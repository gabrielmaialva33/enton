//! The persona, end to end through the compiled binary: the startup line names
//! the one that speaks, the cortex receives it as its system prompt, the soul
//! links every thought to its hash, and `enton why` shows which one spoke and
//! warns when it changed. Enton only ever reads the file.

use std::error::Error;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use enton_adapters::cortex::Persona;
use enton_adapters::soul::{PersonaDigest, PersonaSource};
use enton_adapters::{Soul, SoulConfig};
use enton_core::ThoughtId;

type TestResult<T = ()> = Result<T, Box<dyn Error + Send + Sync>>;

/// Upper bound on anything the binary should do promptly.
const WAIT: Duration = Duration::from_secs(10);
/// A cortex nothing listens on: its calls fail at once.
const NO_CORTEX: &str = "http://127.0.0.1:9/v1";
/// What the owner says in every run: Enton's name and a whole request.
const REQUEST: &str = "enton, que horas são agora?";
/// Two personas in the style of the built-in one.
const CALM: &str = "Você é o Enton, calmo e gentil.\nResponda em uma frase.\n";
const CHEEKY: &str = "Você é o Enton, zoeiro e debochado.\nResponda curto, com gíria.\n";

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A scratch directory removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> TestResult<Self> {
        let count = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("enton-persona-{}-{count}", std::process::id()));
        std::fs::create_dir_all(&path)?;
        Ok(Self(path))
    }

    fn write(&self, name: &str, contents: &[u8]) -> TestResult<PathBuf> {
        let path = self.0.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, contents)?;
        Ok(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.0));
    }
}

/// A loopback OpenAI-compatible server that keeps every system prompt it is
/// sent and streams a one-sentence reply.
struct FakeCortex {
    addr: SocketAddr,
    prompts: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    server: Option<JoinHandle<()>>,
}

impl FakeCortex {
    fn start() -> TestResult<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?;
        let prompts = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (kept, stopped) = (Arc::clone(&prompts), Arc::clone(&stop));
        let server = thread::spawn(move || {
            for stream in listener.incoming() {
                if stopped.load(Ordering::SeqCst) {
                    return;
                }
                let Ok(stream) = stream else { continue };
                let kept = Arc::clone(&kept);
                // A client that hung up has nothing left to report.
                thread::spawn(move || drop(answer(stream, &kept)));
            }
        });
        Ok(Self {
            addr,
            prompts,
            stop,
            server: Some(server),
        })
    }

    fn url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    fn prompts(&self) -> TestResult<Vec<String>> {
        Ok(self.prompts.lock().map_err(|_| "prompts poisoned")?.clone())
    }
}

impl Drop for FakeCortex {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Wake the blocking accept so the server sees the flag.
        drop(TcpStream::connect(self.addr));
        if let Some(server) = self.server.take() {
            drop(server.join());
        }
    }
}

/// Answer one request: the warm-up with an empty completion, a thought with a
/// streamed reply after keeping its system prompt.
fn answer(stream: TcpStream, prompts: &Mutex<Vec<String>>) -> TestResult {
    stream.set_read_timeout(Some(WAIT))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut length = 0;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Err("the request ended inside its headers".into());
        }
        let line = line.trim();
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            length = value.trim().parse()?;
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    let body: serde_json::Value = serde_json::from_slice(&body)?;
    let mut stream = stream;
    if body.get("stream") == Some(&serde_json::Value::Bool(false)) {
        stream.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\
              Connection: close\r\n\r\n{}",
        )?;
        return Ok(());
    }
    let prompt = body
        .pointer("/messages/0/content")
        .and_then(serde_json::Value::as_str)
        .ok_or("no system prompt")?
        .to_owned();
    prompts.lock().map_err(|_| "prompts poisoned")?.push(prompt);
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n\
          data: {\"choices\":[{\"delta\":{\"content\":\"Oi.\"}}]}\n\ndata: [DONE]\n\n",
    )?;
    stream.flush()?;
    Ok(())
}

/// What a finished run printed.
struct Run {
    status: ExitStatus,
    stdout: Vec<String>,
    stderr: String,
}

impl Run {
    fn line(&self, prefix: &str) -> Option<&str> {
        self.stdout
            .iter()
            .map(String::as_str)
            .find(|line| line.starts_with(prefix))
    }
}

/// Run the binary with `args` and `XDG_CONFIG_HOME` at `config`; say
/// [`REQUEST`] `requests` times, each after the reply to the one before, then quit.
fn live(config: &Path, args: &[&str], requests: usize) -> TestResult<Run> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_enton"))
        .args(args)
        .env("XDG_CONFIG_HOME", config)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stdin = child.stdin.take().ok_or("no stdin")?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let mut stderr = child.stderr.take().ok_or("no stderr")?;
    let (lines, printed) = mpsc::channel();
    let reader = thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if lines.send(line).is_err() {
                return;
            }
        }
    });
    let errors = thread::spawn(move || {
        let mut text = String::new();
        drop(stderr.read_to_string(&mut text));
        text
    });

    let speaks = |lines: &[String]| {
        lines
            .iter()
            .filter(|line| line.starts_with("Speak {"))
            .count()
    };
    let mut stdout = Vec::new();
    for _ in 0..requests {
        writeln!(stdin, "{REQUEST}")?;
        let replies = speaks(&stdout);
        let deadline = Instant::now() + WAIT;
        while speaks(&stdout) == replies {
            match printed.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(line) => stdout.push(line),
                Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => {
                    drop(child.kill());
                    return Err(format!("no reply; printed {stdout:?}").into());
                }
            }
        }
    }
    // A binary that refused to start has already closed its end.
    drop(writeln!(stdin, "quit"));
    drop(stdin);
    let deadline = Instant::now() + WAIT;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            drop(child.kill());
            return Err("enton did not exit".into());
        }
        thread::sleep(Duration::from_millis(10));
    };
    reader.join().map_err(|_| "stdout reader panicked")?;
    stdout.extend(printed.try_iter());
    let stderr = errors.join().map_err(|_| "stderr reader panicked")?;
    Ok(Run {
        status,
        stdout,
        stderr,
    })
}

/// `enton why` on the soul at `soul`, with `args`.
fn why(soul: &Path, args: &[&str]) -> TestResult<String> {
    let output = Command::new(env!("CARGO_BIN_EXE_enton"))
        .arg("why")
        .arg("--soul")
        .arg(soul)
        .args(["--profile", "t1-ref"])
        .args(args)
        .output()?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned().into());
    }
    Ok(String::from_utf8(output.stdout)?)
}

fn digest(persona: &Persona) -> PersonaDigest {
    persona.into()
}

fn text(path: &Path) -> TestResult<String> {
    Ok(path.to_str().ok_or("path is not UTF-8")?.to_owned())
}

#[test]
fn the_startup_line_names_the_persona_that_speaks() {
    let scratch = Scratch::new().unwrap();
    let config = scratch.0.join("config");
    std::fs::create_dir_all(&config).unwrap();

    // No file: the built-in persona, and the line says where Enton looked.
    let run = live(&config, &["--no-soul", "--cortex-url", NO_CORTEX], 0).unwrap();
    assert!(run.status.success(), "{}", run.stderr);
    let built_in = digest(&Persona::built_in());
    assert_eq!(
        run.line("[enton] Persona: "),
        Some(
            format!(
                "[enton] Persona: built-in default (no {}), sha256 {}, {} bytes",
                config.join("enton").join("PERSONA.md").display(),
                built_in.short_hex(),
                built_in.bytes
            )
            .as_str()
        )
    );

    // The owner's file, hashed as `sha256sum` would, and left exactly as it was.
    let path = scratch
        .write("config/enton/PERSONA.md", CALM.as_bytes())
        .unwrap();
    let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
    let run = live(&config, &["--no-soul", "--cortex-url", NO_CORTEX], 0).unwrap();
    assert!(run.status.success(), "{}", run.stderr);
    let calm = digest(&Persona::from_file(&path).unwrap());
    assert_eq!(calm.source, PersonaSource::File);
    assert_eq!(
        run.line("[enton] Persona: "),
        Some(
            format!(
                "[enton] Persona: {}, sha256 {}, {} bytes",
                path.display(),
                calm.short_hex(),
                CALM.len()
            )
            .as_str()
        )
    );
    assert_eq!(std::fs::read_to_string(&path).unwrap(), CALM);
    assert_eq!(
        std::fs::metadata(&path).unwrap().modified().unwrap(),
        modified
    );
    let listed: Vec<_> = std::fs::read_dir(config.join("enton"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(listed, ["PERSONA.md"], "nothing was written beside it");
}

#[test]
fn a_persona_that_cannot_be_used_stops_enton_before_it_starts() {
    let scratch = Scratch::new().unwrap();
    let huge = scratch.write("huge.md", &vec![b'a'; 8 * 1024 + 1]).unwrap();
    let missing = scratch.0.join("missing.md");
    for (persona, says) in [
        (&huge, "is 8193 bytes, over the 8192-byte (8 KiB) cap"),
        (&missing, "no persona file at"),
    ] {
        let run = live(
            &scratch.0,
            &[
                "--no-soul",
                "--cortex-url",
                NO_CORTEX,
                "--persona",
                &text(persona).unwrap(),
            ],
            0,
        )
        .unwrap();
        assert!(!run.status.success());
        assert!(run.stderr.contains(says), "{}", run.stderr);
        assert!(run.line("[enton] ").is_none(), "{:?}", run.stdout);
    }
}

#[test]
fn each_thought_is_asked_with_its_persona_and_why_says_which() {
    let scratch = Scratch::new().unwrap();
    let cortex = FakeCortex::start().unwrap();
    let soul = scratch.0.join("soul.sqlite");
    let calm_file = scratch.write("calm.md", CALM.as_bytes()).unwrap();
    let cheeky_file = scratch.write("cheeky.md", CHEEKY.as_bytes()).unwrap();
    let calm = digest(&Persona::from_file(&calm_file).unwrap());
    let cheeky = digest(&Persona::from_file(&cheeky_file).unwrap());

    // Three lives: calm, then cheeky, then calm again.
    let (soul_arg, url) = (text(&soul).unwrap(), cortex.url());
    for persona in [&calm_file, &cheeky_file, &calm_file] {
        let persona = text(persona).unwrap();
        let args = [
            "--soul",
            &soul_arg,
            "--cortex-url",
            &url,
            "--persona",
            &persona,
        ];
        let run = live(&scratch.0, &args, 1).unwrap();
        assert!(run.status.success(), "{}", run.stderr);
    }

    // The cortex was given each persona, trimmed, before the clock line.
    let prompts = cortex.prompts().unwrap();
    assert_eq!(prompts.len(), 3);
    for (prompt, persona) in prompts.iter().zip([CALM, CHEEKY, CALM]) {
        let expected = format!("{}\n\nCurrent local date and time: ", persona.trim_end());
        assert!(prompt.starts_with(&expected), "{prompt}");
    }

    // The soul recorded two personas, and linked each thought to its own.
    let log = Soul::open_read_only(&soul, SoulConfig::default()).unwrap();
    let records = log.personas().unwrap();
    let digests: Vec<PersonaDigest> = records.iter().map(|record| record.digest).collect();
    assert_eq!(digests, [calm, cheeky]);
    for (thought, persona) in [(1, calm), (2, cheeky), (3, calm)] {
        let record = log.thought_persona(ThoughtId(thought)).unwrap();
        assert_eq!(
            record.map(|record| record.digest),
            Some(persona),
            "{thought}"
        );
    }
    drop(log);

    let text = why(&soul, &[]).unwrap();
    let (calm_name, cheeky_name) = (
        format!("{} (file, {} bytes)", calm.short_hex(), CALM.len()),
        format!("{} (file, {} bytes)", cheeky.short_hex(), CHEEKY.len()),
    );
    assert!(
        text.contains(&format!(
            "Personas: {calm_name} first used at seq {}; {cheeky_name} first used at seq {}.",
            records[0].first_seq, records[1].first_seq
        )),
        "{text}"
    );
    let persona_lines: Vec<&str> = text
        .lines()
        .filter(|line| line.starts_with("  persona") || line.starts_with("  warning"))
        .collect();
    assert_eq!(
        persona_lines,
        [
            format!("  persona   {calm_name}"),
            format!("  persona   {cheeky_name}"),
            format!(
                "  warning   the persona changed since thought #1, which was asked with {calm_name}"
            ),
            format!("  persona   {calm_name}"),
            format!(
                "  warning   the persona changed since thought #2, which was asked with {cheeky_name}"
            ),
        ]
    );

    let json: serde_json::Value = serde_json::from_str(&why(&soul, &["--json"]).unwrap()).unwrap();
    let second = &json["cues"][1]["decision"];
    assert_eq!(second["thought"], 2);
    assert_eq!(second["persona"]["sha256"], cheeky.hex());
    assert_eq!(second["persona"]["source"], "file");
    assert_eq!(second["persona"]["bytes"], CHEEKY.len());
    assert_eq!(second["persona_changed"]["thought"], 1);
    assert_eq!(second["persona_changed"]["persona"]["sha256"], calm.hex());
    assert!(json["cues"][0]["decision"].get("persona_changed").is_none());
    assert_eq!(json["personas"].as_array().map(Vec::len), Some(2));
    // The text of a persona is never stored, so never shown.
    assert!(!text.contains("calmo e gentil"), "{text}");
    for suffix in ["", "-wal"] {
        let mut file = soul.clone().into_os_string();
        file.push(suffix);
        let bytes = std::fs::read(&file).unwrap_or_default();
        assert!(
            !bytes
                .windows(18)
                .any(|window| window == b"zoeiro e debochado"),
            "the soul holds a persona's text"
        );
    }
}
