//! The conversation history, end to end through the compiled binary: in text mode
//! Enton remembers each reply whole, exactly as the cortex wrote it (a spoken reply
//! the owner cuts off is remembered only as far as it was heard; the runtime's tests
//! cover that with a mock player). `--no-chime` is accepted whether or not the voice
//! is built in.

use std::error::Error;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

type TestResult<T = ()> = Result<T, Box<dyn Error + Send + Sync>>;

/// Upper bound on anything the binary should do promptly.
const WAIT: Duration = Duration::from_secs(10);
/// What the owner says, in order.
const ASKED: [&str; 2] = ["Enton, conta uma história.", "Enton, e depois?"];
/// What the cortex answers, in order: the first with a stage direction and markup.
const REPLIES: [&str; 2] = [
    "*risos* Era uma vez um robô. Ele morava num **PC**.",
    "Fim.",
];

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A scratch directory removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> TestResult<Self> {
        let count = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("enton-history-{}-{count}", std::process::id()));
        std::fs::create_dir_all(&path)?;
        Ok(Self(path))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.0));
    }
}

/// A message the cortex received: its role and content.
type Message = (String, String);

/// A loopback OpenAI-compatible server that keeps the messages of every thought it
/// is asked and streams [`REPLIES`] in turn.
struct FakeCortex {
    addr: SocketAddr,
    requests: Arc<Mutex<Vec<Vec<Message>>>>,
    stop: Arc<AtomicBool>,
    server: Option<JoinHandle<()>>,
}

impl FakeCortex {
    fn start() -> TestResult<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?;
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (kept, stopped) = (Arc::clone(&requests), Arc::clone(&stop));
        let server = thread::spawn(move || {
            for stream in listener.incoming() {
                if stopped.load(Ordering::SeqCst) {
                    return;
                }
                let Ok(stream) = stream else { continue };
                // A client that hung up has nothing left to report.
                drop(answer(stream, &kept));
            }
        });
        Ok(Self {
            addr,
            requests,
            stop,
            server: Some(server),
        })
    }

    fn url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    fn requests(&self) -> TestResult<Vec<Vec<Message>>> {
        Ok(self
            .requests
            .lock()
            .map_err(|_| "requests poisoned")?
            .clone())
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

/// Answer one request: the warm-up with an empty completion, a thought with the next
/// reply after keeping its messages.
fn answer(stream: TcpStream, requests: &Mutex<Vec<Vec<Message>>>) -> TestResult {
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
    let messages = body
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .ok_or("no messages")?
        .iter()
        .map(|message| {
            let field = |name| {
                message
                    .get(name)
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_owned()
            };
            (field("role"), field("content"))
        })
        .collect();
    let reply = {
        let mut requests = requests.lock().map_err(|_| "requests poisoned")?;
        requests.push(messages);
        REPLIES.get(requests.len() - 1).copied().unwrap_or("Nada.")
    };
    let delta = serde_json::json!({ "choices": [{ "delta": { "content": reply } }] });
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n\
         data: {delta}\n\ndata: [DONE]\n\n"
    );
    stream.write_all(response.as_bytes())?;
    stream.flush()?;
    Ok(())
}

#[test]
fn text_mode_remembers_each_reply_whole() -> TestResult {
    let scratch = Scratch::new()?;
    let cortex = FakeCortex::start()?;
    let mut child = Command::new(env!("CARGO_BIN_EXE_enton"))
        .args(["--no-soul", "--no-chime", "--cortex-url", &cortex.url()])
        // Never the owner's own PERSONA.md or CHECKLIST.md.
        .env("XDG_CONFIG_HOME", scratch.0.join("config"))
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

    // A fragment of each reply, as the printed `Speak` action shows it.
    for (asked, reply) in ASKED.iter().zip(["Era uma vez", "Fim."]) {
        writeln!(stdin, "{asked}")?;
        let deadline = Instant::now() + WAIT;
        loop {
            match printed.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(line) if line.starts_with("Speak {") && line.contains(reply) => break,
                Ok(_) => {}
                Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => {
                    drop(child.kill());
                    return Err(format!("no reply to {asked:?}").into());
                }
            }
        }
    }
    writeln!(stdin, "quit")?;
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
    let stderr = errors.join().map_err(|_| "stderr reader panicked")?;
    assert!(status.success(), "{stderr}");
    assert_eq!(
        stderr.contains("Warning: --no-chime ignored (voice feature disabled)"),
        !cfg!(feature = "voice"),
        "{stderr}"
    );

    let requests = cortex.requests()?;
    assert_eq!(requests.len(), 2, "{requests:?}");
    let pair = |role: &str, content: &str| (role.to_owned(), content.to_owned());
    // The first thought has no history; the second carries the first exchange, the
    // reply whole, stage direction and markup included.
    let second = requests.get(1).ok_or("no second request")?;
    assert_eq!(
        second.get(1..),
        Some(
            &[
                pair("user", ASKED[0]),
                pair("assistant", REPLIES[0]),
                pair("user", ASKED[1]),
            ][..]
        ),
        "{second:?}"
    );
    assert_eq!(
        requests.first().and_then(|first| first.get(1..)),
        Some(&[pair("user", ASKED[0])][..])
    );
    Ok(())
}
