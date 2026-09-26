//! The binary with a checklist and a cortex that is down: the soul records whether the
//! checklist holds something to check (never its text), and a thought the cortex could
//! not answer as a failure, not as a reply.

use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use enton_adapters::{Soul, SoulConfig};
use enton_core::{Event, ThoughtId};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Upper bound on anything the binary should do promptly.
const WAIT: Duration = Duration::from_secs(10);

/// What the owner wrote in the checklist; the soul must never hold it.
const CHECKLIST: &str = "# Hoje\n- [ ] regar as samambaias da varanda\n";

/// A scratch directory for the soul and the config, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> TestResult<Self> {
        let path = std::env::temp_dir().join(format!("enton-checklist-e2e-{}", std::process::id()));
        std::fs::create_dir_all(path.join("config").join("enton"))?;
        Ok(Self(path))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // Best effort: a leftover temporary directory must not fail a test.
        drop(std::fs::remove_dir_all(&self.0));
    }
}

/// Kills the binary if the test fails before it exits.
struct Running(Child);

impl Drop for Running {
    fn drop(&mut self) {
        drop(self.0.kill());
        drop(self.0.wait());
    }
}

/// A local address nothing listens on: a cortex that is down.
fn dead_cortex() -> TestResult<String> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(format!("http://127.0.0.1:{port}/v1"))
}

/// Forward each line of `stream` to a channel, tagged with where it came from.
fn lines(
    stream: impl std::io::Read + Send + 'static,
    tag: &'static str,
    sender: mpsc::Sender<(&'static str, String)>,
) {
    thread::spawn(move || {
        for line in BufReader::new(stream).lines().map_while(Result::ok) {
            if sender.send((tag, line)).is_err() {
                return;
            }
        }
    });
}

/// Every line the binary printed so far, and the channel that brings the rest.
struct Output {
    receiver: mpsc::Receiver<(&'static str, String)>,
    seen: Vec<(&'static str, String)>,
}

impl Output {
    /// Wait for a line from `tag` that contains `needle`, printed at any time so far:
    /// stdout and stderr arrive on separate pipes, in no particular order.
    fn wait_for(&mut self, tag: &str, needle: &str) -> TestResult {
        let matches = |(from, line): &(&str, String)| *from == tag && line.contains(needle);
        if self.seen.iter().any(matches) {
            return Ok(());
        }
        let deadline = Instant::now() + WAIT;
        while let Some(left) = deadline.checked_duration_since(Instant::now()) {
            let Ok(next) = self.receiver.recv_timeout(left) else {
                break;
            };
            let found = matches(&next);
            self.seen.push(next);
            if found {
                return Ok(());
            }
        }
        Err(format!(
            "no {tag} line containing {needle:?} within {WAIT:?}: {:?}",
            self.seen
        )
        .into())
    }
}

#[test]
fn the_soul_records_the_checklist_flag_and_a_failed_thought_but_never_the_checklist() -> TestResult
{
    let scratch = Scratch::new()?;
    let config = scratch.0.join("config");
    std::fs::write(config.join("enton").join("CHECKLIST.md"), CHECKLIST)?;
    let soul = scratch.0.join("soul.sqlite");

    let mut child = Command::new(env!("CARGO_BIN_EXE_enton"))
        .arg("--soul")
        .arg(&soul)
        .arg("--cortex-url")
        .arg(dead_cortex()?)
        .env("XDG_CONFIG_HOME", &config)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stdin = child.stdin.take().ok_or("no stdin")?;
    let (sender, receiver) = mpsc::channel();
    lines(
        child.stdout.take().ok_or("no stdout")?,
        "stdout",
        sender.clone(),
    );
    lines(child.stderr.take().ok_or("no stderr")?, "stderr", sender);
    let mut running = Running(child);
    let mut output = Output {
        receiver,
        seen: Vec::new(),
    };

    output.wait_for("stdout", "[enton] Soul:")?;
    output.wait_for("stderr", "(something to check)")?;
    writeln!(stdin, "enton, que horas sao agora?")?;
    stdin.flush()?;
    output.wait_for("stdout", "Think {")?;
    output.wait_for("stderr", "falling back to offline placeholder")?;
    // The failure reaches the event loop right after that line; give it a moment.
    thread::sleep(Duration::from_millis(500));
    writeln!(stdin, "quit")?;
    stdin.flush()?;
    drop(stdin);
    let deadline = Instant::now() + WAIT;
    let status = loop {
        if let Some(status) = running.0.try_wait()? {
            break status;
        }
        if Instant::now() > deadline {
            return Err("enton did not quit".into());
        }
        thread::sleep(Duration::from_millis(20));
    };
    if !status.success() {
        return Err(format!("enton failed: {status:?}").into());
    }

    let log = Soul::open(&soul, SoulConfig::default())?;
    let events: Vec<Event> = log
        .read_after(0, 4_096)?
        .into_iter()
        .map(|(_, event)| event)
        .collect();
    // The checklist is read before the first tick, and only its flag is recorded.
    let first_checklist = events
        .iter()
        .position(|event| matches!(event, Event::Checklist { .. }));
    let first_tick = events
        .iter()
        .position(|event| matches!(event, Event::Tick { .. }));
    assert!(
        first_checklist.is_some() && first_checklist < first_tick.or(Some(usize::MAX)),
        "{events:?}"
    );
    assert!(matches!(
        first_checklist.and_then(|index| events.get(index)),
        Some(Event::Checklist {
            actionable: true,
            ..
        })
    ));
    // The owner's question failed in the cortex: a failure, never a reply.
    assert!(events.iter().any(|event| matches!(
        event,
        Event::CortexFailed {
            thought: ThoughtId(1),
            ..
        }
    )));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, Event::CortexReply { .. }))
    );
    drop(log);

    // Nothing the owner wrote in the checklist is anywhere in the soul's files.
    for entry in std::fs::read_dir(&scratch.0)? {
        let path = entry?.path();
        if path.is_file() {
            let bytes = std::fs::read(&path)?;
            assert!(
                !bytes.windows(10).any(|window| window == b"samambaias"),
                "{} holds the checklist text",
                path.display()
            );
        }
    }
    Ok(())
}
