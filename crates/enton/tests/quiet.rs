//! The binary with quiet hours around the current minute and a typed quiet command: the
//! soul records the band's flag and each command as a quiet event, never as a speech cue,
//! so a command buys no thought, and a call by name is still answered.

use std::error::Error;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use enton_adapters::initiative::{QUIET_HOURS_VAR, local_minute};
use enton_adapters::{Soul, SoulConfig};
use enton_core::Event;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Upper bound on anything the binary should do promptly.
const WAIT: Duration = Duration::from_secs(10);

/// A scratch directory for the soul and the config, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> TestResult<Self> {
        let path = std::env::temp_dir().join(format!("enton-quiet-e2e-{}", std::process::id()));
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
    /// Wait for a line from `tag` that contains `needle`, printed at any time so far.
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

    /// How many `tag` lines seen so far contain `needle`.
    fn count(&self, tag: &str, needle: &str) -> usize {
        self.seen
            .iter()
            .filter(|(from, line)| *from == tag && line.contains(needle))
            .count()
    }
}

/// A three-minute band of local time around the current minute.
fn band_around_now() -> String {
    let now = local_minute();
    let at = |minute: u16| format!("{:02}:{:02}", minute / 60, minute % 60);
    format!("{}-{}", at((now + 1439) % 1440), at((now + 2) % 1440))
}

#[test]
fn quiet_commands_and_hours_reach_the_soul_as_flags_and_buy_no_thought() -> TestResult {
    let scratch = Scratch::new()?;
    let soul = scratch.0.join("soul.sqlite");
    let mut child = Command::new(env!("CARGO_BIN_EXE_enton"))
        .arg("--soul")
        .arg(&soul)
        .arg("--cortex-url")
        .arg(dead_cortex()?)
        .env("XDG_CONFIG_HOME", scratch.0.join("config"))
        .env(QUIET_HOURS_VAR, band_around_now())
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
    output.wait_for("stderr", "[enton] quiet hours (")?;
    writeln!(stdin, "Enton, silêncio por meia hora")?;
    stdin.flush()?;
    output.wait_for("stderr", "[enton] quiet for 30 min")?;
    writeln!(stdin, "Enton, pode falar")?;
    stdin.flush()?;
    output.wait_for("stderr", "[enton] quiet released")?;
    // Neither command bought a thought.
    assert_eq!(output.count("stdout", "Think {"), 0, "{:?}", output.seen);
    // Called by name during the quiet hours, Enton still thinks.
    writeln!(stdin, "Enton, que horas são?")?;
    stdin.flush()?;
    output.wait_for("stdout", "Think {")?;
    output.wait_for("stderr", "falling back to offline placeholder")?;
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
    // The band is read before the first tick, and only its flag is recorded.
    let band = events
        .iter()
        .position(|event| matches!(event, Event::QuietHours { active: true, .. }));
    let first_tick = events
        .iter()
        .position(|event| matches!(event, Event::Tick { .. }));
    assert!(
        band.is_some() && band < first_tick.or(Some(usize::MAX)),
        "{events:?}"
    );
    // Each command is a quiet event: half an hour, then a release.
    let quiet: Vec<(u64, u64)> = events
        .iter()
        .filter_map(|event| match event {
            Event::Quiet { now, until } => Some((now.0, until.0)),
            _ => None,
        })
        .collect();
    assert!(
        matches!(quiet.as_slice(), [(hushed, until), (released, released_until)]
            if until - hushed == 1_800_000 && released == released_until),
        "{quiet:?}"
    );
    // The only speech cue is the call by name.
    let cues = events
        .iter()
        .filter(|event| matches!(event, Event::Speech { .. }))
        .count();
    assert_eq!(cues, 1, "{events:?}");
    Ok(())
}
