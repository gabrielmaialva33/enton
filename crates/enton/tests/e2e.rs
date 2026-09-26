//! End-to-end integration tests for the Enton digital organism binary.
//!
//! Spawns the compiled `enton` binary against a fake streaming OpenAI-compatible
//! cortex server implemented using only `std::net::TcpListener`. Exercises
//! ignition, streaming cortex deliberation, speech generation, and soul
//! persistence/resumption across process restarts.

use std::error::Error;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

static TEST_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Guard managing a temporary directory containing the organism's SQLite soul.
#[derive(Debug)]
struct TempSoulDir {
    path: PathBuf,
}

impl TempSoulDir {
    fn new() -> Result<Self, Box<dyn Error>> {
        let count = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("enton-e2e-{}-{count}", std::process::id()));
        std::fs::create_dir_all(&path)?;
        Ok(Self { path })
    }

    fn soul_file(&self) -> PathBuf {
        self.path.join("soul.sqlite")
    }
}

impl Drop for TempSoulDir {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.path));
    }
}

/// A reply to the startup warm-up: a successful empty completion.
const WARM_UP_REPLY: &[u8] =
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";

/// Whether a request body is the cortex warm-up (one token, no streaming).
fn is_warm_up(body: &[u8]) -> bool {
    let body = String::from_utf8_lossy(body);
    body.contains("\"max_tokens\":1") && body.contains("\"stream\":false")
}

/// A fake OpenAI-compatible streaming cortex server using only `std::net`.
#[derive(Debug)]
struct FakeCortex {
    addr: SocketAddr,
    request_count: Arc<AtomicUsize>,
    shutdown: Arc<AtomicBool>,
    server_thread: Option<JoinHandle<()>>,
}

impl FakeCortex {
    fn new(responses: Vec<String>) -> Result<Self, Box<dyn Error>> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?;
        listener.set_nonblocking(true)?;

        let request_count = Arc::new(AtomicUsize::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));
        let responses = Arc::new(Mutex::new(responses));

        let req_count_clone = Arc::clone(&request_count);
        let shutdown_clone = Arc::clone(&shutdown);

        let server_thread = thread::spawn(move || {
            while !shutdown_clone.load(Ordering::Relaxed) {
                let (mut stream, _) = match listener.accept() {
                    Ok(conn) => conn,
                    Err(ref err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(_) => break,
                };

                if shutdown_clone.load(Ordering::Relaxed) {
                    break;
                }

                stream.set_nonblocking(false).ok();
                stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
                stream.set_write_timeout(Some(Duration::from_secs(5))).ok();

                // Consume HTTP request headers and body to avoid connection resets.
                let mut content_length = 0;
                {
                    let mut reader = BufReader::new(&mut stream);
                    loop {
                        let mut line = String::new();
                        match reader.read_line(&mut line) {
                            Ok(0) | Err(_) => break,
                            Ok(_) => {
                                let trimmed = line.trim();
                                if trimmed.is_empty() {
                                    break;
                                }
                                let lower = trimmed.to_ascii_lowercase();
                                if let Some(len_str) = lower.strip_prefix("content-length:") {
                                    content_length = len_str.trim().parse::<usize>().unwrap_or(0);
                                }
                            }
                        }
                    }
                    let mut body = vec![0u8; content_length];
                    drop(reader.read_exact(&mut body));
                    // The warm-up at startup only loads the model: answer it, but it
                    // is not a thought.
                    if is_warm_up(&body) {
                        drop(stream.write_all(WARM_UP_REPLY));
                        drop(stream.flush());
                        drop(stream.shutdown(std::net::Shutdown::Both));
                        continue;
                    }
                }

                let req_idx = req_count_clone.fetch_add(1, Ordering::SeqCst);
                let Ok(guard) = responses.lock() else {
                    drop(stream.write_all(b"HTTP/1.1 500 Internal Server Error\r\n\r\n"));
                    drop(stream.flush());
                    drop(stream.shutdown(std::net::Shutdown::Both));
                    break;
                };
                let reply = guard
                    .get(req_idx)
                    .cloned()
                    .unwrap_or_else(|| "Default cortex reply.".to_string());

                let sse_body = format!(
                    "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{reply}\"}}}}]}}\n\ndata: [DONE]\n\n"
                );
                let http_response = format!(
                    "HTTP/1.1 200 OK\r\n\
                     Content-Type: text/event-stream\r\n\
                     Cache-Control: no-cache\r\n\
                     Connection: close\r\n\
                     \r\n\
                     {sse_body}"
                );

                drop(stream.write_all(http_response.as_bytes()));
                drop(stream.flush());
                drop(stream.shutdown(std::net::Shutdown::Both));
            }
        });

        Ok(Self {
            addr,
            request_count,
            shutdown,
            server_thread: Some(server_thread),
        })
    }

    fn url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    fn request_count(&self) -> usize {
        self.request_count.load(Ordering::SeqCst)
    }
}

impl Drop for FakeCortex {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        drop(TcpStream::connect(self.addr));
        if let Some(handle) = self.server_thread.take() {
            drop(handle.join());
        }
    }
}

/// RAII guard ensuring the child process is terminated on unexpected exits.
#[derive(Debug)]
struct ChildGuard(Option<Child>);

impl ChildGuard {
    fn new(child: Child) -> Self {
        Self(Some(child))
    }

    fn wait_for_exit(
        &mut self,
        timeout: Duration,
    ) -> Result<std::process::ExitStatus, Box<dyn Error>> {
        let mut child = self.0.take().ok_or("child handle already consumed")?;
        let start = Instant::now();
        loop {
            if let Some(status) = child.try_wait()? {
                return Ok(status);
            }
            if start.elapsed() > timeout {
                drop(child.kill());
                drop(child.wait());
                return Err(
                    format!("Child process did not exit within timeout of {timeout:?}").into(),
                );
            }
            thread::sleep(Duration::from_millis(20));
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.0.take() {
            drop(child.kill());
            drop(child.wait());
        }
    }
}

/// Waits until the Soul resuming banner appears and extracts resumed timestamp.
fn wait_for_soul_banner(
    rx: &std::sync::mpsc::Receiver<String>,
    stderr_lines: &Arc<Mutex<Vec<String>>>,
    timeout: Duration,
) -> Result<u64, Box<dyn Error>> {
    let start = Instant::now();
    while start.elapsed() < timeout {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(line) => {
                if let Some(pos) = line.find("(resuming at ") {
                    let after = &line[pos + "(resuming at ".len()..];
                    if let Some(end_pos) = after.find(" ms)") {
                        let ms_str = &after[..end_pos];
                        if let Ok(ms) = ms_str.parse::<u64>() {
                            return Ok(ms);
                        }
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    let errs = stderr_lines
        .lock()
        .map_err(|_| "stderr lock poisoned")?
        .join("\n");
    Err(format!("Timed out waiting for Soul resuming banner. Stderr:\n{errs}").into())
}

/// Waits until a `Speak` action containing the expected reply is emitted.
fn wait_for_reply(
    rx: &std::sync::mpsc::Receiver<String>,
    stderr_lines: &Arc<Mutex<Vec<String>>>,
    expected_reply: &str,
    timeout: Duration,
) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    while start.elapsed() < timeout {
        match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(line) => {
                if line.contains("Speak {") && line.contains(expected_reply) {
                    return Ok(());
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    let errs = stderr_lines
        .lock()
        .map_err(|_| "stderr lock poisoned")?
        .join("\n");
    Err(format!(
        "Timed out waiting for Speak action containing {expected_reply:?}. Stderr:\n{errs}"
    )
    .into())
}

/// Runs a single session with the `enton` binary and returns the resumed timestamp in ms.
fn run_session(
    cortex_url: &str,
    soul_path: &Path,
    user_prompt: &str,
    expected_reply: &str,
) -> Result<u64, Box<dyn Error>> {
    let binary = env!("CARGO_BIN_EXE_enton");
    let mut command = Command::new(binary);
    // Never the owner's own PERSONA.md or CHECKLIST.md: a directory that does not
    // exist beside the soul gives the built-in persona and nothing to check.
    let config = soul_path.with_extension("config");
    command
        .arg("--cortex-url")
        .arg(cortex_url)
        .arg("--soul")
        .arg(soul_path)
        .env("XDG_CONFIG_HOME", config)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn()?;

    let mut stdin = child.stdin.take().ok_or("take child stdin failed")?;
    let stdout = child.stdout.take().ok_or("take child stdout failed")?;
    let stderr = child.stderr.take().ok_or("take child stderr failed")?;

    let mut guard = ChildGuard::new(child);

    let (stdout_tx, stdout_rx) = std::sync::mpsc::channel();
    let stdout_handle = thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for l in reader.lines().map_while(Result::ok) {
            if stdout_tx.send(l).is_err() {
                break;
            }
        }
    });

    let stderr_lines = Arc::new(Mutex::new(Vec::new()));
    let stderr_lines_clone = Arc::clone(&stderr_lines);
    let stderr_handle = thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for l in reader.lines().map_while(Result::ok) {
            if let Ok(mut buf) = stderr_lines_clone.lock() {
                buf.push(l);
            }
        }
    });

    let timeout = Duration::from_secs(10);

    // 1. Verify soul resumption timestamp.
    let resumed_ms = wait_for_soul_banner(&stdout_rx, &stderr_lines, timeout)?;

    // Give a brief window to guarantee wall-clock time passes for monotonic timestamps.
    thread::sleep(Duration::from_millis(50));

    // 2. Send user prompt via stdin.
    writeln!(stdin, "{user_prompt}")?;
    stdin.flush()?;

    // 3. Await Speak action containing the expected reply on stdout.
    wait_for_reply(&stdout_rx, &stderr_lines, expected_reply, timeout)?;

    // 4. Send quit command and close stdin.
    writeln!(stdin, "quit")?;
    stdin.flush()?;
    drop(stdin);

    // 5. Verify clean process termination.
    let status = guard.wait_for_exit(timeout)?;
    if !status.success() {
        let errs = stderr_lines
            .lock()
            .map_err(|_| "stderr lock poisoned")?
            .join("\n");
        return Err(format!("Subprocess failed with status {status:?}. Stderr:\n{errs}").into());
    }

    if let Err(err) = stdout_handle.join() {
        return Err(format!("stdout thread panicked: {err:?}").into());
    }
    if let Err(err) = stderr_handle.join() {
        return Err(format!("stderr thread panicked: {err:?}").into());
    }

    Ok(resumed_ms)
}

#[test]
fn e2e_enton_binary_runs_and_restores_soul() {
    let temp_dir = TempSoulDir::new().unwrap();
    let soul_path = temp_dir.soul_file();

    let reply1 = "Ola, o tempo esta ensolarado hoje.";
    let reply2 = "O livro de matematica tinha muitos problemas.";

    let fake_cortex = FakeCortex::new(vec![reply1.to_string(), reply2.to_string()]).unwrap();

    // Session 1: Fresh soul log starting at 0 ms.
    let resumed1 = run_session(
        &fake_cortex.url(),
        &soul_path,
        "enton, qual e a previsao do tempo para hoje?",
        reply1,
    )
    .unwrap();
    assert_eq!(resumed1, 0, "Expected session 1 to resume at 0 ms");

    // Session 2: Resumed soul log continuing from previous timestamp (> 0 ms).
    let resumed2 = run_session(
        &fake_cortex.url(),
        &soul_path,
        "enton, conte uma piada curta agora por favor.",
        reply2,
    )
    .unwrap();
    assert!(
        resumed2 > 0,
        "Expected session 2 to resume at > 0 ms, got {resumed2} ms"
    );

    assert_eq!(
        fake_cortex.request_count(),
        2,
        "Expected fake cortex to handle exactly 2 requests"
    );
}
