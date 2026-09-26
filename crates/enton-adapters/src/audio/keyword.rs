//! Portuguese keyword fallback: bounded transcription followed by a pure rule.

use std::{
    ffi::OsString,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use sherpa_onnx::{
    OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig, OfflineWhisperModelConfig,
};

use super::{AudioError, model_path};

const MAX_KEYWORD_SAMPLES: usize = 48_000;
const MAX_TRANSCRIPT_BYTES: usize = 4_096;
const MAX_STDERR_BYTES: usize = 2_048;

/// Unknown states must not be confused with a successfully rejected keyword.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeywordDecision {
    /// Recognition has not been attempted.
    Pending,
    /// A complete Enton token was recognized.
    Matched,
    /// Recognition succeeded without an Enton token.
    Absent,
    /// Recognition failed; no negative evidence is available.
    Unavailable,
}

/// Match a complete token, case-insensitively. Deliberately do not match
/// Portuguese "então", substrings such as "Benton", or fuzzy ASR guesses.
#[must_use]
pub fn contains_keyword(transcript: &str) -> bool {
    transcript
        .split(|character: char| !character.is_alphanumeric())
        .any(|token| token.eq_ignore_ascii_case("enton"))
}

/// A single-shot, controlled worker executable. It receives a little-endian
/// u32 sample count followed by f32 PCM on stdin, then returns UTF-8 on stdout.
/// Use the listen example's `--keyword-worker` mode locally or over SSH to T3.
/// The worker must not fork processes inheriting its standard streams.
#[derive(Debug, Clone)]
pub struct KeywordCommand {
    /// Worker executable path.
    pub program: PathBuf,
    /// Arguments passed verbatim to the worker executable.
    pub arguments: Vec<OsString>,
    /// Wall-clock deadline, greater than zero and at most 30 seconds.
    pub timeout: Duration,
}

struct SpawnedWorker {
    child: Child,
    input: std::process::ChildStdin,
    output: std::process::ChildStdout,
    err_pipe: std::process::ChildStderr,
}

fn spawn_worker(command: &KeywordCommand) -> Result<SpawnedWorker, AudioError> {
    let mut child = Command::new(&command.program)
        .args(&command.arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|source| worker_io("spawn", source))?;
    let Some(input) = child.stdin.take() else {
        stop_worker(&mut child)?;
        return Err(AudioError::WorkerProtocol("missing stdin"));
    };
    let Some(output) = child.stdout.take() else {
        stop_worker(&mut child)?;
        return Err(AudioError::WorkerProtocol("missing stdout"));
    };
    let Some(err_pipe) = child.stderr.take() else {
        stop_worker(&mut child)?;
        return Err(AudioError::WorkerProtocol("missing stderr"));
    };
    Ok(SpawnedWorker {
        child,
        input,
        output,
        err_pipe,
    })
}

fn build_payload(samples: &[f32]) -> (usize, Vec<u8>) {
    let count = samples.len().min(MAX_KEYWORD_SAMPLES);
    if count == 0 {
        return (0, Vec::new());
    }
    let mut payload = Vec::with_capacity(4 + count * 4);
    payload.extend_from_slice(&u32::try_from(count).unwrap_or(48_000).to_le_bytes());
    for sample in samples.iter().take(count) {
        payload.extend_from_slice(&sample.to_le_bytes());
    }
    (count, payload)
}

impl KeywordCommand {
    pub(crate) fn validate(&self) -> Result<(), AudioError> {
        if self.timeout.is_zero() || self.timeout > Duration::from_secs(30) {
            return Err(AudioError::Configuration(
                "keyword timeout must be positive and at most 30 seconds",
            ));
        }
        Ok(())
    }

    /// Recognize only the first three seconds. No retries, no audio files, and
    /// at most one child per call. The capture pipeline calls this serially.
    /// Input is normalized mono PCM at 16 kHz. Returns worker, protocol,
    /// configuration, or timeout errors; a failure is never a negative match.
    pub fn recognize(&self, samples: &[f32]) -> Result<KeywordDecision, AudioError> {
        self.recognize_cancellable(samples, &AtomicBool::new(false))
    }

    /// Run recognition with an atomic cancellation flag.
    /// Polling terminates immediately if `cancel` becomes true.
    pub fn recognize_cancellable(
        &self,
        samples: &[f32],
        cancel: &AtomicBool,
    ) -> Result<KeywordDecision, AudioError> {
        self.validate()?;
        let (count, payload) = build_payload(samples);
        if count == 0 {
            return Ok(KeywordDecision::Absent);
        }
        let SpawnedWorker {
            mut child,
            mut input,
            output,
            err_pipe,
        } = spawn_worker(self)?;
        let (sender, receiver) = mpsc::sync_channel(1);
        let io_thread = thread::spawn(move || {
            let result = (|| -> std::io::Result<Vec<u8>> {
                input.write_all(&payload)?;
                drop(input);
                let mut text = Vec::new();
                output
                    .take((MAX_TRANSCRIPT_BYTES + 1) as u64)
                    .read_to_end(&mut text)?;
                Ok(text)
            })();
            // A disconnected parent has already timed out; no consumer remains.
            // The I/O result is still observed when the thread is joined.
            sender.send(result)
        });
        let err_thread = thread::spawn(move || {
            let mut text = Vec::new();
            drop(
                err_pipe
                    .take((MAX_STDERR_BYTES + 1) as u64)
                    .read_to_end(&mut text),
            );
            String::from_utf8_lossy(&text).trim().to_string()
        });
        let deadline = Instant::now() + self.timeout;
        let mut response = None;
        let mut worker_failed_status = None;
        let result = loop {
            if cancel.load(Ordering::Relaxed) {
                break Err(AudioError::WorkerTimeout);
            }
            if response.is_none() {
                match receiver.try_recv() {
                    Ok(Ok(bytes)) if bytes.len() <= MAX_TRANSCRIPT_BYTES => response = Some(bytes),
                    Ok(Err(source)) => break Err(worker_io("pipe exchange", source)),
                    Ok(Ok(_)) => {
                        break Err(AudioError::WorkerProtocol("output exceeded 4096 bytes"));
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        break Err(AudioError::WorkerProtocol("pipe thread disconnected"));
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                }
            }
            match child.try_wait() {
                Ok(Some(status)) => {
                    if !status.success() {
                        worker_failed_status = Some(status);
                        break Err(AudioError::WorkerExit {
                            status,
                            stderr: String::new(),
                        });
                    }
                    if let Some(bytes) = response.take() {
                        break String::from_utf8(bytes)
                            .map(|text| {
                                if contains_keyword(&text) {
                                    KeywordDecision::Matched
                                } else {
                                    KeywordDecision::Absent
                                }
                            })
                            .map_err(|_| AudioError::WorkerProtocol("invalid UTF-8"));
                    }
                }
                Ok(None) => {}
                Err(source) => break Err(worker_io("poll exit", source)),
            }
            if Instant::now() >= deadline {
                break Err(AudioError::WorkerTimeout);
            }
            thread::sleep(Duration::from_millis(5));
        };
        // Kill on timeout/oversized output/cancellation before joining pipe I/O, then reap.
        let cleanup = stop_worker(&mut child);
        let joined = io_thread.join().map_err(|_| AudioError::WorkerPanicked);
        let stderr = err_thread.join().unwrap_or_default();
        // Do not swallow cleanup errors. If cleanup failed (child could not be killed/reaped),
        // that is a critical system failure that must be surfaced.
        cleanup?;
        if let Some(status) = worker_failed_status {
            return Err(AudioError::WorkerExit { status, stderr });
        }
        let decision = result?;
        joined?.map_err(|_| AudioError::WorkerProtocol("parent disconnected"))?;
        Ok(decision)
    }
}

fn worker_io(operation: &'static str, source: std::io::Error) -> AudioError {
    AudioError::WorkerIo { operation, source }
}

fn stop_worker(child: &mut Child) -> Result<(), AudioError> {
    // Attempt kill if child hasn't already exited.
    let kill_err = match child.kill() {
        Ok(()) => None,
        Err(source) => {
            // Check if already reaped/exited
            match child.try_wait() {
                Ok(Some(_)) => None,
                _ => Some(worker_io("kill", source)),
            }
        }
    };
    // ALWAYS call child.wait() to reap the zombie process.
    let wait_res = child.wait().map_err(|source| worker_io("reap", source));
    if let Some(err) = kill_err {
        return Err(err);
    }
    wait_res?;
    Ok(())
}

/// Run the one-shot native Whisper worker used by the listen example. Its parent
/// enforces the wall-clock deadline because native decode has no cancellation API.
/// Model files must already exist; this function never downloads anything.
/// Reads bounded 16 kHz PCM from stdin and writes the UTF-8 protocol response
/// to stdout. Returns model, native-runtime, protocol, or pipe I/O errors.
pub fn run_keyword_worker(model_directory: &Path) -> Result<(), AudioError> {
    let mut input = std::io::stdin().lock();
    let mut header = [0; 4];
    input
        .read_exact(&mut header)
        .map_err(|source| worker_io("read PCM header", source))?;
    let count = u32::from_le_bytes(header) as usize;
    if count == 0 || count > MAX_KEYWORD_SAMPLES {
        return Err(AudioError::WorkerProtocol("sample count must be 1..=48000"));
    }
    let mut bytes = vec![0; count * 4];
    input
        .read_exact(&mut bytes)
        .map_err(|source| worker_io("read PCM samples", source))?;
    drop(input);
    let samples: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    if samples.iter().any(|value| !value.is_finite()) {
        return Err(AudioError::WorkerProtocol("PCM contains nonfinite samples"));
    }
    let config = OfflineRecognizerConfig {
        model_config: OfflineModelConfig {
            whisper: OfflineWhisperModelConfig {
                encoder: Some(model_path(&model_directory.join("tiny-encoder.int8.onnx"))?),
                decoder: Some(model_path(&model_directory.join("tiny-decoder.int8.onnx"))?),
                language: Some("pt".to_owned()),
                task: Some("transcribe".to_owned()),
                ..Default::default()
            },
            tokens: Some(model_path(&model_directory.join("tiny-tokens.txt"))?),
            num_threads: 1,
            provider: Some("cpu".to_owned()),
            ..Default::default()
        },
        decoding_method: Some("greedy_search".to_owned()),
        ..Default::default()
    };
    let recognizer = OfflineRecognizer::create(&config)
        .ok_or(AudioError::Native("could not create Whisper recognizer"))?;
    let stream = recognizer.create_stream();
    stream.accept_waveform(16_000, &samples);
    recognizer.decode(&stream);
    let result = stream
        .get_result()
        .ok_or(AudioError::Native("Whisper returned no result"))?;
    if result.text.len() > MAX_TRANSCRIPT_BYTES {
        return Err(AudioError::WorkerProtocol(
            "Whisper transcript exceeded limit",
        ));
    }
    // This is the worker wire protocol, not library diagnostics.
    std::io::stdout()
        .lock()
        .write_all(result.text.as_bytes())
        .map_err(|source| worker_io("write transcript", source))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn keyword_uses_portuguese_safe_token_boundaries() {
        for text in ["Enton", "Oi, ENTON!", "enton, pode me ouvir?"] {
            assert!(contains_keyword(text));
        }
        for text in ["então", "Benton", "enton2", "venton", "", "E n t o n"] {
            assert!(!contains_keyword(text));
        }
    }

    #[test]
    fn worker_timeout_configuration_is_bounded() {
        let command = KeywordCommand {
            program: PathBuf::from("unused"),
            arguments: vec![],
            timeout: Duration::ZERO,
        };
        assert!(command.validate().is_err());
    }

    #[test]
    #[cfg(unix)]
    fn worker_protocol_distinguishes_matches_absence_and_malformed_output() {
        for (script, expected) in [
            (
                "cat >/dev/null; printf 'Oi, ENTON!'",
                Ok(KeywordDecision::Matched),
            ),
            (
                "cat >/dev/null; printf 'Benton'",
                Ok(KeywordDecision::Absent),
            ),
            ("cat >/dev/null; printf '\\377'", Err(())),
        ] {
            let command = KeywordCommand {
                program: PathBuf::from("sh"),
                arguments: vec![OsString::from("-c"), OsString::from(script)],
                timeout: Duration::from_secs(2),
            };
            let result = command.recognize(&[0.0; 512]);
            match expected {
                Ok(decision) => assert_eq!(result.unwrap(), decision),
                Err(()) => assert!(matches!(
                    result,
                    Err(AudioError::WorkerProtocol("invalid UTF-8"))
                )),
            }
        }
    }

    #[test]
    #[cfg(unix)]
    fn slow_worker_is_killed_and_reaped() {
        let command = KeywordCommand {
            program: PathBuf::from("sleep"),
            arguments: vec![OsString::from("5")],
            timeout: Duration::from_millis(40),
        };
        let start = Instant::now();
        assert!(command.recognize(&[0.0; 512]).is_err());
        assert!(start.elapsed() < Duration::from_secs(2));
    }

    #[test]
    #[cfg(unix)]
    fn f4_worker_always_reaped_and_cleanup_errors_surfaced() {
        let mut child = Command::new("sleep")
            .arg("10")
            .spawn()
            .expect("spawn sleep");
        assert!(stop_worker(&mut child).is_ok());
        assert!(child.try_wait().unwrap().is_some());
    }

    #[test]
    #[cfg(unix)]
    fn f5_keyword_recognize_cancellable_exits_promptly_on_cancellation() {
        let command = KeywordCommand {
            program: PathBuf::from("sleep"),
            arguments: vec![OsString::from("10")],
            timeout: Duration::from_secs(10),
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_clone = Arc::clone(&cancel);
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(30));
            cancel_clone.store(true, Ordering::Relaxed);
        });
        let start = Instant::now();
        let result = command.recognize_cancellable(&[0.0; 512], &cancel);
        let elapsed = start.elapsed();
        assert!(result.is_err());
        assert!(
            elapsed < Duration::from_millis(500),
            "cancelled worker must terminate promptly, took {elapsed:?}"
        );
    }

    #[test]
    #[cfg(unix)]
    fn f8_worker_stderr_is_captured_in_error() {
        let command = KeywordCommand {
            program: PathBuf::from("sh"),
            arguments: vec![
                OsString::from("-c"),
                OsString::from("echo 'cuda out of memory error' >&2; exit 42"),
            ],
            timeout: Duration::from_secs(2),
        };
        let result = command.recognize(&[0.0; 512]);
        match result {
            Err(AudioError::WorkerExit { status, stderr }) => {
                assert_eq!(status.code(), Some(42));
                assert!(
                    stderr.contains("cuda out of memory error"),
                    "stderr was: {stderr}"
                );
            }
            other => panic!("expected WorkerExit with captured stderr, got {other:?}"),
        }
    }
}
