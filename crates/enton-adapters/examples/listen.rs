//! Microphone diagnostics; no audio is saved to disk or sent to a cloud service.

// A CLI binary talks to the terminal by design; only libraries must not print.
#![allow(clippy::print_stdout, clippy::print_stderr)]

#[cfg(not(feature = "audio"))]
fn main() {
    eprintln!("Enable audio: cargo run -p enton-adapters --features audio --example listen");
}

#[cfg(feature = "audio")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    enabled::run()
}

#[cfg(feature = "audio")]
mod enabled {
    use std::{
        error::Error,
        ffi::OsString,
        path::PathBuf,
        sync::mpsc::RecvTimeoutError,
        time::{Duration, Instant},
    };

    use enton_adapters::audio::{
        AudioCapture, AudioConfig, KeywordCommand, input_devices, run_keyword_worker,
    };

    pub(super) fn run() -> Result<(), Box<dyn Error>> {
        let arguments: Vec<String> = std::env::args().skip(1).collect();
        if arguments
            .first()
            .is_some_and(|value| value == "--keyword-worker")
        {
            let [_, directory] = arguments.as_slice() else {
                return Err("worker requires exactly one model directory".into());
            };
            run_keyword_worker(&PathBuf::from(directory))?;
            return Ok(());
        }
        let mut models = PathBuf::from(
            std::env::var_os("HOME").ok_or("HOME is unset; cannot locate model cache")?,
        )
        .join(".cache/enton/models");
        let mut device = None;
        let mut seconds = 10_u64;
        let mut vad_only = false;
        let mut remote = None;
        let mut remote_executable = None;
        let mut remote_models = None;
        let mut args = arguments.iter();
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--list" => {
                    for (id, description) in input_devices()? {
                        println!("{id}\t{description}");
                    }
                    return Ok(());
                }
                "--help" => {
                    println!(
                        "listen [--list] [--seconds 1..3600] [--device ID] [--models DIRECTORY] [--vad-only]\nRemote keyword fallback: --keyword-ssh HOST --remote-listen ABSOLUTE_PATH --remote-models ABSOLUTE_DIRECTORY\nDefault: local Whisper tiny (T2/T3); use --vad-only for T1 diagnostics."
                    );
                    return Ok(());
                }
                "--models" => models = PathBuf::from(args.next().ok_or("--models needs a path")?),
                "--device" => device = Some(args.next().ok_or("--device needs an ID")?.clone()),
                "--seconds" => seconds = args.next().ok_or("--seconds needs a number")?.parse()?,
                "--vad-only" => vad_only = true,
                "--keyword-ssh" => {
                    remote = Some(args.next().ok_or("--keyword-ssh needs a host")?.clone());
                }
                "--remote-listen" => {
                    remote_executable =
                        Some(args.next().ok_or("--remote-listen needs a path")?.clone());
                }
                "--remote-models" => {
                    remote_models =
                        Some(args.next().ok_or("--remote-models needs a path")?.clone());
                }
                _ => return Err(format!("unknown option: {argument}").into()),
            }
        }
        if !(1..=3_600).contains(&seconds) {
            return Err("--seconds must be 1..=3600".into());
        }
        if vad_only && remote.is_some() {
            return Err("--vad-only and --keyword-ssh are mutually exclusive".into());
        }
        if remote.is_none() && (remote_executable.is_some() || remote_models.is_some()) {
            return Err("remote paths require --keyword-ssh".into());
        }
        let mut config = AudioConfig::new(models.join("silero_vad.onnx"));
        config.device_id = device;
        if !vad_only {
            config.keyword = Some(if let Some(host) = remote {
                remote_keyword(host, remote_executable, remote_models)?
            } else {
                local_keyword(models.join("sherpa-onnx-whisper-tiny"))?
            });
        }
        let capture = AudioCapture::start(config)?;
        eprintln!(
            "Listening for {seconds}s; PCM is retained only in a 30s RAM ring. Keyword pending/unavailable is NOT a negative match."
        );
        let deadline = Instant::now() + Duration::from_secs(seconds);
        while Instant::now() < deadline {
            match capture.recv_timeout(Duration::from_millis(100)) {
                Ok(segment) => println!(
                    "start_ms={} end_ms={} samples={} cue={:?} keyword={:?} forced={}",
                    segment.start_sample / 16,
                    segment.end_sample / 16,
                    segment.samples.len(),
                    segment.cue,
                    segment.keyword,
                    segment.forced_endpoint,
                ),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("audio workers disconnected".into());
                }
            }
        }
        eprintln!("Audio statistics: {:?}", capture.stats());
        Ok(())
    }

    fn local_keyword(directory: PathBuf) -> Result<KeywordCommand, Box<dyn Error>> {
        for name in [
            "tiny-encoder.int8.onnx",
            "tiny-decoder.int8.onnx",
            "tiny-tokens.txt",
        ] {
            if !directory.join(name).is_file() {
                return Err(format!(
                    "missing {}; see the README (Voice and microphone) or use --vad-only",
                    directory.join(name).display()
                )
                .into());
            }
        }
        Ok(KeywordCommand {
            program: std::env::current_exe()?,
            arguments: vec![
                OsString::from("--keyword-worker"),
                directory.into_os_string(),
            ],
            timeout: Duration::from_secs(10),
        })
    }

    fn remote_keyword(
        host: String,
        executable: Option<String>,
        directory: Option<String>,
    ) -> Result<KeywordCommand, Box<dyn Error>> {
        let executable = executable.ok_or("remote fallback requires --remote-listen")?;
        let directory = directory.ok_or("remote fallback requires --remote-models")?;
        if !executable.starts_with('/') || !directory.starts_with('/') {
            return Err("remote paths must be absolute".into());
        }
        Ok(KeywordCommand {
            program: PathBuf::from("ssh"),
            arguments: vec![
                "-T".into(),
                "-oBatchMode=yes".into(),
                "-oConnectTimeout=3".into(),
                "--".into(),
                host.into(),
                format!(
                    "exec timeout --signal=KILL 10s {} --keyword-worker {}",
                    quote(&executable),
                    quote(&directory)
                )
                .into(),
            ],
            timeout: Duration::from_secs(12),
        })
    }

    fn quote(value: &str) -> String {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}
