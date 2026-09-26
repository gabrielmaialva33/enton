//! Bounded microphone perception: capture, resampling, VAD, and keyword fallback.
//!
//! Native inference and device I/O stay outside the pure endpoint/keyword rules.
//! The expected model layout is in the README (Voice and microphone).

mod capture;
mod endpoint;
mod keyword;

pub use capture::{AudioCapture, AudioConfig, AudioStats, input_devices};
pub use endpoint::{
    AudioSegment, Endpoint, EndpointDetector, FRAME_MS, FRAME_SAMPLES, RetainedFrame, Retention,
    SAMPLE_RATE,
};
pub use keyword::{KeywordCommand, KeywordDecision, contains_keyword, run_keyword_worker};

use std::path::{Path, PathBuf};

/// Adapter failure with no raw audio or transcript included in its message.
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    /// A configuration value is outside its supported bounds.
    #[error("invalid audio configuration: {0}")]
    Configuration(&'static str),
    /// A model path cannot be passed to the native runtime.
    #[error("model path must be UTF-8 without NUL bytes: {0}")]
    InvalidModelPath(PathBuf),
    /// A required model file is absent.
    #[error("model file not found: {0}")]
    MissingModel(PathBuf),
    /// The audio backend could not enumerate, configure, or start a device.
    #[error("audio device operation failed: {0}")]
    Device(#[from] cpal::Error),
    /// No matching microphone is available.
    #[error("input device is unavailable")]
    NoInputDevice,
    /// The microphone sample format is unsupported.
    #[error("unsupported microphone format: {0}")]
    SampleFormat(cpal::SampleFormat),
    /// Native inference or resampler initialization failed.
    #[error("native audio runtime failed: {0}")]
    Native(&'static str),
    /// The keyword worker could not complete a protocol operation.
    #[error("keyword worker {operation} failed: {source}")]
    WorkerIo {
        /// Operation that failed, excluding audio and transcript content.
        operation: &'static str,
        /// Underlying operating-system failure.
        #[source]
        source: std::io::Error,
    },
    /// A keyword worker violated the bounded PCM/transcript protocol.
    #[error("keyword worker protocol error: {0}")]
    WorkerProtocol(&'static str),
    /// A keyword worker exceeded its configured wall-clock deadline.
    #[error("keyword worker timed out")]
    WorkerTimeout,
    /// A keyword worker exited unsuccessfully.
    #[error("keyword worker failed with status {status}: {stderr}")]
    WorkerExit {
        /// Exit status returned by the process.
        status: std::process::ExitStatus,
        /// Captured standard error output, capped to avoid unbounded allocation.
        stderr: String,
    },
    /// The keyword pipe thread panicked before returning its result.
    #[error("keyword worker I/O thread panicked")]
    WorkerPanicked,
}

fn model_path(path: &Path) -> Result<String, AudioError> {
    let value = path
        .to_str()
        .filter(|value| !value.contains('\0'))
        .ok_or_else(|| AudioError::InvalidModelPath(path.to_owned()))?;
    if !path.is_file() {
        return Err(AudioError::MissingModel(path.to_owned()));
    }
    Ok(value.to_owned())
}
