use std::path::PathBuf;

use cpal::SampleFormat;

/// Errors from voice configuration, synthesis, and playback.
#[derive(Debug, thiserror::Error)]
pub enum VoiceError {
    /// A required model asset is missing or has the wrong file type.
    #[error(
        "voice error: {asset} not found: {path}. See the README (Voice and microphone) for the model layout."
    )]
    MissingAsset {
        /// Name of the model asset.
        asset: &'static str,
        /// Configured file or directory path.
        path: PathBuf,
    },
    /// A required native model path is not UTF-8.
    #[error("voice error: invalid UTF-8 in {0}")]
    InvalidPath(&'static str),
    /// Sentence queue capacity is outside 1..=64.
    #[error("voice error: queue_capacity must be between 1 and 64 (got {0})")]
    QueueCapacity(usize),
    /// Native TTS initialization failed.
    #[error("voice error: failed to create sherpa-onnx OfflineTts instance")]
    Initialization,
    /// Native TTS generation failed; no transcript is included in the error.
    #[error("voice error: sherpa-onnx TTS generation returned no audio")]
    Synthesis,
    /// A resampler could not be created for the requested rates.
    #[error("voice error: failed to create resampler from {input_hz} to {output_hz} Hz")]
    Resampler {
        /// Native model sample rate in hertz.
        input_hz: i32,
        /// Requested output sample rate in hertz.
        output_hz: i32,
    },
    /// The audio backend failed during a named operation.
    #[error("voice error: {operation}: {source}")]
    Device {
        /// Backend operation that failed.
        operation: &'static str,
        /// Original backend error.
        #[source]
        source: cpal::Error,
    },
    /// No matching output device is available; `None` denotes the default.
    #[error("voice error: output device not found: {0:?}")]
    NoOutputDevice(Option<String>),
    /// The output device's sample format is unsupported.
    #[error("voice error: unsupported output sample format: {0:?}")]
    SampleFormat(SampleFormat),
    /// The synthesis worker has disconnected from its bounded job queue.
    #[error("voice error: synthesis queue closed")]
    QueueClosed,
    /// The bounded submission or playback queue is full; the new job is rejected.
    #[error("voice error: queue full; new utterance rejected")]
    QueueFull,
    /// Empty or whitespace-only text cannot identify an utterance.
    #[error("voice error: speech text is empty")]
    EmptyText,
    /// All 16 subscriber slots are occupied.
    #[error("voice error: event subscriber limit reached")]
    SubscriberLimit,
}

/// Configuration for the local neural voice synthesizer (Kokoro via sherpa-onnx).
#[derive(Debug, Clone)]
pub struct VoiceConfig {
    /// Path to the ONNX model file (e.g. `model.onnx`).
    pub model_path: PathBuf,
    /// Path to the speaker voices binary (e.g. `voices.bin`).
    pub voices_path: PathBuf,
    /// Path to the phoneme tokens file (e.g. `tokens.txt`).
    pub tokens_path: PathBuf,
    /// Path to the espeak-ng-data directory.
    pub data_dir: PathBuf,
    /// Optional path to the dict directory (e.g. `dict/`).
    pub dict_dir: Option<PathBuf>,
    /// Optional path to a custom lexicon file.
    pub lexicon_path: Option<PathBuf>,
    /// Language code passed to Kokoro (e.g. `"pt-br"` or `"pt"`).
    pub lang: String,
    /// Speaker ID within voices.bin (`42` for `pf_dora`, `43` for `pm_alex`).
    pub speaker_id: i32,
    /// Playback speed factor (default 1.0).
    pub speed: f32,
    /// Number of worker threads for TTS inference (default 2).
    pub num_threads: i32,
    /// Bound (1..=64) for each synthesis job queue and the ready playback queue.
    /// New jobs are rejected when full; ready-queue overflow emits `Failed`.
    pub queue_capacity: usize,
    /// Specific CPAL device identifier string; `None` uses default output device.
    pub device_id: Option<String>,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        let base = std::env::var("HOME").map_or_else(
            |_| PathBuf::from("."),
            |h| {
                let v1_1 =
                    PathBuf::from(&h).join(".cache/enton/models/kokoro-int8-multi-lang-v1_1");
                if v1_1.is_dir() {
                    v1_1
                } else {
                    PathBuf::from(h).join(".cache/enton/models/kokoro-int8-multi-lang-v1_0")
                }
            },
        );
        let dict = base.join("dict");
        Self {
            model_path: base.join("model.int8.onnx"),
            voices_path: base.join("voices.bin"),
            tokens_path: base.join("tokens.txt"),
            data_dir: base.join("espeak-ng-data"),
            dict_dir: if dict.is_dir() { Some(dict) } else { None },
            lexicon_path: None,
            lang: "pt-br".to_string(),
            speaker_id: 42, // pf_dora (Brazilian Portuguese female)
            speed: 1.0,
            num_threads: 2,
            queue_capacity: 8,
            device_id: None,
        }
    }
}

impl VoiceConfig {
    /// Creates a voice configuration tailored for a specific hardware profile name.
    ///
    /// Configures 8 TTS worker threads for `"desktop"` and 2 worker threads for `"t1-ref"`.
    #[must_use]
    pub fn for_profile(profile_name: &str) -> Self {
        let num_threads = if profile_name == "desktop" { 8 } else { 2 };
        Self {
            num_threads,
            ..Self::default()
        }
    }

    /// Validates that model files exist on disk.
    ///
    /// # Errors
    /// Returns [`VoiceError`] if any required model file or directory is missing.
    pub fn validate(&self) -> Result<(), VoiceError> {
        if !self.model_path.is_file() {
            return Err(VoiceError::MissingAsset {
                asset: "model file",
                path: self.model_path.clone(),
            });
        }
        if !self.voices_path.is_file() {
            return Err(VoiceError::MissingAsset {
                asset: "voices file",
                path: self.voices_path.clone(),
            });
        }
        if !self.tokens_path.is_file() {
            return Err(VoiceError::MissingAsset {
                asset: "tokens file",
                path: self.tokens_path.clone(),
            });
        }
        if !self.data_dir.is_dir() {
            return Err(VoiceError::MissingAsset {
                asset: "espeak-ng-data directory",
                path: self.data_dir.clone(),
            });
        }
        if self.queue_capacity == 0 || self.queue_capacity > 64 {
            return Err(VoiceError::QueueCapacity(self.queue_capacity));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_config_validation_catches_missing_files() {
        let config = VoiceConfig {
            model_path: PathBuf::from("/non/existent/model.onnx"),
            voices_path: PathBuf::from("/non/existent/voices.bin"),
            tokens_path: PathBuf::from("/non/existent/tokens.txt"),
            data_dir: PathBuf::from("/non/existent/espeak-ng-data"),
            ..VoiceConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
