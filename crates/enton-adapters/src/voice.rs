//! Text-to-speech synthesis and audio playback (RFC 0001 §6, Task 0004).
//!
//! Provides a bounded sentence-by-sentence streaming voice synthesizer backed by
//! `sherpa-onnx` Kokoro TTS (Brazilian Portuguese) and `cpal` audio output.
//!
//! Emits lifecycle events ([`PlaybackEvent::Started`], [`PlaybackEvent::Finished`],
//! [`PlaybackEvent::Cancelled`], [`PlaybackEvent::Failed`]) for each utterance to record what was actually heard
//! and supports immediate barge-in interruption upon new stimulus.

mod config;
mod events;
mod playback;
mod player;
mod worker;

pub use config::{VoiceConfig, VoiceError};
pub use enton_core::UtteranceId;
pub use events::{PlaybackEvent, PlaybackEventStats, UtteranceStageTimings, VoiceLatencyBreakdown};
pub use player::VoicePlayer;
