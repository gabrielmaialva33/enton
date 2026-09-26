//! Text-to-speech synthesis and audio playback (RFC 0001 §6, Task 0004).
//!
//! Provides a bounded sentence-by-sentence streaming voice synthesizer backed by
//! `sherpa-onnx` (Brazilian Portuguese) and `cpal` audio output. Two engines speak:
//! Kokoro multi-lang v1.0 ([`VoiceEngine::Kokoro`]) and Piper `pt_BR` faber-medium,
//! a VITS model ([`VoiceEngine::Piper`]). Playback runs at the model's native rate
//! when the output device takes it, so audio is not resampled twice.
//!
//! Emits lifecycle events ([`PlaybackEvent::Started`], [`PlaybackEvent::Finished`],
//! [`PlaybackEvent::Cancelled`], [`PlaybackEvent::Failed`]) for each utterance to record what was actually heard
//! and supports immediate barge-in interruption upon new stimulus.
//!
//! [`speakable`] strips what a reply should not say out loud (stage directions,
//! markup, emojis) before synthesis, and [`VoicePlayer::chime`] plays a short
//! acknowledgement, computed once, through the same queue and events as speech.

mod chime;
mod config;
mod events;
mod playback;
mod player;
mod text;
mod worker;

pub use config::{KokoroVoice, PiperVoice, VoiceConfig, VoiceEngine, VoiceError, VoiceModel};
pub use enton_core::UtteranceId;
pub use events::{PlaybackEvent, PlaybackEventStats, UtteranceStageTimings, VoiceLatencyBreakdown};
pub use player::VoicePlayer;
pub use text::speakable;
