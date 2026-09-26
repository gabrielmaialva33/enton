//! I/O adapters and physical interfaces for the Enton digital organism.

pub mod body;
pub mod clock;

#[cfg(feature = "audio")]
pub mod audio;
#[cfg(feature = "cortex")]
pub mod cortex;
#[cfg(feature = "soul")]
mod organism_snapshot;
#[cfg(feature = "soul")]
pub mod soul;
#[cfg(feature = "voice")]
pub mod voice;

pub use body::{BodyError, read_body_signals, read_body_signals_from};
pub use clock::MonotonicClock;
#[cfg(feature = "cortex")]
pub use cortex::{CortexConfig, CortexError, OpenAiCortex};
#[cfg(feature = "soul")]
pub use soul::{ActionStatus, SeqNo, Soul, SoulConfig};
#[cfg(feature = "voice")]
pub use voice::{PlaybackEvent, UtteranceId, VoiceConfig, VoiceError, VoicePlayer};
