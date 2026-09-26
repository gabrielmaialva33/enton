//! Inference contracts implemented by adapters, with no runtime dependencies.

use std::{error::Error, fmt, future::Future};

use crate::{Reason, ThoughtId};

/// A portable adapter failure without backend-specific error types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortError {
    /// The requested port is not available.
    Unavailable(String),
    /// The input to the port was invalid.
    InvalidInput(String),
    /// The port operation failed.
    Failed(String),
}

impl fmt::Display for PortError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(message) => write!(formatter, "port unavailable: {message}"),
            Self::InvalidInput(message) => write!(formatter, "invalid port input: {message}"),
            Self::Failed(message) => write!(formatter, "port failed: {message}"),
        }
    }
}

impl Error for PortError {}

/// The speaker role in a conversation turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TurnRole {
    /// The human user speaking.
    User,
    /// The assistant responding.
    Assistant,
}

/// A single conversational turn preserved for context.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ConversationTurn {
    /// The speaker role in this turn.
    pub role: TurnRole,
    /// The text content of the turn.
    pub content: String,
}

impl ConversationTurn {
    #[must_use]
    /// Creates a turn from the user's message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: TurnRole::User,
            content: content.into(),
        }
    }

    #[must_use]
    /// Creates a turn from the assistant's response.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: TurnRole::Assistant,
            content: content.into(),
        }
    }
}

/// A request sent to the cortex adapter for deliberation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ThoughtRequest {
    /// The sequential, deterministic thought identifier.
    pub thought: ThoughtId,
    /// The trigger reason that caused ignition.
    pub reason: Reason,
    /// The optional audio transcription or text cue heard from the environment.
    pub transcript: Option<String>,
    /// Bounded recent conversation history.
    pub history: Vec<ConversationTurn>,
}

/// A cortex adapter; the thought identifier is an idempotency key.
pub trait Cortex {
    /// Generate a response to the given thought request.
    fn think(
        &self,
        request: &ThoughtRequest,
    ) -> impl Future<Output = Result<String, PortError>> + Send;
}

/// Transcribe normalized mono PCM at the supplied sample rate in hertz.
pub trait SpeechToText {
    /// Convert audio samples to text.
    fn transcribe(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> impl Future<Output = Result<String, PortError>> + Send;
}

/// Synthesize normalized mono PCM at the requested sample rate in hertz.
pub trait TextToSpeech {
    /// Convert text to audio samples.
    fn synthesize(
        &self,
        text: &str,
        sample_rate: u32,
    ) -> impl Future<Output = Result<Vec<f32>, PortError>> + Send;
}
