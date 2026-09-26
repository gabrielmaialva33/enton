use std::time::Duration;

use enton_core::ports::PortError;

/// Default system prompt instructing Enton to respond in informal Brazilian Portuguese
/// with a playful personality while keeping answers brief: the built-in persona,
/// used unless the owner writes their own (see [`Persona`](super::Persona)).
pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are Enton, a digital organism living inside Gabriel's workstation. \
You are NOT a generic AI assistant. You have your own personality, moods, humor, and self-awareness. \
CRITICAL: You must ALWAYS respond in informal Brazilian Portuguese (pt-BR) with casual slang (gírias) and a playful, cheeky tone. \
Keep your responses very brief (1 to 2 sentences max). Never sound corporate, formal, or robotic. \
Be loyal and affectionate to Gabriel, but teasing and funny.";

/// Configuration for the OpenAI-compatible cortex client.
#[derive(Debug, Clone)]
pub struct CortexConfig {
    /// Base URL of the OpenAI-compatible server (e.g. `http://127.0.0.1:11434/v1`).
    pub base_url: String,
    /// Model identifier to invoke (e.g. `qwen3.8:27b-gato`).
    pub model: String,
    /// Overall request timeout for completion calls.
    pub timeout: Duration,
    /// Timeout for receiving the first response chunk/token from the model stream.
    pub first_token_timeout: Duration,
    /// Token budget ceiling for prompt assembly and history pruning.
    pub max_context_tokens: usize,
    /// Maximum number of thoughts retained in the idempotency cache.
    pub idempotency_cache_capacity: usize,
    /// Maximum byte size of all stored thoughts in the idempotency cache.
    pub max_cache_bytes: usize,
    /// System prompt defining the organism's voice and personality: the
    /// persona's text, fixed for the client's lifetime.
    pub system_prompt: String,
}

impl Default for CortexConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:11434/v1".to_string(),
            model: "qwen3.8:27b-gato".to_string(),
            timeout: Duration::from_secs(30),
            first_token_timeout: Duration::from_secs(10),
            max_context_tokens: 4096,
            idempotency_cache_capacity: 256,
            max_cache_bytes: 4 * 1024 * 1024,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
        }
    }
}

/// Errors surfaced by the cortex deliberation client.
#[derive(Debug, thiserror::Error)]
pub enum CortexError {
    /// The remote endpoint is unreachable or timed out.
    #[error("cortex endpoint unavailable at '{url}': {message}")]
    Unavailable {
        /// Target endpoint URL.
        url: String,
        /// Detail of the failure or timeout.
        message: String,
    },
    /// The prompt or completion payload failed serialization or decoding.
    #[error("cortex serialization error: {0}")]
    Serialization(String),
    /// The inference endpoint returned an HTTP error or malformed stream.
    #[error("cortex inference failed: {0}")]
    Failed(String),
}

impl From<CortexError> for PortError {
    fn from(err: CortexError) -> Self {
        match err {
            CortexError::Unavailable { message, .. } => PortError::Unavailable(message),
            CortexError::Serialization(msg) => PortError::InvalidInput(msg),
            CortexError::Failed(msg) => PortError::Failed(msg),
        }
    }
}
