//! Cortex client for OpenAI-compatible LLM servers (RFC 0001 §6).
//!
//! Provides an asynchronous [`Cortex`] adapter targeting Ollama or OpenAI-compatible
//! endpoints with middle-out history pruning, single-flight locking, and thought idempotency.

mod cache;
mod chunking;
mod client;
mod config;
mod prompt;
mod wire;

pub use chunking::{
    MIN_FIRST_CLAUSE_CHARS, extract_completed_chunks, extract_completed_sentences,
    extract_reply_chunks,
};
pub use client::{OpenAiCortex, ThoughtStageTimings};
pub use config::{CortexConfig, CortexError, DEFAULT_SYSTEM_PROMPT};
pub use prompt::prune_history_middle_out;
