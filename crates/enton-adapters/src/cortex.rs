//! Cortex client for OpenAI-compatible LLM servers (RFC 0001 §6).
//!
//! Provides an asynchronous [`Cortex`] adapter targeting Ollama or OpenAI-compatible
//! endpoints with middle-out history pruning, single-flight locking, and thought idempotency.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use enton_core::ThoughtId;
use enton_core::ports::{ConversationTurn, Cortex, PortError, ThoughtRequest, TurnRole};
use serde::{Deserialize, Serialize};

/// Default system prompt instructing Enton to respond in informal Brazilian Portuguese
/// with a playful personality while keeping answers brief.
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
    /// System prompt defining the organism's voice and personality.
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

/// Prunes conversation history from the middle outward to fit within a token budget.
///
/// Adapted from Goose (Apache-2.0):
/// <https://github.com/aaif-goose/goose/blob/main/crates/goose-context-management/src/summarize.rs>
///
/// This keeps the earliest conversational context and the most recent turns while
/// removing stale turns from the middle first.
#[must_use]
pub fn prune_history_middle_out(
    history: &[ConversationTurn],
    max_tokens: usize,
) -> Vec<ConversationTurn> {
    if history.is_empty() {
        return Vec::new();
    }

    // Heuristic estimation: ~4 chars per token plus 4 tokens message framing overhead.
    let turn_tokens = |turn: &ConversationTurn| -> usize { turn.content.len().div_ceil(4) + 4 };

    let mut total_tokens: usize = history.iter().map(turn_tokens).sum();
    if total_tokens <= max_tokens {
        return history.to_vec();
    }

    let count = history.len();
    let mid = count / 2;

    // Sort candidate removal indices by distance to the middle (closest removed first).
    let mut removal_order: Vec<usize> = (0..count).collect();
    removal_order.sort_by_key(|&idx| idx.abs_diff(mid));

    let mut removed = vec![false; count];
    for idx in removal_order {
        if total_tokens <= max_tokens {
            break;
        }
        if let Some(turn) = history.get(idx) {
            if let Some(r) = removed.get_mut(idx) {
                *r = true;
            }
            total_tokens = total_tokens.saturating_sub(turn_tokens(turn));
        }
    }

    history
        .iter()
        .zip(removed.iter())
        .filter_map(
            |(turn, &is_removed)| {
                if is_removed { None } else { Some(turn.clone()) }
            },
        )
        .collect()
}

#[derive(Debug, Serialize)]
struct OutgoingChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest<'a> {
    model: &'a str,
    messages: Vec<OutgoingChatMessage<'a>>,
    stream: bool,
    reasoning_effort: &'a str,
}

#[derive(Debug, Deserialize)]
struct StreamCompletionChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Debug, Deserialize)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
}

/// Minimum character length required before emitting an early first clause to TTS.
///
/// Kokoro is a neural TTS model whose phonemizer and prosody predictor require
/// sufficient phonetic context to generate natural pitch contours and prevent
/// choppy, clipped audio on short fragments (e.g. "Oi," or "Sim,"). A threshold
/// of 20 characters provides 3–5 words of prosodic context while enabling
/// sub-second time-to-first-audio (TTFA).
pub const MIN_FIRST_CLAUSE_CHARS: usize = 20;

/// Known abbreviations that should not trigger sentence termination when ending in a dot.
const KNOWN_ABBREVIATIONS: &[&str] = &[
    "dr", "dra", "sr", "sra", "prof", "profa", "ex", "etc", "vs", "p.ex", "mr", "mrs", "ms",
];

/// Checks if a dot or comma at `byte_idx` in `text` is part of a decimal number (e.g. "3,5" or "3.14").
fn is_decimal_separator(text: &str, byte_idx: usize) -> bool {
    let bytes = text.as_bytes();
    if byte_idx == 0 || byte_idx + 1 >= bytes.len() {
        return false;
    }
    let prev = bytes.get(byte_idx.saturating_sub(1)).copied();
    let next = bytes.get(byte_idx + 1).copied();
    match (prev, next) {
        (Some(p), Some(n)) => p.is_ascii_digit() && n.is_ascii_digit(),
        _ => false,
    }
}

/// Checks if the dot at `dot_byte_pos` in `text` belongs to a known abbreviation.
fn is_abbreviation_dot(text: &str, dot_byte_pos: usize) -> bool {
    let Some(prefix) = text.get(..dot_byte_pos) else {
        return false;
    };
    let mut word_start = dot_byte_pos;
    for (idx, ch) in prefix.char_indices().rev() {
        if ch.is_alphabetic() || ch == '.' {
            word_start = idx;
        } else {
            break;
        }
    }
    if word_start >= dot_byte_pos {
        return false;
    }
    let Some(word) = prefix.get(word_start..dot_byte_pos) else {
        return false;
    };
    let lower = word.to_lowercase();
    KNOWN_ABBREVIATIONS.contains(&lower.as_str())
}

#[derive(Debug)]
struct BoundaryMatch {
    final_end_pos: usize,
    remainder_offset: usize,
    #[expect(dead_code, reason = "retained for debugging and inspection")]
    is_clause: bool,
}

#[expect(
    clippy::too_many_lines,
    reason = "scans clauses, sentences, decimals, abbreviations, and quotes in a single pass"
)]
fn find_boundary(buffer: &str, search_start: usize, allow_clause: bool) -> Option<BoundaryMatch> {
    let slice = buffer.get(search_start..)?;
    let mut chars = slice.char_indices().peekable();

    while let Some((rel_idx, ch)) = chars.next() {
        let abs_pos = search_start + rel_idx;

        if ch == '.' || ch == '!' || ch == '?' || ch == '\n' {
            if ch == '.' && is_decimal_separator(buffer, abs_pos) {
                continue;
            }
            if ch == '.' && is_abbreviation_dot(buffer, abs_pos) {
                continue;
            }

            let mut end_pos = abs_pos + ch.len_utf8();
            while let Some(&(_, next_ch)) = chars.peek() {
                if next_ch == '.' || next_ch == '!' || next_ch == '?' {
                    end_pos += next_ch.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }

            let mut final_end_pos = end_pos;
            while let Some(&(_, next_ch)) = chars.peek() {
                if next_ch == '"'
                    || next_ch == '\''
                    || next_ch == '”'
                    || next_ch == '’'
                    || next_ch == ')'
                    || next_ch == ']'
                {
                    final_end_pos += next_ch.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }

            let is_terminal = if let Some(trailing) = buffer.get(final_end_pos..) {
                trailing.chars().next().is_none_or(char::is_whitespace)
            } else {
                true
            };

            if is_terminal {
                let remainder_offset = buffer
                    .get(final_end_pos..)
                    .and_then(|rem| {
                        rem.char_indices()
                            .find(|(_, c)| !c.is_whitespace())
                            .map(|(offset, _)| final_end_pos + offset)
                    })
                    .unwrap_or(buffer.len());

                return Some(BoundaryMatch {
                    final_end_pos,
                    remainder_offset,
                    is_clause: false,
                });
            }
        } else if allow_clause && (ch == ',' || ch == ';' || ch == ':' || ch == '—') {
            if ch == ',' && is_decimal_separator(buffer, abs_pos) {
                continue;
            }

            let end_pos = abs_pos + ch.len_utf8();
            let mut final_end_pos = end_pos;
            while let Some(&(_, next_ch)) = chars.peek() {
                if next_ch == '"'
                    || next_ch == '\''
                    || next_ch == '”'
                    || next_ch == '’'
                    || next_ch == ')'
                    || next_ch == ']'
                {
                    final_end_pos += next_ch.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }

            let is_valid_clause = if ch == '—' {
                true
            } else if let Some(trailing) = buffer.get(final_end_pos..) {
                trailing.chars().next().is_none_or(char::is_whitespace)
            } else {
                true
            };

            if is_valid_clause {
                let candidate_chars = buffer
                    .get(..final_end_pos)
                    .map_or(0, |s| s.trim().chars().count());

                if candidate_chars >= MIN_FIRST_CLAUSE_CHARS {
                    let remainder_offset = buffer
                        .get(final_end_pos..)
                        .and_then(|rem| {
                            rem.char_indices()
                                .find(|(_, c)| !c.is_whitespace())
                                .map(|(offset, _)| final_end_pos + offset)
                        })
                        .unwrap_or(buffer.len());

                    return Some(BoundaryMatch {
                        final_end_pos,
                        remainder_offset,
                        is_clause: true,
                    });
                }
            }
        }
    }

    None
}

fn extract_completed_chunks_inner(
    buffer: &mut String,
    first_chunk_sent: &mut bool,
    allow_first_clause: bool,
) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut search_start = 0;

    while search_start < buffer.len() {
        let allow_clause = allow_first_clause && !*first_chunk_sent;
        let Some(boundary) = find_boundary(buffer, search_start, allow_clause) else {
            break;
        };

        let chunk = buffer
            .get(..boundary.final_end_pos)
            .map(|s| s.trim().to_string())
            .unwrap_or_default();

        buffer.drain(..boundary.remainder_offset.min(buffer.len()));
        search_start = 0;

        if !chunk.is_empty() {
            *first_chunk_sent = true;
            chunks.push(chunk);
        }
    }

    chunks
}

/// Extracts speech chunks from an accumulator buffer.
///
/// If `first_chunk_sent` is false, this extracts the first chunk at the first
/// clause boundary (',', ';', ':', '—') once the accumulated text reaches
/// [`MIN_FIRST_CLAUSE_CHARS`], or at the first terminal sentence boundary
/// ('.', '!', '?', '\n').
///
/// Once the first chunk has been emitted (`*first_chunk_sent == true`), all
/// subsequent chunks maintain full sentence granularity.
///
/// Respects decimals (e.g. "3,5"), abbreviations ("Dr.", "Sr."), and ellipses ("...").
/// Drains emitted chunks from `buffer` and leaves trailing incomplete text.
#[must_use]
pub fn extract_completed_chunks(buffer: &mut String, first_chunk_sent: &mut bool) -> Vec<String> {
    extract_completed_chunks_inner(buffer, first_chunk_sent, true)
}

/// Extracts completed sentences from an accumulator buffer without clause splitting.
///
/// Looks for terminal punctuation ('.', '!', '?', or '\n') followed by whitespace,
/// while respecting ellipses ('...'), abbreviations ("Dr."), and trailing quotes.
/// Drains completed sentences from `buffer` and leaves trailing incomplete text.
#[must_use]
pub fn extract_completed_sentences(buffer: &mut String) -> Vec<String> {
    let mut dummy_first = true;
    extract_completed_chunks_inner(buffer, &mut dummy_first, false)
}

/// Splits an entire text reply into synthesis chunks.
///
/// Emits the first clause early if it reaches [`MIN_FIRST_CLAUSE_CHARS`],
/// and splits all subsequent text by sentence boundaries.
#[must_use]
pub fn extract_reply_chunks(text: &str) -> Vec<String> {
    let mut buf = text.to_string();
    let mut first_sent = false;
    let mut chunks = extract_completed_chunks(&mut buf, &mut first_sent);
    let leftover = buf.trim();
    if !leftover.is_empty() {
        chunks.push(leftover.to_string());
    }
    chunks
}

/// Bounded FIFO idempotency cache tracking the most recent thoughts and their responses.
#[derive(Debug)]
pub(crate) struct IdempotencyCache {
    capacity: usize,
    max_bytes: usize,
    total_bytes: usize,
    entries: HashMap<ThoughtId, String>,
    order: VecDeque<ThoughtId>,
}

impl IdempotencyCache {
    pub(crate) fn new(capacity: usize, max_bytes: usize) -> Self {
        Self {
            capacity,
            max_bytes,
            total_bytes: 0,
            entries: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
        }
    }

    pub(crate) fn get(&self, thought: ThoughtId) -> Option<&String> {
        self.entries.get(&thought)
    }

    pub(crate) fn insert(&mut self, thought: ThoughtId, reply: String) {
        if self.capacity == 0 || reply.len() > self.max_bytes {
            return;
        }

        if let Some(old) = self.entries.get(&thought) {
            let old_len = old.len();
            self.total_bytes = self.total_bytes.saturating_sub(old_len);
            while (self.total_bytes + reply.len() > self.max_bytes) && !self.entries.is_empty() {
                if let Some(oldest) = self.order.pop_front() {
                    if oldest == thought {
                        continue;
                    }
                    if let Some(removed) = self.entries.remove(&oldest) {
                        self.total_bytes = self.total_bytes.saturating_sub(removed.len());
                    }
                } else {
                    break;
                }
            }
            self.total_bytes += reply.len();
            self.entries.insert(thought, reply);
            return;
        }

        while (self.entries.len() >= self.capacity
            || (self.total_bytes + reply.len() > self.max_bytes))
            && !self.entries.is_empty()
        {
            if let Some(oldest) = self.order.pop_front() {
                if let Some(removed) = self.entries.remove(&oldest) {
                    self.total_bytes = self.total_bytes.saturating_sub(removed.len());
                }
            } else {
                break;
            }
        }

        self.total_bytes += reply.len();
        self.entries.insert(thought, reply);
        self.order.push_back(thought);
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.total_bytes = 0;
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Monotonic timestamps recorded across the lifecycle of a cortex deliberation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThoughtStageTimings {
    /// Identifier of the deliberation thought.
    pub thought: ThoughtId,
    /// Instant when the deliberation request was initiated.
    pub request_start: Instant,
    /// Instant when the first token delta was parsed from the SSE stream.
    pub first_token: Option<Instant>,
    /// Instant when the first chunk (clause or sentence) was emitted to the receiver.
    pub first_chunk: Option<Instant>,
    /// Instant when the stream completed ([DONE]) or terminated.
    pub stream_end: Option<Instant>,
}

/// An OpenAI-compatible cortex client with middle-out history pruning,
/// single-flight locking, and thought idempotency caching.
#[derive(Debug, Clone)]
pub struct OpenAiCortex {
    config: CortexConfig,
    client: reqwest::Client,
    in_flight: Arc<tokio::sync::Mutex<()>>,
    cache: Arc<tokio::sync::RwLock<IdempotencyCache>>,
    stage_timings: Arc<tokio::sync::RwLock<VecDeque<ThoughtStageTimings>>>,
}

impl OpenAiCortex {
    /// Creates a new cortex client with the supplied configuration.
    #[must_use]
    pub fn new(config: CortexConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .unwrap_or_default();

        let cache =
            IdempotencyCache::new(config.idempotency_cache_capacity, config.max_cache_bytes);

        Self {
            config,
            client,
            in_flight: Arc::new(tokio::sync::Mutex::new(())),
            cache: Arc::new(tokio::sync::RwLock::new(cache)),
            stage_timings: Arc::new(tokio::sync::RwLock::new(VecDeque::with_capacity(64))),
        }
    }

    /// Creates a cortex client targeting a specific base URL and model name.
    #[must_use]
    pub fn with_endpoint(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(CortexConfig {
            base_url: base_url.into(),
            model: model.into(),
            ..CortexConfig::default()
        })
    }

    /// Returns the stage timings recorded for a specific thought, if available.
    pub async fn stage_timings(&self, thought: ThoughtId) -> Option<ThoughtStageTimings> {
        let guard = self.stage_timings.read().await;
        guard.iter().find(|t| t.thought == thought).copied()
    }

    /// Returns the most recently recorded thought stage timings.
    pub async fn last_stage_timings(&self) -> Option<ThoughtStageTimings> {
        let guard = self.stage_timings.read().await;
        guard.back().copied()
    }

    /// Clears the idempotency cache.
    pub async fn clear_cache(&self) {
        self.cache.write().await.clear();
    }

    /// Returns the number of cached thought entries.
    #[must_use]
    pub async fn cache_len(&self) -> usize {
        self.cache.read().await.len()
    }

    /// Returns the total bytes of responses currently stored in the idempotency cache.
    #[must_use]
    pub async fn cache_bytes(&self) -> usize {
        self.cache.read().await.total_bytes()
    }

    /// Returns `true` if the idempotency cache contains no entries.
    #[must_use]
    pub async fn is_cache_empty(&self) -> bool {
        self.cache.read().await.is_empty()
    }

    /// Returns a reference to the active configuration.
    #[must_use]
    pub fn config(&self) -> &CortexConfig {
        &self.config
    }

    /// Deliberates on a stimulus by streaming SSE completion chunks and yielding
    /// completed sentences as they arrive.
    ///
    /// Respects the bounded idempotency cache and single-flight inference lock.
    ///
    /// # Errors
    /// Returns [`PortError`] if the endpoint is unreachable or returns an error.
    #[expect(
        clippy::significant_drop_tightening,
        reason = "single-flight inference lock must remain held across HTTP send and stream consumption"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "deliberation setup, payload serialization, HTTP dispatch, and stream initialization"
    )]
    pub async fn think_stream(
        &self,
        request: &ThoughtRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<String>, PortError> {
        let request_start = Instant::now();
        {
            let mut timings = self.stage_timings.write().await;
            if timings.len() >= 64 {
                timings.pop_front();
            }
            timings.push_back(ThoughtStageTimings {
                thought: request.thought,
                request_start,
                first_token: None,
                first_chunk: None,
                stream_end: None,
            });
        }

        if let Some(rx) = reply_from_cache(&self.cache, request.thought, &self.stage_timings).await
        {
            return Ok(rx);
        }

        // Single-flight lock: only one inference request may be in flight to the LLM
        // at a time (RFC 0001 §3). The lock is transferred to consume_sse_stream to hold
        // until the final SSE token is received.
        let flight_guard = Arc::clone(&self.in_flight).lock_owned().await;

        if let Some(rx) = reply_from_cache(&self.cache, request.thought, &self.stage_timings).await
        {
            return Ok(rx);
        }

        let user_prompt = match &request.transcript {
            Some(transcript) if !transcript.trim().is_empty() => transcript.clone(),
            _ => format!("[Trigger reason: {:?}]", request.reason),
        };

        let fixed_tokens =
            (self.config.system_prompt.len().div_ceil(4) + 4) + (user_prompt.len().div_ceil(4) + 4);
        let history_budget = self.config.max_context_tokens.saturating_sub(fixed_tokens);
        let pruned_history = prune_history_middle_out(&request.history, history_budget);

        let messages = assemble_messages(&self.config, &user_prompt, &pruned_history);

        let endpoint_url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );
        let payload = ChatCompletionRequest {
            model: &self.config.model,
            messages,
            stream: true,
            reasoning_effort: "none",
        };

        let send_future = self.client.post(&endpoint_url).json(&payload).send();
        let response = tokio::time::timeout(self.config.first_token_timeout, send_future)
            .await
            .map_err(|_| {
                PortError::Unavailable(format!(
                    "request timed out after {:?} waiting for response headers",
                    self.config.first_token_timeout
                ))
            })?
            .map_err(|err| {
                if err.is_timeout() {
                    PortError::Unavailable(format!(
                        "request timed out after {:?}",
                        self.config.timeout
                    ))
                } else if err.is_connect() {
                    PortError::Unavailable(format!("endpoint unreachable at {endpoint_url}: {err}"))
                } else {
                    PortError::Failed(format!("HTTP error: {err}"))
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(PortError::Failed(format!("HTTP {status}: {body}")));
        }

        let mut response = response;
        let first_chunk = tokio::time::timeout(self.config.first_token_timeout, response.chunk())
            .await
            .map_err(|_| {
                PortError::Unavailable(format!(
                    "stream timed out after {:?} waiting for first token",
                    self.config.first_token_timeout
                ))
            })?
            .map_err(|err| PortError::Failed(format!("HTTP chunk error: {err}")))?;

        let Some(first_chunk) = first_chunk else {
            return Err(PortError::Unavailable(
                "cortex returned empty stream".to_string(),
            ));
        };

        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let cache = Arc::clone(&self.cache);
        let stage_timings = Arc::clone(&self.stage_timings);
        let thought_id = request.thought;
        let total_timeout = self.config.timeout;
        let first_chunk_vec = first_chunk.to_vec();

        tokio::spawn(consume_sse_stream(
            response,
            first_chunk_vec,
            tx,
            cache,
            stage_timings,
            thought_id,
            flight_guard,
            total_timeout,
        ));

        Ok(rx)
    }
}

async fn reply_from_cache(
    cache: &tokio::sync::RwLock<IdempotencyCache>,
    thought_id: ThoughtId,
    stage_timings: &Arc<tokio::sync::RwLock<VecDeque<ThoughtStageTimings>>>,
) -> Option<tokio::sync::mpsc::Receiver<String>> {
    let cached = {
        let guard = cache.read().await;
        guard.get(thought_id)?.clone()
    };
    let timings = Arc::clone(stage_timings);
    let (tx, rx) = tokio::sync::mpsc::channel(16);
    tokio::spawn(async move {
        let mut buf = cached;
        let mut first_chunk_sent = false;
        let chunks = extract_completed_chunks(&mut buf, &mut first_chunk_sent);
        let mut first_recorded = false;
        for s in chunks {
            if !first_recorded {
                first_recorded = true;
                let now = Instant::now();
                let mut guard = timings.write().await;
                if let Some(t) = guard.iter_mut().find(|t| t.thought == thought_id) {
                    if t.first_token.is_none() {
                        t.first_token = Some(now);
                    }
                    if t.first_chunk.is_none() {
                        t.first_chunk = Some(now);
                    }
                }
            }
            if tx.send(s).await.is_err() {
                return;
            }
        }
        let leftover = buf.trim();
        if !leftover.is_empty() {
            if !first_recorded {
                let now = Instant::now();
                let mut guard = timings.write().await;
                if let Some(t) = guard.iter_mut().find(|t| t.thought == thought_id) {
                    if t.first_token.is_none() {
                        t.first_token = Some(now);
                    }
                    if t.first_chunk.is_none() {
                        t.first_chunk = Some(now);
                    }
                }
            }
            drop(tx.send(leftover.to_string()).await);
        }
        let now = Instant::now();
        let mut guard = timings.write().await;
        if let Some(t) = guard.iter_mut().find(|t| t.thought == thought_id) {
            t.stream_end = Some(now);
        }
    });
    Some(rx)
}

fn assemble_messages<'a>(
    config: &'a CortexConfig,
    user_prompt: &'a str,
    pruned_history: &'a [ConversationTurn],
) -> Vec<OutgoingChatMessage<'a>> {
    let mut messages = Vec::with_capacity(pruned_history.len() + 2);
    messages.push(OutgoingChatMessage {
        role: "system",
        content: &config.system_prompt,
    });

    for turn in pruned_history {
        let role_str = match turn.role {
            TurnRole::User => "user",
            TurnRole::Assistant => "assistant",
        };
        messages.push(OutgoingChatMessage {
            role: role_str,
            content: &turn.content,
        });
    }

    messages.push(OutgoingChatMessage {
        role: "user",
        content: user_prompt,
    });

    messages
}

const MAX_RAW_SSE_BYTES: usize = 64 * 1024;
const MAX_COMPLETION_BYTES: usize = 512 * 1024;

async fn send_sentence_bounded(
    tx: &tokio::sync::mpsc::Sender<String>,
    sentence: String,
    deadline: tokio::time::Instant,
) -> bool {
    let now = tokio::time::Instant::now();
    if now >= deadline || tx.is_closed() {
        return false;
    }
    matches!(
        tokio::time::timeout_at(deadline, tx.send(sentence)).await,
        Ok(Ok(()))
    )
}

#[expect(clippy::too_many_arguments)]
async fn process_raw_sse_bytes(
    raw_bytes: &mut Vec<u8>,
    sentence_buf: &mut String,
    accumulated_full: &mut String,
    tx: &tokio::sync::mpsc::Sender<String>,
    first_chunk_sent: &mut bool,
    token_seen: &mut bool,
    first_chunk_recorded: &mut bool,
    thought_id: ThoughtId,
    stage_timings: &Arc<tokio::sync::RwLock<VecDeque<ThoughtStageTimings>>>,
    is_done: &mut bool,
    deadline: tokio::time::Instant,
) {
    while let Some(newline_pos) = raw_bytes.iter().position(|&b| b == b'\n') {
        let line_bytes = raw_bytes.drain(..=newline_pos).collect::<Vec<u8>>();
        let line_str = String::from_utf8_lossy(&line_bytes);
        let line = line_str.trim();

        if line.is_empty() || line.starts_with(':') {
            continue;
        }

        if let Some(data) = line.strip_prefix("data:") {
            let trimmed = data.trim();
            if trimmed == "[DONE]" {
                *is_done = true;
                break;
            }

            if let Ok(chunk) = serde_json::from_str::<StreamCompletionChunk>(trimmed) {
                for choice in chunk.choices {
                    if let Some(content) = choice.delta.content
                        && !content.is_empty()
                    {
                        if !*token_seen {
                            *token_seen = true;
                            let now = Instant::now();
                            let mut guard = stage_timings.write().await;
                            if let Some(t) = guard.iter_mut().find(|t| t.thought == thought_id)
                                && t.first_token.is_none()
                            {
                                t.first_token = Some(now);
                            }
                        }
                        accumulated_full.push_str(&content);
                        sentence_buf.push_str(&content);
                        let chunks = extract_completed_chunks(sentence_buf, first_chunk_sent);
                        for s in chunks {
                            if !*first_chunk_recorded {
                                *first_chunk_recorded = true;
                                let now = Instant::now();
                                let mut guard = stage_timings.write().await;
                                if let Some(t) = guard.iter_mut().find(|t| t.thought == thought_id)
                                    && t.first_chunk.is_none()
                                {
                                    t.first_chunk = Some(now);
                                }
                            }
                            if !send_sentence_bounded(tx, s, deadline).await {
                                *is_done = true;
                                break;
                            }
                        }
                        if *is_done || accumulated_full.len() > MAX_COMPLETION_BYTES {
                            *is_done = true;
                            break;
                        }
                    }
                }
            }
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "SSE streaming consumer managing timeouts, buffering, and stage timing updates"
)]
#[expect(clippy::too_many_arguments)]
async fn consume_sse_stream(
    mut response: reqwest::Response,
    first_chunk: Vec<u8>,
    tx: tokio::sync::mpsc::Sender<String>,
    cache: Arc<tokio::sync::RwLock<IdempotencyCache>>,
    stage_timings: Arc<tokio::sync::RwLock<VecDeque<ThoughtStageTimings>>>,
    thought_id: ThoughtId,
    _flight_guard: tokio::sync::OwnedMutexGuard<()>,
    total_timeout: Duration,
) {
    let deadline = tokio::time::Instant::now() + total_timeout;
    let mut raw_bytes = Vec::new();
    let mut sentence_buf = String::new();
    let mut accumulated_full = String::new();
    let mut first_chunk_sent = false;
    let mut token_seen = false;
    let mut first_chunk_recorded = false;
    let mut is_done = false;

    raw_bytes.extend_from_slice(&first_chunk);
    process_raw_sse_bytes(
        &mut raw_bytes,
        &mut sentence_buf,
        &mut accumulated_full,
        &tx,
        &mut first_chunk_sent,
        &mut token_seen,
        &mut first_chunk_recorded,
        thought_id,
        &stage_timings,
        &mut is_done,
        deadline,
    )
    .await;

    while !is_done {
        if tx.is_closed() || tokio::time::Instant::now() >= deadline {
            break;
        }

        let chunk = tokio::select! {
            () = tx.closed() => {
                break;
            }
            res = tokio::time::timeout_at(deadline, response.chunk()) => {
                match res {
                    Ok(Ok(Some(bytes))) => bytes,
                    _ => break,
                }
            }
        };

        raw_bytes.extend_from_slice(&chunk);
        if raw_bytes.len() > MAX_RAW_SSE_BYTES {
            break;
        }

        process_raw_sse_bytes(
            &mut raw_bytes,
            &mut sentence_buf,
            &mut accumulated_full,
            &tx,
            &mut first_chunk_sent,
            &mut token_seen,
            &mut first_chunk_recorded,
            thought_id,
            &stage_timings,
            &mut is_done,
            deadline,
        )
        .await;

        if accumulated_full.len() > MAX_COMPLETION_BYTES {
            break;
        }
    }

    // Only if the stream properly completed with [DONE] do we flush and cache (C3)
    if is_done {
        if !raw_bytes.is_empty() {
            let line_str = String::from_utf8_lossy(&raw_bytes);
            let line = line_str.trim();
            if let Some(data) = line.strip_prefix("data:") {
                let trimmed = data.trim();
                if trimmed != "[DONE]"
                    && let Ok(chunk) = serde_json::from_str::<StreamCompletionChunk>(trimmed)
                {
                    for choice in chunk.choices {
                        if let Some(content) = choice.delta.content {
                            if !token_seen {
                                token_seen = true;
                                let now = Instant::now();
                                let mut guard = stage_timings.write().await;
                                if let Some(t) = guard.iter_mut().find(|t| t.thought == thought_id)
                                    && t.first_token.is_none()
                                {
                                    t.first_token = Some(now);
                                }
                            }
                            accumulated_full.push_str(&content);
                            sentence_buf.push_str(&content);
                        }
                    }
                }
            }
        }

        let remaining_chunks = extract_completed_chunks(&mut sentence_buf, &mut first_chunk_sent);
        for s in remaining_chunks {
            if !first_chunk_recorded {
                first_chunk_recorded = true;
                let now = Instant::now();
                let mut guard = stage_timings.write().await;
                if let Some(t) = guard.iter_mut().find(|t| t.thought == thought_id)
                    && t.first_chunk.is_none()
                {
                    t.first_chunk = Some(now);
                }
            }
            if !send_sentence_bounded(&tx, s, deadline).await {
                break;
            }
        }
        let leftover = sentence_buf.trim();
        if !leftover.is_empty() {
            if !first_chunk_recorded {
                let now = Instant::now();
                let mut guard = stage_timings.write().await;
                if let Some(t) = guard.iter_mut().find(|t| t.thought == thought_id)
                    && t.first_chunk.is_none()
                {
                    t.first_chunk = Some(now);
                }
            }
            let _ = send_sentence_bounded(&tx, leftover.to_string(), deadline).await;
        }

        {
            let now = Instant::now();
            let mut guard = stage_timings.write().await;
            if let Some(t) = guard.iter_mut().find(|t| t.thought == thought_id) {
                t.stream_end = Some(now);
            }
        }

        let cleaned = accumulated_full.trim().to_string();
        if !cleaned.is_empty() {
            cache.write().await.insert(thought_id, cleaned);
        }
    }
}

impl Default for OpenAiCortex {
    fn default() -> Self {
        Self::new(CortexConfig::default())
    }
}

impl Cortex for OpenAiCortex {
    async fn think(&self, request: &ThoughtRequest) -> Result<String, PortError> {
        let mut rx = self.think_stream(request).await?;
        let mut full = String::new();
        while let Some(sentence) = rx.recv().await {
            if !full.is_empty() && !full.ends_with(' ') {
                full.push(' ');
            }
            full.push_str(&sentence);
        }

        // Return successfully only if the response was fully completed and cached (C3)
        if let Some(cached) = self.cache.read().await.get(request.thought) {
            return Ok(cached.clone());
        }

        Err(PortError::Failed(
            "cortex stream terminated without successful completion".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use enton_core::Reason;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    #[test]
    fn chat_completion_request_serializes_reasoning_effort_and_omits_think() {
        let req = ChatCompletionRequest {
            model: "test-model",
            messages: vec![OutgoingChatMessage {
                role: "user",
                content: "hello",
            }],
            stream: true,
            reasoning_effort: "none",
        };
        let json = serde_json::to_string(&req).expect("serialization succeeds");
        assert!(json.contains(r#""reasoning_effort":"none""#));
        assert!(!json.contains("think"));
    }

    #[tokio::test]
    async fn first_token_timeout_triggers_on_hanging_headers() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let port = listener.local_addr().expect("local addr").port();

        // Spawn mock server that accepts but does nothing (headers hang)
        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                // Sleep for longer than the client's first_token_timeout
                tokio::time::sleep(Duration::from_millis(500)).await;
                if let Err(_err) = socket.write_all(b"HTTP/1.1 200 OK\r\n\r\n").await {
                    // Non-fatal if the client already timed out and closed the connection.
                }
            }
        });

        let config = CortexConfig {
            base_url: format!("http://127.0.0.1:{port}"),
            model: "test-model".to_string(),
            first_token_timeout: Duration::from_millis(50),
            timeout: Duration::from_secs(1),
            ..CortexConfig::default()
        };
        let cortex = OpenAiCortex::new(config);

        let request = ThoughtRequest {
            thought: ThoughtId(999),
            reason: Reason::Keyword,
            transcript: Some("test".to_string()),
            history: Vec::new(),
        };

        let result = cortex.think_stream(&request).await;
        assert!(
            result.is_err(),
            "think_stream should fail on first token timeout"
        );
        match result.unwrap_err() {
            PortError::Unavailable(msg) => {
                assert!(
                    msg.contains("timed out"),
                    "expected timeout message, got: {msg}"
                );
            }
            other => panic!("expected PortError::Unavailable, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn first_token_timeout_triggers_on_hanging_first_chunk() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let port = listener.local_addr().expect("local addr").port();

        // Spawn mock server that sends headers immediately, but no body chunks
        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                if let Err(_err) = socket
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
                    .await
                {
                    // Non-fatal if connection closed early
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        });

        let config = CortexConfig {
            base_url: format!("http://127.0.0.1:{port}"),
            model: "test-model".to_string(),
            first_token_timeout: Duration::from_millis(50),
            timeout: Duration::from_secs(1),
            ..CortexConfig::default()
        };
        let cortex = OpenAiCortex::new(config);

        let request = ThoughtRequest {
            thought: ThoughtId(1001),
            reason: Reason::Keyword,
            transcript: Some("test".to_string()),
            history: Vec::new(),
        };

        let result = cortex.think_stream(&request).await;
        assert!(
            result.is_err(),
            "think_stream should fail when first chunk times out"
        );
        match result.unwrap_err() {
            PortError::Unavailable(msg) => {
                assert!(
                    msg.contains("stream timed out") || msg.contains("timed out"),
                    "expected timeout message, got: {msg}"
                );
            }
            other => panic!("expected PortError::Unavailable, got {other:?}"),
        }
    }

    #[test]
    fn prune_history_keeps_head_and_tail_when_middle_dropped() {
        let turns: Vec<ConversationTurn> = (0..5)
            .map(|i| ConversationTurn::user(format!("Turn {i}: test turn content")))
            .collect();

        // Budget that fits exactly the 2 boundary turns, dropping the 3 middle turns:
        let pruned = prune_history_middle_out(&turns, 30);
        assert_eq!(pruned.len(), 2);
        assert_eq!(pruned[0].content, turns[0].content);
        assert_eq!(pruned[1].content, turns[4].content);
    }

    #[test]
    fn prune_history_within_budget_retains_all() {
        let turns = vec![
            ConversationTurn::user("hi"),
            ConversationTurn::assistant("hello"),
        ];
        let pruned = prune_history_middle_out(&turns, 1000);
        assert_eq!(pruned.len(), 2);
        assert_eq!(pruned[0].content, "hi");
        assert_eq!(pruned[1].content, "hello");
    }

    #[tokio::test]
    async fn idempotency_cache_prevents_duplicate_calls() {
        let cortex = OpenAiCortex::with_endpoint("http://127.0.0.1:9", "dummy-model");
        let id = ThoughtId(42);

        // Pre-populate cache directly
        {
            let mut cache = cortex.cache.write().await;
            cache.insert(id, "cached response".to_string());
        }

        let request = ThoughtRequest {
            thought: id,
            reason: Reason::Keyword,
            transcript: Some("enton".to_string()),
            history: Vec::new(),
        };

        // Should return the cached value without trying to connect to port 9
        let result = cortex
            .think(&request)
            .await
            .expect("cached read should succeed");
        assert_eq!(result, "cached response");
    }

    #[tokio::test]
    async fn idempotency_cache_bounds_capacity_and_preserves_most_recent() {
        let capacity = 16;
        let config = CortexConfig {
            base_url: "http://127.0.0.1:9".to_string(),
            model: "dummy-model".to_string(),
            idempotency_cache_capacity: capacity,
            ..CortexConfig::default()
        };
        let cortex = OpenAiCortex::new(config);

        assert!(cortex.cache.read().await.is_empty());

        let total = capacity + 10;
        {
            let mut cache = cortex.cache.write().await;
            for i in 1..=total {
                let id = ThoughtId(u64::try_from(i).expect("valid id"));
                cache.insert(id, format!("response {i}"));
            }
            drop(cache);
        }

        // Verify that the cache size is capped at exactly capacity (N)
        assert_eq!(cortex.cache_len().await, capacity);

        // Verify that the oldest 10 entries (1..=10) were evicted
        {
            let cache = cortex.cache.read().await;
            for i in 1..=10 {
                let id = ThoughtId(u64::try_from(i).expect("valid id"));
                assert!(
                    cache.get(id).is_none(),
                    "ThoughtId({i}) should have been evicted"
                );
            }
            // Verify that the latest entries (11..=total) remain cached
            for i in 11..=total {
                let id = ThoughtId(u64::try_from(i).expect("valid id"));
                assert!(
                    cache.get(id).is_some(),
                    "ThoughtId({i}) should remain in cache"
                );
            }
            drop(cache);
        }

        // Verify that the most recent thought continues to be deduplicated without network call
        let recent_request = ThoughtRequest {
            thought: ThoughtId(u64::try_from(total).expect("valid id")),
            reason: Reason::Keyword,
            transcript: Some("test".to_string()),
            history: Vec::new(),
        };
        let response = cortex
            .think(&recent_request)
            .await
            .expect("most recent thought should be deduplicated from cache");
        assert_eq!(response, format!("response {total}"));

        // Verify that an evicted thought is not in cache and fails when attempting network call
        let evicted_request = ThoughtRequest {
            thought: ThoughtId(1),
            reason: Reason::Keyword,
            transcript: Some("test".to_string()),
            history: Vec::new(),
        };
        assert!(
            cortex.think(&evicted_request).await.is_err(),
            "evicted thought should attempt network and fail on dummy port"
        );
    }

    #[test]
    fn test_extract_completed_sentences_various_cases() {
        let mut buffer = "E aí, Gabriel! Beleza? Tô de boa por aqui.".to_string();
        let sentences = extract_completed_sentences(&mut buffer);
        assert_eq!(
            sentences,
            vec!["E aí, Gabriel!", "Beleza?", "Tô de boa por aqui."]
        );
        assert!(buffer.is_empty());

        let mut buffer = "Testando... mais um teste.".to_string();
        let sentences = extract_completed_sentences(&mut buffer);
        assert_eq!(sentences, vec!["Testando...", "mais um teste."]);
        assert!(buffer.is_empty());

        let mut buffer = "Frase com \"aspas!\" Mais texto depois.".to_string();
        let sentences = extract_completed_sentences(&mut buffer);
        assert_eq!(
            sentences,
            vec!["Frase com \"aspas!\"", "Mais texto depois."]
        );
        assert!(buffer.is_empty());

        // Incomplete sentence should stay in buffer
        let mut buffer = "Olá, Gabriel! Como vai vo".to_string();
        let sentences = extract_completed_sentences(&mut buffer);
        assert_eq!(sentences, vec!["Olá, Gabriel!"]);
        assert_eq!(buffer, "Como vai vo");

        // Multi-line sentences
        let mut buffer = "Linha um.\nLinha dois!\n".to_string();
        let sentences = extract_completed_sentences(&mut buffer);
        assert_eq!(sentences, vec!["Linha um.", "Linha dois!"]);
        assert!(buffer.is_empty());
    }

    #[tokio::test]
    async fn think_stream_cached_thought_yields_sentences() {
        let cortex = OpenAiCortex::with_endpoint("http://127.0.0.1:9", "dummy-model");
        let id = ThoughtId(100);

        {
            let mut cache = cortex.cache.write().await;
            cache.insert(
                id,
                "E aí Gabriel! Tô aqui no terminal. Manda ver.".to_string(),
            );
        }

        let request = ThoughtRequest {
            thought: id,
            reason: Reason::Speech,
            transcript: Some("enton oi".to_string()),
            history: Vec::new(),
        };

        let mut rx = cortex
            .think_stream(&request)
            .await
            .expect("stream from cache should succeed");

        let mut received = Vec::new();
        while let Some(sentence) = rx.recv().await {
            received.push(sentence);
        }

        assert_eq!(
            received,
            vec!["E aí Gabriel!", "Tô aqui no terminal.", "Manda ver."]
        );
    }

    #[tokio::test]
    async fn c1_cached_response_with_more_than_16_sentences_does_not_deadlock() {
        let cortex = OpenAiCortex::with_endpoint("http://127.0.0.1:9", "dummy-model");
        let id_17 = ThoughtId(201);
        let id_16_frag = ThoughtId(202);

        // Case 1: Exactly 17 completed sentences (previously deadlocked a 16-slot channel)
        let seventeen_sentences = (1..=17)
            .map(|i| format!("Frase número {i}."))
            .collect::<Vec<_>>()
            .join(" ");

        // Case 2: 16 completed sentences plus one trailing fragment
        let sixteen_plus_frag = (1..=16)
            .map(|i| format!("Sentença {i}."))
            .collect::<Vec<_>>()
            .join(" ")
            + " Fragmento final";

        {
            let mut cache = cortex.cache.write().await;
            cache.insert(id_17, seventeen_sentences);
            cache.insert(id_16_frag, sixteen_plus_frag);
        }

        // Test 17 sentences with timeout protection
        let req17 = ThoughtRequest {
            thought: id_17,
            reason: Reason::Speech,
            transcript: Some("test 17".to_string()),
            history: Vec::new(),
        };
        let mut rx = tokio::time::timeout(Duration::from_secs(2), cortex.think_stream(&req17))
            .await
            .expect("think_stream must not hang")
            .expect("should return rx");

        let mut received = Vec::new();
        while let Ok(Some(s)) = tokio::time::timeout(Duration::from_secs(1), rx.recv()).await {
            received.push(s);
        }
        assert_eq!(
            received.len(),
            17,
            "must receive all 17 sentences without channel deadlock"
        );

        // Test 16 sentences + fragment with timeout protection
        let req16 = ThoughtRequest {
            thought: id_16_frag,
            reason: Reason::Speech,
            transcript: Some("test 16 frag".to_string()),
            history: Vec::new(),
        };
        let mut rx = tokio::time::timeout(Duration::from_secs(2), cortex.think_stream(&req16))
            .await
            .expect("think_stream must not hang")
            .expect("should return rx");

        let mut received = Vec::new();
        while let Ok(Some(s)) = tokio::time::timeout(Duration::from_secs(1), rx.recv()).await {
            received.push(s);
        }
        assert_eq!(
            received.len(),
            17,
            "must receive 16 sentences + fragment without channel deadlock"
        );
        assert_eq!(received[16], "Fragmento final");
    }

    #[tokio::test]
    async fn c2_undrained_receiver_and_dropped_receiver_release_inference_slot() {
        // Scenario A: Dropped receiver during streaming exits immediately and releases flight guard
        let listener_a = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let port_a = listener_a.local_addr().expect("port").port();

        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener_a.accept().await {
                drop(
                    socket
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
                        .await,
                );
                let chunk =
                    b"data: {\"choices\":[{\"delta\":{\"content\":\"Primeira frase. \"}}]}\n\n";
                drop(socket.write_all(chunk).await);
                // Keep connection open without writing more
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        });

        let config_a = CortexConfig {
            base_url: format!("http://127.0.0.1:{port_a}"),
            first_token_timeout: Duration::from_millis(500),
            timeout: Duration::from_secs(5),
            ..CortexConfig::default()
        };
        let cortex_a = OpenAiCortex::new(config_a);

        let req_a = ThoughtRequest {
            thought: ThoughtId(301),
            reason: Reason::Speech,
            transcript: Some("drop test".to_string()),
            history: Vec::new(),
        };
        let rx = cortex_a.think_stream(&req_a).await.expect("should connect");

        // Immediately drop receiver (barge-in / cancellation)
        drop(rx);

        // Wait a brief moment for tokio::select! on tx.closed() to wake consume_sse_stream
        tokio::time::sleep(Duration::from_millis(60)).await;

        // The flight lock must be free now even though the server body is still open
        assert!(
            cortex_a.in_flight.try_lock().is_ok(),
            "flight lock must be released immediately when receiver is dropped"
        );

        // Scenario B: Undrained receiver times out and releases flight guard at deadline
        let listener_b = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let port_b = listener_b.local_addr().expect("port").port();

        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener_b.accept().await {
                drop(
                    socket
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
                        .await,
                );
                // Write many sentences to overflow the 16-slot channel
                for i in 1..=20 {
                    let chunk = format!(
                        "data: {{\"choices\":[{{\"delta\":{{\"content\":\"Frase número {i}. \"}}}}]}}\n\n"
                    );
                    drop(socket.write_all(chunk.as_bytes()).await);
                }
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        });

        let config_b = CortexConfig {
            base_url: format!("http://127.0.0.1:{port_b}"),
            first_token_timeout: Duration::from_millis(500),
            timeout: Duration::from_millis(150), // Short overall deadline
            ..CortexConfig::default()
        };
        let cortex_b = OpenAiCortex::new(config_b);

        let req_b = ThoughtRequest {
            thought: ThoughtId(302),
            reason: Reason::Speech,
            transcript: Some("backpressure test".to_string()),
            history: Vec::new(),
        };
        let _retained_rx = cortex_b.think_stream(&req_b).await.expect("should connect");

        // Retain _retained_rx without reading. After timeout (150ms), consumer must exit and release lock.
        tokio::time::sleep(Duration::from_millis(250)).await;

        assert!(
            cortex_b.in_flight.try_lock().is_ok(),
            "flight lock must be released after timeout even if receiver was retained and undrained"
        );
    }

    #[tokio::test]
    async fn c3_failed_or_cancelled_stream_is_not_cached() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let port = listener.local_addr().expect("port").port();

        // Server sends one sentence, then abruptly closes connection without data: [DONE]
        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                drop(
                    socket
                        .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
                        .await,
                );
                let chunk = b"data: {\"choices\":[{\"delta\":{\"content\":\"Primeira frase antes da queda. \"}}]}\n\n";
                drop(socket.write_all(chunk).await);
                // Abrupt close
                drop(socket);
            }
        });

        let config = CortexConfig {
            base_url: format!("http://127.0.0.1:{port}"),
            first_token_timeout: Duration::from_millis(500),
            timeout: Duration::from_secs(1),
            ..CortexConfig::default()
        };
        let cortex = OpenAiCortex::new(config);

        let request = ThoughtRequest {
            thought: ThoughtId(401),
            reason: Reason::Speech,
            transcript: Some("c3 test".to_string()),
            history: Vec::new(),
        };

        // think() should fail because the stream was truncated before [DONE]
        let result = cortex.think(&request).await;
        assert!(
            result.is_err(),
            "think() must return error on incomplete stream without [DONE]"
        );

        // Crucially, the partial response must NOT be stored in the idempotency cache!
        assert_eq!(
            cortex.cache_len().await,
            0,
            "incomplete/failed stream must not be inserted into cache"
        );
    }

    #[tokio::test]
    async fn c4_idempotency_cache_evicts_by_byte_cap() {
        let config = CortexConfig {
            idempotency_cache_capacity: 10,
            max_cache_bytes: 100, // Small 100-byte cap
            ..CortexConfig::default()
        };
        let cortex = OpenAiCortex::new(config);

        let t1 = ThoughtId(501);
        let t2 = ThoughtId(502);

        // String 1 is 60 bytes
        let reply_1 = "A".repeat(60);
        // String 2 is 60 bytes
        let reply_2 = "B".repeat(60);

        let (len_after_insert, bytes_after_insert, has_t1, has_t2) = {
            let mut cache = cortex.cache.write().await;
            cache.insert(t1, reply_1);
            assert_eq!(cache.total_bytes(), 60);
            assert_eq!(cache.len(), 1);

            // Inserting reply_2 brings total to 120 > 100, so t1 must be evicted
            cache.insert(t2, reply_2);
            (
                cache.len(),
                cache.total_bytes(),
                cache.get(t1).is_some(),
                cache.get(t2).is_some(),
            )
        };

        assert_eq!(len_after_insert, 1, "t1 should be evicted by byte limit");
        assert_eq!(bytes_after_insert, 60);
        assert!(!has_t1, "t1 must be evicted");
        assert!(has_t2, "t2 must remain in cache");

        assert_eq!(cortex.cache_bytes().await, 60);
    }
    #[tokio::test]
    async fn cortex_stage_timings_recorded_on_thought_stream() {
        let cortex = OpenAiCortex::with_endpoint("http://127.0.0.1:9", "dummy-model");
        let id = ThoughtId(888);
        let cached_reply = "Com certeza Gabriel, estou pronto para te ajudar! Vamos nessa.";

        {
            let mut cache = cortex.cache.write().await;
            cache.insert(id, cached_reply.to_string());
        }

        let request = ThoughtRequest {
            thought: id,
            reason: Reason::Speech,
            transcript: Some("test timings".to_string()),
            history: Vec::new(),
        };

        let mut rx = cortex
            .think_stream(&request)
            .await
            .expect("stream from cache should succeed");

        let mut received = Vec::new();
        while let Some(chunk) = rx.recv().await {
            received.push(chunk);
        }

        assert_eq!(
            received,
            vec![
                "Com certeza Gabriel,",
                "estou pronto para te ajudar!",
                "Vamos nessa."
            ]
        );

        let timings = cortex
            .stage_timings(id)
            .await
            .expect("stage timings must be recorded for thought");

        assert_eq!(timings.thought, id);
        let tok = timings.first_token.expect("first_token recorded");
        let chunk = timings.first_chunk.expect("first_chunk recorded");
        let end = timings.stream_end.expect("stream_end recorded");

        assert!(tok >= timings.request_start);
        assert!(chunk >= tok);
        assert!(end >= chunk);
    }

    #[test]
    fn property_chunks_concatenate_to_original_text_without_empty_chunks() {
        let samples = [
            "",
            "   ",
            "Olá!",
            "Oi, tudo bem?",
            "Sim. Vamos lá.",
            "Com certeza Gabriel, eu vou te ajudar hoje! Vamos começar.",
            "O processador opera a 3,5 GHz, o que garante 3.14 vezes mais velocidade.",
            "O Dr. Gabriel ligou; ele volta amanhã com certeza. Até mais.",
            "Pensando bem... tudo vai dar certo no final! Com certeza.",
            "Esta é a primeira cláusula longa — e esta continua depois do travessão. Fim.",
            "Aqui está o relatório: todos os dados foram conferidos com calma.",
            "\"Com certeza Gabriel, podemos fazer isso!\", disse ela com alegria.",
            "Frase um! Frase dois? Frase três. Frase quatro...\nFrase cinco.",
            "Texto longo sem pontuação terminal mas que deve ser retornado intacto no leftover",
        ];

        for text in samples {
            let chunks = extract_reply_chunks(text);

            // Property 1: No chunk is empty
            for chunk in &chunks {
                assert!(
                    !chunk.trim().is_empty(),
                    "chunk must not be empty for input: {text:?}"
                );
            }

            // Property 2: Modulo trimming and spacing, concatenating reproduces the original text
            if text.trim().is_empty() {
                assert!(chunks.is_empty());
            } else {
                let joined = chunks.join(" ");
                let original_words: Vec<&str> = text.split_whitespace().collect();
                let joined_words: Vec<&str> = joined.split_whitespace().collect();
                assert_eq!(
                    joined_words, original_words,
                    "reproduced words must match original for input: {text:?}"
                );
            }
        }
    }

    #[test]
    fn first_clause_splits_at_comma_when_length_reaches_threshold() {
        let text =
            "Com certeza meu caro amigo Gabriel, estou pronto para te ajudar! Vamos em frente.";
        let chunks = extract_reply_chunks(text);

        assert_eq!(
            chunks,
            vec![
                "Com certeza meu caro amigo Gabriel,",
                "estou pronto para te ajudar!",
                "Vamos em frente."
            ]
        );
        assert!(chunks[0].chars().count() >= MIN_FIRST_CLAUSE_CHARS);
    }

    #[test]
    fn first_clause_does_not_split_short_comma_fragment() {
        let text = "Oi, tudo bem com você? Vamos começar agora.";
        let chunks = extract_reply_chunks(text);

        // "Oi," is only 3 chars < MIN_FIRST_CLAUSE_CHARS, so it stays with the first sentence.
        assert_eq!(
            chunks,
            vec!["Oi, tudo bem com você?", "Vamos começar agora."]
        );
    }

    #[test]
    fn later_chunks_maintain_sentence_granularity_even_with_commas() {
        let text =
            "Com certeza Gabriel, estou pronto! Na próxima etapa, faremos o teste, com calma.";
        let chunks = extract_reply_chunks(text);

        // First chunk splits at clause boundary.
        // Later chunks keep sentence granularity despite containing commas.
        assert_eq!(
            chunks,
            vec![
                "Com certeza Gabriel,",
                "estou pronto!",
                "Na próxima etapa, faremos o teste, com calma."
            ]
        );
    }

    #[test]
    fn boundaries_respect_decimals_in_portuguese_and_english() {
        // Portuguese decimal "3,5" uses a comma
        let pt_text = "O sistema consumiu 3,5 watts de energia, o que é excelente.";
        let pt_chunks = extract_reply_chunks(pt_text);
        assert_eq!(
            pt_chunks,
            vec![
                "O sistema consumiu 3,5 watts de energia,",
                "o que é excelente."
            ]
        );

        // English decimal "2.5" uses a dot
        let en_text = "A versão 2.5 foi liberada agora, aproveite para atualizar.";
        let en_chunks = extract_reply_chunks(en_text);
        assert_eq!(
            en_chunks,
            vec![
                "A versão 2.5 foi liberada agora,",
                "aproveite para atualizar."
            ]
        );
    }

    #[test]
    fn boundaries_respect_abbreviations_and_ellipses() {
        let text = "O Dr. Gabriel chegou agora há pouco; vamos conversar com ele.";
        let chunks = extract_reply_chunks(text);
        assert_eq!(
            chunks,
            vec![
                "O Dr. Gabriel chegou agora há pouco;",
                "vamos conversar com ele."
            ]
        );

        let ellipsis_text = "Esperando um pouco... talvez seja melhor agora. Tudo certo.";
        let ellipsis_chunks = extract_reply_chunks(ellipsis_text);
        assert_eq!(
            ellipsis_chunks,
            vec![
                "Esperando um pouco...",
                "talvez seja melhor agora.",
                "Tudo certo."
            ]
        );
    }

    #[test]
    fn boundaries_support_em_dash_colon_and_semicolon() {
        let em_dash = "Com certeza meu caro amigo — vamos resolver essa pendência hoje.";
        let chunks = extract_reply_chunks(em_dash);
        assert_eq!(
            chunks,
            vec![
                "Com certeza meu caro amigo —",
                "vamos resolver essa pendência hoje."
            ]
        );

        let colon = "Aqui está a solução completa: todos os testes passaram.";
        let colon_chunks = extract_reply_chunks(colon);
        assert_eq!(
            colon_chunks,
            vec!["Aqui está a solução completa:", "todos os testes passaram."]
        );
    }
}
