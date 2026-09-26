use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use enton_core::ThoughtId;
use enton_core::ports::{Cortex, PortError, ThoughtRequest};

use super::cache::IdempotencyCache;
use super::chunking::extract_completed_chunks;
use super::config::CortexConfig;
use super::prompt::{assemble_messages, prune_history_middle_out};
use super::wire::{ChatCompletionRequest, StreamCompletionChunk};

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
    pub(crate) in_flight: Arc<tokio::sync::Mutex<()>>,
    pub(crate) cache: Arc<tokio::sync::RwLock<IdempotencyCache>>,
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
}
