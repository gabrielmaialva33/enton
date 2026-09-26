use std::collections::{HashMap, VecDeque};

use enton_core::ThoughtId;

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

#[cfg(test)]
mod tests {
    use super::super::client::OpenAiCortex;
    use super::super::config::CortexConfig;
    use super::*;
    use enton_core::Reason;
    use enton_core::ports::{Cortex, ThoughtRequest};

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
}
