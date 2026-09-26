//! The soul on its own thread, so durable writes never block the event loop.
//!
//! The event loop is a single-threaded tokio runtime; a `synchronous=FULL`
//! SQLite write there would stall timers and cortex streaming. The worker owns
//! the [`Soul`] and answers each request through a oneshot channel, so the loop
//! can await durability before acting (write-ahead: record, then reduce, then
//! act).

use std::path::Path;
use std::thread::JoinHandle;

use enton_adapters::{SeqNo, Soul, SoulConfig, soul};
use enton_core::{Event, Organism, Profile, ThoughtId};
use tokio::sync::{mpsc, oneshot};

/// Bound on requests waiting for the worker.
const QUEUE: usize = 64;

/// Retention: about 14 hours of one-second ticks, two snapshots, 64 MiB.
fn config() -> SoulConfig {
    SoulConfig {
        max_events: Some(50_000),
        max_db_bytes: Some(64 * 1024 * 1024),
        max_snapshots: Some(2),
        ..SoulConfig::default()
    }
}

/// A journal failure: the soul itself failed, or its worker is gone.
#[derive(Debug, thiserror::Error)]
pub(crate) enum JournalError {
    /// The durable log rejected the operation.
    #[error("soul: {0}")]
    Soul(#[from] soul::Error),
    /// The worker thread stopped before answering.
    #[error("soul worker stopped")]
    WorkerGone,
}

/// What [`Journal::open`] restored from disk.
#[derive(Debug)]
pub(crate) struct Restored {
    /// The organism as of the last recorded event.
    pub(crate) organism: Organism,
    /// Thoughts a crash left pending; they are marked failed, never re-run.
    pub(crate) abandoned: Vec<ThoughtId>,
}

enum Request {
    Append {
        event: Event,
        reply: oneshot::Sender<Result<SeqNo, soul::Error>>,
    },
    Pending {
        thought: ThoughtId,
        at_seq: SeqNo,
        reply: oneshot::Sender<Result<(), soul::Error>>,
    },
    Resolve {
        thought: ThoughtId,
        done: bool,
        result_json: String,
        reply: oneshot::Sender<Result<(), soul::Error>>,
    },
    Snapshot {
        at_seq: SeqNo,
        organism: Box<Organism>,
        reply: oneshot::Sender<Result<(), soul::Error>>,
    },
}

/// Async handle to the soul worker.
#[derive(Debug)]
pub(crate) struct Journal {
    requests: Option<mpsc::Sender<Request>>,
    worker: Option<JoinHandle<()>>,
}

impl Journal {
    /// Open (or create) the soul at `path`, abandon thoughts a crash left
    /// pending, restore the organism for `profile` and start the worker.
    ///
    /// Blocking: call it from `spawn_blocking` or before the event loop runs.
    pub(crate) fn open(path: &Path, profile: &Profile) -> Result<(Self, Restored), JournalError> {
        if let Some(directory) = path.parent() {
            std::fs::create_dir_all(directory).map_err(soul::Error::from)?;
        }
        let soul = Soul::open(path, config())?;
        let abandoned: Vec<ThoughtId> = soul
            .pending_actions()?
            .into_iter()
            .map(|(thought, _)| thought)
            .collect();
        for thought in &abandoned {
            soul.mark_failed(*thought, r#"{"reason":"abandoned at restart"}"#)?;
        }
        // Replayed actions describe the past; executing them again would repeat effects.
        let (organism, _replayed) = soul.replay_organism(profile)?;

        let (requests, inbox) = mpsc::channel(QUEUE);
        let worker = std::thread::Builder::new()
            .name("enton-soul".to_owned())
            .spawn(move || serve(&soul, inbox))
            .map_err(soul::Error::from)?;
        let journal = Self {
            requests: Some(requests),
            worker: Some(worker),
        };
        Ok((
            journal,
            Restored {
                organism,
                abandoned,
            },
        ))
    }

    /// Durably append `event` and return its sequence number.
    pub(crate) async fn append(&self, event: &Event) -> Result<SeqNo, JournalError> {
        let event = event.clone();
        self.call(|reply| Request::Append { event, reply }).await
    }

    /// Record that `thought`, decided at `at_seq`, is about to reach the cortex.
    pub(crate) async fn pending(
        &self,
        thought: ThoughtId,
        at_seq: SeqNo,
    ) -> Result<(), JournalError> {
        self.call(|reply| Request::Pending {
            thought,
            at_seq,
            reply,
        })
        .await
    }

    /// Resolve a pending thought as done or failed, with a small JSON result.
    pub(crate) async fn resolve(
        &self,
        thought: ThoughtId,
        done: bool,
        result_json: String,
    ) -> Result<(), JournalError> {
        self.call(|reply| Request::Resolve {
            thought,
            done,
            result_json,
            reply,
        })
        .await
    }

    /// Persist `organism` as the state after reducing event `at_seq`, then
    /// enforce retention.
    pub(crate) async fn snapshot(
        &self,
        at_seq: SeqNo,
        organism: &Organism,
    ) -> Result<(), JournalError> {
        let organism = Box::new(organism.clone());
        self.call(|reply| Request::Snapshot {
            at_seq,
            organism,
            reply,
        })
        .await
    }

    /// Stop accepting requests and wait for the worker to finish the queue.
    pub(crate) async fn close(mut self) -> Result<(), JournalError> {
        self.requests = None;
        let Some(worker) = self.worker.take() else {
            return Ok(());
        };
        match tokio::task::spawn_blocking(move || worker.join()).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_)) | Err(_) => Err(JournalError::WorkerGone),
        }
    }

    async fn call<T>(
        &self,
        request: impl FnOnce(oneshot::Sender<Result<T, soul::Error>>) -> Request,
    ) -> Result<T, JournalError> {
        let requests = self.requests.as_ref().ok_or(JournalError::WorkerGone)?;
        let (reply, answer) = oneshot::channel();
        requests
            .send(request(reply))
            .await
            .map_err(|_| JournalError::WorkerGone)?;
        Ok(answer.await.map_err(|_| JournalError::WorkerGone)??)
    }
}

fn serve(soul: &Soul, mut inbox: mpsc::Receiver<Request>) {
    while let Some(request) = inbox.blocking_recv() {
        // A dropped reply means the caller stopped waiting; the write still happened.
        let _delivered = match request {
            Request::Append { event, reply } => reply.send(soul.append_event(&event)).is_ok(),
            Request::Pending {
                thought,
                at_seq,
                reply,
            } => reply.send(soul.record_pending(thought, at_seq)).is_ok(),
            Request::Resolve {
                thought,
                done,
                result_json,
                reply,
            } => {
                let result = if done {
                    soul.mark_done(thought, &result_json)
                } else {
                    soul.mark_failed(thought, &result_json)
                };
                reply.send(result).is_ok()
            }
            Request::Snapshot {
                at_seq,
                organism,
                reply,
            } => {
                let result = soul
                    .save_organism_snapshot(at_seq, &organism)
                    .and_then(|()| soul.retain().map(|_| ()));
                reply.send(result).is_ok()
            }
        };
    }
}
