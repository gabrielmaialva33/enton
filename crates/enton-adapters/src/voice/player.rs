use std::collections::VecDeque;
use std::fmt;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use enton_core::UtteranceId;
use enton_core::ports::{PortError, TextToSpeech};

use super::chime::chime;
use super::config::{VoiceConfig, VoiceError};
use super::events::{
    EventHub, PlaybackEvent, PlaybackEventStats, UtteranceStageTimings, emit_event,
};
use super::playback::{
    PlaybackQueueState, QueuedSentence, open_output_device, spawn_simulated_device,
    start_output_stream,
};
use super::worker::{
    PortJob, SynthesisJob, SynthesisWorkerParams, SynthesizerBackend, create_offline_tts,
    native_sample_rate, spawn_port_worker, spawn_synthesis_worker, synthesis_response,
};

/// Voice synthesizer and audio player.
///
/// Delivers sentence-by-sentence speech synthesis with bounded buffering,
/// accurate lifecycle event reporting, and instantaneous barge-in cancellation.
pub struct VoicePlayer {
    config: VoiceConfig,
    next_id: AtomicU64,
    pub(super) generation: Arc<AtomicU64>,
    pub(super) subscribers: Arc<EventHub>,
    pub(super) queue_state: Arc<Mutex<PlaybackQueueState>>,
    pub(super) port_tx: SyncSender<PortJob>,
    pub(super) last_error: Arc<Mutex<Option<VoiceError>>>,
    pub(super) job_tx: SyncSender<SynthesisJob>,
    _worker_handle: Option<JoinHandle<()>>,
    _stream: Option<cpal::Stream>,
    pub(super) stopped: Arc<AtomicBool>,
    native_sample_rate: u32,
    device_sample_rate: u32,
    /// The acknowledgement chime at the output rate, computed once.
    chime: Vec<f32>,
}

impl fmt::Debug for VoicePlayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VoicePlayer")
            .field("config", &self.config)
            .field("next_id", &self.next_id.load(Ordering::Relaxed))
            .field("generation", &self.generation.load(Ordering::Relaxed))
            .field("native_sample_rate", &self.native_sample_rate)
            .field("device_sample_rate", &self.device_sample_rate)
            .finish_non_exhaustive()
    }
}

impl VoicePlayer {
    /// Creates and starts a new voice player with the supplied configuration and default audio device.
    ///
    /// The output stream runs at the synthesizer's native rate when the device
    /// takes it; otherwise at the device's default rate, with each sentence
    /// resampled before playback (see [`VoicePlayer::is_resampling`]).
    ///
    /// # Errors
    /// Returns [`VoiceError`] if model files are missing, invalid, or no CPAL output device is found.
    pub fn new(config: VoiceConfig) -> Result<Self, VoiceError> {
        config.validate()?;
        let tts = create_offline_tts(&config)?;
        let native_rate = native_sample_rate(&tts)?;
        let output = open_output_device(config.device_id.as_deref(), native_rate)?;

        let subscribers = Arc::new(EventHub::new());
        let last_error = Arc::new(Mutex::new(None));
        let generation = Arc::new(AtomicU64::new(0));
        let queue_state = Arc::new(Mutex::new(PlaybackQueueState::new(config.queue_capacity)));

        let start = |stream_config| {
            start_output_stream(
                &output.device,
                stream_config,
                &queue_state,
                &subscribers,
                &last_error,
            )
        };
        let (stream, sample_rate) = match start(&output.preferred) {
            Ok(stream) => (stream, output.preferred.sample_rate()),
            // A device can list a native-rate config and then refuse it; its
            // default config still plays, through the resampler.
            Err(_) if output.preferred != output.default => {
                (start(&output.default)?, output.default.sample_rate())
            }
            Err(error) => return Err(error),
        };

        let synthesizer = Arc::new(SynthesizerBackend::Offline(Mutex::new(tts)));
        let (job_tx, job_rx) = sync_channel::<SynthesisJob>(config.queue_capacity);
        let stopped = Arc::new(AtomicBool::new(false));

        let worker_params = SynthesisWorkerParams {
            job_rx,
            synthesizer: Arc::clone(&synthesizer),
            last_error: Arc::clone(&last_error),
            queue_state: Arc::clone(&queue_state),
            generation: Arc::clone(&generation),
            subscribers: Arc::clone(&subscribers),
            stopped: Arc::clone(&stopped),
            device_sample_rate: sample_rate,
            speed: config.speed,
            speaker_id: config.engine.speaker_id(),
            queue_capacity: config.queue_capacity,
        };

        let worker_handle = spawn_synthesis_worker(worker_params);
        let port_tx = spawn_port_worker(
            synthesizer,
            Arc::clone(&stopped),
            config.queue_capacity,
            config.speed,
            config.engine.speaker_id(),
        );

        Ok(Self {
            config,
            next_id: AtomicU64::new(1),
            generation,
            subscribers,
            queue_state,
            port_tx,
            last_error,
            job_tx,
            _worker_handle: Some(worker_handle),
            _stream: Some(stream),
            stopped,
            native_sample_rate: native_rate,
            device_sample_rate: sample_rate,
            chime: chime(sample_rate),
        })
    }

    /// Creates a mock voice player that runs without requiring model files or physical audio devices.
    ///
    /// It is the real player with two stand-ins: every sentence synthesizes to 20 ms
    /// of a 440 Hz tone, and a simulated device plays the queue at 24 kHz in about real
    /// time, through the same callback and lifecycle events as a sound card.
    /// Ideal for headless testing and CI environments.
    #[must_use]
    pub fn mock() -> Self {
        const RATE: u32 = 24_000;
        let config = VoiceConfig::default();
        let capacity = config.queue_capacity;
        let subscribers = Arc::new(EventHub::new());
        let last_error = Arc::new(Mutex::new(None));
        let generation = Arc::new(AtomicU64::new(0));
        let queue_state = Arc::new(Mutex::new(PlaybackQueueState::new(capacity)));
        let synthesizer = Arc::new(SynthesizerBackend::Mock { sample_rate: RATE });
        let (job_tx, job_rx) = sync_channel::<SynthesisJob>(capacity);
        let stopped = Arc::new(AtomicBool::new(false));

        let worker_handle = spawn_synthesis_worker(SynthesisWorkerParams {
            job_rx,
            synthesizer: Arc::clone(&synthesizer),
            last_error: Arc::clone(&last_error),
            queue_state: Arc::clone(&queue_state),
            generation: Arc::clone(&generation),
            subscribers: Arc::clone(&subscribers),
            stopped: Arc::clone(&stopped),
            device_sample_rate: RATE,
            speed: 1.0,
            speaker_id: 0,
            queue_capacity: capacity,
        });
        spawn_simulated_device(
            Arc::clone(&queue_state),
            Arc::clone(&subscribers),
            Arc::clone(&stopped),
            RATE,
        );
        let port_tx = spawn_port_worker(synthesizer, Arc::clone(&stopped), capacity, 1.0, 42);
        Self {
            config,
            next_id: AtomicU64::new(1),
            generation,
            subscribers,
            queue_state,
            port_tx,
            last_error,
            job_tx,
            _worker_handle: Some(worker_handle),
            _stream: None,
            stopped,
            native_sample_rate: RATE,
            device_sample_rate: RATE,
            chime: chime(RATE),
        }
    }

    /// Subscribe to playback lifecycle events with a 64-event queue.
    /// At most 16 subscribers are admitted. The dispatch mailbox also holds 64
    /// events. Both queues drop new events on overflow, counted in `event_stats`;
    /// delivery never waits for a slow subscriber. Disconnections are counted.
    /// Returns [`VoiceError::SubscriberLimit`] when all subscriber slots are used.
    pub fn subscribe(&self) -> Result<Receiver<PlaybackEvent>, VoiceError> {
        self.subscribers.subscribe()
    }

    /// Snapshot cumulative event loss. Any nonzero loss invalidates a complete audit.
    #[must_use]
    pub fn event_stats(&self) -> PlaybackEventStats {
        self.subscribers.stats()
    }

    /// Returns recorded stage timings for a given utterance ID, if available.
    #[must_use]
    pub fn stage_timings(&self, id: UtteranceId) -> Option<UtteranceStageTimings> {
        let state = self
            .queue_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.timings.iter().find(|t| t.id == id).copied()
    }

    /// Returns recorded stage timings for the most recently submitted utterance.
    #[must_use]
    pub fn last_stage_timings(&self) -> Option<UtteranceStageTimings> {
        let state = self
            .queue_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.timings.back().copied()
    }

    /// Queues a sentence for synthesis and playback.
    ///
    /// Rejects new submissions immediately when the bounded job queue is full.
    /// Accepted synthesis jobs can still fail asynchronously (including playback
    /// queue overflow); these produce [`PlaybackEvent::Failed`] with their ID.
    /// Returns the assigned [`UtteranceId`].
    ///
    /// # Errors
    /// Returns [`VoiceError::EmptyText`], [`VoiceError::QueueFull`], or
    /// [`VoiceError::QueueClosed`] if admission fails. No ID is returned on failure.
    pub fn speak(&self, text: impl Into<String>) -> Result<UtteranceId, VoiceError> {
        let text = text.into();
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(VoiceError::EmptyText);
        }

        let id = UtteranceId(self.next_id.fetch_add(1, Ordering::SeqCst));
        let current_gen = self.generation.load(Ordering::SeqCst);
        let handed_to_tts = std::time::Instant::now();

        let job = SynthesisJob {
            id,
            generation: current_gen,
            text: trimmed.to_string(),
        };

        self.queue_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .record(UtteranceStageTimings {
                id,
                generation: current_gen,
                handed_to_tts,
                synthesis_done: None,
                first_sample_played: None,
                playback_finished: None,
            });

        self.job_tx.try_send(job).map_err(|error| match error {
            TrySendError::Full(_) => VoiceError::QueueFull,
            TrySendError::Disconnected(_) => VoiceError::QueueClosed,
        })?;

        Ok(id)
    }

    /// Queues the acknowledgement chime (two soft rising tones, about 200 ms) for
    /// playback, as an utterance of its own: it emits [`PlaybackEvent::Started`], with
    /// empty text, then [`PlaybackEvent::Finished`], or [`PlaybackEvent::Cancelled`]
    /// like any sentence. Computed when the player started, it skips synthesis: it
    /// plays right after what is already queued, before sentences still being
    /// synthesized. Returns its [`UtteranceId`].
    ///
    /// # Errors
    /// Returns [`VoiceError::QueueFull`] when the playback queue is full; no ID is
    /// returned then.
    pub fn chime(&self) -> Result<UtteranceId, VoiceError> {
        let samples = self.chime.clone();
        let mut state = self
            .queue_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.queue.len() >= self.config.queue_capacity {
            return Err(VoiceError::QueueFull);
        }
        let id = UtteranceId(self.next_id.fetch_add(1, Ordering::SeqCst));
        let generation = state.generation;
        let now = std::time::Instant::now();
        state.record(UtteranceStageTimings {
            id,
            generation,
            handed_to_tts: now,
            synthesis_done: Some(now),
            first_sample_played: None,
            playback_finished: None,
        });
        state.queue.push_back(QueuedSentence {
            id,
            generation,
            text: String::new(),
            samples,
        });
        Ok(id)
    }

    /// Cancels currently playing and all queued utterances immediately (barge-in).
    ///
    /// Emits [`PlaybackEvent::Cancelled`] for every active or pending utterance,
    /// clears the audio buffer, and transitions playback immediately to silence.
    pub fn cancel(&self) {
        let replacement = VecDeque::with_capacity(self.config.queue_capacity);
        let mut state = self
            .queue_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let active = state.current_utterance.take();
        let samples = std::mem::take(&mut state.current_samples);
        let queued = std::mem::replace(&mut state.queue, replacement);
        state.current_pos = 0;
        drop(state);
        drop(samples);
        if let Some(id) = active {
            emit_event(&self.subscribers, PlaybackEvent::Cancelled { id });
        }
        for item in queued {
            emit_event(&self.subscribers, PlaybackEvent::Cancelled { id: item.id });
        }
    }

    /// Returns `true` if an utterance is currently playing.
    #[must_use]
    pub fn is_playing(&self) -> bool {
        let state = self
            .queue_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.current_utterance.is_some() || !state.queue.is_empty()
    }

    /// Returns the currently playing utterance ID, if any.
    #[must_use]
    pub fn active_utterance(&self) -> Option<UtteranceId> {
        let state = self
            .queue_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.current_utterance
    }

    /// Take the latest asynchronous synthesis or output-stream error.
    /// A new error replaces the previous one, bounding retained diagnostics to
    /// one failure. Each synthesis failure is also published as a distinct
    /// [`PlaybackEvent::Failed`]; inspect `event_stats` to detect delivery loss.
    #[must_use]
    pub fn take_error(&self) -> Option<VoiceError> {
        self.last_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }

    /// Returns the active voice configuration.
    #[must_use]
    pub fn config(&self) -> &VoiceConfig {
        &self.config
    }

    /// Returns the active output device sample rate in hertz.
    #[must_use]
    pub fn device_sample_rate(&self) -> u32 {
        self.device_sample_rate
    }

    /// Returns the synthesizer's native sample rate in hertz (24 kHz for Kokoro,
    /// 22.05 kHz for Piper `pt_BR` faber-medium).
    #[must_use]
    pub fn native_sample_rate(&self) -> u32 {
        self.native_sample_rate
    }

    /// Returns `true` when the device did not take the native rate, so each
    /// sentence is resampled to [`VoicePlayer::device_sample_rate`] before playback.
    #[must_use]
    pub fn is_resampling(&self) -> bool {
        self.native_sample_rate != self.device_sample_rate
    }
}

impl TextToSpeech for VoicePlayer {
    /// Submit lazily to a dedicated blocking worker without requiring an async runtime.
    /// Rejects empty text or a full/disconnected port queue. Dropping the future
    /// skips queued work; native inference already running finishes off-thread
    /// and its result is discarded. The port queue shares `queue_capacity` as its bound.
    fn synthesize(
        &self,
        text: &str,
        sample_rate: u32,
    ) -> impl Future<Output = Result<Vec<f32>, PortError>> + Send {
        let sender = self.port_tx.clone();
        async move {
            if text.trim().is_empty() {
                return Err(PortError::InvalidInput(VoiceError::EmptyText.to_string()));
            }
            let (response, reply) = synthesis_response();
            sender
                .try_send(PortJob {
                    text: text.to_owned(),
                    sample_rate,
                    reply,
                })
                .map_err(|error| {
                    PortError::Unavailable(match error {
                        TrySendError::Full(_) => VoiceError::QueueFull.to_string(),
                        TrySendError::Disconnected(_) => VoiceError::QueueClosed.to_string(),
                    })
                })?;
            response.await
        }
    }
}

impl Drop for VoicePlayer {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use std::task::{Context, Poll, Waker};
    use std::thread;
    use std::time::Duration;

    use super::*;

    fn wait_for<F: Future>(future: F) -> F::Output {
        struct ThreadWake(thread::Thread);
        impl std::task::Wake for ThreadWake {
            fn wake(self: Arc<Self>) {
                self.0.unpark();
            }
        }
        let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
        let mut context = Context::from_waker(&waker);
        let mut future = std::pin::pin!(future);
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            if let Poll::Ready(result) = future.as_mut().poll(&mut context) {
                return result;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "worker failed to wake its caller"
            );
            thread::park_timeout(Duration::from_millis(50));
        }
    }

    #[test]
    fn full_submission_queue_returns_without_waiting_for_worker() {
        let mut player = VoicePlayer::mock();
        let (sender, receiver) = sync_channel(1);
        player.job_tx = sender;
        player.speak("first").unwrap();
        let (done_tx, done_rx) = sync_channel(1);
        let handle = thread::spawn(move || {
            done_tx.send(player.speak("overflow")).unwrap();
        });
        let early = done_rx.recv_timeout(Duration::from_millis(200));
        drop(receiver); // Always release a buggy blocking sender before asserting.
        handle.join().unwrap();
        assert!(early.is_ok(), "submission blocked the caller");
        assert!(early.unwrap().is_err(), "overflow must be rejected");
    }

    #[test]
    fn port_synthesis_is_lazy_and_runs_off_the_polling_thread() {
        let calls = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let observed = Arc::clone(&calls);
        let mut player = VoicePlayer::mock();
        let polling_thread = thread::current().id();
        player.port_tx = spawn_port_worker(
            Arc::new(SynthesizerBackend::Test(Arc::new(move || {
                assert_ne!(thread::current().id(), polling_thread);
                observed.fetch_add(1, Ordering::SeqCst);
                Ok(vec![0.25])
            }))),
            Arc::clone(&player.stopped),
            1,
            1.0,
            0,
        );
        let future = player.synthesize("hello", 24_000);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "inference ran before polling"
        );
        assert_eq!(wait_for(future).unwrap().len(), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cancellation_releases_queue_before_publishing_events() {
        let player = Arc::new(VoicePlayer::mock());
        {
            // Ten seconds to play, so only the cancellation can clear it.
            let mut state = player.queue_state.lock().unwrap();
            state.current_utterance = Some(UtteranceId(77));
            state.current_samples = vec![0.0; 240_000];
        }
        let subscribers = player.subscribers.state.senders.lock().unwrap();
        let cancelling = Arc::clone(&player);
        let handle = thread::spawn(move || cancelling.cancel());
        let deadline = std::time::Instant::now() + Duration::from_millis(200);
        let mut cleared = false;
        while std::time::Instant::now() < deadline {
            if let Ok(state) = player.queue_state.try_lock()
                && state.current_utterance.is_none()
            {
                cleared = true;
                break;
            }
            thread::yield_now();
        }
        drop(subscribers);
        handle.join().unwrap();
        assert!(
            cleared,
            "event publication retained the playback queue lock"
        );
    }

    #[test]
    fn empty_speech_is_rejected_without_returning_an_utterance_id() {
        let player = VoicePlayer::mock();
        assert!(player.speak(" \t\n").is_err());
        assert_eq!(player.speak("real sentence").unwrap(), UtteranceId(1));
    }

    #[test]
    fn slow_subscriber_has_a_bounded_backlog() {
        let player = VoicePlayer::mock();
        let receiver = player.subscribe().unwrap();
        for id in 0..100 {
            player.subscribers.state.publish(&PlaybackEvent::Finished {
                id: UtteranceId(id),
            });
        }
        assert_eq!(
            receiver.try_iter().count(),
            super::super::events::EVENT_CAPACITY
        );
        assert_eq!(
            player.event_stats().subscriber_dropped,
            100 - super::super::events::EVENT_CAPACITY as u64
        );
    }

    #[test]
    fn synthesis_failures_are_reported_for_each_utterance() {
        let (job_tx, job_rx) = sync_channel(2);
        let subscribers = Arc::new(EventHub::new());
        let event_rx = subscribers.subscribe().unwrap();
        let stopped = Arc::new(AtomicBool::new(false));
        let worker = spawn_synthesis_worker(SynthesisWorkerParams {
            job_rx,
            synthesizer: Arc::new(SynthesizerBackend::Test(Arc::new(|| {
                Err(VoiceError::Synthesis)
            }))),
            last_error: Arc::new(Mutex::new(None)),
            queue_state: Arc::new(Mutex::new(PlaybackQueueState {
                generation: 0,
                current_utterance: None,
                current_samples: Vec::new(),
                current_pos: 0,
                queue: VecDeque::with_capacity(2),
                timings: VecDeque::new(),
            })),
            generation: Arc::new(AtomicU64::new(0)),
            subscribers,
            stopped: Arc::clone(&stopped),
            device_sample_rate: 24_000,
            speed: 1.0,
            speaker_id: 0,
            queue_capacity: 2,
        });
        for id in 1..=2 {
            job_tx
                .send(SynthesisJob {
                    id: UtteranceId(id),
                    generation: 0,
                    text: "hello".into(),
                })
                .unwrap();
        }
        let events: Vec<_> = (0..2)
            .map(|_| event_rx.recv_timeout(Duration::from_secs(2)).unwrap())
            .collect();
        stopped.store(true, Ordering::Relaxed);
        drop(job_tx);
        worker.join().unwrap();
        assert_eq!(
            events,
            vec![
                PlaybackEvent::Failed {
                    id: UtteranceId(1),
                    reason: VoiceError::Synthesis.to_string()
                },
                PlaybackEvent::Failed {
                    id: UtteranceId(2),
                    reason: VoiceError::Synthesis.to_string()
                },
            ]
        );
    }

    #[test]
    fn port_queue_full_returns_an_error_without_waiting() {
        let mut player = VoicePlayer::mock();
        let (sender, _receiver) = sync_channel(1);
        player.port_tx = sender;
        let mut pending = Box::pin(player.synthesize("first", 24_000));
        let mut context = Context::from_waker(Waker::noop());
        assert!(pending.as_mut().poll(&mut context).is_pending());
        let error = wait_for(player.synthesize("second", 24_000)).unwrap_err();
        assert!(matches!(error, PortError::Unavailable(_)));
        drop(pending);
    }

    #[test]
    fn cancellation_during_inference_never_enqueues_stale_speech() {
        let (entered_tx, entered_rx) = sync_channel(1);
        let (release_tx, release_rx) = sync_channel(1);
        let release_rx = Mutex::new(release_rx);
        let backend = Arc::new(SynthesizerBackend::Test(Arc::new(move || {
            entered_tx.send(()).unwrap();
            release_rx
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(2))
                .unwrap();
            Ok(vec![0.25])
        })));
        let mut player = VoicePlayer::mock();
        let (sender, receiver) = sync_channel(2);
        player.job_tx = sender;
        let events = player.subscribe().unwrap();
        let worker = spawn_synthesis_worker(SynthesisWorkerParams {
            job_rx: receiver,
            synthesizer: backend,
            last_error: Arc::clone(&player.last_error),
            queue_state: Arc::clone(&player.queue_state),
            generation: Arc::clone(&player.generation),
            subscribers: Arc::clone(&player.subscribers),
            stopped: Arc::clone(&player.stopped),
            device_sample_rate: 24_000,
            speed: 1.0,
            speaker_id: 0,
            queue_capacity: 2,
        });
        let active = player.speak("inference in progress").unwrap();
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let queued = player.speak("queued before cancellation").unwrap();
        player.cancel();
        release_tx.send(()).unwrap();
        assert_eq!(
            events.recv_timeout(Duration::from_secs(2)).unwrap(),
            PlaybackEvent::Cancelled { id: active }
        );
        assert_eq!(
            events.recv_timeout(Duration::from_secs(2)).unwrap(),
            PlaybackEvent::Cancelled { id: queued }
        );
        assert!(player.queue_state.lock().unwrap().queue.is_empty());
        assert!(
            entered_rx.try_recv().is_err(),
            "stale queued job reached inference"
        );
        player.stopped.store(true, Ordering::Relaxed);
        worker.join().unwrap();
    }

    #[test]
    fn mock_player_emits_started_and_finished_events() {
        let player = VoicePlayer::mock();
        let rx = player.subscribe().unwrap();

        let id = player.speak("E aí Gabriel!").expect("speak should succeed");
        assert_eq!(id, UtteranceId(1));

        let first_event = rx
            .recv_timeout(Duration::from_millis(500))
            .expect("event received");
        assert_eq!(
            first_event,
            PlaybackEvent::Started {
                id: UtteranceId(1),
                text: "E aí Gabriel!".to_string(),
            }
        );

        let second_event = rx
            .recv_timeout(Duration::from_millis(500))
            .expect("event received");
        assert_eq!(second_event, PlaybackEvent::Finished { id: UtteranceId(1) });
    }

    #[test]
    fn mock_player_barge_in_cancels_active_and_queued_utterances() {
        let player = VoicePlayer::mock();
        let rx = player.subscribe().unwrap();

        let id1 = player.speak("Primeira frase longa.").expect("speak 1");
        let id2 = player.speak("Segunda frase.").expect("speak 2");
        assert_eq!(id1, UtteranceId(1));
        assert_eq!(id2, UtteranceId(2));

        let ev1 = rx
            .recv_timeout(Duration::from_millis(500))
            .expect("event 1");
        assert_eq!(
            ev1,
            PlaybackEvent::Started {
                id: id1,
                text: "Primeira frase longa.".to_string(),
            }
        );

        player.cancel();

        let mut events = Vec::new();
        while let Ok(ev) = rx.recv_timeout(Duration::from_millis(100)) {
            events.push(ev);
        }

        assert!(
            events.contains(&PlaybackEvent::Cancelled { id: id1 })
                || events.contains(&PlaybackEvent::Cancelled { id: id2 })
        );
        assert!(!player.is_playing());
    }

    /// Events from `rx` until `last` arrives, or none arrives for a second.
    fn events_until(rx: &Receiver<PlaybackEvent>, last: &PlaybackEvent) -> Vec<PlaybackEvent> {
        let mut events = Vec::new();
        while let Ok(event) = rx.recv_timeout(Duration::from_secs(1)) {
            let done = event == *last;
            events.push(event);
            if done {
                break;
            }
        }
        events
    }

    #[test]
    fn the_chime_plays_as_an_utterance_of_its_own() {
        let player = VoicePlayer::mock();
        let rx = player.subscribe().unwrap();
        let chime = player.chime().unwrap();
        assert_eq!(
            events_until(&rx, &PlaybackEvent::Finished { id: chime }),
            [
                PlaybackEvent::Started {
                    id: chime,
                    text: String::new()
                },
                PlaybackEvent::Finished { id: chime },
            ]
        );
        let timings = player.stage_timings(chime).unwrap();
        assert!(timings.first_sample_played.is_some());
        assert!(timings.playback_finished.is_some());
        // Speech that follows takes the next ID and plays as before.
        let spoken = player.speak("Oi.").unwrap();
        assert_eq!(spoken, UtteranceId(chime.0 + 1));
        assert_eq!(
            events_until(&rx, &PlaybackEvent::Finished { id: spoken }),
            [
                PlaybackEvent::Started {
                    id: spoken,
                    text: "Oi.".to_owned()
                },
                PlaybackEvent::Finished { id: spoken },
            ]
        );
    }

    #[test]
    fn a_barge_in_cuts_the_chime_like_speech() {
        let player = VoicePlayer::mock();
        let rx = player.subscribe().unwrap();
        let chime = player.chime().unwrap();
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            PlaybackEvent::Started {
                id: chime,
                text: String::new()
            }
        );
        player.cancel();
        assert_eq!(
            events_until(&rx, &PlaybackEvent::Cancelled { id: chime }),
            [PlaybackEvent::Cancelled { id: chime }]
        );
        assert!(!player.is_playing());
        assert!(
            player
                .stage_timings(chime)
                .unwrap()
                .playback_finished
                .is_none()
        );
    }

    #[test]
    fn the_chime_is_refused_when_the_playback_queue_is_full() {
        let player = VoicePlayer::mock();
        {
            // Hold the device on a long sentence while the queue fills up.
            let mut state = player.queue_state.lock().unwrap();
            state.current_utterance = Some(UtteranceId(900));
            state.current_samples = vec![0.0; 240_000];
        }
        for _ in 0..player.config().queue_capacity {
            player.chime().unwrap();
        }
        assert!(matches!(player.chime(), Err(VoiceError::QueueFull)));
        player.cancel();
        assert!(player.chime().is_ok());
    }

    #[test]
    fn text_to_speech_trait_synthesizes_samples() {
        let player = VoicePlayer::mock();
        let samples = wait_for(player.synthesize("Olá Gabriel", 24_000)).unwrap();
        assert!(!samples.is_empty());
    }

    #[test]
    fn test_real_voice_player_lifecycle() {
        let config = VoiceConfig::default();
        if config.validate().is_err() {
            return;
        }
        let Ok(player) = VoicePlayer::new(config) else {
            return;
        };
        let rx = player.subscribe().unwrap();
        let Ok(id) = player.speak("Olá Gabriel!") else {
            panic!("failed to speak");
        };
        let ev = rx.recv_timeout(Duration::from_secs(5));
        println!("Received event for {id}: {ev:?}");
    }

    #[test]
    fn test_stage_timings_recorded_on_mock_playback() {
        let player = VoicePlayer::mock();
        let rx = player.subscribe().expect("subscribe should succeed");

        let id = player
            .speak("Primeira frase para teste de tempo.")
            .expect("speak should succeed");

        // Immediately after speak, handed_to_tts must be present
        let immediate_timing = player
            .stage_timings(id)
            .expect("timing should exist for id");
        assert_eq!(immediate_timing.id, id);
        assert_eq!(player.last_stage_timings().map(|t| t.id), Some(id));

        // Drain lifecycle events until Finished
        while let Ok(event) = rx.recv_timeout(Duration::from_millis(200)) {
            if matches!(event, PlaybackEvent::Finished { id: finished_id } if finished_id == id) {
                break;
            }
        }

        let completed_timing = player
            .stage_timings(id)
            .expect("timing should exist after finish");
        let synthesis_done = completed_timing
            .synthesis_done
            .expect("synthesis_done should be recorded");
        let first_sample = completed_timing
            .first_sample_played
            .expect("first_sample_played should be recorded");
        let playback_finished = completed_timing
            .playback_finished
            .expect("playback_finished should be recorded");

        assert!(synthesis_done >= completed_timing.handed_to_tts);
        assert!(first_sample >= synthesis_done);
        assert!(playback_finished >= first_sample);
    }

    #[test]
    fn test_output_stream_stays_open_across_multiple_utterances_and_cancellation() {
        let player = VoicePlayer::mock();
        let rx = player.subscribe().expect("subscribe should succeed");

        // First utterance
        let id1 = player.speak("Primeira frase").expect("speak 1");
        assert_eq!(id1, UtteranceId(1));

        // Let it start and finish
        let mut got_id1_finished = false;
        while let Ok(event) = rx.recv_timeout(Duration::from_millis(150)) {
            if matches!(event, PlaybackEvent::Finished { id } if id == id1) {
                got_id1_finished = true;
                break;
            }
        }
        assert!(got_id1_finished, "id1 should finish normally");

        // Second utterance immediately follows without reopening stream
        let id2 = player.speak("Segunda frase").expect("speak 2");
        assert_eq!(id2, UtteranceId(2));

        // Barge-in cancel
        player.cancel();

        // Third utterance immediately succeeds on same player
        let id3 = player
            .speak("Terceira frase após cancelamento")
            .expect("speak 3");
        assert_eq!(id3, UtteranceId(3));

        // Check that id3 completes normally
        let mut got_id3_finished = false;
        while let Ok(event) = rx.recv_timeout(Duration::from_millis(200)) {
            if matches!(event, PlaybackEvent::Finished { id } if id == id3) {
                got_id3_finished = true;
                break;
            }
        }
        assert!(
            got_id3_finished,
            "id3 should finish normally after cancellation"
        );
    }
}
