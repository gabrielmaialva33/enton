//! Text-to-speech synthesis and audio playback (RFC 0001 §6, Task 0004).
//!
//! Provides a bounded sentence-by-sentence streaming voice synthesizer backed by
//! `sherpa-onnx` Kokoro TTS (Brazilian Portuguese) and `cpal` audio output.
//!
//! Emits lifecycle events ([`PlaybackEvent::Started`], [`PlaybackEvent::Finished`],
//! [`PlaybackEvent::Cancelled`], [`PlaybackEvent::Failed`]) for each utterance to record what was actually heard
//! and supports immediate barge-in interruption upon new stimulus.

use std::collections::VecDeque;
use std::fmt;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};
use enton_core::ports::{PortError, TextToSpeech};
use serde::{Deserialize, Serialize};
use sherpa_onnx::{
    GenerationConfig, LinearResampler, OfflineTts, OfflineTtsConfig, OfflineTtsKokoroModelConfig,
    OfflineTtsModelConfig,
};

pub use enton_core::UtteranceId;

/// Playback lifecycle events emitted by the voice player.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlaybackEvent {
    /// Playback of the utterance has started on the audio device.
    Started {
        /// Utterance identifier.
        id: UtteranceId,
        /// Full text of the spoken sentence.
        text: String,
    },
    /// Playback of the utterance completed normally.
    Finished {
        /// Utterance identifier.
        id: UtteranceId,
    },
    /// Synthesis or playback admission failed after accepting an utterance.
    Failed {
        /// Identifier returned by the successful submission.
        id: UtteranceId,
        /// Failure description, excluding the spoken text.
        reason: String,
    },
    /// Playback or queued utterance was cancelled (e.g. by barge-in).
    Cancelled {
        /// Utterance identifier.
        id: UtteranceId,
    },
}

/// Monotonic timestamps recorded across the lifecycle of an utterance in the voice pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UtteranceStageTimings {
    /// Utterance identifier.
    pub id: UtteranceId,
    /// Generation counter at time of submission.
    pub generation: u64,
    /// Instant when the chunk/sentence was submitted to the player (`speak`).
    pub handed_to_tts: std::time::Instant,
    /// Instant when native TTS synthesis finished producing audio samples.
    pub synthesis_done: Option<std::time::Instant>,
    /// Instant when the first audio sample was dispatched to the hardware audio buffer.
    pub first_sample_played: Option<std::time::Instant>,
    /// Instant when playback finished.
    pub playback_finished: Option<std::time::Instant>,
}

/// Consolidated latency breakdown across cortex deliberation and voice playback.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoiceLatencyBreakdown {
    /// Duration from thought request dispatch to first token received from cortex.
    pub time_to_first_token: Option<Duration>,
    /// Duration from first token to first chunk handed to TTS.
    pub time_to_first_chunk: Option<Duration>,
    /// Duration spent synthesizing the first chunk in Kokoro TTS.
    pub synthesis_duration: Option<Duration>,
    /// Duration from synthesis completion to first audio sample output on device.
    pub audio_device_dispatch: Option<Duration>,
    /// Total time to first audio (TTFA) from thought request dispatch to first sample played.
    pub total_time_to_first_audio: Option<Duration>,
}

impl VoiceLatencyBreakdown {
    /// Computes the latency breakdown from stage time points.
    #[must_use]
    pub fn compute(
        request_start: std::time::Instant,
        first_token: Option<std::time::Instant>,
        handed_to_tts: std::time::Instant,
        synthesis_done: Option<std::time::Instant>,
        first_sample_played: Option<std::time::Instant>,
    ) -> Self {
        let time_to_first_token =
            first_token.map(|tok| tok.saturating_duration_since(request_start));
        let time_to_first_chunk =
            first_token.map(|tok| handed_to_tts.saturating_duration_since(tok));
        let synthesis_duration =
            synthesis_done.map(|syn| syn.saturating_duration_since(handed_to_tts));
        let audio_device_dispatch = match (synthesis_done, first_sample_played) {
            (Some(syn), Some(play)) => Some(play.saturating_duration_since(syn)),
            _ => None,
        };
        let total_time_to_first_audio =
            first_sample_played.map(|play| play.saturating_duration_since(request_start));

        Self {
            time_to_first_token,
            time_to_first_chunk,
            synthesis_duration,
            audio_device_dispatch,
            total_time_to_first_audio,
        }
    }
}

/// Errors from voice configuration, synthesis, and playback.
#[derive(Debug, thiserror::Error)]
pub enum VoiceError {
    /// A required model asset is missing or has the wrong file type.
    #[error(
        "voice error: {asset} not found: {path}. See the README (Voice and microphone) for the model layout."
    )]
    MissingAsset {
        /// Name of the model asset.
        asset: &'static str,
        /// Configured file or directory path.
        path: PathBuf,
    },
    /// A required native model path is not UTF-8.
    #[error("voice error: invalid UTF-8 in {0}")]
    InvalidPath(&'static str),
    /// Sentence queue capacity is outside 1..=64.
    #[error("voice error: queue_capacity must be between 1 and 64 (got {0})")]
    QueueCapacity(usize),
    /// Native TTS initialization failed.
    #[error("voice error: failed to create sherpa-onnx OfflineTts instance")]
    Initialization,
    /// Native TTS generation failed; no transcript is included in the error.
    #[error("voice error: sherpa-onnx TTS generation returned no audio")]
    Synthesis,
    /// A resampler could not be created for the requested rates.
    #[error("voice error: failed to create resampler from {input_hz} to {output_hz} Hz")]
    Resampler {
        /// Native model sample rate in hertz.
        input_hz: i32,
        /// Requested output sample rate in hertz.
        output_hz: i32,
    },
    /// The audio backend failed during a named operation.
    #[error("voice error: {operation}: {source}")]
    Device {
        /// Backend operation that failed.
        operation: &'static str,
        /// Original backend error.
        #[source]
        source: cpal::Error,
    },
    /// No matching output device is available; `None` denotes the default.
    #[error("voice error: output device not found: {0:?}")]
    NoOutputDevice(Option<String>),
    /// The output device's sample format is unsupported.
    #[error("voice error: unsupported output sample format: {0:?}")]
    SampleFormat(SampleFormat),
    /// The synthesis worker has disconnected from its bounded job queue.
    #[error("voice error: synthesis queue closed")]
    QueueClosed,
    /// The bounded submission or playback queue is full; the new job is rejected.
    #[error("voice error: queue full; new utterance rejected")]
    QueueFull,
    /// Empty or whitespace-only text cannot identify an utterance.
    #[error("voice error: speech text is empty")]
    EmptyText,
    /// All 16 subscriber slots are occupied.
    #[error("voice error: event subscriber limit reached")]
    SubscriberLimit,
}

const EVENT_CAPACITY: usize = 64;
const MAX_SUBSCRIBERS: usize = 16;
const MAX_PLAYBACK_QUEUE: usize = 64;
// One active utterance can finish, and each queued item can start and finish.
const CALLBACK_EVENTS: usize = 1 + 2 * MAX_PLAYBACK_QUEUE;

/// Cumulative loss counters for the bounded playback event transport.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PlaybackEventStats {
    /// New events dropped because the dispatch mailbox was full or disconnected.
    pub mailbox_dropped: u64,
    /// New deliveries dropped because a subscriber's 64-event queue was full.
    pub subscriber_dropped: u64,
    /// Disconnected subscribers removed during delivery.
    pub subscribers_disconnected: u64,
}

#[derive(Default)]
struct EventSubscribers {
    senders: Mutex<Vec<SyncSender<PlaybackEvent>>>,
    mailbox_dropped: AtomicU64,
    subscriber_dropped: AtomicU64,
    subscribers_disconnected: AtomicU64,
}

impl EventSubscribers {
    fn publish(&self, event: &PlaybackEvent) {
        self.senders
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|sender| match sender.try_send(event.clone()) {
                Ok(()) => true,
                Err(TrySendError::Full(_)) => {
                    self.subscriber_dropped.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(TrySendError::Disconnected(_)) => {
                    self.subscribers_disconnected
                        .fetch_add(1, Ordering::Relaxed);
                    false
                }
            });
    }
}

struct EventHub {
    sender: SyncSender<PlaybackEvent>,
    state: Arc<EventSubscribers>,
}

impl EventHub {
    fn new() -> Self {
        let (sender, receiver) = sync_channel::<PlaybackEvent>(EVENT_CAPACITY);
        let state = Arc::new(EventSubscribers::default());
        let publishing = Arc::clone(&state);
        // Cloning text and subscriber fan-out happen here, never on the audio callback.
        thread::spawn(move || {
            while let Ok(event) = receiver.recv() {
                publishing.publish(&event);
            }
        });
        Self { sender, state }
    }

    fn subscribe(&self) -> Result<Receiver<PlaybackEvent>, VoiceError> {
        let (sender, receiver) = sync_channel(EVENT_CAPACITY);
        let mut senders = self
            .state
            .senders
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if senders.len() >= MAX_SUBSCRIBERS {
            return Err(VoiceError::SubscriberLimit);
        }
        senders.push(sender);
        drop(senders);
        Ok(receiver)
    }

    fn stats(&self) -> PlaybackEventStats {
        PlaybackEventStats {
            mailbox_dropped: self.state.mailbox_dropped.load(Ordering::Relaxed),
            subscriber_dropped: self.state.subscriber_dropped.load(Ordering::Relaxed),
            subscribers_disconnected: self.state.subscribers_disconnected.load(Ordering::Relaxed),
        }
    }
}

fn emit_event(hub: &EventHub, event: PlaybackEvent) {
    if hub.sender.try_send(event).is_err() {
        hub.state.mailbox_dropped.fetch_add(1, Ordering::Relaxed);
    }
}

struct EventBatch {
    events: [Option<PlaybackEvent>; CALLBACK_EVENTS],
    length: usize,
}

impl EventBatch {
    fn new() -> Self {
        Self {
            events: std::array::from_fn(|_| None),
            length: 0,
        }
    }

    fn push(&mut self, event: PlaybackEvent, hub: &EventHub) {
        if let Some(slot) = self.events.get_mut(self.length) {
            *slot = Some(event);
            self.length += 1;
        } else {
            // Defensive accounting if a future producer violates the playback bound.
            hub.state.mailbox_dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn publish(self, hub: &EventHub) {
        for event in self.events.into_iter().flatten() {
            emit_event(hub, event);
        }
    }
}

/// Configuration for the Kokoro TTS engine and CPAL audio playback.
#[derive(Debug, Clone)]
pub struct VoiceConfig {
    /// Path to the ONNX model file (e.g. `model.onnx`).
    pub model_path: PathBuf,
    /// Path to the speaker voices binary (e.g. `voices.bin`).
    pub voices_path: PathBuf,
    /// Path to the phoneme tokens file (e.g. `tokens.txt`).
    pub tokens_path: PathBuf,
    /// Path to the espeak-ng-data directory.
    pub data_dir: PathBuf,
    /// Optional path to the dict directory (e.g. `dict/`).
    pub dict_dir: Option<PathBuf>,
    /// Optional path to a custom lexicon file.
    pub lexicon_path: Option<PathBuf>,
    /// Language code passed to Kokoro (e.g. `"pt-br"` or `"pt"`).
    pub lang: String,
    /// Speaker ID within voices.bin (`42` for `pf_dora`, `43` for `pm_alex`).
    pub speaker_id: i32,
    /// Playback speed factor (default 1.0).
    pub speed: f32,
    /// Number of worker threads for TTS inference (default 2).
    pub num_threads: i32,
    /// Bound (1..=64) for each synthesis job queue and the ready playback queue.
    /// New jobs are rejected when full; ready-queue overflow emits `Failed`.
    pub queue_capacity: usize,
    /// Specific CPAL device identifier string; `None` uses default output device.
    pub device_id: Option<String>,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        let base = std::env::var("HOME").map_or_else(
            |_| PathBuf::from("."),
            |h| {
                let v1_1 =
                    PathBuf::from(&h).join(".cache/enton/models/kokoro-int8-multi-lang-v1_1");
                if v1_1.is_dir() {
                    v1_1
                } else {
                    PathBuf::from(h).join(".cache/enton/models/kokoro-int8-multi-lang-v1_0")
                }
            },
        );
        let dict = base.join("dict");
        Self {
            model_path: base.join("model.int8.onnx"),
            voices_path: base.join("voices.bin"),
            tokens_path: base.join("tokens.txt"),
            data_dir: base.join("espeak-ng-data"),
            dict_dir: if dict.is_dir() { Some(dict) } else { None },
            lexicon_path: None,
            lang: "pt-br".to_string(),
            speaker_id: 42, // pf_dora (Brazilian Portuguese female)
            speed: 1.0,
            num_threads: 2,
            queue_capacity: 8,
            device_id: None,
        }
    }
}

impl VoiceConfig {
    /// Creates a voice configuration tailored for a specific hardware profile name.
    ///
    /// Configures 8 TTS worker threads for `"desktop"` and 2 worker threads for `"t1-ref"`.
    #[must_use]
    pub fn for_profile(profile_name: &str) -> Self {
        let num_threads = if profile_name == "desktop" { 8 } else { 2 };
        Self {
            num_threads,
            ..Self::default()
        }
    }

    /// Validates that model files exist on disk.
    ///
    /// # Errors
    /// Returns [`VoiceError`] if any required model file or directory is missing.
    pub fn validate(&self) -> Result<(), VoiceError> {
        if !self.model_path.is_file() {
            return Err(VoiceError::MissingAsset {
                asset: "model file",
                path: self.model_path.clone(),
            });
        }
        if !self.voices_path.is_file() {
            return Err(VoiceError::MissingAsset {
                asset: "voices file",
                path: self.voices_path.clone(),
            });
        }
        if !self.tokens_path.is_file() {
            return Err(VoiceError::MissingAsset {
                asset: "tokens file",
                path: self.tokens_path.clone(),
            });
        }
        if !self.data_dir.is_dir() {
            return Err(VoiceError::MissingAsset {
                asset: "espeak-ng-data directory",
                path: self.data_dir.clone(),
            });
        }
        if self.queue_capacity == 0 || self.queue_capacity > 64 {
            return Err(VoiceError::QueueCapacity(self.queue_capacity));
        }
        Ok(())
    }
}

/// A queued sentence ready for audio playback.
struct QueuedSentence {
    id: UtteranceId,
    generation: u64,
    text: String,
    samples: Vec<f32>,
}

/// Internal mutable state shared between worker threads and audio callback.
struct PlaybackQueueState {
    generation: u64,
    current_utterance: Option<UtteranceId>,
    current_samples: Vec<f32>,
    current_pos: usize,
    queue: VecDeque<QueuedSentence>,
    timings: VecDeque<UtteranceStageTimings>,
}

enum SynthesizerBackend {
    Offline(Mutex<OfflineTts>),
    Mock {
        sample_rate: u32,
    },
    #[cfg(test)]
    Test(Arc<dyn Fn() -> Result<Vec<f32>, VoiceError> + Send + Sync>),
}

impl SynthesizerBackend {
    fn synthesize(
        &self,
        text: &str,
        target_sample_rate: u32,
        speed: f32,
        speaker_id: i32,
    ) -> Result<Vec<f32>, VoiceError> {
        match self {
            #[cfg(test)]
            Self::Test(generate) => generate(),
            Self::Offline(tts_mutex) => {
                let gen_config = GenerationConfig {
                    speed,
                    sid: speaker_id,
                    silence_scale: 0.2,
                    ..GenerationConfig::default()
                };

                let tts = tts_mutex
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let audio = tts
                    .generate_with_config(text, &gen_config, None::<fn(&[f32], f32) -> bool>)
                    .ok_or(VoiceError::Synthesis)?;
                drop(tts);

                let samples = audio.samples();
                let native_rate = u32::try_from(audio.sample_rate()).unwrap_or(24_000);

                if native_rate == target_sample_rate || target_sample_rate == 0 {
                    Ok(samples.to_vec())
                } else {
                    let in_rate = i32::try_from(native_rate).unwrap_or(24_000);
                    let out_rate = i32::try_from(target_sample_rate).unwrap_or(48_000);
                    let resampler = LinearResampler::create(in_rate, out_rate).ok_or(
                        VoiceError::Resampler {
                            input_hz: in_rate,
                            output_hz: out_rate,
                        },
                    )?;
                    let mut converted = resampler.resample(samples, false);
                    converted.extend(resampler.resample(&[], true));
                    Ok(converted)
                }
            }
            Self::Mock { sample_rate } => {
                let count = usize::try_from(*sample_rate / 50).unwrap_or(480);
                let mut samples = Vec::with_capacity(count);
                for i in 0..count {
                    let phase =
                        (i as f32) * 440.0 * 2.0 * std::f32::consts::PI / (*sample_rate as f32);
                    samples.push((phase.sin() * 0.1).clamp(-1.0, 1.0));
                }
                Ok(samples)
            }
        }
    }
}

// A runtime-independent one-shot response keeps `voice` usable without the
// optional Tokio dependency. The producer wakes outside the short state lock.
#[derive(Default)]
struct ResponseState {
    result: Option<Result<Vec<f32>, PortError>>,
    waker: Option<Waker>,
    cancelled: bool,
}

struct SynthesisResponse(Arc<Mutex<ResponseState>>);
struct SynthesisReply(Option<Arc<Mutex<ResponseState>>>);

fn synthesis_response() -> (SynthesisResponse, SynthesisReply) {
    let state = Arc::new(Mutex::new(ResponseState::default()));
    (
        SynthesisResponse(Arc::clone(&state)),
        SynthesisReply(Some(state)),
    )
}

impl Future for SynthesisResponse {
    type Output = Result<Vec<f32>, PortError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(result) = state.result.take() {
            Poll::Ready(result)
        } else {
            state.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

impl Drop for SynthesisResponse {
    fn drop(&mut self) {
        let mut state = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.cancelled = true;
        state.waker = None;
    }
}

impl SynthesisReply {
    fn is_cancelled(&self) -> bool {
        self.0.as_ref().is_none_or(|state| {
            state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .cancelled
        })
    }

    fn complete(mut self, result: Result<Vec<f32>, PortError>) {
        if let Some(state) = self.0.take() {
            finish_response(&state, result);
        }
    }
}

fn finish_response(shared: &Mutex<ResponseState>, result: Result<Vec<f32>, PortError>) {
    let mut state = shared
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.cancelled {
        return;
    }
    state.result = Some(result);
    let waker = state.waker.take();
    drop(state);
    if let Some(waker) = waker {
        waker.wake();
    }
}

impl Drop for SynthesisReply {
    fn drop(&mut self) {
        if let Some(state) = self.0.take() {
            finish_response(
                &state,
                Err(PortError::Unavailable(
                    "synthesis worker stopped".to_owned(),
                )),
            );
        }
    }
}

struct PortJob {
    text: String,
    sample_rate: u32,
    reply: SynthesisReply,
}

fn spawn_port_worker(
    synthesizer: Arc<SynthesizerBackend>,
    stopped: Arc<AtomicBool>,
    capacity: usize,
    speed: f32,
    speaker_id: i32,
) -> SyncSender<PortJob> {
    let (sender, receiver) = sync_channel::<PortJob>(capacity);
    thread::spawn(move || {
        while !stopped.load(Ordering::Relaxed) {
            let job = match receiver.recv_timeout(Duration::from_millis(50)) {
                Ok(job) => job,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            };
            if job.reply.is_cancelled() {
                continue;
            }
            let result = synthesizer
                .synthesize(&job.text, job.sample_rate, speed, speaker_id)
                .map_err(|error| PortError::Failed(error.to_string()));
            job.reply.complete(result);
        }
    });
    sender
}

struct SynthesisJob {
    id: UtteranceId,
    generation: u64,
    text: String,
}

/// Voice synthesizer and audio player.
///
/// Delivers sentence-by-sentence speech synthesis with bounded buffering,
/// accurate lifecycle event reporting, and instantaneous barge-in cancellation.
pub struct VoicePlayer {
    config: VoiceConfig,
    next_id: AtomicU64,
    generation: Arc<AtomicU64>,
    subscribers: Arc<EventHub>,
    queue_state: Arc<Mutex<PlaybackQueueState>>,
    port_tx: SyncSender<PortJob>,
    last_error: Arc<Mutex<Option<VoiceError>>>,
    job_tx: SyncSender<SynthesisJob>,
    _worker_handle: Option<JoinHandle<()>>,
    _stream: Option<cpal::Stream>,
    stopped: Arc<AtomicBool>,
    device_sample_rate: u32,
}

impl fmt::Debug for VoicePlayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VoicePlayer")
            .field("config", &self.config)
            .field("next_id", &self.next_id.load(Ordering::Relaxed))
            .field("generation", &self.generation.load(Ordering::Relaxed))
            .field("device_sample_rate", &self.device_sample_rate)
            .finish_non_exhaustive()
    }
}

fn create_offline_tts(config: &VoiceConfig) -> Result<OfflineTts, VoiceError> {
    let kokoro_config = OfflineTtsKokoroModelConfig {
        model: Some(
            config
                .model_path
                .to_str()
                .ok_or(VoiceError::InvalidPath("model_path"))?
                .to_string(),
        ),
        voices: Some(
            config
                .voices_path
                .to_str()
                .ok_or(VoiceError::InvalidPath("voices_path"))?
                .to_string(),
        ),
        tokens: Some(
            config
                .tokens_path
                .to_str()
                .ok_or(VoiceError::InvalidPath("tokens_path"))?
                .to_string(),
        ),
        data_dir: Some(
            config
                .data_dir
                .to_str()
                .ok_or(VoiceError::InvalidPath("data_dir"))?
                .to_string(),
        ),
        dict_dir: config
            .dict_dir
            .as_ref()
            .and_then(|p| p.to_str().map(ToString::to_string)),
        lexicon: config
            .lexicon_path
            .as_ref()
            .and_then(|p| p.to_str().map(ToString::to_string)),
        lang: Some(config.lang.clone()),
        length_scale: 1.0,
    };

    let tts_config = OfflineTtsConfig {
        model: OfflineTtsModelConfig {
            kokoro: kokoro_config,
            num_threads: config.num_threads,
            debug: false,
            provider: Some("cpu".to_string()),
            ..OfflineTtsModelConfig::default()
        },
        rule_fsts: None,
        max_num_sentences: 1,
        rule_fars: None,
        silence_scale: 0.2,
    };

    OfflineTts::create(&tts_config).ok_or(VoiceError::Initialization)
}

fn open_output_device(
    device_id: Option<&str>,
) -> Result<(cpal::Device, cpal::SupportedStreamConfig), VoiceError> {
    let host = cpal::default_host();
    let device = match device_id {
        Some(target_id) => host
            .output_devices()
            .map_err(|source| VoiceError::Device {
                operation: "failed to query output devices",
                source,
            })?
            .find(|d| d.id().is_ok_and(|id| id.to_string() == target_id))
            .ok_or(VoiceError::NoOutputDevice(Some(target_id.to_owned())))?,
        None => host
            .default_output_device()
            .ok_or(VoiceError::NoOutputDevice(None))?,
    };

    let supported_config = device
        .default_output_config()
        .map_err(|source| VoiceError::Device {
            operation: "failed to query default output config",
            source,
        })?;

    Ok((device, supported_config))
}

impl VoicePlayer {
    /// Creates and starts a new voice player with the supplied configuration and default audio device.
    ///
    /// # Errors
    /// Returns [`VoiceError`] if model files are missing, invalid, or no CPAL output device is found.
    pub fn new(config: VoiceConfig) -> Result<Self, VoiceError> {
        config.validate()?;
        let tts = create_offline_tts(&config)?;

        let (device, supported_config) = open_output_device(config.device_id.as_deref())?;
        let sample_rate = supported_config.sample_rate();
        let channels = usize::from(supported_config.channels());

        let subscribers = Arc::new(EventHub::new());
        let last_error = Arc::new(Mutex::new(None));
        let generation = Arc::new(AtomicU64::new(0));
        let queue_state = Arc::new(Mutex::new(PlaybackQueueState {
            generation: 0,
            current_utterance: None,
            current_samples: Vec::new(),
            current_pos: 0,
            queue: VecDeque::with_capacity(config.queue_capacity),
            timings: VecDeque::with_capacity(64),
        }));

        let stream = match supported_config.sample_format() {
            SampleFormat::F32 => build_cpal_stream::<f32>(
                &device,
                &supported_config,
                channels,
                Arc::clone(&queue_state),
                Arc::clone(&subscribers),
                Arc::clone(&last_error),
            )?,
            SampleFormat::I16 => build_cpal_stream::<i16>(
                &device,
                &supported_config,
                channels,
                Arc::clone(&queue_state),
                Arc::clone(&subscribers),
                Arc::clone(&last_error),
            )?,
            SampleFormat::U16 => build_cpal_stream::<u16>(
                &device,
                &supported_config,
                channels,
                Arc::clone(&queue_state),
                Arc::clone(&subscribers),
                Arc::clone(&last_error),
            )?,
            other => {
                return Err(VoiceError::SampleFormat(other));
            }
        };

        stream.play().map_err(|source| VoiceError::Device {
            operation: "failed to start cpal playback stream",
            source,
        })?;

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
            speaker_id: config.speaker_id,
            queue_capacity: config.queue_capacity,
        };

        let worker_handle = spawn_synthesis_worker(worker_params);
        let port_tx = spawn_port_worker(
            synthesizer,
            Arc::clone(&stopped),
            config.queue_capacity,
            config.speed,
            config.speaker_id,
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
            device_sample_rate: sample_rate,
        })
    }

    /// Creates a mock voice player that runs without requiring model files or physical audio devices.
    ///
    /// Ideal for headless testing and CI environments.
    #[must_use]
    pub fn mock() -> Self {
        let subscribers = Arc::new(EventHub::new());
        let last_error = Arc::new(Mutex::new(None));
        let generation = Arc::new(AtomicU64::new(0));
        let queue_state = Arc::new(Mutex::new(PlaybackQueueState {
            generation: 0,
            current_utterance: None,
            current_samples: Vec::new(),
            current_pos: 0,
            queue: VecDeque::with_capacity(8),
            timings: VecDeque::with_capacity(64),
        }));

        let synthesizer = Arc::new(SynthesizerBackend::Mock {
            sample_rate: 24_000,
        });
        let (job_tx, job_rx) = sync_channel::<SynthesisJob>(8);
        let stopped = Arc::new(AtomicBool::new(false));

        let worker_handle = spawn_mock_worker(
            job_rx,
            Arc::clone(&queue_state),
            Arc::clone(&generation),
            Arc::clone(&subscribers),
            Arc::clone(&stopped),
        );

        let port_tx = spawn_port_worker(synthesizer, Arc::clone(&stopped), 8, 1.0, 42);
        Self {
            config: VoiceConfig::default(),
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
            device_sample_rate: 24_000,
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

        {
            let mut state = self
                .queue_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.timings.len() >= 64 {
                state.timings.pop_front();
            }
            state.timings.push_back(UtteranceStageTimings {
                id,
                generation: current_gen,
                handed_to_tts,
                synthesis_done: None,
                first_sample_played: None,
                playback_finished: None,
            });
        }

        self.job_tx.try_send(job).map_err(|error| match error {
            TrySendError::Full(_) => VoiceError::QueueFull,
            TrySendError::Disconnected(_) => VoiceError::QueueClosed,
        })?;

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

struct SynthesisWorkerParams {
    job_rx: Receiver<SynthesisJob>,
    synthesizer: Arc<SynthesizerBackend>,
    last_error: Arc<Mutex<Option<VoiceError>>>,
    queue_state: Arc<Mutex<PlaybackQueueState>>,
    generation: Arc<AtomicU64>,
    subscribers: Arc<EventHub>,
    stopped: Arc<AtomicBool>,
    device_sample_rate: u32,
    speed: f32,
    speaker_id: i32,
    queue_capacity: usize,
}

fn spawn_synthesis_worker(params: SynthesisWorkerParams) -> JoinHandle<()> {
    thread::spawn(move || {
        while !params.stopped.load(Ordering::Relaxed) {
            let job = match params.job_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(job) => job,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            };

            if job.generation != params.generation.load(Ordering::SeqCst) {
                emit_event(&params.subscribers, PlaybackEvent::Cancelled { id: job.id });
                continue;
            }

            let samples = match params.synthesizer.synthesize(
                &job.text,
                params.device_sample_rate,
                params.speed,
                params.speaker_id,
            ) {
                Ok(s) => s,
                Err(err) => {
                    emit_event(
                        &params.subscribers,
                        PlaybackEvent::Failed {
                            id: job.id,
                            reason: err.to_string(),
                        },
                    );
                    *params
                        .last_error
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(err);
                    continue;
                }
            };

            let synthesis_done_now = std::time::Instant::now();
            if job.generation != params.generation.load(Ordering::SeqCst) {
                emit_event(&params.subscribers, PlaybackEvent::Cancelled { id: job.id });
                continue;
            }

            let mut state = params
                .queue_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(timing) = state.timings.iter_mut().find(|t| t.id == job.id) {
                timing.synthesis_done = Some(synthesis_done_now);
            }
            if job.generation != state.generation {
                drop(state);
                emit_event(&params.subscribers, PlaybackEvent::Cancelled { id: job.id });
                continue;
            }

            if state.queue.len() >= params.queue_capacity {
                drop(state);
                emit_event(
                    &params.subscribers,
                    PlaybackEvent::Failed {
                        id: job.id,
                        reason: VoiceError::QueueFull.to_string(),
                    },
                );
                continue;
            }
            state.queue.push_back(QueuedSentence {
                id: job.id,
                generation: job.generation,
                text: job.text,
                samples,
            });
        }
    })
}

fn spawn_mock_worker(
    job_rx: Receiver<SynthesisJob>,
    queue_state: Arc<Mutex<PlaybackQueueState>>,
    generation: Arc<AtomicU64>,
    subscribers: Arc<EventHub>,
    stopped: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        while !stopped.load(Ordering::Relaxed) {
            let job = match job_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(job) => job,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            };

            let current_gen = generation.load(Ordering::SeqCst);
            if job.generation != current_gen {
                emit_event(&subscribers, PlaybackEvent::Cancelled { id: job.id });
                continue;
            }

            let mock_now = std::time::Instant::now();
            {
                let mut state = queue_state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if job.generation != state.generation {
                    drop(state);
                    emit_event(&subscribers, PlaybackEvent::Cancelled { id: job.id });
                    continue;
                }
                state.current_utterance = Some(job.id);
                if let Some(timing) = state.timings.iter_mut().find(|t| t.id == job.id) {
                    timing.synthesis_done = Some(mock_now);
                    timing.first_sample_played = Some(mock_now);
                }
            }

            emit_event(
                &subscribers,
                PlaybackEvent::Started {
                    id: job.id,
                    text: job.text.clone(),
                },
            );

            let mut played_ms = 0;
            let mut cancelled = false;
            while played_ms < 25 {
                thread::sleep(Duration::from_millis(5));
                played_ms += 5;
                if generation.load(Ordering::SeqCst) != current_gen {
                    cancelled = true;
                    break;
                }
            }

            let mut state = queue_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !cancelled && state.generation == current_gen {
                state.current_utterance = None;
                let finish_now = std::time::Instant::now();
                if let Some(timing) = state.timings.iter_mut().find(|t| t.id == job.id) {
                    timing.playback_finished = Some(finish_now);
                }
                drop(state);
                emit_event(&subscribers, PlaybackEvent::Finished { id: job.id });
            }
        }
    })
}

fn build_cpal_stream<T>(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    channels: usize,
    queue_state: Arc<Mutex<PlaybackQueueState>>,
    subscribers: Arc<EventHub>,
    last_error: Arc<Mutex<Option<VoiceError>>>,
) -> Result<cpal::Stream, VoiceError>
where
    T: SizedSample + FromSample<f32>,
{
    let err_fn = move |source| {
        *last_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(VoiceError::Device {
            operation: "output stream",
            source,
        });
    };

    let stream = device
        .build_output_stream(
            (*config).into(),
            move |data: &mut [T], _| {
                process_output_callback(data, channels, &queue_state, &subscribers);
            },
            err_fn,
            None,
        )
        .map_err(|source| VoiceError::Device {
            operation: "failed to build cpal output stream",
            source,
        })?;

    Ok(stream)
}

fn process_output_callback<T>(
    data: &mut [T],
    channels: usize,
    queue_state: &Arc<Mutex<PlaybackQueueState>>,
    subscribers: &Arc<EventHub>,
) where
    T: SizedSample + FromSample<f32>,
{
    let Ok(mut state) = queue_state.try_lock() else {
        for sample in data.iter_mut() {
            *sample = T::from_sample_(0.0);
        }
        return;
    };

    // Device configurations supply nonzero channels. Keep malformed input total.
    if channels == 0 {
        drop(state);
        data.fill(T::from_sample_(0.0));
        return;
    }
    let mut events = EventBatch::new();
    let mut frames = data.chunks_exact_mut(channels);
    for frame in frames.by_ref() {
        if state.current_pos >= state.current_samples.len() {
            if let Some(id) = state.current_utterance.take() {
                let finish_now = std::time::Instant::now();
                if let Some(timing) = state.timings.iter_mut().find(|t| t.id == id) {
                    timing.playback_finished = Some(finish_now);
                }
                events.push(PlaybackEvent::Finished { id }, subscribers);
            }
            state.current_samples.clear();
            state.current_pos = 0;

            let mut next_item = None;
            while let Some(item) = state.queue.pop_front() {
                if item.generation == state.generation {
                    next_item = Some(item);
                    break;
                }
                events.push(PlaybackEvent::Cancelled { id: item.id }, subscribers);
            }

            if let Some(item) = next_item {
                let play_now = std::time::Instant::now();
                if let Some(timing) = state.timings.iter_mut().find(|t| t.id == item.id) {
                    timing.first_sample_played = Some(play_now);
                }
                events.push(
                    PlaybackEvent::Started {
                        id: item.id,
                        text: item.text,
                    },
                    subscribers,
                );
                state.current_utterance = Some(item.id);
                state.current_samples = item.samples;
                state.current_pos = 0;
            } else {
                frame.fill(T::from_sample_(0.0));
                break;
            }
        }

        let sample = state
            .current_samples
            .get(state.current_pos)
            .copied()
            .unwrap_or(0.0);
        state.current_pos += 1;
        frame.fill(T::from_sample_(sample));
    }
    drop(state);
    events.publish(subscribers);
    for frame in frames {
        frame.fill(T::from_sample_(0.0));
    }
}

#[cfg(test)]
mod tests {
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
        let calls = Arc::new(AtomicU64::new(0));
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
        player.queue_state.lock().unwrap().current_utterance = Some(UtteranceId(77));
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
        assert_eq!(receiver.try_iter().count(), EVENT_CAPACITY);
        assert_eq!(
            player.event_stats().subscriber_dropped,
            100 - EVENT_CAPACITY as u64
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
    fn event_overflow_and_disconnection_are_counted_without_blocking() {
        let hub = EventHub::new();
        let disconnected = hub.subscribe().unwrap();
        drop(disconnected);
        hub.state
            .publish(&PlaybackEvent::Finished { id: UtteranceId(0) });
        assert_eq!(hub.stats().subscribers_disconnected, 1);

        // A mailbox with no consumer makes overflow deterministic without timing.
        let (sender, receiver) = sync_channel(1);
        let hub = EventHub {
            sender,
            state: Arc::new(EventSubscribers::default()),
        };
        emit_event(&hub, PlaybackEvent::Finished { id: UtteranceId(1) });
        emit_event(&hub, PlaybackEvent::Finished { id: UtteranceId(2) });
        assert_eq!(hub.stats().mailbox_dropped, 1);
        assert_eq!(
            receiver.try_recv().unwrap(),
            PlaybackEvent::Finished { id: UtteranceId(1) }
        );
    }

    #[test]
    fn subscriber_registration_is_bounded() {
        let hub = EventHub::new();
        let receivers: Vec<_> = (0..MAX_SUBSCRIBERS)
            .map(|_| hub.subscribe().unwrap())
            .collect();
        assert!(matches!(hub.subscribe(), Err(VoiceError::SubscriberLimit)));
        drop(receivers);
        hub.state
            .publish(&PlaybackEvent::Finished { id: UtteranceId(1) });
        assert_eq!(hub.stats().subscribers_disconnected, MAX_SUBSCRIBERS as u64);
        assert!(hub.subscribe().is_ok());
    }

    #[test]
    fn callback_does_not_wait_for_slow_event_subscribers() {
        let hub = Arc::new(EventHub::new());
        let receiver = hub.subscribe().unwrap();
        let subscribers = hub.state.senders.lock().unwrap();
        let state = Arc::new(Mutex::new(PlaybackQueueState {
            generation: 0,
            current_utterance: Some(UtteranceId(1)),
            current_samples: Vec::new(),
            current_pos: 0,
            queue: VecDeque::new(),
            timings: VecDeque::new(),
        }));
        let mut samples = [1.0_f32; 2];
        process_output_callback(&mut samples, 2, &state, &hub);
        assert!(state.try_lock().is_ok());
        assert_eq!(samples.map(f32::to_bits), [0.0_f32.to_bits(); 2]);
        drop(subscribers);
        assert_eq!(
            receiver.recv_timeout(Duration::from_secs(2)).unwrap(),
            PlaybackEvent::Finished { id: UtteranceId(1) }
        );
    }

    #[test]
    fn abandoned_port_jobs_skip_inference_and_worker_exit_wakes_waiters() {
        let calls = Arc::new(AtomicU64::new(0));
        let observed = Arc::clone(&calls);
        let (response, reply) = synthesis_response();
        drop(response);
        assert!(reply.is_cancelled());
        let stopped = Arc::new(AtomicBool::new(false));
        let sender = spawn_port_worker(
            Arc::new(SynthesizerBackend::Test(Arc::new(move || {
                observed.fetch_add(1, Ordering::SeqCst);
                Ok(vec![0.25])
            }))),
            Arc::clone(&stopped),
            2,
            1.0,
            0,
        );
        sender
            .send(PortJob {
                text: "cancelled".into(),
                sample_rate: 24_000,
                reply,
            })
            .unwrap();
        let (response, reply) = synthesis_response();
        sender
            .send(PortJob {
                text: "live".into(),
                sample_rate: 24_000,
                reply,
            })
            .unwrap();
        assert!(wait_for(response).is_ok());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        stopped.store(true, Ordering::Relaxed);
        drop(sender);

        let (response, reply) = synthesis_response();
        drop(reply);
        assert!(matches!(wait_for(response), Err(PortError::Unavailable(_))));
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
    fn callback_preserves_stereo_samples_silence_and_lifecycle_order() {
        let subscribers = Arc::new(EventHub::new());
        let rx = subscribers.subscribe().unwrap();
        let state = Arc::new(Mutex::new(PlaybackQueueState {
            generation: 1,
            current_utterance: None,
            current_samples: Vec::new(),
            current_pos: 0,
            queue: VecDeque::from([
                QueuedSentence {
                    id: UtteranceId(1),
                    generation: 0,
                    text: "stale".to_owned(),
                    samples: vec![0.9],
                },
                QueuedSentence {
                    id: UtteranceId(2),
                    generation: 1,
                    text: "first".to_owned(),
                    samples: vec![0.25, -0.5],
                },
                QueuedSentence {
                    id: UtteranceId(3),
                    generation: 1,
                    text: "second".to_owned(),
                    samples: vec![0.75],
                },
            ]),
            timings: VecDeque::new(),
        }));
        let mut data = [1.0_f32; 10];
        process_output_callback(&mut data, 2, &state, &subscribers);
        assert_eq!(
            data.map(f32::to_bits),
            [0.25, 0.25, -0.5, -0.5, 0.75, 0.75, 0.0, 0.0, 0.0, 0.0].map(f32::to_bits)
        );
        assert_eq!(
            (0..5)
                .map(|_| rx.recv_timeout(Duration::from_secs(2)).unwrap())
                .collect::<Vec<_>>(),
            vec![
                PlaybackEvent::Cancelled { id: UtteranceId(1) },
                PlaybackEvent::Started {
                    id: UtteranceId(2),
                    text: "first".to_owned()
                },
                PlaybackEvent::Finished { id: UtteranceId(2) },
                PlaybackEvent::Started {
                    id: UtteranceId(3),
                    text: "second".to_owned()
                },
                PlaybackEvent::Finished { id: UtteranceId(3) },
            ]
        );
    }

    #[test]
    fn callback_defers_finished_until_the_frame_after_the_last_sample() {
        let subscribers = Arc::new(EventHub::new());
        let rx = subscribers.subscribe().unwrap();
        let state = Arc::new(Mutex::new(PlaybackQueueState {
            generation: 0,
            current_utterance: Some(UtteranceId(1)),
            current_samples: vec![0.25],
            current_pos: 0,
            queue: VecDeque::new(),
            timings: VecDeque::new(),
        }));
        let mut data = [1.0_f32; 2];
        process_output_callback(&mut data, 2, &state, &subscribers);
        assert_eq!(data.map(f32::to_bits), [0.25_f32.to_bits(); 2]);
        assert!(rx.try_recv().is_err());
        process_output_callback(&mut data, 2, &state, &subscribers);
        assert_eq!(data.map(f32::to_bits), [0.0_f32.to_bits(); 2]);
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(2)).unwrap(),
            PlaybackEvent::Finished { id: UtteranceId(1) }
        );
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn callback_outputs_silence_without_advancing_when_queue_is_locked() {
        let subscribers = Arc::new(EventHub::new());
        let state = Arc::new(Mutex::new(PlaybackQueueState {
            generation: 0,
            current_utterance: Some(UtteranceId(1)),
            current_samples: vec![0.25],
            current_pos: 0,
            queue: VecDeque::new(),
            timings: VecDeque::new(),
        }));
        let guard = state.lock().unwrap();
        let mut data = [1.0_f32; 2];
        process_output_callback(&mut data, 2, &state, &subscribers);
        assert_eq!(guard.current_pos, 0);
        drop(guard);
        assert_eq!(data.map(f32::to_bits), [0.0_f32.to_bits(); 2]);
    }

    #[test]
    fn utterance_id_formatting_and_ordering() {
        let id1 = UtteranceId(1);
        let id2 = UtteranceId(2);
        assert_eq!(format!("{id1}"), "#1");
        assert!(id1 < id2);
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

    #[test]
    fn text_to_speech_trait_synthesizes_samples() {
        let player = VoicePlayer::mock();
        let samples = wait_for(player.synthesize("Olá Gabriel", 24_000)).unwrap();
        assert!(!samples.is_empty());
    }

    #[test]
    fn voice_config_validation_catches_missing_files() {
        let config = VoiceConfig {
            model_path: PathBuf::from("/non/existent/model.onnx"),
            voices_path: PathBuf::from("/non/existent/voices.bin"),
            tokens_path: PathBuf::from("/non/existent/tokens.txt"),
            data_dir: PathBuf::from("/non/existent/espeak-ng-data"),
            ..VoiceConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_official_kokoro_model_if_present() {
        let config = VoiceConfig::default();
        if !config.model_path.is_file() {
            return;
        }
        let tts = create_offline_tts(&config).expect("create_offline_tts should succeed");
        let gen_config = GenerationConfig {
            speed: 1.0,
            sid: config.speaker_id,
            silence_scale: 0.2,
            ..GenerationConfig::default()
        };
        let audio = tts.generate_with_config(
            "Olá Gabriel! Eu sou o Enton.",
            &gen_config,
            None::<fn(&[f32], f32) -> bool>,
        );
        assert!(audio.is_some(), "generation should succeed");
        let audio = audio.unwrap();
        assert!(!audio.samples().is_empty());
        assert_eq!(audio.sample_rate(), 24_000);
    }

    #[test]
    fn test_real_voice_player_lifecycle() {
        let config = VoiceConfig::default();
        if !config.model_path.is_file() {
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
    fn test_voice_latency_breakdown_compute() {
        let t0 = std::time::Instant::now();
        let t1_token = t0 + Duration::from_millis(120);
        let t2_chunk = t0 + Duration::from_millis(180);
        let t3_synth = t0 + Duration::from_millis(260);
        let t4_sample = t0 + Duration::from_millis(280);

        let breakdown = VoiceLatencyBreakdown::compute(
            t0,
            Some(t1_token),
            t2_chunk,
            Some(t3_synth),
            Some(t4_sample),
        );

        assert_eq!(
            breakdown.time_to_first_token,
            Some(Duration::from_millis(120))
        );
        assert_eq!(
            breakdown.time_to_first_chunk,
            Some(Duration::from_millis(60))
        );
        assert_eq!(
            breakdown.synthesis_duration,
            Some(Duration::from_millis(80))
        );
        assert_eq!(
            breakdown.audio_device_dispatch,
            Some(Duration::from_millis(20))
        );
        assert_eq!(
            breakdown.total_time_to_first_audio,
            Some(Duration::from_millis(280))
        );
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
