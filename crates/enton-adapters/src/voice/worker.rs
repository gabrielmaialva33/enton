use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use enton_core::UtteranceId;
use enton_core::ports::PortError;
use sherpa_onnx::{
    GenerationConfig, LinearResampler, OfflineTts, OfflineTtsConfig, OfflineTtsKokoroModelConfig,
    OfflineTtsModelConfig,
};

use super::config::{VoiceConfig, VoiceError};
use super::events::{EventHub, PlaybackEvent, emit_event};
use super::playback::{PlaybackQueueState, QueuedSentence};

pub(super) enum SynthesizerBackend {
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

pub(super) struct SynthesisResponse(Arc<Mutex<ResponseState>>);
pub(super) struct SynthesisReply(Option<Arc<Mutex<ResponseState>>>);

pub(super) fn synthesis_response() -> (SynthesisResponse, SynthesisReply) {
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
    pub(super) fn is_cancelled(&self) -> bool {
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

pub(super) struct PortJob {
    pub(super) text: String,
    pub(super) sample_rate: u32,
    pub(super) reply: SynthesisReply,
}

pub(super) fn spawn_port_worker(
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

pub(super) struct SynthesisJob {
    pub(super) id: UtteranceId,
    pub(super) generation: u64,
    pub(super) text: String,
}

pub(super) fn create_offline_tts(config: &VoiceConfig) -> Result<OfflineTts, VoiceError> {
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

pub(super) struct SynthesisWorkerParams {
    pub(super) job_rx: Receiver<SynthesisJob>,
    pub(super) synthesizer: Arc<SynthesizerBackend>,
    pub(super) last_error: Arc<Mutex<Option<VoiceError>>>,
    pub(super) queue_state: Arc<Mutex<PlaybackQueueState>>,
    pub(super) generation: Arc<std::sync::atomic::AtomicU64>,
    pub(super) subscribers: Arc<EventHub>,
    pub(super) stopped: Arc<AtomicBool>,
    pub(super) device_sample_rate: u32,
    pub(super) speed: f32,
    pub(super) speaker_id: i32,
    pub(super) queue_capacity: usize,
}

pub(super) fn spawn_synthesis_worker(params: SynthesisWorkerParams) -> JoinHandle<()> {
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

pub(super) fn spawn_mock_worker(
    job_rx: Receiver<SynthesisJob>,
    queue_state: Arc<Mutex<PlaybackQueueState>>,
    generation: Arc<std::sync::atomic::AtomicU64>,
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
    fn abandoned_port_jobs_skip_inference_and_worker_exit_wakes_waiters() {
        let calls = Arc::new(std::sync::atomic::AtomicU64::new(0));
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
}
