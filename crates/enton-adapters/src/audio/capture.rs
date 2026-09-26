//! Device callbacks only downmix and enqueue fixed blocks; workers own inference.

use std::{
    fmt,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, SyncSender},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use cpal::{
    FromSample, Sample, SampleFormat, SizedSample,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use sherpa_onnx::{LinearResampler, SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

use super::{
    AudioError, AudioSegment, EndpointDetector, FRAME_SAMPLES, KeywordCommand, KeywordDecision,
    RetainedFrame, Retention, SAMPLE_RATE, model_path,
};

const INPUT_SAMPLES: usize = 1_024;
const INPUT_QUEUE: usize = 16;
const POLL: Duration = Duration::from_millis(50);

/// No models are downloaded implicitly and keyword fallback is opt-in.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Exact CPAL device ID, as returned by [`input_devices`]; None uses default.
    pub device_id: Option<String>,
    /// Path to an installed Silero VAD ONNX model.
    pub vad_model: PathBuf,
    /// Rolling raw-audio retention in milliseconds (1000..=60000).
    pub retention_ms: u32,
    /// Maximum utterance duration in milliseconds (1000..=15000).
    pub max_segment_ms: u32,
    /// Applies independently to the keyword and output segment queues (1..=8).
    pub segment_queue_capacity: usize,
    /// Optional bounded keyword worker; absent leaves decisions pending.
    pub keyword: Option<KeywordCommand>,
}

impl AudioConfig {
    #[must_use]
    /// Set a model path with 30 s retention, 8 s segments, and four queue slots.
    pub fn new(vad_model: PathBuf) -> Self {
        Self {
            device_id: None,
            vad_model,
            retention_ms: 30_000,
            max_segment_ms: 8_000,
            segment_queue_capacity: 4,
            keyword: None,
        }
    }

    fn validate(&self) -> Result<(), AudioError> {
        if !(1..=8).contains(&self.segment_queue_capacity)
            || self.retention_ms < self.max_segment_ms.saturating_add(192)
        {
            return Err(AudioError::Configuration(
                "queue must be 1..=8 and retention must cover segment plus pre-roll",
            ));
        }
        if let Some(keyword) = &self.keyword {
            keyword.validate()?;
        }
        Ok(())
    }
}

/// Explicit loss accounting. Any capture loss invalidates a lossless E1 tape.
#[derive(Debug, Default, Clone, Copy)]
pub struct AudioStats {
    /// Number of 32 ms frames submitted to VAD.
    pub frames_processed: u64,
    /// Input blocks rejected by the full or disconnected capture queue.
    pub input_blocks_dropped: u64,
    /// Device errors and failed capture workers.
    pub capture_errors: u64,
    /// Detected gaps that reset the resampler, VAD, and endpoint detector.
    pub discontinuities: u64,
    /// Speech bursts shorter than the 96 ms acceptance threshold.
    pub short_segments_dropped: u64,
    /// Segments lost to queue overflow, disconnection, or retention gaps.
    pub output_segments_dropped: u64,
    /// Segments discarded after waiting more than two seconds.
    pub stale_segments_dropped: u64,
    /// Keyword worker failures, recorded as unavailable decisions.
    pub keyword_failures: u64,
}

#[derive(Default)]
struct Counters {
    frames_processed: AtomicU64,
    input_blocks_dropped: AtomicU64,
    capture_errors: AtomicU64,
    discontinuities: AtomicU64,
    short_segments_dropped: AtomicU64,
    output_segments_dropped: AtomicU64,
    stale_segments_dropped: AtomicU64,
    keyword_failures: AtomicU64,
}

impl Counters {
    fn snapshot(&self) -> AudioStats {
        AudioStats {
            frames_processed: self.frames_processed.load(Ordering::Relaxed),
            input_blocks_dropped: self.input_blocks_dropped.load(Ordering::Relaxed),
            capture_errors: self.capture_errors.load(Ordering::Relaxed),
            discontinuities: self.discontinuities.load(Ordering::Relaxed),
            short_segments_dropped: self.short_segments_dropped.load(Ordering::Relaxed),
            output_segments_dropped: self.output_segments_dropped.load(Ordering::Relaxed),
            stale_segments_dropped: self.stale_segments_dropped.load(Ordering::Relaxed),
            keyword_failures: self.keyword_failures.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone)]
struct Shared {
    stopped: Arc<AtomicBool>,
    counters: Arc<Counters>,
    retention: Arc<Mutex<Retention>>,
}

/// Owns the device stream and two bounded processing workers. Dropping it stops
/// capture and joins workers; an active keyword process can take its timeout.
pub struct AudioCapture {
    stream: Option<cpal::Stream>,
    receiver: Receiver<AudioSegment>,
    workers: Vec<JoinHandle<()>>,
    shared: Shared,
}

impl fmt::Debug for AudioCapture {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AudioCapture")
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

/// Device IDs, not `PipeWire` node names. With the default Linux ALSA backend,
/// the `pipewire` ALSA device can be routed with `PIPEWIRE_PROPS`.
/// Returns a device error when enumeration or device metadata fails.
pub fn input_devices() -> Result<Vec<(String, String)>, AudioError> {
    cpal::default_host()
        .input_devices()
        .map_err(AudioError::Device)?
        .map(|device| {
            Ok((
                device.id().map_err(AudioError::Device)?.to_string(),
                device
                    .description()
                    .map_err(AudioError::Device)?
                    .name()
                    .to_owned(),
            ))
        })
        .collect()
}

impl AudioCapture {
    /// Validate configuration, load VAD, and start capture and keyword workers.
    /// Returns configuration, model, or device errors without downloading models.
    pub fn start(config: AudioConfig) -> Result<Self, AudioError> {
        config.validate()?;
        let endpoint = EndpointDetector::new(config.max_segment_ms)?;
        let shared = Shared {
            stopped: Arc::new(AtomicBool::new(false)),
            counters: Arc::new(Counters::default()),
            retention: Arc::new(Mutex::new(Retention::new(config.retention_ms)?)),
        };
        let vad = make_vad(&config)?;
        let host = cpal::default_host();
        let device = if let Some(id) = &config.device_id {
            host.device_by_id(&id.parse().map_err(AudioError::Device)?)
        } else {
            host.default_input_device()
        }
        .ok_or(AudioError::NoInputDevice)?;
        let supported = device.default_input_config().map_err(AudioError::Device)?;
        let rate = supported.sample_rate();
        let channels = supported.channels();
        if !(8_000..=192_000).contains(&rate) || !(1..=32).contains(&channels) {
            return Err(AudioError::Configuration(
                "input must have 1..=32 channels at 8000..=192000 Hz",
            ));
        }
        let resampler = LinearResampler::create(i32::try_from(rate).unwrap_or(16_000), 16_000)
            .ok_or(AudioError::Native("could not create audio resampler"))?;
        let ring_buffer = Arc::new(SpscRingBuffer::new(INPUT_QUEUE));
        let (segment_tx, segment_rx) = mpsc::sync_channel(config.segment_queue_capacity);
        let (output_tx, output_rx) = mpsc::sync_channel(config.segment_queue_capacity);
        let stream = build_stream(
            &device,
            &supported,
            Arc::clone(&ring_buffer),
            Arc::clone(&shared.counters),
        )?;
        let processing = shared.clone();
        let ring_capture = Arc::clone(&ring_buffer);
        let capture_worker = thread::spawn(move || {
            capture_loop(
                &ring_capture,
                &segment_tx,
                &vad,
                &resampler,
                endpoint,
                rate,
                &processing,
            );
        });
        let processing = shared.clone();
        let keyword_worker = thread::spawn(move || {
            keyword_loop(
                &segment_rx,
                &output_tx,
                config.keyword.as_ref(),
                &processing,
            );
        });
        let capture = Self {
            stream: Some(stream),
            receiver: output_rx,
            workers: vec![capture_worker, keyword_worker],
            shared,
        };
        if let Some(stream) = &capture.stream {
            stream.play().map_err(AudioError::Device)?;
        }
        Ok(capture)
    }

    /// Wait at most `timeout` for a completed segment.
    /// Returns timeout or disconnection when no segment can be received.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<AudioSegment, RecvTimeoutError> {
        self.receiver.recv_timeout(timeout)
    }

    #[must_use]
    /// Snapshot cumulative processing and loss counters without resetting them.
    pub fn stats(&self) -> AudioStats {
        self.shared.counters.snapshot()
    }

    /// In-memory audit window, including noise and rejected/undelivered segments.
    /// The caller owns and must bound any snapshots it retains.
    #[must_use]
    pub fn retained_audio(&self) -> Vec<RetainedFrame> {
        self.shared
            .retention
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .snapshot()
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        self.shared.stopped.store(true, Ordering::Relaxed);
        self.stream.take();
        for worker in self.workers.drain(..) {
            if worker.join().is_err() {
                self.shared
                    .counters
                    .capture_errors
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

fn make_vad(config: &AudioConfig) -> Result<VoiceActivityDetector, AudioError> {
    let model = VadModelConfig {
        silero_vad: SileroVadModelConfig {
            model: Some(model_path(&config.vad_model)?),
            threshold: 0.5,
            // Keep native hangover minimal; the pure endpoint detector owns it.
            min_silence_duration: 0.001,
            min_speech_duration: 0.001,
            window_size: 512,
            max_speech_duration: 20.0,
        },
        sample_rate: 16_000,
        num_threads: 1,
        provider: Some("cpu".to_owned()),
        ..Default::default()
    };
    VoiceActivityDetector::create(&model, 20.0)
        .ok_or(AudioError::Native("could not load Silero VAD model"))
}

struct InputSlot {
    first_sample: AtomicU64,
    generation: AtomicU64,
    samples: [AtomicU32; INPUT_SAMPLES],
}

impl InputSlot {
    fn new() -> Self {
        Self {
            first_sample: AtomicU64::new(0),
            generation: AtomicU64::new(0),
            samples: std::array::from_fn(|_| AtomicU32::new(0)),
        }
    }
}

pub(crate) struct SpscRingBuffer {
    slots: Vec<InputSlot>,
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl SpscRingBuffer {
    pub(crate) fn new(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(InputSlot::new());
        }
        Self {
            slots,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    pub(crate) fn try_push(
        &self,
        first_sample: u64,
        generation: u64,
        samples: &[f32; INPUT_SAMPLES],
    ) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head.wrapping_sub(tail) >= self.slots.len() {
            return false;
        }
        let slot_idx = head % self.slots.len();
        if let Some(slot) = self.slots.get(slot_idx) {
            slot.first_sample.store(first_sample, Ordering::Relaxed);
            slot.generation.store(generation, Ordering::Relaxed);
            for (idx, sample) in samples.iter().enumerate() {
                if let Some(target) = slot.samples.get(idx) {
                    target.store(sample.to_bits(), Ordering::Relaxed);
                }
            }
            self.head.store(head.wrapping_add(1), Ordering::Release);
            true
        } else {
            false
        }
    }

    pub(crate) fn try_pop(&self, out_samples: &mut [f32; INPUT_SAMPLES]) -> Option<(u64, u64)> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);
        if tail == head {
            return None;
        }
        let slot_idx = tail % self.slots.len();
        let slot = self.slots.get(slot_idx)?;
        let first_sample = slot.first_sample.load(Ordering::Relaxed);
        let generation = slot.generation.load(Ordering::Relaxed);
        for (idx, out) in out_samples.iter_mut().enumerate() {
            if let Some(source) = slot.samples.get(idx) {
                *out = f32::from_bits(source.load(Ordering::Relaxed));
            }
        }
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Some((first_sample, generation))
    }
}

fn build_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    ring_buffer: Arc<SpscRingBuffer>,
    counters: Arc<Counters>,
) -> Result<cpal::Stream, AudioError> {
    match config.sample_format() {
        SampleFormat::I8 => typed_stream::<i8>(device, config, ring_buffer, counters),
        SampleFormat::I16 => typed_stream::<i16>(device, config, ring_buffer, counters),
        SampleFormat::I32 => typed_stream::<i32>(device, config, ring_buffer, counters),
        SampleFormat::I64 => typed_stream::<i64>(device, config, ring_buffer, counters),
        SampleFormat::U8 => typed_stream::<u8>(device, config, ring_buffer, counters),
        SampleFormat::U16 => typed_stream::<u16>(device, config, ring_buffer, counters),
        SampleFormat::U32 => typed_stream::<u32>(device, config, ring_buffer, counters),
        SampleFormat::U64 => typed_stream::<u64>(device, config, ring_buffer, counters),
        SampleFormat::F32 => typed_stream::<f32>(device, config, ring_buffer, counters),
        SampleFormat::F64 => typed_stream::<f64>(device, config, ring_buffer, counters),
        format => Err(AudioError::SampleFormat(format)),
    }
}

fn typed_stream<T>(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    ring_buffer: Arc<SpscRingBuffer>,
    counters: Arc<Counters>,
) -> Result<cpal::Stream, AudioError>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = usize::from(config.channels());
    let errors = Arc::clone(&counters);
    let mut samples = [0.0; INPUT_SAMPLES];
    let mut length = 0;
    let mut position = 0;
    let mut generation = 0;
    device
        .build_input_stream(
            (*config).into(),
            move |data: &[T], _| {
                let current = counters.capture_errors.load(Ordering::Relaxed);
                if current != generation {
                    length = 0;
                    generation = current;
                }
                for frame in data.chunks_exact(channels) {
                    let Some(slot) = samples.get_mut(length) else {
                        counters.capture_errors.fetch_add(1, Ordering::Relaxed);
                        length = 0;
                        continue;
                    };
                    *slot = downmix(frame);
                    length += 1;
                    position += 1;
                    if length == INPUT_SAMPLES {
                        let first_sample = position - INPUT_SAMPLES as u64;
                        if !ring_buffer.try_push(first_sample, generation, &samples) {
                            counters
                                .input_blocks_dropped
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        length = 0;
                    }
                }
            },
            move |_| {
                errors.capture_errors.fetch_add(1, Ordering::Relaxed);
            },
            Some(Duration::from_secs(1)),
        )
        .map_err(AudioError::Device)
}

fn downmix<T: Sample>(frame: &[T]) -> f32
where
    f32: FromSample<T>,
{
    let sum: f32 = frame
        .iter()
        .map(|value| {
            let value = f32::from_sample(*value);
            if value.is_finite() {
                value.clamp(-1.0, 1.0)
            } else {
                0.0
            }
        })
        .sum();
    sum / frame.len() as f32
}

struct QueuedSegment {
    queued_at: Instant,
    segment: AudioSegment,
}

fn frame_energy(frame: &[f32; FRAME_SAMPLES]) -> f32 {
    let rms = (frame.iter().map(|value| value * value).sum::<f32>() / FRAME_SAMPLES as f32).sqrt();
    (rms / 0.1).min(1.0)
}

fn process_completed_frame(
    frame: &[f32; FRAME_SAMPLES],
    next_output: u64,
    vad: &VoiceActivityDetector,
    endpoint: &mut EndpointDetector,
    sender: &SyncSender<QueuedSegment>,
    shared: &Shared,
    native_frames: &mut u32,
) {
    vad.accept_waveform(frame);
    shared
        .counters
        .frames_processed
        .fetch_add(1, Ordering::Relaxed);
    let speech = vad.detected();
    vad.clear();
    let outcome = endpoint.push(next_output, speech, frame_energy(frame));
    let mut retention = shared
        .retention
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    retention.push(RetainedFrame {
        start_sample: next_output,
        samples: *frame,
        speech,
    });
    let segment = outcome
        .filter(|ep| ep.accepted)
        .and_then(|ep| retention.segment(ep));
    drop(retention);
    if let Some(outcome) = outcome {
        if !outcome.accepted {
            shared
                .counters
                .short_segments_dropped
                .fetch_add(1, Ordering::Relaxed);
        } else if let Some(segment) = segment {
            if sender
                .try_send(QueuedSegment {
                    queued_at: Instant::now(),
                    segment,
                })
                .is_err()
            {
                shared
                    .counters
                    .output_segments_dropped
                    .fetch_add(1, Ordering::Relaxed);
            }
        } else {
            shared
                .counters
                .output_segments_dropped
                .fetch_add(1, Ordering::Relaxed);
        }
        if !outcome.forced {
            vad.reset();
            *native_frames = 0;
        }
    }
    *native_frames += 1;
    if *native_frames >= 1_875 && !speech && !endpoint.is_active() {
        vad.reset();
        *native_frames = 0;
    }
}

fn capture_loop(
    ring_buffer: &Arc<SpscRingBuffer>,
    sender: &SyncSender<QueuedSegment>,
    vad: &VoiceActivityDetector,
    resampler: &LinearResampler,
    mut endpoint: EndpointDetector,
    input_rate: u32,
    shared: &Shared,
) {
    let mut block_samples = [0.0; INPUT_SAMPLES];
    let mut frame = [0.0; FRAME_SAMPLES];
    let mut length = 0;
    let mut next_input = 0;
    let mut next_output = 0;
    let mut generation = 0;
    let mut native_frames = 0;
    while !shared.stopped.load(Ordering::Relaxed) {
        let Some((first_sample, block_generation)) = ring_buffer.try_pop(&mut block_samples) else {
            thread::sleep(Duration::from_millis(5));
            continue;
        };
        if first_sample != next_input || block_generation != generation {
            shared
                .counters
                .discontinuities
                .fetch_add(1, Ordering::Relaxed);
            next_output =
                first_sample.saturating_mul(u64::from(SAMPLE_RATE)) / u64::from(input_rate);
            endpoint.reset_at(next_output);
            vad.reset();
            resampler.reset();
            length = 0;
            shared
                .retention
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clear();
        }
        generation = block_generation;
        next_input = first_sample + INPUT_SAMPLES as u64;
        let converted = resampler.resample(&block_samples, false);
        for sample in converted {
            let Some(slot) = frame.get_mut(length) else {
                shared
                    .counters
                    .capture_errors
                    .fetch_add(1, Ordering::Relaxed);
                length = 0;
                continue;
            };
            *slot = sample;
            length += 1;
            if length != FRAME_SAMPLES {
                continue;
            }
            process_completed_frame(
                &frame,
                next_output,
                vad,
                &mut endpoint,
                sender,
                shared,
                &mut native_frames,
            );
            next_output += FRAME_SAMPLES as u64;
            length = 0;
        }
    }
}

fn keyword_loop(
    receiver: &Receiver<QueuedSegment>,
    sender: &SyncSender<AudioSegment>,
    keyword: Option<&KeywordCommand>,
    shared: &Shared,
) {
    while !shared.stopped.load(Ordering::Relaxed) {
        let mut queued = match receiver.recv_timeout(POLL) {
            Ok(segment) => segment,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        if queued.queued_at.elapsed() > Duration::from_secs(2) {
            shared
                .counters
                .stale_segments_dropped
                .fetch_add(1, Ordering::Relaxed);
            continue;
        }
        if let Some(keyword) = keyword {
            queued.segment.keyword = keyword
                .recognize_cancellable(&queued.segment.samples, &shared.stopped)
                .unwrap_or_else(|_| {
                    shared
                        .counters
                        .keyword_failures
                        .fetch_add(1, Ordering::Relaxed);
                    KeywordDecision::Unavailable
                });
            queued.segment.cue.keyword = queued.segment.keyword == KeywordDecision::Matched;
        }
        if sender.try_send(queued.segment).is_err() {
            shared
                .counters
                .output_segments_dropped
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_normalizes_formats_and_sanitizes_bad_samples() {
        assert!((downmix(&[1.0_f32, -1.0]) - 0.0).abs() < f32::EPSILON);
        assert!((downmix(&[i16::MAX, i16::MAX]) - 1.0).abs() < 0.0001);
        assert!(downmix(&[f32::NAN, f32::INFINITY]).abs() < f32::EPSILON);
    }

    #[test]
    fn configuration_cannot_hide_unbounded_buffers() {
        let mut config = AudioConfig::new(PathBuf::from("unused"));
        assert!(config.validate().is_ok());
        config.segment_queue_capacity = 100;
        assert!(config.validate().is_err());
        config.segment_queue_capacity = 1;
        config.retention_ms = 1_000;
        assert!(config.validate().is_err());
    }

    #[test]
    fn resampler_preserves_duration_across_chunk_boundaries() {
        let resampler = LinearResampler::create(48_000, 16_000).unwrap();
        let mut samples = Vec::new();
        for _ in 0..48 {
            samples.extend(resampler.resample(&[0.5; 1_000], false));
        }
        samples.extend(resampler.resample(&[], true));
        assert_eq!(samples.len(), 16_000);
        assert!((samples[8_000] - 0.5).abs() < 0.001);
    }

    #[test]
    fn full_output_queue_drops_without_blocking_or_erasing_retention() {
        let shared = Shared {
            stopped: Arc::new(AtomicBool::new(false)),
            counters: Arc::new(Counters::default()),
            retention: Arc::new(Mutex::new(Retention::new(1_000).unwrap())),
        };
        shared.retention.lock().unwrap().push(RetainedFrame {
            start_sample: 0,
            samples: [0.5; 512],
            speech: true,
        });
        let (input, receiver) = mpsc::sync_channel(4);
        let (output, emitted) = mpsc::sync_channel(1);
        for _ in 0..4 {
            input
                .send(QueuedSegment {
                    queued_at: Instant::now(),
                    segment: AudioSegment {
                        start_sample: 0,
                        end_sample: 512,
                        cue: enton_core::SpeechCue::default(),
                        samples: vec![0.5; 512],
                        keyword: KeywordDecision::Pending,
                        forced_endpoint: false,
                    },
                })
                .unwrap();
        }
        drop(input);
        keyword_loop(&receiver, &output, None, &shared);
        assert_eq!(shared.counters.snapshot().output_segments_dropped, 3);
        assert_eq!(
            emitted.try_recv().unwrap().keyword,
            KeywordDecision::Pending
        );
        assert_eq!(shared.retention.lock().unwrap().snapshot().len(), 1);
    }

    #[test]
    #[ignore = "requires the Enton audio model cache; set ENTON_AUDIO_MODELS"]
    fn native_silero_keeps_a_minute_of_silence_quiet() {
        let directory =
            PathBuf::from(std::env::var_os("ENTON_AUDIO_MODELS").expect("set ENTON_AUDIO_MODELS"));
        let vad = make_vad(&AudioConfig::new(directory.join("silero_vad.onnx"))).unwrap();
        for _ in 0..1_875 {
            vad.accept_waveform(&[0.0; 512]);
            assert!(!vad.detected());
            vad.clear();
        }
    }

    #[test]
    fn f7_spsc_ring_buffer_lock_free_and_bit_exact() {
        let ring = Arc::new(SpscRingBuffer::new(4));
        let ring_producer = Arc::clone(&ring);

        // 1. Bit exact fidelity test
        let mut original_samples = [0.0; INPUT_SAMPLES];
        for (i, slot) in original_samples.iter_mut().enumerate() {
            *slot = (i as f32) * 0.001 - 0.5;
        }
        assert!(ring.try_push(1234, 1, &original_samples));

        let mut popped = [0.0; INPUT_SAMPLES];
        let (first, generation) = ring.try_pop(&mut popped).expect("should pop");
        assert_eq!(first, 1234);
        assert_eq!(generation, 1);
        for (orig, pop) in original_samples.iter().zip(popped.iter()) {
            assert_eq!(orig.to_bits(), pop.to_bits());
        }

        // 2. Capacity & overflow test (ring size 4)
        for i in 0..4 {
            assert!(ring.try_push(i * 1024, 1, &[0.0; INPUT_SAMPLES]));
        }
        // 5th push must fail wait-free (buffer full)
        assert!(!ring.try_push(4096, 1, &[0.0; INPUT_SAMPLES]));

        // Drain 4 items
        for i in 0..4 {
            let (first, _) = ring.try_pop(&mut popped).expect("should pop");
            assert_eq!(first, i * 1024);
        }
        // Next pop must be empty
        assert!(ring.try_pop(&mut popped).is_none());

        // 3. Multi-threaded producer-consumer test
        let handle = thread::spawn(move || {
            for i in 0..100 {
                let mut data = [0.0; INPUT_SAMPLES];
                data[0] = i as f32;
                while !ring_producer.try_push(i * 1024, 1, &data) {
                    thread::yield_now();
                }
            }
        });

        for i in 0..100 {
            loop {
                if let Some((first, _)) = ring.try_pop(&mut popped) {
                    assert_eq!(first, i * 1024);
                    assert!((popped[0] - i as f32).abs() < f32::EPSILON);
                    break;
                }
                thread::yield_now();
            }
        }
        handle.join().unwrap();
    }
}
