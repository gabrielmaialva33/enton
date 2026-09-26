//! Pure, sample-clocked endpointing and bounded raw-frame retention.

use std::collections::VecDeque;

use enton_core::SpeechCue;

use super::{AudioError, KeywordDecision};

/// Mono PCM sample rate in hertz.
pub const SAMPLE_RATE: u32 = 16_000;
/// Samples in each 32 ms processing frame.
pub const FRAME_SAMPLES: usize = 512;
/// Processing frame duration in milliseconds.
pub const FRAME_MS: u32 = 32;
const PRE_ROLL_FRAMES: u32 = 5;
const MIN_SPEECH_FRAMES: u32 = 3;

/// Completed, bounded utterance. A pending/unavailable keyword is not a negative.
#[derive(Debug, Clone)]
pub struct AudioSegment {
    /// Inclusive position in the resampled 16 kHz sample clock.
    pub start_sample: u64,
    /// Exclusive position in the resampled 16 kHz sample clock.
    pub end_sample: u64,
    /// Energy, speech duration, VAD confidence, and keyword evidence.
    pub cue: SpeechCue,
    /// Mono 16 kHz normalized PCM, including pre-roll and endpoint silence.
    pub samples: Vec<f32>,
    /// Keyword recognition state, including pending and unavailable results.
    pub keyword: KeywordDecision,
    /// Whether the maximum utterance duration ended this segment.
    pub forced_endpoint: bool,
}

/// Endpoint indices use the resampled 16 kHz sample clock, not wall time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Endpoint {
    /// Inclusive position in the resampled 16 kHz sample clock.
    pub start_sample: u64,
    /// Exclusive position in the resampled 16 kHz sample clock.
    pub end_sample: u64,
    /// Energy, speech duration, VAD confidence, and keyword evidence.
    pub cue: SpeechCue,
    /// False for speech bursts shorter than 96 ms; audio remains in retention.
    pub accepted: bool,
    /// Whether the maximum utterance duration forced this endpoint.
    pub forced: bool,
}

/// Scalar state only: no microphone, model, wall clock, or audio allocations.
#[derive(Debug, Clone)]
pub struct EndpointDetector {
    max_frames: u32,
    first_voice: Option<u64>,
    voiced_frames: u32,
    total_frames: u32,
    quiet_frames: u32,
    energy_sum: f32,
    continuing: bool,
    min_start_sample: u64,
}

impl EndpointDetector {
    /// Maximum utterance span, rounded down to 32 ms; allowed range 1 to 15 seconds.
    /// Returns a configuration error outside that range.
    pub fn new(max_segment_ms: u32) -> Result<Self, AudioError> {
        if !(1_000..=15_000).contains(&max_segment_ms) {
            return Err(AudioError::Configuration(
                "maximum segment must be 1000..=15000 ms",
            ));
        }
        Ok(Self {
            max_frames: max_segment_ms / FRAME_MS,
            first_voice: None,
            voiced_frames: 0,
            total_frames: 0,
            quiet_frames: 0,
            energy_sum: 0.0,
            continuing: false,
            min_start_sample: 0,
        })
    }

    /// Consume one 32 ms frame. `speech` is the VAD decision; `energy` is
    /// normalized RMS. Call `reset` before supplying discontinuous audio.
    pub fn push(&mut self, start: u64, speech: bool, energy: f32) -> Option<Endpoint> {
        if speech {
            self.first_voice.get_or_insert(start);
            self.voiced_frames += 1;
            self.quiet_frames = 0;
            self.energy_sum += if energy.is_finite() {
                energy.clamp(0.0, 1.0)
            } else {
                0.0
            };
        } else if self.first_voice.is_some() {
            self.quiet_frames += 1;
        } else {
            self.continuing = false;
        }
        let first = self.first_voice?;
        self.total_frames += 1;
        // Short utterances get an extra frame of silence to protect brief pauses.
        let silence_frames = if self.voiced_frames * FRAME_MS < 600 {
            8
        } else {
            7
        };
        let forced = self.total_frames >= self.max_frames;
        if !forced && self.quiet_frames < silence_frames {
            return None;
        }
        let was_continuing = self.continuing;
        let speech_span_frames = self.total_frames - self.quiet_frames;
        let raw_start = if was_continuing {
            first
        } else {
            first.saturating_sub(u64::from(PRE_ROLL_FRAMES) * FRAME_SAMPLES as u64)
        };
        let start_sample = raw_start.max(self.min_start_sample);
        let endpoint = Endpoint {
            start_sample,
            end_sample: start + FRAME_SAMPLES as u64,
            cue: SpeechCue {
                energy: self.energy_sum / self.voiced_frames as f32,
                duration_ms: speech_span_frames * FRAME_MS,
                // sherpa's safe VAD API exposes activity, not posterior probability.
                vad_confidence: self.voiced_frames as f32 / speech_span_frames as f32,
                keyword: false,
                speaker_sim: None,
                media: None,
                turn_complete: None,
                directed: None,
                direction: None,
            },
            accepted: self.voiced_frames >= MIN_SPEECH_FRAMES || was_continuing,
            forced,
        };
        if forced {
            self.first_voice = None;
            self.voiced_frames = 0;
            self.total_frames = 0;
            self.quiet_frames = 0;
            self.energy_sum = 0.0;
            self.continuing = true;
        } else {
            self.reset_at(self.min_start_sample);
        }
        Some(endpoint)
    }

    /// Discard an incomplete candidate after a capture gap or device error.
    pub fn reset(&mut self) {
        self.reset_at(0);
    }

    /// Reset candidate tracking and set a lower sample-clock boundary for pre-roll.
    pub fn reset_at(&mut self, min_start: u64) {
        self.first_voice = None;
        self.voiced_frames = 0;
        self.total_frames = 0;
        self.quiet_frames = 0;
        self.energy_sum = 0.0;
        self.continuing = false;
        self.min_start_sample = min_start;
    }

    pub(crate) fn is_active(&self) -> bool {
        self.first_voice.is_some()
    }
}

/// Retained regardless of VAD/keyword decisions or downstream queue overflow.
#[derive(Debug, Clone)]
pub struct RetainedFrame {
    /// Inclusive position in the resampled 16 kHz sample clock.
    pub start_sample: u64,
    /// One frame of normalized mono PCM at 16 kHz.
    pub samples: [f32; FRAME_SAMPLES],
    /// Whether the VAD detected speech in this frame.
    pub speech: bool,
}

/// A rolling sample-time window. Eviction does not depend on STT success.
#[derive(Debug)]
pub struct Retention {
    frames: VecDeque<RetainedFrame>,
    capacity: usize,
}

impl Retention {
    /// Allocate a rolling window of 1000..=60000 ms, rounded down to frames.
    /// Returns a configuration error for durations outside that range.
    pub fn new(retention_ms: u32) -> Result<Self, AudioError> {
        if !(1_000..=60_000).contains(&retention_ms) {
            return Err(AudioError::Configuration(
                "retention must be 1000..=60000 ms",
            ));
        }
        let capacity = (retention_ms / FRAME_MS) as usize;
        Ok(Self {
            frames: VecDeque::with_capacity(capacity),
            capacity,
        })
    }

    /// Retain a frame, evicting oldest frames by sample age or capacity.
    /// Uses storage allocated at construction; never grows the buffer.
    pub fn push(&mut self, frame: RetainedFrame) {
        let span = self.capacity as u64 * FRAME_SAMPLES as u64;
        while self
            .frames
            .front()
            .is_some_and(|old| frame.start_sample.saturating_sub(old.start_sample) >= span)
            || self.frames.len() >= self.capacity
        {
            self.frames.pop_front();
        }
        self.frames.push_back(frame);
    }

    /// Clear all retained frames after a stream gap or device error.
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// Copy the current bounded window for offline E1 inspection. Gaps remain
    /// visible in frame indices; this function never invents missing samples.
    #[must_use]
    pub fn snapshot(&self) -> Vec<RetainedFrame> {
        self.frames.iter().cloned().collect()
    }

    pub(crate) fn segment(&self, endpoint: Endpoint) -> Option<AudioSegment> {
        let front = self.frames.front()?;
        if front.start_sample > endpoint.start_sample {
            // Older frames required by this endpoint were already evicted from retention.
            // Returning None accounts for output loss rather than emitting clipped audio.
            return None;
        }
        let first = endpoint.start_sample;
        let mut next = first;
        let capacity = usize::try_from(endpoint.end_sample.saturating_sub(first)).unwrap_or(0);
        let mut samples = Vec::with_capacity(capacity);
        for frame in self
            .frames
            .iter()
            .filter(|frame| frame.start_sample >= first && frame.start_sample < endpoint.end_sample)
        {
            if frame.start_sample != next {
                return None;
            }
            samples.extend_from_slice(&frame.samples);
            next += FRAME_SAMPLES as u64;
        }
        if next != endpoint.end_sample {
            return None;
        }
        Some(AudioSegment {
            start_sample: first,
            end_sample: next,
            cue: endpoint.cue,
            samples,
            keyword: KeywordDecision::Pending,
            forced_endpoint: endpoint.forced,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(voiced: u32, silence: u32) -> Option<Endpoint> {
        let mut detector = EndpointDetector::new(8_000).unwrap();
        let mut result = None;
        for frame in 0..voiced + silence {
            result = detector.push(u64::from(frame) * 512, frame < voiced, 0.5);
        }
        result
    }

    #[test]
    fn adaptive_silence_is_224_or_256_ms() {
        assert!(run(5, 7).is_none());
        let short = run(5, 8).unwrap();
        assert!(short.accepted);
        assert_eq!(short.cue.duration_ms, 160);
        assert!(run(25, 6).is_none());
        assert!(run(25, 7).unwrap().accepted);
    }

    #[test]
    fn a_pause_does_not_split_speech_and_silence_never_emits() {
        let mut detector = EndpointDetector::new(8_000).unwrap();
        for frame in 0..50 {
            assert!(detector.push(frame * 512, false, 0.0).is_none());
        }
        for frame in 50..60 {
            assert!(detector.push(frame * 512, true, 0.5).is_none());
        }
        for frame in 60..66 {
            assert!(detector.push(frame * 512, false, 0.0).is_none());
        }
        assert!(detector.push(66 * 512, true, 0.5).is_none());
    }

    #[test]
    fn short_bursts_are_rejected_and_long_speech_is_capped() {
        assert!(!run(2, 8).unwrap().accepted);
        let mut detector = EndpointDetector::new(1_000).unwrap();
        for frame in 0..30 {
            assert!(detector.push(frame * 512, true, 0.5).is_none());
        }
        let endpoint = detector.push(30 * 512, true, 0.5).unwrap();
        assert!(endpoint.forced);
        assert_eq!(endpoint.cue.duration_ms, 992);
    }

    #[test]
    fn retention_keeps_rejected_audio_then_evicts_by_time() {
        let mut retention = Retention::new(1_000).unwrap();
        for frame in 0..100 {
            retention.push(RetainedFrame {
                start_sample: frame * 512,
                samples: [0.25; FRAME_SAMPLES],
                speech: false,
            });
        }
        let snapshot = retention.snapshot();
        assert_eq!(snapshot.len(), 31);
        assert_eq!(snapshot[0].start_sample, 69 * 512);
        retention.push(RetainedFrame {
            start_sample: 1_000_000,
            samples: [0.0; 512],
            speech: false,
        });
        assert_eq!(retention.snapshot().len(), 1);
    }

    #[test]
    fn retained_segment_includes_pre_roll_but_refuses_gaps() {
        let endpoint = run(5, 8).unwrap();
        let mut retention = Retention::new(1_000).unwrap();
        for frame in 0..13 {
            retention.push(RetainedFrame {
                start_sample: frame * 512,
                samples: [0.5; 512],
                speech: frame < 5,
            });
        }
        assert_eq!(retention.segment(endpoint).unwrap().samples.len(), 13 * 512);
        retention.frames.remove(3);
        assert!(retention.segment(endpoint).is_none());
    }

    #[test]
    fn reset_discards_partial_speech_and_parameters_are_bounded() {
        let mut detector = EndpointDetector::new(8_000).unwrap();
        detector.push(0, true, 0.5);
        detector.reset();
        for frame in 1..20 {
            assert!(detector.push(frame * 512, false, 0.0).is_none());
        }
        assert!(EndpointDetector::new(0).is_err());
        assert!(Retention::new(60_001).is_err());
    }

    #[test]
    fn f2_continuous_speech_slices_without_duplicate_preroll_or_tail_drop() {
        let mut detector = EndpointDetector::new(1_000).unwrap();
        for frame in 0..30 {
            assert!(detector.push(frame * 512, true, 0.5).is_none());
        }
        let forced = detector.push(30 * 512, true, 0.5).unwrap();
        assert!(forced.forced);
        let end1 = forced.end_sample;
        assert_eq!(end1, 31 * 512);

        // Continuation speech: speaker speaks for 2 more frames followed by silence.
        assert!(detector.push(31 * 512, true, 0.5).is_none());
        assert!(detector.push(32 * 512, true, 0.5).is_none());
        let mut second = None;
        for s in 0..8 {
            if let Some(ep) = detector.push((33 + s) * 512, false, 0.0) {
                second = Some(ep);
                break;
            }
        }
        let second = second.expect("second segment must endpoint naturally");
        assert!(!second.forced);
        assert_eq!(
            second.start_sample, end1,
            "continuation segment must not duplicate pre-roll"
        );
        assert!(
            second.accepted,
            "short tail of continuous speech must be accepted"
        );
    }

    #[test]
    fn f3_retention_segment_rejects_evicted_preroll_and_recovers_after_clear() {
        let mut retention = Retention::new(1_000).unwrap();
        for frame in 0..40 {
            retention.push(RetainedFrame {
                start_sample: frame * 512,
                samples: [0.1; 512],
                speech: true,
            });
        }
        assert_eq!(retention.frames.front().unwrap().start_sample, 9 * 512);

        let endpoint_with_evicted_onset = Endpoint {
            start_sample: 5 * 512,
            end_sample: 30 * 512,
            cue: enton_core::SpeechCue::default(),
            accepted: true,
            forced: false,
        };
        assert!(
            retention.segment(endpoint_with_evicted_onset).is_none(),
            "retention must return None when onset/pre-roll was evicted"
        );

        retention.clear();
        assert_eq!(retention.frames.len(), 0);

        let mut detector = EndpointDetector::new(8_000).unwrap();
        detector.reset_at(50_000);
        for frame in 0..5 {
            let sample_idx = 50_000 + frame * 512;
            retention.push(RetainedFrame {
                start_sample: sample_idx,
                samples: [0.5; 512],
                speech: true,
            });
            assert!(detector.push(sample_idx, true, 0.5).is_none());
        }
        let mut natural_ep = None;
        for s in 0..8 {
            let sample_idx = 50_000 + (5 + s) * 512;
            retention.push(RetainedFrame {
                start_sample: sample_idx,
                samples: [0.0; 512],
                speech: false,
            });
            if let Some(ep) = detector.push(sample_idx, false, 0.0) {
                natural_ep = Some(ep);
                break;
            }
        }
        let ep = natural_ep.expect("should endpoint");
        assert_eq!(ep.start_sample, 50_000);
        let segment = retention
            .segment(ep)
            .expect("segment must succeed without gap failure");
        assert_eq!(segment.start_sample, 50_000);
    }

    #[test]
    fn f6_retention_segment_preallocates_exact_capacity() {
        let endpoint = run(5, 8).unwrap();
        let mut retention = Retention::new(1_000).unwrap();
        for frame in 0..13 {
            retention.push(RetainedFrame {
                start_sample: frame * 512,
                samples: [0.5; 512],
                speech: frame < 5,
            });
        }
        let segment = retention.segment(endpoint).unwrap();
        let expected_samples =
            usize::try_from(endpoint.end_sample - endpoint.start_sample).unwrap();
        assert_eq!(segment.samples.len(), expected_samples);
        assert_eq!(segment.samples.capacity(), expected_samples);
    }
}
