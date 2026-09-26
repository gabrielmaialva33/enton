//! Protocol 2.0.0 populations. Deliberately does not import the cognitive Profile.

use crate::{
    Annotation, EpisodeId, Error, Interval, Record, SegmentId, Stimulus, Tape, TapeKind, Turn,
    TurnId,
};
use enton_core::{Event, Millis, SpeechCue};

/// Vigna's reference `SplitMix64`: fixed-increment state and mixed output.
/// Reference: <https://prng.di.unimi.it/splitmix64.c> (public domain).
#[derive(Debug, Clone, Copy)]
pub struct SplitMix64(u64);
impl SplitMix64 {
    /// Initialize the 64-bit state with the supplied seed.
    #[must_use]
    pub const fn with_seed(seed: u64) -> Self {
        Self(seed)
    }
    /// Advance the state and return the mixed 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn range(&mut self, low: u32, high: u32) -> u32 {
        let width = u64::from(high - low) + 1;
        let reject_below = width.wrapping_neg() % width;
        loop {
            let value = self.next_u64();
            if value >= reject_below {
                // The remainder is at most high-low, which fits u32.
                return low + u32::try_from(value % width).unwrap_or(0);
            }
        }
    }
    fn real(&mut self, low: f32, high: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / 16_777_216.0;
        low + (high - low) * unit
    }
}

/// Lower bound of the nominal speaker similarity range for legitimate user speech.
pub const USER_SPEAKER_SIM_MIN: f32 = 0.62;
/// Upper bound of the nominal speaker similarity range for legitimate user speech.
pub const USER_SPEAKER_SIM_MAX: f32 = 0.95;

/// Lower bound of the nominal speaker similarity range for impostor speech.
pub const IMPOSTOR_SPEAKER_SIM_MIN: f32 = 0.05;
/// Upper bound of the nominal speaker similarity range for impostor speech.
pub const IMPOSTOR_SPEAKER_SIM_MAX: f32 = 0.55;

/// Lower bound of the speaker similarity range for non-speech acoustic segments.
pub const NON_SPEECH_SPEAKER_SIM_MIN: f32 = 0.0;
/// Upper bound of the speaker similarity range for non-speech acoustic segments.
pub const NON_SPEECH_SPEAKER_SIM_MAX: f32 = 0.3;

/// Probability of crossover error in speaker verification (bad audio for user, confusable voices for impostor).
pub const SPEAKER_SIM_CROSSOVER_RATE: f32 = 0.03;

/// Stream salt for the independent speaker-similarity RNG stream.
const SPEAKER_STREAM_SALT: u64 = 0x5350_4541_4b45_5231;

/// Lower bound of the nominal media likelihood range for television and reproduced media.
pub const TV_MEDIA_MIN: f32 = 0.6;
/// Upper bound of the nominal media likelihood range for television and reproduced media.
pub const TV_MEDIA_MAX: f32 = 0.98;

/// Probability that the audio tagger misses a TV segment and tags it within the live range.
pub const TV_MEDIA_MISS_RATE: f32 = 0.05;

/// Lower bound of the media likelihood range for live voices.
pub const LIVE_VOICE_MEDIA_MIN: f32 = 0.0;
/// Upper bound of the media likelihood range for live voices.
pub const LIVE_VOICE_MEDIA_MAX: f32 = 0.35;

/// Probability of a false alarm where a live voice is tagged in the media range.
pub const LIVE_VOICE_MEDIA_FALSE_ALARM_RATE: f32 = 0.03;

/// Lower bound of the media likelihood range for non-speech acoustic segments.
pub const NON_SPEECH_MEDIA_MIN: f32 = 0.0;
/// Upper bound of the media likelihood range for non-speech acoustic segments.
pub const NON_SPEECH_MEDIA_MAX: f32 = 0.3;

/// Stream salt for the independent media-tagger RNG stream.
const MEDIA_STREAM_SALT: u64 = 0x4d45_4449_415f_5331;

/// Lower bound of the nominal turn-completion score for complete user turns.
pub const COMPLETE_USER_TURN_MIN: f32 = 0.6;
/// Upper bound of the nominal turn-completion score for complete user turns.
pub const COMPLETE_USER_TURN_MAX: f32 = 0.99;
/// Probability that an acoustic end-of-turn model misses a complete user turn.
pub const COMPLETE_USER_TURN_MISS_RATE: f32 = 0.02;

/// Lower bound of the nominal turn-completion score for incomplete user segments.
pub const INCOMPLETE_USER_TURN_MIN: f32 = 0.01;
/// Upper bound of the nominal turn-completion score for incomplete user segments.
pub const INCOMPLETE_USER_TURN_MAX: f32 = 0.4;
/// Probability that an acoustic end-of-turn model prematurely signals completion on an incomplete turn.
pub const INCOMPLETE_USER_TURN_PREMATURE_RATE: f32 = 0.03;

/// Lower bound of the turn-completion score for non-user acoustic segments.
pub const NON_USER_TURN_MIN: f32 = 0.2;
/// Upper bound of the turn-completion score for non-user acoustic segments.
pub const NON_USER_TURN_MAX: f32 = 0.9;

/// Stream salt for the independent turn-completion RNG stream.
const TURN_STREAM_SALT: u64 = 0x5455_524e_5f53_3130;

/// Synthesized turn role classifying segment completion expectations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnRole {
    /// Incomplete user turn segment preceding a pause before the final segment.
    Incomplete,
    /// Final or only segment of a user turn.
    Complete,
    /// Non-user audio (TV, background noise, or other talkers) without turn structure.
    Other,
}

struct Builder {
    rng: SplitMix64,
    speaker_rng: SplitMix64,
    media_rng: SplitMix64,
    turn_rng: SplitMix64,
    records: Vec<Record>,
    turns: Vec<Turn>,
    next_segment: u32,
    next_turn: u32,
}
impl Builder {
    fn new(seed: u64) -> Self {
        Self {
            rng: SplitMix64::with_seed(seed),
            speaker_rng: SplitMix64::with_seed(seed ^ SPEAKER_STREAM_SALT),
            media_rng: SplitMix64::with_seed(seed ^ MEDIA_STREAM_SALT),
            turn_rng: SplitMix64::with_seed(seed ^ TURN_STREAM_SALT),
            records: Vec::new(),
            turns: Vec::new(),
            next_segment: 0,
            next_turn: 0,
        }
    }
    fn draw_speaker_sim(&mut self, source: Stimulus) -> f32 {
        match source {
            Stimulus::Request(_) | Stimulus::BargeIn(_) => {
                if self.speaker_rng.real(0.0, 1.0) < SPEAKER_SIM_CROSSOVER_RATE {
                    self.speaker_rng
                        .real(IMPOSTOR_SPEAKER_SIM_MIN, IMPOSTOR_SPEAKER_SIM_MAX)
                } else {
                    self.speaker_rng
                        .real(USER_SPEAKER_SIM_MIN, USER_SPEAKER_SIM_MAX)
                }
            }
            Stimulus::Tv | Stimulus::OtherSpeech | Stimulus::FalseKeyword => {
                if self.speaker_rng.real(0.0, 1.0) < SPEAKER_SIM_CROSSOVER_RATE {
                    self.speaker_rng
                        .real(USER_SPEAKER_SIM_MIN, USER_SPEAKER_SIM_MAX)
                } else {
                    self.speaker_rng
                        .real(IMPOSTOR_SPEAKER_SIM_MIN, IMPOSTOR_SPEAKER_SIM_MAX)
                }
            }
            Stimulus::Noise | Stimulus::Motor | Stimulus::Ventilation | Stimulus::SelfEcho => self
                .speaker_rng
                .real(NON_SPEECH_SPEAKER_SIM_MIN, NON_SPEECH_SPEAKER_SIM_MAX),
        }
    }
    fn draw_media(&mut self, source: Stimulus) -> f32 {
        match source {
            Stimulus::Tv => {
                if self.media_rng.real(0.0, 1.0) < TV_MEDIA_MISS_RATE {
                    self.media_rng
                        .real(LIVE_VOICE_MEDIA_MIN, LIVE_VOICE_MEDIA_MAX)
                } else {
                    self.media_rng.real(TV_MEDIA_MIN, TV_MEDIA_MAX)
                }
            }
            Stimulus::Request(_)
            | Stimulus::BargeIn(_)
            | Stimulus::OtherSpeech
            | Stimulus::FalseKeyword => {
                if self.media_rng.real(0.0, 1.0) < LIVE_VOICE_MEDIA_FALSE_ALARM_RATE {
                    self.media_rng.real(TV_MEDIA_MIN, TV_MEDIA_MAX)
                } else {
                    self.media_rng
                        .real(LIVE_VOICE_MEDIA_MIN, LIVE_VOICE_MEDIA_MAX)
                }
            }
            Stimulus::Noise | Stimulus::Motor | Stimulus::Ventilation | Stimulus::SelfEcho => self
                .media_rng
                .real(NON_SPEECH_MEDIA_MIN, NON_SPEECH_MEDIA_MAX),
        }
    }
    fn draw_turn_complete(&mut self, role: TurnRole) -> f32 {
        match role {
            TurnRole::Complete => {
                if self.turn_rng.real(0.0, 1.0) < COMPLETE_USER_TURN_MISS_RATE {
                    self.turn_rng
                        .real(INCOMPLETE_USER_TURN_MIN, INCOMPLETE_USER_TURN_MAX)
                } else {
                    self.turn_rng
                        .real(COMPLETE_USER_TURN_MIN, COMPLETE_USER_TURN_MAX)
                }
            }
            TurnRole::Incomplete => {
                if self.turn_rng.real(0.0, 1.0) < INCOMPLETE_USER_TURN_PREMATURE_RATE {
                    self.turn_rng
                        .real(COMPLETE_USER_TURN_MIN, COMPLETE_USER_TURN_MAX)
                } else {
                    self.turn_rng
                        .real(INCOMPLETE_USER_TURN_MIN, INCOMPLETE_USER_TURN_MAX)
                }
            }
            TurnRole::Other => self.turn_rng.real(NON_USER_TURN_MIN, NON_USER_TURN_MAX),
        }
    }
    fn segment(
        &mut self,
        end: u64,
        mut cue: SpeechCue,
        episode: EpisodeId,
        source: Stimulus,
        role: TurnRole,
    ) -> SegmentId {
        cue.speaker_sim = Some(self.draw_speaker_sim(source));
        cue.media = Some(self.draw_media(source));
        cue.turn_complete = Some(self.draw_turn_complete(role));
        let id = SegmentId(self.next_segment);
        // All generator loops are protocol-bounded to fewer than MAX_EVENTS entries.
        self.next_segment += 1;
        self.records.push(Record {
            event: Event::Speech {
                now: Millis(end),
                cue,
            },
            annotation: Annotation::Speech {
                segment: id,
                episode: Some(episode),
                source,
            },
        });
        id
    }
    fn user_cue(&mut self, duration_ms: u32, keyword: bool) -> SpeechCue {
        SpeechCue {
            duration_ms,
            keyword,
            energy: self.rng.real(0.65, 1.0),
            vad_confidence: self.rng.real(0.65, 1.0),
            speaker_sim: None,
            media: None,
            turn_complete: None,
        }
    }
    fn request(
        &mut self,
        start: u64,
        episode: EpisodeId,
        duration: u32,
        keyword: bool,
        barge_in: bool,
        meta: (crate::tape::TurnKind, Option<u32>),
    ) -> u64 {
        let (kind, gap) = meta;
        let id = TurnId(self.next_turn);
        self.next_turn += 1;
        let end = start + u64::from(duration);
        let cue = self.user_cue(duration, keyword);
        let source = if barge_in {
            Stimulus::BargeIn(id)
        } else {
            Stimulus::Request(id)
        };
        let segment = self.segment(end, cue, episode, source, TurnRole::Complete);
        self.turns.push(Turn {
            id,
            episode,
            segments: vec![segment],
            available_at: Millis(end),
            deadline: Millis(end + 10_000),
            kind,
            gap,
        });
        end
    }
    fn split(&mut self, start: u64, episode: EpisodeId) -> u64 {
        let id = TurnId(self.next_turn);
        self.next_turn += 1;
        let name_duration = self.rng.range(250, 650);
        let name_end = start + u64::from(name_duration);
        let name = self.user_cue(name_duration, true);
        let first = self.segment(
            name_end,
            name,
            episode,
            Stimulus::Request(id),
            TurnRole::Incomplete,
        );
        let pause = self.rng.range(150, 700);
        let duration = self.rng.range(700, 1600);
        let end = name_end + u64::from(pause) + u64::from(duration);
        let cue = self.user_cue(duration, false);
        let second = self.segment(end, cue, episode, Stimulus::Request(id), TurnRole::Complete);
        self.turns.push(Turn {
            id,
            episode,
            segments: vec![first, second],
            available_at: Millis(end),
            deadline: Millis(end + 10_000),
            kind: crate::tape::TurnKind::Split,
            gap: None,
        });
        name_end
    }
    fn conversation(&mut self, start: u64, episode: EpisodeId, block: u32) {
        const GAPS: [u32; 7] = [1500, 3000, 4900, 5000, 5100, 9000, 12_000];
        let duration = self.rng.range(1000, 1800);
        let mut end = self.request(
            start,
            episode,
            duration,
            true,
            false,
            (crate::tape::TurnKind::Conversation, None),
        );
        for turn in 0..4 {
            let gap = GAPS
                .get(((block * 4 + turn) % 7) as usize)
                .copied()
                .unwrap_or(1500);
            let jitter = self.rng.range(0, 200);
            let next_end = end + u64::from(gap - 100 + jitter);
            let duration = self.rng.range(600, 1200);
            end = self.request(
                next_end - u64::from(duration),
                episode,
                duration,
                false,
                false,
                (crate::tape::TurnKind::Conversation, Some(gap)),
            );
        }
    }
    fn other(&mut self, end: u64, episode: EpisodeId, duration: u32) {
        let cue = SpeechCue {
            duration_ms: duration,
            keyword: false,
            energy: self.rng.real(0.55, 0.95),
            vad_confidence: self.rng.real(0.65, 1.0),
            speaker_sim: None,
            media: None,
            turn_complete: None,
        };
        self.segment(end, cue, episode, Stimulus::OtherSpeech, TurnRole::Other);
    }
    fn noise(&mut self, start: u64, episode: EpisodeId) {
        let duration_ms = self.rng.range(250, 1000);
        let cue = SpeechCue {
            duration_ms,
            keyword: false,
            energy: self.rng.real(0.1, 0.8),
            vad_confidence: self.rng.real(0.02, 0.35),
            speaker_sim: None,
            media: None,
            turn_complete: None,
        };
        self.segment(
            start + u64::from(duration_ms),
            cue,
            episode,
            Stimulus::Noise,
            TurnRole::Other,
        );
    }
    fn television(&mut self, start: u64, episode: EpisodeId) {
        let span = u64::from(self.rng.range(120_000, 165_000));
        let energy = self.rng.real(0.75, 0.95);
        let vad = self.rng.real(0.80, 0.98);
        let mut now = start;
        while now + 2000 <= start + span {
            let duration_ms = self.rng.range(1000, 2000);
            self.segment(
                now + u64::from(duration_ms),
                SpeechCue {
                    duration_ms,
                    keyword: false,
                    energy,
                    vad_confidence: vad,
                    speaker_sim: None,
                    media: None,
                    turn_complete: None,
                },
                episode,
                Stimulus::Tv,
                TurnRole::Other,
            );
            now += u64::from(self.rng.range(3000, 6000));
        }
    }
    fn finish(
        mut self,
        kind: TapeKind,
        seed: u64,
        duration: u64,
        noise: Vec<Interval>,
    ) -> Result<Tape, Error> {
        // Insert ticks first at equal timestamps, without changing speech order.
        let ticks = (0..=duration).step_by(1000).map(|now| Record {
            event: Event::Tick { now: Millis(now) },
            annotation: Annotation::Clock,
        });
        let mut records: Vec<_> = ticks.collect();
        records.append(&mut self.records);
        records.sort_by_key(|record| record.event.now());
        Tape::new(kind, seed, Millis(duration), records, self.turns, noise)
    }
}

/// Generate the declared E1a mixed population; no controller is run or consulted.
pub fn e1a(seed: u64) -> Result<Tape, Error> {
    let mut b = Builder::new(seed);
    for block in 0u32..20 {
        let origin = u64::from(block) * 180_000;
        let episode = block * 5;
        let mut starts = [0u64; 5];
        for (value, offset) in starts
            .iter_mut()
            .zip([5000, 30_000, 55_000, 90_000, 120_000])
        {
            *value = origin + offset + u64::from(b.rng.range(0, 2000));
        }
        let [single, split, conversation, short, interruption] = starts;
        let duration = b.rng.range(1000, 1800);
        let first_end = b.request(
            single,
            EpisodeId(episode),
            duration,
            true,
            false,
            (crate::tape::TurnKind::Single, None),
        );
        let name_end = b.split(split, EpisodeId(episode + 1));
        b.conversation(conversation, EpisodeId(episode + 2), block);
        let duration = b.rng.range(250, 700);
        b.request(
            short,
            EpisodeId(episode + 3),
            duration,
            true,
            false,
            (crate::tape::TurnKind::Short, None),
        );
        let duration = b.rng.range(1000, 1800);
        let end = b.request(
            interruption,
            EpisodeId(episode + 4),
            duration,
            true,
            false,
            (crate::tape::TurnKind::Interruption, None),
        );
        b.request(
            end + 600,
            EpisodeId(episode + 4),
            600,
            false,
            true,
            (crate::tape::TurnKind::Interruption, None),
        );
        let distractor = episode + 100;
        b.television(origin, EpisodeId(distractor));
        b.other(first_end + 1000, EpisodeId(distractor + 1), 700);
        b.segment(
            name_end + 400,
            SpeechCue {
                energy: 0.8,
                vad_confidence: 0.8,
                duration_ms: 250,
                keyword: true,
                speaker_sim: None,
                media: None,
                turn_complete: None,
            },
            EpisodeId(distractor + 2),
            Stimulus::FalseKeyword,
            TurnRole::Other,
        );
        for n in 0..5 {
            b.noise(origin + 60_000 + n * 3000, EpisodeId(distractor + 3));
        }
        let duration = b.rng.range(700, 1600);
        b.other(origin + 110_000, EpisodeId(distractor + 4), duration);
    }
    b.finish(TapeKind::E1a, seed, 3_600_000, vec![])
}

/// Generate ten commands and fifty minutes of household noise cues over one hour.
pub fn e1b(seed: u64) -> Result<Tape, Error> {
    let mut b = Builder::new(seed);
    let mut intervals = Vec::with_capacity(10);
    for block in 0u32..10 {
        let origin = u64::from(block) * 360_000;
        intervals.push(Interval {
            start: Millis(origin),
            end: Millis(origin + 300_000),
        });
        for n in 0u32..100 {
            let start = origin + u64::from(n) * 3000 + u64::from(b.rng.range(0, 500));
            let (energy, vad, duration_ms, source) = match n % 4 {
                0 => (
                    b.rng.real(0.3, 0.8),
                    b.rng.real(0.05, 0.3),
                    b.rng.range(250, 1000),
                    Stimulus::Motor,
                ),
                1 => (
                    b.rng.real(0.1, 0.4),
                    b.rng.real(0.02, 0.2),
                    b.rng.range(250, 1000),
                    Stimulus::Ventilation,
                ),
                2 => (
                    b.rng.real(0.7, 0.95),
                    b.rng.real(0.8, 0.98),
                    b.rng.range(1200, 2400),
                    Stimulus::Tv,
                ),
                _ => (
                    b.rng.real(0.55, 0.95),
                    b.rng.real(0.65, 1.0),
                    b.rng.range(700, 1600),
                    Stimulus::OtherSpeech,
                ),
            };
            b.segment(
                start + u64::from(duration_ms),
                SpeechCue {
                    energy,
                    vad_confidence: vad,
                    duration_ms,
                    keyword: false,
                    speaker_sim: None,
                    media: None,
                    turn_complete: None,
                },
                EpisodeId(100 + block),
                source,
                TurnRole::Other,
            );
        }
        let start = origin + 300_000 + u64::from(b.rng.range(1000, 54_000));
        let duration = b.rng.range(1000, 1800);
        b.request(
            start,
            EpisodeId(block),
            duration,
            true,
            false,
            (crate::tape::TurnKind::Command, None),
        );
    }
    b.finish(TapeKind::E1b, seed, 3_600_000, intervals)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_c_known_answer_vectors() {
        // Compiled Vigna's linked reference C on 2026-09-25, setting its x to
        // each seed. These literals are not calculated by this Rust code.
        for (seed, expected) in [
            (
                0,
                [
                    0xe220_a839_7b1d_cdaf,
                    0x6e78_9e6a_a1b9_65f4,
                    0x06c4_5d18_8009_454f,
                    0xf88b_b8a8_724c_81ec,
                    0x1b39_896a_51a8_749b,
                    0x53cb_9f0c_747e_a2ea,
                    0x2c82_9abe_1f45_32e1,
                    0xc584_133a_c916_ab3c,
                ],
            ),
            (
                256,
                [
                    0x6602_d201_e324_653f,
                    0xf648_421a_27ca_8ae0,
                    0x7a63_9a53_5657_02ad,
                    0x963d_391a_95bf_a6ac,
                    0xa014_bb22_a044_6111,
                    0x8498_2fdb_4d57_da0a,
                    0xa9bd_6932_63ce_3676,
                    0x66e8_bcf0_c694_e249,
                ],
            ),
        ] {
            let mut rng = SplitMix64::with_seed(seed);
            for value in expected {
                assert_eq!(rng.next_u64(), value);
            }
        }
    }
    #[test]
    fn old_colliding_seeds_now_produce_diverse_stimuli() {
        let a = e1a(0).unwrap();
        let b = e1a(256).unwrap();
        let features = |t: &Tape| {
            t.records()
                .iter()
                .filter_map(|r| match r.event {
                    Event::Speech { cue, .. } => Some(cue),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let af = features(&a);
        let bf = features(&b);
        let different = af.iter().zip(&bf).filter(|(a, b)| a != b).count();
        assert!(different * 10 > af.len() * 8);
        assert_eq!(a, e1a(0).unwrap());
    }
    #[test]
    fn calibration_populations_have_full_counts_and_noise() {
        // Never cross the calibration boundary, including under the full gate.
        for seed in [0, 1, 7, 42, 256, 999] {
            let a = e1a(seed).unwrap();
            let b = e1b(seed).unwrap();
            let relevant: std::collections::BTreeSet<_> =
                a.turns().iter().map(|t| t.episode).collect();
            let distractors: std::collections::BTreeSet<_> = a
                .records()
                .iter()
                .filter_map(|r| match r.annotation {
                    Annotation::Speech {
                        episode, source, ..
                    } if source.turn().is_none() => episode,
                    _ => None,
                })
                .collect();
            assert_eq!(relevant.len(), 100);
            assert_eq!(distractors.len(), 100);
            assert_eq!(a.turns().len(), 200);
            assert_eq!(b.turns().len(), 10);
            assert_eq!(
                b.noise_intervals()
                    .iter()
                    .map(|s| s.end.0 - s.start.0)
                    .sum::<u64>(),
                3_000_000
            );
            assert_eq!(
                b.records()
                    .iter()
                    .filter(|r| matches!(r.event, Event::Speech { .. })
                        && r.annotation.turn().is_none())
                    .count(),
                1000
            );
            assert!(a.records().windows(3).any(|rows| {
                rows.iter().any(|r| r.annotation.turn().is_some())
                    && rows.iter().any(|r| {
                        matches!(
                            r.annotation.source(),
                            Some(Stimulus::Tv | Stimulus::OtherSpeech)
                        )
                    })
            }));
        }
    }
    #[test]
    fn speaker_sim_distribution_properties() {
        let mut pooled_user = Vec::new();
        let mut pooled_impostor = Vec::new();

        for seed in [0, 1, 7, 42] {
            let a = e1a(seed).unwrap();
            let b = e1b(seed).unwrap();
            let mut user_sims = Vec::new();
            let mut impostor_sims = Vec::new();

            for tape in [&a, &b] {
                for record in tape.records() {
                    if let Event::Speech { cue, .. } = &record.event {
                        let sim = cue.speaker_sim.expect(
                            "every speech cue in a generated tape must have Some speaker_sim",
                        );
                        match record.annotation.source() {
                            Some(Stimulus::Request(_) | Stimulus::BargeIn(_)) => {
                                user_sims.push(sim);
                                pooled_user.push(sim);
                            }
                            Some(Stimulus::Tv | Stimulus::OtherSpeech | Stimulus::FalseKeyword) => {
                                impostor_sims.push(sim);
                                pooled_impostor.push(sim);
                            }
                            _ => {}
                        }
                    }
                }
            }

            let user_mean = user_sims.iter().copied().sum::<f32>() / user_sims.len() as f32;
            let impostor_mean =
                impostor_sims.iter().copied().sum::<f32>() / impostor_sims.len() as f32;
            let impostor_ge_06 = impostor_sims.iter().filter(|&&s| s >= 0.6).count() as f32
                / impostor_sims.len() as f32;

            assert!(
                user_mean > 0.7,
                "seed {seed}: user mean was {user_mean}, expected > 0.7"
            );
            assert!(
                impostor_mean < 0.4,
                "seed {seed}: impostor mean was {impostor_mean}, expected < 0.4"
            );
            assert!(
                (0.01..=0.06).contains(&impostor_ge_06),
                "seed {seed}: impostor share >= 0.6 was {impostor_ge_06}, expected in 0.01..=0.06"
            );
        }

        let pooled_user_mean = pooled_user.iter().copied().sum::<f32>() / pooled_user.len() as f32;
        let pooled_impostor_mean =
            pooled_impostor.iter().copied().sum::<f32>() / pooled_impostor.len() as f32;
        let pooled_impostor_ge_06 = pooled_impostor.iter().filter(|&&s| s >= 0.6).count() as f32
            / pooled_impostor.len() as f32;

        assert!(
            pooled_user_mean > 0.7,
            "pooled user mean was {pooled_user_mean}, expected > 0.7"
        );
        assert!(
            pooled_impostor_mean < 0.4,
            "pooled impostor mean was {pooled_impostor_mean}, expected < 0.4"
        );
        assert!(
            (0.01..=0.06).contains(&pooled_impostor_ge_06),
            "pooled impostor share >= 0.6 was {pooled_impostor_ge_06}, expected in 0.01..=0.06"
        );
    }
    #[test]
    fn media_distribution_properties() {
        let mut pooled_tv = Vec::new();
        let mut pooled_live = Vec::new();

        for seed in [0, 1, 7, 42] {
            let a = e1a(seed).unwrap();
            let b = e1b(seed).unwrap();
            let mut tv_media = Vec::new();
            let mut live_media = Vec::new();

            for tape in [&a, &b] {
                for record in tape.records() {
                    if let Event::Speech { cue, .. } = &record.event {
                        let media = cue
                            .media
                            .expect("every speech cue in a generated tape must have Some media");
                        match record.annotation.source() {
                            Some(Stimulus::Tv) => {
                                tv_media.push(media);
                                pooled_tv.push(media);
                            }
                            Some(
                                Stimulus::Request(_)
                                | Stimulus::BargeIn(_)
                                | Stimulus::OtherSpeech
                                | Stimulus::FalseKeyword,
                            ) => {
                                live_media.push(media);
                                pooled_live.push(media);
                            }
                            _ => {}
                        }
                    }
                }
            }

            let tv_mean = tv_media.iter().copied().sum::<f32>() / tv_media.len() as f32;
            let live_mean = live_media.iter().copied().sum::<f32>() / live_media.len() as f32;
            let tv_lt_05 =
                tv_media.iter().filter(|&&m| m < 0.5).count() as f32 / tv_media.len() as f32;
            let live_ge_05 =
                live_media.iter().filter(|&&m| m >= 0.5).count() as f32 / live_media.len() as f32;

            assert!(
                tv_mean > 0.7,
                "seed {seed}: tv mean was {tv_mean}, expected > 0.7"
            );
            assert!(
                live_mean < 0.25,
                "seed {seed}: live mean was {live_mean}, expected < 0.25"
            );
            assert!(
                (0.01..=0.08).contains(&tv_lt_05),
                "seed {seed}: tv share < 0.5 was {tv_lt_05}, expected in 0.01..=0.08"
            );
            assert!(
                (0.01..=0.08).contains(&live_ge_05),
                "seed {seed}: live share >= 0.5 was {live_ge_05}, expected in 0.01..=0.08"
            );
        }

        let pooled_tv_mean = pooled_tv.iter().copied().sum::<f32>() / pooled_tv.len() as f32;
        let pooled_live_mean = pooled_live.iter().copied().sum::<f32>() / pooled_live.len() as f32;
        let pooled_tv_lt_05 =
            pooled_tv.iter().filter(|&&m| m < 0.5).count() as f32 / pooled_tv.len() as f32;
        let pooled_live_ge_05 =
            pooled_live.iter().filter(|&&m| m >= 0.5).count() as f32 / pooled_live.len() as f32;

        assert!(
            pooled_tv_mean > 0.7,
            "pooled tv mean was {pooled_tv_mean}, expected > 0.7"
        );
        assert!(
            pooled_live_mean < 0.25,
            "pooled live mean was {pooled_live_mean}, expected < 0.25"
        );
        assert!(
            (0.01..=0.08).contains(&pooled_tv_lt_05),
            "pooled tv share < 0.5 was {pooled_tv_lt_05}, expected in 0.01..=0.08"
        );
        assert!(
            (0.01..=0.08).contains(&pooled_live_ge_05),
            "pooled live share >= 0.5 was {pooled_live_ge_05}, expected in 0.01..=0.08"
        );
    }
    #[test]
    fn speaker_sim_draws_are_independent_of_media_stream() {
        let tape = e1a(42).unwrap();
        let mut sim_rng = SplitMix64::with_seed(0x2a ^ SPEAKER_STREAM_SALT);
        let mut speech_records: Vec<_> = tape
            .records()
            .iter()
            .filter(|r| matches!(r.event, Event::Speech { .. }))
            .collect();
        speech_records.sort_by_key(|r| match r.annotation {
            Annotation::Speech { segment, .. } => segment.0,
            Annotation::Clock => 0,
        });
        for record in speech_records {
            if let Event::Speech { cue, .. } = &record.event {
                let source = record.annotation.source().unwrap();
                let expected_sim = match source {
                    Stimulus::Request(_) | Stimulus::BargeIn(_) => {
                        if sim_rng.real(0.0, 1.0) < SPEAKER_SIM_CROSSOVER_RATE {
                            sim_rng.real(IMPOSTOR_SPEAKER_SIM_MIN, IMPOSTOR_SPEAKER_SIM_MAX)
                        } else {
                            sim_rng.real(USER_SPEAKER_SIM_MIN, USER_SPEAKER_SIM_MAX)
                        }
                    }
                    Stimulus::Tv | Stimulus::OtherSpeech | Stimulus::FalseKeyword => {
                        if sim_rng.real(0.0, 1.0) < SPEAKER_SIM_CROSSOVER_RATE {
                            sim_rng.real(USER_SPEAKER_SIM_MIN, USER_SPEAKER_SIM_MAX)
                        } else {
                            sim_rng.real(IMPOSTOR_SPEAKER_SIM_MIN, IMPOSTOR_SPEAKER_SIM_MAX)
                        }
                    }
                    Stimulus::Noise
                    | Stimulus::Motor
                    | Stimulus::Ventilation
                    | Stimulus::SelfEcho => {
                        sim_rng.real(NON_SPEECH_SPEAKER_SIM_MIN, NON_SPEECH_SPEAKER_SIM_MAX)
                    }
                };
                assert_eq!(
                    cue.speaker_sim,
                    Some(expected_sim),
                    "speaker_sim must match independent stream unaffected by media"
                );
            }
        }
    }
    #[test]
    fn turn_complete_distribution_properties() {
        let mut pooled_complete = Vec::new();
        let mut pooled_incomplete = Vec::new();

        for seed in 0..=31 {
            let a = e1a(seed).unwrap();
            let b = e1b(seed).unwrap();

            for tape in [&a, &b] {
                let mut incomplete_segments = std::collections::BTreeSet::new();
                let mut complete_segments = std::collections::BTreeSet::new();
                for turn in tape.turns() {
                    if let Some((&last, prefix)) = turn.segments.split_last() {
                        complete_segments.insert(last);
                        for &seg in prefix {
                            incomplete_segments.insert(seg);
                        }
                    }
                }

                for record in tape.records() {
                    if let Event::Speech { cue, .. } = &record.event {
                        let score = cue.turn_complete.expect(
                            "every speech cue in a generated tape must have Some turn_complete",
                        );
                        if let Annotation::Speech { segment, .. } = record.annotation {
                            if complete_segments.contains(&segment) {
                                pooled_complete.push(score);
                            } else if incomplete_segments.contains(&segment) {
                                pooled_incomplete.push(score);
                            }
                        }
                    }
                }
            }
        }

        let pooled_complete_mean =
            pooled_complete.iter().copied().sum::<f32>() / pooled_complete.len() as f32;
        let pooled_incomplete_mean =
            pooled_incomplete.iter().copied().sum::<f32>() / pooled_incomplete.len() as f32;
        let pooled_complete_lt_05 = pooled_complete.iter().filter(|&&t| t < 0.5).count() as f32
            / pooled_complete.len() as f32;
        let pooled_incomplete_ge_05 = pooled_incomplete.iter().filter(|&&t| t >= 0.5).count()
            as f32
            / pooled_incomplete.len() as f32;

        assert!(
            pooled_complete_mean > 0.75,
            "pooled complete mean was {pooled_complete_mean}, expected > 0.75"
        );
        assert!(
            pooled_incomplete_mean < 0.25,
            "pooled incomplete mean was {pooled_incomplete_mean}, expected < 0.25"
        );
        assert!(
            pooled_complete_lt_05 < 0.05,
            "pooled complete share < 0.5 was {pooled_complete_lt_05}, expected < 0.05"
        );
        assert!(
            pooled_incomplete_ge_05 < 0.05,
            "pooled incomplete share >= 0.5 was {pooled_incomplete_ge_05}, expected < 0.05"
        );
    }
    #[test]
    fn speaker_sim_and_media_draws_are_independent_of_turn_stream() {
        let tape = e1a(42).unwrap();
        let mut sim_rng = SplitMix64::with_seed(0x2a ^ SPEAKER_STREAM_SALT);
        let mut media_rng = SplitMix64::with_seed(0x2a ^ MEDIA_STREAM_SALT);

        let mut speech_records: Vec<_> = tape
            .records()
            .iter()
            .filter(|r| matches!(r.event, Event::Speech { .. }))
            .collect();
        speech_records.sort_by_key(|r| match r.annotation {
            Annotation::Speech { segment, .. } => segment.0,
            Annotation::Clock => 0,
        });
        for record in speech_records {
            if let Event::Speech { cue, .. } = &record.event {
                let source = match record.annotation {
                    Annotation::Speech { source, .. } => source,
                    Annotation::Clock => continue,
                };
                let expected_sim = match source {
                    Stimulus::Request(_) | Stimulus::BargeIn(_) => {
                        if sim_rng.real(0.0, 1.0) < SPEAKER_SIM_CROSSOVER_RATE {
                            sim_rng.real(IMPOSTOR_SPEAKER_SIM_MIN, IMPOSTOR_SPEAKER_SIM_MAX)
                        } else {
                            sim_rng.real(USER_SPEAKER_SIM_MIN, USER_SPEAKER_SIM_MAX)
                        }
                    }
                    Stimulus::Tv | Stimulus::OtherSpeech | Stimulus::FalseKeyword => {
                        if sim_rng.real(0.0, 1.0) < SPEAKER_SIM_CROSSOVER_RATE {
                            sim_rng.real(USER_SPEAKER_SIM_MIN, USER_SPEAKER_SIM_MAX)
                        } else {
                            sim_rng.real(IMPOSTOR_SPEAKER_SIM_MIN, IMPOSTOR_SPEAKER_SIM_MAX)
                        }
                    }
                    Stimulus::Noise
                    | Stimulus::Motor
                    | Stimulus::Ventilation
                    | Stimulus::SelfEcho => {
                        sim_rng.real(NON_SPEECH_SPEAKER_SIM_MIN, NON_SPEECH_SPEAKER_SIM_MAX)
                    }
                };
                let expected_media = match source {
                    Stimulus::Tv => {
                        if media_rng.real(0.0, 1.0) < TV_MEDIA_MISS_RATE {
                            media_rng.real(LIVE_VOICE_MEDIA_MIN, LIVE_VOICE_MEDIA_MAX)
                        } else {
                            media_rng.real(TV_MEDIA_MIN, TV_MEDIA_MAX)
                        }
                    }
                    Stimulus::Request(_)
                    | Stimulus::BargeIn(_)
                    | Stimulus::OtherSpeech
                    | Stimulus::FalseKeyword => {
                        if media_rng.real(0.0, 1.0) < LIVE_VOICE_MEDIA_FALSE_ALARM_RATE {
                            media_rng.real(TV_MEDIA_MIN, TV_MEDIA_MAX)
                        } else {
                            media_rng.real(LIVE_VOICE_MEDIA_MIN, LIVE_VOICE_MEDIA_MAX)
                        }
                    }
                    Stimulus::Noise
                    | Stimulus::Motor
                    | Stimulus::Ventilation
                    | Stimulus::SelfEcho => {
                        media_rng.real(NON_SPEECH_MEDIA_MIN, NON_SPEECH_MEDIA_MAX)
                    }
                };

                assert_eq!(
                    cue.speaker_sim,
                    Some(expected_sim),
                    "speaker_sim must match independent stream unaffected by turn_complete"
                );
                assert_eq!(
                    cue.media,
                    Some(expected_media),
                    "media must match independent stream unaffected by turn_complete"
                );
            }
        }
    }
    #[test]
    fn turn_complete_draws_match_independent_stream() {
        let tape = e1a(42).unwrap();
        let mut turn_rng = SplitMix64::with_seed(0x2a ^ TURN_STREAM_SALT);

        let mut incomplete_segments = std::collections::BTreeSet::new();
        let mut complete_segments = std::collections::BTreeSet::new();
        for turn in tape.turns() {
            if let Some((&last, prefix)) = turn.segments.split_last() {
                complete_segments.insert(last);
                for &seg in prefix {
                    incomplete_segments.insert(seg);
                }
            }
        }

        let mut speech_records: Vec<_> = tape
            .records()
            .iter()
            .filter(|r| matches!(r.event, Event::Speech { .. }))
            .collect();
        speech_records.sort_by_key(|r| match r.annotation {
            Annotation::Speech { segment, .. } => segment.0,
            Annotation::Clock => 0,
        });
        for record in speech_records {
            if let Event::Speech { cue, .. } = &record.event {
                let seg = match record.annotation {
                    Annotation::Speech { segment, .. } => segment,
                    Annotation::Clock => continue,
                };
                let role = if complete_segments.contains(&seg) {
                    TurnRole::Complete
                } else if incomplete_segments.contains(&seg) {
                    TurnRole::Incomplete
                } else {
                    TurnRole::Other
                };
                let expected_turn = match role {
                    TurnRole::Complete => {
                        if turn_rng.real(0.0, 1.0) < COMPLETE_USER_TURN_MISS_RATE {
                            turn_rng.real(INCOMPLETE_USER_TURN_MIN, INCOMPLETE_USER_TURN_MAX)
                        } else {
                            turn_rng.real(COMPLETE_USER_TURN_MIN, COMPLETE_USER_TURN_MAX)
                        }
                    }
                    TurnRole::Incomplete => {
                        if turn_rng.real(0.0, 1.0) < INCOMPLETE_USER_TURN_PREMATURE_RATE {
                            turn_rng.real(COMPLETE_USER_TURN_MIN, COMPLETE_USER_TURN_MAX)
                        } else {
                            turn_rng.real(INCOMPLETE_USER_TURN_MIN, INCOMPLETE_USER_TURN_MAX)
                        }
                    }
                    TurnRole::Other => turn_rng.real(NON_USER_TURN_MIN, NON_USER_TURN_MAX),
                };

                assert_eq!(
                    cue.turn_complete,
                    Some(expected_turn),
                    "turn_complete must match independent stream"
                );
            }
        }
    }
}

#[cfg(test)]
mod coverage_tests {
    use super::*;
    #[test]
    fn sustained_tv_interruptions_and_both_sides_of_the_window_are_present() {
        let tape = e1a(7).unwrap();
        let mut tv = std::collections::BTreeMap::<EpisodeId, (u64, u64)>::new();
        for record in tape.records() {
            if let Annotation::Speech {
                episode: Some(episode),
                source: Stimulus::Tv,
                ..
            } = record.annotation
            {
                let time = record.event.now().0;
                let span = tv.entry(episode).or_insert((time, time));
                span.1 = time;
            }
        }
        assert_eq!(tv.len(), 20);
        assert!(tv.values().all(|(start, end)| end - start > 110_000));
        let mut conversations = std::collections::BTreeMap::<EpisodeId, Vec<u64>>::new();
        for turn in tape.turns() {
            conversations
                .entry(turn.episode)
                .or_default()
                .push(turn.available_at.0);
        }
        let gaps: Vec<_> = conversations
            .values()
            .filter(|ends| ends.len() == 5)
            .flat_map(|ends| ends.windows(2).map(|pair| pair[1] - pair[0]))
            .collect();
        assert!(gaps.iter().any(|gap| (4800..5000).contains(gap)));
        assert!(gaps.iter().any(|gap| (5001..=5200).contains(gap)));
        assert!(gaps.iter().any(|gap| *gap >= 8900));
        for episode in (0..100).step_by(5) {
            let request = tape
                .turns()
                .iter()
                .find(|turn| turn.episode == EpisodeId(episode))
                .unwrap();
            assert!(tape.records().iter().any(|record| record.event.now().0
                == request.available_at.0 + 1000
                && record.annotation.source() == Some(Stimulus::OtherSpeech)));
        }
    }
}
