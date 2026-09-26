//! Calibrated sensor evidence: what a reading says, in nats, about a cue.
//!
//! A sensor reading is not a verdict. Each model below says how a sensor scores
//! one class of sound at a given duration, and the log-likelihood ratio between
//! two classes is how far a reading should move the organism's belief.
//!
//! All classes of one sensor share one spread, so every ratio is linear in the
//! reading and needs only IEEE 754 arithmetic: replay is bit-identical on every
//! platform. Each ratio is capped at `max_llr`, because the tails are where a
//! calibration is least trustworthy.

use serde::{Deserialize, Serialize};

use crate::SpeechCue;

/// Durations, in milliseconds, at which calibrated means are given. Between
/// anchors a mean is interpolated linearly in duration; outside them the
/// nearest anchor holds.
pub const ANCHORS_MS: [u32; 3] = [750, 1_500, 3_000];

/// Durations, in milliseconds, that split cues into the three end-of-turn bands:
/// below the first, up to the second, and longer.
pub const TURN_BANDS_MS: [u32; 2] = [1_000, 2_000];

/// End-of-turn scores that split a band into four bins.
pub const TURN_EDGES: [f32; 3] = [0.1, 0.5, 0.9];

/// Directedness scores that split a detector's output into three bands: clearly
/// addressed to someone else below the first, ambiguous up to the second, and
/// clearly addressed to Enton from it on.
pub const DIRECTED_EDGES: [f32; 2] = [0.3, 0.75];

/// Narrowest spread a calibration may declare: tighter than any real sensor,
/// wide enough that no ratio overflows or divides by zero.
pub const MIN_SD: f32 = 0.01;

/// How speaker verification scores each kind of voice against the owner's voiceprint.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VoiceModel {
    /// Mean similarity of the owner's own voice at each anchor.
    pub owner: [f32; 3],
    /// Mean similarity of another live person in the home.
    pub other: [f32; 3],
    /// Mean similarity of a voice reproduced by a loudspeaker (TV, radio).
    pub reproduced: [f32; 3],
    /// Spread shared by the three classes.
    pub sd: f32,
}

/// How the audio tagger scores live sound and sound from a loudspeaker.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SourceModel {
    /// Mean media likelihood of a live voice in the room at each anchor.
    pub live: [f32; 3],
    /// Mean media likelihood of reproduced audio at each anchor.
    pub reproduced: [f32; 3],
    /// Spread shared by both classes.
    pub sd: f32,
}

/// How the end-of-turn model separates a finished turn from a pause inside one.
/// Its errors are confident (scores pile up near zero and one on both sides), so
/// it is calibrated by bins rather than by a spread.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TurnModel {
    /// Log-likelihood ratio, finished over unfinished, for each duration band
    /// (rows, split at [`TURN_BANDS_MS`]) and score bin (columns, split at
    /// [`TURN_EDGES`]).
    pub llr: [[f32; 4]; 3],
}

/// How a device-directedness detector separates speech addressed to Enton from
/// speech addressed to someone else in the room. Like the end-of-turn model its
/// errors are confident (a command-shaped aside to a person scores near one), so
/// it is calibrated by bands rather than by a spread.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DirectedModel {
    /// Log-likelihood ratio, addressed to Enton over addressed to someone else,
    /// for each score band (split at [`DIRECTED_EDGES`]).
    pub llr: [f32; 3],
}

/// Calibration of every sensor that may describe a speech cue.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Senses {
    /// Speaker verification against the owner's voiceprint.
    pub voice: VoiceModel,
    /// Live sound versus a loudspeaker.
    pub source: SourceModel,
    /// Finished turn versus a pause inside one.
    pub turn: TurnModel,
    /// Speech addressed to Enton versus speech addressed to someone else.
    #[serde(default)]
    pub directed: DirectedModel,
    /// Largest weight, in nats, that any single ratio may carry.
    pub max_llr: f32,
}

/// What one cue says, in nats. Positive favors the first-named class; a sensor
/// that did not run says nothing (zero).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    /// The owner's voice over another live person's.
    pub owner_over_other: f32,
    /// The owner's voice over a voice from a loudspeaker.
    pub owner_over_reproduced: f32,
    /// Live sound over sound from a loudspeaker.
    pub live_over_reproduced: f32,
    /// A finished turn over a pause inside one.
    pub finished_over_unfinished: f32,
    /// Speech addressed to Enton over speech addressed to someone else.
    #[serde(default)]
    pub addressed_over_not: f32,
}

impl Evidence {
    /// The owner speaking live, over the more plausible alternative: another
    /// person in the room, or a loudspeaker. Voice and source are added only
    /// against the loudspeaker, the one alternative both sensors describe.
    #[must_use]
    pub fn owner_live(&self) -> f32 {
        self.owner_over_other
            .min(self.owner_over_reproduced + self.live_over_reproduced)
    }
}

impl Senses {
    /// Calibration measured for the sensors Enton ships, pooled over a home in
    /// which the owner is near the device a quarter of the time and a TV plays
    /// in the background half of the time.
    ///
    /// - Voice: CAM++ (3D-Speaker, zh-cn common) against a 10 s voiceprint
    ///   enrolled near the device, on simulated rooms; "other" is a same-gender
    ///   relative in the same room, the hardest common case.
    /// - Source: published single-microphone live-versus-media detectors
    ///   (about 30% of TV dialogue missed, 7 to 35% of live voices flagged).
    /// - Turn: Smart Turn v3.2 at about 250 ms of silence, from per-pause
    ///   predictions on real Portuguese turns.
    /// - Directed: see [`DirectedModel::calibrated`].
    ///
    /// A device should replace the voice rows with what `owner_probe` measures
    /// on its own microphone.
    #[must_use]
    pub const fn calibrated() -> Self {
        Self {
            voice: VoiceModel {
                owner: [0.38, 0.54, 0.61],
                other: [0.33, 0.44, 0.50],
                reproduced: [0.24, 0.34, 0.38],
                sd: 0.16,
            },
            source: SourceModel {
                live: [0.37, 0.33, 0.29],
                reproduced: [0.58, 0.63, 0.67],
                sd: 0.20,
            },
            turn: TurnModel {
                llr: [
                    [-1.06, 0.02, 0.51, 1.45],
                    [-1.64, -0.69, -0.18, 1.02],
                    [-1.28, -0.61, -0.23, 0.67],
                ],
            },
            directed: DirectedModel::calibrated(),
            max_llr: 3.0,
        }
    }

    /// Read a cue. Readings are canonicalized first, so a non-finite score is a
    /// sensor that did not run.
    #[must_use]
    pub fn read(&self, cue: &SpeechCue) -> Evidence {
        let cue = cue.canonical();
        let duration = cue.duration_ms;
        let voice = self.voice;
        let ratio = |reading: Option<f32>, a: [f32; 3], b: [f32; 3], sd: f32| {
            reading.map_or(0.0, |x| {
                self.capped(linear_llr(
                    x,
                    mean_at(a, duration),
                    mean_at(b, duration),
                    sd,
                ))
            })
        };
        Evidence {
            owner_over_other: ratio(cue.speaker_sim, voice.owner, voice.other, voice.sd),
            owner_over_reproduced: ratio(cue.speaker_sim, voice.owner, voice.reproduced, voice.sd),
            // The tagger scores how reproduced a sound is, so live comes second.
            live_over_reproduced: -ratio(
                cue.media,
                self.source.reproduced,
                self.source.live,
                self.source.sd,
            ),
            finished_over_unfinished: cue
                .turn_complete
                .map_or(0.0, |score| self.capped(self.turn.llr_at(score, duration))),
            addressed_over_not: cue
                .directed
                .map_or(0.0, |score| self.capped(self.directed.llr_at(score))),
        }
    }

    /// Whether every mean lies in the unit interval, every ratio is finite, every
    /// spread lies between [`MIN_SD`] and one, and the cap is positive.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let voice = self.voice;
        let source = self.source;
        let means = [
            voice.owner,
            voice.other,
            voice.reproduced,
            source.live,
            source.reproduced,
        ];
        means
            .iter()
            .flatten()
            .all(|mean| (0.0..=1.0).contains(mean))
            && [voice.sd, source.sd]
                .iter()
                .all(|sd| (MIN_SD..=1.0).contains(sd))
            && self.max_llr.is_finite()
            && self.max_llr > 0.0
            && self.turn.llr.iter().flatten().all(|llr| llr.is_finite())
            && self.directed.llr.iter().all(|llr| llr.is_finite())
    }

    fn capped(&self, llr: f32) -> f32 {
        if llr.is_nan() {
            return 0.0;
        }
        llr.clamp(-self.max_llr, self.max_llr)
    }
}

impl Default for Senses {
    fn default() -> Self {
        Self::calibrated()
    }
}

impl DirectedModel {
    /// A text-based device-directedness detector that reads the transcript with the
    /// previous turn as context. Published follow-up detectors reach 8 to 14% equal
    /// error rate; this is the conservative end, against the hardest alternative:
    /// the owner, in the same voice, talking to someone else in the room. Per band,
    /// addressed to Enton over such an aside: ln(0.05 / 0.70), ln(0.10 / 0.18) and
    /// ln(0.85 / 0.12).
    #[must_use]
    pub const fn calibrated() -> Self {
        Self {
            llr: [-2.64, -0.59, 1.96],
        }
    }

    fn llr_at(&self, score: f32) -> f32 {
        let band = DIRECTED_EDGES.iter().filter(|edge| score >= **edge).count();
        self.llr.get(band).copied().unwrap_or(0.0)
    }
}

impl Default for DirectedModel {
    fn default() -> Self {
        Self::calibrated()
    }
}

impl TurnModel {
    fn llr_at(&self, score: f32, duration_ms: u32) -> f32 {
        let band = if duration_ms < TURN_BANDS_MS[0] {
            0
        } else if duration_ms <= TURN_BANDS_MS[1] {
            1
        } else {
            2
        };
        let bin = TURN_EDGES.iter().filter(|edge| score >= **edge).count();
        self.llr
            .get(band)
            .and_then(|row| row.get(bin))
            .copied()
            .unwrap_or(0.0)
    }
}

/// A calibrated mean at `duration_ms`: linear between anchors, flat outside them.
fn mean_at(means: [f32; 3], duration_ms: u32) -> f32 {
    let [first, middle, last] = ANCHORS_MS;
    let (from, to, a, b) = if duration_ms <= first {
        return means[0];
    } else if duration_ms <= middle {
        (first, middle, means[0], means[1])
    } else if duration_ms <= last {
        (middle, last, means[1], means[2])
    } else {
        return means[2];
    };
    let weight = (duration_ms - from) as f32 / (to - from) as f32;
    a + weight * (b - a)
}

/// Log-likelihood ratio of class `a` over class `b` for reading `x`, both
/// Gaussian with means `a` and `b` and the same spread: linear in `x`, zero
/// halfway between the means.
fn linear_llr(x: f32, a: f32, b: f32, sd: f32) -> f32 {
    (a - b) * (2.0 * x - a - b) / (2.0 * sd * sd)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(duration_ms: u32, sim: Option<f32>, media: Option<f32>, turn: Option<f32>) -> SpeechCue {
        SpeechCue {
            energy: 0.8,
            duration_ms,
            vad_confidence: 0.9,
            keyword: false,
            speaker_sim: sim,
            media,
            turn_complete: turn,
            directed: None,
        }
    }

    /// A cue that only a directedness detector described.
    fn addressed(directed: Option<f32>) -> SpeechCue {
        SpeechCue {
            directed,
            ..cue(1_500, None, None, None)
        }
    }

    #[test]
    fn a_missing_sensor_says_nothing() {
        let evidence = Senses::calibrated().read(&cue(1_500, None, None, None));
        assert_eq!(evidence, Evidence::default());
        assert!(evidence.owner_live().abs() < f32::EPSILON);
    }

    #[test]
    fn a_non_finite_reading_is_a_sensor_that_did_not_run() {
        let evidence = Senses::calibrated().read(&SpeechCue {
            directed: Some(f32::NAN),
            ..cue(1_500, Some(f32::NAN), Some(f32::INFINITY), Some(f32::NAN))
        });
        assert_eq!(evidence, Evidence::default());
    }

    #[test]
    fn directedness_bands_split_at_their_edges() {
        let senses = Senses::calibrated();
        let weigh = |score| senses.read(&addressed(Some(score))).addressed_over_not;
        let [undirected, ambiguous, directed] = senses.directed.llr;
        for (score, expected) in [
            (0.0, undirected),
            (0.299, undirected),
            (0.3, ambiguous),
            (0.749, ambiguous),
            (0.75, directed),
            (1.0, directed),
        ] {
            assert_eq!(weigh(score).to_bits(), expected.to_bits(), "{score}");
        }
        // Out of range readings are clamped into the unit interval first.
        assert_eq!(weigh(7.0).to_bits(), directed.to_bits());
        assert_eq!(weigh(-1.0).to_bits(), undirected.to_bits());
        // No detector, no evidence, whatever the other sensors say.
        assert_eq!(
            senses.read(&addressed(None)).addressed_over_not.to_bits(),
            0
        );
    }

    #[test]
    fn only_a_clearly_addressed_reading_favors_enton() {
        let senses = Senses::calibrated();
        let weigh = |score| senses.read(&addressed(Some(score))).addressed_over_not;
        assert!(weigh(0.1) < -2.0, "an aside-like score is strong evidence");
        assert!(
            (-1.0..0.0).contains(&weigh(0.5)),
            "an ambiguous score leans away"
        );
        assert!(weigh(0.9) > 1.5);
    }

    #[test]
    fn the_same_directedness_reading_always_weighs_the_same() {
        // A table lookup and a clamp: these exact bits reproduce on every platform.
        let senses = Senses::calibrated();
        let bits = [0.1, 0.5, 0.9].map(|score| {
            senses
                .read(&addressed(Some(score)))
                .addressed_over_not
                .to_bits()
        });
        assert_eq!(bits, [0xc028_f5c3, 0xbf17_0a3d, 0x3ffa_e148]);
    }

    #[test]
    fn a_calibration_stored_without_directedness_reads_back_with_the_shipped_model() {
        let mut stored = serde_json::to_value(Senses::calibrated()).unwrap();
        stored.as_object_mut().unwrap().remove("directed");
        let senses: Senses = serde_json::from_value(stored).unwrap();
        assert_eq!(senses, Senses::calibrated());
    }

    #[test]
    fn a_directedness_ratio_is_capped_too() {
        let mut senses = Senses::calibrated();
        senses.directed.llr = [-9.0, 0.0, 9.0];
        let weigh = |score| senses.read(&addressed(Some(score))).addressed_over_not;
        assert_eq!(weigh(0.0).to_bits(), (-senses.max_llr).to_bits());
        assert_eq!(weigh(1.0).to_bits(), senses.max_llr.to_bits());
    }

    #[test]
    fn a_reading_halfway_between_two_means_is_neutral() {
        let senses = Senses::calibrated();
        let halfway = f32::midpoint(0.54, 0.44);
        let evidence = senses.read(&cue(1_500, Some(halfway), None, None));
        assert!(evidence.owner_over_other.abs() < 1e-5);
    }

    #[test]
    fn the_owner_mean_favors_the_owner_and_the_other_mean_the_other() {
        let senses = Senses::calibrated();
        let owner = senses.read(&cue(1_500, Some(0.54), None, None));
        let other = senses.read(&cue(1_500, Some(0.44), None, None));
        assert!(owner.owner_over_other > 0.0);
        assert!((owner.owner_over_other + other.owner_over_other).abs() < 1e-5);
        // d' of 0.1 / 0.16 is weak evidence: under one nat either way.
        assert!(owner.owner_over_other < 1.0);
    }

    #[test]
    fn a_tv_like_reading_points_away_from_the_owner_speaking_live() {
        let senses = Senses::calibrated();
        let tv = senses.read(&cue(1_500, Some(0.34), Some(0.63), None));
        let owner = senses.read(&cue(1_500, Some(0.54), Some(0.33), None));
        assert!(tv.live_over_reproduced < 0.0);
        assert!(tv.owner_live() < -1.0);
        assert!(owner.owner_live() > 0.0);
    }

    #[test]
    fn every_ratio_is_capped() {
        let senses = Senses::calibrated();
        let extreme = senses.read(&cue(3_000, Some(0.0), Some(1.0), None));
        for llr in [
            extreme.owner_over_other,
            extreme.owner_over_reproduced,
            extreme.live_over_reproduced,
        ] {
            assert!(llr >= -senses.max_llr, "{llr}");
        }
        assert!((extreme.owner_over_reproduced + senses.max_llr).abs() < f32::EPSILON);
    }

    #[test]
    fn means_hold_outside_the_anchors_and_interpolate_between_them() {
        let means = [0.2, 0.5, 0.8];
        assert!((mean_at(means, 100) - 0.2).abs() < f32::EPSILON);
        assert!((mean_at(means, 750) - 0.2).abs() < f32::EPSILON);
        assert!((mean_at(means, 1_125) - 0.35).abs() < 1e-6);
        assert!((mean_at(means, 1_500) - 0.5).abs() < f32::EPSILON);
        assert!((mean_at(means, 2_250) - 0.65).abs() < 1e-6);
        assert!((mean_at(means, 3_000) - 0.8).abs() < f32::EPSILON);
        assert!((mean_at(means, 60_000) - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn turn_bins_follow_duration_bands_and_score_edges() {
        let senses = Senses::calibrated();
        let turn = |duration, score| {
            senses
                .read(&cue(duration, None, None, Some(score)))
                .finished_over_unfinished
        };
        let expected = senses.turn.llr;
        assert!((turn(999, 0.05) - expected[0][0]).abs() < f32::EPSILON);
        assert!((turn(1_000, 0.1) - expected[1][1]).abs() < f32::EPSILON);
        assert!((turn(2_000, 0.5) - expected[1][2]).abs() < f32::EPSILON);
        assert!((turn(2_001, 0.95) - expected[2][3]).abs() < f32::EPSILON);
    }

    #[test]
    fn the_same_reading_always_weighs_the_same() {
        // Arithmetic only: these exact bits must reproduce on every platform.
        let evidence = Senses::calibrated().read(&cue(1_200, Some(0.61), Some(0.4), Some(0.95)));
        let bits = [
            evidence.owner_over_other,
            evidence.owner_over_reproduced,
            evidence.live_over_reproduced,
            evidence.finished_over_unfinished,
        ]
        .map(f32::to_bits);
        // The same sequence of float32 operations in numpy gives these bits too.
        assert_eq!(bits, [0x3f0b_3335, 0x3fc3_5c29, 0x3f03_c9ef, 0x3f82_8f5c]);
    }

    #[test]
    fn the_calibration_is_valid_and_rejects_nonsense() {
        let mut senses = Senses::calibrated();
        assert!(senses.is_valid());
        senses.voice.sd = 0.0;
        assert!(!senses.is_valid());
        let mut senses = Senses::calibrated();
        senses.source.live[1] = 1.5;
        assert!(!senses.is_valid());
        let mut senses = Senses::calibrated();
        senses.turn.llr[2][0] = f32::NAN;
        assert!(!senses.is_valid());
        let mut senses = Senses::calibrated();
        senses.directed.llr[1] = f32::INFINITY;
        assert!(!senses.is_valid());
        // A spread so narrow that 2 sd^2 underflows would divide zero by zero.
        let mut senses = Senses::calibrated();
        senses.voice.sd = 1e-30;
        assert!(!senses.is_valid());
    }
}
