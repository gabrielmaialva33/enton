//! Inputs to the organism, produced by adapters and consumed by the core.

use serde::{Deserialize, Serialize};

use crate::action::ThoughtId;

/// Monotonic milliseconds supplied by an adapter, never read by the core.
/// Keeping time explicit makes replay deterministic.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Millis(pub u64);

impl Millis {
    /// Milliseconds elapsed since `earlier`, or zero if time moved backward.
    #[must_use]
    pub fn since(self, earlier: Millis) -> u64 {
        self.0.saturating_sub(earlier.0)
    }
}

/// A sequential, deterministic utterance identifier for synthesized speech.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct UtteranceId(pub u64);

impl core::fmt::Display for UtteranceId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// A stimulus reduced to the information the brainstem needs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Event {
    /// Passage of time, emitted periodically by the clock adapter.
    Tick {
        /// The current time.
        now: Millis,
    },
    /// An interoceptive hardware reading (thesis D1).
    Body {
        /// The current time.
        now: Millis,
        /// Physical signals from sensors.
        signals: BodySignals,
    },
    /// A cheap speech cue, before transcription (perception by surprise, P5).
    Speech {
        /// The current time.
        now: Millis,
        /// Voice activity detection results before transcription.
        cue: SpeechCue,
    },
    /// The cortex's response to a thought requested by the core.
    CortexReply {
        /// The current time.
        now: Millis,
        /// The sequential thought identifier.
        thought: ThoughtId,
        /// The generated text from the cortex.
        text: String,
    },
    /// Playback of an utterance has begun on the audio output.
    PlaybackStarted {
        /// The current time.
        now: Millis,
        /// The utterance being played.
        utterance: UtteranceId,
    },
    /// Playback of an utterance has finished on the audio output.
    PlaybackFinished {
        /// The current time.
        now: Millis,
        /// The utterance that finished playing.
        utterance: UtteranceId,
        /// Whether the utterance was cut off: cancelled while it played, or before it
        /// started, so it was not heard to its end. The reducer treats a cut playback
        /// exactly like a finished one; the flag tells the audit what was cut. Omitted
        /// from JSON when false, so an utterance that played to its end is stored exactly
        /// as it was before the flag existed.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        interrupted: bool,
    },
    /// The owner's checklist (`CHECKLIST.md`) was read: at startup and whenever the file
    /// changes. Only whether it holds something to check crosses into the core; its text
    /// never does, so it never reaches the reducer or the soul.
    Checklist {
        /// The current time.
        now: Millis,
        /// Whether the checklist holds anything beyond blank lines, headings and empty
        /// list items: something a drive thought could bring up.
        actionable: bool,
    },
    /// A thought the core requested failed in the cortex (unreachable, timed out, or an
    /// empty answer where one was owed). A thought abandoned for a newer one, or cut off
    /// by a shutdown, did not fail and is never reported.
    CortexFailed {
        /// The current time.
        now: Millis,
        /// The thought that failed.
        thought: ThoughtId,
    },
    /// The owner told Enton to keep quiet, or released it: a typed or transcribed command
    /// ("Enton, silêncio", "Enton, pode falar") that the adapter recognized and sent in
    /// place of its speech cue, so the command itself buys no thought. Until `until`, every
    /// thought of Enton's own abstains (`Quiet`); being called by name is still answered.
    /// An `until` at or before `now` releases it.
    Quiet {
        /// The current time.
        now: Millis,
        /// When quiet mode ends on its own.
        until: Millis,
    },
    /// The owner's quiet hours (a band of local time, 23:00 to 07:00 by default) began or
    /// ended. The core has no wall clock: the adapter reads it and sends only the flag, at
    /// startup and at each edge of the band, so replay stays exact.
    QuietHours {
        /// The current time.
        now: Millis,
        /// Whether the band is on.
        active: bool,
    },
}

impl Event {
    /// The event with any speech cue or body reading in canonical form (see
    /// [`SpeechCue::canonical`] and [`BodySignals::canonical`]): what the reducer
    /// decides on and what a durable log should store.
    #[must_use]
    pub fn canonical(self) -> Self {
        match self {
            Event::Speech { now, cue } => Event::Speech {
                now,
                cue: cue.canonical(),
            },
            Event::Body { now, signals } => Event::Body {
                now,
                signals: signals.canonical(),
            },
            other => other,
        }
    }

    /// The instant at which the event occurred.
    #[must_use]
    pub fn now(&self) -> Millis {
        match self {
            Event::Tick { now }
            | Event::Body { now, .. }
            | Event::Speech { now, .. }
            | Event::CortexReply { now, .. }
            | Event::PlaybackStarted { now, .. }
            | Event::PlaybackFinished { now, .. }
            | Event::Checklist { now, .. }
            | Event::CortexFailed { now, .. }
            | Event::Quiet { now, .. }
            | Event::QuietHours { now, .. } => *now,
        }
    }
}

/// Physical body signals. Missing sensors are represented by `None`:
/// desktops may lack a battery and some boards expose no thermal zones.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct BodySignals {
    /// Highest temperature among thermal zones, in degrees Celsius.
    pub temperature_c: Option<f32>,
    /// Battery charge, from zero to one.
    pub battery: Option<f32>,
    /// CPU load normalized by the core count, from zero to one.
    pub cpu_load: f32,
}

impl BodySignals {
    /// The readings safe to reduce and to serialize. A non-finite temperature or
    /// battery becomes unknown (`None`) and a non-finite load becomes zero, which is
    /// also what JSON reads back for them, so a replayed body decides exactly like
    /// the live one. Battery and load are clamped to the unit interval.
    #[must_use]
    pub fn canonical(self) -> Self {
        let unit = |value: f32| value.clamp(0.0, 1.0);
        Self {
            temperature_c: self.temperature_c.filter(|value| value.is_finite()),
            battery: self.battery.filter(|value| value.is_finite()).map(unit),
            cpu_load: if self.cpu_load.is_finite() {
                unit(self.cpu_load)
            } else {
                0.0
            },
        }
    }
}

/// Shortest and longest vector a direction reading may be: an estimator reports a unit
/// vector, so one far from unit length is a broken reading, not a direction.
const DIRECTION_LENGTH: core::ops::RangeInclusive<f32> = 0.5..=1.5;

/// How far from one a reading's length may be for it to count as a unit vector already.
/// Normalizing leaves a length within a few units in the last place of one, far inside
/// this slack, so a canonical reading canonicalizes to itself.
const UNIT_SLACK: f32 = 1e-5;

/// `[x, y]` scaled to unit length, or `None` when it is not a direction at all: not
/// finite, or too far from unit length. IEEE 754 square root and division are correctly
/// rounded, so every platform computes the same bits.
fn unit_direction([x, y]: [f32; 2]) -> Option<[f32; 2]> {
    let length = (x * x + y * y).sqrt();
    if !DIRECTION_LENGTH.contains(&length) {
        return None;
    }
    if (length - 1.0).abs() <= UNIT_SLACK {
        Some([x, y])
    } else {
        Some([x / length, y / length])
    }
}

impl SpeechCue {
    /// The cue with every measurement safe to reduce and to serialize. Non-finite
    /// energy or VAD become zero and non-finite likelihoods become unknown (`None`),
    /// which is also what JSON reads back for them, so a replayed cue decides exactly
    /// like the live one. Finite values are clamped to the unit interval. A direction
    /// becomes unknown unless both components are finite and its length lies within
    /// 0.5 and 1.5; then it is scaled to unit length.
    #[must_use]
    pub fn canonical(self) -> Self {
        let level = |value: f32| {
            if value.is_finite() {
                value.clamp(0.0, 1.0)
            } else {
                0.0
            }
        };
        let likelihood = |value: Option<f32>| {
            value
                .filter(|value| value.is_finite())
                .map(|value| value.clamp(0.0, 1.0))
        };
        Self {
            energy: level(self.energy),
            vad_confidence: level(self.vad_confidence),
            speaker_sim: likelihood(self.speaker_sim),
            media: likelihood(self.media),
            turn_complete: likelihood(self.turn_complete),
            directed: likelihood(self.directed),
            direction: self.direction.and_then(unit_direction),
            ..self
        }
    }
}

/// Cues from voice activity detection, without a transcript.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SpeechCue {
    /// Signal energy, from zero to one.
    pub energy: f32,
    /// Duration of the speech segment in milliseconds.
    pub duration_ms: u32,
    /// Voice activity confidence, from zero to one.
    pub vad_confidence: f32,
    /// Whether the keyword (Enton's name) was detected.
    pub keyword: bool,
    /// Similarity of this voice to the owner's enrolled voiceprint, from zero to
    /// one; `None` when no speaker verification ran.
    #[serde(default)]
    pub speaker_sim: Option<f32>,
    /// Likelihood, from zero to one, that the segment is reproduced media (TV, radio,
    /// music) rather than a live voice in the room; `None` when no audio tagger ran.
    #[serde(default)]
    pub media: Option<f32>,
    /// Likelihood, from zero to one, that the speaker finished their turn with this
    /// segment (an end-of-turn model such as Smart Turn); `None` when none ran.
    #[serde(default)]
    pub turn_complete: Option<f32>,
    /// Likelihood, from zero to one, that the speech is addressed to Enton rather
    /// than to someone else in the room (a device-directedness detector); `None`
    /// when no detector ran.
    #[serde(default)]
    pub directed: Option<f32>,
    /// Direction of arrival that a microphone array estimated for the segment, as the
    /// unit vector `[cos, sin]` of its azimuth in the array's own frame; `None` when no
    /// array ran. Omitted from JSON when absent, so a cue without the sensor is stored
    /// exactly as it was before the sensor existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub direction: Option<[f32; 2]>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pointing(direction: [f32; 2]) -> Option<[f32; 2]> {
        SpeechCue {
            direction: Some(direction),
            ..SpeechCue::default()
        }
        .canonical()
        .direction
    }

    #[test]
    fn a_direction_is_scaled_to_unit_length() {
        assert_eq!(pointing([1.2, 0.0]), Some([1.0, 0.0]));
        assert_eq!(pointing([0.0, -0.6]), Some([0.0, -1.0]));
        // Length 1.25, exact in binary: each component is the nearest float to 0.6 and 0.8.
        assert_eq!(pointing([0.75, 1.0]), Some([0.6, 0.8]));
        // A reading already of unit length keeps its bits.
        let unit = [0.6_f32, 0.8];
        assert_eq!(
            pointing(unit).map(|v| v.map(f32::to_bits)),
            Some(unit.map(f32::to_bits))
        );
    }

    #[test]
    fn a_broken_direction_is_no_direction() {
        for broken in [
            [f32::NAN, 1.0],
            [1.0, f32::INFINITY],
            [f32::NEG_INFINITY, 0.0],
            [0.0, 0.0],
            [0.3, 0.3],
            [1.2, 1.2],
            [f32::MAX, 0.0],
            [-3.0, 0.0],
        ] {
            assert_eq!(pointing(broken), None, "{broken:?}");
        }
        // The length limits are inclusive.
        assert_eq!(pointing([0.5, 0.0]), Some([1.0, 0.0]));
        assert_eq!(pointing([0.0, 1.5]), Some([0.0, 1.0]));
    }

    #[test]
    fn canonical_directions_canonicalize_to_themselves() {
        // Directions every few degrees around the circle, at lengths across the whole
        // accepted range: normalizing twice must give the same bits as once.
        for step in 0..720_u16 {
            let angle = f32::from(step) * core::f32::consts::PI / 360.0;
            for length in [0.51_f32, 0.73, 0.999_9, 1.0, 1.000_2, 1.31, 1.49] {
                let once = pointing([length * angle.cos(), length * angle.sin()]).unwrap();
                let twice = pointing(once).unwrap();
                assert_eq!(
                    once.map(f32::to_bits),
                    twice.map(f32::to_bits),
                    "{angle} {length}"
                );
                let norm = (once[0] * once[0] + once[1] * once[1]).sqrt();
                assert!((norm - 1.0).abs() <= UNIT_SLACK, "{norm}");
            }
        }
    }

    #[test]
    fn a_cue_without_a_direction_serializes_as_before_the_sensor_existed() {
        let cue = SpeechCue {
            energy: 0.5,
            duration_ms: 900,
            vad_confidence: 0.8,
            ..SpeechCue::default()
        };
        let json = serde_json::to_string(&cue).unwrap();
        assert!(!json.contains("direction"), "{json}");
        assert_eq!(serde_json::from_str::<SpeechCue>(&json).unwrap(), cue);
        let pointed = SpeechCue {
            direction: Some([0.6, 0.8]),
            ..cue
        };
        let json = serde_json::to_string(&pointed).unwrap();
        assert!(json.contains(r#""direction":[0.6,0.8]"#), "{json}");
        assert_eq!(serde_json::from_str::<SpeechCue>(&json).unwrap(), pointed);
    }
}
