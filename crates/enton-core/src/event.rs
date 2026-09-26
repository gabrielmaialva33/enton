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
    },
}

impl Event {
    /// The instant at which the event occurred.
    #[must_use]
    pub fn now(&self) -> Millis {
        match self {
            Event::Tick { now }
            | Event::Body { now, .. }
            | Event::Speech { now, .. }
            | Event::CortexReply { now, .. }
            | Event::PlaybackStarted { now, .. }
            | Event::PlaybackFinished { now, .. } => *now,
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
    /// Similarity of this voice to whoever last addressed Enton by name, from
    /// zero to one; `None` when no speaker verification ran.
    #[serde(default)]
    pub speaker_sim: Option<f32>,
}
