//! Decisions from the organism. Adapters execute them; the core only decides.

use serde::{Deserialize, Serialize};

use crate::event::Millis;

/// A sequential, deterministic thought identifier. Replay produces the same
/// identifiers, which the cortex adapter can use as idempotency keys.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct ThoughtId(pub u64);

/// Why a stimulus deserved attention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Reason {
    /// Enton was addressed by name.
    Keyword,
    /// Salient speech without the keyword.
    Speech,
    /// Speech within the active attention window following an addressed turn.
    FollowUp,
    /// An internal drive crossed the threshold, identified by name.
    Drive(String),
}

/// Why the cortex did not wake up (a counterfactual, thesis D2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Abstention {
    /// Salience did not reach the threshold.
    BelowThreshold,
    /// The previous ignition was too recent, or hysteresis has not rearmed.
    Cooldown,
    /// The budget could not pay for a thought.
    OutOfEnergy,
    /// The body is in torpor due to fever or a critical battery level.
    Torpor,
    /// Salience was suppressed by habituation to repeated similar stimuli.
    Habituation,
    /// Stimulus coincided with active playback or room reverberation and lacked barge-in evidence.
    SelfEcho,
    /// Inside an attention window, the voice did not match whoever addressed Enton.
    OtherSpeaker,
}

/// A brainstem decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Action {
    /// Wake the cortex (LLM).
    Think {
        /// Sequential identifier for this thought.
        thought: ThoughtId,
        /// Why this stimulus deserved attention.
        reason: Reason,
        /// Priority measure from 0 to 1, computed from stimulus features.
        salience: f32,
    },
    /// Speak a response.
    Speak {
        /// The text to speak, extracted from cortex reply.
        text: String,
    },
    /// Wait for a continuation turn within the attention window.
    Attend {
        /// Monotonic deadline for receiving a continuation before timing out.
        until: Millis,
    },
    /// A stimulus did not wake the cortex; retain the reason for auditing
    /// false negatives.
    Abstain {
        /// Why the stimulus was potentially interesting.
        reason: Reason,
        /// How much attention it drew, from 0 to 1.
        salience: f32,
        /// Why the cortex was not called.
        why: Abstention,
    },
}
