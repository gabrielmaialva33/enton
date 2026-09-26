//! Versioned, causal E1 cue benchmark (RFC 0001 §7).

pub mod baseline;
mod economy;
mod invariants;
pub mod offpolicy;
pub mod report;
pub mod run;
mod scoring;
pub mod synth;
pub mod tape;

pub use economy::Economy;
pub use report::{
    Criterion, CriterionSummary, NoiseBreakdown, NoiseReason, NoiseStimulus, Report, Status,
    Summary, WasteBreakdown, WasteKey, WasteReason, WasteStimulus, clopper_pearson_upper_bound,
};
pub use run::{Controller, ControllerResult, ExperimentRun, Sensors, run_tape, run_tape_with};
pub use scoring::Tally;
pub use synth::{SplitMix64, TurnRole, e1a, e1b};
pub use tape::{
    Annotation, ConditionKey, Distance, EpisodeId, Interval, Record, RoomCondition, SegmentId,
    Stimulus, Tape, TapeKind, Turn, TurnId, TvBackground, TvContent,
};

/// Version of distributions, feedback, attribution and the report contract.
/// Version 3.0.0 adds asides, in-window and adjacent distractors; populations differ from 2.x;
/// sensor readings follow measured models with per-tape, per-block and per-turn correlated errors.
/// Version 3.1.0 adds a simulated device-directedness reading to every speech cue, drawn on its
/// own random stream so no other reading moved; runs withhold it unless asked (see [`Sensors`]).
/// Version 3.2.0 adds off-policy evaluation ([`offpolicy`]): the organism as an exploring
/// logging policy and a family of threshold candidates estimated from its log. Tapes and
/// the default runs are unchanged.
/// Version 3.3.0 adds a simulated direction of arrival from a microphone array to every
/// speech cue, drawn on its own random stream so no other reading moved; runs withhold it
/// unless asked, and then decide every call exactly as 3.2.0 did.
/// Version 3.4.0 opens every run with the owner's checklist holding something to check
/// (see [`run::CHECKLIST_ACTIONABLE`]), and lets a tape read the checklist itself. Generated tapes
/// are unchanged and carry no checklist reading of their own.
/// Version 3.5.0 lets a tape carry the owner's quiet commands and the quiet hours (`Quiet` and
/// `QuietHours` records), and records the drive whose deferred intent rides a paid thought.
/// Generated tapes carry neither, and no generated tape readies a drive, so every run decides
/// every call exactly as 3.4.0 did.
pub const BENCHMARK_VERSION: &str = "3.5.0";

/// Invalid experiments fail explicitly rather than dropping stimuli or granting free calls.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An annotation, timestamp, population or configuration is inconsistent.
    #[error("invalid benchmark input: {0}")]
    Invalid(String),
    /// The organism broke one of its reducer invariants: a bug, never bad luck.
    #[error("organism invariant violated: {0}")]
    Invariant(String),
    /// An explicit resource bound was exceeded.
    #[error("benchmark limit exceeded: {0}")]
    Limit(&'static str),
    /// The core requested work its identical external account could not fund.
    #[error("organism budget invariant: balance {balance}, requested cost {cost}")]
    BudgetInvariant {
        /// Remaining common-account units.
        balance: f64,
        /// Required common-account units.
        cost: f64,
    },
    /// Reading or writing a tape failed.
    #[error("tape I/O: {0}")]
    Io(#[from] std::io::Error),
    /// A tape could not be encoded or decoded.
    #[error("tape JSON: {0}")]
    Json(#[from] serde_json::Error),
}
