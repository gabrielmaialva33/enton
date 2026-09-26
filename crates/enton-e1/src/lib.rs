//! Versioned, causal E1 cue benchmark (RFC 0001 §7).

pub mod baseline;
mod economy;
mod invariants;
pub mod report;
pub mod run;
mod scoring;
pub mod synth;
pub mod tape;

pub use economy::Economy;
pub use report::{
    Criterion, CriterionSummary, NoiseBreakdown, NoiseReason, NoiseStimulus, Report, Status,
    Summary, clopper_pearson_upper_bound,
};
pub use run::{Controller, ControllerResult, ExperimentRun, run_tape};
pub use synth::{SplitMix64, e1a, e1b};
pub use tape::{
    Annotation, EpisodeId, Interval, Record, SegmentId, Stimulus, Tape, TapeKind, Turn, TurnId,
};

/// Version of distributions, feedback, attribution and the report contract.
pub const BENCHMARK_VERSION: &str = "2.2.1";

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
