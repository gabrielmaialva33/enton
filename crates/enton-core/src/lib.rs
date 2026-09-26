//! Enton's brainstem (RFC 0001).
//!
//! This crate is **pure**: no I/O, clock reads, or entropy of its own.
//! Inputs arrive as [`Event`] values and decisions leave as [`Action`] values.
//! Replaying the same event tape with the same profile reproduces the same
//! decisions: even exploration's coin flips come from a generator seeded by the
//! profile and kept in the organism's state (see [`ExplorationPolicy`]).
//!
//! The LLM is the cortex, not the heart: the organism only requests a thought
//! ([`Action::Think`]) when ignition and the budget permit it. Abstentions
//! explain rejected stimuli so adapters can audit the cost of not thinking.

pub mod action;
pub mod drive;
pub mod energy;
pub mod event;
pub mod evidence;
pub mod ignition;
pub mod organism;
pub mod ports;
pub mod profile;

pub use action::{Abstention, Action, Reason, ThoughtId};
pub use drive::{Drive, DriveTable};
pub use energy::{Budget, PriceTable};
pub use event::{BodySignals, Event, Millis, SpeechCue, UtteranceId};
pub use evidence::{
    DirectedModel, DirectionModel, Evidence, Senses, SourceModel, TurnModel, VoiceModel,
};
pub use ignition::Ignition;
pub use organism::{Organism, PlaybackStatus, REDUCER_VERSION, contains_keyword_word};
pub use profile::{
    AttentionPolicy, BodyLimits, BudgetPolicy, EchoPolicy, ExplorationPolicy, HabituationPolicy,
    IgnitionPolicy, InvalidProfile, Profile, SaliencePolicy, SourcePolicy, TvCautionConfinement,
};
