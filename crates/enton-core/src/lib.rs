//! Enton's brainstem (RFC 0001).
//!
//! This crate is **pure**: no I/O, clock reads, or internal randomness.
//! Inputs arrive as [`Event`] values and decisions leave as [`Action`] values.
//! Replaying the same event tape with the same profile reproduces the same
//! decisions.
//!
//! The LLM is the cortex, not the heart: the organism only requests a thought
//! ([`Action::Think`]) when ignition and the budget permit it. Abstentions
//! explain rejected stimuli so adapters can audit the cost of not thinking.

pub mod action;
pub mod drive;
pub mod energy;
pub mod event;
pub mod ignition;
pub mod organism;
pub mod ports;

pub use action::{Abstention, Action, Reason, ThoughtId};
pub use drive::{Drive, DriveTable};
pub use energy::{Budget, PriceTable};
pub use event::{BodySignals, Event, Millis, SpeechCue, UtteranceId};
pub use ignition::Ignition;
pub use organism::{Organism, PlaybackStatus, Profile, REDUCER_VERSION, contains_keyword_word};
