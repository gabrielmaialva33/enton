//! Durable event log: the organism's soul (RFC 0001 P4).
//!
//! This module adapts the durable event-log discipline of `IronClaw`'s event
//! store, an Apache-2.0-licensed implementation from the nearAI project:
//! <https://github.com/nearai/ironclaw/tree/main/crates/events/ironclaw_event_store>
//!
//! One connection, one writer, WAL journal mode, synchronous=FULL. The store
//! keeps only reduced cues and decisions (no raw media, RFC 0001 section 6);
//! the event tape is replayable, gap-detectable, and bounded by retention and
//! a database size cap.

use std::path::PathBuf;

use enton_core::ThoughtId;
use rusqlite::Connection;

/// Monotonic position of an event in the log. The first appended event
/// returns one; every later append returns a strictly larger value.
pub type SeqNo = u64;

/// Lifecycle status of a recorded thought action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionStatus {
    /// The action is recorded before executing its side-effect.
    Pending,
    /// The action's side-effect completed successfully.
    Done,
    /// The action's side-effect failed.
    Failed,
}

impl ActionStatus {
    /// Returns the database string representation for this status.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Done => "done",
            Self::Failed => "failed",
        }
    }
}

/// Errors surfaced by the durable log.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A database operation failed.
    #[error("database operation failed: {0}")]
    Storage(#[from] rusqlite::Error),
    /// Storing or replaying an event payload failed.
    #[error("event serialization failed: {0}")]
    Json(#[from] serde_json::Error),
    /// Measuring the database file failed.
    #[error("I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// The stored events are no longer contiguous after the cursor.
    #[error("replay gap: expected sequence number {expected}, found {found}")]
    ReplayGap {
        /// The first sequence number the log should contain.
        expected: SeqNo,
        /// The first sequence number it actually contains.
        found: SeqNo,
    },
    /// A thought that was never recorded got resolved.
    #[error("thought {0:?} was never recorded")]
    UnknownAction(ThoughtId),
    /// The database schema was not written by this version of the code.
    #[error("unsupported schema version {found}, expected {expected}")]
    UnsupportedSchema {
        /// Schema version found in the database file.
        found: u32,
        /// Schema version this code expects.
        expected: u32,
    },
    #[error("retention failed to meet cap: {0}")]
    /// Retention failed to meet its configured caps.
    RetentionCap(String),
    /// The soul configuration is invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),
    /// The file is damaged in a way that opening it would silently hide.
    #[error("damaged soul: {0}")]
    Damaged(&'static str),
    /// The database contains events or snapshots from an incompatible reducer version.
    #[error("incompatible {kind} reducer version: found {found}, expected {expected}")]
    IncompatibleHistory {
        /// What kind of record had the incompatible version ("snapshot" or "event").
        kind: &'static str,
        /// The reducer version found in the database.
        found: u32,
        /// The reducer version expected by the configuration.
        expected: u32,
    },
    /// The database has no snapshots and no events, but the sequence number is advanced.
    #[error("incompatible history: pruned database with no snapshots and no events")]
    PrunedWithoutAnchor,
    /// A snapshot was attempted with an invalid sequence number.
    #[error("invalid snapshot sequence: {0}")]
    InvalidSnapshotSequence(&'static str),
    /// The database checkpoint is busy.
    #[error("database checkpoint is busy")]
    CheckpointBusy,
    /// The organism profile failed validation.
    #[error("invalid organism profile: {0}")]
    InvalidProfile(#[from] enton_core::InvalidProfile),
}

/// The organism's durable memory: a single-writer, append-only event log with
/// crash-reconciliation for recorded actions.
///
/// `rusqlite::Connection` is not `Sync`, so every operation takes exclusive
/// access and the single-writer discipline is enforced by the type system.
#[derive(Debug)]
pub struct Soul {
    conn: Connection,
    config: SoulConfig,
    path: PathBuf,
}

/// Retention and size policy for a [`Soul`] instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoulConfig {
    /// Version of the reducer that consumed the stored events.
    pub reducer_version: u32,
    /// Version of the hardware and cognition policy that produced the events.
    pub config_version: u32,
    /// Maximum number of events kept by retention; `None` disables the cap.
    pub max_events: Option<usize>,
    /// Maximum size of the database file, in bytes; `None` disables the cap.
    pub max_db_bytes: Option<u64>,
    /// Maximum number of snapshots kept; `None` disables the cap.
    pub max_snapshots: Option<usize>,
}

impl Default for SoulConfig {
    fn default() -> Self {
        Self {
            reducer_version: enton_core::REDUCER_VERSION,
            config_version: 1,
            max_events: None,
            max_db_bytes: None,
            max_snapshots: None,
        }
    }
}

impl Soul {
    const SCHEMA_VERSION: u32 = 3;
    const PRUNE_BATCH: usize = 512;
    const REPLAY_PAGE: usize = 1024;
}

mod schema;
mod snapshot;
mod store;

#[cfg(test)]
mod tests;
