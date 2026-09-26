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

use std::path::{Path, PathBuf};

use enton_core::{Action, Event, ThoughtId};
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

    /// Open a log at `path`, creating the file when it does not exist yet.
    ///
    /// Existing rows are never modified here. Returns
    /// [`Error::UnsupportedSchema`] when the file was written by a newer
    /// version of this code.
    pub fn open(path: impl AsRef<Path>, config: SoulConfig) -> Result<Self, Error> {
        if config.max_snapshots == Some(0)
            && (config.max_events.is_some() || config.max_db_bytes.is_some())
        {
            return Err(Error::InvalidConfig(
                "max_snapshots cannot be 0 when pruning is enabled",
            ));
        }

        let path = path.as_ref().to_path_buf();
        let conn = Connection::open(&path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "FULL")?;
        let found = Self::schema_version(&conn)?;
        if found > Self::SCHEMA_VERSION {
            return Err(Error::UnsupportedSchema {
                found,
                expected: Self::SCHEMA_VERSION,
            });
        }
        Self::migrate(&conn)?;

        // Check for incompatible history
        {
            let mut stmt =
                conn.prepare("SELECT reducer_version FROM snapshots ORDER BY seq DESC LIMIT 1")?;
            let mut rows = stmt.query([])?;
            if let Some(row) = rows.next()? {
                let rv: u32 = row.get(0)?;
                if rv != config.reducer_version {
                    return Err(Error::IncompatibleHistory {
                        kind: "snapshot",
                        found: rv,
                        expected: config.reducer_version,
                    });
                }
            } else {
                let mut stmt =
                    conn.prepare("SELECT reducer_version FROM events ORDER BY seq DESC LIMIT 1")?;
                let mut rows = stmt.query([])?;
                if let Some(row) = rows.next()? {
                    let rv: u32 = row.get(0)?;
                    if rv != config.reducer_version {
                        return Err(Error::IncompatibleHistory {
                            kind: "event",
                            found: rv,
                            expected: config.reducer_version,
                        });
                    }
                } else {
                    let mut stmt =
                        conn.prepare("SELECT seq FROM sqlite_sequence WHERE name = 'events'")?;
                    let mut rows = stmt.query([])?;
                    if let Some(row) = rows.next()? {
                        let seq: i64 = row.get(0)?;
                        if seq > 0 {
                            return Err(Error::PrunedWithoutAnchor);
                        }
                    }
                }
            }
        }

        Ok(Self { conn, config, path })
    }

    /// The database file backing this log. The journal keeps `-wal` and
    /// `-shm` siblings next to it while the log is open.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Save an opaque state snapshot at a given sequence number.
    pub fn save_snapshot(&self, at_seq: SeqNo, blob: &[u8]) -> Result<(), Error> {
        if at_seq == 0 {
            return Err(Error::InvalidSnapshotSequence("snapshot seq cannot be 0"));
        }
        let max_seq: i64 =
            self.conn
                .query_row("SELECT IFNULL(MAX(seq), 0) FROM events", [], |row| {
                    row.get(0)
                })?;
        let max_seq_u64 = u64::try_from(max_seq).unwrap_or(0);
        if at_seq > max_seq_u64 {
            return Err(Error::InvalidSnapshotSequence("snapshot seq out of bounds"));
        }

        let seq_i64 = i64::try_from(at_seq)
            .map_err(|_| Error::Storage(rusqlite::Error::IntegralValueOutOfRange(0, i64::MAX)))?;
        self.conn.execute(
            "INSERT OR REPLACE INTO snapshots (seq, reducer_version, blob) VALUES (?1, ?2, ?3)",
            rusqlite::params![seq_i64, self.config.reducer_version, blob],
        )?;
        Ok(())
    }

    /// Return the latest snapshot (if any) matching the current reducer version.
    pub fn latest_snapshot(&self) -> Result<Option<(SeqNo, Vec<u8>)>, Error> {
        let mut stmt = self
            .conn
            .prepare("SELECT seq, blob FROM snapshots ORDER BY seq DESC LIMIT 1")?;
        let mut rows = stmt.query([])?;
        if let Some(row) = rows.next()? {
            let raw: i64 = row.get(0)?;
            let seq = u64::try_from(raw).unwrap_or(0);
            let blob: Vec<u8> = row.get(1)?;
            Ok(Some((seq, blob)))
        } else {
            Ok(None)
        }
    }

    /// Appends a new event to the soul.
    pub fn append_event(&self, event: &Event) -> Result<SeqNo, Error> {
        let at_ms = i64::try_from(event.now().0)
            .map_err(|_| Error::Storage(rusqlite::Error::IntegralValueOutOfRange(0, i64::MAX)))?;
        let payload = serde_json::to_string(event)?;
        let reducer_version = i64::from(self.config.reducer_version);
        let config_version = i64::from(self.config.config_version);
        self.conn.execute(
            "INSERT INTO events (at_ms, payload_json, reducer_version, config_version) \
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![at_ms, payload, reducer_version, config_version],
        )?;
        let rowid = self.conn.last_insert_rowid();
        let seq = u64::try_from(rowid)
            .map_err(|_| Error::Storage(rusqlite::Error::IntegralValueOutOfRange(0, rowid)))?;
        Ok(seq)
    }

    /// Record that a thought is about to run, before its effect, so a crash
    /// mid-effect leaves a row to reconcile.
    pub fn record_pending(&self, thought: ThoughtId, created_seq: SeqNo) -> Result<(), Error> {
        let thought_id = i64::try_from(thought.0)
            .map_err(|_| Error::Storage(rusqlite::Error::IntegralValueOutOfRange(0, i64::MAX)))?;
        let created_seq_i64 = i64::try_from(created_seq)
            .map_err(|_| Error::Storage(rusqlite::Error::IntegralValueOutOfRange(0, i64::MAX)))?;
        self.conn.execute(
            "INSERT INTO actions (thought_id, status, created_seq, result_json) \
             VALUES (?1, ?2, ?3, NULL)",
            rusqlite::params![thought_id, ActionStatus::Pending.as_str(), created_seq_i64],
        )?;
        Ok(())
    }

    /// Resolve a recorded thought with its successful effect result.
    pub fn mark_done(&self, thought: ThoughtId, result_json: &str) -> Result<(), Error> {
        self.mark(thought, ActionStatus::Done, result_json)
    }

    /// Resolve a recorded thought with its failure result.
    pub fn mark_failed(&self, thought: ThoughtId, result_json: &str) -> Result<(), Error> {
        self.mark(thought, ActionStatus::Failed, result_json)
    }

    /// The recorded actions that never resolved, oldest first: what a
    /// restarted process must reconcile.
    pub fn pending_actions(&self) -> Result<Vec<(ThoughtId, SeqNo)>, Error> {
        let mut stmt = self.conn.prepare(
            "SELECT thought_id, created_seq FROM actions \
             WHERE status = ?1 ORDER BY created_seq ASC, thought_id ASC",
        )?;
        let rows = stmt
            .query_map([ActionStatus::Pending.as_str()], |row| {
                let tid: i64 = row.get(0)?;
                let seq: i64 = row.get(1)?;
                Ok((
                    ThoughtId(u64::try_from(tid).unwrap_or(0)),
                    u64::try_from(seq).unwrap_or(0),
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Read up to `limit` events after `cursor`, in sequence order.
    ///
    /// Returns [`Error::ReplayGap`] when the first event read is not the one
    /// immediately after the cursor, so a truncated log is never replayed
    /// silently.
    pub fn read_after(&self, cursor: SeqNo, limit: usize) -> Result<Vec<(SeqNo, Event)>, Error> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let cursor_i64 = i64::try_from(cursor)
            .map_err(|_| Error::Storage(rusqlite::Error::IntegralValueOutOfRange(0, i64::MAX)))?;
        let limit_i64 = i64::try_from(limit)
            .map_err(|_| Error::Storage(rusqlite::Error::IntegralValueOutOfRange(0, i64::MAX)))?;
        let mut stmt = self.conn.prepare(
            "SELECT seq, payload_json, reducer_version FROM events WHERE seq > ?1 ORDER BY seq ASC LIMIT ?2",
        )?;
        let mut rows = stmt.query(rusqlite::params![cursor_i64, limit_i64])?;
        let mut out = Vec::with_capacity(limit.min(Self::REPLAY_PAGE));
        let mut expected = cursor.saturating_add(1);
        while let Some(row) = rows.next()? {
            let raw: i64 = row.get(0)?;
            let payload: String = row.get(1)?;
            let rv: u32 = row.get(2)?;
            if rv != self.config.reducer_version {
                return Err(Error::IncompatibleHistory {
                    kind: "event",
                    found: rv,
                    expected: self.config.reducer_version,
                });
            }
            let seq = u64::try_from(raw).unwrap_or(0);
            if seq != expected {
                return Err(Error::ReplayGap {
                    expected,
                    found: seq,
                });
            }
            out.push((seq, serde_json::from_str(&payload)?));
            expected = seq.saturating_add(1);
        }
        Ok(out)
    }

    /// Feed every stored event, in order, into a fresh organism and return its
    /// actions. Deterministic for the same log and the same profile.
    pub fn replay<S, I, R, E>(
        &self,
        init: I,
        restore: R,
        mut step: E,
    ) -> Result<(S, Vec<Action>), Error>
    where
        I: FnOnce() -> S,
        R: FnOnce(&[u8]) -> Result<S, Error>,
        E: FnMut(&mut S, &Event) -> Vec<Action>,
    {
        let latest = self.latest_snapshot()?;
        let (mut cursor, mut state) = if let Some((seq, blob)) = latest {
            (seq, restore(&blob)?)
        } else {
            (0, init())
        };

        let mut out = Vec::new();
        loop {
            let page = self.read_after(cursor, Self::REPLAY_PAGE)?;
            if page.is_empty() {
                return Ok((state, out));
            }
            for (_, event) in &page {
                out.extend(step(&mut state, event));
            }
            cursor = page.last().map_or(cursor, |(seq, _)| *seq);
            if page.len() < Self::REPLAY_PAGE {
                return Ok((state, out));
            }
        }
    }

    /// Enforce the retention and size caps by dropping the oldest events that
    /// are not needed for crash recovery. Meets caps or returns an error.
    pub fn retain(&self) -> Result<usize, Error> {
        if self.config.max_events.is_none()
            && self.config.max_db_bytes.is_none()
            && self.config.max_snapshots.is_none()
        {
            return Ok(0);
        }
        let target_events = self.config.max_events.unwrap_or(usize::MAX);
        let target_bytes = self.config.max_db_bytes.unwrap_or(u64::MAX);

        let latest_snap = self.latest_snapshot()?;
        let max_droppable_seq = latest_snap.map_or(0, |(seq, _)| seq);

        let mut snapshots_deleted = 0;
        if let Some(target_snapshots) = self.config.max_snapshots {
            let target_i64 = i64::try_from(target_snapshots).unwrap_or(i64::MAX);
            snapshots_deleted = self.conn.execute(
                "DELETE FROM snapshots WHERE reducer_version = ?1 AND seq NOT IN (
                    SELECT seq FROM snapshots WHERE reducer_version = ?1 ORDER BY seq DESC LIMIT ?2
                )",
                rusqlite::params![self.config.reducer_version, target_i64],
            )?;
        }

        let mut deleted = 0;

        loop {
            let current_count = self.event_count()?;
            let current_size = self.db_size_bytes()?;

            if current_count <= target_events && current_size <= target_bytes {
                break;
            }

            let over_events = current_count.saturating_sub(target_events);
            let limit = if over_events > 0 {
                over_events
            } else {
                Self::PRUNE_BATCH
            }
            .min(Self::PRUNE_BATCH);
            let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
            let max_seq_i64 = i64::try_from(max_droppable_seq).unwrap_or(i64::MAX);

            let n = self.conn.execute(
                "DELETE FROM events WHERE seq IN (
                    SELECT seq FROM events
                    WHERE seq <= ?1
                    ORDER BY seq ASC
                    LIMIT ?2
                )",
                rusqlite::params![max_seq_i64, limit_i64],
            )?;

            if n == 0 {
                break;
            }
            deleted += n;
        }

        if deleted > 0 || snapshots_deleted > 0 {
            // Checkpoint and VACUUM to reclaim space
            let busy: i32 = self
                .conn
                .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |row| row.get(0))?;
            if busy != 0 {
                return Err(Error::CheckpointBusy);
            }
            self.conn.execute("VACUUM", [])?;
        }

        let final_count = self.event_count()?;
        let final_size = self.db_size_bytes()?;

        if final_count > target_events {
            return Err(Error::RetentionCap(format!(
                "could not meet event cap of {target_events}; {final_count} events remain because they are pinned by lack of snapshots"
            )));
        }
        if final_size > target_bytes {
            return Err(Error::RetentionCap(format!(
                "could not meet size cap of {target_bytes} bytes; file is {final_size} bytes (vacuum failed or data pinned)"
            )));
        }

        Ok(deleted)
    }

    fn event_count(&self) -> Result<usize, Error> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM events", [], |row| row.get(0))?;
        Ok(usize::try_from(n.max(0)).unwrap_or(0))
    }

    fn mark(
        &self,
        thought: ThoughtId,
        status: ActionStatus,
        result_json: &str,
    ) -> Result<(), Error> {
        let thought_id = i64::try_from(thought.0)
            .map_err(|_| Error::Storage(rusqlite::Error::IntegralValueOutOfRange(0, i64::MAX)))?;
        let updated = self.conn.execute(
            "UPDATE actions SET status = ?1, result_json = ?2 WHERE thought_id = ?3",
            rusqlite::params![status.as_str(), result_json, thought_id],
        )?;
        if updated == 0 {
            return Err(Error::UnknownAction(thought));
        }
        Ok(())
    }

    fn db_size_bytes(&self) -> Result<u64, Error> {
        let mut total = std::fs::metadata(&self.path)?.len();
        let name = self
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        let parent = self
            .path
            .parent()
            .map_or(String::new(), |p| p.to_string_lossy().into_owned());
        let wal = format!("{parent}/{name}-wal");
        if let Ok(meta) = std::fs::metadata(&wal) {
            total += meta.len();
        }
        let shm = format!("{parent}/{name}-shm");
        if let Ok(meta) = std::fs::metadata(&shm) {
            total += meta.len();
        }
        Ok(total)
    }

    fn schema_version(conn: &Connection) -> Result<u32, Error> {
        let version: u32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        Ok(version)
    }

    fn migrate(conn: &Connection) -> Result<(), Error> {
        let mut current_version = Self::schema_version(conn)?;
        if current_version == Self::SCHEMA_VERSION {
            return Ok(());
        }

        if current_version == 0 {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS events (
                     seq INTEGER PRIMARY KEY AUTOINCREMENT,
                     at_ms INTEGER NOT NULL,
                     payload_json TEXT NOT NULL,
                     reducer_version INTEGER NOT NULL,
                     config_version INTEGER NOT NULL
                 );
                 CREATE TABLE IF NOT EXISTS actions (
                     thought_id INTEGER PRIMARY KEY,
                     status TEXT NOT NULL CHECK (status IN ('pending', 'done', 'failed')),
                     created_seq INTEGER NOT NULL,
                     result_json TEXT
                 );
                 CREATE INDEX IF NOT EXISTS idx_actions_pending
                     ON actions (status, created_seq);
                 PRAGMA auto_vacuum = FULL;
                 VACUUM;",
            )?;
            current_version = 1;
        }

        if current_version == 1 {
            conn.execute_batch(
                "CREATE TABLE events_v2 (
                     seq INTEGER PRIMARY KEY AUTOINCREMENT,
                     at_ms INTEGER NOT NULL,
                     payload_json TEXT NOT NULL,
                     reducer_version INTEGER NOT NULL,
                     config_version INTEGER NOT NULL
                 );
                 INSERT INTO events_v2 (seq, at_ms, payload_json, reducer_version, config_version)
                     SELECT seq, at_ms, payload_json, reducer_version, config_version FROM events;
                 DROP TABLE events;
                 ALTER TABLE events_v2 RENAME TO events;
                 PRAGMA auto_vacuum = FULL;
                 VACUUM;",
            )?;
            current_version = 2;
        }

        if current_version == 2 {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS snapshots (
                     seq INTEGER PRIMARY KEY,
                     reducer_version INTEGER NOT NULL,
                     blob BLOB NOT NULL
                 );",
            )?;
        }

        conn.pragma_update(None, "user_version", Self::SCHEMA_VERSION)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn pruned_without_anchor_is_rejected() {
        let path = temp_path("pruned_no_anchor");

        {
            let config = SoulConfig::default();
            let soul = Soul::open(&path, config).expect("open");
            soul.append_event(&tick(100)).expect("append");
            // we delete the events manually and advance the sqlite_sequence to simulate a pruned history without snapshots
            soul.conn.execute("DELETE FROM events", []).unwrap();
        }

        let config2 = SoulConfig::default();
        let err = Soul::open(&path, config2).unwrap_err();
        assert!(matches!(err, Error::PrunedWithoutAnchor));
    }

    #[test]
    fn reject_zero_snapshot_cap_with_pruning() {
        let path = temp_path("zero_snap_cap");
        let config = SoulConfig {
            max_snapshots: Some(0),
            max_events: Some(10), // pruning enabled
            ..SoulConfig::default()
        };
        let err = Soul::open(&path, config).unwrap_err();
        assert!(matches!(err, Error::InvalidConfig(_)));
    }

    #[test]
    fn incompatible_history_is_rejected() {
        let path = temp_path("incompatible_history");

        {
            let config = SoulConfig {
                reducer_version: 1,
                ..SoulConfig::default()
            };
            let soul = Soul::open(&path, config).expect("open");
            soul.append_event(&tick(100)).expect("append");
            soul.save_snapshot(1, b"snap").expect("save");
        }

        // Reopen with different version should fail
        let config2 = SoulConfig {
            reducer_version: 2,
            ..SoulConfig::default()
        };
        let err = Soul::open(&path, config2).unwrap_err();
        assert!(matches!(err, Error::IncompatibleHistory { .. }));
    }

    #[test]
    fn snapshot_rejects_future_sequence() {
        let path = temp_path("snapshot_future_seq");
        let soul = Soul::open(&path, SoulConfig::default()).expect("open");

        soul.append_event(&tick(100)).expect("append 1");

        // Attempting to snapshot at seq 2 (future) should fail
        let err = soul.save_snapshot(2, b"blob").unwrap_err();
        assert!(matches!(err, Error::InvalidSnapshotSequence(_)));

        // Attempting to snapshot at seq 0 should also fail
        let err_zero = soul.save_snapshot(0, b"blob").unwrap_err();
        assert!(matches!(err_zero, Error::InvalidSnapshotSequence(_)));

        // Valid snapshot
        soul.save_snapshot(1, b"blob").expect("save 1");

        cleanup(&path);
    }

    #[test]
    fn retain_exceeds_prune_batch_budget() {
        let path = temp_path("retain_budget");
        let config = SoulConfig {
            max_events: Some(10),
            ..SoulConfig::default()
        };
        let soul = Soul::open(&path, config).expect("open");

        // Add more than PRUNE_BATCH (512) events, e.g., 600
        for i in 1..=600 {
            soul.append_event(&tick(i)).expect("append");
        }

        // Save snapshot at 600
        soul.save_snapshot(600, b"snap").expect("save");

        // Retain should delete 590 events, which requires more than one batch of 512
        let deleted = soul.retain().expect("retain");
        assert_eq!(deleted, 590);
        assert_eq!(soul.event_count().unwrap(), 10);

        cleanup(&path);
    }

    #[test]
    fn snapshots_are_pruned() {
        let path = temp_path("snapshots_prune");
        let config = SoulConfig {
            max_snapshots: Some(2),
            ..SoulConfig::default()
        };
        let soul = Soul::open(&path, config).expect("open");

        for i in 1..=4 {
            soul.append_event(&tick(i * 100)).expect("append");
            soul.save_snapshot(i, b"snap").expect("save");
        }

        // Before retain, we have 4 snapshots
        let snap_count: i64 = soul
            .conn
            .query_row("SELECT COUNT(*) FROM snapshots", [], |r| r.get(0))
            .unwrap();
        assert_eq!(snap_count, 4);

        soul.retain().expect("retain");

        // After retain, only the latest 2 should remain (seq 3 and 4)
        let snap_count: i64 = soul
            .conn
            .query_row("SELECT COUNT(*) FROM snapshots", [], |r| r.get(0))
            .unwrap();
        assert_eq!(snap_count, 2);

        let seqs: Vec<i64> = soul
            .conn
            .prepare("SELECT seq FROM snapshots ORDER BY seq ASC")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(seqs, vec![3, 4]);

        cleanup(&path);
    }

    #[test]
    fn migration_sets_auto_vacuum_full() {
        let path = temp_path("auto_vacuum");
        let soul = Soul::open(&path, SoulConfig::default()).expect("open");

        let auto_vacuum: i32 = soul
            .conn
            .query_row("PRAGMA auto_vacuum", [], |r| r.get(0))
            .unwrap();
        assert_eq!(auto_vacuum, 1, "auto_vacuum should be 1 (FULL)");

        cleanup(&path);
    }

    use super::*;
    use enton_core::{Millis, SpeechCue};
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_path(tag: &str) -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("enton_soul_{id}_{tag}_{}", std::process::id()))
    }

    fn cleanup(path: &Path) {
        for suffix in ["", "-wal", "-shm"] {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            let parent = path
                .parent()
                .map_or(String::new(), |p| p.to_string_lossy().into_owned());
            if let Err(_err) = std::fs::remove_file(format!("{parent}/{name}{suffix}")) {}
        }
    }

    fn tick(now: u64) -> Event {
        Event::Tick { now: Millis(now) }
    }

    fn keyword_speech(now: u64) -> Event {
        Event::Speech {
            now: Millis(now),
            cue: SpeechCue {
                energy: 0.8,
                duration_ms: 900,
                vad_confidence: 0.9,
                keyword: true,
            },
        }
    }

    fn seqs(path: &Path) -> Vec<SeqNo> {
        let conn = Connection::open(path).expect("reopen for read");
        let mut stmt = conn
            .prepare("SELECT seq FROM events ORDER BY seq ASC")
            .expect("prepare");
        let rows: Vec<i64> = stmt
            .query_map([], |row| row.get(0))
            .expect("query")
            .collect::<Result<Vec<_>, _>>()
            .expect("collect");
        rows.into_iter()
            .map(|s| u64::try_from(s).unwrap_or(0))
            .collect()
    }

    // Fake state for replay tests
    #[derive(Debug, PartialEq, Eq, Default)]
    struct FakeState {
        ticks: u64,
        speech_count: u64,
    }

    #[test]
    fn replay_generic_with_fake_state() {
        let path = temp_path("replay_generic");
        let soul = Soul::open(&path, SoulConfig::default()).expect("open");

        soul.append_event(&tick(100)).expect("append");
        soul.append_event(&tick(200)).expect("append");
        soul.append_event(&keyword_speech(300)).expect("append");

        let init = || FakeState::default();
        let restore = |_blob: &[u8]| -> Result<FakeState, Error> {
            Ok(FakeState {
                ticks: 99,
                speech_count: 99,
            }) // not used since no snapshot
        };
        let step = |state: &mut FakeState, event: &Event| -> Vec<Action> {
            match event {
                Event::Tick { .. } => state.ticks += 1,
                Event::Speech { .. } => state.speech_count += 1,
                _ => {}
            }
            Vec::new()
        };

        let (state, actions) = soul.replay(init, restore, step).expect("replay");
        assert_eq!(
            state,
            FakeState {
                ticks: 2,
                speech_count: 1
            }
        );
        assert!(actions.is_empty());

        cleanup(&path);
    }

    #[test]
    fn replay_from_snapshot() {
        let path = temp_path("snapshot_replay");
        let soul = Soul::open(&path, SoulConfig::default()).expect("open");

        soul.append_event(&tick(100)).expect("append"); // seq 1
        let snap_seq = soul.append_event(&tick(200)).expect("append"); // seq 2

        // Save snapshot at seq 2
        soul.save_snapshot(snap_seq, b"some_blob").expect("save");

        soul.append_event(&tick(300)).expect("append"); // seq 3
        soul.append_event(&keyword_speech(400)).expect("append"); // seq 4

        let init = || FakeState::default();
        let restore = |blob: &[u8]| -> Result<FakeState, Error> {
            assert_eq!(blob, b"some_blob");
            Ok(FakeState {
                ticks: 2,
                speech_count: 0,
            }) // restored state representing 2 ticks
        };
        let step = |state: &mut FakeState, event: &Event| -> Vec<Action> {
            match event {
                Event::Tick { .. } => state.ticks += 1,
                Event::Speech { .. } => state.speech_count += 1,
                _ => {}
            }
            Vec::new()
        };

        let (state, actions) = soul.replay(init, restore, step).expect("replay");
        // State should have 2 from snapshot + 1 tick from tail (seq 3) + 1 speech from tail (seq 4)
        assert_eq!(
            state,
            FakeState {
                ticks: 3,
                speech_count: 1
            }
        );
        assert!(actions.is_empty());

        cleanup(&path);
    }

    #[test]
    fn replay_empty_tail_after_snapshot() {
        let path = temp_path("empty_tail");
        let soul = Soul::open(&path, SoulConfig::default()).expect("open");

        let snap_seq = soul.append_event(&tick(100)).expect("append"); // seq 1
        soul.save_snapshot(snap_seq, b"snap").expect("save");

        // No events after snapshot (empty tail)

        let init = || FakeState::default();
        let restore = |_blob: &[u8]| -> Result<FakeState, Error> {
            Ok(FakeState {
                ticks: 10,
                speech_count: 10,
            })
        };
        let step = |_state: &mut FakeState, _event: &Event| -> Vec<Action> { Vec::new() };

        let (state, _actions) = soul.replay(init, restore, step).expect("replay");
        assert_eq!(
            state,
            FakeState {
                ticks: 10,
                speech_count: 10
            }
        );

        cleanup(&path);
    }

    #[test]
    fn retention_prunes_only_before_snapshot() {
        let path = temp_path("retention_prune");
        let config = SoulConfig {
            max_events: Some(2), // tiny cap to force pruning
            ..SoulConfig::default()
        };
        let soul = Soul::open(&path, config).expect("open");

        soul.append_event(&tick(100)).expect("append"); // seq 1
        soul.append_event(&tick(200)).expect("append"); // seq 2
        soul.append_event(&tick(300)).expect("append"); // seq 3
        soul.append_event(&tick(400)).expect("append"); // seq 4

        // Without snapshot, nothing is pruned because max_droppable_seq = 0
        let res = soul.retain();
        assert!(matches!(res, Err(Error::RetentionCap(_))));
        assert_eq!(seqs(&path), vec![1, 2, 3, 4]);

        // Take snapshot at seq 3
        soul.save_snapshot(3, b"snap").expect("save snapshot");

        // Now retain can prune up to seq 3
        let deleted = soul.retain().expect("retain success");
        // Target is 2 events. Total was 4. So it deletes seq 1 and seq 2.
        assert_eq!(deleted, 2);
        assert_eq!(seqs(&path), vec![3, 4]);

        cleanup(&path);
    }

    #[test]
    fn retention_fails_if_cap_cannot_be_met() {
        let path = temp_path("retention_fail");
        let config = SoulConfig {
            max_events: Some(1), // cap is 1
            ..SoulConfig::default()
        };
        let soul = Soul::open(&path, config).expect("open");

        soul.append_event(&tick(100)).expect("append"); // 1
        soul.append_event(&tick(200)).expect("append"); // 2
        soul.append_event(&tick(300)).expect("append"); // 3

        // Snapshot at seq 1
        soul.save_snapshot(1, b"snap").expect("save");

        // Can only drop up to seq 1. We have 3 events, cap is 1. We need to drop 2, but can only drop 1.
        let res = soul.retain();
        assert!(matches!(res, Err(Error::RetentionCap(_))));

        // Seq 1 should be deleted, leaving 2 and 3.
        assert_eq!(seqs(&path), vec![2, 3]);

        cleanup(&path);
    }

    #[test]
    fn v1_snapshot_is_rejected_as_incompatible_by_v2_default() {
        let path = temp_path("v1_snapshot_incompatible");
        // Open soul with explicit reducer_version = 1 (simulating organism v1)
        let v1_config = SoulConfig {
            reducer_version: 1,
            ..SoulConfig::default()
        };
        let soul_v1 = Soul::open(&path, v1_config).expect("open v1");
        let seq = soul_v1.append_event(&tick(100)).expect("append");
        soul_v1
            .save_snapshot(seq, b"v1_state")
            .expect("save v1 snapshot");
        drop(soul_v1);

        // Opening with default config (v2) must reject the v1 snapshot
        let err = Soul::open(&path, SoulConfig::default()).expect_err("should reject v1 snapshot");
        assert!(matches!(
            err,
            Error::IncompatibleHistory {
                kind: "snapshot",
                found: 1,
                expected: 2,
            }
        ));

        cleanup(&path);
    }
}
