//! Appending, resolving, reading, replaying and pruning the event log.

use std::path::Path;

use enton_core::{Action, Event, ThoughtId};
use rusqlite::Connection;

use super::{ActionStatus, Error, SeqNo, Soul, SoulConfig};

impl Soul {
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

    /// Appends a new event to the soul.
    pub fn append_event(&self, event: &Event) -> Result<SeqNo, Error> {
        let at_ms = i64::try_from(event.now().0)
            .map_err(|_| Error::Storage(rusqlite::Error::IntegralValueOutOfRange(0, i64::MAX)))?;
        // Canonical cues only: a non-finite measurement would serialize as `null` and
        // make the event unreadable (or read back differently) on replay.
        let payload = serde_json::to_string(&event.clone().canonical())?;
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

    pub(super) fn event_count(&self) -> Result<usize, Error> {
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
}
