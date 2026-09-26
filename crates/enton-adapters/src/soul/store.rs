//! Appending, resolving, reading, replaying and pruning the event log.

use std::path::Path;

use enton_core::{Action, Event, ThoughtId};
use rusqlite::{Connection, OpenFlags};

use super::chain::{self, EVENT_CHECKSUM, EVENT_COLUMNS, StoredEvent, sql_seq};
use super::{ActionStatus, Error, SeqNo, Soul, SoulConfig};

/// Size of the header every SQLite database file starts with.
const SQLITE_HEADER_BYTES: u64 = 100;

impl Soul {
    /// Open a log at `path`, creating the file when it does not exist yet.
    ///
    /// Existing rows are only rewritten by a schema migration, which runs in one
    /// transaction; migrating to schema 4 computes the checksums of the rows
    /// already stored, trusting them as they stand. Returns
    /// [`Error::UnsupportedSchema`] when the file was written by a newer
    /// version of this code, and [`Error::Truncated`] when events recorded at
    /// the end of the log are gone.
    pub fn open(path: impl AsRef<Path>, config: SoulConfig) -> Result<Self, Error> {
        if config.max_snapshots == Some(0)
            && (config.max_events.is_some() || config.max_db_bytes.is_some())
        {
            return Err(Error::InvalidConfig(
                "max_snapshots cannot be 0 when pruning is enabled",
            ));
        }

        let path = path.as_ref().to_path_buf();
        // SQLite reads a file shorter than its header as an empty database and
        // would write a fresh schema over it: a soul cut that short lost its history.
        if let Ok(meta) = std::fs::metadata(&path)
            && (1..SQLITE_HEADER_BYTES).contains(&meta.len())
        {
            return Err(Error::Damaged("the file is shorter than a database header"));
        }
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
        Self::check_history(&conn, &config)?;

        Ok(Self { conn, config, path })
    }

    /// Open an existing log at `path` for reading only, to audit it.
    ///
    /// Nothing is created, migrated or written, so this is safe while Enton runs:
    /// in WAL mode a reader never blocks the writer. A log closed cleanly (no
    /// `-wal` file beside it, so no writer has it open) is opened immutable, which
    /// leaves no `-wal` or `-shm` behind; a writer that opens it during the read
    /// only appends to its own `-wal`, which this reader then does not see.
    /// Appends, resolutions, snapshots and retention through the returned handle
    /// fail with a storage error. Returns a storage error when the file does not
    /// exist and [`Error::UnsupportedSchema`] unless the file has exactly this
    /// version's schema (an older one needs the migration [`Soul::open`] performs).
    pub fn open_read_only(path: impl AsRef<Path>, config: SoulConfig) -> Result<Self, Error> {
        let path = path.as_ref().to_path_buf();
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        let mut wal = path.clone().into_os_string();
        wal.push("-wal");
        let immutable = if Path::new(&wal).exists() {
            None
        } else {
            immutable_uri(&path)
        };
        let conn = match immutable {
            Some(uri) => Connection::open_with_flags(uri, flags | OpenFlags::SQLITE_OPEN_URI)?,
            None => Connection::open_with_flags(&path, flags)?,
        };
        let found = Self::schema_version(&conn)?;
        if found != Self::SCHEMA_VERSION {
            return Err(Error::UnsupportedSchema {
                found,
                expected: Self::SCHEMA_VERSION,
            });
        }
        Self::check_history(&conn, &config)?;
        Ok(Self { conn, config, path })
    }

    /// Reject a log whose latest snapshot (or, without one, latest event) was
    /// reduced by another reducer version, a pruned log left with no anchor, and
    /// a log whose last recorded events are gone.
    fn check_history(conn: &Connection, config: &SoulConfig) -> Result<(), Error> {
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
        chain::head(conn)?.check()
    }

    /// The database file backing this log. The journal keeps `-wal` and
    /// `-shm` siblings next to it while the log is open.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Appends a new event to the soul, chained after the latest one.
    ///
    /// Returns [`Error::Truncated`] when events recorded at the end of the log
    /// are gone, since the new event would have nothing to chain from.
    pub fn append_event(&self, event: &Event) -> Result<SeqNo, Error> {
        let at_ms = i64::try_from(event.now().0)
            .map_err(|_| Error::Storage(rusqlite::Error::IntegralValueOutOfRange(0, i64::MAX)))?;
        // Canonical cues only: a non-finite measurement would serialize as `null` and
        // make the event unreadable (or read back differently) on replay.
        let payload = serde_json::to_string(&event.clone().canonical())?;
        let reducer_version = i64::from(self.config.reducer_version);
        let config_version = i64::from(self.config.config_version);
        let head = chain::head(&self.conn)?;
        head.check()?;
        let seq = head.seq.saturating_add(1);
        let checksum = chain::event_checksum(
            &head.checksum,
            seq,
            at_ms,
            reducer_version,
            config_version,
            payload.as_bytes(),
        );
        // One statement, so the event and its checksum commit together. Naming
        // the sequence number keeps AUTOINCREMENT's record of it up to date.
        self.conn.execute(
            "INSERT INTO events (seq, at_ms, payload_json, reducer_version, config_version, checksum) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                sql_seq(seq)?,
                at_ms,
                payload,
                reducer_version,
                config_version,
                checksum.as_slice()
            ],
        )?;
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

    /// The recorded status of `thought` and the small JSON result it was
    /// resolved with, or `None` when the thought was never recorded (a log
    /// written without a cortex, or a thought pruned by retention).
    pub fn thought_status(
        &self,
        thought: ThoughtId,
    ) -> Result<Option<(ActionStatus, Option<String>)>, Error> {
        let thought_id = i64::try_from(thought.0)
            .map_err(|_| Error::Storage(rusqlite::Error::IntegralValueOutOfRange(0, i64::MAX)))?;
        let mut stmt = self
            .conn
            .prepare("SELECT status, result_json FROM actions WHERE thought_id = ?1")?;
        let mut rows = stmt.query([thought_id])?;
        let Some(row) = rows.next()? else {
            return Ok(None);
        };
        let status: String = row.get(0)?;
        let result: Option<String> = row.get(1)?;
        // The schema's CHECK constraint admits only these three strings.
        let status = match status.as_str() {
            "done" => ActionStatus::Done,
            "failed" => ActionStatus::Failed,
            _ => ActionStatus::Pending,
        };
        Ok(Some((status, result)))
    }

    /// Read up to `limit` events after `cursor`, in sequence order, verifying
    /// each against the hash chain.
    ///
    /// The chain continues from the checksum held for the cursor's event (its
    /// own, as stored, or the retention anchor's), which a snapshot at the
    /// cursor must agree with. Returns [`Error::ReplayGap`] when the first event
    /// read is not the one immediately after the cursor, so a truncated log is
    /// never replayed silently, and [`Error::Corrupt`] naming the first event
    /// that fails its checksum, whose payload is never parsed.
    pub fn read_after(&self, cursor: SeqNo, limit: usize) -> Result<Vec<(SeqNo, Event)>, Error> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let cursor_i64 = sql_seq(cursor)?;
        let limit_i64 = i64::try_from(limit)
            .map_err(|_| Error::Storage(rusqlite::Error::IntegralValueOutOfRange(0, i64::MAX)))?;
        let mut prev = chain::link(&self.conn, cursor)?;
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {EVENT_COLUMNS} FROM events WHERE seq > ?1 ORDER BY seq ASC LIMIT ?2"
        ))?;
        let mut rows = stmt.query(rusqlite::params![cursor_i64, limit_i64])?;
        let mut out = Vec::with_capacity(limit.min(Self::REPLAY_PAGE));
        let mut expected = cursor.saturating_add(1);
        while let Some(row) = rows.next()? {
            let event = StoredEvent::read(row)?;
            if event.seq != expected {
                return Err(Error::ReplayGap {
                    expected,
                    found: event.seq,
                });
            }
            // Nothing vouches for the event before: the chain cannot reach this one.
            let link = prev.ok_or(Error::Corrupt {
                kind: "event",
                seq: event.seq,
            })?;
            prev = Some(event.verify(row, EVENT_CHECKSUM, &link)?);
            if event.reducer_version != i64::from(self.config.reducer_version) {
                return Err(Error::IncompatibleHistory {
                    kind: "event",
                    found: u32::try_from(event.reducer_version).unwrap_or(u32::MAX),
                    expected: self.config.reducer_version,
                });
            }
            out.push((event.seq, serde_json::from_slice(&event.payload)?));
            expected = event.seq.saturating_add(1);
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

            let n = self.prune(max_droppable_seq, limit)?;
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

    /// Drop up to `limit` of the oldest events, none after `through`, and move
    /// the chain anchor to the last one dropped, in one transaction.
    ///
    /// The dropped events are verified first: pruning never erases the evidence
    /// of damage, it reports it as [`Error::Corrupt`] and drops nothing.
    fn prune(&self, through: SeqNo, limit: usize) -> Result<usize, Error> {
        let tx = self.conn.unchecked_transaction()?;
        let mut stmt = tx.prepare(&format!(
            "SELECT {EVENT_COLUMNS} FROM events WHERE seq <= ?1 ORDER BY seq ASC LIMIT ?2"
        ))?;
        let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
        let through_i64 = i64::try_from(through).unwrap_or(i64::MAX);
        let mut rows = stmt.query(rusqlite::params![through_i64, limit_i64])?;
        let mut last: Option<(SeqNo, chain::Checksum)> = None;
        let mut count = 0;
        while let Some(row) = rows.next()? {
            let event = StoredEvent::read(row)?;
            let link = match last {
                Some((seq, checksum)) if event.seq == seq.saturating_add(1) => checksum,
                Some((seq, _)) => {
                    return Err(Error::ReplayGap {
                        expected: seq.saturating_add(1),
                        found: event.seq,
                    });
                }
                None => chain::link(&tx, event.seq.saturating_sub(1))?.ok_or(Error::Corrupt {
                    kind: "event",
                    seq: event.seq,
                })?,
            };
            last = Some((event.seq, event.verify(row, EVENT_CHECKSUM, &link)?));
            count += 1;
        }
        drop(rows);
        drop(stmt);
        let Some((seq, checksum)) = last else {
            return Ok(0);
        };
        tx.execute("DELETE FROM events WHERE seq <= ?1", [sql_seq(seq)?])?;
        chain::set_anchor(&tx, seq, &checksum)?;
        tx.commit()?;
        Ok(count)
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

/// An SQLite URI that opens `path` immutable: no locks, no `-wal` or `-shm`
/// files. `None` for a path that is not UTF-8, which then opens normally.
pub(super) fn immutable_uri(path: &Path) -> Option<String> {
    let text = path.to_str()?;
    // An absolute path gets an empty authority, so a leading `//` stays a path.
    let mut uri = String::from(if text.starts_with('/') {
        "file://"
    } else {
        "file:"
    });
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || b"-._~/".contains(&byte) {
            uri.push(char::from(byte));
        } else {
            uri.push('%');
            for nibble in [byte >> 4, byte & 0x0f] {
                let digit = char::from_digit(u32::from(nibble), 16).unwrap_or('0');
                uri.push(digit.to_ascii_uppercase());
            }
        }
    }
    uri.push_str("?immutable=1");
    Some(uri)
}
