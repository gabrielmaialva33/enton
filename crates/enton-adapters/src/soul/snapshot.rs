//! Snapshots: opaque blobs at a sequence number, and the typed organism
//! snapshot built on them so replay only has to reduce the tail.

use super::chain::{self, SNAPSHOT_COLUMNS, StoredSnapshot, sql_seq};
use super::{Error, SeqNo, Soul};
use enton_core::{Action, Organism, Profile};
use serde::{Deserialize, Serialize, de::Error as _};

// This tag versions the blob layout independently of SoulConfig's reducer
// version. Unknown formats must fail restoration, never start a fresh organism.
#[derive(Serialize, Deserialize)]
#[serde(tag = "format", deny_unknown_fields)]
enum Snapshot<T> {
    #[serde(rename = "enton-organism-v1")]
    V1 { state: T },
}

impl Soul {
    /// Persist the complete organism state after reducing event `at_seq`.
    ///
    /// The caller must pass the state at exactly that log sequence and must not
    /// append concurrently. The blob includes the full profile, both budgets,
    /// attention, habituation, ignition, drives, time and the next thought ID.
    /// Returns storage or serialization errors; zero/future sequences are rejected
    /// by the underlying snapshot API. No actuators are invoked.
    pub fn save_organism_snapshot(&self, at_seq: SeqNo, organism: &Organism) -> Result<(), Error> {
        let blob = serde_json::to_vec(&Snapshot::V1 { state: organism })?;
        self.save_snapshot(at_seq, &blob)
    }

    /// Restore a compatible snapshot and reduce only its event tail.
    ///
    /// Without a snapshot, starts at sequence zero with `profile`. A snapshot
    /// must contain exactly this profile; changing policy requires an explicit
    /// migration. Use the original reducer/config versions in [`crate::SoulConfig`],
    /// and keep at least one snapshot when pruning (`max_snapshots` must not be zero).
    /// Malformed/unknown blob formats and profile mismatches return a JSON error;
    /// missing events return a replay gap; a snapshot or tail event that fails
    /// its checksum returns [`Error::Corrupt`] with its sequence number. This
    /// never falls back after a failed restore and never executes returned
    /// actions (including `Speak`).
    ///
    /// Returned actions cover only the replayed tail, not the snapshotted prefix.
    /// As with [`Soul::replay`], callers must bound the log with retention to bound
    /// the returned action vector. Run this blocking API outside async executors.
    ///
    /// Drive the restored organism with
    /// [`MonotonicClock::resuming_at`](crate::MonotonicClock::resuming_at) at
    /// [`Organism::last_seen`]: a fresh clock starts at zero, and the reducer
    /// ignores time that runs backward.
    pub fn replay_organism(&self, profile: &Profile) -> Result<(Organism, Vec<Action>), Error> {
        let fresh = Organism::new(profile.clone())?;
        self.replay(
            || fresh,
            |blob| decode_organism(blob, profile),
            Organism::step,
        )
    }

    /// The organism at the earliest point from which the stored log replays
    /// without a gap, and the sequence number it stands at.
    ///
    /// [`Soul::replay_organism`] starts from the latest snapshot, the fastest way
    /// to resume; after a clean shutdown its tail is empty. An audit wants the
    /// opposite: every stored event. This starts from a fresh organism while the
    /// log still holds its first event, and otherwise from the earliest
    /// compatible snapshot that the stored events continue. Reduce the rest by
    /// paging [`Soul::read_after`] from the returned sequence number through
    /// [`Organism::step`]: the reducer is deterministic, so this recomputes every
    /// stored decision exactly.
    ///
    /// Snapshots are checked against `profile` even when replay starts fresh, so
    /// a wrong profile fails here instead of replaying different decisions.
    /// Returns [`Error::ReplayGap`] when the log was pruned and no snapshot
    /// anchors what is left, and [`Error::Corrupt`] when the snapshot fails its
    /// checksum. Read-only; no actuators are invoked.
    pub fn earliest_organism(&self, profile: &Profile) -> Result<(Organism, SeqNo), Error> {
        let first: Option<i64> = self
            .conn
            .query_row("SELECT MIN(seq) FROM events", [], |row| row.get(0))?;
        let first = first.map(|seq| u64::try_from(seq).unwrap_or(0));
        // A snapshot anchors the log when the event right after it is stored.
        let floor = first.map_or(0, |seq| seq.saturating_sub(1));
        let floor = i64::try_from(floor)
            .map_err(|_| Error::Storage(rusqlite::Error::IntegralValueOutOfRange(0, i64::MAX)))?;
        let anchor = {
            let mut stmt = self.conn.prepare(&format!(
                "SELECT {SNAPSHOT_COLUMNS} FROM snapshots WHERE reducer_version = ?1 AND seq >= ?2 \
                 ORDER BY seq ASC LIMIT 1"
            ))?;
            let mut rows = stmt.query(rusqlite::params![self.config.reducer_version, floor])?;
            rows.next()?.map(StoredSnapshot::read).transpose()?
        };
        match (first, anchor) {
            (Some(1), anchor) => {
                if let Some(snapshot) = anchor {
                    decode_organism(&snapshot.blob, profile)?;
                }
                Ok((Organism::new(profile.clone())?, 0))
            }
            (_, Some(snapshot)) => Ok((decode_organism(&snapshot.blob, profile)?, snapshot.seq)),
            (None, None) => Ok((Organism::new(profile.clone())?, 0)),
            (Some(found), None) => Err(Error::ReplayGap { expected: 1, found }),
        }
    }
}

/// Decode an organism snapshot blob, which must hold exactly `profile`.
fn decode_organism(blob: &[u8], profile: &Profile) -> Result<Organism, Error> {
    let Snapshot::V1 { state } = serde_json::from_slice::<Snapshot<Organism>>(blob)?;
    if state.profile() != profile {
        return Err(serde_json::Error::custom(
            "organism snapshot profile differs from the requested replay profile",
        )
        .into());
    }
    Ok(state)
}

impl Soul {
    /// Save an opaque state snapshot at a given sequence number.
    ///
    /// The snapshot's checksum binds it to the checksum held for event
    /// `at_seq`, so that event must still be stored (or be the one retention
    /// pruned last).
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

        let chain = chain::held(&self.conn, at_seq)?.ok_or(Error::InvalidSnapshotSequence(
            "the event at the snapshot seq was pruned",
        ))?;
        let reducer_version = i64::from(self.config.reducer_version);
        let checksum = chain::snapshot_checksum(at_seq, reducer_version, &chain, blob);
        self.conn.execute(
            "INSERT OR REPLACE INTO snapshots (seq, reducer_version, blob, chain, checksum) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                sql_seq(at_seq)?,
                reducer_version,
                blob,
                chain.as_slice(),
                checksum.as_slice()
            ],
        )?;
        Ok(())
    }

    /// Return the latest snapshot, if any, after verifying its checksum
    /// ([`Error::Corrupt`] when it fails).
    pub fn latest_snapshot(&self) -> Result<Option<(SeqNo, Vec<u8>)>, Error> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {SNAPSHOT_COLUMNS} FROM snapshots ORDER BY seq DESC LIMIT 1"
        ))?;
        let mut rows = stmt.query([])?;
        let snapshot = rows.next()?.map(StoredSnapshot::read).transpose()?;
        Ok(snapshot.map(|snapshot| (snapshot.seq, snapshot.blob)))
    }
}
