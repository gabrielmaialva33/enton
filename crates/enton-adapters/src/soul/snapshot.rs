//! Snapshots: opaque blobs at a sequence number, and the typed organism
//! snapshot built on them so replay only has to reduce the tail.

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
    /// missing events return a replay gap. This never falls back after a failed
    /// restore and never executes returned actions (including `Speak`).
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
            |blob| {
                let Snapshot::V1 { state } = serde_json::from_slice::<Snapshot<Organism>>(blob)?;
                if state.profile() != profile {
                    return Err(serde_json::Error::custom(
                        "organism snapshot profile differs from the requested replay profile",
                    )
                    .into());
                }
                Ok(state)
            },
            Organism::step,
        )
    }
}

impl Soul {
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
}
