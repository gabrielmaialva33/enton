//! Typed organism snapshots on top of the soul's opaque blob API.

use enton_core::{Action, Organism, Profile};
use serde::{Deserialize, Serialize, de::Error as _};

use crate::soul::{Error, SeqNo, Soul};

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
        self.replay(
            || Organism::new(profile.clone()),
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
