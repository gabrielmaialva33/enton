//! Checksums: a SHA-256 hash chain over the event log and a checksum on every
//! snapshot, so damage and tampering are reported instead of replayed.
//!
//! Each event's checksum covers the checksum of the event before it, its own
//! sequence number, every stored column and its payload. Changing, removing,
//! renumbering or reordering an event therefore breaks the chain at the first
//! event affected, and every read that feeds replay verifies the events it
//! returns. The first event chains from [`GENESIS`]. Retention keeps an anchor,
//! the sequence number and checksum of the last event it pruned, so the earliest
//! kept event still verifies.
//!
//! A snapshot's checksum covers its sequence number, reducer version and blob,
//! and the checksum of the event it was taken at (its chain link). That binds the
//! snapshot to the history it summarizes: the event after it must continue the
//! chain from that link, and when the snapshot's own event was pruned without an
//! anchor the chain continues from the snapshot.
//!
//! The chain has no key. It detects media damage and any edit that does not
//! also recompute every later checksum; someone who can rewrite the whole file
//! can rewrite the chain too, which only a head checksum kept elsewhere reveals.
//! Deleting events from the end leaves a valid chain, so it is detected through
//! the last sequence number SQLite assigned (see [`head`]).
//!
//! Persona records and the links from thoughts to them sit beside the chain,
//! as snapshots do (see `persona.rs`). A persona record's checksum covers its
//! hash, length and origin, the sequence number of the event its first thought
//! was decided at and that event's checksum; a thought's link covers the
//! thought, its event and the checksum of the persona record it names. So an
//! altered, moved or swapped record or link is reported as damage.

use rusqlite::types::{Value, ValueRef};
use rusqlite::{Connection, OptionalExtension, Row};
use sha2::{Digest, Sha256};

use super::persona::PersonaDigest;
use super::{Error, SeqNo};

/// A SHA-256 digest.
pub(super) type Checksum = [u8; 32];

/// The checksum the first event chains from.
pub(super) const GENESIS: Checksum = [0; 32];

/// Domain tags keep a checksum of one kind of record from ever equalling one of
/// another kind.
const EVENT_DOMAIN: &[u8] = b"enton-soul-event-v1\0";
const SNAPSHOT_DOMAIN: &[u8] = b"enton-soul-snapshot-v1\0";
const PERSONA_DOMAIN: &[u8] = b"enton-soul-persona-v1\0";
const THOUGHT_DOMAIN: &[u8] = b"enton-soul-thought-v1\0";

/// The event columns [`StoredEvent::read`] expects, in order, then the checksum.
pub(super) const EVENT_COLUMNS: &str =
    "seq, at_ms, payload_json, reducer_version, config_version, checksum";
/// Index of the checksum in [`EVENT_COLUMNS`].
pub(super) const EVENT_CHECKSUM: usize = 5;
/// The snapshot columns [`StoredSnapshot::read`] expects, in order.
pub(super) const SNAPSHOT_COLUMNS: &str = "seq, reducer_version, blob, chain, checksum";

/// The checksum of an event: every field is fixed-width except the payload,
/// which comes last, so no two different events hash the same input.
pub(super) fn event_checksum(
    prev: &Checksum,
    seq: SeqNo,
    at_ms: i64,
    reducer_version: i64,
    config_version: i64,
    payload: &[u8],
) -> Checksum {
    let mut hasher = Sha256::new();
    hasher.update(EVENT_DOMAIN);
    hasher.update(prev);
    hasher.update(seq.to_be_bytes());
    hasher.update(at_ms.to_be_bytes());
    hasher.update(reducer_version.to_be_bytes());
    hasher.update(config_version.to_be_bytes());
    hasher.update(payload);
    hasher.finalize().into()
}

/// The checksum of a snapshot taken at `seq`, whose event holds `chain`.
pub(super) fn snapshot_checksum(
    seq: SeqNo,
    reducer_version: i64,
    chain: &Checksum,
    blob: &[u8],
) -> Checksum {
    let mut hasher = Sha256::new();
    hasher.update(SNAPSHOT_DOMAIN);
    hasher.update(seq.to_be_bytes());
    hasher.update(reducer_version.to_be_bytes());
    hasher.update(chain);
    hasher.update(blob);
    hasher.finalize().into()
}

/// The checksum of a persona record, first used by a thought decided at event
/// `first_seq`, whose checksum is `chain`. Every field is fixed-width.
pub(super) fn persona_checksum(
    first_seq: SeqNo,
    chain: &Checksum,
    persona: &PersonaDigest,
) -> Checksum {
    let mut hasher = Sha256::new();
    hasher.update(PERSONA_DOMAIN);
    hasher.update(first_seq.to_be_bytes());
    hasher.update(chain);
    hasher.update(persona.sha256);
    hasher.update(persona.bytes.to_be_bytes());
    hasher.update([persona.source.tag()]);
    hasher.finalize().into()
}

/// The checksum of the link from `thought`, decided at event `created_seq`, to
/// the persona record whose checksum is `persona`. Every field is fixed-width.
pub(super) fn thought_checksum(thought: u64, created_seq: SeqNo, persona: &Checksum) -> Checksum {
    let mut hasher = Sha256::new();
    hasher.update(THOUGHT_DOMAIN);
    hasher.update(thought.to_be_bytes());
    hasher.update(created_seq.to_be_bytes());
    hasher.update(persona);
    hasher.finalize().into()
}

/// An event row as stored, before its checksum is compared.
pub(super) struct StoredEvent {
    pub(super) seq: SeqNo,
    at_ms: i64,
    pub(super) payload: Vec<u8>,
    pub(super) reducer_version: i64,
    config_version: i64,
}

impl StoredEvent {
    /// Read the first five columns of [`EVENT_COLUMNS`]. A column holding the
    /// wrong type was damaged: it is reported as [`Error::Corrupt`].
    pub(super) fn read(row: &Row<'_>) -> Result<Self, Error> {
        let raw: i64 = row.get(0)?;
        // A negative sequence number is damage; zero then fails the gap check.
        let seq = u64::try_from(raw).unwrap_or(0);
        let integer = |index: usize| match row.get_ref(index)? {
            ValueRef::Integer(value) => Ok(value),
            _ => Err(Error::Corrupt { kind: "event", seq }),
        };
        let payload = match row.get_ref(2)? {
            ValueRef::Text(text) => text.to_vec(),
            _ => return Err(Error::Corrupt { kind: "event", seq }),
        };
        Ok(Self {
            seq,
            at_ms: integer(1)?,
            payload,
            reducer_version: integer(3)?,
            config_version: integer(4)?,
        })
    }

    /// This event's checksum when it follows an event holding `prev`.
    pub(super) fn checksum(&self, prev: &Checksum) -> Checksum {
        event_checksum(
            prev,
            self.seq,
            self.at_ms,
            self.reducer_version,
            self.config_version,
            &self.payload,
        )
    }

    /// Check the stored checksum at `index` of `row` against this event chained
    /// after `prev`, and return it for the next event.
    pub(super) fn verify(
        &self,
        row: &Row<'_>,
        index: usize,
        prev: &Checksum,
    ) -> Result<Checksum, Error> {
        let stored = checksum_in(row.get(index)?, "event", self.seq)?;
        if self.checksum(prev) == stored {
            Ok(stored)
        } else {
            Err(Error::Corrupt {
                kind: "event",
                seq: self.seq,
            })
        }
    }
}

/// A snapshot row whose checksum matched.
pub(super) struct StoredSnapshot {
    pub(super) seq: SeqNo,
    pub(super) blob: Vec<u8>,
    /// The checksum of the event the snapshot was taken at.
    pub(super) chain: Checksum,
}

impl StoredSnapshot {
    /// Read a row selected with [`SNAPSHOT_COLUMNS`] and verify its checksum.
    pub(super) fn read(row: &Row<'_>) -> Result<Self, Error> {
        let raw: i64 = row.get(0)?;
        let seq = u64::try_from(raw).unwrap_or(0);
        let corrupt = || Error::Corrupt {
            kind: "snapshot",
            seq,
        };
        let ValueRef::Integer(reducer_version) = row.get_ref(1)? else {
            return Err(corrupt());
        };
        let ValueRef::Blob(blob) = row.get_ref(2)? else {
            return Err(corrupt());
        };
        let blob = blob.to_vec();
        let chain = checksum_in(row.get(3)?, "snapshot", seq)?;
        let stored = checksum_in(row.get(4)?, "snapshot", seq)?;
        if snapshot_checksum(seq, reducer_version, &chain, &blob) != stored {
            return Err(corrupt());
        }
        Ok(Self { seq, blob, chain })
    }
}

/// A stored checksum: exactly 32 bytes in a blob, or the record is damaged.
pub(super) fn checksum_in(value: Value, kind: &'static str, seq: SeqNo) -> Result<Checksum, Error> {
    match value {
        Value::Blob(bytes) => {
            Checksum::try_from(bytes.as_slice()).map_err(|_| Error::Corrupt { kind, seq })
        }
        _ => Err(Error::Corrupt { kind, seq }),
    }
}

/// A sequence number as SQLite stores it.
pub(super) fn sql_seq(seq: SeqNo) -> Result<i64, Error> {
    i64::try_from(seq)
        .map_err(|_| Error::Storage(rusqlite::Error::IntegralValueOutOfRange(0, i64::MAX)))
}

/// The last event retention pruned: its sequence number and checksum.
fn anchor(conn: &Connection) -> Result<Option<(SeqNo, Checksum)>, Error> {
    let row: Option<(i64, Value)> = conn
        .query_row(
            "SELECT seq, checksum FROM chain_anchor WHERE id = 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    row.map(|(raw, value)| {
        let seq = u64::try_from(raw).unwrap_or(0);
        Ok((seq, checksum_in(value, "anchor", seq)?))
    })
    .transpose()
}

/// Move the anchor to the last pruned event.
pub(super) fn set_anchor(conn: &Connection, seq: SeqNo, checksum: &Checksum) -> Result<(), Error> {
    conn.execute(
        "INSERT OR REPLACE INTO chain_anchor (id, seq, checksum) VALUES (1, ?1, ?2)",
        rusqlite::params![sql_seq(seq)?, checksum.as_slice()],
    )?;
    Ok(())
}

/// The checksum the log holds for event `seq`: the event's own, the anchor's
/// when retention pruned it, and [`GENESIS`] for sequence number zero. The
/// event's own checksum is returned as stored, not verified.
pub(super) fn held(conn: &Connection, seq: SeqNo) -> Result<Option<Checksum>, Error> {
    if seq == 0 {
        return Ok(Some(GENESIS));
    }
    let stored: Option<Value> = conn
        .query_row(
            "SELECT checksum FROM events WHERE seq = ?1",
            [sql_seq(seq)?],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(value) = stored {
        return checksum_in(value, "event", seq).map(Some);
    }
    Ok(anchor(conn)?.and_then(|(at, checksum)| (at == seq).then_some(checksum)))
}

/// The snapshot taken at `seq`, verified, if there is one.
pub(super) fn snapshot_at(conn: &Connection, seq: SeqNo) -> Result<Option<StoredSnapshot>, Error> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {SNAPSHOT_COLUMNS} FROM snapshots WHERE seq = ?1"
    ))?;
    let mut rows = stmt.query([sql_seq(seq)?])?;
    rows.next()?.map(StoredSnapshot::read).transpose()
}

/// The checksum event `seq + 1` chains from, or `None` when nothing the log
/// keeps vouches for event `seq`.
///
/// A snapshot at `seq` must agree with the checksum held for its event, which
/// is then reported as [`Error::Corrupt`]; when that event was pruned without
/// an anchor, the chain continues from the snapshot.
pub(super) fn link(conn: &Connection, seq: SeqNo) -> Result<Option<Checksum>, Error> {
    let held = held(conn, seq)?;
    let snapshot = snapshot_at(conn, seq)?.map(|snapshot| snapshot.chain);
    match (held, snapshot) {
        (Some(held), Some(chain)) if held != chain => Err(Error::Corrupt { kind: "event", seq }),
        (Some(held), _) => Ok(Some(held)),
        (None, chain) => Ok(chain),
    }
}

/// Whether event `seq` holds the checksum its place in the chain gives it:
/// `false` only when the event is stored, the log vouches for the one before,
/// and the event fails against it.
pub(super) fn event_verifies(conn: &Connection, seq: SeqNo) -> Result<bool, Error> {
    let Some(prev) = link(conn, seq.saturating_sub(1))? else {
        return Ok(true);
    };
    let mut stmt = conn.prepare(&format!(
        "SELECT {EVENT_COLUMNS} FROM events WHERE seq = ?1"
    ))?;
    let mut rows = stmt.query([sql_seq(seq)?])?;
    let Some(row) = rows.next()? else {
        return Ok(true);
    };
    let event = StoredEvent::read(row)?;
    Ok(event.verify(row, EVENT_CHECKSUM, &prev).is_ok())
}

/// Where the log ends.
pub(super) struct Head {
    /// The last sequence number SQLite assigned, zero before the first event.
    pub(super) recorded: SeqNo,
    /// The latest event the log keeps (or, with every event pruned, the anchor).
    pub(super) seq: SeqNo,
    /// The checksum held for it.
    pub(super) checksum: Checksum,
}

impl Head {
    /// Fails with [`Error::Truncated`] when events the log recorded are gone
    /// from its end, which leaves the chain itself intact.
    pub(super) fn check(&self) -> Result<(), Error> {
        if self.recorded > self.seq {
            return Err(Error::Truncated {
                recorded: self.recorded,
                found: self.seq,
            });
        }
        Ok(())
    }
}

/// Where the log ends: the last sequence number assigned and the latest event kept.
pub(super) fn head(conn: &Connection) -> Result<Head, Error> {
    let recorded: i64 = conn.query_row(
        "SELECT IFNULL(MAX(seq), 0) FROM sqlite_sequence WHERE name = 'events'",
        [],
        |row| row.get(0),
    )?;
    let recorded = u64::try_from(recorded).unwrap_or(0);
    let latest: Option<(i64, Value)> = conn
        .query_row(
            "SELECT seq, checksum FROM events ORDER BY seq DESC LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let (seq, checksum) = match latest {
        Some((raw, value)) => {
            let seq = u64::try_from(raw).unwrap_or(0);
            (seq, checksum_in(value, "event", seq)?)
        }
        None => anchor(conn)?.unwrap_or((0, GENESIS)),
    };
    Ok(Head {
        recorded,
        seq,
        checksum,
    })
}

/// Compute the checksums of a log written before schema 4, inside its
/// migration transaction.
///
/// Its rows are trusted as they stand at this moment: whatever they hold is
/// what the chain vouches for from now on. A row whose columns hold the wrong
/// type fails with [`Error::Corrupt`], since it cannot be trusted as a record.
/// When older code pruned the log, the checksum of the last pruned event is
/// unknowable, so the anchor before the earliest kept event (or, with none
/// kept, at the last sequence number assigned) holds [`GENESIS`]; so does the
/// chain link of a snapshot whose event was pruned.
pub(super) fn seal(conn: &Connection) -> Result<(), Error> {
    let first: Option<i64> = conn.query_row("SELECT MIN(seq) FROM events", [], |row| row.get(0))?;
    let before_first = match first {
        Some(first) => u64::try_from(first).unwrap_or(0).saturating_sub(1),
        None => head(conn)?.recorded,
    };
    if before_first > 0 {
        set_anchor(conn, before_first, &GENESIS)?;
    }

    let mut prev = GENESIS;
    let mut cursor = i64::MIN;
    loop {
        let page = {
            let mut stmt = conn.prepare(
                "SELECT seq, at_ms, payload_json, reducer_version, config_version \
                 FROM events WHERE seq > ?1 ORDER BY seq ASC LIMIT 1024",
            )?;
            let mut rows = stmt.query([cursor])?;
            let mut page = Vec::new();
            while let Some(row) = rows.next()? {
                page.push((row.get::<_, i64>(0)?, StoredEvent::read(row)?));
            }
            page
        };
        if page.is_empty() {
            break;
        }
        for (raw, event) in &page {
            let checksum = event.checksum(&prev);
            conn.execute(
                "UPDATE events SET checksum = ?1 WHERE seq = ?2",
                rusqlite::params![checksum.as_slice(), raw],
            )?;
            prev = checksum;
            cursor = *raw;
        }
    }

    let snapshots: Vec<(i64, i64, Vec<u8>)> = {
        let mut stmt = conn.prepare("SELECT seq, reducer_version, blob FROM snapshots")?;
        let mut rows = stmt.query([])?;
        let mut snapshots = Vec::new();
        while let Some(row) = rows.next()? {
            let raw: i64 = row.get(0)?;
            let seq = u64::try_from(raw).unwrap_or(0);
            let (ValueRef::Integer(reducer_version), ValueRef::Blob(blob)) =
                (row.get_ref(1)?, row.get_ref(2)?)
            else {
                return Err(Error::Corrupt {
                    kind: "snapshot",
                    seq,
                });
            };
            snapshots.push((raw, reducer_version, blob.to_vec()));
        }
        snapshots
    };
    for (raw, reducer_version, blob) in snapshots {
        let seq = u64::try_from(raw).unwrap_or(0);
        let chain = held(conn, seq)?.unwrap_or(GENESIS);
        let checksum = snapshot_checksum(seq, reducer_version, &chain, &blob);
        conn.execute(
            "UPDATE snapshots SET chain = ?1, checksum = ?2 WHERE seq = ?3",
            rusqlite::params![chain.as_slice(), checksum.as_slice(), raw],
        )?;
    }
    Ok(())
}
