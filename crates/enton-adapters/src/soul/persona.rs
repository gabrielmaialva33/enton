//! Personas: which system prompt each thought was asked with, kept as a hash.
//!
//! The persona is the text that gives the cortex Enton's voice: the built-in
//! default or the owner's `PERSONA.md`. The soul never keeps that text. Like a
//! transcript, it is not needed to replay a decision (the reducer never sees
//! it), and the owner keeps the file, in git if they want its history. The soul
//! keeps what proves which version spoke: the SHA-256 of the persona's bytes
//! (for a file, what `sha256sum` prints), their length, and whether it was the
//! built-in default or a file.
//!
//! A persona is recorded once, in the `personas` table, together with the first
//! thought asked with it, and every thought's action row links to the persona
//! record it was asked with. Neither is an event, so neither is in the hash
//! chain; they sit beside it as snapshots do. A record's checksum covers the
//! checksum of the event its first thought was decided at, and a link's covers
//! the checksum of the record it names (see `chain.rs`), so an altered, moved or
//! swapped record or link fails its checksum and is reported as damage. Like
//! the chain's, those checksums have no key: they catch damaged media and hand
//! edits, not someone who recomputes them.

use enton_core::ThoughtId;
use rusqlite::types::ValueRef;
use rusqlite::{Connection, Row};

use super::chain::{self, Checksum, checksum_in, sql_seq};
use super::{Error, SeqNo, Soul};

/// The persona columns [`StoredPersona::read`] expects, in order.
const PERSONA_COLUMNS: &str = "id, sha256, bytes, source, first_seq, chain, checksum";
/// The action columns [`linked`] expects, in order.
const LINK_COLUMNS: &str = "thought_id, created_seq, persona, persona_checksum";
/// Hex digits in [`PersonaDigest::short_hex`].
const SHORT_HEX: usize = 12;

/// Where the text of a persona came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PersonaSource {
    /// The default compiled into the binary.
    BuiltIn,
    /// A file the owner wrote.
    File,
}

impl PersonaSource {
    /// The database string representation for this source.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BuiltIn => "built-in",
            Self::File => "file",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "built-in" => Some(Self::BuiltIn),
            "file" => Some(Self::File),
            _ => None,
        }
    }

    /// The byte a checksum covers for this source.
    pub(super) const fn tag(self) -> u8 {
        match self {
            Self::BuiltIn => 0,
            Self::File => 1,
        }
    }
}

/// What the soul keeps of a persona: never its text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PersonaDigest {
    /// SHA-256 of the persona's bytes; for a file, what `sha256sum` prints.
    pub sha256: [u8; 32],
    /// The length of those bytes.
    pub bytes: u64,
    /// Whether it was the built-in default or a file.
    pub source: PersonaSource,
}

impl PersonaDigest {
    /// The hash in lowercase hex, as `sha256sum` prints it.
    #[must_use]
    pub fn hex(&self) -> String {
        self.sha256
            .iter()
            .flat_map(|byte| [byte >> 4, byte & 0x0f])
            .map(|nibble| char::from_digit(u32::from(nibble), 16).unwrap_or('0'))
            .collect()
    }

    /// The first 12 hex digits of the hash, enough to tell personas apart.
    #[must_use]
    pub fn short_hex(&self) -> String {
        self.hex().chars().take(SHORT_HEX).collect()
    }
}

/// A persona the soul recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersonaRecord {
    /// The persona.
    pub digest: PersonaDigest,
    /// The event the first thought asked with it was decided at.
    pub first_seq: SeqNo,
}

/// A persona row whose checksum matched.
struct StoredPersona {
    id: i64,
    record: PersonaRecord,
    /// The checksum of the event at `first_seq` when the record was written.
    chain: Checksum,
    checksum: Checksum,
}

impl StoredPersona {
    /// Read a row selected with [`PERSONA_COLUMNS`] and verify its checksum. A
    /// column holding the wrong type or value was damaged.
    fn read(row: &Row<'_>) -> Result<Self, Error> {
        let id: i64 = row.get(0)?;
        let ValueRef::Integer(first_seq) = row.get_ref(4)? else {
            return Err(Error::Corrupt {
                kind: "persona",
                seq: 0,
            });
        };
        let first_seq = u64::try_from(first_seq).unwrap_or(0);
        let corrupt = || Error::Corrupt {
            kind: "persona",
            seq: first_seq,
        };
        let sha256 = match row.get_ref(1)? {
            ValueRef::Blob(bytes) => <[u8; 32]>::try_from(bytes).map_err(|_| corrupt())?,
            _ => return Err(corrupt()),
        };
        let bytes = match row.get_ref(2)? {
            ValueRef::Integer(bytes) => u64::try_from(bytes).map_err(|_| corrupt())?,
            _ => return Err(corrupt()),
        };
        let source = match row.get_ref(3)? {
            ValueRef::Text(text) => std::str::from_utf8(text)
                .ok()
                .and_then(PersonaSource::parse)
                .ok_or_else(corrupt)?,
            _ => return Err(corrupt()),
        };
        let chain = checksum_in(row.get(5)?, "persona", first_seq)?;
        let checksum = checksum_in(row.get(6)?, "persona", first_seq)?;
        let record = PersonaRecord {
            digest: PersonaDigest {
                sha256,
                bytes,
                source,
            },
            first_seq,
        };
        if chain::persona_checksum(first_seq, &chain, &record.digest) != checksum {
            return Err(corrupt());
        }
        Ok(Self {
            id,
            record,
            chain,
            checksum,
        })
    }

    /// Read and verify a row, then check it against the log: while the event
    /// it was bound to is kept, that event must still hold the same checksum.
    /// When it does not, either the event's checksum was damaged or the record
    /// belongs to another history (it was moved or copied in); the error names
    /// the event when the event fails its own place in the chain, and the
    /// record otherwise.
    fn verified(conn: &Connection, row: &Row<'_>) -> Result<Self, Error> {
        let stored = Self::read(row)?;
        let seq = stored.record.first_seq;
        match chain::held(conn, seq)? {
            Some(held) if held != stored.chain => {
                let kind = if chain::event_verifies(conn, seq)? {
                    "persona"
                } else {
                    "event"
                };
                Err(Error::Corrupt { kind, seq })
            }
            _ => Ok(stored),
        }
    }
}

/// Every persona record, verified, oldest first.
fn all(conn: &Connection) -> Result<Vec<StoredPersona>, Error> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {PERSONA_COLUMNS} FROM personas ORDER BY first_seq ASC, id ASC"
    ))?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(StoredPersona::verified(conn, row)?);
    }
    Ok(out)
}

/// Verify every persona record; the soul checks them when it opens.
pub(super) fn verify(conn: &Connection) -> Result<(), Error> {
    all(conn).map(drop)
}

/// The record of `persona`, written when no thought was asked with it before,
/// bound to event `created_seq`. Returns its row id and checksum.
pub(super) fn adopt(
    conn: &Connection,
    persona: &PersonaDigest,
    created_seq: SeqNo,
) -> Result<(i64, Checksum), Error> {
    let found = {
        let mut stmt = conn.prepare(&format!(
            "SELECT {PERSONA_COLUMNS} FROM personas WHERE sha256 = ?1 AND source = ?2"
        ))?;
        let mut rows = stmt.query(rusqlite::params![
            persona.sha256.as_slice(),
            persona.source.as_str()
        ])?;
        rows.next()?
            .map(|row| StoredPersona::verified(conn, row))
            .transpose()?
    };
    if let Some(stored) = found {
        // The same hash is the same text, so the same length.
        if stored.record.digest != *persona {
            return Err(Error::Corrupt {
                kind: "persona",
                seq: stored.record.first_seq,
            });
        }
        return Ok((stored.id, stored.checksum));
    }
    let chain = chain::held(conn, created_seq)?.ok_or(Error::UnknownEvent(created_seq))?;
    let checksum = chain::persona_checksum(created_seq, &chain, persona);
    let bytes = i64::try_from(persona.bytes)
        .map_err(|_| Error::Storage(rusqlite::Error::IntegralValueOutOfRange(2, i64::MAX)))?;
    conn.execute(
        "INSERT INTO personas (sha256, bytes, source, first_seq, chain, checksum) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![
            persona.sha256.as_slice(),
            bytes,
            persona.source.as_str(),
            sql_seq(created_seq)?,
            chain.as_slice(),
            checksum.as_slice()
        ],
    )?;
    Ok((conn.last_insert_rowid(), checksum))
}

/// The thought in an action row selected with [`LINK_COLUMNS`], and the persona
/// it was asked with, both verified. A thought recorded before the soul kept
/// personas links to none.
fn linked(conn: &Connection, row: &Row<'_>) -> Result<(ThoughtId, Option<PersonaRecord>), Error> {
    let thought = u64::try_from(row.get::<_, i64>(0)?).unwrap_or(0);
    let seq = match row.get_ref(1)? {
        ValueRef::Integer(seq) => u64::try_from(seq).unwrap_or(0),
        _ => 0,
    };
    let corrupt = || Error::Corrupt {
        kind: "thought",
        seq,
    };
    let (id, link) = match (row.get_ref(2)?, row.get_ref(3)?) {
        (ValueRef::Null, ValueRef::Null) => return Ok((ThoughtId(thought), None)),
        (ValueRef::Integer(id), ValueRef::Blob(link)) => {
            (id, Checksum::try_from(link).map_err(|_| corrupt())?)
        }
        _ => return Err(corrupt()),
    };
    let stored = {
        let mut stmt = conn.prepare(&format!(
            "SELECT {PERSONA_COLUMNS} FROM personas WHERE id = ?1"
        ))?;
        let mut rows = stmt.query([id])?;
        rows.next()?
            .map(|row| StoredPersona::verified(conn, row))
            .transpose()?
            .ok_or_else(corrupt)?
    };
    if chain::thought_checksum(thought, seq, &stored.checksum) != link {
        return Err(corrupt());
    }
    Ok((ThoughtId(thought), Some(stored.record)))
}

impl Soul {
    /// Every persona the soul recorded, in the order they were first used, each
    /// verified ([`Error::Corrupt`] names the first damaged record).
    pub fn personas(&self) -> Result<Vec<PersonaRecord>, Error> {
        Ok(all(&self.conn)?
            .into_iter()
            .map(|stored| stored.record)
            .collect())
    }

    /// The persona `thought` was asked with, verified together with the
    /// thought's link to it. `None` when the thought was never recorded, or was
    /// recorded before the soul kept personas.
    pub fn thought_persona(&self, thought: ThoughtId) -> Result<Option<PersonaRecord>, Error> {
        let thought_id = i64::try_from(thought.0)
            .map_err(|_| Error::Storage(rusqlite::Error::IntegralValueOutOfRange(0, i64::MAX)))?;
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {LINK_COLUMNS} FROM actions WHERE thought_id = ?1"
        ))?;
        let mut rows = stmt.query([thought_id])?;
        let linked = rows
            .next()?
            .map(|row| linked(&self.conn, row))
            .transpose()?;
        Ok(linked.and_then(|(_, persona)| persona))
    }

    /// The latest thought recorded before `thought`, with the persona it was
    /// asked with (verified, `None` when it predates persona records).
    pub fn thought_before(
        &self,
        thought: ThoughtId,
    ) -> Result<Option<(ThoughtId, Option<PersonaRecord>)>, Error> {
        let bound = i64::try_from(thought.0).unwrap_or(i64::MAX);
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {LINK_COLUMNS} FROM actions WHERE thought_id < ?1 \
             ORDER BY thought_id DESC LIMIT 1"
        ))?;
        let mut rows = stmt.query([bound])?;
        rows.next()?.map(|row| linked(&self.conn, row)).transpose()
    }
}
