//! The soul's checksums, tested by damaging its database behind its back the
//! way a bad disk or a hand edit would: a flipped byte in an event, its checksum
//! or a snapshot, a deleted, renumbered or swapped event and a lost tail are each
//! reported with the record they hit, never replayed. Logs written before the
//! checksums migrate and verify, and pruned logs keep verifying from their anchor.
//! Persona records and the links from thoughts to them are checked the same way.
#![cfg(feature = "soul")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use enton_adapters::soul::{Error, PersonaDigest, PersonaSource};
use enton_adapters::{SeqNo, Soul, SoulConfig};
use enton_core::{Event, Millis, Organism, Profile, SpeechCue, ThoughtId};
use rusqlite::Connection;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

/// A scratch directory for one database, removed on drop.
struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> std::io::Result<Self> {
        let id = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("enton-soul-integrity-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&path)?;
        Ok(Self(path))
    }

    fn database(&self) -> PathBuf {
        self.0.join("soul.sqlite")
    }

    /// A second connection to the database, as a bad disk or an editor would use.
    fn tamper(&self) -> rusqlite::Result<Connection> {
        let conn = Connection::open(self.database())?;
        conn.pragma_update(None, "synchronous", "OFF")?;
        Ok(conn)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        // Best effort: a leftover temporary directory must not fail a test.
        drop(std::fs::remove_dir_all(&self.0));
    }
}

/// The `index`-th event of a short life: ticks and speech cues.
fn event(index: u32) -> Event {
    let now = Millis(1_000 * u64::from(index));
    if index.is_multiple_of(3) {
        Event::Tick { now }
    } else {
        Event::Speech {
            now,
            cue: SpeechCue {
                energy: 0.8,
                duration_ms: 700 + 10 * index,
                vad_confidence: 0.9,
                keyword: index % 4 == 1,
                ..SpeechCue::default()
            },
        }
    }
}

/// Append events `from..=to` while `organism` reduces them; with `snapshot_at`,
/// snapshot the organism after that event.
fn live(
    soul: &Soul,
    organism: &mut Organism,
    from: u32,
    to: u32,
    snapshot_at: Option<SeqNo>,
) -> TestResult {
    for index in from..=to {
        let seq = soul.append_event(&event(index))?;
        organism.step(&event(index));
        if Some(seq) == snapshot_at {
            soul.save_organism_snapshot(seq, organism)?;
        }
    }
    Ok(())
}

/// A soul holding events `1..=count` and the organism that lived them.
fn soul_with(
    directory: &TestDirectory,
    count: u32,
    config: SoulConfig,
    snapshot_at: Option<SeqNo>,
) -> TestResult<(Soul, Organism)> {
    let soul = Soul::open(directory.database(), config)?;
    let mut organism = Organism::new(Profile::t1_ref())?;
    live(&soul, &mut organism, 1, count, snapshot_at)?;
    Ok((soul, organism))
}

/// The bytes of a column of event `seq`, as stored.
fn event_bytes(conn: &Connection, column: &str, seq: SeqNo) -> TestResult<Vec<u8>> {
    let seq = i64::try_from(seq)?;
    Ok(conn.query_row(
        &format!("SELECT CAST({column} AS BLOB) FROM events WHERE seq = ?1"),
        [seq],
        |row| row.get(0),
    )?)
}

/// Overwrite the payload of event `seq` with raw bytes, kept as text.
fn set_payload(conn: &Connection, seq: SeqNo, bytes: &[u8]) -> TestResult {
    let seq = i64::try_from(seq)?;
    conn.execute(
        "UPDATE events SET payload_json = CAST(?1 AS TEXT) WHERE seq = ?2",
        rusqlite::params![bytes, seq],
    )?;
    Ok(())
}

/// Overwrite a blob column of the row of `table` at `seq`.
fn set_blob(conn: &Connection, table: &str, column: &str, seq: SeqNo, bytes: &[u8]) -> TestResult {
    let seq = i64::try_from(seq)?;
    conn.execute(
        &format!("UPDATE {table} SET {column} = ?1 WHERE seq = ?2"),
        rusqlite::params![bytes, seq],
    )?;
    Ok(())
}

/// `bytes` with the byte at `at` flipped by the bits of `mask`.
fn flipped(bytes: &[u8], at: usize, mask: u8) -> Vec<u8> {
    let mut out = bytes.to_vec();
    if let Some(byte) = out.get_mut(at) {
        *byte ^= mask;
    }
    out
}

fn is_corrupt<T>(result: &Result<T, Error>, kind: &str, at: SeqNo) -> bool {
    matches!(result, Err(Error::Corrupt { kind: found, seq }) if *found == kind && *seq == at)
}

// ---------------------------------------------------------------------------
// Flipped bytes

#[test]
fn a_flipped_payload_byte_is_reported_with_its_sequence_number() {
    let directory = TestDirectory::new().unwrap();
    let (soul, _) = soul_with(&directory, 10, SoulConfig::default(), None).unwrap();
    let tamper = directory.tamper().unwrap();
    let original = event_bytes(&tamper, "payload_json", 5).unwrap();

    // The dangerous case: one bit turns a digit into another digit, and the
    // payload still parses as a valid event that was never recorded.
    let digit = original.iter().position(u8::is_ascii_digit).unwrap();
    let forged = flipped(&original, digit, 0x01);
    let forged_event: Event = serde_json::from_slice(&forged).unwrap();
    assert_ne!(forged_event, event(5).canonical());
    set_payload(&tamper, 5, &forged).unwrap();
    assert!(is_corrupt(&soul.read_after(0, 64), "event", 5));
    assert!(is_corrupt(
        &soul.replay_organism(&Profile::t1_ref()),
        "event",
        5
    ));
    let read_only = Soul::open_read_only(directory.database(), SoulConfig::default()).unwrap();
    assert!(is_corrupt(&read_only.read_after(0, 64), "event", 5));
    // The events before the damage still read.
    assert_eq!(soul.read_after(0, 4).unwrap().len(), 4);

    // Every byte, flipped in the low bit, a letter-case bit and the high bit
    // (which leaves invalid UTF-8 in the text).
    for at in 0..original.len() {
        for mask in [0x01, 0x20, 0x80] {
            set_payload(&tamper, 5, &flipped(&original, at, mask)).unwrap();
            let read = soul.read_after(0, 64);
            assert!(
                is_corrupt(&read, "event", 5),
                "byte {at} ^ {mask:#04x}: {read:?}"
            );
        }
    }
    set_payload(&tamper, 5, &original).unwrap();
    assert_eq!(soul.read_after(0, 64).unwrap().len(), 10);
}

#[test]
fn a_flipped_checksum_or_column_is_reported_with_its_sequence_number() {
    let directory = TestDirectory::new().unwrap();
    let (soul, organism) = soul_with(&directory, 10, SoulConfig::default(), Some(5)).unwrap();
    let tamper = directory.tamper().unwrap();

    for seq in [1, 5, 10] {
        let original = event_bytes(&tamper, "checksum", seq).unwrap();
        assert_eq!(original.len(), 32);
        for at in 0..original.len() {
            set_blob(
                &tamper,
                "events",
                "checksum",
                seq,
                &flipped(&original, at, 0x01),
            )
            .unwrap();
            assert!(
                is_corrupt(&soul.read_after(0, 64), "event", seq),
                "{seq}/{at}"
            );
        }
        let longer = [original.as_slice(), &[0_u8]].concat();
        for wrong in [&[][..], &original[..31], longer.as_slice()] {
            set_blob(&tamper, "events", "checksum", seq, wrong).unwrap();
            assert!(is_corrupt(&soul.read_after(0, 64), "event", seq));
        }
        set_blob(&tamper, "events", "checksum", seq, &original).unwrap();
    }

    // The snapshot at 5 vouches for event 5's checksum: replay from it notices
    // a damaged one even though event 5 itself is not replayed.
    let original = event_bytes(&tamper, "checksum", 5).unwrap();
    set_blob(
        &tamper,
        "events",
        "checksum",
        5,
        &flipped(&original, 7, 0x10),
    )
    .unwrap();
    assert!(is_corrupt(
        &soul.replay_organism(&Profile::t1_ref()),
        "event",
        5
    ));
    set_blob(&tamper, "events", "checksum", 5, &original).unwrap();

    // The other columns are covered too.
    for (column, value) in [
        ("at_ms", 12_345),
        ("config_version", 2),
        ("reducer_version", 99),
    ] {
        let seq: i64 = 7;
        let before: i64 = tamper
            .query_row(
                &format!("SELECT {column} FROM events WHERE seq = 7"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        tamper
            .execute(
                &format!("UPDATE events SET {column} = ?1 WHERE seq = ?2"),
                [value, seq],
            )
            .unwrap();
        assert!(is_corrupt(&soul.read_after(0, 64), "event", 7), "{column}");
        tamper
            .execute(
                &format!("UPDATE events SET {column} = ?1 WHERE seq = ?2"),
                [before, seq],
            )
            .unwrap();
    }
    assert_eq!(
        soul.replay_organism(&Profile::t1_ref()).unwrap().0,
        organism
    );
}

#[test]
fn a_flipped_snapshot_byte_is_reported_with_its_sequence_number() {
    let directory = TestDirectory::new().unwrap();
    let (soul, organism) = soul_with(&directory, 10, SoulConfig::default(), Some(6)).unwrap();
    let tamper = directory.tamper().unwrap();
    let profile = Profile::t1_ref();
    let column = |name: &str| -> Vec<u8> {
        tamper
            .query_row(
                &format!("SELECT {name} FROM snapshots WHERE seq = 6"),
                [],
                |row| row.get(0),
            )
            .unwrap()
    };

    for name in ["blob", "chain", "checksum"] {
        let original = column(name);
        for at in 0..original.len() {
            set_blob(&tamper, "snapshots", name, 6, &flipped(&original, at, 0x01)).unwrap();
            assert!(
                is_corrupt(&soul.latest_snapshot(), "snapshot", 6),
                "{name}/{at}"
            );
            assert!(is_corrupt(&soul.replay_organism(&profile), "snapshot", 6));
        }
        set_blob(&tamper, "snapshots", name, 6, &original).unwrap();
    }

    // A snapshot moved to another event, or relabelled with another reducer.
    tamper
        .execute("UPDATE snapshots SET seq = 7 WHERE seq = 6", [])
        .unwrap();
    assert!(is_corrupt(&soul.replay_organism(&profile), "snapshot", 7));
    tamper
        .execute("UPDATE snapshots SET seq = 6 WHERE seq = 7", [])
        .unwrap();
    tamper
        .execute(
            "UPDATE snapshots SET reducer_version = reducer_version + 1",
            [],
        )
        .unwrap();
    assert!(is_corrupt(&soul.latest_snapshot(), "snapshot", 6));
    tamper
        .execute(
            "UPDATE snapshots SET reducer_version = reducer_version - 1",
            [],
        )
        .unwrap();
    assert_eq!(soul.replay_organism(&profile).unwrap().0, organism);

    // `enton why` starts from the earliest snapshot a pruned log continues.
    tamper
        .execute("DELETE FROM events WHERE seq <= 6", [])
        .unwrap();
    let original = column("blob");
    set_blob(
        &tamper,
        "snapshots",
        "blob",
        6,
        &flipped(&original, 40, 0x02),
    )
    .unwrap();
    let read_only = Soul::open_read_only(directory.database(), SoulConfig::default()).unwrap();
    assert!(is_corrupt(
        &read_only.earliest_organism(&profile),
        "snapshot",
        6
    ));
}

// ---------------------------------------------------------------------------
// Deleted, renumbered and swapped events

#[test]
fn a_deleted_event_is_detected() {
    let directory = TestDirectory::new().unwrap();
    let (soul, _) = soul_with(&directory, 10, SoulConfig::default(), None).unwrap();
    let tamper = directory.tamper().unwrap();
    tamper
        .execute("DELETE FROM events WHERE seq = 5", [])
        .unwrap();
    assert!(matches!(
        soul.read_after(0, 64),
        Err(Error::ReplayGap {
            expected: 5,
            found: 6
        })
    ));

    // Closing the gap by renumbering the later events breaks the chain instead.
    for seq in 6..=10 {
        tamper
            .execute("UPDATE events SET seq = ?1 WHERE seq = ?2", [seq - 1, seq])
            .unwrap();
    }
    assert!(is_corrupt(&soul.read_after(0, 64), "event", 5));
}

#[test]
fn events_deleted_from_the_end_are_detected() {
    let directory = TestDirectory::new().unwrap();
    let (soul, _) = soul_with(&directory, 10, SoulConfig::default(), Some(7)).unwrap();
    let tamper = directory.tamper().unwrap();
    // What is left is a valid chain; only the sequence numbers SQLite assigned remember more.
    tamper
        .execute("DELETE FROM events WHERE seq > 8", [])
        .unwrap();
    assert!(matches!(
        soul.append_event(&event(11)),
        Err(Error::Truncated {
            recorded: 10,
            found: 8
        })
    ));
    drop(soul);
    for opened in [
        Soul::open(directory.database(), SoulConfig::default()),
        Soul::open_read_only(directory.database(), SoulConfig::default()),
    ] {
        assert!(matches!(
            opened,
            Err(Error::Truncated {
                recorded: 10,
                found: 8
            })
        ));
    }
}

#[test]
fn swapped_events_are_detected() {
    let directory = TestDirectory::new().unwrap();
    let (soul, _) = soul_with(&directory, 10, SoulConfig::default(), None).unwrap();
    let tamper = directory.tamper().unwrap();
    // Whole rows, checksums included, trade places.
    tamper
        .execute_batch(
            "UPDATE events SET seq = -1 WHERE seq = 4;
             UPDATE events SET seq = 4 WHERE seq = 5;
             UPDATE events SET seq = 5 WHERE seq = -1;",
        )
        .unwrap();
    assert!(is_corrupt(&soul.read_after(0, 64), "event", 4));
    let read_only = Soul::open_read_only(directory.database(), SoulConfig::default()).unwrap();
    assert!(is_corrupt(&read_only.read_after(0, 64), "event", 4));
}

// ---------------------------------------------------------------------------
// Pruning

#[test]
fn a_pruned_log_verifies_from_its_anchor() {
    let directory = TestDirectory::new().unwrap();
    let config = SoulConfig {
        max_events: Some(3),
        max_snapshots: Some(2),
        ..SoulConfig::default()
    };
    let profile = Profile::t1_ref();
    let (soul, mut organism) = soul_with(&directory, 10, config.clone(), Some(8)).unwrap();
    assert_eq!(soul.retain().unwrap(), 7);
    // Event 8 chains from the anchor retention left for event 7.
    assert_eq!(soul.read_after(7, 64).unwrap().len(), 3);
    assert_eq!(soul.replay_organism(&profile).unwrap().0, organism);

    // Pruned right up to the snapshot: the anchor and the snapshot both stand
    // for event 17, and must agree.
    live(&soul, &mut organism, 11, 20, Some(17)).unwrap();
    assert_eq!(soul.retain().unwrap(), 10);
    drop(soul);
    let soul = Soul::open(directory.database(), config).unwrap();
    assert_eq!(soul.read_after(17, 64).unwrap().len(), 3);
    assert_eq!(soul.replay_organism(&profile).unwrap().0, organism);
    let read_only = Soul::open_read_only(directory.database(), SoulConfig::default()).unwrap();
    let (mut audited, cursor) = read_only.earliest_organism(&profile).unwrap();
    assert_eq!(cursor, 17);
    for (_, event) in read_only.read_after(cursor, 64).unwrap() {
        audited.step(&event);
    }
    assert_eq!(audited, organism);

    // A damaged anchor no longer matches the snapshot at its event.
    let tamper = directory.tamper().unwrap();
    let anchor: Vec<u8> = tamper
        .query_row("SELECT checksum FROM chain_anchor", [], |row| row.get(0))
        .unwrap();
    tamper
        .execute(
            "UPDATE chain_anchor SET checksum = ?1",
            [flipped(&anchor, 3, 0x01)],
        )
        .unwrap();
    assert!(is_corrupt(&soul.read_after(17, 64), "event", 17));
    assert!(is_corrupt(&soul.replay_organism(&profile), "event", 17));
}

#[test]
fn a_log_pruned_to_nothing_keeps_its_chain() {
    let directory = TestDirectory::new().unwrap();
    let config = SoulConfig {
        max_events: Some(0),
        max_snapshots: Some(1),
        ..SoulConfig::default()
    };
    let profile = Profile::t1_ref();
    let (soul, mut organism) = soul_with(&directory, 6, config.clone(), Some(6)).unwrap();
    assert_eq!(soul.retain().unwrap(), 6);
    drop(soul);
    let soul = Soul::open(directory.database(), config).unwrap();
    live(&soul, &mut organism, 7, 8, None).unwrap();
    assert_eq!(soul.read_after(6, 64).unwrap().len(), 2);
    assert_eq!(soul.replay_organism(&profile).unwrap().0, organism);
}

#[test]
fn pruning_never_erases_damage() {
    let directory = TestDirectory::new().unwrap();
    let config = SoulConfig {
        max_events: Some(3),
        ..SoulConfig::default()
    };
    let (soul, _) = soul_with(&directory, 10, config, Some(8)).unwrap();
    let tamper = directory.tamper().unwrap();
    let original = event_bytes(&tamper, "payload_json", 2).unwrap();
    set_payload(&tamper, 2, &flipped(&original, 3, 0x01)).unwrap();
    assert!(is_corrupt(&soul.retain(), "event", 2));
    assert_eq!(
        soul.read_after(0, 1).unwrap().len(),
        1,
        "nothing was dropped"
    );
}

// ---------------------------------------------------------------------------
// Migration

/// Write a log as schema `version` (2 or 3) left it: events `first..=last` of
/// [`event`], and with schema 3 the given organism snapshots.
fn legacy_soul(
    directory: &TestDirectory,
    version: u32,
    first: u32,
    last: u32,
    snapshots: &[(SeqNo, &Organism)],
) -> TestResult {
    let conn = Connection::open(directory.database())?;
    conn.execute_batch(
        "CREATE TABLE events (
             seq INTEGER PRIMARY KEY AUTOINCREMENT,
             at_ms INTEGER NOT NULL,
             payload_json TEXT NOT NULL,
             reducer_version INTEGER NOT NULL,
             config_version INTEGER NOT NULL
         );
         CREATE TABLE actions (
             thought_id INTEGER PRIMARY KEY,
             status TEXT NOT NULL CHECK (status IN ('pending', 'done', 'failed')),
             created_seq INTEGER NOT NULL,
             result_json TEXT
         );
         CREATE INDEX idx_actions_pending ON actions (status, created_seq);",
    )?;
    if version >= 3 {
        conn.execute_batch(
            "CREATE TABLE snapshots (
                 seq INTEGER PRIMARY KEY,
                 reducer_version INTEGER NOT NULL,
                 blob BLOB NOT NULL
             );",
        )?;
    }
    conn.pragma_update(None, "user_version", version)?;
    for index in 1..=last {
        let event = event(index);
        conn.execute(
            "INSERT INTO events (seq, at_ms, payload_json, reducer_version, config_version) \
             VALUES (?1, ?2, ?3, ?4, 1)",
            rusqlite::params![
                index,
                i64::try_from(event.now().0)?,
                serde_json::to_string(&event.canonical())?,
                enton_core::REDUCER_VERSION
            ],
        )?;
    }
    for (seq, organism) in snapshots {
        let blob = serde_json::to_vec(&serde_json::json!({
            "format": "enton-organism-v1",
            "state": organism,
        }))?;
        conn.execute(
            "INSERT INTO snapshots (seq, reducer_version, blob) VALUES (?1, ?2, ?3)",
            rusqlite::params![i64::try_from(*seq)?, enton_core::REDUCER_VERSION, blob],
        )?;
    }
    // Older code pruned without an anchor.
    conn.execute("DELETE FROM events WHERE seq < ?1", [first])?;
    Ok(())
}

/// The organism after events `1..=count`.
fn lived(count: u32) -> TestResult<Organism> {
    let mut organism = Organism::new(Profile::t1_ref())?;
    for index in 1..=count {
        organism.step(&event(index));
    }
    Ok(organism)
}

#[test]
fn a_schema_2_soul_migrates_and_verifies() {
    let directory = TestDirectory::new().unwrap();
    legacy_soul(&directory, 2, 1, 8, &[]).unwrap();
    let soul = Soul::open(directory.database(), SoulConfig::default()).unwrap();
    let version: u32 = soul_version(&directory);
    assert_eq!(version, 5);
    let read = soul.read_after(0, 64).unwrap();
    assert_eq!(read.len(), 8);
    for (seq, stored) in &read {
        assert_eq!(stored, &event(u32::try_from(*seq).unwrap()).canonical());
    }
    let profile = Profile::t1_ref();
    let mut organism = lived(8).unwrap();
    assert_eq!(soul.replay_organism(&profile).unwrap().0, organism);

    // New events continue the chain the migration started.
    live(&soul, &mut organism, 9, 10, Some(10)).unwrap();
    drop(soul);
    let soul = Soul::open_read_only(directory.database(), SoulConfig::default()).unwrap();
    assert_eq!(soul.read_after(0, 64).unwrap().len(), 10);
    assert_eq!(soul.replay_organism(&profile).unwrap().0, organism);
    // Opened immutable (no writer had it open), it would not see a later edit.
    drop(soul);

    let tamper = directory.tamper().unwrap();
    let original = event_bytes(&tamper, "payload_json", 3).unwrap();
    set_payload(&tamper, 3, &flipped(&original, 5, 0x01)).unwrap();
    let soul = Soul::open_read_only(directory.database(), SoulConfig::default()).unwrap();
    assert!(is_corrupt(&soul.read_after(0, 64), "event", 3));
}

#[test]
fn a_pruned_schema_3_soul_migrates_and_verifies() {
    let directory = TestDirectory::new().unwrap();
    let (at_four, at_seven) = (lived(4).unwrap(), lived(7).unwrap());
    legacy_soul(&directory, 3, 5, 10, &[(4, &at_four), (7, &at_seven)]).unwrap();
    let profile = Profile::t1_ref();
    let soul = Soul::open(directory.database(), SoulConfig::default()).unwrap();
    assert_eq!(soul_version(&directory), 5);

    // The earliest kept event chains from the anchor the migration left.
    let (mut audited, cursor) = soul.earliest_organism(&profile).unwrap();
    assert_eq!((cursor, &audited), (4, &at_four));
    for (_, event) in soul.read_after(cursor, 64).unwrap() {
        audited.step(&event);
    }
    let mut organism = lived(10).unwrap();
    assert_eq!(audited, organism);
    assert_eq!(soul.replay_organism(&profile).unwrap().0, organism);

    live(&soul, &mut organism, 11, 11, None).unwrap();
    drop(soul);
    let soul = Soul::open(directory.database(), SoulConfig::default()).unwrap();
    assert_eq!(soul.read_after(4, 64).unwrap().len(), 7);
    assert_eq!(soul.replay_organism(&profile).unwrap().0, organism);

    let tamper = directory.tamper().unwrap();
    let original = event_bytes(&tamper, "payload_json", 6).unwrap();
    set_payload(&tamper, 6, &flipped(&original, 5, 0x01)).unwrap();
    assert!(is_corrupt(&soul.read_after(4, 64), "event", 6));
}

fn soul_version(directory: &TestDirectory) -> u32 {
    Connection::open(directory.database())
        .and_then(|conn| conn.query_row("PRAGMA user_version", [], |row| row.get(0)))
        .unwrap_or(0)
}

#[test]
fn the_error_names_the_damaged_record() {
    let damaged = Error::Corrupt {
        kind: "event",
        seq: 42,
    };
    assert_eq!(
        damaged.to_string(),
        "damaged soul: the event at sequence number 42 fails its checksum"
    );
}

// ---------------------------------------------------------------------------
// Personas

/// A persona whose hash is `id` repeated.
fn persona(id: u8, source: PersonaSource) -> PersonaDigest {
    PersonaDigest {
        sha256: [id; 32],
        bytes: 500 + u64::from(id),
        source,
    }
}

/// A soul holding events `1..=10` in which thought 1, decided at event 2, was
/// asked with the built-in persona and thought 2, at event 5, with a file.
fn soul_with_personas(directory: &TestDirectory) -> TestResult<Soul> {
    let (soul, _) = soul_with(directory, 10, SoulConfig::default(), None)?;
    soul.record_pending(ThoughtId(1), 2, &persona(1, PersonaSource::BuiltIn))?;
    soul.record_pending(ThoughtId(2), 5, &persona(2, PersonaSource::File))?;
    Ok(soul)
}

/// Overwrite a column of the file persona's record, first used at event 5.
fn set_persona<T: rusqlite::ToSql>(conn: &Connection, column: &str, value: T) -> TestResult {
    conn.execute(
        &format!("UPDATE personas SET {column} = ?1 WHERE source = 'file'"),
        [value],
    )?;
    Ok(())
}

/// Opening the soul, listing its personas and reading thought 2's persona all
/// fail on the persona record at `seq`, while its events still read.
fn refused(directory: &TestDirectory, soul: &Soul, seq: SeqNo) -> bool {
    let config = SoulConfig::default;
    is_corrupt(&Soul::open(directory.database(), config()), "persona", seq)
        && is_corrupt(
            &Soul::open_read_only(directory.database(), config()),
            "persona",
            seq,
        )
        && is_corrupt(&soul.personas(), "persona", seq)
        && is_corrupt(&soul.thought_persona(ThoughtId(2)), "persona", seq)
        && soul
            .read_after(0, 64)
            .is_ok_and(|events| events.len() == 10)
}

#[test]
fn a_tampered_persona_record_is_reported_with_the_event_it_was_first_used_at() {
    let directory = TestDirectory::new().unwrap();
    let soul = soul_with_personas(&directory).unwrap();
    let tamper = directory.tamper().unwrap();
    for column in ["sha256", "chain", "checksum"] {
        let original: Vec<u8> = tamper
            .query_row(
                &format!("SELECT {column} FROM personas WHERE source = 'file'"),
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(original.len(), 32);
        for at in 0..original.len() {
            set_persona(&tamper, column, flipped(&original, at, 0x01)).unwrap();
            assert!(refused(&directory, &soul, 5), "{column}/{at}");
        }
        set_persona(&tamper, column, original).unwrap();
    }

    // Its length, its origin (a file passed off as the built-in persona) and
    // the event it was first used at are covered too.
    set_persona(&tamper, "bytes", 503).unwrap();
    assert!(refused(&directory, &soul, 5));
    set_persona(&tamper, "bytes", 502).unwrap();
    tamper
        .execute(
            "UPDATE personas SET source = 'built-in' WHERE first_seq = 5",
            [],
        )
        .unwrap();
    assert!(is_corrupt(&soul.personas(), "persona", 5));
    tamper
        .execute(
            "UPDATE personas SET source = 'file' WHERE first_seq = 5",
            [],
        )
        .unwrap();
    set_persona(&tamper, "first_seq", 6).unwrap();
    assert!(refused(&directory, &soul, 6));
    set_persona(&tamper, "first_seq", 5).unwrap();

    // Put back, everything reads again.
    assert_eq!(soul.personas().unwrap().len(), 2);
    assert!(soul.thought_persona(ThoughtId(2)).unwrap().is_some());
    Soul::open(directory.database(), SoulConfig::default()).unwrap();
}

#[test]
fn a_persona_record_from_another_history_is_refused() {
    let ours = TestDirectory::new().unwrap();
    let soul = soul_with_personas(&ours).unwrap();
    // Another soul, where the same persona spoke at the same event of another life.
    let theirs = TestDirectory::new().unwrap();
    let other = Soul::open(theirs.database(), SoulConfig::default()).unwrap();
    for index in 11..=20 {
        other.append_event(&event(index)).unwrap();
    }
    other
        .record_pending(ThoughtId(9), 5, &persona(2, PersonaSource::File))
        .unwrap();
    let (chain, checksum): (Vec<u8>, Vec<u8>) = theirs
        .tamper()
        .unwrap()
        .query_row("SELECT chain, checksum FROM personas", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();

    // Copied in, the record is valid on its own but bound to an event this log
    // never held, and the event it names is intact: the record is to blame.
    let tamper = ours.tamper().unwrap();
    set_persona(&tamper, "chain", chain).unwrap();
    set_persona(&tamper, "checksum", checksum).unwrap();
    assert!(refused(&ours, &soul, 5));
}

#[test]
fn a_damaged_event_under_a_persona_record_is_named_as_the_event() {
    let directory = TestDirectory::new().unwrap();
    let soul = soul_with_personas(&directory).unwrap();
    let tamper = directory.tamper().unwrap();
    let original = event_bytes(&tamper, "checksum", 5).unwrap();
    set_blob(
        &tamper,
        "events",
        "checksum",
        5,
        &flipped(&original, 0, 0x01),
    )
    .unwrap();
    assert!(is_corrupt(&soul.personas(), "event", 5));
    assert!(is_corrupt(&soul.read_after(0, 64), "event", 5));
    assert!(is_corrupt(
        &Soul::open(directory.database(), SoulConfig::default()),
        "event",
        5
    ));
}

#[test]
fn a_tampered_link_from_a_thought_to_its_persona_is_reported() {
    let directory = TestDirectory::new().unwrap();
    let soul = soul_with_personas(&directory).unwrap();
    let tamper = directory.tamper().unwrap();
    tamper
        .execute_batch("CREATE TEMP TABLE kept AS SELECT * FROM actions;")
        .unwrap();
    let put_back = || {
        tamper
            .execute_batch("DELETE FROM actions; INSERT INTO actions SELECT * FROM kept;")
            .unwrap();
        assert!(soul.thought_persona(ThoughtId(2)).unwrap().is_some());
    };

    for (edit, seq) in [
        // The file persona's thought relinked to the built-in persona's record.
        (
            "UPDATE actions SET persona = (SELECT persona FROM kept WHERE thought_id = 1) \
             WHERE thought_id = 2",
            5,
        ),
        ("UPDATE actions SET persona = 99 WHERE thought_id = 2", 5),
        ("UPDATE actions SET persona = NULL WHERE thought_id = 2", 5),
        (
            "UPDATE actions SET persona_checksum = zeroblob(32) WHERE thought_id = 2",
            5,
        ),
        (
            "UPDATE actions SET persona_checksum = zeroblob(31) WHERE thought_id = 2",
            5,
        ),
        // Moved to another event, or given another thought's number and link.
        ("UPDATE actions SET created_seq = 6 WHERE thought_id = 2", 6),
        (
            "DELETE FROM actions WHERE thought_id = 2; \
             UPDATE actions SET thought_id = 2 WHERE thought_id = 1",
            2,
        ),
    ] {
        tamper.execute_batch(edit).unwrap();
        assert!(
            is_corrupt(&soul.thought_persona(ThoughtId(2)), "thought", seq),
            "{edit}"
        );
        assert!(
            is_corrupt(&soul.thought_before(ThoughtId(3)), "thought", seq),
            "{edit}"
        );
        put_back();
    }

    // A deleted record leaves its thought's link pointing at nothing.
    tamper
        .execute("DELETE FROM personas WHERE source = 'file'", [])
        .unwrap();
    assert!(is_corrupt(
        &soul.thought_persona(ThoughtId(2)),
        "thought",
        5
    ));
}
