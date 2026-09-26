//! The on-disk schema and its forward-only migrations.

use rusqlite::Connection;

use super::{Error, Soul, chain};

/// `PRAGMA auto_vacuum` value for FULL.
const AUTO_VACUUM_FULL: u32 = 1;

impl Soul {
    pub(super) fn schema_version(conn: &Connection) -> Result<u32, Error> {
        let version: u32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        Ok(version)
    }

    pub(super) fn migrate(conn: &Connection) -> Result<(), Error> {
        if Self::schema_version(conn)? != Self::SCHEMA_VERSION {
            // Every schema change and the version that records it commit together:
            // a crash leaves the old schema or the new one, never half of each.
            let tx = conn.unchecked_transaction()?;
            Self::migrate_steps(&tx)?;
            tx.pragma_update(None, "user_version", Self::SCHEMA_VERSION)?;
            tx.commit()?;
        }
        // `auto_vacuum` only takes effect through a VACUUM, which cannot run in a
        // transaction; checked on every open, so a crash between the two heals.
        let auto_vacuum: u32 = conn.query_row("PRAGMA auto_vacuum", [], |row| row.get(0))?;
        if auto_vacuum != AUTO_VACUUM_FULL {
            conn.execute_batch("PRAGMA auto_vacuum = FULL; VACUUM;")?;
        }
        Ok(())
    }

    fn migrate_steps(conn: &Connection) -> Result<(), Error> {
        let mut current_version = Self::schema_version(conn)?;
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
                     ON actions (status, created_seq);",
            )?;
            current_version = 1;
        }

        if current_version == 1 {
            // Older code ran each statement of this step on its own, so a crash may
            // have left it half done. Finish an interrupted rename, drop a stray
            // copy, then copy again.
            if !Self::table_exists(conn, "events")? && Self::table_exists(conn, "events_v2")? {
                conn.execute_batch("ALTER TABLE events_v2 RENAME TO events;")?;
            }
            conn.execute_batch(
                "DROP TABLE IF EXISTS events_v2;
                 CREATE TABLE events_v2 (
                     seq INTEGER PRIMARY KEY AUTOINCREMENT,
                     at_ms INTEGER NOT NULL,
                     payload_json TEXT NOT NULL,
                     reducer_version INTEGER NOT NULL,
                     config_version INTEGER NOT NULL
                 );
                 INSERT INTO events_v2 (seq, at_ms, payload_json, reducer_version, config_version)
                     SELECT seq, at_ms, payload_json, reducer_version, config_version FROM events;
                 DROP TABLE events;
                 ALTER TABLE events_v2 RENAME TO events;",
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
            current_version = 3;
        }

        if current_version == 3 {
            // Checksums (see `chain.rs`). Rows already stored are trusted as they
            // stand now and get theirs computed here, in the same transaction.
            conn.execute_batch(
                "ALTER TABLE events ADD COLUMN checksum BLOB NOT NULL DEFAULT x'';
                 ALTER TABLE snapshots ADD COLUMN chain BLOB NOT NULL DEFAULT x'';
                 ALTER TABLE snapshots ADD COLUMN checksum BLOB NOT NULL DEFAULT x'';
                 CREATE TABLE chain_anchor (
                     id INTEGER PRIMARY KEY CHECK (id = 1),
                     seq INTEGER NOT NULL,
                     checksum BLOB NOT NULL
                 );",
            )?;
            chain::seal(conn)?;
            current_version = 4;
        }

        if current_version == 4 {
            // Personas (see `persona.rs`): one record per persona a thought was
            // asked with, and each thought's link to its record. The text is never
            // stored. Thoughts recorded before this step keep no persona.
            conn.execute_batch(
                "CREATE TABLE personas (
                     id INTEGER PRIMARY KEY,
                     sha256 BLOB NOT NULL,
                     bytes INTEGER NOT NULL,
                     source TEXT NOT NULL CHECK (source IN ('built-in', 'file')),
                     first_seq INTEGER NOT NULL,
                     chain BLOB NOT NULL,
                     checksum BLOB NOT NULL,
                     UNIQUE (sha256, source)
                 );
                 ALTER TABLE actions ADD COLUMN persona INTEGER;
                 ALTER TABLE actions ADD COLUMN persona_checksum BLOB;",
            )?;
        }
        Ok(())
    }

    fn table_exists(conn: &Connection, name: &str) -> Result<bool, Error> {
        let found: u32 = conn.query_row(
            "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
            [name],
            |row| row.get(0),
        )?;
        Ok(found > 0)
    }
}
