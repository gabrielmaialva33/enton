//! The on-disk schema and its forward-only migrations.

use rusqlite::Connection;

use super::{Error, Soul};

impl Soul {
    pub(super) fn schema_version(conn: &Connection) -> Result<u32, Error> {
        let version: u32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        Ok(version)
    }

    pub(super) fn migrate(conn: &Connection) -> Result<(), Error> {
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
