//! `enton why`: replay the soul and explain the latest speech decisions.
//!
//! The soul keeps every reduced event and the reducer is deterministic, so
//! stepping an organism through the log recomputes each decision exactly,
//! together with the state it was made in. Replay starts as early as the log
//! allows (a fresh organism, or the earliest snapshot a pruned log continues).
//!
//! Only what the soul stores is shown: cue measurements, decisions, the
//! evidence recomputed from those measurements with the profile's calibration,
//! the organism's own state, how recorded thoughts resolved, and the persona
//! each was asked with (a hash, since the soul never keeps the persona's text).
//! The soul keeps no transcripts, and the text of Enton's replies is never read
//! back here.

mod audit;
mod explain;
mod render;

use std::path::PathBuf;

use enton_adapters::soul;

use self::audit::audit;
use self::render::render;
pub(crate) use crate::cli::WhyConfig;

/// Why the audit could not run.
#[derive(Debug, thiserror::Error)]
pub(crate) enum WhyError {
    /// No `--soul` and no home directory to derive the default location from.
    #[error("no soul path: set HOME or XDG_DATA_HOME, or pass --soul <PATH>")]
    NoPath,
    /// The log does not exist.
    #[error("no soul at {}: Enton has not recorded anything there yet", .0.display())]
    Missing(PathBuf),
    /// The log was written with an older schema, which only a live run migrates.
    #[error(
        "the soul at {} has schema {found}, older than this enton's {expected}: run enton on it once to migrate it (enton why only reads)",
        path.display()
    )]
    OldSchema {
        /// The log.
        path: PathBuf,
        /// Its schema version.
        found: u32,
        /// The version this build reads.
        expected: u32,
    },
    /// The log exists but could not be opened or replayed.
    #[error("cannot replay the soul at {}: {source}", path.display())]
    Soul {
        /// The log that failed.
        path: PathBuf,
        /// What failed.
        source: soul::Error,
    },
    /// Serializing the report failed.
    #[error("JSON output failed: {0}")]
    Json(#[from] serde_json::Error),
}

/// Replay the soul named by `config` and explain its latest speech cues, as
/// text or, with `--json`, as one JSON document.
pub(crate) fn run(config: &WhyConfig) -> Result<String, WhyError> {
    let path = config.soul.as_deref().ok_or(WhyError::NoPath)?;
    let report = audit(path, &config.profile, config.last, config.since_ms)?;
    if config.json {
        let mut out = serde_json::to_string_pretty(&report)?;
        out.push('\n');
        Ok(out)
    } else {
        Ok(render(&report))
    }
}
