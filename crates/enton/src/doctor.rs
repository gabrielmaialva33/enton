//! Local-only diagnostics: no live runtime, migrations, model calls or audio devices.
use std::fmt;
use std::path::Path;

use enton_adapters::cortex::Persona;
use enton_adapters::{Soul, SoulConfig, soul};
use enton_core::{Organism, Profile};
use serde::Serialize;

use crate::cli::{DoctorConfig, default_persona_path};

const MAX_AUDIT_EVENTS: usize = 100_000;
const PAGE: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Status {
    Ok,
    Warning,
    Error,
    NotEvaluated,
}

#[derive(Debug, Serialize)]
struct Check {
    name: &'static str,
    status: Status,
    message: String,
}
impl Check {
    fn new(name: &'static str, status: Status, message: impl Into<String>) -> Self {
        Self {
            name,
            status,
            message: message.into(),
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct Report {
    schema_version: u32,
    mode: &'static str,
    profile: String,
    checks: Vec<Check>,
}
impl Report {
    pub(crate) fn has_errors(&self) -> bool {
        self.checks
            .iter()
            .any(|check| check.status == Status::Error)
    }
}
impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Enton doctor | offline, read-only | profile {}",
            self.profile
        )?;
        for check in &self.checks {
            writeln!(f, "  {:?} {}: {}", check.status, check.name, check.message)?;
        }
        writeln!(
            f,
            "No repairs, inference, audio capture or schema migrations were performed."
        )
    }
}

pub(crate) fn inspect(config: &DoctorConfig) -> Report {
    let profile = match Organism::new(config.profile.clone()) {
        Ok(_) => Check::new(
            "profile",
            Status::Ok,
            "Profile passes the core's validation.",
        ),
        Err(_) => Check::new(
            "profile",
            Status::Error,
            "Invalid profile; use a supported profile for this build.",
        ),
    };
    Report {
        schema_version: 1,
        mode: "offline",
        profile: config.profile.name.clone(),
        checks: vec![
            profile,
            check_persona(config.persona.as_deref()),
            check_soul(config),
            Check::new(
                "cortex",
                Status::NotEvaluated,
                "Endpoint reachability, model availability and response quality are not tested offline; no model was loaded.",
            ),
            Check::new(
                "audio",
                Status::NotEvaluated,
                "Microphone, echo cancellation, speech and physical latency require a separate, explicit audio test.",
            ),
            Check::new(
                "soak",
                Status::NotEvaluated,
                "24-hour RSS stability and real power consumption are not measured by this command.",
            ),
        ],
    }
}

fn check_persona(explicit: Option<&Path>) -> Check {
    let default = explicit.is_none().then(default_persona_path).flatten();
    let path = explicit.or(default.as_deref());
    let persona = match path {
        None => Ok(Persona::built_in()),
        Some(path) => match std::fs::metadata(path) {
            Ok(metadata) if metadata.is_file() => Persona::from_file(path),
            Err(error) if explicit.is_none() && error.kind() == std::io::ErrorKind::NotFound => {
                Ok(Persona::built_in())
            }
            _ => {
                return Check::new(
                    "persona",
                    Status::Error,
                    "Persona must be a readable regular file; check --persona or the config directory.",
                );
            }
        },
    };
    match persona {
        Ok(persona) => Check::new(
            "persona",
            Status::Ok,
            format!(
                "Valid persona ({} bytes); contents are not displayed.",
                persona.bytes()
            ),
        ),
        Err(_) => Check::new(
            "persona",
            Status::Error,
            "Cannot load persona; check UTF-8 encoding, permissions and the 8 KiB limit. Contents are not displayed.",
        ),
    }
}

fn check_soul(config: &DoctorConfig) -> Check {
    let Some(path) = config.soul.as_deref() else {
        return Check::new(
            "soul",
            Status::Warning,
            "No data directory configured; set HOME/XDG_DATA_HOME or pass --soul. Nothing was created.",
        );
    };
    match std::fs::metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Check::new(
                "soul",
                if config.explicit_soul {
                    Status::Error
                } else {
                    Status::Warning
                },
                "Soul does not exist here. Check --soul, or record a session later; nothing was created.",
            );
        }
        Ok(metadata) if metadata.is_file() => {}
        _ => {
            return Check::new(
                "soul",
                Status::Error,
                "Soul must be a readable regular SQLite file; check the path and permissions.",
            );
        }
    }
    match audit_soul(path, &config.profile) {
        Ok((events, true)) => Check::new(
            "soul",
            Status::Ok,
            format!(
                "Read-only replay verified {events} retained events and the compatible anchor; no historical actions were executed."
            ),
        ),
        Ok((events, false)) => Check::new(
            "soul",
            Status::Warning,
            format!(
                "Audit stopped at its {events}-event bound; remaining events are NOT verified. Use enton why for a full audit."
            ),
        ),
        Err(error) => Check::new("soul", Status::Error, soul_error(&error)),
    }
}

fn audit_soul(path: &Path, profile: &Profile) -> Result<(usize, bool), soul::Error> {
    let soul = Soul::open_read_only(path, SoulConfig::default())?;
    let (mut organism, mut cursor) = soul.earliest_organism(profile)?;
    let mut events = 0;
    loop {
        let remaining = MAX_AUDIT_EVENTS - events;
        let page = soul.read_after(cursor, PAGE.min(remaining + 1))?;
        if page.is_empty() {
            return Ok((events, true));
        }
        if page.len() > remaining {
            return Ok((events, false));
        }
        for (seq, event) in page {
            // Deterministic replay only: returned actions are deliberately never executed.
            drop(organism.step(&event));
            cursor = seq;
            events += 1;
        }
    }
}

fn soul_error(error: &soul::Error) -> String {
    // Do not print database payloads or decoder errors that may contain stored text.
    match error {
        soul::Error::UnsupportedSchema { found, expected } => format!("Schema {found} is incompatible with {expected}; use the matching build or an explicit backed-up migration. Doctor never migrates."),
        soul::Error::IncompatibleHistory { found, expected, .. } => format!("Reducer history {found} is incompatible with {expected}; use the matching build. History was not modified."),
        soul::Error::Corrupt { kind, seq } => format!("Checksum failure in {kind} at sequence {seq}; preserve the file and recover from a trusted backup."),
        soul::Error::ReplayGap { expected, found } => format!("Replay gap: expected sequence {expected}, found {found}; preserve the file and check retention/backups."),
        soul::Error::Json(_) | soul::Error::InvalidProfile(_) => "Stored state is incompatible or invalid; check --profile and the build that wrote it. Stored contents are not displayed.".into(),
        _ => "Soul cannot be verified; check integrity, permissions, schema and a trusted backup. No repairs were attempted.".into(),
    }
}

pub(crate) fn run(config: &DoctorConfig) -> std::process::ExitCode {
    let report = inspect(config);
    if config.json {
        if let Ok(text) = serde_json::to_string_pretty(&report) {
            println!("{text}");
        } else {
            eprintln!("Cannot serialize doctor report.");
            return std::process::ExitCode::FAILURE;
        }
    } else {
        print!("{report}");
    }
    if report.has_errors() {
        std::process::ExitCode::FAILURE
    } else {
        std::process::ExitCode::SUCCESS
    }
}
