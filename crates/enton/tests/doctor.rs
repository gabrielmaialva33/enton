//! The doctor is a local-only, read-only check, isolated from the owner's files.
// Fixture helpers must fail the test on errors; AGENTS.md permits unwraps in tests.
#![allow(clippy::unwrap_used)]
use serde_json::Value;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!(
            "enton-doctor-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
    fn run(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_enton"))
            .args(["doctor", "--json"])
            .args(args)
            .env("HOME", self.0.join("home"))
            .env("XDG_CONFIG_HOME", self.0.join("config"))
            .env("XDG_DATA_HOME", self.0.join("data"))
            .output()
            .unwrap()
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.0));
    }
}
fn check<'a>(report: &'a Value, name: &str) -> &'a Value {
    report["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap()
}
#[test]
fn fresh_install_is_reported_without_creating_files_or_starting_inference() {
    let scratch = Scratch::new();
    let output = scratch.run(&[]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["mode"], "offline");
    assert_eq!(check(&report, "soul")["status"], "warning");
    assert_eq!(check(&report, "cortex")["status"], "not_evaluated");
    assert_eq!(check(&report, "audio")["status"], "not_evaluated");
    assert_eq!(std::fs::read_dir(&scratch.0).unwrap().count(), 0);
}
#[test]
fn an_explicit_missing_soul_is_an_error_not_a_new_database() {
    let scratch = Scratch::new();
    let path = scratch.0.join("missing/soul.sqlite");
    let output = scratch.run(&["--soul", path.to_str().unwrap()]);
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(check(&report, "soul")["status"], "error");
    assert!(!path.parent().unwrap().exists());
}
#[test]
fn invalid_persona_does_not_leak_its_contents() {
    let scratch = Scratch::new();
    let path = scratch.0.join("persona.md");
    let secret = "DO_NOT_PRINT_THIS_PERSONA".repeat(400);
    std::fs::write(&path, &secret).unwrap();
    let output = scratch.run(&["--persona", path.to_str().unwrap()]);
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(check(&report, "persona")["status"], "error");
    assert!(!String::from_utf8_lossy(&output.stdout).contains("DO_NOT_PRINT"));
    assert!(!String::from_utf8_lossy(&output.stderr).contains("DO_NOT_PRINT"));
    assert_eq!(std::fs::read_to_string(path).unwrap(), secret);
}
#[test]
fn corrupt_soul_is_reported_without_rewriting_it() {
    let scratch = Scratch::new();
    let path = scratch.0.join("soul.sqlite");
    std::fs::write(&path, b"not a database").unwrap();
    let output = scratch.run(&["--soul", path.to_str().unwrap()]);
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(check(&report, "soul")["status"], "error");
    assert_eq!(std::fs::read(path).unwrap(), b"not a database");
}

#[test]
fn a_valid_soul_is_audited_without_writes_and_wrong_profile_is_rejected() {
    use enton_adapters::{Soul, SoulConfig};
    use enton_core::{Event, Millis, Organism, Profile};
    let scratch = Scratch::new();
    let path = scratch.0.join("soul.sqlite");
    let mut organism = Organism::new(Profile::desktop()).unwrap();
    {
        let soul = Soul::open(&path, SoulConfig::default()).unwrap();
        let event = Event::Tick { now: Millis(1_000) };
        let seq = soul.append_event(&event).unwrap();
        organism.step(&event);
        soul.save_organism_snapshot(seq, &organism).unwrap();
    }
    let before = std::fs::read(&path).unwrap();
    let output = scratch.run(&["--soul", path.to_str().unwrap(), "--profile=desktop"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(check(&report, "soul")["status"], "ok");
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(
        std::fs::read_dir(&scratch.0).unwrap().count(),
        1,
        "doctor created SQLite sidecars"
    );
    let wrong = scratch.run(&["--soul", path.to_str().unwrap(), "--profile=t1-ref"]);
    assert!(!wrong.status.success());
    let report: Value = serde_json::from_slice(&wrong.stdout).unwrap();
    assert_eq!(check(&report, "soul")["status"], "error");
    assert_eq!(std::fs::read(path).unwrap(), before);
}

#[test]
fn incompatible_reducer_history_is_not_migrated_or_rewritten() {
    use enton_adapters::{Soul, SoulConfig};
    use enton_core::{Event, Millis, REDUCER_VERSION};
    let scratch = Scratch::new();
    let path = scratch.0.join("old.sqlite");
    {
        let soul = Soul::open(
            &path,
            SoulConfig {
                reducer_version: REDUCER_VERSION + 1,
                ..SoulConfig::default()
            },
        )
        .unwrap();
        soul.append_event(&Event::Tick { now: Millis(1) }).unwrap();
    }
    let before = std::fs::read(&path).unwrap();
    let output = scratch.run(&["--soul", path.to_str().unwrap()]);
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(check(&report, "soul")["status"], "error");
    assert!(
        check(&report, "soul")["message"]
            .as_str()
            .unwrap()
            .contains("Reducer history")
    );
    assert_eq!(std::fs::read(path).unwrap(), before);
}

#[test]
fn directories_and_live_runtime_options_are_rejected() {
    let scratch = Scratch::new();
    for flag in ["--persona", "--soul"] {
        let output = scratch.run(&[flag, scratch.0.to_str().unwrap()]);
        assert!(!output.status.success());
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            check(&report, flag.trim_start_matches('-'))["status"],
            "error"
        );
    }
    for args in [
        ["--voice", ""],
        ["--json=false", ""],
        ["--cortex-url", "http://127.0.0.1:9"],
    ] {
        let args: Vec<_> = args.into_iter().filter(|arg| !arg.is_empty()).collect();
        assert!(!scratch.run(&args).status.success());
    }
}
