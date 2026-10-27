//! Offline diagnosis must observe the real benchmark, never replace its policy.
// Fixture helpers must fail the test on errors; AGENTS.md permits unwraps in tests.
#![allow(clippy::unwrap_used)]
use serde_json::Value;
use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_e1-sim"))
        .args(args)
        .output()
        .unwrap()
}

fn diagnosis(extra: &[&str]) -> Value {
    let mut args = vec![
        "--seed",
        "100",
        "--diagnose",
        "e1a",
        "--episode",
        "1",
        "--json",
    ];
    args.extend_from_slice(extra);
    let out = run(&args);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

#[test]
fn diagnosis_is_repeatable_and_reports_real_episode_decisions() {
    let report = diagnosis(&[]);
    assert_eq!(report, diagnosis(&[]));
    assert_eq!(report["seed"], 100);
    assert_eq!(report["episode"], 1);
    assert!(!report["steps"].as_array().unwrap().is_empty());
    assert!(!report["turns"].as_array().unwrap().is_empty());
    assert_eq!(report["omitted_steps"], 0);
    assert!(
        report["steps"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| !s["actions"].as_array().unwrap().is_empty())
    );
}

#[test]
fn masked_sensors_do_not_reappear_in_diagnostic_events() {
    let mut speech_events = 0;
    for step in diagnosis(&[])["steps"].as_array().unwrap() {
        if let Some(speech) = step["event"].get("Speech") {
            speech_events += 1;
            assert!(speech["cue"]["directed"].is_null());
            assert!(speech["cue"]["direction"].is_null());
        }
    }
    assert!(
        speech_events > 0,
        "sensor assertions must exercise actual speech"
    );
}

#[test]
fn invalid_diagnostic_combinations_are_rejected() {
    for args in [
        vec!["--diagnose", "e1a"],
        vec!["--episode", "1"],
        vec!["--diagnose", "unknown", "--episode", "1"],
        vec![
            "--diagnose",
            "e1a",
            "--episode",
            "1",
            "--seeds",
            "100..=101",
        ],
        vec!["--diagnose", "e1a", "--episode", "1", "--summary"],
        vec!["--diagnose", "e1a", "--episode", "1", "--off-policy"],
        vec!["--diagnose", "e1a", "--episode", "4294967295"],
        vec!["--diagnose", "e1a", "--episode", "1", "--seed", "1000"],
    ] {
        assert!(!run(&args).status.success(), "accepted {args:?}");
    }
}

#[test]
fn observation_preserves_the_entire_ordinary_result_including_prehistory() {
    use enton_core::Profile;
    use enton_e1::diagnostics::diagnose_episode;
    use enton_e1::{Sensors, e1a, e1b, run_tape_with};
    let profile = Profile::t1_ref();
    for tape in [e1a(100).unwrap(), e1b(100).unwrap()] {
        for sensors in [
            Sensors::DEFAULT,
            Sensors::WITH_DIRECTED,
            Sensors::WITH_DIRECTION,
        ] {
            let ordinary = run_tape_with(&tape, &profile, sensors).unwrap();
            for turn in [tape.turns().first().unwrap(), tape.turns().last().unwrap()] {
                let diagnosis = diagnose_episode(&tape, &profile, sensors, turn.episode).unwrap();
                assert_eq!(diagnosis.result, ordinary.organism);
                for outcome in &diagnosis.turns {
                    let credited = ordinary.organism.thoughts.iter().any(|thought| {
                        thought.credit == enton_e1::run::Credit::Served(outcome.turn.id)
                    });
                    assert_eq!(outcome.served, credited);
                }
            }
        }
    }
}

#[test]
fn diagnostics_show_misses_rather_than_reclassifying_them_as_served() {
    use enton_core::Profile;
    use enton_e1::diagnostics::diagnose_episode;
    use enton_e1::{Sensors, e1a, run_tape};
    let tape = e1a(100).unwrap();
    let ordinary = run_tape(&tape).unwrap();
    let missed = tape
        .turns()
        .iter()
        .find(|turn| {
            !ordinary
                .organism
                .thoughts
                .iter()
                .any(|thought| thought.credit == enton_e1::run::Credit::Served(turn.id))
        })
        .unwrap();
    let diagnosis =
        diagnose_episode(&tape, &Profile::t1_ref(), Sensors::DEFAULT, missed.episode).unwrap();
    assert!(
        diagnosis
            .turns
            .iter()
            .any(|outcome| outcome.turn.id == missed.id && !outcome.served)
    );
    assert!(diagnosis.steps.iter().any(|step| !step.actions.is_empty()));
}
