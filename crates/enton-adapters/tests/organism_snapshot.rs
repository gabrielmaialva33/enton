//! Snapshot + retained-tail recovery through the public soul/core APIs.
#![cfg(feature = "soul")]
// Test fixtures may fail loudly; the quality bar permits unwrap in tests.
#![allow(clippy::unwrap_used)]

use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use enton_adapters::{MonotonicClock, Soul, SoulConfig, soul::Error};
use enton_core::{
    Abstention, Action, BodySignals, Event, Millis, Organism, Profile, SpeechCue, ThoughtId,
};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let id = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("enton-b2-{}-{id}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn database(&self) -> PathBuf {
        self.0.join("soul.sqlite")
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn speech(now: u64, keyword: bool, duration_ms: u32) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 0.83,
            vad_confidence: 0.91,
            duration_ms,
            keyword,
            speaker_sim: None,
            media: None,
            turn_complete: None,
            directed: None,
            direction: None,
        },
    }
}

fn events() -> Vec<Event> {
    vec![
        Event::Tick { now: Millis(100) },
        speech(200, false, 1300),
        speech(300, false, 1300),
        Event::Tick { now: Millis(1100) },
        speech(1200, true, 400), // Snapshot while Attend is pending.
        speech(1500, false, 1100),
        Event::CortexReply {
            now: Millis(1700),
            thought: ThoughtId(2),
            text: "Ready".into(),
        },
        speech(2200, false, 800),
        Event::Body {
            now: Millis(2300),
            signals: BodySignals {
                temperature_c: Some(95.0),
                ..BodySignals::default()
            },
        },
        speech(2400, false, 800), // Torpor and the active attention deadline persist.
        speech(3000, true, 300),
        Event::Tick { now: Millis(8000) }, // Pending keyword times out.
        Event::Body {
            now: Millis(8100),
            signals: BodySignals::default(),
        },
        Event::Tick {
            now: Millis(18_100),
        },
        speech(18_200, false, 1200),
    ]
}

fn thought_ids(actions: &[Action]) -> Vec<ThoughtId> {
    actions
        .iter()
        .filter_map(|action| match action {
            Action::Think { thought, .. } => Some(*thought),
            _ => None,
        })
        .collect()
}

#[test]
fn every_snapshot_boundary_survives_pruning_restart_and_replay() {
    let tape = events();
    for cut in 1..=tape.len() {
        let directory = TestDirectory::new();
        let config = SoulConfig {
            max_events: Some(tape.len() - cut),
            max_snapshots: Some(1),
            ..SoulConfig::default()
        };
        let mut expected = Organism::new(Profile::t1_ref()).unwrap();
        let mut expected_tail = Vec::new();
        {
            let soul = Soul::open(directory.database(), config.clone()).unwrap();
            for (index, event) in tape.iter().enumerate() {
                let seq = soul.append_event(event).unwrap();
                let actions = expected.step(event);
                if index >= cut {
                    expected_tail.extend(actions);
                }
                if index + 1 == cut {
                    soul.save_organism_snapshot(seq, &expected).unwrap();
                }
            }
            assert_eq!(soul.retain().unwrap(), cut);
            let rows = soul.read_after(cut as u64, tape.len()).unwrap();
            assert_eq!(rows.len(), tape.len() - cut);
            assert!(rows.iter().all(|(seq, _)| *seq > cut as u64));
        } // Close SQLite, including WAL, before reopening.

        let soul = Soul::open(directory.database(), config).unwrap();
        let (mut restored, tail) = soul.replay_organism(&Profile::t1_ref()).unwrap();
        assert_eq!(restored, expected, "state at cut {cut}");
        assert_eq!(
            thought_ids(&tail),
            thought_ids(&expected_tail),
            "IDs at cut {cut}"
        );
        assert_eq!(tail, expected_tail, "actions at cut {cut}");

        // Recovery must also preserve the next ID, not just past actions.
        let next = speech(19_000, true, 1200);
        let expected_actions = expected.step(&next);
        let actual_actions = restored.step(&next);
        assert!(!thought_ids(&actual_actions).is_empty());
        assert_eq!(actual_actions, expected_actions);
        assert_eq!(restored, expected);
    }
}

#[test]
fn a_restored_organism_resumes_time_where_it_stopped() {
    let directory = TestDirectory::new();
    let five_hours = 5 * 3_600_000;
    {
        let soul = Soul::open(directory.database(), SoulConfig::default()).unwrap();
        let mut organism = Organism::new(Profile::desktop()).unwrap();
        let mut last_seq = 0;
        for event in [
            Event::Tick {
                now: Millis(five_hours),
            },
            // A paid thought after the last tick: the resume point must see it.
            speech(five_hours + 500, false, 1000),
        ] {
            last_seq = soul.append_event(&event).unwrap();
            organism.step(&event);
        }
        soul.save_organism_snapshot(last_seq, &organism).unwrap();
    }

    let soul = Soul::open(directory.database(), SoulConfig::default()).unwrap();
    let (restored, _) = soul.replay_organism(&Profile::desktop()).unwrap();
    assert_eq!(restored.last_seen(), Millis(five_hours + 500));

    // A clock restarting at zero reads as time running backward: stuck in cooldown.
    let mut naive = restored.clone();
    assert!(matches!(
        naive.step(&speech(60_000, false, 1000)).as_slice(),
        [Action::Abstain {
            why: Abstention::Cooldown,
            ..
        }]
    ));

    // The resumed clock keeps counting, so the same cue a minute later is paid for.
    let mut resumed = restored;
    let now = MonotonicClock::resuming_at(resumed.last_seen()).now();
    assert!(now >= resumed.last_seen());
    let actions = resumed.step(&speech(now.0 + 60_000, false, 1000));
    assert_eq!(thought_ids(&actions), vec![ThoughtId(2)]);

    let before = resumed.discretionary_budget().available;
    resumed.step(&Event::Tick {
        now: Millis(now.0 + 3_600_000),
    });
    assert!(resumed.discretionary_budget().available > before);
}

#[test]
fn replay_without_snapshot_and_across_multiple_tail_pages_is_identical() {
    let directory = TestDirectory::new();
    let soul = Soul::open(directory.database(), SoulConfig::default()).unwrap();
    let mut expected = Organism::new(Profile::desktop()).unwrap();
    let mut actions = Vec::new();
    assert_eq!(
        soul.replay_organism(&Profile::desktop()).unwrap(),
        (expected.clone(), vec![])
    );
    for index in 0..2100 {
        let event = if index % 200 == 0 {
            speech(index * 1000, true, 1200)
        } else {
            Event::Tick {
                now: Millis(index * 1000),
            }
        };
        soul.append_event(&event).unwrap();
        actions.extend(expected.step(&event));
    }
    assert_eq!(
        soul.replay_organism(&Profile::desktop()).unwrap(),
        (expected, actions)
    );
}

#[test]
fn retention_preserves_unsnapshotted_tail_and_reports_unreachable_caps() {
    let directory = TestDirectory::new();
    let soul = Soul::open(
        directory.database(),
        SoulConfig {
            max_events: Some(0),
            ..SoulConfig::default()
        },
    )
    .unwrap();
    let mut expected = Organism::new(Profile::t1_ref()).unwrap();
    let event = speech(0, true, 400);
    let seq = soul.append_event(&event).unwrap();
    expected.step(&event);
    assert!(matches!(soul.retain(), Err(Error::RetentionCap(_))));
    soul.save_organism_snapshot(seq, &expected).unwrap();
    let event = speech(300, false, 1000);
    soul.append_event(&event).unwrap();
    let actions = expected.step(&event);
    assert!(matches!(soul.retain(), Err(Error::RetentionCap(_))));
    assert_eq!(soul.read_after(seq, 10).unwrap().len(), 1);
    assert_eq!(
        soul.replay_organism(&Profile::t1_ref()).unwrap(),
        (expected, actions)
    );
}

#[test]
fn corrupt_unknown_format_and_mismatched_profiles_do_not_fall_back() {
    let directory = TestDirectory::new();
    let soul = Soul::open(directory.database(), SoulConfig::default()).unwrap();
    let event = speech(0, true, 1200);
    let seq = soul.append_event(&event).unwrap();
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    organism.step(&event);
    soul.save_organism_snapshot(seq, &organism).unwrap();
    let mut changed = Profile::t1_ref();
    changed.attention.attention_ms += 1;
    assert!(matches!(
        soul.replay_organism(&changed),
        Err(Error::Json(_))
    ));
    for blob in [
        b"broken".as_slice(),
        br#"{"format":"enton-organism-v999","state":{}}"#,
    ] {
        soul.save_snapshot(seq, blob).unwrap();
        assert!(matches!(
            soul.replay_organism(&Profile::t1_ref()),
            Err(Error::Json(_))
        ));
    }
}

#[test]
fn snapshots_preserve_custom_profile_names_and_policy() {
    let directory = TestDirectory::new();
    let soul = Soul::open(directory.database(), SoulConfig::default()).unwrap();
    let mut profile = Profile::t1_ref();
    profile.name = "custom-owned-policy".into();
    profile.budgets.obligation_budget_per_hour = 17.0;
    profile.attention.attention_ms = 2345;
    let mut organism = Organism::new(profile.clone()).unwrap();
    let event = speech(123, true, 300);
    let seq = soul.append_event(&event).unwrap();
    organism.step(&event);
    soul.save_organism_snapshot(seq, &organism).unwrap();
    assert_eq!(soul.replay_organism(&profile).unwrap(), (organism, vec![]));
}

#[test]
fn snapshot_recovery_replays_multiple_pages_after_reopening() {
    let directory = TestDirectory::new();
    let config = SoulConfig {
        max_events: Some(1200),
        max_snapshots: Some(1),
        ..SoulConfig::default()
    };
    let mut expected = Organism::new(Profile::t1_ref()).unwrap();
    let mut expected_tail = Vec::new();
    {
        let soul = Soul::open(directory.database(), config.clone()).unwrap();
        for index in 0..1401 {
            let event = if index % 100 == 0 {
                speech(index * 1000, true, 1200)
            } else {
                Event::Tick {
                    now: Millis(index * 1000),
                }
            };
            let seq = soul.append_event(&event).unwrap();
            let actions = expected.step(&event);
            if index == 200 {
                soul.save_organism_snapshot(seq, &expected).unwrap();
            } else if index > 200 {
                expected_tail.extend(actions);
            }
        }
        assert_eq!(soul.retain().unwrap(), 201);
    }
    let soul = Soul::open(directory.database(), config).unwrap();
    assert_eq!(
        soul.replay_organism(&Profile::t1_ref()).unwrap(),
        (expected, expected_tail)
    );
}

#[test]
fn an_unreachable_byte_cap_is_reported_without_losing_the_snapshot() {
    let directory = TestDirectory::new();
    let soul = Soul::open(
        directory.database(),
        SoulConfig {
            max_db_bytes: Some(1),
            max_snapshots: Some(1),
            ..SoulConfig::default()
        },
    )
    .unwrap();
    let event = speech(0, true, 300);
    let seq = soul.append_event(&event).unwrap();
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    organism.step(&event);
    soul.save_organism_snapshot(seq, &organism).unwrap();
    let Error::RetentionCap(message) = soul.retain().unwrap_err() else {
        panic!("expected an explicit retention-cap error");
    };
    assert!(message.contains("size cap of 1 bytes"));
    assert_eq!(
        soul.replay_organism(&Profile::t1_ref()).unwrap(),
        (organism, vec![])
    );
}

#[test]
fn replay_organism_rejects_invalid_profile() {
    let directory = TestDirectory::new();
    let soul = Soul::open(directory.database(), SoulConfig::default()).unwrap();
    let mut invalid = Profile::t1_ref();
    invalid.ignition.hysteresis = invalid.ignition.threshold;
    assert!(matches!(
        soul.replay_organism(&invalid),
        Err(Error::InvalidProfile(_))
    ));
}
