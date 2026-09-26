//! Unit tests for the durable log: schema, pruning, retention and replay.

use enton_core::{Action, Event};

use super::*;
use enton_core::{Millis, SpeechCue};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

#[test]
fn pruned_without_anchor_is_rejected() {
    let path = temp_path("pruned_no_anchor");

    {
        let config = SoulConfig::default();
        let soul = Soul::open(&path, config).expect("open");
        soul.append_event(&tick(100)).expect("append");
        // we delete the events manually and advance the sqlite_sequence to simulate a pruned history without snapshots
        soul.conn.execute("DELETE FROM events", []).unwrap();
    }

    let config2 = SoulConfig::default();
    let err = Soul::open(&path, config2).unwrap_err();
    assert!(matches!(err, Error::PrunedWithoutAnchor));
}

#[test]
fn reject_zero_snapshot_cap_with_pruning() {
    let path = temp_path("zero_snap_cap");
    let config = SoulConfig {
        max_snapshots: Some(0),
        max_events: Some(10), // pruning enabled
        ..SoulConfig::default()
    };
    let err = Soul::open(&path, config).unwrap_err();
    assert!(matches!(err, Error::InvalidConfig(_)));
}

#[test]
fn incompatible_history_is_rejected() {
    let path = temp_path("incompatible_history");

    {
        let config = SoulConfig {
            reducer_version: 1,
            ..SoulConfig::default()
        };
        let soul = Soul::open(&path, config).expect("open");
        soul.append_event(&tick(100)).expect("append");
        soul.save_snapshot(1, b"snap").expect("save");
    }

    // Reopen with different version should fail
    let config2 = SoulConfig {
        reducer_version: 2,
        ..SoulConfig::default()
    };
    let err = Soul::open(&path, config2).unwrap_err();
    assert!(matches!(err, Error::IncompatibleHistory { .. }));
}

#[test]
fn snapshot_rejects_future_sequence() {
    let path = temp_path("snapshot_future_seq");
    let soul = Soul::open(&path, SoulConfig::default()).expect("open");

    soul.append_event(&tick(100)).expect("append 1");

    // Attempting to snapshot at seq 2 (future) should fail
    let err = soul.save_snapshot(2, b"blob").unwrap_err();
    assert!(matches!(err, Error::InvalidSnapshotSequence(_)));

    // Attempting to snapshot at seq 0 should also fail
    let err_zero = soul.save_snapshot(0, b"blob").unwrap_err();
    assert!(matches!(err_zero, Error::InvalidSnapshotSequence(_)));

    // Valid snapshot
    soul.save_snapshot(1, b"blob").expect("save 1");

    cleanup(&path);
}

#[test]
fn retain_exceeds_prune_batch_budget() {
    let path = temp_path("retain_budget");
    let config = SoulConfig {
        max_events: Some(10),
        ..SoulConfig::default()
    };
    let soul = Soul::open(&path, config).expect("open");

    // Add more than PRUNE_BATCH (512) events, e.g., 600
    for i in 1..=600 {
        soul.append_event(&tick(i)).expect("append");
    }

    // Save snapshot at 600
    soul.save_snapshot(600, b"snap").expect("save");

    // Retain should delete 590 events, which requires more than one batch of 512
    let deleted = soul.retain().expect("retain");
    assert_eq!(deleted, 590);
    assert_eq!(soul.event_count().unwrap(), 10);

    cleanup(&path);
}

#[test]
fn snapshots_are_pruned() {
    let path = temp_path("snapshots_prune");
    let config = SoulConfig {
        max_snapshots: Some(2),
        ..SoulConfig::default()
    };
    let soul = Soul::open(&path, config).expect("open");

    for i in 1..=4 {
        soul.append_event(&tick(i * 100)).expect("append");
        soul.save_snapshot(i, b"snap").expect("save");
    }

    // Before retain, we have 4 snapshots
    let snap_count: i64 = soul
        .conn
        .query_row("SELECT COUNT(*) FROM snapshots", [], |r| r.get(0))
        .unwrap();
    assert_eq!(snap_count, 4);

    soul.retain().expect("retain");

    // After retain, only the latest 2 should remain (seq 3 and 4)
    let snap_count: i64 = soul
        .conn
        .query_row("SELECT COUNT(*) FROM snapshots", [], |r| r.get(0))
        .unwrap();
    assert_eq!(snap_count, 2);

    let seqs: Vec<i64> = soul
        .conn
        .prepare("SELECT seq FROM snapshots ORDER BY seq ASC")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    assert_eq!(seqs, vec![3, 4]);

    cleanup(&path);
}

#[test]
fn migration_sets_auto_vacuum_full() {
    let path = temp_path("auto_vacuum");
    let soul = Soul::open(&path, SoulConfig::default()).expect("open");

    let auto_vacuum: i32 = soul
        .conn
        .query_row("PRAGMA auto_vacuum", [], |r| r.get(0))
        .unwrap();
    assert_eq!(auto_vacuum, 1, "auto_vacuum should be 1 (FULL)");

    cleanup(&path);
}

static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(tag: &str) -> PathBuf {
    let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("enton_soul_{id}_{tag}_{}", std::process::id()))
}

fn cleanup(path: &Path) {
    for suffix in ["", "-wal", "-shm"] {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        let parent = path
            .parent()
            .map_or(String::new(), |p| p.to_string_lossy().into_owned());
        if let Err(_err) = std::fs::remove_file(format!("{parent}/{name}{suffix}")) {}
    }
}

fn tick(now: u64) -> Event {
    Event::Tick { now: Millis(now) }
}

fn keyword_speech(now: u64) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 0.8,
            duration_ms: 900,
            vad_confidence: 0.9,
            keyword: true,
            speaker_sim: None,
            media: None,
            turn_complete: None,
        },
    }
}

fn seqs(path: &Path) -> Vec<SeqNo> {
    let conn = Connection::open(path).expect("reopen for read");
    let mut stmt = conn
        .prepare("SELECT seq FROM events ORDER BY seq ASC")
        .expect("prepare");
    let rows: Vec<i64> = stmt
        .query_map([], |row| row.get(0))
        .expect("query")
        .collect::<Result<Vec<_>, _>>()
        .expect("collect");
    rows.into_iter()
        .map(|s| u64::try_from(s).unwrap_or(0))
        .collect()
}

// Fake state for replay tests
#[derive(Debug, PartialEq, Eq, Default)]
struct FakeState {
    ticks: u64,
    speech_count: u64,
}

#[test]
fn replay_generic_with_fake_state() {
    let path = temp_path("replay_generic");
    let soul = Soul::open(&path, SoulConfig::default()).expect("open");

    soul.append_event(&tick(100)).expect("append");
    soul.append_event(&tick(200)).expect("append");
    soul.append_event(&keyword_speech(300)).expect("append");

    let init = || FakeState::default();
    let restore = |_blob: &[u8]| -> Result<FakeState, Error> {
        Ok(FakeState {
            ticks: 99,
            speech_count: 99,
        }) // not used since no snapshot
    };
    let step = |state: &mut FakeState, event: &Event| -> Vec<Action> {
        match event {
            Event::Tick { .. } => state.ticks += 1,
            Event::Speech { .. } => state.speech_count += 1,
            _ => {}
        }
        Vec::new()
    };

    let (state, actions) = soul.replay(init, restore, step).expect("replay");
    assert_eq!(
        state,
        FakeState {
            ticks: 2,
            speech_count: 1
        }
    );
    assert!(actions.is_empty());

    cleanup(&path);
}

#[test]
fn replay_from_snapshot() {
    let path = temp_path("snapshot_replay");
    let soul = Soul::open(&path, SoulConfig::default()).expect("open");

    soul.append_event(&tick(100)).expect("append"); // seq 1
    let snap_seq = soul.append_event(&tick(200)).expect("append"); // seq 2

    // Save snapshot at seq 2
    soul.save_snapshot(snap_seq, b"some_blob").expect("save");

    soul.append_event(&tick(300)).expect("append"); // seq 3
    soul.append_event(&keyword_speech(400)).expect("append"); // seq 4

    let init = || FakeState::default();
    let restore = |blob: &[u8]| -> Result<FakeState, Error> {
        assert_eq!(blob, b"some_blob");
        Ok(FakeState {
            ticks: 2,
            speech_count: 0,
        }) // restored state representing 2 ticks
    };
    let step = |state: &mut FakeState, event: &Event| -> Vec<Action> {
        match event {
            Event::Tick { .. } => state.ticks += 1,
            Event::Speech { .. } => state.speech_count += 1,
            _ => {}
        }
        Vec::new()
    };

    let (state, actions) = soul.replay(init, restore, step).expect("replay");
    // State should have 2 from snapshot + 1 tick from tail (seq 3) + 1 speech from tail (seq 4)
    assert_eq!(
        state,
        FakeState {
            ticks: 3,
            speech_count: 1
        }
    );
    assert!(actions.is_empty());

    cleanup(&path);
}

#[test]
fn replay_empty_tail_after_snapshot() {
    let path = temp_path("empty_tail");
    let soul = Soul::open(&path, SoulConfig::default()).expect("open");

    let snap_seq = soul.append_event(&tick(100)).expect("append"); // seq 1
    soul.save_snapshot(snap_seq, b"snap").expect("save");

    // No events after snapshot (empty tail)

    let init = || FakeState::default();
    let restore = |_blob: &[u8]| -> Result<FakeState, Error> {
        Ok(FakeState {
            ticks: 10,
            speech_count: 10,
        })
    };
    let step = |_state: &mut FakeState, _event: &Event| -> Vec<Action> { Vec::new() };

    let (state, _actions) = soul.replay(init, restore, step).expect("replay");
    assert_eq!(
        state,
        FakeState {
            ticks: 10,
            speech_count: 10
        }
    );

    cleanup(&path);
}

#[test]
fn retention_prunes_only_before_snapshot() {
    let path = temp_path("retention_prune");
    let config = SoulConfig {
        max_events: Some(2), // tiny cap to force pruning
        ..SoulConfig::default()
    };
    let soul = Soul::open(&path, config).expect("open");

    soul.append_event(&tick(100)).expect("append"); // seq 1
    soul.append_event(&tick(200)).expect("append"); // seq 2
    soul.append_event(&tick(300)).expect("append"); // seq 3
    soul.append_event(&tick(400)).expect("append"); // seq 4

    // Without snapshot, nothing is pruned because max_droppable_seq = 0
    let res = soul.retain();
    assert!(matches!(res, Err(Error::RetentionCap(_))));
    assert_eq!(seqs(&path), vec![1, 2, 3, 4]);

    // Take snapshot at seq 3
    soul.save_snapshot(3, b"snap").expect("save snapshot");

    // Now retain can prune up to seq 3
    let deleted = soul.retain().expect("retain success");
    // Target is 2 events. Total was 4. So it deletes seq 1 and seq 2.
    assert_eq!(deleted, 2);
    assert_eq!(seqs(&path), vec![3, 4]);

    cleanup(&path);
}

#[test]
fn retention_fails_if_cap_cannot_be_met() {
    let path = temp_path("retention_fail");
    let config = SoulConfig {
        max_events: Some(1), // cap is 1
        ..SoulConfig::default()
    };
    let soul = Soul::open(&path, config).expect("open");

    soul.append_event(&tick(100)).expect("append"); // 1
    soul.append_event(&tick(200)).expect("append"); // 2
    soul.append_event(&tick(300)).expect("append"); // 3

    // Snapshot at seq 1
    soul.save_snapshot(1, b"snap").expect("save");

    // Can only drop up to seq 1. We have 3 events, cap is 1. We need to drop 2, but can only drop 1.
    let res = soul.retain();
    assert!(matches!(res, Err(Error::RetentionCap(_))));

    // Seq 1 should be deleted, leaving 2 and 3.
    assert_eq!(seqs(&path), vec![2, 3]);

    cleanup(&path);
}

#[test]
fn v1_snapshot_is_rejected_as_incompatible_by_the_current_default() {
    let path = temp_path("v1_snapshot_incompatible");
    // Open soul with explicit reducer_version = 1 (simulating organism v1)
    let v1_config = SoulConfig {
        reducer_version: 1,
        ..SoulConfig::default()
    };
    let soul_v1 = Soul::open(&path, v1_config).expect("open v1");
    let seq = soul_v1.append_event(&tick(100)).expect("append");
    soul_v1
        .save_snapshot(seq, b"v1_state")
        .expect("save v1 snapshot");
    drop(soul_v1);

    // Opening with the default config (the current reducer) must reject the v1 snapshot.
    let err = Soul::open(&path, SoulConfig::default()).expect_err("should reject v1 snapshot");
    assert!(matches!(
        err,
        Error::IncompatibleHistory {
            kind: "snapshot",
            found: 1,
            expected,
        } if expected == enton_core::REDUCER_VERSION
    ));

    cleanup(&path);
}
