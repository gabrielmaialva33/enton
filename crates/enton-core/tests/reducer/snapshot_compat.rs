//! A golden snapshot guards the organism's serialized form. The soul stores whole
//! organisms, so within one reducer version a layout change must not orphan them;
//! across versions the fixture is regenerated on purpose and its diff reviewed.

use enton_core::{
    BodySignals, Event, Millis, Organism, Profile, REDUCER_VERSION, SpeechCue, ThoughtId,
    UtteranceId,
};

/// Reducer version that wrote the fixture. After bumping `REDUCER_VERSION`, regenerate it
/// with `cargo test -p enton-core --test reducer -- --ignored regenerate_the_snapshot_fixture`
/// and review the diff: it shows exactly how the organism's state changed.
const FIXTURE_REDUCER_VERSION: u32 = 14;

/// State after `run_tape`, as the soul would store it.
const FIXTURE: &str = include_str!("../fixtures/organism-snapshot.json");

/// Fields removed on purpose without a reducer bump, as `(object, key)`: an empty
/// object name means the organism itself.
const REMOVED: &[(&str, &str)] = &[];

/// A cue with the speaker, media and end-of-turn sensors reporting:
/// `(speaker_sim, media, turn_complete)`.
fn cue(
    energy: f32,
    vad: f32,
    duration_ms: u32,
    keyword: bool,
    sensors: (f32, f32, f32),
) -> SpeechCue {
    let (speaker_sim, media, turn_complete) = sensors;
    SpeechCue {
        energy,
        duration_ms,
        vad_confidence: vad,
        keyword,
        speaker_sim: Some(speaker_sim),
        media: Some(media),
        turn_complete: Some(turn_complete),
        directed: None,
        direction: None,
    }
}

/// The same cue with a directedness detector's reading as well.
fn addressed(cue: SpeechCue, directed: f32) -> SpeechCue {
    SpeechCue {
        directed: Some(directed),
        ..cue
    }
}

/// A tape that leaves almost every piece of organism state non-trivial: a
/// checklist, a conversation, echo adaptation, a follow-up whose thought failed
/// (so the cortex backs off), a habituated TV whose direction the array read, an
/// hour of drives and a pending "Enton?" still waiting for its continuation.
fn run_tape(mut organism: Organism) -> Organism {
    let speech = |now: u64, cue: SpeechCue| Event::Speech {
        now: Millis(now),
        cue,
    };
    let mut events = vec![
        Event::Tick { now: Millis(0) },
        Event::Checklist {
            now: Millis(400),
            actionable: true,
        },
        Event::Body {
            now: Millis(500),
            signals: BodySignals {
                temperature_c: Some(55.0),
                battery: Some(0.8),
                cpu_load: 0.3,
            },
        },
        speech(1_000, cue(0.9, 0.9, 1_500, true, (0.9, 0.1, 0.9))),
        Event::CortexReply {
            now: Millis(2_000),
            thought: ThoughtId(1),
            text: "Oi, Gabriel!".into(),
        },
        Event::PlaybackStarted {
            now: Millis(2_100),
            utterance: UtteranceId(1),
        },
        speech(2_500, cue(0.45, 0.55, 250, false, (0.15, 0.4, 0.8))),
        Event::PlaybackFinished {
            now: Millis(3_000),
            utterance: UtteranceId(1),
            interrupted: false,
        },
        // An aside inside the window, then the follow-up, both read by a directedness detector.
        speech(
            3_200,
            addressed(cue(0.9, 0.9, 500, false, (0.85, 0.1, 0.9)), 0.1),
        ),
        speech(
            3_600,
            addressed(cue(0.9, 0.9, 1_200, false, (0.85, 0.1, 0.9)), 0.9),
        ),
        Event::CortexFailed {
            now: Millis(5_000),
            thought: ThoughtId(2),
        },
        Event::Tick {
            now: Millis(60_000),
        },
    ];
    for i in 0..10 {
        events.push(speech(
            70_000 + i * 4_000,
            SpeechCue {
                direction: Some([0.6, 0.8]),
                ..cue(0.8, 0.9, 1_500, false, (0.2, 0.9, 0.5))
            },
        ));
    }
    events.push(Event::Tick {
        now: Millis(3_600_000),
    });
    // An unfinished "Enton..." is still waiting when the snapshot is taken.
    events.push(speech(3_600_500, cue(0.9, 0.9, 400, true, (0.9, 0.1, 0.2))));

    for event in &events {
        organism.step(event);
    }
    organism
}

#[test]
fn the_fixture_was_written_by_the_current_reducer() {
    assert_eq!(
        FIXTURE_REDUCER_VERSION, REDUCER_VERSION,
        "REDUCER_VERSION changed: regenerate the snapshot fixture (see FIXTURE_REDUCER_VERSION)"
    );
}

#[test]
#[ignore = "rewrites the fixture; run on purpose after bumping REDUCER_VERSION"]
fn regenerate_the_snapshot_fixture() {
    let organism = run_tape(Organism::new(Profile::t1_ref()).unwrap());
    let json = serde_json::to_string_pretty(&organism).unwrap() + "\n";
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/organism-snapshot.json"
    );
    std::fs::write(path, json).unwrap();
}

#[test]
fn a_stored_snapshot_restores_to_the_same_state() {
    let organism = run_tape(Organism::new(Profile::t1_ref()).unwrap());
    let stored: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();

    let restored: Organism = serde_json::from_value(stored.clone()).unwrap();
    assert_eq!(
        restored, organism,
        "the stored state reads back as the live one"
    );

    // Restored mid-request, it finishes the caller's request exactly like the live one.
    let rest = Event::Speech {
        now: Millis(3_601_200),
        cue: addressed(cue(0.9, 0.9, 600, false, (0.9, 0.1, 0.9)), 0.9),
    };
    let (mut live, mut resumed) = (organism.clone(), restored);
    let finished = live.step(&rest);
    assert!(matches!(
        finished.as_slice(),
        [enton_core::Action::Think { .. }]
    ));
    assert_eq!(resumed.step(&rest), finished);
    assert_eq!(resumed, live);

    // The wire shape is unchanged, apart from the fields removed on purpose.
    let mut expected = stored;
    for (object, key) in REMOVED {
        let target = if object.is_empty() {
            &mut expected
        } else {
            &mut expected[*object]
        };
        target.as_object_mut().unwrap().remove(*key);
    }
    // Compare what the soul would write (text), not `to_value`, which widens f32 to f64.
    let written: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&organism).unwrap()).unwrap();
    assert_eq!(written, expected);
}
