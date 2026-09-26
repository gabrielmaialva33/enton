//! Snapshots written by an earlier reducer must keep restoring: the soul stores
//! whole organisms, and a layout change must not orphan them.

use enton_core::{
    BodySignals, Event, Millis, Organism, Profile, SpeechCue, ThoughtId, UtteranceId,
};

/// State after `run_tape`, serialized by reducer version 5 (flat profile, budgets
/// with a reserve).
const REDUCER_V5: &str = include_str!("../fixtures/organism-reducer-v5.json");

/// Fields later versions removed on purpose, as `(object, key)`: an empty
/// object name means the organism itself.
const REMOVED: &[(&str, &str)] = &[
    // Dropped when the profile was grouped: legacy budget aliases and the unused reserve.
    // Decisions did not change, so REDUCER_VERSION stays and old snapshots still restore.
    ("profile", "budget_per_hour"),
    ("profile", "reserve_fraction"),
    ("obligation_budget", "reserve"),
    ("discretionary_budget", "reserve"),
];

fn cue(energy: f32, vad: f32, duration_ms: u32, keyword: bool, speaker_sim: f32) -> SpeechCue {
    SpeechCue {
        energy,
        duration_ms,
        vad_confidence: vad,
        keyword,
        speaker_sim: Some(speaker_sim),
    }
}

/// A tape that leaves almost every piece of organism state non-trivial: a
/// conversation, echo adaptation, a habituated TV, an hour of drives and a
/// pending "Enton?" still waiting for its continuation.
fn run_tape(mut organism: Organism) -> Organism {
    let speech = |now: u64, cue: SpeechCue| Event::Speech {
        now: Millis(now),
        cue,
    };
    let mut events = vec![
        Event::Tick { now: Millis(0) },
        Event::Body {
            now: Millis(500),
            signals: BodySignals {
                temperature_c: Some(55.0),
                battery: Some(0.8),
                cpu_load: 0.3,
            },
        },
        speech(1_000, cue(0.9, 0.9, 1_500, true, 0.9)),
        Event::CortexReply {
            now: Millis(2_000),
            thought: ThoughtId(1),
            text: "Oi, Gabriel!".into(),
        },
        Event::PlaybackStarted {
            now: Millis(2_100),
            utterance: UtteranceId(1),
        },
        speech(2_500, cue(0.45, 0.55, 250, false, 0.15)),
        Event::PlaybackFinished {
            now: Millis(3_000),
            utterance: UtteranceId(1),
        },
        speech(3_600, cue(0.9, 0.9, 1_200, false, 0.85)),
        Event::Tick {
            now: Millis(60_000),
        },
    ];
    for i in 0..10 {
        events.push(speech(70_000 + i * 4_000, cue(0.8, 0.9, 1_500, false, 0.2)));
    }
    events.push(Event::Tick {
        now: Millis(3_600_000),
    });
    events.push(speech(3_600_500, cue(0.9, 0.9, 400, true, 0.9)));

    for event in &events {
        organism.step(event);
    }
    organism
}

#[test]
fn a_reducer_v5_snapshot_still_restores_to_the_same_state() {
    let organism = run_tape(Organism::new(Profile::t1_ref()).unwrap());
    let stored: serde_json::Value = serde_json::from_str(REDUCER_V5).unwrap();

    let restored: Organism = serde_json::from_value(stored.clone()).unwrap();
    assert_eq!(
        restored, organism,
        "the stored state reads back as the live one"
    );

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
