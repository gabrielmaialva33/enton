use enton_core::{Action, Event, Millis, Organism, Profile, ThoughtId};

use super::support::{body, speech};

#[test]
fn identical_tapes_reproduce_actions_and_final_state() {
    let tape = [
        Event::Tick {
            now: Millis(60_000),
        },
        speech(60_001, false),
        speech(60_002, false),
        speech(60_003, true),
        Event::CortexReply {
            now: Millis(60_004),
            thought: ThoughtId(2),
            text: "I am listening.".to_owned(),
        },
        body(60_005, Some(100.0), None),
        speech(60_006, false),
        Event::Tick {
            now: Millis(120_000),
        },
        body(120_001, None, Some(0.9)),
        speech(120_002, false),
    ];
    let mut first = Organism::new(Profile::t1_ref()).unwrap();
    let mut second = Organism::new(Profile::t1_ref()).unwrap();
    let first_actions: Vec<_> = tape.iter().map(|event| first.step(event)).collect();
    let second_actions: Vec<_> = tape.iter().map(|event| second.step(event)).collect();
    assert_eq!(first_actions, second_actions);
    assert_eq!(first, second);
    assert_eq!(
        first_actions[4],
        vec![Action::Speak {
            text: "I am listening.".to_owned()
        }]
    );
}

#[test]
fn a_cue_with_broken_measurements_replays_exactly_like_it_ran_live() {
    use enton_core::SpeechCue;
    let broken = Event::Speech {
        now: Millis(1_000),
        cue: SpeechCue {
            energy: f32::NAN,
            duration_ms: 1_500,
            vad_confidence: f32::INFINITY,
            keyword: true,
            speaker_sim: Some(f32::NAN),
            media: Some(f32::NEG_INFINITY),
            turn_complete: Some(f32::NAN),
            directed: Some(f32::INFINITY),
            direction: None,
        },
    };
    let mut live = Organism::new(Profile::t1_ref()).unwrap();
    let live_actions = live.step(&broken);

    // What a durable log stores, and what replay reads back.
    let stored = serde_json::to_string(&broken.clone().canonical()).unwrap();
    let replayed_event: Event = serde_json::from_str(&stored).unwrap();
    let mut replayed = Organism::new(Profile::t1_ref()).unwrap();
    assert_eq!(replayed.step(&replayed_event), live_actions);
    assert_eq!(replayed, live);
}
