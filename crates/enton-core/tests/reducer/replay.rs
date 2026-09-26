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
