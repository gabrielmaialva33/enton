//! An end-of-turn signal, when present, decides whether a keyword cue is a whole
//! request or the start of one.

use enton_core::{Action, Event, Millis, Organism, Reason, SpeechCue};

use super::support::{assert_thought, lab_profile};

fn called(duration_ms: u32, turn_complete: Option<f32>) -> Event {
    Event::Speech {
        now: Millis(1_000),
        cue: SpeechCue {
            energy: 1.0,
            duration_ms,
            vad_confidence: 1.0,
            keyword: true,
            speaker_sim: Some(0.9),
            media: Some(0.1),
            turn_complete,
            directed: None,
            direction: None,
        },
    }
}

fn attends(actions: &[Action]) -> bool {
    matches!(actions, [Action::Attend { .. }])
}

#[test]
fn a_short_complete_command_is_answered_at_once() {
    // "Enton, para!" is under 900 ms, but it is a whole request.
    let mut organism = Organism::new(lab_profile()).unwrap();
    assert_thought(&organism.step(&called(500, Some(0.9))), 1, &Reason::Keyword);
}

#[test]
fn a_long_unfinished_call_waits_for_the_rest() {
    // "Enton, você pode..." followed by a pause is not a request yet.
    let mut organism = Organism::new(lab_profile()).unwrap();
    assert!(attends(&organism.step(&called(1_400, Some(0.2)))));
    assert!(organism.is_attending());
}

#[test]
fn without_an_end_of_turn_model_duration_decides_as_before() {
    let mut organism = Organism::new(lab_profile()).unwrap();
    assert!(attends(&organism.step(&called(500, None))));
    let mut organism = Organism::new(lab_profile()).unwrap();
    assert_thought(&organism.step(&called(1_400, None)), 1, &Reason::Keyword);
}

#[test]
fn another_voice_saying_the_name_does_not_get_the_shortcut() {
    // Someone else says "Enton" in passing: even if the end-of-turn model calls it
    // complete, the voice is known not to be the caller's, so the name waits.
    let mut organism = Organism::new(lab_profile()).unwrap();
    let passing = Event::Speech {
        now: Millis(1_000),
        cue: SpeechCue {
            energy: 1.0,
            duration_ms: 250,
            vad_confidence: 1.0,
            keyword: true,
            speaker_sim: Some(0.2),
            media: Some(0.1),
            turn_complete: Some(0.9),
            directed: None,
            direction: None,
        },
    };
    assert!(attends(&organism.step(&passing)));
}

fn cue_at(now: u64, keyword: bool, duration_ms: u32, sim: f32, media: f32, complete: f32) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 1.0,
            duration_ms,
            vad_confidence: 1.0,
            keyword,
            speaker_sim: Some(sim),
            media: Some(media),
            turn_complete: Some(complete),
            directed: None,
            direction: None,
        },
    }
}

/// The caller says "Enton..." and stops short (an unfinished name ending at 1 s).
fn unfinished_name(mut organism: Organism) -> Organism {
    assert!(attends(
        &organism.step(&cue_at(1_000, true, 400, 0.9, 0.1, 0.2))
    ));
    organism
}

#[test]
fn closeness_to_an_unfinished_name_excuses_one_sensor_error_not_two() {
    // Each cue starts 300 ms after the name and finishes the turn.
    let continuation = |sim, media| cue_at(2_200, false, 900, sim, media, 0.9);
    let fresh = || unfinished_name(Organism::new(lab_profile()).unwrap());

    // The voice check misfired once: still the caller going on.
    assert_thought(&fresh().step(&continuation(0.3, 0.1)), 1, &Reason::Keyword);
    // The media tagger misfired once: still the caller going on.
    assert_thought(&fresh().step(&continuation(0.9, 0.8)), 1, &Reason::Keyword);
    // Both reject it: that is a TV or someone else, however close in time.
    let mut organism = fresh();
    assert!(matches!(
        organism.step(&continuation(0.3, 0.8)).as_slice(),
        [Action::Abstain { .. }]
    ));
    assert!(organism.is_attending());
}

#[test]
fn an_unfinished_continuation_keeps_the_request_open() {
    let mut organism = unfinished_name(Organism::new(lab_profile()).unwrap());
    // "... você pode..." is still unfinished: wait again, from its end.
    assert!(attends(
        &organism.step(&cue_at(2_200, false, 900, 0.9, 0.1, 0.2))
    ));
    assert!(organism.is_attending());
    // "... me lembrar da reunião?" finishes it: one request, one thought.
    assert_thought(
        &organism.step(&cue_at(3_600, false, 1_000, 0.9, 0.1, 0.9)),
        1,
        &Reason::Keyword,
    );
}

#[test]
fn a_distant_cue_gets_no_such_benefit() {
    // Starts 2 s after the name: the per-segment vetoes apply again.
    let mut organism = unfinished_name(Organism::new(lab_profile()).unwrap());
    assert!(matches!(
        organism
            .step(&cue_at(3_900, false, 900, 0.3, 0.8, 0.9))
            .as_slice(),
        [Action::Abstain { .. }]
    ));
    assert!(organism.is_attending());
}

#[test]
fn another_voice_saying_the_name_does_not_take_over_a_pending_turn() {
    let mut organism = unfinished_name(Organism::new(lab_profile()).unwrap());
    // Someone else says "Enton" in passing, 400 ms later.
    assert!(matches!(
        organism
            .step(&cue_at(1_650, true, 250, 0.2, 0.1, 0.9))
            .as_slice(),
        [Action::Abstain { .. }]
    ));
    assert!(organism.is_attending());
    // The caller's continuation, adjacent to the caller's own name, still joins it.
    assert_thought(
        &organism.step(&cue_at(2_200, false, 900, 0.9, 0.1, 0.9)),
        1,
        &Reason::Keyword,
    );
}
