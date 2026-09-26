use enton_core::{
    Abstention, Action, Event, Millis, Organism, Reason, SpeechCue, ThoughtId, UtteranceId,
};

use super::support::{assert_abstention, assert_thought, lab_profile, voice};

#[test]
fn another_voice_inside_the_window_is_not_a_follow_up() {
    let mut organism = Organism::new(lab_profile()).unwrap();
    assert_thought(
        &organism.step(&voice(1_000, true, 1_500, 0.9)),
        1,
        &Reason::Keyword,
    );
    organism.step(&Event::CortexReply {
        now: Millis(2_000),
        thought: ThoughtId(1),
        text: "Oi!".into(),
    });
    let window = organism.attention_until();

    // The TV talks inside the window: no follow-up, and the window does not grow.
    assert_abstention(
        &organism.step(&voice(3_000, false, 1_500, 0.2)),
        Abstention::OtherSpeaker,
    );
    assert_eq!(organism.attention_until(), window);

    // The person who asked keeps the conversation going.
    assert_thought(
        &organism.step(&voice(3_500, false, 1_500, 0.9)),
        2,
        &Reason::FollowUp,
    );
}

#[test]
fn a_pending_name_call_waits_for_its_own_speaker() {
    let mut organism = Organism::new(lab_profile()).unwrap();
    assert!(matches!(
        organism.step(&voice(1_000, true, 400, 0.9)).as_slice(),
        [Action::Attend { .. }]
    ));

    assert_abstention(
        &organism.step(&voice(2_000, false, 1_200, 0.2)),
        Abstention::OtherSpeaker,
    );
    assert!(
        organism.is_attending(),
        "the TV did not consume the pending turn"
    );

    assert_thought(
        &organism.step(&voice(3_000, false, 1_200, 0.9)),
        1,
        &Reason::Keyword,
    );
}

#[test]
fn only_the_addressed_voice_barges_in_without_saying_the_name() {
    let mut organism = Organism::new(lab_profile()).unwrap();
    assert_thought(
        &organism.step(&voice(1_000, true, 1_500, 0.9)),
        1,
        &Reason::Keyword,
    );
    organism.step(&Event::PlaybackStarted {
        now: Millis(1_500),
        utterance: UtteranceId(1),
    });

    // A loud other voice over Enton's playback is not an interruption, and it is
    // named for what it is: loud enough, wrong voice. It must not teach the echo model.
    let expectation = organism.echo_energy_expectation();
    assert_abstention(
        &organism.step(&voice(2_000, false, 1_000, 0.2)),
        Abstention::OtherSpeaker,
    );
    assert!(organism.is_speaking());
    assert!((organism.echo_energy_expectation() - expectation).abs() < f32::EPSILON);

    // The same loudness in the addressed voice is a barge-in.
    assert_thought(
        &organism.step(&voice(2_500, false, 1_000, 0.9)),
        2,
        &Reason::FollowUp,
    );
}

#[test]
fn only_the_verified_voice_keeps_a_conversation_through_a_long_pause() {
    let converse = || {
        let mut organism = Organism::new(lab_profile()).unwrap();
        assert_thought(
            &organism.step(&voice(1_000, true, 1_500, 0.9)),
            1,
            &Reason::Keyword,
        );
        organism.step(&Event::CortexReply {
            now: Millis(2_000),
            thought: ThoughtId(1),
            text: "Oi!".into(),
        });
        organism
    };
    // Nine seconds later the short window has closed, the verified one has not.
    let late = 11_000;

    let mut organism = converse();
    assert_thought(
        &organism.step(&voice(late, false, 1_200, 0.9)),
        2,
        &Reason::FollowUp,
    );

    // Without speaker verification the pause ends the conversation, as before.
    let mut organism = converse();
    let unverified = Event::Speech {
        now: Millis(late),
        cue: SpeechCue {
            energy: 1.0,
            duration_ms: 1_200,
            vad_confidence: 1.0,
            keyword: false,
            speaker_sim: None,
            media: None,
            turn_complete: None,
        },
    };
    assert!(!matches!(
        organism.step(&unverified).as_slice(),
        [Action::Think {
            reason: Reason::FollowUp,
            ..
        }]
    ));

    // Nor can anyone else step into the long window.
    let mut organism = converse();
    assert!(!matches!(
        organism.step(&voice(late, false, 1_200, 0.2)).as_slice(),
        [Action::Think {
            reason: Reason::FollowUp,
            ..
        }]
    ));
}
