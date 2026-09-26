//! A device-directedness detector says whether speech is addressed to Enton. Inside
//! an attention window, speech addressed to someone else is turned away; speech
//! clearly addressed to Enton may use the longer window.

use enton_core::{
    Abstention, Action, Event, Millis, Organism, Profile, Reason, SpeechCue, ThoughtId,
};

use super::support::{assert_abstention, assert_thought, lab_profile, speech};

/// A cue with every sensor reporting: `(speaker_sim, media, turn_complete, directed)`.
fn heard(
    now: u64,
    keyword: bool,
    duration_ms: u32,
    sensors: (f32, f32, f32, Option<f32>),
) -> Event {
    let (speaker_sim, media, turn_complete, directed) = sensors;
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 1.0,
            duration_ms,
            vad_confidence: 1.0,
            keyword,
            speaker_sim: Some(speaker_sim),
            media: Some(media),
            turn_complete: Some(turn_complete),
            directed,
            direction: None,
        },
    }
}

/// The owner, live, finishing a turn, with the given directedness reading.
fn owner(now: u64, directed: Option<f32>) -> Event {
    heard(now, false, 1_500, (0.9, 0.1, 0.95, directed))
}

/// A cue that only a directedness detector described.
fn only_directed(now: u64, duration_ms: u32, directed: Option<f32>) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 1.0,
            duration_ms,
            vad_confidence: 1.0,
            keyword: false,
            speaker_sim: None,
            media: None,
            turn_complete: None,
            directed,
            direction: None,
        },
    }
}

/// The owner called Enton and got a reply at 2 s: the short window runs to 7 s, the
/// long one to 12 s.
fn in_conversation(mut organism: Organism) -> Organism {
    assert_thought(
        &organism.step(&heard(1_000, true, 1_500, (0.9, 0.1, 0.95, Some(0.9)))),
        1,
        &Reason::Keyword,
    );
    organism.step(&Event::CortexReply {
        now: Millis(2_000),
        thought: ThoughtId(1),
        text: "Oi!".into(),
    });
    organism
}

fn is_follow_up(actions: &[Action]) -> bool {
    matches!(
        actions,
        [Action::Think {
            reason: Reason::FollowUp,
            ..
        }]
    )
}

#[test]
fn speech_addressed_to_someone_else_neither_continues_nor_extends_the_window() {
    let mut organism = in_conversation(Organism::new(lab_profile()).unwrap());
    let window = organism.attention_until();
    // The owner, live and verified, turns to someone in the room.
    assert_abstention(
        &organism.step(&owner(3_000, Some(0.1))),
        Abstention::Undirected,
    );
    assert_eq!(organism.attention_until(), window);
    // Then turns back to Enton.
    assert_thought(
        &organism.step(&owner(3_500, Some(0.9))),
        2,
        &Reason::FollowUp,
    );
}

#[test]
fn an_ambiguous_reading_is_not_enough_to_turn_a_follow_up_away() {
    let mut organism = in_conversation(Organism::new(lab_profile()).unwrap());
    assert_thought(
        &organism.step(&owner(3_000, Some(0.5))),
        2,
        &Reason::FollowUp,
    );
}

#[test]
fn objections_are_named_in_order_media_other_voice_then_undirected() {
    let at = |sensors| {
        let mut organism = in_conversation(Organism::new(lab_profile()).unwrap());
        organism.step(&heard(3_000, false, 1_500, sensors))
    };
    assert_abstention(&at((0.2, 0.9, 0.95, Some(0.1))), Abstention::Media);
    assert_abstention(&at((0.2, 0.1, 0.95, Some(0.1))), Abstention::OtherSpeaker);
    assert_abstention(&at((0.9, 0.1, 0.95, Some(0.1))), Abstention::Undirected);
}

#[test]
fn a_pending_name_is_not_consumed_by_an_aside() {
    let mut organism = Organism::new(lab_profile()).unwrap();
    // "Enton..." and a pause: the name alone reads as unfinished.
    assert!(matches!(
        organism
            .step(&heard(1_000, true, 400, (0.9, 0.1, 0.2, Some(0.9))))
            .as_slice(),
        [Action::Attend { .. }]
    ));
    // Two seconds later the owner says something to someone else.
    assert_abstention(
        &organism.step(&heard(4_000, false, 1_000, (0.9, 0.1, 0.95, Some(0.1)))),
        Abstention::Undirected,
    );
    assert!(
        organism.is_attending(),
        "the aside did not consume the turn"
    );
    // The request itself follows.
    assert_thought(
        &organism.step(&heard(5_000, false, 800, (0.9, 0.1, 0.95, Some(0.9)))),
        1,
        &Reason::Keyword,
    );
}

#[test]
fn closeness_to_an_unfinished_name_excuses_one_of_three_objections_never_two() {
    // Each continuation starts 300 ms after "Enton..." and finishes the turn.
    let continue_with = |speaker_sim, media, directed| {
        let mut organism = Organism::new(lab_profile()).unwrap();
        assert!(matches!(
            organism
                .step(&heard(1_000, true, 400, (0.9, 0.1, 0.2, Some(0.9))))
                .as_slice(),
            [Action::Attend { .. }]
        ));
        let actions = organism.step(&heard(
            2_200,
            false,
            900,
            (speaker_sim, media, 0.95, Some(directed)),
        ));
        (actions, organism.is_attending())
    };
    // One sensor objects, whichever it is: still the caller going on.
    for (speaker_sim, media, directed) in [(0.9, 0.1, 0.1), (0.3, 0.1, 0.9), (0.9, 0.8, 0.9)] {
        let (actions, attending) = continue_with(speaker_sim, media, directed);
        assert_thought(&actions, 1, &Reason::Keyword);
        assert!(!attending);
    }
    // Any two object: someone else, however close in time; the name keeps waiting.
    for (speaker_sim, media, directed, why) in [
        (0.3, 0.1, 0.1, Abstention::OtherSpeaker),
        (0.9, 0.8, 0.1, Abstention::Media),
        (0.3, 0.8, 0.9, Abstention::Media),
    ] {
        let (actions, attending) = continue_with(speaker_sim, media, directed);
        assert_abstention(&actions, why);
        assert!(attending);
    }
}

#[test]
fn clearly_addressed_speech_may_use_the_long_window() {
    // At 11 s the short window has closed and the long one has not. Speaker
    // verification did not run: only the directedness detector speaks for the cue.
    let late = |profile: Profile, directed| {
        let mut organism = in_conversation(Organism::new(profile).unwrap());
        organism.step(&only_directed(11_000, 1_200, directed))
    };
    assert_thought(&late(lab_profile(), Some(0.9)), 2, &Reason::FollowUp);
    // Ambiguous or missing readings do not open it.
    assert!(!is_follow_up(&late(lab_profile(), Some(0.5))));
    assert!(!is_follow_up(&late(lab_profile(), None)));
    // The profile can switch that use off.
    let mut profile = lab_profile();
    profile.attention.directed_extends_window = false;
    assert!(!is_follow_up(&late(profile, Some(0.9))));
    // A higher bar than the reading reaches keeps it shut too.
    let mut profile = lab_profile();
    profile.attention.directed_window_llr = 5.0;
    assert!(!is_follow_up(&late(profile, Some(0.9))));
}

#[test]
fn clearly_addressed_speech_in_another_voice_stays_out_of_the_long_window() {
    let mut organism = in_conversation(Organism::new(lab_profile()).unwrap());
    let other = heard(11_000, false, 1_200, (0.2, 0.1, 0.95, Some(0.9)));
    assert!(!is_follow_up(&organism.step(&other)));
    // Nor does the long window outlive its span for anyone.
    let mut organism = in_conversation(Organism::new(lab_profile()).unwrap());
    assert!(!is_follow_up(&organism.step(&only_directed(
        12_500,
        1_200,
        Some(0.9)
    ))));
}

#[test]
fn directedness_settings_do_not_touch_cues_without_the_detector() {
    // The same sensorless tape under the lab profile and under extreme directedness
    // settings: every decision and the final state agree.
    let tape = [
        speech(1_000, true),
        Event::CortexReply {
            now: Millis(2_000),
            thought: ThoughtId(1),
            text: "Oi!".into(),
        },
        // Inside the short window, a follow-up; then only the long window is open.
        only_directed(3_000, 1_200, None),
        Event::Tick { now: Millis(8_500) },
        only_directed(11_000, 1_500, None),
        only_directed(30_000, 900, None),
    ];
    let mut strict = lab_profile();
    strict.attention.undirected_llr = 0.01;
    strict.attention.directed_window_llr = 0.01;
    let mut off = lab_profile();
    off.attention.directed_extends_window = false;
    let run = |profile: Profile| {
        let mut organism = Organism::new(profile).unwrap();
        let actions: Vec<_> = tape.iter().map(|event| organism.step(event)).collect();
        (
            actions,
            organism.attention_until(),
            organism.verified_attention_until(),
        )
    };
    let reference = run(lab_profile());
    assert!(
        reference
            .0
            .iter()
            .flatten()
            .any(|action| matches!(action, Action::Think { .. })),
        "the tape exercises decisions: {reference:?}"
    );
    assert_eq!(run(strict), reference);
    assert_eq!(run(off), reference);
}
