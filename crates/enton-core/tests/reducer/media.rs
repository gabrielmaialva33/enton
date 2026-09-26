//! Reproduced media (TV, radio, music) is never a live voice to Enton.

use enton_core::{
    Abstention, Action, Event, Millis, Organism, Profile, Reason, SpeechCue, ThoughtId, UtteranceId,
};

use super::support::{assert_abstention, assert_thought};

fn heard(now: u64, keyword: bool, speaker_sim: f32, media: f32) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 1.0,
            duration_ms: 1_500,
            vad_confidence: 1.0,
            keyword,
            speaker_sim: Some(speaker_sim),
            media: Some(media),
            turn_complete: None,
            directed: None,
        },
    }
}

/// Enton called by name, and its reply delivered: an attention window is open.
fn addressed(mut organism: Organism) -> Organism {
    assert_thought(
        &organism.step(&heard(1_000, true, 0.9, 0.1)),
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

#[test]
fn media_inside_the_window_neither_continues_nor_extends_it() {
    let mut organism = addressed(Organism::new(Profile::t1_ref()).unwrap());
    let window = organism.attention_until();
    // Even a TV voice the verifier confuses with the caller is not a follow-up.
    assert_abstention(
        &organism.step(&heard(3_000, false, 0.9, 0.9)),
        Abstention::Media,
    );
    assert_eq!(organism.attention_until(), window);
    assert_thought(
        &organism.step(&heard(3_500, false, 0.9, 0.1)),
        2,
        &Reason::FollowUp,
    );
}

#[test]
fn overheard_media_never_buys_a_discretionary_thought() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    assert_abstention(
        &organism.step(&heard(1_000, false, 0.2, 0.9)),
        Abstention::Media,
    );
    // The same live voice would have ignited.
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    assert_thought(
        &organism.step(&heard(1_000, false, 0.2, 0.1)),
        1,
        &Reason::Speech,
    );
}

#[test]
fn media_over_playback_does_not_teach_the_echo_model() {
    let mut organism = addressed(Organism::new(Profile::t1_ref()).unwrap());
    organism.step(&Event::PlaybackStarted {
        now: Millis(2_100),
        utterance: UtteranceId(1),
    });
    let expectation = organism.echo_energy_expectation();
    assert_abstention(
        &organism.step(&heard(2_500, false, 0.2, 0.9)),
        Abstention::Media,
    );
    assert!((organism.echo_energy_expectation() - expectation).abs() < f32::EPSILON);
    assert!(organism.is_speaking());
}

#[test]
fn the_name_is_heard_even_from_something_that_sounds_like_media() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    assert!(matches!(
        organism.step(&heard(1_000, true, 0.9, 0.9)).as_slice(),
        [Action::Think {
            reason: Reason::Keyword,
            ..
        }]
    ));
}
