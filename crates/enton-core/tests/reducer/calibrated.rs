//! Behavior under the calibration measured for the sensors Enton ships, where a
//! single segment says little about who is speaking.

use enton_core::{Abstention, Event, Millis, Organism, Profile, Reason, SpeechCue, ThoughtId};

use super::support::{assert_abstention, assert_thought};

fn heard(now: u64, keyword: bool, duration_ms: u32, sim: f32, media: f32) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 0.9,
            duration_ms,
            vad_confidence: 0.9,
            keyword,
            speaker_sim: Some(sim),
            media: Some(media),
            turn_complete: None,
        },
    }
}

/// The owner called Enton and got a text reply at 2 s: the windows run from there.
fn in_conversation(mut organism: Organism) -> Organism {
    assert_thought(
        &organism.step(&heard(1_000, true, 1_500, 0.55, 0.3)),
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
fn the_owner_across_the_room_is_still_heard() {
    // 0.45 is how the owner often scores across the room with a TV on. A fixed
    // 0.6 threshold called that another voice; calibrated, it is barely evidence.
    let mut organism = in_conversation(Organism::new(Profile::t1_ref()).unwrap());
    assert_thought(
        &organism.step(&heard(3_500, false, 1_500, 0.45, 0.35)),
        2,
        &Reason::FollowUp,
    );
}

#[test]
fn one_segment_cannot_tell_a_relative_from_the_owner() {
    // A relative in the same room scores about 0.44 on a 1.5 s line: under a fifth
    // of a nat against the owner. Neither verified nor ruled out, it is treated as
    // the conversation going on.
    let mut organism = in_conversation(Organism::new(Profile::t1_ref()).unwrap());
    assert_thought(
        &organism.step(&heard(3_500, false, 1_500, 0.44, 0.3)),
        2,
        &Reason::FollowUp,
    );
}

#[test]
fn a_clear_tv_line_inside_the_window_is_not_a_follow_up() {
    // A TV voice (0.25) that the tagger also hears as a loudspeaker (0.8): over two
    // nats for reproduced sound.
    let mut organism = in_conversation(Organism::new(Profile::t1_ref()).unwrap());
    assert_abstention(
        &organism.step(&heard(3_500, false, 1_500, 0.25, 0.8)),
        Abstention::Media,
    );
}

#[test]
fn only_positive_voice_evidence_keeps_the_long_window() {
    // Eight seconds after the reply the short window is over. A 3 s line at 0.75
    // is most of a nat for the owner: the conversation goes on.
    let mut organism = in_conversation(Organism::new(Profile::t1_ref()).unwrap());
    assert_thought(
        &organism.step(&heard(10_000, false, 3_000, 0.75, 0.2)),
        2,
        &Reason::FollowUp,
    );

    // The same line at 0.5 says nothing either way, which is not enough so late:
    // it is overheard speech, not a follow-up.
    let mut organism = in_conversation(Organism::new(Profile::t1_ref()).unwrap());
    let actions = organism.step(&heard(10_000, false, 3_000, 0.5, 0.2));
    assert!(
        !matches!(
            actions.as_slice(),
            [enton_core::Action::Think {
                reason: Reason::FollowUp,
                ..
            }]
        ),
        "{actions:?}"
    );
}

fn sensed(now: u64, keyword: bool, duration_ms: u32, vad: f32, sensors: (f32, f32, f32)) -> Event {
    let (sim, media, turn) = sensors;
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 0.9,
            duration_ms,
            vad_confidence: vad,
            keyword,
            speaker_sim: Some(sim),
            media: Some(media),
            turn_complete: Some(turn),
        },
    }
}

#[test]
fn right_after_an_unfinished_name_one_tagger_error_is_excused() {
    // "Enton..." (400 ms, reads unfinished), then the rest 300 ms later in a voice
    // that favors the owner, which the tagger alone mistakes for a loudspeaker.
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    assert!(matches!(
        organism
            .step(&sensed(1_000, true, 400, 0.9, (0.6, 0.2, 0.2)))
            .as_slice(),
        [enton_core::Action::Attend { .. }]
    ));
    assert_thought(
        &organism.step(&sensed(2_800, false, 1_500, 0.9, (0.6, 0.9, 0.95))),
        1,
        &Reason::Keyword,
    );
}

#[test]
fn only_speech_tells_enton_a_tv_is_on() {
    // Clicks and hums that a tagger scores as reproduced carry no voice to judge.
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    for n in 0..6 {
        organism.step(&sensed(
            1_000 + n * 2_000,
            false,
            1_500,
            0.1,
            (0.1, 0.95, 0.5),
        ));
    }
    assert_eq!(organism.tv_presence().to_bits(), 0);

    // The same lines as speech are the TV talking.
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    for n in 0..6 {
        organism.step(&sensed(
            1_000 + n * 2_000,
            false,
            1_500,
            0.9,
            (0.1, 0.95, 0.5),
        ));
    }
    assert!(organism.tv_presence() > 0.5, "{}", organism.tv_presence());
}
