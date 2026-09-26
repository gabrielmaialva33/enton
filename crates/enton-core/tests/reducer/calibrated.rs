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
            directed: None,
            direction: None,
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
            directed: None,
            direction: None,
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

/// A line in the owner's voice as the shipped sensors hear it, with a directedness reading.
fn addressed(now: u64, duration_ms: u32, sim: f32, media: f32, directed: f32) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 0.9,
            duration_ms,
            vad_confidence: 0.9,
            keyword: false,
            speaker_sim: Some(sim),
            media: Some(media),
            turn_complete: None,
            directed: Some(directed),
            direction: None,
        },
    }
}

#[test]
fn an_aside_the_detector_hears_clearly_is_turned_away() {
    // The owner's own voice, which one segment cannot tell from a follow-up: only
    // the detector's reading (clearly addressed to someone else) turns it away.
    let mut organism = in_conversation(Organism::new(Profile::t1_ref()).unwrap());
    assert_abstention(
        &organism.step(&addressed(3_500, 1_500, 0.55, 0.3, 0.1)),
        Abstention::Undirected,
    );
    // An ambiguous reading leans away by little more than half a nat: still a follow-up.
    let mut organism = in_conversation(Organism::new(Profile::t1_ref()).unwrap());
    assert_thought(
        &organism.step(&addressed(3_500, 1_500, 0.55, 0.3, 0.5)),
        2,
        &Reason::FollowUp,
    );
}

#[test]
fn a_command_shaped_aside_is_the_detector_s_blind_spot() {
    // "Anota aí também o sabão em pó", said to someone else, scores as addressed to
    // Enton: the detector's confident errors buy a thought like a follow-up would.
    let mut organism = in_conversation(Organism::new(Profile::t1_ref()).unwrap());
    assert_thought(
        &organism.step(&addressed(3_500, 1_500, 0.55, 0.3, 0.95)),
        2,
        &Reason::FollowUp,
    );
}

#[test]
fn clearly_addressed_speech_keeps_the_long_window_when_the_voice_says_nothing() {
    // The 3 s line at 0.5 that says nothing about the voice is overheard speech this
    // late (see `only_positive_voice_evidence_keeps_the_long_window`); clearly
    // addressed to Enton, it continues the conversation.
    let mut organism = in_conversation(Organism::new(Profile::t1_ref()).unwrap());
    assert_thought(
        &organism.step(&addressed(10_000, 3_000, 0.5, 0.2, 0.9)),
        2,
        &Reason::FollowUp,
    );
    // A voice that rules the owner out gets no such benefit.
    let mut organism = in_conversation(Organism::new(Profile::t1_ref()).unwrap());
    let actions = organism.step(&addressed(10_000, 3_000, 0.2, 0.2, 0.9));
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

/// The same event with a microphone array's reading, if any.
fn pointed(event: Event, direction: Option<[f32; 2]>) -> Event {
    match event {
        Event::Speech { now, cue } => Event::Speech {
            now,
            cue: SpeechCue { direction, ..cue },
        },
        other => other,
    }
}

/// Where the TV stands, as the array sees it, and a place across the room from it.
const TV: [f32; 2] = [0.6, 0.8];
const ACROSS: [f32; 2] = [0.8, -0.6];

/// Twenty-four TV lines, four seconds apart from 1 s, from the TV's direction: voice and
/// tagger mark each as the TV (about 2.7 nats), so they teach the shipped profile, which
/// wants twenty recent lines, where the TV stands.
fn after_a_show(mut organism: Organism) -> Organism {
    for n in 0..24 {
        if n == 18 {
            assert_eq!(organism.tv_direction_as_of(Millis(72_000)), None);
        }
        organism.step(&pointed(
            heard(1_000 + n * 4_000, false, 1_500, 0.3, 0.7),
            Some(TV),
        ));
    }
    organism
}

#[test]
fn the_shipped_array_learns_where_the_tv_is_and_weighs_readings_against_it() {
    let organism = after_a_show(Organism::new(Profile::t1_ref()).unwrap());
    let learned = organism.tv_direction_as_of(Millis(93_000)).unwrap();
    assert!((learned[0] - TV[0]).abs() < 1e-5 && (learned[1] - TV[1]).abs() < 1e-5);
    let weigh = |direction| {
        let Event::Speech { cue, .. } = pointed(heard(0, false, 1_500, 0.5, 0.4), Some(direction))
        else {
            unreachable!()
        };
        organism
            .evidence_as_of(Millis(94_000), &cue)
            .from_tv_direction
    };
    let model = organism.profile().senses.direction;
    assert_eq!(weigh(TV).to_bits(), model.max_llr.to_bits());
    assert_eq!(weigh(ACROSS).to_bits(), model.min_llr.to_bits());
}

/// The owner across the room with the TV on, as the shipped sensors hear them on average
/// (0.47 and 0.38 on a 1.5 s line): after calling Enton at 95 s and a reply at 96 s.
fn owner_follow_up(organism: Organism, direction: Option<[f32; 2]>) -> Vec<enton_core::Action> {
    let mut organism = after_a_show(organism);
    assert_thought(
        &organism.step(&heard(95_000, true, 1_500, 0.55, 0.3)),
        1,
        &Reason::Keyword,
    );
    organism.step(&Event::CortexReply {
        now: Millis(96_000),
        thought: ThoughtId(1),
        text: "Oi!".into(),
    });
    organism.step(&pointed(heard(97_000, false, 1_500, 0.47, 0.38), direction))
}

#[test]
fn the_shipped_profile_lets_the_direction_add_evidence_but_never_loosen_a_bar() {
    // With the TV on, the tagger's reading alone turns the owner away, wherever they sit.
    for direction in [None, Some(ACROSS), Some(TV)] {
        assert_abstention(
            &owner_follow_up(Organism::new(Profile::t1_ref()).unwrap(), direction),
            Abstention::Media,
        );
    }
}

#[test]
fn confined_the_owner_away_from_the_tv_is_heard_with_the_tv_on() {
    let mut confined = Profile::t1_ref();
    confined.source.direction_confines_tv_caution = true;
    assert_thought(
        &owner_follow_up(Organism::new(confined.clone()).unwrap(), Some(ACROSS)),
        2,
        &Reason::FollowUp,
    );
    // In line with the TV, the direction is over two nats for a loudspeaker.
    assert_abstention(
        &owner_follow_up(Organism::new(confined.clone()).unwrap(), Some(TV)),
        Abstention::OtherSpeaker,
    );
    // Without the array, the whole caution applies, as it always did.
    assert_abstention(
        &owner_follow_up(Organism::new(confined).unwrap(), None),
        Abstention::Media,
    );
}
