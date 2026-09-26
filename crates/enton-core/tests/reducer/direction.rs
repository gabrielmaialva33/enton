//! A microphone array's direction of arrival. Enton learns where the TV is from the lines
//! that voice and tagger already mark as the TV, then weighs each cue's direction against
//! it while the TV is on: evidence for a loudspeaker, never proof, and one independent
//! objection among the others.

use enton_core::{
    Abstention, Action, DirectionModel, Event, Millis, Organism, Profile, Reason, SpeechCue,
    ThoughtId, TvCautionConfinement, UtteranceId,
};

use super::support::{assert_abstention, assert_thought, lab_profile};

/// Where the TV stands, as the array sees it.
const TV: [f32; 2] = [1.0, 0.0];
/// Straight across the room from the TV, as far as a reading can be from it.
const AWAY: [f32; 2] = [0.0, 1.0];

/// The lab calibration with a direction model strong enough to sway a borderline voice
/// on its own, trusting a TV direction after three lines.
fn lab() -> Profile {
    let mut profile = lab_profile();
    profile.senses.direction = DirectionModel {
        kappa: 15.0,
        max_llr: 5.0,
        min_llr: -5.0,
    };
    profile.source.tv_direction_min_lines = 3.0;
    profile
}

/// A speech cue with every sensor reporting: `(speaker_sim, media, turn_complete)`.
fn heard(
    now: u64,
    keyword: bool,
    duration_ms: u32,
    sensors: (f32, f32, f32),
    direction: Option<[f32; 2]>,
) -> Event {
    let (speaker_sim, media, turn_complete) = sensors;
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 0.8,
            duration_ms,
            vad_confidence: 0.9,
            keyword,
            speaker_sim: Some(speaker_sim),
            media: Some(media),
            turn_complete: Some(turn_complete),
            directed: None,
            direction,
        },
    }
}

/// A line the lab's voice and tagger mark as the TV beyond doubt.
fn tv_line(now: u64, direction: [f32; 2]) -> Event {
    heard(now, false, 1_500, (0.2, 0.9, 0.5), Some(direction))
}

/// The unit vector `degrees` from the TV.
fn off_tv(degrees: f32) -> [f32; 2] {
    let (sin, cos) = degrees.to_radians().sin_cos();
    [cos, sin]
}

/// Degrees between two unit vectors.
fn degrees_between(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] * b[0] + a[1] * b[1])
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

/// Four TV lines from the TV's direction, two seconds apart from 1 s: the TV is on and
/// Enton knows where it stands.
fn with_the_tv_on(mut organism: Organism) -> Organism {
    for n in 0..4 {
        organism.step(&tv_line(1_000 + n * 2_000, TV));
    }
    assert!(organism.tv_presence_as_of(Millis(8_000)) >= 0.5);
    assert!(organism.tv_direction_as_of(Millis(8_000)).is_some());
    organism
}

/// The owner called Enton at 10 s and got a text reply at 11 s: the short window runs
/// to 16 s.
fn in_conversation(mut organism: Organism) -> Organism {
    assert_thought(
        &organism.step(&heard(10_000, true, 1_500, (0.85, 0.1, 0.95), None)),
        1,
        &Reason::Keyword,
    );
    organism.step(&Event::CortexReply {
        now: Millis(11_000),
        thought: ThoughtId(1),
        text: "Oi!".into(),
    });
    organism
}

#[test]
fn tv_lines_teach_the_tv_direction_once_enough_of_them_agree() {
    let mut organism = Organism::new(lab()).unwrap();
    // Three lines, each a little older than the last, weigh just under three.
    let readings = [off_tv(-8.0), off_tv(10.0), off_tv(2.0), off_tv(-4.0)];
    for (n, reading) in (0_u64..).zip(readings) {
        assert_eq!(organism.tv_direction_as_of(Millis(n * 2_000)), None);
        organism.step(&tv_line(1_000 + n * 2_000, reading));
    }
    let learned = organism.tv_direction_as_of(Millis(8_000)).unwrap();
    let length = (learned[0] * learned[0] + learned[1] * learned[1]).sqrt();
    assert!((length - 1.0).abs() < 1e-6, "{learned:?}");
    assert!(degrees_between(learned, TV) < 2.0, "{learned:?}");
}

#[test]
fn only_speech_that_voice_and_tagger_mark_as_the_tv_teaches_it() {
    let mut eager = lab();
    eager.source.tv_direction_min_lines = 1.0;
    let taught = |events: &[Event]| {
        let mut organism = Organism::new(eager.clone()).unwrap();
        for event in events {
            organism.step(event);
        }
        organism.tv_direction_as_of(events.last().unwrap().now())
    };
    let only_direction = |now| Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 0.8,
            duration_ms: 1_500,
            vad_confidence: 0.9,
            direction: Some(TV),
            ..SpeechCue::default()
        },
    };
    let quiet_tv = |now| Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            vad_confidence: 0.1,
            ..match tv_line(now, TV) {
                Event::Speech { cue, .. } => cue,
                _ => unreachable!(),
            }
        },
    };
    // The owner's voice, a cue no voice or tagger described, the name, and a hum.
    assert_eq!(
        taught(&[heard(1_000, false, 1_500, (0.85, 0.1, 0.95), Some(TV))]),
        None
    );
    assert_eq!(taught(&[only_direction(1_000)]), None);
    assert_eq!(
        taught(&[heard(1_000, true, 1_500, (0.2, 0.9, 0.95), Some(TV))]),
        None
    );
    assert_eq!(taught(&[quiet_tv(1_000)]), None);
    // A TV line without a reading teaches nothing either; with one, it does.
    assert_eq!(
        taught(&[heard(1_000, false, 1_500, (0.2, 0.9, 0.5), None)]),
        None
    );
    assert!(taught(&[tv_line(1_000, TV)]).is_some());
    // Nor does a TV line heard over Enton's own playback.
    let over_playback = [
        Event::PlaybackStarted {
            now: Millis(500),
            utterance: UtteranceId(1),
        },
        tv_line(1_000, TV),
    ];
    assert_eq!(taught(&over_playback), None);
}

#[test]
fn the_tv_direction_is_forgotten_and_follows_a_tv_that_moved() {
    let mut organism = Organism::new(lab()).unwrap();
    for n in 0..5 {
        organism.step(&tv_line(1_000 + n * 2_000, TV));
    }
    let half_life = organism.profile().source.tv_direction_half_life_ms;
    assert!(
        organism
            .tv_direction_as_of(Millis(9_000 + half_life / 4))
            .is_some()
    );
    // Five lines, one half-life later, weigh less than the three it takes.
    assert_eq!(organism.tv_direction_as_of(Millis(9_000 + half_life)), None);

    // An hour later the TV talks from across the room: the old lines hardly count.
    let later = 9_000 + 6 * half_life;
    for n in 0..5 {
        organism.step(&tv_line(later + n * 2_000, AWAY));
    }
    let learned = organism.tv_direction_as_of(Millis(later + 8_000)).unwrap();
    assert!(degrees_between(learned, AWAY) < 2.0, "{learned:?}");
}

#[test]
fn lines_that_disagree_name_no_direction() {
    let mut organism = Organism::new(lab()).unwrap();
    for n in 0..20 {
        let reading = if n % 2 == 0 { TV } else { [-1.0, 0.0] };
        organism.step(&tv_line(1_000 + n * 2_000, reading));
    }
    assert_eq!(organism.tv_direction_as_of(Millis(40_000)), None);
}

/// A follow-up whose voice and tagger clear the TV caution with little to spare:
/// `(speaker_sim, media)` give 1.8 nats over another voice and 5.1 over a loudspeaker.
const BORDERLINE: (f32, f32, f32) = (0.58, 0.481_25, 0.95);

#[test]
fn a_follow_up_from_the_tv_direction_is_heard_as_the_tv() {
    let follow_up = |direction| {
        let mut organism = in_conversation(with_the_tv_on(Organism::new(lab()).unwrap()));
        organism.step(&heard(12_000, false, 1_500, BORDERLINE, direction))
    };
    assert_thought(&follow_up(Some(AWAY)), 2, &Reason::FollowUp);
    assert_thought(&follow_up(None), 2, &Reason::FollowUp);
    assert_abstention(&follow_up(Some(TV)), Abstention::OtherSpeaker);
}

#[test]
fn the_direction_is_weighed_only_while_the_tv_is_on_and_enton_is_silent() {
    let mut organism = with_the_tv_on(Organism::new(lab()).unwrap());
    let Event::Speech { cue, .. } = heard(0, false, 1_500, BORDERLINE, Some(TV)) else {
        unreachable!()
    };
    assert!(
        organism
            .evidence_as_of(Millis(8_000), &cue)
            .from_tv_direction
            > 4.0
    );
    // Three minutes later the TV belief has decayed: it is off, though its place is known.
    let quiet = Millis(187_000);
    assert!(organism.tv_presence_as_of(quiet) < 0.5);
    assert!(organism.tv_direction_as_of(quiet).is_some());
    assert_eq!(
        organism
            .evidence_as_of(quiet, &cue)
            .from_tv_direction
            .to_bits(),
        0
    );
    // While Enton itself talks, its loudspeaker dominates the array.
    organism.step(&Event::PlaybackStarted {
        now: Millis(8_500),
        utterance: UtteranceId(1),
    });
    assert!(organism.tv_presence_as_of(Millis(9_000)) >= 0.5);
    assert_eq!(
        organism
            .evidence_as_of(Millis(9_000), &cue)
            .from_tv_direction
            .to_bits(),
        0
    );
}

#[test]
fn with_the_tv_off_the_owner_in_line_with_it_is_heard_as_before() {
    // The TV's place is known, but it went quiet three minutes ago.
    let late = |direction| {
        let mut organism = with_the_tv_on(Organism::new(lab()).unwrap());
        assert_thought(
            &organism.step(&heard(180_000, true, 1_500, (0.85, 0.1, 0.95), None)),
            1,
            &Reason::Keyword,
        );
        organism.step(&Event::CortexReply {
            now: Millis(181_000),
            thought: ThoughtId(1),
            text: "Oi!".into(),
        });
        assert!(organism.tv_direction_as_of(Millis(182_000)).is_some());
        // With the TV weighed this reading would fall just past the other-voice bar.
        organism.step(&heard(
            182_000,
            false,
            1_500,
            (0.58, 0.495, 0.95),
            direction,
        ))
    };
    assert_thought(&late(Some(TV)), 2, &Reason::FollowUp);
    assert_eq!(late(Some(TV)), late(None));
}

/// "Enton..." at 8 s, read as unfinished, with the TV on and its place known.
fn after_an_unfinished_name(organism: Organism) -> Organism {
    let mut organism = with_the_tv_on(organism);
    assert!(matches!(
        organism
            .step(&heard(8_000, true, 400, (0.85, 0.1, 0.2), None))
            .as_slice(),
        [Action::Attend { .. }]
    ));
    organism
}

#[test]
fn the_direction_is_one_independent_objection_to_a_continuation() {
    // Each continuation starts 300 ms after the name and finishes the turn.
    let continue_with = |media: f32, direction| {
        let mut organism = after_an_unfinished_name(Organism::new(lab()).unwrap());
        let actions = organism.step(&heard(9_200, false, 900, (0.6, media, 0.95), direction));
        (actions, organism.is_attending())
    };
    // The tagger alone objects (with the TV on, a reading of 0.5 is enough): excused.
    let (actions, attending) = continue_with(0.5, Some(AWAY));
    assert_thought(&actions, 1, &Reason::Keyword);
    assert!(!attending);
    // The tagger and the direction both object: someone else, however close in time.
    let (actions, attending) = continue_with(0.5, Some(TV));
    assert_abstention(&actions, Abstention::Media);
    assert!(attending);
    // The direction alone never turns a cue away.
    let (actions, _) = continue_with(0.4625, Some(TV));
    assert_thought(&actions, 1, &Reason::Keyword);
}

#[test]
fn one_sensor_never_raises_two_objections() {
    // Voice and tagger clear their own bars; the direction's weight on the loudspeaker
    // alternative is what makes the voice another's. That is still one sensor objecting,
    // so right after the unfinished name it is excused...
    let cue = |now, duration_ms| heard(now, false, duration_ms, (0.6, 0.486_25, 0.95), Some(TV));
    let mut organism = after_an_unfinished_name(Organism::new(lab()).unwrap());
    assert_thought(&organism.step(&cue(9_200, 900)), 1, &Reason::Keyword);
    // ...while two seconds after the name, closeness excuses nothing.
    let mut organism = after_an_unfinished_name(Organism::new(lab()).unwrap());
    assert_abstention(&organism.step(&cue(11_200, 900)), Abstention::OtherSpeaker);
    assert!(organism.is_attending());
}

#[test]
fn confined_the_tv_caution_weighs_on_the_loudspeaker_alone() {
    // An owner's follow-up that voice and tagger alone cannot tell from the TV: no
    // evidence either way against another person, half a nat for a live source.
    let owner_like = (0.55, 0.493_75, 0.95);
    let follow_up = |confined: bool, direction| {
        let mut profile = lab();
        profile.source.tv_caution_confinement = if confined {
            TvCautionConfinement::Always
        } else {
            TvCautionConfinement::Never
        };
        let mut organism = in_conversation(with_the_tv_on(Organism::new(profile).unwrap()));
        organism.step(&heard(12_000, false, 1_500, owner_like, direction))
    };
    // The whole caution on every sensor: the tagger's reading alone turns it away.
    assert_abstention(&follow_up(false, Some(AWAY)), Abstention::Media);
    // Confined, the direction that places it away from the TV lets it through...
    assert_thought(&follow_up(true, Some(AWAY)), 2, &Reason::FollowUp);
    // ...and the one in line with the TV turns it away as a loudspeaker.
    assert_abstention(&follow_up(true, Some(TV)), Abstention::OtherSpeaker);
    // Without a reading, the caution applies as it always did.
    assert_abstention(&follow_up(true, None), Abstention::Media);
}

#[test]
fn by_default_the_caution_is_confined_only_where_directedness_also_judged() {
    // The same owner-like follow-up from away from the TV, with the shipped setting:
    // without a directedness reading nobody answers for other people, so the whole
    // caution stays; with one, the caution weighs on the loudspeaker alone.
    let owner_like = (0.55, 0.493_75, 0.95);
    let follow_up = |directed: Option<f32>| {
        let mut organism = in_conversation(with_the_tv_on(Organism::new(lab()).unwrap()));
        let Event::Speech { now, mut cue } = heard(12_000, false, 1_500, owner_like, Some(AWAY))
        else {
            unreachable!("heard builds speech")
        };
        cue.directed = directed;
        organism.step(&Event::Speech { now, cue })
    };
    assert_eq!(
        Profile::t1_ref().source.tv_caution_confinement,
        TvCautionConfinement::WithDirectedness
    );
    assert_abstention(&follow_up(None), Abstention::Media);
    assert_thought(&follow_up(Some(0.9)), 2, &Reason::FollowUp);
}

#[test]
fn a_snapshot_keeps_what_was_learned_and_an_old_one_starts_from_nothing() {
    let organism = with_the_tv_on(Organism::new(lab()).unwrap());
    let blob = serde_json::to_string(&organism).unwrap();
    let mut restored: Organism = serde_json::from_str(&blob).unwrap();
    assert_eq!(restored, organism);
    let mut live = in_conversation(organism);
    restored = in_conversation(restored);
    let follow_up = heard(12_000, false, 1_500, BORDERLINE, Some(TV));
    let decided = live.step(&follow_up);
    assert_abstention(&decided, Abstention::OtherSpeaker);
    assert_eq!(restored.step(&follow_up), decided);
    assert_eq!(restored, live);

    // A snapshot written before the array existed restores with nothing learned.
    let mut stored: serde_json::Value = serde_json::from_str(&blob).unwrap();
    assert!(
        stored
            .as_object_mut()
            .unwrap()
            .remove("tv_direction")
            .is_some()
    );
    let old: Organism = serde_json::from_value(stored).unwrap();
    assert_eq!(old.tv_direction_as_of(Millis(8_000)), None);
}

#[test]
fn a_profile_stored_before_the_array_reads_back_with_its_defaults() {
    for shipped in [Profile::t1_ref(), Profile::desktop()] {
        let mut stored = serde_json::to_value(&shipped).unwrap();
        let fields = stored.as_object_mut().unwrap();
        for key in [
            "tv_direction_half_life_ms",
            "tv_direction_min_lines",
            "tv_caution_confinement",
        ] {
            assert!(fields.remove(key).is_some(), "{key}");
        }
        fields["senses"]
            .as_object_mut()
            .unwrap()
            .remove("direction");
        let profile: Profile = serde_json::from_value(stored).unwrap();
        assert_eq!(profile, shipped);
    }
}

#[test]
fn the_profile_rejects_a_direction_that_could_never_be_trusted_or_forgotten() {
    for broken in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let mut profile = Profile::t1_ref();
        profile.source.tv_direction_min_lines = broken;
        assert!(profile.validate().is_err(), "min lines {broken}");
    }
    let mut profile = Profile::t1_ref();
    profile.source.tv_direction_half_life_ms = 0;
    assert!(profile.validate().is_err());
}
