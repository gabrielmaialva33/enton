//! Deferred intents: a drive that gets ready while Enton is in a conversation holds its
//! intent instead of cutting the owner off, and rides the next answer the owner asks for,
//! at no extra paid call. Once the conversation is over it may think alone instead; with
//! neither before its deferral runs out, it lets the intent go, logged once.

use enton_core::{
    Abstention, Action, Deferred, Event, Millis, Organism, Profile, Reason, ThoughtId,
};

use super::support::{
    assert_abstention, assert_thought, body, checklist, lab_profile, owner_nearby, quiet,
    quiet_hours, speech,
};

/// The lab calibration with drives that ignite within the hour and no cooldown.
fn drives_profile() -> Profile {
    let mut profile = lab_profile();
    profile.ignition.threshold = 0.01;
    profile.ignition.hysteresis = 0.002;
    profile.ignition.ema_alpha = 1.0;
    profile.ignition.cooldown_ms = 0;
    profile
}

fn tick(now: u64) -> Event {
    Event::Tick { now: Millis(now) }
}

fn reply(now: u64, thought: u64, text: &str) -> Event {
    Event::CortexReply {
        now: Millis(now),
        thought: ThoughtId(thought),
        text: text.to_owned(),
    }
}

fn failed(now: u64, thought: u64) -> Event {
    Event::CortexFailed {
        now: Millis(now),
        thought: ThoughtId(thought),
    }
}

fn curiosity() -> Reason {
    Reason::Drive("curiosity".to_owned())
}

/// A drive's level; not a number when there is no such drive, which fails any check.
fn level(organism: &Organism, drive: &str) -> f32 {
    organism
        .drives()
        .as_slice()
        .iter()
        .find(|candidate| candidate.name == drive)
        .map_or(f32::NAN, |found| found.level)
}

/// The owner at home with something on the checklist asks Enton something at 3 599 s,
/// and the drives, an hour old, get ready at the next tick, mid-conversation: curiosity
/// holds its intent. The answer to the request comes at 3 601 s.
fn mid_conversation(mut organism: Organism) -> Organism {
    assert!(organism.step(&checklist(0, true)).is_empty());
    organism.step(&owner_nearby(3_000_000));
    assert_thought(
        &organism.step(&speech(3_599_000, true)),
        1,
        &Reason::Keyword,
    );
    // The drive does not cut in: it holds its intent, silently.
    assert!(organism.step(&tick(3_600_000)).is_empty());
    assert_eq!(
        organism.deferred(),
        Some(&Deferred {
            drive: "curiosity".to_owned(),
            since: Millis(3_600_000),
            expires: Millis(3_600_000 + organism.profile().discretion.deferral_ms),
        })
    );
    assert_eq!(
        organism.step(&reply(3_601_000, 1, "Oi!")),
        vec![Action::Speak {
            text: "Oi!".to_owned()
        }]
    );
    organism
}

/// A thought the owner asked for, carrying `drive`'s intent.
fn assert_ride(actions: &[Action], expected: u64, expected_reason: &Reason, drive: &str) {
    assert!(
        matches!(actions, [Action::Think { thought, reason, rider: Some(rider), propensity: None, .. }]
            if *thought == ThoughtId(expected) && reason == expected_reason && rider == drive),
        "expected thought {expected} for {expected_reason:?} carrying {drive}, got {actions:?}"
    );
}

// ---------------------------------------------------------------------------
// Riding

#[test]
fn a_drive_ready_mid_conversation_rides_the_owners_next_request() {
    let mut organism = mid_conversation(Organism::new(drives_profile()).unwrap());
    // The conversation goes on; the intent waits without a word.
    for now in (3_602_000..3_605_000).step_by(1_000) {
        assert!(organism.step(&tick(now)).is_empty(), "{now} ms");
    }
    // The owner's follow-up is answered, and the answer carries the drive's intent.
    assert_ride(
        &organism.step(&speech(3_605_000, false)),
        2,
        &Reason::FollowUp,
        "curiosity",
    );
    assert_eq!(organism.drive_thought(), Some((ThoughtId(2), "curiosity")));
    // While the ride is out, the drive neither asks again nor lets go.
    for now in (3_605_500..3_607_000).step_by(500) {
        assert!(organism.step(&tick(now)).is_empty(), "{now} ms");
    }
    // Its reply answers the drive, as the drive's own thought would.
    organism.step(&reply(3_607_000, 2, "Claro. E as plantas, já regou?"));
    assert!(level(&organism, "curiosity").abs() < f32::EPSILON);
    assert_eq!(organism.deferred(), None);
    assert_eq!(organism.drive_thought(), None);
    // No thought of Enton's own was bought: the discretionary account is untouched...
    let budget = organism.discretionary_budget();
    assert!((budget.available - budget.capacity).abs() < f32::EPSILON);
    // ...and the answered drive does not ask again after the conversation.
    for now in (3_608_000..3_608_000 + 600_000).step_by(10_000) {
        assert!(organism.step(&tick(now)).is_empty(), "{now} ms");
    }
}

#[test]
fn without_a_request_to_ride_the_drive_thinks_alone_once_the_conversation_ends() {
    let mut organism = mid_conversation(Organism::new(drives_profile()).unwrap());
    // The reply reopened the windows: 5 s for anyone, 10 s for the owner's voice.
    for now in (3_602_000..3_611_000).step_by(1_000) {
        assert!(organism.step(&tick(now)).is_empty(), "{now} ms");
    }
    let actions = organism.step(&tick(3_611_000));
    assert_thought(&actions, 2, &curiosity());
    assert!(matches!(
        actions.as_slice(),
        [Action::Think { rider: None, .. }]
    ));
    // Its own thought took the intent's place.
    assert_eq!(organism.deferred(), None);
    assert_eq!(organism.drive_thought(), Some((ThoughtId(2), "curiosity")));
}

#[test]
fn a_call_by_name_carries_the_intent_too() {
    let mut organism = mid_conversation(Organism::new(drives_profile()).unwrap());
    // The owner calls again by name inside the window: a whole request, answered at once.
    assert_ride(
        &organism.step(&speech(3_603_000, true)),
        2,
        &Reason::Keyword,
        "curiosity",
    );
}

#[test]
fn an_unfinished_name_waits_and_the_answer_that_follows_carries_the_intent() {
    let mut organism = mid_conversation(Organism::new(drives_profile()).unwrap());
    // "Enton..." alone: Enton waits for the rest, and no thought is bought yet.
    let short = Event::Speech {
        now: Millis(3_603_000),
        cue: enton_core::SpeechCue {
            duration_ms: 400,
            ..match speech(0, true) {
                Event::Speech { cue, .. } => cue,
                _ => unreachable!("speech builds a speech cue"),
            }
        },
    };
    assert!(matches!(
        organism.step(&short).as_slice(),
        [Action::Attend { .. }]
    ));
    // Nobody continues: the wait times out into an answer, which the intent rides.
    let until = organism.attention_until().unwrap();
    assert_ride(
        &organism.step(&tick(until.0)),
        2,
        &Reason::Keyword,
        "curiosity",
    );
}

#[test]
fn a_ride_lost_to_a_newer_request_moves_to_it() {
    let mut organism = mid_conversation(Organism::new(drives_profile()).unwrap());
    assert_ride(
        &organism.step(&speech(3_605_000, false)),
        2,
        &Reason::FollowUp,
        "curiosity",
    );
    // The owner goes on before the answer: the runtime drops thought 2 for thought 3,
    // and the intent rides the newer one.
    assert_ride(
        &organism.step(&speech(3_606_000, false)),
        3,
        &Reason::FollowUp,
        "curiosity",
    );
    // A late reply to the dropped thought answers nothing.
    organism.step(&reply(3_606_500, 2, "Oi."));
    assert!(level(&organism, "curiosity") > 0.2);
    assert!(organism.deferred().is_some());
    organism.step(&reply(3_607_000, 3, "Pronto."));
    assert!(level(&organism, "curiosity").abs() < f32::EPSILON);
    assert_eq!(organism.deferred(), None);
}

#[test]
fn a_failed_ride_keeps_the_intent_for_the_next_request() {
    let mut organism = mid_conversation(Organism::new(drives_profile()).unwrap());
    assert_ride(
        &organism.step(&speech(3_605_000, false)),
        2,
        &Reason::FollowUp,
        "curiosity",
    );
    organism.step(&failed(3_605_500, 2));
    assert_eq!(organism.drive_thought(), None);
    assert!(organism.deferred().is_some());
    // The cortex backs off thoughts of Enton's own, but a ride buys none: the owner's
    // next request carries the intent all the same.
    assert!(organism.backoff_until().is_some());
    assert_ride(
        &organism.step(&speech(3_606_000, true)),
        3,
        &Reason::Keyword,
        "curiosity",
    );
}

// ---------------------------------------------------------------------------
// Expiry

#[test]
fn an_intent_with_no_turn_to_take_expires_once_and_lets_go() {
    let mut organism = mid_conversation(Organism::new(drives_profile()).unwrap());
    let expires = organism.deferred().unwrap().expires.0;
    // The owner asks for quiet: the conversation is over, and neither a ride nor a
    // thought of the drive's own may come before the intent runs out.
    assert!(
        organism
            .step(&quiet(3_602_000, 3_602_000 + 3_600_000))
            .is_empty()
    );
    assert_abstention(&organism.step(&tick(3_603_000)), Abstention::Quiet);
    for now in (3_613_000..expires).step_by(10_000) {
        assert!(organism.step(&tick(now)).is_empty(), "{now} ms");
    }
    let actions = organism.step(&tick(expires));
    assert_abstention(&actions, Abstention::Expired);
    assert!(matches!(
        actions.as_slice(),
        [Action::Abstain { reason, .. }] if *reason == curiosity()
    ));
    // Let go as if answered by silence: nothing more is logged while its pressure is low.
    assert_eq!(organism.deferred(), None);
    assert!(level(&organism, "curiosity").abs() < f32::EPSILON);
    for now in (expires + 10_000..expires + 600_000).step_by(10_000) {
        assert!(organism.step(&tick(now)).is_empty(), "{now} ms");
    }
}

#[test]
fn a_ride_in_flight_is_waited_for_past_the_deadline() {
    let mut profile = drives_profile();
    profile.discretion.deferral_ms = 4_000;
    let mut organism = mid_conversation(Organism::new(profile).unwrap());
    assert_ride(
        &organism.step(&speech(3_603_000, false)),
        2,
        &Reason::FollowUp,
        "curiosity",
    );
    // The deadline passes with the ride out: nothing expires...
    assert!(organism.step(&tick(3_605_000)).is_empty());
    // ...and when the ride fails, the intent has run out and lets go at the next tick.
    organism.step(&failed(3_605_500, 2));
    assert_abstention(&organism.step(&tick(3_606_000)), Abstention::Expired);
    assert_eq!(organism.deferred(), None);
}

// ---------------------------------------------------------------------------
// What a ride needs

#[test]
fn an_intent_is_held_only_when_the_drive_could_think_but_for_the_conversation() {
    let hold = |setup: &[Event]| {
        let mut organism = Organism::new(drives_profile()).unwrap();
        organism.step(&checklist(0, true));
        organism.step(&owner_nearby(3_000_000));
        for event in setup {
            organism.step(event);
        }
        assert_thought(
            &organism.step(&speech(3_599_000, true)),
            1,
            &Reason::Keyword,
        );
        organism.step(&tick(3_600_000));
        organism.deferred().cloned()
    };
    assert!(hold(&[]).is_some());
    assert_eq!(hold(&[checklist(1_000, false)]), None, "nothing to check");
    assert_eq!(hold(&[quiet_hours(1_000, true)]), None, "quiet hours");
    assert_eq!(hold(&[body(1_000, Some(100.0), None)]), None, "torpor");
}

#[test]
fn an_answer_carries_no_intent_while_quiet_hours_hold_and_does_again_after() {
    let mut organism = mid_conversation(Organism::new(drives_profile()).unwrap());
    organism.step(&quiet_hours(3_602_000, true));
    let actions = organism.step(&speech(3_603_000, false));
    assert_thought(&actions, 2, &Reason::FollowUp);
    assert!(matches!(
        actions.as_slice(),
        [Action::Think { rider: None, .. }]
    ));
    assert_eq!(organism.drive_thought(), None);
    // The intent is still held: once the band ends, the next request carries it.
    organism.step(&reply(3_604_000, 2, "Tá."));
    organism.step(&quiet_hours(3_604_500, false));
    assert_ride(
        &organism.step(&speech(3_605_000, true)),
        3,
        &Reason::Keyword,
        "curiosity",
    );
}

#[test]
fn overheard_speech_never_carries_an_intent() {
    let mut organism = mid_conversation(Organism::new(drives_profile()).unwrap());
    // Past the windows, overheard speech buys a thought of Enton's own, and the drive
    // thinks alone at the same tick it would anyway: nothing rides a discretionary thought.
    for now in (3_602_000..3_611_000).step_by(1_000) {
        organism.step(&tick(now));
    }
    let actions = organism.step(&speech(3_610_500, false));
    assert!(
        matches!(
            actions.as_slice(),
            [Action::Think {
                reason: Reason::FollowUp | Reason::Speech,
                rider: None,
                ..
            }]
        ),
        "{actions:?}"
    );
}

// ---------------------------------------------------------------------------
// Replay

#[test]
fn deferrals_rides_and_expiries_replay_from_json_and_survive_a_snapshot() {
    let tape = [
        checklist(0, true),
        owner_nearby(3_000_000),
        speech(3_599_000, true),
        tick(3_600_000),
        reply(3_601_000, 1, "Oi!"),
        speech(3_605_000, false),
        speech(3_606_000, false),
        reply(3_606_500, 2, "Oi."),
        failed(3_607_000, 3),
        tick(3_607_500),
        speech(3_608_000, true),
        reply(3_609_000, 4, "Pronto."),
        // Half an hour later the drives are ready again, mid-conversation, and the owner
        // asks for quiet: the new intent expires.
        speech(5_399_000, true),
        tick(5_400_000),
        quiet(5_401_000, 9_000_000),
        tick(5_402_000),
        tick(6_000_000),
        tick(6_010_000),
    ];
    let mut live = Organism::new(drives_profile()).unwrap();
    let decided: Vec<_> = tape.iter().map(|event| live.step(event)).collect();
    let flat: Vec<&Action> = decided.iter().flatten().collect();
    assert!(
        flat.iter()
            .any(|action| matches!(action, Action::Think { rider: Some(_), .. })),
        "a ride in {flat:?}"
    );
    assert!(
        flat.iter().any(|action| matches!(
            action,
            Action::Abstain {
                why: Abstention::Expired,
                ..
            }
        )),
        "an expiry in {flat:?}"
    );

    let mut replayed = Organism::new(drives_profile()).unwrap();
    for (event, expected) in tape.iter().zip(&decided) {
        let stored = serde_json::to_string(&event.clone().canonical()).unwrap();
        let read: Event = serde_json::from_str(&stored).unwrap();
        assert_eq!(&replayed.step(&read), expected, "{stored}");
    }
    assert_eq!(replayed, live);
    // A decision log stores the ride with the thought, and reads it back.
    for action in &flat {
        let stored = serde_json::to_string(action).unwrap();
        assert_eq!(&&serde_json::from_str::<Action>(&stored).unwrap(), action);
    }

    for cut in 0..=tape.len() {
        let mut first = Organism::new(drives_profile()).unwrap();
        for event in &tape[..cut] {
            first.step(event);
        }
        let blob = serde_json::to_string(&first).unwrap();
        let mut resumed: Organism = serde_json::from_str(&blob).unwrap();
        assert_eq!(resumed, first, "cut {cut}");
        for (event, expected) in tape[cut..].iter().zip(&decided[cut..]) {
            assert_eq!(&resumed.step(event), expected, "cut {cut}");
        }
        assert_eq!(resumed, live, "cut {cut}");
    }
}

#[test]
fn a_thought_nothing_rides_is_stored_as_before_rides() {
    let think = Action::Think {
        thought: ThoughtId(7),
        reason: Reason::Keyword,
        salience: 0.5,
        propensity: None,
        rider: None,
    };
    let json = serde_json::to_string(&think).unwrap();
    assert!(!json.contains("rider"), "{json}");
    assert_eq!(serde_json::from_str::<Action>(&json).unwrap(), think);
    let ridden = Action::Think {
        thought: ThoughtId(7),
        reason: Reason::Keyword,
        salience: 0.5,
        propensity: None,
        rider: Some("curiosity".to_owned()),
    };
    let json = serde_json::to_string(&ridden).unwrap();
    assert!(json.contains(r#""rider":"curiosity""#), "{json}");
    assert_eq!(serde_json::from_str::<Action>(&json).unwrap(), ridden);
}
