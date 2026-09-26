//! What a discretionary thought needs before it is bought: something on the checklist
//! (drives), the owner at home, and a cortex that answers. Obligations need none of it.
//! Drive thoughts are answered by their reply, silence included, and a failed or
//! superseded one leaves its drive free to ask again later.

use enton_core::{Abstention, Action, Event, Millis, Organism, Profile, Reason, ThoughtId};

use super::support::{
    assert_abstention, assert_thought, body, checklist, lab_profile, owner_nearby, speech, voice,
};

/// The lab calibration with drives that ignite within the hour (see `sensitive_profile`).
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

/// Something on the checklist, and the owner heard nearby at `now`.
fn home_with_something_to_check(mut organism: Organism, now: u64) -> Organism {
    assert!(organism.step(&checklist(0, true)).is_empty());
    organism.step(&owner_nearby(now));
    organism
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

// ---------------------------------------------------------------------------
// Something to check

#[test]
fn a_drive_with_nothing_to_check_waits_once_and_spends_nothing() {
    let mut organism = Organism::new(drives_profile()).unwrap();
    organism.step(&owner_nearby(3_000_000));
    // Before any checklist reading, nothing is known: there is nothing to check.
    assert_abstention(&organism.step(&tick(3_600_000)), Abstention::NothingToCheck);
    // The wait is logged when it starts, not on every tick, and costs nothing.
    for second in 1..=30 {
        assert!(organism.step(&tick(3_600_000 + second * 1_000)).is_empty());
    }
    let budget = organism.discretionary_budget();
    assert!((budget.available - budget.capacity).abs() < f32::EPSILON);
    // A checklist of headings and empty items says the same, silently.
    assert!(organism.step(&checklist(3_631_000, false)).is_empty());
    assert!(organism.step(&tick(3_632_000)).is_empty());
    assert!(!organism.checklist_actionable());
    // Once the owner writes something to check, the drive thinks.
    organism.step(&checklist(3_633_000, true));
    assert_thought(&organism.step(&tick(3_634_000)), 1, &curiosity());
}

#[test]
fn an_empty_checklist_never_holds_back_an_obligation_or_overheard_speech() {
    let mut organism = Organism::new(lab_profile()).unwrap();
    organism.step(&checklist(0, false));
    assert_thought(&organism.step(&speech(1_000, true)), 1, &Reason::Keyword);
    organism.step(&tick(30_000));
    assert_thought(&organism.step(&speech(30_000, false)), 2, &Reason::Speech);
}

// ---------------------------------------------------------------------------
// Silence as an outcome

#[test]
fn a_silent_reply_answers_the_drive_that_asked_and_it_asks_again_only_later() {
    let mut organism =
        home_with_something_to_check(Organism::new(drives_profile()).unwrap(), 3_000_000);
    assert_thought(&organism.step(&tick(3_600_000)), 1, &curiosity());
    assert_eq!(organism.drive_thought(), Some((ThoughtId(1), "curiosity")));
    assert!(level(&organism, "curiosity") > 0.2);

    // Nothing on the checklist needed saying: the cortex stays silent, and nothing is said.
    assert!(organism.step(&reply(3_602_000, 1, "")).is_empty());
    assert_eq!(organism.drive_thought(), None);
    assert!(level(&organism, "curiosity").abs() < f32::EPSILON);
    assert_eq!(organism.cortex_failures(), 0);

    // A quiet check does not fire again at the next tick, nor for a long while.
    for second in 1..=120 {
        assert!(
            organism.step(&tick(3_602_000 + second * 1_000)).is_empty(),
            "second {second}"
        );
    }
    // Once its pressure builds back up, with the owner still around, it asks again.
    organism.step(&owner_nearby(5_000_000));
    assert_thought(&organism.step(&tick(5_400_000)), 2, &curiosity());
}

#[test]
fn a_reply_with_something_to_say_answers_the_drive_and_is_spoken() {
    let mut organism =
        home_with_something_to_check(Organism::new(drives_profile()).unwrap(), 3_000_000);
    assert_thought(&organism.step(&tick(3_600_000)), 1, &curiosity());
    assert_eq!(
        organism.step(&reply(3_602_000, 1, "Lembrete: regar as plantas.")),
        vec![Action::Speak {
            text: "Lembrete: regar as plantas.".to_owned()
        }]
    );
    assert!(level(&organism, "curiosity").abs() < f32::EPSILON);
}

#[test]
fn only_the_reply_to_the_drive_thought_answers_the_drive() {
    let mut organism =
        home_with_something_to_check(Organism::new(drives_profile()).unwrap(), 3_000_000);
    // The owner called earlier; that thought's reply comes late, after a drive asked.
    assert_thought(
        &organism.step(&speech(3_500_000, true)),
        1,
        &Reason::Keyword,
    );
    assert_thought(&organism.step(&tick(3_600_000)), 2, &curiosity());
    let before = level(&organism, "curiosity");
    organism.step(&reply(3_600_500, 1, "Oi!"));
    assert_eq!(organism.drive_thought(), Some((ThoughtId(2), "curiosity")));
    assert!((level(&organism, "curiosity") - before).abs() < f32::EPSILON);
    organism.step(&reply(3_601_000, 2, ""));
    assert!(level(&organism, "curiosity").abs() < f32::EPSILON);
}

#[test]
fn a_drive_superseded_by_the_owner_waits_for_the_conversation_then_asks_again() {
    let mut organism =
        home_with_something_to_check(Organism::new(drives_profile()).unwrap(), 3_000_000);
    assert_thought(&organism.step(&tick(3_600_000)), 1, &curiosity());
    // The owner calls while the drive's thought is out: the runtime drops it for theirs.
    assert_thought(
        &organism.step(&speech(3_601_000, true)),
        2,
        &Reason::Keyword,
    );
    assert_eq!(organism.drive_thought(), None);
    organism.step(&reply(3_603_000, 2, "Oi!"));
    // The drive, unanswered, never cuts into the conversation...
    for now in (3_604_000..3_613_000).step_by(1_000) {
        assert!(organism.step(&tick(now)).is_empty(), "{now} ms");
    }
    // ...and asks again once it is over.
    assert_thought(&organism.step(&tick(3_613_000)), 3, &curiosity());
}

#[test]
fn a_call_by_name_does_not_use_up_a_waiting_drive() {
    let mut organism = Organism::new(drives_profile()).unwrap();
    organism.step(&checklist(0, true));
    // Nobody is home when the drive gets ready.
    assert_abstention(&organism.step(&tick(3_600_000)), Abstention::NobodyHome);
    // The owner comes home and calls Enton; answering them does not answer the drive.
    assert_thought(
        &organism.step(&speech(3_601_000, true)),
        1,
        &Reason::Keyword,
    );
    organism.step(&reply(3_602_000, 1, "Oi!"));
    for now in (3_603_000..3_612_000).step_by(1_000) {
        assert!(organism.step(&tick(now)).is_empty(), "{now} ms");
    }
    assert_thought(&organism.step(&tick(3_612_000)), 2, &curiosity());
}

// ---------------------------------------------------------------------------
// Nobody home

#[test]
fn overheard_speech_needs_the_owner_home_and_a_call_by_name_never_does() {
    let profile = lab_profile();
    let window = profile.discretion.presence_window_ms;
    let mut organism = Organism::new(profile).unwrap();
    // Nobody has been heard yet: speech worth a thought abstains, and spends nothing.
    assert_abstention(
        &organism.step(&speech(1_000, false)),
        Abstention::NobodyHome,
    );
    assert_eq!(organism.owner_heard_at(), None);
    // Calling Enton by name is always answered, and says someone is home.
    assert_thought(&organism.step(&speech(20_000, true)), 1, &Reason::Keyword);
    assert_eq!(organism.owner_heard_at(), Some(Millis(20_000)));
    organism.step(&tick(40_000));
    assert_thought(&organism.step(&speech(40_000, false)), 2, &Reason::Speech);
    // Presence lasts the window, counted from when the owner was last heard.
    organism.step(&tick(20_000 + window));
    assert!(!organism.owner_present_as_of(Millis(20_000 + window)));
    assert_abstention(
        &organism.step(&speech(20_000 + window, false)),
        Abstention::NobodyHome,
    );
    // The owner's verified voice, without a word to Enton, says they are home too.
    assert_thought(
        &organism.step(&voice(20_001 + window, false, 1_500, 0.9)),
        3,
        &Reason::Speech,
    );
}

#[test]
fn another_voice_or_a_loudspeaker_does_not_say_the_owner_is_home() {
    let mut organism = Organism::new(lab_profile()).unwrap();
    // A voice the verifier rules out, speaking without the name: nobody home.
    assert_abstention(
        &organism.step(&voice(1_000, false, 1_500, 0.25)),
        Abstention::NobodyHome,
    );
    assert_eq!(organism.owner_heard_at(), None);
}

#[test]
fn a_drive_waits_for_somebody_home_and_thinks_when_the_owner_is_heard() {
    let mut organism = Organism::new(drives_profile()).unwrap();
    organism.step(&checklist(0, true));
    assert_abstention(&organism.step(&tick(3_600_000)), Abstention::NobodyHome);
    assert!(organism.step(&tick(3_601_000)).is_empty());
    organism.step(&owner_nearby(3_601_500));
    assert_thought(&organism.step(&tick(3_602_000)), 1, &curiosity());
}

// ---------------------------------------------------------------------------
// Backing off after cortex failures

#[test]
fn failures_back_off_discretionary_thoughts_exponentially_and_a_reply_ends_it() {
    let profile = lab_profile();
    let base = profile.discretion.cortex_backoff_base_ms;
    let mut organism = Organism::new(profile).unwrap();
    // The owner calls, and the cortex fails to answer.
    assert_thought(&organism.step(&speech(1_000, true)), 1, &Reason::Keyword);
    assert!(organism.step(&failed(2_000, 1)).is_empty());
    assert_eq!(organism.cortex_failures(), 1);
    assert_eq!(organism.backoff_until(), Some(Millis(2_000 + base)));
    // The failed answer no longer holds the conversation open.
    assert_eq!(organism.conversation_thought(), None);

    // Speech worth a thought, with the owner home, waits out the backoff.
    organism.step(&tick(20_000));
    assert_abstention(&organism.step(&speech(20_000, false)), Abstention::Backoff);
    // The owner calling again is always tried; when it fails too, the wait doubles.
    assert_thought(&organism.step(&speech(30_000, true)), 2, &Reason::Keyword);
    organism.step(&failed(31_000, 2));
    assert_eq!(organism.backoff_until(), Some(Millis(31_000 + 2 * base)));
    // A failure of a thought never issued changes nothing.
    organism.step(&failed(31_500, 99));
    assert_eq!(organism.cortex_failures(), 2);

    // Past the wait, discretionary thoughts try again.
    let after = 31_000 + 2 * base;
    organism.step(&tick(after));
    assert_thought(&organism.step(&speech(after, false)), 3, &Reason::Speech);
    // A reply ends the backoff at once.
    organism.step(&failed(after + 1_000, 3));
    assert_eq!(organism.cortex_failures(), 3);
    assert_thought(
        &organism.step(&speech(after + 2_000, true)),
        4,
        &Reason::Keyword,
    );
    organism.step(&reply(after + 3_000, 4, "Voltei."));
    assert_eq!(organism.cortex_failures(), 0);
    assert_eq!(organism.backoff_until(), None);
}

#[test]
fn a_failed_drive_thought_asks_again_only_after_the_backoff() {
    let mut organism =
        home_with_something_to_check(Organism::new(drives_profile()).unwrap(), 3_000_000);
    let base = organism.profile().discretion.cortex_backoff_base_ms;
    assert_thought(&organism.step(&tick(3_600_000)), 1, &curiosity());
    organism.step(&failed(3_601_000, 1));
    assert_eq!(organism.drive_thought(), None);
    // Its drive is still unanswered, but no tick hammers a dead cortex: the wait is
    // logged once...
    assert_abstention(&organism.step(&tick(3_602_000)), Abstention::Backoff);
    for now in (3_603_000..3_601_000 + base).step_by(1_000) {
        assert!(organism.step(&tick(now)).is_empty(), "{now} ms");
    }
    // ...and the drive asks again when it ends.
    assert_thought(&organism.step(&tick(3_601_000 + base)), 2, &curiosity());
}

#[test]
fn the_backoff_doubles_up_to_its_cap_without_overflow() {
    let policy = Profile::t1_ref().discretion;
    let base = policy.cortex_backoff_base_ms;
    assert_eq!(policy.backoff_ms(0), 0);
    assert_eq!(policy.backoff_ms(1), base);
    assert_eq!(policy.backoff_ms(2), 2 * base);
    assert_eq!(policy.backoff_ms(4), 8 * base);
    for failures in [7, 30, 63, 64, 65, u32::MAX] {
        assert_eq!(policy.backoff_ms(failures), policy.cortex_backoff_cap_ms);
    }
}

// ---------------------------------------------------------------------------
// Order of the gates

#[test]
fn overheard_speech_is_held_back_by_nobody_home_then_backoff_then_energy() {
    let mut profile = lab_profile();
    profile.budgets.discretionary_budget_per_hour = 1.0;
    profile.discretion.cortex_backoff_base_ms = 7_200_000;
    profile.discretion.cortex_backoff_cap_ms = 7_200_000;
    let window = profile.discretion.presence_window_ms;
    let mut organism = Organism::new(profile).unwrap();
    // The owner's own words buy the only discretionary thought there is, which fails.
    assert_thought(
        &organism.step(&voice(0, false, 1_500, 0.9)),
        1,
        &Reason::Speech,
    );
    organism.step(&failed(500, 1));
    // The cooldown after that thought comes first.
    assert_abstention(&organism.step(&speech(5_000, false)), Abstention::Cooldown);
    // Out of energy and backing off: the backoff is named, not the budget.
    assert_abstention(&organism.step(&speech(20_000, false)), Abstention::Backoff);
    // Gone long enough, nobody home comes before the backoff.
    organism.step(&tick(window + 1));
    assert_abstention(
        &organism.step(&speech(window + 1, false)),
        Abstention::NobodyHome,
    );
}

#[test]
fn a_ready_drive_names_torpor_then_the_checklist_then_nobody_home_then_backoff() {
    let mut organism = Organism::new(drives_profile()).unwrap();
    // One wait after another, each logged once, as its cause changes.
    organism.step(&body(0, Some(100.0), None));
    assert_abstention(&organism.step(&tick(3_600_000)), Abstention::Torpor);
    organism.step(&body(3_600_500, None, None));
    assert_abstention(&organism.step(&tick(3_601_000)), Abstention::NothingToCheck);
    organism.step(&checklist(3_601_500, true));
    assert_abstention(&organism.step(&tick(3_602_000)), Abstention::NobodyHome);
    // The owner calls and the answer fails; after the conversation, the drive backs off.
    assert_thought(
        &organism.step(&speech(3_602_500, true)),
        1,
        &Reason::Keyword,
    );
    organism.step(&failed(3_603_000, 1));
    for now in (3_603_000..3_612_500).step_by(500) {
        assert!(organism.step(&tick(now)).is_empty(), "{now} ms");
    }
    assert_abstention(&organism.step(&tick(3_612_500)), Abstention::Backoff);
}

// ---------------------------------------------------------------------------
// Replay

#[test]
fn checklist_presence_and_failures_replay_from_json_and_survive_a_snapshot() {
    let tape = [
        checklist(0, false),
        tick(1_800_000),
        owner_nearby(3_000_000),
        tick(3_600_000),
        checklist(3_600_500, true),
        tick(3_601_000),
        failed(3_601_500, 1),
        tick(3_602_000),
        speech(3_650_000, true),
        reply(3_651_000, 2, "Oi!"),
        tick(3_700_000),
        reply(3_701_000, 3, ""),
        speech(3_702_000, false),
        tick(5_500_000),
        speech(5_500_001, false),
    ];
    let mut live = Organism::new(drives_profile()).unwrap();
    let decided: Vec<_> = tape.iter().map(|event| live.step(event)).collect();
    // Every kind of decision the rules can take shows up.
    let whys: Vec<Abstention> = decided
        .iter()
        .flatten()
        .filter_map(|action| match action {
            Action::Abstain { why, .. } => Some(*why),
            _ => None,
        })
        .collect();
    for expected in [
        Abstention::NothingToCheck,
        Abstention::Backoff,
        Abstention::NobodyHome,
    ] {
        assert!(whys.contains(&expected), "{expected:?} in {whys:?}");
    }

    let mut replayed = Organism::new(drives_profile()).unwrap();
    for (event, expected) in tape.iter().zip(&decided) {
        let stored = serde_json::to_string(&event.clone().canonical()).unwrap();
        let read: Event = serde_json::from_str(&stored).unwrap();
        assert_eq!(&replayed.step(&read), expected, "{stored}");
    }
    assert_eq!(replayed, live);

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
