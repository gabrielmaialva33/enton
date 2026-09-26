//! Quiet mode and quiet hours: the owner's command ("Enton, silêncio") and the night band
//! hold back every thought of Enton's own, and never an answer the owner asks for. The
//! command itself is an instruction, not a request: it buys no thought.

use enton_core::{
    Abstention, Action, Event, Millis, Organism, Profile, Reason, SpeechCue, ThoughtId,
};

use super::support::{
    assert_abstention, assert_thought, checklist, lab_profile, owner_nearby, quiet, quiet_hours,
    speech,
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

fn curiosity() -> Reason {
    Reason::Drive("curiosity".to_owned())
}

/// "Enton..." alone: the name, too short to hold a request.
fn name_alone(now: u64) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 1.0,
            duration_ms: 400,
            vad_confidence: 1.0,
            keyword: true,
            ..SpeechCue::default()
        },
    }
}

fn full(organism: &Organism) -> bool {
    let budgets = [
        organism.obligation_budget(),
        organism.discretionary_budget(),
    ];
    budgets
        .iter()
        .all(|budget| (budget.available - budget.capacity).abs() < f32::EPSILON)
}

// ---------------------------------------------------------------------------
// Quiet mode

#[test]
fn a_quiet_command_buys_no_thought_and_holds_back_everything_of_enton_s_own() {
    let mut organism = Organism::new(drives_profile()).unwrap();
    organism.step(&checklist(0, true));
    // "Enton, silêncio por uma hora": decided on the spot, for free.
    assert!(organism.step(&quiet(1_000, 3_601_000)).is_empty());
    assert_eq!(organism.quiet_until(), Some(Millis(3_601_000)));
    assert!(organism.quiet_as_of(Millis(1_000)));
    assert!(full(&organism));
    // It called Enton by name: the owner is home.
    assert_eq!(organism.owner_heard_at(), Some(Millis(1_000)));

    // Speech worth a thought, with the owner home, waits.
    assert_abstention(&organism.step(&speech(30_000, false)), Abstention::Quiet);
    // So does a ready drive, logged once.
    assert_abstention(&organism.step(&tick(3_000_000)), Abstention::Quiet);
    assert!(organism.step(&tick(3_001_000)).is_empty());

    // Called by name, Enton answers: the first thought bought since the command.
    assert_thought(
        &organism.step(&speech(3_100_000, true)),
        1,
        &Reason::Keyword,
    );
    organism.step(&reply(3_101_000, 1, "Oi."));
    // The drive may not ride that answer either: quiet holds back all of its own.
    for now in (3_102_000..3_111_000).step_by(1_000) {
        assert!(organism.step(&tick(now)).is_empty(), "{now} ms");
    }
    assert_eq!(organism.deferred(), None);

    // When the conversation ends (the verified window, 10 s after the reply), the wait is
    // logged again, once.
    assert_abstention(&organism.step(&tick(3_111_000)), Abstention::Quiet);
    // Quiet runs out on its own, and the drive thinks at the next tick.
    for now in (3_121_000..3_601_000).step_by(10_000) {
        assert!(organism.step(&tick(now)).is_empty(), "{now} ms");
    }
    assert!(!organism.quiet_as_of(Millis(3_601_000)));
    assert_thought(&organism.step(&tick(3_601_000)), 2, &curiosity());
}

#[test]
fn the_owner_releases_quiet_by_name() {
    let mut organism = Organism::new(drives_profile()).unwrap();
    organism.step(&checklist(0, true));
    organism.step(&quiet(1_000, 7_201_000));
    assert_abstention(&organism.step(&tick(3_000_000)), Abstention::Quiet);
    // "Enton, pode falar": a release, for free as well.
    assert!(organism.step(&quiet(3_010_000, 3_010_000)).is_empty());
    assert_eq!(organism.quiet_until(), None);
    assert_thought(&organism.step(&tick(3_011_000)), 1, &curiosity());
}

#[test]
fn the_latest_command_wins() {
    let mut organism = Organism::new(drives_profile()).unwrap();
    organism.step(&quiet(0, 3_600_000));
    // A shorter quiet asked later replaces the longer one.
    organism.step(&quiet(100_000, 200_000));
    assert_eq!(organism.quiet_until(), Some(Millis(200_000)));
    assert!(organism.quiet_as_of(Millis(199_999)));
    assert!(!organism.quiet_as_of(Millis(200_000)));
    // A release whose clock reads behind the last command still releases.
    organism.step(&quiet(150_000, 0));
    assert_eq!(organism.quiet_until(), None);
}

#[test]
fn a_quiet_command_ends_a_waiting_name_and_closes_the_windows() {
    let mut organism = Organism::new(lab_profile()).unwrap();
    // "Enton..." and then "Enton, fica quieto": the command was the rest of that turn.
    assert!(matches!(
        organism.step(&name_alone(1_000)).as_slice(),
        [Action::Attend { .. }]
    ));
    assert!(organism.step(&quiet(2_000, 3_602_000)).is_empty());
    assert!(!organism.is_attending());
    assert_eq!(organism.attention_until(), None);
    assert_eq!(organism.verified_attention_until(), None);
    // The wait does not time out into an answer...
    assert!(organism.step(&tick(6_000)).is_empty());
    // ...and what the owner says next, without the name, is not a follow-up.
    assert_abstention(&organism.step(&speech(7_000, false)), Abstention::Quiet);
    // The name still gets an answer.
    assert_thought(&organism.step(&speech(8_000, true)), 1, &Reason::Keyword);
}

#[test]
fn a_release_leaves_the_conversation_as_it_was() {
    let mut organism = Organism::new(lab_profile()).unwrap();
    assert_thought(&organism.step(&speech(1_000, true)), 1, &Reason::Keyword);
    organism.step(&reply(2_000, 1, "Oi!"));
    let windows = (
        organism.attention_until(),
        organism.verified_attention_until(),
    );
    organism.step(&quiet(3_000, 3_000));
    assert_eq!(
        (
            organism.attention_until(),
            organism.verified_attention_until()
        ),
        windows
    );
    // A follow-up in the window is still answered.
    assert_thought(&organism.step(&speech(4_000, false)), 2, &Reason::FollowUp);
}

// ---------------------------------------------------------------------------
// Quiet hours

#[test]
fn quiet_hours_hold_back_enton_s_own_thoughts_and_never_an_answer() {
    let mut organism = Organism::new(drives_profile()).unwrap();
    organism.step(&checklist(0, true));
    organism.step(&owner_nearby(1_000));
    assert!(organism.step(&quiet_hours(2_000, true)).is_empty());
    assert!(organism.in_quiet_hours());
    // Overheard speech worth a thought waits for the morning.
    assert_abstention(
        &organism.step(&speech(30_000, false)),
        Abstention::QuietHours,
    );
    // The owner calling by name is answered, and so is their follow-up.
    assert_thought(&organism.step(&speech(60_000, true)), 1, &Reason::Keyword);
    organism.step(&reply(61_000, 1, "Boa noite."));
    assert_thought(&organism.step(&speech(63_000, false)), 2, &Reason::FollowUp);
    organism.step(&reply(64_000, 2, "Durma bem."));
    // A ready drive waits too, logged once, whoever is home.
    assert_abstention(&organism.step(&tick(3_600_000)), Abstention::QuietHours);
    assert!(organism.step(&tick(3_601_000)).is_empty());
    // Morning: the band ends; the drive still needs somebody home.
    assert!(organism.step(&quiet_hours(3_700_000, false)).is_empty());
    assert_abstention(&organism.step(&tick(3_700_000)), Abstention::NobodyHome);
    organism.step(&owner_nearby(3_701_000));
    assert_thought(&organism.step(&tick(3_702_000)), 3, &curiosity());
}

#[test]
fn quiet_mode_is_named_before_quiet_hours_and_both_before_the_checklist() {
    let mut organism = Organism::new(drives_profile()).unwrap();
    organism.step(&quiet_hours(0, true));
    organism.step(&quiet(1_000, 7_200_000));
    assert_abstention(&organism.step(&tick(3_600_000)), Abstention::Quiet);
    organism.step(&quiet(3_601_000, 3_601_000));
    assert_abstention(&organism.step(&tick(3_602_000)), Abstention::QuietHours);
    organism.step(&quiet_hours(3_602_500, false));
    assert_abstention(&organism.step(&tick(3_603_000)), Abstention::NothingToCheck);
    organism.step(&checklist(3_603_500, true));
    assert_thought(&organism.step(&tick(3_604_000)), 1, &curiosity());
}

// ---------------------------------------------------------------------------
// Replay

#[test]
fn quiet_and_quiet_hours_replay_from_json_and_survive_a_snapshot() {
    let tape = [
        checklist(0, true),
        quiet(1_000, 3_601_000),
        speech(30_000, false),
        tick(3_000_000),
        speech(3_100_000, true),
        reply(3_101_000, 1, "Oi."),
        name_alone(3_200_000),
        quiet(3_201_000, 3_201_000),
        tick(3_205_000),
        quiet_hours(3_300_000, true),
        speech(3_300_500, false),
        tick(3_400_000),
        quiet_hours(3_500_000, false),
        tick(3_500_500),
    ];
    let mut live = Organism::new(drives_profile()).unwrap();
    let decided: Vec<_> = tape.iter().map(|event| live.step(event)).collect();
    let whys: Vec<Abstention> = decided
        .iter()
        .flatten()
        .filter_map(|action| match action {
            Action::Abstain { why, .. } => Some(*why),
            _ => None,
        })
        .collect();
    for expected in [Abstention::Quiet, Abstention::QuietHours] {
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

#[test]
fn a_snapshot_stored_before_quiet_and_deferrals_reads_back_without_them() {
    let organism = Organism::new(drives_profile()).unwrap();
    let mut stored = serde_json::to_value(&organism).unwrap();
    let fields = stored.as_object_mut().unwrap();
    for key in ["deferred", "quiet_until", "quiet_hours"] {
        assert!(fields.remove(key).is_some(), "{key}");
    }
    let restored: Organism = serde_json::from_value(stored).unwrap();
    assert_eq!(restored, organism);
}
