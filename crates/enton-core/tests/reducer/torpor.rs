use enton_core::{Abstention, Event, Millis, Organism, Profile, Reason, SpeechCue};

use super::support::{assert_abstention, assert_thought, body, sensitive_profile, speech};

#[test]
fn fever_blocks_speech_but_not_keywords_and_a_new_body_reading_clears_torpor() {
    let profile = Profile::t1_ref();
    let fever = profile.body.fever_c;
    let cooldown = profile.ignition.cooldown_ms;
    let mut organism = Organism::new(profile).unwrap();
    assert!(organism.step(&body(0, Some(fever), None)).is_empty());
    assert_abstention(&organism.step(&speech(0, false)), Abstention::Torpor);
    assert_thought(&organism.step(&speech(0, true)), 1, &Reason::Keyword);
    assert!(organism.step(&body(cooldown, None, None)).is_empty());
    assert_thought(&organism.step(&speech(cooldown, false)), 2, &Reason::Speech);
}

#[test]
fn critical_battery_enters_torpor_at_the_exact_boundary() {
    let profile = Profile::t1_ref();
    let battery = profile.body.lethargy_battery;
    let mut organism = Organism::new(profile).unwrap();
    organism.step(&body(0, None, Some(battery)));
    assert_abstention(&organism.step(&speech(0, false)), Abstention::Torpor);
    organism.step(&body(1, None, Some(battery + 0.01)));
    assert_thought(&organism.step(&speech(1, false)), 1, &Reason::Speech);
}

#[test]
fn rejection_priority_is_torpor_then_threshold_then_cooldown_then_energy() {
    let mut organism = Organism::new({
        let mut profile = Profile::t1_ref();
        profile.budgets.discretionary_budget_per_hour = 1.0;
        profile
    })
    .unwrap();
    assert_thought(&organism.step(&speech(0, false)), 1, &Reason::Speech);
    let weak = Event::Speech {
        now: Millis(1),
        cue: SpeechCue::default(),
    };
    organism.step(&body(1, Some(100.0), None));
    assert_abstention(&organism.step(&weak), Abstention::Torpor);
    organism.step(&body(1, None, None));
    assert_abstention(&organism.step(&weak), Abstention::BelowThreshold);
    assert_abstention(&organism.step(&speech(1, false)), Abstention::Cooldown);
    assert_abstention(
        &organism.step(&speech(10_000, false)),
        Abstention::OutOfEnergy,
    );
}

#[test]
fn a_ready_drive_records_torpor_without_consuming_ignition() {
    let mut organism = Organism::new(sensitive_profile()).unwrap();
    organism.step(&body(0, Some(100.0), None));
    assert_abstention(
        &organism.step(&Event::Tick {
            now: Millis(3_600_000),
        }),
        Abstention::Torpor,
    );
    organism.step(&body(3_600_001, None, None));
    assert_thought(
        &organism.step(&Event::Tick {
            now: Millis(3_601_000),
        }),
        1,
        &Reason::Drive("curiosity".to_owned()),
    );
}

#[test]
fn a_body_reading_that_is_not_a_number_is_unknown_live_and_in_replay() {
    // A sensor that reads "inf" is a broken sensor, not a fever: live and replayed
    // organisms must agree, and JSON can only carry it as unknown.
    let events = [
        body(1, Some(f32::INFINITY), Some(f32::NEG_INFINITY)),
        speech(2, false),
    ];
    let mut live = Organism::new(Profile::t1_ref()).unwrap();
    let mut replayed = live.clone();
    let live_actions: Vec<_> = events.iter().map(|event| live.step(event)).collect();
    let replayed_actions: Vec<_> = events
        .iter()
        .map(|event| {
            let stored = serde_json::to_string(&event.clone().canonical()).unwrap();
            replayed.step(&serde_json::from_str(&stored).unwrap())
        })
        .collect();
    assert_eq!(live_actions, replayed_actions);
    assert_eq!(live, replayed);
    assert_thought(&live_actions[1], 1, &Reason::Speech);
}
