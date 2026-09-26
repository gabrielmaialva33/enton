use enton_core::{Abstention, Action, Event, Millis, Organism, Profile, Reason, SpeechCue};

use super::support::{assert_abstention, assert_thought, sensitive_profile, speech};

#[test]
fn directed_requests_served_when_discretionary_zeroed() {
    let mut organism = Organism::new(Profile {
        obligation_budget_per_hour: 10.0,
        discretionary_budget_per_hour: 0.0,
        cooldown_ms: 0,
        ..Profile::t1_ref()
    })
    .unwrap();
    // Directed turns: Keyword and FollowUp succeed despite discretionary being 0
    assert_thought(&organism.step(&speech(0, true)), 1, &Reason::Keyword);
    assert_thought(&organism.step(&speech(1, false)), 2, &Reason::FollowUp);

    // Close attention window
    organism.step(&Event::Tick { now: Millis(6_000) });

    // Undirected optional speech fails immediately because discretionary is 0
    assert_abstention(
        &organism.step(&speech(6_001, false)),
        Abstention::OutOfEnergy,
    );
}

#[test]
fn optional_thoughts_never_spend_obligation() {
    let mut organism = Organism::new(Profile {
        obligation_budget_per_hour: 2.0,
        discretionary_budget_per_hour: 10.0,
        cooldown_ms: 0,
        ..Profile::t1_ref()
    })
    .unwrap();
    // Discretionary speech spends discretionary budget
    assert_thought(&organism.step(&speech(0, false)), 1, &Reason::Speech);
    assert_thought(&organism.step(&speech(1, false)), 2, &Reason::Speech);

    // Obligation budget is completely untouched (still has 2.0)
    assert_thought(&organism.step(&speech(2, true)), 3, &Reason::Keyword);
    assert_thought(&organism.step(&speech(3, true)), 4, &Reason::Keyword);
    // Now obligation is exhausted
    assert_abstention(&organism.step(&speech(4, true)), Abstention::OutOfEnergy);
}

#[test]
fn only_new_tick_time_refills_the_budget() {
    let mut organism = Organism::new(Profile {
        obligation_budget_per_hour: 2.0,
        discretionary_budget_per_hour: 0.0,
        cooldown_ms: 0,
        ..Profile::t1_ref()
    })
    .unwrap();
    organism.step(&speech(0, true));
    organism.step(&speech(1, true));
    assert_abstention(&organism.step(&speech(2, true)), Abstention::OutOfEnergy);
    organism.step(&Event::Tick {
        now: Millis(1_800_000),
    });
    assert_abstention(
        &organism.step(&speech(1_800_000, false)),
        Abstention::OutOfEnergy,
    );
    assert_thought(
        &organism.step(&speech(1_800_000, true)),
        3,
        &Reason::Keyword,
    );
    let snapshot = organism.clone();
    assert!(
        organism
            .step(&Event::Tick {
                now: Millis(900_000)
            })
            .is_empty()
    );
    assert!(
        organism
            .step(&Event::Tick {
                now: Millis(1_800_000)
            })
            .is_empty()
    );
    assert_eq!(organism, snapshot);
    organism.step(&Event::Tick {
        now: Millis(3_600_000),
    });
    assert_thought(
        &organism.step(&speech(3_600_000, true)),
        4,
        &Reason::Keyword,
    );
}

#[test]
fn a_ready_drive_cannot_spend_a_keyword_only_budget() {
    let mut organism = Organism::new(Profile {
        discretionary_budget_per_hour: 0.0,
        ..sensitive_profile()
    })
    .unwrap();
    assert_abstention(
        &organism.step(&Event::Tick {
            now: Millis(3_600_000),
        }),
        Abstention::OutOfEnergy,
    );
    assert_thought(
        &organism.step(&speech(3_600_001, true)),
        1,
        &Reason::Keyword,
    );
}

#[test]
fn a7_timeout_spends_after_elapsed_refill_without_intermediate_ticks() {
    let profile = Profile {
        obligation_budget_per_hour: 1.0,
        think_cost: 1.0,
        ..Profile::t1_ref()
    };
    let mut organism = Organism::new(profile).unwrap();

    // Spend initial 1.0 obligation energy at time 0
    let actions = organism.step(&Event::Speech {
        now: Millis(0),
        cue: SpeechCue {
            keyword: true,
            duration_ms: 1_200,
            ..SpeechCue::default()
        },
    });
    assert_thought(&actions, 1, &Reason::Keyword);
    assert!(organism.obligation_budget().available < 0.001);

    // Advance to 3,599,000 ms: budget has refilled almost 1.0 (3599/3600 = 0.9997), but < 1.0
    organism.step(&Event::Tick {
        now: Millis(3_599_000),
    });
    assert!(organism.obligation_budget().available < 1.0);

    // Attend keyword-only turn at 3,599,000 ms (window until 3,604,000 ms)
    let actions = organism.step(&Event::Speech {
        now: Millis(3_599_000),
        cue: SpeechCue {
            keyword: true,
            duration_ms: 300,
            ..SpeechCue::default()
        },
    });
    assert_eq!(
        actions,
        vec![Action::Attend {
            until: Millis(3_604_000)
        }]
    );

    // Advance directly to deadline tick at 3,604,000 ms without intermediate ticks.
    // The 5,000 ms elapsed between 3,599,000 and 3,604,000 ms crosses the 1.0 energy boundary.
    // Refill must apply BEFORE evaluating the timeout so the thought is paid and fires.
    let timeout_actions = organism.step(&Event::Tick {
        now: Millis(3_604_000),
    });
    assert_thought(&timeout_actions, 2, &Reason::Keyword);
    assert!(!organism.is_attending());
}

#[test]
fn a7_timeout_spends_after_elapsed_refill_with_intermediate_ticks() {
    let profile = Profile {
        obligation_budget_per_hour: 1.0,
        think_cost: 1.0,
        ..Profile::t1_ref()
    };
    let mut organism = Organism::new(profile).unwrap();

    // Spend initial 1.0 obligation energy at time 0
    let actions = organism.step(&Event::Speech {
        now: Millis(0),
        cue: SpeechCue {
            keyword: true,
            duration_ms: 1_200,
            ..SpeechCue::default()
        },
    });
    assert_thought(&actions, 1, &Reason::Keyword);

    // Advance to 3,599,000 ms
    organism.step(&Event::Tick {
        now: Millis(3_599_000),
    });

    // Attend keyword-only turn at 3,599,000 ms
    organism.step(&Event::Speech {
        now: Millis(3_599_000),
        cue: SpeechCue {
            keyword: true,
            duration_ms: 300,
            ..SpeechCue::default()
        },
    });

    // Intermediate tick at 3,600,000 ms
    organism.step(&Event::Tick {
        now: Millis(3_600_000),
    });

    // Deadline tick at 3,604,000 ms
    let timeout_actions = organism.step(&Event::Tick {
        now: Millis(3_604_000),
    });
    assert_thought(&timeout_actions, 2, &Reason::Keyword);
}
