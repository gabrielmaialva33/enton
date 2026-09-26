//! Deterministic scaffold checks, not the full audio-based RFC E1 experiment.

use enton_core::{
    Abstention, Action, BodySignals, Event, Millis, Organism, PlaybackStatus, Profile, Reason,
    SpeechCue, ThoughtId, UtteranceId, contains_keyword_word,
};

fn speech(now: u64, keyword: bool) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 1.0,
            duration_ms: 1_500,
            vad_confidence: 1.0,
            keyword,
            speaker_sim: None,
        },
    }
}

fn body(now: u64, temperature_c: Option<f32>, battery: Option<f32>) -> Event {
    Event::Body {
        now: Millis(now),
        signals: BodySignals {
            temperature_c,
            battery,
            cpu_load: 0.0,
        },
    }
}

fn assert_thought(actions: &[Action], expected: u64, expected_reason: &Reason) {
    assert!(
        matches!(actions, [Action::Think { thought, reason, .. }]
            if *thought == ThoughtId(expected) && reason == expected_reason),
        "expected thought {expected} for {expected_reason:?}, got {actions:?}"
    );
}

fn assert_abstention(actions: &[Action], expected: Abstention) {
    assert!(
        matches!(actions, [Action::Abstain { why, .. }] if *why == expected),
        "expected {expected:?}, got {actions:?}"
    );
}

#[test]
fn an_hour_of_ticks_never_ignites_or_emits_idle_abstentions() {
    for profile in [Profile::t1_ref(), Profile::desktop()] {
        let mut organism = Organism::new(profile).unwrap();
        for second in 0..=3_600 {
            assert!(
                organism
                    .step(&Event::Tick {
                        now: Millis(second * 1_000)
                    })
                    .is_empty(),
                "unexpected idle action at second {second}"
            );
        }
    }
}

#[test]
fn keyword_ignites_even_without_other_speech_evidence() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    // Full keyword cue (>= keyword_only_ms) ignites directly
    let actions = organism.step(&Event::Speech {
        now: Millis(0),
        cue: SpeechCue {
            keyword: true,
            duration_ms: 1_200,
            ..SpeechCue::default()
        },
    });
    assert_thought(&actions, 1, &Reason::Keyword);
}

#[test]
fn weak_speech_records_below_threshold() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    assert_abstention(
        &organism.step(&Event::Speech {
            now: Millis(0),
            cue: SpeechCue {
                energy: 0.1,
                duration_ms: 100,
                vad_confidence: 0.2,
                keyword: false,
                speaker_sim: None,
            },
        }),
        Abstention::BelowThreshold,
    );
}

#[test]
fn speech_observes_cooldown_but_keywords_bypass_it() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    assert_thought(&organism.step(&speech(0, false)), 1, &Reason::Speech);
    assert_abstention(&organism.step(&speech(1_000, false)), Abstention::Cooldown);
    // Keyword bypasses cooldown and opens attention window
    assert_thought(&organism.step(&speech(1_001, true)), 2, &Reason::Keyword);
    // Inside attention window, speech is treated as FollowUp
    assert_thought(&organism.step(&speech(1_002, false)), 3, &Reason::FollowUp);
    // Close attention window via tick after attention_ms (5_000 ms)
    organism.step(&Event::Tick { now: Millis(6_500) });
    // Outside attention window, but still within cooldown from thought 3 (1_002 + 10_000 = 11_002)
    assert_abstention(&organism.step(&speech(7_000, false)), Abstention::Cooldown);
    // Advance time past cooldown and let habituation decay
    organism.step(&Event::Tick {
        now: Millis(30_000),
    });
    assert_thought(&organism.step(&speech(30_001, false)), 4, &Reason::Speech);
}

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
fn fever_blocks_speech_but_not_keywords_and_a_new_body_reading_clears_torpor() {
    let profile = Profile::t1_ref();
    let fever = profile.fever_c;
    let cooldown = profile.cooldown_ms;
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
    let battery = profile.lethargy_battery;
    let mut organism = Organism::new(profile).unwrap();
    organism.step(&body(0, None, Some(battery)));
    assert_abstention(&organism.step(&speech(0, false)), Abstention::Torpor);
    organism.step(&body(1, None, Some(battery + 0.01)));
    assert_thought(&organism.step(&speech(1, false)), 1, &Reason::Speech);
}

#[test]
fn identical_tapes_reproduce_actions_and_final_state() {
    let tape = [
        Event::Tick {
            now: Millis(60_000),
        },
        speech(60_001, false),
        speech(60_002, false),
        speech(60_003, true),
        Event::CortexReply {
            now: Millis(60_004),
            thought: ThoughtId(2),
            text: "I am listening.".to_owned(),
        },
        body(60_005, Some(100.0), None),
        speech(60_006, false),
        Event::Tick {
            now: Millis(120_000),
        },
        body(120_001, None, Some(0.9)),
        speech(120_002, false),
    ];
    let mut first = Organism::new(Profile::t1_ref()).unwrap();
    let mut second = Organism::new(Profile::t1_ref()).unwrap();
    let first_actions: Vec<_> = tape.iter().map(|event| first.step(event)).collect();
    let second_actions: Vec<_> = tape.iter().map(|event| second.step(event)).collect();
    assert_eq!(first_actions, second_actions);
    assert_eq!(first, second);
    assert_eq!(
        first_actions[4],
        vec![Action::Speak {
            text: "I am listening.".to_owned()
        }]
    );
}

#[test]
fn rejection_priority_is_torpor_then_threshold_then_cooldown_then_energy() {
    let mut organism = Organism::new(Profile {
        discretionary_budget_per_hour: 1.0,
        ..Profile::t1_ref()
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

fn sensitive_profile() -> Profile {
    Profile {
        threshold: 0.01,
        hysteresis: 0.002,
        ema_alpha: 1.0,
        cooldown_ms: 0,
        ..Profile::t1_ref()
    }
}

#[test]
fn drive_ignition_reports_the_strongest_contributor_and_does_not_repeat() {
    let mut organism = Organism::new(sensitive_profile()).unwrap();
    assert_thought(
        &organism.step(&Event::Tick {
            now: Millis(3_600_000),
        }),
        1,
        &Reason::Drive("curiosity".to_owned()),
    );
    assert!(
        organism
            .step(&Event::Tick {
                now: Millis(3_601_000)
            })
            .is_empty()
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
fn speech_salience_uses_the_specified_weights_and_caps_duration() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    let actions = organism.step(&Event::Speech {
        now: Millis(0),
        cue: SpeechCue {
            energy: 0.8,
            duration_ms: 3_000,
            vad_confidence: 1.0,
            keyword: true,
            speaker_sim: None,
        },
    });
    let [Action::Think { salience, .. }] = actions.as_slice() else {
        panic!("expected a thought, got {actions:?}");
    };
    // 0.60 * 1.0 (vad) + 0.25 * 0.8 (energy) + 0.15 * 1.0 (capped duration) + 1.0 (keyword) = 1.95
    assert!((salience - 1.95).abs() < 0.000_001);
}

#[test]
fn malformed_speech_cannot_introduce_nonfinite_salience() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    let actions = organism.step(&Event::Speech {
        now: Millis(0),
        cue: SpeechCue {
            energy: f32::NAN,
            vad_confidence: f32::INFINITY,
            ..SpeechCue::default()
        },
    });
    assert_abstention(&actions, Abstention::BelowThreshold);
    assert!(matches!(&actions[0], Action::Abstain { salience, .. } if salience.is_finite()));
}

#[test]
fn salience_short_clear_direct_speech_passes_and_low_vad_never_passes() {
    let profile = Profile::t1_ref();
    let threshold = profile.threshold;

    // 1. Short clear direct speech: energy >= 0.8, VAD >= 0.8, 250 ms passes t1-ref threshold (0.7)
    let mut organism = Organism::new(profile).unwrap();
    let actions = organism.step(&Event::Speech {
        now: Millis(0),
        cue: SpeechCue {
            energy: 0.8,
            duration_ms: 250,
            vad_confidence: 0.8,
            keyword: false,
            speaker_sim: None,
        },
    });
    let [
        Action::Think {
            salience, reason, ..
        },
    ] = actions.as_slice()
    else {
        panic!("expected direct speech to ignite, got {actions:?}");
    };
    assert_eq!(*reason, Reason::Speech);
    assert!(
        *salience >= threshold,
        "salience {salience} must be >= threshold {threshold}"
    );

    // 2. Low-VAD sound (VAD <= 0.4) NEVER passes threshold at ANY energy or duration
    for &energy in &[0.0f32, 0.2, 0.5, 0.8, 1.0] {
        for &vad in &[0.0f32, 0.1, 0.2, 0.3, 0.4] {
            for &dur in &[50u32, 100, 250, 500, 1000, 5000, 10_000] {
                let mut fresh_organism = Organism::new(Profile::t1_ref()).unwrap();
                let actions = fresh_organism.step(&Event::Speech {
                    now: Millis(0),
                    cue: SpeechCue {
                        energy,
                        duration_ms: dur,
                        vad_confidence: vad,
                        keyword: false,
                        speaker_sim: None,
                    },
                });
                assert!(
                    matches!(&actions[0], Action::Abstain { why: Abstention::BelowThreshold, salience, .. } if *salience < threshold),
                    "VAD <= 0.4 must never reach threshold (energy={energy}, vad={vad}, dur={dur}), got {actions:?}"
                );
            }
        }
    }
}

#[test]
fn keyword_only_turn_attends_and_merges_continuation() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();

    // 1. Keyword cue shorter than keyword_only_ms (900 ms) emits Action::Attend { until }
    let actions1 = organism.step(&Event::Speech {
        now: Millis(100),
        cue: SpeechCue {
            energy: 0.9,
            duration_ms: 400,
            vad_confidence: 0.95,
            keyword: true,
            speaker_sim: None,
        },
    });
    assert_eq!(
        actions1,
        vec![Action::Attend {
            until: Millis(5_100)
        }]
    );
    assert!(organism.is_attending());
    assert_eq!(organism.attention_until(), Some(Millis(5_100)));

    // 2. Pause of 300 ms (Tick)
    let tick_actions = organism.step(&Event::Tick { now: Millis(400) });
    assert!(tick_actions.is_empty());
    assert!(organism.is_attending());

    // 3. Continuation speech arrives at 400 ms (within window, no keyword)
    let actions2 = organism.step(&Event::Speech {
        now: Millis(400),
        cue: SpeechCue {
            energy: 0.85,
            duration_ms: 800,
            vad_confidence: 0.9,
            keyword: false,
            speaker_sim: None,
        },
    });

    // Produces a single Think with Reason::Keyword covering both segments
    let [
        Action::Think {
            thought, reason, ..
        },
    ] = actions2.as_slice()
    else {
        panic!("expected single Think for continuation, got {actions2:?}");
    };
    assert_eq!(*thought, ThoughtId(1));
    assert_eq!(*reason, Reason::Keyword);
    assert!(!organism.is_attending());
    // Attention window extended after served continuation
    assert_eq!(organism.attention_until(), Some(Millis(5_400)));
}

#[test]
fn keyword_only_turn_thinks_on_timeout_when_no_continuation_arrives() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();

    // Keyword cue shorter than 900 ms emits Attend { until: 5000 }
    let actions1 = organism.step(&Event::Speech {
        now: Millis(0),
        cue: SpeechCue {
            energy: 0.9,
            duration_ms: 350,
            vad_confidence: 0.9,
            keyword: true,
            speaker_sim: None,
        },
    });
    assert_eq!(
        actions1,
        vec![Action::Attend {
            until: Millis(5_000)
        }]
    );

    // Ticks before timeout do not ignite
    assert!(
        organism
            .step(&Event::Tick { now: Millis(2_000) })
            .is_empty()
    );
    assert!(organism.is_attending());

    // Tick at deadline (5000 ms) fires Think with Reason::Keyword ("Enton?" alone still answered)
    let actions = organism.step(&Event::Tick { now: Millis(5_000) });
    let [
        Action::Think {
            thought, reason, ..
        },
    ] = actions.as_slice()
    else {
        panic!("expected Think on timeout, got {actions:?}");
    };
    assert_eq!(*thought, ThoughtId(1));
    assert_eq!(*reason, Reason::Keyword);
    assert!(!organism.is_attending());
}

#[test]
fn attention_window_follow_up_bypasses_cooldown_and_pays_normal_energy() {
    let mut organism = Organism::new(Profile {
        obligation_budget_per_hour: 2.0,
        ..Profile::t1_ref()
    })
    .unwrap();

    // 1. Initial direct request with keyword opens attention window (5000 ms)
    let actions1 = organism.step(&Event::Speech {
        now: Millis(0),
        cue: SpeechCue {
            energy: 0.9,
            duration_ms: 1_200,
            vad_confidence: 0.95,
            keyword: true,
            speaker_sim: None,
        },
    });
    assert_thought(&actions1, 1, &Reason::Keyword);
    assert_eq!(organism.attention_until(), Some(Millis(5_000)));

    // 2. Follow-up speech inside window (1500 ms) without keyword
    let actions2 = organism.step(&Event::Speech {
        now: Millis(1_500),
        cue: SpeechCue {
            energy: 0.7,
            duration_ms: 500,
            vad_confidence: 0.6, // meets minimal VAD (0.5)
            keyword: false,
            speaker_sim: None,
        },
    });
    // Bypasses cooldown! Becomes Reason::FollowUp
    assert_thought(&actions2, 2, &Reason::FollowUp);
    // Window extended by served follow-up
    assert_eq!(organism.attention_until(), Some(Millis(6_500)));

    // 3. Normal energy is now exhausted (only keyword reserve remaining).
    // Another follow-up inside window cannot spend reserve and yields OutOfEnergy!
    let actions3 = organism.step(&Event::Speech {
        now: Millis(2_000),
        cue: SpeechCue {
            energy: 0.7,
            duration_ms: 500,
            vad_confidence: 0.6,
            keyword: false,
            speaker_sim: None,
        },
    });
    assert_abstention(&actions3, Abstention::OutOfEnergy);
}

#[test]
fn habituation_suppresses_repeated_similar_non_addressed_speech() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();

    // First burst of non-addressed loud speech (TV)
    let actions1 = organism.step(&Event::Speech {
        now: Millis(0),
        cue: SpeechCue {
            energy: 0.85,
            duration_ms: 1_200,
            vad_confidence: 0.9,
            keyword: false,
            speaker_sim: None,
        },
    });
    assert_thought(&actions1, 1, &Reason::Speech);
    assert!(organism.habituation() >= 0.0);

    // Repeated bursts with identical features arrive every 2 seconds
    let mut habituated = false;
    for i in 1..=5 {
        let actions = organism.step(&Event::Speech {
            now: Millis(i * 2_000),
            cue: SpeechCue {
                energy: 0.85,
                duration_ms: 1_200,
                vad_confidence: 0.9,
                keyword: false,
                speaker_sim: None,
            },
        });
        if actions.iter().any(|a| {
            matches!(
                a,
                Action::Abstain {
                    why: Abstention::Habituation,
                    ..
                }
            )
        }) {
            habituated = true;
            break;
        }
    }
    assert!(
        habituated,
        "repeated TV bursts must yield Abstention::Habituation"
    );
    assert!(organism.habituation() > 0.0);

    // Addressed speech (keyword) is NEVER habituated!
    let keyword_actions = organism.step(&Event::Speech {
        now: Millis(12_000),
        cue: SpeechCue {
            energy: 0.85,
            duration_ms: 1_200,
            vad_confidence: 0.9,
            keyword: true,
            speaker_sim: None,
        },
    });
    assert_thought(&keyword_actions, 2, &Reason::Keyword);

    // Habituation decays on Tick
    let hab_before = organism.habituation();
    organism.step(&Event::Tick {
        now: Millis(30_000),
    });
    assert!(
        organism.habituation() < hab_before,
        "habituation must decay over time"
    );
}

#[test]
fn novelty_adds_salience_on_prediction_error() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();

    // Initial cue sets baseline expectation
    let _ = organism.step(&Event::Speech {
        now: Millis(0),
        cue: SpeechCue {
            energy: 0.5,
            duration_ms: 500,
            vad_confidence: 0.5,
            keyword: false,
            speaker_sim: None,
        },
    });

    // Sudden different cue (high energy, high VAD) creates prediction error
    let actions = organism.step(&Event::Speech {
        now: Millis(15_000),
        cue: SpeechCue {
            energy: 0.9,
            duration_ms: 1_000,
            vad_confidence: 0.9,
            keyword: false,
            speaker_sim: None,
        },
    });
    assert_thought(&actions, 1, &Reason::Speech);
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

#[test]
fn a8_slow_reply_anchors_attention_independent_of_tick_interleaving() {
    // 1. With closing tick at 6,000 ms:
    let mut org_with_tick = Organism::new(Profile::t1_ref()).unwrap();
    let act1 = org_with_tick.step(&speech(0, true));
    assert_thought(&act1, 1, &Reason::Keyword);
    assert_eq!(org_with_tick.attention_until(), Some(Millis(5_000)));

    // Closing tick at 6,000 ms closes the initial attention window
    org_with_tick.step(&Event::Tick { now: Millis(6_000) });
    assert_eq!(org_with_tick.attention_until(), None);

    // Slow reply arrives at 6,001 ms for thought 1
    let reply_actions = org_with_tick.step(&Event::CortexReply {
        now: Millis(6_001),
        thought: ThoughtId(1),
        text: "Resposta do Enton".to_owned(),
    });
    assert_eq!(
        reply_actions,
        vec![Action::Speak {
            text: "Resposta do Enton".to_owned()
        }]
    );
    // Attention is anchored to completion of reply: 6,001 + 5,000 = 11,001 ms
    assert_eq!(org_with_tick.attention_until(), Some(Millis(11_001)));

    // Follow-up inside this window is accepted
    let follow_up_cue = SpeechCue {
        energy: 0.8,
        duration_ms: 600,
        vad_confidence: 0.9,
        keyword: false,
        speaker_sim: None,
    };
    let follow_up = org_with_tick.step(&Event::Speech {
        now: Millis(7_000),
        cue: follow_up_cue,
    });
    assert_thought(&follow_up, 2, &Reason::FollowUp);
    assert_eq!(org_with_tick.attention_until(), Some(Millis(12_000)));

    // 2. Run WITHOUT closing tick between 5,000 and 6,001 ms:
    let mut org_without_tick = Organism::new(Profile::t1_ref()).unwrap();
    let _ = org_without_tick.step(&speech(0, true));
    let _ = org_without_tick.step(&Event::CortexReply {
        now: Millis(6_001),
        thought: ThoughtId(1),
        text: "Resposta do Enton".to_owned(),
    });
    // Immediately after reply, both have identical attention window (11,001 ms)
    assert_eq!(org_without_tick.attention_until(), Some(Millis(11_001)));

    let follow_up_no_tick = org_without_tick.step(&Event::Speech {
        now: Millis(7_000),
        cue: follow_up_cue,
    });
    assert_thought(&follow_up_no_tick, 2, &Reason::FollowUp);
    assert_eq!(org_without_tick.attention_until(), Some(Millis(12_000)));

    // Both produce the exact same attention window regardless of tick interleaving!
    assert_eq!(
        org_with_tick.attention_until(),
        org_without_tick.attention_until()
    );
}

#[test]
fn a8_stale_and_other_replies_never_reopen_attention() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();

    // User speaks keyword -> ThoughtId(1)
    let _ = organism.step(&speech(0, true));
    // Reply for ThoughtId(1) at 1,000 ms extends window to 6,000 ms
    let _ = organism.step(&Event::CortexReply {
        now: Millis(1_000),
        thought: ThoughtId(1),
        text: "Olá".to_owned(),
    });
    assert_eq!(organism.attention_until(), Some(Millis(6_000)));

    // Ticks advance past window to 10,000 ms
    organism.step(&Event::Tick {
        now: Millis(10_000),
    });
    assert_eq!(organism.attention_until(), None);

    // Stale/duplicate reply for ThoughtId(1) arrives at 10_001 ms
    organism.step(&Event::CortexReply {
        now: Millis(10_001),
        thought: ThoughtId(1),
        text: "Stale reply".to_owned(),
    });
    // Stale reply must NOT reopen attention!
    assert_eq!(organism.attention_until(), None);

    // Reply for another/unknown ThoughtId(999) must NOT open attention!
    organism.step(&Event::CortexReply {
        now: Millis(10_002),
        thought: ThoughtId(999),
        text: "Other reply".to_owned(),
    });
    assert_eq!(organism.attention_until(), None);
}

#[test]
fn a8_reordered_backward_reply_timestamp_does_not_shorten_window() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();

    // Keyword at 10,000 ms opens window until 15,000 ms
    let _ = organism.step(&speech(10_000, true));
    assert_eq!(organism.attention_until(), Some(Millis(15_000)));

    // Reply arrives with a backward/reordered timestamp earlier than the current window (e.g. 2,000 ms)
    let _ = organism.step(&Event::CortexReply {
        now: Millis(2_000),
        thought: ThoughtId(1),
        text: "Resposta reordenada".to_owned(),
    });
    // Must NOT shorten window to 7,000 ms; must remain 15,000 ms
    assert_eq!(organism.attention_until(), Some(Millis(15_000)));
}

#[test]
fn a9_marginal_cue_fate_changes_with_novelty() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();

    // Threshold is 0.70.
    // Marginal cue: vad=0.8, energy=0.6, dur=333ms (dur_norm=0.333).
    // base_salience = 0.60 * 0.8 + 0.25 * 0.6 + 0.15 * 0.333 = 0.48 + 0.15 + 0.05 = 0.68 (< 0.70).
    let marginal_cue = SpeechCue {
        energy: 0.6,
        duration_ms: 333,
        vad_confidence: 0.8,
        keyword: false,
        speaker_sim: None,
    };

    // 1. Initial quiet background sound sets running expectation (energy=0.05, vad=0.05, dur=100ms)
    let _ = organism.step(&Event::Speech {
        now: Millis(0),
        cue: SpeechCue {
            energy: 0.05,
            duration_ms: 100,
            vad_confidence: 0.05,
            keyword: false,
            speaker_sim: None,
        },
    });

    // 2. Marginal cue arrives at 15_000 ms (past cooldown).
    // Large prediction error against background -> novelty bonus (~0.047) pushes salience:
    // 0.68 + 0.047 = 0.727 >= 0.70 threshold.
    let novel_actions = organism.step(&Event::Speech {
        now: Millis(15_000),
        cue: marginal_cue,
    });
    assert_thought(&novel_actions, 1, &Reason::Speech);

    // 3. Tick to 30,000 ms (cooldown expires)
    organism.step(&Event::Tick {
        now: Millis(30_000),
    });

    // Repeated identical cues make running expectation match marginal_cue exactly
    for i in 1..=4 {
        organism.step(&Event::Speech {
            now: Millis(30_000 + i * 200),
            cue: marginal_cue,
        });
    }

    // Tick to 45,000 ms (cooldown expires)
    organism.step(&Event::Tick {
        now: Millis(45_000),
    });

    // 4. Same marginal cue arrives when familiar (prediction error = 0, novelty = 0).
    // Base salience (0.68) is below threshold (0.70) -> must ABSTAIN!
    let familiar_actions = organism.step(&Event::Speech {
        now: Millis(45_000),
        cue: marginal_cue,
    });
    assert!(
        matches!(
            familiar_actions.as_slice(),
            [Action::Abstain {
                why: Abstention::BelowThreshold | Abstention::Habituation,
                ..
            }]
        ),
        "familiar marginal cue must abstain, got {familiar_actions:?}"
    );
}

#[test]
fn a9_silence_tv_to_novel_speech_resets_habituation() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();

    // 10 low-level TV bursts build up habituation
    for i in 0..10 {
        organism.step(&Event::Speech {
            now: Millis(i * 1_000),
            cue: SpeechCue {
                energy: 0.25,
                duration_ms: 250,
                vad_confidence: 0.25,
                keyword: false,
                speaker_sim: None,
            },
        });
    }
    assert!(
        organism.habituation() > 0.5,
        "habituation should be built up from repeated TV cues"
    );

    // Tick after 15 seconds
    organism.step(&Event::Tick {
        now: Millis(25_000),
    });

    // Sudden novel speech cue with high energy and VAD
    let novel_actions = organism.step(&Event::Speech {
        now: Millis(25_001),
        cue: SpeechCue {
            energy: 0.95,
            duration_ms: 900,
            vad_confidence: 0.95,
            keyword: false,
            speaker_sim: None,
        },
    });

    // Maximal prediction error resets habituation and ignites (must NOT be rejected as Habituation!)
    assert_thought(&novel_actions, 1, &Reason::Speech);
    assert!(organism.habituation().abs() < f32::EPSILON);
}

#[test]
fn a9_tunables_in_profile_govern_similarity_and_expectation() {
    let mut profile = Profile::t1_ref();
    assert!((profile.similarity_cutoff - 0.4).abs() < f32::EPSILON);
    assert!((profile.expectation_coefficient - 0.25).abs() < f32::EPSILON);

    // Configure a stricter similarity cutoff
    profile.similarity_cutoff = 0.8;
    profile.expectation_coefficient = 0.50;
    let mut organism = Organism::new(profile).unwrap();

    // First cue sets expectation
    organism.step(&Event::Speech {
        now: Millis(0),
        cue: SpeechCue {
            energy: 0.5,
            duration_ms: 500,
            vad_confidence: 0.5,
            keyword: false,
            speaker_sim: None,
        },
    });

    // Moderately different cue: similarity ~ 0.70.
    // With default cutoff 0.4, this would increase habituation.
    // With cutoff 0.8, similarity (0.70) <= cutoff (0.8), so it resets habituation!
    organism.step(&Event::Speech {
        now: Millis(1_000),
        cue: SpeechCue {
            energy: 0.8,
            duration_ms: 500,
            vad_confidence: 0.8,
            keyword: false,
            speaker_sim: None,
        },
    });
    assert!(organism.habituation().abs() < f32::EPSILON);
}

#[test]
fn self_echo_during_playback_is_suppressed() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    organism.step(&Event::PlaybackStarted {
        now: Millis(1000),
        utterance: UtteranceId(1),
    });
    assert!(organism.is_speaking());

    let initial_energy = organism.obligation_budget().available;

    let actions = organism.step(&Event::Speech {
        now: Millis(1200),
        cue: SpeechCue {
            energy: 0.70,
            duration_ms: 500,
            vad_confidence: 0.85,
            keyword: false,
            speaker_sim: None,
        },
    });

    assert_eq!(actions.len(), 1);
    assert_abstention(&actions, Abstention::SelfEcho);
    assert!((organism.obligation_budget().available - initial_energy).abs() < f32::EPSILON);
    assert!(organism.is_speaking());
}

#[test]
fn self_echo_during_hangover_is_suppressed() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    let init_actions = organism.step(&Event::Speech {
        now: Millis(500),
        cue: SpeechCue {
            energy: 0.85,
            duration_ms: 1000,
            vad_confidence: 0.95,
            keyword: true,
            speaker_sim: None,
        },
    });
    assert_thought(&init_actions, 1, &Reason::Keyword);

    organism.step(&Event::PlaybackStarted {
        now: Millis(1000),
        utterance: UtteranceId(1),
    });
    organism.step(&Event::PlaybackFinished {
        now: Millis(2000),
        utterance: UtteranceId(1),
    });
    assert!(organism.is_hangover());

    let actions = organism.step(&Event::Speech {
        now: Millis(2100),
        cue: SpeechCue {
            energy: 0.70,
            duration_ms: 300,
            vad_confidence: 0.80,
            keyword: false,
            speaker_sim: None,
        },
    });

    assert_eq!(actions.len(), 1);
    assert_abstention(&actions, Abstention::SelfEcho);

    let actions_after = organism.step(&Event::Speech {
        now: Millis(2300),
        cue: SpeechCue {
            energy: 0.70,
            duration_ms: 600,
            vad_confidence: 0.85,
            keyword: false,
            speaker_sim: None,
        },
    });
    assert_thought(&actions_after, 2, &Reason::FollowUp);
}

#[test]
fn predicted_keyword_requires_full_barge_in_margin() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    let reply_actions = organism.step(&Event::CortexReply {
        now: Millis(1000),
        thought: ThoughtId(1),
        text: "Olá! Eu sou o Enton.".to_string(),
    });
    assert!(organism.self_speech_has_keyword());
    assert_eq!(
        reply_actions,
        vec![Action::Speak {
            text: "Olá! Eu sou o Enton.".to_string()
        }]
    );

    organism.step(&Event::PlaybackStarted {
        now: Millis(1200),
        utterance: UtteranceId(1),
    });

    let actions = organism.step(&Event::Speech {
        now: Millis(1400),
        cue: SpeechCue {
            energy: 0.85,
            duration_ms: 400,
            vad_confidence: 0.90,
            keyword: true,
            speaker_sim: None,
        },
    });

    assert_eq!(actions.len(), 1);
    assert_abstention(&actions, Abstention::SelfEcho);
    assert!(organism.is_speaking());
}

#[test]
fn unpredicted_keyword_rejects_sub_echo_false_positives() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    organism.step(&Event::CortexReply {
        now: Millis(1000),
        thought: ThoughtId(1),
        text: "Tudo bem, vou ajustar a temperatura.".to_string(),
    });
    assert!(!organism.self_speech_has_keyword());

    organism.step(&Event::PlaybackStarted {
        now: Millis(1200),
        utterance: UtteranceId(1),
    });

    let actions = organism.step(&Event::Speech {
        now: Millis(1400),
        cue: SpeechCue {
            energy: 0.70,
            duration_ms: 300,
            vad_confidence: 0.85,
            keyword: true,
            speaker_sim: None,
        },
    });

    assert_eq!(actions.len(), 1);
    assert_abstention(&actions, Abstention::SelfEcho);
    assert!(organism.is_speaking());
}

#[test]
fn unpredicted_keyword_above_echo_triggers_barge_in() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    organism.step(&Event::CortexReply {
        now: Millis(1000),
        thought: ThoughtId(1),
        text: "Tudo bem, vou ajustar a temperatura.".to_string(),
    });
    assert!(!organism.self_speech_has_keyword());

    organism.step(&Event::PlaybackStarted {
        now: Millis(1200),
        utterance: UtteranceId(1),
    });

    let actions = organism.step(&Event::Speech {
        now: Millis(1400),
        cue: SpeechCue {
            energy: 0.80,
            duration_ms: 400,
            vad_confidence: 0.90,
            keyword: true,
            speaker_sim: None,
        },
    });

    assert_thought(&actions, 1, &Reason::Keyword);
    assert!(organism.is_hangover());
    assert_eq!(
        organism.playback_status(),
        PlaybackStatus::Hangover {
            utterance: UtteranceId(1),
            until: Millis(1600),
        }
    );
    assert!(!organism.is_speaking());
}

#[test]
fn loud_speech_barge_in_triggers_follow_up() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    organism.step(&Event::PlaybackStarted {
        now: Millis(1000),
        utterance: UtteranceId(1),
    });

    let exp_before = organism.echo_energy_expectation();

    let actions = organism.step(&Event::Speech {
        now: Millis(1300),
        cue: SpeechCue {
            energy: 0.95,
            duration_ms: 700,
            vad_confidence: 0.95,
            keyword: false,
            speaker_sim: None,
        },
    });

    assert_thought(&actions, 1, &Reason::FollowUp);
    assert!(organism.is_hangover());
    assert_eq!(
        organism.playback_status(),
        PlaybackStatus::Hangover {
            utterance: UtteranceId(1),
            until: Millis(1500),
        }
    );
    assert!(!organism.is_speaking());
    assert!((organism.echo_energy_expectation() - exp_before).abs() < f32::EPSILON);
}

#[test]
fn consecutive_barge_in_ratchet_breaks_underestimated_echo_loop() {
    let mut profile = Profile::t1_ref();
    profile.echo_initial_energy = 0.20;
    profile.echo_barge_in_margin = 0.15;
    let mut organism = Organism::new(profile).unwrap();

    organism.step(&Event::Speech {
        now: Millis(1000),
        cue: SpeechCue {
            energy: 0.85,
            duration_ms: 1000,
            vad_confidence: 0.95,
            keyword: true,
            speaker_sim: None,
        },
    });

    organism.step(&Event::CortexReply {
        now: Millis(1100),
        thought: ThoughtId(1),
        text: "Resposta 1".to_string(),
    });
    organism.step(&Event::PlaybackStarted {
        now: Millis(1200),
        utterance: UtteranceId(1),
    });

    let actions1 = organism.step(&Event::Speech {
        now: Millis(1300),
        cue: SpeechCue {
            energy: 0.70,
            duration_ms: 500,
            vad_confidence: 0.90,
            keyword: false,
            speaker_sim: None,
        },
    });
    assert_thought(&actions1, 2, &Reason::FollowUp);

    organism.step(&Event::CortexReply {
        now: Millis(1400),
        thought: ThoughtId(2),
        text: "Resposta 2".to_string(),
    });
    organism.step(&Event::PlaybackStarted {
        now: Millis(1500),
        utterance: UtteranceId(2),
    });

    let actions2 = organism.step(&Event::Speech {
        now: Millis(1600),
        cue: SpeechCue {
            energy: 0.70,
            duration_ms: 500,
            vad_confidence: 0.90,
            keyword: false,
            speaker_sim: None,
        },
    });
    assert_thought(&actions2, 3, &Reason::FollowUp);
    assert!(organism.echo_energy_expectation() >= 0.70);

    organism.step(&Event::CortexReply {
        now: Millis(1700),
        thought: ThoughtId(3),
        text: "Resposta 3".to_string(),
    });
    organism.step(&Event::PlaybackStarted {
        now: Millis(1800),
        utterance: UtteranceId(3),
    });

    let actions3 = organism.step(&Event::Speech {
        now: Millis(1900),
        cue: SpeechCue {
            energy: 0.70,
            duration_ms: 500,
            vad_confidence: 0.90,
            keyword: false,
            speaker_sim: None,
        },
    });
    assert_abstention(&actions3, Abstention::SelfEcho);
    assert!(organism.is_speaking());
}

#[test]
fn dual_mode_attention_anchoring() {
    let profile = Profile::t1_ref();

    let mut text_org = Organism::new(profile.clone()).unwrap();
    text_org.step(&Event::Speech {
        now: Millis(1000),
        cue: SpeechCue {
            energy: 0.85,
            duration_ms: 1000,
            vad_confidence: 0.95,
            keyword: true,
            speaker_sim: None,
        },
    });
    text_org.step(&Event::CortexReply {
        now: Millis(1500),
        thought: ThoughtId(1),
        text: "Text reply".to_string(),
    });
    assert_eq!(text_org.attention_until(), Some(Millis(6500)));

    let mut voice_org = Organism::new(profile).unwrap();
    voice_org.step(&Event::Speech {
        now: Millis(1000),
        cue: SpeechCue {
            energy: 0.85,
            duration_ms: 1000,
            vad_confidence: 0.95,
            keyword: true,
            speaker_sim: None,
        },
    });
    voice_org.step(&Event::CortexReply {
        now: Millis(1500),
        thought: ThoughtId(1),
        text: "Voice reply".to_string(),
    });
    voice_org.step(&Event::PlaybackStarted {
        now: Millis(1700),
        utterance: UtteranceId(1),
    });
    assert_eq!(voice_org.attention_until(), None);

    voice_org.step(&Event::PlaybackFinished {
        now: Millis(3700),
        utterance: UtteranceId(1),
    });
    assert_eq!(voice_org.attention_until(), Some(Millis(8900)));
}

#[test]
#[allow(clippy::too_many_lines)]
fn closed_loop_feedback_immunity_pure_core() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();

    let actions1 = organism.step(&Event::Speech {
        now: Millis(5_000),
        cue: SpeechCue {
            energy: 0.85,
            duration_ms: 1000,
            vad_confidence: 0.95,
            keyword: true,
            speaker_sim: None,
        },
    });
    assert_thought(&actions1, 1, &Reason::Keyword);

    organism.step(&Event::CortexReply {
        now: Millis(6_000),
        thought: ThoughtId(1),
        text: "Olá! Eu sou o Enton. Como posso ajudar?".to_string(),
    });

    organism.step(&Event::PlaybackStarted {
        now: Millis(6_400),
        utterance: UtteranceId(1),
    });

    let e1 = organism.step(&Event::Speech {
        now: Millis(6_800),
        cue: SpeechCue {
            energy: 0.70,
            duration_ms: 400,
            vad_confidence: 0.90,
            keyword: false,
            speaker_sim: None,
        },
    });
    assert_abstention(&e1, Abstention::SelfEcho);

    let e2 = organism.step(&Event::Speech {
        now: Millis(7_200),
        cue: SpeechCue {
            energy: 0.72,
            duration_ms: 400,
            vad_confidence: 0.92,
            keyword: true,
            speaker_sim: None,
        },
    });
    assert_abstention(&e2, Abstention::SelfEcho);

    organism.step(&Event::PlaybackFinished {
        now: Millis(8_400),
        utterance: UtteranceId(1),
    });

    let e3 = organism.step(&Event::Speech {
        now: Millis(8_500),
        cue: SpeechCue {
            energy: 0.65,
            duration_ms: 200,
            vad_confidence: 0.85,
            keyword: false,
            speaker_sim: None,
        },
    });
    assert_abstention(&e3, Abstention::SelfEcho);

    let actions2 = organism.step(&Event::Speech {
        now: Millis(10_000),
        cue: SpeechCue {
            energy: 0.80,
            duration_ms: 600,
            vad_confidence: 0.90,
            keyword: false,
            speaker_sim: None,
        },
    });
    assert_thought(&actions2, 2, &Reason::FollowUp);

    organism.step(&Event::CortexReply {
        now: Millis(11_000),
        thought: ThoughtId(2),
        text: "Tudo bem!".to_string(),
    });

    organism.step(&Event::PlaybackStarted {
        now: Millis(11_400),
        utterance: UtteranceId(2),
    });

    let e4 = organism.step(&Event::Speech {
        now: Millis(11_800),
        cue: SpeechCue {
            energy: 0.68,
            duration_ms: 300,
            vad_confidence: 0.88,
            keyword: true,
            speaker_sim: None,
        },
    });
    assert_abstention(&e4, Abstention::SelfEcho);

    organism.step(&Event::PlaybackFinished {
        now: Millis(13_400),
        utterance: UtteranceId(2),
    });

    for t in 14..=60 {
        let tick_actions = organism.step(&Event::Tick {
            now: Millis(t * 1_000),
        });
        assert!(tick_actions.is_empty(), "tick action at second {t}");
    }

    assert_eq!(organism.conversation_thought(), None);

    let next_thought_actions = organism.step(&Event::Speech {
        now: Millis(60_000),
        cue: SpeechCue {
            energy: 0.85,
            duration_ms: 1000,
            vad_confidence: 0.95,
            keyword: true,
            speaker_sim: None,
        },
    });
    assert_thought(&next_thought_actions, 3, &Reason::Keyword);
}

#[test]
fn playback_watchdog_terminates_stuck_playback_to_prevent_deafness() {
    let mut profile = Profile::t1_ref();
    profile.max_playback_ms = 10_000;
    let mut organism = Organism::new(profile).unwrap();

    organism.step(&Event::PlaybackStarted {
        now: Millis(1000),
        utterance: UtteranceId(1),
    });
    assert!(organism.is_speaking());

    organism.step(&Event::Tick {
        now: Millis(10_000),
    });
    assert!(organism.is_speaking());

    organism.step(&Event::Tick {
        now: Millis(11_000),
    });
    assert!(!organism.is_speaking());
    assert_eq!(organism.playback_status(), PlaybackStatus::Idle);

    let actions = organism.step(&Event::Speech {
        now: Millis(11_500),
        cue: SpeechCue {
            energy: 0.85,
            duration_ms: 1000,
            vad_confidence: 0.95,
            keyword: true,
            speaker_sim: None,
        },
    });
    assert_thought(&actions, 1, &Reason::Keyword);
}

#[test]
fn non_finite_cues_during_playback_do_not_corrupt_echo_expectation_and_subsequent_barge_in_works() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();

    organism.step(&Event::PlaybackStarted {
        now: Millis(1000),
        utterance: UtteranceId(1),
    });
    assert!(organism.is_speaking());

    let initial_exp = organism.echo_energy_expectation();

    // 1. Cue with NaN energy arrives during playback: must not corrupt forward model
    let actions_nan = organism.step(&Event::Speech {
        now: Millis(1200),
        cue: SpeechCue {
            energy: f32::NAN,
            duration_ms: 300,
            vad_confidence: 0.90,
            keyword: false,
            speaker_sim: None,
        },
    });
    assert_abstention(&actions_nan, Abstention::SelfEcho);
    assert!((organism.echo_energy_expectation() - initial_exp).abs() < f32::EPSILON);

    // 2. Cue with positive infinity energy arrives during playback: must not corrupt forward model
    let actions_inf = organism.step(&Event::Speech {
        now: Millis(1400),
        cue: SpeechCue {
            energy: f32::INFINITY,
            duration_ms: 300,
            vad_confidence: 0.90,
            keyword: false,
            speaker_sim: None,
        },
    });
    assert_abstention(&actions_inf, Abstention::SelfEcho);
    assert!((organism.echo_energy_expectation() - initial_exp).abs() < f32::EPSILON);

    // 3. Cue with negative infinity energy arrives during playback: must not corrupt forward model
    let actions_neg_inf = organism.step(&Event::Speech {
        now: Millis(1600),
        cue: SpeechCue {
            energy: f32::NEG_INFINITY,
            duration_ms: 300,
            vad_confidence: 0.90,
            keyword: false,
            speaker_sim: None,
        },
    });
    assert_abstention(&actions_neg_inf, Abstention::SelfEcho);
    assert!((organism.echo_energy_expectation() - initial_exp).abs() < f32::EPSILON);

    // 4. Real barge-in with keyword arrives afterwards: works cleanly!
    let actions_barge_in = organism.step(&Event::Speech {
        now: Millis(1800),
        cue: SpeechCue {
            energy: 0.85,
            duration_ms: 500,
            vad_confidence: 0.95,
            keyword: true,
            speaker_sim: None,
        },
    });
    assert_thought(&actions_barge_in, 1, &Reason::Keyword);
    assert!(organism.is_hangover());
}

#[test]
fn residual_echo_after_barge_in_is_self_echo_and_does_not_open_follow_up() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();

    // 1. Playback starts
    organism.step(&Event::PlaybackStarted {
        now: Millis(1000),
        utterance: UtteranceId(1),
    });
    assert!(organism.is_speaking());

    // 2. Legitimate barge-in arrives at t=1200ms
    let actions_barge_in = organism.step(&Event::Speech {
        now: Millis(1200),
        cue: SpeechCue {
            energy: 0.85,
            duration_ms: 400,
            vad_confidence: 0.95,
            keyword: true,
            speaker_sim: None,
        },
    });
    assert_thought(&actions_barge_in, 1, &Reason::Keyword);
    // Organism transitions to Hangover until 1200 + 200 = 1400ms
    assert!(organism.is_hangover());
    assert_eq!(
        organism.playback_status(),
        PlaybackStatus::Hangover {
            utterance: UtteranceId(1),
            until: Millis(1400),
        }
    );

    // 3. Acoustic residual echo arrives 50ms later (t=1250ms), within hangover window
    let actions_echo = organism.step(&Event::Speech {
        now: Millis(1250),
        cue: SpeechCue {
            energy: 0.70, // below barge-in margin
            duration_ms: 100,
            vad_confidence: 0.80,
            keyword: false,
            speaker_sim: None,
        },
    });
    // Must be classified as SelfEcho abstention, NOT trigger a FollowUp thought!
    assert_abstention(&actions_echo, Abstention::SelfEcho);
    // Does not open a new thought
    assert_eq!(organism.conversation_thought(), Some(ThoughtId(1)));

    // 4. After hangover expires (t=1450ms), inside active attention window (1200 + 5000 = 6200ms),
    // a real user follow-up triggers FollowUp thought
    let actions_follow_up = organism.step(&Event::Speech {
        now: Millis(1450),
        cue: SpeechCue {
            energy: 0.70,
            duration_ms: 600,
            vad_confidence: 0.85,
            keyword: false,
            speaker_sim: None,
        },
    });
    assert_thought(&actions_follow_up, 2, &Reason::FollowUp);
}

#[test]
fn self_speech_keyword_detection_respects_unicode_word_boundaries() {
    // Test exact cases required by user
    assert!(contains_keyword_word("Enton,", "enton"));
    assert!(contains_keyword_word("ENTON", "enton"));
    assert!(!contains_keyword_word("entonação", "enton"));
    assert!(contains_keyword_word("o Enton.", "enton"));

    // Additional boundary and Unicode cases
    assert!(contains_keyword_word("enton", "enton"));
    assert!(contains_keyword_word("Olá, Enton!", "enton"));
    assert!(contains_keyword_word("¿Enton?", "enton"));
    assert!(!contains_keyword_word("desentonação", "enton"));
    assert!(!contains_keyword_word("", "enton"));
    assert!(!contains_keyword_word("qualquer palavra", "enton"));

    // Verify organism integration via CortexReply
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();

    // 'entonação' must NOT flag self_speech_has_keyword
    organism.step(&Event::CortexReply {
        now: Millis(1000),
        thought: ThoughtId(1),
        text: "Qual é a entonação correta?".to_string(),
    });
    assert!(!organism.self_speech_has_keyword());

    // 'Enton,' must flag self_speech_has_keyword
    organism.step(&Event::CortexReply {
        now: Millis(2000),
        thought: ThoughtId(2),
        text: "Enton, preciso de ajuda.".to_string(),
    });
    assert!(organism.self_speech_has_keyword());

    // 'ENTON' must flag self_speech_has_keyword
    organism.step(&Event::CortexReply {
        now: Millis(3000),
        thought: ThoughtId(3),
        text: "EU SOU O ENTON".to_string(),
    });
    assert!(organism.self_speech_has_keyword());

    // 'o Enton.' must flag self_speech_has_keyword
    organism.step(&Event::CortexReply {
        now: Millis(4000),
        thought: ThoughtId(4),
        text: "Chame o Enton.".to_string(),
    });
    assert!(organism.self_speech_has_keyword());
}

#[test]
fn playback_finished_discretionary_opens_no_attention_window_so_cue_is_unaddressed() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();

    // 1. Unaddressed speech cue triggers discretionary Speech think
    let actions_initial = organism.step(&speech(1_000, false));
    assert_thought(&actions_initial, 1, &Reason::Speech);
    assert!(!organism.speaking_for_obligation());

    // 2. Playback starts and finishes
    organism.step(&Event::PlaybackStarted {
        now: Millis(1_500),
        utterance: UtteranceId(1),
    });
    assert_eq!(organism.attention_until(), None);

    organism.step(&Event::PlaybackFinished {
        now: Millis(3_000),
        utterance: UtteranceId(1),
    });
    // Discretionary thought must not open a post-playback attention window
    assert_eq!(organism.attention_until(), None);

    // 3. Right after hangover expires (t = 3000 + 200 = 3200ms), a non-keyword cue arrives
    let actions_subsequent = organism.step(&speech(3_250, false));

    // Because there is no attention window, it is NOT a FollowUp think: it goes through
    // the unaddressed path (blocked here by cooldown from the earlier discretionary think).
    assert_abstention(&actions_subsequent, Abstention::Cooldown);
    let [Action::Abstain { reason, .. }] = actions_subsequent.as_slice() else {
        panic!("expected Abstain action, got {actions_subsequent:?}");
    };
    assert_eq!(*reason, Reason::Speech);
    assert_ne!(*reason, Reason::FollowUp);
}

#[test]
fn playback_finished_obligation_opens_attention_window_for_follow_up() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();

    // 1. Keyword speech cue triggers obligation Keyword think
    let actions_initial = organism.step(&speech(1_000, true));
    assert_thought(&actions_initial, 1, &Reason::Keyword);
    assert!(organism.speaking_for_obligation());

    // 2. Playback starts and finishes
    organism.step(&Event::PlaybackStarted {
        now: Millis(1_500),
        utterance: UtteranceId(1),
    });
    assert_eq!(organism.attention_until(), None);

    organism.step(&Event::PlaybackFinished {
        now: Millis(3_000),
        utterance: UtteranceId(1),
    });
    // Obligation thought opens post-playback attention window: 3000 + 200 (hangover) + 5000 = 8200ms
    assert_eq!(organism.attention_until(), Some(Millis(8_200)));

    // 3. Right after hangover expires (t = 3000 + 200 = 3200ms), a non-keyword cue arrives inside window
    let actions_subsequent = organism.step(&speech(3_250, false));

    // Inside attention window, non-keyword continuation IS a FollowUp think
    assert_thought(&actions_subsequent, 2, &Reason::FollowUp);
}

fn tv_cue(now: u64) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 0.8,
            duration_ms: 1_000,
            vad_confidence: 0.9,
            keyword: false,
            speaker_sim: None,
        },
    }
}

/// Twenty identical TV-like cues, three seconds apart, with no tick in between.
fn habituate_to_tv(organism: &mut Organism) {
    for i in 0..20 {
        organism.step(&tv_cue(i * 3_000));
    }
}

#[test]
fn slow_habituation_outlasts_a_quiet_gap_and_keeps_the_tv_muted() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    habituate_to_tv(&mut organism);
    let slow = organism.slow_habituation();
    assert!(
        slow > 0.3,
        "twenty similar cues accrue long-term habituation"
    );

    // A two-minute pause erases most of the fast component, not the slow one.
    organism.step(&Event::Tick {
        now: Millis(117_000),
    });
    assert!(organism.habituation() < 0.1);
    assert!(organism.slow_habituation() > 0.9 * slow);
    assert_abstention(&organism.step(&tv_cue(117_000)), Abstention::Habituation);

    // Without the long-term component the same cue would have ignited.
    let mut forgetful = Profile::t1_ref();
    forgetful.slow_habituation_rate = 0.0;
    let mut organism = Organism::new(forgetful).unwrap();
    habituate_to_tv(&mut organism);
    organism.step(&Event::Tick {
        now: Millis(117_000),
    });
    assert_thought(&organism.step(&tv_cue(117_000)), 2, &Reason::Speech);
}

#[test]
fn slow_habituation_never_mutes_a_novel_cue_and_survives_it() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    habituate_to_tv(&mut organism);
    let slow = organism.slow_habituation();
    organism.step(&Event::Tick {
        now: Millis(117_000),
    });

    // Maximal prediction error: the fast component resets (A9) and the cue ignites.
    let novel = organism.step(&Event::Speech {
        now: Millis(117_001),
        cue: SpeechCue {
            energy: 0.1,
            duration_ms: 100,
            vad_confidence: 0.45,
            keyword: false,
            speaker_sim: None,
        },
    });
    assert!(
        !matches!(
            novel.as_slice(),
            [Action::Abstain {
                why: Abstention::Habituation,
                ..
            }]
        ),
        "a novel cue is never rejected as habituation, got {novel:?}"
    );
    assert!(organism.habituation().abs() < f32::EPSILON);
    assert!(organism.slow_habituation() > 0.9 * slow);
}

fn voice(now: u64, keyword: bool, duration_ms: u32, speaker_sim: f32) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 1.0,
            duration_ms,
            vad_confidence: 1.0,
            keyword,
            speaker_sim: Some(speaker_sim),
        },
    }
}

#[test]
fn another_voice_inside_the_window_is_not_a_follow_up() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    assert_thought(
        &organism.step(&voice(1_000, true, 1_500, 0.9)),
        1,
        &Reason::Keyword,
    );
    organism.step(&Event::CortexReply {
        now: Millis(2_000),
        thought: ThoughtId(1),
        text: "Oi!".into(),
    });
    let window = organism.attention_until();

    // The TV talks inside the window: no follow-up, and the window does not grow.
    assert_abstention(
        &organism.step(&voice(3_000, false, 1_500, 0.2)),
        Abstention::OtherSpeaker,
    );
    assert_eq!(organism.attention_until(), window);

    // The person who asked keeps the conversation going.
    assert_thought(
        &organism.step(&voice(3_500, false, 1_500, 0.9)),
        2,
        &Reason::FollowUp,
    );
}

#[test]
fn a_pending_name_call_waits_for_its_own_speaker() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    assert!(matches!(
        organism.step(&voice(1_000, true, 400, 0.9)).as_slice(),
        [Action::Attend { .. }]
    ));

    assert_abstention(
        &organism.step(&voice(2_000, false, 1_200, 0.2)),
        Abstention::OtherSpeaker,
    );
    assert!(
        organism.is_attending(),
        "the TV did not consume the pending turn"
    );

    assert_thought(
        &organism.step(&voice(3_000, false, 1_200, 0.9)),
        1,
        &Reason::Keyword,
    );
}

#[test]
fn only_the_addressed_voice_barges_in_without_saying_the_name() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    assert_thought(
        &organism.step(&voice(1_000, true, 1_500, 0.9)),
        1,
        &Reason::Keyword,
    );
    organism.step(&Event::PlaybackStarted {
        now: Millis(1_500),
        utterance: UtteranceId(1),
    });

    // A loud other voice over Enton's playback is not an interruption.
    assert_abstention(
        &organism.step(&voice(2_000, false, 1_000, 0.2)),
        Abstention::SelfEcho,
    );
    assert!(organism.is_speaking());

    // The same loudness in the addressed voice is a barge-in.
    assert_thought(
        &organism.step(&voice(2_500, false, 1_000, 0.9)),
        2,
        &Reason::FollowUp,
    );
}
