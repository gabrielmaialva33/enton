use enton_core::{Abstention, Action, Event, Millis, Organism, Profile, Reason, SpeechCue};

use super::support::{assert_abstention, assert_thought, sensitive_profile, speech};

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
                media: None,
                turn_complete: None,
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
            media: None,
            turn_complete: None,
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
    let threshold = profile.ignition.threshold;

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
            media: None,
            turn_complete: None,
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
                        media: None,
                        turn_complete: None,
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
