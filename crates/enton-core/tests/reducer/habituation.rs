use enton_core::{Abstention, Action, Event, Millis, Organism, Profile, Reason, SpeechCue};

use super::support::{assert_abstention, assert_thought, tv_cue};

/// Twenty identical TV-like cues, three seconds apart, with no tick in between.
fn habituate_to_tv(organism: &mut Organism) {
    for i in 0..20 {
        organism.step(&tv_cue(i * 3_000));
    }
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
    assert!((profile.habituation.similarity_cutoff - 0.4).abs() < f32::EPSILON);
    assert!((profile.habituation.expectation_coefficient - 0.25).abs() < f32::EPSILON);

    // Configure a stricter similarity cutoff
    profile.habituation.similarity_cutoff = 0.8;
    profile.habituation.expectation_coefficient = 0.50;
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
    forgetful.habituation.slow_habituation_rate = 0.0;
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
