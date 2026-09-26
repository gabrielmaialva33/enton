use enton_core::{
    Abstention, Action, Event, Millis, Organism, PlaybackStatus, Profile, Reason, SpeechCue,
    ThoughtId, UtteranceId, contains_keyword_word,
};

use super::support::{assert_abstention, assert_thought};

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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
            duration_ms: 1_000, // a whole request, not the name alone
            vad_confidence: 0.90,
            keyword: true,
            speaker_sim: None,
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
    profile.echo.echo_initial_energy = 0.20;
    profile.echo.echo_barge_in_margin = 0.15;
    let mut organism = Organism::new(profile).unwrap();

    organism.step(&Event::Speech {
        now: Millis(1000),
        cue: SpeechCue {
            energy: 0.85,
            duration_ms: 1000,
            vad_confidence: 0.95,
            keyword: true,
            speaker_sim: None,
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
        },
    });
    assert_abstention(&actions3, Abstention::SelfEcho);
    assert!(organism.is_speaking());
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
        },
    });
    assert_thought(&next_thought_actions, 3, &Reason::Keyword);
}

#[test]
fn playback_watchdog_terminates_stuck_playback_to_prevent_deafness() {
    let mut profile = Profile::t1_ref();
    profile.echo.max_playback_ms = 10_000;
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
        },
    });
    assert_abstention(&actions_neg_inf, Abstention::SelfEcho);
    assert!((organism.echo_energy_expectation() - initial_exp).abs() < f32::EPSILON);

    // 4. Real barge-in with keyword arrives afterwards: works cleanly!
    let actions_barge_in = organism.step(&Event::Speech {
        now: Millis(1800),
        cue: SpeechCue {
            energy: 0.85,
            duration_ms: 1_000, // a whole request, not the name alone
            vad_confidence: 0.95,
            keyword: true,
            speaker_sim: None,
            media: None,
            turn_complete: None,
            directed: None,
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
            duration_ms: 1_000, // a whole request, not the name alone
            vad_confidence: 0.95,
            keyword: true,
            speaker_sim: None,
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
fn an_unfinished_name_over_playback_cuts_it_and_waits_for_the_rest() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    organism.step(&Event::PlaybackStarted {
        now: Millis(1_000),
        utterance: UtteranceId(1),
    });
    // A loud "Enton..." that the end-of-turn model says is unfinished.
    let actions = organism.step(&Event::Speech {
        now: Millis(1_500),
        cue: SpeechCue {
            energy: 1.0,
            duration_ms: 400,
            vad_confidence: 1.0,
            keyword: true,
            speaker_sim: Some(0.9),
            media: Some(0.1),
            turn_complete: Some(0.2),
            directed: None,
        },
    });
    assert!(
        matches!(actions.as_slice(), [Action::Attend { .. }]),
        "{actions:?}"
    );
    assert!(!organism.is_speaking(), "the playback was cut");
    assert!(organism.is_attending());
}
