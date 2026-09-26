use enton_core::{
    Abstention, Action, Event, Millis, Organism, Profile, Reason, SpeechCue, ThoughtId, UtteranceId,
};

use super::support::{assert_abstention, assert_thought, speech};

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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
    let mut organism = Organism::new({
        let mut profile = Profile::t1_ref();
        profile.budgets.obligation_budget_per_hour = 2.0;
        profile
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
        },
    });
    assert_abstention(&actions3, Abstention::OutOfEnergy);
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
        media: None,
        turn_complete: None,
        directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
            media: None,
            turn_complete: None,
            directed: None,
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
