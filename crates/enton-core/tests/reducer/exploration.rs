//! Logged exploration: a cue that an evidence objection turns away close to its
//! threshold thinks anyway with the profile's probability, paid by the discretionary
//! account, and every coin flip is logged with the probability of the side that came up.

use enton_core::{
    Abstention, Action, Event, Millis, Organism, Profile, Reason, SpeechCue, ThoughtId, UtteranceId,
};

use super::support::{assert_abstention, assert_thought};

/// t1-ref exploring with `probability`. The directedness detector's clearly undirected
/// band weighs 2.64 nats against Enton, 0.14 past this 2.5-nat threshold: borderline
/// within the one-nat margin. The discretionary account is large, so it never binds.
fn exploring(probability: f32) -> Profile {
    let mut profile = Profile::t1_ref();
    profile.attention.undirected_llr = 2.5;
    profile.budgets.discretionary_budget_per_hour = 10_000.0;
    profile.exploration.explore_probability = probability;
    profile.exploration.explore_seed = 7;
    profile
}

/// A loud, clear 1.5 s cue with the given readings: `(speaker_sim, media, directed)`.
fn said(now: u64, keyword: bool, readings: (Option<f32>, Option<f32>, Option<f32>)) -> Event {
    let (speaker_sim, media, directed) = readings;
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 1.0,
            duration_ms: 1_500,
            vad_confidence: 1.0,
            keyword,
            speaker_sim,
            media,
            turn_complete: None,
            directed,
        },
    }
}

/// A whole request that names Enton: it thinks at once and opens the windows.
fn request(now: u64) -> Event {
    said(now, true, (None, None, None))
}

/// Speech addressed to someone else: the clearly undirected band.
fn aside(now: u64) -> Event {
    said(now, false, (None, None, Some(0.1)))
}

/// `trials` minutes, each with a tick, a request and, 2 s later, an aside in its window.
fn trials(trials: u64) -> Vec<Event> {
    (1..=trials)
        .flat_map(|minute| {
            let start = minute * 60_000;
            [
                Event::Tick { now: Millis(start) },
                request(start + 1_000),
                aside(start + 3_000),
            ]
        })
        .collect()
}

fn logged(action: &Action) -> Option<f32> {
    match action {
        Action::Think { propensity, .. } | Action::Abstain { propensity, .. } => *propensity,
        Action::Speak { .. } | Action::Attend { .. } => None,
    }
}

fn generator(organism: &Organism) -> Option<u64> {
    serde_json::to_value(organism).ok().and_then(|state| {
        state
            .get("explore_state")
            .and_then(serde_json::Value::as_u64)
    })
}

#[test]
fn without_a_probability_nothing_is_flipped_and_the_other_settings_do_not_matter() {
    let mut off = Organism::new(exploring(0.0)).unwrap();
    let mut elsewhere = exploring(0.0);
    elsewhere.exploration.explore_margin_nats = 0.0;
    elsewhere.exploration.explore_seed = 99;
    let mut other = Organism::new(elsewhere).unwrap();
    for event in trials(40) {
        let actions = off.step(&event);
        assert_eq!(actions, other.step(&event));
        assert!(
            actions.iter().all(|action| logged(action).is_none()),
            "{actions:?}"
        );
    }
    assert_eq!(generator(&off), Some(7), "the generator never moved");
}

#[test]
fn a_borderline_objection_is_explored_with_the_logged_probability() {
    let mut organism = Organism::new(exploring(0.25)).unwrap();
    let (mut explored, mut kept) = (0_u32, 0_u32);
    for event in trials(400) {
        let actions = organism.step(&event);
        let aside = matches!(&event, Event::Speech { cue, .. } if cue.directed.is_some());
        for action in &actions {
            match (aside, action) {
                (
                    true,
                    Action::Think {
                        reason: Reason::FollowUp,
                        propensity: Some(p),
                        ..
                    },
                ) => {
                    assert_eq!(p.to_bits(), 0.25_f32.to_bits());
                    explored += 1;
                }
                (
                    true,
                    Action::Abstain {
                        why: Abstention::Undirected,
                        propensity: Some(p),
                        ..
                    },
                ) => {
                    assert_eq!(p.to_bits(), 0.75_f32.to_bits());
                    kept += 1;
                }
                (false, action) => assert!(logged(action).is_none(), "{action:?}"),
                (true, action) => panic!("an aside decided {action:?}"),
            }
        }
    }
    assert_eq!(explored + kept, 400);
    // Binomial(400, 0.25): mean 100, sd 8.7; four sd either way.
    assert!((65..=135).contains(&explored), "{explored} explored of 400");
}

#[test]
fn an_objection_beyond_the_margin_is_never_explored() {
    let mut profile = exploring(1.0);
    // Back to 1.5 nats: the band is now 1.14 nats past it, beyond the one-nat margin.
    profile.attention.undirected_llr = 1.5;
    let mut organism = Organism::new(profile).unwrap();
    for event in trials(20) {
        let actions = organism.step(&event);
        if matches!(&event, Event::Speech { cue, .. } if cue.directed.is_some()) {
            assert!(matches!(
                actions.as_slice(),
                [Action::Abstain {
                    why: Abstention::Undirected,
                    propensity: None,
                    ..
                }]
            ));
        }
    }
    assert_eq!(generator(&organism), Some(7));
}

#[test]
fn an_explored_thought_is_paid_by_the_discretionary_account_and_never_on_credit() {
    let mut profile = exploring(1.0);
    profile.budgets.discretionary_budget_per_hour = 1.0;
    let mut organism = Organism::new(profile).unwrap();
    assert_thought(&organism.step(&request(1_000)), 1, &Reason::Keyword);
    let obligation = organism.obligation_budget().available;
    let discretionary = organism.discretionary_budget().available;
    assert!(matches!(
        organism.step(&aside(3_000)).as_slice(),
        [Action::Think {
            reason: Reason::FollowUp,
            propensity: Some(_),
            ..
        }]
    ));
    assert_eq!(
        organism.obligation_budget().available.to_bits(),
        obligation.to_bits()
    );
    assert_eq!(
        organism.discretionary_budget().available.to_bits(),
        (discretionary - 1.0).to_bits()
    );

    // The account is empty: the next borderline aside keeps its objection, for sure, and
    // the coin is not flipped at all.
    let before = generator(&organism);
    assert!(matches!(
        organism.step(&aside(3_500)).as_slice(),
        [Action::Abstain {
            why: Abstention::Undirected,
            propensity: None,
            ..
        }]
    ));
    assert_eq!(generator(&organism), before);
}

#[test]
fn exploring_replays_and_survives_a_snapshot_bit_for_bit() {
    let tape = trials(60);
    let (first, rest) = tape.split_at(tape.len() / 2);
    let mut live = Organism::new(exploring(0.5)).unwrap();
    let mut straight = Vec::new();
    for event in &tape {
        straight.push(live.step(event));
    }
    let mut resumed = Organism::new(exploring(0.5)).unwrap();
    let mut around = Vec::new();
    for event in first {
        around.push(resumed.step(event));
    }
    let blob = serde_json::to_string(&resumed).unwrap();
    let mut resumed: Organism = serde_json::from_str(&blob).unwrap();
    for event in rest {
        around.push(resumed.step(event));
    }
    assert_eq!(around, straight);
    assert_eq!(resumed, live);
    let flips: Vec<_> = straight.iter().flatten().filter_map(logged).collect();
    assert!(flips.contains(&0.5) && flips.len() == 60, "{flips:?}");
    // Both sides came up.
    let explored = straight
        .iter()
        .flatten()
        .filter(|action| {
            matches!(
                action,
                Action::Think {
                    propensity: Some(_),
                    ..
                }
            )
        })
        .count();
    assert!((1..60).contains(&explored), "{explored}");
}

#[test]
fn enton_s_own_playback_is_never_explored() {
    let mut organism = Organism::new(exploring(1.0)).unwrap();
    assert_thought(&organism.step(&request(1_000)), 1, &Reason::Keyword);
    organism.step(&Event::PlaybackStarted {
        now: Millis(1_400),
        utterance: UtteranceId(1),
    });
    // A voice 0.1 nats short of verification, loud enough to barge in: while Enton
    // talks it may not interrupt, and no coin decides otherwise.
    let actions = organism.step(&said(2_000, false, (Some(0.49), None, None)));
    assert!(matches!(
        actions.as_slice(),
        [Action::Abstain {
            why: Abstention::OtherSpeaker,
            propensity: None,
            ..
        }]
    ));
    assert_eq!(generator(&organism), Some(7));
}

/// Enton answered a request at 2 s: the short window closes at 7 s, the long one at 12 s.
fn after_a_reply(mut organism: Organism) -> Organism {
    assert_thought(&organism.step(&request(1_000)), 1, &Reason::Keyword);
    organism.step(&Event::CortexReply {
        now: Millis(2_000),
        thought: ThoughtId(1),
        text: "Oi!".into(),
    });
    organism
}

#[test]
fn a_voice_just_short_of_verification_explores_the_longer_window() {
    // At 9 s only the long window is open, and a similarity of 0.49 leaves the owner
    // 0.1 nats short of verification: heard as overheard speech, still in the thought
    // cooldown, it abstains. Exploration heard it in the window.
    let unverified = said(9_000, false, (Some(0.49), None, None));
    let mut sure = after_a_reply(Organism::new(exploring(1.0)).unwrap());
    let actions = sure.step(&unverified);
    assert!(
        matches!(
            actions.as_slice(),
            [Action::Think {
                reason: Reason::FollowUp,
                propensity: Some(p),
                ..
            }] if p.to_bits() == 1.0_f32.to_bits()
        ),
        "{actions:?}"
    );
    // At the smallest probability the generator resolves, the coin keeps the objection:
    // the overheard path's own abstention, logged with its probability.
    let mut rare = after_a_reply(Organism::new(exploring(1e-9)).unwrap());
    let actions = rare.step(&unverified);
    assert!(
        matches!(
            actions.as_slice(),
            [Action::Abstain {
                why: Abstention::Cooldown,
                propensity: Some(p),
                ..
            }] if p.to_bits() == (1.0_f32 - 1.0 / 16_777_216.0).to_bits()
        ),
        "{actions:?}"
    );
    // Without exploration the cue is the same abstention, logged as certain.
    let mut plain = after_a_reply(Organism::new(exploring(0.0)).unwrap());
    assert!(matches!(
        plain.step(&unverified).as_slice(),
        [Action::Abstain {
            why: Abstention::Cooldown,
            propensity: None,
            ..
        }]
    ));
}

#[test]
fn overheard_media_near_the_threshold_explores_on_the_discretionary_account() {
    // The tagger's 0.65 on a 1.5 s cue is 0.27 nats past the loudspeaker threshold;
    // 1.0 is 2 nats past it, beyond the margin.
    let near = said(1_000, false, (None, Some(0.65), None));
    let far = said(1_000, false, (None, Some(1.0), None));
    let mut organism = Organism::new(exploring(1.0)).unwrap();
    let discretionary = organism.discretionary_budget().available;
    assert!(matches!(
        organism.step(&near).as_slice(),
        [Action::Think {
            reason: Reason::Speech,
            propensity: Some(_),
            ..
        }]
    ));
    assert_eq!(
        organism.discretionary_budget().available.to_bits(),
        (discretionary - 1.0).to_bits()
    );
    assert!(!organism.speaking_for_obligation());
    let mut organism = Organism::new(exploring(1.0)).unwrap();
    assert!(matches!(
        organism.step(&far).as_slice(),
        [Action::Abstain {
            why: Abstention::Media,
            propensity: None,
            ..
        }]
    ));
}

#[test]
fn a_counterfactual_asks_another_profile_without_moving_the_state() {
    let mut organism = after_a_reply(Organism::new(exploring(0.0)).unwrap());
    let before = organism.clone();
    let mut lifted = exploring(0.0);
    lifted.attention.undirected_llr = 3.0;
    let actions = organism.counterfactual(&lifted, &aside(3_000)).unwrap();
    assert_thought(&actions, 2, &Reason::FollowUp);
    assert_eq!(organism, before, "the state did not move");
    // Asked of its own profile, it is exactly the next step.
    let own = organism
        .counterfactual(&exploring(0.0), &aside(3_000))
        .unwrap();
    assert_eq!(own, organism.step(&aside(3_000)));
    assert_abstention(&own, Abstention::Undirected);
    // An invalid profile is refused.
    let mut broken = exploring(0.0);
    broken.exploration.explore_probability = 2.0;
    assert!(organism.counterfactual(&broken, &aside(4_000)).is_err());
}
