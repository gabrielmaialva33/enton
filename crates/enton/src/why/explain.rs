use enton_core::{Abstention, Profile, Reason};

use super::audit::{Context, CueRecord, Decision, Fate, Then};
use super::render::{SECOND_MS, duration, nats};

/// One plain-language line: what was decided, and the evidence or state behind it.
pub(super) fn explain(record: &CueRecord, profile: &Profile) -> String {
    match &record.decision {
        Decision::Think {
            thought,
            reason,
            salience,
            propensity,
            fate,
        } => {
            let cause = if *reason == Reason::Keyword {
                if record.cue.keyword {
                    "Enton was called by name".to_owned()
                } else {
                    "this finished the request that began with \"Enton...\"".to_owned()
                }
            } else if *reason == Reason::FollowUp {
                "a follow-up inside the conversation window".to_owned()
            } else if *reason == Reason::Speech {
                format!(
                    "overheard speech was salient enough to think about (salience {salience:.2})"
                )
            } else {
                format!("{} ignited a thought", reason_name(reason))
            };
            let coin = propensity
                .map(|p| {
                    format!(
                        "; the cue was borderline, and a coin flip explored it (probability {p:.2})"
                    )
                })
                .unwrap_or_default();
            format!(
                "Thought #{thought} ({}): {cause}{coin}{}.",
                reason_name(reason),
                fate_clause(fate.as_ref())
            )
        }
        Decision::Attend { until_ms } => format!(
            "Waited (Attend): Enton heard its name and waited up to {} for the rest of the request; {}.",
            duration(until_ms.saturating_sub(record.at_ms)),
            then_clause(record)
        ),
        Decision::Abstain {
            reason,
            salience,
            why,
            propensity,
        } => format!(
            "Abstained ({why:?}): {}{}.",
            abstention_cause(record, profile, reason, *salience, *why),
            propensity
                .map(|p| format!(
                    "; the cue was borderline, and a coin flip passed on it (probability {p:.2})"
                ))
                .unwrap_or_default()
        ),
        Decision::Other => "No decision this version of enton why can show.".to_owned(),
    }
}

fn fate_clause(fate: Option<&Fate>) -> String {
    match fate {
        Some(Fate::Done {
            reply_chars: Some(chars),
        }) => format!("; the cortex replied ({chars} characters)"),
        Some(Fate::Done { reply_chars: None }) => "; the cortex replied".to_owned(),
        Some(Fate::Failed {
            failure: Some(failure),
        }) => format!(", but the thought failed: {failure}"),
        Some(Fate::Failed { failure: None }) => ", but the thought failed".to_owned(),
        Some(Fate::Pending) => {
            ", but the thought never resolved (still running, or cut off by a crash)".to_owned()
        }
        Some(Fate::Unrecorded) | None => String::new(),
    }
}

fn then_clause(record: &CueRecord) -> String {
    match &record.then {
        Some(Then::Timeout { at_ms, decision }) => {
            let after = duration(at_ms.saturating_sub(record.at_ms));
            match decision {
                Decision::Think { thought, fate, .. } => format!(
                    "nothing followed, so after {after} it answered the name alone (thought #{thought}{})",
                    fate_clause(fate.as_ref())
                ),
                Decision::Abstain { why, .. } => format!(
                    "nothing followed, and when the window closed after {after} it abstained ({why:?})"
                ),
                Decision::Attend { .. } | Decision::Other => {
                    format!("nothing followed, and the window closed after {after}")
                }
            }
        }
        Some(Then::Continued { at_ms, seq }) => format!(
            "the cue at seq {seq}, {} later, carried the turn on",
            duration(at_ms.saturating_sub(record.at_ms))
        ),
        None => "it was still waiting at the last recorded event".to_owned(),
    }
}

/// While the TV counted as on, how long it had been.
fn tv_clause(context: &Context) -> String {
    match context.tv_on_for_ms {
        Some(ms) if ms >= SECOND_MS => {
            format!("; the TV had been on for {}", duration(ms))
        }
        Some(_) => "; the TV had just come on".to_owned(),
        None => String::new(),
    }
}

// `Abstention` is not `#[non_exhaustive]`, so a wildcard after its nine variants is
// unreachable today. It stays so that `enton why` keeps compiling, with a generic
// line, while the core grows a new abstention.
#[allow(unreachable_patterns)]
fn abstention_cause(
    record: &CueRecord,
    profile: &Profile,
    reason: &Reason,
    salience: f32,
    why: Abstention,
) -> String {
    let evidence = &record.evidence;
    let context = &record.context;
    let owner_live = evidence
        .owner_live
        .map(|llr| format!(" (owner-live {})", nats(llr)))
        .unwrap_or_default();
    let addressed = *reason == Reason::Keyword || *reason == Reason::FollowUp;
    match why {
        Abstention::Media => {
            let llr = evidence
                .live_over_reproduced
                .map(|llr| format!(", {}", nats(llr)))
                .unwrap_or_default();
            let over = if context.enton_speaking {
                " over Enton's own voice"
            } else {
                ""
            };
            format!(
                "the tagger heard a loudspeaker{over}{llr}{}",
                tv_clause(context)
            )
        }
        Abstention::OtherSpeaker => {
            if context.enton_speaking {
                format!(
                    "Enton was speaking, and a voice not verified as the owner's{owner_live} cannot interrupt it"
                )
            } else if *reason == Reason::Keyword {
                format!(
                    "someone else said the name{owner_live} while an \"Enton...\" was still waiting for its own speaker"
                )
            } else {
                format!(
                    "the voice did not match whoever addressed Enton{owner_live}, so it could not continue the conversation{}",
                    tv_clause(context)
                )
            }
        }
        Abstention::Undirected => format!(
            "the directedness detector heard speech addressed to someone else{}",
            evidence
                .addressed_over_not
                .map(|llr| format!(", {}", nats(llr)))
                .unwrap_or_default()
        ),
        Abstention::SelfEcho => format!(
            "it overlapped Enton's own voice and did not stand out from the echo as an interruption (energy {:.2}, echo expected near {:.2})",
            record.cue.energy, profile.echo.echo_initial_energy
        ),
        Abstention::BelowThreshold if *reason == Reason::FollowUp => format!(
            "the voice activity was too weak for a follow-up (VAD {:.2}, needs {:.2})",
            record.cue.vad_confidence, profile.attention.follow_up_min_vad
        ),
        Abstention::BelowThreshold => format!(
            "nobody called Enton, and the speech was not salient enough to think about (salience {salience:.2}, needs {:.2})",
            profile.ignition.discretionary_threshold
        ),
        Abstention::Cooldown => format!(
            "nobody called Enton, and it had thought less than {} before; overheard speech waits out that cooldown",
            duration(profile.ignition.cooldown_ms)
        ),
        Abstention::Habituation => format!(
            "similar sounds kept repeating and Enton got used to them (salience {salience:.2} after habituation, needs {:.2})",
            profile.ignition.discretionary_threshold
        ),
        Abstention::OutOfEnergy => {
            let (account, budget) = if addressed {
                ("obligation", context.obligation_budget)
            } else {
                ("discretionary", context.discretionary_budget)
            };
            format!(
                "the {account} budget could not pay for a thought ({:.2} left, a thought costs {:.2})",
                budget.available, context.think_cost
            )
        }
        Abstention::Torpor => {
            "the body was in torpor (fever or a low battery), and then only a call by name buys a thought"
                .to_owned()
        }
        _ => "the reducer gave no reason this version of enton why can explain".to_owned(),
    }
}

pub(super) fn summary(decision: &Decision) -> String {
    match decision {
        Decision::Think {
            thought,
            reason,
            salience,
            ..
        } => format!(
            "Think #{thought} ({}, salience {salience:.2})",
            reason_name(reason)
        ),
        Decision::Attend { until_ms } => format!("Attend until t = {until_ms} ms"),
        Decision::Abstain {
            reason,
            salience,
            why,
            ..
        } => format!(
            "Abstain: {why:?} ({}, salience {salience:.2})",
            reason_name(reason)
        ),
        Decision::Other => "no decision".to_owned(),
    }
}

fn reason_name(reason: &Reason) -> String {
    if let Reason::Drive(name) = reason {
        format!("drive {name}")
    } else {
        format!("{reason:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::super::audit::{BudgetLeft, SensorEvidence};
    use super::*;
    use enton_core::SpeechCue;

    fn quiet_context() -> Context {
        Context {
            attention_left_ms: None,
            verified_attention_left_ms: None,
            name_pending: false,
            tv_presence: 0.0,
            tv_on_for_ms: None,
            torpor: false,
            enton_speaking: false,
            obligation_budget: BudgetLeft {
                available: 0.4,
                capacity: 120.0,
            },
            discretionary_budget: BudgetLeft {
                available: 12.0,
                capacity: 12.0,
            },
            think_cost: 1.0,
        }
    }

    fn record(cue: SpeechCue, decision: Decision, evidence: SensorEvidence) -> CueRecord {
        CueRecord {
            seq: 7,
            at_ms: 60_000,
            ago_ms: 0,
            cue,
            decision,
            evidence,
            context: quiet_context(),
            then: None,
            explanation: String::new(),
        }
    }

    fn abstain(reason: Reason, why: Abstention) -> Decision {
        Decision::Abstain {
            reason,
            salience: 0.5,
            why,
            propensity: None,
        }
    }

    /// An overheard line that only an audio tagger described: it sounded like a TV.
    fn tv_line() -> SpeechCue {
        SpeechCue {
            energy: 0.6,
            duration_ms: 1_500,
            vad_confidence: 0.9,
            media: Some(0.8),
            ..SpeechCue::default()
        }
    }

    fn name_alone() -> SpeechCue {
        SpeechCue {
            energy: 0.8,
            duration_ms: 300,
            vad_confidence: 0.95,
            keyword: true,
            ..SpeechCue::default()
        }
    }

    #[test]
    fn each_abstention_gets_a_plain_reason() {
        let profile = Profile::t1_ref();
        let line = |record: &CueRecord| explain(record, &profile);
        let voice = SensorEvidence {
            owner_over_other: Some(-1.8),
            owner_over_reproduced: Some(-0.4),
            owner_live: Some(-1.8),
            ..SensorEvidence::default()
        };

        assert_eq!(
            line(&record(
                name_alone(),
                abstain(Reason::Keyword, Abstention::OutOfEnergy),
                SensorEvidence::default()
            )),
            "Abstained (OutOfEnergy): the obligation budget could not pay for a thought (0.40 left, a thought costs 1.00)."
        );
        assert_eq!(
            line(&record(
                tv_line(),
                abstain(Reason::FollowUp, Abstention::OtherSpeaker),
                voice
            )),
            "Abstained (OtherSpeaker): the voice did not match whoever addressed Enton (owner-live -1.8 nats), so it could not continue the conversation."
        );
        assert_eq!(
            line(&record(
                name_alone(),
                abstain(Reason::Keyword, Abstention::OtherSpeaker),
                voice
            )),
            "Abstained (OtherSpeaker): someone else said the name (owner-live -1.8 nats) while an \"Enton...\" was still waiting for its own speaker."
        );
        assert_eq!(
            line(&record(
                SpeechCue {
                    directed: Some(0.1),
                    ..tv_line()
                },
                abstain(Reason::FollowUp, Abstention::Undirected),
                SensorEvidence {
                    addressed_over_not: Some(-2.64),
                    ..SensorEvidence::default()
                }
            )),
            "Abstained (Undirected): the directedness detector heard speech addressed to someone else, -2.6 nats."
        );
        assert_eq!(
            line(&record(
                SpeechCue {
                    vad_confidence: 0.3,
                    ..tv_line()
                },
                abstain(Reason::FollowUp, Abstention::BelowThreshold),
                SensorEvidence::default()
            )),
            "Abstained (BelowThreshold): the voice activity was too weak for a follow-up (VAD 0.30, needs 0.50)."
        );
        assert_eq!(
            line(&record(
                tv_line(),
                abstain(Reason::Speech, Abstention::Torpor),
                SensorEvidence::default()
            )),
            "Abstained (Torpor): the body was in torpor (fever or a low battery), and then only a call by name buys a thought."
        );
        assert_eq!(
            line(&record(
                tv_line(),
                abstain(Reason::Speech, Abstention::Cooldown),
                SensorEvidence::default()
            )),
            "Abstained (Cooldown): nobody called Enton, and it had thought less than 10 s before; overheard speech waits out that cooldown."
        );
    }

    #[test]
    fn a_thought_says_whether_the_cortex_answered() {
        let profile = Profile::t1_ref();
        let think = |fate| Decision::Think {
            thought: 3,
            reason: Reason::Keyword,
            salience: 1.6,
            propensity: None,
            fate,
        };
        let failed = record(
            name_alone(),
            think(Some(Fate::Failed {
                failure: Some("cortex unavailable".to_owned()),
            })),
            SensorEvidence::default(),
        );
        assert_eq!(
            explain(&failed, &profile),
            "Thought #3 (Keyword): Enton was called by name, but the thought failed: cortex unavailable."
        );
        let done = record(
            SpeechCue {
                keyword: false,
                ..name_alone()
            },
            think(Some(Fate::Done {
                reply_chars: Some(42),
            })),
            SensorEvidence::default(),
        );
        assert_eq!(
            explain(&done, &profile),
            "Thought #3 (Keyword): this finished the request that began with \"Enton...\"; the cortex replied (42 characters)."
        );
        assert_eq!(summary(&done.decision), "Think #3 (Keyword, salience 1.60)");
    }

    #[test]
    fn a_coin_flip_at_a_borderline_cue_is_named() {
        let profile = Profile::t1_ref();
        let explored = record(
            tv_line(),
            Decision::Think {
                thought: 4,
                reason: Reason::FollowUp,
                salience: 0.6,
                propensity: Some(0.05),
                fate: Some(Fate::Done {
                    reply_chars: Some(12),
                }),
            },
            SensorEvidence::default(),
        );
        assert_eq!(
            explain(&explored, &profile),
            "Thought #4 (FollowUp): a follow-up inside the conversation window; the cue was borderline, and a coin flip explored it (probability 0.05); the cortex replied (12 characters)."
        );
        let passed = record(
            tv_line(),
            Decision::Abstain {
                reason: Reason::FollowUp,
                salience: 0.6,
                why: Abstention::OtherSpeaker,
                propensity: Some(0.95),
            },
            SensorEvidence {
                owner_live: Some(-1.2),
                ..SensorEvidence::default()
            },
        );
        assert_eq!(
            explain(&passed, &profile),
            "Abstained (OtherSpeaker): the voice did not match whoever addressed Enton (owner-live -1.2 nats), so it could not continue the conversation; the cue was borderline, and a coin flip passed on it (probability 0.95)."
        );
        let json = serde_json::to_value(&passed.decision).unwrap();
        assert!((json["propensity"].as_f64().unwrap() - 0.95).abs() < 1e-6);
        let json = serde_json::to_value(abstain(Reason::Speech, Abstention::Media)).unwrap();
        assert!(json.get("propensity").is_none());
    }
}
