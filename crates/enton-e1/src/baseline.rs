//! The two simple policies. Payment is always enforced by the outer harness.
use enton_core::{Abstention, Action, Event, Millis, Reason, ThoughtId};

#[derive(Debug, Clone)]
pub(crate) struct Baseline {
    fixed_window: bool,
    last_think: Option<Millis>,
    until: Option<Millis>,
    pending_window: Option<ThoughtId>,
    next_thought: u64,
}
impl Baseline {
    pub(crate) fn new(fixed_window: bool) -> Self {
        Self {
            fixed_window,
            last_think: None,
            until: None,
            pending_window: None,
            next_thought: 1,
        }
    }
    pub(crate) fn step(&mut self, event: &Event, can_pay: bool) -> Vec<Action> {
        if let Event::CortexReply { thought, now, .. } = event {
            if self.pending_window == Some(*thought) {
                self.pending_window = None;
                self.until = Some(Millis(now.0.saturating_add(5000)));
            }
            return vec![];
        }
        let Event::Speech { now, cue } = event else {
            return vec![];
        };
        let continuation =
            !cue.keyword && self.fixed_window && self.until.is_some_and(|until| *now < until);
        let reason = if cue.keyword {
            Reason::Keyword
        } else if continuation {
            Reason::FollowUp
        } else {
            Reason::Speech
        };
        let eligible = cue.keyword
            || if continuation {
                cue.vad_confidence >= 0.5
            } else {
                cue.vad_confidence >= 0.8 && cue.energy >= 0.8
            };
        let why = if !eligible {
            Some(Abstention::BelowThreshold)
        } else if !continuation && self.last_think.is_some_and(|last| now.since(last) < 5000) {
            Some(Abstention::Cooldown)
        } else if !can_pay {
            Some(Abstention::OutOfEnergy)
        } else {
            None
        };
        if let Some(why) = why {
            return vec![Action::Abstain {
                reason,
                salience: cue.vad_confidence,
                why,
            }];
        }
        let thought = ThoughtId(self.next_thought);
        self.next_thought += 1;
        self.last_think = Some(*now);
        if self.fixed_window && (cue.keyword || continuation) {
            self.pending_window = Some(thought);
        }
        vec![Action::Think {
            thought,
            reason,
            salience: cue.vad_confidence,
        }]
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn fixed_window_echo_handling() {
        let mut window = Baseline::new(true);

        // 1. Initial keyword speech triggers Think
        window.step(&speech(1000, true), true);

        // 2. An echo arrives BEFORE CortexReply.
        // It has no keyword, and its VAD is 0.55.
        // For a continuation, 0.55 >= 0.5, so it would pass.
        // But since the window hasn't opened yet, it must pass the simple threshold (0.8).
        // It fails the simple threshold, so it should Abstain (BelowThreshold).
        let echo = Event::Speech {
            now: Millis(1500),
            cue: SpeechCue {
                energy: 0.45,
                vad_confidence: 0.55,
                duration_ms: 250,
                keyword: false,
                speaker_sim: None,
                media: None,
                turn_complete: None,
            },
        };
        assert!(matches!(
            window.step(&echo, true).as_slice(),
            [Action::Abstain {
                why: Abstention::BelowThreshold,
                ..
            }]
        ));

        // 3. CortexReply opens the window.
        let reply = Event::CortexReply {
            thought: ThoughtId(1),
            now: Millis(2000),
            text: "reply".into(),
        };
        window.step(&reply, true);

        // 4. A continuation arrives AFTER CortexReply.
        // It has no keyword, VAD 0.55. Now the window is open, so 0.55 >= 0.5 is sufficient.
        let cont = Event::Speech {
            now: Millis(2500),
            cue: SpeechCue {
                energy: 0.45,
                vad_confidence: 0.55,
                duration_ms: 250,
                keyword: false,
                speaker_sim: None,
                media: None,
                turn_complete: None,
            },
        };
        assert!(matches!(
            window.step(&cont, true).as_slice(),
            [Action::Think {
                reason: Reason::FollowUp,
                ..
            }]
        ));
    }

    use super::*;
    use enton_core::SpeechCue;
    fn speech(now: u64, keyword: bool) -> Event {
        Event::Speech {
            now: Millis(now),
            cue: SpeechCue {
                energy: 0.9,
                vad_confidence: 0.9,
                duration_ms: 1000,
                keyword,
                speaker_sim: None,
                media: None,
                turn_complete: None,
            },
        }
    }
    #[test]
    fn unpaid_calls_cannot_advance_cooldown_ids_or_the_window() {
        let mut b = Baseline::new(true);
        assert!(matches!(
            b.step(&speech(1000, true), false).as_slice(),
            [Action::Abstain {
                why: Abstention::OutOfEnergy,
                ..
            }]
        ));
        assert_eq!(b.until, None);
        assert!(matches!(
            b.step(&speech(1001, true), true).as_slice(),
            [Action::Think {
                thought: ThoughtId(1),
                ..
            }]
        ));
    }
    #[test]
    fn third_controller_serves_continuations_but_simple_keeps_cooldown() {
        let mut simple = Baseline::new(false);
        let mut window = Baseline::new(true);
        simple.step(&speech(1000, true), true);
        window.step(&speech(1000, true), true);

        let reply = Event::CortexReply {
            thought: ThoughtId(1),
            now: Millis(1200),
            text: "reply".into(),
        };
        window.step(&reply, true);
        assert!(matches!(
            simple.step(&speech(2000, false), true).as_slice(),
            [Action::Abstain {
                why: Abstention::Cooldown,
                ..
            }]
        ));
        assert!(matches!(
            window.step(&speech(2000, false), true).as_slice(),
            [Action::Think {
                reason: Reason::FollowUp,
                ..
            }]
        ));
        assert!(matches!(
            window.step(&speech(7000, false), true).as_slice(),
            [Action::Think {
                reason: Reason::Speech,
                ..
            }]
        ));
    }
}
