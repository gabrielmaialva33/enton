//! Reducer invariants checked after every organism step, in the spirit of
//! the `TigerBeetle` VOPR: the synthetic tapes explore, the watch asserts, and a
//! violation stops the run naming the step and the event that broke it.

use enton_core::{Action, Event, Millis, Organism, Reason, ThoughtId};

use crate::Error;

/// Steps between snapshot round trips. The restored shadow then steps in
/// lockstep with the live organism until the next round trip.
const SHADOW_EVERY: u64 = 1_000;

/// Tolerance for budget bounds, which accumulate f32 refills.
const EPSILON: f32 = 1e-3;

/// Tracks what one organism has done so far and checks each new step.
#[derive(Debug, Default)]
pub(crate) struct Watch {
    steps: u64,
    last_thought: Option<ThoughtId>,
    last_seen: Millis,
    shadow: Option<Box<Organism>>,
}

impl Watch {
    /// Check the organism right after it reduced `event` into `actions`.
    pub(crate) fn after_step(
        &mut self,
        organism: &Organism,
        event: &Event,
        actions: &[Action],
    ) -> Result<(), Error> {
        self.steps += 1;
        self.check_decisions(event, actions)?;
        self.check_thought_ids(event, actions)?;
        self.check_time(organism, event)?;
        self.check_bounds(organism, event)?;
        self.check_shadow(organism, event, actions)
    }

    fn violation(&self, event: &Event, what: &str) -> Error {
        Error::Invariant(format!(
            "step {} at {} ms ({}): {what}",
            self.steps,
            event.now().0,
            kind(event)
        ))
    }

    fn check_decisions(&self, event: &Event, actions: &[Action]) -> Result<(), Error> {
        let decisions = actions
            .iter()
            .filter(|action| {
                matches!(
                    action,
                    Action::Think { .. } | Action::Attend { .. } | Action::Abstain { .. }
                )
            })
            .count();
        let speaks = actions
            .iter()
            .filter(|action| matches!(action, Action::Speak { .. }))
            .count();
        let broken = match event {
            Event::Speech { .. } => (decisions != 1 || speaks != 0)
                .then_some("a speech cue yields exactly one decision"),
            Event::CortexReply { .. } => (decisions != 0 || speaks != 1)
                .then_some("a cortex reply yields exactly one Speak and nothing else"),
            Event::Tick { .. } => actions
                .iter()
                .any(|action| !tick_may_emit(action))
                .then_some("a tick only thinks or abstains, for a drive or an attend timeout"),
            Event::Body { .. } | Event::PlaybackStarted { .. } | Event::PlaybackFinished { .. } => {
                (!actions.is_empty()).then_some("body and playback events decide nothing")
            }
        };
        broken.map_or(Ok(()), |what| Err(self.violation(event, what)))
    }

    fn check_thought_ids(&mut self, event: &Event, actions: &[Action]) -> Result<(), Error> {
        for action in actions {
            if let Action::Think { thought, .. } = action {
                let expected = self.last_thought.map_or(1, |last| last.0 + 1);
                if thought.0 != expected {
                    return Err(self.violation(event, "thought IDs increase by exactly one"));
                }
                self.last_thought = Some(*thought);
            }
        }
        Ok(())
    }

    fn check_time(&mut self, organism: &Organism, event: &Event) -> Result<(), Error> {
        let seen = organism.last_seen();
        if seen < event.now() || seen < self.last_seen {
            return Err(self.violation(event, "the observed time never runs backward"));
        }
        self.last_seen = seen;
        Ok(())
    }

    fn check_bounds(&self, organism: &Organism, event: &Event) -> Result<(), Error> {
        for budget in [
            organism.obligation_budget(),
            organism.discretionary_budget(),
        ] {
            if !(-EPSILON..=budget.capacity + EPSILON).contains(&budget.available) {
                return Err(self.violation(event, "a budget stays within zero and its capacity"));
            }
        }
        let unit = [
            organism.habituation(),
            organism.slow_habituation(),
            organism.echo_energy_expectation(),
            organism.tv_presence(),
        ];
        if unit.iter().any(|level| !(0.0..=1.0).contains(level)) {
            return Err(self.violation(event, "habituation, echo and TV levels stay within [0, 1]"));
        }
        let profile = organism.profile();
        let hangover = profile.echo.echo_hangover_ms;
        let windows = [
            (organism.attention_until(), profile.attention.attention_ms),
            (
                organism.verified_attention_until(),
                profile.attention.verified_attention_ms,
            ),
        ];
        let seen = organism.last_seen().0;
        if windows.iter().any(|(until, span)| {
            until.is_some_and(|until| until.0 > seen.saturating_add(span + hangover))
        }) {
            return Err(self.violation(event, "an attention window never reaches past its span"));
        }
        Ok(())
    }

    fn check_shadow(
        &mut self,
        organism: &Organism,
        event: &Event,
        actions: &[Action],
    ) -> Result<(), Error> {
        if let Some(shadow) = self.shadow.as_mut() {
            let replayed = shadow.step(event);
            if replayed != actions || **shadow != *organism {
                return Err(
                    self.violation(event, "a snapshot round trip diverged from the live state")
                );
            }
        }
        if self.steps.is_multiple_of(SHADOW_EVERY) {
            let blob = serde_json::to_vec(organism)?;
            self.shadow = Some(Box::new(serde_json::from_slice(&blob)?));
        }
        Ok(())
    }
}

/// On a tick the organism may only resolve a pending attend or act on a drive.
fn tick_may_emit(action: &Action) -> bool {
    match action {
        Action::Think { reason, .. } | Action::Abstain { reason, .. } => {
            matches!(reason, Reason::Keyword | Reason::Drive(_))
        }
        Action::Speak { .. } | Action::Attend { .. } => false,
    }
}

fn kind(event: &Event) -> &'static str {
    match event {
        Event::Tick { .. } => "Tick",
        Event::Body { .. } => "Body",
        Event::Speech { .. } => "Speech",
        Event::CortexReply { .. } => "CortexReply",
        Event::PlaybackStarted { .. } => "PlaybackStarted",
        Event::PlaybackFinished { .. } => "PlaybackFinished",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use enton_core::{Abstention, Profile, SpeechCue};

    fn organism() -> Organism {
        Organism::new(Profile::t1_ref()).unwrap()
    }

    fn cue(now: u64) -> Event {
        Event::Speech {
            now: Millis(now),
            cue: SpeechCue {
                energy: 0.2,
                duration_ms: 200,
                vad_confidence: 0.2,
                keyword: false,
                speaker_sim: None,
                media: None,
                turn_complete: None,
            },
        }
    }

    fn abstain() -> Action {
        Action::Abstain {
            reason: Reason::Speech,
            salience: 0.1,
            why: Abstention::BelowThreshold,
        }
    }

    fn think(id: u64) -> Action {
        Action::Think {
            thought: ThoughtId(id),
            reason: Reason::Keyword,
            salience: 1.5,
        }
    }

    #[test]
    fn real_steps_pass_the_watch() {
        let mut organism = organism();
        let mut watch = Watch::default();
        for now in [100, 200, 300] {
            let event = cue(now);
            let actions = organism.step(&event);
            watch.after_step(&organism, &event, &actions).unwrap();
        }
    }

    #[test]
    fn the_snapshot_shadow_starts_and_tracks_a_long_run() {
        let mut organism = organism();
        let mut watch = Watch::default();
        for second in 0..2_500 {
            let event = if second % 7 == 0 {
                cue(second * 1_000 + 1)
            } else {
                Event::Tick {
                    now: Millis(second * 1_000),
                }
            };
            let actions = organism.step(&event);
            watch.after_step(&organism, &event, &actions).unwrap();
        }
        assert!(
            watch.shadow.is_some(),
            "a round trip happened and was checked"
        );
    }

    #[test]
    fn a_speech_cue_with_two_decisions_is_a_violation() {
        let mut organism = organism();
        let event = cue(100);
        organism.step(&event);
        let err = Watch::default()
            .after_step(&organism, &event, &[abstain(), abstain()])
            .unwrap_err();
        assert!(err.to_string().contains("exactly one decision"), "{err}");
    }

    #[test]
    fn skipped_thought_ids_are_a_violation() {
        let mut organism = organism();
        let mut watch = Watch::default();
        let first = cue(100);
        organism.step(&first);
        watch.after_step(&organism, &first, &[think(1)]).unwrap();
        let second = cue(200);
        organism.step(&second);
        let err = watch
            .after_step(&organism, &second, &[think(3)])
            .unwrap_err();
        assert!(err.to_string().contains("increase by exactly one"), "{err}");
    }

    #[test]
    fn a_tick_that_attends_is_a_violation() {
        let mut organism = organism();
        let event = Event::Tick { now: Millis(100) };
        organism.step(&event);
        let err = Watch::default()
            .after_step(
                &organism,
                &event,
                &[Action::Attend {
                    until: Millis(5_100),
                }],
            )
            .unwrap_err();
        assert!(err.to_string().contains("a tick only"), "{err}");
    }
}
