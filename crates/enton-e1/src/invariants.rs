//! Reducer invariants checked after every organism step, in the spirit of
//! the `TigerBeetle` VOPR: the synthetic tapes explore, the watch asserts, and a
//! violation stops the run naming the step and the event that broke it.

use enton_core::{Abstention, Action, Budget, Deferred, Event, Millis, Organism, Reason, ThoughtId};

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
    /// Obligation and discretionary budgets after the previous step. Only ticks refill,
    /// so a speech cue finds exactly these.
    budgets: Option<(Budget, Budget)>,
    /// Whether the last checklist event said there is something to check (none yet: no).
    checklist_actionable: bool,
    /// Whether the cortex failed since its last reply, so a backoff may be under way.
    failed_since_reply: bool,
    /// Until when the owner asked for quiet, as of the last quiet command (none yet: never).
    quiet_until: Option<Millis>,
    /// Whether the quiet hours are on, as of the last reading (none yet: off).
    quiet_hours: bool,
    /// The intent a drive held before this step, if one was held.
    deferred: Option<Deferred>,
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
        self.check_discretion(event, actions)?;
        self.check_initiative(organism, event, actions)?;
        self.check_thought_ids(event, actions)?;
        self.check_time(organism, event)?;
        self.check_bounds(organism, event)?;
        self.check_exploration(organism, event, actions)?;
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
            Event::CortexReply { text, .. } => {
                let expected = usize::from(!text.trim().is_empty());
                (decisions != 0 || speaks != expected).then_some(
                    "a cortex reply yields exactly one Speak, none when silent, and nothing else",
                )
            }
            Event::Tick { .. } => actions
                .iter()
                .any(|action| !tick_may_emit(action))
                .then_some("a tick only thinks or abstains, for a drive or an attend timeout"),
            Event::Body { .. }
            | Event::PlaybackStarted { .. }
            | Event::PlaybackFinished { .. }
            | Event::Checklist { .. }
            | Event::CortexFailed { .. }
            | Event::Quiet { .. }
            | Event::QuietHours { .. } => (!actions.is_empty()).then_some(
                "body, playback, checklist, failure and quiet events decide nothing",
            ),
        };
        broken.map_or(Ok(()), |what| Err(self.violation(event, what)))
    }

    /// What holds a discretionary thought back never touches an obligation, and each
    /// reason holds only when what it names was seen: a drive thinks only with something
    /// on the checklist, abstains as `NothingToCheck` only without it, and backs off only
    /// after a cortex failure that no reply has answered yet; no thought of Enton's own is
    /// bought in quiet mode or quiet hours, and each abstains for them only while they hold.
    fn check_discretion(&mut self, event: &Event, actions: &[Action]) -> Result<(), Error> {
        // Only outcomes of thoughts the organism issued count, as in the reducer.
        let issued = |thought: &ThoughtId| {
            thought.0 >= 1 && self.last_thought.is_some_and(|last| *thought <= last)
        };
        match event {
            Event::Checklist { actionable, .. } => self.checklist_actionable = *actionable,
            Event::CortexFailed { thought, .. } if issued(thought) => {
                self.failed_since_reply = true;
            }
            Event::CortexReply { thought, .. } if issued(thought) => {
                self.failed_since_reply = false;
            }
            Event::Quiet { now, until } => self.quiet_until = (until > now).then_some(*until),
            Event::QuietHours { active, .. } => self.quiet_hours = *active,
            _ => {}
        }
        let now = event.now();
        let quiet = self.quiet_until.is_some_and(|until| now < until);
        for action in actions {
            let (reason, why) = match action {
                Action::Think {
                    reason: Reason::Drive(_),
                    ..
                } if !self.checklist_actionable => {
                    return Err(
                        self.violation(event, "a drive thinks only with something to check")
                    );
                }
                Action::Think {
                    reason, propensity, ..
                } if (matches!(reason, Reason::Drive(_) | Reason::Speech)
                    || propensity.is_some())
                    && (quiet || self.quiet_hours) =>
                {
                    return Err(self.violation(
                        event,
                        "no thought of Enton's own is bought in quiet mode or quiet hours",
                    ));
                }
                Action::Abstain { reason, why, .. } => (reason, *why),
                _ => continue,
            };
            let discretionary = matches!(reason, Reason::Drive(_) | Reason::Speech);
            let broken = match why {
                Abstention::NothingToCheck => (!matches!(reason, Reason::Drive(_))
                    || self.checklist_actionable)
                    .then_some("only a drive abstains for nothing to check, and only without it"),
                Abstention::NobodyHome => (!discretionary)
                    .then_some("only a discretionary thought waits for somebody home"),
                Abstention::Backoff => (!discretionary || !self.failed_since_reply).then_some(
                    "only a discretionary thought backs off, and only after an unanswered failure",
                ),
                Abstention::Quiet => (!discretionary || !quiet)
                    .then_some("only a discretionary thought waits for quiet, and only in it"),
                Abstention::QuietHours => (!discretionary || !self.quiet_hours || quiet)
                    .then_some("only a discretionary thought waits for the quiet hours, in them"),
                _ => None,
            };
            if let Some(what) = broken {
                return Err(self.violation(event, what));
            }
        }
        Ok(())
    }

    /// A drive's intent is held only at a tick, while nothing but a conversation holds it
    /// back; it rides only an answer the owner asked for, and only the intent held before
    /// the step; it expires only at a tick, once its deadline has passed, and only once.
    fn check_initiative(
        &mut self,
        organism: &Organism,
        event: &Event,
        actions: &[Action],
    ) -> Result<(), Error> {
        let now = event.now();
        let before = self.deferred.take();
        let after = organism.deferred().cloned();
        let free = !organism.quiet_as_of(now)
            && !organism.in_quiet_hours()
            && self.checklist_actionable
            && !organism.is_torpid()
            && organism.owner_present_as_of(now);
        for action in actions {
            let broken = match action {
                Action::Think {
                    reason,
                    propensity,
                    rider: Some(drive),
                    ..
                } => {
                    if !matches!(reason, Reason::Keyword | Reason::FollowUp) || propensity.is_some()
                    {
                        Some("only an answer the owner asked for carries a drive's intent")
                    } else if !free {
                        Some("an intent rides only when its drive could think but for the talk")
                    } else if before.as_ref().is_none_or(|held| held.drive != *drive) {
                        Some("a ride carries the intent held before it")
                    } else {
                        None
                    }
                }
                Action::Abstain {
                    reason,
                    why: Abstention::Expired,
                    ..
                } => {
                    let expired = before.as_ref().is_some_and(|held| {
                        now >= held.expires && *reason == Reason::Drive(held.drive.clone())
                    });
                    (!matches!(event, Event::Tick { .. }) || !expired || after.is_some())
                        .then_some("an intent expires once, at a tick past its deadline")
                }
                _ => None,
            };
            if let Some(what) = broken {
                return Err(self.violation(event, what));
            }
        }
        if let Some(held) = &after
            && before.is_none()
            && (!matches!(event, Event::Tick { .. }) || held.since != now || !free)
        {
            return Err(self.violation(
                event,
                "an intent is held only at a tick, when its drive could think but for the talk",
            ));
        }
        self.deferred = after;
        Ok(())
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
        if organism
            .tv_direction_as_of(organism.last_seen())
            .is_some_and(|[x, y]| ((x * x + y * y).sqrt() - 1.0).abs() > 1e-5)
        {
            return Err(self.violation(event, "a learned TV direction is a unit vector"));
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

    /// A coin flip is logged only at a speech cue, with the profile's applied probability
    /// (a thought) or its complement (an abstention), and an explored thought is paid by
    /// the discretionary account alone.
    fn check_exploration(
        &mut self,
        organism: &Organism,
        event: &Event,
        actions: &[Action],
    ) -> Result<(), Error> {
        let before = self.budgets.replace((
            *organism.obligation_budget(),
            *organism.discretionary_budget(),
        ));
        let explore = organism.profile().exploration.applied_probability();
        for action in actions {
            let (logged, expected, explored) = match action {
                Action::Think {
                    propensity: Some(logged),
                    ..
                } => (*logged, explore, true),
                Action::Abstain {
                    propensity: Some(logged),
                    ..
                } => (*logged, 1.0 - explore, false),
                _ => continue,
            };
            if !matches!(event, Event::Speech { .. }) {
                return Err(self.violation(event, "only a speech cue flips the exploration coin"));
            }
            if logged.to_bits() != expected.to_bits() {
                return Err(self.violation(
                    event,
                    "a logged propensity is the applied exploration probability or its complement",
                ));
            }
            let cost = organism.profile().budgets.think_cost;
            let paid_alone = before.is_none_or(|(obligation, discretionary)| {
                organism.obligation_budget().available.to_bits() == obligation.available.to_bits()
                    && organism.discretionary_budget().available.to_bits()
                        == (discretionary.available - cost).to_bits()
            });
            if explored && !paid_alone {
                return Err(self.violation(
                    event,
                    "an explored thought is paid by the discretionary account alone",
                ));
            }
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
        Event::Checklist { .. } => "Checklist",
        Event::CortexFailed { .. } => "CortexFailed",
        Event::Quiet { .. } => "Quiet",
        Event::QuietHours { .. } => "QuietHours",
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
                directed: None,
                direction: None,
            },
        }
    }

    fn abstain() -> Action {
        Action::Abstain {
            reason: Reason::Speech,
            salience: 0.1,
            why: Abstention::BelowThreshold,
            propensity: None,
        }
    }

    fn think(id: u64) -> Action {
        Action::Think {
            thought: ThoughtId(id),
            reason: Reason::Keyword,
            salience: 1.5,
            propensity: None,
            rider: None,
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
    fn a_coin_flip_the_profile_did_not_make_is_a_violation() {
        // Exploration is off: a thought would log probability zero, an abstention one.
        let mut organism = organism();
        let first = cue(100);
        let mut watch = Watch::default();
        let actions = organism.step(&first);
        watch.after_step(&organism, &first, &actions).unwrap();
        let second = cue(200);
        organism.step(&second);
        let halved = Action::Abstain {
            reason: Reason::Speech,
            salience: 0.1,
            why: Abstention::BelowThreshold,
            propensity: Some(0.5),
        };
        let err = watch.after_step(&organism, &second, &[halved]).unwrap_err();
        assert!(
            err.to_string().contains("applied exploration probability"),
            "{err}"
        );

        // A thought logged as explored that the discretionary account did not pay for.
        let free = Action::Think {
            thought: ThoughtId(1),
            reason: Reason::FollowUp,
            salience: 1.0,
            propensity: Some(0.0),
            rider: None,
        };
        let err = watch.after_step(&organism, &second, &[free]).unwrap_err();
        assert!(
            err.to_string().contains("discretionary account alone"),
            "{err}"
        );

        let tick = Event::Tick { now: Millis(300) };
        organism.step(&tick);
        let on_a_tick = Action::Abstain {
            reason: Reason::Drive("social".into()),
            salience: 0.1,
            why: Abstention::Cooldown,
            propensity: Some(1.0),
        };
        let err = Watch::default()
            .after_step(&organism, &tick, &[on_a_tick])
            .unwrap_err();
        assert!(err.to_string().contains("only a speech cue"), "{err}");
    }

    #[test]
    fn checklist_and_failure_events_decide_nothing() {
        let mut organism = organism();
        let mut watch = Watch::default();
        for event in [
            Event::Checklist {
                now: Millis(100),
                actionable: true,
            },
            Event::CortexFailed {
                now: Millis(200),
                thought: ThoughtId(1),
            },
        ] {
            let actions = organism.step(&event);
            assert!(actions.is_empty());
            watch.after_step(&organism, &event, &actions).unwrap();
            let err = watch
                .after_step(&organism, &event, &[abstain()])
                .unwrap_err();
            assert!(err.to_string().contains("decide nothing"), "{err}");
        }
    }

    #[test]
    fn a_silent_reply_speaks_nothing() {
        let mut organism = organism();
        let first = cue(100);
        let mut watch = Watch::default();
        let actions = organism.step(&first);
        watch.after_step(&organism, &first, &actions).unwrap();
        let silent = Event::CortexReply {
            now: Millis(200),
            thought: ThoughtId(1),
            text: String::new(),
        };
        let actions = organism.step(&silent);
        assert!(actions.is_empty());
        watch.after_step(&organism, &silent, &actions).unwrap();
        let spoken = Action::Speak {
            text: String::new(),
        };
        let err = watch.after_step(&organism, &silent, &[spoken]).unwrap_err();
        assert!(err.to_string().contains("none when silent"), "{err}");
    }

    #[test]
    fn a_discretionary_gate_where_it_cannot_hold_is_a_violation() {
        let gated = |reason: Reason, why| Action::Abstain {
            reason,
            salience: 0.8,
            why,
            propensity: None,
        };
        let mut organism = organism();
        let event = cue(100);
        organism.step(&event);
        // Nothing to check holds back a drive only, never speech.
        let err = Watch::default()
            .after_step(
                &organism,
                &event,
                &[gated(Reason::Speech, Abstention::NothingToCheck)],
            )
            .unwrap_err();
        assert!(err.to_string().contains("nothing to check"), "{err}");
        // An obligation never waits for somebody home.
        let err = Watch::default()
            .after_step(
                &organism,
                &event,
                &[gated(Reason::Keyword, Abstention::NobodyHome)],
            )
            .unwrap_err();
        assert!(err.to_string().contains("somebody home"), "{err}");
        // Backing off takes a failure first.
        let err = Watch::default()
            .after_step(
                &organism,
                &event,
                &[gated(Reason::Speech, Abstention::Backoff)],
            )
            .unwrap_err();
        assert!(err.to_string().contains("backs off"), "{err}");

        // A drive thought with nothing on the checklist.
        let tick = Event::Tick { now: Millis(300) };
        organism.step(&tick);
        let drive = Action::Think {
            thought: ThoughtId(1),
            reason: Reason::Drive("curiosity".into()),
            salience: 0.8,
            propensity: None,
            rider: None,
        };
        let err = Watch::default()
            .after_step(&organism, &tick, &[drive])
            .unwrap_err();
        assert!(err.to_string().contains("something to check"), "{err}");
    }

    #[test]
    fn quiet_events_decide_nothing() {
        let mut organism = organism();
        let mut watch = Watch::default();
        for event in [
            Event::Quiet {
                now: Millis(100),
                until: Millis(3_600_100),
            },
            Event::QuietHours {
                now: Millis(200),
                active: true,
            },
        ] {
            let actions = organism.step(&event);
            assert!(actions.is_empty());
            watch.after_step(&organism, &event, &actions).unwrap();
            let err = watch
                .after_step(&organism, &event, &[abstain()])
                .unwrap_err();
            assert!(err.to_string().contains("decide nothing"), "{err}");
        }
    }

    #[test]
    fn quiet_that_the_events_do_not_support_is_a_violation() {
        let gated = |reason: Reason, why| Action::Abstain {
            reason,
            salience: 0.8,
            why,
            propensity: None,
        };
        let mut organism = organism();
        let event = cue(100);
        organism.step(&event);
        // Nobody asked for quiet, and the night band never began.
        let err = Watch::default()
            .after_step(
                &organism,
                &event,
                &[gated(Reason::Speech, Abstention::Quiet)],
            )
            .unwrap_err();
        assert!(err.to_string().contains("waits for quiet"), "{err}");
        let err = Watch::default()
            .after_step(
                &organism,
                &event,
                &[gated(Reason::Speech, Abstention::QuietHours)],
            )
            .unwrap_err();
        assert!(err.to_string().contains("quiet hours"), "{err}");

        // In quiet mode, speech overheard buys no thought, and an answer is no quiet wait.
        let mut watch = Watch::default();
        let hush = Event::Quiet {
            now: Millis(200),
            until: Millis(3_600_200),
        };
        organism.step(&hush);
        watch.after_step(&organism, &hush, &[]).unwrap();
        let event = cue(300);
        organism.step(&event);
        let overheard = Action::Think {
            thought: ThoughtId(1),
            reason: Reason::Speech,
            salience: 0.9,
            propensity: None,
            rider: None,
        };
        let err = watch
            .after_step(&organism, &event, &[overheard])
            .unwrap_err();
        assert!(err.to_string().contains("quiet mode or quiet hours"), "{err}");
        let err = watch
            .after_step(
                &organism,
                &event,
                &[gated(Reason::Keyword, Abstention::Quiet)],
            )
            .unwrap_err();
        assert!(err.to_string().contains("waits for quiet"), "{err}");
    }

    #[test]
    fn a_ride_or_an_expiry_without_a_held_intent_is_a_violation() {
        let mut organism = organism();
        organism.step(&Event::Checklist {
            now: Millis(0),
            actionable: true,
        });
        let mut watch = Watch::default();
        watch
            .after_step(
                &organism,
                &Event::Checklist {
                    now: Millis(0),
                    actionable: true,
                },
                &[],
            )
            .unwrap();
        let event = cue(100);
        organism.step(&event);
        let ride = |reason: Reason, propensity| Action::Think {
            thought: ThoughtId(1),
            reason,
            salience: 1.5,
            propensity,
            rider: Some("curiosity".into()),
        };
        // Nothing was held: nothing may ride.
        let err = watch
            .after_step(&organism, &event, &[ride(Reason::Keyword, None)])
            .unwrap_err();
        assert!(err.to_string().contains("intent held before it"), "{err}");
        // A discretionary thought, or an explored one, never carries an intent.
        for (reason, propensity) in [(Reason::Speech, None), (Reason::FollowUp, Some(0.5))] {
            let err = watch
                .after_step(&organism, &event, &[ride(reason, propensity)])
                .unwrap_err();
            assert!(err.to_string().contains("owner asked for"), "{err}");
        }
        // Nothing was held: nothing expires.
        let tick = Event::Tick { now: Millis(200) };
        organism.step(&tick);
        let expired = Action::Abstain {
            reason: Reason::Drive("curiosity".into()),
            salience: 0.8,
            why: Abstention::Expired,
            propensity: None,
        };
        let err = watch
            .after_step(&organism, &tick, &[expired])
            .unwrap_err();
        assert!(err.to_string().contains("expires once"), "{err}");
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
