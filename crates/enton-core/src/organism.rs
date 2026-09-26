//! The deterministic brainstem reducer.

use serde::{Deserialize, Serialize};

use crate::profile::{InvalidProfile, Profile};
use crate::{
    Abstention, Action, Budget, DriveTable, Event, Ignition, Millis, PriceTable, Reason, SpeechCue,
    ThoughtId, UtteranceId,
};

fn default_echo_energy_expectation() -> f32 {
    0.75
}

/// Running expectation of recent cues for novelty detection (RFC P5).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct CueFeatures {
    energy: f32,
    vad: f32,
    dur: f32,
}

/// A pending keyword-only turn awaiting continuation speech.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PendingAttend {
    until: Millis,
    salience: f32,
}

/// The version of the brainstem reducer and snapshot schema.
pub const REDUCER_VERSION: u32 = 5;

/// The physical playback / vocalization state of the organism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PlaybackStatus {
    /// Silent: not playing audio through the DAC.
    #[default]
    Idle,
    /// Actively playing an utterance on the DAC.
    Speaking {
        /// Utterance currently playing.
        utterance: UtteranceId,
        /// Monotonic time when playback started.
        started_at: Millis,
    },
    /// Playback finished; reverberation hangover clearing until deadline.
    Hangover {
        /// Utterance that just finished.
        utterance: UtteranceId,
        /// Monotonic deadline when hangover period ends.
        until: Millis,
    },
}

/// Pure state machine. Its time origin is [`Millis`] zero; only ticks advance
/// drives and refill energy. Repeated or backward ticks do not advance time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Organism {
    profile: Profile,
    drives: DriveTable,
    prices: PriceTable,
    obligation_budget: Budget,
    discretionary_budget: Budget,
    ignition: Ignition,
    last_tick: Millis,
    #[serde(default)]
    last_seen: Millis,
    torpor: bool,
    next_thought: u64,
    attention_until: Option<Millis>,
    pending_attend: Option<PendingAttend>,
    cue_expectation: Option<CueFeatures>,
    habituation: f32,
    #[serde(default)]
    slow_habituation: f32,
    #[serde(default)]
    conversation_thought: Option<ThoughtId>,
    #[serde(default)]
    playback_status: PlaybackStatus,
    #[serde(default)]
    self_speech_has_keyword: bool,
    #[serde(default = "default_echo_energy_expectation")]
    echo_energy_expectation: f32,
    #[serde(default)]
    consecutive_barge_ins: u32,
    #[serde(default)]
    speaking_for_obligation: bool,
}

impl Organism {
    /// Create a fresh organism with full obligation and discretionary budgets and thought IDs starting at one.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidProfile`] if the profile fails validation.
    pub fn new(profile: Profile) -> Result<Self, InvalidProfile> {
        profile.validate()?;
        let echo_initial_energy = profile.echo.echo_initial_energy;
        Ok(Self {
            drives: DriveTable::default_m1(),
            prices: PriceTable {
                think: profile.budgets.think_cost,
            },
            obligation_budget: Budget::new(profile.budgets.obligation_budget_per_hour),
            discretionary_budget: Budget::new(profile.budgets.discretionary_budget_per_hour),
            ignition: Ignition::new(
                profile.ignition.threshold,
                profile.ignition.hysteresis,
                profile.ignition.cooldown_ms,
                profile.ignition.ema_alpha,
            ),
            profile,
            last_tick: Millis(0),
            last_seen: Millis(0),
            torpor: false,
            next_thought: 1,
            attention_until: None,
            pending_attend: None,
            cue_expectation: None,
            habituation: 0.0,
            slow_habituation: 0.0,
            conversation_thought: None,
            playback_status: PlaybackStatus::Idle,
            self_speech_has_keyword: false,
            echo_energy_expectation: echo_initial_energy,
            consecutive_barge_ins: 0,
            speaking_for_obligation: false,
        })
    }

    /// The immutable policy used by this organism.
    #[must_use]
    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    /// Returns the current obligation budget.
    #[must_use]
    pub fn obligation_budget(&self) -> &Budget {
        &self.obligation_budget
    }

    /// Returns the current discretionary budget.
    #[must_use]
    pub fn discretionary_budget(&self) -> &Budget {
        &self.discretionary_budget
    }

    /// Active attention window deadline, if Enton is currently attending or in conversation.
    #[must_use]
    pub fn attention_until(&self) -> Option<Millis> {
        self.attention_until
    }

    /// Current habituation level to repeated non-addressed cues, from 0 to 1.
    #[must_use]
    pub fn habituation(&self) -> f32 {
        self.habituation
    }

    /// Long-term habituation, from 0 to 1: it decays slowly and a novel cue does
    /// not erase it, but it only suppresses cues similar to the habituated one.
    #[must_use]
    pub fn slow_habituation(&self) -> f32 {
        self.slow_habituation
    }

    /// Whether Enton is actively waiting for a continuation of a keyword-only turn.
    #[must_use]
    pub fn is_attending(&self) -> bool {
        self.pending_attend.is_some()
    }

    /// Active conversation thought awaiting reply, if any.
    #[must_use]
    pub fn conversation_thought(&self) -> Option<ThoughtId> {
        self.conversation_thought
    }

    /// Returns the current playback status.
    #[must_use]
    pub fn playback_status(&self) -> PlaybackStatus {
        self.playback_status
    }

    /// Whether Enton is actively vocalizing audio through the DAC.
    #[must_use]
    pub fn is_speaking(&self) -> bool {
        matches!(self.playback_status, PlaybackStatus::Speaking { .. })
    }

    /// Whether Enton is in the post-playback acoustic hangover window.
    #[must_use]
    pub fn is_hangover(&self) -> bool {
        matches!(self.playback_status, PlaybackStatus::Hangover { .. })
    }

    /// Expected echo energy under the adaptive forward model.
    #[must_use]
    pub fn echo_energy_expectation(&self) -> f32 {
        self.echo_energy_expectation
    }

    /// Whether Enton's currently playing or pending utterance includes its own name.
    #[must_use]
    pub fn self_speech_has_keyword(&self) -> bool {
        self.self_speech_has_keyword
    }

    /// Whether the most recent paid thought answered an obligation turn (keyword or follow-up)
    /// rather than a discretionary thought.
    #[must_use]
    pub fn speaking_for_obligation(&self) -> bool {
        self.speaking_for_obligation
    }

    /// The latest instant this organism has observed, from any event.
    ///
    /// A restored organism must resume its clock here. The reducer ignores time
    /// that runs backward, so a clock restarting at zero would freeze drives,
    /// budget refills and cooldowns until it caught up with the old uptime.
    #[must_use]
    pub fn last_seen(&self) -> Millis {
        self.last_seen.max(self.last_tick)
    }

    /// Consume a recorded event and return decisions without executing effects.
    pub fn step(&mut self, event: &Event) -> Vec<Action> {
        self.last_seen = self.last_seen.max(event.now());
        match event {
            Event::Tick { now } => self.tick(*now),
            Event::Body { signals, .. } => {
                self.torpor = signals
                    .temperature_c
                    .is_some_and(|value| value >= self.profile.body.fever_c)
                    || signals
                        .battery
                        .is_some_and(|value| value <= self.profile.body.lethargy_battery);
                Vec::new()
            }
            Event::Speech { now, cue } => vec![self.speech(*now, cue)],
            Event::CortexReply { now, text, thought } => {
                self.drives.satisfy("social", 0.3);
                self.self_speech_has_keyword = contains_keyword_word(text, "enton");
                if Some(*thought) == self.conversation_thought {
                    self.conversation_thought = None;
                    let new_until =
                        Millis(now.0.saturating_add(self.profile.attention.attention_ms));
                    if new_until > self.last_tick {
                        self.attention_until = Some(match self.attention_until {
                            Some(current) => current.max(new_until),
                            None => new_until,
                        });
                    }
                }
                vec![Action::Speak { text: text.clone() }]
            }
            Event::PlaybackStarted { now, utterance } => {
                self.playback_status = PlaybackStatus::Speaking {
                    utterance: *utterance,
                    started_at: *now,
                };
                // Voice mode precedence: playback supersedes text anchor and suspends the window
                self.attention_until = None;
                Vec::new()
            }
            Event::PlaybackFinished { now, utterance } => {
                if let PlaybackStatus::Speaking {
                    utterance: active, ..
                } = self.playback_status
                    && active == *utterance
                {
                    let hangover_until =
                        Millis(now.0.saturating_add(self.profile.echo.echo_hangover_ms));
                    self.playback_status = PlaybackStatus::Hangover {
                        utterance: *utterance,
                        until: hangover_until,
                    };
                    if self.speaking_for_obligation {
                        self.attention_until = Some(Millis(
                            hangover_until
                                .0
                                .saturating_add(self.profile.attention.attention_ms),
                        ));
                    }
                }
                Vec::new()
            }
        }
    }

    fn advance_playback_status(&mut self, now: Millis) {
        match self.playback_status {
            PlaybackStatus::Speaking { started_at, .. } => {
                if now.since(started_at) >= self.profile.echo.max_playback_ms {
                    // Watchdog: terminate playback to prevent permanent deafness
                    self.playback_status = PlaybackStatus::Idle;
                    self.self_speech_has_keyword = false;
                }
            }
            PlaybackStatus::Hangover { until, .. } => {
                if now >= until {
                    self.playback_status = PlaybackStatus::Idle;
                    self.self_speech_has_keyword = false;
                }
            }
            PlaybackStatus::Idle => {}
        }
    }

    fn is_in_echo_period(&self, now: Millis) -> bool {
        match self.playback_status {
            PlaybackStatus::Speaking { started_at, .. } => {
                now.since(started_at) < self.profile.echo.max_playback_ms
            }
            PlaybackStatus::Hangover { until, .. } => now < until,
            PlaybackStatus::Idle => false,
        }
    }

    fn tick(&mut self, now: Millis) -> Vec<Action> {
        let dt_ms = now.since(self.last_tick);
        if dt_ms > 0 {
            self.last_tick = now;

            // 1. Metabolic updates applied BEFORE any decision at this timestamp (A7)
            self.drives.advance(dt_ms);
            self.obligation_budget
                .refill(dt_ms, self.profile.budgets.obligation_budget_per_hour);
            self.discretionary_budget
                .refill(dt_ms, self.profile.budgets.discretionary_budget_per_hour);
            self.ignition.advance(self.drives.pressure());

            // Decay both habituation components with their half-lives on tick
            self.habituation = decay(
                self.habituation,
                dt_ms,
                self.profile.habituation.habituation_decay_half_life_ms,
            );
            self.slow_habituation = decay(
                self.slow_habituation,
                dt_ms,
                self.profile.habituation.slow_habituation_half_life_ms,
            );
        }

        // Advance playback status (watchdog and hangover expiration)
        self.advance_playback_status(now);

        let mut actions = Vec::new();

        // 2. Evaluate keyword timeout exactly once (A7)
        if self.pending_attend.as_ref().is_some_and(|p| now >= p.until)
            && let Some(pending) = self.pending_attend.take()
        {
            // Window closed with no continuation: think then ("Enton?" alone still gets an answer)
            // Obligation turn: spends obligation_budget; keyword bypasses torpor
            let action = self.pay_and_think_obligation(now, Reason::Keyword, pending.salience);
            if let Action::Think { thought, .. } = action {
                self.conversation_thought = Some(thought);
            }
            actions.push(action);
        }

        // Close attention window if deadline passed
        if self.attention_until.is_some_and(|until| now >= until) {
            self.attention_until = None;
        }

        if !self.ignition.drive_ready(now) {
            return actions;
        }
        let Some(drive) = self.drives.strongest() else {
            return actions;
        };
        let reason = Reason::Drive(drive.name.clone());
        let salience = self.ignition.salience();
        actions.push(if self.torpor {
            Action::Abstain {
                reason,
                salience,
                why: Abstention::Torpor,
            }
        } else {
            // Discretionary drive thought spends discretionary budget
            self.pay_and_think_discretionary(now, reason, salience)
        });
        actions
    }

    fn speech_during_echo(
        &mut self,
        now: Millis,
        cue: &SpeechCue,
        norm_energy: f32,
        norm_vad: f32,
        norm_dur: f32,
    ) -> Action {
        let is_barge_in = if cue.keyword {
            if self.self_speech_has_keyword {
                // Predicted keyword in self-speech: requires full double-talk margin
                norm_energy > self.echo_energy_expectation + self.profile.echo.echo_barge_in_margin
            } else {
                // Unpredicted keyword: requires reduced margin over expected echo to reject TTS phonetic false positives
                norm_energy
                    >= self.echo_energy_expectation + self.profile.echo.keyword_barge_in_margin
            }
        } else {
            // Non-keyword speech cue: requires full double-talk margin, minimal follow-up VAD
            // and the addressed speaker's voice (anyone may still interrupt by name)
            norm_energy > self.echo_energy_expectation + self.profile.echo.echo_barge_in_margin
                && norm_vad >= self.profile.attention.follow_up_min_vad
                && self.is_addressed_speaker(cue)
        };

        if !is_barge_in {
            // Stimulus rejected as self-echo: adapt forward model on rejected cues only,
            // guarding against non-finite (NaN, inf) values from upstream audio bugs.
            if cue.energy.is_finite() {
                let alpha = self.profile.echo.echo_learning_rate;
                self.echo_energy_expectation = (self.echo_energy_expectation
                    + alpha * (norm_energy - self.echo_energy_expectation))
                    .clamp(0.0, 1.0);
            }

            let base_salience = self.calculate_base_salience(norm_energy, norm_vad, norm_dur);
            let salience = if cue.keyword {
                base_salience + 1.0
            } else {
                base_salience
            };
            let reason = if cue.keyword {
                Reason::Keyword
            } else {
                Reason::FollowUp
            };
            return Action::Abstain {
                reason,
                salience,
                why: Abstention::SelfEcho,
            };
        }

        // Legitimate barge-in accepted: cancel playback, transition to hangover for residual tail
        let utterance = match self.playback_status {
            PlaybackStatus::Speaking { utterance, .. }
            | PlaybackStatus::Hangover { utterance, .. } => utterance,
            PlaybackStatus::Idle => UtteranceId(0),
        };
        let hangover_until = Millis(now.0.saturating_add(self.profile.echo.echo_hangover_ms));
        self.playback_status = PlaybackStatus::Hangover {
            utterance,
            until: hangover_until,
        };

        // Consecutive barge-in ratchet: breaks runaway loops if initial echo was underestimated
        self.consecutive_barge_ins += 1;
        if self.consecutive_barge_ins >= 2 {
            self.echo_energy_expectation = self.echo_energy_expectation.max(norm_energy);
        }

        if cue.keyword {
            let base_salience = self.calculate_base_salience(norm_energy, norm_vad, norm_dur) + 1.0;
            self.attention_until = Some(Millis(
                now.0.saturating_add(self.profile.attention.attention_ms),
            ));
            let action = self.pay_and_think_obligation(now, Reason::Keyword, base_salience);
            if let Action::Think { thought, .. } = action {
                self.conversation_thought = Some(thought);
            }
            return action;
        }

        let salience = self.calculate_base_salience(norm_energy, norm_vad, norm_dur);
        if self.torpor {
            return Action::Abstain {
                reason: Reason::FollowUp,
                salience,
                why: Abstention::Torpor,
            };
        }
        self.attention_until = Some(Millis(
            now.0.saturating_add(self.profile.attention.attention_ms),
        ));
        let action = self.pay_and_think_obligation(now, Reason::FollowUp, salience);
        if let Action::Think { thought, .. } = action {
            self.conversation_thought = Some(thought);
        }
        action
    }

    fn speech(&mut self, now: Millis, cue: &SpeechCue) -> Action {
        self.advance_playback_status(now);

        let norm_energy = normalized(cue.energy);
        let norm_vad = normalized(cue.vad_confidence);
        let norm_dur = (cue
            .duration_ms
            .min(self.profile.salience.salience_duration_max_ms) as f32)
            / (self.profile.salience.salience_duration_max_ms as f32);

        if self.is_in_echo_period(now) {
            return self.speech_during_echo(now, cue, norm_energy, norm_vad, norm_dur);
        }

        // Quiet period (outside playback and hangover): user speech resets consecutive barge-in ratchet
        self.consecutive_barge_ins = 0;

        // Case 1: Addressed by keyword
        if cue.keyword {
            let base_salience = self.calculate_base_salience(norm_energy, norm_vad, norm_dur) + 1.0;
            // Addressed speech is never habituated
            if cue.duration_ms < self.profile.attention.keyword_only_ms {
                let until = Millis(now.0.saturating_add(self.profile.attention.attention_ms));
                self.attention_until = Some(until);
                self.pending_attend = Some(PendingAttend {
                    until,
                    salience: base_salience,
                });
                self.conversation_thought = None;
                return Action::Attend { until };
            }

            self.pending_attend = None;
            self.attention_until = Some(Millis(
                now.0.saturating_add(self.profile.attention.attention_ms),
            ));
            // Keyword bypasses torpor and non-keyword cooldown; spends obligation_budget
            let action = self.pay_and_think_obligation(now, Reason::Keyword, base_salience);
            if let Action::Think { thought, .. } = action {
                self.conversation_thought = Some(thought);
            }
            return action;
        }

        // Case 2: Inside active attention window
        let in_attention_window = self.attention_until.is_some_and(|until| now < until);
        if in_attention_window {
            if !self.is_addressed_speaker(cue) {
                // Someone else talking inside the window neither continues the turn nor
                // extends the window; a pending "Enton?" keeps waiting for its speaker.
                return Action::Abstain {
                    reason: Reason::FollowUp,
                    salience: self.calculate_base_salience(norm_energy, norm_vad, norm_dur),
                    why: Abstention::OtherSpeaker,
                };
            }
            // Addressed speech is never habituated
            if let Some(pending) = self.pending_attend.take() {
                if norm_vad < self.profile.attention.follow_up_min_vad {
                    // Not valid continuation: restore pending attend and record abstention
                    self.pending_attend = Some(pending);
                    let salience = self.calculate_base_salience(norm_energy, norm_vad, norm_dur);
                    return Action::Abstain {
                        reason: Reason::FollowUp,
                        salience,
                        why: Abstention::BelowThreshold,
                    };
                }

                // Valid continuation: produces a single Think with Reason::Keyword covering both segments
                let continuation_salience =
                    self.calculate_base_salience(norm_energy, norm_vad, norm_dur);
                let combined_salience = pending.salience.max(continuation_salience + 1.0);
                self.attention_until = Some(Millis(
                    now.0.saturating_add(self.profile.attention.attention_ms),
                ));
                let action = self.pay_and_think_obligation(now, Reason::Keyword, combined_salience);
                if let Action::Think { thought, .. } = action {
                    self.conversation_thought = Some(thought);
                }
                return action;
            }

            // Normal follow-up in an ongoing conversation
            let salience = self.calculate_base_salience(norm_energy, norm_vad, norm_dur);
            if self.torpor {
                return Action::Abstain {
                    reason: Reason::FollowUp,
                    salience,
                    why: Abstention::Torpor,
                };
            }
            if norm_vad < self.profile.attention.follow_up_min_vad {
                return Action::Abstain {
                    reason: Reason::FollowUp,
                    salience,
                    why: Abstention::BelowThreshold,
                };
            }

            // Follow-up needs only minimal VAD, bypasses non-keyword cooldown, pays obligation_budget
            let action = self.pay_and_think_obligation(now, Reason::FollowUp, salience);
            if let Action::Think { thought, .. } = action {
                self.conversation_thought = Some(thought);
                self.attention_until = Some(Millis(
                    now.0.saturating_add(self.profile.attention.attention_ms),
                ));
            }
            return action;
        }

        // Case 3: Non-addressed speech outside attention window
        self.speech_unaddressed(now, norm_energy, norm_vad, norm_dur)
    }

    fn speech_unaddressed(
        &mut self,
        now: Millis,
        norm_energy: f32,
        norm_vad: f32,
        norm_dur: f32,
    ) -> Action {
        let (novelty, similarity) =
            self.update_novelty_and_similarity(norm_energy, norm_vad, norm_dur);

        // A sufficient prediction error resets habituation (A9)
        if similarity <= self.profile.habituation.similarity_cutoff {
            self.habituation = 0.0;
        }

        let base_salience = self.calculate_base_salience(norm_energy, norm_vad, norm_dur);
        let salience_with_novelty = if norm_vad <= 0.4 {
            // Sound with VAD <= 0.4 never reaches threshold
            base_salience.min(self.profile.ignition.discretionary_threshold - 0.01)
        } else {
            base_salience + novelty
        };

        // Habituation suppresses in proportion to similarity with habituated expectation (A9).
        // The long-term component is stimulus-specific: it never touches a novel cue.
        let familiar = similarity > self.profile.habituation.similarity_cutoff;
        let long_term = if familiar { self.slow_habituation } else { 0.0 };
        let suppression = (self.habituation + long_term).min(1.0) * similarity;
        let effective_salience = (salience_with_novelty - suppression).max(0.0);

        // Update habituation for subsequent cues if stimulus is similar to running expectation
        if familiar {
            let hab_inc = (similarity - self.profile.habituation.similarity_cutoff)
                * self.profile.habituation.habituation_step;
            self.habituation = (self.habituation + hab_inc).clamp(0.0, 1.0);
            self.slow_habituation = (self.slow_habituation
                + hab_inc * self.profile.habituation.slow_habituation_rate)
                .clamp(0.0, 1.0);
        }

        if self.torpor {
            return Action::Abstain {
                reason: Reason::Speech,
                salience: effective_salience,
                why: Abstention::Torpor,
            };
        }

        if salience_with_novelty < self.profile.ignition.discretionary_threshold {
            return Action::Abstain {
                reason: Reason::Speech,
                salience: effective_salience,
                why: Abstention::BelowThreshold,
            };
        }

        if self.ignition.in_cooldown(now) {
            return Action::Abstain {
                reason: Reason::Speech,
                salience: effective_salience,
                why: Abstention::Cooldown,
            };
        }

        if effective_salience < self.profile.ignition.discretionary_threshold {
            return Action::Abstain {
                reason: Reason::Speech,
                salience: effective_salience,
                why: Abstention::Habituation,
            };
        }

        self.pay_and_think_discretionary(now, Reason::Speech, effective_salience)
    }

    /// Whether a cue may speak for the current turn: a matching voice, or no verification.
    fn is_addressed_speaker(&self, cue: &SpeechCue) -> bool {
        cue.speaker_sim
            .is_none_or(|sim| sim >= self.profile.attention.follow_up_min_speaker_sim)
    }

    fn calculate_base_salience(&self, norm_energy: f32, norm_vad: f32, norm_dur: f32) -> f32 {
        self.profile.salience.salience_vad_weight * norm_vad
            + self.profile.salience.salience_energy_weight * norm_energy
            + self.profile.salience.salience_duration_weight * norm_dur
    }

    fn update_novelty_and_similarity(
        &mut self,
        norm_energy: f32,
        norm_vad: f32,
        norm_dur: f32,
    ) -> (f32, f32) {
        if let Some(exp) = &mut self.cue_expectation {
            let diff_energy = (norm_energy - exp.energy).abs();
            let diff_vad = (norm_vad - exp.vad).abs();
            let diff_dur = (norm_dur - exp.dur).abs();
            let error = (diff_energy + diff_vad + diff_dur) / 3.0;
            let similarity = (1.0 - error).clamp(0.0, 1.0);
            let novelty = (error * self.profile.salience.novelty_weight)
                .min(self.profile.salience.novelty_max);

            let alpha = self.profile.habituation.expectation_coefficient;
            exp.energy += alpha * (norm_energy - exp.energy);
            exp.vad += alpha * (norm_vad - exp.vad);
            exp.dur += alpha * (norm_dur - exp.dur);

            (novelty, similarity)
        } else {
            self.cue_expectation = Some(CueFeatures {
                energy: norm_energy,
                vad: norm_vad,
                dur: norm_dur,
            });
            (self.profile.salience.novelty_max, 0.0)
        }
    }

    fn pay_and_think_obligation(&mut self, now: Millis, reason: Reason, salience: f32) -> Action {
        if !self.obligation_budget.try_spend(self.prices.think) {
            return Action::Abstain {
                reason,
                salience,
                why: Abstention::OutOfEnergy,
            };
        }
        let thought = ThoughtId(self.next_thought);
        self.next_thought += 1;
        self.ignition.fired(now);
        self.speaking_for_obligation = true;
        Action::Think {
            thought,
            reason,
            salience,
        }
    }

    fn pay_and_think_discretionary(
        &mut self,
        now: Millis,
        reason: Reason,
        salience: f32,
    ) -> Action {
        if !self.discretionary_budget.try_spend(self.prices.think) {
            return Action::Abstain {
                reason,
                salience,
                why: Abstention::OutOfEnergy,
            };
        }
        let thought = ThoughtId(self.next_thought);
        self.next_thought += 1;
        self.ignition.fired(now);
        self.speaking_for_obligation = false;
        Action::Think {
            thought,
            reason,
            salience,
        }
    }
}

/// Exponential decay of `level` over `dt_ms` with the given half-life.
fn decay(level: f32, dt_ms: u64, half_life_ms: u64) -> f32 {
    if level <= 0.0 {
        return 0.0;
    }
    let factor = (-std::f32::consts::LN_2 * (dt_ms as f32) / half_life_ms as f32).exp();
    (level * factor).clamp(0.0, 1.0)
}

fn normalized(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Checks whether `text` contains `keyword` as a whole word, delimited by non-alphanumeric Unicode characters.
#[must_use]
pub fn contains_keyword_word(text: &str, keyword: &str) -> bool {
    text.split(|c: char| !c.is_alphanumeric())
        .any(|word| word.eq_ignore_ascii_case(keyword))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_validation() {
        assert!(Profile::t1_ref().validate().is_ok());
        assert!(Profile::desktop().validate().is_ok());

        let mut invalid = Profile::t1_ref();
        invalid.ignition.hysteresis = invalid.ignition.threshold;
        assert_eq!(
            invalid.validate(),
            Err(InvalidProfile {
                name: invalid.name.clone(),
            })
        );
        assert_eq!(
            Organism::new(invalid.clone()),
            Err(InvalidProfile { name: invalid.name })
        );
    }
}
