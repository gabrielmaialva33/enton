//! The deterministic brainstem reducer.

use serde::{Deserialize, Serialize};

use crate::profile::{InvalidProfile, Profile};
use crate::{
    Abstention, Action, Budget, DriveTable, Event, Evidence, Ignition, Millis, PriceTable, Reason,
    SpeechCue, ThoughtId, UtteranceId,
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
    /// When the name segment ended, to tell a continuation that follows right after it.
    #[serde(default)]
    name_ended_at: Millis,
    /// The evidence, finished over unfinished in nats, by which the name was judged
    /// unfinished, when an end-of-turn model ran.
    #[serde(default)]
    name_finished: Option<f32>,
}

/// What stands against a cue continuing the owner's turn.
struct Vetoes {
    /// The tagger heard a loudspeaker.
    media: bool,
    /// Voice and source together rule out the owner speaking live.
    other_voice: bool,
    /// The voice alone rules out the owner: an objection independent of the
    /// tagger's, so closeness to an unfinished name excuses one sensor, never two.
    voice_alone: bool,
}

/// A continuation may start this much before the name's recorded end: endpoint jitter.
const CONTINUATION_JITTER_MS: u64 = 50;

/// The version of the brainstem reducer and snapshot schema.
pub const REDUCER_VERSION: u32 = 9;

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
    #[serde(default)]
    verified_attention_until: Option<Millis>,
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
    /// Belief, from zero to one, that a TV (or radio) is playing, as of `tv_heard_at`.
    #[serde(default)]
    tv_presence: f32,
    #[serde(default)]
    tv_heard_at: Millis,
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
            verified_attention_until: None,
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
            tv_presence: 0.0,
            tv_heard_at: Millis(0),
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

    /// Deadline of the longer window open only to the verified voice of whoever
    /// addressed Enton, if one is open.
    #[must_use]
    pub fn verified_attention_until(&self) -> Option<Millis> {
        self.verified_attention_until
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

    /// Belief, from zero to one, that a TV is playing, as of the last overheard line.
    #[must_use]
    pub fn tv_presence(&self) -> f32 {
        self.tv_presence
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
            Event::Speech { now, cue } => vec![self.speech(*now, &cue.canonical())],
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
                        let verified = Millis(
                            now.0
                                .saturating_add(self.profile.attention.verified_attention_ms),
                        );
                        self.verified_attention_until = Some(
                            self.verified_attention_until
                                .map_or(verified, |current| current.max(verified)),
                        );
                    }
                }
                vec![Action::Speak { text: text.clone() }]
            }
            Event::PlaybackStarted { now, utterance } => {
                self.playback_status = PlaybackStatus::Speaking {
                    utterance: *utterance,
                    started_at: *now,
                };
                // Voice mode precedence: playback supersedes text anchor and suspends the windows
                self.attention_until = None;
                self.verified_attention_until = None;
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
                        self.open_attention(hangover_until);
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
        if self
            .verified_attention_until
            .is_some_and(|until| now >= until)
        {
            self.verified_attention_until = None;
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
        if !cue.keyword && self.is_media(cue) {
            // Media over Enton's playback is neither a barge-in nor evidence about
            // the echo path: the forward model must not learn the TV's loudness.
            return Action::Abstain {
                reason: Reason::FollowUp,
                salience: self.calculate_base_salience(norm_energy, norm_vad, norm_dur),
                why: Abstention::Media,
            };
        }
        let echo = self.profile.echo;
        // Three separate pieces of evidence: loud enough over the expected echo, speech-like
        // enough, and a voice allowed to interrupt. Only loudness says anything about the
        // echo path, so only a cue that failed it may train the echo model.
        let (loud_enough, speech_like, voice_allowed) = if cue.keyword {
            let loud = if self.self_speech_has_keyword || self.is_unverified_voice(cue) {
                // Predicted keyword in self-speech, or a voice that verification did not
                // confirm as the owner's (while Enton talks, its own echo is the likeliest
                // voice): requires full double-talk margin
                norm_energy > self.echo_energy_expectation + echo.echo_barge_in_margin
            } else {
                // Unpredicted keyword: requires reduced margin over expected echo to reject TTS phonetic false positives
                norm_energy >= self.echo_energy_expectation + echo.keyword_barge_in_margin
            };
            // Anyone may interrupt by name.
            (loud, true, true)
        } else {
            // Non-keyword speech cue: double-talk margin (smaller for the caller's verified
            // voice), minimal follow-up VAD, and the addressed speaker's voice
            let margin = if self.is_verified_speaker(cue) {
                echo.verified_barge_in_margin
            } else {
                echo.echo_barge_in_margin
            };
            (
                norm_energy > self.echo_energy_expectation + margin,
                norm_vad >= self.profile.attention.follow_up_min_vad,
                !self.is_unverified_voice(cue),
            )
        };
        let is_barge_in = loud_enough && speech_like && voice_allowed;

        if !is_barge_in {
            // Only a cue that failed the loudness test is evidence about the echo path;
            // one rejected for its voice must not teach the model the caller's loudness.
            // A zero reading (including a non-finite one, canonicalized to zero so replay
            // decides the same) carries no evidence either.
            if !loud_enough && norm_energy > 0.0 {
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
            let why = if loud_enough && speech_like {
                Abstention::OtherSpeaker
            } else {
                Abstention::SelfEcho
            };
            return Action::Abstain {
                reason,
                salience,
                why,
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
            // The playback is cut; the name itself is judged as it would be in silence:
            // a whole request is answered, an unfinished "Enton..." waits for the rest.
            return self.speech_addressed(now, cue, norm_energy, norm_vad, norm_dur);
        }

        let salience = self.calculate_base_salience(norm_energy, norm_vad, norm_dur);
        if self.torpor {
            return Action::Abstain {
                reason: Reason::FollowUp,
                salience,
                why: Abstention::Torpor,
            };
        }
        self.open_attention(now);
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
        // Outside Enton's own playback, a loudspeaker voice is someone else's: the TV.
        // Only speech says so; a click or a hum carries no voice to judge.
        if !cue.keyword && norm_vad >= self.profile.attention.follow_up_min_vad {
            self.track_tv(now, cue);
        }

        if cue.keyword {
            return self.speech_addressed(now, cue, norm_energy, norm_vad, norm_dur);
        }
        if self.is_in_attention_window(now, cue) {
            return self.speech_in_window(now, cue, norm_energy, norm_vad, norm_dur);
        }

        // Overheard speech, outside any attention window: the TV makes a loudspeaker
        // likelier here too.
        let media = self.window_vetoes(now, cue).media;
        self.speech_unaddressed(now, media, norm_energy, norm_vad, norm_dur)
    }

    /// Enton was called by name: answer now, or wait for the rest of the request.
    fn speech_addressed(
        &mut self,
        now: Millis,
        cue: &SpeechCue,
        norm_energy: f32,
        norm_vad: f32,
        norm_dur: f32,
    ) -> Action {
        let base_salience = self.calculate_base_salience(norm_energy, norm_vad, norm_dur) + 1.0;
        if self.pending_attend.is_some()
            && self.is_known_other_speaker(cue)
            && !self.is_whole_request(cue)
        {
            // Someone else saying the name does not take over a caller's unfinished turn.
            return Action::Abstain {
                reason: Reason::Keyword,
                salience: base_salience,
                why: Abstention::OtherSpeaker,
            };
        }
        // Addressed speech is never habituated
        if !self.is_whole_request(cue) {
            let until = Millis(now.0.saturating_add(self.profile.attention.attention_ms));
            self.open_attention(now);
            self.pending_attend = Some(PendingAttend {
                until,
                salience: base_salience,
                name_ended_at: now,
                name_finished: cue.turn_complete.map(|_| self.whole_request_log_odds(cue)),
            });
            self.conversation_thought = None;
            return Action::Attend { until };
        }

        self.pending_attend = None;
        self.open_attention(now);
        // Keyword bypasses torpor and non-keyword cooldown; spends obligation_budget
        let action = self.pay_and_think_obligation(now, Reason::Keyword, base_salience);
        if let Action::Think { thought, .. } = action {
            self.conversation_thought = Some(thought);
        }
        action
    }

    /// Whether a cue falls inside an attention window: the short one for any voice,
    /// the long one only for the caller's verified voice.
    fn is_in_attention_window(&self, now: Millis, cue: &SpeechCue) -> bool {
        self.attention_until.is_some_and(|until| now < until)
            || (self
                .verified_attention_until
                .is_some_and(|until| now < until)
                && self.is_verified_speaker(cue))
    }

    /// Speech inside an attention window: a continuation, a follow-up, or someone else.
    fn speech_in_window(
        &mut self,
        now: Millis,
        cue: &SpeechCue,
        norm_energy: f32,
        norm_vad: f32,
        norm_dur: f32,
    ) -> Action {
        // Closeness to an unfinished name outweighs one sensor's veto, never both: a cue
        // that the media tagger and the speaker check both reject is someone else.
        let Vetoes {
            media,
            other_voice,
            voice_alone,
        } = self.window_vetoes(now, cue);
        let excused = self.is_adjacent_continuation(now, cue, norm_vad) && !(media && voice_alone);
        if !excused && media {
            // Reproduced media inside the window neither continues nor extends it.
            return Action::Abstain {
                reason: Reason::FollowUp,
                salience: self.calculate_base_salience(norm_energy, norm_vad, norm_dur),
                why: Abstention::Media,
            };
        }
        if !excused && other_voice {
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
            let turn = self.turn_evidence(cue);
            if turn.is_some_and(|finished| finished < 0.0) {
                // The request is still going ("Enton, você pode... hã..."): keep waiting,
                // now anchored at this segment's end.
                let until = Millis(now.0.saturating_add(self.profile.attention.attention_ms));
                self.open_attention(now);
                self.pending_attend = Some(PendingAttend {
                    until,
                    salience: combined_salience,
                    name_ended_at: now,
                    name_finished: turn,
                });
                return Action::Attend { until };
            }
            self.open_attention(now);
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
            self.open_attention(now);
        }
        action
    }

    fn speech_unaddressed(
        &mut self,
        now: Millis,
        media: bool,
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

        // Media still trains habituation above, so the few TV cues the tagger misses
        // arrive already familiar; tagged ones never buy a thought.
        if media {
            return Action::Abstain {
                reason: Reason::Speech,
                salience: effective_salience,
                why: Abstention::Media,
            };
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

    /// Open (or reopen) the attention windows at `from`: the short one for any cue,
    /// the long one for the verified voice of whoever addressed Enton.
    fn open_attention(&mut self, from: Millis) {
        let policy = self.profile.attention;
        self.attention_until = Some(Millis(from.0.saturating_add(policy.attention_ms)));
        self.verified_attention_until =
            Some(Millis(from.0.saturating_add(policy.verified_attention_ms)));
    }

    /// What the calibrated sensors say about a cue.
    fn evidence(&self, cue: &SpeechCue) -> Evidence {
        self.profile.senses.read(cue)
    }

    /// End-of-turn evidence, finished over unfinished, when an end-of-turn model ran.
    fn turn_evidence(&self, cue: &SpeechCue) -> Option<f32> {
        cue.turn_complete
            .map(|_| self.evidence(cue).finished_over_unfinished)
    }

    /// Whether the evidence verifies the owner speaking live (not merely unknown).
    fn is_verified_speaker(&self, cue: &SpeechCue) -> bool {
        self.evidence(cue).owner_live() >= self.profile.attention.verified_voice_llr
    }

    /// Whether a keyword cue already carries a whole request. Its length says how
    /// likely that is (the name alone is short) and an end-of-turn model adds its
    /// evidence, except for a voice known to be someone else or reproduced media,
    /// whose turn is not Enton's to judge: then length alone decides, so a name
    /// dropped mid-sentence by another person waits instead of buying a thought.
    /// Without the model, a cue shorter than `keyword_only_ms` is the name alone
    /// ("Enton?") and Enton waits.
    fn is_whole_request(&self, cue: &SpeechCue) -> bool {
        self.whole_request_log_odds(cue) >= 0.0
    }

    /// Log-odds, in nats, that a keyword cue holds a whole request: its length,
    /// plus the end-of-turn evidence when the voice is the owner's to judge.
    fn whole_request_log_odds(&self, cue: &SpeechCue) -> f32 {
        let attention = self.profile.attention;
        let beyond_name_s = (cue.duration_ms as f32 - attention.keyword_only_ms as f32) / 1_000.0;
        let length = attention.whole_request_llr_per_s * beyond_name_s;
        let turn = if self.is_known_other_speaker(cue) || self.is_media(cue) {
            0.0
        } else {
            self.evidence(cue).finished_over_unfinished
        };
        length + turn
    }

    /// Whether the cue is the rest of a name left unfinished just before: the name
    /// read as unfinished, this cue reads as finished, and it starts within
    /// `continuation_gap_ms` of the name's end (allowing endpointing jitter).
    fn is_adjacent_continuation(&self, now: Millis, cue: &SpeechCue, norm_vad: f32) -> bool {
        let attention = self.profile.attention;
        let Some(pending) = &self.pending_attend else {
            return false;
        };
        let name_unfinished = pending.name_finished.is_some_and(|name| name < 0.0);
        let finishes = self
            .turn_evidence(cue)
            .is_some_and(|finished| finished > 0.0);
        let started = now.0.saturating_sub(u64::from(cue.duration_ms));
        let name_end = pending.name_ended_at.0;
        let close = started.saturating_add(CONTINUATION_JITTER_MS) >= name_end
            && started <= name_end.saturating_add(u64::from(attention.continuation_gap_ms));
        name_unfinished && finishes && close && norm_vad >= attention.follow_up_min_vad
    }

    /// Whether verification ran and did not confirm the owner. While Enton talks its
    /// own echo is the likeliest voice, so interrupting it takes positive evidence;
    /// without verification, loudness alone decides, as it always did.
    fn is_unverified_voice(&self, cue: &SpeechCue) -> bool {
        cue.speaker_sim.is_some() && !self.is_verified_speaker(cue)
    }

    /// The TV belief at `now`, decayed since the last line that fed it.
    fn tv_presence_at(&self, now: Millis) -> f32 {
        decay(
            self.tv_presence,
            now.since(self.tv_heard_at),
            self.profile.source.tv_half_life_ms,
        )
    }

    /// Feed the TV belief with an overheard line: a voice that sounds like a
    /// loudspeaker rather than the owner moves it toward certainty.
    fn track_tv(&mut self, now: Millis, cue: &SpeechCue) {
        let source = self.profile.source;
        let evidence = self.evidence(cue);
        let loudspeaker = evidence.owner_over_reproduced + evidence.live_over_reproduced;
        let mut presence = self.tv_presence_at(now);
        if loudspeaker <= -source.tv_line_llr {
            presence += source.tv_line_weight * (1.0 - presence);
        }
        self.tv_presence = presence.clamp(0.0, 1.0);
        self.tv_heard_at = now;
    }

    /// Whether a cue is reproduced media, and whether it is someone else's voice.
    /// While the TV is on, both take `tv_caution_llr` less evidence, but only for a
    /// sensor that ran: a cue without one is never turned away for the TV.
    fn window_vetoes(&self, now: Millis, cue: &SpeechCue) -> Vetoes {
        let source = self.profile.source;
        let caution = if self.tv_presence_at(now) >= source.tv_on_level {
            source.tv_caution_llr
        } else {
            0.0
        };
        let evidence = self.evidence(cue);
        let other_llr = -(self.profile.attention.other_voice_llr - caution);
        let voice = evidence
            .owner_over_other
            .min(evidence.owner_over_reproduced);
        Vetoes {
            media: cue.media.is_some()
                && evidence.live_over_reproduced <= -(source.media_llr - caution),
            other_voice: cue.speaker_sim.is_some() && evidence.owner_live() <= other_llr,
            voice_alone: cue.speaker_sim.is_some() && voice <= other_llr,
        }
    }

    /// Whether the evidence says the cue came from a loudspeaker (TV, radio, music).
    fn is_media(&self, cue: &SpeechCue) -> bool {
        self.evidence(cue).live_over_reproduced <= -self.profile.source.media_llr
    }

    /// Whether the evidence says the cue is not the owner speaking live.
    fn is_known_other_speaker(&self, cue: &SpeechCue) -> bool {
        self.evidence(cue).owner_live() <= -self.profile.attention.other_voice_llr
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

/// Exponential decay of `level` over `dt_ms` with the given half-life, in IEEE
/// arithmetic only: whole half-lives halve exactly and the fraction left uses a
/// fixed polynomial, so every platform computes the same bits (a library `exp`
/// is not specified to).
fn decay(level: f32, dt_ms: u64, half_life_ms: u64) -> f32 {
    if level <= 0.0 || half_life_ms == 0 {
        return 0.0;
    }
    let halvings = dt_ms / half_life_ms;
    if halvings >= 64 {
        return 0.0;
    }
    // 2^-halvings, exactly: the biased exponent of a power of two.
    let whole = u32::try_from((127 - halvings) << 23).map_or(0.0, f32::from_bits);
    let fraction = (dt_ms % half_life_ms) as f32 / half_life_ms as f32;
    (level * whole * exp2_neg(fraction)).clamp(0.0, 1.0)
}

/// 2^-x for x in [0, 1): the Taylor series of e^t at t = -x ln 2, to degree 10
/// in Horner form, within a few units in the last place.
fn exp2_neg(x: f32) -> f32 {
    let t = -x * std::f32::consts::LN_2;
    (1..=10u8)
        .rev()
        .fold(1.0, |acc, k| 1.0 + t * acc / f32::from(k))
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
    fn decay_halves_exactly_at_each_half_life() {
        assert_eq!(decay(1.0, 30_000, 30_000).to_bits(), 0.5_f32.to_bits());
        assert_eq!(decay(0.8, 60_000, 30_000).to_bits(), 0.2_f32.to_bits());
        assert_eq!(decay(0.8, 0, 30_000).to_bits(), 0.8_f32.to_bits());
        assert_eq!(decay(1.0, 64 * 30_000, 30_000).to_bits(), 0);
    }

    #[test]
    fn decay_follows_the_exponential_between_half_lives() {
        for dt in [1_u64, 250, 7_500, 15_000, 29_999, 45_000, 1_234_567] {
            let exact = (-(dt as f64) / 30_000.0 * std::f64::consts::LN_2).exp();
            let got = f64::from(decay(1.0, dt, 30_000));
            assert!((got - exact).abs() < 1e-6, "{dt} ms: {got} vs {exact}");
        }
    }

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
