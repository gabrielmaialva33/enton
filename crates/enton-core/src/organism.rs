//! The deterministic brainstem reducer.

use core::cmp::Ordering;

use serde::{Deserialize, Serialize};

use crate::profile::{DRAW_SCALE, InvalidProfile, Profile, TvCautionConfinement};
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
    /// What each independent sensor says on its own.
    alone: Objections,
    /// Voice, source and direction together rule out the owner speaking live.
    other_voice: bool,
}

/// Where the TV caution applies to one cue, in nats.
struct Cautions {
    /// To each sensor's own objection and to another person's voice.
    sensors: f32,
    /// To the loudspeaker alternative: voice, tagger and direction together.
    loudspeaker: f32,
    /// Whether the cue's direction of arrival is weighed at all.
    direction: bool,
    /// Whether the caution weighs on the loudspeaker alternative alone.
    confined: bool,
}

/// The objections of sensors that err independently of one another, so closeness
/// to an unfinished name may excuse one of them, never two. Each sensor raises at
/// most one: the voice objects on the voice alone and the tagger on its reading alone,
/// even though both, with the direction, also weigh in on the loudspeaker alternative.
// One independent flag per sensor, which is what the rule counts: the flags are not the
// states of one machine, the case this pedantic lint guards against.
#[allow(clippy::struct_excessive_bools)]
struct Objections {
    /// The tagger heard a loudspeaker.
    media: bool,
    /// The voice alone rules out the owner.
    voice: bool,
    /// The directedness detector heard speech addressed to someone else.
    undirected: bool,
    /// The direction of arrival alone points at the TV.
    direction: bool,
}

impl Objections {
    /// How many of the sensors object.
    fn count(&self) -> usize {
        [self.media, self.voice, self.undirected, self.direction]
            .into_iter()
            .filter(|objects| *objects)
            .count()
    }
}

/// Where overheard TV lines came from: the sum of their direction readings, each
/// discounted by its age, as of `at`. Only lines that the voice and the tagger mark as
/// the TV teach it, so the estimate never leans on itself.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
struct TvDirection {
    /// Sum of the lines' unit vectors, weighted as of `at`.
    sum: [f32; 2],
    /// Sum of the lines' weights as of `at`: how many lines, discounted by age.
    lines: f32,
    /// When the last line was added.
    at: Millis,
}

/// Mean resultant length below which the lines name no direction: readings that
/// scatter this much come from a TV that moved, or from voices mistaken for a TV.
const TV_DIRECTION_AGREEMENT: f32 = 0.5;

/// A continuation may start this much before the name's recorded end: endpoint jitter.
const CONTINUATION_JITTER_MS: u64 = 50;

/// What the owner's checklist holds, as far as the core knows: whether there is
/// something a drive could bring up. Its text never reaches the core.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Checklist {
    /// Nothing to check: no checklist, an effectively empty one, or none read yet.
    #[default]
    Empty,
    /// Something to check.
    Actionable,
}

/// A drive's thought awaiting its outcome, and the drive it answers: the drive's own
/// thought, or an answer to the owner that its deferred intent rides.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct DriveThought {
    thought: ThoughtId,
    drive: String,
}

/// A ready drive's intent, held while Enton is in a conversation: rather than cut the
/// owner off with a thought of its own, the drive rides the next thought the owner asks
/// for, which brings up what the drive wanted to at no extra paid call. Once the
/// conversation is over the drive may think alone instead; past `expires`, with neither,
/// it lets the intent go.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Deferred {
    /// The drive whose intent waits.
    pub drive: String,
    /// When it began to wait.
    pub since: Millis,
    /// When it lets go, unless a ride or a thought of its own came first.
    pub expires: Millis,
}

/// The version of the brainstem reducer and snapshot schema.
pub const REDUCER_VERSION: u32 = 15;

/// Which account pays for a thought.
#[derive(Debug, Clone, Copy)]
enum Payment {
    /// A turn Enton owes an answer: the obligation account.
    Obligation,
    /// A thought Enton chose on its own: the discretionary account.
    Discretionary,
    /// A borderline cue explored with this probability: the discretionary account pays
    /// and the thought carries the probability as its propensity.
    Explored(f32),
}

/// One flip of the exploration coin at a borderline cue, with the probability of the
/// side that came up.
#[derive(Debug, Clone, Copy)]
enum Draw {
    /// Think anyway.
    Explore(f32),
    /// Keep the objection and abstain.
    Keep(f32),
}

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
    /// Where the TV's lines have come from, for weighing each cue's direction of arrival.
    #[serde(default)]
    tv_direction: TvDirection,
    /// State of the exploration generator (`SplitMix64`), seeded from the profile and
    /// advanced once per coin flip, so a snapshot resumes the same sequence of draws.
    #[serde(default)]
    explore_state: u64,
    /// What the owner's checklist holds, as of the last `Checklist` event. Before one,
    /// nothing is known: nothing to check.
    #[serde(default)]
    checklist: Checklist,
    /// When Enton last heard the owner: an addressed cue it accepted, or a cue in the
    /// owner's verified voice. `None` until then: nobody has been home.
    #[serde(default)]
    owner_heard_at: Option<Millis>,
    /// The drive thought awaiting its outcome, if one is out.
    #[serde(default)]
    drive_thought: Option<DriveThought>,
    /// What a ready drive last abstained for, so a wait is logged when it starts and when
    /// its cause changes, not on every tick.
    #[serde(default)]
    drive_waiting: Option<Abstention>,
    /// Cortex failures in a row since the last reply.
    #[serde(default)]
    cortex_failures: u32,
    /// Until when discretionary thoughts back off after those failures.
    #[serde(default)]
    backoff_until: Option<Millis>,
    /// A ready drive's intent, held while Enton is in a conversation, to ride the
    /// owner's next request.
    #[serde(default)]
    deferred: Option<Deferred>,
    /// Until when the owner asked for quiet, as of the last `Quiet` command; `None`
    /// before one, or once released.
    #[serde(default)]
    quiet_until: Option<Millis>,
    /// Whether the owner's quiet hours are on, as of the last `QuietHours` event.
    #[serde(default)]
    quiet_hours: bool,
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
        let explore_state = profile.exploration.explore_seed;
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
            tv_direction: TvDirection::default(),
            explore_state,
            checklist: Checklist::Empty,
            owner_heard_at: None,
            drive_thought: None,
            drive_waiting: None,
            cortex_failures: 0,
            backoff_until: None,
            deferred: None,
            quiet_until: None,
            quiet_hours: false,
        })
    }

    /// The decisions `profile` would take on `event` from this organism's exact state,
    /// which is left untouched: what an offline evaluation asks of every logged decision.
    ///
    /// Only the policy that reads the state changes. The state itself (budgets, drives,
    /// ignition, windows, beliefs and the exploration generator) stays as this organism
    /// built it, so the answer is exact for profiles that differ in how they read
    /// evidence (the evidence thresholds and exploration) and approximate for profiles
    /// that would also have shaped the state differently (budget capacities and the
    /// ignition threshold, which the state holds).
    ///
    /// # Errors
    ///
    /// Returns [`InvalidProfile`] if `profile` fails validation.
    pub fn counterfactual(
        &self,
        profile: &Profile,
        event: &Event,
    ) -> Result<Vec<Action>, InvalidProfile> {
        profile.validate()?;
        let mut twin = self.clone();
        twin.profile = profile.clone();
        twin.prices = PriceTable {
            think: profile.budgets.think_cost,
        };
        Ok(twin.step(event))
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

    /// Whether the body is in torpor (fever or a critical battery level), as of
    /// the last body reading. Read-only, for audits such as `enton why`.
    #[must_use]
    pub fn is_torpid(&self) -> bool {
        self.torpor
    }

    /// The drives and their current levels. Read-only, for audits and tests.
    #[must_use]
    pub fn drives(&self) -> &DriveTable {
        &self.drives
    }

    /// Whether the owner's checklist holds something a drive could bring up, as of the
    /// last `Checklist` event (false before one).
    #[must_use]
    pub fn checklist_actionable(&self) -> bool {
        self.checklist == Checklist::Actionable
    }

    /// When Enton last heard the owner: an addressed cue it accepted, or a cue in the
    /// owner's verified voice. `None` if it never has.
    #[must_use]
    pub fn owner_heard_at(&self) -> Option<Millis> {
        self.owner_heard_at
    }

    /// Whether the owner counts as around at `now`: heard within the profile's
    /// presence window. Discretionary thoughts need it.
    #[must_use]
    pub fn owner_present_as_of(&self, now: Millis) -> bool {
        self.owner_present(now)
    }

    /// The drive thought awaiting its outcome, and the drive it answers, if one is out.
    #[must_use]
    pub fn drive_thought(&self) -> Option<(ThoughtId, &str)> {
        self.drive_thought
            .as_ref()
            .map(|pending| (pending.thought, pending.drive.as_str()))
    }

    /// Cortex failures in a row since the last reply.
    #[must_use]
    pub fn cortex_failures(&self) -> u32 {
        self.cortex_failures
    }

    /// Until when discretionary thoughts back off after cortex failures, if they do.
    #[must_use]
    pub fn backoff_until(&self) -> Option<Millis> {
        self.backoff_until
    }

    /// The intent a ready drive holds while Enton is in a conversation, to ride the
    /// owner's next request, if one is held.
    #[must_use]
    pub fn deferred(&self) -> Option<&Deferred> {
        self.deferred.as_ref()
    }

    /// Until when the owner asked for quiet, as of the last `Quiet` command: `None`
    /// before one, or once released. Quiet holds before this instant.
    #[must_use]
    pub fn quiet_until(&self) -> Option<Millis> {
        self.quiet_until
    }

    /// Whether the owner's quiet mode holds at `now`.
    #[must_use]
    pub fn quiet_as_of(&self, now: Millis) -> bool {
        self.quiet(now)
    }

    /// Whether the owner's quiet hours are on, as of the last `QuietHours` event.
    #[must_use]
    pub fn in_quiet_hours(&self) -> bool {
        self.quiet_hours
    }

    /// Belief, from zero to one, that a TV is playing at `now`: the value a cue
    /// at that instant is weighed against, decayed since the last overheard line.
    /// Read-only, for audits such as `enton why`.
    #[must_use]
    pub fn tv_presence_as_of(&self, now: Millis) -> f32 {
        self.tv_presence_at(now)
    }

    /// The TV's learned direction at `now`, a unit vector in the microphone array's
    /// frame: `None` until enough recent TV lines agree on one. Read-only, for audits
    /// such as `enton why`.
    #[must_use]
    pub fn tv_direction_as_of(&self, now: Millis) -> Option<[f32; 2]> {
        self.tv_direction_at(now)
    }

    /// What the calibrated sensors say about `cue` at `now`, its direction weighed
    /// against the TV's direction learned so far: the evidence a cue at that instant
    /// is judged on. Read-only, for audits such as `enton why`.
    #[must_use]
    pub fn evidence_as_of(&self, now: Millis, cue: &SpeechCue) -> Evidence {
        self.evidence_at(now, cue)
    }

    /// Consume a recorded event and return decisions without executing effects.
    pub fn step(&mut self, event: &Event) -> Vec<Action> {
        self.last_seen = self.last_seen.max(event.now());
        match event {
            Event::Tick { now } => self.tick(*now),
            Event::Body { signals, .. } => {
                let signals = signals.canonical();
                self.torpor = signals
                    .temperature_c
                    .is_some_and(|value| value >= self.profile.body.fever_c)
                    || signals
                        .battery
                        .is_some_and(|value| value <= self.profile.body.lethargy_battery);
                Vec::new()
            }
            Event::Speech { now, cue } => vec![self.speech(*now, &cue.canonical())],
            Event::Checklist { actionable, .. } => {
                self.checklist = if *actionable {
                    Checklist::Actionable
                } else {
                    Checklist::Empty
                };
                Vec::new()
            }
            Event::CortexFailed { now, thought } => {
                self.cortex_failed(*now, *thought);
                Vec::new()
            }
            Event::Quiet { now, until } => {
                self.hush(*now, *until);
                Vec::new()
            }
            Event::QuietHours { active, .. } => {
                self.quiet_hours = *active;
                Vec::new()
            }
            Event::CortexReply { now, text, thought } => {
                self.drives.satisfy("social", 0.3);
                if self.issued(*thought) {
                    // The cortex answers again: discretionary thoughts stop backing off.
                    self.cortex_failures = 0;
                    self.backoff_until = None;
                }
                // Any reply to a drive's thought answers it, silence included: the drive
                // checked what there was to check, and asks again only once its pressure
                // builds back up.
                if let Some(answered) = self
                    .drive_thought
                    .take_if(|pending| pending.thought == *thought)
                {
                    self.drives.satisfy(&answered.drive, 1.0);
                    self.ignition.settle(self.drives.pressure());
                    // A ride's reply answers the intent it carried, as the drive's own would.
                    if self
                        .deferred
                        .as_ref()
                        .is_some_and(|deferred| deferred.drive == answered.drive)
                    {
                        self.deferred = None;
                    }
                }
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
                // Silence is an outcome, not a failure: a blank reply says nothing.
                if text.trim().is_empty() {
                    Vec::new()
                } else {
                    vec![Action::Speak { text: text.clone() }]
                }
            }
            Event::PlaybackStarted { now, utterance } => {
                self.playback_status = PlaybackStatus::Speaking {
                    utterance: *utterance,
                    started_at: *now,
                };
                // Voice mode precedence: playback supersedes text anchor and suspends the
                // windows. Not while an "Enton..." waits for the rest of its request: what
                // plays then (the runtime's acknowledgement chime) leaves the caller's turn open.
                if self.pending_attend.is_none() {
                    self.attention_until = None;
                    self.verified_attention_until = None;
                }
                Vec::new()
            }
            // A cut playback ends like a finished one: the flag is for the audit.
            Event::PlaybackFinished { now, utterance, .. } => {
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
            let action =
                self.pay_and_think(now, Reason::Keyword, pending.salience, Payment::Obligation);
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

        // An intent that found nothing to ride, and no turn of its own, lets go.
        if let Some(expired) = self.expire_deferral(now) {
            self.drive_waiting = None;
            actions.push(expired);
            return actions;
        }
        // A drive waits for a conversation to end: a thought of its own would cut the
        // owner off, and would supersede the answer Enton owes them. It holds its intent
        // instead, to ride the owner's next request.
        if self.in_conversation(now) {
            if self.ignition.drive_pressing() {
                self.defer(now);
            }
            self.drive_waiting = None;
            return actions;
        }
        if !self.ignition.drive_ready(now) {
            self.drive_waiting = None;
            return actions;
        }
        let Some(drive) = self.drives.strongest().map(|drive| drive.name.clone()) else {
            return actions;
        };
        let reason = Reason::Drive(drive.clone());
        let salience = self.ignition.salience();
        let held_back = if self.torpor {
            Some(Abstention::Torpor)
        } else {
            self.discretion_gate(now, true).or_else(|| {
                (self.thoughts_exhausted()
                    || !self.discretionary_budget.can_spend(self.prices.think))
                .then_some(Abstention::OutOfEnergy)
            })
        };
        if let Some(why) = held_back {
            // A ready drive that cannot think logs its wait when it starts and whenever
            // the cause changes; the ticks in between stay silent, spend nothing, and
            // leave the ignition armed.
            if self.drive_waiting != Some(why) {
                self.drive_waiting = Some(why);
                actions.push(Action::Abstain {
                    reason,
                    salience,
                    why,
                    propensity: None,
                });
            }
        } else {
            self.drive_waiting = None;
            // Discretionary drive thought spends discretionary budget
            let action = self.pay_and_think(now, reason, salience, Payment::Discretionary);
            if let Action::Think { thought, .. } = action {
                self.drive_thought = Some(DriveThought { thought, drive });
            }
            actions.push(action);
        }
        actions
    }

    /// Whether Enton is in a conversation at `now`: waiting for the rest of a request,
    /// inside an attention window, or speaking (its echo included).
    fn in_conversation(&self, now: Millis) -> bool {
        self.pending_attend.is_some()
            || self.attention_until.is_some_and(|until| now < until)
            || self
                .verified_attention_until
                .is_some_and(|until| now < until)
            || self.is_in_echo_period(now)
    }

    /// What holds a discretionary thought back at `now`, in order: the owner's quiet
    /// mode, then their quiet hours; for a drive, a checklist with nothing to check; for
    /// any, nobody home or a cortex backoff.
    fn discretion_gate(&self, now: Millis, drive: bool) -> Option<Abstention> {
        if self.quiet(now) {
            Some(Abstention::Quiet)
        } else if self.quiet_hours {
            Some(Abstention::QuietHours)
        } else if drive && self.checklist == Checklist::Empty {
            Some(Abstention::NothingToCheck)
        } else if !self.owner_present(now) {
            Some(Abstention::NobodyHome)
        } else if self.backing_off(now) {
            Some(Abstention::Backoff)
        } else {
            None
        }
    }

    /// Whether the owner asked for quiet and it still holds at `now`.
    fn quiet(&self, now: Millis) -> bool {
        self.quiet_until.is_some_and(|until| now < until)
    }

    /// The owner's quiet command at `now`: quiet until `until`, or released when that is
    /// not after `now`. It called Enton by name, so the owner is home. It ends the turn it
    /// may have finished, so a pending "Enton..." does not time out into an answer, and
    /// quiet closes the attention windows: the owner is done talking for now.
    fn hush(&mut self, now: Millis, until: Millis) {
        self.heard_owner(now);
        self.pending_attend = None;
        if until > now {
            self.quiet_until = Some(until);
            self.attention_until = None;
            self.verified_attention_until = None;
        } else {
            self.quiet_until = None;
        }
    }

    /// Whether a drive's intent may ride an answer to the owner at `now` (or begin to
    /// wait for one): nothing but the conversation holds the drive back. A ride buys no
    /// thought, so neither the budget nor a cortex backoff applies; the body, quiet, the
    /// checklist and somebody home do.
    fn may_ride(&self, now: Millis) -> bool {
        !self.torpor
            && !self.quiet(now)
            && !self.quiet_hours
            && self.checklist == Checklist::Actionable
            && self.owner_present(now)
    }

    /// A drive presses to fire during a conversation: hold the strongest drive's intent,
    /// unless one is already held or something besides the conversation holds it back.
    fn defer(&mut self, now: Millis) {
        if self.deferred.is_some() || !self.may_ride(now) {
            return;
        }
        let Some(drive) = self.drives.strongest().map(|drive| drive.name.clone()) else {
            return;
        };
        let expires = Millis(now.0.saturating_add(self.profile.discretion.deferral_ms));
        self.deferred = Some(Deferred {
            drive,
            since: now,
            expires,
        });
    }

    /// A held intent whose deferral ran out, with no ride in flight, lets go: its drive is
    /// answered as if by silence and asks again only once its pressure builds back up.
    /// The abstention is logged once, at the tick that drops it.
    fn expire_deferral(&mut self, now: Millis) -> Option<Action> {
        if self.drive_thought.is_some() {
            // A ride in flight waits for its outcome.
            return None;
        }
        let expired = self
            .deferred
            .take_if(|deferred| now >= deferred.expires)?;
        let salience = self.ignition.salience();
        self.drives.satisfy(&expired.drive, 1.0);
        self.ignition.settle(self.drives.pressure());
        Some(Action::Abstain {
            reason: Reason::Drive(expired.drive),
            salience,
            why: Abstention::Expired,
            propensity: None,
        })
    }

    /// An answer the owner asked for, `thought`, carries the held intent when it may: the
    /// thought now answers the drive, as the drive's own would, and its reply, failure or
    /// supersession resolves it like one. The intent stays held until it is answered, so a
    /// ride that is lost may ride the owner's next request.
    fn ride(&mut self, now: Millis, thought: ThoughtId) -> Option<String> {
        if !self.may_ride(now) {
            return None;
        }
        let drive = self.deferred.as_ref()?.drive.clone();
        self.drive_thought = Some(DriveThought {
            thought,
            drive: drive.clone(),
        });
        Some(drive)
    }

    /// Whether the owner was heard within the presence window before `now`.
    fn owner_present(&self, now: Millis) -> bool {
        self.owner_heard_at
            .is_some_and(|at| now.since(at) < self.profile.discretion.presence_window_ms)
    }

    /// Record that the owner was heard at `now`.
    fn heard_owner(&mut self, now: Millis) {
        self.owner_heard_at = Some(self.owner_heard_at.map_or(now, |at| at.max(now)));
    }

    /// Whether discretionary thoughts are still backing off after cortex failures.
    fn backing_off(&self, now: Millis) -> bool {
        self.backoff_until.is_some_and(|until| now < until)
    }

    /// Whether `thought` is one this organism issued.
    fn issued(&self, thought: ThoughtId) -> bool {
        thought.0 >= 1 && thought.0 < self.next_thought
    }

    /// A thought failed in the cortex: count the failure and back off discretionary
    /// thoughts, exponentially in the failures in a row. A failed drive thought did not
    /// answer its drive, which may ask again once the backoff allows; a failed answer
    /// to the owner ends the conversation's wait for it.
    fn cortex_failed(&mut self, now: Millis, thought: ThoughtId) {
        if !self.issued(thought) {
            return;
        }
        self.cortex_failures = self.cortex_failures.saturating_add(1);
        let wait = self.profile.discretion.backoff_ms(self.cortex_failures);
        let until = Millis(now.0.saturating_add(wait));
        self.backoff_until = Some(self.backoff_until.map_or(until, |at| at.max(until)));
        if self
            .drive_thought
            .take_if(|pending| pending.thought == thought)
            .is_some()
        {
            self.ignition.rearm();
        }
        if self.conversation_thought == Some(thought) {
            self.conversation_thought = None;
        }
    }

    /// A newer thought, or speech the runtime accepts, supersedes the drive thought in
    /// flight: the runtime abandons it, so it will neither answer nor fail. Its drive
    /// was not answered and may ask again later.
    fn supersede_drive_thought(&mut self) {
        if self.drive_thought.take().is_some() {
            self.ignition.rearm();
        }
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
                propensity: None,
            };
        }
        let echo = self.profile.echo;
        // Three separate pieces of evidence: loud enough over the expected echo, speech-like
        // enough, and a voice allowed to interrupt. Only loudness says anything about the
        // echo path, so only a cue that failed it may train the echo model.
        let (loud_enough, speech_like, voice_allowed) = if cue.keyword {
            let loud = if self.self_speech_has_keyword || self.is_unverified_voice(now, cue) {
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
            let margin = if self.is_verified_speaker(now, cue) {
                echo.verified_barge_in_margin
            } else {
                echo.echo_barge_in_margin
            };
            (
                norm_energy > self.echo_energy_expectation + margin,
                norm_vad >= self.profile.attention.follow_up_min_vad,
                !self.is_unverified_voice(now, cue),
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
                propensity: None,
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
        if self.pending_attend.is_some() {
            // The rest of a request Enton was waiting for, over its own acknowledgement:
            // it finishes that request, as it would in silence, and only once.
            return self.accept_in_window(
                now,
                cue,
                norm_energy,
                norm_vad,
                norm_dur,
                Payment::Obligation,
            );
        }

        // An accepted barge-in in the caller's voice: the owner is home.
        self.heard_owner(now);
        let salience = self.calculate_base_salience(norm_energy, norm_vad, norm_dur);
        if self.torpor {
            return Action::Abstain {
                reason: Reason::FollowUp,
                salience,
                why: Abstention::Torpor,
                propensity: None,
            };
        }
        self.open_attention(now);
        let action = self.pay_and_think(now, Reason::FollowUp, salience, Payment::Obligation);
        if let Action::Think { thought, .. } = action {
            self.conversation_thought = Some(thought);
        }
        action
    }

    fn speech(&mut self, now: Millis, cue: &SpeechCue) -> Action {
        self.advance_playback_status(now);
        // The owner's verified voice says they are home, whatever the cue turns out to be.
        if cue.speaker_sim.is_some() && self.is_verified_speaker(now, cue) {
            self.heard_owner(now);
        }

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
        let tv_line = if !cue.keyword && norm_vad >= self.profile.attention.follow_up_min_vad {
            self.track_tv(now, cue)
        } else {
            None
        };
        let action = self.speech_heard(now, cue, norm_energy, norm_vad, norm_dur);
        // A cue is judged against where the TV was heard before it; only then does a TV
        // line teach its own direction.
        if let Some(direction) = tv_line {
            self.learn_tv_direction(now, direction);
        }
        action
    }

    /// Speech outside Enton's own playback: its name, a cue in a window, or overheard.
    fn speech_heard(
        &mut self,
        now: Millis,
        cue: &SpeechCue,
        norm_energy: f32,
        norm_vad: f32,
        norm_dur: f32,
    ) -> Action {
        if cue.keyword {
            return self.speech_addressed(now, cue, norm_energy, norm_vad, norm_dur);
        }
        if self.is_in_attention_window(now, cue) {
            return self.speech_in_window(now, cue, norm_energy, norm_vad, norm_dur);
        }
        if let Some(action) = self.explore_unverified(now, cue, norm_energy, norm_vad, norm_dur) {
            return action;
        }

        // Overheard speech, outside any attention window: the TV makes a loudspeaker
        // likelier here too.
        let media = self.media_objection(now, cue);
        self.speech_unaddressed(now, media, true, norm_energy, norm_vad, norm_dur)
    }

    /// A cue inside the longer window only, in a voice that fell short of verification,
    /// is heard as overheard speech. When that would abstain, the shortfall (and any
    /// objection the window itself would raise) is borderline and, heard in the window,
    /// the cue would buy a thought, flip the exploration coin: think as a follow-up, or
    /// abstain as overheard speech, both logged with their probability. `None` leaves the
    /// cue to the overheard path.
    fn explore_unverified(
        &mut self,
        now: Millis,
        cue: &SpeechCue,
        norm_energy: f32,
        norm_vad: f32,
        norm_dur: f32,
    ) -> Option<Action> {
        let policy = self.profile.exploration;
        let in_longer_window = self
            .verified_attention_until
            .is_some_and(|until| now < until);
        if policy.explore_probability <= 0.0 || !in_longer_window || cue.speaker_sim.is_none() {
            return None;
        }
        let shortfall =
            self.profile.attention.verified_voice_llr - self.evidence_at(now, cue).owner_live();
        let depth = if self.window_objection(now, cue, norm_vad).is_some() {
            shortfall.max(self.objection_depth(now, cue, norm_vad))
        } else {
            shortfall
        };
        let thinks_without = self.thinks_in_window(cue, norm_vad);
        if depth > policy.explore_margin_nats || !thinks_without {
            return None;
        }
        // Only a cue that the overheard path turns away is an abstention to explore: ask
        // a copy, so this state moves only along the path actually taken.
        let media = self.media_objection(now, cue);
        let heard =
            self.clone()
                .speech_unaddressed(now, media, false, norm_energy, norm_vad, norm_dur);
        if !matches!(heard, Action::Abstain { .. }) {
            return None;
        }
        Some(match self.flip(now, depth, thinks_without)? {
            Draw::Explore(probability) => self.accept_in_window(
                now,
                cue,
                norm_energy,
                norm_vad,
                norm_dur,
                Payment::Explored(probability),
            ),
            Draw::Keep(probability) => logged(
                self.speech_unaddressed(now, media, false, norm_energy, norm_vad, norm_dur),
                probability,
            ),
        })
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
            && self.is_known_other_speaker(now, cue)
            && !self.is_whole_request(now, cue)
        {
            // Someone else saying the name does not take over a caller's unfinished turn.
            return Action::Abstain {
                reason: Reason::Keyword,
                salience: base_salience,
                why: Abstention::OtherSpeaker,
                propensity: None,
            };
        }
        // Someone called Enton by name: the owner is home.
        self.heard_owner(now);
        // Addressed speech is never habituated
        if !self.is_whole_request(now, cue) {
            let until = Millis(now.0.saturating_add(self.profile.attention.attention_ms));
            self.supersede_drive_thought();
            self.open_attention(now);
            self.pending_attend = Some(PendingAttend {
                until,
                salience: base_salience,
                name_ended_at: now,
                name_finished: cue
                    .turn_complete
                    .map(|_| self.whole_request_log_odds(now, cue)),
            });
            self.conversation_thought = None;
            return Action::Attend { until };
        }

        self.pending_attend = None;
        self.open_attention(now);
        // Keyword bypasses torpor and non-keyword cooldown; spends obligation_budget
        let action = self.pay_and_think(now, Reason::Keyword, base_salience, Payment::Obligation);
        if let Action::Think { thought, .. } = action {
            self.conversation_thought = Some(thought);
        }
        action
    }

    /// Whether a cue falls inside an attention window: the short one for any voice,
    /// the long one for the caller's verified voice or for speech clearly addressed
    /// to Enton.
    fn is_in_attention_window(&self, now: Millis, cue: &SpeechCue) -> bool {
        self.attention_until.is_some_and(|until| now < until)
            || (self
                .verified_attention_until
                .is_some_and(|until| now < until)
                && (self.is_verified_speaker(now, cue) || self.is_clearly_addressed(now, cue)))
    }

    /// What turns a cue inside an attention window away, in order: a loudspeaker,
    /// another voice, then speech addressed to someone else. None of them extends the
    /// window, and a pending "Enton?" keeps waiting for its own speaker. Closeness to
    /// an unfinished name outweighs one independent sensor's objection, never two: a
    /// cue that two of them reject is someone else, however close in time.
    fn window_objection(&self, now: Millis, cue: &SpeechCue, norm_vad: f32) -> Option<Abstention> {
        let vetoes = self.window_vetoes(now, cue);
        if self.is_adjacent_continuation(now, cue, norm_vad) && vetoes.alone.count() <= 1 {
            return None;
        }
        if vetoes.alone.media {
            Some(Abstention::Media)
        } else if vetoes.other_voice {
            Some(Abstention::OtherSpeaker)
        } else if vetoes.alone.undirected {
            Some(Abstention::Undirected)
        } else {
            None
        }
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
        if let Some(why) = self.window_objection(now, cue, norm_vad) {
            let salience = self.calculate_base_salience(norm_energy, norm_vad, norm_dur);
            let abstain = |propensity| Action::Abstain {
                reason: Reason::FollowUp,
                salience,
                why,
                propensity,
            };
            let depth = self.objection_depth(now, cue, norm_vad);
            let thinks_without = self.thinks_in_window(cue, norm_vad);
            return match self.flip(now, depth, thinks_without) {
                Some(Draw::Explore(probability)) => self.accept_in_window(
                    now,
                    cue,
                    norm_energy,
                    norm_vad,
                    norm_dur,
                    Payment::Explored(probability),
                ),
                Some(Draw::Keep(probability)) => abstain(Some(probability)),
                None => abstain(None),
            };
        }
        self.accept_in_window(
            now,
            cue,
            norm_energy,
            norm_vad,
            norm_dur,
            Payment::Obligation,
        )
    }

    /// Whether a cue in a window, with no evidence against it, would buy a thought rather
    /// than wait for the rest of a request or abstain for its VAD or for torpor.
    fn thinks_in_window(&self, cue: &SpeechCue, norm_vad: f32) -> bool {
        norm_vad >= self.profile.attention.follow_up_min_vad
            && if self.pending_attend.is_some() {
                !self
                    .turn_evidence(cue)
                    .is_some_and(|finished| finished < 0.0)
            } else {
                !self.torpor
            }
    }

    /// A cue in a window that no evidence turned away (or that exploration let through):
    /// the rest of a pending request or a follow-up, paid as `payment` says.
    fn accept_in_window(
        &mut self,
        now: Millis,
        cue: &SpeechCue,
        norm_energy: f32,
        norm_vad: f32,
        norm_dur: f32,
        payment: Payment,
    ) -> Action {
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
                    propensity: None,
                };
            }

            // A continuation the evidence let through is the owner at home; one that only
            // a coin flip let through is not evidence of anyone.
            if !matches!(payment, Payment::Explored(_)) {
                self.heard_owner(now);
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
                self.supersede_drive_thought();
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
            let action = self.pay_and_think(now, Reason::Keyword, combined_salience, payment);
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
                propensity: None,
            };
        }
        if norm_vad < self.profile.attention.follow_up_min_vad {
            return Action::Abstain {
                reason: Reason::FollowUp,
                salience,
                why: Abstention::BelowThreshold,
                propensity: None,
            };
        }

        if !matches!(payment, Payment::Explored(_)) {
            self.heard_owner(now);
        }
        // Follow-up needs only minimal VAD, bypasses non-keyword cooldown, pays obligation_budget
        let action = self.pay_and_think(now, Reason::FollowUp, salience, payment);
        if let Action::Think { thought, .. } = action {
            self.conversation_thought = Some(thought);
            self.open_attention(now);
        }
        action
    }

    /// Overheard speech: novelty, habituation and the discretionary threshold decide.
    /// `media` is how far past its threshold the tagger's objection is, when it objects;
    /// `explore` lets a borderline objection flip the exploration coin.
    fn speech_unaddressed(
        &mut self,
        now: Millis,
        media: Option<f32>,
        explore: bool,
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
        // arrive already familiar; tagged ones never buy a thought, unless a borderline
        // one is explored.
        if let Some(depth) = media {
            let abstain = |propensity| Action::Abstain {
                reason: Reason::Speech,
                salience: effective_salience,
                why: Abstention::Media,
                propensity,
            };
            let threshold = self.profile.ignition.discretionary_threshold;
            let thinks_without = !self.torpor
                && salience_with_novelty >= threshold
                && !self.ignition.in_cooldown(now)
                && effective_salience >= threshold;
            let draw = if explore {
                self.flip(now, depth, thinks_without)
            } else {
                None
            };
            return match draw {
                Some(Draw::Explore(probability)) => self.pay_and_think(
                    now,
                    Reason::Speech,
                    effective_salience,
                    Payment::Explored(probability),
                ),
                Some(Draw::Keep(probability)) => abstain(Some(probability)),
                None => abstain(None),
            };
        }

        if self.torpor {
            return Action::Abstain {
                reason: Reason::Speech,
                salience: effective_salience,
                why: Abstention::Torpor,
                propensity: None,
            };
        }

        if salience_with_novelty < self.profile.ignition.discretionary_threshold {
            return Action::Abstain {
                reason: Reason::Speech,
                salience: effective_salience,
                why: Abstention::BelowThreshold,
                propensity: None,
            };
        }

        if self.ignition.in_cooldown(now) {
            return Action::Abstain {
                reason: Reason::Speech,
                salience: effective_salience,
                why: Abstention::Cooldown,
                propensity: None,
            };
        }

        if effective_salience < self.profile.ignition.discretionary_threshold {
            return Action::Abstain {
                reason: Reason::Speech,
                salience: effective_salience,
                why: Abstention::Habituation,
                propensity: None,
            };
        }

        // Worth a thought, but a discretionary one: nobody home, or a cortex backing off,
        // holds it back.
        if let Some(why) = self.discretion_gate(now, false) {
            return Action::Abstain {
                reason: Reason::Speech,
                salience: effective_salience,
                why,
                propensity: None,
            };
        }

        self.pay_and_think(
            now,
            Reason::Speech,
            effective_salience,
            Payment::Discretionary,
        )
    }

    /// Open (or reopen) the attention windows at `from`: the short one for any cue,
    /// the long one for the verified voice of whoever addressed Enton.
    fn open_attention(&mut self, from: Millis) {
        let policy = self.profile.attention;
        self.attention_until = Some(Millis(from.0.saturating_add(policy.attention_ms)));
        self.verified_attention_until =
            Some(Millis(from.0.saturating_add(policy.verified_attention_ms)));
    }

    /// What the calibrated sensors say about a cue, apart from its direction, which
    /// only [`Self::evidence_at`] weighs: for the ratios that do not involve it.
    fn evidence(&self, cue: &SpeechCue) -> Evidence {
        self.profile.senses.read(cue)
    }

    /// What the calibrated sensors say about a cue at `now`, its direction of arrival
    /// weighed against the TV's direction learned so far.
    fn evidence_at(&self, now: Millis, cue: &SpeechCue) -> Evidence {
        self.profile
            .senses
            .read_with_tv(cue, self.tv_direction_weighed(now))
    }

    /// The TV direction that a cue's direction is weighed against at `now`: only while the
    /// TV is on, because with it off nothing plays from there and a voice from its
    /// direction is someone sitting in line with it; and never while Enton's own playback
    /// (or its hangover) fills the room, because the loudest source at the array is then
    /// the device's own loudspeaker and a reading says little about who talks over it.
    fn tv_direction_weighed(&self, now: Millis) -> Option<[f32; 2]> {
        let tv_on = self.tv_presence_at(now) >= self.profile.source.tv_on_level;
        if tv_on && !self.is_in_echo_period(now) {
            self.tv_direction_at(now)
        } else {
            None
        }
    }

    /// Whether a cue's direction of arrival is weighed at `now`: an array read it and
    /// Enton knows where the TV is (see [`Self::tv_direction_weighed`]).
    fn direction_speaks(&self, now: Millis, cue: &SpeechCue) -> bool {
        cue.canonical().direction.is_some() && self.tv_direction_weighed(now).is_some()
    }

    /// End-of-turn evidence, finished over unfinished, when an end-of-turn model ran.
    fn turn_evidence(&self, cue: &SpeechCue) -> Option<f32> {
        cue.turn_complete
            .map(|_| self.evidence(cue).finished_over_unfinished)
    }

    /// Whether the evidence verifies the owner speaking live (not merely unknown).
    fn is_verified_speaker(&self, now: Millis, cue: &SpeechCue) -> bool {
        self.evidence_at(now, cue).owner_live() >= self.profile.attention.verified_voice_llr
    }

    /// Whether a directedness detector ran and heard speech clearly addressed to Enton,
    /// in a voice that the window's own voice objection (stricter while the TV is on)
    /// does not rule out: when the profile allows it, such a cue may use the longer
    /// window like the owner's verified voice.
    fn is_clearly_addressed(&self, now: Millis, cue: &SpeechCue) -> bool {
        let attention = self.profile.attention;
        attention.directed_extends_window
            && cue.directed.is_some()
            && self.evidence(cue).addressed_over_not >= attention.directed_window_llr
            && !self.window_vetoes(now, cue).alone.voice
    }

    /// Whether a keyword cue already carries a whole request. Its length says how
    /// likely that is (the name alone is short) and an end-of-turn model adds its
    /// evidence, except for a voice known to be someone else or reproduced media,
    /// whose turn is not Enton's to judge: then length alone decides, so a name
    /// dropped mid-sentence by another person waits instead of buying a thought.
    /// Without the model, a cue shorter than `keyword_only_ms` is the name alone
    /// ("Enton?") and Enton waits.
    fn is_whole_request(&self, now: Millis, cue: &SpeechCue) -> bool {
        self.whole_request_log_odds(now, cue) >= 0.0
    }

    /// Log-odds, in nats, that a keyword cue holds a whole request: its length,
    /// plus the end-of-turn evidence when the voice is the owner's to judge.
    fn whole_request_log_odds(&self, now: Millis, cue: &SpeechCue) -> f32 {
        let attention = self.profile.attention;
        let beyond_name_s = (cue.duration_ms as f32 - attention.keyword_only_ms as f32) / 1_000.0;
        let length = attention.whole_request_llr_per_s * beyond_name_s;
        let turn = if self.is_known_other_speaker(now, cue) || self.is_media(cue) {
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
    fn is_unverified_voice(&self, now: Millis, cue: &SpeechCue) -> bool {
        cue.speaker_sim.is_some() && !self.is_verified_speaker(now, cue)
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
    /// loudspeaker rather than the owner, or comes from where the TV is, moves it toward
    /// certainty. Returns the line's direction when the voice and the tagger alone mark
    /// it as the TV: the lines that may teach where the TV is, since an estimate that
    /// learned from its own verdicts would confirm a wrong start forever.
    fn track_tv(&mut self, now: Millis, cue: &SpeechCue) -> Option<[f32; 2]> {
        let source = self.profile.source;
        let evidence = self.evidence_at(now, cue);
        let loudspeaker = evidence.owner_over_loudspeaker();
        let mut presence = self.tv_presence_at(now);
        if loudspeaker <= -source.tv_line_llr {
            presence += source.tv_line_weight * (1.0 - presence);
        }
        self.tv_presence = presence.clamp(0.0, 1.0);
        self.tv_heard_at = now;
        let voice_and_tagger = evidence.owner_over_reproduced + evidence.live_over_reproduced;
        cue.canonical()
            .direction
            .filter(|_| voice_and_tagger <= -source.tv_line_llr)
    }

    /// Add a TV line's direction to what Enton knows of where the TV is.
    fn learn_tv_direction(&mut self, now: Millis, direction: [f32; 2]) {
        let half_life = self.profile.source.tv_direction_half_life_ms;
        let learned = &mut self.tv_direction;
        let kept = decay(1.0, now.since(learned.at), half_life);
        let [x, y] = learned.sum;
        let [line_x, line_y] = direction;
        learned.sum = [x * kept + line_x, y * kept + line_y];
        learned.lines = learned.lines * kept + 1.0;
        // A line from a clock that ran backward counts as heard at the latest time.
        learned.at = learned.at.max(now);
    }

    /// The TV's direction at `now`, a unit vector, once enough recent lines taught it
    /// and they agree on one.
    fn tv_direction_at(&self, now: Millis) -> Option<[f32; 2]> {
        let source = self.profile.source;
        let learned = self.tv_direction;
        let kept = decay(1.0, now.since(learned.at), source.tv_direction_half_life_ms);
        // Fewer lines than the bar, or a count that is not a number: nothing is known.
        let lines = learned.lines * kept;
        if lines
            .partial_cmp(&source.tv_direction_min_lines)
            .is_none_or(Ordering::is_lt)
        {
            return None;
        }
        // Age scales the sum and the count alike, so neither the direction nor the
        // agreement between the lines depends on it.
        let [x, y] = learned.sum;
        let length = (x * x + y * y).sqrt();
        if length
            .partial_cmp(&(TV_DIRECTION_AGREEMENT * learned.lines))
            .is_none_or(Ordering::is_lt)
        {
            return None;
        }
        Some([x / length, y / length])
    }

    /// How many nats less evidence a loudspeaker or another voice needs to be turned
    /// away at `now`: `tv_caution_llr` while the TV is on, none otherwise.
    fn tv_caution(&self, now: Millis) -> f32 {
        let source = self.profile.source;
        if self.tv_presence_at(now) >= source.tv_on_level {
            source.tv_caution_llr
        } else {
            0.0
        }
    }

    /// How far past its threshold the tagger's objection to an overheard cue is, in
    /// nats, when the tagger objects at all. Overheard speech is weighed as it always
    /// was: with the whole TV caution, whatever the direction says.
    fn media_objection(&self, now: Millis, cue: &SpeechCue) -> Option<f32> {
        let limit = -(self.profile.source.media_llr - self.tv_caution(now));
        let live_over_reproduced = self.evidence(cue).live_over_reproduced;
        (cue.media.is_some() && live_over_reproduced <= limit)
            .then_some(limit - live_over_reproduced)
    }

    /// The TV caution at `now` split by where it applies to `cue`: to each sensor on its
    /// own and to another person's voice, and to the loudspeaker alternative. They are
    /// the same unless the cue's direction speaks and the profile confines the caution
    /// to the loudspeaker alternative (see [`TvCautionConfinement`]).
    fn cautions(&self, now: Millis, cue: &SpeechCue) -> Cautions {
        let caution = self.tv_caution(now);
        let direction = self.direction_speaks(now, cue);
        let confined = direction
            && match self.profile.source.tv_caution_confinement {
                TvCautionConfinement::Never => false,
                TvCautionConfinement::WithDirectedness => cue.directed.is_some(),
                TvCautionConfinement::Always => true,
            };
        Cautions {
            sensors: if confined { 0.0 } else { caution },
            loudspeaker: caution,
            direction,
            confined,
        }
    }

    /// The smallest loosening, in nats, of every evidence threshold at once that would let
    /// a cue through a window: how far past its threshold the deepest objection that
    /// matters is. A sensor that did not run never objects. An adjacent continuation needs
    /// only all but one independent sensor lifted (see [`Self::window_objection`]).
    fn objection_depth(&self, now: Millis, cue: &SpeechCue, norm_vad: f32) -> f32 {
        let attention = self.profile.attention;
        let cautions = self.cautions(now, cue);
        let evidence = self.evidence_at(now, cue);
        let other_llr = -(attention.other_voice_llr - cautions.sensors);
        let media_limit = self.profile.source.media_llr - cautions.sensors;
        let ran = |sensor: Option<f32>, depth: f32| sensor.map_or(f32::NEG_INFINITY, |_| depth);
        let media = ran(cue.media, -media_limit - evidence.live_over_reproduced);
        let voice = ran(
            cue.speaker_sim,
            other_llr
                - evidence
                    .owner_over_other
                    .min(evidence.owner_over_reproduced),
        );
        let other = if cautions.confined {
            let loudspeaker_llr = -(attention.other_voice_llr - cautions.loudspeaker);
            ran(cue.speaker_sim, other_llr - evidence.owner_over_other)
                .max(loudspeaker_llr - evidence.owner_over_loudspeaker())
        } else {
            ran(cue.speaker_sim, other_llr - evidence.owner_live())
        };
        let undirected = ran(
            cue.directed,
            -attention.undirected_llr - evidence.addressed_over_not,
        );
        let direction = if cautions.direction {
            evidence.from_tv_direction - media_limit
        } else {
            f32::NEG_INFINITY
        };
        let every = media.max(other).max(undirected);
        if !self.is_adjacent_continuation(now, cue, norm_vad) {
            return every;
        }
        // Lifting all but the deepest of the independent sensors: the second deepest of
        // the four, which is the second deepest of the first three unless the direction
        // lies between that and the deepest of them.
        let second_of_three = media.min(voice).max(media.max(voice).min(undirected));
        let deepest_of_three = media.max(voice).max(undirected);
        let second = second_of_three.max(deepest_of_three.min(direction));
        every.min(second)
    }

    /// Whether a cue is reproduced media, whether it is someone else's voice, whether it
    /// was addressed to someone else, and whether its direction alone points at the TV.
    /// While the TV is on, the voice, the tagger and the direction take `tv_caution_llr`
    /// less evidence, but only for a sensor that ran: a cue without one is never turned
    /// away for the TV. The direction never turns a cue away on its own, because the owner
    /// sometimes sits in line with the TV; it weighs in on the loudspeaker alternative of
    /// another voice and counts as one independent objection.
    fn window_vetoes(&self, now: Millis, cue: &SpeechCue) -> Vetoes {
        let source = self.profile.source;
        let attention = self.profile.attention;
        let cautions = self.cautions(now, cue);
        let evidence = self.evidence_at(now, cue);
        let other_llr = -(attention.other_voice_llr - cautions.sensors);
        let media_limit = source.media_llr - cautions.sensors;
        let voice = evidence
            .owner_over_other
            .min(evidence.owner_over_reproduced);
        let other_voice = if cautions.confined {
            // The caution weighs on the loudspeaker alternative only, where voice, tagger
            // and direction speak together; another person's voice is judged as with the
            // TV off.
            let loudspeaker_llr = -(attention.other_voice_llr - cautions.loudspeaker);
            (cue.speaker_sim.is_some() && evidence.owner_over_other <= other_llr)
                || evidence.owner_over_loudspeaker() <= loudspeaker_llr
        } else {
            cue.speaker_sim.is_some() && evidence.owner_live() <= other_llr
        };
        Vetoes {
            alone: Objections {
                media: cue.media.is_some() && evidence.live_over_reproduced <= -media_limit,
                voice: cue.speaker_sim.is_some() && voice <= other_llr,
                undirected: cue.directed.is_some()
                    && evidence.addressed_over_not <= -attention.undirected_llr,
                direction: cautions.direction && evidence.from_tv_direction >= media_limit,
            },
            other_voice,
        }
    }

    /// Whether the evidence says the cue came from a loudspeaker (TV, radio, music).
    fn is_media(&self, cue: &SpeechCue) -> bool {
        self.evidence(cue).live_over_reproduced <= -self.profile.source.media_llr
    }

    /// Whether the evidence says the cue is not the owner speaking live.
    fn is_known_other_speaker(&self, now: Millis, cue: &SpeechCue) -> bool {
        self.evidence_at(now, cue).owner_live() <= -self.profile.attention.other_voice_llr
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

    /// Whether the thought counter has no fresh ID left. Only a corrupt snapshot gets
    /// here (a thought per millisecond would take half a billion years); issuing a
    /// repeated ID would break idempotency, so Enton stops thinking, as if broke,
    /// without spending anything.
    fn thoughts_exhausted(&self) -> bool {
        self.next_thought == u64::MAX
    }

    /// Buy a thought from the account `payment` names, or abstain as out of energy.
    fn pay_and_think(
        &mut self,
        now: Millis,
        reason: Reason,
        salience: f32,
        payment: Payment,
    ) -> Action {
        let exhausted = self.thoughts_exhausted();
        let cost = self.prices.think;
        let account = match payment {
            Payment::Obligation => &mut self.obligation_budget,
            Payment::Discretionary | Payment::Explored(_) => &mut self.discretionary_budget,
        };
        if exhausted || !account.try_spend(cost) {
            return Action::Abstain {
                reason,
                salience,
                why: Abstention::OutOfEnergy,
                propensity: None,
            };
        }
        let thought = ThoughtId(self.next_thought);
        self.next_thought = self.next_thought.saturating_add(1);
        self.supersede_drive_thought();
        // An answer the owner asked for carries a held intent; the drive's own thought
        // takes its place.
        let rider = match payment {
            Payment::Obligation => self.ride(now, thought),
            Payment::Discretionary | Payment::Explored(_) => None,
        };
        if matches!(reason, Reason::Drive(_)) {
            self.deferred = None;
        }
        if matches!(reason, Reason::Drive(_)) || rider.is_some() {
            self.ignition.fired(now);
        } else {
            self.ignition.paid(now);
        }
        let (answers_turn, propensity) = match payment {
            Payment::Obligation => (true, None),
            Payment::Discretionary => (false, None),
            // An explored thought answers a turn exactly when its reason is one.
            Payment::Explored(probability) => (
                matches!(reason, Reason::Keyword | Reason::FollowUp),
                Some(probability),
            ),
        };
        self.speaking_for_obligation = answers_turn;
        Action::Think {
            thought,
            reason,
            salience,
            propensity,
            rider,
        }
    }

    /// Flip the exploration coin for a borderline cue: one an evidence objection turns
    /// away, whose deepest objecting sensor is `depth` nats past its threshold, and that
    /// would have bought a thought without the objection (`thinks_without`). `None` when
    /// the cue may not explore: exploration is off, the cue is not borderline, the body is
    /// in torpor, the discretionary account cannot pay for the thought, or a discretionary
    /// thought is held back at `now` (nobody home, or a cortex backoff). The generator
    /// advances only when the coin is flipped.
    fn flip(&mut self, now: Millis, depth: f32, thinks_without: bool) -> Option<Draw> {
        let policy = self.profile.exploration;
        let eligible = policy.explore_probability > 0.0
            && thinks_without
            && depth <= policy.explore_margin_nats
            && !self.torpor
            && !self.thoughts_exhausted()
            && self.discretionary_budget.can_spend(self.prices.think)
            && self.discretion_gate(now, false).is_none();
        if !eligible {
            return None;
        }
        // The draw and the cut are integers of at most 24 bits, exact in f32, so the coin
        // comes up `explore` exactly with the applied probability, and the two logged
        // propensities sum to exactly one.
        let explore = policy.applied_probability();
        Some(if (self.next_draw() as f32) < explore * DRAW_SCALE {
            Draw::Explore(explore)
        } else {
            Draw::Keep(1.0 - explore)
        })
    }

    /// The next 24 uniform bits of the exploration generator (`SplitMix64`).
    fn next_draw(&mut self) -> u32 {
        self.explore_state = self.explore_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.explore_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        u32::try_from((z ^ (z >> 31)) >> 40).unwrap_or(0)
    }
}

/// The same decision, logged with the probability with which it was taken.
fn logged(action: Action, probability: f32) -> Action {
    match action {
        Action::Think {
            thought,
            reason,
            salience,
            rider,
            ..
        } => Action::Think {
            thought,
            reason,
            salience,
            propensity: Some(probability),
            rider,
        },
        Action::Abstain {
            reason,
            salience,
            why,
            ..
        } => Action::Abstain {
            reason,
            salience,
            why,
            propensity: Some(probability),
        },
        other => other,
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

        for broken in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut profile = Profile::t1_ref();
            profile.attention.undirected_llr = broken;
            assert!(profile.validate().is_err(), "undirected {broken}");
            let mut profile = Profile::t1_ref();
            profile.attention.directed_window_llr = broken;
            assert!(profile.validate().is_err(), "directed window {broken}");
        }
        for broken in [-0.1, 1.5, f32::NAN, f32::INFINITY] {
            let mut profile = Profile::t1_ref();
            profile.exploration.explore_probability = broken;
            assert!(profile.validate().is_err(), "probability {broken}");
        }
        for broken in [-0.1, f32::NAN, f32::INFINITY] {
            let mut profile = Profile::t1_ref();
            profile.exploration.explore_margin_nats = broken;
            assert!(profile.validate().is_err(), "margin {broken}");
        }
        for fine in [0.0, 1.0] {
            let mut profile = Profile::t1_ref();
            profile.exploration.explore_probability = fine;
            profile.exploration.explore_margin_nats = fine;
            assert!(profile.validate().is_ok(), "{fine}");
        }
    }

    #[test]
    fn a_profile_stored_before_exploration_reads_back_with_it_off() {
        for shipped in [Profile::t1_ref(), Profile::desktop()] {
            assert_eq!(shipped.exploration, crate::ExplorationPolicy::default());
            let mut stored = serde_json::to_value(&shipped).unwrap();
            let fields = stored.as_object_mut().unwrap();
            for key in ["explore_probability", "explore_margin_nats", "explore_seed"] {
                assert!(fields.remove(key).is_some(), "{key}");
            }
            let profile: Profile = serde_json::from_value(stored).unwrap();
            assert_eq!(profile, shipped);
            assert_eq!(profile.exploration.explore_probability.to_bits(), 0);
        }
    }

    #[test]
    fn the_applied_probability_is_a_multiple_of_the_draw_resolution() {
        let applied = |probability| {
            crate::ExplorationPolicy {
                explore_probability: probability,
                ..crate::ExplorationPolicy::default()
            }
            .applied_probability()
        };
        for exact in [0.0_f32, 0.25, 0.5, 1.0] {
            assert_eq!(applied(exact).to_bits(), exact.to_bits());
        }
        // 0.1 is not a multiple of 2^-24: it rounds up by less than one step.
        let step = 1.0 / DRAW_SCALE;
        assert!(applied(0.1) >= 0.1 && applied(0.1) - 0.1 < step);
        assert_eq!((applied(0.1) * DRAW_SCALE).fract().to_bits(), 0);
        assert_eq!(applied(1e-12).to_bits(), step.to_bits());
    }

    #[test]
    fn a_profile_stored_before_discretion_reads_back_with_its_defaults() {
        for shipped in [Profile::t1_ref(), Profile::desktop()] {
            assert_eq!(shipped.discretion, crate::DiscretionPolicy::default());
            let mut stored = serde_json::to_value(&shipped).unwrap();
            let fields = stored.as_object_mut().unwrap();
            for key in [
                "presence_window_ms",
                "cortex_backoff_base_ms",
                "cortex_backoff_cap_ms",
            ] {
                assert!(fields.remove(key).is_some(), "{key}");
            }
            let profile: Profile = serde_json::from_value(stored).unwrap();
            assert_eq!(profile, shipped);
        }
    }

    #[test]
    fn discretion_needs_a_presence_window_and_a_backoff_capped_above_its_base() {
        let broken: [fn(&mut Profile); 3] = [
            |profile| profile.discretion.presence_window_ms = 0,
            |profile| profile.discretion.cortex_backoff_base_ms = 0,
            |profile| {
                profile.discretion.cortex_backoff_cap_ms =
                    profile.discretion.cortex_backoff_base_ms - 1;
            },
        ];
        for (index, breaks) in broken.into_iter().enumerate() {
            let mut profile = Profile::t1_ref();
            breaks(&mut profile);
            assert!(profile.validate().is_err(), "case {index}");
        }
        let mut equal = Profile::t1_ref();
        equal.discretion.cortex_backoff_cap_ms = equal.discretion.cortex_backoff_base_ms;
        assert!(equal.validate().is_ok());
    }

    #[test]
    fn a_profile_stored_before_directedness_reads_back_with_its_defaults() {
        for shipped in [Profile::t1_ref(), Profile::desktop()] {
            let mut stored = serde_json::to_value(&shipped).unwrap();
            let fields = stored.as_object_mut().unwrap();
            for key in [
                "undirected_llr",
                "directed_extends_window",
                "directed_window_llr",
            ] {
                assert!(fields.remove(key).is_some(), "{key}");
            }
            fields["senses"].as_object_mut().unwrap().remove("directed");
            let profile: Profile = serde_json::from_value(stored).unwrap();
            assert_eq!(profile, shipped);
        }
    }
}
