//! Equal-cost admission and closed-loop execution; annotations are only for scoring.
pub use crate::scoring::{Credit, Trigger};
use crate::{
    Annotation, BENCHMARK_VERSION, Economy, Error, Interval, Record, SegmentId, Stimulus, Tape,
    TapeKind,
};
use crate::{
    baseline::Baseline,
    economy::Account,
    invariants::Watch,
    offpolicy::{Decision, DecisionLog},
    report::{
        NoiseBreakdown, NoiseReason, NoiseStimulus, WasteBreakdown, WasteReason, WasteStimulus,
    },
    scoring::Scorer,
};
use enton_core::{
    Abstention, Action, Event, Millis, Organism, Profile, Reason, SpeechCue, ThoughtId, UtteranceId,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

const MAX_PROCESSED: usize = 100_000;
const MAX_PENDING: usize = 4096;

/// What E1 assumes of the owner's checklist: something on it, read when the run starts.
///
/// E1's tapes carry no checklist, and a drive with nothing to check never thinks, so
/// without an assumption no drive could ever think in E1. Assuming something to check
/// keeps drives deciding as they did before checklists existed, so E1 stays comparable
/// across versions. It changes nothing on the generated tapes either way: they last at
/// most two hours, and t1-ref's drives need about three and a half hours of unrelieved
/// pressure to reach their threshold. A tape may read the checklist itself (a
/// `Checklist` record); a reading at time zero follows this one and overrides it.
pub const CHECKLIST_ACTIONABLE: bool = true;

/// Policy identity; all policies receive the same exogenous events and outer economy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Controller {
    /// T1-ref organism with its private two-ledger policy.
    Organism,
    /// VAD/keyword and a fixed paid-call cooldown.
    Simple,
    /// Simple policy with the separately declared fixed continuation window.
    FixedWindow,
}
impl Controller {
    /// Stable name for reports.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Organism => "organism",
            Self::Simple => "simple",
            Self::FixedWindow => "simple + fixed window",
        }
    }
}

/// Auditable paid action; no historical actuator is executed.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PaidThought {
    /// Core/controller's deterministic thought ID.
    pub thought: ThoughtId,
    /// Decision timestamp in milliseconds.
    pub at: Millis,
    /// Policy's reason, independent of annotations.
    pub reason: Reason,
    /// Exact triggering segment, timeout, or internal decision.
    pub trigger: Trigger,
    /// Single-turn credit or waste.
    pub credit: Credit,
    /// The drive whose deferred intent rode this thought, at no extra paid call. Omitted
    /// from JSON when none did, so reports read as they did before rides.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rider: Option<String>,
}

/// Results for one policy; service is a paid-ignition proxy, not answer correctness.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ControllerResult {
    /// Policy identity.
    pub controller: Controller,
    /// Every paid call, including early, duplicate and feedback-triggered work.
    pub paid_calls: u64,
    /// Price times paid calls, in abstract units.
    pub total_cost: f64,
    /// Actual refill credited after capacity clipping.
    pub credited_refill: f64,
    /// Explicit credit for f32 rounding, bounded by 0.0001 thought price per debit.
    pub rounding_credit: f64,
    /// Final common-account balance.
    pub final_balance: f64,
    /// Keyword/FollowUp abstentions, regardless of cause or ground-truth relevance.
    pub rejected_obligations: u64,
    /// Speech/Drive rejections specifically due to insufficient funds.
    pub discretionary_exhaustion: u64,
    /// Fully served request/conversation episodes.
    pub served_requests: u64,
    /// Number of independently annotated relevant episodes.
    pub total_requests: u64,
    /// Timely, fully available turns receiving at least one paid thought.
    pub served_turns: u64,
    /// Number of independently annotated response-required turns.
    pub total_turns: u64,
    /// Strata breakdown of turns (served / total).
    pub strata: std::collections::BTreeMap<crate::scoring::Stratum, crate::scoring::Tally>,
    /// Turns served / total, grouped by the block condition of each turn's first segment.
    pub turns_by_condition:
        std::collections::BTreeMap<crate::tape::ConditionKey, crate::scoring::Tally>,
    /// Calls not earning new timely turn credit, including duplicates.
    pub wasted_calls: u64,
    /// Timely calls for an already served turn, included in waste.
    pub duplicate_calls: u64,
    /// All calls at timestamps inside E1b's half-open noise intervals.
    pub calls_in_noise: u64,
    /// Breakdown of noise-interval calls by reason and triggering stimulus.
    pub noise_breakdown: NoiseBreakdown,
    /// Breakdown of wasted and duplicate calls by reason and triggering stimulus.
    pub waste_breakdown: WasteBreakdown,
    /// What the controller decided on each segment that belongs to a turn: `think`,
    /// `attend`, or `abstain:` and the reason. Shows where requests are lost.
    pub turn_segment_decisions: BTreeMap<String, u64>,
    /// Calls whose direct trigger was synthetic self-echo (including its timeout).
    pub synthetic_self_ignitions: u64,
    /// Barge-in segments in the identical exogenous tape.
    pub barge_in_segments: u64,
    /// Such segments actually overlapping this controller's simulated playback.
    pub overlapping_barge_in_segments: u64,
    /// Generated feedback events actually processed during the observation horizon.
    pub feedback_events: u64,
    /// Feedback beyond the fixed observation horizon, explicitly censored.
    pub feedback_after_horizon: u64,
    /// Bounded per-thought causal audit.
    pub thoughts: Vec<PaidThought>,
}
impl ControllerResult {
    fn new(controller: Controller, tape: &Tape) -> Self {
        let requests: BTreeSet<_> = tape.turns().iter().map(|t| t.episode).collect();
        Self {
            controller,
            paid_calls: 0,
            total_cost: 0.0,
            credited_refill: 0.0,
            rounding_credit: 0.0,
            final_balance: 0.0,
            rejected_obligations: 0,
            discretionary_exhaustion: 0,
            served_requests: 0,
            total_requests: requests.len() as u64,
            served_turns: 0,
            total_turns: tape.turns().len() as u64,
            strata: std::collections::BTreeMap::new(),
            turns_by_condition: std::collections::BTreeMap::new(),
            wasted_calls: 0,
            duplicate_calls: 0,
            calls_in_noise: 0,
            noise_breakdown: NoiseBreakdown::new(),
            waste_breakdown: WasteBreakdown::new(),
            turn_segment_decisions: BTreeMap::new(),
            synthetic_self_ignitions: 0,
            barge_in_segments: 0,
            overlapping_barge_in_segments: 0,
            feedback_events: 0,
            feedback_after_horizon: 0,
            thoughts: Vec::new(),
        }
    }
}

/// Which sensor readings reach the controllers. A tape always carries every reading;
/// a sensor that is off is stripped at the policy boundary, from exogenous and feedback
/// cues alike, so an ablation never changes the tape itself.
// One independent on/off switch per sensor, which is what an ablation varies: the
// flags are not the states of one machine, the case this pedantic lint guards against.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Sensors {
    /// Speaker verification against the owner's voiceprint (`speaker_sim`).
    pub speaker: bool,
    /// Live voice versus loudspeaker audio tagger (`media`).
    pub media: bool,
    /// End-of-turn model (`turn_complete`).
    pub turn: bool,
    /// Device-directedness detector (`directed`). Off by default: it needs
    /// speech-to-text inside the window and about 0.5 s of CPU per segment for a 4B
    /// model, so it is a desktop-first sensor.
    pub directed: bool,
    /// Direction of arrival from a microphone array (`direction`). Off by default: it
    /// needs an array of two or more microphones, which a single-microphone device
    /// lacks. Serialized only when on, so a report without it reads as before it existed.
    #[serde(skip_serializing_if = "is_off")]
    pub direction: bool,
}

/// Whether a sensor is off, for fields serialized only when on.
// Serde's `skip_serializing_if` passes a reference.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_off(on: &bool) -> bool {
    !*on
}

impl Sensors {
    /// The sensors every E1 report has used: speaker, media and end of turn.
    pub const DEFAULT: Self = Self {
        speaker: true,
        media: true,
        turn: true,
        directed: false,
        direction: false,
    };

    /// The default sensors plus the device-directedness detector.
    pub const WITH_DIRECTED: Self = Self {
        directed: true,
        ..Self::DEFAULT
    };

    /// The default sensors plus the microphone array's direction of arrival.
    pub const WITH_DIRECTION: Self = Self {
        direction: true,
        ..Self::DEFAULT
    };

    /// The cue as the controllers perceive it: readings of a sensor that is off
    /// become `None`, what a cue without that sensor carries.
    #[must_use]
    pub fn sense(self, cue: SpeechCue) -> SpeechCue {
        SpeechCue {
            speaker_sim: cue.speaker_sim.filter(|_| self.speaker),
            media: cue.media.filter(|_| self.media),
            turn_complete: cue.turn_complete.filter(|_| self.turn),
            directed: cue.directed.filter(|_| self.directed),
            direction: cue.direction.filter(|_| self.direction),
            ..cue
        }
    }

    /// The event as the controllers perceive it (see [`Sensors::sense`]).
    #[must_use]
    pub fn perceive(self, event: &Event) -> Event {
        match event {
            Event::Speech { now, cue } => Event::Speech {
                now: *now,
                cue: self.sense(*cue),
            },
            other => other.clone(),
        }
    }

    fn names(self) -> Vec<&'static str> {
        [
            (self.speaker, "speaker verification"),
            (self.media, "media tagger"),
            (self.turn, "end of turn"),
            (self.directed, "directedness"),
            (self.direction, "direction of arrival"),
        ]
        .into_iter()
        .filter_map(|(on, name)| on.then_some(name))
        .collect()
    }
}

impl Default for Sensors {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl std::fmt::Display for Sensors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names = self.names();
        if names.is_empty() {
            f.write_str("none")
        } else {
            f.write_str(&names.join(", "))
        }
    }
}

/// Three independently executed controllers on one versioned exogenous tape.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExperimentRun {
    /// Benchmark protocol version.
    pub version: String,
    /// Population identity.
    pub kind: TapeKind,
    /// Tape seed.
    pub seed: u64,
    /// Sensor readings that reached the controllers.
    pub sensors: Sensors,
    /// Identical account/price contract for all controllers.
    pub economy: Economy,
    /// Organism result.
    pub organism: ControllerResult,
    /// Original simple controller result.
    pub simple: ControllerResult,
    /// Secondary fixed-window cost comparator result.
    pub fixed_window: ControllerResult,
}

/// Run all three policies on the same validated tape without audio, network or models,
/// with the default sensors. Returns errors on resource overflow or a violated
/// shared-budget invariant.
pub fn run_tape(tape: &Tape) -> Result<ExperimentRun, Error> {
    run_tape_with(tape, &Profile::t1_ref(), Sensors::DEFAULT)
}

/// Run one tape with a candidate organism profile and a sensor set, for calibration
/// sweeps and ablations on calibration seeds. The comparators and the shared account
/// keep the reference profile, so only the organism differs between candidates; every
/// controller perceives the same sensors.
///
/// # Errors
///
/// Returns an error if the tape is invalid or a limit is exceeded.
pub fn run_tape_with(
    tape: &Tape,
    organism_profile: &Profile,
    sensors: Sensors,
) -> Result<ExperimentRun, Error> {
    let profile = Profile::t1_ref();
    let economy = Economy::from_profile(&profile)?;
    let run = |controller, profile| run_controller(tape, controller, profile, economy, sensors);
    Ok(ExperimentRun {
        version: BENCHMARK_VERSION.into(),
        kind: tape.kind(),
        seed: tape.seed(),
        sensors,
        economy,
        organism: run(Controller::Organism, organism_profile)?,
        simple: run(Controller::Simple, &profile)?,
        fixed_window: run(Controller::FixedWindow, &profile)?,
    })
}

enum Machine {
    Core(Box<Organism>, Box<Watch>),
    Simple(Baseline),
}
impl Machine {
    fn new(controller: Controller, profile: &Profile) -> Result<Self, Error> {
        match controller {
            Controller::Organism => {
                let org =
                    Organism::new(profile.clone()).map_err(|e| Error::Invalid(e.to_string()))?;
                Ok(Self::Core(Box::new(org), Box::default()))
            }
            Controller::Simple => Ok(Self::Simple(Baseline::new(false))),
            Controller::FixedWindow => Ok(Self::Simple(Baseline::new(true))),
        }
    }
    /// Step the controller; the organism is checked against its invariants.
    fn step(&mut self, event: &Event, can_pay: bool) -> Result<Vec<Action>, Error> {
        match self {
            Self::Core(organism, watch) => {
                let actions = organism.step(event);
                watch.after_step(organism, event, &actions)?;
                Ok(actions)
            }
            Self::Simple(b) => Ok(b.step(event, can_pay)),
        }
    }
    fn attending(&self) -> bool {
        match self {
            Self::Core(o, _) => o.is_attending(),
            Self::Simple(_) => false,
        }
    }
}

/// Enton's own synthesized voice does not match the user.
const SELF_ECHO_SPEAKER_SIM: f32 = 0.15;
/// Clean synthesized speech through the loudspeaker lacks the broadcast cues a media tagger keys on.
const SELF_ECHO_MEDIA: f32 = 0.4;
/// Enton speaks whole sentences.
const SELF_ECHO_TURN_COMPLETE: f32 = 0.8;
/// Enton's replies are addressed to the owner, not to Enton: clearly undirected.
const SELF_ECHO_DIRECTED: f32 = 0.15;

struct Feedback {
    pending: BTreeMap<(Millis, u64), Record>,
    next_order: u64,
    next_segment: u32,
    echo_segments: BTreeSet<SegmentId>,
    playbacks: BTreeMap<UtteranceId, Interval>,
    horizon: Millis,
    censored: u64,
}
impl Feedback {
    fn new(horizon: Millis) -> Self {
        Self {
            pending: BTreeMap::new(),
            next_order: 0,
            next_segment: 50_000,
            echo_segments: BTreeSet::new(),
            playbacks: BTreeMap::new(),
            horizon,
            censored: 0,
        }
    }
    fn schedule(&mut self, record: Record) -> Result<(), Error> {
        if record.event.now() > self.horizon {
            self.censored += 1;
            return Ok(());
        }
        if self.pending.len() >= MAX_PENDING {
            return Err(Error::Limit("pending feedback events"));
        }
        self.pending
            .insert((record.event.now(), self.next_order), record);
        self.next_order += 1;
        Ok(())
    }
    fn reply(&mut self, now: Millis, thought: ThoughtId) -> Result<(), Error> {
        if self.playbacks.len() >= MAX_PROCESSED {
            return Err(Error::Limit("playback audit"));
        }
        self.playbacks.insert(
            UtteranceId(thought.0),
            Interval {
                start: Millis(now.0 + 400),
                end: Millis(now.0 + 2000),
            },
        );
        self.schedule(Record {
            event: Event::PlaybackStarted {
                now: Millis(now.0 + 400),
                utterance: UtteranceId(thought.0),
            },
            annotation: Annotation::Clock,
        })?;
        for (index, offset) in [900, 1700].into_iter().enumerate() {
            let segment = SegmentId(self.next_segment);
            self.next_segment += 1;
            self.echo_segments.insert(segment);
            self.schedule(Record {
                event: Event::Speech {
                    now: Millis(now.0 + offset),
                    cue: SpeechCue {
                        energy: 0.45,
                        vad_confidence: 0.55,
                        duration_ms: 250,
                        keyword: thought.0.is_multiple_of(4) && index == 0,
                        speaker_sim: Some(SELF_ECHO_SPEAKER_SIM),
                        media: Some(SELF_ECHO_MEDIA),
                        turn_complete: Some(SELF_ECHO_TURN_COMPLETE),
                        directed: Some(SELF_ECHO_DIRECTED),
                        // Enton's own loudspeaker sits in the device, too close to the
                        // array for a far-field direction: an estimator reports none.
                        direction: None,
                    },
                },
                annotation: Annotation::Speech {
                    segment,
                    episode: None,
                    source: Stimulus::SelfEcho,
                    pause_style: None,
                },
            })?;
        }
        self.schedule(Record {
            event: Event::CortexReply {
                now: Millis(now.0 + 2000),
                thought,
                text: "synthetic reply completed".into(),
            },
            annotation: Annotation::Clock,
        })?;
        self.schedule(Record {
            event: Event::PlaybackFinished {
                now: Millis(now.0 + 2000),
                utterance: UtteranceId(thought.0),
                interrupted: false,
            },
            annotation: Annotation::Clock,
        })
    }
    /// Mirror the runtime when a new thought starts or speech is accepted: the thought in
    /// flight is abandoned and the player is cancelled. Its reply never arrives, the echoes
    /// it had not produced yet never happen, and a playback already under way ends now,
    /// as the runtime reports a cancelled utterance.
    fn supersede(&mut self, now: Millis) -> Result<(), Error> {
        let mut unstarted = BTreeSet::new();
        let mut unfinished = BTreeSet::new();
        for record in self.pending.values() {
            match record.event {
                Event::PlaybackStarted { utterance, .. } => {
                    unstarted.insert(utterance);
                }
                Event::PlaybackFinished { utterance, .. } => {
                    unfinished.insert(utterance);
                }
                _ => {}
            }
        }
        // Everything still pending belongs to earlier thoughts.
        self.pending.clear();
        for utterance in &unstarted {
            self.playbacks.remove(utterance);
        }
        for utterance in unfinished.difference(&unstarted) {
            if let Some(interval) = self.playbacks.get_mut(utterance) {
                interval.end = now;
            }
            self.schedule(Record {
                event: Event::PlaybackFinished {
                    now,
                    utterance: *utterance,
                    interrupted: true,
                },
                annotation: Annotation::Clock,
            })?;
        }
        Ok(())
    }
    fn caused_by_echo(&self, trigger: Trigger) -> bool {
        match trigger {
            Trigger::Segment(id) | Trigger::AttendTimeout(id) => self.echo_segments.contains(&id),
            Trigger::Internal => false,
        }
    }
    fn observe_barge_in(&self, record: &Record, result: &mut ControllerResult) {
        if matches!(record.annotation.source(), Some(Stimulus::BargeIn(_))) {
            result.barge_in_segments += 1;
            if let Event::Speech { now, cue } = record.event {
                let start = Millis(now.0 - u64::from(cue.duration_ms));
                if self
                    .playbacks
                    .values()
                    .any(|interval| start < interval.end && *now_ref(&now) > interval.start)
                {
                    result.overlapping_barge_in_segments += 1;
                }
            }
        }
    }
}
// Keep interval arithmetic in the same timestamp type as core events.
fn now_ref(now: &Millis) -> &Millis {
    now
}

struct Execution<'a> {
    machine: Machine,
    scorer: Scorer<'a>,
    account: Account,
    feedback: Feedback,
    result: ControllerResult,
    economy: Economy,
    tape: &'a Tape,
    segment_sources: BTreeMap<SegmentId, Stimulus>,
    sensors: Sensors,
    /// Every decision with its propensity and each candidate's choice, when the run is a
    /// logging policy for off-policy evaluation.
    log: Option<DecisionLog<'a>>,
}
impl<'a> Execution<'a> {
    fn new(
        controller: Controller,
        profile: &Profile,
        economy: Economy,
        tape: &'a Tape,
        horizon: Millis,
        sensors: Sensors,
    ) -> Result<Self, Error> {
        let mut segment_sources = BTreeMap::new();
        for r in tape.records() {
            if let Annotation::Speech {
                segment, source, ..
            } = r.annotation
            {
                segment_sources.insert(segment, source);
            }
        }
        Ok(Self {
            machine: Machine::new(controller, profile)?,
            scorer: Scorer::new(tape),
            account: Account::new(economy),
            feedback: Feedback::new(horizon),
            result: ControllerResult::new(controller, tape),
            economy,
            tape,
            segment_sources,
            sensors,
            log: None,
        })
    }

    fn resolve_stimulus(&self, trigger: Trigger, record: &Record) -> NoiseStimulus {
        if self.feedback.caused_by_echo(trigger) {
            return NoiseStimulus::SelfEcho;
        }
        match trigger {
            Trigger::Segment(seg_id) => {
                if let Some(source) = record.annotation.source() {
                    stimulus_to_noise_stimulus(source)
                } else {
                    self.lookup_segment_stimulus(seg_id)
                }
            }
            Trigger::AttendTimeout(seg_id) => self.lookup_segment_stimulus(seg_id),
            Trigger::Internal => NoiseStimulus::NoneInternal,
        }
    }

    fn lookup_segment_stimulus(&self, seg_id: SegmentId) -> NoiseStimulus {
        self.segment_sources
            .get(&seg_id)
            .copied()
            .map_or(NoiseStimulus::NoneInternal, stimulus_to_noise_stimulus)
    }

    fn resolve_waste_stimulus(&self, trigger: Trigger, record: &Record) -> WasteStimulus {
        if self.feedback.caused_by_echo(trigger) {
            return WasteStimulus::SelfEcho;
        }
        match trigger {
            Trigger::Segment(seg_id) => {
                if let Some(source) = record.annotation.source() {
                    stimulus_to_waste_stimulus(source)
                } else {
                    self.lookup_segment_waste_stimulus(seg_id)
                }
            }
            Trigger::AttendTimeout(seg_id) => self.lookup_segment_waste_stimulus(seg_id),
            Trigger::Internal => WasteStimulus::Internal,
        }
    }

    fn lookup_segment_waste_stimulus(&self, seg_id: SegmentId) -> WasteStimulus {
        self.segment_sources
            .get(&seg_id)
            .copied()
            .map_or(WasteStimulus::Internal, stimulus_to_waste_stimulus)
    }

    fn process(&mut self, record: &Record) -> Result<(), Error> {
        let now = record.event.now();
        self.account.advance(now);
        self.feedback.observe_barge_in(record, &mut self.result);
        // Only Event crosses the policy boundary. No labels, segment IDs or turns, and
        // no reading of a sensor that is off.
        let event = self.sensors.perceive(&record.event);
        // Each candidate's choice is asked of the state the cue found, before it moves.
        let asked = match (&self.log, &self.machine) {
            (Some(log), Machine::Core(organism, _)) => log.ask(organism, &event)?,
            _ => None,
        };
        let actions = self.machine.step(&event, self.account.can_pay())?;
        let attending = self.machine.attending();
        if record.annotation.turn().is_some() {
            self.count_turn_segment_decision(&actions);
        }
        if actions
            .iter()
            .any(|action| matches!(action, Action::Think { .. } | Action::Attend { .. }))
        {
            self.feedback.supersede(now)?;
        }
        let paid_before = self.result.thoughts.len();
        for action in &actions {
            match action {
                Action::Think {
                    thought,
                    reason,
                    rider,
                    ..
                } => {
                    self.paid_thought(record, *thought, reason.clone(), rider.clone(), attending)?;
                }
                Action::Abstain { reason, why, .. } => {
                    if matches!(reason, Reason::Keyword | Reason::FollowUp) {
                        self.result.rejected_obligations += 1;
                    } else if *why == Abstention::OutOfEnergy {
                        self.result.discretionary_exhaustion += 1;
                    }
                }
                other => self.scorer.observe_non_think(record, other),
            }
        }
        if let Some(log) = self.log.as_mut() {
            let paid = self.result.thoughts.get(paid_before..).unwrap_or(&[]);
            log.record(record, &actions, asked.as_ref(), paid)?;
        }
        self.scorer.finish_event(now, attending);
        Ok(())
    }
    fn count_turn_segment_decision(&mut self, actions: &[Action]) {
        let decision = actions.iter().find_map(|action| match action {
            Action::Think { .. } => Some("think".to_owned()),
            Action::Attend { .. } => Some("attend".to_owned()),
            Action::Abstain { why, .. } => Some(format!("abstain:{why:?}")),
            Action::Speak { .. } => None,
        });
        if let Some(decision) = decision {
            *self
                .result
                .turn_segment_decisions
                .entry(decision)
                .or_default() += 1;
        }
    }
    /// Pay for `thought`, taken for `reason` and carrying `rider`'s deferred intent if one
    /// rides it, and score it.
    fn paid_thought(
        &mut self,
        record: &Record,
        thought: ThoughtId,
        reason: Reason,
        rider: Option<String>,
        attending: bool,
    ) -> Result<(), Error> {
        self.account.pay()?;
        if self.result.thoughts.len() >= MAX_PROCESSED {
            return Err(Error::Limit("paid thought audit"));
        }
        let now = record.event.now();
        let (trigger, credit) = self.scorer.think(record, &reason, attending);
        self.result.paid_calls += 1;
        self.result.total_cost += self.economy.thought_cost;
        match credit {
            Credit::Served(_) => {}
            Credit::Duplicate(_) => {
                self.result.duplicate_calls += 1;
                self.result.wasted_calls += 1;
            }
            Credit::Waste => self.result.wasted_calls += 1,
        }
        if matches!(credit, Credit::Waste | Credit::Duplicate(_)) {
            let waste_reason = WasteReason::from(&reason);
            let waste_stimulus = self.resolve_waste_stimulus(trigger, record);
            self.result
                .waste_breakdown
                .record(waste_reason, waste_stimulus);
        }
        if self
            .tape
            .noise_intervals()
            .iter()
            .any(|span| span.contains(now))
        {
            self.result.calls_in_noise += 1;
            let noise_reason = NoiseReason::from(&reason);
            let noise_stimulus = self.resolve_stimulus(trigger, record);
            self.result
                .noise_breakdown
                .record(noise_reason, noise_stimulus);
        }
        if self.feedback.caused_by_echo(trigger) {
            self.result.synthetic_self_ignitions += 1;
        }
        self.result.thoughts.push(PaidThought {
            thought,
            at: now,
            reason,
            trigger,
            credit,
            rider,
        });
        self.feedback.reply(now, thought)
    }
}

fn stimulus_to_noise_stimulus(source: Stimulus) -> NoiseStimulus {
    match source {
        Stimulus::Tv => NoiseStimulus::Tv,
        Stimulus::OtherSpeech
        | Stimulus::Request(_)
        | Stimulus::BargeIn(_)
        | Stimulus::FalseKeyword
        // Asides are live speech from the caller addressed to another person in the room.
        | Stimulus::Aside => NoiseStimulus::OtherPerson,
        Stimulus::Motor | Stimulus::Noise => NoiseStimulus::Motor,
        Stimulus::Ventilation => NoiseStimulus::Ventilation,
        Stimulus::SelfEcho => NoiseStimulus::SelfEcho,
    }
}

fn stimulus_to_waste_stimulus(source: Stimulus) -> WasteStimulus {
    match source {
        Stimulus::Request(_) => WasteStimulus::Request,
        Stimulus::BargeIn(_) => WasteStimulus::BargeIn,
        Stimulus::Tv => WasteStimulus::Tv,
        Stimulus::OtherSpeech => WasteStimulus::OtherPerson,
        Stimulus::Aside => WasteStimulus::Aside,
        Stimulus::FalseKeyword => WasteStimulus::FalseKeyword,
        Stimulus::SelfEcho => WasteStimulus::SelfEcho,
        Stimulus::Noise | Stimulus::Motor | Stimulus::Ventilation => WasteStimulus::Noise,
    }
}

fn run_controller(
    tape: &Tape,
    controller: Controller,
    profile: &Profile,
    economy: Economy,
    sensors: Sensors,
) -> Result<ControllerResult, Error> {
    let execution = Execution::new(controller, profile, economy, tape, horizon(tape), sensors)?;
    Ok(drive(execution, tape)?.result)
}

/// Run the organism alone on one tape, with the shared account of every E1 run.
pub(crate) fn run_organism(
    tape: &Tape,
    profile: &Profile,
    sensors: Sensors,
) -> Result<ControllerResult, Error> {
    let economy = Economy::from_profile(&Profile::t1_ref())?;
    run_controller(tape, Controller::Organism, profile, economy, sensors)
}

/// Run the organism as a logging policy: alongside its result, every decision it took,
/// with its propensity, its outcome, and what each candidate profile would have chosen
/// in the same state.
pub(crate) fn run_logged(
    tape: &Tape,
    profile: &Profile,
    sensors: Sensors,
    candidates: &[Profile],
) -> Result<(ControllerResult, Vec<Decision>), Error> {
    let economy = Economy::from_profile(&Profile::t1_ref())?;
    let mut execution = Execution::new(
        Controller::Organism,
        profile,
        economy,
        tape,
        horizon(tape),
        sensors,
    )?;
    execution.log = Some(DecisionLog::new(tape, candidates));
    let execution = drive(execution, tape)?;
    let decisions = execution.log.map(DecisionLog::finish).unwrap_or_default();
    Ok((execution.result, decisions))
}

/// Feedback is observed for 12 s past the tape's end.
fn horizon(tape: &Tape) -> Millis {
    Millis(tape.duration().0 + 12_000)
}

/// Feed the checklist E1 assumes, the tape, the 12 s drain and the closed-loop feedback to
/// one controller, in time order, and settle its result.
fn drive<'a>(mut execution: Execution<'a>, tape: &'a Tape) -> Result<Execution<'a>, Error> {
    let checklist = Record {
        event: Event::Checklist {
            now: Millis(0),
            actionable: CHECKLIST_ACTIONABLE,
        },
        annotation: Annotation::Clock,
    };
    let drain = (1..=12).map(|second| Record {
        event: Event::Tick {
            now: Millis(tape.duration().0 + second * 1000),
        },
        annotation: Annotation::Clock,
    });
    let mut events = std::iter::once(checklist)
        .chain(tape.records().iter().cloned())
        .chain(drain)
        .peekable();
    let mut processed = 0;
    loop {
        let endogenous = match (events.peek(), execution.feedback.pending.first_key_value()) {
            (Some(exogenous), Some(((at, _), _))) => *at < exogenous.event.now(),
            (None, Some(_)) => true,
            _ => false,
        };
        let record = if endogenous {
            execution.result.feedback_events += 1;
            execution
                .feedback
                .pending
                .pop_first()
                .map(|(_, record)| record)
        } else {
            events.next()
        };
        let Some(record) = record else {
            break;
        };
        if processed >= MAX_PROCESSED {
            return Err(Error::Limit("processed events"));
        }
        processed += 1;
        execution.process(&record)?;
    }
    execution.result.served_turns = execution.scorer.served.len() as u64;
    execution.result.served_requests = execution.scorer.served_requests() as u64;
    execution.result.strata = execution.scorer.strata();
    execution.result.turns_by_condition = execution.scorer.turns_by_condition(tape);
    execution.result.final_balance = execution.account.balance;
    execution.result.credited_refill = execution.account.credited_refill;
    execution.result.rounding_credit = execution.account.rounding_credit;
    execution.result.feedback_after_horizon = execution.feedback.censored;
    Ok(execution)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EpisodeId, Turn, TurnId};
    fn fixture(relevant: bool) -> Tape {
        let record = Record {
            event: Event::Speech {
                now: Millis(1000),
                cue: SpeechCue {
                    energy: 0.9,
                    vad_confidence: 0.9,
                    duration_ms: 1000,
                    keyword: true,
                    speaker_sim: None,
                    media: None,
                    turn_complete: None,
                    directed: None,
                    direction: None,
                },
            },
            annotation: Annotation::Speech {
                segment: SegmentId(0),
                episode: Some(EpisodeId(0)),
                source: if relevant {
                    Stimulus::Request(TurnId(0))
                } else {
                    Stimulus::FalseKeyword
                },
                pause_style: None,
            },
        };
        let turns = if relevant {
            vec![Turn {
                id: TurnId(0),
                episode: EpisodeId(0),
                segments: vec![SegmentId(0)],
                available_at: Millis(1000),
                deadline: Millis(11_000),
                kind: crate::tape::TurnKind::Single,
                gap: None,
                pause_style: None,
            }]
        } else {
            vec![]
        };
        Tape::new(
            TapeKind::Fixture,
            0,
            Millis(3000),
            vec![record],
            turns,
            vec![],
            vec![crate::tape::RoomCondition::default()],
        )
        .unwrap()
    }
    fn decision_trace(r: &ControllerResult) -> Vec<(ThoughtId, Millis, &Reason)> {
        r.thoughts
            .iter()
            .map(|t| (t.thought, t.at, &t.reason))
            .collect()
    }
    #[test]
    fn relabeling_changes_only_scoring_not_policy_payment_or_feedback() {
        let a = run_tape(&fixture(true)).unwrap();
        let b = run_tape(&fixture(false)).unwrap();
        for (left, right) in [
            (&a.organism, &b.organism),
            (&a.simple, &b.simple),
            (&a.fixed_window, &b.fixed_window),
        ] {
            assert_eq!(decision_trace(left), decision_trace(right));
            assert!((left.total_cost - right.total_cost).abs() < 1e-9);
            assert_eq!(left.feedback_events, right.feedback_events);
            assert_eq!(left.served_turns, 1);
            assert_eq!(right.served_turns, 0);
        }
    }
    #[test]
    fn common_economy_and_closed_loop_conserve_every_paid_call() {
        let run = run_tape(&fixture(true)).unwrap();
        for result in [&run.organism, &run.simple, &run.fixed_window] {
            assert!(result.feedback_events >= 3);
            assert_eq!(result.paid_calls, result.served_turns + result.wasted_calls);
            assert_eq!(result.paid_calls, result.thoughts.len() as u64);
            assert!(
                (result.total_cost - result.paid_calls as f64 * run.economy.thought_cost).abs()
                    < 1e-9
            );
            assert!(
                (run.economy.capacity + result.credited_refill + result.rounding_credit
                    - result.total_cost
                    - result.final_balance)
                    .abs()
                    < 1e-7
            );
            assert!(result.final_balance >= 0.0);
        }
    }
    #[test]
    fn every_thought_in_a_multi_action_event_is_paid_and_scored() {
        let tape = fixture(true);
        let profile = Profile::t1_ref();
        let economy = Economy::from_profile(&profile).unwrap();
        let mut e = Execution::new(
            Controller::Simple,
            &profile,
            economy,
            &tape,
            Millis(15_000),
            Sensors::DEFAULT,
        )
        .unwrap();
        let record = tape.records().first().unwrap();
        e.paid_thought(record, ThoughtId(1), Reason::Keyword, None, false)
            .unwrap();
        e.paid_thought(record, ThoughtId(2), Reason::Keyword, None, false)
            .unwrap();
        assert_eq!(e.result.paid_calls, 2);
        assert_eq!(e.result.duplicate_calls, 1);
        assert_eq!(e.result.wasted_calls, 1);
        assert!((e.result.total_cost - 2.0 * economy.thought_cost).abs() < 1e-9);
    }
    #[test]
    fn feedback_is_per_paid_thought_and_overflow_is_an_error() {
        let mut f = Feedback::new(Millis(10_000));
        f.reply(Millis(1000), ThoughtId(4)).unwrap();
        assert_eq!(f.pending.len(), 5);
        let record = &f.pending.first_key_value().unwrap().1;
        assert!(matches!(
            record.event,
            Event::PlaybackStarted {
                now: Millis(1400),
                utterance: UtteranceId(4)
            }
        ));
        assert!(record.annotation.turn().is_none());
        while f.pending.len() < MAX_PENDING {
            f.schedule(Record {
                event: Event::Tick { now: Millis(9000) },
                annotation: Annotation::Clock,
            })
            .unwrap();
        }
        assert!(matches!(
            f.schedule(Record {
                event: Event::Tick { now: Millis(9000) },
                annotation: Annotation::Clock
            }),
            Err(Error::Limit(_))
        ));
        let mut f = Feedback::new(Millis(1000));
        f.reply(Millis(1000), ThoughtId(1)).unwrap();
        assert_eq!(f.censored, 5);
        assert!(f.pending.is_empty());
    }
    #[test]
    fn closed_loop_feedback_emits_playback_started_and_finished_for_all_controllers() {
        let mut f = Feedback::new(Millis(10_000));
        f.reply(Millis(1000), ThoughtId(7)).unwrap();
        let events: Vec<_> = f.pending.values().map(|r| r.event.clone()).collect();
        assert_eq!(events.len(), 5);
        assert_eq!(
            events[0],
            Event::PlaybackStarted {
                now: Millis(1400),
                utterance: UtteranceId(7),
            }
        );
        assert_eq!(
            events[4],
            Event::PlaybackFinished {
                now: Millis(3000),
                utterance: UtteranceId(7),
                interrupted: false,
            }
        );

        // Verify baseline ignores both events without returning actions
        let mut simple = Baseline::new(false);
        assert!(simple.step(&events[0], true).is_empty());
        assert!(simple.step(&events[4], true).is_empty());

        let mut fixed = Baseline::new(true);
        assert!(fixed.step(&events[0], true).is_empty());
        assert!(fixed.step(&events[4], true).is_empty());

        // Verify organism tracks playback state on these events
        let mut organism = Organism::new(Profile::t1_ref()).unwrap();
        assert!(!organism.is_speaking());
        organism.step(&events[0]);
        assert!(organism.is_speaking());
        organism.step(&events[4]);
        assert!(!organism.is_speaking());
        assert!(organism.is_hangover());
    }
    #[test]
    fn calibration_42_end_to_end_is_auditable_and_cannot_report_overall_pass() {
        let a = run_tape(&crate::e1a(42).unwrap()).unwrap();
        let b = run_tape(&crate::e1b(42).unwrap()).unwrap();
        for run in [&a, &b] {
            for result in [&run.organism, &run.simple, &run.fixed_window] {
                assert_eq!(result.paid_calls, result.served_turns + result.wasted_calls);
                assert_eq!(result.waste_breakdown.total(), result.wasted_calls);
                assert_eq!(
                    result.waste_breakdown.by_reason.values().sum::<u64>(),
                    result.wasted_calls
                );
                assert_eq!(
                    result.waste_breakdown.by_stimulus.values().sum::<u64>(),
                    result.wasted_calls
                );
                assert_eq!(result.noise_breakdown.total(), result.calls_in_noise);
                assert_eq!(
                    result.noise_breakdown.by_reason.values().sum::<u64>(),
                    result.calls_in_noise
                );
                assert_eq!(
                    result.noise_breakdown.by_stimulus.values().sum::<u64>(),
                    result.calls_in_noise
                );
                assert!(
                    (run.economy.capacity + result.credited_refill + result.rounding_credit
                        - result.total_cost
                        - result.final_balance)
                        .abs()
                        < 1e-7
                );
                assert!(result.synthetic_self_ignitions <= result.wasted_calls);
            }
        }
        let report = crate::Report::new(a, b).unwrap();
        assert_ne!(report.overall, crate::Status::Pass);
        let text = report.to_string();
        assert!(text.contains(BENCHMARK_VERSION));
        assert!(text.contains("not evaluated"));
        assert!(text.contains("noise by reason:"));
        assert!(text.contains("noise by stimulus:"));
        assert!(text.contains("waste by stimulus:"));
        assert!(text.contains("turn segments:"));
        assert!(!text.contains("Overall: PASS"));
        assert_eq!(
            report
                .criteria
                .iter()
                .filter(|c| c.status == crate::Status::NotEvaluated)
                .count(),
            3
        );
    }
    #[test]
    fn waste_breakdown_sums_to_wasted_calls() {
        let a = run_tape(&crate::e1a(42).unwrap()).unwrap();
        let b = run_tape(&crate::e1b(42).unwrap()).unwrap();
        for run in [&a, &b] {
            for result in [&run.organism, &run.simple, &run.fixed_window] {
                assert_eq!(result.waste_breakdown.total(), result.wasted_calls);
                assert_eq!(
                    result.waste_breakdown.by_reason.values().sum::<u64>(),
                    result.wasted_calls
                );
                assert_eq!(
                    result.waste_breakdown.by_stimulus.values().sum::<u64>(),
                    result.wasted_calls
                );
                assert_eq!(
                    result.waste_breakdown.by_cell.values().sum::<u64>(),
                    result.wasted_calls
                );
            }
        }
    }
    // Bounded regression test asserts accounting breakdown across multiple stimuli.
    #[allow(clippy::too_many_lines)]
    #[test]
    fn noise_breakdown_regression_and_accounting() {
        let noise_interval = Interval {
            start: Millis(500),
            end: Millis(30_000),
        };
        let records = vec![
            Record {
                event: Event::Speech {
                    now: Millis(1000),
                    cue: SpeechCue {
                        energy: 0.5,
                        vad_confidence: 0.1,
                        duration_ms: 500,
                        keyword: false,
                        speaker_sim: None,
                        media: None,
                        turn_complete: None,
                        directed: None,
                        direction: None,
                    },
                },
                annotation: Annotation::Speech {
                    segment: SegmentId(1),
                    episode: Some(EpisodeId(10)),
                    source: Stimulus::Motor,
                    pause_style: None,
                },
            },
            Record {
                event: Event::Speech {
                    now: Millis(4000),
                    cue: SpeechCue {
                        energy: 0.2,
                        vad_confidence: 0.05,
                        duration_ms: 500,
                        keyword: false,
                        speaker_sim: None,
                        media: None,
                        turn_complete: None,
                        directed: None,
                        direction: None,
                    },
                },
                annotation: Annotation::Speech {
                    segment: SegmentId(2),
                    episode: Some(EpisodeId(10)),
                    source: Stimulus::Ventilation,
                    pause_style: None,
                },
            },
            Record {
                event: Event::Speech {
                    now: Millis(15_000),
                    cue: SpeechCue {
                        energy: 0.9,
                        vad_confidence: 0.9,
                        duration_ms: 1500,
                        keyword: false,
                        speaker_sim: None,
                        media: None,
                        turn_complete: None,
                        directed: None,
                        direction: None,
                    },
                },
                annotation: Annotation::Speech {
                    segment: SegmentId(3),
                    episode: Some(EpisodeId(10)),
                    source: Stimulus::Tv,
                    pause_style: None,
                },
            },
        ];
        let tape = Tape::new(
            TapeKind::Fixture,
            0,
            Millis(35_000),
            records,
            vec![],
            vec![noise_interval],
            vec![crate::tape::RoomCondition::default()],
        )
        .unwrap();

        let run = run_tape(&tape).unwrap();
        for result in [&run.organism, &run.simple, &run.fixed_window] {
            assert_eq!(result.noise_breakdown.total(), result.calls_in_noise);
            assert_eq!(
                result.noise_breakdown.by_reason.values().sum::<u64>(),
                result.calls_in_noise
            );
            assert_eq!(
                result.noise_breakdown.by_stimulus.values().sum::<u64>(),
                result.calls_in_noise
            );
        }

        assert!(run.simple.calls_in_noise >= 1);
        assert_eq!(
            run.simple.noise_breakdown.reason_count(NoiseReason::Speech),
            run.simple.calls_in_noise
        );
        assert_eq!(
            run.simple.noise_breakdown.stimulus_count(NoiseStimulus::Tv),
            run.simple.calls_in_noise
        );
        assert_eq!(
            run.simple
                .noise_breakdown
                .stimulus_count(NoiseStimulus::Motor),
            0
        );
        assert_eq!(
            run.simple
                .noise_breakdown
                .stimulus_count(NoiseStimulus::Ventilation),
            0
        );
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use crate::{EpisodeId, Turn, TurnId};
    #[test]
    fn original_baseline_does_not_serve_the_unheard_split_continuation() {
        let segments = [(1000, 250, true), (2000, 700, false)]
            .into_iter()
            .enumerate()
            .map(|(index, (end, duration, keyword))| Record {
                event: Event::Speech {
                    now: Millis(end),
                    cue: SpeechCue {
                        energy: 0.9,
                        vad_confidence: 0.9,
                        duration_ms: duration,
                        keyword,
                        speaker_sim: None,
                        media: None,
                        turn_complete: None,
                        directed: None,
                        direction: None,
                    },
                },
                annotation: Annotation::Speech {
                    segment: SegmentId(u32::try_from(index).unwrap()),
                    episode: Some(EpisodeId(0)),
                    source: Stimulus::Request(TurnId(0)),
                    pause_style: None,
                },
            })
            .collect();
        let tape = Tape::new(
            TapeKind::Fixture,
            0,
            Millis(3000),
            segments,
            vec![Turn {
                id: TurnId(0),
                episode: EpisodeId(0),
                segments: vec![SegmentId(0), SegmentId(1)],
                available_at: Millis(2000),
                deadline: Millis(12_000),
                kind: crate::tape::TurnKind::Single,
                gap: None,
                pause_style: None,
            }],
            vec![],
            vec![crate::tape::RoomCondition::default()],
        )
        .unwrap();
        let run = run_tape(&tape).unwrap();
        assert_eq!(run.simple.paid_calls, 1);
        assert_eq!(run.simple.served_turns, 0);
        assert_eq!(run.simple.wasted_calls, 1);
    }
    #[test]
    fn outer_pool_rejects_baseline_work_after_exhaustion() {
        let records = [1000, 7000, 13_000]
            .into_iter()
            .enumerate()
            .map(|(index, now)| Record {
                event: Event::Speech {
                    now: Millis(now),
                    cue: SpeechCue {
                        energy: 0.9,
                        vad_confidence: 0.9,
                        duration_ms: 1000,
                        keyword: true,
                        speaker_sim: None,
                        media: None,
                        turn_complete: None,
                        directed: None,
                        direction: None,
                    },
                },
                annotation: Annotation::Speech {
                    segment: SegmentId(u32::try_from(index).unwrap()),
                    episode: Some(EpisodeId(0)),
                    source: Stimulus::FalseKeyword,
                    pause_style: None,
                },
            })
            .collect();
        let tape = Tape::new(
            TapeKind::Fixture,
            0,
            Millis(15_000),
            records,
            vec![],
            vec![],
            vec![crate::tape::RoomCondition::default()],
        )
        .unwrap();
        let economy = Economy {
            capacity: 2.0,
            refill_per_hour: 2.0,
            thought_cost: 1.0,
        };
        let result = run_controller(
            &tape,
            Controller::Simple,
            &Profile::t1_ref(),
            economy,
            Sensors::DEFAULT,
        )
        .unwrap();
        assert_eq!(result.paid_calls, 2);
        assert_eq!(result.rejected_obligations, 1);
        assert!(result.final_balance < 1.0);
    }
    #[test]
    fn a_new_thought_cuts_the_playback_in_flight_like_the_runtime() {
        let mut feedback = Feedback::new(Millis(10_000));
        // Playback 1400..3000, echoes at 1900 and 2700, reply at 3000.
        feedback.reply(Millis(1000), ThoughtId(1)).unwrap();
        feedback.pending.pop_first().unwrap();
        feedback.supersede(Millis(2000)).unwrap();
        let left: Vec<_> = feedback.pending.values().map(|r| r.event.clone()).collect();
        assert_eq!(
            left,
            vec![Event::PlaybackFinished {
                now: Millis(2000),
                utterance: UtteranceId(1),
                interrupted: true,
            }]
        );
        assert_eq!(
            feedback.playbacks.get(&UtteranceId(1)).map(|i| i.end),
            Some(Millis(2000))
        );

        // A thought superseded before its playback starts is never heard at all.
        feedback.pending.clear();
        feedback.reply(Millis(5000), ThoughtId(2)).unwrap();
        feedback.supersede(Millis(5100)).unwrap();
        assert!(feedback.pending.is_empty());
        assert!(!feedback.playbacks.contains_key(&UtteranceId(2)));
    }
    #[test]
    fn barge_in_overlap_uses_real_playback_intervals_not_its_label_alone() {
        let tape = Tape::new(
            TapeKind::Fixture,
            0,
            Millis(4000),
            vec![],
            vec![],
            vec![],
            vec![crate::tape::RoomCondition::default()],
        )
        .unwrap();
        let mut result = ControllerResult::new(Controller::Simple, &tape);
        let mut feedback = Feedback::new(Millis(10_000));
        feedback.reply(Millis(1000), ThoughtId(1)).unwrap();
        for now in [2200, 4000] {
            feedback.observe_barge_in(
                &Record {
                    event: Event::Speech {
                        now: Millis(now),
                        cue: SpeechCue {
                            duration_ms: 600,
                            ..SpeechCue::default()
                        },
                    },
                    annotation: Annotation::Speech {
                        segment: SegmentId(0),
                        episode: Some(EpisodeId(0)),
                        source: Stimulus::BargeIn(TurnId(0)),
                        pause_style: None,
                    },
                },
                &mut result,
            );
        }
        assert_eq!(result.barge_in_segments, 2);
        assert_eq!(result.overlapping_barge_in_segments, 1);
    }
}

#[cfg(test)]
mod ablation_tests {
    use super::*;
    use crate::{Report, Summary, e1a, e1b};

    /// 64-bit FNV-1a, enough to fingerprint a tape or a result against a known answer.
    struct Fnv(u64);
    impl Fnv {
        fn new() -> Self {
            Self(0xcbf2_9ce4_8422_2325)
        }
        fn bytes(&mut self, bytes: &[u8]) {
            for byte in bytes {
                self.0 ^= u64::from(*byte);
                self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
            }
        }
        fn u64(&mut self, value: u64) {
            self.bytes(&value.to_le_bytes());
        }
        fn f32(&mut self, value: f32) {
            self.u64(u64::from(value.to_bits()));
        }
        fn reading(&mut self, value: Option<f32>) {
            self.u64(value.map_or(u64::MAX, |x| u64::from(x.to_bits())));
        }
    }

    /// Every field a tape of benchmark 3.0.0 had: timing, features, the three
    /// original readings, annotations, turns and conditions. Not the directedness
    /// reading, which 3.0.0 did not have.
    fn fingerprint(tape: &Tape) -> u64 {
        let mut h = Fnv::new();
        for record in tape.records() {
            h.u64(record.event.now().0);
            match (&record.event, &record.annotation) {
                (
                    Event::Speech { cue, .. },
                    Annotation::Speech {
                        segment,
                        episode,
                        source,
                        pause_style,
                    },
                ) => {
                    h.f32(cue.energy);
                    h.u64(u64::from(cue.duration_ms));
                    h.f32(cue.vad_confidence);
                    h.u64(u64::from(cue.keyword));
                    h.reading(cue.speaker_sim);
                    h.reading(cue.media);
                    h.reading(cue.turn_complete);
                    h.u64(u64::from(segment.0));
                    h.u64(episode.map_or(u64::MAX, |e| u64::from(e.0)));
                    h.bytes(format!("{source:?}").as_bytes());
                    h.reading(*pause_style);
                }
                _ => h.u64(0),
            }
        }
        for turn in tape.turns() {
            h.u64(u64::from(turn.id.0));
            for segment in &turn.segments {
                h.u64(u64::from(segment.0));
            }
            h.u64(turn.available_at.0);
            h.u64(turn.deadline.0);
        }
        for c in tape.conditions() {
            h.bytes(format!("{:?}{:?}{:?}", c.distance, c.tv, c.tv_content).as_bytes());
            h.f32(c.acoustics);
            h.f32(c.show);
        }
        h.0
    }

    /// Every controller's full result, as the JSON report writes it.
    fn results(run: &ExperimentRun) -> u64 {
        let mut h = Fnv::new();
        for result in [&run.organism, &run.simple, &run.fixed_window] {
            h.bytes(serde_json::to_string(result).unwrap().as_bytes());
        }
        h.0
    }

    #[test]
    fn the_default_sensors_reproduce_benchmark_3_0_0_exactly() {
        // Tape fingerprints computed with the benchmark 3.0.0 generator (commit e54e8bf):
        // no later stream moved a reading. Outcomes are reducer v13's: a run without the
        // detector decided every call as v9 did until v13, whose presence gate turns away
        // overheard speech before the owner was ever heard (seed 7's E1a, seed 42's E1b;
        // the other two runs are v9's to the bit).
        for (seed, tapes, outcomes) in [
            (
                7,
                [0x8c07_f602_b73b_bf00, 0x5090_69ff_e027_008b],
                [0xc1f0_c11a_9094_60fc, 0xb04d_2f78_5781_6f6f],
            ),
            (
                42,
                [0x931b_93da_b8ea_fac2, 0x4d66_ddc9_dc28_49c4],
                [0xad0d_7822_cec3_7b65, 0xaa2d_2783_7838_f7b8],
            ),
        ] {
            let pair = [e1a(seed).unwrap(), e1b(seed).unwrap()];
            for ((tape, fingerprinted), outcome) in pair.iter().zip(tapes).zip(outcomes) {
                assert_eq!(fingerprint(tape), fingerprinted, "seed {seed} tape");
                let run = run_tape(tape).unwrap();
                assert_eq!(run.sensors, Sensors::DEFAULT);
                assert_eq!(results(&run), outcome, "seed {seed} results");
            }
        }
    }

    #[test]
    fn sensors_that_are_off_are_stripped_and_the_rest_pass_untouched() {
        let cue = SpeechCue {
            energy: 0.7,
            duration_ms: 900,
            vad_confidence: 0.8,
            keyword: true,
            speaker_sim: Some(0.6),
            media: Some(0.2),
            turn_complete: Some(0.9),
            directed: Some(0.95),
            direction: Some([0.6, 0.8]),
        };
        assert_eq!(
            Sensors::DEFAULT.sense(cue),
            SpeechCue {
                directed: None,
                direction: None,
                ..cue
            }
        );
        assert_eq!(
            Sensors::WITH_DIRECTED.sense(cue),
            SpeechCue {
                direction: None,
                ..cue
            }
        );
        assert_eq!(
            Sensors::WITH_DIRECTION.sense(cue),
            SpeechCue {
                directed: None,
                ..cue
            }
        );
        let all = Sensors {
            directed: true,
            ..Sensors::WITH_DIRECTION
        };
        assert_eq!(all.sense(cue), cue);
        let none = Sensors {
            speaker: false,
            media: false,
            turn: false,
            directed: false,
            direction: false,
        };
        assert_eq!(
            none.sense(cue),
            SpeechCue {
                speaker_sim: None,
                media: None,
                turn_complete: None,
                directed: None,
                direction: None,
                ..cue
            }
        );
        let tick = Event::Tick { now: Millis(5) };
        assert_eq!(none.perceive(&tick), tick);
        assert_eq!(
            Sensors::DEFAULT.perceive(&Event::Speech {
                now: Millis(1_000),
                cue
            }),
            Event::Speech {
                now: Millis(1_000),
                cue: Sensors::DEFAULT.sense(cue)
            }
        );
        // Enton's own echo carries a reading too, stripped by the same boundary.
        let mut feedback = Feedback::new(Millis(10_000));
        feedback.reply(Millis(1_000), ThoughtId(1)).unwrap();
        let echoes: Vec<_> = feedback
            .pending
            .values()
            .filter_map(|record| match record.event {
                Event::Speech { cue, .. } => Some(cue),
                _ => None,
            })
            .collect();
        assert_eq!(echoes.len(), 2);
        for echo in echoes {
            assert_eq!(echo.directed, Some(SELF_ECHO_DIRECTED));
            assert_eq!(Sensors::DEFAULT.sense(echo).directed, None);
        }
        assert_eq!(none.to_string(), "none");
        assert_eq!(
            Sensors::WITH_DIRECTED.to_string(),
            "speaker verification, media tagger, end of turn, directedness"
        );
        assert_eq!(
            all.to_string(),
            "speaker verification, media tagger, end of turn, directedness, direction of arrival"
        );
        // A report without the array serializes its sensors as before the array existed.
        let json = serde_json::to_value(Sensors::WITH_DIRECTED).unwrap();
        assert!(json.get("direction").is_none(), "{json}");
        assert_eq!(
            serde_json::to_value(all).unwrap()["direction"],
            serde_json::Value::Bool(true)
        );
    }

    #[test]
    fn without_the_array_no_direction_setting_moves_a_call() {
        // Tapes carry a direction reading on every cue; withheld, the most eager settings
        // decide every call as the shipped ones do.
        let mut eager = Profile::t1_ref();
        eager.source.tv_direction_min_lines = 1.0;
        eager.source.tv_direction_half_life_ms = 1;
        eager.source.tv_caution_confinement = enton_core::TvCautionConfinement::Always;
        for tape in [e1a(7).unwrap(), e1b(7).unwrap()] {
            for sensors in [Sensors::DEFAULT, Sensors::WITH_DIRECTED] {
                let shipped = run_tape_with(&tape, &Profile::t1_ref(), sensors).unwrap();
                let other = run_tape_with(&tape, &eager, sensors).unwrap();
                assert_eq!(results(&shipped), results(&other));
            }
        }
    }

    #[test]
    fn only_the_organism_hears_the_array_and_reports_say_so() {
        let (a, b) = (e1a(42).unwrap(), e1b(42).unwrap());
        let profile = Profile::t1_ref();
        let both = Sensors {
            directed: true,
            ..Sensors::WITH_DIRECTION
        };
        for sensors in [Sensors::WITH_DIRECTION, both] {
            let without = Sensors {
                direction: false,
                ..sensors
            };
            let off = run_tape_with(&a, &profile, without).unwrap();
            let on = run_tape_with(&a, &profile, sensors).unwrap();
            assert_eq!(off.simple, on.simple);
            assert_eq!(off.fixed_window, on.fixed_window);
            assert_eq!(on.sensors, sensors);
            assert_ne!(off.organism.thoughts, on.organism.thoughts);
        }
        let report = Report::new(
            run_tape_with(&a, &profile, Sensors::WITH_DIRECTION).unwrap(),
            run_tape_with(&b, &profile, Sensors::WITH_DIRECTION).unwrap(),
        )
        .unwrap();
        assert!(report.to_string().contains(
            "Sensors: speaker verification, media tagger, end of turn, direction of arrival\n"
        ));
        let summary = Summary::from_reports(std::slice::from_ref(&report)).unwrap();
        let text = summary.to_string();
        assert!(text.contains(
            "sensors: speaker verification, media tagger, end of turn, direction of arrival"
        ));
        assert!(text.contains("  turns by TV: off "), "{text}");
        let json = serde_json::to_value(&summary).unwrap();
        assert_eq!(json["sensors"]["direction"], serde_json::Value::Bool(true));
        let pooled: u64 = summary
            .e1a_turns_by_condition
            .values()
            .map(|tally| tally.total)
            .sum();
        assert_eq!(pooled, summary.e1a_total_turns);
    }

    #[test]
    fn only_the_organism_hears_the_detector_and_reports_say_so() {
        let (a, b) = (e1a(42).unwrap(), e1b(42).unwrap());
        let profile = Profile::t1_ref();
        let without = [
            run_tape_with(&a, &profile, Sensors::DEFAULT).unwrap(),
            run_tape_with(&b, &profile, Sensors::DEFAULT).unwrap(),
        ];
        let with = [
            run_tape_with(&a, &profile, Sensors::WITH_DIRECTED).unwrap(),
            run_tape_with(&b, &profile, Sensors::WITH_DIRECTED).unwrap(),
        ];
        // The comparators read no sensor: the same calls either way.
        for (off, on) in without.iter().zip(&with) {
            assert_eq!(off.simple, on.simple);
            assert_eq!(off.fixed_window, on.fixed_window);
            assert_eq!(on.sensors, Sensors::WITH_DIRECTED);
        }
        assert_ne!(without[0].organism.thoughts, with[0].organism.thoughts);

        let [a_off, b_off] = without;
        let [a_on, b_on] = with;
        // A report pairs runs that perceived the same sensors, and says which.
        assert!(Report::new(a_off.clone(), b_on.clone()).is_err());
        let off = Report::new(a_off, b_off).unwrap();
        let on = Report::new(a_on, b_on).unwrap();
        assert!(
            off.to_string()
                .contains("Sensors: speaker verification, media tagger, end of turn\n")
        );
        assert!(
            on.to_string().contains(
                "Sensors: speaker verification, media tagger, end of turn, directedness\n"
            )
        );
        let json = serde_json::to_value(&on).unwrap();
        assert_eq!(json["sensors"]["directed"], serde_json::Value::Bool(true));
        assert_eq!(
            json["e1a"]["sensors"]["directed"],
            serde_json::Value::Bool(true)
        );
        // So does a summary, which never pools different sensor sets.
        let summary = Summary::from_reports(std::slice::from_ref(&on)).unwrap();
        assert!(
            summary
                .to_string()
                .contains("sensors: speaker verification, media tagger, end of turn, directedness")
        );
        assert!(Summary::from_reports(&[off, on]).is_err());
    }
}

#[cfg(test)]
mod checklist_tests {
    use super::*;
    use crate::tape::{RoomCondition, TurnKind};
    use crate::{EpisodeId, Turn, TurnId};

    /// Half an hour of ticks; with `owner`, the owner asks Enton something two seconds in,
    /// so someone is home all along; with `checklist`, the tape reads the checklist at the
    /// start.
    fn quiet_half_hour(checklist: Option<bool>, owner: bool) -> Tape {
        let duration = 1_800_000;
        let mut records: Vec<Record> = checklist
            .map(|actionable| Record {
                event: Event::Checklist {
                    now: Millis(0),
                    actionable,
                },
                annotation: Annotation::Clock,
            })
            .into_iter()
            .collect();
        for now in (0..=duration).step_by(1_000) {
            records.push(Record {
                event: Event::Tick { now: Millis(now) },
                annotation: Annotation::Clock,
            });
            if owner && now == 1_000 {
                records.push(Record {
                    event: Event::Speech {
                        now: Millis(2_000),
                        cue: SpeechCue {
                            energy: 0.9,
                            vad_confidence: 0.9,
                            duration_ms: 1_500,
                            keyword: true,
                            ..SpeechCue::default()
                        },
                    },
                    annotation: Annotation::Speech {
                        segment: SegmentId(0),
                        episode: Some(EpisodeId(0)),
                        source: Stimulus::Request(TurnId(0)),
                        pause_style: None,
                    },
                });
            }
        }
        let turns = if owner {
            vec![Turn {
                id: TurnId(0),
                episode: EpisodeId(0),
                segments: vec![SegmentId(0)],
                available_at: Millis(2_000),
                deadline: Millis(12_000),
                kind: TurnKind::Single,
                gap: None,
                pause_style: None,
            }]
        } else {
            Vec::new()
        };
        Tape::new(
            TapeKind::Fixture,
            0,
            Millis(duration),
            records,
            turns,
            vec![],
            vec![RoomCondition::default(); 10],
        )
        .unwrap()
    }

    /// t1-ref with drives that reach their threshold in about 25 minutes.
    fn eager() -> Profile {
        let mut profile = Profile::t1_ref();
        profile.ignition.threshold = 0.01;
        profile.ignition.hysteresis = 0.002;
        profile.ignition.ema_alpha = 1.0;
        profile
    }

    fn drive_thoughts(tape: &Tape) -> Vec<PaidThought> {
        run_organism(tape, &eager(), Sensors::DEFAULT)
            .unwrap()
            .thoughts
            .into_iter()
            .filter(|thought| matches!(thought.reason, Reason::Drive(_)))
            .collect()
    }

    #[test]
    fn e1_assumes_something_to_check_so_a_ready_drive_thinks_with_the_owner_home() {
        let thoughts = drive_thoughts(&quiet_half_hour(None, true));
        let [thought] = thoughts.as_slice() else {
            panic!("expected one drive thought, got {thoughts:?}");
        };
        assert_eq!(thought.reason, Reason::Drive("curiosity".into()));
        assert_eq!(thought.trigger, Trigger::Internal);
        assert_eq!(thought.credit, Credit::Waste);
        // The tape saying so explicitly changes nothing.
        assert_eq!(drive_thoughts(&quiet_half_hour(Some(true), true)), thoughts);
    }

    #[test]
    fn a_tape_whose_checklist_has_nothing_to_check_never_buys_a_drive_thought() {
        assert!(drive_thoughts(&quiet_half_hour(Some(false), true)).is_empty());
        // The owner's request is served all the same.
        let result = run_organism(
            &quiet_half_hour(Some(false), true),
            &eager(),
            Sensors::DEFAULT,
        )
        .unwrap();
        assert_eq!(result.served_turns, 1);
    }

    #[test]
    fn with_nobody_home_a_ready_drive_never_thinks() {
        assert!(drive_thoughts(&quiet_half_hour(None, false)).is_empty());
    }

    #[test]
    fn the_generated_tapes_never_ready_a_drive_at_t1_ref() {
        for tape in [crate::e1a(100).unwrap(), crate::e1b(100).unwrap()] {
            let result = run_organism(&tape, &Profile::t1_ref(), Sensors::DEFAULT).unwrap();
            assert!(
                result
                    .thoughts
                    .iter()
                    .all(|thought| !matches!(thought.reason, Reason::Drive(_)))
            );
        }
    }
}
