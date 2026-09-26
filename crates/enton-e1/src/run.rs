//! Equal-cost admission and closed-loop execution; annotations are only for scoring.
pub use crate::scoring::{Credit, Trigger};
use crate::{
    Annotation, BENCHMARK_VERSION, Economy, Error, Interval, Record, SegmentId, Stimulus, Tape,
    TapeKind,
};
use crate::{
    baseline::Baseline,
    economy::Account,
    report::{NoiseBreakdown, NoiseReason, NoiseStimulus},
    scoring::Scorer,
};
use enton_core::{
    Abstention, Action, Event, Millis, Organism, Profile, Reason, SpeechCue, ThoughtId, UtteranceId,
};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

const MAX_PROCESSED: usize = 100_000;
const MAX_PENDING: usize = 4096;

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
    /// Calls not earning new timely turn credit, including duplicates.
    pub wasted_calls: u64,
    /// Timely calls for an already served turn, included in waste.
    pub duplicate_calls: u64,
    /// All calls at timestamps inside E1b's half-open noise intervals.
    pub calls_in_noise: u64,
    /// Breakdown of noise-interval calls by reason and triggering stimulus.
    pub noise_breakdown: NoiseBreakdown,
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
            wasted_calls: 0,
            duplicate_calls: 0,
            calls_in_noise: 0,
            noise_breakdown: NoiseBreakdown::new(),
            synthetic_self_ignitions: 0,
            barge_in_segments: 0,
            overlapping_barge_in_segments: 0,
            feedback_events: 0,
            feedback_after_horizon: 0,
            thoughts: Vec::new(),
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
    /// Identical account/price contract for all controllers.
    pub economy: Economy,
    /// Organism result.
    pub organism: ControllerResult,
    /// Original simple controller result.
    pub simple: ControllerResult,
    /// Secondary fixed-window cost comparator result.
    pub fixed_window: ControllerResult,
}

/// Run all three policies on the same validated tape without audio, network or models.
/// Returns errors on resource overflow or a violated shared-budget invariant.
pub fn run_tape(tape: &Tape) -> Result<ExperimentRun, Error> {
    let profile = Profile::t1_ref();
    let economy = Economy::from_profile(&profile)?;
    Ok(ExperimentRun {
        version: BENCHMARK_VERSION.into(),
        kind: tape.kind(),
        seed: tape.seed(),
        economy,
        organism: run_controller(tape, Controller::Organism, &profile, economy)?,
        simple: run_controller(tape, Controller::Simple, &profile, economy)?,
        fixed_window: run_controller(tape, Controller::FixedWindow, &profile, economy)?,
    })
}

enum Machine {
    Core(Box<Organism>),
    Simple(Baseline),
}
impl Machine {
    fn new(controller: Controller, profile: &Profile) -> Self {
        match controller {
            Controller::Organism => Self::Core(Box::new(Organism::new(profile.clone()))),
            Controller::Simple => Self::Simple(Baseline::new(false)),
            Controller::FixedWindow => Self::Simple(Baseline::new(true)),
        }
    }
    fn step(&mut self, event: &Event, can_pay: bool) -> Vec<Action> {
        match self {
            Self::Core(o) => o.step(event),
            Self::Simple(b) => b.step(event, can_pay),
        }
    }
    fn attending(&self) -> bool {
        match self {
            Self::Core(o) => o.is_attending(),
            Self::Simple(_) => false,
        }
    }
}

struct Feedback {
    pending: BTreeMap<(Millis, u64), Record>,
    next_order: u64,
    next_segment: u32,
    echo_segments: BTreeSet<SegmentId>,
    playbacks: Vec<Interval>,
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
            playbacks: vec![],
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
        self.playbacks.push(Interval {
            start: Millis(now.0 + 400),
            end: Millis(now.0 + 2000),
        });
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
                    },
                },
                annotation: Annotation::Speech {
                    segment,
                    episode: None,
                    source: Stimulus::SelfEcho,
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
            },
            annotation: Annotation::Clock,
        })
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
                    .iter()
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
}
impl<'a> Execution<'a> {
    fn new(
        controller: Controller,
        profile: &Profile,
        economy: Economy,
        tape: &'a Tape,
        horizon: Millis,
    ) -> Self {
        let mut segment_sources = BTreeMap::new();
        for r in tape.records() {
            if let Annotation::Speech {
                segment, source, ..
            } = r.annotation
            {
                segment_sources.insert(segment, source);
            }
        }
        Self {
            machine: Machine::new(controller, profile),
            scorer: Scorer::new(tape),
            account: Account::new(economy),
            feedback: Feedback::new(horizon),
            result: ControllerResult::new(controller, tape),
            economy,
            tape,
            segment_sources,
        }
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

    fn process(&mut self, record: &Record) -> Result<(), Error> {
        let now = record.event.now();
        self.account.advance(now);
        self.feedback.observe_barge_in(record, &mut self.result);
        // Only Event crosses the policy boundary. No labels, segment IDs or turns.
        let actions = self.machine.step(&record.event, self.account.can_pay());
        let attending = self.machine.attending();
        for action in actions {
            match action {
                Action::Think {
                    thought, reason, ..
                } => self.paid_thought(record, thought, reason, attending)?,
                Action::Abstain { reason, why, .. } => {
                    if matches!(reason, Reason::Keyword | Reason::FollowUp) {
                        self.result.rejected_obligations += 1;
                    } else if why == Abstention::OutOfEnergy {
                        self.result.discretionary_exhaustion += 1;
                    }
                }
                other => self.scorer.observe_non_think(record, &other),
            }
        }
        self.scorer.finish_event(now, attending);
        Ok(())
    }
    fn paid_thought(
        &mut self,
        record: &Record,
        thought: ThoughtId,
        reason: Reason,
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
        | Stimulus::FalseKeyword => NoiseStimulus::OtherPerson,
        Stimulus::Motor | Stimulus::Noise => NoiseStimulus::Motor,
        Stimulus::Ventilation => NoiseStimulus::Ventilation,
        Stimulus::SelfEcho => NoiseStimulus::SelfEcho,
    }
}

fn run_controller(
    tape: &Tape,
    controller: Controller,
    profile: &Profile,
    economy: Economy,
) -> Result<ControllerResult, Error> {
    let horizon = Millis(tape.duration().0 + 12_000);
    let drain = (1..=12).map(|second| Record {
        event: Event::Tick {
            now: Millis(tape.duration().0 + second * 1000),
        },
        annotation: Annotation::Clock,
    });
    let mut events = tape.records().iter().cloned().chain(drain).peekable();
    let mut execution = Execution::new(controller, profile, economy, tape, horizon);
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
    execution.result.final_balance = execution.account.balance;
    execution.result.credited_refill = execution.account.credited_refill;
    execution.result.rounding_credit = execution.account.rounding_credit;
    execution.result.feedback_after_horizon = execution.feedback.censored;
    Ok(execution.result)
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
        let mut e = Execution::new(Controller::Simple, &profile, economy, &tape, Millis(15_000));
        let record = tape.records().first().unwrap();
        e.paid_thought(record, ThoughtId(1), Reason::Keyword, false)
            .unwrap();
        e.paid_thought(record, ThoughtId(2), Reason::Keyword, false)
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
        let mut organism = Organism::new(Profile::t1_ref());
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
                    },
                },
                annotation: Annotation::Speech {
                    segment: SegmentId(1),
                    episode: Some(EpisodeId(10)),
                    source: Stimulus::Motor,
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
                    },
                },
                annotation: Annotation::Speech {
                    segment: SegmentId(2),
                    episode: Some(EpisodeId(10)),
                    source: Stimulus::Ventilation,
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
                    },
                },
                annotation: Annotation::Speech {
                    segment: SegmentId(3),
                    episode: Some(EpisodeId(10)),
                    source: Stimulus::Tv,
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
                    },
                },
                annotation: Annotation::Speech {
                    segment: SegmentId(u32::try_from(index).unwrap()),
                    episode: Some(EpisodeId(0)),
                    source: Stimulus::Request(TurnId(0)),
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
            }],
            vec![],
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
                    },
                },
                annotation: Annotation::Speech {
                    segment: SegmentId(u32::try_from(index).unwrap()),
                    episode: Some(EpisodeId(0)),
                    source: Stimulus::FalseKeyword,
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
        )
        .unwrap();
        let economy = Economy {
            capacity: 2.0,
            refill_per_hour: 2.0,
            thought_cost: 1.0,
        };
        let result =
            run_controller(&tape, Controller::Simple, &Profile::t1_ref(), economy).unwrap();
        assert_eq!(result.paid_calls, 2);
        assert_eq!(result.rejected_obligations, 1);
        assert!(result.final_balance < 1.0);
    }
    #[test]
    fn barge_in_overlap_uses_real_playback_intervals_not_its_label_alone() {
        let tape = Tape::new(TapeKind::Fixture, 0, Millis(4000), vec![], vec![], vec![]).unwrap();
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
                    },
                },
                &mut result,
            );
        }
        assert_eq!(result.barge_in_segments, 2);
        assert_eq!(result.overlapping_barge_in_segments, 1);
    }
}
