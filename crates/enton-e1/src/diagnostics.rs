//! Opt-in, bounded episode traces from the actual closed-loop E1 executor.
//!
//! The whole tape runs, including its prehistory and synthetic feedback. Ground-truth
//! annotations select the displayed window, never what the organism perceives.
use std::fmt;

use enton_core::{Action, Budget, Event, Millis, Organism, Profile};
use serde::Serialize;

use crate::run::{Credit, PaidThought};
use crate::{
    Annotation, BENCHMARK_VERSION, ControllerResult, EpisodeId, Error, Sensors, Tape, TapeKind,
    Turn,
};

/// Maximum events retained for a single episode; later entries are counted, not hidden.
pub const MAX_TRACE_STEPS: usize = 512;

/// Pre-decision state, before the event changes budgets or attention.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Context {
    /// Short attention deadline, if any.
    pub attention_until: Option<Millis>,
    /// Longer verified-voice deadline, if any.
    pub verified_attention_until: Option<Millis>,
    /// Whether the core is waiting for the rest of a request.
    pub attending: bool,
    /// Reserved direct-request balance before this event.
    pub obligation: Budget,
    /// Optional-work balance before this event.
    pub discretionary: Budget,
    /// Decayed television belief at the event's timestamp.
    pub tv_presence: f32,
    /// Whether the body currently suppresses optional activity.
    pub torpid: bool,
    /// Voice/source evidence in nats, only for a speech event.
    pub evidence: Option<SensorEvidence>,
}

/// Evidence recomputed from exactly the masked cue supplied to the policy.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SensorEvidence {
    /// Combined evidence for the owner speaking live.
    pub owner_live: f32,
    /// Live over reproduced audio.
    pub live_over_reproduced: f32,
    /// Finished over unfinished speech.
    pub finished_over_unfinished: f32,
    /// Speech addressed to Enton over speech addressed elsewhere.
    pub addressed_over_not: f32,
    /// Evidence for the learned television direction.
    pub from_tv_direction: f32,
}

impl Context {
    pub(crate) fn capture(organism: &Organism, event: &Event) -> Self {
        let now = event.now();
        let evidence = match event {
            Event::Speech { cue, .. } => {
                let evidence = organism.evidence_as_of(now, cue);
                Some(SensorEvidence {
                    owner_live: evidence.owner_live(),
                    live_over_reproduced: evidence.live_over_reproduced,
                    finished_over_unfinished: evidence.finished_over_unfinished,
                    addressed_over_not: evidence.addressed_over_not,
                    from_tv_direction: evidence.from_tv_direction,
                })
            }
            _ => None,
        };
        Self {
            attention_until: organism.attention_until(),
            verified_attention_until: organism.verified_attention_until(),
            attending: organism.is_attending(),
            obligation: *organism.obligation_budget(),
            discretionary: *organism.discretionary_budget(),
            tv_presence: organism.tv_presence_as_of(now),
            torpid: organism.is_torpid(),
            evidence,
        }
    }
}

/// One observed execution step, including feedback and intervening distractors.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Step {
    /// Sensor-masked event delivered to the organism.
    pub event: Event,
    /// Evaluator-only attribution, never delivered to the policy.
    pub annotation: Annotation,
    /// State before reducing this event.
    pub before: Context,
    /// Actual actions, in their actual order.
    pub actions: Vec<Action>,
    /// Actual paid calls and their scoring credits for this event.
    pub paid: Vec<PaidThought>,
}

/// An annotated turn and whether the original scorer credited it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TurnOutcome {
    /// Original ground truth, including segments and deadline.
    pub turn: Turn,
    /// A timely paid call served this whole turn (not an answer-quality measurement).
    pub served: bool,
}

/// Reproducible diagnosis of one response-required episode, never an E1 pass report.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Diagnosis {
    /// Diagnostic format version, independent of benchmark distributions.
    pub schema_version: u32,
    /// Original benchmark version.
    pub benchmark_version: String,
    /// Reducer that made the decisions.
    pub reducer_version: u32,
    /// Full profile, including all calibration parameters.
    pub profile: Profile,
    /// Sensors visible to the policy.
    pub sensors: Sensors,
    /// Source population.
    pub kind: TapeKind,
    /// Reproduction seed.
    pub seed: u64,
    /// Selected episode identity.
    pub episode: EpisodeId,
    /// Inclusive displayed interval, in milliseconds.
    pub window: [Millis; 2],
    /// All response-required turns of this episode, including misses.
    pub turns: Vec<TurnOutcome>,
    /// Bounded event trace within the selected interval.
    pub steps: Vec<Step>,
    /// Further steps in the interval not retained because of the trace cap.
    pub omitted_steps: usize,
    /// Full-tape organism result, identical to an ordinary E1 run.
    pub result: ControllerResult,
}

pub(crate) struct Trace {
    pub(crate) episode: EpisodeId,
    pub(crate) start: Millis,
    pub(crate) end: Millis,
    pub(crate) turns: Vec<Turn>,
    pub(crate) steps: Vec<Step>,
    pub(crate) omitted: usize,
}

impl Trace {
    pub(crate) fn new(tape: &Tape, episode: EpisodeId) -> Result<Self, Error> {
        let turns: Vec<_> = tape
            .turns()
            .iter()
            .filter(|turn| turn.episode == episode)
            .cloned()
            .collect();
        let end = turns
            .iter()
            .map(|turn| turn.deadline)
            .max()
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "episode {} has no response-required turns",
                    episode.0
                ))
            })?;
        let start = tape
            .records()
            .iter()
            .filter(|record| {
                matches!(record.annotation,
            Annotation::Speech { episode: Some(id), .. } if id == episode)
            })
            .map(|record| record.event.now())
            .min()
            .ok_or_else(|| {
                Error::Invalid(format!("episode {} has no speech segments", episode.0))
            })?;
        Ok(Self {
            episode,
            start,
            end,
            turns,
            steps: Vec::new(),
            omitted: 0,
        })
    }

    pub(crate) fn contains(&self, now: Millis) -> bool {
        self.start <= now && now <= self.end
    }

    pub(crate) fn record(
        &mut self,
        event: &Event,
        annotation: &Annotation,
        before: Context,
        actions: &[Action],
        paid: &[PaidThought],
    ) {
        if self.steps.len() == MAX_TRACE_STEPS {
            self.omitted += 1;
            return;
        }
        self.steps.push(Step {
            event: event.clone(),
            annotation: annotation.clone(),
            before,
            actions: actions.to_vec(),
            paid: paid.to_vec(),
        });
    }
}

/// Diagnose one episode without audio, network, model calls or policy changes.
/// The full tape is replayed so earlier TV beliefs, budgets and feedback are preserved.
pub fn diagnose_episode(
    tape: &Tape,
    profile: &Profile,
    sensors: Sensors,
    episode: EpisodeId,
) -> Result<Diagnosis, Error> {
    let trace = Trace::new(tape, episode)?;
    let (result, trace) = crate::run::run_diagnosed(tape, profile, sensors, trace)?;
    let turns = trace
        .turns
        .into_iter()
        .map(|turn| {
            let served = result
                .thoughts
                .iter()
                .any(|thought| thought.credit == Credit::Served(turn.id));
            TurnOutcome { turn, served }
        })
        .collect();
    Ok(Diagnosis {
        schema_version: 1,
        benchmark_version: BENCHMARK_VERSION.to_owned(),
        reducer_version: enton_core::REDUCER_VERSION,
        profile: profile.clone(),
        sensors,
        kind: tape.kind(),
        seed: tape.seed(),
        episode: trace.episode,
        window: [trace.start, trace.end],
        turns,
        steps: trace.steps,
        omitted_steps: trace.omitted,
        result,
    })
}

impl fmt::Display for Diagnosis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "E1 episode diagnosis | {:?} seed {} episode {} | benchmark {} reducer {}",
            self.kind, self.seed, self.episode.0, self.benchmark_version, self.reducer_version
        )?;
        writeln!(
            f,
            "Full tape replayed; displayed window {:?}. Synthetic paid-call proxy, not answer quality or physical E1 validation.",
            self.window
        )?;
        for outcome in &self.turns {
            writeln!(
                f,
                "  turn {} ({}) {}: segments {:?}, available {} ms, deadline {} ms",
                outcome.turn.id.0,
                outcome.turn.kind,
                if outcome.served { "served" } else { "MISSED" },
                outcome.turn.segments,
                outcome.turn.available_at.0,
                outcome.turn.deadline.0
            )?;
        }
        for step in &self.steps {
            writeln!(
                f,
                "  {} ms {:?} -> {:?}",
                step.event.now().0,
                step.annotation,
                step.actions
            )?;
            writeln!(
                f,
                "    before: attending={}, obligation={:.2}, discretionary={:.2}, TV={:.2}; evidence={:?}",
                step.before.attending,
                step.before.obligation.available,
                step.before.discretionary.available,
                step.before.tv_presence,
                step.before.evidence
            )?;
            if !step.paid.is_empty() {
                writeln!(f, "    paid: {:?}", step.paid)?;
            }
        }
        writeln!(
            f,
            "Trace: {} shown, {} omitted (cap {}).",
            self.steps.len(),
            self.omitted_steps,
            MAX_TRACE_STEPS
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_trace_is_bounded_and_explicit_about_omissions() {
        let tape = crate::e1a(100).unwrap();
        let episode = tape.turns().first().unwrap().episode;
        let mut trace = Trace::new(&tape, episode).unwrap();
        let organism = Organism::new(Profile::t1_ref()).unwrap();
        let event = Event::Tick { now: trace.start };
        for _ in 0..MAX_TRACE_STEPS + 3 {
            trace.record(
                &event,
                &Annotation::Clock,
                Context::capture(&organism, &event),
                &[],
                &[],
            );
        }
        assert_eq!(trace.steps.len(), MAX_TRACE_STEPS);
        assert_eq!(trace.omitted, 3);
    }
}
