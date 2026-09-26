//! Off-policy evaluation of the evidence thresholds from one logged, exploring run.
//!
//! The organism runs as a logging policy: a cue that an evidence objection turns away,
//! close to the threshold, thinks anyway with a small probability, and every such coin
//! flip is logged with its propensity (see [`enton_core::ExplorationPolicy`]). A family
//! of candidate profiles is then evaluated from that single log: at every logged decision
//! the candidate's own choice is asked of the logging organism's exact state
//! ([`enton_core::Organism::counterfactual`]), and an outcome is known only where the log
//! thought (the turn it served, or waste), as it would be in production. The tape's
//! ground truth is that outcome, and it is read for nothing else, except to measure the
//! estimator: each candidate is also run for real.
//!
//! Three numbers are compared for every candidate:
//!
//! - **actual**: the candidate run on the tape, closed loop;
//! - **in context**: the per-decision model with every outcome known, the candidate's
//!   choice at each logged state credited with the ground truth. It is what the
//!   estimators converge to with unlimited exploration, so its gap to `actual` is the
//!   bias of evaluating one decision at a time: policies interact over time (a served
//!   turn keeps the window open for the next, a thought moves budgets, cooldowns and
//!   Enton's own echo), and the log only visits the states its own choices led to;
//! - **estimates** from the log alone: inverse propensity scoring (IPS) and its
//!   self-normalized form, whose gap to `in context` is sampling error, plus whatever
//!   the log cannot support.
//!
//! Served turns are estimated per decision. A request (an episode) is served only if all
//! of its turns are, so it is estimated per episode: the candidate's outcome is known
//! when every thought it would take inside the episode's span was also a logged thought,
//! weighted by the inverse probability of that; episodes where the candidate would think
//! where the log never could are unsupported and counted apart. Paid calls need no
//! outcome at all: they are counted directly from the candidates' choices.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use enton_core::{Action, Event, Millis, Organism, PlaybackStatus, Profile};
use serde::Serialize;

use crate::run::{Credit, PaidThought, run_logged, run_organism};
use crate::{Annotation, BENCHMARK_VERSION, Error, Record, Sensors, Tape, TurnId, e1a, e1b};

/// At most this many decisions are logged per tape.
const MAX_DECISIONS: usize = 100_000;

/// Two-sided 95% normal quantile.
const Z95: f64 = 1.959_964;

/// What a decision was, as far as its outcome is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Choice {
    /// A paid thought.
    Think,
    /// Waiting for the rest of a request.
    Attend,
    /// No thought.
    Abstain,
}

impl Choice {
    fn of(action: &Action) -> Option<Self> {
        match action {
            Action::Think { .. } => Some(Self::Think),
            Action::Attend { .. } => Some(Self::Attend),
            Action::Abstain { .. } => Some(Self::Abstain),
            Action::Speak { .. } => None,
        }
    }
}

/// Every candidate's choice at one speech cue, and whether the cue fell in Enton's own
/// playback or its hangover.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Asked {
    choices: Vec<Choice>,
    echo: bool,
    /// Whether the discretionary account could not pay for a thought, so no coin could
    /// be flipped.
    broke: bool,
}

/// One logged decision.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Decision {
    at: Millis,
    choice: Choice,
    /// Probability with which the logging policy took `choice`.
    propensity: f32,
    /// Whether the exploration coin was flipped here.
    flipped: bool,
    /// The turn the logged thought was timely for (served, or served again).
    credited: Option<TurnId>,
    /// Whether the logged thought was the first to serve its turn.
    served: bool,
    /// Ground truth, for measuring the estimator only: the turn a thought here would be
    /// timely for.
    timely: Option<TurnId>,
    /// Whether the decision falls in one of E1b's noise intervals.
    in_noise: bool,
    /// What the log did, for the report on where a candidate leaves the log's support:
    /// `think`, `attend` or `abstain:` and the reason, prefixed `echo ` during Enton's
    /// own playback and its hangover, or `broke ` when the discretionary account could
    /// not have paid for an explored thought.
    logged: String,
    /// Each candidate's choice in this exact state; `None` where every candidate follows
    /// the log (a tick, which reads no evidence).
    candidates: Option<Vec<Choice>>,
}

impl Decision {
    fn candidate(&self, index: usize) -> Choice {
        self.candidates
            .as_ref()
            .and_then(|choices| choices.get(index).copied())
            .unwrap_or(self.choice)
    }

    /// The candidate would think here and the log, which did not, never could have.
    fn unsupported(&self, index: usize) -> bool {
        self.candidate(index) == Choice::Think && self.choice != Choice::Think && !self.flipped
    }
}

/// Collects the decisions of a logging run.
pub(crate) struct DecisionLog<'a> {
    tape: &'a Tape,
    candidates: &'a [Profile],
    decisions: Vec<Decision>,
}

impl<'a> DecisionLog<'a> {
    pub(crate) fn new(tape: &'a Tape, candidates: &'a [Profile]) -> Self {
        Self {
            tape,
            candidates,
            decisions: Vec::new(),
        }
    }

    /// Every candidate's choice on a speech cue, asked of the state it found. Other
    /// events read no evidence, so every candidate would do what the log does.
    pub(crate) fn ask(&self, organism: &Organism, event: &Event) -> Result<Option<Asked>, Error> {
        if !matches!(event, Event::Speech { .. }) {
            return Ok(None);
        }
        let echo = organism.is_speaking()
            || matches!(
                organism.playback_status(),
                PlaybackStatus::Hangover { until, .. } if event.now() < until
            );
        let choices = self
            .candidates
            .iter()
            .map(|profile| {
                let actions = organism
                    .counterfactual(profile, event)
                    .map_err(|error| Error::Invalid(error.to_string()))?;
                actions.iter().find_map(Choice::of).ok_or_else(|| {
                    Error::Invariant("a counterfactual speech cue yields a decision".into())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let broke = !organism
            .discretionary_budget()
            .can_spend(organism.profile().budgets.think_cost);
        Ok(Some(Asked {
            choices,
            echo,
            broke,
        }))
    }

    /// Log the decisions the organism took on `record`; `paid` are the thoughts they
    /// bought, in order.
    pub(crate) fn record(
        &mut self,
        record: &Record,
        actions: &[Action],
        asked: Option<&Asked>,
        paid: &[PaidThought],
    ) -> Result<(), Error> {
        let at = record.event.now();
        let speech = matches!(record.event, Event::Speech { .. });
        let truth = record.annotation.turn().filter(|id| {
            self.tape
                .turns()
                .iter()
                .any(|turn| turn.id == *id && turn.available_at <= at && at <= turn.deadline)
        });
        let in_noise = self
            .tape
            .noise_intervals()
            .iter()
            .any(|span| span.contains(at));
        let echo = match asked {
            Some(asked) if asked.echo => "echo ",
            Some(asked) if asked.broke => "broke ",
            _ => "",
        };
        let mut paid = paid.iter();
        for action in actions {
            let Some(choice) = Choice::of(action) else {
                continue;
            };
            if self.decisions.len() >= MAX_DECISIONS {
                return Err(Error::Limit("logged decisions"));
            }
            let thought = (choice == Choice::Think).then(|| paid.next()).flatten();
            let credited = thought.and_then(|thought| match thought.credit {
                Credit::Served(id) | Credit::Duplicate(id) => Some(id),
                Credit::Waste => None,
            });
            let (flipped, logged) = match action {
                Action::Think { propensity, .. } => (propensity.is_some(), "think".to_owned()),
                Action::Abstain {
                    why, propensity, ..
                } => (propensity.is_some(), format!("abstain:{why:?}")),
                _ => (false, "attend".to_owned()),
            };
            self.decisions.push(Decision {
                at,
                choice,
                propensity: action.propensity(),
                flipped,
                credited,
                served: thought.is_some_and(|thought| matches!(thought.credit, Credit::Served(_))),
                timely: if speech { truth } else { credited },
                in_noise,
                logged: format!("{echo}{logged}"),
                candidates: asked.map(|asked| asked.choices.clone()),
            });
        }
        Ok(())
    }

    pub(crate) fn finish(self) -> Vec<Decision> {
        self.decisions
    }
}

/// Exploration settings of the logging policy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Exploration {
    /// Probability that a borderline cue thinks anyway.
    pub probability: f32,
    /// How far past a threshold, in nats, a cue may be and still be borderline.
    pub margin_nats: f32,
}

impl Exploration {
    /// The settings chosen on calibration seeds 100 to 131 by a rule fixed beforehand:
    /// over probabilities 0.05, 0.1, 0.2 and 0.3 and margins 0.5 and 1.0 nats, the
    /// smallest mean absolute IPS error (E1a turns and requests, every candidate that
    /// loosens by at most 0.5 nats) among settings costing at most 2% more paid calls and
    /// no call in E1b's noise.
    pub const CALIBRATED: Self = Self {
        probability: 0.1,
        margin_nats: 1.0,
    };

    /// The logging profile: `base` exploring with these settings, its generator seeded
    /// with `seed`.
    #[must_use]
    pub fn logging_profile(self, base: &Profile, seed: u64) -> Profile {
        let mut profile = base.clone();
        profile.exploration.explore_probability = self.probability;
        profile.exploration.explore_margin_nats = self.margin_nats;
        profile.exploration.explore_seed = seed;
        profile
    }
}

/// A candidate profile: the base profile with its evidence thresholds moved and no
/// exploration of its own.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Candidate {
    /// Short description, such as `other_voice 1.50`.
    pub label: String,
    /// Evidence against the owner speaking live that turns a cue away, in nats.
    pub other_voice_llr: f32,
    /// Evidence for a loudspeaker that turns a cue away, in nats.
    pub media_llr: f32,
    /// Evidence for speech addressed to someone else that turns a cue away, in nats.
    pub undirected_llr: f32,
    /// Evidence for the owner speaking live that admits a cue to the longer window.
    pub verified_voice_llr: f32,
}

impl Candidate {
    fn of(base: &Profile, label: String) -> Self {
        Self {
            label,
            other_voice_llr: base.attention.other_voice_llr,
            media_llr: base.source.media_llr,
            undirected_llr: base.attention.undirected_llr,
            verified_voice_llr: base.attention.verified_voice_llr,
        }
    }

    /// The candidate as a full profile.
    #[must_use]
    pub fn profile(&self, base: &Profile) -> Profile {
        let mut profile = base.clone();
        profile.attention.other_voice_llr = self.other_voice_llr;
        profile.source.media_llr = self.media_llr;
        profile.attention.undirected_llr = self.undirected_llr;
        profile.attention.verified_voice_llr = self.verified_voice_llr;
        profile.exploration.explore_probability = 0.0;
        profile
    }

    /// How far, in nats, the candidate lets through more than `base` on its most loosened
    /// threshold (zero when it loosens nothing). Past the exploration margin, the log
    /// holds no outcome for some of the cues the candidate would think on.
    #[must_use]
    pub fn loosening(&self, base: &Profile) -> f32 {
        [
            self.other_voice_llr - base.attention.other_voice_llr,
            self.media_llr - base.source.media_llr,
            self.undirected_llr - base.attention.undirected_llr,
            base.attention.verified_voice_llr - self.verified_voice_llr,
        ]
        .into_iter()
        .fold(0.0, f32::max)
    }
}

/// The family evaluated from one log: the base profile, then each threshold moved on its
/// own (tightened by 0.5 and 0.25 nats, loosened by 0.25, 0.5 and 1.0), then all of them
/// loosened and all of them tightened by 0.5. The verified threshold must stay above zero,
/// so it loosens only to 0.01, and tightens by 0.25, 0.5 and 1.0. Directedness is varied
/// only when its detector runs.
#[must_use]
pub fn candidates(base: &Profile, sensors: Sensors) -> Vec<Candidate> {
    let steps = [-0.5_f32, -0.25, 0.25, 0.5, 1.0];
    let attention = base.attention;
    let mut family = vec![Candidate::of(base, "base".into())];
    let mut vary = |name: &str, from: f32, set: fn(&mut Candidate, f32)| {
        for step in steps {
            let mut candidate = Candidate::of(base, format!("{name} {:.2}", from + step));
            set(&mut candidate, from + step);
            family.push(candidate);
        }
    };
    vary("other_voice", attention.other_voice_llr, |c, v| {
        c.other_voice_llr = v;
    });
    vary("media", base.source.media_llr, |c, v| c.media_llr = v);
    if sensors.directed {
        vary("undirected", attention.undirected_llr, |c, v| {
            c.undirected_llr = v;
        });
    }
    for verified in [0.01_f32, 0.35, 0.6, 1.1] {
        let mut candidate = Candidate::of(base, format!("verified {verified:.2}"));
        candidate.verified_voice_llr = verified;
        family.push(candidate);
    }
    let mut loose = Candidate::of(base, "all loosened 0.5".into());
    loose.other_voice_llr += 0.5;
    loose.media_llr += 0.5;
    loose.verified_voice_llr = 0.01;
    let mut tight = Candidate::of(base, "all tightened 0.5".into());
    tight.other_voice_llr -= 0.5;
    tight.media_llr -= 0.5;
    tight.verified_voice_llr += 0.5;
    if sensors.directed {
        loose.undirected_llr += 0.5;
        tight.undirected_llr -= 0.5;
    }
    family.push(loose);
    family.push(tight);
    family
}

/// An episode of E1a: its turns and the span in which its decisions fall, from the start
/// of its first segment to its last turn's deadline.
struct Episode {
    turns: BTreeSet<TurnId>,
    start: Millis,
    end: Millis,
}

fn episodes(tape: &Tape) -> Vec<Episode> {
    let starts: BTreeMap<_, _> = tape
        .records()
        .iter()
        .filter_map(|record| match (&record.annotation, &record.event) {
            (Annotation::Speech { segment, .. }, Event::Speech { now, cue }) => Some((
                *segment,
                Millis(now.0.saturating_sub(u64::from(cue.duration_ms))),
            )),
            _ => None,
        })
        .collect();
    let mut by_episode: BTreeMap<_, Episode> = BTreeMap::new();
    for turn in tape.turns() {
        let start = turn
            .segments
            .iter()
            .filter_map(|segment| starts.get(segment).copied())
            .min()
            .unwrap_or(turn.available_at);
        let episode = by_episode.entry(turn.episode).or_insert(Episode {
            turns: BTreeSet::new(),
            start,
            end: turn.deadline,
        });
        episode.turns.insert(turn.id);
        episode.start = episode.start.min(start);
        episode.end = episode.end.max(turn.deadline);
    }
    by_episode.into_values().collect()
}

/// Running sums for one candidate.
#[derive(Debug, Clone, Copy, Default)]
struct Sums {
    turns_in_context: f64,
    turns_ips: f64,
    turns_ips_variance: f64,
    /// Outcomes where the coin was not flipped: known with probability one.
    turns_fixed: f64,
    flipped: f64,
    flipped_weight: f64,
    flipped_weighted: f64,
    requests_in_context: f64,
    requests_ips: f64,
    requests_ips_variance: f64,
    requests_weight: f64,
    requests_supported: f64,
    requests_total: f64,
    calls_direct: f64,
    noise_direct: f64,
}

impl Sums {
    fn add(&mut self, other: &Self) {
        self.turns_in_context += other.turns_in_context;
        self.turns_ips += other.turns_ips;
        self.turns_ips_variance += other.turns_ips_variance;
        self.turns_fixed += other.turns_fixed;
        self.flipped += other.flipped;
        self.flipped_weight += other.flipped_weight;
        self.flipped_weighted += other.flipped_weighted;
        self.requests_in_context += other.requests_in_context;
        self.requests_ips += other.requests_ips;
        self.requests_ips_variance += other.requests_ips_variance;
        self.requests_weight += other.requests_weight;
        self.requests_supported += other.requests_supported;
        self.requests_total += other.requests_total;
        self.calls_direct += other.calls_direct;
        self.noise_direct += other.noise_direct;
    }

    /// Only the paid calls: what E1b, whose commands are not E1a turns, contributes.
    fn calls_only(self) -> Self {
        Self {
            calls_direct: self.calls_direct,
            noise_direct: self.noise_direct,
            ..Self::default()
        }
    }

    /// Turns served, self-normalized: the flipped decisions' weighted outcomes rescaled
    /// so that their weights sum to their count.
    fn turns_self_normalized(&self) -> f64 {
        if self.flipped_weight > 0.0 {
            self.turns_fixed + self.flipped * self.flipped_weighted / self.flipped_weight
        } else {
            self.turns_fixed
        }
    }

    /// Requests served, self-normalized over the observed supported episodes and scaled
    /// to all of them: unsupported episodes are assumed to fare like supported ones.
    fn requests_self_normalized(&self) -> f64 {
        if self.requests_weight > 0.0 {
            self.requests_total * self.requests_ips / self.requests_weight
        } else {
            0.0
        }
    }
}

/// Estimate, from one tape's log, what candidate `index` would have achieved.
fn estimate(decisions: &[Decision], episodes: &[Episode], index: usize) -> Sums {
    let mut sums = Sums::default();
    let mut timely = BTreeSet::new();
    for decision in decisions {
        let candidate = decision.candidate(index);
        if candidate == Choice::Think {
            sums.calls_direct += 1.0;
            sums.noise_direct += f64::from(u8::from(decision.in_noise));
            if let Some(turn) = decision.timely {
                timely.insert(turn);
            }
        }
        let reward = f64::from(u8::from(decision.served));
        let agrees = candidate == decision.choice;
        if decision.flipped {
            let weight = if agrees {
                1.0 / f64::from(decision.propensity)
            } else {
                0.0
            };
            sums.flipped += 1.0;
            sums.flipped_weight += weight;
            sums.flipped_weighted += weight * reward;
            sums.turns_ips += weight * reward;
            // An unbiased estimate of one term's variance: (w r)^2 (1 - p).
            sums.turns_ips_variance +=
                (weight * reward).powi(2) * (1.0 - f64::from(decision.propensity));
        } else if agrees {
            sums.turns_fixed += reward;
            sums.turns_ips += reward;
        }
    }
    sums.turns_in_context = timely.len() as f64;
    for episode in episodes {
        sums.requests_total += 1.0;
        if episode.turns.is_subset(&timely) {
            sums.requests_in_context += 1.0;
        }
        let mut weight = 1.0;
        let mut observed = true;
        let mut supported = true;
        let mut served = BTreeSet::new();
        let span = decisions
            .iter()
            .filter(|decision| episode.start <= decision.at && decision.at <= episode.end);
        for decision in span {
            if decision.candidate(index) != Choice::Think {
                continue;
            }
            if decision.choice == Choice::Think {
                weight /= f64::from(decision.propensity);
                if let Some(turn) = decision.credited {
                    served.insert(turn);
                }
            } else if decision.flipped {
                observed = false;
            } else {
                supported = false;
            }
        }
        if supported {
            sums.requests_supported += 1.0;
            if observed {
                let outcome = f64::from(u8::from(episode.turns.is_subset(&served)));
                sums.requests_weight += weight;
                sums.requests_ips += weight * outcome;
                sums.requests_ips_variance += (weight * outcome).powi(2) * (1.0 - 1.0 / weight);
            }
        }
    }
    sums
}

/// What a candidate achieved when run for real.
#[derive(Debug, Clone, Copy, Default)]
struct Actual {
    requests: f64,
    turns: f64,
    calls: f64,
    noise: f64,
}

/// Per-seed errors of an estimate against the actual value.
#[derive(Debug, Clone, Default)]
struct Errors(Vec<f64>);

impl Errors {
    /// 95% interval of the pooled error (the sum over seeds), normal approximation over
    /// seeds.
    fn interval(&self) -> (f64, f64) {
        let n = self.0.len() as f64;
        let total: f64 = self.0.iter().sum();
        if self.0.len() < 2 {
            return (total, total);
        }
        let mean = total / n;
        let variance = self.0.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let half = Z95 * (variance * n).sqrt();
        (total - half, total + half)
    }
}

/// Everything gathered for one candidate over the seeds.
#[derive(Debug, Clone, Default)]
struct Tracker {
    sums: Sums,
    actual: Actual,
    requests: Errors,
    turns: Errors,
    calls: Errors,
    noise: Errors,
    unsupported: BTreeMap<String, u64>,
}

impl Tracker {
    fn close_seed(&mut self, seed: &Sums, actual: &Actual) {
        self.sums.add(seed);
        self.actual.requests += actual.requests;
        self.actual.turns += actual.turns;
        self.actual.calls += actual.calls;
        self.actual.noise += actual.noise;
        self.requests.0.push(seed.requests_ips - actual.requests);
        self.turns.0.push(seed.turns_ips - actual.turns);
        self.calls.0.push(seed.calls_direct - actual.calls);
        self.noise.0.push(seed.noise_direct - actual.noise);
    }

    fn result(self, candidate: Candidate, within_margin: bool) -> CandidateResult {
        let sums = self.sums;
        let estimate =
            |actual, in_context, ips, variance: f64, self_normalized, errors: &Errors| {
                let half = Z95 * variance.sqrt();
                let (error_low, error_high) = errors.interval();
                Estimate {
                    actual,
                    in_context,
                    ips,
                    ips_low: ips - half,
                    ips_high: ips + half,
                    self_normalized,
                    error_low,
                    error_high,
                }
            };
        let count = |actual, direct, errors: &Errors| {
            let (error_low, error_high) = errors.interval();
            Count {
                actual,
                direct,
                error_low,
                error_high,
            }
        };
        CandidateResult {
            within_margin,
            requests: estimate(
                self.actual.requests,
                sums.requests_in_context,
                sums.requests_ips,
                sums.requests_ips_variance,
                sums.requests_self_normalized(),
                &self.requests,
            ),
            unsupported_requests: sums.requests_total - sums.requests_supported,
            turns: estimate(
                self.actual.turns,
                sums.turns_in_context,
                sums.turns_ips,
                sums.turns_ips_variance,
                sums.turns_self_normalized(),
                &self.turns,
            ),
            paid_calls: count(self.actual.calls, sums.calls_direct, &self.calls),
            calls_in_noise: count(self.actual.noise, sums.noise_direct, &self.noise),
            unsupported_decisions: self.unsupported,
            candidate,
        }
    }
}

/// One estimated quantity for one candidate, pooled over the seeds.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Estimate {
    /// The candidate run for real.
    pub actual: f64,
    /// The per-decision model with every outcome known: the estimators' target.
    pub in_context: f64,
    /// Inverse propensity scoring, from the log alone.
    pub ips: f64,
    /// Lower end of the IPS 95% interval, from the log alone (exploration noise only).
    pub ips_low: f64,
    /// Upper end of the IPS 95% interval.
    pub ips_high: f64,
    /// Self-normalized importance sampling, from the log alone.
    pub self_normalized: f64,
    /// Lower end of the 95% interval, across seeds, of the IPS error against `actual`.
    pub error_low: f64,
    /// Upper end of that interval.
    pub error_high: f64,
}

/// A count taken directly from the candidates' choices.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Count {
    /// The candidate run for real.
    pub actual: f64,
    /// The candidate's choices at the logged states, counted.
    pub direct: f64,
    /// Lower end of the 95% interval, across seeds, of the error against `actual`.
    pub error_low: f64,
    /// Upper end of that interval.
    pub error_high: f64,
}

/// Every estimate for one candidate.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CandidateResult {
    /// The candidate.
    pub candidate: Candidate,
    /// Whether every threshold the candidate loosens stays within the exploration margin.
    pub within_margin: bool,
    /// E1a requests served.
    pub requests: Estimate,
    /// E1a episodes in which the candidate would think where the log never could.
    pub unsupported_requests: f64,
    /// E1a turns served.
    pub turns: Estimate,
    /// Paid calls over E1a and E1b.
    pub paid_calls: Count,
    /// Calls during E1b's noise.
    pub calls_in_noise: Count,
    /// Decisions (E1a and E1b) at which the candidate would think and the log never could,
    /// by what the log did there.
    pub unsupported_decisions: BTreeMap<String, u64>,
}

/// What exploring cost the logging run, against the same profile without exploration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct ExplorationCost {
    /// Paid calls without exploration (E1a and E1b).
    pub base_paid_calls: u64,
    /// Paid calls of the logging run.
    pub logging_paid_calls: u64,
    /// Calls in E1b's noise without exploration.
    pub base_calls_in_noise: u64,
    /// Calls in E1b's noise while exploring.
    pub logging_calls_in_noise: u64,
    /// E1a turns served without exploration.
    pub base_served_turns: u64,
    /// E1a turns served while exploring.
    pub logging_served_turns: u64,
    /// E1a requests served without exploration.
    pub base_served_requests: u64,
    /// E1a requests served while exploring.
    pub logging_served_requests: u64,
    /// Coin flips.
    pub flips: u64,
    /// Explored thoughts: outcomes of abstentions that would otherwise never be seen.
    pub explored: u64,
    /// Explored thoughts that served a turn: abstentions shown to be misses.
    pub explored_served: u64,
}

/// The off-policy evaluation report.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OffPolicyReport {
    /// Benchmark protocol version.
    pub version: String,
    /// Seeds pooled.
    pub seeds: Vec<u64>,
    /// Sensor readings that reached the organism.
    pub sensors: Sensors,
    /// Exploration settings of the logging policy.
    pub exploration: Exploration,
    /// What exploring cost.
    pub cost: ExplorationCost,
    /// Every candidate, the base profile first.
    pub candidates: Vec<CandidateResult>,
}

/// Run the base profile as an exploring logging policy on each seed's E1a and E1b tapes,
/// estimate every candidate in [`candidates`] from that log, and run each of them for real
/// to measure the estimates.
///
/// # Errors
///
/// Returns an error if a tape is invalid, a limit is exceeded or a candidate profile is
/// invalid.
pub fn evaluate(
    seeds: &[u64],
    base: &Profile,
    exploration: Exploration,
    sensors: Sensors,
) -> Result<OffPolicyReport, Error> {
    let family = candidates(base, sensors);
    let profiles: Vec<Profile> = family.iter().map(|c| c.profile(base)).collect();
    let mut trackers = vec![Tracker::default(); family.len()];
    let mut cost = ExplorationCost::default();
    for &seed in seeds {
        let mut seed_sums = vec![Sums::default(); family.len()];
        let mut seed_actual = vec![Actual::default(); family.len()];
        for (index, tape) in [e1a(seed)?, e1b(seed)?].iter().enumerate() {
            let serving = index == 0;
            let generator = seed.wrapping_mul(2).wrapping_add(u64::from(!serving));
            let logging = exploration.logging_profile(base, generator);
            let (logged, decisions) = run_logged(tape, &logging, sensors, &profiles)?;
            let episodes = if serving { episodes(tape) } else { Vec::new() };
            cost.logging_paid_calls += logged.paid_calls;
            cost.logging_calls_in_noise += logged.calls_in_noise;
            if serving {
                cost.logging_served_turns += logged.served_turns;
                cost.logging_served_requests += logged.served_requests;
            }
            for decision in &decisions {
                let explored = decision.flipped && decision.choice == Choice::Think;
                cost.flips += u64::from(decision.flipped);
                cost.explored += u64::from(explored);
                cost.explored_served += u64::from(explored && decision.served);
            }
            let slots = profiles
                .iter()
                .zip(trackers.iter_mut())
                .zip(seed_sums.iter_mut().zip(seed_actual.iter_mut()));
            for (candidate, ((profile, tracker), (sums, actual))) in slots.enumerate() {
                let tally = estimate(&decisions, &episodes, candidate);
                sums.add(&if serving { tally } else { tally.calls_only() });
                for decision in decisions.iter().filter(|d| d.unsupported(candidate)) {
                    *tracker
                        .unsupported
                        .entry(decision.logged.clone())
                        .or_default() += 1;
                }
                let real = run_organism(tape, profile, sensors)?;
                actual.calls += real.paid_calls as f64;
                actual.noise += real.calls_in_noise as f64;
                if serving {
                    actual.requests += real.served_requests as f64;
                    actual.turns += real.served_turns as f64;
                }
                if candidate == 0 {
                    cost.base_paid_calls += real.paid_calls;
                    cost.base_calls_in_noise += real.calls_in_noise;
                    if serving {
                        cost.base_served_turns += real.served_turns;
                        cost.base_served_requests += real.served_requests;
                    }
                }
            }
        }
        for (tracker, (sums, actual)) in trackers.iter_mut().zip(seed_sums.iter().zip(&seed_actual))
        {
            tracker.close_seed(sums, actual);
        }
    }
    let candidates = family
        .into_iter()
        .zip(trackers)
        .map(|(candidate, tracker)| {
            let within_margin = candidate.loosening(base) <= exploration.margin_nats;
            tracker.result(candidate, within_margin)
        })
        .collect();
    Ok(OffPolicyReport {
        version: BENCHMARK_VERSION.into(),
        seeds: seeds.to_vec(),
        sensors,
        exploration,
        cost,
        candidates,
    })
}

impl OffPolicyReport {
    fn fmt_cost(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cost = &self.cost;
        writeln!(
            f,
            "Logging policy: t1-ref exploring borderline cues with probability {:.2} within {:.2} nats",
            self.exploration.probability, self.exploration.margin_nats
        )?;
        writeln!(
            f,
            "Exploration cost: {} paid calls vs {} without ({:+}), {} calls in E1b noise vs {} ({:+})",
            cost.logging_paid_calls,
            cost.base_paid_calls,
            signed(cost.logging_paid_calls, cost.base_paid_calls),
            cost.logging_calls_in_noise,
            cost.base_calls_in_noise,
            signed(cost.logging_calls_in_noise, cost.base_calls_in_noise),
        )?;
        writeln!(
            f,
            "  served while exploring: {} turns vs {} ({:+}), {} requests vs {} ({:+})",
            cost.logging_served_turns,
            cost.base_served_turns,
            signed(cost.logging_served_turns, cost.base_served_turns),
            cost.logging_served_requests,
            cost.base_served_requests,
            signed(cost.logging_served_requests, cost.base_served_requests),
        )?;
        writeln!(
            f,
            "Information: {} coin flips, {} explored thoughts, {} of them served a turn the log would have missed",
            cost.flips, cost.explored, cost.explored_served
        )
    }

    fn fmt_estimates(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Columns: actual (run for real) | in context (every outcome known, one decision at a time) | IPS [95% from the log] | self-normalized | IPS error [95% over seeds]"
        )?;
        for (title, requests) in [("E1a turns served", false), ("E1a requests served", true)] {
            writeln!(f, "{title}:")?;
            for result in &self.candidates {
                let e = if requests {
                    &result.requests
                } else {
                    &result.turns
                };
                write!(
                    f,
                    "  {:<20}{} actual {:>5.0} | in context {:>5.0} | IPS {:>6.1} [{:.0}, {:.0}] | SN {:>6.1} | error {:+.1} [{:+.0}, {:+.0}]",
                    result.candidate.label,
                    margin_mark(result),
                    e.actual,
                    e.in_context,
                    e.ips,
                    e.ips_low,
                    e.ips_high,
                    e.self_normalized,
                    e.ips - e.actual,
                    e.error_low,
                    e.error_high,
                )?;
                if requests && result.unsupported_requests > 0.0 {
                    write!(f, " | unsupported {:.0}", result.unsupported_requests)?;
                }
                writeln!(f)?;
            }
        }
        Ok(())
    }

    /// What choosing a candidate over the base would change, actual against estimated:
    /// the difference a threshold decision rests on.
    fn fmt_effects(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(base) = self.candidates.first() else {
            return Ok(());
        };
        writeln!(
            f,
            "Effect against the base: actual | IPS | self-normalized (requests, turns), actual | counted (calls):"
        )?;
        for result in self.candidates.iter().skip(1) {
            let delta = |pick: fn(&CandidateResult) -> &Estimate| {
                let (e, b) = (pick(result), pick(base));
                (
                    e.actual - b.actual,
                    e.ips - b.ips,
                    e.self_normalized - b.self_normalized,
                )
            };
            let requests = delta(|r| &r.requests);
            let turns = delta(|r| &r.turns);
            writeln!(
                f,
                "  {:<20}{} requests {:+5.0} | {:+5.0} | {:+5.0}   turns {:+5.0} | {:+5.0} | {:+5.0}   calls {:+5.0} | {:+5.0}",
                result.candidate.label,
                margin_mark(result),
                requests.0,
                requests.1,
                requests.2,
                turns.0,
                turns.1,
                turns.2,
                result.paid_calls.actual - base.paid_calls.actual,
                result.paid_calls.direct - base.paid_calls.direct,
            )?;
        }
        Ok(())
    }

    fn fmt_calls(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Paid calls (E1a + E1b) and calls in E1b noise: actual | counted at the logged states | error [95% over seeds]:"
        )?;
        for result in &self.candidates {
            let (calls, noise) = (&result.paid_calls, &result.calls_in_noise);
            writeln!(
                f,
                "  {:<20}{} calls {:>5.0} | {:>5.0} | {:+.0} [{:+.0}, {:+.0}]   noise {:>3.0} | {:>3.0}",
                result.candidate.label,
                margin_mark(result),
                calls.actual,
                calls.direct,
                calls.direct - calls.actual,
                calls.error_low,
                calls.error_high,
                noise.actual,
                noise.direct,
            )?;
        }
        Ok(())
    }

    fn fmt_support(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Unsupported decisions (the candidate thinks where the log never could), by what the log did:"
        )?;
        for result in &self.candidates {
            if result.unsupported_decisions.is_empty() {
                continue;
            }
            let cells: Vec<_> = result
                .unsupported_decisions
                .iter()
                .map(|(logged, count)| format!("{logged} {count}"))
                .collect();
            writeln!(
                f,
                "  {:<20}{} {}",
                result.candidate.label,
                margin_mark(result),
                cells.join(", ")
            )?;
        }
        writeln!(
            f,
            "* loosens a threshold past the exploration margin: the log holds no outcome for some cues it would think on"
        )
    }
}

impl fmt::Display for OffPolicyReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "E1 off-policy evaluation | benchmark {} | {} seeds: {:?}",
            self.version,
            self.seeds.len(),
            self.seeds
        )?;
        writeln!(f, "Sensors: {}", self.sensors)?;
        self.fmt_cost(f)?;
        self.fmt_estimates(f)?;
        self.fmt_calls(f)?;
        self.fmt_effects(f)?;
        self.fmt_support(f)
    }
}

fn margin_mark(result: &CandidateResult) -> char {
    if result.within_margin { ' ' } else { '*' }
}

/// `a - b` as a signed count.
fn signed(a: u64, b: u64) -> i128 {
    i128::from(a) - i128::from(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run::run_tape_with;

    fn decision(choice: Choice, propensity: f32, flipped: bool, turn: Option<u32>) -> Decision {
        let turn = turn.map(TurnId);
        let think = choice == Choice::Think;
        Decision {
            at: Millis(0),
            choice,
            propensity,
            flipped,
            credited: turn.filter(|_| think),
            served: think && turn.is_some(),
            timely: turn,
            in_noise: false,
            logged: String::new(),
            candidates: None,
        }
    }

    /// A deterministic thought that served turn 1, an explored one (probability 0.25)
    /// that served turn 2, and a kept abstention (0.75) on turn 3's last segment, one
    /// second apart. Candidate 0 does what the log's base would (never thinks at a coin
    /// flip); candidate 1 always thinks.
    fn log() -> Vec<Decision> {
        let mut decisions = vec![
            decision(Choice::Think, 1.0, false, Some(1)),
            decision(Choice::Think, 0.25, true, Some(2)),
            decision(Choice::Abstain, 0.75, true, Some(3)),
        ];
        for (second, decision) in (0..).zip(decisions.iter_mut()) {
            decision.at = Millis(second * 1_000);
            decision.candidates = Some(vec![
                if decision.flipped {
                    Choice::Abstain
                } else {
                    decision.choice
                },
                Choice::Think,
            ]);
        }
        decisions
    }

    fn episode(turn: u32, at: u64) -> Episode {
        Episode {
            turns: BTreeSet::from([TurnId(turn)]),
            start: Millis(at),
            end: Millis(at),
        }
    }

    #[test]
    fn inverse_propensity_weights_what_the_log_saw_and_knows_what_it_did_not() {
        let decisions = log();
        let base = estimate(&decisions, &[], 0);
        assert!((base.turns_ips - 1.0).abs() < 1e-12);
        assert!((base.turns_in_context - 1.0).abs() < 1e-12);
        assert!((base.calls_direct - 1.0).abs() < 1e-12);
        assert!(base.turns_ips_variance.abs() < 1e-12);

        let eager = estimate(&decisions, &[], 1);
        // The explored thought stands for 1 / 0.25 of its kind; the kept abstention's turn
        // is invisible to the log.
        assert!((eager.turns_ips - 5.0).abs() < 1e-12);
        assert!((eager.turns_in_context - 3.0).abs() < 1e-12);
        assert!((eager.calls_direct - 3.0).abs() < 1e-12);
        // (4 x 1)^2 (1 - 0.25).
        assert!((eager.turns_ips_variance - 12.0).abs() < 1e-12);
        // Self-normalized: 1 fixed + 2 flips x (4 / 4).
        assert!((eager.turns_self_normalized() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn a_request_is_weighted_by_every_thought_it_needed() {
        let decisions = log();
        let episodes = [episode(1, 0), episode(2, 1_000), episode(3, 2_000)];
        let eager = estimate(&decisions, &episodes, 1);
        assert!((eager.requests_total - 3.0).abs() < 1e-12);
        assert!((eager.requests_supported - 3.0).abs() < 1e-12);
        assert!((eager.requests_in_context - 3.0).abs() < 1e-12);
        // Seen: episode 1 (weight 1) and episode 2 (weight 4); episode 3 was a coin flip
        // that came up abstain.
        assert!((eager.requests_ips - 5.0).abs() < 1e-12);
        assert!((eager.requests_self_normalized() - 3.0).abs() < 1e-12);

        // A thought where the log abstained for sure leaves the episode unsupported.
        let mut decisions = log();
        if let Some(last) = decisions.last_mut() {
            last.flipped = false;
            last.propensity = 1.0;
        }
        let eager = estimate(&decisions, &episodes, 1);
        assert!((eager.requests_supported - 2.0).abs() < 1e-12);
        assert!(decisions.last().is_some_and(|last| last.unsupported(1)));
        assert!(!decisions.iter().any(|decision| decision.unsupported(0)));
    }

    #[test]
    fn the_error_interval_brackets_the_pooled_error() {
        let errors = Errors(vec![1.0, -1.0, 3.0, 1.0]);
        let (low, high) = errors.interval();
        assert!(low < 4.0 && 4.0 < high);
        assert_eq!(Errors(vec![2.0]).interval(), (2.0, 2.0));
    }

    #[test]
    fn the_family_starts_with_the_base_and_every_candidate_is_valid() {
        let base = Profile::t1_ref();
        let family = candidates(&base, Sensors::DEFAULT);
        let first = family.first().unwrap();
        assert_eq!(first.label, "base");
        assert_eq!(first.profile(&base), base);
        assert!(first.loosening(&base).abs() < f32::EPSILON);
        for candidate in &family {
            let profile = candidate.profile(&base);
            assert!(profile.validate().is_ok(), "{}", candidate.label);
            assert_eq!(profile.exploration.explore_probability.to_bits(), 0);
            assert!(!candidate.label.starts_with("undirected"));
        }
        let loosest = family
            .iter()
            .map(|c| c.loosening(&base))
            .fold(0.0, f32::max);
        assert!((loosest - 1.0).abs() < 1e-6);
        let directed = candidates(&base, Sensors::WITH_DIRECTED);
        assert_eq!(directed.len(), family.len() + 5);
    }

    #[test]
    fn without_exploration_the_base_is_estimated_exactly() {
        let base = Profile::t1_ref();
        let silent = Exploration {
            probability: 0.0,
            margin_nats: 1.0,
        };
        let report = evaluate(&[7], &base, silent, Sensors::DEFAULT).unwrap();
        assert_eq!(report.cost.flips, 0);
        assert_eq!(report.cost.logging_paid_calls, report.cost.base_paid_calls);
        let first = report.candidates.first().unwrap();
        let (a, b) = (
            run_tape_with(&e1a(7).unwrap(), &base, Sensors::DEFAULT).unwrap(),
            run_tape_with(&e1b(7).unwrap(), &base, Sensors::DEFAULT).unwrap(),
        );
        let exact = |estimate: f64, value: u64| (estimate - value as f64).abs() < 1e-9;
        assert!(exact(first.turns.actual, a.organism.served_turns));
        assert!(exact(first.turns.ips, a.organism.served_turns));
        assert!(exact(first.requests.ips, a.organism.served_requests));
        assert!(exact(first.requests.actual, a.organism.served_requests));
        let calls = a.organism.paid_calls + b.organism.paid_calls;
        assert!(exact(first.paid_calls.direct, calls));
        assert!(exact(first.paid_calls.actual, calls));
        assert!(first.unsupported_decisions.is_empty());
    }

    #[test]
    fn an_exploring_log_flips_coins_and_reports_the_same_way_twice() {
        let base = Profile::t1_ref();
        let run = || evaluate(&[100], &base, Exploration::CALIBRATED, Sensors::DEFAULT).unwrap();
        let report = run();
        assert!(report.cost.flips > 0);
        assert!(report.cost.explored <= report.cost.flips);
        assert_eq!(report.version, BENCHMARK_VERSION);
        assert_eq!(
            report.candidates.len(),
            candidates(&base, Sensors::DEFAULT).len()
        );
        assert_eq!(report, run());
        let text = report.to_string();
        assert!(text.contains("E1a turns served:"));
        assert!(text.contains("Unsupported decisions"));
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["exploration"]["probability"], 0.1_f32);
    }
}
