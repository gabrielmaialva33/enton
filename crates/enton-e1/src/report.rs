//! Original RFC criteria first; synthetic proxies cannot establish a full E1 PASS.
use crate::{BENCHMARK_VERSION, ControllerResult, Error, ExperimentRun, TapeKind};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fmt};

/// Cognitive reason for a thought ignited during a noise interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NoiseReason {
    /// Keyword detected (or attend timeout following a keyword).
    Keyword,
    /// Follow-up speech within an active attention window.
    FollowUp,
    /// Salient unaddressed speech.
    Speech,
    /// Internal drive pressure.
    Drive,
}

impl fmt::Display for NoiseReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Keyword => "keyword",
            Self::FollowUp => "follow-up",
            Self::Speech => "speech",
            Self::Drive => "drive",
        })
    }
}

impl From<&enton_core::Reason> for NoiseReason {
    fn from(reason: &enton_core::Reason) -> Self {
        match reason {
            enton_core::Reason::Keyword => Self::Keyword,
            enton_core::Reason::FollowUp => Self::FollowUp,
            enton_core::Reason::Speech => Self::Speech,
            enton_core::Reason::Drive(_) => Self::Drive,
        }
    }
}

/// Triggering stimulus for a thought ignited during a noise interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NoiseStimulus {
    /// Television broadcast audio.
    Tv,
    /// Speech from an unaddressed person.
    OtherPerson,
    /// Motor noise.
    Motor,
    /// Ventilation noise.
    Ventilation,
    /// Self-playback acoustic echo.
    SelfEcho,
    /// Internal drive or clock event with no speech stimulus.
    NoneInternal,
}

impl fmt::Display for NoiseStimulus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Tv => "TV",
            Self::OtherPerson => "other person",
            Self::Motor => "motor",
            Self::Ventilation => "ventilation",
            Self::SelfEcho => "self-echo",
            Self::NoneInternal => "none/internal",
        })
    }
}

/// Descriptive breakdown of noise-interval calls by reason and triggering stimulus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoiseBreakdown {
    /// Breakdown of noise-interval calls by ignition reason.
    pub by_reason: BTreeMap<NoiseReason, u64>,
    /// Breakdown of noise-interval calls by triggering stimulus.
    pub by_stimulus: BTreeMap<NoiseStimulus, u64>,
}

impl Default for NoiseBreakdown {
    fn default() -> Self {
        Self::new()
    }
}

impl NoiseBreakdown {
    /// Construct a new empty breakdown with all keys initialized to zero.
    #[must_use]
    pub fn new() -> Self {
        let mut by_reason = BTreeMap::new();
        by_reason.insert(NoiseReason::Keyword, 0);
        by_reason.insert(NoiseReason::FollowUp, 0);
        by_reason.insert(NoiseReason::Speech, 0);
        by_reason.insert(NoiseReason::Drive, 0);
        let mut by_stimulus = BTreeMap::new();
        by_stimulus.insert(NoiseStimulus::Tv, 0);
        by_stimulus.insert(NoiseStimulus::OtherPerson, 0);
        by_stimulus.insert(NoiseStimulus::Motor, 0);
        by_stimulus.insert(NoiseStimulus::Ventilation, 0);
        by_stimulus.insert(NoiseStimulus::SelfEcho, 0);
        by_stimulus.insert(NoiseStimulus::NoneInternal, 0);
        Self {
            by_reason,
            by_stimulus,
        }
    }

    /// Record one call with its reason and triggering stimulus.
    pub fn record(&mut self, reason: NoiseReason, stimulus: NoiseStimulus) {
        *self.by_reason.entry(reason).or_insert(0) += 1;
        *self.by_stimulus.entry(stimulus).or_insert(0) += 1;
    }

    /// Total calls recorded in the breakdown.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.by_reason.values().sum()
    }

    /// Get count for a specific reason.
    #[must_use]
    pub fn reason_count(&self, reason: NoiseReason) -> u64 {
        self.by_reason.get(&reason).copied().unwrap_or(0)
    }

    /// Get count for a specific stimulus.
    #[must_use]
    pub fn stimulus_count(&self, stimulus: NoiseStimulus) -> u64 {
        self.by_stimulus.get(&stimulus).copied().unwrap_or(0)
    }
}

/// A criterion is explicitly unmeasured rather than implicitly successful.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Status {
    /// This measured criterion meets its threshold.
    Pass,
    /// This measured criterion fails its threshold.
    Fail,
    /// Evidence required by the RFC has not been measured.
    NotEvaluated,
}
impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::NotEvaluated => "not evaluated",
        })
    }
}
/// One original RFC criterion, with provenance and its status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Criterion {
    /// Original threshold, not a replacement waste metric.
    pub name: &'static str,
    /// Pass/fail of a synthetic proxy, or explicitly unmeasured.
    pub status: Status,
    /// Values and scope of the evidence.
    pub details: String,
}
impl Criterion {
    fn measured(name: &'static str, passed: bool, details: String) -> Self {
        Self {
            name,
            status: if passed { Status::Pass } else { Status::Fail },
            details,
        }
    }
    fn unmeasured(name: &'static str) -> Self {
        Self {
            name,
            status: Status::NotEvaluated,
            details: "physical measurement absent; simulation is not a substitute".into(),
        }
    }
}

/// A report with the unchanged RFC thresholds and all three secondary audits.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    /// Distribution/scoring version, always printed.
    pub version: &'static str,
    /// Original criteria in RFC order.
    pub criteria: Vec<Criterion>,
    /// Never Pass while any required criterion is unmeasured.
    pub overall: Status,
    /// E1a metrics and raw causal audit.
    pub e1a: ExperimentRun,
    /// E1b metrics and raw causal audit.
    pub e1b: ExperimentRun,
}
impl Report {
    /// Validate population/controller roles and build the original-criterion report.
    pub fn new(e1a: ExperimentRun, e1b: ExperimentRun) -> Result<Self, Error> {
        if e1a.kind != TapeKind::E1a
            || e1b.kind != TapeKind::E1b
            || e1a.version != BENCHMARK_VERSION
            || e1b.version != BENCHMARK_VERSION
            || e1a.seed != e1b.seed
            || e1a.economy != e1b.economy
        {
            return Err(Error::Invalid(
                "report requires matching-version/seed/economy E1a and E1b runs".into(),
            ));
        }
        for (run, total) in [(&e1a, 100), (&e1b, 10)] {
            for (result, expected) in [
                (&run.organism, crate::Controller::Organism),
                (&run.simple, crate::Controller::Simple),
                (&run.fixed_window, crate::Controller::FixedWindow),
            ] {
                if result.controller != expected
                    || result.total_requests != total
                    || result.served_requests > total
                    || result.total_turns == 0
                    || result.served_turns > result.total_turns
                    || result.paid_calls != result.thoughts.len() as u64
                    || result.served_turns + result.wasted_calls != result.paid_calls
                    || result.noise_breakdown.total() != result.calls_in_noise
                {
                    return Err(Error::Invalid(
                        "invalid report population, role or causal accounting".into(),
                    ));
                }
            }
        }
        let org = e1a.organism.paid_calls + e1b.organism.paid_calls;
        let base = e1a.simple.paid_calls + e1b.simple.paid_calls;
        let reduction = if base == 0 {
            "undefined (baseline has zero calls)".into()
        } else {
            format!("{:.2}%", 100.0 * (1.0 - org as f64 / base as f64))
        };
        let criteria = vec![
            Criterion::measured(
                "RFC §7: >=50% fewer total calls than simple",
                base > 0 && org <= base / 2,
                format!(
                    "synthetic paid-call proxy: organism {org}, simple {base}, reduction {reduction}"
                ),
            ),
            Criterion::measured(
                "RFC §7: >=99/100 relevant requests in E1a",
                e1a.organism.served_requests >= 99,
                format!(
                    "synthetic complete-episode ignition proxy: {}/100; all turns required",
                    e1a.organism.served_requests
                ),
            ),
            Criterion::measured(
                "RFC §7: 10/10 commands in E1b",
                e1b.organism.served_requests == 10,
                format!(
                    "synthetic paid-ignition proxy: {}/10",
                    e1b.organism.served_requests
                ),
            ),
            Criterion::unmeasured("RFC §7: zero physical self-ignitions with barge-in preserved"),
            Criterion::measured(
                "RFC §7: zero calls in E1b's 50 minutes of noise",
                e1b.organism.calls_in_noise == 0,
                format!(
                    "synthetic interval proxy: {} calls",
                    e1b.organism.calls_in_noise
                ),
            ),
            Criterion::unmeasured("RFC §7: added p95 latency <=100 ms"),
            Criterion::unmeasured("RFC §7: stable core RSS over 24 hours"),
        ];
        let overall = overall_status(&criteria);
        Ok(Self {
            version: BENCHMARK_VERSION,
            criteria,
            overall,
            e1a,
            e1b,
        })
    }
}
fn overall_status(criteria: &[Criterion]) -> Status {
    if criteria.iter().any(|c| c.status == Status::Fail) {
        Status::Fail
    } else if criteria.is_empty() || criteria.iter().any(|c| c.status == Status::NotEvaluated) {
        Status::NotEvaluated
    } else {
        Status::Pass
    }
}
fn print_controller(f: &mut fmt::Formatter<'_>, r: &ControllerResult) -> fmt::Result {
    writeln!(
        f,
        "  {}: paid={}, cost={:.3}, requests={}/{}, turns={}/{}, waste={}, duplicates={}",
        r.controller.name(),
        r.paid_calls,
        r.total_cost,
        r.served_requests,
        r.total_requests,
        r.served_turns,
        r.total_turns,
        r.wasted_calls,
        r.duplicate_calls
    )?;
    writeln!(
        f,
        "    rejected obligations={}, discretionary exhaustion={}, refill={:.6}, balance={:.6}, rounding credit={:.9}",
        r.rejected_obligations,
        r.discretionary_exhaustion,
        r.credited_refill,
        r.final_balance,
        r.rounding_credit
    )?;
    writeln!(
        f,
        "    noise-interval calls={}, synthetic self-ignitions={}, actual playback overlap={}/{} barge-in segments, feedback={}, feedback beyond horizon={}",
        r.calls_in_noise,
        r.synthetic_self_ignitions,
        r.overlapping_barge_in_segments,
        r.barge_in_segments,
        r.feedback_events,
        r.feedback_after_horizon
    )?;
    if r.calls_in_noise > 0 {
        let reasons: Vec<_> = r
            .noise_breakdown
            .by_reason
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        writeln!(f, "    noise by reason: {}", reasons.join(", "))?;
        let stimuli: Vec<_> = r
            .noise_breakdown
            .by_stimulus
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        writeln!(f, "    noise by stimulus: {}", stimuli.join(", "))?;
    }
    let mut keys: Vec<_> = r.strata.keys().collect();
    keys.sort();
    for k in keys {
        if let Some(tally) = r.strata.get(k) {
            let crate::scoring::Tally { served, total } = tally;
            match k {
                crate::scoring::Stratum::Kind(kind) => {
                    writeln!(f, "    stratum [type: {kind}]: {served}/{total}")?;
                }
                crate::scoring::Stratum::Gap(gap) => {
                    writeln!(f, "    stratum [gap: {gap} ms]: {served}/{total}")?;
                }
            }
        }
    }
    Ok(())
}
impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "E1 benchmark {} | seed {} | synthetic cue experiment",
            self.version, self.e1a.seed
        )?;
        writeln!(f, "Original RFC §7 criteria (unchanged):")?;
        for criterion in &self.criteria {
            writeln!(
                f,
                "  [{}] {} ({})",
                criterion.status, criterion.name, criterion.details
            )?;
        }
        writeln!(
            f,
            "Overall: {}. Physical E1 validation remains incomplete.",
            self.overall
        )?;
        writeln!(
            f,
            "Common account per controller/tape: initial/capacity {:.3}, refill {:.3}/hour, price {:.3}/call",
            self.e1a.economy.capacity,
            self.e1a.economy.refill_per_hour,
            self.e1a.economy.thought_cost
        )?;
        for (name, run) in [("E1a", &self.e1a), ("E1b", &self.e1b)] {
            writeln!(
                f,
                "Secondary {name} diagnostics (not replacement criteria):"
            )?;
            for result in [&run.organism, &run.simple, &run.fixed_window] {
                print_controller(f, result)?;
            }
        }
        let comparator_qualified = self.e1a.fixed_window.served_requests >= 99
            && self.e1b.fixed_window.served_requests == 10;
        writeln!(
            f,
            "Fixed-window comparator reaches the request service floor: {comparator_qualified}. Its cost is context, never an organism criterion."
        )?;
        writeln!(
            f,
            "Cost units are abstract; no token, joule, audio-quality or wall-clock latency measurements."
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn passing_proxies_cannot_hide_missing_physical_criteria() {
        let criteria = vec![
            Criterion::measured("calls", true, "proxy".into()),
            Criterion::unmeasured("RSS"),
        ];
        assert_eq!(overall_status(&criteria), Status::NotEvaluated);
        assert_eq!(overall_status(&[]), Status::NotEvaluated);
    }
    #[test]
    fn original_failure_is_never_replaced_by_a_secondary_success() {
        assert_eq!(
            overall_status(&[
                Criterion::measured("original calls", false, "only 17%".into()),
                Criterion::unmeasured("RSS")
            ]),
            Status::Fail
        );
    }
    #[test]
    fn noise_breakdown_initializes_zero_and_formats_cleanly() {
        let mut nb = NoiseBreakdown::new();
        assert_eq!(nb.total(), 0);
        assert_eq!(nb.reason_count(NoiseReason::Keyword), 0);
        assert_eq!(nb.stimulus_count(NoiseStimulus::Tv), 0);

        nb.record(NoiseReason::Speech, NoiseStimulus::Tv);
        nb.record(NoiseReason::FollowUp, NoiseStimulus::OtherPerson);
        assert_eq!(nb.total(), 2);
        assert_eq!(nb.reason_count(NoiseReason::Speech), 1);
        assert_eq!(nb.reason_count(NoiseReason::FollowUp), 1);
        assert_eq!(nb.reason_count(NoiseReason::Drive), 0);
        assert_eq!(nb.stimulus_count(NoiseStimulus::Tv), 1);
        assert_eq!(nb.stimulus_count(NoiseStimulus::OtherPerson), 1);
        assert_eq!(nb.stimulus_count(NoiseStimulus::Motor), 0);

        assert_eq!(NoiseReason::Keyword.to_string(), "keyword");
        assert_eq!(NoiseReason::FollowUp.to_string(), "follow-up");
        assert_eq!(NoiseReason::Speech.to_string(), "speech");
        assert_eq!(NoiseReason::Drive.to_string(), "drive");

        assert_eq!(NoiseStimulus::Tv.to_string(), "TV");
        assert_eq!(NoiseStimulus::OtherPerson.to_string(), "other person");
        assert_eq!(NoiseStimulus::Motor.to_string(), "motor");
        assert_eq!(NoiseStimulus::Ventilation.to_string(), "ventilation");
        assert_eq!(NoiseStimulus::SelfEcho.to_string(), "self-echo");
        assert_eq!(NoiseStimulus::NoneInternal.to_string(), "none/internal");
    }
}
