//! Original RFC criteria first; synthetic proxies cannot establish a full E1 PASS.
use crate::tape::{ConditionKey, Distance, TvBackground};
use crate::{BENCHMARK_VERSION, ControllerResult, Error, ExperimentRun, Sensors, TapeKind};
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

/// Cognitive reason for a paid thought recorded as waste or duplicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum WasteReason {
    /// Keyword detected (or attend timeout following a keyword).
    Keyword,
    /// Follow-up speech within an active attention window.
    FollowUp,
    /// Salient unaddressed speech.
    Speech,
    /// Internal drive pressure.
    Drive,
}

impl fmt::Display for WasteReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Keyword => "keyword",
            Self::FollowUp => "follow-up",
            Self::Speech => "speech",
            Self::Drive => "drive",
        })
    }
}

impl From<&enton_core::Reason> for WasteReason {
    fn from(r: &enton_core::Reason) -> Self {
        match r {
            enton_core::Reason::Keyword => Self::Keyword,
            enton_core::Reason::FollowUp => Self::FollowUp,
            enton_core::Reason::Speech => Self::Speech,
            enton_core::Reason::Drive(_) => Self::Drive,
        }
    }
}

/// Triggering stimulus for a paid thought recorded as waste or duplicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum WasteStimulus {
    /// Request turn segment.
    Request,
    /// Barge-in request turn segment.
    BargeIn,
    /// Television broadcast audio.
    Tv,
    /// Speech from an unaddressed person.
    OtherPerson,
    /// Caller voice addressed to someone else.
    Aside,
    /// False positive keyword detection.
    FalseKeyword,
    /// Self-playback acoustic echo.
    SelfEcho,
    /// Household, motor or ventilation noise.
    Noise,
    /// Internal drive or clock event with no speech stimulus.
    Internal,
}

impl fmt::Display for WasteStimulus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Request => "request",
            Self::BargeIn => "barge-in",
            Self::Tv => "tv",
            Self::OtherPerson => "other person",
            Self::Aside => "aside",
            Self::FalseKeyword => "false keyword",
            Self::SelfEcho => "self-echo",
            Self::Noise => "noise",
            Self::Internal => "internal",
        })
    }
}

/// Composite key indexing waste breakdown by reason and triggering stimulus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WasteKey {
    /// Ignition reason.
    pub reason: WasteReason,
    /// Triggering stimulus.
    pub stimulus: WasteStimulus,
}

impl Serialize for WasteKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_str(&format_args!("{}:{}", self.reason, self.stimulus))
    }
}

impl<'de> Deserialize<'de> for WasteKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let (reason_str, stimulus_str) = s
            .split_once(':')
            .ok_or_else(|| serde::de::Error::custom("expected reason:stimulus"))?;
        let reason = match reason_str {
            "keyword" => WasteReason::Keyword,
            "follow-up" => WasteReason::FollowUp,
            "speech" => WasteReason::Speech,
            "drive" => WasteReason::Drive,
            _ => return Err(serde::de::Error::custom("unknown reason")),
        };
        let stimulus = match stimulus_str {
            "request" => WasteStimulus::Request,
            "barge-in" => WasteStimulus::BargeIn,
            "tv" => WasteStimulus::Tv,
            "other person" => WasteStimulus::OtherPerson,
            "aside" => WasteStimulus::Aside,
            "false keyword" => WasteStimulus::FalseKeyword,
            "self-echo" => WasteStimulus::SelfEcho,
            "noise" => WasteStimulus::Noise,
            "internal" => WasteStimulus::Internal,
            _ => return Err(serde::de::Error::custom("unknown stimulus")),
        };
        Ok(Self { reason, stimulus })
    }
}

/// Descriptive breakdown of wasted calls by reason and triggering stimulus.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WasteBreakdown {
    /// Breakdown of wasted calls by ignition reason.
    pub by_reason: BTreeMap<WasteReason, u64>,
    /// Breakdown of wasted calls by triggering stimulus.
    pub by_stimulus: BTreeMap<WasteStimulus, u64>,
    /// Joint breakdown by reason and stimulus.
    pub by_cell: BTreeMap<WasteKey, u64>,
}

impl Default for WasteBreakdown {
    fn default() -> Self {
        Self::new()
    }
}

impl WasteBreakdown {
    /// Construct a new empty breakdown with all reason and stimulus keys initialized to zero.
    #[must_use]
    pub fn new() -> Self {
        let mut by_reason = BTreeMap::new();
        by_reason.insert(WasteReason::Keyword, 0);
        by_reason.insert(WasteReason::FollowUp, 0);
        by_reason.insert(WasteReason::Speech, 0);
        by_reason.insert(WasteReason::Drive, 0);

        let mut by_stimulus = BTreeMap::new();
        by_stimulus.insert(WasteStimulus::Request, 0);
        by_stimulus.insert(WasteStimulus::BargeIn, 0);
        by_stimulus.insert(WasteStimulus::Tv, 0);
        by_stimulus.insert(WasteStimulus::OtherPerson, 0);
        by_stimulus.insert(WasteStimulus::Aside, 0);
        by_stimulus.insert(WasteStimulus::FalseKeyword, 0);
        by_stimulus.insert(WasteStimulus::SelfEcho, 0);
        by_stimulus.insert(WasteStimulus::Noise, 0);
        by_stimulus.insert(WasteStimulus::Internal, 0);

        Self {
            by_reason,
            by_stimulus,
            by_cell: BTreeMap::new(),
        }
    }

    /// Record one wasted or duplicate call with its reason and triggering stimulus.
    pub fn record(&mut self, reason: WasteReason, stimulus: WasteStimulus) {
        *self.by_reason.entry(reason).or_insert(0) += 1;
        *self.by_stimulus.entry(stimulus).or_insert(0) += 1;
        *self
            .by_cell
            .entry(WasteKey { reason, stimulus })
            .or_insert(0) += 1;
    }

    /// Total wasted calls recorded in the breakdown.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.by_reason.values().sum()
    }

    /// Get count for a specific reason.
    #[must_use]
    pub fn reason_count(&self, reason: WasteReason) -> u64 {
        self.by_reason.get(&reason).copied().unwrap_or(0)
    }

    /// Get count for a specific stimulus.
    #[must_use]
    pub fn stimulus_count(&self, stimulus: WasteStimulus) -> u64 {
        self.by_stimulus.get(&stimulus).copied().unwrap_or(0)
    }

    /// Get count for a specific reason-stimulus cell.
    #[must_use]
    pub fn cell_count(&self, reason: WasteReason, stimulus: WasteStimulus) -> u64 {
        self.by_cell
            .get(&WasteKey { reason, stimulus })
            .copied()
            .unwrap_or(0)
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
    /// Sensor readings that reached the controllers, always printed.
    pub sensors: Sensors,
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
            || e1a.sensors != e1b.sensors
        {
            return Err(Error::Invalid(
                "report requires matching-version/seed/economy/sensors E1a and E1b runs".into(),
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
                    || result.waste_breakdown.total() != result.wasted_calls
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
            sensors: e1a.sensors,
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
fn print_controller(f: &mut fmt::Formatter<'_>, r: &ControllerResult, is_e1a: bool) -> fmt::Result {
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
    let non_zero_waste: Vec<_> = r
        .waste_breakdown
        .by_stimulus
        .iter()
        .filter(|(_, v)| **v > 0)
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    if !non_zero_waste.is_empty() {
        writeln!(f, "    waste by stimulus: {}", non_zero_waste.join(", "))?;
    }
    if !r.turn_segment_decisions.is_empty() {
        let decisions: Vec<_> = r
            .turn_segment_decisions
            .iter()
            .map(|(decision, count)| format!("{decision}={count}"))
            .collect();
        writeln!(f, "    turn segments: {}", decisions.join(", "))?;
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
    if is_e1a && !r.turns_by_condition.is_empty() {
        print_conditions(f, r)?;
    }
    Ok(())
}
fn print_conditions(f: &mut fmt::Formatter<'_>, r: &ControllerResult) -> fmt::Result {
    writeln!(
        f,
        "    turns served by condition (TV off | moderate | loud):"
    )?;
    for dist in [Distance::Near, Distance::Far] {
        let [off, moderate, loud] = [
            TvBackground::Off,
            TvBackground::Moderate,
            TvBackground::Loud,
        ]
        .map(|tv| {
            r.turns_by_condition
                .get(&ConditionKey::new(dist, tv))
                .copied()
                .unwrap_or_default()
        });
        writeln!(
            f,
            "      {dist:<4}: off {}/{}, moderate {}/{}, loud {}/{}",
            off.served, off.total, moderate.served, moderate.total, loud.served, loud.total,
        )?;
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
        writeln!(f, "Sensors: {}", self.sensors)?;
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
        for (name, run, is_e1a) in [("E1a", &self.e1a, true), ("E1b", &self.e1b, false)] {
            writeln!(
                f,
                "Secondary {name} diagnostics (not replacement criteria):"
            )?;
            for result in [&run.organism, &run.simple, &run.fixed_window] {
                print_controller(f, result, is_e1a)?;
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

/// One-sided 95% Clopper-Pearson exact upper bound for $k$ misses out of $n$ trials.
///
/// Computes the exact Clopper-Pearson upper bound by finding $p \in [0, 1]$
/// such that $P(X \le k \mid n, p) = 0.05$.
///
/// Bounded bisection search over the binomial CDF computed using incremental log terms.
#[must_use]
pub fn clopper_pearson_upper_bound(k: u64, n: u64) -> f64 {
    if n == 0 || k >= n {
        return 1.0;
    }
    let target = 0.05;
    let mut low = 0.0f64;
    let mut high = 1.0f64;
    let Ok(k_usize) = usize::try_from(k) else {
        return 1.0;
    };
    let mut buffer = Vec::with_capacity(k_usize.saturating_add(1));

    for _ in 0..100 {
        let mid = low.midpoint(high);
        let cdf = binomial_cdf(k, n, mid, &mut buffer);
        if cdf > target {
            low = mid;
        } else {
            high = mid;
        }
    }
    low.midpoint(high)
}

fn binomial_cdf(k: u64, n: u64, p: f64, buffer: &mut Vec<f64>) -> f64 {
    if k >= n {
        return 1.0;
    }
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }
    buffer.clear();
    let Ok(k_usize) = usize::try_from(k) else {
        return 1.0;
    };
    let ln_p = p.ln();
    let ln_1_minus_p = (1.0 - p).ln();
    let log_ratio = ln_p - ln_1_minus_p;

    let mut current_log = (n as f64) * ln_1_minus_p;
    buffer.push(current_log);
    let mut max_log = current_log;

    for i in 1..=k_usize {
        let i_f64 = i as f64;
        let n_minus_i_plus_1 = (n.saturating_sub(i as u64).saturating_add(1)) as f64;
        current_log += n_minus_i_plus_1.ln() - i_f64.ln() + log_ratio;
        buffer.push(current_log);
        if current_log > max_log {
            max_log = current_log;
        }
    }

    let sum: f64 = buffer.iter().map(|&lt| (lt - max_log).exp()).sum();
    (max_log.exp() * sum).clamp(0.0, 1.0)
}

/// Pass, fail, and not-evaluated counts for one criterion across multiple seeds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CriterionSummary {
    /// Criterion threshold name.
    pub name: &'static str,
    /// Number of seeds passing this criterion.
    pub pass: usize,
    /// Number of seeds failing this criterion.
    pub fail: usize,
    /// Number of seeds where this criterion was not evaluated.
    pub not_evaluated: usize,
}

/// Aggregate multi-seed summary for experiment E1.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Summary {
    /// Benchmark protocol version.
    pub version: &'static str,
    /// Sensor readings that reached the controllers, the same for every seed.
    pub sensors: Sensors,
    /// Evaluated seeds in order.
    pub seeds: Vec<u64>,
    /// Criteria pass/fail/not-evaluated counts across seeds.
    pub criteria: Vec<CriterionSummary>,
    /// E1a organism pooled served requests.
    pub e1a_served_requests: u64,
    /// E1a organism pooled total requests.
    pub e1a_total_requests: u64,
    /// E1a organism pooled served turns.
    pub e1a_served_turns: u64,
    /// E1a organism pooled total turns.
    pub e1a_total_turns: u64,
    /// One-sided 95% Clopper-Pearson exact upper bound on the E1a request miss rate.
    pub e1a_request_miss_rate_upper_bound: f64,
    /// E1b organism pooled served requests.
    pub e1b_served_requests: u64,
    /// E1b organism pooled total requests.
    pub e1b_total_requests: u64,
    /// E1b organism total calls in noise intervals.
    pub e1b_calls_in_noise_total: u64,
    /// E1b organism mean calls in noise intervals per seed.
    pub e1b_calls_in_noise_mean: f64,
    /// E1b organism maximum calls in noise intervals across seeds.
    pub e1b_calls_in_noise_max: u64,
    /// Pooled paid calls by organism across E1a and E1b.
    pub pooled_organism_paid_calls: u64,
    /// Pooled paid calls by simple baseline across E1a and E1b.
    pub pooled_simple_paid_calls: u64,
    /// Pooled paid call reduction percentage of organism relative to simple.
    pub pooled_reduction_percentage: f64,
    /// Total synthetic self-ignitions for E1a organism.
    pub e1a_synthetic_self_ignitions: u64,
    /// E1a organism turns served and total, pooled, by the block condition of each
    /// turn's first segment: where the TV background costs turns.
    pub e1a_turns_by_condition: BTreeMap<ConditionKey, crate::scoring::Tally>,
}

impl Summary {
    /// Build an aggregate summary over multiple reports.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Invalid`] if `reports` is empty or mixes sensor sets.
    pub fn from_reports(reports: &[Report]) -> Result<Self, Error> {
        let first = reports
            .first()
            .ok_or_else(|| Error::Invalid("empty reports for summary".into()))?;
        if reports.iter().any(|report| report.sensors != first.sensors) {
            return Err(Error::Invalid("summary mixes sensor sets".into()));
        }

        let seeds: Vec<u64> = reports.iter().map(|r| r.e1a.seed).collect();

        let mut criteria_map = BTreeMap::new();
        let mut ordered_names = Vec::new();
        for report in reports {
            for c in &report.criteria {
                let entry = criteria_map
                    .entry(c.name)
                    .or_insert((0usize, 0usize, 0usize));
                match c.status {
                    Status::Pass => entry.0 += 1,
                    Status::Fail => entry.1 += 1,
                    Status::NotEvaluated => entry.2 += 1,
                }
                if !ordered_names.contains(&c.name) {
                    ordered_names.push(c.name);
                }
            }
        }
        let criteria = ordered_names
            .into_iter()
            .map(|name| {
                let (pass, fail, not_evaluated) =
                    criteria_map.get(&name).copied().unwrap_or((0, 0, 0));
                CriterionSummary {
                    name,
                    pass,
                    fail,
                    not_evaluated,
                }
            })
            .collect();

        let mut e1a_served_requests = 0u64;
        let mut e1a_total_requests = 0u64;
        let mut e1a_served_turns = 0u64;
        let mut e1a_total_turns = 0u64;
        let mut e1a_synthetic_self_ignitions = 0u64;

        let mut e1b_served_requests = 0u64;
        let mut e1b_total_requests = 0u64;
        let mut e1b_calls_in_noise_total = 0u64;
        let mut e1b_calls_in_noise_max = 0u64;

        let mut pooled_organism_paid_calls = 0u64;
        let mut pooled_simple_paid_calls = 0u64;

        for report in reports {
            e1a_served_requests += report.e1a.organism.served_requests;
            e1a_total_requests += report.e1a.organism.total_requests;
            e1a_served_turns += report.e1a.organism.served_turns;
            e1a_total_turns += report.e1a.organism.total_turns;
            e1a_synthetic_self_ignitions += report.e1a.organism.synthetic_self_ignitions;

            e1b_served_requests += report.e1b.organism.served_requests;
            e1b_total_requests += report.e1b.organism.total_requests;
            let noise_calls = report.e1b.organism.calls_in_noise;
            e1b_calls_in_noise_total += noise_calls;
            if noise_calls > e1b_calls_in_noise_max {
                e1b_calls_in_noise_max = noise_calls;
            }

            pooled_organism_paid_calls +=
                report.e1a.organism.paid_calls + report.e1b.organism.paid_calls;
            pooled_simple_paid_calls += report.e1a.simple.paid_calls + report.e1b.simple.paid_calls;
        }

        let e1a_misses = e1a_total_requests.saturating_sub(e1a_served_requests);
        let e1a_request_miss_rate_upper_bound =
            clopper_pearson_upper_bound(e1a_misses, e1a_total_requests);

        let e1b_calls_in_noise_mean = e1b_calls_in_noise_total as f64 / reports.len() as f64;

        let pooled_reduction_percentage = if pooled_simple_paid_calls == 0 {
            0.0
        } else {
            100.0 * (1.0 - pooled_organism_paid_calls as f64 / pooled_simple_paid_calls as f64)
        };

        Ok(Self {
            version: first.version,
            sensors: first.sensors,
            seeds,
            criteria,
            e1a_served_requests,
            e1a_total_requests,
            e1a_served_turns,
            e1a_total_turns,
            e1a_request_miss_rate_upper_bound,
            e1b_served_requests,
            e1b_total_requests,
            e1b_calls_in_noise_total,
            e1b_calls_in_noise_mean,
            e1b_calls_in_noise_max,
            pooled_organism_paid_calls,
            pooled_simple_paid_calls,
            pooled_reduction_percentage,
            e1a_synthetic_self_ignitions,
            e1a_turns_by_condition: pooled_turns_by_condition(reports),
        })
    }

    /// Pooled served requests for E1a organism.
    #[must_use]
    pub fn e1a_organism_served_requests(&self) -> u64 {
        self.e1a_served_requests
    }

    /// Pooled total requests for E1a organism.
    #[must_use]
    pub fn e1a_organism_total_requests(&self) -> u64 {
        self.e1a_total_requests
    }

    /// Pooled served turns for E1a organism.
    #[must_use]
    pub fn e1a_organism_served_turns(&self) -> u64 {
        self.e1a_served_turns
    }

    /// Pooled total turns for E1a organism.
    #[must_use]
    pub fn e1a_organism_total_turns(&self) -> u64 {
        self.e1a_total_turns
    }

    /// Pooled served requests for E1b organism.
    #[must_use]
    pub fn e1b_organism_served_requests(&self) -> u64 {
        self.e1b_served_requests
    }

    /// Pooled total requests for E1b organism.
    #[must_use]
    pub fn e1b_organism_total_requests(&self) -> u64 {
        self.e1b_total_requests
    }

    /// Total synthetic self-ignitions for E1a organism.
    #[must_use]
    pub fn e1a_organism_synthetic_self_ignitions(&self) -> u64 {
        self.e1a_synthetic_self_ignitions
    }
}

/// The E1a organism's turns served and total by block condition, pooled over `reports`.
fn pooled_turns_by_condition(reports: &[Report]) -> BTreeMap<ConditionKey, crate::scoring::Tally> {
    let mut pooled: BTreeMap<ConditionKey, crate::scoring::Tally> = BTreeMap::new();
    for report in reports {
        for (key, tally) in &report.e1a.organism.turns_by_condition {
            let sum = pooled.entry(*key).or_default();
            sum.served += tally.served;
            sum.total += tally.total;
        }
    }
    pooled
}

impl Summary {
    /// Pooled E1a turns by TV background, over both distances and then per distance.
    fn print_turns_by_tv(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tvs = [
            TvBackground::Off,
            TvBackground::Moderate,
            TvBackground::Loud,
        ];
        let tally = |distances: &[Distance], tv| {
            distances
                .iter()
                .filter_map(|distance| {
                    self.e1a_turns_by_condition
                        .get(&ConditionKey::new(*distance, tv))
                })
                .fold(crate::scoring::Tally::default(), |sum, tally| {
                    crate::scoring::Tally {
                        served: sum.served + tally.served,
                        total: sum.total + tally.total,
                    }
                })
        };
        let share = |tally: crate::scoring::Tally| {
            if tally.total == 0 {
                "n/a".to_owned()
            } else {
                format!("{:.1}%", 100.0 * tally.served as f64 / tally.total as f64)
            }
        };
        let [off, moderate, loud] = tvs.map(|tv| tally(&[Distance::Near, Distance::Far], tv));
        writeln!(
            f,
            "  turns by TV: off {}/{} ({}), moderate {}/{} ({}), loud {}/{} ({})",
            off.served,
            off.total,
            share(off),
            moderate.served,
            moderate.total,
            share(moderate),
            loud.served,
            loud.total,
            share(loud),
        )?;
        for distance in [Distance::Near, Distance::Far] {
            let [off, moderate, loud] = tvs.map(|tv| tally(&[distance], tv));
            writeln!(
                f,
                "    {:<4}: off {}/{}, moderate {}/{}, loud {}/{}",
                distance.to_string().to_lowercase(),
                off.served,
                off.total,
                moderate.served,
                moderate.total,
                loud.served,
                loud.total,
            )?;
        }
        Ok(())
    }
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "E1 benchmark summary | {} seeds: {:?}",
            self.seeds.len(),
            self.seeds
        )?;
        writeln!(f, "Benchmark {} | sensors: {}", self.version, self.sensors)?;
        writeln!(
            f,
            "Original RFC §7 criteria over {} seeds:",
            self.seeds.len()
        )?;
        for c in &self.criteria {
            writeln!(
                f,
                "  {}: {} pass, {} fail, {} not evaluated",
                c.name, c.pass, c.fail, c.not_evaluated
            )?;
        }
        writeln!(f, "E1a organism:")?;
        writeln!(
            f,
            "  requests: {}/{} pooled, turns: {}/{} pooled",
            self.e1a_served_requests,
            self.e1a_total_requests,
            self.e1a_served_turns,
            self.e1a_total_turns
        )?;
        writeln!(
            f,
            "  request miss rate: 95% Clopper-Pearson upper bound = {:.5} ({:.2}%)",
            self.e1a_request_miss_rate_upper_bound,
            self.e1a_request_miss_rate_upper_bound * 100.0
        )?;
        writeln!(
            f,
            "  synthetic self-ignitions: {} total",
            self.e1a_synthetic_self_ignitions
        )?;
        self.print_turns_by_tv(f)?;
        writeln!(f, "E1b organism:")?;
        writeln!(
            f,
            "  requests: {}/{} pooled",
            self.e1b_served_requests, self.e1b_total_requests
        )?;
        writeln!(
            f,
            "  calls in noise: {} total, {:.2} mean/seed, {} max",
            self.e1b_calls_in_noise_total,
            self.e1b_calls_in_noise_mean,
            self.e1b_calls_in_noise_max
        )?;
        writeln!(f, "Paid calls (E1a + E1b pooled):")?;
        if self.pooled_simple_paid_calls == 0 {
            writeln!(
                f,
                "  organism: {}, simple: {}, reduction: undefined (baseline has zero calls)",
                self.pooled_organism_paid_calls, self.pooled_simple_paid_calls
            )?;
        } else {
            writeln!(
                f,
                "  organism: {}, simple: {}, reduction: {:.2}%",
                self.pooled_organism_paid_calls,
                self.pooled_simple_paid_calls,
                self.pooled_reduction_percentage
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clopper_pearson_known_values() {
        let tol = 1e-4;
        let v0_100 = clopper_pearson_upper_bound(0, 100);
        assert!(
            (v0_100 - 0.02951).abs() < tol,
            "k=0, n=100 expected ~0.02951, got {v0_100}"
        );

        let v1_100 = clopper_pearson_upper_bound(1, 100);
        assert!(
            (v1_100 - 0.04656).abs() < tol,
            "k=1, n=100 expected ~0.04656, got {v1_100}"
        );

        let v0_3200 = clopper_pearson_upper_bound(0, 3200);
        assert!(
            (v0_3200 - 0.000_936).abs() < tol,
            "k=0, n=3200 expected ~0.000936, got {v0_3200}"
        );
    }

    #[test]
    fn summary_empty_reports_returns_invalid_error() {
        let res = Summary::from_reports(&[]);
        assert!(matches!(res, Err(Error::Invalid(_))));
    }

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
