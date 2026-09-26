//! Verification metrics over genuine and impostor scores: EER, d', FAR and FRR.

/// Fewest recordings per class (owner and others) a metric report accepts.
pub const MIN_FILES_PER_CLASS: usize = 20;

/// Inverse standard normal CDF (Acklam's algorithm).
#[must_use]
pub fn standard_normal_inv_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e+01,
        2.209_460_984_245_205e+02,
        -2.759_285_104_469_687e+02,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e+01,
        2.506_628_277_459_239e+00,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e+01,
        1.615_858_368_580_409e+02,
        -1.556_989_798_598_866e+02,
        6.680_131_188_771_972e+01,
        -1.328_068_155_288_572e+01,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-03,
        -3.223_964_580_411_365e-01,
        -2.400_758_277_161_838e+00,
        -2.549_732_539_343_734e+00,
        4.374_664_141_464_968e+00,
        2.938_163_982_698_783e+00,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-03,
        3.224_671_290_700_398e-01,
        2.445_134_137_142_996e+00,
        3.754_408_661_907_416e+00,
    ];

    let p_clamped = p.clamp(1e-15, 1.0 - 1e-15);
    let p_low = 0.02425_f64;
    let p_high = 1.0 - p_low;

    if p_clamped < p_low {
        let q = (-2.0 * p_clamped.ln()).sqrt();
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        num / den
    } else if p_clamped <= p_high {
        let q = p_clamped - 0.5;
        let r = q * q;
        let num = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q;
        let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
        num / den
    } else {
        let q = (-2.0 * (1.0 - p_clamped).ln()).sqrt();
        let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
        let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
        -num / den
    }
}

/// Speaker verification metrics over genuine (target) and impostor (non-target) scores.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricsReport {
    /// Genuine recordings scored.
    pub target_count: usize,
    /// Impostor recordings scored.
    pub nontarget_count: usize,
    /// Mean genuine cosine score.
    pub target_mean: f32,
    /// Standard deviation of genuine scores.
    pub target_std: f32,
    /// Lowest genuine score.
    pub target_min: f32,
    /// Highest genuine score.
    pub target_max: f32,
    /// Mean impostor cosine score.
    pub nontarget_mean: f32,
    /// Standard deviation of impostor scores.
    pub nontarget_std: f32,
    /// Lowest impostor score.
    pub nontarget_min: f32,
    /// Highest impostor score.
    pub nontarget_max: f32,
    /// d' from the score means and the pooled deviation.
    pub empirical_dprime: f32,
    /// Equal error rate, from zero to one.
    pub eer: f32,
    /// Cosine threshold at the equal error rate.
    pub eer_threshold: f32,
    /// d' implied by the EER under a Gaussian model.
    pub gaussian_dprime: f32,
    /// False rejection rate where false acceptance is 2%.
    pub frr_at_far_2_pct: f32,
    /// Cosine threshold at that operating point.
    pub frr_at_far_2_pct_thresh: f32,
    /// False rejection rate where false acceptance is 6.7%.
    pub frr_at_far_6_7_pct: f32,
    /// Cosine threshold at that operating point.
    pub frr_at_far_6_7_pct_thresh: f32,
    /// False acceptance rate where false rejection is 1%.
    pub far_at_frr_1_pct: f32,
    /// Cosine threshold at that operating point.
    pub far_at_frr_1_pct_thresh: f32,
    /// Audio scored, in seconds.
    pub total_audio_duration_s: f64,
    /// Time spent computing embeddings, in seconds.
    pub total_embedding_duration_s: f64,
    /// Embedding time per second of audio, in milliseconds.
    pub mean_embedding_time_per_sec_ms: f64,
    /// Real-time factor: compute time over audio time.
    pub rtf: f64,
}

impl MetricsReport {
    /// Render the report as the plain-text table the probe prints.
    #[must_use]
    pub fn format_report(&self) -> String {
        let pass = self.gaussian_dprime >= 3.83 && self.eer <= 0.0305;
        let (status_line, interp_line) = if pass {
            (
                    format!(
                        "  Result: PASSED (EER {:.2}% <= 3.0%, d' {:.2} >= 3.83)",
                        self.eer * 100.0,
                        self.gaussian_dprime
                    ),
                    "  Interpretation: Voice verification alone is viable for authenticated attention on this mic.".to_string(),
                )
        } else {
            (
                    format!(
                        "  Result: DEFICIT (EER {:.2}% > 3.0% or d' {:.2} < 3.83)",
                        self.eer * 100.0,
                        self.gaussian_dprime
                    ),
                    "  Interpretation: Voice alone is insufficient. Organism v3 requires multi-modal fusion (temporal prior / DoA) to reach d' >= 3.83.".to_string(),
                )
        };

        let lines = [
            "================================================================================",
            "Enton Owner Voice Probe: Speaker Verification Report (Task 0013)",
            "================================================================================",
            "Counts:",
            &format!("  Target (genuine owner):      {} files", self.target_count),
            &format!(
                "  Non-target (distractors):    {} files",
                self.nontarget_count
            ),
            "",
            "Timing & Throughput:",
            &format!(
                "  Total audio duration:        {:.2} s",
                self.total_audio_duration_s
            ),
            &format!(
                "  Total embedding compute:     {:.2} s",
                self.total_embedding_duration_s
            ),
            &format!(
                "  Mean compute time / sec:     {:.2} ms / s audio  (RTF = {:.4})",
                self.mean_embedding_time_per_sec_ms, self.rtf
            ),
            "",
            "Cosine Score Statistics:",
            &format!(
                "  Target:      mean = {:.4},  std = {:.4}  (min = {:.4}, max = {:.4})",
                self.target_mean, self.target_std, self.target_min, self.target_max
            ),
            &format!(
                "  Non-target:  mean = {:.4},  std = {:.4}  (min = {:.4}, max = {:.4})",
                self.nontarget_mean, self.nontarget_std, self.nontarget_min, self.nontarget_max
            ),
            "",
            "Discriminability & Error Rates:",
            &format!(
                "  EER (Equal Error Rate):      {:.2}%  (cosine threshold = {:.4})",
                self.eer * 100.0,
                self.eer_threshold
            ),
            &format!("  Gaussian d' (from EER):      {:.2}", self.gaussian_dprime),
            &format!(
                "  Empirical d' (from mean/std): {:.2}",
                self.empirical_dprime
            ),
            "",
            "Operational Risk Points:",
            &format!(
                "  FRR at FAR = 2.0%:           {:.2}%  (operational threshold = {:.4})",
                self.frr_at_far_2_pct * 100.0,
                self.frr_at_far_2_pct_thresh
            ),
            &format!(
                "  FRR at FAR = 6.7%:           {:.2}%  (operational threshold = {:.4})",
                self.frr_at_far_6_7_pct * 100.0,
                self.frr_at_far_6_7_pct_thresh
            ),
            &format!(
                "  FAR at FRR = 1.0%:           {:.2}%  (operational threshold = {:.4})",
                self.far_at_frr_1_pct * 100.0,
                self.far_at_frr_1_pct_thresh
            ),
            "",
            "Organism v3 Feasibility Gate:",
            "  Target criterion: combined d' >= 3.83 (voice alone: EER <= 3.0%)",
            &status_line,
            &interp_line,
            "================================================================================",
        ];

        let mut out = lines.join("\n");
        out.push('\n');
        out
    }
}

fn interpolate_roc(roc: &[(f32, f32, f32)], target: f32, at_far: bool) -> (f32, f32) {
    if roc.is_empty() {
        return (0.0, 0.0);
    }
    let best_thresh = roc
        .iter()
        .find(|p| if at_far { p.1 <= target } else { p.2 >= target })
        .map_or(0.0, |p| p.0);

    for i in 0..roc.len().saturating_sub(1) {
        let Some(&(_, far1, frr1)) = roc.get(i) else {
            break;
        };
        let Some(&(_, far2, frr2)) = roc.get(i + 1) else {
            break;
        };
        let (v1, v2, o1, o2) = if at_far {
            (far1, far2, frr1, frr2)
        } else {
            (frr1, frr2, far1, far2)
        };
        if (v1 >= target && v2 <= target) || (v1 <= target && v2 >= target) {
            let d = v2 - v1;
            if d.abs() > 1e-7 {
                let alpha = (target - v1) / d;
                return ((o1 + alpha * (o2 - o1)).clamp(0.0, 1.0), best_thresh);
            }
            return (o1, best_thresh);
        }
    }
    let fallback = if at_far {
        if target >= 1.0 { 0.0 } else { 1.0 }
    } else if target <= 0.0 {
        1.0
    } else {
        0.0
    };
    (fallback, best_thresh)
}

// Scores are f32; f64 only accumulates the sums, so narrowing the results back to
// f32 cannot lose precision the inputs ever had.
#[allow(clippy::cast_possible_truncation)]
fn calc_stats(scores: &[f32]) -> (f32, f32, f32, f32) {
    let n = scores.len();
    let sum: f64 = scores.iter().map(|&x| f64::from(x)).sum();
    let mean = (sum / n as f64) as f32;
    let var: f64 = scores
        .iter()
        .map(|&x| {
            let d = f64::from(x) - f64::from(mean);
            d * d
        })
        .sum::<f64>()
        / (n - 1) as f64;
    let std = var.sqrt() as f32;
    let min = scores.iter().copied().fold(f32::INFINITY, f32::min);
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (mean, std, min, max)
}

/// ROC points `(threshold, FAR, FRR)` over every observed score, sorted by threshold.
fn roc_curve(target_scores: &[f32], nontarget_scores: &[f32]) -> Vec<(f32, f32, f32)> {
    let (t_len, nt_len) = (target_scores.len(), nontarget_scores.len());
    let mut thresholds: Vec<f32> = Vec::with_capacity(t_len + nt_len + 2);
    thresholds.push(-1.01);
    for &s in target_scores {
        thresholds.push(s);
    }
    for &s in nontarget_scores {
        thresholds.push(s);
    }
    thresholds.push(1.01);
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    thresholds.dedup();

    let mut roc: Vec<(f32, f32, f32)> = Vec::with_capacity(thresholds.len());
    for &thresh in &thresholds {
        let far_count = nontarget_scores.iter().filter(|&&s| s >= thresh).count();
        let frr_count = target_scores.iter().filter(|&&s| s < thresh).count();
        let far = far_count as f32 / nt_len as f32;
        let frr = frr_count as f32 / t_len as f32;
        roc.push((thresh, far, frr));
    }
    roc
}

/// Equal error rate and its cosine threshold, interpolated on the ROC curve.
fn equal_error_rate(roc: &[(f32, f32, f32)], t_min: f32, nt_max: f32) -> (f32, f32) {
    let mut eer = 0.0_f32;
    let mut eer_threshold = 0.0_f32;
    if t_min > nt_max {
        eer = 0.0;
        eer_threshold = f32::midpoint(nt_max, t_min);
    } else {
        for i in 0..roc.len().saturating_sub(1) {
            let Some(&(th1, far1, frr1)) = roc.get(i) else {
                break;
            };
            let Some(&(th2, far2, frr2)) = roc.get(i + 1) else {
                break;
            };
            let diff1 = far1 - frr1;
            let diff2 = far2 - frr2;
            if diff1 >= 0.0 && diff2 <= 0.0 {
                let denom = (far2 - far1) - (frr2 - frr1);
                if denom.abs() > 1e-7 {
                    let num = far2 * frr1 - far1 * frr2;
                    let cand_eer = num / denom;
                    eer = cand_eer.clamp(0.0, 1.0);
                    let span = far2 - far1;
                    if span.abs() > 1e-7 {
                        let alpha = (eer - far1) / span;
                        eer_threshold = th1 + alpha * (th2 - th1);
                    } else {
                        eer_threshold = f32::midpoint(th1, th2);
                    }
                } else {
                    eer = far1;
                    eer_threshold = th1;
                }
                break;
            }
        }
    }
    (eer, eer_threshold)
}

/// Calculate verification metrics: EER, empirical and Gaussian d', FAR, FRR.
pub fn calculate_verification_metrics(
    target_scores: &[f32],
    nontarget_scores: &[f32],
    total_audio_s: f64,
    total_compute_s: f64,
) -> Result<MetricsReport, String> {
    let t_len = target_scores.len();
    let nt_len = nontarget_scores.len();

    if t_len < MIN_FILES_PER_CLASS || nt_len < MIN_FILES_PER_CLASS {
        return Err(format!(
            "Refusing to report: fewer than {MIN_FILES_PER_CLASS} files per class. \
                 Target count: {t_len}, Nontarget count: {nt_len}. Minimum required is {MIN_FILES_PER_CLASS} per class."
        ));
    }

    let (t_mean, t_std, t_min, t_max) = calc_stats(target_scores);
    let (nt_mean, nt_std, nt_min, nt_max) = calc_stats(nontarget_scores);

    let pooled_std = f32::midpoint(t_std * t_std, nt_std * nt_std).sqrt();
    let empirical_dprime = if pooled_std > 1e-6 {
        (t_mean - nt_mean) / pooled_std
    } else {
        0.0
    };

    let roc = roc_curve(target_scores, nontarget_scores);

    let (eer, eer_threshold) = equal_error_rate(&roc, t_min, nt_max);

    // The inverse CDF works in f64; d' is reported with the f32 precision of the scores.
    #[allow(clippy::cast_possible_truncation)]
    let gaussian_dprime = (2.0 * standard_normal_inv_cdf(1.0 - f64::from(eer))) as f32;

    let (frr_at_far_2_pct, frr_at_far_2_pct_thresh) = interpolate_roc(&roc, 0.02, true);
    let (frr_at_far_6_7_pct, frr_at_far_6_7_pct_thresh) = interpolate_roc(&roc, 0.067, true);
    let (far_at_frr_1_pct, far_at_frr_1_pct_thresh) = interpolate_roc(&roc, 0.01, false);

    let mean_embedding_time_per_sec_ms = if total_audio_s > 0.0 {
        (total_compute_s / total_audio_s) * 1000.0
    } else {
        0.0
    };
    let rtf = if total_audio_s > 0.0 {
        total_compute_s / total_audio_s
    } else {
        0.0
    };

    Ok(MetricsReport {
        target_count: t_len,
        nontarget_count: nt_len,
        target_mean: t_mean,
        target_std: t_std,
        target_min: t_min,
        target_max: t_max,
        nontarget_mean: nt_mean,
        nontarget_std: nt_std,
        nontarget_min: nt_min,
        nontarget_max: nt_max,
        empirical_dprime,
        eer,
        eer_threshold,
        gaussian_dprime,
        frr_at_far_2_pct,
        frr_at_far_2_pct_thresh,
        frr_at_far_6_7_pct,
        frr_at_far_6_7_pct_thresh,
        far_at_frr_1_pct,
        far_at_frr_1_pct_thresh,
        total_audio_duration_s: total_audio_s,
        total_embedding_duration_s: total_compute_s,
        mean_embedding_time_per_sec_ms,
        rtf,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_normal_inv_cdf_known_values() {
        let z_half = standard_normal_inv_cdf(0.5);
        assert!(z_half.abs() < 1e-12);

        let z_8871 = standard_normal_inv_cdf(0.8871);
        let dprime_8871 = 2.0 * z_8871;
        assert!((dprime_8871 - 2.422).abs() < 0.005);

        let dprime_05 = 2.0 * standard_normal_inv_cdf(0.95);
        assert!((dprime_05 - 3.29).abs() < 0.01);

        let dprime_03 = 2.0 * standard_normal_inv_cdf(0.97);
        assert!((dprime_03 - 3.76).abs() < 0.01);

        let dprime_01 = 2.0 * standard_normal_inv_cdf(0.99);
        assert!((dprime_01 - 4.65).abs() < 0.01);
    }

    #[test]
    fn eer_and_dprime_synthetic_perfect_separation() {
        let targets = vec![0.85_f32; 25];
        let nontargets = vec![0.15_f32; 25];

        let report = calculate_verification_metrics(&targets, &nontargets, 50.0, 1.0)
            .expect("should calculate report");

        assert_eq!(report.target_count, 25);
        assert_eq!(report.nontarget_count, 25);
        assert!(report.eer.abs() < f32::EPSILON);
        assert!(report.frr_at_far_2_pct.abs() < f32::EPSILON);
        assert!(report.frr_at_far_6_7_pct.abs() < f32::EPSILON);
        assert!(report.far_at_frr_1_pct.abs() < f32::EPSILON);
        assert!(report.gaussian_dprime > 5.0);
    }

    #[test]
    fn eer_synthetic_known_overlap() {
        let mut targets = vec![0.60_f32; 20];
        targets.extend(vec![0.80_f32; 20]);

        let mut nontargets = vec![0.40_f32; 20];
        nontargets.extend(vec![0.60_f32; 20]);

        let report = calculate_verification_metrics(&targets, &nontargets, 80.0, 1.0)
            .expect("should calculate report");

        assert!((report.eer - 0.25).abs() < 0.01);
        assert!((report.gaussian_dprime - 1.35).abs() < 0.05);
    }

    #[test]
    fn refuse_fewer_than_20_files_per_class() {
        let targets_19 = vec![0.8_f32; 19];
        let nontargets_20 = vec![0.2_f32; 20];

        let err =
            calculate_verification_metrics(&targets_19, &nontargets_20, 10.0, 0.1).unwrap_err();
        assert!(err.contains("fewer than 20 files per class"));

        let targets_20 = vec![0.8_f32; 20];
        let nontargets_19 = vec![0.2_f32; 19];

        let err2 =
            calculate_verification_metrics(&targets_20, &nontargets_19, 10.0, 0.1).unwrap_err();
        assert!(err2.contains("fewer than 20 files per class"));
    }
}
