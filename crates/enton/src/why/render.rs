use enton_core::SpeechCue;

use super::audit::{Context, Report, SensorEvidence};
use super::explain::summary;

pub(super) const SECOND_MS: u64 = 1_000;
const MINUTE_MS: u64 = 60 * SECOND_MS;
const HOUR_MS: u64 = 60 * MINUTE_MS;
pub(super) const DAY_MS: u64 = 24 * HOUR_MS;

/// A span of Enton's clock in the largest unit that keeps it readable.
pub(super) fn duration(ms: u64) -> String {
    if ms < SECOND_MS {
        format!("{ms} ms")
    } else if ms < 10 * SECOND_MS {
        format!("{:.1} s", ms as f64 / SECOND_MS as f64)
    } else if ms < MINUTE_MS {
        format!("{} s", ms / SECOND_MS)
    } else if ms < HOUR_MS {
        format!("{} min", ms / MINUTE_MS)
    } else if ms < DAY_MS {
        let minutes = ms % HOUR_MS / MINUTE_MS;
        if minutes == 0 {
            format!("{} h", ms / HOUR_MS)
        } else {
            format!("{} h {minutes} min", ms / HOUR_MS)
        }
    } else {
        let hours = ms % DAY_MS / HOUR_MS;
        if hours == 0 {
            format!("{} d", ms / DAY_MS)
        } else {
            format!("{} d {hours} h", ms / DAY_MS)
        }
    }
}

/// A span given on the command line, in the largest unit that divides it exactly.
pub(super) fn exact_duration(ms: u64) -> String {
    [
        (DAY_MS, "d"),
        (HOUR_MS, "h"),
        (MINUTE_MS, "min"),
        (SECOND_MS, "s"),
    ]
    .into_iter()
    .find(|(unit, _)| ms >= *unit && ms.is_multiple_of(*unit))
    .map_or_else(
        || format!("{ms} ms"),
        |(unit, name)| format!("{} {name}", ms / unit),
    )
}

/// How long before the last recorded event something happened.
pub(super) fn ago(ms: u64) -> String {
    if ms == 0 {
        "at the last event".to_owned()
    } else {
        format!("{} ago", duration(ms))
    }
}

pub(super) fn nats(value: f32) -> String {
    format!("{value:+.1} nats")
}

fn cue_line(cue: &SpeechCue) -> String {
    format!(
        "{}, {} long, energy {:.2}, VAD {:.2}",
        if cue.keyword { "name heard" } else { "no name" },
        duration(u64::from(cue.duration_ms)),
        cue.energy,
        cue.vad_confidence
    )
}

pub(super) fn evidence_line(evidence: &SensorEvidence) -> String {
    let parts: Vec<String> = [
        ("owner over other", evidence.owner_over_other),
        ("owner over reproduced", evidence.owner_over_reproduced),
        ("live over reproduced", evidence.live_over_reproduced),
        (
            "finished over unfinished",
            evidence.finished_over_unfinished,
        ),
        ("addressed over not", evidence.addressed_over_not),
        ("owner-live", evidence.owner_live),
    ]
    .into_iter()
    .filter_map(|(name, llr)| llr.map(|llr| format!("{name} {llr:+.2}")))
    .collect();
    if parts.is_empty() {
        "none: no voice, media, end-of-turn or directedness sensor ran".to_owned()
    } else {
        format!("{} nats", parts.join(", "))
    }
}

pub(super) fn context_line(context: &Context) -> String {
    let window = match (
        context.attention_left_ms,
        context.verified_attention_left_ms,
    ) {
        (Some(left), _) => format!("attention window open ({} left)", duration(left)),
        (None, Some(left)) => format!(
            "only the verified-voice window open ({} left)",
            duration(left)
        ),
        (None, None) => "no attention window".to_owned(),
    };
    let pending = if context.name_pending {
        "\"Enton...\" pending"
    } else {
        "no \"Enton...\" pending"
    };
    let tv = match context.tv_on_for_ms {
        Some(ms) => {
            format!("TV on ({:.2}, for {})", context.tv_presence, duration(ms))
        }
        None => format!("TV off ({:.2})", context.tv_presence),
    };
    let body = if context.torpor { "in torpor" } else { "awake" };
    let speaking = if context.enton_speaking {
        ", Enton speaking"
    } else {
        ""
    };
    format!(
        "{window}, {pending}, {tv}, {body}{speaking}, budget {:.1}/{:.1} obligation and {:.1}/{:.1} discretionary (a thought costs {:.1})",
        context.obligation_budget.available,
        context.obligation_budget.capacity,
        context.discretionary_budget.available,
        context.discretionary_budget.capacity,
        context.think_cost
    )
}

/// The report as text for a terminal.
pub(super) fn render(report: &Report) -> String {
    let mut lines = vec![format!(
        "Soul: {} (profile {})",
        report.soul, report.profile
    )];
    let range = match (report.first_seq, report.last_seq) {
        (Some(first), Some(last)) => format!(", seq {first} to {last}"),
        _ => String::new(),
    };
    let start = report.from_snapshot.map_or_else(
        || "from a fresh organism".to_owned(),
        |seq| format!("from the snapshot at seq {seq}; older events were pruned"),
    );
    lines.push(format!(
        "Replayed {} events{range}, {start}.",
        report.events
    ));
    let window = report
        .since_ms
        .map(|since| format!(" in the last {}", exact_duration(since)))
        .unwrap_or_default();
    let shown = if report.cues.len() as u64 == report.speech_cues {
        format!("All {}", report.cues.len())
    } else {
        format!("The last {} of {}", report.cues.len(), report.speech_cues)
    };
    if report.cues.is_empty() {
        lines.push(format!(
            "No speech cues{window} ({} in the soul).",
            report.speech_cues
        ));
    } else {
        lines.push(format!(
            "{shown} speech cues{window}; times count back on Enton's clock from the last recorded event (t = {} ms).",
            report.last_event_ms
        ));
    }
    for record in &report.cues {
        lines.push(String::new());
        lines.push(format!(
            "{} (t = {} ms, seq {}): {}",
            ago(record.ago_ms),
            record.at_ms,
            record.seq,
            summary(&record.decision)
        ));
        lines.push(format!("  cue       {}", cue_line(&record.cue)));
        lines.push(format!("  evidence  {}", evidence_line(&record.evidence)));
        lines.push(format!("  context   {}", context_line(&record.context)));
        lines.push(format!("  why       {}", record.explanation));
    }
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::super::audit::BudgetLeft;
    use super::*;
    use enton_core::{Senses, SpeechCue};

    fn tv_line() -> SpeechCue {
        SpeechCue {
            energy: 0.6,
            duration_ms: 1_500,
            vad_confidence: 0.9,
            media: Some(0.8),
            ..SpeechCue::default()
        }
    }

    fn name_alone() -> SpeechCue {
        SpeechCue {
            energy: 0.8,
            duration_ms: 300,
            vad_confidence: 0.95,
            keyword: true,
            ..SpeechCue::default()
        }
    }

    fn quiet_context() -> Context {
        Context {
            attention_left_ms: None,
            verified_attention_left_ms: None,
            name_pending: false,
            tv_presence: 0.0,
            tv_on_for_ms: None,
            torpor: false,
            enton_speaking: false,
            obligation_budget: BudgetLeft {
                available: 0.4,
                capacity: 120.0,
            },
            discretionary_budget: BudgetLeft {
                available: 12.0,
                capacity: 12.0,
            },
            think_cost: 1.0,
        }
    }

    #[test]
    fn durations_read_in_the_largest_useful_unit() {
        assert_eq!(duration(450), "450 ms");
        assert_eq!(duration(4_000), "4.0 s");
        assert_eq!(duration(42_500), "42 s");
        assert_eq!(duration(180_000), "3 min");
        assert_eq!(duration(3_600_000), "1 h");
        assert_eq!(duration(3_900_000), "1 h 5 min");
        assert_eq!(duration(DAY_MS), "1 d");
        assert_eq!(duration(DAY_MS + 2 * HOUR_MS + 5), "1 d 2 h");
        assert_eq!(exact_duration(90_000), "90 s");
        assert_eq!(exact_duration(300_000), "5 min");
        assert_eq!(exact_duration(7_200_000), "2 h");
        assert_eq!(exact_duration(DAY_MS), "1 d");
        assert_eq!(exact_duration(1_500), "1500 ms");
        assert_eq!(ago(0), "at the last event");
        assert_eq!(ago(125_000), "2 min ago");
    }

    #[test]
    fn evidence_is_listed_only_for_sensors_that_ran() {
        let senses = Senses::calibrated();
        let tagged = SensorEvidence::read(&senses, &tv_line());
        assert!(tagged.owner_over_other.is_none());
        assert!(tagged.owner_over_reproduced.is_none());
        assert!(tagged.finished_over_unfinished.is_none());
        assert!(tagged.addressed_over_not.is_none());
        let live = tagged.live_over_reproduced.unwrap();
        assert!((live + 2.4).abs() < 1e-5, "{live}");
        assert_eq!(tagged.owner_live, Some(live));
        assert_eq!(
            evidence_line(&tagged),
            "live over reproduced -2.40, owner-live -2.40 nats"
        );

        let bare = SensorEvidence::read(&senses, &name_alone());
        assert_eq!(bare, SensorEvidence::default());
        assert!(evidence_line(&bare).starts_with("none: "));

        let json = serde_json::to_value(tagged).unwrap();
        let keys: Vec<&str> = json
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ["live_over_reproduced", "owner_live"]);
    }

    #[test]
    fn the_context_line_shows_windows_tv_body_and_budget() {
        let mut context = quiet_context();
        assert_eq!(
            context_line(&context),
            "no attention window, no \"Enton...\" pending, TV off (0.00), awake, budget 0.4/120.0 obligation and 12.0/12.0 discretionary (a thought costs 1.0)"
        );
        context.attention_left_ms = Some(3_200);
        context.name_pending = true;
        context.tv_presence = 0.67;
        context.tv_on_for_ms = Some(180_000);
        context.torpor = true;
        context.enton_speaking = true;
        assert!(context_line(&context).starts_with(
            "attention window open (3.2 s left), \"Enton...\" pending, TV on (0.67, for 3 min), in torpor, Enton speaking, "
        ));
    }
}
