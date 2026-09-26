use enton_core::SpeechCue;

use super::audit::{
    Context, CueRecord, Decision, Fate, PersonaOrigin, PersonaShown, Report, SensorEvidence, Then,
};
use super::explain::summary;

pub(super) const SECOND_MS: u64 = 1_000;
const MINUTE_MS: u64 = 60 * SECOND_MS;
const HOUR_MS: u64 = 60 * MINUTE_MS;
pub(super) const DAY_MS: u64 = 24 * HOUR_MS;
/// Personas named in the report's header; `--json` lists them all.
const PERSONAS_SHOWN: usize = 3;

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

/// A persona by its short hash, its origin and its length.
pub(super) fn persona_name(persona: &PersonaShown) -> String {
    let origin = match persona.source {
        PersonaOrigin::BuiltIn => "built-in default",
        PersonaOrigin::File => "file",
    };
    format!("{} ({origin}, {} bytes)", persona.short, persona.bytes)
}

/// The header line naming the most recent personas, if any was recorded.
fn personas_line(personas: &[PersonaShown]) -> Option<String> {
    let recent: Vec<String> = personas
        .iter()
        .rev()
        .take(PERSONAS_SHOWN)
        .rev()
        .map(|persona| {
            format!(
                "{} first used at seq {}",
                persona_name(persona),
                persona.first_seq
            )
        })
        .collect();
    let earlier = personas.len().saturating_sub(recent.len());
    let more = if earlier > 0 {
        format!(" ({earlier} earlier in --json)")
    } else {
        String::new()
    };
    let label = if personas.len() == 1 {
        "Persona"
    } else {
        "Personas"
    };
    (!recent.is_empty()).then(|| format!("{label}: {}{more}.", recent.join("; ")))
}

/// The thought behind a cue: its own decision, or the one a timed-out wait made.
fn thought_of(record: &CueRecord) -> Option<&Decision> {
    match (&record.decision, &record.then) {
        (decision @ Decision::Think { .. }, _)
        | (
            _,
            Some(Then::Timeout {
                decision: decision @ Decision::Think { .. },
                ..
            }),
        ) => Some(decision),
        _ => None,
    }
}

/// Which persona a thought was asked with, and a warning when it changed
/// since the thought before.
pub(super) fn persona_lines(decision: &Decision) -> Vec<String> {
    let Decision::Think {
        persona,
        persona_changed,
        fate,
        ..
    } = decision
    else {
        return Vec::new();
    };
    let mut lines = vec![match (persona, fate) {
        (Some(persona), _) => format!("  persona   {}", persona_name(persona)),
        (None, Some(Fate::Unrecorded)) => {
            "  persona   unknown: the soul holds no record of this thought".to_owned()
        }
        (None, _) => "  persona   unknown: recorded before the soul kept personas".to_owned(),
    }];
    if let Some(change) = persona_changed {
        lines.push(format!(
            "  warning   the persona changed since thought #{}, which was asked with {}",
            change.thought,
            persona_name(&change.persona)
        ));
    }
    lines
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
    lines.extend(personas_line(&report.personas));
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
        if record.cut_off {
            lines.push("  playback  cut off: Enton stopped speaking for this cue".to_owned());
        }
        if let Some(thought) = thought_of(record) {
            lines.extend(persona_lines(thought));
        }
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

    fn persona(short: &str, source: PersonaOrigin, first_seq: u64) -> PersonaShown {
        PersonaShown {
            sha256: format!("{short}{}", "0".repeat(52)),
            short: short.to_owned(),
            bytes: 612,
            source,
            first_seq,
        }
    }

    #[test]
    fn a_thought_names_its_persona_and_warns_when_it_changed() {
        let think = |persona, persona_changed, fate| Decision::Think {
            thought: 4,
            reason: enton_core::Reason::Keyword,
            salience: 1.6,
            propensity: None,
            fate,
            persona,
            persona_changed,
        };
        let file = persona("3f1a9c0d22b7", PersonaOrigin::File, 31);
        let built_in = persona("5be0c1d9a4e2", PersonaOrigin::BuiltIn, 2);
        assert_eq!(
            persona_lines(&think(Some(Box::new(built_in.clone())), None, None)),
            ["  persona   5be0c1d9a4e2 (built-in default, 612 bytes)"]
        );
        let changed = super::super::audit::PersonaChange {
            thought: 3,
            persona: built_in.clone(),
        };
        assert_eq!(
            persona_lines(&think(
                Some(Box::new(file.clone())),
                Some(Box::new(changed)),
                None
            )),
            [
                "  persona   3f1a9c0d22b7 (file, 612 bytes)",
                "  warning   the persona changed since thought #3, which was asked with 5be0c1d9a4e2 (built-in default, 612 bytes)"
            ]
        );
        assert_eq!(
            persona_lines(&think(None, None, Some(Fate::Pending))),
            ["  persona   unknown: recorded before the soul kept personas"]
        );
        assert_eq!(
            persona_lines(&think(None, None, Some(Fate::Unrecorded))),
            ["  persona   unknown: the soul holds no record of this thought"]
        );
        assert!(persona_lines(&Decision::Other).is_empty());

        assert_eq!(personas_line(&[]), None);
        assert_eq!(
            personas_line(std::slice::from_ref(&built_in)).as_deref(),
            Some("Persona: 5be0c1d9a4e2 (built-in default, 612 bytes) first used at seq 2.")
        );
        let many = [
            persona("aaaaaaaaaaaa", PersonaOrigin::File, 1),
            built_in,
            persona("bbbbbbbbbbbb", PersonaOrigin::File, 9),
            file,
        ];
        assert_eq!(
            personas_line(&many).as_deref(),
            Some(
                "Personas: 5be0c1d9a4e2 (built-in default, 612 bytes) first used at seq 2; bbbbbbbbbbbb (file, 612 bytes) first used at seq 9; 3f1a9c0d22b7 (file, 612 bytes) first used at seq 31 (1 earlier in --json)."
            )
        );
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
