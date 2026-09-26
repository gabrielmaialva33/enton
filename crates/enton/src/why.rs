//! `enton why`: replay the soul and explain the latest speech decisions.
//!
//! The soul keeps every reduced event and the reducer is deterministic, so
//! stepping an organism through the log recomputes each decision exactly,
//! together with the state it was made in. Replay starts as early as the log
//! allows (a fresh organism, or the earliest snapshot a pruned log continues).
//!
//! Only what the soul stores is shown: cue measurements, decisions, the
//! evidence recomputed from those measurements with the profile's calibration,
//! the organism's own state, and how recorded thoughts resolved. The soul keeps
//! no transcripts, and the text of Enton's replies is never read back here.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use enton_adapters::{ActionStatus, SeqNo, Soul, SoulConfig, soul};
use enton_core::{
    Abstention, Action, Budget, Event, Millis, Organism, PlaybackStatus, Profile, Reason, Senses,
    SpeechCue, ThoughtId,
};
use serde::{Deserialize, Serialize};

use crate::cli::WhyConfig;

/// Events read from the soul per query.
const PAGE: usize = 1024;

const SECOND_MS: u64 = 1_000;
const MINUTE_MS: u64 = 60 * SECOND_MS;
const HOUR_MS: u64 = 60 * MINUTE_MS;
const DAY_MS: u64 = 24 * HOUR_MS;

/// Why the audit could not run.
#[derive(Debug, thiserror::Error)]
pub(crate) enum WhyError {
    /// No `--soul` and no home directory to derive the default location from.
    #[error("no soul path: set HOME or XDG_DATA_HOME, or pass --soul <PATH>")]
    NoPath,
    /// The log does not exist.
    #[error("no soul at {}: Enton has not recorded anything there yet", .0.display())]
    Missing(PathBuf),
    /// The log exists but could not be opened or replayed.
    #[error("cannot replay the soul at {}: {source}", path.display())]
    Soul {
        /// The log that failed.
        path: PathBuf,
        /// What failed.
        source: soul::Error,
    },
    /// Serializing the report failed.
    #[error("JSON output failed: {0}")]
    Json(#[from] serde_json::Error),
}

/// Replay the soul named by `config` and explain its latest speech cues, as
/// text or, with `--json`, as one JSON document.
pub(crate) fn run(config: &WhyConfig) -> Result<String, WhyError> {
    let path = config.soul.as_deref().ok_or(WhyError::NoPath)?;
    let report = audit(path, &config.profile, config.last, config.since_ms)?;
    if config.json {
        let mut out = serde_json::to_string_pretty(&report)?;
        out.push('\n');
        Ok(out)
    } else {
        Ok(render(&report))
    }
}

/// Everything `enton why` reports.
#[derive(Debug, Serialize)]
pub(crate) struct Report {
    /// The log that was replayed.
    soul: String,
    /// The profile whose reducer and calibration replayed it.
    profile: String,
    /// The snapshot replay started from, when older events were pruned.
    from_snapshot: Option<SeqNo>,
    /// How many events were replayed.
    events: u64,
    /// Sequence number of the first replayed event.
    first_seq: Option<SeqNo>,
    /// Sequence number of the last replayed event.
    last_seq: Option<SeqNo>,
    /// Monotonic time of the last recorded event; `ago_ms` counts back from it.
    last_event_ms: u64,
    /// How many speech cues were replayed.
    speech_cues: u64,
    /// The cues shown, oldest first.
    cues: Vec<CueRecord>,
    /// The `--since` window, if one was given.
    since_ms: Option<u64>,
}

/// One speech cue, what was decided and why.
#[derive(Debug, Clone, Serialize)]
struct CueRecord {
    seq: SeqNo,
    at_ms: u64,
    /// Milliseconds on Enton's clock between this cue and the last recorded event.
    ago_ms: u64,
    /// The cue as stored: measurements only, never words.
    cue: SpeechCue,
    decision: Decision,
    evidence: SensorEvidence,
    context: Context,
    /// For an `Attend`: how the wait for the rest of the request ended.
    #[serde(skip_serializing_if = "Option::is_none")]
    then: Option<Then>,
    explanation: String,
}

/// A decision, as `enton why` shows it.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Decision {
    /// The cortex was woken.
    Think {
        thought: u64,
        reason: Reason,
        salience: f32,
        /// The probability of thinking, when a borderline cue flipped the
        /// exploration coin.
        #[serde(skip_serializing_if = "Option::is_none")]
        propensity: Option<f32>,
        /// How the thought resolved, from the soul's action records.
        #[serde(skip_serializing_if = "Option::is_none")]
        fate: Option<Fate>,
    },
    /// Enton waited for the rest of a request.
    Attend { until_ms: u64 },
    /// The cortex was not woken.
    Abstain {
        reason: Reason,
        salience: f32,
        why: Abstention,
        /// The probability of abstaining, when a borderline cue flipped the
        /// exploration coin.
        #[serde(skip_serializing_if = "Option::is_none")]
        propensity: Option<f32>,
    },
    /// No decision, or one this version cannot show.
    Other,
}

/// How a recorded thought resolved.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Fate {
    /// The cortex replied; only the reply's length is stored.
    Done { reply_chars: Option<u64> },
    /// The thought failed, with the reason the runtime recorded.
    Failed { failure: Option<String> },
    /// Never resolved: still running, or cut off by a crash not yet reconciled.
    Pending,
    /// The soul holds no record of the thought.
    Unrecorded,
}

/// How an `Attend` ended.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Then {
    /// The window closed with nothing more, and the name alone was decided on.
    Timeout { at_ms: u64, decision: Decision },
    /// A later cue carried the turn on.
    Continued { at_ms: u64, seq: SeqNo },
}

/// The calibrated evidence of one cue, in nats, only for sensors that ran.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
struct SensorEvidence {
    #[serde(skip_serializing_if = "Option::is_none")]
    owner_over_other: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    owner_over_reproduced: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    live_over_reproduced: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finished_over_unfinished: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    addressed_over_not: Option<f32>,
    /// The owner speaking live, over the likelier alternative.
    #[serde(skip_serializing_if = "Option::is_none")]
    owner_live: Option<f32>,
}

impl SensorEvidence {
    fn read(senses: &Senses, cue: &SpeechCue) -> Self {
        let cue = cue.canonical();
        let evidence = senses.read(&cue);
        let voice = cue.speaker_sim.is_some();
        let source = cue.media.is_some();
        Self {
            owner_over_other: voice.then_some(evidence.owner_over_other),
            owner_over_reproduced: voice.then_some(evidence.owner_over_reproduced),
            live_over_reproduced: source.then_some(evidence.live_over_reproduced),
            finished_over_unfinished: cue.turn_complete.map(|_| evidence.finished_over_unfinished),
            addressed_over_not: cue.directed.map(|_| evidence.addressed_over_not),
            owner_live: (voice || source).then(|| evidence.owner_live()),
        }
    }
}

/// The state the reducer decided a cue in.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct Context {
    /// Time left in the attention window open to any voice, if one was open.
    attention_left_ms: Option<u64>,
    /// Time left in the longer window open only to the verified voice (or to
    /// speech clearly addressed to Enton), if one was open.
    verified_attention_left_ms: Option<u64>,
    /// Whether an "Enton..." was waiting for the rest of its request.
    name_pending: bool,
    /// Belief, from zero to one, that a TV was playing, as the cue was weighed.
    tv_presence: f32,
    /// How long that belief had been at or above the profile's TV-on level, or
    /// `None` while the TV counted as off.
    tv_on_for_ms: Option<u64>,
    /// Whether the body was in torpor (fever or a critical battery).
    torpor: bool,
    /// Whether Enton's own voice, or its echo, overlapped the cue.
    enton_speaking: bool,
    /// What was left in the account for addressed turns.
    obligation_budget: BudgetLeft,
    /// What was left in the account for overheard speech and drives.
    discretionary_budget: BudgetLeft,
    /// What one thought costs.
    think_cost: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct BudgetLeft {
    available: f32,
    capacity: f32,
}

impl From<&Budget> for BudgetLeft {
    fn from(budget: &Budget) -> Self {
        Self {
            available: budget.available,
            capacity: budget.capacity,
        }
    }
}

impl Context {
    /// The organism's state just before it reduces a cue at `now`. The TV fields
    /// are filled after the step, because the reducer tracks the TV first.
    fn before(organism: &Organism, now: Millis) -> Self {
        let profile = organism.profile();
        let open = |until: Option<Millis>| {
            until
                .filter(|until| now < *until)
                .map(|until| until.since(now))
        };
        let status = organism.playback_status();
        let enton_speaking = if let PlaybackStatus::Speaking { started_at, .. } = status {
            now.since(started_at) < profile.echo.max_playback_ms
        } else if let PlaybackStatus::Hangover { until, .. } = status {
            now < until
        } else {
            false
        };
        Self {
            attention_left_ms: open(organism.attention_until()),
            verified_attention_left_ms: open(organism.verified_attention_until()),
            name_pending: organism.is_attending(),
            tv_presence: 0.0,
            tv_on_for_ms: None,
            torpor: organism.is_torpid(),
            enton_speaking,
            obligation_budget: organism.obligation_budget().into(),
            discretionary_budget: organism.discretionary_budget().into(),
            think_cost: profile.budgets.think_cost,
        }
    }
}

/// The single decision among `action`, as `enton why` shows it.
fn decision(action: Option<&Action>) -> Decision {
    let Some(action) = action else {
        return Decision::Other;
    };
    // Below one, the action was a coin flip at a borderline cue.
    let propensity = Some(action.propensity()).filter(|probability| *probability < 1.0);
    if let Action::Think {
        thought,
        reason,
        salience,
        ..
    } = action
    {
        Decision::Think {
            thought: thought.0,
            reason: reason.clone(),
            salience: *salience,
            propensity,
            fate: None,
        }
    } else if let Action::Attend { until, .. } = action {
        Decision::Attend { until_ms: until.0 }
    } else if let Action::Abstain {
        reason,
        salience,
        why,
        ..
    } = action
    {
        Decision::Abstain {
            reason: reason.clone(),
            salience: *salience,
            why: *why,
            propensity,
        }
    } else {
        Decision::Other
    }
}

/// Steps an organism through the log and keeps the most recent speech cues.
#[derive(Debug)]
struct Recorder {
    keep: usize,
    cues: VecDeque<CueRecord>,
    speech_cues: u64,
    events: u64,
    first_seq: Option<SeqNo>,
    last_seq: Option<SeqNo>,
    /// When the TV started counting as on, while it does.
    tv_on_since: Option<Millis>,
    /// The cue whose `Attend` is still waiting for the rest of its request.
    waiting: Option<SeqNo>,
}

impl Recorder {
    fn new(keep: usize) -> Self {
        Self {
            keep: keep.max(1),
            cues: VecDeque::new(),
            speech_cues: 0,
            events: 0,
            first_seq: None,
            last_seq: None,
            tv_on_since: None,
            waiting: None,
        }
    }

    /// Reduce one stored event, recording what a speech cue decided and why.
    fn step(&mut self, organism: &mut Organism, seq: SeqNo, event: &Event) {
        self.events += 1;
        self.first_seq.get_or_insert(seq);
        self.last_seq = Some(seq);
        let now = event.now();
        let waiting_before = organism.is_attending();

        let Event::Speech { cue, .. } = event else {
            let actions = organism.step(event);
            if waiting_before && !organism.is_attending() {
                // Only a tick ends the wait on its own: the name alone is decided on.
                let timeout = actions.iter().find(|action| {
                    matches!(
                        action,
                        Action::Think {
                            reason: Reason::Keyword,
                            ..
                        } | Action::Abstain {
                            reason: Reason::Keyword,
                            ..
                        }
                    )
                });
                self.resolve_wait(Then::Timeout {
                    at_ms: now.0,
                    decision: decision(timeout),
                });
            }
            return;
        };

        let cue = cue.canonical();
        let tv_on_level = organism.profile().source.tv_on_level;
        // The belief only decays between lines, so a dip below the level shows here.
        if organism.tv_presence_as_of(now) < tv_on_level {
            self.tv_on_since = None;
        }
        let mut context = Context::before(organism, now);
        let evidence = SensorEvidence::read(&organism.profile().senses, &cue);
        let decision = decision(organism.step(event).first());

        context.tv_presence = organism.tv_presence_as_of(now);
        if context.tv_presence >= tv_on_level {
            let since = *self.tv_on_since.get_or_insert(now);
            context.tv_on_for_ms = Some(now.since(since));
        } else {
            self.tv_on_since = None;
        }

        let attends = matches!(decision, Decision::Attend { .. });
        if waiting_before && (attends || !organism.is_attending()) {
            self.resolve_wait(Then::Continued { at_ms: now.0, seq });
        }
        if attends {
            self.waiting = Some(seq);
        }

        self.speech_cues += 1;
        self.cues.push_back(CueRecord {
            seq,
            at_ms: now.0,
            ago_ms: 0,
            cue,
            decision,
            evidence,
            context,
            then: None,
            explanation: String::new(),
        });
        if self.cues.len() > self.keep {
            self.cues.pop_front();
        }
    }

    fn resolve_wait(&mut self, then: Then) {
        if let Some(seq) = self.waiting.take()
            && let Some(record) = self.cues.iter_mut().find(|record| record.seq == seq)
        {
            record.then = Some(then);
        }
    }

    /// The cues within `since_ms` of the last event, with their age filled in.
    fn finish(self, last_event: Millis, since_ms: Option<u64>) -> Vec<CueRecord> {
        self.cues
            .into_iter()
            .map(|mut record| {
                record.ago_ms = last_event.since(Millis(record.at_ms));
                record
            })
            .filter(|record| since_ms.is_none_or(|since| record.ago_ms <= since))
            .collect()
    }
}

/// Replay the log at `path` and explain its `last` most recent speech cues.
fn audit(
    path: &Path,
    profile: &Profile,
    last: usize,
    since_ms: Option<u64>,
) -> Result<Report, WhyError> {
    if !path.exists() {
        return Err(WhyError::Missing(path.to_path_buf()));
    }
    let failed = |source| WhyError::Soul {
        path: path.to_path_buf(),
        source,
    };
    let soul = Soul::open_read_only(path, SoulConfig::default()).map_err(failed)?;
    let (mut organism, start) = soul.earliest_organism(profile).map_err(failed)?;

    let mut recorder = Recorder::new(last);
    let mut cursor = start;
    loop {
        let page = soul.read_after(cursor, PAGE).map_err(failed)?;
        for (seq, event) in &page {
            recorder.step(&mut organism, *seq, event);
        }
        match page.last() {
            Some((seq, _)) if page.len() == PAGE => cursor = *seq,
            _ => break,
        }
    }

    let (events, speech_cues, first_seq, last_seq) = (
        recorder.events,
        recorder.speech_cues,
        recorder.first_seq,
        recorder.last_seq,
    );
    let last_event = organism.last_seen();
    let mut cues = recorder.finish(last_event, since_ms);
    for record in &mut cues {
        if let Decision::Think { thought, fate, .. } = &mut record.decision {
            *fate = Some(fate_of(&soul, *thought).map_err(failed)?);
        }
        if let Some(Then::Timeout {
            decision: Decision::Think { thought, fate, .. },
            ..
        }) = &mut record.then
        {
            *fate = Some(fate_of(&soul, *thought).map_err(failed)?);
        }
        record.explanation = explain(record, profile);
    }

    Ok(Report {
        soul: path.display().to_string(),
        profile: profile.name.clone(),
        from_snapshot: (start > 0).then_some(start),
        events,
        first_seq,
        last_seq,
        last_event_ms: last_event.0,
        speech_cues,
        cues,
        since_ms,
    })
}

/// The small result a resolved thought was stored with.
#[derive(Debug, Default, Deserialize)]
struct Resolution {
    chars: Option<u64>,
    reason: Option<String>,
}

fn fate_of(soul: &Soul, thought: u64) -> Result<Fate, soul::Error> {
    let Some((status, result)) = soul.thought_status(ThoughtId(thought))? else {
        return Ok(Fate::Unrecorded);
    };
    let resolution = result
        .and_then(|json| serde_json::from_str::<Resolution>(&json).ok())
        .unwrap_or_default();
    Ok(match status {
        ActionStatus::Done => Fate::Done {
            reply_chars: resolution.chars,
        },
        ActionStatus::Failed => Fate::Failed {
            failure: resolution.reason,
        },
        ActionStatus::Pending => Fate::Pending,
    })
}

/// A span of Enton's clock in the largest unit that keeps it readable.
fn duration(ms: u64) -> String {
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
fn exact_duration(ms: u64) -> String {
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
fn ago(ms: u64) -> String {
    if ms == 0 {
        "at the last event".to_owned()
    } else {
        format!("{} ago", duration(ms))
    }
}

fn nats(value: f32) -> String {
    format!("{value:+.1} nats")
}

fn reason_name(reason: &Reason) -> String {
    if let Reason::Drive(name) = reason {
        format!("drive {name}")
    } else {
        format!("{reason:?}")
    }
}

/// One plain-language line: what was decided, and the evidence or state behind it.
fn explain(record: &CueRecord, profile: &Profile) -> String {
    match &record.decision {
        Decision::Think {
            thought,
            reason,
            salience,
            propensity,
            fate,
        } => {
            let cause = if *reason == Reason::Keyword {
                if record.cue.keyword {
                    "Enton was called by name".to_owned()
                } else {
                    "this finished the request that began with \"Enton...\"".to_owned()
                }
            } else if *reason == Reason::FollowUp {
                "a follow-up inside the conversation window".to_owned()
            } else if *reason == Reason::Speech {
                format!(
                    "overheard speech was salient enough to think about (salience {salience:.2})"
                )
            } else {
                format!("{} ignited a thought", reason_name(reason))
            };
            let coin = propensity
                .map(|p| {
                    format!(
                        "; the cue was borderline, and a coin flip explored it (probability {p:.2})"
                    )
                })
                .unwrap_or_default();
            format!(
                "Thought #{thought} ({}): {cause}{coin}{}.",
                reason_name(reason),
                fate_clause(fate.as_ref())
            )
        }
        Decision::Attend { until_ms } => format!(
            "Waited (Attend): Enton heard its name and waited up to {} for the rest of the request; {}.",
            duration(until_ms.saturating_sub(record.at_ms)),
            then_clause(record)
        ),
        Decision::Abstain {
            reason,
            salience,
            why,
            propensity,
        } => format!(
            "Abstained ({why:?}): {}{}.",
            abstention_cause(record, profile, reason, *salience, *why),
            propensity
                .map(|p| format!(
                    "; the cue was borderline, and a coin flip passed on it (probability {p:.2})"
                ))
                .unwrap_or_default()
        ),
        Decision::Other => "No decision this version of enton why can show.".to_owned(),
    }
}

fn fate_clause(fate: Option<&Fate>) -> String {
    match fate {
        Some(Fate::Done {
            reply_chars: Some(chars),
        }) => format!("; the cortex replied ({chars} characters)"),
        Some(Fate::Done { reply_chars: None }) => "; the cortex replied".to_owned(),
        Some(Fate::Failed {
            failure: Some(failure),
        }) => format!(", but the thought failed: {failure}"),
        Some(Fate::Failed { failure: None }) => ", but the thought failed".to_owned(),
        Some(Fate::Pending) => {
            ", but the thought never resolved (still running, or cut off by a crash)".to_owned()
        }
        Some(Fate::Unrecorded) | None => String::new(),
    }
}

fn then_clause(record: &CueRecord) -> String {
    match &record.then {
        Some(Then::Timeout { at_ms, decision }) => {
            let after = duration(at_ms.saturating_sub(record.at_ms));
            match decision {
                Decision::Think { thought, fate, .. } => format!(
                    "nothing followed, so after {after} it answered the name alone (thought #{thought}{})",
                    fate_clause(fate.as_ref())
                ),
                Decision::Abstain { why, .. } => format!(
                    "nothing followed, and when the window closed after {after} it abstained ({why:?})"
                ),
                Decision::Attend { .. } | Decision::Other => {
                    format!("nothing followed, and the window closed after {after}")
                }
            }
        }
        Some(Then::Continued { at_ms, seq }) => format!(
            "the cue at seq {seq}, {} later, carried the turn on",
            duration(at_ms.saturating_sub(record.at_ms))
        ),
        None => "it was still waiting at the last recorded event".to_owned(),
    }
}

/// While the TV counted as on, how long it had been.
fn tv_clause(context: &Context) -> String {
    match context.tv_on_for_ms {
        Some(ms) if ms >= SECOND_MS => {
            format!("; the TV had been on for {}", duration(ms))
        }
        Some(_) => "; the TV had just come on".to_owned(),
        None => String::new(),
    }
}

// `Abstention` is not `#[non_exhaustive]`, so a wildcard after its nine variants is
// unreachable today. It stays so that `enton why` keeps compiling, with a generic
// line, while the core grows a new abstention.
#[allow(unreachable_patterns)]
fn abstention_cause(
    record: &CueRecord,
    profile: &Profile,
    reason: &Reason,
    salience: f32,
    why: Abstention,
) -> String {
    let evidence = &record.evidence;
    let context = &record.context;
    let owner_live = evidence
        .owner_live
        .map(|llr| format!(" (owner-live {})", nats(llr)))
        .unwrap_or_default();
    let addressed = *reason == Reason::Keyword || *reason == Reason::FollowUp;
    match why {
        Abstention::Media => {
            let llr = evidence
                .live_over_reproduced
                .map(|llr| format!(", {}", nats(llr)))
                .unwrap_or_default();
            let over = if context.enton_speaking {
                " over Enton's own voice"
            } else {
                ""
            };
            format!(
                "the tagger heard a loudspeaker{over}{llr}{}",
                tv_clause(context)
            )
        }
        Abstention::OtherSpeaker => {
            if context.enton_speaking {
                format!(
                    "Enton was speaking, and a voice not verified as the owner's{owner_live} cannot interrupt it"
                )
            } else if *reason == Reason::Keyword {
                format!(
                    "someone else said the name{owner_live} while an \"Enton...\" was still waiting for its own speaker"
                )
            } else {
                format!(
                    "the voice did not match whoever addressed Enton{owner_live}, so it could not continue the conversation{}",
                    tv_clause(context)
                )
            }
        }
        Abstention::Undirected => format!(
            "the directedness detector heard speech addressed to someone else{}",
            evidence
                .addressed_over_not
                .map(|llr| format!(", {}", nats(llr)))
                .unwrap_or_default()
        ),
        Abstention::SelfEcho => format!(
            "it overlapped Enton's own voice and did not stand out from the echo as an interruption (energy {:.2}, echo expected near {:.2})",
            record.cue.energy, profile.echo.echo_initial_energy
        ),
        Abstention::BelowThreshold if *reason == Reason::FollowUp => format!(
            "the voice activity was too weak for a follow-up (VAD {:.2}, needs {:.2})",
            record.cue.vad_confidence, profile.attention.follow_up_min_vad
        ),
        Abstention::BelowThreshold => format!(
            "nobody called Enton, and the speech was not salient enough to think about (salience {salience:.2}, needs {:.2})",
            profile.ignition.discretionary_threshold
        ),
        Abstention::Cooldown => format!(
            "nobody called Enton, and it had thought less than {} before; overheard speech waits out that cooldown",
            duration(profile.ignition.cooldown_ms)
        ),
        Abstention::Habituation => format!(
            "similar sounds kept repeating and Enton got used to them (salience {salience:.2} after habituation, needs {:.2})",
            profile.ignition.discretionary_threshold
        ),
        Abstention::OutOfEnergy => {
            let (account, budget) = if addressed {
                ("obligation", context.obligation_budget)
            } else {
                ("discretionary", context.discretionary_budget)
            };
            format!(
                "the {account} budget could not pay for a thought ({:.2} left, a thought costs {:.2})",
                budget.available, context.think_cost
            )
        }
        Abstention::Torpor => {
            "the body was in torpor (fever or a low battery), and then only a call by name buys a thought"
                .to_owned()
        }
        _ => "the reducer gave no reason this version of enton why can explain".to_owned(),
    }
}

fn summary(decision: &Decision) -> String {
    match decision {
        Decision::Think {
            thought,
            reason,
            salience,
            ..
        } => format!(
            "Think #{thought} ({}, salience {salience:.2})",
            reason_name(reason)
        ),
        Decision::Attend { until_ms } => format!("Attend until t = {until_ms} ms"),
        Decision::Abstain {
            reason,
            salience,
            why,
            ..
        } => format!(
            "Abstain: {why:?} ({}, salience {salience:.2})",
            reason_name(reason)
        ),
        Decision::Other => "no decision".to_owned(),
    }
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

fn evidence_line(evidence: &SensorEvidence) -> String {
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

fn context_line(context: &Context) -> String {
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
fn render(report: &Report) -> String {
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
    use super::*;

    fn tick(now: u64) -> Event {
        Event::Tick { now: Millis(now) }
    }

    fn speech(now: u64, cue: SpeechCue) -> Event {
        Event::Speech {
            now: Millis(now),
            cue,
        }
    }

    /// An overheard line that only an audio tagger described: it sounded like a TV.
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

    /// Step `events` through a fresh T1-ref organism, keeping every cue.
    fn replay(events: &[Event]) -> Result<(Recorder, Organism), enton_core::InvalidProfile> {
        let mut organism = Organism::new(Profile::t1_ref())?;
        let mut recorder = Recorder::new(usize::MAX);
        for (index, event) in events.iter().enumerate() {
            recorder.step(&mut organism, index as u64 + 1, event);
        }
        Ok((recorder, organism))
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

    fn record(cue: SpeechCue, decision: Decision, evidence: SensorEvidence) -> CueRecord {
        CueRecord {
            seq: 7,
            at_ms: 60_000,
            ago_ms: 0,
            cue,
            decision,
            evidence,
            context: quiet_context(),
            then: None,
            explanation: String::new(),
        }
    }

    fn abstain(reason: Reason, why: Abstention) -> Decision {
        Decision::Abstain {
            reason,
            salience: 0.5,
            why,
            propensity: None,
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
    fn a_media_abstention_names_the_tagger_and_how_long_the_tv_was_on() {
        // A TV line every 10 s for four minutes: the belief that a TV is playing
        // first reaches the on level on the third line, dips below it between the
        // next few lines, and holds from the fifth on.
        let events: Vec<Event> = (1..=25).map(|n| speech(n * 10_000, tv_line())).collect();
        let (recorder, organism) = replay(&events).unwrap();
        let cues = recorder.finish(organism.last_seen(), None);
        assert_eq!(cues.len(), 25);
        assert!(cues.iter().all(|record| matches!(
            record.decision,
            Decision::Abstain {
                why: Abstention::Media,
                ..
            }
        )));
        assert!(cues[1].context.tv_on_for_ms.is_none());
        assert_eq!(cues[2].context.tv_on_for_ms, Some(0));
        assert!(cues[3].context.tv_on_for_ms.is_some_and(|ms| ms == 0));
        let last = &cues[24];
        assert_eq!(last.context.tv_on_for_ms, Some(200_000));
        assert_eq!(
            explain(&cues[2], &Profile::t1_ref()),
            "Abstained (Media): the tagger heard a loudspeaker, -2.4 nats; the TV had just come on."
        );
        assert_eq!(
            explain(last, &Profile::t1_ref()),
            "Abstained (Media): the tagger heard a loudspeaker, -2.4 nats; the TV had been on for 3 min."
        );
    }

    #[test]
    fn a_long_silence_turns_the_tv_off() {
        let (recorder, organism) = replay(&[
            speech(10_000, tv_line()),
            speech(20_000, tv_line()),
            speech(30_000, tv_line()),
            speech(40_000, tv_line()),
            // Two minutes later the belief has decayed below the on level.
            speech(160_000, tv_line()),
        ])
        .unwrap();
        let cues = recorder.finish(organism.last_seen(), None);
        assert!(cues[3].context.tv_on_for_ms.is_some());
        assert!(cues[4].context.tv_on_for_ms.is_none());
    }

    #[test]
    fn an_attend_records_how_the_wait_ended() {
        let events = [
            tick(1_000),
            speech(2_000, name_alone()),
            tick(4_000),
            tick(7_000),
            speech(8_000, name_alone()),
            speech(9_000, tv_line()),
            speech(9_500, name_alone()),
        ];
        let (recorder, organism) = replay(&events).unwrap();
        let cues = recorder.finish(organism.last_seen(), None);
        let profile = Profile::t1_ref();

        assert!(matches!(
            cues[0].then,
            Some(Then::Timeout {
                at_ms: 7_000,
                decision: Decision::Think { thought: 1, .. }
            })
        ));
        assert_eq!(
            explain(&cues[0], &profile),
            "Waited (Attend): Enton heard its name and waited up to 5.0 s for the rest of the request; nothing followed, so after 5.0 s it answered the name alone (thought #1)."
        );
        // An overheard TV line inside the window does not end the wait; the next name does.
        assert!(matches!(
            cues[1].then,
            Some(Then::Continued {
                at_ms: 9_500,
                seq: 7
            })
        ));
        assert!(cues[2].context.name_pending);
        assert!(cues[2].context.attention_left_ms.is_some());
        assert_eq!(
            explain(&cues[3], &profile),
            "Waited (Attend): Enton heard its name and waited up to 5.0 s for the rest of the request; it was still waiting at the last recorded event."
        );
        assert_eq!(recorder_counts(&events), (7, 4));
    }

    fn recorder_counts(events: &[Event]) -> (u64, u64) {
        replay(events).map_or((0, 0), |(recorder, _)| {
            (recorder.events, recorder.speech_cues)
        })
    }

    #[test]
    fn only_the_most_recent_cues_within_since_are_kept() {
        let events: Vec<Event> = (1..=5).map(|n| speech(n * 60_000, tv_line())).collect();
        let mut organism = Organism::new(Profile::t1_ref()).unwrap();
        let mut recorder = Recorder::new(3);
        for (index, event) in events.iter().enumerate() {
            recorder.step(&mut organism, index as u64 + 1, event);
        }
        assert_eq!(recorder.speech_cues, 5);
        let cues = recorder.finish(Millis(300_000), Some(90_000));
        let seqs: Vec<SeqNo> = cues.iter().map(|record| record.seq).collect();
        assert_eq!(seqs, [4, 5]);
        assert_eq!(cues[0].ago_ms, 60_000);
    }

    #[test]
    fn each_abstention_gets_a_plain_reason() {
        let profile = Profile::t1_ref();
        let line = |record: &CueRecord| explain(record, &profile);
        let voice = SensorEvidence {
            owner_over_other: Some(-1.8),
            owner_over_reproduced: Some(-0.4),
            owner_live: Some(-1.8),
            ..SensorEvidence::default()
        };

        assert_eq!(
            line(&record(
                name_alone(),
                abstain(Reason::Keyword, Abstention::OutOfEnergy),
                SensorEvidence::default()
            )),
            "Abstained (OutOfEnergy): the obligation budget could not pay for a thought (0.40 left, a thought costs 1.00)."
        );
        assert_eq!(
            line(&record(
                tv_line(),
                abstain(Reason::FollowUp, Abstention::OtherSpeaker),
                voice
            )),
            "Abstained (OtherSpeaker): the voice did not match whoever addressed Enton (owner-live -1.8 nats), so it could not continue the conversation."
        );
        assert_eq!(
            line(&record(
                name_alone(),
                abstain(Reason::Keyword, Abstention::OtherSpeaker),
                voice
            )),
            "Abstained (OtherSpeaker): someone else said the name (owner-live -1.8 nats) while an \"Enton...\" was still waiting for its own speaker."
        );
        assert_eq!(
            line(&record(
                SpeechCue {
                    directed: Some(0.1),
                    ..tv_line()
                },
                abstain(Reason::FollowUp, Abstention::Undirected),
                SensorEvidence {
                    addressed_over_not: Some(-2.64),
                    ..SensorEvidence::default()
                }
            )),
            "Abstained (Undirected): the directedness detector heard speech addressed to someone else, -2.6 nats."
        );
        assert_eq!(
            line(&record(
                SpeechCue {
                    vad_confidence: 0.3,
                    ..tv_line()
                },
                abstain(Reason::FollowUp, Abstention::BelowThreshold),
                SensorEvidence::default()
            )),
            "Abstained (BelowThreshold): the voice activity was too weak for a follow-up (VAD 0.30, needs 0.50)."
        );
        assert_eq!(
            line(&record(
                tv_line(),
                abstain(Reason::Speech, Abstention::Torpor),
                SensorEvidence::default()
            )),
            "Abstained (Torpor): the body was in torpor (fever or a low battery), and then only a call by name buys a thought."
        );
        assert_eq!(
            line(&record(
                tv_line(),
                abstain(Reason::Speech, Abstention::Cooldown),
                SensorEvidence::default()
            )),
            "Abstained (Cooldown): nobody called Enton, and it had thought less than 10 s before; overheard speech waits out that cooldown."
        );
    }

    #[test]
    fn a_thought_says_whether_the_cortex_answered() {
        let profile = Profile::t1_ref();
        let think = |fate| Decision::Think {
            thought: 3,
            reason: Reason::Keyword,
            salience: 1.6,
            propensity: None,
            fate,
        };
        let failed = record(
            name_alone(),
            think(Some(Fate::Failed {
                failure: Some("cortex unavailable".to_owned()),
            })),
            SensorEvidence::default(),
        );
        assert_eq!(
            explain(&failed, &profile),
            "Thought #3 (Keyword): Enton was called by name, but the thought failed: cortex unavailable."
        );
        let done = record(
            SpeechCue {
                keyword: false,
                ..name_alone()
            },
            think(Some(Fate::Done {
                reply_chars: Some(42),
            })),
            SensorEvidence::default(),
        );
        assert_eq!(
            explain(&done, &profile),
            "Thought #3 (Keyword): this finished the request that began with \"Enton...\"; the cortex replied (42 characters)."
        );
        assert_eq!(summary(&done.decision), "Think #3 (Keyword, salience 1.60)");
    }

    #[test]
    fn a_coin_flip_at_a_borderline_cue_is_named() {
        let profile = Profile::t1_ref();
        let explored = record(
            tv_line(),
            Decision::Think {
                thought: 4,
                reason: Reason::FollowUp,
                salience: 0.6,
                propensity: Some(0.05),
                fate: Some(Fate::Done {
                    reply_chars: Some(12),
                }),
            },
            SensorEvidence::default(),
        );
        assert_eq!(
            explain(&explored, &profile),
            "Thought #4 (FollowUp): a follow-up inside the conversation window; the cue was borderline, and a coin flip explored it (probability 0.05); the cortex replied (12 characters)."
        );
        let passed = record(
            tv_line(),
            Decision::Abstain {
                reason: Reason::FollowUp,
                salience: 0.6,
                why: Abstention::OtherSpeaker,
                propensity: Some(0.95),
            },
            SensorEvidence {
                owner_live: Some(-1.2),
                ..SensorEvidence::default()
            },
        );
        assert_eq!(
            explain(&passed, &profile),
            "Abstained (OtherSpeaker): the voice did not match whoever addressed Enton (owner-live -1.2 nats), so it could not continue the conversation; the cue was borderline, and a coin flip passed on it (probability 0.95)."
        );
        let json = serde_json::to_value(&passed.decision).unwrap();
        assert!((json["propensity"].as_f64().unwrap() - 0.95).abs() < 1e-6);
        let json = serde_json::to_value(abstain(Reason::Speech, Abstention::Media)).unwrap();
        assert!(json.get("propensity").is_none());
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
