use std::collections::VecDeque;
use std::path::Path;

use enton_adapters::soul::{PersonaRecord, PersonaSource};
use enton_adapters::{ActionStatus, SeqNo, Soul, SoulConfig, soul};
use enton_core::{
    Action, Budget, Event, Millis, Organism, PlaybackStatus, Profile, Reason, Senses, SpeechCue,
    ThoughtId,
};
use serde::{Deserialize, Serialize};

use super::WhyError;
use super::explain::explain;

/// Events read from the soul per query.
const PAGE: usize = 1024;

/// Everything `enton why` reports.
#[derive(Debug, Serialize)]
pub(super) struct Report {
    /// The log that was replayed.
    pub(super) soul: String,
    /// The profile whose reducer and calibration replayed it.
    pub(super) profile: String,
    /// The snapshot replay started from, when older events were pruned.
    pub(super) from_snapshot: Option<SeqNo>,
    /// How many events were replayed.
    pub(super) events: u64,
    /// Sequence number of the first replayed event.
    pub(super) first_seq: Option<SeqNo>,
    /// Sequence number of the last replayed event.
    pub(super) last_seq: Option<SeqNo>,
    /// Monotonic time of the last recorded event; `ago_ms` counts back from it.
    pub(super) last_event_ms: u64,
    /// How many speech cues were replayed.
    pub(super) speech_cues: u64,
    /// Every persona a recorded thought was asked with, in the order they were
    /// first used.
    pub(super) personas: Vec<PersonaShown>,
    /// The cues shown, oldest first.
    pub(super) cues: Vec<CueRecord>,
    /// The `--since` window, if one was given.
    pub(super) since_ms: Option<u64>,
}

/// One speech cue, what was decided and why.
#[derive(Debug, Clone, Serialize)]
pub(super) struct CueRecord {
    pub(super) seq: SeqNo,
    pub(super) at_ms: u64,
    /// Milliseconds on Enton's clock between this cue and the last recorded event.
    pub(super) ago_ms: u64,
    /// The cue as stored: measurements only, never words.
    pub(super) cue: SpeechCue,
    pub(super) decision: Decision,
    pub(super) evidence: SensorEvidence,
    pub(super) context: Context,
    /// For an `Attend`: how the wait for the rest of the request ended.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) then: Option<Then>,
    /// Whether Enton, speaking, stopped for this cue: an utterance of its was cut off
    /// right after it.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub(super) cut_off: bool,
    pub(super) explanation: String,
}

/// A decision, as `enton why` shows it.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(super) enum Decision {
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
        /// The persona the thought was asked with; `None` when the soul holds
        /// none for it (the thought predates persona records, or was never
        /// recorded).
        #[serde(skip_serializing_if = "Option::is_none")]
        persona: Option<Box<PersonaShown>>,
        /// Set when the thought before it was asked with another persona.
        #[serde(skip_serializing_if = "Option::is_none")]
        persona_changed: Option<Box<PersonaChange>>,
    },
    /// Enton waited for the rest of a request.
    Attend { until_ms: u64 },
    /// The cortex was not woken.
    Abstain {
        reason: Reason,
        salience: f32,
        why: enton_core::Abstention,
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
pub(super) enum Fate {
    /// The cortex replied; only the reply's length is stored.
    Done { reply_chars: Option<u64> },
    /// The thought failed, with the reason the runtime recorded.
    Failed { failure: Option<String> },
    /// Never resolved: still running, or cut off by a crash not yet reconciled.
    Pending,
    /// The soul holds no record of the thought.
    Unrecorded,
}

/// A persona, as the soul recorded it: never its text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct PersonaShown {
    /// SHA-256 of the persona's bytes, in hex; for a file, what `sha256sum` prints.
    pub(super) sha256: String,
    /// The first digits of that hash, as the startup line shows them.
    #[serde(skip)]
    pub(super) short: String,
    /// The length of those bytes.
    pub(super) bytes: u64,
    pub(super) source: PersonaOrigin,
    /// The event the first thought asked with it was decided at.
    pub(super) first_seq: SeqNo,
}

/// Whether a persona was the built-in default or a file the owner wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum PersonaOrigin {
    BuiltIn,
    File,
}

impl From<PersonaRecord> for PersonaShown {
    fn from(record: PersonaRecord) -> Self {
        let digest = record.digest;
        Self {
            sha256: digest.hex(),
            short: digest.short_hex(),
            bytes: digest.bytes,
            source: match digest.source {
                PersonaSource::BuiltIn => PersonaOrigin::BuiltIn,
                PersonaSource::File => PersonaOrigin::File,
            },
            first_seq: record.first_seq,
        }
    }
}

/// The persona of the thought before, when it differs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct PersonaChange {
    /// The thought before.
    pub(super) thought: u64,
    /// The persona it was asked with.
    pub(super) persona: PersonaShown,
}

/// How an `Attend` ended.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(super) enum Then {
    /// The window closed with nothing more, and the name alone was decided on.
    Timeout { at_ms: u64, decision: Decision },
    /// A later cue carried the turn on.
    Continued { at_ms: u64, seq: SeqNo },
}

/// The calibrated evidence of one cue, in nats, only for sensors that ran.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub(super) struct SensorEvidence {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) owner_over_other: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) owner_over_reproduced: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) live_over_reproduced: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) finished_over_unfinished: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) addressed_over_not: Option<f32>,
    /// The owner speaking live, over the likelier alternative.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) owner_live: Option<f32>,
}

impl SensorEvidence {
    pub(super) fn read(senses: &Senses, cue: &SpeechCue) -> Self {
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
pub(super) struct Context {
    /// Time left in the attention window open to any voice, if one was open.
    pub(super) attention_left_ms: Option<u64>,
    /// Time left in the longer window open only to the verified voice (or to
    /// speech clearly addressed to Enton), if one was open.
    pub(super) verified_attention_left_ms: Option<u64>,
    /// Whether an "Enton..." was waiting for the rest of its request.
    pub(super) name_pending: bool,
    /// Belief, from zero to one, that a TV was playing, as the cue was weighed.
    pub(super) tv_presence: f32,
    /// How long that belief had been at or above the profile's TV-on level, or
    /// `None` while the TV counted as off.
    pub(super) tv_on_for_ms: Option<u64>,
    /// Whether the body was in torpor (fever or a critical battery).
    pub(super) torpor: bool,
    /// Whether Enton's own voice, or its echo, overlapped the cue.
    pub(super) enton_speaking: bool,
    /// What was left in the account for addressed turns.
    pub(super) obligation_budget: BudgetLeft,
    /// What was left in the account for overheard speech and drives.
    pub(super) discretionary_budget: BudgetLeft,
    /// What one thought costs.
    pub(super) think_cost: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub(super) struct BudgetLeft {
    pub(super) available: f32,
    pub(super) capacity: f32,
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
            persona: None,
            persona_changed: None,
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
    /// The last cue that took the turn (a thought or a wait): the runtime stops Enton's
    /// voice for such a cue, so an utterance cut off after it was cut for it.
    took_turn: Option<SeqNo>,
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
            took_turn: None,
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
            if let Event::PlaybackFinished {
                interrupted: true, ..
            } = event
            {
                self.cut_off_by(self.took_turn);
            }
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
        if matches!(decision, Decision::Think { .. } | Decision::Attend { .. }) {
            self.took_turn = Some(seq);
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
            cut_off: false,
            explanation: String::new(),
        });
        if self.cues.len() > self.keep {
            self.cues.pop_front();
        }
    }

    /// Record that the cue at `seq`, if it is still kept, cut Enton off.
    fn cut_off_by(&mut self, seq: Option<SeqNo>) {
        if let Some(record) = self.cues.iter_mut().find(|record| Some(record.seq) == seq) {
            record.cut_off = true;
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
pub(super) fn audit(
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
    let soul = Soul::open_read_only(path, SoulConfig::default()).map_err(|err| match err {
        soul::Error::UnsupportedSchema { found, expected } if found < expected => {
            WhyError::OldSchema {
                path: path.to_path_buf(),
                found,
                expected,
            }
        }
        other => failed(other),
    })?;
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
        resolve(&soul, &mut record.decision).map_err(failed)?;
        if let Some(Then::Timeout { decision, .. }) = &mut record.then {
            resolve(&soul, decision).map_err(failed)?;
        }
        record.explanation = explain(record, profile);
    }
    let personas = soul
        .personas()
        .map_err(failed)?
        .into_iter()
        .map(PersonaShown::from)
        .collect();

    Ok(Report {
        soul: path.display().to_string(),
        profile: profile.name.clone(),
        from_snapshot: (start > 0).then_some(start),
        events,
        first_seq,
        last_seq,
        last_event_ms: last_event.0,
        speech_cues,
        personas,
        cues,
        since_ms,
    })
}

/// Fill in what the soul recorded about a thought: how it resolved, the
/// persona it was asked with, and whether that persona differs from the one
/// the thought before was asked with.
fn resolve(soul: &Soul, decision: &mut Decision) -> Result<(), soul::Error> {
    let Decision::Think {
        thought,
        fate,
        persona,
        persona_changed,
        ..
    } = decision
    else {
        return Ok(());
    };
    *fate = Some(fate_of(soul, *thought)?);
    let Some(record) = soul.thought_persona(ThoughtId(*thought))? else {
        return Ok(());
    };
    *persona_changed = match soul.thought_before(ThoughtId(*thought))? {
        Some((before, Some(previous))) if previous.digest != record.digest => {
            Some(Box::new(PersonaChange {
                thought: before.0,
                persona: previous.into(),
            }))
        }
        _ => None,
    };
    *persona = Some(Box::new(record.into()));
    Ok(())
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

    fn recorder_counts(events: &[Event]) -> (u64, u64) {
        replay(events).map_or((0, 0), |(recorder, _)| {
            (recorder.events, recorder.speech_cues)
        })
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
                why: enton_core::Abstention::Media,
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

    #[test]
    fn the_cue_that_cut_enton_off_says_so() {
        let request = SpeechCue {
            energy: 0.9,
            duration_ms: 1_500,
            vad_confidence: 0.95,
            keyword: true,
            ..SpeechCue::default()
        };
        let finished = |now: u64, utterance: u64, interrupted: bool| Event::PlaybackFinished {
            now: Millis(now),
            utterance: enton_core::UtteranceId(utterance),
            interrupted,
        };
        let events = [
            speech(1_000, request),
            Event::PlaybackStarted {
                now: Millis(3_000),
                utterance: enton_core::UtteranceId(1),
            },
            finished(4_000, 1, false),
            Event::PlaybackStarted {
                now: Millis(4_010),
                utterance: enton_core::UtteranceId(2),
            },
            // The owner calls again over the second sentence: it is cut, and so is the
            // third, which never started.
            speech(5_000, request),
            finished(5_010, 2, true),
            finished(5_010, 3, true),
        ];
        let (recorder, organism) = replay(&events).unwrap();
        let cues = recorder.finish(organism.last_seen(), None);
        assert!(!cues[0].cut_off);
        assert!(cues[1].context.enton_speaking);
        assert!(matches!(
            cues[1].decision,
            Decision::Think { thought: 2, .. }
        ));
        assert!(cues[1].cut_off);
        let json = serde_json::to_value(&cues[1]).unwrap();
        assert_eq!(json["cut_off"], serde_json::Value::Bool(true));
        assert!(
            serde_json::to_value(&cues[0])
                .unwrap()
                .get("cut_off")
                .is_none()
        );
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
}
