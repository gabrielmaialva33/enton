//! Explicit ground truth, kept outside every controller's input.

use crate::{BENCHMARK_VERSION, Error};
use enton_core::{Event, Millis};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::File,
    io::{Read, Write},
    path::Path,
};

pub(crate) const MAX_EVENTS: usize = 50_000;
pub(crate) const MAX_TURNS: usize = 512;
const MAX_BYTES: u64 = 32 * 1024 * 1024;

/// Identity of an independently annotated request turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TurnId(pub u32);
/// Identity of one observed speech segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SegmentId(pub u32);
/// Identity of a request/conversation or distractor episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EpisodeId(pub u32);

/// Which independently specified population this tape represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TapeKind {
    /// One hundred request/conversation and one hundred distractor episodes.
    E1a,
    /// Ten commands across an hour, with fifty minutes of noise intervals.
    E1b,
    /// Small regression fixtures, never eligible for an E1 report.
    Fixture,
}

/// Semantic source annotation, never sent to a controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stimulus {
    /// A segment belonging to a request (possibly only its name prefix).
    Request(TurnId),
    /// A request intended to interrupt playback; actual overlap is measured separately.
    BargeIn(TurnId),
    /// Television speech.
    Tv,
    /// Speech addressed to another person.
    OtherSpeech,
    /// Household or general noise.
    Noise,
    /// Motor noise.
    Motor,
    /// Ventilation noise.
    Ventilation,
    /// A detector's false positive on the wake word.
    FalseKeyword,
    /// A controller's own synthetic playback leaking into its microphone.
    SelfEcho,
}

impl Stimulus {
    pub(crate) fn turn(self) -> Option<TurnId> {
        match self {
            Self::Request(id) | Self::BargeIn(id) => Some(id),
            _ => None,
        }
    }
}

/// Annotation attached to an exogenous event or a generated feedback cue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Annotation {
    /// A tick or synthetic completion, with no request attribution.
    Clock,
    /// A speech segment and its independently annotated origin.
    Speech {
        /// Unique segment identity.
        segment: SegmentId,
        /// Exogenous episode; generated self-echo has no episode.
        episode: Option<EpisodeId>,
        /// Semantic source.
        source: Stimulus,
    },
}

impl Annotation {
    pub(crate) fn source(&self) -> Option<Stimulus> {
        match self {
            Self::Clock => None,
            Self::Speech { source, .. } => Some(*source),
        }
    }
    pub(crate) fn turn(&self) -> Option<TurnId> {
        self.source().and_then(Stimulus::turn)
    }
}

/// An event at its end timestamp, paired with ground truth outside the event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// The only value supplied to a controller.
    pub event: Event,
    /// Evaluator-only ground truth.
    pub annotation: Annotation,
}

/// Descriptive classification of a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnKind {
    /// A single uninterrupted request.
    Single,
    /// A request split into a name and the request content.
    Split,
    /// A long conversation with multiple follow-up turns.
    Conversation,
    /// A short direct command/request.
    Short,
    /// A request interrupted by other speech.
    Interruption,
    /// A command (usually E1b).
    Command,
}

impl fmt::Display for TurnKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Single => "single",
            Self::Split => "split",
            Self::Conversation => "conversation",
            Self::Short => "short",
            Self::Interruption => "interruption",
            Self::Command => "command",
        })
    }
}

/// A response-required turn. Segment membership and completion are explicit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Turn {
    /// Unique turn identity.
    pub id: TurnId,
    /// Request/conversation to which this turn belongs.
    pub episode: EpisodeId,
    /// Segment IDs in chronological order, including any incomplete name prefix.
    pub segments: Vec<SegmentId>,
    /// End of the final request-bearing segment, in monotonic milliseconds.
    pub available_at: Millis,
    /// Inclusive last instant at which ignition can serve this turn.
    pub deadline: Millis,
    /// Descriptive classification ("single", "split", "conversation", "short", "interruption").
    pub kind: TurnKind,
    /// Preceding gap in milliseconds, if applicable.
    #[serde(default)]
    pub gap: Option<u32>,
}

/// A half-open interval of monotonic milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Interval {
    /// Inclusive beginning.
    pub start: Millis,
    /// Exclusive end.
    pub end: Millis,
}
impl Interval {
    pub(crate) fn contains(self, now: Millis) -> bool {
        self.start <= now && now < self.end
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct TapeData {
    version: String,
    kind: TapeKind,
    seed: u64,
    duration: Millis,
    records: Vec<Record>,
    turns: Vec<Turn>,
    noise: Vec<Interval>,
}

type SegmentIndex<'a> = BTreeMap<SegmentId, (&'a Record, Option<EpisodeId>, Stimulus)>;

/// A validated, bounded event tape. Serialized input is validated on load.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Tape(TapeData);

impl Tape {
    /// Construct a tape from independently annotated data; reject invalid identities,
    /// missing segments, nonfinite cues, backward time or exceeded bounds.
    pub fn new(
        kind: TapeKind,
        seed: u64,
        duration: Millis,
        records: Vec<Record>,
        turns: Vec<Turn>,
        noise: Vec<Interval>,
    ) -> Result<Self, Error> {
        let tape = Self(TapeData {
            version: BENCHMARK_VERSION.into(),
            kind,
            seed,
            duration,
            records,
            turns,
            noise,
        });
        tape.validate()?;
        Ok(tape)
    }
    /// Population identity.
    #[must_use]
    pub fn kind(&self) -> TapeKind {
        self.0.kind
    }
    /// Generator seed (fixtures may use zero).
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.0.seed
    }
    /// Exogenous tape duration in milliseconds.
    #[must_use]
    pub fn duration(&self) -> Millis {
        self.0.duration
    }
    /// Ordered exogenous events and ground truth.
    #[must_use]
    pub fn records(&self) -> &[Record] {
        &self.0.records
    }
    /// Explicit request turns.
    #[must_use]
    pub fn turns(&self) -> &[Turn] {
        &self.0.turns
    }
    /// Half-open E1b noise intervals.
    #[must_use]
    pub fn noise_intervals(&self) -> &[Interval] {
        &self.0.noise
    }

    /// Save a versioned JSON tape and propagate write/flush failures.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let mut file = std::io::BufWriter::new(File::create(path)?);
        serde_json::to_writer(&mut file, self)?;
        file.flush()?;
        Ok(())
    }
    /// Load at most 32 MiB; reject old versions and invalid annotations.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Error> {
        let mut bytes = Vec::new();
        File::open(path)?
            .take(MAX_BYTES + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_BYTES {
            return Err(Error::Limit("serialized tape bytes"));
        }
        let tape = Self(serde_json::from_slice(&bytes)?);
        tape.validate()?;
        Ok(tape)
    }

    fn validate(&self) -> Result<(), Error> {
        if self.0.version != BENCHMARK_VERSION {
            return Err(Error::Invalid("unsupported benchmark version".into()));
        }
        if self.0.records.len() > MAX_EVENTS
            || self.0.turns.len() > MAX_TURNS
            || self.0.noise.len() > MAX_EVENTS
        {
            return Err(Error::Limit("tape events/turns/intervals"));
        }
        if self.0.duration.0 == 0 || self.0.duration.0 > 7_200_000 {
            return Err(Error::Invalid("duration must be within two hours".into()));
        }
        let segments = self.validate_records()?;
        self.validate_turns(&segments)?;
        self.validate_population()
    }

    fn validate_records(&self) -> Result<SegmentIndex<'_>, Error> {
        let mut previous = Millis(0);
        let mut segments = BTreeMap::new();
        for record in &self.0.records {
            let now = record.event.now();
            if now < previous || now > self.0.duration {
                return Err(Error::Invalid(
                    "event timestamp outside ordered tape".into(),
                ));
            }
            previous = now;
            match (&record.event, &record.annotation) {
                (Event::Tick { .. }, Annotation::Clock) => {}
                (
                    Event::Speech { cue, .. },
                    Annotation::Speech {
                        segment,
                        episode,
                        source,
                    },
                ) => {
                    if cue.duration_ms == 0
                        || now.0 < u64::from(cue.duration_ms)
                        || !(0.0..=1.0).contains(&cue.energy)
                        || !(0.0..=1.0).contains(&cue.vad_confidence)
                    {
                        return Err(Error::Invalid(
                            "invalid speech features or segment end".into(),
                        ));
                    }
                    if segment.0 >= 50_000 || episode.is_none() || *source == Stimulus::SelfEcho {
                        return Err(Error::Invalid(
                            "exogenous speech must have an episode and cannot be self-echo".into(),
                        ));
                    }
                    if segments
                        .insert(*segment, (record, *episode, *source))
                        .is_some()
                    {
                        return Err(Error::Invalid("duplicate segment ID".into()));
                    }
                }
                _ => return Err(Error::Invalid("exogenous event/annotation mismatch".into())),
            }
        }
        Ok(segments)
    }

    fn validate_turns(&self, segments: &SegmentIndex<'_>) -> Result<(), Error> {
        let mut turns = BTreeSet::new();
        let mut claimed = BTreeSet::new();
        for turn in &self.0.turns {
            if !turns.insert(turn.id)
                || turn.segments.is_empty()
                || turn.segments.len() > 16
                || turn.deadline < turn.available_at
                || turn.deadline.0 > self.0.duration.0 + 12_000
            {
                return Err(Error::Invalid(
                    "invalid turn identity, segments or deadline".into(),
                ));
            }
            let mut last = Millis(0);
            for id in &turn.segments {
                let Some((record, episode, source)) = segments.get(id) else {
                    return Err(Error::Invalid("missing request segment".into()));
                };
                if !claimed.insert(*id)
                    || source.turn() != Some(turn.id)
                    || *episode != Some(turn.episode)
                    || record.event.now() < last
                {
                    return Err(Error::Invalid("inconsistent turn membership/order".into()));
                }
                last = record.event.now();
            }
            if last != turn.available_at {
                return Err(Error::Invalid(
                    "availability must equal final segment end".into(),
                ));
            }
        }
        for (id, (_, _, source)) in segments {
            if source.turn().is_some() && !claimed.contains(id) {
                return Err(Error::Invalid("unannotated request segment".into()));
            }
        }
        Ok(())
    }

    fn validate_population(&self) -> Result<(), Error> {
        let mut end = Millis(0);
        for interval in &self.0.noise {
            if interval.start < end
                || interval.end <= interval.start
                || interval.end > self.0.duration
            {
                return Err(Error::Invalid("invalid noise intervals".into()));
            }
            end = interval.end;
        }
        let relevant: BTreeSet<_> = self.0.turns.iter().map(|turn| turn.episode).collect();
        let distractors: BTreeSet<_> = self
            .0
            .records
            .iter()
            .filter_map(|record| match record.annotation {
                Annotation::Speech {
                    episode, source, ..
                } if source.turn().is_none() => episode,
                _ => None,
            })
            .collect();
        if !relevant.is_disjoint(&distractors) {
            return Err(Error::Invalid(
                "episode mixes request and distractor identities".into(),
            ));
        }
        if self.0.kind == TapeKind::E1a
            && (relevant.len() != 100 || distractors.len() != 100 || self.0.duration.0 != 3_600_000)
        {
            return Err(Error::Invalid(
                "E1a requires 100 request and 100 distractor episodes over one hour".into(),
            ));
        }
        if self.0.kind == TapeKind::E1b
            && (relevant.len() != 10
                || self.0.turns.len() != 10
                || self.0.duration.0 != 3_600_000
                || self
                    .0
                    .noise
                    .iter()
                    .map(|span| span.end.0.saturating_sub(span.start.0))
                    .sum::<u64>()
                    != 3_000_000)
        {
            return Err(Error::Invalid(
                "E1b requires ten commands and fifty minutes of noise over one hour".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn v2_json_roundtrip_preserves_explicit_ground_truth() {
        let tape = crate::e1a(7).unwrap();
        let path =
            std::env::temp_dir().join(format!("enton-e1-v2-tape-{}.json", std::process::id()));
        tape.save(&path).unwrap();
        assert_eq!(Tape::load(&path).unwrap(), tape);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn malformed_annotations_and_populations_are_rejected() {
        let tape = crate::e1b(7).unwrap();
        let mut bad = tape.clone();
        bad.0.version = "1".into();
        assert!(bad.validate().is_err());
        let mut bad = tape.clone();
        bad.0.turns.get_mut(0).unwrap().available_at.0 -= 1;
        assert!(bad.validate().is_err());
        let mut bad = tape.clone();
        bad.0.turns.get_mut(0).unwrap().segments.clear();
        assert!(bad.validate().is_err());
        let mut bad = tape.clone();
        bad.0.records.reverse();
        assert!(bad.validate().is_err());
        let mut bad = tape.clone();
        bad.0.noise.clear();
        assert!(bad.validate().is_err());
        let mut bad = tape.clone();
        bad.0.kind = TapeKind::E1a;
        assert!(bad.validate().is_err());
        let mut bad = tape.clone();
        bad.0.noise = vec![
            Interval {
                start: Millis(0),
                end: Millis(u64::MAX)
            };
            2
        ];
        assert!(bad.validate().is_err());
        let mut bad = tape;
        bad.0.records = vec![
            Record {
                event: Event::Tick { now: Millis(0) },
                annotation: Annotation::Clock
            };
            MAX_EVENTS + 1
        ];
        assert!(matches!(bad.validate(), Err(Error::Limit(_))));
    }
    #[test]
    fn availability_accounts_for_all_segment_durations_and_silence() {
        let tape = crate::e1a(7).unwrap();
        for turn in tape.turns().iter().filter(|t| t.segments.len() == 2) {
            let records:Vec<_>=turn.segments.iter().map(|segment|tape.records().iter().find(|r|matches!(r.annotation,Annotation::Speech{segment:id,..} if id==*segment)).unwrap()).collect();
            let first = records.first().unwrap();
            let last = records.last().unwrap();
            let Event::Speech { now, cue } = last.event else {
                panic!("request segment must be speech");
            };
            let silence = now.0 - u64::from(cue.duration_ms) - first.event.now().0;
            assert!((150..=700).contains(&silence));
            assert_eq!(turn.available_at, now);
        }
    }
}
