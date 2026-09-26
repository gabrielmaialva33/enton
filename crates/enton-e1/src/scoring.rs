//! One causal attribution per paid thought; no heuristic reconstruction of turns.
use crate::tape::TurnKind;
use crate::{Annotation, Record, SegmentId, Tape, Turn, TurnId};
use enton_core::{Action, Event, Millis, Reason};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stratum {
    Kind(TurnKind),
    Gap(u32),
}

impl Serialize for Stratum {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Kind(k) => serializer.collect_str(&format_args!("kind:{k}")),
            Self::Gap(g) => serializer.collect_str(&format_args!("gap:{g}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct Tally {
    pub served: u64,
    pub total: u64,
}

/// Observable event that caused a paid thought.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Trigger {
    /// The current speech segment (including irrelevant speech).
    Segment(SegmentId),
    /// A keyword timeout for this exact previously attended segment.
    AttendTimeout(SegmentId),
    /// Internal drive or another unassociated event.
    Internal,
}
/// Each thought serves at most one turn; all other calls count as waste.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Credit {
    /// First timely paid thought for the complete request.
    Served(TurnId),
    /// Another paid thought for an already served turn, also waste.
    Duplicate(TurnId),
    /// Early, late, irrelevant or unassociated work.
    Waste,
}
struct Pending {
    segment: SegmentId,
    turn: Option<TurnId>,
    until: Millis,
}
pub(crate) struct Scorer<'a> {
    turns: BTreeMap<TurnId, &'a Turn>,
    pub(crate) served: BTreeSet<TurnId>,
    pending: Option<Pending>,
}
impl<'a> Scorer<'a> {
    pub(crate) fn new(tape: &'a Tape) -> Self {
        Self {
            turns: tape.turns().iter().map(|t| (t.id, t)).collect(),
            served: BTreeSet::new(),
            pending: None,
        }
    }
    pub(crate) fn attend(&mut self, record: &Record, until: Millis) {
        self.pending = match record.annotation {
            Annotation::Speech { segment, .. } => Some(Pending {
                segment,
                turn: record.annotation.turn(),
                until,
            }),
            Annotation::Clock => None,
        };
    }
    pub(crate) fn think(
        &mut self,
        record: &Record,
        reason: &Reason,
        attending_after: bool,
    ) -> (Trigger, Credit) {
        let now = record.event.now();
        let (trigger, candidate) = match record.annotation {
            Annotation::Speech { segment, .. } if !matches!(reason, Reason::Drive(_)) => {
                (Trigger::Segment(segment), record.annotation.turn())
            }
            Annotation::Clock
                if matches!(record.event, Event::Tick { .. })
                    && *reason == Reason::Keyword
                    && !attending_after =>
            {
                match self.pending.take() {
                    Some(p) if now >= p.until => {
                        // Waiting until availability does not make an incomplete name
                        // prefix contain a subsequently rejected/missing request.
                        let complete = p.turn.filter(|id| {
                            self.turns
                                .get(id)
                                .is_some_and(|turn| turn.segments.last() == Some(&p.segment))
                        });
                        (Trigger::AttendTimeout(p.segment), complete)
                    }
                    _ => (Trigger::Internal, None),
                }
            }
            _ => (Trigger::Internal, None),
        };
        let credit = if let Some(id) = candidate {
            if let Some(turn) = self.turns.get(&id) {
                if now < turn.available_at || now > turn.deadline {
                    Credit::Waste
                } else if self.served.insert(id) {
                    Credit::Served(id)
                } else {
                    Credit::Duplicate(id)
                }
            } else {
                Credit::Waste
            }
        } else {
            Credit::Waste
        };
        (trigger, credit)
    }
    pub(crate) fn finish_event(&mut self, now: Millis, attending_after: bool) {
        if !attending_after
            || self.pending.as_ref().is_some_and(|p| {
                now >= p.until
                    || p.turn
                        .and_then(|id| self.turns.get(&id))
                        .is_some_and(|t| now > t.deadline)
            })
        {
            self.pending = None;
        }
    }
    pub(crate) fn served_requests(&self) -> usize {
        let mut episodes: BTreeMap<_, bool> = BTreeMap::new();
        for turn in self.turns.values() {
            *episodes.entry(turn.episode).or_insert(true) &= self.served.contains(&turn.id);
        }
        episodes.values().filter(|served| **served).count()
    }

    pub(crate) fn strata(&self) -> BTreeMap<Stratum, Tally> {
        let mut result: BTreeMap<Stratum, Tally> = BTreeMap::new();
        for turn in self.turns.values() {
            let is_served = self.served.contains(&turn.id);
            let s = u64::from(is_served);

            // by kind
            let entry = result.entry(Stratum::Kind(turn.kind)).or_default();
            entry.served += s;
            entry.total += 1;

            // by gap
            if let Some(gap) = turn.gap {
                let entry = result.entry(Stratum::Gap(gap)).or_default();
                entry.served += s;
                entry.total += 1;
            }
        }
        result
    }
    pub(crate) fn observe_non_think(&mut self, record: &Record, action: &Action) {
        if let Action::Attend { until } = action {
            self.attend(record, *until);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tape::TurnKind;

    #[test]
    fn strata_tallies_correctly_by_kind_and_gap() {
        let turns = [
            Turn {
                id: TurnId(1),
                episode: EpisodeId(1),
                segments: vec![SegmentId(1)],
                available_at: Millis(1000),
                deadline: Millis(5000),
                kind: TurnKind::Single,
                gap: None,
            },
            Turn {
                id: TurnId(2),
                episode: EpisodeId(1),
                segments: vec![SegmentId(2)],
                available_at: Millis(2000),
                deadline: Millis(6000),
                kind: TurnKind::Conversation,
                gap: Some(3000),
            },
            Turn {
                id: TurnId(3),
                episode: EpisodeId(2),
                segments: vec![SegmentId(3)],
                available_at: Millis(3000),
                deadline: Millis(7000),
                kind: TurnKind::Conversation,
                gap: Some(3000),
            },
        ];

        // We will simulate serving Turn 1 and Turn 2, missing Turn 3.
        let s = Scorer {
            turns: turns.iter().map(|t| (t.id, t)).collect(),
            served: [TurnId(1), TurnId(2)].into_iter().collect(),
            pending: None,
        };

        let strata = s.strata();

        // Turn 1: Single, no gap -> served
        // Turn 2: Conversation, gap 3000 -> served
        // Turn 3: Conversation, gap 3000 -> not served

        assert_eq!(strata[&Stratum::Kind(TurnKind::Single)].served, 1);
        assert_eq!(strata[&Stratum::Kind(TurnKind::Single)].total, 1);

        assert_eq!(strata[&Stratum::Kind(TurnKind::Conversation)].served, 1);
        assert_eq!(strata[&Stratum::Kind(TurnKind::Conversation)].total, 2);

        assert_eq!(strata[&Stratum::Gap(3000)].served, 1);
        assert_eq!(strata[&Stratum::Gap(3000)].total, 2);
    }

    use super::*;
    use crate::{EpisodeId, Stimulus, TapeKind};
    use enton_core::SpeechCue;
    fn speech(segment: u32, end: u64, source: Stimulus) -> Record {
        Record {
            event: Event::Speech {
                now: Millis(end),
                cue: SpeechCue {
                    energy: 0.9,
                    vad_confidence: 0.9,
                    duration_ms: 250,
                    keyword: true,
                },
            },
            annotation: Annotation::Speech {
                segment: SegmentId(segment),
                episode: Some(EpisodeId(if source.turn().is_some() { 1 } else { 2 })),
                source,
            },
        }
    }
    fn fixture() -> Tape {
        Tape::new(
            TapeKind::Fixture,
            0,
            Millis(20_000),
            vec![
                speech(0, 1000, Stimulus::Request(TurnId(1))),
                speech(1, 2000, Stimulus::Request(TurnId(1))),
                speech(2, 10_000, Stimulus::Request(TurnId(2))),
            ],
            vec![
                Turn {
                    id: TurnId(1),
                    episode: EpisodeId(1),
                    segments: vec![SegmentId(0), SegmentId(1)],
                    available_at: Millis(2000),
                    deadline: Millis(12_000),
                    kind: crate::tape::TurnKind::Single,
                    gap: None,
                },
                Turn {
                    id: TurnId(2),
                    episode: EpisodeId(1),
                    segments: vec![SegmentId(2)],
                    available_at: Millis(10_000),
                    deadline: Millis(20_000),
                    kind: crate::tape::TurnKind::Single,
                    gap: None,
                },
            ],
            vec![],
        )
        .unwrap()
    }
    fn tick(now: u64) -> Record {
        Record {
            event: Event::Tick { now: Millis(now) },
            annotation: Annotation::Clock,
        }
    }
    #[test]
    fn split_name_cannot_serve_request_and_each_thought_has_one_credit() {
        let tape = fixture();
        let mut s = Scorer::new(&tape);
        assert_eq!(
            s.think(
                &speech(0, 1000, Stimulus::Request(TurnId(1))),
                &Reason::Keyword,
                false
            )
            .1,
            Credit::Waste
        );
        let complete = speech(1, 2000, Stimulus::Request(TurnId(1)));
        assert_eq!(
            s.think(&complete, &Reason::Speech, false).1,
            Credit::Served(TurnId(1))
        );
        assert_eq!(
            s.think(&complete, &Reason::Speech, false).1,
            Credit::Duplicate(TurnId(1))
        );
        assert_eq!(s.served_requests(), 0); // The second response-required turn is missing.
    }
    #[test]
    fn interrupted_attend_never_credits_later_noise() {
        let tape = fixture();
        let mut s = Scorer::new(&tape);
        s.attend(&speech(0, 1000, Stimulus::Request(TurnId(1))), Millis(6000));
        let noise = speech(3, 1500, Stimulus::Noise);
        assert_eq!(s.think(&noise, &Reason::Keyword, false).1, Credit::Waste);
        s.finish_event(Millis(1500), false);
        assert_eq!(
            s.think(&tick(7000), &Reason::Keyword, false),
            (Trigger::Internal, Credit::Waste)
        );
        assert!(s.served.is_empty());
    }
    #[test]
    fn replaced_attend_uses_only_new_complete_turn() {
        let tape = fixture();
        let mut s = Scorer::new(&tape);
        s.attend(&speech(0, 1000, Stimulus::Request(TurnId(1))), Millis(6000));
        s.attend(
            &speech(2, 10_000, Stimulus::Request(TurnId(2))),
            Millis(15_000),
        );
        assert_eq!(
            s.think(&tick(15_000), &Reason::Keyword, false),
            (
                Trigger::AttendTimeout(SegmentId(2)),
                Credit::Served(TurnId(2))
            )
        );
        assert!(!s.served.contains(&TurnId(1)));
    }
    #[test]
    fn failed_and_expired_attends_are_terminal() {
        let tape = fixture();
        for (now, still_attending) in [(11_000, false), (15_000, true)] {
            let mut s = Scorer::new(&tape);
            s.attend(
                &speech(2, 10_000, Stimulus::Request(TurnId(2))),
                Millis(15_000),
            );
            // false models the core consuming pending state on OutOfEnergy;
            // true without a timeout thought at the deadline is expiry.
            s.finish_event(Millis(now), still_attending);
            assert_eq!(
                s.think(&tick(16_000), &Reason::Keyword, false),
                (Trigger::Internal, Credit::Waste)
            );
        }
    }
    #[test]
    fn rejected_noise_preserves_complete_attend_but_not_incomplete_content() {
        let tape = fixture();
        let mut s = Scorer::new(&tape);
        s.attend(
            &speech(2, 10_000, Stimulus::Request(TurnId(2))),
            Millis(15_000),
        );
        s.finish_event(Millis(11_000), true);
        assert_eq!(
            s.think(&tick(15_000), &Reason::Keyword, false).1,
            Credit::Served(TurnId(2))
        );
        let mut s = Scorer::new(&tape);
        s.attend(&speech(0, 1000, Stimulus::Request(TurnId(1))), Millis(6000));
        s.finish_event(Millis(2000), true); // The request-bearing cue was rejected.
        assert_eq!(
            s.think(&tick(6000), &Reason::Keyword, false).1,
            Credit::Waste
        );
    }
    #[test]
    fn late_or_drive_thoughts_do_not_serve_a_pending_request() {
        let tape = fixture();
        let mut s = Scorer::new(&tape);
        let complete = speech(2, 10_000, Stimulus::Request(TurnId(2)));
        assert_eq!(
            s.think(&complete, &Reason::Drive("social".into()), false).1,
            Credit::Waste
        );
        s.attend(&complete, Millis(25_000));
        assert_eq!(
            s.think(&tick(25_000), &Reason::Keyword, false).1,
            Credit::Waste
        );
    }
}
