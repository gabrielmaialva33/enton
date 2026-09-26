//! Fixture tapes for Enton's own initiative, which E1's generated tapes never exercise:
//! they carry no quiet command and no quiet hours, and no drive of t1-ref gets ready
//! within a tape. With drives made eager, one tape has a drive ready while the owner
//! talks, whose intent rides the owner's follow-up; another has the owner ask for quiet.
//! Each is run against a control tape that lacks what it exercises.
// Test fixtures may fail loudly; the quality bar permits unwrap in tests.
#![allow(clippy::unwrap_used)]

use enton_core::{Event, Millis, Profile, Reason, SpeechCue};
use enton_e1::run::PaidThought;
use enton_e1::tape::TurnKind;
use enton_e1::{
    Annotation, EpisodeId, Record, RoomCondition, SegmentId, Sensors, Stimulus, Tape, TapeKind,
    Turn, TurnId, run_tape_with,
};

/// Forty minutes: long enough for eager drives to get ready (about 25 minutes of
/// pressure, 31 once the owner's replies have eased the social drive).
const DURATION: u64 = 2_400_000;

/// t1-ref with drives that reach their threshold in about 25 minutes.
fn eager() -> Profile {
    let mut profile = Profile::t1_ref();
    profile.ignition.threshold = 0.01;
    profile.ignition.hysteresis = 0.002;
    profile.ignition.ema_alpha = 1.0;
    profile
}

/// A clear 1.5 s segment ending at `now`, Enton's name in it or not.
fn said(now: u64, keyword: bool) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 0.9,
            vad_confidence: 0.9,
            duration_ms: 1_500,
            keyword,
            ..SpeechCue::default()
        },
    }
}

/// What happens in a fixture besides the clock.
enum Happening {
    /// The owner asks something: a request turn of its own, in `episode`.
    Request {
        now: u64,
        keyword: bool,
        episode: u32,
    },
    /// Someone else talks, not to Enton.
    Overheard { now: u64 },
    /// The owner's quiet command, until `until` (a release when not after `now`).
    Quiet { now: u64, until: u64 },
    /// The quiet hours begin or end.
    QuietHours { now: u64, active: bool },
}

impl Happening {
    fn at(&self) -> u64 {
        match self {
            Self::Request { now, .. }
            | Self::Overheard { now }
            | Self::Quiet { now, .. }
            | Self::QuietHours { now, .. } => *now,
        }
    }
}

/// A fixture tape: a tick every second for [`DURATION`], and `happenings` (at instants
/// between ticks) with their annotations and turns.
fn tape(happenings: &[Happening]) -> Tape {
    let mut records = Vec::new();
    let mut turns = Vec::new();
    let mut pending = happenings.iter().peekable();
    let mut segment = 0;
    for second in 0..=DURATION / 1_000 {
        let now = second * 1_000;
        records.push(Record {
            event: Event::Tick { now: Millis(now) },
            annotation: Annotation::Clock,
        });
        while let Some(happening) = pending.next_if(|happening| happening.at() < now + 1_000) {
            let record = match *happening {
                Happening::Request {
                    now,
                    keyword,
                    episode,
                } => {
                    let id = TurnId(segment);
                    turns.push(Turn {
                        id,
                        episode: EpisodeId(episode),
                        segments: vec![SegmentId(segment)],
                        available_at: Millis(now),
                        deadline: Millis(now + 10_000),
                        kind: if keyword {
                            TurnKind::Single
                        } else {
                            TurnKind::Conversation
                        },
                        gap: None,
                        pause_style: None,
                    });
                    Record {
                        event: said(now, keyword),
                        annotation: Annotation::Speech {
                            segment: SegmentId(segment),
                            episode: Some(EpisodeId(episode)),
                            source: Stimulus::Request(id),
                            pause_style: None,
                        },
                    }
                }
                Happening::Overheard { now } => Record {
                    event: said(now, false),
                    annotation: Annotation::Speech {
                        segment: SegmentId(segment),
                        episode: Some(EpisodeId(100 + segment)),
                        source: Stimulus::OtherSpeech,
                        pause_style: None,
                    },
                },
                Happening::Quiet { now, until } => Record {
                    event: Event::Quiet {
                        now: Millis(now),
                        until: Millis(until),
                    },
                    annotation: Annotation::Clock,
                },
                Happening::QuietHours { now, active } => Record {
                    event: Event::QuietHours {
                        now: Millis(now),
                        active,
                    },
                    annotation: Annotation::Clock,
                },
            };
            if matches!(record.annotation, Annotation::Speech { .. }) {
                segment += 1;
            }
            records.push(record);
        }
    }
    let blocks = usize::try_from(DURATION.div_ceil(180_000)).unwrap();
    Tape::new(
        TapeKind::Fixture,
        0,
        Millis(DURATION),
        records,
        turns,
        vec![],
        vec![RoomCondition::default(); blocks],
    )
    .unwrap()
}

/// The organism's paid thoughts on `tape`, with eager drives, and how many turns it served.
fn organism(tape: &Tape) -> (Vec<PaidThought>, u64) {
    let run = run_tape_with(tape, &eager(), Sensors::DEFAULT).unwrap();
    (run.organism.thoughts, run.organism.served_turns)
}

/// Each paid thought as `(id, reason, rider)`.
fn summary(thoughts: &[PaidThought]) -> Vec<(u64, Reason, Option<&str>)> {
    thoughts
        .iter()
        .map(|thought| {
            (
                thought.thought.0,
                thought.reason.clone(),
                thought.rider.as_deref(),
            )
        })
        .collect()
}

fn curiosity() -> Reason {
    Reason::Drive("curiosity".to_owned())
}

// ---------------------------------------------------------------------------
// A deferred intent riding a conversation

/// Nobody is home when the drives get ready, about 25 minutes in; at 30.5 minutes the
/// owner calls Enton, and the ready drive holds its intent while they talk. With
/// `follow_up`, the owner asks something more inside the window.
fn owner_comes_home(follow_up: bool) -> Tape {
    let mut happenings = vec![Happening::Request {
        now: 1_830_500,
        keyword: true,
        episode: 0,
    }];
    if follow_up {
        happenings.push(Happening::Request {
            now: 1_835_500,
            keyword: false,
            episode: 0,
        });
    }
    tape(&happenings)
}

#[test]
fn a_drive_ready_while_the_owner_talks_rides_their_follow_up() {
    let (thoughts, served) = organism(&owner_comes_home(true));
    // The follow-up is answered, and its answer carries the drive's intent: the drive
    // never buys a thought of its own.
    assert_eq!(
        summary(&thoughts),
        vec![
            (1, Reason::Keyword, None),
            (2, Reason::FollowUp, Some("curiosity")),
        ]
    );
    assert_eq!(served, 2);
}

#[test]
fn without_a_follow_up_the_drive_pays_for_a_thought_of_its_own() {
    let (thoughts, served) = organism(&owner_comes_home(false));
    // The same call the follow-up cost above now buys the drive's own thought, after
    // the conversation: a ride gets the owner's answer and the drive's turn for one.
    let summary = summary(&thoughts);
    assert_eq!(
        summary,
        vec![(1, Reason::Keyword, None), (2, curiosity(), None)]
    );
    let drive = thoughts.last().unwrap();
    assert!(drive.at > Millis(1_840_500), "{:?}", drive.at);
    assert_eq!(served, 1);
}

// ---------------------------------------------------------------------------
// Quiet mode and quiet hours

/// The owner calls Enton a minute in, someone talks nearby at two minutes, and the owner
/// asks something at ten; the drives get ready about half an hour in. `quiet` is what
/// holds Enton's own initiative back: the owner's command at 70 s and its release at 35
/// minutes, or the quiet hours over the same span, or nothing.
fn evening(quiet: Option<bool>) -> Tape {
    let mut happenings = vec![
        Happening::Request {
            now: 60_500,
            keyword: true,
            episode: 0,
        },
        Happening::Overheard { now: 120_500 },
        Happening::Request {
            now: 600_500,
            keyword: true,
            episode: 1,
        },
    ];
    match quiet {
        Some(true) => {
            happenings.insert(
                1,
                Happening::Quiet {
                    now: 70_000,
                    until: 70_000 + 3_600_000,
                },
            );
            happenings.push(Happening::Quiet {
                now: 2_100_000,
                until: 2_100_000,
            });
        }
        Some(false) => {
            happenings.insert(
                1,
                Happening::QuietHours {
                    now: 70_000,
                    active: true,
                },
            );
            happenings.push(Happening::QuietHours {
                now: 2_100_000,
                active: false,
            });
        }
        None => {}
    }
    tape(&happenings)
}

#[test]
fn quiet_mode_holds_back_enton_s_own_thoughts_and_answers_every_call() {
    let (control, served) = organism(&evening(None));
    // Without quiet, the overheard talk buys a thought, and so does the drive once ready.
    let control = summary(&control);
    assert_eq!(
        control,
        vec![
            (1, Reason::Keyword, None),
            (2, Reason::Speech, None),
            (3, Reason::Keyword, None),
            (4, curiosity(), None),
        ]
    );
    assert_eq!(served, 2);

    for quiet in [true, false] {
        let (thoughts, served) = organism(&evening(Some(quiet)));
        // The command itself buys nothing; the overheard talk and the drive wait; both
        // of the owner's calls are answered; the drive thinks once released.
        assert_eq!(
            summary(&thoughts),
            vec![
                (1, Reason::Keyword, None),
                (2, Reason::Keyword, None),
                (3, curiosity(), None),
            ],
            "quiet mode: {quiet}"
        );
        let drive = thoughts.last().unwrap();
        assert!(
            (Millis(2_100_000)..=Millis(2_101_000)).contains(&drive.at),
            "{:?}",
            drive.at
        );
        assert_eq!(served, 2);
    }
}
