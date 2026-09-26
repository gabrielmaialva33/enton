//! An acknowledgement played while Enton waits for the rest of a request (the chime
//! the runtime plays on `Attend`) is Enton's own playback: the echo model and the
//! hangover cover it like speech, and it never takes the turn from the caller.

use enton_core::{Abstention, Action, Event, Millis, Organism, Profile, Reason, SpeechCue};
use enton_core::{ThoughtId, UtteranceId};

use super::support::assert_abstention;

/// "Enton" alone: short enough that Enton waits for the rest.
fn name_alone(now: u64) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 0.9,
            duration_ms: 400,
            vad_confidence: 0.95,
            keyword: true,
            ..SpeechCue::default()
        },
    }
}

/// The rest of the request, spoken by the caller.
fn continuation(now: u64) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 0.85,
            duration_ms: 800,
            vad_confidence: 0.9,
            ..SpeechCue::default()
        },
    }
}

/// The chime as the microphone hears it: quiet, and not much like speech.
fn chime_echo(now: u64) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 0.3,
            duration_ms: 200,
            vad_confidence: 0.4,
            ..SpeechCue::default()
        },
    }
}

fn chime(organism: &mut Organism, started: u64, finished: u64) {
    assert!(
        organism
            .step(&Event::PlaybackStarted {
                now: Millis(started),
                utterance: UtteranceId(1),
            })
            .is_empty()
    );
    assert!(
        organism
            .step(&Event::PlaybackFinished {
                now: Millis(finished),
                utterance: UtteranceId(1),
                interrupted: false,
            })
            .is_empty()
    );
}

fn assert_one_keyword_thought(actions: &[Action]) {
    assert!(
        matches!(
            actions,
            [Action::Think {
                thought: ThoughtId(1),
                reason: Reason::Keyword,
                ..
            }]
        ),
        "expected the caller's request as thought 1, got {actions:?}"
    );
}

#[test]
fn the_chime_keeps_the_callers_turn_open() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    assert_eq!(
        organism.step(&name_alone(100)),
        vec![Action::Attend {
            until: Millis(5_100)
        }]
    );
    chime(&mut organism, 120, 320);
    assert!(organism.is_attending());
    assert_eq!(organism.attention_until(), Some(Millis(5_100)));

    assert!(
        organism
            .step(&Event::Tick { now: Millis(1_000) })
            .is_empty()
    );
    assert_one_keyword_thought(&organism.step(&continuation(2_500)));
    assert!(!organism.is_attending());
    // The name and its continuation were one request: the window does not fire again.
    assert!(
        organism
            .step(&Event::Tick { now: Millis(6_000) })
            .is_empty()
    );
}

#[test]
fn the_chime_heard_by_the_microphone_is_its_own_echo() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    organism.step(&name_alone(100));
    assert!(
        organism
            .step(&Event::PlaybackStarted {
                now: Millis(120),
                utterance: UtteranceId(1),
            })
            .is_empty()
    );
    assert_abstention(&organism.step(&chime_echo(250)), Abstention::SelfEcho);
    assert!(
        organism
            .step(&Event::PlaybackFinished {
                now: Millis(320),
                utterance: UtteranceId(1),
                interrupted: false,
            })
            .is_empty()
    );
    assert_abstention(&organism.step(&chime_echo(400)), Abstention::SelfEcho);
    // Still waiting for the caller, who then finishes the request.
    assert!(organism.is_attending());
    assert_one_keyword_thought(&organism.step(&continuation(2_000)));
}

#[test]
fn a_continuation_over_the_chime_finishes_the_request_once() {
    let mut organism = Organism::new(Profile::t1_ref()).unwrap();
    organism.step(&name_alone(100));
    organism.step(&Event::PlaybackStarted {
        now: Millis(120),
        utterance: UtteranceId(1),
    });
    // The caller goes on while the chime still plays, loud enough to interrupt it (a
    // quieter cue over the chime is judged as its echo, like one over speech).
    let Event::Speech { now, cue } = continuation(250) else {
        unreachable!("a continuation is speech")
    };
    let loud = Event::Speech {
        now,
        cue: SpeechCue {
            energy: 0.95,
            ..cue
        },
    };
    assert_one_keyword_thought(&organism.step(&loud));
    assert!(!organism.is_attending());
    organism.step(&Event::PlaybackFinished {
        now: Millis(260),
        utterance: UtteranceId(1),
        interrupted: true,
    });
    // One request, one thought: the window closing later asks nothing more.
    assert!(
        organism
            .step(&Event::Tick { now: Millis(6_000) })
            .is_empty()
    );
}
