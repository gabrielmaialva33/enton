//! Event-driven runtime scenarios; current-thread tasks are cancelled before yielding,
//! so no real inference, network connection or audio device is used.
use super::*;
use enton_core::{Millis, Profile, SpeechCue};

fn state() -> RuntimeState {
    RuntimeState::new(
        Organism::new(Profile::t1_ref()).unwrap(),
        MonotonicClock::new(),
        OpenAiCortex::with_endpoint("http://127.0.0.1:9", "never-run"),
        #[cfg(feature = "voice")]
        None,
        None,
    )
}
fn input(text: &str, now: u64, duration_ms: u32, keyword: bool, vad: f32) -> LoopMessage {
    LoopMessage::SpeechInput {
        text: text.to_owned(),
        event: Event::Speech {
            now: Millis(now),
            cue: SpeechCue {
                energy: vad,
                vad_confidence: vad,
                duration_ms,
                keyword,
                ..SpeechCue::default()
            },
        },
        input_end_time: Instant::now(),
    }
}
fn stop_call(state: &mut RuntimeState) {
    if let Some(task) = state.active_cortex_task.take() {
        task.abort();
    }
}
async fn send(state: &mut RuntimeState, tx: &mpsc::Sender<LoopMessage>, message: LoopMessage) {
    state.handle_message(message, tx).await.unwrap();
    stop_call(state);
}

#[tokio::test(flavor = "current_thread")]
async fn quiet_commands_discard_the_abandoned_partial_transcript() {
    for until in [200, 60_000] {
        let (tx, _rx) = mpsc::channel(8);
        let mut state = state();
        send(
            &mut state,
            &tx,
            input("abandoned request prefix", 100, 300, true, 1.0),
        )
        .await;
        assert!(state.organism.is_attending());
        assert_eq!(
            state.attended_transcript.as_deref(),
            Some("abandoned request prefix")
        );
        send(
            &mut state,
            &tx,
            LoopMessage::Event(Event::Quiet {
                now: Millis(200),
                until: Millis(until),
            }),
        )
        .await;
        assert!(!state.organism.is_attending());
        assert_eq!(
            state.attended_transcript, None,
            "quiet left abandoned words for the next request"
        );
        assert!(state.pending_first_audio_timer.is_none());
        send(
            &mut state,
            &tx,
            input("fresh request prefix", 300, 300, true, 1.0),
        )
        .await;
        assert_eq!(
            state.attended_transcript.as_deref(),
            Some("fresh request prefix")
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn quiet_hours_do_not_discard_an_explicit_pending_request() {
    let (tx, _rx) = mpsc::channel(8);
    let mut state = state();
    send(&mut state, &tx, input("Enton", 100, 300, true, 1.0)).await;
    send(
        &mut state,
        &tx,
        LoopMessage::Event(Event::QuietHours {
            now: Millis(200),
            active: true,
        }),
    )
    .await;
    assert!(state.organism.is_attending());
    assert_eq!(state.attended_transcript.as_deref(), Some("Enton"));
}

#[tokio::test(flavor = "current_thread")]
async fn rejected_noise_keeps_the_prefix_and_timeout_dispatches_once() {
    let (tx, _rx) = mpsc::channel(8);
    let mut state = state();
    send(&mut state, &tx, input("Enton", 100, 300, true, 1.0)).await;
    send(&mut state, &tx, input("noise", 200, 100, false, 0.01)).await;
    assert_eq!(state.attended_transcript.as_deref(), Some("Enton"));
    send(
        &mut state,
        &tx,
        LoopMessage::Event(Event::Tick { now: Millis(5_100) }),
    )
    .await;
    assert_eq!(state.in_flight_thought, Some(ThoughtId(1)));
    assert!(state.attended_transcript.is_none());
    send(
        &mut state,
        &tx,
        LoopMessage::Event(Event::Tick { now: Millis(6_100) }),
    )
    .await;
    assert_eq!(state.in_flight_thought, Some(ThoughtId(1)));
}

#[tokio::test(flavor = "current_thread")]
async fn a_late_success_cannot_pollute_the_new_requests_history() {
    let (tx, _rx) = mpsc::channel(8);
    let mut state = state();
    send(
        &mut state,
        &tx,
        input("first request", 1_000, 1_500, true, 1.0),
    )
    .await;
    send(
        &mut state,
        &tx,
        input("second request", 2_000, 1_500, true, 1.0),
    )
    .await;
    send(
        &mut state,
        &tx,
        LoopMessage::CortexFinished {
            thought: ThoughtId(1),
            text: "stale answer".into(),
            user_prompt: Some("first request".into()),
            outcome: Ok(()),
        },
    )
    .await;
    assert_eq!(state.in_flight_thought, Some(ThoughtId(2)));
    assert_eq!(state.conversation.len(), 0);
    send(
        &mut state,
        &tx,
        LoopMessage::CortexFinished {
            thought: ThoughtId(2),
            text: "current answer".into(),
            user_prompt: Some("second request".into()),
            outcome: Ok(()),
        },
    )
    .await;
    assert_eq!(state.in_flight_thought, None);
    assert_eq!(state.conversation.len(), 2);
    assert_eq!(state.organism.cortex_failures(), 0);
}

#[tokio::test(flavor = "current_thread")]
async fn failure_keeps_the_users_request_but_does_not_invent_an_answer() {
    let (tx, _rx) = mpsc::channel(8);
    let mut state = state();
    send(&mut state, &tx, input("request", 1_000, 1_500, true, 1.0)).await;
    send(
        &mut state,
        &tx,
        LoopMessage::CortexFinished {
            thought: ThoughtId(1),
            text: String::new(),
            user_prompt: Some("request".into()),
            outcome: Err("simulated failure"),
        },
    )
    .await;
    assert_eq!(state.conversation.len(), 1);
    assert_eq!(state.organism.cortex_failures(), 1);
    assert_eq!(state.in_flight_thought, None);
}
