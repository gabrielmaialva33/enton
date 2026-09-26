//! The event loop: record each event in the soul, reduce it, act on the decisions.

use std::collections::VecDeque;
use std::io::Write;
#[cfg(feature = "voice")]
use std::sync::Arc;
use std::time::Instant;

use enton_adapters::cortex::OpenAiCortex;
#[cfg(feature = "voice")]
use enton_adapters::voice::{UtteranceId, VoicePlayer};
use enton_adapters::{MonotonicClock, SeqNo};
use enton_core::ports::{ConversationTurn, ThoughtRequest};
use enton_core::{Action, Event, Organism, ThoughtId};
use tokio::sync::mpsc;

use crate::journal::{Journal, JournalError};
use crate::tasks::spawn_cortex_think;

/// Snapshot the organism after this many recorded events (about ten minutes of ticks).
pub(crate) const SNAPSHOT_EVERY: u32 = 600;

/// How a cortex call ended; failures carry a fixed reason for the soul, never text.
pub(crate) type Outcome = Result<(), &'static str>;

pub(crate) enum LoopMessage {
    Event(Event),
    SpeechInput {
        text: String,
        event: Event,
        input_end_time: Instant,
    },
    #[cfg(feature = "voice")]
    PlaybackStarted {
        id: UtteranceId,
    },
    #[cfg(feature = "voice")]
    PlaybackFinished {
        id: UtteranceId,
    },
    #[cfg(feature = "voice")]
    PlaybackFailed {
        id: UtteranceId,
        reason: String,
    },
    CortexFinished {
        thought: ThoughtId,
        text: String,
        user_prompt: Option<String>,
        outcome: Outcome,
    },
    Quit,
}

pub(crate) fn flush_stdout() {
    // Best-effort flush for interactive CLI output; ignore failure if stdout pipe is broken.
    if let Err(_err) = std::io::stdout().flush() {
        // Ignored: consumer may have closed stdout pipe.
    }
}

pub(crate) struct RuntimeState {
    organism: Organism,
    clock: MonotonicClock,
    cortex: OpenAiCortex,
    #[cfg(feature = "voice")]
    voice_player: Option<Arc<VoicePlayer>>,
    last_transcript: Option<String>,
    attended_transcript: Option<String>,
    history: VecDeque<ConversationTurn>,
    pending_first_audio_timer: Option<Instant>,
    active_cortex_task: Option<tokio::task::JoinHandle<()>>,
    in_flight_thought: Option<ThoughtId>,
    journal: Option<Journal>,
    last_seq: Option<SeqNo>,
    unsnapshotted: u32,
}

impl RuntimeState {
    /// A runtime around a (possibly restored) organism, with no turn in flight.
    pub(crate) fn new(
        organism: Organism,
        clock: MonotonicClock,
        cortex: OpenAiCortex,
        #[cfg(feature = "voice")] voice_player: Option<Arc<VoicePlayer>>,
        journal: Option<Journal>,
    ) -> Self {
        Self {
            organism,
            clock,
            cortex,
            #[cfg(feature = "voice")]
            voice_player,
            last_transcript: None,
            attended_transcript: None,
            history: VecDeque::with_capacity(32),
            pending_first_audio_timer: None,
            active_cortex_task: None,
            in_flight_thought: None,
            journal,
            last_seq: None,
            unsnapshotted: 0,
        }
    }

    /// Write-ahead: record `event` in the soul (when there is one), then reduce it.
    async fn record_and_step(&mut self, event: &Event) -> Result<Vec<Action>, JournalError> {
        if let Some(journal) = &self.journal {
            self.last_seq = Some(journal.append(event).await?);
            self.unsnapshotted += 1;
        }
        Ok(self.organism.step(event))
    }

    async fn step_event(
        &mut self,
        event: &Event,
        tx: &mpsc::Sender<LoopMessage>,
    ) -> Result<(), JournalError> {
        let actions = self.record_and_step(event).await?;
        self.dispatch(actions, tx).await?;
        self.maybe_snapshot().await
    }

    async fn handle_speech_input(
        &mut self,
        text: String,
        event: &Event,
        input_end_time: Instant,
        tx: &mpsc::Sender<LoopMessage>,
    ) -> Result<(), JournalError> {
        let actions = self.record_and_step(event).await?;
        let is_accepted = actions
            .iter()
            .any(|a| matches!(a, Action::Think { .. } | Action::Attend { .. }));

        if is_accepted {
            self.abandon_in_flight("superseded").await?;
            #[cfg(feature = "voice")]
            if let Some(ref player) = self.voice_player {
                player.cancel();
            }

            self.pending_first_audio_timer = Some(input_end_time);
            let full_text = match self.attended_transcript.take() {
                Some(prefix) if !prefix.is_empty() => format!("{prefix} {text}"),
                _ => text,
            };
            self.last_transcript = Some(full_text);
        }

        self.dispatch(actions, tx).await?;
        self.maybe_snapshot().await
    }

    async fn handle_cortex_finished(
        &mut self,
        thought: ThoughtId,
        text: String,
        user_prompt: Option<String>,
        outcome: Outcome,
        tx: &mpsc::Sender<LoopMessage>,
    ) -> Result<(), JournalError> {
        if self.in_flight_thought != Some(thought) {
            return Ok(());
        }
        self.active_cortex_task = None;
        self.in_flight_thought = None;
        self.resolve(thought, outcome.map(|()| text.chars().count()))
            .await?;

        if let Some(user_text) = user_prompt {
            self.history.push_back(ConversationTurn::user(user_text));
        }
        self.history.push_back(ConversationTurn::assistant(&text));
        while self.history.len() > 30 {
            self.history.pop_front();
        }

        let reply = Event::CortexReply {
            now: self.clock.now(),
            thought,
            text,
        };
        self.step_event(&reply, tx).await
    }

    /// Handle one loop message; `Ok(false)` asks the loop to stop.
    async fn handle_message(
        &mut self,
        msg: LoopMessage,
        tx: &mpsc::Sender<LoopMessage>,
    ) -> Result<bool, JournalError> {
        match msg {
            LoopMessage::SpeechInput {
                text,
                event,
                input_end_time,
            } => {
                self.handle_speech_input(text, &event, input_end_time, tx)
                    .await?;
            }
            #[cfg(feature = "voice")]
            LoopMessage::PlaybackStarted { id } => {
                if let Some(start_time) = self.pending_first_audio_timer.take() {
                    let elapsed = start_time.elapsed();
                    let ms = elapsed.as_secs_f64() * 1000.0;
                    println!("[voice] Time to first audio: {ms:.1} ms (utterance {id})");
                    flush_stdout();
                }
                let event = Event::PlaybackStarted {
                    now: self.clock.now(),
                    utterance: id,
                };
                self.step_event(&event, tx).await?;
            }
            #[cfg(feature = "voice")]
            LoopMessage::PlaybackFinished { id } => {
                let event = Event::PlaybackFinished {
                    now: self.clock.now(),
                    utterance: id,
                };
                self.step_event(&event, tx).await?;
            }
            #[cfg(feature = "voice")]
            LoopMessage::PlaybackFailed { id, reason } => {
                eprintln!("[enton-voice] utterance {id} failed: {reason}");
                self.pending_first_audio_timer = None;
                let event = Event::PlaybackFinished {
                    now: self.clock.now(),
                    utterance: id,
                };
                self.step_event(&event, tx).await?;
            }
            LoopMessage::Event(event) => {
                self.step_event(&event, tx).await?;
            }
            LoopMessage::CortexFinished {
                thought,
                text,
                user_prompt,
                outcome,
            } => {
                self.handle_cortex_finished(thought, text, user_prompt, outcome, tx)
                    .await?;
            }
            LoopMessage::Quit => {
                self.abandon_in_flight("shutdown").await?;
                #[cfg(feature = "voice")]
                if let Some(ref player) = self.voice_player {
                    player.cancel();
                }
                self.attended_transcript = None;
                flush_stdout();
                return Ok(false);
            }
        }
        Ok(true)
    }

    async fn dispatch(
        &mut self,
        actions: Vec<Action>,
        tx: &mpsc::Sender<LoopMessage>,
    ) -> Result<(), JournalError> {
        for action in actions {
            println!("{action:?}");
            flush_stdout();
            match action {
                Action::Think {
                    thought, reason, ..
                } => {
                    self.abandon_in_flight("superseded").await?;
                    // Recorded before the effect, so a crash mid-call leaves a row to reconcile.
                    if let (Some(journal), Some(seq)) = (&self.journal, self.last_seq) {
                        journal.pending(thought, seq).await?;
                    }
                    self.in_flight_thought = Some(thought);

                    let user_prompt = self
                        .last_transcript
                        .take()
                        .or_else(|| self.attended_transcript.take());
                    self.attended_transcript = None;

                    let request = ThoughtRequest {
                        thought,
                        reason,
                        transcript: user_prompt,
                        history: self.history.iter().cloned().collect(),
                    };

                    self.active_cortex_task = Some(spawn_cortex_think(
                        self.cortex.clone(),
                        #[cfg(feature = "voice")]
                        self.voice_player.clone(),
                        request,
                        tx.clone(),
                    ));
                }
                Action::Attend { .. } => {
                    self.attended_transcript = self.last_transcript.take();
                }
                Action::Speak { .. } | Action::Abstain { .. } => {}
            }
        }
        Ok(())
    }

    /// Stop the in-flight cortex call, if any, and record why it never finished.
    async fn abandon_in_flight(&mut self, reason: &'static str) -> Result<(), JournalError> {
        if let Some(task) = self.active_cortex_task.take() {
            task.abort();
        }
        match self.in_flight_thought.take() {
            Some(thought) => self.resolve(thought, Err(reason)).await,
            None => Ok(()),
        }
    }

    /// Resolve a recorded thought: done with the reply length, or failed with a reason.
    async fn resolve(
        &self,
        thought: ThoughtId,
        outcome: Result<usize, &'static str>,
    ) -> Result<(), JournalError> {
        let Some(journal) = &self.journal else {
            return Ok(());
        };
        match outcome {
            Ok(chars) => {
                journal
                    .resolve(thought, true, format!(r#"{{"chars":{chars}}}"#))
                    .await
            }
            Err(reason) => {
                journal
                    .resolve(thought, false, format!(r#"{{"reason":"{reason}"}}"#))
                    .await
            }
        }
    }

    async fn maybe_snapshot(&mut self) -> Result<(), JournalError> {
        if self.unsnapshotted < SNAPSHOT_EVERY {
            return Ok(());
        }
        let (Some(journal), Some(seq)) = (&self.journal, self.last_seq) else {
            return Ok(());
        };
        journal.snapshot(seq, &self.organism).await?;
        self.unsnapshotted = 0;
        Ok(())
    }

    /// Snapshot what the log holds beyond the last snapshot, then stop the worker.
    async fn close_journal(&mut self) -> Result<(), JournalError> {
        let Some(journal) = self.journal.take() else {
            return Ok(());
        };
        let snapshot = match self.last_seq {
            Some(seq) if self.unsnapshotted > 0 => journal.snapshot(seq, &self.organism).await,
            _ => Ok(()),
        };
        let closed = journal.close().await;
        snapshot.and(closed)
    }
}

pub(crate) async fn run_event_loop(
    mut state: RuntimeState,
    mut rx: mpsc::Receiver<LoopMessage>,
    tx: mpsc::Sender<LoopMessage>,
) -> Result<(), JournalError> {
    let outcome = loop {
        let Some(msg) = rx.recv().await else {
            break Ok(());
        };
        match state.handle_message(msg, &tx).await {
            Ok(true) => {}
            Ok(false) => break Ok(()),
            Err(err) => break Err(err),
        }
    };
    flush_stdout();
    let closed = state.close_journal().await;
    outcome.and(closed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::speech_cue;
    use enton_core::{Profile, SpeechCue};

    /// A typed line with the given timing and loudness (energy and VAD alike).
    fn typed(text: &str, now: u64, level: f32, duration_ms: u32, keyword: bool) -> LoopMessage {
        LoopMessage::SpeechInput {
            text: text.to_string(),
            event: Event::Speech {
                now: enton_core::Millis(now),
                cue: SpeechCue {
                    energy: level,
                    vad_confidence: level,
                    duration_ms,
                    keyword,
                    speaker_sim: None,
                    media: None,
                    turn_complete: None,
                    directed: None,
                    direction: None,
                },
            },
            input_end_time: Instant::now(),
        }
    }

    #[tokio::test]
    async fn rejected_cues_leave_transcripts_intact() {
        let (tx, _rx) = mpsc::channel(1);
        let mut state = test_state(Organism::new(Profile::t1_ref()).unwrap(), None);

        // 1. Keyword, shorter than 900 ms -> accepted, Attend until later.
        state
            .handle_message(typed("enton", 100, 0.9, 500, true), &tx)
            .await
            .unwrap();

        assert_eq!(state.last_transcript.as_deref(), None);
        assert_eq!(state.attended_transcript.as_deref(), Some("enton")); // Attend sets attended_transcript

        // 2. Rejected noise (low VAD) -> Abstain
        state
            .handle_message(typed("shhh", 200, 0.1, 100, false), &tx)
            .await
            .unwrap();

        // State is intact
        assert_eq!(
            state.attended_transcript.as_deref(),
            Some("enton"),
            "rejected cue left attended transcript intact"
        );
        assert_eq!(state.last_transcript.as_deref(), None);

        // 3. Continuation (high VAD) -> accepted -> Think
        state
            .handle_message(typed("help me", 300, 0.9, 1000, false), &tx)
            .await
            .unwrap();

        assert_eq!(state.last_transcript.as_deref(), None);
        assert_eq!(
            state.attended_transcript, None,
            "Think clears attended_transcript"
        );

        // 4. Accept a Thought, then reject another noise.
        // The Think above set in_flight_thought and active_cortex_task
        assert!(state.in_flight_thought.is_some());

        state
            .handle_message(typed("cough", 400, 0.1, 100, false), &tx)
            .await
            .unwrap();

        assert!(
            state.in_flight_thought.is_some(),
            "rejected cue did not abort flight"
        );
    }

    #[test]
    fn normal_shutdown_returns_and_drops_runtime_resources() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "runtime::tests::shutdown_child",
                "--ignored",
                "--nocapture",
            ])
            .env("ENTON_SHUTDOWN_TEST_CHILD", "1")
            .output()
            .unwrap();
        assert!(output.status.success());
        assert!(
            String::from_utf8(output.stdout)
                .unwrap()
                .contains("runtime resources dropped")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    #[ignore = "invoked only by the shutdown regression in an isolated test process"]
    async fn shutdown_child() {
        assert_eq!(std::env::var("ENTON_SHUTDOWN_TEST_CHILD").unwrap(), "1");
        #[cfg(feature = "voice")]
        let player = Arc::new(VoicePlayer::mock());
        #[cfg(feature = "voice")]
        let weak = Arc::downgrade(&player);

        let (tx, rx) = mpsc::channel(1);
        tx.send(LoopMessage::Quit).await.unwrap();
        let state = test_state(Organism::new(Profile::t1_ref()).unwrap(), None);
        #[cfg(feature = "voice")]
        let state = RuntimeState {
            voice_player: Some(player),
            ..state
        };
        run_event_loop(state, rx, tx).await.unwrap();
        #[cfg(feature = "voice")]
        assert!(weak.upgrade().is_none());
        println!("runtime resources dropped");
    }

    fn test_state(organism: Organism, journal: Option<Journal>) -> RuntimeState {
        RuntimeState::new(
            organism,
            MonotonicClock::new(),
            OpenAiCortex::with_endpoint("http://127.0.0.1:9", "unused"),
            #[cfg(feature = "voice")]
            None,
            journal,
        )
    }

    fn addressed_request() -> LoopMessage {
        let text = "Enton, que horas são?";
        LoopMessage::SpeechInput {
            text: text.to_string(),
            event: Event::Speech {
                now: enton_core::Millis(1_000),
                cue: speech_cue(text),
            },
            input_end_time: Instant::now(),
        }
    }

    #[tokio::test]
    async fn a_crash_mid_thought_is_restored_and_the_thought_abandoned() {
        let dir = std::env::temp_dir().join(format!("enton-bin-crash-{}", std::process::id()));
        let path = dir.join("soul.sqlite");
        let profile = Profile::t1_ref();
        let (journal, restored) = Journal::open(&path, &profile).unwrap();
        assert!(restored.abandoned.is_empty());
        let (tx, _rx) = mpsc::channel(8);
        let mut state = test_state(restored.organism, Some(journal));

        state
            .handle_message(addressed_request(), &tx)
            .await
            .unwrap();
        let thought = state.in_flight_thought.unwrap();
        let before = state.organism.clone();

        // Crash: no Quit, no final snapshot, the thought never resolves.
        state.active_cortex_task.take().unwrap().abort();
        state.journal.take().unwrap().close().await.unwrap();

        let (_journal, restored) = Journal::open(&path, &profile).unwrap();
        assert_eq!(restored.organism, before);
        assert_eq!(restored.organism.last_seen(), enton_core::Millis(1_000));
        assert_eq!(restored.abandoned, vec![thought]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn a_clean_shutdown_resolves_the_thought_and_snapshots() {
        let dir = std::env::temp_dir().join(format!("enton-bin-quit-{}", std::process::id()));
        let path = dir.join("soul.sqlite");
        let profile = Profile::t1_ref();
        let (journal, restored) = Journal::open(&path, &profile).unwrap();
        let (tx, _rx) = mpsc::channel(8);
        let mut state = test_state(restored.organism, Some(journal));

        state
            .handle_message(addressed_request(), &tx)
            .await
            .unwrap();
        assert!(!state.handle_message(LoopMessage::Quit, &tx).await.unwrap());
        let before = state.organism.clone();
        state.close_journal().await.unwrap();

        let (_journal, restored) = Journal::open(&path, &profile).unwrap();
        assert_eq!(restored.organism, before);
        assert!(restored.abandoned.is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[cfg(feature = "voice")]
    #[tokio::test]
    async fn voice_playback_lifecycle_events_forward_to_core() {
        let (tx, _rx) = mpsc::channel(64);
        let mut state = test_state(Organism::new(Profile::t1_ref()).unwrap(), None);

        // 1. PlaybackStarted forwards to core and updates playback status to speaking
        assert!(!state.organism.is_speaking());
        state
            .handle_message(LoopMessage::PlaybackStarted { id: UtteranceId(1) }, &tx)
            .await
            .unwrap();
        assert!(state.organism.is_speaking());

        // 2. PlaybackFinished forwards to core and transitions from speaking to hangover
        state
            .handle_message(LoopMessage::PlaybackFinished { id: UtteranceId(1) }, &tx)
            .await
            .unwrap();
        assert!(!state.organism.is_speaking());
        assert!(state.organism.is_hangover());

        // 3. PlaybackStarted again with utterance 2
        state
            .handle_message(LoopMessage::PlaybackStarted { id: UtteranceId(2) }, &tx)
            .await
            .unwrap();
        assert!(state.organism.is_speaking());

        // 4. PlaybackFailed forwards to core as finish so the watchdog is not waited on
        state
            .handle_message(
                LoopMessage::PlaybackFailed {
                    id: UtteranceId(2),
                    reason: "DAC buffer underrun".into(),
                },
                &tx,
            )
            .await
            .unwrap();
        assert!(!state.organism.is_speaking());
        assert!(state.organism.is_hangover());
    }
}
