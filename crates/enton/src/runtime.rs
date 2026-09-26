//! The event loop: record each event in the soul, reduce it, act on the decisions.

use std::io::Write;
#[cfg(feature = "voice")]
use std::sync::Arc;
use std::time::Instant;

use enton_adapters::checklist::Checklist;
use enton_adapters::cortex::OpenAiCortex;
use enton_adapters::initiative::with_ride;
#[cfg(feature = "voice")]
use enton_adapters::voice::{UtteranceId, VoicePlayer};
use enton_adapters::{MonotonicClock, SeqNo};
use enton_core::ports::{ConversationTurn, ThoughtRequest};
use enton_core::{Action, Event, Organism, Reason, ThoughtId};
use tokio::sync::mpsc;

use crate::conversation::Conversation;
#[cfg(feature = "voice")]
use crate::conversation::Heard;
use crate::journal::{Journal, JournalError};
#[cfg(feature = "voice")]
use crate::tasks::user_turn;
use crate::tasks::{drive_prompt, spawn_cortex_think};

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
    /// An utterance ended: played to its end, or cut off (`interrupted`).
    #[cfg(feature = "voice")]
    PlaybackFinished {
        id: UtteranceId,
        interrupted: bool,
    },
    #[cfg(feature = "voice")]
    PlaybackFailed {
        id: UtteranceId,
        reason: String,
    },
    /// A sentence of `thought`'s reply went to the voice, as `utterance` (`None` when
    /// it had nothing to say out loud or the player refused it).
    #[cfg(feature = "voice")]
    Sentence {
        thought: ThoughtId,
        text: String,
        utterance: Option<UtteranceId>,
    },
    CortexFinished {
        thought: ThoughtId,
        text: String,
        user_prompt: Option<String>,
        outcome: Outcome,
    },
    /// The owner's checklist was read: at startup, and whenever it changed.
    Checklist(Checklist),
    Quit,
}

pub(crate) fn flush_stdout() {
    // Best-effort flush for interactive CLI output; ignore failure if stdout pipe is broken.
    if let Err(_err) = std::io::stdout().flush() {
        // Ignored: consumer may have closed stdout pipe.
    }
}

/// Enton's voice: the player, and whether it chimes while it waits for the rest of a
/// request.
#[cfg(feature = "voice")]
pub(crate) struct Voice {
    pub(crate) player: Arc<VoicePlayer>,
    /// Play the acknowledgement chime on `Attend`.
    pub(crate) chime: bool,
}

pub(crate) struct RuntimeState {
    organism: Organism,
    clock: MonotonicClock,
    cortex: OpenAiCortex,
    #[cfg(feature = "voice")]
    voice: Option<Voice>,
    /// What the owner heard of the replies spoken aloud.
    #[cfg(feature = "voice")]
    heard: Heard,
    /// The last acknowledgement chime, to tell its playback from speech.
    #[cfg(feature = "voice")]
    chime: Option<UtteranceId>,
    last_transcript: Option<String>,
    attended_transcript: Option<String>,
    conversation: Conversation,
    pending_first_audio_timer: Option<Instant>,
    active_cortex_task: Option<tokio::task::JoinHandle<()>>,
    in_flight_thought: Option<ThoughtId>,
    journal: Option<Journal>,
    last_seq: Option<SeqNo>,
    unsnapshotted: u32,
    /// The checklist's text while it holds something to check: what a drive thought
    /// brings to the cortex. The core only ever learns whether there is one.
    checklist: Option<String>,
}

impl RuntimeState {
    /// A runtime around a (possibly restored) organism, with no turn in flight.
    pub(crate) fn new(
        organism: Organism,
        clock: MonotonicClock,
        cortex: OpenAiCortex,
        #[cfg(feature = "voice")] voice: Option<Voice>,
        journal: Option<Journal>,
    ) -> Self {
        Self {
            organism,
            clock,
            cortex,
            #[cfg(feature = "voice")]
            voice,
            #[cfg(feature = "voice")]
            heard: Heard::default(),
            #[cfg(feature = "voice")]
            chime: None,
            last_transcript: None,
            attended_transcript: None,
            conversation: Conversation::default(),
            pending_first_audio_timer: None,
            active_cortex_task: None,
            in_flight_thought: None,
            journal,
            last_seq: None,
            unsnapshotted: 0,
            checklist: None,
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
            #[cfg(feature = "voice")]
            self.cut_off();
            self.abandon_in_flight("superseded").await?;

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
        // A thought superseded or cut off by a shutdown was abandoned (and resolved) when
        // it was dropped: its late end is no outcome, so the core never hears of it.
        if self.in_flight_thought != Some(thought) {
            return Ok(());
        }
        self.active_cortex_task = None;
        self.in_flight_thought = None;
        self.resolve(thought, outcome.map(|()| text.chars().count()))
            .await?;

        if let Some(user_text) = user_prompt {
            self.conversation.push(ConversationTurn::user(user_text));
        }
        let said = !text.trim().is_empty();
        if said {
            // Whole, as the cortex wrote it; a spoken reply the owner cuts off is later
            // cut down to what they heard.
            self.conversation.push(ConversationTurn::assistant(&text));
        }
        #[cfg(feature = "voice")]
        self.heard
            .finished(thought, said.then(|| self.conversation.newest()).flatten());

        let now = self.clock.now();
        let outcome = if outcome.is_ok() {
            Event::CortexReply { now, thought, text }
        } else {
            // A real failure: the core counts it and backs off discretionary thoughts.
            Event::CortexFailed { now, thought }
        };
        self.step_event(&outcome, tx).await
    }

    /// Keep the checklist's text for drive thoughts, and tell the core only whether it
    /// holds something to check.
    async fn handle_checklist(
        &mut self,
        checklist: Checklist,
        tx: &mpsc::Sender<LoopMessage>,
    ) -> Result<(), JournalError> {
        let actionable = checklist.is_actionable();
        self.checklist = checklist.text().map(str::to_owned);
        let event = Event::Checklist {
            now: self.clock.now(),
            actionable,
        };
        self.step_event(&event, tx).await
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
                if self.chime == Some(id) {
                    // The acknowledgement is not the answer: time it, and keep the
                    // answer's timer running.
                    if let Some(start_time) = self.pending_first_audio_timer {
                        let ms = start_time.elapsed().as_secs_f64() * 1000.0;
                        println!("[voice] Chime {ms:.1} ms after the name (utterance {id})");
                        flush_stdout();
                    }
                } else if let Some(start_time) = self.pending_first_audio_timer.take() {
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
            LoopMessage::PlaybackFinished { id, interrupted } => {
                if !interrupted {
                    self.heard.ended(id, true);
                }
                let event = Event::PlaybackFinished {
                    now: self.clock.now(),
                    utterance: id,
                    interrupted,
                };
                self.step_event(&event, tx).await?;
            }
            #[cfg(feature = "voice")]
            LoopMessage::PlaybackFailed { id, reason } => {
                eprintln!("[enton-voice] utterance {id} failed: {reason}");
                self.pending_first_audio_timer = None;
                self.heard.ended(id, false);
                // Not cut off: it failed, which the log above explains.
                let event = Event::PlaybackFinished {
                    now: self.clock.now(),
                    utterance: id,
                    interrupted: false,
                };
                self.step_event(&event, tx).await?;
            }
            #[cfg(feature = "voice")]
            LoopMessage::Sentence {
                thought,
                text,
                utterance,
            } => self.heard.sentence(thought, text, utterance),
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
            LoopMessage::Checklist(checklist) => {
                self.handle_checklist(checklist, tx).await?;
            }
            LoopMessage::Quit => {
                self.abandon_in_flight("shutdown").await?;
                #[cfg(feature = "voice")]
                if let Some(voice) = &self.voice {
                    voice.player.cancel();
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
                    thought,
                    reason,
                    rider,
                    ..
                } => {
                    self.abandon_in_flight("superseded").await?;
                    // Recorded before the effect, so a crash mid-call leaves a row to reconcile.
                    if let (Some(journal), Some(seq)) = (&self.journal, self.last_seq) {
                        journal.pending(thought, seq).await?;
                    }
                    self.in_flight_thought = Some(thought);

                    let transcript = if matches!(reason, Reason::Drive(_)) {
                        // A drive brings up what is on the checklist, never speech heard
                        // earlier, which stays with the turn it belongs to.
                        self.checklist
                            .as_deref()
                            .map(|checklist| drive_prompt(&reason, checklist))
                    } else {
                        let heard = self
                            .last_transcript
                            .take()
                            .or_else(|| self.attended_transcript.take());
                        self.attended_transcript = None;
                        heard
                    };
                    // A drive's deferred intent riding this answer adds one line to it.
                    let transcript =
                        with_ride(transcript, rider.as_deref(), self.checklist.as_deref());

                    let request = ThoughtRequest {
                        thought,
                        reason,
                        transcript,
                        history: self.conversation.turns(),
                    };

                    #[cfg(feature = "voice")]
                    if self.voice.is_some() {
                        self.heard.begin(thought, user_turn(&request));
                    }
                    self.active_cortex_task = Some(spawn_cortex_think(
                        self.cortex.clone(),
                        #[cfg(feature = "voice")]
                        self.voice.as_ref().map(|voice| Arc::clone(&voice.player)),
                        request,
                        tx.clone(),
                    ));
                }
                Action::Attend { .. } => {
                    self.attended_transcript = self.last_transcript.take();
                    #[cfg(feature = "voice")]
                    self.acknowledge();
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
            Some(thought) => {
                #[cfg(feature = "voice")]
                self.heard.forget(thought);
                self.resolve(thought, Err(reason)).await
            }
            None => Ok(()),
        }
    }

    /// The owner took the turn: stop speaking, and keep in the conversation only what
    /// they heard of each reply cut off.
    #[cfg(feature = "voice")]
    fn cut_off(&mut self) {
        let Some(voice) = &self.voice else {
            return;
        };
        voice.player.cancel();
        let player = &voice.player;
        self.heard.cut(&mut self.conversation, |id| {
            player
                .stage_timings(id)
                .is_some_and(|timings| timings.playback_finished.is_some())
        });
    }

    /// Enton heard its name and waits for the rest: chime, so the owner knows it listens.
    #[cfg(feature = "voice")]
    fn acknowledge(&mut self) {
        let Some(voice) = self.voice.as_ref().filter(|voice| voice.chime) else {
            return;
        };
        match voice.player.chime() {
            Ok(id) => self.chime = Some(id),
            Err(err) => eprintln!("[enton-voice] chime not played: {err}"),
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
            voice: Some(Voice {
                player,
                chime: true,
            }),
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

    /// The persona of [`test_state`]'s cortex, which has the default configuration.
    fn built_in_persona() -> enton_adapters::soul::PersonaDigest {
        (&enton_adapters::cortex::Persona::built_in()).into()
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
        let (journal, restored) = Journal::open(&path, &profile, built_in_persona()).unwrap();
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

        let (_journal, restored) = Journal::open(&path, &profile, built_in_persona()).unwrap();
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
        let (journal, restored) = Journal::open(&path, &profile, built_in_persona()).unwrap();
        let (tx, _rx) = mpsc::channel(8);
        let mut state = test_state(restored.organism, Some(journal));

        state
            .handle_message(addressed_request(), &tx)
            .await
            .unwrap();
        assert!(!state.handle_message(LoopMessage::Quit, &tx).await.unwrap());
        let before = state.organism.clone();
        state.close_journal().await.unwrap();

        let (_journal, restored) = Journal::open(&path, &profile, built_in_persona()).unwrap();
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
            .handle_message(
                LoopMessage::PlaybackFinished {
                    id: UtteranceId(1),
                    interrupted: false,
                },
                &tx,
            )
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

/// What reaches the core when a thought ends: a reply, a real failure, or nothing at all
/// for a thought that was dropped; and what a drive thought brings to the cortex.
#[cfg(test)]
mod outcome_tests {
    use super::*;
    use crate::tasks::speech_cue;
    use enton_core::{Millis, Profile};

    fn state(profile: Profile) -> RuntimeState {
        RuntimeState::new(
            Organism::new(profile).unwrap(),
            MonotonicClock::new(),
            OpenAiCortex::with_endpoint("http://127.0.0.1:9", "unused"),
            #[cfg(feature = "voice")]
            None,
            None,
        )
    }

    fn typed(text: &str, now: u64) -> LoopMessage {
        LoopMessage::SpeechInput {
            text: text.to_owned(),
            event: Event::Speech {
                now: Millis(now),
                cue: speech_cue(text),
            },
            input_end_time: Instant::now(),
        }
    }

    fn finished(thought: u64, text: &str, outcome: Outcome) -> LoopMessage {
        LoopMessage::CortexFinished {
            thought: ThoughtId(thought),
            text: text.to_owned(),
            user_prompt: None,
            outcome,
        }
    }

    /// Stop the real call the dispatch started, so only the test's outcomes arrive.
    fn stop_the_call(state: &mut RuntimeState) {
        if let Some(task) = state.active_cortex_task.take() {
            task.abort();
        }
    }

    #[tokio::test]
    async fn a_real_failure_reaches_the_core_and_backs_off() {
        let (tx, _rx) = mpsc::channel(8);
        let mut state = state(Profile::t1_ref());
        state
            .handle_message(typed("Enton, que horas são?", 1_000), &tx)
            .await
            .unwrap();
        assert_eq!(state.in_flight_thought, Some(ThoughtId(1)));
        stop_the_call(&mut state);
        state
            .handle_message(
                finished(1, "(cortex offline in scaffold)", Err("cortex unavailable")),
                &tx,
            )
            .await
            .unwrap();
        assert_eq!(state.organism.cortex_failures(), 1);
        assert!(state.organism.backoff_until().is_some());
        assert_eq!(state.organism.conversation_thought(), None);
    }

    #[tokio::test]
    async fn a_superseded_or_shut_down_thought_is_no_failure() {
        let (tx, _rx) = mpsc::channel(8);
        let mut state = state(Profile::t1_ref());
        state
            .handle_message(typed("Enton, que horas são?", 1_000), &tx)
            .await
            .unwrap();
        stop_the_call(&mut state);
        // The owner asks again before the answer: the first thought is dropped.
        state
            .handle_message(typed("Enton, e amanhã, vai chover?", 3_000), &tx)
            .await
            .unwrap();
        assert_eq!(state.in_flight_thought, Some(ThoughtId(2)));
        stop_the_call(&mut state);
        // Its late end, even a failed one, is no outcome.
        state
            .handle_message(finished(1, "", Err("cortex unavailable")), &tx)
            .await
            .unwrap();
        assert_eq!(state.organism.cortex_failures(), 0);
        // The newer thought's reply is one, and nothing failed.
        state
            .handle_message(finished(2, "Não deve chover.", Ok(())), &tx)
            .await
            .unwrap();
        assert_eq!(state.organism.cortex_failures(), 0);
        assert_eq!(state.in_flight_thought, None);

        // A thought cut off by a shutdown did not fail either.
        state
            .handle_message(typed("Enton, conta uma piada.", 20_000), &tx)
            .await
            .unwrap();
        stop_the_call(&mut state);
        assert!(!state.handle_message(LoopMessage::Quit, &tx).await.unwrap());
        assert_eq!(state.organism.cortex_failures(), 0);
    }

    #[tokio::test]
    async fn the_checklist_reaches_the_core_as_a_flag_and_a_drive_as_its_prompt() {
        let (tx, _rx) = mpsc::channel(8);
        let mut profile = Profile::t1_ref();
        profile.ignition.threshold = 0.01;
        profile.ignition.hysteresis = 0.002;
        profile.ignition.ema_alpha = 1.0;
        let mut state = state(profile);
        state
            .handle_message(LoopMessage::Checklist(Checklist::Empty), &tx)
            .await
            .unwrap();
        assert!(!state.organism.checklist_actionable());
        assert_eq!(state.checklist, None);
        let text = "# Hoje\n- [ ] regar as plantas\n";
        state
            .handle_message(
                LoopMessage::Checklist(Checklist::Actionable(text.to_owned())),
                &tx,
            )
            .await
            .unwrap();
        assert!(state.organism.checklist_actionable());
        assert_eq!(state.checklist.as_deref(), Some(text));

        // The owner is home: they asked something, and the answer came.
        state
            .handle_message(typed("Enton, bom dia!", 1_000), &tx)
            .await
            .unwrap();
        stop_the_call(&mut state);
        state
            .handle_message(finished(1, "Bom dia!", Ok(())), &tx)
            .await
            .unwrap();
        let history = state.conversation.len();

        // Half an hour later a drive asks, with the checklist as its prompt.
        state
            .handle_message(
                LoopMessage::Event(Event::Tick {
                    now: Millis(1_600_000),
                }),
                &tx,
            )
            .await
            .unwrap();
        assert_eq!(state.in_flight_thought, Some(ThoughtId(2)));
        assert_eq!(
            state.organism.drive_thought(),
            Some((ThoughtId(2), "curiosity"))
        );
        stop_the_call(&mut state);
        // Nothing needed saying: silence answers the drive, fails nothing, and leaves the
        // conversation history as it was.
        state
            .handle_message(finished(2, "", Ok(())), &tx)
            .await
            .unwrap();
        assert_eq!(state.organism.drive_thought(), None);
        assert_eq!(state.organism.cortex_failures(), 0);
        assert_eq!(state.conversation.len(), history);
    }
}

/// What the conversation remembers of a reply: all of it in text mode; spoken, only what
/// the owner heard before cutting Enton off. And the chime Enton plays while it waits.
#[cfg(test)]
mod heard_tests {
    use super::*;
    use crate::tasks::speech_cue;
    use enton_core::ports::TurnRole;
    use enton_core::{Millis, Profile};

    const ASKED: &str = "Enton, conta uma história.";
    #[cfg(feature = "voice")]
    const REPLY: [&str; 2] = ["Era uma vez um robô.", "Ele morava num PC."];
    const WHOLE: &str = "Era uma vez um robô. Ele morava num PC.";
    /// Loud and addressed by name: it interrupts Enton mid-sentence.
    const INTERRUPTION: &str = "Enton, para, que horas são?";

    fn state() -> RuntimeState {
        RuntimeState::new(
            Organism::new(Profile::t1_ref()).unwrap(),
            MonotonicClock::new(),
            OpenAiCortex::with_endpoint("http://127.0.0.1:9", "unused"),
            #[cfg(feature = "voice")]
            None,
            None,
        )
    }

    fn typed(text: &str, now: u64) -> LoopMessage {
        LoopMessage::SpeechInput {
            text: text.to_owned(),
            event: Event::Speech {
                now: Millis(now),
                cue: speech_cue(text),
            },
            input_end_time: Instant::now(),
        }
    }

    fn replied(thought: u64, text: &str, asked: &str) -> LoopMessage {
        LoopMessage::CortexFinished {
            thought: ThoughtId(thought),
            text: text.to_owned(),
            user_prompt: Some(asked.to_owned()),
            outcome: Ok(()),
        }
    }

    /// Stop the real call the dispatch started, so only the test's messages arrive.
    fn stop_the_call(state: &mut RuntimeState) {
        if let Some(task) = state.active_cortex_task.take() {
            task.abort();
        }
    }

    fn turns(state: &RuntimeState) -> Vec<(TurnRole, String)> {
        state
            .conversation
            .turns()
            .into_iter()
            .map(|turn| (turn.role, turn.content))
            .collect()
    }

    async fn send(state: &mut RuntimeState, tx: &mpsc::Sender<LoopMessage>, msg: LoopMessage) {
        assert!(state.handle_message(msg, tx).await.unwrap());
    }

    #[tokio::test]
    async fn text_mode_remembers_the_whole_reply() {
        let (tx, _rx) = mpsc::channel(8);
        let mut state = state();
        send(&mut state, &tx, typed(ASKED, 1_000)).await;
        stop_the_call(&mut state);
        send(&mut state, &tx, replied(1, WHOLE, ASKED)).await;
        send(&mut state, &tx, typed(INTERRUPTION, 3_000)).await;
        stop_the_call(&mut state);
        assert_eq!(
            turns(&state),
            [
                (TurnRole::User, ASKED.to_owned()),
                (TurnRole::Assistant, WHOLE.to_owned()),
            ]
        );
    }

    #[cfg(feature = "voice")]
    fn voiced(chime: bool) -> RuntimeState {
        RuntimeState {
            voice: Some(Voice {
                player: Arc::new(VoicePlayer::mock()),
                chime,
            }),
            ..state()
        }
    }

    /// Utterance IDs far above the mock player's own, which starts at one.
    #[cfg(feature = "voice")]
    fn utterance(n: u64) -> UtteranceId {
        UtteranceId(1_000 + n)
    }

    #[cfg(feature = "voice")]
    fn sentence(thought: u64, n: usize, id: u64) -> LoopMessage {
        LoopMessage::Sentence {
            thought: ThoughtId(thought),
            text: REPLY[n].to_owned(),
            utterance: Some(utterance(id)),
        }
    }

    #[cfg(feature = "voice")]
    fn played(id: u64, interrupted: bool) -> LoopMessage {
        LoopMessage::PlaybackFinished {
            id: utterance(id),
            interrupted,
        }
    }

    #[cfg(feature = "voice")]
    #[tokio::test]
    async fn a_reply_cut_off_is_remembered_as_far_as_it_was_heard() {
        let (tx, _rx) = mpsc::channel(8);
        let mut state = voiced(false);
        send(&mut state, &tx, typed(ASKED, 1_000)).await;
        stop_the_call(&mut state);
        send(&mut state, &tx, sentence(1, 0, 1)).await;
        send(&mut state, &tx, sentence(1, 1, 2)).await;
        send(&mut state, &tx, replied(1, WHOLE, ASKED)).await;
        let started = |id| LoopMessage::PlaybackStarted { id: utterance(id) };
        send(&mut state, &tx, started(1)).await;
        send(&mut state, &tx, played(1, false)).await;
        send(&mut state, &tx, started(2)).await;
        assert!(state.organism.is_speaking());

        // The owner talks over the second sentence and takes the turn.
        send(&mut state, &tx, typed(INTERRUPTION, 3_000)).await;
        assert_eq!(state.in_flight_thought, Some(ThoughtId(2)));
        stop_the_call(&mut state);
        let heard = "Era uma vez um robô. [interrupted: the owner heard only this]";
        assert_eq!(
            turns(&state),
            [
                (TurnRole::User, ASKED.to_owned()),
                (TurnRole::Assistant, heard.to_owned()),
            ]
        );
        // The player reports the cut; the conversation stays as the owner heard it.
        send(&mut state, &tx, played(2, true)).await;
        assert_eq!(turns(&state).len(), 2);
    }

    #[cfg(feature = "voice")]
    #[tokio::test]
    async fn a_reply_heard_in_full_is_remembered_whole() {
        let (tx, _rx) = mpsc::channel(8);
        let mut state = voiced(false);
        send(&mut state, &tx, typed(ASKED, 1_000)).await;
        stop_the_call(&mut state);
        send(&mut state, &tx, sentence(1, 0, 1)).await;
        send(&mut state, &tx, sentence(1, 1, 2)).await;
        send(&mut state, &tx, replied(1, WHOLE, ASKED)).await;
        send(&mut state, &tx, played(1, false)).await;
        send(&mut state, &tx, played(2, false)).await;
        send(&mut state, &tx, typed(INTERRUPTION, 9_000)).await;
        stop_the_call(&mut state);
        assert_eq!(
            turns(&state),
            [
                (TurnRole::User, ASKED.to_owned()),
                (TurnRole::Assistant, WHOLE.to_owned()),
            ]
        );
    }

    #[cfg(feature = "voice")]
    #[tokio::test]
    async fn a_reply_cut_off_while_still_being_written_keeps_the_owners_words() {
        let (tx, _rx) = mpsc::channel(8);
        let mut state = voiced(false);
        send(&mut state, &tx, typed(ASKED, 1_000)).await;
        stop_the_call(&mut state);
        send(&mut state, &tx, sentence(1, 0, 1)).await;
        send(
            &mut state,
            &tx,
            LoopMessage::PlaybackStarted { id: utterance(1) },
        )
        .await;
        send(&mut state, &tx, played(1, false)).await;
        // The cortex is still writing when the owner speaks again.
        send(&mut state, &tx, typed(INTERRUPTION, 3_000)).await;
        stop_the_call(&mut state);
        assert_eq!(
            turns(&state),
            [
                (TurnRole::User, ASKED.to_owned()),
                (
                    TurnRole::Assistant,
                    "Era uma vez um robô. [interrupted: the owner heard only this]".to_owned()
                ),
            ]
        );
        // A late sentence or end of the abandoned thought changes nothing.
        send(&mut state, &tx, sentence(1, 1, 2)).await;
        send(&mut state, &tx, replied(1, WHOLE, ASKED)).await;
        assert_eq!(turns(&state).len(), 2);
    }

    #[cfg(feature = "voice")]
    #[tokio::test]
    async fn waiting_for_the_rest_chimes_through_the_player_and_keeps_the_turn_open() {
        let (tx, _rx) = mpsc::channel(8);
        let mut state = voiced(true);
        let events = state.voice.as_ref().unwrap().player.subscribe().unwrap();
        send(&mut state, &tx, typed("enton", 1_000)).await;
        assert!(state.organism.is_attending());
        let chime = state.chime.expect("the wait chimed");
        assert_eq!(
            events
                .recv_timeout(std::time::Duration::from_secs(1))
                .unwrap(),
            enton_adapters::PlaybackEvent::Started {
                id: chime,
                text: String::new()
            }
        );
        // The loop hears the chime like speech: Enton's own playback, then its echo.
        send(&mut state, &tx, LoopMessage::PlaybackStarted { id: chime }).await;
        assert!(state.organism.is_speaking());
        send(
            &mut state,
            &tx,
            LoopMessage::PlaybackFinished {
                id: chime,
                interrupted: false,
            },
        )
        .await;
        assert!(state.organism.is_hangover());
        // The owner goes on after the chime: one request, answered as addressed.
        send(&mut state, &tx, typed("que horas são?", 3_000)).await;
        assert_eq!(state.in_flight_thought, Some(ThoughtId(1)));
        assert_eq!(state.last_transcript, None);
        stop_the_call(&mut state);
        assert!(!state.organism.is_attending());

        // With the chime turned off, the wait is silent.
        let mut quiet = voiced(false);
        send(&mut quiet, &tx, typed("enton", 1_000)).await;
        assert!(quiet.organism.is_attending());
        assert_eq!(quiet.chime, None);
    }
}
