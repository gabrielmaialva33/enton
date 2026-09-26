//! Terminal channel driver and async runtime for the Enton digital organism.

// A CLI binary talks to the terminal by design; only libraries must not print.
#![allow(clippy::print_stdout, clippy::print_stderr)]

mod journal;

use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
#[cfg(feature = "voice")]
use std::sync::Arc;
use std::time::{Duration, Instant};

use enton_adapters::cortex::{CortexConfig, OpenAiCortex};
#[cfg(feature = "voice")]
use enton_adapters::voice::{PlaybackEvent, UtteranceId, VoiceConfig, VoicePlayer};
use enton_adapters::{MonotonicClock, SeqNo, read_body_signals};
use enton_core::ports::{ConversationTurn, Cortex, ThoughtRequest};
use enton_core::{Action, Event, Organism, Profile, SpeechCue, ThoughtId};
use journal::{Journal, JournalError};
use tokio::sync::mpsc;

/// Snapshot the organism after this many recorded events (about ten minutes of ticks).
const SNAPSHOT_EVERY: u32 = 600;

/// How a cortex call ended; failures carry a fixed reason for the soul, never text.
type Outcome = Result<(), &'static str>;

const USAGE: &str = "Usage: enton [--profile t1-ref|desktop] [--cortex-url <URL>] [--model <MODEL>] [--soul <PATH> | --no-soul] [--voice] [--speaker <ID>]";

enum LoopMessage {
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

fn flush_stdout() {
    // Best-effort flush for interactive CLI output; ignore failure if stdout pipe is broken.
    if let Err(_err) = std::io::stdout().flush() {
        // Ignored: consumer may have closed stdout pipe.
    }
}

#[derive(Debug)]
struct CliConfig {
    profile: Profile,
    cortex_url: String,
    model: String,
    soul: Option<PathBuf>,
    #[cfg(feature = "voice")]
    voice: bool,
    #[cfg(feature = "voice")]
    speaker_id: Option<i32>,
}

fn parse_cli_args() -> Result<CliConfig, String> {
    parse_args(std::env::args().skip(1))
}

/// Parse flags given either as `--flag value` or as `--flag=value`.
fn parse_args(args: impl IntoIterator<Item = String>) -> Result<CliConfig, String> {
    let mut args = args.into_iter();
    let mut profile = Profile::t1_ref();
    let mut cortex_url = "http://127.0.0.1:11434/v1".to_string();
    let mut model = "qwen3.8:27b-gato".to_string();
    let mut soul = None;
    let mut no_soul = false;
    #[cfg(feature = "voice")]
    let mut voice = false;
    #[cfg(feature = "voice")]
    let mut speaker_id = None;

    while let Some(arg) = args.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((flag, value)) if flag.starts_with("--") => {
                (flag.to_owned(), Some(value.to_owned()))
            }
            _ => (arg.clone(), None),
        };
        let mut value = || {
            inline
                .clone()
                .or_else(|| args.next())
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag.as_str() {
            "--profile" => {
                profile = match value()?.as_str() {
                    "t1-ref" => Profile::t1_ref(),
                    "desktop" => Profile::desktop(),
                    other => return Err(format!("unknown profile: {other}")),
                };
            }
            "--cortex-url" => cortex_url = value()?,
            "--model" => model = value()?,
            "--soul" => soul = Some(PathBuf::from(value()?)),
            "--no-soul" => no_soul = true,
            #[cfg(feature = "voice")]
            "--voice" => voice = true,
            #[cfg(feature = "voice")]
            "--speaker" => {
                let id = value()?;
                speaker_id = Some(
                    id.parse::<i32>()
                        .map_err(|e| format!("invalid speaker id: {e}"))?,
                );
            }
            #[cfg(not(feature = "voice"))]
            "--voice" | "--speaker" => {
                if flag == "--speaker" {
                    value()?;
                }
                eprintln!("Warning: {flag} ignored (voice feature disabled)");
            }
            "--help" | "-h" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }

    if no_soul && soul.is_some() {
        return Err("--soul and --no-soul are mutually exclusive".to_string());
    }
    let soul = if no_soul {
        None
    } else {
        soul.or_else(|| default_soul_path(&profile.name))
    };

    Ok(CliConfig {
        profile,
        cortex_url,
        model,
        soul,
        #[cfg(feature = "voice")]
        voice,
        #[cfg(feature = "voice")]
        speaker_id,
    })
}

/// `$XDG_DATA_HOME/enton/soul-<profile>.sqlite`, falling back to `~/.local/share`.
/// One log per profile: a snapshot only restores under the profile that wrote it.
fn default_soul_path(profile_name: &str) -> Option<PathBuf> {
    let data = std::env::var_os("XDG_DATA_HOME")
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"))
        })?;
    Some(
        data.join("enton")
            .join(format!("soul-{profile_name}.sqlite")),
    )
}

struct RuntimeState {
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

#[cfg(feature = "voice")]
async fn think_stream_and_speak(
    cortex: &OpenAiCortex,
    player: &Arc<VoicePlayer>,
    request: &ThoughtRequest,
) -> (String, Outcome) {
    match cortex.think_stream(request).await {
        Ok(mut rx) => {
            let mut accumulated = String::new();
            while let Some(sentence) = rx.recv().await {
                if !accumulated.is_empty() && !accumulated.ends_with(' ') {
                    accumulated.push(' ');
                }
                accumulated.push_str(&sentence);
                if let Err(err) = player.speak(sentence) {
                    eprintln!("[enton-voice] playback queue error: {err}");
                }
            }
            if accumulated.trim().is_empty() {
                eprintln!("cortex stream produced no output; falling back to offline placeholder");
                let placeholder = "(cortex offline in scaffold)".to_string();
                if let Err(err) = player.speak(&placeholder) {
                    eprintln!("[enton-voice] playback queue error: {err}");
                }
                (placeholder, Err("empty reply"))
            } else {
                (accumulated, Ok(()))
            }
        }
        Err(err) => {
            eprintln!("cortex unavailable ({err}); falling back to offline placeholder");
            let placeholder = "(cortex offline in scaffold)".to_string();
            if let Err(speak_err) = player.speak(&placeholder) {
                eprintln!("[enton-voice] playback queue error: {speak_err}");
            }
            (placeholder, Err("cortex unavailable"))
        }
    }
}

async fn think_text(cortex: &OpenAiCortex, request: &ThoughtRequest) -> (String, Outcome) {
    match cortex.think(request).await {
        Ok(text) => {
            if text.trim().is_empty() {
                eprintln!("cortex produced empty reply; falling back to offline placeholder");
                (
                    "(cortex offline in scaffold)".to_string(),
                    Err("empty reply"),
                )
            } else {
                (text, Ok(()))
            }
        }
        Err(err) => {
            eprintln!("cortex unavailable ({err}); falling back to offline placeholder");
            (
                "(cortex offline in scaffold)".to_string(),
                Err("cortex unavailable"),
            )
        }
    }
}

fn spawn_cortex_think(
    cortex: OpenAiCortex,
    #[cfg(feature = "voice")] player: Option<Arc<VoicePlayer>>,
    request: ThoughtRequest,
    tx: mpsc::Sender<LoopMessage>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let thought = request.thought;
        let user_prompt = request.transcript.clone();
        #[cfg(feature = "voice")]
        let (text, outcome) = if let Some(ref player) = player {
            think_stream_and_speak(&cortex, player, &request).await
        } else {
            think_text(&cortex, &request).await
        };
        #[cfg(not(feature = "voice"))]
        let (text, outcome) = think_text(&cortex, &request).await;
        if let Err(_err) = tx
            .send(LoopMessage::CortexFinished {
                thought,
                text,
                user_prompt,
                outcome,
            })
            .await
        {
            // Main event loop receiver was closed; organism is terminating.
        }
    })
}

async fn read_sensors(
    read: impl FnOnce() -> enton_core::BodySignals + Send + 'static,
) -> Result<enton_core::BodySignals, tokio::task::JoinError> {
    tokio::task::spawn_blocking(read).await
}

fn spawn_timer_task(tx: mpsc::Sender<LoopMessage>, clock: MonotonicClock) {
    tokio::spawn(async move {
        let mut body_counter: u32 = 0;
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        loop {
            interval.tick().await;
            let now = clock.now();
            if tx
                .send(LoopMessage::Event(Event::Tick { now }))
                .await
                .is_err()
            {
                break;
            }
            body_counter += 1;
            if body_counter >= 10 {
                body_counter = 0;
                let signals = match read_sensors(read_body_signals).await {
                    Ok(signals) => signals,
                    Err(error) => {
                        eprintln!("body sensor worker failed: {error}");
                        continue;
                    }
                };
                if tx
                    .send(LoopMessage::Event(Event::Body { now, signals }))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        }
    });
}

fn speech_cue(line: &str) -> SpeechCue {
    let trimmed = line.trim();
    let char_count = trimmed.chars().count();
    let duration_ms = u32::try_from(char_count.saturating_mul(60))
        .unwrap_or(u32::MAX)
        .min(3000);
    let keyword = enton_core::contains_keyword_word(line, "enton");
    SpeechCue {
        energy: 0.8,
        duration_ms,
        vad_confidence: 1.0,
        keyword,
    }
}

fn spawn_stdin_task(tx: mpsc::Sender<LoopMessage>, clock: MonotonicClock) {
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lines() {
            let Ok(line) = line else {
                break;
            };
            let input_end_time = Instant::now();
            let trimmed = line.trim();
            if trimmed.eq_ignore_ascii_case("quit") {
                if let Err(_err) = tx.blocking_send(LoopMessage::Quit) {
                    // Receiver closed; main loop is terminating.
                }
                return;
            }
            let cue = speech_cue(trimmed);
            let event = Event::Speech {
                now: clock.now(),
                cue,
            };
            if tx
                .blocking_send(LoopMessage::SpeechInput {
                    text: trimmed.to_string(),
                    event,
                    input_end_time,
                })
                .is_err()
            {
                return;
            }
        }
        if let Err(_err) = tx.blocking_send(LoopMessage::Quit) {
            // Receiver closed; main loop is terminating.
        }
    });
}

#[cfg(feature = "voice")]
fn try_init_voice(
    cli_voice: bool,
    cli_speaker_id: Option<i32>,
    profile_name: &str,
) -> Option<Arc<VoicePlayer>> {
    if !cli_voice {
        return None;
    }

    let mut config = VoiceConfig::for_profile(profile_name);
    if let Some(sid) = cli_speaker_id {
        config.speaker_id = sid;
    }

    match VoicePlayer::new(config.clone()) {
        Ok(player) => {
            println!(
                "[enton] Voice output enabled (Kokoro PT-BR speaker #{}, {} threads, cpal)",
                config.speaker_id, config.num_threads
            );
            Some(Arc::new(player))
        }
        Err(err) => {
            eprintln!("[enton] Voice output unavailable ({err})");
            eprintln!(
                "[enton] See the README (Voice and microphone) for the Kokoro PT-BR model layout."
            );
            None
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> std::process::ExitCode {
    let cli = match parse_cli_args() {
        Ok(c) => c,
        Err(err) => {
            eprintln!("Error: {err}");
            eprintln!("{USAGE}");
            return std::process::ExitCode::FAILURE;
        }
    };

    let (organism, journal) = match restore(cli.profile, cli.soul).await {
        Ok(restored) => restored,
        Err(err) => {
            eprintln!("Error: {err}");
            eprintln!("hint: pass --soul <PATH> for another log, or --no-soul to run without one");
            return std::process::ExitCode::FAILURE;
        }
    };

    #[cfg(feature = "voice")]
    let voice_profile_name = organism.profile().name.clone();
    #[cfg(feature = "voice")]
    let voice_player = match tokio::task::spawn_blocking(move || {
        try_init_voice(cli.voice, cli.speaker_id, &voice_profile_name)
    })
    .await
    {
        Ok(player) => player,
        Err(error) => {
            eprintln!("voice initialization worker failed: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    // A restored organism resumes where it stopped; a fresh one starts at zero.
    let clock = MonotonicClock::resuming_at(organism.last_seen());

    let (tx, rx) = mpsc::channel::<LoopMessage>(64);

    #[cfg(feature = "voice")]
    if let Some(ref player) = voice_player {
        spawn_voice_event_listener(player, tx.clone());
    }

    spawn_timer_task(tx.clone(), clock);
    spawn_stdin_task(tx.clone(), clock);

    let state = RuntimeState {
        organism,
        clock,
        cortex: OpenAiCortex::new(CortexConfig {
            base_url: cli.cortex_url,
            model: cli.model,
            ..CortexConfig::default()
        }),
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
    };

    match run_event_loop(state, rx, tx).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            // Acting on events the soul did not record would break replay; stop instead.
            eprintln!("[enton] stopping: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Restore the organism from its soul, or start a fresh one without a log.
async fn restore(
    profile: Profile,
    soul: Option<PathBuf>,
) -> Result<(Organism, Option<Journal>), String> {
    let Some(path) = soul else {
        println!("[enton] Soul disabled: nothing is recorded");
        return Organism::new(profile)
            .map(|organism| (organism, None))
            .map_err(|err| err.to_string());
    };
    let opened = tokio::task::spawn_blocking(move || {
        Journal::open(&path, &profile).map(|(journal, restored)| (journal, restored, path))
    })
    .await
    .map_err(|err| format!("soul worker failed: {err}"))?;
    let (journal, restored, path) = opened.map_err(|err| err.to_string())?;
    println!(
        "[enton] Soul: {} (resuming at {} ms)",
        path.display(),
        restored.organism.last_seen().0
    );
    if !restored.abandoned.is_empty() {
        eprintln!(
            "[enton] {} thought(s) interrupted by the last shutdown were marked failed",
            restored.abandoned.len()
        );
    }
    Ok((restored.organism, Some(journal)))
}

async fn run_event_loop(
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

#[cfg(feature = "voice")]
fn spawn_voice_event_listener(player: &Arc<VoicePlayer>, tx: mpsc::Sender<LoopMessage>) {
    let receiver = match player.subscribe() {
        Ok(receiver) => receiver,
        Err(error) => {
            eprintln!("voice event subscription failed: {error}");
            return;
        }
    };
    let player = Arc::downgrade(player);
    std::thread::spawn(move || {
        let mut previous_loss = 0;
        while let Ok(event) = receiver.recv() {
            if let Some(player) = player.upgrade() {
                let stats = player.event_stats();
                let loss = stats
                    .mailbox_dropped
                    .saturating_add(stats.subscriber_dropped);
                if loss != previous_loss {
                    eprintln!("[enton-voice] incomplete event audit: {loss} dropped deliveries");
                    previous_loss = loss;
                }
            }
            let message = match event {
                PlaybackEvent::Started { id, .. } => LoopMessage::PlaybackStarted { id },
                PlaybackEvent::Finished { id } | PlaybackEvent::Cancelled { id } => {
                    LoopMessage::PlaybackFinished { id }
                }
                PlaybackEvent::Failed { id, reason } => LoopMessage::PlaybackFailed { id, reason },
            };
            if tx.blocking_send(message).is_err() {
                break;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rejected_cues_leave_transcripts_intact() {
        let (tx, _rx) = mpsc::channel(1);
        let mut state = test_state(Organism::new(Profile::t1_ref()).unwrap(), None);

        // 1. Keyword -> accepted, Attend until later.
        let kw_event = Event::Speech {
            now: enton_core::Millis(100),
            cue: enton_core::SpeechCue {
                energy: 0.9,
                vad_confidence: 0.9,
                duration_ms: 500, // < 900ms => Attend
                keyword: true,
            },
        };
        state
            .handle_message(
                LoopMessage::SpeechInput {
                    text: "enton".to_string(),
                    event: kw_event,
                    input_end_time: Instant::now(),
                },
                &tx,
            )
            .await
            .unwrap();

        assert_eq!(state.last_transcript.as_deref(), None);
        assert_eq!(state.attended_transcript.as_deref(), Some("enton")); // Attend sets attended_transcript

        // 2. Rejected noise (low VAD) -> Abstain
        let noise_event = Event::Speech {
            now: enton_core::Millis(200),
            cue: enton_core::SpeechCue {
                energy: 0.1,
                vad_confidence: 0.1,
                duration_ms: 100,
                keyword: false,
            },
        };
        state
            .handle_message(
                LoopMessage::SpeechInput {
                    text: "shhh".to_string(),
                    event: noise_event,
                    input_end_time: Instant::now(),
                },
                &tx,
            )
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
        let cont_event = Event::Speech {
            now: enton_core::Millis(300),
            cue: enton_core::SpeechCue {
                energy: 0.9,
                vad_confidence: 0.9,
                duration_ms: 1000,
                keyword: false,
            },
        };
        state
            .handle_message(
                LoopMessage::SpeechInput {
                    text: "help me".to_string(),
                    event: cont_event,
                    input_end_time: Instant::now(),
                },
                &tx,
            )
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

        let noise2_event = Event::Speech {
            now: enton_core::Millis(400),
            cue: enton_core::SpeechCue {
                energy: 0.1,
                vad_confidence: 0.1,
                duration_ms: 100,
                keyword: false,
            },
        };
        state
            .handle_message(
                LoopMessage::SpeechInput {
                    text: "cough".to_string(),
                    event: noise2_event,
                    input_end_time: Instant::now(),
                },
                &tx,
            )
            .await
            .unwrap();

        assert!(
            state.in_flight_thought.is_some(),
            "rejected cue did not abort flight"
        );
    }
    #[tokio::test(flavor = "current_thread")]
    async fn sensor_reads_run_off_the_executor_thread() {
        let executor_thread = std::thread::current().id();
        let signals = read_sensors(move || {
            assert_ne!(std::thread::current().id(), executor_thread);
            enton_core::BodySignals::default()
        })
        .await
        .unwrap();
        assert_eq!(signals, enton_core::BodySignals::default());
    }

    #[test]
    fn normal_shutdown_returns_and_drops_runtime_resources() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::shutdown_child",
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
        RuntimeState {
            organism,
            clock: MonotonicClock::new(),
            cortex: OpenAiCortex::with_endpoint("http://127.0.0.1:9", "unused"),
            #[cfg(feature = "voice")]
            voice_player: None,
            last_transcript: None,
            attended_transcript: None,
            history: VecDeque::new(),
            pending_first_audio_timer: None,
            active_cortex_task: None,
            in_flight_thought: None,
            journal,
            last_seq: None,
            unsnapshotted: 0,
        }
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

    #[test]
    fn cli_accepts_both_flag_forms_and_rejects_conflicts() {
        let parse = |args: &[&str]| parse_args(args.iter().map(|arg| (*arg).to_owned()));
        let cli = parse(&[
            "--profile=desktop",
            "--model",
            "m",
            "--soul",
            "/tmp/s.sqlite",
        ])
        .unwrap();
        assert_eq!(cli.profile.name, "desktop");
        assert_eq!(cli.model, "m");
        assert_eq!(cli.soul, Some(PathBuf::from("/tmp/s.sqlite")));
        assert!(parse(&["--no-soul"]).unwrap().soul.is_none());
        assert!(parse(&["--soul=/tmp/s.sqlite", "--no-soul"]).is_err());
        assert!(parse(&["--model"]).is_err());
        assert!(parse(&["--profile", "mars"]).is_err());
        assert!(parse(&["--bogus"]).is_err());
    }

    #[test]
    fn speech_cue_keyword_addressing() {
        assert!(speech_cue("enton").keyword);
        assert!(speech_cue("Enton, que horas são?").keyword);
        assert!(speech_cue("ei ENTON!").keyword);

        assert!(!speech_cue("Benton").keyword);
        assert!(!speech_cue("sentenced").keyword);
        assert!(!speech_cue("então").keyword);
    }

    #[test]
    fn speech_cue_duration_rule() {
        assert_eq!(speech_cue("").duration_ms, 0);
        assert_eq!(speech_cue("enton").duration_ms, 300);
        assert_eq!(speech_cue("ei ENTON!").duration_ms, 540);
        assert_eq!(speech_cue("Enton, que horas são?").duration_ms, 1260);

        let cap_boundary = "a".repeat(50);
        assert_eq!(speech_cue(&cap_boundary).duration_ms, 3000);

        let over_cap = "a".repeat(100);
        assert_eq!(speech_cue(&over_cap).duration_ms, 3000);
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

    #[cfg(feature = "voice")]
    #[test]
    fn playback_event_mapping_matches_core_lifecycle() {
        fn map_playback_event(event: PlaybackEvent) -> LoopMessage {
            match event {
                PlaybackEvent::Started { id, .. } => LoopMessage::PlaybackStarted { id },
                PlaybackEvent::Finished { id } | PlaybackEvent::Cancelled { id } => {
                    LoopMessage::PlaybackFinished { id }
                }
                PlaybackEvent::Failed { id, reason } => LoopMessage::PlaybackFailed { id, reason },
            }
        }

        assert!(matches!(
            map_playback_event(PlaybackEvent::Started {
                id: UtteranceId(1),
                text: "hello".into()
            }),
            LoopMessage::PlaybackStarted { id: UtteranceId(1) }
        ));
        assert!(matches!(
            map_playback_event(PlaybackEvent::Finished { id: UtteranceId(1) }),
            LoopMessage::PlaybackFinished { id: UtteranceId(1) }
        ));
        assert!(matches!(
            map_playback_event(PlaybackEvent::Cancelled { id: UtteranceId(1) }),
            LoopMessage::PlaybackFinished { id: UtteranceId(1) }
        ));
        assert!(matches!(
            map_playback_event(PlaybackEvent::Failed {
                id: UtteranceId(1),
                reason: "error".into()
            }),
            LoopMessage::PlaybackFailed {
                id: UtteranceId(1),
                ..
            }
        ));
    }
}
