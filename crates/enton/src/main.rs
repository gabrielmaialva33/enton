//! Terminal channel driver and async runtime for the Enton digital organism.

// A CLI binary talks to the terminal by design; only libraries must not print.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::VecDeque;
use std::io::Write;
#[cfg(feature = "voice")]
use std::sync::Arc;
use std::time::{Duration, Instant};

use enton_adapters::cortex::{CortexConfig, OpenAiCortex};
#[cfg(feature = "voice")]
use enton_adapters::voice::{PlaybackEvent, UtteranceId, VoiceConfig, VoicePlayer};
use enton_adapters::{MonotonicClock, read_body_signals};
use enton_core::ports::{ConversationTurn, Cortex, ThoughtRequest};
use enton_core::{Action, Event, Organism, Profile, SpeechCue, ThoughtId};
use tokio::sync::mpsc;

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
    #[cfg(feature = "voice")]
    voice: bool,
    #[cfg(feature = "voice")]
    speaker_id: Option<i32>,
}

fn parse_cli_args() -> Result<CliConfig, String> {
    let mut args = std::env::args().skip(1);
    let mut profile_name = "t1-ref";
    let mut cortex_url = "http://127.0.0.1:11434/v1".to_string();
    let mut model = "qwen3.8:27b-gato".to_string();
    #[cfg(feature = "voice")]
    let mut voice = false;
    #[cfg(feature = "voice")]
    let mut speaker_id = None;

    while let Some(arg) = args.next() {
        if arg == "--profile" {
            let Some(val) = args.next() else {
                return Err("missing value for --profile".to_string());
            };
            profile_name = match val.as_str() {
                "t1-ref" => "t1-ref",
                "desktop" => "desktop",
                other => return Err(format!("unknown profile: {other}")),
            };
        } else if let Some(val) = arg.strip_prefix("--profile=") {
            profile_name = match val {
                "t1-ref" => "t1-ref",
                "desktop" => "desktop",
                other => return Err(format!("unknown profile: {other}")),
            };
        } else if arg == "--cortex-url" {
            let Some(val) = args.next() else {
                return Err("missing value for --cortex-url".to_string());
            };
            cortex_url = val;
        } else if let Some(val) = arg.strip_prefix("--cortex-url=") {
            cortex_url = val.to_string();
        } else if arg == "--model" {
            let Some(val) = args.next() else {
                return Err("missing value for --model".to_string());
            };
            model = val;
        } else if let Some(val) = arg.strip_prefix("--model=") {
            model = val.to_string();
        } else if arg == "--voice" {
            #[cfg(feature = "voice")]
            {
                voice = true;
            }
            #[cfg(not(feature = "voice"))]
            {
                eprintln!("Warning: --voice flag ignored (voice feature disabled)");
            }
        } else if arg == "--speaker" {
            let Some(val) = args.next() else {
                return Err("missing value for --speaker".to_string());
            };
            #[cfg(feature = "voice")]
            {
                let sid = val
                    .parse::<i32>()
                    .map_err(|e| format!("invalid speaker id: {e}"))?;
                speaker_id = Some(sid);
            }
            #[cfg(not(feature = "voice"))]
            {
                let _ = val;
                eprintln!("Warning: --speaker flag ignored (voice feature disabled)");
            }
        } else if let Some(val) = arg.strip_prefix("--speaker=") {
            #[cfg(feature = "voice")]
            {
                let sid = val
                    .parse::<i32>()
                    .map_err(|e| format!("invalid speaker id: {e}"))?;
                speaker_id = Some(sid);
            }
            #[cfg(not(feature = "voice"))]
            {
                let _ = val;
                eprintln!("Warning: --speaker flag ignored (voice feature disabled)");
            }
        } else if arg == "--help" || arg == "-h" {
            println!(
                "Usage: enton [--profile t1-ref|desktop] [--cortex-url <URL>] [--model <MODEL>] [--voice] [--speaker <ID>]"
            );
            std::process::exit(0);
        } else {
            return Err(format!("unknown argument: {arg}"));
        }
    }

    let profile = match profile_name {
        "t1-ref" => Profile::t1_ref(),
        "desktop" => Profile::desktop(),
        _ => unreachable!(),
    };

    Ok(CliConfig {
        profile,
        cortex_url,
        model,
        #[cfg(feature = "voice")]
        voice,
        #[cfg(feature = "voice")]
        speaker_id,
    })
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
}

impl RuntimeState {
    fn step_event(&mut self, event: &Event, tx: &mpsc::Sender<LoopMessage>) {
        let actions = self.organism.step(event);
        self.dispatch(actions, tx);
    }

    fn handle_speech_input(
        &mut self,
        text: String,
        event: &Event,
        input_end_time: Instant,
        tx: &mpsc::Sender<LoopMessage>,
    ) {
        let actions = self.organism.step(event);
        let is_accepted = actions
            .iter()
            .any(|a| matches!(a, Action::Think { .. } | Action::Attend { .. }));

        if is_accepted {
            if let Some(task) = self.active_cortex_task.take() {
                task.abort();
            }
            self.in_flight_thought = None;
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

        self.dispatch(actions, tx);
    }

    fn handle_cortex_finished(
        &mut self,
        thought: ThoughtId,
        text: String,
        user_prompt: Option<String>,
        tx: &mpsc::Sender<LoopMessage>,
    ) {
        if self.in_flight_thought != Some(thought) {
            return;
        }
        self.active_cortex_task = None;
        self.in_flight_thought = None;

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
        self.step_event(&reply, tx);
    }

    fn handle_message(&mut self, msg: LoopMessage, tx: &mpsc::Sender<LoopMessage>) -> bool {
        match msg {
            LoopMessage::SpeechInput {
                text,
                event,
                input_end_time,
            } => {
                self.handle_speech_input(text, &event, input_end_time, tx);
                true
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
                    utterance: enton_core::UtteranceId(id.0),
                };
                self.step_event(&event, tx);
                true
            }
            #[cfg(feature = "voice")]
            LoopMessage::PlaybackFinished { id } => {
                let event = Event::PlaybackFinished {
                    now: self.clock.now(),
                    utterance: enton_core::UtteranceId(id.0),
                };
                self.step_event(&event, tx);
                true
            }
            #[cfg(feature = "voice")]
            LoopMessage::PlaybackFailed { id, reason } => {
                eprintln!("[enton-voice] utterance {id} failed: {reason}");
                self.pending_first_audio_timer = None;
                let event = Event::PlaybackFinished {
                    now: self.clock.now(),
                    utterance: enton_core::UtteranceId(id.0),
                };
                self.step_event(&event, tx);
                true
            }
            LoopMessage::Event(event) => {
                self.step_event(&event, tx);
                true
            }
            LoopMessage::CortexFinished {
                thought,
                text,
                user_prompt,
            } => {
                self.handle_cortex_finished(thought, text, user_prompt, tx);
                true
            }
            LoopMessage::Quit => {
                if let Some(task) = self.active_cortex_task.take() {
                    task.abort();
                }
                #[cfg(feature = "voice")]
                if let Some(ref player) = self.voice_player {
                    player.cancel();
                }
                self.attended_transcript = None;
                flush_stdout();
                false
            }
        }
    }

    fn dispatch(&mut self, actions: Vec<Action>, tx: &mpsc::Sender<LoopMessage>) {
        for action in actions {
            println!("{action:?}");
            flush_stdout();
            match action {
                Action::Think {
                    thought, reason, ..
                } => {
                    if let Some(old_task) = self.active_cortex_task.take() {
                        old_task.abort();
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
    }
}

#[cfg(feature = "voice")]
async fn think_stream_and_speak(
    cortex: &OpenAiCortex,
    player: &Arc<VoicePlayer>,
    request: &ThoughtRequest,
) -> String {
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
                placeholder
            } else {
                accumulated
            }
        }
        Err(err) => {
            eprintln!("cortex unavailable ({err}); falling back to offline placeholder");
            let placeholder = "(cortex offline in scaffold)".to_string();
            if let Err(speak_err) = player.speak(&placeholder) {
                eprintln!("[enton-voice] playback queue error: {speak_err}");
            }
            placeholder
        }
    }
}

async fn think_text(cortex: &OpenAiCortex, request: &ThoughtRequest) -> String {
    match cortex.think(request).await {
        Ok(text) => {
            if text.trim().is_empty() {
                eprintln!("cortex produced empty reply; falling back to offline placeholder");
                "(cortex offline in scaffold)".to_string()
            } else {
                text
            }
        }
        Err(err) => {
            eprintln!("cortex unavailable ({err}); falling back to offline placeholder");
            "(cortex offline in scaffold)".to_string()
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
        let text = if let Some(ref player) = player {
            think_stream_and_speak(&cortex, player, &request).await
        } else {
            think_text(&cortex, &request).await
        };
        #[cfg(not(feature = "voice"))]
        let text = think_text(&cortex, &request).await;
        if let Err(_err) = tx
            .send(LoopMessage::CortexFinished {
                thought,
                text,
                user_prompt,
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
            let char_count = trimmed.chars().count();
            let duration_ms = u32::try_from(char_count.saturating_mul(60))
                .unwrap_or(u32::MAX)
                .min(3000);
            let keyword = trimmed.to_ascii_lowercase().contains("enton");
            let cue = SpeechCue {
                energy: 0.8,
                duration_ms,
                vad_confidence: 1.0,
                keyword,
            };
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
            eprintln!(
                "Usage: enton [--profile t1-ref|desktop] [--cortex-url <URL>] [--model <MODEL>] [--voice] [--speaker <ID>]"
            );
            return std::process::ExitCode::FAILURE;
        }
    };

    #[cfg(feature = "voice")]
    let voice_profile_name = cli.profile.name.clone();
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
    let clock = MonotonicClock::new();

    let (tx, rx) = mpsc::channel::<LoopMessage>(64);

    #[cfg(feature = "voice")]
    if let Some(ref player) = voice_player {
        spawn_voice_event_listener(player, tx.clone());
    }

    spawn_timer_task(tx.clone(), clock);
    spawn_stdin_task(tx.clone(), clock);

    let state = RuntimeState {
        organism: Organism::new(cli.profile),
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
    };

    run_event_loop(state, rx, tx).await;
    std::process::ExitCode::SUCCESS
}

async fn run_event_loop(
    mut state: RuntimeState,
    mut rx: mpsc::Receiver<LoopMessage>,
    tx: mpsc::Sender<LoopMessage>,
) {
    while let Some(msg) = rx.recv().await {
        let should_continue = state.handle_message(msg, &tx);
        if !should_continue {
            break;
        }
    }
    flush_stdout();
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
        let mut state = RuntimeState {
            organism: Organism::new(Profile::t1_ref()),
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
        };

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
        state.handle_message(
            LoopMessage::SpeechInput {
                text: "enton".to_string(),
                event: kw_event,
                input_end_time: Instant::now(),
            },
            &tx,
        );

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
        state.handle_message(
            LoopMessage::SpeechInput {
                text: "shhh".to_string(),
                event: noise_event,
                input_end_time: Instant::now(),
            },
            &tx,
        );

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
        state.handle_message(
            LoopMessage::SpeechInput {
                text: "help me".to_string(),
                event: cont_event,
                input_end_time: Instant::now(),
            },
            &tx,
        );

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
        state.handle_message(
            LoopMessage::SpeechInput {
                text: "cough".to_string(),
                event: noise2_event,
                input_end_time: Instant::now(),
            },
            &tx,
        );

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
        let state = RuntimeState {
            organism: Organism::new(Profile::t1_ref()),
            clock: MonotonicClock::new(),
            cortex: OpenAiCortex::with_endpoint("http://127.0.0.1:9", "unused"),
            #[cfg(feature = "voice")]
            voice_player: Some(player),
            last_transcript: None,
            attended_transcript: None,
            history: VecDeque::new(),
            pending_first_audio_timer: None,
            active_cortex_task: None,
            in_flight_thought: None,
        };
        run_event_loop(state, rx, tx).await;
        #[cfg(feature = "voice")]
        assert!(weak.upgrade().is_none());
        println!("runtime resources dropped");
    }

    #[cfg(feature = "voice")]
    #[tokio::test]
    async fn voice_playback_lifecycle_events_forward_to_core() {
        let (tx, _rx) = mpsc::channel(64);
        let mut state = RuntimeState {
            organism: Organism::new(Profile::t1_ref()),
            clock: MonotonicClock::new(),
            cortex: OpenAiCortex::with_endpoint("http://127.0.0.1:9", "unused"),
            voice_player: None,
            last_transcript: None,
            attended_transcript: None,
            history: VecDeque::new(),
            pending_first_audio_timer: None,
            active_cortex_task: None,
            in_flight_thought: None,
        };

        // 1. PlaybackStarted forwards to core and updates playback status to speaking
        assert!(!state.organism.is_speaking());
        state.handle_message(LoopMessage::PlaybackStarted { id: UtteranceId(1) }, &tx);
        assert!(state.organism.is_speaking());

        // 2. PlaybackFinished forwards to core and transitions from speaking to hangover
        state.handle_message(LoopMessage::PlaybackFinished { id: UtteranceId(1) }, &tx);
        assert!(!state.organism.is_speaking());
        assert!(state.organism.is_hangover());

        // 3. PlaybackStarted again with utterance 2
        state.handle_message(LoopMessage::PlaybackStarted { id: UtteranceId(2) }, &tx);
        assert!(state.organism.is_speaking());

        // 4. PlaybackFailed forwards to core as finish so the watchdog is not waited on
        state.handle_message(
            LoopMessage::PlaybackFailed {
                id: UtteranceId(2),
                reason: "DAC buffer underrun".into(),
            },
            &tx,
        );
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
