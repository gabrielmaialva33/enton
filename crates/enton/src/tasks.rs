//! Tasks that feed the event loop: cortex calls, the clock and body sensors,
//! typed lines from stdin, and voice playback events.

#[cfg(feature = "voice")]
use std::sync::Arc;
use std::time::{Duration, Instant};

use enton_adapters::cortex::OpenAiCortex;
#[cfg(feature = "voice")]
use enton_adapters::voice::{PlaybackEvent, VoiceConfig, VoicePlayer};
use enton_adapters::{MonotonicClock, read_body_signals};
use enton_core::ports::{Cortex, ThoughtRequest};
use enton_core::{Event, SpeechCue};
use tokio::sync::mpsc;

use crate::runtime::{LoopMessage, Outcome};

#[cfg(feature = "voice")]
pub(crate) async fn think_stream_and_speak(
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

pub(crate) async fn think_text(
    cortex: &OpenAiCortex,
    request: &ThoughtRequest,
) -> (String, Outcome) {
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

pub(crate) fn spawn_cortex_think(
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

pub(crate) async fn read_sensors(
    read: impl FnOnce() -> enton_core::BodySignals + Send + 'static,
) -> Result<enton_core::BodySignals, tokio::task::JoinError> {
    tokio::task::spawn_blocking(read).await
}

pub(crate) fn spawn_timer_task(tx: mpsc::Sender<LoopMessage>, clock: MonotonicClock) {
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

pub(crate) fn speech_cue(line: &str) -> SpeechCue {
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
        speaker_sim: None,
    }
}

pub(crate) fn spawn_stdin_task(tx: mpsc::Sender<LoopMessage>, clock: MonotonicClock) {
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
pub(crate) fn try_init_voice(
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

#[cfg(feature = "voice")]
pub(crate) fn spawn_voice_event_listener(player: &Arc<VoicePlayer>, tx: mpsc::Sender<LoopMessage>) {
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
    #[cfg(feature = "voice")]
    use enton_adapters::voice::UtteranceId;

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
