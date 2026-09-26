//! Tasks that feed the event loop: cortex calls, the clock, body sensors and the
//! owner's checklist, typed lines from stdin, and voice playback events.

#[cfg(feature = "voice")]
use std::sync::Arc;
use std::time::{Duration, Instant};

use enton_adapters::checklist::{self, Checklist, ChecklistWatcher};
use enton_adapters::cortex::OpenAiCortex;
#[cfg(feature = "voice")]
use enton_adapters::voice::{
    PlaybackEvent, UtteranceId, VoiceConfig, VoiceEngine, VoiceModel, VoicePlayer, speakable,
};
use enton_adapters::{MonotonicClock, read_body_signals};
use enton_core::ports::{Cortex, NOTHING_TO_SAY, ThoughtRequest, says_nothing};
use enton_core::{Event, Reason, SpeechCue};
use tokio::sync::mpsc;

#[cfg(feature = "voice")]
use crate::conversation::append_sentence;
use crate::runtime::{LoopMessage, Outcome};

/// Whether the cortex may answer `request` with silence: a drive thought asks whether
/// anything on the checklist is worth saying, and nothing is a valid answer. Any other
/// thought owes one, so an empty reply is a failure.
fn may_stay_silent(request: &ThoughtRequest) -> bool {
    matches!(request.reason, Reason::Drive(_))
}

/// What the owner said to `request`, for the conversation history. A drive thought's
/// transcript is the checklist, which nobody said: it stays out.
pub(crate) fn user_turn(request: &ThoughtRequest) -> Option<String> {
    if may_stay_silent(request) {
        None
    } else {
        request.transcript.clone()
    }
}

/// What a drive thought asks the cortex: the trigger, in the form the cortex uses when
/// a thought has no transcript, then the owner's checklist and how to say nothing.
pub(crate) fn drive_prompt(reason: &Reason, checklist: &str) -> String {
    format!(
        "[Trigger reason: {reason:?}] Nobody spoke to you: this is your own turn. \
         The owner's checklist (CHECKLIST.md) lists what you may bring up on your own:\n\n\
         {}\n\n\
         If nothing on it is worth saying now, reply exactly {NOTHING_TO_SAY}.",
        checklist.trim()
    )
}

/// Hand one sentence of a reply to the player, without what a voice should not read out
/// (see [`speakable`]): the utterance that plays it, or `None` when nothing was left to
/// say or the player refused it.
#[cfg(feature = "voice")]
fn speak_sentence(player: &VoicePlayer, sentence: &str) -> Option<UtteranceId> {
    let spoken = speakable(sentence);
    if spoken.is_empty() {
        return None;
    }
    player
        .speak(spoken)
        .inspect_err(|err| eprintln!("[enton-voice] playback queue error: {err}"))
        .ok()
}

/// Stream `request`'s reply from the cortex and speak it sentence by sentence, telling
/// the loop which utterance carries each sentence (so it can remember only what the
/// owner heard). Returns the whole reply as written, stage directions included.
#[cfg(feature = "voice")]
pub(crate) async fn think_stream_and_speak(
    cortex: &OpenAiCortex,
    player: &Arc<VoicePlayer>,
    request: &ThoughtRequest,
    tx: &mpsc::Sender<LoopMessage>,
) -> (String, Outcome) {
    let silence = may_stay_silent(request);
    match cortex.think_stream(request).await {
        Ok(mut rx) => {
            let mut accumulated = String::new();
            while let Some(sentence) = rx.recv().await {
                // A drive's "nothing to say" is silence, never spoken.
                if silence && says_nothing(&sentence) {
                    continue;
                }
                append_sentence(&mut accumulated, &sentence);
                let utterance = speak_sentence(player, &sentence);
                let spoken = LoopMessage::Sentence {
                    thought: request.thought,
                    text: sentence,
                    utterance,
                };
                if tx.send(spoken).await.is_err() {
                    // The loop is gone: nobody is left to hear the rest.
                    break;
                }
            }
            if accumulated.trim().is_empty() && silence {
                (String::new(), Ok(()))
            } else if accumulated.trim().is_empty() {
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
            if may_stay_silent(request) && says_nothing(&text) {
                // A drive thought found nothing worth saying: silence, not a failure.
                (String::new(), Ok(()))
            } else if text.trim().is_empty() {
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
        let user_prompt = user_turn(&request);
        #[cfg(feature = "voice")]
        let (text, outcome) = if let Some(ref player) = player {
            think_stream_and_speak(&cortex, player, &request, &tx).await
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

/// Poll the checklist off the executor thread: the watcher back, and the checklist when
/// it changed.
async fn poll_checklist(
    mut watcher: ChecklistWatcher,
) -> Result<(ChecklistWatcher, Option<Checklist>), tokio::task::JoinError> {
    tokio::task::spawn_blocking(move || {
        let changed = watcher.poll();
        (watcher, changed)
    })
    .await
}

/// Say what the checklist now holds, on stderr so stdout stays the decision log.
fn report_checklist(watcher: &ChecklistWatcher, checklist: &Checklist) {
    let Some(path) = watcher.path() else {
        eprintln!("[enton] Checklist: no config directory (set HOME or XDG_CONFIG_HOME)");
        return;
    };
    let path = path.display();
    match checklist {
        Checklist::Actionable(_) => eprintln!("[enton] Checklist: {path} (something to check)"),
        Checklist::Empty => eprintln!("[enton] Checklist: {path} holds nothing to check"),
        Checklist::Missing => {
            eprintln!("[enton] Checklist: none at {path}, so drives have nothing to check");
        }
        Checklist::Unusable(why) => {
            eprintln!("[enton] Checklist: {path} is not used, {why}");
        }
    }
}

pub(crate) fn spawn_timer_task(tx: mpsc::Sender<LoopMessage>, clock: MonotonicClock) {
    tokio::spawn(async move {
        let mut checklist = Some(ChecklistWatcher::new(checklist::default_path()));
        let mut body_counter: u32 = 0;
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        loop {
            interval.tick().await;
            // The checklist is read before the first tick, then polled with the body.
            if body_counter == 0
                && let Some(watcher) = checklist.take()
            {
                match poll_checklist(watcher).await {
                    Ok((watcher, changed)) => {
                        if let Some(changed) = changed {
                            report_checklist(&watcher, &changed);
                            if tx.send(LoopMessage::Checklist(changed)).await.is_err() {
                                break;
                            }
                        }
                        checklist = Some(watcher);
                    }
                    Err(error) => eprintln!("checklist worker failed: {error}"),
                }
            }
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
        media: None,
        turn_complete: None,
        // Typed text is addressed to Enton by construction: nobody types into its
        // terminal to talk to someone else.
        directed: Some(1.0),
        direction: None,
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
    voice_model: VoiceModel,
    cli_speaker_id: Option<i32>,
    profile_name: &str,
) -> Option<Arc<VoicePlayer>> {
    if !cli_voice {
        return None;
    }

    let mut config = VoiceConfig::for_profile_with_model(profile_name, voice_model);
    // The command line refuses --speaker for single-speaker voices.
    if let (Some(sid), VoiceEngine::Kokoro(kokoro)) = (cli_speaker_id, &mut config.engine) {
        kokoro.speaker_id = sid;
    }

    match VoicePlayer::new(config) {
        Ok(player) => {
            println!(
                "[enton] Voice output enabled ({}, {} threads, cpal)",
                player.config().engine,
                player.config().num_threads
            );
            if player.is_resampling() {
                println!(
                    "[enton] Voice playback at {} Hz, resampled from the model's {} Hz (the device does not take it)",
                    player.device_sample_rate(),
                    player.native_sample_rate()
                );
            } else {
                println!(
                    "[enton] Voice playback at {} Hz, the model's native rate (no resampling)",
                    player.device_sample_rate()
                );
            }
            Some(Arc::new(player))
        }
        Err(err) => {
            eprintln!("[enton] Voice output unavailable ({err})");
            eprintln!(
                "[enton] See the README (Voice and microphone) for the Kokoro and Piper model layouts."
            );
            None
        }
    }
}

/// What a playback event tells the loop. A cancelled utterance ended too, cut off: the
/// soul records that it was, and it counts as unheard.
#[cfg(feature = "voice")]
fn playback_message(event: PlaybackEvent) -> LoopMessage {
    match event {
        PlaybackEvent::Started { id, .. } => LoopMessage::PlaybackStarted { id },
        PlaybackEvent::Finished { id } => LoopMessage::PlaybackFinished {
            id,
            interrupted: false,
        },
        PlaybackEvent::Cancelled { id } => LoopMessage::PlaybackFinished {
            id,
            interrupted: true,
        },
        PlaybackEvent::Failed { id, reason } => LoopMessage::PlaybackFailed { id, reason },
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
            if tx.blocking_send(playback_message(event)).is_err() {
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
    fn a_drive_asks_with_the_checklist_and_may_stay_silent() {
        let reason = Reason::Drive("curiosity".to_owned());
        let prompt = drive_prompt(&reason, "\n# Hoje\n- [ ] regar as plantas\n\n");
        assert!(
            prompt.starts_with("[Trigger reason: Drive(\"curiosity\")]"),
            "{prompt}"
        );
        assert!(
            prompt.contains("\n\n# Hoje\n- [ ] regar as plantas\n\n"),
            "{prompt}"
        );
        assert!(
            prompt.ends_with("reply exactly NOTHING_TO_SAY."),
            "{prompt}"
        );
        let request = |reason| ThoughtRequest {
            thought: enton_core::ThoughtId(1),
            reason,
            transcript: None,
            history: Vec::new(),
        };
        assert!(may_stay_silent(&request(reason)));
        for owed in [Reason::Keyword, Reason::FollowUp, Reason::Speech] {
            assert!(!may_stay_silent(&request(owed)));
        }
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
        assert!(matches!(
            playback_message(PlaybackEvent::Started {
                id: UtteranceId(1),
                text: "hello".into()
            }),
            LoopMessage::PlaybackStarted { id: UtteranceId(1) }
        ));
        assert!(matches!(
            playback_message(PlaybackEvent::Finished { id: UtteranceId(1) }),
            LoopMessage::PlaybackFinished {
                id: UtteranceId(1),
                interrupted: false
            }
        ));
        // A cancelled utterance ended too, but was cut off.
        assert!(matches!(
            playback_message(PlaybackEvent::Cancelled { id: UtteranceId(1) }),
            LoopMessage::PlaybackFinished {
                id: UtteranceId(1),
                interrupted: true
            }
        ));
        assert!(matches!(
            playback_message(PlaybackEvent::Failed {
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
