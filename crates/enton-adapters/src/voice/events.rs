use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

pub(super) use enton_core::UtteranceId;
use serde::{Deserialize, Serialize};

use super::config::VoiceError;

/// Playback lifecycle events emitted by the voice player.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlaybackEvent {
    /// Playback of the utterance has started on the audio device.
    Started {
        /// Utterance identifier.
        id: UtteranceId,
        /// Full text of the spoken sentence.
        text: String,
    },
    /// Playback of the utterance completed normally.
    Finished {
        /// Utterance identifier.
        id: UtteranceId,
    },
    /// Synthesis or playback admission failed after accepting an utterance.
    Failed {
        /// Identifier returned by the successful submission.
        id: UtteranceId,
        /// Failure description, excluding the spoken text.
        reason: String,
    },
    /// Playback or queued utterance was cancelled (e.g. by barge-in).
    Cancelled {
        /// Utterance identifier.
        id: UtteranceId,
    },
}

/// Monotonic timestamps recorded across the lifecycle of an utterance in the voice pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UtteranceStageTimings {
    /// Utterance identifier.
    pub id: UtteranceId,
    /// Generation counter at time of submission.
    pub generation: u64,
    /// Instant when the chunk/sentence was submitted to the player (`speak`).
    pub handed_to_tts: std::time::Instant,
    /// Instant when native TTS synthesis finished producing audio samples.
    pub synthesis_done: Option<std::time::Instant>,
    /// Instant when the first audio sample was dispatched to the hardware audio buffer.
    pub first_sample_played: Option<std::time::Instant>,
    /// Instant when playback finished.
    pub playback_finished: Option<std::time::Instant>,
}

/// Consolidated latency breakdown across cortex deliberation and voice playback.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoiceLatencyBreakdown {
    /// Duration from thought request dispatch to first token received from cortex.
    pub time_to_first_token: Option<Duration>,
    /// Duration from first token to first chunk handed to TTS.
    pub time_to_first_chunk: Option<Duration>,
    /// Duration spent synthesizing the first chunk in Kokoro TTS.
    pub synthesis_duration: Option<Duration>,
    /// Duration from synthesis completion to first audio sample output on device.
    pub audio_device_dispatch: Option<Duration>,
    /// Total time to first audio (TTFA) from thought request dispatch to first sample played.
    pub total_time_to_first_audio: Option<Duration>,
}

impl VoiceLatencyBreakdown {
    /// Computes the latency breakdown from stage time points.
    #[must_use]
    pub fn compute(
        request_start: std::time::Instant,
        first_token: Option<std::time::Instant>,
        handed_to_tts: std::time::Instant,
        synthesis_done: Option<std::time::Instant>,
        first_sample_played: Option<std::time::Instant>,
    ) -> Self {
        let time_to_first_token =
            first_token.map(|tok| tok.saturating_duration_since(request_start));
        let time_to_first_chunk =
            first_token.map(|tok| handed_to_tts.saturating_duration_since(tok));
        let synthesis_duration =
            synthesis_done.map(|syn| syn.saturating_duration_since(handed_to_tts));
        let audio_device_dispatch = match (synthesis_done, first_sample_played) {
            (Some(syn), Some(play)) => Some(play.saturating_duration_since(syn)),
            _ => None,
        };
        let total_time_to_first_audio =
            first_sample_played.map(|play| play.saturating_duration_since(request_start));

        Self {
            time_to_first_token,
            time_to_first_chunk,
            synthesis_duration,
            audio_device_dispatch,
            total_time_to_first_audio,
        }
    }
}

pub(super) const EVENT_CAPACITY: usize = 64;
const MAX_SUBSCRIBERS: usize = 16;
pub(super) const MAX_PLAYBACK_QUEUE: usize = 64;
// One active utterance can finish, and each queued item can start and finish.
pub(super) const CALLBACK_EVENTS: usize = 1 + 2 * MAX_PLAYBACK_QUEUE;

/// Cumulative loss counters for the bounded playback event transport.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PlaybackEventStats {
    /// New events dropped because the dispatch mailbox was full or disconnected.
    pub mailbox_dropped: u64,
    /// New deliveries dropped because a subscriber's 64-event queue was full.
    pub subscriber_dropped: u64,
    /// Disconnected subscribers removed during delivery.
    pub subscribers_disconnected: u64,
}

#[derive(Default)]
pub(super) struct EventSubscribers {
    pub(super) senders: Mutex<Vec<SyncSender<PlaybackEvent>>>,
    pub(super) mailbox_dropped: AtomicU64,
    pub(super) subscriber_dropped: AtomicU64,
    pub(super) subscribers_disconnected: AtomicU64,
}

impl EventSubscribers {
    pub(super) fn publish(&self, event: &PlaybackEvent) {
        self.senders
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|sender| match sender.try_send(event.clone()) {
                Ok(()) => true,
                Err(TrySendError::Full(_)) => {
                    self.subscriber_dropped.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(TrySendError::Disconnected(_)) => {
                    self.subscribers_disconnected
                        .fetch_add(1, Ordering::Relaxed);
                    false
                }
            });
    }
}

pub(super) struct EventHub {
    pub(super) sender: SyncSender<PlaybackEvent>,
    pub(super) state: Arc<EventSubscribers>,
}

impl EventHub {
    pub(super) fn new() -> Self {
        let (sender, receiver) = sync_channel::<PlaybackEvent>(EVENT_CAPACITY);
        let state = Arc::new(EventSubscribers::default());
        let publishing = Arc::clone(&state);
        // Cloning text and subscriber fan-out happen here, never on the audio callback.
        thread::spawn(move || {
            while let Ok(event) = receiver.recv() {
                publishing.publish(&event);
            }
        });
        Self { sender, state }
    }

    pub(super) fn subscribe(&self) -> Result<Receiver<PlaybackEvent>, VoiceError> {
        let (sender, receiver) = sync_channel(EVENT_CAPACITY);
        let mut senders = self
            .state
            .senders
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if senders.len() >= MAX_SUBSCRIBERS {
            return Err(VoiceError::SubscriberLimit);
        }
        senders.push(sender);
        drop(senders);
        Ok(receiver)
    }

    pub(super) fn stats(&self) -> PlaybackEventStats {
        PlaybackEventStats {
            mailbox_dropped: self.state.mailbox_dropped.load(Ordering::Relaxed),
            subscriber_dropped: self.state.subscriber_dropped.load(Ordering::Relaxed),
            subscribers_disconnected: self.state.subscribers_disconnected.load(Ordering::Relaxed),
        }
    }
}

pub(super) fn emit_event(hub: &EventHub, event: PlaybackEvent) {
    if hub.sender.try_send(event).is_err() {
        hub.state.mailbox_dropped.fetch_add(1, Ordering::Relaxed);
    }
}

pub(super) struct EventBatch {
    events: [Option<PlaybackEvent>; CALLBACK_EVENTS],
    length: usize,
}

impl EventBatch {
    pub(super) fn new() -> Self {
        Self {
            events: std::array::from_fn(|_| None),
            length: 0,
        }
    }

    pub(super) fn push(&mut self, event: PlaybackEvent, hub: &EventHub) {
        if let Some(slot) = self.events.get_mut(self.length) {
            *slot = Some(event);
            self.length += 1;
        } else {
            // Defensive accounting if a future producer violates the playback bound.
            hub.state.mailbox_dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn publish(self, hub: &EventHub) {
        for event in self.events.into_iter().flatten() {
            emit_event(hub, event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utterance_id_formatting_and_ordering() {
        let id1 = UtteranceId(1);
        let id2 = UtteranceId(2);
        assert_eq!(format!("{id1}"), "#1");
        assert!(id1 < id2);
    }

    #[test]
    fn test_voice_latency_breakdown_compute() {
        let t0 = std::time::Instant::now();
        let t1_token = t0 + Duration::from_millis(120);
        let t2_chunk = t0 + Duration::from_millis(180);
        let t3_synth = t0 + Duration::from_millis(260);
        let t4_sample = t0 + Duration::from_millis(280);

        let breakdown = VoiceLatencyBreakdown::compute(
            t0,
            Some(t1_token),
            t2_chunk,
            Some(t3_synth),
            Some(t4_sample),
        );

        assert_eq!(
            breakdown.time_to_first_token,
            Some(Duration::from_millis(120))
        );
        assert_eq!(
            breakdown.time_to_first_chunk,
            Some(Duration::from_millis(60))
        );
        assert_eq!(
            breakdown.synthesis_duration,
            Some(Duration::from_millis(80))
        );
        assert_eq!(
            breakdown.audio_device_dispatch,
            Some(Duration::from_millis(20))
        );
        assert_eq!(
            breakdown.total_time_to_first_audio,
            Some(Duration::from_millis(280))
        );
    }

    #[test]
    fn event_overflow_and_disconnection_are_counted_without_blocking() {
        let hub = EventHub::new();
        let disconnected = hub.subscribe().unwrap();
        drop(disconnected);
        hub.state
            .publish(&PlaybackEvent::Finished { id: UtteranceId(0) });
        assert_eq!(hub.stats().subscribers_disconnected, 1);

        // A mailbox with no consumer makes overflow deterministic without timing.
        let (sender, receiver) = sync_channel(1);
        let hub = EventHub {
            sender,
            state: Arc::new(EventSubscribers::default()),
        };
        emit_event(&hub, PlaybackEvent::Finished { id: UtteranceId(1) });
        emit_event(&hub, PlaybackEvent::Finished { id: UtteranceId(2) });
        assert_eq!(hub.stats().mailbox_dropped, 1);
        assert_eq!(
            receiver.try_recv().unwrap(),
            PlaybackEvent::Finished { id: UtteranceId(1) }
        );
    }

    #[test]
    fn subscriber_registration_is_bounded() {
        let hub = EventHub::new();
        let receivers: Vec<_> = (0..MAX_SUBSCRIBERS)
            .map(|_| hub.subscribe().unwrap())
            .collect();
        assert!(matches!(hub.subscribe(), Err(VoiceError::SubscriberLimit)));
        drop(receivers);
        hub.state
            .publish(&PlaybackEvent::Finished { id: UtteranceId(1) });
        assert_eq!(hub.stats().subscribers_disconnected, MAX_SUBSCRIBERS as u64);
        assert!(hub.subscribe().is_ok());
    }
}
