use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{FromSample, SizedSample};
use enton_core::UtteranceId;

use super::config::VoiceError;
use super::events::{EventBatch, EventHub, PlaybackEvent, UtteranceStageTimings};

/// A queued sentence ready for audio playback.
pub(super) struct QueuedSentence {
    pub(super) id: UtteranceId,
    pub(super) generation: u64,
    pub(super) text: String,
    pub(super) samples: Vec<f32>,
}

/// Internal mutable state shared between worker threads and audio callback.
pub(super) struct PlaybackQueueState {
    pub(super) generation: u64,
    pub(super) current_utterance: Option<UtteranceId>,
    pub(super) current_samples: Vec<f32>,
    pub(super) current_pos: usize,
    pub(super) queue: VecDeque<QueuedSentence>,
    pub(super) timings: VecDeque<UtteranceStageTimings>,
}

pub(super) fn open_output_device(
    device_id: Option<&str>,
) -> Result<(cpal::Device, cpal::SupportedStreamConfig), VoiceError> {
    let host = cpal::default_host();
    let device = match device_id {
        Some(target_id) => host
            .output_devices()
            .map_err(|source| VoiceError::Device {
                operation: "failed to query output devices",
                source,
            })?
            .find(|d| d.id().is_ok_and(|id| id.to_string() == target_id))
            .ok_or(VoiceError::NoOutputDevice(Some(target_id.to_owned())))?,
        None => host
            .default_output_device()
            .ok_or(VoiceError::NoOutputDevice(None))?,
    };

    let supported_config = device
        .default_output_config()
        .map_err(|source| VoiceError::Device {
            operation: "failed to query default output config",
            source,
        })?;

    Ok((device, supported_config))
}

pub(super) fn build_cpal_stream<T>(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    channels: usize,
    queue_state: Arc<Mutex<PlaybackQueueState>>,
    subscribers: Arc<EventHub>,
    last_error: Arc<Mutex<Option<VoiceError>>>,
) -> Result<cpal::Stream, VoiceError>
where
    T: SizedSample + FromSample<f32>,
{
    let err_fn = move |source| {
        *last_error
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(VoiceError::Device {
            operation: "output stream",
            source,
        });
    };

    let stream = device
        .build_output_stream(
            (*config).into(),
            move |data: &mut [T], _| {
                process_output_callback(data, channels, &queue_state, &subscribers);
            },
            err_fn,
            None,
        )
        .map_err(|source| VoiceError::Device {
            operation: "failed to build cpal output stream",
            source,
        })?;

    Ok(stream)
}

pub(super) fn process_output_callback<T>(
    data: &mut [T],
    channels: usize,
    queue_state: &Arc<Mutex<PlaybackQueueState>>,
    subscribers: &Arc<EventHub>,
) where
    T: SizedSample + FromSample<f32>,
{
    let Ok(mut state) = queue_state.try_lock() else {
        for sample in data.iter_mut() {
            *sample = T::from_sample_(0.0);
        }
        return;
    };

    // Device configurations supply nonzero channels. Keep malformed input total.
    if channels == 0 {
        drop(state);
        data.fill(T::from_sample_(0.0));
        return;
    }
    let mut events = EventBatch::new();
    let mut frames = data.chunks_exact_mut(channels);
    for frame in frames.by_ref() {
        if state.current_pos >= state.current_samples.len() {
            if let Some(id) = state.current_utterance.take() {
                let finish_now = std::time::Instant::now();
                if let Some(timing) = state.timings.iter_mut().find(|t| t.id == id) {
                    timing.playback_finished = Some(finish_now);
                }
                events.push(PlaybackEvent::Finished { id }, subscribers);
            }
            state.current_samples.clear();
            state.current_pos = 0;

            let mut next_item = None;
            while let Some(item) = state.queue.pop_front() {
                if item.generation == state.generation {
                    next_item = Some(item);
                    break;
                }
                events.push(PlaybackEvent::Cancelled { id: item.id }, subscribers);
            }

            if let Some(item) = next_item {
                let play_now = std::time::Instant::now();
                if let Some(timing) = state.timings.iter_mut().find(|t| t.id == item.id) {
                    timing.first_sample_played = Some(play_now);
                }
                events.push(
                    PlaybackEvent::Started {
                        id: item.id,
                        text: item.text,
                    },
                    subscribers,
                );
                state.current_utterance = Some(item.id);
                state.current_samples = item.samples;
                state.current_pos = 0;
            } else {
                frame.fill(T::from_sample_(0.0));
                break;
            }
        }

        let sample = state
            .current_samples
            .get(state.current_pos)
            .copied()
            .unwrap_or(0.0);
        state.current_pos += 1;
        frame.fill(T::from_sample_(sample));
    }
    drop(state);
    events.publish(subscribers);
    for frame in frames {
        frame.fill(T::from_sample_(0.0));
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn callback_preserves_stereo_samples_silence_and_lifecycle_order() {
        let subscribers = Arc::new(EventHub::new());
        let rx = subscribers.subscribe().unwrap();
        let state = Arc::new(Mutex::new(PlaybackQueueState {
            generation: 1,
            current_utterance: None,
            current_samples: Vec::new(),
            current_pos: 0,
            queue: VecDeque::from([
                QueuedSentence {
                    id: UtteranceId(1),
                    generation: 0,
                    text: "stale".to_owned(),
                    samples: vec![0.9],
                },
                QueuedSentence {
                    id: UtteranceId(2),
                    generation: 1,
                    text: "first".to_owned(),
                    samples: vec![0.25, -0.5],
                },
                QueuedSentence {
                    id: UtteranceId(3),
                    generation: 1,
                    text: "second".to_owned(),
                    samples: vec![0.75],
                },
            ]),
            timings: VecDeque::new(),
        }));
        let mut data = [1.0_f32; 10];
        process_output_callback(&mut data, 2, &state, &subscribers);
        assert_eq!(
            data.map(f32::to_bits),
            [0.25, 0.25, -0.5, -0.5, 0.75, 0.75, 0.0, 0.0, 0.0, 0.0].map(f32::to_bits)
        );
        assert_eq!(
            (0..5)
                .map(|_| rx.recv_timeout(Duration::from_secs(2)).unwrap())
                .collect::<Vec<_>>(),
            vec![
                PlaybackEvent::Cancelled { id: UtteranceId(1) },
                PlaybackEvent::Started {
                    id: UtteranceId(2),
                    text: "first".to_owned()
                },
                PlaybackEvent::Finished { id: UtteranceId(2) },
                PlaybackEvent::Started {
                    id: UtteranceId(3),
                    text: "second".to_owned()
                },
                PlaybackEvent::Finished { id: UtteranceId(3) },
            ]
        );
    }

    #[test]
    fn callback_defers_finished_until_the_frame_after_the_last_sample() {
        let subscribers = Arc::new(EventHub::new());
        let rx = subscribers.subscribe().unwrap();
        let state = Arc::new(Mutex::new(PlaybackQueueState {
            generation: 0,
            current_utterance: Some(UtteranceId(1)),
            current_samples: vec![0.25],
            current_pos: 0,
            queue: VecDeque::new(),
            timings: VecDeque::new(),
        }));
        let mut data = [1.0_f32; 2];
        process_output_callback(&mut data, 2, &state, &subscribers);
        assert_eq!(data.map(f32::to_bits), [0.25_f32.to_bits(); 2]);
        assert!(rx.try_recv().is_err());
        process_output_callback(&mut data, 2, &state, &subscribers);
        assert_eq!(data.map(f32::to_bits), [0.0_f32.to_bits(); 2]);
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(2)).unwrap(),
            PlaybackEvent::Finished { id: UtteranceId(1) }
        );
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn callback_outputs_silence_without_advancing_when_queue_is_locked() {
        let subscribers = Arc::new(EventHub::new());
        let state = Arc::new(Mutex::new(PlaybackQueueState {
            generation: 0,
            current_utterance: Some(UtteranceId(1)),
            current_samples: vec![0.25],
            current_pos: 0,
            queue: VecDeque::new(),
            timings: VecDeque::new(),
        }));
        let guard = state.lock().unwrap();
        let mut data = [1.0_f32; 2];
        process_output_callback(&mut data, 2, &state, &subscribers);
        assert_eq!(guard.current_pos, 0);
        drop(guard);
        assert_eq!(data.map(f32::to_bits), [0.0_f32.to_bits(); 2]);
    }

    #[test]
    fn callback_does_not_wait_for_slow_event_subscribers() {
        let hub = Arc::new(EventHub::new());
        let receiver = hub.subscribe().unwrap();
        let subscribers = hub.state.senders.lock().unwrap();
        let state = Arc::new(Mutex::new(PlaybackQueueState {
            generation: 0,
            current_utterance: Some(UtteranceId(1)),
            current_samples: Vec::new(),
            current_pos: 0,
            queue: VecDeque::new(),
            timings: VecDeque::new(),
        }));
        let mut samples = [1.0_f32; 2];
        process_output_callback(&mut samples, 2, &state, &hub);
        assert!(state.try_lock().is_ok());
        assert_eq!(samples.map(f32::to_bits), [0.0_f32.to_bits(); 2]);
        drop(subscribers);
        assert_eq!(
            receiver.recv_timeout(Duration::from_secs(2)).unwrap(),
            PlaybackEvent::Finished { id: UtteranceId(1) }
        );
    }
}
