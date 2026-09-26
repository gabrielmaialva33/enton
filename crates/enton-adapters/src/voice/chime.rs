//! The acknowledgement chime: two soft bell tones, rising a fourth, that Enton plays
//! when it heard its name and waits for the rest of the request. Computed once, at
//! the output rate, so playing it costs no synthesis and needs no file.

use std::f32::consts::TAU;

/// The whole chime, in seconds.
const LENGTH_S: f32 = 0.2;

/// Each tone: pitch in hertz (G5, then C6), start and length in seconds, and how fast
/// it decays (the time constant of its exponential envelope, in seconds).
const TONES: [Tone; 2] = [
    Tone {
        hz: 783.99,
        start_s: 0.0,
        length_s: 0.1,
        decay_s: 0.045,
    },
    Tone {
        hz: 1_046.5,
        start_s: 0.08,
        length_s: 0.12,
        decay_s: 0.055,
    },
];

/// Peak amplitude of one tone: soft, well under the level of speech.
const PEAK: f32 = 0.22;
/// Weight of the octave above each tone, which makes it a bell instead of a beep.
const OCTAVE: f32 = 0.2;
/// Attack and release of each tone, in seconds: long enough that no edge clicks.
const RAMP_S: f32 = 0.005;

struct Tone {
    hz: f32,
    start_s: f32,
    length_s: f32,
    decay_s: f32,
}

impl Tone {
    /// The tone's value at `t` seconds from the start of the chime.
    fn at(&self, t: f32) -> f32 {
        let local = t - self.start_s;
        if !(0.0..self.length_s).contains(&local) {
            return 0.0;
        }
        let attack = (local / RAMP_S).min(1.0);
        let release = ((self.length_s - local) / RAMP_S).min(1.0);
        let envelope = attack * release * (-local / self.decay_s).exp();
        let phase = TAU * self.hz * local;
        PEAK * envelope * (phase.sin() + OCTAVE * (2.0 * phase).sin()) / (1.0 + OCTAVE)
    }
}

/// The chime as mono samples at `sample_rate` hertz: about 200 ms, silent at both ends.
#[must_use]
pub(super) fn chime(sample_rate: u32) -> Vec<f32> {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a positive length of a fraction of a second at an audio rate fits usize"
    )]
    let count = (LENGTH_S * sample_rate as f32).round() as usize;
    let rate = sample_rate as f32;
    (0..count)
        .map(|n| {
            let t = n as f32 / rate;
            TONES.iter().map(|tone| tone.at(t)).sum::<f32>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Power of `samples` at `hz` (the Goertzel algorithm), per sample.
    fn power(samples: &[f32], hz: f32, rate: f32) -> f32 {
        let coefficient = 2.0 * (TAU * hz / rate).cos();
        let (mut previous, mut before) = (0.0_f32, 0.0_f32);
        for sample in samples {
            let current = sample + coefficient * previous - before;
            before = previous;
            previous = current;
        }
        (previous * previous + before * before - coefficient * previous * before)
            / samples.len() as f32
    }

    #[test]
    fn the_chime_is_short_soft_and_silent_at_both_ends() {
        for rate in [16_000, 22_050, 24_000, 44_100, 48_000] {
            let samples = chime(rate);
            assert_eq!(samples.len(), (rate as usize) / 5, "{rate} Hz");
            assert!(samples.iter().all(|s| s.is_finite()));
            let peak = samples.iter().fold(0.0_f32, |peak, s| peak.max(s.abs()));
            assert!((0.1..=0.3).contains(&peak), "peak {peak} at {rate} Hz");
            assert_eq!(samples.first().copied(), Some(0.0));
            let tail = samples.last().copied().unwrap_or(1.0);
            assert!(tail.abs() < 1e-3, "tail {tail} at {rate} Hz");
        }
    }

    #[test]
    fn the_chime_rises_from_g5_to_c6() {
        let rate = 48_000.0;
        let samples = chime(48_000);
        // First 70 ms: only the lower tone plays. Last 90 ms: only the higher one.
        let (first, last) = (&samples[..3_360], &samples[5_280..]);
        assert!(power(first, 783.99, rate) > 20.0 * power(first, 1_046.5, rate));
        assert!(power(last, 1_046.5, rate) > 20.0 * power(last, 783.99, rate));
    }

    #[test]
    fn the_same_rate_gives_the_same_chime() {
        assert_eq!(
            chime(24_000)
                .iter()
                .map(|s| s.to_bits())
                .collect::<Vec<_>>(),
            chime(24_000)
                .iter()
                .map(|s| s.to_bits())
                .collect::<Vec<_>>()
        );
    }
}
