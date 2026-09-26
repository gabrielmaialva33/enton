//! Smoothed drive ignition, hysteresis, and a shared thought cooldown.

use serde::{Deserialize, Serialize};

use crate::Millis;

/// Drive hysteresis is separate from speech salience: a new utterance can
/// fire after cooldown without first waiting for an internal drive to decay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ignition {
    threshold: f32,
    hysteresis: f32,
    cooldown_ms: u64,
    ema_alpha: f32,
    smoothed: f32,
    armed: bool,
    last_fire: Option<Millis>,
}

impl Ignition {
    /// Configure ignition with validated profile parameters.
    #[must_use]
    pub fn new(threshold: f32, hysteresis: f32, cooldown_ms: u64, ema_alpha: f32) -> Self {
        Self {
            threshold,
            hysteresis,
            cooldown_ms,
            ema_alpha,
            smoothed: 0.0,
            armed: true,
            last_fire: None,
        }
    }

    /// Update the drive EMA and rearm only strictly below the lower threshold.
    pub fn advance(&mut self, pressure: f32) {
        self.smoothed += self.ema_alpha * (pressure - self.smoothed);
        if self.smoothed < self.threshold - self.hysteresis {
            self.armed = true;
        }
    }

    /// Current drive salience, after smoothing.
    #[must_use]
    pub fn salience(&self) -> f32 {
        self.smoothed
    }

    /// Whether a drive would fire before checking torpor and energy.
    #[must_use]
    pub fn drive_ready(&self, now: Millis) -> bool {
        self.armed && self.smoothed >= self.threshold && !self.in_cooldown(now)
    }

    /// Whether a successful thought is still inside its cooldown window.
    #[must_use]
    pub fn in_cooldown(&self, now: Millis) -> bool {
        self.last_fire
            .is_some_and(|last| now.since(last) < self.cooldown_ms)
    }

    /// Record a paid thought, including keywords. Rejected attempts do not
    /// consume cooldown or hysteresis. Keywords bypass both checks in the caller.
    pub fn fired(&mut self, now: Millis) {
        self.armed = false;
        self.last_fire = Some(self.last_fire.map_or(now, |last| last.max(now)));
    }
}
