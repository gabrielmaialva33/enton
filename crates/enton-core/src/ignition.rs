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
        self.drive_pressing() && !self.in_cooldown(now)
    }

    /// Whether a drive presses to fire, the cooldown aside: armed and at or above the
    /// threshold. Holding its intent to ride the owner's next request needs only this,
    /// since riding a thought the owner asked for buys no thought of its own.
    #[must_use]
    pub fn drive_pressing(&self) -> bool {
        self.armed && self.smoothed >= self.threshold
    }

    /// Whether a successful thought is still inside its cooldown window.
    #[must_use]
    pub fn in_cooldown(&self, now: Millis) -> bool {
        self.last_fire
            .is_some_and(|last| now.since(last) < self.cooldown_ms)
    }

    /// Record a drive's thought, or a thought its deferred intent rides: it starts the shared
    /// cooldown and consumes the hysteresis, so the drive does not ask again while that
    /// thought is out. Rejected attempts do not consume either.
    pub fn fired(&mut self, now: Millis) {
        self.armed = false;
        self.paid(now);
    }

    /// Record any other paid thought (a keyword, a follow-up, overheard speech): it
    /// starts the shared cooldown and leaves the drive hysteresis alone, because only a
    /// drive's own thought answers its pressure. Keywords bypass the cooldown in the caller.
    pub fn paid(&mut self, now: Millis) {
        self.last_fire = Some(self.last_fire.map_or(now, |last| last.max(now)));
    }

    /// A drive's thought ended without answering the pressure (it failed, or a newer
    /// thought superseded it): the drive may ask again once the cooldown, and whatever
    /// else holds it back, allows.
    pub fn rearm(&mut self) {
        self.armed = true;
    }

    /// A drive's thought was answered and its drive satisfied: the smoothed pressure
    /// drops at once to what is left, `pressure`, and the ignition rearms. What is left
    /// fires again only once it reaches the threshold, so an answered pressure never
    /// fires twice and one that nothing answers never locks the drives out.
    pub fn settle(&mut self, pressure: f32) {
        self.smoothed = pressure;
        self.armed = true;
    }
}
