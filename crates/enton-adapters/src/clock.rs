//! Monotonic clock adapter measuring elapsed time since construction, optionally
//! resuming from the last instant a restored organism observed.

use std::time::Instant;

use enton_core::Millis;

/// A monotonic clock returning [`Millis`] elapsed since construction, plus an origin.
#[derive(Debug, Clone, Copy)]
pub struct MonotonicClock {
    start: Instant,
    origin: Millis,
}

impl MonotonicClock {
    /// Creates a new [`MonotonicClock`] reading zero at the current instant.
    #[must_use]
    pub fn new() -> Self {
        Self::resuming_at(Millis(0))
    }

    /// Creates a clock that continues from `origin` instead of zero.
    ///
    /// After restoring an organism, pass
    /// [`Organism::last_seen`](enton_core::Organism::last_seen): cooldowns,
    /// attention windows and budget refills keep counting from where they
    /// stopped. Time spent while the process was down is not credited.
    #[must_use]
    pub fn resuming_at(origin: Millis) -> Self {
        Self {
            start: Instant::now(),
            origin,
        }
    }

    /// Returns the origin plus the monotonic time elapsed since construction.
    #[must_use]
    pub fn now(&self) -> Millis {
        let elapsed_ms = u64::try_from(self.start.elapsed().as_millis()).unwrap_or(u64::MAX);
        Millis(self.origin.0.saturating_add(elapsed_ms))
    }
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn clock_advances_monotonically() {
        let clock = MonotonicClock::new();
        let t1 = clock.now();
        thread::sleep(Duration::from_millis(5));
        let t2 = clock.now();
        assert!(t2.0 >= t1.0);
    }

    #[test]
    fn resumed_clock_never_reads_behind_its_origin() {
        let origin = Millis(5 * 3_600_000);
        let clock = MonotonicClock::resuming_at(origin);
        assert!(clock.now() >= origin);
        assert_eq!(
            MonotonicClock::resuming_at(Millis(u64::MAX)).now(),
            Millis(u64::MAX)
        );
    }

    #[test]
    fn clock_default_creates_instance() {
        let clock = MonotonicClock::default();
        let now = clock.now();
        assert_eq!(now, clock.now());
    }
}
