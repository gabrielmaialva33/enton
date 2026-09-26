//! Monotonic clock adapter measuring elapsed time since construction.

use std::time::Instant;

use enton_core::Millis;

/// A monotonic clock returning [`Millis`] elapsed since construction.
#[derive(Debug, Clone, Copy)]
pub struct MonotonicClock {
    start: Instant,
}

impl MonotonicClock {
    /// Creates a new [`MonotonicClock`] starting at the current instant.
    #[must_use]
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Returns the monotonic elapsed time in milliseconds since this clock was constructed.
    #[must_use]
    pub fn now(&self) -> Millis {
        let elapsed_ms = self.start.elapsed().as_millis();
        let ms_u64 = u64::try_from(elapsed_ms).unwrap_or(u64::MAX);
        Millis(ms_u64)
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
    fn clock_default_creates_instance() {
        let clock = MonotonicClock::default();
        let now = clock.now();
        assert_eq!(now, clock.now());
    }
}
