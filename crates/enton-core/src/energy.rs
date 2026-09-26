//! A bounded budget with a protected keyword reserve.

use serde::{Deserialize, Serialize};

/// Prices in abstract scaffold budget units, not measured joules.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PriceTable {
    /// Cost to ignite a thought.
    pub think: f32,
}

/// Budget amounts in the same units as [`PriceTable`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    /// Maximum amount that can be held.
    pub capacity: f32,
    /// Spendable balance, decreases on use, refilled over time.
    pub available: f32,
    /// Protected balance, expressed in budget units rather than a fraction.
    pub reserve: f32,
}

impl Budget {
    /// Create a full budget; clamp the reserve to the available capacity.
    #[must_use]
    pub fn new(capacity: f32, reserve: f32) -> Self {
        let capacity = if capacity.is_finite() {
            capacity.max(0.0)
        } else {
            0.0
        };
        Self {
            capacity,
            available: capacity,
            reserve: reserve.max(0.0).min(capacity),
        }
    }

    /// Refill at a finite, positive hourly rate, without exceeding capacity.
    pub fn refill(&mut self, dt_ms: u64, per_hour: f32) {
        if per_hour.is_finite() && per_hour > 0.0 {
            let amount = dt_ms as f32 / 3_600_000.0 * per_hour;
            self.available = (self.available + amount).min(self.capacity);
        }
    }

    /// Spend atomically, preserving the reserve unless explicitly allowed.
    /// Invalid or negative costs are rejected without changing the balance.
    #[must_use]
    pub fn try_spend(&mut self, cost: f32, allow_reserve: bool) -> bool {
        if !cost.is_finite() || cost < 0.0 {
            return false;
        }
        let remaining = self.available - cost;
        let floor = if allow_reserve { 0.0 } else { self.reserve };
        if remaining < floor {
            return false;
        }
        self.available = remaining;
        true
    }
}
