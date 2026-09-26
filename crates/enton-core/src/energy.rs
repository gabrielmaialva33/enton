//! A bounded budget that refills over time.

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
}

impl Budget {
    /// Create a full budget; a negative or non-finite capacity becomes zero.
    #[must_use]
    pub fn new(capacity: f32) -> Self {
        let capacity = if capacity.is_finite() {
            capacity.max(0.0)
        } else {
            0.0
        };
        Self {
            capacity,
            available: capacity,
        }
    }

    /// Refill at a finite, positive hourly rate, without exceeding capacity.
    pub fn refill(&mut self, dt_ms: u64, per_hour: f32) {
        if per_hour.is_finite() && per_hour > 0.0 {
            let amount = dt_ms as f32 / 3_600_000.0 * per_hour;
            self.available = (self.available + amount).min(self.capacity);
        }
    }

    /// Whether [`Budget::try_spend`] would accept `cost` now, without spending it.
    #[must_use]
    pub fn can_spend(&self, cost: f32) -> bool {
        cost.is_finite() && cost >= 0.0 && self.available - cost >= 0.0
    }

    /// Spend atomically; a cost the balance cannot cover, or an invalid or
    /// negative cost, is rejected without changing the balance.
    #[must_use]
    pub fn try_spend(&mut self, cost: f32) -> bool {
        if !self.can_spend(cost) {
            return false;
        }
        self.available -= cost;
        true
    }
}
