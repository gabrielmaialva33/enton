//! Data-driven internal pressures, advanced only by explicit elapsed time.

use serde::{Deserialize, Serialize};

/// A homeostatic drive with a normalized level and nonnegative parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Drive {
    /// The unique identifier for this drive.
    pub name: String,
    /// Current pressure, in the inclusive range zero to one.
    pub level: f32,
    /// Weight applied when computing total pressure.
    pub weight: f32,
    /// Amount added per minute of elapsed time.
    pub growth_per_min: f32,
}

/// An ordered drive table. Equal contributions favor the first drive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DriveTable(Vec<Drive>);

impl DriveTable {
    /// Build a table from drives with finite, nonnegative parameters.
    #[must_use]
    pub fn new(drives: Vec<Drive>) -> Self {
        Self(drives)
    }

    /// Conservative scaffold defaults, not calibrated physiological values.
    /// After one hour their combined pressure is below both profile thresholds.
    #[must_use]
    pub fn default_m1() -> Self {
        Self::new(
            [
                ("curiosity", 0.6, 0.004),
                ("social", 0.6, 0.003),
                ("rest", 0.2, 0.002),
            ]
            .into_iter()
            .map(|(name, weight, growth_per_min)| Drive {
                name: name.to_owned(),
                level: 0.0,
                weight,
                growth_per_min,
            })
            .collect(),
        )
    }

    /// Advance every drive by elapsed milliseconds, saturating at one.
    pub fn advance(&mut self, dt_ms: u64) {
        let minutes = dt_ms as f32 / 60_000.0;
        for drive in &mut self.0 {
            drive.level = (drive.level + drive.growth_per_min * minutes).clamp(0.0, 1.0);
        }
    }

    /// Reduce a named drive; unknown names and nonpositive amounts are no-ops.
    pub fn satisfy(&mut self, name: &str, amount: f32) {
        if !amount.is_finite() || amount <= 0.0 {
            return;
        }
        for drive in &mut self.0 {
            if drive.name == name {
                drive.level = (drive.level - amount).max(0.0);
            }
        }
    }

    /// Sum of each drive's weighted squared level.
    #[must_use]
    pub fn pressure(&self) -> f32 {
        self.0.iter().map(contribution).sum()
    }

    /// The drive making the largest contribution to total pressure.
    #[must_use]
    pub fn strongest(&self) -> Option<&Drive> {
        self.0.iter().reduce(|strongest, candidate| {
            if contribution(candidate) > contribution(strongest) {
                candidate
            } else {
                strongest
            }
        })
    }

    /// Drives in their stable evaluation order.
    #[must_use]
    pub fn as_slice(&self) -> &[Drive] {
        &self.0
    }
}

fn contribution(drive: &Drive) -> f32 {
    drive.weight * drive.level * drive.level
}
