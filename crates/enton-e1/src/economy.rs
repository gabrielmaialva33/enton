//! Identical outer accounts, independent of labels and policy ledger allocation.
use crate::Error;
use enton_core::{Millis, Profile};
use serde::Serialize;

/// Common cost contract derived once from the selected core profile.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Economy {
    /// Full initial balance and capacity, in abstract units.
    pub capacity: f64,
    /// Units supplied per elapsed hour to every controller.
    pub refill_per_hour: f64,
    /// Identical cost per paid thought.
    pub thought_cost: f64,
}
impl Economy {
    pub(crate) fn from_profile(profile: &Profile) -> Result<Self, Error> {
        let capacity = f64::from(profile.budgets.obligation_budget_per_hour)
            + f64::from(profile.budgets.discretionary_budget_per_hour);
        let thought_cost = f64::from(profile.budgets.think_cost);
        if !capacity.is_finite()
            || capacity <= 0.0
            || !thought_cost.is_finite()
            || thought_cost <= 0.0
        {
            return Err(Error::Invalid(
                "positive finite common budget and price required".into(),
            ));
        }
        Ok(Self {
            capacity,
            refill_per_hour: capacity,
            thought_cost,
        })
    }
}

pub(crate) struct Account {
    economy: Economy,
    last: Millis,
    pub(crate) balance: f64,
    pub(crate) credited_refill: f64,
    pub(crate) rounding_credit: f64,
}
impl Account {
    pub(crate) fn new(economy: Economy) -> Self {
        Self {
            economy,
            last: Millis(0),
            balance: economy.capacity,
            credited_refill: 0.0,
            rounding_credit: 0.0,
        }
    }
    pub(crate) fn advance(&mut self, now: Millis) {
        let increment = now.since(self.last) as f64 / 3_600_000.0 * self.economy.refill_per_hour;
        self.last = self.last.max(now);
        let updated = (self.balance + increment).min(self.economy.capacity);
        self.credited_refill += updated - self.balance;
        self.balance = updated;
    }
    pub(crate) fn can_pay(&self) -> bool {
        self.balance + self.economy.thought_cost * 0.0001 >= self.economy.thought_cost
    }
    pub(crate) fn pay(&mut self) -> Result<(), Error> {
        if !self.can_pay() {
            return Err(Error::BudgetInvariant {
                balance: self.balance,
                cost: self.economy.thought_cost,
            });
        }
        let difference = self.balance - self.economy.thought_cost;
        self.rounding_credit += (-difference).max(0.0);
        self.balance = difference.max(0.0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn equal_accounts_exhaust_refill_and_conserve_cost() {
        let economy = Economy {
            capacity: 2.0,
            refill_per_hour: 2.0,
            thought_cost: 1.0,
        };
        for _controller in 0..3 {
            let mut a = Account::new(economy);
            a.pay().unwrap();
            a.pay().unwrap();
            assert!(!a.can_pay());
            assert!(a.pay().is_err());
            a.advance(Millis(1_800_000));
            assert!(a.can_pay());
            a.pay().unwrap();
            a.advance(Millis(1_000));
            assert!(!a.can_pay());
            assert!(
                (economy.capacity + a.credited_refill + a.rounding_credit - 3.0 - a.balance).abs()
                    < 1e-9
            );
        }
    }
}
