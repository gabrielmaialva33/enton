//! Boundary checks for the budget, drive table, and ignition state machine.

use enton_core::{Budget, Drive, DriveTable, Ignition, Millis};

#[test]
fn spending_stops_at_zero_and_never_overdraws() {
    let mut budget = Budget::new(4.0);
    assert!(budget.try_spend(2.0));
    assert!(budget.try_spend(2.0));
    assert!(!budget.try_spend(0.1));
    assert!(budget.available.abs() < f32::EPSILON);
}

#[test]
fn refill_is_proportional_to_elapsed_time_and_capped() {
    let mut budget = Budget::new(4.0);
    assert!(budget.try_spend(4.0));
    budget.refill(900_000, 4.0);
    assert!((budget.available - 1.0).abs() < f32::EPSILON);
    budget.refill(u64::MAX, 4.0);
    assert!((budget.available - budget.capacity).abs() < f32::EPSILON);
}

#[test]
fn invalid_costs_and_rates_do_not_create_energy() {
    let mut budget = Budget::new(4.0);
    assert!(budget.try_spend(2.0));
    let previous = budget;
    for invalid in [-1.0, f32::NAN, f32::INFINITY] {
        assert!(!budget.try_spend(invalid));
        budget.refill(3_600_000, invalid);
        assert_eq!(budget, previous);
    }
}

#[test]
fn drive_growth_and_satisfaction_are_bounded() {
    let mut drives = DriveTable::new(vec![Drive {
        name: "custom".to_owned(),
        level: 0.2,
        weight: 2.0,
        growth_per_min: 0.1,
    }]);
    drives.advance(60_000);
    assert!((drives.pressure() - 0.18).abs() < 0.000_001);
    drives.satisfy("unknown", 1.0);
    assert!((drives.pressure() - 0.18).abs() < 0.000_001);
    drives.advance(u64::MAX);
    assert!((drives.pressure() - 2.0).abs() < f32::EPSILON);
    drives.satisfy("custom", 3.0);
    assert!(drives.pressure().abs() < f32::EPSILON);
    drives.satisfy("custom", -1.0);
    assert!(drives.pressure().abs() < f32::EPSILON);
}

#[test]
fn strongest_uses_weighted_pressure_and_stable_tie_breaking() {
    let drives = DriveTable::new(vec![
        Drive {
            name: "high_level".to_owned(),
            level: 1.0,
            weight: 0.1,
            growth_per_min: 0.0,
        },
        Drive {
            name: "high_weight".to_owned(),
            level: 0.5,
            weight: 1.0,
            growth_per_min: 0.0,
        },
        Drive {
            name: "tie".to_owned(),
            level: 0.5,
            weight: 1.0,
            growth_per_min: 0.0,
        },
    ]);
    assert_eq!(
        drives.strongest().map(|drive| drive.name.as_str()),
        Some("high_weight")
    );
    let empty = DriveTable::new(Vec::new());
    assert!(empty.strongest().is_none());
    assert!(empty.pressure().abs() < f32::EPSILON);
}

#[test]
fn ignition_smooths_pressure_before_testing_the_threshold() {
    let mut ignition = Ignition::new(0.75, 0.25, 0, 0.5);
    ignition.advance(1.0);
    assert!(!ignition.drive_ready(Millis(0)));
    ignition.advance(1.0);
    assert!(ignition.drive_ready(Millis(0)));
}

#[test]
fn hysteresis_requires_a_strict_crossing_of_the_lower_threshold() {
    let mut ignition = Ignition::new(0.75, 0.25, 0, 1.0);
    ignition.advance(1.0);
    assert!(ignition.drive_ready(Millis(0)));
    ignition.fired(Millis(0));
    ignition.advance(0.5);
    ignition.advance(1.0);
    assert!(!ignition.drive_ready(Millis(10_000)));
    ignition.advance(0.49);
    ignition.advance(0.75);
    assert!(ignition.drive_ready(Millis(10_000)));
}

#[test]
fn cooldown_ends_at_the_exact_boundary_and_backward_time_does_not_bypass_it() {
    let mut ignition = Ignition::new(0.75, 0.25, 1_000, 1.0);
    assert!(!ignition.in_cooldown(Millis(0)));
    ignition.fired(Millis(100));
    assert!(ignition.in_cooldown(Millis(0)));
    assert!(ignition.in_cooldown(Millis(1_099)));
    assert!(!ignition.in_cooldown(Millis(1_100)));
    ignition.fired(Millis(50));
    assert!(ignition.in_cooldown(Millis(1_099)));
}
