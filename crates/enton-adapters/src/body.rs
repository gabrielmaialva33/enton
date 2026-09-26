//! Hardware interoception adapter reading physical signals from sysfs and procfs.

use std::fs;
use std::path::Path;

use enton_core::BodySignals;

/// Reads physical body signals from the host system root (`/`).
#[must_use]
pub fn read_body_signals() -> BodySignals {
    read_body_signals_from(Path::new("/"))
}

/// Reads physical body signals using a specified filesystem root path.
///
/// Useful for unit testing with synthetic sysfs and procfs structures.
#[must_use]
pub fn read_body_signals_from(root: &Path) -> BodySignals {
    let thermal_dir = root.join("sys/class/thermal");
    let power_supply_dir = root.join("sys/class/power_supply");
    let loadavg_path = root.join("proc/loadavg");

    let temperature_c = read_max_temperature(&thermal_dir);
    let battery = read_battery_capacity(&power_supply_dir);
    let cpu_load = read_cpu_load(&loadavg_path);

    BodySignals {
        temperature_c,
        battery,
        cpu_load,
    }
}

fn read_max_temperature(thermal_dir: &Path) -> Option<f32> {
    let entries = fs::read_dir(thermal_dir).ok()?;
    let mut max_temp: Option<f32> = None;

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with("thermal_zone") {
            let temp_file = entry.path().join("temp");
            if let Ok(content) = fs::read_to_string(temp_file)
                && let Ok(milli_c) = content.trim().parse::<f32>()
            {
                let temp_c = milli_c / 1000.0;
                max_temp = Some(match max_temp {
                    Some(current) => current.max(temp_c),
                    None => temp_c,
                });
            }
        }
    }

    max_temp
}

fn read_battery_capacity(power_supply_dir: &Path) -> Option<f32> {
    let entries = fs::read_dir(power_supply_dir).ok()?;
    let mut capacities = Vec::new();

    for entry in entries.flatten() {
        let path = entry.path();
        let type_file = path.join("type");
        if let Ok(type_content) = fs::read_to_string(type_file)
            && type_content.trim().eq_ignore_ascii_case("battery")
        {
            let capacity_file = path.join("capacity");
            if let Ok(cap_content) = fs::read_to_string(capacity_file)
                && let Ok(val) = cap_content.trim().parse::<f32>()
                && val.is_finite()
                && (0.0..=100.0).contains(&val)
            {
                capacities.push(val / 100.0);
            }
        }
    }

    if capacities.is_empty() {
        None
    } else {
        let sum: f32 = capacities.iter().sum();
        let count = capacities.len() as f32;
        Some(sum / count)
    }
}

fn read_cpu_load(proc_loadavg_path: &Path) -> f32 {
    let Ok(content) = fs::read_to_string(proc_loadavg_path) else {
        return 0.0;
    };

    let Some(first_token) = content.split_whitespace().next() else {
        return 0.0;
    };

    let Ok(load_1min) = first_token.parse::<f32>() else {
        return 0.0;
    };

    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get) as f32;

    (load_1min / cores).max(0.0)
}

/// Errors surfaced by physical hardware interoception.
#[derive(Debug, thiserror::Error)]
pub enum BodyError {
    /// Reading a sysfs or procfs entry failed.
    #[error("failed to read sensor at '{path}': {source}")]
    Io {
        /// Sensor file path that failed to read.
        path: std::path::PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// Parsing a numeric sensor metric failed.
    #[error("failed to parse sensor metric at '{path}': {value}")]
    Parse {
        /// Sensor file path with invalid content.
        path: std::path::PathBuf,
        /// String value that failed to parse.
        value: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDir {
        path: std::path::PathBuf,
    }

    impl TempDir {
        fn new() -> Self {
            let id = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("enton_adapters_test_{id}_{}", std::process::id()));
            if let Err(_err) = fs::remove_dir_all(&path) {
                // Non-fatal if the directory did not previously exist.
            }
            fs::create_dir_all(&path).expect("failed to create temp dir");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            if let Err(_err) = fs::remove_dir_all(&self.path) {
                // Non-fatal if temp directory was already removed.
            }
        }
    }

    #[test]
    fn missing_sensors_return_defaults() {
        let temp = TempDir::new();
        let signals = read_body_signals_from(temp.path());
        assert_eq!(signals.temperature_c, None);
        assert_eq!(signals.battery, None);
        assert!(signals.cpu_load.abs() < f32::EPSILON);
    }

    #[test]
    fn thermal_zones_report_max_temperature() {
        let temp = TempDir::new();
        let tz0 = temp.path().join("sys/class/thermal/thermal_zone0");
        let tz1 = temp.path().join("sys/class/thermal/thermal_zone1");
        fs::create_dir_all(&tz0).unwrap();
        fs::create_dir_all(&tz1).unwrap();

        fs::write(tz0.join("temp"), "35000\n").unwrap();
        fs::write(tz1.join("temp"), "48500\n").unwrap();

        let signals = read_body_signals_from(temp.path());
        let temp_c = signals
            .temperature_c
            .expect("temperature should be present");
        assert!((temp_c - 48.5).abs() < 1e-4);
    }

    #[test]
    fn battery_reports_normalized_capacity() {
        let temp = TempDir::new();
        let bat0 = temp.path().join("sys/class/power_supply/BAT0");
        let ac = temp.path().join("sys/class/power_supply/AC");
        fs::create_dir_all(&bat0).unwrap();
        fs::create_dir_all(&ac).unwrap();

        fs::write(bat0.join("type"), "Battery\n").unwrap();
        fs::write(bat0.join("capacity"), "85\n").unwrap();

        fs::write(ac.join("type"), "Mains\n").unwrap();
        fs::write(ac.join("online"), "1\n").unwrap();

        let signals = read_body_signals_from(temp.path());
        let battery = signals.battery.expect("battery should be present");
        assert!((battery - 0.85).abs() < 1e-4);
    }

    #[test]
    fn battery_reports_accurate_boundaries_and_sanitizes_input() {
        let temp = TempDir::new();
        let bat = temp.path().join("sys/class/power_supply/BAT0");
        fs::create_dir_all(&bat).unwrap();
        fs::write(bat.join("type"), "Battery\n").unwrap();

        // 0% -> 0.0
        fs::write(bat.join("capacity"), "0\n").unwrap();
        let signals = read_body_signals_from(temp.path());
        assert_eq!(signals.battery, Some(0.0));

        // 1% -> 0.01 (regression for C6: previously returned 1.0)
        fs::write(bat.join("capacity"), "1\n").unwrap();
        let signals = read_body_signals_from(temp.path());
        let bat_val = signals.battery.expect("battery present");
        assert!((bat_val - 0.01).abs() < 1e-4);

        // 2% -> 0.02
        fs::write(bat.join("capacity"), "2\n").unwrap();
        let signals = read_body_signals_from(temp.path());
        let bat_val = signals.battery.expect("battery present");
        assert!((bat_val - 0.02).abs() < 1e-4);

        // 100% -> 1.0
        fs::write(bat.join("capacity"), "100\n").unwrap();
        let signals = read_body_signals_from(temp.path());
        assert_eq!(signals.battery, Some(1.0));

        // Malformed string -> None
        fs::write(bat.join("capacity"), "unknown\n").unwrap();
        let signals = read_body_signals_from(temp.path());
        assert_eq!(signals.battery, None);

        // Out of range negative -> None
        fs::write(bat.join("capacity"), "-5\n").unwrap();
        let signals = read_body_signals_from(temp.path());
        assert_eq!(signals.battery, None);

        // Out of range > 100 -> None
        fs::write(bat.join("capacity"), "101\n").unwrap();
        let signals = read_body_signals_from(temp.path());
        assert_eq!(signals.battery, None);

        // Non-finite (NaN / inf) -> None
        fs::write(bat.join("capacity"), "NaN\n").unwrap();
        let signals = read_body_signals_from(temp.path());
        assert_eq!(signals.battery, None);
    }

    #[test]
    fn cpu_load_computes_from_loadavg() {
        let temp = TempDir::new();
        let proc_dir = temp.path().join("proc");
        fs::create_dir_all(&proc_dir).unwrap();

        let cores =
            std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get) as f32;
        let expected_load = 0.5;
        let load1 = expected_load * cores;
        fs::write(
            proc_dir.join("loadavg"),
            format!("{load1:.2} 1.50 1.00 2/500 12345\n"),
        )
        .unwrap();

        let signals = read_body_signals_from(temp.path());
        assert!((signals.cpu_load - expected_load).abs() < 0.05);
    }

    #[test]
    fn live_read_does_not_panic() {
        let signals = read_body_signals();
        assert!(signals.cpu_load >= 0.0);
    }
}
