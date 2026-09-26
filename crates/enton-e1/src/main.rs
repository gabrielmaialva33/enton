//! Bounded CLI for calibration and the freeze owner's explicit held-out evaluation.
// A binary owns its terminal output; only libraries must not print.
#![allow(clippy::print_stdout, clippy::print_stderr)]
use enton_core::Profile;
use enton_e1::offpolicy::{Exploration, evaluate};
use enton_e1::{BENCHMARK_VERSION, Error, Report, Sensors, Summary, e1a, e1b, run_tape_with};

#[derive(Debug, PartialEq)]
struct Cli {
    seeds: Vec<u64>,
    sensors: Sensors,
    json: bool,
    summary: bool,
    /// Exploration settings of the logging policy, when estimating candidates off policy.
    off_policy: Option<Exploration>,
    help: bool,
}
fn parse_seeds(value: &str) -> Result<Vec<u64>, Error> {
    let parse = |text: &str| {
        text.parse::<u64>()
            .map_err(|_| Error::Invalid(format!("invalid seed: {text}")))
    };
    if let Some((low, high)) = value.split_once("..") {
        let start = parse(low)?;
        let end = parse(high.trim_start_matches('='))?;
        if end < start || end - start >= 32 {
            return Err(Error::Invalid(
                "seed range must contain 1 to 32 entries (inclusive)".into(),
            ));
        }
        Ok((start..=end).collect())
    } else {
        let parts: Vec<_> = value.split(',').collect();
        if parts.len() > 32 {
            return Err(Error::Limit("CLI seeds"));
        }
        parts.into_iter().map(parse).collect()
    }
}
fn parse_fraction(value: Option<String>, what: &str) -> Result<f32, Error> {
    let value = value.ok_or_else(|| Error::Invalid(format!("missing {what} value")))?;
    value
        .parse::<f32>()
        .ok()
        .filter(|parsed| parsed.is_finite() && *parsed >= 0.0)
        .ok_or_else(|| Error::Invalid(format!("invalid {what}: {value}")))
}
fn parse_cli(args: impl IntoIterator<Item = String>) -> Result<Cli, Error> {
    let mut args = args.into_iter();
    let mut seeds = vec![42];
    let mut held_out = false;
    let mut sensors = Sensors::DEFAULT;
    let mut json = false;
    let mut summary = false;
    let mut off_policy = false;
    let mut exploration = Exploration::CALIBRATED;
    let mut help = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--off-policy" => off_policy = true,
            "--explore-probability" => {
                exploration.probability = parse_fraction(args.next(), "exploration probability")?;
            }
            "--explore-margin" => {
                exploration.margin_nats = parse_fraction(args.next(), "exploration margin")?;
            }
            "--seed" | "--seeds" => {
                let value = args
                    .next()
                    .ok_or_else(|| Error::Invalid("missing seed value".into()))?;
                seeds = parse_seeds(&value)?;
            }
            "--held-out" => held_out = true,
            "--with-directed" => sensors = Sensors::WITH_DIRECTED,
            "--json" => json = true,
            "--summary" => summary = true,
            "--help" | "-h" => help = true,
            _ => {
                if let Some(value) = arg
                    .strip_prefix("--seed=")
                    .or_else(|| arg.strip_prefix("--seeds="))
                {
                    seeds = parse_seeds(value)?;
                } else {
                    return Err(Error::Invalid(format!("unknown argument: {arg}")));
                }
            }
        }
    }
    if !held_out && seeds.iter().any(|seed| *seed >= 1000) {
        return Err(Error::Invalid("seeds >=1000 require --held-out and the freeze owner's completed manifest; not calibration".into()));
    }
    Ok(Cli {
        seeds,
        sensors,
        json,
        summary,
        off_policy: off_policy.then_some(exploration),
        help,
    })
}
fn execute() -> Result<(), Error> {
    let cli = parse_cli(std::env::args().skip(1))?;
    if cli.help {
        println!(
            "E1 benchmark {BENCHMARK_VERSION}\nUsage: e1-sim [--seed N | --seeds A..=B | --seeds A,B] [--with-directed] [--json] [--summary] [--held-out]\n       e1-sim --off-policy [--explore-probability P] [--explore-margin NATS] [--seeds ...] [--with-directed] [--json]\nRanges are inclusive, at most 32 seeds. Calibration seeds: 0 to 999. Held-out mode is reserved for the frozen-manifest owner.\nSensors: speaker verification, media tagger and end of turn; --with-directed adds the simulated device-directedness detector.\nOff policy: run t1-ref as an exploring logging policy (default probability {}, margin {} nats), estimate a family of threshold candidates from that log, and run each for real to measure the estimates.",
            Exploration::CALIBRATED.probability,
            Exploration::CALIBRATED.margin_nats,
        );
        return Ok(());
    }
    let profile = Profile::t1_ref();
    if let Some(exploration) = cli.off_policy {
        let report = evaluate(&cli.seeds, &profile, exploration, cli.sensors)?;
        if cli.json {
            println!("{}", serde_json::to_string(&report)?);
        } else {
            print!("{report}");
        }
        return Ok(());
    }
    let report = |seed| {
        Report::new(
            run_tape_with(&e1a(seed)?, &profile, cli.sensors)?,
            run_tape_with(&e1b(seed)?, &profile, cli.sensors)?,
        )
    };
    if cli.summary {
        let mut reports = Vec::with_capacity(cli.seeds.len());
        for &seed in &cli.seeds {
            reports.push(report(seed)?);
        }
        let summary = Summary::from_reports(&reports)?;
        if cli.json {
            println!("{}", serde_json::to_string(&summary)?);
        } else {
            println!("{summary}");
        }
    } else {
        for &seed in &cli.seeds {
            let report = report(seed)?;
            if cli.json {
                println!("{}", serde_json::to_string(&report)?);
            } else {
                println!("{report}");
            }
        }
    }
    Ok(())
}
fn main() -> std::process::ExitCode {
    match execute() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("E1 benchmark {BENCHMARK_VERSION}: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_held_out_seeds_without_generating_them() {
        assert!(parse_cli(["--seed".into(), "1000".into()]).is_err());
        assert!(parse_seeds("0..=999").is_err());
        assert_eq!(parse_seeds("0..=2").unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn parses_summary_flag() {
        let cli = parse_cli(["--seeds=0..=2".into(), "--summary".into()]).unwrap();
        assert!(cli.summary);
        assert!(!cli.json);
        assert_eq!(cli.seeds, vec![0, 1, 2]);

        let cli_default = parse_cli(["--seed=42".into()]).unwrap();
        assert!(!cli_default.summary);

        let cli_json_summary = parse_cli(["--summary".into(), "--json".into()]).unwrap();
        assert!(cli_json_summary.summary);
        assert!(cli_json_summary.json);
    }

    #[test]
    fn off_policy_runs_with_the_calibrated_exploration_unless_told_otherwise() {
        assert_eq!(parse_cli(["--seed=1".into()]).unwrap().off_policy, None);
        let cli = parse_cli(["--off-policy".into(), "--seeds=100..=103".into()]).unwrap();
        assert_eq!(cli.off_policy, Some(Exploration::CALIBRATED));
        let cli = parse_cli([
            "--off-policy".into(),
            "--explore-probability".into(),
            "0.2".into(),
            "--explore-margin".into(),
            "0.5".into(),
        ])
        .unwrap();
        assert_eq!(
            cli.off_policy,
            Some(Exploration {
                probability: 0.2,
                margin_nats: 0.5,
            })
        );
        for broken in ["-0.1", "nan", "inf", "x"] {
            assert!(parse_cli(["--explore-probability".into(), broken.into()]).is_err());
        }
        assert!(parse_cli(["--explore-margin".into()]).is_err());
    }

    #[test]
    fn the_directedness_detector_runs_only_when_asked() {
        assert_eq!(
            parse_cli(["--seed=1".into()]).unwrap().sensors,
            Sensors::DEFAULT
        );
        let with = parse_cli(["--with-directed".into(), "--seed=1".into()]).unwrap();
        assert_eq!(with.sensors, Sensors::WITH_DIRECTED);
        assert!(with.sensors.directed);
    }
}
