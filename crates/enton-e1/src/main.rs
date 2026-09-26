//! Bounded CLI for calibration and the freeze owner's explicit held-out evaluation.
// A binary owns its terminal output; only libraries must not print.
#![allow(clippy::print_stdout, clippy::print_stderr)]
use enton_e1::{BENCHMARK_VERSION, Error, Report, e1a, e1b, run_tape};

#[derive(Debug)]
struct Cli {
    seeds: Vec<u64>,
    json: bool,
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
fn parse_cli(args: impl IntoIterator<Item = String>) -> Result<Cli, Error> {
    let mut args = args.into_iter();
    let mut seeds = vec![42];
    let mut held_out = false;
    let mut json = false;
    let mut help = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" | "--seeds" => {
                let value = args
                    .next()
                    .ok_or_else(|| Error::Invalid("missing seed value".into()))?;
                seeds = parse_seeds(&value)?;
            }
            "--held-out" => held_out = true,
            "--json" => json = true,
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
    Ok(Cli { seeds, json, help })
}
fn execute() -> Result<(), Error> {
    let cli = parse_cli(std::env::args().skip(1))?;
    if cli.help {
        println!(
            "E1 benchmark {BENCHMARK_VERSION}\nUsage: e1-sim [--seed N | --seeds A..=B | --seeds A,B] [--json] [--held-out]\nRanges are inclusive, at most 32 seeds. Calibration seeds: 0 to 999. Held-out mode is reserved for the frozen-manifest owner."
        );
        return Ok(());
    }
    for seed in cli.seeds {
        let report = Report::new(run_tape(&e1a(seed)?)?, run_tape(&e1b(seed)?)?)?;
        if cli.json {
            println!("{}", serde_json::to_string(&report)?);
        } else {
            println!("{report}");
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
}
