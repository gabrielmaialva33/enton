//! Command line flags.

use std::path::PathBuf;

use enton_core::Profile;

/// Printed on `--help` and after a flag error.
pub(crate) const USAGE: &str = "Usage: enton [--profile t1-ref|desktop] [--cortex-url <URL>] [--model <MODEL>] [--soul <PATH> | --no-soul] [--voice] [--speaker <ID>]
       enton why [--soul <PATH>] [--profile t1-ref|desktop] [--last <N>] [--since <DURATION>] [--json]";

/// How many speech cues `enton why` explains when `--last` is not given.
const DEFAULT_LAST: usize = 10;

/// What the binary was asked to do.
#[derive(Debug)]
pub(crate) enum Command {
    /// Live: perceive, think and speak.
    Run(CliConfig),
    /// Audit the soul: replay it and explain the latest speech decisions.
    Why(WhyConfig),
}

#[derive(Debug)]
pub(crate) struct CliConfig {
    pub(crate) profile: Profile,
    pub(crate) cortex_url: String,
    pub(crate) model: String,
    pub(crate) soul: Option<PathBuf>,
    #[cfg(feature = "voice")]
    pub(crate) voice: bool,
    #[cfg(feature = "voice")]
    pub(crate) speaker_id: Option<i32>,
}

/// Flags of `enton why`.
#[derive(Debug)]
pub(crate) struct WhyConfig {
    /// The profile whose reducer and calibration replay the log.
    pub(crate) profile: Profile,
    /// The log to audit; `None` when no default location could be derived.
    pub(crate) soul: Option<PathBuf>,
    /// How many of the most recent speech cues to explain, at least one.
    pub(crate) last: usize,
    /// Only cues this many milliseconds before the last recorded event, or later.
    pub(crate) since_ms: Option<u64>,
    /// Print machine-readable JSON instead of text.
    pub(crate) json: bool,
}

pub(crate) fn parse_cli_args() -> Result<Command, String> {
    parse_command(std::env::args().skip(1))
}

/// Parse a subcommand and its flags; without one, the flags of a live run.
pub(crate) fn parse_command(args: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let mut args = args.into_iter().peekable();
    if args.next_if(|arg| arg == "why").is_some() {
        return parse_why_args(args).map(Command::Why);
    }
    parse_args(args).map(Command::Run)
}

/// Split `--flag=value` into its flag and inline value; anything else is a bare flag.
fn split_flag(arg: &str) -> (String, Option<String>) {
    match arg.split_once('=') {
        Some((flag, value)) if flag.starts_with("--") => (flag.to_owned(), Some(value.to_owned())),
        _ => (arg.to_owned(), None),
    }
}

fn parse_profile(name: &str) -> Result<Profile, String> {
    match name {
        "t1-ref" => Ok(Profile::t1_ref()),
        "desktop" => Ok(Profile::desktop()),
        other => Err(format!("unknown profile: {other}")),
    }
}

/// Parse flags given either as `--flag value` or as `--flag=value`.
pub(crate) fn parse_args(args: impl IntoIterator<Item = String>) -> Result<CliConfig, String> {
    let mut args = args.into_iter();
    let mut profile = Profile::t1_ref();
    let mut cortex_url = "http://127.0.0.1:11434/v1".to_string();
    let mut model = "qwen3.8:27b-gato".to_string();
    let mut soul = None;
    let mut no_soul = false;
    #[cfg(feature = "voice")]
    let mut voice = false;
    #[cfg(feature = "voice")]
    let mut speaker_id = None;

    while let Some(arg) = args.next() {
        let (flag, inline) = split_flag(&arg);
        let mut value = || {
            inline
                .clone()
                .or_else(|| args.next())
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag.as_str() {
            "--profile" => profile = parse_profile(&value()?)?,
            "--cortex-url" => cortex_url = value()?,
            "--model" => model = value()?,
            "--soul" => soul = Some(PathBuf::from(value()?)),
            "--no-soul" => no_soul = true,
            #[cfg(feature = "voice")]
            "--voice" => voice = true,
            #[cfg(feature = "voice")]
            "--speaker" => {
                let id = value()?;
                speaker_id = Some(
                    id.parse::<i32>()
                        .map_err(|e| format!("invalid speaker id: {e}"))?,
                );
            }
            #[cfg(not(feature = "voice"))]
            "--voice" | "--speaker" => {
                if flag == "--speaker" {
                    value()?;
                }
                eprintln!("Warning: {flag} ignored (voice feature disabled)");
            }
            "--help" | "-h" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }

    if no_soul && soul.is_some() {
        return Err("--soul and --no-soul are mutually exclusive".to_string());
    }
    let soul = if no_soul {
        None
    } else {
        soul.or_else(|| default_soul_path(&profile.name))
    };

    Ok(CliConfig {
        profile,
        cortex_url,
        model,
        soul,
        #[cfg(feature = "voice")]
        voice,
        #[cfg(feature = "voice")]
        speaker_id,
    })
}

/// Parse the flags of `enton why`, in either `--flag value` or `--flag=value` form.
pub(crate) fn parse_why_args(args: impl IntoIterator<Item = String>) -> Result<WhyConfig, String> {
    let mut args = args.into_iter();
    let mut profile = Profile::t1_ref();
    let mut soul = None;
    let mut last = DEFAULT_LAST;
    let mut since_ms = None;
    let mut json = false;

    while let Some(arg) = args.next() {
        let (flag, inline) = split_flag(&arg);
        let mut value = || {
            inline
                .clone()
                .or_else(|| args.next())
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag.as_str() {
            "--profile" => profile = parse_profile(&value()?)?,
            "--soul" => soul = Some(PathBuf::from(value()?)),
            "--last" => {
                last = value()?
                    .parse::<usize>()
                    .ok()
                    .filter(|count| *count > 0)
                    .ok_or("--last takes a count of at least 1")?;
            }
            "--since" => since_ms = Some(parse_duration(&value()?)?),
            "--json" => json = true,
            "--help" | "-h" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument for why: {arg}")),
        }
    }

    Ok(WhyConfig {
        soul: soul.or_else(|| default_soul_path(&profile.name)),
        profile,
        last,
        since_ms,
        json,
    })
}

/// A duration such as `500ms`, `90s`, `5m` (or `5min`), `2h` or `1d`, in
/// milliseconds. A bare number is seconds.
pub(crate) fn parse_duration(text: &str) -> Result<u64, String> {
    let invalid = || format!("invalid duration: {text} (try 90s, 5m, 2h or 1d)");
    let text = text.trim();
    let digits = text
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(text.len());
    let (number, unit) = text.split_at_checked(digits).ok_or_else(invalid)?;
    let number: u64 = number.parse().map_err(|_| invalid())?;
    let scale: u64 = match unit.trim() {
        "ms" => 1,
        "" | "s" => 1_000,
        "m" | "min" => 60_000,
        "h" => 3_600_000,
        "d" => 86_400_000,
        _ => return Err(invalid()),
    };
    number.checked_mul(scale).ok_or_else(invalid)
}

/// `$XDG_DATA_HOME/enton/soul-<profile>.sqlite`, falling back to `~/.local/share`.
/// One log per profile: a snapshot only restores under the profile that wrote it.
pub(crate) fn default_soul_path(profile_name: &str) -> Option<PathBuf> {
    let data = std::env::var_os("XDG_DATA_HOME")
        .filter(|dir| !dir.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"))
        })?;
    Some(
        data.join("enton")
            .join(format!("soul-{profile_name}.sqlite")),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| (*arg).to_owned()).collect()
    }

    #[test]
    fn cli_accepts_both_flag_forms_and_rejects_conflicts() {
        let parse = |args: &[&str]| parse_args(strings(args));
        let cli = parse(&[
            "--profile=desktop",
            "--model",
            "m",
            "--soul",
            "/tmp/s.sqlite",
        ])
        .unwrap();
        assert_eq!(cli.profile.name, "desktop");
        assert_eq!(cli.model, "m");
        assert_eq!(cli.soul, Some(PathBuf::from("/tmp/s.sqlite")));
        assert!(parse(&["--no-soul"]).unwrap().soul.is_none());
        assert!(parse(&["--soul=/tmp/s.sqlite", "--no-soul"]).is_err());
        assert!(parse(&["--model"]).is_err());
        assert!(parse(&["--profile", "mars"]).is_err());
        assert!(parse(&["--bogus"]).is_err());
    }

    #[test]
    fn why_is_a_subcommand_with_its_own_flags() {
        let Command::Why(why) = parse_command(strings(&[
            "why",
            "--soul=/tmp/s.sqlite",
            "--profile",
            "desktop",
            "--last",
            "3",
            "--since=5m",
            "--json",
        ]))
        .unwrap() else {
            panic!("expected the why subcommand");
        };
        assert_eq!(why.profile.name, "desktop");
        assert_eq!(why.soul, Some(PathBuf::from("/tmp/s.sqlite")));
        assert_eq!(why.last, 3);
        assert_eq!(why.since_ms, Some(300_000));
        assert!(why.json);

        let Command::Why(defaults) = parse_command(strings(&["why"])).unwrap() else {
            panic!("expected the why subcommand");
        };
        assert_eq!(defaults.last, DEFAULT_LAST);
        assert!(defaults.since_ms.is_none());
        assert!(!defaults.json);

        assert!(matches!(
            parse_command(strings(&["--profile", "desktop"])).unwrap(),
            Command::Run(_)
        ));
        // Live-run flags are not why flags, and a count must be positive.
        assert!(parse_command(strings(&["why", "--no-soul"])).is_err());
        assert!(parse_command(strings(&["why", "--last", "0"])).is_err());
        assert!(parse_command(strings(&["why", "--last", "many"])).is_err());
        assert!(parse_command(strings(&["why", "--since"])).is_err());
    }

    #[test]
    fn durations_take_a_unit_or_default_to_seconds() {
        assert_eq!(parse_duration("500ms"), Ok(500));
        assert_eq!(parse_duration("90"), Ok(90_000));
        assert_eq!(parse_duration("90s"), Ok(90_000));
        assert_eq!(parse_duration("5m"), Ok(300_000));
        assert_eq!(parse_duration("5min"), Ok(300_000));
        assert_eq!(parse_duration(" 2h "), Ok(7_200_000));
        assert_eq!(parse_duration("1d"), Ok(86_400_000));
        for bad in ["", "m", "5 weeks", "-5m", "1.5h", "99999999999999999999d"] {
            assert!(parse_duration(bad).is_err(), "{bad}");
        }
    }
}
