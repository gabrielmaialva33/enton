//! Command line flags.

use std::path::PathBuf;

use enton_core::Profile;

/// Printed on `--help` and after a flag error.
pub(crate) const USAGE: &str = "Usage: enton [--profile t1-ref|desktop] [--cortex-url <URL>] [--model <MODEL>] [--soul <PATH> | --no-soul] [--voice] [--speaker <ID>]";

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

pub(crate) fn parse_cli_args() -> Result<CliConfig, String> {
    parse_args(std::env::args().skip(1))
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
        let (flag, inline) = match arg.split_once('=') {
            Some((flag, value)) if flag.starts_with("--") => {
                (flag.to_owned(), Some(value.to_owned()))
            }
            _ => (arg.clone(), None),
        };
        let mut value = || {
            inline
                .clone()
                .or_else(|| args.next())
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag.as_str() {
            "--profile" => {
                profile = match value()?.as_str() {
                    "t1-ref" => Profile::t1_ref(),
                    "desktop" => Profile::desktop(),
                    other => return Err(format!("unknown profile: {other}")),
                };
            }
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

    #[test]
    fn cli_accepts_both_flag_forms_and_rejects_conflicts() {
        let parse = |args: &[&str]| parse_args(args.iter().map(|arg| (*arg).to_owned()));
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
}
