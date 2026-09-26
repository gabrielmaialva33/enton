//! Terminal channel driver and async runtime for the Enton digital organism.

// A CLI binary talks to the terminal by design; only libraries must not print.
#![allow(clippy::print_stdout, clippy::print_stderr)]

mod cli;
mod conversation;
mod journal;
mod runtime;
mod tasks;
mod why;

use std::path::{Path, PathBuf};
use std::time::Duration;

/// How long the cortex warm-up may wait for a local server to load a large model.
const CORTEX_WARM_UP_TIMEOUT: Duration = Duration::from_secs(120);

use enton_adapters::MonotonicClock;
use enton_adapters::cortex::{CortexConfig, OpenAiCortex, Persona, PersonaError, PersonaOrigin};
use enton_adapters::soul::PersonaDigest;
use enton_core::{Organism, Profile};
use tokio::sync::mpsc;

use cli::{CliConfig, Command, USAGE, default_persona_path, parse_cli_args};
use journal::Journal;
#[cfg(feature = "voice")]
use runtime::Voice;
use runtime::{LoopMessage, RuntimeState, run_event_loop};
use tasks::{spawn_stdin_task, spawn_timer_task};
#[cfg(feature = "voice")]
use tasks::{spawn_voice_event_listener, try_init_voice};

fn main() -> std::process::ExitCode {
    match parse_cli_args() {
        Ok(Command::Run(cli)) => match load_persona(cli.persona.as_deref()) {
            Ok(persona) => run(cli, &persona),
            Err(err) => {
                eprintln!("Error: {err}");
                eprintln!(
                    "hint: without --persona, Enton reads $XDG_CONFIG_HOME/enton/PERSONA.md (or ~/.config/enton/PERSONA.md), or uses its built-in persona when that file does not exist"
                );
                std::process::ExitCode::FAILURE
            }
        },
        Ok(Command::Why(config)) => explain(&config),
        Err(err) => {
            eprintln!("Error: {err}");
            eprintln!("{USAGE}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Read the persona once, before anything runs: `--persona`, else `PERSONA.md`
/// in the config directory when it is there, else the built-in one. Nothing in
/// Enton ever writes it (see `enton_adapters::cortex::Persona`).
fn load_persona(explicit: Option<&Path>) -> Result<Persona, PersonaError> {
    let default = explicit.is_none().then(default_persona_path).flatten();
    let persona = match (explicit, &default) {
        (Some(path), _) => Persona::from_file(path)?,
        (None, Some(path)) => Persona::from_file_or_built_in(path)?,
        (None, None) => Persona::built_in(),
    };
    println!("{}", persona_line(&persona, default.as_deref()));
    Ok(persona)
}

/// The startup line naming the persona, its short hash (the one `enton why`
/// shows) and its length; `looked_at` is where a missing file was looked for.
fn persona_line(persona: &Persona, looked_at: Option<&Path>) -> String {
    let origin = match persona.origin() {
        PersonaOrigin::File(path) => path.display().to_string(),
        PersonaOrigin::BuiltIn => match looked_at {
            Some(path) => format!("built-in default (no {})", path.display()),
            None => "built-in default (no config directory: HOME and XDG_CONFIG_HOME are unset)"
                .to_owned(),
        },
    };
    format!(
        "[enton] Persona: {origin}, sha256 {}, {} bytes",
        PersonaDigest::from(persona).short_hex(),
        persona.bytes()
    )
}

/// `enton why`: a read-only audit, so it needs no async runtime.
fn explain(config: &cli::WhyConfig) -> std::process::ExitCode {
    match why::run(config) {
        Ok(output) => {
            print!("{output}");
            std::process::ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("Error: {err}");
            eprintln!("hint: pass --soul <PATH> (and the --profile that wrote it) for another log");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Live: perceive, think and speak until stdin closes or says `quit`.
#[tokio::main(flavor = "current_thread")]
async fn run(cli: CliConfig, persona: &Persona) -> std::process::ExitCode {
    let spoken = PersonaDigest::from(persona);
    let (organism, journal) = match restore(cli.profile, cli.soul, spoken).await {
        Ok(restored) => restored,
        Err(err) => {
            eprintln!("Error: {err}");
            eprintln!("hint: pass --soul <PATH> for another log, or --no-soul to run without one");
            return std::process::ExitCode::FAILURE;
        }
    };

    #[cfg(feature = "voice")]
    let voice_profile_name = organism.profile().name.clone();
    #[cfg(feature = "voice")]
    let voice_player = match tokio::task::spawn_blocking(move || {
        try_init_voice(
            cli.voice,
            cli.voice_model,
            cli.speaker_id,
            &voice_profile_name,
        )
    })
    .await
    {
        Ok(player) => player,
        Err(error) => {
            eprintln!("voice initialization worker failed: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    // A restored organism resumes where it stopped; a fresh one starts at zero.
    let clock = MonotonicClock::resuming_at(organism.last_seen());

    let (tx, rx) = mpsc::channel::<LoopMessage>(64);

    #[cfg(feature = "voice")]
    if let Some(ref player) = voice_player {
        spawn_voice_event_listener(player, tx.clone());
    }

    spawn_timer_task(tx.clone(), clock);
    spawn_stdin_task(tx.clone(), clock);

    let cortex = OpenAiCortex::new(CortexConfig {
        base_url: cli.cortex_url,
        model: cli.model,
        system_prompt: persona.text().to_owned(),
        ..CortexConfig::default()
    });
    // A local server unloads an idle model; load it now, so the first thought does
    // not time out waiting for it.
    let warming = cortex.clone();
    tokio::spawn(async move {
        match warming.warm_up(CORTEX_WARM_UP_TIMEOUT).await {
            Ok(took) => eprintln!("[enton] Cortex ready ({:.1} s)", took.as_secs_f32()),
            Err(err) => eprintln!("[enton] Cortex warm-up failed: {err}"),
        }
    });
    let state = RuntimeState::new(
        organism,
        clock,
        cortex,
        #[cfg(feature = "voice")]
        voice_player.map(|player| Voice {
            player,
            chime: cli.chime,
        }),
        journal,
    );

    match run_event_loop(state, rx, tx).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(err) => {
            // Acting on events the soul did not record would break replay; stop instead.
            eprintln!("[enton] stopping: {err}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Restore the organism from its soul, or start a fresh one without a log.
/// Thoughts recorded in the soul are linked to `persona`.
async fn restore(
    profile: Profile,
    soul: Option<PathBuf>,
    persona: PersonaDigest,
) -> Result<(Organism, Option<Journal>), String> {
    let Some(path) = soul else {
        println!("[enton] Soul disabled: nothing is recorded");
        return Organism::new(profile)
            .map(|organism| (organism, None))
            .map_err(|err| err.to_string());
    };
    let opened = tokio::task::spawn_blocking(move || {
        Journal::open(&path, &profile, persona).map(|(journal, restored)| (journal, restored, path))
    })
    .await
    .map_err(|err| format!("soul worker failed: {err}"))?;
    let (journal, restored, path) = opened.map_err(|err| err.to_string())?;
    println!(
        "[enton] Soul: {} (resuming at {} ms)",
        path.display(),
        restored.organism.last_seen().0
    );
    if !restored.abandoned.is_empty() {
        eprintln!(
            "[enton] {} thought(s) interrupted by the last shutdown were marked failed",
            restored.abandoned.len()
        );
    }
    Ok((restored.organism, Some(journal)))
}
