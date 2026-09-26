//! Terminal channel driver and async runtime for the Enton digital organism.

// A CLI binary talks to the terminal by design; only libraries must not print.
#![allow(clippy::print_stdout, clippy::print_stderr)]

mod cli;
mod journal;
mod runtime;
mod tasks;

use std::path::PathBuf;

use enton_adapters::MonotonicClock;
use enton_adapters::cortex::{CortexConfig, OpenAiCortex};
use enton_core::{Organism, Profile};
use tokio::sync::mpsc;

use cli::{USAGE, parse_cli_args};
use journal::Journal;
use runtime::{LoopMessage, RuntimeState, run_event_loop};
use tasks::{spawn_stdin_task, spawn_timer_task};
#[cfg(feature = "voice")]
use tasks::{spawn_voice_event_listener, try_init_voice};

#[tokio::main(flavor = "current_thread")]
async fn main() -> std::process::ExitCode {
    let cli = match parse_cli_args() {
        Ok(c) => c,
        Err(err) => {
            eprintln!("Error: {err}");
            eprintln!("{USAGE}");
            return std::process::ExitCode::FAILURE;
        }
    };

    let (organism, journal) = match restore(cli.profile, cli.soul).await {
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
        try_init_voice(cli.voice, cli.speaker_id, &voice_profile_name)
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
        ..CortexConfig::default()
    });
    let state = RuntimeState::new(
        organism,
        clock,
        cortex,
        #[cfg(feature = "voice")]
        voice_player,
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
async fn restore(
    profile: Profile,
    soul: Option<PathBuf>,
) -> Result<(Organism, Option<Journal>), String> {
    let Some(path) = soul else {
        println!("[enton] Soul disabled: nothing is recorded");
        return Organism::new(profile)
            .map(|organism| (organism, None))
            .map_err(|err| err.to_string());
    };
    let opened = tokio::task::spawn_blocking(move || {
        Journal::open(&path, &profile).map(|(journal, restored)| (journal, restored, path))
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
