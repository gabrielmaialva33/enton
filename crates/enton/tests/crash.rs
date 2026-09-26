#![cfg(target_os = "linux")]
//! Crash safety of the soul, driven through the compiled `enton` binary.
//!
//! Enton records every event in its soul before it acts on it (write-ahead) and
//! records a thought as pending before it calls the cortex. These tests break the
//! process and its files the ways a machine does, read the soul back through its
//! public API and check that nothing recorded is lost, nothing is recorded twice,
//! nothing torn is replayed and nothing unrecorded is acted on:
//!
//! - SIGKILL at seeded points of a scripted conversation (including right after a
//!   thought starts and while the cortex streams its reply), then a restart that
//!   some seeds kill again while it recovers;
//! - a file size limit (`RLIMIT_FSIZE`, with `SIGXFSZ` ignored so writes fail with
//!   `EFBIG`) so the soul cannot grow, then a restart without it;
//! - the write-ahead log and the database file cut short at many offsets;
//! - one bit of a stored record flipped inside the database file, which SQLite
//!   cannot see: the soul's checksums must name the record.
//!
//! The cortex is a loopback OpenAI-compatible server on `std::net` that streams
//! each reply in two parts, so a kill can land between them.

use std::collections::HashSet;
use std::error::Error;
use std::fs::OpenOptions;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::ops::Range;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use enton_adapters::{SeqNo, Soul, SoulConfig, soul};
use enton_core::{Action, Event, Organism, Profile, ThoughtId};

type TestResult<T = ()> = Result<T, Box<dyn Error + Send + Sync>>;

/// Upper bound on anything the binary should do promptly.
const WAIT: Duration = Duration::from_secs(10);
/// Pacing bound while conversing: how long to wait for a decision or a reply.
const STEP: Duration = Duration::from_secs(3);
/// How long the fake cortex holds a reply open between its two parts.
const REPLY_GAP: Duration = Duration::from_millis(30);
/// The signal number of SIGKILL.
const SIGKILL: i32 = 9;

/// A scripted conversation: some lines address Enton by name, some do not. Every
/// line has a different length, so each recorded speech cue names its line.
const SCRIPT: [&str; 10] = [
    "bom dia pessoal",
    "enton, qual e a previsao do tempo hoje?",
    "acho que vai chover mais tarde",
    "enton",
    "liga a luz da sala por favor",
    "o jantar esta quase pronto",
    "enton, conta uma piada curta sobre gatos",
    "ninguem viu o controle da tv?",
    "enton, que horas sao agora?",
    "vou sair para comprar pao",
];

/// Said after every recovery, to show the restored organism still converses.
const FOLLOW_UP: &str = "enton, voce ainda esta ai comigo?";

/// The fake cortex's reply, streamed in two parts.
const REPLY_HEAD: &str = "Tudo certo por aqui.";
const REPLY_TAIL: &str = " Pode perguntar.";

/// Seeds for the kill points; each one picks where a conversation dies.
const KILL_SEEDS: [u64; 24] = [
    7, 19, 23, 42, 57, 64, 99, 101, 128, 256, 311, 404, 512, 777, 1_001, 1_024, 1_337, 2_024,
    2_048, 4_096, 8_191, 9_001, 31_337, 65_535,
];

/// The size of SQLite's WAL index (`-shm`), which it allocates on open.
const WAL_INDEX_BYTES: u64 = 32 * 1024;
/// Room for the write-ahead log past the WAL index under the disk-full limit.
const WAL_ROOM_BYTES: u64 = 40 * 1024;
/// One WAL frame at SQLite's default 4 KiB page: each step of the disk-full
/// limit moves the failing write to a later commit.
const FRAME_BYTES: u64 = 4096 + 24;
/// SQLite WAL layout: a 32-byte file header, then frames of a 24-byte header and a page.
const WAL_HEADER: usize = 32;
const FRAME_HEADER: usize = 24;

// ---------------------------------------------------------------------------
// Tests

#[test]
fn sigkill_at_seeded_points_0_to_5_loses_and_repeats_nothing() {
    kill_at_seeded_points(0..6).unwrap_or_else(|err| panic!("{err}"));
}

#[test]
fn sigkill_at_seeded_points_6_to_11_loses_and_repeats_nothing() {
    kill_at_seeded_points(6..12).unwrap_or_else(|err| panic!("{err}"));
}

#[test]
fn sigkill_at_seeded_points_12_to_17_loses_and_repeats_nothing() {
    kill_at_seeded_points(12..18).unwrap_or_else(|err| panic!("{err}"));
}

#[test]
fn sigkill_at_seeded_points_18_to_23_loses_and_repeats_nothing() {
    kill_at_seeded_points(18..24).unwrap_or_else(|err| panic!("{err}"));
}

// Twelve limits, one WAL frame apart, fail every kind of write in turn: the
// pending record of a thought, a speech append, the resolve after the cortex
// answered, and the append of its reply.

#[test]
fn a_full_disk_at_limit_steps_0_to_3_stops_enton_before_it_acts_on_unrecorded_writes() {
    disk_full_at_limits(0..4).unwrap_or_else(|err| panic!("{err}"));
}

#[test]
fn a_full_disk_at_limit_steps_4_to_7_stops_enton_before_it_acts_on_unrecorded_writes() {
    disk_full_at_limits(4..8).unwrap_or_else(|err| panic!("{err}"));
}

#[test]
fn a_full_disk_at_limit_steps_8_to_11_stops_enton_before_it_acts_on_unrecorded_writes() {
    disk_full_at_limits(8..12).unwrap_or_else(|err| panic!("{err}"));
}

#[test]
fn a_soul_without_room_for_its_wal_index_refuses_to_start() {
    no_room_at_startup().unwrap_or_else(|err| panic!("{err}"));
}

#[test]
fn a_torn_write_ahead_log_restores_its_last_committed_prefix() {
    torn_established_wal().unwrap_or_else(|err| panic!("{err}"));
}

#[test]
fn a_cut_first_write_ahead_log_restores_a_prefix_or_refuses_clearly() {
    cut_fresh_wal(Expect::RestoreOrRefuse).unwrap_or_else(|err| panic!("{err}"));
}

#[test]
fn a_crash_while_the_first_schema_is_written_leaves_a_soul_that_opens() {
    cut_fresh_wal(Expect::Restore).unwrap_or_else(|err| panic!("{err}"));
}

#[test]
fn a_truncated_database_restores_a_prefix_or_refuses_clearly() {
    truncated_database().unwrap_or_else(|err| panic!("{err}"));
}

#[test]
fn a_damaged_soul_is_never_silently_replaced_by_a_fresh_one() {
    tiny_database().unwrap_or_else(|err| panic!("{err}"));
}

#[test]
fn a_flipped_bit_in_a_stored_record_is_refused_naming_the_record() {
    flipped_bit().unwrap_or_else(|err| panic!("{err}"));
}

// ---------------------------------------------------------------------------
// Kill mid-flight

/// Where a conversation dies.
#[derive(Debug, Clone, Copy)]
enum KillAt {
    /// Once this many decisions were printed (zero: as soon as the banner shows).
    Decisions(usize),
    /// Right after the n-th thought starts (its `Think` is printed).
    Thought(usize),
    /// While the cortex streams its n-th reply, held open after the first part.
    Streaming(usize),
    /// Right after the n-th reply is spoken.
    Spoken(usize),
}

impl KillAt {
    fn reached(self, enton: &Enton) -> bool {
        match self {
            Self::Decisions(count) => enton.decisions().len() >= count,
            Self::Thought(count) => enton.count("Think {") >= count,
            Self::Streaming(count) => enton.streams >= count,
            Self::Spoken(count) => enton.count("Speak {") >= count,
        }
    }
}

/// One seeded crash: where the first life dies, and whether its first recovery
/// dies too (after the given delay) before the final one.
#[derive(Debug, Clone, Copy)]
struct KillPlan {
    at: KillAt,
    second_crash: Option<Duration>,
}

impl KillPlan {
    /// The kind of kill cycles with `index`, so every kind is covered; the seed picks
    /// the point and the second crash.
    fn new(index: usize, seed: u64) -> Self {
        let mut random = SplitMix64(seed);
        let at = match index % 4 {
            0 => KillAt::Decisions(random.below(19)),
            1 => KillAt::Thought(1 + random.below(8)),
            2 => KillAt::Streaming(1 + random.below(7)),
            _ => KillAt::Spoken(1 + random.below(7)),
        };
        // Startup takes a few milliseconds, so the second kill lands before, during
        // or after the restore that reconciles interrupted thoughts.
        let second_crash =
            (random.below(3) == 0).then(|| Duration::from_millis(random.next() % 12));
        Self { at, second_crash }
    }
}

/// `SplitMix64`: a tiny deterministic generator, enough to spread kill points.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: u64) -> usize {
        usize::try_from(self.next() % bound).unwrap_or(0)
    }
}

fn kill_at_seeded_points(indices: Range<usize>) -> TestResult {
    for index in indices {
        let seed = *KILL_SEEDS.get(index).ok_or("no such kill seed")?;
        let plan = KillPlan::new(index, seed);
        crash_and_recover(plan)
            .map_err(|err| format!("kill point {index} (seed {seed}, {plan:?}): {err}"))?;
    }
    Ok(())
}

fn crash_and_recover(plan: KillPlan) -> TestResult {
    let scratch = Scratch::new("kill")?;
    let cortex = FakeCortex::start()?;
    if let KillAt::Streaming(request) = plan.at {
        cortex.stall(request);
    }
    let script = script();
    let (mut recovered, heard) = first_life(&scratch, &cortex, &script, plan.at)?;
    if let Some(delay) = plan.second_crash {
        recovered = crash_while_recovering(&scratch, &cortex, &recovered, delay)?;
    }
    let heard = script.get(..heard).ok_or("heard more lines than said")?;
    recover_for_good(&scratch, &cortex, &recovered, heard)
}

/// A fresh soul, killed at `at`. Returns what the soul holds afterwards and how
/// many script lines it recorded.
fn first_life(
    scratch: &Scratch,
    cortex: &FakeCortex,
    script: &[String],
    at: KillAt,
) -> TestResult<(Inspection, usize)> {
    let mut enton = Enton::spawn(&scratch.soul(), cortex, None)?;
    let resumed = enton.banner()?;
    ensure(resumed == 0, || {
        format!("a fresh soul resumed at {resumed} ms")
    })?;
    let reached = converse(&mut enton, script, Some(at));
    let killed = enton.kill()?;
    ensure(killed.status.signal() == Some(SIGKILL), || {
        format!("enton stopped before the kill: {}", killed.describe())
    })?;

    let copy = copy_soul(&scratch.soul(), &scratch.dir("killed")?)?;
    let soul = inspect(&copy)?;
    let heard = check_log(&soul.events, script)?;
    check_decisions(&soul, 0, &killed.decisions(), false)?;
    check_pending(&soul)?;
    let requests = cortex.requests()?;
    check_effects_recorded(&copy, &soul, 0, script, &requests)?;
    if reached && matches!(at, KillAt::Streaming(_)) && requests.len() == killed.count("Think {") {
        // The stalled reply never finished, so the thought that asked for it was
        // recorded as pending before the call and must still be pending.
        let printed = killed.decisions().len();
        let last = soul
            .decisions
            .iter()
            .flatten()
            .take(printed)
            .filter_map(thought_of)
            .last();
        let pending: Vec<ThoughtId> = soul.pending.iter().map(|(thought, _)| *thought).collect();
        ensure(last.is_some() && pending == Vec::from_iter(last), || {
            format!("killed mid-reply to {last:?}, but pending thoughts are {pending:?}")
        })?;
    }
    Ok((soul, heard))
}

/// Restart on the crashed soul and kill that too, after `delay`, while it recovers.
fn crash_while_recovering(
    scratch: &Scratch,
    cortex: &FakeCortex,
    before: &Inspection,
    delay: Duration,
) -> TestResult<Inspection> {
    let mut enton = Enton::spawn(&scratch.soul(), cortex, None)?;
    thread::sleep(delay);
    let killed = enton.kill()?;
    ensure(killed.status.signal() == Some(SIGKILL), || {
        format!(
            "the recovering enton stopped before the kill: {}",
            killed.describe()
        )
    })?;
    let after = inspect(&copy_soul(&scratch.soul(), &scratch.dir("killed-again")?)?)?;
    ensure(
        after.events.get(..before.events.len()) == Some(before.events.as_slice()),
        || "a crash during recovery rewrote recorded history".to_owned(),
    )?;
    check_decisions(&after, before.events.len(), &killed.decisions(), false)?;
    check_pending(&after)?;
    if let Some(resumed) = resumed_at(&killed.stdout) {
        let expected = before.clean.last_seen().0;
        ensure(resumed == expected, || {
            format!("recovery resumed at {resumed} ms, the log ends at {expected} ms")
        })?;
    }
    if let Some(interrupted) = killed.interrupted()? {
        ensure(interrupted == before.pending.len(), || {
            format!(
                "{interrupted} thoughts reported interrupted, {:?} pending",
                before.pending
            )
        })?;
    }
    Ok(after)
}

/// Restart for good: the soul restores, reports the interrupted thoughts, keeps
/// talking and shuts down cleanly with its history intact.
fn recover_for_good(
    scratch: &Scratch,
    cortex: &FakeCortex,
    before: &Inspection,
    heard: &[String],
) -> TestResult {
    let mut enton = Enton::spawn(&scratch.soul(), cortex, None)?;
    let resumed = enton.banner()?;
    let expected = before.clean.last_seen().0;
    ensure(resumed == expected, || {
        format!("restart resumed at {resumed} ms, the log ends at {expected} ms")
    })?;
    converse(&mut enton, &[FOLLOW_UP.to_owned()], None);
    let finished = enton.quit()?;
    ensure(finished.status.success(), || {
        format!("the recovered enton failed: {}", finished.describe())
    })?;
    let interrupted = finished.interrupted()?.unwrap_or(0);
    ensure(interrupted == before.pending.len(), || {
        format!(
            "{interrupted} thoughts reported interrupted, {:?} were pending",
            before.pending
        )
    })?;

    let after = inspect(&scratch.soul())?;
    ensure(after.pending.is_empty(), || {
        format!("thoughts still pending after recovery: {:?}", after.pending)
    })?;
    ensure(
        after.events.get(..before.events.len()) == Some(before.events.as_slice()),
        || "recovery rewrote recorded history".to_owned(),
    )?;
    let mut lines = heard.to_vec();
    lines.push(FOLLOW_UP.to_owned());
    let recorded = check_log(&after.events, &lines)?;
    ensure(recorded == lines.len(), || {
        "the follow-up after recovery was not recorded".to_owned()
    })?;
    check_decisions(&after, before.events.len(), &finished.decisions(), true)?;
    check_clock(&after.events, before.events.len(), resumed)
}

// ---------------------------------------------------------------------------
// Disk full

fn disk_full_at_limits(steps: Range<u64>) -> TestResult {
    for step in steps {
        disk_full(step).map_err(|err| format!("limit step {step}: {err}"))?;
    }
    Ok(())
}

/// One soul through three lives: healthy, then under a file size limit (raised
/// by `step` WAL frames, so each step fails a later write) until a write fails,
/// then healthy again with its history intact.
fn disk_full(step: u64) -> TestResult {
    let scratch = Scratch::new("full")?;
    let soul = scratch.soul();
    let cortex = FakeCortex::start()?;
    let opening: Vec<String> = script().into_iter().take(2).collect();

    // A healthy first life: the soul exists and was shut down cleanly.
    let mut enton = Enton::spawn(&soul, &cortex, None)?;
    enton.banner()?;
    converse(&mut enton, &opening, None);
    let first = enton.quit()?;
    ensure(first.status.success(), || {
        format!("first life failed: {}", first.describe())
    })?;
    let before = inspect(&copy_soul(&soul, &scratch.dir("before")?)?)?;

    let room = WAL_ROOM_BYTES + step * FRAME_BYTES;
    let limit = (std::fs::metadata(&soul)?.len().max(WAL_INDEX_BYTES) + room).next_multiple_of(512);
    let full = life_without_room(&scratch, &cortex, &before, &opening, limit)?;

    // Room again: the soul is intact, replays and keeps talking.
    let mut enton = Enton::spawn(&soul, &cortex, None)?;
    let resumed = enton.banner()?;
    let expected = full.clean.last_seen().0;
    ensure(resumed == expected, || {
        format!("restart resumed at {resumed} ms, the log ends at {expected} ms")
    })?;
    converse(&mut enton, &[FOLLOW_UP.to_owned()], None);
    let healed = enton.quit()?;
    ensure(healed.status.success(), || {
        format!("restart failed: {}", healed.describe())
    })?;
    let interrupted = healed.interrupted()?.unwrap_or(0);
    ensure(interrupted == full.pending.len(), || {
        format!(
            "{interrupted} thoughts reported interrupted, {:?} pending",
            full.pending
        )
    })?;
    let after = inspect(&soul)?;
    ensure(after.pending.is_empty(), || {
        format!("still pending: {:?}", after.pending)
    })?;
    ensure(
        after.events.get(..full.events.len()) == Some(full.events.as_slice()),
        || "the restart rewrote recorded history".to_owned(),
    )?;
    check_decisions(&after, full.events.len(), &healed.decisions(), true)
}

/// Enton must stop on its own, report the soul failure, and not crash.
fn check_stopped_cleanly(stopped: &Finished) -> TestResult {
    ensure(stopped.status.code() == Some(1), || {
        format!("expected a clean failure exit: {}", stopped.describe())
    })?;
    ensure(
        stopped
            .stderr
            .iter()
            .any(|line| line.starts_with("[enton] stopping: soul: ")),
        || format!("no soul failure reported: {}", stopped.describe()),
    )?;
    ensure(!stopped.panicked(), || {
        format!("enton panicked: {}", stopped.describe())
    })
}

/// A life where no file may grow past `limit` bytes: Enton must stop by itself
/// at the first write that fails, report it, and never act on what it could not
/// record. Returns what the soul holds afterwards.
fn life_without_room(
    scratch: &Scratch,
    cortex: &FakeCortex,
    before: &Inspection,
    opening: &[String],
    limit: u64,
) -> TestResult<Inspection> {
    let soul = scratch.soul();
    let lines: Vec<String> = SCRIPT
        .iter()
        .cycle()
        .zip(0..60)
        .map(|(line, turn)| format!("{line} (fala {turn})"))
        .collect();
    let calls_before = cortex.requests()?.len();
    let mut enton = Enton::spawn(&soul, cortex, Some(limit))?;
    let resumed = enton.banner()?;
    ensure(resumed == before.clean.last_seen().0, || {
        format!("resumed at {resumed} ms under the limit")
    })?;
    converse(&mut enton, &lines, None);
    let stopped = enton.wait_exit()?;
    check_stopped_cleanly(&stopped)?;
    for file in soul_files(&soul) {
        if file.try_exists()? {
            let size = std::fs::metadata(&file)?.len();
            ensure(size <= limit, || {
                format!("{} grew to {size} bytes past {limit}", file.display())
            })?;
        }
    }

    let copy = copy_soul(&soul, &scratch.dir("full")?)?;
    let full = inspect(&copy)?;
    ensure(
        full.events.get(..before.events.len()) == Some(before.events.as_slice()),
        || "the full disk rewrote recorded history".to_owned(),
    )?;
    let said: Vec<String> = opening.iter().chain(&lines).cloned().collect();
    let consumed = check_log(&full.events, &said)?.saturating_sub(opening.len());
    ensure(consumed < lines.len(), || {
        "the file size limit never bit".to_owned()
    })?;
    check_decisions(&full, before.events.len(), &stopped.decisions(), false)?;
    check_pending(&full)?;
    let requests = cortex.requests()?;
    let calls = requests.get(calls_before..).unwrap_or_default();
    // A line the soul never recorded never reached the cortex, not even as history.
    for line in lines.iter().skip(consumed) {
        ensure(
            !calls.iter().any(|body| body.contains(line.as_str())),
            || format!("the cortex heard {line:?}, which the soul never recorded"),
        )?;
    }
    check_effects_recorded(&copy, &full, before.events.len(), &said, calls)?;
    Ok(full)
}

/// No effect without its record. Every cortex call of a run (`calls`, for the
/// events after the first `from`) is backed by a thought the run recorded, and a
/// line whose thought reached the cortex had that thought recorded first.
///
/// Whether a thought ever got a row (pending, done or failed) is probed on a
/// throwaway copy of the soul: resolving a thought that was never recorded fails
/// with `UnknownAction`.
fn check_effects_recorded(
    soul: &Path,
    inspection: &Inspection,
    from: usize,
    lines: &[String],
    calls: &[String],
) -> TestResult {
    let scratch = Scratch::new("probe")?;
    let log = Soul::open(
        copy_soul(soul, &scratch.dir("probe")?)?,
        SoulConfig::default(),
    )?;
    let mut spoken = 0;
    let mut recorded = 0;
    for (index, ((_, event), decided)) in inspection
        .events
        .iter()
        .zip(&inspection.decisions)
        .enumerate()
    {
        let line = if matches!(event, Event::Speech { .. }) {
            spoken += 1;
            lines.get(spoken - 1)
        } else {
            None
        };
        if index < from {
            continue;
        }
        for thought in decided.iter().filter_map(thought_of) {
            let on_record = match log.mark_failed(thought, "{}") {
                Ok(()) => true,
                Err(soul::Error::UnknownAction(_)) => false,
                Err(err) => return Err(err.into()),
            };
            recorded += usize::from(on_record);
            // The request carries the line as (the end of) a message's content.
            let heard = line.is_some_and(|line| {
                let content = format!("{line}\"");
                calls.iter().any(|body| body.contains(&content))
            });
            ensure(on_record || !heard, || {
                format!("the cortex heard {line:?} for {thought:?}, which was never recorded")
            })?;
        }
    }
    ensure(calls.len() <= recorded, || {
        format!(
            "{} cortex calls for {recorded} recorded thoughts",
            calls.len()
        )
    })
}

fn no_room_at_startup() -> TestResult {
    let scratch = Scratch::new("no-room")?;
    let soul = scratch.soul();
    let cortex = FakeCortex::start()?;
    let mut enton = Enton::spawn(&soul, &cortex, None)?;
    enton.banner()?;
    converse(
        &mut enton,
        &script().into_iter().take(2).collect::<Vec<_>>(),
        None,
    );
    ensure(enton.quit()?.status.success(), || {
        "first life failed".to_owned()
    })?;
    let before = inspect(&copy_soul(&soul, &scratch.dir("before")?)?)?;
    let requests = cortex.requests()?.len();

    // Half the WAL index: opening the soul already needs more room than that.
    let mut enton = Enton::spawn(&soul, &cortex, Some(WAL_INDEX_BYTES / 2))?;
    let refused = enton.wait_exit()?;
    ensure(refused_clearly(&refused), || {
        format!("expected a clear refusal: {}", refused.describe())
    })?;
    ensure(
        refused.decisions().is_empty() && cortex.requests()?.len() == requests,
        || "enton acted on a soul it could not open".to_owned(),
    )?;

    let after = inspect(&copy_soul(&soul, &scratch.dir("after")?)?)?;
    ensure(after.events == before.events, || {
        "the refused start changed the log".to_owned()
    })?;
    let finished = run_briefly(&soul, &cortex)?;
    ensure(
        finished.status.success()
            && resumed_at(&finished.stdout) == Some(before.clean.last_seen().0),
        || {
            format!(
                "the soul did not restore once there was room: {}",
                finished.describe()
            )
        },
    )
}

// ---------------------------------------------------------------------------
// Truncated files

/// How a damaged soul must come back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expect {
    /// Restore a consistent prefix of the log: the damage is a legitimate crash state.
    Restore,
    /// Restore a consistent prefix, or refuse with a clear error.
    RestoreOrRefuse,
}

/// Whether a cut lands on a commit boundary (a state some crash could leave) or tears a write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cut {
    Commit,
    Torn,
}

/// What a damaged soul restored through the soul API: how many events of the
/// original log it holds, or why it refused.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Outcome {
    Prefix(usize),
    Refused(String),
}

/// The undamaged log, and the organism after each of its prefixes.
struct Reference {
    events: Vec<(SeqNo, Event)>,
    states: Vec<Organism>,
}

impl Reference {
    fn of(soul: &Path) -> TestResult<Self> {
        let inspection = inspect(soul)?;
        let mut organism = Organism::new(Profile::t1_ref())?;
        let mut states = vec![organism.clone()];
        for (_, event) in &inspection.events {
            organism.step(event);
            states.push(organism.clone());
        }
        Ok(Self {
            events: inspection.events,
            states,
        })
    }
}

fn torn_established_wal() -> TestResult {
    let scratch = Scratch::new("torn-wal")?;
    let cortex = FakeCortex::start()?;
    let script = script();
    let soul = scratch.soul();

    // First life closes cleanly (its events end up in the database file); the
    // second is killed mid-reply, so its events and a pending thought live only in the WAL.
    let mut enton = Enton::spawn(&soul, &cortex, None)?;
    enton.banner()?;
    converse(&mut enton, script.get(..3).unwrap_or_default(), None);
    ensure(enton.quit()?.status.success(), || {
        "first life failed".to_owned()
    })?;
    cortex.stall(cortex.requests()?.len() + 3);
    let mut enton = Enton::spawn(&soul, &cortex, None)?;
    enton.banner()?;
    converse(
        &mut enton,
        script.get(3..).unwrap_or_default(),
        Some(KillAt::Streaming(3)),
    );
    ensure(enton.kill()?.status.signal() == Some(SIGKILL), || {
        "second life died early".to_owned()
    })?;

    let crashed = copy_soul(&soul, &scratch.dir("crashed")?)?;
    let reference = Reference::of(&copy_soul(&crashed, &scratch.dir("reference")?)?)?;
    let outcomes = cut_wal_everywhere(&crashed, &reference, &cortex, Expect::Restore)?;
    let last = outcomes.last().map(|(_, _, outcome)| outcome);
    ensure(
        last == Some(&Outcome::Prefix(reference.events.len())),
        || {
            format!(
                "the whole WAL restored {last:?} of {} events",
                reference.events.len()
            )
        },
    )
}

fn cut_fresh_wal(expect: Expect) -> TestResult {
    let scratch = Scratch::new("fresh-wal")?;
    let cortex = FakeCortex::start()?;
    let soul = scratch.soul();

    // Killed early in its first life: the WAL still holds every schema commit.
    let mut enton = Enton::spawn(&soul, &cortex, None)?;
    enton.banner()?;
    converse(&mut enton, script().get(..2).unwrap_or_default(), None);
    ensure(enton.kill()?.status.signal() == Some(SIGKILL), || {
        "first life died early".to_owned()
    })?;

    let crashed = copy_soul(&soul, &scratch.dir("crashed")?)?;
    let reference = Reference::of(&copy_soul(&crashed, &scratch.dir("reference")?)?)?;
    cut_wal_everywhere(&crashed, &reference, &cortex, expect).map(drop)
}

/// Cut the WAL of `source` at every commit boundary and inside every frame.
///
/// A cut on a commit boundary must restore no fewer events than the boundary
/// before it; a torn cut must restore exactly what the last commit before it did.
/// Every commit cut, and one cut in eight of the rest, also runs the binary (a
/// torn cut must restore what its commit cut did, which the binary checked).
fn cut_wal_everywhere(
    source: &Path,
    reference: &Reference,
    cortex: &FakeCortex,
    expect: Expect,
) -> TestResult<Vec<(usize, Cut, Outcome)>> {
    let wal = std::fs::read(sibling(source, "-wal"))?;
    let layout = wal_layout(&wal)?;
    let mut outcomes: Vec<(usize, Cut, Outcome)> = Vec::new();
    for (index, (offset, cut)) in wal_cuts(&layout).into_iter().enumerate() {
        let run_binary = cut == Cut::Commit || index % 8 == 0;
        let damage = |soul: &Path| truncate(&sibling(soul, "-wal"), offset);
        let outcome = restore_damaged(source, &damage, reference, cortex, expect, run_binary)
            .map_err(|err| format!("WAL cut at byte {offset} ({cut:?}): {err}"))?;
        // What the last commit at or before this offset restored (the database alone before any frame).
        let committed = outcomes
            .iter()
            .rev()
            .find(|(at, kind, _)| *kind == Cut::Commit && *at <= offset)
            .map(|(_, _, outcome)| outcome.clone());
        match (cut, committed) {
            (Cut::Torn, Some(committed)) => ensure(outcome == committed, || {
                format!(
                    "WAL torn at byte {offset} restored {outcome:?}, its last commit {committed:?}"
                )
            })?,
            (Cut::Commit, Some(Outcome::Prefix(earlier))) => {
                if let Outcome::Prefix(now) = outcome {
                    ensure(now >= earlier, || {
                        format!("WAL cut at byte {offset} lost events: {now} after {earlier}")
                    })?;
                }
            }
            _ => {}
        }
        outcomes.push((offset, cut, outcome));
    }
    // Cuts inside the WAL header come before the empty-WAL boundary; check them now.
    let empty = outcomes
        .iter()
        .find(|(at, _, _)| *at == WAL_HEADER)
        .map(|(_, _, outcome)| outcome.clone());
    for (offset, _, outcome) in outcomes.iter().filter(|(at, _, _)| *at < WAL_HEADER) {
        ensure(Some(outcome) == empty.as_ref(), || {
            format!("WAL header torn at byte {offset} restored {outcome:?}, an empty WAL {empty:?}")
        })?;
    }
    Ok(outcomes)
}

fn truncated_database() -> TestResult {
    let scratch = Scratch::new("truncated-db")?;
    let cortex = FakeCortex::start()?;
    let script = script();
    let soul = scratch.soul();

    // A cleanly closed soul: everything lives in the database file.
    let mut enton = Enton::spawn(&soul, &cortex, None)?;
    enton.banner()?;
    converse(&mut enton, script.get(..4).unwrap_or_default(), None);
    ensure(enton.quit()?.status.success(), || {
        "first life failed".to_owned()
    })?;
    let closed = copy_soul(&soul, &scratch.dir("closed")?)?;
    cut_database_everywhere(&closed, &cortex)?;

    // The same soul killed in a second life: the database file plus a WAL.
    let mut enton = Enton::spawn(&soul, &cortex, None)?;
    enton.banner()?;
    converse(&mut enton, script.get(4..).unwrap_or_default(), None);
    ensure(enton.kill()?.status.signal() == Some(SIGKILL), || {
        "second life died early".to_owned()
    })?;
    let crashed = copy_soul(&soul, &scratch.dir("crashed")?)?;
    cut_database_everywhere(&crashed, &cortex)
}

/// Cut the database file of `source` at the start, inside its header, and at and
/// inside every page; each cut must restore a prefix or refuse clearly.
fn cut_database_everywhere(source: &Path, cortex: &FakeCortex) -> TestResult {
    let dir = source.parent().ok_or("a soul outside any directory")?;
    let reference = Reference::of(&copy_soul(source, &dir.join("reference"))?)?;
    let length = usize::try_from(std::fs::metadata(source)?.len())?;
    let page = 4096;
    let mut offsets = vec![0, 1, 50, 99, 100, 1000, length.saturating_sub(1)];
    for start in (0..length).step_by(page) {
        offsets.extend([start, start + page / 2]);
    }
    offsets.retain(|offset| *offset < length);
    offsets.sort_unstable();
    offsets.dedup();
    for offset in offsets {
        let damage = |soul: &Path| truncate(soul, offset);
        restore_damaged(
            source,
            &damage,
            &reference,
            cortex,
            Expect::RestoreOrRefuse,
            true,
        )
        .map_err(|err| format!("database cut at byte {offset}: {err}"))?;
    }
    Ok(())
}

/// A soul with history, cut to a few bytes: no longer a database, and not the
/// empty file of a soul that was never written either. Enton must refuse it
/// rather than start a fresh organism over it.
fn tiny_database() -> TestResult {
    let scratch = Scratch::new("tiny-db")?;
    let cortex = FakeCortex::start()?;
    let soul = scratch.soul();
    let mut enton = Enton::spawn(&soul, &cortex, None)?;
    enton.banner()?;
    converse(&mut enton, script().get(..2).unwrap_or_default(), None);
    ensure(enton.quit()?.status.success(), || {
        "first life failed".to_owned()
    })?;
    for length in [1, 2, 16, 50, 99] {
        let cut = copy_soul(&soul, &scratch.dir(&format!("cut-{length}"))?)?;
        truncate(&cut, length)?;
        let finished = run_briefly(&cut, &cortex)?;
        ensure(refused_clearly(&finished), || {
            format!(
                "a soul cut to {length} bytes was not refused: {}",
                finished.describe()
            )
        })?;
    }
    Ok(())
}

/// Media damage SQLite cannot see: one bit of a stored record flips inside a
/// well-formed page of the database file, and the digit it hits becomes another
/// digit, so the record still parses. The binary must refuse the soul and name
/// the record: at startup for the snapshot it restores from, and in `enton why`
/// for an event of the audited log.
fn flipped_bit() -> TestResult {
    let scratch = Scratch::new("flipped-bit")?;
    let cortex = FakeCortex::start()?;
    let soul = scratch.soul();
    let mut enton = Enton::spawn(&soul, &cortex, None)?;
    enton.banner()?;
    converse(&mut enton, script().get(..3).unwrap_or_default(), None);
    ensure(enton.quit()?.status.success(), || {
        "first life failed".to_owned()
    })?;
    // A clean close leaves every record in the database file.
    let log = Soul::open_read_only(&soul, SoulConfig::default())?;
    let events = read_log(&log)?;
    let (snapshot_seq, blob) = log
        .latest_snapshot()?
        .ok_or("the close saved no snapshot")?;
    drop(log);

    let damaged = copy_soul(&soul, &scratch.dir("snapshot")?)?;
    flip_a_digit_of(&damaged, &blob)?;
    let finished = run_briefly(&damaged, &cortex)?;
    let named = format!("the snapshot at sequence number {snapshot_seq} fails its checksum");
    ensure(
        refused_clearly(&finished) && finished.stderr.iter().any(|line| line.contains(&named)),
        || {
            format!(
                "a damaged snapshot was not refused by name: {}",
                finished.describe()
            )
        },
    )?;

    // Restoring starts from the snapshot; only the audit replays older events.
    let (seq, speech) = events
        .iter()
        .find(|(_, event)| matches!(event, Event::Speech { .. }))
        .ok_or("no speech was recorded")?;
    let damaged = copy_soul(&soul, &scratch.dir("event")?)?;
    flip_a_digit_of(&damaged, &serde_json::to_vec(speech)?)?;
    let audit = Command::new(env!("CARGO_BIN_EXE_enton"))
        .arg("why")
        .arg("--soul")
        .arg(&damaged)
        .arg("--profile")
        .arg("t1-ref")
        .output()?;
    let stderr = String::from_utf8_lossy(&audit.stderr);
    let named = format!("the event at sequence number {seq} fails its checksum");
    ensure(
        audit.status.code() == Some(1) && stderr.contains(&named) && !stderr.contains("panicked"),
        || format!("enton why did not name the damaged event: {stderr}"),
    )
}

/// Flip the low bit of a digit in the one copy of `record` the database file
/// holds, found through a window of the record that occurs there exactly once.
/// The search runs from the end of the record, where the digits are numbers
/// (a record starts with its type tags, and a snapshot's names its format).
fn flip_a_digit_of(soul: &Path, record: &[u8]) -> TestResult {
    let mut file = std::fs::read(soul)?;
    let at = record
        .windows(24)
        .rev()
        .find_map(|window| {
            let digit = window.iter().rposition(u8::is_ascii_digit)?;
            let mut copies = file
                .windows(window.len())
                .enumerate()
                .filter(|(_, candidate)| *candidate == window);
            match (copies.next(), copies.next()) {
                (Some((at, _)), None) => Some(at + digit),
                _ => None,
            }
        })
        .ok_or("the record is not stored contiguously in the file")?;
    let byte = file
        .get_mut(at)
        .ok_or("the digit is past the end of the file")?;
    *byte ^= 0x01;
    std::fs::write(soul, file)?;
    Ok(())
}

/// Damage a copy of `source`, restore it through the soul API and (when
/// `run_binary`) through the binary. Neither may restore anything but a prefix
/// of the reference log, both must agree, and the binary must never panic.
fn restore_damaged(
    source: &Path,
    damage: &dyn Fn(&Path) -> io::Result<()>,
    reference: &Reference,
    cortex: &FakeCortex,
    expect: Expect,
    run_binary: bool,
) -> TestResult<Outcome> {
    let scratch = Scratch::new("cut")?;
    let api = copy_soul(source, &scratch.dir("api")?)?;
    damage(&api)?;
    let binary = copy_soul(&api, &scratch.dir("binary")?)?;

    let restored = restore_view(&api);
    let outcome = match &restored {
        Err(err) => Outcome::Refused(err.to_string()),
        Ok((organism, _)) => Outcome::Prefix(restored_prefix(&api, organism, reference)?),
    };
    if expect == Expect::Restore {
        ensure(matches!(outcome, Outcome::Prefix(_)), || {
            format!("a crash state must restore, but the soul refused: {outcome:?}")
        })?;
    }
    if !run_binary {
        return Ok(outcome);
    }

    let finished = run_briefly(&binary, cortex)?;
    ensure(
        !finished.panicked() && finished.status.signal().is_none(),
        || format!("enton crashed on a damaged soul: {}", finished.describe()),
    )?;
    match (&restored, resumed_at(&finished.stdout)) {
        (Ok((organism, pending)), Some(resumed)) => {
            let expected = organism.last_seen().0;
            ensure(resumed == expected, || {
                format!("enton resumed at {resumed} ms, the soul API at {expected} ms")
            })?;
            let interrupted = finished.interrupted()?.unwrap_or(0);
            ensure(interrupted == *pending, || {
                format!("{interrupted} thoughts reported interrupted, {pending} pending")
            })?;
            ensure(
                finished.status.success()
                    || finished
                        .stderr
                        .iter()
                        .any(|line| line.starts_with("[enton] stopping: soul: ")),
                || {
                    format!(
                        "enton restored but then failed unclearly: {}",
                        finished.describe()
                    )
                },
            )
        }
        (Ok(_), None) => ensure(
            expect == Expect::RestoreOrRefuse && refused_clearly(&finished),
            || {
                format!(
                    "the soul API restores but enton did not: {}",
                    finished.describe()
                )
            },
        ),
        (Err(err), Some(_)) => Err(format!("enton restored a soul the API refuses ({err})").into()),
        (Err(_), None) => ensure(refused_clearly(&finished), || {
            format!("expected a clear refusal: {}", finished.describe())
        }),
    }?;
    Ok(outcome)
}

/// Restore the way the binary does: open, list pending thoughts, replay from
/// the latest snapshot. Returns the organism and the number of pending thoughts.
fn restore_view(soul: &Path) -> Result<(Organism, usize), soul::Error> {
    let log = Soul::open(soul, SoulConfig::default())?;
    let pending = log.pending_actions()?.len();
    let (organism, _) = log.replay_organism(&Profile::t1_ref())?;
    Ok((organism, pending))
}

/// How many events of the reference a restored organism stands for. When the
/// whole log is still readable it must be exactly a prefix of the reference; if
/// only the snapshot and its tail are, the organism must still equal the state
/// after some prefix. Anything else replayed a torn event.
fn restored_prefix(soul: &Path, organism: &Organism, reference: &Reference) -> TestResult<usize> {
    let readable = Soul::open(soul, SoulConfig::default()).and_then(|log| read_log(&log));
    let Ok(events) = readable else {
        return reference
            .states
            .iter()
            .position(|state| state == organism)
            .ok_or_else(|| "restored a state no prefix of the log reduces to".into());
    };
    ensure(
        reference.events.get(..events.len()) == Some(events.as_slice()),
        || {
            format!(
                "the damaged log holds {} events that are not a prefix of the original",
                events.len()
            )
        },
    )?;
    ensure(reference.states.get(events.len()) == Some(organism), || {
        format!(
            "replay of {} events differs from a clean run of them",
            events.len()
        )
    })?;
    Ok(events.len())
}

/// Where each frame of a write-ahead log ends and whether it commits.
struct WalLayout {
    frame_len: usize,
    frames: Vec<(usize, bool)>,
}

fn wal_layout(wal: &[u8]) -> TestResult<WalLayout> {
    let word = |at: usize| -> TestResult<u32> {
        let bytes: [u8; 4] = wal
            .get(at..at + 4)
            .ok_or("WAL shorter than its header")?
            .try_into()?;
        Ok(u32::from_be_bytes(bytes))
    };
    let frame_len = FRAME_HEADER + usize::try_from(word(8)?)?;
    let salt = wal.get(16..24).ok_or("WAL shorter than its header")?;
    let mut frames = Vec::new();
    let mut end = WAL_HEADER + frame_len;
    while let Some(frame) = wal.get(end - frame_len..end) {
        // Frames left over from an earlier generation of the log carry an old salt.
        if frame.get(8..16) != Some(salt) {
            break;
        }
        frames.push((end, frame.get(4..8).is_some_and(|size| size != [0; 4])));
        end += frame_len;
    }
    ensure(frames.iter().any(|(_, commits)| *commits), || {
        "the WAL holds no commit".to_owned()
    })?;
    Ok(WalLayout { frame_len, frames })
}

/// Offsets to cut a WAL at: inside its header, at every frame boundary, and inside
/// every frame header and page. Sorted by offset.
fn wal_cuts(layout: &WalLayout) -> Vec<(usize, Cut)> {
    let mut cuts = vec![
        (0, Cut::Torn),
        (WAL_HEADER / 2, Cut::Torn),
        (WAL_HEADER, Cut::Commit),
    ];
    for &(end, commits) in &layout.frames {
        let start = end - layout.frame_len;
        cuts.push((start + FRAME_HEADER / 2, Cut::Torn));
        cuts.push((
            start + FRAME_HEADER + (layout.frame_len - FRAME_HEADER) / 2,
            Cut::Torn,
        ));
        cuts.push((end - 1, Cut::Torn));
        cuts.push((end, if commits { Cut::Commit } else { Cut::Torn }));
    }
    cuts
}

fn truncate(path: &Path, length: usize) -> io::Result<()> {
    let length = u64::try_from(length).map_err(io::Error::other)?;
    OpenOptions::new().write(true).open(path)?.set_len(length)
}

/// Start the binary on `soul` and, if it restores, stop it again at once.
fn run_briefly(soul: &Path, cortex: &FakeCortex) -> TestResult<Finished> {
    let mut enton = Enton::spawn(soul, cortex, None)?;
    if enton.pump_until(WAIT, |enton| resumed_at(&enton.stdout).is_some()) {
        enton.quit()
    } else {
        enton.wait_exit()
    }
}

/// Refused at startup: no banner, a failure exit, and the soul error on stderr.
fn refused_clearly(finished: &Finished) -> bool {
    resumed_at(&finished.stdout).is_none()
        && finished.status.code() == Some(1)
        && finished
            .stderr
            .iter()
            .any(|line| line.starts_with("Error: soul: "))
        && !finished.panicked()
}

// ---------------------------------------------------------------------------
// Reading the soul back

/// A soul read back through its public API, with a clean in-process run of its log.
struct Inspection {
    events: Vec<(SeqNo, Event)>,
    /// What a fresh organism decides on each event, in log order.
    decisions: Vec<Vec<Action>>,
    /// The organism after the whole log.
    clean: Organism,
    pending: Vec<(ThoughtId, SeqNo)>,
}

/// Read every event, reduce them from scratch, and check the soul's own replay
/// (latest snapshot plus tail) lands on the same organism with the same decisions.
fn inspect(soul: &Path) -> TestResult<Inspection> {
    let profile = Profile::t1_ref();
    let log = Soul::open(soul, SoulConfig::default())?;
    let events = read_log(&log)?;
    let mut clean = Organism::new(profile.clone())?;
    let decisions: Vec<Vec<Action>> = events.iter().map(|(_, event)| clean.step(event)).collect();
    let (replayed, tail) = log.replay_organism(&profile)?;
    ensure(replayed == clean, || {
        "replaying the soul differs from a clean run of its log".to_owned()
    })?;
    let snapshot = log.latest_snapshot()?.map_or(0, |(seq, _)| seq);
    let expected = events
        .iter()
        .zip(&decisions)
        .filter(|((seq, _), _)| *seq > snapshot)
        .flat_map(|(_, actions)| actions);
    ensure(tail.iter().eq(expected), || {
        format!("replay after snapshot {snapshot} decided {tail:?}, unlike a clean run")
    })?;
    let pending = log.pending_actions()?;
    Ok(Inspection {
        events,
        decisions,
        clean,
        pending,
    })
}

/// Every event, page by page: `read_after` rejects gaps and unreadable payloads.
fn read_log(log: &Soul) -> Result<Vec<(SeqNo, Event)>, soul::Error> {
    let mut events: Vec<(SeqNo, Event)> = Vec::new();
    loop {
        let cursor = events.last().map_or(0, |(seq, _)| *seq);
        let page = log.read_after(cursor, 256)?;
        if page.is_empty() {
            return Ok(events);
        }
        events.extend(page);
    }
}

/// The log starts at one with no gap, holds the said `lines` exactly once and in
/// order (a prefix of them), and each cortex reply once. Returns how many lines it holds.
fn check_log(events: &[(SeqNo, Event)], lines: &[String]) -> TestResult<usize> {
    for (position, (seq, _)) in (1..).zip(events) {
        ensure(*seq == position, || {
            format!("event {position} is recorded as seq {seq}")
        })?;
    }
    let heard: Vec<(u32, bool)> = events
        .iter()
        .filter_map(|(_, event)| match event {
            Event::Speech { cue, .. } => Some((cue.duration_ms, cue.keyword)),
            _ => None,
        })
        .collect();
    ensure(heard.len() <= lines.len(), || {
        format!(
            "{} speech events for {} lines said: some recorded twice",
            heard.len(),
            lines.len()
        )
    })?;
    for (index, (cue, line)) in heard.iter().zip(lines).enumerate() {
        ensure(*cue == signature(line), || {
            format!("speech event {index} is {cue:?}, but line {index} was {line:?}")
        })?;
    }
    let mut replies = HashSet::new();
    for (seq, event) in events {
        if let Event::CortexReply { thought, .. } = event {
            ensure(replies.insert(*thought), || {
                format!("reply to {thought:?} recorded twice (seq {seq})")
            })?;
        }
    }
    Ok(heard.len())
}

/// A run's printed decisions against a clean run of the events it recorded (the
/// log after its first `from` events): every printed decision was recorded first,
/// and a crash can only have cut the decisions of the last recorded event. A
/// `complete` run printed them all.
fn check_decisions(
    inspection: &Inspection,
    from: usize,
    printed: &[&str],
    complete: bool,
) -> TestResult {
    let per_event = inspection
        .decisions
        .get(from..)
        .ok_or("the log lost events")?;
    let clean: Vec<String> = per_event
        .iter()
        .flatten()
        .map(|action| format!("{action:?}"))
        .collect();
    if let Some((index, (clean, printed))) = clean
        .iter()
        .zip(printed)
        .enumerate()
        .find(|(_, (clean, printed))| clean != printed)
    {
        return Err(
            format!("decision {index} was printed as {printed} but replays as {clean}").into(),
        );
    }
    ensure(printed.len() <= clean.len(), || {
        format!(
            "{} decisions printed, only {} recorded: acted on an unrecorded event ({:?})",
            printed.len(),
            clean.len(),
            printed.get(clean.len()..),
        )
    })?;
    let settled: usize = per_event
        .split_last()
        .map_or(0, |(_, earlier)| earlier.iter().map(Vec::len).sum());
    let floor = if complete { clean.len() } else { settled };
    ensure(printed.len() >= floor, || {
        format!(
            "{} decisions printed, but {floor} recorded ones were settled",
            printed.len()
        )
    })
}

/// At most one thought is in flight. A pending one was decided by the event its
/// row points at, never got a reply, and no thought was decided after it except
/// by the last recorded event (whose dispatch the crash may have cut).
fn check_pending(inspection: &Inspection) -> TestResult {
    ensure(inspection.pending.len() <= 1, || {
        format!("several thoughts in flight: {:?}", inspection.pending)
    })?;
    let last = inspection.decisions.len().saturating_sub(1);
    for &(thought, created) in &inspection.pending {
        let index = usize::try_from(created)?
            .checked_sub(1)
            .ok_or("pending at seq 0")?;
        let decided = inspection
            .decisions
            .get(index)
            .ok_or("pending row points past the log")?;
        ensure(
            decided
                .iter()
                .any(|action| thought_of(action) == Some(thought)),
            || format!("{thought:?} is pending at seq {created}, which did not decide it"),
        )?;
        let answered = inspection.events.iter().any(
            |(_, event)| matches!(event, Event::CortexReply { thought: replied, .. } if *replied == thought),
        );
        ensure(!answered, || {
            format!("{thought:?} is pending but its reply is recorded")
        })?;
        let later: Vec<usize> = inspection
            .decisions
            .iter()
            .enumerate()
            .flat_map(|(event, actions)| {
                actions
                    .iter()
                    .filter_map(thought_of)
                    .map(move |id| (event, id))
            })
            .skip_while(|(_, id)| *id != thought)
            .skip(1)
            .map(|(event, _)| event)
            .collect();
        ensure(later.iter().all(|event| *event == last), || {
            format!("{thought:?} is still pending although later thoughts superseded it")
        })?;
    }
    Ok(())
}

/// Time never runs backward across a restart: events recorded after the first
/// `from` are no earlier than where the organism resumed.
fn check_clock(events: &[(SeqNo, Event)], from: usize, resumed: u64) -> TestResult {
    for (seq, event) in events.iter().skip(from) {
        let at = event.now().0;
        ensure(at >= resumed, || {
            format!("seq {seq} at {at} ms is before the resume point {resumed} ms")
        })?;
    }
    Ok(())
}

fn thought_of(action: &Action) -> Option<ThoughtId> {
    match action {
        Action::Think { thought, .. } => Some(*thought),
        _ => None,
    }
}

fn script() -> Vec<String> {
    SCRIPT.iter().map(|line| (*line).to_owned()).collect()
}

/// What a typed line leaves in its recorded speech cue: its length (60 ms per
/// character, capped at 3 s) and whether it names Enton. Mirrors the binary's
/// stdin adapter; the script's lines all differ in length, so each cue names its line.
fn signature(line: &str) -> (u32, bool) {
    let chars = line.trim().chars().count();
    let duration = u32::try_from(chars.saturating_mul(60))
        .unwrap_or(u32::MAX)
        .min(3_000);
    (duration, enton_core::contains_keyword_word(line, "enton"))
}

/// Fail with `message` unless `condition` holds.
fn ensure(condition: bool, message: impl FnOnce() -> String) -> TestResult {
    if condition {
        Ok(())
    } else {
        Err(message().into())
    }
}

// ---------------------------------------------------------------------------
// Files

static NEXT_SCRATCH: AtomicUsize = AtomicUsize::new(0);

/// A temporary directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> TestResult<Self> {
        let id = NEXT_SCRATCH.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("enton-crash-{}-{tag}-{id}", std::process::id()));
        std::fs::create_dir_all(&path)?;
        Ok(Self(path))
    }

    fn soul(&self) -> PathBuf {
        self.0.join("soul.sqlite")
    }

    fn dir(&self, name: &str) -> TestResult<PathBuf> {
        let path = self.0.join(name);
        std::fs::create_dir_all(&path)?;
        Ok(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // Best effort: a leftover temporary directory must not fail a test.
        drop(std::fs::remove_dir_all(&self.0));
    }
}

/// The database file and the siblings SQLite keeps next to it in WAL mode.
fn soul_files(soul: &Path) -> [PathBuf; 3] {
    [
        soul.to_path_buf(),
        sibling(soul, "-wal"),
        sibling(soul, "-shm"),
    ]
}

fn sibling(soul: &Path, suffix: &str) -> PathBuf {
    let mut name = soul.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// Copy a stopped soul (database, WAL and WAL index, as present) into `dir`.
fn copy_soul(soul: &Path, dir: &Path) -> TestResult<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let copy = dir.join("soul.sqlite");
    for (from, to) in soul_files(soul).iter().zip(soul_files(&copy)) {
        if from.try_exists()? {
            std::fs::copy(from, to)?;
        }
    }
    Ok(copy)
}

// ---------------------------------------------------------------------------
// The binary

/// What the driver hears, in arrival order.
enum Signal {
    /// A line the binary printed on stdout.
    Stdout(String),
    /// The binary's stdout reached end of file.
    StdoutClosed,
    /// The cortex sent the first part of a reply and holds the rest back.
    Streaming,
}

/// The `enton` binary on a soul, fed through stdin and watched through stdout.
struct Enton {
    child: Child,
    stdin: Option<ChildStdin>,
    signals: Receiver<Signal>,
    stdout: Vec<String>,
    stdout_open: bool,
    /// Replies the cortex started streaming during this run.
    streams: usize,
    stderr: Arc<Mutex<Vec<String>>>,
    readers: Vec<JoinHandle<()>>,
}

/// How a run of the binary ended.
struct Finished {
    status: ExitStatus,
    stdout: Vec<String>,
    stderr: Vec<String>,
}

impl Enton {
    /// Start the binary on `soul`. With `file_limit`, no file it writes may grow
    /// past that many bytes, and a write past it fails instead of killing it.
    fn spawn(soul: &Path, cortex: &FakeCortex, file_limit: Option<u64>) -> TestResult<Self> {
        let binary = env!("CARGO_BIN_EXE_enton");
        let mut command = if let Some(bytes) = file_limit {
            // The script ignores SIGXFSZ and sets RLIMIT_FSIZE (`ulimit -f` counts
            // 512-byte blocks), and `exec` keeps both: writes past the limit fail
            // with EFBIG. Integration tests may not use `unsafe` (`pre_exec`).
            let mut shell = Command::new("sh");
            shell
                .arg("-c")
                .arg(r#"trap '' XFSZ && ulimit -f "$1" && shift && exec "$@""#)
                .arg("enton-file-limit")
                .arg((bytes / 512).to_string())
                .arg(binary);
            shell
        } else {
            Command::new(binary)
        };
        // Never the owner's own PERSONA.md or CHECKLIST.md: a directory that does not
        // exist beside the soul gives the built-in persona and nothing to check.
        let config = soul.with_extension("config");
        command
            .arg("--soul")
            .arg(soul)
            .arg("--cortex-url")
            .arg(cortex.url())
            .env("XDG_CONFIG_HOME", config)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let (sender, signals) = mpsc::channel();
        let mut child = command.spawn()?;
        let stdout = child.stdout.take();
        let stderr_pipe = child.stderr.take();
        let mut enton = Self {
            stdin: child.stdin.take(),
            child,
            signals,
            stdout: Vec::new(),
            stdout_open: true,
            streams: 0,
            stderr: Arc::new(Mutex::new(Vec::new())),
            readers: Vec::new(),
        };
        let (Some(stdout), Some(stderr_pipe)) = (stdout, stderr_pipe) else {
            return Err("the binary's pipes are missing".into());
        };
        cortex.observe(sender.clone())?;
        enton.readers.push(thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if sender.send(Signal::Stdout(line)).is_err() {
                    return;
                }
            }
            drop(sender.send(Signal::StdoutClosed));
        }));
        let lines = Arc::clone(&enton.stderr);
        enton.readers.push(thread::spawn(move || {
            for line in BufReader::new(stderr_pipe).lines().map_while(Result::ok) {
                if let Ok(mut lines) = lines.lock() {
                    lines.push(line);
                }
            }
        }));
        Ok(enton)
    }

    fn send(&mut self, line: &str) -> io::Result<()> {
        let stdin = self.stdin.as_mut().ok_or(io::ErrorKind::BrokenPipe)?;
        writeln!(stdin, "{line}")?;
        stdin.flush()
    }

    fn absorb(&mut self, signal: Signal) {
        match signal {
            Signal::Stdout(line) => self.stdout.push(line),
            Signal::StdoutClosed => self.stdout_open = false,
            Signal::Streaming => self.streams += 1,
        }
    }

    /// Absorb signals until `done` holds (true), or `timeout` passes or stdout
    /// closes first (false).
    fn pump_until(&mut self, timeout: Duration, done: impl Fn(&Self) -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if done(self) {
                return true;
            }
            if !self.stdout_open {
                return false;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            match self.signals.recv_timeout(left) {
                Ok(signal) => self.absorb(signal),
                Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => {
                    return done(self);
                }
            }
        }
    }

    fn decisions(&self) -> Vec<&str> {
        decisions(&self.stdout)
    }

    fn count(&self, kind: &str) -> usize {
        self.decisions()
            .iter()
            .filter(|decision| decision.starts_with(kind))
            .count()
    }

    /// Whether the last thought printed has no reply printed after it yet.
    fn awaiting_reply(&self) -> bool {
        let decisions = self.decisions();
        let last = |kind: &str| {
            decisions
                .iter()
                .rposition(|decision| decision.starts_with(kind))
        };
        last("Think {") > last("Speak {")
    }

    /// Wait for the startup banner; returns where the organism resumed, in ms.
    fn banner(&mut self) -> TestResult<u64> {
        self.pump_until(WAIT, |enton| resumed_at(&enton.stdout).is_some());
        resumed_at(&self.stdout)
            .ok_or_else(|| format!("no startup banner; stderr: {:?}", self.stderr_lines()).into())
    }

    fn stderr_lines(&self) -> Vec<String> {
        self.stderr
            .lock()
            .map(|lines| lines.clone())
            .unwrap_or_default()
    }

    fn kill(&mut self) -> TestResult<Finished> {
        self.child.kill()?;
        let status = self.child.wait()?;
        self.finish(status)
    }

    fn quit(&mut self) -> TestResult<Finished> {
        // A binary that already stopped reading reports why through its exit status.
        drop(self.send("quit"));
        self.stdin = None;
        self.wait_exit()
    }

    /// Wait for the binary to exit on its own.
    fn wait_exit(&mut self) -> TestResult<Finished> {
        let deadline = Instant::now() + WAIT;
        loop {
            if let Some(status) = self.child.try_wait()? {
                return self.finish(status);
            }
            if Instant::now() >= deadline {
                return Err(
                    format!("enton did not exit; stderr: {:?}", self.stderr_lines()).into(),
                );
            }
            if !self.pump_until(Duration::from_millis(10), |_| false) && !self.stdout_open {
                thread::sleep(Duration::from_millis(5));
            }
        }
    }

    /// Collect everything the exited binary printed.
    fn finish(&mut self, status: ExitStatus) -> TestResult<Finished> {
        let deadline = Instant::now() + WAIT;
        while self.stdout_open {
            let signal = self
                .signals
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .map_err(|_| "stdout stayed open after the binary exited")?;
            self.absorb(signal);
        }
        for reader in self.readers.drain(..) {
            reader.join().map_err(|_| "a pipe reader panicked")?;
        }
        self.stdin = None;
        Ok(Finished {
            status,
            stdout: self.stdout.clone(),
            stderr: self.stderr_lines(),
        })
    }
}

impl Drop for Enton {
    fn drop(&mut self) {
        // Never leak a binary from a failed test; it may already be gone.
        if matches!(self.child.try_wait(), Ok(None)) {
            drop(self.child.kill());
        }
        drop(self.child.wait());
    }
}

impl Finished {
    fn decisions(&self) -> Vec<&str> {
        decisions(&self.stdout)
    }

    fn count(&self, kind: &str) -> usize {
        self.decisions()
            .iter()
            .filter(|decision| decision.starts_with(kind))
            .count()
    }

    /// How many thoughts the startup reported interrupted and marked failed, if it said.
    fn interrupted(&self) -> TestResult<Option<usize>> {
        let Some(line) = self
            .stderr
            .iter()
            .find(|line| line.contains("thought(s) interrupted"))
        else {
            return Ok(None);
        };
        let count = line
            .trim_start_matches("[enton] ")
            .split(' ')
            .next()
            .unwrap_or_default();
        Ok(Some(count.parse()?))
    }

    fn panicked(&self) -> bool {
        self.stderr.iter().any(|line| line.contains("panicked"))
    }

    fn describe(&self) -> String {
        format!("{}; stderr: {:?}", self.status, self.stderr)
    }
}

/// Decisions are the stdout lines that are not `[enton]` notices.
fn decisions(stdout: &[String]) -> Vec<&str> {
    stdout
        .iter()
        .map(String::as_str)
        .filter(|line| !line.starts_with('['))
        .collect()
}

/// The time the startup banner says the organism resumed at, in ms.
fn resumed_at(stdout: &[String]) -> Option<u64> {
    stdout.iter().find_map(|line| {
        let (_, rest) = line.split_once("(resuming at ")?;
        rest.strip_suffix(" ms)")?.parse().ok()
    })
}

/// Say each line in turn, waiting for its decision and, for a thought, its spoken
/// reply; stop early once `kill_at` is reached. Returns whether it was reached.
fn converse(enton: &mut Enton, lines: &[String], kill_at: Option<KillAt>) -> bool {
    let reached = |enton: &Enton| kill_at.is_some_and(|at| at.reached(enton));
    for line in lines {
        if reached(enton) {
            return true;
        }
        if enton.send(line).is_err() {
            // The binary stopped reading: it is exiting.
            return false;
        }
        let before = enton.decisions().len();
        enton.pump_until(STEP, |enton| {
            reached(enton) || enton.decisions().len() > before
        });
        enton.pump_until(STEP, |enton| reached(enton) || !enton.awaiting_reply());
    }
    reached(enton)
}

// ---------------------------------------------------------------------------
// The cortex

struct CortexState {
    stop: AtomicBool,
    /// Bodies of the chat completion requests, in arrival order.
    requests: Mutex<Vec<String>>,
    /// Where to announce a reply that started streaming.
    observer: Mutex<Option<Sender<Signal>>>,
    /// The 1-based request whose reply stalls after its first part (0: none).
    stall: AtomicUsize,
}

/// A loopback OpenAI-compatible cortex that streams every reply in two parts.
struct FakeCortex {
    addr: SocketAddr,
    state: Arc<CortexState>,
    server: Option<JoinHandle<()>>,
}

impl FakeCortex {
    fn start() -> TestResult<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?;
        let state = Arc::new(CortexState {
            stop: AtomicBool::new(false),
            requests: Mutex::new(Vec::new()),
            observer: Mutex::new(None),
            stall: AtomicUsize::new(0),
        });
        let shared = Arc::clone(&state);
        let server = thread::spawn(move || {
            for stream in listener.incoming() {
                if shared.stop.load(Ordering::SeqCst) {
                    return;
                }
                let Ok(stream) = stream else { continue };
                let shared = Arc::clone(&shared);
                // A reply cut short is a client that was killed; nothing to report.
                thread::spawn(move || drop(serve(stream, &shared)));
            }
        });
        Ok(Self {
            addr,
            state,
            server: Some(server),
        })
    }

    fn url(&self) -> String {
        format!("http://{}/v1", self.addr)
    }

    /// Hold the `request`-th reply (1-based, counting every run) open after its
    /// first part until the cortex stops.
    fn stall(&self, request: usize) {
        self.state.stall.store(request, Ordering::SeqCst);
    }

    fn observe(&self, observer: Sender<Signal>) -> TestResult {
        *self
            .state
            .observer
            .lock()
            .map_err(|_| "cortex observer poisoned")? = Some(observer);
        Ok(())
    }

    fn requests(&self) -> TestResult<Vec<String>> {
        Ok(self
            .state
            .requests
            .lock()
            .map_err(|_| "cortex requests poisoned")?
            .clone())
    }
}

impl Drop for FakeCortex {
    fn drop(&mut self) {
        self.state.stop.store(true, Ordering::SeqCst);
        // Wake the blocking accept so the server sees the flag.
        drop(TcpStream::connect(self.addr));
        if let Some(server) = self.server.take() {
            drop(server.join());
        }
    }
}

/// Answer one streaming chat completion: read the request, send the first part of
/// the reply, announce it, wait, then send the rest.
fn serve(stream: TcpStream, state: &CortexState) -> TestResult {
    stream.set_read_timeout(Some(WAIT))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut length = 0;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Err("the request ended inside its headers".into());
        }
        let line = line.trim();
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            length = value.trim().parse()?;
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    let body = String::from_utf8(body)?;
    // The warm-up at startup only loads the model: answer it, but it is no thought,
    // so it is neither recorded nor numbered.
    if body.contains("\"max_tokens\":1") && body.contains("\"stream\":false") {
        let mut stream = stream;
        stream.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\
              Connection: close\r\n\r\n{}",
        )?;
        return Ok(());
    }
    let number = {
        let mut requests = state
            .requests
            .lock()
            .map_err(|_| "cortex requests poisoned")?;
        requests.push(body);
        requests.len()
    };

    let mut stream = stream;
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
    stream.write_all(format!("{head}{}", sse(REPLY_HEAD)).as_bytes())?;
    stream.flush()?;
    if let Some(observer) = state
        .observer
        .lock()
        .map_err(|_| "cortex observer poisoned")?
        .as_ref()
    {
        // A driver that already finished no longer listens.
        drop(observer.send(Signal::Streaming));
    }
    let gap = if state.stall.load(Ordering::SeqCst) == number {
        WAIT * 6
    } else {
        REPLY_GAP
    };
    let resume = Instant::now() + gap;
    while Instant::now() < resume && !state.stop.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(2));
    }
    stream.write_all(format!("{}data: [DONE]\n\n", sse(REPLY_TAIL)).as_bytes())?;
    stream.flush()?;
    stream.shutdown(Shutdown::Both)?;
    Ok(())
}

fn sse(content: &str) -> String {
    format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"{content}\"}}}}]}}\n\n")
}
