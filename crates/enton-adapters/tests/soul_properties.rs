//! Property-based tests for the soul's durable encodings: every appended event
//! reads back in canonical form and replays to the live organism, and no
//! snapshot blob or stored event payload, however corrupt, panics a replay or
//! restores an organism it should not.
#![cfg(feature = "soul")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use enton_adapters::{Soul, SoulConfig};
use enton_core::{
    Action, BodySignals, Event, Millis, Organism, Profile, SpeechCue, ThoughtId, UtteranceId,
};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::Index;
use proptest::test_runner::{Config, TestCaseError, TestRunner};
use serde_json::Value;

/// Explicit case counts keep the suite fast in debug builds (every append is a
/// synchronous commit); failures are not written next to the sources.
fn config(cases: u32) -> Config {
    Config {
        cases,
        failure_persistence: None,
        ..Config::default()
    }
}

fn fail(error: impl std::fmt::Display) -> TestCaseError {
    TestCaseError::fail(error.to_string())
}

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

/// A scratch directory for one database, removed on drop.
struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> std::io::Result<Self> {
        let id = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("enton-soul-properties-{}-{id}", std::process::id()));
        std::fs::create_dir_all(&path)?;
        Ok(Self(path))
    }

    fn database(&self) -> PathBuf {
        self.0.join("soul.sqlite")
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        // Best effort: a leftover temporary directory must not fail a property.
        drop(std::fs::remove_dir_all(&self.0));
    }
}

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

/// A normalized reading: mostly in range, sometimes out of it or not a number.
fn reading() -> impl Strategy<Value = f32> {
    prop_oneof![
        8 => 0.0_f32..=1.0,
        1 => -2.0_f32..=3.0,
        1 => prop::num::f32::ANY,
    ]
}

/// Every field is set today; the struct update keeps this compiling when
/// `SpeechCue` gains a field, which then takes its default (sensor did not run).
#[allow(clippy::needless_update)]
fn speech_cue(
    (energy, duration_ms, vad_confidence, keyword): (f32, u32, f32, bool),
    (speaker_sim, media, turn_complete, directed): (
        Option<f32>,
        Option<f32>,
        Option<f32>,
        Option<f32>,
    ),
) -> SpeechCue {
    SpeechCue {
        energy,
        duration_ms,
        vad_confidence,
        keyword,
        speaker_sim,
        media,
        turn_complete,
        directed,
        ..SpeechCue::default()
    }
}

fn cue() -> impl Strategy<Value = SpeechCue> {
    (
        (
            reading(),
            0_u32..=4_000,
            reading(),
            prop::bool::weighted(0.3),
        ),
        (
            prop::option::of(reading()),
            prop::option::of(reading()),
            prop::option::of(reading()),
            prop::option::of(reading()),
        ),
    )
        .prop_map(|(levels, sensors)| speech_cue(levels, sensors))
}

/// A body measurement in `range`, or, when `broken` is set, sometimes any float.
fn measurement(range: std::ops::RangeInclusive<f32>, broken: bool) -> BoxedStrategy<f32> {
    if broken {
        prop_oneof![6 => range, 1 => prop::num::f32::ANY].boxed()
    } else {
        range.boxed()
    }
}

fn body_signals(broken: bool) -> impl Strategy<Value = BodySignals> {
    (
        prop::option::of(measurement(20.0..=110.0, broken)),
        prop::option::of(measurement(-0.2..=1.2, broken)),
        measurement(0.0..=4.0, broken),
    )
        .prop_map(|(temperature_c, battery, cpu_load)| BodySignals {
            temperature_c,
            battery,
            cpu_load,
        })
}

/// An event without its time yet.
#[derive(Debug, Clone)]
enum Kind {
    Tick,
    Body(BodySignals),
    Speech(SpeechCue),
    Reply(ThoughtId, String),
    Started(UtteranceId),
    Finished(UtteranceId),
}

impl Kind {
    fn at(self, now: Millis) -> Event {
        match self {
            Kind::Tick => Event::Tick { now },
            Kind::Body(signals) => Event::Body { now, signals },
            Kind::Speech(cue) => Event::Speech { now, cue },
            Kind::Reply(thought, text) => Event::CortexReply { now, thought, text },
            Kind::Started(utterance) => Event::PlaybackStarted { now, utterance },
            Kind::Finished(utterance) => Event::PlaybackFinished { now, utterance },
        }
    }
}

fn kind(broken_body: bool) -> impl Strategy<Value = Kind> {
    prop_oneof![
        3 => Just(Kind::Tick),
        1 => body_signals(broken_body).prop_map(Kind::Body),
        5 => cue().prop_map(Kind::Speech),
        1 => (1_u64..=4, any::<String>())
            .prop_map(|(thought, text)| Kind::Reply(ThoughtId(thought), text)),
        1 => (1_u64..=3).prop_map(|id| Kind::Started(UtteranceId(id))),
        1 => (1_u64..=3).prop_map(|id| Kind::Finished(UtteranceId(id))),
    ]
}

/// Up to 40 events: each append is a synchronous commit.
fn tape(broken_body: bool) -> impl Strategy<Value = Vec<Event>> {
    vec((0_u64..=3_000, kind(broken_body)), 0..=40).prop_map(|steps| {
        let mut now = 0_u64;
        steps
            .into_iter()
            .map(|(dt, kind)| {
                now += dt;
                kind.at(Millis(now))
            })
            .collect()
    })
}

// ---------------------------------------------------------------------------
// Appended events read back canonical and replay to the live organism
// ---------------------------------------------------------------------------

/// Append `events` while a live organism reduces them, snapshot it after `cut`
/// events, then read the log back and replay it.
fn append_read_replay(events: &[Event], cut: Index) -> Result<(), TestCaseError> {
    let directory = TestDirectory::new().map_err(fail)?;
    let soul = Soul::open(directory.database(), SoulConfig::default()).map_err(fail)?;
    let profile = Profile::t1_ref();
    let mut live = Organism::new(profile.clone()).map_err(fail)?;
    let cut = cut.index(events.len() + 1);
    let mut tail = Vec::new();
    for (index, event) in events.iter().enumerate() {
        let seq = soul.append_event(event).map_err(fail)?;
        prop_assert_eq!(
            Some(seq),
            u64::try_from(index + 1).ok(),
            "sequence numbers count from one"
        );
        let actions = live.step(event);
        if index < cut {
            if index + 1 == cut {
                soul.save_organism_snapshot(seq, &live).map_err(fail)?;
            }
        } else {
            tail.extend(actions);
        }
    }

    let read = soul
        .read_after(0, events.len() + 1)
        .map_err(|error| fail(format!("the log does not read back: {error}")))?;
    prop_assert_eq!(read.len(), events.len());
    for ((_, stored), event) in read.iter().zip(events) {
        prop_assert_eq!(
            stored,
            &event.clone().canonical(),
            "an event reads back canonical"
        );
    }

    let (restored, replayed) = soul
        .replay_organism(&profile)
        .map_err(|error| fail(format!("the log does not replay: {error}")))?;
    prop_assert_eq!(
        &replayed,
        &tail,
        "the replayed tail decides like the live one"
    );
    prop_assert_eq!(
        &restored,
        &live,
        "the replay ends where the live organism did"
    );
    Ok(())
}

proptest! {
    #![proptest_config(config(24))]

    /// Every appended event reads back as its canonical form, and a snapshot plus
    /// the replayed tail reproduce the live organism and its decisions. Body
    /// readings stay finite here; the next property covers broken ones.
    #[test]
    fn appended_events_read_back_canonical_and_replay(events in tape(false), cut in any::<Index>()) {
        append_read_replay(&events, cut)?;
    }

    /// Bug: `append_event` stores `event.canonical()`, which only canonicalizes
    /// speech cues, so a body signal that is not finite is written as JSON `null`:
    ///
    /// - shrunk: `[Body { now: 0, signals: { temperature_c: Some(inf), battery:
    ///   None, cpu_load: 0.0 } }]` reads back with `temperature_c: None`, so the
    ///   replayed organism is not in torpor where the live one was;
    /// - appending `Body { signals: { cpu_load: NaN, .. }, .. }` succeeds, and from
    ///   then on `read_after` and `replay_organism` fail with `invalid type: null,
    ///   expected f32`: the organism can no longer be restored from its soul.
    ///
    /// Same root cause as the core property
    /// `body_events_replay_from_json_like_they_ran_live`.
    #[test]
    fn appended_body_events_read_back_canonical_and_replay(events in tape(true), cut in any::<Index>()) {
        append_read_replay(&events, cut)?;
    }
}

// ---------------------------------------------------------------------------
// Corrupt snapshots and payloads
// ---------------------------------------------------------------------------

fn speech(now: u64, keyword: bool, duration_ms: u32, energy: f32) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: speech_cue(
            (energy, duration_ms, 0.9, keyword),
            (None, None, None, None),
        ),
    }
}

/// A short life the snapshot is taken at the end of: a conversation, then an
/// unfinished "Enton?" still pending.
fn lived() -> Vec<Event> {
    vec![
        Event::Tick { now: Millis(1_000) },
        speech(1_100, true, 1_600, 0.8),
        Event::CortexReply {
            now: Millis(1_200),
            thought: ThoughtId(1),
            text: "Enton here.".to_owned(),
        },
        speech(1_300, false, 700, 0.6),
        speech(1_400, true, 300, 0.8),
    ]
}

/// What the restored organism must still reduce: the pending turn times out, a
/// request is answered and interrupted twice, and an hour passes.
fn tail() -> Vec<Event> {
    vec![
        Event::Tick { now: Millis(7_000) },
        speech(7_100, true, 1_600, 0.8),
        Event::PlaybackStarted {
            now: Millis(7_200),
            utterance: UtteranceId(1),
        },
        speech(7_300, false, 1_200, 1.0),
        speech(7_400, false, 1_200, 1.0),
        Event::Tick {
            now: Millis(3_607_400),
        },
    ]
}

/// A soul holding `lived()`, a valid organism snapshot after it, and `tail()`.
/// Returns the soul, the snapshot's sequence number and its blob.
fn soul_with_snapshot(
    directory: &TestDirectory,
    profile: &Profile,
) -> Result<(Soul, u64, Vec<u8>), TestCaseError> {
    let soul = Soul::open(directory.database(), SoulConfig::default()).map_err(fail)?;
    let mut organism = Organism::new(profile.clone()).map_err(fail)?;
    let mut seq = 0;
    for event in lived() {
        seq = soul.append_event(&event).map_err(fail)?;
        organism.step(&event);
    }
    soul.save_organism_snapshot(seq, &organism).map_err(fail)?;
    for event in tail() {
        soul.append_event(&event).map_err(fail)?;
    }
    let (_, blob) = soul
        .latest_snapshot()
        .map_err(fail)?
        .ok_or_else(|| fail("the snapshot was saved"))?;
    Ok((soul, seq, blob))
}

/// JSON pointers to every number in `value`.
fn numbers(value: &Value, at: &str, out: &mut Vec<String>) {
    match value {
        Value::Number(_) => out.push(at.to_owned()),
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                numbers(item, &format!("{at}/{index}"), out);
            }
        }
        Value::Object(fields) => {
            for (key, field) in fields {
                let key = key.replace('~', "~0").replace('/', "~1");
                numbers(field, &format!("{at}/{key}"), out);
            }
        }
        Value::Null | Value::Bool(_) | Value::String(_) => {}
    }
}

/// `json` with the number `at` picks replaced by `replacement`.
fn replace_number(mut json: Value, at: Index, replacement: &Value) -> Value {
    let mut pointers = Vec::new();
    numbers(&json, "", &mut pointers);
    if !pointers.is_empty()
        && let Some(slot) = json.pointer_mut(at.get(&pointers))
    {
        *slot = replacement.clone();
    }
    json
}

/// What a corrupt or hand-edited record might hold in place of a number. The
/// counters' own limits are left to `a_restored_thought_counter_keeps_counting`.
fn odd_number() -> impl Strategy<Value = Value> {
    prop::sample::select(vec![
        Value::from(0),
        Value::from(-1),
        Value::from(u64::from(u32::MAX) + 1),
        Value::from(1_u64 << 53),
        Value::from(i64::MIN),
        Value::from(0.5),
        Value::from(1e300),
        Value::from(-1e300),
        Value::Null,
        Value::from("1"),
    ])
}

/// How a snapshot blob gets corrupted.
#[derive(Debug, Clone)]
enum Corruption {
    /// Bytes that were never a snapshot.
    Garbage(Vec<u8>),
    /// A snapshot cut short.
    Truncated(Index),
    /// One byte overwritten.
    Byte(Index, u8),
    /// One number replaced.
    Number(Index, Value),
}

fn corruption() -> impl Strategy<Value = Corruption> {
    prop_oneof![
        vec(any::<u8>(), 0..256).prop_map(Corruption::Garbage),
        any::<Index>().prop_map(Corruption::Truncated),
        (any::<Index>(), any::<u8>()).prop_map(|(at, byte)| Corruption::Byte(at, byte)),
        (any::<Index>(), odd_number()).prop_map(|(at, value)| Corruption::Number(at, value)),
    ]
}

/// The corrupted blob, and whether it can no longer be a valid snapshot.
fn corrupt(valid: &[u8], corruption: &Corruption) -> Result<(Vec<u8>, bool), TestCaseError> {
    Ok(match corruption {
        Corruption::Garbage(bytes) => (bytes.clone(), true),
        Corruption::Truncated(at) => (
            valid.iter().take(at.index(valid.len())).copied().collect(),
            true,
        ),
        Corruption::Byte(at, byte) => {
            let mut bytes = valid.to_vec();
            if let Some(slot) = bytes.get_mut(at.index(valid.len())) {
                *slot = *byte;
            }
            (bytes, false)
        }
        Corruption::Number(at, replacement) => {
            let json = serde_json::from_slice(valid).map_err(fail)?;
            let json = replace_number(json, *at, replacement);
            (serde_json::to_vec(&json).map_err(fail)?, false)
        }
    })
}

/// A corrupt snapshot never panics a replay. Garbage and truncated blobs never
/// restore; a blob that still parses restores only with the requested profile;
/// the valid blob always restores. Blobs are stored and returned byte for byte.
#[test]
fn a_corrupt_snapshot_never_panics_or_restores_garbage() {
    let directory = TestDirectory::new().unwrap();
    let profile = Profile::t1_ref();
    let (soul, seq, valid) = soul_with_snapshot(&directory, &profile).unwrap();
    let mut runner = TestRunner::new(config(256));
    runner
        .run(&corruption(), |corruption| {
            let (blob, garbage) = corrupt(&valid, &corruption)?;
            soul.save_snapshot(seq, &blob).map_err(fail)?;
            let stored = soul.latest_snapshot().map_err(fail)?;
            prop_assert_eq!(stored, Some((seq, blob.clone())), "blobs are opaque");
            match soul.replay_organism(&profile) {
                Ok((organism, _)) => {
                    prop_assert!(!garbage, "garbage restored an organism");
                    prop_assert_eq!(organism.profile(), &profile);
                }
                Err(error) => prop_assert!(blob != valid, "the valid snapshot failed: {error}"),
            }
            Ok(())
        })
        .unwrap();
}

/// How a stored event payload gets corrupted on disk, and which one.
#[derive(Debug, Clone)]
enum Payload {
    /// Arbitrary bytes, stored as a blob.
    Bytes(Index, Vec<u8>),
    /// Arbitrary text.
    Text(Index, String),
    /// The stored payload with one number replaced.
    Number(Index, Index, Value),
}

fn payload() -> impl Strategy<Value = Payload> {
    prop_oneof![
        (any::<Index>(), vec(any::<u8>(), 0..128))
            .prop_map(|(event, bytes)| Payload::Bytes(event, bytes)),
        (any::<Index>(), any::<String>()).prop_map(|(event, text)| Payload::Text(event, text)),
        (any::<Index>(), any::<Index>(), odd_number())
            .prop_map(|(event, at, value)| Payload::Number(event, at, value)),
    ]
}

/// Overwrite the payload of `seq` through a second connection, as a bad disk would.
fn overwrite(
    database: &rusqlite::Connection,
    seq: u64,
    payload: &rusqlite::types::Value,
) -> Result<(), TestCaseError> {
    let seq = i64::try_from(seq).map_err(fail)?;
    database
        .execute(
            "UPDATE events SET payload_json = ?1 WHERE seq = ?2",
            rusqlite::params![payload, seq],
        )
        .map_err(fail)?;
    Ok(())
}

/// A corrupt event payload in the replayed tail never panics reading or
/// replaying the log, and bytes that are not text never read back as an event.
#[test]
fn a_corrupt_event_payload_never_panics_a_replay() {
    let directory = TestDirectory::new().unwrap();
    let profile = Profile::t1_ref();
    let (soul, snapshot_seq, _) = soul_with_snapshot(&directory, &profile).unwrap();
    // Only the tail after the snapshot is replayed, so that is what gets corrupted.
    let stored: Vec<(u64, String)> = soul
        .read_after(snapshot_seq, 64)
        .unwrap()
        .iter()
        .map(|(seq, event)| (*seq, serde_json::to_string(event).unwrap()))
        .collect();
    let database = rusqlite::Connection::open(directory.database()).unwrap();
    let mut runner = TestRunner::new(config(128));
    runner
        .run(&payload(), |payload| {
            for (seq, json) in &stored {
                overwrite(&database, *seq, &json.clone().into())?;
            }
            let (event, corrupted) = match &payload {
                Payload::Bytes(event, bytes) => (event, bytes.clone().into()),
                Payload::Text(event, text) => (event, text.clone().into()),
                Payload::Number(event, at, replacement) => {
                    let json = serde_json::from_str(&event.get(&stored).1).map_err(fail)?;
                    (
                        event,
                        replace_number(json, *at, replacement).to_string().into(),
                    )
                }
            };
            overwrite(&database, event.get(&stored).0, &corrupted)?;
            let read = soul.read_after(0, 64);
            let replay = soul.replay_organism(&profile);
            if matches!(payload, Payload::Bytes(..)) {
                prop_assert!(read.is_err(), "a blob payload read back: {read:?}");
                prop_assert!(replay.is_err(), "a blob payload replayed");
            }
            Ok(())
        })
        .unwrap();
}

/// The actions returned for a restored snapshot are its tail's, never the prefix's.
#[test]
fn a_restored_snapshot_replays_only_its_tail() {
    let directory = TestDirectory::new().unwrap();
    let profile = Profile::t1_ref();
    let (soul, _, _) = soul_with_snapshot(&directory, &profile).unwrap();
    let mut live = Organism::new(profile.clone()).unwrap();
    for event in lived() {
        live.step(&event);
    }
    let expected: Vec<Action> = tail().iter().flat_map(|event| live.step(event)).collect();
    assert!(
        expected
            .iter()
            .any(|action| matches!(action, Action::Think { .. })),
        "the tail thinks, so the thought counter is exercised"
    );
    let (restored, replayed) = soul.replay_organism(&profile).unwrap();
    assert_eq!(replayed, expected);
    assert_eq!(restored, live);
}

fn thought_counter() -> impl Strategy<Value = u64> {
    prop_oneof![any::<u64>(), (u64::MAX - 8)..=u64::MAX]
}

proptest! {
    #![proptest_config(config(32))]

    /// A snapshot restores any thought counter, and the replayed tail keeps
    /// counting from it by exactly one.
    ///
    /// Bug: `Organism` increments `next_thought` with `+= 1`, and a restored
    /// snapshot is trusted as is, so a corrupt counter at its limit panics the
    /// replay with "attempt to add with overflow" in debug builds, against the
    /// rule that library code never panics; release builds wrap to `ThoughtId(0)`
    /// and reuse idempotency keys. Shrunk reproducer: `next_thought` set to
    /// 18446744073709551612 (`u64::MAX - 3`) in the snapshot of `lived()`, then
    /// `replay_organism` over `tail()`, whose fourth thought overflows.
    #[test]
    fn a_restored_thought_counter_keeps_counting(next in thought_counter()) {
        let directory = TestDirectory::new().map_err(fail)?;
        let profile = Profile::t1_ref();
        let (soul, seq, valid) = soul_with_snapshot(&directory, &profile)?;
        let mut json: Value = serde_json::from_slice(&valid).map_err(fail)?;
        let counter = json
            .pointer_mut("/state/next_thought")
            .ok_or_else(|| fail("the snapshot holds the thought counter"))?;
        *counter = Value::from(next);
        soul.save_snapshot(seq, &serde_json::to_vec(&json).map_err(fail)?).map_err(fail)?;
        let (_, replayed) = soul.replay_organism(&profile).map_err(fail)?;
        let mut expected = Some(next);
        for action in &replayed {
            if let Action::Think { thought, .. } = action {
                prop_assert_eq!(Some(thought.0), expected, "thought IDs count by one");
                expected = thought.0.checked_add(1);
            }
        }
    }
}
