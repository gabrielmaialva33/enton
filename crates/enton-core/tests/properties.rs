//! Property-based tests for the brainstem reducer and its calibrated evidence.
//!
//! Generated tapes mix ticks whose clock mostly runs forward but sometimes
//! repeats or runs backward, body signals, speech cues with arbitrary
//! measurements (including NaN, infinities, negatives and values above one),
//! cortex replies for thoughts the organism issued, and playback of the
//! utterances it spoke. After every step the reducer invariants must hold on
//! both shipped profiles (and a variant with budgets a short tape can drain),
//! and a snapshot taken at a random step must restore an organism that decides
//! the rest of the tape exactly like the live one.
//!
//! Failures are never persisted next to the sources: a shrunk reproducer goes
//! into an explicit test instead.

use std::ops::RangeInclusive;

use enton_core::evidence::MIN_SD;
use enton_core::{
    Action, BodySignals, Budget, DirectedModel, Event, Evidence, Millis, Organism, Profile, Reason,
    Senses, SourceModel, SpeechCue, ThoughtId, TurnModel, UtteranceId, VoiceModel,
};
use proptest::prelude::*;
use proptest::sample::Index;
use proptest::test_runner::{Config, TestCaseError};

/// Tolerance for budget bounds, which accumulate f32 refills.
const EPSILON: f32 = 1e-3;

/// Longest generated tape: a few hundred events.
const MAX_TAPE: usize = 300;

/// Explicit case counts keep the suite fast in debug builds; failures are not
/// written to `proptest-regressions` files next to the sources.
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

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

/// A float no sensor should produce, but a buggy one might.
fn broken_float() -> impl Strategy<Value = f32> {
    prop_oneof![
        3 => prop::sample::select(vec![
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            f32::MAX,
            f32::MIN,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1.0 + f32::EPSILON,
        ]),
        // Unlike `any::<f32>()`, this includes infinities and NaNs of either sign.
        1 => prop::num::f32::ANY,
    ]
}

/// A normalized sensor reading: mostly in range, sometimes out of it or not a
/// number at all.
fn reading() -> impl Strategy<Value = f32> {
    prop_oneof![
        12 => 0.0_f32..=1.0,
        3 => -2.0_f32..=3.0,
        2 => broken_float(),
    ]
}

fn duration() -> impl Strategy<Value = u32> {
    prop_oneof![
        16 => 0_u32..=4_000,
        2 => 4_000_u32..=60_000,
        1 => any::<u32>(),
    ]
}

/// Optional readings of the four sensors: speaker, media, end of turn, directedness.
type Readings = (Option<f32>, Option<f32>, Option<f32>, Option<f32>);

/// Every field is set today; the struct update keeps this compiling when
/// `SpeechCue` gains a field, which then takes its default (sensor did not run).
#[allow(clippy::needless_update)]
fn speech_cue(
    energy: f32,
    duration_ms: u32,
    vad_confidence: f32,
    keyword: bool,
    (speaker_sim, media, turn_complete, directed): Readings,
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
        reading(),
        duration(),
        reading(),
        prop::bool::weighted(0.3),
        (
            prop::option::of(reading()),
            prop::option::of(reading()),
            prop::option::of(reading()),
            prop::option::of(reading()),
        ),
    )
        .prop_map(|(energy, duration_ms, vad, keyword, readings)| {
            speech_cue(energy, duration_ms, vad, keyword, readings)
        })
}

/// A body measurement in `range`, or, when `broken` is set, sometimes any float.
fn measurement(range: RangeInclusive<f32>, broken: bool) -> BoxedStrategy<f32> {
    if broken {
        prop_oneof![8 => range, 1 => broken_float()].boxed()
    } else {
        range.boxed()
    }
}

/// Body signals around the torpor limits of both profiles; with `broken`, some
/// readings are not finite.
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

/// How the adapter clock moves before an event.
#[derive(Debug, Clone, Copy)]
enum Clock {
    Forward(u64),
    Repeat,
    Backward(u64),
}

fn clock() -> impl Strategy<Value = Clock> {
    prop_oneof![
        12 => (1_u64..=1_500).prop_map(Clock::Forward),
        4 => (1_500_u64..=60_000).prop_map(Clock::Forward),
        // Hours pass now and then, so drives build up and budgets refill.
        1 => (60_000_u64..=7_200_000).prop_map(Clock::Forward),
        2 => Just(Clock::Repeat),
        1 => (1_u64..=20_000).prop_map(Clock::Backward),
    ]
}

/// What happens next, before it is bound to the thoughts and utterances that
/// exist at that point of the tape.
#[derive(Debug, Clone)]
enum Stimulus {
    Tick,
    Body(BodySignals),
    Speech(SpeechCue),
    Reply { thought: Index, text: String },
    PlaybackStarted(Index),
    PlaybackFinished(Index),
}

fn reply_text() -> impl Strategy<Value = String> {
    prop_oneof![
        4 => prop::sample::select(vec![
            "",
            "Sure.",
            "Enton here.",
            "I am enton!",
            "entonces",
            "ENTON?",
        ])
        .prop_map(str::to_owned),
        1 => any::<String>(),
    ]
}

fn stimulus(broken_body: bool) -> impl Strategy<Value = Stimulus> {
    prop_oneof![
        5 => Just(Stimulus::Tick),
        1 => body_signals(broken_body).prop_map(Stimulus::Body),
        8 => cue().prop_map(Stimulus::Speech),
        2 => (any::<Index>(), reply_text())
            .prop_map(|(thought, text)| Stimulus::Reply { thought, text }),
        2 => any::<Index>().prop_map(Stimulus::PlaybackStarted),
        2 => any::<Index>().prop_map(Stimulus::PlaybackFinished),
    ]
}

type Tape = Vec<(Clock, Stimulus)>;

fn tape(broken_body: bool) -> impl Strategy<Value = Tape> {
    prop::collection::vec((clock(), stimulus(broken_body)), 0..=MAX_TAPE)
}

// ---------------------------------------------------------------------------
// The world around the organism
// ---------------------------------------------------------------------------

/// What the adapters know: the clock, the thoughts the organism asked for and
/// the utterances it spoke, so replies and playback refer to real ones.
#[derive(Debug, Default)]
struct World {
    now: u64,
    thoughts: Vec<ThoughtId>,
    utterances: Vec<UtteranceId>,
    next_utterance: u64,
}

fn pick<T: Copy>(items: &[T], index: Index) -> Option<T> {
    (!items.is_empty()).then(|| *index.get(items))
}

impl World {
    /// Bind a stimulus to the current world. A reply or playback with nothing
    /// to refer to yet is a plain tick.
    fn event(&mut self, clock: Clock, stimulus: &Stimulus) -> Event {
        self.now = match clock {
            Clock::Forward(dt) => self.now.saturating_add(dt),
            Clock::Repeat => self.now,
            Clock::Backward(dt) => self.now.saturating_sub(dt),
        };
        let now = Millis(self.now);
        let tick = Event::Tick { now };
        match stimulus {
            Stimulus::Tick => tick,
            Stimulus::Body(signals) => Event::Body {
                now,
                signals: *signals,
            },
            Stimulus::Speech(cue) => Event::Speech { now, cue: *cue },
            // Any issued thought: the current one, or a late or repeated reply.
            Stimulus::Reply { thought, text } => {
                pick(&self.thoughts, *thought).map_or(tick, |thought| Event::CortexReply {
                    now,
                    thought,
                    text: text.clone(),
                })
            }
            Stimulus::PlaybackStarted(index) => pick(&self.utterances, *index)
                .map_or(tick, |utterance| Event::PlaybackStarted { now, utterance }),
            // Not necessarily the utterance playing: a stale finish must be ignored.
            Stimulus::PlaybackFinished(index) => pick(&self.utterances, *index)
                .map_or(tick, |utterance| Event::PlaybackFinished { now, utterance }),
        }
    }

    fn observe(&mut self, actions: &[Action]) {
        for action in actions {
            match action {
                Action::Think { thought, .. } => self.thoughts.push(*thought),
                Action::Speak { .. } => {
                    self.next_utterance += 1;
                    self.utterances.push(UtteranceId(self.next_utterance));
                }
                Action::Attend { .. } | Action::Abstain { .. } => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Invariants checked after every step
// ---------------------------------------------------------------------------

/// Tracks one organism across a tape and checks every step.
#[derive(Debug)]
struct Watch {
    next_thought: u64,
    last_seen: Millis,
    capacities: [f32; 2],
    /// Budgets after the previous step: a speech cue refills nothing, so it finds these.
    budgets: [Budget; 2],
}

impl Watch {
    fn new(organism: &Organism) -> Self {
        Self {
            next_thought: 1,
            last_seen: organism.last_seen(),
            capacities: budgets(organism).map(|budget| budget.capacity),
            budgets: budgets(organism),
        }
    }

    fn after_step(
        &mut self,
        organism: &Organism,
        event: &Event,
        actions: &[Action],
    ) -> Result<(), TestCaseError> {
        check_decisions(event, actions)?;
        check_actions(event, actions)?;
        self.check_thought_ids(actions)?;
        self.check_time(organism, event)?;
        self.check_budgets(organism)?;
        self.check_exploration(organism, event, actions)?;
        check_levels(organism)
    }

    /// A coin flip happens only at a speech cue, logs the applied exploration probability
    /// (a thought) or its complement (an abstention), and an explored thought is paid by
    /// the discretionary account alone.
    fn check_exploration(
        &mut self,
        organism: &Organism,
        event: &Event,
        actions: &[Action],
    ) -> Result<(), TestCaseError> {
        let [obligation, discretionary] = std::mem::replace(&mut self.budgets, budgets(organism));
        let explore = organism.profile().exploration.applied_probability();
        for action in actions {
            let (logged, expected, explored) = match action {
                Action::Think {
                    propensity: Some(logged),
                    ..
                } => (*logged, explore, true),
                Action::Abstain {
                    propensity: Some(logged),
                    ..
                } => (*logged, 1.0 - explore, false),
                _ => continue,
            };
            prop_assert!(
                matches!(event, Event::Speech { .. }) && explore > 0.0,
                "only a speech cue flips a coin, and only while exploring: {action:?}"
            );
            prop_assert_eq!(logged.to_bits(), expected.to_bits(), "{:?}", action);
            if explored {
                let cost = organism.profile().budgets.think_cost;
                prop_assert_eq!(
                    organism.obligation_budget().available.to_bits(),
                    obligation.available.to_bits(),
                    "an explored thought never touches the obligation account"
                );
                prop_assert_eq!(
                    organism.discretionary_budget().available.to_bits(),
                    (discretionary.available - cost).to_bits(),
                    "an explored thought is paid by the discretionary account"
                );
            }
        }
        Ok(())
    }

    fn check_thought_ids(&mut self, actions: &[Action]) -> Result<(), TestCaseError> {
        for action in actions {
            if let Action::Think { thought, .. } = action {
                prop_assert_eq!(
                    thought.0,
                    self.next_thought,
                    "thought IDs increase by exactly one"
                );
                self.next_thought += 1;
            }
        }
        Ok(())
    }

    fn check_time(&mut self, organism: &Organism, event: &Event) -> Result<(), TestCaseError> {
        let seen = organism.last_seen();
        prop_assert!(
            seen >= event.now() && seen >= self.last_seen,
            "last_seen went backward: {seen:?} after {:?}, event at {:?}",
            self.last_seen,
            event.now()
        );
        self.last_seen = seen;
        Ok(())
    }

    fn check_budgets(&self, organism: &Organism) -> Result<(), TestCaseError> {
        for (budget, capacity) in budgets(organism).into_iter().zip(self.capacities) {
            prop_assert_eq!(
                budget.capacity.to_bits(),
                capacity.to_bits(),
                "a budget's capacity never changes"
            );
            prop_assert!(
                (-EPSILON..=budget.capacity + EPSILON).contains(&budget.available),
                "a budget stays within zero and its capacity: {budget:?}"
            );
        }
        Ok(())
    }
}

fn budgets(organism: &Organism) -> [Budget; 2] {
    [
        *organism.obligation_budget(),
        *organism.discretionary_budget(),
    ]
}

fn is_decision(action: &Action) -> bool {
    matches!(
        action,
        Action::Think { .. } | Action::Attend { .. } | Action::Abstain { .. }
    )
}

fn reason(action: &Action) -> Option<&Reason> {
    match action {
        Action::Think { reason, .. } | Action::Abstain { reason, .. } => Some(reason),
        Action::Speak { .. } | Action::Attend { .. } => None,
    }
}

fn check_decisions(event: &Event, actions: &[Action]) -> Result<(), TestCaseError> {
    let decisions = actions.iter().filter(|action| is_decision(action)).count();
    let speaks = actions
        .iter()
        .filter(|action| matches!(action, Action::Speak { .. }))
        .count();
    match event {
        Event::Speech { .. } => prop_assert!(
            decisions == 1 && speaks == 0,
            "a speech cue yields exactly one decision and no Speak: {actions:?}"
        ),
        Event::CortexReply { text, .. } => prop_assert!(
            matches!(actions, [Action::Speak { text: spoken }] if spoken == text),
            "a cortex reply yields exactly its Speak and nothing else: {actions:?}"
        ),
        Event::Tick { .. } => {
            // A tick may only resolve a pending attend or act on a drive, once each.
            let keyword = actions
                .iter()
                .filter(|action| reason(action) == Some(&Reason::Keyword))
                .count();
            let drive = actions
                .iter()
                .filter(|action| matches!(reason(action), Some(Reason::Drive(_))))
                .count();
            prop_assert!(
                actions
                    .iter()
                    .all(|action| matches!(action, Action::Think { .. } | Action::Abstain { .. }))
                    && keyword <= 1
                    && drive <= 1
                    && keyword + drive == actions.len(),
                "a tick only thinks or abstains, for one timeout and one drive: {actions:?}"
            );
        }
        Event::Body { .. } | Event::PlaybackStarted { .. } | Event::PlaybackFinished { .. } => {
            prop_assert!(
                actions.is_empty(),
                "body and playback events decide nothing: {actions:?}"
            );
        }
    }
    Ok(())
}

/// Saliences land in audit logs as JSON, where a non-finite value is unreadable.
fn check_actions(event: &Event, actions: &[Action]) -> Result<(), TestCaseError> {
    for action in actions {
        match action {
            Action::Think { salience, .. } | Action::Abstain { salience, .. } => prop_assert!(
                salience.is_finite() && *salience >= 0.0,
                "a salience is finite and non-negative: {action:?}"
            ),
            Action::Attend { until } => prop_assert!(
                *until > event.now(),
                "an attend deadline lies after its cue: {action:?}"
            ),
            Action::Speak { .. } => {}
        }
    }
    Ok(())
}

fn check_levels(organism: &Organism) -> Result<(), TestCaseError> {
    for (name, level) in [
        ("habituation", organism.habituation()),
        ("slow habituation", organism.slow_habituation()),
        ("echo expectation", organism.echo_energy_expectation()),
        ("TV presence", organism.tv_presence()),
    ] {
        prop_assert!((0.0..=1.0).contains(&level), "{name} left [0, 1]: {level}");
    }
    let profile = organism.profile();
    let reach = profile.echo.echo_hangover_ms;
    let seen = organism.last_seen().0;
    for (until, span) in [
        (organism.attention_until(), profile.attention.attention_ms),
        (
            organism.verified_attention_until(),
            profile.attention.verified_attention_ms,
        ),
    ] {
        prop_assert!(
            until.is_none_or(|until| until.0 <= seen.saturating_add(span + reach)),
            "an attention window never reaches past its span: {until:?} at {seen} ms"
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------

/// `t1_ref` with budgets a short tape can drain, and a price that does not divide
/// them, so the bounds are tested near zero and `OutOfEnergy` is reached: the
/// shipped budgets outlast a few hundred events.
fn starved() -> Profile {
    let mut profile = Profile::t1_ref();
    "starved".clone_into(&mut profile.name);
    profile.budgets.think_cost = 1.5;
    profile.budgets.obligation_budget_per_hour = 4.0;
    profile.budgets.discretionary_budget_per_hour = 2.0;
    profile
}

/// `base` flipping a fair coin at any cue an objection turns away within three nats of
/// its threshold, so random tapes reach exploration often.
fn exploring(base: Profile) -> Profile {
    let mut profile = base;
    profile.exploration.explore_probability = 0.5;
    profile.exploration.explore_margin_nats = 3.0;
    profile.exploration.explore_seed = 11;
    profile
}

/// Run `tape` on `profile`, checking every step, then restore a JSON snapshot
/// taken before step `cut` and check that it decides the rest of the tape, and
/// ends, exactly like the live organism.
fn run_with_snapshot(profile: Profile, tape: &Tape, cut: Index) -> Result<(), TestCaseError> {
    let mut organism = Organism::new(profile).map_err(fail)?;
    let mut watch = Watch::new(&organism);
    let mut world = World::default();
    let cut = cut.index(tape.len() + 1);
    let mut snapshot = None;
    let mut events = Vec::with_capacity(tape.len());
    let mut decisions = Vec::with_capacity(tape.len());
    for (step, (clock, stimulus)) in tape.iter().enumerate() {
        if step == cut {
            snapshot = Some((
                serde_json::to_string(&organism).map_err(fail)?,
                organism.clone(),
            ));
        }
        let event = world.event(*clock, stimulus);
        let actions = organism.step(&event);
        watch
            .after_step(&organism, &event, &actions)
            .map_err(|error| fail(format!("step {step}, {event:?}: {error}")))?;
        world.observe(&actions);
        events.push(event);
        decisions.push(actions);
    }
    let (blob, at_cut) = match snapshot {
        Some(snapshot) => snapshot,
        None => (
            serde_json::to_string(&organism).map_err(fail)?,
            organism.clone(),
        ),
    };
    let mut restored: Organism = serde_json::from_str(&blob).map_err(fail)?;
    prop_assert_eq!(&restored, &at_cut, "a snapshot restores an equal organism");
    for (step, (event, expected)) in events.iter().zip(&decisions).enumerate().skip(cut) {
        prop_assert_eq!(
            &restored.step(event),
            expected,
            "the restored organism decided step {} differently",
            step
        );
    }
    prop_assert_eq!(
        &restored,
        &organism,
        "the restored organism ended elsewhere"
    );
    Ok(())
}

/// The soul stores `event.canonical()` as JSON (RFC P4): an organism fed the
/// stored events must decide exactly like the live one fed the raw events.
fn replay_from_json(profile: Profile, tape: &Tape) -> Result<(), TestCaseError> {
    let mut live = Organism::new(profile.clone()).map_err(fail)?;
    let mut replayed = Organism::new(profile).map_err(fail)?;
    let mut world = World::default();
    for (step, (clock, stimulus)) in tape.iter().enumerate() {
        let event = world.event(*clock, stimulus);
        let actions = live.step(&event);
        world.observe(&actions);
        let stored = serde_json::to_string(&event.clone().canonical()).map_err(fail)?;
        let read: Event = serde_json::from_str(&stored)
            .map_err(|error| fail(format!("step {step}: {stored} does not read back: {error}")))?;
        prop_assert_eq!(
            &replayed.step(&read),
            &actions,
            "step {}: {:?} stored as {} decides differently",
            step,
            event,
            stored
        );
    }
    prop_assert_eq!(&replayed, &live);
    Ok(())
}

proptest! {
    #![proptest_config(config(192))]

    /// Every step keeps the reducer invariants on both shipped profiles, a starved
    /// one and two exploring ones, and a snapshot at a random step replays the rest
    /// exactly, coin flips included.
    #[test]
    fn every_step_keeps_the_reducer_invariants(tape in tape(true), cut in any::<Index>()) {
        run_with_snapshot(Profile::t1_ref(), &tape, cut)?;
        run_with_snapshot(Profile::desktop(), &tape, cut)?;
        run_with_snapshot(starved(), &tape, cut)?;
        run_with_snapshot(exploring(Profile::t1_ref()), &tape, cut)?;
        run_with_snapshot(exploring(starved()), &tape, cut)?;
    }

    /// Canonical events read back from JSON decide exactly like the live ones, coin
    /// flips included. Body readings stay finite here; the next property covers
    /// broken ones.
    #[test]
    fn canonical_events_replay_from_json_like_they_ran_live(tape in tape(false)) {
        replay_from_json(Profile::t1_ref(), &tape)?;
        replay_from_json(Profile::desktop(), &tape)?;
        replay_from_json(exploring(Profile::t1_ref()), &tape)?;
    }

    /// Bug: `Event::canonical` canonicalizes speech cues only, so a body signal
    /// that is not finite is stored as JSON `null`:
    ///
    /// - shrunk: `[Body { now: 1, signals: { temperature_c: None, battery: None,
    ///   cpu_load: NaN } }]` stores `"cpu_load":null`, which does not read back
    ///   (`invalid type: null, expected f32`), so a soul holding it cannot replay;
    /// - `[Body { now: 1, temperature_c: Some(inf) }, Speech { now: 2, energy: 1.0,
    ///   vad_confidence: 1.0, duration_ms: 1500 }]` on `t1_ref`: the live organism
    ///   is in torpor and abstains, the replayed one reads `null` as `None` and
    ///   issues `ThoughtId(1)`; `battery: Some(-inf)` diverges the same way.
    #[test]
    fn body_events_replay_from_json_like_they_ran_live(tape in tape(true)) {
        replay_from_json(Profile::t1_ref(), &tape)?;
        replay_from_json(Profile::desktop(), &tape)?;
    }
}

// ---------------------------------------------------------------------------
// Evidence and canonical cues
// ---------------------------------------------------------------------------

fn llrs(evidence: &Evidence) -> [(&'static str, f32); 5] {
    [
        ("owner over other", evidence.owner_over_other),
        ("owner over reproduced", evidence.owner_over_reproduced),
        ("live over reproduced", evidence.live_over_reproduced),
        (
            "finished over unfinished",
            evidence.finished_over_unfinished,
        ),
        ("addressed over not", evidence.addressed_over_not),
    ]
}

/// Each ratio is finite and within the cap. `owner_live` adds two capped ratios
/// against the loudspeaker (the cap bounds "any single ratio"), so it may reach
/// twice the cap below zero, but never above one cap.
fn check_evidence(senses: &Senses, cue: &SpeechCue) -> Result<(), TestCaseError> {
    let evidence = senses.read(cue);
    let cap = senses.max_llr;
    for (name, llr) in llrs(&evidence) {
        prop_assert!(
            llr.is_finite() && llr.abs() <= cap,
            "{name} = {llr} is not within the cap {cap} for {cue:?}"
        );
    }
    let owner_live = evidence.owner_live();
    prop_assert!(
        owner_live.is_finite() && (-2.0 * cap..=cap).contains(&owner_live),
        "owner live = {owner_live} is not within [-2 cap, cap] for {cue:?}"
    );
    Ok(())
}

fn means() -> impl Strategy<Value = [f32; 3]> {
    [0.0_f32..=1.0, 0.0_f32..=1.0, 0.0_f32..=1.0]
}

/// Any calibration `Senses::is_valid` accepts.
fn senses() -> impl Strategy<Value = Senses> {
    let voice =
        (means(), means(), means(), MIN_SD..=1.0).prop_map(|(owner, other, reproduced, sd)| {
            VoiceModel {
                owner,
                other,
                reproduced,
                sd,
            }
        });
    let source = (means(), means(), MIN_SD..=1.0).prop_map(|(live, reproduced, sd)| SourceModel {
        live,
        reproduced,
        sd,
    });
    let row = || {
        [
            -1e30_f32..=1e30,
            -1e30_f32..=1e30,
            -1e30_f32..=1e30,
            -1e30_f32..=1e30,
        ]
    };
    let turn = [row(), row(), row()].prop_map(|llr| TurnModel { llr });
    let directed = [-1e30_f32..=1e30, -1e30_f32..=1e30, -1e30_f32..=1e30]
        .prop_map(|llr| DirectedModel { llr });
    let max_llr = prop_oneof![f32::MIN_POSITIVE..=10.0, Just(f32::MAX)];
    (voice, source, turn, directed, max_llr).prop_map(|(voice, source, turn, directed, max_llr)| {
        Senses {
            voice,
            source,
            turn,
            directed,
            max_llr,
        }
    })
}

proptest! {
    #![proptest_config(config(1024))]

    /// Every ratio the shipped calibration reads from any cue is finite and capped.
    #[test]
    fn calibrated_evidence_is_finite_and_capped(cue in cue()) {
        check_evidence(&Senses::calibrated(), &cue)?;
    }

    /// The same holds for any calibration that passes validation.
    #[test]
    fn any_valid_calibration_reads_finite_capped_evidence(senses in senses(), cue in cue()) {
        prop_assert!(senses.is_valid(), "the generator builds valid calibrations");
        check_evidence(&senses, &cue)?;
    }

    /// A sensor that did not run, or read something that is not a number, says nothing.
    #[test]
    fn a_sensor_that_did_not_run_says_nothing(cue in cue()) {
        let evidence = Senses::calibrated().read(&cue);
        let ran = |reading: Option<f32>| reading.is_some_and(f32::is_finite);
        if !ran(cue.speaker_sim) {
            prop_assert!(evidence.owner_over_other == 0.0 && evidence.owner_over_reproduced == 0.0);
        }
        if !ran(cue.media) {
            prop_assert!(evidence.live_over_reproduced == 0.0);
        }
        if !ran(cue.turn_complete) {
            prop_assert!(evidence.finished_over_unfinished == 0.0);
        }
        if !ran(cue.directed) {
            prop_assert!(evidence.addressed_over_not == 0.0);
        }
        prop_assert_eq!(evidence, Senses::calibrated().read(&cue.canonical()));
    }

    /// In the shipped calibration the owner scores above other voices and
    /// loudspeakers, a loudspeaker scores above a live voice, and a higher
    /// end-of-turn score is never weaker evidence of a finished turn: a higher
    /// reading can only move each ratio its own way.
    #[test]
    fn calibrated_evidence_is_monotonic_in_each_reading(
        duration_ms in duration(),
        low in -1.0_f32..=2.0,
        high in -1.0_f32..=2.0,
    ) {
        let (low, high) = if low <= high { (low, high) } else { (high, low) };
        let senses = Senses::calibrated();
        let read = |reading: f32| {
            let reading = Some(reading);
            senses.read(&speech_cue(0.5, duration_ms, 0.5, false, (reading, reading, reading, reading)))
        };
        let (a, b) = (read(low), read(high));
        prop_assert!(a.owner_over_other <= b.owner_over_other, "{a:?} vs {b:?}");
        prop_assert!(a.owner_over_reproduced <= b.owner_over_reproduced, "{a:?} vs {b:?}");
        prop_assert!(a.live_over_reproduced >= b.live_over_reproduced, "{a:?} vs {b:?}");
        prop_assert!(a.finished_over_unfinished <= b.finished_over_unfinished, "{a:?} vs {b:?}");
        prop_assert!(a.addressed_over_not <= b.addressed_over_not, "{a:?} vs {b:?}");
    }

    /// Canonicalization is idempotent, lands every reading in the unit interval,
    /// keeps what it does not measure, and never alters a reading already valid.
    #[test]
    fn canonical_cues_are_idempotent_and_in_range(cue in cue()) {
        let once = cue.canonical();
        prop_assert_eq!(once.canonical(), once);
        prop_assert_eq!(once.duration_ms, cue.duration_ms);
        prop_assert_eq!(once.keyword, cue.keyword);
        let unit = |value: f32| value.is_finite() && (0.0..=1.0).contains(&value);
        for (raw, canonical) in [
            (Some(cue.energy), Some(once.energy)),
            (Some(cue.vad_confidence), Some(once.vad_confidence)),
            (cue.speaker_sim, once.speaker_sim),
            (cue.media, once.media),
            (cue.turn_complete, once.turn_complete),
            (cue.directed, once.directed),
        ] {
            prop_assert!(canonical.is_none_or(unit), "{canonical:?} from {raw:?}");
            if let Some(raw) = raw.filter(|raw| unit(*raw)) {
                prop_assert_eq!(canonical.map(f32::to_bits), Some(raw.to_bits()));
            }
        }
    }

    /// A canonical cue is exactly what JSON reads back, bit for bit.
    #[test]
    fn canonical_cues_survive_json_exactly(cue in cue()) {
        let canonical = cue.canonical();
        let json = serde_json::to_string(&canonical).map_err(fail)?;
        let read: SpeechCue = serde_json::from_str(&json).map_err(fail)?;
        prop_assert_eq!(read, canonical, "{}", json);
    }
}
