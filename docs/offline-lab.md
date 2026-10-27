# Offline laboratory

Develop and debug Enton without capturing audio, running a language model or changing live services. This laboratory complements the microphone tests; it does not replace acoustic measurements or evaluation of real answers.

## Local operational checks

From the repository root:

```bash
cargo run --locked -p enton -- doctor --profile desktop
cargo run --locked -p enton -- doctor --profile desktop --json
```

`doctor` validates the selected built-in profile, loads the selected/default persona without printing its contents, and audits the selected/default soul using the existing read-only SQLite API. It never starts the live runtime, warms up the cortex, captures audio, speaks, migrates a schema or repairs a file.

Use `--soul PATH` and `--persona PATH` to check explicit files. The profile must match the one that wrote a soul snapshot. Default paths follow the existing XDG/HOME conventions. A missing default persona uses the built-in persona; an explicitly requested missing persona is an error.

JSON has `schema_version`, `mode`, `profile` and `checks`. Each check has a name, status and message:

| Status | Meaning |
| --- | --- |
| `ok` | The specified local check passed. |
| `warning` | A fresh default soul is absent, or the bounded audit is incomplete. |
| `error` | A requested file is absent/unusable, or validation failed. |
| `not_evaluated` | No measurement was attempted for this capability. |

Any error returns a nonzero exit code. Warnings alone return success and must not be interpreted as full readiness. An explicitly named missing soul is an error; a missing default soul is a warning. Neither creates directories or a database.

The soul audit checks the compatible replay anchor and up to 100,000 retained events in pages of 256. Exceeding that cap produces an explicit warning, not a clean integrity result. `enton why` remains available for a full replay audit. Closed fixture databases are tested byte-for-byte unchanged, with no new SQLite sidecars. Existing SQLite WAL-reader semantics apply to a log that is actively being written.

Network reachability, model availability/answer quality, audio, physical latency, 24-hour memory stability and power consumption are explicitly **not evaluated**. The command is a local diagnostic, not a live health check or an E1 pass certificate.

## E1 episode diagnosis

```bash
cargo run --locked -p enton-e1 --bin e1-sim -- \
  --seed 100 --diagnose e1a --episode 1

cargo run --locked -p enton-e1 --bin e1-sim -- \
  --seed 100 --diagnose e1a --episode 1 --json
```

Select `e1a` or `e1b`, one seed, and the actual episode ID from the tape. Only episodes with response-required turns qualify; missing/distractor-only IDs fail rather than selecting another episode. Use `--with-directed` or `--with-direction` to retain the corresponding simulated sensors. Incompatible `--summary`/`--off-policy` combinations are rejected.

The entire tape executes through the original E1 engine, including earlier TV beliefs, budgets, feedback, playback and the drain interval. The displayed interval begins at the selected episode's first speech event and ends at its latest response deadline. Earlier events still affect the result; they are not discarded to make a small isolated simulation.

The report includes:

- Original turns, segment IDs, availability/deadlines and `served`/`MISSED` outcomes from actual paid-call credits.
- Masked events, actual actions and paid calls, including intervening distractors and feedback in the displayed interval.
- Pre-event attention, budgets, TV belief and sensor evidence, computed from the same masked inputs the organism receives.
- The full profile, sensors, benchmark/reducer versions and full-tape organism result for comparison.

At most 512 steps are retained. `omitted_steps` reports every additional step in the interval. A truncated display is never a complete causal trace, although the full execution and turn scoring still finish. The diagnostic JSON format has its own `schema_version`; benchmark distributions/scoring remain at their existing version.

These are synthetic **paid-call service proxies**, not proof that a real model answered correctly. Evidence is what the reducer was given; it is not a claim that a simulated sensor measured the physical room.

## Reproducible tests

```bash
# Episode repeatability, sensor masking and equality with ordinary execution
CARGO_BUILD_JOBS=2 cargo test --locked -p enton-e1 --test diagnostics -- --test-threads=2

# Deterministic runtime scenarios, no live model or audio device
CARGO_BUILD_JOBS=2 cargo test --locked -p enton --bin enton runtime::offline_tests -- --test-threads=2

# Doctor in isolated temporary configuration/data directories
CARGO_BUILD_JOBS=2 cargo test --locked -p enton --test doctor -- --test-threads=2

# Existing calibration report (not threshold tuning)
cargo run --locked -p enton-e1 --bin e1-sim -- --seeds 100..=131 --summary
```

Runtime scenarios feed timestamped events and controlled completions into the real runtime. They use a current-thread executor and abort spawned cortex tasks before yielding, so inference is never performed. Existing loopback-server integration tests still cover the HTTP boundary separately.

One reproduced regression is covered: a quiet command cancelled the core's pending request but left its partial transcript in the runtime. The next accepted request could inherit those abandoned words. Runtime cleanup now follows cancellation, while preserving a legitimate timeout dispatch and keeping explicit requests alive when quiet hours begin. Other scenarios cover rejected noise, stale completions and failed thoughts.

## Evaluation discipline

Use calibration seeds 100 through 131 for development. Keep report seeds 0 through 31 separate, and do not use held-out seeds >=1000 without the freeze owner's authorization and manifest. Diagnostic support does not authorize tuning on held-out cases.

Do not alter thresholds, population labels, deadlines or success criteria to obtain a favorable report. Compare ordinary and diagnosed runs before changing policy. The offline work here does not claim to have solved the existing E1 request misses or measured the physical criteria.
