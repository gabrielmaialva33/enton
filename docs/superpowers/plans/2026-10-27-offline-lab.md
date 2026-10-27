# Offline Laboratory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Opt-in episode diagnosis, deterministic runtime regressions and a read-only offline doctor.

**Architecture:** Reuse the E1 execution loop and soul audit; keep diagnostics separate from policy/inference.

**Tech Stack:** Existing Rust, serde/serde_json, Tokio and SQLite adapter; no new production dependencies.

**Spec:** `docs/superpowers/specs/2026-10-27-offline-lab-design.md`

## Global Constraints
English code/docs; no microphone, inference, installs, service changes, commits, branches or pushes. Preserve E1 scoring/distributions/thresholds. Calibration seeds 100..131, no held-out seeds >=1000. Bound traces and label unmeasured checks.

## Review Focus
Invalid episode selectors; masked sensors in evidence; stale partial transcripts after quiet commands; absent default versus explicitly requested soul paths; secret-free errors and JSON with unmeasured physical checks.

## Task 1: E1 diagnosis
**Files:** `crates/enton-e1/src/{diagnostics,run,lib,main}.rs`, `tests/diagnostics.rs`.
**Interface:** `diagnose_episode(&Tape, &Profile, Sensors, EpisodeId) -> Result<Diagnosis, Error>`; CLI `--diagnose e1a|e1b --episode N --seed N [--json]`.
- [x] Add CLI tests and observe unknown-option failure.
- [x] Add a bounded optional execution observer and per-turn outcomes from actual credits.
- [x] Wire single-seed mode; reject incomplete/incompatible selectors.
- [x] Prove deterministic output, unchanged results and sensor masking.

## Task 2: Runtime scenarios and regression
**Files:** `crates/enton/src/runtime.rs`, `runtime/offline_tests.rs`.
- [x] Reproduce retention of an abandoned partial request across quiet mode.
- [x] Synchronize transcript lifetime with core pending state, preserving legitimate timeout input.
- [x] Test rejection, quiet cancellation, timeout, stale completion and failure without inference.

## Task 3: Offline doctor
**Files:** `crates/enton/src/{doctor,cli,main}.rs`, `tests/doctor.rs`.
**Interface:** `enton doctor [--profile t1-ref|desktop] [--soul PATH] [--persona PATH] [--json]`.
- [x] Add isolated tests for missing data, invalid persona, corrupt/incompatible log and clean JSON.
- [x] Use read-only APIs; never start the live runtime or warm up a model.
- [x] Errors fail; fresh default installations warn; external checks are not evaluated.

## Task 4: Verification and docs
**Files:** `README.md`, `docs/offline-lab.md`, this ledger.
- [x] Run formatting, targeted/workspace tests and strict Clippy, including safe all-feature checks.
- [x] Compare calibration with baseline and document usage/limitations.
- [x] Review changes, update ledger and leave changes uncommitted.

## Execution ledger
Baseline: clean master at afda4b6.
Ruling: work in place as requested; AGENTS.md prohibits unrequested branch changes.
Pre-flight: optional E1 observer; doctor uses read-only inspection; runtime fix does not change core policy.

### Implementation and observed verification

- E1: diagnostic CLI initially failed with unknown `--diagnose`; implemented through the existing executor. Five integration tests cover repeatability, masked inputs, invalid selectors, ordinary-result equivalence and missed turns. One unit test proves the 512-step cap reports omissions.
- Runtime: quiet-cancellation regression failed with the abandoned prefix still present, then passed after cleanup. Five scenarios exercise the actual runtime without allowing inference tasks to execute. Review constrained transcript preservation to an actual keyword timeout thought, not an unrelated discretionary thought.
- Doctor: four initial CLI tests failed before implementation. Seven isolated integration tests now cover fresh/missing/corrupt/incompatible inputs, profile mismatch, unchanged bytes/no sidecars and secret-free output.
- Default workspace test run: 481 passed, 2 intentionally ignored, zero failed. All 18 new tests also passed when run with all features.
- Default and all-feature strict workspace/all-target Clippy passed. All-feature workspace test binaries compiled (`--no-run`); the full all-feature suite was not executed, to avoid unrelated audio/model fixtures. The new offline subset was executed instead.
- Formatting and whitespace checks passed. `cargo machete` found no unused dependencies.
- Calibration 100..131 unchanged: 1776/3200 E1a requests, 3894/6400 turns; 5938 vs 12778 total paid calls (53.53% reduction); E1b 320/320 requests, 9 noise calls; zero synthetic self-ignitions. Original E1 criteria remain unmet and physical criteria remain unmeasured.
- `cargo deny --all-features check` failed on 15 advisory matches in the pre-existing lockfile; bans/licenses/sources passed. Recorded in `docs/dependency-audit-2026-10-27.md`. No dependency updates or advisory suppression were mixed into this work.
- No microphone, real inference, running-service changes, global tool installations, branch changes, commits or pushes. Cargo populated its normal cache for existing optional dependencies during compilation; no dependencies were added to the manifests/lockfile.
