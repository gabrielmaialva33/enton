# Offline laboratory and diagnostics

## Intent and scope
The owner approved implementation in the existing project on 2026-10-27: make Enton easier to investigate without microphone testing or real inference.

1. Opt-in E1 episode diagnosis through the existing closed-loop execution engine: full prehistory, unchanged scoring/budgets/sensor masking, actual decisions and unserved turns, bounded trace with explicit truncation.
2. Deterministic runtime scenarios for partial requests, discarded context, late completions, failure and quiet mode; no inference tasks allowed to execute.
3. Fix only a reproduced regression; do not tune thresholds or alter E1 populations.
4. Offline, read-only `enton doctor` with text/JSON and failure exit on errors: profile, persona and existing soul compatibility/integrity. Network, model quality, microphone and physical latency remain explicitly unmeasured.

## Approach and alternatives
Reuse E1's execution loop instead of a second simulator, which could disagree about feedback. Reuse the soul's read-only audit instead of opening its writer or copying the database. Controlled runtime events complement existing loopback integration tests. These implementation choices preserve the approved offline-first approach.

## Constraints and non-goals
Existing Rust workspace/dependencies and crate boundaries; English code/docs. No microphone, real inference/audio, installs, service changes, commits, branches or pushes. Work in place as requested. Validate against isolated fixtures, never the owner's persona or soul. Calibration seeds 100..131; no held-out seeds >=1000. No new framework, vision, memory service or dashboard.

## Acceptance
Observe failing tests before implementation. Diagnostic and ordinary E1 policy results match; repeated diagnosis is identical. Doctor creates no missing directories/database and performs no migration. Errors do not expose persona contents. Cancelled partial requests cannot leak into subsequent requests. Formatting, tests and strict Clippy pass for the configurations actually exercised. Simulation does not prove physical E1 criteria or LLM quality.
