# Dependency audit and remediation — 2026-10-27

## Current result

Both `cargo deny check` and `cargo deny --all-features check` now exit successfully: **advisories ok, bans ok, licenses ok, sources ok**. The original 15 advisory matches and the yanked-package warning no longer appear. This is the result for the resolved graph and advisory database checked on this date, not proof that every component is free of vulnerabilities.

The checks still print non-blocking duplicate-version warnings. The default-feature run also reports allowed sources/licenses that are not encountered because the optional audio build graph is absent. No warning levels were weakened to hide these findings.

The initial offline-laboratory delivery left dependencies unchanged. This subsequent, owner-approved remediation changes `Cargo.lock`, adds a narrowly pinned upstream patch in `Cargo.toml`, and records source/license permissions in `deny.toml`. Existing offline-laboratory edits remain intact and uncommitted.

## Before and after

Baseline: `master` at `afda4b6`, before dependency remediation. Stage 1 applied compatible patch releases, reducing all-feature advisory matches from 15 to 6 and making the default audit pass. Stage 2 adopted an official build-dependency fix, clearing the remaining six.

| Package | Before | Resolved after remediation | Original advisory IDs |
| --- | --- | --- | --- |
| rustls | 0.23.45 | 0.23.46 | RUSTSEC-2026-0346, 0347, 0348, 0349 |
| rustls-webpki | 0.103.15 | 0.103.16 | RUSTSEC-2026-0382 |
| tokio | 1.53.1 | 1.53.3 | RUSTSEC-2026-0386 |
| webpki-root-certs | 1.0.9 | 1.0.10 | RUSTSEC-2026-0383 |
| webpki-roots (1.x parent) | 1.0.9 | 1.0.10 | RUSTSEC-2026-0381 |
| zip | 2.4.2 | 9.1.2 | RUSTSEC-2026-0367, 0368, 0369 |
| zstd | 0.13.3 | 0.14.3 | RUSTSEC-2026-0391 |
| zstd-safe | 7.3.0 | 8.2.0 | RUSTSEC-2026-0398, 0403 |
| zstd-sys | 2.1.0+zstd.1.5.7 | 2.1.2+zstd.1.5.7 | RUSTSEC-2026-0410 |
| yoke-derive | 0.8.3 | 0.8.4 | Yanked version, not an advisory match |
| sherpa-onnx-sys | Registry 1.13.8 | Official Git revision, still 1.13.8 | Introduces the corrected ZIP build requirement |

The `webpki-roots` 0.26.11 compatibility wrapper remains, resolving to the corrected 1.0.10 parent. Cargo generated all lockfile checksums. Additional ZIP transitive changes are consequences of its new dependency requirements, not a whole-workspace update.

The advisory paths covered both runtime TLS dependencies through reqwest and build-only archive/compression dependencies through sherpa-onnx-sys. The initial package matches did not establish exploitability in Enton's actual configuration; no such claim is made here.

## Preserve the speech stack while fixing its build dependencies

No newer compatible registry release of sherpa-onnx 1.x was available during this check. Instead of migrating the speech stack to 2.x, the workspace uses the official fix from [upstream PR #4059](https://github.com/k2-fsa/sherpa-onnx/pull/4059), pinned to the complete commit:

`5821f230f309c084d17d871c8d62ad3218a66ccc`

The normal `sherpa-onnx` crate remains registry version 1.13.8. Only `sherpa-onnx-sys` comes from the pinned official repository. Its package version stays 1.13.8 and its manifest changes the ZIP build requirement from 2 to 9. The selected corrected ZIP chain remains compatible with Rust 1.88.

Verified against upstream tag v1.13.8:

- Identical sys source tree: `a0c5bc7187ea17acacc728f416158cd21f4e562a`.
- Identical build script blob: `34a3404824a01d026593a793490a94a687690e1d`.

The downloaded crates.io 1.13.8 package was also compared directly against the fetched Git revision: `cmp` of `build.rs` and recursive `diff` of `src` both succeeded. This checks the actual published files, not only the release tag. No local dependency fork, modified native bindings or forced override of an incompatible ZIP version was used.

The existing build script uses ZIP for iOS xcframework archives and tar/bzip2 for the Linux archive. Its code and native release selection are unchanged. Fixing the dependency graph is still worthwhile; being build-only is not treated as an exemption from advisories. Native archive provenance, microphone behavior and pronunciation were not independently audited in this task.

### Temporary patch maintenance

`deny.toml` permits only the official `https://github.com/k2-fsa/sherpa-onnx` Git repository for this patch. `Cargo.toml` and `Cargo.lock` pin the full revision; there is no moving branch reference. Clean builds need access to that revision or an existing Cargo cache.

Remove the patch and this source permission together when a compatible published 1.x fix can replace it, or during a separately validated speech-stack migration. Regenerate the lockfile and rerun both feature configurations, MSRV, offline tests and audits. Do not simply drop the patch back to the vulnerable registry graph.

## Reviewed license exception, not an advisory exception

ZIP's new build chain brings `libbz2-rs-sys 0.2.5`, declaring `bzip2-1.0.6`. Its packaged LICENSE was read in full and checked against the [SPDX license text](https://spdx.org/licenses/bzip2-1.0.6.html). The policy permits this license **only for `libbz2-rs-sys@0.2.5`**, not globally or for future releases.

Keep the required source notices, accurately identify origin and modified sources, and do not imply author endorsement. The reviewed file's SHA-256 is `54e1fd7bd53273e601c9599eb162a004dacddee031ae297ee7109a7beb93b1c2`. This record does not replace the dependency's license terms.

The advisory ignore list remains empty. No package was excluded from the audit graph, and no vulnerability severity was downgraded.

## Verification performed

| Check | Observed result |
| --- | --- |
| Default workspace tests, pinned Rust 1.98.1 | 481 passed, 2 intentionally ignored, 0 failed |
| Strict workspace/all-target Clippy, default and all features | Passed |
| Entire all-feature test suite, compile/link only (`--no-run`) | Passed |
| New offline tests executed with all features | 18 passed: runtime 5, doctor 7, episode diagnosis 5, trace bound 1 |
| Rust 1.88.0 workspace/all-target checks, default and all features | Passed |
| Persistence property tests actually executed with Rust 1.88.0 | 6 passed |
| Default and all-feature cargo-deny, all checks | Passed; known advisory matches 0 |
| Formatting and whitespace checks | Passed |
| cargo-machete | No unused dependencies found |

Rust 1.88 initially rejected an existing property-test expression with E0308/E0277: `json.pointer_mut(at.get(&pointers))`. Explicitly sampling `String` with `at.get::<String>(&pointers)` resolved the inference ambiguity. The failure was observed before the change, then both MSRV configurations and the property tests passed. This is a test-helper correction; no production behavior or version floor changed.

The full all-feature test suite was compiled, **not executed**. Only the audio-independent offline subset was executed with all features. No microphone capture, live model inference, audio playback or installed-service replacement was used. Cross-target builds and hosted CI were not run in this task.

### E1 calibration unchanged

The final run on calibration seeds 100 through 131 reproduced the earlier results:

- E1a: 1776/3200 complete requests and 3894/6400 turns served.
- E1b: 320/320 commands served; 9 total calls in noise intervals.
- Paid calls: organism 5938, simple controller 12778; reduction 53.53%.
- Synthetic self-ignitions: 0.

The core policy, populations, deadlines and success criteria were not altered. The outstanding E1 service-rate goals remain outstanding, and physical latency, microphone accuracy and 24-hour memory stability remain unmeasured. Held-out seeds were not used.

## Reproduction

From the repository root, use the lockfile's exact dependency resolutions rather than an unrestricted update:

```bash
cargo deny check
cargo deny --all-features check
CARGO_BUILD_JOBS=2 cargo test --locked --workspace -- --test-threads=2
CARGO_BUILD_JOBS=2 cargo clippy --locked --workspace --all-targets -- -D warnings
CARGO_BUILD_JOBS=2 cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
CARGO_BUILD_JOBS=2 cargo test --locked --workspace --all-features --no-run
CARGO_BUILD_JOBS=2 cargo +1.88.0 check --locked --workspace --all-targets
CARGO_BUILD_JOBS=2 cargo +1.88.0 check --locked --workspace --all-targets --all-features
cargo run --locked -q -p enton-e1 --bin e1-sim -- --seeds 100..=131 --summary
cargo fmt --all -- --check
cargo machete
```

Changes remain in the existing working tree, without a commit or push. No running service or installed executable was replaced.
