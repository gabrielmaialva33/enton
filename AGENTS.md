# enton

Enton is a digital organism that lives on this machine: it perceives, spends
energy thinking, and explains why it did not.

- **Purpose:** run a cheap, continuous digital life; make thought expensive
  and rare; keep the whole economy auditable.
- **Design source:** RFC 0001 *"Organismo"* (accepted). The document is not kept
  in this repository; the thesis, M1 scope and E1 criteria below carry its binding
  decisions, and section names such as P1, D1 or §7 in the code refer to it.

## Thesis (three lines)

1. Enton is a digital organism: its body is the hardware, its economy is
   auditable, and its learning is reversible.
2. Life is cheap and continuous; thought is expensive and rare.
3. The thesis only holds if it is measured — experiment E1 (RFC §7) exists to
   refute it.

## M1 scope

- P1: metabolic cognition — drives, ignition, budget (RFC §6).
- P4: soul = durable event log, replayable after restart.
- P5: audio perception by surprise — VAD + keyword before any STT.
- D1: interoception (body signals) + self-echo cancellation (PipeWire AEC).
- D2: per-profile price table, counterfactuals, offline evaluation.
- E1 must pass (RFC §7).

Out of scope for M1: vision, distillation, multiple bodies, WASM skills.

## Crate map and boundaries

| Crate                   | Responsibility                                                                                                                                                                                                                                                                                               |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `crates/enton-core`     | The pure cognitive core: `(state, event) → (state, actions)`. **No I/O, no clock, no randomness, fully deterministic** — time and coin flips enter as data. Drive tables, budgets, ignition, and the port traits for inference (`Cortex`, `SpeechToText`, `TextToSpeech`) live here with no implementations. |
| `crates/enton-adapters` | All I/O: body-signal readers (sysfs/procfs), monotonic clock, audio, log, cortex clients.                                                                                                                                                                                                                    |
| `crates/enton`          | The binary: composes core + adapters according to the hardware profile (`--profile t1-ref\|desktop`).                                                                                                                                                                                                        |

Cross a boundary only through the core's public API.

## Language

Everything in the repository — code, comments, doc comments, identifiers,
commits, and these docs — is in **English**.

## Quality bar

This bar is binding: high-level, idiomatic Rust. The workspace lints enforce
most of it (no `unwrap`/`expect`/`panic`/unchecked indexing in library code, no
swallowed `Result`s, libraries do not print, bounded everything); review covers
the rest. Never silence a lint without a comment explaining why the bar allows it.

**Workspace hygiene.** Never leave helper scripts, patch files or backups in the
repository (`patch_*.py`, `fix_*.py`, `*.bak`, `*.orig`). Edit files with your file
tools; if you truly need a throwaway script, put it under `/tmp` and delete it
when done. 56 such scripts had to be removed from the root on 2026-09-25.

## Agent roster (measured 2026-09-25)

- **Codex / Antigravity** — multi-file features, audio, async, anything with FFI.
- **laguna-xs-2.1** (local, free) — small bounded units (1–2 files with tests);
  its output always gets an external check: it has reported "done" with a red build.
- **qwen3.8:27b-gato** (local, free) — reviews (`@revisor`), docs, synthesis; not
  long compile-fix loops.
- **ornith-1.5:35b-gato** (local) — fast (~158 tok/s) but, under OpenCode, writes code as chat text
  instead of calling the file tools; failed the same bounded task twice. Not used as a coding agent.
- **Cline / Gemini 3.1 Pro** — paid per token (cost shown in its TUI); good at bounded module work. As a reviewer it is
  weak: review 0003 had 3 of 10 findings real (it cited code it had misread), so verify every claim before dispatching
  fixes.
- **Knowing when an agent finished (herdr):** `herdr agent prompt … --wait` / `agent wait` return on the
  first settled state, but agy reports `done` while still working and cline flips to `done` between
  steps. Treat an agent as finished only when herdr is not `working` **and** the pane footer shows no
  cancel hint (`esc to cancel` for agy/cline, `esc to interrupt` for codex) for ~20 s.
- Only **one local model at a time**, and never while Enton's cortex is being
  measured: the RTX 4090 holds one of them, and a collision wedged Ollama once.

## Commands

```sh
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt
cargo deny check
cargo machete
```

Every crate must pass all of them. Lints are workspace-level: clippy pedantic and
`unsafe_code = forbid`. CI (`.github/workflows/ci.yml`) runs them with default
features and with `--all-features`; `cpal` needs the ALSA headers
(`libasound2-dev`) for the audio features.

If the checkout moves, run `cargo clean`: build-script outputs and
`env!("CARGO_MANIFEST_DIR")` keep the old absolute path, and Cargo reuses them.

## Cross compilation and the T1-ref cage

- CI cross target: `aarch64-unknown-linux-musl`, built with `cargo zigbuild`
  (zig compiles `aws-lc-sys`), smoke-tested under `qemu-user` and held to the
  RFC §4 core budget.
- `armv5te-unknown-linux-musleabi` is a target **only for the camera** (T0),
  never for the mind.

## Build profiles

The core binary size is measured to ensure it fits the RFC 0001 §4 budget (< 20 MB):

| Profile            | Command                                                                | Executable | Shared libraries                                                |
|--------------------|------------------------------------------------------------------------|------------|-----------------------------------------------------------------|
| T1 (lean, default) | `cargo build --release -p enton`                                       | 4.7 MiB    | none                                                            |
| Desktop (static)   | `cargo build --release -p enton --features audio,voice`                | 33.0 MiB   | none (ONNX Runtime linked in)                                   |
| Desktop (shared)   | `cargo build --release -p enton --features audio,voice,shared-runtime` | 4.9 MiB    | `libsherpa-onnx-c-api.so` 4.9 MiB, `libonnxruntime.so` 25.8 MiB |

Measured 2026-09-25 (release, stripped). The default build is the lean T1 one; the
desktop needs the `audio,voice` features. Only the shared desktop build and the lean
build meet the RFC §4 core budget (< 20 MB with native runtimes outside the binary).

Note: When using the `shared-runtime` feature, the `libsherpa-onnx-c-api.so` and `libonnxruntime.so` shared libraries
must be accessible in the library path (e.g. `LD_LIBRARY_PATH` or a system directory).

The T1-ref reference environment is a cgroup cage on the Acer notebook — it
reproduces constraints, not the performance or power draw of a real ARM board:

```sh
systemd-run --user --scope -p MemoryMax=512M -p MemorySwapMax=0 -p CPUQuota=100% \
  target/release/enton --profile t1-ref
```

## E1 success criteria (RFC §7)

The implementation of the thesis is refuted if it fails any of these:

- ≥ 50% fewer cortex calls than the simple controller (VAD + keyword + fixed
  cooldown), using the same models, audio, and budget on the T1-ref profile;
- ≥ 99/100 relevant requests served in E1a, 10/10 in E1b;
- zero self-ignitions;
- zero cortex calls during the 50 min of noise in E1b;
- ≤ 100 ms additional p95 latency;
- stable core RSS (no sustained growth over 24 h).

## Git

The repository is under git: `origin` is `github.com/gabrielmaialva33/enton` and
`main` is the default branch. The history before the Rust rewrite is the Python
prototype (v0); RFC decision 3 ties it to a `v0-python` tag on its last commit.

- Read-only commands (`status`, `log`, `diff`, `show`) are always fine.
- Commit, branch, tag, push or rewrite history only when the owner asks.
- Commit messages are in English, like everything else in the repository.
