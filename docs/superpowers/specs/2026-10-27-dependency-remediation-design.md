# Compatible dependency security updates

## Intent
Apply the compatible security updates approved by the owner on 2026-10-27, preserving the offline laboratory and avoiding microphone testing or changes to running services.

## Scope and constraints
- Work in the existing checkout, preserving all pre-existing modifications.
- Use Cargo to regenerate the lockfile; do not hand-edit dependency checksums.
- Prefer the smallest published patch that closes each advisory: rustls 0.23.46, tokio 1.53.3, rustls-webpki 0.103.16, webpki-root-certs 1.0.10, webpki-roots 1.0.10 and zstd-sys 2.1.2. Replace yanked yoke-derive 0.8.3 with 0.8.4.
- Preserve Rust 1.88 MSRV, the pinned 1.98.1 toolchain, crate features, reducer policy and all E1 criteria.
- No advisory ignores, graph pruning, dependency forks, forced incompatible transitive versions, global installations, microphone capture, real inference, playback, service changes, commits or pushes.
- Registry access and fetching ordinary build dependencies, including a verified immutable official upstream revision, are allowed; testing remains independent of live models.

## Approaches
A whole-workspace update would add unrelated changes and make failures harder to attribute. Updating sherpa-onnx to 2.x changes its native speech stack and is not equivalent to a lockfile patch. The approved approach is targeted compatible patches first, followed by inspection of the audio build dependency path. An incompatible audio migration remains a distinct deliverable rather than silently changing speech behavior.

Investigation found a third, compatible route: official upstream PR #4059, commit `5821f230f309c084d17d871c8d62ad3218a66ccc`, retains package version 1.13.8 and changes only its ZIP build requirement from 2 to 9. The `src` tree (`a0c5bc7187ea17acacc728f416158cd21f4e562a`) and `build.rs` blob (`34a3404824a01d026593a793490a94a687690e1d`) match tag v1.13.8 exactly. Use a revision-pinned Cargo patch to that official repository, not a fork or a forced incompatible override of ZIP itself. Explicitly allow that repository in the source policy; leave advisory ignores empty. Validate the newly resolved ZIP/compression chain, native test linking and MSRV before claiming closure. Document removal conditions for the temporary patch.

## Verification
Reproduce the advisory failure before changes. After targeted updates, run the default and all-feature audits and report their actual scope separately. Run the default workspace suite, strict Clippy for both feature sets, all-feature test compilation and the new offline tests under all features. Check the installed MSRV without installing toolchains, and compare E1 calibration seeds 100..131. Never run held-out seeds.

## Acceptance and limitations
The patched packages must no longer match their advisories; the lockfile diff must contain only justified resolutions. Existing offline functionality must retain passing tests. Any remaining all-feature findings must stay visible with exact dependency paths, upstream evidence and the reason they require more than a compatible patch. A passing audit is not proof of absence of all vulnerabilities; compilation is not a microphone or pronunciation test.
