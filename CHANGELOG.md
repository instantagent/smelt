# Changelog

## 0.3.1 — 2026-07-21

- Bundle the Pi integration as a SmeltCLI resource and require the exact
  launcher-supplied `smelt` executable instead of searching for an installed
  command.
- Enforce the manifest-only `.agent` artifact contract when loading,
  installing, and publishing overlays.
- Restore the curated external-consumer release gate and remove the remaining
  live Instant Agent identity from project policy and guidance.

## 0.3.0 — 2026-07-21

- Integrate the former Instant Agent policy layer as the contained internal
  `SmeltAgent` target and expose it only through `smelt agent`.
- Preserve thin version-1 `.agent` overlays and portable registry layout while
  moving agent storage and environment names into the Smelt namespace.
- Integrate Pi authoring and interactive execution with the current `smelt`
  executable; no separate agent command is required or provided.
- Add containment and external-consumer gates proving that lower Smelt layers
  remain agent-independent and that the agent core still builds solely against
  the public runtime and serving libraries.

## 0.2.0 — 2026-07-20

This release established Smelt as a lower-level model compiler, package,
runtime, serving, and optimization toolkit. Its former standalone Instant
Agent consumer was integrated into Smelt in 0.3.0.

- Consolidate inspection, profiling, benchmarking, kernel sweeps, and
  correctness checks under the discoverable `smelt lab` command. The former
  `smelt-probe` executable and retired top-level aliases are removed.
- Replace the former bake product contract with package-native optional
  prepared artifacts for reusable prompt-prefix and grammar state.
- Expose descriptor-based admission, runtime inventory, and serving through
  the `SmeltRuntime` and `SmeltServe` libraries so consumers do not need a
  separately installed `smelt` executable.
- Deduplicate installed packages through the content-addressed
  `SmeltPackageStore` on a best-effort basis while retaining a portable
  materialization boundary for registries and archives.
- Run SkinTokens articulation through the generic component-package and block
  graph runtime, including its compiler, kernels, reference verifier, and
  deterministic rigging path.
- Speed up exact dense contractions, tiled attention, and Qwen 3.5 0.8B B8
  prefill while retaining package-faithful correctness gates and receipts.
- Add the Smelt overview, public visual identity, and standalone Metal research
  spikes for mmap-backed buffers, paged attention, speculative decoding, and
  Whisper transcription.

## 0.1.1 — 2026-07-19

- Remove unsafe Clang-importer flags from the public Swift package products so
  downstream consumers can build Smelt without inheriting unsafe settings.
- Use the expected 32-bit LAPACK integer representation in the GPTQ
  path, preserving source-package compatibility without those unsafe flags.

## 0.1.0 — 2026-07-14

The first public Smelt source release.

- Build, inspect, run, and serve sealed `.smeltpkg` model packages on Apple
  Silicon from the `smelt` command-line tool.
- Provide reusable `SmeltSchema`, `SmeltCompiler`, `SmeltRuntime`,
  `SmeltServe`, `SmeltModuleAuthoring`, and `SmeltModels` Swift products.
- Compile and run canonical Qwen 3.5 0.8B and 2B text packages and the Qwen3-TTS
  0.6B CustomVoice package with native Metal kernels.
- Restore optional prepared prefix state and compiled grammar vocabulary rather
  than rebuilding fixed work for each request.
- Store packages and shared model blobs by content identity.
- Ship public cold-start, throughput, and structured-quality receipts.
