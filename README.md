<p align="center"><img src="docs/assets/smelt-mark.png" width="96" alt="Smelt"></p>

# Smelt

**The model compiler and native runtime for Apple Silicon.** Smelt turns
checkpoints — chat models, voices, whole multi-stage systems — into sealed
programs, verified bit-exact against the original model.

Smelt compiles a whole model system — checkpoints, block graph, tokenizer or
modality adapters, kernels, runtime policy, and integrity data — into one
executable `.smeltpkg`, and gives it one toolchain for bring-up, running,
serving, inspecting, and optimizing. It owns the package ABI, compiler,
runtime, Metal kernels, model definitions, rigging pipeline, and
correctness/performance harnesses. One surface drives every modality:
`smelt run` executes the graph the package declares, whether that graph is a
chat model, a streaming speech pipeline, or a mesh auto-rigger.

## What it runs today

- **Text** — Qwen 3.5 (0.8B/2B/4B, hybrid linear attention) and Gemma 4
  (E2B/E4B): ~300 tok/s decode on the 0.8B and 144.7 tok/s on E2B at the
  production build config; grammar-constrained JSON decodes at 97.5% of
  unconstrained speed. Binary and ternary Bonsai 27B run native low-bit all
  the way to Metal, with binary decode at 47.9 tok/s median. Figures are from
  the canonical M2 Max bench machine.
- **Speech** — Qwen3-TTS (0.6B/1.7B talker + streaming codec): 83 ms warm
  time-to-first-audio on the 0.6B, and chunked streaming decode is
  memcmp-identical to offline decode.
- **Articulation** — [SkinTokens](https://github.com/VAST-AI-Research/SkinTokens),
  VAST-AI Research's open auto-rigger (Michelangelo shape encoder, skinning
  VAE, and a Qwen-trunk autoregressive generator — 595M parameters, one BF16
  checkpoint): a GLB mesh in, a skinned, re-importable GLB out, through nine
  package-declared stages. The deterministic reference rig is byte-identical,
  SHA-pinned.

```console
swift build -c release
.build/release/smelt build Models/qwen35_text.module.json --output artifacts
.build/release/smelt run artifacts/model.smeltpkg --prompt "hello"
.build/release/smelt serve artifacts/model.smeltpkg --transport http
.build/release/smelt lab verify artifacts/model.smeltpkg
```

Compiled models are `.smeltpkg` directories. A package declares its block
graph, run contract, tokenizer or modality adapters, runtime policy, kernels,
weights, and integrity data. `smelt run` executes one package-authored request;
`smelt serve` keeps the same package-faithful runtime resident.

Products ship on top of Smelt.
[Instant Agent](https://github.com/instantagent/instantagent) is the first: a downstream repository built on
the public `SmeltRuntime` and `SmeltServe` products. Its `.agent` artifact is a
thin semantic overlay referencing a Smelt package by content identity, so
multiple agents can share the same stored package and file-backed weight pages.
`SmeltPackageStore.materialize(identity:at:)` is the portability boundary for
registries and archives: it resolves local CAS links into ordinary package
files while retaining clone-on-write where possible. Smelt HTTP servers expose
the loaded identity as `X-Smelt-Package-Identity` on `/v1/models`, allowing a
consumer to reject accidental routing to the wrong resident package.

## Products

- `SmeltSchema`: package ABI, block graph, run contracts, typed ports/options.
- `SmeltCompiler`: checkpoint adaptation, quantization, lowering, and assembly.
- `SmeltRuntime`: package loading, sessions, prepared state, inference, and
  generic file-transform dispatch.
- `SmeltServe`: reusable model admission, OpenAI-compatible serving, and HTTP or
  stdio transports. Consumers do not need the `smelt` executable.
- `SmeltModuleAuthoring`: Swift model/module authoring helpers.
- `SmeltModels`: maintained model definitions.
- `smelt` and `smelt-models`: toolkit and model-definition executables. Performance,
  correctness, profiling, and bring-up tools live under `smelt lab`.

The mesh-skinning vertical slice is a Smelt model workflow. Its compiler,
runtime, shaders, SkinTokens component package, PyTorch reference tooling, U0
verifier, and parity fixtures stay here and execute through CAM-authored
`smelt build` and the generic package run contract.

## Validation

GitHub runs only the seconds-level syntax/data lint lane:

```console
bash tools/ci-lint.sh
```

Before a code PR is opened or merged, run the local correctness gate:

```console
bash tools/test-default.sh -c release --parallel --num-workers 3 --quiet
```

Run the slow, model, rig, performance, maintainer, and release gates locally
when the changed surface requires them. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Lab workflow

All correctness, benchmarking, profiling, inspection, and kernel experiments
live under one discoverable namespace:

```console
.build/release/smelt lab help
.build/release/smelt lab inspect dispatches model.smeltpkg --table prefill
.build/release/smelt lab profile prefill model.smeltpkg --tokens 8 --iterations 20
.build/release/smelt lab sweep qmm model.smeltpkg --batches 7,8,9
.build/release/smelt lab verify candidate.smeltpkg --baseline baseline.smeltpkg
```

Treat a kernel sweep as discovery, not a production performance claim. Rebuild
the canonical package, verify its artifact identity and output parity, then run
alternating baseline/candidate package benchmarks at the selected route and its
neighboring boundaries.
