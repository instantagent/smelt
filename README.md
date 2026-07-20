# Smelt

Smelt is an Apple Silicon model toolkit for configuring, compiling, bringing
up, running, serving, inspecting, and optimizing local models. It owns the
package ABI, compiler, runtime, Metal kernels, model definitions, rigging
pipeline, and correctness/performance harnesses.

```console
swift build -c release
.build/release/smelt build Models/qwen35_text.module.json --output artifacts
.build/release/smelt run artifacts/model.smeltpkg --prompt "hello"
.build/release/smelt serve artifacts/model.smeltpkg --transport http
.build/release/smelt verify artifacts/model.smeltpkg
```

Compiled models are `.smeltpkg` directories. A package declares its block
graph, run contract, tokenizer or modality adapters, runtime policy, kernels,
weights, and integrity data. `smelt run` executes one package-authored request;
`smelt serve` keeps the same package-faithful runtime resident.

Smelt is independently useful and has no dependency on an agent product.
[Instant Agent](https://github.com/instantagent/instantagent) is a separate downstream repository that uses
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
- `smelt`, `smelt-probe`, and `smelt-models`: toolkit and bring-up executables.

The mesh-rigging vertical slice is a Smelt model workflow. Its compiler,
runtime, shaders, TokenRig package, PyTorch reference tooling, U0 verifier, and
parity fixtures stay here and execute through the generic package run contract.

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
