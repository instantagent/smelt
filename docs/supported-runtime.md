# Supported runtime

This is Smelt's public runtime contract. A model family is supported only when
it has a canonical module definition, source/build recipe, package output,
integrity checks, correctness coverage, and performance evidence. Loading a
checkpoint once is not sufficient.

## Canonical packages

| Package | Source model | Canonical output |
|---|---|---|
| Qwen 3.5 0.8B | `Qwen/Qwen3.5-0.8B` | `artifacts/qwen35-0.8b-qmm16x128/Qwen_Qwen3.5-0.8B.smeltpkg` |
| Qwen 3.5 2B | `Qwen/Qwen3.5-2B` | `artifacts/qwen35-2b-qmm16x128/Qwen_Qwen3.5-2B.smeltpkg` |
| Qwen3-TTS 0.6B | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | `qwen3-tts.smeltpkg` |

The checked-in authorities are:

- `Models/source-build-plan.tsv` for checkpoint inputs and source assembly;
- `Models/package-build-plan.tsv` for canonical package outputs and evidence;
- each module's authored performance profile for release bounds.

## Execution contract

`smelt run` directly executes the graph and run contract declared by one
`.smeltpkg`:

```bash
smelt run model.smeltpkg --prompt "hello"
cat input.txt | smelt run model.smeltpkg
```

`smelt serve` admits the same package into a resident HTTP or stdio transport.
The reusable `SmeltRuntime` and `SmeltServe` products expose those capabilities
without requiring a `smelt` executable in a consuming application.

Package identity, content-addressed installation, and best-effort shared
file-backed storage are Smelt runtime concerns. Model packages remain complete
and runnable without agent artifacts, Pi, or agent storage.

## Agent overlay

`smelt agent` is an optional policy leaf over the supported model runtime:

```bash
smelt agent create triage --model model.smeltpkg --system-file instructions.md
smelt agent run triage.agent "route this report"
smelt agent run triage.agent --interactive
```

A version-1 `.agent` contains only `agent.json`: its name, one Smelt package
identity, optional instructions, tool names, and the default `once` or
`interactive` mode. It contains no weights and no absolute model path. The
underlying package is resolved through `SmeltPackageStore`, so multiple agents
can reuse the same package pages in memory.

Interactive mode loads `integrations/pi-smelt-agent`. The Swift launcher passes
the exact current `smelt` executable path to the adapter, which uses the hidden
`smelt agent _serve-model` transport command. Generic serving remains
`smelt serve`; there is no public agent-serve command or separate `ia` binary.

## Verification

Every public checkout can run the hermetic product gate:

```bash
swift build -c release --product smelt
bash tools/test-default.sh -c release --parallel --num-workers 3 --quiet
```

Performance and correctness work belongs under `smelt lab`. A release claim
must use a canonical rebuilt package, verify its artifact identity and output
parity, and compare alternating baseline/candidate measurements on the
canonical machine. Generated packages, weights, metallibs, and local benchmark
artifacts remain untracked.
