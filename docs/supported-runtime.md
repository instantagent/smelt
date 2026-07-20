# Supported runtime

This is the release contract for Instant Agent. A model family is supported
only when it has a canonical source row, build recipe, output path, integrity
checks, output/prefill parity coverage, and performance gates. Loading once is
not sufficient.

## Canonical packages

| Package | Source model | Build | Canonical output |
|---|---|---|---|
| Qwen 3.5 0.8B | `Qwen/Qwen3.5-0.8B` | `bash tools/build-qwen35-0.8b.sh` | `artifacts/qwen35-0.8b-qmm16x128/Qwen_Qwen3.5-0.8B.agent` |
| Qwen 3.5 2B | `Qwen/Qwen3.5-2B` | `bash tools/build-qwen35-2b.sh` | `artifacts/qwen35-2b-qmm16x128/Qwen_Qwen3.5-2B.agent` |
| Qwen3-TTS 0.6B | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | `bash tools/build-module-package-plan.sh --id qwen3_tts` | `qwen3-tts.agent` |

The text packages use `affine_u4` weights with Metal prefill. The TTS package
uses grouped `u4` projections and carries its own reference parity, streamed vs
offline bit-exactness, and audio-baseline gates.

The checked-in source of truth is:

- `Models/source-build-plan.tsv` for source checkpoints and build commands;
- `Models/package-build-plan.tsv` for canonical outputs and build evidence;
- each module's authored performance profile for release bounds.

## Execution contract

`ia run` is the only public execution command.

```bash
ia run package.agent --once "one result"
ia run package.agent -i
cat input.txt | ia run package.agent
```

For text agents, the supported one-shot path preserves the package's prompt
template, sealed persona, sealed grammar, declared arguments, context limit,
and sampler selection. Interactive mode uses the same package contract through
the bundled Pi integration. A fixed first turn at temperature zero must agree
with the one-shot path at the generated-token and structured-output levels.

For text-to-PCM agents, `ia run` writes audio to the speakers on a terminal and
streams WAV on stdout when piped. Interactive mode is rejected because the
current terminal session surface is text-generation only.

## Run-mode contract

An optional version-1 `args.json` can declare:

```json
{
  "version": 1,
  "run": { "defaultMode": "auto" },
  "args": []
}
```

Valid modes are `interactive`, `once`, and `auto`. CLI flags override the
declaration. Explicit prompt input selects one-shot mode. `auto` enters a
conversation only when stdin and stdout are terminals. No migration or legacy
alias is part of the contract.

## Native session policy

The stateful runtime owns decoding-policy resolution and context pressure.
Callers supply an `AgentDecodingPolicyRequest`; they do not resolve the final
sampler themselves. Resolved policies and their signals are recorded on
`text_start` events and traces.

Supported request fields include policy name, phase, explicit/text/tool
temperatures, seed, latest user text, tool presence, and context pressure. Tool
call JSON remains argmax unless a positive tool temperature is explicitly
requested. Prompt overflow is rejected with structured context-pressure data;
an oversized output request is reduced to the available context room.

This lower-level contract is exercised independently of the Pi UI by the
stateful runtime and serve-handler Swift tests.

## Pi integration

`ia run -i` passes one resolved package to
`integrations/pi-instant-agent`. The provider has no fixed model registry: the
selected package path, stable identity, display name, and server port come from
the launcher. Pi owns terminal rendering and session persistence; Instant Agent
owns package semantics and generation.

`ia create <name>` uses the separate `author.ts` extension. Its tools are
confined to the draft source files and real `ia create`/`ia run --once` test
loops.

## Verification

Every public checkout can run the hermetic build and test gate:

```bash
swift build -c release --product ia
bash tools/test-default.sh
```

Built text packages can also be checked directly:

```bash
.build/release/ia verify path/to/model.agent
```

The release process additionally rebuilds the canonical text and voice
packages, exercises their parity and performance profiles, and records dated
[receipts](receipts/cold-start-methodology.md).

New package sets are not advertised until they carry equivalent build,
correctness, and performance evidence. Generated `.agent` directories,
weights, metallibs, and local benchmark artifacts remain untracked.
