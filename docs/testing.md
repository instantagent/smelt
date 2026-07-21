# Test Gate Matrix

Smelt has one hosted lint lane and several local test tiers. The split keeps
GitHub usage negligible while preserving the full correctness gate before
merge.

## Tiers

| Tier | Runner | Trigger | What it covers |
|---|---|---|---|
| Hosted lint | `bash tools/ci-lint.sh` | every non-doc PR + push to `main` (`.github/workflows/default.yml`) | Shell and Python syntax plus checked-in JSON parsing. No Swift build, cache, model setup, or Metal runner. |
| Default | `bash tools/test-default.sh -c release --parallel --num-workers 3 --quiet` | locally before every code PR and merge | Hermetic product suite — schema, dispatch table, tokenizer behavior, model adapters, runtime sessions, optimizer rewrites, Metal kernel parity. No external assets. |
| Slow | `bash tools/test-slow.sh` | manual, run locally | Quantizer perf and quality sweeps, extended prefill kernel coverage. Sets `SMELT_RUN_SLOW_TESTS=1`. Hermetic — no external assets, just slow. |
| Maintainer | `SMELT_INCLUDE_MAINTAINER_TESTS=1 swift test` | release verification and module migration work | Private release-evidence harnesses, completion-matrix validation, selector-deletion scans, and CAM command/source migration lints. |

## Default suite

`tools/test-default.sh` runs the hermetic Swift suite in several minutes while
excluding multi-GB package-build fixtures. Package.swift also excludes about
20K lines of maintainer-only release evidence and source-migration tests from
the default test targets, avoiding their compile cost rather than merely
skipping them after compilation. Anything that needs network access, a
downloaded checkpoint, or a built canonical package is gated behind one of the
opt-in environment variables below.

## Opt-in env vars

Each variable is set to `1` to enable. They compose — set multiple to run
multiple tiers in one invocation.

| Variable | What it enables | Required asset |
|---|---|---|
| `SMELT_RUN_SLOW_TESTS` | Quantizer perf + quality sweeps; large-batch prefill kernel coverage. | None — slow but hermetic. |
| `SMELT_INCLUDE_MAINTAINER_TESTS` | Private release-evidence, completion-matrix, selector-deletion, and source-migration suites. | None; `tools/verify-release.sh` enables this automatically. |
| `SMELT_RUN_TEXT_SMOKE` | Qwen 3.5 end-to-end smoke (prefill ↔ decode parity, prefix caching, GPU temperature sampling) and extended large-batch prefill coverage. | Built canonical Qwen packages. |
| `SMELT_RUN_EXPERIMENTAL_QMM_FFN_TESTS` | Experimental QMM-on-FFN tests. Subject to change. | None. |

Pass-through variables for the smoke tests — useful when canonical packages
live outside the repo:

| Variable | Effect |
|---|---|
| `SMELT_TEXT_PACKAGE` | Override path to the Qwen 2B canonical package. |
| `SMELT_QWEN_0_8B_PACKAGE` | Override path to the Qwen 0.8B canonical package. |
| `SMELT_QWEN_4B_PACKAGE` | Override path to the Qwen 4B package. |

## CI and pre-merge policy

GitHub runs only `bash tools/ci-lint.sh` on Linux, with a two-minute runaway
limit and no run for Markdown-only changes. The script has no downloaded
dependencies and normally completes in seconds. Superseded runs on the same
ref are canceled.

The standing correctness gate is local: before opening or merging a code pull
request, run `bash tools/test-default.sh -c release --parallel --num-workers 3
--quiet` and record the result in the pull request. The Qwen3 TTS Metal suite
runs afterward in one XCTest process because concurrent GPU-heavy workers can
return empty command-buffer outputs. Slow, integration, maintainer, and release
verification remain local-only. Adding a hosted macOS or full Swift-test lane
requires explicit repository-owner approval.
