#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Skip heavy package-building integration tests (multi-GB synthetic
# weights.bin fixtures exhaust the CI runner's disk). They run in the
# manual `integration` workflow instead. See
# Tests/SmeltCompilerTests/Helpers/IntegrationGate.swift.
export SMELT_SKIP_INTEGRATION_TESTS=1
export SMELT_INCLUDE_MAINTAINER_TESTS=0

# Release-evidence and source-migration scanners are private maintainer
# machinery (public-release-plan amendment 9). Package.swift excludes their
# roughly 20K lines from the default test targets, so CI does not spend time
# compiling tests it will skip. tools/verify-release.sh opts them back in.
parallel_skips=(
  --skip Qwen3TTSCodecGPUTests
)

swift test "${parallel_skips[@]}" "$@"

# These Metal tests intermittently return empty output buffers when separate
# XCTest worker processes submit many shader-heavy command buffers at once.
# Keep their full coverage, but run the suite after the parallel CPU lane so
# the runner's GPU driver sees one test process at a time.
serial_args=()
skip_next=0
for arg in "$@"; do
  if (( skip_next )); then
    skip_next=0
    continue
  fi
  case "$arg" in
    --parallel|--no-parallel)
      ;;
    --num-workers)
      skip_next=1
      ;;
    --num-workers=*)
      ;;
    *)
      serial_args+=("$arg")
      ;;
  esac
done

swift test --filter Qwen3TTSCodecGPUTests "${serial_args[@]}" --no-parallel
