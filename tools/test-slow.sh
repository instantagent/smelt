#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SMELT_RUN_SLOW_TESTS=1 swift test "$@"
