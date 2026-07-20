#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -ne 0 ]]; then
  echo "This build entrypoint is module package-plan driven and takes no model-specific build arguments." >&2
  exit 1
fi

exec bash tools/build-module-package-plan.sh --id qwen35_text
