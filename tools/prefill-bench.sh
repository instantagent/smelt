#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  echo "Usage: bash tools/prefill-bench.sh <model.smeltpkg> [--iterations N] [TOKENS...]" >&2
  exit 1
}

[[ $# -ge 1 ]] || usage

pkg_path=$1
shift
iterations=10
token_counts=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iterations)
      [[ $# -ge 2 ]] || usage
      iterations=$2
      shift 2
      ;;
    *)
      token_counts+=("$1")
      shift
      ;;
  esac
done

if [[ ${#token_counts[@]} -eq 0 ]]; then
  token_counts=(64 128 256)
fi

if [[ -x .build/release/smelt ]]; then
  agent_cmd=(.build/release/smelt)
else
  agent_cmd=(swift run -c release smelt)
fi

for tokens in "${token_counts[@]}"; do
  output=$("${agent_cmd[@]}" prefill-bench "$pkg_path" --tokens "$tokens" --iterations "$iterations" 2>&1)
  wall_ms=$(printf '%s\n' "$output" | sed -n 's/^  Wall time:    \([0-9.]*\)ms\/prefill.*/\1/p' | tail -n 1)
  tok_s=$(printf '%s\n' "$output" | sed -n 's/^  Tokens\/sec:   \([0-9.]*\)$/\1/p' | tail -n 1)
  if [[ -z "$wall_ms" || -z "$tok_s" ]]; then
    printf '%s\n' "$output" >&2
    echo "Failed to parse prefill-bench output for $tokens tokens" >&2
    exit 1
  fi
  printf '%4s tokens  %8sms/prefill  %8stok/s\n' "$tokens" "$wall_ms" "$tok_s"
done
