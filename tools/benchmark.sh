#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  echo "Usage: bash tools/benchmark.sh <model.smeltpkg> [--decode-iterations N] [--decode-warmup N] [--decode-fixed-position N] [--decode-positions 0,32,64,...] [--decode-trace-start N --decode-trace-length N] [--prefill-iterations N] [--prefill-warmup N] [--prefill-tokens 64,256] [--benchmark-settle-timeout N] [--benchmark-settle-interval N] [--min-decode-tps N] [--max-decode-p95-ms N] [--min-prefill-tps 64=...,256=...] [--max-prefill-p95-ms 64=...,256=...] [--output-json FILE] [--use-package-profile] [--defer-startup-profile]" >&2
  exit 1
}

[[ $# -ge 1 ]] || usage

pkg_path=$1
shift
decode_iterations=20
decode_warmup=5
decode_fixed_position=
decode_positions=
decode_trace_start=
decode_trace_length=
prefill_iterations=5
prefill_warmup=2
prefill_tokens=64,256
benchmark_settle_timeout=
benchmark_settle_interval=
min_decode_tps=
max_decode_p95_ms=
min_prefill_tps=
max_prefill_p95_ms=
use_package_profile=0
defer_startup_profile=0
output_json=
profile_gate=
profile_required_output_metrics=()
profile_min_bound_lines=()
profile_max_bound_lines=()
profile_deferred_bound_lines=()

lookup_threshold() {
  local spec=$1
  local key=$2
  local entry lhs rhs
  [[ -n "$spec" ]] || return 1
  IFS=',' read -r -a entries <<< "$spec"
  for entry in "${entries[@]}"; do
    lhs=${entry%%=*}
    rhs=${entry#*=}
    lhs=${lhs//[[:space:]]/}
    rhs=${rhs//[[:space:]]/}
    if [[ -n "$lhs" && -n "$rhs" && "$lhs" == "$key" ]]; then
      printf '%s\n' "$rhs"
      return 0
    fi
  done
  return 1
}

float_ge() {
  awk -v a="$1" -v b="$2" 'BEGIN { exit((a + 0 >= b + 0) ? 0 : 1) }'
}

float_le() {
  awk -v a="$1" -v b="$2" 'BEGIN { exit((a + 0 <= b + 0) ? 0 : 1) }'
}

set_threshold_if_missing() {
  local spec=$1
  local key=$2
  local value=$3
  if lookup_threshold "$spec" "$key" >/dev/null 2>&1; then
    printf '%s\n' "$spec"
  elif [[ -n "$spec" ]]; then
    printf '%s,%s=%s\n' "$spec" "$key" "$value"
  else
    printf '%s=%s\n' "$key" "$value"
  fi
}

metric_seen() {
  local needle=$1
  shift
  local metric
  for metric in "$@"; do
    [[ "$metric" == "$needle" ]] && return 0
  done
  return 1
}

profile_prefill_tokens() {
  local current=$1
  shift
  python3 - "$current" "$@" <<'PY'
import re
import sys

tokens = []
seen = set()

def add(value):
    value = str(value).strip()
    if value and value not in seen:
        seen.add(value)
        tokens.append(value)

for part in sys.argv[1].split(","):
    add(part)

pattern = re.compile(r"^prefill([0-9]+)_(?:wall_ms|tokens_per_second|p95_ms)$")
for metric in sys.argv[2:]:
    match = pattern.match(metric)
    if match:
        add(match.group(1))

print(",".join(tokens))
PY
}

read_package_profile() {
  local manifest_path=$1
  local canonical_profile
  local model_name
  model_name=$(python3 - "$manifest_path" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    manifest = json.load(handle)
camel = manifest.get("modelName")
snake = manifest.get("model_name")
if camel is not None and (not isinstance(camel, str) or not camel):
    raise SystemExit("text package manifest has invalid modelName")
if snake is not None and (not isinstance(snake, str) or not snake):
    raise SystemExit("text package manifest has invalid model_name")
if camel is not None and snake is not None and camel != snake:
    raise SystemExit("text package manifest has conflicting modelName/model_name")
print(camel if camel is not None else snake or "")
PY
  )
  local profile_args=(lab package-profile text.decode-prefill-startup)
  if [[ -n "$model_name" ]]; then
    profile_args+=(--model-name "$model_name")
  fi
  if [[ -x .build/release/smelt ]]; then
    canonical_profile=$(.build/release/smelt "${profile_args[@]}")
  else
    canonical_profile=$(swift run -c release smelt "${profile_args[@]}")
  fi
  SMELT_CANONICAL_PROFILE="$canonical_profile" python3 - "$manifest_path" <<'PY'
import json
import math
import os
import sys

manifest_path = sys.argv[1]

def load_canonical_profile(raw):
    profile = {
        "required_trace_labels": [],
        "required_output_metrics": [],
        "max_bounds": [],
    }
    for line in raw.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        kind = parts[0]
        if kind in ("gate", "command"):
            if len(parts) != 2:
                print(f"invalid canonical profile {kind} row", file=sys.stderr)
                sys.exit(1)
            profile[kind] = parts[1]
        elif kind == "required_trace_label":
            if len(parts) != 2:
                print("invalid canonical profile required_trace_label row", file=sys.stderr)
                sys.exit(1)
            profile["required_trace_labels"].append(parts[1])
        elif kind == "required_output_metric":
            if len(parts) != 2:
                print("invalid canonical profile required_output_metric row", file=sys.stderr)
                sys.exit(1)
            profile["required_output_metrics"].append(parts[1])
        elif kind == "max_bound":
            if len(parts) != 4:
                print("invalid canonical profile max_bound row", file=sys.stderr)
                sys.exit(1)
            try:
                maximum = float(parts[2])
            except ValueError:
                print("invalid canonical profile max_bound value", file=sys.stderr)
                sys.exit(1)
            profile["max_bounds"].append((parts[1], maximum, parts[3]))
    for key in ("gate", "command"):
        if key not in profile:
            print(f"canonical profile missing {key}", file=sys.stderr)
            sys.exit(1)
    return profile

canonical = load_canonical_profile(os.environ.get("SMELT_CANONICAL_PROFILE", ""))
gate_id = canonical["gate"]

try:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
except Exception as error:
    print(f"failed to read text package manifest: {error}", file=sys.stderr)
    sys.exit(1)

validation = manifest.get("validation")
if not isinstance(validation, dict):
    print("text package manifest has no validation block", file=sys.stderr)
    sys.exit(1)

declared_gate = validation.get("performance_gate")
if declared_gate not in (None, gate_id):
    print(
        f"text package validation performance_gate is '{declared_gate}', expected '{gate_id}'",
        file=sys.stderr,
    )
    sys.exit(1)

profile = validation.get("performance_profile")
if not isinstance(profile, dict):
    print("text package validation has no performance_profile", file=sys.stderr)
    sys.exit(1)
if profile.get("gate") != gate_id:
    print(
        f"text package performance_profile gate is '{profile.get('gate')}', expected '{gate_id}'",
        file=sys.stderr,
    )
    sys.exit(1)
if profile.get("command") != canonical["command"]:
    print(
        f"text package performance_profile command is '{profile.get('command')}', "
        f"expected '{canonical['command']}'",
        file=sys.stderr,
    )
    sys.exit(1)

def require_strings(values, required, label):
    if not isinstance(values, list):
        print(f"text package performance_profile {label}s must be a list", file=sys.stderr)
        sys.exit(1)
    for value in values:
        if not isinstance(value, str) or not value:
            print(f"text package performance_profile has an invalid {label}", file=sys.stderr)
            sys.exit(1)
    value_set = set(values)
    for value in required:
        if value not in value_set:
            print(
                f"text package performance_profile missing canonical {label}: {value}",
                file=sys.stderr,
            )
            sys.exit(1)

require_strings(
    profile.get("required_trace_labels"),
    canonical["required_trace_labels"],
    "required trace label",
)
require_strings(
    profile.get("required_output_metrics"),
    canonical["required_output_metrics"],
    "required output metric",
)

print(f"gate\t{gate_id}")
for metric in profile.get("required_output_metrics", []):
    if not isinstance(metric, str) or not metric:
        print("text package performance_profile has an invalid required output metric", file=sys.stderr)
        sys.exit(1)
    print(f"metric\t{metric}")

for bound in profile.get("min_bounds", []):
    metric = bound.get("metric")
    min_value = bound.get("min")
    unit = bound.get("unit")
    if (
        not isinstance(metric, str)
        or not metric
        or not isinstance(min_value, (int, float))
        or not math.isfinite(min_value)
        or min_value <= 0
        or not isinstance(unit, str)
        or not unit
    ):
        print("text package performance_profile has an invalid min-bound", file=sys.stderr)
        sys.exit(1)
    print(f"min_bound\t{metric}\t{min_value:g}\t{unit}")

for bound in profile.get("max_bounds", []):
    metric = bound.get("metric")
    max_value = bound.get("max")
    unit = bound.get("unit")
    if (
        not isinstance(metric, str)
        or not metric
        or not isinstance(max_value, (int, float))
        or not math.isfinite(max_value)
        or max_value <= 0
        or not isinstance(unit, str)
        or not unit
    ):
        print("text package performance_profile has an invalid max-bound", file=sys.stderr)
        sys.exit(1)
    print(f"max_bound\t{metric}\t{max_value:g}\t{unit}")

for metric, maximum, unit in canonical["max_bounds"]:
    found = False
    for bound in profile.get("max_bounds", []):
        if (
            isinstance(bound, dict)
            and bound.get("metric") == metric
            and bound.get("unit") == unit
            and isinstance(bound.get("max"), (int, float))
            and math.isfinite(bound.get("max"))
            and bound.get("max") <= maximum
        ):
            found = True
            break
    if not found:
        print(
            f"text package performance_profile missing canonical max-bound: {metric}",
            file=sys.stderr,
        )
        sys.exit(1)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --decode-iterations)
      [[ $# -ge 2 ]] || usage
      decode_iterations=$2
      shift 2
      ;;
    --decode-warmup)
      [[ $# -ge 2 ]] || usage
      decode_warmup=$2
      shift 2
      ;;
    --decode-fixed-position)
      [[ $# -ge 2 ]] || usage
      decode_fixed_position=$2
      shift 2
      ;;
    --decode-positions)
      [[ $# -ge 2 ]] || usage
      decode_positions=$2
      shift 2
      ;;
    --decode-trace-start)
      [[ $# -ge 2 ]] || usage
      decode_trace_start=$2
      shift 2
      ;;
    --decode-trace-length)
      [[ $# -ge 2 ]] || usage
      decode_trace_length=$2
      shift 2
      ;;
    --prefill-iterations)
      [[ $# -ge 2 ]] || usage
      prefill_iterations=$2
      shift 2
      ;;
    --prefill-warmup)
      [[ $# -ge 2 ]] || usage
      prefill_warmup=$2
      shift 2
      ;;
    --prefill-tokens)
      [[ $# -ge 2 ]] || usage
      prefill_tokens=$2
      shift 2
      ;;
    --benchmark-settle-timeout)
      [[ $# -ge 2 ]] || usage
      benchmark_settle_timeout=$2
      shift 2
      ;;
    --benchmark-settle-interval)
      [[ $# -ge 2 ]] || usage
      benchmark_settle_interval=$2
      shift 2
      ;;
    --min-decode-tps)
      [[ $# -ge 2 ]] || usage
      min_decode_tps=$2
      shift 2
      ;;
    --max-decode-p95-ms)
      [[ $# -ge 2 ]] || usage
      max_decode_p95_ms=$2
      shift 2
      ;;
    --min-prefill-tps)
      [[ $# -ge 2 ]] || usage
      min_prefill_tps=$2
      shift 2
      ;;
    --max-prefill-p95-ms)
      [[ $# -ge 2 ]] || usage
      max_prefill_p95_ms=$2
      shift 2
      ;;
    --output-json)
      [[ $# -ge 2 ]] || usage
      output_json=$2
      shift 2
      ;;
    --use-package-profile)
      use_package_profile=1
      shift
      ;;
    --defer-startup-profile)
      defer_startup_profile=1
      shift
      ;;
    --gate-*)
      echo "tools/benchmark.sh: model gate flags were removed; use --use-package-profile or explicit threshold flags" >&2
      exit 1
      ;;
    *)
      usage
      ;;
  esac
done

if [[ ! -f "$pkg_path/manifest.json" ]]; then
  echo "Missing package at $pkg_path" >&2
  exit 1
fi

if [[ "$use_package_profile" != 0 ]]; then
  profile_dump=$(read_package_profile "$pkg_path/manifest.json")
  while IFS=$'\t' read -r kind metric bound_value unit; do
    case "$kind" in
      gate)
        profile_gate=$metric
        ;;
      metric)
        profile_required_output_metrics+=("$metric")
        ;;
      min_bound)
        if [[ "$unit" != "tok/s" ]]; then
          echo "text package profile min-bound '$metric' uses unsupported unit '$unit'." >&2
          exit 1
        fi
        case "$metric" in
          decode_tokens_per_second)
            [[ -n "$min_decode_tps" ]] || min_decode_tps=$bound_value
            ;;
          prefill*_tokens_per_second)
            if [[ "$metric" =~ ^prefill([0-9]+)_tokens_per_second$ ]]; then
              min_prefill_tps=$(set_threshold_if_missing \
                "$min_prefill_tps" "${BASH_REMATCH[1]}" "$bound_value")
            else
              echo "text package profile min-bound '$metric' is not enforced by this harness." >&2
              exit 1
            fi
            ;;
          *)
            echo "text package profile min-bound '$metric' is not enforced by this harness." >&2
            exit 1
            ;;
        esac
        profile_min_bound_lines+=("$metric $bound_value$unit")
        ;;
      max_bound)
        if [[ "$unit" != "ms" ]]; then
          echo "text package profile max-bound '$metric' uses unsupported unit '$unit'." >&2
          exit 1
        fi
        case "$metric" in
          decode_p95_ms_per_token)
            [[ -n "$max_decode_p95_ms" ]] || max_decode_p95_ms=$bound_value
            profile_max_bound_lines+=("$metric $bound_value$unit")
            ;;
          prefill*_p95_ms)
            if [[ "$metric" =~ ^prefill([0-9]+)_p95_ms$ ]]; then
              max_prefill_p95_ms=$(set_threshold_if_missing \
                "$max_prefill_p95_ms" "${BASH_REMATCH[1]}" "$bound_value")
              profile_max_bound_lines+=("$metric $bound_value$unit")
            else
              echo "text package profile max-bound '$metric' is not enforced by this harness." >&2
              exit 1
            fi
            ;;
          trace_first_token_ms)
            profile_deferred_bound_lines+=("$metric $bound_value$unit")
            ;;
          *)
            echo "text package profile max-bound '$metric' is not enforced by this harness." >&2
            exit 1
            ;;
        esac
        ;;
    esac
  done <<<"$profile_dump"
  startup_profile_deferred=0
  if [[ ${#profile_required_output_metrics[@]} -gt 0 ]] &&
    metric_seen trace_first_token_ms "${profile_required_output_metrics[@]}"
  then
    startup_profile_deferred=1
  fi
  if [[ ${#profile_deferred_bound_lines[@]} -gt 0 ]]; then
    startup_profile_deferred=1
  fi
  if [[ $startup_profile_deferred -eq 1 && $defer_startup_profile -eq 0 ]]; then
    echo "text package profile includes startup metric trace_first_token_ms; run tools/benchmark-text-startup.sh --use-package-profile first, then pass --defer-startup-profile for this decode/prefill-only harness." >&2
    exit 1
  fi
  prefill_tokens=$(profile_prefill_tokens "$prefill_tokens" "${profile_required_output_metrics[@]}")
fi

if [[ -x .build/release/smelt ]]; then
  agent_cmd=(.build/release/smelt)
else
  agent_cmd=(swift run -c release smelt)
fi

bench_args=(lab bench decode "$pkg_path" --iterations "$decode_iterations" --warmup "$decode_warmup")
if [[ -n "$decode_fixed_position" ]]; then
  bench_args+=(--fixed-position "$decode_fixed_position")
fi
if [[ -n "$decode_positions" ]]; then
  bench_args+=(--positions "$decode_positions")
fi
if [[ -n "$decode_trace_start" || -n "$decode_trace_length" ]]; then
  [[ -n "$decode_trace_start" && -n "$decode_trace_length" ]] || usage
  bench_args+=(--trace-start "$decode_trace_start" --trace-length "$decode_trace_length")
fi

if [[ "$use_package_profile" != 0 ]]; then
  echo "text-profile-gate $profile_gate"
  for metric in "${profile_required_output_metrics[@]}"; do
    echo "text-profile-required-output-metric $metric"
  done
  if [[ ${#profile_min_bound_lines[@]} -gt 0 ]]; then
    for line in "${profile_min_bound_lines[@]}"; do
      echo "text-profile-min-bound $line"
    done
  fi
  if [[ ${#profile_max_bound_lines[@]} -gt 0 ]]; then
    for line in "${profile_max_bound_lines[@]}"; do
      echo "text-profile-max-bound $line"
    done
  fi
  if [[ ${#profile_deferred_bound_lines[@]} -gt 0 ]]; then
    for line in "${profile_deferred_bound_lines[@]}"; do
      echo "text-profile-deferred-bound $line"
    done
  fi
fi

bench_out=$("${agent_cmd[@]}" "${bench_args[@]}" 2>&1)
if [[ -n "$decode_positions" || -n "$decode_trace_start" ]]; then
  printf '%s\n' "$bench_out"
  exit 0
fi

position=$(printf '%s\n' "$bench_out" | sed -E -n 's/^  Position:[[:space:]]+(.+)$/\1/p' | tail -n 1)
position_line=$(printf '%s\n' "$bench_out" | sed -E -n 's/^  Position:[[:space:]]+(.+)$/decode-position \1/p' | tail -n 1)
decode_median_ms=$(printf '%s\n' "$bench_out" | sed -E -n 's/^  Median:[[:space:]]+([0-9.]+)ms\/tok[[:space:]]+\([0-9.]+ tok\/s\)$/\1/p' | tail -n 1)
decode_line=$(printf '%s\n' "$bench_out" | sed -E -n 's/^  Median:[[:space:]]+([0-9.]+)ms\/tok[[:space:]]+\(([0-9.]+) tok\/s\)$/decode-median \1ms\/tok  \2 tok\/s/p' | tail -n 1)
decode_tps=$(printf '%s\n' "$bench_out" | sed -E -n 's/^  Median:[[:space:]]+[0-9.]+ms\/tok[[:space:]]+\(([0-9.]+) tok\/s\)$/\1/p' | tail -n 1)
decode_p95_line=$(printf '%s\n' "$bench_out" | sed -E -n 's/^  P95:[[:space:]]+([0-9.]+)ms\/tok$/decode-p95 \1ms\/tok/p' | tail -n 1)
decode_p95_ms=$(printf '%s\n' "$bench_out" | sed -E -n 's/^  P95:[[:space:]]+([0-9.]+)ms\/tok$/\1/p' | tail -n 1)
pure_gpu_med_line=$(printf '%s\n' "$bench_out" | sed -E -n 's/^  Pure GPU med:[[:space:]]+([0-9.]+)ms\/tok$/decode-pure-gpu-median \1ms\/tok/p' | tail -n 1)
pure_gpu_p95_line=$(printf '%s\n' "$bench_out" | sed -E -n 's/^  Pure GPU p95:[[:space:]]+([0-9.]+)ms\/tok$/decode-pure-gpu-p95 \1ms\/tok/p' | tail -n 1)
cpu_med_line=$(printf '%s\n' "$bench_out" | sed -E -n 's/^  CPU med:[[:space:]]+([0-9.]+)ms\/tok$/decode-cpu-median \1ms\/tok/p' | tail -n 1)
cpu_p95_line=$(printf '%s\n' "$bench_out" | sed -E -n 's/^  CPU p95:[[:space:]]+([0-9.]+)ms\/tok$/decode-cpu-p95 \1ms\/tok/p' | tail -n 1)
submit_med_line=$(printf '%s\n' "$bench_out" | sed -E -n 's/^  Submit med:[[:space:]]+([0-9.]+)ms\/tok$/decode-submit-median \1ms\/tok/p' | tail -n 1)
submit_p95_line=$(printf '%s\n' "$bench_out" | sed -E -n 's/^  Submit p95:[[:space:]]+([0-9.]+)ms\/tok$/decode-submit-p95 \1ms\/tok/p' | tail -n 1)
read_med_line=$(printf '%s\n' "$bench_out" | sed -E -n 's/^  Read med:[[:space:]]+([0-9.]+)ms\/tok$/decode-read-median \1ms\/tok/p' | tail -n 1)
read_p95_line=$(printf '%s\n' "$bench_out" | sed -E -n 's/^  Read p95:[[:space:]]+([0-9.]+)ms\/tok$/decode-read-p95 \1ms\/tok/p' | tail -n 1)
observed_metrics=(
  decode_median_ms_per_token
  decode_tokens_per_second
)
if [[ -n "$decode_p95_ms" ]]; then
  observed_metrics+=(decode_p95_ms_per_token)
fi

[[ -n "$decode_line" ]] || {
  printf '%s\n' "$bench_out" >&2
  echo "Failed to parse decode bench output" >&2
  exit 1
}

[[ -n "$position_line" ]] && printf '%s\n' "$position_line"
printf '%s\n' "$decode_line"
[[ -n "$decode_p95_line" ]] && printf '%s\n' "$decode_p95_line"
[[ -n "$pure_gpu_med_line" ]] && printf '%s\n' "$pure_gpu_med_line"
[[ -n "$pure_gpu_p95_line" ]] && printf '%s\n' "$pure_gpu_p95_line"
[[ -n "$cpu_med_line" ]] && printf '%s\n' "$cpu_med_line"
[[ -n "$cpu_p95_line" ]] && printf '%s\n' "$cpu_p95_line"
[[ -n "$submit_med_line" ]] && printf '%s\n' "$submit_med_line"
[[ -n "$submit_p95_line" ]] && printf '%s\n' "$submit_p95_line"
[[ -n "$read_med_line" ]] && printf '%s\n' "$read_med_line"
[[ -n "$read_p95_line" ]] && printf '%s\n' "$read_p95_line"

gate_failures=()
if [[ -n "$min_decode_tps" && -n "$decode_tps" ]] && ! float_ge "$decode_tps" "$min_decode_tps"; then
  gate_failures+=("decode tok/s $decode_tps < $min_decode_tps")
fi
if [[ -n "$max_decode_p95_ms" && -n "$decode_p95_ms" ]] && ! float_le "$decode_p95_ms" "$max_decode_p95_ms"; then
  gate_failures+=("decode p95 ${decode_p95_ms}ms > ${max_decode_p95_ms}ms")
fi

IFS=',' read -r -a token_counts <<< "$prefill_tokens"
prefill_metric_lines=()
for tokens in "${token_counts[@]}"; do
  prefill_args=(lab bench prefill "$pkg_path" --tokens "$tokens" --iterations "$prefill_iterations" --warmup "$prefill_warmup")
  min_tok_s=$(lookup_threshold "$min_prefill_tps" "$tokens" || true)
  max_p95_ms=$(lookup_threshold "$max_prefill_p95_ms" "$tokens" || true)
  if [[ -n "$min_tok_s" ]]; then
    prefill_args+=(--min-tps "$min_tok_s")
  fi
  if [[ -n "$max_p95_ms" ]]; then
    prefill_args+=(--max-p95-ms "$max_p95_ms")
  fi
  if [[ -n "$benchmark_settle_timeout" ]]; then
    prefill_args+=(--benchmark-settle-timeout "$benchmark_settle_timeout")
  fi
  if [[ -n "$benchmark_settle_interval" ]]; then
    prefill_args+=(--benchmark-settle-interval "$benchmark_settle_interval")
  fi
  if ! prefill_out=$("${agent_cmd[@]}" "${prefill_args[@]}" 2>&1); then
    printf '%s\n' "$prefill_out" >&2
    gate_failures+=("prefill-$tokens command failed")
    continue
  fi
  wall_ms=$(printf '%s\n' "$prefill_out" | sed -n 's/^  Wall time:    \([0-9.]*\)ms\/prefill.*/\1/p' | tail -n 1)
  p95_ms=$(printf '%s\n' "$prefill_out" | sed -n 's/^  P95:          \([0-9.]*\)ms\/prefill$/\1/p' | tail -n 1)
  tok_s=$(printf '%s\n' "$prefill_out" | sed -n 's/^  Tokens\/sec:   \([0-9.]*\)$/\1/p' | tail -n 1)
  if [[ -z "$wall_ms" || -z "$tok_s" || -z "$p95_ms" ]]; then
    printf '%s\n' "$prefill_out" >&2
    gate_failures+=("prefill-$tokens missing required metrics")
    continue
  fi
  observed_metrics+=(
    "prefill${tokens}_wall_ms"
    "prefill${tokens}_tokens_per_second"
    "prefill${tokens}_p95_ms"
  )
  prefill_metric_lines+=("$tokens"$'\t'"$wall_ms"$'\t'"$p95_ms"$'\t'"$tok_s")
  printf 'prefill-%s-median %sms/prefill  %s tok/s\n' "$tokens" "$wall_ms" "$tok_s"
  printf 'prefill-%s-p95 %sms/prefill\n' "$tokens" "$p95_ms"
  if [[ -n "$min_tok_s" ]] && ! float_ge "$tok_s" "$min_tok_s"; then
    gate_failures+=("prefill-$tokens tok/s $tok_s < $min_tok_s")
  fi
  if [[ -n "$max_p95_ms" ]] && ! float_le "$p95_ms" "$max_p95_ms"; then
    gate_failures+=("prefill-$tokens p95 ${p95_ms}ms > ${max_p95_ms}ms")
  fi
done

if [[ "$use_package_profile" != 0 ]]; then
  missing_profile_metrics=()
  if [[ ${#profile_required_output_metrics[@]} -gt 0 ]]; then
    for metric in "${profile_required_output_metrics[@]}"; do
      case "$metric" in
        trace_first_token_ms)
          continue
          ;;
      esac
      if ! metric_seen "$metric" "${observed_metrics[@]}"; then
        missing_profile_metrics+=("$metric")
      fi
    done
  fi
  if [[ ${#missing_profile_metrics[@]} -gt 0 ]]; then
    for metric in "${missing_profile_metrics[@]}"; do
      printf 'package profile required output metric missing: %s\n' "$metric" >&2
    done
    exit 1
  fi
fi

write_output_json() {
  [[ -n "$output_json" ]] || return 0
  mkdir -p "$(dirname "$output_json")"
  PREFILL_METRICS_TSV=""
  OBSERVED_METRICS_TSV=""
  GATE_FAILURES_TSV=""
  PROFILE_REQUIRED_OUTPUT_METRICS_TSV=""
  PROFILE_MIN_BOUNDS_TSV=""
  PROFILE_MAX_BOUNDS_TSV=""
  PROFILE_DEFERRED_BOUNDS_TSV=""
  if [[ ${#prefill_metric_lines[@]} -gt 0 ]]; then
    PREFILL_METRICS_TSV=$(printf '%s\n' "${prefill_metric_lines[@]}")
  fi
  if [[ ${#observed_metrics[@]} -gt 0 ]]; then
    OBSERVED_METRICS_TSV=$(printf '%s\n' "${observed_metrics[@]}")
  fi
  if [[ ${#gate_failures[@]} -gt 0 ]]; then
    GATE_FAILURES_TSV=$(printf '%s\n' "${gate_failures[@]}")
  fi
  if [[ ${#profile_required_output_metrics[@]} -gt 0 ]]; then
    PROFILE_REQUIRED_OUTPUT_METRICS_TSV=$(printf '%s\n' "${profile_required_output_metrics[@]}")
  fi
  if [[ ${#profile_min_bound_lines[@]} -gt 0 ]]; then
    PROFILE_MIN_BOUNDS_TSV=$(printf '%s\n' "${profile_min_bound_lines[@]}")
  fi
  if [[ ${#profile_max_bound_lines[@]} -gt 0 ]]; then
    PROFILE_MAX_BOUNDS_TSV=$(printf '%s\n' "${profile_max_bound_lines[@]}")
  fi
  if [[ ${#profile_deferred_bound_lines[@]} -gt 0 ]]; then
    PROFILE_DEFERRED_BOUNDS_TSV=$(printf '%s\n' "${profile_deferred_bound_lines[@]}")
  fi
  BENCH_COMMAND_TSV=$(printf '%s\t' "${agent_cmd[@]}" "${bench_args[@]}")
  export PREFILL_METRICS_TSV OBSERVED_METRICS_TSV GATE_FAILURES_TSV
  export PROFILE_REQUIRED_OUTPUT_METRICS_TSV PROFILE_MIN_BOUNDS_TSV
  export PROFILE_MAX_BOUNDS_TSV PROFILE_DEFERRED_BOUNDS_TSV BENCH_COMMAND_TSV
  python3 - "$output_json" "$pkg_path" "$profile_gate" "$use_package_profile" \
    "$position" "$decode_median_ms" "$decode_tps" "$decode_p95_ms" \
    "$decode_iterations" "$decode_warmup" "$prefill_iterations" "$prefill_warmup" <<'PY'
import json
import math
import os
import sys

(
    output_path,
    package_path,
    profile_gate,
    use_package_profile,
    position,
    decode_median_ms,
    decode_tps,
    decode_p95_ms,
    decode_iterations,
    decode_warmup,
    prefill_iterations,
    prefill_warmup,
) = sys.argv[1:]


def number(value):
    if value == "":
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    return parsed


def lines(name):
    text = os.environ.get(name, "")
    return [line for line in text.splitlines() if line]


prefill = {}
for line in lines("PREFILL_METRICS_TSV"):
    tokens, wall_ms, p95_ms, tok_s = line.split("\t")
    prefill[tokens] = {
        "wall_ms": number(wall_ms),
        "p95_ms": number(p95_ms),
        "tokens_per_second": number(tok_s),
    }

command = [
    part
    for part in os.environ.get("BENCH_COMMAND_TSV", "").split("\t")
    if part
]
doc = {
    "schema": 1,
    "gate": profile_gate or None,
    "package_profile": use_package_profile != "0",
    "package": os.path.realpath(package_path),
    "command": command,
    "decode": {
        "position": position or None,
        "iterations": int(decode_iterations),
        "warmup": int(decode_warmup),
        "median_ms_per_token": number(decode_median_ms),
        "tokens_per_second": number(decode_tps),
        "p95_ms_per_token": number(decode_p95_ms),
    },
    "prefill": {
        "iterations": int(prefill_iterations),
        "warmup": int(prefill_warmup),
        "metrics": prefill,
    },
    "observed_metrics": lines("OBSERVED_METRICS_TSV"),
    "required_output_metrics": lines("PROFILE_REQUIRED_OUTPUT_METRICS_TSV"),
    "profile_min_bounds": lines("PROFILE_MIN_BOUNDS_TSV"),
    "profile_max_bounds": lines("PROFILE_MAX_BOUNDS_TSV"),
    "profile_deferred_bounds": lines("PROFILE_DEFERRED_BOUNDS_TSV"),
    "gate_failures": lines("GATE_FAILURES_TSV"),
}
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(doc, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
  echo "benchmark-json $output_json"
}

write_output_json

if [[ ${#gate_failures[@]} -gt 0 ]]; then
  printf 'benchmark-gates failed\n' >&2
  for failure in "${gate_failures[@]}"; do
    printf '  %s\n' "$failure" >&2
  done
  exit 2
fi
