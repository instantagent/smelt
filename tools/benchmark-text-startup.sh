#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat >&2 <<'USAGE'
Usage: bash tools/benchmark-text-startup.sh <model.smeltpkg> [options]

Options:
  --prompt TEXT              Prompt to run. Default: hi
  --max-tokens N             Tokens to generate. Default: 1
  --iterations N             Fresh-process timed runs. Default: 5
  --warmup N                 Fresh-process warmup runs before timing. Default: 0
  --template NAME            Forwarded to `smelt run`.
  --system TEXT              Forwarded to `smelt run`.
  --system-file FILE         Forwarded to `smelt run`.
  --temp T                   Forwarded to `smelt run`. Default: 0
  --seed N                   Forwarded to `smelt run` when set.
  --max-median-ms N          Fail unless median process wall time <= N.
  --max-first-ms N           Fail unless first timed process wall time <= N.
  --max-trace-first-ms N     Fail unless median trace-derived first-token time <= N.
  --use-package-profile      Load startup trace labels, required metrics, and bounds from manifest validation.
  --output-json FILE         Write raw measurements. Default: /tmp/smelt-text-startup-*.json
  --skip-build               Use existing .build/release/smelt.
USAGE
  exit 1
}

ensure_release_ia() {
  if [[ ! -d third_party/llguidance/CLLGuidance.xcframework ]]; then
    bash tools/build-llguidance.sh >/dev/null
  fi
  swift package describe --type json >/dev/null
  swift build -c release >/dev/null
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
  local profile_args=(module-profile text.decode-prefill-startup)
  if [[ -n "$model_name" ]]; then
    profile_args+=(--model-name "$model_name")
  fi
  canonical_profile=$(.build/release/smelt "${profile_args[@]}")
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

labels = profile.get("required_trace_labels")
require_strings(labels, canonical["required_trace_labels"], "required trace label")
metrics = profile.get("required_output_metrics")
require_strings(metrics, canonical["required_output_metrics"], "required output metric")

print(f"gate\t{gate_id}")
for label in labels:
    if not isinstance(label, str) or not label:
        print("text package performance_profile has an invalid required trace label", file=sys.stderr)
        sys.exit(1)
    print(f"trace_label\t{label}")

for metric in metrics:
    if not isinstance(metric, str) or not metric:
        print("text package performance_profile has an invalid required output metric", file=sys.stderr)
        sys.exit(1)
    print(f"metric\t{metric}")

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
    print(f"bound\t{metric}\t{max_value:g}\t{unit}")

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

json_array() {
  python3 - "$@" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1:], separators=(",", ":")))
PY
}

[[ $# -ge 1 ]] || usage
package_path=$1
shift

prompt=${SMELT_TEXT_STARTUP_PROMPT:-hi}
max_tokens=${SMELT_TEXT_STARTUP_MAX_TOKENS:-1}
iterations=${SMELT_TEXT_STARTUP_ITERATIONS:-5}
warmup=${SMELT_TEXT_STARTUP_WARMUP:-0}
template=${SMELT_TEXT_STARTUP_TEMPLATE:-}
system=${SMELT_TEXT_STARTUP_SYSTEM:-}
system_file=${SMELT_TEXT_STARTUP_SYSTEM_FILE:-}
temp=${SMELT_TEXT_STARTUP_TEMP:-0}
seed=${SMELT_TEXT_STARTUP_SEED:-}
max_median_ms=${SMELT_TEXT_STARTUP_MAX_MEDIAN_MS:-}
max_first_ms=${SMELT_TEXT_STARTUP_MAX_FIRST_MS:-}
max_trace_first_ms=${SMELT_TEXT_STARTUP_MAX_TRACE_FIRST_MS:-}
use_package_profile=${SMELT_TEXT_STARTUP_USE_PACKAGE_PROFILE:-0}
profile_gate=
profile_max_bound_lines=()
profile_required_output_metrics=()
required_trace_labels=("exec -> main (dyld)" "tokenizer load" "SmeltModel init (total)")
json_path=${SMELT_TEXT_STARTUP_JSON:-}
skip_build=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      [[ $# -ge 2 ]] || usage
      prompt=$2
      shift 2
      ;;
    --max-tokens)
      [[ $# -ge 2 ]] || usage
      max_tokens=$2
      shift 2
      ;;
    --iterations)
      [[ $# -ge 2 ]] || usage
      iterations=$2
      shift 2
      ;;
    --warmup)
      [[ $# -ge 2 ]] || usage
      warmup=$2
      shift 2
      ;;
    --template)
      [[ $# -ge 2 ]] || usage
      template=$2
      shift 2
      ;;
    --system)
      [[ $# -ge 2 ]] || usage
      system=$2
      shift 2
      ;;
    --system-file)
      [[ $# -ge 2 ]] || usage
      system_file=$2
      shift 2
      ;;
    --temp)
      [[ $# -ge 2 ]] || usage
      temp=$2
      shift 2
      ;;
    --seed)
      [[ $# -ge 2 ]] || usage
      seed=$2
      shift 2
      ;;
    --max-median-ms)
      [[ $# -ge 2 ]] || usage
      max_median_ms=$2
      shift 2
      ;;
    --max-first-ms)
      [[ $# -ge 2 ]] || usage
      max_first_ms=$2
      shift 2
      ;;
    --max-trace-first-ms)
      [[ $# -ge 2 ]] || usage
      max_trace_first_ms=$2
      shift 2
      ;;
    --use-package-profile)
      use_package_profile=1
      shift
      ;;
    --output-json)
      [[ $# -ge 2 ]] || usage
      json_path=$2
      shift 2
      ;;
    --skip-build)
      skip_build=1
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      usage
      ;;
  esac
done

if [[ ! -f "$package_path/manifest.json" ]]; then
  echo "Missing package at $package_path" >&2
  exit 1
fi

if [[ $skip_build -eq 0 ]]; then
  ensure_release_ia
elif [[ ! -x .build/release/smelt ]]; then
  echo "Missing .build/release/smelt; omit --skip-build to build it." >&2
  exit 1
fi

if [[ "$use_package_profile" != 0 ]]; then
  required_trace_labels=()
  profile_dump=$(read_package_profile "$package_path/manifest.json")
  while IFS=$'\t' read -r kind metric max_value unit; do
    case "$kind" in
      gate)
        profile_gate=$metric
        ;;
      trace_label)
        required_trace_labels+=("$metric")
        ;;
      metric)
        case "$metric" in
          trace_first_token_ms)
            profile_required_output_metrics+=("$metric")
            ;;
          *)
            ;;
        esac
        ;;
      bound)
        if [[ "$unit" != "ms" ]]; then
          echo "text package profile max-bound '$metric' uses unsupported unit '$unit'." >&2
          exit 1
        fi
        case "$metric" in
          trace_first_token_ms)
            [[ -n "$max_trace_first_ms" ]] || max_trace_first_ms=$max_value
            ;;
          *)
            echo "text package profile max-bound '$metric' is not enforced by this harness." >&2
            exit 1
            ;;
        esac
        profile_max_bound_lines+=("$metric $max_value$unit")
        ;;
    esac
  done <<<"$profile_dump"
fi

run_id="$(basename "$package_path" .smeltpkg)-$$"
json_path=${json_path:-${TMPDIR:-/tmp}/smelt-text-startup-${run_id}.json}

cmd=(.build/release/smelt run "$package_path" --prompt "$prompt" --max-tokens "$max_tokens" --temp "$temp")
if [[ -n "$seed" ]]; then
  cmd+=(--seed "$seed")
fi
if [[ -n "$template" ]]; then
  cmd+=(--template "$template")
fi
if [[ -n "$system" ]]; then
  cmd+=(--system "$system")
fi
if [[ -n "$system_file" ]]; then
  cmd+=(--system-file "$system_file")
fi

required_trace_labels_json=$(json_array "${required_trace_labels[@]}")
if ((${#profile_required_output_metrics[@]} > 0)); then
  required_output_metrics_json=$(json_array "${profile_required_output_metrics[@]}")
else
  required_output_metrics_json=$(json_array)
fi
if [[ "$use_package_profile" != 0 ]]; then
  echo "text-profile-gate $profile_gate"
  if ((${#profile_required_output_metrics[@]} > 0)); then
    for metric in "${profile_required_output_metrics[@]}"; do
      echo "text-profile-required-output-metric $metric"
    done
  fi
  echo "text-profile-required-trace-labels ${required_trace_labels[*]}"
  if ((${#profile_max_bound_lines[@]} > 0)); then
    for line in "${profile_max_bound_lines[@]}"; do
      echo "text-profile-max-bound $line"
    done
  fi
fi

python3 - "$json_path" "$iterations" "$warmup" "$max_median_ms" "$max_first_ms" \
  "$max_trace_first_ms" "$required_trace_labels_json" "$required_output_metrics_json" \
  "$use_package_profile" "$profile_gate" \
  "${cmd[@]}" <<'PY'
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time

json_path = sys.argv[1]
iterations = int(sys.argv[2])
warmup = int(sys.argv[3])
max_median_ms = sys.argv[4]
max_first_ms = sys.argv[5]
max_trace_first_ms = sys.argv[6]
required_trace_labels = json.loads(sys.argv[7])
required_output_metrics = json.loads(sys.argv[8])
package_profile_mode = sys.argv[9] != "0"
profile_gate = sys.argv[10] or None
cmd = sys.argv[11:]

if iterations <= 0 or warmup < 0:
    raise SystemExit("iterations must be > 0 and warmup must be >= 0")

startup_re = re.compile(r"startup:\s*([+-]?\d+(?:\.\d+)?)ms\s+(.+)$")
timing_re = re.compile(
    r"Timing: prefill ([0-9.eE+-]+)ms, generate ([0-9.eE+-]+)ms, "
    r"([0-9.eE+-]+|inf) tok/s"
)
generated_re = re.compile(r"Generated token IDs:\s*(\[.*\])")
prompt_tokens_re = re.compile(r"Prompt tokens:\s*(\d+)")


def sha256_file(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def run_once(index, phase):
    env = dict(**os.environ, SMELT_STARTUP_TRACE="1", SMELT_CAS="0")
    start = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    elapsed = time.perf_counter() - start
    stderr = proc.stderr
    stdout = proc.stdout
    if proc.returncode != 0:
        sys.stderr.write(stdout)
        sys.stderr.write(stderr)
        raise SystemExit(proc.returncode)

    startup = []
    startup_by_label = {}
    for line in stderr.splitlines():
        match = startup_re.match(line)
        if match:
            label = match.group(2).strip()
            ms = float(match.group(1))
            startup.append({
                "label": label,
                "ms": ms,
            })
            startup_by_label[label] = ms
    timing = timing_re.search(stderr)
    generated = generated_re.search(stderr)
    prompt_tokens = prompt_tokens_re.search(stderr)
    if package_profile_mode:
        if prompt_tokens is None:
            sys.stderr.write(stderr)
            raise SystemExit("missing prompt token count required by package profile")
        if generated is None:
            sys.stderr.write(stderr)
            raise SystemExit("missing generated token IDs required by package profile")
        try:
            generated_tokens = json.loads(generated.group(1))
        except json.JSONDecodeError as error:
            sys.stderr.write(stderr)
            raise SystemExit(f"invalid generated token IDs required by package profile: {error}")
        if not isinstance(generated_tokens, list) or not generated_tokens:
            sys.stderr.write(stderr)
            raise SystemExit("empty generated token IDs required by package profile")
    prefill_ms = float(timing.group(1)) if timing else None
    trace_first_ms = None
    core_trace_labels = [
        "exec -> main (dyld)",
        "tokenizer load",
        "SmeltModel init (total)",
    ]
    missing_trace_labels = [
        label for label in required_trace_labels
        if label not in startup_by_label
    ]
    missing_core_trace_labels = [
        label for label in core_trace_labels
        if label not in startup_by_label
    ]
    requires_trace_first = "trace_first_token_ms" in required_output_metrics
    if (max_trace_first_ms or requires_trace_first) and (
        prefill_ms is None
        or missing_trace_labels
        or missing_core_trace_labels
    ):
        if prefill_ms is None:
            sys.stderr.write(stderr)
            reason = "--max-trace-first-ms" if max_trace_first_ms else "trace_first_token_ms"
            raise SystemExit(f"missing timing line required by {reason}")
        missing = list(dict.fromkeys(missing_trace_labels + missing_core_trace_labels))
        if missing:
            reason = "--max-trace-first-ms" if max_trace_first_ms else "trace_first_token_ms"
            raise SystemExit(
                f"missing startup trace labels required by {reason}: "
                + ", ".join(missing)
            )
    if prefill_ms is not None and not missing_core_trace_labels:
        exec_ms = startup_by_label["exec -> main (dyld)"]
        tokenizer_ms = startup_by_label["tokenizer load"]
        model_ms = startup_by_label["SmeltModel init (total)"]
        trace_first_ms = exec_ms + tokenizer_ms + model_ms + prefill_ms
    return {
        "index": index,
        "phase": phase,
        "wall_ms": elapsed * 1000.0,
        "startup": startup,
        "prompt_tokens": int(prompt_tokens.group(1)) if prompt_tokens else None,
        "prefill_ms": prefill_ms,
        "generate_ms": float(timing.group(2)) if timing else None,
        "tok_s": float(timing.group(3)) if timing else None,
        "trace_first_token_ms": trace_first_ms,
        "generated": generated.group(1) if generated else None,
        "stdout_preview": stdout[:200],
    }

samples = []
for i in range(warmup):
    samples.append(run_once(i, "warmup"))
timed = []
for i in range(iterations):
    sample = run_once(i, "timed")
    timed.append(sample)
    samples.append(sample)

walls = [sample["wall_ms"] for sample in timed]
trace_first_values = [
    sample["trace_first_token_ms"]
    for sample in timed
    if sample["trace_first_token_ms"] is not None
]
median = statistics.median(walls)
p95 = sorted(walls)[max(0, min(len(walls) - 1, int(len(walls) * 0.95) - 1))]
first = walls[0]
trace_first_first = timed[0]["trace_first_token_ms"] if timed else None
trace_first_median = statistics.median(trace_first_values) if trace_first_values else None
missing_output_metrics = []
if "trace_first_token_ms" in required_output_metrics and trace_first_first is None:
    missing_output_metrics.append("trace_first_token_ms")
summary = {
    "schema": 1,
    "kind": "smelt.module.text_startup_evidence",
    "command": cmd,
    "package": os.path.realpath(cmd[2]) if len(cmd) > 2 else None,
    "package_realpath": os.path.realpath(cmd[2]) if len(cmd) > 2 else None,
    "manifest_sha256": sha256_file(os.path.join(cmd[2], "manifest.json")) if len(cmd) > 2 else None,
    "package_profile": package_profile_mode,
    "profile_gate": profile_gate,
    "iterations": iterations,
    "warmup": warmup,
    "first_ms": first,
    "median_ms": median,
    "p95_ms": p95,
    "trace_first_token_first_ms": trace_first_first,
    "trace_first_token_median_ms": trace_first_median,
    "required_output_metrics": required_output_metrics,
    "required_trace_labels": required_trace_labels,
    "min_ms": min(walls),
    "max_ms": max(walls),
    "samples": samples,
}
with open(json_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print(f"text-package {cmd[2]}")
print("text-command " + " ".join(cmd))
print(f"text-startup-json {json_path}")
print(f"text-startup-first-wall {first:.3f}ms")
print(f"text-startup-median-wall {median:.3f}ms")
print(f"text-startup-p95-wall {p95:.3f}ms")
if trace_first_median is not None:
    print(f"text-startup-median-trace-first-token {trace_first_median:.3f}ms")
if missing_output_metrics:
    print(
        "missing startup output metrics required by package profile: "
        + ", ".join(missing_output_metrics),
        file=sys.stderr,
    )
    raise SystemExit(1)
if timed[0]["startup"]:
    print("text-startup-first-startup " + json.dumps(timed[0]["startup"], separators=(",", ":")))
if timed[0]["prefill_ms"] is not None:
    print(f"text-startup-first-prefill {timed[0]['prefill_ms']:.3f}ms")
if timed[0]["trace_first_token_ms"] is not None:
    print(f"text-startup-first-trace-first-token {timed[0]['trace_first_token_ms']:.3f}ms")
if timed[0]["generate_ms"] is not None:
    print(f"text-startup-first-generate {timed[0]['generate_ms']:.3f}ms")

failures = []
if max_median_ms and median > float(max_median_ms):
    failures.append(f"median wall {median:.3f}ms > {float(max_median_ms):.3f}ms")
if max_first_ms and first > float(max_first_ms):
    failures.append(f"first wall {first:.3f}ms > {float(max_first_ms):.3f}ms")
if (
    max_trace_first_ms
    and trace_first_first is not None
    and trace_first_first > float(max_trace_first_ms)
):
    failures.append(
        f"first trace first token {trace_first_first:.3f}ms "
        f"> {float(max_trace_first_ms):.3f}ms"
    )
if max_trace_first_ms and trace_first_first is None:
    failures.append("first trace first token unavailable")
if failures:
    print("text-startup-gates failed", file=sys.stderr)
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    raise SystemExit(2)
PY
