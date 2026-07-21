#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat >&2 <<'USAGE'
Usage: bash tools/build-module-source-plan.sh [options]

Options:
  --plan FILE          TSV source plan. Default: Models/source-build-plan.tsv.
  --package-plan FILE  TSV package plan to cross-check source_package paths.
                       Default: Models/package-build-plan.tsv.
  --id NAME            Build one row by id. Repeatable.
  --all                Build every row in the plan.
  --skip-tool-build    Use existing .build/release/smelt.
  -h, --help           Show this help.

Plan columns:
  id<TAB>module_path<TAB>source_default<TAB>source_package<TAB>package_result<TAB>build_argv
USAGE
  exit 1
}

plan=${SMELT_MODULE_SOURCE_BUILD_PLAN:-Models/source-build-plan.tsv}
package_plan=${SMELT_MODULE_PACKAGE_BUILD_PLAN:-Models/package-build-plan.tsv}
skip_tool_build=0
build_all=0
ids=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan)
      [[ $# -ge 2 ]] || usage
      plan=$2
      shift 2
      ;;
    --package-plan)
      [[ $# -ge 2 ]] || usage
      package_plan=$2
      shift 2
      ;;
    --id)
      [[ $# -ge 2 ]] || usage
      ids+=("$2")
      shift 2
      ;;
    --all)
      build_all=1
      shift
      ;;
    --skip-tool-build)
      skip_tool_build=1
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

if ((build_all == 0 && ${#ids[@]} == 0)); then
  usage
fi

if [[ ! -f "$plan" ]]; then
  echo "module source build plan not found: $plan" >&2
  exit 1
fi
if [[ ! -f "$package_plan" ]]; then
  echo "module package build plan not found: $package_plan" >&2
  exit 1
fi

contains() {
  local needle=$1
  shift
  local value
  for value in "$@"; do
    [[ "$value" == "$needle" ]] && return 0
  done
  return 1
}

expand_path() {
  local raw=$1
  case "$raw" in
    "~") printf '%s\n' "$HOME" ;;
    "~/"*) printf '%s\n' "$HOME/${raw#"~/"}" ;;
    *) printf '%s\n' "$raw" ;;
  esac
}

source_env_name() {
  local row_id=$1
  local suffix
  suffix=$(printf '%s' "$row_id" | sed 's/[^A-Za-z0-9]/_/g' | tr '[:lower:]' '[:upper:]')
  printf 'SMELT_MODULE_SOURCE_%s\n' "$suffix"
}

package_plan_artifact_root() {
  local wanted_id=$1
  local row_id module_path artifact_root output_package evidence_json rest
  while IFS=$'\t' read -r row_id module_path artifact_root output_package evidence_json rest; do
    [[ -z "${row_id:-}" || "${row_id:0:1}" == "#" ]] && continue
    if [[ "$row_id" == "$wanted_id" ]]; then
      printf '%s\n' "$artifact_root"
      return 0
    fi
  done < "$package_plan"
  return 1
}

# The module descriptor is the authored `<id>.module.json` (JSON), not the
# grammar `.cam` oracle. Decode the `weights` source declaration structurally
# rather than scraping grammar text.
module_weights_locator() {
  local module_path=$1
  python3 - "$module_path" <<'PY'
import json, sys
try:
    doc = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
for src in doc.get("sources", []):
    if src.get("id") == "weights" and src.get("kind") == "hf":
        print(src.get("locator", ""))
        break
PY
}

module_weights_checkpoint_map() {
  local module_path=$1
  python3 - "$module_path" <<'PY'
import json, sys
try:
    doc = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
for src in doc.get("sources", []):
    if src.get("id") == "weights":
        cm = src.get("checkpointMap")
        if cm:
            print(cm)
        break
PY
}

build_argv_uses_module_source_package() {
  local build_argv=$1
  local raw_args=()
  local arg
  IFS='|' read -r -a raw_args <<< "$build_argv"
  for arg in "${raw_args[@]}"; do
    [[ "$arg" == "--module-source-package" ]] && return 0
  done
  return 1
}

require_module_source_package_argv_targets_cam() {
  local row_id=$1
  local module_path=$2
  local build_argv=$3
  local raw_args=()
  IFS='|' read -r -a raw_args <<< "$build_argv"
  if ((${#raw_args[@]} < 2)) || [[ "${raw_args[0]}" != "build" ]]; then
    echo "module source row $row_id uses --module-source-package but build command is not a smelt build." >&2
    exit 1
  fi
  if [[ "${raw_args[1]}" != "{module}" && "${raw_args[1]}" != "$module_path" ]]; then
    echo "module source row $row_id uses --module-source-package but build input is not the module row path." >&2
    echo "  expected: {module} or $module_path" >&2
    echo "  got:      ${raw_args[1]}" >&2
    exit 1
  fi
}

require_checkpoint_map_alignment() {
  local row_id=$1
  local module_path=$2
  local build_argv=$3
  local module_map spec_map
  module_map=$(module_weights_checkpoint_map "$module_path" || true)
  if build_argv_uses_module_source_package "$build_argv"; then
    require_module_source_package_argv_targets_cam "$row_id" "$module_path" "$build_argv"
    if [[ -z "$module_map" ]]; then
      echo "module source row $row_id uses module source package build but module source weights has no checkpoint-map." >&2
      exit 1
    fi
    return 0
  fi
  # Non-CAM-source-package build rows used the legacy model-spec DSL, which was retired;
  # no such rows remain, so a build spec never declares its own checkpoint_map.
  spec_map=""
  if [[ -z "$module_map" && -n "$spec_map" ]]; then
    echo "module source row $row_id build spec declares checkpoint_map '$spec_map' but module source weights has none." >&2
    exit 1
  fi
  if [[ -n "$module_map" && -z "$spec_map" ]]; then
    echo "module source row $row_id declares checkpoint-map '$module_map' but build spec has none." >&2
    exit 1
  fi
  if [[ -n "$module_map" && "$module_map" != "$spec_map" ]]; then
    echo "module source row $row_id checkpoint-map mismatch." >&2
    echo "  module:     $module_map" >&2
    echo "  build spec: $spec_map" >&2
    exit 1
  fi
}

has_carried_tensor_artifact() {
  local input_path=$1
  python3 - "$input_path" <<'PY'
import json, os, sys

root = sys.argv[1]

def complete_index(path):
    try:
        with open(path, "rb") as handle:
            document = json.load(handle)
        shards = set(document.get("weight_map", {}).values())
    except (OSError, ValueError, TypeError):
        return False
    if not shards:
        return False
    parent = os.path.dirname(path)
    return all(
        isinstance(shard, str)
        and os.path.isfile(os.path.join(parent, shard))
        and os.path.getsize(os.path.join(parent, shard)) > 0
        for shard in shards
    )

def supported_file(path):
    return path.endswith((".safetensors", ".npz", ".bin")) \
        and os.path.isfile(path) and os.path.getsize(path) > 0

if os.path.isfile(root):
    ok = complete_index(root) if root.endswith(".safetensors.index.json") \
        else supported_file(root)
    sys.exit(0 if ok else 1)
if not os.path.isdir(root):
    sys.exit(1)

indexes = []
artifacts = []
for directory, names, files in os.walk(root):
    names[:] = [name for name in names if name != ".cache"]
    for name in files:
        path = os.path.join(directory, name)
        if name.endswith(".safetensors.index.json"):
            indexes.append(path)
        elif supported_file(path):
            artifacts.append(path)

# An index is an all-shards contract. Once present, a few downloaded shards
# must not masquerade as a complete checkpoint.
ok = any(complete_index(path) for path in indexes) if indexes else bool(artifacts)
sys.exit(0 if ok else 1)
PY
}

describe_incomplete_indexed_checkpoint() {
  local input_path=$1
  python3 - "$input_path" <<'PY'
import json, os, sys

root = sys.argv[1]
if os.path.isfile(root):
    indexes = [root] if root.endswith(".safetensors.index.json") else []
elif os.path.isdir(root):
    indexes = []
    for directory, names, files in os.walk(root):
        names[:] = [name for name in names if name != ".cache"]
        indexes.extend(
            os.path.join(directory, name)
            for name in files if name.endswith(".safetensors.index.json")
        )
else:
    indexes = []

for index in sorted(indexes):
    try:
        with open(index, "rb") as handle:
            shards = sorted(set(json.load(handle).get("weight_map", {}).values()))
    except (OSError, ValueError, TypeError):
        print(f"Indexed safetensors checkpoint is unreadable: {index}", file=sys.stderr)
        continue
    parent = os.path.dirname(index)
    missing = [
        shard for shard in shards
        if not isinstance(shard, str)
        or not os.path.isfile(os.path.join(parent, shard))
        or os.path.getsize(os.path.join(parent, shard)) == 0
    ]
    if missing:
        preview = ", ".join(map(str, missing[:3]))
        suffix = "" if len(missing) <= 3 else f", ... (+{len(missing) - 3})"
        print(
            f"Indexed safetensors checkpoint is incomplete: {len(missing)} "
            f"missing/empty shard(s) from {os.path.basename(index)}: {preview}{suffix}",
            file=sys.stderr,
        )
PY
}

substitute_build_arg() {
  local arg=$1
  local source_path=$2
  local module_path=$3
  local output_path=$4
  local stage_path=$5
  arg=${arg//\{source\}/$source_path}
  arg=${arg//\{module\}/$module_path}
  arg=${arg//\{output\}/$output_path}
  arg=${arg//\{stage\}/$stage_path}
  printf '%s\n' "$arg"
}

run_row_build_command() {
  local build_argv=$1
  local source_path=$2
  local module_path=$3
  local output_path=$4
  local stage_path=$5
  local raw_args=()
  local argv=()
  local arg
  IFS='|' read -r -a raw_args <<< "$build_argv"
  if ((${#raw_args[@]} == 0)); then
    echo "module source build command is empty" >&2
    exit 1
  fi
  for arg in "${raw_args[@]}"; do
    if [[ -z "$arg" ]]; then
      echo "module source build command contains an empty argument" >&2
      exit 1
    fi
    argv+=("$(substitute_build_arg "$arg" "$source_path" "$module_path" "$output_path" "$stage_path")")
  done
  .build/release/smelt "${argv[@]}"
}

require_source_input() {
  local row_id=$1
  local module_path=$2
  local env_name=$3
  local input_path=$4
  if [[ -e "$input_path" ]] && has_carried_tensor_artifact "$input_path"; then
    return 0
  fi
  local locator
  locator=$(module_weights_locator "$module_path" || true)
  if [[ ! -e "$input_path" && -n "$locator" ]]; then
    echo "Missing module source input for $row_id ($locator): $input_path" >&2
  elif [[ ! -e "$input_path" ]]; then
    echo "Missing module source input for $row_id: $input_path" >&2
  else
    echo "module source input for $row_id has no carried tensor artifact: $input_path" >&2
    describe_incomplete_indexed_checkpoint "$input_path"
    echo "Expected *.safetensors, *.safetensors.index.json, *.npz, or *.bin." >&2
  fi
  echo "Set $env_name to a local checkpoint or optimized artifact path." >&2
  if [[ "$plan" == "Models/source-build-plan.tsv" ]]; then
    echo "Or fetch the CAM-declared source with: bash tools/fetch-module-source-inputs.sh --id $row_id" >&2
  else
    echo "Or fetch the CAM-declared source with: bash tools/fetch-module-source-inputs.sh --plan $plan --id $row_id" >&2
  fi
  exit 1
}

ensure_single_staged_package() {
  local stage=$1
  local row_id=$2
  local found=()
  while IFS= read -r package; do
    found+=("$package")
  done < <(find "$stage" -maxdepth 1 -type d -name '*.smeltpkg' | sort)
  if ((${#found[@]} != 1)); then
    echo "Expected exactly one staged source package for $row_id in $stage; found ${#found[@]}." >&2
    exit 1
  fi
  printf '%s\n' "${found[0]}"
}

if [[ "$skip_tool_build" == 0 ]]; then
  swift build -c release >/dev/null
elif [[ ! -x .build/release/smelt ]]; then
  echo "Missing .build/release/smelt; omit --skip-tool-build to build it." >&2
  exit 1
fi

matched=0
while IFS=$'\t' read -r row_id module_path source_default source_package package_result build_argv rest; do
  [[ -z "${row_id:-}" || "${row_id:0:1}" == "#" ]] && continue
  if [[ -n "${rest:-}" ]]; then
    echo "Invalid module source build plan row for $row_id: too many columns" >&2
    exit 1
  fi
  if [[ -z "${row_id:-}" || -z "${module_path:-}" ||
        -z "${source_default:-}" || -z "${source_package:-}" ||
        -z "${package_result:-}" || -z "${build_argv:-}" ]]; then
    echo "Invalid module source build plan row: missing required column" >&2
    exit 1
  fi
  row_selected=0
  if ((build_all == 1)); then
    row_selected=1
  elif ((${#ids[@]} > 0)) && contains "$row_id" "${ids[@]}"; then
    row_selected=1
  fi
  if ((row_selected == 0)); then
    continue
  fi

  matched=1
  if [[ ! -f "$module_path" ]]; then
    echo "module not found for $row_id: $module_path" >&2
    exit 1
  fi
  expected_source_package=$(package_plan_artifact_root "$row_id" || true)
  if [[ -z "$expected_source_package" ]]; then
    echo "module source row $row_id has no matching package plan row." >&2
    exit 1
  fi
  if [[ "$source_package" != "$expected_source_package" ]]; then
    echo "module source row $row_id source_package does not match package plan artifact_root." >&2
    echo "  source plan:  $source_package" >&2
    echo "  package plan: $expected_source_package" >&2
    exit 1
  fi
  require_checkpoint_map_alignment "$row_id" "$module_path" "$build_argv"

  source_env=$(source_env_name "$row_id")
  raw_input=${!source_env:-$source_default}
  input_path=$(expand_path "$raw_input")
  require_source_input "$row_id" "$module_path" "$source_env" "$input_path"

  echo "module source build $row_id"
  rm -rf "$source_package"
  mkdir -p "$(dirname "$source_package")"

  case "$package_result" in
    staged-single)
      source_stage=".build/module-source-build/$row_id"
      rm -rf "$source_stage"
      mkdir -p "$source_stage"
      run_row_build_command "$build_argv" "$input_path" "$module_path" "$source_package" "$source_stage"
      staged_package=$(ensure_single_staged_package "$source_stage" "$row_id")
      mv "$staged_package" "$source_package"
      rm -rf "$source_stage"
      ;;
    direct)
      run_row_build_command "$build_argv" "$input_path" "$module_path" "$source_package" ""
      ;;
    *)
      echo "Unsupported module source package_result for $row_id: $package_result" >&2
      exit 1
      ;;
  esac

  if [[ ! -d "$source_package" ]]; then
    echo "module source build did not produce source package for $row_id: $source_package" >&2
    exit 1
  fi

  # The checked package build regenerates the canonical module descriptor and
  # requires generated payloads to be absent from its source root. The source
  # package is also allowed to carry compiler byproducts that are not declared
  # in source_package_files. Drop both classes at the producer so a canonical
  # package build is repeatable.
  rm -f \
    "$source_package/module.json" \
    "$source_package/SmeltGeneratedKernels.metal" \
    "$source_package/gptq_capture_points.json" \
    "$source_package/model.metalarchive" \
    "$source_package/trace_markers.json"
done < "$plan"

if [[ "$matched" == 0 ]]; then
  echo "No module source build plan rows matched." >&2
  exit 1
fi
