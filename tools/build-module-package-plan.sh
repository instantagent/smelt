#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat >&2 <<'USAGE'
Usage: bash tools/build-module-package-plan.sh [options]

Options:
  --plan FILE          TSV plan. Default: Models/package-build-plan.tsv.
  --source-plan FILE   TSV source materialization plan.
                       Default: Models/source-build-plan.tsv.
  --id NAME            Build one row by id. Repeatable.
  --all                Build every row in the plan.
  --evidence-dir DIR   module build evidence output dir. Default: .build/module-build-evidence.
  --skip-source-build  Use existing source artifact roots.
  --skip-tool-build    Use existing .build/release/smelt.
  -h, --help           Show this help.

Plan columns:
  id<TAB>module_path<TAB>artifact_root<TAB>output_package<TAB>evidence_json
USAGE
  exit 1
}

plan=${SMELT_MODULE_PACKAGE_BUILD_PLAN:-Models/package-build-plan.tsv}
source_plan=${SMELT_MODULE_SOURCE_BUILD_PLAN:-Models/source-build-plan.tsv}
evidence_dir=${SMELT_MODULE_BUILD_EVIDENCE_DIR:-.build/module-build-evidence}
skip_source_build=0
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
    --source-plan)
      [[ $# -ge 2 ]] || usage
      source_plan=$2
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
    --evidence-dir)
      [[ $# -ge 2 ]] || usage
      evidence_dir=$2
      shift 2
      ;;
    --skip-source-build)
      skip_source_build=1
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
  echo "module package build plan not found: $plan" >&2
  exit 1
fi
if [[ "$skip_source_build" == 0 && ! -f "$source_plan" ]]; then
  echo "module source build plan not found: $source_plan" >&2
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

if [[ "$skip_tool_build" == 0 ]]; then
  swift build -c release >/dev/null
elif [[ ! -x .build/release/smelt ]]; then
  echo "Missing .build/release/smelt; omit --skip-tool-build to build it." >&2
  exit 1
fi

mkdir -p "$evidence_dir"

matched=0
while IFS=$'\t' read -r row_id module_path artifact_root output_package evidence_json rest; do
  [[ -z "${row_id:-}" || "${row_id:0:1}" == "#" ]] && continue
  if [[ -n "${rest:-}" ]]; then
    echo "Invalid module package build plan row for $row_id: too many columns" >&2
    exit 1
  fi
  if [[ -z "${row_id:-}" || -z "${module_path:-}" || -z "${artifact_root:-}" ||
        -z "${output_package:-}" || -z "${evidence_json:-}" ]]; then
    echo "Invalid module package build plan row: missing required column" >&2
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
  mkdir -p "$(dirname "$output_package")" "$(dirname "$evidence_dir/$evidence_json")"
  if [[ "$skip_source_build" == 0 ]]; then
    bash tools/build-module-source-plan.sh \
      --plan "$source_plan" \
      --package-plan "$plan" \
      --id "$row_id" \
      --skip-tool-build
  fi
  echo "module build $row_id"
  .build/release/smelt build "$module_path" \
    --output "$output_package" \
    --module-artifact-root "$artifact_root" \
    --module-build-evidence-json "$evidence_dir/$evidence_json"
done < "$plan"

if [[ "$matched" == 0 ]]; then
  echo "No module package build plan rows matched." >&2
  exit 1
fi
