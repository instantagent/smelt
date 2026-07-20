#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat >&2 <<'USAGE'
Usage: bash tools/fetch-module-source-inputs.sh [options]

Options:
  --plan FILE       TSV source plan. Default: Models/source-build-plan.tsv.
  --id NAME         Fetch one row by id. Repeatable.
  --all             Fetch every row in the plan.
  --dry-run         Ask hf to report downloads without writing files.
  --force-download  Force hf to redownload files.
  -h, --help        Show this help.

The target directory comes from SMELT_MODULE_SOURCE_<ID> if set, otherwise source_default.
USAGE
  exit 1
}

plan=${SMELT_MODULE_SOURCE_BUILD_PLAN:-Models/source-build-plan.tsv}
build_all=0
dry_run=0
force_download=0
ids=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan)
      [[ $# -ge 2 ]] || usage
      plan=$2
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
    --dry-run)
      dry_run=1
      shift
      ;;
    --force-download)
      force_download=1
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
if ! command -v hf >/dev/null 2>&1; then
  echo "Missing hf CLI. Install it and authenticate before fetching module sources." >&2
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

# The module descriptor is the authored `<id>.module.json` (JSON), not the
# grammar `.cam` oracle. Decode the `weights` source declaration structurally.
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

module_weights_revision() {
  local module_path=$1
  local revision
  revision=$(python3 - "$module_path" <<'PY'
import json, sys
try:
    doc = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
for src in doc.get("sources", []):
    if src.get("id") == "weights" and src.get("kind") == "hf":
        print(src.get("revision", ""))
        break
PY
)
  printf '%s\n' "${revision:-main}"
}

matched=0
while IFS=$'\t' read -r row_id module_path source_default source_package package_result build_argv rest; do
  [[ -z "${row_id:-}" || "${row_id:0:1}" == "#" ]] && continue
  if [[ -n "${rest:-}" ]]; then
    echo "Invalid module source build plan row for $row_id: too many columns" >&2
    exit 1
  fi
  if [[ -z "${row_id:-}" || -z "${module_path:-}" ||
        -z "${source_default:-}" ]]; then
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

  repo=$(module_weights_locator "$module_path")
  if [[ -z "$repo" ]]; then
    echo "module has no HF source weights declaration for $row_id: $module_path" >&2
    exit 1
  fi
  revision=$(module_weights_revision "$module_path")
  source_env=$(source_env_name "$row_id")
  raw_target=${!source_env:-$source_default}
  target=$(expand_path "$raw_target")
  target_existed=0
  if [[ -e "$target" ]]; then
    target_existed=1
  fi

  echo "module source fetch $row_id -> $target"
  cmd=(hf download "$repo" --revision "$revision" --local-dir "$target")
  if ((dry_run == 1)); then
    cmd+=(--dry-run)
  fi
  if ((force_download == 1)); then
    cmd+=(--force-download)
  fi
  if "${cmd[@]}"; then
    fetch_status=0
  else
    fetch_status=$?
  fi
  if ((dry_run == 1 && target_existed == 0 && fetch_status == 0)) && [[ -e "$target" ]]; then
    rm -rf "$target"
  fi
  if ((fetch_status != 0)); then
    exit "$fetch_status"
  fi
done < "$plan"

if [[ "$matched" == 0 ]]; then
  echo "No module source build plan rows matched." >&2
  exit 1
fi
