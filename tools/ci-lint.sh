#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

repo_search() {
  if command -v rg >/dev/null 2>&1; then
    rg "$@"
  else
    grep -Er "$@"
  fi
}

if repo_search -n 'import (InstantAgent|Agent[A-Z][A-Za-z0-9_]*)' Package.swift Sources; then
  echo "Smelt must not import an Instant Agent product target" >&2
  exit 1
fi

if find Sources -mindepth 1 -maxdepth 1 -type d -name 'Agent*' -print -quit | grep -q .; then
  echo "Smelt source targets must use the Smelt namespace" >&2
  exit 1
fi

if repo_search -n '^[[:space:]]*import SmeltAgent\b' \
  Sources/SmeltSchema Sources/SmeltCompiler Sources/SmeltRuntime \
  Sources/SmeltServe Sources/SmeltLab Sources/SmeltModels \
  Sources/SmeltModuleAuthoring; then
  echo "lower-level Smelt targets must not import SmeltAgent" >&2
  exit 1
fi

if repo_search -n '\.library\(name: "SmeltAgent"' Package.swift; then
  echo "SmeltAgent is an internal target, not a public Swift product" >&2
  exit 1
fi

identity_paths=(
  Sources/SmeltAgent Sources/SmeltCLI/Commands/Agent.swift
  Sources/SmeltCLI/Helpers/AgentPiLauncher.swift Tests/SmeltAgentTests
  Sources/SmeltCLI/Resources/pi-smelt-agent README.md SECURITY.md
  CODE_OF_CONDUCT.md CONTRIBUTING.md CLAUDE.md docs/VISION.md
  docs/supported-runtime.md docs/testing.md docs/PERF_STATUS.md
  docs/ASSUMPTIONS.md docs/model-bringup-framework.md
  docs/r2-host-loop-protocol.md Models/evidence/inkling-cost/README.md
)
existing_identity_paths=()
for path in "${identity_paths[@]}"; do
  [[ -e "$path" ]] && existing_identity_paths+=("$path")
done
if repo_search -n '\bInstantAgent\b|Instant Agent|\bIA_[A-Z0-9_]+\b|\bia (create|run|install|publish|build|serve|module|lab|verify|bench|prefill-bench)\b' \
  "${existing_identity_paths[@]}"; then
  echo "retired Instant Agent identity remains in a live product surface" >&2
  exit 1
fi

for retired in integrations/pi-instant-agent integrations/pi-smelt-agent \
  Formula/instant-agent.rb site; do
  if [[ -e "$retired" ]]; then
    echo "Instant Agent product residue remains in Smelt: $retired" >&2
    exit 1
  fi
done

pi_resource=Sources/SmeltCLI/Resources/pi-smelt-agent
for required in README.md author.ts coding-tools.json index.ts package.json; do
  if [[ ! -f "$pi_resource/$required" ]]; then
    echo "SmeltCLI Pi resource is missing $required" >&2
    exit 1
  fi
done
if ! repo_search -q 'resources: \[\.copy\("Resources/pi-smelt-agent"\)\]' Package.swift; then
  echo "SmeltCLI must bundle the Pi integration resource" >&2
  exit 1
fi
if repo_search -n 'return "smelt"|\|\| "smelt"|SMELT_AGENT_PI_HOME|\.build.*release.*smelt' \
  "$pi_resource"/*.ts; then
  echo "Pi integration must use the exact launcher-supplied Smelt executable" >&2
  exit 1
fi

python3 - <<'PY'
import re
from pathlib import Path

main = Path("Sources/SmeltCLI/main.swift").read_text()
commands = set()
for match in re.finditer(r'^case\s+([^:]+):', main, re.MULTILINE):
    commands.update(re.findall(r'"([^"]+)"', match.group(1)))

expected = {
    "help", "--help", "-h", "run", "build", "module", "lab",
    "linger-worker", "serve", "agent", "cas",
}
if commands != expected:
    missing = sorted(expected - commands)
    extra = sorted(commands - expected)
    raise SystemExit(
        f"top-level smelt command drift: missing={missing} extra={extra}"
    )

retired_paths = [
    "Sources/SmeltCLI/Commands/PreparedPrompt.swift",
    "Sources/SmeltCLI/Commands/Qwen35VisionBuild.swift",
    "Sources/SmeltCompiler/SmeltQwen35VisionArtifactBuilder.swift",
    "Sources/SmeltRuntime/SmeltBakedPrefix.swift",
    "Sources/SmeltSchema/SmeltBakeManifest.swift",
    "tools/showcase.sh",
]
remaining = [path for path in retired_paths if Path(path).exists()]
if remaining:
    raise SystemExit(f"retired CLI surface returned: {remaining}")

source_text = "\n".join(
    path.read_text()
    for path in Path("Sources").rglob("*.swift")
)
retired_bake_contracts = [
    "SmeltBakeManifest", "SmeltBakedPrefix", '"baked.json"',
    '"bake_manifest"', '"baked-inline"', '"baked-sidecar"',
]
stale_bake_contracts = [
    spelling for spelling in retired_bake_contracts if spelling in source_text
]
if stale_bake_contracts:
    raise SystemExit(
        f"retired bake product contract returned: {stale_bake_contracts}"
    )

lab = Path("Sources/SmeltCLI/Commands/Lab.swift").read_text()
qmm = Path("Sources/SmeltCLI/Commands/QMMSweep.swift").read_text()
if re.search(r"\bargs\s*=", lab + "\n" + qmm):
    raise SystemExit("smelt lab forwarding must not mutate process-global argv")
if "_ body: ([String]) -> Void" not in lab:
    raise SystemExit("smelt lab forwarding must pass argv explicitly to leaf commands")

lab_tool = Path("Sources/SmeltLab/SmeltLabTool.swift")
vision_tool = Path("Sources/SmeltLab/SmeltLabQwen35Vision.swift")
if not vision_tool.is_file() or len(lab_tool.read_text().splitlines()) > 2_800:
    raise SystemExit("oversized SmeltLab probe implementation was recombined")

active_plan_path = Path("docs/smelt-agent-integration-plan.md")
retired_spellings = [
    "smelt-probe", "`smelt verify`", "`smelt bench`",
    "`smelt prefill-bench`", "`smelt trace`", "`smelt replay`",
    "`smelt profile`", "`smelt optimizer-report`", "`smelt kernels`",
    "`smelt dispatches`", "`smelt kernel-lab`",
]
if active_plan_path.is_file():
    active_plan = active_plan_path.read_text()
    stale = [spelling for spelling in retired_spellings if spelling in active_plan]
    if stale:
        raise SystemExit(f"active split plan advertises retired CLI surface: {stale}")

print("cli surface: canonical")
PY

enumerate_repository_files() {
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git ls-files --cached --others --exclude-standard -z
  else
    find . \
      \( -path './.git' -o -path './.build' -o -path './.swiftpm' \) -prune \
      -o -type f -print0
  fi
}

shell_count=0
while IFS= read -r -d '' file; do
  [[ -f "$file" ]] || continue
  first_line=$(head -n 1 "$file" || true)
  case "$file:$first_line" in
    *.sh:*|*:*bash*|*:*'/sh'*)
      bash -n "$file"
      ((shell_count += 1))
      ;;
  esac
done < <(enumerate_repository_files)

python3 - <<'PY'
import ast
import json
from pathlib import Path
import subprocess

try:
    raw_paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        stderr=subprocess.DEVNULL,
    ).split(b"\0")
    paths = [Path(raw.decode("utf-8")) for raw in raw_paths if raw]
except (subprocess.CalledProcessError, FileNotFoundError):
    ignored_roots = {".git", ".build", ".swiftpm"}
    paths = [
        path
        for path in Path(".").rglob("*")
        if path.is_file() and not ignored_roots.intersection(path.parts)
    ]

python_count = 0
json_count = 0
for path in paths:
    if not path.is_file():
        continue
    data = path.read_bytes()
    is_python = path.suffix == ".py" or data.startswith(b"#!/usr/bin/env python")
    if is_python:
        ast.parse(data.decode("utf-8-sig"), filename=str(path))
        python_count += 1
    if path.suffix == ".json":
        json.loads(data)
        json_count += 1

print(f"python syntax: {python_count} files")
print(f"json syntax: {json_count} files")
PY

echo "shell syntax: ${shell_count} files"
echo "CI lint passed"
