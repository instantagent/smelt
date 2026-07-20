#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if rg -n 'import (InstantAgent|Agent[A-Z][A-Za-z0-9_]*)' Package.swift Sources; then
  echo "Smelt must not import an Instant Agent product target" >&2
  exit 1
fi

if find Sources -mindepth 1 -maxdepth 1 -type d -name 'Agent*' -print -quit | grep -q .; then
  echo "Smelt source targets must use the Smelt namespace" >&2
  exit 1
fi

for retired in integrations/pi-instant-agent Formula/instant-agent.rb site; do
  if [[ -e "$retired" ]]; then
    echo "Instant Agent product residue remains in Smelt: $retired" >&2
    exit 1
  fi
done

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
done < <(git ls-files --cached --others --exclude-standard -z)

python3 - <<'PY'
import ast
import json
from pathlib import Path
import subprocess

paths = [
    Path(raw.decode("utf-8"))
    for raw in subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    ).split(b"\0")
    if raw
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
