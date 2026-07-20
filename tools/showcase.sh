#!/usr/bin/env bash
# Guided tour of what a .smeltpkg can do, end to end, on real models:
#
#   1. Cold start          — exec → first token, no daemon, no warmup
#   2. Create an agent     — system prompt + JSON schema sealed into a package
#   3. Pipe it             — guaranteed-valid JSON straight into jq
#   4. Keep it warm        — --linger reuses a worker over a Unix socket
#   5. Share the weights   — `smelt cas` dedups N agents to one weights.bin
#   6. Install it          — publish to a registry, install by name anywhere
#
# Everything runs on throwaway copies under a temp dir; the canonical
# package and your real CAS store are never touched.
#
# Usage: bash tools/showcase.sh [package.smeltpkg] [--fast]
#   --fast    skip the press-Enter pauses (also implied when stdin
#             isn't a TTY)

set -euo pipefail
cd "$(dirname "$0")/.."

PKG="artifacts/qwen35-0.8b-qmm16x128/Qwen_Qwen3.5-0.8B.smeltpkg"
FAST=0
for arg in "$@"; do
    case "$arg" in
        --fast) FAST=1 ;;
        *) PKG="$arg" ;;
    esac
done
[ -t 0 ] || FAST=1

if [ ! -d "$PKG" ]; then
    echo "No package at $PKG"
    echo "Build one first:  bash tools/build-qwen35-0.8b.sh"
    exit 1
fi

echo "Building smelt (release)..."
swift build -c release >/dev/null
AGENT="$(swift build -c release --show-bin-path)/smelt"

DEMO="$(mktemp -d /tmp/smelt-showcase.XXXXXX)"
trap 'rm -rf "$DEMO"' EXIT
export SMELT_CAS_DIR="$DEMO/store"   # sandbox the store for the demo

bold=$(tput bold 2>/dev/null || true)
dim=$(tput dim 2>/dev/null || true)
reset=$(tput sgr0 2>/dev/null || true)

step() {
    echo
    echo "${bold}── $1 ──${reset}"
    echo "${dim}$2${reset}"
    echo
}

run() {
    echo "${dim}\$ $*${reset}"
    "$@"
}

pause() {
    [ "$FAST" = "1" ] && return
    echo
    read -r -p "${dim}[Enter]${reset} " _
}

step "1/6 · Cold start" \
    "One process, no daemon. exec → first token in ~100ms on an M2 Max."
run env SMELT_CAS=0 "$AGENT" run "$PKG" "Say something nice about UNIX pipes." --max-tokens 24
pause

step "2/6 · Create an agent" \
    "Seal a persona (prefilled to KV state) and a JSON schema (compiled
to an llguidance token trie) into a copy of the package. The output
contract now ships inside the artifact."
run "$AGENT" create triage --model "$PKG" --output "$DEMO/triage.smeltpkg" \
    --system-file demo/triage-smelt.txt \
    --json-schema demo/triage-schema.json
pause

step "3/6 · Pipe it" \
    "The schema is in the package — jq cannot be handed invalid JSON.
Constrained decode runs at ~97% of unconstrained speed."
echo '2026-06-10T15:02:11Z ERROR checkout-api: connection pool exhausted (32/32 in use), p99 4.1s, 503s rising' > "$DEMO/error.log"
echo "${dim}\$ tail -1 error.log | smelt run triage.smeltpkg --system-file demo/triage-smelt.txt | jq .${reset}"
tail -1 "$DEMO/error.log" \
    | env SMELT_CAS=0 "$AGENT" run "$DEMO/triage.smeltpkg" \
        --system-file demo/triage-smelt.txt --max-tokens 64 2>/dev/null \
    | jq .
pause

step "4/6 · Keep it warm" \
    "--linger leaves a worker behind for N idle seconds (it self-destructs;
there is no daemon to manage). Repeat invocations skip package load."
run env SMELT_CAS=0 "$AGENT" run "$DEMO/triage.smeltpkg" \
    --system-file demo/triage-smelt.txt \
    --prompt "WARN retry 3/5 for s3 upload, backing off" \
    --max-tokens 48 --linger 20
echo
run env SMELT_CAS=0 "$AGENT" run "$DEMO/triage.smeltpkg" \
    --system-file demo/triage-smelt.txt \
    --prompt "DEBUG cache miss for key user:1842" \
    --max-tokens 48 --linger 20
pause

step "5/6 · Share the weights" \
    "Two agents from one base model. 'smelt cas adopt' moves identical
artifacts into a content-addressed store and leaves symlinks — N
packages, one weights.bin on disk (and one set of pages in RAM)."
run "$AGENT" create scribe --model "$PKG" --output "$DEMO/scribe.smeltpkg"
echo "Before: $(du -sh "$DEMO" | cut -f1) for two packages"
run "$AGENT" cas adopt "$DEMO/triage.smeltpkg" --quiet
run "$AGENT" cas adopt "$DEMO/scribe.smeltpkg"
echo
echo "After:  $(du -sh "$DEMO" | cut -f1) — two packages for the price of one"
pause

step "6/6 · Install it" \
    "A registry is just a hosted directory (a GitHub repo works as-is).
The thin agent is tens of KB; blobs the local store already has are
never downloaded — the second agent on a base model installs in <100ms."
export SMELT_AGENTS_DIR="$DEMO/agents"
mkdir -p "$DEMO/registry"
run "$AGENT" publish "$DEMO/triage.smeltpkg" --name triage --version 0.1.0 \
    --description "log triage: severity/component/summary" \
    --registry "$DEMO/registry"
echo
run "$AGENT" install triage --registry "$DEMO/registry"
echo
echo "${dim}\$ tail -1 error.log | smelt run triage --system-file demo/triage-smelt.txt | jq .severity${reset}"
tail -1 "$DEMO/error.log" \
    | "$AGENT" run triage \
        --system-file demo/triage-smelt.txt --max-tokens 64 2>/dev/null \
    | jq .severity

echo
echo "${bold}That's the knife.${reset} smelt create · run · cas · publish · install"
echo "${dim}Docs: README.md · vision: docs/VISION.md · release contract: docs/supported-runtime.md${reset}"
