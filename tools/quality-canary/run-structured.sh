#!/usr/bin/env bash
# Structured-task quality canary: serve a package, run the JSON-extraction /
# tool-call corpus, score, tear down. Wrap the invocation in flock on a shared
# machine. Artifacts land in bench-results/structured-<timestamp>/.
#
# Usage: tools/quality-canary/run-structured.sh <package.smeltpkg>
# Env: PORT (default 8791), BIN (default .build/release/smelt)
set -uo pipefail
cd "$(dirname "$0")/../.."

PACKAGE=${1:?usage: run-structured.sh <package.smeltpkg>}
PORT=${PORT:-8791}
BIN=${BIN:-.build/release/smelt}
CORPUS=tools/quality-canary/structured-corpus.jsonl

[ -d "$PACKAGE" ] || { echo "package not found: $PACKAGE" >&2; exit 1; }
[ -x "$BIN" ] || { echo "build smelt first: swift build -c release" >&2; exit 1; }
[ -f "$CORPUS" ] || { echo "corpus not found: $CORPUS" >&2; exit 1; }

TS=$(date +%Y%m%d-%H%M%S)
RUN_DIR="bench-results/structured-$TS"
mkdir -p "$RUN_DIR"
echo "[canary] package: $PACKAGE" >&2
echo "[canary] uptime: $(uptime)" >&2

if PIDS=$(lsof -ti :"$PORT" 2>/dev/null) && [ -n "$PIDS" ]; then
  echo "[canary] port $PORT busy (pids $PIDS); pick a free PORT" >&2; exit 1
fi

"$BIN" serve "$PACKAGE" --transport http --port "$PORT" > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; true' EXIT

echo "[canary] waiting for serve on :$PORT (pid $SERVER_PID)..." >&2
MODEL_ID=""
for _ in $(seq 1 60); do
  if MODEL_ID=$(curl -sS "http://localhost:$PORT/v1/models" 2>/dev/null \
      | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null); then
    [ -n "$MODEL_ID" ] && break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[canary] serve died; see $RUN_DIR/server.log" >&2; exit 1
  fi
  sleep 1
done
[ -n "$MODEL_ID" ] || { echo "[canary] serve never became reachable; see $RUN_DIR/server.log" >&2; exit 1; }
echo "[canary] serve ready: model=$MODEL_ID" >&2

python3 tools/quality-canary/run-structured.py \
  --base-url "http://localhost:$PORT/v1" \
  --model "$MODEL_ID" \
  --corpus "$CORPUS" \
  --out "$RUN_DIR/structured.json"
RC=$?

echo "[canary] artifact: $RUN_DIR/structured.json" >&2
echo "$RUN_DIR" > "$RUN_DIR/.rundir"
exit $RC
