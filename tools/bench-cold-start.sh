#!/usr/bin/env bash
# =============================================================================
# bench-cold-start.sh — the cold-start receipt harness (L1 item 2)
#
# Measures process-exec -> first-generated-token for `smelt` vs the field
# (Ollama, llama.cpp, mlx_lm) on the SAME model (Qwen3.5-0.8B) at the 4-bit tier,
# plus TTS time-to-first-audio (TTFA) for the Qwen3-TTS package.
# See docs/receipts/cold-start-methodology.md for the full write-up.
#
# Two wall-clock metrics per trial (tools/coldstart_timer.py, monotonic clock):
#   ttft_ms = spawn -> first byte of generated output on stdout (child pinned to
#             generate exactly ONE token; stderr carries all log/preamble noise).
#   exit_ms = spawn -> process exit (ttft + teardown). Robust cross-check.
#
# "Cold": fresh process; page cache evicted with `vmtouch -e` on the model's real
# files, then residency VERIFIED with a `vmtouch` query (no -e): exact
# resident/total PAGE COUNTS recorded per trial to the raw (`# resid` lines);
# zero resident pages is REQUIRED — on a dirty check the eviction is retried once
# and the cell then FAILS the run (fail-loud, never proceed-on-dirty). No
# passwordless sudo on this box, so no global `purge` — vmtouch targets exactly
# the model bytes without root. Ollama has a persistent server that `ollama run`
# does not self-start, so Ollama-cold = server up, model unloaded (`ollama stop`,
# then polled via `ollama ps` until gone — recorded per trial; ps failure or
# timeout FAILS the cell) + its manifest-referenced blobs evicted (its genuine
# cold-model-load latency; the always-resident daemon's startup is not in the
# number). The Metal pipeline/PSO
# cache is NOT controlled (no non-root purge) and warms after first compile.
# "Warm" = identical run immediately repeated, no eviction. "Linger" = smelt's
# warm-worker path.
#
# Shared machine: every cell holds the bench lock for its whole trial batch; the
# load average is logged to the raw artifact before each batch.
#
# Reproduce (one command):  tools/bench-cold-start.sh
# =============================================================================
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMER="$HERE/tools/coldstart_timer.py"
LOCK="${BENCH_LOCK:-/private/tmp/claude-501/-Users-jud-Projects-smelt/bench.lock}"
N="${N:-12}"
PROMPT="${PROMPT:-Explain gravity in one short sentence.}"
OUTDIR="${OUTDIR:-$HERE/docs/receipts/raw}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RAW_TSV="${RAW_TSV:-$OUTDIR/cold-start-raw-$STAMP.tsv}"
RUN_LOG="${RUN_LOG:-$OUTDIR/cold-start-run-$STAMP.log}"
WORK="$(mktemp -d)"

SMELT="${SMELT:-$HERE/.build/release/smelt}"
SMELT_PKG="${SMELT_PKG:-$HERE/bench-packages/Qwen_Qwen3.5-0.8B.smeltpkg}"
TTS_PKG="${TTS_PKG:-$HERE/bench-packages/qwen3-tts.smeltpkg}"
GGUF="${GGUF:-/tmp/gguf-models/Qwen3.5-0.8B-Q4_0-ggmlorg.gguf}"   # ggml-org Q4_0 (loads in llama.cpp 8680; bartowski Q4_K_M is missing an SSM tensor)
MLX_MODEL="${MLX_MODEL:-mlx-community/Qwen3.5-0.8B-4bit}"
MLX_PY="${MLX_PY:-/tmp/mlx-venv/bin/python}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen35-08b-q4}"                     # created from the same ggml-org GGUF via Modelfile
LLAMA_COMPLETION="${LLAMA_COMPLETION:-llama-completion}"

CELLS="${CELLS:-smelt_cold smelt_warm smelt_linger llamacpp_cold llamacpp_warm ollama_cold ollama_warm mlx_cold mlx_warm tts_cold tts_warm tts_linger trace}"

mkdir -p "$OUTDIR"
printf "cell\ttrial\tttft_ms\texit_ms\trc\tstdout_bytes\tnote\n" > "$RAW_TSV"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN_LOG"; }
loadavg(){ uptime | sed 's/.*load average[s]*: //'; }
pkg_real_files(){ find -L "$1" -type f 2>/dev/null | while IFS= read -r f; do
  python3 -c "import os,sys;print(os.path.realpath(sys.argv[1]))" "$f"; done | sort -u; }

record(){ # cell trial resultline note ; nonzero on an invalid row (backstop)
  local cell="$1" trial="$2" line="$3" note="$4" ttft exit_ms rc bytes first
  ttft=$(sed -n 's/.*ttft_ms=\([0-9.na]*\).*/\1/p' <<<"$line")
  exit_ms=$(sed -n 's/.*exit_ms=\([0-9.]*\).*/\1/p' <<<"$line")
  rc=$(sed -n 's/.*rc=\([0-9-]*\).*/\1/p' <<<"$line")
  bytes=$(sed -n 's/.*stdout_bytes=\([0-9]*\).*/\1/p' <<<"$line")
  first=$(sed -n 's/.*first=\(.*\)$/\1/p' <<<"$line")   # repr-escaped stamped-byte preview (kept as evidence)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$cell" "$trial" "${ttft:-nan}" "${exit_ms:-nan}" "${rc:-?}" "${bytes:-0}" "$note; first=${first:-?}" >> "$RAW_TSV"
  if [ "${rc:-1}" != "0" ] || [ "${ttft:-nan}" = "nan" ] || [ -z "$ttft" ] || [ -z "$exit_ms" ] || [ "${bytes:-0}" = "0" ]; then
    warn "FATAL $cell #$trial: invalid row (rc=${rc:-?} ttft=${ttft:-nan} exit=${exit_ms:-nan} bytes=${bytes:-0})"
    return 9
  fi
}
emit(){ # cell note < timer-lines ; assigns sequential trial numbers; exits on invalid row
  local cell="$1" note="$2" T=0 line
  while IFS= read -r line; do [ -z "$line" ] && continue; T=$((T+1)); record "$cell" "$T" "$line" "$note" || exit 9; echo "  $cell #$T: $line"; done
}
batch_header(){ log "CELL $1  load: $(loadavg)"; echo "# $1 load: $(loadavg)" >> "$RAW_TSV"; }

# ---- eviction verification + ollama unload poll (fail-loud) ---------------
warn(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN_LOG" >&2; }   # -> run log + stderr, NOT the emit pipe
cellfail(){ warn "FATAL: cell $1 failed (cold-state verification or trial execution); aborting run"; exit 1; }
# Evict a newline-separated file list from the page cache (vmtouch -e), then
# VERIFY with a query (no -e): parse the exact resident/total PAGE COUNTS from
# vmtouch's "Resident Pages: N/M" line and require N == 0 (an integer percent
# can read 0% with pages still resident). On query failure or N > 0, retry the
# eviction once; if still dirty, record it and return nonzero so the caller
# FAILS the cell. Every check is recorded to the raw as
# `# resid <cell> #<trial>: resident_pages=N/M` (ignored by the summary parser).
evict_verify(){ # cell trial <newline-separated file list>
  local cell="$1" trial="$2" files="$3" attempt=1 counts="" f
  while IFS= read -r f; do   # every target must EXIST (vmtouch reports 0/0 for a
    [ -z "$f" ] && continue  # missing file, which would fake a clean check)
    [ -f "$f" ] || { printf '# resid %s #%s: resident_pages=missing-file:%s\n' "$cell" "$trial" "$f" >> "$RAW_TSV"
                     warn "FATAL $cell #$trial: eviction target missing: $f"; return 1; }
  done <<<"$files"
  for attempt in 1 2; do
    printf '%s\n' "$files" | while IFS= read -r f; do [ -n "$f" ] && vmtouch -e "$f" >/dev/null 2>&1; done
    counts=$(printf '%s\n' "$files" | grep -v '^[[:space:]]*$' | tr '\n' '\0' \
             | xargs -0 vmtouch 2>/dev/null | awk '/Resident Pages:/{print $3}' | tail -1)
    case "$counts" in 0/0|"") : ;; 0/*) break ;; esac   # clean = N==0 AND M>0
  done
  printf '# resid %s #%s: resident_pages=%s attempts=%s\n' "$cell" "$trial" "${counts:-query-failed}" "$attempt" >> "$RAW_TSV"
  case "$counts" in
    0/0|"") warn "FATAL $cell #$trial: residency query covered no pages (counts=${counts:-query-failed})"; return 1 ;;
    0/*)  : ;;
    *)    warn "FATAL $cell #$trial: eviction not verified (resident_pages=$counts) after retry"; return 1 ;;
  esac
}
# Resolve the SELECTED Ollama model's blob files (config + layer digests) from
# its manifest, so eviction targets exactly the model bytes. Nonzero on any
# resolution failure (missing manifest/blob, parse error) — caller decides.
ollama_model_blobs(){ # model -> newline-separated blob paths on stdout
  local model="$1" name tag mf out p
  name="${model%%:*}"; tag="latest"; case "$model" in *:*) tag="${model##*:}";; esac
  mf=$(find "$HOME/.ollama/models/manifests" -type f -path "*/$name/$tag" 2>/dev/null | head -1)
  [ -n "$mf" ] || { warn "ollama manifest not found for $model"; return 1; }
  out=$(python3 -c 'import json,sys,os
m=json.load(open(sys.argv[1]))
digs=[m["config"]["digest"]]+[l["digest"] for l in m["layers"]]
print("\n".join(os.path.join(sys.argv[2],d.replace(":","-")) for d in digs))' \
       "$mf" "$HOME/.ollama/models/blobs" 2>/dev/null) || { warn "ollama manifest parse failed: $mf"; return 1; }
  printf '%s\n' "$out" | while IFS= read -r p; do
    [ -f "$p" ] || { warn "ollama blob missing: $p"; exit 1; }
  done || return 1
  printf '%s\n' "$out"
}
# After `ollama stop`, poll `ollama ps` until MODEL is no longer listed (~10s
# timeout). An `ollama ps` FAILURE is distinguished from a genuinely empty
# list; on ps failure or timeout, returns nonzero so the caller FAILS the cell.
# Success is recorded to the raw as `# unload <cell> #<trial>: ok`.
ollama_wait_unloaded(){ # model cell trial
  local model="$1" cell="$2" trial="$3" t=0 out
  while :; do
    if ! out=$(ollama ps 2>&1); then
      warn "FATAL $cell #$trial: 'ollama ps' failed: $out"; return 1
    fi
    printf '%s\n' "$out" | tail -n +2 | grep -q "$model" || break
    t=$((t+1))
    if [ "$t" -ge 20 ]; then
      warn "FATAL $cell #$trial: '$model' still resident after 'ollama stop' (~10s timeout)"; return 1
    fi
    sleep 0.5
  done
  printf '# unload %s #%s: ok\n' "$cell" "$trial" >> "$RAW_TSV"
}
export RAW_TSV RUN_LOG
export -f warn evict_verify ollama_wait_unloaded

# ---- smelt text (cold/warm) --------------------------------------------------
run_smelt(){ local mode="$1" cell="ia_$1"; local files; files="$(pkg_real_files "$SMELT_PKG")"
  batch_header "$cell"
  flock -o "$LOCK" bash -c '
    SMELT="$1";PKG="$2";PROMPT="$3";TIMER="$4";N="$5";MODE="$6";WORK="$7";FILES="$8";CELL="$9"
    for i in $(seq 1 "$N"); do
      if [ "$MODE" = cold ]; then evict_verify "$CELL" "$i" "$FILES" || exit 97; fi
      python3 "$TIMER" --out "$WORK/o" --err "$WORK/e" -- "$SMELT" run "$PKG" "$PROMPT" --max-tokens 1 --temp 0 || exit 97
    done' _ "$SMELT" "$SMELT_PKG" "$PROMPT" "$TIMER" "$N" "$mode" "$WORK" "$files" "$cell" | emit "$cell" "$mode"
  [ "${PIPESTATUS[0]}:${PIPESTATUS[1]}" = "0:0" ] || cellfail "$cell"
}
kill_lingerworkers(){ pkill -f "$HERE/.build/release/smelt linger-worker" 2>/dev/null; sleep 1; }
run_smelt_linger(){ local cell="smelt_linger"; batch_header "$cell"
  flock -o "$LOCK" bash -c '
    SMELT="$1";PKG="$2";PROMPT="$3";TIMER="$4";N="$5";WORK="$6"
    "$SMELT" run "$PKG" "$PROMPT" --max-tokens 1 --temp 0 --linger 30 >/dev/null 2>&1; sleep 1
    for i in $(seq 1 "$N"); do python3 "$TIMER" --out "$WORK/o" --err "$WORK/e" -- "$SMELT" run "$PKG" "$PROMPT" --max-tokens 1 --temp 0 --linger 30 || exit 97; done
    ' _ "$SMELT" "$SMELT_PKG" "$PROMPT" "$TIMER" "$N" "$WORK" | emit "$cell" "linger"
  [ "${PIPESTATUS[0]}:${PIPESTATUS[1]}" = "0:0" ] || cellfail "$cell"
  kill_lingerworkers
}
# ---- llama.cpp ------------------------------------------------------------
run_llamacpp(){ local mode="$1" cell="llamacpp_$1"; batch_header "$cell"
  flock -o "$LOCK" bash -c '
    LC="$1";GGUF="$2";PROMPT="$3";TIMER="$4";N="$5";MODE="$6";WORK="$7";CELL="$8"
    for i in $(seq 1 "$N"); do
      if [ "$MODE" = cold ]; then evict_verify "$CELL" "$i" "$GGUF" || exit 97; fi
      python3 "$TIMER" --out "$WORK/o" --err "$WORK/e" -- "$LC" -m "$GGUF" -p "$PROMPT" -n 1 --temp 0 --no-display-prompt --no-warmup </dev/null || exit 97
    done' _ "$LLAMA_COMPLETION" "$GGUF" "$PROMPT" "$TIMER" "$N" "$mode" "$WORK" "$cell" | emit "$cell" "$mode"
  [ "${PIPESTATUS[0]}:${PIPESTATUS[1]}" = "0:0" ] || cellfail "$cell"
}
# ---- ollama ---------------------------------------------------------------
ollama_ensure_server(){ pgrep -f "ollama serve" >/dev/null || { OLLAMA_FLASH_ATTENTION=1 nohup ollama serve >/tmp/ollama-serve.log 2>&1 & sleep 2; }; }
run_ollama(){ local mode="$1" cell="ollama_$1"; ollama_ensure_server; batch_header "$cell"
  # Eviction set: the model's manifest-referenced blobs (exact). If manifest
  # resolution fails, fall back to ALL local blobs — a SUPERSET (still a valid
  # cold state; the doc discloses the fallback) — recording which set was used.
  local blobs evset="manifest"
  blobs="$(ollama_model_blobs "$OLLAMA_MODEL")" || {
    evset="all-blobs-superset"; blobs="$(ls "$HOME"/.ollama/models/blobs/sha256-* 2>/dev/null)"
    warn "$cell: manifest resolution failed; falling back to all-blobs SUPERSET eviction"; }
  [ "$mode" = cold ] && printf '# evictset %s: %s (%s files)\n' "$cell" "$evset" "$(printf '%s\n' "$blobs" | grep -c .)" >> "$RAW_TSV"
  flock -o "$LOCK" bash -c '
    MODEL="$1";PROMPT="$2";TIMER="$3";N="$4";MODE="$5";WORK="$6";CELL="$7";BLOBS="$8"
    if [ "$MODE" = warm ]; then ollama run "$MODEL" "$PROMPT" >/dev/null 2>&1; fi   # preload once
    for i in $(seq 1 "$N"); do
      if [ "$MODE" = cold ]; then
        ollama stop "$MODEL" >/dev/null 2>&1                     # unload from server RAM
        ollama_wait_unloaded "$MODEL" "$CELL" "$i" || exit 97    # verified unload, recorded; fail-loud
        evict_verify "$CELL" "$i" "$BLOBS" || exit 97            # evict + verify page counts; fail-loud
      fi
      python3 "$TIMER" --out "$WORK/o" --err "$WORK/e" -- ollama run "$MODEL" "$PROMPT" || exit 97
    done' _ "$OLLAMA_MODEL" "$PROMPT" "$TIMER" "$N" "$mode" "$WORK" "$cell" "$blobs" | emit "$cell" "$mode"
  [ "${PIPESTATUS[0]}:${PIPESTATUS[1]}" = "0:0" ] || cellfail "$cell"
}
# ---- mlx_lm ---------------------------------------------------------------
run_mlx(){ local mode="$1" cell="mlx_$1"
  local mdir; mdir="$($MLX_PY -c "from huggingface_hub import snapshot_download;print(snapshot_download('$MLX_MODEL'))" 2>/dev/null)"
  local files; files="$(pkg_real_files "$mdir")"; batch_header "$cell"
  # Marker includes the banner's trailing newline so the stamped byte is the
  # first TOKEN byte (the banner is block-buffered into the same first pipe
  # chunk as the token; evidence: docs/receipts/raw/mlx-stdout-probe.txt).
  local marker=$'==========\n'
  flock -o "$LOCK" bash -c '
    PY="$1";MODEL="$2";PROMPT="$3";TIMER="$4";N="$5";MODE="$6";WORK="$7";FILES="$8";CELL="$9";MARKER="${10}"
    for i in $(seq 1 "$N"); do
      if [ "$MODE" = cold ]; then evict_verify "$CELL" "$i" "$FILES" || exit 97; fi
      KMP_DUPLICATE_LIB_OK=TRUE python3 "$TIMER" --out "$WORK/o" --err "$WORK/e" --start-after "$MARKER" -- \
        "$PY" -m mlx_lm generate --model "$MODEL" --prompt "$PROMPT" --max-tokens 1 --temp 0 || exit 97
    done' _ "$MLX_PY" "$MLX_MODEL" "$PROMPT" "$TIMER" "$N" "$mode" "$WORK" "$files" "$cell" "$marker" | emit "$cell" "$mode"
  [ "${PIPESTATUS[0]}:${PIPESTATUS[1]}" = "0:0" ] || cellfail "$cell"
}
# ---- TTS (TTFA) -----------------------------------------------------------
run_tts(){ local mode="$1" cell="tts_$1"; local files; files="$(pkg_real_files "$TTS_PKG")"
  local lf=""; [ "$mode" = linger ] && lf="--linger 30"; batch_header "$cell"
  flock -o "$LOCK" bash -c '
    SMELT="$1";PKG="$2";TIMER="$3";N="$4";MODE="$5";WORK="$6";FILES="$7";LF="$8";CELL="$9"
    [ "$MODE" = linger ] && { "$SMELT" run "$PKG" "warm up" --max-frames 4 --greedy $LF >/dev/null 2>&1; sleep 1; }
    for i in $(seq 1 "$N"); do
      if [ "$MODE" = cold ]; then evict_verify "$CELL" "$i" "$FILES" || exit 97; fi
      # timer ttft = first WAV byte on stdout (exec->first-audio byte); exit_ms =
      # exec->exit. The WAV is piped into the timer, previewed to --out; the smelt
      # internal load->first-audio stamp lands on --err.
      python3 "$TIMER" --out "$WORK/o-$MODE-$i" --err "$WORK/tts-$MODE-$i" -- \
        "$SMELT" run "$PKG" "Hello, this is a first audio latency test." --max-frames 1 --greedy $LF > "$WORK/tline-$MODE-$i" 2>/dev/null
    done' _ "$SMELT" "$TTS_PKG" "$TIMER" "$N" "$mode" "$WORK" "$files" "$lf" "$cell" || cellfail "$cell"
  local T=0 i ef line ttft exit_ms ttfa rc bytes wav first
  for i in $(seq 1 "$N"); do T=$((T+1)); ef="$WORK/tts-$mode-$i"; line="$(cat "$WORK/tline-$mode-$i" 2>/dev/null)"
    ttft=$(sed -n "s/.*ttft_ms=\([0-9.na]*\).*/\1/p" <<<"$line")
    exit_ms=$(sed -n "s/.*exit_ms=\([0-9.]*\).*/\1/p" <<<"$line")
    rc=$(sed -n "s/.*rc=\([0-9-]*\).*/\1/p" <<<"$line")
    bytes=$(sed -n "s/.*stdout_bytes=\([0-9]*\).*/\1/p" <<<"$line")
    case "$line" in *"first='RIFF"*|*'first="RIFF'*) wav=ok;; *) wav=BAD;; esac
    first=$(sed -n "s/.*first=\(.*\)$/\1/p" <<<"$line")
    ttfa=$(sed -n "s/.*first audio \([0-9.]*\)ms.*/\1/p" "$ef" 2>/dev/null | head -1)
    # ttft_ms col = exec->first-audio byte; exit_ms col = exec->exit; rc and
    # stdout_bytes are the child's REAL values parsed from the timer line;
    # wav_magic=ok iff the stdout preview begins with the RIFF WAV magic;
    # first= retains the stamped-byte preview as per-trial evidence.
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$cell" "$T" "${ttft:-nan}" "${exit_ms:-nan}" "${rc:-?}" "${bytes:-0}" "ttft_ms=exec->first-audio-byte; exit_ms=exec->exit; wav_magic=$wav; smelt_load_to_first_audio_ms=${ttfa:-nan}; first=${first:-?}" >> "$RAW_TSV"
    echo "  $cell #$T: first_audio_byte_ms=${ttft:-nan} exec_exit_ms=${exit_ms:-nan} rc=${rc:-?} wav_magic=$wav smelt_load_to_first_audio_ms=${ttfa:-nan}"
    if [ "${rc:-1}" != "0" ] || [ "$wav" != "ok" ] || [ "${ttfa:-nan}" = "nan" ]; then
      warn "FATAL $cell #$T: child rc=${rc:-missing} wav_magic=$wav smelt_stamp=${ttfa:-nan} (timer line: $line)"; cellfail "$cell"
    fi
  done
  [ "$mode" = linger ] && kill_lingerworkers
}
# ---- smelt startup trace (3 cold samples; decomposition evidence) -------------
# Captures SMELT_STARTUP_TRACE stage timings (FULL stderr, incl. the smelt
# `Timing: prefill/generate` line) for 3 page-cache-evicted runs, into the raw
# dir. The methodology doc's decomposition table is rebuilt from this capture.
run_trace(){ local cell="trace"; local files; files="$(pkg_real_files "$SMELT_PKG")"
  local out="$OUTDIR/smelt-startup-trace-$STAMP.txt"; batch_header "$cell"
  flock -o "$LOCK" bash -c '
    SMELT="$1";PKG="$2";PROMPT="$3";FILES="$4";OUT="$5";CELL="$6"
    echo "# smelt SMELT_STARTUP_TRACE, cold (page cache evicted + verified per sample), full stderr, $(date)" > "$OUT"
    for i in 1 2 3; do
      evict_verify "$CELL" "$i" "$FILES" || exit 97
      echo "=== cold sample $i ===" >> "$OUT"
      SMELT_STARTUP_TRACE=1 "$SMELT" run "$PKG" "$PROMPT" --max-tokens 1 --temp 0 2>>"$OUT" >/dev/null || exit 97
    done
    grep -q "^startup:" "$OUT" || exit 97' _ "$SMELT" "$SMELT_PKG" "$PROMPT" "$files" "$out" "$cell" || cellfail "$cell"
  log "startup trace -> $out"
}

# ---- provenance (recorded BEFORE any timed cell) ----------------------------
# Backs the doc's build/version/machine-state claims with the run log itself.
# NOTE: hashing reads warm the page cache; harmless — every cold trial evicts
# and VERIFIES afterwards.
sha(){ shasum -a 256 "$1" 2>/dev/null | awk '{print $1}'; }
realp(){ python3 -c "import os,sys;print(os.path.realpath(sys.argv[1]))" "$1"; }
provenance(){
  local mdir mf name tag
  log "--- provenance ---"
  log "config: CELLS=\"$CELLS\" N=$N PROMPT=\"$PROMPT\""
  log "config: SMELT=$SMELT SMELT_PKG=$SMELT_PKG TTS_PKG=$TTS_PKG GGUF=$GGUF"
  log "config: MLX_MODEL=$MLX_MODEL MLX_PY=$MLX_PY OLLAMA_MODEL=$OLLAMA_MODEL LLAMA_COMPLETION=$LLAMA_COMPLETION"
  log "git HEAD: $(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo n/a) (branch $(git -C "$HERE" branch --show-current 2>/dev/null || echo n/a))"
  local dirty; dirty="$(git -C "$HERE" status --porcelain 2>/dev/null)"
  log "git dirty files: $(printf '%s\n' "$dirty" | grep -c .) (this unit's uncommitted changes are expected; full list follows)"
  printf '%s\n' "$dirty" | tee -a "$RUN_LOG"
  log "harness: $HERE/tools/bench-cold-start.sh  sha256=$(sha "$HERE/tools/bench-cold-start.sh")"
  log "timer: $TIMER  sha256=$(sha "$TIMER")"
  log "methodology doc: $HERE/docs/receipts/cold-start-methodology.md  sha256=$(sha "$HERE/docs/receipts/cold-start-methodology.md")"
  log "uptime: $(uptime)"
  log "process snapshot (top 15 by %cpu):"
  ps -Ao pcpu,comm -r | head -15 | tee -a "$RUN_LOG"
  log "versions: llama.cpp $($LLAMA_COMPLETION --version 2>&1 | grep '^version:' | head -1) | $(ollama --version 2>/dev/null | head -1) | $(vmtouch -h 2>&1 | grep -o 'vmtouch v[0-9.]*' | head -1)"
  log "versions: system $(python3 --version 2>&1) | mlx-venv $("$MLX_PY" --version 2>&1) | mlx_lm $("$MLX_PY" -c 'import mlx_lm;print(mlx_lm.__version__)' 2>/dev/null) | mlx $("$MLX_PY" -c 'import mlx.core as mx;print(mx.__version__)' 2>/dev/null)"
  log "smelt binary: $(realp "$SMELT")  sha256=$(sha "$SMELT")"
  # weight files are selected by SIZE (>50 MB), not extension — smelt's CAS-backed
  # weights blob has no extension (its real path IS a sha256 name)
  bigfiles(){ while IFS= read -r f; do [ -n "$f" ] && [ "$(stat -f%z "$f" 2>/dev/null || echo 0)" -gt 52428800 ] && printf '%s\n' "$f"; done; }
  log "smelt package eviction list ($SMELT_PKG):"; pkg_real_files "$SMELT_PKG" | tee -a "$RUN_LOG"
  pkg_real_files "$SMELT_PKG" | bigfiles | while IFS= read -r f; do log "smelt weights: $f  sha256=$(sha "$f")"; done
  log "tts package eviction list ($TTS_PKG):"; pkg_real_files "$TTS_PKG" | tee -a "$RUN_LOG"
  pkg_real_files "$TTS_PKG" | bigfiles | while IFS= read -r f; do log "tts weights: $f  sha256=$(sha "$f")"; done
  log "gguf: $(realp "$GGUF")  sha256=$(sha "$GGUF")"
  mdir="$($MLX_PY -c "from huggingface_hub import snapshot_download;print(snapshot_download('$MLX_MODEL'))" 2>/dev/null)"
  log "mlx snapshot (revision = path tail): ${mdir:-UNRESOLVED}"
  log "mlx eviction list:"; pkg_real_files "$mdir" | tee -a "$RUN_LOG"
  name="${OLLAMA_MODEL%%:*}"; tag="latest"; case "$OLLAMA_MODEL" in *:*) tag="${OLLAMA_MODEL##*:}";; esac
  mf=$(find "$HOME/.ollama/models/manifests" -type f -path "*/$name/$tag" 2>/dev/null | head -1)
  if [ -n "$mf" ]; then log "ollama manifest ($mf):"; tee -a "$RUN_LOG" < "$mf"; echo | tee -a "$RUN_LOG"
  else log "ollama manifest: NOT FOUND for $OLLAMA_MODEL"; fi
  log "ollama eviction list (manifest-referenced blobs):"
  ollama_model_blobs "$OLLAMA_MODEL" | tee -a "$RUN_LOG" || log "(resolution failed; ollama_cold will fall back to the all-blobs superset)"
  # full per-file sha256 manifests of both package trees (pins manifest/routing
  # files, not just the weights) -> sibling raw file
  local manifests="$OUTDIR/package-manifests-$STAMP.txt"
  { echo "# per-file sha256 manifests (find -L <pkg> -type f | sort | shasum -a 256), run $STAMP"
    echo "## smelt package: $SMELT_PKG"
    find -L "$SMELT_PKG" -type f -print0 2>/dev/null | sort -z | xargs -0 shasum -a 256
    echo "## tts package: $TTS_PKG"
    find -L "$TTS_PKG" -type f -print0 2>/dev/null | sort -z | xargs -0 shasum -a 256
  } > "$manifests"
  log "package tree manifests -> $manifests  sha256=$(sha "$manifests")"
  log "invocation templates (resolved; the timer wraps each with t0->first-byte/exit stamps):"
  log "  smelt_cold|smelt_warm: $SMELT run $SMELT_PKG \"$PROMPT\" --max-tokens 1 --temp 0"
  log "  smelt_linger: $SMELT run $SMELT_PKG \"$PROMPT\" --max-tokens 1 --temp 0 --linger 30"
  log "  llamacpp_*: $LLAMA_COMPLETION -m $GGUF -p \"$PROMPT\" -n 1 --temp 0 --no-display-prompt --no-warmup </dev/null"
  log "  ollama_*: ollama run $OLLAMA_MODEL \"$PROMPT\""
  log "  mlx_*: $MLX_PY -m mlx_lm generate --model $MLX_MODEL --prompt \"$PROMPT\" --max-tokens 1 --temp 0  (timer --start-after banner+newline)"
  log "  tts_*: $SMELT run $TTS_PKG \"Hello, this is a first audio latency test.\" --max-frames 1 --greedy  [tts_linger adds --linger 30]"
  log "  trace: SMELT_STARTUP_TRACE=1 $SMELT run $SMELT_PKG \"$PROMPT\" --max-tokens 1 --temp 0  (3 samples, full stderr)"
  log "--- end provenance ---"
}

# ---- driver ---------------------------------------------------------------
log "=== cold-start receipt run $STAMP ==="
log "machine: $(sysctl -n machdep.cpu.brand_string) | macOS $(sw_vers -productVersion) ($(sw_vers -buildVersion)) | $(( $(sysctl -n hw.memsize)/1024/1024/1024 ))GB"
log "N=$N prompt=\"$PROMPT\"  raw -> $RAW_TSV"
provenance
for cell in $CELLS; do case "$cell" in
  smelt_cold) run_smelt cold;; smelt_warm) run_smelt warm;; smelt_linger) run_smelt_linger;;
  llamacpp_cold) run_llamacpp cold;; llamacpp_warm) run_llamacpp warm;;
  ollama_cold) run_ollama cold;; ollama_warm) run_ollama warm;;
  mlx_cold) run_mlx cold;; mlx_warm) run_mlx warm;;
  tts_cold) run_tts cold;; tts_warm) run_tts warm;; tts_linger) run_tts linger;;
  trace) run_trace;;
  *) log "unknown cell: $cell";; esac; done

log "=== summary (median / min / max) ==="
# Medians are computed EXACTLY (Decimal; inputs are 1-decimal ms, so a 12-trial
# median is exact at 2 decimals) and printed to 2 decimals; min/max as recorded.
# The summary FAILS the run unless every cell has exactly N finite rows.
python3 - "$RAW_TSV" "$N" <<'PY' | tee -a "$RUN_LOG"
import sys,csv
from decimal import Decimal
N=int(sys.argv[2])
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter="\t") if r["cell"] and not r["cell"].startswith("#")]
cells={}; bad=[]
for r in rows:
    c=cells.setdefault(r["cell"],{"tt":[],"ex":[],"n":0}); c["n"]+=1
    for k,dest in (("ttft_ms","tt"),("exit_ms","ex")):
        try:
            v=Decimal(r[k])
            if not v.is_finite(): raise ValueError
            c[dest].append(v)
        except Exception:
            bad.append(f'{r["cell"]} #{r["trial"]}: nonfinite {k}={r[k]!r}')
def med(vals):
    s=sorted(vals); n=len(s)
    m=s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2
    return f'{m:.2f}'
print(f'{"cell":16}{"n":>4}{"ttft_med":>11}{"ttft_min":>10}{"ttft_max":>10}{"exit_med":>11}')
for name,c in cells.items():
    tm=med(c["tt"]) if c["tt"] else "nan"; em=med(c["ex"]) if c["ex"] else "nan"
    lo=str(min(c["tt"])) if c["tt"] else "nan"; hi=str(max(c["tt"])) if c["tt"] else "nan"
    print(f'{name:16}{c["n"]:>4}{tm:>11}{lo:>10}{hi:>10}{em:>11}')
    if c["n"]!=N: bad.append(f'{name}: expected exactly N={N} rows, got {c["n"]}')
if bad:
    print("FATAL: summary validation failed:")
    for b in bad: print("  "+b)
    sys.exit(1)
PY
[ "${PIPESTATUS[0]}" = 0 ] || { log "FATAL: summary validation failed (see above)"; rm -rf "$WORK"; exit 1; }
log "done. raw: $RAW_TSV"
rm -rf "$WORK"
