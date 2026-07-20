# Cold-start receipt — methodology & results

**What this is.** A reproducible measurement of *process-exec → first-generated-token*
("cold start") for the `ia` agent runtime versus the field — Ollama, llama.cpp,
and mlx_lm — on the **same model at the same quant tier**, plus time-to-first-audio
(TTFA) for the Qwen3-TTS package. This is the launch post's spine: the number
nobody else reports with a repeatable harness.

Written in the style of `docs/ASSUMPTIONS.md`: every published number traces to a
raw per-trial artifact, and "cold" is defined by exactly what the harness controls
— no stronger claim than what is measured.

## Current fresh-process update — 2026-07-14

The v0.1 release includes a later Qwen 3.5 0.8B kernel pass than the page-cold
field comparison below. Three independent 12-run cohorts measured the current
canonical package from a fresh process with OS pages resident and no lingering
worker:

| Metric | Median | Range | Trials |
|---|---:|---:|---:|
| traced `exec` → first token | **82.8 ms** | 69.7–93.9 ms | 36 |
| external launch → process output wall time | **99.11 ms** | 86.12–109.99 ms | 36 |

The first value begins at process exec and is the number used on
instantagent.dev. The second includes the parent harness's spawn overhead and
process teardown. Neither is described as page-cache cold: weights were
resident, while the model process itself was new on every trial. The separate
264.05 ms result below remains the full page-evicted comparison.

Raw evidence:

- [`qwen35-0.8b-startup-cohort-1.json`](raw/qwen35-0.8b-startup-cohort-1.json)
- [`qwen35-0.8b-startup-cohort-2.json`](raw/qwen35-0.8b-startup-cohort-2.json)
- [`qwen35-0.8b-startup-cohort-3.json`](raw/qwen35-0.8b-startup-cohort-3.json)
- [`qwen35-0.8b-perf-catalog.json`](raw/qwen35-0.8b-perf-catalog.json)

- **Date:** 2026-07-13 (published run `20260713-133404`)
- **Harness:** `tools/bench-cold-start.sh` (+ `tools/coldstart_timer.py`)
- **Raw artifacts:** `docs/receipts/raw/cold-start-raw-<stamp>.tsv` (every trial),
  `docs/receipts/raw/cold-start-run-<stamp>.log` (run log incl. the provenance
  block + load averages), `ia-startup-trace-<stamp>.txt` (3-sample trace),
  `package-manifests-<stamp>.txt` (per-file sha256 of both package trees);
  published run: `cold-start-raw-20260713-133404.tsv`.
- **One-command reproduce:** `tools/bench-cold-start.sh`

## Hardware / OS / build

| | |
|---|---|
| Machine | Apple MacBook Pro, **Apple M2 Max** (12-core: 8P+4E), 32 GB unified memory |
| GPU | M2 Max integrated (Metal) |
| OS | macOS 15.7.5, build 24G624 |
| `ia` build | release build of `AgentCLI` at commit `08faf23d` (branch `l1-coldstart`, off `release-phase3-rename`); the run log's provenance block records the binary path + sha256 and the exact git HEAD + dirty-file list at run time |
| Uptime at run | 33 days (see run log for per-batch `uptime` load averages) |

This is a **shared developer machine**; the published run was taken in a quiet
window (per-batch 1-min load 2.9–3.8, recorded to the raw artifact immediately
before each batch). The harness records a **provenance block** in the run log at
launch — git HEAD + dirty-state list, resolved artifact paths + sha256 hashes
(binary, weights, harness, timer, this doc), full per-file sha256 manifests of
both package trees (sibling raw file `package-manifests-<stamp>.txt`), tool
versions, the effective config with per-cell resolved invocation templates, full
`uptime`, and a process snapshot (`ps -Ao pcpu,comm -r`) — all in the published
run's `cold-start-run-20260713-133404.log`, so machine-state and build/version
claims are backed by the recorded log, not narration. Every measured cell holds a machine-wide `flock` for its entire trial
batch; the lock excludes only *cooperating* jobs that take it (sibling bench
agents do), not arbitrary processes. Policy: cells taken under a visible load
spike are re-run in a quieter window rather than published — the retained
first-pass runs in `raw/`, taken beside four sibling bench agents, are exactly
that history.

## The metric

Measured externally by `tools/coldstart_timer.py` with a monotonic wall clock,
identically for every contender:

- **`ttft_ms`** — from just-before-`Popen` (t0, which includes fork/exec + dyld)
  to the **first byte of generated output on the child's stdout**. Each contender
  is configured to generate **exactly one token** (`--max-tokens 1` / `-n 1` /
  `num_predict 1`) and to send all log/preamble noise to stderr, so the first
  stdout byte is the first token. For a non-streaming tool that prints its single
  token at the end (`ia`), first-byte ≈ the token; for streaming tools it is the
  first token. Pinning `n=1` makes the two definitions converge — every tool does
  "load cold → produce one token → finish." Each raw row retains the timer's
  first-bytes preview (`first=…` in the note field), so the
  stamped-byte-was-a-token claim is auditable per trial.
- **`exit_ms`** — t0 → process exit (`ttft` + teardown). Robust cross-check that
  needs no stdout parsing; reported alongside `ttft`.

For `mlx_lm` (`--verbose` defaults to `True`) the CLI prints a `==========`
banner to stdout, but Python **block-buffers** the pipe: the banner is not
flushed until the first *token* print (`flush=True`), so the banner and the first
token land in the **same first observed pipe chunk** (a single flush), at the
moment the first token is generated.
`ttft` therefore stamps exec→first-token *regardless* of the marker. The harness
passes `--start-after` the banner **plus its trailing newline** (11 bytes,
`==========\n`), so the stamped byte is the first token byte itself — the marker
does **not** skip an early-printed banner (there is none). Byte order and flush
behavior are evidenced by a separate instrumented probe retained at
`docs/receipts/raw/mlx-stdout-probe.txt`: the first observed pipe chunk is
`==========\n<first-token>` at ≈2.3 s, and process exit follows ≈80 ms later
(final stats + teardown) — `ttft` is a genuine first-token latency, distinct
from `exit`.

**Seed / determinism policy.** All text runs use **greedy decoding (temp = 0 /
argmax)**, so the first token is deterministic and there is no sampling variance
to sweep — the only variance is system-level, captured by the N trials. (The
ASSUMPTIONS.md seed-sweep gate targets throughput/quality claims; it does not
apply to a deterministic first-token latency.)

**Trials.** N = 12 per cell (≥ 10 required). Reported: **median** (headline),
**min**, **max**. Medians are computed exactly (inputs are 1-decimal ms, so a
12-trial median is exact at 2 decimals) and reported to 2 decimals by the
harness summary; min/max and all other stats as recorded. The summary fails the
run unless every cell has exactly N finite rows. All raw per-trial values are in
the TSV.

## What "cold" means (honest definition)

Passwordless `sudo` is **not** available on this machine (`sudo -n purge` →
"a password is required"), so a full system page-cache `purge` is **not** used.
Instead the harness controls cold state as follows, and claims nothing beyond it:

1. **Fresh process every trial.** No warm/reused process; no `ia --linger` worker
   running.
2. **Page cache — evicted, then verified, fail-closed.** Before each cold trial
   the harness runs `vmtouch -e` on the model's *real* backing files (resolving
   `ia`'s content-addressed-store symlinks to the real blobs; the GGUF; the mlx
   safetensors; the Ollama model's manifest-referenced blobs). `vmtouch -e`
   evicts a file's pages from the unified page cache **without root** for files
   you own, and it targets *exactly* the model bytes; it does **not** evict other
   OS caches — a narrower "cold" than a global `purge`. Residency after eviction
   is verified with a `vmtouch` query (no `-e`): the exact resident/total **page
   counts** are recorded per trial in the raw artifact (`# resid` lines), zero
   resident pages is **required**, and on a dirty or failed check the harness
   retries the eviction once and then **aborts the cell** rather than proceeding.
3. **No daemon warmth (where applicable).** `ia`, `llama.cpp`, and `mlx_lm` have
   no daemon — every run is a full process + model load. **Ollama is different:**
   it has a persistent server. Its `ollama run` requires the server to be up (it
   does not self-start), so Ollama "cold" here means **server up, model verified
   unloaded from server RAM (`ollama stop`, then polled via `ollama ps` until it
   no longer appears — success recorded per trial as `# unload` lines; a poll
   failure or timeout aborts the cell) and its manifest-referenced blobs evicted
   from page cache** — i.e. Ollama's genuine *cold-model-load* latency. (If
   manifest resolution ever fails, the harness falls back to evicting **all**
   local Ollama blobs — a superset — and records which set was used in the raw.)
   The one-time server-daemon startup is **not** included in Ollama's number
   (the daemon is normally always-resident). This
   *favors* Ollama relative to `ia` (which pays full process init every time); it
   is called out here so the comparison stays honest.

**Not controlled — the Metal pipeline/shader cache.** There is no non-root way to
purge the GPU driver's compiled-pipeline (PSO) cache, so it is **not** evicted; it
warms after the first per-boot pipeline compile and persists across trials. It
is uncontrolled for every Metal-using contender. To show how much of `ia`'s cold
number is pipeline/init vs. weight load, the harness also captures `ia`'s
`AGENT_STARTUP_TRACE` stage decomposition (see the run log / raw dir).

## "Warm" and "linger"

- **Warm** — the identical run immediately repeated with **no eviction** (page
  cache hot, pipeline cache hot, fresh process). For Ollama: server up + model
  resident.
- **Linger** (`ia` only) — `ia run … --linger N` keeps the loaded model resident
  in a detached worker; subsequent runs forward the prompt over a Unix socket and
  skip tokenizer + model load. This is `ia`'s answer to a warm daemon. (The linger
  reply is non-streaming, so it measures warm end-to-end for one token.)

## Contenders — models & quant tiers

All four run **Qwen3.5-0.8B** (a hybrid SSM/attention "qwen3_5" architecture) at
the **4-bit tier** — the strongest cross-framework comparison available: same base
checkpoint, same task, same machine, same measurement wrapper. The contenders are
**independent 4-bit conversions of that one base checkpoint**, not bit-identical
weights — each ecosystem ships its own 4-bit scheme (u4 / Q4_0 / mlx-4bit), so the
quantizations differ; weight-file sizes are listed below so the tier is auditable.
(The `qwen3_5` architecture is brand-new; `llama.cpp` b8680 loads the **official
ggml-org** GGUF but *fails* on bartowski's Q4_K_M with `missing tensor
'blk.24.ssm_conv1d.weight'` — an SSM/hybrid quantization gap — which is why the
Q4_0 build is used.)

| Contender | Version | Model / package | 4-bit scheme | Weight file | Invocation (1 token, greedy) |
|---|---|---|---|---|---|
| **ia** | `AgentCLI` @ `08faf23d` | `Qwen_Qwen3.5-0.8B.agent` (shipped u4 release package, CAS-backed) | affine-**u4**, groupSize 64 | **404 MiB** (`weights.bin`) | `ia run <pkg> "<prompt>" --max-tokens 1 --temp 0` |
| **llama.cpp** | b8680 (`15f786e65`), Homebrew | `ggml-org/Qwen3.5-0.8B-GGUF` → `Qwen3.5-0.8B-Q4_0.gguf` | GGUF **Q4_0** | 537 MiB | `llama-completion -m <gguf> -p "<prompt>" -n 1 --temp 0 --no-display-prompt --no-warmup </dev/null` |
| **Ollama** | 0.31.1, Homebrew | same ggml-org Q4_0 GGUF, imported via Modelfile as `qwen35-08b-q4` (`num_predict 1`, `temperature 0`) | GGUF **Q4_0** | 537 MiB blob | `ollama run qwen35-08b-q4 "<prompt>"` |
| **mlx_lm** | 0.31.3 (venv, Python 3.12.2) | `mlx-community/Qwen3.5-0.8B-4bit` | MLX **4-bit**, groupSize 64 | 596 MiB (`model.safetensors`) | `python -m mlx_lm generate --model <id> --prompt "<prompt>" --max-tokens 1 --temp 0` |

`vmtouch` 1.3.1 performs the page-cache eviction. Note that `ia`'s u4 package is
the *smallest* on disk (404 MiB) — a tighter 4-bit than Q4_0/mlx-4bit — so it also
reads the fewest bytes on a cold page-cache miss.

## Results

All numbers are the **median of N=12** trials, computed exactly and reported to
2 decimals; min/max show the spread and span **all 12 trials of a cell — nothing
is excluded**. Every cell is from the single run `20260713-133404`
(`docs/receipts/raw/cold-start-raw-20260713-133404.tsv` + run log with the full
provenance block — git HEAD + dirty-file list, harness/timer/doc sha256s,
per-cell resolved invocation templates — plus the package-tree manifests file
`package-manifests-20260713-133404.txt`): per-batch 1-min load 2.9–3.8;
cold-state verification recorded per trial — **63/63** residency checks read
`resident_pages=0/M` (5 cold cells × 12 trials + 3 trace samples), **12/12**
Ollama unload polls recorded `ok`, the Ollama eviction set was manifest-resolved
(`# evictset ollama_cold: manifest (3 files)` — exactly the model's blobs),
**36/36** TTS trials carry `wav_magic=ok` with real per-trial
`rc`/`stdout_bytes`, and **144/144** rows retain the timer's `first=` preview
(per-trial proof the stamped byte was a token/WAV byte); the summary's
exactly-N-finite-rows enforcement passed. Earlier runs — `20260711-224824`
(+ its mlx/TTS quiet re-run), `20260713-120716`, and `20260713-130028` — are
retained in `raw/` as history. Per-cell load averages are in the run logs.

### Text — exec → first token (ms), Qwen3.5-0.8B @ 4-bit, greedy

| Runtime | **COLD** median | min–max | WARM median | Daemon/resident |
|---|---|---|---|---|
| **`ia`** (u4) | **264.05** | 252.4–276.1 | **106.65** | **29.20** (`--linger`) |
| llama.cpp (Q4_0) | 1337.40 | 1291.9–1688.3 | 1063.35 | — (no daemon) |
| Ollama (Q4_0) | 1342.95 | 1337.8–1603.1 | — | 247.35 (server-resident) |
| mlx_lm (4-bit) | 2207.00 | 2190.4–2572.8 | 2140.15 | — (no daemon) |

**`ia`'s COLD start is 5.1× faster than llama.cpp's, 5.1× faster than Ollama's, and
8.4× faster than mlx_lm's** on the same base checkpoint. Daemon-vs-daemon (both keep
the model resident): `ia --linger` 29.20 ms vs Ollama's warm server 247.35 ms — **8.5×**.

mlx_lm's cold-vs-warm gap is small in every run (≈67 ms of a ≈2.2 s total in
the published run; ≈84 ms and ≈0 in the retained runs): evicting the weights
from the page cache changes the number only marginally, i.e. the cold weight
read is a minor component of mlx_lm's startup. (What the remainder is spent on
is not measured here. The mlx invocation gained an explicit `--temp 0` in this
run; the ≈110 ms median shift vs the retained `130028` run is published as
measured, with no mechanism claimed.)

Elevated trials this run (all retained; min–max always spans all 12): the three
non-`ia` cold batches' maxima are each batch's **trial #1** (llama.cpp 1688.3,
Ollama 1603.1, mlx 2572.8 — the same first-trial-of-batch effect appears, milder,
in `ia_cold` 276.1, `ia_linger` 45.8, and `tts_linger` 328.3); `ia_warm` has a
single mid-batch elevated trial (#6, 157.7 ms); and `tts_cold`'s **last four
trials (#9–12) drift to 1.21–1.95 s** against 0.93–1.01 s for trials #1–8 — a
tail-of-batch pattern with no identified mechanism. The medians are computed
over all 12 (tts_cold's 974.55 ms falls within its #1–8 range).

### What is inside `ia`'s 264.05 ms COLD median — `AGENT_STARTUP_TRACE` decomposition

The stage decomposition below is from the published run itself: its `trace` cell
(`docs/receipts/raw/ia-startup-trace-20260713-133404.txt`) captured 3 cold
samples — page cache evicted and **verified** per sample, same binary — with
full stderr including the `Timing:` prefill line. Values are the 3-sample spread.

| Stage | ms (3-sample spread) |
|---|---|
| exec → `main` (dyld) | 4.7–4.9 |
| tokenizer load | 2.0–2.7 |
| manifest decode / bake honesty / mmap | ~0.2 |
| Metal device + queue (overlapped) | 41.5–51.6 |
| metallib + dispatch tables + **pipeline states (async, 25 deferred)** + buffer alloc | 3.8–4.0 (**0.2 ms** for PSOs) |
| **weights join + residency** (cold read of 404 MiB u4) | **148.4–149.1** |
| — `AgentModel` init total | **195.3–205.0** |
| pipeline materialization wait (blocking) | 0.0 (all 3 samples) |
| + prefill + first token (the `Timing:` line) | 30.2–33.8 |

Stages sum to ≈237–242 ms per sample, against the published **264.05 ms** COLD
median. The ≈22–27 ms difference is **not itemized** by the trace: it covers
spawn overhead before `exec → main` as seen by the external timer (whose t0
precedes fork/exec), first-token emit up to the first stdout byte, and
trial-to-trial variance (the trace samples are separate cold runs within the
same published run, not the measured trials themselves). The cold number is
dominated by the **cold weight read (148–149 ms)** and **Metal device init
(42–52 ms)**. For pipeline states, the trace records the **async PSO launch:
0.2 ms** and the **later blocking wait: 0.0 ms** in all 3 samples — the baked
metallib lets pipeline-state creation launch async off the critical path, and it
had completed before first use; the compilation cost itself was overlapped and
**not independently measured**.

### TTS — cold-start time-to-first-audio (Qwen3-TTS, `--max-frames 1`, greedy)

The headline is the honest first-WAV-byte latency: the timer's `ttft` = wall time
from exec to the **first byte of audio (WAV) on `ia`'s stdout**. `exit` (exec →
process exit) is a labeled cross-check, and `ia`'s own internal stamp is the
load-complete → first-audio streaming latency. Numbers are from the published
run's TSV (`cold-start-raw-20260713-133404.tsv`).

| Cell | **exec → first-audio byte** (`ttft`) median (ms) | min–max | exec → exit (cross-check, ms) | `ia` internal load→first-audio stamp (ms) |
|---|---|---|---|---|
| `tts_cold` | **974.55** | 933.9–1946.3³ | 1014.30 | 293 |
| `tts_warm` | 692.40 | 667.2–714.6 | 734.15 | 295 |
| `tts_linger` | **100.80** | 98.1–328.3² | 102.50 | 83 |

Three numbers, each labeled for what it is:
- **Full COLD exec → first-audio byte ≈ 0.97 s** (`ttft`; fresh process, page
  cache evicted): process spawn + cold load of the 813 MB package + prewarm + the
  first 12 Hz frame's first WAV byte. This is the honest cold TTFA.
- **exec → exit ≈ 1.01 s** (cross-check): the same COLD run's full wall time
  (first-audio byte + finishing the one-frame write + teardown), ≈40 ms above the
  first-audio byte.
- **`ia`'s internal load→first-audio stamp ≈ 0.29 s** (COLD median 293 ms): the
  *load-complete → first-audio* streaming latency; it excludes the cold model
  load. With a resident worker (`--linger`) the full exec → first-audio byte
  drops to **~0.10 s**.

² `tts_linger` trial #1 (328.3 ms `ttft` / 330.0 ms exit) is the worker's first
post-warmup frame; trials #2–12 are 98.1–102.0 ms. It is retained in the raw and
in the min–max.

³ `tts_cold` trials #9–12 drifted to 1.21–1.95 s (vs 0.93–1.01 s for #1–8) — a
tail-of-batch pattern with no identified mechanism; all 12 are retained and the
median (974.55 ms) falls within the #1–8 range. See "Elevated trials" above.

### Reproduce

```
tools/bench-cold-start.sh                 # all cells, N=12, writes docs/receipts/raw/
N=12 CELLS="ia_cold llamacpp_cold" tools/bench-cold-start.sh   # subset
```
Config via env (`IA`, `IA_PKG`, `GGUF`, `MLX_MODEL`, `OLLAMA_MODEL`, `N`, `PROMPT`);
defaults are the exact artifacts documented above.

## Caveats

1. **No global page-cache purge.** Passwordless `sudo` is unavailable, so
   `sudo purge` is not used. Page-cache cold state is achieved with per-file
   `vmtouch -e` eviction of the model's real bytes; residency after eviction is
   verified per trial (a `vmtouch` query with no `-e`): the exact resident/total
   page counts are recorded in the raw (`# resid` lines) and the run **aborts**
   unless zero pages remain resident (after one eviction retry). All 63 checks
   in the published run read `resident_pages=0/M` (5 cold cells × 12 trials + 3
   trace samples). `vmtouch -e` targets exactly
   the model files; it does not evict other OS caches — a narrower "cold"
   definition than a global purge, documented honestly.
2. **Metal pipeline/PSO cache is not controlled.** No non-root way to purge the
   GPU driver's compiled-pipeline cache; it warms after the first per-boot compile
   and persists (uptime at this run: 33 days). It is uncontrolled for all
   contenders, and no cell measures a first-of-boot compile. For `ia`, the trace
   records the async PSO launch (0.2 ms) and a 0.0 ms blocking wait; the compile
   cost itself was overlapped and not independently measured.
3. **Ollama cold excludes daemon startup.** `ollama run` requires an already-running
   server (it does not self-start). Ollama-cold here is *cold model load into a
   warm server* (`ollama stop`, verified unloaded by polling `ollama ps` —
   recorded per trial, **12/12 `ok`** in the published run, failure or timeout
   aborts the cell — plus eviction of the model's manifest-referenced blobs; the
   published run's eviction set was manifest-resolved, exactly the model's 3 blob
   files, per its `# evictset` line); the always-resident daemon's own startup is
   not in the number. This *favors*
   Ollama vs. `ia` (which pays full process init every run) and is disclosed so
   the comparison stays honest.
4. **Shared machine, quiet window.** The published run's per-batch 1-min load was
   2.9–3.8, recorded in the raw/run log. Its provenance block records launch
   state directly: uptime 33 days (load averages 3.22/2.47/2.57) and a process
   snapshot (top consumers at launch: WindowServer ~29 %, opendirectoryd ~17 %,
   a `claude` helper ~17 %, iTerm2 ~12 %) — an ordinary working desktop, not an
   idle lab machine; no sibling bench jobs held the bench lock during the run.
   Every cell holds the lock for its whole batch (the lock excludes only
   cooperating jobs that take it). Thermal state is not recorded by the harness.
   (The retained first-pass runs were taken beside four sibling bench agents plus
   an unrelated ~490 %-CPU process — the reason for re-running.)
5. **4-bit schemes differ across frameworks** (u4 / Q4_0 / mlx-4bit). All are the
   "4-bit tier" but not bit-identical quantizations; weight-file sizes are listed
   so the tier is auditable. This is inherent to a cross-framework comparison.
6. **mlx first-token via banner marker.** `mlx_lm` (`--verbose=True` by default)
   prints a `==========` banner to stdout, but Python block-buffers the pipe, so
   the banner is flushed *together with* the first token's `print(…, flush=True)` —
   both land in the **same first observed pipe chunk** (a single flush) at
   first-token time (evidence: the instrumented probe retained at
   `docs/receipts/raw/mlx-stdout-probe.txt`, whose first observed pipe chunk is
   `==========\n<token>`; the probe reads the pipe in 4 KiB chunks, so it proves
   same-chunk arrival, not a single `write()` syscall). The harness passes
   `--start-after` the banner **plus its trailing newline** (11 bytes), so the
   timer's stamp lands on the first token byte itself; the marker does **not**
   skip an early-printed banner. `ttft` (≈2207 ms cold) is thus exec→first-token
   and sits ≈80 ms below `exit` (final stats + teardown) — it is **not**
   indistinguishable from `exit`.
   Both columns are in the raw.
7. **Greedy decoding** (`temp=0`/argmax) → deterministic first token; the N trials
   capture system variance only (no seed sweep needed for a deterministic latency).
8. **TTS three numbers.** For the `tts_*` cells the timer's `ttft` is the wall time
   to the **first WAV byte** on `ia`'s stdout (exec → first-audio byte); `exit` is
   the full run wall time (exec → exit); and `ia`'s own `text-to-pcm: first audio
   Xms` stamp (in the raw note field) is the *load-complete → first-audio* latency,
   which excludes the cold model load. Headline TTFA is the first-audio byte
   (`ttft`); the other two are cross-checks. (The oldest retained first-pass raw
   stored the `ia` stamp in the `ttft` column for `tts_*` cells; the published run
   stores the timer's first-audio byte there and names each number in the note
   field.)
