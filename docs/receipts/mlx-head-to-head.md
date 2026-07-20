# MLX head-to-head — Qwen3.5 0.8B & 2B (u4)

Launch receipt for the MLX head-to-head comparison. Smelt's shipped
u4 text SKUs benchmarked against the closest MLX equivalents on the same machine,
same quant tier, same named base model.

- **Date:** 2026-07-13 (aligned re-run; first pass 2026-07-11)
- **Machine:** Apple M2 Max (`Mac14,5`), 12 CPU cores, 32 GB unified memory
- **OS:** macOS 15.7.5 (build 24G624)
- **Smelt build:** both packages built from `release-phase3-rename` @ `08faf23d` (post
  CAM->module rename); `ia` release binary rebuilt in this worktree and includes this
  unit's purely additive decode "Window wall" output line (uncommitted at build time;
  binary sha256 recorded in `mlx-h2h-env.txt` and, with per-file hashes of both packages,
  in `mlx-h2h-package-manifests.txt`)
- **MLX:** `mlx` 0.31.1, `mlx-lm` 0.31.1, CPython 3.14.6 (`/opt/homebrew/bin/python3`)
  (versions, snapshot revisions, ia binary sha256, and a process snapshot recorded in
  `mlx-h2h-env.txt`)
- **Harness:** `ia bench` (5 launches per rep for decode) / `ia prefill-bench` (Smelt)
  and `python3 -m mlx_lm.benchmark` (MLX), driven directly per rep inside one `flock`
  window; the literal command lines are echoed into every raw, and load average +
  `ProcessInfo` thermal state are recorded pre-lock, inside-lock, and post-run (see
  "Exact invocations"). The first pass used `tools/compare-mlx.sh`, which couples MLX
  decode to the 256-token prompt — the decode-alignment bug fixed below.

## Headline (no spin)

Throughput ratios, Smelt / MLX, median of the per-rep ratios over 6 reps per condition
(higher = Smelt faster). Quiet load is the headline regime; "back-to-back" is the
same-day no-cooldown contrast (thermal remained nominal this run):

| SKU | wall-clock decode (Smelt / MLX) | prefill-256 (Smelt / MLX) |
|-----|--------------------------------|----------------------------|
| Qwen3.5-0.8B u4 | quiet **0.97x — Smelt -2.5%**; back-to-back 0.93x — Smelt -6.8% | quiet **1.14x — Smelt +14.3%**; back-to-back 1.11x — Smelt +10.5% |
| Qwen3.5-2B u4   | quiet **0.97x — Smelt -3.4%**; back-to-back 0.94x — Smelt -5.9% | quiet **1.01x — Smelt +1.1%**; back-to-back 1.05x — Smelt +5.1% |

- **Decode: MLX wins wall-clock decode in all four conditions** — Smelt -2.5% (0.8B
  quiet), -3.4% (2B quiet), -6.8% (0.8B back-to-back), -5.9% (2B back-to-back). Both
  engines measured as whole-window throughput over the same 128-token window at KV
  positions 5-132 (see "Decode workload alignment"). Stated plainly, no softening.
- **Prefill-256: Smelt wins in all four conditions** — +14.3% (0.8B quiet), +10.5%
  (0.8B back-to-back), +1.1% (2B quiet; fixed-order run — see "Exact invocations"),
  +5.1% (2B back-to-back).
- **Known worst case, disclosed:** the first-pass high-load regime (1-min load ~5-6
  recorded per rep, back-to-back, Smelt decode drifting 218 -> 178 tok/s across the
  batch — consistent with thermal throttling, though thermal state was not recorded in
  the first pass) measured **Smelt -14% vs MLX on 2B prefill-256 under high load**
  (per-rep ratio 0.86x; a legacy mixed-estimator diagnostic — MLX mean-of-5-trials vs
  Smelt median-of-5 — indicative of the regime, not directly comparable to the aligned V2
  cells). This re-run recorded thermal-nominal at every check and measured
  Smelt +5.1% in its back-to-back 2B prefill batch. Both results stand, conditions
  stated — see "Known worst case" below.
- Smelt's orthogonal, durable advantages — instant cold start (ready-to-run artifact:
  no per-run weight download, no JIT compile) and constrained/structured decoding — are
  not captured by this throughput-only comparison and are measured elsewhere.

## Fair-comparison note (why these are the closest equivalents)

Both sides are built from the **same named upstream model** at the **same quantization
tier**, so this is a like-for-like tier comparison, not a quant-tier mismatch:

| | Smelt package | MLX model |
|---|---|---|
| 0.8B | `Qwen/Qwen3.5-0.8B` -> `affine_u4`, group_size 64 | `mlx-community/Qwen3.5-0.8B-4bit` — `{bits:4, group_size:64, mode:affine}` |
| 2B | `Qwen/Qwen3.5-2B` -> `affine_u4`, group_size 64 | `mlx-community/Qwen3.5-2B-4bit` — `{bits:4, group_size:64, mode:affine}` |

- What is matched is **architecture, parameter count, and 4-bit tier**. Smelt's packages
  build from `Qwen/Qwen3.5-0.8B` / `Qwen/Qwen3.5-2B`; the exact upstream variant of the
  `mlx-community` conversions is ambiguous in their own metadata (the YAML `base_model`
  names `Qwen3.5-*-Base` while the prose card names `Qwen3.5-*` — an internal
  inconsistency in the model cards), and upstream revision hashes are not independently
  recorded on either side. The throughput comparison is unaffected by the variant choice
  (identical tensor shapes and compute), and this receipt makes no quality claims.
- Architecture is `qwen3_5` (hybrid linear/full attention) on both sides.
- Quant scheme is identical: **affine 4-bit, group size 64** (independent 4-bit
  conversions of the same named base model, not the same quantized weights).
- Smelt additionally u4-quantizes the token embedding and ties the LM head
  (`quantizeEmbedding: true`, `tiedLMHead: true`); MLX 4-bit quantizes embed/lm_head by
  default as well, so this is not a divergence in favor of either side.
- MLX model revisions pinned: 0.8B `da28692b5f139cb0ec58a356b437486b7dac7462`,
  2B `674aaa7240b91e8012fcad5d791b7dfe5ba90207`.

## Measurement config & hygiene

- **Prefill window = 256 tokens** on both engines — like-for-like. Smelt =
  `prefill-bench --tokens 256 --iterations 5 --warmup 2`, whose "Wall time" line is the
  **median of 5 passes**; MLX = **median of the 5 per-trial `prompt_tps` values** from
  `--prompt-tokens 256 --generation-tokens 1 --num-trials 5` (the "Averages" line in the
  raws is a mean and is not used for the comparison).
- **Decode window = 128 tokens over KV positions 5-132** on both engines, and the **same
  statistic on both sides: whole-window throughput (128 tokens / window wall-clock),
  median of 5 windows per rep**. Smelt: 5 fresh `ia bench --iterations 128 --warmup 5`
  launches per rep, taking each launch's "Window wall" tok/s; per-rep value = median of
  the 5. MLX: median of the 5 per-trial `generation_tps` values from one
  `--prompt-tokens 5 --generation-tokens 128 --num-trials 5` launch. See "Decode
  workload alignment".
- **6 independent reps per condition** -> median + range per cell (the >=6 seed-sweep gate
  in `docs/ASSUMPTIONS.md`). **Quiet load is the headline regime for both SKUs.** (The
  "measure-tokens sweep" gate in that doc is spec-decode-specific and does not apply to
  this single-stream throughput A/B — no drafter.)
- **Regimes.** *Quiet:* each rep gated on 1-min load <= 4 and thermal nominal/fair, with a
  >= 25 s lock-released cooldown between reps. *Back-to-back:* no cooldown, reps run
  consecutively. This run shows no recorded thermal stress in either regime:
  `ProcessInfo` thermal state was **nominal at every pre-lock/inside-lock/post-run check
  across all 24 reps**, and there is no monotonic intra-batch decline suggestive of
  heat-soak (Smelt whole-window decode is simply the most variable metric — per-condition
  spread up to ~9% — and one 2B back-to-back rep contained a transient disturbance,
  disclosed below). The no-cooldown batches are therefore labeled back-to-back, not
  high/sustained load. (The first pass *did* record a high-load regime on the 2B; see
  "Known worst case".)
- **Shared machine:** every measured rep ran under
  `flock /private/tmp/claude-501/-Users-jud-Projects-smelt/bench.lock`, which excludes
  cooperating bench jobs (all our agents take this lock); it does not by itself guarantee
  an otherwise-idle GPU. Load average and thermal state were recorded pre-lock,
  inside-lock, and post-run for every rep; the inside-lock 1-min load range is shown per
  condition below.
- **Load sensitivity.** In the first pass, the 2B prefill ratio shifted sharply under a
  high-load regime (1-min load ~5-6 recorded per rep; thermal state was not recorded) —
  see "Known worst case". In this run, per-rep ratio medians moved up to ~4 points
  between regimes, in mixed directions (0.8B decode -4.3 points back-to-back, 2B prefill
  +3.9 points back-to-back), with quiet and back-to-back load ranges overlapping
  (2.2-3.8 vs 2.2-3.7); quiet remains the headline as the gated, cooled regime.

### Decode workload alignment

**Position alignment.** The first pass measured decode on mismatched contexts: `ia bench`
decodes from ~zero context (its 5-iteration run measured sequential KV positions 5-9),
while MLX's `--prompt-tokens 256 --generation-tokens 128` measured `generation_tps` over
KV positions 256-384 (decode *after* a 256-token prompt). Decode cost grows with KV
length, so this favored whichever engine was measured at the lower context (Smelt) — not
like-for-like. The first-pass decode ratios are therefore not comparable to this run's
and are not quoted. This run aligns the window using existing flags on both sides: both
engines decode the **same 128-token window at KV positions 5-132**:
- Smelt: `ia bench <pkg> --iterations 128 --warmup 5` — warmup fills positions 0-4, then
  the 128 measured single-token decode steps run at positions 5-132 (varying-sequential,
  wrap-at-200 not reached).
- MLX: `mlx_lm.benchmark --prompt-tokens 5 --generation-tokens 128` — the 5-token prompt
  fills positions 0-4, then the 128 generated tokens (`generation_tps`) run at positions
  5-132.

**Statistic alignment.** `ia bench`'s long-standing "Median" line is
1000 / median(single-step ms) — a per-step median that discounts the engine's own slow
tail and ignores inter-step gaps — while MLX's `generation_tps` is whole-generation
tokens / elapsed, a whole-window average. Those are different statistics, and the
per-step median flatters Smelt's spikier step distribution. The comparison statistic is
therefore **whole-window throughput on both sides**: `ia bench` gained a purely additive
"Window wall" output line (monotonic wall-clock from just before the first measured step
to just after the last; tok/s = 128 / elapsed), and each side's per-rep decode value is
the **median of 5 whole windows** — 5 fresh `ia bench` launches for Smelt, the 5
in-process trials for MLX. This estimator change is what flips the decode result vs the
per-step view: Smelt's per-step median is higher than MLX's whole-window rate, but its
tail steps and submit overhead cost the window (see the diagnostics note in "Results").

`prefill-256` was already like-for-like in both passes (both engines prefill exactly 256
tokens); it is taken from a **separate** MLX invocation (`--prompt-tokens 256`), so the
two MLX metrics do not share one coupled call.

### Exact invocations

Per rep, all commands run inside one `flock` window in a fixed order (all Smelt
invocations, then all MLX); the within-rep ratio therefore controls for slow drift
between reps, not for order effects within a rep. Load average and `ProcessInfo` thermal
state are recorded pre-lock, inside-lock, and post-run, and each literal command line is
echoed into the raw. `<pkg>` / `<mlx-model>` per SKU from the
"Fair-comparison note" table.

```
# Smelt decode — run 5x per rep; per-rep decode = MEDIAN of the 5 "Window wall" tok/s values
.build/release/ia bench <pkg> --iterations 128 --warmup 5
#   -> "  Window wall:  <ms>ms  (<tok/s> tok/s over 128 steps)"  = one window's decode tok/s
#      (the per-step "Median"/"Pure GPU med" lines are diagnostics, not the comparison stat)

# Smelt prefill-256 — one launch; the "Tokens/sec" line is already the median of 5 passes
.build/release/ia prefill-bench <pkg> --tokens 256 --iterations 5 --warmup 2
#   -> "Tokens/sec:" = prefill-256 tok/s;  "P95:" = prefill p95

# MLX decode — one launch; per-rep decode = MEDIAN of the 5 "Trial i:" generation_tps values
KMP_DUPLICATE_LIB_OK=TRUE python3 -m mlx_lm.benchmark --model <mlx-model> \
  --prompt-tokens 5 --generation-tokens 128 --num-trials 5 --prefill-step-size 2048

# MLX prefill-256 — one launch; per-rep prefill = MEDIAN of the 5 "Trial i:" prompt_tps values
KMP_DUPLICATE_LIB_OK=TRUE python3 -m mlx_lm.benchmark --model <mlx-model> \
  --prompt-tokens 256 --generation-tokens 1 --num-trials 5 --prefill-step-size 2048
```

Parsing note: MLX values come from the per-trial `Trial i:` lines (median of 5); the
`Averages:` line is a mean and is not used. (`KMP_DUPLICATE_LIB_OK=TRUE` is required:
importing `mlx_lm` under this Python otherwise aborts with OpenMP "Error #15" — duplicate
`libomp`.)

## Results

Throughput in tokens/sec (higher is better). Per rep: **Smelt decode** = median of 5
`ia bench` launches' "Window wall" tok/s (whole 128-step window, wall-clock); **MLX
decode** = median of the 5 per-trial `generation_tps` values; **Smelt prefill** =
`prefill-bench` "Tokens/sec" (median of 5 passes); **MLX prefill** = median of the 5
per-trial `prompt_tps` values. "per-rep ratio" = median (and range) of the Smelt/MLX
ratio computed within each rep. "load" = inside-lock 1-min load average. Thermal state
was nominal at every check in every rep below.

**Diagnostics, not competing decode claims:** each condition also lists Smelt's per-step
median and kernel-only pure-GPU rates. Smelt's median per-step latency is lower than its
whole-window rate implies — tail steps and submit overhead cost the window — and pure-GPU
shows kernel-only headroom. The wall-clock decode claim is the window number in the
tables, not these diagnostics.

### Qwen3.5-0.8B u4

MLX peak process memory, max observed across all trials: 0.795 GB (prefill-256
invocation) / 0.454 GB (5-token-prompt decode invocation); the "Averages" line in the
raws is a mean of per-trial peaks. On-disk: Smelt package 443 MB; MLX snapshot 622 MiB,
of which ~192 MiB is `vision_tower` weights that `mlx_lm` discards for this text-only
benchmark — the like-for-like on-disk text footprint is ~430 MiB. (The vision-weight
sizes here and for the 2B below were verified from the safetensors headers of the
benchmarked snapshots.)

**(a) Quiet — headline — 6 reps, load 2.2-3.8:**

| metric | Smelt median | Smelt range | MLX median | MLX range | per-rep ratio |
|---|---|---|---|---|---|
| decode (window) tok/s | 327.2 | 304.7–331.5 | **335.4** | 330.4–336.0 | 0.975x (0.915–1.003) |
| prefill-256 tok/s | **3229.4** | 3200.8–3248.9 | 2821.1 | 2726.5–2914.7 | 1.143x (1.104–1.189) |

- Diagnostics: Smelt per-step median 384.0 tok/s (361.0–388.5); pure-GPU median 441.5
  tok/s (423.7–450.5).

| rep | load | Smelt dec | MLX dec | dec ratio | Smelt pre | MLX pre | pre ratio |
|---|---|---|---|---|---|---|---|
| 1 | 3.1 | 324.3 | 335.3 | 0.967 | 3248.9 | 2914.7 | 1.115 |
| 2 | 3.8 | 320.1 | 335.5 | 0.954 | 3242.0 | 2726.5 | 1.189 |
| 3 | 2.7 | 330.1 | 336.0 | 0.982 | 3244.2 | 2745.3 | 1.182 |
| 4 | 2.5 | 330.2 | 335.8 | 0.983 | 3216.9 | 2747.4 | 1.171 |
| 5 | 2.2 | 331.5 | 330.4 | 1.003 | 3200.8 | 2894.8 | 1.106 |
| 6 | 3.0 | 304.7 | 333.0 | 0.915 | 3201.3 | 2900.2 | 1.104 |

**(b) Back-to-back (no cooldown; thermal remained nominal this run) — 6 reps, load
3.2-3.7:**

| metric | Smelt median | Smelt range | MLX median | MLX range | per-rep ratio |
|---|---|---|---|---|---|
| decode (window) tok/s | 313.9 | 303.7–331.5 | **336.9** | 333.0–337.6 | 0.932x (0.900–0.995) |
| prefill-256 tok/s | **3234.4** | 3202.6–3245.3 | 2916.0 | 2789.0–2939.6 | 1.105x (1.098–1.163) |

- Diagnostics: Smelt per-step median 363.9 tok/s (356.8–383.0); pure-GPU median 421.1
  tok/s (418.4–442.5).

| rep | load | Smelt dec | MLX dec | dec ratio | Smelt pre | MLX pre | pre ratio |
|---|---|---|---|---|---|---|---|
| 1 | 3.6 | 331.5 | 333.0 | 0.995 | 3226.4 | 2939.2 | 1.098 |
| 2 | 3.7 | 311.6 | 336.3 | 0.927 | 3202.6 | 2895.6 | 1.106 |
| 3 | 3.5 | 316.2 | 337.2 | 0.938 | 3242.4 | 2936.4 | 1.104 |
| 4 | 3.2 | 306.3 | 336.6 | 0.910 | 3245.3 | 2939.6 | 1.104 |
| 5 | 3.4 | 303.7 | 337.6 | 0.900 | 3218.6 | 2789.1 | 1.154 |
| 6 | 3.4 | 318.2 | 337.5 | 0.943 | 3243.5 | 2789.0 | 1.163 |

### Qwen3.5-2B u4

MLX peak process memory, max observed across all trials: 1.523 GB (prefill-256
invocation) / 1.090 GB (5-token-prompt decode invocation); the "Averages" line in the
raws is a mean of per-trial peaks. On-disk: Smelt package 1.0 GB; MLX snapshot 1.6 GiB,
of which ~632 MiB is `vision_tower` weights that `mlx_lm` discards for this text-only
benchmark — the like-for-like on-disk text footprint is ~1.0 GiB.

**(a) Quiet — headline — 6 reps, load 2.3-3.8:**

| metric | Smelt median | Smelt range | MLX median | MLX range | per-rep ratio |
|---|---|---|---|---|---|
| decode (window) tok/s | 191.9 | 185.3–195.3 | **199.5** | 197.0–200.2 | 0.966x (0.930–0.975) |
| prefill-256 tok/s | **1367.2** | 1363.7–1380.8 | 1353.2 | 1342.7–1367.9 | 1.011x (0.999–1.019) |

- Diagnostics: Smelt per-step median 209.2 tok/s (206.4–211.3); pure-GPU median 226.8
  tok/s (225.7–230.4).

| rep | load | Smelt dec | MLX dec | dec ratio | Smelt pre | MLX pre | pre ratio |
|---|---|---|---|---|---|---|---|
| 1 | 2.6 | 190.4 | 200.2 | 0.951 | 1368.2 | 1342.7 | 1.019 |
| 2 | 2.3 | 195.3 | 200.2 | 0.975 | 1380.8 | 1367.9 | 1.009 |
| 3 | 2.5 | 194.7 | 199.6 | 0.975 | 1363.7 | 1365.0 | 0.999 |
| 4 | 2.8 | 189.2 | 197.0 | 0.960 | 1369.8 | 1344.1 | 1.019 |
| 5 | 3.8 | 193.5 | 199.2 | 0.972 | 1366.3 | 1348.6 | 1.013 |
| 6 | 3.4 | 185.3 | 199.3 | 0.930 | 1365.2 | 1357.8 | 1.005 |

**(b) Back-to-back (no cooldown; thermal remained nominal this run) — 6 reps, load
2.2-3.0:**

| metric | Smelt median | Smelt range | MLX median | MLX range | per-rep ratio |
|---|---|---|---|---|---|
| decode (window) tok/s | 189.7 | 183.6–193.6 | **201.3** | 174.3–201.6 | 0.941x (0.914–1.101) |
| prefill-256 tok/s | **1371.6** | 918.9–1378.7 | 1303.6 | 1288.8–1335.5 | 1.051x (0.709–1.066) |

- Diagnostics: Smelt per-step median 208.2 tok/s (206.1–209.6); pure-GPU median 227.0
  tok/s (226.2–228.3).

| rep | load | Smelt dec | MLX dec | dec ratio | Smelt pre | MLX pre | pre ratio |
|---|---|---|---|---|---|---|---|
| 1 | 3.0 | 189.0 | 201.6 | 0.938 | 1378.7 | 1306.1 | 1.056 |
| 2 | 2.6 | 186.5 | 201.1 | 0.927 | 1374.4 | 1288.8 | 1.066 |
| 3 | 2.4 | 190.4 | 201.5 | 0.945 | 1370.2 | 1301.0 | 1.053 |
| 4 | 2.3 | 193.6 | 201.5 | 0.961 | 1367.0 | 1335.5 | 1.024 |
| 5 | 2.2 | 183.6 | 200.9 | 0.914 | 1372.9 | 1309.9 | 1.048 |
| 6 | 2.5 | 192.0 | 174.3 | 1.101 | 918.9 | 1295.2 | 0.709 |

**Rep-6 transient (disclosed, not excluded):** in 2B back-to-back rep 6, Smelt prefill
(918.9 tok/s vs 1367.0-1378.7 in every other rep) and MLX decode (174.3 tok/s vs
200.9-201.6) are BOTH depressed in the same rep — a transient machine disturbance, not
attributed (thermal stayed nominal; inside-lock 1-min load 2.5). The rep is retained in
the raws and the table; the medians are robust to it; its ratio cells (dec 1.101, pre
0.709) are the visible outliers in the ratio ranges above.

### Known worst case — 2B prefill-256 under high load (first pass, 2026-07-11)

The first pass measured the 2B under a high-load regime: 1-min load ~5-6 recorded per
rep, reps back-to-back with no cooldown, and Smelt decode drifting 218 -> 178 tok/s
across the batch — consistent with thermal throttling, though thermal state was not
recorded in the first pass (its raws record load averages and metrics only, not thermal
state or the identity of the other load on the box). In that regime, prefill-256 — which
was already like-for-like in the first pass (both engines prefill exactly 256 tokens;
only decode was misaligned) — measured **Smelt -14% vs MLX on 2B prefill-256 under high
load**: Smelt median 998.6 tok/s (859.7–1180.8) vs MLX 1157.5 (1050.2–1183.6), per-rep
ratio 0.86x (0.82–1.00). These first-pass numbers are a **legacy mixed-estimator
diagnostic**: the first-pass harness compared MLX's "Averages" line (a mean of 5 trials)
against Smelt's median-of-5, and its raws do not contain per-trial values, so the aligned
V2 estimator cannot be reconstructed — indicative of the high-load regime, not a
like-for-like cell comparable to the V2 numbers above. Smelt prefill gave up substantially more than MLX prefill in
that regime; the mechanism (a thermal-throttling asymmetry is the hypothesis) is not
established by the recorded data. The high-load regime remains the candidate
optimization target.

This re-run did not reproduce that regime (1-min load stayed ~2-4, thermal nominal at
every check); its back-to-back 2B batch measured Smelt +5.1% on prefill-256. No spin in
either direction: +5.1% is what this run measured; -14% is what the first-pass
high-load regime measured; both stand, conditions stated.

Backing raws (first pass):
- `docs/receipts/raw/mlx-h2h-2b-highload-firstpass-rep1.txt`
- `docs/receipts/raw/mlx-h2h-2b-highload-firstpass-rep2.txt`
- `docs/receipts/raw/mlx-h2h-2b-highload-firstpass-rep3.txt`
- `docs/receipts/raw/mlx-h2h-2b-highload-firstpass-rep4.txt`
- `docs/receipts/raw/mlx-h2h-2b-highload-firstpass-rep5.txt`
- `docs/receipts/raw/mlx-h2h-2b-highload-firstpass-rep6.txt`

## Reading of the result

- **Decode: MLX wins wall-clock decode in all four conditions** — Smelt -2.5% (0.8B
  quiet), -3.4% (2B quiet), -6.8% (0.8B back-to-back), -5.9% (2B back-to-back). MLX's
  whole-window decode is also much steadier (per-condition range <= ~1.7%, excluding the
  one transient-disturbance rep) than Smelt's (spread up to ~9%): Smelt's tail steps and
  submit overhead both slow and destabilize its window throughput. That gap — per-step
  median 384 tok/s vs 327 window tok/s on 0.8B quiet — is the concrete decode
  optimization target.
- **Prefill: Smelt wins in all four conditions** — +14.3%/+10.5% (0.8B quiet/back-to-back),
  +1.1% (fixed-order run)/+5.1% (2B). **In the first-pass high-load regime, 2B prefill dropped to Smelt
  -14% vs MLX; not reproduced in this thermally-nominal run** (legacy mixed-estimator
  diagnostic; see "Known worst case").
- The single most defensible one-line summary: **on the same named base model at the same
  4-bit tier (independent 4-bit conversions), MLX wins wall-clock decode in every
  condition (Smelt -2.5% to -6.8%); Smelt wins prefill-256 in every condition (+1.1% to
  +14.3%); the first-pass high-load run additionally measured Smelt -14% on 2B
  prefill-256 (legacy mixed-estimator diagnostic; not reproduced thermally-nominal).**
- Not measured here: **cold start** (ready-to-run artifact vs. download+load+compile) and
  **structured/constrained output**, which is where the product wins independently of
  steady-state throughput.

## Raw output

Shipped with this receipt under `docs/receipts/raw/` (committed in the same change), one
file per rep. Each aligned-run file records the pre-lock/inside-lock load averages and
thermal states, the literal command lines, the full output of the per-rep invocations,
and the post-run load average and thermal state.

- Environment (python/mlx/mlx-lm versions, ia binary sha256 + worktree HEAD, HF snapshot
  revisions, uptime, process snapshot): `docs/receipts/raw/mlx-h2h-env.txt`
- Package/binary provenance (per-file sha256 of every file in both Smelt packages + the
  rebuilt ia binary): `docs/receipts/raw/mlx-h2h-package-manifests.txt`
- 0.8B quiet (headline):
  - `docs/receipts/raw/mlx-h2h-0.8b-quiet-rep1.txt`
  - `docs/receipts/raw/mlx-h2h-0.8b-quiet-rep2.txt`
  - `docs/receipts/raw/mlx-h2h-0.8b-quiet-rep3.txt`
  - `docs/receipts/raw/mlx-h2h-0.8b-quiet-rep4.txt`
  - `docs/receipts/raw/mlx-h2h-0.8b-quiet-rep5.txt`
  - `docs/receipts/raw/mlx-h2h-0.8b-quiet-rep6.txt`
- 0.8B back-to-back:
  - `docs/receipts/raw/mlx-h2h-0.8b-b2b-rep1.txt`
  - `docs/receipts/raw/mlx-h2h-0.8b-b2b-rep2.txt`
  - `docs/receipts/raw/mlx-h2h-0.8b-b2b-rep3.txt`
  - `docs/receipts/raw/mlx-h2h-0.8b-b2b-rep4.txt`
  - `docs/receipts/raw/mlx-h2h-0.8b-b2b-rep5.txt`
  - `docs/receipts/raw/mlx-h2h-0.8b-b2b-rep6.txt`
- 2B quiet (headline):
  - `docs/receipts/raw/mlx-h2h-2b-quiet-rep1.txt`
  - `docs/receipts/raw/mlx-h2h-2b-quiet-rep2.txt`
  - `docs/receipts/raw/mlx-h2h-2b-quiet-rep3.txt`
  - `docs/receipts/raw/mlx-h2h-2b-quiet-rep4.txt`
  - `docs/receipts/raw/mlx-h2h-2b-quiet-rep5.txt`
  - `docs/receipts/raw/mlx-h2h-2b-quiet-rep6.txt`
- 2B back-to-back:
  - `docs/receipts/raw/mlx-h2h-2b-b2b-rep1.txt`
  - `docs/receipts/raw/mlx-h2h-2b-b2b-rep2.txt`
  - `docs/receipts/raw/mlx-h2h-2b-b2b-rep3.txt`
  - `docs/receipts/raw/mlx-h2h-2b-b2b-rep4.txt`
  - `docs/receipts/raw/mlx-h2h-2b-b2b-rep5.txt`
  - `docs/receipts/raw/mlx-h2h-2b-b2b-rep6.txt`
- 2B high load, first pass (worst case):
  - `docs/receipts/raw/mlx-h2h-2b-highload-firstpass-rep1.txt`
  - `docs/receipts/raw/mlx-h2h-2b-highload-firstpass-rep2.txt`
  - `docs/receipts/raw/mlx-h2h-2b-highload-firstpass-rep3.txt`
  - `docs/receipts/raw/mlx-h2h-2b-highload-firstpass-rep4.txt`
  - `docs/receipts/raw/mlx-h2h-2b-highload-firstpass-rep5.txt`
  - `docs/receipts/raw/mlx-h2h-2b-highload-firstpass-rep6.txt`

The remaining first-pass batches (0.8B at load ~6; the 4-rep 2B quiet batch) are
superseded by the aligned re-run and stay uncommitted (git-ignored `artifacts/l1-mlx-raw/`).
