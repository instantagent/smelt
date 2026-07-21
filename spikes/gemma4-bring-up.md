# Gemma 4 Bring-Up Status

This note captures the current state of the Gemma 4 text-model bring-up on main, with the focus on the official `google/gemma-4-E2B` path.

## Scope

- Canonical packaged path: affine/u4 build via `bash tools/build-gemma4-e2b.sh`
- Correctness bring-up path: FP16 build via `tools/gemma4-e2b-official-fp16.smelt`
- Current status: both FP16 and canonical affine/u4 E2B packages now restore decode/prefill parity on the active bring-up prompt

## What Is Working

- Gemma parser/IR support is in place for the staged fixtures and official E2B/E4B shapes.
- HuggingFace text-only checkpoint filtering and canonical tensor-name mapping are in place.
- Gemma-specific decode lowering is working for:
  - mixed sliding/global attention families
  - Gemma Q/K weighted per-head RMS norms
  - Gemma V no-scale per-head RMS norm
  - Gemma attention score softcapping
  - Gemma proportional split-half RoPE layout
  - Gemma per-layer residual inputs
  - Gemma residual `layer_scalar` weights
- Metal prefill package generation works for Gemma, including shared-KV lowering and per-layer residual inputs.
- The FP16 E2B bring-up package builds and runs.

## Confirmed Fixes Landed

1. Missing `layer_scalar` weights were added to Gemma weight layout and both decode and prefill residual branches now apply them.
2. Weight-build provenance now includes weight-layout names, shapes, and dtypes, so old `weights.bin` files are invalidated when Gemma layout changes.
3. Gemma proportional RoPE was fixed in both decode and prefill:
   - `ropeLayout == 2` now pairs `dim` with `dim + headDim / 2`
   - this applies to both normal decode RoPE and fused prefill RoPE/KV write
4. Gemma FP16 FFN-down prefill now keeps the down-projection transient in FP32 before the following RMSNorm.
5. Shared-KV Gemma prefill now still applies Q RoPE on shared layers while suppressing duplicate K/V writes.
6. CLI trace dumping was fixed for variable-width batched buffers and FP32 slots, which removed earlier false positives from the trace harness.
7. Generic decode attention kernels now snapshot aliased query inputs before writing the output buffer, which fixed the wide-head Gemma `L4` global-attention parity break.
8. Metal prefill now applies final `logit_cap` before argmax, matching decode and restoring exact logits parity after `final_norm`.
9. `smelt-probe` now provides lightweight row-level and final-slot decode/prefill comparison commands so Gemma debugging does not depend on repeated heavyweight CLI startup.
10. Gemma decode now routes FFN-down through the FP32 transient only when the down projection is actually FP16. This fixed the canonical affine/u4 decode path, which had been treating packed half output as FP32 input for `post_ffn_norm`.

## Current Runtime State

Using the FP16 correctness package built from:

```bash
bash tools/build-qwen35-2b.sh --spec tools/gemma4-e2b-official-fp16.smelt --output artifacts/gemma4-e2b-fp16
```

Current smoke result on prompt `The capital of France is`:

- normal Metal prefill returns first token `9079` (`Paris`)
- the current forced-decode validation path also returns first token `9079` (`Paris`)
- `smelt-probe compare-final-slot` now reports exact logits parity on the active six-token bring-up prompt for both the FP16 and affine/u4 packages:

```text
FP16:   maxDiff=0.0000 idx=0 prefill=-14.1250 decode=-14.1250 above0.01=0
Affine: maxDiff=0.0000 idx=0 prefill=-17.1875 decode=-17.1875 above0.01=0
```

This means the bring-up is no longer in the "immediate wrong first token" state and the previous Metal-prefill correctness blockers are cleared for the active FP16 and canonical affine/u4 repros.

## Recent Validation

- `swift build -c release` passed on the current Gemma bring-up branch state.
- Focused compiler/kernel coverage passed for the touched Gemma paths:
  - `GemmaCheckpointAdapterTests`
  - `WeightPackerTests`
  - `KernelCatalogTests`
  - `TopLevelEmitterTests`
  - `PrefillEmitterTests`
  - `AttentionPluginTests`
  - `PrefillKernelTests/testRopeAndKvCachePrefill_GemmaProportionalSplitHalfLayout`
  - `RoPETableTests`
- Opt-in runtime parity coverage now passes on both local Gemma E2B packages:
  - `SMELT_RUN_GEMMA_SMOKE=1 swift test --filter GemmaSmokeTests`
  - covers both `artifacts/gemma4-e2b-fp16/...` and `artifacts/gemma4-e2b-affine/...`
  - compares prefill vs decode final logits across both short prompts and a chunked-prefill prompt that exceeds `max_prefill_batch`
- Rebuilding the FP16 E2B package reuses `weights.bin` only when the expanded Gemma layout provenance still matches.

## Initial Performance Baseline

Initial benchmark baseline captured on 2026-04-16 against the two local E2B packages:

- canonical affine/u4: `artifacts/gemma4-e2b-affine/google_gemma-4-E2B.smeltpkg`
- correctness/reference FP16: `artifacts/gemma4-e2b-fp16/google_gemma-4-E2B.smeltpkg`

Steady-state decode and synthetic prefill were measured with:

```bash
bash tools/benchmark.sh artifacts/gemma4-e2b-affine/google_gemma-4-E2B.smeltpkg --decode-iterations 30 --decode-warmup 5 --prefill-iterations 5 --prefill-warmup 2 --prefill-tokens 64,256,512
bash tools/benchmark.sh artifacts/gemma4-e2b-fp16/google_gemma-4-E2B.smeltpkg --decode-iterations 30 --decode-warmup 5 --prefill-iterations 5 --prefill-warmup 2 --prefill-tokens 64,256,512
```

Results:

```text
Affine decode:      34.20ms/tok   29.2 tok/s   p95 48.93ms
Affine prefill-64:   494.8ms      129.3 tok/s  p95 506.5ms
Affine prefill-256: 4600.2ms       55.6 tok/s  p95 4909.9ms
Affine prefill-512: 15820.3ms      32.4 tok/s  p95 16143.5ms

FP16 decode:        42.94ms/tok   23.3 tok/s   p95 57.75ms
FP16 prefill-64:     976.5ms       65.5 tok/s  p95 1120.3ms
FP16 prefill-256:   7312.7ms       35.0 tok/s  p95 8022.6ms
FP16 prefill-512:  17358.6ms       29.5 tok/s  p95 20816.5ms
```

User-visible one-token CLI smoke was measured with:

```bash
/usr/bin/time -lp .build/release/smelt run artifacts/gemma4-e2b-affine/google_gemma-4-E2B.smeltpkg --prompt 'The capital of France is' --template '' --max-tokens 1
/usr/bin/time -lp .build/release/smelt run artifacts/gemma4-e2b-fp16/google_gemma-4-E2B.smeltpkg --prompt 'The capital of France is' --template '' --max-tokens 1
```

That coarse first-token check produced:

```text
Affine CLI smoke: prefill 486.7ms, wall 4.52s, completion Paris
FP16 CLI smoke:   prefill 2417.4ms, wall 9.03s, completion Paris
```

Takeaways from the initial baseline:

- affine/u4 is already materially ahead of FP16 on both decode and prefill, so it remains the right primary perf target
- prompt prefill falls off sharply once Gemma starts chunking above `prefill.max_prefill_batch = 64`
- the next performance work should focus on chunked prefill first, then decode tok/s
- `smelt bench --fixed-position` is not the cheap steady-state decode path for this CLI; it replays a zero-token prefix before each measured sample and should not be used for quick Gemma baselines

## Current Perf Tuning State

The first Gemma-specific affine tuning pass is now landed on top of the correctness bring-up:

- added fixed-shape affine decode kernels for the main Gemma E2B `group_size = 128` hot shapes
- added fixed-shape batched affine prefill kernels for the same `g128` hot shapes
- added a fixed-shape `rms_norm_1pw_d1536` decode and batched specialization
- added fixed-shape batched dual-affine K/V prefill kernels for Gemma `cols = 1536`

Kept result:

- decode now uses the new fixed-shape `g128` affine kernels
- prefill now uses fixed batched `g128` affine kernels, not the qmm/full path

Important constraint discovered during tuning:

- a first attempt to use Gemma `g128` qmm/full prefill kernels was faster but broke affine prefill-vs-decode logits parity on the prompt sweep
- that path was not kept
- the current kept path is the parity-clean fixed batched path, validated with:

```bash
SMELT_RUN_GEMMA_SMOKE=1 swift test --filter testGemmaE2BAffinePromptSweepPrefillMatchesDecodeFinalLogits
```

Current canonical affine/u4 benchmark after the kept tuning pass:

```text
Decode:        32.29ms/tok   31.0 tok/s   p95 47.51ms
Prefill-64:   416.3ms        153.7 tok/s  p95 419.4ms
Prefill-256: 4028.8ms         63.5 tok/s  p95 4036.9ms
Prefill-512: 15111.7ms        33.9 tok/s  p95 15140.9ms
```

Compared with the initial affine baseline, that is:

- decode: `29.2 -> 31.0 tok/s`
- prefill-64: `129.3 -> 153.7 tok/s`
- prefill-256: `55.6 -> 63.5 tok/s`
- prefill-512: `32.4 -> 33.9 tok/s`

Second Gemma prefill tuning pass:

- added a cached-score fast path to `attention_prefill` and `attention_prefill_softcap` for active attention spans up to `512`
- the kept version preserves the old `256`-thread launch shape so prefill stays numerically aligned with decode on the Gemma runtime sweep
- an intermediate attempt to cut Gemma prefill attention to `64/128` threads was faster but broke chunked-prompt decode/prefill logits parity and was not kept

Current canonical affine/u4 benchmark after the kept attention-prefill pass:

```text
Decode:        32.6ms/tok   30.6 tok/s   p95 47.6ms
Prefill-64:   186.9ms       342.5 tok/s  p95 190.9ms
Prefill-256: 830.4ms        308.3 tok/s  p95 832.2ms
Prefill-512: 1863.6ms       274.7 tok/s  p95 1873.3ms
```

Compared with the previous kept affine pass, that is:

- decode: effectively flat at about `31 tok/s`
- prefill-64: `153.7 -> 342.5 tok/s`
- prefill-256: `63.5 -> 308.3 tok/s`
- prefill-512: `33.9 -> 274.7 tok/s`

Current bottleneck after this pass:

- chunked prefill is no longer the main Gemma problem on the current benchmark shapes
- steady-state decode is now the clear next performance target
- decode is still materially limited by generic attention plus the remaining unfused affine buckets tied to `group_size = 128`

Decode tuning pass after the kept prefill work:

- added Gemma decode attention kernels for:
  - `attention_decode_d256_h8_kv1`
  - `attention_decode_d256_h8_kv1_sdpa`
  - `attention_decode_d512_h8_kv1`
  - `attention_decode_d512_h8_kv1_sdpa`
- the short vectorized kernels match generic decode in isolated shader tests, but enabling them in the runtime introduced prompt-sweep logit drift and they were not kept in the emitted dispatch path
- the kept runtime path now uses:
  - generic decode attention for `position + 1 < 128`
  - Gemma SDPA decode attention for `position + 1 >= 128`
- that guard keeps the affine Gemma prompt sweep exact again while still enabling the long-context decode path where steady-state generation actually spends time

Current affine/u4 status after the kept decode pass:

```text
Sequential decode bench: 33.50ms/tok   29.8 tok/s   p95 48.11ms
Prefill-64:             190.5ms        336.0 tok/s  p95 193.2ms
Prefill-256:            832.6ms        307.5 tok/s  p95 837.8ms
Prefill-512:           1905.6ms        268.7 tok/s  p95 2008.8ms
```

Fixed-position kernel profile still shows low attention share, but the `smelt kernels --position ...` subcommand is not yet trustworthy for dispatch-path attribution:

```text
attention_decode_d256_h8_kv1     28 dispatches   446us   4.3%
attention_decode_d512_h8_kv1      7 dispatches   159us   1.5%
TOTAL                                           10293us
```

Implications of that split:

- attention is no longer dominating the fixed-position GPU profile
- however, `smelt kernels --position 160` still reports the short Gemma kernels, even though the emitted decode guard now switches to SDPA at `position + 1 >= 128`
- so this profiler path is still good for rough bucket sizing, but not for proving which guarded decode kernel actually executed
- until that harness is fixed, runtime-level parity checks and capture-based inspection remain the source of truth for guarded-path attribution

Decode rows4 follow-up:

- added Gemma decode `rows4` affine kernels for the hottest `group_size = 128` shapes:
  - `affine_matvec_c1536_r2048_g128_rows4`
  - `affine_matvec_c1536_r256_g128_rows4`
  - `affine_matvec_c1536_r6144_g128_rows4`
  - `affine_matvec_c1536_r12288_g128_rows4`
  - `affine_matvec_c256_r1536_g128_rows4`
  - `affine_matvec_c2048_r1536_g128_rows4`
  - `affine_matvec_c6144_r1536_g128_rows4`
  - `affine_matvec_c12288_r1536_g128_rows4`
  - `affine_matvec_c1536_r262144_g128_rows4` for the tied LM head
- exactness checks passed against the generic affine kernel for all of those shapes, including the `262144 x 1536` LM-head path
- the canonical affine/u4 Gemma runtime smoke stayed exact after the rows4 selection changes:
  - `SMELT_RUN_GEMMA_SMOKE=1 swift test --filter testGemmaE2BAffinePromptSweepPrefillMatchesDecodeFinalLogits`
- the rebuilt generated package now emits those rows4 kernels directly in decode; for example:
  - Q projection uses `p[190]`
  - O projection uses `p[193]`
  - FFN gate/up use `p[192]`
  - FFN down uses `p[195]`
  - LM head uses `p[196]`

Current affine/u4 status after the rows4 decode follow-up:

```text
Sequential decode bench: 32.65ms/tok   30.6 tok/s   p95 47.58ms
Prefill-64:             190.6ms        335.8 tok/s  p95 192.4ms
Prefill-256:            825.6ms        310.1 tok/s  p95 835.0ms
Prefill-512:           1860.8ms        275.2 tok/s  p95 1884.4ms
```

Fixed-position profile after both decode rows4 passes, including the new `256 <-> 1536` rows4 buckets:

```text
affine_matvec_c1536_r12288_g...   40 dispatches   1261us   14.9%
rms_norm_1pw_d1536               176 dispatches    884us   10.4%
affine_matvec_c12288_r1536_g...   20 dispatches    807us    9.5%
affine_matvec_c1536_r262144...     1 dispatch      570us    6.7%
affine_matvec_c1536_r6144_g...    30 dispatches    557us    6.6%
elementwise_add                  106 dispatches    420us    5.0%
affine_matvec_c1536_r256_g...     59 dispatches    394us    4.6%
affine_matvec_c256_r1536_g...     35 dispatches    360us    4.2%
TOTAL                                            8484us    8.5ms/tok
```

Current decode implication:

- the added `256 <-> 1536` rows4 kernels are live in the fixed-position hot list and exact against generic
- the rows4 selection is real and safe, but it only moved sequential decode marginally
- the remaining decode wall is now mostly structural:
  - `rms_norm_1pw_d1536`
  - the small repeated `1536 -> 256` and `256 -> 1536` affine paths
  - the still-heavy large affine buckets themselves
- the compiler already has a cooperative norm-fusion pass, but it currently only recognizes:
  - generic `rms_norm_1pw`
  - generic `affine_matvec`
- Gemma decode is now mostly on specialized fixed-shape affine kernels plus `rms_norm_1pw_d1536`, so that pass does not currently fire on the hot Gemma decode chains
- the next serious decode target should therefore be norm fusion for `rms_norm_1pw_d1536` into Gemma’s specialized affine consumers, not more blind shape-pack churn

Residual-add fusion follow-up:

- tried extending `FuseMatvecResidualAddPass` so Gemma fixed-shape decode output/down projections could lower to generic `fused_affine_matvec_add`
- added optimizer coverage for the rows4 `2048 -> 1536` Gemma output-projection case
- however, the real Gemma decode package still shows `elementwise_add` at `106` dispatches, and the dispatch table size does not drop
- the reason is structural: the hot Gemma residual sites record trace markers such as `L*.attn_out` and `L*.ffn_down` between the matvec and the residual add
- the current peephole matcher only fuses truly adjacent dispatches, and simply skipping those markers would change trace semantics because the intermediate buffer would no longer exist
- that means this is not an unfused-kernel problem; it is a trace-contract problem
- if we want this win, we need one of:
  - a trace-aware fusion design that explicitly redefines those intermediate boundaries
  - an emitter-side fused residual path that records different markers
  - a no-trace / reduced-trace performance build mode

Stripped-mode norm+add spike:

- built a `traceMode = stripped` path so performance-only packages can omit exported trace markers while still preserving those marker ops internally as optimizer barriers
- that surfaced one real emitter bug: the post-FFN fallback path had accidentally tied the residual add to the trace-marker condition; stripped fallback now still emits the add
- also prototyped a fused `rms_norm_1pw_d1536_add` kernel for Gemma decode and wired it into the stripped emitter
- isolated shader tests stayed close, but not exact:
  - `rms_norm_1pw_d1536_add` vs the real decode reference path (`rms_norm_1pw_d1536` then scalar `elementwise_add`) still diverges by `0.000977` max diff, including the aliased `alt -> alt` case
- that one-half-ULP per-op drift accumulates across Gemma decode and breaks the runtime prompt sweep at the final logits:
  - observed runtime drift was about `0.035 ... 0.047` max diff on the current affine prompt set
- because correctness remains the hard requirement, that fused norm+add path is not kept in emitted packages right now
- current status after backing it out:
  - canonical full Gemma affine package restored to the known-good `993`-dispatch decode path
  - clean full-package runtime sweep re-confirmed with `SMELT_GEMMA_AFFINE_PACKAGE=.../gemma4-e2b-affine-fulltmp/... SMELT_RUN_GEMMA_SMOKE=1 swift test --filter testGemmaE2BAffinePromptSweepPrefillMatchesDecodeFinalLogits`
- implication:
  - stripped/no-trace packaging is now safer as infrastructure, but it is not yet a Gemma decode perf win
  - exact decode tuning should move back to other targets before revisiting norm+add fusion

## Debugging Constraint

CLI startup for the Gemma E2B package is too expensive to use as the normal inner-loop debugger.

- repeated `smelt run --debug ...` probes are heavy enough to stall local development and can leave multiple long-lived processes around if they are interrupted
- for Gemma bring-up work, CLI-based probes should be treated as final verification, not the primary iteration loop
- the preferred next step is a direct `SmeltRuntime` probe or a dedicated local debug harness that:
  - loads the package once
  - steps decode or prefill in-process
  - dumps specific slots or trace points without paying full CLI startup every time

## What Is Still Pending

There is no longer a known FP16 E2B decode/prefill parity blocker on the active bring-up prompt.

What is still not proven yet:

- broader prompt coverage beyond the current six-token `Paris` repro
- wider regression coverage so the fixed attention-aliasing and logit-cap issues cannot silently reappear
- wider regression coverage so the fixed affine Gemma FFN-down routing issue cannot silently reappear

The last real bugs that were blocking the current bring-up were:

1. the generic decode attention kernels were reading `query` in-place while also writing the aliased output buffer, which corrupted the upper half of wide Gemma `headDim = 512` decode outputs
2. prefill emitted LM head directly into argmax without the configured final `logit_cap`, so `final_norm` matched but logits did not
3. decode routed affine Gemma FFN-down through the FP32 transient path even though affine matvec writes half output, so `post_ffn_norm` was normalizing reinterpreted half data as floats

Those are fixed on the current branch state.

The remaining work is bring-up hardening, not the same parity bug hunt.

## Decode GeGLU Fusion

A safe affine GeGLU decode fusion is now in place for the hot Gemma E2B FFN shapes:

- added `fused_affine_gate_up_geglu`
- added fixed `rows4` decode specializations for:
  - `fused_affine_gate_up_geglu_c1536_r6144_g128_rows4`
  - `fused_affine_gate_up_geglu_c1536_r12288_g128_rows4`
- decode now routes affine Gemma GeGLU FFNs through that fused path instead of:
  - affine gate matvec
  - affine up matvec
  - standalone `geglu_fused`

Correctness status:

- kernel tests are exact against the iterated GPU reference path
- specialized Gemma kernels are exact against the generic fused kernel
- canonical affine package smoke still passes:

```bash
SMELT_RUN_GEMMA_SMOKE=1 swift test --filter testGemmaE2BAffinePromptSweepPrefillMatchesDecodeFinalLogits
```

Package/runtime effects:

- canonical affine decode dispatch table dropped from `993` to `923` ops
- runtime needed one loader fix so the base `fused_affine_gate_up_geglu` placeholder is created with dummy function constants, the same way the other FC-backed fused affine kernels already are

Measured decode result on the current sequential harness:

```text
run 1: 33.08 ms/tok, 30.2 tok/s
run 2: 34.05 ms/tok, 29.4 tok/s
```

So this change is correctness-safe and reduces dispatch count materially, but it did **not** deliver a meaningful steady-state decode speedup by itself.

The fixed-position kernel profile at `position 128` shows why the next work needs to move elsewhere:

```text
TOTAL: 8483 us
fused_affine_gate_up_geglu_c1536_r6144_g128_rows4 + _r12288_: 1899 us combined
rms_norm_1pw_d1536: 870 us
affine_matvec_c12288_r1536_g128_rows4: 825 us
affine_matvec_c1536_r262144_g128_rows4: 572 us
elementwise_add: 531 us
```

The next high-value decode targets are therefore:

1. exact-safe residual/add fusion so the `elementwise_add` wall can actually move
2. the remaining `rms_norm_1pw_d1536 -> affine consumer` chain
3. the large `12288 -> 1536` down projection path

## Ungated-Q Copy Removal

The next obvious decode cleanup after GeGLU fusion was structural, not shader math:

- non-gated Gemma attention layers were still projecting Q into `attnQBuf`
- decode then paid an explicit `buffer_copy` into `attnOutBuf`
- Gemma E2B has `35` attention layers, so this was one wasted decode dispatch per layer

That path is now fixed:

- ungated Q projection writes directly into `attnOutBuf`
- `L*.q_proj` trace markers now follow the real ungated-Q output slot instead of the old staging slot
- focused compiler coverage now proves:
  - Gemma sliding attention no longer emits `buffer_copy`
  - `L0.q_proj` resolves to `attnOutBuf`

Correctness status:

- attention plugin tests passed
- canonical affine runtime smoke still passes:

```bash
SMELT_RUN_GEMMA_SMOKE=1 swift test --filter testGemmaE2BAffinePromptSweepPrefillMatchesDecodeFinalLogits
```

Package/runtime effects:

- canonical affine decode dispatch table dropped from `923` to `888` ops
- this is the expected `35`-dispatch win from removing the ungated-Q copy from every Gemma attention layer

Measured decode result on the same sequential harness:

```text
32.32 ms/tok, 30.9 tok/s, p95 47.23 ms
```

Updated fixed-position profile at `position 128`:

```text
TOTAL: 8421 us
fused_affine_gate_up_geglu_c1536_r6144/_r12288: 1854 us combined
rms_norm_1pw_d1536: 1018 us
affine_matvec_c12288_r1536_g128_rows4: 812 us
affine_matvec_c1536_r262144_g128_rows4: 571 us
elementwise_add: 536 us
```

Implication:

- this is a real op-count win, not a harness ghost
- but the remaining decode wall is still dominated by:
  - `elementwise_add`
  - `rms_norm_1pw_d1536`
  - the repeated small and large affine buckets

## Repro Commands

Build:

```bash
swift build -c release
bash tools/build-qwen35-2b.sh --spec tools/gemma4-e2b-official-fp16.smelt --output artifacts/gemma4-e2b-fp16
```

Smoke:

```bash
.build/release/smelt run artifacts/gemma4-e2b-fp16/google_gemma-4-E2B.smeltpkg --prompt 'The capital of France is' --template '' --max-tokens 1
.build/release/smelt run artifacts/gemma4-e2b-fp16/google_gemma-4-E2B.smeltpkg --prompt 'The capital of France is' --template '' --max-tokens 1 --force-decode
.build/release/smelt run artifacts/gemma4-e2b-affine/google_gemma-4-E2B.smeltpkg --prompt 'The capital of France is' --template '' --max-tokens 1
```

Targeted parity probes:

```bash
.build/arm64-apple-macosx/release/smelt-probe compare-rows --package artifacts/gemma4-e2b-fp16/google_gemma-4-E2B.smeltpkg --ids 2,818,5279,529,7001,563 --slot 23 --row-width 4096 --prefill-dispatch 3394 --decode-dispatch 138
.build/arm64-apple-macosx/release/smelt-probe compare-rows --package artifacts/gemma4-e2b-fp16/google_gemma-4-E2B.smeltpkg --ids 2,818,5279,529,7001,563 --slot 8 --row-width 1536 --prefill-dispatch 24808 --decode-dispatch 885
.build/arm64-apple-macosx/release/smelt-probe compare-final-slot --package artifacts/gemma4-e2b-fp16/google_gemma-4-E2B.smeltpkg --ids 2,818,5279,529,7001,563 --slot 16 --count 262144
.build/arm64-apple-macosx/release/smelt-probe compare-final-slot --package artifacts/gemma4-e2b-affine/google_gemma-4-E2B.smeltpkg --ids 2,818,5279,529,7001,563 --slot 16 --count 262144
SMELT_RUN_GEMMA_SMOKE=1 swift test --filter GemmaSmokeTests
```

## Next Steps

1. Start Gemma performance tuning, using the canonical affine/u4 E2B package as the primary target and keeping `GemmaSmokeTests` plus `smelt-probe` as guardrails.
2. Prioritize the biggest user-visible cost centers first:
   - prompt prefill latency on the canonical affine package
   - decode tokens/second after the first generated token
3. Keep CLI debug as final verification only; the runtime smoke and `smelt-probe` path should stay the default inner loop.
4. Treat broader prompt-corpus expansion as follow-up hardening work, not as a blocker for perf tuning.

## Exactness Contract For The Next Perf Pass

Gemma decode tuning should now use a staged affine/u4 reference as the implementation oracle for each hot chain.

That means:

- local fused-vs-staged quantized tests come before end-to-end smoke
- FP16 and FP32 remain model-quality references, not fusion oracles
- non-exact region experiments, including the recent cooperative norm-fusion attempt, should stay out of the default optimizer path until they match the staged quantized contract

The longer-term compiler direction is documented separately in:

- `spikes/compile-time-fusion-aot-vision.md`

Local oracle status after tightening the staged residual contract:

- generic `fused_affine_matvec_add` still does not match staged `affine_matvec -> elementwise_add` exactly on Gemma `g128` decode shapes
- current max diffs on the local Metal kernel oracle are still:
  - `cols=2048`: `0.015625`
  - `cols=4096`: `0.001953125`
  - `cols=6144`: `0.015625`
  - `cols=12288`: `0.0078125`
- Gemma-specific fixed fused residual kernels are now exact against the staged specialized affine path after forcing a real `device` half store/reload before the residual add:
  - `fused_affine_matvec_add_c2048_r1536_g128_rows4`: `0.0`
  - `fused_affine_matvec_add_c4096_r1536_g128`: `0.0`
  - `fused_affine_matvec_add_c6144_r1536_g128_rows4`: `0.0`
  - `fused_affine_matvec_add_c12288_r1536_g128_rows4`: `0.0`
- the optimizer now selects those Gemma-specialized fused residual kernels again in local compiler tests
- that result identified the real blocker: live Gemma decode was not emitting `matvec -> elementwise_add` at the hot sites. The generated graph was `matvec -> norm -> add`, so the exact fused residual kernels were correct but structurally unreachable in the real package
- `rms_norm_1pw_d1536_add` is now exact for the non-aliased decode contract Gemma actually uses (`cur -> alt`), after forcing a real `device` half store/reload before the residual add
- the aliased `alt -> alt` form is still not exact and remains outside the current emitter contract
- `TopLevelEmitter` now emits `rmsNorm1PWD1536Add` for stripped Gemma decode, and the canonical `tools/build-gemma4-e2b.sh` wrapper now defaults to `--trace-mode stripped`
- rebuilding the canonical affine package now produces an `818`-op decode table instead of `888`, and the full affine Gemma runtime smoke still passes on that package
- updated canonical affine/u4 baseline after the op drop:
  - decode: `32.68 ms/tok`, `30.6 tok/s`, p95 `47.22 ms`
  - prefill `64/256/512`: `340.2 / 309.6 / 269.2 tok/s`
- that means the graph-reachability problem is fixed; the remaining decode wall is now throughput inside the surviving hot kernels, not the missing norm+add fusion
- follow-up on cooperative norm fusion:
  - tried making the path exact by forcing the fused `norm_scale_affine_matvec` kernels to consume half-rounded normalized lanes and by adding a specialized `rms_norm_scale_only_d1536` prepass so the RMS reduction geometry matched `rms_norm_1pw_d1536`
  - local kernel tests stayed exact against the generic fused kernels, but full Gemma runtime smoke still drifted by about `0.033 ... 0.047` on final logits
  - result: `CooperativeNormFusionPass` remains out of `defaultPasses`; the extra kernels stay in-tree as research, not product path
- follow-up on decode FFN tiling:
  - added rows8 GeGLU specializations for `fused_affine_gate_up_geglu_c1536_r6144_g128` and `...r12288...`
  - those rows8 kernels are exact against the generic fused GeGLU kernel
  - they did not improve the real Gemma benchmark enough to keep; a measured release run landed at about `33.35 ms/tok`, `30.0 tok/s`, with worse prefill too
  - result: rows8 GeGLU kernels remain available for future tuning, but `SmeltQwenShapePacks.decodeFusedGeGLUPipeline` stays on the rows4 variants
- follow-up on the final decode LM-head chain:
  - stripped Gemma decode now fuses the final `rms_norm_1pw_d1536 -> lm_head` path through `rms_norm_scale_only_d1536` + `norm_scale_affine_matvec_c1536_r262144_g128_rows4`
  - the canonical affine package emits that fused path in `SmeltGenerated.swift`, and the full affine Gemma runtime smoke still passes on it
  - this is a throughput win, not an op-count win: the decode table stays at `818` ops because the old `final_norm + lm_head` pair became `scale_only + norm-scaled lm_head`
  - clean post-rebuild benchmarks on the canonical affine package landed in the same band:
    - run 1: decode `26.92 ms/tok`, `37.2 tok/s`, p95 `37.44 ms`; prefill `339.2 / 301.4 / 251.2 tok/s`
    - run 2: decode `27.29 ms/tok`, `36.6 tok/s`, p95 `36.82 ms`; prefill `331.7 / 309.8 / 267.0 tok/s`
  - compared with the prior kept baseline (`32.68 ms/tok`, `30.6 tok/s`), this is the first material Gemma decode win after the graph-reachability cleanup
  - fixed-position profiling is still not a trustworthy absolute harness, but it does confirm the local change in the final chain: `norm_scale_affine_matvec_c1536_r262144_g128_rows4` now shows up at about `777 µs`, while `rms_norm_scale_only_d1536` is effectively free (`7 µs`)
- follow-up on decode affine rows8 tiling for the `1536 -> 256` bucket:
  - added `affine_matvec_c1536_r256_g128_rows8` and proved it exact against the generic affine kernel in local shader tests
  - enabling it in the real Gemma decode selector regressed the clean release benchmark to about `31.10 ms/tok`, `32.2 tok/s`
  - result: the rows8 `1536 -> 256` kernel remains in-tree as research, but `SmeltQwenShapePacks.decodeAffineDecodePipeline` stays on the rows4 selector for the canonical package
- follow-up on decode-side fused K/V projection:
  - routed Gemma sliding/global decode through the generic `fused_dual_affine_matvec` path for `k_proj + v_proj`, which reduced the canonical decode table from `818` ops to `803`
  - local compiler and kernel tests were clean, and the full affine Gemma runtime smoke still passed on the `803`-op package
  - real performance still regressed enough to reject it:
    - decode: `30.46 ms/tok`, `32.8 tok/s`, p95 `41.42 ms`
    - prefill `64/256/512`: `160.0 / 246.6 / 258.8 tok/s`
  - result: the decode K/V fusion wiring was backed back out, and the canonical package remains on the separate K/V projections despite the higher op count
- follow-up on decode per-head RMS threadgroup retuning:
  - tried reducing Gemma decode `per_head_rms_norm*` launches from `256` threads to `128` for `headDim = 256/512`
  - this was easy to reach in the compiler and looked plausible in isolation, but it is not exact under the current Gemma contract
  - the full affine smoke regressed immediately with final-logit drift around `0.03125 ... 0.04492`
  - result: decode per-head RMS launches stay at `min(headDim, 256)` in the canonical package
- follow-up on stripped pre-FFN norm-scale GeGLU fusion:
  - tried a Gemma-only stripped decode path that replaced `pre_feedforward_layernorm -> fused_affine_gate_up_geglu` with `rms_norm_scale_only_d1536 -> norm-scaled GeGLU rows4`
  - the kernel was reachable in the stripped decode graph, and the package still built cleanly at `818` ops, but end-to-end exactness failed
  - the full affine smoke drifted by about `0.037 ... 0.055` on final logits, so this path is not safe to keep
  - result: the experiment was fully backed out and the canonical package was rebuilt back to the last known-good `818`-op / `36.6 ... 37.2 tok/s` baseline
  - fresh restore benchmark on the rebuilt canonical package:
    - decode: `26.86 ms/tok`, `37.2 tok/s`, p95 `37.92 ms`
    - prefill `64/256/512`: `331.8 / 310.0 / 270.1 tok/s`
- root-cause finding from the exactness deep dive:
  - a real debugging-tool bug was also fixed during this pass: `smelt run --debug --dump-trace-label` had been selecting the first matching marker, but batched prefill records the same label once per token. The CLI now defaults to the last matching occurrence and supports `--dump-trace-occurrence first|last|N`, so prefill/decode label comparisons no longer silently read the wrong token.
  - a second trace-capture bug was also fixed: repeated prefill labels were still being read with a "last token row" offset even when the selected marker came from a per-token scratch dispatch that writes row `0`. That made several branch-local Gemma comparisons meaningless.
  - after fixing both trace-read bugs, the earlier `L4.per_layer_gate` theory was false. On both the `full` and `stripped-markers` Gemma packages:
    - `L4.pre_per_layer_branch` is exact
    - `L4.per_layer_gate` is exact
    - `L4.per_layer_int` is exact
    - `L4.per_layer_proj` is exact
    - `L4.post_per_layer_norm` is exact
    - `L4.post_per_layer_residual` is exact
  - the first real stripped-only drift is earlier and smaller than that: `L3.layer_scaled` is the first full-row boundary that diverges on the stripped package, while the full package remains exact.
  - more specifically, the first stripped-only suspect boundary is now `L3.mid`, which is immediately after the post-attention fused `rms_norm_1pw_d1536_add` path. The full package is exact there; the stripped package accumulates small but real drift there at later prompt positions.
  - the shared drift source is not "fused kernels" in general and it is not the `per_head_rms_norm` threadgroup geometry by itself; isolated `128`- vs `256`-thread `per_head_rms_norm` runs matched exactly, both out-of-place and in-place
  - the real contract boundary is `RMSNorm -> consumer`
  - the exact `norm_scale_affine_matvec` family is exact specifically because it reproduces the staged `float -> half -> float` round-trip on the normalized lanes before using them in the affine dot; the kernel explicitly casts `x0...x15` to `half`, writes `normOutput`, and then reloads those rounded `half` values into `float` before the dot product
  - that behavior is now covered by a staged-oracle test in `PrefillKernelTests`: `testNormScaleAffineMatvecGemmaDecodeSpecializations_MatchStagedNormPlusAffine` passed at `0.000000` max norm diff and `0.000000` max output diff for the hot Gemma `1536 -> {2048,12288,262144}` decode shapes
  - the Gemma per-layer gate kernel itself is no longer a plausible suspect:
    - `testAffineMatvecGemmaDecodeSpecializations_MatchGeneric` still passes at `0.000000` max diff for `affine_matvec_c1536_r2048_g128_rows4`
    - the new offset-binding regression `testAffineMatvecGemmaDecodeSpecialization_InputOffsetMatchesDirectBuffer` also passed at `0.000000` max diff, so the prefill `emitMatvecVarSlice(..., inputOffset: ...)` path is numerically identical to binding the same input directly
  - a second diagnostic test now proves why the failed norm-crossing experiments drifted: feeding affine and GeGLU consumers unrounded normalized `float` lanes instead of the staged rounded `half` buffer changes outputs materially even before any end-to-end decode loop is involved
    - affine consumer max diff: `0.031250`
    - GeGLU consumer max diff: `2.000000`
  - reusable rule going forward: any fusion that crosses an RMSNorm boundary must prove exactness against the staged quantized contract, and that contract includes the normalized-lane half round-trip. Raw "same math in float" equivalence is not sufficient.
- one additional exactness note from the same pass: `fused_affine_matvec_add_c6144_r2048_g64` is exact against staged Qwen `affine_matvec -> elementwise_add`, but `fused_affine_matvec_add_c2048_r2048_g64` is still not exact against the staged qmm oracle
- follow-up on the Gemma exactness deep dive:
  - the new dispatch-level `smelt-probe` tooling proved the earlier `L2.pre_per_layer_branch` suspicion was a false lead:
    - on the first bad Gemma token (prefix length `15`), the decode-side `rms_norm_1pw_d1536_add` output matches a CPU staged reference exactly on the real runtime tensors
    - the observed `L2.pre_per_layer_branch` drift was the amplification of a smaller upstream FFN drift, not a bug in the fused norm+add kernel
  - the real first live exactness failure is earlier, at stripped `L2.ffn_int`
    - prefill uses staged `gate -> up -> geglu_fused`
    - decode uses `norm_scale_affine_gate_up_geglu_c1536_r6144_g128_rows4`
    - on the failing prompt, `L2.ffn_int` differs by about `0.001`, `L2.ffn_down` differs by about `0.0005`, and that is enough to become a `~0.039` final-logit divergence on the canonical affine package
  - the canonical package was checked directly with `smelt-probe compare-final-slot --slot 16 --count 262144` on the failing prompt and reproduced the real issue before the fix:
    - `maxDiff=0.0391`, `above0.01=44846`
  - root cause: the stripped pre-FFN `norm_scale_affine_gate_up_geglu` path is not exact enough for Gemma’s quantized contract, even though existing synthetic shader tests only showed sub-`0.002` drift
  - fix kept:
    - disabled `emitStrippedNormScaleGeGLUFFNIfPossible(...)` in the exact stripped Gemma path
    - kept the exact stripped wins that are still safe: fused post-attention/post-FFN `rms_norm_1pw_d1536_add` and fused final `norm_scale_affine_matvec` LM-head
  - revalidation after rebuild:
    - `smelt-probe compare-final-slot` on the same canonical prompt returned `maxDiff=0.0000`, `above0.01=0`
    - `SMELT_RUN_GEMMA_SMOKE=1 swift test --filter testGemmaE2BAffinePromptSweepPrefillMatchesDecodeFinalLogits` passed again (`87.8s`)
  - result: Gemma is back on a trustworthy exact baseline for further performance work; the stripped pre-FFN norm-scale GeGLU path should now be treated as research-only until it can be proven exact end-to-end on real prompts

## Post-Fix Exact Perf Baseline

Fresh benchmark on the rebuilt exact canonical package (`artifacts/gemma4-e2b-affine/google_gemma-4-E2B.smeltpkg`) after removing the non-exact stripped pre-FFN `norm_scale_affine_gate_up_geglu` path:

```bash
bash tools/benchmark.sh artifacts/gemma4-e2b-affine/google_gemma-4-E2B.smeltpkg \
  --decode-iterations 30 --decode-warmup 5 \
  --prefill-iterations 5 --prefill-warmup 2 \
  --prefill-tokens 64,256,512
```

Measured result:

```text
decode-position varying sequential (wrap at 200)
decode-median 32.21ms/tok  31.0 tok/s
decode-p95 47.35ms/tok
decode-pure-gpu-median 31.58ms/tok
decode-pure-gpu-p95 46.72ms/tok
decode-cpu-median 0.328ms/tok
decode-submit-median 0.230ms/tok

prefill-64-median   184.2ms/prefill   347.4 tok/s
prefill-256-median  812.3ms/prefill   315.2 tok/s
prefill-512-median 1843.0ms/prefill   277.8 tok/s
```

Notes:

- this is the real exact baseline, not the earlier faster non-exact stripped GeGLU path
- prefill remains strong; decode is the main remaining problem again
- CPU/read/submit time is negligible relative to GPU time, so further decode wins have to come from kernel/runtime GPU work, not host-side cleanup

Fixed-position kernel bucket profile at `position 128` on the same exact package:

```text
TOTAL: 11048 us
attention_decode_d512_h8_kv1        2742 us   24.8%
fused_affine_gate_up_geglu*         2025 us   18.3% combined
affine_matvec_c12288_r1536_g128      852 us    7.7%
norm_scale_affine_matvec_c1536...    699 us    6.3%
rms_norm_1pw_d1536                   593 us    5.4%
rms_norm_1pw_d1536_add               447 us    4.0%
```

Interpretation:

- guarded attention attribution in `smelt kernels --position ...` is still not fully trustworthy, so the exact attention mix should not be over-read from this profile alone
- even with that caveat, the stable exact-safe decode targets are now clear:
  - `fused_affine_gate_up_geglu_c1536_r6144/_r12288_g128_rows4`
  - `affine_matvec_c12288_r1536_g128_rows4`
  - `norm_scale_affine_matvec_c1536_r262144_g128_rows4`
  - `rms_norm_1pw_d1536` / `rms_norm_1pw_d1536_add`

### LM-Head Rows8 Follow-Up

Kept exact win:

- added exact `rows8` specializations for the tied Gemma LM head:
  - `affine_matvec_c1536_r262144_g128_rows8`
  - `norm_scale_affine_matvec_c1536_r262144_g128_rows8`
- both kernels were proven exact against:
  - the generic decode kernels
  - the staged quantized `rms_norm_1pw_d1536 -> affine_matvec` oracle
- the canonical package now routes the final stripped LM-head path through `norm_scale_affine_matvec_c1536_r262144_g128_rows8`

Measured on real primed decode state with `smelt-probe bench-pipeline`:

```text
previous rows4 LM head: ~700.7 us median
rows8 LM head after keep: 592.9 .. 638.5 us median
```

The range above came from repeated direct dispatch microbenches on the rebuilt canonical package; both measurements were on the live final-logits dispatch, not synthetic shader tests.

Rebuilt exact-package benchmark after keeping the LM-head rows8 path and rejecting the slower down-projection rows8 experiment:

```text
decode-median 31.85ms/tok  31.4 tok/s
decode-p95 46.80ms/tok
decode-pure-gpu-median 31.41ms/tok

prefill-64-median   186.9ms/prefill   342.4 tok/s
prefill-256-median  824.1ms/prefill   310.7 tok/s
prefill-512-median 1852.4ms/prefill   276.4 tok/s
```

Fixed-position kernel profile at `position 128` on the kept package:

```text
TOTAL: 7942 us
fused_affine_gate_up_geglu*         1756 us   22.1% combined
affine_matvec_c12288_r1536_g128      826 us   10.4%
norm_scale_affine_matvec_c1536...    606 us    7.6%
rms_norm_1pw_d1536                   534 us    6.7%
rms_norm_1pw_d1536_add               488 us    6.1%
```

Rejected follow-up:

- tried a `rows8` decode specialization for `affine_matvec_c12288_r1536_g128`
- it remained exact against the generic affine kernel
- real primed-dispatch microbench regressed:
  - rows4 baseline: `~29.63 us` median
  - rows8 candidate: `~30.45 us` median
- result: the `12288 -> 1536` path stays on `rows4` in the canonical package

### Short-Context Attention SDPA Investigation

This is the clearest remaining step-function opportunity.

Current exact Gemma decode still keeps `position + 1 < 128` on the generic attention path. Profiling the exact canonical package at `position 64` showed that this path is catastrophically expensive:

```text
TOTAL: 83412 us
attention_decode: 75999 us (91.1%)
```

That is the first hard proof that the current short-context Gemma attention threshold is masking a much larger win than any remaining affine bucket.

Two lower-threshold experiments were tried:

- `gemmaDecodeSDPASwitchSeqLen = 64`
- `gemmaDecodeSDPASwitchSeqLen = 96`

Both experiments:

- compiled cleanly
- rebuilt the canonical package cleanly
- materially collapsed the `position 64` kernel profile when SDPA was actually used

For the `64` switch, `smelt kernels --position 64` dropped from the `83.4 ms/tok` generic wall above to about:

```text
TOTAL: 8228 us
```

The probe tooling needed one more fix before these chunked-prefill comparisons were trustworthy. `smelt-probe` was still driving prefill as a single `prefillStep` / `debugPrefillStep` call on the full prompt, which is wrong once `max_prefill_batch = 64` forces chunking. That is now fixed in `Sources/SmeltProbe/main.swift`: prefill-sensitive commands replay all prior chunks fully, then inspect the active chunk with the correct `startPos` and row offset.

After that probe fix, the `96` switch still fails on the real 112-token chunked prompt, but with cleaner numbers:

- threshold `96`:
  - `maxDiff=0.025390625`
  - `above0.01=11990`

The key diagnostic from the fixed probe is where the divergence starts. On the failing `SDPA=96` trace-enabled package:

- `L0.out ... L9.out` stay below the real failure threshold
- `L10.out` is the first layer output over `0.01`
- inside `L10`, `q_post_rope`, `k_post_rope`, and the normalized Q/K/V prep are still exact enough
- `L10.attn_raw` is the first clean attention-side label over `0.01`
  - prefill pipeline: `attention_prefill`
  - decode pipeline: `attention_decode_d256_h8_kv1_sdpa`
  - `maxDiff=0.0176`
  - `above0.01=4`

That is the strongest current evidence that the short-context Gemma `d256` SDPA decode path itself is the real mismatch, not the upstream projection or RoPE prep path.

The compiler source has been restored to the known-good `128` threshold after this investigation.

Interpretation:

- the next genuinely large Gemma decode win is not another small affine tile tweak
- it is making the Gemma SDPA decode path numerically exact enough to replace the generic short-context path earlier
- until that is fixed, positions below `128` will keep paying a large attention tax, and that will continue to dominate the full sequential benchmark

#### Follow-Up: Short Sliding Kernel Is Not Yet a Safe Substitute

I also tested the obvious compiler-side escape hatch: route Gemma sliding layers on `position + 1 < 128` through the existing `attention_decode_d256_h8_kv1` kernel instead of the generic `attention_decode` path, while leaving SDPA at `>= 128`.

That looked promising in isolation:

- the direct kernel test against the generic decode kernel could be tightened from `max diff = 0.000122` down to `0.000000`
- the kernel still compiled cleanly and the package rebuilt cleanly

But the real runtime smoke still failed once the compiler actually emitted that path:

- initial variant: all four Gemma prompts failed with `maxDiff ~= 0.035 .. 0.041`
- after tightening the short kernel reduction order again: the full smoke still failed, with the chunked prompt worst at `maxDiff = 0.044921875`

Conclusion:

- `attention_decode_d256_h8_kv1 vs generic` is not a strong enough oracle for exact-mode Gemma routing decisions
- a kernel can look exact in the isolated generic-comparison harness and still be non-exact against the real prefill-vs-decode contract
- for Gemma attention, only the full runtime smoke should be treated as authoritative before changing the emitted decode path

The compiler routing was restored to the known-good generic-short / SDPA-128 baseline after this experiment.

## Relevant Files

- `tools/gemma4-e2b-official-fp16.smelt`
- `tools/build-gemma4-e2b.sh`
- `Sources/SmeltCompiler/GemmaCheckpointAdapter.swift`
- `Sources/SmeltCompiler/TopLevelEmitter.swift`
- `Sources/SmeltCompiler/PrefillEmitter.swift`
- `Sources/SmeltProbe/main.swift`
- `Tests/SmeltRuntimeTests/GemmaSmokeTests.swift`
- `Resources/Shaders/attention.metal`
- `Resources/Shaders/prefill_attention.metal`
- `Resources/Shaders/prefill_rope_kv.metal`
- `Resources/Shaders/norms_fp32.metal`
