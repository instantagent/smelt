#include <metal_stdlib>
using namespace metal;

/// SIMD-parallel full-causal GQA, fp32 — BIT-EXACT replacement for causal_gqa_attn_f32, which
/// runs ONE THREAD per (t, head) (~11K sequential ops each; measured 1.5ms/layer = 42ms of the
/// 30-token talker prefill). One 32-lane SIMD per (t, head) instead:
///   - scores: lanes take keys s = lane + 32·i; each score's inner d-loop is SOURCE-IDENTICAL
///     to the scalar kernel (sequential mul+add over d) → per-score bits unchanged.
///   - mx: fp max is exact under any association → reduction over the staged scores == the
///     scalar kernel's sequential max.
///   - denom: every lane redundantly sums exp(sc[s]-mx) over s ASCENDING — the scalar order.
///   - out: lanes take dims d = lane + 32·j; per d the s-loop is sequential ascending with the
///     SAME e·v mul+add — the scalar kernel's accumulation order per output element.
/// exp() is the same metal::exp in the same metallib → identical results.
///
/// Buffers: 0 q [frames, heads·headDim], 1 k, 2 v [frames, kvHeads·headDim], 3 out
/// Constants: 4 frames, 5 heads, 6 kvHeads, 7 headDim
/// Dispatch: grid (frames·32, heads, 1), threadsPerThreadgroup (32, 1, 1). frames <= 1024
/// (threadgroup score stage; the dispatcher falls back to the scalar kernel beyond).
#define CAUSAL_SIMD_MAX_FRAMES 1024

kernel void causal_gqa_attn_simd_f32(
    device const float* q       [[buffer(0)]],
    device const float* k       [[buffer(1)]],
    device const float* v       [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      frames  [[buffer(4)]],
    constant uint&      heads   [[buffer(5)]],
    constant uint&      kvHeads [[buffer(6)]],
    constant uint&      headDim [[buffer(7)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]]
) {
    threadgroup float sc[CAUSAL_SIMD_MAX_FRAMES];   // dot·scaling per key position
    uint t = gid.x;
    uint qh = gid.y;
    if (t >= frames || qh >= heads) return;

    uint group = heads / kvHeads;
    uint kvh = qh / group;
    uint qDim = heads * headDim;
    uint kvDim = kvHeads * headDim;
    float scaling = 1.0f / sqrt(float(headDim));
    uint qb = t * qDim + qh * headDim;

    // Scores, lane-parallel across key positions; inner loop source-identical to the scalar kernel.
    for (uint s = lane; s <= t; s += 32) {
        uint kb = s * kvDim + kvh * headDim;
        float dot = 0.0f;
        for (uint d = 0; d < headDim; d++) dot += q[qb + d] * k[kb + d];
        sc[s] = dot * scaling;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Max (exact under any order) + the scalar kernel's sequential-ascending denom, replicated
    // per lane (identical fp sequence in every lane — cheap at <=frames adds).
    float mx = -INFINITY;
    for (uint s = 0; s <= t; s++) mx = max(mx, sc[s]);
    float denom = 0.0f;
    for (uint s = 0; s <= t; s++) denom += exp(sc[s] - mx);

    // Output dims lane-parallel; per dim the s-loop is sequential ascending (scalar order).
    uint ob = t * qDim + qh * headDim;
    for (uint d = lane; d < headDim; d += 32) {
        float acc = 0.0f;
        for (uint s = 0; s <= t; s++) {
            float e = exp(sc[s] - mx);
            acc += e * v[s * kvDim + kvh * headDim + d];
        }
        out[ob + d] = acc / denom;
    }
}

/// Start-position-aware fp32 GQA — the cross-chunk prefill sibling of
/// causal_gqa_attn_simd_f32 (docs/talker-trunk-crosschunk-prefill-plan.md, B3.2b).
/// A chunk of `frames` queries at absolute base `startPos` attends the FULL KV cache
/// rows [0, startPos + localQuery]. The cache is the row-contiguous [rows, kvDim] fp32
/// trunk ABI; prior chunks (committed earlier on the same queue) populated rows
/// [0, startPos), this chunk's K/V-write dispatches populated [startPos, startPos+frames).
///
/// BIT-EXACT to causal_gqa_attn_simd_f32 at startPos == 0 (absT == t, identical loops)
/// — preserves the W2 single-chunk prefill parity gate. For startPos > 0 the score,
/// max, denom, and output loops are SOURCE-IDENTICAL, only ranging over absolute
/// [0, absT] instead of [0, t]; q/out stay chunk-local (q, out are chunk slabs), k/v
/// index absolute s (the full cache). exp() is the same metal::exp in the same metallib.
///
/// Buffers: 0 q [frames, heads·headDim] (chunk-local), 1 kCache, 2 vCache
/// [rows, kvHeads·headDim] (full), 3 out [frames, heads·headDim] (chunk-local)
/// Constants: 4 frames(=chunkLen), 5 heads, 6 kvHeads, 7 headDim, 8 startPos
/// Dispatch: grid (frames·32, heads, 1), tg (32,1,1). startPos + frames <= 2048.
#define CAUSAL_CACHED_MAX_LEN 2048

kernel void causal_gqa_attn_cached_f32(
    device const float* q       [[buffer(0)]],
    device const float* k       [[buffer(1)]],
    device const float* v       [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      frames  [[buffer(4)]],
    constant uint&      heads   [[buffer(5)]],
    constant uint&      kvHeads [[buffer(6)]],
    constant uint&      headDim [[buffer(7)]],
    constant uint&      startPos [[buffer(8)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]]
) {
    threadgroup float sc[CAUSAL_CACHED_MAX_LEN];   // dot·scaling per absolute key position
    uint t = gid.x;          // chunk-local query index
    uint qh = gid.y;
    if (t >= frames || qh >= heads) return;

    uint absT = startPos + t;   // absolute position of this query
    uint group = heads / kvHeads;
    uint kvh = qh / group;
    uint qDim = heads * headDim;
    uint kvDim = kvHeads * headDim;
    float scaling = 1.0f / sqrt(float(headDim));
    uint qb = t * qDim + qh * headDim;   // q is chunk-local

    // Scores over absolute key positions [0, absT]; lane-parallel; inner loop
    // source-identical to the scalar kernel. k is the full cache (absolute s).
    for (uint s = lane; s <= absT; s += 32) {
        uint kb = s * kvDim + kvh * headDim;
        float dot = 0.0f;
        for (uint d = 0; d < headDim; d++) dot += q[qb + d] * k[kb + d];
        sc[s] = dot * scaling;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Max (exact under any order) + the scalar kernel's sequential-ascending denom.
    float mx = -INFINITY;
    for (uint s = 0; s <= absT; s++) mx = max(mx, sc[s]);
    float denom = 0.0f;
    for (uint s = 0; s <= absT; s++) denom += exp(sc[s] - mx);

    // Output dims lane-parallel; per dim the s-loop is sequential ascending. out is
    // chunk-local; v is the full cache (absolute s).
    uint ob = t * qDim + qh * headDim;
    for (uint d = lane; d < headDim; d += 32) {
        float acc = 0.0f;
        for (uint s = 0; s <= absT; s++) {
            float e = exp(sc[s] - mx);
            acc += e * v[s * kvDim + kvh * headDim + d];
        }
        out[ob + d] = acc / denom;
    }
}

/// Start-position-aware SCALAR streaming GQA — the UNCAPPED sibling of
/// causal_gqa_attn_cached_f32 (Phase 4 U1, variable-length compiled prefill). Same
/// cross-chunk semantics (chunk-local q/out at absolute base startPos; full KV cache
/// k/v indexed absolutely), but ONE THREAD per (t, head) that streams keys in two
/// passes — max, then a combined denom + weighted-V — recomputing each dot and
/// holding only acc[headDim] in registers. NO threadgroup score array, so NO length
/// cap (the sc[2048] in causal_gqa_attn_cached_f32 was the ONLY cap).
///
/// NUMERICALLY EQUIVALENT to causal_gqa_attn_cached_f32 (~1e-6, <=~14 ULP) — NOT
/// byte-identical: this kernel contracts `dot*scaling - mx` into one FMA, while the SIMD
/// kernel stores `dot*scaling` to sc[s] then subtracts mx (no FMA across the store). Same
/// reassociation class as the non-cached scalar-vs-SIMD pair. It IS the same algorithm
/// (a full first-pass max, then denom + each acc[d] accumulated over s strictly ASCENDING)
/// — NOT flash attention (no online accumulator rescale). Used ONLY for the genuinely
/// over-cap (cache > 2048) chunks the SIMD kernel cannot represent; the <=2048 path stays
/// on the SIMD kernel, so no byte-exact gate is perturbed.
///
/// Buffers: 0 q [frames, heads·headDim] (chunk-local), 1 kCache, 2 vCache
/// [rows, kvHeads·headDim] (full), 3 out [frames, heads·headDim] (chunk-local)
/// Constants: 4 frames(=chunkLen), 5 heads, 6 kvHeads, 7 headDim, 8 startPos
/// Dispatch: grid (frames, heads, 1), tg (1,1,1) — one thread per (t,head). Uncapped in
/// sequence length; headDim <= 128 (the acc[] register array — the F32 talker trunk guard).
/// Runs correctly on the SIMD kernel's (frames*32, heads) over-launch grid too (lanes with
/// t >= frames return early), which is what the runtime substitution dispatches.
kernel void causal_gqa_attn_cached_scalar_f32(
    device const float* q       [[buffer(0)]],
    device const float* k       [[buffer(1)]],
    device const float* v       [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      frames  [[buffer(4)]],
    constant uint&      heads   [[buffer(5)]],
    constant uint&      kvHeads [[buffer(6)]],
    constant uint&      headDim [[buffer(7)]],
    constant uint&      startPos [[buffer(8)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint t = tid.x;          // chunk-local query index
    uint qh = tid.y;
    if (t >= frames || qh >= heads) return;

    uint absT = startPos + t;   // absolute position of this query
    uint group = heads / kvHeads;
    uint kvh = qh / group;
    uint qDim = heads * headDim;
    uint kvDim = kvHeads * headDim;
    float scaling = 1.0f / sqrt(float(headDim));
    uint qb = t * qDim + qh * headDim;   // q is chunk-local

    // Pass 1: global max over absolute key positions [0, absT] (fp max is exact).
    float mx = -INFINITY;
    for (uint s = 0; s <= absT; s++) {
        uint kb = s * kvDim + kvh * headDim;
        float dot = 0.0f;
        for (uint d = 0; d < headDim; d++) dot += q[qb + d] * k[kb + d];
        mx = max(mx, dot * scaling);
    }

    // Pass 2: denom + weighted-V, ascending s; only acc[headDim] in registers.
    float acc[128];  // headDim <= 128 (F32TalkerTrunkGuards)
    for (uint d = 0; d < headDim; d++) acc[d] = 0.0f;
    float denom = 0.0f;
    for (uint s = 0; s <= absT; s++) {
        uint kb = s * kvDim + kvh * headDim;
        float dot = 0.0f;
        for (uint d = 0; d < headDim; d++) dot += q[qb + d] * k[kb + d];
        float e = exp(dot * scaling - mx);
        denom += e;
        uint vb = s * kvDim + kvh * headDim;
        for (uint d = 0; d < headDim; d++) acc[d] += e * v[vb + d];
    }

    uint ob = t * qDim + qh * headDim;   // out is chunk-local
    for (uint d = 0; d < headDim; d++) out[ob + d] = acc[d] / denom;
}
