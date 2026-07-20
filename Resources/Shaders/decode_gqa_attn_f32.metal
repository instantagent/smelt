#include <metal_stdlib>
using namespace metal;

/// Single-query grouped-query attention for KV-cache decode, fp32. q is one row
/// [heads*headDim]; k/v are the cache [cacheLen, kvHeads*headDim]. The query (the newest
/// token) attends to ALL cacheLen keys — no causal mask, because it is the last position.
/// Query head qh uses kv head qh/(heads/kvHeads). Two-pass softmax (max, then denom+output),
/// scaling 1/sqrt(headDim).
///
/// ONE THREADGROUP (32-lane SIMD) PER QUERY HEAD: q is staged into threadgroup memory once, then
/// the 32 lanes split the cacheLen keys (lane l owns keys l, l+32, …) so the K/V global reads issue
/// in parallel and their latency is hidden — the prior one-thread-per-head form scanned the whole
/// cache serially per head (~1.8ms/dispatch, the dominant decode cost). `simd_max`/`simd_sum` reduce
/// the per-lane max, denom, and per-dim V-accumulation across the SIMD (no barriers beyond the one
/// q-stage). Reassociates the softmax sums (tree vs sequential) → fp32-equivalent, rides the
/// cosine/relL2 gate. Dispatch grid = heads*32 threads, threadsPerThreadgroup = 32.
///
/// Buffers: 0 q [heads*headDim], 1 k [cacheLen,kvHeads*headDim], 2 v, 3 out [heads*headDim]
/// Constants: 4 cacheLen, 5 heads, 6 kvHeads, 7 headDim
kernel void decode_gqa_attn_f32(
    device const float* q        [[buffer(0)]],
    device const float* k        [[buffer(1)]],
    device const float* v        [[buffer(2)]],
    device float*       out      [[buffer(3)]],
    constant uint&      cacheLen [[buffer(4)]],
    constant uint&      heads    [[buffer(5)]],
    constant uint&      kvHeads  [[buffer(6)]],
    constant uint&      headDim  [[buffer(7)]],
    uint qh   [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (qh >= heads) return;

    uint group = heads / kvHeads;
    uint kvh = qh / group;
    uint kvDim = kvHeads * headDim;
    float scaling = 1.0f / sqrt(float(headDim));
    uint qb = qh * headDim;

    // Stage q[head] into threadgroup memory once (avoids re-reading it from global per key).
    threadgroup float qs[128];  // headDim <= 128
    for (uint d = lane; d < headDim; d += 32) qs[d] = q[qb + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 1: per-key scaled dot, max across all keys (lanes split the keys, then simd_max). When the
    // lane's key count fits SCORE_CAP, the scaled dots are cached so pass 2 reuses them BIT-IDENTICALLY
    // (no K re-read / re-dot); longer caches fall back to recompute. lane owns keys lane, lane+32, …, so
    // it holds ceil(cacheLen/32) scores — capped at SCORE_CAP (= 256 keys).
    const uint SCORE_CAP = 8;
    float scores[SCORE_CAP];
    bool cacheScores = cacheLen <= SCORE_CAP * 32;
    float localMax = -INFINITY;
    uint i = 0;
    for (uint s = lane; s < cacheLen; s += 32, i++) {
        uint kb = s * kvDim + kvh * headDim;
        float dot = 0.0f;
        for (uint d = 0; d < headDim; d++) dot += qs[d] * k[kb + d];
        float sc = dot * scaling;
        if (cacheScores) scores[i] = sc;
        localMax = max(localMax, sc);
    }
    float mx = simd_max(localMax);

    // Pass 2: denom + weighted-V accumulation (lanes split keys, each keeps a private acc[headDim]).
    float acc[128];
    for (uint d = 0; d < headDim; d++) acc[d] = 0.0f;
    float denom = 0.0f;
    uint j = 0;
    for (uint s = lane; s < cacheLen; s += 32, j++) {
        uint kb = s * kvDim + kvh * headDim;
        float sc;
        if (cacheScores) {
            sc = scores[j];
        } else {
            float dot = 0.0f;
            for (uint d = 0; d < headDim; d++) dot += qs[d] * k[kb + d];
            sc = dot * scaling;
        }
        float e = exp(sc - mx);
        denom += e;
        for (uint d = 0; d < headDim; d++) acc[d] += e * v[kb + d];
    }
    denom = simd_sum(denom);
    for (uint d = 0; d < headDim; d++) {
        float a = simd_sum(acc[d]);
        if (lane == 0) out[qb + d] = a / denom;
    }
}
