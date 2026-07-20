#include <metal_stdlib>
using namespace metal;

/// Full-causal grouped-query attention, fp32. q is [frames, heads*headDim]; k/v are
/// [frames, kvHeads*headDim]. Query head qh uses kv head qh/(heads/kvHeads). Query t attends
/// to all keys 0..t (full causal, NO window). Two-pass softmax (max, then denom+output) so
/// there is no fixed-size scratch — safe for arbitrarily long/growing sequences. Matches the
/// attention in Qwen3TTSTalker.forward (scaling = 1/sqrt(headDim)). One thread per (head, t).
///
/// Buffers: 0 q, 1 k, 2 v, 3 out
/// Constants: 4 frames, 5 heads, 6 kvHeads, 7 headDim
kernel void causal_gqa_attn_f32(
    device const float* q       [[buffer(0)]],
    device const float* k       [[buffer(1)]],
    device const float* v       [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      frames  [[buffer(4)]],
    constant uint&      heads   [[buffer(5)]],
    constant uint&      kvHeads [[buffer(6)]],
    constant uint&      headDim [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint t = tid.x;
    uint qh = tid.y;
    if (t >= frames || qh >= heads) return;

    uint group = heads / kvHeads;
    uint kvh = qh / group;
    uint qDim = heads * headDim;
    uint kvDim = kvHeads * headDim;
    float scaling = 1.0f / sqrt(float(headDim));
    uint qb = t * qDim + qh * headDim;

    float mx = -INFINITY;
    for (uint s = 0; s <= t; s++) {
        uint kb = s * kvDim + kvh * headDim;
        float dot = 0.0f;
        for (uint d = 0; d < headDim; d++) dot += q[qb + d] * k[kb + d];
        mx = max(mx, dot * scaling);
    }

    float acc[128];  // headDim <= 128
    for (uint d = 0; d < headDim; d++) acc[d] = 0.0f;
    float denom = 0.0f;
    for (uint s = 0; s <= t; s++) {
        uint kb = s * kvDim + kvh * headDim;
        float dot = 0.0f;
        for (uint d = 0; d < headDim; d++) dot += q[qb + d] * k[kb + d];
        float e = exp(dot * scaling - mx);
        denom += e;
        uint vb = s * kvDim + kvh * headDim;
        for (uint d = 0; d < headDim; d++) acc[d] += e * v[vb + d];
    }

    uint ob = t * qDim + qh * headDim;
    for (uint d = 0; d < headDim; d++) out[ob + d] = acc[d] / denom;
}
