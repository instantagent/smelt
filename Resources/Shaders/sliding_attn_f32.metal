#include <metal_stdlib>
using namespace metal;

/// Sliding-window causal multi-head attention, fp32. q/k/v are [frames, heads*headDim]
/// (sequence-major, RoPE already applied to q/k). Query t attends to keys in
/// (t-window, t] (inclusive), softmax, weighted sum of v. Full MHA (each query head uses
/// its own kv head). One thread per (head, query t). Matches the attention in
/// Qwen3TTSCodec.preTransformer (scaling = 1/sqrt(headDim)).
///
/// Buffers: 0 q, 1 k, 2 v, 3 out  (all [frames, heads*headDim])
/// Constants: 4 frames, 5 heads, 6 headDim, 7 window
kernel void sliding_attn_f32(
    device const float* q       [[buffer(0)]],
    device const float* k       [[buffer(1)]],
    device const float* v       [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      frames  [[buffer(4)]],
    constant uint&      heads   [[buffer(5)]],
    constant uint&      headDim [[buffer(6)]],
    constant uint&      window  [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint t = tid.x;   // query position
    uint hd = tid.y;  // head
    if (t >= frames || hd >= heads) return;

    uint attnDim = heads * headDim;
    uint qb = t * attnDim + hd * headDim;
    float scaling = 1.0f / sqrt(float(headDim));
    uint lo = (t + 1 > window) ? (t + 1 - window) : 0u;

    float sc[72];  // window <= 72
    float mx = -INFINITY;
    for (uint s = lo; s <= t; s++) {
        uint kb = s * attnDim + hd * headDim;
        float dot = 0.0f;
        for (uint d = 0; d < headDim; d++) dot += q[qb + d] * k[kb + d];
        float v0 = dot * scaling;
        sc[s - lo] = v0;
        mx = max(mx, v0);
    }
    float denom = 0.0f;
    for (uint s = lo; s <= t; s++) { float e = exp(sc[s - lo] - mx); sc[s - lo] = e; denom += e; }

    uint ob = t * attnDim + hd * headDim;
    for (uint d = 0; d < headDim; d++) {
        float acc = 0.0f;
        for (uint s = lo; s <= t; s++) acc += (sc[s - lo] / denom) * v[s * attnDim + hd * headDim + d];
        out[ob + d] = acc;
    }
}
