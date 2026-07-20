#include <metal_stdlib>
using namespace metal;

/// Per-head RMSNorm, fp32, on a [frames, heads*headDim] buffer: for each (t, head),
/// normalize over headDim and scale by weight[d] DIRECTLY (x * rsqrt(mean(x^2)+eps) * w[d]).
/// The talker's q_norm/k_norm (applied per head, before RoPE). Matches Qwen3TTSTalker.headRMSNorm.
///
/// ONE THREADGROUP (32-lane SIMD) PER (frame, head): lanes stride headDim coalesced, one `simd_sum`
/// reduces the partial sums-of-squares, all lanes write the normalized head strided. The prior
/// one-thread-per-(frame,head) form ran a single thread per head at decode (frames=1) over the whole
/// headDim with unhidden memory latency. Reassociates the reduction → fp32-equivalent (cosine/relL2
/// gate). The (frame,head) pair is row-major in threadgroup_position_in_grid (= t*heads + h), so
/// `base = tgid*headDim` matches the [frames, heads*headDim] layout. Grid = frames*heads*32, tg = 32.
///
/// Buffers: 0 x [frames, heads*headDim], 1 weight [headDim], 2 out
/// Constants: 3 frames, 4 heads, 5 headDim, 6 eps (float)
kernel void rms_norm_head_f32(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float*       out     [[buffer(2)]],
    constant uint&      frames  [[buffer(3)]],
    constant uint&      heads   [[buffer(4)]],
    constant uint&      headDim [[buffer(5)]],
    constant float&     eps     [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (tgid >= frames * heads) return;
    uint base = tgid * headDim;

    float partial = 0.0f;
    for (uint d = lane; d < headDim; d += 32) { float v = x[base + d]; partial += v * v; }
    float ms = simd_sum(partial) / float(headDim);
    float inv = 1.0f / sqrt(ms + eps);

    for (uint d = lane; d < headDim; d += 32) out[base + d] = x[base + d] * inv * weight[d];
}

/// BF16-scale sibling of `rms_norm_head_f32`; only the scale load widens from
/// BF16. The row reduction and normalized output expression are unchanged.
kernel void rms_norm_head_bf16w_f32(
    device const float*  x       [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device float*        out     [[buffer(2)]],
    constant uint&       frames  [[buffer(3)]],
    constant uint&       heads   [[buffer(4)]],
    constant uint&       headDim [[buffer(5)]],
    constant float&      eps     [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (tgid >= frames * heads) return;
    uint base = tgid * headDim;

    float partial = 0.0f;
    for (uint d = lane; d < headDim; d += 32) { float v = x[base + d]; partial += v * v; }
    float ms = simd_sum(partial) / float(headDim);
    float inv = 1.0f / sqrt(ms + eps);

    for (uint d = lane; d < headDim; d += 32) out[base + d] = x[base + d] * inv * float(weight[d]);
}
