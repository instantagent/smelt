#include <metal_stdlib>
using namespace metal;

/// Per-head RMSNorm + RoPE fused for an M=1 decode step (2 dispatches → 1). One 32-lane SIMD per
/// head: the norm uses rms_norm_head_f32's EXACT reduction (lane-strided x² partials, one
/// simd_sum, `1/sqrt(ms+eps)`); each rotation pair then reads a/b built with the norm kernel's
/// exact per-element expression (x·inv·w[d]) and applies rope_apply_f32's exact output
/// expressions. Bit-identical to rms_norm_head_f32 → rope_apply_f32 at frames=1. `out` may be
/// bound at an offset (the post-RoPE K straight into its KV-cache row).
///
/// Buffers: 0 x [heads·headDim] float, 1 normW [headDim] float, 2 cos [headDim] float,
///          3 sin [headDim] float, 4 out [heads·headDim] float
/// Constants: 5 heads, 6 headDim, 7 eps (float)
/// Dispatch: grid (heads·32, 1, 1), threadsPerThreadgroup (32, 1, 1). headDim even.
kernel void head_norm_rope_f32(
    device const float* x       [[buffer(0)]],
    device const float* normW   [[buffer(1)]],
    device const float* cosT    [[buffer(2)]],
    device const float* sinT    [[buffer(3)]],
    device float*       out     [[buffer(4)]],
    constant uint&      heads   [[buffer(5)]],
    constant uint&      headDim [[buffer(6)]],
    constant float&     eps     [[buffer(7)]],
    uint hd   [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (hd >= heads) return;
    uint base = hd * headDim;
    // rms_norm_head_f32's reduction, verbatim (frames=1 → tgid == hd).
    float partial = 0.0f;
    for (uint d = lane; d < headDim; d += 32) { float v = x[base + d]; partial += v * v; }
    float ms = simd_sum(partial) / float(headDim);
    float inv = 1.0f / sqrt(ms + eps);

    uint halfDim = headDim / 2;
    for (uint j = lane; j < halfDim; j += 32) {
        // a/b == rms_norm_head_f32's x·inv·w per element; outputs == rope_apply_f32 (t=0 row).
        float a = x[base + j] * inv * normW[j];
        float b = x[base + j + halfDim] * inv * normW[j + halfDim];
        out[base + j]           = a * cosT[j]           - b * sinT[j];
        out[base + j + halfDim] = b * cosT[j + halfDim] + a * sinT[j + halfDim];
    }
}

/// BF16-scale sibling of `head_norm_rope_f32`; the normalization reduction and
/// RoPE expressions are deliberately identical after exact BF16-to-FP32 scale
/// widening.
kernel void head_norm_rope_bf16w_f32(
    device const float*  x       [[buffer(0)]],
    device const bfloat* normW   [[buffer(1)]],
    device const float*  cosT    [[buffer(2)]],
    device const float*  sinT    [[buffer(3)]],
    device float*        out     [[buffer(4)]],
    constant uint&       heads   [[buffer(5)]],
    constant uint&       headDim [[buffer(6)]],
    constant float&      eps     [[buffer(7)]],
    uint hd   [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (hd >= heads) return;
    uint base = hd * headDim;
    float partial = 0.0f;
    for (uint d = lane; d < headDim; d += 32) { float v = x[base + d]; partial += v * v; }
    float ms = simd_sum(partial) / float(headDim);
    float inv = 1.0f / sqrt(ms + eps);

    uint halfDim = headDim / 2;
    for (uint j = lane; j < halfDim; j += 32) {
        float a = x[base + j] * inv * float(normW[j]);
        float b = x[base + j + halfDim] * inv * float(normW[j + halfDim]);
        out[base + j]           = a * cosT[j]           - b * sinT[j];
        out[base + j + halfDim] = b * cosT[j + halfDim] + a * sinT[j + halfDim];
    }
}
