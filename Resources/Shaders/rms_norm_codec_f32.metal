#include <metal_stdlib>
using namespace metal;

/// Codec RMSNorm, fp32, on [frames, dim] (sequence-major): per frame, normalize over the
/// feature dim and scale by weight DIRECTLY (y = x * rsqrt(mean(x^2)+eps) * w[i]). This is
/// the codec convention — NOT the LLM `(1 + weight)` form — so it has its own kernel.
/// Matches Qwen3TTSCodec.rmsNorm (eps 1e-5).
///
/// ONE THREADGROUP (32-lane SIMD) PER FRAME: each lane strides the feature dim by 32 (coalesced
/// reads), a single `simd_sum` reduces the 32 partial sums-of-squares (no barriers), then all lanes
/// write the normalized row strided. The prior one-thread-per-frame form collapsed to a single thread
/// at decode (frames=1), serially reducing a 2048-wide row with unhidden memory latency (~55× slower
/// than occupied). Reassociates the reduction (tree vs sequential) → fp32-equivalent, rides the
/// cosine/relL2 gate. Dispatch with grid = frames*32 threads, threadsPerThreadgroup = 32.
///
/// Buffers: 0 input [frames,dim], 1 weight [dim], 2 output [frames,dim]
/// Constants: 3 frames, 4 dim, 5 eps (float)
kernel void rms_norm_codec_f32(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant uint&      frames [[buffer(3)]],
    constant uint&      dim    [[buffer(4)]],
    constant float&     eps    [[buffer(5)]],
    uint t    [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (t >= frames) return;
    uint base = t * dim;

    float partial = 0.0f;
    for (uint i = lane; i < dim; i += 32) { float v = input[base + i]; partial += v * v; }
    float ms = simd_sum(partial) / float(dim);
    float inv = 1.0f / sqrt(ms + eps);

    for (uint i = lane; i < dim; i += 32) output[base + i] = input[base + i] * inv * weight[i];
}

/// BF16-scale sibling of `rms_norm_codec_f32`. The scale tensor stays in its
/// authoritative checkpoint representation and is widened exactly at the load;
/// every reduction and arithmetic expression otherwise remains byte-for-byte
/// the FP32-scale route.
kernel void rms_norm_codec_bf16w_f32(
    device const float*  input  [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float*        output [[buffer(2)]],
    constant uint&       frames [[buffer(3)]],
    constant uint&       dim    [[buffer(4)]],
    constant float&      eps    [[buffer(5)]],
    uint t    [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (t >= frames) return;
    uint base = t * dim;

    float partial = 0.0f;
    for (uint i = lane; i < dim; i += 32) { float v = input[base + i]; partial += v * v; }
    float ms = simd_sum(partial) / float(dim);
    float inv = 1.0f / sqrt(ms + eps);

    for (uint i = lane; i < dim; i += 32) output[base + i] = input[base + i] * inv * float(weight[i]);
}
