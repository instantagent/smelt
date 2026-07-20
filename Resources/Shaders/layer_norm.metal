#include <metal_stdlib>
using namespace metal;

/// Standard LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
/// Uses simd_sum + cross-SIMD reduction (same pattern as norms.metal).
/// 2-pass: (1) sum + sum_sq, (2) normalize + affine.
///
/// Buffers:
///   0: input   [rows, dim]  FP16
///   1: weight  [dim]        FP16 (gamma)
///   2: bias    [dim]        FP16 (beta)
///   3: output  [rows, dim]  FP16
/// Constants:
///   0: dim     uint
///   1: eps     float
kernel void layer_norm(
    device const half* input   [[buffer(0)]],
    device const half* weight  [[buffer(1)]],
    device const half* bias    [[buffer(2)]],
    device half*       output  [[buffer(3)]],
    constant uint&     dim     [[buffer(4)]],
    constant float&    eps     [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint base = row * dim;

    // Pass 1: accumulate sum and sum-of-squares in one read
    float local_sum = 0;
    float local_sq = 0;
    for (uint i = tid; i < dim; i += tg_size) {
        float x = float(input[base + i]);
        local_sum += x;
        local_sq += x * x;
    }

    // SIMD reduction (hardware single-cycle on Apple Silicon)
    local_sum = simd_sum(local_sum);
    local_sq = simd_sum(local_sq);

    // Cross-SIMD reduction via threadgroup memory (one entry per SIMD group)
    threadgroup float partial_sum[32];
    threadgroup float partial_sq[32];
    if (simd_lane == 0) {
        partial_sum[simd_group] = local_sum;
        partial_sq[simd_group] = local_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 accumulates across SIMD groups, then broadcasts mean + inv_std back through the
    // already-allocated partial arrays (slots [0..1], consumed above) — a threadgroup ARRAY write
    // avoids the -Wsometimes-uninitialized false positive a conditionally-written scalar triggers.
    if (tid == 0) {
        float total_sum = 0;
        float total_sq = 0;
        uint num_groups = (tg_size + 31) / 32;
        for (uint s = 0; s < num_groups; s++) {
            total_sum += partial_sum[s];
            total_sq += partial_sq[s];
        }
        float mean = total_sum / float(dim);
        float var = total_sq / float(dim) - mean * mean;
        partial_sum[0] = mean;
        partial_sq[0] = rsqrt(var + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: normalize + affine
    float mean = partial_sum[0];
    float inv_std = partial_sq[0];
    for (uint i = tid; i < dim; i += tg_size) {
        float x = float(input[base + i]);
        float norm = (x - mean) * inv_std;
        float y = norm * float(weight[i]) + float(bias[i]);
        output[base + i] = half(y);
    }
}
