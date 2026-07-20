#include <metal_stdlib>
using namespace metal;

// ─── FP32 verification variants ───
// Identical math to the FP16 versions but with float inputs/outputs.
// Used to verify the algorithm is correct (maxDiff should be < 1e-5).
// FP16 precision loss is measured separately.

kernel void rms_norm_1pw_fp32(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float*       output  [[buffer(2)]],
    constant uint&      dim     [[buffer(3)]],
    constant float&     eps     [[buffer(4)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgs        [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    float sumSq = 0.0f;
    for (uint i = tid; i < dim; i += tgs) {
        float v = input[i];
        sumSq += v * v;
    }

    sumSq = simd_sum(sumSq);

    threadgroup float partial[32];
    if (simd_lane == 0) { partial[simd_group] = sumSq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_rsqrt = 0.0f;
    if (tid == 0) {
        float total = 0;
        for (uint s = 0; s < tgs / 32; s++) { total += partial[s]; }
        float mean = total / float(dim);
        shared_rsqrt = rsqrt(mean + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;

    for (uint i = tid; i < dim; i += tgs) {
        float x = input[i];
        float w = weight[i];
        output[i] = x * rs * (1.0f + w);
    }
}

kernel void rms_norm_1pw_from_fp32(
    device const float* input   [[buffer(0)]],
    device const half*  weight  [[buffer(1)]],
    device half*        output  [[buffer(2)]],
    constant uint&      dim     [[buffer(3)]],
    constant float&     eps     [[buffer(4)]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgs        [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    float sumSq = 0.0f;
    for (uint i = tid; i < dim; i += tgs) {
        float v = input[i];
        sumSq += v * v;
    }

    sumSq = simd_sum(sumSq);

    threadgroup float partial[32];
    if (simd_lane == 0) { partial[simd_group] = sumSq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_rsqrt = 0.0f;
    if (tid == 0) {
        float total = 0;
        for (uint s = 0; s < tgs / 32; s++) { total += partial[s]; }
        float mean = total / float(dim);
        shared_rsqrt = rsqrt(mean + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;
    for (uint i = tid; i < dim; i += tgs) {
        float x = input[i];
        float w = float(weight[i]);
        output[i] = half(clamp(x * rs * (1.0f + w), -65504.0f, 65504.0f));
    }
}

kernel void rms_norm_1pw_from_fp32_batched(
    device const float* input   [[buffer(0)]],  // [B, dim]
    device const half*  weight  [[buffer(1)]],  // [dim]
    device half*        output  [[buffer(2)]],  // [B, dim]
    constant uint&      dim     [[buffer(3)]],
    constant float&     eps     [[buffer(4)]],
    uint batch      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgs        [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint offset = batch * dim;

    float sumSq = 0.0f;
    for (uint i = tid; i < dim; i += tgs) {
        float v = input[offset + i];
        sumSq += v * v;
    }

    sumSq = simd_sum(sumSq);

    threadgroup float partial[32];
    if (simd_lane == 0) { partial[simd_group] = sumSq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_rsqrt = 0.0f;
    if (tid == 0) {
        float total = 0;
        for (uint s = 0; s < tgs / 32; s++) { total += partial[s]; }
        float mean = total / float(dim);
        shared_rsqrt = rsqrt(mean + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;
    for (uint i = tid; i < dim; i += tgs) {
        float x = input[offset + i];
        float w = float(weight[i]);
        output[offset + i] = half(clamp(x * rs * (1.0f + w), -65504.0f, 65504.0f));
    }
}

kernel void rms_norm_gated_fp32(
    device const float* input   [[buffer(0)]],
    device const float* gate    [[buffer(1)]],
    device const float* weight  [[buffer(2)]],
    device float*       output  [[buffer(3)]],
    constant uint&      headDim [[buffer(4)]],
    constant float&     eps     [[buffer(5)]],
    uint head       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgs        [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint offset = head * headDim;

    float sumSq = 0.0f;
    for (uint i = tid; i < headDim; i += tgs) {
        float v = input[offset + i];
        sumSq += v * v;
    }
    sumSq = simd_sum(sumSq);

    threadgroup float partial[32];
    if (simd_lane == 0) { partial[simd_group] = sumSq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_rsqrt = 0.0f;
    if (tid == 0) {
        float total = 0;
        for (uint s = 0; s < tgs / 32; s++) { total += partial[s]; }
        float mean = total / float(headDim);
        shared_rsqrt = rsqrt(mean + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;

    for (uint i = tid; i < headDim; i += tgs) {
        float x = input[offset + i];
        float g = gate[offset + i];
        float w = weight[i];
        float silu_g = g / (1.0f + exp(-g));
        output[offset + i] = w * (x * rs) * silu_g;
    }
}

kernel void l2_normalize_fp32(
    device float*       data    [[buffer(0)]],
    constant uint&      headDim [[buffer(1)]],
    constant float&     eps     [[buffer(2)]],
    uint head       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgs        [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint offset = head * headDim;

    float sumSq = 0.0f;
    for (uint i = tid; i < headDim; i += tgs) {
        float v = data[offset + i];
        sumSq += v * v;
    }
    sumSq = simd_sum(sumSq);

    threadgroup float partial[32];
    if (simd_lane == 0) { partial[simd_group] = sumSq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_scale = 0.0f;
    if (tid == 0) {
        float total = 0;
        for (uint s = 0; s < tgs / 32; s++) { total += partial[s]; }
        float norm = sqrt(total);
        shared_scale = 1.0f / max(norm, eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float scale = shared_scale;
    for (uint i = tid; i < headDim; i += tgs) {
        data[offset + i] *= scale;
    }
}
