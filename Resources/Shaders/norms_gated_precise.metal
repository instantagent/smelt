#include <metal_stdlib>
using namespace metal;

// Qwen DeltaNet's graph materializes RMSNorm in fp16, then evaluates SiLU
// and the final product in fp32 before casting back to fp16. Safe-math
// compilation is required to preserve that graph-owned arithmetic topology.
inline half rms_norm_gated_d128_value(
    half input,
    half gate,
    half weight,
    float inverseRms
) {
    const half normalized = weight * half(float(input) * inverseRms);
    const float g = float(gate);
    const float tail = 1.0f / (1.0f + metal::precise::exp(abs(g)));
    const float sigmoid = g < 0.0f ? tail : 1.0f - tail;
    return half(float(normalized) * (g * sigmoid));
}

kernel void rms_norm_gated_d128(
    device const half* input  [[buffer(0)]],
    device const half* gate   [[buffer(1)]],
    device const half* weight [[buffer(2)]],
    device half*       output [[buffer(3)]],
    uint head       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 128;
    constexpr uint N_READS = 4;
    constexpr float eps = 1e-6f;
    const uint offset = head * D + tid * N_READS;

    float sumSq = 0.0f;
    for (uint i = 0; i < N_READS; ++i) {
        const float x = float(input[offset + i]);
        sumSq += x * x;
    }
    sumSq = simd_sum(sumSq);

    threadgroup float inverseRms;
    if (simd_lane == 0) {
        inverseRms = metal::precise::rsqrt(sumSq / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < N_READS; ++i) {
        output[offset + i] = rms_norm_gated_d128_value(
            input[offset + i], gate[offset + i], weight[tid * N_READS + i], inverseRms
        );
    }
}

kernel void rms_norm_gated_d128_batched(
    device const half* input     [[buffer(0)]],
    device const half* gate      [[buffer(1)]],
    device const half* weight    [[buffer(2)]],
    device half*       output    [[buffer(3)]],
    constant uint&     numHeads  [[buffer(4)]],
    uint2 group      [[threadgroup_position_in_grid]],
    uint tid         [[thread_index_in_threadgroup]],
    uint simd_lane   [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 128;
    constexpr uint N_READS = 4;
    constexpr float eps = 1e-6f;
    const uint offset = (group.y * numHeads + group.x) * D + tid * N_READS;

    float sumSq = 0.0f;
    for (uint i = 0; i < N_READS; ++i) {
        const float x = float(input[offset + i]);
        sumSq += x * x;
    }
    sumSq = simd_sum(sumSq);

    threadgroup float inverseRms;
    if (simd_lane == 0) {
        inverseRms = metal::precise::rsqrt(sumSq / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < N_READS; ++i) {
        output[offset + i] = rms_norm_gated_d128_value(
            input[offset + i], gate[offset + i], weight[tid * N_READS + i], inverseRms
        );
    }
}
