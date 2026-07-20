#include <metal_stdlib>
using namespace metal;

// ─── Fused gate split ───
// Splits interleaved Q+gate output into separate query and gate buffers.
// Input: [nHeads × headDim*2] — each head has headDim query then headDim gate values.
// Output: query [nHeads × headDim], gate [nHeads × headDim]
// Uses 2D grid: threadgroup per head, threads split the headDim work.
// Vectorized half4 loads where headDim is aligned.

kernel void gate_split(
    device const half* input   [[buffer(0)]],  // [nHeads * headDim * 2]
    device half*       query   [[buffer(1)]],  // [nHeads * headDim]
    device half*       gate    [[buffer(2)]],  // [nHeads * headDim]
    constant uint&     nHeads  [[buffer(3)]],
    constant uint&     headDim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]      // (dim, head)
) {
    uint head = gid.y;
    uint dim = gid.x;
    if (head >= nHeads || dim >= headDim) return;

    uint srcBase = head * headDim * 2;
    uint dstIdx = head * headDim + dim;
    query[dstIdx] = input[srcBase + dim];
    gate[dstIdx] = input[srcBase + headDim + dim];
}

// ─── Fused per-head RMSNorm (1 + weight) ───
// Applies RMSNorm with (1+weight) independently per head. One dispatch for all heads.
// Input: [nHeads × headDim] — contiguous heads.
// Weight: [headDim] — shared across heads.
// Output: [nHeads × headDim] — can be in-place.
// One threadgroup per head. Uses threadgroup memory to cache input (avoids double read).
// Host contract: dispatch nHeads threadgroups, each with threads_per_threadgroup
// that is a multiple of 32 (SIMD width) and >= 32.

kernel void per_head_rms_norm_1pw(
    device const half* input   [[buffer(0)]],  // [nHeads * headDim]
    device const half* weight  [[buffer(1)]],  // [headDim]
    device half*       output  [[buffer(2)]],  // [nHeads * headDim]
    constant uint&     headDim [[buffer(3)]],
    constant float&    eps     [[buffer(4)]],
    uint head       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgs        [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint offset = head * headDim;

    // Cache input values in threadgroup memory to avoid double device read.
    // Gemma global attention uses headDim=512.
    threadgroup float cached[512];

    // Pass 1: load input to threadgroup memory + accumulate sum of squares
    float sumSq = 0.0f;
    for (uint i = tid; i < headDim; i += tgs) {
        float v = float(input[offset + i]);
        cached[i] = v;
        sumSq += v * v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SIMD reduction
    sumSq = simd_sum(sumSq);

    // Cross-SIMD reduction — use (tgs + 31) / 32 to handle non-32-aligned sizes
    threadgroup float partial[32];
    if (simd_lane == 0) { partial[simd_group] = sumSq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_rsqrt = 0.0f;
    if (tid == 0) {
        float total = 0;
        uint nSimds = (tgs + 31) / 32;  // ceiling division, handles partial SIMD groups
        for (uint s = 0; s < nSimds; s++) { total += partial[s]; }
        float mean = total / float(headDim);
        shared_rsqrt = rsqrt(mean + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;

    // Pass 2: apply normalization from cached values (no second device read)
    for (uint i = tid; i < headDim; i += tgs) {
        float x = cached[i];
        float w = float(weight[i]);
        output[offset + i] = half(x * rs * (1.0f + w));
    }
}

kernel void per_head_rms_norm(
    device const half* input   [[buffer(0)]],  // [nHeads * headDim]
    device const half* weight  [[buffer(1)]],  // [headDim]
    device half*       output  [[buffer(2)]],  // [nHeads * headDim]
    constant uint&     headDim [[buffer(3)]],
    constant float&    eps     [[buffer(4)]],
    uint head       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgs        [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint reads = 4;
    uint offset = head * headDim;
    uint localBase = tid * reads;
    half values[reads];
    float sumSq = 0.0f;
    for (uint i = 0; i < reads; ++i) {
        uint local = localBase + i;
        half value = local < headDim ? input[offset + local] : half(0);
        values[i] = value;
        float x = float(value);
        sumSq += x * x;
    }
    sumSq = simd_sum(sumSq);

    // Match MLX's four-contiguous-reads RMSNorm reduction topology. Keeping
    // this topology in the generic direct-weight norm brick matters because a
    // one-ULP normalization change is amplified by every cached attention use.
    threadgroup float partial[32];
    if (simd_group == 0) { partial[simd_lane] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) { partial[simd_group] = sumSq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_rsqrt;
    if (simd_group == 0) {
        float total = simd_sum(partial[simd_lane]);
        if (simd_lane == 0) {
            shared_rsqrt = metal::precise::rsqrt(total / float(headDim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All input reads complete before an in-place caller can overwrite them.
    for (uint i = 0; i < reads; ++i) {
        uint local = localBase + i;
        if (local < headDim) {
            const half normalized = half(float(values[i]) * shared_rsqrt);
            output[offset + local] = weight[local] * normalized;
        }
    }
}

kernel void per_head_rms_norm_noscale(
    device half*       data    [[buffer(0)]],  // [nHeads * headDim] in-place
    constant uint&     headDim [[buffer(1)]],
    constant float&    eps     [[buffer(2)]],
    uint head       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint tgs        [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint offset = head * headDim;

    threadgroup float cached[512];

    float sumSq = 0.0f;
    for (uint i = tid; i < headDim; i += tgs) {
        float v = float(data[offset + i]);
        cached[i] = v;
        sumSq += v * v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumSq = simd_sum(sumSq);

    threadgroup float partial[32];
    if (simd_lane == 0) { partial[simd_group] = sumSq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_rsqrt = 0.0f;
    if (tid == 0) {
        float total = 0.0f;
        uint nSimds = (tgs + 31) / 32;
        for (uint s = 0; s < nSimds; s++) { total += partial[s]; }
        float mean = total / float(headDim);
        shared_rsqrt = rsqrt(mean + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;
    for (uint i = tid; i < headDim; i += tgs) {
        data[offset + i] = half(cached[i] * rs);
    }
}

kernel void per_head_rms_norm_1pw_batched(
    device const half* input   [[buffer(0)]],  // [batch, nHeads * headDim]
    device const half* weight  [[buffer(1)]],  // [headDim]
    device half*       output  [[buffer(2)]],  // [batch, nHeads * headDim]
    constant uint&     nHeads  [[buffer(3)]],
    constant uint&     headDim [[buffer(4)]],
    constant float&    eps     [[buffer(5)]],
    uint2 group     [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint head = group.x;
    uint batch = group.y;
    uint rowStride = nHeads * headDim;
    uint offset = batch * rowStride + head * headDim;
    uint tgs = min(headDim, 256u);

    threadgroup float cached[512];

    float sumSq = 0.0f;
    for (uint i = tid; i < headDim; i += tgs) {
        float v = float(input[offset + i]);
        cached[i] = v;
        sumSq += v * v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumSq = simd_sum(sumSq);

    threadgroup float partial[32];
    if (simd_lane == 0) { partial[simd_group] = sumSq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_rsqrt = 0.0f;
    if (tid == 0) {
        float total = 0.0f;
        uint nSimds = (tgs + 31) / 32;
        for (uint s = 0; s < nSimds; s++) { total += partial[s]; }
        float mean = total / float(headDim);
        shared_rsqrt = rsqrt(mean + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;
    for (uint i = tid; i < headDim; i += tgs) {
        float x = cached[i];
        float w = float(weight[i]);
        output[offset + i] = half(x * rs * (1.0f + w));
    }
}

kernel void per_head_rms_norm_batched(
    device const half* input   [[buffer(0)]],  // [batch, nHeads * headDim]
    device const half* weight  [[buffer(1)]],  // [headDim]
    device half*       output  [[buffer(2)]],  // [batch, nHeads * headDim]
    constant uint&     nHeads  [[buffer(3)]],
    constant uint&     headDim [[buffer(4)]],
    constant float&    eps     [[buffer(5)]],
    uint2 group     [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint head = group.x;
    uint batch = group.y;
    uint rowStride = nHeads * headDim;
    uint offset = batch * rowStride + head * headDim;
    constexpr uint reads = 4;
    uint localBase = tid * reads;
    half values[reads];
    float sumSq = 0.0f;
    for (uint i = 0; i < reads; ++i) {
        uint local = localBase + i;
        half value = local < headDim ? input[offset + local] : half(0);
        values[i] = value;
        float x = float(value);
        sumSq += x * x;
    }
    sumSq = simd_sum(sumSq);

    threadgroup float partial[32];
    if (simd_group == 0) { partial[simd_lane] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0) { partial[simd_group] = sumSq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_rsqrt;
    if (simd_group == 0) {
        float total = simd_sum(partial[simd_lane]);
        if (simd_lane == 0) {
            shared_rsqrt = metal::precise::rsqrt(total / float(headDim) + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < reads; ++i) {
        uint local = localBase + i;
        if (local < headDim) {
            const half normalized = half(float(values[i]) * shared_rsqrt);
            output[offset + local] = weight[local] * normalized;
        }
    }
}

kernel void per_head_rms_norm_noscale_batched(
    device half*       data    [[buffer(0)]],  // [batch, nHeads * headDim] in-place
    constant uint&     nHeads  [[buffer(1)]],
    constant uint&     headDim [[buffer(2)]],
    constant float&    eps     [[buffer(3)]],
    uint2 group     [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint head = group.x;
    uint batch = group.y;
    uint rowStride = nHeads * headDim;
    uint offset = batch * rowStride + head * headDim;
    uint tgs = min(headDim, 256u);

    threadgroup float cached[512];

    float sumSq = 0.0f;
    for (uint i = tid; i < headDim; i += tgs) {
        float v = float(data[offset + i]);
        cached[i] = v;
        sumSq += v * v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumSq = simd_sum(sumSq);

    threadgroup float partial[32];
    if (simd_lane == 0) { partial[simd_group] = sumSq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float shared_rsqrt = 0.0f;
    if (tid == 0) {
        float total = 0.0f;
        uint nSimds = (tgs + 31) / 32;
        for (uint s = 0; s < nSimds; s++) { total += partial[s]; }
        float mean = total / float(headDim);
        shared_rsqrt = rsqrt(mean + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rs = shared_rsqrt;
    for (uint i = tid; i < headDim; i += tgs) {
        data[offset + i] = half(cached[i] * rs);
    }
}
