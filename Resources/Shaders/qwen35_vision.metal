#include <metal_stdlib>
using namespace metal;

kernel void qwen35_vision_softmax_rows_f32(
    device float*   scores [[buffer(0)]],
    constant uint&  rows   [[buffer(1)]],
    constant uint&  cols   [[buffer(2)]],
    constant float& scale  [[buffer(3)]],
    uint3 tg       [[threadgroup_position_in_grid]],
    uint tid       [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]],
    uint3 tgs      [[threads_per_threadgroup]]
) {
    uint row = tg.x;
    if (row >= rows) return;
    device float* values = scores + row * cols;
    threadgroup float partial[32];

    float localMaximum = -INFINITY;
    for (uint column = tid; column < cols; column += tgs.x) {
        localMaximum = max(localMaximum, values[column] * scale);
    }
    localMaximum = simd_max(localMaximum);
    if (lane == 0u) partial[simdGroup] = localMaximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float maximum = -INFINITY;
        uint groups = (tgs.x + 31u) / 32u;
        for (uint index = 0u; index < groups; ++index) {
            maximum = max(maximum, partial[index]);
        }
        partial[0] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = partial[0];

    float localSum = 0.0f;
    for (uint column = tid; column < cols; column += tgs.x) {
        float value = exp(values[column] * scale - maximum);
        values[column] = value;
        localSum += value;
    }
    localSum = simd_sum(localSum);
    if (lane == 0u) partial[simdGroup] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float sum = 0.0f;
        uint groups = (tgs.x + 31u) / 32u;
        for (uint index = 0u; index < groups; ++index) sum += partial[index];
        partial[0] = 1.0f / sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverseSum = partial[0];
    for (uint column = tid; column < cols; column += tgs.x) {
        values[column] *= inverseSum;
    }
}
kernel void qwen35_vision_check_finite_f32(
    device const float* values      [[buffer(0)]],
    device atomic_uint* firstBad    [[buffer(1)]],
    constant uint&      count       [[buffer(2)]],
    constant uint&      stage       [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count && !isfinite(values[tid])) {
        atomic_fetch_min_explicit(firstBad, stage, memory_order_relaxed);
    }
}

kernel void qwen35_vision_add_f32(
    device const float* lhs   [[buffer(0)]],
    device const float* rhs   [[buffer(1)]],
    device float*       out   [[buffer(2)]],
    constant uint&      count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) out[tid] = lhs[tid] + rhs[tid];
}

kernel void qwen35_vision_add_bias_rows_f32(
    device float*       values [[buffer(0)]],
    device const float* bias   [[buffer(1)]],
    constant uint&      rows   [[buffer(2)]],
    constant uint&      cols   [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    if (tid.x < cols && tid.y < rows) {
        values[tid.y * cols + tid.x] += bias[tid.x];
    }
}

/// Row-major LayerNorm for [rows, dim] fp32 activations and parameters.
/// One threadgroup owns one row; reductions stay fp32.
kernel void qwen35_vision_layer_norm_f32(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint&      rows   [[buffer(4)]],
    constant uint&      dim    [[buffer(5)]],
    constant float&     eps    [[buffer(6)]],
    uint3 tg        [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint lane       [[thread_index_in_simdgroup]],
    uint simdGroup  [[simdgroup_index_in_threadgroup]],
    uint3 tgs       [[threads_per_threadgroup]]
) {
    uint row = tg.x;
    if (row >= rows) return;
    device const float* x = input + row * dim;
    device float* y = output + row * dim;

    float localSum = 0.0f;
    for (uint i = tid; i < dim; i += tgs.x) localSum += x[i];
    localSum = simd_sum(localSum);
    threadgroup float partial[32];
    if (lane == 0) partial[simdGroup] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        uint groups = (tgs.x + 31u) / 32u;
        for (uint i = 0; i < groups; ++i) total += partial[i];
        partial[0] = total / float(dim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = partial[0];

    float localVar = 0.0f;
    for (uint i = tid; i < dim; i += tgs.x) {
        float d = x[i] - mean;
        localVar += d * d;
    }
    localVar = simd_sum(localVar);
    if (lane == 0) partial[simdGroup] = localVar;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        uint groups = (tgs.x + 31u) / 32u;
        for (uint i = 0; i < groups; ++i) total += partial[i];
        partial[0] = rsqrt(total / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = partial[0];
    for (uint i = tid; i < dim; i += tgs.x) {
        y[i] = (x[i] - mean) * inv * weight[i] + bias[i];
    }
}

/// Split [T, 3H] QKV and apply Qwen's split-half 2D vision rotary to Q/K.
/// cos/sin are [T, D] and already contain the repeated row/column frequencies.
kernel void qwen35_vision_split_rope_f32(
    device const float* qkv     [[buffer(0)]],
    device const float* cosines [[buffer(1)]],
    device const float* sines   [[buffer(2)]],
    device float*       qOut    [[buffer(3)]],
    device float*       kOut    [[buffer(4)]],
    device float*       vOut    [[buffer(5)]],
    constant uint&      tokens  [[buffer(6)]],
    constant uint&      heads   [[buffer(7)]],
    constant uint&      headDim [[buffer(8)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint pair = tid.x;
    uint head = tid.y;
    uint token = tid.z;
    uint halfDim = headDim / 2u;
    if (token >= tokens || head >= heads || pair >= halfDim) return;
    uint hidden = heads * headDim;
    uint qBase = token * 3u * hidden + head * headDim;
    uint kBase = qBase + hidden;
    uint vBase = kBase + hidden;
    uint outBase = token * hidden + head * headDim;
    float c0 = cosines[token * headDim + pair];
    float s0 = sines[token * headDim + pair];
    float c1 = cosines[token * headDim + pair + halfDim];
    float s1 = sines[token * headDim + pair + halfDim];
    float q0 = qkv[qBase + pair];
    float q1 = qkv[qBase + pair + halfDim];
    float k0 = qkv[kBase + pair];
    float k1 = qkv[kBase + pair + halfDim];
    qOut[outBase + pair] = q0 * c0 - q1 * s0;
    qOut[outBase + pair + halfDim] = q1 * c1 + q0 * s1;
    kOut[outBase + pair] = k0 * c0 - k1 * s0;
    kOut[outBase + pair + halfDim] = k1 * c1 + k0 * s1;
    vOut[outBase + pair] = qkv[vBase + pair];
    vOut[outBase + pair + halfDim] = qkv[vBase + pair + halfDim];
}

/// Full non-causal attention over independently packed frame chunks. One SIMD
/// owns a (query, head); two head-dimension values are accumulated per lane.
kernel void qwen35_vision_attention_f32(
    device const float* q          [[buffer(0)]],
    device const float* k          [[buffer(1)]],
    device const float* v          [[buffer(2)]],
    device const uint*  chunkStart [[buffer(3)]],
    device const uint*  chunkEnd   [[buffer(4)]],
    device float*       output     [[buffer(5)]],
    constant uint&      tokens     [[buffer(6)]],
    constant uint&      heads      [[buffer(7)]],
    constant uint&      headDim    [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    uint query = tg.x;
    uint head = tg.y;
    if (query >= tokens || head >= heads || lane >= 32u) return;
    uint hidden = heads * headDim;
    uint d0 = lane;
    uint d1 = lane + 32u;
    uint qBase = query * hidden + head * headDim;
    float q0 = d0 < headDim ? q[qBase + d0] : 0.0f;
    float q1 = d1 < headDim ? q[qBase + d1] : 0.0f;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float scale = rsqrt(float(headDim));
    uint begin = chunkStart[query];
    uint end = chunkEnd[query];
    for (uint source = begin; source < end; ++source) {
        uint sourceBase = source * hidden + head * headDim;
        float partial = 0.0f;
        if (d0 < headDim) partial += q0 * k[sourceBase + d0];
        if (d1 < headDim) partial += q1 * k[sourceBase + d1];
        float score = simd_sum(partial) * scale;
        float nextMaximum = max(maximum, score);
        float oldScale = maximum == -INFINITY ? 0.0f : exp(maximum - nextMaximum);
        float weight = exp(score - nextMaximum);
        denominator = denominator * oldScale + weight;
        if (d0 < headDim) acc0 = acc0 * oldScale + weight * v[sourceBase + d0];
        if (d1 < headDim) acc1 = acc1 * oldScale + weight * v[sourceBase + d1];
        maximum = nextMaximum;
    }
    uint outBase = query * hidden + head * headDim;
    if (d0 < headDim) output[outBase + d0] = acc0 / denominator;
    if (d1 < headDim) output[outBase + d1] = acc1 / denominator;
}

/// PyTorch's approximate="tanh" GELU used by Qwen3.5 vision.
kernel void qwen35_vision_gelu_tanh_f32(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      count  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    float x = input[tid];
    constexpr float rootTwoOverPi = 0.7978845608028654f;
    // The literal cubic overflows for a large but still-finite x. GELU's
    // correctly rounded limits are x for large positive values and +0 for
    // large negative values, so take those limits before evaluating x^3.
    if (x > 10.0f) {
        output[tid] = x;
    } else if (x < -10.0f) {
        output[tid] = 0.0f;
    } else {
        output[tid] = 0.5f * x
            * (1.0f + tanh(rootTwoOverPi * (x + 0.044715f * x * x * x)));
    }
}
