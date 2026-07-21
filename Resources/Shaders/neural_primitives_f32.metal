#include <metal_stdlib>
using namespace metal;

inline float dense_erf_approx(float x) {
    float sign = (x < 0.0f) ? -1.0f : 1.0f;
    float absolute = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * absolute);
    float value = 1.0f
        - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
            - 0.284496736f) * t + 0.254829592f) * t
            * exp(-absolute * absolute);
    return sign * value;
}

/// Token-major dense projection with BF16 checkpoint weights/bias and FP32
/// activations/accumulation. The float4 body is source-aligned with Smelt's
/// retained gemm_bf16w_f32; the scalar tail covers the rig model's K=54 and K=5
/// projections without perturbing divisible-by-four reduction order.
kernel void dense_bf16w_f32(
    device const float*  input   [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device const bfloat* bias    [[buffer(2)]],
    device float*        output  [[buffer(3)]],
    constant uint&       rows    [[buffer(4)]],
    constant uint&       outDim  [[buffer(5)]],
    constant uint&       inDim   [[buffer(6)]],
    constant uint&       hasBias [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    uint outputColumn = gid.x;
    uint row = gid.y;
    if (outputColumn >= outDim || row >= rows) return;
    uint chunks = inDim >> 2;
    device const float4* input4 = (device const float4*)(input + row * inDim);
    device const bfloat4* weight4 =
        (device const bfloat4*)(weight + outputColumn * inDim);
    float partial = 0.0f;
    for (uint chunk = lane; chunk < chunks; chunk += 32u) {
        partial += dot(input4[chunk], float4(weight4[chunk]));
    }
    uint tail = (chunks << 2) + lane;
    if (tail < inDim) {
        partial += input[row * inDim + tail]
            * float(weight[outputColumn * inDim + tail]);
    }
    float total = simd_sum(partial);
    if (lane == 0u) {
        output[row * outDim + outputColumn] = total
            + (hasBias != 0u ? float(bias[outputColumn]) : 0.0f);
    }
}

/// Four-row BF16-weight dense projection. Each SIMD preserves the scalar
/// kernel's exact per-row float4 accumulation and simd_sum order while the
/// threadgroup loads one weight row for four independent activation rows.
/// The host routes only K <= 3072 so the complete weight row fits in the
/// fixed threadgroup tile.
kernel void dense_bf16w_f32_rows4(
    device const float*  input   [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device const bfloat* bias    [[buffer(2)]],
    device float*        output  [[buffer(3)]],
    constant uint&       rows    [[buffer(4)]],
    constant uint&       outDim  [[buffer(5)]],
    constant uint&       inDim   [[buffer(6)]],
    constant uint&       hasBias [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint rowsPerThreadgroup = 4u;
    constexpr uint maximumChunks = 768u;
    threadgroup bfloat4 cachedWeight[maximumChunks];
    uint outputColumn = tg.x;
    uint chunks = inDim >> 2;
    if (outputColumn >= outDim || chunks > maximumChunks) return;
    device const bfloat4* weight4 =
        (device const bfloat4*)(weight + outputColumn * inDim);
    for (uint chunk = tid; chunk < chunks; chunk += 128u) {
        cachedWeight[chunk] = weight4[chunk];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = tg.y * rowsPerThreadgroup + simdGroup;
    float partial = 0.0f;
    if (row < rows) {
        device const float4* input4 =
            (device const float4*)(input + row * inDim);
        for (uint chunk = lane; chunk < chunks; chunk += 32u) {
            partial += dot(input4[chunk], float4(cachedWeight[chunk]));
        }
        uint tail = (chunks << 2) + lane;
        if (tail < inDim) {
            partial += input[row * inDim + tail]
                * float(weight[outputColumn * inDim + tail]);
        }
    }
    float total = simd_sum(partial);
    if (lane == 0u && row < rows) {
        output[row * outDim + outputColumn] = total
            + (hasBias != 0u ? float(bias[outputColumn]) : 0.0f);
    }
}

/// Eight-row companion to dense_bf16w_f32_rows4. Per-row arithmetic is
/// identical; the larger tile amortizes each BF16 weight-row load across eight
/// activation rows when the caller has enough independent rows.
kernel void dense_bf16w_f32_rows8(
    device const float*  input   [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device const bfloat* bias    [[buffer(2)]],
    device float*        output  [[buffer(3)]],
    constant uint&       rows    [[buffer(4)]],
    constant uint&       outDim  [[buffer(5)]],
    constant uint&       inDim   [[buffer(6)]],
    constant uint&       hasBias [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint rowsPerThreadgroup = 8u;
    constexpr uint maximumChunks = 768u;
    threadgroup bfloat4 cachedWeight[maximumChunks];
    uint outputColumn = tg.x;
    uint chunks = inDim >> 2;
    if (outputColumn >= outDim || chunks > maximumChunks) return;
    device const bfloat4* weight4 =
        (device const bfloat4*)(weight + outputColumn * inDim);
    for (uint chunk = tid; chunk < chunks; chunk += 256u) {
        cachedWeight[chunk] = weight4[chunk];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = tg.y * rowsPerThreadgroup + simdGroup;
    float partial = 0.0f;
    if (row < rows) {
        device const float4* input4 =
            (device const float4*)(input + row * inDim);
        for (uint chunk = lane; chunk < chunks; chunk += 32u) {
            partial += dot(input4[chunk], float4(cachedWeight[chunk]));
        }
        uint tail = (chunks << 2) + lane;
        if (tail < inDim) {
            partial += input[row * inDim + tail]
                * float(weight[outputColumn * inDim + tail]);
        }
    }
    float total = simd_sum(partial);
    if (lane == 0u && row < rows) {
        output[row * outDim + outputColumn] = total
            + (hasBias != 0u ? float(bias[outputColumn]) : 0.0f);
    }
}

/// Eight-row dense projection with a generic exact epilogue. Epilogue 1 is
/// the retained fp32 GELU expression; epilogue 2 is an fp32 residual add.
/// The reduction body is source-identical to dense_bf16w_f32_rows8.
kernel void dense_bf16w_f32_rows8_epilogue(
    device const float*  input    [[buffer(0)]],
    device const bfloat* weight   [[buffer(1)]],
    device const bfloat* bias     [[buffer(2)]],
    device const float*  residual [[buffer(3)]],
    device float*        output   [[buffer(4)]],
    constant uint&       rows     [[buffer(5)]],
    constant uint&       outDim   [[buffer(6)]],
    constant uint&       inDim    [[buffer(7)]],
    constant uint&       hasBias  [[buffer(8)]],
    constant uint&       epilogue [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint rowsPerThreadgroup = 8u;
    constexpr uint maximumChunks = 768u;
    threadgroup bfloat4 cachedWeight[maximumChunks];
    uint outputColumn = tg.x;
    uint chunks = inDim >> 2;
    if (outputColumn >= outDim || chunks > maximumChunks) return;
    device const bfloat4* weight4 =
        (device const bfloat4*)(weight + outputColumn * inDim);
    for (uint chunk = tid; chunk < chunks; chunk += 256u) {
        cachedWeight[chunk] = weight4[chunk];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = tg.y * rowsPerThreadgroup + simdGroup;
    float partial = 0.0f;
    if (row < rows) {
        device const float4* input4 =
            (device const float4*)(input + row * inDim);
        for (uint chunk = lane; chunk < chunks; chunk += 32u) {
            partial += dot(input4[chunk], float4(cachedWeight[chunk]));
        }
        uint tail = (chunks << 2) + lane;
        if (tail < inDim) {
            partial += input[row * inDim + tail]
                * float(weight[outputColumn * inDim + tail]);
        }
    }
    float total = simd_sum(partial);
    if (lane == 0u && row < rows) {
        uint index = row * outDim + outputColumn;
        float result = total
            + (hasBias != 0u ? float(bias[outputColumn]) : 0.0f);
        if (epilogue == 1u) {
            float x = result;
            result = 0.5f * x
                * (1.0f + dense_erf_approx(x * 0.70710678f));
        } else if (epilogue == 2u) {
            result = residual[index] + result;
        }
        output[index] = result;
    }
}

/// Computes multiple output columns per eight-row threadgroup while preserving
/// the independent-column kernel's reduction order for every output element.
/// The tile shares each activation vector across the selected weight rows.
template <uint outputColumns>
inline void dense_bf16w_f32_rows8_columns_body(
    device const float*  input,
    device const bfloat* weight,
    device const bfloat* bias,
    device const float*  residual,
    device float*        output,
    uint                 rows,
    uint                 outDim,
    uint                 inDim,
    uint                 hasBias,
    uint                 epilogue,
    uint2                tg,
    uint                 tid,
    uint                 lane,
    uint                 simdGroup,
    threadgroup bfloat4* cachedWeight
) {
    constexpr uint rowsPerThreadgroup = 8u;
    constexpr uint maximumChunks = 768u;
    uint outputColumnBase = tg.x * outputColumns;
    uint chunks = inDim >> 2;
    if (outputColumnBase >= outDim || chunks > maximumChunks) return;
    uint activeColumns = min(outputColumns, outDim - outputColumnBase);
    uint cachedChunks = activeColumns * chunks;
    for (uint index = tid; index < cachedChunks; index += 256u) {
        uint localColumn = index / chunks;
        uint chunk = index - localColumn * chunks;
        device const bfloat4* weight4 = (device const bfloat4*)(
            weight + (outputColumnBase + localColumn) * inDim
        );
        cachedWeight[localColumn * maximumChunks + chunk] = weight4[chunk];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint row = tg.y * rowsPerThreadgroup + simdGroup;
    float partial[outputColumns];
    for (uint localColumn = 0u; localColumn < outputColumns; ++localColumn) {
        partial[localColumn] = 0.0f;
    }
    if (row < rows) {
        device const float4* input4 =
            (device const float4*)(input + row * inDim);
        for (uint chunk = lane; chunk < chunks; chunk += 32u) {
            float4 activation = input4[chunk];
            for (uint localColumn = 0u;
                 localColumn < activeColumns;
                 ++localColumn) {
                partial[localColumn] += dot(
                    activation,
                    float4(cachedWeight[localColumn * maximumChunks + chunk])
                );
            }
        }
        uint tail = (chunks << 2) + lane;
        if (tail < inDim) {
            float activation = input[row * inDim + tail];
            for (uint localColumn = 0u;
                 localColumn < activeColumns;
                 ++localColumn) {
                partial[localColumn] += activation * float(
                    weight[(outputColumnBase + localColumn) * inDim + tail]
                );
            }
        }
    }
    for (uint localColumn = 0u;
         localColumn < outputColumns;
         ++localColumn) {
        float total = simd_sum(partial[localColumn]);
        if (lane == 0u && row < rows && localColumn < activeColumns) {
            uint outputColumn = outputColumnBase + localColumn;
            uint outputIndex = row * outDim + outputColumn;
            float result = total
                + (hasBias != 0u ? float(bias[outputColumn]) : 0.0f);
            if (epilogue == 1u) {
                float x = result;
                result = 0.5f * x
                    * (1.0f + dense_erf_approx(x * 0.70710678f));
            } else if (epilogue == 2u) {
                result = residual[outputIndex] + result;
            }
            output[outputIndex] = result;
        }
    }
}

kernel void dense_bf16w_f32_rows8_cols2_epilogue(
    device const float*  input    [[buffer(0)]],
    device const bfloat* weight   [[buffer(1)]],
    device const bfloat* bias     [[buffer(2)]],
    device const float*  residual [[buffer(3)]],
    device float*        output   [[buffer(4)]],
    constant uint&       rows     [[buffer(5)]],
    constant uint&       outDim   [[buffer(6)]],
    constant uint&       inDim    [[buffer(7)]],
    constant uint&       hasBias  [[buffer(8)]],
    constant uint&       epilogue [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]]
) {
    threadgroup bfloat4 cachedWeight[2u * 768u];
    dense_bf16w_f32_rows8_columns_body<2u>(
        input, weight, bias, residual, output,
        rows, outDim, inDim, hasBias, epilogue,
        tg, tid, lane, simdGroup, cachedWeight
    );
}

/// Row-major LayerNorm for [rows, dim] fp32 activations and parameters.
/// One threadgroup owns one row; reductions stay fp32. This is intentionally
/// source-identical to the retained Qwen3.5 vision implementation so the
/// rig model port has an internal raw-bit oracle.
kernel void layer_norm_rows_f32(
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

/// LayerNorm with the same fp32 activation/reduction path and authoritative
/// BF16 checkpoint parameters. rig model keeps every normalization parameter in
/// BF16; widening happens only at this consumer boundary.
kernel void layer_norm_rows_bf16w_f32(
    device const float*  input  [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const bfloat* bias   [[buffer(2)]],
    device float*        output [[buffer(3)]],
    constant uint&       rows   [[buffer(4)]],
    constant uint&       dim    [[buffer(5)]],
    constant float&      eps    [[buffer(6)]],
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
        y[i] = (x[i] - mean) * inv * float(weight[i]) + float(bias[i]);
    }
}

/// Extract one Q/K/V part from PyTorch's head-interleaved linear layout:
/// `[token, head, part, headDim]` -> `[token, head * headDim]`.
kernel void extract_interleaved_head_part_f32(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      tokens [[buffer(2)]],
    constant uint&      heads [[buffer(3)]],
    constant uint&      headDim [[buffer(4)]],
    constant uint&      parts [[buffer(5)]],
    constant uint&      selectedPart [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint token = gid.y;
    uint column = gid.x;
    uint hidden = heads * headDim;
    if (token >= tokens || column >= hidden || selectedPart >= parts) return;
    uint head = column / headDim;
    uint withinHead = column % headDim;
    uint sourceColumn = (head * parts + selectedPart) * headDim + withinHead;
    output[token * hidden + column] =
        input[token * hidden * parts + sourceColumn];
}

/// Reproduce Tripo2AttnProcessor2_0's authored `cat(...).view(heads, parts,
/// headDim).split(...)` layout from separate projection buffers. Although this
/// ordering is unusual, it is checkpoint semantics and cannot be "cleaned up".
kernel void repack_concatenated_head_parts_f32(
    device const float* input0 [[buffer(0)]],
    device const float* input1 [[buffer(1)]],
    device const float* input2 [[buffer(2)]],
    device float* output0 [[buffer(3)]],
    device float* output1 [[buffer(4)]],
    device float* output2 [[buffer(5)]],
    constant uint& tokens [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant uint& headDim [[buffer(8)]],
    constant uint& parts [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint token = gid.y;
    uint column = gid.x;
    uint selectedPart = gid.z;
    uint hidden = heads * headDim;
    if (token >= tokens || column >= hidden || selectedPart >= parts) return;
    uint head = column / headDim;
    uint withinHead = column % headDim;
    uint concatenatedColumn = (head * parts + selectedPart) * headDim + withinHead;
    uint sourceTensor = concatenatedColumn / hidden;
    uint sourceColumn = concatenatedColumn % hidden;
    float value = sourceTensor == 0u
        ? input0[token * hidden + sourceColumn]
        : (sourceTensor == 1u
            ? input1[token * hidden + sourceColumn]
            : input2[token * hidden + sourceColumn]);
    if (selectedPart == 0u) output0[token * hidden + column] = value;
    else if (selectedPart == 1u) output1[token * hidden + column] = value;
    else output2[token * hidden + column] = value;
}

/// Full non-causal multi-head attention with independent query and key/value
/// token counts. One SIMD owns a (query, head); two head-dimension values are
/// accumulated per lane. the rig model's Michelangelo and SkinVAE attention blocks
/// all use headDim == 64, which is the largest shape this kernel represents.
kernel void noncausal_attention_f32(
    device const float* q      [[buffer(0)]],
    device const float* k      [[buffer(1)]],
    device const float* v      [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint&      queryTokens    [[buffer(4)]],
    constant uint&      keyValueTokens [[buffer(5)]],
    constant uint&      heads          [[buffer(6)]],
    constant uint&      headDim        [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    uint query = tg.x;
    uint head = tg.y;
    if (query >= queryTokens || head >= heads || lane >= 32u) return;
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
    for (uint source = 0u; source < keyValueTokens; ++source) {
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

/// Eight-query non-causal attention tile. Each SIMD retains the monolithic
/// kernel's complete per-query recurrence while the threadgroup stages eight
/// source K/V rows once for all eight independent queries.
kernel void noncausal_attention_q8_f32(
    device const float* q      [[buffer(0)]],
    device const float* k      [[buffer(1)]],
    device const float* v      [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint&      queryTokens    [[buffer(4)]],
    constant uint&      keyValueTokens [[buffer(5)]],
    constant uint&      heads          [[buffer(6)]],
    constant uint&      headDim        [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint queryTile = 8u;
    constexpr uint sourceTile = 8u;
    constexpr uint maximumHeadDim = 64u;
    threadgroup float cachedKey[sourceTile * maximumHeadDim];
    threadgroup float cachedValue[sourceTile * maximumHeadDim];

    uint head = tg.y;
    if (head >= heads || headDim > maximumHeadDim) return;
    uint query = tg.x * queryTile + simdGroup;
    bool active = query < queryTokens;
    uint hidden = heads * headDim;
    uint d0 = lane;
    uint d1 = lane + 32u;
    uint qBase = query * hidden + head * headDim;
    float q0 = active && d0 < headDim ? q[qBase + d0] : 0.0f;
    float q1 = active && d1 < headDim ? q[qBase + d1] : 0.0f;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float scale = rsqrt(float(headDim));
    for (uint sourceBase = 0u;
         sourceBase < keyValueTokens;
         sourceBase += sourceTile) {
        uint sourceCount = min(sourceTile, keyValueTokens - sourceBase);
        uint scalarCount = sourceCount * headDim;
        for (uint index = tid; index < scalarCount; index += 256u) {
            uint source = sourceBase + index / headDim;
            uint dimension = index % headDim;
            uint offset = source * hidden + head * headDim + dimension;
            cachedKey[index] = k[offset];
            cachedValue[index] = v[offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint localSource = 0u;
             localSource < sourceCount;
             ++localSource) {
            uint localBase = localSource * headDim;
            float partial = 0.0f;
            if (active && d0 < headDim) {
                partial += q0 * cachedKey[localBase + d0];
            }
            if (active && d1 < headDim) {
                partial += q1 * cachedKey[localBase + d1];
            }
            float score = simd_sum(partial) * scale;
            if (active) {
                float nextMaximum = max(maximum, score);
                float oldScale = maximum == -INFINITY
                    ? 0.0f
                    : exp(maximum - nextMaximum);
                float weightValue = exp(score - nextMaximum);
                denominator = denominator * oldScale + weightValue;
                if (d0 < headDim) {
                    acc0 = acc0 * oldScale
                        + weightValue * cachedValue[localBase + d0];
                }
                if (d1 < headDim) {
                    acc1 = acc1 * oldScale
                        + weightValue * cachedValue[localBase + d1];
                }
                maximum = nextMaximum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        uint outBase = query * hidden + head * headDim;
        if (d0 < headDim) output[outBase + d0] = acc0 / denominator;
        if (d1 < headDim) output[outBase + d1] = acc1 / denominator;
    }
}

/// Sixteen-query non-causal attention tile. Each SIMD retains the monolithic
/// kernel's complete per-query recurrence while the threadgroup stages 32
/// source K/V rows once for all sixteen independent queries.
kernel void noncausal_attention_q16_f32(
    device const float* q      [[buffer(0)]],
    device const float* k      [[buffer(1)]],
    device const float* v      [[buffer(2)]],
    device float*       output [[buffer(3)]],
    constant uint&      queryTokens    [[buffer(4)]],
    constant uint&      keyValueTokens [[buffer(5)]],
    constant uint&      heads          [[buffer(6)]],
    constant uint&      headDim        [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint queryTile = 16u;
    constexpr uint sourceTile = 32u;
    constexpr uint maximumHeadDim = 64u;
    threadgroup float cachedKey[sourceTile * maximumHeadDim];
    threadgroup float cachedValue[sourceTile * maximumHeadDim];

    uint head = tg.y;
    if (head >= heads || headDim > maximumHeadDim) return;
    uint query = tg.x * queryTile + simdGroup;
    bool active = query < queryTokens;
    uint hidden = heads * headDim;
    uint d0 = lane;
    uint d1 = lane + 32u;
    uint qBase = query * hidden + head * headDim;
    float q0 = active && d0 < headDim ? q[qBase + d0] : 0.0f;
    float q1 = active && d1 < headDim ? q[qBase + d1] : 0.0f;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float scale = rsqrt(float(headDim));
    for (uint sourceBase = 0u;
         sourceBase < keyValueTokens;
         sourceBase += sourceTile) {
        uint sourceCount = min(sourceTile, keyValueTokens - sourceBase);
        uint scalarCount = sourceCount * headDim;
        for (uint index = tid; index < scalarCount; index += 512u) {
            uint source = sourceBase + index / headDim;
            uint dimension = index % headDim;
            uint offset = source * hidden + head * headDim + dimension;
            cachedKey[index] = k[offset];
            cachedValue[index] = v[offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint localSource = 0u;
             localSource < sourceCount;
             ++localSource) {
            uint localBase = localSource * headDim;
            float partial = 0.0f;
            if (active && d0 < headDim) {
                partial += q0 * cachedKey[localBase + d0];
            }
            if (active && d1 < headDim) {
                partial += q1 * cachedKey[localBase + d1];
            }
            float score = simd_sum(partial) * scale;
            if (active) {
                float nextMaximum = max(maximum, score);
                float oldScale = maximum == -INFINITY
                    ? 0.0f
                    : exp(maximum - nextMaximum);
                float weightValue = exp(score - nextMaximum);
                denominator = denominator * oldScale + weightValue;
                if (d0 < headDim) {
                    acc0 = acc0 * oldScale
                        + weightValue * cachedValue[localBase + d0];
                }
                if (d1 < headDim) {
                    acc1 = acc1 * oldScale
                        + weightValue * cachedValue[localBase + d1];
                }
                maximum = nextMaximum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        uint outBase = query * hidden + head * headDim;
        if (d0 < headDim) output[outBase + d0] = acc0 / denominator;
        if (d1 < headDim) output[outBase + d1] = acc1 / denominator;
    }
}

/// Stateful form of non-causal attention for key/value sequences too long for
/// one reliable GPU command buffer. Chunks are submitted in source order and
/// persist the exact online-softmax state at FP32 boundaries, preserving the
/// monolithic kernel's operation order for every source token.
kernel void noncausal_attention_update_f32(
    device const float* q           [[buffer(0)]],
    device const float* k           [[buffer(1)]],
    device const float* v           [[buffer(2)]],
    device float* accumulator       [[buffer(3)]],
    device float* maximumState      [[buffer(4)]],
    device float* denominatorState  [[buffer(5)]],
    constant uint& queryTokens      [[buffer(6)]],
    constant uint& keyValueTokens   [[buffer(7)]],
    constant uint& heads            [[buffer(8)]],
    constant uint& headDim          [[buffer(9)]],
    constant uint& sourceStart      [[buffer(10)]],
    constant uint& sourceCount      [[buffer(11)]],
    constant uint& finalize         [[buffer(12)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    uint query = tg.x;
    uint head = tg.y;
    if (query >= queryTokens || head >= heads || lane >= 32u) return;
    uint hidden = heads * headDim;
    uint d0 = lane;
    uint d1 = lane + 32u;
    uint qBase = query * hidden + head * headDim;
    uint stateIndex = query * heads + head;
    float q0 = d0 < headDim ? q[qBase + d0] : 0.0f;
    float q1 = d1 < headDim ? q[qBase + d1] : 0.0f;
    float maximum = maximumState[stateIndex];
    float denominator = denominatorState[stateIndex];
    float acc0 = d0 < headDim ? accumulator[qBase + d0] : 0.0f;
    float acc1 = d1 < headDim ? accumulator[qBase + d1] : 0.0f;
    float scale = rsqrt(float(headDim));
    uint sourceEnd = min(sourceStart + sourceCount, keyValueTokens);
    for (uint source = sourceStart; source < sourceEnd; ++source) {
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
    if (d0 < headDim) {
        accumulator[qBase + d0] = finalize != 0u ? acc0 / denominator : acc0;
    }
    if (d1 < headDim) {
        accumulator[qBase + d1] = finalize != 0u ? acc1 / denominator : acc1;
    }
    if (lane == 0u) {
        maximumState[stateIndex] = maximum;
        denominatorState[stateIndex] = denominator;
    }
}

/// Michelangelo / SkinVAE Fourier position embedding. Output layout matches
/// PyTorch's flatten + concat exactly:
///   [input?], sin([x0*f0...xN*fF]), cos([x0*f0...xN*fF]).
/// PMPE adds a phase sinusoid to each corresponding Fourier component.
kernel void fourier_position_embedding_f32(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      rows         [[buffer(2)]],
    constant uint&      inputDim     [[buffer(3)]],
    constant uint&      numFreqs     [[buffer(4)]],
    constant uint&      includeInput [[buffer(5)]],
    constant uint&      includePi    [[buffer(6)]],
    constant uint&      usePMPE      [[buffer(7)]],
    constant uint&      inputStride  [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint embeddingScalars = inputDim * numFreqs;
    uint inputScalars = includeInput != 0u ? inputDim : 0u;
    uint outputDim = inputScalars + 2u * embeddingScalars;
    uint column = gid.x;
    if (row >= rows || column >= outputDim) return;

    uint outputBase = row * outputDim;
    uint inputBase = row * inputStride;
    if (column < inputScalars) {
        output[outputBase + column] = input[inputBase + column];
        return;
    }

    uint trigIndex = column - inputScalars;
    bool cosine = trigIndex >= embeddingScalars;
    uint flattened = trigIndex % embeddingScalars;
    uint coordinate = flattened / numFreqs;
    uint frequencyIndex = flattened % numFreqs;
    float x = input[inputBase + coordinate];
    float frequency = float(1u << frequencyIndex);
    if (includePi != 0u) frequency *= M_PI_F;
    float angle = x * frequency;
    float value = cosine ? cos(angle) : sin(angle);

    if (usePMPE != 0u) {
        float fraction = float(frequencyIndex + 1u) / float(numFreqs);
        float phase = (pow(float(numFreqs), 1.0f - fraction) + fraction)
            * (2.0f * M_PI_F);
        float phaseAngle = x * (0.5f * M_PI_F) + phase;
        value += cosine ? cos(phaseAngle) : sin(phaseAngle);
    }
    output[outputBase + column] = value;
}

inline float round_bf16_exact(float value) {
    return float(bfloat(value));
}

/// Decoder-query PMPE with the authored BF16 tensor semantics. Upstream casts
/// sampled queries and the embedder's frequency/phase buffers to BF16 before
/// these elementwise operations; every operation below therefore rounds back
/// to BF16. Results are widened for the staged FP32 decoder.
kernel void pmpe_bf16_semantics_f32(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      rows         [[buffer(2)]],
    constant uint&      inputDim     [[buffer(3)]],
    constant uint&      numFreqs     [[buffer(4)]],
    constant uint&      includeInput [[buffer(5)]],
    constant uint&      includePi    [[buffer(6)]],
    constant uint&      usePMPE      [[buffer(7)]],
    constant uint&      inputStride  [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint embeddingScalars = inputDim * numFreqs;
    uint inputScalars = includeInput != 0u ? inputDim : 0u;
    uint outputDim = inputScalars + 2u * embeddingScalars;
    uint column = gid.x;
    if (row >= rows || column >= outputDim) return;

    uint outputBase = row * outputDim;
    uint inputBase = row * inputStride;
    float x = round_bf16_exact(input[inputBase + min(column, inputDim - 1u)]);
    if (column < inputScalars) {
        output[outputBase + column] = x;
        return;
    }

    uint trigIndex = column - inputScalars;
    bool cosine = trigIndex >= embeddingScalars;
    uint flattened = trigIndex % embeddingScalars;
    uint coordinate = flattened / numFreqs;
    uint frequencyIndex = flattened % numFreqs;
    x = round_bf16_exact(input[inputBase + coordinate]);
    float frequency = float(1u << frequencyIndex);
    if (includePi != 0u) {
        frequency = round_bf16_exact(frequency * M_PI_F);
    }
    float angle = round_bf16_exact(x * frequency);
    float value = round_bf16_exact(cosine ? cos(angle) : sin(angle));

    if (usePMPE != 0u) {
        float fraction = float(frequencyIndex + 1u) / float(numFreqs);
        float phase = round_bf16_exact(
            (pow(float(numFreqs), 1.0f - fraction) + fraction)
                * (2.0f * M_PI_F)
        );
        float phaseProduct = round_bf16_exact(x * M_PI_F);
        float phaseAngle = round_bf16_exact(
            round_bf16_exact(phaseProduct * 0.5f) + phase
        );
        float phaseValue = round_bf16_exact(
            cosine ? cos(phaseAngle) : sin(phaseAngle)
        );
        value = round_bf16_exact(value + phaseValue);
    }
    output[outputBase + column] = value;
}

/// Concatenate a dense base embedding with a strided feature suffix. rig model
/// uses this to append normals from interleaved [xyz, normal] point rows.
kernel void append_strided_features_f32(
    device const float* base     [[buffer(0)]],
    device const float* features [[buffer(1)]],
    device float*       output   [[buffer(2)]],
    constant uint&      rows          [[buffer(3)]],
    constant uint&      baseDim       [[buffer(4)]],
    constant uint&      featureStride [[buffer(5)]],
    constant uint&      featureOffset [[buffer(6)]],
    constant uint&      featureCount  [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint column = gid.x;
    uint outputDim = baseDim + featureCount;
    if (row >= rows || column >= outputDim) return;
    if (column < baseDim) {
        output[row * outputDim + column] = base[row * baseDim + column];
    } else {
        uint feature = column - baseDim;
        output[row * outputDim + column] =
            features[row * featureStride + featureOffset + feature];
    }
}

/// Decode SkinVAE's FSQ index into five independent level-8 codes. Every
/// output is one of {-1, -.75, -.5, -.25, 0, .25, .5, .75} and therefore
/// exactly representable in fp32.
kernel void fsq_base8x5_decode_f32(
    device const uint* indices [[buffer(0)]],
    device float*      codes   [[buffer(1)]],
    constant uint&     count   [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    uint index = indices[tid];
    uint basis = 1u;
    for (uint level = 0u; level < 5u; ++level) {
        uint digit = (index / basis) % 8u;
        codes[tid * 5u + level] = (float(digit) - 4.0f) * 0.25f;
        basis *= 8u;
    }
}

/// FP32 residual add shared by staged rig model transformer blocks.
kernel void add_rows_f32(
    device const float* lhs   [[buffer(0)]],
    device const float* rhs   [[buffer(1)]],
    device float*       output [[buffer(2)]],
    constant uint&      count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) output[tid] = lhs[tid] + rhs[tid];
}

/// Final SkinVAE scalar activation.
kernel void sigmoid_f32(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      count  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        float x = input[tid];
        output[tid] = 1.0f / (1.0f + exp(-x));
    }
}

/// Row-major RMSNorm with an authoritative BF16 scale. the rig model's output
/// `nn.RMSNorm` has no explicit epsilon, so the runtime passes BF16 finfo eps
/// (0.0078125) rather than Qwen's unrelated 1e-6 value.
kernel void rms_norm_rows_bf16w_f32(
    device const float*  input  [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float*        output [[buffer(2)]],
    constant uint&       rows [[buffer(3)]],
    constant uint&       dim [[buffer(4)]],
    constant float&      eps [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdGroup [[simdgroup_index_in_threadgroup]],
    uint3 tgs [[threads_per_threadgroup]]
) {
    uint row = tg.x;
    if (row >= rows) return;
    device const float* x = input + row * dim;
    device float* y = output + row * dim;
    float localSquares = 0.0f;
    for (uint i = tid; i < dim; i += tgs.x) localSquares += x[i] * x[i];
    localSquares = simd_sum(localSquares);
    threadgroup float partial[32];
    if (lane == 0) partial[simdGroup] = localSquares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        uint groups = (tgs.x + 31u) / 32u;
        for (uint i = 0; i < groups; ++i) total += partial[i];
        partial[0] = rsqrt(total / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverseRMS = partial[0];
    for (uint i = tid; i < dim; i += tgs.x) {
        y[i] = x[i] * inverseRMS * float(weight[i]);
    }
}
