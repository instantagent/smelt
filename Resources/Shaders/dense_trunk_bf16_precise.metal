#include <metal_stdlib>
using namespace metal;

// BF16-activation variants of the dense trunk bricks. Accumulations that the
// source graph performs in FP32 remain FP32; device/local BF16 values preserve
// every materialized source boundary, including boundaries hidden by fusion.

inline bool bf16_rounding_boundary(float value) {
    uint remainder = as_type<uint>(value) & 0xFFFFu;
    return abs(value) < 0.015625f
        || (remainder >= 0x7000u && remainder <= 0x9000u);
}

inline float bf16_reference_rsqrt(float value) {
    return 1.0f / sqrt(value);
}

// SLEEF single-precision exp, 1-ULP contract, in the FMA topology used by
// arm64 vector softmax. Keeping this local makes softmax parity independent of
// the device driver's transcendental implementation.
inline float sleef_exp_u10(float value) {
    constexpr float reciprocalLn2 = 1.4426950408889634073599246810018921374f;
    constexpr float ln2Upper = 0.693145751953125f;
    constexpr float ln2Lower = 1.428606765330187045e-06f;
    int exponent = int(rint(value * reciprocalLn2));
    float reduced = fma(float(exponent), -ln2Upper, value);
    reduced = fma(float(exponent), -ln2Lower, reduced);

    float polynomial = 0.000198527617612853646278381f;
    polynomial = fma(polynomial, reduced, 0.00139304355252534151077271f);
    polynomial = fma(polynomial, reduced, 0.00833336077630519866943359f);
    polynomial = fma(polynomial, reduced, 0.0416664853692054748535156f);
    polynomial = fma(polynomial, reduced, 0.166666671633720397949219f);
    polynomial = fma(polynomial, reduced, 0.5f);
    polynomial = 1.0f + fma(reduced * reduced, polynomial, reduced);

    int firstExponent = exponent >> 1;
    int secondExponent = exponent - firstExponent;
    float firstScale = as_type<float>(uint(firstExponent + 127) << 23);
    float secondScale = as_type<float>(uint(secondExponent + 127) << 23);
    float result = (polynomial * firstScale) * secondScale;
    if (value < -104.0f) result = 0.0f;
    if (value > 100.0f) result = INFINITY;
    return result;
}

inline float softmax_reciprocal_sleef(
    threadgroup const bfloat* scores,
    uint count,
    float maximum
) {
    float4 sums = 0.0f;
    uint position = 0;
    for (; position + 3 < count; position += 4) {
        sums[0] += sleef_exp_u10(float(scores[position]) - maximum);
        sums[1] += sleef_exp_u10(float(scores[position + 1]) - maximum);
        sums[2] += sleef_exp_u10(float(scores[position + 2]) - maximum);
        sums[3] += sleef_exp_u10(float(scores[position + 3]) - maximum);
    }
    for (; position < count; ++position) {
        sums[position & 3] += sleef_exp_u10(float(scores[position]) - maximum);
    }
    float denominator = (sums.x + sums.z) + (sums.y + sums.w);
    return 1.0f / denominator;
}

// Contiguous BF16 square reduction with the four-row/four-level cascade used
// by the reference FP32 sum kernel. This is a shape-neutral reduction brick:
// head norms and trunk norms share it at different last-dimension widths.
inline float cascade_bf16_square_sum(
    device const bfloat* input,
    uint count
) {
    constexpr uint rowCount = 4;
    constexpr uint levelCount = 4;
    float4 cascade[levelCount][rowCount];
    for (uint level = 0; level < levelCount; ++level) {
        for (uint row = 0; row < rowCount; ++row) {
            cascade[level][row] = 0.0f;
        }
    }

    uint vectorCount = count / 4;
    uint interleavedRows = vectorCount / rowCount;
    uint ceilLog2 = 0;
    uint ceiling = 1;
    while (ceiling < interleavedRows) {
        ceiling <<= 1;
        ++ceilLog2;
    }
    uint levelPower = max(4u, ceilLog2 / levelCount);
    uint levelStep = 1u << levelPower;
    uint levelMask = levelStep - 1u;
    uint interleavedRow = 0;
    for (; interleavedRow + levelStep <= interleavedRows;) {
        for (uint localRow = 0; localRow < levelStep; ++localRow, ++interleavedRow) {
            for (uint row = 0; row < rowCount; ++row) {
                uint vectorIndex = interleavedRow * rowCount + row;
                float4 value = float4(
                    *((device const bfloat4*)(input + vectorIndex * 4))
                );
                cascade[0][row] += fma(value, value, 0.0f);
            }
        }
        for (uint level = 1; level < levelCount; ++level) {
            for (uint row = 0; row < rowCount; ++row) {
                cascade[level][row] += cascade[level - 1][row];
                cascade[level - 1][row] = 0.0f;
            }
            uint mask = levelMask << (level * levelPower);
            if ((interleavedRow & mask) != 0) break;
        }
    }
    for (; interleavedRow < interleavedRows; ++interleavedRow) {
        for (uint row = 0; row < rowCount; ++row) {
            uint vectorIndex = interleavedRow * rowCount + row;
            float4 value = float4(
                *((device const bfloat4*)(input + vectorIndex * 4))
            );
            cascade[0][row] += fma(value, value, 0.0f);
        }
    }
    for (uint level = 1; level < levelCount; ++level) {
        for (uint row = 0; row < rowCount; ++row) {
            cascade[0][row] += cascade[level][row];
        }
    }

    uint vectorIndex = interleavedRows * rowCount;
    for (; vectorIndex < vectorCount; ++vectorIndex) {
        float4 value = float4(
            *((device const bfloat4*)(input + vectorIndex * 4))
        );
        cascade[0][0] += fma(value, value, 0.0f);
    }
    for (uint row = 1; row < rowCount; ++row) {
        cascade[0][0] += cascade[0][row];
    }

    float total = 0.0f;
    total += cascade[0][0].x;
    total += cascade[0][0].y;
    total += cascade[0][0].z;
    total += cascade[0][0].w;
    for (uint index = vectorCount * 4; index < count; ++index) {
        float value = float(input[index]);
        total += fma(value, value, 0.0f);
    }
    return total;
}

// FP32-accumulating BF16 dot with the 8x4 accumulator topology used by the
// canonical arm64 reference backend. This is an arithmetic contract, not a
// model contract: any dense BF16 projection may request it near a BF16
// rounding midpoint.
inline float pairwise8x4_bf16_dot(
    device const bfloat* input,
    device const bfloat* weight,
    uint columns
) {
    float4 sums[8];
    for (uint accumulator = 0; accumulator < 8; ++accumulator) {
        sums[accumulator] = 0.0f;
    }

    uint column = 0;
    for (; column + 31 < columns; column += 32) {
        for (uint accumulator = 0; accumulator < 8; ++accumulator) {
            uint offset = column + accumulator * 4;
            float4 x = float4(*((device const bfloat4*)(input + offset)));
            float4 w = float4(*((device const bfloat4*)(weight + offset)));
            sums[accumulator] = fma(x, w, sums[accumulator]);
        }
    }
    for (uint offset = 4; offset > 0; offset >>= 1) {
        for (uint accumulator = 0; accumulator < offset; ++accumulator) {
            sums[accumulator] += sums[accumulator + offset];
        }
    }
    float total = (sums[0].x + sums[0].y) + (sums[0].z + sums[0].w);

    // The reference backend consumes an eight-element vector tail as two
    // consecutive four-wide FMAs, then reduces it independently.
    float4 tail = 0.0f;
    for (; column + 7 < columns; column += 8) {
        float4 x = float4(*((device const bfloat4*)(input + column)));
        float4 w = float4(*((device const bfloat4*)(weight + column)));
        tail = fma(x, w, tail);
        x = float4(*((device const bfloat4*)(input + column + 4)));
        w = float4(*((device const bfloat4*)(weight + column + 4)));
        tail = fma(x, w, tail);
    }
    total += (tail.x + tail.y) + (tail.z + tail.w);
    for (; column < columns; ++column) {
        total += float(input[column]) * float(weight[column]);
    }
    return total;
}

// BF16 dot for a contiguous coefficient vector and a strided matrix column.
// The four interleaved FP32 accumulators are the generic GEMM reduction
// contract used when the right-hand operand is not row-contiguous.
inline float ilp4_bf16_strided_dot(
    threadgroup const bfloat* coefficients,
    device const bfloat* values,
    uint activeCount,
    uint contractionCount,
    uint valueStride
) {
    float partials[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint index = 0;
    for (; index + 3 < activeCount; index += 4) {
        partials[0] = fma(
            float(coefficients[index]),
            float(values[index * valueStride]),
            partials[0]
        );
        partials[1] = fma(
            float(coefficients[index + 1]),
            float(values[(index + 1) * valueStride]),
            partials[1]
        );
        partials[2] = fma(
            float(coefficients[index + 2]),
            float(values[(index + 2) * valueStride]),
            partials[2]
        );
        partials[3] = fma(
            float(coefficients[index + 3]),
            float(values[(index + 3) * valueStride]),
            partials[3]
        );
    }
    // The reference GEMM puts a real contraction tail into accumulator zero.
    // A masked causal row still contracts across the padded full sequence;
    // its inactive zeros leave active tail values in their absolute ILP lane.
    bool paddedContraction = activeCount < contractionCount;
    for (; index < activeCount; ++index) {
        uint accumulator = paddedContraction ? (index & 3) : 0u;
        partials[accumulator] = fma(
            float(coefficients[index]),
            float(values[index * valueStride]),
            partials[accumulator]
        );
    }
    partials[0] += partials[1];
    partials[0] += partials[2];
    partials[0] += partials[3];
    return partials[0];
}

inline float stable_bf16_dot(
    float parallelTotal,
    device const bfloat* input,
    device const bfloat* weight,
    uint columns
) {
    return bf16_rounding_boundary(parallelTotal)
        ? pairwise8x4_bf16_dot(input, weight, columns)
        : parallelTotal;
}

kernel void rms_norm_codec_bf16(
    device const bfloat* input  [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device bfloat* output       [[buffer(2)]],
    constant uint& frames       [[buffer(3)]],
    constant uint& dim          [[buffer(4)]],
    constant float& eps         [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup float sharedInverse[1];
    if (row >= frames) return;
    uint base = row * dim;
    if (lane == 0) {
        float squareSum = cascade_bf16_square_sum(input + base, dim);
        sharedInverse[0] = bf16_reference_rsqrt(squareSum / float(dim) + eps);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float inverse = sharedInverse[0];
    for (uint i = lane; i < dim; i += 32) {
        bfloat normalized = bfloat(float(input[base + i]) * inverse);
        output[base + i] = bfloat(float(normalized) * float(weight[i]));
    }
}

kernel void rms_norm_head_bf16(
    device const bfloat* input  [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device bfloat* output       [[buffer(2)]],
    constant uint& frames       [[buffer(3)]],
    constant uint& heads        [[buffer(4)]],
    constant uint& headDim      [[buffer(5)]],
    constant float& eps         [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup float sharedInverse[1];
    if (row >= frames * heads) return;
    uint base = row * headDim;
    if (lane == 0) {
        float squareSum = cascade_bf16_square_sum(input + base, headDim);
        sharedInverse[0] = bf16_reference_rsqrt(squareSum / float(headDim) + eps);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float inverse = sharedInverse[0];
    for (uint d = lane; d < headDim; d += 32) {
        bfloat normalized = bfloat(float(input[base + d]) * inverse);
        output[base + d] = bfloat(float(normalized) * float(weight[d]));
    }
}

kernel void head_norm_rope_bf16(
    device const bfloat* input  [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const bfloat* cosines [[buffer(2)]],
    device const bfloat* sines   [[buffer(3)]],
    device bfloat* output       [[buffer(4)]],
    constant uint& heads        [[buffer(5)]],
    constant uint& headDim      [[buffer(6)]],
    constant float& eps         [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup float sharedInverse[1];
    if (head >= heads) return;
    uint base = head * headDim;
    if (lane == 0) {
        float squareSum = cascade_bf16_square_sum(input + base, headDim);
        sharedInverse[0] = bf16_reference_rsqrt(
            squareSum / float(headDim) + eps
        );
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float inverse = sharedInverse[0];
    uint halfDim = headDim / 2;
    for (uint j = lane; j < halfDim; j += 32) {
        bfloat normalizedA = bfloat(float(input[base + j]) * inverse);
        bfloat normalizedB = bfloat(float(input[base + j + halfDim]) * inverse);
        bfloat a = bfloat(float(normalizedA) * float(weight[j]));
        bfloat b = bfloat(float(normalizedB) * float(weight[j + halfDim]));
        bfloat first = bfloat(float(a) * float(cosines[j]));
        bfloat second = bfloat(-float(b) * float(sines[j]));
        output[base + j] = bfloat(float(first) + float(second));
        first = bfloat(float(b) * float(cosines[j + halfDim]));
        second = bfloat(float(a) * float(sines[j + halfDim]));
        output[base + j + halfDim] = bfloat(float(first) + float(second));
    }
}

kernel void gemv_qkv_bf16(
    device const bfloat* input  [[buffer(0)]],
    device const bfloat* qWeight [[buffer(1)]],
    device const bfloat* kWeight [[buffer(2)]],
    device const bfloat* vWeight [[buffer(3)]],
    device bfloat* qOutput      [[buffer(4)]],
    device bfloat* kOutput      [[buffer(5)]],
    device bfloat* vOutput      [[buffer(6)]],
    constant uint& qRows        [[buffer(7)]],
    constant uint& kRows        [[buffer(8)]],
    constant uint& vRows        [[buffer(9)]],
    constant uint& columns      [[buffer(10)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    uint totalRows = qRows + kRows + vRows;
    if (row >= totalRows) return;
    device const bfloat* weight;
    device bfloat* output;
    uint localRow;
    if (row < qRows) {
        weight = qWeight;
        output = qOutput;
        localRow = row;
    } else if (row < qRows + kRows) {
        weight = kWeight;
        output = kOutput;
        localRow = row - qRows;
    } else {
        weight = vWeight;
        output = vOutput;
        localRow = row - qRows - kRows;
    }
    uint chunks = columns >> 2;
    device const bfloat4* input4 = (device const bfloat4*)input;
    device const bfloat4* weight4 =
        (device const bfloat4*)(weight + localRow * columns);
    float partial = 0.0f;
    for (uint chunk = lane; chunk < chunks; chunk += 32) {
        partial += dot(float4(input4[chunk]), float4(weight4[chunk]));
    }
    float total = simd_sum(partial);
    if (lane == 0) {
        total = stable_bf16_dot(total, input, weight + localRow * columns, columns);
        output[localRow] = bfloat(total);
    }
}

kernel void gemv_add_bf16(
    device const bfloat* input  [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const bfloat* residual [[buffer(2)]],
    device bfloat* output       [[buffer(3)]],
    constant uint& rows         [[buffer(4)]],
    constant uint& columns      [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (row >= rows) return;
    uint chunks = columns >> 2;
    device const bfloat4* input4 = (device const bfloat4*)input;
    device const bfloat4* weight4 =
        (device const bfloat4*)(weight + row * columns);
    float partial = 0.0f;
    for (uint chunk = lane; chunk < chunks; chunk += 32) {
        partial += dot(float4(input4[chunk]), float4(weight4[chunk]));
    }
    float total = simd_sum(partial);
    if (lane == 0) {
        total = stable_bf16_dot(total, input, weight + row * columns, columns);
        bfloat projected = bfloat(total);
        output[row] = bfloat(float(projected) + float(residual[row]));
    }
}

kernel void gemv_gateup_swiglu_bf16(
    device const bfloat* input  [[buffer(0)]],
    device const bfloat* gateWeight [[buffer(1)]],
    device const bfloat* upWeight   [[buffer(2)]],
    device bfloat* output       [[buffer(3)]],
    constant uint& rows         [[buffer(4)]],
    constant uint& columns      [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (row >= rows) return;
    uint chunks = columns >> 2;
    device const bfloat4* input4 = (device const bfloat4*)input;
    device const bfloat4* gate4 =
        (device const bfloat4*)(gateWeight + row * columns);
    device const bfloat4* up4 =
        (device const bfloat4*)(upWeight + row * columns);
    float gatePartial = 0.0f;
    float upPartial = 0.0f;
    for (uint chunk = lane; chunk < chunks; chunk += 32) {
        float4 value = float4(input4[chunk]);
        gatePartial += dot(value, float4(gate4[chunk]));
        upPartial += dot(value, float4(up4[chunk]));
    }
    float gateTotal = simd_sum(gatePartial);
    float upTotal = simd_sum(upPartial);
    if (lane == 0) {
        gateTotal = stable_bf16_dot(
            gateTotal,
            input,
            gateWeight + row * columns,
            columns
        );
        upTotal = stable_bf16_dot(
            upTotal,
            input,
            upWeight + row * columns,
            columns
        );
        bfloat gate = bfloat(gateTotal);
        bfloat up = bfloat(upTotal);
        bfloat silu = bfloat(float(gate) / (1.0f + exp(-float(gate))));
        output[row] = bfloat(float(silu) * float(up));
    }
}

#define DENSE_TRUNK_BF16_MAX_CONTEXT 3192

kernel void decode_gqa_attn_bf16(
    device const bfloat* query [[buffer(0)]],
    device const bfloat* keyCache [[buffer(1)]],
    device const bfloat* valueCache [[buffer(2)]],
    device bfloat* output      [[buffer(3)]],
    constant uint& cacheLength [[buffer(4)]],
    constant uint& heads       [[buffer(5)]],
    constant uint& kvHeads     [[buffer(6)]],
    constant uint& headDim     [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup bfloat scores[DENSE_TRUNK_BF16_MAX_CONTEXT];
    threadgroup float probabilityScale[1];
    if (head >= heads || cacheLength > DENSE_TRUNK_BF16_MAX_CONTEXT) return;
    uint kvHead = head / (heads / kvHeads);
    uint kvDim = kvHeads * headDim;
    uint queryBase = head * headDim;
    float scale = bf16_reference_rsqrt(float(headDim));
    for (uint position = lane; position < cacheLength; position += 32) {
        uint keyBase = position * kvDim + kvHead * headDim;
        float dotValue = pairwise8x4_bf16_dot(
            query + queryBase,
            keyCache + keyBase,
            headDim
        );
        bfloat matmul = bfloat(dotValue);
        scores[position] = bfloat(float(matmul) * scale);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float maximum = -INFINITY;
        for (uint position = 0; position < cacheLength; ++position) {
            maximum = max(maximum, float(scores[position]));
        }
        probabilityScale[0] = softmax_reciprocal_sleef(scores, cacheLength, maximum);
        for (uint position = 0; position < cacheLength; ++position) {
            scores[position] = bfloat(
                sleef_exp_u10(float(scores[position]) - maximum) * probabilityScale[0]
            );
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = lane; d < headDim; d += 32) {
        uint valueBase = kvHead * headDim + d;
        float accumulated = ilp4_bf16_strided_dot(
            scores,
            valueCache + valueBase,
            cacheLength,
            cacheLength,
            kvDim
        );
        output[queryBase + d] = bfloat(accumulated);
    }
}

kernel void gemm_bf16(
    device const bfloat* input  [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const bfloat* bias   [[buffer(2)]],
    device bfloat* output       [[buffer(3)]],
    constant uint& inputRows    [[buffer(4)]],
    constant uint& outputRows   [[buffer(5)]],
    constant uint& columns      [[buffer(6)]],
    constant uint& hasBias      [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    uint outputRow = group.x;
    uint inputRow = group.y;
    if (outputRow >= outputRows || inputRow >= inputRows) return;
    uint chunks = columns >> 2;
    device const bfloat4* input4 =
        (device const bfloat4*)(input + inputRow * columns);
    device const bfloat4* weight4 =
        (device const bfloat4*)(weight + outputRow * columns);
    float partial = 0.0f;
    for (uint chunk = lane; chunk < chunks; chunk += 32) {
        partial += dot(float4(input4[chunk]), float4(weight4[chunk]));
    }
    float total = simd_sum(partial);
    if (lane == 0) {
        total = stable_bf16_dot(
            total,
            input + inputRow * columns,
            weight + outputRow * columns,
            columns
        );
        output[inputRow * outputRows + outputRow] = bfloat(
            total + (hasBias != 0 ? float(bias[outputRow]) : 0.0f)
        );
    }
}

kernel void rope_apply_bf16(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* cosines [[buffer(1)]],
    device const bfloat* sines [[buffer(2)]],
    device bfloat* output      [[buffer(3)]],
    constant uint& frames      [[buffer(4)]],
    constant uint& heads       [[buffer(5)]],
    constant uint& headDim     [[buffer(6)]],
    uint index [[thread_position_in_grid]]
) {
    uint halfDim = headDim / 2;
    if (index >= frames * heads * halfDim) return;
    uint column = index % halfDim;
    uint remaining = index / halfDim;
    uint head = remaining % heads;
    uint frame = remaining / heads;
    uint base = frame * heads * headDim + head * headDim;
    uint ropeBase = frame * headDim;
    bfloat a = input[base + column];
    bfloat b = input[base + column + halfDim];
    bfloat first = bfloat(float(a) * float(cosines[ropeBase + column]));
    bfloat second = bfloat(-float(b) * float(sines[ropeBase + column]));
    output[base + column] = bfloat(float(first) + float(second));
    first = bfloat(float(b) * float(cosines[ropeBase + column + halfDim]));
    second = bfloat(float(a) * float(sines[ropeBase + column + halfDim]));
    output[base + column + halfDim] = bfloat(float(first) + float(second));
}

kernel void causal_gqa_attn_cached_bf16(
    device const bfloat* query [[buffer(0)]],
    device const bfloat* keyCache [[buffer(1)]],
    device const bfloat* valueCache [[buffer(2)]],
    device bfloat* output      [[buffer(3)]],
    constant uint& frames      [[buffer(4)]],
    constant uint& heads       [[buffer(5)]],
    constant uint& kvHeads     [[buffer(6)]],
    constant uint& headDim     [[buffer(7)]],
    constant uint& startPosition [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup bfloat scores[DENSE_TRUNK_BF16_MAX_CONTEXT];
    threadgroup float probabilityScale[1];
    uint frame = group.x;
    uint head = group.y;
    if (frame >= frames || head >= heads) return;
    uint absoluteFrame = startPosition + frame;
    if (absoluteFrame >= DENSE_TRUNK_BF16_MAX_CONTEXT) return;
    uint kvHead = head / (heads / kvHeads);
    uint queryDim = heads * headDim;
    uint kvDim = kvHeads * headDim;
    uint queryBase = frame * queryDim + head * headDim;
    float scale = bf16_reference_rsqrt(float(headDim));
    for (uint position = lane; position <= absoluteFrame; position += 32) {
        uint keyBase = position * kvDim + kvHead * headDim;
        float dotValue = pairwise8x4_bf16_dot(
            query + queryBase,
            keyCache + keyBase,
            headDim
        );
        bfloat matmul = bfloat(dotValue);
        scores[position] = bfloat(float(matmul) * scale);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint scoreCount = absoluteFrame + 1;
    if (lane == 0) {
        float maximum = -INFINITY;
        for (uint position = 0; position < scoreCount; ++position) {
            maximum = max(maximum, float(scores[position]));
        }
        probabilityScale[0] = softmax_reciprocal_sleef(scores, scoreCount, maximum);
        for (uint position = 0; position < scoreCount; ++position) {
            scores[position] = bfloat(
                sleef_exp_u10(float(scores[position]) - maximum) * probabilityScale[0]
            );
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint outputBase = frame * queryDim + head * headDim;
    for (uint d = lane; d < headDim; d += 32) {
        uint valueBase = kvHead * headDim + d;
        float accumulated = ilp4_bf16_strided_dot(
            scores,
            valueCache + valueBase,
            absoluteFrame + 1,
            startPosition + frames,
            kvDim
        );
        output[outputBase + d] = bfloat(accumulated);
    }
}

kernel void scale_residual_tc_bf16(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* residual [[buffer(1)]],
    device const bfloat* scale [[buffer(2)]],
    device bfloat* output      [[buffer(3)]],
    constant uint& channels    [[buffer(4)]],
    constant uint& frames      [[buffer(5)]],
    constant uint& hasScale    [[buffer(6)]],
    uint2 index [[thread_position_in_grid]]
) {
    uint frame = index.x;
    uint channel = index.y;
    if (frame >= frames || channel >= channels) return;
    uint offset = frame * channels + channel;
    bfloat value = hasScale != 0
        ? bfloat(float(scale[channel]) * float(input[offset]))
        : input[offset];
    output[offset] = bfloat(float(residual[offset]) + float(value));
}

kernel void swiglu_bf16(
    device const bfloat* gate [[buffer(0)]],
    device const bfloat* up   [[buffer(1)]],
    device bfloat* output     [[buffer(2)]],
    constant uint& count      [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    bfloat silu = bfloat(float(gate[index]) / (1.0f + exp(-float(gate[index]))));
    output[index] = bfloat(float(silu) * float(up[index]));
}

kernel void gather_row_bf16(
    device const bfloat* table [[buffer(0)]],
    device const uint* ids     [[buffer(1)]],
    device bfloat* output      [[buffer(2)]],
    constant uint& dim         [[buffer(3)]],
    constant uint& slot        [[buffer(4)]],
    uint column [[thread_position_in_grid]]
) {
    if (column >= dim) return;
    output[slot * dim + column] = table[ids[slot] * dim + column];
}

kernel void dense_bf16(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device const bfloat* bias [[buffer(2)]],
    device bfloat* output     [[buffer(3)]],
    constant uint& inputRows  [[buffer(4)]],
    constant uint& outputRows [[buffer(5)]],
    constant uint& columns    [[buffer(6)]],
    constant uint& hasBias    [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    uint outputRow = group.x;
    uint inputRow = group.y;
    if (outputRow >= outputRows || inputRow >= inputRows) return;
    uint chunks = columns >> 2;
    device const bfloat4* input4 =
        (device const bfloat4*)(input + inputRow * columns);
    device const bfloat4* weight4 =
        (device const bfloat4*)(weight + outputRow * columns);
    float partial = 0.0f;
    for (uint chunk = lane; chunk < chunks; chunk += 32) {
        partial += dot(float4(input4[chunk]), float4(weight4[chunk]));
    }
    float total = simd_sum(partial);
    if (lane == 0) {
        total = stable_bf16_dot(
            total,
            input + inputRow * columns,
            weight + outputRow * columns,
            columns
        );
        output[inputRow * outputRows + outputRow] = bfloat(
            total + (hasBias != 0 ? float(bias[outputRow]) : 0.0f)
        );
    }
}
