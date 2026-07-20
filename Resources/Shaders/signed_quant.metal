// Native signed binary/ternary weight kernels.
//
// Canonical storage is row-major LSB-first packed codes plus one fp16 scale per
// g128 column group. Binary uses one bit per weight; ternary uses the semantic
// two-bit codes 0/1/2 for -1/0/+1. This is the same source-neutral spelling
// consumed by AgentSignedQuantCodec and avoids a package-build transpose.

#include <metal_stdlib>
using namespace metal;

// Shared semantic boundary for every fused signed-view SwiGLU consumer.
// Keep this identical to activations_precise.metal: MLX materializes the fp16
// sigmoid, fp16 SiLU activation, and fp16 outer multiply as separate graph
// values. Fusion may remove dispatches, but it must not remove those rounds.
inline half smelt_signed_swiglu_staged_half(half gate, half up) {
    const half tail = 1.0h / (1.0h + exp(abs(gate)));
    const half sigmoid = gate < 0.0h ? tail : 1.0h - tail;
    const half activated = gate * sigmoid;
    return activated * up;
}

inline half smelt_binary_dot4(half4 x, uint nibble) {
    const half4 signs = select(
        half4(-1.0h),
        half4(1.0h),
        (uint4(nibble) & uint4(1u, 2u, 4u, 8u)) != uint4(0u));
    return dot(x, signs);
}

inline float smelt_ternary_planar_dot4(
    half4 x,
    uint positive,
    uint negative
) {
    return 0.5f * (
        float(smelt_binary_dot4(x, positive))
        - float(smelt_binary_dot4(x, negative)));
}

// Convert 32 interleaved two-bit ternary codes into positive/negative masks
// when a popcount-based consumer needs sign planes. Exact affine kernels read
// the packed codes directly and do not pay this conversion.
inline uint smelt_ternary_compact_even_bits(uint value) {
    value &= 0x55555555u;
    value = (value | (value >> 1)) & 0x33333333u;
    value = (value | (value >> 2)) & 0x0f0f0f0fu;
    value = (value | (value >> 4)) & 0x00ff00ffu;
    value = (value | (value >> 8)) & 0x0000ffffu;
    return value;
}

inline uint2 smelt_ternary_sign_masks_32(uint2 packed) {
    uint positive = 0u;
    uint negative = 0u;
    for (uint part = 0; part < 2; ++part) {
        const uint word = packed[part];
        const uint low = word & 0x55555555u;
        const uint high = (word >> 1) & 0x55555555u;
        positive |= smelt_ternary_compact_even_bits(high & ~low) << (part * 16);
        negative |= smelt_ternary_compact_even_bits(~(high | low)) << (part * 16);
    }
    return uint2(positive, negative);
}

inline void smelt_ternary_sign_masks_128(
    device const uchar *codes,
    thread uint4 &positive,
    thread uint4 &negative
) {
    const device uint4 *packed = (device const uint4 *)codes;
    const uint4 first = packed[0];
    const uint4 second = packed[1];
    const uint2 masks0 = smelt_ternary_sign_masks_32(uint2(first.x, first.y));
    const uint2 masks1 = smelt_ternary_sign_masks_32(uint2(first.z, first.w));
    const uint2 masks2 = smelt_ternary_sign_masks_32(uint2(second.x, second.y));
    const uint2 masks3 = smelt_ternary_sign_masks_32(uint2(second.z, second.w));
    positive = uint4(masks0.x, masks1.x, masks2.x, masks3.x);
    negative = uint4(masks0.y, masks1.y, masks2.y, masks3.y);
}

inline void smelt_binary_matvec_g128_rows4(
    device const uchar *codes,
    device const half *scales,
    device const half *input,
    device half *output,
    uint rows,
    uint cols,
    uint2 tgid,
    uint simd_gid,
    uint lane
) {
    constexpr uint GROUP_SIZE = 128;
    constexpr uint ROW_TILE = 4;
    constexpr uint COLS_PER_LANE = 32;
    constexpr uint COLS_PER_SIMDGROUP = COLS_PER_LANE * 32;
    const uint row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint groups = cols / GROUP_SIZE;
    const uint row_bytes = cols / 8;
    device const half *batch_input = input + tgid.y * cols;

    float4 acc = 0.0f;
    for (uint block = 0; block < cols; block += COLS_PER_SIMDGROUP) {
        const uint col = block + lane * COLS_PER_LANE;
        if (col >= cols) continue;
        const uint group = col / GROUP_SIZE;
        uint4 packed = 0u;
        half4 scale = 0.0h;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row >= rows) continue;
            packed[tile] = *(device const uint *)(
                codes + row * row_bytes + col / 8);
            scale[tile] = scales[row * groups + group];
        }

        half4 partial = 0.0h;
        for (uint i = 0; i < COLS_PER_LANE; i += 4) {
            const half4 x = *(device const half4 *)(batch_input + col + i);
            partial.x += smelt_binary_dot4(x, packed.x >> i);
            partial.y += smelt_binary_dot4(x, packed.y >> i);
            partial.z += smelt_binary_dot4(x, packed.z >> i);
            partial.w += smelt_binary_dot4(x, packed.w >> i);
        }
        acc += float4(partial * scale);
    }

    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    acc.z = simd_sum(acc.z);
    acc.w = simd_sum(acc.w);
    if (lane == 0) {
        device half *batch_output = output + tgid.y * rows;
        if (row0 + 0 < rows) batch_output[row0 + 0] = half(acc.x);
        if (row0 + 1 < rows) batch_output[row0 + 1] = half(acc.y);
        if (row0 + 2 < rows) batch_output[row0 + 2] = half(acc.z);
        if (row0 + 3 < rows) batch_output[row0 + 3] = half(acc.w);
    }
}

inline void smelt_binary_gate_up_swiglu_g128_rows4(
    device const uchar *gate_codes,
    device const half *gate_scales,
    device const uchar *up_codes,
    device const half *up_scales,
    device const half *input,
    device half *output,
    uint rows,
    uint cols,
    uint2 tgid,
    uint simd_gid,
    uint lane
) {
    constexpr uint GROUP_SIZE = 128;
    constexpr uint ROW_TILE = 4;
    constexpr uint COLS_PER_LANE = 32;
    constexpr uint COLS_PER_SIMDGROUP = COLS_PER_LANE * 32;
    const uint row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint groups = cols / GROUP_SIZE;
    const uint row_bytes = cols / 8;
    device const half *batch_input = input + tgid.y * cols;

    float4 gate_acc = 0.0f;
    float4 up_acc = 0.0f;
    for (uint block = 0; block < cols; block += COLS_PER_SIMDGROUP) {
        const uint col = block + lane * COLS_PER_LANE;
        if (col >= cols) continue;
        const uint group = col / GROUP_SIZE;
        uint4 gate_packed = 0u;
        uint4 up_packed = 0u;
        half4 gate_scale = 0.0h;
        half4 up_scale = 0.0h;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row >= rows) continue;
            gate_packed[tile] = *(device const uint *)(
                gate_codes + row * row_bytes + col / 8);
            up_packed[tile] = *(device const uint *)(
                up_codes + row * row_bytes + col / 8);
            gate_scale[tile] = gate_scales[row * groups + group];
            up_scale[tile] = up_scales[row * groups + group];
        }

        half4 gate_partial = 0.0h;
        half4 up_partial = 0.0h;
        for (uint i = 0; i < COLS_PER_LANE; i += 4) {
            const half4 x = *(device const half4 *)(batch_input + col + i);
            gate_partial.x += smelt_binary_dot4(x, gate_packed.x >> i);
            gate_partial.y += smelt_binary_dot4(x, gate_packed.y >> i);
            gate_partial.z += smelt_binary_dot4(x, gate_packed.z >> i);
            gate_partial.w += smelt_binary_dot4(x, gate_packed.w >> i);
            up_partial.x += smelt_binary_dot4(x, up_packed.x >> i);
            up_partial.y += smelt_binary_dot4(x, up_packed.y >> i);
            up_partial.z += smelt_binary_dot4(x, up_packed.z >> i);
            up_partial.w += smelt_binary_dot4(x, up_packed.w >> i);
        }
        gate_acc += float4(gate_partial * gate_scale);
        up_acc += float4(up_partial * up_scale);
    }

    gate_acc.x = simd_sum(gate_acc.x);
    gate_acc.y = simd_sum(gate_acc.y);
    gate_acc.z = simd_sum(gate_acc.z);
    gate_acc.w = simd_sum(gate_acc.w);
    up_acc.x = simd_sum(up_acc.x);
    up_acc.y = simd_sum(up_acc.y);
    up_acc.z = simd_sum(up_acc.z);
    up_acc.w = simd_sum(up_acc.w);
    if (lane == 0) {
        const half4 gate = half4(gate_acc);
        const half4 up = half4(up_acc);
        device half *batch_output = output + tgid.y * rows;
        if (row0 + 0 < rows) batch_output[row0 + 0] =
            smelt_signed_swiglu_staged_half(gate.x, up.x);
        if (row0 + 1 < rows) batch_output[row0 + 1] =
            smelt_signed_swiglu_staged_half(gate.y, up.y);
        if (row0 + 2 < rows) batch_output[row0 + 2] =
            smelt_signed_swiglu_staged_half(gate.z, up.z);
        if (row0 + 3 < rows) batch_output[row0 + 3] =
            smelt_signed_swiglu_staged_half(gate.w, up.w);
    }
}

inline void smelt_binary_matvec_add_g128_rows4(
    device const uchar *codes,
    device const half *scales,
    device const half *input,
    device half *output,
    device const half *residual,
    uint rows,
    uint cols,
    uint2 tgid,
    uint simd_gid,
    uint lane
) {
    constexpr uint GROUP_SIZE = 128;
    constexpr uint ROW_TILE = 4;
    constexpr uint COLS_PER_LANE = 32;
    constexpr uint COLS_PER_SIMDGROUP = COLS_PER_LANE * 32;
    const uint row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint groups = cols / GROUP_SIZE;
    const uint row_bytes = cols / 8;
    device const half *batch_input = input + tgid.y * cols;

    float4 acc = 0.0f;
    for (uint block = 0; block < cols; block += COLS_PER_SIMDGROUP) {
        const uint col = block + lane * COLS_PER_LANE;
        if (col >= cols) continue;
        const uint group = col / GROUP_SIZE;
        uint4 packed = 0u;
        half4 scale = 0.0h;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row >= rows) continue;
            packed[tile] = *(device const uint *)(
                codes + row * row_bytes + col / 8);
            scale[tile] = scales[row * groups + group];
        }

        half4 partial = 0.0h;
        for (uint i = 0; i < COLS_PER_LANE; i += 4) {
            const half4 x = *(device const half4 *)(batch_input + col + i);
            partial.x += smelt_binary_dot4(x, packed.x >> i);
            partial.y += smelt_binary_dot4(x, packed.y >> i);
            partial.z += smelt_binary_dot4(x, packed.z >> i);
            partial.w += smelt_binary_dot4(x, packed.w >> i);
        }
        acc += float4(partial * scale);
    }

    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    acc.z = simd_sum(acc.z);
    acc.w = simd_sum(acc.w);
    if (lane == 0) {
        device half *batch_output = output + tgid.y * rows;
        device const half *batch_residual = residual + tgid.y * rows;
        if (row0 + 0 < rows) batch_output[row0 + 0] = half(
            float(half(acc.x)) + float(batch_residual[row0 + 0]));
        if (row0 + 1 < rows) batch_output[row0 + 1] = half(
            float(half(acc.y)) + float(batch_residual[row0 + 1]));
        if (row0 + 2 < rows) batch_output[row0 + 2] = half(
            float(half(acc.z)) + float(batch_residual[row0 + 2]));
        if (row0 + 3 < rows) batch_output[row0 + 3] = half(
            float(half(acc.w)) + float(batch_residual[row0 + 3]));
    }
}

inline void smelt_ternary_matvec_g128_rows4(
    device const uchar *codes,
    device const half *scales,
    device const half *input,
    device half *output,
    uint rows,
    uint cols,
    uint2 tgid,
    uint simd_gid,
    uint lane
) {
    constexpr uint GROUP_SIZE = 128;
    constexpr uint COLS_PER_LANE = 32;
    constexpr uint COLS_PER_SIMDGROUP = COLS_PER_LANE * 32;
    const uint row = tgid.x * 2 + simd_gid;
    if (row >= rows) return;
    const uint groups = cols / GROUP_SIZE;
    const uint row_bytes = cols / 4;
    device const half *batch_input = input + tgid.y * cols;

    float acc = 0.0f;
    for (uint block = 0; block < cols; block += COLS_PER_SIMDGROUP) {
        const uint col = block + lane * COLS_PER_LANE;
        if (col >= cols) continue;
        const uint group = col / GROUP_SIZE;
        const uint2 packed = *(device const uint2 *)(
            codes + row * row_bytes + col / 4);
        const uint2 masks = smelt_ternary_sign_masks_32(packed);
        const float scale = float(scales[row * groups + group]);

        float partial = 0.0f;
        for (uint i = 0; i < COLS_PER_LANE; i += 4) {
            const half4 x = *(device const half4 *)(batch_input + col + i);
            partial += smelt_ternary_planar_dot4(
                x, masks.x >> i, masks.y >> i);
        }
        acc += partial * scale;
    }

    acc = simd_sum(acc);
    if (lane == 0) {
        device half *batch_output = output + tgid.y * rows;
        batch_output[row] = half(acc);
    }
}

// Batch-tiled signed kernels keep the per-token accumulation order identical
// to the scalar-batch path, but load each packed weight and scale tile once for
// four activation rows. This is selected by storage geometry plus batch shape;
// it is not tied to a model or to speculative decoding.
inline void smelt_binary_matvec_g128_rows4_batched_b4(
    device const uchar *codes,
    device const half *scales,
    device const half *input,
    device half *output,
    uint rows,
    uint cols,
    uint actual_batch,
    uint2 tgid,
    uint simd_gid,
    uint lane
) {
    constexpr uint GROUP_SIZE = 128;
    constexpr uint ROW_TILE = 4;
    constexpr uint BATCH_TILE = 4;
    constexpr uint COLS_PER_LANE = 32;
    constexpr uint COLS_PER_SIMDGROUP = COLS_PER_LANE * 32;
    const uint row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint batch0 = tgid.y * BATCH_TILE;
    const uint groups = cols / GROUP_SIZE;
    const uint row_bytes = cols / 8;
    float4 acc[BATCH_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint block = 0; block < cols; block += COLS_PER_SIMDGROUP) {
        const uint col = block + lane * COLS_PER_LANE;
        if (col >= cols) continue;
        const uint group = col / GROUP_SIZE;
        uint4 packed = 0u;
        half4 scale = 0.0h;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row >= rows) continue;
            packed[tile] = *(device const uint *)(
                codes + row * row_bytes + col / 8);
            scale[tile] = scales[row * groups + group];
        }

        for (uint b = 0; b < BATCH_TILE; ++b) {
            if (batch0 + b >= actual_batch) continue;
            device const half *batch_input = input + (batch0 + b) * cols;
            half4 partial = 0.0h;
            for (uint i = 0; i < COLS_PER_LANE; i += 4) {
                const half4 x = *(device const half4 *)(batch_input + col + i);
                partial.x += smelt_binary_dot4(x, packed.x >> i);
                partial.y += smelt_binary_dot4(x, packed.y >> i);
                partial.z += smelt_binary_dot4(x, packed.z >> i);
                partial.w += smelt_binary_dot4(x, packed.w >> i);
            }
            acc[b] += float4(partial * scale);
        }
    }

    for (uint b = 0; b < BATCH_TILE; ++b) {
        if (batch0 + b >= actual_batch) continue;
        acc[b].x = simd_sum(acc[b].x);
        acc[b].y = simd_sum(acc[b].y);
        acc[b].z = simd_sum(acc[b].z);
        acc[b].w = simd_sum(acc[b].w);
        if (lane == 0) {
            device half *batch_output = output + (batch0 + b) * rows;
            if (row0 + 0 < rows) batch_output[row0 + 0] = half(acc[b].x);
            if (row0 + 1 < rows) batch_output[row0 + 1] = half(acc[b].y);
            if (row0 + 2 < rows) batch_output[row0 + 2] = half(acc[b].z);
            if (row0 + 3 < rows) batch_output[row0 + 3] = half(acc[b].w);
        }
    }
}

inline void smelt_ternary_matvec_g128_rows4_batched_b4(
    device const uchar *codes,
    device const half *scales,
    device const half *input,
    device half *output,
    uint rows,
    uint cols,
    uint actual_batch,
    uint2 tgid,
    uint simd_gid,
    uint lane
) {
    constexpr uint GROUP_SIZE = 128;
    constexpr uint ROW_TILE = 4;
    constexpr uint BATCH_TILE = 4;
    constexpr uint COLS_PER_LANE = 32;
    constexpr uint COLS_PER_SIMDGROUP = COLS_PER_LANE * 32;
    const uint row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint batch0 = tgid.y * BATCH_TILE;
    const uint groups = cols / GROUP_SIZE;
    const uint row_bytes = cols / 4;
    float4 acc[BATCH_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint block = 0; block < cols; block += COLS_PER_SIMDGROUP) {
        const uint col = block + lane * COLS_PER_LANE;
        if (col >= cols) continue;
        const uint group = col / GROUP_SIZE;
        uint4 positive = 0u;
        uint4 negative = 0u;
        float4 scale = 0.0f;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row >= rows) continue;
            const uint2 packed = *(device const uint2 *)(
                codes + row * row_bytes + col / 4);
            const uint2 masks = smelt_ternary_sign_masks_32(packed);
            positive[tile] = masks.x;
            negative[tile] = masks.y;
            scale[tile] = float(scales[row * groups + group]);
        }

        for (uint b = 0; b < BATCH_TILE; ++b) {
            if (batch0 + b >= actual_batch) continue;
            device const half *batch_input = input + (batch0 + b) * cols;
            float4 partial = 0.0f;
            for (uint i = 0; i < COLS_PER_LANE; i += 4) {
                const half4 x = *(device const half4 *)(batch_input + col + i);
                partial.x += smelt_ternary_planar_dot4(
                    x, positive.x >> i, negative.x >> i);
                partial.y += smelt_ternary_planar_dot4(
                    x, positive.y >> i, negative.y >> i);
                partial.z += smelt_ternary_planar_dot4(
                    x, positive.z >> i, negative.z >> i);
                partial.w += smelt_ternary_planar_dot4(
                    x, positive.w >> i, negative.w >> i);
            }
            acc[b] += partial * scale;
        }
    }

    for (uint b = 0; b < BATCH_TILE; ++b) {
        if (batch0 + b >= actual_batch) continue;
        acc[b].x = simd_sum(acc[b].x);
        acc[b].y = simd_sum(acc[b].y);
        acc[b].z = simd_sum(acc[b].z);
        acc[b].w = simd_sum(acc[b].w);
        if (lane == 0) {
            device half *batch_output = output + (batch0 + b) * rows;
            if (row0 + 0 < rows) batch_output[row0 + 0] = half(acc[b].x);
            if (row0 + 1 < rows) batch_output[row0 + 1] = half(acc[b].y);
            if (row0 + 2 < rows) batch_output[row0 + 2] = half(acc[b].z);
            if (row0 + 3 < rows) batch_output[row0 + 3] = half(acc[b].w);
        }
    }
}

inline void smelt_binary_gate_up_swiglu_g128_rows4_batched_b4(
    device const uchar *gate_codes,
    device const half *gate_scales,
    device const uchar *up_codes,
    device const half *up_scales,
    device const half *input,
    device half *output,
    uint rows,
    uint cols,
    uint actual_batch,
    uint2 tgid,
    uint simd_gid,
    uint lane
) {
    constexpr uint GROUP_SIZE = 128;
    constexpr uint ROW_TILE = 4;
    constexpr uint BATCH_TILE = 4;
    constexpr uint COLS_PER_LANE = 32;
    constexpr uint COLS_PER_SIMDGROUP = COLS_PER_LANE * 32;
    const uint row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint batch0 = tgid.y * BATCH_TILE;
    const uint groups = cols / GROUP_SIZE;
    const uint row_bytes = cols / 8;
    float4 gate_acc[BATCH_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 up_acc[BATCH_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint block = 0; block < cols; block += COLS_PER_SIMDGROUP) {
        const uint col = block + lane * COLS_PER_LANE;
        if (col >= cols) continue;
        const uint group = col / GROUP_SIZE;
        uint4 gate_packed = 0u;
        uint4 up_packed = 0u;
        half4 gate_scale = 0.0h;
        half4 up_scale = 0.0h;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row >= rows) continue;
            gate_packed[tile] = *(device const uint *)(
                gate_codes + row * row_bytes + col / 8);
            up_packed[tile] = *(device const uint *)(
                up_codes + row * row_bytes + col / 8);
            gate_scale[tile] = gate_scales[row * groups + group];
            up_scale[tile] = up_scales[row * groups + group];
        }

        for (uint b = 0; b < BATCH_TILE; ++b) {
            if (batch0 + b >= actual_batch) continue;
            device const half *batch_input = input + (batch0 + b) * cols;
            half4 gate_partial = 0.0h;
            half4 up_partial = 0.0h;
            for (uint i = 0; i < COLS_PER_LANE; i += 4) {
                const half4 x = *(device const half4 *)(batch_input + col + i);
                gate_partial.x += smelt_binary_dot4(x, gate_packed.x >> i);
                gate_partial.y += smelt_binary_dot4(x, gate_packed.y >> i);
                gate_partial.z += smelt_binary_dot4(x, gate_packed.z >> i);
                gate_partial.w += smelt_binary_dot4(x, gate_packed.w >> i);
                up_partial.x += smelt_binary_dot4(x, up_packed.x >> i);
                up_partial.y += smelt_binary_dot4(x, up_packed.y >> i);
                up_partial.z += smelt_binary_dot4(x, up_packed.z >> i);
                up_partial.w += smelt_binary_dot4(x, up_packed.w >> i);
            }
            gate_acc[b] += float4(gate_partial * gate_scale);
            up_acc[b] += float4(up_partial * up_scale);
        }
    }

    for (uint b = 0; b < BATCH_TILE; ++b) {
        if (batch0 + b >= actual_batch) continue;
        gate_acc[b].x = simd_sum(gate_acc[b].x);
        gate_acc[b].y = simd_sum(gate_acc[b].y);
        gate_acc[b].z = simd_sum(gate_acc[b].z);
        gate_acc[b].w = simd_sum(gate_acc[b].w);
        up_acc[b].x = simd_sum(up_acc[b].x);
        up_acc[b].y = simd_sum(up_acc[b].y);
        up_acc[b].z = simd_sum(up_acc[b].z);
        up_acc[b].w = simd_sum(up_acc[b].w);
        if (lane == 0) {
            const half4 gate = half4(gate_acc[b]);
            const half4 up = half4(up_acc[b]);
            device half *batch_output = output + (batch0 + b) * rows;
            if (row0 + 0 < rows) batch_output[row0 + 0] =
                smelt_signed_swiglu_staged_half(gate.x, up.x);
            if (row0 + 1 < rows) batch_output[row0 + 1] =
                smelt_signed_swiglu_staged_half(gate.y, up.y);
            if (row0 + 2 < rows) batch_output[row0 + 2] =
                smelt_signed_swiglu_staged_half(gate.z, up.z);
            if (row0 + 3 < rows) batch_output[row0 + 3] =
                smelt_signed_swiglu_staged_half(gate.w, up.w);
        }
    }
}

kernel void signed_binary_matvec_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device const half *input [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &cols [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_binary_matvec_g128_rows4(
        codes, scales, input, output, rows, cols, tgid, simd_gid, lane);
}



kernel void signed_binary_matvec_g128_rows8_batched_b4(
    device const uchar *codes [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device const half *input [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &cols [[buffer(5)]],
    constant uint &actual_batch [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_binary_matvec_g128_rows4_batched_b4(
        codes, scales, input, output, rows, cols, actual_batch,
        tgid, simd_gid, lane);
}

kernel void signed_ternary_matvec_g128_rows8_batched_b4(
    device const uchar *codes [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device const half *input [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &cols [[buffer(5)]],
    constant uint &actual_batch [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_ternary_matvec_g128_rows4_batched_b4(
        codes, scales, input, output, rows, cols, actual_batch,
        tgid, simd_gid, lane);
}

kernel void signed_binary_gate_up_swiglu_g128_rows8(
    device const uchar *gate_codes [[buffer(0)]],
    device const half *gate_scales [[buffer(1)]],
    device const uchar *up_codes [[buffer(2)]],
    device const half *up_scales [[buffer(3)]],
    device const half *input [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &cols [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_binary_gate_up_swiglu_g128_rows4(
        gate_codes, gate_scales, up_codes, up_scales, input, output,
        rows, cols, tgid, simd_gid, lane);
}

kernel void signed_binary_gate_up_swiglu_g128_rows8_batched_b4(
    device const uchar *gate_codes [[buffer(0)]],
    device const half *gate_scales [[buffer(1)]],
    device const uchar *up_codes [[buffer(2)]],
    device const half *up_scales [[buffer(3)]],
    device const half *input [[buffer(4)]],
    device half *output [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &cols [[buffer(7)]],
    constant uint &actual_batch [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_binary_gate_up_swiglu_g128_rows4_batched_b4(
        gate_codes, gate_scales, up_codes, up_scales, input, output,
        rows, cols, actual_batch, tgid, simd_gid, lane);
}

kernel void signed_binary_matvec_add_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device const half *input [[buffer(2)]],
    device half *output [[buffer(3)]],
    device const half *residual [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_binary_matvec_add_g128_rows4(
        codes, scales, input, output, residual,
        rows, cols, tgid, simd_gid, lane);
}

kernel void signed_ternary_matvec_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device const half *input [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &cols [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_ternary_matvec_g128_rows4(
        codes, scales, input, output, rows, cols, tgid, simd_gid, lane);
}

template <bool TERNARY>
inline void smelt_signed_embedding_gather_g128(
    device const uchar *codes,
    device const half *scales,
    device const int *token_ids,
    device half *output,
    uint hidden_size,
    uint2 gid
) {
    const uint col = gid.x;
    if (col >= hidden_size) return;
    const uint row = uint(token_ids[gid.y]);
    const uint groups = hidden_size / 128;
    const uint row_bytes = TERNARY ? hidden_size / 4 : hidden_size / 8;
    float value;
    if (TERNARY) {
        const uint code_byte = row * row_bytes + col / 4;
        const uint packed = (codes[code_byte] >> (2u * (col & 3u))) & 3u;
        value = float(int(packed) - 1);
    } else {
        const uint code_byte = row * row_bytes + col / 8;
        const uint packed = (codes[code_byte] >> (col & 7u)) & 1u;
        value = packed == 0u ? -1.0f : 1.0f;
    }
    output[gid.y * hidden_size + col] = half(
        value * float(scales[row * groups + col / 128]));
}

kernel void signed_binary_embedding_gather_g128(
    device const uchar *codes [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device const int *token_ids [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &hidden_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    smelt_signed_embedding_gather_g128<false>(
        codes, scales, token_ids, output, hidden_size, gid);
}

kernel void signed_ternary_embedding_gather_g128(
    device const uchar *codes [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device const int *token_ids [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &hidden_size [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    smelt_signed_embedding_gather_g128<true>(
        codes, scales, token_ids, output, hidden_size, gid);
}

// Signed activation views keep one sign plane plus PLANES-1 magnitude planes.
// The represented integer is unchanged from symmetric signed quantization, but
// projection consumers only need PLANES-1 weighted popcount dots. This encoding
// is shared by every binary/ternary producer and consumer; it is a property of
// the graph-owned view, not of an individual model or projection family.
template <uint PLANES>
inline uint smelt_signed_magnitude_plane_bit(int q, uint plane) {
    if (plane == PLANES - 1) return q < 0 ? 1u : 0u;
    return (uint(abs(q)) >> plane) & 1u;
}

kernel void signed_activation_bitplanes_i4_g128(
    device const half *input [[buffer(0)]],
    device uint *planes [[buffer(1)]],
    device half *activation_scales [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint GROUP_SIZE = 128;
    constexpr uint PLANES = 4;
    constexpr uint WORDS = GROUP_SIZE / 32;
    const uint groups = cols / GROUP_SIZE;
    if (group >= groups) return;

    const uint base = group * GROUP_SIZE;
    float max_abs = 0.0f;
    for (uint word = 0; word < WORDS; ++word) {
        max_abs = max(max_abs, abs(float(input[base + word * 32 + lane])));
    }
    max_abs = simd_max(max_abs);
    // Quantize with the exact fp16 scale consumed by the matvec. Using the
    // pre-rounded fp32 value here makes codes and reconstruction disagree.
    const half stored_scale = half(max_abs > 0.0f ? max_abs / 7.0f : 1.0f);
    const float scale = float(stored_scale);
    if (lane == 0) activation_scales[group] = stored_scale;

    for (uint word = 0; word < WORDS; ++word) {
        const float x = float(input[base + word * 32 + lane]);
        const int q = int(round(clamp(x / scale, -7.0f, 7.0f)));
        for (uint plane = 0; plane < PLANES; ++plane) {
            const uint lane_bit =
                smelt_signed_magnitude_plane_bit<PLANES>(q, plane) << lane;
            const uint packed = simd_sum(lane_bit);
            if (lane == 0) {
                planes[(group * PLANES + plane) * WORDS + word] = packed;
            }
        }
    }
}

template <uint PLANES, int MAX_Q, bool MATCH_STORED_SCALE>
inline void smelt_signed_activation_bitplanes_lowbit_g128(
    device const half *input,
    device uint *planes,
    device half *activation_scales,
    uint cols,
    uint group,
    uint lane
) {
    constexpr uint GROUP_SIZE = 128;
    constexpr uint WORDS = GROUP_SIZE / 32;
    const uint groups = cols / GROUP_SIZE;
    if (group >= groups) return;
    const uint base = group * GROUP_SIZE;
    float max_abs = 0.0f;
    for (uint word = 0; word < WORDS; ++word) {
        max_abs = max(max_abs, abs(float(input[base + word * 32 + lane])));
    }
    max_abs = simd_max(max_abs);
    const float producer_scale = max_abs > 0.0f
        ? max_abs / float(MAX_Q) : 1.0f;
    const half stored_scale = half(producer_scale);
    const float scale = MATCH_STORED_SCALE
        ? float(stored_scale) : producer_scale;
    if (lane == 0) activation_scales[group] = stored_scale;
    for (uint word = 0; word < WORDS; ++word) {
        const float x = float(input[base + word * 32 + lane]);
        const int q = int(round(clamp(x / scale, -float(MAX_Q), float(MAX_Q))));
        for (uint plane = 0; plane < PLANES; ++plane) {
            const uint lane_bit =
                smelt_signed_magnitude_plane_bit<PLANES>(q, plane) << lane;
            const uint packed = simd_sum(lane_bit);
            if (lane == 0) {
                planes[(group * PLANES + plane) * WORDS + word] = packed;
            }
        }
    }
}

kernel void signed_activation_bitplanes_i3_g128(
    device const half *input [[buffer(0)]],
    device uint *planes [[buffer(1)]],
    device half *activation_scales [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_activation_bitplanes_lowbit_g128<3, 3, false>(
        input, planes, activation_scales, cols, group, lane);
}

kernel void signed_activation_bitplanes_i2_g128(
    device const half *input [[buffer(0)]],
    device uint *planes [[buffer(1)]],
    device half *activation_scales [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_activation_bitplanes_lowbit_g128<2, 1, false>(
        input, planes, activation_scales, cols, group, lane);
}

kernel void signed_activation_bitplanes_i5_g128(
    device const half *input [[buffer(0)]],
    device uint *planes [[buffer(1)]],
    device half *activation_scales [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_activation_bitplanes_lowbit_g128<5, 15, true>(
        input, planes, activation_scales, cols, group, lane);
}

// Graph producer for SwiGLU -> signed i5/g128 activation view. Values are
// materialized through the exact staged fp16 clamp boundary before scale
// reduction and quantization, so the view matches swiglu_fused followed by the
// ordinary i5 builder without writing the intermediate vector to device memory.
kernel void swiglu_signed_activation_bitplanes_i5_g128(
    device const half *gate [[buffer(0)]],
    device const half *up [[buffer(1)]],
    device uint *planes [[buffer(2)]],
    device half *activation_scales [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint GROUP_SIZE = 128;
    constexpr uint PLANES = 5;
    constexpr uint WORDS = GROUP_SIZE / 32;
    if (group >= cols / GROUP_SIZE) return;

    const uint base = group * GROUP_SIZE;
    half values[WORDS];
    float max_abs = 0.0f;
    for (uint word = 0; word < WORDS; ++word) {
        const uint col = base + word * 32 + lane;
        values[word] = smelt_signed_swiglu_staged_half(
            gate[col], up[col]);
        max_abs = max(max_abs, abs(float(values[word])));
    }
    max_abs = simd_max(max_abs);
    const half stored_scale = half(max_abs > 0.0f ? max_abs / 15.0f : 1.0f);
    const float scale = float(stored_scale);
    if (lane == 0) activation_scales[group] = stored_scale;

    for (uint word = 0; word < WORDS; ++word) {
        const int q = int(round(clamp(
            float(values[word]) / scale, -15.0f, 15.0f)));
        for (uint plane = 0; plane < PLANES; ++plane) {
            const uint lane_bit =
                smelt_signed_magnitude_plane_bit<PLANES>(q, plane) << lane;
            const uint packed = simd_sum(lane_bit);
            if (lane == 0) {
                planes[(group * PLANES + plane) * WORDS + word] = packed;
            }
        }
    }
}

kernel void signed_activation_bitplanes_i6_g128(
    device const half *input [[buffer(0)]],
    device uint *planes [[buffer(1)]],
    device half *activation_scales [[buffer(2)]],
    constant uint &cols [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_activation_bitplanes_lowbit_g128<6, 31, true>(
        input, planes, activation_scales, cols, group, lane);
}

// Prefill owns the same activation-view representation as decode, with one
// token-major view per active row. Keeping the batch dimension outside the
// canonical single-token builder makes every bit layout and scale-rounding
// rule identical to the already-validated decode contract.
template <uint PLANES, int MAX_Q, bool MATCH_STORED_SCALE>
inline void smelt_signed_activation_bitplanes_batched_g128(
    device const half *input,
    device uint *planes,
    device half *activation_scales,
    uint cols,
    uint2 tgid,
    uint lane
) {
    constexpr uint WORDS = 4;
    const uint groups = cols / 128;
    const uint batch = tgid.y;
    smelt_signed_activation_bitplanes_lowbit_g128<
        PLANES, MAX_Q, MATCH_STORED_SCALE>(
            input + batch * cols,
            planes + batch * groups * PLANES * WORDS,
            activation_scales + batch * groups,
            cols, tgid.x, lane);
}

#define SMELT_DECLARE_SIGNED_ACTIVATION_BITPLANES_BATCHED( \
    NAME, PLANES, MAX_Q, MATCH_STORED_SCALE) \
kernel void NAME( \
    device const half *input [[buffer(0)]], \
    device uint *planes [[buffer(1)]], \
    device half *activation_scales [[buffer(2)]], \
    constant uint &cols [[buffer(3)]], \
    uint2 tgid [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]] \
) { \
    smelt_signed_activation_bitplanes_batched_g128< \
        PLANES, MAX_Q, MATCH_STORED_SCALE>( \
            input, planes, activation_scales, cols, tgid, lane); \
}

SMELT_DECLARE_SIGNED_ACTIVATION_BITPLANES_BATCHED(
    signed_activation_bitplanes_i2_g128_batched, 2, 1, false)
SMELT_DECLARE_SIGNED_ACTIVATION_BITPLANES_BATCHED(
    signed_activation_bitplanes_i3_g128_batched, 3, 3, false)
SMELT_DECLARE_SIGNED_ACTIVATION_BITPLANES_BATCHED(
    signed_activation_bitplanes_i4_g128_batched, 4, 7, true)
SMELT_DECLARE_SIGNED_ACTIVATION_BITPLANES_BATCHED(
    signed_activation_bitplanes_i5_g128_batched, 5, 15, true)
SMELT_DECLARE_SIGNED_ACTIVATION_BITPLANES_BATCHED(
    signed_activation_bitplanes_i6_g128_batched, 6, 31, true)

// Fused producer for the graph pattern
//   sigmoidMul([groups, 128]) -> signed i6/g128 activation view.
//
// Each lane materializes the same four fp16 values as the staged sigmoid_mul
// before applying the exact stored-scale i6 builder contract.
kernel void sigmoid_mul_signed_activation_bitplanes_i6_g128(
    device const half *input_a [[buffer(0)]],
    device const half *input_b [[buffer(1)]],
    device uint *planes [[buffer(2)]],
    device half *activation_scales [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint GROUP_SIZE = 128;
    constexpr uint PLANES = 6;
    constexpr uint WORDS = GROUP_SIZE / 32;
    const uint groups = cols / GROUP_SIZE;
    if (group >= groups) return;

    const uint base = group * GROUP_SIZE;
    half values[WORDS];
    float max_abs = 0.0f;
    for (uint word = 0; word < WORDS; ++word) {
        const uint col = base + word * 32 + lane;
        const half b = input_b[col];
        const half tail = 1.0h / (1.0h + exp(abs(b)));
        const half sig_b = b < 0.0h ? tail : 1.0h - tail;
        values[word] = input_a[col] * sig_b;
        max_abs = max(max_abs, abs(float(values[word])));
    }
    max_abs = simd_max(max_abs);
    const half stored_scale = half(max_abs > 0.0f ? max_abs / 31.0f : 1.0f);
    const float scale = float(stored_scale);
    if (lane == 0) activation_scales[group] = stored_scale;

    for (uint word = 0; word < WORDS; ++word) {
        const float x = float(values[word]);
        const int q = int(round(clamp(x / scale, -31.0f, 31.0f)));
        for (uint plane = 0; plane < PLANES; ++plane) {
            const uint lane_bit =
                smelt_signed_magnitude_plane_bit<PLANES>(q, plane) << lane;
            const uint packed = simd_sum(lane_bit);
            if (lane == 0) {
                planes[(group * PLANES + plane) * WORDS + word] = packed;
            }
        }
    }
}

// Fused producer for the graph pattern
//   RMSNormGated([groups, 128]) -> signed i6/g128 activation view.
//
// The first half deliberately mirrors the generic rms_norm_gated reduction
// and fp16 materialization for its runtime D=128 contract. The first SIMD consumes
// that threadgroup-local fp16 value using the exact stored-scale i6 contract.
// Trace builds run this exact producer and then replay the original norm only
// to materialize the observed edge, so the kernel under audit never changes.
inline void smelt_rms_norm_gated_d128_signed_activation_bitplanes_i6_g128(
    device const half *input,
    device const half *gate,
    device const half *weight,
    device uint *planes,
    device half *activation_scales,
    threadgroup float *cached_input,
    threadgroup float *cached_gate,
    threadgroup float *partial,
    threadgroup float *shared_rsqrt,
    threadgroup half *normalized,
    uint head_dim,
    float eps,
    uint tgs,
    uint group,
    uint tid,
    uint lane,
    uint simd_group
) {
    constexpr uint GROUP_SIZE = 128;
    constexpr uint PLANES = 6;
    constexpr uint WORDS = GROUP_SIZE / 32;
    const uint base = group * head_dim;

    if (tgs == 32 && head_dim == 128) {
        // Match rms_norm_gated_d128 exactly: one SIMD, four adjacent values
        // per lane, precise rsqrt, an fp16 weight*normalized boundary, and
        // the stable sigmoid spelling used by the specialized producer.
        constexpr uint N_READS = 4;
        const uint local = tid * N_READS;
        float sum_sq = 0.0f;
        for (uint i = 0; i < N_READS; ++i) {
            const float x = float(input[base + local + i]);
            sum_sq += x * x;
        }
        sum_sq = simd_sum(sum_sq);
        if (lane == 0) {
            shared_rsqrt[0] = metal::precise::rsqrt(
                sum_sq / float(head_dim) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < N_READS; ++i) {
            const uint index = local + i;
            const half norm_value = weight[index]
                * half(float(input[base + index]) * shared_rsqrt[0]);
            const float g = float(gate[base + index]);
            const float tail = 1.0f / (1.0f + exp(abs(g)));
            const float sigmoid = g < 0.0f ? tail : 1.0f - tail;
            normalized[index] = half(float(norm_value) * (g * sigmoid));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
        float sum_sq = 0.0f;
        for (uint i = tid; i < head_dim; i += tgs) {
            const float x = float(input[base + i]);
            const float g = float(gate[base + i]);
            cached_input[i] = x;
            cached_gate[i] = g;
            sum_sq += x * x;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum_sq = simd_sum(sum_sq);
        if (lane == 0) partial[simd_group] = sum_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float total = 0.0f;
            for (uint s = 0; s < tgs / 32; ++s) total += partial[s];
            const float mean = total / float(head_dim);
            shared_rsqrt[0] = rsqrt(mean + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float rs = shared_rsqrt[0];
        for (uint i = tid; i < head_dim; i += tgs) {
            const float x = cached_input[i];
            const float g = cached_gate[i];
            const float w = float(weight[i]);
            const float silu_g = g / (1.0f + exp(-g));
            normalized[i] = half(w * (x * rs) * silu_g);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_group != 0) return;

    float max_abs = 0.0f;
    for (uint word = 0; word < WORDS; ++word) {
        max_abs = max(max_abs, abs(float(normalized[word * 32 + lane])));
    }
    max_abs = simd_max(max_abs);
    const half stored_scale = half(max_abs > 0.0f ? max_abs / 31.0f : 1.0f);
    const float scale = float(stored_scale);
    if (lane == 0) activation_scales[group] = stored_scale;

    for (uint word = 0; word < WORDS; ++word) {
        const float x = float(normalized[word * 32 + lane]);
        const int q = int(round(clamp(x / scale, -31.0f, 31.0f)));
        for (uint plane = 0; plane < PLANES; ++plane) {
            const uint lane_bit =
                smelt_signed_magnitude_plane_bit<PLANES>(q, plane) << lane;
            const uint packed = simd_sum(lane_bit);
            if (lane == 0) {
                planes[(group * PLANES + plane) * WORDS + word] = packed;
            }
        }
    }
}

kernel void rms_norm_gated_d128_signed_activation_bitplanes_i6_g128(
    device const half *input [[buffer(0)]],
    device const half *gate [[buffer(1)]],
    device const half *weight [[buffer(2)]],
    device uint *planes [[buffer(3)]],
    device half *activation_scales [[buffer(4)]],
    constant uint &head_dim [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float cached_input[256];
    threadgroup float cached_gate[256];
    threadgroup float partial[32];
    threadgroup float shared_rsqrt = 0.0f;
    threadgroup half normalized[128];
    smelt_rms_norm_gated_d128_signed_activation_bitplanes_i6_g128(
        input, gate, weight, planes, activation_scales,
        cached_input, cached_gate, partial, &shared_rsqrt, normalized,
        head_dim, eps, tgs,
        group, tid, lane, simd_group);
}

template <uint PLANES, int MAX_Q, bool MATCH_STORED_SCALE>
inline void smelt_norm_scale_signed_activation_bitplanes_g128(
    device const float *norm_scale,
    device const half *input,
    device const half *norm_weight,
    device uint *planes,
    device half *activation_scales,
    uint cols,
    uint group,
    uint lane
) {
    constexpr uint GROUP_SIZE = 128;
    constexpr uint WORDS = GROUP_SIZE / 32;
    const uint groups = cols / GROUP_SIZE;
    if (group >= groups) return;
    const uint base = group * GROUP_SIZE;
    const float rs = norm_scale[0];
    float values[WORDS];
    float max_abs = 0.0f;
    for (uint word = 0; word < WORDS; ++word) {
        const uint col = base + word * 32 + lane;
        // Match rms_norm_1pw's two fp16 operands and multiplication boundary
        // before the ordinary activation-view producer reads the result.
        const half normalized = half(float(input[col]) * rs);
        const half direct_weight = half(1.0f + float(norm_weight[col]));
        const float value = float(normalized * direct_weight);
        values[word] = value;
        max_abs = max(max_abs, abs(value));
    }
    max_abs = simd_max(max_abs);
    const float producer_scale = max_abs > 0.0f
        ? max_abs / float(MAX_Q) : 1.0f;
    const half stored_scale = half(producer_scale);
    const float quant_scale = MATCH_STORED_SCALE
        ? float(stored_scale) : producer_scale;
    if (lane == 0) activation_scales[group] = stored_scale;
    for (uint word = 0; word < WORDS; ++word) {
        const int q = int(round(clamp(
            values[word] / quant_scale, -float(MAX_Q), float(MAX_Q))));
        for (uint plane = 0; plane < PLANES; ++plane) {
            const uint lane_bit =
                smelt_signed_magnitude_plane_bit<PLANES>(q, plane) << lane;
            const uint packed = simd_sum(lane_bit);
            if (lane == 0) {
                planes[(group * PLANES + plane) * WORDS + word] = packed;
            }
        }
    }
}

kernel void norm_scale_signed_activation_bitplanes_i4_g128(
    device const float *norm_scale [[buffer(0)]],
    device const half *input [[buffer(1)]],
    device const half *norm_weight [[buffer(2)]],
    device uint *planes [[buffer(3)]],
    device half *activation_scales [[buffer(4)]],
    constant uint &cols [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_norm_scale_signed_activation_bitplanes_g128<4, 7, true>(
        norm_scale, input, norm_weight, planes, activation_scales,
        cols, group, lane);
}

kernel void norm_scale_signed_activation_bitplanes_i3_g128(
    device const float *norm_scale [[buffer(0)]],
    device const half *input [[buffer(1)]],
    device const half *norm_weight [[buffer(2)]],
    device uint *planes [[buffer(3)]],
    device half *activation_scales [[buffer(4)]],
    constant uint &cols [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_norm_scale_signed_activation_bitplanes_g128<3, 3, false>(
        norm_scale, input, norm_weight, planes, activation_scales,
        cols, group, lane);
}

kernel void norm_scale_signed_activation_bitplanes_i2_g128(
    device const float *norm_scale [[buffer(0)]],
    device const half *input [[buffer(1)]],
    device const half *norm_weight [[buffer(2)]],
    device uint *planes [[buffer(3)]],
    device half *activation_scales [[buffer(4)]],
    constant uint &cols [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_norm_scale_signed_activation_bitplanes_g128<2, 1, false>(
        norm_scale, input, norm_weight, planes, activation_scales,
        cols, group, lane);
}

kernel void norm_scale_signed_activation_bitplanes_i5_g128(
    device const float *norm_scale [[buffer(0)]],
    device const half *input [[buffer(1)]],
    device const half *norm_weight [[buffer(2)]],
    device uint *planes [[buffer(3)]],
    device half *activation_scales [[buffer(4)]],
    constant uint &cols [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_norm_scale_signed_activation_bitplanes_g128<5, 15, true>(
        norm_scale, input, norm_weight, planes, activation_scales,
        cols, group, lane);
}

kernel void norm_scale_signed_activation_bitplanes_i6_g128(
    device const float *norm_scale [[buffer(0)]],
    device const half *input [[buffer(1)]],
    device const half *norm_weight [[buffer(2)]],
    device uint *planes [[buffer(3)]],
    device half *activation_scales [[buffer(4)]],
    constant uint &cols [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_norm_scale_signed_activation_bitplanes_g128<6, 31, true>(
        norm_scale, input, norm_weight, planes, activation_scales,
        cols, group, lane);
}

inline uint smelt_signed_magnitude_active_count(uint4 activation_magnitude) {
    const uint4 active = popcount(activation_magnitude);
    return active.x + active.y + active.z + active.w;
}

inline int smelt_binary_signed_magnitude_dot_known_active(
    uint4 positive_weight,
    uint4 activation_sign,
    uint4 activation_magnitude,
    uint active_count
) {
    const uint4 agreement = popcount(
        (positive_weight ^ activation_sign) & activation_magnitude);
    const uint agreement_count =
        agreement.x + agreement.y + agreement.z + agreement.w;
    return int(agreement_count * 2u) - int(active_count);
}

inline int smelt_binary_signed_magnitude_dot(
    uint4 positive_weight,
    uint4 activation_sign,
    uint4 activation_magnitude
) {
    return smelt_binary_signed_magnitude_dot_known_active(
        positive_weight, activation_sign, activation_magnitude,
        smelt_signed_magnitude_active_count(activation_magnitude));
}

inline int smelt_ternary_signed_magnitude_dot(
    uint4 positive_weight,
    uint4 negative_weight,
    uint4 activation_sign,
    uint4 activation_magnitude
) {
    const uint4 agreement_mask =
        (positive_weight & ~activation_sign)
        | (negative_weight & activation_sign);
    const uint4 nonzero_weight = positive_weight | negative_weight;
    const uint4 agreement = popcount(agreement_mask & activation_magnitude);
    const uint4 active = popcount(nonzero_weight & activation_magnitude);
    const uint agreement_count =
        agreement.x + agreement.y + agreement.z + agreement.w;
    const uint active_count = active.x + active.y + active.z + active.w;
    return int(agreement_count * 2u) - int(active_count);
}

// True tiled signed bit-GEMM for prompt rows. Each SIMD loads a weight tile
// once, then consumes four independently quantized activation rows. This is
// deliberately not four matvecs hidden in one dispatch: the expensive packed
// weights and scales are shared across the B4 tile.
template <uint PLANES, uint ROW_TILE, uint BATCH_TILE>
inline void smelt_signed_binary_bitplane_matvec_g128_rows8_batched_b4(
    device const uchar *codes,
    device const half *weight_scales,
    device const uint *activation_planes,
    device const half *activation_scales,
    device half *output,
    uint rows,
    uint cols,
    uint actual_batch,
    uint2 tgid,
    uint simd_gid,
    uint lane
) {
    constexpr uint WORDS = 4;
    constexpr uint ROWS_PER_THREADGROUP = 8;
    const uint row0 = tgid.x * ROWS_PER_THREADGROUP + simd_gid * ROW_TILE;
    const uint batch0 = tgid.y * BATCH_TILE;
    const uint groups = cols / 128;
    const uint row_bytes = cols / 8;
    const uint plane_words_per_batch = groups * PLANES * WORDS;
    float4 acc[BATCH_TILE];
    for (uint b = 0; b < BATCH_TILE; ++b) acc[b] = 0.0f;

    for (uint group_block = 0; group_block < groups; group_block += 32) {
        const uint group = group_block + lane;
        if (group >= groups) continue;

        uint4 weight[ROW_TILE];
        half4 weight_scale = 0.0h;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            weight[tile] = 0u;
            if (row >= rows) continue;
            weight[tile] = *(device const uint4 *)(
                codes + row * row_bytes + group * 16);
            weight_scale[tile] = weight_scales[row * groups + group];
        }

        for (uint b = 0; b < BATCH_TILE; ++b) {
            const uint batch = batch0 + b;
            if (batch >= actual_batch) continue;
            device const uint *batch_planes =
                activation_planes + batch * plane_words_per_batch;
            const uint4 sign = *(device const uint4 *)(
                batch_planes + (group * PLANES + PLANES - 1) * WORDS);
            int4 quantized_dot = 0;
            for (uint bit = 0; bit < PLANES - 1; ++bit) {
                const uint4 plane = *(device const uint4 *)(
                    batch_planes + (group * PLANES + bit) * WORDS);
                const uint active_count =
                    smelt_signed_magnitude_active_count(plane);
                const int coefficient = int(1u << bit);
                for (uint tile = 0; tile < ROW_TILE; ++tile) {
                    quantized_dot[tile] += coefficient
                        * smelt_binary_signed_magnitude_dot_known_active(
                            weight[tile], sign, plane, active_count);
                }
            }
            acc[b] += float4(quantized_dot)
                * float(activation_scales[batch * groups + group])
                * float4(weight_scale);
        }
    }

    for (uint b = 0; b < BATCH_TILE; ++b) {
        const uint batch = batch0 + b;
        if (batch >= actual_batch) continue;
        acc[b].x = simd_sum(acc[b].x);
        acc[b].y = simd_sum(acc[b].y);
        acc[b].z = simd_sum(acc[b].z);
        acc[b].w = simd_sum(acc[b].w);
        if (lane == 0) {
            device half *batch_output = output + batch * rows;
            if (row0 + 0 < rows) batch_output[row0 + 0] = half(acc[b].x);
            if (ROW_TILE > 1 && row0 + 1 < rows) {
                batch_output[row0 + 1] = half(acc[b].y);
            }
            if (ROW_TILE > 2 && row0 + 2 < rows) {
                batch_output[row0 + 2] = half(acc[b].z);
            }
            if (ROW_TILE > 3 && row0 + 3 < rows) {
                batch_output[row0 + 3] = half(acc[b].w);
            }
        }
    }
}

#define SMELT_DECLARE_SIGNED_BINARY_BITPLANE_MATVEC_BATCHED_B4(NAME, PLANES) \
kernel void NAME( \
    device const uchar *codes [[buffer(0)]], \
    device const half *weight_scales [[buffer(1)]], \
    device const uint *activation_planes [[buffer(2)]], \
    device const half *activation_scales [[buffer(3)]], \
    device half *output [[buffer(4)]], \
    constant uint &rows [[buffer(5)]], \
    constant uint &cols [[buffer(6)]], \
    constant uint &actual_batch [[buffer(7)]], \
    uint2 tgid [[threadgroup_position_in_grid]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]] \
) { \
    smelt_signed_binary_bitplane_matvec_g128_rows8_batched_b4< \
        PLANES, 4, 4>( \
        codes, weight_scales, activation_planes, activation_scales, output, \
        rows, cols, actual_batch, tgid, simd_gid, lane); \
}

SMELT_DECLARE_SIGNED_BINARY_BITPLANE_MATVEC_BATCHED_B4(
    signed_binary_bitplane_i2_matvec_g128_rows8_batched_b4, 2)
SMELT_DECLARE_SIGNED_BINARY_BITPLANE_MATVEC_BATCHED_B4(
    signed_binary_bitplane_i3_matvec_g128_rows8_batched_b4, 3)
SMELT_DECLARE_SIGNED_BINARY_BITPLANE_MATVEC_BATCHED_B4(
    signed_binary_bitplane_i4_matvec_g128_rows8_batched_b4, 4)
SMELT_DECLARE_SIGNED_BINARY_BITPLANE_MATVEC_BATCHED_B4(
    signed_binary_bitplane_i5_matvec_g128_rows8_batched_b4, 5)
SMELT_DECLARE_SIGNED_BINARY_BITPLANE_MATVEC_BATCHED_B4(
    signed_binary_bitplane_i6_matvec_g128_rows8_batched_b4, 6)

#define SMELT_DECLARE_SIGNED_BINARY_BITPLANE_MATVEC_BATCHED( \
    NAME, PLANES, ROW_TILE, BATCH_TILE) \
kernel void NAME( \
    device const uchar *codes [[buffer(0)]], \
    device const half *weight_scales [[buffer(1)]], \
    device const uint *activation_planes [[buffer(2)]], \
    device const half *activation_scales [[buffer(3)]], \
    device half *output [[buffer(4)]], \
    constant uint &rows [[buffer(5)]], \
    constant uint &cols [[buffer(6)]], \
    constant uint &actual_batch [[buffer(7)]], \
    uint2 tgid [[threadgroup_position_in_grid]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]] \
) { \
    smelt_signed_binary_bitplane_matvec_g128_rows8_batched_b4< \
        PLANES, ROW_TILE, BATCH_TILE>( \
            codes, weight_scales, activation_planes, activation_scales, \
            output, rows, cols, actual_batch, tgid, simd_gid, lane); \
}

#define SMELT_DECLARE_SIGNED_BINARY_BITPLANE_BATCH_VARIANTS(PLANES) \
SMELT_DECLARE_SIGNED_BINARY_BITPLANE_MATVEC_BATCHED( \
    signed_binary_bitplane_i##PLANES##_matvec_g128_rows8_batched_b16, \
    PLANES, 1, 16)

SMELT_DECLARE_SIGNED_BINARY_BITPLANE_BATCH_VARIANTS(2)
SMELT_DECLARE_SIGNED_BINARY_BITPLANE_BATCH_VARIANTS(3)
SMELT_DECLARE_SIGNED_BINARY_BITPLANE_BATCH_VARIANTS(4)
SMELT_DECLARE_SIGNED_BINARY_BITPLANE_BATCH_VARIANTS(5)
SMELT_DECLARE_SIGNED_BINARY_BITPLANE_BATCH_VARIANTS(6)

// One frozen B8 table also serves latency-sensitive tiny prompts. B<=4 keeps
// the proven B4 row geometry and idles the second half of the threadgroup;
// larger prompts use the measured B8 weight-reuse tile. This is a generic
// shape decision inside the kernel, not a model/runtime dispatch fork.
template <uint PLANES>
inline void smelt_signed_binary_bitplane_matvec_g128_rows8_adaptive_b8(
    device const uchar *codes,
    device const half *weight_scales,
    device const uint *activation_planes,
    device const half *activation_scales,
    device half *output,
    uint rows,
    uint cols,
    uint actual_batch,
    uint2 tgid,
    uint simd_gid,
    uint lane
) {
    if (actual_batch <= 4) {
        if (simd_gid < 2) {
            smelt_signed_binary_bitplane_matvec_g128_rows8_batched_b4<
                PLANES, 4, 4>(
                    codes, weight_scales, activation_planes,
                    activation_scales, output, rows, cols, actual_batch,
                    tgid, simd_gid, lane);
        }
        return;
    }
    smelt_signed_binary_bitplane_matvec_g128_rows8_batched_b4<PLANES, 2, 8>(
        codes, weight_scales, activation_planes, activation_scales, output,
        rows, cols, actual_batch, tgid, simd_gid, lane);
}

#define SMELT_DECLARE_SIGNED_BINARY_BITPLANE_ADAPTIVE_B8(NAME, PLANES) \
kernel void NAME( \
    device const uchar *codes [[buffer(0)]], \
    device const half *weight_scales [[buffer(1)]], \
    device const uint *activation_planes [[buffer(2)]], \
    device const half *activation_scales [[buffer(3)]], \
    device half *output [[buffer(4)]], \
    constant uint &rows [[buffer(5)]], \
    constant uint &cols [[buffer(6)]], \
    constant uint &actual_batch [[buffer(7)]], \
    uint2 tgid [[threadgroup_position_in_grid]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]] \
) { \
    smelt_signed_binary_bitplane_matvec_g128_rows8_adaptive_b8<PLANES>( \
        codes, weight_scales, activation_planes, activation_scales, output, \
        rows, cols, actual_batch, tgid, simd_gid, lane); \
}

SMELT_DECLARE_SIGNED_BINARY_BITPLANE_ADAPTIVE_B8(
    signed_binary_bitplane_i2_matvec_g128_rows8_batched_b8, 2)
SMELT_DECLARE_SIGNED_BINARY_BITPLANE_ADAPTIVE_B8(
    signed_binary_bitplane_i3_matvec_g128_rows8_batched_b8, 3)
SMELT_DECLARE_SIGNED_BINARY_BITPLANE_ADAPTIVE_B8(
    signed_binary_bitplane_i4_matvec_g128_rows8_batched_b8, 4)
SMELT_DECLARE_SIGNED_BINARY_BITPLANE_ADAPTIVE_B8(
    signed_binary_bitplane_i5_matvec_g128_rows8_batched_b8, 5)
SMELT_DECLARE_SIGNED_BINARY_BITPLANE_ADAPTIVE_B8(
    signed_binary_bitplane_i6_matvec_g128_rows8_batched_b8, 6)

kernel void signed_binary_bitplane_i4_matvec_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *weight_scales [[buffer(1)]],
    device const uint *activation_planes [[buffer(2)]],
    device const half *activation_scales [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint ROW_TILE = 4;
    constexpr uint PLANES = 4;
    constexpr uint WORDS = 4;
    const uint row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint groups = cols / 128;
    const uint row_bytes = cols / 8;
    float4 acc = 0.0f;

    for (uint group_block = 0; group_block < groups; group_block += 32) {
        const uint group = group_block + lane;
        if (group >= groups) continue;

        uint4 weight[ROW_TILE];
        half4 weight_scale = 0.0h;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            weight[tile] = 0u;
            if (row >= rows) continue;
            weight[tile] = *(device const uint4 *)(
                codes + row * row_bytes + group * 16);
            weight_scale[tile] = weight_scales[row * groups + group];
        }

        const uint4 sign = *(device const uint4 *)(
            activation_planes + (group * PLANES + PLANES - 1) * WORDS);
        int4 quantized_dot = 0;
        for (uint bit = 0; bit < PLANES - 1; ++bit) {
            const uint4 plane = *(device const uint4 *)(
                activation_planes + (group * PLANES + bit) * WORDS);
            const uint active_count =
                smelt_signed_magnitude_active_count(plane);
            const int coefficient = int(1u << bit);
            quantized_dot.x += coefficient
                * smelt_binary_signed_magnitude_dot_known_active(
                    weight[0], sign, plane, active_count);
            quantized_dot.y += coefficient
                * smelt_binary_signed_magnitude_dot_known_active(
                    weight[1], sign, plane, active_count);
            quantized_dot.z += coefficient
                * smelt_binary_signed_magnitude_dot_known_active(
                    weight[2], sign, plane, active_count);
            quantized_dot.w += coefficient
                * smelt_binary_signed_magnitude_dot_known_active(
                    weight[3], sign, plane, active_count);
        }
        acc += float4(quantized_dot)
            * float(activation_scales[group])
            * float4(weight_scale);
    }

    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    acc.z = simd_sum(acc.z);
    acc.w = simd_sum(acc.w);
    if (lane == 0) {
        if (row0 + 0 < rows) output[row0 + 0] = half(acc.x);
        if (row0 + 1 < rows) output[row0 + 1] = half(acc.y);
        if (row0 + 2 < rows) output[row0 + 2] = half(acc.z);
        if (row0 + 3 < rows) output[row0 + 3] = half(acc.w);
    }
}

template <uint PLANES, bool ADD_RESIDUAL>
inline void smelt_signed_binary_bitplane_lowbit_matvec_g128_rows8(
    device const uchar *codes,
    device const half *weight_scales,
    device const uint *activation_planes,
    device const half *activation_scales,
    device half *output,
    device const half *residual,
    uint rows,
    uint cols,
    uint2 tgid,
    uint simd_gid,
    uint lane
) {
    constexpr uint ROW_TILE = 4;
    constexpr uint WORDS = 4;
    const uint row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint groups = cols / 128;
    const uint row_bytes = cols / 8;
    float4 acc = 0.0f;
    for (uint group_block = 0; group_block < groups; group_block += 32) {
        const uint group = group_block + lane;
        if (group >= groups) continue;
        uint4 weight[ROW_TILE];
        half4 weight_scale = 0.0h;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            weight[tile] = 0u;
            if (row >= rows) continue;
            weight[tile] = *(device const uint4 *)(
                codes + row * row_bytes + group * 16);
            weight_scale[tile] = weight_scales[row * groups + group];
        }
        const uint4 sign = *(device const uint4 *)(
            activation_planes + (group * PLANES + PLANES - 1) * WORDS);
        int4 quantized_dot = 0;
        for (uint bit = 0; bit < PLANES - 1; ++bit) {
            const uint4 plane = *(device const uint4 *)(
                activation_planes + (group * PLANES + bit) * WORDS);
            const uint active_count =
                smelt_signed_magnitude_active_count(plane);
            const int coefficient = int(1u << bit);
            quantized_dot.x += coefficient
                * smelt_binary_signed_magnitude_dot_known_active(
                    weight[0], sign, plane, active_count);
            quantized_dot.y += coefficient
                * smelt_binary_signed_magnitude_dot_known_active(
                    weight[1], sign, plane, active_count);
            quantized_dot.z += coefficient
                * smelt_binary_signed_magnitude_dot_known_active(
                    weight[2], sign, plane, active_count);
            quantized_dot.w += coefficient
                * smelt_binary_signed_magnitude_dot_known_active(
                    weight[3], sign, plane, active_count);
        }
        acc += float4(quantized_dot)
            * float(activation_scales[group])
            * float4(weight_scale);
    }
    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    acc.z = simd_sum(acc.z);
    acc.w = simd_sum(acc.w);
    if (lane == 0) {
        if (row0 + 0 < rows) output[row0 + 0] = half(
            float(half(acc.x)) + (ADD_RESIDUAL ? float(residual[row0 + 0]) : 0.0f));
        if (row0 + 1 < rows) output[row0 + 1] = half(
            float(half(acc.y)) + (ADD_RESIDUAL ? float(residual[row0 + 1]) : 0.0f));
        if (row0 + 2 < rows) output[row0 + 2] = half(
            float(half(acc.z)) + (ADD_RESIDUAL ? float(residual[row0 + 2]) : 0.0f));
        if (row0 + 3 < rows) output[row0 + 3] = half(
            float(half(acc.w)) + (ADD_RESIDUAL ? float(residual[row0 + 3]) : 0.0f));
    }
}

kernel void signed_binary_bitplane_i3_matvec_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *weight_scales [[buffer(1)]],
    device const uint *activation_planes [[buffer(2)]],
    device const half *activation_scales [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_binary_bitplane_lowbit_matvec_g128_rows8<3, false>(
        codes, weight_scales, activation_planes, activation_scales,
        output, output, rows, cols, tgid, simd_gid, lane);
}

// Consume the CAM-owned low-bit activation view once for paired binary gate/up
// projections, then apply the exact staged fp16 SwiGLU boundary.
template <uint PLANES>
inline void smelt_signed_binary_bitplane_gate_up_swiglu_g128_rows8(
    device const uchar *gate_codes,
    device const half *gate_weight_scales,
    device const uchar *up_codes,
    device const half *up_weight_scales,
    device const uint *activation_planes,
    device const half *activation_scales,
    device half *output,
    uint rows,
    uint cols,
    uint2 tgid,
    uint simd_gid,
    uint lane
) {
    constexpr uint ROW_TILE = 4;
    constexpr uint WORDS = 4;
    const uint row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint groups = cols / 128;
    const uint row_bytes = cols / 8;
    float4 gate_acc = 0.0f;
    float4 up_acc = 0.0f;

    for (uint group_block = 0; group_block < groups; group_block += 32) {
        const uint group = group_block + lane;
        if (group >= groups) continue;
        uint4 gate_weight[ROW_TILE];
        uint4 up_weight[ROW_TILE];
        half4 gate_weight_scale = 0.0h;
        half4 up_weight_scale = 0.0h;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            gate_weight[tile] = 0u;
            up_weight[tile] = 0u;
            if (row >= rows) continue;
            gate_weight[tile] = *(device const uint4 *)(
                gate_codes + row * row_bytes + group * 16);
            up_weight[tile] = *(device const uint4 *)(
                up_codes + row * row_bytes + group * 16);
            gate_weight_scale[tile] = gate_weight_scales[row * groups + group];
            up_weight_scale[tile] = up_weight_scales[row * groups + group];
        }

        const uint4 sign = *(device const uint4 *)(
            activation_planes + (group * PLANES + PLANES - 1) * WORDS);
        int4 gate_quantized_dot = 0;
        int4 up_quantized_dot = 0;
        for (uint bit = 0; bit < PLANES - 1; ++bit) {
            const uint4 plane = *(device const uint4 *)(
                activation_planes + (group * PLANES + bit) * WORDS);
            const uint active_count =
                smelt_signed_magnitude_active_count(plane);
            const int coefficient = int(1u << bit);
            for (uint tile = 0; tile < ROW_TILE; ++tile) {
                gate_quantized_dot[tile] += coefficient
                    * smelt_binary_signed_magnitude_dot_known_active(
                        gate_weight[tile], sign, plane, active_count);
                up_quantized_dot[tile] += coefficient
                    * smelt_binary_signed_magnitude_dot_known_active(
                        up_weight[tile], sign, plane, active_count);
            }
        }
        const float activation_scale = float(activation_scales[group]);
        gate_acc += float4(gate_quantized_dot)
            * activation_scale * float4(gate_weight_scale);
        up_acc += float4(up_quantized_dot)
            * activation_scale * float4(up_weight_scale);
    }

    gate_acc.x = simd_sum(gate_acc.x);
    gate_acc.y = simd_sum(gate_acc.y);
    gate_acc.z = simd_sum(gate_acc.z);
    gate_acc.w = simd_sum(gate_acc.w);
    up_acc.x = simd_sum(up_acc.x);
    up_acc.y = simd_sum(up_acc.y);
    up_acc.z = simd_sum(up_acc.z);
    up_acc.w = simd_sum(up_acc.w);
    if (lane == 0) {
        const half4 gate = half4(gate_acc);
        const half4 up = half4(up_acc);
        if (row0 + 0 < rows) output[row0 + 0] =
            smelt_signed_swiglu_staged_half(gate.x, up.x);
        if (row0 + 1 < rows) output[row0 + 1] =
            smelt_signed_swiglu_staged_half(gate.y, up.y);
        if (row0 + 2 < rows) output[row0 + 2] =
            smelt_signed_swiglu_staged_half(gate.z, up.z);
        if (row0 + 3 < rows) output[row0 + 3] =
            smelt_signed_swiglu_staged_half(gate.w, up.w);
    }
}

kernel void signed_binary_bitplane_i3_gate_up_swiglu_g128_rows8(
    device const uchar *gate_codes [[buffer(0)]],
    device const half *gate_weight_scales [[buffer(1)]],
    device const uchar *up_codes [[buffer(2)]],
    device const half *up_weight_scales [[buffer(3)]],
    device const uint *activation_planes [[buffer(4)]],
    device const half *activation_scales [[buffer(5)]],
    device half *output [[buffer(6)]],
    constant uint &rows [[buffer(7)]],
    constant uint &cols [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_binary_bitplane_gate_up_swiglu_g128_rows8<3>(
        gate_codes, gate_weight_scales, up_codes, up_weight_scales,
        activation_planes, activation_scales, output,
        rows, cols, tgid, simd_gid, lane);
}

kernel void signed_binary_bitplane_i4_gate_up_swiglu_g128_rows8(
    device const uchar *gate_codes [[buffer(0)]],
    device const half *gate_weight_scales [[buffer(1)]],
    device const uchar *up_codes [[buffer(2)]],
    device const half *up_weight_scales [[buffer(3)]],
    device const uint *activation_planes [[buffer(4)]],
    device const half *activation_scales [[buffer(5)]],
    device half *output [[buffer(6)]],
    constant uint &rows [[buffer(7)]],
    constant uint &cols [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_binary_bitplane_gate_up_swiglu_g128_rows8<4>(
        gate_codes, gate_weight_scales, up_codes, up_weight_scales,
        activation_planes, activation_scales, output,
        rows, cols, tgid, simd_gid, lane);
}

kernel void signed_binary_bitplane_i5_gate_up_swiglu_g128_rows8(
    device const uchar *gate_codes [[buffer(0)]],
    device const half *gate_weight_scales [[buffer(1)]],
    device const uchar *up_codes [[buffer(2)]],
    device const half *up_weight_scales [[buffer(3)]],
    device const uint *activation_planes [[buffer(4)]],
    device const half *activation_scales [[buffer(5)]],
    device half *output [[buffer(6)]],
    constant uint &rows [[buffer(7)]],
    constant uint &cols [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_binary_bitplane_gate_up_swiglu_g128_rows8<5>(
        gate_codes, gate_weight_scales, up_codes, up_weight_scales,
        activation_planes, activation_scales, output,
        rows, cols, tgid, simd_gid, lane);
}

kernel void signed_binary_bitplane_i6_gate_up_swiglu_g128_rows8(
    device const uchar *gate_codes [[buffer(0)]],
    device const half *gate_weight_scales [[buffer(1)]],
    device const uchar *up_codes [[buffer(2)]],
    device const half *up_weight_scales [[buffer(3)]],
    device const uint *activation_planes [[buffer(4)]],
    device const half *activation_scales [[buffer(5)]],
    device half *output [[buffer(6)]],
    constant uint &rows [[buffer(7)]],
    constant uint &cols [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_binary_bitplane_gate_up_swiglu_g128_rows8<6>(
        gate_codes, gate_weight_scales, up_codes, up_weight_scales,
        activation_planes, activation_scales, output,
        rows, cols, tgid, simd_gid, lane);
}

// Representation-specific consumer for the same CAM gate/up bank contract as
// the binary kernel above. Keep ternary's proven one-row-per-SIMD accumulation
// order while sharing activation-plane loads across the two projections.
template <uint PLANES>
inline void smelt_signed_ternary_bitplane_gate_up_swiglu_g128_rows8(
    device const uchar *gate_codes,
    device const half *gate_weight_scales,
    device const uchar *up_codes,
    device const half *up_weight_scales,
    device const uint *activation_planes,
    device const half *activation_scales,
    device half *output,
    uint rows,
    uint cols,
    uint2 tgid,
    uint simd_gid,
    uint lane
) {
    constexpr uint WORDS = 4;
    constexpr uint ROWS_PER_THREADGROUP = 4;
    const uint row = tgid.x * ROWS_PER_THREADGROUP + simd_gid;
    if (row >= rows) return;
    const uint groups = cols / 128;
    const uint row_bytes = cols / 4;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (uint group_block = 0; group_block < groups; group_block += 32) {
        const uint group = group_block + lane;
        if (group >= groups) continue;
        uint4 gate_positive;
        uint4 gate_negative;
        uint4 up_positive;
        uint4 up_negative;
        smelt_ternary_sign_masks_128(
            gate_codes + row * row_bytes + group * 32,
            gate_positive, gate_negative);
        smelt_ternary_sign_masks_128(
            up_codes + row * row_bytes + group * 32,
            up_positive, up_negative);
        const uint4 sign = *(device const uint4 *)(
            activation_planes + (group * PLANES + PLANES - 1) * WORDS);
        int gate_quantized_dot = 0;
        int up_quantized_dot = 0;
        for (uint bit = 0; bit < PLANES - 1; ++bit) {
            const uint4 plane = *(device const uint4 *)(
                activation_planes + (group * PLANES + bit) * WORDS);
            const int coefficient = int(1u << bit);
            gate_quantized_dot += coefficient
                * smelt_ternary_signed_magnitude_dot(
                    gate_positive, gate_negative, sign, plane);
            up_quantized_dot += coefficient
                * smelt_ternary_signed_magnitude_dot(
                    up_positive, up_negative, sign, plane);
        }
        const float activation_scale = float(activation_scales[group]);
        gate_acc += float(gate_quantized_dot) * activation_scale
            * float(gate_weight_scales[row * groups + group]);
        up_acc += float(up_quantized_dot) * activation_scale
            * float(up_weight_scales[row * groups + group]);
    }

    gate_acc = simd_sum(gate_acc);
    up_acc = simd_sum(up_acc);
    if (lane == 0) {
        output[row] = smelt_signed_swiglu_staged_half(
            half(gate_acc), half(up_acc));
    }
}

#define SMELT_DECLARE_TERNARY_BITPLANE_GATE_UP_SWIGLU(NAME, PLANES) \
kernel void NAME( \
    device const uchar *gate_codes [[buffer(0)]], \
    device const half *gate_weight_scales [[buffer(1)]], \
    device const uchar *up_codes [[buffer(2)]], \
    device const half *up_weight_scales [[buffer(3)]], \
    device const uint *activation_planes [[buffer(4)]], \
    device const half *activation_scales [[buffer(5)]], \
    device half *output [[buffer(6)]], \
    constant uint &rows [[buffer(7)]], \
    constant uint &cols [[buffer(8)]], \
    uint2 tgid [[threadgroup_position_in_grid]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]] \
) { \
    smelt_signed_ternary_bitplane_gate_up_swiglu_g128_rows8<PLANES>( \
        gate_codes, gate_weight_scales, up_codes, up_weight_scales, \
        activation_planes, activation_scales, output, \
        rows, cols, tgid, simd_gid, lane); \
}

SMELT_DECLARE_TERNARY_BITPLANE_GATE_UP_SWIGLU(
    signed_ternary_bitplane_i4_gate_up_swiglu_g128_rows8, 4)
SMELT_DECLARE_TERNARY_BITPLANE_GATE_UP_SWIGLU(
    signed_ternary_bitplane_i5_gate_up_swiglu_g128_rows8, 5)
SMELT_DECLARE_TERNARY_BITPLANE_GATE_UP_SWIGLU(
    signed_ternary_bitplane_i6_gate_up_swiglu_g128_rows8, 6)

kernel void signed_binary_bitplane_i2_matvec_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *weight_scales [[buffer(1)]],
    device const uint *activation_planes [[buffer(2)]],
    device const half *activation_scales [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_binary_bitplane_lowbit_matvec_g128_rows8<2, false>(
        codes, weight_scales, activation_planes, activation_scales,
        output, output, rows, cols, tgid, simd_gid, lane);
}

kernel void signed_binary_bitplane_i5_matvec_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *weight_scales [[buffer(1)]],
    device const uint *activation_planes [[buffer(2)]],
    device const half *activation_scales [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_binary_bitplane_lowbit_matvec_g128_rows8<5, false>(
        codes, weight_scales, activation_planes, activation_scales,
        output, output, rows, cols, tgid, simd_gid, lane);
}

kernel void signed_binary_bitplane_i6_matvec_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *weight_scales [[buffer(1)]],
    device const uint *activation_planes [[buffer(2)]],
    device const half *activation_scales [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_binary_bitplane_lowbit_matvec_g128_rows8<6, false>(
        codes, weight_scales, activation_planes, activation_scales,
        output, output, rows, cols, tgid, simd_gid, lane);
}

// Ternary activation-view consumers derive transient sign masks from the
// canonical interleaved codes, then retain the existing popcount arithmetic.
template <uint PLANES>
inline void smelt_signed_ternary_bitplane_matvec_g128_rows8(
    device const uchar *codes,
    device const half *weight_scales,
    device const uint *activation_planes,
    device const half *activation_scales,
    device half *output,
    uint rows,
    uint cols,
    uint2 tgid,
    uint simd_gid,
    uint lane
) {
    constexpr uint WORDS = 4;
    constexpr uint ROWS_PER_THREADGROUP = 4;
    const uint row = tgid.x * ROWS_PER_THREADGROUP + simd_gid;
    if (row >= rows) return;
    const uint groups = cols / 128;
    const uint row_bytes = cols / 4;
    float acc = 0.0f;

    for (uint group_block = 0; group_block < groups; group_block += 32) {
        const uint group = group_block + lane;
        if (group >= groups) continue;
        uint4 positive;
        uint4 negative;
        smelt_ternary_sign_masks_128(
            codes + row * row_bytes + group * 32, positive, negative);
        const uint4 sign = *(device const uint4 *)(
            activation_planes + (group * PLANES + PLANES - 1) * WORDS);

        int quantized_dot = 0;
        for (uint bit = 0; bit < PLANES - 1; ++bit) {
            const uint4 plane = *(device const uint4 *)(
                activation_planes + (group * PLANES + bit) * WORDS);
            quantized_dot += int(1u << bit)
                * smelt_ternary_signed_magnitude_dot(
                    positive, negative, sign, plane);
        }
        acc += float(quantized_dot)
            * float(activation_scales[group])
            * float(weight_scales[row * groups + group]);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        output[row] = half(acc);
    }
}

// Two rows per SIMD trades some register pressure for activation-view reuse.
// It is a measured win for wide-input projections; the emitter keeps the
// one-row kernel for shapes where greater SIMD occupancy is faster.
template <uint PLANES>
inline void smelt_signed_ternary_bitplane_matvec_g128_rows2_wide(
    device const uchar *codes,
    device const half *weight_scales,
    device const uint *activation_planes,
    device const half *activation_scales,
    device half *output,
    uint rows,
    uint cols,
    uint2 tgid,
    uint simd_gid,
    uint lane
) {
    constexpr uint WORDS = 4;
    constexpr uint ROW_TILE = 2;
    constexpr uint ROWS_PER_THREADGROUP = 4;
    const uint row0 = tgid.x * ROWS_PER_THREADGROUP + simd_gid * ROW_TILE;
    if (row0 >= rows) return;
    const uint groups = cols / 128;
    const uint row_bytes = cols / 4;
    float2 acc = 0.0f;

    for (uint group_block = 0; group_block < groups; group_block += 32) {
        const uint group = group_block + lane;
        if (group >= groups) continue;
        uint4 positive[ROW_TILE];
        uint4 negative[ROW_TILE];
        half2 weight_scale = 0.0h;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            positive[tile] = 0u;
            negative[tile] = 0u;
            if (row >= rows) continue;
            smelt_ternary_sign_masks_128(
                codes + row * row_bytes + group * 32,
                positive[tile], negative[tile]);
            weight_scale[tile] = weight_scales[row * groups + group];
        }
        const uint4 sign = *(device const uint4 *)(
            activation_planes + (group * PLANES + PLANES - 1) * WORDS);
        int2 quantized_dot = 0;
        for (uint bit = 0; bit < PLANES - 1; ++bit) {
            const uint4 plane = *(device const uint4 *)(
                activation_planes + (group * PLANES + bit) * WORDS);
            const int coefficient = int(1u << bit);
            quantized_dot.x += coefficient
                * smelt_ternary_signed_magnitude_dot(
                    positive[0], negative[0], sign, plane);
            quantized_dot.y += coefficient
                * smelt_ternary_signed_magnitude_dot(
                    positive[1], negative[1], sign, plane);
        }
        acc += float2(quantized_dot)
            * float(activation_scales[group])
            * float2(weight_scale);
    }
    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    if (lane == 0) {
        output[row0] = half(acc.x);
        if (row0 + 1 < rows) output[row0 + 1] = half(acc.y);
    }
}

#define SMELT_DECLARE_TERNARY_BITPLANE_ROWS2_WIDE(NAME, PLANES) \
kernel void NAME( \
    device const uchar *codes [[buffer(0)]], \
    device const half *weight_scales [[buffer(1)]], \
    device const uint *activation_planes [[buffer(2)]], \
    device const half *activation_scales [[buffer(3)]], \
    device half *output [[buffer(4)]], \
    constant uint &rows [[buffer(5)]], \
    constant uint &cols [[buffer(6)]], \
    uint2 tgid [[threadgroup_position_in_grid]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]] \
) { \
    smelt_signed_ternary_bitplane_matvec_g128_rows2_wide<PLANES>( \
        codes, weight_scales, activation_planes, activation_scales, \
        output, rows, cols, tgid, simd_gid, lane); \
}

SMELT_DECLARE_TERNARY_BITPLANE_ROWS2_WIDE(
    signed_ternary_bitplane_i4_matvec_g128_rows2_wide, 4)
SMELT_DECLARE_TERNARY_BITPLANE_ROWS2_WIDE(
    signed_ternary_bitplane_i5_matvec_g128_rows2_wide, 5)
SMELT_DECLARE_TERNARY_BITPLANE_ROWS2_WIDE(
    signed_ternary_bitplane_i6_matvec_g128_rows2_wide, 6)

kernel void signed_ternary_bitplane_i4_matvec_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *weight_scales [[buffer(1)]],
    device const uint *activation_planes [[buffer(2)]],
    device const half *activation_scales [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_ternary_bitplane_matvec_g128_rows8<4>(
        codes, weight_scales, activation_planes, activation_scales,
        output, rows, cols, tgid, simd_gid, lane);
}

kernel void signed_ternary_bitplane_i5_matvec_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *weight_scales [[buffer(1)]],
    device const uint *activation_planes [[buffer(2)]],
    device const half *activation_scales [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_ternary_bitplane_matvec_g128_rows8<5>(
        codes, weight_scales, activation_planes, activation_scales,
        output, rows, cols, tgid, simd_gid, lane);
}

kernel void signed_ternary_bitplane_i6_matvec_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *weight_scales [[buffer(1)]],
    device const uint *activation_planes [[buffer(2)]],
    device const half *activation_scales [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_ternary_bitplane_matvec_g128_rows8<6>(
        codes, weight_scales, activation_planes, activation_scales,
        output, rows, cols, tgid, simd_gid, lane);
}

kernel void signed_binary_bitplane_i4_matvec_add_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *weight_scales [[buffer(1)]],
    device const uint *activation_planes [[buffer(2)]],
    device const half *activation_scales [[buffer(3)]],
    device half *output [[buffer(4)]],
    device const half *residual [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &cols [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint ROW_TILE = 4;
    constexpr uint PLANES = 4;
    constexpr uint WORDS = 4;
    const uint row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint groups = cols / 128;
    const uint row_bytes = cols / 8;
    float4 acc = 0.0f;

    for (uint group_block = 0; group_block < groups; group_block += 32) {
        const uint group = group_block + lane;
        if (group >= groups) continue;

        uint4 weight[ROW_TILE];
        half4 weight_scale = 0.0h;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            weight[tile] = 0u;
            if (row >= rows) continue;
            weight[tile] = *(device const uint4 *)(
                codes + row * row_bytes + group * 16);
            weight_scale[tile] = weight_scales[row * groups + group];
        }

        const uint4 sign = *(device const uint4 *)(
            activation_planes + (group * PLANES + PLANES - 1) * WORDS);
        int4 quantized_dot = 0;
        for (uint bit = 0; bit < PLANES - 1; ++bit) {
            const uint4 plane = *(device const uint4 *)(
                activation_planes + (group * PLANES + bit) * WORDS);
            const uint active_count =
                smelt_signed_magnitude_active_count(plane);
            const int coefficient = int(1u << bit);
            quantized_dot.x += coefficient
                * smelt_binary_signed_magnitude_dot_known_active(
                    weight[0], sign, plane, active_count);
            quantized_dot.y += coefficient
                * smelt_binary_signed_magnitude_dot_known_active(
                    weight[1], sign, plane, active_count);
            quantized_dot.z += coefficient
                * smelt_binary_signed_magnitude_dot_known_active(
                    weight[2], sign, plane, active_count);
            quantized_dot.w += coefficient
                * smelt_binary_signed_magnitude_dot_known_active(
                    weight[3], sign, plane, active_count);
        }
        acc += float4(quantized_dot)
            * float(activation_scales[group])
            * float4(weight_scale);
    }

    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    acc.z = simd_sum(acc.z);
    acc.w = simd_sum(acc.w);
    if (lane == 0) {
        if (row0 + 0 < rows) output[row0 + 0] = half(
            float(half(acc.x)) + float(residual[row0 + 0]));
        if (row0 + 1 < rows) output[row0 + 1] = half(
            float(half(acc.y)) + float(residual[row0 + 1]));
        if (row0 + 2 < rows) output[row0 + 2] = half(
            float(half(acc.z)) + float(residual[row0 + 2]));
        if (row0 + 3 < rows) output[row0 + 3] = half(
            float(half(acc.w)) + float(residual[row0 + 3]));
    }
}

kernel void signed_binary_bitplane_i3_matvec_add_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *weight_scales [[buffer(1)]],
    device const uint *activation_planes [[buffer(2)]],
    device const half *activation_scales [[buffer(3)]],
    device half *output [[buffer(4)]],
    device const half *residual [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &cols [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_binary_bitplane_lowbit_matvec_g128_rows8<3, true>(
        codes, weight_scales, activation_planes, activation_scales,
        output, residual, rows, cols, tgid, simd_gid, lane);
}

kernel void signed_binary_bitplane_i2_matvec_add_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *weight_scales [[buffer(1)]],
    device const uint *activation_planes [[buffer(2)]],
    device const half *activation_scales [[buffer(3)]],
    device half *output [[buffer(4)]],
    device const half *residual [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &cols [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_binary_bitplane_lowbit_matvec_g128_rows8<2, true>(
        codes, weight_scales, activation_planes, activation_scales,
        output, residual, rows, cols, tgid, simd_gid, lane);
}

kernel void signed_binary_bitplane_i5_matvec_add_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *weight_scales [[buffer(1)]],
    device const uint *activation_planes [[buffer(2)]],
    device const half *activation_scales [[buffer(3)]],
    device half *output [[buffer(4)]],
    device const half *residual [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &cols [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_binary_bitplane_lowbit_matvec_g128_rows8<5, true>(
        codes, weight_scales, activation_planes, activation_scales,
        output, residual, rows, cols, tgid, simd_gid, lane);
}

kernel void signed_binary_bitplane_i6_matvec_add_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *weight_scales [[buffer(1)]],
    device const uint *activation_planes [[buffer(2)]],
    device const half *activation_scales [[buffer(3)]],
    device half *output [[buffer(4)]],
    device const half *residual [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &cols [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_binary_bitplane_lowbit_matvec_g128_rows8<6, true>(
        codes, weight_scales, activation_planes, activation_scales,
        output, residual, rows, cols, tgid, simd_gid, lane);
}

template <uint PLANES>
inline void smelt_signed_binary_bitplane_bank4_matvec_g128_rows8(
    device const uchar *codes,
    device const half *weight_scales,
    device const uint *activation_planes,
    device const half *activation_scales,
    device half *output0,
    device half *output1,
    device half *output2,
    device half *output3,
    uint rows0,
    uint rows1,
    uint rows2,
    uint rows3,
    uint cols,
    uint2 tgid,
    uint simd_gid,
    uint lane
) {
    constexpr uint ROW_TILE = 4;
    constexpr uint WORDS = 4;
    const uint row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint boundary1 = rows0 + rows1;
    const uint boundary2 = boundary1 + rows2;
    const uint total_rows = boundary2 + rows3;
    if (row0 >= total_rows) return;
    const uint groups = cols / 128;
    const uint row_bytes = cols / 8;
    float4 acc = 0.0f;
    for (uint group_block = 0; group_block < groups; group_block += 32) {
        const uint group = group_block + lane;
        if (group >= groups) continue;
        uint4 weight[ROW_TILE];
        half4 weight_scale = 0.0h;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            weight[tile] = 0u;
            if (row >= total_rows) continue;
            weight[tile] = *(device const uint4 *)(
                codes + row * row_bytes + group * 16);
            weight_scale[tile] = weight_scales[row * groups + group];
        }
        const uint4 sign = *(device const uint4 *)(
            activation_planes + (group * PLANES + PLANES - 1) * WORDS);
        int4 quantized_dot = 0;
        for (uint bit = 0; bit < PLANES - 1; ++bit) {
            const uint4 plane = *(device const uint4 *)(
                activation_planes + (group * PLANES + bit) * WORDS);
            const uint active_count =
                smelt_signed_magnitude_active_count(plane);
            const int coefficient = int(1u << bit);
            for (uint tile = 0; tile < ROW_TILE; ++tile) {
                quantized_dot[tile] += coefficient
                    * smelt_binary_signed_magnitude_dot_known_active(
                        weight[tile], sign, plane, active_count);
            }
        }
        acc += float4(quantized_dot)
            * float(activation_scales[group])
            * float4(weight_scale);
    }
    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    acc.z = simd_sum(acc.z);
    acc.w = simd_sum(acc.w);
    if (lane == 0) {
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row >= total_rows) continue;
            const half value = half(acc[tile]);
            if (row < rows0) output0[row] = value;
            else if (row < boundary1) output1[row - rows0] = value;
            else if (row < boundary2) output2[row - boundary1] = value;
            else output3[row - boundary2] = value;
        }
    }
}

kernel void signed_binary_bitplane_i3_bank4_matvec_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *weight_scales [[buffer(1)]],
    device const uint *activation_planes [[buffer(2)]],
    device const half *activation_scales [[buffer(3)]],
    device half *output0 [[buffer(4)]],
    device half *output1 [[buffer(5)]],
    device half *output2 [[buffer(6)]],
    device half *output3 [[buffer(7)]],
    constant uint &rows0 [[buffer(8)]],
    constant uint &rows1 [[buffer(9)]],
    constant uint &rows2 [[buffer(10)]],
    constant uint &rows3 [[buffer(11)]],
    constant uint &cols [[buffer(12)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_binary_bitplane_bank4_matvec_g128_rows8<3>(
        codes, weight_scales, activation_planes, activation_scales,
        output0, output1, output2, output3,
        rows0, rows1, rows2, rows3, cols, tgid, simd_gid, lane);
}

kernel void signed_binary_bitplane_i2_bank4_matvec_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *weight_scales [[buffer(1)]],
    device const uint *activation_planes [[buffer(2)]],
    device const half *activation_scales [[buffer(3)]],
    device half *output0 [[buffer(4)]],
    device half *output1 [[buffer(5)]],
    device half *output2 [[buffer(6)]],
    device half *output3 [[buffer(7)]],
    constant uint &rows0 [[buffer(8)]],
    constant uint &rows1 [[buffer(9)]],
    constant uint &rows2 [[buffer(10)]],
    constant uint &rows3 [[buffer(11)]],
    constant uint &cols [[buffer(12)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    smelt_signed_binary_bitplane_bank4_matvec_g128_rows8<2>(
        codes, weight_scales, activation_planes, activation_scales,
        output0, output1, output2, output3,
        rows0, rows1, rows2, rows3, cols, tgid, simd_gid, lane);
}

template <uint PLANES>
inline void smelt_signed_ternary_bitplane_bank4_matvec_g128_rows8(
    device const uchar *codes,
    device const half *weight_scales,
    device const uint *activation_planes,
    device const half *activation_scales,
    device half *output0,
    device half *output1,
    device half *output2,
    device half *output3,
    uint rows0,
    uint rows1,
    uint rows2,
    uint rows3,
    uint cols,
    uint2 tgid,
    uint simd_gid,
    uint lane
) {
    constexpr uint WORDS = 4;
    constexpr uint ROW_TILE = 2;
    constexpr uint ROWS_PER_THREADGROUP = 4;
    const uint row0 = tgid.x * ROWS_PER_THREADGROUP + simd_gid * ROW_TILE;
    const uint boundary1 = rows0 + rows1;
    const uint boundary2 = boundary1 + rows2;
    const uint total_rows = boundary2 + rows3;
    if (row0 >= total_rows) return;
    const uint groups = cols / 128;
    const uint row_bytes = cols / 4;
    float2 acc = 0.0f;
    for (uint group_block = 0; group_block < groups; group_block += 32) {
        const uint group = group_block + lane;
        if (group >= groups) continue;
        uint4 positive[ROW_TILE];
        uint4 negative[ROW_TILE];
        half2 weight_scale = 0.0h;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            positive[tile] = 0u;
            negative[tile] = 0u;
            if (row >= total_rows) continue;
            smelt_ternary_sign_masks_128(
                codes + row * row_bytes + group * 32,
                positive[tile], negative[tile]);
            weight_scale[tile] = weight_scales[row * groups + group];
        }
        const uint4 sign = *(device const uint4 *)(
            activation_planes + (group * PLANES + PLANES - 1) * WORDS);
        int2 quantized_dot = 0;
        for (uint bit = 0; bit < PLANES - 1; ++bit) {
            const uint4 plane = *(device const uint4 *)(
                activation_planes + (group * PLANES + bit) * WORDS);
            const int coefficient = int(1u << bit);
            quantized_dot.x += coefficient
                * smelt_ternary_signed_magnitude_dot(
                    positive[0], negative[0], sign, plane);
            quantized_dot.y += coefficient
                * smelt_ternary_signed_magnitude_dot(
                    positive[1], negative[1], sign, plane);
        }
        acc += float2(quantized_dot)
            * float(activation_scales[group])
            * float2(weight_scale);
    }
    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    if (lane == 0) {
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row >= total_rows) continue;
            const half value = half(acc[tile]);
            if (row < rows0) output0[row] = value;
            else if (row < boundary1) output1[row - rows0] = value;
            else if (row < boundary2) output2[row - boundary1] = value;
            else output3[row - boundary2] = value;
        }
    }
}

#define SMELT_DECLARE_TERNARY_BITPLANE_BANK4(NAME, PLANES) \
kernel void NAME( \
    device const uchar *codes [[buffer(0)]], \
    device const half *weight_scales [[buffer(1)]], \
    device const uint *activation_planes [[buffer(2)]], \
    device const half *activation_scales [[buffer(3)]], \
    device half *output0 [[buffer(4)]], \
    device half *output1 [[buffer(5)]], \
    device half *output2 [[buffer(6)]], \
    device half *output3 [[buffer(7)]], \
    constant uint &rows0 [[buffer(8)]], \
    constant uint &rows1 [[buffer(9)]], \
    constant uint &rows2 [[buffer(10)]], \
    constant uint &rows3 [[buffer(11)]], \
    constant uint &cols [[buffer(12)]], \
    uint2 tgid [[threadgroup_position_in_grid]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]] \
) { \
    smelt_signed_ternary_bitplane_bank4_matvec_g128_rows8<PLANES>( \
        codes, weight_scales, activation_planes, activation_scales, \
        output0, output1, output2, output3, \
        rows0, rows1, rows2, rows3, cols, tgid, simd_gid, lane); \
}

SMELT_DECLARE_TERNARY_BITPLANE_BANK4(
    signed_ternary_bitplane_i4_bank4_matvec_g128_rows8, 4)
SMELT_DECLARE_TERNARY_BITPLANE_BANK4(
    signed_ternary_bitplane_i5_bank4_matvec_g128_rows8, 5)
SMELT_DECLARE_TERNARY_BITPLANE_BANK4(
    signed_ternary_bitplane_i6_bank4_matvec_g128_rows8, 6)

// Exact two-consumer projection-bank lowering. Each SIMD lane loads an
// activation tile once and advances two independent accumulators in the same
// order as the standalone binary matvec. Rows may differ; the tail of the
// larger projection continues without the smaller consumer.
kernel void signed_binary_dual_matvec_g128_rows8(
    device const uchar *first_codes [[buffer(0)]],
    device const half *first_scales [[buffer(1)]],
    device const uchar *second_codes [[buffer(2)]],
    device const half *second_scales [[buffer(3)]],
    device const half *input [[buffer(4)]],
    device half *first_output [[buffer(5)]],
    device half *second_output [[buffer(6)]],
    constant uint &first_rows [[buffer(7)]],
    constant uint &second_rows [[buffer(8)]],
    constant uint &cols [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint ROW_TILE = 4;
    constexpr uint COLS_PER_LANE = 32;
    constexpr uint COLS_PER_SIMDGROUP = COLS_PER_LANE * 32;
    const uint row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint groups = cols / 128;
    const uint row_bytes = cols / 8;
    const bool has_second = row0 < second_rows;
    float4 first_acc = 0.0f;
    float4 second_acc = 0.0f;

    for (uint block = 0; block < cols; block += COLS_PER_SIMDGROUP) {
        const uint col = block + lane * COLS_PER_LANE;
        if (col >= cols) continue;
        const uint group = col / 128;
        uint4 first_packed = 0u;
        uint4 second_packed = 0u;
        half4 first_scale = 0.0h;
        half4 second_scale = 0.0h;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row < first_rows) {
                first_packed[tile] = *(device const uint *)(
                    first_codes + row * row_bytes + col / 8);
                first_scale[tile] = first_scales[row * groups + group];
            }
            if (row < second_rows) {
                second_packed[tile] = *(device const uint *)(
                    second_codes + row * row_bytes + col / 8);
                second_scale[tile] = second_scales[row * groups + group];
            }
        }

        half4 first_partial = 0.0h;
        half4 second_partial = 0.0h;
        for (uint i = 0; i < COLS_PER_LANE; i += 4) {
            const half4 x = *(device const half4 *)(input + col + i);
            first_partial.x += smelt_binary_dot4(x, first_packed.x >> i);
            first_partial.y += smelt_binary_dot4(x, first_packed.y >> i);
            first_partial.z += smelt_binary_dot4(x, first_packed.z >> i);
            first_partial.w += smelt_binary_dot4(x, first_packed.w >> i);
            if (has_second) {
                second_partial.x += smelt_binary_dot4(x, second_packed.x >> i);
                second_partial.y += smelt_binary_dot4(x, second_packed.y >> i);
                second_partial.z += smelt_binary_dot4(x, second_packed.z >> i);
                second_partial.w += smelt_binary_dot4(x, second_packed.w >> i);
            }
        }
        first_acc += float4(first_partial * first_scale);
        second_acc += float4(second_partial * second_scale);
    }

    first_acc.x = simd_sum(first_acc.x);
    first_acc.y = simd_sum(first_acc.y);
    first_acc.z = simd_sum(first_acc.z);
    first_acc.w = simd_sum(first_acc.w);
    second_acc.x = simd_sum(second_acc.x);
    second_acc.y = simd_sum(second_acc.y);
    second_acc.z = simd_sum(second_acc.z);
    second_acc.w = simd_sum(second_acc.w);
    if (lane == 0) {
        if (row0 + 0 < first_rows) first_output[row0 + 0] = half(first_acc.x);
        if (row0 + 1 < first_rows) first_output[row0 + 1] = half(first_acc.y);
        if (row0 + 2 < first_rows) first_output[row0 + 2] = half(first_acc.z);
        if (row0 + 3 < first_rows) first_output[row0 + 3] = half(first_acc.w);
        if (row0 + 0 < second_rows) second_output[row0 + 0] = half(second_acc.x);
        if (row0 + 1 < second_rows) second_output[row0 + 1] = half(second_acc.y);
        if (row0 + 2 < second_rows) second_output[row0 + 2] = half(second_acc.z);
        if (row0 + 3 < second_rows) second_output[row0 + 3] = half(second_acc.w);
    }
}

// Exact projection-bank lowering that concatenates logical row spaces without
// changing physical weight ownership. It removes a dispatch boundary while
// retaining the standalone kernel's single accumulator chain and row tile.
kernel void signed_binary_virtual_bank_matvec_g128_rows8(
    device const uchar *first_codes [[buffer(0)]],
    device const half *first_scales [[buffer(1)]],
    device const uchar *second_codes [[buffer(2)]],
    device const half *second_scales [[buffer(3)]],
    device const half *input [[buffer(4)]],
    device half *first_output [[buffer(5)]],
    device half *second_output [[buffer(6)]],
    constant uint &first_rows [[buffer(7)]],
    constant uint &second_rows [[buffer(8)]],
    constant uint &cols [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint ROW_TILE = 4;
    constexpr uint COLS_PER_LANE = 32;
    constexpr uint COLS_PER_SIMDGROUP = COLS_PER_LANE * 32;
    const uint bank_row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint total_rows = first_rows + second_rows;
    if (bank_row0 >= total_rows) return;
    const bool first = bank_row0 < first_rows;
    const uint row0 = first ? bank_row0 : bank_row0 - first_rows;
    const uint active_rows = first ? first_rows : second_rows;
    const uint groups = cols / 128;
    const uint row_bytes = cols / 8;
    float4 acc = 0.0f;

    for (uint block = 0; block < cols; block += COLS_PER_SIMDGROUP) {
        const uint col = block + lane * COLS_PER_LANE;
        if (col >= cols) continue;
        const uint group = col / 128;
        uint4 packed = 0u;
        half4 scale = 0.0h;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row >= active_rows) continue;
            if (first) {
                packed[tile] = *(device const uint *)(
                    first_codes + row * row_bytes + col / 8);
                scale[tile] = first_scales[row * groups + group];
            } else {
                packed[tile] = *(device const uint *)(
                    second_codes + row * row_bytes + col / 8);
                scale[tile] = second_scales[row * groups + group];
            }
        }
        half4 partial = 0.0h;
        for (uint i = 0; i < COLS_PER_LANE; i += 4) {
            const half4 x = *(device const half4 *)(input + col + i);
            partial.x += smelt_binary_dot4(x, packed.x >> i);
            partial.y += smelt_binary_dot4(x, packed.y >> i);
            partial.z += smelt_binary_dot4(x, packed.z >> i);
            partial.w += smelt_binary_dot4(x, packed.w >> i);
        }
        acc += float4(partial * scale);
    }

    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    acc.z = simd_sum(acc.z);
    acc.w = simd_sum(acc.w);
    if (lane == 0) {
        device half *output = first ? first_output : second_output;
        if (row0 + 0 < active_rows) output[row0 + 0] = half(acc.x);
        if (row0 + 1 < active_rows) output[row0 + 1] = half(acc.y);
        if (row0 + 2 < active_rows) output[row0 + 2] = half(acc.z);
        if (row0 + 3 < active_rows) output[row0 + 3] = half(acc.w);
    }
}

// Exact packed projection-bank lowering. CAM bank members are laid out as one
// contiguous code slab and one contiguous scale slab, so the hot loop is the
// standalone matvec over a larger row space. Only lane zero partitions the
// completed rows into their graph-owned consumer buffers.
kernel void signed_binary_packed_bank_matvec_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device const half *input [[buffer(2)]],
    device half *first_output [[buffer(3)]],
    device half *second_output [[buffer(4)]],
    constant uint &first_rows [[buffer(5)]],
    constant uint &second_rows [[buffer(6)]],
    constant uint &cols [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint ROW_TILE = 4;
    constexpr uint COLS_PER_LANE = 32;
    constexpr uint COLS_PER_SIMDGROUP = COLS_PER_LANE * 32;
    const uint row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint total_rows = first_rows + second_rows;
    if (row0 >= total_rows) return;
    const uint groups = cols / 128;
    const uint row_bytes = cols / 8;
    float4 acc = 0.0f;

    for (uint block = 0; block < cols; block += COLS_PER_SIMDGROUP) {
        const uint col = block + lane * COLS_PER_LANE;
        if (col >= cols) continue;
        const uint group = col / 128;
        uint4 packed = 0u;
        half4 scale = 0.0h;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row >= total_rows) continue;
            packed[tile] = *(device const uint *)(
                codes + row * row_bytes + col / 8);
            scale[tile] = scales[row * groups + group];
        }
        half4 partial = 0.0h;
        for (uint i = 0; i < COLS_PER_LANE; i += 4) {
            const half4 x = *(device const half4 *)(input + col + i);
            partial.x += smelt_binary_dot4(x, packed.x >> i);
            partial.y += smelt_binary_dot4(x, packed.y >> i);
            partial.z += smelt_binary_dot4(x, packed.z >> i);
            partial.w += smelt_binary_dot4(x, packed.w >> i);
        }
        acc += float4(partial * scale);
    }

    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    acc.z = simd_sum(acc.z);
    acc.w = simd_sum(acc.w);
    if (lane == 0) {
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row >= total_rows) continue;
            const half value = half(acc[tile]);
            if (row < first_rows) first_output[row] = value;
            else second_output[row - first_rows] = value;
        }
    }
}

// Production projection-bank brick: up to four graph outputs share one input
// and one physically packed row space. Zero-sized trailing members make the
// same pipeline usable for two-, three-, and four-member banks.
kernel void signed_binary_packed_bank4_matvec_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device const half *input [[buffer(2)]],
    device half *output0 [[buffer(3)]],
    device half *output1 [[buffer(4)]],
    device half *output2 [[buffer(5)]],
    device half *output3 [[buffer(6)]],
    constant uint &rows0 [[buffer(7)]],
    constant uint &rows1 [[buffer(8)]],
    constant uint &rows2 [[buffer(9)]],
    constant uint &rows3 [[buffer(10)]],
    constant uint &cols [[buffer(11)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint ROW_TILE = 4;
    constexpr uint COLS_PER_LANE = 32;
    constexpr uint COLS_PER_SIMDGROUP = COLS_PER_LANE * 32;
    const uint row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint boundary1 = rows0 + rows1;
    const uint boundary2 = boundary1 + rows2;
    const uint total_rows = boundary2 + rows3;
    if (row0 >= total_rows) return;
    const uint groups = cols / 128;
    const uint row_bytes = cols / 8;
    float4 acc = 0.0f;

    for (uint block = 0; block < cols; block += COLS_PER_SIMDGROUP) {
        const uint col = block + lane * COLS_PER_LANE;
        if (col >= cols) continue;
        const uint group = col / 128;
        uint4 packed = 0u;
        half4 scale = 0.0h;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row >= total_rows) continue;
            packed[tile] = *(device const uint *)(
                codes + row * row_bytes + col / 8);
            scale[tile] = scales[row * groups + group];
        }
        half4 partial = 0.0h;
        for (uint i = 0; i < COLS_PER_LANE; i += 4) {
            const half4 x = *(device const half4 *)(input + col + i);
            partial.x += smelt_binary_dot4(x, packed.x >> i);
            partial.y += smelt_binary_dot4(x, packed.y >> i);
            partial.z += smelt_binary_dot4(x, packed.z >> i);
            partial.w += smelt_binary_dot4(x, packed.w >> i);
        }
        acc += float4(partial * scale);
    }

    acc.x = simd_sum(acc.x);
    acc.y = simd_sum(acc.y);
    acc.z = simd_sum(acc.z);
    acc.w = simd_sum(acc.w);
    if (lane == 0) {
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row >= total_rows) continue;
            const half value = half(acc[tile]);
            if (row < rows0) output0[row] = value;
            else if (row < boundary1) output1[row - rows0] = value;
            else if (row < boundary2) output2[row - boundary1] = value;
            else output3[row - boundary2] = value;
        }
    }
}
