#include <metal_simdgroup>
#include <metal_stdlib>

#include "agent_affine_qmm.h"

using namespace metal;

inline float smelt_precise_affine_qdot16(
    uint packed_codes,
    thread const float *x_thread,
    float scale,
    float bias,
    float sum
) {
    float accum = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        const uint packed = (packed_codes >> (i * 8)) & 0xffu;
        accum +=
            (x_thread[4 * i + 0] * (packed & 0x03u) +
             x_thread[4 * i + 1] * (packed & 0x0cu) +
             x_thread[4 * i + 2] * (packed & 0x30u) +
             x_thread[4 * i + 3] * (packed & 0xc0u));
    }
    return scale * accum + sum * bias;
}

// Preserve the three graph-owned fp16 materialization points from
// activations_precise.metal::swiglu_fused. Keeping this helper in the precise
// signed translation unit lets projection fusion remove dispatches without
// changing the staged MLX arithmetic contract.
inline half smelt_precise_swiglu_staged_half(half gate, half up) {
    const half tail = 1.0h / (1.0h + exp(abs(gate)));
    const half sigmoid = gate < 0.0h ? tail : 1.0h - tail;
    const half activated = gate * sigmoid;
    return activated * up;
}

// Arithmetic-compatible with MLX affine_qmv_fast<half, 128, 2>. The package
// keeps the checkpoint's LSB-first interleaved 2-bit spelling, so each lane
// loads the exact UInt32 word consumed by the affine dot expression.
kernel void signed_ternary_affine_matvec_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device const half *bias_scales [[buffer(2)]],
    device const half *input [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint GROUP_SIZE = 128;
    constexpr uint ROW_TILE = 4;
    constexpr uint VALUES_PER_LANE = 16;
    constexpr uint BLOCK_SIZE = VALUES_PER_LANE * 32;
    const uint row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint groups = cols / GROUP_SIZE;
    const uint row_bytes = cols / 4;
    const uint lane_col = lane * VALUES_PER_LANE;
    // MLX's small-M affine_qmv_fast launch gives every activation row an
    // independent grid slice. Preserve that topology for both decode (H=1)
    // and short prefill (H=M) without changing the arithmetic kernel.
    device const half *batch_input = input + tgid.y * cols;
    device half *batch_output = output + tgid.y * rows;

    float result[ROW_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = 0; block < cols; block += BLOCK_SIZE) {
        const uint col = block + lane_col;
        float x_thread[VALUES_PER_LANE];
        float sum = 0.0f;
        for (uint i = 0; i < VALUES_PER_LANE; i += 4) {
            // Preserve MLX's half-typed addition expression before it widens
            // into the float accumulator.
            sum += batch_input[col + i + 0] + batch_input[col + i + 1]
                + batch_input[col + i + 2] + batch_input[col + i + 3];
            x_thread[i + 0] = batch_input[col + i + 0];
            x_thread[i + 1] = batch_input[col + i + 1] / 4.0f;
            x_thread[i + 2] = batch_input[col + i + 2] / 16.0f;
            x_thread[i + 3] = batch_input[col + i + 3] / 64.0f;
        }

        const uint group = col / GROUP_SIZE;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row >= rows) continue;
            const uint packed_codes = *(device const uint *)(
                codes + row * row_bytes + col / 4);
            const uint scale_index = row * groups + group;
            result[tile] += smelt_precise_affine_qdot16(
                packed_codes,
                x_thread,
                scales[scale_index],
                -bias_scales[scale_index],
                sum);
        }
    }

    for (uint tile = 0; tile < ROW_TILE; ++tile) {
        result[tile] = simd_sum(result[tile]);
    }
    if (lane == 0) {
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row < rows) batch_output[row] = half(result[tile]);
        }
    }
}

// Exact affine projection + residual epilogue. The projection result is
// explicitly rounded to fp16 before the fp16 residual addition, matching the
// staged matvec store followed by activations.metal::elementwise_add.
kernel void signed_ternary_affine_matvec_add_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device const half *bias_scales [[buffer(2)]],
    device const half *input [[buffer(3)]],
    device half *output [[buffer(4)]],
    device const half *residual [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    constant uint &cols [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint GROUP_SIZE = 128;
    constexpr uint ROW_TILE = 4;
    constexpr uint VALUES_PER_LANE = 16;
    constexpr uint BLOCK_SIZE = VALUES_PER_LANE * 32;
    const uint row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint groups = cols / GROUP_SIZE;
    const uint row_bytes = cols / 4;
    const uint lane_col = lane * VALUES_PER_LANE;
    device const half *batch_input = input + tgid.y * cols;
    device half *batch_output = output + tgid.y * rows;
    device const half *batch_residual = residual + tgid.y * rows;

    float result[ROW_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = 0; block < cols; block += BLOCK_SIZE) {
        const uint col = block + lane_col;
        float x_thread[VALUES_PER_LANE];
        float sum = 0.0f;
        for (uint i = 0; i < VALUES_PER_LANE; i += 4) {
            sum += batch_input[col + i + 0] + batch_input[col + i + 1]
                + batch_input[col + i + 2] + batch_input[col + i + 3];
            x_thread[i + 0] = batch_input[col + i + 0];
            x_thread[i + 1] = batch_input[col + i + 1] / 4.0f;
            x_thread[i + 2] = batch_input[col + i + 2] / 16.0f;
            x_thread[i + 3] = batch_input[col + i + 3] / 64.0f;
        }

        const uint group = col / GROUP_SIZE;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row >= rows) continue;
            const uint packed_codes = *(device const uint *)(
                codes + row * row_bytes + col / 4);
            const uint scale_index = row * groups + group;
            result[tile] += smelt_precise_affine_qdot16(
                packed_codes,
                x_thread,
                scales[scale_index],
                -bias_scales[scale_index],
                sum);
        }
    }

    for (uint tile = 0; tile < ROW_TILE; ++tile) {
        result[tile] = simd_sum(result[tile]);
    }
    if (lane == 0) {
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row < rows) {
                const half projected = half(result[tile]);
                batch_output[row] = projected + batch_residual[row];
            }
        }
    }
}

// Exact common-input projection bank. CAM owns the member order and the
// package writer lays compatible code/scale rows contiguously, so one virtual
// row space can replace up to four independent dispatches. Every row executes
// the same affine-QMV arithmetic as signed_ternary_affine_matvec_g128_rows8.
kernel void signed_ternary_affine_bank4_matvec_g128_rows8(
    device const uchar *codes [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device const half *bias_scales [[buffer(2)]],
    device const half *input [[buffer(3)]],
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
    constexpr uint GROUP_SIZE = 128;
    constexpr uint ROW_TILE = 4;
    constexpr uint VALUES_PER_LANE = 16;
    constexpr uint BLOCK_SIZE = VALUES_PER_LANE * 32;
    const uint row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint boundary1 = rows0 + rows1;
    const uint boundary2 = boundary1 + rows2;
    const uint total_rows = boundary2 + rows3;
    if (row0 >= total_rows) return;
    const uint groups = cols / GROUP_SIZE;
    const uint row_bytes = cols / 4;
    const uint lane_col = lane * VALUES_PER_LANE;

    float result[ROW_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = 0; block < cols; block += BLOCK_SIZE) {
        const uint col = block + lane_col;
        float x_thread[VALUES_PER_LANE];
        float sum = 0.0f;
        for (uint i = 0; i < VALUES_PER_LANE; i += 4) {
            sum += input[col + i + 0] + input[col + i + 1]
                + input[col + i + 2] + input[col + i + 3];
            x_thread[i + 0] = input[col + i + 0];
            x_thread[i + 1] = input[col + i + 1] / 4.0f;
            x_thread[i + 2] = input[col + i + 2] / 16.0f;
            x_thread[i + 3] = input[col + i + 3] / 64.0f;
        }

        const uint group = col / GROUP_SIZE;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row >= total_rows) continue;
            const uint packed_codes = *(device const uint *)(
                codes + row * row_bytes + col / 4);
            const uint scale_index = row * groups + group;
            result[tile] += smelt_precise_affine_qdot16(
                packed_codes,
                x_thread,
                scales[scale_index],
                -bias_scales[scale_index],
                sum);
        }
    }

    for (uint tile = 0; tile < ROW_TILE; ++tile) {
        result[tile] = simd_sum(result[tile]);
    }
    if (lane == 0) {
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row >= total_rows) continue;
            const half value = half(result[tile]);
            if (row < rows0) output0[row] = value;
            else if (row < boundary1) output1[row - rows0] = value;
            else if (row < boundary2) output2[row - boundary1] = value;
            else output3[row - boundary2] = value;
        }
    }
}

// Exact paired gate/up projection. Each projection retains the independent
// affine_qmv_fast accumulation and simd_sum order used by
// signed_ternary_affine_matvec_g128_rows8. The only shared work is loading and
// expanding the common fp16 activation tile. Gate and up are each rounded to
// fp16 before the staged fp16 SwiGLU helper, matching the three-dispatch graph.
kernel void signed_ternary_affine_gate_up_swiglu_g128_rows8(
    device const uchar *gate_codes [[buffer(0)]],
    device const half *gate_scales [[buffer(1)]],
    device const half *gate_bias_scales [[buffer(2)]],
    device const uchar *up_codes [[buffer(3)]],
    device const half *up_scales [[buffer(4)]],
    device const half *up_bias_scales [[buffer(5)]],
    device const half *input [[buffer(6)]],
    device half *output [[buffer(7)]],
    constant uint &rows [[buffer(8)]],
    constant uint &cols [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint GROUP_SIZE = 128;
    constexpr uint ROW_TILE = 4;
    constexpr uint VALUES_PER_LANE = 16;
    constexpr uint BLOCK_SIZE = VALUES_PER_LANE * 32;
    const uint row0 = tgid.x * (ROW_TILE * 2) + simd_gid * ROW_TILE;
    const uint groups = cols / GROUP_SIZE;
    const uint row_bytes = cols / 4;
    const uint lane_col = lane * VALUES_PER_LANE;
    device const half *batch_input = input + tgid.y * cols;
    device half *batch_output = output + tgid.y * rows;

    float gate_result[ROW_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_result[ROW_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = 0; block < cols; block += BLOCK_SIZE) {
        const uint col = block + lane_col;
        float x_thread[VALUES_PER_LANE];
        float sum = 0.0f;
        for (uint i = 0; i < VALUES_PER_LANE; i += 4) {
            sum += batch_input[col + i + 0] + batch_input[col + i + 1]
                + batch_input[col + i + 2] + batch_input[col + i + 3];
            x_thread[i + 0] = batch_input[col + i + 0];
            x_thread[i + 1] = batch_input[col + i + 1] / 4.0f;
            x_thread[i + 2] = batch_input[col + i + 2] / 16.0f;
            x_thread[i + 3] = batch_input[col + i + 3] / 64.0f;
        }

        const uint group = col / GROUP_SIZE;
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row >= rows) continue;
            const uint scale_index = row * groups + group;
            const uint gate_packed = *(device const uint *)(
                gate_codes + row * row_bytes + col / 4);
            gate_result[tile] += smelt_precise_affine_qdot16(
                gate_packed,
                x_thread,
                gate_scales[scale_index],
                -gate_bias_scales[scale_index],
                sum);

            const uint up_packed = *(device const uint *)(
                up_codes + row * row_bytes + col / 4);
            up_result[tile] += smelt_precise_affine_qdot16(
                up_packed,
                x_thread,
                up_scales[scale_index],
                -up_bias_scales[scale_index],
                sum);
        }
    }

    for (uint tile = 0; tile < ROW_TILE; ++tile) {
        gate_result[tile] = simd_sum(gate_result[tile]);
        up_result[tile] = simd_sum(up_result[tile]);
    }
    if (lane == 0) {
        for (uint tile = 0; tile < ROW_TILE; ++tile) {
            const uint row = row0 + tile;
            if (row < rows) {
                const half gate = half(gate_result[tile]);
                const half up = half(up_result[tile]);
                batch_output[row] = smelt_precise_swiglu_staged_half(gate, up);
            }
        }
    }
}

// MLX-compatible affine QMM topology for canonical interleaved ternary weights.
//
// MLX routes sufficiently tall affine quantized projections through a
// BM32/BN32/BK32 simdgroup-matrix kernel. This brick dequantizes each 32x32
// source-code tile into threadgroup memory before executing the same 2x2-warp,
// 8x8-fragment accumulation schedule.
// Routing is a compiler decision based on storage/shape/batch capabilities;
// the kernel contains no model-family assumptions.
kernel void signed_ternary_affine_qmm_g128_bm32_bn32_bk32(
    device const uchar *codes [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device const half *bias_scales [[buffer(2)]],
    device const half *input [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &cols [[buffer(6)]],
    constant uint &actual_batch [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint BM = 32;
    constexpr uint BN = 32;
    constexpr uint BK = 32;
    constexpr uint BK_PADDED = 40;
    constexpr uint GROUP_SIZE = 128;

    threadgroup half Xs[BM * BK_PADDED];
    threadgroup half Ws[BN * BK_PADDED];

    using Frag = AgentMMAFrag88f;
    using FragT = Frag::FragT;

    FragT c00 = FragT(0.0f);
    FragT c01 = FragT(0.0f);
    FragT c10 = FragT(0.0f);
    FragT c11 = FragT(0.0f);
    FragT a0 = FragT(0.0f);
    FragT a1 = FragT(0.0f);
    FragT b0 = FragT(0.0f);
    FragT b1 = FragT(0.0f);

    const uint batch_base = tgid.y * BM;
    const uint row_base = tgid.x * BN;
    const uint groups = cols / GROUP_SIZE;
    const uint row_bytes = cols / 4;
    const uint x_row = tid / 4;
    const uint x_col = (tid & 3u) * 8;
    const uint weight_row = tid / 4;
    const uint weight_col = (tid & 3u) * 8;

    for (uint k_base = 0; k_base < cols; k_base += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup half *x_dst = Xs + x_row * BK_PADDED + x_col;
        if (batch_base + x_row < actual_batch) {
            device const half *x_src = input
                + (batch_base + x_row) * cols + k_base + x_col;
            for (uint i = 0; i < 8; ++i) x_dst[i] = x_src[i];
        } else {
            for (uint i = 0; i < 8; ++i) x_dst[i] = 0.0h;
        }

        threadgroup half *w_dst = Ws + weight_row * BK_PADDED + weight_col;
        const uint global_row = row_base + weight_row;
        if (global_row < rows) {
            const uint global_col = k_base + weight_col;
            const ushort packed_codes = *(device const ushort *)(
                codes + global_row * row_bytes + global_col / 4);
            const uint scale_index = global_row * groups + k_base / GROUP_SIZE;
            const half scale = scales[scale_index];
            const half bias = -bias_scales[scale_index];
            // MLX's 2-bit dequantizer does not first unpack a logical code and
            // multiply it by `scale`. Each byte lane uses a separately rounded
            // fp16 subscale (`scale`, `scale / 4`, `scale / 16`, `scale / 64`)
            // against the still-shifted two-bit field. Those expressions are
            // mathematically equivalent but not bit-equivalent for arbitrary
            // fp16 scales. Preserve that graph boundary before the MMA tile;
            // otherwise only some activation rows happen to round back to the
            // MLX result and recurrent history eventually exposes the error.
            const half lane_scales[4] = {
                scale,
                scale / half(4.0h),
                scale / half(16.0h),
                scale / half(64.0h),
            };
            for (uint i = 0; i < 8; ++i) {
                const uint code = (uint(packed_codes) >> (2u * i)) & 3u;
                const uint byte_lane = i & 3u;
                const uint shifted_code = code << (2u * byte_lane);
                w_dst[i] = lane_scales[byte_lane] * half(shifted_code) + bias;
            }
        } else {
            for (uint i = 0; i < 8; ++i) w_dst[i] = 0.0h;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_m = (simd_gid / 2) * 8;
        const uint tile_n = (simd_gid & 1u) * 8;
        const short2 coord = Frag::getCoord(ushort(lane));
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);
            Frag::load(a0, Xs + tile_m * BK_PADDED + kk, BK_PADDED, 1, coord);
            Frag::load(a1, Xs + (tile_m + 16) * BK_PADDED + kk, BK_PADDED, 1, coord);

            simdgroup_barrier(mem_flags::mem_none);
            Frag::load(b0, Ws + tile_n * BK_PADDED + kk, 1, BK_PADDED, coord);
            Frag::load(b1, Ws + (tile_n + 16) * BK_PADDED + kk, 1, BK_PADDED, coord);

            simdgroup_barrier(mem_flags::mem_none);
            // Match MLX BlockMMA's serpentine fragment visitation.
            Frag::mma(c00, a0, b0, c00);
            Frag::mma(c01, a0, b1, c01);
            Frag::mma(c11, a1, b1, c11);
            Frag::mma(c10, a1, b0, c10);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    const short2 coord = Frag::getCoord(ushort(lane));
    const uint tile_m = (simd_gid / 2) * 8;
    const uint tile_n = (simd_gid & 1u) * 8;
    const uint output_m0 = batch_base + tile_m + uint(coord.y);
    const uint output_n0 = row_base + tile_n + uint(coord.x);
    if (output_m0 < actual_batch) {
        if (output_n0 + 0 < rows) output[output_m0 * rows + output_n0 + 0] = half(c00[0]);
        if (output_n0 + 1 < rows) output[output_m0 * rows + output_n0 + 1] = half(c00[1]);
        if (output_n0 + 16 < rows) output[output_m0 * rows + output_n0 + 16] = half(c01[0]);
        if (output_n0 + 17 < rows) output[output_m0 * rows + output_n0 + 17] = half(c01[1]);
    }
    const uint output_m1 = output_m0 + 16;
    if (output_m1 < actual_batch) {
        if (output_n0 + 0 < rows) output[output_m1 * rows + output_n0 + 0] = half(c10[0]);
        if (output_n0 + 1 < rows) output[output_m1 * rows + output_n0 + 1] = half(c10[1]);
        if (output_n0 + 16 < rows) output[output_m1 * rows + output_n0 + 16] = half(c11[0]);
        if (output_n0 + 17 < rows) output[output_m1 * rows + output_n0 + 17] = half(c11[1]);
    }
}
