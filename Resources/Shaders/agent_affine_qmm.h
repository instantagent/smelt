#pragma once

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>

using namespace metal;

struct AgentMMAFrag88f {
    using MatT = simdgroup_matrix<float, 8, 8>;
    using FragT = vec<float, 2>;

    static inline short2 getCoord(ushort simdLane) {
        const short qid = simdLane / 4;
        const short fm = (qid & 4) + ((simdLane / 2) % 4);
        const short fn = (qid & 2) * 2 + (simdLane % 2) * 2;
        return short2(fn, fm);
    }

    template <typename SrcPtrType>
    static inline void load(
        thread FragT& dst,
        SrcPtrType src,
        int strX,
        int strY,
        short2 coord
    ) {
        dst[0] = static_cast<float>(src[coord.y * strX + (coord.x + 0) * strY]);
        dst[1] = static_cast<float>(src[coord.y * strX + (coord.x + 1) * strY]);
    }

    template <typename DstPtrType>
    static inline void store(
        const thread FragT& src,
        DstPtrType dst,
        int strX,
        int strY,
        short2 coord
    ) {
        dst[coord.y * strX + (coord.x + 0) * strY] = static_cast<half>(src[0]);
        dst[coord.y * strX + (coord.x + 1) * strY] = static_cast<half>(src[1]);
    }

    static inline void mma(
        thread FragT& d,
        thread FragT& a,
        thread FragT& b,
        thread FragT& c
    ) {
        MatT dMat;
        MatT aMat;
        MatT bMat;
        MatT cMat;

        reinterpret_cast<thread FragT&>(aMat.thread_elements()) = a;
        reinterpret_cast<thread FragT&>(bMat.thread_elements()) = b;
        reinterpret_cast<thread FragT&>(cMat.thread_elements()) = c;

        simdgroup_multiply_accumulate(dMat, aMat, bMat, cMat);
        d = reinterpret_cast<thread FragT&>(dMat.thread_elements());
    }
};

inline void agent_dequantize_u4x16_to_float(
    const device uint8_t* src,
    half scale,
    half bias,
    threadgroup float* dst
) {
    float scaleF = float(scale);
    float biasF = float(bias);
    for (uint i = 0; i < 8; i++) {
        uint packed = src[i];
        dst[2 * i] = scaleF * float(packed & 0x0Fu) + biasF;
        dst[2 * i + 1] = scaleF * float((packed >> 4) & 0x0Fu) + biasF;
    }
}

inline void agent_dequantize_u4x8_to_float(
    const device uint8_t* src,
    half scale,
    half bias,
    threadgroup float* dst
) {
    float scaleF = float(scale);
    float biasF = float(bias);
    for (uint i = 0; i < 4; i++) {
        uint packed = src[i];
        dst[2 * i] = scaleF * float(packed & 0x0Fu) + biasF;
        dst[2 * i + 1] = scaleF * float((packed >> 4) & 0x0Fu) + biasF;
    }
}

inline void agent_dequantize_u4x8_to_float_scaled(
    const device uint8_t* src,
    half scale,
    threadgroup float* dst
) {
    float scaleF = float(scale);
    for (uint i = 0; i < 4; i++) {
        uint packed = src[i];
        dst[2 * i] = scaleF * float(packed & 0x0Fu);
        dst[2 * i + 1] = scaleF * float((packed >> 4) & 0x0Fu);
    }
}

template <
    uint FIXED_ROWS,
    uint FIXED_COLS,
    uint FIXED_GROUP_SIZE,
    uint BATCH_TILE,
    bool CLEAR_PADDING = true
>
inline void agent_affine_qmm_fixed_batched_full(
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device const half*    input,
    device half*          output,
    uint                  actualBatch,
    threadgroup half*     Xs,
    threadgroup float*    Ws,
    uint2                 tgid,
    uint                  tid,
    uint                  simd_lane,
    uint                  simd_group
) {
    static_assert(
        BATCH_TILE == 8 || BATCH_TILE == 16,
        "qmm full affine path expects batch tile 8 or 16"
    );

    constexpr ushort BM = BATCH_TILE;
    constexpr ushort BN = 32;
    constexpr ushort BK = 32;
    constexpr ushort BK_PADDED = CLEAR_PADDING ? BK + 8 : BK + 2;
    constexpr ushort NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr ushort OUTPUT_COLS_PER_SIMDGROUP = 16;
    constexpr ushort SIMDGROUPS_N = 2;
    constexpr ushort ROWS_PER_SIMDGROUP = 8;

    using Frag = AgentMMAFrag88f;
    using FragT = Frag::FragT;

    const uint batchBase = tgid.y * BM;
    const uint rowBase = tgid.x * BN;
    const uint threadIdx = tid;
    const short2 coord = Frag::getCoord(ushort(simd_lane));
    const bool fullTile = batchBase + BM <= actualBatch;

    FragT c0 = FragT(0.0f);
    FragT c1 = FragT(0.0f);
    FragT a = FragT(0.0f);
    FragT b0 = FragT(0.0f);
    FragT b1 = FragT(0.0f);

    if (fullTile) {
        for (uint kBase = 0; kBase < FIXED_COLS; kBase += BK) {
            const uint xRow = threadIdx / 8;
            const uint xCol = (threadIdx % 8) * 4;
            device const half* xSrc = input + (batchBase + xRow) * FIXED_COLS + kBase + xCol;
            threadgroup half* xDst = Xs + xRow * BK_PADDED + xCol;
            xDst[0] = xSrc[0];
            xDst[1] = xSrc[1];
            xDst[2] = xSrc[2];
            xDst[3] = xSrc[3];

            if (CLEAR_PADDING && threadIdx < BM) {
                threadgroup half* pad = Xs + threadIdx * BK_PADDED + BK;
                pad[0] = half(0.0h);
                pad[1] = half(0.0h);
                pad[2] = half(0.0h);
                pad[3] = half(0.0h);
                pad[4] = half(0.0h);
                pad[5] = half(0.0h);
                pad[6] = half(0.0h);
                pad[7] = half(0.0h);
            }

            const uint groupIdx = kBase / FIXED_GROUP_SIZE;
            constexpr uint WEIGHT_LOAD_PASSES = 16 / BATCH_TILE;
            for (uint pass = 0; pass < WEIGHT_LOAD_PASSES; pass++) {
                const uint weightThread = threadIdx + pass * BATCH_TILE * 8;
                const uint wRow = weightThread >> 2;
                const uint wQuarter = weightThread & 3u;
                const uint colBase = wQuarter * 8;
                device const uint8_t* wSrc =
                    weights + (rowBase + wRow) * (FIXED_COLS / 2)
                        + (kBase / 2) + (colBase / 2);
                threadgroup float* wDst = Ws + wRow * BK_PADDED + colBase;
                agent_dequantize_u4x8_to_float(
                    wSrc,
                    scales[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                    biases[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                    wDst
                );
            }

            if (CLEAR_PADDING && threadIdx < BN) {
                threadgroup float* pad = Ws + threadIdx * BK_PADDED + BK;
                pad[0] = 0.0f;
                pad[1] = 0.0f;
                pad[2] = 0.0f;
                pad[3] = 0.0f;
                pad[4] = 0.0f;
                pad[5] = 0.0f;
                pad[6] = 0.0f;
                pad[7] = 0.0f;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint simdRowBase = (simd_group / SIMDGROUPS_N) * ROWS_PER_SIMDGROUP;
            const uint simdColBase = (simd_group % SIMDGROUPS_N) * OUTPUT_COLS_PER_SIMDGROUP;
            for (uint kk = 0; kk < BK; kk += 8) {
                simdgroup_barrier(mem_flags::mem_none);
                Frag::load(a, Xs + simdRowBase * BK_PADDED + kk, BK_PADDED, 1, coord);
                Frag::load(b0, Ws + simdColBase * BK_PADDED + kk, 1, BK_PADDED, coord);
                Frag::load(b1, Ws + (simdColBase + 8) * BK_PADDED + kk, 1, BK_PADDED, coord);
                simdgroup_barrier(mem_flags::mem_none);
                Frag::mma(c0, a, b0, c0);
                Frag::mma(c1, a, b1, c1);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    } else {
        for (uint kBase = 0; kBase < FIXED_COLS; kBase += BK) {
            const uint xRow = threadIdx / 8;
            const uint xCol = (threadIdx % 8) * 4;
            threadgroup half* xDst = Xs + xRow * BK_PADDED + xCol;
            if (batchBase + xRow < actualBatch) {
                device const half* xSrc = input + (batchBase + xRow) * FIXED_COLS + kBase + xCol;
                xDst[0] = xSrc[0];
                xDst[1] = xSrc[1];
                xDst[2] = xSrc[2];
                xDst[3] = xSrc[3];
            } else {
                xDst[0] = half(0.0h);
                xDst[1] = half(0.0h);
                xDst[2] = half(0.0h);
                xDst[3] = half(0.0h);
            }

            if (CLEAR_PADDING && threadIdx < BM) {
                threadgroup half* pad = Xs + threadIdx * BK_PADDED + BK;
                pad[0] = half(0.0h);
                pad[1] = half(0.0h);
                pad[2] = half(0.0h);
                pad[3] = half(0.0h);
                pad[4] = half(0.0h);
                pad[5] = half(0.0h);
                pad[6] = half(0.0h);
                pad[7] = half(0.0h);
            }

            const uint groupIdx = kBase / FIXED_GROUP_SIZE;
            constexpr uint WEIGHT_LOAD_PASSES = 16 / BATCH_TILE;
            for (uint pass = 0; pass < WEIGHT_LOAD_PASSES; pass++) {
                const uint weightThread = threadIdx + pass * BATCH_TILE * 8;
                const uint wRow = weightThread >> 2;
                const uint wQuarter = weightThread & 3u;
                const uint colBase = wQuarter * 8;
                device const uint8_t* wSrc =
                    weights + (rowBase + wRow) * (FIXED_COLS / 2)
                        + (kBase / 2) + (colBase / 2);
                threadgroup float* wDst = Ws + wRow * BK_PADDED + colBase;
                agent_dequantize_u4x8_to_float(
                    wSrc,
                    scales[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                    biases[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                    wDst
                );
            }

            if (CLEAR_PADDING && threadIdx < BN) {
                threadgroup float* pad = Ws + threadIdx * BK_PADDED + BK;
                pad[0] = 0.0f;
                pad[1] = 0.0f;
                pad[2] = 0.0f;
                pad[3] = 0.0f;
                pad[4] = 0.0f;
                pad[5] = 0.0f;
                pad[6] = 0.0f;
                pad[7] = 0.0f;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint simdRowBase = (simd_group / SIMDGROUPS_N) * ROWS_PER_SIMDGROUP;
            const uint simdColBase = (simd_group % SIMDGROUPS_N) * OUTPUT_COLS_PER_SIMDGROUP;
            for (uint kk = 0; kk < BK; kk += 8) {
                simdgroup_barrier(mem_flags::mem_none);
                Frag::load(a, Xs + simdRowBase * BK_PADDED + kk, BK_PADDED, 1, coord);
                Frag::load(b0, Ws + simdColBase * BK_PADDED + kk, 1, BK_PADDED, coord);
                Frag::load(b1, Ws + (simdColBase + 8) * BK_PADDED + kk, 1, BK_PADDED, coord);
                simdgroup_barrier(mem_flags::mem_none);
                Frag::mma(c0, a, b0, c0);
                Frag::mma(c1, a, b1, c1);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint simdRowBase = (simd_group / SIMDGROUPS_N) * ROWS_PER_SIMDGROUP;
    const uint simdColBase = (simd_group % SIMDGROUPS_N) * OUTPUT_COLS_PER_SIMDGROUP;
    device half* outBase =
        output + batchBase * FIXED_ROWS + rowBase + simdColBase;
    if (fullTile || batchBase + simdRowBase + coord.y < actualBatch) {
        outBase[(simdRowBase + coord.y) * FIXED_ROWS + coord.x + 0] = half(c0[0]);
        outBase[(simdRowBase + coord.y) * FIXED_ROWS + coord.x + 1] = half(c0[1]);
        outBase[(simdRowBase + coord.y) * FIXED_ROWS + coord.x + 8] = half(c1[0]);
        outBase[(simdRowBase + coord.y) * FIXED_ROWS + coord.x + 9] = half(c1[1]);
    }
}

template <ushort ACTIVATION>
inline half agent_qmm_gate_up_product(float gate, float up) {
    if (ACTIVATION == 1) {
        gate = float(half(gate));
        up = float(half(up));
        float gate3 = gate * gate * gate;
        float inner = 0.7978845608f * (gate + 0.044715f * gate3);
        inner = clamp(inner, -20.0f, 20.0f);
        float geluGate = 0.5f * gate * (1.0f + tanh(inner));
        return half(clamp(geluGate * up, -65504.0f, 65504.0f));
    }
    return half(gate / (1.0f + exp(-gate)) * up);
}

template <uint NUM_COL_GROUPS>
inline void agent_apply_gate_up_bias_group(
    thread AgentMMAFrag88f::FragT& gateBias0,
    thread AgentMMAFrag88f::FragT& gateBias1,
    thread AgentMMAFrag88f::FragT& upBias0,
    thread AgentMMAFrag88f::FragT& upBias1,
    device const half*             gateBiases,
    device const half*             upBiases,
    uint                           rowBase,
    uint                           simdColBase,
    short2                         coord,
    uint                           biasGroup,
    float                          xsum
) {
    const uint row0 = rowBase + simdColBase + coord.x + 0;
    const uint row1 = rowBase + simdColBase + coord.x + 1;
    const uint row8 = rowBase + simdColBase + coord.x + 8;
    const uint row9 = rowBase + simdColBase + coord.x + 9;
    gateBias0[0] += float(gateBiases[row0 * NUM_COL_GROUPS + biasGroup]) * xsum;
    gateBias0[1] += float(gateBiases[row1 * NUM_COL_GROUPS + biasGroup]) * xsum;
    gateBias1[0] += float(gateBiases[row8 * NUM_COL_GROUPS + biasGroup]) * xsum;
    gateBias1[1] += float(gateBiases[row9 * NUM_COL_GROUPS + biasGroup]) * xsum;
    upBias0[0] += float(upBiases[row0 * NUM_COL_GROUPS + biasGroup]) * xsum;
    upBias0[1] += float(upBiases[row1 * NUM_COL_GROUPS + biasGroup]) * xsum;
    upBias1[0] += float(upBiases[row8 * NUM_COL_GROUPS + biasGroup]) * xsum;
    upBias1[1] += float(upBiases[row9 * NUM_COL_GROUPS + biasGroup]) * xsum;
}

template <
    uint FIXED_ROWS,
    uint FIXED_COLS,
    uint FIXED_GROUP_SIZE,
    uint BATCH_TILE,
    ushort ACTIVATION = 0,
    bool CLEAR_PADDING = true
>
inline void agent_fused_affine_gate_up_qmm_fixed_batched_full(
    device const uint8_t* gateWeights,
    device const half*    gateScales,
    device const half*    gateBiases,
    device const uint8_t* upWeights,
    device const half*    upScales,
    device const half*    upBiases,
    device const half*    input,
    device half*          output,
    uint                  actualBatch,
    threadgroup half*     Xs,
    threadgroup float*    Wg,
    threadgroup float*    Wu,
    uint2                 tgid,
    uint                  tid,
    uint                  simd_lane,
    uint                  simd_group
) {
    static_assert(BATCH_TILE == 16, "qmm full fused affine path currently expects batch tile 16");

    constexpr ushort BM = BATCH_TILE;
    constexpr ushort BN = 32;
    constexpr ushort BK = 32;
    constexpr ushort BK_PADDED = CLEAR_PADDING ? BK + 8 : BK + 2;
    constexpr ushort NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr ushort OUTPUT_COLS_PER_SIMDGROUP = 16;
    constexpr ushort SIMDGROUPS_N = 2;
    constexpr ushort ROWS_PER_SIMDGROUP = 8;

    using Frag = AgentMMAFrag88f;
    using FragT = Frag::FragT;

    const uint batchBase = tgid.y * BM;
    const uint rowBase = tgid.x * BN;
    const uint threadIdx = tid;
    const short2 coord = Frag::getCoord(ushort(simd_lane));
    const bool fullTile = batchBase + BM <= actualBatch;

    FragT gate0 = FragT(0.0f);
    FragT gate1 = FragT(0.0f);
    FragT up0 = FragT(0.0f);
    FragT up1 = FragT(0.0f);
    FragT gateBias0 = FragT(0.0f);
    FragT gateBias1 = FragT(0.0f);
    FragT upBias0 = FragT(0.0f);
    FragT upBias1 = FragT(0.0f);
    FragT a = FragT(0.0f);
    FragT b0 = FragT(0.0f);
    FragT b1 = FragT(0.0f);

    if (fullTile) {
        const uint simdColBase = (simd_group % SIMDGROUPS_N) * OUTPUT_COLS_PER_SIMDGROUP;
        uint xsumBiasGroup = 0;
        // Hoist GeGLU bias correction across k tiles in the same per-tile xsum order.
        float xsumGroupSum = 0.0f;
        for (uint kBase = 0; kBase < FIXED_COLS; kBase += BK) {
            const uint xRow = threadIdx / 8;
            const uint xCol = (threadIdx % 8) * 4;
            device const half* xSrc = input + (batchBase + xRow) * FIXED_COLS + kBase + xCol;
            threadgroup half* xDst = Xs + xRow * BK_PADDED + xCol;
            xDst[0] = xSrc[0];
            xDst[1] = xSrc[1];
            xDst[2] = xSrc[2];
            xDst[3] = xSrc[3];

            if (CLEAR_PADDING && threadIdx < BM) {
                threadgroup half* pad = Xs + threadIdx * BK_PADDED + BK;
                pad[0] = half(0.0);
                pad[1] = half(0.0);
                pad[2] = half(0.0);
                pad[3] = half(0.0);
                pad[4] = half(0.0);
                pad[5] = half(0.0);
                pad[6] = half(0.0);
                pad[7] = half(0.0);
            }

            const uint wRow = threadIdx >> 2;
            const uint wQuarter = threadIdx & 3u;
            const uint colBase = wQuarter * 8;
            const uint groupIdx = kBase / FIXED_GROUP_SIZE;

            device const uint8_t* gateSrc =
                gateWeights + (rowBase + wRow) * (FIXED_COLS / 2) + (kBase / 2) + (colBase / 2);
            device const uint8_t* upSrc =
                upWeights + (rowBase + wRow) * (FIXED_COLS / 2) + (kBase / 2) + (colBase / 2);

            threadgroup float* gateDst = Wg + wRow * BK_PADDED + colBase;
            threadgroup float* upDst = Wu + wRow * BK_PADDED + colBase;
            if (ACTIVATION == 1) {
                agent_dequantize_u4x8_to_float_scaled(
                    gateSrc,
                    gateScales[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                    gateDst
                );
                agent_dequantize_u4x8_to_float_scaled(
                    upSrc,
                    upScales[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                    upDst
                );
            } else {
                agent_dequantize_u4x8_to_float(
                    gateSrc,
                    gateScales[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                    gateBiases[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                    gateDst
                );
                agent_dequantize_u4x8_to_float(
                    upSrc,
                    upScales[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                    upBiases[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                    upDst
                );
            }

            if (CLEAR_PADDING && threadIdx < BN) {
                threadgroup float* gatePad = Wg + threadIdx * BK_PADDED + BK;
                threadgroup float* upPad = Wu + threadIdx * BK_PADDED + BK;
                gatePad[0] = 0.0f;
                gatePad[1] = 0.0f;
                gatePad[2] = 0.0f;
                gatePad[3] = 0.0f;
                gatePad[4] = 0.0f;
                gatePad[5] = 0.0f;
                gatePad[6] = 0.0f;
                gatePad[7] = 0.0f;
                upPad[0] = 0.0f;
                upPad[1] = 0.0f;
                upPad[2] = 0.0f;
                upPad[3] = 0.0f;
                upPad[4] = 0.0f;
                upPad[5] = 0.0f;
                upPad[6] = 0.0f;
                upPad[7] = 0.0f;
            }

            if (ACTIVATION == 1) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (threadIdx < BM) {
                    threadgroup half* xSumRow = Xs + threadIdx * BK_PADDED;
                    float xsum0 = 0.0f;
                    float xsum1 = 0.0f;
                    for (uint i = 0; i < 16; i++) {
                        xsum0 += float(xSumRow[i]);
                        xsum1 += float(xSumRow[16 + i]);
                    }
                    threadgroup float* sumDst = Wg + threadIdx * BK_PADDED + BK;
                    sumDst[0] = xsum0;
                    sumDst[1] = xsum1;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const uint simdRowBase = (simd_group / SIMDGROUPS_N) * ROWS_PER_SIMDGROUP;
            if (ACTIVATION == 1 && groupIdx != xsumBiasGroup) {
                agent_apply_gate_up_bias_group<NUM_COL_GROUPS>(
                    gateBias0, gateBias1, upBias0, upBias1,
                    gateBiases, upBiases, rowBase, simdColBase, coord,
                    xsumBiasGroup, xsumGroupSum
                );
                xsumGroupSum = 0.0f;
                xsumBiasGroup = groupIdx;
            }
            if (ACTIVATION == 1) {
                threadgroup float* xSums = Wg + (simdRowBase + coord.y) * BK_PADDED + BK;
                const float xsum0 = xSums[0];
                const float xsum1 = xSums[1];
                const float xsum = xsum0 + xsum1;
                xsumGroupSum += xsum;
            }
            for (uint kk = 0; kk < BK; kk += 8) {
                simdgroup_barrier(mem_flags::mem_none);
                Frag::load(a, Xs + simdRowBase * BK_PADDED + kk, BK_PADDED, 1, coord);

                Frag::load(b0, Wg + simdColBase * BK_PADDED + kk, 1, BK_PADDED, coord);
                Frag::load(b1, Wg + (simdColBase + 8) * BK_PADDED + kk, 1, BK_PADDED, coord);
                simdgroup_barrier(mem_flags::mem_none);
                Frag::mma(gate0, a, b0, gate0);
                Frag::mma(gate1, a, b1, gate1);

                Frag::load(b0, Wu + simdColBase * BK_PADDED + kk, 1, BK_PADDED, coord);
                Frag::load(b1, Wu + (simdColBase + 8) * BK_PADDED + kk, 1, BK_PADDED, coord);
                simdgroup_barrier(mem_flags::mem_none);
                Frag::mma(up0, a, b0, up0);
                Frag::mma(up1, a, b1, up1);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (ACTIVATION == 1) {
            agent_apply_gate_up_bias_group<NUM_COL_GROUPS>(
                gateBias0, gateBias1, upBias0, upBias1,
                gateBiases, upBiases, rowBase, simdColBase, coord,
                xsumBiasGroup, xsumGroupSum
            );
        }
    } else {
        const uint simdColBase = (simd_group % SIMDGROUPS_N) * OUTPUT_COLS_PER_SIMDGROUP;
        uint xsumBiasGroup = 0;
        float xsumGroupSum = 0.0f;
        for (uint kBase = 0; kBase < FIXED_COLS; kBase += BK) {
            const uint xRow = threadIdx / 8;
            const uint xCol = (threadIdx % 8) * 4;
            threadgroup half* xDst = Xs + xRow * BK_PADDED + xCol;
            if (batchBase + xRow < actualBatch) {
                device const half* xSrc = input + (batchBase + xRow) * FIXED_COLS + kBase + xCol;
                xDst[0] = xSrc[0];
                xDst[1] = xSrc[1];
                xDst[2] = xSrc[2];
                xDst[3] = xSrc[3];
            } else {
                xDst[0] = half(0.0h);
                xDst[1] = half(0.0h);
                xDst[2] = half(0.0h);
                xDst[3] = half(0.0h);
            }

            if (CLEAR_PADDING && threadIdx < BM) {
                threadgroup half* pad = Xs + threadIdx * BK_PADDED + BK;
                pad[0] = half(0.0);
                pad[1] = half(0.0);
                pad[2] = half(0.0);
                pad[3] = half(0.0);
                pad[4] = half(0.0);
                pad[5] = half(0.0);
                pad[6] = half(0.0);
                pad[7] = half(0.0);
            }

            const uint wRow = threadIdx >> 2;
            const uint wQuarter = threadIdx & 3u;
            const uint colBase = wQuarter * 8;
            const uint groupIdx = kBase / FIXED_GROUP_SIZE;

            device const uint8_t* gateSrc =
                gateWeights + (rowBase + wRow) * (FIXED_COLS / 2) + (kBase / 2) + (colBase / 2);
            device const uint8_t* upSrc =
                upWeights + (rowBase + wRow) * (FIXED_COLS / 2) + (kBase / 2) + (colBase / 2);

            threadgroup float* gateDst = Wg + wRow * BK_PADDED + colBase;
            threadgroup float* upDst = Wu + wRow * BK_PADDED + colBase;
            if (ACTIVATION == 1) {
                agent_dequantize_u4x8_to_float_scaled(
                    gateSrc,
                    gateScales[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                    gateDst
                );
                agent_dequantize_u4x8_to_float_scaled(
                    upSrc,
                    upScales[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                    upDst
                );
            } else {
                agent_dequantize_u4x8_to_float(
                    gateSrc,
                    gateScales[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                    gateBiases[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                    gateDst
                );
                agent_dequantize_u4x8_to_float(
                    upSrc,
                    upScales[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                    upBiases[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                    upDst
                );
            }

            if (CLEAR_PADDING && threadIdx < BN) {
                threadgroup float* gatePad = Wg + threadIdx * BK_PADDED + BK;
                threadgroup float* upPad = Wu + threadIdx * BK_PADDED + BK;
                gatePad[0] = 0.0f;
                gatePad[1] = 0.0f;
                gatePad[2] = 0.0f;
                gatePad[3] = 0.0f;
                gatePad[4] = 0.0f;
                gatePad[5] = 0.0f;
                gatePad[6] = 0.0f;
                gatePad[7] = 0.0f;
                upPad[0] = 0.0f;
                upPad[1] = 0.0f;
                upPad[2] = 0.0f;
                upPad[3] = 0.0f;
                upPad[4] = 0.0f;
                upPad[5] = 0.0f;
                upPad[6] = 0.0f;
                upPad[7] = 0.0f;
            }

            if (ACTIVATION == 1) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (threadIdx < BM) {
                    threadgroup half* xSumRow = Xs + threadIdx * BK_PADDED;
                    float xsum0 = 0.0f;
                    float xsum1 = 0.0f;
                    for (uint i = 0; i < 16; i++) {
                        xsum0 += float(xSumRow[i]);
                        xsum1 += float(xSumRow[16 + i]);
                    }
                    threadgroup float* sumDst = Wg + threadIdx * BK_PADDED + BK;
                    sumDst[0] = xsum0;
                    sumDst[1] = xsum1;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            } else {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const uint simdRowBase = (simd_group / SIMDGROUPS_N) * ROWS_PER_SIMDGROUP;
            if (ACTIVATION == 1 && groupIdx != xsumBiasGroup) {
                agent_apply_gate_up_bias_group<NUM_COL_GROUPS>(
                    gateBias0, gateBias1, upBias0, upBias1,
                    gateBiases, upBiases, rowBase, simdColBase, coord,
                    xsumBiasGroup, xsumGroupSum
                );
                xsumGroupSum = 0.0f;
                xsumBiasGroup = groupIdx;
            }
            if (ACTIVATION == 1) {
                threadgroup float* xSums = Wg + (simdRowBase + coord.y) * BK_PADDED + BK;
                const float xsum0 = xSums[0];
                const float xsum1 = xSums[1];
                const float xsum = xsum0 + xsum1;
                xsumGroupSum += xsum;
            }
            for (uint kk = 0; kk < BK; kk += 8) {
                simdgroup_barrier(mem_flags::mem_none);
                Frag::load(a, Xs + simdRowBase * BK_PADDED + kk, BK_PADDED, 1, coord);

                Frag::load(b0, Wg + simdColBase * BK_PADDED + kk, 1, BK_PADDED, coord);
                Frag::load(b1, Wg + (simdColBase + 8) * BK_PADDED + kk, 1, BK_PADDED, coord);
                simdgroup_barrier(mem_flags::mem_none);
                Frag::mma(gate0, a, b0, gate0);
                Frag::mma(gate1, a, b1, gate1);

                Frag::load(b0, Wu + simdColBase * BK_PADDED + kk, 1, BK_PADDED, coord);
                Frag::load(b1, Wu + (simdColBase + 8) * BK_PADDED + kk, 1, BK_PADDED, coord);
                simdgroup_barrier(mem_flags::mem_none);
                Frag::mma(up0, a, b0, up0);
                Frag::mma(up1, a, b1, up1);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (ACTIVATION == 1) {
            agent_apply_gate_up_bias_group<NUM_COL_GROUPS>(
                gateBias0, gateBias1, upBias0, upBias1,
                gateBiases, upBiases, rowBase, simdColBase, coord,
                xsumBiasGroup, xsumGroupSum
            );
        }
    }

    const uint simdRowBase = (simd_group / SIMDGROUPS_N) * ROWS_PER_SIMDGROUP;
    const uint simdColBase = (simd_group % SIMDGROUPS_N) * OUTPUT_COLS_PER_SIMDGROUP;
    device half* outBase =
        output + batchBase * FIXED_ROWS + rowBase + simdColBase;
    if (fullTile || batchBase + simdRowBase + coord.y < actualBatch) {
        if (ACTIVATION == 1) {
            gate0 += gateBias0;
            gate1 += gateBias1;
            up0 += upBias0;
            up1 += upBias1;
        }
        outBase[(simdRowBase + coord.y) * FIXED_ROWS + coord.x + 0] =
            agent_qmm_gate_up_product<ACTIVATION>(gate0[0], up0[0]);
        outBase[(simdRowBase + coord.y) * FIXED_ROWS + coord.x + 1] =
            agent_qmm_gate_up_product<ACTIVATION>(gate0[1], up0[1]);
        outBase[(simdRowBase + coord.y) * FIXED_ROWS + coord.x + 8] =
            agent_qmm_gate_up_product<ACTIVATION>(gate1[0], up1[0]);
        outBase[(simdRowBase + coord.y) * FIXED_ROWS + coord.x + 9] =
            agent_qmm_gate_up_product<ACTIVATION>(gate1[1], up1[1]);
    }
}

template <uint FIXED_COLS, uint BK_PADDED>
inline void agent_load_norm_scaled_x_tile(
    device const float* scalePtr,
    device const half*  normInput,
    device const half*  normWeight,
    device half*        normOutput,
    threadgroup half*   Xs,
    uint                batchIndex,
    uint                xRow,
    uint                kBase,
    uint                xCol,
    bool                validBatch,
    bool                writeNormOutput
) {
    threadgroup half* xDst = Xs + xRow * BK_PADDED + xCol;
    if (!validBatch) {
        xDst[0] = half(0.0h);
        xDst[1] = half(0.0h);
        xDst[2] = half(0.0h);
        xDst[3] = half(0.0h);
        return;
    }

    const float rs = scalePtr[batchIndex];
    device const half* xSrc = normInput + batchIndex * FIXED_COLS + kBase + xCol;
    device const half* wSrc = normWeight + kBase + xCol;

    half h0 = half(float(xSrc[0]) * rs * (1.0f + float(wSrc[0])));
    half h1 = half(float(xSrc[1]) * rs * (1.0f + float(wSrc[1])));
    half h2 = half(float(xSrc[2]) * rs * (1.0f + float(wSrc[2])));
    half h3 = half(float(xSrc[3]) * rs * (1.0f + float(wSrc[3])));

    xDst[0] = h0;
    xDst[1] = h1;
    xDst[2] = h2;
    xDst[3] = h3;

    if (writeNormOutput) {
        device half* out = normOutput + batchIndex * FIXED_COLS + kBase + xCol;
        out[0] = h0;
        out[1] = h1;
        out[2] = h2;
        out[3] = h3;
    }
}

template <
    uint FIXED_ROWS,
    uint FIXED_COLS,
    uint FIXED_GROUP_SIZE,
    uint BATCH_TILE,
    bool CLEAR_PADDING = true
>
inline void agent_norm_scale_affine_qmm_fixed_batched_full(
    device const float*   scalePtr,
    device const half*    normInput,
    device const half*    normWeight,
    device half*          normOutput,
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device half*          output,
    uint                  actualBatch,
    threadgroup half*     Xs,
    threadgroup float*    Ws,
    uint2                 tgid,
    uint                  tid,
    uint                  simd_lane,
    uint                  simd_group
) {
    static_assert(BATCH_TILE == 16, "qmm full affine path currently expects batch tile 16");

    constexpr ushort BM = BATCH_TILE;
    constexpr ushort BN = 32;
    constexpr ushort BK = 32;
    constexpr ushort BK_PADDED = CLEAR_PADDING ? BK + 8 : BK + 2;
    constexpr ushort NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr ushort OUTPUT_COLS_PER_SIMDGROUP = 16;
    constexpr ushort SIMDGROUPS_N = 2;
    constexpr ushort ROWS_PER_SIMDGROUP = 8;

    using Frag = AgentMMAFrag88f;
    using FragT = Frag::FragT;

    const uint batchBase = tgid.y * BM;
    const uint rowBase = tgid.x * BN;
    const uint threadIdx = tid;
    const short2 coord = Frag::getCoord(ushort(simd_lane));
    const bool fullTile = batchBase + BM <= actualBatch;
    const bool writeNormOutput = rowBase == 0;

    FragT c0 = FragT(0.0f);
    FragT c1 = FragT(0.0f);
    FragT a = FragT(0.0f);
    FragT b0 = FragT(0.0f);
    FragT b1 = FragT(0.0f);

    for (uint kBase = 0; kBase < FIXED_COLS; kBase += BK) {
        const uint xRow = threadIdx / 8;
        const uint xCol = (threadIdx % 8) * 4;
        const bool validBatch = fullTile || batchBase + xRow < actualBatch;
        agent_load_norm_scaled_x_tile<FIXED_COLS, BK_PADDED>(
            scalePtr, normInput, normWeight, normOutput, Xs,
            batchBase + xRow, xRow, kBase, xCol, validBatch, writeNormOutput
        );

        if (CLEAR_PADDING && threadIdx < BM) {
            threadgroup half* pad = Xs + threadIdx * BK_PADDED + BK;
            pad[0] = half(0.0h);
            pad[1] = half(0.0h);
            pad[2] = half(0.0h);
            pad[3] = half(0.0h);
            pad[4] = half(0.0h);
            pad[5] = half(0.0h);
            pad[6] = half(0.0h);
            pad[7] = half(0.0h);
        }

        const uint wRow = threadIdx >> 2;
        const uint wQuarter = threadIdx & 3u;
        const uint colBase = wQuarter * 8;
        const uint groupIdx = kBase / FIXED_GROUP_SIZE;
        device const uint8_t* wSrc =
            weights + (rowBase + wRow) * (FIXED_COLS / 2) + (kBase / 2) + (colBase / 2);
        threadgroup float* wDst = Ws + wRow * BK_PADDED + colBase;
        agent_dequantize_u4x8_to_float(
            wSrc,
            scales[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
            biases[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
            wDst
        );

        if (CLEAR_PADDING && threadIdx < BN) {
            threadgroup float* pad = Ws + threadIdx * BK_PADDED + BK;
            pad[0] = 0.0f;
            pad[1] = 0.0f;
            pad[2] = 0.0f;
            pad[3] = 0.0f;
            pad[4] = 0.0f;
            pad[5] = 0.0f;
            pad[6] = 0.0f;
            pad[7] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint simdRowBase = (simd_group / SIMDGROUPS_N) * ROWS_PER_SIMDGROUP;
        const uint simdColBase = (simd_group % SIMDGROUPS_N) * OUTPUT_COLS_PER_SIMDGROUP;
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);
            Frag::load(a, Xs + simdRowBase * BK_PADDED + kk, BK_PADDED, 1, coord);
            Frag::load(b0, Ws + simdColBase * BK_PADDED + kk, 1, BK_PADDED, coord);
            Frag::load(b1, Ws + (simdColBase + 8) * BK_PADDED + kk, 1, BK_PADDED, coord);
            simdgroup_barrier(mem_flags::mem_none);
            Frag::mma(c0, a, b0, c0);
            Frag::mma(c1, a, b1, c1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint simdRowBase = (simd_group / SIMDGROUPS_N) * ROWS_PER_SIMDGROUP;
    const uint simdColBase = (simd_group % SIMDGROUPS_N) * OUTPUT_COLS_PER_SIMDGROUP;
    device half* outBase =
        output + batchBase * FIXED_ROWS + rowBase + simdColBase;
    if (fullTile || batchBase + simdRowBase + coord.y < actualBatch) {
        outBase[(simdRowBase + coord.y) * FIXED_ROWS + coord.x + 0] = half(c0[0]);
        outBase[(simdRowBase + coord.y) * FIXED_ROWS + coord.x + 1] = half(c0[1]);
        outBase[(simdRowBase + coord.y) * FIXED_ROWS + coord.x + 8] = half(c1[0]);
        outBase[(simdRowBase + coord.y) * FIXED_ROWS + coord.x + 9] = half(c1[1]);
    }
}

template <
    uint FIXED_ROWS,
    uint FIXED_COLS,
    uint FIXED_GROUP_SIZE,
    uint BATCH_TILE,
    ushort ACTIVATION = 0,
    bool CLEAR_PADDING = true
>
inline void agent_norm_scale_fused_affine_gate_up_qmm_fixed_batched_full(
    device const float*   scalePtr,
    device const half*    normInput,
    device const half*    normWeight,
    device half*          normOutput,
    device const uint8_t* gateWeights,
    device const half*    gateScales,
    device const half*    gateBiases,
    device const uint8_t* upWeights,
    device const half*    upScales,
    device const half*    upBiases,
    device half*          output,
    uint                  actualBatch,
    threadgroup half*     Xs,
    threadgroup float*    Wg,
    threadgroup float*    Wu,
    uint2                 tgid,
    uint                  tid,
    uint                  simd_lane,
    uint                  simd_group
) {
    static_assert(BATCH_TILE == 16, "qmm full fused affine path currently expects batch tile 16");

    constexpr ushort BM = BATCH_TILE;
    constexpr ushort BN = 32;
    constexpr ushort BK = 32;
    constexpr ushort BK_PADDED = CLEAR_PADDING ? BK + 8 : BK + 2;
    constexpr ushort NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr ushort OUTPUT_COLS_PER_SIMDGROUP = 16;
    constexpr ushort SIMDGROUPS_N = 2;
    constexpr ushort ROWS_PER_SIMDGROUP = 8;

    using Frag = AgentMMAFrag88f;
    using FragT = Frag::FragT;

    const uint batchBase = tgid.y * BM;
    const uint rowBase = tgid.x * BN;
    const uint threadIdx = tid;
    const short2 coord = Frag::getCoord(ushort(simd_lane));
    const bool fullTile = batchBase + BM <= actualBatch;
    const bool writeNormOutput = rowBase == 0;

    FragT gate0 = FragT(0.0f);
    FragT gate1 = FragT(0.0f);
    FragT up0 = FragT(0.0f);
    FragT up1 = FragT(0.0f);
    FragT gateBias0 = FragT(0.0f);
    FragT gateBias1 = FragT(0.0f);
    FragT upBias0 = FragT(0.0f);
    FragT upBias1 = FragT(0.0f);
    FragT a = FragT(0.0f);
    FragT b0 = FragT(0.0f);
    FragT b1 = FragT(0.0f);

    const uint simdColBase = (simd_group % SIMDGROUPS_N) * OUTPUT_COLS_PER_SIMDGROUP;
    uint xsumBiasGroup = 0;
    float xsumGroupSum = 0.0f;
    for (uint kBase = 0; kBase < FIXED_COLS; kBase += BK) {
        const uint xRow = threadIdx / 8;
        const uint xCol = (threadIdx % 8) * 4;
        const bool validBatch = fullTile || batchBase + xRow < actualBatch;
        agent_load_norm_scaled_x_tile<FIXED_COLS, BK_PADDED>(
            scalePtr, normInput, normWeight, normOutput, Xs,
            batchBase + xRow, xRow, kBase, xCol, validBatch, writeNormOutput
        );

        if (CLEAR_PADDING && threadIdx < BM) {
            threadgroup half* pad = Xs + threadIdx * BK_PADDED + BK;
            pad[0] = half(0.0h);
            pad[1] = half(0.0h);
            pad[2] = half(0.0h);
            pad[3] = half(0.0h);
            pad[4] = half(0.0h);
            pad[5] = half(0.0h);
            pad[6] = half(0.0h);
            pad[7] = half(0.0h);
        }

        const uint wRow = threadIdx >> 2;
        const uint wQuarter = threadIdx & 3u;
        const uint colBase = wQuarter * 8;
        const uint groupIdx = kBase / FIXED_GROUP_SIZE;

        device const uint8_t* gateSrc =
            gateWeights + (rowBase + wRow) * (FIXED_COLS / 2) + (kBase / 2) + (colBase / 2);
        device const uint8_t* upSrc =
            upWeights + (rowBase + wRow) * (FIXED_COLS / 2) + (kBase / 2) + (colBase / 2);

        threadgroup float* gateDst = Wg + wRow * BK_PADDED + colBase;
        threadgroup float* upDst = Wu + wRow * BK_PADDED + colBase;
        // ACTIVATION==1 (GeGLU) mirrors the non-norm-scale FFN path's
        // precision profile: scaled-only dequant + per-row `bias * xsum`
        // accumulation after the MMA, instead of baking biases into every
        // dequantized weight. Algebraically equal but fp16 rounds
        // differently — bit-parity matters so that toggling the cooperative
        // fusion in and out of the dispatch table doesn't shift the
        // drafter's α (e.g. 2.75 ↔ 2.92).
        if (ACTIVATION == 1) {
            agent_dequantize_u4x8_to_float_scaled(
                gateSrc,
                gateScales[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                gateDst
            );
            agent_dequantize_u4x8_to_float_scaled(
                upSrc,
                upScales[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                upDst
            );
        } else {
            agent_dequantize_u4x8_to_float(
                gateSrc,
                gateScales[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                gateBiases[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                gateDst
            );
            agent_dequantize_u4x8_to_float(
                upSrc,
                upScales[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                upBiases[(rowBase + wRow) * NUM_COL_GROUPS + groupIdx],
                upDst
            );
        }

        if (CLEAR_PADDING && threadIdx < BN) {
            threadgroup float* gatePad = Wg + threadIdx * BK_PADDED + BK;
            threadgroup float* upPad = Wu + threadIdx * BK_PADDED + BK;
            gatePad[0] = 0.0f;
            gatePad[1] = 0.0f;
            gatePad[2] = 0.0f;
            gatePad[3] = 0.0f;
            gatePad[4] = 0.0f;
            gatePad[5] = 0.0f;
            gatePad[6] = 0.0f;
            gatePad[7] = 0.0f;
            upPad[0] = 0.0f;
            upPad[1] = 0.0f;
            upPad[2] = 0.0f;
            upPad[3] = 0.0f;
            upPad[4] = 0.0f;
            upPad[5] = 0.0f;
            upPad[6] = 0.0f;
            upPad[7] = 0.0f;
        }

        if (ACTIVATION == 1) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (threadIdx < BM) {
                threadgroup half* xSumRow = Xs + threadIdx * BK_PADDED;
                float xsum0 = 0.0f;
                float xsum1 = 0.0f;
                for (uint i = 0; i < 16; i++) {
                    xsum0 += float(xSumRow[i]);
                    xsum1 += float(xSumRow[16 + i]);
                }
                threadgroup float* sumDst = Wg + threadIdx * BK_PADDED + BK;
                sumDst[0] = xsum0;
                sumDst[1] = xsum1;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const uint simdRowBase = (simd_group / SIMDGROUPS_N) * ROWS_PER_SIMDGROUP;
        if (ACTIVATION == 1 && groupIdx != xsumBiasGroup) {
            agent_apply_gate_up_bias_group<NUM_COL_GROUPS>(
                gateBias0, gateBias1, upBias0, upBias1,
                gateBiases, upBiases, rowBase, simdColBase, coord,
                xsumBiasGroup, xsumGroupSum
            );
            xsumGroupSum = 0.0f;
            xsumBiasGroup = groupIdx;
        }
        if (ACTIVATION == 1) {
            threadgroup float* xSums = Wg + (simdRowBase + coord.y) * BK_PADDED + BK;
            const float xsum0 = xSums[0];
            const float xsum1 = xSums[1];
            const float xsum = xsum0 + xsum1;
            xsumGroupSum += xsum;
        }
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_barrier(mem_flags::mem_none);
            Frag::load(a, Xs + simdRowBase * BK_PADDED + kk, BK_PADDED, 1, coord);

            Frag::load(b0, Wg + simdColBase * BK_PADDED + kk, 1, BK_PADDED, coord);
            Frag::load(b1, Wg + (simdColBase + 8) * BK_PADDED + kk, 1, BK_PADDED, coord);
            simdgroup_barrier(mem_flags::mem_none);
            Frag::mma(gate0, a, b0, gate0);
            Frag::mma(gate1, a, b1, gate1);

            Frag::load(b0, Wu + simdColBase * BK_PADDED + kk, 1, BK_PADDED, coord);
            Frag::load(b1, Wu + (simdColBase + 8) * BK_PADDED + kk, 1, BK_PADDED, coord);
            simdgroup_barrier(mem_flags::mem_none);
            Frag::mma(up0, a, b0, up0);
            Frag::mma(up1, a, b1, up1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (ACTIVATION == 1) {
        agent_apply_gate_up_bias_group<NUM_COL_GROUPS>(
            gateBias0, gateBias1, upBias0, upBias1,
            gateBiases, upBiases, rowBase, simdColBase, coord,
            xsumBiasGroup, xsumGroupSum
        );
    }

    const uint simdRowBase = (simd_group / SIMDGROUPS_N) * ROWS_PER_SIMDGROUP;
    device half* outBase =
        output + batchBase * FIXED_ROWS + rowBase + simdColBase;
    if (fullTile || batchBase + simdRowBase + coord.y < actualBatch) {
        if (ACTIVATION == 1) {
            gate0 += gateBias0;
            gate1 += gateBias1;
            up0 += upBias0;
            up1 += upBias1;
        }
        outBase[(simdRowBase + coord.y) * FIXED_ROWS + coord.x + 0] =
            agent_qmm_gate_up_product<ACTIVATION>(gate0[0], up0[0]);
        outBase[(simdRowBase + coord.y) * FIXED_ROWS + coord.x + 1] =
            agent_qmm_gate_up_product<ACTIVATION>(gate0[1], up0[1]);
        outBase[(simdRowBase + coord.y) * FIXED_ROWS + coord.x + 8] =
            agent_qmm_gate_up_product<ACTIVATION>(gate1[0], up1[0]);
        outBase[(simdRowBase + coord.y) * FIXED_ROWS + coord.x + 9] =
            agent_qmm_gate_up_product<ACTIVATION>(gate1[1], up1[1]);
    }
}
