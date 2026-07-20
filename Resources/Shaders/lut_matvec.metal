#include <metal_stdlib>
#include "agent_affine_qmm.h"
using namespace metal;

// ─── Fused grouped-LUT u4 matrix-vector multiply ───
// Weight matrix [R, C] stored as packed u4 (2 values per byte).
// Each group of `groupSize` rows shares a 16-entry FP16 LUT.
// Input [C], Output [R].
//
// 2 simdgroups × 4 rows each = 8 output rows per threadgroup.
// Each simdgroup independently reduces via simd_sum — no cross-simdgroup
// barrier or threadgroup reduction needed. Smaller TGs = better occupancy.
// LUT in threadgroup SRAM (Apple's fast path for broadcast reads).
//
// Dispatch: ceil(R/8) threadgroups × 64 threads.

constant uint MATVEC_TPG = 64;
constant uint ROWS_PER_SG = 4;
constant uint ROWS_PER_TG = 8;

// Function constants for compile-time specialization.
// When set at pipeline creation, the compiler can unroll the inner loop
// and optimize buffer access patterns for the known column count / group size.
// Must be provided via MTLFunctionConstantValues at pipeline creation time.
constant uint FC_COLS       [[function_constant(0)]];
constant uint FC_GROUP_SIZE [[function_constant(1)]];

kernel void fused_lut_matvec(
    device const uint8_t* indices   [[buffer(0)]],  // [R, C/2] packed u4
    device const half*    lut       [[buffer(1)]],  // [nGroups, 16]
    device const half*    input     [[buffer(2)]],  // [C]
    device half*          output    [[buffer(3)]],  // [R]
    constant uint&        numRows   [[buffer(4)]],  // R (for bounds on last TG)
    uint tgid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint baseRow = tgid * ROWS_PER_TG;
    uint halfCols = FC_COLS / 2;
    uint validRows = min(ROWS_PER_TG, numRows - baseRow);

    // Each simdgroup owns 4 consecutive rows
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint sgValidRows = (validRows > simd_group * ROWS_PER_SG)
        ? min(ROWS_PER_SG, validRows - simd_group * ROWS_PER_SG)
        : 0u;

    // Load LUT(s) into threadgroup memory. With 8 rows and FC_GROUP_SIZE=16,
    // at most 2 LUTs are needed (when rows straddle a group boundary).
    uint group0 = baseRow / FC_GROUP_SIZE;
    uint groupLast = (baseRow + validRows - 1) / FC_GROUP_SIZE;

    threadgroup half lr0[16];
    threadgroup half lr1[16];
    if (tid < 16) {
        lr0[tid] = lut[group0 * 16 + tid];
        if (group0 != groupLast) {
            lr1[tid] = lut[groupLast * 16 + tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgValidRows == 0) return;

    // Select LUT per row (precompute outside inner loop)
    threadgroup const half* lut_r0 = (sgBaseRow / FC_GROUP_SIZE == group0) ? lr0 : lr1;
    threadgroup const half* lut_r1 = ((sgBaseRow + 1) / FC_GROUP_SIZE == group0) ? lr0 : lr1;
    threadgroup const half* lut_r2 = ((sgBaseRow + 2) / FC_GROUP_SIZE == group0) ? lr0 : lr1;
    threadgroup const half* lut_r3 = ((sgBaseRow + 3) / FC_GROUP_SIZE == group0) ? lr0 : lr1;

    // Accumulators for up to 4 rows
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // Row pointers as uint2 for 8-byte reads (16 nibbles per load)
    uint eighthCols = halfCols / 8;  // number of uint2 chunks
    device const uint2* r0 = (device const uint2*)(indices + sgBaseRow * halfCols);
    device const uint2* r1 = (sgValidRows > 1) ? (device const uint2*)(indices + (sgBaseRow + 1) * halfCols) : r0;
    device const uint2* r2 = (sgValidRows > 2) ? (device const uint2*)(indices + (sgBaseRow + 2) * halfCols) : r0;
    device const uint2* r3 = (sgValidRows > 3) ? (device const uint2*)(indices + (sgBaseRow + 3) * halfCols) : r0;

    // 32 threads stride across columns — each uint2 covers 16 nibbles = 16 elements
    for (uint j = simd_lane; j < eighthCols; j += 32) {
        uint col = j * 16;

        // Load 16 input values (four half4 loads)
        half4 in0 = *(device const half4*)(input + col);
        half4 in1 = *(device const half4*)(input + col + 4);
        half4 in2 = *(device const half4*)(input + col + 8);
        half4 in3 = *(device const half4*)(input + col + 12);

        // Row 0 (always valid)
        {
            uint2 pp = r0[j];
            acc0 += float(lut_r0[ pp.x        & 0xF]) * float(in0[0]);
            acc0 += float(lut_r0[(pp.x >>  4) & 0xF]) * float(in0[1]);
            acc0 += float(lut_r0[(pp.x >>  8) & 0xF]) * float(in0[2]);
            acc0 += float(lut_r0[(pp.x >> 12) & 0xF]) * float(in0[3]);
            acc0 += float(lut_r0[(pp.x >> 16) & 0xF]) * float(in1[0]);
            acc0 += float(lut_r0[(pp.x >> 20) & 0xF]) * float(in1[1]);
            acc0 += float(lut_r0[(pp.x >> 24) & 0xF]) * float(in1[2]);
            acc0 += float(lut_r0[(pp.x >> 28)       ]) * float(in1[3]);
            acc0 += float(lut_r0[ pp.y        & 0xF]) * float(in2[0]);
            acc0 += float(lut_r0[(pp.y >>  4) & 0xF]) * float(in2[1]);
            acc0 += float(lut_r0[(pp.y >>  8) & 0xF]) * float(in2[2]);
            acc0 += float(lut_r0[(pp.y >> 12) & 0xF]) * float(in2[3]);
            acc0 += float(lut_r0[(pp.y >> 16) & 0xF]) * float(in3[0]);
            acc0 += float(lut_r0[(pp.y >> 20) & 0xF]) * float(in3[1]);
            acc0 += float(lut_r0[(pp.y >> 24) & 0xF]) * float(in3[2]);
            acc0 += float(lut_r0[(pp.y >> 28)       ]) * float(in3[3]);
        }

        if (sgValidRows > 1) {
            uint2 pp = r1[j];
            acc1 += float(lut_r1[ pp.x        & 0xF]) * float(in0[0]);
            acc1 += float(lut_r1[(pp.x >>  4) & 0xF]) * float(in0[1]);
            acc1 += float(lut_r1[(pp.x >>  8) & 0xF]) * float(in0[2]);
            acc1 += float(lut_r1[(pp.x >> 12) & 0xF]) * float(in0[3]);
            acc1 += float(lut_r1[(pp.x >> 16) & 0xF]) * float(in1[0]);
            acc1 += float(lut_r1[(pp.x >> 20) & 0xF]) * float(in1[1]);
            acc1 += float(lut_r1[(pp.x >> 24) & 0xF]) * float(in1[2]);
            acc1 += float(lut_r1[(pp.x >> 28)       ]) * float(in1[3]);
            acc1 += float(lut_r1[ pp.y        & 0xF]) * float(in2[0]);
            acc1 += float(lut_r1[(pp.y >>  4) & 0xF]) * float(in2[1]);
            acc1 += float(lut_r1[(pp.y >>  8) & 0xF]) * float(in2[2]);
            acc1 += float(lut_r1[(pp.y >> 12) & 0xF]) * float(in2[3]);
            acc1 += float(lut_r1[(pp.y >> 16) & 0xF]) * float(in3[0]);
            acc1 += float(lut_r1[(pp.y >> 20) & 0xF]) * float(in3[1]);
            acc1 += float(lut_r1[(pp.y >> 24) & 0xF]) * float(in3[2]);
            acc1 += float(lut_r1[(pp.y >> 28)       ]) * float(in3[3]);
        }

        if (sgValidRows > 2) {
            uint2 pp = r2[j];
            acc2 += float(lut_r2[ pp.x        & 0xF]) * float(in0[0]);
            acc2 += float(lut_r2[(pp.x >>  4) & 0xF]) * float(in0[1]);
            acc2 += float(lut_r2[(pp.x >>  8) & 0xF]) * float(in0[2]);
            acc2 += float(lut_r2[(pp.x >> 12) & 0xF]) * float(in0[3]);
            acc2 += float(lut_r2[(pp.x >> 16) & 0xF]) * float(in1[0]);
            acc2 += float(lut_r2[(pp.x >> 20) & 0xF]) * float(in1[1]);
            acc2 += float(lut_r2[(pp.x >> 24) & 0xF]) * float(in1[2]);
            acc2 += float(lut_r2[(pp.x >> 28)       ]) * float(in1[3]);
            acc2 += float(lut_r2[ pp.y        & 0xF]) * float(in2[0]);
            acc2 += float(lut_r2[(pp.y >>  4) & 0xF]) * float(in2[1]);
            acc2 += float(lut_r2[(pp.y >>  8) & 0xF]) * float(in2[2]);
            acc2 += float(lut_r2[(pp.y >> 12) & 0xF]) * float(in2[3]);
            acc2 += float(lut_r2[(pp.y >> 16) & 0xF]) * float(in3[0]);
            acc2 += float(lut_r2[(pp.y >> 20) & 0xF]) * float(in3[1]);
            acc2 += float(lut_r2[(pp.y >> 24) & 0xF]) * float(in3[2]);
            acc2 += float(lut_r2[(pp.y >> 28)       ]) * float(in3[3]);
        }

        if (sgValidRows > 3) {
            uint2 pp = r3[j];
            acc3 += float(lut_r3[ pp.x        & 0xF]) * float(in0[0]);
            acc3 += float(lut_r3[(pp.x >>  4) & 0xF]) * float(in0[1]);
            acc3 += float(lut_r3[(pp.x >>  8) & 0xF]) * float(in0[2]);
            acc3 += float(lut_r3[(pp.x >> 12) & 0xF]) * float(in0[3]);
            acc3 += float(lut_r3[(pp.x >> 16) & 0xF]) * float(in1[0]);
            acc3 += float(lut_r3[(pp.x >> 20) & 0xF]) * float(in1[1]);
            acc3 += float(lut_r3[(pp.x >> 24) & 0xF]) * float(in1[2]);
            acc3 += float(lut_r3[(pp.x >> 28)       ]) * float(in1[3]);
            acc3 += float(lut_r3[ pp.y        & 0xF]) * float(in2[0]);
            acc3 += float(lut_r3[(pp.y >>  4) & 0xF]) * float(in2[1]);
            acc3 += float(lut_r3[(pp.y >>  8) & 0xF]) * float(in2[2]);
            acc3 += float(lut_r3[(pp.y >> 12) & 0xF]) * float(in2[3]);
            acc3 += float(lut_r3[(pp.y >> 16) & 0xF]) * float(in3[0]);
            acc3 += float(lut_r3[(pp.y >> 20) & 0xF]) * float(in3[1]);
            acc3 += float(lut_r3[(pp.y >> 24) & 0xF]) * float(in3[2]);
            acc3 += float(lut_r3[(pp.y >> 28)       ]) * float(in3[3]);
        }
    }

    // Tail for remainder not covered by uint2 chunks (0-7 packed bytes)
    uint tailStart = eighthCols * 8;  // in packed bytes
    for (uint j = tailStart + simd_lane; j < halfCols; j += 32) {
        uint col = j * 2;
        half i0 = input[col];
        half i1 = input[col + 1];

        uint8_t p0 = indices[sgBaseRow * halfCols + j];
        acc0 += float(lut_r0[p0 & 0xF]) * float(i0) + float(lut_r0[p0 >> 4]) * float(i1);

        if (sgValidRows > 1) {
            uint8_t p1 = indices[(sgBaseRow + 1) * halfCols + j];
            acc1 += float(lut_r1[p1 & 0xF]) * float(i0) + float(lut_r1[p1 >> 4]) * float(i1);
        }
        if (sgValidRows > 2) {
            uint8_t p2 = indices[(sgBaseRow + 2) * halfCols + j];
            acc2 += float(lut_r2[p2 & 0xF]) * float(i0) + float(lut_r2[p2 >> 4]) * float(i1);
        }
        if (sgValidRows > 3) {
            uint8_t p3 = indices[(sgBaseRow + 3) * halfCols + j];
            acc3 += float(lut_r3[p3 & 0xF]) * float(i0) + float(lut_r3[p3 >> 4]) * float(i1);
        }
    }

    // SIMD reduction only — no threadgroup barrier needed
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    // Lane 0 of each simdgroup writes its 4 rows directly
    if (simd_lane == 0) {
        output[sgBaseRow] = half(acc0);
        if (sgValidRows > 1) output[sgBaseRow + 1] = half(acc1);
        if (sgValidRows > 2) output[sgBaseRow + 2] = half(acc2);
        if (sgValidRows > 3) output[sgBaseRow + 3] = half(acc3);
    }
}

// ─── Fused LUT matvec + residual add ───
// Same as fused_lut_matvec but adds result to a residual buffer:
//   output[r] = residual[r] + dot(weights[r], input)
// Eliminates separate elementwise_add dispatch after out_proj / down_proj.
// Dispatch: ceil(R/8) threadgroups × 64 threads.

kernel void fused_lut_matvec_add(
    device const uint8_t* indices   [[buffer(0)]],  // [R, C/2] packed u4
    device const half*    lut       [[buffer(1)]],  // [nGroups, 16]
    device const half*    input     [[buffer(2)]],  // [C]
    device half*          output    [[buffer(3)]],  // [R] — written as residual + matvec result
    device const half*    residual  [[buffer(4)]],  // [R] — existing values to add to
    constant uint&        numRows   [[buffer(5)]],  // R
    uint tgid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint baseRow = tgid * ROWS_PER_TG;
    uint halfCols = FC_COLS / 2;
    uint validRows = min(ROWS_PER_TG, numRows - baseRow);

    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint sgValidRows = (validRows > simd_group * ROWS_PER_SG)
        ? min(ROWS_PER_SG, validRows - simd_group * ROWS_PER_SG)
        : 0u;

    uint group0 = baseRow / FC_GROUP_SIZE;
    uint groupLast = (baseRow + validRows - 1) / FC_GROUP_SIZE;

    threadgroup half lr0[16];
    threadgroup half lr1[16];
    if (tid < 16) {
        lr0[tid] = lut[group0 * 16 + tid];
        if (group0 != groupLast) {
            lr1[tid] = lut[groupLast * 16 + tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgValidRows == 0) return;

    threadgroup const half* lut_r0 = (sgBaseRow / FC_GROUP_SIZE == group0) ? lr0 : lr1;
    threadgroup const half* lut_r1 = ((sgBaseRow + 1) / FC_GROUP_SIZE == group0) ? lr0 : lr1;
    threadgroup const half* lut_r2 = ((sgBaseRow + 2) / FC_GROUP_SIZE == group0) ? lr0 : lr1;
    threadgroup const half* lut_r3 = ((sgBaseRow + 3) / FC_GROUP_SIZE == group0) ? lr0 : lr1;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    uint eighthCols = halfCols / 8;
    device const uint2* r0 = (device const uint2*)(indices + sgBaseRow * halfCols);
    device const uint2* r1 = (sgValidRows > 1) ? (device const uint2*)(indices + (sgBaseRow + 1) * halfCols) : r0;
    device const uint2* r2 = (sgValidRows > 2) ? (device const uint2*)(indices + (sgBaseRow + 2) * halfCols) : r0;
    device const uint2* r3 = (sgValidRows > 3) ? (device const uint2*)(indices + (sgBaseRow + 3) * halfCols) : r0;

    for (uint j = simd_lane; j < eighthCols; j += 32) {
        uint col = j * 16;
        half4 in0 = *(device const half4*)(input + col);
        half4 in1 = *(device const half4*)(input + col + 4);
        half4 in2 = *(device const half4*)(input + col + 8);
        half4 in3 = *(device const half4*)(input + col + 12);

        { uint2 pp = r0[j];
          acc0 += float(lut_r0[ pp.x        & 0xF]) * float(in0[0]) + float(lut_r0[(pp.x >>  4) & 0xF]) * float(in0[1])
                + float(lut_r0[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(lut_r0[(pp.x >> 12) & 0xF]) * float(in0[3])
                + float(lut_r0[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(lut_r0[(pp.x >> 20) & 0xF]) * float(in1[1])
                + float(lut_r0[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(lut_r0[(pp.x >> 28)       ]) * float(in1[3])
                + float(lut_r0[ pp.y        & 0xF]) * float(in2[0]) + float(lut_r0[(pp.y >>  4) & 0xF]) * float(in2[1])
                + float(lut_r0[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(lut_r0[(pp.y >> 12) & 0xF]) * float(in2[3])
                + float(lut_r0[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(lut_r0[(pp.y >> 20) & 0xF]) * float(in3[1])
                + float(lut_r0[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(lut_r0[(pp.y >> 28)       ]) * float(in3[3]); }

        if (sgValidRows > 1) { uint2 pp = r1[j];
          acc1 += float(lut_r1[ pp.x        & 0xF]) * float(in0[0]) + float(lut_r1[(pp.x >>  4) & 0xF]) * float(in0[1])
                + float(lut_r1[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(lut_r1[(pp.x >> 12) & 0xF]) * float(in0[3])
                + float(lut_r1[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(lut_r1[(pp.x >> 20) & 0xF]) * float(in1[1])
                + float(lut_r1[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(lut_r1[(pp.x >> 28)       ]) * float(in1[3])
                + float(lut_r1[ pp.y        & 0xF]) * float(in2[0]) + float(lut_r1[(pp.y >>  4) & 0xF]) * float(in2[1])
                + float(lut_r1[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(lut_r1[(pp.y >> 12) & 0xF]) * float(in2[3])
                + float(lut_r1[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(lut_r1[(pp.y >> 20) & 0xF]) * float(in3[1])
                + float(lut_r1[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(lut_r1[(pp.y >> 28)       ]) * float(in3[3]); }

        if (sgValidRows > 2) { uint2 pp = r2[j];
          acc2 += float(lut_r2[ pp.x        & 0xF]) * float(in0[0]) + float(lut_r2[(pp.x >>  4) & 0xF]) * float(in0[1])
                + float(lut_r2[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(lut_r2[(pp.x >> 12) & 0xF]) * float(in0[3])
                + float(lut_r2[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(lut_r2[(pp.x >> 20) & 0xF]) * float(in1[1])
                + float(lut_r2[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(lut_r2[(pp.x >> 28)       ]) * float(in1[3])
                + float(lut_r2[ pp.y        & 0xF]) * float(in2[0]) + float(lut_r2[(pp.y >>  4) & 0xF]) * float(in2[1])
                + float(lut_r2[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(lut_r2[(pp.y >> 12) & 0xF]) * float(in2[3])
                + float(lut_r2[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(lut_r2[(pp.y >> 20) & 0xF]) * float(in3[1])
                + float(lut_r2[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(lut_r2[(pp.y >> 28)       ]) * float(in3[3]); }

        if (sgValidRows > 3) { uint2 pp = r3[j];
          acc3 += float(lut_r3[ pp.x        & 0xF]) * float(in0[0]) + float(lut_r3[(pp.x >>  4) & 0xF]) * float(in0[1])
                + float(lut_r3[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(lut_r3[(pp.x >> 12) & 0xF]) * float(in0[3])
                + float(lut_r3[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(lut_r3[(pp.x >> 20) & 0xF]) * float(in1[1])
                + float(lut_r3[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(lut_r3[(pp.x >> 28)       ]) * float(in1[3])
                + float(lut_r3[ pp.y        & 0xF]) * float(in2[0]) + float(lut_r3[(pp.y >>  4) & 0xF]) * float(in2[1])
                + float(lut_r3[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(lut_r3[(pp.y >> 12) & 0xF]) * float(in2[3])
                + float(lut_r3[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(lut_r3[(pp.y >> 20) & 0xF]) * float(in3[1])
                + float(lut_r3[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(lut_r3[(pp.y >> 28)       ]) * float(in3[3]); }
    }

    uint tailStart = eighthCols * 8;
    for (uint j = tailStart + simd_lane; j < halfCols; j += 32) {
        uint col = j * 2;
        half i0 = input[col];
        half i1 = input[col + 1];
        uint8_t p0 = indices[sgBaseRow * halfCols + j];
        acc0 += float(lut_r0[p0 & 0xF]) * float(i0) + float(lut_r0[p0 >> 4]) * float(i1);
        if (sgValidRows > 1) { uint8_t p1 = indices[(sgBaseRow+1)*halfCols+j]; acc1 += float(lut_r1[p1&0xF])*float(i0)+float(lut_r1[p1>>4])*float(i1); }
        if (sgValidRows > 2) { uint8_t p2 = indices[(sgBaseRow+2)*halfCols+j]; acc2 += float(lut_r2[p2&0xF])*float(i0)+float(lut_r2[p2>>4])*float(i1); }
        if (sgValidRows > 3) { uint8_t p3 = indices[(sgBaseRow+3)*halfCols+j]; acc3 += float(lut_r3[p3&0xF])*float(i0)+float(lut_r3[p3>>4])*float(i1); }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    // Add residual and write
    if (simd_lane == 0) {
        output[sgBaseRow] = half(float(residual[sgBaseRow]) + acc0);
        if (sgValidRows > 1) output[sgBaseRow + 1] = half(float(residual[sgBaseRow + 1]) + acc1);
        if (sgValidRows > 2) output[sgBaseRow + 2] = half(float(residual[sgBaseRow + 2]) + acc2);
        if (sgValidRows > 3) output[sgBaseRow + 3] = half(float(residual[sgBaseRow + 3]) + acc3);
    }
}

// ─── Fused dual LUT matvec: two weight matrices, same input, two outputs ───
// Computes w1 @ input → output1 and w2 @ input → output2 in one dispatch.
// Both matrices must have the same dimensions [R, C].
// Same thread structure as fused_lut_matvec: 64 threads, 2 SG, 4 rows/SG.
// Input loaded once, reused for both projections.
// Function constants FC_COLS and FC_GROUP_SIZE used.
//
// Dispatch: ceil(R/8) threadgroups × 64 threads.

kernel void fused_dual_lut_matvec(
    device const uint8_t* w1_indices  [[buffer(0)]],  // [R, C/2] packed u4
    device const half*    w1_lut      [[buffer(1)]],  // [nGroups, 16]
    device const uint8_t* w2_indices  [[buffer(2)]],  // [R, C/2] packed u4
    device const half*    w2_lut      [[buffer(3)]],  // [nGroups, 16]
    device const half*    input       [[buffer(4)]],  // [C]
    device half*          output1     [[buffer(5)]],  // [R]
    device half*          output2     [[buffer(6)]],  // [R]
    constant uint&        numRows     [[buffer(7)]],  // R
    uint tgid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint baseRow = tgid * ROWS_PER_TG;
    uint halfCols = FC_COLS / 2;
    uint validRows = min(ROWS_PER_TG, numRows - baseRow);

    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint sgValidRows = (validRows > simd_group * ROWS_PER_SG)
        ? min(ROWS_PER_SG, validRows - simd_group * ROWS_PER_SG)
        : 0u;

    // Load LUTs into threadgroup memory — w1 and w2, 2 LUTs each (for group boundary)
    uint group0 = baseRow / FC_GROUP_SIZE;
    uint groupLast = (baseRow + validRows - 1) / FC_GROUP_SIZE;

    threadgroup half w1_lr0[16], w1_lr1[16], w2_lr0[16], w2_lr1[16];
    if (tid < 16) {
        w1_lr0[tid] = w1_lut[group0 * 16 + tid];
        if (group0 != groupLast) {
            w1_lr1[tid] = w1_lut[groupLast * 16 + tid];
        }
    }
    if (tid >= 16 && tid < 32) {
        uint idx = tid - 16;
        w2_lr0[idx] = w2_lut[group0 * 16 + idx];
        if (group0 != groupLast) {
            w2_lr1[idx] = w2_lut[groupLast * 16 + idx];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgValidRows == 0) return;

    // Select w1 LUT per row
    threadgroup const half* w1lut_r0 = (sgBaseRow / FC_GROUP_SIZE == group0) ? w1_lr0 : w1_lr1;
    threadgroup const half* w1lut_r1 = ((sgBaseRow + 1) / FC_GROUP_SIZE == group0) ? w1_lr0 : w1_lr1;
    threadgroup const half* w1lut_r2 = ((sgBaseRow + 2) / FC_GROUP_SIZE == group0) ? w1_lr0 : w1_lr1;
    threadgroup const half* w1lut_r3 = ((sgBaseRow + 3) / FC_GROUP_SIZE == group0) ? w1_lr0 : w1_lr1;

    // Select w2 LUT per row
    threadgroup const half* w2lut_r0 = (sgBaseRow / FC_GROUP_SIZE == group0) ? w2_lr0 : w2_lr1;
    threadgroup const half* w2lut_r1 = ((sgBaseRow + 1) / FC_GROUP_SIZE == group0) ? w2_lr0 : w2_lr1;
    threadgroup const half* w2lut_r2 = ((sgBaseRow + 2) / FC_GROUP_SIZE == group0) ? w2_lr0 : w2_lr1;
    threadgroup const half* w2lut_r3 = ((sgBaseRow + 3) / FC_GROUP_SIZE == group0) ? w2_lr0 : w2_lr1;

    // 8 accumulators: w1_0-3 and w2_0-3
    float w1acc0 = 0.0f, w1acc1 = 0.0f, w1acc2 = 0.0f, w1acc3 = 0.0f;
    float w2acc0 = 0.0f, w2acc1 = 0.0f, w2acc2 = 0.0f, w2acc3 = 0.0f;

    // Row pointers for w1 and w2 weight matrices
    uint eighthCols = halfCols / 8;
    device const uint2* w1r0 = (device const uint2*)(w1_indices + sgBaseRow * halfCols);
    device const uint2* w1r1 = (sgValidRows > 1) ? (device const uint2*)(w1_indices + (sgBaseRow + 1) * halfCols) : w1r0;
    device const uint2* w1r2 = (sgValidRows > 2) ? (device const uint2*)(w1_indices + (sgBaseRow + 2) * halfCols) : w1r0;
    device const uint2* w1r3 = (sgValidRows > 3) ? (device const uint2*)(w1_indices + (sgBaseRow + 3) * halfCols) : w1r0;

    device const uint2* w2r0 = (device const uint2*)(w2_indices + sgBaseRow * halfCols);
    device const uint2* w2r1 = (sgValidRows > 1) ? (device const uint2*)(w2_indices + (sgBaseRow + 1) * halfCols) : w2r0;
    device const uint2* w2r2 = (sgValidRows > 2) ? (device const uint2*)(w2_indices + (sgBaseRow + 2) * halfCols) : w2r0;
    device const uint2* w2r3 = (sgValidRows > 3) ? (device const uint2*)(w2_indices + (sgBaseRow + 3) * halfCols) : w2r0;

    // Inner loop: load input ONCE, load w1 AND w2 weights, accumulate both
    for (uint j = simd_lane; j < eighthCols; j += 32) {
        uint col = j * 16;

        // Load 16 input values (four half4 loads) — shared by both projections
        half4 in0 = *(device const half4*)(input + col);
        half4 in1 = *(device const half4*)(input + col + 4);
        half4 in2 = *(device const half4*)(input + col + 8);
        half4 in3 = *(device const half4*)(input + col + 12);

        // Row 0 (always valid) — w1
        { uint2 pp = w1r0[j];
          w1acc0 += float(w1lut_r0[ pp.x        & 0xF]) * float(in0[0]) + float(w1lut_r0[(pp.x >>  4) & 0xF]) * float(in0[1])
                  + float(w1lut_r0[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(w1lut_r0[(pp.x >> 12) & 0xF]) * float(in0[3])
                  + float(w1lut_r0[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(w1lut_r0[(pp.x >> 20) & 0xF]) * float(in1[1])
                  + float(w1lut_r0[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(w1lut_r0[(pp.x >> 28)       ]) * float(in1[3])
                  + float(w1lut_r0[ pp.y        & 0xF]) * float(in2[0]) + float(w1lut_r0[(pp.y >>  4) & 0xF]) * float(in2[1])
                  + float(w1lut_r0[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(w1lut_r0[(pp.y >> 12) & 0xF]) * float(in2[3])
                  + float(w1lut_r0[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(w1lut_r0[(pp.y >> 20) & 0xF]) * float(in3[1])
                  + float(w1lut_r0[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(w1lut_r0[(pp.y >> 28)       ]) * float(in3[3]); }
        // Row 0 — w2
        { uint2 pp = w2r0[j];
          w2acc0 += float(w2lut_r0[ pp.x        & 0xF]) * float(in0[0]) + float(w2lut_r0[(pp.x >>  4) & 0xF]) * float(in0[1])
                  + float(w2lut_r0[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(w2lut_r0[(pp.x >> 12) & 0xF]) * float(in0[3])
                  + float(w2lut_r0[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(w2lut_r0[(pp.x >> 20) & 0xF]) * float(in1[1])
                  + float(w2lut_r0[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(w2lut_r0[(pp.x >> 28)       ]) * float(in1[3])
                  + float(w2lut_r0[ pp.y        & 0xF]) * float(in2[0]) + float(w2lut_r0[(pp.y >>  4) & 0xF]) * float(in2[1])
                  + float(w2lut_r0[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(w2lut_r0[(pp.y >> 12) & 0xF]) * float(in2[3])
                  + float(w2lut_r0[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(w2lut_r0[(pp.y >> 20) & 0xF]) * float(in3[1])
                  + float(w2lut_r0[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(w2lut_r0[(pp.y >> 28)       ]) * float(in3[3]); }

        if (sgValidRows > 1) {
            // Row 1 — w1
            { uint2 pp = w1r1[j];
              w1acc1 += float(w1lut_r1[ pp.x        & 0xF]) * float(in0[0]) + float(w1lut_r1[(pp.x >>  4) & 0xF]) * float(in0[1])
                      + float(w1lut_r1[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(w1lut_r1[(pp.x >> 12) & 0xF]) * float(in0[3])
                      + float(w1lut_r1[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(w1lut_r1[(pp.x >> 20) & 0xF]) * float(in1[1])
                      + float(w1lut_r1[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(w1lut_r1[(pp.x >> 28)       ]) * float(in1[3])
                      + float(w1lut_r1[ pp.y        & 0xF]) * float(in2[0]) + float(w1lut_r1[(pp.y >>  4) & 0xF]) * float(in2[1])
                      + float(w1lut_r1[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(w1lut_r1[(pp.y >> 12) & 0xF]) * float(in2[3])
                      + float(w1lut_r1[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(w1lut_r1[(pp.y >> 20) & 0xF]) * float(in3[1])
                      + float(w1lut_r1[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(w1lut_r1[(pp.y >> 28)       ]) * float(in3[3]); }
            // Row 1 — w2
            { uint2 pp = w2r1[j];
              w2acc1 += float(w2lut_r1[ pp.x        & 0xF]) * float(in0[0]) + float(w2lut_r1[(pp.x >>  4) & 0xF]) * float(in0[1])
                      + float(w2lut_r1[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(w2lut_r1[(pp.x >> 12) & 0xF]) * float(in0[3])
                      + float(w2lut_r1[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(w2lut_r1[(pp.x >> 20) & 0xF]) * float(in1[1])
                      + float(w2lut_r1[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(w2lut_r1[(pp.x >> 28)       ]) * float(in1[3])
                      + float(w2lut_r1[ pp.y        & 0xF]) * float(in2[0]) + float(w2lut_r1[(pp.y >>  4) & 0xF]) * float(in2[1])
                      + float(w2lut_r1[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(w2lut_r1[(pp.y >> 12) & 0xF]) * float(in2[3])
                      + float(w2lut_r1[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(w2lut_r1[(pp.y >> 20) & 0xF]) * float(in3[1])
                      + float(w2lut_r1[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(w2lut_r1[(pp.y >> 28)       ]) * float(in3[3]); }
        }

        if (sgValidRows > 2) {
            // Row 2 — w1
            { uint2 pp = w1r2[j];
              w1acc2 += float(w1lut_r2[ pp.x        & 0xF]) * float(in0[0]) + float(w1lut_r2[(pp.x >>  4) & 0xF]) * float(in0[1])
                      + float(w1lut_r2[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(w1lut_r2[(pp.x >> 12) & 0xF]) * float(in0[3])
                      + float(w1lut_r2[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(w1lut_r2[(pp.x >> 20) & 0xF]) * float(in1[1])
                      + float(w1lut_r2[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(w1lut_r2[(pp.x >> 28)       ]) * float(in1[3])
                      + float(w1lut_r2[ pp.y        & 0xF]) * float(in2[0]) + float(w1lut_r2[(pp.y >>  4) & 0xF]) * float(in2[1])
                      + float(w1lut_r2[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(w1lut_r2[(pp.y >> 12) & 0xF]) * float(in2[3])
                      + float(w1lut_r2[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(w1lut_r2[(pp.y >> 20) & 0xF]) * float(in3[1])
                      + float(w1lut_r2[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(w1lut_r2[(pp.y >> 28)       ]) * float(in3[3]); }
            // Row 2 — w2
            { uint2 pp = w2r2[j];
              w2acc2 += float(w2lut_r2[ pp.x        & 0xF]) * float(in0[0]) + float(w2lut_r2[(pp.x >>  4) & 0xF]) * float(in0[1])
                      + float(w2lut_r2[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(w2lut_r2[(pp.x >> 12) & 0xF]) * float(in0[3])
                      + float(w2lut_r2[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(w2lut_r2[(pp.x >> 20) & 0xF]) * float(in1[1])
                      + float(w2lut_r2[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(w2lut_r2[(pp.x >> 28)       ]) * float(in1[3])
                      + float(w2lut_r2[ pp.y        & 0xF]) * float(in2[0]) + float(w2lut_r2[(pp.y >>  4) & 0xF]) * float(in2[1])
                      + float(w2lut_r2[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(w2lut_r2[(pp.y >> 12) & 0xF]) * float(in2[3])
                      + float(w2lut_r2[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(w2lut_r2[(pp.y >> 20) & 0xF]) * float(in3[1])
                      + float(w2lut_r2[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(w2lut_r2[(pp.y >> 28)       ]) * float(in3[3]); }
        }

        if (sgValidRows > 3) {
            // Row 3 — w1
            { uint2 pp = w1r3[j];
              w1acc3 += float(w1lut_r3[ pp.x        & 0xF]) * float(in0[0]) + float(w1lut_r3[(pp.x >>  4) & 0xF]) * float(in0[1])
                      + float(w1lut_r3[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(w1lut_r3[(pp.x >> 12) & 0xF]) * float(in0[3])
                      + float(w1lut_r3[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(w1lut_r3[(pp.x >> 20) & 0xF]) * float(in1[1])
                      + float(w1lut_r3[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(w1lut_r3[(pp.x >> 28)       ]) * float(in1[3])
                      + float(w1lut_r3[ pp.y        & 0xF]) * float(in2[0]) + float(w1lut_r3[(pp.y >>  4) & 0xF]) * float(in2[1])
                      + float(w1lut_r3[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(w1lut_r3[(pp.y >> 12) & 0xF]) * float(in2[3])
                      + float(w1lut_r3[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(w1lut_r3[(pp.y >> 20) & 0xF]) * float(in3[1])
                      + float(w1lut_r3[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(w1lut_r3[(pp.y >> 28)       ]) * float(in3[3]); }
            // Row 3 — w2
            { uint2 pp = w2r3[j];
              w2acc3 += float(w2lut_r3[ pp.x        & 0xF]) * float(in0[0]) + float(w2lut_r3[(pp.x >>  4) & 0xF]) * float(in0[1])
                      + float(w2lut_r3[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(w2lut_r3[(pp.x >> 12) & 0xF]) * float(in0[3])
                      + float(w2lut_r3[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(w2lut_r3[(pp.x >> 20) & 0xF]) * float(in1[1])
                      + float(w2lut_r3[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(w2lut_r3[(pp.x >> 28)       ]) * float(in1[3])
                      + float(w2lut_r3[ pp.y        & 0xF]) * float(in2[0]) + float(w2lut_r3[(pp.y >>  4) & 0xF]) * float(in2[1])
                      + float(w2lut_r3[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(w2lut_r3[(pp.y >> 12) & 0xF]) * float(in2[3])
                      + float(w2lut_r3[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(w2lut_r3[(pp.y >> 20) & 0xF]) * float(in3[1])
                      + float(w2lut_r3[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(w2lut_r3[(pp.y >> 28)       ]) * float(in3[3]); }
        }
    }

    // Tail for remainder not covered by uint2 chunks
    uint tailStart = eighthCols * 8;
    for (uint j = tailStart + simd_lane; j < halfCols; j += 32) {
        uint col = j * 2;
        half i0 = input[col];
        half i1 = input[col + 1];

        uint8_t w1p0 = w1_indices[sgBaseRow * halfCols + j];
        w1acc0 += float(w1lut_r0[w1p0 & 0xF]) * float(i0) + float(w1lut_r0[w1p0 >> 4]) * float(i1);
        uint8_t w2p0 = w2_indices[sgBaseRow * halfCols + j];
        w2acc0 += float(w2lut_r0[w2p0 & 0xF]) * float(i0) + float(w2lut_r0[w2p0 >> 4]) * float(i1);

        if (sgValidRows > 1) {
            uint8_t w1p1 = w1_indices[(sgBaseRow+1)*halfCols+j]; w1acc1 += float(w1lut_r1[w1p1&0xF])*float(i0)+float(w1lut_r1[w1p1>>4])*float(i1);
            uint8_t w2p1 = w2_indices[(sgBaseRow+1)*halfCols+j]; w2acc1 += float(w2lut_r1[w2p1&0xF])*float(i0)+float(w2lut_r1[w2p1>>4])*float(i1);
        }
        if (sgValidRows > 2) {
            uint8_t w1p2 = w1_indices[(sgBaseRow+2)*halfCols+j]; w1acc2 += float(w1lut_r2[w1p2&0xF])*float(i0)+float(w1lut_r2[w1p2>>4])*float(i1);
            uint8_t w2p2 = w2_indices[(sgBaseRow+2)*halfCols+j]; w2acc2 += float(w2lut_r2[w2p2&0xF])*float(i0)+float(w2lut_r2[w2p2>>4])*float(i1);
        }
        if (sgValidRows > 3) {
            uint8_t w1p3 = w1_indices[(sgBaseRow+3)*halfCols+j]; w1acc3 += float(w1lut_r3[w1p3&0xF])*float(i0)+float(w1lut_r3[w1p3>>4])*float(i1);
            uint8_t w2p3 = w2_indices[(sgBaseRow+3)*halfCols+j]; w2acc3 += float(w2lut_r3[w2p3&0xF])*float(i0)+float(w2lut_r3[w2p3>>4])*float(i1);
        }
    }

    // SIMD reduction on all 8 accumulators
    w1acc0 = simd_sum(w1acc0); w1acc1 = simd_sum(w1acc1);
    w1acc2 = simd_sum(w1acc2); w1acc3 = simd_sum(w1acc3);
    w2acc0 = simd_sum(w2acc0); w2acc1 = simd_sum(w2acc1);
    w2acc2 = simd_sum(w2acc2); w2acc3 = simd_sum(w2acc3);

    // Lane 0 writes both outputs
    if (simd_lane == 0) {
        output1[sgBaseRow] = half(w1acc0);
        output2[sgBaseRow] = half(w2acc0);
        if (sgValidRows > 1) { output1[sgBaseRow + 1] = half(w1acc1); output2[sgBaseRow + 1] = half(w2acc1); }
        if (sgValidRows > 2) { output1[sgBaseRow + 2] = half(w1acc2); output2[sgBaseRow + 2] = half(w2acc2); }
        if (sgValidRows > 3) { output1[sgBaseRow + 3] = half(w1acc3); output2[sgBaseRow + 3] = half(w2acc3); }
    }
}

// ─── Fused gate+up+SwiGLU: output[r] = silu(gate_dot(r)) * up_dot(r) ───
// Computes two LUT matvecs (gate_proj, up_proj) on the same input, applies
// SwiGLU activation, and writes the fused result. Eliminates 2 intermediate
// buffers and 2 dispatch transitions.
//
// Same thread structure as fused_lut_matvec: 64 threads, 2 SG, 4 rows/SG.
// Each SG processes 4 rows of BOTH weight matrices simultaneously.
// Input loaded once, reused for both projections.
// Function constants FC_COLS and FC_GROUP_SIZE used.
//
// Dispatch: ceil(R/8) threadgroups × 64 threads.

kernel void fused_gate_up_swiglu(
    device const uint8_t* gate_indices [[buffer(0)]],  // [R, C/2] gate weights
    device const half*    gate_lut     [[buffer(1)]],  // [nGroups, 16] gate LUT
    device const uint8_t* up_indices   [[buffer(2)]],  // [R, C/2] up weights
    device const half*    up_lut       [[buffer(3)]],  // [nGroups, 16] up LUT
    device const half*    input        [[buffer(4)]],  // [C]
    device half*          output       [[buffer(5)]],  // [R] SwiGLU result
    constant uint&        numRows      [[buffer(6)]],  // R
    uint tgid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint baseRow = tgid * ROWS_PER_TG;
    uint halfCols = FC_COLS / 2;
    uint validRows = min(ROWS_PER_TG, numRows - baseRow);

    // Each simdgroup owns 4 consecutive rows
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint sgValidRows = (validRows > simd_group * ROWS_PER_SG)
        ? min(ROWS_PER_SG, validRows - simd_group * ROWS_PER_SG)
        : 0u;

    // Load LUTs into threadgroup memory — gate and up, 2 LUTs each (for group boundary)
    uint group0 = baseRow / FC_GROUP_SIZE;
    uint groupLast = (baseRow + validRows - 1) / FC_GROUP_SIZE;

    threadgroup half g_lr0[16], g_lr1[16], u_lr0[16], u_lr1[16];
    if (tid < 16) {
        g_lr0[tid] = gate_lut[group0 * 16 + tid];
        if (group0 != groupLast) {
            g_lr1[tid] = gate_lut[groupLast * 16 + tid];
        }
    }
    if (tid >= 16 && tid < 32) {
        uint idx = tid - 16;
        u_lr0[idx] = up_lut[group0 * 16 + idx];
        if (group0 != groupLast) {
            u_lr1[idx] = up_lut[groupLast * 16 + idx];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgValidRows == 0) return;

    // Select gate LUT per row
    threadgroup const half* glut_r0 = (sgBaseRow / FC_GROUP_SIZE == group0) ? g_lr0 : g_lr1;
    threadgroup const half* glut_r1 = ((sgBaseRow + 1) / FC_GROUP_SIZE == group0) ? g_lr0 : g_lr1;
    threadgroup const half* glut_r2 = ((sgBaseRow + 2) / FC_GROUP_SIZE == group0) ? g_lr0 : g_lr1;
    threadgroup const half* glut_r3 = ((sgBaseRow + 3) / FC_GROUP_SIZE == group0) ? g_lr0 : g_lr1;

    // Select up LUT per row
    threadgroup const half* ulut_r0 = (sgBaseRow / FC_GROUP_SIZE == group0) ? u_lr0 : u_lr1;
    threadgroup const half* ulut_r1 = ((sgBaseRow + 1) / FC_GROUP_SIZE == group0) ? u_lr0 : u_lr1;
    threadgroup const half* ulut_r2 = ((sgBaseRow + 2) / FC_GROUP_SIZE == group0) ? u_lr0 : u_lr1;
    threadgroup const half* ulut_r3 = ((sgBaseRow + 3) / FC_GROUP_SIZE == group0) ? u_lr0 : u_lr1;

    // 8 accumulators: gate0-3 and up0-3
    float gacc0 = 0.0f, gacc1 = 0.0f, gacc2 = 0.0f, gacc3 = 0.0f;
    float uacc0 = 0.0f, uacc1 = 0.0f, uacc2 = 0.0f, uacc3 = 0.0f;

    // Row pointers for gate and up weight matrices
    uint eighthCols = halfCols / 8;
    device const uint2* gr0 = (device const uint2*)(gate_indices + sgBaseRow * halfCols);
    device const uint2* gr1 = (sgValidRows > 1) ? (device const uint2*)(gate_indices + (sgBaseRow + 1) * halfCols) : gr0;
    device const uint2* gr2 = (sgValidRows > 2) ? (device const uint2*)(gate_indices + (sgBaseRow + 2) * halfCols) : gr0;
    device const uint2* gr3 = (sgValidRows > 3) ? (device const uint2*)(gate_indices + (sgBaseRow + 3) * halfCols) : gr0;

    device const uint2* ur0 = (device const uint2*)(up_indices + sgBaseRow * halfCols);
    device const uint2* ur1 = (sgValidRows > 1) ? (device const uint2*)(up_indices + (sgBaseRow + 1) * halfCols) : ur0;
    device const uint2* ur2 = (sgValidRows > 2) ? (device const uint2*)(up_indices + (sgBaseRow + 2) * halfCols) : ur0;
    device const uint2* ur3 = (sgValidRows > 3) ? (device const uint2*)(up_indices + (sgBaseRow + 3) * halfCols) : ur0;

    // Inner loop: load input ONCE, load gate AND up weights, accumulate both
    for (uint j = simd_lane; j < eighthCols; j += 32) {
        uint col = j * 16;

        // Load 16 input values (four half4 loads) — shared by both projections
        half4 in0 = *(device const half4*)(input + col);
        half4 in1 = *(device const half4*)(input + col + 4);
        half4 in2 = *(device const half4*)(input + col + 8);
        half4 in3 = *(device const half4*)(input + col + 12);

        // Row 0 (always valid) — gate
        { uint2 pp = gr0[j];
          gacc0 += float(glut_r0[ pp.x        & 0xF]) * float(in0[0]) + float(glut_r0[(pp.x >>  4) & 0xF]) * float(in0[1])
                 + float(glut_r0[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(glut_r0[(pp.x >> 12) & 0xF]) * float(in0[3])
                 + float(glut_r0[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(glut_r0[(pp.x >> 20) & 0xF]) * float(in1[1])
                 + float(glut_r0[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(glut_r0[(pp.x >> 28)       ]) * float(in1[3])
                 + float(glut_r0[ pp.y        & 0xF]) * float(in2[0]) + float(glut_r0[(pp.y >>  4) & 0xF]) * float(in2[1])
                 + float(glut_r0[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(glut_r0[(pp.y >> 12) & 0xF]) * float(in2[3])
                 + float(glut_r0[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(glut_r0[(pp.y >> 20) & 0xF]) * float(in3[1])
                 + float(glut_r0[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(glut_r0[(pp.y >> 28)       ]) * float(in3[3]); }
        // Row 0 — up
        { uint2 pp = ur0[j];
          uacc0 += float(ulut_r0[ pp.x        & 0xF]) * float(in0[0]) + float(ulut_r0[(pp.x >>  4) & 0xF]) * float(in0[1])
                 + float(ulut_r0[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(ulut_r0[(pp.x >> 12) & 0xF]) * float(in0[3])
                 + float(ulut_r0[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(ulut_r0[(pp.x >> 20) & 0xF]) * float(in1[1])
                 + float(ulut_r0[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(ulut_r0[(pp.x >> 28)       ]) * float(in1[3])
                 + float(ulut_r0[ pp.y        & 0xF]) * float(in2[0]) + float(ulut_r0[(pp.y >>  4) & 0xF]) * float(in2[1])
                 + float(ulut_r0[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(ulut_r0[(pp.y >> 12) & 0xF]) * float(in2[3])
                 + float(ulut_r0[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(ulut_r0[(pp.y >> 20) & 0xF]) * float(in3[1])
                 + float(ulut_r0[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(ulut_r0[(pp.y >> 28)       ]) * float(in3[3]); }

        if (sgValidRows > 1) {
            // Row 1 — gate
            { uint2 pp = gr1[j];
              gacc1 += float(glut_r1[ pp.x        & 0xF]) * float(in0[0]) + float(glut_r1[(pp.x >>  4) & 0xF]) * float(in0[1])
                     + float(glut_r1[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(glut_r1[(pp.x >> 12) & 0xF]) * float(in0[3])
                     + float(glut_r1[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(glut_r1[(pp.x >> 20) & 0xF]) * float(in1[1])
                     + float(glut_r1[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(glut_r1[(pp.x >> 28)       ]) * float(in1[3])
                     + float(glut_r1[ pp.y        & 0xF]) * float(in2[0]) + float(glut_r1[(pp.y >>  4) & 0xF]) * float(in2[1])
                     + float(glut_r1[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(glut_r1[(pp.y >> 12) & 0xF]) * float(in2[3])
                     + float(glut_r1[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(glut_r1[(pp.y >> 20) & 0xF]) * float(in3[1])
                     + float(glut_r1[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(glut_r1[(pp.y >> 28)       ]) * float(in3[3]); }
            // Row 1 — up
            { uint2 pp = ur1[j];
              uacc1 += float(ulut_r1[ pp.x        & 0xF]) * float(in0[0]) + float(ulut_r1[(pp.x >>  4) & 0xF]) * float(in0[1])
                     + float(ulut_r1[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(ulut_r1[(pp.x >> 12) & 0xF]) * float(in0[3])
                     + float(ulut_r1[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(ulut_r1[(pp.x >> 20) & 0xF]) * float(in1[1])
                     + float(ulut_r1[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(ulut_r1[(pp.x >> 28)       ]) * float(in1[3])
                     + float(ulut_r1[ pp.y        & 0xF]) * float(in2[0]) + float(ulut_r1[(pp.y >>  4) & 0xF]) * float(in2[1])
                     + float(ulut_r1[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(ulut_r1[(pp.y >> 12) & 0xF]) * float(in2[3])
                     + float(ulut_r1[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(ulut_r1[(pp.y >> 20) & 0xF]) * float(in3[1])
                     + float(ulut_r1[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(ulut_r1[(pp.y >> 28)       ]) * float(in3[3]); }
        }

        if (sgValidRows > 2) {
            // Row 2 — gate
            { uint2 pp = gr2[j];
              gacc2 += float(glut_r2[ pp.x        & 0xF]) * float(in0[0]) + float(glut_r2[(pp.x >>  4) & 0xF]) * float(in0[1])
                     + float(glut_r2[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(glut_r2[(pp.x >> 12) & 0xF]) * float(in0[3])
                     + float(glut_r2[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(glut_r2[(pp.x >> 20) & 0xF]) * float(in1[1])
                     + float(glut_r2[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(glut_r2[(pp.x >> 28)       ]) * float(in1[3])
                     + float(glut_r2[ pp.y        & 0xF]) * float(in2[0]) + float(glut_r2[(pp.y >>  4) & 0xF]) * float(in2[1])
                     + float(glut_r2[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(glut_r2[(pp.y >> 12) & 0xF]) * float(in2[3])
                     + float(glut_r2[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(glut_r2[(pp.y >> 20) & 0xF]) * float(in3[1])
                     + float(glut_r2[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(glut_r2[(pp.y >> 28)       ]) * float(in3[3]); }
            // Row 2 — up
            { uint2 pp = ur2[j];
              uacc2 += float(ulut_r2[ pp.x        & 0xF]) * float(in0[0]) + float(ulut_r2[(pp.x >>  4) & 0xF]) * float(in0[1])
                     + float(ulut_r2[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(ulut_r2[(pp.x >> 12) & 0xF]) * float(in0[3])
                     + float(ulut_r2[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(ulut_r2[(pp.x >> 20) & 0xF]) * float(in1[1])
                     + float(ulut_r2[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(ulut_r2[(pp.x >> 28)       ]) * float(in1[3])
                     + float(ulut_r2[ pp.y        & 0xF]) * float(in2[0]) + float(ulut_r2[(pp.y >>  4) & 0xF]) * float(in2[1])
                     + float(ulut_r2[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(ulut_r2[(pp.y >> 12) & 0xF]) * float(in2[3])
                     + float(ulut_r2[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(ulut_r2[(pp.y >> 20) & 0xF]) * float(in3[1])
                     + float(ulut_r2[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(ulut_r2[(pp.y >> 28)       ]) * float(in3[3]); }
        }

        if (sgValidRows > 3) {
            // Row 3 — gate
            { uint2 pp = gr3[j];
              gacc3 += float(glut_r3[ pp.x        & 0xF]) * float(in0[0]) + float(glut_r3[(pp.x >>  4) & 0xF]) * float(in0[1])
                     + float(glut_r3[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(glut_r3[(pp.x >> 12) & 0xF]) * float(in0[3])
                     + float(glut_r3[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(glut_r3[(pp.x >> 20) & 0xF]) * float(in1[1])
                     + float(glut_r3[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(glut_r3[(pp.x >> 28)       ]) * float(in1[3])
                     + float(glut_r3[ pp.y        & 0xF]) * float(in2[0]) + float(glut_r3[(pp.y >>  4) & 0xF]) * float(in2[1])
                     + float(glut_r3[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(glut_r3[(pp.y >> 12) & 0xF]) * float(in2[3])
                     + float(glut_r3[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(glut_r3[(pp.y >> 20) & 0xF]) * float(in3[1])
                     + float(glut_r3[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(glut_r3[(pp.y >> 28)       ]) * float(in3[3]); }
            // Row 3 — up
            { uint2 pp = ur3[j];
              uacc3 += float(ulut_r3[ pp.x        & 0xF]) * float(in0[0]) + float(ulut_r3[(pp.x >>  4) & 0xF]) * float(in0[1])
                     + float(ulut_r3[(pp.x >>  8) & 0xF]) * float(in0[2]) + float(ulut_r3[(pp.x >> 12) & 0xF]) * float(in0[3])
                     + float(ulut_r3[(pp.x >> 16) & 0xF]) * float(in1[0]) + float(ulut_r3[(pp.x >> 20) & 0xF]) * float(in1[1])
                     + float(ulut_r3[(pp.x >> 24) & 0xF]) * float(in1[2]) + float(ulut_r3[(pp.x >> 28)       ]) * float(in1[3])
                     + float(ulut_r3[ pp.y        & 0xF]) * float(in2[0]) + float(ulut_r3[(pp.y >>  4) & 0xF]) * float(in2[1])
                     + float(ulut_r3[(pp.y >>  8) & 0xF]) * float(in2[2]) + float(ulut_r3[(pp.y >> 12) & 0xF]) * float(in2[3])
                     + float(ulut_r3[(pp.y >> 16) & 0xF]) * float(in3[0]) + float(ulut_r3[(pp.y >> 20) & 0xF]) * float(in3[1])
                     + float(ulut_r3[(pp.y >> 24) & 0xF]) * float(in3[2]) + float(ulut_r3[(pp.y >> 28)       ]) * float(in3[3]); }
        }
    }

    // Tail for remainder not covered by uint2 chunks
    uint tailStart = eighthCols * 8;
    for (uint j = tailStart + simd_lane; j < halfCols; j += 32) {
        uint col = j * 2;
        half i0 = input[col];
        half i1 = input[col + 1];

        uint8_t gp0 = gate_indices[sgBaseRow * halfCols + j];
        gacc0 += float(glut_r0[gp0 & 0xF]) * float(i0) + float(glut_r0[gp0 >> 4]) * float(i1);
        uint8_t up0 = up_indices[sgBaseRow * halfCols + j];
        uacc0 += float(ulut_r0[up0 & 0xF]) * float(i0) + float(ulut_r0[up0 >> 4]) * float(i1);

        if (sgValidRows > 1) {
            uint8_t gp1 = gate_indices[(sgBaseRow+1)*halfCols+j]; gacc1 += float(glut_r1[gp1&0xF])*float(i0)+float(glut_r1[gp1>>4])*float(i1);
            uint8_t up1 = up_indices[(sgBaseRow+1)*halfCols+j];   uacc1 += float(ulut_r1[up1&0xF])*float(i0)+float(ulut_r1[up1>>4])*float(i1);
        }
        if (sgValidRows > 2) {
            uint8_t gp2 = gate_indices[(sgBaseRow+2)*halfCols+j]; gacc2 += float(glut_r2[gp2&0xF])*float(i0)+float(glut_r2[gp2>>4])*float(i1);
            uint8_t up2 = up_indices[(sgBaseRow+2)*halfCols+j];   uacc2 += float(ulut_r2[up2&0xF])*float(i0)+float(ulut_r2[up2>>4])*float(i1);
        }
        if (sgValidRows > 3) {
            uint8_t gp3 = gate_indices[(sgBaseRow+3)*halfCols+j]; gacc3 += float(glut_r3[gp3&0xF])*float(i0)+float(glut_r3[gp3>>4])*float(i1);
            uint8_t up3 = up_indices[(sgBaseRow+3)*halfCols+j];   uacc3 += float(ulut_r3[up3&0xF])*float(i0)+float(ulut_r3[up3>>4])*float(i1);
        }
    }

    // SIMD reduction on all 8 accumulators
    gacc0 = simd_sum(gacc0); gacc1 = simd_sum(gacc1);
    gacc2 = simd_sum(gacc2); gacc3 = simd_sum(gacc3);
    uacc0 = simd_sum(uacc0); uacc1 = simd_sum(uacc1);
    uacc2 = simd_sum(uacc2); uacc3 = simd_sum(uacc3);

    // Lane 0 applies SwiGLU: silu(gate) * up, where silu(x) = x / (1 + exp(-x))
    if (simd_lane == 0) {
        { float g = gacc0; float s = g / (1.0f + exp(-g)); output[sgBaseRow] = half(s * uacc0); }
        if (sgValidRows > 1) { float g = gacc1; float s = g / (1.0f + exp(-g)); output[sgBaseRow + 1] = half(s * uacc1); }
        if (sgValidRows > 2) { float g = gacc2; float s = g / (1.0f + exp(-g)); output[sgBaseRow + 2] = half(s * uacc2); }
        if (sgValidRows > 3) { float g = gacc3; float s = g / (1.0f + exp(-g)); output[sgBaseRow + 3] = half(s * uacc3); }
    }
}

// ─── FP16 dense matrix-vector multiply ───
// Weight matrix [R, C] stored as FP16. Input [C], Output [R].
// One threadgroup per output row, SIMD reduction.
// Used for tied LM head (embed_tokens weight reused for logits).

kernel void fp16_matvec(
    device const half* weight   [[buffer(0)]],  // [R, C]
    device const half* input    [[buffer(1)]],  // [C]
    device half*       output   [[buffer(2)]],  // [R]
    constant uint&     cols     [[buffer(3)]],  // C
    uint tgid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid;
    device const half* rowPtr = weight + row * cols;

    // Vectorized dot product — half4 loads for 4× memory throughput
    float acc = 0.0f;
    uint cols4 = cols / 4;
    device const half4* rowPtr4 = (device const half4*)rowPtr;
    device const half4* input4  = (device const half4*)input;
    for (uint j = tid; j < cols4; j += MATVEC_TPG) {
        acc += dot(float4(rowPtr4[j]), float4(input4[j]));
    }
    // Scalar tail for non-4-aligned remainder
    for (uint j = cols4 * 4 + tid; j < cols; j += MATVEC_TPG) {
        acc += float(rowPtr[j]) * float(input[j]);
    }

    acc = simd_sum(acc);
    threadgroup float partial[8];
    if (simd_lane == 0) { partial[simd_group] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0;
        for (uint s = 0; s < MATVEC_TPG / 32; s++) { total += partial[s]; }
        output[row] = half(clamp(total, -65504.0f, 65504.0f));
    }
}

kernel void fp16_matvec_fp32_out(
    device const half* weight   [[buffer(0)]],  // [R, C]
    device const half* input    [[buffer(1)]],  // [C]
    device float*      output   [[buffer(2)]],  // [R]
    constant uint&     cols     [[buffer(3)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid;
    device const half* rowPtr = weight + row * cols;

    float acc = 0.0f;
    uint cols4 = cols / 4;
    device const half4* rowPtr4 = (device const half4*)rowPtr;
    device const half4* input4  = (device const half4*)input;
    for (uint j = tid; j < cols4; j += MATVEC_TPG) {
        acc += dot(float4(rowPtr4[j]), float4(input4[j]));
    }
    for (uint j = cols4 * 4 + tid; j < cols; j += MATVEC_TPG) {
        acc += float(rowPtr[j]) * float(input[j]);
    }

    acc = simd_sum(acc);
    threadgroup float partial[8];
    if (simd_lane == 0) { partial[simd_group] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0;
        for (uint s = 0; s < MATVEC_TPG / 32; s++) { total += partial[s]; }
        output[row] = total;
    }
}

// ─── FP16-activation dense matvec, bf16 / fp32 WEIGHTS ───
// The fp16_matvec SHAPE (fp16 input + output, one threadgroup per output row, SIMD + threadgroup
// reduction, half clamp) with a bf16 / fp32 weight matrix instead of fp16 — so a bf16 (the
// checkpoint's native dtype) or fp32 projection weight is a kernel LEGO in the fp16-activation LLM
// path, not a hole the gateway throws on (U2 of docs/dtype-building-blocks-plan.md). Only the
// weight load differs from fp16_matvec: bf16→float and fp32 are EXACT/widening, accumulate in
// float — never an instantiation of the fp32-ACTIVATION gemv_*_f32 shape (those take fp32 input).

kernel void fp16_matvec_bf16w(
    device const bfloat* weight   [[buffer(0)]],  // [R, C] bf16
    device const half*   input    [[buffer(1)]],  // [C]
    device half*         output   [[buffer(2)]],  // [R]
    constant uint&       cols     [[buffer(3)]],  // C
    uint tgid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid;
    device const bfloat* rowPtr = weight + row * cols;

    float acc = 0.0f;
    uint cols4 = cols / 4;
    device const bfloat4* rowPtr4 = (device const bfloat4*)rowPtr;
    device const half4*   input4  = (device const half4*)input;
    for (uint j = tid; j < cols4; j += MATVEC_TPG) {
        acc += dot(float4(rowPtr4[j]), float4(input4[j]));
    }
    for (uint j = cols4 * 4 + tid; j < cols; j += MATVEC_TPG) {
        acc += float(rowPtr[j]) * float(input[j]);
    }

    acc = simd_sum(acc);
    threadgroup float partial[8];
    if (simd_lane == 0) { partial[simd_group] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0;
        for (uint s = 0; s < MATVEC_TPG / 32; s++) { total += partial[s]; }
        output[row] = half(clamp(total, -65504.0f, 65504.0f));
    }
}

kernel void fp16_matvec_fp32w(
    device const float* weight   [[buffer(0)]],  // [R, C] fp32
    device const half*  input    [[buffer(1)]],  // [C]
    device half*        output   [[buffer(2)]],  // [R]
    constant uint&      cols     [[buffer(3)]],  // C
    uint tgid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid;
    device const float* rowPtr = weight + row * cols;

    float acc = 0.0f;
    uint cols4 = cols / 4;
    device const float4* rowPtr4 = (device const float4*)rowPtr;
    device const half4*  input4  = (device const half4*)input;
    for (uint j = tid; j < cols4; j += MATVEC_TPG) {
        acc += dot(rowPtr4[j], float4(input4[j]));
    }
    for (uint j = cols4 * 4 + tid; j < cols; j += MATVEC_TPG) {
        acc += rowPtr[j] * float(input[j]);
    }

    acc = simd_sum(acc);
    threadgroup float partial[8];
    if (simd_lane == 0) { partial[simd_group] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0;
        for (uint s = 0; s < MATVEC_TPG / 32; s++) { total += partial[s]; }
        output[row] = half(clamp(total, -65504.0f, 65504.0f));
    }
}

// ─── FP16 embedding gather ───
// Fetches one row from the embedding table. Dispatch with width = hidden_dim.
kernel void embedding_gather(
    device const half* table  [[buffer(0)]],  // [vocab, hidden]
    device const int*  index  [[buffer(1)]],  // [1] token id
    device half*       output [[buffer(2)]],  // [hidden]
    constant uint&     hidden [[buffer(3)]],  // hidden dim
    uint tid [[thread_position_in_grid]]
) {
    if (tid < hidden) {
        uint row = uint(index[0]);
        output[tid] = table[row * hidden + tid];
    }
}

// ─── Quantized embedding gather (u4+LUT) ───
// Same layout as fused_lut_matvec weights. Each thread dequants 2 elements.
// Dispatch with width = hidden_dim / 2.
kernel void lut_embedding_gather(
    device const uint8_t* indices   [[buffer(0)]],  // [vocab, hidden/2] packed u4
    device const half*    lut       [[buffer(1)]],  // [nGroups, 16] FP16
    device const int*     tokenId   [[buffer(2)]],  // [1]
    device half*          output    [[buffer(3)]],  // [hidden]
    constant uint&        cols      [[buffer(4)]],  // hidden dim
    constant uint&        groupSize [[buffer(5)]],  // rows per group (16)
    uint tid [[thread_position_in_grid]]
) {
    uint halfCols = cols / 2;
    if (tid >= halfCols) return;

    uint row = uint(tokenId[0]);
    uint group = row / groupSize;
    device const half* groupLUT = lut + group * 16;

    uint byteIdx = row * halfCols + tid;
    uint8_t packed = indices[byteIdx];

    output[tid * 2]     = groupLUT[packed & 0xF];
    output[tid * 2 + 1] = groupLUT[packed >> 4];
}

// ─── Quantized embedding gather (u4+affine) ───
// Same packed layout as affine_matvec weights:
//   dequant = nibble * scale + bias, with groups along columns.
// Dispatch with width = hidden_dim / 2.
kernel void affine_embedding_gather(
    device const uint8_t* weights   [[buffer(0)]],  // [vocab, hidden/2] packed u4
    device const half*    scales    [[buffer(1)]],  // [vocab, hidden/groupSize]
    device const half*    biases    [[buffer(2)]],  // [vocab, hidden/groupSize]
    device const int*     tokenId   [[buffer(3)]],  // [1]
    device half*          output    [[buffer(4)]],  // [hidden]
    constant uint&        cols      [[buffer(5)]],  // hidden dim
    constant uint&        groupSize [[buffer(6)]],  // columns per affine group
    uint tid [[thread_position_in_grid]]
) {
    uint halfCols = cols / 2;
    if (tid >= halfCols) return;

    uint row = uint(tokenId[0]);
    uint numColGroups = cols / groupSize;
    uint byteIdx = row * halfCols + tid;
    uint8_t packed = weights[byteIdx];

    uint col0 = tid * 2;
    uint col1 = col0 + 1;
    uint group0 = col0 / groupSize;
    uint group1 = col1 / groupSize;
    uint sbBase = row * numColGroups;

    float scale0 = float(scales[sbBase + group0]);
    float bias0 = float(biases[sbBase + group0]);
    float scale1 = float(scales[sbBase + group1]);
    float bias1 = float(biases[sbBase + group1]);

    output[col0] = half(float(packed & 0xF) * scale0 + bias0);
    output[col1] = half(float(packed >> 4) * scale1 + bias1);
}

// ─── TurboQuant-H embedding gather ───
// Dispatch: 1D grid (num_groups * G, 1, 1). Each thread
// independently computes one output column.
//
// For output column pos (group g = pos/G, posInGroup p = pos%G):
//   output[pos] = (1/√G) * Σ_{j=0..G-1} H[p,j] * centroid[code[g,j]]
//
// H is the normalized Sylvester Hadamard; H[p,j]'s sign is
// determined by popcount(p & j) — even → +1, odd → -1. Per
// thread does O(G) lookups + signed adds (G=128 → ~128 ops),
// no threadgroup memory needed. The independent-per-thread shape
// keeps the dispatcher unconstrained on threadgroup size and
// makes partial-final-group handling a single pos<cols write
// guard.
//
// Codes layout: packed 4-per-byte from LSB,
//   codes[row * codesPerRow + (pos / 4)] bits (pos%4)*2 .. (pos%4)*2+1
// Codebook layout: [num_groups, 4] fp16 row-major.
kernel void tqh_embedding_gather(
    device const uint8_t* codes        [[buffer(0)]],
    device const half*    codebook     [[buffer(1)]],
    device const int*     tokenId      [[buffer(2)]],
    device half*          output       [[buffer(3)]],
    constant uint&        codesPerRow  [[buffer(4)]],
    constant uint&        cols         [[buffer(5)]],
    uint                  gid          [[thread_position_in_grid]]
) {
    constexpr uint G = 128;
    // 1.0f / sqrt(128.0f); sqrt isn't constexpr in MSL, so literal.
    constexpr float inv_sqrt_G = 0.0883883476483184405f;

    uint group = gid / G;
    uint posInGroup = gid % G;
    uint row = uint(tokenId[0]);
    uint rowBase = row * codesPerRow + group * (G / 4);
    uint cbBase = group * 4;

    float acc = 0.0f;
    for (uint j = 0; j < G; ++j) {
        uint byteIdx = rowBase + j / 4;
        uint shift = (j & 3) * 2;
        uint code = uint((codes[byteIdx] >> shift) & 0x3);
        float c = float(codebook[cbBase + code]);
        uint parity = popcount(posInGroup & j) & 1;
        acc += (parity != 0) ? -c : c;
    }

    if (gid < cols) {
        output[gid] = half(acc * inv_sqrt_G);
    }
}

// ─── Batched quantized embedding gather (u4+affine, prefill) ───
// Dispatch: 2D grid (hidden/2, B).
kernel void affine_embedding_gather_batched(
    device const uint8_t* weights   [[buffer(0)]],  // [vocab, hidden/2] packed u4
    device const half*    scales    [[buffer(1)]],  // [vocab, hidden/groupSize]
    device const half*    biases    [[buffer(2)]],  // [vocab, hidden/groupSize]
    device const int*     tokenIds  [[buffer(3)]],  // [B]
    device half*          output    [[buffer(4)]],  // [B, hidden] row-major
    constant uint&        cols      [[buffer(5)]],  // hidden dim
    constant uint&        batchSize [[buffer(6)]],  // actual batch
    constant uint&        groupSize [[buffer(7)]],  // columns per affine group
    uint2 tid [[thread_position_in_grid]]
) {
    uint halfCols = cols / 2;
    uint halfCol = tid.x;
    uint batch = tid.y;
    if (halfCol >= halfCols || batch >= batchSize) return;

    uint row = uint(tokenIds[batch]);
    uint numColGroups = cols / groupSize;
    uint byteIdx = row * halfCols + halfCol;
    uint8_t packed = weights[byteIdx];

    uint col0 = halfCol * 2;
    uint col1 = col0 + 1;
    uint group0 = col0 / groupSize;
    uint group1 = col1 / groupSize;
    uint sbBase = row * numColGroups;
    uint outBase = batch * cols;

    float scale0 = float(scales[sbBase + group0]);
    float bias0 = float(biases[sbBase + group0]);
    float scale1 = float(scales[sbBase + group1]);
    float bias1 = float(biases[sbBase + group1]);

    output[outBase + col0] = half(float(packed & 0xF) * scale0 + bias0);
    output[outBase + col1] = half(float(packed >> 4) * scale1 + bias1);
}

// ─── Batched FP16 embedding gather (prefill) ───
// Fetches B rows from the embedding table at once.
// Dispatch: 2D grid (hidden, B).
kernel void embedding_gather_batched(
    device const half* table   [[buffer(0)]],  // [vocab, hidden]
    device const int*  indices [[buffer(1)]],  // [B] token ids
    device half*       output  [[buffer(2)]],  // [B, hidden] row-major
    constant uint&     hidden  [[buffer(3)]],
    constant uint&     batchSize [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]       // (dim, batch)
) {
    uint dim = tid.x;
    uint batch = tid.y;
    if (dim >= hidden || batch >= batchSize) return;
    uint row = uint(indices[batch]);
    output[batch * hidden + dim] = table[row * hidden + dim];
}

// ─── Affine u4 matrix-vector multiply ───
// Weight matrix [R, C] stored as packed u4 (2 values per byte).
// Groups are along COLUMNS: each group of `groupSize` consecutive columns
// shares a per-row scale+bias. value = scale * nibble + bias.
//
// 2 simdgroups × 4 rows each = 8 output rows per threadgroup.
// simd_sum reduction only — no threadgroup barriers for reduction.
// MLX pre-scaling trick: mask nibbles without shifting, pre-divide input.
//
// Dispatch: ceil(R/8) threadgroups × 64 threads.

// ─── Affine u4 matrix-vector multiply ───
// Same structure as fused_lut_matvec but with affine dequantization:
//   dequant = raw_nibble * scale + bias (per group)
// Uses FC_COLS/FC_GROUP_SIZE function constants for compiler specialization.
// uint2 reads (16 nibbles per load) matching the LUT kernel's memory pattern.
//
// Math: result[r] = sum_g [ scale[r,g] * dot(nibbles_g, x_g) + bias[r,g] * sum(x_g) ]
// Since scale/bias are constant within a group, we can apply per-chunk and accumulate.
// Dispatch: ceil(R/8) threadgroups × 64 threads.

kernel void affine_matvec(
    device const uint8_t* weights  [[buffer(0)]],  // [R, C/2] packed u4
    device const half*    scales   [[buffer(1)]],  // [R, C/FC_GROUP_SIZE]
    device const half*    biases   [[buffer(2)]],  // [R, C/FC_GROUP_SIZE]
    device const half*    input    [[buffer(3)]],  // [C]
    device half*          output   [[buffer(4)]],  // [R]
    constant uint&        numRows  [[buffer(5)]],  // R
    uint tgid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint baseRow = tgid * ROWS_PER_TG;
    uint halfCols = FC_COLS / 2;
    uint validRows = min(ROWS_PER_TG, numRows - baseRow);

    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint sgValidRows = (validRows > simd_group * ROWS_PER_SG)
        ? min(ROWS_PER_SG, validRows - simd_group * ROWS_PER_SG)
        : 0u;

    if (sgValidRows == 0) return;

    uint numColGroups = FC_COLS / FC_GROUP_SIZE;
    uint eighthCols = halfCols / 8;  // iteration count (16 nibbles per step)
    uint u16PerRow = halfCols / 2;   // uint16 elements per row

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // Row pointers as uint16_t — zero shifts needed (MLX pattern)
    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * halfCols);
    device const uint16_t* r1 = (sgValidRows > 1) ? r0 + u16PerRow : r0;
    device const uint16_t* r2 = (sgValidRows > 2) ? r0 + 2 * u16PerRow : r0;
    device const uint16_t* r3 = (sgValidRows > 3) ? r0 + 3 * u16PerRow : r0;

    // Scale/bias row pointers
    device const half* sc0 = scales + sgBaseRow * numColGroups;
    device const half* sc1 = sc0 + numColGroups;
    device const half* sc2 = sc0 + 2 * numColGroups;
    device const half* sc3 = sc0 + 3 * numColGroups;
    device const half* bi0 = biases + sgBaseRow * numColGroups;
    device const half* bi1 = bi0 + numColGroups;
    device const half* bi2 = bi0 + 2 * numColGroups;
    device const half* bi3 = bi0 + 3 * numColGroups;

    // 32 threads stride across columns. Pre-divide trick: zero shifts in inner loop.
    for (uint j = simd_lane; j < eighthCols; j += 32) {
        uint col = j * 16;
        uint g = col / FC_GROUP_SIZE;

        // Load 16 input values
        device const half* xp = input + col;
        float x0  = float(xp[0]);
        float x1  = float(xp[1]);
        float x2  = float(xp[2]);
        float x3  = float(xp[3]);
        float x4  = float(xp[4]);
        float x5  = float(xp[5]);
        float x6  = float(xp[6]);
        float x7  = float(xp[7]);
        float x8  = float(xp[8]);
        float x9  = float(xp[9]);
        float x10 = float(xp[10]);
        float x11 = float(xp[11]);
        float x12 = float(xp[12]);
        float x13 = float(xp[13]);
        float x14 = float(xp[14]);
        float x15 = float(xp[15]);

        // Bias term: sum of original (non-pre-scaled) input values
        float xsum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
                   + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15;

        // Pre-divide: every 4th value scaled by 1/16^k so masks work without shifts.
        // x_pre[1] * (nibble << 4) = (x/16) * (nibble*16) = x * nibble
        x1  *= (1.0f / 16.0f);
        x2  *= (1.0f / 256.0f);
        x3  *= (1.0f / 4096.0f);
        x5  *= (1.0f / 16.0f);
        x6  *= (1.0f / 256.0f);
        x7  *= (1.0f / 4096.0f);
        x9  *= (1.0f / 16.0f);
        x10 *= (1.0f / 256.0f);
        x11 *= (1.0f / 4096.0f);
        x13 *= (1.0f / 16.0f);
        x14 *= (1.0f / 256.0f);
        x15 *= (1.0f / 4096.0f);

// Macro: dot product of 16 pre-scaled x values with uint2 weight, mask-only (no shifts).
// Reads uint2 as 4 × uint16, masks each nibble in-position.
// Zero-shift affine dot: reads 4 x uint16_t (matching MLX qdot pattern).
// Each uint16 contains 4 nibbles masked in-position. No shifts needed.
#define AFFINE_DOT16(w) \
    ( x0  * float((w)[0] & 0x000Fu) + x1  * float((w)[0] & 0x00F0u) \
    + x2  * float((w)[0] & 0x0F00u) + x3  * float((w)[0] & 0xF000u) \
    + x4  * float((w)[1] & 0x000Fu) + x5  * float((w)[1] & 0x00F0u) \
    + x6  * float((w)[1] & 0x0F00u) + x7  * float((w)[1] & 0xF000u) \
    + x8  * float((w)[2] & 0x000Fu) + x9  * float((w)[2] & 0x00F0u) \
    + x10 * float((w)[2] & 0x0F00u) + x11 * float((w)[2] & 0xF000u) \
    + x12 * float((w)[3] & 0x000Fu) + x13 * float((w)[3] & 0x00F0u) \
    + x14 * float((w)[3] & 0x0F00u) + x15 * float((w)[3] & 0xF000u) )

        // Row 0 (always valid) — reads 4 × uint16_t via pointer (no shifts)
        acc0 += float(sc0[g]) * AFFINE_DOT16(r0 + j * 4) + float(bi0[g]) * xsum;

        if (sgValidRows > 1)
        acc1 += float(sc1[g]) * AFFINE_DOT16(r1 + j * 4) + float(bi1[g]) * xsum;

        if (sgValidRows > 2)
        acc2 += float(sc2[g]) * AFFINE_DOT16(r2 + j * 4) + float(bi2[g]) * xsum;

        if (sgValidRows > 3)
        acc3 += float(sc3[g]) * AFFINE_DOT16(r3 + j * 4) + float(bi3[g]) * xsum;

#undef AFFINE_DOT16
    }

    // Single SIMD reduction
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    // Lane 0 writes results
    if (simd_lane == 0) {
        output[sgBaseRow] = half(acc0);
        if (sgValidRows > 1) output[sgBaseRow + 1] = half(acc1);
        if (sgValidRows > 2) output[sgBaseRow + 2] = half(acc2);
        if (sgValidRows > 3) output[sgBaseRow + 3] = half(acc3);
    }
}

// ─── Norm-scale affine matvec ───
// Reads a pre-computed RMS scale and normalizes input inline:
//   x_norm = input * scale * (1 + norm_weight)
// Optionally writes the normalized vector as a side effect for later consumers.

kernel void norm_scale_affine_matvec(
    device const float*    scalePtr   [[buffer(0)]],  // [1]
    device const half*     normInput  [[buffer(1)]],  // [C]
    device const half*     normWeight [[buffer(2)]],  // [C]
    device half*           normOutput [[buffer(3)]],  // [C]
    device const uint8_t*  weights    [[buffer(4)]],  // [R, C/2] packed u4
    device const half*     scales     [[buffer(5)]],  // [R, C/FC_GROUP_SIZE]
    device const half*     biases     [[buffer(6)]],  // [R, C/FC_GROUP_SIZE]
    device half*           output     [[buffer(7)]],  // [R]
    constant uint&         numRows    [[buffer(8)]],  // R
    uint tgid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint baseRow = tgid * ROWS_PER_TG;
    uint halfCols = FC_COLS / 2;
    uint validRows = min(ROWS_PER_TG, numRows - baseRow);

    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint sgValidRows = (validRows > simd_group * ROWS_PER_SG)
        ? min(ROWS_PER_SG, validRows - simd_group * ROWS_PER_SG)
        : 0u;

    if (sgValidRows == 0) return;

    float rs = scalePtr[0];
    uint numColGroups = FC_COLS / FC_GROUP_SIZE;
    uint eighthCols = halfCols / 8;
    uint u16PerRow = halfCols / 2;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * halfCols);
    device const uint16_t* r1 = (sgValidRows > 1) ? r0 + u16PerRow : r0;
    device const uint16_t* r2 = (sgValidRows > 2) ? r0 + 2 * u16PerRow : r0;
    device const uint16_t* r3 = (sgValidRows > 3) ? r0 + 3 * u16PerRow : r0;

    device const half* sc0 = scales + sgBaseRow * numColGroups;
    device const half* sc1 = sc0 + numColGroups;
    device const half* sc2 = sc0 + 2 * numColGroups;
    device const half* sc3 = sc0 + 3 * numColGroups;
    device const half* bi0 = biases + sgBaseRow * numColGroups;
    device const half* bi1 = bi0 + numColGroups;
    device const half* bi2 = bi0 + 2 * numColGroups;
    device const half* bi3 = bi0 + 3 * numColGroups;

    for (uint j = simd_lane; j < eighthCols; j += 32) {
        uint col = j * 16;
        uint g = col / FC_GROUP_SIZE;

        device const half* ip = normInput + col;
        device const half* wp = normWeight + col;

        float x0  = float(ip[0])  * rs * (1.0f + float(wp[0]));
        float x1  = float(ip[1])  * rs * (1.0f + float(wp[1]));
        float x2  = float(ip[2])  * rs * (1.0f + float(wp[2]));
        float x3  = float(ip[3])  * rs * (1.0f + float(wp[3]));
        float x4  = float(ip[4])  * rs * (1.0f + float(wp[4]));
        float x5  = float(ip[5])  * rs * (1.0f + float(wp[5]));
        float x6  = float(ip[6])  * rs * (1.0f + float(wp[6]));
        float x7  = float(ip[7])  * rs * (1.0f + float(wp[7]));
        float x8  = float(ip[8])  * rs * (1.0f + float(wp[8]));
        float x9  = float(ip[9])  * rs * (1.0f + float(wp[9]));
        float x10 = float(ip[10]) * rs * (1.0f + float(wp[10]));
        float x11 = float(ip[11]) * rs * (1.0f + float(wp[11]));
        float x12 = float(ip[12]) * rs * (1.0f + float(wp[12]));
        float x13 = float(ip[13]) * rs * (1.0f + float(wp[13]));
        float x14 = float(ip[14]) * rs * (1.0f + float(wp[14]));
        float x15 = float(ip[15]) * rs * (1.0f + float(wp[15]));

        half hx0 = half(x0);
        half hx1 = half(x1);
        half hx2 = half(x2);
        half hx3 = half(x3);
        half hx4 = half(x4);
        half hx5 = half(x5);
        half hx6 = half(x6);
        half hx7 = half(x7);
        half hx8 = half(x8);
        half hx9 = half(x9);
        half hx10 = half(x10);
        half hx11 = half(x11);
        half hx12 = half(x12);
        half hx13 = half(x13);
        half hx14 = half(x14);
        half hx15 = half(x15);

        if (tgid == 0 && simd_group == 0) {
            normOutput[col + 0] = hx0;
            normOutput[col + 1] = hx1;
            normOutput[col + 2] = hx2;
            normOutput[col + 3] = hx3;
            normOutput[col + 4] = hx4;
            normOutput[col + 5] = hx5;
            normOutput[col + 6] = hx6;
            normOutput[col + 7] = hx7;
            normOutput[col + 8] = hx8;
            normOutput[col + 9] = hx9;
            normOutput[col + 10] = hx10;
            normOutput[col + 11] = hx11;
            normOutput[col + 12] = hx12;
            normOutput[col + 13] = hx13;
            normOutput[col + 14] = hx14;
            normOutput[col + 15] = hx15;
        }

        x0 = float(hx0);
        x1 = float(hx1);
        x2 = float(hx2);
        x3 = float(hx3);
        x4 = float(hx4);
        x5 = float(hx5);
        x6 = float(hx6);
        x7 = float(hx7);
        x8 = float(hx8);
        x9 = float(hx9);
        x10 = float(hx10);
        x11 = float(hx11);
        x12 = float(hx12);
        x13 = float(hx13);
        x14 = float(hx14);
        x15 = float(hx15);

        float xsum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
                   + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15;

        x1  *= (1.0f / 16.0f);
        x2  *= (1.0f / 256.0f);
        x3  *= (1.0f / 4096.0f);
        x5  *= (1.0f / 16.0f);
        x6  *= (1.0f / 256.0f);
        x7  *= (1.0f / 4096.0f);
        x9  *= (1.0f / 16.0f);
        x10 *= (1.0f / 256.0f);
        x11 *= (1.0f / 4096.0f);
        x13 *= (1.0f / 16.0f);
        x14 *= (1.0f / 256.0f);
        x15 *= (1.0f / 4096.0f);

#define AFFINE_DOT16(w) \
    ( x0  * float((w)[0] & 0x000Fu) + x1  * float((w)[0] & 0x00F0u) \
    + x2  * float((w)[0] & 0x0F00u) + x3  * float((w)[0] & 0xF000u) \
    + x4  * float((w)[1] & 0x000Fu) + x5  * float((w)[1] & 0x00F0u) \
    + x6  * float((w)[1] & 0x0F00u) + x7  * float((w)[1] & 0xF000u) \
    + x8  * float((w)[2] & 0x000Fu) + x9  * float((w)[2] & 0x00F0u) \
    + x10 * float((w)[2] & 0x0F00u) + x11 * float((w)[2] & 0xF000u) \
    + x12 * float((w)[3] & 0x000Fu) + x13 * float((w)[3] & 0x00F0u) \
    + x14 * float((w)[3] & 0x0F00u) + x15 * float((w)[3] & 0xF000u) )

        acc0 += float(sc0[g]) * AFFINE_DOT16(r0 + j * 4) + float(bi0[g]) * xsum;
        if (sgValidRows > 1)
        acc1 += float(sc1[g]) * AFFINE_DOT16(r1 + j * 4) + float(bi1[g]) * xsum;
        if (sgValidRows > 2)
        acc2 += float(sc2[g]) * AFFINE_DOT16(r2 + j * 4) + float(bi2[g]) * xsum;
        if (sgValidRows > 3)
        acc3 += float(sc3[g]) * AFFINE_DOT16(r3 + j * 4) + float(bi3[g]) * xsum;

#undef AFFINE_DOT16
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (simd_lane == 0) {
        output[sgBaseRow] = half(acc0);
        if (sgValidRows > 1) output[sgBaseRow + 1] = half(acc1);
        if (sgValidRows > 2) output[sgBaseRow + 2] = half(acc2);
        if (sgValidRows > 3) output[sgBaseRow + 3] = half(acc3);
    }
}

// ─── Norm-scale fused affine gate+up+SwiGLU ───
// Reads a pre-computed RMS scale and normalizes input inline before both affine matvecs.

kernel void norm_scale_affine_gate_up_swiglu(
    device const float*    scalePtr     [[buffer(0)]],  // [1]
    device const half*     normInput    [[buffer(1)]],  // [C]
    device const half*     normWeight   [[buffer(2)]],  // [C]
    device const uint8_t*  gate_weights [[buffer(3)]],  // [R, C/2]
    device const half*     gate_scales  [[buffer(4)]],  // [R, C/GS]
    device const half*     gate_biases  [[buffer(5)]],  // [R, C/GS]
    device const uint8_t*  up_weights   [[buffer(6)]],  // [R, C/2]
    device const half*     up_scales    [[buffer(7)]],  // [R, C/GS]
    device const half*     up_biases    [[buffer(8)]],  // [R, C/GS]
    device half*           output       [[buffer(9)]],  // [R]
    constant uint&         numRows      [[buffer(10)]], // R
    uint tgid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint baseRow = tgid * ROWS_PER_TG;
    uint halfCols = FC_COLS / 2;
    uint validRows = min(ROWS_PER_TG, numRows - baseRow);
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint sgValidRows = (validRows > simd_group * ROWS_PER_SG)
        ? min(ROWS_PER_SG, validRows - simd_group * ROWS_PER_SG) : 0u;
    if (sgValidRows == 0) return;

    float rs = scalePtr[0];
    uint numColGroups = FC_COLS / FC_GROUP_SIZE;
    uint eighthCols = halfCols / 8;

    float g0=0,g1=0,g2=0,g3=0;
    float u0=0,u1=0,u2=0,u3=0;

    uint u16PerRow = halfCols / 2;
    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * halfCols);
    device const uint16_t* gr1 = (sgValidRows > 1) ? gr0 + u16PerRow : gr0;
    device const uint16_t* gr2 = (sgValidRows > 2) ? gr0 + 2 * u16PerRow : gr0;
    device const uint16_t* gr3 = (sgValidRows > 3) ? gr0 + 3 * u16PerRow : gr0;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * halfCols);
    device const uint16_t* ur1 = (sgValidRows > 1) ? ur0 + u16PerRow : ur0;
    device const uint16_t* ur2 = (sgValidRows > 2) ? ur0 + 2 * u16PerRow : ur0;
    device const uint16_t* ur3 = (sgValidRows > 3) ? ur0 + 3 * u16PerRow : ur0;

    device const half* gs0=gate_scales+sgBaseRow*numColGroups; device const half* gs1=gs0+numColGroups; device const half* gs2=gs0+2*numColGroups; device const half* gs3=gs0+3*numColGroups;
    device const half* gb0=gate_biases+sgBaseRow*numColGroups; device const half* gb1=gb0+numColGroups; device const half* gb2=gb0+2*numColGroups; device const half* gb3=gb0+3*numColGroups;
    device const half* us0=up_scales+sgBaseRow*numColGroups;   device const half* us1=us0+numColGroups; device const half* us2=us0+2*numColGroups; device const half* us3=us0+3*numColGroups;
    device const half* ub0=up_biases+sgBaseRow*numColGroups;   device const half* ub1=ub0+numColGroups; device const half* ub2=ub0+2*numColGroups; device const half* ub3=ub0+3*numColGroups;

    for (uint j = simd_lane; j < eighthCols; j += 32) {
        uint col = j * 16;
        uint gidx = col / FC_GROUP_SIZE;
        device const half* ip = normInput + col;
        device const half* wp = normWeight + col;

        float x0  = float(ip[0])  * rs * (1.0f + float(wp[0]));
        float x1  = float(ip[1])  * rs * (1.0f + float(wp[1]));
        float x2  = float(ip[2])  * rs * (1.0f + float(wp[2]));
        float x3  = float(ip[3])  * rs * (1.0f + float(wp[3]));
        float x4  = float(ip[4])  * rs * (1.0f + float(wp[4]));
        float x5  = float(ip[5])  * rs * (1.0f + float(wp[5]));
        float x6  = float(ip[6])  * rs * (1.0f + float(wp[6]));
        float x7  = float(ip[7])  * rs * (1.0f + float(wp[7]));
        float x8  = float(ip[8])  * rs * (1.0f + float(wp[8]));
        float x9  = float(ip[9])  * rs * (1.0f + float(wp[9]));
        float x10 = float(ip[10]) * rs * (1.0f + float(wp[10]));
        float x11 = float(ip[11]) * rs * (1.0f + float(wp[11]));
        float x12 = float(ip[12]) * rs * (1.0f + float(wp[12]));
        float x13 = float(ip[13]) * rs * (1.0f + float(wp[13]));
        float x14 = float(ip[14]) * rs * (1.0f + float(wp[14]));
        float x15 = float(ip[15]) * rs * (1.0f + float(wp[15]));

        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

#define D16(w) \
    (x0*float((w)[0]&0x000Fu)+x1*float((w)[0]&0x00F0u)+x2*float((w)[0]&0x0F00u)+x3*float((w)[0]&0xF000u) \
    +x4*float((w)[1]&0x000Fu)+x5*float((w)[1]&0x00F0u)+x6*float((w)[1]&0x0F00u)+x7*float((w)[1]&0xF000u) \
    +x8*float((w)[2]&0x000Fu)+x9*float((w)[2]&0x00F0u)+x10*float((w)[2]&0x0F00u)+x11*float((w)[2]&0xF000u) \
    +x12*float((w)[3]&0x000Fu)+x13*float((w)[3]&0x00F0u)+x14*float((w)[3]&0x0F00u)+x15*float((w)[3]&0xF000u))

        { g0+=float(gs0[gidx])*D16(gr0+j*4)+float(gb0[gidx])*xsum; u0+=float(us0[gidx])*D16(ur0+j*4)+float(ub0[gidx])*xsum; }
        if (sgValidRows>1) { g1+=float(gs1[gidx])*D16(gr1+j*4)+float(gb1[gidx])*xsum; u1+=float(us1[gidx])*D16(ur1+j*4)+float(ub1[gidx])*xsum; }
        if (sgValidRows>2) { g2+=float(gs2[gidx])*D16(gr2+j*4)+float(gb2[gidx])*xsum; u2+=float(us2[gidx])*D16(ur2+j*4)+float(ub2[gidx])*xsum; }
        if (sgValidRows>3) { g3+=float(gs3[gidx])*D16(gr3+j*4)+float(gb3[gidx])*xsum; u3+=float(us3[gidx])*D16(ur3+j*4)+float(ub3[gidx])*xsum; }
#undef D16
    }

    g0=simd_sum(g0); g1=simd_sum(g1); g2=simd_sum(g2); g3=simd_sum(g3);
    u0=simd_sum(u0); u1=simd_sum(u1); u2=simd_sum(u2); u3=simd_sum(u3);

    if (simd_lane == 0) {
        output[sgBaseRow] = half(g0 / (1.0f + exp(-g0)) * u0);
        if (sgValidRows>1) output[sgBaseRow+1] = half(g1 / (1.0f + exp(-g1)) * u1);
        if (sgValidRows>2) output[sgBaseRow+2] = half(g2 / (1.0f + exp(-g2)) * u2);
        if (sgValidRows>3) output[sgBaseRow+3] = half(g3 / (1.0f + exp(-g3)) * u3);
    }
}

// ─── Norm-scale fused affine gate+up+GeGLU ───
// Reads a pre-computed RMS scale and normalizes input inline before both affine
// matvecs. Exactness requires matching the staged half round-trip on the
// normalized lanes before using them in the gate/up dot products.

inline half agent_geglu_product(float gate, float up);

kernel void norm_scale_affine_gate_up_geglu(
    device const float*    scalePtr     [[buffer(0)]],  // [1]
    device const half*     normInput    [[buffer(1)]],  // [C]
    device const half*     normWeight   [[buffer(2)]],  // [C]
    device const uint8_t*  gate_weights [[buffer(3)]],  // [R, C/2]
    device const half*     gate_scales  [[buffer(4)]],  // [R, C/GS]
    device const half*     gate_biases  [[buffer(5)]],  // [R, C/GS]
    device const uint8_t*  up_weights   [[buffer(6)]],  // [R, C/2]
    device const half*     up_scales    [[buffer(7)]],  // [R, C/GS]
    device const half*     up_biases    [[buffer(8)]],  // [R, C/GS]
    device half*           output       [[buffer(9)]],  // [R]
    constant uint&         numRows      [[buffer(10)]], // R
    uint tgid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint baseRow = tgid * ROWS_PER_TG;
    uint halfCols = FC_COLS / 2;
    uint validRows = min(ROWS_PER_TG, numRows - baseRow);
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint sgValidRows = (validRows > simd_group * ROWS_PER_SG)
        ? min(ROWS_PER_SG, validRows - simd_group * ROWS_PER_SG) : 0u;
    if (sgValidRows == 0) return;

    float rs = scalePtr[0];
    uint numColGroups = FC_COLS / FC_GROUP_SIZE;
    uint eighthCols = halfCols / 8;

    float g0=0,g1=0,g2=0,g3=0;
    float u0=0,u1=0,u2=0,u3=0;

    uint u16PerRow = halfCols / 2;
    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * halfCols);
    device const uint16_t* gr1 = (sgValidRows > 1) ? gr0 + u16PerRow : gr0;
    device const uint16_t* gr2 = (sgValidRows > 2) ? gr0 + 2 * u16PerRow : gr0;
    device const uint16_t* gr3 = (sgValidRows > 3) ? gr0 + 3 * u16PerRow : gr0;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * halfCols);
    device const uint16_t* ur1 = (sgValidRows > 1) ? ur0 + u16PerRow : ur0;
    device const uint16_t* ur2 = (sgValidRows > 2) ? ur0 + 2 * u16PerRow : ur0;
    device const uint16_t* ur3 = (sgValidRows > 3) ? ur0 + 3 * u16PerRow : ur0;

    device const half* gs0=gate_scales+sgBaseRow*numColGroups; device const half* gs1=gs0+numColGroups; device const half* gs2=gs0+2*numColGroups; device const half* gs3=gs0+3*numColGroups;
    device const half* gb0=gate_biases+sgBaseRow*numColGroups; device const half* gb1=gb0+numColGroups; device const half* gb2=gb0+2*numColGroups; device const half* gb3=gb0+3*numColGroups;
    device const half* us0=up_scales+sgBaseRow*numColGroups;   device const half* us1=us0+numColGroups; device const half* us2=us0+2*numColGroups; device const half* us3=us0+3*numColGroups;
    device const half* ub0=up_biases+sgBaseRow*numColGroups;   device const half* ub1=ub0+numColGroups; device const half* ub2=ub0+2*numColGroups; device const half* ub3=ub0+3*numColGroups;

    for (uint j = simd_lane; j < eighthCols; j += 32) {
        uint col = j * 16;
        uint gidx = col / FC_GROUP_SIZE;
        device const half* ip = normInput + col;
        device const half* wp = normWeight + col;

        float x0  = float(ip[0])  * rs * (1.0f + float(wp[0]));
        float x1  = float(ip[1])  * rs * (1.0f + float(wp[1]));
        float x2  = float(ip[2])  * rs * (1.0f + float(wp[2]));
        float x3  = float(ip[3])  * rs * (1.0f + float(wp[3]));
        float x4  = float(ip[4])  * rs * (1.0f + float(wp[4]));
        float x5  = float(ip[5])  * rs * (1.0f + float(wp[5]));
        float x6  = float(ip[6])  * rs * (1.0f + float(wp[6]));
        float x7  = float(ip[7])  * rs * (1.0f + float(wp[7]));
        float x8  = float(ip[8])  * rs * (1.0f + float(wp[8]));
        float x9  = float(ip[9])  * rs * (1.0f + float(wp[9]));
        float x10 = float(ip[10]) * rs * (1.0f + float(wp[10]));
        float x11 = float(ip[11]) * rs * (1.0f + float(wp[11]));
        float x12 = float(ip[12]) * rs * (1.0f + float(wp[12]));
        float x13 = float(ip[13]) * rs * (1.0f + float(wp[13]));
        float x14 = float(ip[14]) * rs * (1.0f + float(wp[14]));
        float x15 = float(ip[15]) * rs * (1.0f + float(wp[15]));

        half hx0 = half(x0);
        half hx1 = half(x1);
        half hx2 = half(x2);
        half hx3 = half(x3);
        half hx4 = half(x4);
        half hx5 = half(x5);
        half hx6 = half(x6);
        half hx7 = half(x7);
        half hx8 = half(x8);
        half hx9 = half(x9);
        half hx10 = half(x10);
        half hx11 = half(x11);
        half hx12 = half(x12);
        half hx13 = half(x13);
        half hx14 = half(x14);
        half hx15 = half(x15);

        x0 = float(hx0);
        x1 = float(hx1);
        x2 = float(hx2);
        x3 = float(hx3);
        x4 = float(hx4);
        x5 = float(hx5);
        x6 = float(hx6);
        x7 = float(hx7);
        x8 = float(hx8);
        x9 = float(hx9);
        x10 = float(hx10);
        x11 = float(hx11);
        x12 = float(hx12);
        x13 = float(hx13);
        x14 = float(hx14);
        x15 = float(hx15);

        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

#define D16(w) \
    (x0*float((w)[0]&0x000Fu)+x1*float((w)[0]&0x00F0u)+x2*float((w)[0]&0x0F00u)+x3*float((w)[0]&0xF000u) \
    +x4*float((w)[1]&0x000Fu)+x5*float((w)[1]&0x00F0u)+x6*float((w)[1]&0x0F00u)+x7*float((w)[1]&0xF000u) \
    +x8*float((w)[2]&0x000Fu)+x9*float((w)[2]&0x00F0u)+x10*float((w)[2]&0x0F00u)+x11*float((w)[2]&0xF000u) \
    +x12*float((w)[3]&0x000Fu)+x13*float((w)[3]&0x00F0u)+x14*float((w)[3]&0x0F00u)+x15*float((w)[3]&0xF000u))

        { g0+=float(gs0[gidx])*D16(gr0+j*4)+float(gb0[gidx])*xsum; u0+=float(us0[gidx])*D16(ur0+j*4)+float(ub0[gidx])*xsum; }
        if (sgValidRows>1) { g1+=float(gs1[gidx])*D16(gr1+j*4)+float(gb1[gidx])*xsum; u1+=float(us1[gidx])*D16(ur1+j*4)+float(ub1[gidx])*xsum; }
        if (sgValidRows>2) { g2+=float(gs2[gidx])*D16(gr2+j*4)+float(gb2[gidx])*xsum; u2+=float(us2[gidx])*D16(ur2+j*4)+float(ub2[gidx])*xsum; }
        if (sgValidRows>3) { g3+=float(gs3[gidx])*D16(gr3+j*4)+float(gb3[gidx])*xsum; u3+=float(us3[gidx])*D16(ur3+j*4)+float(ub3[gidx])*xsum; }
#undef D16
    }

    g0=simd_sum(g0); g1=simd_sum(g1); g2=simd_sum(g2); g3=simd_sum(g3);
    u0=simd_sum(u0); u1=simd_sum(u1); u2=simd_sum(u2); u3=simd_sum(u3);

    if (simd_lane == 0) {
        output[sgBaseRow] = agent_geglu_product(g0, u0);
        if (sgValidRows>1) output[sgBaseRow+1] = agent_geglu_product(g1, u1);
        if (sgValidRows>2) output[sgBaseRow+2] = agent_geglu_product(g2, u2);
        if (sgValidRows>3) output[sgBaseRow+3] = agent_geglu_product(g3, u3);
    }
}

// ─── Fused affine matvec + residual add ───
// Same as affine_matvec but adds a residual vector at writeback:
//   output[r] = matvec_result[r] + residual[r]
// Eliminates a separate elementwise_add dispatch.
// Dispatch: ceil(R/8) threadgroups × 64 threads.

kernel void fused_affine_matvec_add(
    device const uint8_t* weights  [[buffer(0)]],  // [R, C/2] packed u4
    device const half*    scales   [[buffer(1)]],  // [R, C/FC_GROUP_SIZE]
    device const half*    biases   [[buffer(2)]],  // [R, C/FC_GROUP_SIZE]
    device const half*    input    [[buffer(3)]],  // [C]
    device half*          output   [[buffer(4)]],  // [R]
    device const half*    residual [[buffer(5)]],  // [R]
    constant uint&        numRows  [[buffer(6)]],  // R
    uint tgid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint baseRow = tgid * ROWS_PER_TG;
    uint halfCols = FC_COLS / 2;
    uint validRows = min(ROWS_PER_TG, numRows - baseRow);

    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint sgValidRows = (validRows > simd_group * ROWS_PER_SG)
        ? min(ROWS_PER_SG, validRows - simd_group * ROWS_PER_SG)
        : 0u;

    if (sgValidRows == 0) return;

    uint numColGroups = FC_COLS / FC_GROUP_SIZE;
    uint eighthCols = halfCols / 8;
    uint u16PerRow = halfCols / 2;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * halfCols);
    device const uint16_t* r1 = (sgValidRows > 1) ? r0 + u16PerRow : r0;
    device const uint16_t* r2 = (sgValidRows > 2) ? r0 + 2 * u16PerRow : r0;
    device const uint16_t* r3 = (sgValidRows > 3) ? r0 + 3 * u16PerRow : r0;

    device const half* sc0 = scales + sgBaseRow * numColGroups;
    device const half* sc1 = sc0 + numColGroups;
    device const half* sc2 = sc0 + 2 * numColGroups;
    device const half* sc3 = sc0 + 3 * numColGroups;
    device const half* bi0 = biases + sgBaseRow * numColGroups;
    device const half* bi1 = bi0 + numColGroups;
    device const half* bi2 = bi0 + 2 * numColGroups;
    device const half* bi3 = bi0 + 3 * numColGroups;

    for (uint j = simd_lane; j < eighthCols; j += 32) {
        uint col = j * 16;
        uint g = col / FC_GROUP_SIZE;

        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

#define AFFINE_DOT16(w) \
    (x0*float((w)[0]&0x000Fu)+x1*float((w)[0]&0x00F0u)+x2*float((w)[0]&0x0F00u)+x3*float((w)[0]&0xF000u) \
    +x4*float((w)[1]&0x000Fu)+x5*float((w)[1]&0x00F0u)+x6*float((w)[1]&0x0F00u)+x7*float((w)[1]&0xF000u) \
    +x8*float((w)[2]&0x000Fu)+x9*float((w)[2]&0x00F0u)+x10*float((w)[2]&0x0F00u)+x11*float((w)[2]&0xF000u) \
    +x12*float((w)[3]&0x000Fu)+x13*float((w)[3]&0x00F0u)+x14*float((w)[3]&0x0F00u)+x15*float((w)[3]&0xF000u))

        acc0 += float(sc0[g]) * AFFINE_DOT16(r0 + j*4) + float(bi0[g]) * xsum;
        if (sgValidRows > 1)
        acc1 += float(sc1[g]) * AFFINE_DOT16(r1 + j*4) + float(bi1[g]) * xsum;
        if (sgValidRows > 2)
        acc2 += float(sc2[g]) * AFFINE_DOT16(r2 + j*4) + float(bi2[g]) * xsum;
        if (sgValidRows > 3)
        acc3 += float(sc3[g]) * AFFINE_DOT16(r3 + j*4) + float(bi3[g]) * xsum;

#undef AFFINE_DOT16
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    // Add residual and write
    if (simd_lane == 0) {
        device half* out = output + sgBaseRow;
        out[0] = half(acc0);
        out[0] = out[0] + residual[sgBaseRow];
        if (sgValidRows > 1) {
            out[1] = half(acc1);
            out[1] = out[1] + residual[sgBaseRow + 1];
        }
        if (sgValidRows > 2) {
            out[2] = half(acc2);
            out[2] = out[2] + residual[sgBaseRow + 2];
        }
        if (sgValidRows > 3) {
            out[3] = half(acc3);
            out[3] = out[3] + residual[sgBaseRow + 3];
        }
    }
}

// ─── Fused dual affine matvec ───
// Two [R, C] affine matvecs sharing the same input, writing to two outputs.
// Replaces 2 separate affine_matvec dispatches (e.g., A+B projections in DeltaNet).
// Dispatch: ceil(R/8) threadgroups × 64 threads.

kernel void fused_dual_affine_matvec(
    device const uint8_t* w1_weights [[buffer(0)]],   // [R, C/2]
    device const half*    w1_scales  [[buffer(1)]],   // [R, C/GS]
    device const half*    w1_biases  [[buffer(2)]],   // [R, C/GS]
    device const uint8_t* w2_weights [[buffer(3)]],   // [R, C/2]
    device const half*    w2_scales  [[buffer(4)]],   // [R, C/GS]
    device const half*    w2_biases  [[buffer(5)]],   // [R, C/GS]
    device const half*    input      [[buffer(6)]],   // [C]
    device half*          output1    [[buffer(7)]],   // [R]
    device half*          output2    [[buffer(8)]],   // [R]
    constant uint&        numRows    [[buffer(9)]],   // R
    uint tgid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint baseRow = tgid * ROWS_PER_TG;
    uint halfCols = FC_COLS / 2;
    uint validRows = min(ROWS_PER_TG, numRows - baseRow);
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint sgValidRows = (validRows > simd_group * ROWS_PER_SG)
        ? min(ROWS_PER_SG, validRows - simd_group * ROWS_PER_SG) : 0u;
    if (sgValidRows == 0) return;

    uint numColGroups = FC_COLS / FC_GROUP_SIZE;
    uint eighthCols = halfCols / 8;

    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;  // w1 accumulators
    float b0 = 0, b1 = 0, b2 = 0, b3 = 0;  // w2 accumulators

    uint u16PerRow = halfCols / 2;
    device const uint16_t* w1r0 = (device const uint16_t*)(w1_weights + sgBaseRow * halfCols);
    device const uint16_t* w1r1 = (sgValidRows > 1) ? w1r0 + u16PerRow : w1r0;
    device const uint16_t* w1r2 = (sgValidRows > 2) ? w1r0 + 2 * u16PerRow : w1r0;
    device const uint16_t* w1r3 = (sgValidRows > 3) ? w1r0 + 3 * u16PerRow : w1r0;
    device const uint16_t* w2r0 = (device const uint16_t*)(w2_weights + sgBaseRow * halfCols);
    device const uint16_t* w2r1 = (sgValidRows > 1) ? w2r0 + u16PerRow : w2r0;
    device const uint16_t* w2r2 = (sgValidRows > 2) ? w2r0 + 2 * u16PerRow : w2r0;
    device const uint16_t* w2r3 = (sgValidRows > 3) ? w2r0 + 3 * u16PerRow : w2r0;

    device const half* w1s0 = w1_scales + sgBaseRow * numColGroups;
    device const half* w1s1 = w1s0 + numColGroups; device const half* w1s2 = w1s0 + 2 * numColGroups; device const half* w1s3 = w1s0 + 3 * numColGroups;
    device const half* w1b0 = w1_biases + sgBaseRow * numColGroups;
    device const half* w1b1 = w1b0 + numColGroups; device const half* w1b2 = w1b0 + 2 * numColGroups; device const half* w1b3 = w1b0 + 3 * numColGroups;
    device const half* w2s0 = w2_scales + sgBaseRow * numColGroups;
    device const half* w2s1 = w2s0 + numColGroups; device const half* w2s2 = w2s0 + 2 * numColGroups; device const half* w2s3 = w2s0 + 3 * numColGroups;
    device const half* w2b0 = w2_biases + sgBaseRow * numColGroups;
    device const half* w2b1 = w2b0 + numColGroups; device const half* w2b2 = w2b0 + 2 * numColGroups; device const half* w2b3 = w2b0 + 3 * numColGroups;

    for (uint j = simd_lane; j < eighthCols; j += 32) {
        uint col = j * 16;
        uint g = col / FC_GROUP_SIZE;
        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

#define D16(w) \
    (x0*float((w)[0]&0x000Fu)+x1*float((w)[0]&0x00F0u)+x2*float((w)[0]&0x0F00u)+x3*float((w)[0]&0xF000u) \
    +x4*float((w)[1]&0x000Fu)+x5*float((w)[1]&0x00F0u)+x6*float((w)[1]&0x0F00u)+x7*float((w)[1]&0xF000u) \
    +x8*float((w)[2]&0x000Fu)+x9*float((w)[2]&0x00F0u)+x10*float((w)[2]&0x0F00u)+x11*float((w)[2]&0xF000u) \
    +x12*float((w)[3]&0x000Fu)+x13*float((w)[3]&0x00F0u)+x14*float((w)[3]&0x0F00u)+x15*float((w)[3]&0xF000u))

        { a0+=float(w1s0[g])*D16(w1r0+j*4)+float(w1b0[g])*xsum; b0+=float(w2s0[g])*D16(w2r0+j*4)+float(w2b0[g])*xsum; }
        if (sgValidRows>1) { a1+=float(w1s1[g])*D16(w1r1+j*4)+float(w1b1[g])*xsum; b1+=float(w2s1[g])*D16(w2r1+j*4)+float(w2b1[g])*xsum; }
        if (sgValidRows>2) { a2+=float(w1s2[g])*D16(w1r2+j*4)+float(w1b2[g])*xsum; b2+=float(w2s2[g])*D16(w2r2+j*4)+float(w2b2[g])*xsum; }
        if (sgValidRows>3) { a3+=float(w1s3[g])*D16(w1r3+j*4)+float(w1b3[g])*xsum; b3+=float(w2s3[g])*D16(w2r3+j*4)+float(w2b3[g])*xsum; }
#undef D16
    }

    a0=simd_sum(a0); a1=simd_sum(a1); a2=simd_sum(a2); a3=simd_sum(a3);
    b0=simd_sum(b0); b1=simd_sum(b1); b2=simd_sum(b2); b3=simd_sum(b3);
    if (simd_lane == 0) {
        output1[sgBaseRow]=half(a0); output2[sgBaseRow]=half(b0);
        if (sgValidRows>1) { output1[sgBaseRow+1]=half(a1); output2[sgBaseRow+1]=half(b1); }
        if (sgValidRows>2) { output1[sgBaseRow+2]=half(a2); output2[sgBaseRow+2]=half(b2); }
        if (sgValidRows>3) { output1[sgBaseRow+3]=half(a3); output2[sgBaseRow+3]=half(b3); }
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_dual_affine_matvec_fixed(
    device const uint8_t* w1_weights,
    device const half*    w1_scales,
    device const half*    w1_biases,
    device const uint8_t* w2_weights,
    device const half*    w2_scales,
    device const half*    w2_biases,
    device const half*    input,
    device half*          output1,
    device half*          output2,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;

    uint baseRow = tgid * ROWS_PER_TG;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;

    device const uint16_t* w1r0 = (device const uint16_t*)(w1_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* w1r1 = w1r0 + U16_PER_ROW;
    device const uint16_t* w1r2 = w1r1 + U16_PER_ROW;
    device const uint16_t* w1r3 = w1r2 + U16_PER_ROW;
    device const uint16_t* w2r0 = (device const uint16_t*)(w2_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* w2r1 = w2r0 + U16_PER_ROW;
    device const uint16_t* w2r2 = w2r1 + U16_PER_ROW;
    device const uint16_t* w2r3 = w2r2 + U16_PER_ROW;

    device const half* w1s0 = w1_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* w1s1 = w1s0 + NUM_COL_GROUPS;
    device const half* w1s2 = w1s1 + NUM_COL_GROUPS;
    device const half* w1s3 = w1s2 + NUM_COL_GROUPS;
    device const half* w1b0 = w1_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* w1b1 = w1b0 + NUM_COL_GROUPS;
    device const half* w1b2 = w1b1 + NUM_COL_GROUPS;
    device const half* w1b3 = w1b2 + NUM_COL_GROUPS;
    device const half* w2s0 = w2_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* w2s1 = w2s0 + NUM_COL_GROUPS;
    device const half* w2s2 = w2s1 + NUM_COL_GROUPS;
    device const half* w2s3 = w2s2 + NUM_COL_GROUPS;
    device const half* w2b0 = w2_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* w2b1 = w2b0 + NUM_COL_GROUPS;
    device const half* w2b2 = w2b1 + NUM_COL_GROUPS;
    device const half* w2b3 = w2b2 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;
        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

#define D16_FIXED(w) \
    (x0*float((w)[0]&0x000Fu)+x1*float((w)[0]&0x00F0u)+x2*float((w)[0]&0x0F00u)+x3*float((w)[0]&0xF000u) \
    +x4*float((w)[1]&0x000Fu)+x5*float((w)[1]&0x00F0u)+x6*float((w)[1]&0x0F00u)+x7*float((w)[1]&0xF000u) \
    +x8*float((w)[2]&0x000Fu)+x9*float((w)[2]&0x00F0u)+x10*float((w)[2]&0x0F00u)+x11*float((w)[2]&0xF000u) \
    +x12*float((w)[3]&0x000Fu)+x13*float((w)[3]&0x00F0u)+x14*float((w)[3]&0x0F00u)+x15*float((w)[3]&0xF000u))

        a0 += float(w1s0[g]) * D16_FIXED(w1r0 + j * 4) + float(w1b0[g]) * xsum;
        b0 += float(w2s0[g]) * D16_FIXED(w2r0 + j * 4) + float(w2b0[g]) * xsum;
        a1 += float(w1s1[g]) * D16_FIXED(w1r1 + j * 4) + float(w1b1[g]) * xsum;
        b1 += float(w2s1[g]) * D16_FIXED(w2r1 + j * 4) + float(w2b1[g]) * xsum;
        a2 += float(w1s2[g]) * D16_FIXED(w1r2 + j * 4) + float(w1b2[g]) * xsum;
        b2 += float(w2s2[g]) * D16_FIXED(w2r2 + j * 4) + float(w2b2[g]) * xsum;
        a3 += float(w1s3[g]) * D16_FIXED(w1r3 + j * 4) + float(w1b3[g]) * xsum;
        b3 += float(w2s3[g]) * D16_FIXED(w2r3 + j * 4) + float(w2b3[g]) * xsum;
#undef D16_FIXED
    }

    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
    b0 = simd_sum(b0); b1 = simd_sum(b1); b2 = simd_sum(b2); b3 = simd_sum(b3);

    if (simd_lane == 0) {
        output1[sgBaseRow] = half(a0); output2[sgBaseRow] = half(b0);
        output1[sgBaseRow + 1] = half(a1); output2[sgBaseRow + 1] = half(b1);
        output1[sgBaseRow + 2] = half(a2); output2[sgBaseRow + 2] = half(b2);
        output1[sgBaseRow + 3] = half(a3); output2[sgBaseRow + 3] = half(b3);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_dual_affine_matvec_fixed_batched(
    device const uint8_t* w1_weights,
    device const half*    w1_scales,
    device const half*    w1_biases,
    device const uint8_t* w2_weights,
    device const half*    w2_scales,
    device const half*    w2_biases,
    device const half*    input,
    device half*          output1,
    device half*          output2,
    uint2 tgid,
    uint simd_lane,
    uint simd_group
) {
    fused_dual_affine_matvec_fixed<FIXED_ROWS, FIXED_COLS, FIXED_GROUP_SIZE>(
        w1_weights, w1_scales, w1_biases,
        w2_weights, w2_scales, w2_biases,
        input + tgid.y * FIXED_COLS,
        output1 + tgid.y * FIXED_ROWS,
        output2 + tgid.y * FIXED_ROWS,
        tgid.x, simd_lane, simd_group
    );
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void norm_scale_fused_dual_affine_matvec_fixed_batched(
    device const float*   scalePtr,
    device const half*    normInput,
    device const half*    normWeight,
    device const uint8_t* w1_weights,
    device const half*    w1_scales,
    device const half*    w1_biases,
    device const uint8_t* w2_weights,
    device const half*    w2_scales,
    device const half*    w2_biases,
    device half*          output1,
    device half*          output2,
    uint2 tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;

    uint baseRow = tgid.x * ROWS_PER_TG;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint batchOffset = tgid.y * FIXED_COLS;
    float rs = scalePtr[tgid.y];

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;

    device const uint16_t* w1r0 = (device const uint16_t*)(w1_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* w1r1 = w1r0 + U16_PER_ROW;
    device const uint16_t* w1r2 = w1r1 + U16_PER_ROW;
    device const uint16_t* w1r3 = w1r2 + U16_PER_ROW;
    device const uint16_t* w2r0 = (device const uint16_t*)(w2_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* w2r1 = w2r0 + U16_PER_ROW;
    device const uint16_t* w2r2 = w2r1 + U16_PER_ROW;
    device const uint16_t* w2r3 = w2r2 + U16_PER_ROW;

    device const half* w1s0 = w1_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* w1s1 = w1s0 + NUM_COL_GROUPS;
    device const half* w1s2 = w1s1 + NUM_COL_GROUPS;
    device const half* w1s3 = w1s2 + NUM_COL_GROUPS;
    device const half* w1b0 = w1_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* w1b1 = w1b0 + NUM_COL_GROUPS;
    device const half* w1b2 = w1b1 + NUM_COL_GROUPS;
    device const half* w1b3 = w1b2 + NUM_COL_GROUPS;
    device const half* w2s0 = w2_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* w2s1 = w2s0 + NUM_COL_GROUPS;
    device const half* w2s2 = w2s1 + NUM_COL_GROUPS;
    device const half* w2s3 = w2s2 + NUM_COL_GROUPS;
    device const half* w2b0 = w2_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* w2b1 = w2b0 + NUM_COL_GROUPS;
    device const half* w2b2 = w2b1 + NUM_COL_GROUPS;
    device const half* w2b3 = w2b2 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;
        device const half* xp = normInput + batchOffset + col;
        device const half* wp = normWeight + col;

        half hx0  = half(float(xp[0])  * rs * (1.0f + float(wp[0])));
        half hx1  = half(float(xp[1])  * rs * (1.0f + float(wp[1])));
        half hx2  = half(float(xp[2])  * rs * (1.0f + float(wp[2])));
        half hx3  = half(float(xp[3])  * rs * (1.0f + float(wp[3])));
        half hx4  = half(float(xp[4])  * rs * (1.0f + float(wp[4])));
        half hx5  = half(float(xp[5])  * rs * (1.0f + float(wp[5])));
        half hx6  = half(float(xp[6])  * rs * (1.0f + float(wp[6])));
        half hx7  = half(float(xp[7])  * rs * (1.0f + float(wp[7])));
        half hx8  = half(float(xp[8])  * rs * (1.0f + float(wp[8])));
        half hx9  = half(float(xp[9])  * rs * (1.0f + float(wp[9])));
        half hx10 = half(float(xp[10]) * rs * (1.0f + float(wp[10])));
        half hx11 = half(float(xp[11]) * rs * (1.0f + float(wp[11])));
        half hx12 = half(float(xp[12]) * rs * (1.0f + float(wp[12])));
        half hx13 = half(float(xp[13]) * rs * (1.0f + float(wp[13])));
        half hx14 = half(float(xp[14]) * rs * (1.0f + float(wp[14])));
        half hx15 = half(float(xp[15]) * rs * (1.0f + float(wp[15])));

        float x0=float(hx0),x1=float(hx1),x2=float(hx2),x3=float(hx3);
        float x4=float(hx4),x5=float(hx5),x6=float(hx6),x7=float(hx7);
        float x8=float(hx8),x9=float(hx9),x10=float(hx10),x11=float(hx11);
        float x12=float(hx12),x13=float(hx13),x14=float(hx14),x15=float(hx15);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

#define NORM_SCALE_D16_FIXED(w) \
    (x0*float((w)[0]&0x000Fu)+x1*float((w)[0]&0x00F0u)+x2*float((w)[0]&0x0F00u)+x3*float((w)[0]&0xF000u) \
    +x4*float((w)[1]&0x000Fu)+x5*float((w)[1]&0x00F0u)+x6*float((w)[1]&0x0F00u)+x7*float((w)[1]&0xF000u) \
    +x8*float((w)[2]&0x000Fu)+x9*float((w)[2]&0x00F0u)+x10*float((w)[2]&0x0F00u)+x11*float((w)[2]&0xF000u) \
    +x12*float((w)[3]&0x000Fu)+x13*float((w)[3]&0x00F0u)+x14*float((w)[3]&0x0F00u)+x15*float((w)[3]&0xF000u))

        a0 += float(w1s0[g]) * NORM_SCALE_D16_FIXED(w1r0 + j * 4) + float(w1b0[g]) * xsum;
        b0 += float(w2s0[g]) * NORM_SCALE_D16_FIXED(w2r0 + j * 4) + float(w2b0[g]) * xsum;
        a1 += float(w1s1[g]) * NORM_SCALE_D16_FIXED(w1r1 + j * 4) + float(w1b1[g]) * xsum;
        b1 += float(w2s1[g]) * NORM_SCALE_D16_FIXED(w2r1 + j * 4) + float(w2b1[g]) * xsum;
        a2 += float(w1s2[g]) * NORM_SCALE_D16_FIXED(w1r2 + j * 4) + float(w1b2[g]) * xsum;
        b2 += float(w2s2[g]) * NORM_SCALE_D16_FIXED(w2r2 + j * 4) + float(w2b2[g]) * xsum;
        a3 += float(w1s3[g]) * NORM_SCALE_D16_FIXED(w1r3 + j * 4) + float(w1b3[g]) * xsum;
        b3 += float(w2s3[g]) * NORM_SCALE_D16_FIXED(w2r3 + j * 4) + float(w2b3[g]) * xsum;
#undef NORM_SCALE_D16_FIXED
    }

    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
    b0 = simd_sum(b0); b1 = simd_sum(b1); b2 = simd_sum(b2); b3 = simd_sum(b3);

    if (simd_lane == 0) {
        device half* out1 = output1 + tgid.y * FIXED_ROWS + sgBaseRow;
        device half* out2 = output2 + tgid.y * FIXED_ROWS + sgBaseRow;
        out1[0] = half(a0); out2[0] = half(b0);
        out1[1] = half(a1); out2[1] = half(b1);
        out1[2] = half(a2); out2[2] = half(b2);
        out1[3] = half(a3); out2[3] = half(b3);
    }
}

// ─── Fused affine gate+up+SwiGLU ───
// Two [R, C] affine matvecs (gate + up) + SwiGLU: output = silu(gate) * up.
// Replaces 3 separate dispatches. Dispatch: ceil(R/8) threadgroups × 64 threads.

kernel void fused_affine_gate_up_swiglu(
    device const uint8_t* gate_weights [[buffer(0)]],  // [R, C/2]
    device const half*    gate_scales  [[buffer(1)]],  // [R, C/GS]
    device const half*    gate_biases  [[buffer(2)]],  // [R, C/GS]
    device const uint8_t* up_weights   [[buffer(3)]],  // [R, C/2]
    device const half*    up_scales    [[buffer(4)]],  // [R, C/GS]
    device const half*    up_biases    [[buffer(5)]],  // [R, C/GS]
    device const half*    input        [[buffer(6)]],  // [C]
    device half*          output       [[buffer(7)]],  // [R]
    constant uint&        numRows      [[buffer(8)]],  // R
    uint tgid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint baseRow = tgid * ROWS_PER_TG;
    uint halfCols = FC_COLS / 2;
    uint validRows = min(ROWS_PER_TG, numRows - baseRow);
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint sgValidRows = (validRows > simd_group * ROWS_PER_SG)
        ? min(ROWS_PER_SG, validRows - simd_group * ROWS_PER_SG) : 0u;
    if (sgValidRows == 0) return;

    uint numColGroups = FC_COLS / FC_GROUP_SIZE;
    uint eighthCols = halfCols / 8;

    float g0=0,g1=0,g2=0,g3=0;  // gate accumulators
    float u0=0,u1=0,u2=0,u3=0;  // up accumulators

    uint u16PerRow = halfCols / 2;
    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * halfCols);
    device const uint16_t* gr1 = (sgValidRows > 1) ? gr0 + u16PerRow : gr0;
    device const uint16_t* gr2 = (sgValidRows > 2) ? gr0 + 2 * u16PerRow : gr0;
    device const uint16_t* gr3 = (sgValidRows > 3) ? gr0 + 3 * u16PerRow : gr0;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * halfCols);
    device const uint16_t* ur1 = (sgValidRows > 1) ? ur0 + u16PerRow : ur0;
    device const uint16_t* ur2 = (sgValidRows > 2) ? ur0 + 2 * u16PerRow : ur0;
    device const uint16_t* ur3 = (sgValidRows > 3) ? ur0 + 3 * u16PerRow : ur0;

    device const half* gs0=gate_scales+sgBaseRow*numColGroups; device const half* gs1=gs0+numColGroups; device const half* gs2=gs0+2*numColGroups; device const half* gs3=gs0+3*numColGroups;
    device const half* gb0=gate_biases+sgBaseRow*numColGroups; device const half* gb1=gb0+numColGroups; device const half* gb2=gb0+2*numColGroups; device const half* gb3=gb0+3*numColGroups;
    device const half* us0=up_scales+sgBaseRow*numColGroups;   device const half* us1=us0+numColGroups; device const half* us2=us0+2*numColGroups; device const half* us3=us0+3*numColGroups;
    device const half* ub0=up_biases+sgBaseRow*numColGroups;   device const half* ub1=ub0+numColGroups; device const half* ub2=ub0+2*numColGroups; device const half* ub3=ub0+3*numColGroups;

    for (uint j = simd_lane; j < eighthCols; j += 32) {
        uint col = j * 16;
        uint gidx = col / FC_GROUP_SIZE;
        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

#define D16(w) \
    (x0*float((w)[0]&0x000Fu)+x1*float((w)[0]&0x00F0u)+x2*float((w)[0]&0x0F00u)+x3*float((w)[0]&0xF000u) \
    +x4*float((w)[1]&0x000Fu)+x5*float((w)[1]&0x00F0u)+x6*float((w)[1]&0x0F00u)+x7*float((w)[1]&0xF000u) \
    +x8*float((w)[2]&0x000Fu)+x9*float((w)[2]&0x00F0u)+x10*float((w)[2]&0x0F00u)+x11*float((w)[2]&0xF000u) \
    +x12*float((w)[3]&0x000Fu)+x13*float((w)[3]&0x00F0u)+x14*float((w)[3]&0x0F00u)+x15*float((w)[3]&0xF000u))

        { g0+=float(gs0[gidx])*D16(gr0+j*4)+float(gb0[gidx])*xsum; u0+=float(us0[gidx])*D16(ur0+j*4)+float(ub0[gidx])*xsum; }
        if (sgValidRows>1) { g1+=float(gs1[gidx])*D16(gr1+j*4)+float(gb1[gidx])*xsum; u1+=float(us1[gidx])*D16(ur1+j*4)+float(ub1[gidx])*xsum; }
        if (sgValidRows>2) { g2+=float(gs2[gidx])*D16(gr2+j*4)+float(gb2[gidx])*xsum; u2+=float(us2[gidx])*D16(ur2+j*4)+float(ub2[gidx])*xsum; }
        if (sgValidRows>3) { g3+=float(gs3[gidx])*D16(gr3+j*4)+float(gb3[gidx])*xsum; u3+=float(us3[gidx])*D16(ur3+j*4)+float(ub3[gidx])*xsum; }
#undef D16
    }

    g0=simd_sum(g0); g1=simd_sum(g1); g2=simd_sum(g2); g3=simd_sum(g3);
    u0=simd_sum(u0); u1=simd_sum(u1); u2=simd_sum(u2); u3=simd_sum(u3);

    // SwiGLU: output = silu(gate) * up = gate * sigmoid(gate) * up
    if (simd_lane == 0) {
        output[sgBaseRow] = half(g0 / (1.0f + exp(-g0)) * u0);
        if (sgValidRows>1) output[sgBaseRow+1] = half(g1 / (1.0f + exp(-g1)) * u1);
        if (sgValidRows>2) output[sgBaseRow+2] = half(g2 / (1.0f + exp(-g2)) * u2);
        if (sgValidRows>3) output[sgBaseRow+3] = half(g3 / (1.0f + exp(-g3)) * u3);
    }
}

inline half agent_geglu_product(float gate, float up) {
    gate = float(half(gate));
    up = float(half(up));
    float gate3 = gate * gate * gate;
    float inner = 0.7978845608f * (gate + 0.044715f * gate3);
    inner = clamp(inner, -20.0f, 20.0f);
    float geluGate = 0.5f * gate * (1.0f + tanh(inner));
    return half(clamp(geluGate * up, -65504.0f, 65504.0f));
}

kernel void fused_affine_gate_up_geglu(
    device const uint8_t* gate_weights [[buffer(0)]],  // [R, C/2]
    device const half*    gate_scales  [[buffer(1)]],  // [R, C/GS]
    device const half*    gate_biases  [[buffer(2)]],  // [R, C/GS]
    device const uint8_t* up_weights   [[buffer(3)]],  // [R, C/2]
    device const half*    up_scales    [[buffer(4)]],  // [R, C/GS]
    device const half*    up_biases    [[buffer(5)]],  // [R, C/GS]
    device const half*    input        [[buffer(6)]],  // [C]
    device half*          output       [[buffer(7)]],  // [R]
    constant uint&        numRows      [[buffer(8)]],  // R
    uint tgid       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint baseRow = tgid * ROWS_PER_TG;
    uint halfCols = FC_COLS / 2;
    uint validRows = min(ROWS_PER_TG, numRows - baseRow);
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint sgValidRows = (validRows > simd_group * ROWS_PER_SG)
        ? min(ROWS_PER_SG, validRows - simd_group * ROWS_PER_SG) : 0u;
    if (sgValidRows == 0) return;

    uint numColGroups = FC_COLS / FC_GROUP_SIZE;
    uint eighthCols = halfCols / 8;

    float g0=0,g1=0,g2=0,g3=0;
    float u0=0,u1=0,u2=0,u3=0;

    uint u16PerRow = halfCols / 2;
    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * halfCols);
    device const uint16_t* gr1 = (sgValidRows > 1) ? gr0 + u16PerRow : gr0;
    device const uint16_t* gr2 = (sgValidRows > 2) ? gr0 + 2 * u16PerRow : gr0;
    device const uint16_t* gr3 = (sgValidRows > 3) ? gr0 + 3 * u16PerRow : gr0;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * halfCols);
    device const uint16_t* ur1 = (sgValidRows > 1) ? ur0 + u16PerRow : ur0;
    device const uint16_t* ur2 = (sgValidRows > 2) ? ur0 + 2 * u16PerRow : ur0;
    device const uint16_t* ur3 = (sgValidRows > 3) ? ur0 + 3 * u16PerRow : ur0;

    device const half* gs0=gate_scales+sgBaseRow*numColGroups; device const half* gs1=gs0+numColGroups; device const half* gs2=gs0+2*numColGroups; device const half* gs3=gs0+3*numColGroups;
    device const half* gb0=gate_biases+sgBaseRow*numColGroups; device const half* gb1=gb0+numColGroups; device const half* gb2=gb0+2*numColGroups; device const half* gb3=gb0+3*numColGroups;
    device const half* us0=up_scales+sgBaseRow*numColGroups;   device const half* us1=us0+numColGroups; device const half* us2=us0+2*numColGroups; device const half* us3=us0+3*numColGroups;
    device const half* ub0=up_biases+sgBaseRow*numColGroups;   device const half* ub1=ub0+numColGroups; device const half* ub2=ub0+2*numColGroups; device const half* ub3=ub0+3*numColGroups;

    for (uint j = simd_lane; j < eighthCols; j += 32) {
        uint col = j * 16;
        uint gidx = col / FC_GROUP_SIZE;
        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

#define D16(w) \
    (x0*float((w)[0]&0x000Fu)+x1*float((w)[0]&0x00F0u)+x2*float((w)[0]&0x0F00u)+x3*float((w)[0]&0xF000u) \
    +x4*float((w)[1]&0x000Fu)+x5*float((w)[1]&0x00F0u)+x6*float((w)[1]&0x0F00u)+x7*float((w)[1]&0xF000u) \
    +x8*float((w)[2]&0x000Fu)+x9*float((w)[2]&0x00F0u)+x10*float((w)[2]&0x0F00u)+x11*float((w)[2]&0xF000u) \
    +x12*float((w)[3]&0x000Fu)+x13*float((w)[3]&0x00F0u)+x14*float((w)[3]&0x0F00u)+x15*float((w)[3]&0xF000u))

        { g0+=float(gs0[gidx])*D16(gr0+j*4)+float(gb0[gidx])*xsum; u0+=float(us0[gidx])*D16(ur0+j*4)+float(ub0[gidx])*xsum; }
        if (sgValidRows>1) { g1+=float(gs1[gidx])*D16(gr1+j*4)+float(gb1[gidx])*xsum; u1+=float(us1[gidx])*D16(ur1+j*4)+float(ub1[gidx])*xsum; }
        if (sgValidRows>2) { g2+=float(gs2[gidx])*D16(gr2+j*4)+float(gb2[gidx])*xsum; u2+=float(us2[gidx])*D16(ur2+j*4)+float(ub2[gidx])*xsum; }
        if (sgValidRows>3) { g3+=float(gs3[gidx])*D16(gr3+j*4)+float(gb3[gidx])*xsum; u3+=float(us3[gidx])*D16(ur3+j*4)+float(ub3[gidx])*xsum; }
#undef D16
    }

    g0=simd_sum(g0); g1=simd_sum(g1); g2=simd_sum(g2); g3=simd_sum(g3);
    u0=simd_sum(u0); u1=simd_sum(u1); u2=simd_sum(u2); u3=simd_sum(u3);

    if (simd_lane == 0) {
        output[sgBaseRow] = agent_geglu_product(g0, u0);
        if (sgValidRows>1) output[sgBaseRow+1] = agent_geglu_product(g1, u1);
        if (sgValidRows>2) output[sgBaseRow+2] = agent_geglu_product(g2, u2);
        if (sgValidRows>3) output[sgBaseRow+3] = agent_geglu_product(g3, u3);
    }
}

#define AGENT_AFFINE_DOT16(w) \
    (x0*float((w)[0]&0x000Fu)+x1*float((w)[0]&0x00F0u)+x2*float((w)[0]&0x0F00u)+x3*float((w)[0]&0xF000u) \
    +x4*float((w)[1]&0x000Fu)+x5*float((w)[1]&0x00F0u)+x6*float((w)[1]&0x0F00u)+x7*float((w)[1]&0xF000u) \
    +x8*float((w)[2]&0x000Fu)+x9*float((w)[2]&0x00F0u)+x10*float((w)[2]&0x0F00u)+x11*float((w)[2]&0xF000u) \
    +x12*float((w)[3]&0x000Fu)+x13*float((w)[3]&0x00F0u)+x14*float((w)[3]&0x0F00u)+x15*float((w)[3]&0xF000u))

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE, uint ROWS_PER_TG_LOCAL>
inline void fused_dual_affine_matvec_fixed_two_rows_per_simdgroup(
    device const uint8_t* w1_weights,
    device const half*    w1_scales,
    device const half*    w1_biases,
    device const uint8_t* w2_weights,
    device const half*    w2_scales,
    device const half*    w2_biases,
    device const half*    input,
    device half*          output1,
    device half*          output2,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 2;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;

    float a0 = 0.0f, a1 = 0.0f;
    float b0 = 0.0f, b1 = 0.0f;

    device const uint16_t* w1r0 = (device const uint16_t*)(w1_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* w1r1 = w1r0 + U16_PER_ROW;
    device const uint16_t* w2r0 = (device const uint16_t*)(w2_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* w2r1 = w2r0 + U16_PER_ROW;

    device const half* w1s0 = w1_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* w1s1 = w1s0 + NUM_COL_GROUPS;
    device const half* w1b0 = w1_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* w1b1 = w1b0 + NUM_COL_GROUPS;
    device const half* w2s0 = w2_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* w2s1 = w2s0 + NUM_COL_GROUPS;
    device const half* w2b0 = w2_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* w2b1 = w2b0 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;
        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

        a0 += float(w1s0[g]) * AGENT_AFFINE_DOT16(w1r0 + j * 4) + float(w1b0[g]) * xsum;
        b0 += float(w2s0[g]) * AGENT_AFFINE_DOT16(w2r0 + j * 4) + float(w2b0[g]) * xsum;
        a1 += float(w1s1[g]) * AGENT_AFFINE_DOT16(w1r1 + j * 4) + float(w1b1[g]) * xsum;
        b1 += float(w2s1[g]) * AGENT_AFFINE_DOT16(w2r1 + j * 4) + float(w2b1[g]) * xsum;
    }

    a0 = simd_sum(a0); a1 = simd_sum(a1);
    b0 = simd_sum(b0); b1 = simd_sum(b1);

    if (simd_lane == 0) {
        output1[sgBaseRow] = half(a0); output2[sgBaseRow] = half(b0);
        output1[sgBaseRow + 1] = half(a1); output2[sgBaseRow + 1] = half(b1);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void affine_matvec_fixed(
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device const half*    input,
    device half*          output,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;

    uint baseRow = tgid * ROWS_PER_TG;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * HALF_COLS);
    device const uint16_t* r1 = r0 + U16_PER_ROW;
    device const uint16_t* r2 = r1 + U16_PER_ROW;
    device const uint16_t* r3 = r2 + U16_PER_ROW;

    device const half* sc0 = scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* sc1 = sc0 + NUM_COL_GROUPS;
    device const half* sc2 = sc1 + NUM_COL_GROUPS;
    device const half* sc3 = sc2 + NUM_COL_GROUPS;
    device const half* bi0 = biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* bi1 = bi0 + NUM_COL_GROUPS;
    device const half* bi2 = bi1 + NUM_COL_GROUPS;
    device const half* bi3 = bi2 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;

        device const half* xp = input + col;
        float x0  = float(xp[0]);
        float x1  = float(xp[1]);
        float x2  = float(xp[2]);
        float x3  = float(xp[3]);
        float x4  = float(xp[4]);
        float x5  = float(xp[5]);
        float x6  = float(xp[6]);
        float x7  = float(xp[7]);
        float x8  = float(xp[8]);
        float x9  = float(xp[9]);
        float x10 = float(xp[10]);
        float x11 = float(xp[11]);
        float x12 = float(xp[12]);
        float x13 = float(xp[13]);
        float x14 = float(xp[14]);
        float x15 = float(xp[15]);

        float xsum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
                   + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15;

        x1  *= (1.0f / 16.0f);
        x2  *= (1.0f / 256.0f);
        x3  *= (1.0f / 4096.0f);
        x5  *= (1.0f / 16.0f);
        x6  *= (1.0f / 256.0f);
        x7  *= (1.0f / 4096.0f);
        x9  *= (1.0f / 16.0f);
        x10 *= (1.0f / 256.0f);
        x11 *= (1.0f / 4096.0f);
        x13 *= (1.0f / 16.0f);
        x14 *= (1.0f / 256.0f);
        x15 *= (1.0f / 4096.0f);

        acc0 += float(sc0[g]) * AGENT_AFFINE_DOT16(r0 + j * 4) + float(bi0[g]) * xsum;
        acc1 += float(sc1[g]) * AGENT_AFFINE_DOT16(r1 + j * 4) + float(bi1[g]) * xsum;
        acc2 += float(sc2[g]) * AGENT_AFFINE_DOT16(r2 + j * 4) + float(bi2[g]) * xsum;
        acc3 += float(sc3[g]) * AGENT_AFFINE_DOT16(r3 + j * 4) + float(bi3[g]) * xsum;
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (simd_lane == 0) {
        output[sgBaseRow] = half(acc0);
        output[sgBaseRow + 1] = half(acc1);
        output[sgBaseRow + 2] = half(acc2);
        output[sgBaseRow + 3] = half(acc3);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void affine_matvec_fixed_rows4(
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device const half*    input,
    device half*          output,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 2;
    constexpr uint ROWS_PER_TG_LOCAL = 4;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;

    float acc0 = 0.0f, acc1 = 0.0f;

    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * HALF_COLS);
    device const uint16_t* r1 = r0 + U16_PER_ROW;

    device const half* sc0 = scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* sc1 = sc0 + NUM_COL_GROUPS;
    device const half* bi0 = biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* bi1 = bi0 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;

        device const half* xp = input + col;
        float x0  = float(xp[0]);
        float x1  = float(xp[1]);
        float x2  = float(xp[2]);
        float x3  = float(xp[3]);
        float x4  = float(xp[4]);
        float x5  = float(xp[5]);
        float x6  = float(xp[6]);
        float x7  = float(xp[7]);
        float x8  = float(xp[8]);
        float x9  = float(xp[9]);
        float x10 = float(xp[10]);
        float x11 = float(xp[11]);
        float x12 = float(xp[12]);
        float x13 = float(xp[13]);
        float x14 = float(xp[14]);
        float x15 = float(xp[15]);

        float xsum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
                   + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15;

        x1  *= (1.0f / 16.0f);
        x2  *= (1.0f / 256.0f);
        x3  *= (1.0f / 4096.0f);
        x5  *= (1.0f / 16.0f);
        x6  *= (1.0f / 256.0f);
        x7  *= (1.0f / 4096.0f);
        x9  *= (1.0f / 16.0f);
        x10 *= (1.0f / 256.0f);
        x11 *= (1.0f / 4096.0f);
        x13 *= (1.0f / 16.0f);
        x14 *= (1.0f / 256.0f);
        x15 *= (1.0f / 4096.0f);

        acc0 += float(sc0[g]) * AGENT_AFFINE_DOT16(r0 + j * 4) + float(bi0[g]) * xsum;
        acc1 += float(sc1[g]) * AGENT_AFFINE_DOT16(r1 + j * 4) + float(bi1[g]) * xsum;
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (simd_lane == 0) {
        output[sgBaseRow] = half(acc0);
        output[sgBaseRow + 1] = half(acc1);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void affine_matvec_fixed_rows8(
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device const half*    input,
    device half*          output,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 4;
    constexpr uint ROWS_PER_TG_LOCAL = 8;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * HALF_COLS);
    device const uint16_t* r1 = r0 + U16_PER_ROW;
    device const uint16_t* r2 = r1 + U16_PER_ROW;
    device const uint16_t* r3 = r2 + U16_PER_ROW;

    device const half* sc0 = scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* sc1 = sc0 + NUM_COL_GROUPS;
    device const half* sc2 = sc1 + NUM_COL_GROUPS;
    device const half* sc3 = sc2 + NUM_COL_GROUPS;
    device const half* bi0 = biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* bi1 = bi0 + NUM_COL_GROUPS;
    device const half* bi2 = bi1 + NUM_COL_GROUPS;
    device const half* bi3 = bi2 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;

        device const half* xp = input + col;
        float x0  = float(xp[0]);
        float x1  = float(xp[1]);
        float x2  = float(xp[2]);
        float x3  = float(xp[3]);
        float x4  = float(xp[4]);
        float x5  = float(xp[5]);
        float x6  = float(xp[6]);
        float x7  = float(xp[7]);
        float x8  = float(xp[8]);
        float x9  = float(xp[9]);
        float x10 = float(xp[10]);
        float x11 = float(xp[11]);
        float x12 = float(xp[12]);
        float x13 = float(xp[13]);
        float x14 = float(xp[14]);
        float x15 = float(xp[15]);

        float xsum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
                   + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15;

        x1  *= (1.0f / 16.0f);
        x2  *= (1.0f / 256.0f);
        x3  *= (1.0f / 4096.0f);
        x5  *= (1.0f / 16.0f);
        x6  *= (1.0f / 256.0f);
        x7  *= (1.0f / 4096.0f);
        x9  *= (1.0f / 16.0f);
        x10 *= (1.0f / 256.0f);
        x11 *= (1.0f / 4096.0f);
        x13 *= (1.0f / 16.0f);
        x14 *= (1.0f / 256.0f);
        x15 *= (1.0f / 4096.0f);

        acc0 += float(sc0[g]) * AGENT_AFFINE_DOT16(r0 + j * 4) + float(bi0[g]) * xsum;
        acc1 += float(sc1[g]) * AGENT_AFFINE_DOT16(r1 + j * 4) + float(bi1[g]) * xsum;
        acc2 += float(sc2[g]) * AGENT_AFFINE_DOT16(r2 + j * 4) + float(bi2[g]) * xsum;
        acc3 += float(sc3[g]) * AGENT_AFFINE_DOT16(r3 + j * 4) + float(bi3[g]) * xsum;
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (simd_lane == 0) {
        output[sgBaseRow] = half(acc0);
        output[sgBaseRow + 1] = half(acc1);
        output[sgBaseRow + 2] = half(acc2);
        output[sgBaseRow + 3] = half(acc3);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void affine_matvec_fixed_rows4_sg1(
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device const half*    input,
    device half*          output,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_TG_LOCAL = 4;

    uint row = tgid * ROWS_PER_TG_LOCAL + simd_group;
    float acc = 0.0f;

    device const uint16_t* r = (device const uint16_t*)(weights + row * HALF_COLS);
    device const half* sc = scales + row * NUM_COL_GROUPS;
    device const half* bi = biases + row * NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;

        device const half* xp = input + col;
        float x0  = float(xp[0]);
        float x1  = float(xp[1]);
        float x2  = float(xp[2]);
        float x3  = float(xp[3]);
        float x4  = float(xp[4]);
        float x5  = float(xp[5]);
        float x6  = float(xp[6]);
        float x7  = float(xp[7]);
        float x8  = float(xp[8]);
        float x9  = float(xp[9]);
        float x10 = float(xp[10]);
        float x11 = float(xp[11]);
        float x12 = float(xp[12]);
        float x13 = float(xp[13]);
        float x14 = float(xp[14]);
        float x15 = float(xp[15]);

        float xsum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
                   + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15;

        x1  *= (1.0f / 16.0f);
        x2  *= (1.0f / 256.0f);
        x3  *= (1.0f / 4096.0f);
        x5  *= (1.0f / 16.0f);
        x6  *= (1.0f / 256.0f);
        x7  *= (1.0f / 4096.0f);
        x9  *= (1.0f / 16.0f);
        x10 *= (1.0f / 256.0f);
        x11 *= (1.0f / 4096.0f);
        x13 *= (1.0f / 16.0f);
        x14 *= (1.0f / 256.0f);
        x15 *= (1.0f / 4096.0f);

        acc += float(sc[g]) * AGENT_AFFINE_DOT16(r + j * 4) + float(bi[g]) * xsum;
    }

    acc = simd_sum(acc);

    if (simd_lane == 0) {
        output[row] = half(acc);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void norm_scale_affine_matvec_fixed_rows4(
    device const float*    scalePtr,
    device const half*     normInput,
    device const half*     normWeight,
    device half*           normOutput,
    device const uint8_t*  weights,
    device const half*     scales,
    device const half*     biases,
    device half*           output,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 2;
    constexpr uint ROWS_PER_TG_LOCAL = 4;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;
    float rs = scalePtr[0];

    float acc0 = 0.0f, acc1 = 0.0f;

    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * HALF_COLS);
    device const uint16_t* r1 = r0 + U16_PER_ROW;

    device const half* sc0 = scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* sc1 = sc0 + NUM_COL_GROUPS;
    device const half* bi0 = biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* bi1 = bi0 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;

        device const half* xp = normInput + col;
        device const half* wp = normWeight + col;

        float x0  = float(xp[0])  * rs * (1.0f + float(wp[0]));
        float x1  = float(xp[1])  * rs * (1.0f + float(wp[1]));
        float x2  = float(xp[2])  * rs * (1.0f + float(wp[2]));
        float x3  = float(xp[3])  * rs * (1.0f + float(wp[3]));
        float x4  = float(xp[4])  * rs * (1.0f + float(wp[4]));
        float x5  = float(xp[5])  * rs * (1.0f + float(wp[5]));
        float x6  = float(xp[6])  * rs * (1.0f + float(wp[6]));
        float x7  = float(xp[7])  * rs * (1.0f + float(wp[7]));
        float x8  = float(xp[8])  * rs * (1.0f + float(wp[8]));
        float x9  = float(xp[9])  * rs * (1.0f + float(wp[9]));
        float x10 = float(xp[10]) * rs * (1.0f + float(wp[10]));
        float x11 = float(xp[11]) * rs * (1.0f + float(wp[11]));
        float x12 = float(xp[12]) * rs * (1.0f + float(wp[12]));
        float x13 = float(xp[13]) * rs * (1.0f + float(wp[13]));
        float x14 = float(xp[14]) * rs * (1.0f + float(wp[14]));
        float x15 = float(xp[15]) * rs * (1.0f + float(wp[15]));

        half hx0 = half(x0);
        half hx1 = half(x1);
        half hx2 = half(x2);
        half hx3 = half(x3);
        half hx4 = half(x4);
        half hx5 = half(x5);
        half hx6 = half(x6);
        half hx7 = half(x7);
        half hx8 = half(x8);
        half hx9 = half(x9);
        half hx10 = half(x10);
        half hx11 = half(x11);
        half hx12 = half(x12);
        half hx13 = half(x13);
        half hx14 = half(x14);
        half hx15 = half(x15);

        if (tgid == 0 && simd_group == 0) {
            normOutput[col + 0] = hx0;
            normOutput[col + 1] = hx1;
            normOutput[col + 2] = hx2;
            normOutput[col + 3] = hx3;
            normOutput[col + 4] = hx4;
            normOutput[col + 5] = hx5;
            normOutput[col + 6] = hx6;
            normOutput[col + 7] = hx7;
            normOutput[col + 8] = hx8;
            normOutput[col + 9] = hx9;
            normOutput[col + 10] = hx10;
            normOutput[col + 11] = hx11;
            normOutput[col + 12] = hx12;
            normOutput[col + 13] = hx13;
            normOutput[col + 14] = hx14;
            normOutput[col + 15] = hx15;
        }

        x0 = float(hx0);
        x1 = float(hx1);
        x2 = float(hx2);
        x3 = float(hx3);
        x4 = float(hx4);
        x5 = float(hx5);
        x6 = float(hx6);
        x7 = float(hx7);
        x8 = float(hx8);
        x9 = float(hx9);
        x10 = float(hx10);
        x11 = float(hx11);
        x12 = float(hx12);
        x13 = float(hx13);
        x14 = float(hx14);
        x15 = float(hx15);

        float xsum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
                   + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15;

        x1  *= (1.0f / 16.0f);
        x2  *= (1.0f / 256.0f);
        x3  *= (1.0f / 4096.0f);
        x5  *= (1.0f / 16.0f);
        x6  *= (1.0f / 256.0f);
        x7  *= (1.0f / 4096.0f);
        x9  *= (1.0f / 16.0f);
        x10 *= (1.0f / 256.0f);
        x11 *= (1.0f / 4096.0f);
        x13 *= (1.0f / 16.0f);
        x14 *= (1.0f / 256.0f);
        x15 *= (1.0f / 4096.0f);

#define NORM_SCALE_AFFINE_DOT16(w) \
    ( x0  * float((w)[0] & 0x000Fu) + x1  * float((w)[0] & 0x00F0u) \
    + x2  * float((w)[0] & 0x0F00u) + x3  * float((w)[0] & 0xF000u) \
    + x4  * float((w)[1] & 0x000Fu) + x5  * float((w)[1] & 0x00F0u) \
    + x6  * float((w)[1] & 0x0F00u) + x7  * float((w)[1] & 0xF000u) \
    + x8  * float((w)[2] & 0x000Fu) + x9  * float((w)[2] & 0x00F0u) \
    + x10 * float((w)[2] & 0x0F00u) + x11 * float((w)[2] & 0xF000u) \
    + x12 * float((w)[3] & 0x000Fu) + x13 * float((w)[3] & 0x00F0u) \
    + x14 * float((w)[3] & 0x0F00u) + x15 * float((w)[3] & 0xF000u) )

        acc0 += float(sc0[g]) * NORM_SCALE_AFFINE_DOT16(r0 + j * 4) + float(bi0[g]) * xsum;
        acc1 += float(sc1[g]) * NORM_SCALE_AFFINE_DOT16(r1 + j * 4) + float(bi1[g]) * xsum;

#undef NORM_SCALE_AFFINE_DOT16
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (simd_lane == 0) {
        output[sgBaseRow] = half(acc0);
        output[sgBaseRow + 1] = half(acc1);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void norm_add_scale_affine_matvec_fixed_rows4(
    device const float*    scalePtr,
    device const half*     normInput,
    device const half*     normWeight,
    device const half*     residual,
    device half*           normOutput,
    device const uint8_t*  weights,
    device const half*     scales,
    device const half*     biases,
    device half*           output,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 2;
    constexpr uint ROWS_PER_TG_LOCAL = 4;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;
    float rs = scalePtr[0];

    float acc0 = 0.0f, acc1 = 0.0f;

    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * HALF_COLS);
    device const uint16_t* r1 = r0 + U16_PER_ROW;

    device const half* sc0 = scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* sc1 = sc0 + NUM_COL_GROUPS;
    device const half* bi0 = biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* bi1 = bi0 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;

        device const half* xp = normInput + col;
        device const half* wp = normWeight + col;
        device const half* rp = residual + col;

        half hx0 = half(float(xp[0]) * rs * (1.0f + float(wp[0])));
        half hx1 = half(float(xp[1]) * rs * (1.0f + float(wp[1])));
        half hx2 = half(float(xp[2]) * rs * (1.0f + float(wp[2])));
        half hx3 = half(float(xp[3]) * rs * (1.0f + float(wp[3])));
        half hx4 = half(float(xp[4]) * rs * (1.0f + float(wp[4])));
        half hx5 = half(float(xp[5]) * rs * (1.0f + float(wp[5])));
        half hx6 = half(float(xp[6]) * rs * (1.0f + float(wp[6])));
        half hx7 = half(float(xp[7]) * rs * (1.0f + float(wp[7])));
        half hx8 = half(float(xp[8]) * rs * (1.0f + float(wp[8])));
        half hx9 = half(float(xp[9]) * rs * (1.0f + float(wp[9])));
        half hx10 = half(float(xp[10]) * rs * (1.0f + float(wp[10])));
        half hx11 = half(float(xp[11]) * rs * (1.0f + float(wp[11])));
        half hx12 = half(float(xp[12]) * rs * (1.0f + float(wp[12])));
        half hx13 = half(float(xp[13]) * rs * (1.0f + float(wp[13])));
        half hx14 = half(float(xp[14]) * rs * (1.0f + float(wp[14])));
        half hx15 = half(float(xp[15]) * rs * (1.0f + float(wp[15])));

        half ax0 = half(hx0 + rp[0]);
        half ax1 = half(hx1 + rp[1]);
        half ax2 = half(hx2 + rp[2]);
        half ax3 = half(hx3 + rp[3]);
        half ax4 = half(hx4 + rp[4]);
        half ax5 = half(hx5 + rp[5]);
        half ax6 = half(hx6 + rp[6]);
        half ax7 = half(hx7 + rp[7]);
        half ax8 = half(hx8 + rp[8]);
        half ax9 = half(hx9 + rp[9]);
        half ax10 = half(hx10 + rp[10]);
        half ax11 = half(hx11 + rp[11]);
        half ax12 = half(hx12 + rp[12]);
        half ax13 = half(hx13 + rp[13]);
        half ax14 = half(hx14 + rp[14]);
        half ax15 = half(hx15 + rp[15]);

        if (tgid == 0 && simd_group == 0) {
            normOutput[col + 0] = ax0;
            normOutput[col + 1] = ax1;
            normOutput[col + 2] = ax2;
            normOutput[col + 3] = ax3;
            normOutput[col + 4] = ax4;
            normOutput[col + 5] = ax5;
            normOutput[col + 6] = ax6;
            normOutput[col + 7] = ax7;
            normOutput[col + 8] = ax8;
            normOutput[col + 9] = ax9;
            normOutput[col + 10] = ax10;
            normOutput[col + 11] = ax11;
            normOutput[col + 12] = ax12;
            normOutput[col + 13] = ax13;
            normOutput[col + 14] = ax14;
            normOutput[col + 15] = ax15;
        }

        float x0 = float(ax0);
        float x1 = float(ax1);
        float x2 = float(ax2);
        float x3 = float(ax3);
        float x4 = float(ax4);
        float x5 = float(ax5);
        float x6 = float(ax6);
        float x7 = float(ax7);
        float x8 = float(ax8);
        float x9 = float(ax9);
        float x10 = float(ax10);
        float x11 = float(ax11);
        float x12 = float(ax12);
        float x13 = float(ax13);
        float x14 = float(ax14);
        float x15 = float(ax15);

        float xsum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
                   + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15;

        x1  *= (1.0f / 16.0f);
        x2  *= (1.0f / 256.0f);
        x3  *= (1.0f / 4096.0f);
        x5  *= (1.0f / 16.0f);
        x6  *= (1.0f / 256.0f);
        x7  *= (1.0f / 4096.0f);
        x9  *= (1.0f / 16.0f);
        x10 *= (1.0f / 256.0f);
        x11 *= (1.0f / 4096.0f);
        x13 *= (1.0f / 16.0f);
        x14 *= (1.0f / 256.0f);
        x15 *= (1.0f / 4096.0f);

#define NORM_ADD_SCALE_AFFINE_DOT16(w) \
    ( x0  * float((w)[0] & 0x000Fu) + x1  * float((w)[0] & 0x00F0u) \
    + x2  * float((w)[0] & 0x0F00u) + x3  * float((w)[0] & 0xF000u) \
    + x4  * float((w)[1] & 0x000Fu) + x5  * float((w)[1] & 0x00F0u) \
    + x6  * float((w)[1] & 0x0F00u) + x7  * float((w)[1] & 0xF000u) \
    + x8  * float((w)[2] & 0x000Fu) + x9  * float((w)[2] & 0x00F0u) \
    + x10 * float((w)[2] & 0x0F00u) + x11 * float((w)[2] & 0xF000u) \
    + x12 * float((w)[3] & 0x000Fu) + x13 * float((w)[3] & 0x00F0u) \
    + x14 * float((w)[3] & 0x0F00u) + x15 * float((w)[3] & 0xF000u) )

        acc0 += float(sc0[g]) * NORM_ADD_SCALE_AFFINE_DOT16(r0 + j * 4) + float(bi0[g]) * xsum;
        acc1 += float(sc1[g]) * NORM_ADD_SCALE_AFFINE_DOT16(r1 + j * 4) + float(bi1[g]) * xsum;

#undef NORM_ADD_SCALE_AFFINE_DOT16
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (simd_lane == 0) {
        output[sgBaseRow] = half(acc0);
        output[sgBaseRow + 1] = half(acc1);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void norm_scale_affine_matvec_fixed_rows8(
    device const float*    scalePtr,
    device const half*     normInput,
    device const half*     normWeight,
    device half*           normOutput,
    device const uint8_t*  weights,
    device const half*     scales,
    device const half*     biases,
    device half*           output,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 4;
    constexpr uint ROWS_PER_TG_LOCAL = 8;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;
    float rs = scalePtr[0];

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * HALF_COLS);
    device const uint16_t* r1 = r0 + U16_PER_ROW;
    device const uint16_t* r2 = r1 + U16_PER_ROW;
    device const uint16_t* r3 = r2 + U16_PER_ROW;

    device const half* sc0 = scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* sc1 = sc0 + NUM_COL_GROUPS;
    device const half* sc2 = sc1 + NUM_COL_GROUPS;
    device const half* sc3 = sc2 + NUM_COL_GROUPS;
    device const half* bi0 = biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* bi1 = bi0 + NUM_COL_GROUPS;
    device const half* bi2 = bi1 + NUM_COL_GROUPS;
    device const half* bi3 = bi2 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;

        device const half* xp = normInput + col;
        device const half* wp = normWeight + col;

        float x0  = float(xp[0])  * rs * (1.0f + float(wp[0]));
        float x1  = float(xp[1])  * rs * (1.0f + float(wp[1]));
        float x2  = float(xp[2])  * rs * (1.0f + float(wp[2]));
        float x3  = float(xp[3])  * rs * (1.0f + float(wp[3]));
        float x4  = float(xp[4])  * rs * (1.0f + float(wp[4]));
        float x5  = float(xp[5])  * rs * (1.0f + float(wp[5]));
        float x6  = float(xp[6])  * rs * (1.0f + float(wp[6]));
        float x7  = float(xp[7])  * rs * (1.0f + float(wp[7]));
        float x8  = float(xp[8])  * rs * (1.0f + float(wp[8]));
        float x9  = float(xp[9])  * rs * (1.0f + float(wp[9]));
        float x10 = float(xp[10]) * rs * (1.0f + float(wp[10]));
        float x11 = float(xp[11]) * rs * (1.0f + float(wp[11]));
        float x12 = float(xp[12]) * rs * (1.0f + float(wp[12]));
        float x13 = float(xp[13]) * rs * (1.0f + float(wp[13]));
        float x14 = float(xp[14]) * rs * (1.0f + float(wp[14]));
        float x15 = float(xp[15]) * rs * (1.0f + float(wp[15]));

        half hx0 = half(x0);
        half hx1 = half(x1);
        half hx2 = half(x2);
        half hx3 = half(x3);
        half hx4 = half(x4);
        half hx5 = half(x5);
        half hx6 = half(x6);
        half hx7 = half(x7);
        half hx8 = half(x8);
        half hx9 = half(x9);
        half hx10 = half(x10);
        half hx11 = half(x11);
        half hx12 = half(x12);
        half hx13 = half(x13);
        half hx14 = half(x14);
        half hx15 = half(x15);

        if (tgid == 0 && simd_group == 0) {
            normOutput[col + 0] = hx0;
            normOutput[col + 1] = hx1;
            normOutput[col + 2] = hx2;
            normOutput[col + 3] = hx3;
            normOutput[col + 4] = hx4;
            normOutput[col + 5] = hx5;
            normOutput[col + 6] = hx6;
            normOutput[col + 7] = hx7;
            normOutput[col + 8] = hx8;
            normOutput[col + 9] = hx9;
            normOutput[col + 10] = hx10;
            normOutput[col + 11] = hx11;
            normOutput[col + 12] = hx12;
            normOutput[col + 13] = hx13;
            normOutput[col + 14] = hx14;
            normOutput[col + 15] = hx15;
        }

        x0 = float(hx0);
        x1 = float(hx1);
        x2 = float(hx2);
        x3 = float(hx3);
        x4 = float(hx4);
        x5 = float(hx5);
        x6 = float(hx6);
        x7 = float(hx7);
        x8 = float(hx8);
        x9 = float(hx9);
        x10 = float(hx10);
        x11 = float(hx11);
        x12 = float(hx12);
        x13 = float(hx13);
        x14 = float(hx14);
        x15 = float(hx15);

        float xsum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
                   + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15;

        x1  *= (1.0f / 16.0f);
        x2  *= (1.0f / 256.0f);
        x3  *= (1.0f / 4096.0f);
        x5  *= (1.0f / 16.0f);
        x6  *= (1.0f / 256.0f);
        x7  *= (1.0f / 4096.0f);
        x9  *= (1.0f / 16.0f);
        x10 *= (1.0f / 256.0f);
        x11 *= (1.0f / 4096.0f);
        x13 *= (1.0f / 16.0f);
        x14 *= (1.0f / 256.0f);
        x15 *= (1.0f / 4096.0f);

#define NORM_SCALE_AFFINE_ROWS8_DOT16(w) \
    ( x0  * float((w)[0] & 0x000Fu) + x1  * float((w)[0] & 0x00F0u) \
    + x2  * float((w)[0] & 0x0F00u) + x3  * float((w)[0] & 0xF000u) \
    + x4  * float((w)[1] & 0x000Fu) + x5  * float((w)[1] & 0x00F0u) \
    + x6  * float((w)[1] & 0x0F00u) + x7  * float((w)[1] & 0xF000u) \
    + x8  * float((w)[2] & 0x000Fu) + x9  * float((w)[2] & 0x00F0u) \
    + x10 * float((w)[2] & 0x0F00u) + x11 * float((w)[2] & 0xF000u) \
    + x12 * float((w)[3] & 0x000Fu) + x13 * float((w)[3] & 0x00F0u) \
    + x14 * float((w)[3] & 0x0F00u) + x15 * float((w)[3] & 0xF000u) )

        acc0 += float(sc0[g]) * NORM_SCALE_AFFINE_ROWS8_DOT16(r0 + j * 4) + float(bi0[g]) * xsum;
        acc1 += float(sc1[g]) * NORM_SCALE_AFFINE_ROWS8_DOT16(r1 + j * 4) + float(bi1[g]) * xsum;
        acc2 += float(sc2[g]) * NORM_SCALE_AFFINE_ROWS8_DOT16(r2 + j * 4) + float(bi2[g]) * xsum;
        acc3 += float(sc3[g]) * NORM_SCALE_AFFINE_ROWS8_DOT16(r3 + j * 4) + float(bi3[g]) * xsum;

#undef NORM_SCALE_AFFINE_ROWS8_DOT16
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (simd_lane == 0) {
        output[sgBaseRow] = half(acc0);
        output[sgBaseRow + 1] = half(acc1);
        output[sgBaseRow + 2] = half(acc2);
        output[sgBaseRow + 3] = half(acc3);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE, uint BATCH_TILE>
inline void affine_matvec_fixed_batched_full(
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device const half*    input,
    device half*          output,
    uint                  actualBatch,
    threadgroup half*     Xs,
    threadgroup float*    Ws,
    uint2 tgid,
    uint tid,
    uint simd_lane,
    uint simd_group
) {
    agent_affine_qmm_fixed_batched_full<FIXED_ROWS, FIXED_COLS, FIXED_GROUP_SIZE, BATCH_TILE>(
        weights, scales, biases, input, output,
        actualBatch,
        Xs, Ws,
        tgid, tid, simd_lane, simd_group
    );
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE, uint BATCH_TILE>
inline void affine_matvec_fixed_batched(
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device const half*    input,
    device half*          output,
    constant uint&        actualBatch,
    uint2 tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;

    uint baseRow = tgid.x * ROWS_PER_TG;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint batchBase = tgid.y * BATCH_TILE;
    bool fullTile = batchBase + BATCH_TILE <= actualBatch;

    float acc[BATCH_TILE][ROWS_PER_SG];
    for (uint b = 0; b < BATCH_TILE; b++) {
        for (uint r = 0; r < ROWS_PER_SG; r++) {
            acc[b][r] = 0.0f;
        }
    }

    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * HALF_COLS);
    device const uint16_t* r1 = r0 + U16_PER_ROW;
    device const uint16_t* r2 = r1 + U16_PER_ROW;
    device const uint16_t* r3 = r2 + U16_PER_ROW;

    device const half* sc0 = scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* sc1 = sc0 + NUM_COL_GROUPS;
    device const half* sc2 = sc1 + NUM_COL_GROUPS;
    device const half* sc3 = sc2 + NUM_COL_GROUPS;
    device const half* bi0 = biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* bi1 = bi0 + NUM_COL_GROUPS;
    device const half* bi2 = bi1 + NUM_COL_GROUPS;
    device const half* bi3 = bi2 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;
        device const uint16_t* w0 = r0 + j * 4;
        device const uint16_t* w1 = r1 + j * 4;
        device const uint16_t* w2 = r2 + j * 4;
        device const uint16_t* w3 = r3 + j * 4;

        for (uint b = 0; b < BATCH_TILE; b++) {
            if (!fullTile && batchBase + b >= actualBatch) continue;
            device const half* xp = input + (batchBase + b) * FIXED_COLS + col;
            float x0  = float(xp[0]);
            float x1  = float(xp[1]);
            float x2  = float(xp[2]);
            float x3  = float(xp[3]);
            float x4  = float(xp[4]);
            float x5  = float(xp[5]);
            float x6  = float(xp[6]);
            float x7  = float(xp[7]);
            float x8  = float(xp[8]);
            float x9  = float(xp[9]);
            float x10 = float(xp[10]);
            float x11 = float(xp[11]);
            float x12 = float(xp[12]);
            float x13 = float(xp[13]);
            float x14 = float(xp[14]);
            float x15 = float(xp[15]);

            float xsum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
                       + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15;

            x1  *= (1.0f / 16.0f);
            x2  *= (1.0f / 256.0f);
            x3  *= (1.0f / 4096.0f);
            x5  *= (1.0f / 16.0f);
            x6  *= (1.0f / 256.0f);
            x7  *= (1.0f / 4096.0f);
            x9  *= (1.0f / 16.0f);
            x10 *= (1.0f / 256.0f);
            x11 *= (1.0f / 4096.0f);
            x13 *= (1.0f / 16.0f);
            x14 *= (1.0f / 256.0f);
            x15 *= (1.0f / 4096.0f);

            acc[b][0] += float(sc0[g]) * AGENT_AFFINE_DOT16(w0) + float(bi0[g]) * xsum;
            acc[b][1] += float(sc1[g]) * AGENT_AFFINE_DOT16(w1) + float(bi1[g]) * xsum;
            acc[b][2] += float(sc2[g]) * AGENT_AFFINE_DOT16(w2) + float(bi2[g]) * xsum;
            acc[b][3] += float(sc3[g]) * AGENT_AFFINE_DOT16(w3) + float(bi3[g]) * xsum;
        }
    }

    for (uint b = 0; b < BATCH_TILE; b++) {
        acc[b][0] = simd_sum(acc[b][0]);
        acc[b][1] = simd_sum(acc[b][1]);
        acc[b][2] = simd_sum(acc[b][2]);
        acc[b][3] = simd_sum(acc[b][3]);
    }

    if (simd_lane == 0) {
        for (uint b = 0; b < BATCH_TILE; b++) {
            if (!fullTile && batchBase + b >= actualBatch) continue;
            device half* out = output + (batchBase + b) * FIXED_ROWS + sgBaseRow;
            out[0] = half(acc[b][0]);
            out[1] = half(acc[b][1]);
            out[2] = half(acc[b][2]);
            out[3] = half(acc[b][3]);
        }
    }
}

// llama.cpp-style small-batch qmv-ext template for affine_u4.
// Each 8-lane row team owns one output row; 4 row teams per simdgroup;
// 2 simdgroups per TG → 64 threads, 8 output rows per TG.
// Weight dequant lives outside the BATCH_TILE loop so 5 batch rows
// share one weight read instead of paying 5× dequant arithmetic.
// Final reduction is 3 simd_shuffle_down steps over an 8-lane row team
// (vs simd_sum's 32-lane log-reduce).
template <
    uint FIXED_ROWS,
    uint FIXED_COLS,
    uint FIXED_GROUP_SIZE,
    uint BATCH_TILE,
    uint NXPSG,
    uint NSG,
    uint CHPT
>
inline void affine_matvec_fixed_ext_config(
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device const half*    input,
    device half*          output,
    constant uint&        actualBatch,
    uint2 tgid,
    uint  tiisg,
    uint  sgitg
) {
    constexpr uint NYPSG = 32 / NXPSG;
    constexpr uint R0PTG = NYPSG * NSG;
    constexpr uint VALS_PER_CHUNK = 16;
    constexpr uint VPI = CHPT * VALS_PER_CHUNK;
    constexpr uint K_STRIDE = NXPSG * VPI;

    const uint tx = tiisg % NXPSG;
    const uint ty = tiisg / NXPSG;
    const uint outRow = tgid.x * R0PTG + NYPSG * sgitg + ty;
    const uint batchBase = tgid.y * BATCH_TILE;
    const bool fullTile = batchBase + BATCH_TILE <= actualBatch;
    if (outRow >= FIXED_ROWS) return;

    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;

    device const uint16_t* rowBase =
        (device const uint16_t*)(weights + outRow * HALF_COLS);
    device const half* scRow = scales + outRow * NUM_COL_GROUPS;
    device const half* biRow = biases + outRow * NUM_COL_GROUPS;

    uint batchRow[BATCH_TILE];
    #pragma unroll
    for (uint b = 0; b < BATCH_TILE; ++b) {
        batchRow[b] = (fullTile || batchBase + b < actualBatch)
            ? (batchBase + b) : 0;
    }

    float sumf[BATCH_TILE];
    #pragma unroll
    for (uint b = 0; b < BATCH_TILE; ++b) sumf[b] = 0.0f;

    // Derive pointers from k each iteration. Carrying device pointers
    // across the final iteration would leave them past end-of-buffer
    // for the last row/batch and trip Metal's address validation
    // even though those values are never dereferenced.
    for (uint k = tx * VPI; k < FIXED_COLS; k += K_STRIDE) {
        float wbuf[CHPT * VALS_PER_CHUNK];
        #pragma unroll
        for (uint ch = 0; ch < CHPT; ++ch) {
            const uint col = k + ch * VALS_PER_CHUNK;
            const uint g = col / FIXED_GROUP_SIZE;
            const float s = float(scRow[g]);
            const float bi = float(biRow[g]);
            device const uint16_t* wptr =
                rowBase + (k / 4) + ch * (VALS_PER_CHUNK / 4);
            const uint16_t w[4] = { wptr[0], wptr[1], wptr[2], wptr[3] };
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                const uint16_t p = w[i];
                wbuf[ch * 16 + i * 4 + 0] = float((p >>  0) & 0xF) * s + bi;
                wbuf[ch * 16 + i * 4 + 1] = float((p >>  4) & 0xF) * s + bi;
                wbuf[ch * 16 + i * 4 + 2] = float((p >>  8) & 0xF) * s + bi;
                wbuf[ch * 16 + i * 4 + 3] = float((p >> 12) & 0xF) * s + bi;
            }
        }

        #pragma unroll
        for (uint b = 0; b < BATCH_TILE; ++b) {
            if (!fullTile && batchBase + b >= actualBatch) continue;
            device const half* xPtr = input + batchRow[b] * FIXED_COLS + k;
            #pragma unroll
            for (uint i = 0; i < CHPT * VALS_PER_CHUNK; ++i) {
                sumf[b] += wbuf[i] * float(xPtr[i]);
            }
        }
    }

    #pragma unroll
    for (uint b = 0; b < BATCH_TILE; ++b) {
        #pragma unroll
        for (uint offset = NXPSG / 2; offset > 0; offset >>= 1) {
            sumf[b] += simd_shuffle_down(sumf[b], offset);
        }
    }

    if (tx == 0) {
        #pragma unroll
        for (uint b = 0; b < BATCH_TILE; ++b) {
            if (!fullTile && batchBase + b >= actualBatch) continue;
            output[(batchBase + b) * FIXED_ROWS + outRow] = half(sumf[b]);
        }
    }
}

// Four-row specialization of the small-batch affine path. A float4 carries
// the independent accumulators for the batch rows while every dequantized
// weight is broadcast once. The arithmetic order within each row is unchanged
// from affine_matvec_fixed_ext_config, and partial tiles mask only their stores.
template <
    uint FIXED_ROWS,
    uint FIXED_COLS,
    uint FIXED_GROUP_SIZE,
    uint NXPSG,
    uint NSG,
    uint CHPT
>
inline void affine_matvec_fixed_ext_vec4(
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device const half*    input,
    device half*          output,
    constant uint&        actualBatch,
    uint2 tgid,
    uint  tiisg,
    uint  sgitg
) {
    constexpr uint BATCH_TILE = 4;
    constexpr uint NYPSG = 32 / NXPSG;
    constexpr uint R0PTG = NYPSG * NSG;
    constexpr uint VALS_PER_CHUNK = 16;
    constexpr uint VPI = CHPT * VALS_PER_CHUNK;
    constexpr uint K_STRIDE = NXPSG * VPI;
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;

    const uint tx = tiisg % NXPSG;
    const uint ty = tiisg / NXPSG;
    const uint outRow = tgid.x * R0PTG + NYPSG * sgitg + ty;
    const uint batchBase = tgid.y * BATCH_TILE;
    if (outRow >= FIXED_ROWS) return;

    const bool4 active = bool4(
        batchBase + 0 < actualBatch,
        batchBase + 1 < actualBatch,
        batchBase + 2 < actualBatch,
        batchBase + 3 < actualBatch
    );
    const uint4 batchRow = uint4(
        active[0] ? batchBase + 0 : 0,
        active[1] ? batchBase + 1 : 0,
        active[2] ? batchBase + 2 : 0,
        active[3] ? batchBase + 3 : 0
    );

    device const uint16_t* rowBase =
        (device const uint16_t*)(weights + outRow * HALF_COLS);
    device const half* scRow = scales + outRow * NUM_COL_GROUPS;
    device const half* biRow = biases + outRow * NUM_COL_GROUPS;
    float4 sumf = float4(0.0f);

    for (uint k = tx * VPI; k < FIXED_COLS; k += K_STRIDE) {
        float wbuf[CHPT * VALS_PER_CHUNK];
        #pragma unroll
        for (uint ch = 0; ch < CHPT; ++ch) {
            const uint col = k + ch * VALS_PER_CHUNK;
            const uint g = col / FIXED_GROUP_SIZE;
            const float s = float(scRow[g]);
            const float bi = float(biRow[g]);
            device const uint16_t* wptr =
                rowBase + (k / 4) + ch * (VALS_PER_CHUNK / 4);
            const uint16_t w[4] = { wptr[0], wptr[1], wptr[2], wptr[3] };
            #pragma unroll
            for (uint i = 0; i < 4; ++i) {
                const uint16_t p = w[i];
                wbuf[ch * 16 + i * 4 + 0] = float((p >>  0) & 0xF) * s + bi;
                wbuf[ch * 16 + i * 4 + 1] = float((p >>  4) & 0xF) * s + bi;
                wbuf[ch * 16 + i * 4 + 2] = float((p >>  8) & 0xF) * s + bi;
                wbuf[ch * 16 + i * 4 + 3] = float((p >> 12) & 0xF) * s + bi;
            }
        }

        device const half* x0 = input + batchRow[0] * FIXED_COLS + k;
        device const half* x1 = input + batchRow[1] * FIXED_COLS + k;
        device const half* x2 = input + batchRow[2] * FIXED_COLS + k;
        device const half* x3 = input + batchRow[3] * FIXED_COLS + k;
        #pragma unroll
        for (uint i = 0; i < CHPT * VALS_PER_CHUNK; ++i) {
            const float4 x = float4(
                float(x0[i]), float(x1[i]), float(x2[i]), float(x3[i])
            );
            sumf += wbuf[i] * x;
        }
    }

    #pragma unroll
    for (uint offset = NXPSG / 2; offset > 0; offset >>= 1) {
        sumf += simd_shuffle_down(sumf, offset);
    }

    if (tx == 0) {
        #pragma unroll
        for (uint b = 0; b < BATCH_TILE; ++b) {
            if (active[b]) {
                output[(batchBase + b) * FIXED_ROWS + outRow] = half(sumf[b]);
            }
        }
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE, uint BATCH_TILE>
inline void affine_matvec_fixed_ext_b5(
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device const half*    input,
    device half*          output,
    constant uint&        actualBatch,
    uint2 tgid,
    uint  tiisg,
    uint  sgitg
) {
    affine_matvec_fixed_ext_config<
        FIXED_ROWS, FIXED_COLS, FIXED_GROUP_SIZE, BATCH_TILE, 8, 2, 2
    >(
        weights, scales, biases, input, output,
        actualBatch, tgid, tiisg, sgitg
    );
}

// Small-batch SwiGLU companion to affine_matvec_fixed_ext_b5. The two
// projection weights share one dispatch and one input traversal policy while
// retaining the row-team geometry that won over padded QMM at B2/B4. This is
// shape-generic; package-local wrappers provide only compile-time dimensions.
template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE, uint BATCH_TILE>
inline void fused_affine_gate_up_swiglu_fixed_ext(
    device const uint8_t* gateWeights,
    device const half*    gateScales,
    device const half*    gateBiases,
    device const uint8_t* upWeights,
    device const half*    upScales,
    device const half*    upBiases,
    device const half*    input,
    device half*          output,
    constant uint&        actualBatch,
    uint2 tgid,
    uint  tiisg,
    uint  sgitg
) {
    constexpr uint NSG = 2;
    constexpr uint NXPSG = 8;
    constexpr uint NYPSG = 32 / NXPSG;
    constexpr uint R0PTG = NYPSG * NSG;
    constexpr uint VALS_PER_CHUNK = 16;
    constexpr uint CHPT = 2;
    constexpr uint VPI = CHPT * VALS_PER_CHUNK;
    constexpr uint K_STRIDE = NXPSG * VPI;
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;

    const uint tx = tiisg % NXPSG;
    const uint ty = tiisg / NXPSG;
    const uint outRow = tgid.x * R0PTG + NYPSG * sgitg + ty;
    const uint batchBase = tgid.y * BATCH_TILE;
    const bool fullTile = batchBase + BATCH_TILE <= actualBatch;
    if (outRow >= FIXED_ROWS) return;

    uint batchRow[BATCH_TILE];
    #pragma unroll
    for (uint b = 0; b < BATCH_TILE; ++b) {
        batchRow[b] = (fullTile || batchBase + b < actualBatch)
            ? (batchBase + b) : 0;
    }

    float sums[2][BATCH_TILE];
    #pragma unroll
    for (uint projection = 0; projection < 2; ++projection) {
        #pragma unroll
        for (uint b = 0; b < BATCH_TILE; ++b) sums[projection][b] = 0.0f;

        device const uint8_t* weights = projection == 0 ? gateWeights : upWeights;
        device const half* scales = projection == 0 ? gateScales : upScales;
        device const half* biases = projection == 0 ? gateBiases : upBiases;
        device const uint16_t* rowBase =
            (device const uint16_t*)(weights + outRow * HALF_COLS);
        device const half* scRow = scales + outRow * NUM_COL_GROUPS;
        device const half* biRow = biases + outRow * NUM_COL_GROUPS;

        for (uint k = tx * VPI; k < FIXED_COLS; k += K_STRIDE) {
            float wbuf[CHPT * VALS_PER_CHUNK];
            #pragma unroll
            for (uint ch = 0; ch < CHPT; ++ch) {
                const uint col = k + ch * VALS_PER_CHUNK;
                const uint g = col / FIXED_GROUP_SIZE;
                const float s = float(scRow[g]);
                const float bi = float(biRow[g]);
                device const uint16_t* wptr =
                    rowBase + (k / 4) + ch * (VALS_PER_CHUNK / 4);
                const uint16_t w[4] = { wptr[0], wptr[1], wptr[2], wptr[3] };
                #pragma unroll
                for (uint i = 0; i < 4; ++i) {
                    const uint16_t p = w[i];
                    wbuf[ch * 16 + i * 4 + 0] = float((p >>  0) & 0xF) * s + bi;
                    wbuf[ch * 16 + i * 4 + 1] = float((p >>  4) & 0xF) * s + bi;
                    wbuf[ch * 16 + i * 4 + 2] = float((p >>  8) & 0xF) * s + bi;
                    wbuf[ch * 16 + i * 4 + 3] = float((p >> 12) & 0xF) * s + bi;
                }
            }

            #pragma unroll
            for (uint b = 0; b < BATCH_TILE; ++b) {
                if (!fullTile && batchBase + b >= actualBatch) continue;
                device const half* xPtr = input + batchRow[b] * FIXED_COLS + k;
                #pragma unroll
                for (uint i = 0; i < CHPT * VALS_PER_CHUNK; ++i) {
                    sums[projection][b] += wbuf[i] * float(xPtr[i]);
                }
            }
        }

        #pragma unroll
        for (uint b = 0; b < BATCH_TILE; ++b) {
            sums[projection][b] += simd_shuffle_down(sums[projection][b], 4);
            sums[projection][b] += simd_shuffle_down(sums[projection][b], 2);
            sums[projection][b] += simd_shuffle_down(sums[projection][b], 1);
        }
    }

    if (tx == 0) {
        #pragma unroll
        for (uint b = 0; b < BATCH_TILE; ++b) {
            if (!fullTile && batchBase + b >= actualBatch) continue;
            float gate = sums[0][b];
            float up = sums[1][b];
            float silu = gate / (1.0f + exp(-gate));
            output[(batchBase + b) * FIXED_ROWS + outRow] =
                half(clamp(silu * up, -65504.0f, 65504.0f));
        }
    }
}

// Four-row vector-accumulator companion to
// fused_affine_gate_up_swiglu_fixed_ext. The compiler selects this only for a
// four-row small-batch capability; larger prefill tiles keep their QMM route.
template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_affine_gate_up_swiglu_fixed_ext_vec4(
    device const uint8_t* gateWeights,
    device const half*    gateScales,
    device const half*    gateBiases,
    device const uint8_t* upWeights,
    device const half*    upScales,
    device const half*    upBiases,
    device const half*    input,
    device half*          output,
    constant uint&        actualBatch,
    uint2 tgid,
    uint  tiisg,
    uint  sgitg
) {
    constexpr uint BATCH_TILE = 4;
    constexpr uint NSG = 2;
    constexpr uint NXPSG = 8;
    constexpr uint NYPSG = 32 / NXPSG;
    constexpr uint R0PTG = NYPSG * NSG;
    constexpr uint VALS_PER_CHUNK = 16;
    constexpr uint CHPT = 2;
    constexpr uint VPI = CHPT * VALS_PER_CHUNK;
    constexpr uint K_STRIDE = NXPSG * VPI;
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;

    const uint tx = tiisg % NXPSG;
    const uint ty = tiisg / NXPSG;
    const uint outRow = tgid.x * R0PTG + NYPSG * sgitg + ty;
    const uint batchBase = tgid.y * BATCH_TILE;
    if (outRow >= FIXED_ROWS) return;

    const bool4 active = bool4(
        batchBase + 0 < actualBatch,
        batchBase + 1 < actualBatch,
        batchBase + 2 < actualBatch,
        batchBase + 3 < actualBatch
    );
    const uint4 batchRow = uint4(
        active[0] ? batchBase + 0 : 0,
        active[1] ? batchBase + 1 : 0,
        active[2] ? batchBase + 2 : 0,
        active[3] ? batchBase + 3 : 0
    );
    float4 sums[2] = { float4(0.0f), float4(0.0f) };

    #pragma unroll
    for (uint projection = 0; projection < 2; ++projection) {
        device const uint8_t* weights = projection == 0
            ? gateWeights : upWeights;
        device const half* scales = projection == 0
            ? gateScales : upScales;
        device const half* biases = projection == 0
            ? gateBiases : upBiases;
        device const uint16_t* rowBase =
            (device const uint16_t*)(weights + outRow * HALF_COLS);
        device const half* scRow = scales + outRow * NUM_COL_GROUPS;
        device const half* biRow = biases + outRow * NUM_COL_GROUPS;

        for (uint k = tx * VPI; k < FIXED_COLS; k += K_STRIDE) {
            float wbuf[CHPT * VALS_PER_CHUNK];
            #pragma unroll
            for (uint ch = 0; ch < CHPT; ++ch) {
                const uint col = k + ch * VALS_PER_CHUNK;
                const uint g = col / FIXED_GROUP_SIZE;
                const float s = float(scRow[g]);
                const float bi = float(biRow[g]);
                device const uint16_t* wptr =
                    rowBase + (k / 4) + ch * (VALS_PER_CHUNK / 4);
                const uint16_t w[4] = { wptr[0], wptr[1], wptr[2], wptr[3] };
                #pragma unroll
                for (uint i = 0; i < 4; ++i) {
                    const uint16_t p = w[i];
                    wbuf[ch * 16 + i * 4 + 0] = float((p >>  0) & 0xF) * s + bi;
                    wbuf[ch * 16 + i * 4 + 1] = float((p >>  4) & 0xF) * s + bi;
                    wbuf[ch * 16 + i * 4 + 2] = float((p >>  8) & 0xF) * s + bi;
                    wbuf[ch * 16 + i * 4 + 3] = float((p >> 12) & 0xF) * s + bi;
                }
            }

            device const half* x0 = input + batchRow[0] * FIXED_COLS + k;
            device const half* x1 = input + batchRow[1] * FIXED_COLS + k;
            device const half* x2 = input + batchRow[2] * FIXED_COLS + k;
            device const half* x3 = input + batchRow[3] * FIXED_COLS + k;
            #pragma unroll
            for (uint i = 0; i < CHPT * VALS_PER_CHUNK; ++i) {
                const float4 x = float4(
                    float(x0[i]), float(x1[i]), float(x2[i]), float(x3[i])
                );
                sums[projection] += wbuf[i] * x;
            }
        }

        sums[projection] += simd_shuffle_down(sums[projection], 4);
        sums[projection] += simd_shuffle_down(sums[projection], 2);
        sums[projection] += simd_shuffle_down(sums[projection], 1);
    }

    if (tx == 0) {
        #pragma unroll
        for (uint b = 0; b < BATCH_TILE; ++b) {
            if (active[b]) {
                const float gate = sums[0][b];
                const float up = sums[1][b];
                const float silu = gate / (1.0f + exp(-gate));
                output[(batchBase + b) * FIXED_ROWS + outRow] =
                    half(clamp(silu * up, -65504.0f, 65504.0f));
            }
        }
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE, uint BATCH_TILE>
inline void fused_dual_affine_matvec_fixed_ext(
    device const uint8_t* firstWeights,
    device const half*    firstScales,
    device const half*    firstBiases,
    device const uint8_t* secondWeights,
    device const half*    secondScales,
    device const half*    secondBiases,
    device const half*    input,
    device half*          firstOutput,
    device half*          secondOutput,
    constant uint&        actualBatch,
    uint2 tgid,
    uint  tiisg,
    uint  sgitg
) {
    constexpr uint NSG = 2;
    constexpr uint NXPSG = 8;
    constexpr uint NYPSG = 32 / NXPSG;
    constexpr uint R0PTG = NYPSG * NSG;
    constexpr uint VALS_PER_CHUNK = 16;
    constexpr uint CHPT = 2;
    constexpr uint VPI = CHPT * VALS_PER_CHUNK;
    constexpr uint K_STRIDE = NXPSG * VPI;
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;

    const uint tx = tiisg % NXPSG;
    const uint ty = tiisg / NXPSG;
    const uint outRow = tgid.x * R0PTG + NYPSG * sgitg + ty;
    const uint batchBase = tgid.y * BATCH_TILE;
    const bool fullTile = batchBase + BATCH_TILE <= actualBatch;
    if (outRow >= FIXED_ROWS) return;

    uint batchRow[BATCH_TILE];
    #pragma unroll
    for (uint b = 0; b < BATCH_TILE; ++b) {
        batchRow[b] = (fullTile || batchBase + b < actualBatch)
            ? (batchBase + b) : 0;
    }

    float sums[2][BATCH_TILE];
    #pragma unroll
    for (uint projection = 0; projection < 2; ++projection) {
        #pragma unroll
        for (uint b = 0; b < BATCH_TILE; ++b) sums[projection][b] = 0.0f;

        device const uint8_t* weights = projection == 0
            ? firstWeights : secondWeights;
        device const half* scales = projection == 0
            ? firstScales : secondScales;
        device const half* biases = projection == 0
            ? firstBiases : secondBiases;
        device const uint16_t* rowBase =
            (device const uint16_t*)(weights + outRow * HALF_COLS);
        device const half* scRow = scales + outRow * NUM_COL_GROUPS;
        device const half* biRow = biases + outRow * NUM_COL_GROUPS;

        for (uint k = tx * VPI; k < FIXED_COLS; k += K_STRIDE) {
            float wbuf[CHPT * VALS_PER_CHUNK];
            #pragma unroll
            for (uint ch = 0; ch < CHPT; ++ch) {
                const uint col = k + ch * VALS_PER_CHUNK;
                const uint g = col / FIXED_GROUP_SIZE;
                const float s = float(scRow[g]);
                const float bi = float(biRow[g]);
                device const uint16_t* wptr =
                    rowBase + (k / 4) + ch * (VALS_PER_CHUNK / 4);
                const uint16_t w[4] = { wptr[0], wptr[1], wptr[2], wptr[3] };
                #pragma unroll
                for (uint i = 0; i < 4; ++i) {
                    const uint16_t p = w[i];
                    wbuf[ch * 16 + i * 4 + 0] = float((p >>  0) & 0xF) * s + bi;
                    wbuf[ch * 16 + i * 4 + 1] = float((p >>  4) & 0xF) * s + bi;
                    wbuf[ch * 16 + i * 4 + 2] = float((p >>  8) & 0xF) * s + bi;
                    wbuf[ch * 16 + i * 4 + 3] = float((p >> 12) & 0xF) * s + bi;
                }
            }

            #pragma unroll
            for (uint b = 0; b < BATCH_TILE; ++b) {
                if (!fullTile && batchBase + b >= actualBatch) continue;
                device const half* xPtr = input + batchRow[b] * FIXED_COLS + k;
                #pragma unroll
                for (uint i = 0; i < CHPT * VALS_PER_CHUNK; ++i) {
                    sums[projection][b] += wbuf[i] * float(xPtr[i]);
                }
            }
        }

        #pragma unroll
        for (uint b = 0; b < BATCH_TILE; ++b) {
            sums[projection][b] += simd_shuffle_down(sums[projection][b], 4);
            sums[projection][b] += simd_shuffle_down(sums[projection][b], 2);
            sums[projection][b] += simd_shuffle_down(sums[projection][b], 1);
        }
    }

    if (tx == 0) {
        #pragma unroll
        for (uint b = 0; b < BATCH_TILE; ++b) {
            if (!fullTile && batchBase + b >= actualBatch) continue;
            const uint offset = (batchBase + b) * FIXED_ROWS + outRow;
            firstOutput[offset] = half(sums[0][b]);
            secondOutput[offset] = half(sums[1][b]);
        }
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE, uint BATCH_TILE>
inline void affine_matvec_fixed_batched_sg4(
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device const half*    input,
    device half*          output,
    constant uint&        actualBatch,
    uint2 tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint ROWS_PER_SG_LOCAL = 4;
    constexpr uint ROWS_PER_TG_LOCAL = 16;
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;

    uint baseRow = tgid.x * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;
    uint batchBase = tgid.y * BATCH_TILE;
    bool fullTile = batchBase + BATCH_TILE <= actualBatch;

    float acc[BATCH_TILE][ROWS_PER_SG_LOCAL];
    for (uint b = 0; b < BATCH_TILE; b++) {
        for (uint r = 0; r < ROWS_PER_SG_LOCAL; r++) {
            acc[b][r] = 0.0f;
        }
    }

    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * HALF_COLS);
    device const uint16_t* r1 = r0 + U16_PER_ROW;
    device const uint16_t* r2 = r1 + U16_PER_ROW;
    device const uint16_t* r3 = r2 + U16_PER_ROW;

    device const half* sc0 = scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* sc1 = sc0 + NUM_COL_GROUPS;
    device const half* sc2 = sc1 + NUM_COL_GROUPS;
    device const half* sc3 = sc2 + NUM_COL_GROUPS;
    device const half* bi0 = biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* bi1 = bi0 + NUM_COL_GROUPS;
    device const half* bi2 = bi1 + NUM_COL_GROUPS;
    device const half* bi3 = bi2 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;
        device const uint16_t* w0 = r0 + j * 4;
        device const uint16_t* w1 = r1 + j * 4;
        device const uint16_t* w2 = r2 + j * 4;
        device const uint16_t* w3 = r3 + j * 4;

        for (uint b = 0; b < BATCH_TILE; b++) {
            if (!fullTile && batchBase + b >= actualBatch) continue;
            device const half* xp = input + (batchBase + b) * FIXED_COLS + col;
            float x0  = float(xp[0]);
            float x1  = float(xp[1]);
            float x2  = float(xp[2]);
            float x3  = float(xp[3]);
            float x4  = float(xp[4]);
            float x5  = float(xp[5]);
            float x6  = float(xp[6]);
            float x7  = float(xp[7]);
            float x8  = float(xp[8]);
            float x9  = float(xp[9]);
            float x10 = float(xp[10]);
            float x11 = float(xp[11]);
            float x12 = float(xp[12]);
            float x13 = float(xp[13]);
            float x14 = float(xp[14]);
            float x15 = float(xp[15]);

            float xsum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
                       + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15;

            x1  *= (1.0f / 16.0f);
            x2  *= (1.0f / 256.0f);
            x3  *= (1.0f / 4096.0f);
            x5  *= (1.0f / 16.0f);
            x6  *= (1.0f / 256.0f);
            x7  *= (1.0f / 4096.0f);
            x9  *= (1.0f / 16.0f);
            x10 *= (1.0f / 256.0f);
            x11 *= (1.0f / 4096.0f);
            x13 *= (1.0f / 16.0f);
            x14 *= (1.0f / 256.0f);
            x15 *= (1.0f / 4096.0f);

            acc[b][0] += float(sc0[g]) * AGENT_AFFINE_DOT16(w0) + float(bi0[g]) * xsum;
            acc[b][1] += float(sc1[g]) * AGENT_AFFINE_DOT16(w1) + float(bi1[g]) * xsum;
            acc[b][2] += float(sc2[g]) * AGENT_AFFINE_DOT16(w2) + float(bi2[g]) * xsum;
            acc[b][3] += float(sc3[g]) * AGENT_AFFINE_DOT16(w3) + float(bi3[g]) * xsum;
        }
    }

    for (uint b = 0; b < BATCH_TILE; b++) {
        acc[b][0] = simd_sum(acc[b][0]);
        acc[b][1] = simd_sum(acc[b][1]);
        acc[b][2] = simd_sum(acc[b][2]);
        acc[b][3] = simd_sum(acc[b][3]);
    }

    if (simd_lane == 0) {
        for (uint b = 0; b < BATCH_TILE; b++) {
            if (!fullTile && batchBase + b >= actualBatch) continue;
            device half* out = output + (batchBase + b) * FIXED_ROWS + sgBaseRow;
            out[0] = half(acc[b][0]);
            out[1] = half(acc[b][1]);
            out[2] = half(acc[b][2]);
            out[3] = half(acc[b][3]);
        }
    }
}

// Argmax key packing: agent_argmax_pack_key encodes the lookup
// index into the lower bits of the .y field so the reduction can
// be a single uint2 comparison. AGENT_ARGMAX_INDEX_BITS bounds the
// supported vocab size; raise it (and the reducer's MAX_INDEX
// decode) before adding a kernel for vocab > 2^AGENT_ARGMAX_INDEX_BITS.
#define AGENT_ARGMAX_INDEX_BITS 18
#define AGENT_ARGMAX_MAX_INDEX  ((1u << AGENT_ARGMAX_INDEX_BITS) - 1u)

static inline uint agent_ordered_half_key(half value) {
    uint bits = uint(as_type<ushort>(value));
    // half(-0) = 0x8000 and half(+0) = 0x0000 compare equal
    // numerically, but their raw bits sort apart. Collapse -0 to
    // +0 so signed-zero ties resolve via the index slot and the
    // earlier-index winner matches the CPU argmax semantics.
    if ((bits & 0x7FFFu) == 0u) {
        bits = 0u;
    }
    return (bits & 0x8000u) != 0u
        ? (0xFFFFu - bits)
        : (bits ^ 0x8000u);
}

static inline uint2 agent_argmax_pack_key(half value, uint index) {
    uint orderedValue = agent_ordered_half_key(value);
    uint inverseIndex = (AGENT_ARGMAX_MAX_INDEX - index) & AGENT_ARGMAX_MAX_INDEX;
    return uint2(orderedValue, inverseIndex);
}

static inline bool agent_argmax_key_gt(uint2 lhs, uint2 rhs) {
    return lhs.x > rhs.x || (lhs.x == rhs.x && lhs.y > rhs.y);
}

static inline uint2 agent_argmax_key_max(uint2 lhs, uint2 rhs) {
    return agent_argmax_key_gt(lhs, rhs) ? lhs : rhs;
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE, uint BATCH_TILE>
inline void affine_matvec_argmax_fixed_batched(
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device const half*    input,
    device uint2*         partialKeys,
    uint                  actualBatch,
    float                 logitCap,
    threadgroup uint2*    localKeys,
    uint2 tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;

    uint baseRow = tgid.x * ROWS_PER_TG;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint batchBase = tgid.y * BATCH_TILE;
    if (batchBase >= actualBatch) {
        return;
    }
    bool fullTile = batchBase + BATCH_TILE <= actualBatch;

    float acc[BATCH_TILE][ROWS_PER_SG];
    for (uint b = 0; b < BATCH_TILE; b++) {
        for (uint r = 0; r < ROWS_PER_SG; r++) {
            acc[b][r] = 0.0f;
        }
    }

    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * HALF_COLS);
    device const uint16_t* r1 = r0 + U16_PER_ROW;
    device const uint16_t* r2 = r1 + U16_PER_ROW;
    device const uint16_t* r3 = r2 + U16_PER_ROW;

    device const half* sc0 = scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* sc1 = sc0 + NUM_COL_GROUPS;
    device const half* sc2 = sc1 + NUM_COL_GROUPS;
    device const half* sc3 = sc2 + NUM_COL_GROUPS;
    device const half* bi0 = biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* bi1 = bi0 + NUM_COL_GROUPS;
    device const half* bi2 = bi1 + NUM_COL_GROUPS;
    device const half* bi3 = bi2 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;
        device const uint16_t* w0 = r0 + j * 4;
        device const uint16_t* w1 = r1 + j * 4;
        device const uint16_t* w2 = r2 + j * 4;
        device const uint16_t* w3 = r3 + j * 4;

        for (uint b = 0; b < BATCH_TILE; b++) {
            if (!fullTile && batchBase + b >= actualBatch) continue;
            device const half* xp = input + (batchBase + b) * FIXED_COLS + col;
            float x0  = float(xp[0]);
            float x1  = float(xp[1]);
            float x2  = float(xp[2]);
            float x3  = float(xp[3]);
            float x4  = float(xp[4]);
            float x5  = float(xp[5]);
            float x6  = float(xp[6]);
            float x7  = float(xp[7]);
            float x8  = float(xp[8]);
            float x9  = float(xp[9]);
            float x10 = float(xp[10]);
            float x11 = float(xp[11]);
            float x12 = float(xp[12]);
            float x13 = float(xp[13]);
            float x14 = float(xp[14]);
            float x15 = float(xp[15]);

            float xsum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
                       + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15;

            x1  *= (1.0f / 16.0f);
            x2  *= (1.0f / 256.0f);
            x3  *= (1.0f / 4096.0f);
            x5  *= (1.0f / 16.0f);
            x6  *= (1.0f / 256.0f);
            x7  *= (1.0f / 4096.0f);
            x9  *= (1.0f / 16.0f);
            x10 *= (1.0f / 256.0f);
            x11 *= (1.0f / 4096.0f);
            x13 *= (1.0f / 16.0f);
            x14 *= (1.0f / 256.0f);
            x15 *= (1.0f / 4096.0f);

            acc[b][0] += float(sc0[g]) * AGENT_AFFINE_DOT16(w0) + float(bi0[g]) * xsum;
            acc[b][1] += float(sc1[g]) * AGENT_AFFINE_DOT16(w1) + float(bi1[g]) * xsum;
            acc[b][2] += float(sc2[g]) * AGENT_AFFINE_DOT16(w2) + float(bi2[g]) * xsum;
            acc[b][3] += float(sc3[g]) * AGENT_AFFINE_DOT16(w3) + float(bi3[g]) * xsum;
        }
    }

    for (uint b = 0; b < BATCH_TILE; b++) {
        acc[b][0] = simd_sum(acc[b][0]);
        acc[b][1] = simd_sum(acc[b][1]);
        acc[b][2] = simd_sum(acc[b][2]);
        acc[b][3] = simd_sum(acc[b][3]);
    }

    if (simd_lane == 0) {
        for (uint b = 0; b < BATCH_TILE; b++) {
            if (!fullTile && batchBase + b >= actualBatch) continue;
            // Apply logit cap before fp16 rounding to match the
            // decode/full-logits path. activations.metal:170 uses
            // `cap * tanh(x / cap)` for the standalone logit_cap
            // kernel; mirror that exact expression so the rounded
            // half values match bit-for-bit.
            float v0 = acc[b][0];
            float v1 = acc[b][1];
            float v2 = acc[b][2];
            float v3 = acc[b][3];
            if (logitCap > 0.0f) {
                v0 = logitCap * tanh(v0 / logitCap);
                v1 = logitCap * tanh(v1 / logitCap);
                v2 = logitCap * tanh(v2 / logitCap);
                v3 = logitCap * tanh(v3 / logitCap);
            }
            uint2 best = agent_argmax_pack_key(half(v0), sgBaseRow + 0);
            best = agent_argmax_key_max(
                best,
                agent_argmax_pack_key(half(v1), sgBaseRow + 1)
            );
            best = agent_argmax_key_max(
                best,
                agent_argmax_pack_key(half(v2), sgBaseRow + 2)
            );
            best = agent_argmax_key_max(
                best,
                agent_argmax_pack_key(half(v3), sgBaseRow + 3)
            );
            localKeys[b * 2 + simd_group] = best;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0 && simd_lane == 0) {
        constexpr uint PARTIALS_PER_BATCH = FIXED_ROWS / ROWS_PER_TG;
        for (uint b = 0; b < BATCH_TILE; b++) {
            if (!fullTile && batchBase + b >= actualBatch) continue;
            uint batch = batchBase + b;
            uint2 best = agent_argmax_key_max(
                localKeys[b * 2],
                localKeys[b * 2 + 1]
            );
            partialKeys[batch * PARTIALS_PER_BATCH + tgid.x] = best;
        }
    }
}

template <uint FIXED_ROWS>
inline void lm_head_argmax_reduce_fixed(
    device const uint2* partialKeys,
    device int*        output,
    uint               actualBatch,
    uint               batch,
    uint               tid,
    threadgroup uint2* scratch
) {
    if (batch >= actualBatch) {
        return;
    }

    constexpr uint PARTIALS_PER_BATCH = FIXED_ROWS / ROWS_PER_TG;
    constexpr uint THREADS = 256;
    uint2 best = uint2(0, 0);
    device const uint2* row = partialKeys + batch * PARTIALS_PER_BATCH;
    for (uint idx = tid; idx < PARTIALS_PER_BATCH; idx += THREADS) {
        best = agent_argmax_key_max(best, row[idx]);
    }
    scratch[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = agent_argmax_key_max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[batch] = int(AGENT_ARGMAX_MAX_INDEX - scratch[0].y);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE, uint BATCH_TILE>
inline void norm_scale_affine_matvec_fixed_batched(
    device const float*   scalePtr,
    device const half*    normInput,
    device const half*    normWeight,
    device half*          normOutput,
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device half*          output,
    constant uint&        actualBatch,
    uint2 tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;

    uint baseRow = tgid.x * ROWS_PER_TG;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint batchBase = tgid.y * BATCH_TILE;
    bool fullTile = batchBase + BATCH_TILE <= actualBatch;
    bool writeNormOutput = tgid.x == 0 && simd_group == 0;

    float acc[BATCH_TILE][ROWS_PER_SG];
    for (uint b = 0; b < BATCH_TILE; b++) {
        for (uint r = 0; r < ROWS_PER_SG; r++) {
            acc[b][r] = 0.0f;
        }
    }

    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * HALF_COLS);
    device const uint16_t* r1 = r0 + U16_PER_ROW;
    device const uint16_t* r2 = r1 + U16_PER_ROW;
    device const uint16_t* r3 = r2 + U16_PER_ROW;

    device const half* sc0 = scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* sc1 = sc0 + NUM_COL_GROUPS;
    device const half* sc2 = sc1 + NUM_COL_GROUPS;
    device const half* sc3 = sc2 + NUM_COL_GROUPS;
    device const half* bi0 = biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* bi1 = bi0 + NUM_COL_GROUPS;
    device const half* bi2 = bi1 + NUM_COL_GROUPS;
    device const half* bi3 = bi2 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;
        device const uint16_t* w0 = r0 + j * 4;
        device const uint16_t* w1 = r1 + j * 4;
        device const uint16_t* w2 = r2 + j * 4;
        device const uint16_t* w3 = r3 + j * 4;

        for (uint b = 0; b < BATCH_TILE; b++) {
            if (!fullTile && batchBase + b >= actualBatch) continue;

            uint batchIndex = batchBase + b;
            float rs = scalePtr[batchIndex];
            device const half* xp = normInput + batchIndex * FIXED_COLS + col;
            device const half* wp = normWeight + col;

            float x0  = float(xp[0])  * rs * (1.0f + float(wp[0]));
            float x1  = float(xp[1])  * rs * (1.0f + float(wp[1]));
            float x2  = float(xp[2])  * rs * (1.0f + float(wp[2]));
            float x3  = float(xp[3])  * rs * (1.0f + float(wp[3]));
            float x4  = float(xp[4])  * rs * (1.0f + float(wp[4]));
            float x5  = float(xp[5])  * rs * (1.0f + float(wp[5]));
            float x6  = float(xp[6])  * rs * (1.0f + float(wp[6]));
            float x7  = float(xp[7])  * rs * (1.0f + float(wp[7]));
            float x8  = float(xp[8])  * rs * (1.0f + float(wp[8]));
            float x9  = float(xp[9])  * rs * (1.0f + float(wp[9]));
            float x10 = float(xp[10]) * rs * (1.0f + float(wp[10]));
            float x11 = float(xp[11]) * rs * (1.0f + float(wp[11]));
            float x12 = float(xp[12]) * rs * (1.0f + float(wp[12]));
            float x13 = float(xp[13]) * rs * (1.0f + float(wp[13]));
            float x14 = float(xp[14]) * rs * (1.0f + float(wp[14]));
            float x15 = float(xp[15]) * rs * (1.0f + float(wp[15]));

            half hx0 = half(x0);
            half hx1 = half(x1);
            half hx2 = half(x2);
            half hx3 = half(x3);
            half hx4 = half(x4);
            half hx5 = half(x5);
            half hx6 = half(x6);
            half hx7 = half(x7);
            half hx8 = half(x8);
            half hx9 = half(x9);
            half hx10 = half(x10);
            half hx11 = half(x11);
            half hx12 = half(x12);
            half hx13 = half(x13);
            half hx14 = half(x14);
            half hx15 = half(x15);

            if (writeNormOutput) {
                device half* out = normOutput + batchIndex * FIXED_COLS + col;
                out[0] = hx0;
                out[1] = hx1;
                out[2] = hx2;
                out[3] = hx3;
                out[4] = hx4;
                out[5] = hx5;
                out[6] = hx6;
                out[7] = hx7;
                out[8] = hx8;
                out[9] = hx9;
                out[10] = hx10;
                out[11] = hx11;
                out[12] = hx12;
                out[13] = hx13;
                out[14] = hx14;
                out[15] = hx15;
            }

            x0 = float(hx0);
            x1 = float(hx1);
            x2 = float(hx2);
            x3 = float(hx3);
            x4 = float(hx4);
            x5 = float(hx5);
            x6 = float(hx6);
            x7 = float(hx7);
            x8 = float(hx8);
            x9 = float(hx9);
            x10 = float(hx10);
            x11 = float(hx11);
            x12 = float(hx12);
            x13 = float(hx13);
            x14 = float(hx14);
            x15 = float(hx15);

            float xsum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
                       + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15;

            x1  *= (1.0f / 16.0f);
            x2  *= (1.0f / 256.0f);
            x3  *= (1.0f / 4096.0f);
            x5  *= (1.0f / 16.0f);
            x6  *= (1.0f / 256.0f);
            x7  *= (1.0f / 4096.0f);
            x9  *= (1.0f / 16.0f);
            x10 *= (1.0f / 256.0f);
            x11 *= (1.0f / 4096.0f);
            x13 *= (1.0f / 16.0f);
            x14 *= (1.0f / 256.0f);
            x15 *= (1.0f / 4096.0f);

            acc[b][0] += float(sc0[g]) * AGENT_AFFINE_DOT16(w0) + float(bi0[g]) * xsum;
            acc[b][1] += float(sc1[g]) * AGENT_AFFINE_DOT16(w1) + float(bi1[g]) * xsum;
            acc[b][2] += float(sc2[g]) * AGENT_AFFINE_DOT16(w2) + float(bi2[g]) * xsum;
            acc[b][3] += float(sc3[g]) * AGENT_AFFINE_DOT16(w3) + float(bi3[g]) * xsum;
        }
    }

    for (uint b = 0; b < BATCH_TILE; b++) {
        acc[b][0] = simd_sum(acc[b][0]);
        acc[b][1] = simd_sum(acc[b][1]);
        acc[b][2] = simd_sum(acc[b][2]);
        acc[b][3] = simd_sum(acc[b][3]);
    }

    if (simd_lane == 0) {
        for (uint b = 0; b < BATCH_TILE; b++) {
            if (!fullTile && batchBase + b >= actualBatch) continue;
            device half* out = output + (batchBase + b) * FIXED_ROWS + sgBaseRow;
            out[0] = half(acc[b][0]);
            out[1] = half(acc[b][1]);
            out[2] = half(acc[b][2]);
            out[3] = half(acc[b][3]);
        }
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_affine_matvec_add_fixed(
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device const half*    input,
    device half*          output,
    device const half*    residual,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;

    uint baseRow = tgid * ROWS_PER_TG;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * HALF_COLS);
    device const uint16_t* r1 = r0 + U16_PER_ROW;
    device const uint16_t* r2 = r1 + U16_PER_ROW;
    device const uint16_t* r3 = r2 + U16_PER_ROW;

    device const half* sc0 = scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* sc1 = sc0 + NUM_COL_GROUPS;
    device const half* sc2 = sc1 + NUM_COL_GROUPS;
    device const half* sc3 = sc2 + NUM_COL_GROUPS;
    device const half* bi0 = biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* bi1 = bi0 + NUM_COL_GROUPS;
    device const half* bi2 = bi1 + NUM_COL_GROUPS;
    device const half* bi3 = bi2 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;

        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

        acc0 += float(sc0[g]) * AGENT_AFFINE_DOT16(r0 + j*4) + float(bi0[g]) * xsum;
        acc1 += float(sc1[g]) * AGENT_AFFINE_DOT16(r1 + j*4) + float(bi1[g]) * xsum;
        acc2 += float(sc2[g]) * AGENT_AFFINE_DOT16(r2 + j*4) + float(bi2[g]) * xsum;
        acc3 += float(sc3[g]) * AGENT_AFFINE_DOT16(r3 + j*4) + float(bi3[g]) * xsum;
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (simd_lane == 0) {
        device half* out = output + sgBaseRow;
        out[0] = half(acc0);
        out[1] = half(acc1);
        out[2] = half(acc2);
        out[3] = half(acc3);
        out[0] = out[0] + residual[sgBaseRow];
        out[1] = out[1] + residual[sgBaseRow + 1];
        out[2] = out[2] + residual[sgBaseRow + 2];
        out[3] = out[3] + residual[sgBaseRow + 3];
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_affine_matvec_add_fixed_rows4(
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device const half*    input,
    device half*          output,
    device const half*    residual,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 2;
    constexpr uint ROWS_PER_TG_LOCAL = 4;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;

    float acc0 = 0.0f, acc1 = 0.0f;

    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * HALF_COLS);
    device const uint16_t* r1 = r0 + U16_PER_ROW;

    device const half* sc0 = scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* sc1 = sc0 + NUM_COL_GROUPS;
    device const half* bi0 = biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* bi1 = bi0 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;

        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

        acc0 += float(sc0[g]) * AGENT_AFFINE_DOT16(r0 + j*4) + float(bi0[g]) * xsum;
        acc1 += float(sc1[g]) * AGENT_AFFINE_DOT16(r1 + j*4) + float(bi1[g]) * xsum;
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (simd_lane == 0) {
        device half* out = output + sgBaseRow;
        out[0] = half(acc0);
        out[1] = half(acc1);
        out[0] = out[0] + residual[sgBaseRow];
        out[1] = out[1] + residual[sgBaseRow + 1];
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_affine_matvec_add_fixed_rows4_sbcache(
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device const half*    input,
    device half*          output,
    device const half*    residual,
    threadgroup half*     scaleBiasCache,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 2;
    constexpr uint ROWS_PER_TG_LOCAL = 4;
    constexpr uint CACHE_COUNT = ROWS_PER_TG_LOCAL * NUM_COL_GROUPS;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;
    uint tid = simd_group * 32 + simd_lane;

    threadgroup half* scCache = scaleBiasCache;
    threadgroup half* biCache = scaleBiasCache + CACHE_COUNT;
    for (uint i = tid; i < CACHE_COUNT; i += 64) {
        uint localRow = i / NUM_COL_GROUPS;
        uint group = i - localRow * NUM_COL_GROUPS;
        uint row = baseRow + localRow;
        scCache[i] = scales[row * NUM_COL_GROUPS + group];
        biCache[i] = biases[row * NUM_COL_GROUPS + group];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc0 = 0.0f, acc1 = 0.0f;

    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * HALF_COLS);
    device const uint16_t* r1 = r0 + U16_PER_ROW;

    uint localSgRow = simd_group * ROWS_PER_SG_LOCAL;
    threadgroup half* sc0 = scCache + localSgRow * NUM_COL_GROUPS;
    threadgroup half* sc1 = sc0 + NUM_COL_GROUPS;
    threadgroup half* bi0 = biCache + localSgRow * NUM_COL_GROUPS;
    threadgroup half* bi1 = bi0 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;

        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

        acc0 += float(sc0[g]) * AGENT_AFFINE_DOT16(r0 + j*4) + float(bi0[g]) * xsum;
        acc1 += float(sc1[g]) * AGENT_AFFINE_DOT16(r1 + j*4) + float(bi1[g]) * xsum;
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (simd_lane == 0) {
        device half* out = output + sgBaseRow;
        out[0] = half(acc0);
        out[1] = half(acc1);
        out[0] = out[0] + residual[sgBaseRow];
        out[1] = out[1] + residual[sgBaseRow + 1];
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_dual_affine_matvec_add_fixed_rows4(
    device const uint8_t* w1_weights,
    device const half*    w1_scales,
    device const half*    w1_biases,
    device const uint8_t* w2_weights,
    device const half*    w2_scales,
    device const half*    w2_biases,
    device const half*    input,
    device half*          output1,
    device half*          output2,
    device const half*    residual1,
    device const half*    residual2,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 2;
    constexpr uint ROWS_PER_TG_LOCAL = 4;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;

    float a0 = 0.0f, a1 = 0.0f;
    float b0 = 0.0f, b1 = 0.0f;

    device const uint16_t* w1r0 = (device const uint16_t*)(w1_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* w1r1 = w1r0 + U16_PER_ROW;
    device const uint16_t* w2r0 = (device const uint16_t*)(w2_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* w2r1 = w2r0 + U16_PER_ROW;

    device const half* w1s0 = w1_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* w1s1 = w1s0 + NUM_COL_GROUPS;
    device const half* w1b0 = w1_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* w1b1 = w1b0 + NUM_COL_GROUPS;
    device const half* w2s0 = w2_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* w2s1 = w2s0 + NUM_COL_GROUPS;
    device const half* w2b0 = w2_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* w2b1 = w2b0 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;

        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

        a0 += float(w1s0[g]) * AGENT_AFFINE_DOT16(w1r0 + j*4) + float(w1b0[g]) * xsum;
        b0 += float(w2s0[g]) * AGENT_AFFINE_DOT16(w2r0 + j*4) + float(w2b0[g]) * xsum;
        a1 += float(w1s1[g]) * AGENT_AFFINE_DOT16(w1r1 + j*4) + float(w1b1[g]) * xsum;
        b1 += float(w2s1[g]) * AGENT_AFFINE_DOT16(w2r1 + j*4) + float(w2b1[g]) * xsum;
    }

    a0 = simd_sum(a0);
    a1 = simd_sum(a1);
    b0 = simd_sum(b0);
    b1 = simd_sum(b1);

    if (simd_lane == 0) {
        output1[sgBaseRow] = half(a0) + residual1[sgBaseRow];
        output2[sgBaseRow] = half(b0) + residual2[sgBaseRow];
        output1[sgBaseRow + 1] = half(a1) + residual1[sgBaseRow + 1];
        output2[sgBaseRow + 1] = half(b1) + residual2[sgBaseRow + 1];
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_affine_matvec_add_fixed_rows4_tginput(
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device const half*    input,
    device half*          output,
    device const half*    residual,
    threadgroup half*     inputScratch,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 2;
    constexpr uint ROWS_PER_TG_LOCAL = 4;
    constexpr uint THREADS_PER_TG_LOCAL = 64;

    uint tid = simd_group * 32 + simd_lane;
    for (uint idx = tid; idx < FIXED_COLS; idx += THREADS_PER_TG_LOCAL) {
        inputScratch[idx] = input[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;

    float acc0 = 0.0f, acc1 = 0.0f;

    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * HALF_COLS);
    device const uint16_t* r1 = r0 + U16_PER_ROW;

    device const half* sc0 = scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* sc1 = sc0 + NUM_COL_GROUPS;
    device const half* bi0 = biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* bi1 = bi0 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;

        threadgroup const half* xp = inputScratch + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

        acc0 += float(sc0[g]) * AGENT_AFFINE_DOT16(r0 + j*4) + float(bi0[g]) * xsum;
        acc1 += float(sc1[g]) * AGENT_AFFINE_DOT16(r1 + j*4) + float(bi1[g]) * xsum;
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (simd_lane == 0) {
        device half* out = output + sgBaseRow;
        out[0] = half(acc0);
        out[1] = half(acc1);
        out[0] = out[0] + residual[sgBaseRow];
        out[1] = out[1] + residual[sgBaseRow + 1];
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_affine_matvec_add_fixed_rows4_sg1(
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device const half*    input,
    device half*          output,
    device const half*    residual,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_TG_LOCAL = 4;

    uint row = tgid * ROWS_PER_TG_LOCAL + simd_group;
    float acc = 0.0f;

    device const uint16_t* r = (device const uint16_t*)(weights + row * HALF_COLS);
    device const half* sc = scales + row * NUM_COL_GROUPS;
    device const half* bi = biases + row * NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;

        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

        acc += float(sc[g]) * AGENT_AFFINE_DOT16(r + j * 4) + float(bi[g]) * xsum;
    }

    acc = simd_sum(acc);

    if (simd_lane == 0) {
        output[row] = half(acc) + residual[row];
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE, uint ROWS_PER_TG_LOCAL>
inline void fused_affine_matvec_add_fixed_rows4sg_tiled(
    device const uint8_t* weights,
    device const half*    scales,
    device const half*    biases,
    device const half*    input,
    device half*          output,
    device const half*    residual,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 4;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    device const uint16_t* r0 = (device const uint16_t*)(weights + sgBaseRow * HALF_COLS);
    device const uint16_t* r1 = r0 + U16_PER_ROW;
    device const uint16_t* r2 = r1 + U16_PER_ROW;
    device const uint16_t* r3 = r2 + U16_PER_ROW;

    device const half* sc0 = scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* sc1 = sc0 + NUM_COL_GROUPS;
    device const half* sc2 = sc1 + NUM_COL_GROUPS;
    device const half* sc3 = sc2 + NUM_COL_GROUPS;
    device const half* bi0 = biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* bi1 = bi0 + NUM_COL_GROUPS;
    device const half* bi2 = bi1 + NUM_COL_GROUPS;
    device const half* bi3 = bi2 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint g = col / FIXED_GROUP_SIZE;

        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

        acc0 += float(sc0[g]) * AGENT_AFFINE_DOT16(r0 + j*4) + float(bi0[g]) * xsum;
        acc1 += float(sc1[g]) * AGENT_AFFINE_DOT16(r1 + j*4) + float(bi1[g]) * xsum;
        acc2 += float(sc2[g]) * AGENT_AFFINE_DOT16(r2 + j*4) + float(bi2[g]) * xsum;
        acc3 += float(sc3[g]) * AGENT_AFFINE_DOT16(r3 + j*4) + float(bi3[g]) * xsum;
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (simd_lane == 0) {
        device half* out = output + sgBaseRow;
        out[0] = half(acc0);
        out[1] = half(acc1);
        out[2] = half(acc2);
        out[3] = half(acc3);
        out[0] = out[0] + residual[sgBaseRow];
        out[1] = out[1] + residual[sgBaseRow + 1];
        out[2] = out[2] + residual[sgBaseRow + 2];
        out[3] = out[3] + residual[sgBaseRow + 3];
    }
}


template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_affine_gate_up_swiglu_fixed(
    device const uint8_t* gate_weights,
    device const half*    gate_scales,
    device const half*    gate_biases,
    device const uint8_t* up_weights,
    device const half*    up_scales,
    device const half*    up_biases,
    device const half*    input,
    device half*          output,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;

    uint baseRow = tgid * ROWS_PER_TG;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;

    float g0 = 0, g1 = 0, g2 = 0, g3 = 0;
    float u0 = 0, u1 = 0, u2 = 0, u3 = 0;

    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* gr1 = gr0 + U16_PER_ROW;
    device const uint16_t* gr2 = gr1 + U16_PER_ROW;
    device const uint16_t* gr3 = gr2 + U16_PER_ROW;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* ur1 = ur0 + U16_PER_ROW;
    device const uint16_t* ur2 = ur1 + U16_PER_ROW;
    device const uint16_t* ur3 = ur2 + U16_PER_ROW;

    device const half* gs0=gate_scales+sgBaseRow*NUM_COL_GROUPS; device const half* gs1=gs0+NUM_COL_GROUPS; device const half* gs2=gs1+NUM_COL_GROUPS; device const half* gs3=gs2+NUM_COL_GROUPS;
    device const half* gb0=gate_biases+sgBaseRow*NUM_COL_GROUPS; device const half* gb1=gb0+NUM_COL_GROUPS; device const half* gb2=gb1+NUM_COL_GROUPS; device const half* gb3=gb2+NUM_COL_GROUPS;
    device const half* us0=up_scales+sgBaseRow*NUM_COL_GROUPS;   device const half* us1=us0+NUM_COL_GROUPS; device const half* us2=us1+NUM_COL_GROUPS; device const half* us3=us2+NUM_COL_GROUPS;
    device const half* ub0=up_biases+sgBaseRow*NUM_COL_GROUPS;   device const half* ub1=ub0+NUM_COL_GROUPS; device const half* ub2=ub1+NUM_COL_GROUPS; device const half* ub3=ub2+NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint gidx = col / FIXED_GROUP_SIZE;
        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

        g0 += float(gs0[gidx]) * AGENT_AFFINE_DOT16(gr0 + j*4) + float(gb0[gidx]) * xsum;
        u0 += float(us0[gidx]) * AGENT_AFFINE_DOT16(ur0 + j*4) + float(ub0[gidx]) * xsum;
        g1 += float(gs1[gidx]) * AGENT_AFFINE_DOT16(gr1 + j*4) + float(gb1[gidx]) * xsum;
        u1 += float(us1[gidx]) * AGENT_AFFINE_DOT16(ur1 + j*4) + float(ub1[gidx]) * xsum;
        g2 += float(gs2[gidx]) * AGENT_AFFINE_DOT16(gr2 + j*4) + float(gb2[gidx]) * xsum;
        u2 += float(us2[gidx]) * AGENT_AFFINE_DOT16(ur2 + j*4) + float(ub2[gidx]) * xsum;
        g3 += float(gs3[gidx]) * AGENT_AFFINE_DOT16(gr3 + j*4) + float(gb3[gidx]) * xsum;
        u3 += float(us3[gidx]) * AGENT_AFFINE_DOT16(ur3 + j*4) + float(ub3[gidx]) * xsum;
    }

    g0=simd_sum(g0); g1=simd_sum(g1); g2=simd_sum(g2); g3=simd_sum(g3);
    u0=simd_sum(u0); u1=simd_sum(u1); u2=simd_sum(u2); u3=simd_sum(u3);

    if (simd_lane == 0) {
        output[sgBaseRow] = half(g0 / (1.0f + fast::exp(-g0)) * u0);
        output[sgBaseRow + 1] = half(g1 / (1.0f + fast::exp(-g1)) * u1);
        output[sgBaseRow + 2] = half(g2 / (1.0f + fast::exp(-g2)) * u2);
        output[sgBaseRow + 3] = half(g3 / (1.0f + fast::exp(-g3)) * u3);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_affine_gate_up_swiglu_fixed_rows4(
    device const uint8_t* gate_weights,
    device const half*    gate_scales,
    device const half*    gate_biases,
    device const uint8_t* up_weights,
    device const half*    up_scales,
    device const half*    up_biases,
    device const half*    input,
    device half*          output,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 2;
    constexpr uint ROWS_PER_TG_LOCAL = 4;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;

    float g0 = 0.0f, g1 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f;

    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* gr1 = gr0 + U16_PER_ROW;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* ur1 = ur0 + U16_PER_ROW;

    device const half* gs0 = gate_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* gs1 = gs0 + NUM_COL_GROUPS;
    device const half* gb0 = gate_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* gb1 = gb0 + NUM_COL_GROUPS;
    device const half* us0 = up_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* us1 = us0 + NUM_COL_GROUPS;
    device const half* ub0 = up_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* ub1 = ub0 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint gidx = col / FIXED_GROUP_SIZE;
        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

        g0 += float(gs0[gidx]) * AGENT_AFFINE_DOT16(gr0 + j * 4) + float(gb0[gidx]) * xsum;
        u0 += float(us0[gidx]) * AGENT_AFFINE_DOT16(ur0 + j * 4) + float(ub0[gidx]) * xsum;
        g1 += float(gs1[gidx]) * AGENT_AFFINE_DOT16(gr1 + j * 4) + float(gb1[gidx]) * xsum;
        u1 += float(us1[gidx]) * AGENT_AFFINE_DOT16(ur1 + j * 4) + float(ub1[gidx]) * xsum;
    }

    g0 = simd_sum(g0);
    g1 = simd_sum(g1);
    u0 = simd_sum(u0);
    u1 = simd_sum(u1);

    if (simd_lane == 0) {
        output[sgBaseRow] = half(g0 / (1.0f + fast::exp(-g0)) * u0);
        output[sgBaseRow + 1] = half(g1 / (1.0f + fast::exp(-g1)) * u1);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_affine_gate_up_swiglu_fixed_rows4_sbcache(
    device const uint8_t* gate_weights,
    device const half*    gate_scales,
    device const half*    gate_biases,
    device const uint8_t* up_weights,
    device const half*    up_scales,
    device const half*    up_biases,
    device const half*    input,
    device half*          output,
    threadgroup half4*    scaleBiasCache,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 2;
    constexpr uint ROWS_PER_TG_LOCAL = 4;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;
    uint localSgRow = simd_group * ROWS_PER_SG_LOCAL;

    constexpr uint CACHE_COUNT_PER_SIMDGROUP = ROWS_PER_SG_LOCAL * NUM_COL_GROUPS;
    for (uint localIndex = simd_lane;
         localIndex < CACHE_COUNT_PER_SIMDGROUP;
         localIndex += 32) {
        uint localRow = localSgRow + localIndex / NUM_COL_GROUPS;
        uint group = localIndex % NUM_COL_GROUPS;
        uint row = baseRow + localRow;
        uint sourceIndex = row * NUM_COL_GROUPS + group;
        uint cacheIndex = localRow * NUM_COL_GROUPS + group;
        scaleBiasCache[cacheIndex] = half4(
            gate_scales[sourceIndex], gate_biases[sourceIndex],
            up_scales[sourceIndex], up_biases[sourceIndex]
        );
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    float g0 = 0.0f, g1 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f;

    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* gr1 = gr0 + U16_PER_ROW;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* ur1 = ur0 + U16_PER_ROW;

    threadgroup half4* sb0 = scaleBiasCache + localSgRow * NUM_COL_GROUPS;
    threadgroup half4* sb1 = sb0 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint gidx = col / FIXED_GROUP_SIZE;
        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

        half4 sbv0 = sb0[gidx];
        half4 sbv1 = sb1[gidx];
        g0 += float(sbv0[0]) * AGENT_AFFINE_DOT16(gr0 + j * 4) + float(sbv0[1]) * xsum;
        u0 += float(sbv0[2]) * AGENT_AFFINE_DOT16(ur0 + j * 4) + float(sbv0[3]) * xsum;
        g1 += float(sbv1[0]) * AGENT_AFFINE_DOT16(gr1 + j * 4) + float(sbv1[1]) * xsum;
        u1 += float(sbv1[2]) * AGENT_AFFINE_DOT16(ur1 + j * 4) + float(sbv1[3]) * xsum;
    }

    g0 = simd_sum(g0);
    g1 = simd_sum(g1);
    u0 = simd_sum(u0);
    u1 = simd_sum(u1);

    if (simd_lane == 0) {
        output[sgBaseRow] = half(g0 / (1.0f + fast::exp(-g0)) * u0);
        output[sgBaseRow + 1] = half(g1 / (1.0f + fast::exp(-g1)) * u1);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_affine_gate_up_swiglu_fixed_rows4_exp2(
    device const uint8_t* gate_weights,
    device const half*    gate_scales,
    device const half*    gate_biases,
    device const uint8_t* up_weights,
    device const half*    up_scales,
    device const half*    up_biases,
    device const half*    input,
    device half*          output,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 2;
    constexpr uint ROWS_PER_TG_LOCAL = 4;
    constexpr float INV_LN2 = 1.4426950408889634f;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;

    float g0 = 0.0f, g1 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f;

    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* gr1 = gr0 + U16_PER_ROW;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* ur1 = ur0 + U16_PER_ROW;

    device const half* gs0 = gate_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* gs1 = gs0 + NUM_COL_GROUPS;
    device const half* gb0 = gate_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* gb1 = gb0 + NUM_COL_GROUPS;
    device const half* us0 = up_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* us1 = us0 + NUM_COL_GROUPS;
    device const half* ub0 = up_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* ub1 = ub0 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint gidx = col / FIXED_GROUP_SIZE;
        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

        g0 += float(gs0[gidx]) * AGENT_AFFINE_DOT16(gr0 + j * 4) + float(gb0[gidx]) * xsum;
        u0 += float(us0[gidx]) * AGENT_AFFINE_DOT16(ur0 + j * 4) + float(ub0[gidx]) * xsum;
        g1 += float(gs1[gidx]) * AGENT_AFFINE_DOT16(gr1 + j * 4) + float(gb1[gidx]) * xsum;
        u1 += float(us1[gidx]) * AGENT_AFFINE_DOT16(ur1 + j * 4) + float(ub1[gidx]) * xsum;
    }

    g0 = simd_sum(g0);
    g1 = simd_sum(g1);
    u0 = simd_sum(u0);
    u1 = simd_sum(u1);

    if (simd_lane == 0) {
        output[sgBaseRow] = half(g0 / (1.0f + fast::exp2(-g0 * INV_LN2)) * u0);
        output[sgBaseRow + 1] = half(g1 / (1.0f + fast::exp2(-g1 * INV_LN2)) * u1);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_affine_gate_up_swiglu_fixed_rows4_tginput(
    device const uint8_t* gate_weights,
    device const half*    gate_scales,
    device const half*    gate_biases,
    device const uint8_t* up_weights,
    device const half*    up_scales,
    device const half*    up_biases,
    device const half*    input,
    device half*          output,
    threadgroup half*     inputScratch,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 2;
    constexpr uint ROWS_PER_TG_LOCAL = 4;
    constexpr uint THREADS_PER_TG_LOCAL = 64;

    uint tid = simd_group * 32 + simd_lane;
    for (uint idx = tid; idx < FIXED_COLS; idx += THREADS_PER_TG_LOCAL) {
        inputScratch[idx] = input[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;

    float g0 = 0.0f, g1 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f;

    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* gr1 = gr0 + U16_PER_ROW;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* ur1 = ur0 + U16_PER_ROW;

    device const half* gs0 = gate_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* gs1 = gs0 + NUM_COL_GROUPS;
    device const half* gb0 = gate_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* gb1 = gb0 + NUM_COL_GROUPS;
    device const half* us0 = up_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* us1 = us0 + NUM_COL_GROUPS;
    device const half* ub0 = up_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* ub1 = ub0 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint gidx = col / FIXED_GROUP_SIZE;
        threadgroup const half* xp = inputScratch + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

        g0 += float(gs0[gidx]) * AGENT_AFFINE_DOT16(gr0 + j * 4) + float(gb0[gidx]) * xsum;
        u0 += float(us0[gidx]) * AGENT_AFFINE_DOT16(ur0 + j * 4) + float(ub0[gidx]) * xsum;
        g1 += float(gs1[gidx]) * AGENT_AFFINE_DOT16(gr1 + j * 4) + float(gb1[gidx]) * xsum;
        u1 += float(us1[gidx]) * AGENT_AFFINE_DOT16(ur1 + j * 4) + float(ub1[gidx]) * xsum;
    }

    g0 = simd_sum(g0);
    g1 = simd_sum(g1);
    u0 = simd_sum(u0);
    u1 = simd_sum(u1);

    if (simd_lane == 0) {
        output[sgBaseRow] = half(g0 / (1.0f + fast::exp(-g0)) * u0);
        output[sgBaseRow + 1] = half(g1 / (1.0f + fast::exp(-g1)) * u1);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE, uint ROWS_PER_TG_LOCAL>
inline void fused_affine_gate_up_swiglu_fixed_rows4sg_tiled(
    device const uint8_t* gate_weights,
    device const half*    gate_scales,
    device const half*    gate_biases,
    device const uint8_t* up_weights,
    device const half*    up_scales,
    device const half*    up_biases,
    device const half*    input,
    device half*          output,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 4;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;

    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;

    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* gr1 = gr0 + U16_PER_ROW;
    device const uint16_t* gr2 = gr1 + U16_PER_ROW;
    device const uint16_t* gr3 = gr2 + U16_PER_ROW;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* ur1 = ur0 + U16_PER_ROW;
    device const uint16_t* ur2 = ur1 + U16_PER_ROW;
    device const uint16_t* ur3 = ur2 + U16_PER_ROW;

    device const half* gs0 = gate_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* gs1 = gs0 + NUM_COL_GROUPS;
    device const half* gs2 = gs1 + NUM_COL_GROUPS;
    device const half* gs3 = gs2 + NUM_COL_GROUPS;
    device const half* gb0 = gate_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* gb1 = gb0 + NUM_COL_GROUPS;
    device const half* gb2 = gb1 + NUM_COL_GROUPS;
    device const half* gb3 = gb2 + NUM_COL_GROUPS;
    device const half* us0 = up_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* us1 = us0 + NUM_COL_GROUPS;
    device const half* us2 = us1 + NUM_COL_GROUPS;
    device const half* us3 = us2 + NUM_COL_GROUPS;
    device const half* ub0 = up_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* ub1 = ub0 + NUM_COL_GROUPS;
    device const half* ub2 = ub1 + NUM_COL_GROUPS;
    device const half* ub3 = ub2 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint gidx = col / FIXED_GROUP_SIZE;
        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

        g0 += float(gs0[gidx]) * AGENT_AFFINE_DOT16(gr0 + j * 4) + float(gb0[gidx]) * xsum;
        u0 += float(us0[gidx]) * AGENT_AFFINE_DOT16(ur0 + j * 4) + float(ub0[gidx]) * xsum;
        g1 += float(gs1[gidx]) * AGENT_AFFINE_DOT16(gr1 + j * 4) + float(gb1[gidx]) * xsum;
        u1 += float(us1[gidx]) * AGENT_AFFINE_DOT16(ur1 + j * 4) + float(ub1[gidx]) * xsum;
        g2 += float(gs2[gidx]) * AGENT_AFFINE_DOT16(gr2 + j * 4) + float(gb2[gidx]) * xsum;
        u2 += float(us2[gidx]) * AGENT_AFFINE_DOT16(ur2 + j * 4) + float(ub2[gidx]) * xsum;
        g3 += float(gs3[gidx]) * AGENT_AFFINE_DOT16(gr3 + j * 4) + float(gb3[gidx]) * xsum;
        u3 += float(us3[gidx]) * AGENT_AFFINE_DOT16(ur3 + j * 4) + float(ub3[gidx]) * xsum;
    }

    g0 = simd_sum(g0);
    g1 = simd_sum(g1);
    g2 = simd_sum(g2);
    g3 = simd_sum(g3);
    u0 = simd_sum(u0);
    u1 = simd_sum(u1);
    u2 = simd_sum(u2);
    u3 = simd_sum(u3);

    if (simd_lane == 0) {
        output[sgBaseRow] = half(g0 / (1.0f + fast::exp(-g0)) * u0);
        output[sgBaseRow + 1] = half(g1 / (1.0f + fast::exp(-g1)) * u1);
        output[sgBaseRow + 2] = half(g2 / (1.0f + fast::exp(-g2)) * u2);
        output[sgBaseRow + 3] = half(g3 / (1.0f + fast::exp(-g3)) * u3);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_affine_gate_up_swiglu_fixed_rows4_sg1(
    device const uint8_t* gate_weights,
    device const half*    gate_scales,
    device const half*    gate_biases,
    device const uint8_t* up_weights,
    device const half*    up_scales,
    device const half*    up_biases,
    device const half*    input,
    device half*          output,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_TG_LOCAL = 4;

    uint row = tgid * ROWS_PER_TG_LOCAL + simd_group;

    float gacc = 0.0f;
    float uacc = 0.0f;

    device const uint16_t* gr = (device const uint16_t*)(gate_weights + row * HALF_COLS);
    device const uint16_t* ur = (device const uint16_t*)(up_weights + row * HALF_COLS);

    device const half* gs = gate_scales + row * NUM_COL_GROUPS;
    device const half* gb = gate_biases + row * NUM_COL_GROUPS;
    device const half* us = up_scales + row * NUM_COL_GROUPS;
    device const half* ub = up_biases + row * NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint group = col / FIXED_GROUP_SIZE;
        device const half* xp = input + col;
        float x0 = float(xp[0]), x1 = float(xp[1]), x2 = float(xp[2]), x3 = float(xp[3]);
        float x4 = float(xp[4]), x5 = float(xp[5]), x6 = float(xp[6]), x7 = float(xp[7]);
        float x8 = float(xp[8]), x9 = float(xp[9]), x10 = float(xp[10]), x11 = float(xp[11]);
        float x12 = float(xp[12]), x13 = float(xp[13]), x14 = float(xp[14]), x15 = float(xp[15]);
        float xsum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
                   + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15;
        x1 *= (1.0f / 16.0f);
        x2 *= (1.0f / 256.0f);
        x3 *= (1.0f / 4096.0f);
        x5 *= (1.0f / 16.0f);
        x6 *= (1.0f / 256.0f);
        x7 *= (1.0f / 4096.0f);
        x9 *= (1.0f / 16.0f);
        x10 *= (1.0f / 256.0f);
        x11 *= (1.0f / 4096.0f);
        x13 *= (1.0f / 16.0f);
        x14 *= (1.0f / 256.0f);
        x15 *= (1.0f / 4096.0f);

        gacc += float(gs[group]) * AGENT_AFFINE_DOT16(gr + j * 4) + float(gb[group]) * xsum;
        uacc += float(us[group]) * AGENT_AFFINE_DOT16(ur + j * 4) + float(ub[group]) * xsum;
    }

    gacc = simd_sum(gacc);
    uacc = simd_sum(uacc);

    if (simd_lane == 0) {
        output[row] = half(gacc / (1.0f + fast::exp(-gacc)) * uacc);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_affine_gate_up_swiglu_fixed_rows8_sg2(
    device const uint8_t* gate_weights,
    device const half*    gate_scales,
    device const half*    gate_biases,
    device const uint8_t* up_weights,
    device const half*    up_scales,
    device const half*    up_biases,
    device const half*    input,
    device half*          output,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 2;
    constexpr uint ROWS_PER_TG_LOCAL = 8;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;

    float g0 = 0.0f, g1 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f;

    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* gr1 = gr0 + U16_PER_ROW;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* ur1 = ur0 + U16_PER_ROW;

    device const half* gs0 = gate_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* gs1 = gs0 + NUM_COL_GROUPS;
    device const half* gb0 = gate_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* gb1 = gb0 + NUM_COL_GROUPS;
    device const half* us0 = up_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* us1 = us0 + NUM_COL_GROUPS;
    device const half* ub0 = up_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* ub1 = ub0 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint gidx = col / FIXED_GROUP_SIZE;
        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

        g0 += float(gs0[gidx]) * AGENT_AFFINE_DOT16(gr0 + j * 4) + float(gb0[gidx]) * xsum;
        u0 += float(us0[gidx]) * AGENT_AFFINE_DOT16(ur0 + j * 4) + float(ub0[gidx]) * xsum;
        g1 += float(gs1[gidx]) * AGENT_AFFINE_DOT16(gr1 + j * 4) + float(gb1[gidx]) * xsum;
        u1 += float(us1[gidx]) * AGENT_AFFINE_DOT16(ur1 + j * 4) + float(ub1[gidx]) * xsum;
    }

    g0 = simd_sum(g0);
    g1 = simd_sum(g1);
    u0 = simd_sum(u0);
    u1 = simd_sum(u1);

    if (simd_lane == 0) {
        output[sgBaseRow] = half(g0 / (1.0f + fast::exp(-g0)) * u0);
        output[sgBaseRow + 1] = half(g1 / (1.0f + fast::exp(-g1)) * u1);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_affine_gate_up_swiglu_fixed_rows4_hacc(
    device const uint8_t* gate_weights,
    device const half*    gate_scales,
    device const half*    gate_biases,
    device const uint8_t* up_weights,
    device const half*    up_scales,
    device const half*    up_biases,
    device const half*    input,
    device half*          output,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 2;
    constexpr uint ROWS_PER_TG_LOCAL = 4;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;

    half g0 = half(0.0h), g1 = half(0.0h);
    half u0 = half(0.0h), u1 = half(0.0h);

    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* gr1 = gr0 + U16_PER_ROW;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* ur1 = ur0 + U16_PER_ROW;

    device const half* gs0 = gate_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* gs1 = gs0 + NUM_COL_GROUPS;
    device const half* gb0 = gate_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* gb1 = gb0 + NUM_COL_GROUPS;
    device const half* us0 = up_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* us1 = us0 + NUM_COL_GROUPS;
    device const half* ub0 = up_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* ub1 = ub0 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint gidx = col / FIXED_GROUP_SIZE;
        device const half* xp = input + col;
        half x0=xp[0],x1=xp[1],x2=xp[2],x3=xp[3];
        half x4=xp[4],x5=xp[5],x6=xp[6],x7=xp[7];
        half x8=xp[8],x9=xp[9],x10=xp[10],x11=xp[11];
        half x12=xp[12],x13=xp[13],x14=xp[14],x15=xp[15];
        half xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=half(1.0h/16.0h); x2*=half(1.0h/256.0h); x3*=half(1.0h/4096.0h);
        x5*=half(1.0h/16.0h); x6*=half(1.0h/256.0h); x7*=half(1.0h/4096.0h);
        x9*=half(1.0h/16.0h); x10*=half(1.0h/256.0h); x11*=half(1.0h/4096.0h);
        x13*=half(1.0h/16.0h); x14*=half(1.0h/256.0h); x15*=half(1.0h/4096.0h);

#define AGENT_AFFINE_DOT16H(w) \
    ( x0*half(float((w)[0]&0x000Fu)) + x1*half(float((w)[0]&0x00F0u)) \
    + x2*half(float((w)[0]&0x0F00u)) + x3*half(float((w)[0]&0xF000u)) \
    + x4*half(float((w)[1]&0x000Fu)) + x5*half(float((w)[1]&0x00F0u)) \
    + x6*half(float((w)[1]&0x0F00u)) + x7*half(float((w)[1]&0xF000u)) \
    + x8*half(float((w)[2]&0x000Fu)) + x9*half(float((w)[2]&0x00F0u)) \
    + x10*half(float((w)[2]&0x0F00u)) + x11*half(float((w)[2]&0xF000u)) \
    + x12*half(float((w)[3]&0x000Fu)) + x13*half(float((w)[3]&0x00F0u)) \
    + x14*half(float((w)[3]&0x0F00u)) + x15*half(float((w)[3]&0xF000u)) )

        g0 += gs0[gidx] * AGENT_AFFINE_DOT16H(gr0 + j * 4) + gb0[gidx] * xsum;
        u0 += us0[gidx] * AGENT_AFFINE_DOT16H(ur0 + j * 4) + ub0[gidx] * xsum;
        g1 += gs1[gidx] * AGENT_AFFINE_DOT16H(gr1 + j * 4) + gb1[gidx] * xsum;
        u1 += us1[gidx] * AGENT_AFFINE_DOT16H(ur1 + j * 4) + ub1[gidx] * xsum;
#undef AGENT_AFFINE_DOT16H
    }

    float g0f = simd_sum(float(g0));
    float g1f = simd_sum(float(g1));
    float u0f = simd_sum(float(u0));
    float u1f = simd_sum(float(u1));

    if (simd_lane == 0) {
        output[sgBaseRow] = half(g0f / (1.0f + fast::exp(-g0f)) * u0f);
        output[sgBaseRow + 1] = half(g1f / (1.0f + fast::exp(-g1f)) * u1f);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_affine_gate_up_geglu_fixed_rows4(
    device const uint8_t* gate_weights,
    device const half*    gate_scales,
    device const half*    gate_biases,
    device const uint8_t* up_weights,
    device const half*    up_scales,
    device const half*    up_biases,
    device const half*    input,
    device half*          output,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 2;
    constexpr uint ROWS_PER_TG_LOCAL = 4;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;

    float g0 = 0.0f, g1 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f;

    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* gr1 = gr0 + U16_PER_ROW;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* ur1 = ur0 + U16_PER_ROW;

    device const half* gs0 = gate_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* gs1 = gs0 + NUM_COL_GROUPS;
    device const half* gb0 = gate_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* gb1 = gb0 + NUM_COL_GROUPS;
    device const half* us0 = up_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* us1 = us0 + NUM_COL_GROUPS;
    device const half* ub0 = up_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* ub1 = ub0 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint gidx = col / FIXED_GROUP_SIZE;
        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

        g0 += float(gs0[gidx]) * AGENT_AFFINE_DOT16(gr0 + j * 4) + float(gb0[gidx]) * xsum;
        u0 += float(us0[gidx]) * AGENT_AFFINE_DOT16(ur0 + j * 4) + float(ub0[gidx]) * xsum;
        g1 += float(gs1[gidx]) * AGENT_AFFINE_DOT16(gr1 + j * 4) + float(gb1[gidx]) * xsum;
        u1 += float(us1[gidx]) * AGENT_AFFINE_DOT16(ur1 + j * 4) + float(ub1[gidx]) * xsum;
    }

    g0 = simd_sum(g0);
    g1 = simd_sum(g1);
    u0 = simd_sum(u0);
    u1 = simd_sum(u1);

    if (simd_lane == 0) {
        output[sgBaseRow] = agent_geglu_product(g0, u0);
        output[sgBaseRow + 1] = agent_geglu_product(g1, u1);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void norm_scale_affine_gate_up_geglu_fixed_rows4(
    device const float*    scalePtr,
    device const half*     normInput,
    device const half*     normWeight,
    device const uint8_t*  gate_weights,
    device const half*     gate_scales,
    device const half*     gate_biases,
    device const uint8_t*  up_weights,
    device const half*     up_scales,
    device const half*     up_biases,
    device half*           output,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 2;
    constexpr uint ROWS_PER_TG_LOCAL = 4;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;
    float rs = scalePtr[0];

    float g0 = 0.0f, g1 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f;

    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* gr1 = gr0 + U16_PER_ROW;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* ur1 = ur0 + U16_PER_ROW;

    device const half* gs0 = gate_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* gs1 = gs0 + NUM_COL_GROUPS;
    device const half* gb0 = gate_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* gb1 = gb0 + NUM_COL_GROUPS;
    device const half* us0 = up_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* us1 = us0 + NUM_COL_GROUPS;
    device const half* ub0 = up_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* ub1 = ub0 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint gidx = col / FIXED_GROUP_SIZE;
        device const half* xp = normInput + col;
        device const half* wp = normWeight + col;

        float x0  = float(xp[0])  * rs * (1.0f + float(wp[0]));
        float x1  = float(xp[1])  * rs * (1.0f + float(wp[1]));
        float x2  = float(xp[2])  * rs * (1.0f + float(wp[2]));
        float x3  = float(xp[3])  * rs * (1.0f + float(wp[3]));
        float x4  = float(xp[4])  * rs * (1.0f + float(wp[4]));
        float x5  = float(xp[5])  * rs * (1.0f + float(wp[5]));
        float x6  = float(xp[6])  * rs * (1.0f + float(wp[6]));
        float x7  = float(xp[7])  * rs * (1.0f + float(wp[7]));
        float x8  = float(xp[8])  * rs * (1.0f + float(wp[8]));
        float x9  = float(xp[9])  * rs * (1.0f + float(wp[9]));
        float x10 = float(xp[10]) * rs * (1.0f + float(wp[10]));
        float x11 = float(xp[11]) * rs * (1.0f + float(wp[11]));
        float x12 = float(xp[12]) * rs * (1.0f + float(wp[12]));
        float x13 = float(xp[13]) * rs * (1.0f + float(wp[13]));
        float x14 = float(xp[14]) * rs * (1.0f + float(wp[14]));
        float x15 = float(xp[15]) * rs * (1.0f + float(wp[15]));

        half hx0 = half(x0);
        half hx1 = half(x1);
        half hx2 = half(x2);
        half hx3 = half(x3);
        half hx4 = half(x4);
        half hx5 = half(x5);
        half hx6 = half(x6);
        half hx7 = half(x7);
        half hx8 = half(x8);
        half hx9 = half(x9);
        half hx10 = half(x10);
        half hx11 = half(x11);
        half hx12 = half(x12);
        half hx13 = half(x13);
        half hx14 = half(x14);
        half hx15 = half(x15);

        x0 = float(hx0);
        x1 = float(hx1);
        x2 = float(hx2);
        x3 = float(hx3);
        x4 = float(hx4);
        x5 = float(hx5);
        x6 = float(hx6);
        x7 = float(hx7);
        x8 = float(hx8);
        x9 = float(hx9);
        x10 = float(hx10);
        x11 = float(hx11);
        x12 = float(hx12);
        x13 = float(hx13);
        x14 = float(hx14);
        x15 = float(hx15);

        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

        g0 += float(gs0[gidx]) * AGENT_AFFINE_DOT16(gr0 + j * 4) + float(gb0[gidx]) * xsum;
        u0 += float(us0[gidx]) * AGENT_AFFINE_DOT16(ur0 + j * 4) + float(ub0[gidx]) * xsum;
        g1 += float(gs1[gidx]) * AGENT_AFFINE_DOT16(gr1 + j * 4) + float(gb1[gidx]) * xsum;
        u1 += float(us1[gidx]) * AGENT_AFFINE_DOT16(ur1 + j * 4) + float(ub1[gidx]) * xsum;
    }

    g0 = simd_sum(g0);
    g1 = simd_sum(g1);
    u0 = simd_sum(u0);
    u1 = simd_sum(u1);

    if (simd_lane == 0) {
        output[sgBaseRow] = agent_geglu_product(g0, u0);
        output[sgBaseRow + 1] = agent_geglu_product(g1, u1);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE>
inline void fused_affine_gate_up_geglu_fixed_rows8(
    device const uint8_t* gate_weights,
    device const half*    gate_scales,
    device const half*    gate_biases,
    device const uint8_t* up_weights,
    device const half*    up_scales,
    device const half*    up_biases,
    device const half*    input,
    device half*          output,
    uint tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_SG_LOCAL = 4;
    constexpr uint ROWS_PER_TG_LOCAL = 8;

    uint baseRow = tgid * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;

    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;

    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* gr1 = gr0 + U16_PER_ROW;
    device const uint16_t* gr2 = gr1 + U16_PER_ROW;
    device const uint16_t* gr3 = gr2 + U16_PER_ROW;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* ur1 = ur0 + U16_PER_ROW;
    device const uint16_t* ur2 = ur1 + U16_PER_ROW;
    device const uint16_t* ur3 = ur2 + U16_PER_ROW;

    device const half* gs0 = gate_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* gs1 = gs0 + NUM_COL_GROUPS;
    device const half* gs2 = gs1 + NUM_COL_GROUPS;
    device const half* gs3 = gs2 + NUM_COL_GROUPS;
    device const half* gb0 = gate_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* gb1 = gb0 + NUM_COL_GROUPS;
    device const half* gb2 = gb1 + NUM_COL_GROUPS;
    device const half* gb3 = gb2 + NUM_COL_GROUPS;
    device const half* us0 = up_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* us1 = us0 + NUM_COL_GROUPS;
    device const half* us2 = us1 + NUM_COL_GROUPS;
    device const half* us3 = us2 + NUM_COL_GROUPS;
    device const half* ub0 = up_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* ub1 = ub0 + NUM_COL_GROUPS;
    device const half* ub2 = ub1 + NUM_COL_GROUPS;
    device const half* ub3 = ub2 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint gidx = col / FIXED_GROUP_SIZE;
        device const half* xp = input + col;
        float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
        float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
        float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
        float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
        float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
        x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
        x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
        x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
        x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

        g0 += float(gs0[gidx]) * AGENT_AFFINE_DOT16(gr0 + j * 4) + float(gb0[gidx]) * xsum;
        u0 += float(us0[gidx]) * AGENT_AFFINE_DOT16(ur0 + j * 4) + float(ub0[gidx]) * xsum;
        g1 += float(gs1[gidx]) * AGENT_AFFINE_DOT16(gr1 + j * 4) + float(gb1[gidx]) * xsum;
        u1 += float(us1[gidx]) * AGENT_AFFINE_DOT16(ur1 + j * 4) + float(ub1[gidx]) * xsum;
        g2 += float(gs2[gidx]) * AGENT_AFFINE_DOT16(gr2 + j * 4) + float(gb2[gidx]) * xsum;
        u2 += float(us2[gidx]) * AGENT_AFFINE_DOT16(ur2 + j * 4) + float(ub2[gidx]) * xsum;
        g3 += float(gs3[gidx]) * AGENT_AFFINE_DOT16(gr3 + j * 4) + float(gb3[gidx]) * xsum;
        u3 += float(us3[gidx]) * AGENT_AFFINE_DOT16(ur3 + j * 4) + float(ub3[gidx]) * xsum;
    }

    g0 = simd_sum(g0);
    g1 = simd_sum(g1);
    g2 = simd_sum(g2);
    g3 = simd_sum(g3);
    u0 = simd_sum(u0);
    u1 = simd_sum(u1);
    u2 = simd_sum(u2);
    u3 = simd_sum(u3);

    if (simd_lane == 0) {
        output[sgBaseRow] = agent_geglu_product(g0, u0);
        output[sgBaseRow + 1] = agent_geglu_product(g1, u1);
        output[sgBaseRow + 2] = agent_geglu_product(g2, u2);
        output[sgBaseRow + 3] = agent_geglu_product(g3, u3);
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE, uint BATCH_TILE>
inline void fused_affine_gate_up_swiglu_fixed_batched(
    device const uint8_t* gate_weights,
    device const half*    gate_scales,
    device const half*    gate_biases,
    device const uint8_t* up_weights,
    device const half*    up_scales,
    device const half*    up_biases,
    device const half*    input,
    device half*          output,
    uint2 tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;

    uint baseRow = tgid.x * ROWS_PER_TG;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint batchBase = tgid.y * BATCH_TILE;

    float gateAcc[BATCH_TILE][ROWS_PER_SG];
    float upAcc[BATCH_TILE][ROWS_PER_SG];
    for (uint b = 0; b < BATCH_TILE; b++) {
        for (uint r = 0; r < ROWS_PER_SG; r++) {
            gateAcc[b][r] = 0.0f;
            upAcc[b][r] = 0.0f;
        }
    }

    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* gr1 = gr0 + U16_PER_ROW;
    device const uint16_t* gr2 = gr1 + U16_PER_ROW;
    device const uint16_t* gr3 = gr2 + U16_PER_ROW;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* ur1 = ur0 + U16_PER_ROW;
    device const uint16_t* ur2 = ur1 + U16_PER_ROW;
    device const uint16_t* ur3 = ur2 + U16_PER_ROW;

    device const half* gs0 = gate_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* gs1 = gs0 + NUM_COL_GROUPS;
    device const half* gs2 = gs1 + NUM_COL_GROUPS;
    device const half* gs3 = gs2 + NUM_COL_GROUPS;
    device const half* gb0 = gate_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* gb1 = gb0 + NUM_COL_GROUPS;
    device const half* gb2 = gb1 + NUM_COL_GROUPS;
    device const half* gb3 = gb2 + NUM_COL_GROUPS;
    device const half* us0 = up_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* us1 = us0 + NUM_COL_GROUPS;
    device const half* us2 = us1 + NUM_COL_GROUPS;
    device const half* us3 = us2 + NUM_COL_GROUPS;
    device const half* ub0 = up_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* ub1 = ub0 + NUM_COL_GROUPS;
    device const half* ub2 = ub1 + NUM_COL_GROUPS;
    device const half* ub3 = ub2 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint gidx = col / FIXED_GROUP_SIZE;
        device const uint16_t* gw0 = gr0 + j * 4;
        device const uint16_t* gw1 = gr1 + j * 4;
        device const uint16_t* gw2 = gr2 + j * 4;
        device const uint16_t* gw3 = gr3 + j * 4;
        device const uint16_t* uw0 = ur0 + j * 4;
        device const uint16_t* uw1 = ur1 + j * 4;
        device const uint16_t* uw2 = ur2 + j * 4;
        device const uint16_t* uw3 = ur3 + j * 4;

        for (uint b = 0; b < BATCH_TILE; b++) {
            device const half* xp = input + (batchBase + b) * FIXED_COLS + col;
            float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
            float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
            float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
            float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
            float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
            x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
            x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
            x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
            x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

            gateAcc[b][0] += float(gs0[gidx]) * AGENT_AFFINE_DOT16(gw0) + float(gb0[gidx]) * xsum;
            upAcc[b][0] += float(us0[gidx]) * AGENT_AFFINE_DOT16(uw0) + float(ub0[gidx]) * xsum;
            gateAcc[b][1] += float(gs1[gidx]) * AGENT_AFFINE_DOT16(gw1) + float(gb1[gidx]) * xsum;
            upAcc[b][1] += float(us1[gidx]) * AGENT_AFFINE_DOT16(uw1) + float(ub1[gidx]) * xsum;
            gateAcc[b][2] += float(gs2[gidx]) * AGENT_AFFINE_DOT16(gw2) + float(gb2[gidx]) * xsum;
            upAcc[b][2] += float(us2[gidx]) * AGENT_AFFINE_DOT16(uw2) + float(ub2[gidx]) * xsum;
            gateAcc[b][3] += float(gs3[gidx]) * AGENT_AFFINE_DOT16(gw3) + float(gb3[gidx]) * xsum;
            upAcc[b][3] += float(us3[gidx]) * AGENT_AFFINE_DOT16(uw3) + float(ub3[gidx]) * xsum;
        }
    }

    for (uint b = 0; b < BATCH_TILE; b++) {
        gateAcc[b][0] = simd_sum(gateAcc[b][0]);
        gateAcc[b][1] = simd_sum(gateAcc[b][1]);
        gateAcc[b][2] = simd_sum(gateAcc[b][2]);
        gateAcc[b][3] = simd_sum(gateAcc[b][3]);
        upAcc[b][0] = simd_sum(upAcc[b][0]);
        upAcc[b][1] = simd_sum(upAcc[b][1]);
        upAcc[b][2] = simd_sum(upAcc[b][2]);
        upAcc[b][3] = simd_sum(upAcc[b][3]);
    }

    if (simd_lane == 0) {
        for (uint b = 0; b < BATCH_TILE; b++) {
            device half* out = output + (batchBase + b) * FIXED_ROWS + sgBaseRow;
            out[0] = half(gateAcc[b][0] / (1.0f + exp(-gateAcc[b][0])) * upAcc[b][0]);
            out[1] = half(gateAcc[b][1] / (1.0f + exp(-gateAcc[b][1])) * upAcc[b][1]);
            out[2] = half(gateAcc[b][2] / (1.0f + exp(-gateAcc[b][2])) * upAcc[b][2]);
            out[3] = half(gateAcc[b][3] / (1.0f + exp(-gateAcc[b][3])) * upAcc[b][3]);
        }
    }
}

template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE, uint BATCH_TILE>
inline void fused_affine_gate_up_swiglu_fixed_batched_guarded(
    device const uint8_t* gate_weights,
    device const half*    gate_scales,
    device const half*    gate_biases,
    device const uint8_t* up_weights,
    device const half*    up_scales,
    device const half*    up_biases,
    device const half*    input,
    device half*          output,
    constant uint&        actualBatch,
    uint2 tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;

    uint baseRow = tgid.x * ROWS_PER_TG;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint batchBase = tgid.y * BATCH_TILE;
    bool fullTile = batchBase + BATCH_TILE <= actualBatch;

    float gateAcc[BATCH_TILE][ROWS_PER_SG];
    float upAcc[BATCH_TILE][ROWS_PER_SG];
    for (uint b = 0; b < BATCH_TILE; b++) {
        for (uint r = 0; r < ROWS_PER_SG; r++) {
            gateAcc[b][r] = 0.0f;
            upAcc[b][r] = 0.0f;
        }
    }

    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* gr1 = gr0 + U16_PER_ROW;
    device const uint16_t* gr2 = gr1 + U16_PER_ROW;
    device const uint16_t* gr3 = gr2 + U16_PER_ROW;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* ur1 = ur0 + U16_PER_ROW;
    device const uint16_t* ur2 = ur1 + U16_PER_ROW;
    device const uint16_t* ur3 = ur2 + U16_PER_ROW;

    device const half* gs0 = gate_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* gs1 = gs0 + NUM_COL_GROUPS;
    device const half* gs2 = gs1 + NUM_COL_GROUPS;
    device const half* gs3 = gs2 + NUM_COL_GROUPS;
    device const half* gb0 = gate_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* gb1 = gb0 + NUM_COL_GROUPS;
    device const half* gb2 = gb1 + NUM_COL_GROUPS;
    device const half* gb3 = gb2 + NUM_COL_GROUPS;
    device const half* us0 = up_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* us1 = us0 + NUM_COL_GROUPS;
    device const half* us2 = us1 + NUM_COL_GROUPS;
    device const half* us3 = us2 + NUM_COL_GROUPS;
    device const half* ub0 = up_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* ub1 = ub0 + NUM_COL_GROUPS;
    device const half* ub2 = ub1 + NUM_COL_GROUPS;
    device const half* ub3 = ub2 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint gidx = col / FIXED_GROUP_SIZE;
        device const uint16_t* gw0 = gr0 + j * 4;
        device const uint16_t* gw1 = gr1 + j * 4;
        device const uint16_t* gw2 = gr2 + j * 4;
        device const uint16_t* gw3 = gr3 + j * 4;
        device const uint16_t* uw0 = ur0 + j * 4;
        device const uint16_t* uw1 = ur1 + j * 4;
        device const uint16_t* uw2 = ur2 + j * 4;
        device const uint16_t* uw3 = ur3 + j * 4;

        for (uint b = 0; b < BATCH_TILE; b++) {
            if (!fullTile && batchBase + b >= actualBatch) continue;
            device const half* xp = input + (batchBase + b) * FIXED_COLS + col;
            float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
            float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
            float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
            float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
            float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
            x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
            x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
            x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
            x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

            gateAcc[b][0] += float(gs0[gidx]) * AGENT_AFFINE_DOT16(gw0) + float(gb0[gidx]) * xsum;
            upAcc[b][0] += float(us0[gidx]) * AGENT_AFFINE_DOT16(uw0) + float(ub0[gidx]) * xsum;
            gateAcc[b][1] += float(gs1[gidx]) * AGENT_AFFINE_DOT16(gw1) + float(gb1[gidx]) * xsum;
            upAcc[b][1] += float(us1[gidx]) * AGENT_AFFINE_DOT16(uw1) + float(ub1[gidx]) * xsum;
            gateAcc[b][2] += float(gs2[gidx]) * AGENT_AFFINE_DOT16(gw2) + float(gb2[gidx]) * xsum;
            upAcc[b][2] += float(us2[gidx]) * AGENT_AFFINE_DOT16(uw2) + float(ub2[gidx]) * xsum;
            gateAcc[b][3] += float(gs3[gidx]) * AGENT_AFFINE_DOT16(gw3) + float(gb3[gidx]) * xsum;
            upAcc[b][3] += float(us3[gidx]) * AGENT_AFFINE_DOT16(uw3) + float(ub3[gidx]) * xsum;
        }
    }

    for (uint b = 0; b < BATCH_TILE; b++) {
        gateAcc[b][0] = simd_sum(gateAcc[b][0]);
        gateAcc[b][1] = simd_sum(gateAcc[b][1]);
        gateAcc[b][2] = simd_sum(gateAcc[b][2]);
        gateAcc[b][3] = simd_sum(gateAcc[b][3]);
        upAcc[b][0] = simd_sum(upAcc[b][0]);
        upAcc[b][1] = simd_sum(upAcc[b][1]);
        upAcc[b][2] = simd_sum(upAcc[b][2]);
        upAcc[b][3] = simd_sum(upAcc[b][3]);
    }

    if (simd_lane == 0) {
        for (uint b = 0; b < BATCH_TILE; b++) {
            if (!fullTile && batchBase + b >= actualBatch) continue;
            device half* out = output + (batchBase + b) * FIXED_ROWS + sgBaseRow;
            out[0] = half(gateAcc[b][0] / (1.0f + exp(-gateAcc[b][0])) * upAcc[b][0]);
            out[1] = half(gateAcc[b][1] / (1.0f + exp(-gateAcc[b][1])) * upAcc[b][1]);
            out[2] = half(gateAcc[b][2] / (1.0f + exp(-gateAcc[b][2])) * upAcc[b][2]);
            out[3] = half(gateAcc[b][3] / (1.0f + exp(-gateAcc[b][3])) * upAcc[b][3]);
        }
    }
}

// SG_COUNT = simdgroups per threadgroup. The body otherwise mirrors
// the legacy 2-SG path; ROWS_PER_TG_LOCAL = SG_COUNT * ROWS_PER_SG so
// each simdgroup still owns ROWS_PER_SG=4 consecutive rows. Increasing
// SG_COUNT widens the threadgroup (better memory-latency hiding at the
// cost of register pressure per TG and fewer threadgroups in flight).
template <uint FIXED_ROWS, uint FIXED_COLS, uint FIXED_GROUP_SIZE, uint BATCH_TILE, uint SG_COUNT>
inline void fused_affine_gate_up_geglu_fixed_batched(
    device const uint8_t* gate_weights,
    device const half*    gate_scales,
    device const half*    gate_biases,
    device const uint8_t* up_weights,
    device const half*    up_scales,
    device const half*    up_biases,
    device const half*    input,
    device half*          output,
    constant uint&        actualBatch,
    uint2 tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    constexpr uint ROWS_PER_TG_LOCAL = SG_COUNT * ROWS_PER_SG;

    uint baseRow = tgid.x * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG;
    uint batchBase = tgid.y * BATCH_TILE;
    bool fullTile = batchBase + BATCH_TILE <= actualBatch;

    float gateAcc[BATCH_TILE][ROWS_PER_SG];
    float upAcc[BATCH_TILE][ROWS_PER_SG];
    for (uint b = 0; b < BATCH_TILE; b++) {
        for (uint r = 0; r < ROWS_PER_SG; r++) {
            gateAcc[b][r] = 0.0f;
            upAcc[b][r] = 0.0f;
        }
    }

    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* gr1 = gr0 + U16_PER_ROW;
    device const uint16_t* gr2 = gr1 + U16_PER_ROW;
    device const uint16_t* gr3 = gr2 + U16_PER_ROW;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* ur1 = ur0 + U16_PER_ROW;
    device const uint16_t* ur2 = ur1 + U16_PER_ROW;
    device const uint16_t* ur3 = ur2 + U16_PER_ROW;

    device const half* gs0 = gate_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* gs1 = gs0 + NUM_COL_GROUPS;
    device const half* gs2 = gs1 + NUM_COL_GROUPS;
    device const half* gs3 = gs2 + NUM_COL_GROUPS;
    device const half* gb0 = gate_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* gb1 = gb0 + NUM_COL_GROUPS;
    device const half* gb2 = gb1 + NUM_COL_GROUPS;
    device const half* gb3 = gb2 + NUM_COL_GROUPS;
    device const half* us0 = up_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* us1 = us0 + NUM_COL_GROUPS;
    device const half* us2 = us1 + NUM_COL_GROUPS;
    device const half* us3 = us2 + NUM_COL_GROUPS;
    device const half* ub0 = up_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* ub1 = ub0 + NUM_COL_GROUPS;
    device const half* ub2 = ub1 + NUM_COL_GROUPS;
    device const half* ub3 = ub2 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint gidx = col / FIXED_GROUP_SIZE;
        device const uint16_t* gw0 = gr0 + j * 4;
        device const uint16_t* gw1 = gr1 + j * 4;
        device const uint16_t* gw2 = gr2 + j * 4;
        device const uint16_t* gw3 = gr3 + j * 4;
        device const uint16_t* uw0 = ur0 + j * 4;
        device const uint16_t* uw1 = ur1 + j * 4;
        device const uint16_t* uw2 = ur2 + j * 4;
        device const uint16_t* uw3 = ur3 + j * 4;

        for (uint b = 0; b < BATCH_TILE; b++) {
            if (!fullTile && batchBase + b >= actualBatch) continue;
            device const half* xp = input + (batchBase + b) * FIXED_COLS + col;
            float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
            float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
            float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
            float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
            float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
            x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
            x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
            x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
            x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

            gateAcc[b][0] += float(gs0[gidx]) * AGENT_AFFINE_DOT16(gw0) + float(gb0[gidx]) * xsum;
            upAcc[b][0] += float(us0[gidx]) * AGENT_AFFINE_DOT16(uw0) + float(ub0[gidx]) * xsum;
            gateAcc[b][1] += float(gs1[gidx]) * AGENT_AFFINE_DOT16(gw1) + float(gb1[gidx]) * xsum;
            upAcc[b][1] += float(us1[gidx]) * AGENT_AFFINE_DOT16(uw1) + float(ub1[gidx]) * xsum;
            gateAcc[b][2] += float(gs2[gidx]) * AGENT_AFFINE_DOT16(gw2) + float(gb2[gidx]) * xsum;
            upAcc[b][2] += float(us2[gidx]) * AGENT_AFFINE_DOT16(uw2) + float(ub2[gidx]) * xsum;
            gateAcc[b][3] += float(gs3[gidx]) * AGENT_AFFINE_DOT16(gw3) + float(gb3[gidx]) * xsum;
            upAcc[b][3] += float(us3[gidx]) * AGENT_AFFINE_DOT16(uw3) + float(ub3[gidx]) * xsum;
        }
    }

    for (uint b = 0; b < BATCH_TILE; b++) {
        gateAcc[b][0] = simd_sum(gateAcc[b][0]);
        gateAcc[b][1] = simd_sum(gateAcc[b][1]);
        gateAcc[b][2] = simd_sum(gateAcc[b][2]);
        gateAcc[b][3] = simd_sum(gateAcc[b][3]);
        upAcc[b][0] = simd_sum(upAcc[b][0]);
        upAcc[b][1] = simd_sum(upAcc[b][1]);
        upAcc[b][2] = simd_sum(upAcc[b][2]);
        upAcc[b][3] = simd_sum(upAcc[b][3]);
    }

    if (simd_lane == 0) {
        for (uint b = 0; b < BATCH_TILE; b++) {
            if (!fullTile && batchBase + b >= actualBatch) continue;
            device half* out = output + (batchBase + b) * FIXED_ROWS + sgBaseRow;
            out[0] = agent_geglu_product(gateAcc[b][0], upAcc[b][0]);
            out[1] = agent_geglu_product(gateAcc[b][1], upAcc[b][1]);
            out[2] = agent_geglu_product(gateAcc[b][2], upAcc[b][2]);
            out[3] = agent_geglu_product(gateAcc[b][3], upAcc[b][3]);
        }
    }
}

inline void fused_affine_gate_up_swiglu_fixed_batched_qwen_2x2_full(
    device const uint8_t* gate_weights,
    device const half*    gate_scales,
    device const half*    gate_biases,
    device const uint8_t* up_weights,
    device const half*    up_scales,
    device const half*    up_biases,
    device const half*    input,
    device half*          output,
    uint2 tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint FIXED_ROWS = 6144;
    constexpr uint FIXED_COLS = 2048;
    constexpr uint FIXED_GROUP_SIZE = 64;
    constexpr uint ROWS_PER_SG_LOCAL = 2;
    constexpr uint ROWS_PER_TG_LOCAL = 4;
    constexpr uint BATCH_TILE = 2;
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    uint baseRow = tgid.x * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;
    uint batchBase = tgid.y * BATCH_TILE;

    float gateAcc[BATCH_TILE][ROWS_PER_SG_LOCAL];
    float upAcc[BATCH_TILE][ROWS_PER_SG_LOCAL];
    for (uint b = 0; b < BATCH_TILE; b++) {
        for (uint r = 0; r < ROWS_PER_SG_LOCAL; r++) {
            gateAcc[b][r] = 0.0f;
            upAcc[b][r] = 0.0f;
        }
    }

    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* gr1 = gr0 + U16_PER_ROW;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* ur1 = ur0 + U16_PER_ROW;

    device const half* gs0 = gate_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* gs1 = gs0 + NUM_COL_GROUPS;
    device const half* gb0 = gate_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* gb1 = gb0 + NUM_COL_GROUPS;
    device const half* us0 = up_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* us1 = us0 + NUM_COL_GROUPS;
    device const half* ub0 = up_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* ub1 = ub0 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint gidx = col / FIXED_GROUP_SIZE;
        device const uint16_t* gw0 = gr0 + j * 4;
        device const uint16_t* gw1 = gr1 + j * 4;
        device const uint16_t* uw0 = ur0 + j * 4;
        device const uint16_t* uw1 = ur1 + j * 4;

        for (uint b = 0; b < BATCH_TILE; b++) {
            device const half* xp = input + (batchBase + b) * FIXED_COLS + col;
            float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
            float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
            float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
            float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
            float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
            x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
            x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
            x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
            x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

            gateAcc[b][0] += float(gs0[gidx]) * AGENT_AFFINE_DOT16(gw0) + float(gb0[gidx]) * xsum;
            upAcc[b][0] += float(us0[gidx]) * AGENT_AFFINE_DOT16(uw0) + float(ub0[gidx]) * xsum;
            gateAcc[b][1] += float(gs1[gidx]) * AGENT_AFFINE_DOT16(gw1) + float(gb1[gidx]) * xsum;
            upAcc[b][1] += float(us1[gidx]) * AGENT_AFFINE_DOT16(uw1) + float(ub1[gidx]) * xsum;
        }
    }

    for (uint b = 0; b < BATCH_TILE; b++) {
        gateAcc[b][0] = simd_sum(gateAcc[b][0]);
        gateAcc[b][1] = simd_sum(gateAcc[b][1]);
        upAcc[b][0] = simd_sum(upAcc[b][0]);
        upAcc[b][1] = simd_sum(upAcc[b][1]);
    }

    if (simd_lane == 0) {
        for (uint b = 0; b < BATCH_TILE; b++) {
            device half* out = output + (batchBase + b) * FIXED_ROWS + sgBaseRow;
            out[0] = half(gateAcc[b][0] / (1.0f + exp(-gateAcc[b][0])) * upAcc[b][0]);
            out[1] = half(gateAcc[b][1] / (1.0f + exp(-gateAcc[b][1])) * upAcc[b][1]);
        }
    }
}

inline void fused_affine_gate_up_swiglu_fixed_batched_qwen_2x2(
    device const uint8_t* gate_weights,
    device const half*    gate_scales,
    device const half*    gate_biases,
    device const uint8_t* up_weights,
    device const half*    up_scales,
    device const half*    up_biases,
    device const half*    input,
    device half*          output,
    constant uint&        actualBatch,
    uint2 tgid,
    uint simd_lane,
    uint simd_group
) {
    constexpr uint FIXED_ROWS = 6144;
    constexpr uint FIXED_COLS = 2048;
    constexpr uint FIXED_GROUP_SIZE = 64;
    constexpr uint ROWS_PER_SG_LOCAL = 2;
    constexpr uint ROWS_PER_TG_LOCAL = 4;
    constexpr uint BATCH_TILE = 2;
    constexpr uint HALF_COLS = FIXED_COLS / 2;
    constexpr uint EIGHTH_COLS = HALF_COLS / 8;
    constexpr uint U16_PER_ROW = HALF_COLS / 2;
    constexpr uint NUM_COL_GROUPS = FIXED_COLS / FIXED_GROUP_SIZE;
    uint baseRow = tgid.x * ROWS_PER_TG_LOCAL;
    uint sgBaseRow = baseRow + simd_group * ROWS_PER_SG_LOCAL;
    uint batchBase = tgid.y * BATCH_TILE;
    bool fullTile = batchBase + BATCH_TILE <= actualBatch;

    float gateAcc[BATCH_TILE][ROWS_PER_SG_LOCAL];
    float upAcc[BATCH_TILE][ROWS_PER_SG_LOCAL];
    for (uint b = 0; b < BATCH_TILE; b++) {
        for (uint r = 0; r < ROWS_PER_SG_LOCAL; r++) {
            gateAcc[b][r] = 0.0f;
            upAcc[b][r] = 0.0f;
        }
    }

    device const uint16_t* gr0 = (device const uint16_t*)(gate_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* gr1 = gr0 + U16_PER_ROW;
    device const uint16_t* ur0 = (device const uint16_t*)(up_weights + sgBaseRow * HALF_COLS);
    device const uint16_t* ur1 = ur0 + U16_PER_ROW;

    device const half* gs0 = gate_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* gs1 = gs0 + NUM_COL_GROUPS;
    device const half* gb0 = gate_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* gb1 = gb0 + NUM_COL_GROUPS;
    device const half* us0 = up_scales + sgBaseRow * NUM_COL_GROUPS;
    device const half* us1 = us0 + NUM_COL_GROUPS;
    device const half* ub0 = up_biases + sgBaseRow * NUM_COL_GROUPS;
    device const half* ub1 = ub0 + NUM_COL_GROUPS;

    for (uint j = simd_lane; j < EIGHTH_COLS; j += 32) {
        uint col = j * 16;
        uint gidx = col / FIXED_GROUP_SIZE;

        device const uint16_t* gw0 = gr0 + j * 4;
        device const uint16_t* gw1 = gr1 + j * 4;
        device const uint16_t* uw0 = ur0 + j * 4;
        device const uint16_t* uw1 = ur1 + j * 4;

        for (uint b = 0; b < BATCH_TILE; b++) {
            if (!fullTile && batchBase + b >= actualBatch) continue;
            device const half* xp = input + (batchBase + b) * FIXED_COLS + col;
            float x0=float(xp[0]),x1=float(xp[1]),x2=float(xp[2]),x3=float(xp[3]);
            float x4=float(xp[4]),x5=float(xp[5]),x6=float(xp[6]),x7=float(xp[7]);
            float x8=float(xp[8]),x9=float(xp[9]),x10=float(xp[10]),x11=float(xp[11]);
            float x12=float(xp[12]),x13=float(xp[13]),x14=float(xp[14]),x15=float(xp[15]);
            float xsum = x0+x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11+x12+x13+x14+x15;
            x1*=(1.f/16.f); x2*=(1.f/256.f); x3*=(1.f/4096.f);
            x5*=(1.f/16.f); x6*=(1.f/256.f); x7*=(1.f/4096.f);
            x9*=(1.f/16.f); x10*=(1.f/256.f); x11*=(1.f/4096.f);
            x13*=(1.f/16.f); x14*=(1.f/256.f); x15*=(1.f/4096.f);

            gateAcc[b][0] += float(gs0[gidx]) * AGENT_AFFINE_DOT16(gw0) + float(gb0[gidx]) * xsum;
            upAcc[b][0] += float(us0[gidx]) * AGENT_AFFINE_DOT16(uw0) + float(ub0[gidx]) * xsum;
            gateAcc[b][1] += float(gs1[gidx]) * AGENT_AFFINE_DOT16(gw1) + float(gb1[gidx]) * xsum;
            upAcc[b][1] += float(us1[gidx]) * AGENT_AFFINE_DOT16(uw1) + float(ub1[gidx]) * xsum;
        }
    }

    for (uint b = 0; b < BATCH_TILE; b++) {
        gateAcc[b][0] = simd_sum(gateAcc[b][0]);
        gateAcc[b][1] = simd_sum(gateAcc[b][1]);
        upAcc[b][0] = simd_sum(upAcc[b][0]);
        upAcc[b][1] = simd_sum(upAcc[b][1]);
    }

    if (simd_lane == 0) {
        for (uint b = 0; b < BATCH_TILE; b++) {
            if (!fullTile && batchBase + b >= actualBatch) continue;
            device half* out = output + (batchBase + b) * FIXED_ROWS + sgBaseRow;
            out[0] = half(gateAcc[b][0] / (1.0f + exp(-gateAcc[b][0])) * upAcc[b][0]);
            out[1] = half(gateAcc[b][1] / (1.0f + exp(-gateAcc[b][1])) * upAcc[b][1]);
        }
    }
}

kernel void fused_affine_gate_up_swiglu_c2048_r6144_g64(
    device const uint8_t* gate_weights [[buffer(0)]],
    device const half*    gate_scales  [[buffer(1)]],
    device const half*    gate_biases  [[buffer(2)]],
    device const uint8_t* up_weights   [[buffer(3)]],
    device const half*    up_scales    [[buffer(4)]],
    device const half*    up_biases    [[buffer(5)]],
    device const half*    input        [[buffer(6)]],
    device half*          output       [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half4 scaleBiasCache[4 * (2048 / 64)];
    fused_affine_gate_up_swiglu_fixed_rows4_sbcache<6144, 2048, 64>(
        gate_weights, gate_scales, gate_biases,
        up_weights, up_scales, up_biases,
        input, output, scaleBiasCache,
        tgid, simd_lane, simd_group
    );
}

kernel void fused_affine_gate_up_swiglu_c2048_r6144_g64_batched(
    device const uint8_t* gate_weights [[buffer(0)]],
    device const half*    gate_scales  [[buffer(1)]],
    device const half*    gate_biases  [[buffer(2)]],
    device const uint8_t* up_weights   [[buffer(3)]],
    device const half*    up_scales    [[buffer(4)]],
    device const half*    up_biases    [[buffer(5)]],
    device const half*    input        [[buffer(6)]],
    device half*          output       [[buffer(7)]],
    constant uint&        actualBatch  [[buffer(8)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    fused_affine_gate_up_swiglu_fixed_batched_qwen_2x2(
        gate_weights, gate_scales, gate_biases,
        up_weights, up_scales, up_biases,
        input, output,
        actualBatch,
        tgid, simd_lane, simd_group
    );
}

kernel void fused_affine_gate_up_swiglu_c2048_r6144_g64_batched_full(
    device const uint8_t* gate_weights [[buffer(0)]],
    device const half*    gate_scales  [[buffer(1)]],
    device const half*    gate_biases  [[buffer(2)]],
    device const uint8_t* up_weights   [[buffer(3)]],
    device const half*    up_scales    [[buffer(4)]],
    device const half*    up_biases    [[buffer(5)]],
    device const half*    input        [[buffer(6)]],
    device half*          output       [[buffer(7)]],
    constant uint&        actualBatch  [[buffer(8)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint tid         [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half Xs[16 * 40];
    threadgroup float Wg[32 * 40];
    threadgroup float Wu[32 * 40];
    agent_fused_affine_gate_up_qmm_fixed_batched_full<6144, 2048, 64, 16>(
        gate_weights, gate_scales, gate_biases,
        up_weights, up_scales, up_biases,
        input, output,
        actualBatch,
        Xs, Wg, Wu,
        tgid, tid, simd_lane, simd_group
    );
}

kernel void fused_affine_matvec_add_c2048_r2048_g64(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    device const half*    residual [[buffer(5)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    fused_affine_matvec_add_fixed_rows4<2048, 2048, 64>(
        weights, scales, biases, input, output, residual,
        tgid, simd_lane, simd_group
    );
}

kernel void fused_affine_matvec_add_c6144_r2048_g64(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    device const half*    residual [[buffer(5)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    fused_affine_matvec_add_fixed<2048, 6144, 64>(
        weights, scales, biases, input, output, residual,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c2048_r2048_g64(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed_rows4<2048, 2048, 64>(
        weights, scales, biases, input, output,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c2048_r2048_g64_batched(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed_batched<2048, 2048, 64, 8>(
        weights, scales, biases, input, output,
        actualBatch,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c2048_r2048_g64_batched_full(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint tid         [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half Xs[16 * 40];
    threadgroup float Ws[32 * 40];
    affine_matvec_fixed_batched_full<2048, 2048, 64, 16>(
        weights, scales, biases, input, output,
        actualBatch,
        Xs, Ws,
        tgid, tid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c2048_r6144_g64(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed<6144, 2048, 64>(
        weights, scales, biases, input, output,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c6144_r2048_g64(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed<2048, 6144, 64>(
        weights, scales, biases, input, output,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c2048_r6144_g64_batched(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed_batched<6144, 2048, 64, 8>(
        weights, scales, biases, input, output,
        actualBatch,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c2048_r6144_g64_batched_full(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint tid         [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half Xs[16 * 40];
    threadgroup float Ws[32 * 40];
    affine_matvec_fixed_batched_full<6144, 2048, 64, 16>(
        weights, scales, biases, input, output,
        actualBatch,
        Xs, Ws,
        tgid, tid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c2048_r4096_g64(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed<4096, 2048, 64>(
        weights, scales, biases, input, output,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c2048_r4096_g64_batched(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed_batched<4096, 2048, 64, 8>(
        weights, scales, biases, input, output,
        actualBatch,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c2048_r4096_g64_batched_full(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint tid         [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half Xs[16 * 40];
    threadgroup float Ws[32 * 40];
    affine_matvec_fixed_batched_full<4096, 2048, 64, 16>(
        weights, scales, biases, input, output,
        actualBatch,
        Xs, Ws,
        tgid, tid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c2048_r512_g64(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed<512, 2048, 64>(
        weights, scales, biases, input, output,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c2048_r248320_g64(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed<248320, 2048, 64>(
        weights, scales, biases, input, output,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c2048_r512_g64_batched(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed_batched<512, 2048, 64, 8>(
        weights, scales, biases, input, output,
        actualBatch,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c2048_r512_g64_batched_full(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint tid         [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half Xs[16 * 40];
    threadgroup float Ws[32 * 40];
    affine_matvec_fixed_batched_full<512, 2048, 64, 16>(
        weights, scales, biases, input, output,
        actualBatch,
        Xs, Ws,
        tgid, tid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c6144_r2048_g64_batched(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed_batched<2048, 6144, 64, 8>(
        weights, scales, biases, input, output,
        actualBatch,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c6144_r2048_g64_batched_full(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint tid         [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half Xs[16 * 40];
    threadgroup float Ws[32 * 40];
    affine_matvec_fixed_batched_full<2048, 6144, 64, 16>(
        weights, scales, biases, input, output,
        actualBatch,
        Xs, Ws,
        tgid, tid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c1536_r2048_g128(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed<2048, 1536, 128>(
        weights, scales, biases, input, output,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c1536_r4096_g128(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed<4096, 1536, 128>(
        weights, scales, biases, input, output,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c1536_r256_g128(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed<256, 1536, 128>(
        weights, scales, biases, input, output,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c1536_r512_g128(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed<512, 1536, 128>(
        weights, scales, biases, input, output,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c1536_r6144_g128(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed<6144, 1536, 128>(
        weights, scales, biases, input, output,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c1536_r12288_g128(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed<12288, 1536, 128>(
        weights, scales, biases, input, output,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c2048_r1536_g128(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed<1536, 2048, 128>(
        weights, scales, biases, input, output,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c4096_r1536_g128(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed<1536, 4096, 128>(
        weights, scales, biases, input, output,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c6144_r1536_g128(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed<1536, 6144, 128>(
        weights, scales, biases, input, output,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c12288_r1536_g128(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed<1536, 12288, 128>(
        weights, scales, biases, input, output,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c1536_r2048_g128_batched(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed_batched<2048, 1536, 128, 8>(
        weights, scales, biases, input, output,
        actualBatch,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c1536_r4096_g128_batched(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed_batched<4096, 1536, 128, 8>(
        weights, scales, biases, input, output,
        actualBatch,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c1536_r6144_g128_batched(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed_batched<6144, 1536, 128, 8>(
        weights, scales, biases, input, output,
        actualBatch,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c1536_r12288_g128_batched(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed_batched<12288, 1536, 128, 8>(
        weights, scales, biases, input, output,
        actualBatch,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c2048_r1536_g128_batched(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed_batched<1536, 2048, 128, 8>(
        weights, scales, biases, input, output,
        actualBatch,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c4096_r1536_g128_batched(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed_batched<1536, 4096, 128, 8>(
        weights, scales, biases, input, output,
        actualBatch,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c6144_r1536_g128_batched(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed_batched<1536, 6144, 128, 8>(
        weights, scales, biases, input, output,
        actualBatch,
        tgid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c12288_r1536_g128_batched(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    affine_matvec_fixed_batched<1536, 12288, 128, 8>(
        weights, scales, biases, input, output,
        actualBatch,
        tgid, simd_lane, simd_group
    );
}

#define AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          output   [[buffer(4)]], \
    constant uint&        actualBatch [[buffer(5)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    affine_matvec_fixed_batched<ROWS, COLS, GROUP_SIZE, 8>( \
        weights, scales, biases, input, output, \
        actualBatch, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP_TILE(NAME, ROWS, COLS, GROUP_SIZE, BATCH_TILE) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          output   [[buffer(4)]], \
    constant uint&        actualBatch [[buffer(5)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    affine_matvec_fixed_batched<ROWS, COLS, GROUP_SIZE, BATCH_TILE>( \
        weights, scales, biases, input, output, \
        actualBatch, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_SG4_GROUP_TILE(NAME, ROWS, COLS, GROUP_SIZE, BATCH_TILE) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          output   [[buffer(4)]], \
    constant uint&        actualBatch [[buffer(5)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    affine_matvec_fixed_batched_sg4<ROWS, COLS, GROUP_SIZE, BATCH_TILE>( \
        weights, scales, biases, input, output, \
        actualBatch, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_AFFINE_MATVEC_FIXED_EXT_B5(NAME, ROWS, COLS, GROUP_SIZE, BATCH_TILE) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          output   [[buffer(4)]], \
    constant uint&        actualBatch [[buffer(5)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint tiisg       [[thread_index_in_simdgroup]], \
    uint sgitg       [[simdgroup_index_in_threadgroup]] \
) { \
    affine_matvec_fixed_ext_b5<ROWS, COLS, GROUP_SIZE, BATCH_TILE>( \
        weights, scales, biases, input, output, \
        actualBatch, \
        tgid, tiisg, sgitg \
    ); \
}

#define AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_BATCHED_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const float*   scalePtr   [[buffer(0)]], \
    device const half*    normInput  [[buffer(1)]], \
    device const half*    normWeight [[buffer(2)]], \
    device half*          normOutput [[buffer(3)]], \
    device const uint8_t* weights    [[buffer(4)]], \
    device const half*    scales     [[buffer(5)]], \
    device const half*    biases     [[buffer(6)]], \
    device half*          output     [[buffer(7)]], \
    constant uint&        actualBatch [[buffer(8)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    norm_scale_affine_matvec_fixed_batched<ROWS, COLS, GROUP_SIZE, 8>( \
        scalePtr, normInput, normWeight, normOutput, \
        weights, scales, biases, output, \
        actualBatch, \
        tgid, simd_lane, simd_group \
    ); \
}

AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP(affine_matvec_c2560_r2048_g128_batched, 2048, 2560, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP_TILE(affine_matvec_c2560_r2048_g128_batched_tile3, 2048, 2560, 128, 3)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_SG4_GROUP_TILE(affine_matvec_c2560_r2048_g128_batched_sg4_bt5, 2048, 2560, 128, 5)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP(affine_matvec_c2560_r4096_g128_batched, 4096, 2560, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP_TILE(affine_matvec_c2560_r4096_g128_batched_tile3, 4096, 2560, 128, 3)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP(affine_matvec_c2560_r512_g128_batched, 512, 2560, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP(affine_matvec_c2560_r1024_g128_batched, 1024, 2560, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP(affine_matvec_c2560_r10240_g128_batched, 10240, 2560, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_SG4_GROUP_TILE(affine_matvec_c2560_r10240_g128_batched_sg4_bt5, 10240, 2560, 128, 5)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_EXT_B5(affine_matvec_c2560_r10240_g128_batched_ext_b5, 10240, 2560, 128, 5)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_EXT_B5(affine_matvec_c10240_r2560_g128_batched_ext_b5, 2560, 10240, 128, 5)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_EXT_B5(affine_matvec_c2560_r2048_g128_batched_ext_b5, 2048, 2560, 128, 5)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_EXT_B5(affine_matvec_c2048_r2560_g128_batched_ext_b5, 2560, 2048, 128, 5)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_EXT_B5(affine_matvec_c2560_r10240_g128_batched_ext_b4, 10240, 2560, 128, 4)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_EXT_B5(affine_matvec_c10240_r2560_g128_batched_ext_b4, 2560, 10240, 128, 4)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_EXT_B5(affine_matvec_c2560_r2048_g128_batched_ext_b4, 2048, 2560, 128, 4)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_EXT_B5(affine_matvec_c2048_r2560_g128_batched_ext_b4, 2560, 2048, 128, 4)
AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_BATCHED_GROUP(norm_scale_affine_matvec_c2560_r10240_g128_batched, 10240, 2560, 128)
AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_BATCHED_GROUP(norm_scale_affine_matvec_c2560_r2048_g128_batched, 2048, 2560, 128)
AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_BATCHED_GROUP(norm_scale_affine_matvec_c2560_r4096_g128_batched, 4096, 2560, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP(affine_matvec_c2560_r10752_g128_batched, 10752, 2560, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP(affine_matvec_c2048_r2560_g128_batched, 2560, 2048, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP_TILE(affine_matvec_c2048_r2560_g128_batched_tile3, 2560, 2048, 128, 3)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_SG4_GROUP_TILE(affine_matvec_c2048_r2560_g128_batched_sg4_bt5, 2560, 2048, 128, 5)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP(affine_matvec_c4096_r2560_g128_batched, 2560, 4096, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP_TILE(affine_matvec_c4096_r2560_g128_batched_tile3, 2560, 4096, 128, 3)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP(affine_matvec_c10240_r2560_g128_batched, 2560, 10240, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP_TILE(affine_matvec_c10240_r2560_g128_batched_tile4, 2560, 10240, 128, 4)
// BT=3 variant for K=3 stochastic verify (3 active rows after row-0 priming).
// Codex single-seed bench (prior session) measured down 7.37 -> 8.08 ms — but
// single-seed spec bench has variance 1.5-2.9 across seeds 42-47, so retrying
// under multi-seed methodology established in Phase G.
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP_TILE(affine_matvec_c10240_r2560_g128_batched_tile3, 2560, 10240, 128, 3)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_SG4_GROUP_TILE(affine_matvec_c10240_r2560_g128_batched_sg4_bt5, 2560, 10240, 128, 5)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP(affine_matvec_c2560_r256_g128_batched, 256, 2560, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP(affine_matvec_c256_r2560_g128_batched, 2560, 256, 128)
// LM-head full-logits batched. Verify's prefillAllLogits previously
// fell through to generic affine_matvec (4 separate per-row dispatches
// reading the 33 MB W tensor 4 times = ~4 ms verify). Batching to one
// dispatch reads W once and writes K rows of logits in a single pass.
AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP(affine_matvec_c2560_r262144_g128_batched, 262144, 2560, 128)

#undef AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP
#undef AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_GROUP_TILE
#undef AGENT_DECLARE_AFFINE_MATVEC_FIXED_BATCHED_SG4_GROUP_TILE
#undef AGENT_DECLARE_AFFINE_MATVEC_FIXED_EXT_B5
#undef AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_BATCHED_GROUP

kernel void affine_matvec_c1536_r2048_g128_batched_full(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint tid         [[thread_index_in_threadgroup]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half Xs[16 * 40];
    threadgroup float Ws[32 * 40];
    affine_matvec_fixed_batched_full<2048, 1536, 128, 16>(
        weights, scales, biases, input, output,
        actualBatch,
        Xs, Ws,
        tgid, tid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c1536_r4096_g128_batched_full(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint tid         [[thread_index_in_threadgroup]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half Xs[16 * 40];
    threadgroup float Ws[32 * 40];
    affine_matvec_fixed_batched_full<4096, 1536, 128, 16>(
        weights, scales, biases, input, output,
        actualBatch,
        Xs, Ws,
        tgid, tid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c1536_r6144_g128_batched_full(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint tid         [[thread_index_in_threadgroup]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half Xs[16 * 40];
    threadgroup float Ws[32 * 40];
    affine_matvec_fixed_batched_full<6144, 1536, 128, 16>(
        weights, scales, biases, input, output,
        actualBatch,
        Xs, Ws,
        tgid, tid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c1536_r12288_g128_batched_full(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint tid         [[thread_index_in_threadgroup]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half Xs[16 * 40];
    threadgroup float Ws[32 * 40];
    affine_matvec_fixed_batched_full<12288, 1536, 128, 16>(
        weights, scales, biases, input, output,
        actualBatch,
        Xs, Ws,
        tgid, tid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c2048_r1536_g128_batched_full(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint tid         [[thread_index_in_threadgroup]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half Xs[16 * 40];
    threadgroup float Ws[32 * 40];
    affine_matvec_fixed_batched_full<1536, 2048, 128, 16>(
        weights, scales, biases, input, output,
        actualBatch,
        Xs, Ws,
        tgid, tid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c4096_r1536_g128_batched_full(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint tid         [[thread_index_in_threadgroup]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half Xs[16 * 40];
    threadgroup float Ws[32 * 40];
    affine_matvec_fixed_batched_full<1536, 4096, 128, 16>(
        weights, scales, biases, input, output,
        actualBatch,
        Xs, Ws,
        tgid, tid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c6144_r1536_g128_batched_full(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint tid         [[thread_index_in_threadgroup]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half Xs[16 * 40];
    threadgroup float Ws[32 * 40];
    affine_matvec_fixed_batched_full<1536, 6144, 128, 16>(
        weights, scales, biases, input, output,
        actualBatch,
        Xs, Ws,
        tgid, tid, simd_lane, simd_group
    );
}

kernel void affine_matvec_c12288_r1536_g128_batched_full(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    constant uint&        actualBatch [[buffer(5)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint tid         [[thread_index_in_threadgroup]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half Xs[16 * 40];
    threadgroup float Ws[32 * 40];
    affine_matvec_fixed_batched_full<1536, 12288, 128, 16>(
        weights, scales, biases, input, output,
        actualBatch,
        Xs, Ws,
        tgid, tid, simd_lane, simd_group
    );
}

kernel void fused_dual_affine_matvec_c2048_r16_g64_batched(
    device const uint8_t* w1_weights [[buffer(0)]],
    device const half*    w1_scales  [[buffer(1)]],
    device const half*    w1_biases  [[buffer(2)]],
    device const uint8_t* w2_weights [[buffer(3)]],
    device const half*    w2_scales  [[buffer(4)]],
    device const half*    w2_biases  [[buffer(5)]],
    device const half*    input      [[buffer(6)]],
    device half*          output1    [[buffer(7)]],
    device half*          output2    [[buffer(8)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    fused_dual_affine_matvec_fixed_batched<16, 2048, 64>(
        w1_weights, w1_scales, w1_biases,
        w2_weights, w2_scales, w2_biases,
        input, output1, output2,
        tgid, simd_lane, simd_group
    );
}

kernel void fused_dual_affine_matvec_c1536_r256_g128_batched(
    device const uint8_t* w1_weights [[buffer(0)]],
    device const half*    w1_scales  [[buffer(1)]],
    device const half*    w1_biases  [[buffer(2)]],
    device const uint8_t* w2_weights [[buffer(3)]],
    device const half*    w2_scales  [[buffer(4)]],
    device const half*    w2_biases  [[buffer(5)]],
    device const half*    input      [[buffer(6)]],
    device half*          output1    [[buffer(7)]],
    device half*          output2    [[buffer(8)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    fused_dual_affine_matvec_fixed_batched<256, 1536, 128>(
        w1_weights, w1_scales, w1_biases,
        w2_weights, w2_scales, w2_biases,
        input, output1, output2,
        tgid, simd_lane, simd_group
    );
}

kernel void fused_dual_affine_matvec_c1536_r512_g128_batched(
    device const uint8_t* w1_weights [[buffer(0)]],
    device const half*    w1_scales  [[buffer(1)]],
    device const half*    w1_biases  [[buffer(2)]],
    device const uint8_t* w2_weights [[buffer(3)]],
    device const half*    w2_scales  [[buffer(4)]],
    device const half*    w2_biases  [[buffer(5)]],
    device const half*    input      [[buffer(6)]],
    device half*          output1    [[buffer(7)]],
    device half*          output2    [[buffer(8)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    fused_dual_affine_matvec_fixed_batched<512, 1536, 128>(
        w1_weights, w1_scales, w1_biases,
        w2_weights, w2_scales, w2_biases,
        input, output1, output2,
        tgid, simd_lane, simd_group
    );
}

#define AGENT_DECLARE_FUSED_DUAL_AFFINE_MATVEC_FIXED_BATCHED_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* w1_weights [[buffer(0)]], \
    device const half*    w1_scales  [[buffer(1)]], \
    device const half*    w1_biases  [[buffer(2)]], \
    device const uint8_t* w2_weights [[buffer(3)]], \
    device const half*    w2_scales  [[buffer(4)]], \
    device const half*    w2_biases  [[buffer(5)]], \
    device const half*    input      [[buffer(6)]], \
    device half*          output1    [[buffer(7)]], \
    device half*          output2    [[buffer(8)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    fused_dual_affine_matvec_fixed_batched<ROWS, COLS, GROUP_SIZE>( \
        w1_weights, w1_scales, w1_biases, \
        w2_weights, w2_scales, w2_biases, \
        input, output1, output2, \
        tgid, simd_lane, simd_group \
    ); \
}

AGENT_DECLARE_FUSED_DUAL_AFFINE_MATVEC_FIXED_BATCHED_GROUP(fused_dual_affine_matvec_c2560_r512_g128_batched, 512, 2560, 128)
AGENT_DECLARE_FUSED_DUAL_AFFINE_MATVEC_FIXED_BATCHED_GROUP(fused_dual_affine_matvec_c2560_r1024_g128_batched, 1024, 2560, 128)

#undef AGENT_DECLARE_FUSED_DUAL_AFFINE_MATVEC_FIXED_BATCHED_GROUP

#define AGENT_DECLARE_NORM_SCALE_FUSED_DUAL_AFFINE_MATVEC_FIXED_BATCHED_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const float*   scalePtr   [[buffer(0)]], \
    device const half*    normInput  [[buffer(1)]], \
    device const half*    normWeight [[buffer(2)]], \
    device const uint8_t* w1_weights [[buffer(3)]], \
    device const half*    w1_scales  [[buffer(4)]], \
    device const half*    w1_biases  [[buffer(5)]], \
    device const uint8_t* w2_weights [[buffer(6)]], \
    device const half*    w2_scales  [[buffer(7)]], \
    device const half*    w2_biases  [[buffer(8)]], \
    device half*          output1    [[buffer(9)]], \
    device half*          output2    [[buffer(10)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    norm_scale_fused_dual_affine_matvec_fixed_batched<ROWS, COLS, GROUP_SIZE>( \
        scalePtr, normInput, normWeight, \
        w1_weights, w1_scales, w1_biases, \
        w2_weights, w2_scales, w2_biases, \
        output1, output2, \
        tgid, simd_lane, simd_group \
    ); \
}

AGENT_DECLARE_NORM_SCALE_FUSED_DUAL_AFFINE_MATVEC_FIXED_BATCHED_GROUP(norm_scale_fused_dual_affine_matvec_c2560_r512_g128_batched, 512, 2560, 128)
AGENT_DECLARE_NORM_SCALE_FUSED_DUAL_AFFINE_MATVEC_FIXED_BATCHED_GROUP(norm_scale_fused_dual_affine_matvec_c2560_r1024_g128_batched, 1024, 2560, 128)

#undef AGENT_DECLARE_NORM_SCALE_FUSED_DUAL_AFFINE_MATVEC_FIXED_BATCHED_GROUP

kernel void fused_dual_affine_matvec_c2048_r16_g64(
    device const uint8_t* w1_weights [[buffer(0)]],
    device const half*    w1_scales  [[buffer(1)]],
    device const half*    w1_biases  [[buffer(2)]],
    device const uint8_t* w2_weights [[buffer(3)]],
    device const half*    w2_scales  [[buffer(4)]],
    device const half*    w2_biases  [[buffer(5)]],
    device const half*    input      [[buffer(6)]],
    device half*          output1    [[buffer(7)]],
    device half*          output2    [[buffer(8)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    fused_dual_affine_matvec_fixed<16, 2048, 64>(
        w1_weights, w1_scales, w1_biases,
        w2_weights, w2_scales, w2_biases,
        input, output1, output2,
        tgid, simd_lane, simd_group
    );
}

kernel void fused_dual_affine_matvec_c2048_r512_g64_batched(
    device const uint8_t* w1_weights [[buffer(0)]],
    device const half*    w1_scales  [[buffer(1)]],
    device const half*    w1_biases  [[buffer(2)]],
    device const uint8_t* w2_weights [[buffer(3)]],
    device const half*    w2_scales  [[buffer(4)]],
    device const half*    w2_biases  [[buffer(5)]],
    device const half*    input      [[buffer(6)]],
    device half*          output1    [[buffer(7)]],
    device half*          output2    [[buffer(8)]],
    uint2 tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    fused_dual_affine_matvec_fixed_batched<512, 2048, 64>(
        w1_weights, w1_scales, w1_biases,
        w2_weights, w2_scales, w2_biases,
        input, output1, output2,
        tgid, simd_lane, simd_group
    );
}

#define AGENT_DECLARE_AFFINE_MATVEC_FIXED(NAME, ROWS, COLS) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          output   [[buffer(4)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    affine_matvec_fixed<ROWS, COLS, 64>( \
        weights, scales, biases, input, output, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4(NAME, ROWS, COLS) \
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(NAME, ROWS, COLS, 64)

#define AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          output   [[buffer(4)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    affine_matvec_fixed_rows4<ROWS, COLS, GROUP_SIZE>( \
        weights, scales, biases, input, output, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_SG1_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          output   [[buffer(4)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    affine_matvec_fixed_rows4_sg1<ROWS, COLS, GROUP_SIZE>( \
        weights, scales, biases, input, output, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS8_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          output   [[buffer(4)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    affine_matvec_fixed_rows8<ROWS, COLS, GROUP_SIZE>( \
        weights, scales, biases, input, output, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_AFFINE_MATVEC_ARGMAX_FIXED_BATCHED_GROUP(NAME, ROWS, COLS, GROUP_SIZE, BATCH_TILE) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device uint2*         partialKeys [[buffer(4)]], \
    constant uint&        actualBatch [[buffer(5)]], \
    constant float&       logitCap [[buffer(6)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup uint2 localKeys[BATCH_TILE * 2]; \
    affine_matvec_argmax_fixed_batched<ROWS, COLS, GROUP_SIZE, BATCH_TILE>( \
        weights, scales, biases, input, partialKeys, actualBatch, logitCap, localKeys, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_LM_HEAD_ARGMAX_REDUCE(NAME, ROWS) \
kernel void NAME( \
    device const uint2* partialKeys [[buffer(0)]], \
    device int*         output      [[buffer(1)]], \
    constant uint&      actualBatch [[buffer(2)]], \
    uint batch [[threadgroup_position_in_grid]], \
    uint tid   [[thread_index_in_threadgroup]] \
) { \
    threadgroup uint2 scratch[256]; \
    lm_head_argmax_reduce_fixed<ROWS>(partialKeys, output, actualBatch, batch, tid, scratch); \
}

#define AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          output   [[buffer(4)]], \
    device const half*    residual [[buffer(5)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    fused_affine_matvec_add_fixed<ROWS, COLS, GROUP_SIZE>( \
        weights, scales, biases, input, output, residual, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          output   [[buffer(4)]], \
    device const half*    residual [[buffer(5)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    fused_affine_matvec_add_fixed_rows4<ROWS, COLS, GROUP_SIZE>( \
        weights, scales, biases, input, output, residual, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_TGINPUT_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          output   [[buffer(4)]], \
    device const half*    residual [[buffer(5)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half inputScratch[COLS]; \
    fused_affine_matvec_add_fixed_rows4_tginput<ROWS, COLS, GROUP_SIZE>( \
        weights, scales, biases, input, output, residual, inputScratch, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_SG1_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          output   [[buffer(4)]], \
    device const half*    residual [[buffer(5)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    fused_affine_matvec_add_fixed_rows4_sg1<ROWS, COLS, GROUP_SIZE>( \
        weights, scales, biases, input, output, residual, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_SBCACHE_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          output   [[buffer(4)]], \
    device const half*    residual [[buffer(5)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half scaleBiasCache[2 * 4 * (COLS / GROUP_SIZE)]; \
    fused_affine_matvec_add_fixed_rows4_sbcache<ROWS, COLS, GROUP_SIZE>( \
        weights, scales, biases, input, output, residual, scaleBiasCache, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS16_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          output   [[buffer(4)]], \
    device const half*    residual [[buffer(5)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    fused_affine_matvec_add_fixed_rows4sg_tiled<ROWS, COLS, GROUP_SIZE, 16>( \
        weights, scales, biases, input, output, residual, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS32_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          output   [[buffer(4)]], \
    device const half*    residual [[buffer(5)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    fused_affine_matvec_add_fixed_rows4sg_tiled<ROWS, COLS, GROUP_SIZE, 32>( \
        weights, scales, biases, input, output, residual, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL(NAME, ROWS, COLS) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          output   [[buffer(4)]], \
    constant uint&        actualBatch [[buffer(5)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint tid         [[thread_index_in_threadgroup]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half Xs[16 * (32 + 8)]; \
    threadgroup float Ws[32 * (32 + 8)]; \
    agent_affine_qmm_fixed_batched_full<ROWS, COLS, 64, 16, false>( \
        weights, scales, biases, input, output, \
        actualBatch, \
        Xs, Ws, \
        tgid, tid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL_PADDED_REFERENCE(NAME, ROWS, COLS) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          output   [[buffer(4)]], \
    constant uint&        actualBatch [[buffer(5)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint tid         [[thread_index_in_threadgroup]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half Xs[16 * (32 + 8)]; \
    threadgroup float Ws[32 * (32 + 8)]; \
    agent_affine_qmm_fixed_batched_full<ROWS, COLS, 64, 16, true>( \
        weights, scales, biases, input, output, \
        actualBatch, \
        Xs, Ws, \
        tgid, tid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_FULL(NAME, ROWS, COLS) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          matvecOutput [[buffer(4)]], \
    device const half*    residual [[buffer(5)]], \
    device half*          output   [[buffer(6)]], \
    constant uint&        actualBatch [[buffer(7)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint tid         [[thread_index_in_threadgroup]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half Xs[16 * (32 + 8)]; \
    threadgroup float Ws[32 * (32 + 8)]; \
    const uint batchBase = tgid.y * 16; \
    const uint rowBase = tgid.x * 32; \
    const short2 coord = AgentMMAFrag88f::getCoord(ushort(simd_lane)); \
    const uint simdRowBase = (simd_group / 2) * 8; \
    const uint simdColBase = (simd_group % 2) * 16; \
    const bool valid = batchBase + simdRowBase + coord.y < actualBatch; \
    const uint outIndex0 = (simdRowBase + coord.y) * ROWS + coord.x + 0; \
    const uint outIndex1 = (simdRowBase + coord.y) * ROWS + coord.x + 1; \
    const uint outIndex2 = (simdRowBase + coord.y) * ROWS + coord.x + 8; \
    const uint outIndex3 = (simdRowBase + coord.y) * ROWS + coord.x + 9; \
    device half* matvecBase = matvecOutput + batchBase * ROWS + rowBase + simdColBase; \
    device half* outputBase = output + batchBase * ROWS + rowBase + simdColBase; \
    device const half* residualBase = residual + batchBase * ROWS + rowBase + simdColBase; \
    half r0 = half(0.0h); \
    half r1 = half(0.0h); \
    half r2 = half(0.0h); \
    half r3 = half(0.0h); \
    if (valid) { \
        r0 = residualBase[outIndex0]; \
        r1 = residualBase[outIndex1]; \
        r2 = residualBase[outIndex2]; \
        r3 = residualBase[outIndex3]; \
    } \
    agent_affine_qmm_fixed_batched_full<ROWS, COLS, 64, 16, false>( \
        weights, scales, biases, input, matvecOutput, \
        actualBatch, \
        Xs, Ws, \
        tgid, tid, simd_lane, simd_group \
    ); \
    if (valid) { \
        outputBase[outIndex0] = matvecBase[outIndex0] + r0; \
        outputBase[outIndex1] = matvecBase[outIndex1] + r1; \
        outputBase[outIndex2] = matvecBase[outIndex2] + r2; \
        outputBase[outIndex3] = matvecBase[outIndex3] + r3; \
    } \
}

#define AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL_SCALAR(NAME, ROWS, COLS) \
kernel void NAME( \
    device const uint8_t* weights  [[buffer(0)]], \
    device const half*    scales   [[buffer(1)]], \
    device const half*    biases   [[buffer(2)]], \
    device const half*    input    [[buffer(3)]], \
    device half*          output   [[buffer(4)]], \
    constant uint&        actualBatch [[buffer(5)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    affine_matvec_fixed_batched<ROWS, COLS, 64, 8>( \
        weights, scales, biases, input, output, \
        actualBatch, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const float*    scalePtr   [[buffer(0)]], \
    device const half*     normInput  [[buffer(1)]], \
    device const half*     normWeight [[buffer(2)]], \
    device half*           normOutput [[buffer(3)]], \
    device const uint8_t*  weights    [[buffer(4)]], \
    device const half*     scales     [[buffer(5)]], \
    device const half*     biases     [[buffer(6)]], \
    device half*           output     [[buffer(7)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    norm_scale_affine_matvec_fixed_rows4<ROWS, COLS, GROUP_SIZE>( \
        scalePtr, normInput, normWeight, normOutput, \
        weights, scales, biases, output, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_ROWS8_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const float*    scalePtr   [[buffer(0)]], \
    device const half*     normInput  [[buffer(1)]], \
    device const half*     normWeight [[buffer(2)]], \
    device half*           normOutput [[buffer(3)]], \
    device const uint8_t*  weights    [[buffer(4)]], \
    device const half*     scales     [[buffer(5)]], \
    device const half*     biases     [[buffer(6)]], \
    device half*           output     [[buffer(7)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    norm_scale_affine_matvec_fixed_rows8<ROWS, COLS, GROUP_SIZE>( \
        scalePtr, normInput, normWeight, normOutput, \
        weights, scales, biases, output, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_NORM_ADD_SCALE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const float*    scalePtr   [[buffer(0)]], \
    device const half*     normInput  [[buffer(1)]], \
    device const half*     normWeight [[buffer(2)]], \
    device const half*     residual   [[buffer(3)]], \
    device half*           normOutput [[buffer(4)]], \
    device const uint8_t*  weights    [[buffer(5)]], \
    device const half*     scales     [[buffer(6)]], \
    device const half*     biases     [[buffer(7)]], \
    device half*           output     [[buffer(8)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    norm_add_scale_affine_matvec_fixed_rows4<ROWS, COLS, GROUP_SIZE>( \
        scalePtr, normInput, normWeight, residual, normOutput, \
        weights, scales, biases, output, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED(NAME, ROWS, COLS) \
kernel void NAME( \
    device const uint8_t* gate_weights [[buffer(0)]], \
    device const half*    gate_scales  [[buffer(1)]], \
    device const half*    gate_biases  [[buffer(2)]], \
    device const uint8_t* up_weights   [[buffer(3)]], \
    device const half*    up_scales    [[buffer(4)]], \
    device const half*    up_biases    [[buffer(5)]], \
    device const half*    input        [[buffer(6)]], \
    device half*          output       [[buffer(7)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    fused_affine_gate_up_swiglu_fixed_rows4<ROWS, COLS, 64>( \
        gate_weights, gate_scales, gate_biases, \
        up_weights, up_scales, up_biases, \
        input, output, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_ROWS4_TGINPUT_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* gate_weights [[buffer(0)]], \
    device const half*    gate_scales  [[buffer(1)]], \
    device const half*    gate_biases  [[buffer(2)]], \
    device const uint8_t* up_weights   [[buffer(3)]], \
    device const half*    up_scales    [[buffer(4)]], \
    device const half*    up_biases    [[buffer(5)]], \
    device const half*    input        [[buffer(6)]], \
    device half*          output       [[buffer(7)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half inputScratch[COLS]; \
    fused_affine_gate_up_swiglu_fixed_rows4_tginput<ROWS, COLS, GROUP_SIZE>( \
        gate_weights, gate_scales, gate_biases, \
        up_weights, up_scales, up_biases, \
        input, output, inputScratch, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_ROWS4_EXP2_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* gate_weights [[buffer(0)]], \
    device const half*    gate_scales  [[buffer(1)]], \
    device const half*    gate_biases  [[buffer(2)]], \
    device const uint8_t* up_weights   [[buffer(3)]], \
    device const half*    up_scales    [[buffer(4)]], \
    device const half*    up_biases    [[buffer(5)]], \
    device const half*    input        [[buffer(6)]], \
    device half*          output       [[buffer(7)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    fused_affine_gate_up_swiglu_fixed_rows4_exp2<ROWS, COLS, GROUP_SIZE>( \
        gate_weights, gate_scales, gate_biases, \
        up_weights, up_scales, up_biases, \
        input, output, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_ROWS4_SBCACHE_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* gate_weights [[buffer(0)]], \
    device const half*    gate_scales  [[buffer(1)]], \
    device const half*    gate_biases  [[buffer(2)]], \
    device const uint8_t* up_weights   [[buffer(3)]], \
    device const half*    up_scales    [[buffer(4)]], \
    device const half*    up_biases    [[buffer(5)]], \
    device const half*    input        [[buffer(6)]], \
    device half*          output       [[buffer(7)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half4 scaleBiasCache[4 * (COLS / GROUP_SIZE)]; \
    fused_affine_gate_up_swiglu_fixed_rows4_sbcache<ROWS, COLS, GROUP_SIZE>( \
        gate_weights, gate_scales, gate_biases, \
        up_weights, up_scales, up_biases, \
        input, output, scaleBiasCache, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_ROWS16_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* gate_weights [[buffer(0)]], \
    device const half*    gate_scales  [[buffer(1)]], \
    device const half*    gate_biases  [[buffer(2)]], \
    device const uint8_t* up_weights   [[buffer(3)]], \
    device const half*    up_scales    [[buffer(4)]], \
    device const half*    up_biases    [[buffer(5)]], \
    device const half*    input        [[buffer(6)]], \
    device half*          output       [[buffer(7)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    fused_affine_gate_up_swiglu_fixed_rows4sg_tiled<ROWS, COLS, GROUP_SIZE, 16>( \
        gate_weights, gate_scales, gate_biases, \
        up_weights, up_scales, up_biases, \
        input, output, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_ROWS32_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* gate_weights [[buffer(0)]], \
    device const half*    gate_scales  [[buffer(1)]], \
    device const half*    gate_biases  [[buffer(2)]], \
    device const uint8_t* up_weights   [[buffer(3)]], \
    device const half*    up_scales    [[buffer(4)]], \
    device const half*    up_biases    [[buffer(5)]], \
    device const half*    input        [[buffer(6)]], \
    device half*          output       [[buffer(7)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    fused_affine_gate_up_swiglu_fixed_rows4sg_tiled<ROWS, COLS, GROUP_SIZE, 32>( \
        gate_weights, gate_scales, gate_biases, \
        up_weights, up_scales, up_biases, \
        input, output, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_GATE_UP_GEGLU_FIXED_ROWS4_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* gate_weights [[buffer(0)]], \
    device const half*    gate_scales  [[buffer(1)]], \
    device const half*    gate_biases  [[buffer(2)]], \
    device const uint8_t* up_weights   [[buffer(3)]], \
    device const half*    up_scales    [[buffer(4)]], \
    device const half*    up_biases    [[buffer(5)]], \
    device const half*    input        [[buffer(6)]], \
    device half*          output       [[buffer(7)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    fused_affine_gate_up_geglu_fixed_rows4<ROWS, COLS, GROUP_SIZE>( \
        gate_weights, gate_scales, gate_biases, \
        up_weights, up_scales, up_biases, \
        input, output, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_GATE_UP_GEGLU_FIXED_ROWS8_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* gate_weights [[buffer(0)]], \
    device const half*    gate_scales  [[buffer(1)]], \
    device const half*    gate_biases  [[buffer(2)]], \
    device const uint8_t* up_weights   [[buffer(3)]], \
    device const half*    up_scales    [[buffer(4)]], \
    device const half*    up_biases    [[buffer(5)]], \
    device const half*    input        [[buffer(6)]], \
    device half*          output       [[buffer(7)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    fused_affine_gate_up_geglu_fixed_rows8<ROWS, COLS, GROUP_SIZE>( \
        gate_weights, gate_scales, gate_biases, \
        up_weights, up_scales, up_biases, \
        input, output, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_GATE_UP_GEGLU_FIXED_BT_SG(NAME, ROWS, COLS, GROUP_SIZE, BT, SG_COUNT) \
kernel void NAME( \
    device const uint8_t* gate_weights [[buffer(0)]], \
    device const half*    gate_scales  [[buffer(1)]], \
    device const half*    gate_biases  [[buffer(2)]], \
    device const uint8_t* up_weights   [[buffer(3)]], \
    device const half*    up_scales    [[buffer(4)]], \
    device const half*    up_biases    [[buffer(5)]], \
    device const half*    input        [[buffer(6)]], \
    device half*          output       [[buffer(7)]], \
    constant uint&        actualBatch  [[buffer(8)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    fused_affine_gate_up_geglu_fixed_batched<ROWS, COLS, GROUP_SIZE, BT, SG_COUNT>( \
        gate_weights, gate_scales, gate_biases, \
        up_weights, up_scales, up_biases, \
        input, output, \
        actualBatch, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_GEGLU_FIXED_ROWS4_GROUP(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const float*    scalePtr     [[buffer(0)]], \
    device const half*     normInput    [[buffer(1)]], \
    device const half*     normWeight   [[buffer(2)]], \
    device const uint8_t*  gate_weights [[buffer(3)]], \
    device const half*     gate_scales  [[buffer(4)]], \
    device const half*     gate_biases  [[buffer(5)]], \
    device const uint8_t*  up_weights   [[buffer(6)]], \
    device const half*     up_scales    [[buffer(7)]], \
    device const half*     up_biases    [[buffer(8)]], \
    device half*           output       [[buffer(9)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    norm_scale_affine_gate_up_geglu_fixed_rows4<ROWS, COLS, GROUP_SIZE>( \
        scalePtr, normInput, normWeight, \
        gate_weights, gate_scales, gate_biases, \
        up_weights, up_scales, up_biases, \
        output, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_FULL(NAME, ROWS, COLS) \
kernel void NAME( \
    device const uint8_t* gate_weights [[buffer(0)]], \
    device const half*    gate_scales  [[buffer(1)]], \
    device const half*    gate_biases  [[buffer(2)]], \
    device const uint8_t* up_weights   [[buffer(3)]], \
    device const half*    up_scales    [[buffer(4)]], \
    device const half*    up_biases    [[buffer(5)]], \
    device const half*    input        [[buffer(6)]], \
    device half*          output       [[buffer(7)]], \
    constant uint&        actualBatch  [[buffer(8)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint tid         [[thread_index_in_threadgroup]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half Xs[16 * (32 + 8)]; \
    threadgroup float Wg[32 * (32 + 8)]; \
    threadgroup float Wu[32 * (32 + 8)]; \
    agent_fused_affine_gate_up_qmm_fixed_batched_full<ROWS, COLS, 64, 16, 0, false>( \
        gate_weights, gate_scales, gate_biases, \
        up_weights, up_scales, up_biases, \
        input, output, \
        actualBatch, \
        Xs, Wg, Wu, \
        tgid, tid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_FULL_PADDED_REFERENCE(NAME, ROWS, COLS) \
kernel void NAME( \
    device const uint8_t* gate_weights [[buffer(0)]], \
    device const half*    gate_scales  [[buffer(1)]], \
    device const half*    gate_biases  [[buffer(2)]], \
    device const uint8_t* up_weights   [[buffer(3)]], \
    device const half*    up_scales    [[buffer(4)]], \
    device const half*    up_biases    [[buffer(5)]], \
    device const half*    input        [[buffer(6)]], \
    device half*          output       [[buffer(7)]], \
    constant uint&        actualBatch  [[buffer(8)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint tid         [[thread_index_in_threadgroup]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half Xs[16 * (32 + 8)]; \
    threadgroup float Wg[32 * (32 + 8)]; \
    threadgroup float Wu[32 * (32 + 8)]; \
    agent_fused_affine_gate_up_qmm_fixed_batched_full<ROWS, COLS, 64, 16, 0, true>( \
        gate_weights, gate_scales, gate_biases, \
        up_weights, up_scales, up_biases, \
        input, output, \
        actualBatch, \
        Xs, Wg, Wu, \
        tgid, tid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_AFFINE_GATE_UP_GEGLU_FIXED_FULL(NAME, ROWS, COLS, GROUP_SIZE) \
kernel void NAME( \
    device const uint8_t* gate_weights [[buffer(0)]], \
    device const half*    gate_scales  [[buffer(1)]], \
    device const half*    gate_biases  [[buffer(2)]], \
    device const uint8_t* up_weights   [[buffer(3)]], \
    device const half*    up_scales    [[buffer(4)]], \
    device const half*    up_biases    [[buffer(5)]], \
    device const half*    input        [[buffer(6)]], \
    device half*          output       [[buffer(7)]], \
    constant uint&        actualBatch  [[buffer(8)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint tid         [[thread_index_in_threadgroup]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half Xs[16 * (32 + 8)]; \
    threadgroup float Wg[32 * (32 + 8)]; \
    threadgroup float Wu[32 * (32 + 8)]; \
    agent_fused_affine_gate_up_qmm_fixed_batched_full<ROWS, COLS, GROUP_SIZE, 16, 1>( \
        gate_weights, gate_scales, gate_biases, \
        up_weights, up_scales, up_biases, \
        input, output, \
        actualBatch, \
        Xs, Wg, Wu, \
        tgid, tid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_FULL(NAME, ROWS, COLS) \
kernel void NAME( \
    device const float*   scalePtr   [[buffer(0)]], \
    device const half*    normInput  [[buffer(1)]], \
    device const half*    normWeight [[buffer(2)]], \
    device half*          normOutput [[buffer(3)]], \
    device const uint8_t* weights    [[buffer(4)]], \
    device const half*    scales     [[buffer(5)]], \
    device const half*    biases     [[buffer(6)]], \
    device half*          output     [[buffer(7)]], \
    constant uint&        actualBatch [[buffer(8)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint tid         [[thread_index_in_threadgroup]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half Xs[16 * (32 + 8)]; \
    threadgroup float Ws[32 * (32 + 8)]; \
    agent_norm_scale_affine_qmm_fixed_batched_full<ROWS, COLS, 64, 16, false>( \
        scalePtr, normInput, normWeight, normOutput, \
        weights, scales, biases, output, actualBatch, \
        Xs, Ws, \
        tgid, tid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_FULL_PADDED_REFERENCE(NAME, ROWS, COLS) \
kernel void NAME( \
    device const float*   scalePtr   [[buffer(0)]], \
    device const half*    normInput  [[buffer(1)]], \
    device const half*    normWeight [[buffer(2)]], \
    device half*          normOutput [[buffer(3)]], \
    device const uint8_t* weights    [[buffer(4)]], \
    device const half*    scales     [[buffer(5)]], \
    device const half*    biases     [[buffer(6)]], \
    device half*          output     [[buffer(7)]], \
    constant uint&        actualBatch [[buffer(8)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint tid         [[thread_index_in_threadgroup]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half Xs[16 * (32 + 8)]; \
    threadgroup float Ws[32 * (32 + 8)]; \
    agent_norm_scale_affine_qmm_fixed_batched_full<ROWS, COLS, 64, 16, true>( \
        scalePtr, normInput, normWeight, normOutput, \
        weights, scales, biases, output, actualBatch, \
        Xs, Ws, \
        tgid, tid, simd_lane, simd_group \
    ); \
}

// ACTIVATION: 0 = SwiGLU (gate / (1+exp(-gate)) * up), 1 = GeGLU
// (tanh-based GELU on gate * up). See agent_qmm_gate_up_product.
#define AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_GLU_FIXED_FULL(NAME, ROWS, COLS, GROUP_SIZE, ACTIVATION) \
kernel void NAME( \
    device const float*   scalePtr     [[buffer(0)]], \
    device const half*    normInput    [[buffer(1)]], \
    device const half*    normWeight   [[buffer(2)]], \
    device half*          normOutput   [[buffer(3)]], \
    device const uint8_t* gate_weights [[buffer(4)]], \
    device const half*    gate_scales  [[buffer(5)]], \
    device const half*    gate_biases  [[buffer(6)]], \
    device const uint8_t* up_weights   [[buffer(7)]], \
    device const half*    up_scales    [[buffer(8)]], \
    device const half*    up_biases    [[buffer(9)]], \
    device half*          output       [[buffer(10)]], \
    constant uint&        actualBatch  [[buffer(11)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint tid         [[thread_index_in_threadgroup]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half Xs[16 * (32 + 8)]; \
    threadgroup float Wg[32 * (32 + 8)]; \
    threadgroup float Wu[32 * (32 + 8)]; \
    agent_norm_scale_fused_affine_gate_up_qmm_fixed_batched_full<ROWS, COLS, GROUP_SIZE, 16, ACTIVATION, (ACTIVATION != 0)>( \
        scalePtr, normInput, normWeight, normOutput, \
        gate_weights, gate_scales, gate_biases, \
        up_weights, up_scales, up_biases, output, actualBatch, \
        Xs, Wg, Wu, \
        tgid, tid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_SWIGLU_FIXED_FULL(NAME, ROWS, COLS) \
    AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_GLU_FIXED_FULL(NAME, ROWS, COLS, 64, 0)

#define AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_SWIGLU_FIXED_FULL_PADDED_REFERENCE(NAME, ROWS, COLS) \
kernel void NAME( \
    device const float*   scalePtr     [[buffer(0)]], \
    device const half*    normInput    [[buffer(1)]], \
    device const half*    normWeight   [[buffer(2)]], \
    device half*          normOutput   [[buffer(3)]], \
    device const uint8_t* gate_weights [[buffer(4)]], \
    device const half*    gate_scales  [[buffer(5)]], \
    device const half*    gate_biases  [[buffer(6)]], \
    device const uint8_t* up_weights   [[buffer(7)]], \
    device const half*    up_scales    [[buffer(8)]], \
    device const half*    up_biases    [[buffer(9)]], \
    device half*          output       [[buffer(10)]], \
    constant uint&        actualBatch  [[buffer(11)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint tid         [[thread_index_in_threadgroup]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    threadgroup half Xs[16 * (32 + 8)]; \
    threadgroup float Wg[32 * (32 + 8)]; \
    threadgroup float Wu[32 * (32 + 8)]; \
    agent_norm_scale_fused_affine_gate_up_qmm_fixed_batched_full<ROWS, COLS, 64, 16, 0, true>( \
        scalePtr, normInput, normWeight, normOutput, \
        gate_weights, gate_scales, gate_biases, \
        up_weights, up_scales, up_biases, output, actualBatch, \
        Xs, Wg, Wu, \
        tgid, tid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_GEGLU_FIXED_FULL(NAME, ROWS, COLS, GROUP_SIZE) \
    AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_GLU_FIXED_FULL(NAME, ROWS, COLS, GROUP_SIZE, 1)

#define AGENT_DECLARE_FUSED_DUAL_AFFINE_FIXED(NAME, ROWS, COLS) \
kernel void NAME( \
    device const uint8_t* w1_weights [[buffer(0)]], \
    device const half*    w1_scales  [[buffer(1)]], \
    device const half*    w1_biases  [[buffer(2)]], \
    device const uint8_t* w2_weights [[buffer(3)]], \
    device const half*    w2_scales  [[buffer(4)]], \
    device const half*    w2_biases  [[buffer(5)]], \
    device const half*    input      [[buffer(6)]], \
    device half*          output1    [[buffer(7)]], \
    device half*          output2    [[buffer(8)]], \
    uint tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane  [[thread_index_in_simdgroup]], \
    uint simd_group [[simdgroup_index_in_threadgroup]] \
) { \
    fused_dual_affine_matvec_fixed<ROWS, COLS, 64>( \
        w1_weights, w1_scales, w1_biases, \
        w2_weights, w2_scales, w2_biases, \
        input, output1, output2, \
        tgid, simd_lane, simd_group \
    ); \
}

#define AGENT_DECLARE_FUSED_DUAL_AFFINE_BATCHED(NAME, ROWS, COLS) \
kernel void NAME( \
    device const uint8_t* w1_weights [[buffer(0)]], \
    device const half*    w1_scales  [[buffer(1)]], \
    device const half*    w1_biases  [[buffer(2)]], \
    device const uint8_t* w2_weights [[buffer(3)]], \
    device const half*    w2_scales  [[buffer(4)]], \
    device const half*    w2_biases  [[buffer(5)]], \
    device const half*    input      [[buffer(6)]], \
    device half*          output1    [[buffer(7)]], \
    device half*          output2    [[buffer(8)]], \
    uint2 tgid       [[threadgroup_position_in_grid]], \
    uint simd_lane   [[thread_index_in_simdgroup]], \
    uint simd_group  [[simdgroup_index_in_threadgroup]] \
) { \
    fused_dual_affine_matvec_fixed_batched<ROWS, COLS, 64>( \
        w1_weights, w1_scales, w1_biases, \
        w2_weights, w2_scales, w2_biases, \
        input, output1, output2, \
        tgid, simd_lane, simd_group \
    ); \
}

kernel void fused_affine_gate_up_swiglu_c1024_r3584_g64(
    device const uint8_t* gate_weights [[buffer(0)]],
    device const half*    gate_scales  [[buffer(1)]],
    device const half*    gate_biases  [[buffer(2)]],
    device const uint8_t* up_weights   [[buffer(3)]],
    device const half*    up_scales    [[buffer(4)]],
    device const half*    up_biases    [[buffer(5)]],
    device const half*    input        [[buffer(6)]],
    device half*          output       [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    fused_affine_gate_up_swiglu_fixed<3584, 1024, 64>(
        gate_weights, gate_scales, gate_biases,
        up_weights, up_scales, up_biases,
        input, output,
        tgid, simd_lane, simd_group
    );
}

kernel void fused_affine_gate_up_swiglu_c1024_r3584_g64_rows4(
    device const uint8_t* gate_weights [[buffer(0)]],
    device const half*    gate_scales  [[buffer(1)]],
    device const half*    gate_biases  [[buffer(2)]],
    device const uint8_t* up_weights   [[buffer(3)]],
    device const half*    up_scales    [[buffer(4)]],
    device const half*    up_biases    [[buffer(5)]],
    device const half*    input        [[buffer(6)]],
    device half*          output       [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half4 scaleBiasCache[4 * (1024 / 64)];
    fused_affine_gate_up_swiglu_fixed_rows4_sbcache<3584, 1024, 64>(
        gate_weights, gate_scales, gate_biases,
        up_weights, up_scales, up_biases,
        input, output, scaleBiasCache,
        tgid, simd_lane, simd_group
    );
}
AGENT_DECLARE_AFFINE_MATVEC_FIXED(affine_matvec_c1024_r2048_g64, 2048, 1024)
AGENT_DECLARE_AFFINE_MATVEC_FIXED(affine_matvec_c1024_r3584_g64, 3584, 1024)
AGENT_DECLARE_AFFINE_MATVEC_FIXED(affine_matvec_c1024_r4096_g64, 4096, 1024)
AGENT_DECLARE_AFFINE_MATVEC_FIXED(affine_matvec_c1024_r512_g64, 512, 1024)
AGENT_DECLARE_AFFINE_MATVEC_FIXED(affine_matvec_c1024_r6144_g64, 6144, 1024)
AGENT_DECLARE_AFFINE_MATVEC_FIXED(affine_matvec_c2048_r1024_g64, 1024, 2048)
AGENT_DECLARE_AFFINE_MATVEC_FIXED(affine_matvec_c3584_r1024_g64, 1024, 3584)
AGENT_DECLARE_AFFINE_MATVEC_FIXED(affine_matvec_c1024_r248320_g64, 248320, 1024)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4(affine_matvec_c1024_r2048_g64_rows4, 2048, 1024)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4(affine_matvec_c1024_r6144_g64_rows4, 6144, 1024)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4(affine_matvec_c2048_r1024_g64_rows4, 1024, 2048)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4(affine_matvec_c3584_r1024_g64_rows4, 1024, 3584)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4(affine_matvec_c1024_r248320_g64_rows4, 248320, 1024)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS8_GROUP(affine_matvec_c2048_r151936_g64_rows8, 151936, 2048, 64)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c1536_r2048_g128_rows4, 2048, 1536, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c1536_r256_g128_rows4, 256, 1536, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS8_GROUP(affine_matvec_c1536_r256_g128_rows8, 256, 1536, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c1536_r6144_g128_rows4, 6144, 1536, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c1536_r12288_g128_rows4, 12288, 1536, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c1536_r262144_g128_rows4, 262144, 1536, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS8_GROUP(affine_matvec_c1536_r262144_g128_rows8, 262144, 1536, 128)
AGENT_DECLARE_AFFINE_MATVEC_ARGMAX_FIXED_BATCHED_GROUP(affine_matvec_argmax_c1536_r262144_g128_batched, 262144, 1536, 128, 4)
AGENT_DECLARE_LM_HEAD_ARGMAX_REDUCE(lm_head_argmax_reduce_r262144, 262144)
AGENT_DECLARE_FUSED_AFFINE_GATE_UP_GEGLU_FIXED_ROWS4_GROUP(fused_affine_gate_up_geglu_c1536_r6144_g128_rows4, 6144, 1536, 128)
AGENT_DECLARE_FUSED_AFFINE_GATE_UP_GEGLU_FIXED_ROWS4_GROUP(fused_affine_gate_up_geglu_c1536_r12288_g128_rows4, 12288, 1536, 128)
AGENT_DECLARE_FUSED_AFFINE_GATE_UP_GEGLU_FIXED_ROWS8_GROUP(fused_affine_gate_up_geglu_c1536_r6144_g128_rows8, 6144, 1536, 128)
AGENT_DECLARE_FUSED_AFFINE_GATE_UP_GEGLU_FIXED_ROWS8_GROUP(fused_affine_gate_up_geglu_c1536_r12288_g128_rows8, 12288, 1536, 128)
AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_GEGLU_FIXED_ROWS4_GROUP(norm_scale_affine_gate_up_geglu_c1536_r6144_g128_rows4, 6144, 1536, 128)
AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_GEGLU_FIXED_ROWS4_GROUP(norm_scale_affine_gate_up_geglu_c1536_r12288_g128_rows4, 12288, 1536, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c256_r1536_g128_rows4, 1536, 256, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c2048_r1536_g128_rows4, 1536, 2048, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c6144_r1536_g128_rows4, 1536, 6144, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c12288_r1536_g128_rows4, 1536, 12288, 128)
AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_GROUP(fused_affine_matvec_add_c2048_r1536_g128_rows4, 1536, 2048, 128)
AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED(fused_affine_matvec_add_c4096_r1536_g128, 1536, 4096, 128)
AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_GROUP(fused_affine_matvec_add_c6144_r1536_g128_rows4, 1536, 6144, 128)
AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_GROUP(fused_affine_matvec_add_c12288_r1536_g128_rows4, 1536, 12288, 128)
AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(norm_scale_affine_matvec_c1536_r2048_g128_rows4, 2048, 1536, 128)
AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(norm_scale_affine_matvec_c1536_r12288_g128_rows4, 12288, 1536, 128)
AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(norm_scale_affine_matvec_c1536_r262144_g128_rows4, 262144, 1536, 128)
AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_ROWS8_GROUP(norm_scale_affine_matvec_c1536_r262144_g128_rows8, 262144, 1536, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c2560_r2048_g128_rows4, 2048, 2560, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c2560_r4096_g128_rows4, 4096, 2560, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c2560_r512_g128_rows4, 512, 2560, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c2560_r1024_g128_rows4, 1024, 2560, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c2560_r10240_g128_rows4, 10240, 2560, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c2560_r10752_g128_rows4, 10752, 2560, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c2048_r2560_g128_rows4, 2560, 2048, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c4096_r2560_g128_rows4, 2560, 4096, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c10240_r2560_g128_rows4, 2560, 10240, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_SG1_GROUP(affine_matvec_c10240_r2560_g128_rows4_sg1, 2560, 10240, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c2560_r256_g128_rows4, 256, 2560, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c256_r2560_g128_rows4, 2560, 256, 128)
AGENT_DECLARE_FUSED_AFFINE_GATE_UP_GEGLU_FIXED_ROWS4_GROUP(fused_affine_gate_up_geglu_c256_r2048_g128_rows4, 2048, 256, 128)
AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_GEGLU_FIXED_ROWS4_GROUP(norm_scale_affine_gate_up_geglu_c256_r2048_g128_rows4, 2048, 256, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c256_r1024_g128_rows4, 1024, 256, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c256_r2048_g128_rows4, 2048, 256, 128)
AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(norm_scale_affine_matvec_c256_r1024_g128_rows4, 1024, 256, 128)
AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(norm_scale_affine_matvec_c256_r2048_g128_rows4, 2048, 256, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c1024_r256_g128_rows4, 256, 1024, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(affine_matvec_c2048_r256_g128_rows4, 256, 2048, 128)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS8_GROUP(affine_matvec_c2560_r262144_g128_rows8, 262144, 2560, 128)
AGENT_DECLARE_AFFINE_MATVEC_ARGMAX_FIXED_BATCHED_GROUP(affine_matvec_argmax_c2560_r262144_g128_batched, 262144, 2560, 128, 4)
AGENT_DECLARE_FUSED_AFFINE_GATE_UP_GEGLU_FIXED_ROWS4_GROUP(fused_affine_gate_up_geglu_c2560_r10240_g128_rows4, 10240, 2560, 128)
AGENT_DECLARE_FUSED_AFFINE_GATE_UP_GEGLU_FIXED_BT_SG(fused_affine_gate_up_geglu_c2560_r10240_g128_batched, 10240, 2560, 128, 3, 2)
AGENT_DECLARE_FUSED_AFFINE_GATE_UP_GEGLU_FIXED_BT_SG(fused_affine_gate_up_geglu_c2560_r10240_g128_batched_bt4_sg4, 10240, 2560, 128, 4, 4)
AGENT_DECLARE_FUSED_AFFINE_GATE_UP_GEGLU_FIXED_FULL(fused_affine_gate_up_geglu_c2560_r10240_g128_batched_full, 10240, 2560, 128)
AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_GEGLU_FIXED_ROWS4_GROUP(norm_scale_affine_gate_up_geglu_c2560_r10240_g128_rows4, 10240, 2560, 128)
AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_GEGLU_FIXED_FULL(norm_scale_affine_gate_up_geglu_c2560_r10240_g128_batched_full, 10240, 2560, 128)
AGENT_DECLARE_NORM_ADD_SCALE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(norm_add_scale_affine_matvec_c2560_r256_g128_rows4, 256, 2560, 128)
AGENT_DECLARE_FUSED_DUAL_AFFINE_FIXED(fused_dual_affine_matvec_c1024_r16_g64, 16, 1024)
kernel void fused_dual_affine_matvec_c1024_r16_g64_rows4(
    device const uint8_t* w1_weights [[buffer(0)]],
    device const half*    w1_scales  [[buffer(1)]],
    device const half*    w1_biases  [[buffer(2)]],
    device const uint8_t* w2_weights [[buffer(3)]],
    device const half*    w2_scales  [[buffer(4)]],
    device const half*    w2_biases  [[buffer(5)]],
    device const half*    input      [[buffer(6)]],
    device half*          output1    [[buffer(7)]],
    device half*          output2    [[buffer(8)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    fused_dual_affine_matvec_fixed_two_rows_per_simdgroup<16, 1024, 64, 4>(
        w1_weights, w1_scales, w1_biases,
        w2_weights, w2_scales, w2_biases,
        input, output1, output2,
        tgid, simd_lane, simd_group
    );
}
AGENT_DECLARE_FUSED_DUAL_AFFINE_BATCHED(fused_dual_affine_matvec_c1024_r512_g64_batched, 512, 1024)
AGENT_DECLARE_FUSED_DUAL_AFFINE_BATCHED(fused_dual_affine_matvec_c1024_r16_g64_batched, 16, 1024)
AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_GROUP(fused_affine_matvec_add_c2048_r1024_g64_rows4, 1024, 2048, 64)
AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_FULL(fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full, 3584, 1024)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL(affine_matvec_c1024_r2048_g64_batched_full, 2048, 1024)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL(affine_matvec_c1024_r3584_g64_batched_full, 3584, 1024)
AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_FULL(fused_affine_matvec_add_c2048_r1024_g64_batched_full, 1024, 2048)
AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_FULL(fused_affine_matvec_add_c3584_r1024_g64_batched_full, 1024, 3584)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL(affine_matvec_c1024_r4096_g64_batched_full, 4096, 1024)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL(affine_matvec_c1024_r512_g64_batched_full, 512, 1024)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL(affine_matvec_c1024_r6144_g64_batched_full, 6144, 1024)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL_PADDED_REFERENCE(affine_matvec_c1024_r2048_g64_batched_full_padded_reference, 2048, 1024)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL_PADDED_REFERENCE(affine_matvec_c2048_r2048_g64_batched_full_padded_reference, 2048, 2048)
AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_FULL_PADDED_REFERENCE(fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full_padded_reference, 3584, 1024)
AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_FULL_PADDED_REFERENCE(fused_affine_gate_up_swiglu_c2048_r6144_g64_batched_full_padded_reference, 6144, 2048)
AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED(fused_affine_gate_up_swiglu_c1024_r3584_g64_rows4_nocache_reference, 3584, 1024)
AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED(fused_affine_gate_up_swiglu_c2048_r6144_g64_rows4_nocache_reference, 6144, 2048)
AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_FULL_PADDED_REFERENCE(norm_scale_affine_matvec_c1024_r6144_g64_batched_full_padded_reference, 6144, 1024)
AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_FULL_PADDED_REFERENCE(norm_scale_affine_matvec_c2048_r6144_g64_batched_full_padded_reference, 6144, 2048)
AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_SWIGLU_FIXED_FULL_PADDED_REFERENCE(norm_scale_affine_gate_up_swiglu_c1024_r3584_g64_batched_full_padded_reference, 3584, 1024)
AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_SWIGLU_FIXED_FULL_PADDED_REFERENCE(norm_scale_affine_gate_up_swiglu_c2048_r6144_g64_batched_full_padded_reference, 6144, 2048)
AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_SWIGLU_FIXED_FULL(norm_scale_affine_gate_up_swiglu_c1024_r3584_g64_batched_full, 3584, 1024)
AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_FULL(norm_scale_affine_matvec_c1024_r4096_g64_batched_full, 4096, 1024)
AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_FULL(norm_scale_affine_matvec_c1024_r6144_g64_batched_full, 6144, 1024)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL(affine_matvec_c2048_r1024_g64_batched_full, 1024, 2048)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL(affine_matvec_c3584_r1024_g64_batched_full, 1024, 3584)
AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_FULL(fused_affine_gate_up_swiglu_c2048_r8192_g64_batched_full, 8192, 2048)
AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_FULL(fused_affine_matvec_add_c2048_r2048_g64_batched_full, 2048, 2048)
AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_FULL(fused_affine_matvec_add_c6144_r2048_g64_batched_full, 2048, 6144)
AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_FULL(fused_affine_matvec_add_c8192_r2048_g64_batched_full, 2048, 8192)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL(affine_matvec_c8192_r2048_g64_batched_full, 2048, 8192)
AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_SWIGLU_FIXED_FULL(norm_scale_affine_gate_up_swiglu_c2048_r6144_g64_batched_full, 6144, 2048)
AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_SWIGLU_FIXED_FULL(norm_scale_affine_gate_up_swiglu_c2048_r8192_g64_batched_full, 8192, 2048)
AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_FULL(norm_scale_affine_matvec_c2048_r2048_g64_batched_full, 2048, 2048)
AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_FULL(norm_scale_affine_matvec_c2048_r4096_g64_batched_full, 4096, 2048)
AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_FULL(norm_scale_affine_matvec_c2048_r6144_g64_batched_full, 6144, 2048)

AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_ROWS4_SBCACHE_GROUP(fused_affine_gate_up_swiglu_c2048_r11008_g64, 11008, 2048, 64)

AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_ROWS4_SBCACHE_GROUP(fused_affine_gate_up_swiglu_c2560_r9216_g64, 9216, 2560, 64)
AGENT_DECLARE_AFFINE_MATVEC_FIXED(affine_matvec_c2560_r8192_g64, 8192, 2560)
AGENT_DECLARE_AFFINE_MATVEC_FIXED(affine_matvec_c2560_r4096_g64, 4096, 2560)
AGENT_DECLARE_AFFINE_MATVEC_FIXED(affine_matvec_c2560_r1024_g64, 1024, 2560)
AGENT_DECLARE_AFFINE_MATVEC_FIXED(affine_matvec_c4096_r2560_g64, 2560, 4096)
AGENT_DECLARE_AFFINE_MATVEC_FIXED(affine_matvec_c9216_r2560_g64, 2560, 9216)
AGENT_DECLARE_AFFINE_MATVEC_FIXED(affine_matvec_c2560_r248320_g64, 248320, 2560)
AGENT_DECLARE_FUSED_DUAL_AFFINE_FIXED(fused_dual_affine_matvec_c2560_r32_g64, 32, 2560)
AGENT_DECLARE_FUSED_DUAL_AFFINE_BATCHED(fused_dual_affine_matvec_c2560_r1024_g64_batched, 1024, 2560)
AGENT_DECLARE_FUSED_DUAL_AFFINE_BATCHED(fused_dual_affine_matvec_c2560_r32_g64_batched, 32, 2560)
AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_FULL(fused_affine_gate_up_swiglu_c2560_r9216_g64_batched_full, 9216, 2560)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL(affine_matvec_c2560_r8192_g64_batched_full, 8192, 2560)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL(affine_matvec_c2560_r4096_g64_batched_full, 4096, 2560)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL(affine_matvec_c2560_r1024_g64_batched_full, 1024, 2560)
AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_FULL(fused_affine_matvec_add_c4096_r2560_g64_batched_full, 2560, 4096)
AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_FULL(fused_affine_matvec_add_c9216_r2560_g64_batched_full, 2560, 9216)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL(affine_matvec_c4096_r2560_g64_batched_full, 2560, 4096)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL(affine_matvec_c9216_r2560_g64_batched_full, 2560, 9216)
AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_SWIGLU_FIXED_FULL(norm_scale_affine_gate_up_swiglu_c2560_r9216_g64_batched_full, 9216, 2560)
AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_FULL(norm_scale_affine_matvec_c2560_r8192_g64_batched_full, 8192, 2560)
AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED(fused_affine_gate_up_swiglu_c3072_r8192_g64, 8192, 3072)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4(affine_matvec_c3072_r3072_g64, 3072, 3072)
AGENT_DECLARE_AFFINE_MATVEC_FIXED(affine_matvec_c3072_r1024_g64, 1024, 3072)
AGENT_DECLARE_AFFINE_MATVEC_FIXED(affine_matvec_c3072_r8192_g64, 8192, 3072)
AGENT_DECLARE_AFFINE_MATVEC_FIXED(affine_matvec_c8192_r3072_g64, 3072, 8192)
AGENT_DECLARE_FUSED_DUAL_AFFINE_BATCHED(fused_dual_affine_matvec_c3072_r1024_g64_batched, 1024, 3072)
AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_FULL(fused_affine_gate_up_swiglu_c3072_r8192_g64_batched_full, 8192, 3072)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL(affine_matvec_c3072_r3072_g64_batched_full, 3072, 3072)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL(affine_matvec_c3072_r1024_g64_batched_full, 1024, 3072)
AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL(affine_matvec_c8192_r3072_g64_batched_full, 3072, 8192)
AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_FULL(fused_affine_matvec_add_c3072_r3072_g64_batched_full, 3072, 3072)
AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_FULL(fused_affine_matvec_add_c8192_r3072_g64_batched_full, 3072, 8192)
AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_SWIGLU_FIXED_FULL(norm_scale_affine_gate_up_swiglu_c3072_r8192_g64_batched_full, 8192, 3072)
AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_FULL(norm_scale_affine_matvec_c3072_r3072_g64_batched_full, 3072, 3072)

#undef AGENT_DECLARE_AFFINE_MATVEC_FIXED
#undef AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4
#undef AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP
#undef AGENT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_SG1_GROUP
#undef AGENT_DECLARE_AFFINE_MATVEC_ARGMAX_FIXED_BATCHED_GROUP
#undef AGENT_DECLARE_LM_HEAD_ARGMAX_REDUCE
#undef AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_ROWS4_GROUP
#undef AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL
#undef AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL_PADDED_REFERENCE
#undef AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_FULL_PADDED_REFERENCE
#undef AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_FULL_PADDED_REFERENCE
#undef AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_SWIGLU_FIXED_FULL_PADDED_REFERENCE
#undef AGENT_DECLARE_AFFINE_MATVEC_FIXED_FULL_SCALAR
#undef AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_TGINPUT_GROUP
#undef AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_SG1_GROUP
#undef AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_SBCACHE_GROUP
#undef AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS16_GROUP
#undef AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS32_GROUP
#undef AGENT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_FULL
#undef AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED
#undef AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_ROWS4_TGINPUT_GROUP
#undef AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_ROWS4_EXP2_GROUP
#undef AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_ROWS4_SBCACHE_GROUP
#undef AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_ROWS16_GROUP
#undef AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_ROWS32_GROUP
#undef AGENT_DECLARE_FUSED_AFFINE_GATE_UP_GEGLU_FIXED_ROWS4_GROUP
#undef AGENT_DECLARE_FUSED_AFFINE_GATE_UP_GEGLU_FIXED_ROWS8_GROUP
#undef AGENT_DECLARE_FUSED_AFFINE_GATE_UP_GEGLU_FIXED_BT_SG
#undef AGENT_DECLARE_FUSED_AFFINE_GATE_UP_GEGLU_FIXED_FULL
#undef AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_GEGLU_FIXED_ROWS4_GROUP
#undef AGENT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_FULL
#undef AGENT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_FULL
#undef AGENT_DECLARE_NORM_SCALE_AFFINE_GATE_UP_SWIGLU_FIXED_FULL
#undef AGENT_DECLARE_FUSED_DUAL_AFFINE_FIXED
#undef AGENT_DECLARE_FUSED_DUAL_AFFINE_BATCHED

#undef AGENT_AFFINE_DOT16

// ─── Argmax over FP16 logits ───

kernel void argmax_fp16_partials(
    device const half* logits     [[buffer(0)]],
    device uint2*      partials   [[buffer(1)]],
    constant uint&     vocabSize  [[buffer(2)]],
    constant uint&     chunkSize  [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_index_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    uint start = tgid * chunkSize;
    uint end = min(start + chunkSize, vocabSize);
    uint2 best = uint2(0, 0);
    for (uint i = start + ltid; i < end; i += tgs) {
        best = agent_argmax_key_max(best, agent_argmax_pack_key(logits[i], i));
    }

    threadgroup uint2 simdBest[32];
    uint lane = ltid & 31u;
    uint simdID = ltid >> 5u;
    for (uint offset = 16; offset > 0; offset >>= 1) {
        uint2 other = simd_shuffle_down(best, offset);
        best = agent_argmax_key_max(best, other);
    }
    if (lane == 0) {
        simdBest[simdID] = best;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdID == 0) {
        uint simdCount = tgs >> 5u;
        best = lane < simdCount ? simdBest[lane] : uint2(0, 0);
        for (uint offset = 16; offset > 0; offset >>= 1) {
            uint2 other = simd_shuffle_down(best, offset);
            best = agent_argmax_key_max(best, other);
        }
        if (lane == 0) {
            partials[tgid] = best;
        }
    }
}

kernel void argmax_key_reduce(
    device const uint2* partials    [[buffer(0)]],
    device int*         output      [[buffer(1)]],
    constant uint&      numPartials [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint THREADS = 256;
    uint2 best = uint2(0, 0);
    for (uint i = tid; i < numPartials; i += THREADS) {
        best = agent_argmax_key_max(best, partials[i]);
    }

    threadgroup uint2 scratch[THREADS];
    scratch[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = agent_argmax_key_max(scratch[tid], scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        output[0] = int(AGENT_ARGMAX_MAX_INDEX - scratch[0].y);
    }
}

// Single-threadgroup: all threads stride over the entire vocab.
// IMPORTANT: dispatch exactly 1 threadgroup — no cross-TG reduction is performed.

kernel void argmax_fp16(
    device const half* logits   [[buffer(0)]],  // [vocab]
    device atomic_int* result   [[buffer(1)]],  // [2]: {best_idx, best_val_as_int}
    constant uint&     vocabSize [[buffer(2)]],
    uint tid  [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_index_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    // Each thread finds its local max
    float bestVal = -INFINITY;
    int bestIdx = 0;
    for (uint i = ltid; i < vocabSize; i += tgs) {  // stride over vocab
        float v = float(logits[i]);
        if (v > bestVal) { bestVal = v; bestIdx = int(i); }
    }

    // SIMD reduction
    for (uint offset = 16; offset > 0; offset >>= 1) {
        float otherVal = simd_shuffle_down(bestVal, offset);
        int otherIdx = simd_shuffle_down(bestIdx, offset);
        if (otherVal > bestVal) { bestVal = otherVal; bestIdx = otherIdx; }
    }

    // Threadgroup reduction
    threadgroup float tgBestVal[32];
    threadgroup int tgBestIdx[32];
    uint simd_id = ltid / 32;
    uint lane = ltid % 32;
    if (lane == 0) { tgBestVal[simd_id] = bestVal; tgBestIdx[simd_id] = bestIdx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (ltid == 0) {
        float localBest = -INFINITY;
        int localIdx = 0;
        uint nSimds = tgs / 32;
        for (uint s = 0; s < nSimds; s++) {
            if (tgBestVal[s] > localBest) { localBest = tgBestVal[s]; localIdx = tgBestIdx[s]; }
        }
        // Atomic compare-exchange with global result
        // Encode val+idx into result[0]=idx, use result[1] for val comparison
        // Simple approach: just write if better (race-free if single threadgroup)
        atomic_store_explicit(&result[0], localIdx, memory_order_relaxed);
    }
}

static inline ulong agent_splitmix64(ulong value) {
    value += 0x9E3779B97F4A7C15ull;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

static inline float agent_deterministic_unit(ulong seed, uint position) {
    ulong bits = agent_splitmix64(
        seed + 0x9E3779B97F4A7C15ull + ulong(position)
    );
    uint mantissa = uint((bits >> 40) & 0x00FF'FFFFull);
    float unit = (float(mantissa) + 0.5f) * (1.0f / 16777216.0f);
    return clamp(unit, FLT_MIN, 1.0f - FLT_EPSILON);
}

// ─── Temperature sampling over FP16 logits ───
// Two implementations:
//   sample_temperature_fp16 — exact inverse-CDF sampling. Phase 3 is a
//     serial scan over vocab on thread 0, costing ~5 ms for 262K vocab.
//   sample_temperature_gumbel_fp16 — Gumbel-max equivalent. argmax_i(logit_i/T
//     + g_i) where g_i = -log(-log(u_i)) draws the same distribution as
//     softmax(logits/T) sampling, but in a single PARALLEL argmax pass.
//     Per-token uniform u_i is keyed on (seed, position, i) so reproducible
//     across runs (different stream than the inverse-CDF path).
// Both dispatch exactly 1 threadgroup.
kernel void sample_temperature_fp16(
    device const half* logits      [[buffer(0)]],  // [vocab]
    device int*        result      [[buffer(1)]],  // [1]: sampled token
    constant uint&     vocabSize   [[buffer(2)]],
    constant float&    invTemp     [[buffer(3)]],
    constant ulong&    seed        [[buffer(4)]],
    constant uint&     position    [[buffer(5)]],
    uint ltid [[thread_index_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    float bestVal = -INFINITY;
    int bestIdx = 0;
    for (uint i = ltid; i < vocabSize; i += tgs) {
        float v = float(logits[i]);
        if (v > bestVal) {
            bestVal = v;
            bestIdx = int(i);
        }
    }

    for (uint offset = 16; offset > 0; offset >>= 1) {
        float otherVal = simd_shuffle_down(bestVal, offset);
        int otherIdx = simd_shuffle_down(bestIdx, offset);
        if (otherVal > bestVal) {
            bestVal = otherVal;
            bestIdx = otherIdx;
        }
    }

    threadgroup float tgBestVal[32];
    threadgroup int tgBestIdx[32];
    threadgroup float tgMass[32];
    threadgroup float globalMax;
    threadgroup int fallbackIdx;

    uint simdID = ltid / 32;
    uint lane = ltid % 32;
    if (lane == 0) {
        tgBestVal[simdID] = bestVal;
        tgBestIdx[simdID] = bestIdx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (ltid == 0) {
        float localBest = -INFINITY;
        int localIdx = 0;
        uint nSimds = tgs / 32;
        for (uint s = 0; s < nSimds; s++) {
            if (tgBestVal[s] > localBest) {
                localBest = tgBestVal[s];
                localIdx = tgBestIdx[s];
            }
        }
        globalMax = localBest;
        fallbackIdx = localIdx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float localMass = 0.0f;
    for (uint i = ltid; i < vocabSize; i += tgs) {
        localMass += exp((float(logits[i]) - globalMax) * invTemp);
    }
    localMass = simd_sum(localMass);
    if (lane == 0) {
        tgMass[simdID] = localMass;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (ltid == 0) {
        float totalMass = 0.0f;
        uint nSimds = tgs / 32;
        for (uint s = 0; s < nSimds; s++) {
            totalMass += tgMass[s];
        }

        if (!isfinite(totalMass) || totalMass <= 0.0f) {
            result[0] = fallbackIdx;
        } else {
            float threshold = agent_deterministic_unit(seed, position) * totalMass;
            float runningMass = 0.0f;
            int sampledIdx = fallbackIdx;
            for (uint i = 0; i < vocabSize; i++) {
                runningMass += exp((float(logits[i]) - globalMax) * invTemp);
                if (runningMass >= threshold) {
                    sampledIdx = int(i);
                    break;
                }
            }
            result[0] = sampledIdx;
        }
    }
}

// Gumbel-max temperature sampler: single parallel pass that draws from
// softmax(logits/T). For each token i, score = logit_i * invTemp + g_i where
// g_i = -log(-log(u_i)) is Gumbel(0,1) noise. Argmax of scores is a sample
// from the softmax distribution. Per-token u_i is hashed from (seed, position, i)
// using splitmix64 so the stream is reproducible. Same parallel-reduce structure
// as the argmax kernels, so cost ≈ argmax cost (~50-100 µs for 262K vocab).
kernel void sample_temperature_gumbel_fp16(
    device const half* logits      [[buffer(0)]],
    device int*        result      [[buffer(1)]],
    constant uint&     vocabSize   [[buffer(2)]],
    constant float&    invTemp     [[buffer(3)]],
    constant ulong&    seed        [[buffer(4)]],
    constant uint&     position    [[buffer(5)]],
    uint ltid [[thread_index_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    float bestVal = -INFINITY;
    int bestIdx = 0;

    // Mix (seed, position) once so per-token rehash is cheap.
    ulong base = agent_splitmix64(seed + 0x9E3779B97F4A7C15ull + ulong(position));

    for (uint i = ltid; i < vocabSize; i += tgs) {
        // Stream RNG: per-token uniform from splitmix64(base ^ i).
        // The (mantissa + 0.5) / 2^24 form keeps u in (~3e-8, 1-3e-8),
        // already inside the log-log domain — no clamp needed.
        ulong bits = agent_splitmix64(base ^ ulong(i));
        uint mantissa = uint((bits >> 40) & 0x00FF'FFFFull);
        float u = (float(mantissa) + 0.5f) * (1.0f / 16777216.0f);
        float g = -log(-log(u));
        float v = float(logits[i]) * invTemp + g;
        if (v > bestVal) {
            bestVal = v;
            bestIdx = int(i);
        }
    }

    for (uint offset = 16; offset > 0; offset >>= 1) {
        float otherVal = simd_shuffle_down(bestVal, offset);
        int otherIdx = simd_shuffle_down(bestIdx, offset);
        if (otherVal > bestVal) {
            bestVal = otherVal;
            bestIdx = otherIdx;
        }
    }

    threadgroup float tgBestVal[32];
    threadgroup int tgBestIdx[32];

    uint simdID = ltid / 32;
    uint lane = ltid % 32;
    if (lane == 0) {
        tgBestVal[simdID] = bestVal;
        tgBestIdx[simdID] = bestIdx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (ltid == 0) {
        float localBest = -INFINITY;
        int localIdx = 0;
        uint nSimds = tgs / 32;
        for (uint s = 0; s < nSimds; s++) {
            if (tgBestVal[s] > localBest) {
                localBest = tgBestVal[s];
                localIdx = tgBestIdx[s];
            }
        }
        result[0] = localIdx;
    }
}

// ─── TurboQuant-H matvec (two kernels) ───
//
// Why two kernels: the per-row dequant W[r,:] would be O(G^2) per
// group (G=128) per row. For LM head shapes (262144 rows) the naive
// path costs ~G^2 * num_groups * rows ≈ 86B fp ops. The matvec
// decomposes:
//   y[r] = sum_c x[c] * W[r,c]
//        = sum_g sum_p x[g*G+p] * inv_sqrt_G * sum_j sign(p,j) * cb[g, code[r,g,j]]
//        = inv_sqrt_G * sum_g sum_j cb[g, code[r,g,j]] * X_hat[g,j]
//   X_hat[g,j] = sum_p sign(p,j) * x[g*G+p]
// so X_hat is shared across rows. Precompute it once per matvec,
// then the per-row work drops to O(G * num_groups) — a G× win.
// `sign(p,j)` is the Sylvester Hadamard element, identical to the
// convention in tqh_embedding_gather above.

kernel void tqh_matvec_prepare_input(
    device const half*    input    [[buffer(0)]],   // [cols] fp16
    device float*         xHat     [[buffer(1)]],   // [numGroups * G] fp32 scratch
    constant uint&        cols     [[buffer(2)]],
    uint                  gid      [[thread_position_in_grid]]
) {
    constexpr uint G = 128;

    uint g = gid / G;
    uint j = gid % G;
    uint groupBase = g * G;

    float acc = 0.0f;
    for (uint p = 0; p < G; ++p) {
        uint colIdx = groupBase + p;
        if (colIdx >= cols) break;  // partial final group: x is 0 in pad slots
        uint parity = popcount(p & j) & 1;
        float xp = float(input[colIdx]);
        acc += (parity != 0) ? -xp : xp;
    }
    xHat[gid] = acc;
}

// Decode-path TQH matvec. Mirrors the batched SG4 design: 4 rows
// per simdgroup, threads stride across the row's 160 u32 codes
// for coalesced 128-byte memory loads. The original per-thread-
// per-row design striped 32 threads in a SIMD across 32 different
// rows (640-byte stride), which Apple Silicon can't coalesce.
kernel void tqh_matvec(
    device const uint8_t* codes        [[buffer(0)]],  // [rows, codesPerRow] packed 2-bit
    device const half*    codebook     [[buffer(1)]],  // [numGroups, 4] fp16
    device const float*   xHat         [[buffer(2)]],  // [numGroups * G] fp32 from prepare
    device half*          output       [[buffer(3)]],  // [rows] fp16
    constant uint&        numRows      [[buffer(4)]],
    constant uint&        cols         [[buffer(5)]],
    constant uint&        codesPerRow  [[buffer(6)]],
    uint                  tgid_x       [[threadgroup_position_in_grid]],
    uint                  simd_lane    [[thread_index_in_simdgroup]],
    uint                  simd_group   [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint G = 128;
    constexpr uint codesPerU32 = 16;
    constexpr uint u32PerGroup = G / 16;
    constexpr uint ROWS_PER_SG = 4;
    constexpr uint SG_PER_TG = 4;
    constexpr float inv_sqrt_G = 0.0883883476483184405f;

    uint baseRow = tgid_x * (SG_PER_TG * ROWS_PER_SG) + simd_group * ROWS_PER_SG;
    if (baseRow >= numRows) return;

    uint numGroups = (cols + G - 1) / G;
    uint u32PerRow = numGroups * u32PerGroup;

    device const uint* codes32 = (device const uint*)codes;
    uint codesPerRowU32 = codesPerRow / 4;

    // Hoist row-tail guards out of the hot loop. Numbers divisible by
    // 16 (LM head vocab=262144, FFN down rows=10240) always take the
    // full-tile fast path on every TG; partial-row TGs only happen
    // at oddly-sized weight matrices.
    bool has1 = baseRow + 1 < numRows;
    bool has2 = baseRow + 2 < numRows;
    bool has3 = baseRow + 3 < numRows;
    uint r0Base = (baseRow + 0) * codesPerRowU32;
    uint r1Base = has1 ? (baseRow + 1) * codesPerRowU32 : r0Base;
    uint r2Base = has2 ? (baseRow + 2) * codesPerRowU32 : r0Base;
    uint r3Base = has3 ? (baseRow + 3) * codesPerRowU32 : r0Base;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // groupIdx is monotone-non-decreasing across a thread's iters
    // (w grows by 32 each iter, groupIdx = w >> 3), so we only need
    // to refresh the 4 codebook centroids when groupIdx actually
    // changes. Saves ~3 of every 5 fp16 codebook loads at cols=2560.
    uint prevGroupIdx = 0xFFFFFFFFu;
    float cb0 = 0, cb1 = 0, cb2 = 0, cb3 = 0;

    for (uint w = simd_lane; w < u32PerRow; w += 32) {
        uint groupIdx = w >> 3;          // / u32PerGroup (=8)
        uint groupOffset = w & 0x7u;     // % u32PerGroup
        if (groupIdx != prevGroupIdx) {
            uint cbBase = groupIdx * 4;
            cb0 = float(codebook[cbBase + 0]);
            cb1 = float(codebook[cbBase + 1]);
            cb2 = float(codebook[cbBase + 2]);
            cb3 = float(codebook[cbBase + 3]);
            prevGroupIdx = groupIdx;
        }
        float cb[4] = { cb0, cb1, cb2, cb3 };

        uint xHatBase = groupIdx * G + groupOffset * codesPerU32;

        uint p0 = codes32[r0Base + w];
        uint p1 = has1 ? codes32[r1Base + w] : 0u;
        uint p2 = has2 ? codes32[r2Base + w] : 0u;
        uint p3 = has3 ? codes32[r3Base + w] : 0u;

        #pragma unroll
        for (uint bb = 0; bb < codesPerU32; ++bb) {
            float xv = xHat[xHatBase + bb];
            uint c0 = (p0 >> (bb * 2)) & 0x3u;
            uint c1 = (p1 >> (bb * 2)) & 0x3u;
            uint c2 = (p2 >> (bb * 2)) & 0x3u;
            uint c3 = (p3 >> (bb * 2)) & 0x3u;
            acc0 += cb[c0] * xv;
            acc1 += cb[c1] * xv;
            acc2 += cb[c2] * xv;
            acc3 += cb[c3] * xv;
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (simd_lane == 0) {
        output[baseRow + 0] = half(acc0 * inv_sqrt_G);
        if (has1) output[baseRow + 1] = half(acc1 * inv_sqrt_G);
        if (has2) output[baseRow + 2] = half(acc2 * inv_sqrt_G);
        if (has3) output[baseRow + 3] = half(acc3 * inv_sqrt_G);
    }
}

// ─── TurboQuant-H matvec, batched (prefill) ───
//
// Mirrors the decode pair but adds a B (batch) dimension. The
// matvec dispatches over (numRows, B) so each (b, r) thread
// produces output[b, r] using xHat[b, :] which is precomputed
// per b by the prepare kernel. xHat scratch sizing must include
// the B factor: scratchPerBatch = numGroups * G fp32, total =
// B * numGroups * G fp32. PrefillEmitter dispatches use
// `dynamicGridH: .seqLen` so the runtime only fires positions
// 0..seqLen-1 each iteration.

kernel void tqh_matvec_prepare_input_batched(
    device const half*    input    [[buffer(0)]],   // [B, cols] fp16
    device float*         xHat     [[buffer(1)]],   // [B, numGroups, G] fp32
    constant uint&        cols     [[buffer(2)]],
    uint2                 gid      [[thread_position_in_grid]]
) {
    constexpr uint G = 128;

    uint b = gid.y;
    uint groupAndLane = gid.x;
    uint g = groupAndLane / G;
    uint j = groupAndLane % G;
    uint groupBase = g * G;

    uint numGroups = (cols + G - 1) / G;
    uint inputBatchBase = b * cols;
    uint xHatBatchBase = b * numGroups * G;

    float acc = 0.0f;
    for (uint p = 0; p < G; ++p) {
        uint colIdx = groupBase + p;
        if (colIdx >= cols) break;
        uint parity = popcount(p & j) & 1;
        float xp = float(input[inputBatchBase + colIdx]);
        acc += (parity != 0) ? -xp : xp;
    }
    xHat[xHatBatchBase + groupAndLane] = acc;
}

// 4 rows per simdgroup. 32 threads stride across the row's u32-packed
// codes (each thread covers 5 of the 160 u32s per row), so codes are
// loaded coalesced within each row — thread t at iter i reads
// codes32[rowBase + i*32 + t] for the same row, which is 32 consecutive
// u32s = 128 bytes per warp memory transaction. The original
// one-thread-per-row design striped the same SIMD across 32 different
// rows, each separated by codesPerRow bytes (640 at cols=2560), and
// Apple Silicon can't coalesce that — empirically stuck at ~4% of
// peak bandwidth. Mirrors affine_matvec_fixed_batched_sg4.
kernel void tqh_matvec_batched(
    device const uint8_t* codes        [[buffer(0)]],  // [numRows, codesPerRow]
    device const half*    codebook     [[buffer(1)]],  // [numGroups, 4]
    device const float*   xHat         [[buffer(2)]],  // [B, numGroups, G]
    device half*          output       [[buffer(3)]],  // [B, numRows]
    constant uint&        numRows      [[buffer(4)]],
    constant uint&        cols         [[buffer(5)]],
    constant uint&        codesPerRow  [[buffer(6)]],
    uint2                 tgid         [[threadgroup_position_in_grid]],
    uint                  simd_lane    [[thread_index_in_simdgroup]],
    uint                  simd_group   [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint G = 128;
    constexpr uint codesPerU32 = 16;
    constexpr uint u32PerGroup = G / 16;
    constexpr uint ROWS_PER_SG = 4;
    constexpr uint SG_PER_TG = 4;
    constexpr float inv_sqrt_G = 0.0883883476483184405f;

    uint baseRow = tgid.x * (SG_PER_TG * ROWS_PER_SG) + simd_group * ROWS_PER_SG;
    uint b = tgid.y;
    if (baseRow >= numRows) return;

    uint numGroups = (cols + G - 1) / G;
    uint u32PerRow = numGroups * u32PerGroup;

    device const uint* codes32 = (device const uint*)codes;
    uint codesPerRowU32 = codesPerRow / 4;
    uint xHatBatchBase = b * numGroups * G;

    // Same hoists as the decode-path tqh_matvec — see that kernel's
    // comment for the rationale.
    bool has1 = baseRow + 1 < numRows;
    bool has2 = baseRow + 2 < numRows;
    bool has3 = baseRow + 3 < numRows;
    uint r0Base = (baseRow + 0) * codesPerRowU32;
    uint r1Base = has1 ? (baseRow + 1) * codesPerRowU32 : r0Base;
    uint r2Base = has2 ? (baseRow + 2) * codesPerRowU32 : r0Base;
    uint r3Base = has3 ? (baseRow + 3) * codesPerRowU32 : r0Base;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    uint prevGroupIdx = 0xFFFFFFFFu;
    float cb0 = 0, cb1 = 0, cb2 = 0, cb3 = 0;

    for (uint w = simd_lane; w < u32PerRow; w += 32) {
        uint groupIdx = w >> 3;
        uint groupOffset = w & 0x7u;
        if (groupIdx != prevGroupIdx) {
            uint cbBase = groupIdx * 4;
            cb0 = float(codebook[cbBase + 0]);
            cb1 = float(codebook[cbBase + 1]);
            cb2 = float(codebook[cbBase + 2]);
            cb3 = float(codebook[cbBase + 3]);
            prevGroupIdx = groupIdx;
        }
        float cb[4] = { cb0, cb1, cb2, cb3 };

        uint xHatBase = xHatBatchBase + groupIdx * G + groupOffset * codesPerU32;

        uint p0 = codes32[r0Base + w];
        uint p1 = has1 ? codes32[r1Base + w] : 0u;
        uint p2 = has2 ? codes32[r2Base + w] : 0u;
        uint p3 = has3 ? codes32[r3Base + w] : 0u;

        #pragma unroll
        for (uint bb = 0; bb < codesPerU32; ++bb) {
            float xv = xHat[xHatBase + bb];
            uint c0 = (p0 >> (bb * 2)) & 0x3u;
            uint c1 = (p1 >> (bb * 2)) & 0x3u;
            uint c2 = (p2 >> (bb * 2)) & 0x3u;
            uint c3 = (p3 >> (bb * 2)) & 0x3u;
            acc0 += cb[c0] * xv;
            acc1 += cb[c1] * xv;
            acc2 += cb[c2] * xv;
            acc3 += cb[c3] * xv;
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2);
    acc3 = simd_sum(acc3);

    if (simd_lane == 0) {
        uint outBase = b * numRows + baseRow;
        output[outBase + 0] = half(acc0 * inv_sqrt_G);
        if (has1) output[outBase + 1] = half(acc1 * inv_sqrt_G);
        if (has2) output[outBase + 2] = half(acc2 * inv_sqrt_G);
        if (has3) output[outBase + 3] = half(acc3 * inv_sqrt_G);
    }
}
