#include <metal_stdlib>
using namespace metal;

// ─── Fused grouped-LUT u4 matrix-matrix multiply (prefill) ───
// Weight matrix [R, C] stored as packed u4 (2 values per byte).
// Each group of `groupSize` rows shares a 16-entry FP16 LUT.
// Input [B, C] row-major, Output [B, R] row-major.
// Dispatch: (R, B) threadgroups, 256 threads each.
// Each threadgroup computes one output element: output[batch, row].

constant uint MATMUL_TPG = 256;

kernel void fused_lut_matmul(
    device const uint8_t* indices   [[buffer(0)]],  // [R, C/2] packed u4
    device const half*    lut       [[buffer(1)]],  // [nGroups, 16]
    device const half*    input     [[buffer(2)]],  // [B, C] row-major
    device half*          output    [[buffer(3)]],  // [B, R] row-major
    constant uint&        cols      [[buffer(4)]],  // C
    constant uint&        groupSize [[buffer(5)]],  // rows per group
    constant uint&        numRows   [[buffer(6)]],  // R
    uint2 tgid      [[threadgroup_position_in_grid]],  // (row, batch)
    uint tid         [[thread_index_in_threadgroup]],
    uint simd_lane   [[thread_index_in_simdgroup]],
    uint simd_group  [[simdgroup_index_in_threadgroup]]
) {
    uint row = tgid.x;
    uint batch = tgid.y;
    uint halfCols = cols / 2;

    // Load group LUT into registers
    uint group = row / groupSize;
    device const half* groupLUT = lut + group * 16;
    half lr[16];
    for (uint i = 0; i < 16; i++) { lr[i] = groupLUT[i]; }

    // Pointer to this batch's input vector
    device const half* batchInput = input + batch * cols;

    // Striped dot product (same reduction as fused_lut_matvec)
    float acc = 0.0f;
    device const uint8_t* rowIdx = indices + row * halfCols;
    for (uint j = tid; j < halfCols; j += MATMUL_TPG) {
        uint8_t packed = rowIdx[j];
        uint col = j * 2;
        acc += float(lr[packed & 0xF]) * float(batchInput[col]);
        acc += float(lr[packed >> 4])  * float(batchInput[col + 1]);
    }

    // SIMD + threadgroup reduction
    acc = simd_sum(acc);
    threadgroup float partial[8];
    if (simd_lane == 0) { partial[simd_group] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0;
        for (uint s = 0; s < MATMUL_TPG / 32; s++) { total += partial[s]; }
        output[batch * numRows + row] = half(total);
    }
}
