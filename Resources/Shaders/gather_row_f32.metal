#include <metal_stdlib>
using namespace metal;

/// Gather one row of a [rows, dim] fp32 table, where the row INDEX is read from a GPU buffer (the
/// argmax output of the previous MTP sub-pass) — so the gather chains on-GPU with no CPU readback.
/// `out[d] = table[idx[slot] * dim + d]`. One thread per output element (flat grid = dim).
///
/// Buffers: 0 table [rows,dim] float, 1 idx [>=slot+1] uint, 2 out [dim] float
/// Constants: 3 dim, 4 slot
kernel void gather_row_f32(
    device const float* table [[buffer(0)]],
    device const uint*  idx   [[buffer(1)]],
    device float*       out   [[buffer(2)]],
    constant uint&      dim   [[buffer(3)]],
    constant uint&      slot  [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dim) return;
    out[gid] = table[idx[slot] * dim + gid];
}
