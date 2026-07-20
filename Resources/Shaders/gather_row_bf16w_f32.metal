#include <metal_stdlib>
using namespace metal;

/// bf16-table variant of gather_row_f32: the [rows, dim] table is bfloat (the BF16-source codec
/// embeddings stored bf16 to halve footprint), widened to fp32 on read (`float(bfloat)` = bits<<16,
/// EXACT for a bf16 source) so the gathered row is bit-identical to the f32-stored table.
/// `out[d] = float(table[idx[slot]·dim + d])`. One thread per output element (flat grid = dim).
///
/// Buffers: 0 table [rows,dim] bfloat, 1 idx [>=slot+1] uint, 2 out [dim] float
/// Constants: 3 dim, 4 slot
kernel void gather_row_bf16w_f32(
    device const bfloat* table [[buffer(0)]],
    device const uint*   idx   [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       dim   [[buffer(3)]],
    constant uint&       slot  [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= dim) return;
    out[gid] = float(table[idx[slot] * dim + gid]);
}
