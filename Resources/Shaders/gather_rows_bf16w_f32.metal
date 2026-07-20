#include <metal_stdlib>
using namespace metal;

/// bf16-table variant of gather_rows_f32: the [rows, dim] table is bfloat (the BF16-source
/// text_embedding stored bf16 to halve footprint), widened to fp32 on read (`float(bfloat)` = bits<<16,
/// EXACT for a bf16 source) so each gathered row is bit-identical to the host weightRow widen. Signed
/// ids: id < 0 ⇒ a ZERO row (same projection-only sentinel as gather_rows_f32); valid ids are
/// host-prechecked to 0 <= id < rows. One thread per output element (flat grid = n*dim).
///
/// Buffers: 0 table [rows,dim] bfloat, 1 ids [n] int, 2 out [n,dim] float
/// Constants: 3 n, 4 dim
kernel void gather_rows_bf16w_f32(
    device const bfloat* table [[buffer(0)]],
    device const int*    ids   [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       n     [[buffer(3)]],
    constant uint&       dim   [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n * dim) return;
    uint t = gid / dim, d = gid % dim;
    int id = ids[t];
    out[gid] = (id < 0) ? 0.0f : float(table[uint(id) * dim + d]);
}
