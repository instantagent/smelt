#include <metal_stdlib>
using namespace metal;

/// Dense matrix transpose: src is row-major [rows, cols], dst is row-major [cols, rows], with
/// dst[c, r] = src[r, c]. The codec's CT<->TC layout bridge between the conv stages (channel-major
/// [C, T]) and the pre_transformer (time-major [T, C]); replaces the hand `transposeF32` CPU helper
/// (a `.contents()` readback + CPU loop) so the codec graph stays on-GPU through a record table.
/// Pure data movement -> bit-exact with the CPU transpose.
///
/// Buffers: 0 src float [rows, cols], 1 dst float [cols, rows]
/// Constants: 2 rows, 3 cols
/// Dispatch: grid (cols, rows, 1) — thread (c, r) writes dst[c * rows + r] = src[r * cols + c].
kernel void transpose_f32(
    device const float* src  [[buffer(0)]],
    device float*       dst  [[buffer(1)]],
    constant uint&      rows [[buffer(2)]],
    constant uint&      cols [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint c = tid.x, r = tid.y;
    if (r >= rows || c >= cols) return;
    dst[c * rows + r] = src[r * cols + c];
}
