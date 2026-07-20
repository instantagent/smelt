#include <metal_stdlib>
using namespace metal;

/// Multi-row gather: out[t, :] = table[ids[t], :] for t in [0, n), reading the per-row index from an
/// ids slot (the TTS front-end's host-resolved text/codec ids — the ids-from-slot generalization of the
/// single-row gather_row_f32, which the LLM uses for the next-MTP-codebook gather). `ids` is SIGNED:
/// id < 0 ⇒ write a ZERO row (the projection-only codec sentinel, so scale_residual_tc(proj + 0) = proj,
/// bit-identical to the host front-end which leaves a nil-codec row's contribution zero). A valid id is
/// host-prechecked to 0 <= id < rows before dispatch (a baked table can't precheck), so the kernel never
/// silently clamps an out-of-range index — the only special case is the id < 0 zero sentinel.
/// One thread per output element (flat grid = n*dim).
///
/// Buffers: 0 table [rows,dim] float, 1 ids [n] int, 2 out [n,dim] float
/// Constants: 3 n, 4 dim
kernel void gather_rows_f32(
    device const float* table [[buffer(0)]],
    device const int*   ids   [[buffer(1)]],
    device float*       out   [[buffer(2)]],
    constant uint&      n     [[buffer(3)]],
    constant uint&      dim   [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n * dim) return;
    uint t = gid / dim, d = gid % dim;
    int id = ids[t];
    out[gid] = (id < 0) ? 0.0f : table[uint(id) * dim + d];
}
