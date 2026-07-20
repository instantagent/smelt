#include <metal_stdlib>
using namespace metal;

/// Apply RoPE (rotate_half) to a [frames, heads*headDim] buffer using cos/sin tables
/// [frames, headDim] in cat(freqs,freqs) layout. Matches the per-head RoPE in
/// Qwen3TTSCodec.preTransformer.
///
/// ONE THREAD PER (frame, head, pair j): RoPE has no reduction, so the maximal-occupancy form is one
/// thread per rotated pair — each computes out[base+j] and out[base+j+halfDim]. The prior
/// one-thread-per-(frame,head) form ran a single thread per head at decode (frames=1) looping halfDim
/// with unhidden latency. Flattened grid id = (t*heads + hd)*halfDim + j; grid = frames*heads*halfDim.
///
/// Buffers: 0 x, 1 cos [frames,headDim], 2 sin [frames,headDim], 3 out
/// Constants: 4 frames, 5 heads, 6 headDim
kernel void rope_apply_f32(
    device const float* x       [[buffer(0)]],
    device const float* cosT    [[buffer(1)]],
    device const float* sinT    [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      frames  [[buffer(4)]],
    constant uint&      heads   [[buffer(5)]],
    constant uint&      headDim [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint halfDim = headDim / 2;
    if (gid >= frames * heads * halfDim) return;
    uint j = gid % halfDim;
    uint rest = gid / halfDim;
    uint hd = rest % heads;
    uint t = rest / heads;
    uint base = t * heads * headDim + hd * headDim;
    uint cb = t * headDim;

    float a = x[base + j], b = x[base + j + halfDim];
    out[base + j]           = a * cosT[cb + j]           - b * sinT[cb + j];
    out[base + j + halfDim] = b * cosT[cb + j + halfDim] + a * sinT[cb + j + halfDim];
}
