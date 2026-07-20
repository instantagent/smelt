#include <metal_stdlib>
using namespace metal;

/// Per-channel strided copy: dst[c, dstOff + l] = src[c, srcOff + l] for l in [0, copyLen), over
/// `channels` rows with independent src/dst row strides. The destination must be pre-zeroed (outF32),
/// so this serves the codec's causal-conv LEFT-pad (dstLen = paddedLen, dstOff = leftPad, the zeros
/// come from the memset) and the transposed-conv RIGHT-trim (dstLen < srcLen, copy the kept prefix) —
/// both previously done with a CPU `.contents()` read of the prior kernel's GPU output, which forced a
/// per-conv sync and blocked command-buffer batching. On-GPU, the pad/trim chains in the command stream.
/// Pure data movement → bit-exact.
///
/// Buffers: 0 src float, 1 dst float
/// Constants: 2 channels, 3 srcLen, 4 dstLen, 5 srcOff, 6 dstOff, 7 copyLen
/// Dispatch: grid (copyLen, channels, 1).
kernel void channel_copy_f32(
    device const float* src     [[buffer(0)]],
    device float*       dst     [[buffer(1)]],
    constant uint&      channels [[buffer(2)]],
    constant uint&      srcLen  [[buffer(3)]],
    constant uint&      dstLen  [[buffer(4)]],
    constant uint&      srcOff  [[buffer(5)]],
    constant uint&      dstOff  [[buffer(6)]],
    constant uint&      copyLen [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint l = tid.x, c = tid.y;
    if (c >= channels || l >= copyLen) return;
    dst[c * dstLen + dstOff + l] = src[c * srcLen + srcOff + l];
}
