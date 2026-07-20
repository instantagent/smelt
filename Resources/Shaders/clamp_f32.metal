#include <metal_stdlib>
using namespace metal;

/// Elementwise clamp to [lo, hi], fp32. The codec final waveform clamp to [-1, 1].
/// One thread per element. Buffers 0 input, 1 output. Constants 2 count, 3 lo, 4 hi.
kernel void clamp_f32(
    device const float* input  [[buffer(0)]],
    device float*       output [[buffer(1)]],
    constant uint&      count  [[buffer(2)]],
    constant float&     lo     [[buffer(3)]],
    constant float&     hi     [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    output[tid] = clamp(input[tid], lo, hi);
}
