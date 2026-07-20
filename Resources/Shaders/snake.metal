#include <metal_stdlib>
using namespace metal;

/// Snake activation: y = x + sin(alpha * x)^2 / alpha
/// Per-channel learnable alpha parameter.
///
/// Buffers:
///   0: input   [channels, length]  FP16
///   1: alpha   [channels]          FP16 (learnable per-channel parameter)
///   2: output  [channels, length]  FP16
/// Constants:
///   0: channels  uint
///   1: length    uint
kernel void snake_activation(
    device const half* input   [[buffer(0)]],
    device const half* alpha   [[buffer(1)]],
    device half*       output  [[buffer(2)]],
    constant uint&     channels [[buffer(3)]],
    constant uint&     length   [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint l = tid.x;  // position (contiguous in memory for coalesced access)
    uint c = tid.y;  // channel

    if (c >= channels || l >= length) return;

    uint idx = c * length + l;
    float x = float(input[idx]);
    float a = float(alpha[c]);

    // snake(x) = x + sin(a*x)^2 / a
    float s = sin(a * x);
    float y = x + (s * s) / a;

    output[idx] = half(y);
}
