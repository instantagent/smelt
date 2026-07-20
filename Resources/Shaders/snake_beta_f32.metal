#include <metal_stdlib>
using namespace metal;

/// SnakeBeta activation (fp32): y = x + (1 / (exp(beta) + 1e-9)) * sin(x * exp(alpha))^2
/// Per-channel learnable alpha, beta. Matches Qwen3TTSCodec.snakeBeta (codec is fp32).
///
/// Buffers:
///   0: input   [channels, length]  float  (channel-major, length contiguous)
///   1: alpha   [channels]          float
///   2: beta    [channels]          float
///   3: output  [channels, length]  float
/// Constants:
///   4: channels  uint
///   5: length    uint
kernel void snake_beta_f32(
    device const float* input    [[buffer(0)]],
    device const float* alpha    [[buffer(1)]],
    device const float* beta     [[buffer(2)]],
    device float*       output   [[buffer(3)]],
    constant uint&      channels [[buffer(4)]],
    constant uint&      length   [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint l = tid.x;  // position (contiguous in memory)
    uint c = tid.y;  // channel

    if (c >= channels || l >= length) return;

    uint idx = c * length + l;
    float x = input[idx];
    float a = exp(alpha[c]);
    float invB = 1.0f / (exp(beta[c]) + 1e-9f);

    float s = sin(x * a);
    output[idx] = x + invB * s * s;
}
