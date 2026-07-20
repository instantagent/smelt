#include <metal_stdlib>
using namespace metal;

/// Dense fp32 matmul matching nn.Linear: y[M,N] = x[M,K] · Wᵀ + b, with W row-major
/// [N,K] (out-features × in-features). One thread per output element (naive; tiling is a
/// later perf concern). Used by the codec pre_transformer (q/k/v/o, MLP), ConvNeXt
/// pointwise (1×1 conv == matmul), and RVQ output_proj. Matches Qwen3TTSTalker.linearBias.
///
/// Buffers:
///   0: x      [M, K]   float
///   1: W      [N, K]   float
///   2: bias   [N]      float
///   3: out    [M, N]   float
/// Constants:
///   4: M, 5: N, 6: K, 7: has_bias (uint; 0 = no bias)
kernel void matmul_f32(
    device const float* x        [[buffer(0)]],
    device const float* W        [[buffer(1)]],
    device const float* bias     [[buffer(2)]],
    device float*       out      [[buffer(3)]],
    constant uint&      M        [[buffer(4)]],
    constant uint&      N        [[buffer(5)]],
    constant uint&      K        [[buffer(6)]],
    constant uint&      has_bias [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]]
) {
    uint n = tid.x;  // output feature
    uint m = tid.y;  // row
    if (m >= M || n >= N) return;

    float acc = (has_bias != 0) ? bias[n] : 0.0f;
    uint xb = m * K, wb = n * K;
    for (uint k = 0; k < K; k++) acc += x[xb + k] * W[wb + k];
    out[m * N + n] = acc;
}
