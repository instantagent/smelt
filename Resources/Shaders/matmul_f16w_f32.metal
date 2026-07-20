#include <metal_stdlib>
using namespace metal;

/// fp16-weight × fp32-activation matmul, fp32 accumulate: y[M,N] = x[M,K] · Wᵀ + b, with W
/// row-major [N,K] stored as `half` (out-features × in-features). Identical to matmul_f32 except
/// W is fp16 — this halves the weight-read bandwidth that dominates the M=1 decode matmuls (once
/// the model is KV-cached + batched). Accumulation stays fp32, so only the weight loses precision
/// (~5e-4 relative); activations remain fp32. One thread per output element.
///
/// Buffers:
///   0: x      [M, K]   float
///   1: W      [N, K]   half
///   2: bias   [N]      float
///   3: out    [M, N]   float
/// Constants:
///   4: M, 5: N, 6: K, 7: has_bias (uint; 0 = no bias)
kernel void matmul_f16w_f32(
    device const float* x        [[buffer(0)]],
    device const half*  W        [[buffer(1)]],
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
    for (uint k = 0; k < K; k++) acc += x[xb + k] * float(W[wb + k]);
    out[m * N + n] = acc;
}
