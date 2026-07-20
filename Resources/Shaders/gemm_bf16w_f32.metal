#include <metal_stdlib>
using namespace metal;

/// bf16-weight variant of the coalesced batched-GEMV gemm_f32 (M>1 prefill): W is bf16, activations +
/// accumulate stay fp32. SAME layout/dispatch as gemm_f16w_f32 — only W's element type (bfloat4 chunks)
/// differs. bf16→float is an exact widen, so the per-(n,m) reduction matches gemv_bf16w_f32 on that row.
///
/// Buffers: 0 x [M,K] float, 1 W [N,K] bfloat, 2 bias [N] float, 3 out [M,N] float
/// Constants: 4 M, 5 N, 6 K, 7 has_bias
/// Dispatch: grid (N·32, M, 1), threadsPerThreadgroup (32, 1, 1).
kernel void gemm_bf16w_f32(
    device const float*  x        [[buffer(0)]],
    device const bfloat* W        [[buffer(1)]],
    device const float*  bias     [[buffer(2)]],
    device float*        out      [[buffer(3)]],
    constant uint&       M        [[buffer(4)]],
    constant uint&       N        [[buffer(5)]],
    constant uint&       K        [[buffer(6)]],
    constant uint&       has_bias [[buffer(7)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]]
) {
    uint n = gid.x, m = gid.y;
    if (n >= N || m >= M) return;
    uint chunks = K >> 2;
    device const float4*  x4 = (device const float4*)(x + m * K);
    device const bfloat4* W4 = (device const bfloat4*)(W + n * K);
    float partial = 0.0f;
    for (uint c = lane; c < chunks; c += 32) partial += dot(x4[c], float4(W4[c]));
    float total = simd_sum(partial);
    if (lane == 0) out[m * N + n] = total + (has_bias != 0 ? bias[n] : 0.0f);
}
