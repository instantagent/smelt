#include <metal_stdlib>
using namespace metal;

/// Coalesced GEMV for M=1: out[n] = x[K] · W[n][K] + b[n]. One SIMD (32 threads) per output
/// feature n; each lane sums K/32 contiguous float4 chunks (coalesced + vectorized), then a single
/// `simd_sum` reduces the 32 partials — NO threadgroup barriers. For the M=1 decode matmuls that
/// dominate the talker/MTP, vs matmul_f32's strided one-thread-per-output (bandwidth-inefficient).
/// K is a multiple of 4 for every talker/MTP projection. Dispatch with threadsPerThreadgroup = 32.
///
/// Buffers: 0 x [K] float, 1 W [N,K] float, 2 bias [N] float, 3 out [N] float
/// Constants: 4 M(=1), 5 N, 6 K, 7 has_bias
kernel void gemv_f32(
    device const float* x        [[buffer(0)]],
    device const float* W        [[buffer(1)]],
    device const float* bias     [[buffer(2)]],
    device float*       out      [[buffer(3)]],
    constant uint&      M        [[buffer(4)]],
    constant uint&      N        [[buffer(5)]],
    constant uint&      K        [[buffer(6)]],
    constant uint&      has_bias [[buffer(7)]],
    uint n   [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (n >= N) return;
    uint chunks = K >> 2;
    device const float4* x4 = (device const float4*)x;
    device const float4* W4 = (device const float4*)(W + n * K);
    float partial = 0.0f;
    for (uint c = lane; c < chunks; c += 32) partial += dot(x4[c], W4[c]);
    float total = simd_sum(partial);
    if (lane == 0) out[n] = total + (has_bias != 0 ? bias[n] : 0.0f);
}
