#include <metal_stdlib>
using namespace metal;

/// bf16-weight coalesced GEMV for M=1: out[n] = x[K] · W[n][K] + b[n]. Identical to gemv_f16w_f32
/// except W is bf16 (the source checkpoint's native dtype). bf16→float is an EXACT widen (bf16 is the
/// top 16 bits of an fp32, mantissa zero-padded), so `float4(bfloat4)` reproduces the same fp32 value
/// the old fp32-storage path fed the gemv → bit-identical result at HALF the weight footprint.
/// Activations + accumulation stay fp32. K is a multiple of 4. Dispatch threadsPerThreadgroup = 32.
///
/// Buffers: 0 x [K] float, 1 W [N,K] bfloat, 2 bias [N] float, 3 out [N] float
/// Constants: 4 M(=1), 5 N, 6 K, 7 has_bias
kernel void gemv_bf16w_f32(
    device const float*  x        [[buffer(0)]],
    device const bfloat* W        [[buffer(1)]],
    device const float*  bias     [[buffer(2)]],
    device float*        out      [[buffer(3)]],
    constant uint&       M        [[buffer(4)]],
    constant uint&       N        [[buffer(5)]],
    constant uint&       K        [[buffer(6)]],
    constant uint&       has_bias [[buffer(7)]],
    uint n    [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (n >= N) return;
    uint chunks = K >> 2;
    device const float4*  x4 = (device const float4*)x;
    device const bfloat4* W4 = (device const bfloat4*)(W + n * K);
    float partial = 0.0f;
    for (uint c = lane; c < chunks; c += 32) partial += dot(x4[c], float4(W4[c]));
    float total = simd_sum(partial);
    if (lane == 0) out[n] = total + (has_bias != 0 ? bias[n] : 0.0f);
}
