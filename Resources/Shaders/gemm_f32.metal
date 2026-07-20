#include <metal_stdlib>
using namespace metal;

/// Coalesced batched GEMV for the M>1 prefill: out[M,N] = x[M,K] · W[N,K]ᵀ + b[N]. One SIMD (32
/// lanes) per output element (n,m); each lane strides K contiguously (coalesced weight + activation
/// reads) and a single simd_sum reduces — exactly gemv_f32 generalized with an M grid dimension.
///
/// vs matmul_f32's one-thread-per-output (strided W[n*K+k] reads, uncoalesced): the prefill measured
/// ~53 ms/token, ~2× the decode forward's coalesced 27 ms/token — pure coalescing loss. This reads
/// the weight M× (once per row) like the naive kernel, but coalesced (the machine's 333 GB/s vs the
/// strided ~150), keeping the naive kernel's high occupancy (N×M SIMDs) — which the weight-amortizing
/// tiled GEMM sacrificed to barriers. Per-(n,m) reduction is bit-identical to gemv_f32 on that row →
/// identical numerics to the decode path, so codes==gen_codes holds. K a multiple of 4.
///
/// Buffers: 0 x [M,K] float, 1 W [N,K] float, 2 bias [N] float, 3 out [M,N] float
/// Constants: 4 M, 5 N, 6 K, 7 has_bias
/// Dispatch: grid (N·32, M, 1), threadsPerThreadgroup (32, 1, 1).

kernel void gemm_f32(
    device const float* x        [[buffer(0)]],
    device const float* W        [[buffer(1)]],
    device const float* bias     [[buffer(2)]],
    device float*       out      [[buffer(3)]],
    constant uint&      M        [[buffer(4)]],
    constant uint&      N        [[buffer(5)]],
    constant uint&      K        [[buffer(6)]],
    constant uint&      has_bias [[buffer(7)]],
    uint2 gid  [[threadgroup_position_in_grid]],   // (n, m)
    uint  lane [[thread_index_in_threadgroup]]
) {
    uint n = gid.x, m = gid.y;
    if (n >= N || m >= M) return;
    uint chunks = K >> 2;
    device const float4* x4 = (device const float4*)(x + m * K);
    device const float4* W4 = (device const float4*)(W + n * K);
    float partial = 0.0f;
    for (uint c = lane; c < chunks; c += 32) partial += dot(x4[c], W4[c]);
    float total = simd_sum(partial);
    if (lane == 0) out[m * N + n] = total + (has_bias != 0 ? bias[n] : 0.0f);
}
