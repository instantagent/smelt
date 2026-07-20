#include <metal_stdlib>
using namespace metal;

/// f16-weight variant of the direct multi-N batched GEMV gemm_tn_bf16w_f32 — see that file for
/// the shape rationale. BIT-EXACT per output (n,m) vs gemv_f16w_f32/gemm_f16w_f32 (same
/// lane-strided chunk order, one accumulator per n, one simd_sum). f16→float is an exact widen.
///
/// Buffers: 0 x [M,K] float, 1 W [N,K] half, 2 bias [N] float, 3 out [M,N] float
/// Constants: 4 M, 5 N, 6 K, 7 has_bias
/// Dispatch: grid (ceil(N/GEMM_TN)·32, ceil(M/TG_M)·TG_M, 1), threadsPerThreadgroup (32, TG_M, 1).
/// K % 4 == 0. GEMM_TN / GEMM_TN_TGM must match `gemmTNFeatures` / `gemmTNTileM` in
/// Qwen3TTSGPUCodec.swift.
#define GEMM_TN 4
#define GEMM_TN_TGM 16

kernel void gemm_tn_f16w_f32(
    device const float* x        [[buffer(0)]],
    device const half*  W        [[buffer(1)]],
    device const float* bias     [[buffer(2)]],
    device float*       out      [[buffer(3)]],
    constant uint&      M        [[buffer(4)]],
    constant uint&      N        [[buffer(5)]],
    constant uint&      K        [[buffer(6)]],
    constant uint&      has_bias [[buffer(7)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]]
) {
    uint lane = lid.x;
    uint m    = tg.y * GEMM_TN_TGM + lid.y;
    uint n0   = tg.x * GEMM_TN;
    if (m >= M || n0 >= N) return;   // whole SIMD exits together; no barriers in this kernel
    uint chunks = K >> 2;
    device const float4* x4 = (device const float4*)(x + m * K);
    device const half4*  W4 = (device const half4*)W;
    float acc[GEMM_TN];
    #pragma unroll
    for (uint nn = 0; nn < GEMM_TN; ++nn) acc[nn] = 0.0f;
    for (uint c = lane; c < chunks; c += 32) {
        float4 xv = x4[c];
        #pragma unroll
        for (uint nn = 0; nn < GEMM_TN; ++nn) {
            uint n = n0 + nn;
            if (n < N) acc[nn] += dot(xv, float4(W4[n * chunks + c]));
        }
    }
    #pragma unroll
    for (uint nn = 0; nn < GEMM_TN; ++nn) {
        uint n = n0 + nn;
        float total = simd_sum(acc[nn]);
        if (lane == 0 && n < N) out[m * N + n] = total + (has_bias != 0 ? bias[n] : 0.0f);
    }
}
