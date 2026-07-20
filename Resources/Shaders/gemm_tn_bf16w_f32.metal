#include <metal_stdlib>
using namespace metal;

/// Direct multi-N batched GEMV, bf16 weights (codex-prescribed shape, the measured winner of the
/// M>1 family): a (32 × TG_M) threadgroup holds TG_M row-SIMDs; each SIMD computes GEMM_TN output
/// features for its row, reading each x chunk ONCE and TN weight chunks per step — no threadgroup
/// memory, no barriers. Attacks the threadgroup-count wall directly (gemm_tg at TN=1 was ~1.15M
/// tiny TGs across a 30-token prefill = launch-granularity-bound at ~54GB/s): TG count divides by
/// TN, work per TG multiplies by TN, and the TG's row-SIMDs read the same W addresses together
/// (cache-deduped). Weight traffic stays ceil(M/TG_M)× (the gemm_mt/tg dedup).
/// BIT-EXACT per output (n,m) vs gemv/gemm: lane owns chunks c = lane + 32t, t increasing, one
/// lane-local accumulator per n, one simd_sum. bf16→float is an exact widen.
///
/// Buffers: 0 x [M,K] float, 1 W [N,K] bfloat, 2 bias [N] float, 3 out [M,N] float
/// Constants: 4 M, 5 N, 6 K, 7 has_bias
/// Dispatch: grid (ceil(N/GEMM_TN)·32, ceil(M/TG_M)·TG_M, 1), threadsPerThreadgroup (32, TG_M, 1).
/// K % 4 == 0. GEMM_TN / GEMM_TN_TGM must match `gemmTNFeatures` / `gemmTNTileM` in
/// Qwen3TTSGPUCodec.swift.
#define GEMM_TN 4
#define GEMM_TN_TGM 16

kernel void gemm_tn_bf16w_f32(
    device const float*  x        [[buffer(0)]],
    device const bfloat* W        [[buffer(1)]],
    device const float*  bias     [[buffer(2)]],
    device float*        out      [[buffer(3)]],
    constant uint&       M        [[buffer(4)]],
    constant uint&       N        [[buffer(5)]],
    constant uint&       K        [[buffer(6)]],
    constant uint&       has_bias [[buffer(7)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]]
) {
    uint lane = lid.x;
    uint m    = tg.y * GEMM_TN_TGM + lid.y;
    uint n0   = tg.x * GEMM_TN;
    if (m >= M || n0 >= N) return;   // whole SIMD exits together; no barriers in this kernel
    uint chunks = K >> 2;
    device const float4*  x4 = (device const float4*)(x + m * K);
    device const bfloat4* W4 = (device const bfloat4*)W;
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
