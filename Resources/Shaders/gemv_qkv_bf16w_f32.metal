#include <metal_stdlib>
using namespace metal;

/// Fused q/k/v projection GEMV, bf16 weights (M=1 decode/MTP step): the three per-layer
/// projections share one input row and were three separate dispatches — at ~1200 tiny dispatches
/// per generated frame the per-dispatch floor IS the decode/MTP wall (codex-confirmed; batching
/// already removed all inter-dispatch gaps). One dispatch covers all Nq+Nk+Nv output rows; each
/// row is the SAME one-SIMD gemv reduction as gemv_bf16w_f32 (lane-strided float4 chunks,
/// sequential accumulate, one simd_sum) → bit-identical outputs. No biases (the talker/MTP
/// q/k/v have none). v writes at an offset (the KV-cache row) via the buffer binding.
///
/// Buffers: 0 x [K] float, 1 Wq [Nq,K] bfloat, 2 Wk [Nk,K] bfloat, 3 Wv [Nv,K] bfloat,
///          4 outQ [Nq] float, 5 outK [Nk] float, 6 outV [Nv] float
/// Constants: 7 Nq, 8 Nk, 9 Nv, 10 K
/// Dispatch: grid ((Nq+Nk+Nv)·32, 1, 1), threadsPerThreadgroup (32, 1, 1). K % 4 == 0.
kernel void gemv_qkv_bf16w_f32(
    device const float*  x    [[buffer(0)]],
    device const bfloat* Wq   [[buffer(1)]],
    device const bfloat* Wk   [[buffer(2)]],
    device const bfloat* Wv   [[buffer(3)]],
    device float*        outQ [[buffer(4)]],
    device float*        outK [[buffer(5)]],
    device float*        outV [[buffer(6)]],
    constant uint&       Nq   [[buffer(7)]],
    constant uint&       Nk   [[buffer(8)]],
    constant uint&       Nv   [[buffer(9)]],
    constant uint&       K    [[buffer(10)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (row >= Nq + Nk + Nv) return;
    device const bfloat* W;
    device float* out;
    uint n;
    if (row < Nq)           { W = Wq; out = outQ; n = row; }
    else if (row < Nq + Nk) { W = Wk; out = outK; n = row - Nq; }
    else                    { W = Wv; out = outV; n = row - Nq - Nk; }
    uint chunks = K >> 2;
    device const float4*  x4 = (device const float4*)x;
    device const bfloat4* W4 = (device const bfloat4*)(W + n * K);
    float partial = 0.0f;
    for (uint c = lane; c < chunks; c += 32) partial += dot(x4[c], float4(W4[c]));
    float total = simd_sum(partial);
    if (lane == 0) out[n] = total;
}
