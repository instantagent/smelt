#include <metal_stdlib>
using namespace metal;

/// GEMV + residual-add epilogue, bf16 weights (M=1 decode/MTP step): out[n] = x·W[n] + res[n] —
/// the o_proj/down_proj matmul and the following residual add in one dispatch (2 → 1). The dot
/// is gemv_bf16w_f32's exact reduction; fp32 addition is bit-commutative, so `total + res[n]`
/// equals the standalone residual kernel's add regardless of operand order. No bias (the
/// talker/MTP o/down projections have none).
///
/// Buffers: 0 x [K] float, 1 W [N,K] bfloat, 2 res [N] float, 3 out [N] float
/// Constants: 4 N, 5 K
/// Dispatch: grid (N·32, 1, 1), threadsPerThreadgroup (32, 1, 1). K % 4 == 0.
kernel void gemv_add_bf16w_f32(
    device const float*  x   [[buffer(0)]],
    device const bfloat* W   [[buffer(1)]],
    device const float*  res [[buffer(2)]],
    device float*        out [[buffer(3)]],
    constant uint&       N   [[buffer(4)]],
    constant uint&       K   [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (row >= N) return;
    uint chunks = K >> 2;
    device const float4*  x4 = (device const float4*)x;
    device const bfloat4* W4 = (device const bfloat4*)(W + row * K);
    float acc = 0.0f;
    for (uint c = lane; c < chunks; c += 32) acc += dot(x4[c], float4(W4[c]));
    float total = simd_sum(acc);
    if (lane == 0) out[row] = total + res[row];
}
