#include <metal_stdlib>
using namespace metal;

/// int4-weight variant of the coalesced batched-GEMV gemm_f32 (M>1 prefill). Same group-wise affine
/// dequant as gemv_u4_f32 (packed nibbles + per-group fp16 scale/bias staged into threadgroup memory),
/// with an added M grid dim: one SIMD per (n,m). The per-(n,m) reduction matches gemv_u4_f32 on row n.
/// Same CONTRACT as gemv_u4_f32 (enforced by the builder):
/// K and group_size multiples of 4, group_size ≥ 4, ceil(K/group_size) ≤ U4_MAX_GROUPS.
///
/// Buffers: 0 x [M,K] float, 1 Wq [N,K/2] uchar, 2 scales [N,G] half, 3 biases [N,G] half,
///          4 bias [N] float, 5 out [M,N] float
/// Constants: 6 M, 7 N, 8 K, 9 has_bias, 10 group_size
/// Dispatch: grid (N·32, M, 1), threadsPerThreadgroup (32, 1, 1).
constant uint U4_MAX_GROUPS = 256;

kernel void gemm_u4_f32(
    device const float* x          [[buffer(0)]],
    device const uchar* Wq         [[buffer(1)]],
    device const half*  scales     [[buffer(2)]],
    device const half*  biases     [[buffer(3)]],
    device const float* bias       [[buffer(4)]],
    device float*       out        [[buffer(5)]],
    constant uint&      M          [[buffer(6)]],
    constant uint&      N          [[buffer(7)]],
    constant uint&      K          [[buffer(8)]],
    constant uint&      has_bias   [[buffer(9)]],
    constant uint&      group_size [[buffer(10)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]]
) {
    uint n = gid.x, m = gid.y;
    if (n >= N || m >= M) return;
    uint groups = (K + group_size - 1) / group_size;
    threadgroup float sScale[U4_MAX_GROUPS];
    threadgroup float sBias[U4_MAX_GROUPS];
    for (uint i = lane; i < groups; i += 32) {
        sScale[i] = float(scales[n * groups + i]);
        sBias[i]  = float(biases[n * groups + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint chunks = K >> 2;
    uint chunksPerGroup = group_size >> 2;
    device const float4* x4 = (device const float4*)(x + m * K);
    device const uchar2* w2 = (device const uchar2*)(Wq + n * (K >> 1));
    float partial = 0.0f;
    for (uint c = lane; c < chunks; c += 32) {
        uchar2 b = w2[c];
        float4 nib = float4(float(b.x & 0xF), float(b.x >> 4), float(b.y & 0xF), float(b.y >> 4));
        uint g = c / chunksPerGroup;
        float4 xc = x4[c];
        partial += sScale[g] * dot(xc, nib) + sBias[g] * (xc.x + xc.y + xc.z + xc.w);
    }
    float total = simd_sum(partial);
    if (lane == 0) out[m * N + n] = total + (has_bias != 0 ? bias[n] : 0.0f);
}
