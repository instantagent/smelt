#include <metal_stdlib>
using namespace metal;

/// Coalesced GEMV for M=1 with group-wise affine int4 weights:
///   out[n] = Σ_k x[k]·(nibble(n,k)·scale[n][g] + bias[n][g]),  g = k/group_size
/// One SIMD (32 lanes) per output feature n. Weights are packed nibbles (2 per byte, LOW nibble =
/// even column); per-group fp16 scale/bias are staged into threadgroup memory ONCE per row, then each
/// lane sweeps coalesced float4 chunks (4 columns = 2 packed bytes) indexing the staged scale/bias.
/// Staging is the point: a strided lane crosses a new group every chunk, so reading scale/bias from
/// device per chunk would cost more bytes than the nibbles — staging makes it a threadgroup-memory read.
/// Per chunk the dequant identity Σ x·(q·s+b) = s·dot(x,q) + b·Σx holds because group_size is a
/// multiple of 4 (a 4-column chunk never straddles a group). Accumulate in fp32; simd_sum reduces.
/// Dispatch threadsPerThreadgroup = 32. CONTRACT enforced by the packer/dispatch site (the builder),
/// not here: K and group_size are multiples of 4, group_size ≥ 4, and ceil(K/group_size) ≤ U4_MAX_GROUPS.
///
/// Buffers: 0 x [K] float, 1 Wq [N,K/2] uchar, 2 scales [N,G] half, 3 biases [N,G] half,
///          4 bias [N] float, 5 out [N] float
/// Constants: 6 M(=1), 7 N, 8 K, 9 has_bias, 10 group_size
constant uint U4_MAX_GROUPS = 256;

kernel void gemv_u4_f32(
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
    uint n    [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (n >= N) return;
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
    device const float4* x4 = (device const float4*)x;
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
    if (lane == 0) out[n] = total + (has_bias != 0 ? bias[n] : 0.0f);
}
