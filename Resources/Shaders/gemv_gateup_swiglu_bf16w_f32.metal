#include <metal_stdlib>
using namespace metal;

/// Fused gate/up projection + SwiGLU GEMV, bf16 weights (M=1 decode/MTP step): gate matmul, up
/// matmul, and the elementwise SwiGLU were three dispatches per layer — part of the ~1200-tiny-
/// dispatch decode/MTP wall. One SIMD per intermediate feature r computes BOTH dots (each the
/// SAME lane-strided reduction as gemv_bf16w_f32 on that row → bit-identical g and u), then lane 0
/// applies swiglu_f32's exact expression `(g / (1 + exp(-g))) * u` — identical bytes out, one
/// dispatch instead of three. No biases (the talker/MTP MLPs have none).
///
/// Buffers: 0 x [K] float, 1 Wg [N,K] bfloat, 2 Wu [N,K] bfloat, 3 act [N] float
/// Constants: 4 N, 5 K
/// Dispatch: grid (N·32, 1, 1), threadsPerThreadgroup (32, 1, 1). K % 4 == 0.
kernel void gemv_gateup_swiglu_bf16w_f32(
    device const float*  x   [[buffer(0)]],
    device const bfloat* Wg  [[buffer(1)]],
    device const bfloat* Wu  [[buffer(2)]],
    device float*        act [[buffer(3)]],
    constant uint&       N   [[buffer(4)]],
    constant uint&       K   [[buffer(5)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]
) {
    if (row >= N) return;
    uint chunks = K >> 2;
    device const float4*  x4 = (device const float4*)x;
    device const bfloat4* G4 = (device const bfloat4*)(Wg + row * K);
    device const bfloat4* U4 = (device const bfloat4*)(Wu + row * K);
    float pg = 0.0f, pu = 0.0f;
    for (uint c = lane; c < chunks; c += 32) {
        float4 xv = x4[c];
        pg += dot(xv, float4(G4[c]));
        pu += dot(xv, float4(U4[c]));
    }
    float g = simd_sum(pg);
    float u = simd_sum(pu);
    if (lane == 0) act[row] = (g / (1.0f + exp(-g))) * u;
}
