#include <metal_stdlib>

using namespace metal;

// Match MLX nn.silu(gate) * up for fp16 graphs. MLX's sigmoid primitive
// returns the input dtype, nn.silu materializes that fp16 activation, and the
// outer multiply rounds to fp16 again. Safe-math compilation preserves those
// graph-owned boundaries inside this single dispatch.
kernel void swiglu_fused(
    device const half *gate [[buffer(0)]],
    device const half *up [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const half g = gate[tid];
    const half tail = 1.0h / (1.0h + exp(abs(g)));
    const half sigmoid = g < 0.0h ? tail : 1.0h - tail;
    const half activated = g * sigmoid;
    output[tid] = activated * up[tid];
}

// Match the two fp16 graph nodes `attention * sigmoid(gate)`: the sigmoid
// result is materialized as fp16 before the outer fp16 multiply.
kernel void sigmoid_mul(
    device const half *attention [[buffer(0)]],
    device const half *gate [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    const half g = gate[tid];
    const half tail = 1.0h / (1.0h + exp(abs(g)));
    const half sigmoid = g < 0.0h ? tail : 1.0h - tail;
    output[tid] = attention[tid] * sigmoid;
}
