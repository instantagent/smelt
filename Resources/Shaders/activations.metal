#include <metal_stdlib>
using namespace metal;

// ─── SiLU (swish): x * sigmoid(x) ───
kernel void silu(
    device const half* input  [[buffer(0)]],
    device half*       output [[buffer(1)]],
    constant uint&     count  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        float x = float(input[tid]);
        output[tid] = half(x / (1.0f + exp(-x)));
    }
}

// ─── Sigmoid ───
kernel void sigmoid_kernel(
    device const half* input  [[buffer(0)]],
    device half*       output [[buffer(1)]],
    constant uint&     count  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        float x = float(input[tid]);
        output[tid] = half(1.0f / (1.0f + exp(-x)));
    }
}

// ─── Softplus: log(1 + exp(x)) ───
kernel void softplus(
    device const half* input  [[buffer(0)]],
    device half*       output [[buffer(1)]],
    constant uint&     count  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        float x = float(input[tid]);
        output[tid] = half((x > 20.0f) ? x : log(1.0f + exp(x)));
    }
}

// ─── Buffer copy: out = src ───
kernel void buffer_copy(
    device const half* src    [[buffer(0)]],
    device half*       dst    [[buffer(1)]],
    constant uint&     count  [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        dst[tid] = src[tid];
    }
}

// ─── Elementwise add: out = a + b ───
kernel void elementwise_add(
    device const half* inputA [[buffer(0)]],
    device const half* inputB [[buffer(1)]],
    device half*       output [[buffer(2)]],
    constant uint&     count  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        output[tid] = inputA[tid] + inputB[tid];
    }
}

// --- Batched projection bias add: out[b,row] = input[b,row] + bias[row] ---
kernel void projection_bias_add_batched(
    device const half* input [[buffer(0)]],
    device const half* bias  [[buffer(1)]],
    device half*       output [[buffer(2)]],
    constant uint&     rows  [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.x;
    uint batch = gid.y;
    if (row < rows) {
        uint index = batch * rows + row;
        output[index] = input[index] + bias[row];
    }
}

// ─── Elementwise multiply: out = a * b ───
kernel void elementwise_mul(
    device const half* inputA [[buffer(0)]],
    device const half* inputB [[buffer(1)]],
    device half*       output [[buffer(2)]],
    constant uint&     count  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        output[tid] = inputA[tid] * inputB[tid];
    }
}

// ─── Scalar multiply: out = a * scalar ───
kernel void scalar_mul(
    device const half* input  [[buffer(0)]],
    device half*       output [[buffer(1)]],
    constant float&    scalar [[buffer(2)]],
    constant uint&     count  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        output[tid] = half(float(input[tid]) * scalar);
    }
}

// ─── Scalar multiply from a packed half weight: out = a * scalar[0] ───
kernel void scalar_mul_weight(
    device const half* input  [[buffer(0)]],
    device const half* scalar [[buffer(1)]],
    device half*       output [[buffer(2)]],
    constant uint&     count  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        output[tid] = half(float(input[tid]) * float(scalar[0]));
    }
}

// ─── Fused GeGLU: out = gelu(gate) * up ───
kernel void geglu_fused(
    device const half* gate   [[buffer(0)]],  // [D] from gate_proj
    device const half* up     [[buffer(1)]],  // [D] from up_proj
    device half*       output [[buffer(2)]],  // [D]
    constant uint&     count  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        float g = float(gate[tid]);
        float u = float(up[tid]);
        float g3 = g * g * g;
        float inner = 0.7978845608f * (g + 0.044715f * g3);
        inner = clamp(inner, -20.0f, 20.0f);
        float gelu_g = 0.5f * g * (1.0f + tanh(inner));
        output[tid] = half(clamp(gelu_g * u, -65504.0f, 65504.0f));
    }
}

kernel void geglu_fused_strided_batched(
    device const half* gate     [[buffer(0)]],
    device const half* up       [[buffer(1)]],
    device half*       output   [[buffer(2)]],
    constant uint&     count    [[buffer(3)]],
    constant uint&     upStride [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint tid = gid.x;
    uint batch = gid.y;
    if (tid < count) {
        uint packedIndex = batch * count + tid;
        uint upIndex = batch * upStride + tid;
        float g = float(gate[packedIndex]);
        float u = float(up[upIndex]);
        float g3 = g * g * g;
        float inner = 0.7978845608f * (g + 0.044715f * g3);
        inner = clamp(inner, -20.0f, 20.0f);
        float gelu_g = 0.5f * g * (1.0f + tanh(inner));
        output[packedIndex] = half(clamp(gelu_g * u, -65504.0f, 65504.0f));
    }
}

// ─── Logit cap: out = cap * tanh(logits / cap) ───
kernel void logit_cap(
    device const half* input  [[buffer(0)]],
    device half*       output [[buffer(1)]],
    constant uint&     count  [[buffer(2)]],
    constant float&    cap    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        float x = float(input[tid]);
        output[tid] = half(cap * tanh(x / cap));
    }
}
