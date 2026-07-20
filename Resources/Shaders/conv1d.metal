#include <metal_stdlib>
using namespace metal;

inline half smelt_silu_staged_half(half x) {
    const half tail = half(1.0h / (1.0h + exp(abs(x))));
    const half sigmoid = x < 0.0h ? tail : half(1.0h - tail);
    const half result = x * sigmoid;
    // MLX canonicalizes underflowed SiLU results to +0. Signed zero is
    // numerically inert but breaks frozen-boundary bit parity and hashes. Use
    // a bit test: fast-math is otherwise allowed to fold a value comparison
    // back into the sign-preserving result.
    const ushort bits = as_type<ushort>(result);
    return (bits & 0x7fffu) == 0u ? as_type<half>(ushort(0)) : result;
}

// MLX's depthwise_conv_1d accumulates one tap at a time from a float zero.
// `dot(float4, float4)` has a different reduction topology and can move the
// fp16 result even though the real-valued expression is equivalent.
inline float smelt_conv4_acc(half4 x, half4 w) {
    float acc = 0.0f;
    acc += float(x.x) * float(w.x);
    acc += float(x.y) * float(w.y);
    acc += float(x.z) * float(w.z);
    acc += float(x.w) * float(w.w);
    return acc;
}

// ─── Conv1d update for DeltaNet decode step ───
// Shift state left by 1, append new value, dot product per channel, SiLU.
// state: [channels, kernelSize=4], new_val: [channels], conv_weight: [channels, kernelSize=4]
// Output: [channels] (activated), state updated in-place.
// One thread per channel.

kernel void conv1d_update_silu_c6144_k4(
    device half*       state       [[buffer(0)]],
    device const half* new_val     [[buffer(1)]],
    device const half* conv_weight [[buffer(2)]],
    device half*       output      [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= 6144) return;

    uint stateOffset = tid * 4;
    half4 shifted = half4(
        state[stateOffset + 1],
        state[stateOffset + 2],
        state[stateOffset + 3],
        new_val[tid]
    );

    state[stateOffset + 0] = shifted.x;
    state[stateOffset + 1] = shifted.y;
    state[stateOffset + 2] = shifted.z;
    state[stateOffset + 3] = shifted.w;

    half4 weights = half4(
        conv_weight[stateOffset + 0],
        conv_weight[stateOffset + 1],
        conv_weight[stateOffset + 2],
        conv_weight[stateOffset + 3]
    );
    float acc = smelt_conv4_acc(shifted, weights);
    output[tid] = smelt_silu_staged_half(half(acc));
}

// Qwen 0.8B decode specialization: the first 32 heads are Q/K and are
// L2-normalized immediately after the convolution; the final 16 V heads keep
// the rounded convolution output. This preserves the staged FP16 boundary.
kernel void conv1d_update_silu_l2_qk_c6144_k4_d128_h16(
    device half*       state       [[buffer(0)]],
    device const half* new_val     [[buffer(1)]],
    device const half* conv_weight [[buffer(2)]],
    device half*       output      [[buffer(3)]],
    uint head       [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint D = 128;
    constexpr uint QK_HEADS = 32;
    constexpr float eps = 1e-6f;
    uint channel = head * D + tid;
    uint stateOffset = channel * 4;

    half4 shifted = half4(
        state[stateOffset + 1],
        state[stateOffset + 2],
        state[stateOffset + 3],
        new_val[channel]
    );
    state[stateOffset + 0] = shifted.x;
    state[stateOffset + 1] = shifted.y;
    state[stateOffset + 2] = shifted.z;
    state[stateOffset + 3] = shifted.w;

    half4 weights = half4(
        conv_weight[stateOffset + 0],
        conv_weight[stateOffset + 1],
        conv_weight[stateOffset + 2],
        conv_weight[stateOffset + 3]
    );
    float acc = smelt_conv4_acc(shifted, weights);
    half convolved = smelt_silu_staged_half(half(acc));
    output[channel] = convolved;

    if (head < QK_HEADS) {
        threadgroup half cached[D];
        threadgroup float partial[4];
        threadgroup float shared_scale = 0.0f;
        cached[tid] = convolved;
        float v = float(convolved);
        float sumSq = simd_sum(v * v);
        if (simd_lane == 0) {
            partial[simd_group] = sumSq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = partial[0] + partial[1] + partial[2] + partial[3];
            shared_scale = 1.0f / max(sqrt(total), eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        output[channel] = half(float(cached[tid]) * shared_scale);
    }
}

// Prompt-prefill specialization for Qwen DeltaNet.
// Processes one channel across the full prompt chunk sequentially, keeping the
// 4-tap conv state in registers and writing it back once at the end.
kernel void conv1d_update_silu_c6144_k4_prefill(
    device half*       state       [[buffer(0)]],  // [6144, 4]
    device half*       qkv         [[buffer(1)]],  // [T, 6144] in-place
    device const half* conv_weight [[buffer(2)]],  // [6144, 4]
    constant uint&     seqLen      [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= 6144) return;

    uint stateOffset = tid * 4;
    float s0 = float(state[stateOffset + 0]);
    float s1 = float(state[stateOffset + 1]);
    float s2 = float(state[stateOffset + 2]);
    float s3 = float(state[stateOffset + 3]);

    float w0 = float(conv_weight[stateOffset + 0]);
    float w1 = float(conv_weight[stateOffset + 1]);
    float w2 = float(conv_weight[stateOffset + 2]);
    float w3 = float(conv_weight[stateOffset + 3]);

    for (uint pos = 0; pos < seqLen; pos++) {
        uint idx = pos * 6144 + tid;
        float x = float(qkv[idx]);

        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = x;

        float acc = 0.0f;
        acc = metal::precise::fma(s0, w0, acc);
        acc = metal::precise::fma(s1, w1, acc);
        acc = metal::precise::fma(s2, w2, acc);
        acc = metal::precise::fma(s3, w3, acc);
        const half convolved = half(acc);
        qkv[idx] = smelt_silu_staged_half(convolved);
    }

    state[stateOffset + 0] = half(s0);
    state[stateOffset + 1] = half(s1);
    state[stateOffset + 2] = half(s2);
    state[stateOffset + 3] = half(s3);
}

kernel void conv1d_update_silu_prefill(
    device half*       state       [[buffer(0)]],  // [channels, kernelSize]
    device half*       qkv         [[buffer(1)]],  // [seqLen, channels] in-place
    device const half* conv_weight [[buffer(2)]],  // [channels, kernelSize]
    constant uint&     seqLen      [[buffer(3)]],
    constant uint&     channels    [[buffer(4)]],
    constant uint&     kernelSize  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= channels) return;

    uint stateOffset = tid * kernelSize;
    uint weightOffset = tid * kernelSize;

    float cached[8];
    for (uint k = 0; k < kernelSize; k++) {
        cached[k] = float(state[stateOffset + k]);
    }

    for (uint pos = 0; pos < seqLen; pos++) {
        uint idx = pos * channels + tid;
        float x = float(qkv[idx]);

        for (uint k = 0; k < kernelSize - 1; k++) {
            cached[k] = cached[k + 1];
        }
        cached[kernelSize - 1] = x;

        float acc = 0.0f;
        for (uint k = 0; k < kernelSize; k++) {
            acc += cached[k] * float(conv_weight[weightOffset + k]);
        }
        const half convolved = half(acc);
        qkv[idx] = smelt_silu_staged_half(convolved);
    }

    for (uint k = 0; k < kernelSize; k++) {
        state[stateOffset + k] = half(cached[k]);
    }
}

// Verify transaction variant. In addition to updating the live convolution
// state, retain the entry state and every per-token successor so the runtime
// can commit only the accepted speculative prefix without replaying tokens.
kernel void conv1d_update_silu_prefill_checkpoint(
    device half*       state       [[buffer(0)]],  // [channels, kernelSize]
    device half*       qkv         [[buffer(1)]],  // [seqLen, channels] in-place
    device const half* conv_weight [[buffer(2)]],  // [channels, kernelSize]
    constant uint&     seqLen      [[buffer(3)]],
    constant uint&     channels    [[buffer(4)]],
    constant uint&     kernelSize  [[buffer(5)]],
    device half*       history     [[buffer(6)]],  // [seqLen + 1, channels, kernelSize]
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= channels) return;

    uint stateOffset = tid * kernelSize;
    uint weightOffset = tid * kernelSize;
    uint historyStride = channels * kernelSize;

    float cached[8];
    for (uint k = 0; k < kernelSize; k++) {
        cached[k] = float(state[stateOffset + k]);
        history[stateOffset + k] = half(cached[k]);
    }

    for (uint pos = 0; pos < seqLen; pos++) {
        uint idx = pos * channels + tid;
        float x = float(qkv[idx]);

        for (uint k = 0; k < kernelSize - 1; k++) {
            cached[k] = cached[k + 1];
        }
        cached[kernelSize - 1] = x;

        float acc = 0.0f;
        for (uint k = 0; k < kernelSize; k++) {
            acc += cached[k] * float(conv_weight[weightOffset + k]);
        }
        qkv[idx] = smelt_silu_staged_half(half(acc));

        uint historyBase = (pos + 1) * historyStride + stateOffset;
        for (uint k = 0; k < kernelSize; k++) {
            const half rounded = half(cached[k]);
            history[historyBase + k] = rounded;
            // A sequential decode stores FP16 state after every token and
            // reloads that rounded value on the next token. Verification must
            // feed the checkpoint boundary back into its live trajectory too;
            // merely recording it lets hidden FP32 state drift across rows.
            cached[k] = float(rounded);
        }
    }

    for (uint k = 0; k < kernelSize; k++) {
        state[stateOffset + k] = half(cached[k]);
    }
}

kernel void conv1d_update_silu(
    device half*       state      [[buffer(0)]],  // [C, K] — updated in-place
    device const half* new_val    [[buffer(1)]],  // [C] — new QKV column
    device const half* conv_weight[[buffer(2)]],  // [C, K] — conv weights (no bias)
    device half*       output     [[buffer(3)]],  // [C] — activated output
    constant uint&     channels   [[buffer(4)]],  // C (6144)
    constant uint&     kernelSize [[buffer(5)]],  // K (4)
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= channels) return;

    uint ch = tid;
    uint stateOffset = ch * kernelSize;
    uint weightOffset = ch * kernelSize;

    if (kernelSize == 4) {
        half4 shifted = half4(
            state[stateOffset + 1],
            state[stateOffset + 2],
            state[stateOffset + 3],
            new_val[ch]
        );

        state[stateOffset + 0] = shifted.x;
        state[stateOffset + 1] = shifted.y;
        state[stateOffset + 2] = shifted.z;
        state[stateOffset + 3] = shifted.w;

        half4 weights = half4(
            conv_weight[weightOffset + 0],
            conv_weight[weightOffset + 1],
            conv_weight[weightOffset + 2],
            conv_weight[weightOffset + 3]
        );
        float acc = smelt_conv4_acc(shifted, weights);
        output[ch] = smelt_silu_staged_half(half(acc));
        return;
    }

    // Read shifted values into registers to avoid device write-then-reread.
    // shifted = [state[1], state[2], ..., state[K-1], new_val]
    float cached[8];  // max supported kernel size
    for (uint k = 0; k < kernelSize - 1; k++) {
        cached[k] = float(state[stateOffset + k + 1]);
    }
    cached[kernelSize - 1] = float(new_val[ch]);

    // Write shifted state back to device
    for (uint k = 0; k < kernelSize; k++) {
        state[stateOffset + k] = half(cached[k]);
    }

    // Dot product from cached registers (no device re-read)
    float acc = 0.0f;
    for (uint k = 0; k < kernelSize; k++) {
        acc += cached[k] * float(conv_weight[weightOffset + k]);
    }

    // SiLU activation
    output[ch] = smelt_silu_staged_half(half(acc));
}
