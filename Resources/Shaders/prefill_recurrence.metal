#include <metal_stdlib>
using namespace metal;

inline half smelt_sigmoid_staged_half(half x) {
    const half tail = half(1.0h / (1.0h + exp(abs(x))));
    return x < 0.0h ? tail : half(1.0h - tail);
}

inline float smelt_log1p(float x) {
    const float xp1 = 1.0f + x;
    if (xp1 == 1.0f) {
        return x;
    }
    return x * (metal::log(xp1) / (xp1 - 1.0f));
}

inline half smelt_softplus_staged_half(half x) {
    const half maxValue = max(x, 0.0h);
    const half minValue = min(x, 0.0h);
    return half(float(maxValue) + smelt_log1p(float(metal::exp(minValue - maxValue))));
}

// ─── Fused DeltaNet recurrence for prefill ───
//
// Fuses conv1d + L2 norm Q/K + compute gates + 5 recurrence sub-kernels
// into a single kernel that loops over all B positions per head.
// Eliminates ~576 dispatches per layer (9 kernels × 64 positions → 1 dispatch).
//
// Each threadgroup owns one head and processes all positions sequentially.
// Intermediates (gates, kv_mem, delta) live in threadgroup memory.
//
// Layout assumptions:
//   qkvBuf:  [B, qkvDim] row-major (from batched matmul)
//   bBuf:    [B, numHeads] row-major
//   aBuf:    [B, numHeads] row-major
//   recOut:  [B, numHeads * headDim] row-major
//   state:   [numHeads, headDim, headDim] FP16 (persistent per layer, [Dv, Dk])
//   convSt:  [qkvDim, convKernel] (persistent per layer, shared across heads)
//
// Dispatch: numHeads threadgroups, headDim threads each.

kernel void deltanet_recurrence_prefill(
    device half*        state      [[buffer(0)]],   // [H, D, D] FP16
    device half*        convState  [[buffer(1)]],   // [C, K] conv state (shared)
    device half*        qkvBuf     [[buffer(2)]],   // [B, C] QKV (in-place after conv)
    device const half*  convWeight [[buffer(3)]],   // [C, K] conv weights
    device const half*  aLog       [[buffer(4)]],   // [H] log decay param
    device const half*  dtBias     [[buffer(5)]],   // [H] dt bias param
    device const half*  bBuf       [[buffer(6)]],   // [B, H] beta projections
    device const half*  aBuf       [[buffer(7)]],   // [B, H] alpha projections
    device half*        recOut     [[buffer(8)]],   // [B, H*D] output
    constant uint&      headDim    [[buffer(9)]],   // D = 128
    constant uint&      numHeads   [[buffer(10)]],  // H = 16
    constant uint&      seqLen     [[buffer(11)]],  // B (actual positions)
    constant uint&      qkvDim     [[buffer(12)]],  // C = 6144 = 3*H*D
    constant uint&      convKernel [[buffer(13)]],  // K = 4
    constant float&     headScale  [[buffer(14)]],  // 1/sqrt(D)
    constant float&     l2Eps      [[buffer(15)]],  // 1e-6
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    // Per-head geometry
    uint D = headDim;
    uint H = numHeads;
    uint C = qkvDim;           // 3 * H * D
    uint HD = H * D;           // one "block" = H*D elements (Q, K, or V)

    // This head's slice within Q/K/V
    // QKV layout per position: [Q0..Q_{H-1}, K0..K_{H-1}, V0..V_{H-1}]
    // Q[head] starts at head*D, K[head] at HD + head*D, V[head] at 2*HD + head*D
    uint qBase = head * D;
    uint kBase = HD + head * D;
    uint vBase = 2 * HD + head * D;

    // State offset for this head
    uint stateOff = head * D * D;

    // Threadgroup memory for intermediates
    threadgroup float tg_kv_mem[256];   // [D] — kv_mem_readout result
    threadgroup float tg_delta[256];    // [D] — compute_delta result
    threadgroup float tg_key[256];      // [D] — cached key for outer product
    threadgroup float tg_beta;          // scalar per head
    threadgroup float tg_g;             // scalar per head

    for (uint pos = 0; pos < seqLen; pos++) {
        uint posOff = pos * C;  // offset into qkvBuf for this position

        // ═══════════════════════════════════════════════
        // 1. Conv1d update + SiLU (this head's channels)
        // ═══════════════════════════════════════════════
        // Each head owns D channels in Q, K, V = 3*D channels total in QKV.
        // Conv1d operates on ALL channels (shared convState), but we only
        // process this head's channels here.
        // Wait — conv1d needs ALL channels done before L2 norm.
        // Since each threadgroup is one head, we need all heads to complete
        // conv1d before any head does L2 norm. This requires a global barrier
        // which Metal doesn't have within a single dispatch.
        //
        // SOLUTION: Process conv1d for this head's channels only.
        // Q channels: [head*D .. (head+1)*D)
        // K channels: [HD + head*D .. HD + (head+1)*D)
        // V channels: [2*HD + head*D .. 2*HD + (head+1)*D)
        // Total: 3*D channels per head, each thread handles 3*D/tgs channels.

        for (uint i = tid; i < 3 * D; i += tgs) {
            uint ch;
            if (i < D) {
                ch = head * D + i;           // Q channel
            } else if (i < 2 * D) {
                ch = HD + head * D + (i - D); // K channel
            } else {
                ch = 2 * HD + head * D + (i - 2 * D); // V channel
            }

            uint stOff = ch * convKernel;
            uint wOff = ch * convKernel;

            // Shift state, append new value
            float cached[8];
            for (uint k = 0; k < convKernel - 1; k++) {
                cached[k] = float(convState[stOff + k + 1]);
            }
            cached[convKernel - 1] = float(qkvBuf[posOff + ch]);

            for (uint k = 0; k < convKernel; k++) {
                convState[stOff + k] = half(cached[k]);
            }

            // Dot product + SiLU
            float acc = 0.0f;
            for (uint k = 0; k < convKernel; k++) {
                acc += cached[k] * float(convWeight[wOff + k]);
            }
            float activated = acc / (1.0f + exp(-acc));
            qkvBuf[posOff + ch] = half(activated);
        }
        threadgroup_barrier(mem_flags::mem_device);

        // ═══════════════════════════════════════════
        // 2. L2 normalize Q (this head's slice)
        // ═══════════════════════════════════════════
        {
            float sumSq = 0.0f;
            for (uint i = tid; i < D; i += tgs) {
                float v = float(qkvBuf[posOff + qBase + i]);
                sumSq += v * v;
            }
            sumSq = simd_sum(sumSq);

            threadgroup float partials[32];
            uint simd_lane = tid % 32;
            uint simd_group = tid / 32;
            if (simd_lane == 0) partials[simd_group] = sumSq;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup float shared_scale;
            if (tid == 0) {
                float total = 0;
                for (uint s = 0; s < tgs / 32; s++) total += partials[s];
                float norm = sqrt(total);
                shared_scale = 1.0f / max(norm, l2Eps);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = tid; i < D; i += tgs) {
                qkvBuf[posOff + qBase + i] = half(
                    float(qkvBuf[posOff + qBase + i]) * shared_scale
                );
            }
        }

        // ═══════════════════════════════════════════
        // 3. L2 normalize K (this head's slice)
        // ═══════════════════════════════════════════
        {
            float sumSq = 0.0f;
            for (uint i = tid; i < D; i += tgs) {
                float v = float(qkvBuf[posOff + kBase + i]);
                sumSq += v * v;
            }
            sumSq = simd_sum(sumSq);

            threadgroup float partials[32];
            uint simd_lane = tid % 32;
            uint simd_group = tid / 32;
            if (simd_lane == 0) partials[simd_group] = sumSq;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup float shared_scale;
            if (tid == 0) {
                float total = 0;
                for (uint s = 0; s < tgs / 32; s++) total += partials[s];
                float norm = sqrt(total);
                shared_scale = 1.0f / max(norm, l2Eps);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = tid; i < D; i += tgs) {
                qkvBuf[posOff + kBase + i] = half(
                    float(qkvBuf[posOff + kBase + i]) * shared_scale
                );
            }
        }

        // ═══════════════════════════════════════════
        // 4. Compute gates (thread 0 only, per-head scalars)
        // ═══════════════════════════════════════════
        if (tid == 0) {
            float b = float(bBuf[pos * H + head]);
            tg_beta = 1.0f / (1.0f + exp(-b));

            float a = float(aBuf[pos * H + head]);
            float al = float(aLog[head]);
            float db = float(dtBias[head]);
            float x_sp = a + db;
            float sp = (x_sp > 20.0f) ? x_sp : log(1.0f + exp(x_sp));
            tg_g = -exp(al) * sp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ═══════════════════════════════════════════
        // 5. State decay: state *= exp(g)
        // ═══════════════════════════════════════════
        {
            float decay = exp(tg_g);
            for (uint i = tid; i < D * D; i += tgs) {
                state[stateOff + i] = half(float(state[stateOff + i]) * decay);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // Cache key in threadgroup memory for steps 6-8
        for (uint i = tid; i < D; i += tgs) {
            tg_key[i] = float(qkvBuf[posOff + kBase + i]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ═══════════════════════════════════════════
        // 6. KV memory readout: kv_mem[v] = sum_k state[v,k] * key[k]
        // ═══════════════════════════════════════════
        for (uint v = tid; v < D; v += tgs) {
            float acc = 0.0f;
            for (uint k = 0; k < D; k++) {
                acc += float(state[stateOff + v * D + k]) * tg_key[k];
            }
            tg_kv_mem[v] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ═══════════════════════════════════════════
        // 7. Compute delta: delta[v] = (value[v] - kv_mem[v]) * beta
        // ═══════════════════════════════════════════
        for (uint v = tid; v < D; v += tgs) {
            float val = float(qkvBuf[posOff + vBase + v]);
            tg_delta[v] = (val - tg_kv_mem[v]) * tg_beta;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ═══════════════════════════════════════════
        // 8. Outer product update: state[v,k] += delta[v] * key[k]
        // ═══════════════════════════════════════════
        for (uint idx = tid; idx < D * D; idx += tgs) {
            uint v = idx / D;
            uint k = idx % D;
            float oldVal = float(state[stateOff + idx]);
            state[stateOff + idx] = half(oldVal + tg_delta[v] * tg_key[k]);
        }
        threadgroup_barrier(mem_flags::mem_device);

        // ═══════════════════════════════════════════
        // 9. Query readout: output[v] = sum_k state[v,k] * q[k] * scale
        // ═══════════════════════════════════════════
        uint outOff = pos * H * D + head * D;
        for (uint v = tid; v < D; v += tgs) {
            float acc = 0.0f;
            for (uint k = 0; k < D; k++) {
                float q = float(qkvBuf[posOff + qBase + k]) * headScale;
                acc += float(state[stateOff + v * D + k]) * q;
            }
            recOut[outOff + v] = half(acc);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// D128/H16 prompt-prefill capability using the MLX tiled recurrence geometry.
// Expects qkv to be pre-convolved and Q/K to be L2-normalized in-place.
// Launch: threads=(32, 128, 16), threadgroup=(32, 4, 1).
kernel void deltanet_recurrence_mlx_prefill_d128_h16(
    device half*        state   [[buffer(0)]],  // [16, 128, 128] FP16
    device const half*  qkv     [[buffer(1)]],  // [T, 3 * 16 * 128]
    device const half*  b_proj  [[buffer(2)]],  // [T, 16]
    device const half*  a_proj  [[buffer(3)]],  // [T, 16]
    device const half*  a_log   [[buffer(4)]],  // [16]
    device const half*  dt_bias [[buffer(5)]],  // [16]
    device half*        output  [[buffer(6)]],  // [T, 16, 128]
    constant uint&      seqLen  [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 128;
    constexpr uint H = 16;
    constexpr uint C = 3 * H * D;
    constexpr uint HIDDEN = H * D;

    uint head = gid.z;
    uint dvIdx = gid.y;

    uint qBase = head * D;
    uint kBase = HIDDEN + qBase;
    uint vBase = 2 * HIDDEN + qBase;
    uint stateBase = (head * D + dvIdx) * D;

    // Preserve MLX's lane ownership and reduction order. Unlike repeated
    // decode calls, MLX batched prefill keeps the recurrent state in FP32
    // registers across the entire chunk and rounds only when it persists the
    // final state.
    uint dk0 = lane * 4;
    uint dk1 = lane * 4 + 1;
    uint dk2 = lane * 4 + 2;
    uint dk3 = lane * 4 + 3;

    float s0 = float(state[stateBase + dk0]);
    float s1 = float(state[stateBase + dk1]);
    float s2 = float(state[stateBase + dk2]);
    float s3 = float(state[stateBase + dk3]);

    threadgroup float tgDecay = 0.0f;
    threadgroup float tgBeta = 0.0f;

    for (uint pos = 0; pos < seqLen; pos++) {
        uint qkvBase = pos * C;
        uint projBase = pos * H;

        float k0 = float(qkv[qkvBase + kBase + dk0]);
        float k1 = float(qkv[qkvBase + kBase + dk1]);
        float k2 = float(qkv[qkvBase + kBase + dk2]);
        float k3 = float(qkv[qkvBase + kBase + dk3]);

        float q0 = float(qkv[qkvBase + qBase + dk0]);
        float q1 = float(qkv[qkvBase + qBase + dk1]);
        float q2 = float(qkv[qkvBase + qBase + dk2]);
        float q3 = float(qkv[qkvBase + qBase + dk3]);

        if (lid.x == 0 && lid.y == 0) {
            tgBeta = float(smelt_sigmoid_staged_half(b_proj[projBase + head]));

            half xh = a_proj[projBase + head] + dt_bias[head];
            float al = float(a_log[head]);
            float sp = float(smelt_softplus_staged_half(xh));
            tgDecay = float(half(metal::precise::exp(-metal::precise::exp(al) * sp)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float decay = tgDecay;
        s0 *= decay;
        s1 *= decay;
        s2 *= decay;
        s3 *= decay;

        float kvMem = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
        kvMem = simd_sum(kvMem);

        float delta = (float(qkv[qkvBase + vBase + dvIdx]) - kvMem) * tgBeta;

        s0 += k0 * delta;
        s1 += k1 * delta;
        s2 += k2 * delta;
        s3 += k3 * delta;

        float out = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
        out = simd_sum(out);

        if (lane == 0) {
            output[pos * HIDDEN + head * D + dvIdx] = half(out);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

    }

    state[stateBase + dk0] = half(s0);
    state[stateBase + dk1] = half(s1);
    state[stateBase + dk2] = half(s2);
    state[stateBase + dk3] = half(s3);
}

kernel void deltanet_recurrence_mlx_prefill_d128_h32_qk16(
    device half*        state   [[buffer(0)]],  // [32, 128, 128] FP16
    device const half*  qkv     [[buffer(1)]],  // [T, 8192]
    device const half*  b_proj  [[buffer(2)]],  // [T, 32]
    device const half*  a_proj  [[buffer(3)]],  // [T, 32]
    device const half*  a_log   [[buffer(4)]],  // [32]
    device const half*  dt_bias [[buffer(5)]],  // [32]
    device half*        output  [[buffer(6)]],  // [T, 32, 128]
    constant uint&      seqLen  [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 128;
    constexpr uint Hqk = 16;
    constexpr uint Hv = 32;
    constexpr uint C = 2 * Hqk * D + Hv * D;
    constexpr uint HIDDEN = Hv * D;

    uint head = gid.z;
    uint dvIdx = gid.y;
    uint qkHead = head >> 1;

    uint qBase = qkHead * D;
    uint kBase = Hqk * D + qBase;
    uint vBase = 2 * Hqk * D + head * D;
    uint stateBase = (head * D + dvIdx) * D;

    // Preserve MLX's lane ownership and reduction order while carrying the
    // batched prefill state in FP32 until the final persistent-state write.
    uint dk0 = lane * 4;
    uint dk1 = lane * 4 + 1;
    uint dk2 = lane * 4 + 2;
    uint dk3 = lane * 4 + 3;

    float s0 = float(state[stateBase + dk0]);
    float s1 = float(state[stateBase + dk1]);
    float s2 = float(state[stateBase + dk2]);
    float s3 = float(state[stateBase + dk3]);

    threadgroup float tgDecay = 0.0f;
    threadgroup float tgBeta = 0.0f;

    for (uint pos = 0; pos < seqLen; pos++) {
        uint qkvBase = pos * C;
        uint projBase = pos * Hv;

        float k0 = float(qkv[qkvBase + kBase + dk0]);
        float k1 = float(qkv[qkvBase + kBase + dk1]);
        float k2 = float(qkv[qkvBase + kBase + dk2]);
        float k3 = float(qkv[qkvBase + kBase + dk3]);

        float q0 = float(qkv[qkvBase + qBase + dk0]);
        float q1 = float(qkv[qkvBase + qBase + dk1]);
        float q2 = float(qkv[qkvBase + qBase + dk2]);
        float q3 = float(qkv[qkvBase + qBase + dk3]);

        if (lid.x == 0 && lid.y == 0) {
            tgBeta = float(smelt_sigmoid_staged_half(b_proj[projBase + head]));

            half xh = a_proj[projBase + head] + dt_bias[head];
            float al = float(a_log[head]);
            float sp = float(smelt_softplus_staged_half(xh));
            tgDecay = float(half(metal::precise::exp(-metal::precise::exp(al) * sp)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float decay = tgDecay;
        s0 *= decay;
        s1 *= decay;
        s2 *= decay;
        s3 *= decay;

        float kvMem = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
        kvMem = simd_sum(kvMem);

        float delta = (float(qkv[qkvBase + vBase + dvIdx]) - kvMem) * tgBeta;

        s0 += k0 * delta;
        s1 += k1 * delta;
        s2 += k2 * delta;
        s3 += k3 * delta;

        float out = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
        out = simd_sum(out);

        if (lane == 0) {
            output[pos * HIDDEN + head * D + dvIdx] = half(out);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

    }

    state[stateBase + dk0] = half(s0);
    state[stateBase + dk1] = half(s1);
    state[stateBase + dk2] = half(s2);
    state[stateBase + dk3] = half(s3);
}

// Capability specialization for DeltaNet cells with D=128, 48 value heads,
// and 16 Q/K heads. Arithmetic order and lane ownership mirror MLX's generated
// batched kernel; recurrent state remains FP32 across the prompt chunk.
kernel void deltanet_recurrence_mlx_prefill_d128_h48_qk16(
    device half*        state   [[buffer(0)]],  // [48, 128, 128] FP16
    device const half*  qkv     [[buffer(1)]],  // [T, 10240]
    device const half*  b_proj  [[buffer(2)]],  // [T, 48]
    device const half*  a_proj  [[buffer(3)]],  // [T, 48]
    device const half*  a_log   [[buffer(4)]],  // [48]
    device const half*  dt_bias [[buffer(5)]],  // [48]
    device half*        output  [[buffer(6)]],  // [T, 48, 128]
    constant uint&      seqLen  [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 128;
    constexpr uint Hqk = 16;
    constexpr uint Hv = 48;
    constexpr uint C = 2 * Hqk * D + Hv * D;
    constexpr uint HIDDEN = Hv * D;

    uint head = gid.z;
    uint dvIdx = gid.y;
    uint qkHead = head / 3;

    uint qBase = qkHead * D;
    uint kBase = Hqk * D + qBase;
    uint vBase = 2 * Hqk * D + head * D;
    uint stateBase = (head * D + dvIdx) * D;

    constexpr uint elemsPerLane = D / 32;
    float stateReg[8];
    for (uint i = 0; i < elemsPerLane; i++) {
        uint dkIdx = lane * elemsPerLane + i;
        stateReg[i] = float(state[stateBase + dkIdx]);
    }

    threadgroup float tgDecay = 0.0f;
    threadgroup float tgBeta = 0.0f;

    for (uint pos = 0; pos < seqLen; pos++) {
        uint qkvBase = pos * C;
        uint projBase = pos * Hv;

        float keyReg[8];
        float queryReg[8];
        for (uint i = 0; i < elemsPerLane; i++) {
            uint dkIdx = lane * elemsPerLane + i;
            keyReg[i] = float(qkv[qkvBase + kBase + dkIdx]);
            queryReg[i] = float(qkv[qkvBase + qBase + dkIdx]);
        }

        if (lid.x == 0 && lid.y == 0) {
            tgBeta = float(smelt_sigmoid_staged_half(b_proj[projBase + head]));

            half xh = a_proj[projBase + head] + dt_bias[head];
            float al = float(a_log[head]);
            float sp = float(smelt_softplus_staged_half(xh));
            tgDecay = float(half(metal::precise::exp(-metal::precise::exp(al) * sp)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float kvMem = 0.0f;
        for (uint i = 0; i < elemsPerLane; i++) {
            stateReg[i] *= tgDecay;
            kvMem += stateReg[i] * keyReg[i];
        }
        kvMem = simd_sum(kvMem);

        float delta = (float(qkv[qkvBase + vBase + dvIdx]) - kvMem) * tgBeta;

        float out = 0.0f;
        for (uint i = 0; i < elemsPerLane; i++) {
            stateReg[i] += keyReg[i] * delta;
            out += stateReg[i] * queryReg[i];
        }
        out = simd_sum(out);

        if (lane == 0) {
            output[pos * HIDDEN + head * D + dvIdx] = half(out);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

    }

    for (uint i = 0; i < elemsPerLane; i++) {
        uint dkIdx = lane * elemsPerLane + i;
        state[stateBase + dkIdx] = half(stateReg[i]);
    }
}

kernel void deltanet_recurrence_mlx_prefill(
    device half*        state      [[buffer(0)]],  // [Hv, D, D] FP16
    device const half*  qkv        [[buffer(1)]],  // [T, 2 * Hqk * D + Hv * D]
    device const half*  b_proj     [[buffer(2)]],  // [T, Hv]
    device const half*  a_proj     [[buffer(3)]],  // [T, Hv]
    device const half*  a_log      [[buffer(4)]],  // [Hv]
    device const half*  dt_bias    [[buffer(5)]],  // [Hv]
    device half*        output     [[buffer(6)]],  // [T, Hv, D]
    constant uint&      headDim    [[buffer(7)]],
    constant uint&      valueHeads [[buffer(8)]],
    constant uint&      qkHeads    [[buffer(9)]],
    constant uint&      seqLen     [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint head = gid.z;
    uint dvIdx = gid.y;
    if (head >= valueHeads || dvIdx >= headDim) return;

    uint elemsPerLane = headDim / 32;
    if (elemsPerLane == 0 || elemsPerLane > 8) return;

    uint repeatFactor = valueHeads / qkHeads;
    uint qkHead = head / repeatFactor;
    uint qBase = qkHead * headDim;
    uint kBase = qkHeads * headDim + qBase;
    uint vBase = 2 * qkHeads * headDim + head * headDim;
    uint hiddenStride = (2 * qkHeads + valueHeads) * headDim;
    uint stateBase = (head * headDim + dvIdx) * headDim;
    float stateReg[8];
    for (uint i = 0; i < elemsPerLane; i++) {
        uint dkIdx = lane * elemsPerLane + i;
        stateReg[i] = float(state[stateBase + dkIdx]);
    }

    threadgroup float tgDecay = 0.0f;
    threadgroup float tgBeta = 0.0f;

    for (uint pos = 0; pos < seqLen; pos++) {
        uint qkvBase = pos * hiddenStride;
        uint projBase = pos * valueHeads;

        float keyReg[8];
        float queryReg[8];
        for (uint i = 0; i < elemsPerLane; i++) {
            uint dkIdx = lane * elemsPerLane + i;
            keyReg[i] = float(qkv[qkvBase + kBase + dkIdx]);
            queryReg[i] = float(qkv[qkvBase + qBase + dkIdx]);
        }

        if (lid.x == 0 && lid.y == 0) {
            tgBeta = float(smelt_sigmoid_staged_half(b_proj[projBase + head]));

            half xh = a_proj[projBase + head] + dt_bias[head];
            float al = float(a_log[head]);
            float sp = float(smelt_softplus_staged_half(xh));
            tgDecay = float(half(metal::precise::exp(-metal::precise::exp(al) * sp)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float kvMem = 0.0f;
        for (uint i = 0; i < elemsPerLane; i++) {
            stateReg[i] *= tgDecay;
            kvMem += stateReg[i] * keyReg[i];
        }
        kvMem = simd_sum(kvMem);

        float delta = (float(qkv[qkvBase + vBase + dvIdx]) - kvMem) * tgBeta;

        float out = 0.0f;
        for (uint i = 0; i < elemsPerLane; i++) {
            stateReg[i] += keyReg[i] * delta;
            out += stateReg[i] * queryReg[i];
        }
        out = simd_sum(out);

        if (lane == 0) {
            output[pos * valueHeads * headDim + head * headDim + dvIdx] = half(out);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

    }

    for (uint i = 0; i < elemsPerLane; i++) {
        uint dkIdx = lane * elemsPerLane + i;
        state[stateBase + dkIdx] = half(stateReg[i]);
    }
}

// Verify transaction variant. Every thread owns one strided slice of the
// recurrent matrix, so it can checkpoint the entry state and each rounded
// prefix successor without synchronization or an extra full-state pass. Feed
// every FP16 snapshot back into the live trajectory so each row has the same
// state boundary as an independent sequential decode.
kernel void deltanet_recurrence_mlx_prefill_checkpoint(
    device half*        state      [[buffer(0)]],  // [Hv, D, D] FP16
    device const half*  qkv        [[buffer(1)]],  // [T, 2 * Hqk * D + Hv * D]
    device const half*  b_proj     [[buffer(2)]],  // [T, Hv]
    device const half*  a_proj     [[buffer(3)]],  // [T, Hv]
    device const half*  a_log      [[buffer(4)]],  // [Hv]
    device const half*  dt_bias    [[buffer(5)]],  // [Hv]
    device half*        output     [[buffer(6)]],  // [T, Hv, D]
    constant uint&      headDim    [[buffer(7)]],
    constant uint&      valueHeads [[buffer(8)]],
    constant uint&      qkHeads    [[buffer(9)]],
    constant uint&      seqLen     [[buffer(10)]],
    device half*        history    [[buffer(11)]], // [T + 1, Hv, D, D]
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint head = gid.z;
    uint dvIdx = gid.y;
    if (head >= valueHeads || dvIdx >= headDim) return;

    uint elemsPerLane = headDim / 32;
    if (elemsPerLane == 0 || elemsPerLane > 8) return;

    uint repeatFactor = valueHeads / qkHeads;
    uint qkHead = head / repeatFactor;
    uint qBase = qkHead * headDim;
    uint kBase = qkHeads * headDim + qBase;
    uint vBase = 2 * qkHeads * headDim + head * headDim;
    uint hiddenStride = (2 * qkHeads + valueHeads) * headDim;
    uint stateBase = (head * headDim + dvIdx) * headDim;
    uint historyStride = valueHeads * headDim * headDim;
    float stateReg[8];
    for (uint i = 0; i < elemsPerLane; i++) {
        uint dkIdx = lane * elemsPerLane + i;
        stateReg[i] = float(state[stateBase + dkIdx]);
        history[stateBase + dkIdx] = half(stateReg[i]);
    }

    threadgroup float tgDecay = 0.0f;
    threadgroup float tgBeta = 0.0f;

    for (uint pos = 0; pos < seqLen; pos++) {
        uint qkvBase = pos * hiddenStride;
        uint projBase = pos * valueHeads;

        float keyReg[8];
        float queryReg[8];
        for (uint i = 0; i < elemsPerLane; i++) {
            uint dkIdx = lane * elemsPerLane + i;
            keyReg[i] = float(qkv[qkvBase + kBase + dkIdx]);
            queryReg[i] = float(qkv[qkvBase + qBase + dkIdx]);
        }

        if (lid.x == 0 && lid.y == 0) {
            tgBeta = float(smelt_sigmoid_staged_half(b_proj[projBase + head]));

            half xh = a_proj[projBase + head] + dt_bias[head];
            float al = float(a_log[head]);
            float sp = float(smelt_softplus_staged_half(xh));
            tgDecay = float(half(metal::precise::exp(-metal::precise::exp(al) * sp)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float kvMem = 0.0f;
        for (uint i = 0; i < elemsPerLane; i++) {
            stateReg[i] *= tgDecay;
            kvMem += stateReg[i] * keyReg[i];
        }
        kvMem = simd_sum(kvMem);

        float delta = (float(qkv[qkvBase + vBase + dvIdx]) - kvMem) * tgBeta;

        float out = 0.0f;
        for (uint i = 0; i < elemsPerLane; i++) {
            stateReg[i] += keyReg[i] * delta;
            out += stateReg[i] * queryReg[i];
        }
        out = simd_sum(out);

        if (lane == 0) {
            output[pos * valueHeads * headDim + head * headDim + dvIdx] = half(out);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint historyBase = (pos + 1) * historyStride + stateBase;
        for (uint i = 0; i < elemsPerLane; i++) {
            uint dkIdx = lane * elemsPerLane + i;
            const half rounded = half(stateReg[i]);
            history[historyBase + dkIdx] = rounded;
            // Match the ordinary decode boundary: its recurrent matrix is
            // materialized as FP16 after every token, so the following token
            // starts from the rounded state rather than this kernel's private
            // FP32 accumulator.
            stateReg[i] = float(rounded);
        }
    }

    for (uint i = 0; i < elemsPerLane; i++) {
        uint dkIdx = lane * elemsPerLane + i;
        state[stateBase + dkIdx] = half(stateReg[i]);
    }
}
