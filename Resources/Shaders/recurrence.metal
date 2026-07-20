#include <metal_stdlib>
using namespace metal;

inline half smelt_decode_sigmoid_staged_half(half x) {
    const half tail = half(1.0h / (1.0h + exp(abs(x))));
    return x < 0.0h ? tail : half(1.0h - tail);
}

inline float smelt_decode_log1p(float x) {
    const float xp1 = 1.0f + x;
    if (xp1 == 1.0f) return x;
    return x * (metal::log(xp1) / (xp1 - 1.0f));
}

inline half smelt_decode_softplus_staged_half(half x) {
    const half maxValue = max(x, 0.0h);
    const half minValue = min(x, 0.0h);
    return half(
        float(maxValue)
        + smelt_decode_log1p(float(metal::exp(minValue - maxValue))));
}

// ─── DeltaNet recurrence (single decode step) ───
//
// Per head (16 independent heads):
//   scale = 1 / sqrt(128)
//   q = q * scale
//   state = state * exp(g)                           // decay
//   kv_mem = state^T @ k                             // readout: [128]
//   delta = (v - kv_mem) * beta                      // gated update: [128]
//   state = state + k outer delta                    // rank-1 update: [128, 128]
//   output = state^T @ q                             // query readout: [128]
//
// State is FP16 [16, 128, 128]. Q/K/V are FP16 [16, 128].
// Gates g [16] and beta [16] are FP16 scalars per head.
//
// We break this into sub-kernels for clarity and correctness:
//   1. state_decay: state *= exp(g)
//   2. kv_mem_readout: kv_mem = state^T @ k
//   3. compute_delta: delta = (v - kv_mem) * beta
//   4. outer_product_update: state += k outer delta
//   5. query_readout: output = state^T @ q * scale

// ─── 1. State decay ───
// state[h, i, j] *= exp(g[h]) for all i, j
// Dispatch: [16 * 128 * 128] threads or [16] threadgroups × [128*128/tpg] each

kernel void state_decay(
    device half*       state   [[buffer(0)]],  // [H, D, D] FP16
    device const half* g_val   [[buffer(1)]],  // [H] decay gate (already negated: g = -exp(A_log)*softplus(a+bias))
    constant uint&     headDim [[buffer(2)]],  // D = 128
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    float decay = exp(float(g_val[head]));
    uint stateSize = headDim * headDim;
    uint offset = head * stateSize;

    for (uint i = tid; i < stateSize; i += tgs) {
        state[offset + i] = half(float(state[offset + i]) * decay);
    }
}

// ─── 2. KV memory readout: kv_mem[h, v] = sum_k(state[h, k, v] * key[h, k]) ───
// For each head, compute state^T @ k: [D, D]^T @ [D] = [D]
// This is a matvec: for each output element v, dot product of state column v with k.
// Dispatch: H threadgroups × D threads (each thread computes one output element)

kernel void kv_mem_readout(
    device const half*  state   [[buffer(0)]],  // [H, D, D] FP16
    device const half*  key     [[buffer(1)]],  // [H, D] FP16
    device float*       kv_mem  [[buffer(2)]],  // [H, D] FP32 output
    constant uint&      headDim [[buffer(3)]],  // D = 128
    uint head [[threadgroup_position_in_grid]],
    uint vid  [[thread_index_in_threadgroup]]     // value dimension index
) {
    if (vid >= headDim) return;

    uint stateOffset = head * headDim * headDim;
    uint keyOffset = head * headDim;

    // kv_mem[v] = sum_k state[k, v] * key[k]
    // state is row-major [D, D], so state[k, v] = state[k * D + v]
    float acc = 0.0f;
    for (uint k = 0; k < headDim; k++) {
        acc += float(state[stateOffset + k * headDim + vid]) * float(key[keyOffset + k]);
    }
    kv_mem[head * headDim + vid] = acc;
}

// ─── 3. Compute delta: delta[h, v] = (value[h, v] - kv_mem[h, v]) * beta[h] ───
// Dispatch: [H * D] threads

kernel void compute_delta(
    device const half*  value  [[buffer(0)]],  // [H, D] FP16
    device const float* kv_mem [[buffer(1)]],  // [H, D] FP32
    device const half*  beta   [[buffer(2)]],  // [H] FP16
    device float*       delta  [[buffer(3)]],  // [H, D] FP32 output
    constant uint&      headDim [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint head = tid / headDim;
    float v = float(value[tid]);
    float mem = kv_mem[tid];
    float b = float(beta[head]);
    delta[tid] = (v - mem) * b;
}

// ─── 4. Outer product update: state[h, k, v] += key[h, k] * delta[h, v] ───
// Dispatch: H threadgroups × (D*D / some factor) threads

kernel void outer_product_update(
    device half*        state  [[buffer(0)]],  // [H, D, D] FP16
    device const half*  key    [[buffer(1)]],  // [H, D] FP16
    device const float* delta  [[buffer(2)]],  // [H, D] FP32
    constant uint&      headDim [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    uint stateSize = headDim * headDim;
    uint stateOffset = head * stateSize;
    uint headOffset = head * headDim;

    // Cache key and delta in threadgroup memory — avoids repeated device reads
    // across the D*D iteration space (16384 iterations for D=128)
    threadgroup float key_tg[256];
    threadgroup float delta_tg[256];
    for (uint i = tid; i < headDim; i += tgs) {
        key_tg[i] = float(key[headOffset + i]);
        delta_tg[i] = delta[headOffset + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < stateSize; idx += tgs) {
        uint k = idx / headDim;
        uint v = idx % headDim;
        float oldVal = float(state[stateOffset + idx]);
        state[stateOffset + idx] = half(oldVal + key_tg[k] * delta_tg[v]);
    }
}

// ─── 2+3 Fused: KV readout + delta in one pass ───
// Reads UN-decayed state, applies decay scalar to readout, computes delta inline.
// decay * (state^T @ K) == (state * decay)^T @ K since decay is scalar per head.
// Eliminates kv_mem intermediate buffer and compute_delta dispatch.
// Dispatch: H threadgroups × D threads

kernel void kv_readout_delta(
    device const half*  state   [[buffer(0)]],  // [H, D, D] FP16 (un-decayed)
    device const half*  key     [[buffer(1)]],  // [H, D] FP16
    device const half*  value   [[buffer(2)]],  // [H, D] FP16
    device const half*  g_val   [[buffer(3)]],  // [H] decay gate
    device const half*  beta    [[buffer(4)]],  // [H] beta gate
    device float*       delta   [[buffer(5)]],  // [H, D] FP32 output
    constant uint&      headDim [[buffer(6)]],  // D = 128
    uint head [[threadgroup_position_in_grid]],
    uint vid  [[thread_index_in_threadgroup]]
) {
    if (vid >= headDim) return;

    uint stateOffset = head * headDim * headDim;
    uint keyOffset = head * headDim;
    float decay = exp(float(g_val[head]));

    // kv_mem = decay * state^T @ key  (applying decay to readout, not state)
    float acc = 0.0f;
    for (uint k = 0; k < headDim; k++) {
        acc += float(state[stateOffset + k * headDim + vid]) * float(key[keyOffset + k]);
    }
    float kv_mem = acc * decay;

    // delta = (value - kv_mem) * beta
    float v = float(value[head * headDim + vid]);
    float b = float(beta[head]);
    delta[head * headDim + vid] = (v - kv_mem) * b;
}

// ─── Fused: gates + KV readout + delta ───
// Computes beta=sigmoid(b), g=-exp(A_log)*softplus(a+dt_bias) inline,
// then performs KV readout + delta. Eliminates compute_gates dispatch.
// Thread 0 computes gates and broadcasts via threadgroup memory.
// Dispatch: H threadgroups × D threads

kernel void gates_kv_readout_delta(
    device const half*  state    [[buffer(0)]],   // [H, D, D] FP16
    device const half*  key      [[buffer(1)]],   // [H, D] FP16
    device const half*  value    [[buffer(2)]],   // [H, D] FP16
    device const half*  b_proj   [[buffer(3)]],   // [H] beta input
    device const half*  a_proj   [[buffer(4)]],   // [H] alpha input
    device const half*  a_log    [[buffer(5)]],   // [H] log decay param
    device const half*  dt_bias  [[buffer(6)]],   // [H] dt bias
    device float*       delta    [[buffer(7)]],   // [H, D] FP32 output
    device half*        g_out    [[buffer(8)]],   // [H] decay gate output (for state_decay_update)
    constant uint&      headDim  [[buffer(9)]],   // D = 128
    uint head [[threadgroup_position_in_grid]],
    uint vid  [[thread_index_in_threadgroup]]
) {
    // Thread 0 computes gates, broadcasts to all threads
    threadgroup float tg_decay;
    threadgroup float tg_beta;

    if (vid == 0) {
        // Beta = sigmoid(b)
        float b = float(b_proj[head]);
        tg_beta = 1.0f / (1.0f + exp(-b));

        // G = -exp(A_log) * softplus(a + dt_bias)
        float a = float(a_proj[head]);
        float al = float(a_log[head]);
        float db = float(dt_bias[head]);
        float x_sp = a + db;
        float sp = (x_sp > 20.0f) ? x_sp : log(1.0f + exp(x_sp));
        float g = -exp(al) * sp;
        tg_decay = exp(g);
        g_out[head] = half(g);  // state_decay_update needs this
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (vid >= headDim) return;

    float decay = tg_decay;
    float beta = tg_beta;

    uint stateOffset = head * headDim * headDim;
    uint keyOffset = head * headDim;

    // kv_mem = decay * state^T @ key
    float acc = 0.0f;
    for (uint k = 0; k < headDim; k++) {
        acc += float(state[stateOffset + k * headDim + vid]) * float(key[keyOffset + k]);
    }
    float kv_mem = acc * decay;

    // delta = (value - kv_mem) * beta
    float v = float(value[head * headDim + vid]);
    delta[head * headDim + vid] = (v - kv_mem) * beta;
}

// ─── 1+4 Fused: state decay + outer product update in one pass ───
// state[h,i,j] = state[h,i,j] * exp(g[h]) + key[h,i] * delta[h,j]
// Saves one full state matrix read+write round-trip per layer.
// Dispatch: H threadgroups × 256 threads

kernel void state_decay_update(
    device half*        state   [[buffer(0)]],  // [H, D, D] FP16
    device const half*  g_val   [[buffer(1)]],  // [H] decay gate
    device const half*  key     [[buffer(2)]],  // [H, D] FP16
    device const float* delta   [[buffer(3)]],  // [H, D] FP32
    constant uint&      headDim [[buffer(4)]],  // D = 128
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]]
) {
    float decay = exp(float(g_val[head]));
    uint stateSize = headDim * headDim;
    uint stateOffset = head * stateSize;
    uint headOffset = head * headDim;

    // Cache key and delta in threadgroup memory
    threadgroup float key_tg[256];
    threadgroup float delta_tg[256];
    for (uint i = tid; i < headDim; i += tgs) {
        key_tg[i] = float(key[headOffset + i]);
        delta_tg[i] = delta[headOffset + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < stateSize; idx += tgs) {
        uint k = idx / headDim;
        uint v = idx % headDim;
        float oldVal = float(state[stateOffset + idx]);
        state[stateOffset + idx] = half(oldVal * decay + key_tg[k] * delta_tg[v]);
    }
}

// ─── 5. Query readout: output[h, v] = sum_k(state[h, k, v] * q_scaled[h, k]) ───
// Same structure as kv_mem_readout but with scaled query.
// Scale applied inline: q_scaled = q * (1/sqrt(D))

kernel void query_readout(
    device const half*  state   [[buffer(0)]],  // [H, D, D] FP16
    device const half*  query   [[buffer(1)]],  // [H, D] FP16
    device half*        output  [[buffer(2)]],  // [H, D] FP16 output
    constant uint&      headDim [[buffer(3)]],  // D = 128
    constant float&     scale   [[buffer(4)]],  // 1/sqrt(D)
    uint head [[threadgroup_position_in_grid]],
    uint vid  [[thread_index_in_threadgroup]]
) {
    if (vid >= headDim) return;

    uint stateOffset = head * headDim * headDim;
    uint queryOffset = head * headDim;

    float acc = 0.0f;
    for (uint k = 0; k < headDim; k++) {
        float q = float(query[queryOffset + k]) * scale;
        acc += float(state[stateOffset + k * headDim + vid]) * q;
    }
    output[head * headDim + vid] = half(acc);
}

// ─── Mega-fused DeltaNet block: conv1d + L2 norm + recurrence ───
// Absorbs conv1d_update_silu, L2 normalize, tiled recurrence, and gated RMS norm.
// Unlike the old mega-fused implementation, the recurrent state is treated as
// row-major [Dv, Dk] and each simdgroup owns one output row at a time.
// Dispatch: H threadgroups × 256 threads.

kernel void deltanet_recurrence_fused(
    device half*        state      [[buffer(0)]],   // [H, Dv, Dk] FP16 (row-major, read-write)
    device half*        qkv        [[buffer(1)]],   // [qkvDim] Q+K+V input from matvec
    device const half*  b_proj     [[buffer(2)]],   // [H]
    device const half*  a_proj     [[buffer(3)]],   // [H]
    device const half*  a_log      [[buffer(4)]],   // [H]
    device const half*  dt_bias    [[buffer(5)]],   // [H]
    device half*        output     [[buffer(6)]],   // [H, D] FP16 (gated-norm applied)
    device half*        convState  [[buffer(7)]],   // [qkvDim, convK] conv state
    device const half*  convWeight [[buffer(8)]],   // [qkvDim, convK] conv weights
    device const half*  z_proj     [[buffer(9)]],   // [H*D] z projection for gating
    device const half*  normWeight [[buffer(10)]],  // [D] gated RMS norm weight
    constant uint&      headDim    [[buffer(11)]],  // D (128)
    constant float&     headScale  [[buffer(12)]],  // 1/sqrt(D)
    constant uint&      qkvDim     [[buffer(13)]],  // Q+K+V total dim (e.g. 6144)
    constant uint&      convK      [[buffer(14)]],  // conv kernel size (e.g. 4)
    constant uint&      numHeads   [[buffer(15)]],  // H
    constant float&     rmsEps     [[buffer(16)]],  // RMS norm epsilon
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint D = headDim;
    uint H = numHeads;
    uint simdCount = tgs / 32;
    uint elemsPerLane = D / 32;
    if (elemsPerLane == 0 || elemsPerLane > 8) return;

    uint qBase = head * D;
    uint kBase = H * D + qBase;
    uint vBase = 2 * H * D + qBase;
    uint stateBase = head * D * D;

    threadgroup float tg_q[256];
    threadgroup float tg_k[256];
    threadgroup float tg_v[256];
    threadgroup float tg_z[256];
    threadgroup float tg_out[256];
    threadgroup float partial0[8];
    threadgroup float partial1[8];
    threadgroup float tg_q_scale;
    threadgroup float tg_k_scale;
    threadgroup float tg_beta;
    threadgroup float tg_decay;
    threadgroup float tg_rms_scale;

    // Phase 0: conv1d update + SiLU for this head's Q, K, V channels.
    for (uint i = tid; i < 3 * D; i += tgs) {
        bool isQ = i < D;
        bool isK = !isQ && i < 2 * D;
        uint localIdx = isQ ? i : (isK ? i - D : i - 2 * D);
        uint ch = isQ ? (qBase + localIdx)
            : (isK ? (kBase + localIdx) : (vBase + localIdx));

        uint stateOffset = ch * convK;
        uint weightOffset = ch * convK;
        float cached[8];
        for (uint k = 0; k < convK - 1; k++) {
            cached[k] = float(convState[stateOffset + k + 1]);
        }
        cached[convK - 1] = float(qkv[ch]);

        float acc = 0.0f;
        for (uint k = 0; k < convK; k++) {
            convState[stateOffset + k] = half(cached[k]);
            acc += cached[k] * float(convWeight[weightOffset + k]);
        }
        float activated = acc / (1.0f + exp(-acc));
        if (isQ) {
            tg_q[localIdx] = activated;
        } else if (isK) {
            tg_k[localIdx] = activated;
        } else {
            tg_v[localIdx] = activated;
        }
    }
    for (uint i = tid; i < D; i += tgs) {
        tg_z[i] = float(z_proj[qBase + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: L2 normalize Q and K in threadgroup memory, and compute gates.
    float qSumSq = 0.0f;
    float kSumSq = 0.0f;
    for (uint i = tid; i < D; i += tgs) {
        float q = tg_q[i];
        float k = tg_k[i];
        qSumSq += q * q;
        kSumSq += k * k;
    }
    qSumSq = simd_sum(qSumSq);
    kSumSq = simd_sum(kSumSq);
    if (simd_lane == 0) {
        partial0[simd_group] = qSumSq;
        partial1[simd_group] = kSumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float totalQ = 0.0f;
        float totalK = 0.0f;
        for (uint s = 0; s < simdCount; s++) {
            totalQ += partial0[s];
            totalK += partial1[s];
        }
        tg_q_scale = 1.0f / max(sqrt(totalQ), 1e-6f);
        tg_k_scale = 1.0f / max(sqrt(totalK), 1e-6f);

        float b = float(b_proj[head]);
        tg_beta = 1.0f / (1.0f + exp(-b));

        float a = float(a_proj[head]);
        float al = float(a_log[head]);
        float db = float(dt_bias[head]);
        float x_sp = a + db;
        float sp = (x_sp > 20.0f) ? x_sp : log(1.0f + exp(x_sp));
        tg_decay = exp(-exp(al) * sp);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < D; i += tgs) {
        tg_q[i] *= tg_q_scale * headScale;
        tg_k[i] *= tg_k_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float keyReg[8];
    float queryReg[8];
    for (uint i = 0; i < elemsPerLane; i++) {
        uint dkIdx = i * 32 + simd_lane;
        keyReg[i] = tg_k[dkIdx];
        queryReg[i] = tg_q[dkIdx];
    }

    // Phase 2: MLX-style recurrence, processing simdCount output rows per pass.
    for (uint rowBase = 0; rowBase < D; rowBase += simdCount) {
        uint row = rowBase + simd_group;
        if (row >= D) continue;

        uint rowStateBase = stateBase + row * D;
        float stateReg[8];
        float kvMem = 0.0f;
        for (uint i = 0; i < elemsPerLane; i++) {
            uint dkIdx = i * 32 + simd_lane;
            stateReg[i] = float(state[rowStateBase + dkIdx]) * tg_decay;
            kvMem += stateReg[i] * keyReg[i];
        }
        kvMem = simd_sum(kvMem);

        float delta = (tg_v[row] - kvMem) * tg_beta;
        float out = 0.0f;
        for (uint i = 0; i < elemsPerLane; i++) {
            uint dkIdx = i * 32 + simd_lane;
            stateReg[i] += keyReg[i] * delta;
            out += stateReg[i] * queryReg[i];
            state[rowStateBase + dkIdx] = half(stateReg[i]);
        }
        out = simd_sum(out);
        if (simd_lane == 0) {
            tg_out[row] = out;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: gated RMS norm and final output write.
    float sumSq = 0.0f;
    for (uint i = tid; i < D; i += tgs) {
        float v = tg_out[i];
        sumSq += v * v;
    }
    sumSq = simd_sum(sumSq);
    if (simd_lane == 0) {
        partial0[simd_group] = sumSq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < simdCount; s++) {
            total += partial0[s];
        }
        tg_rms_scale = rsqrt(total / float(D) + rmsEps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < D; i += tgs) {
        float raw = tg_out[i];
        float z = tg_z[i];
        float silu_z = z / (1.0f + exp(-z));
        output[qBase + i] = half(raw * tg_rms_scale * float(normWeight[i]) * silu_z);
    }
}

// ─── Split decode recurrence core with MLX-style tiling ───
// Expects qkv to already contain conv1d+SiLU output with Q and K L2-normalized.
// State layout is [H, Dv, Dk] row-major, matching one contiguous Dk row per output value.
// Dispatch: threads=(32, headDim, numHeads), threadgroup=(32, 4, 1).
// Each simdgroup owns one output row and keeps Dk/32 values in registers.

kernel void deltanet_recurrence_mlx_decode(
    device half*        state      [[buffer(0)]],  // [H, Dv, Dk] FP16
    device const half*  qkv        [[buffer(1)]],  // [2 * Hqk * D + Hv * D]
    device const half*  b_proj     [[buffer(2)]],  // [Hv]
    device const half*  a_proj     [[buffer(3)]],  // [Hv]
    device const half*  a_log      [[buffer(4)]],  // [Hv]
    device const half*  dt_bias    [[buffer(5)]],  // [Hv]
    device half*        output     [[buffer(6)]],  // [Hv, D]
    constant uint&      headDim    [[buffer(7)]],  // D
    constant float&     headScale  [[buffer(8)]],  // 1/sqrt(D)
    constant uint&      valueHeads [[buffer(9)]],  // Hv
    constant uint&      qkHeads    [[buffer(10)]], // Hqk
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint head = gid.z;
    uint dvIdx = gid.y;
    if (head >= valueHeads || dvIdx >= headDim) return;

    // Known DeltaNet configs use headDim <= 256 and headDim % 32 == 0.
    uint elemsPerLane = headDim / 32;
    if (elemsPerLane == 0 || elemsPerLane > 8) return;

    uint repeatFactor = valueHeads / qkHeads;
    uint qkHead = head / repeatFactor;
    uint qBase = qkHead * headDim;
    uint kBase = qkHeads * headDim + qBase;
    uint vBase = 2 * qkHeads * headDim + head * headDim;
    uint stateBase = (head * headDim + dvIdx) * headDim;

    float stateReg[8];
    float keyReg[8];
    float queryReg[8];
    for (uint i = 0; i < elemsPerLane; i++) {
        uint dkIdx = lane * elemsPerLane + i;
        stateReg[i] = float(state[stateBase + dkIdx]);
        keyReg[i] = float(qkv[kBase + dkIdx]);
        queryReg[i] = float(qkv[qBase + dkIdx]);
    }

    threadgroup float tgDecay = 0.0f;
    threadgroup float tgBeta = 0.0f;
    if (lid.x == 0 && lid.y == 0) {
        tgBeta = float(smelt_decode_sigmoid_staged_half(b_proj[head]));

        half xh = a_proj[head] + dt_bias[head];
        float al = float(a_log[head]);
        float sp = float(smelt_decode_softplus_staged_half(xh));
        tgDecay = float(half(metal::precise::exp(-metal::precise::exp(al) * sp)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float kvMem = 0.0f;
    for (uint i = 0; i < elemsPerLane; i++) {
        stateReg[i] *= tgDecay;
        kvMem += stateReg[i] * keyReg[i];
    }
    kvMem = simd_sum(kvMem);

    float delta = (float(qkv[vBase + dvIdx]) - kvMem) * tgBeta;

    float out = 0.0f;
    for (uint i = 0; i < elemsPerLane; i++) {
        stateReg[i] += keyReg[i] * delta;
        out += stateReg[i] * queryReg[i];
    }
    out = simd_sum(out);

    if (lane == 0) {
        output[head * headDim + dvIdx] = half(out);
    }

    for (uint i = 0; i < elemsPerLane; i++) {
        uint dkIdx = lane * elemsPerLane + i;
        state[stateBase + dkIdx] = half(stateReg[i]);
    }
}

kernel void deltanet_recurrence_mlx_decode_d128_h16(
    device half*        state   [[buffer(0)]],  // [16, 128, 128] FP16
    device const half*  qkv     [[buffer(1)]],  // [3 * 16 * 128]
    device const half*  b_proj  [[buffer(2)]],  // [16]
    device const half*  a_proj  [[buffer(3)]],  // [16]
    device const half*  a_log   [[buffer(4)]],  // [16]
    device const half*  dt_bias [[buffer(5)]],  // [16]
    device half*        output  [[buffer(6)]],  // [16, 128]
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 128;
    constexpr uint H = 16;

    uint head = gid.z;
    uint dvIdx = gid.y;

    uint qBase = head * D;
    uint kBase = H * D + qBase;
    uint vBase = 2 * H * D + qBase;
    uint stateBase = (head * D + dvIdx) * D;

    uint dk0 = lane * 4;
    uint dk1 = lane * 4 + 1;
    uint dk2 = lane * 4 + 2;
    uint dk3 = lane * 4 + 3;

    float s0 = float(state[stateBase + dk0]);
    float s1 = float(state[stateBase + dk1]);
    float s2 = float(state[stateBase + dk2]);
    float s3 = float(state[stateBase + dk3]);

    float k0 = float(qkv[kBase + dk0]);
    float k1 = float(qkv[kBase + dk1]);
    float k2 = float(qkv[kBase + dk2]);
    float k3 = float(qkv[kBase + dk3]);

    float q0 = float(qkv[qBase + dk0]);
    float q1 = float(qkv[qBase + dk1]);
    float q2 = float(qkv[qBase + dk2]);
    float q3 = float(qkv[qBase + dk3]);

    threadgroup float tgDecay = 0.0f;
    threadgroup float tgBeta = 0.0f;
    if (lid.x == 0 && lid.y == 0) {
        tgBeta = float(smelt_decode_sigmoid_staged_half(b_proj[head]));

        half xh = a_proj[head] + dt_bias[head];
        float al = float(a_log[head]);
        float sp = float(smelt_decode_softplus_staged_half(xh));
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

    float delta = (float(qkv[vBase + dvIdx]) - kvMem) * tgBeta;

    s0 += k0 * delta;
    s1 += k1 * delta;
    s2 += k2 * delta;
    s3 += k3 * delta;

    float out = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
    out = simd_sum(out);

    if (lane == 0) {
        output[head * D + dvIdx] = half(out);
    }

    state[stateBase + dk0] = half(s0);
    state[stateBase + dk1] = half(s1);
    state[stateBase + dk2] = half(s2);
    state[stateBase + dk3] = half(s3);
}

kernel void deltanet_recurrence_mlx_decode_d128_h32_qk16(
    device half*        state   [[buffer(0)]],  // [32, 128, 128] FP16
    device const half*  qkv     [[buffer(1)]],  // [8192]
    device const half*  b_proj  [[buffer(2)]],  // [32]
    device const half*  a_proj  [[buffer(3)]],  // [32]
    device const half*  a_log   [[buffer(4)]],  // [32]
    device const half*  dt_bias [[buffer(5)]],  // [32]
    device half*        output  [[buffer(6)]],  // [32, 128]
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 128;
    constexpr uint Hqk = 16;

    uint head = gid.z;
    uint dvIdx = gid.y;
    uint qkHead = head >> 1;

    uint qBase = qkHead * D;
    uint kBase = Hqk * D + qBase;
    uint vBase = 2 * Hqk * D + head * D;
    uint stateBase = (head * D + dvIdx) * D;

    uint dk0 = lane * 4;
    uint dk1 = lane * 4 + 1;
    uint dk2 = lane * 4 + 2;
    uint dk3 = lane * 4 + 3;

    float s0 = float(state[stateBase + dk0]);
    float s1 = float(state[stateBase + dk1]);
    float s2 = float(state[stateBase + dk2]);
    float s3 = float(state[stateBase + dk3]);

    float k0 = float(qkv[kBase + dk0]);
    float k1 = float(qkv[kBase + dk1]);
    float k2 = float(qkv[kBase + dk2]);
    float k3 = float(qkv[kBase + dk3]);

    float q0 = float(qkv[qBase + dk0]);
    float q1 = float(qkv[qBase + dk1]);
    float q2 = float(qkv[qBase + dk2]);
    float q3 = float(qkv[qBase + dk3]);

    threadgroup float tgDecay = 0.0f;
    threadgroup float tgBeta = 0.0f;
    if (lid.x == 0 && lid.y == 0) {
        tgBeta = float(smelt_decode_sigmoid_staged_half(b_proj[head]));

        half xh = a_proj[head] + dt_bias[head];
        float al = float(a_log[head]);
        float sp = float(smelt_decode_softplus_staged_half(xh));
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

    float delta = (float(qkv[vBase + dvIdx]) - kvMem) * tgBeta;

    s0 += k0 * delta;
    s1 += k1 * delta;
    s2 += k2 * delta;
    s3 += k3 * delta;

    float out = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
    out = simd_sum(out);

    if (lane == 0) {
        output[head * D + dvIdx] = half(out);
    }

    state[stateBase + dk0] = half(s0);
    state[stateBase + dk1] = half(s1);
    state[stateBase + dk2] = half(s2);
    state[stateBase + dk3] = half(s3);
}

// Capability specialization for DeltaNet cells with D=128, 48 value heads,
// and 16 Q/K heads (three value heads share each Q/K head). The storage ABI,
// launch geometry, and arithmetic order match the generic recurrence exactly;
// only shape-derived address/control values become compile-time constants.
kernel void deltanet_recurrence_mlx_decode_d128_h48_qk16(
    device half*        state   [[buffer(0)]],  // [48, 128, 128] FP16
    device const half*  qkv     [[buffer(1)]],  // [10240]
    device const half*  b_proj  [[buffer(2)]],  // [48]
    device const half*  a_proj  [[buffer(3)]],  // [48]
    device const half*  a_log   [[buffer(4)]],  // [48]
    device const half*  dt_bias [[buffer(5)]],  // [48]
    device half*        output  [[buffer(6)]],  // [48, 128]
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 128;
    constexpr uint Hqk = 16;

    uint head = gid.z;
    uint dvIdx = gid.y;
    uint elemsPerLane = D / 32;
    uint qkHead = head / 3;

    uint qBase = qkHead * D;
    uint kBase = Hqk * D + qBase;
    uint vBase = 2 * Hqk * D + head * D;
    uint stateBase = (head * D + dvIdx) * D;

    float stateReg[8];
    float keyReg[8];
    float queryReg[8];
    for (uint i = 0; i < elemsPerLane; i++) {
        uint dkIdx = lane * elemsPerLane + i;
        stateReg[i] = float(state[stateBase + dkIdx]);
        keyReg[i] = float(qkv[kBase + dkIdx]);
        queryReg[i] = float(qkv[qBase + dkIdx]);
    }

    threadgroup float tgDecay = 0.0f;
    threadgroup float tgBeta = 0.0f;
    if (lid.x == 0 && lid.y == 0) {
        tgBeta = float(smelt_decode_sigmoid_staged_half(b_proj[head]));

        half xh = a_proj[head] + dt_bias[head];
        float al = float(a_log[head]);
        float sp = float(smelt_decode_softplus_staged_half(xh));
        tgDecay = float(half(metal::precise::exp(-metal::precise::exp(al) * sp)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float kvMem = 0.0f;
    for (uint i = 0; i < elemsPerLane; i++) {
        stateReg[i] *= tgDecay;
        kvMem += stateReg[i] * keyReg[i];
    }
    kvMem = simd_sum(kvMem);

    float delta = (float(qkv[vBase + dvIdx]) - kvMem) * tgBeta;

    float out = 0.0f;
    for (uint i = 0; i < elemsPerLane; i++) {
        stateReg[i] += keyReg[i] * delta;
        out += stateReg[i] * queryReg[i];
    }
    out = simd_sum(out);

    if (lane == 0) {
        output[head * D + dvIdx] = half(out);
    }

    for (uint i = 0; i < elemsPerLane; i++) {
        uint dkIdx = lane * elemsPerLane + i;
        state[stateBase + dkIdx] = half(stateReg[i]);
    }
}

// ─── Fused gates: beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias) ───
// Dispatch: H threads (one per head)

kernel void compute_gates(
    device const half*  b_proj   [[buffer(0)]],  // [H] — beta input
    device const half*  a_proj   [[buffer(1)]],  // [H] — alpha input
    device const half*  a_log    [[buffer(2)]],  // [H] — log decay param
    device const half*  dt_bias  [[buffer(3)]],  // [H] — dt bias param
    device half*        beta_out [[buffer(4)]],  // [H] — sigmoid(b)
    device half*        g_out    [[buffer(5)]],  // [H] — decay gate
    constant uint&      numHeads [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= numHeads) return;

    // Beta = sigmoid(b)
    float b = float(b_proj[tid]);
    beta_out[tid] = half(1.0f / (1.0f + exp(-b)));

    // G = -exp(A_log) * softplus(a + dt_bias)
    float a = float(a_proj[tid]);
    float al = float(a_log[tid]);
    float db = float(dt_bias[tid]);
    float x_sp = a + db;
    float sp = (x_sp > 20.0f) ? x_sp : log(1.0f + exp(x_sp));
    g_out[tid] = half(-exp(al) * sp);
}
