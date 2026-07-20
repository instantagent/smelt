#include <metal_stdlib>
using namespace metal;

// ─── RoPE (Rotary Position Embeddings) ───
// Applied to Q [numHeads, headDim] and K [numKVHeads, headDim].
// Only the first ropeDim (64) elements are rotated; rest pass through.
// rotate_half: swap pairs and negate: [x0,x1,x2,x3,...] → [-x1,x0,-x3,x2,...]
// result = x * cos + rotate_half(x) * sin
// Dispatch: numHeads * headDim threads

static inline void agent_rope_trig(
    uint pair,
    uint halfRope,
    uint position,
    float baseLog2,
    uint mathMode,
    device const half* cos_val,
    device const half* sin_val,
    uint c0Index,
    uint c1Index,
    thread float& c0,
    thread float& s0,
    thread float& c1,
    thread float& s1
) {
    if (mathMode == 1) {
        // Match MLX fast RoPE: derive the inverse frequency and trig values
        // in float on the GPU. An fp16 lookup table cannot reproduce this
        // boundary for arbitrary fp16 activations.
        float d = float(pair) / float(halfRope);
        float invFreq = metal::exp2(-d * baseLog2);
        float angle = float(position) * invFreq;
        float c = metal::fast::cos(angle);
        float s = metal::fast::sin(angle);
        c0 = c;
        s0 = s;
        c1 = c;
        s1 = s;
        return;
    }

    c0 = float(cos_val[c0Index]);
    s0 = float(sin_val[c0Index]);
    c1 = float(cos_val[c1Index]);
    s1 = float(sin_val[c1Index]);
}

kernel void apply_rope(
    device half*       qk      [[buffer(0)]],  // [H, D] — modified in place
    device const half* cos_val [[buffer(1)]],  // [ropeDim]
    device const half* sin_val [[buffer(2)]],  // [ropeDim]
    constant uint&     headDim [[buffer(3)]],  // D = 128
    constant uint&     ropeDim [[buffer(4)]],  // 64
    constant uint&     numHeads [[buffer(5)]],
    constant uint&     ropeLayout [[buffer(6)]], // 0 = adjacent pairs, 1 = split-half, 2 = proportional split-half
    constant uint&     position [[buffer(7)]],
    constant float&    baseLog2 [[buffer(8)]],
    constant uint&     mathMode [[buffer(9)]], // 0 = table, 1 = MLX-compatible analytic
    uint tid [[thread_position_in_grid]]
) {
    uint total = numHeads * headDim;
    if (tid >= total) return;

    uint head = tid / headDim;
    uint dim = tid % headDim;

    // One thread per pair — eliminates the in-place read/write race that existed
    // when two threads shared a pair. Only first ropeDim/2 threads per head work.
    uint halfRope = ropeDim / 2;
    if (dim >= halfRope) return;

    uint offset = head * headDim;
    uint d0;
    uint d1;
    uint c0Index;
    uint c1Index;
    if (ropeLayout == 1) {
        d0 = dim;
        d1 = dim + halfRope;
        c0Index = d0;
        c1Index = d1;
    } else if (ropeLayout == 2) {
        // Gemma full-attention proportional RoPE rotates the active prefix
        // against the midpoint of the full head, while the compact table only
        // stores the active rotary frequencies.
        d0 = dim;
        d1 = dim + headDim / 2;
        c0Index = dim;
        c1Index = dim + halfRope;
    } else {
        d0 = dim * 2;
        d1 = d0 + 1;
        c0Index = d0;
        c1Index = d1;
    }

    // Read both elements before writing either
    float x0 = float(qk[offset + d0]);
    float x1 = float(qk[offset + d1]);

    float c0, s0, c1, s1;
    agent_rope_trig(
        dim, halfRope, position, baseLog2, mathMode,
        cos_val, sin_val, c0Index, c1Index,
        c0, s0, c1, s1
    );

    // rotate_half: result[2i] = x[2i]*cos - x[2i+1]*sin
    //              result[2i+1] = x[2i+1]*cos + x[2i]*sin
    qk[offset + d0] = half(x0 * c0 - x1 * s0);
    // Preserve the reference source order. Although addition is
    // mathematically commutative, the compiler may contract the first
    // product with the add; swapping the operands moves fp16 boundaries.
    qk[offset + d1] = half(x0 * s1 + x1 * c1);
}

// ─── KV cache update ───
// Write new K/V vectors into cache at position `pos`.
// K_new: [numKVHeads, headDim], cache: [numKVHeads, maxSeqLen, headDim]
// Dispatch: numKVHeads * headDim threads

kernel void kv_cache_update(
    device half*       cache    [[buffer(0)]],  // [numKVHeads, cacheSeqCapacity, headDim]
    device const half* new_kv   [[buffer(1)]],  // [numKVHeads, headDim]
    constant uint&     cacheSeqCapacity [[buffer(2)]],
    constant uint&     headDim  [[buffer(3)]],  // 128
    constant uint&     pos      [[buffer(4)]],  // current position
    constant uint&     numHeads [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = numHeads * headDim;
    if (tid >= total) return;

    uint head = tid / headDim;
    uint dim = tid % headDim;

    // cache[head, pos, dim] = new_kv[head, dim]
    cache[head * cacheSeqCapacity * headDim + pos * headDim + dim] = new_kv[tid];
}

kernel void rope_kv_cache_update(
    device half*       cache    [[buffer(0)]],  // [numKVHeads, cacheSeqCapacity, headDim]
    device const half* new_kv   [[buffer(1)]],  // [numKVHeads, headDim]
    device const half* cos_val  [[buffer(2)]],  // [ropeDim]
    device const half* sin_val  [[buffer(3)]],  // [ropeDim]
    constant uint&     cacheSeqCapacity [[buffer(4)]],
    constant uint&     headDim [[buffer(5)]],
    constant uint&     pos [[buffer(6)]],
    constant uint&     numHeads [[buffer(7)]],
    constant uint&     ropeDim [[buffer(8)]],
    constant uint&     ropeLayout [[buffer(9)]],
    constant float&    baseLog2 [[buffer(10)]],
    constant uint&     mathMode [[buffer(11)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = numHeads * headDim;
    if (tid >= total) return;

    uint head = tid / headDim;
    uint dim = tid % headDim;
    uint headBase = head * headDim;
    uint cacheBase = head * cacheSeqCapacity * headDim + pos * headDim;
    uint halfRope = ropeDim / 2;

    if (dim < halfRope) {
        uint d0;
        uint d1;
        uint c0Index;
        uint c1Index;
        if (ropeLayout == 1) {
            d0 = dim;
            d1 = dim + halfRope;
            c0Index = d0;
            c1Index = d1;
        } else if (ropeLayout == 2) {
            d0 = dim;
            d1 = dim + headDim / 2;
            c0Index = dim;
            c1Index = dim + halfRope;
        } else {
            d0 = dim * 2;
            d1 = d0 + 1;
            c0Index = d0;
            c1Index = d1;
        }

        float x0 = float(new_kv[headBase + d0]);
        float x1 = float(new_kv[headBase + d1]);
        float c0, s0, c1, s1;
        agent_rope_trig(
            dim, halfRope, pos, baseLog2, mathMode,
            cos_val, sin_val, c0Index, c1Index,
            c0, s0, c1, s1
        );
        cache[cacheBase + d0] = half(x0 * c0 - x1 * s0);
        cache[cacheBase + d1] = half(x1 * c1 + x0 * s1);
        return;
    }

    bool rotatedSecond = false;
    if (ropeLayout == 2) {
        rotatedSecond = dim >= headDim / 2 && dim < headDim / 2 + halfRope;
    } else {
        rotatedSecond = dim < ropeDim;
    }
    if (!rotatedSecond) {
        cache[cacheBase + dim] = new_kv[headBase + dim];
    }
}

// ─── Single-token softmax attention with GQA ───
// Q: [numQHeads, headDim], K_cache: [numKVHeads, seqLen, headDim], V_cache: same
// GQA: numQHeads / numKVHeads query heads share each KV head
// Output: [numQHeads, headDim]
// Dispatch: numQHeads threadgroups

kernel void attention_decode(
    device const half* query    [[buffer(0)]],   // [numQHeads, headDim]
    device const half* k_cache  [[buffer(1)]],   // [numKVHeads, cacheSeqCapacity, headDim]
    device const half* v_cache  [[buffer(2)]],   // [numKVHeads, cacheSeqCapacity, headDim]
    device const half* attn_mask [[buffer(3)]],  // unused in decode; seqLen is already causal
    device half*       output   [[buffer(4)]],   // [numQHeads, headDim]
    constant uint&     headDim  [[buffer(5)]],   // 128
    constant uint&     cacheSeqCapacity [[buffer(6)]],
    constant uint&     seqLen   [[buffer(7)]],   // actual sequence length (pos + 1)
    constant uint&     numKVHeads [[buffer(8)]],  // 2
    constant float&    scale    [[buffer(9)]],  // 1/sqrt(headDim)
    constant uint&     slidingWindow [[buffer(10)]],  // 0 = full causal
    uint3 tgid      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint3 tgs_v     [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint3 tgCount   [[threadgroups_per_grid]]
) {
    uint qHead = tgid.x;
    uint tgs = tgs_v.x;
    uint numQHeads = tgCount.x;
    uint gqaRatio = numQHeads / numKVHeads;
    uint kvHead = qHead / gqaRatio;
    uint qOffset = qHead * headDim;
    uint kvStride = cacheSeqCapacity * headDim;  // stride between KV heads
    threadgroup float qShared[512];

    // Generic decode aliases query and output on the same slot. Snapshot Q
    // before any thread writes output so upper dimensions cannot read a
    // partially-overwritten query vector.
    for (uint d = tid; d < headDim; d += tgs) {
        qShared[d] = float(query[qOffset + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint seqStart = 0;
    if (slidingWindow > 0 && slidingWindow < seqLen) {
        seqStart = seqLen - slidingWindow;
    }

    float maxScore = -INFINITY;
    for (uint s = seqStart + tid; s < seqLen; s += tgs) {
        float dot = 0.0f;
        for (uint d = 0; d < headDim; d++) {
            dot += qShared[d] * float(k_cache[kvHead * kvStride + s * headDim + d]);
        }
        dot *= scale;
        maxScore = max(maxScore, dot);
    }

    // Reduce max across threads
    maxScore = simd_max(maxScore);
    threadgroup float tgMax[8];
    if (simd_lane == 0) { tgMax[simd_group] = maxScore; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = -INFINITY;
        for (uint s = 0; s < tgs / 32; s++) { m = max(m, tgMax[s]); }
        tgMax[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxScore = tgMax[0];

    float sumExp = 0.0f;
    for (uint s = seqStart + tid; s < seqLen; s += tgs) {
        float dot = 0.0f;
        for (uint d = 0; d < headDim; d++) {
            dot += qShared[d] * float(k_cache[kvHead * kvStride + s * headDim + d]);
        }
        sumExp += exp(dot * scale - maxScore);
    }
    sumExp = simd_sum(sumExp);
    threadgroup float tgSum[8];
    if (simd_lane == 0) { tgSum[simd_group] = sumExp; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0;
        for (uint s = 0; s < tgs / 32; s++) { total += tgSum[s]; }
        tgSum[0] = 1.0f / total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invSum = tgSum[0];

    for (uint d = tid; d < headDim; d += tgs) {
        float acc = 0.0f;
        for (uint s = seqStart; s < seqLen; s++) {
            float dot = 0.0f;
            for (uint qd = 0; qd < headDim; qd++) {
                dot += qShared[qd]
                    * float(k_cache[kvHead * kvStride + s * headDim + qd]);
            }
            float weight = exp(dot * scale - maxScore) * invSum;
            acc += weight * float(v_cache[kvHead * kvStride + s * headDim + d]);
        }
        output[qOffset + d] = half(acc);
    }
}

// Generic decode attention with Gemma-style score softcapping:
//   score = softcap * tanh((QK^T * scale) / softcap)
// applied before softmax.
kernel void attention_decode_softcap(
    device const half* query    [[buffer(0)]],   // [numQHeads, headDim]
    device const half* k_cache  [[buffer(1)]],   // [numKVHeads, cacheSeqCapacity, headDim]
    device const half* v_cache  [[buffer(2)]],   // [numKVHeads, cacheSeqCapacity, headDim]
    device const half* attn_mask [[buffer(3)]],  // unused in decode; seqLen is already causal
    device half*       output   [[buffer(4)]],   // [numQHeads, headDim]
    constant uint&     headDim  [[buffer(5)]],
    constant uint&     cacheSeqCapacity [[buffer(6)]],
    constant uint&     seqLen   [[buffer(7)]],
    constant uint&     numKVHeads [[buffer(8)]],
    constant float&    scale    [[buffer(9)]],
    constant uint&     slidingWindow [[buffer(10)]],
    constant float&    softcap  [[buffer(11)]],
    uint3 tgid      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint3 tgs_v     [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint3 tgCount   [[threadgroups_per_grid]]
) {
    uint qHead = tgid.x;
    uint tgs = tgs_v.x;
    uint numQHeads = tgCount.x;
    uint gqaRatio = numQHeads / numKVHeads;
    uint kvHead = qHead / gqaRatio;
    uint qOffset = qHead * headDim;
    uint kvStride = cacheSeqCapacity * headDim;
    threadgroup float qShared[512];

    for (uint d = tid; d < headDim; d += tgs) {
        qShared[d] = float(query[qOffset + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint seqStart = 0;
    if (slidingWindow > 0 && slidingWindow < seqLen) {
        seqStart = seqLen - slidingWindow;
    }

    float maxScore = -INFINITY;
    for (uint s = seqStart + tid; s < seqLen; s += tgs) {
        float dot = 0.0f;
        for (uint d = 0; d < headDim; d++) {
            dot += qShared[d] * float(k_cache[kvHead * kvStride + s * headDim + d]);
        }
        dot *= scale;
        dot = softcap * tanh(dot / softcap);
        maxScore = max(maxScore, dot);
    }

    maxScore = simd_max(maxScore);
    threadgroup float tgMax[8];
    if (simd_lane == 0) { tgMax[simd_group] = maxScore; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = -INFINITY;
        for (uint s = 0; s < tgs / 32; s++) { m = max(m, tgMax[s]); }
        tgMax[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxScore = tgMax[0];

    float sumExp = 0.0f;
    for (uint s = seqStart + tid; s < seqLen; s += tgs) {
        float dot = 0.0f;
        for (uint d = 0; d < headDim; d++) {
            dot += qShared[d] * float(k_cache[kvHead * kvStride + s * headDim + d]);
        }
        dot = softcap * tanh((dot * scale) / softcap);
        sumExp += exp(dot - maxScore);
    }
    sumExp = simd_sum(sumExp);
    threadgroup float tgSum[8];
    if (simd_lane == 0) { tgSum[simd_group] = sumExp; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0;
        for (uint s = 0; s < tgs / 32; s++) { total += tgSum[s]; }
        tgSum[0] = 1.0f / total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invSum = tgSum[0];

    for (uint d = tid; d < headDim; d += tgs) {
        float acc = 0.0f;
        for (uint s = seqStart; s < seqLen; s++) {
            float dot = 0.0f;
            for (uint qd = 0; qd < headDim; qd++) {
                dot += qShared[qd]
                    * float(k_cache[kvHead * kvStride + s * headDim + qd]);
            }
            dot = softcap * tanh((dot * scale) / softcap);
            float weight = exp(dot - maxScore) * invSum;
            acc += weight * float(v_cache[kvHead * kvStride + s * headDim + d]);
        }
        output[qOffset + d] = half(acc);
    }
}

// Qwen 3.5 2B decode specialization:
//   qHeads=8, kvHeads=2, headDim=256, maxSeq=256, gqa=4, scale=1/16.
// Query and output alias the same buffer. The causal mask buffer is not needed
// because decode only attends over [0, seqLen).
kernel void attention_decode_d256_h8_kv2(
    device half*       q_out    [[buffer(0)]],  // [8, 256] in-place
    device const half* k_cache  [[buffer(1)]],  // [2, cacheSeqCapacity, 256]
    device const half* v_cache  [[buffer(2)]],  // [2, cacheSeqCapacity, 256]
    constant uint&     seqLen   [[buffer(3)]],  // actual decode length (pos + 1)
    constant uint&     cacheSeqCapacity [[buffer(4)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint headDim = 256;
    constexpr uint headDim4 = headDim / 4;
    constexpr uint maxSeq = 256;
    constexpr uint gqaRatio = 4;
    constexpr float scaleLog2 = 0.0625f * M_LOG2E_F;

    uint kvHead = qHead / gqaRatio;
    uint qOffset = qHead * headDim;
    uint kvBase = kvHead * cacheSeqCapacity * headDim;

    device half4* out4 = reinterpret_cast<device half4*>(q_out + qOffset);
    const device half4* k4Base = reinterpret_cast<const device half4*>(k_cache + kvBase);
    const device half4* v4Base = reinterpret_cast<const device half4*>(v_cache + kvBase);

    threadgroup float4 qShared[headDim4];
    threadgroup float scores[maxSeq];
    threadgroup float partial[2];

    qShared[tid] = float4(out4[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float maxScore = -INFINITY;
    for (uint s = tid; s < seqLen; s += headDim4) {
        const device half4* k4 = k4Base + s * headDim4;
        float accDot = 0.0f;
        for (uint d4 = 0; d4 < headDim4; d4++) {
            accDot += dot(qShared[d4], float4(k4[d4]));
        }
        float score = accDot * scaleLog2;
        scores[s] = score;
        maxScore = max(maxScore, score);
    }

    maxScore = simd_max(maxScore);
    if (simd_lane == 0) { partial[simd_group] = maxScore; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        partial[0] = max(partial[0], partial[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scoreMax = partial[0];

    float sumExp = 0.0f;
    for (uint s = tid; s < seqLen; s += headDim4) {
        float e = fast::exp2(scores[s] - scoreMax);
        scores[s] = e;
        sumExp += e;
    }

    sumExp = simd_sum(sumExp);
    if (simd_lane == 0) { partial[simd_group] = sumExp; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        partial[0] = 1.0f / (partial[0] + partial[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invSum = partial[0];

    for (uint s = tid; s < seqLen; s += headDim4) {
        scores[s] *= invSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 acc = 0.0f;
    for (uint s = 0; s < seqLen; s++) {
        acc += scores[s] * float4(v4Base[s * headDim4 + tid]);
    }
    out4[tid] = half4(acc);
}

kernel void attention_decode_d256_h16_kv4(
    device half*       q_out    [[buffer(0)]],  // [16, 256] in-place
    device const half* k_cache  [[buffer(1)]],  // [4, cacheSeqCapacity, 256]
    device const half* v_cache  [[buffer(2)]],  // [4, cacheSeqCapacity, 256]
    constant uint&     seqLen   [[buffer(3)]],  // actual decode length (pos + 1)
    constant uint&     cacheSeqCapacity [[buffer(4)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint headDim = 256;
    constexpr uint headDim4 = headDim / 4;
    constexpr uint maxSeq = 256;
    constexpr uint gqaRatio = 4;
    constexpr float scaleLog2 = 0.0625f * M_LOG2E_F;

    uint kvHead = qHead / gqaRatio;
    uint qOffset = qHead * headDim;
    uint kvBase = kvHead * cacheSeqCapacity * headDim;

    device half4* out4 = reinterpret_cast<device half4*>(q_out + qOffset);
    const device half4* k4Base = reinterpret_cast<const device half4*>(k_cache + kvBase);
    const device half4* v4Base = reinterpret_cast<const device half4*>(v_cache + kvBase);

    threadgroup float4 qShared[headDim4];
    threadgroup float scores[maxSeq];
    threadgroup float partial[2];

    qShared[tid] = float4(out4[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float maxScore = -INFINITY;
    for (uint s = tid; s < seqLen; s += headDim4) {
        const device half4* k4 = k4Base + s * headDim4;
        float accDot = 0.0f;
        for (uint d4 = 0; d4 < headDim4; d4++) {
            accDot += dot(qShared[d4], float4(k4[d4]));
        }
        float score = accDot * scaleLog2;
        scores[s] = score;
        maxScore = max(maxScore, score);
    }

    maxScore = simd_max(maxScore);
    if (simd_lane == 0) { partial[simd_group] = maxScore; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        partial[0] = max(partial[0], partial[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scoreMax = partial[0];

    float sumExp = 0.0f;
    for (uint s = tid; s < seqLen; s += headDim4) {
        float e = fast::exp2(scores[s] - scoreMax);
        scores[s] = e;
        sumExp += e;
    }

    sumExp = simd_sum(sumExp);
    if (simd_lane == 0) { partial[simd_group] = sumExp; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        partial[0] = 1.0f / (partial[0] + partial[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invSum = partial[0];

    for (uint s = tid; s < seqLen; s += headDim4) {
        scores[s] *= invSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 acc = 0.0f;
    for (uint s = 0; s < seqLen; s++) {
        acc += scores[s] * float4(v4Base[s * headDim4 + tid]);
    }
    out4[tid] = half4(acc);
}

// Qwen 3.5/3.6 27B decode specialization:
//   qHeads=24, kvHeads=4, headDim=256, maxSeq=256, gqa=6, scale=1/16.
// This is the same reusable short-context vector attention brick as the
// H8/KV2 and H16/KV4 variants, with the geometry declared by the model.
kernel void attention_decode_d256_h24_kv4(
    device half*       q_out    [[buffer(0)]],  // [24, 256] in-place
    device const half* k_cache  [[buffer(1)]],  // [4, cacheSeqCapacity, 256]
    device const half* v_cache  [[buffer(2)]],  // [4, cacheSeqCapacity, 256]
    constant uint&     seqLen   [[buffer(3)]],  // actual decode length (pos + 1)
    constant uint&     cacheSeqCapacity [[buffer(4)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint headDim = 256;
    constexpr uint headDim4 = headDim / 4;
    constexpr uint maxSeq = 256;
    constexpr uint gqaRatio = 6;
    constexpr float scaleLog2 = 0.0625f * M_LOG2E_F;

    uint kvHead = qHead / gqaRatio;
    uint qOffset = qHead * headDim;
    uint kvBase = kvHead * cacheSeqCapacity * headDim;

    device half4* out4 = reinterpret_cast<device half4*>(q_out + qOffset);
    const device half4* k4Base = reinterpret_cast<const device half4*>(k_cache + kvBase);
    const device half4* v4Base = reinterpret_cast<const device half4*>(v_cache + kvBase);

    threadgroup float4 qShared[headDim4];
    threadgroup float scores[maxSeq];
    threadgroup float partial[2];

    qShared[tid] = float4(out4[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float maxScore = -INFINITY;
    for (uint s = tid; s < seqLen; s += headDim4) {
        const device half4* k4 = k4Base + s * headDim4;
        float accDot = 0.0f;
        for (uint d4 = 0; d4 < headDim4; d4++) {
            accDot += dot(qShared[d4], float4(k4[d4]));
        }
        float score = accDot * scaleLog2;
        scores[s] = score;
        maxScore = max(maxScore, score);
    }

    maxScore = simd_max(maxScore);
    if (simd_lane == 0) { partial[simd_group] = maxScore; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        partial[0] = max(partial[0], partial[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scoreMax = partial[0];

    float sumExp = 0.0f;
    for (uint s = tid; s < seqLen; s += headDim4) {
        float e = fast::exp2(scores[s] - scoreMax);
        scores[s] = e;
        sumExp += e;
    }

    sumExp = simd_sum(sumExp);
    if (simd_lane == 0) { partial[simd_group] = sumExp; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        partial[0] = 1.0f / (partial[0] + partial[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invSum = partial[0];

    for (uint s = tid; s < seqLen; s += headDim4) {
        scores[s] *= invSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 acc = 0.0f;
    for (uint s = 0; s < seqLen; s++) {
        acc += scores[s] * float4(v4Base[s * headDim4 + tid]);
    }
    out4[tid] = half4(acc);
}

// MLX-style decode attention specialization for the same Qwen geometry.
// This is a direct adaptation of MLX's sdpa_vector kernel body for:
//   qHeads=8, kvHeads=2, headDim=256, valueDim=256, maxSeq=256, gqa=4.
// Launch: threadgroups=(8, 1, 1), threadsPerThreadgroup=(1024, 1, 1).
kernel void attention_decode_d256_h8_kv2_sdpa(
    device half*       q_out    [[buffer(0)]],  // [8, 256] in-place
    device const half* k_cache  [[buffer(1)]],  // [2, cacheSeqCapacity, 256]
    device const half* v_cache  [[buffer(2)]],  // [2, cacheSeqCapacity, 256]
    constant uint&     seqLen   [[buffer(3)]],  // actual decode length (pos + 1)
    constant uint&     cacheSeqCapacity [[buffer(4)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]]
) {
    constexpr uint headDim = 256;
    constexpr uint gqaRatio = 4;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint qkPerThread = headDim / BD;
    constexpr uint vPerThread = headDim / BD;
    constexpr float scale = 0.0625f;

    const uint kvHead = qHead / gqaRatio;
    const uint qOffset = qHead * headDim;
    const uint kvBase = kvHead * cacheSeqCapacity * headDim;

    device const half* query = q_out + qOffset + simd_lane * qkPerThread;
    device const half* keys = k_cache + kvBase + simd_group * headDim + simd_lane * qkPerThread;
    device const half* values =
        v_cache + kvBase + simd_group * headDim + simd_lane * vPerThread;
    device half* out = q_out + qOffset + simd_group * vPerThread;

    thread float q[qkPerThread];
    thread float k[qkPerThread];
    thread float o[vPerThread];

    threadgroup float outputs[BN * BD];
    threadgroup float maxScores[BN];
    threadgroup float sumExpScores[BN];

    for (uint i = 0; i < qkPerThread; ++i) {
        q[i] = scale * float(query[i]);
    }
    for (uint i = 0; i < vPerThread; ++i) {
        o[i] = 0.0f;
    }

    float maxScore = -INFINITY;
    float sumExpScore = 0.0f;

    for (uint s = simd_group; s < seqLen; s += BN) {
        for (uint i = 0; i < qkPerThread; ++i) {
            k[i] = float(keys[i]);
        }

        float score = 0.0f;
        for (uint i = 0; i < qkPerThread; ++i) {
            score += q[i] * k[i];
        }
        score = simd_sum(score);

        float newMax = max(maxScore, score);
        float factor = fast::exp(maxScore - newMax);
        float expScore = fast::exp(score - newMax);

        maxScore = newMax;
        sumExpScore = sumExpScore * factor + expScore;

        for (uint i = 0; i < vPerThread; ++i) {
            o[i] = o[i] * factor + expScore * float(values[i]);
        }

        keys += BN * headDim;
        values += BN * headDim;
    }

    if (simd_lane == 0) {
        maxScores[simd_group] = maxScore;
        sumExpScores[simd_group] = sumExpScore;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    maxScore = maxScores[simd_lane];
    float newMax = simd_max(maxScore);
    float factor = fast::exp(maxScore - newMax);
    sumExpScore = simd_sum(sumExpScores[simd_lane] * factor);

    for (uint i = 0; i < vPerThread; ++i) {
        outputs[simd_lane * BD + simd_group] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_group * BD + simd_lane] * factor);
        if (sumExpScore != 0.0f) {
            o[i] /= sumExpScore;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lane == 0) {
        for (uint i = 0; i < vPerThread; ++i) {
            out[i] = half(o[i]);
        }
    }
}

kernel void attention_decode_d256_h16_kv4_sdpa(
    device half*       q_out    [[buffer(0)]],  // [16, 256] in-place
    device const half* k_cache  [[buffer(1)]],  // [4, cacheSeqCapacity, 256]
    device const half* v_cache  [[buffer(2)]],  // [4, cacheSeqCapacity, 256]
    constant uint&     seqLen   [[buffer(3)]],  // actual decode length (pos + 1)
    constant uint&     cacheSeqCapacity [[buffer(4)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]]
) {
    constexpr uint headDim = 256;
    constexpr uint gqaRatio = 4;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint qkPerThread = headDim / BD;
    constexpr uint vPerThread = headDim / BD;
    constexpr float scale = 0.0625f;

    const uint kvHead = qHead / gqaRatio;
    const uint qOffset = qHead * headDim;
    const uint kvBase = kvHead * cacheSeqCapacity * headDim;

    device const half* query = q_out + qOffset + simd_lane * qkPerThread;
    device const half* keys = k_cache + kvBase + simd_group * headDim + simd_lane * qkPerThread;
    device const half* values =
        v_cache + kvBase + simd_group * headDim + simd_lane * vPerThread;
    device half* out = q_out + qOffset + simd_group * vPerThread;

    thread float q[qkPerThread];
    thread float k[qkPerThread];
    thread float o[vPerThread];

    threadgroup float outputs[BN * BD];
    threadgroup float maxScores[BN];
    threadgroup float sumExpScores[BN];

    for (uint i = 0; i < qkPerThread; ++i) {
        q[i] = scale * float(query[i]);
    }
    for (uint i = 0; i < vPerThread; ++i) {
        o[i] = 0.0f;
    }

    float maxScore = -INFINITY;
    float sumExpScore = 0.0f;

    for (uint s = simd_group; s < seqLen; s += BN) {
        for (uint i = 0; i < qkPerThread; ++i) {
            k[i] = float(keys[i]);
        }

        float score = 0.0f;
        for (uint i = 0; i < qkPerThread; ++i) {
            score += q[i] * k[i];
        }
        score = simd_sum(score);

        float newMax = max(maxScore, score);
        float factor = fast::exp(maxScore - newMax);
        float expScore = fast::exp(score - newMax);

        maxScore = newMax;
        sumExpScore = sumExpScore * factor + expScore;

        for (uint i = 0; i < vPerThread; ++i) {
            o[i] = o[i] * factor + expScore * float(values[i]);
        }

        keys += BN * headDim;
        values += BN * headDim;
    }

    if (simd_lane == 0) {
        maxScores[simd_group] = maxScore;
        sumExpScores[simd_group] = sumExpScore;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    maxScore = maxScores[simd_lane];
    float newMax = simd_max(maxScore);
    float factor = fast::exp(maxScore - newMax);
    sumExpScore = simd_sum(sumExpScores[simd_lane] * factor);

    for (uint i = 0; i < vPerThread; ++i) {
        outputs[simd_lane * BD + simd_group] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_group * BD + simd_lane] * factor);
        if (sumExpScore != 0.0f) {
            o[i] /= sumExpScore;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lane == 0) {
        for (uint i = 0; i < vPerThread; ++i) {
            out[i] = half(o[i]);
        }
    }
}

// MLX-style long-context decode attention for the declared D256/H24/KV4
// geometry. The implementation is the same reusable SDPA vector brick as the
// H8/KV2 and H16/KV4 variants; only the package-declared GQA ratio differs.
// Launch: threadgroups=(24, 1, 1), threadsPerThreadgroup=(1024, 1, 1).
kernel void attention_decode_d256_h24_kv4_sdpa(
    device half*       q_out    [[buffer(0)]],  // [24, 256] in-place
    device const half* k_cache  [[buffer(1)]],  // [4, cacheSeqCapacity, 256]
    device const half* v_cache  [[buffer(2)]],  // [4, cacheSeqCapacity, 256]
    constant uint&     seqLen   [[buffer(3)]],  // actual decode length (pos + 1)
    constant uint&     cacheSeqCapacity [[buffer(4)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]]
) {
    constexpr uint headDim = 256;
    constexpr uint gqaRatio = 6;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint qkPerThread = headDim / BD;
    constexpr uint vPerThread = headDim / BD;
    constexpr float scale = 0.0625f;

    const uint kvHead = qHead / gqaRatio;
    const uint qOffset = qHead * headDim;
    const uint kvBase = kvHead * cacheSeqCapacity * headDim;

    device const half* query = q_out + qOffset + simd_lane * qkPerThread;
    device const half* keys = k_cache + kvBase + simd_group * headDim + simd_lane * qkPerThread;
    device const half* values =
        v_cache + kvBase + simd_group * headDim + simd_lane * vPerThread;
    device half* out = q_out + qOffset + simd_group * vPerThread;

    thread float q[qkPerThread];
    thread float k[qkPerThread];
    thread float o[vPerThread];

    threadgroup float outputs[BN * BD];
    threadgroup float maxScores[BN];
    threadgroup float sumExpScores[BN];

    for (uint i = 0; i < qkPerThread; ++i) {
        q[i] = scale * float(query[i]);
    }
    for (uint i = 0; i < vPerThread; ++i) {
        o[i] = 0.0f;
    }

    float maxScore = -INFINITY;
    float sumExpScore = 0.0f;

    for (uint s = simd_group; s < seqLen; s += BN) {
        for (uint i = 0; i < qkPerThread; ++i) {
            k[i] = float(keys[i]);
        }

        float score = 0.0f;
        for (uint i = 0; i < qkPerThread; ++i) {
            score += q[i] * k[i];
        }
        score = simd_sum(score);

        float newMax = max(maxScore, score);
        float factor = fast::exp(maxScore - newMax);
        float expScore = fast::exp(score - newMax);

        maxScore = newMax;
        sumExpScore = sumExpScore * factor + expScore;

        for (uint i = 0; i < vPerThread; ++i) {
            o[i] = o[i] * factor + expScore * float(values[i]);
        }

        keys += BN * headDim;
        values += BN * headDim;
    }

    if (simd_lane == 0) {
        maxScores[simd_group] = maxScore;
        sumExpScores[simd_group] = sumExpScore;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    maxScore = maxScores[simd_lane];
    float newMax = simd_max(maxScore);
    float factor = fast::exp(maxScore - newMax);
    sumExpScore = simd_sum(sumExpScores[simd_lane] * factor);

    for (uint i = 0; i < vPerThread; ++i) {
        outputs[simd_lane * BD + simd_group] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_group * BD + simd_lane] * factor);
        if (sumExpScore != 0.0f) {
            o[i] /= sumExpScore;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lane == 0) {
        for (uint i = 0; i < vPerThread; ++i) {
            out[i] = half(o[i]);
        }
    }
}

// Shape-generic MLX sdpa_vector brick for D=256 decode attention. Model
// identity is deliberately absent: qHeads/kvHeads and the cache geometry are
// package data, while the compiler admits only compatible no-mask/no-softcap
// attention semantics.
kernel void attention_decode_mlx_vector_d256(
    device half*       q_out    [[buffer(0)]],
    device const half* k_cache  [[buffer(1)]],
    device const half* v_cache  [[buffer(2)]],
    constant uint&     seqLen   [[buffer(3)]],
    constant uint&     cacheSeqCapacity [[buffer(4)]],
    constant uint&     qHeads   [[buffer(5)]],
    constant uint&     kvHeads  [[buffer(6)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 256;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint valuesPerLane = D / BD;
    constexpr float scale = 0.0625f;

    if (qHead >= qHeads || kvHeads == 0 || (qHeads % kvHeads) != 0) return;
    const uint gqa = qHeads / kvHeads;
    const uint kvHead = qHead / gqa;
    const uint qOffset = qHead * D;
    const uint kvBase = kvHead * cacheSeqCapacity * D;

    device const half* query = q_out + qOffset + simd_lane * valuesPerLane;
    device const half* keys =
        k_cache + kvBase + simd_group * D + simd_lane * valuesPerLane;
    device const half* values =
        v_cache + kvBase + simd_group * D + simd_lane * valuesPerLane;
    device half* out = q_out + qOffset + simd_group * valuesPerLane;

    thread float q[valuesPerLane];
    thread float k[valuesPerLane];
    thread float o[valuesPerLane];
    threadgroup float outputs[BN * BD];
    threadgroup float maxScores[BN];
    threadgroup float sumExpScores[BN];

    for (uint i = 0; i < valuesPerLane; ++i) {
        q[i] = scale * float(query[i]);
        o[i] = 0.0f;
    }

    float maxScore = -3.402823466e+38f;
    float sumExpScore = 0.0f;
    for (uint s = simd_group; s < seqLen; s += BN) {
        for (uint i = 0; i < valuesPerLane; ++i) k[i] = float(keys[i]);

        float score = 0.0f;
        for (uint i = 0; i < valuesPerLane; ++i) score += q[i] * k[i];
        score = simd_sum(score);

        const float newMax = max(maxScore, score);
        const float factor = fast::exp(maxScore - newMax);
        const float expScore = fast::exp(score - newMax);
        maxScore = newMax;
        sumExpScore = sumExpScore * factor + expScore;
        for (uint i = 0; i < valuesPerLane; ++i) {
            o[i] = o[i] * factor + expScore * float(values[i]);
        }
        keys += BN * D;
        values += BN * D;
    }

    if (simd_lane == 0) {
        maxScores[simd_group] = maxScore;
        sumExpScores[simd_group] = sumExpScore;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maxScore = maxScores[simd_lane];
    const float newMax = simd_max(maxScore);
    const float factor = fast::exp(maxScore - newMax);
    sumExpScore = simd_sum(sumExpScores[simd_lane] * factor);

    for (uint i = 0; i < valuesPerLane; ++i) {
        outputs[simd_lane * BD + simd_group] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_group * BD + simd_lane] * factor);
        o[i] = sumExpScore == 0.0f ? o[i] : (o[i] / sumExpScore);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (simd_lane == 0) {
        for (uint i = 0; i < valuesPerLane; ++i) out[i] = half(o[i]);
    }
}

// First half of MLX's long-key sdpa_vector_2pass topology. A threadgroup owns
// one KV head and one key block; its Y simdgroups independently cover every
// GQA query head. Partial values intentionally materialize as fp16 while the
// normalization statistics remain fp32, matching the frozen MLX graph.
kernel void attention_decode_mlx_vector_2pass_1_d256_b128(
    device const half* q       [[buffer(0)]],
    device const half* k_cache [[buffer(1)]],
    device const half* v_cache [[buffer(2)]],
    device half*       partials [[buffer(3)]],
    device float*      stats    [[buffer(4)]],
    constant uint&     seqLen   [[buffer(5)]],
    constant uint&     cacheSeqCapacity [[buffer(6)]],
    constant uint&     qHeads   [[buffer(7)]],
    constant uint&     kvHeads  [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid  [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 256;
    constexpr uint BLOCKS = 128;
    constexpr uint valuesPerLane = D / 32;
    constexpr float scale = 0.0625f;

    if (kvHeads == 0 || (qHeads % kvHeads) != 0) return;
    const uint gqa = qHeads / kvHeads;
    const uint kvHead = tgid.x;
    const uint block = tgid.z;
    const uint qWithinKV = lid.y;
    const uint qHead = kvHead * gqa + qWithinKV;
    if (kvHead >= kvHeads || block >= BLOCKS || qWithinKV >= gqa || qHead >= qHeads) return;

    device const half* query = q + qHead * D + simd_lane * valuesPerLane;
    device const half* keys = k_cache
        + kvHead * cacheSeqCapacity * D + block * D + simd_lane * valuesPerLane;
    device const half* values = v_cache
        + kvHead * cacheSeqCapacity * D + block * D + simd_lane * valuesPerLane;
    device half* out = partials
        + (qHead * BLOCKS + block) * D + simd_lane * valuesPerLane;

    thread float queryValues[valuesPerLane];
    thread float outputValues[valuesPerLane];
    for (uint i = 0; i < valuesPerLane; ++i) {
        queryValues[i] = scale * float(query[i]);
        outputValues[i] = 0.0f;
    }

    float maxScore = -3.402823466e+38f;
    float sumExpScore = 0.0f;
    for (uint key = block; key < seqLen; key += BLOCKS) {
        float score = 0.0f;
        for (uint i = 0; i < valuesPerLane; ++i) {
            score += queryValues[i] * float(keys[i]);
        }
        score = simd_sum(score);

        const float newMax = max(maxScore, score);
        const float factor = fast::exp(maxScore - newMax);
        const float expScore = fast::exp(score - newMax);
        maxScore = newMax;
        sumExpScore = sumExpScore * factor + expScore;
        for (uint i = 0; i < valuesPerLane; ++i) {
            outputValues[i] = outputValues[i] * factor
                + expScore * float(values[i]);
        }
        keys += BLOCKS * D;
        values += BLOCKS * D;
    }

    const uint statIndex = qHead * BLOCKS + block;
    if (simd_lane == 0) {
        stats[statIndex] = sumExpScore;
        stats[qHeads * BLOCKS + statIndex] = maxScore;
    }
    for (uint i = 0; i < valuesPerLane; ++i) out[i] = half(outputValues[i]);
}

// Final half of the fixed-B128 MLX two-pass topology. Launching 32 simdgroups
// per query head preserves the reference block visitation and transpose/reduce
// order exactly.
kernel void attention_decode_mlx_vector_2pass_2_d256_b128(
    device const half*  partials [[buffer(0)]],
    device const float* stats    [[buffer(1)]],
    device half*        out      [[buffer(2)]],
    constant uint&      qHeads   [[buffer(3)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]]
) {
    constexpr uint D = 256;
    constexpr uint BLOCKS = 128;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint valuesPerLane = D / BD;
    if (qHead >= qHeads) return;

    device const half* partial = partials
        + (qHead * BLOCKS + simd_group) * D + simd_lane * valuesPerLane;
    device const float* sums = stats + qHead * BLOCKS;
    device const float* maxs = stats + qHeads * BLOCKS + qHead * BLOCKS;
    device half* output = out + qHead * D + simd_group * valuesPerLane;

    thread float values[valuesPerLane] = {0.0f};
    threadgroup float transpose[BN * BD];

    float maxScore = -3.402823466e+38f;
    for (uint b = 0; b < BLOCKS / BN; ++b) {
        maxScore = max(maxScore, maxs[simd_lane + BN * b]);
    }
    maxScore = simd_max(maxScore);

    float sumExpScore = 0.0f;
    for (uint b = 0; b < BLOCKS / BN; ++b) {
        const uint block = simd_lane + BN * b;
        const float factor = fast::exp(maxs[block] - maxScore);
        sumExpScore += factor * sums[block];
    }
    sumExpScore = simd_sum(sumExpScore);

    for (uint b = 0; b < BLOCKS / BN; ++b) {
        const uint block = simd_group + BN * b;
        const float factor = fast::exp(maxs[block] - maxScore);
        for (uint i = 0; i < valuesPerLane; ++i) {
            values[i] += factor * float(partial[i]);
        }
        partial += BN * D;
    }

    for (uint i = 0; i < valuesPerLane; ++i) {
        transpose[simd_lane * BD + simd_group] = values[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        values[i] = simd_sum(transpose[simd_group * BD + simd_lane]);
        values[i] = sumExpScore == 0.0f
            ? values[i]
            : (values[i] / sumExpScore);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (simd_lane == 0) {
        for (uint i = 0; i < valuesPerLane; ++i) output[i] = half(values[i]);
    }
}

// MLX-style decode attention specialization for VibeThinker-class Qwen geometry:
//   qHeads=16, kvHeads=2, headDim=128, valueDim=128, gqa=8.
// Launch: threadgroups=(16, 1, 1), threadsPerThreadgroup=(1024, 1, 1).
kernel void attention_decode_d128_h16_kv2_sdpa(
    device half*       q_out    [[buffer(0)]],  // [16, 128] in-place
    device const half* k_cache  [[buffer(1)]],  // [2, cacheSeqCapacity, 128]
    device const half* v_cache  [[buffer(2)]],  // [2, cacheSeqCapacity, 128]
    constant uint&     seqLen   [[buffer(3)]],  // actual decode length (pos + 1)
    constant uint&     cacheSeqCapacity [[buffer(4)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]]
) {
    constexpr uint headDim = 128;
    constexpr uint gqaRatio = 8;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint qkPerThread = headDim / BD;
    constexpr uint vPerThread = headDim / BD;
    constexpr float scale = 0.08838834764831845f;

    const uint kvHead = qHead / gqaRatio;
    const uint qOffset = qHead * headDim;
    const uint kvBase = kvHead * cacheSeqCapacity * headDim;

    device const half* query = q_out + qOffset + simd_lane * qkPerThread;
    device const half* keys = k_cache + kvBase + simd_group * headDim + simd_lane * qkPerThread;
    device const half* values =
        v_cache + kvBase + simd_group * headDim + simd_lane * vPerThread;
    device half* out = q_out + qOffset + simd_group * vPerThread;

    thread float q[qkPerThread];
    thread float k[qkPerThread];
    thread float o[vPerThread];

    threadgroup float outputs[BN * BD];
    threadgroup float maxScores[BN];
    threadgroup float sumExpScores[BN];

    for (uint i = 0; i < qkPerThread; ++i) {
        q[i] = scale * float(query[i]);
    }
    for (uint i = 0; i < vPerThread; ++i) {
        o[i] = 0.0f;
    }

    float maxScore = -INFINITY;
    float sumExpScore = 0.0f;

    for (uint s = simd_group; s < seqLen; s += BN) {
        for (uint i = 0; i < qkPerThread; ++i) {
            k[i] = float(keys[i]);
        }

        float score = 0.0f;
        for (uint i = 0; i < qkPerThread; ++i) {
            score += q[i] * k[i];
        }
        score = simd_sum(score);

        float newMax = max(maxScore, score);
        float factor = fast::exp(maxScore - newMax);
        float expScore = fast::exp(score - newMax);

        maxScore = newMax;
        sumExpScore = sumExpScore * factor + expScore;

        for (uint i = 0; i < vPerThread; ++i) {
            o[i] = o[i] * factor + expScore * float(values[i]);
        }

        keys += BN * headDim;
        values += BN * headDim;
    }

    if (simd_lane == 0) {
        maxScores[simd_group] = maxScore;
        sumExpScores[simd_group] = sumExpScore;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    maxScore = maxScores[simd_lane];
    float newMax = simd_max(maxScore);
    float factor = fast::exp(maxScore - newMax);
    sumExpScore = simd_sum(sumExpScores[simd_lane] * factor);

    for (uint i = 0; i < vPerThread; ++i) {
        outputs[simd_lane * BD + simd_group] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_group * BD + simd_lane] * factor);
        if (sumExpScore != 0.0f) {
            o[i] /= sumExpScore;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lane == 0) {
        for (uint i = 0; i < vPerThread; ++i) {
            out[i] = half(o[i]);
        }
    }
}

static inline float gemma_apply_softcap_if_needed(
    float score,
    bool useSoftcap,
    float softcap
) {
    if (!useSoftcap) { return score; }
    return softcap * tanh(score / softcap);
}

static inline void gemma_apply_split_half_rope(
    threadgroup float* vec,
    const device half* cos_row,
    const device half* sin_row,
    uint headDim,
    uint ropeDim,
    uint ropeLayout,
    uint tid,
    uint tgs
) {
    uint halfRope = ropeDim / 2;
    for (uint pair = tid; pair < halfRope; pair += tgs) {
        uint d0 = pair;
        uint d1 = ropeLayout == 2 ? (pair + headDim / 2) : (pair + halfRope);
        uint c0Index = pair;
        uint c1Index = ropeLayout == 2 ? (pair + halfRope) : d1;

        float x0 = vec[d0];
        float x1 = vec[d1];
        float c0 = float(cos_row[c0Index]);
        float s0 = float(sin_row[c0Index]);
        float c1 = float(cos_row[c1Index]);
        float s1 = float(sin_row[c1Index]);

        vec[d0] = x0 * c0 - x1 * s0;
        vec[d1] = x1 * c1 + x0 * s1;
    }
}

static inline void gemma_decode_d256_h8_kv1_fused_shared_impl(
    device half*       q_out,
    const device half* cos_row,
    const device half* sin_row,
    const device half* k_cache,
    const device half* v_cache,
    uint               seqLen,
    uint               cacheSeqCapacity,
    uint               slidingWindow,
    bool               useSoftcap,
    float              softcap,
    uint               qHead,
    uint               tid,
    uint               simd_lane,
    uint               simd_group,
    threadgroup float* qShared,
    threadgroup float* scores,
    threadgroup float* reductions
) {
    constexpr uint headDim = 256;
    constexpr uint headDim4 = headDim / 4;
    constexpr uint maxActiveSeq = 512;
    constexpr uint gqaRatio = 8;
    constexpr uint ropeDim = 256;
    constexpr uint tgs = 256;

    if (seqLen == 0) { return; }

    uint seqStart = 0;
    if (slidingWindow > 0 && slidingWindow < seqLen) {
        seqStart = seqLen - slidingWindow;
    }
    uint activeLen = seqLen - seqStart;
    if (activeLen == 0 || activeLen > maxActiveSeq) { return; }

    uint kvHead = qHead / gqaRatio;
    uint qOffset = qHead * headDim;
    uint kvBase = kvHead * cacheSeqCapacity * headDim;
    device half4* qOut4 = reinterpret_cast<device half4*>(q_out + qOffset);

    for (uint d = tid; d < headDim; d += tgs) {
        qShared[d] = float(q_out[qOffset + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    gemma_apply_split_half_rope(qShared, cos_row, sin_row, headDim, ropeDim, 1, tid, tgs);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d4 = tid; d4 < headDim4; d4 += tgs) {
        qOut4[d4] = half4(
            qShared[d4 * 4 + 0],
            qShared[d4 * 4 + 1],
            qShared[d4 * 4 + 2],
            qShared[d4 * 4 + 3]
        );
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint d = tid; d < headDim; d += tgs) {
        qShared[d] = float(q_out[qOffset + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float maxScore = -INFINITY;
    for (uint idx = tid; idx < activeLen; idx += tgs) {
        uint absPos = seqStart + idx;
        float dot = 0.0f;
        uint cacheOffset = kvBase + absPos * headDim;
        for (uint d = 0; d < headDim; d++) {
            dot += qShared[d] * float(k_cache[cacheOffset + d]);
        }
        float score = gemma_apply_softcap_if_needed(dot, useSoftcap, softcap);
        scores[idx] = score;
        maxScore = max(maxScore, score);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    maxScore = simd_max(maxScore);
    if (simd_lane == 0) { reductions[simd_group] = maxScore; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float reducedMax = -INFINITY;
        for (uint group = 0; group < tgs / 32; group++) {
            reducedMax = max(reducedMax, reductions[group]);
        }
        reductions[0] = reducedMax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scoreMax = reductions[0];

    float sumExp = 0.0f;
    for (uint idx = tid; idx < activeLen; idx += tgs) {
        float e = exp(scores[idx] - scoreMax);
        scores[idx] = e;
        sumExp += e;
    }
    sumExp = simd_sum(sumExp);
    if (simd_lane == 0) { reductions[simd_group] = sumExp; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float reducedSum = 0.0f;
        for (uint group = 0; group < tgs / 32; group++) {
            reducedSum += reductions[group];
        }
        reductions[0] = reducedSum != 0.0f ? (1.0f / reducedSum) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invSum = reductions[0];
    for (uint d = tid; d < headDim; d += tgs) {
        float acc = 0.0f;
        for (uint idx = 0; idx < activeLen; idx++) {
            uint absPos = seqStart + idx;
            acc += (scores[idx] * invSum) * float(v_cache[kvBase + absPos * headDim + d]);
        }
        q_out[qOffset + d] = half(acc);
    }
}

static inline void gemma_decode_d256_h8_kv1_fused_impl(
    device half*       q_out,
    const device half* k_in,
    const device half* v_in,
    const device half* cos_row,
    const device half* sin_row,
    device half*       key_cache,
    device half*       val_cache,
    uint               seqLen,
    uint               cacheSeqCapacity,
    uint               slidingWindow,
    bool               useSoftcap,
    float              softcap,
    uint               qHead,
    uint               tid,
    uint               simd_lane,
    uint               simd_group,
    threadgroup float* qShared,
    threadgroup float* kLocal,
    threadgroup float* vLocal,
    threadgroup float* scores,
    threadgroup float* reductions
) {
    constexpr uint headDim = 256;
    constexpr uint headDim4 = headDim / 4;
    constexpr uint maxActiveSeq = 512;
    constexpr uint gqaRatio = 8;
    constexpr uint ropeDim = 256;
    constexpr uint tgs = 256;

    if (seqLen == 0) { return; }

    uint seqStart = 0;
    if (slidingWindow > 0 && slidingWindow < seqLen) {
        seqStart = seqLen - slidingWindow;
    }
    uint activeLen = seqLen - seqStart;
    if (activeLen == 0 || activeLen > maxActiveSeq) { return; }

    uint currentAbsPos = seqLen - 1;
    uint kvHead = qHead / gqaRatio;
    uint qOffset = qHead * headDim;
    uint kvBase = kvHead * cacheSeqCapacity * headDim;
    device half4* qOut4 = reinterpret_cast<device half4*>(q_out + qOffset);

    for (uint d = tid; d < headDim; d += tgs) {
        qShared[d] = float(q_out[qOffset + d]);
        kLocal[d] = float(k_in[d]);
        vLocal[d] = float(v_in[d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    gemma_apply_split_half_rope(qShared, cos_row, sin_row, headDim, ropeDim, 1, tid, tgs);
    gemma_apply_split_half_rope(kLocal, cos_row, sin_row, headDim, ropeDim, 1, tid, tgs);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d4 = tid; d4 < headDim4; d4 += tgs) {
        qOut4[d4] = half4(
            qShared[d4 * 4 + 0],
            qShared[d4 * 4 + 1],
            qShared[d4 * 4 + 2],
            qShared[d4 * 4 + 3]
        );
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint d = tid; d < headDim; d += tgs) {
        qShared[d] = float(q_out[qOffset + d]);
        kLocal[d] = float(half(kLocal[d]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device half4* keyCache4 = reinterpret_cast<device half4*>(
        key_cache + kvBase + currentAbsPos * headDim
    );
    device half4* valCache4 = reinterpret_cast<device half4*>(
        val_cache + kvBase + currentAbsPos * headDim
    );
    for (uint d4 = tid; d4 < headDim4; d4 += tgs) {
        keyCache4[d4] = half4(
            kLocal[d4 * 4 + 0],
            kLocal[d4 * 4 + 1],
            kLocal[d4 * 4 + 2],
            kLocal[d4 * 4 + 3]
        );
        valCache4[d4] = half4(
            vLocal[d4 * 4 + 0],
            vLocal[d4 * 4 + 1],
            vLocal[d4 * 4 + 2],
            vLocal[d4 * 4 + 3]
        );
    }
    threadgroup_barrier(mem_flags::mem_device);

    float maxScore = -INFINITY;
    for (uint idx = tid; idx < activeLen; idx += tgs) {
        float dot = 0.0f;
        uint absPos = seqStart + idx;
        uint cacheOffset = kvBase + absPos * headDim;
        for (uint d = 0; d < headDim; d++) {
            dot += qShared[d] * float(key_cache[cacheOffset + d]);
        }
        float score = gemma_apply_softcap_if_needed(dot, useSoftcap, softcap);
        scores[idx] = score;
        maxScore = max(maxScore, score);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    maxScore = simd_max(maxScore);
    if (simd_lane == 0) { reductions[simd_group] = maxScore; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float reducedMax = -INFINITY;
        for (uint group = 0; group < tgs / 32; group++) {
            reducedMax = max(reducedMax, reductions[group]);
        }
        reductions[0] = reducedMax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scoreMax = reductions[0];

    float sumExp = 0.0f;
    for (uint idx = tid; idx < activeLen; idx += tgs) {
        float e = exp(scores[idx] - scoreMax);
        scores[idx] = e;
        sumExp += e;
    }
    sumExp = simd_sum(sumExp);
    if (simd_lane == 0) { reductions[simd_group] = sumExp; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float reducedSum = 0.0f;
        for (uint group = 0; group < tgs / 32; group++) {
            reducedSum += reductions[group];
        }
        reductions[0] = reducedSum != 0.0f ? (1.0f / reducedSum) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invSum = reductions[0];
    for (uint d = tid; d < headDim; d += tgs) {
        float acc = 0.0f;
        for (uint idx = 0; idx < activeLen; idx++) {
            uint absPos = seqStart + idx;
            acc += (scores[idx] * invSum) * float(val_cache[kvBase + absPos * headDim + d]);
        }
        q_out[qOffset + d] = half(acc);
    }
}

static inline void gemma_decode_d512_h8_kv1_fused_shared_impl(
    device half*       q_out,
    const device half* cos_row,
    const device half* sin_row,
    const device half* k_cache,
    const device half* v_cache,
    uint               seqLen,
    uint               cacheSeqCapacity,
    uint               slidingWindow,
    bool               useSoftcap,
    float              softcap,
    uint               qHead,
    uint               tid,
    threadgroup float* qShared,
    threadgroup float4* qPacked,
    threadgroup float* scores,
    threadgroup float* partial,
    uint               simd_lane,
    uint               simd_group
) {
    constexpr uint headDim = 512;
    constexpr uint headDim4 = headDim / 4;
    constexpr uint maxActiveSeq = 128;
    constexpr uint gqaRatio = 8;
    constexpr uint ropeDim = 128;
    constexpr uint tgs = 128;

    if (seqLen == 0) { return; }

    uint seqStart = 0;
    if (slidingWindow > 0 && slidingWindow < seqLen) {
        seqStart = seqLen - slidingWindow;
    }
    uint activeLen = seqLen - seqStart;
    if (activeLen == 0 || activeLen > maxActiveSeq) { return; }

    uint kvHead = qHead / gqaRatio;
    uint qOffset = qHead * headDim;
    uint kvBase = kvHead * cacheSeqCapacity * headDim;

    for (uint d = tid; d < headDim; d += tgs) {
        qShared[d] = float(q_out[qOffset + d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    gemma_apply_split_half_rope(qShared, cos_row, sin_row, headDim, ropeDim, 2, tid, tgs);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < headDim; d += tgs) {
        qShared[d] = float(half(qShared[d]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device half4* out4 = reinterpret_cast<device half4*>(q_out + qOffset);
    for (uint d4 = tid; d4 < headDim4; d4 += tgs) {
        out4[d4] = half4(
            qShared[d4 * 4 + 0],
            qShared[d4 * 4 + 1],
            qShared[d4 * 4 + 2],
            qShared[d4 * 4 + 3]
        );
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint d4 = tid; d4 < headDim4; d4 += tgs) {
        qPacked[d4] = float4(out4[d4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float maxScore = -INFINITY;
    if (tid < activeLen) {
        uint absPos = seqStart + tid;
        uint cacheOffset = kvBase + absPos * headDim;
        float accDot = 0.0f;
        for (uint d = 0; d < headDim; d++) {
            accDot += qShared[d] * float(k_cache[cacheOffset + d]);
        }
        float score = gemma_apply_softcap_if_needed(accDot, useSoftcap, softcap);
        scores[tid] = score;
        maxScore = score;
    }
    maxScore = simd_max(maxScore);
    if (simd_lane == 0) { partial[simd_group] = maxScore; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float reduced = partial[0];
        for (uint i = 1; i < 4; i++) {
            reduced = max(reduced, partial[i]);
        }
        partial[0] = reduced;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scoreMax = partial[0];

    float sumExp = 0.0f;
    if (tid < activeLen) {
        float e = exp(scores[tid] - scoreMax);
        scores[tid] = e;
        sumExp = e;
    }
    sumExp = simd_sum(sumExp);
    if (simd_lane == 0) { partial[simd_group] = sumExp; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float reduced = 0.0f;
        for (uint i = 0; i < 4; i++) {
            reduced += partial[i];
        }
        partial[0] = reduced != 0.0f ? (1.0f / reduced) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invSum = partial[0];

    if (tid < activeLen) {
        scores[tid] *= invSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < headDim4) {
        float4 acc = float4(0.0f);
        for (uint idx = 0; idx < activeLen; idx++) {
            const device half4* v4 = reinterpret_cast<const device half4*>(
                v_cache + kvBase + (seqStart + idx) * headDim
            );
            acc += scores[idx] * float4(v4[tid]);
        }
        out4[tid] = half4(acc);
    }
}

static inline void gemma_decode_d512_h8_kv1_fused_impl(
    device half*       q_out,
    const device half* k_in,
    const device half* v_in,
    const device half* cos_row,
    const device half* sin_row,
    device half*       key_cache,
    device half*       val_cache,
    uint               seqLen,
    uint               cacheSeqCapacity,
    uint               slidingWindow,
    bool               useSoftcap,
    float              softcap,
    uint               qHead,
    uint               tid,
    threadgroup float* qShared,
    threadgroup float4* qPacked,
    threadgroup float* kLocal,
    threadgroup float4* kPacked,
    threadgroup float* vLocal,
    threadgroup float4* vPacked,
    threadgroup float* scores,
    threadgroup float* partial,
    uint               simd_lane,
    uint               simd_group
) {
    constexpr uint headDim = 512;
    constexpr uint headDim4 = headDim / 4;
    constexpr uint maxActiveSeq = 128;
    constexpr uint gqaRatio = 8;
    constexpr uint ropeDim = 128;
    constexpr uint tgs = 128;

    if (seqLen == 0) { return; }

    uint seqStart = 0;
    if (slidingWindow > 0 && slidingWindow < seqLen) {
        seqStart = seqLen - slidingWindow;
    }
    uint activeLen = seqLen - seqStart;
    if (activeLen == 0 || activeLen > maxActiveSeq) { return; }

    uint currentAbsPos = seqLen - 1;
    uint kvHead = qHead / gqaRatio;
    uint qOffset = qHead * headDim;
    uint kvBase = kvHead * cacheSeqCapacity * headDim;

    for (uint d = tid; d < headDim; d += tgs) {
        qShared[d] = float(q_out[qOffset + d]);
        kLocal[d] = float(k_in[d]);
        vLocal[d] = float(v_in[d]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    gemma_apply_split_half_rope(qShared, cos_row, sin_row, headDim, ropeDim, 2, tid, tgs);
    gemma_apply_split_half_rope(kLocal, cos_row, sin_row, headDim, ropeDim, 2, tid, tgs);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < headDim; d += tgs) {
        qShared[d] = float(half(qShared[d]));
        kLocal[d] = float(half(kLocal[d]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device half4* out4 = reinterpret_cast<device half4*>(q_out + qOffset);
    for (uint d4 = tid; d4 < headDim4; d4 += tgs) {
        out4[d4] = half4(
            qShared[d4 * 4 + 0],
            qShared[d4 * 4 + 1],
            qShared[d4 * 4 + 2],
            qShared[d4 * 4 + 3]
        );
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint d4 = tid; d4 < headDim4; d4 += tgs) {
        qPacked[d4] = float4(out4[d4]);
        kPacked[d4] = float4(
            kLocal[d4 * 4 + 0],
            kLocal[d4 * 4 + 1],
            kLocal[d4 * 4 + 2],
            kLocal[d4 * 4 + 3]
        );
        vPacked[d4] = float4(
            vLocal[d4 * 4 + 0],
            vLocal[d4 * 4 + 1],
            vLocal[d4 * 4 + 2],
            vLocal[d4 * 4 + 3]
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Every Q-head threadgroup writes the same rounded KV row before reading
    // it back. This avoids cross-threadgroup ordering assumptions for the
    // current token while preserving staged "write cache then attend" semantics.
    device half4* keyCache4 = reinterpret_cast<device half4*>(
        key_cache + kvBase + currentAbsPos * headDim
    );
    device half4* valCache4 = reinterpret_cast<device half4*>(
        val_cache + kvBase + currentAbsPos * headDim
    );
    for (uint d4 = tid; d4 < headDim4; d4 += tgs) {
        keyCache4[d4] = half4(kPacked[d4]);
        valCache4[d4] = half4(vPacked[d4]);
    }
    threadgroup_barrier(mem_flags::mem_device);

    float maxScore = -INFINITY;
    if (tid < activeLen) {
        uint absPos = seqStart + tid;
        uint cacheOffset = kvBase + absPos * headDim;
        float accDot = 0.0f;
        for (uint d = 0; d < headDim; d++) {
            accDot += qShared[d] * float(key_cache[cacheOffset + d]);
        }
        float score = gemma_apply_softcap_if_needed(accDot, useSoftcap, softcap);
        scores[tid] = score;
        maxScore = score;
    }
    maxScore = simd_max(maxScore);
    if (simd_lane == 0) { partial[simd_group] = maxScore; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float reduced = partial[0];
        for (uint i = 1; i < 4; i++) {
            reduced = max(reduced, partial[i]);
        }
        partial[0] = reduced;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scoreMax = partial[0];

    float sumExp = 0.0f;
    if (tid < activeLen) {
        float e = exp(scores[tid] - scoreMax);
        scores[tid] = e;
        sumExp = e;
    }
    sumExp = simd_sum(sumExp);
    if (simd_lane == 0) { partial[simd_group] = sumExp; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float reduced = 0.0f;
        for (uint i = 0; i < 4; i++) {
            reduced += partial[i];
        }
        partial[0] = reduced != 0.0f ? (1.0f / reduced) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invSum = partial[0];

    if (tid < activeLen) {
        scores[tid] *= invSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < headDim4) {
        float4 acc = float4(0.0f);
        for (uint idx = 0; idx < activeLen; idx++) {
            const device half4* v4 = reinterpret_cast<const device half4*>(
                val_cache + kvBase + (seqStart + idx) * headDim
            );
            acc += scores[idx] * float4(v4[tid]);
        }
        out4[tid] = half4(acc);
    }
}

// Experimental fused Gemma sliding decode path.
// Combines Q RoPE, optional K RoPE + KV update, and decode attention.
kernel void attention_decode_d256_h8_kv1_fused(
    device half*       q_out    [[buffer(0)]],
    device const half* k_in     [[buffer(1)]],
    device const half* v_in     [[buffer(2)]],
    device const half* cos_row  [[buffer(3)]],
    device const half* sin_row  [[buffer(4)]],
    device half*       key_cache [[buffer(5)]],
    device half*       val_cache [[buffer(6)]],
    constant uint&     seqLen   [[buffer(7)]],
    constant uint&     cacheSeqCapacity [[buffer(8)]],
    constant uint&     slidingWindow [[buffer(9)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float qShared[256];
    threadgroup float kLocal[256];
    threadgroup float vLocal[256];
    threadgroup float scores[512];
    threadgroup float reductions[8];
    gemma_decode_d256_h8_kv1_fused_impl(
        q_out, k_in, v_in, cos_row, sin_row, key_cache, val_cache,
        seqLen, cacheSeqCapacity, slidingWindow,
        false, 0.0f,
        qHead, tid, simd_lane, simd_group, qShared, kLocal, vLocal, scores, reductions
    );
}

kernel void attention_decode_d256_h8_kv1_fused_softcap(
    device half*       q_out    [[buffer(0)]],
    device const half* k_in     [[buffer(1)]],
    device const half* v_in     [[buffer(2)]],
    device const half* cos_row  [[buffer(3)]],
    device const half* sin_row  [[buffer(4)]],
    device half*       key_cache [[buffer(5)]],
    device half*       val_cache [[buffer(6)]],
    constant uint&     seqLen   [[buffer(7)]],
    constant uint&     cacheSeqCapacity [[buffer(8)]],
    constant uint&     slidingWindow [[buffer(9)]],
    constant float&    softcap  [[buffer(10)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float qShared[256];
    threadgroup float kLocal[256];
    threadgroup float vLocal[256];
    threadgroup float scores[512];
    threadgroup float reductions[8];
    gemma_decode_d256_h8_kv1_fused_impl(
        q_out, k_in, v_in, cos_row, sin_row, key_cache, val_cache,
        seqLen, cacheSeqCapacity, slidingWindow,
        true, softcap,
        qHead, tid, simd_lane, simd_group, qShared, kLocal, vLocal, scores, reductions
    );
}

kernel void attention_decode_d256_h8_kv1_fused_shared(
    device half*       q_out    [[buffer(0)]],
    device const half* cos_row  [[buffer(1)]],
    device const half* sin_row  [[buffer(2)]],
    device const half* k_cache  [[buffer(3)]],
    device const half* v_cache  [[buffer(4)]],
    constant uint&     seqLen   [[buffer(5)]],
    constant uint&     cacheSeqCapacity [[buffer(6)]],
    constant uint&     slidingWindow [[buffer(7)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float qShared[256];
    threadgroup float scores[512];
    threadgroup float reductions[8];
    gemma_decode_d256_h8_kv1_fused_shared_impl(
        q_out, cos_row, sin_row, k_cache, v_cache,
        seqLen, cacheSeqCapacity, slidingWindow,
        false, 0.0f,
        qHead, tid, simd_lane, simd_group, qShared, scores, reductions
    );
}

kernel void attention_decode_d256_h8_kv1_fused_shared_softcap(
    device half*       q_out    [[buffer(0)]],
    device const half* cos_row  [[buffer(1)]],
    device const half* sin_row  [[buffer(2)]],
    device const half* k_cache  [[buffer(3)]],
    device const half* v_cache  [[buffer(4)]],
    constant uint&     seqLen   [[buffer(5)]],
    constant uint&     cacheSeqCapacity [[buffer(6)]],
    constant uint&     slidingWindow [[buffer(7)]],
    constant float&    softcap  [[buffer(8)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float qShared[256];
    threadgroup float scores[512];
    threadgroup float reductions[8];
    gemma_decode_d256_h8_kv1_fused_shared_impl(
        q_out, cos_row, sin_row, k_cache, v_cache,
        seqLen, cacheSeqCapacity, slidingWindow,
        true, softcap,
        qHead, tid, simd_lane, simd_group, qShared, scores, reductions
    );
}

// Experimental fused Gemma global short-context decode path.
// Combines Q RoPE, optional K RoPE + KV update, and decode attention for seqLen < 128.
kernel void attention_decode_d512_h8_kv1_fused(
    device half*       q_out    [[buffer(0)]],
    device const half* k_in     [[buffer(1)]],
    device const half* v_in     [[buffer(2)]],
    device const half* cos_row  [[buffer(3)]],
    device const half* sin_row  [[buffer(4)]],
    device half*       key_cache [[buffer(5)]],
    device half*       val_cache [[buffer(6)]],
    constant uint&     seqLen   [[buffer(7)]],
    constant uint&     cacheSeqCapacity [[buffer(8)]],
    constant uint&     slidingWindow [[buffer(9)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float qShared[512];
    threadgroup float4 qPacked[128];
    threadgroup float kLocal[512];
    threadgroup float4 kPacked[128];
    threadgroup float vLocal[512];
    threadgroup float4 vPacked[128];
    threadgroup float scores[128];
    threadgroup float partial[4];
    gemma_decode_d512_h8_kv1_fused_impl(
        q_out, k_in, v_in, cos_row, sin_row, key_cache, val_cache,
        seqLen, cacheSeqCapacity, slidingWindow,
        false, 0.0f,
        qHead, tid, qShared, qPacked, kLocal, kPacked, vLocal, vPacked, scores, partial,
        simd_lane, simd_group
    );
}

kernel void attention_decode_d512_h8_kv1_fused_softcap(
    device half*       q_out    [[buffer(0)]],
    device const half* k_in     [[buffer(1)]],
    device const half* v_in     [[buffer(2)]],
    device const half* cos_row  [[buffer(3)]],
    device const half* sin_row  [[buffer(4)]],
    device half*       key_cache [[buffer(5)]],
    device half*       val_cache [[buffer(6)]],
    constant uint&     seqLen   [[buffer(7)]],
    constant uint&     cacheSeqCapacity [[buffer(8)]],
    constant uint&     slidingWindow [[buffer(9)]],
    constant float&    softcap  [[buffer(10)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float qShared[512];
    threadgroup float4 qPacked[128];
    threadgroup float kLocal[512];
    threadgroup float4 kPacked[128];
    threadgroup float vLocal[512];
    threadgroup float4 vPacked[128];
    threadgroup float scores[128];
    threadgroup float partial[4];
    gemma_decode_d512_h8_kv1_fused_impl(
        q_out, k_in, v_in, cos_row, sin_row, key_cache, val_cache,
        seqLen, cacheSeqCapacity, slidingWindow,
        true, softcap,
        qHead, tid, qShared, qPacked, kLocal, kPacked, vLocal, vPacked, scores, partial,
        simd_lane, simd_group
    );
}

kernel void attention_decode_d512_h8_kv1_fused_shared(
    device half*       q_out    [[buffer(0)]],
    device const half* cos_row  [[buffer(1)]],
    device const half* sin_row  [[buffer(2)]],
    device const half* k_cache  [[buffer(3)]],
    device const half* v_cache  [[buffer(4)]],
    constant uint&     seqLen   [[buffer(5)]],
    constant uint&     cacheSeqCapacity [[buffer(6)]],
    constant uint&     slidingWindow [[buffer(7)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float qShared[512];
    threadgroup float4 qPacked[128];
    threadgroup float scores[128];
    threadgroup float partial[4];
    gemma_decode_d512_h8_kv1_fused_shared_impl(
        q_out, cos_row, sin_row, k_cache, v_cache,
        seqLen, cacheSeqCapacity, slidingWindow,
        false, 0.0f,
        qHead, tid, qShared, qPacked, scores, partial, simd_lane, simd_group
    );
}

kernel void attention_decode_d512_h8_kv1_fused_shared_softcap(
    device half*       q_out    [[buffer(0)]],
    device const half* cos_row  [[buffer(1)]],
    device const half* sin_row  [[buffer(2)]],
    device const half* k_cache  [[buffer(3)]],
    device const half* v_cache  [[buffer(4)]],
    constant uint&     seqLen   [[buffer(5)]],
    constant uint&     cacheSeqCapacity [[buffer(6)]],
    constant uint&     slidingWindow [[buffer(7)]],
    constant float&    softcap  [[buffer(8)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float qShared[512];
    threadgroup float4 qPacked[128];
    threadgroup float scores[128];
    threadgroup float partial[4];
    gemma_decode_d512_h8_kv1_fused_shared_impl(
        q_out, cos_row, sin_row, k_cache, v_cache,
        seqLen, cacheSeqCapacity, slidingWindow,
        true, softcap,
        qHead, tid, qShared, qPacked, scores, partial, simd_lane, simd_group
    );
}

// Gemma 4 E2B decode specialization for sliding-attention layers:
//   qHeads=8, kvHeads=1, headDim=256, gqa=8, slidingWindow=512, scale=1.
kernel void attention_decode_d256_h8_kv1(
    device half*       q_out    [[buffer(0)]],  // [8, 256] in-place
    device const half* k_cache  [[buffer(1)]],  // [1, cacheSeqCapacity, 256]
    device const half* v_cache  [[buffer(2)]],  // [1, cacheSeqCapacity, 256]
    constant uint&     seqLen   [[buffer(3)]],  // actual decode length (pos + 1)
    constant uint&     cacheSeqCapacity [[buffer(4)]],
    constant uint&     slidingWindow [[buffer(5)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint headDim = 256;
    constexpr uint headDim4 = headDim / 4;
    constexpr uint maxActiveSeq = 512;
    constexpr uint gqaRatio = 8;

    const uint kvHead = qHead / gqaRatio;
    const uint qOffset = qHead * headDim;
    const uint kvBase = kvHead * cacheSeqCapacity * headDim;

    uint seqStart = 0;
    if (slidingWindow > 0 && slidingWindow < seqLen) {
        seqStart = seqLen - slidingWindow;
    }
    const uint activeLen = seqLen - seqStart;

    device half4* out4 = reinterpret_cast<device half4*>(q_out + qOffset);
    const device half4* k4Base =
        reinterpret_cast<const device half4*>(k_cache + kvBase + seqStart * headDim);
    const device half4* v4Base =
        reinterpret_cast<const device half4*>(v_cache + kvBase + seqStart * headDim);

    threadgroup float4 qShared[headDim4];
    threadgroup float scores[maxActiveSeq];
    qShared[tid] = float4(out4[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < activeLen; idx += headDim4) {
        const device half4* k4 = k4Base + idx * headDim4;
        float accDot = 0.0f;
        for (uint d4 = 0; d4 < headDim4; d4++) {
            float4 qv = qShared[d4];
            float4 kv = float4(k4[d4]);
            accDot += qv[0] * kv[0];
            accDot += qv[1] * kv[1];
            accDot += qv[2] * kv[2];
            accDot += qv[3] * kv[3];
        }
        float score = accDot;
        scores[idx] = score;
    }
    if (tid == 0) {
        float maxScore = -INFINITY;
        for (uint idx = 0; idx < activeLen; idx++) {
            maxScore = max(maxScore, scores[idx]);
        }
        scores[maxActiveSeq - 1] = maxScore;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scoreMax = scores[maxActiveSeq - 1];

    if (tid == 0) {
        float sumExp = 0.0f;
        for (uint idx = 0; idx < activeLen; idx++) {
            float e = exp(scores[idx] - scoreMax);
            scores[idx] = e;
            sumExp += e;
        }
        const float invSum = sumExp != 0.0f ? (1.0f / sumExp) : 0.0f;
        for (uint idx = 0; idx < activeLen; idx++) {
            scores[idx] *= invSum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 acc = 0.0f;
    for (uint idx = 0; idx < activeLen; idx++) {
        acc += scores[idx] * float4(v4Base[idx * headDim4 + tid]);
    }
    out4[tid] = half4(acc);
}

// Gemma 4 E2B decode specialization for global-attention layers:
//   qHeads=8, kvHeads=1, headDim=512, gqa=8, scale=1.
kernel void attention_decode_d512_h8_kv1(
    device half*       q_out    [[buffer(0)]],  // [8, 512] in-place
    device const half* k_cache  [[buffer(1)]],  // [1, cacheSeqCapacity, 512]
    device const half* v_cache  [[buffer(2)]],  // [1, cacheSeqCapacity, 512]
    constant uint&     seqLen   [[buffer(3)]],  // actual decode length (pos + 1)
    constant uint&     cacheSeqCapacity [[buffer(4)]],
    constant uint&     slidingWindow [[buffer(5)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint headDim = 512;
    constexpr uint headDim4 = headDim / 4;
    constexpr uint maxActiveSeq = 64;
    constexpr uint gqaRatio = 8;

    const uint kvHead = qHead / gqaRatio;
    const uint qOffset = qHead * headDim;
    const uint kvBase = kvHead * cacheSeqCapacity * headDim;

    uint seqStart = 0;
    if (slidingWindow > 0 && slidingWindow < seqLen) {
        seqStart = seqLen - slidingWindow;
    }
    const uint activeLen = seqLen - seqStart;

    device half4* out4 = reinterpret_cast<device half4*>(q_out + qOffset);
    const device half4* k4Base =
        reinterpret_cast<const device half4*>(k_cache + kvBase + seqStart * headDim);
    const device half4* v4Base =
        reinterpret_cast<const device half4*>(v_cache + kvBase + seqStart * headDim);

    threadgroup float4 qShared[headDim4];
    threadgroup float scores[maxActiveSeq];
    threadgroup float partial[4];

    qShared[tid] = float4(out4[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float maxScore = -INFINITY;
    for (uint idx = tid; idx < activeLen; idx += headDim4) {
        const device half4* k4 = k4Base + idx * headDim4;
        float accDot = 0.0f;
        for (uint d4 = 0; d4 < headDim4; d4++) {
            accDot += dot(qShared[d4], float4(k4[d4]));
        }
        float score = accDot;
        scores[idx] = score;
        maxScore = max(maxScore, score);
    }

    maxScore = simd_max(maxScore);
    if (simd_lane == 0) { partial[simd_group] = maxScore; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float reduced = partial[0];
        for (uint i = 1; i < 4; i++) {
            reduced = max(reduced, partial[i]);
        }
        partial[0] = reduced;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scoreMax = partial[0];

    float sumExp = 0.0f;
    for (uint idx = tid; idx < activeLen; idx += headDim4) {
        float e = exp(scores[idx] - scoreMax);
        scores[idx] = e;
        sumExp += e;
    }

    sumExp = simd_sum(sumExp);
    if (simd_lane == 0) { partial[simd_group] = sumExp; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float reduced = 0.0f;
        for (uint i = 0; i < 4; i++) {
            reduced += partial[i];
        }
        partial[0] = 1.0f / reduced;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float invSum = partial[0];

    for (uint idx = tid; idx < activeLen; idx += headDim4) {
        scores[idx] *= invSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 acc = 0.0f;
    for (uint idx = 0; idx < activeLen; idx++) {
        acc += scores[idx] * float4(v4Base[idx * headDim4 + tid]);
    }
    out4[tid] = half4(acc);
}

// MLX-style Gemma decode attention specialization for long sliding contexts.
kernel void attention_decode_d256_h8_kv1_sdpa(
    device half*       q_out    [[buffer(0)]],  // [8, 256] in-place
    device const half* k_cache  [[buffer(1)]],  // [1, cacheSeqCapacity, 256]
    device const half* v_cache  [[buffer(2)]],  // [1, cacheSeqCapacity, 256]
    constant uint&     seqLen   [[buffer(3)]],  // actual decode length (pos + 1)
    constant uint&     cacheSeqCapacity [[buffer(4)]],
    constant uint&     slidingWindow [[buffer(5)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]]
) {
    constexpr uint headDim = 256;
    constexpr uint gqaRatio = 8;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint qkPerThread = headDim / BD;
    constexpr uint vPerThread = headDim / BD;
    constexpr float scale = 1.0f;

    const uint kvHead = qHead / gqaRatio;
    const uint qOffset = qHead * headDim;

    uint seqStart = 0;
    if (slidingWindow > 0 && slidingWindow < seqLen) {
        seqStart = seqLen - slidingWindow;
    }
    const uint activeLen = seqLen - seqStart;
    const uint kvBase = kvHead * cacheSeqCapacity * headDim + seqStart * headDim;

    device const half* query = q_out + qOffset + simd_lane * qkPerThread;
    device const half* keys = k_cache + kvBase + simd_group * headDim + simd_lane * qkPerThread;
    device const half* values =
        v_cache + kvBase + simd_group * headDim + simd_lane * vPerThread;
    device half* out = q_out + qOffset + simd_group * vPerThread;

    thread float q[qkPerThread];
    thread float k[qkPerThread];
    thread float o[vPerThread];

    threadgroup float outputs[BN * BD];
    threadgroup float maxScores[BN];
    threadgroup float sumExpScores[BN];

    for (uint i = 0; i < qkPerThread; ++i) {
        q[i] = scale * float(query[i]);
    }
    for (uint i = 0; i < vPerThread; ++i) {
        o[i] = 0.0f;
    }

    float maxScore = -INFINITY;
    float sumExpScore = 0.0f;

    for (uint s = simd_group; s < activeLen; s += BN) {
        for (uint i = 0; i < qkPerThread; ++i) {
            k[i] = float(keys[i]);
        }

        float score = 0.0f;
        for (uint i = 0; i < qkPerThread; ++i) {
            score += q[i] * k[i];
        }
        score = simd_sum(score);

        float newMax = max(maxScore, score);
        float factor = fast::exp(maxScore - newMax);
        float expScore = fast::exp(score - newMax);

        maxScore = newMax;
        sumExpScore = sumExpScore * factor + expScore;

        for (uint i = 0; i < vPerThread; ++i) {
            o[i] = o[i] * factor + expScore * float(values[i]);
        }

        keys += BN * headDim;
        values += BN * headDim;
    }

    if (simd_lane == 0) {
        maxScores[simd_group] = maxScore;
        sumExpScores[simd_group] = sumExpScore;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    maxScore = maxScores[simd_lane];
    float newMax = simd_max(maxScore);
    float factor = fast::exp(maxScore - newMax);
    sumExpScore = simd_sum(sumExpScores[simd_lane] * factor);

    for (uint i = 0; i < vPerThread; ++i) {
        outputs[simd_lane * BD + simd_group] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_group * BD + simd_lane] * factor);
        if (sumExpScore != 0.0f) {
            o[i] /= sumExpScore;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lane == 0) {
        for (uint i = 0; i < vPerThread; ++i) {
            out[i] = half(o[i]);
        }
    }
}

// MLX-style Gemma decode attention specialization for long global contexts.
kernel void attention_decode_d512_h8_kv1_sdpa(
    device half*       q_out    [[buffer(0)]],  // [8, 512] in-place
    device const half* k_cache  [[buffer(1)]],  // [1, cacheSeqCapacity, 512]
    device const half* v_cache  [[buffer(2)]],  // [1, cacheSeqCapacity, 512]
    constant uint&     seqLen   [[buffer(3)]],  // actual decode length (pos + 1)
    constant uint&     cacheSeqCapacity [[buffer(4)]],
    constant uint&     slidingWindow [[buffer(5)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]]
) {
    constexpr uint headDim = 512;
    constexpr uint gqaRatio = 8;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint qkPerThread = headDim / BD;
    constexpr uint vPerThread = headDim / BD;
    constexpr float scale = 1.0f;

    const uint kvHead = qHead / gqaRatio;
    const uint qOffset = qHead * headDim;

    uint seqStart = 0;
    if (slidingWindow > 0 && slidingWindow < seqLen) {
        seqStart = seqLen - slidingWindow;
    }
    const uint activeLen = seqLen - seqStart;
    const uint kvBase = kvHead * cacheSeqCapacity * headDim + seqStart * headDim;

    device const half* query = q_out + qOffset + simd_lane * qkPerThread;
    device const half* keys = k_cache + kvBase + simd_group * headDim + simd_lane * qkPerThread;
    device const half* values =
        v_cache + kvBase + simd_group * headDim + simd_lane * vPerThread;
    device half* out = q_out + qOffset + simd_group * vPerThread;

    thread float q[qkPerThread];
    thread float k[qkPerThread];
    thread float o[vPerThread];

    threadgroup float outputs[BN * BD];
    threadgroup float maxScores[BN];
    threadgroup float sumExpScores[BN];

    for (uint i = 0; i < qkPerThread; ++i) {
        q[i] = scale * float(query[i]);
    }
    for (uint i = 0; i < vPerThread; ++i) {
        o[i] = 0.0f;
    }

    float maxScore = -INFINITY;
    float sumExpScore = 0.0f;

    for (uint s = simd_group; s < activeLen; s += BN) {
        for (uint i = 0; i < qkPerThread; ++i) {
            k[i] = float(keys[i]);
        }

        float score = 0.0f;
        for (uint i = 0; i < qkPerThread; ++i) {
            score += q[i] * k[i];
        }
        score = simd_sum(score);

        float newMax = max(maxScore, score);
        float factor = fast::exp(maxScore - newMax);
        float expScore = fast::exp(score - newMax);

        maxScore = newMax;
        sumExpScore = sumExpScore * factor + expScore;

        for (uint i = 0; i < vPerThread; ++i) {
            o[i] = o[i] * factor + expScore * float(values[i]);
        }

        keys += BN * headDim;
        values += BN * headDim;
    }

    if (simd_lane == 0) {
        maxScores[simd_group] = maxScore;
        sumExpScores[simd_group] = sumExpScore;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    maxScore = maxScores[simd_lane];
    float newMax = simd_max(maxScore);
    float factor = fast::exp(maxScore - newMax);
    sumExpScore = simd_sum(sumExpScores[simd_lane] * factor);

    for (uint i = 0; i < vPerThread; ++i) {
        outputs[simd_lane * BD + simd_group] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_group * BD + simd_lane] * factor);
        if (sumExpScore != 0.0f) {
            o[i] /= sumExpScore;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lane == 0) {
        for (uint i = 0; i < vPerThread; ++i) {
            out[i] = half(o[i]);
        }
    }
}

// MLX-style Gemma decode attention specialization for E4B sliding KV2 contexts.
kernel void attention_decode_d256_h8_kv2_sdpa_reserved(
    device half*       q_out    [[buffer(0)]],  // [8, 256] in-place
    device const half* k_cache  [[buffer(1)]],  // [2, cacheSeqCapacity, 256]
    device const half* v_cache  [[buffer(2)]],  // [2, cacheSeqCapacity, 256]
    constant uint&     seqLen   [[buffer(3)]],  // actual decode length (pos + 1)
    constant uint&     cacheSeqCapacity [[buffer(4)]],
    constant uint&     slidingWindow [[buffer(5)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]]
) {
    constexpr uint headDim = 256;
    constexpr uint gqaRatio = 4;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint qkPerThread = headDim / BD;
    constexpr uint vPerThread = headDim / BD;
    constexpr float scale = 1.0f;

    const uint kvHead = qHead / gqaRatio;
    const uint qOffset = qHead * headDim;

    uint seqStart = 0;
    if (slidingWindow > 0 && slidingWindow < seqLen) {
        seqStart = seqLen - slidingWindow;
    }
    const uint activeLen = seqLen - seqStart;
    const uint kvBase = kvHead * cacheSeqCapacity * headDim + seqStart * headDim;

    device const half* query = q_out + qOffset + simd_lane * qkPerThread;
    device const half* keys = k_cache + kvBase + simd_group * headDim + simd_lane * qkPerThread;
    device const half* values =
        v_cache + kvBase + simd_group * headDim + simd_lane * vPerThread;
    device half* out = q_out + qOffset + simd_group * vPerThread;

    thread float q[qkPerThread];
    thread float k[qkPerThread];
    thread float o[vPerThread];

    threadgroup float outputs[BN * BD];
    threadgroup float maxScores[BN];
    threadgroup float sumExpScores[BN];

    for (uint i = 0; i < qkPerThread; ++i) {
        q[i] = scale * float(query[i]);
    }
    for (uint i = 0; i < vPerThread; ++i) {
        o[i] = 0.0f;
    }

    float maxScore = -INFINITY;
    float sumExpScore = 0.0f;

    for (uint s = simd_group; s < activeLen; s += BN) {
        for (uint i = 0; i < qkPerThread; ++i) {
            k[i] = float(keys[i]);
        }

        float score = 0.0f;
        for (uint i = 0; i < qkPerThread; ++i) {
            score += q[i] * k[i];
        }
        score = simd_sum(score);

        float newMax = max(maxScore, score);
        float factor = fast::exp(maxScore - newMax);
        float expScore = fast::exp(score - newMax);

        maxScore = newMax;
        sumExpScore = sumExpScore * factor + expScore;

        for (uint i = 0; i < vPerThread; ++i) {
            o[i] = o[i] * factor + expScore * float(values[i]);
        }

        keys += BN * headDim;
        values += BN * headDim;
    }

    if (simd_lane == 0) {
        maxScores[simd_group] = maxScore;
        sumExpScores[simd_group] = sumExpScore;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    maxScore = maxScores[simd_lane];
    float newMax = simd_max(maxScore);
    float factor = fast::exp(maxScore - newMax);
    sumExpScore = simd_sum(sumExpScores[simd_lane] * factor);

    for (uint i = 0; i < vPerThread; ++i) {
        outputs[simd_lane * BD + simd_group] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_group * BD + simd_lane] * factor);
        if (sumExpScore != 0.0f) {
            o[i] /= sumExpScore;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lane == 0) {
        for (uint i = 0; i < vPerThread; ++i) {
            out[i] = half(o[i]);
        }
    }
}

// MLX-style Gemma decode attention specialization for E4B global KV2 contexts.
kernel void attention_decode_d512_h8_kv2_sdpa_reserved(
    device half*       q_out    [[buffer(0)]],  // [8, 512] in-place
    device const half* k_cache  [[buffer(1)]],  // [2, cacheSeqCapacity, 512]
    device const half* v_cache  [[buffer(2)]],  // [2, cacheSeqCapacity, 512]
    constant uint&     seqLen   [[buffer(3)]],  // actual decode length (pos + 1)
    constant uint&     cacheSeqCapacity [[buffer(4)]],
    constant uint&     slidingWindow [[buffer(5)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]]
) {
    constexpr uint headDim = 512;
    constexpr uint gqaRatio = 4;
    constexpr uint BN = 32;
    constexpr uint BD = 32;
    constexpr uint qkPerThread = headDim / BD;
    constexpr uint vPerThread = headDim / BD;
    constexpr float scale = 1.0f;

    const uint kvHead = qHead / gqaRatio;
    const uint qOffset = qHead * headDim;

    uint seqStart = 0;
    if (slidingWindow > 0 && slidingWindow < seqLen) {
        seqStart = seqLen - slidingWindow;
    }
    const uint activeLen = seqLen - seqStart;
    const uint kvBase = kvHead * cacheSeqCapacity * headDim + seqStart * headDim;

    device const half* query = q_out + qOffset + simd_lane * qkPerThread;
    device const half* keys = k_cache + kvBase + simd_group * headDim + simd_lane * qkPerThread;
    device const half* values =
        v_cache + kvBase + simd_group * headDim + simd_lane * vPerThread;
    device half* out = q_out + qOffset + simd_group * vPerThread;

    thread float q[qkPerThread];
    thread float k[qkPerThread];
    thread float o[vPerThread];

    threadgroup float outputs[BN * BD];
    threadgroup float maxScores[BN];
    threadgroup float sumExpScores[BN];

    for (uint i = 0; i < qkPerThread; ++i) {
        q[i] = scale * float(query[i]);
    }
    for (uint i = 0; i < vPerThread; ++i) {
        o[i] = 0.0f;
    }

    float maxScore = -INFINITY;
    float sumExpScore = 0.0f;

    for (uint s = simd_group; s < activeLen; s += BN) {
        for (uint i = 0; i < qkPerThread; ++i) {
            k[i] = float(keys[i]);
        }

        float score = 0.0f;
        for (uint i = 0; i < qkPerThread; ++i) {
            score += q[i] * k[i];
        }
        score = simd_sum(score);

        float newMax = max(maxScore, score);
        float factor = fast::exp(maxScore - newMax);
        float expScore = fast::exp(score - newMax);

        maxScore = newMax;
        sumExpScore = sumExpScore * factor + expScore;

        for (uint i = 0; i < vPerThread; ++i) {
            o[i] = o[i] * factor + expScore * float(values[i]);
        }

        keys += BN * headDim;
        values += BN * headDim;
    }

    if (simd_lane == 0) {
        maxScores[simd_group] = maxScore;
        sumExpScores[simd_group] = sumExpScore;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    maxScore = maxScores[simd_lane];
    float newMax = simd_max(maxScore);
    float factor = fast::exp(maxScore - newMax);
    sumExpScore = simd_sum(sumExpScores[simd_lane] * factor);

    for (uint i = 0; i < vPerThread; ++i) {
        outputs[simd_lane * BD + simd_group] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_group * BD + simd_lane] * factor);
        if (sumExpScore != 0.0f) {
            o[i] /= sumExpScore;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lane == 0) {
        for (uint i = 0; i < vPerThread; ++i) {
            out[i] = half(o[i]);
        }
    }
}

// Decode attention for the Gemma 4 *-it-assistant drafter sliding layers
// (hidden=256, q_heads=4, kv_heads=2, head_dim=256, rope_dim=256,
// rope_layout=1). Folds per-head Q RMS norm + Q RoPE into the SDPA
// loop so a single dispatch replaces the standalone trio
// per_head_rms_norm -> apply_rope -> attention_decode. K/V cache is
// read-only (external_kv).
//
// The score/softmax/output accumulation mirrors the generic
// attention_decode (3-pass full dot products) rather than the
// online-softmax SDPA variant: the drafter's distilled weights are
// tuned against the generic algorithm, and the alternate accumulation
// order in online-softmax produces enough fp drift across the
// drafter's 4 layers to shift argmax token selection.
//
// q_norm and q_rope output values are explicitly rounded through fp16
// to match the precision profile of the standalone three-dispatch
// chain, which writes fp16 to device memory between norm and rope and
// between rope and attention.
kernel void attention_decode_d256_h4_kv2_qnorm_rope_shared(
    device half*       q_out          [[buffer(0)]],   // [4, 256] in-place
    device const half* q_norm_weight  [[buffer(1)]],   // [256]
    device const half* cos_row        [[buffer(2)]],   // [256] cos at current decode position
    device const half* sin_row        [[buffer(3)]],   // [256] sin at current decode position
    device const half* k_cache        [[buffer(4)]],   // [2, cacheSeqCapacity, 256]
    device const half* v_cache        [[buffer(5)]],   // [2, cacheSeqCapacity, 256]
    constant uint&     seqLen         [[buffer(6)]],   // pos + 1
    constant uint&     cacheSeqCapacity [[buffer(7)]],
    constant uint&     slidingWindow [[buffer(8)]],
    constant float&    eps            [[buffer(9)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint headDim = 256;
    constexpr uint gqaRatio = 2;
    constexpr uint ropeDim = 256;
    constexpr uint tgs = 256;

    threadgroup float qShared[headDim];
    threadgroup float reductions[tgs / 32];
    threadgroup float qNormInvRMS;

    if (seqLen == 0) { return; }

    uint seqStart = 0;
    if (slidingWindow > 0 && slidingWindow < seqLen) {
        seqStart = seqLen - slidingWindow;
    }

    const uint kvHead = qHead / gqaRatio;
    const uint qOffset = qHead * headDim;
    const uint kvStride = cacheSeqCapacity * headDim;

    float sumSq = 0.0f;
    for (uint d = tid; d < headDim; d += tgs) {
        float v = float(q_out[qOffset + d]);
        qShared[d] = v;
        sumSq += v * v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumSq = simd_sum(sumSq);
    if (simd_lane == 0) { reductions[simd_group] = sumSq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < tgs / 32; ++s) { total += reductions[s]; }
        qNormInvRMS = rsqrt(total / float(headDim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < headDim; d += tgs) {
        float w = float(q_norm_weight[d]);
        qShared[d] = float(half(qShared[d] * qNormInvRMS * w));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    gemma_apply_split_half_rope(qShared, cos_row, sin_row, headDim, ropeDim, 1, tid, tgs);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < headDim; d += tgs) {
        qShared[d] = float(half(qShared[d]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The three SDPA passes recompute Q·K per-position instead of caching
    // scores in a threadgroup array; this matches the standalone
    // attention_decode kernel and keeps the dispatch bounded only by
    // seqLen, not by a fixed scratch tile.

    float maxScore = -INFINITY;
    for (uint s = seqStart + tid; s < seqLen; s += tgs) {
        float dot = 0.0f;
        uint cacheOffset = kvHead * kvStride + s * headDim;
        for (uint d = 0; d < headDim; ++d) {
            dot += qShared[d] * float(k_cache[cacheOffset + d]);
        }
        maxScore = max(maxScore, dot);
    }
    maxScore = simd_max(maxScore);
    if (simd_lane == 0) { reductions[simd_group] = maxScore; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float reducedMax = -INFINITY;
        for (uint g = 0; g < tgs / 32; ++g) {
            reducedMax = max(reducedMax, reductions[g]);
        }
        reductions[0] = reducedMax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scoreMax = reductions[0];

    float sumExp = 0.0f;
    for (uint s = seqStart + tid; s < seqLen; s += tgs) {
        float dot = 0.0f;
        uint cacheOffset = kvHead * kvStride + s * headDim;
        for (uint d = 0; d < headDim; ++d) {
            dot += qShared[d] * float(k_cache[cacheOffset + d]);
        }
        sumExp += exp(dot - scoreMax);
    }
    sumExp = simd_sum(sumExp);
    if (simd_lane == 0) { reductions[simd_group] = sumExp; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float reducedSum = 0.0f;
        for (uint g = 0; g < tgs / 32; ++g) { reducedSum += reductions[g]; }
        reductions[0] = reducedSum != 0.0f ? (1.0f / reducedSum) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invSum = reductions[0];

    for (uint d = tid; d < headDim; d += tgs) {
        float acc = 0.0f;
        for (uint s = seqStart; s < seqLen; ++s) {
            float dot = 0.0f;
            uint kOffset = kvHead * kvStride + s * headDim;
            for (uint qd = 0; qd < headDim; ++qd) {
                dot += qShared[qd] * float(k_cache[kOffset + qd]);
            }
            float weight = exp(dot - scoreMax) * invSum;
            acc += weight * float(v_cache[kOffset + d]);
        }
        q_out[qOffset + d] = half(acc);
    }
}

// Decode attention for the Gemma 4 *-it-assistant drafter global layer
// (q_heads=4, kv_heads=2, head_dim=512, rope_dim=128, rope_layout=2).
// Same fusion shape as the d256 sibling — q_norm + Q RoPE + 3-pass
// SDPA in one dispatch; K/V cache read-only (external_kv).
kernel void attention_decode_d512_h4_kv2_qnorm_rope_shared(
    device half*       q_out          [[buffer(0)]],   // [4, 512] in-place
    device const half* q_norm_weight  [[buffer(1)]],   // [512]
    device const half* cos_row        [[buffer(2)]],   // [128]
    device const half* sin_row        [[buffer(3)]],   // [128]
    device const half* k_cache        [[buffer(4)]],   // [2, cacheSeqCapacity, 512]
    device const half* v_cache        [[buffer(5)]],   // [2, cacheSeqCapacity, 512]
    constant uint&     seqLen         [[buffer(6)]],
    constant uint&     cacheSeqCapacity [[buffer(7)]],
    constant uint&     slidingWindow [[buffer(8)]],
    constant float&    eps            [[buffer(9)]],
    uint qHead      [[threadgroup_position_in_grid]],
    uint tid        [[thread_index_in_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint headDim = 512;
    constexpr uint gqaRatio = 2;
    constexpr uint ropeDim = 128;
    constexpr uint tgs = 256;

    threadgroup float qShared[headDim];
    threadgroup float reductions[tgs / 32];
    threadgroup float qNormInvRMS;

    if (seqLen == 0) { return; }

    uint seqStart = 0;
    if (slidingWindow > 0 && slidingWindow < seqLen) {
        seqStart = seqLen - slidingWindow;
    }

    const uint kvHead = qHead / gqaRatio;
    const uint qOffset = qHead * headDim;
    const uint kvStride = cacheSeqCapacity * headDim;

    float sumSq = 0.0f;
    for (uint d = tid; d < headDim; d += tgs) {
        float v = float(q_out[qOffset + d]);
        qShared[d] = v;
        sumSq += v * v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumSq = simd_sum(sumSq);
    if (simd_lane == 0) { reductions[simd_group] = sumSq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < tgs / 32; ++s) { total += reductions[s]; }
        qNormInvRMS = rsqrt(total / float(headDim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < headDim; d += tgs) {
        float w = float(q_norm_weight[d]);
        qShared[d] = float(half(qShared[d] * qNormInvRMS * w));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // rope_layout=2: only the active rotary prefix rotates; dims beyond ropeDim/2 keep the post-norm value.
    gemma_apply_split_half_rope(qShared, cos_row, sin_row, headDim, ropeDim, 2, tid, tgs);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < headDim; d += tgs) {
        qShared[d] = float(half(qShared[d]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The three SDPA passes recompute Q·K per-position so the dispatch
    // is bounded only by seqLen (no fixed-size scratch tile). Critical
    // for the global layer where slidingWindow=0 lets seqLen grow with
    // the conversation length.

    float maxScore = -INFINITY;
    for (uint s = seqStart + tid; s < seqLen; s += tgs) {
        float dot = 0.0f;
        uint cacheOffset = kvHead * kvStride + s * headDim;
        for (uint d = 0; d < headDim; ++d) {
            dot += qShared[d] * float(k_cache[cacheOffset + d]);
        }
        maxScore = max(maxScore, dot);
    }
    maxScore = simd_max(maxScore);
    if (simd_lane == 0) { reductions[simd_group] = maxScore; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float reducedMax = -INFINITY;
        for (uint g = 0; g < tgs / 32; ++g) {
            reducedMax = max(reducedMax, reductions[g]);
        }
        reductions[0] = reducedMax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scoreMax = reductions[0];

    float sumExp = 0.0f;
    for (uint s = seqStart + tid; s < seqLen; s += tgs) {
        float dot = 0.0f;
        uint cacheOffset = kvHead * kvStride + s * headDim;
        for (uint d = 0; d < headDim; ++d) {
            dot += qShared[d] * float(k_cache[cacheOffset + d]);
        }
        sumExp += exp(dot - scoreMax);
    }
    sumExp = simd_sum(sumExp);
    if (simd_lane == 0) { reductions[simd_group] = sumExp; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float reducedSum = 0.0f;
        for (uint g = 0; g < tgs / 32; ++g) { reducedSum += reductions[g]; }
        reductions[0] = reducedSum != 0.0f ? (1.0f / reducedSum) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float invSum = reductions[0];

    for (uint d = tid; d < headDim; d += tgs) {
        float acc = 0.0f;
        for (uint s = seqStart; s < seqLen; ++s) {
            float dot = 0.0f;
            uint kOffset = kvHead * kvStride + s * headDim;
            for (uint qd = 0; qd < headDim; ++qd) {
                dot += qShared[qd] * float(k_cache[kOffset + qd]);
            }
            float weight = exp(dot - scoreMax) * invSum;
            acc += weight * float(v_cache[kOffset + d]);
        }
        q_out[qOffset + d] = half(acc);
    }
}
