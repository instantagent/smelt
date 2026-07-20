#include <metal_stdlib>
using namespace metal;

struct AgentRopePair {
    uint d0;
    uint d1;
    uint c0Index;
    uint c1Index;
};

static inline AgentRopePair agent_rope_pair_indices(
    uint pair,
    uint ropeLayout,
    uint headDim,
    uint ropeDim
) {
    uint halfRope = ropeDim / 2;
    AgentRopePair indices;
    if (ropeLayout == 1) {
        indices.d0 = pair;
        indices.d1 = pair + halfRope;
        indices.c0Index = indices.d0;
        indices.c1Index = indices.d1;
    } else if (ropeLayout == 2) {
        indices.d0 = pair;
        indices.d1 = pair + headDim / 2;
        indices.c0Index = pair;
        indices.c1Index = pair + halfRope;
    } else {
        indices.d0 = pair * 2;
        indices.d1 = indices.d0 + 1;
        indices.c0Index = indices.d0;
        indices.c1Index = indices.d1;
    }
    return indices;
}

static inline bool agent_rope_dim_active(
    uint d,
    uint ropeLayout,
    uint ropeDim,
    uint headDim
) {
    if (ropeLayout == 2) {
        uint halfRope = ropeDim / 2;
        uint secondStart = headDim / 2;
        return d < halfRope || (d >= secondStart && d < secondStart + halfRope);
    }
    return d < ropeDim;
}

// ─── Batched RoPE + KV cache write for prefill ───
//
// Applies RoPE to Q and K for all positions in parallel, then writes
// K and V to the KV cache. Replaces 4 × B unrolled dispatches with 1.
//
// Layout: Q [B, qHeads, headDim], K [B, kvHeads, headDim], V same
// RoPE cos/sin tables: [maxSeqLen, ropeDim]
// KV cache: [kvHeads, cacheSeqCapacity, headDim]
//
// Dispatch: (B, max(qHeads, kvHeads)) threadgroups, headDim threads each.
// Each threadgroup handles one (position, head) pair.

kernel void rope_and_kv_cache_prefill(
    device half*       queries    [[buffer(0)]],  // [B, qHeads, headDim] in-place RoPE
    device half*       keys       [[buffer(1)]],  // [B, kvHeads, headDim] in-place RoPE
    device const half* values     [[buffer(2)]],  // [B, kvHeads, headDim]
    device const half* cos_table  [[buffer(3)]],  // [maxSeqLen, ropeDim]
    device const half* sin_table  [[buffer(4)]],  // [maxSeqLen, ropeDim]
    device half*       key_cache  [[buffer(5)]],  // [kvHeads, cacheSeqCapacity, headDim]
    device half*       val_cache  [[buffer(6)]],  // [kvHeads, cacheSeqCapacity, headDim]
    constant uint&     headDim    [[buffer(7)]],
    constant uint&     ropeDim    [[buffer(8)]],
    constant uint&     qHeads     [[buffer(9)]],
    constant uint&     kvHeads    [[buffer(10)]],
    constant uint&     seqLen     [[buffer(11)]],
    constant uint&     startPos   [[buffer(12)]],
    constant uint&     cacheSeqCapacity [[buffer(13)]],
    constant uint&     ropeLayout [[buffer(14)]], // 0 = adjacent pairs, 1 = split-half, 2 = proportional split-half
    uint3 tgid [[threadgroup_position_in_grid]],   // (position, head)
    uint tid   [[thread_index_in_threadgroup]],
    uint3 tgs_v [[threads_per_threadgroup]]
) {
    uint pos = tgid.x;
    uint head = tgid.y;
    uint tgs = tgs_v.x;

    if (pos >= seqLen) return;

    uint absPos = startPos + pos;  // absolute position for RoPE lookup

    // RoPE cos/sin for this position
    device const half* cos_row = cos_table + absPos * ropeDim;
    device const half* sin_row = sin_table + absPos * ropeDim;

    // Apply RoPE to Q (if this head index < qHeads).
    // One thread owns one even/odd pair to avoid the in-place race where
    // two threads read/write the same pair concurrently.
    if (head < qHeads) {
        uint qOff = pos * qHeads * headDim + head * headDim;
        uint halfRope = ropeDim / 2;
        for (uint pair = tid; pair < halfRope; pair += tgs) {
            uint d0;
            uint d1;
            uint c0Index;
            uint c1Index;
            if (ropeLayout == 1) {
                d0 = pair;
                d1 = pair + halfRope;
                c0Index = d0;
                c1Index = d1;
            } else if (ropeLayout == 2) {
                d0 = pair;
                d1 = pair + headDim / 2;
                c0Index = pair;
                c1Index = pair + halfRope;
            } else {
                d0 = pair * 2;
                d1 = d0 + 1;
                c0Index = d0;
                c1Index = d1;
            }
            float x_even = float(queries[qOff + d0]);
            float x_odd = float(queries[qOff + d1]);
            float cos_e = float(cos_row[c0Index]);
            float sin_e = float(sin_row[c0Index]);
            float cos_o = float(cos_row[c1Index]);
            float sin_o = float(sin_row[c1Index]);

            queries[qOff + d0] = half(x_even * cos_e - x_odd * sin_e);
            queries[qOff + d1] = half(x_odd * cos_o + x_even * sin_o);
        }
    }

    // Apply RoPE to K + write KV cache (if this head index < kvHeads)
    if (head < kvHeads) {
        uint kOff = pos * kvHeads * headDim + head * headDim;

        // RoPE on K with the same one-thread-per-pair ownership.
        uint halfRope = ropeDim / 2;
        for (uint pair = tid; pair < halfRope; pair += tgs) {
            uint d0;
            uint d1;
            uint c0Index;
            uint c1Index;
            if (ropeLayout == 1) {
                d0 = pair;
                d1 = pair + halfRope;
                c0Index = d0;
                c1Index = d1;
            } else if (ropeLayout == 2) {
                d0 = pair;
                d1 = pair + headDim / 2;
                c0Index = pair;
                c1Index = pair + halfRope;
            } else {
                d0 = pair * 2;
                d1 = d0 + 1;
                c0Index = d0;
                c1Index = d1;
            }
            float x_even = float(keys[kOff + d0]);
            float x_odd = float(keys[kOff + d1]);
            float cos_e = float(cos_row[c0Index]);
            float sin_e = float(sin_row[c0Index]);
            float cos_o = float(cos_row[c1Index]);
            float sin_o = float(sin_row[c1Index]);

            keys[kOff + d0] = half(x_even * cos_e - x_odd * sin_e);
            keys[kOff + d1] = half(x_odd * cos_o + x_even * sin_o);
        }

        threadgroup_barrier(mem_flags::mem_device);

        // Write K to cache: cache[head, absPos, dim]
        uint cacheOff = head * cacheSeqCapacity * headDim + absPos * headDim;
        for (uint d = tid; d < headDim; d += tgs) {
            key_cache[cacheOff + d] = keys[kOff + d];
        }

        // Write V to cache
        uint vOff = pos * kvHeads * headDim + head * headDim;
        for (uint d = tid; d < headDim; d += tgs) {
            val_cache[cacheOff + d] = values[vOff + d];
        }
    }
}

kernel void fused_norm_rope_and_kv_cache_prefill(
    device half*       queries    [[buffer(0)]],  // [B, qHeads, headDim] in-place RoPE
    device half*       keys       [[buffer(1)]],  // keys, or Q norm input when kvHeads == 0
    device const half* values     [[buffer(2)]],  // values, or Q norm weight when kvHeads == 0
    device const half* cos_table  [[buffer(3)]],  // [maxSeqLen, ropeDim]
    device const half* sin_table  [[buffer(4)]],  // [maxSeqLen, ropeDim]
    device half*       key_cache  [[buffer(5)]],  // [kvHeads, cacheSeqCapacity, headDim]
    device half*       val_cache  [[buffer(6)]],  // [kvHeads, cacheSeqCapacity, headDim]
    constant uint&     headDim    [[buffer(7)]],
    constant uint&     ropeDim    [[buffer(8)]],
    constant uint&     qHeads     [[buffer(9)]],
    constant uint&     kvHeads    [[buffer(10)]],
    constant uint&     seqLen     [[buffer(11)]],
    constant uint&     startPos   [[buffer(12)]],
    constant uint&     cacheSeqCapacity [[buffer(13)]],
    constant uint&     ropeLayout [[buffer(14)]],
    constant float&    rmsEps     [[buffer(15)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint3 tgs_v [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint pos = tgid.x;
    uint head = tgid.y;
    uint tgs = tgs_v.x;

    if (pos >= seqLen) return;

    uint absPos = startPos + pos;
    device const half* cos_row = cos_table + absPos * ropeDim;
    device const half* sin_row = sin_table + absPos * ropeDim;

    if (head < qHeads) {
        uint qOff = pos * qHeads * headDim + head * headDim;
        uint halfRope = ropeDim / 2;

        if (kvHeads == 0) {
            device const half* qNormInput = keys;
            device const half* qNormWeight = values;

            float sumSq = 0.0f;
            for (uint d = tid; d < headDim; d += tgs) {
                float v = float(qNormInput[qOff + d]);
                sumSq += v * v;
            }

            sumSq = simd_sum(sumSq);

            threadgroup float partial[32];
            if (simd_lane == 0) {
                partial[simd_group] = sumSq;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup float shared_rsqrt = 0.0f;
            if (tid == 0) {
                float total = 0.0f;
                uint nSimds = (tgs + 31) / 32;
                for (uint s = 0; s < nSimds; s++) {
                    total += partial[s];
                }
                float mean = total / float(headDim);
                shared_rsqrt = rsqrt(mean + rmsEps);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float rs = shared_rsqrt;
            for (uint pair = tid; pair < halfRope; pair += tgs) {
                AgentRopePair indices = agent_rope_pair_indices(
                    pair,
                    ropeLayout,
                    headDim,
                    ropeDim
                );
                float x_even = float(qNormInput[qOff + indices.d0])
                    * rs * float(qNormWeight[indices.d0]);
                float x_odd = float(qNormInput[qOff + indices.d1])
                    * rs * float(qNormWeight[indices.d1]);
                float cos_e = float(cos_row[indices.c0Index]);
                float sin_e = float(sin_row[indices.c0Index]);
                float cos_o = float(cos_row[indices.c1Index]);
                float sin_o = float(sin_row[indices.c1Index]);

                queries[qOff + indices.d0] = half(x_even * cos_e - x_odd * sin_e);
                queries[qOff + indices.d1] = half(x_odd * cos_o + x_even * sin_o);
            }

            for (uint d = tid; d < headDim; d += tgs) {
                if (!agent_rope_dim_active(d, ropeLayout, ropeDim, headDim)) {
                    queries[qOff + d] = half(
                        float(qNormInput[qOff + d]) * rs * float(qNormWeight[d])
                    );
                }
            }
        } else {
            for (uint pair = tid; pair < halfRope; pair += tgs) {
                uint d0;
                uint d1;
                uint c0Index;
                uint c1Index;
                if (ropeLayout == 1) {
                    d0 = pair;
                    d1 = pair + halfRope;
                    c0Index = d0;
                    c1Index = d1;
                } else if (ropeLayout == 2) {
                    d0 = pair;
                    d1 = pair + headDim / 2;
                    c0Index = pair;
                    c1Index = pair + halfRope;
                } else {
                    d0 = pair * 2;
                    d1 = d0 + 1;
                    c0Index = d0;
                    c1Index = d1;
                }
                float x_even = float(queries[qOff + d0]);
                float x_odd = float(queries[qOff + d1]);
                float cos_e = float(cos_row[c0Index]);
                float sin_e = float(sin_row[c0Index]);
                float cos_o = float(cos_row[c1Index]);
                float sin_o = float(sin_row[c1Index]);

                queries[qOff + d0] = half(x_even * cos_e - x_odd * sin_e);
                queries[qOff + d1] = half(x_odd * cos_o + x_even * sin_o);
            }
        }
    }

    if (head < kvHeads) {
        uint kOff = pos * kvHeads * headDim + head * headDim;

        uint halfRope = ropeDim / 2;
        for (uint pair = tid; pair < halfRope; pair += tgs) {
            uint d0;
            uint d1;
            uint c0Index;
            uint c1Index;
            if (ropeLayout == 1) {
                d0 = pair;
                d1 = pair + halfRope;
                c0Index = d0;
                c1Index = d1;
            } else if (ropeLayout == 2) {
                d0 = pair;
                d1 = pair + headDim / 2;
                c0Index = pair;
                c1Index = pair + halfRope;
            } else {
                d0 = pair * 2;
                d1 = d0 + 1;
                c0Index = d0;
                c1Index = d1;
            }
            float x_even = float(keys[kOff + d0]);
            float x_odd = float(keys[kOff + d1]);
            float cos_e = float(cos_row[c0Index]);
            float sin_e = float(sin_row[c0Index]);
            float cos_o = float(cos_row[c1Index]);
            float sin_o = float(sin_row[c1Index]);

            keys[kOff + d0] = half(x_even * cos_e - x_odd * sin_e);
            keys[kOff + d1] = half(x_odd * cos_o + x_even * sin_o);
        }

        threadgroup_barrier(mem_flags::mem_device);

        uint cacheOff = head * cacheSeqCapacity * headDim + absPos * headDim;
        for (uint d = tid; d < headDim; d += tgs) {
            key_cache[cacheOff + d] = keys[kOff + d];
        }

        uint vOff = pos * kvHeads * headDim + head * headDim;
        float sumSq = 0.0f;
        for (uint d = tid; d < headDim; d += tgs) {
            float v = float(values[vOff + d]);
            sumSq += v * v;
        }

        sumSq = simd_sum(sumSq);

        threadgroup float partial[32];
        if (simd_lane == 0) {
            partial[simd_group] = sumSq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float shared_rsqrt = 0.0f;
        if (tid == 0) {
            float total = 0.0f;
            uint nSimds = (tgs + 31) / 32;
            for (uint s = 0; s < nSimds; s++) {
                total += partial[s];
            }
            float mean = total / float(headDim);
            shared_rsqrt = rsqrt(mean + rmsEps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rs = shared_rsqrt;
        for (uint d = tid; d < headDim; d += tgs) {
            val_cache[cacheOff + d] = half(float(values[vOff + d]) * rs);
        }
    }
}
