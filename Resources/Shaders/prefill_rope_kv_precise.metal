#include <metal_stdlib>
using namespace metal;

// Analytic RoPE + KV-cache write for unscaled rotary embeddings. This is the
// batched form of attention.metal's MLX-compatible apply_rope path: each
// (position, head) owns its pairs, and trig stays FP32 until the output store.
// Scaled-RoPE models continue to use the table-based prefill brick.
kernel void rope_and_kv_cache_prefill_analytic(
    device half* queries        [[buffer(0)]],
    device half* keys           [[buffer(1)]],
    device const half* values   [[buffer(2)]],
    device half* key_cache      [[buffer(3)]],
    device half* val_cache      [[buffer(4)]],
    constant uint& packedDims   [[buffer(5)]], // headDim | ropeDim << 16
    constant uint& packedHeads  [[buffer(6)]], // qHeads | kvHeads << 16
    constant uint& seqLen       [[buffer(7)]],
    constant uint& startPos     [[buffer(8)]],
    constant uint& cacheSeqCapacity [[buffer(9)]],
    constant uint& ropeLayout   [[buffer(10)]],
    constant float& baseLog2    [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tgs_v [[threads_per_threadgroup]]
) {
    const uint headDim = packedDims & 0xffffu;
    const uint ropeDim = packedDims >> 16;
    const uint qHeads = packedHeads & 0xffffu;
    const uint kvHeads = packedHeads >> 16;
    const uint pos = tgid.x;
    const uint head = tgid.y;
    const uint tgs = tgs_v.x;
    if (pos >= seqLen) return;

    const uint absPos = startPos + pos;
    const uint halfRope = ropeDim / 2;

    if (head < qHeads) {
        const uint qOff = pos * qHeads * headDim + head * headDim;
        for (uint pair = tid; pair < halfRope; pair += tgs) {
            uint d0;
            uint d1;
            if (ropeLayout == 1) {
                d0 = pair;
                d1 = pair + halfRope;
            } else if (ropeLayout == 2) {
                d0 = pair;
                d1 = pair + headDim / 2;
            } else {
                d0 = pair * 2;
                d1 = d0 + 1;
            }

            const float d = float(pair) / float(halfRope);
            const float invFreq = metal::exp2(-d * baseLog2);
            const float angle = float(absPos) * invFreq;
            const float c = metal::fast::cos(angle);
            const float s = metal::fast::sin(angle);
            const float x0 = float(queries[qOff + d0]);
            const float x1 = float(queries[qOff + d1]);
            queries[qOff + d0] = half(x0 * c - x1 * s);
            queries[qOff + d1] = half(x0 * s + x1 * c);
        }
    }

    if (head < kvHeads) {
        const uint kOff = pos * kvHeads * headDim + head * headDim;
        for (uint pair = tid; pair < halfRope; pair += tgs) {
            uint d0;
            uint d1;
            if (ropeLayout == 1) {
                d0 = pair;
                d1 = pair + halfRope;
            } else if (ropeLayout == 2) {
                d0 = pair;
                d1 = pair + headDim / 2;
            } else {
                d0 = pair * 2;
                d1 = d0 + 1;
            }

            const float d = float(pair) / float(halfRope);
            const float invFreq = metal::exp2(-d * baseLog2);
            const float angle = float(absPos) * invFreq;
            const float c = metal::fast::cos(angle);
            const float s = metal::fast::sin(angle);
            const float x0 = float(keys[kOff + d0]);
            const float x1 = float(keys[kOff + d1]);
            keys[kOff + d0] = half(x0 * c - x1 * s);
            keys[kOff + d1] = half(x0 * s + x1 * c);
        }

        threadgroup_barrier(mem_flags::mem_device);
        const uint cacheOff =
            head * cacheSeqCapacity * headDim + absPos * headDim;
        const uint vOff = pos * kvHeads * headDim + head * headDim;
        for (uint dim = tid; dim < headDim; dim += tgs) {
            key_cache[cacheOff + dim] = keys[kOff + dim];
            val_cache[cacheOff + dim] = values[vOff + dim];
        }
    }
}
