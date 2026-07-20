#include <metal_stdlib>
using namespace metal;

/// Temperature + top-k categorical sampling over an fp32 logit vector, writing the drawn index to
/// out[slot]. The MTP code-predictor calls this instead of argmax_f32 when sampling, so the 15
/// sub-passes still chain in one command buffer (the next sub-pass gathers its embedding from the
/// sampled idxBuf with no CPU round-trip). Mirrors the host Qwen3TTSSampler.sampleTopK: threshold =
/// the k-th largest logit, keep every token with logit >= threshold (HF TopKLogitsWarper tie
/// semantics), weight exp((logit - max)/temperature), inverse-CDF walk in index order against
/// uniforms[slot]*total. fp32 vs the host's fp64 accumulation, so this is distribution-equivalent
/// (gated by a frequency sweep), not bit-identical. Requires temperature > 0 (greedy is argmax_f32).
///
/// Threshold extraction runs k SIMD rounds: each round reduces, across the 32 lanes, the largest
/// ordered key strictly below the previous round's winner (so ties resolve by index, exactly k
/// tokens are extracted). All lanes track maxVal (round 0) and threshold (round k-1) from the
/// broadcast winner; lane 0 then walks the CDF serially (fp-consistent total + walk, with the
/// last-kept-token fallback for the u≈1 rounding boundary).
///
/// Buffers: 0 logits [n] float, 1 uniforms [>=slot+1] float, 2 out [>=slot+1] uint
/// Constants: 3 n, 4 slot, 5 temperature (float), 6 topK

#define STK_INDEX_BITS 24
#define STK_MAX_INDEX  ((1u << STK_INDEX_BITS) - 1u)

static inline uint stk_ordered_key(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7FFFFFFFu) == 0u) bits = 0u;          // -0 == +0
    return (bits & 0x80000000u) != 0u ? (~bits) : (bits | 0x80000000u);
}
static inline bool stk_key_gt(uint2 a, uint2 b) { return a.x > b.x || (a.x == b.x && a.y > b.y); }
static inline bool stk_key_lt(uint2 a, uint2 b) { return a.x < b.x || (a.x == b.x && a.y < b.y); }

kernel void sample_topk_f32(
    device const float* logits      [[buffer(0)]],
    device const float* uniforms    [[buffer(1)]],
    device uint*        out         [[buffer(2)]],
    constant uint&      n           [[buffer(3)]],
    constant uint&      slot        [[buffer(4)]],
    constant float&     temperature [[buffer(5)]],
    constant uint&      topK        [[buffer(6)]],
    uint lane [[thread_position_in_threadgroup]]
) {
    uint k = min(max(topK, 1u), n);                     // mirror host clamp; topK==0 would degenerate
    uint2 lastKey = uint2(0xFFFFFFFFu, 0xFFFFFFFFu);    // sentinel above every real key
    float maxVal = -INFINITY, threshold = -INFINITY;
    for (uint r = 0; r < k; r++) {
        // Largest key strictly below lastKey across this lane's strided slice.
        uint2 best = uint2(0u, 0u);
        for (uint i = lane; i < n; i += 32) {
            uint2 key = uint2(stk_ordered_key(logits[i]), STK_MAX_INDEX - i);
            if (stk_key_lt(key, lastKey) && stk_key_gt(key, best)) best = key;
        }
        for (uint off = 16; off > 0; off >>= 1) {
            uint2 o = uint2(simd_shuffle_down(best.x, off), simd_shuffle_down(best.y, off));
            if (stk_key_gt(o, best)) best = o;
        }
        best = uint2(simd_broadcast(best.x, 0), simd_broadcast(best.y, 0));  // winner to all lanes
        lastKey = best;
        float v = logits[STK_MAX_INDEX - best.y];
        if (r == 0) maxVal = v;
        threshold = v;                                   // round k-1's value sticks
    }
    if (lane != 0) return;
    float invT = 1.0f / temperature;
    // `> -INFINITY` excludes suppressed tokens even when topK reaches into the suppressed tail
    // (threshold == -inf): a zero-weight -inf entry must never be selectable. No-op when threshold
    // is finite (the normal case). Mirrors the host Qwen3TTSSampler.sampleTopK keep filter.
    float total = 0.0f;
    for (uint i = 0; i < n; i++) {
        if (logits[i] > -INFINITY && logits[i] >= threshold) total += exp((logits[i] - maxVal) * invT);
    }
    float target = uniforms[slot] * total;
    float acc = 0.0f;
    uint last = 0;
    for (uint i = 0; i < n; i++) {
        if (logits[i] > -INFINITY && logits[i] >= threshold) {
            acc += exp((logits[i] - maxVal) * invT);
            last = i;
            if (acc >= target) { out[slot] = i; return; }
        }
    }
    out[slot] = last;                                    // u≈1 rounding: the last kept token
}
