#include <metal_stdlib>
using namespace metal;

/// On-GPU codebook-0 SAMPLED selection = applyCb0Processors + temperature/top-k sampling — the sampled
/// twin of cb0_argmax_f32 (processors + argmax). It folds the cb0 logit processors (repetition penalty,
/// range-suppress, min-new-token eos-suppress) into sample_topk_f32's threshold+CDF body, so a sampled
/// cb0 never round-trips 3072 logits + host processors + host sampleTopK — only the drawn cb0 (1 uint)
/// comes back (for the EOS check). The processors are recomputed per logit read via cb0_effective_logit,
/// called by BOTH the threshold-extraction pass AND the CDF re-read (sample_topk reads each logit twice),
/// so the two passes can never drift. fp32 CDF accumulation vs the host's fp64 makes this
/// distribution-equivalent (gated by a seed-fixed frequency sweep), exactly like sample_topk_f32; the
/// processors + the uniform draw are bit-identical (cb0_argmax already proves the processed logits are
/// bit-exact), only the CDF rounding differs. Requires temperature > 0 (greedy is cb0_argmax_f32).
///
/// Buffers: 0 logits [n] float, 1 history [>=historyLen] uint, 2 uniforms [>=1] float, 3 out [>=1] uint
/// Constants: 4 n, 5 historyLen, 6 frame, 7 penalty (float), 8 suppressFrom, 9 eos, 10 minNewTokens,
///            11 temperature (float), 12 topK

#define STK_INDEX_BITS 24
#define STK_MAX_INDEX  ((1u << STK_INDEX_BITS) - 1u)

static inline uint stk_ordered_key(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7FFFFFFFu) == 0u) bits = 0u;          // -0 == +0
    return (bits & 0x80000000u) != 0u ? (~bits) : (bits | 0x80000000u);
}
static inline bool stk_key_gt(uint2 a, uint2 b) { return a.x > b.x || (a.x == b.x && a.y > b.y); }
static inline bool stk_key_lt(uint2 a, uint2 b) { return a.x < b.x || (a.x == b.x && a.y < b.y); }

/// The cb0 effective logit at index `i`: logits[i] after the three processors, IEEE-correctly-rounded so
/// the result is bit-identical to Qwen3TTSGenerator.applyCb0Processors (same order: repetition penalty,
/// then range-suppress, then eos-suppress overwrite). Matches cb0_argmax_f32's per-lane prelude exactly.
static inline float cb0_effective_logit(
    device const float* logits, device const uint* history, uint i,
    uint historyLen, uint frame, float penalty, uint suppressFrom, uint eos, uint minNewTokens
) {
    float v = logits[i];
    bool inHist = false;
    for (uint j = 0; j < historyLen; j++) { if (history[j] == i) { inHist = true; break; } }
    if (inHist) v = (v > 0.0f) ? (v / penalty) : (v * penalty);
    if (i >= suppressFrom && i != eos) v = -INFINITY;
    if (i == eos && frame < minNewTokens) v = -INFINITY;
    return v;
}

kernel void cb0_sample_topk_f32(
    device const float* logits       [[buffer(0)]],
    device const uint*  history      [[buffer(1)]],
    device const float* uniforms     [[buffer(2)]],
    device uint*        out          [[buffer(3)]],
    constant uint&      n            [[buffer(4)]],
    constant uint&      historyLen   [[buffer(5)]],
    constant uint&      frame        [[buffer(6)]],
    constant float&     penalty      [[buffer(7)]],
    constant uint&      suppressFrom [[buffer(8)]],
    constant uint&      eos          [[buffer(9)]],
    constant uint&      minNewTokens [[buffer(10)]],
    constant float&     temperature  [[buffer(11)]],
    constant uint&      topK         [[buffer(12)]],
    uint lane [[thread_position_in_threadgroup]]
) {
    uint k = min(max(topK, 1u), n);                     // mirror host clamp; topK==0 would degenerate
    uint2 lastKey = uint2(0xFFFFFFFFu, 0xFFFFFFFFu);    // sentinel above every real key
    float maxVal = -INFINITY, threshold = -INFINITY;
    for (uint r = 0; r < k; r++) {
        // Largest effective-logit key strictly below lastKey across this lane's strided slice.
        uint2 best = uint2(0u, 0u);
        for (uint i = lane; i < n; i += 32) {
            float eff = cb0_effective_logit(logits, history, i, historyLen, frame, penalty,
                                            suppressFrom, eos, minNewTokens);
            uint2 key = uint2(stk_ordered_key(eff), STK_MAX_INDEX - i);
            if (stk_key_lt(key, lastKey) && stk_key_gt(key, best)) best = key;
        }
        for (uint off = 16; off > 0; off >>= 1) {
            uint2 o = uint2(simd_shuffle_down(best.x, off), simd_shuffle_down(best.y, off));
            if (stk_key_gt(o, best)) best = o;
        }
        best = uint2(simd_broadcast(best.x, 0), simd_broadcast(best.y, 0));  // winner to all lanes
        lastKey = best;
        float v = cb0_effective_logit(logits, history, STK_MAX_INDEX - best.y, historyLen, frame,
                                      penalty, suppressFrom, eos, minNewTokens);
        if (r == 0) maxVal = v;
        threshold = v;                                   // round k-1's value sticks
    }
    if (lane != 0) return;
    float invT = 1.0f / temperature;
    // `> -INFINITY` excludes suppressed tokens even when topK reaches into the suppressed tail
    // (threshold == -inf): a zero-weight -inf entry must never be selectable. No-op when threshold is
    // finite (the normal case). Mirrors the host Qwen3TTSSampler.sampleTopK keep filter.
    float total = 0.0f;
    for (uint i = 0; i < n; i++) {
        float eff = cb0_effective_logit(logits, history, i, historyLen, frame, penalty,
                                        suppressFrom, eos, minNewTokens);
        if (eff > -INFINITY && eff >= threshold) total += exp((eff - maxVal) * invT);
    }
    // total == 0 only if EVERY token is suppressed (all `eff == -inf`) — unreachable for a valid cb0
    // config: the audio range [0, suppressFrom) is never suppressed, so finite tokens in [0, suppressFrom)
    // always exist. (The host sampler traps this case; here a malformed all-suppressed config would fall
    // through to out[0] = last == 0 rather than trap. Documented, not defended — the real package config
    // makes it impossible.)
    float target = uniforms[0] * total;
    float acc = 0.0f;
    uint last = 0;
    for (uint i = 0; i < n; i++) {
        float eff = cb0_effective_logit(logits, history, i, historyLen, frame, penalty,
                                        suppressFrom, eos, minNewTokens);
        if (eff > -INFINITY && eff >= threshold) {
            acc += exp((eff - maxVal) * invT);
            last = i;
            if (acc >= target) { out[0] = i; return; }
        }
    }
    out[0] = last;                                       // u≈1 rounding: the last kept token
}
