#include <metal_stdlib>
using namespace metal;

/// On-GPU codebook-0 selection = applyCb0Processors + argmax, exactly matching the CPU reference
/// (Qwen3TTSGenerator.applyCb0Processors then `for v where l[v] > mx`), so the talker frame doesn't
/// round-trip 3072 logits + CPU argmax to the host — only the resulting cb0 (1 uint) comes back (for
/// the EOS check). One 32-lane SIMD; each lane scans a strided slice computing the EFFECTIVE logit:
///   1. repetition penalty: if v ∈ history, scale (v>0 ? v/penalty : v*penalty) — applied once (break);
///   2. suppress [suppressFrom, n) except eos → -inf;
///   3. min_new_tokens: v == eos and frame < minNewTokens → -inf.
/// Then the SAME exact ordered-key argmax as argmax_f32 (strict-greater ⇒ lowest index wins ties;
/// `eff > -FLT_MAX` eligibility ⇒ the -inf-suppressed tokens can never win — matching the CPU `> mx`).
/// The processor order matches the CPU (penalty, then range-suppress, then eos-suppress overwrite), and
/// /,* are IEEE-correctly-rounded so the effective logits are bit-identical → cb0 bit-identical.
///
/// Buffers: 0 logits [n] float, 1 history [>=historyLen] uint, 2 out [>=1] uint
/// Constants: 3 n, 4 historyLen, 5 frame, 6 penalty (float), 7 suppressFrom, 8 eos, 9 minNewTokens

#define AGENT_CB0_INDEX_BITS 24
#define AGENT_CB0_MAX_INDEX  ((1u << AGENT_CB0_INDEX_BITS) - 1u)

static inline uint agent_cb0_ordered_key(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7FFFFFFFu) == 0u) bits = 0u;          // -0 == +0
    return (bits & 0x80000000u) != 0u ? (~bits) : (bits | 0x80000000u);
}
static inline bool agent_cb0_key_gt(uint2 a, uint2 b) {
    return a.x > b.x || (a.x == b.x && a.y > b.y);
}

kernel void cb0_argmax_f32(
    device const float* logits      [[buffer(0)]],
    device const uint*  history     [[buffer(1)]],
    device uint*        out         [[buffer(2)]],
    constant uint&      n           [[buffer(3)]],
    constant uint&      historyLen  [[buffer(4)]],
    constant uint&      frame       [[buffer(5)]],
    constant float&     penalty     [[buffer(6)]],
    constant uint&      suppressFrom [[buffer(7)]],
    constant uint&      eos         [[buffer(8)]],
    constant uint&      minNewTokens [[buffer(9)]],
    uint lane [[thread_position_in_threadgroup]]
) {
    uint2 best = uint2(0u, AGENT_CB0_MAX_INDEX);
    for (uint i = lane; i < n; i += 32) {
        float v = logits[i];
        bool inHist = false;
        for (uint j = 0; j < historyLen; j++) { if (history[j] == i) { inHist = true; break; } }
        if (inHist) v = (v > 0.0f) ? (v / penalty) : (v * penalty);
        if (i >= suppressFrom && i != eos) v = -INFINITY;
        if (i == eos && frame < minNewTokens) v = -INFINITY;
        uint2 key = (v > -FLT_MAX)
            ? uint2(agent_cb0_ordered_key(v), AGENT_CB0_MAX_INDEX - i)
            : uint2(0u, 0u);
        if (agent_cb0_key_gt(key, best)) best = key;
    }
    for (uint off = 16; off > 0; off >>= 1) {
        uint2 other = uint2(simd_shuffle_down(best.x, off), simd_shuffle_down(best.y, off));
        if (agent_cb0_key_gt(other, best)) best = other;
    }
    if (lane == 0) out[0] = AGENT_CB0_MAX_INDEX - best.y;
}
