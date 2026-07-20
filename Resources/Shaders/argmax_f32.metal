#include <metal_stdlib>
using namespace metal;

/// Exact argmax over an fp32 vector, matching the CPU `for v where logits[v] > mx` semantics
/// (strict-greater ⇒ LOWEST index wins ties). Used by the MTP code-predictor to pick each sub-pass's
/// codebook id ON-GPU, so the 15 sub-passes chain in one command buffer with no CPU readback.
///
/// One threadgroup (32-lane SIMD). Each lane scans its strided slice building a per-lane best key
/// `(orderedValue, inverseIndex)`: `orderedValue` is a monotonic uint mapping of the float (so uint
/// compare == float compare), `inverseIndex = MAX_INDEX - i` (so the LOWEST index has the LARGEST
/// inverseIndex and wins ties — matching the CPU). A non-finite logit is given the minimum key so it
/// can never win (CPU's `>` skips NaN). The 32 lane-keys are reduced by an EXACT lexicographic
/// butterfly (`simd_shuffle_down` on both fields — not a lossy packed uint, not scalar simd_max), and
/// lane 0 writes `index = MAX_INDEX - winner.y` to `out[slot]`.
///
/// Buffers: 0 logits [n] float, 1 out [>=slot+1] uint
/// Constants: 2 n, 3 slot (output index this argmax writes to)

#define AGENT_ARGMAX_F32_INDEX_BITS 24
#define AGENT_ARGMAX_F32_MAX_INDEX  ((1u << AGENT_ARGMAX_F32_INDEX_BITS) - 1u)

// Monotonic uint key: for f1 < f2 (finite), key(f1) < key(f2). Negative floats: flip all 32 bits;
// non-negative: set the sign bit. Collapse -0 to +0 so signed-zero ties resolve via the index slot.
static inline uint agent_ordered_f32_key(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7FFFFFFFu) == 0u) bits = 0u;          // -0 == +0
    return (bits & 0x80000000u) != 0u ? (~bits) : (bits | 0x80000000u);
}

// Lexicographic: larger orderedValue wins; on equal value, larger inverseIndex (= lower index) wins.
static inline bool agent_key_gt(uint2 a, uint2 b) {
    return a.x > b.x || (a.x == b.x && a.y > b.y);
}

kernel void argmax_f32(
    device const float* logits [[buffer(0)]],
    device uint*        out    [[buffer(1)]],
    constant uint&      n      [[buffer(2)]],
    constant uint&      slot   [[buffer(3)]],
    uint lane [[thread_position_in_threadgroup]]
) {
    // Sentinel: orderedValue 0 sits below every non-NaN value (always replaced by a real logit), and
    // inverseIndex = MAX ⇒ if NOTHING wins (all-NaN, which never happens with finite logits) the result
    // is index 0 — matching the CPU's `best = 0` default rather than an out-of-range sentinel index.
    uint2 best = uint2(0u, AGENT_ARGMAX_F32_MAX_INDEX);
    for (uint i = lane; i < n; i += 32) {
        float v = logits[i];
        // Eligible iff `v > -FLT_MAX` — EXACTLY the CPU's `lg[v] > mx` with mx initialized to
        // -Float.greatestFiniteMagnitude. This one test subsumes every never-selectable value: NaN
        // (`NaN > x` is false), -inf, and -FLT_MAX itself all fail it and get the minimum key, so they
        // can never win (matching the CPU, which leaves `best` at its current value — index 0 default).
        // +inf is eligible and its ordered key ranks above all finite (CPU `+inf > mx` wins).
        uint2 key = (v > -FLT_MAX)
            ? uint2(agent_ordered_f32_key(v), AGENT_ARGMAX_F32_MAX_INDEX - i)
            : uint2(0u, 0u);
        if (agent_key_gt(key, best)) best = key;
    }
    // Exact 32-lane lexicographic butterfly reduction (no barriers; both fields shuffled).
    for (uint off = 16; off > 0; off >>= 1) {
        uint2 other = uint2(simd_shuffle_down(best.x, off), simd_shuffle_down(best.y, off));
        if (agent_key_gt(other, best)) best = other;
    }
    if (lane == 0) out[slot] = AGENT_ARGMAX_F32_MAX_INDEX - best.y;
}
