#include <metal_stdlib>
using namespace metal;

/// SIMD-parallel sliding-window causal MHA, fp32 — BIT-EXACT replacement for sliding_attn_f32
/// (one THREAD per (t, head), ~3 sequential passes over window×headDim — the same scalar-thread
/// pathology as causal_gqa_attn_f32). One 32-lane SIMD per (t, head):
///   - scores: lanes take keys s = lo + lane + 32·i; the inner d-loop is SOURCE-IDENTICAL to the
///     scalar kernel → per-score bits unchanged. Staged in threadgroup memory (window <= 72).
///   - mx: fp max is exact under any association.
///   - denom: every lane redundantly accumulates exp(sc[i]-mx) over s ASCENDING (scalar order).
///   - out: lanes take dims d; per d the s-loop is sequential ascending with the scalar kernel's
///     exact term shape acc += (e/denom)·v (the divide INSIDE the sum, per term, as the scalar
///     kernel does after its in-place e overwrite).
/// exp() is the same metal::exp in the same metallib → identical results.
///
/// Buffers: 0 q, 1 k, 2 v, 3 out  (all [frames, heads*headDim])
/// Constants: 4 frames, 5 heads, 6 headDim, 7 window
/// Dispatch: grid (frames·32, heads, 1), threadsPerThreadgroup (32, 1, 1). window <= 72.
kernel void sliding_attn_simd_f32(
    device const float* q       [[buffer(0)]],
    device const float* k       [[buffer(1)]],
    device const float* v       [[buffer(2)]],
    device float*       out     [[buffer(3)]],
    constant uint&      frames  [[buffer(4)]],
    constant uint&      heads   [[buffer(5)]],
    constant uint&      headDim [[buffer(6)]],
    constant uint&      window  [[buffer(7)]],
    uint2 gid  [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_threadgroup]]
) {
    threadgroup float sc[72];   // window <= 72 (scalar kernel's local array, now shared)
    uint t = gid.x;
    uint hd = gid.y;
    if (t >= frames || hd >= heads) return;

    uint attnDim = heads * headDim;
    uint qb = t * attnDim + hd * headDim;
    float scaling = 1.0f / sqrt(float(headDim));
    uint lo = (t + 1 > window) ? (t + 1 - window) : 0u;

    // Scores, lane-parallel across key positions; inner loop source-identical to the scalar kernel.
    for (uint s = lo + lane; s <= t; s += 32) {
        uint kb = s * attnDim + hd * headDim;
        float dot = 0.0f;
        for (uint d = 0; d < headDim; d++) dot += q[qb + d] * k[kb + d];
        sc[s - lo] = dot * scaling;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Exact max + the scalar kernel's sequential-ascending denom, replicated per lane.
    float mx = -INFINITY;
    for (uint s = lo; s <= t; s++) mx = max(mx, sc[s - lo]);
    float denom = 0.0f;
    for (uint s = lo; s <= t; s++) denom += exp(sc[s - lo] - mx);

    // Output dims lane-parallel; per dim the s-loop is sequential ascending with the scalar
    // kernel's per-term (e/denom)·v shape.
    uint ob = t * attnDim + hd * headDim;
    for (uint d = lane; d < headDim; d += 32) {
        float acc = 0.0f;
        for (uint s = lo; s <= t; s++) {
            float e = exp(sc[s - lo] - mx);
            acc += (e / denom) * v[s * attnDim + hd * headDim + d];
        }
        out[ob + d] = acc;
    }
}
