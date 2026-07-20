// cluster.metal — EAGLE-class drafter sparse lm_head.
//
// Replaces the dense lm_head matvec over the full vocab with a top-k
// centroid gather. For Gemma 4 E2B drafter (num_centroids=2048,
// top_k=32, tokens_per_cluster=128, vocab=262144), only 4096 of
// 262144 logits are computed; non-selected positions get `-INFINITY`.
//
// Dispatch shape (set by TopLevelEmitter):
//   threadgroups: num_centroids × 1 × 1
//   threads/group: tokens_per_cluster × 1 × 1
//
//   threadgroup_id        = my centroid index (0..num_centroids-1)
//   thread_id_in_group    = my token slot within the cluster
//                            (0..tokens_per_cluster-1)
//   permuted_vocab_pos    = my_centroid * tokens_per_cluster + my_slot
//   real_vocab_index      = token_ordering[permuted_vocab_pos]
//
// Each threadgroup independently determines whether its centroid is
// in the top-k by counting how many other centroid_logits are
// strictly greater than its own. Rank < top_k → selected. No global
// synchronization, no top-k indices buffer; cost is
// O(num_centroids) reads per threadgroup, O(num_centroids^2) total
// (~4 M reads for the canonical 2048-centroid shape — negligible
// next to the matvec on selected clusters).
//
// Coverage invariant for well-formed packages: every vocab slot is
// written by exactly one thread. validateAgentIR enforces
// `tokens_per_cluster <= 256` so the dispatch's tgWidth cap covers
// every slot in every cluster, and `vocab_size % num_centroids == 0`
// so cluster boundaries align. A malformed `token_ordering` (out of
// range or sign-extended) leaves a vocab slot uninitialized — the
// thread returns silently rather than writing to an OOB index. v1
// trusts the converter; runtime tooling can pre-fill -inf if
// paranoia is warranted.

#include <metal_stdlib>
using namespace metal;

// ─── cluster_sparse_lm_head ───
//
// Inputs:
//   centroid_logits [num_centroids] fp16     — output of the centroid
//                                                projection matvec.
//   lm_head_weight  [vocab, hidden] fp16    — flat row-major; row i
//                                                spans hidden elements
//                                                starting at i*hidden.
//   hidden_state    [hidden] fp16             — the post-final-norm
//                                                drafter hidden state.
//   token_ordering  [vocab] int32             — permutation buffer
//                                                grouping vocab into
//                                                contiguous clusters.
//   logits          [vocab] fp16                — output. Selected
//                                                positions get the
//                                                dot-product score;
//                                                others get -INFINITY.
//
// Constants:
//   num_centroids, top_k, vocab_size, hidden_size, tokens_per_cluster,
//   logit_cap (Gemma-style softcap; 0 = disabled).

kernel void cluster_sparse_lm_head(
    device const half* centroid_logits  [[buffer(0)]],
    device const half* lm_head_weight   [[buffer(1)]],
    device const half* hidden_state     [[buffer(2)]],
    device const int*  token_ordering   [[buffer(3)]],
    device half*       logits           [[buffer(4)]],
    constant uint&     num_centroids    [[buffer(5)]],
    constant uint&     top_k            [[buffer(6)]],
    constant uint&     vocab_size       [[buffer(7)]],
    constant uint&     hidden_size      [[buffer(8)]],
    constant uint&     tokens_per_cluster [[buffer(9)]],
    constant float&    logit_cap        [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]]
) {
    // Bounds: tgid < num_centroids, tid < tokens_per_cluster. Under
    // a validateAgentIR-clean config these are uniform across the
    // threadgroup (tokens_per_cluster <= 256 = tgWidth cap,
    // num_centroids = grid size), so any one returning early returns
    // for every lane — safe wrt the threadgroup_barrier below.
    if (tid >= tokens_per_cluster) return;
    if (tgid >= num_centroids) return;

    uint permuted_pos = tgid * tokens_per_cluster + tid;
    if (permuted_pos >= vocab_size) return;

    // token_ordering is data-dependent — a malformed entry on a
    // single lane would cause that lane to skip the barrier below
    // and deadlock the threadgroup, since Metal barriers require
    // every lane to participate. Read the token now, capture the
    // validity flag, and defer the bad-value early-out until after
    // the rank broadcast barrier.
    int real_token = token_ordering[permuted_pos];
    bool token_ok = (real_token >= 0 && uint(real_token) < vocab_size);

    // Compute the rank of THIS threadgroup's centroid exactly once,
    // not per-thread. Without the hoist, every one of the
    // tokens_per_cluster threads independently runs the
    // O(num_centroids) scan — for canonical E2B (2048 × 128) that's
    // 2048 * 128 * 2048 ≈ 537 M reads per decode step, more than
    // the matvec on selected clusters does. With the hoist plus a
    // threadgroup-memory broadcast it's 2048 * 2048 ≈ 4 M reads.
    //
    // Cast to float before compare — fp16 has ~3-decimal-digit
    // precision, so two logits that differ only in their bottom bits
    // can flip their compare result depending on which value gets
    // promoted first inside the SIMD compare lanes. fp32 widens to
    // ~7 digits and removes the lane-dependent flip; rank then
    // matches a deterministic stable sort.
    threadgroup uint shared_rank = 0;
    if (tid == 0) {
        float my_centroid_logit = float(centroid_logits[tgid]);
        uint rank = 0;
        for (uint c = 0; c < num_centroids; c++) {
            if (c == tgid) continue;
            float other = float(centroid_logits[c]);
            if (other > my_centroid_logit) {
                rank++;
            } else if (other == my_centroid_logit && c < tgid) {
                rank++;  // stable tie-break: lower index wins
            }
        }
        shared_rank = rank;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // All threads read the threadgroup-shared rank into a per-thread
    // copy. The threadgroup-uint init at declaration silences the
    // compiler's "used uninitialized whenever the if is false"
    // warning while the runtime semantic stays correct: tid==0
    // writes the real rank, the barrier orders that write before
    // all reads, and every lane sees the same value.
    uint tg_rank = shared_rank;
    bool selected = (tg_rank < top_k);

    // Bad-token early-out is safe HERE because the barrier is past;
    // any lane that exits now leaves the rest of the threadgroup
    // already past the synchronization point.
    if (!token_ok) return;

    if (!selected) {
        logits[uint(real_token)] = -HUGE_VALH;
        return;
    }

    // Selected: dot-product lm_head[real_token, :] with hidden_state.
    // Row stride = hidden_size half-elements. Apply Gemma-style
    // softcap when configured: TopLevelEmitter skips the generic
    // logit-cap pass for cluster packages because that pass would
    // turn the -inf sentinels into a finite -cap, so the cap belongs
    // here where it fires only on real scores.
    device const half* row = lm_head_weight + uint(real_token) * hidden_size;
    float acc = 0.0f;
    for (uint h = 0; h < hidden_size; h++) {
        acc += float(row[h]) * float(hidden_state[h]);
    }
    if (logit_cap > 0.0f) {
        acc = logit_cap * tanh(acc / logit_cap);
    }
    logits[uint(real_token)] = half(acc);
}
