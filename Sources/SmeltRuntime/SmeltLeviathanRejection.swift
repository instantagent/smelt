// SmeltLeviathanRejection — CPU-side acceptance/rejection for
// speculative decoding.
//
// Two modes; pick by the drafter's sampling policy:
//
// - `step` (Leviathan et al. 2022 §3.1, arXiv:2211.17192):
//     for drafters that *sample* from their distribution `q`.
//     Accepts the candidate `c` with prob min(1, p(c)/q(c));
//     on reject samples from `(p - q)+` renormalised. Preserves
//     the target distribution exactly under stochastic drafting.
//
// - `greedyStep`: for drafters that pick `c = argmax(q)` (the
//     current SmeltGemma4Drafter). Accept if `c == argmax(p)`,
//     else reject and replace with `argmax(p)`. The Leviathan
//     ratio is incorrect for argmax drafts (q(c) measured by
//     softmax under-counts an argmax pick), so v1's greedy
//     drafter pairs with the greedy verifier here.
//
// `bonus` samples / picks the next-position token after a full
// K-accept. Use the matching mode for the drafter's policy.
//
// Dense entry points take fp16 logit rows (matches
// SmeltDrafterQ.dense and SmeltRuntime.allLogitsHalf); sparse
// entry points take `SmeltSparseLogitRow` (matches
// SmeltDrafterQ.sparse). Softmax is computed in fp32 with
// subtract-max for numerical stability; sentinels (-inf) drop to
// exp(0)=0 cleanly.

import Accelerate
import Foundation

public enum SmeltLeviathanRejection {

    public struct Decision: Sendable, Equatable {
        public let accepted: Bool
        /// On accept: the candidate. On reject: a token sampled
        /// from `(p - q)+` renormalised.
        public let token: Int32
    }

    /// Block-level verification outcome (Sun et al., ICLR 2025).
    /// `acceptedCount` is the length of the accepted prefix τ* ∈ [0, K];
    /// `token` is the bonus when τ* = K, or the residual replacement
    /// at position τ*+1 otherwise. Committed sequence length is
    /// `acceptedCount + 1`, matching the token-level `step` contract.
    public struct BlockDecision: Sendable, Equatable {
        public let acceptedCount: Int
        public let token: Int32
    }

    /// One position's accept-or-resample decision. `targetLogits`
    /// and `drafterLogits` must agree on vocab size. `temperature`
    /// is applied to both distributions before softmax — caller's
    /// drafter must have sampled from the same temperature, or
    /// the Leviathan ratio `p(c)/q(c)` no longer matches the actual
    /// proposal distribution and the rejection-sampling correctness
    /// proof breaks.
    ///
    /// `step` performs two sequential `next()` draws on `rng` (the
    /// accept uniform, then a residual categorical on reject), so a
    /// single `SmeltDeterministicRng(masterSeed:domain:
    /// .acceptUniform, position:)` covers both safely. Use a
    /// *separate* RNG for `bonus()` and for drafter-side sampling.
    public static func step<R: RandomNumberGenerator>(
        targetLogits: [Float16],
        drafterLogits: [Float16],
        candidate: Int32,
        temperature: Float = 1.0,
        using rng: inout R
    ) -> Decision {
        precondition(targetLogits.count == drafterLogits.count,
                     "SmeltLeviathanRejection: vocab size mismatch")
        let cIdx = Int(candidate)
        precondition(cIdx >= 0 && cIdx < targetLogits.count,
                     "SmeltLeviathanRejection: candidate out of vocab range")
        preconditionTemperature(temperature)

        let p = softmax(targetLogits, temperature: temperature)
        let q = softmax(drafterLogits, temperature: temperature)

        let pC = p[cIdx]
        let qC = q[cIdx]
        // q(c) == 0 implies the drafter never proposes this token;
        // since this argmax pick came from q, q(c) > 0 in any
        // realistic input. Guard anyway: ratio is then 1 (always
        // accept), which matches the bring-up doc's degenerate-case
        // expectation.
        let acceptanceProb: Float = qC > 0 ? min(1, pC / qC) : 1

        let r = Float.random(in: 0 ..< 1, using: &rng)
        if r < acceptanceProb {
            return Decision(accepted: true, token: candidate)
        }

        return Decision(
            accepted: false,
            token: sampleResidualToken(
                scalar: 1, p: p, q: q, using: &rng
            )
        )
    }

    /// Sample a bonus token from the target's distribution at the
    /// position after a full K-token accept. `temperature` matches
    /// the policy that produced the accepted prefix.
    public static func bonus<R: RandomNumberGenerator>(
        targetLogits: [Float16],
        temperature: Float = 1.0,
        using rng: inout R
    ) -> Int32 {
        preconditionTemperature(temperature)
        let p = softmax(targetLogits, temperature: temperature)
        return sampleCategorical(p, using: &rng)
    }

    /// Greedy variant: accept iff `candidate == argmax(p)`. On
    /// reject, replace with `argmax(p)`. Use this when the drafter
    /// produces argmax candidates (no stochastic sampling); the
    /// Leviathan ratio is incorrect for argmax drafts because
    /// softmax(q)[c] under-counts an argmax pick from q.
    public static func greedyStep(
        targetLogits: [Float16], candidate: Int32
    ) -> Decision {
        let pArgmax = argmaxIndex(targetLogits)
        if pArgmax == candidate {
            return Decision(accepted: true, token: candidate)
        }
        return Decision(accepted: false, token: pArgmax)
    }

    /// Greedy bonus: argmax(p). Pairs with greedyStep for the
    /// full-accept path.
    public static func greedyBonus(targetLogits: [Float16]) -> Int32 {
        argmaxIndex(targetLogits)
    }

    /// Greedy variant when the target argmax is already known (e.g.
    /// emitted by a GPU-side argmax kernel during verify). Avoids
    /// the full-vocab CPU argmax that `greedyStep` would otherwise
    /// run.
    public static func greedyStepFromArgmax(
        _ targetArgmax: Int32, candidate: Int32
    ) -> Decision {
        if targetArgmax == candidate {
            return Decision(accepted: true, token: candidate)
        }
        return Decision(accepted: false, token: targetArgmax)
    }

    /// Argmax over a Float16 vector. Pure-Swift iteration is ~20 ms
    /// for vocab=262 K on Apple silicon and dominates the spec-decode
    /// CPU loop (K+1 calls/round). The Accelerate two-pass —
    /// vImage fp16→fp32 conversion (SIMD) + `vDSP_maxvi` (SIMD argmax)
    /// — drops that to roughly 1 ms per call.
    ///
    /// `vDSP_maxvi`'s tie-breaking ("first or last index of max") is
    /// not documented to match the Swift loop's first-occurrence
    /// behavior, but exact-tie maxima on a softmax-style fp16 logit
    /// vector are vanishingly rare in practice, and the drafter's
    /// accept decision only cares about whether the index matches the
    /// candidate token — not which of multiple equal-logit indices
    /// gets returned.
    ///
    /// `NaN`-in-logits diverges from the pre-Accelerate loop: the
    /// old Swift form treated `NaN > bestVal` as false and skipped,
    /// landing on the first non-NaN max (or index 0 if all-NaN).
    /// `vDSP_maxvi` on NaN is undefined per Apple's docs. Callers
    /// must produce NaN-free logits — Smelt's target/drafter kernels
    /// do under normal operation. A NaN regression should be fixed
    /// at the offending kernel, not papered over by re-adding a
    /// NaN-skip CPU pass on this hot path.
    private static func argmaxIndex(_ logits: [Float16]) -> Int32 {
        let n = logits.count
        guard n > 0 else { return 0 }
        var maxIdx: vDSP_Length = 0
        // withUnsafeTemporaryAllocation stack-allocates the fp32
        // scratch (1 MB at vocab=262 K) when the runtime allows it,
        // otherwise falls back to a heap allocation. Either way we
        // avoid the `[Float](repeating: 0, count: n)` ARC array that
        // the original draft of this kernel paid for once per call.
        withUnsafeTemporaryAllocation(
            of: Float.self, capacity: n
        ) { (scratch: UnsafeMutableBufferPointer<Float>) in
            logits.withUnsafeBytes { src in
                var srcBuf = vImage_Buffer(
                    data: UnsafeMutableRawPointer(
                        mutating: src.baseAddress!
                    ),
                    height: 1,
                    width: vImagePixelCount(n),
                    rowBytes: n * MemoryLayout<Float16>.stride
                )
                var dstBuf = vImage_Buffer(
                    data: UnsafeMutableRawPointer(scratch.baseAddress!),
                    height: 1,
                    width: vImagePixelCount(n),
                    rowBytes: n * MemoryLayout<Float>.stride
                )
                _ = vImageConvert_Planar16FtoPlanarF(
                    &srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags)
                )
            }
            var maxVal: Float = 0
            vDSP_maxvi(
                scratch.baseAddress!, 1, &maxVal, &maxIdx, vDSP_Length(n)
            )
        }
        return Int32(maxIdx)
    }

    /// Block Verification (Sun et al., ICLR 2025, arxiv 2403.10444,
    /// Algorithm 2). Verifies all K drafter candidates jointly under
    /// a coupling that strictly dominates per-token rejection on
    /// E[accepted length]:
    ///     min(prod p_i, prod q_i) >= prod min(p_i, q_i)
    /// Same target marginal as `step` per Theorem 1 — committed
    /// sequence is distributed exactly as the target.
    ///
    /// Inputs:
    ///   - `targetLogits`: K+1 fp16 rows; positions 1..K plus the
    ///     bonus position K+1.
    ///   - `drafterLogits`: K fp16 rows; positions 1..K.
    ///   - `candidates`: K drafter samples X_1..X_K.
    ///   - `temperature`: applied symmetrically to p and q.
    ///   - `acceptRng`: K sequential `next()` draws (η_1..η_K).
    ///   - `residualOrBonusRng`: one categorical draw (residual on
    ///     partial accept, bonus on full accept). Caller must use
    ///     domain-separated RNGs across `acceptRng` and
    ///     `residualOrBonusRng` so cumulative-ratio decisions don't
    ///     correlate with the replacement draw.
    public static func blockStep<R: RandomNumberGenerator>(
        targetLogits: [[Float16]],
        drafterLogits: [[Float16]],
        candidates: [Int32],
        temperature: Float = 1.0,
        acceptRng: inout R,
        residualOrBonusRng: inout R
    ) -> BlockDecision {
        let K = candidates.count
        precondition(K >= 1, "SmeltLeviathanRejection.blockStep: K >= 1")
        precondition(targetLogits.count == K + 1,
                     "blockStep: targetLogits must have K+1 rows (verify positions + bonus)")
        precondition(drafterLogits.count == K,
                     "blockStep: drafterLogits must have K rows")
        preconditionTemperature(temperature)
        let vocab = targetLogits[0].count
        for row in targetLogits {
            precondition(row.count == vocab, "blockStep: target row width mismatch")
        }
        for row in drafterLogits {
            precondition(row.count == vocab, "blockStep: drafter row width mismatch")
        }

        let pDist = targetLogits.map { softmax($0, temperature: temperature) }
        let qDist = drafterLogits.map { softmax($0, temperature: temperature) }

        var pPrev: Float = 1.0
        var pAtBoundary: Float = 1.0
        var tauStar = 0

        for i in 0 ..< K {
            let cIdx = Int(candidates[i])
            precondition(cIdx >= 0 && cIdx < vocab,
                         "blockStep: candidate \(i)=\(candidates[i]) out of vocab range")
            let qC = qDist[i][cIdx]
            let pC = pDist[i][cIdx]
            let ratioStep: Float = qC > 0 ? pC / qC : 1
            let pNext = min(pPrev * ratioStep, 1)

            let h: Float
            if i < K - 1 {
                // S_i must use the NEXT row's distributions (the
                // position the residual would sample from if this
                // step rejects), not the current candidate's row.
                // Using `i` instead of `i+1` here biases the
                // accepted-prefix probabilities at K>1 and breaks
                // the claimed target-marginal exactness — codex
                // P1 finding on the chunk-close review.
                var s: Float = 0
                for x in 0 ..< vocab {
                    let v = pNext * pDist[i + 1][x] - qDist[i + 1][x]
                    if v > 0 { s += v }
                }
                let denom = s + (1 - pNext)
                // denom == 0 ⇒ s == 0 AND pNext == 1, which forces
                // dist_p_{i+1} == dist_q_{i+1} exactly. The next
                // position is trivially covered. Falling back to
                // h=0 here would drop already-matched prefixes
                // whenever a later step rejects.
                h = denom > 0 ? s / denom : 1
            } else {
                h = pNext
            }
            let eta = Float.random(in: 0 ..< 1, using: &acceptRng)
            if eta <= h {
                tauStar = i + 1
                pAtBoundary = pNext
            }
            pPrev = pNext
        }

        if tauStar == K {
            return BlockDecision(
                acceptedCount: K,
                token: sampleCategorical(pDist[K], using: &residualOrBonusRng)
            )
        }

        // Partial accept (including τ* = 0): residual at the
        // rejecting position τ*+1 uses the PRE-rejecting cumulative
        // scalar (paper p_{τ*}, tracked here as `pAtBoundary`,
        // updated only when a step accepts). At τ*=0 it stays at
        // the initial 1.0, so K=1 rejection collapses to the
        // token-level (p - q)+ formula — preserving the equivalence
        // to Leviathan rejection. Using the post-rejecting scalar
        // here would shrink the residual mass and let the rejected
        // candidate slip back in via the sum=0 fallback.
        return BlockDecision(
            acceptedCount: tauStar,
            token: sampleResidualToken(
                scalar: pAtBoundary,
                p: pDist[tauStar],
                q: qDist[tauStar],
                using: &residualOrBonusRng
            )
        )
    }

    /// Hierarchical Speculative Decoding (Zhou et al., arxiv
    /// 2601.05724v2, Algorithm 2 with capped branch resampling).
    /// Same I/O shape as `blockStep` — drop-in replacement, NOT a
    /// composition. Backward scan from t=K down to 1, breaking on
    /// the first position whose `h_t^HSD` clears its independent
    /// uniform draw. Uses capped branch divergences instead of BV's
    /// joint OT coupling; published delta over BV is ~+2–3%
    /// absolute α on EAGLE drafters, no proven dominance.
    ///
    /// Acceptance threshold (Eq. 11):
    ///   h_K   = min(r*(X_{1:K}), 1)
    ///   h_t<K = D*_pq / max(D*_pq, D*_qp)
    /// where r*(X_{1:t}) caps the cumulative ratio at the prefix's
    /// historical max (Eq. 16) and D*_pq / D*_qp are the capped
    /// branch divergences over the vocab at position t (Eq. 17).
    ///
    /// Residual on partial accept (Eq. 20):
    ///   p_res(x | X_{1:τ*}) ∝ max(scalar_{τ*+1} · p_{τ*+1}(x)
    ///                              − q_{τ*+1}(x), 0)
    /// where `scalar_{τ*+1} = exp(logR[τ*] − logR[m[τ*+1]])` when
    /// `m[τ*+1] > 0`, otherwise `exp(logR[τ*])`.
    /// Bonus on full accept is identical to BV: sample from
    /// `p_{K+1}` directly.
    public static func hsdStep<R: RandomNumberGenerator>(
        targetLogits: [[Float16]],
        drafterLogits: [[Float16]],
        candidates: [Int32],
        temperature: Float = 1.0,
        acceptRng: inout R,
        residualOrBonusRng: inout R
    ) -> BlockDecision {
        let K = candidates.count
        precondition(K >= 1, "SmeltLeviathanRejection.hsdStep: K >= 1")
        precondition(targetLogits.count == K + 1,
                     "hsdStep: targetLogits must have K+1 rows")
        precondition(drafterLogits.count == K,
                     "hsdStep: drafterLogits must have K rows")
        preconditionTemperature(temperature)
        let vocab = targetLogits[0].count
        for row in targetLogits {
            precondition(row.count == vocab, "hsdStep: target row width mismatch")
        }
        for row in drafterLogits {
            precondition(row.count == vocab, "hsdStep: drafter row width mismatch")
        }

        let pDist = targetLogits.map { softmax($0, temperature: temperature) }
        let qDist = drafterLogits.map { softmax($0, temperature: temperature) }

        // logR[t] = Σ_{i=1..t} log(p_i(X_i) / q_i(X_i)). logR[0] = 0.
        // Use a tiny floor so log of structurally-zero probs collapses
        // to -∞ predictably rather than -nan.
        let floor = Float.leastNormalMagnitude
        var logR = [Float](repeating: 0, count: K + 1)
        for t in 0 ..< K {
            let cIdx = Int(candidates[t])
            precondition(cIdx >= 0 && cIdx < vocab,
                         "hsdStep: candidate \(t)=\(candidates[t]) out of vocab range")
            let pC = max(pDist[t][cIdx], floor)
            let qC = max(qDist[t][cIdx], floor)
            logR[t + 1] = logR[t] + log(pC) - log(qC)
        }

        // m[t] = argmax_{1≤i<t} r(X_{1:i}), or 0 if all r ≤ 1.
        // Equivalently in log-space: argmax_{1≤i<t} logR[i] if any
        // positive, else 0. logR[0] = 0 is the "no cap" sentinel.
        var mIdx = [Int](repeating: 0, count: K + 1)
        var bestI: Int = 0
        var bestLogR: Float = 0
        for t in 1 ... K {
            mIdx[t] = bestI
            if t < K, logR[t] > bestLogR {
                bestLogR = logR[t]
                bestI = t
            }
        }

        // For varying x_t at position t, r*(X_{1:t-1}, x_t) =
        // exp(logR[t-1] − logR[m[t]]) · p_t(x_t)/q_t(x_t) (or
        // exp(logR[t-1]) when m[t] = 0).
        var scalarAtT = [Float](repeating: 1, count: K + 1)
        for t in 1 ... K {
            let offset = mIdx[t] > 0 ? logR[mIdx[t]] : 0
            scalarAtT[t] = exp(logR[t - 1] - offset)
        }

        // Per-position branch divergences (only for paper t < K).
        // h_t uses D*_Branch(p,q | X_{1:t}) — the divergence at the
        // NEXT position (t+1), summing over x_{t+1} with cumulative
        // cap scalarAtT[t+1]. Using the current-position row
        // collapses to the trivial p/q TV identity at t=1, forcing
        // h_1 = 1 and committing q's sample without target-marginal
        // correction (P1 chunk-close finding).
        var dPQ = [Float](repeating: 0, count: K - 1)
        var dQP = [Float](repeating: 0, count: K - 1)
        for t in 1 ..< K {
            let s = scalarAtT[t + 1]
            var sumPQ: Float = 0
            var sumQP: Float = 0
            let p = pDist[t]
            let q = qDist[t]
            for x in 0 ..< vocab {
                let scaledP = s * p[x]
                let diff = scaledP - q[x]
                if diff > 0 {
                    sumPQ += diff
                } else {
                    sumQP -= diff
                }
            }
            dPQ[t - 1] = sumPQ
            dQP[t - 1] = sumQP
        }

        // Backward scan: t from K down to 1. First t whose h_t ≥ η_t
        // wins. h_K = min(r*(X_{1:K}), 1); for t < K, h_t = dPQ /
        // max(dPQ, dQP), with the dPQ/dQP guard for degenerate rows.
        var tauStar = 0
        let cK = Int(candidates[K - 1])
        let pCK = max(pDist[K - 1][cK], floor)
        let qCK = max(qDist[K - 1][cK], floor)
        let rStarK = scalarAtT[K] * pCK / qCK
        let hK = min(rStarK, 1)
        let etaK = Float.random(in: 0 ..< 1, using: &acceptRng)
        if hK >= etaK {
            tauStar = K
        }
        var sIdx = K - 1
        while sIdx >= 1, tauStar == 0 {
            let pq = dPQ[sIdx - 1]
            let qp = dQP[sIdx - 1]
            let denom = max(pq, qp)
            // denom == 0 ⇒ both divergences are zero ⇒ p and q match
            // (after the cap) over the entire vocab at this position.
            // Trivially commit — falling back to 0 would drop matched
            // prefixes the same way the BV bug did.
            let h: Float = denom > 0 ? pq / denom : 1
            let eta = Float.random(in: 0 ..< 1, using: &acceptRng)
            if h >= eta {
                tauStar = sIdx
                break
            }
            sIdx -= 1
        }

        if tauStar == K {
            return BlockDecision(
                acceptedCount: K,
                token: sampleCategorical(pDist[K], using: &residualOrBonusRng)
            )
        }

        // Partial accept (τ* < K). Residual at position τ*+1 uses
        // scalarAtT[τ*+1] · p_{τ*+1}(x) − q_{τ*+1}(x) clipped at 0,
        // normalized by D*_pq at that position (Eq. 20).
        return BlockDecision(
            acceptedCount: tauStar,
            token: sampleResidualToken(
                scalar: scalarAtT[tauStar + 1],
                p: pDist[tauStar],
                q: qDist[tauStar],
                using: &residualOrBonusRng
            )
        )
    }

    /// Sparse expansion identity (Sq = support of q):
    ///     Σ_x max(s·p(x) − q(x), 0)
    ///   = s·(1 − Σ_{x∈Sq} p(x)) + Σ_{x∈Sq} max(s·p(x) − q(x), 0)+
    /// Tokens outside Sq have q=0, so max collapses to s·p which sums
    /// to s·(1 − Σ_Sq p). Drops the O(vocab) sweep to O(|Sq|).
    public static func blockStepSparse<R: RandomNumberGenerator>(
        targetLogits: [[Float16]],
        drafterSparseLogits: [SmeltSparseLogitRow],
        candidates: [Int32],
        temperature: Float = 1.0,
        acceptRng: inout R,
        residualOrBonusRng: inout R
    ) -> BlockDecision {
        let K = candidates.count
        precondition(K >= 1, "blockStepSparse: K >= 1")
        precondition(
            targetLogits.count == K + 1,
            "blockStepSparse: targetLogits must have K+1 rows"
        )
        precondition(
            drafterSparseLogits.count == K,
            "blockStepSparse: drafterSparseLogits must have K rows"
        )
        preconditionTemperature(temperature)
        let vocab = targetLogits[0].count

        let pDist = targetLogits.map { softmax($0, temperature: temperature) }
        let qDist: [[Int32: Float]] = drafterSparseLogits.map {
            sparseSoftmax($0, temperature: temperature, vocab: vocab)
        }

        var pPrev: Float = 1.0
        var pAtBoundary: Float = 1.0
        var tauStar = 0

        for i in 0 ..< K {
            let cIdx = Int(candidates[i])
            precondition(
                cIdx >= 0 && cIdx < vocab,
                "blockStepSparse: candidate \(i)=\(candidates[i]) out of range"
            )
            let qC = qDist[i][candidates[i]] ?? 0
            let pC = pDist[i][cIdx]
            let ratioStep: Float = qC > 0 ? pC / qC : 1
            let pNext = min(pPrev * ratioStep, 1)

            let h: Float
            if i < K - 1 {
                let qNext = qDist[i + 1]
                let pNextRow = pDist[i + 1]
                var sumPInQ: Float = 0
                var positiveDiffs: Float = 0
                // Sort by token id for fp-deterministic accumulation;
                // Swift Dictionary iteration is hash-randomized across
                // processes and the sum order would otherwise flip
                // h's accept/reject decision at boundary cases.
                for tok in qNext.keys.sorted() {
                    let qProb = qNext[tok]!
                    let pVal = pNextRow[Int(tok)]
                    sumPInQ += pVal
                    let term = pNext * pVal - qProb
                    if term > 0 { positiveDiffs += term }
                }
                // `1 − sumPInQ` can dip negative on fp drift when q's
                // support covers high-mass p tokens; clamp the out-of-Sq
                // mass term at 0 so `s` stays non-negative.
                let s = max(pNext * (1 - sumPInQ), 0) + positiveDiffs
                let denom = s + (1 - pNext)
                h = denom > 0 ? s / denom : 1
            } else {
                h = pNext
            }
            let eta = Float.random(in: 0 ..< 1, using: &acceptRng)
            if eta <= h {
                tauStar = i + 1
                pAtBoundary = pNext
            }
            pPrev = pNext
        }

        if tauStar == K {
            return BlockDecision(
                acceptedCount: K,
                token: sampleCategorical(
                    pDist[K], using: &residualOrBonusRng
                )
            )
        }
        return BlockDecision(
            acceptedCount: tauStar,
            token: sampleResidualTokenSparse(
                scalar: pAtBoundary,
                p: pDist[tauStar],
                q: qDist[tauStar],
                using: &residualOrBonusRng
            )
        )
    }

    /// Sparse-q `hsdStep`; see `blockStepSparse` for the support-sum
    /// identity used by the dPQ/dQP divergences.
    public static func hsdStepSparse<R: RandomNumberGenerator>(
        targetLogits: [[Float16]],
        drafterSparseLogits: [SmeltSparseLogitRow],
        candidates: [Int32],
        temperature: Float = 1.0,
        acceptRng: inout R,
        residualOrBonusRng: inout R
    ) -> BlockDecision {
        let K = candidates.count
        precondition(K >= 1, "hsdStepSparse: K >= 1")
        precondition(
            targetLogits.count == K + 1,
            "hsdStepSparse: targetLogits must have K+1 rows"
        )
        precondition(
            drafterSparseLogits.count == K,
            "hsdStepSparse: drafterSparseLogits must have K rows"
        )
        preconditionTemperature(temperature)
        let vocab = targetLogits[0].count

        let pDist = targetLogits.map { softmax($0, temperature: temperature) }
        let qDist: [[Int32: Float]] = drafterSparseLogits.map {
            sparseSoftmax($0, temperature: temperature, vocab: vocab)
        }
        let floor = Float.leastNormalMagnitude

        var logR = [Float](repeating: 0, count: K + 1)
        for t in 0 ..< K {
            let cIdx = Int(candidates[t])
            precondition(
                cIdx >= 0 && cIdx < vocab,
                "hsdStepSparse: candidate \(t) out of range"
            )
            let pC = max(pDist[t][cIdx], floor)
            let qC = max(qDist[t][candidates[t]] ?? 0, floor)
            logR[t + 1] = logR[t] + log(pC) - log(qC)
        }

        var mIdx = [Int](repeating: 0, count: K + 1)
        var bestI: Int = 0
        var bestLogR: Float = 0
        for t in 1 ... K {
            mIdx[t] = bestI
            if t < K, logR[t] > bestLogR {
                bestLogR = logR[t]
                bestI = t
            }
        }

        var scalarAtT = [Float](repeating: 1, count: K + 1)
        for t in 1 ... K {
            let offset = mIdx[t] > 0 ? logR[mIdx[t]] : 0
            scalarAtT[t] = exp(logR[t - 1] - offset)
        }

        var dPQ = [Float](repeating: 0, count: K - 1)
        var dQP = [Float](repeating: 0, count: K - 1)
        for t in 1 ..< K {
            let s = scalarAtT[t + 1]
            let qT = qDist[t]
            let pT = pDist[t]
            // For x not in Sq: s·p[x] > 0 contributes only to dPQ.
            // For x in Sq: split into (s·p − q)+ → dPQ and (q − s·p)+ → dQP.
            var sumPInQ: Float = 0
            var sumPQ: Float = 0
            var sumQP: Float = 0
            // Sort by token id for fp-deterministic accumulation
            // across processes (Dictionary iteration is randomized).
            for tok in qT.keys.sorted() {
                let qProb = qT[tok]!
                let pVal = pT[Int(tok)]
                sumPInQ += pVal
                let scaledP = s * pVal
                let diff = scaledP - qProb
                if diff > 0 {
                    sumPQ += diff
                } else {
                    sumQP -= diff
                }
            }
            // Clamp out-of-Sq mass at 0 against fp drift in sumPInQ.
            sumPQ += max(s * (1 - sumPInQ), 0)
            dPQ[t - 1] = sumPQ
            dQP[t - 1] = sumQP
        }

        var tauStar = 0
        let cK = Int(candidates[K - 1])
        let pCK = max(pDist[K - 1][cK], floor)
        let qCK = max(qDist[K - 1][candidates[K - 1]] ?? 0, floor)
        let rStarK = scalarAtT[K] * pCK / qCK
        let hK = min(rStarK, 1)
        let etaK = Float.random(in: 0 ..< 1, using: &acceptRng)
        if hK >= etaK {
            tauStar = K
        }
        var sIdx = K - 1
        while sIdx >= 1, tauStar == 0 {
            let pq = dPQ[sIdx - 1]
            let qp = dQP[sIdx - 1]
            let denom = max(pq, qp)
            let h: Float = denom > 0 ? pq / denom : 1
            let eta = Float.random(in: 0 ..< 1, using: &acceptRng)
            if h >= eta {
                tauStar = sIdx
                break
            }
            sIdx -= 1
        }

        if tauStar == K {
            return BlockDecision(
                acceptedCount: K,
                token: sampleCategorical(
                    pDist[K], using: &residualOrBonusRng
                )
            )
        }
        return BlockDecision(
            acceptedCount: tauStar,
            token: sampleResidualTokenSparse(
                scalar: scalarAtT[tauStar + 1],
                p: pDist[tauStar],
                q: qDist[tauStar],
                using: &residualOrBonusRng
            )
        )
    }

    private static func sparseSoftmax(
        _ row: SmeltSparseLogitRow, temperature: Float, vocab: Int
    ) -> [Int32: Float] {
        let entries = row.entries
        if entries.isEmpty { return [:] }
        let invT = 1 / temperature
        var maxScaled = -Float.infinity
        for (_, logit) in entries {
            let scaled = logit * invT
            if scaled > maxScaled { maxScaled = scaled }
        }
        var sum: Float = 0
        var result: [Int32: Float] = [:]
        result.reserveCapacity(entries.count)
        for (tok, logit) in entries {
            precondition(
                tok >= 0 && Int(tok) < vocab,
                "sparseSoftmax: token \(tok) out of vocab range [0, \(vocab))"
            )
            let e = exp(logit * invT - maxScaled)
            result[tok] = e
            sum += e
        }
        if sum > 0 {
            for (tok, _) in entries {
                result[tok] = result[tok]! / sum
            }
        }
        return result
    }

    /// Bounded loop count for residual rejection sampling before
    /// falling back to inverse-CDF over the out-of-Sq partition.
    private static let sparseResidualRejectionCap = 32

    /// Residual mass for tokens NOT in Sq is `scalar·p(x)` (q(x)=0),
    /// routed via rejection sampling from p and rejecting draws in Sq.
    /// Tokens in Sq with `scalar·p − q > 0` form a small in-set bucket
    /// sampled directly.
    private static func sampleResidualTokenSparse<R: RandomNumberGenerator>(
        scalar: Float, p: [Float], q: [Int32: Float], using rng: inout R
    ) -> Int32 {
        var sumPInQ: Float = 0
        var positiveDiffs: Float = 0
        var posBuckets: [(Int32, Float)] = []
        posBuckets.reserveCapacity(q.count)
        // Sorted iteration: Swift Dictionary order is hash-randomized
        // across processes; without sorting, seeded RNG would map the
        // same u2 to different replacement tokens across runs.
        for tok in q.keys.sorted() {
            let qProb = q[tok]!
            let pVal = p[Int(tok)]
            sumPInQ += pVal
            let diff = scalar * pVal - qProb
            if diff > 0 {
                positiveDiffs += diff
                posBuckets.append((tok, diff))
            }
        }
        let outSetMass = max(scalar - scalar * sumPInQ, 0)
        let totalMass = outSetMass + positiveDiffs
        if totalMass <= 0 {
            return sampleCategorical(p, using: &rng)
        }
        let u = Float.random(in: 0 ..< totalMass, using: &rng)
        if u < outSetMass {
            // Common case: rejection-sample from p restricted to x ∉ Sq.
            // Pr(reject) = Σ_Sq p; usually small but suffix histograms
            // can push it high enough to exceed 32 iters meaningfully.
            for _ in 0 ..< Self.sparseResidualRejectionCap {
                let candidate = sampleCategorical(p, using: &rng)
                if q[candidate] == nil { return candidate }
            }
            // Fallback: inverse CDF over x ∉ Sq (preserves the
            // residual's "support ⊆ {x : q(x)=0}" invariant). O(vocab)
            // but only fires when rejection has consistently missed.
            let outSetTotal = max(1 - sumPInQ, Float.leastNormalMagnitude)
            var u2 = Float.random(in: 0 ..< outSetTotal, using: &rng)
            for x in 0 ..< p.count {
                if q[Int32(x)] != nil { continue }
                u2 -= p[x]
                if u2 < 0 { return Int32(x) }
            }
            return Int32(p.count - 1)
        }
        var u2 = u - outSetMass
        for (tok, mass) in posBuckets {
            if u2 < mass { return tok }
            u2 -= mass
        }
        return posBuckets.last!.0
    }

    /// Sample one token from the residual distribution shared
    /// across `step`, `blockStep`, and `hsdStep`:
    ///     r(x) ∝ max(scalar · p(x) − q(x), 0)
    /// renormalised. When the residual sums to zero (q dominates
    /// scalar·p everywhere) fall back to sampling from `p` — keeps
    /// the per-call output in the target's support rather than
    /// NaN-dividing or trapping.
    private static func sampleResidualToken<R: RandomNumberGenerator>(
        scalar: Float, p: [Float], q: [Float], using rng: inout R
    ) -> Int32 {
        let n = p.count
        var residual = [Float](repeating: 0, count: n)
        var sum: Float = 0
        for x in 0 ..< n {
            let v = scalar * p[x] - q[x]
            if v > 0 {
                residual[x] = v
                sum += v
            }
        }
        let dist: [Float] = sum > 0
            ? residual.map { $0 / sum }
            : p
        return sampleCategorical(dist, using: &rng)
    }

    private static func preconditionTemperature(
        _ t: Float, function: StaticString = #function
    ) {
        precondition(
            t.isFinite && t > 0 && (1 / t).isFinite,
            "\(function): temperature must be finite, positive, and not subnormal"
        )
    }

    /// Subtract-max fp32 softmax. -inf sentinels (cluster-sparse
    /// drafter logits) collapse to 0 contribution via `exp(-inf) = 0`;
    /// if every logit is -inf the function returns a uniform
    /// distribution rather than NaN-dividing.
    private static func softmax(
        _ logits: [Float16], temperature: Float
    ) -> [Float] {
        let n = logits.count
        guard n > 0 else { return [] }
        var result = [Float](repeating: 0, count: n)
        result.withUnsafeMutableBufferPointer { dst in
            logits.withUnsafeBytes { src in
                var srcBuf = vImage_Buffer(
                    data: UnsafeMutableRawPointer(
                        mutating: src.baseAddress!
                    ),
                    height: 1,
                    width: vImagePixelCount(n),
                    rowBytes: n * MemoryLayout<Float16>.stride
                )
                var dstBuf = vImage_Buffer(
                    data: UnsafeMutableRawPointer(dst.baseAddress!),
                    height: 1,
                    width: vImagePixelCount(n),
                    rowBytes: n * MemoryLayout<Float>.stride
                )
                _ = vImageConvert_Planar16FtoPlanarF(
                    &srcBuf, &dstBuf, vImage_Flags(kvImageNoFlags)
                )
            }
            var maxRaw: Float = 0
            vDSP_maxv(dst.baseAddress!, 1, &maxRaw, vDSP_Length(n))
            guard maxRaw > -.infinity else {
                // All logits are -inf (e.g. an empty cluster-sparse
                // row); fall back to uniform rather than divide by 0.
                let u = Float(1) / Float(n)
                var uniform = u
                vDSP_vfill(&uniform, dst.baseAddress!, 1, vDSP_Length(n))
                return
            }
            let invTemp = 1 / temperature
            var bias = -invTemp * maxRaw
            var scale = invTemp
            vDSP_vsmsa(
                dst.baseAddress!, 1, &scale, &bias,
                dst.baseAddress!, 1, vDSP_Length(n)
            )
            var nInt = Int32(n)
            vvexpf(dst.baseAddress!, dst.baseAddress!, &nInt)
            var sum: Float = 0
            vDSP_sve(dst.baseAddress!, 1, &sum, vDSP_Length(n))
            guard sum > 0 else {
                // Numerical underflow with extreme negative inputs;
                // fall back to uniform.
                let u = Float(1) / Float(n)
                var uniform = u
                vDSP_vfill(&uniform, dst.baseAddress!, 1, vDSP_Length(n))
                return
            }
            vDSP_vsdiv(
                dst.baseAddress!, 1, &sum,
                dst.baseAddress!, 1, vDSP_Length(n)
            )
        }
        return result
    }

    /// Inverse-CDF categorical sample. `p` must be non-negative and
    /// sum to ~1; numeric drift makes the fallback at the tail
    /// load-bearing (final cum can land just under r).
    private static func sampleCategorical<R: RandomNumberGenerator>(
        _ p: [Float], using rng: inout R
    ) -> Int32 {
        let r = Float.random(in: 0 ..< 1, using: &rng)
        var cum: Float = 0
        for i in 0 ..< p.count {
            cum += p[i]
            if r < cum { return Int32(i) }
        }
        return Int32(p.count - 1)
    }
}
