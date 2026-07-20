// SmeltSpeculativeDecode — Orchestration surface for K-step
// speculative decoding (drafter + target verify + Leviathan
// rejection sampling).
//
// Algorithmic source: Leviathan et al. 2022 (arXiv:2211.17192)
// for the rejection sampler; EAGLE (Li et al. 2024,
// arXiv:2401.15077) for the drafter shape — the drafter
// cross-attends to the target's last-layer K/V instead of
// running its own vanilla causal-LM next-token loop.
//
// Per-decode contract:
//   1. Target's `decodeStep` produced the most-recent token plus
//      side effects: post-final-norm hidden state in `normOutBuf`
//      and last-layer K/V cache writes for sliding/global.
//   2. SmeltSpeculativeRuntime calls `drafter.draftStep` with the
//      target's `SmeltDrafterTaps`. The drafter generates K
//      candidate tokens.
//   3. Target chunked-prefills [last_token, candidate_0, …,
//      candidate_{K-1}] in one batched call, producing K+1
//      next-token argmaxes on the greedy fast path or K+1
//      next-token logit distributions on the full-logits path.
//   4. Leviathan rejection sampler accepts the longest prefix of
//      candidates the target agrees with, samples a replacement
//      at the rejection point (or a bonus after a full accept),
//      and commits 1..(K+1) tokens. The experimental drafted-bonus
//      path drafts one extra candidate and, on full accept, commits
//      that verified tail instead of sampling an unverified target
//      bonus.

import Foundation
import Metal
import SmeltSchema

/// Sparse-logit representation for drafters whose proposal q is
/// concentrated on a handful of tokens (suffix-lookup histograms,
/// future model-free drafters). Entries are `(token, logit)` pairs
/// where unlisted tokens carry an implicit `-inf` logit; the verify
/// path's sparse softmax converts to probabilities over just the
/// listed entries, skipping the full-vocab dense pass.
public struct SmeltSparseLogitRow: Sendable {
    public let entries: [(token: Int32, logit: Float)]

    public init(entries: [(token: Int32, logit: Float)]) {
        // Reject duplicate tokens: sparse softmax would double-count
        // a duplicate's mass in the normalizer but overwrite the key
        // with the last logit, biasing q. Drafters that need to
        // coalesce duplicates should pre-merge.
        var seen = Set<Int32>()
        seen.reserveCapacity(entries.count)
        for (tok, _) in entries {
            precondition(
                seen.insert(tok).inserted,
                "SmeltSparseLogitRow: duplicate token \(tok) in entries"
            )
        }
        self.entries = entries
    }

    /// Convenience init for empirical-histogram drafters: stores
    /// each token's log-probability `log(count / denominator)`.
    public init(histogram: [Int32: Int], denominator: Int) {
        let denom = Float(denominator)
        var built: [(token: Int32, logit: Float)] = []
        built.reserveCapacity(histogram.count)
        for (tok, count) in histogram {
            built.append((token: tok, logit: Foundation.log(Float(count) / denom)))
        }
        self.entries = built
    }
}

/// Per-step proposal distribution carried by a `SmeltDraftBatch`.
/// `.none` is argmax-only (verify uses targetArgmaxes); `.dense`
/// is the legacy full-vocab Float16 row per step; `.sparse` carries
/// only the support of q for verify's sparse BV/HSD path.
public enum SmeltDrafterQ: Sendable {
    case none
    case dense([[Float16]])
    case sparse([SmeltSparseLogitRow])

    public var rowCount: Int? {
        switch self {
        case .none: return nil
        case .dense(let rows): return rows.count
        case .sparse(let rows): return rows.count
        }
    }
}

/// Up to K candidate tokens with their per-step proposal distribution.
/// An empty batch is an explicit decline: the speculative runtime takes the
/// ordinary one-token target path. History-only conditional drafters should
/// expose that decision through `SmeltPreflightDrafter` so a miss happens
/// before target taps or speculative state preparation.
public struct SmeltDraftBatch: Sendable {
    public let candidates: [Int32]
    public let q: SmeltDrafterQ

    public init(candidates: [Int32], q: SmeltDrafterQ) throws {
        if let rowCount = q.rowCount {
            guard rowCount == candidates.count else {
                throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                    "SmeltDraftBatch: q must have K rows "
                    + "(got \(rowCount) vs \(candidates.count))"
                )
            }
        }
        self.candidates = candidates
        self.q = q
    }

    public var K: Int { candidates.count }
}

/// Architecture-agnostic drafter consumed by
/// `SmeltSpeculativeRuntime`. Concrete impls wrap a drafter
/// `.smeltpkg`'s `SmeltRuntime`; the surface fits EAGLE-{1,2,3}
/// and Medusa drafters with only the `draftStep` body changing.
public protocol SmeltDrafter {
    /// Step the drafter K times starting from the target's most recent
    /// committed state. A drafter normally produces exactly K candidates; a
    /// conditional drafter may return an empty batch to decline the round.
    ///
    /// `position` is the target's decode position at which
    /// `lastToken` lives (i.e., the position whose K/V the target
    /// just wrote into the bound cache). The drafter's RoPE
    /// rotates Q at this position and its attention reads K/V
    /// over `[0..position]`. Passing `0` for any draft after the
    /// first decode silently throws away most of the prefill +
    /// decode history; `SmeltSpeculativeRuntime` is responsible
    /// for threading the live position through.
    ///
    /// `selectionMode` controls per-step candidate selection.
    /// `.argmax` (or `.temperature` with non-positive/non-finite
    /// temp) requests the argmax fast path: drafter returns
    /// `q: .none`. `.temperature` with positive finite temp
    /// requests stochastic sampling: drafter MUST populate
    /// `batch.q` with `.dense` or `.sparse` so the verify-side
    /// Leviathan ratio matches the policy that produced each
    /// candidate.
    func draftStep(
        targetTaps: SmeltDrafterTaps,
        lastToken: Int32,
        position: Int32,
        K: Int,
        selectionMode: SmeltSelectionMode
    ) throws -> SmeltDraftBatch

    /// Declared on the protocol (not just the extension) so callers
    /// holding `SmeltDrafter` dispatch dynamically and reach the
    /// concrete impl; extension defaults are static-dispatched.
    func resetSuffixCache(promptTokens: [Int32])
    func recordGeneratedTokens(_ tokens: [Int32])
}

/// A drafter that can produce or decline a complete proposal from request
/// history alone. The speculative runtime invokes this surface before asking
/// the target for taps or entering any transactional verification contract.
/// An empty batch is therefore a true routing decision: the target takes its
/// ordinary one-token path without speculative state preparation.
public protocol SmeltPreflightDrafter: SmeltDrafter {
    func preflightDraft(
        lastToken: Int32,
        position: Int32,
        K: Int,
        selectionMode: SmeltSelectionMode
    ) throws -> SmeltDraftBatch
}

/// A drafter whose first candidate is seeded by the target's current row-zero
/// prediction. The rest of speculative verification stays identical; this
/// protocol only makes the otherwise implicit seed dependency explicit.
public protocol SmeltTargetSeededDrafter: SmeltDrafter {
    func draftStep(
        targetTaps: SmeltDrafterTaps,
        targetNextToken: Int32,
        lastToken: Int32,
        position: Int32,
        K: Int,
        selectionMode: SmeltSelectionMode
    ) throws -> SmeltDraftBatch

    func primeTargetContext(
        promptTokens: [Int32],
        targetNextToken: Int32,
        targetHiddenStates: [Data]
    ) throws

    func commitDraft(
        batch: SmeltDraftBatch,
        acceptedCount: Int,
        committedTokens: [Int32],
        startPosition: Int32
    ) throws
}

public extension SmeltTargetSeededDrafter {
    func primeTargetContext(
        promptTokens: [Int32],
        targetNextToken: Int32,
        targetHiddenStates: [Data]
    ) throws { }

    func commitDraft(
        batch: SmeltDraftBatch,
        acceptedCount: Int,
        committedTokens: [Int32],
        startPosition: Int32
    ) throws { }
}

public extension SmeltDrafter {
    func draftStep(
        targetTaps: SmeltDrafterTaps,
        lastToken: Int32,
        position: Int32,
        K: Int
    ) throws -> SmeltDraftBatch {
        try draftStep(
            targetTaps: targetTaps,
            lastToken: lastToken,
            position: position,
            K: K,
            selectionMode: .argmax
        )
    }

    func resetSuffixCache(promptTokens: [Int32]) { }
    func recordGeneratedTokens(_ tokens: [Int32]) { }
}

/// One round of the speculative-decode loop. Carries the tokens
/// committed this step, the count of drafter candidates the target
/// accepted (`acceptedCount` in `0...K`; `K+1` total committed
/// tokens on a full accept thanks to the bonus draw in the default
/// path), and the
/// state needed to start the next round (`nextToken`,
/// `nextPosition`).
public struct SmeltSpeculativeDecodeResult: Sendable {
    /// Tokens decoded this round, in order. In the default path,
    /// length is `acceptedCount + 1`: the accepted prefix plus
    /// either a bonus (full accept) or a replacement (rejection).
    /// In the drafted-bonus experiment, a full accept commits the
    /// verified draft tail directly, so length is `acceptedCount`.
    public let committedTokens: [Int32]
    /// Number of drafter candidates the target accepted. Range is
    /// normally `0...K`. With drafted-bonus enabled, the runtime
    /// asks the drafter for `K+1` candidates and a full accept is
    /// reported as `K+1` with no extra target bonus token appended.
    public let acceptedCount: Int
    /// The last committed token (replacement or bonus). Feed this
    /// as `lastToken` to the next `decodeStep` call.
    public let nextToken: Int32
    /// Position of `nextToken`. Feed as `position` to the next
    /// `decodeStep` call.
    public let nextPosition: Int32
    /// Wall-clock breakdown of this round's three GPU-bound phases
    /// plus the bookkeeping overhead. Sums to total round time
    /// within `CFAbsoluteTimeGetCurrent` resolution. Caller-side
    /// perf hunts (e.g. "verify cost dominates spec round at E4B
    /// scale") inspect these without re-instrumenting.
    public let phaseTimings: SmeltSpeculativePhaseTimings
}

/// Per-round timing breakdown. All fields in seconds.
public struct SmeltSpeculativePhaseTimings: Sendable {
    /// Cumulative drafter `draftStep` time, including the
    /// `target.drafterTaps()` call that lives just above it (taps
    /// is a manifest-walk small enough that bundling it into the
    /// drafter phase keeps the breakdown to four buckets).
    public let drafterSeconds: Double
    /// Target verify time: chunked-prefill argmax/full-logits or
    /// the sequential `verifyDraft` fallback. Single call per round.
    public let verifySeconds: Double
    /// CPU-side Leviathan greedy rejection loop. With the
    /// argmax-only verify table this compares K Int32s and appends
    /// the bonus Int32; the full-logits fallback still does K(+1)
    /// fp16-vocab argmaxes here.
    public let leviathanSeconds: Double
    /// End-of-round `target.decodeStep` that resets normOutBuf and
    /// K/V[committed.count - 1] to the committed-last-token state
    /// the next round's drafter call needs.
    public let refreshSeconds: Double
    /// Total `decodeStep` wall-clock (taps through result compose).
    /// Sum of the four phases above plus residual CPU overhead in
    /// guards, struct compose, etc.
    public let totalSeconds: Double
}

/// Holds a target `SmeltRuntime` and a `SmeltDrafter` for K-step
/// speculative decoding.
public final class SmeltSpeculativeRuntime {
    public let target: SmeltRuntime
    public let drafter: SmeltDrafter
    /// Default draft horizon. K=3 is the empirically measured
    /// tok/s optimum on Gemma 4 E4B-IT after the Phase 16-18 kernel
    /// work made verify and refresh much cheaper (70.15 tok/s vs
    /// K=4's 66.9 and K=5's 68.96 in 5x bench). With fast verify,
    /// the cost of generating more draft tokens (K=5 vs K=3)
    /// outweighs the higher α — fewer tokens per round, but each
    /// round is faster, and committed-tok throughput wins.
    /// Previously K=5 was optimal on the slower Phase 12 path
    /// (50.24 tok/s baseline). K=4 remains the E2B bring-up sweet
    /// spot.
    public static let defaultK: Int = 3

    public let K: Int
    private var logitsBufPrimed = false
    private var primedRow0Argmax: Int32?
    /// Skip the per-round target.decodeStep that re-anchors hidden
    /// state at the commit boundary. Opt-in via SMELT_SPEC_NO_REFRESH=1
    /// because the no-refresh path's missing prime makes the verify-
    /// argmax fast path disagree with plain greedy on most prompts
    /// (α→0).
    private let noRefresh: Bool
    /// Use HSD (Zhou et al. arxiv 2601.05724) instead of BV for the
    /// stochastic accept rule. Opt-in via SMELT_SPEC_USE_HSD=1
    /// because the published HSD-vs-BV α delta is empirical (~+2-3%)
    /// and not proven to dominate.
    private let useHSD: Bool
    /// Experimental K=3 stochastic Block Verification path that keeps
    /// target/drafter logits resident on Metal and reads back only the
    /// final accept decision.
    private let useGPUBV: Bool
    /// Experimental stochastic path: request one additional draft
    /// candidate, verify it with the same target prefill, and on full
    /// accept commit that verified tail instead of sampling a target
    /// bonus that would require a separate refresh decode.
    private let useDraftBonus: Bool
    /// Phase J: Daliri Gumbel-coupling (arXiv:2408.07978). When set,
    /// stochastic verify skips BV's softmax-and-ratio math and
    /// instead dispatches `sample_temperature_gumbel_fp16` once per
    /// verify row with `seed = baseSeed + k * golden`. The drafter
    /// uses the same per-step seed in its GPU sampler, so shared
    /// Gumbel noise couples drafter's argmax(q+g) to target's
    /// argmax(p+g). Accept iff equal. Output distribution = softmax(p).
    private let useGumbelCoupled: Bool

    public init(target: SmeltRuntime, drafter: SmeltDrafter, K: Int = defaultK) throws {
        guard K > 0 else {
            throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                "K must be positive; got \(K)"
            )
        }
        let env = ProcessInfo.processInfo.environment
        let useNoRefresh = env["SMELT_SPEC_NO_REFRESH"] == "1"
        let chunkedDisabled = env["SMELT_SPEC_DECODE_DISABLE_CHUNKED_VERIFY"] == "1"
        let tokenCount = K + 1
        let chunkedAvailable = !chunkedDisabled &&
            (target.canChunkedPrefillVerifyArgmax(tokenCount: tokenCount)
             || target.canChunkedPrefillVerify(tokenCount: tokenCount))
        self.target = target
        self.drafter = drafter
        self.K = K
        self.noRefresh = chunkedAvailable && useNoRefresh
        self.useHSD = env["SMELT_SPEC_USE_HSD"] == "1"
        self.useGPUBV = env["SMELT_SPEC_GPU_BV"] == "1"
        self.useDraftBonus = env["SMELT_SPEC_DRAFT_BONUS"] == "1"
        self.useGumbelCoupled = env["SMELT_SPEC_GUMBEL_COUPLED"] == "1"
    }

    public convenience init(
        target: SmeltRuntime,
        drafterPath: String,
        K: Int = defaultK
    ) throws {
        let manifest = try SmeltManifest.decode(
            from: Data(contentsOf: URL(fileURLWithPath: drafterPath)
                .appendingPathComponent("manifest.json"))
        )
        let drafter: SmeltDrafter
        if manifest.config.inputFusion?.postProjectionWidth == nil,
           manifest.config.inputFusion != nil {
            drafter = try SmeltTargetSeededAuxiliaryDrafter(
                packagePath: drafterPath,
                target: target
            )
        } else {
            drafter = try SmeltGemma4Drafter(
                packagePath: drafterPath,
                target: target
            )
        }
        try self.init(target: target, drafter: drafter, K: K)
    }

    /// Convenience: open the target and drafter packages from disk and
    /// instantiate a SmeltSpeculativeRuntime in one call. The opt-in
    /// surface for integrators that just need "target package +
    /// drafter package → spec-decode runtime."
    ///
    /// Only the Gemma 4 family of drafters has a concrete `SmeltDrafter`
    /// impl today, so the drafter package is assumed to be Gemma 4;
    /// `SmeltGemma4Drafter.init` cross-validates structural compat
    /// against the target. A second drafter family will turn this site
    /// into a dispatch on a manifest discriminator.
    public convenience init(
        targetPath: String,
        drafterPath: String,
        K: Int = defaultK,
        device: MTLDevice? = nil
    ) throws {
        let target = try SmeltRuntime(packagePath: targetPath, device: device)
        try self.init(target: target, drafterPath: drafterPath, K: K)
    }

    /// Call after running ops on `target` outside this runtime
    /// (plain `decodeStep`, `prefillStep`, etc.) so the next
    /// `decodeStep` re-primes instead of trusting stale logitsBuf.
    public func invalidateLogitsCache() {
        logitsBufPrimed = false
        primedRow0Argmax = nil
    }

    /// Adopt logits already produced by an external target prompt/decode.
    /// Recurrent targets cannot safely re-run the final prompt token merely to
    /// prime row zero, so callers that prepare the target before entering the
    /// speculative loop must hand that result across explicitly.
    public func adoptCurrentTargetLogits(argmax: Int32? = nil) {
        logitsBufPrimed = true
        primedRow0Argmax = argmax
    }

    public func resetSuffixCache(promptTokens: [Int32]) {
        drafter.resetSuffixCache(promptTokens: promptTokens)
    }

    public func recordGeneratedTokens(_ tokens: [Int32]) {
        drafter.recordGeneratedTokens(tokens)
    }

    /// Prime a target-seeded auxiliary module from the same prompt hidden rows
    /// that initialized the target. Non-target-seeded drafters ignore this
    /// surface because they own independent prompt/cache policy.
    public func primeTargetSeededDrafter(
        promptTokens: [Int32],
        targetNextToken: Int32,
        targetHiddenStates: [Data]
    ) throws {
        guard let seeded = drafter as? any SmeltTargetSeededDrafter else {
            return
        }
        try seeded.primeTargetContext(
            promptTokens: promptTokens,
            targetNextToken: targetNextToken,
            targetHiddenStates: targetHiddenStates
        )
    }

    /// Run one round: drafter produces K candidates → target
    /// verifies via K+1 sequential decodes → rejection sampling
    /// commits the longest-matching prefix plus replacement-or-
    /// bonus → one final decode at the commit boundary refreshes
    /// the runtime's K/V and normOut so the next round's drafter
    /// sees correct state.
    ///
    /// `selectionMode` picks the verifier rule. `.argmax` (or
    /// `.temperature` with non-positive/non-finite temp) keeps the
    /// greedy verify-argmax fast path: drafter argmax-picks each
    /// candidate, target argmax-verifies, accept iff equal.
    /// `.temperature` with positive finite temp routes through the
    /// full-logit verify (`prefillAllLogits` or `verifyDraft`) and
    /// runs Leviathan rejection (`step` / `bonus`) with the same
    /// temperature applied to both proposal `q` and target `p`.
    ///
    /// Side effect: target's K/V and normOut after the call
    /// reflect a decode of `nextToken` at `nextPosition`. Caller
    /// must not run their own `target.decodeStep` between rounds
    /// — feed `nextToken` / `nextPosition` straight into the
    /// next `decodeStep` call.
    public func decodeStep(
        lastToken: Int32, position: Int32,
        selectionMode: SmeltSelectionMode = .argmax
    ) throws -> SmeltSpeculativeDecodeResult {
        let usesUnsupportedFilteredSpeculation: Bool
        if case .filteredTemperature = selectionMode {
            usesUnsupportedFilteredSpeculation = true
        } else {
            usesUnsupportedFilteredSpeculation = false
        }
        let useArgmaxFastPath = selectionMode.usesArgmaxFastPath
        let temperature: Float
        let masterSeed: UInt64
        if !useArgmaxFastPath, case let .temperature(t, s) = selectionMode {
            temperature = t
            masterSeed = s
        } else {
            temperature = 1.0
            masterSeed = 0
        }
        let draftBonusActive = useDraftBonus && !useArgmaxFastPath
        let draftK = draftBonusActive ? K + 1 : K
        let totalStart = CFAbsoluteTimeGetCurrent()
        var preflightBatch: SmeltDraftBatch?
        var preflightSeconds = 0.0
        if let preflight = drafter as? any SmeltPreflightDrafter {
            let preflightStart = CFAbsoluteTimeGetCurrent()
            let batch = try preflight.preflightDraft(
                lastToken: lastToken,
                position: position,
                K: draftK,
                selectionMode: selectionMode
            )
            preflightSeconds = CFAbsoluteTimeGetCurrent() - preflightStart
            try validateDraftBatch(batch, expectedCount: draftK)
            if batch.K == 0 || usesUnsupportedFilteredSpeculation {
                return try decodeDeclinedRound(
                    lastToken: lastToken,
                    position: position,
                    selectionMode: selectionMode,
                    drafterSeconds: preflightSeconds,
                    totalStart: totalStart
                )
            }
            preflightBatch = batch
        }
        if usesUnsupportedFilteredSpeculation {
            throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                "speculative decode does not yet preserve top-k/top-p target distributions"
            )
        }
        let chunkedDisabled =
            ProcessInfo.processInfo.environment[
                "SMELT_SPEC_DECODE_DISABLE_CHUNKED_VERIFY"
            ] == "1"
        // Recurrent targets are safe only on the greedy argmax table whose
        // kernels checkpoint every conv/recurrent successor. Reject before the
        // drafter or target mutates anything; stochastic/full-logit and the
        // no-refresh experiment have no transactional state contract yet.
        if target.numDeltaLayers > 0 {
            guard useArgmaxFastPath,
                  !noRefresh,
                  !chunkedDisabled,
                  logitsBufPrimed,
                  target.canChunkedPrefillVerifyArgmax(tokenCount: K + 1)
            else {
                throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                    "target has \(target.numDeltaLayers) recurrent layers; "
                        + "speculative decode requires greedy transactional "
                        + "verify-argmax capacity for \(K + 1) inputs and "
                        + "adoptCurrentTargetLogits after prompt priming"
                )
            }
        }

        // Default verify writes K/V at positions [position,
        // position+K] and the final refresh decodes at position+K+1.
        // Drafted-bonus instead verifies K+1 draft tokens and ends on
        // that verified tail, so the maximum touched position is the
        // same. Refusing here keeps SmeltRuntime.decodeStep's
        // `position < contextLimit` precondition from trapping
        // mid-pipeline on a valid-looking call near the end.
        let needed = Int(position) + K + 2
        let limit = target.maxContextTokens
        guard needed <= limit else {
            throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                "decodeStep at position \(position) with K=\(K) needs "
                + "\(needed) context slots; target limit is \(limit). "
                + "Caller must fall back to plain decodeStep before "
                + "exhausting context."
            )
        }

        let gpuBVDrafter: SmeltGemma4Drafter?
        let gpuBVActive: Bool
        if draftBonusActive {
            guard !useHSD else {
                throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                    "SMELT_SPEC_DRAFT_BONUS=1 supports BV only; unset SMELT_SPEC_USE_HSD"
                )
            }
            guard !useGPUBV else {
                throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                    "SMELT_SPEC_DRAFT_BONUS=1 does not compose with SMELT_SPEC_GPU_BV=1"
                )
            }
            guard target.canChunkedPrefillVerify(tokenCount: draftK + 1) else {
                throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                    "SMELT_SPEC_DRAFT_BONUS=1 requires chunked full-logits verify "
                    + "for \(draftK + 1) tokens"
                )
            }
        }

        if useGPUBV && !useArgmaxFastPath {
            guard K == 3 else {
                throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                    "SMELT_SPEC_GPU_BV=1 currently supports K=3 only; got K=\(K)"
                )
            }
            guard !useHSD else {
                throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                    "SMELT_SPEC_GPU_BV=1 supports BV only; unset SMELT_SPEC_USE_HSD"
                )
            }
            guard target.supportsGPUBlockVerificationK3 else {
                throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                    "SMELT_SPEC_GPU_BV=1 requires target model.metallib to contain "
                    + "spec_bv_k3; rebuild the target package"
                )
            }
            guard target.canChunkedPrefillVerify(tokenCount: K + 1) else {
                throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                    "SMELT_SPEC_GPU_BV=1 requires chunked full-logits verify"
                )
            }
            guard let gemma = drafter as? SmeltGemma4Drafter else {
                throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                    "SMELT_SPEC_GPU_BV=1 currently supports SmeltGemma4Drafter only"
                )
            }
            gpuBVDrafter = gemma
            gpuBVActive = true
        } else {
            gpuBVDrafter = nil
            gpuBVActive = false
        }

        let batch: SmeltDraftBatch
        let drafterSeconds: Double
        if let preflightBatch {
            batch = preflightBatch
            drafterSeconds = preflightSeconds
        } else {
            let drafterStart = CFAbsoluteTimeGetCurrent()
            let taps = try target.drafterTaps()
            if let seeded = drafter as? any SmeltTargetSeededDrafter {
                guard useArgmaxFastPath, logitsBufPrimed else {
                    throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                        "target-seeded drafter requires greedy decode and primed target row-zero logits"
                    )
                }
                let targetNextToken = primedRow0Argmax
                    ?? SmeltLeviathanRejection.greedyBonus(
                        targetLogits: target.allLogitsHalf()
                    )
                primedRow0Argmax = targetNextToken
                batch = try seeded.draftStep(
                    targetTaps: taps,
                    targetNextToken: targetNextToken,
                    lastToken: lastToken,
                    position: position,
                    K: draftK,
                    selectionMode: selectionMode
                )
            } else {
                batch = try drafter.draftStep(
                    targetTaps: taps,
                    lastToken: lastToken,
                    position: position,
                    K: draftK,
                    selectionMode: selectionMode
                )
            }
            drafterSeconds = CFAbsoluteTimeGetCurrent() - drafterStart
        }
        try validateDraftBatch(batch, expectedCount: draftK)
        if batch.K == 0 {
            return try decodeDeclinedRound(
                lastToken: lastToken,
                position: position,
                selectionMode: selectionMode,
                drafterSeconds: drafterSeconds,
                totalStart: totalStart
            )
        }
        if !useArgmaxFastPath && !gpuBVActive {
            // Row-count parity (`q.rowCount == candidates.count`) is
            // already enforced by `SmeltDraftBatch.init`; only the
            // .none rejection and the dense per-row vocab-width check
            // need to fire here.
            switch batch.q {
            case .none:
                throw SmeltSpeculativeRuntimeError.targetDrafterMismatch(
                    "stochastic decodeStep requires q to be .dense or "
                    + ".sparse; drafter returned .none"
                )
            case .sparse:
                break
            case .dense(let rows):
                let targetVocab = Int(target.vocabSize)
                for (k, row) in rows.enumerated() {
                    guard row.count == targetVocab else {
                        throw SmeltSpeculativeRuntimeError.targetDrafterMismatch(
                            "stochastic decodeStep: drafter logit row \(k) "
                            + "width \(row.count) != target vocab \(targetVocab)"
                        )
                    }
                }
            }
        }

        // emit_all_logits-built targets with enough prefill batch
        // capacity get chunked-prefill verify (one batched GPU call
        // producing the verify rows plus the bonus/next row).
        // Otherwise fall back to sequential decode + skip-first
        // optimization.
        //
        // SMELT_SPEC_DECODE_DISABLE_CHUNKED_VERIFY=1 forces the
        // sequential path even on capable targets — a diagnostic
        // for when chunked-prefill correctness is suspect (e.g.,
        // E4B α=0 hunt: rules out chunked-prefill as the divergence
        // surface vs the drafter wrapper itself).
        let inputs = [lastToken] + batch.candidates
        var pLogits: [[Float16]]?
        var targetArgmaxes: [Int32]?
        // For recurrent verification, row 0 is the live entry state. An
        // unprimed verify also consumes `lastToken`, so the accepted candidate
        // boundary is shifted by one row. nil means the row-0 gate rejected
        // before a verify pass was needed.
        var recurrentVerifyAcceptedRowBase: Int?

        let vocab = Int32(target.vocabSize)
        for (i, c) in batch.candidates.enumerated() {
            guard c >= 0, c < vocab else {
                throw SmeltSpeculativeRuntimeError.targetDrafterMismatch(
                    "candidate[\(i)] = \(c) out of vocab range [0, \(vocab))"
                )
            }
        }

        var precomputedCand0: SmeltLeviathanRejection.Decision?
        // Batched prefill row whose hidden/logits correspond to the
        // last drafted candidate. Only meaningful when
        // `draftBonusActive` and the whole draft is accepted.
        var draftedTailPrefillRow: Int?

        // Phase J: gumbel-coupled verify decision, populated when
        // SMELT_SPEC_GUMBEL_COUPLED=1 and stochastic. Carries the
        // accept count + the target's token to commit (either the
        // disagreement pick or the bonus on full accept).
        var gumbelCoupledDecision: (acceptedCount: Int, token: Int32)?
        let isSparseQ: Bool
        if case .sparse = batch.q { isSparseQ = true } else { isSparseQ = false }
        // Coupling requires the drafter to have sampled with the same
        // Gumbel-max kernel the target will use. Suffix-lookup hits
        // (sparse q) come from `sampleFromHistogram`, not the GPU
        // sampler, so shared Gumbel noise doesn't align — coupling
        // would silently degrade to "random tokens agree." Fall back
        // to BV's full softmax+ratio path for sparse q.
        let gumbelCoupledActive = useGumbelCoupled
            && !useArgmaxFastPath
            && !gpuBVActive
            && !useHSD
            && !draftBonusActive
            && !isSparseQ
            && target.canChunkedPrefillVerify(tokenCount: inputs.count)

        let verifyStart = CFAbsoluteTimeGetCurrent()
        if gumbelCoupledActive {
            target.armProfileForNextPrefill()
            try target.prefillAllLogitsResident(
                tokens: inputs, startPos: position
            )
            gumbelCoupledDecision = try target.runGumbelCoupledStep(
                candidates: batch.candidates,
                baseSeed: masterSeed,
                position: position,
                temperature: temperature
            )
            // The refresh path below will reset logitsBufPrimed=true;
            // this assignment marks "stale" intent but is overwritten.
            primedRow0Argmax = nil
        } else if gpuBVActive {
            if logitsBufPrimed {
                try target.stageCurrentLogitsForSpecBV(row: 0)
                target.armProfileForNextPrefill()
                try target.prefillAllLogitsResident(
                    tokens: batch.candidates, startPos: position + 1
                )
                try target.stageResidentPrefillLogitsForSpecBV(
                    rowCount: draftK, destinationStartRow: 1
                )
            } else {
                target.armProfileForNextPrefill()
                try target.prefillAllLogitsResident(
                    tokens: inputs, startPos: position
                )
                try target.stageResidentPrefillLogitsForSpecBV(
                    rowCount: draftK + 1, destinationStartRow: 0
                )
            }
            logitsBufPrimed = true
        } else if useArgmaxFastPath, !chunkedDisabled,
           target.canChunkedPrefillVerifyArgmax(tokenCount: inputs.count) {
            if logitsBufPrimed {
                let row0Argmax = primedRow0Argmax
                    ?? SmeltLeviathanRejection.greedyBonus(
                        targetLogits: target.allLogitsHalf()
                    )
                primedRow0Argmax = row0Argmax
                let cand0 = SmeltLeviathanRejection.greedyStepFromArgmax(
                    row0Argmax, candidate: batch.candidates[0]
                )
                precomputedCand0 = cand0
                if !cand0.accepted {
                    targetArgmaxes = [row0Argmax]
                } else {
                    let prefillArgmaxes = try target.prefillVerifyArgmax(
                        tokens: batch.candidates, startPos: position + 1
                    )
                    if target.numDeltaLayers > 0 {
                        recurrentVerifyAcceptedRowBase = 0
                    }
                    targetArgmaxes = [row0Argmax] + prefillArgmaxes
                    draftedTailPrefillRow = batch.candidates.count - 1
                }
            } else {
                targetArgmaxes = try target.prefillVerifyArgmax(
                    tokens: inputs, startPos: position
                )
                draftedTailPrefillRow = batch.candidates.count
                if target.numDeltaLayers > 0 {
                    recurrentVerifyAcceptedRowBase = 1
                }
            }
            logitsBufPrimed = true
        } else if !chunkedDisabled,
           target.canChunkedPrefillVerify(tokenCount: inputs.count) {
            // When cand_0 fails the row-0 gate, the prefill is
            // skipped entirely. K/V[position+1] gets written by
            // refresh below with the replacement token;
            // K/V[position+2..position+K] stays stale but the next
            // round only reads up to its own commit boundary, so
            // the stale tail is self-healing.
            if logitsBufPrimed, useArgmaxFastPath {
                let row0 = target.allLogitsHalf()
                let cand0 = SmeltLeviathanRejection.greedyStep(
                    targetLogits: row0, candidate: batch.candidates[0]
                )
                precomputedCand0 = cand0
                primedRow0Argmax = cand0.token
                if !cand0.accepted {
                    pLogits = [row0]
                } else {
                    target.armProfileForNextPrefill()
                    let prefillLogits = try target.prefillAllLogits(
                        tokens: batch.candidates, startPos: position + 1
                    )
                    pLogits = [row0] + prefillLogits
                }
            } else if logitsBufPrimed {
                let row0 = target.allLogitsHalf()
                target.armProfileForNextPrefill()
                let prefillLogits = try target.prefillAllLogits(
                    tokens: batch.candidates, startPos: position + 1
                )
                pLogits = [row0] + prefillLogits
                if draftBonusActive {
                    draftedTailPrefillRow = batch.candidates.count - 1
                }
            } else {
                target.armProfileForNextPrefill()
                pLogits = try target.prefillAllLogits(
                    tokens: inputs, startPos: position
                )
                if draftBonusActive {
                    draftedTailPrefillRow = batch.candidates.count
                }
            }
            logitsBufPrimed = true
        } else {
            // Validate lastToken's range BEFORE priming so a
            // malformed id throws cleanly instead of triggering
            // an OOB embed_tokens read inside decodeStep.
            if !logitsBufPrimed {
                guard lastToken >= 0, lastToken < vocab else {
                    throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                        "lastToken \(lastToken) out of vocab range [0, \(vocab))"
                    )
                }
                primedRow0Argmax = try target.decodeStep(
                    tokenId: lastToken, position: position
                )
                logitsBufPrimed = true
            }

            pLogits = try target.verifyDraft(
                tokens: inputs,
                startPosition: position,
                firstRowFromLogitsBuf: true
            )
        }
        let verifySeconds = CFAbsoluteTimeGetCurrent() - verifyStart
        let greedyVerifiedTailAvailable = useArgmaxFastPath
            && targetArgmaxes?.count == K + 1
            && draftedTailPrefillRow != nil

        let leviathanStart = CFAbsoluteTimeGetCurrent()
        var committed: [Int32] = []
        committed.reserveCapacity(draftBonusActive ? draftK : K + 1)
        var acceptedCount = 0
        if let gd = gumbelCoupledDecision {
            // Phase J: target's gumbel-coupled picks decided acceptance
            // already; nothing to do CPU-side beyond committing.
            acceptedCount = gd.acceptedCount
            for k in 0 ..< acceptedCount {
                committed.append(batch.candidates[k])
            }
            committed.append(gd.token)
        } else if useArgmaxFastPath {
            for k in 0 ..< K {
                let d: SmeltLeviathanRejection.Decision
                if k == 0, let cached = precomputedCand0 {
                    d = cached
                } else if let targetArgmaxes {
                    d = SmeltLeviathanRejection.greedyStepFromArgmax(
                        targetArgmaxes[k], candidate: batch.candidates[k]
                    )
                } else {
                    d = SmeltLeviathanRejection.greedyStep(
                        targetLogits: pLogits![k], candidate: batch.candidates[k]
                    )
                }
                committed.append(d.token)
                if !d.accepted { break }
                acceptedCount += 1
            }
            if acceptedCount == K {
                if !greedyVerifiedTailAvailable {
                    let bonusToken: Int32
                    if let targetArgmaxes {
                        bonusToken = targetArgmaxes[K]
                    } else {
                        bonusToken = SmeltLeviathanRejection.greedyBonus(
                            targetLogits: pLogits![K]
                        )
                    }
                    committed.append(bonusToken)
                }
            }
        } else if gpuBVActive {
            guard let draftLogits = gpuBVDrafter?.stagedGPUDraftLogits else {
                throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                    "SMELT_SPEC_GPU_BV=1 drafter did not stage q logits"
                )
            }
            let decision = try target.runSpecBVK3(
                draftLogits: draftLogits,
                candidates: batch.candidates,
                temperature: temperature,
                seed: masterSeed,
                position: position
            )
            acceptedCount = decision.acceptedCount
            for k in 0 ..< acceptedCount {
                committed.append(batch.candidates[k])
            }
            committed.append(decision.token)
        } else {
            var acceptRng = SmeltDeterministicRng(
                masterSeed: masterSeed,
                domain: .acceptUniform,
                position: position
            )
            var residualRng = SmeltDeterministicRng(
                masterSeed: masterSeed,
                domain: .residualCategorical,
                position: position
            )
            let block: SmeltLeviathanRejection.BlockDecision
            switch batch.q {
            case .none:
                throw SmeltSpeculativeRuntimeError.targetDrafterMismatch(
                    "stochastic verify path reached with q=.none"
                )
            case .dense(let rows):
                block = useHSD
                    ? SmeltLeviathanRejection.hsdStep(
                        targetLogits: pLogits!,
                        drafterLogits: rows,
                        candidates: batch.candidates,
                        temperature: temperature,
                        acceptRng: &acceptRng,
                        residualOrBonusRng: &residualRng
                    )
                    : SmeltLeviathanRejection.blockStep(
                        targetLogits: pLogits!,
                        drafterLogits: rows,
                        candidates: batch.candidates,
                        temperature: temperature,
                        acceptRng: &acceptRng,
                        residualOrBonusRng: &residualRng
                    )
            case .sparse(let rows):
                block = useHSD
                    ? SmeltLeviathanRejection.hsdStepSparse(
                        targetLogits: pLogits!,
                        drafterSparseLogits: rows,
                        candidates: batch.candidates,
                        temperature: temperature,
                        acceptRng: &acceptRng,
                        residualOrBonusRng: &residualRng
                    )
                    : SmeltLeviathanRejection.blockStepSparse(
                        targetLogits: pLogits!,
                        drafterSparseLogits: rows,
                        candidates: batch.candidates,
                        temperature: temperature,
                        acceptRng: &acceptRng,
                        residualOrBonusRng: &residualRng
                    )
            }
            acceptedCount = block.acceptedCount
            if draftBonusActive && acceptedCount == draftK {
                committed.append(contentsOf: batch.candidates)
            } else {
                for k in 0 ..< acceptedCount {
                    committed.append(batch.candidates[k])
                }
                committed.append(block.token)
            }
        }
        let leviathanSeconds = CFAbsoluteTimeGetCurrent() - leviathanStart

        // `committed.last!` safe by construction: the K-loop appends
        // on every iteration (K >= 1 enforced at init), and the
        // full-accept branch above also appends.
        let nextToken = committed.last!
        let nextPosition = position + Int32(committed.count)

        if let rowBase = recurrentVerifyAcceptedRowBase {
            try target.enqueueRecurrentVerifyStateCommit(
                historyRow: rowBase + acceptedCount
            )
        }

        // Refresh K/V + normOut at the commit boundary. In the
        // default path, partial accept overwrites the rejected
        // candidate's K/V at pos+acceptedCount+1 with the replacement;
        // full accept writes K/V[pos+K+1] for the target bonus. The
        // drafted-bonus experiment instead lands on a verified draft
        // tail whose K/V already exists, so it only moves the prefill
        // hidden/logits rows back to the decode head. Either way the
        // runtime is now positioned at the last committed token and
        // row-0 logits are primed for the next verify.
        //
        // The opt-in no-refresh path memcpys verify's predecessor-
        // position hidden to head and SKIPS the forward pass — K/V
        // at nextPosition stays unwritten until next verify's row 0
        // overwrites it. See `noRefresh` doc above.
        //
        // Phase C Unit 1 (refresh-elim in stochastic) was tried and
        // REVERTED here: the Google Gemma 4 E4B-IT assistant drafter
        // is Q-only with external KV against the target. Its first
        // decodeStep at the committed boundary attends to target's
        // K/V at that position — which is the cell refresh writes.
        // Skipping refresh leaves K/V at the boundary as stale data,
        // corrupting the drafter's attention and tanking α
        // (2.90 → 2.00 on Gemma 4 E4B-IT K=3 bench). The 11.2 ms
        // refresh is real necessary work, not bookkeeping overhead.
        let refreshStart = CFAbsoluteTimeGetCurrent()
        if greedyVerifiedTailAvailable && acceptedCount == K {
            guard let row = draftedTailPrefillRow,
                  let nextArgmax = targetArgmaxes?[K]
            else {
                throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                    "greedy verified-tail commit is missing its hidden row or next argmax"
                )
            }
            try target.copyVerifyHiddenToHead(offsetTokens: row)
            logitsBufPrimed = true
            primedRow0Argmax = nextArgmax
        } else if draftBonusActive && acceptedCount == draftK {
            guard let row = draftedTailPrefillRow else {
                throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                    "SMELT_SPEC_DRAFT_BONUS=1 full accept has no verified tail row"
                )
            }
            try target.copyVerifyHiddenToHead(offsetTokens: row)
            try target.copyPrefillLogitsToHead(offsetTokens: row)
            logitsBufPrimed = true
            primedRow0Argmax = nil
        } else if noRefresh {
            try target.copyVerifyHiddenToHead(offsetTokens: acceptedCount)
            logitsBufPrimed = false
            primedRow0Argmax = nil
        } else {
            let refreshedArgmax = try target.decodeStep(
                tokenId: nextToken, position: nextPosition
            )
            logitsBufPrimed = true
            // Stochastic mode reads next round's row-0 from
            // `target.allLogitsHalf()` directly; caching the argmax
            // here would invite a greedy comparison against drafter
            // q on the wrong policy if a later call slips back into
            // argmax mode mid-session.
            primedRow0Argmax = useArgmaxFastPath ? refreshedArgmax : nil
        }
        if let seeded = drafter as? any SmeltTargetSeededDrafter {
            try seeded.commitDraft(
                batch: batch,
                acceptedCount: acceptedCount,
                committedTokens: committed,
                startPosition: position
            )
        }
        let refreshSeconds = CFAbsoluteTimeGetCurrent() - refreshStart

        let totalSeconds = CFAbsoluteTimeGetCurrent() - totalStart
        return SmeltSpeculativeDecodeResult(
            committedTokens: committed,
            acceptedCount: acceptedCount,
            nextToken: nextToken,
            nextPosition: nextPosition,
            phaseTimings: SmeltSpeculativePhaseTimings(
                drafterSeconds: drafterSeconds,
                verifySeconds: verifySeconds,
                leviathanSeconds: leviathanSeconds,
                refreshSeconds: refreshSeconds,
                totalSeconds: totalSeconds
            )
        )
    }

    private func validateDraftBatch(
        _ batch: SmeltDraftBatch,
        expectedCount: Int
    ) throws {
        guard batch.K == 0 || batch.K == expectedCount else {
            throw SmeltSpeculativeRuntimeError.targetDrafterMismatch(
                "drafter returned K=\(batch.K) candidates, expected "
                    + "\(expectedCount). SmeltDrafter contract permits only "
                    + "a full block or an empty conditional decline."
            )
        }
    }

    /// Commit one ordinary target token after a conditional drafter declines.
    /// This helper deliberately has no verifier/state-transaction dependency.
    /// When row-zero logits are already live it selects directly from them;
    /// otherwise it performs the same initial target decode a plain caller
    /// would need. The final decode advances the target to the committed-token
    /// boundary so the next round observes normal decode state.
    private func decodeDeclinedRound(
        lastToken: Int32,
        position: Int32,
        selectionMode: SmeltSelectionMode,
        drafterSeconds: Double,
        totalStart: CFAbsoluteTime
    ) throws -> SmeltSpeculativeDecodeResult {
        let nextPosition = position + 1
        guard Int(nextPosition) < target.maxContextTokens else {
            throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                "conditional decline at position \(position) needs context "
                    + "slot \(nextPosition); target limit is "
                    + "\(target.maxContextTokens)"
            )
        }

        let refreshStart = CFAbsoluteTimeGetCurrent()
        let nextToken: Int32
        if logitsBufPrimed {
            nextToken = target.selectCurrentToken(
                position: position,
                selectionMode: selectionMode
            )
        } else {
            nextToken = try target.decodeStep(
                tokenId: lastToken,
                position: position,
                selectionMode: selectionMode
            )
        }
        let refreshedToken = try target.decodeStep(
            tokenId: nextToken,
            position: nextPosition
        )
        logitsBufPrimed = true
        primedRow0Argmax = selectionMode.usesArgmaxFastPath
            ? refreshedToken : nil
        let refreshSeconds = CFAbsoluteTimeGetCurrent() - refreshStart
        return SmeltSpeculativeDecodeResult(
            committedTokens: [nextToken],
            acceptedCount: 0,
            nextToken: nextToken,
            nextPosition: nextPosition,
            phaseTimings: SmeltSpeculativePhaseTimings(
                drafterSeconds: drafterSeconds,
                verifySeconds: 0,
                leviathanSeconds: 0,
                refreshSeconds: refreshSeconds,
                totalSeconds: CFAbsoluteTimeGetCurrent() - totalStart
            )
        )
    }
}

public enum SmeltSpeculativeRuntimeError: Error, CustomStringConvertible {
    case invalidConfiguration(String)
    /// Drafter and target disagree on a structural invariant
    /// (tokenizer hash, drafter's backbone hidden size vs
    /// target's hidden size). Caught at composition, not decode.
    case targetDrafterMismatch(String)

    public var description: String {
        switch self {
        case .invalidConfiguration(let detail):
            return "Speculative runtime configuration: \(detail)"
        case .targetDrafterMismatch(let detail):
            return "Target/drafter mismatch: \(detail)"
        }
    }
}
