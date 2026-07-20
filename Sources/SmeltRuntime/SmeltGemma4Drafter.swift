// SmeltGemma4Drafter — concrete EAGLE-class drafter for the Gemma 4
// E2B/E4B *-it-assistant family. Wraps a drafter `.smeltpkg`'s
// `SmeltRuntime` and consumes the target's `SmeltDrafterTaps` to
// produce K candidate tokens per `draftStep`.
//
// Per-step shape:
//   1. Look up `embed_tokens[lastToken]` from the TARGET runtime
//      (the Gemma 4 assistant ties its lm-head weight to the
//      target's embedding row, then drafts on the cluster path —
//      so the drafter's pre_projection input begins with the
//      target's embedding row, not the drafter's).
//   2. Concat with the target's post-final-norm hidden state to
//      form the `pre_projection` input. Layout: `[embed_row,
//      hidden_row]` fp16, total `2 * hiddenSize * 2` bytes.
//   3. Bind each drafter attention layer's `keyCache_<n>` /
//      `valCache_<n>` slot to the target's family-matched K/V
//      buffer. Drafter never writes these slots under
//      `external_kv`.
//   4. Run drafter's `decodeStep`. The pre_projection dispatch
//      overrides hiddenA before any layer reads it; the
//      embedding gather output is dead.
//   5. Read `allLogits()`, take argmax, and return as a single-
//      candidate `SmeltDraftBatch`.
//
// V1 scope:
//   - Argmax sampling.
//   - Target `embed_tokens` may be fp16 or quantized; draftStep
//     calls `target.embedToken`, which returns fp16 row bytes for
//     the pre_projection input.
//   - Concrete to Gemma 4. Llama / Qwen / Mistral drafters get
//     their own concrete `SmeltDrafter` impls — the trade-off
//     against a single generic drafter is per-arch concat layout
//     and tied-vs-untied lm_head policy, neither of which is
//     uniform across the EAGLE family.

import Foundation
import Metal
import SmeltSchema

public final class SmeltGemma4Drafter: SmeltDrafter {
    private let inner: SmeltRuntime
    private let target: SmeltRuntime
    private let hiddenBSlot: Int
    /// The contextCapacity `inner` was last prepared with. Tracks
    /// the bound target cache length so we only re-prepare when
    /// the target reallocates (e.g., on a longer prompt). 0 = not
    /// yet prepared; first draftStep prepares from the live taps.
    private var preparedContextCapacity: Int = 0
    private var gpuDraftLogitsBuffer: MTLBuffer?

    /// Cached env reads; the K-loop is on the hot path and re-reading
    /// `ProcessInfo.environment` per step is wasteful.
    private let gpuBVStagingEnabled: Bool
    private let drafterGPUSampleEnabled: Bool
    private let drafterGreedyEnabled: Bool
    private let suffixLookupDebug: Bool
    private let suffixLookupDrafter: SmeltSuffixLookupDrafter?

    private struct LayerTap {
        let family: String
        let keySlot: Int
        let valSlot: Int
    }
    private let layerTaps: [LayerTap]

    public var stagedGPUDraftLogits: MTLBuffer? {
        gpuDraftLogitsBuffer
    }

    public init(packagePath: String, target: SmeltRuntime) throws {
        // Construct on the target's Metal device — bindExternalKVBuffer
        // hands MTLBuffer references between runtimes, and Metal
        // resources are device-owned. A cross-device bind would fail
        // at encode time on multi-GPU setups.
        self.inner = try SmeltRuntime(
            packagePath: packagePath, device: target.metalDevice
        )
        self.target = target

        // Prepare lazily in draftStep, sized to the target's live
        // K/V buffers. Preparing here at a placeholder capacity
        // would force the drafter cache to a stale size and the
        // bindExternalKVBuffer length check would reject the
        // target's actual buffers (different length).

        let manifestData = try Data(
            contentsOf: URL(fileURLWithPath: "\(packagePath)/manifest.json")
        )
        let manifest = try SmeltManifest.decode(from: manifestData)

        // The other compat checks SmeltDrafterTargetCompat designs for
        // (backbone-hidden-size match, tokenizer-hash match) need their
        // source fields emitted into the runtime manifest before we can
        // enforce them; backboneHiddenSize on a drafter package is its
        // own internal width (256 on E2B) not the target backbone it
        // expects (1536/2560), and tokenizer hashes aren't emitted at
        // all yet. Vocab equality is the strongest cross-package
        // invariant we can check today.
        let targetVocab = target.vocabSize
        let drafterVocab = manifest.config.vocabSize
        guard targetVocab == drafterVocab else {
            throw SmeltSpeculativeRuntimeError.targetDrafterMismatch(
                "Gemma4Drafter: target vocab \(targetVocab) != drafter "
                + "vocab \(drafterVocab) at \(packagePath); tokenizer "
                + "disagreement makes verify token-id comparisons "
                + "meaningless"
            )
        }

        guard let hiddenB = manifest.buffers.slots.first(
            where: { $0.name == "hiddenB" }
        ) else {
            throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                "Gemma4Drafter: drafter package missing hiddenB slot"
            )
        }
        self.hiddenBSlot = hiddenB.index

        guard let provenance = manifest.buildProvenance else {
            // The drafter needs the materialised layer pattern to
            // route per-layer K/V binds; that pattern lives on the
            // build provenance block. Legacy manifests predating
            // provenance need a rebuild before drafter use.
            throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                "Gemma4Drafter: drafter manifest missing buildProvenance; "
                + "rebuild the assistant package"
            )
        }
        let pattern = provenance.resolvedOptions.expandedLayerPattern
        var taps: [LayerTap] = []
        var attnIdx = -1
        for layer in pattern {
            guard SmeltAttentionFamily.known.contains(layer) else { continue }
            attnIdx += 1
            let keyName = "keyCache_\(attnIdx)"
            let valName = "valCache_\(attnIdx)"
            guard let keySlot = manifest.buffers.slots.first(where: { $0.name == keyName }),
                  let valSlot = manifest.buffers.slots.first(where: { $0.name == valName })
            else {
                throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                    "Gemma4Drafter: drafter manifest missing \(keyName)/\(valName) slot"
                )
            }
            taps.append(LayerTap(
                family: layer, keySlot: keySlot.index, valSlot: valSlot.index
            ))
        }
        guard !taps.isEmpty else {
            throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                "Gemma4Drafter: drafter package has no attention layers"
            )
        }
        self.layerTaps = taps

        let env = ProcessInfo.processInfo.environment
        self.gpuBVStagingEnabled = env["SMELT_SPEC_GPU_BV"] == "1"
        // GPU-side sampling in the K-loop is the default. BV still gets
        // the full q distribution via allLogitsHalf. SMELT_LEGACY_SAMPLER=1
        // falls back to the CPU sampler for bit-exact compat.
        let useLegacySampler = env["SMELT_LEGACY_SAMPLER"] == "1"
        self.drafterGPUSampleEnabled = !useLegacySampler
        // Greedy drafter: experiment knob that proposes the LM-head argmax
        // instead of a stochastic sample. **WARNING — mathematically
        // incorrect** for stochastic BV/HSD: those Leviathan-rejection
        // schemes require proposals sampled from q. With argmax proposals
        // the joint coupling is biased (e.g., if p==q the argmax token is
        // accepted with probability 1 instead of p(x)). Keep this knob only
        // for offline α-comparison experiments; never enable in production
        // stochastic decode.
        self.drafterGreedyEnabled = env["SMELT_SPEC_DRAFTER_GREEDY"] == "1"
        self.suffixLookupDebug = env["SMELT_SPEC_SUFFIX_DEBUG"] == "1"
        self.suffixLookupDrafter = env["SMELT_SPEC_SUFFIX_LOOKUP"] == "1"
            ? SmeltSuffixLookupDrafter()
            : nil
    }

    public func resetSuffixCache(promptTokens: [Int32]) {
        guard let suffixLookupDrafter else { return }
        let attempts = suffixLookupDrafter.hits + suffixLookupDrafter.misses
        if suffixLookupDebug, attempts > 0 {
            let rate = Double(suffixLookupDrafter.hits) / Double(attempts)
            fputs(
                "[suffix] hits=\(suffixLookupDrafter.hits) "
                + "misses=\(suffixLookupDrafter.misses) "
                + "rate=\(String(format: "%.2f", rate))\n",
                stderr
            )
        }
        suffixLookupDrafter.resetSuffixCache(promptTokens: promptTokens)
    }

    public func recordGeneratedTokens(_ tokens: [Int32]) {
        // Skip when the feature is off: retaining the caller's buffer
        // would defeat CoW on its next append and add O(n^2) copies to
        // bench measurements. Codex P2 finding on chunk close.
        suffixLookupDrafter?.recordGeneratedTokens(tokens)
    }

    private func ensureGPUDraftLogitsBuffer(K: Int) throws -> MTLBuffer {
        let bytes = K * target.vocabSize * MemoryLayout<Float16>.stride
        if let buffer = gpuDraftLogitsBuffer, buffer.length >= bytes {
            return buffer
        }
        guard let buffer = target.metalDevice.makeBuffer(
            length: max(bytes, 16), options: .storageModeShared
        ) else {
            throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                "Gemma4Drafter: failed to allocate GPU draft logits buffer "
                + "(\(bytes) bytes)"
            )
        }
        memset(buffer.contents(), 0, buffer.length)
        buffer.label = "smelt.spec_bv.draft_logits"
        gpuDraftLogitsBuffer = buffer
        return buffer
    }

    public func draftStep(
        targetTaps: SmeltDrafterTaps,
        lastToken: Int32,
        position: Int32,
        K: Int,
        selectionMode: SmeltSelectionMode
    ) throws -> SmeltDraftBatch {
        guard K >= 1 else {
            throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                "Gemma4Drafter: K must be >= 1; got \(K)"
            )
        }
        let useArgmaxFastPath = selectionMode.usesArgmaxFastPath
        let gpuBVStaging = !useArgmaxFastPath && gpuBVStagingEnabled
        let drafterGPUSample =
            !useArgmaxFastPath && !gpuBVStaging && drafterGPUSampleEnabled

        if let suffixLookupDrafter, !gpuBVStaging {
            let lookupBatch = try suffixLookupDrafter.draftStep(
                targetTaps: targetTaps,
                lastToken: lastToken,
                position: position,
                K: K,
                selectionMode: selectionMode
            )
            if lookupBatch.K > 0 {
                return lookupBatch
            }
        }

        // Resize drafter K/V slots to match the live target cache
        // before any bind. bindExternalKVBuffer requires byte-exact
        // length match between the slot's allocation and the
        // external buffer; preparing once at init at a fixed
        // capacity (e.g., 1) would lock us into single-token
        // decodes. contextCapacity is uniform across families per
        // package (SmeltBufferPlan.swift:395), so the first tap's
        // value is authoritative.
        guard let firstTap = targetTaps.attention.values.first else {
            throw SmeltSpeculativeRuntimeError.targetDrafterMismatch(
                "Gemma4Drafter: target taps carry no attention families"
            )
        }
        let targetCap = firstTap.contextCapacity
        if preparedContextCapacity != targetCap {
            try inner.prepareForRequest(
                batchCapacity: 1, contextCapacity: targetCap
            )
            preparedContextCapacity = targetCap
        }

        // Hoisted out of the K-loop: target taps are stable for the
        // lifetime of this draftStep call (single-threaded contract;
        // no target prepareForRequest fires between iterations), and
        // drafter dispatches are external_kv (no writes). Re-binding
        // K times would be a no-op.
        for tap in layerTaps {
            guard let attTap = targetTaps.attention[tap.family] else {
                throw SmeltSpeculativeRuntimeError.targetDrafterMismatch(
                    "Gemma4Drafter: target taps missing attention family '\(tap.family)' "
                    + "needed by drafter layer pattern"
                )
            }
            try inner.bindExternalKVBuffer(at: tap.keySlot, buffer: attTap.keyCache)
            try inner.bindExternalKVBuffer(at: tap.valSlot, buffer: attTap.valueCache)
        }

        let fp16Stride = MemoryLayout<Float16>.stride
        let hiddenSize = targetTaps.hiddenSize
        let rowBytes = hiddenSize * fp16Stride

        // Reads from offset 0 — correct for post-decode (single
        // row at the head). Chunked-prefill targets put the last
        // token at `(seqLen - 1) * hiddenSize * 2`; see the
        // `lastHiddenState` doc on SmeltDrafterTaps for why an
        // active-offset field is owed before this drafter handles
        // prefill input.
        var hiddenBytes = Data(
            bytes: targetTaps.lastHiddenState.contents(), count: rowBytes
        )
        var currentToken = lastToken
        var candidates: [Int32] = []
        candidates.reserveCapacity(K)
        // Argmax fast path: `inner.decodeStep` returns the candidate
        // through the fp16 argmax slot, so per-step logits never
        // leave GPU memory. Stochastic mode needs the full fp16
        // logits row to match the verify-side Leviathan ratio
        // against the proposal `q` that produced each candidate;
        // `inner.allLogitsHalf()` is one stride-copy per step.
        var perStepLogits: [[Float16]] = []
        if !useArgmaxFastPath && !gpuBVStaging {
            perStepLogits.reserveCapacity(K)
        }
        let gpuLogits = gpuBVStaging
            ? try ensureGPUDraftLogitsBuffer(K: K)
            : nil

        // Gemma 4 multiplies embed_tokens output by sqrt(hidden_size)
        // before feeding the transformer. The target's normal forward
        // path applies this (TopLevelEmitter.swift:135-142,
        // PrefillEmitter.swift:171-179, gated on blockTopology==.gemma),
        // and HF's `target_model_input_embeddings` wrapper also
        // applies it before the assistant drafter consumes the embed
        // (Gemma4TextScaledWordEmbedding.forward). The bench builds
        // drafter inputs from raw `embedToken` bytes, so we must
        // apply the same scale here — without it, the drafter's
        // pre_projection sees an embed half that's ~39x too small,
        // and the drafter's argmax stops tracking the target. This
        // is the structural cause of universal α=0 in Phase 14
        // bench runs across diverse prompts.
        let embedScale = Float(hiddenSize).squareRoot()
        let profile = SmeltDecodeProfile.enabled
        if profile {
            // Drain prior-round records (refresh at boundary, etc.)
            // so the K-loop's breakdown reflects only drafter steps.
            _ = SmeltDecodeProfile.flush()
        }
        // `decodeUsAcc` (wrapper wall-clock around inner.decodeStep)
        // overlaps the encode+submit+gpuWait sum reported below — that
        // overlap is intentional, used as a cross-check that the inner
        // SmeltDecodeProfile captures sum to the outer measurement.
        // `prepUsAcc` and `readbackUsAcc` cover wrapper-only work (embed
        // scale + concat install; readSlotBytes) that the inner records
        // can't see, so they're not redundant.
        var prepUsAcc: Double = 0
        var decodeUsAcc: Double = 0
        var readbackUsAcc: Double = 0
        for step in 0 ..< K {
            let stepT0 = profile ? CFAbsoluteTimeGetCurrent() : 0
            let rawEmbed = try target.embedToken(currentToken)
            guard rawEmbed.count == rowBytes else {
                throw SmeltSpeculativeRuntimeError.targetDrafterMismatch(
                    "Gemma4Drafter: target embedToken returned "
                    + "\(rawEmbed.count) bytes, expected \(rowBytes) "
                    + "(hiddenSize=\(hiddenSize))"
                )
            }
            var scaledEmbed = Data(count: rawEmbed.count)
            scaledEmbed.withUnsafeMutableBytes { dst in
                rawEmbed.withUnsafeBytes { src in
                    let srcPtr = src.bindMemory(to: Float16.self)
                    let dstPtr = dst.bindMemory(to: Float16.self)
                    for i in 0 ..< srcPtr.count {
                        dstPtr[i] = Float16(Float(srcPtr[i]) * embedScale)
                    }
                }
            }

            var concat = Data()
            concat.reserveCapacity(2 * rowBytes)
            concat.append(scaledEmbed)
            concat.append(hiddenBytes)
            // Keep the successful install off Swift's `throws` ABI
            // boundary — release arm64 builds otherwise surfaced
            // stale swift_error register state after the raw copy,
            // causing the next `try` to branch to a corrupt error
            // pointer (EXC_BREAKPOINT in swift_getErrorValue). See
            // installSlotBytesFailure for details.
            if let error = inner.installSlotBytesFailure(
                at: hiddenBSlot, bytes: concat
            ) {
                throw error
            }

            // Position is frozen across K steps — the drafter never
            // advances position_ids during the K-step draft loop
            // (per spikes/mtp-drafter-bring-up.md).
            let stepT1 = profile ? CFAbsoluteTimeGetCurrent() : 0
            // Decode position is frozen across the K-loop (per drafter
            // contract), but the GPU sampler's RNG mixes seed+position —
            // so reuse of the same (seed, position) across K steps would
            // collide. Mix step into the seed via splitmix64's golden
            // constant so each step samples from an independent stream.
            let stepSelectionMode: SmeltSelectionMode
            if drafterGreedyEnabled {
                // Forces the LM-head argmax path; skips the GPU sampler.
                stepSelectionMode = .argmax
            } else if drafterGPUSample,
               case .temperature(let temp, let baseSeed) = selectionMode {
                let stepSeed = baseSeed
                    &+ UInt64(step) &* 0x9E37_79B9_7F4A_7C15
                stepSelectionMode = .temperature(temp, seed: stepSeed)
            } else if case let .filteredTemperature(
                temp, topK, topP, baseSeed
            ) = selectionMode {
                let stepSeed = baseSeed
                    &+ UInt64(step) &* 0x9E37_79B9_7F4A_7C15
                stepSelectionMode = .filteredTemperature(
                    temp, topK: topK, topP: topP, seed: stepSeed
                )
            } else {
                stepSelectionMode = .argmax
            }
            // decodeStep returns whatever was written to the argmax slot —
            // the LM-head argmax in greedy mode, or the GPU sampler's pick
            // when stepSelectionMode is temperature + sampler is wired.
            let gpuPickedToken = try inner.decodeStep(
                tokenId: 0, position: position,
                selectionMode: stepSelectionMode
            )
            let stepT2 = profile ? CFAbsoluteTimeGetCurrent() : 0
            let candidate: Int32
            if useArgmaxFastPath {
                candidate = gpuPickedToken
            } else if gpuBVStaging {
                candidate = try inner.sampleCurrentLogitsGPU(
                    position: position &+ Int32(step),
                    selectionMode: selectionMode
                )
                try inner.copyCurrentLogits(
                    to: gpuLogits!,
                    destinationOffset: step * target.vocabSize
                        * MemoryLayout<Float16>.stride
                )
            } else if drafterGPUSample {
                // BV still needs the full q distribution per step.
                candidate = gpuPickedToken
                perStepLogits.append(inner.allLogitsHalf())
            } else {
                let logits = inner.allLogitsHalf()
                candidate = logits.withUnsafeBufferPointer { buf in
                    SmeltLogitsSelector.select(
                        logits: buf,
                        position: position &+ Int32(step),
                        mode: selectionMode
                    )
                }
                perStepLogits.append(logits)
            }
            candidates.append(candidate)

            if step < K - 1 {
                // post_projection writes its [1, backbone] output to
                // hiddenB[0..rowBytes] at the tail of the dispatch
                // table (TopLevelEmitter.swift:727).
                hiddenBytes = try inner.readSlotBytes(
                    at: hiddenBSlot, offset: 0, length: rowBytes
                )
                currentToken = candidate
            }
            if profile {
                let stepT3 = CFAbsoluteTimeGetCurrent()
                prepUsAcc += (stepT1 - stepT0) * 1_000_000
                decodeUsAcc += (stepT2 - stepT1) * 1_000_000
                readbackUsAcc += (stepT3 - stepT2) * 1_000_000
            }
        }

        if profile {
            let records = SmeltDecodeProfile.flush()
            let encodeUs = records.reduce(0) { $0 + $1.encodeUs }
            let submitUs = records.reduce(0) { $0 + $1.submitUs }
            let gpuWaitUs = records.reduce(0) { $0 + $1.gpuWaitUs }
            fputs(
                String(
                    format:
                        "[drafter-profile] K=%d totals(µs): "
                        + "prep=%.1f decode=%.1f readback=%.1f "
                        + "| decode_breakdown(µs): encode=%.1f submit=%.1f gpuWait=%.1f\n",
                    K,
                    prepUsAcc, decodeUsAcc, readbackUsAcc,
                    encodeUs, submitUs, gpuWaitUs
                ),
                stderr
            )
        }

        let q: SmeltDrafterQ =
            (useArgmaxFastPath || gpuBVStaging) ? .none : .dense(perStepLogits)
        return try SmeltDraftBatch(candidates: candidates, q: q)
    }
}
