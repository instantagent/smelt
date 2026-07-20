// SmeltTargetSeededAuxiliaryDrafter — generic autoregressive composition for
// a compiled target-seeded MTP package. The inner package is an ordinary
// SmeltRuntime: install fused sources, run decodeStep, retain its own KV cache.

import Foundation
import SmeltSchema

public final class SmeltTargetSeededAuxiliaryDrafter: SmeltTargetSeededDrafter {
    private let inner: SmeltRuntime
    private let target: SmeltRuntime
    private let hiddenSize: Int
    private let rowBytes: Int
    private var preparedContextCapacity = 0

    private var primedPosition: Int32?
    private var primedSeed: Int32?
    private var primedPrediction: Int32?
    private var primedHidden: Data?

    private var roundStartHidden: Data?
    /// Hidden after consuming candidate input i at auxiliary cache position
    /// `roundStartPosition + i`.
    private var roundHiddens: [Data] = []

    public init(packagePath: String, target: SmeltRuntime) throws {
        self.inner = try SmeltRuntime(
            packagePath: packagePath,
            device: target.metalDevice
        )
        self.target = target

        let manifestURL = URL(fileURLWithPath: packagePath)
            .appendingPathComponent("manifest.json")
        let manifest = try SmeltManifest.decode(
            from: Data(contentsOf: manifestURL)
        )
        guard let fusion = manifest.config.inputFusion,
              fusion.sourceCount == 2,
              fusion.sourceWidth == target.hiddenSize,
              manifest.config.hiddenSize == target.hiddenSize
        else {
            throw SmeltSpeculativeRuntimeError.targetDrafterMismatch(
                "target-seeded auxiliary package must declare two target-hidden-width fusion sources"
            )
        }
        guard manifest.config.vocabSize == target.vocabSize else {
            throw SmeltSpeculativeRuntimeError.targetDrafterMismatch(
                "target-seeded auxiliary vocab \(manifest.config.vocabSize) != target vocab \(target.vocabSize)"
            )
        }
        self.hiddenSize = manifest.config.hiddenSize
        self.rowBytes = manifest.config.hiddenSize * MemoryLayout<Float16>.stride
    }

    public func draftStep(
        targetTaps: SmeltDrafterTaps,
        lastToken: Int32,
        position: Int32,
        K: Int,
        selectionMode: SmeltSelectionMode
    ) throws -> SmeltDraftBatch {
        throw SmeltSpeculativeRuntimeError.invalidConfiguration(
            "target-seeded auxiliary drafter must be invoked with the target next-token seed"
        )
    }

    public func draftStep(
        targetTaps: SmeltDrafterTaps,
        targetNextToken: Int32,
        lastToken: Int32,
        position: Int32,
        K: Int,
        selectionMode: SmeltSelectionMode
    ) throws -> SmeltDraftBatch {
        guard K > 0, selectionMode.usesArgmaxFastPath else {
            throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                "target-seeded auxiliary drafter currently requires K > 0 greedy selection"
            )
        }
        try ensurePrepared(targetTaps: targetTaps)
        guard targetTaps.hiddenSize == hiddenSize else {
            throw SmeltSpeculativeRuntimeError.targetDrafterMismatch(
                "target hidden width \(targetTaps.hiddenSize) != auxiliary width \(hiddenSize)"
            )
        }

        let targetHidden = Data(
            bytes: targetTaps.lastHiddenState.contents(),
            count: rowBytes
        )
        roundStartHidden = targetHidden
        roundHiddens.removeAll(keepingCapacity: true)

        var candidates = [targetNextToken]
        candidates.reserveCapacity(K)

        if K > 1,
           primedPosition == position,
           primedSeed == targetNextToken,
           let prediction = primedPrediction,
           let hidden = primedHidden {
            candidates.append(prediction)
            roundHiddens.append(hidden)
        } else if K > 1 {
            let step = try runModule(
                token: targetNextToken,
                hidden: targetHidden,
                position: position
            )
            candidates.append(step.token)
            roundHiddens.append(step.hidden)
        }

        while candidates.count < K {
            let inputIndex = roundHiddens.count
            let step = try runModule(
                token: candidates[inputIndex],
                hidden: roundHiddens[inputIndex - 1],
                position: position + Int32(inputIndex)
            )
            candidates.append(step.token)
            roundHiddens.append(step.hidden)
        }

        primedPosition = nil
        primedSeed = nil
        primedPrediction = nil
        primedHidden = nil
        return try SmeltDraftBatch(candidates: candidates, q: .none)
    }

    public func primeTargetContext(
        promptTokens: [Int32],
        targetNextToken: Int32,
        targetHiddenStates: [Data]
    ) throws {
        guard !promptTokens.isEmpty,
              promptTokens.count == targetHiddenStates.count
        else {
            throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                "target-seeded prompt priming needs one target hidden row per prompt token"
            )
        }
        let taps = try target.drafterTaps()
        try ensurePrepared(targetTaps: taps)
        inner.resetWorkingBuffers()

        var finalStep: (token: Int32, hidden: Data)?
        for index in promptTokens.indices {
            let hidden = targetHiddenStates[index]
            guard hidden.count == rowBytes else {
                throw SmeltSpeculativeRuntimeError.targetDrafterMismatch(
                    "target prompt hidden row \(index) has \(hidden.count) bytes; expected \(rowBytes)"
                )
            }
            let shiftedToken = index + 1 < promptTokens.count
                ? promptTokens[index + 1]
                : targetNextToken
            finalStep = try runModule(
                token: shiftedToken,
                hidden: hidden,
                position: Int32(index)
            )
        }
        primedPosition = Int32(promptTokens.count - 1)
        primedSeed = targetNextToken
        primedPrediction = finalStep?.token
        primedHidden = finalStep?.hidden
    }

    public func commitDraft(
        batch: SmeltDraftBatch,
        acceptedCount: Int,
        committedTokens: [Int32],
        startPosition: Int32
    ) throws {
        guard acceptedCount >= 0,
              acceptedCount <= batch.K,
              committedTokens.count == acceptedCount
                || committedTokens.count == acceptedCount + 1,
              var hidden = roundStartHidden
        else {
            throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                "target-seeded draft commit shape is inconsistent"
            )
        }

        // Preserve accepted speculative cache rows. If the final accepted
        // candidate was only predicted (not yet consumed), materialize it now.
        for inputIndex in 0 ..< acceptedCount {
            if inputIndex < roundHiddens.count {
                hidden = roundHiddens[inputIndex]
            } else {
                let step = try runModule(
                    token: batch.candidates[inputIndex],
                    hidden: hidden,
                    position: startPosition + Int32(inputIndex)
                )
                hidden = step.hidden
            }
        }

        if committedTokens.count == acceptedCount + 1,
           let replacementOrBonus = committedTokens.last {
            // Overwrite the rejected speculative row, or append the target
            // bonus, so the cache reaches the new committed boundary.
            _ = try runModule(
                token: replacementOrBonus,
                hidden: hidden,
                position: startPosition + Int32(acceptedCount)
            )
        }
        roundStartHidden = nil
        roundHiddens.removeAll(keepingCapacity: true)
    }

    public func resetSuffixCache(promptTokens: [Int32]) { }
    public func recordGeneratedTokens(_ tokens: [Int32]) { }

    private func ensurePrepared(targetTaps: SmeltDrafterTaps) throws {
        let capacity = targetTaps.attention.values.first?.contextCapacity
            ?? target.maxContextTokens
        if capacity != preparedContextCapacity {
            try inner.prepareForRequest(
                batchCapacity: 1,
                contextCapacity: capacity
            )
            preparedContextCapacity = capacity
        }
    }

    private func runModule(
        token: Int32,
        hidden: Data,
        position: Int32
    ) throws -> (token: Int32, hidden: Data) {
        let embedding = try target.embedToken(token)
        try inner.installInputFusionSources([embedding, hidden])
        let prediction = try inner.decodeStep(
            tokenId: 0,
            position: position,
            selectionMode: .argmax
        )
        return (prediction, try inner.currentHiddenStateBytes())
    }
}
