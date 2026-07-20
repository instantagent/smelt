// SmeltSuffixLookupDrafter — model-free prompt-lookup speculation.
//
// This is deliberately an ordinary SmeltPreflightDrafter brick. It proposes a block
// only when the live token suffix occurred earlier in the prompt/generated
// corpus with enough following tokens to fill the requested horizon. On a
// miss it returns an empty batch before the target exposes taps or enters a
// speculative state transaction.

import Foundation

public final class SmeltSuffixLookupDrafter: SmeltPreflightDrafter {
    public let maxNeedleLength: Int

    private var promptTokens: [Int32] = []
    private var generatedTokens: [Int32] = []
    /// Positions in the logical prompt+generation corpus, indexed by token.
    /// A lookup only visits occurrences of the current suffix tail instead of
    /// rescanning every token at every decode step.
    private var positionsByToken: [Int32: [Int]] = [:]

    public private(set) var hits = 0
    public private(set) var misses = 0
    public private(set) var lastNeedleLength: Int?
    public private(set) var lastCandidates: [Int32] = []

    public init(maxNeedleLength: Int = 4) {
        precondition(maxNeedleLength > 0, "suffix max needle length must be positive")
        self.maxNeedleLength = maxNeedleLength
    }

    public func resetSuffixCache(promptTokens: [Int32]) {
        self.promptTokens = promptTokens
        generatedTokens = []
        rebuildPositionIndex()
        hits = 0
        misses = 0
        lastNeedleLength = nil
        lastCandidates = []
    }

    public func recordGeneratedTokens(_ tokens: [Int32]) {
        if tokens.count >= generatedTokens.count,
           tokens.starts(with: generatedTokens) {
            let base = promptTokens.count
            for index in generatedTokens.count..<tokens.count {
                positionsByToken[tokens[index], default: []].append(base + index)
            }
            generatedTokens = tokens
        } else {
            // Branch/edit callers may replace or truncate the committed tail.
            // Rebuilding keeps the history brick correct without imposing an
            // append-only session contract on the generic runtime.
            generatedTokens = tokens
            rebuildPositionIndex()
        }
    }

    public func draftStep(
        targetTaps: SmeltDrafterTaps,
        lastToken: Int32,
        position: Int32,
        K: Int,
        selectionMode: SmeltSelectionMode
    ) throws -> SmeltDraftBatch {
        try preflightDraft(
            lastToken: lastToken,
            position: position,
            K: K,
            selectionMode: selectionMode
        )
    }

    public func preflightDraft(
        lastToken: Int32,
        position: Int32,
        K: Int,
        selectionMode: SmeltSelectionMode
    ) throws -> SmeltDraftBatch {
        if let batch = try draftIfMatch(
            lastToken: lastToken,
            position: position,
            K: K,
            selectionMode: selectionMode
        ) {
            hits += 1
            lastCandidates = batch.candidates
            return batch
        }
        misses += 1
        lastNeedleLength = nil
        lastCandidates = []
        return try SmeltDraftBatch(candidates: [], q: .none)
    }

    /// The optional form lets a neural drafter use prompt lookup as a cheap
    /// first choice and retain its own fallback on a miss.
    public func draftIfMatch(
        lastToken: Int32,
        position: Int32,
        K: Int,
        selectionMode: SmeltSelectionMode
    ) throws -> SmeltDraftBatch? {
        guard K > 0, let hit = lookup(lastToken: lastToken, K: K) else {
            return nil
        }
        lastNeedleLength = hit.needleLength
        if selectionMode.usesArgmaxFastPath {
            let candidates = hit.frequency.map(Self.argmaxTieBreakByTokenID)
            return try SmeltDraftBatch(candidates: candidates, q: .none)
        }

        let baseSeed: UInt64
        switch selectionMode {
        case .argmax:
            baseSeed = 0
        case .temperature(_, let seed),
             .filteredTemperature(_, _, _, let seed):
            baseSeed = seed
        }

        var candidates: [Int32] = []
        var sparseLogits: [SmeltSparseLogitRow] = []
        candidates.reserveCapacity(hit.frequency.count)
        sparseLogits.reserveCapacity(hit.frequency.count)
        for (k, stepFrequency) in hit.frequency.enumerated() {
            let stepSeed = baseSeed
                &+ UInt64(k) &* 0x9E37_79B9_7F4A_7C15
            var rng = SmeltDeterministicRng(
                masterSeed: stepSeed,
                domain: .drafter,
                position: position
            )
            candidates.append(
                Self.sample(
                    stepFrequency,
                    denominator: hit.denominator,
                    rng: &rng
                )
            )
            sparseLogits.append(
                SmeltSparseLogitRow(
                    histogram: stepFrequency,
                    denominator: hit.denominator
                )
            )
        }
        return try SmeltDraftBatch(
            candidates: candidates,
            q: .sparse(sparseLogits)
        )
    }

    private struct Hit {
        let frequency: [[Int32: Int]]
        let denominator: Int
        let needleLength: Int
    }

    private func lookup(lastToken: Int32, K: Int) -> Hit? {
        let totalLength = promptTokens.count + generatedTokens.count
        guard totalLength >= K + 1 else { return nil }

        let token: (Int) -> Int32 = { position in
            position < self.promptTokens.count
                ? self.promptTokens[position]
                : self.generatedTokens[position - self.promptTokens.count]
        }

        // A caller that forgot to publish committed tokens must not draft
        // from stale corpus state.
        guard token(totalLength - 1) == lastToken else { return nil }

        let upperNeedleLength = min(maxNeedleLength, totalLength - K)
        guard upperNeedleLength >= 1,
              let candidateEnds = positionsByToken[lastToken]
        else { return nil }
        for needleLength in stride(
            from: upperNeedleLength, through: 1, by: -1
        ) {
            let needleStart = totalLength - needleLength
            var matchPositions: [Int] = []
            for end in candidateEnds.reversed() {
                // The complete K-token continuation must already exist in the
                // corpus. This also excludes the live suffix occurrence.
                guard end + K < totalLength else { continue }
                let start = end - needleLength + 1
                guard start >= 0 else { continue }
                var matches = true
                for i in 0 ..< needleLength
                where token(start + i) != token(needleStart + i) {
                    matches = false
                    break
                }
                if matches { matchPositions.append(start) }
            }
            if matchPositions.isEmpty { continue }

            var frequency = Array(repeating: [Int32: Int](), count: K)
            for matchStart in matchPositions {
                let proposalStart = matchStart + needleLength
                for k in 0 ..< K {
                    frequency[k][token(proposalStart + k), default: 0] += 1
                }
            }
            return Hit(
                frequency: frequency,
                denominator: matchPositions.count,
                needleLength: needleLength
            )
        }
        return nil
    }

    private func rebuildPositionIndex() {
        positionsByToken.removeAll(keepingCapacity: true)
        positionsByToken.reserveCapacity(
            min(promptTokens.count + generatedTokens.count, 4_096)
        )
        for (position, token) in promptTokens.enumerated() {
            positionsByToken[token, default: []].append(position)
        }
        let base = promptTokens.count
        for (index, token) in generatedTokens.enumerated() {
            positionsByToken[token, default: []].append(base + index)
        }
    }

    private static func argmaxTieBreakByTokenID(
        _ histogram: [Int32: Int]
    ) -> Int32 {
        var bestCount = 0
        var bestToken = Int32.max
        for (token, count) in histogram {
            if count > bestCount || (count == bestCount && token < bestToken) {
                bestCount = count
                bestToken = token
            }
        }
        return bestToken
    }

    private static func sample(
        _ histogram: [Int32: Int],
        denominator: Int,
        rng: inout SmeltDeterministicRng
    ) -> Int32 {
        let entries = histogram.sorted { $0.key < $1.key }
        let uniform = Double(rng.next() >> 11) / Double(1 << 53)
        let target = uniform * Double(denominator)
        var cumulative = 0.0
        for (token, count) in entries {
            cumulative += Double(count)
            if cumulative > target { return token }
        }
        return entries.last!.key
    }
}
