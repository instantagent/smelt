import Foundation

/// Whether generation mirrors upstream masking quirks or enforces a decodable
/// four-skin-token-per-joint stream.
public enum SmeltRigPolicyMode: Sendable, Equatable {
    case sourceCompatible
    case validated
}

/// the rig model's fixed vocabulary contract from the pinned tokenizer/checkpoint.
public enum SmeltRigVocabulary {
    public static let coordinateRange = 0..<256
    public static let branch = 256
    public static let skeletonBOS = 257
    public static let skeletonEOS = 258
    public static let pad = 259
    public static let spring = 260
    public static let body = 261
    public static let hand = 262
    public static let noClass = 263
    public static let rigNetClass = 264
    public static let vroidClass = 265
    public static let articulationClass = 266
    public static let skinTokenBase = 267
    public static let skinTokenCount = 32_768
    public static let modelEOS = 33_035
    public static let vocabularySize = 33_036
    public static let skinTokenRange = skinTokenBase..<(skinTokenBase + skinTokenCount)
}

/// Detokenized joint hierarchy in the normalized `[-1, 1]` model space.
public struct SmeltSkeleton: Sendable, Equatable {
    public struct Joint: Sendable, Equatable {
        public let position: SIMD3<Float>
        public let parent: Int

        public init(position: SIMD3<Float>, parent: Int) {
            self.position = position
            self.parent = parent
        }
    }

    public let joints: [Joint]

    public init(joints: [Joint]) {
        self.joints = joints
    }
}

/// Pure-Swift rig model skeleton tokenizer/FSM.
public enum SmeltSkeletonTokenizer {
    private enum State {
        case expectBOS
        case classPartOrJoint
        case partOrJoint
        case joint2
        case joint3
        case branchPartOrJoint
        case joint1
        case strictBranchParent1
        case strictBranchParent2
        case strictBranchParent3
        case strictBranchJoint1
        case strictBranchJoint2
        case strictBranchJoint3
    }

    /// Returns the exact next-token skeleton grammar for a valid prefix.
    public static func allowedSkeletonTokens(
        after tokens: [Int],
        mode: SmeltRigPolicyMode
    ) throws -> [Int] {
        let state = try parseState(tokens, mode: mode)
        switch state {
        case .expectBOS:
            return [SmeltRigVocabulary.skeletonBOS]
        case .classPartOrJoint:
            return classTokens + partTokens + coordinateTokens
        case .partOrJoint:
            return partTokens + coordinateTokens + [SmeltRigVocabulary.skeletonEOS]
        case .joint2, .joint3, .joint1,
             .strictBranchParent1, .strictBranchParent2, .strictBranchParent3,
             .strictBranchJoint1, .strictBranchJoint2, .strictBranchJoint3:
            return coordinateTokens
        case .branchPartOrJoint:
            return coordinateTokens + partTokens + [
                SmeltRigVocabulary.branch,
                SmeltRigVocabulary.skeletonEOS,
            ]
        }
    }

    /// Counts complete bones using the pinned upstream state machine. This is
    /// deliberately separate from strict detokenization because the upstream
    /// logits mask opens after the first three coordinates of a branch record.
    public static func sourceCompatibleBoneCount(_ tokens: [Int]) throws -> Int {
        var state = State.expectBOS
        var isBranch = false
        var bones = 0
        for token in tokens {
            if token == SmeltRigVocabulary.skeletonEOS { break }
            switch state {
            case .expectBOS:
                guard token == SmeltRigVocabulary.skeletonBOS else {
                    throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
                }
                state = .classPartOrJoint
            case .classPartOrJoint:
                state = isCoordinate(token) ? .joint2
                    : (isClass(token) ? .partOrJoint : .joint1)
            case .partOrJoint:
                state = isCoordinate(token) ? .joint2 : .partOrJoint
            case .joint2:
                guard isCoordinate(token) else {
                    throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
                }
                state = .joint3
            case .joint3:
                guard isCoordinate(token) else {
                    throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
                }
                if !isBranch { bones += 1 }
                isBranch = false
                state = .branchPartOrJoint
            case .branchPartOrJoint:
                if token == SmeltRigVocabulary.branch {
                    state = .joint1
                    isBranch = true
                } else if isCoordinate(token) {
                    state = .joint2
                } else {
                    state = .joint1
                }
            case .joint1:
                guard isCoordinate(token) else {
                    throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
                }
                state = .joint2
            case .strictBranchParent1, .strictBranchParent2, .strictBranchParent3,
                 .strictBranchJoint1, .strictBranchJoint2, .strictBranchJoint3:
                throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
            }
        }
        return bones
    }

    /// Strictly detokenizes complete skeleton records and reconstructs parents
    /// with the same reverse nearest-parent rule as upstream.
    public static func detokenize(_ tokens: [Int]) throws -> SmeltSkeleton {
        guard tokens.first == SmeltRigVocabulary.skeletonBOS,
              tokens.last == SmeltRigVocabulary.skeletonEOS
        else {
            throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
        }
        let body = tokens.dropFirst().dropLast()
        var cursor = body.startIndex
        var positions: [SIMD3<Float>] = []
        var parentPositions: [SIMD3<Float>] = []
        var lastJoint: SIMD3<Float>?
        var branch = false
        while cursor < body.endIndex {
            let token = body[cursor]
            if isCoordinate(token) {
                let parentPosition: SIMD3<Float>
                let position: SIMD3<Float>
                if branch {
                    parentPosition = try coordinates(body, cursor: &cursor)
                    position = try coordinates(body, cursor: &cursor)
                } else {
                    position = try coordinates(body, cursor: &cursor)
                    parentPosition = lastJoint ?? position
                }
                positions.append(position)
                parentPositions.append(parentPosition)
                lastJoint = position
                branch = false
            } else if token == SmeltRigVocabulary.branch {
                branch = true
                lastJoint = nil
                cursor = body.index(after: cursor)
            } else if isPart(token) || isClass(token) {
                cursor = body.index(after: cursor)
            } else {
                throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
            }
        }
        guard !branch, !positions.isEmpty else {
            throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
        }
        var joints: [SmeltSkeleton.Joint] = []
        joints.reserveCapacity(positions.count)
        for index in positions.indices {
            if index == 0 {
                joints.append(.init(position: positions[index], parent: -1))
                continue
            }
            var best = -1
            var bestDistance = Float.greatestFiniteMagnitude
            for candidate in (0..<index).reversed() {
                let delta = positions[candidate] - parentPositions[index]
                let distance = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z
                if distance < bestDistance {
                    best = candidate
                    bestDistance = distance
                }
            }
            joints.append(.init(position: positions[index], parent: best))
        }
        return SmeltSkeleton(joints: joints)
    }

    private static func parseState(
        _ tokens: [Int],
        mode: SmeltRigPolicyMode
    ) throws -> State {
        var state = State.expectBOS
        for token in tokens {
            switch state {
            case .expectBOS:
                guard token == SmeltRigVocabulary.skeletonBOS else {
                    throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
                }
                state = .classPartOrJoint
            case .classPartOrJoint:
                if isCoordinate(token) { state = .joint2 }
                else if isClass(token) { state = .partOrJoint }
                else if isPart(token) { state = .joint1 }
                else { throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens) }
            case .partOrJoint:
                if isCoordinate(token) { state = .joint2 }
                else if isPart(token) { state = .partOrJoint }
                else { throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens) }
            case .joint1:
                guard isCoordinate(token) else {
                    throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
                }
                state = .joint2
            case .joint2:
                guard isCoordinate(token) else {
                    throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
                }
                state = .joint3
            case .joint3:
                guard isCoordinate(token) else {
                    throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
                }
                state = .branchPartOrJoint
            case .branchPartOrJoint:
                if token == SmeltRigVocabulary.branch {
                    state = mode == .sourceCompatible ? .joint1 : .strictBranchParent1
                } else if isCoordinate(token) {
                    state = .joint2
                } else if isPart(token) {
                    state = .joint1
                } else {
                    throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
                }
            case .strictBranchParent1:
                guard isCoordinate(token) else {
                    throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
                }
                state = .strictBranchParent2
            case .strictBranchParent2:
                guard isCoordinate(token) else {
                    throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
                }
                state = .strictBranchParent3
            case .strictBranchParent3:
                guard isCoordinate(token) else {
                    throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
                }
                state = .strictBranchJoint1
            case .strictBranchJoint1:
                guard isCoordinate(token) else {
                    throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
                }
                state = .strictBranchJoint2
            case .strictBranchJoint2:
                guard isCoordinate(token) else {
                    throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
                }
                state = .strictBranchJoint3
            case .strictBranchJoint3:
                guard isCoordinate(token) else {
                    throw SmeltRigGenerationError.invalidSkeletonPrefix(tokens)
                }
                state = .branchPartOrJoint
            }
        }
        return state
    }

    private static func coordinates(
        _ tokens: ArraySlice<Int>,
        cursor: inout ArraySlice<Int>.Index
    ) throws -> SIMD3<Float> {
        var values: [Float] = []
        values.reserveCapacity(3)
        for _ in 0..<3 {
            guard cursor < tokens.endIndex, isCoordinate(tokens[cursor]) else {
                throw SmeltRigGenerationError.incompleteCoordinate
            }
            values.append((Float(tokens[cursor]) + 0.5) / 128 - 1)
            cursor = tokens.index(after: cursor)
        }
        return SIMD3(values[0], values[1], values[2])
    }

    private static let coordinateTokens = Array(SmeltRigVocabulary.coordinateRange)
    private static let classTokens = Array(
        SmeltRigVocabulary.noClass...SmeltRigVocabulary.articulationClass
    )
    private static let partTokens = Array(
        SmeltRigVocabulary.spring...SmeltRigVocabulary.hand
    )

    private static func isCoordinate(_ token: Int) -> Bool {
        SmeltRigVocabulary.coordinateRange.contains(token)
    }

    private static func isClass(_ token: Int) -> Bool {
        (SmeltRigVocabulary.noClass...SmeltRigVocabulary.articulationClass)
            .contains(token)
    }

    private static func isPart(_ token: Int) -> Bool {
        (SmeltRigVocabulary.spring...SmeltRigVocabulary.hand).contains(token)
    }
}

/// Token mask and sampling policy for rig model generation.
public enum SmeltRigGenerationPolicy {
    public struct Sample: Sendable, Equatable {
        public let token: Int
        public let logProbability: Double

        public init(token: Int, logProbability: Double) {
            self.token = token
            self.logProbability = logProbability
        }
    }

    /// Returns legal next tokens across skeleton, skin, and final-EOS phases.
    public static func allowedNextTokens(
        sequence: [Int],
        mode: SmeltRigPolicyMode = .validated,
        remainingTokenBudget: Int? = nil
    ) throws -> [Int] {
        if sequence.contains(SmeltRigVocabulary.modelEOS) { return [] }
        guard let switchIndex = sequence.firstIndex(
            of: SmeltRigVocabulary.skeletonEOS
        ) else {
            let allowed = try SmeltSkeletonTokenizer.allowedSkeletonTokens(
                after: sequence,
                mode: mode
            )
            return try completionBudgetConstrainedTokens(
                allowed,
                sequence: sequence,
                mode: mode,
                remainingTokenBudget: remainingTokenBudget
            )
        }
        let skeleton = Array(sequence[...switchIndex])
        let jointCount: Int
        switch mode {
        case .sourceCompatible:
            jointCount = try SmeltSkeletonTokenizer.sourceCompatibleBoneCount(skeleton)
        case .validated:
            jointCount = try SmeltSkeletonTokenizer.detokenize(skeleton).joints.count
        }
        guard jointCount > 0 else {
            throw SmeltRigGenerationError.invalidJointCount(jointCount)
        }
        let tokensAfterSwitch = sequence.count - switchIndex - 1
        switch mode {
        case .sourceCompatible:
            // Mirrors the pinned processor literally: mask[skeletonEOS...] and
            // `(length - switchIndex) == joints * 4` (the switch is included).
            if tokensAfterSwitch + 1 == jointCount * 4 {
                return [SmeltRigVocabulary.modelEOS]
            }
            return Array(
                SmeltRigVocabulary.skeletonEOS..<SmeltRigVocabulary.modelEOS
            )
        case .validated:
            let required = jointCount * 4
            if tokensAfterSwitch == required {
                return [SmeltRigVocabulary.modelEOS]
            }
            guard tokensAfterSwitch < required else {
                throw SmeltRigGenerationError.tooManySkinTokens(
                    expected: required,
                    got: tokensAfterSwitch
                )
            }
            return Array(SmeltRigVocabulary.skinTokenRange)
        }
    }

    private static func completionBudgetConstrainedTokens(
        _ allowed: [Int],
        sequence: [Int],
        mode: SmeltRigPolicyMode,
        remainingTokenBudget: Int?
    ) throws -> [Int] {
        guard mode == .validated,
              let remainingTokenBudget,
              allowed.contains(SmeltRigVocabulary.skeletonEOS)
        else {
            return allowed
        }
        guard remainingTokenBudget > 0 else { return [] }

        // At a skeleton record boundary, keep only choices that leave enough
        // context for the shortest legal path through the new record, the
        // skeleton terminator, four skin codes per joint, and model EOS. This
        // converts the model's finite context into a grammar invariant instead
        // of letting a valid-but-unfinishable skeleton consume the last row.
        let joints = try SmeltSkeletonTokenizer.sourceCompatibleBoneCount(sequence)
        let closeNow = joints * 4 + 2
        return allowed.filter { token in
            switch token {
            case SmeltRigVocabulary.skeletonEOS:
                return joints > 0 && remainingTokenBudget >= closeNow
            case SmeltRigVocabulary.branch:
                return remainingTokenBudget >= closeNow + 11
            case SmeltRigVocabulary.spring...SmeltRigVocabulary.hand:
                return remainingTokenBudget >= closeNow + 8
            case SmeltRigVocabulary.coordinateRange:
                return remainingTokenBudget >= closeNow + 7
            default:
                return false
            }
        }
    }

    /// Applies repetition penalty, grammar masking, temperature, top-k, and
    /// nucleus filtering, then draws deterministically from `uniform`.
    public static func sample(
        logits: [Float],
        history: [Int],
        allowedTokens: [Int],
        repetitionPenalty: Float,
        temperature: Float,
        topK: Int,
        topP: Float,
        uniform: Float
    ) throws -> Sample {
        guard uniform >= 0, uniform < 1 else {
            throw SmeltRigGenerationError.invalidSamplingPolicy
        }
        let distribution = try filteredDistribution(
            logits: logits,
            history: history,
            allowedTokens: allowedTokens,
            repetitionPenalty: repetitionPenalty,
            temperature: temperature,
            topK: topK,
            topP: topP
        )
        let target = Double(uniform)
        var running = 0.0
        for item in distribution {
            running += Foundation.exp(item.logProbability)
            if running >= target {
                return item
            }
        }
        return distribution[distribution.count - 1]
    }

    static func filteredDistribution(
        logits: [Float],
        history: [Int],
        allowedTokens: [Int],
        repetitionPenalty: Float,
        temperature: Float,
        topK: Int,
        topP: Float
    ) throws -> [Sample] {
        guard logits.count == SmeltRigVocabulary.vocabularySize else {
            throw SmeltRigGenerationError.invalidLogitCount(logits.count)
        }
        guard repetitionPenalty.isFinite, repetitionPenalty > 0,
              temperature.isFinite, temperature > 0,
              topK > 0,
              topP.isFinite, topP > 0, topP <= 1
        else {
            throw SmeltRigGenerationError.invalidSamplingPolicy
        }
        let allowed = Set(allowedTokens)
        guard !allowed.isEmpty else {
            throw SmeltRigGenerationError.noAllowedTokens
        }
        guard allowed.allSatisfy({ $0 >= 0 && $0 < logits.count }) else {
            throw SmeltRigGenerationError.invalidAllowedToken
        }
        var processed = logits
        for token in Set(history)
        where token >= 0 && token < processed.count {
            processed[token] = processed[token] < 0
                ? processed[token] * repetitionPenalty
                : processed[token] / repetitionPenalty
        }
        var ranked = allowed.map { token -> (token: Int, logit: Float) in
            (token, processed[token])
        }.filter { $0.logit.isFinite }
        ranked.sort {
            $0.logit == $1.logit ? $0.token < $1.token : $0.logit > $1.logit
        }
        guard !ranked.isEmpty else {
            throw SmeltRigGenerationError.noFiniteAllowedLogits
        }
        let k = min(topK, ranked.count)
        let threshold = ranked[k - 1].logit
        ranked = ranked.filter { $0.logit >= threshold }
        let maximum = Double(ranked[0].logit)
        var weighted = ranked.map { item in
            (
                token: item.token,
                logit: item.logit,
                weight: Foundation.exp((Double(item.logit) - maximum) / Double(temperature))
            )
        }
        let total = weighted.reduce(0.0) { $0 + $1.weight }
        var cumulative = 0.0
        var retained: [(token: Int, logit: Float, weight: Double)] = []
        for item in weighted {
            retained.append(item)
            cumulative += item.weight / total
            if cumulative >= Double(topP) { break }
        }
        weighted = retained
        let retainedTotal = weighted.reduce(0.0) { $0 + $1.weight }
        return weighted.map {
            Sample(
                token: $0.token,
                logProbability: Foundation.log($0.weight / retainedTotal)
            )
        }
    }
}

/// Runtime decode strategy for the correctness-first rig model generator.
public enum SmeltRigDecodeMode: Sendable, Equatable {
    /// Selects the lowest-ID argmax after all authored processors.
    case greedy
    /// Draws one deterministic path from the authored filtered distribution.
    case sampled(seed: UInt64)
    /// Runs deterministic seeded beam sampling while snapshotting each beam's
    /// complete GPU KV state.
    case beamSampled(seed: UInt64, width: Int)
}

/// rig model autoregressive generation controls.
public struct SmeltRigGenerationConfiguration: Sendable, Equatable {
    public let policyMode: SmeltRigPolicyMode
    public let decodeMode: SmeltRigDecodeMode
    /// Maximum autoregressive tokens, or `nil` to use all context remaining
    /// after the mesh and caller-provided token prefix.
    public let maximumGeneratedTokens: Int?
    public let repetitionPenalty: Float
    public let temperature: Float
    public let topK: Int
    public let topP: Float

    public init(
        policyMode: SmeltRigPolicyMode = .validated,
        decodeMode: SmeltRigDecodeMode = .greedy,
        maximumGeneratedTokens: Int? = nil,
        repetitionPenalty: Float = 2,
        temperature: Float = 1,
        topK: Int = 5,
        topP: Float = 0.95
    ) {
        self.policyMode = policyMode
        self.decodeMode = decodeMode
        self.maximumGeneratedTokens = maximumGeneratedTokens
        self.repetitionPenalty = repetitionPenalty
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
    }
}

/// A complete, strictly decodable rig model token stream.
public struct SmeltRigGenerationResult: Sendable, Equatable {
    public let tokenIDs: [Int]
    public let generatedTokenIDs: [Int]
    public let skeleton: SmeltSkeleton
    /// Four zero-based FSQ codebook indices for each skeleton joint.
    public let skinCodeIndices: [[Int]]

    public init(
        tokenIDs: [Int],
        generatedTokenIDs: [Int],
        skeleton: SmeltSkeleton,
        skinCodeIndices: [[Int]]
    ) {
        self.tokenIDs = tokenIDs
        self.generatedTokenIDs = generatedTokenIDs
        self.skeleton = skeleton
        self.skinCodeIndices = skinCodeIndices
    }
}

/// Raw rig model generation output, including source-compatible malformed
/// streams retained for parity diagnostics.
public struct SmeltRigTokenSequence: Sendable, Equatable {
    public let tokenIDs: [Int]
    public let generatedTokenIDs: [Int]

    public init(tokenIDs: [Int], generatedTokenIDs: [Int]) {
        self.tokenIDs = tokenIDs
        self.generatedTokenIDs = generatedTokenIDs
    }
}

/// Pure-Smelt rig model autoregressive skeleton generation over a 512-row Michelangelo prefix.
public final class SmeltSkeletonGenerator {
    private static let meshPrefixRows = 512

    public let languageModel: SmeltSkeletonLanguageRuntime

    public init(languageModel: SmeltSkeletonLanguageRuntime) {
        self.languageModel = languageModel
    }

    /// Generates one rig. `startTokenIDs` may be a class head or a complete
    /// teacher-provided skeleton; it must begin with skeleton BOS.
    public func generate(
        meshEmbeddings: [Float],
        startTokenIDs: [Int] = [
            SmeltRigVocabulary.skeletonBOS,
            SmeltRigVocabulary.noClass,
        ],
        configuration: SmeltRigGenerationConfiguration = .init()
    ) throws -> SmeltRigGenerationResult {
        let tokens = try generateTokenSequence(
            meshEmbeddings: meshEmbeddings,
            startTokenIDs: startTokenIDs,
            configuration: configuration
        )
        return try Self.makeResult(
            sequence: tokens.tokenIDs,
            generated: tokens.generatedTokenIDs
        )
    }

    /// Generates raw IDs without requiring the result to be decodable. This is
    /// the parity surface for the pinned source processor's known skin-count bug.
    public func generateTokenSequence(
        meshEmbeddings: [Float],
        startTokenIDs: [Int] = [
            SmeltRigVocabulary.skeletonBOS,
            SmeltRigVocabulary.noClass,
        ],
        configuration: SmeltRigGenerationConfiguration = .init()
    ) throws -> SmeltRigTokenSequence {
        let hiddenSize = languageModel.artifact.manifest.configuration.languageHiddenSize
        guard meshEmbeddings.count == Self.meshPrefixRows * hiddenSize else {
            throw SmeltRigGenerationError.invalidMeshEmbeddingCount(
                expected: Self.meshPrefixRows * hiddenSize,
                got: meshEmbeddings.count
            )
        }
        guard startTokenIDs.first == SmeltRigVocabulary.skeletonBOS else {
            throw SmeltRigGenerationError.invalidSkeletonPrefix(startTokenIDs)
        }
        if let configuredMaximum = configuration.maximumGeneratedTokens,
           configuredMaximum <= 0
        {
            throw SmeltRigGenerationError.invalidMaximumGeneratedTokens(
                configuredMaximum
            )
        }
        if case .beamSampled(_, let width) = configuration.decodeMode,
           width <= 0
        {
            throw SmeltRigGenerationError.invalidBeamWidth(width)
        }
        let prefixLength = Self.meshPrefixRows + startTokenIDs.count
        let maximumPositions =
            languageModel.artifact.manifest.configuration.languageMaximumPositions
        let availableGeneratedTokens = maximumPositions - prefixLength
        let maximumGeneratedTokens =
            configuration.maximumGeneratedTokens ?? availableGeneratedTokens
        guard prefixLength < maximumPositions,
              maximumGeneratedTokens <= availableGeneratedTokens
        else {
            throw SmeltRigGenerationError.contextLimitExceeded(
                requested: prefixLength + maximumGeneratedTokens,
                maximum: maximumPositions
            )
        }

        // Reserve the whole planned sequence before prefill. Growing the KV
        // cache one row at a time during autoregressive decode repeatedly
        // reallocates every context-scoped Metal buffer while preserving its
        // prefix. Besides being quadratic work, those retired allocations can
        // remain resident until the GPU drains and exhaust unified memory on a
        // normal full-length generation.
        try languageModel.trunk.ensureContextCapacity(
            prefixLength + maximumGeneratedTokens
        )

        var embeddings = meshEmbeddings
        embeddings.append(contentsOf: try languageModel.embeddings(tokenIDs: startTokenIDs))
        var logits = try languageModel.prefill(
            embeddings: embeddings,
            sequenceLength: prefixLength
        ).finalLogits
        if case .beamSampled(let seed, let width) = configuration.decodeMode {
            return try generateBeamSampled(
                initialLogits: logits,
                prefixLength: prefixLength,
                startTokenIDs: startTokenIDs,
                seed: seed,
                width: width,
                maximumGeneratedTokens: maximumGeneratedTokens,
                configuration: configuration
            )
        }
        var sequence = startTokenIDs
        var generated: [Int] = []
        generated.reserveCapacity(maximumGeneratedTokens)

        for step in 0..<maximumGeneratedTokens {
            let allowed = try SmeltRigGenerationPolicy.allowedNextTokens(
                sequence: sequence,
                mode: configuration.policyMode,
                remainingTokenBudget: maximumGeneratedTokens - step
            )
            let token = try select(
                logits: logits,
                generatedHistory: generated,
                allowedTokens: allowed,
                step: step,
                configuration: configuration
            )
            sequence.append(token)
            generated.append(token)
            if token == SmeltRigVocabulary.modelEOS {
                return SmeltRigTokenSequence(
                    tokenIDs: sequence,
                    generatedTokenIDs: generated
                )
            }
            let position = Self.meshPrefixRows + sequence.count - 1
            logits = try languageModel.decodeTeacherForced(
                tokenID: token,
                position: position
            ).logits
        }
        throw SmeltRigGenerationError.generationDidNotTerminate(
            maximumGeneratedTokens
        )
    }

    private func select(
        logits: [Float],
        generatedHistory: [Int],
        allowedTokens: [Int],
        step: Int,
        configuration: SmeltRigGenerationConfiguration
    ) throws -> Int {
        switch configuration.decodeMode {
        case .greedy:
            return try SmeltRigGenerationPolicy.sample(
                logits: logits,
                history: generatedHistory,
                allowedTokens: allowedTokens,
                repetitionPenalty: configuration.repetitionPenalty,
                temperature: 1,
                topK: 1,
                topP: 1,
                uniform: 0
            ).token
        case .sampled(let seed):
            return try SmeltRigGenerationPolicy.sample(
                logits: logits,
                history: generatedHistory,
                allowedTokens: allowedTokens,
                repetitionPenalty: configuration.repetitionPenalty,
                temperature: configuration.temperature,
                topK: configuration.topK,
                topP: configuration.topP,
                uniform: SmeltSamplingRandom.uniform(
                    seed: seed,
                    step: step,
                    stream: 0
                )
            ).token
        case .beamSampled:
            preconditionFailure("beam sampling is handled by generateBeamSampled")
        }
    }

    private struct Beam {
        let sequence: [Int]
        let generated: [Int]
        let logits: [Float]
        let score: Double
        let snapshot: SmeltDevicePromptSnapshot
    }

    private struct BeamCandidate {
        let parent: Int
        let token: Int
        let score: Double
    }

    private struct CompletedBeam {
        let sequence: [Int]
        let generated: [Int]
        let score: Double
    }

    private func generateBeamSampled(
        initialLogits: [Float],
        prefixLength: Int,
        startTokenIDs: [Int],
        seed: UInt64,
        width: Int,
        maximumGeneratedTokens: Int,
        configuration: SmeltRigGenerationConfiguration
    ) throws -> SmeltRigTokenSequence {
        let initialSnapshot = try languageModel.trunk.captureDevicePromptSnapshot(
            capturedLength: prefixLength
        )
        var beams = [
            Beam(
                sequence: startTokenIDs,
                generated: [],
                logits: initialLogits,
                score: 0,
                snapshot: initialSnapshot
            ),
        ]
        var completed: [CompletedBeam] = []

        for step in 0..<maximumGeneratedTokens {
            var candidates: [BeamCandidate] = []
            for (parent, beam) in beams.enumerated() {
                let allowed = try SmeltRigGenerationPolicy.allowedNextTokens(
                    sequence: beam.sequence,
                    mode: configuration.policyMode,
                    remainingTokenBudget: maximumGeneratedTokens - step
                )
                let distribution = try SmeltRigGenerationPolicy.filteredDistribution(
                    logits: beam.logits,
                    history: beam.generated,
                    allowedTokens: allowed,
                    repetitionPenalty: configuration.repetitionPenalty,
                    temperature: configuration.temperature,
                    topK: configuration.topK,
                    topP: configuration.topP
                )
                candidates.append(contentsOf: distribution.map {
                    BeamCandidate(
                        parent: parent,
                        token: $0.token,
                        score: beam.score + $0.logProbability
                    )
                })
            }
            guard !candidates.isEmpty else {
                throw SmeltRigGenerationError.noAllowedTokens
            }
            let sampled = sampleWithoutReplacement(
                candidates,
                count: min(width * 2, candidates.count),
                seed: seed,
                step: step
            ).sorted {
                if $0.score != $1.score { return $0.score > $1.score }
                if $0.token != $1.token { return $0.token < $1.token }
                return $0.parent < $1.parent
            }

            var next: [Beam] = []
            next.reserveCapacity(width)
            for (rank, candidate) in sampled.enumerated() {
                let parent = beams[candidate.parent]
                let sequence = parent.sequence + [candidate.token]
                let generated = parent.generated + [candidate.token]
                if candidate.token == SmeltRigVocabulary.modelEOS {
                    if rank < width {
                        completed.append(
                            CompletedBeam(
                                sequence: sequence,
                                generated: generated,
                                score: candidate.score
                            )
                        )
                    }
                    continue
                }
                guard next.count < width else { continue }
                try languageModel.trunk.restoreDevicePromptSnapshot(parent.snapshot)
                let position = prefixLength + parent.generated.count
                let logits = try languageModel.decodeTeacherForced(
                    tokenID: candidate.token,
                    position: position
                ).logits
                let snapshot = try languageModel.trunk.captureDevicePromptSnapshot(
                    capturedLength: position + 1
                )
                next.append(
                    Beam(
                        sequence: sequence,
                        generated: generated,
                        logits: logits,
                        score: candidate.score,
                        snapshot: snapshot
                    )
                )
            }
            if next.isEmpty {
                if let best = completed.max(by: { $0.score < $1.score }) {
                    return SmeltRigTokenSequence(
                        tokenIDs: best.sequence,
                        generatedTokenIDs: best.generated
                    )
                }
                throw SmeltRigGenerationError.noAllowedTokens
            }
            beams = next
        }
        if let best = completed.max(by: { $0.score < $1.score }) {
            return SmeltRigTokenSequence(
                tokenIDs: best.sequence,
                generatedTokenIDs: best.generated
            )
        }
        throw SmeltRigGenerationError.generationDidNotTerminate(
            maximumGeneratedTokens
        )
    }

    private func sampleWithoutReplacement(
        _ candidates: [BeamCandidate],
        count: Int,
        seed: UInt64,
        step: Int
    ) -> [BeamCandidate] {
        var available = candidates
        var sampled: [BeamCandidate] = []
        sampled.reserveCapacity(count)
        for draw in 0..<count {
            let maximum = available.map(\.score).max() ?? 0
            let weights = available.map { Foundation.exp($0.score - maximum) }
            let total = weights.reduce(0, +)
            let uniform = Double(
                SmeltSamplingRandom.uniform(
                    seed: seed,
                    step: step * max(count, 1) + draw,
                    stream: 0
                )
            )
            let target = uniform * total
            var running = 0.0
            var selected = available.count - 1
            for index in available.indices {
                running += weights[index]
                if running >= target {
                    selected = index
                    break
                }
            }
            sampled.append(available.remove(at: selected))
        }
        return sampled
    }

    private static func makeResult(
        sequence: [Int],
        generated: [Int]
    ) throws -> SmeltRigGenerationResult {
        guard sequence.last == SmeltRigVocabulary.modelEOS,
              let switchIndex = sequence.firstIndex(
                  of: SmeltRigVocabulary.skeletonEOS
              )
        else {
            throw SmeltRigGenerationError.invalidCompletedSequence(sequence)
        }
        let skeletonTokens = Array(sequence[...switchIndex])
        let skeleton = try SmeltSkeletonTokenizer.detokenize(skeletonTokens)
        let skinTokens = sequence[(switchIndex + 1)..<(sequence.count - 1)]
        guard skinTokens.count == skeleton.joints.count * 4,
              skinTokens.allSatisfy(SmeltRigVocabulary.skinTokenRange.contains)
        else {
            throw SmeltRigGenerationError.invalidCompletedSequence(sequence)
        }
        var codes: [[Int]] = []
        codes.reserveCapacity(skeleton.joints.count)
        for joint in 0..<skeleton.joints.count {
            let start = skinTokens.index(skinTokens.startIndex, offsetBy: joint * 4)
            let end = skinTokens.index(start, offsetBy: 4)
            codes.append(
                skinTokens[start..<end].map {
                    $0 - SmeltRigVocabulary.skinTokenBase
                }
            )
        }
        return SmeltRigGenerationResult(
            tokenIDs: sequence,
            generatedTokenIDs: generated,
            skeleton: skeleton,
            skinCodeIndices: codes
        )
    }
}

public enum SmeltRigGenerationError: Error, Equatable {
    case invalidSkeletonPrefix([Int])
    case incompleteCoordinate
    case invalidJointCount(Int)
    case tooManySkinTokens(expected: Int, got: Int)
    case invalidLogitCount(Int)
    case invalidSamplingPolicy
    case noAllowedTokens
    case invalidAllowedToken
    case noFiniteAllowedLogits
    case invalidMeshEmbeddingCount(expected: Int, got: Int)
    case invalidMaximumGeneratedTokens(Int)
    case invalidBeamWidth(Int)
    case contextLimitExceeded(requested: Int, maximum: Int)
    case generationDidNotTerminate(Int)
    case invalidCompletedSequence([Int])
}
