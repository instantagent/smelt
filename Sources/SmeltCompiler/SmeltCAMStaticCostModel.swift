import Foundation
import SmeltSchema

/// Storage choices used for architecture-level feasibility accounting. These
/// are logical checkpoint formats, not promises about a particular kernel.
public enum SmeltCAMWeightStorageProfile: String, Codable, CaseIterable, Sendable {
    case bf16
    case affineU4G64 = "affine-u4-g64"
    case nvfp4

    fileprivate var fallbackBytesPerParameter: Double {
        switch self {
        case .bf16:
            return 2
        case .affineU4G64:
            // Four-bit codes plus one FP16 scale and one FP16 bias per group.
            return 0.5 + (4.0 / 64.0)
        case .nvfp4:
            // NVFP4 block metadata varies by checkpoint. Callers should pass
            // an exact index total; this fallback remains explicitly estimated.
            return 0.625
        }
    }
}

public struct SmeltCAMStaticCostScenario: Codable, Equatable, Sendable {
    public let storage: SmeltCAMWeightStorageProfile
    public let contextLength: Int
    public let stateBytesPerElement: Int
    public let exactCheckpointBytes: UInt64?
    public let memoryLimitBytes: UInt64?
    public let sustainedMemoryBandwidthBytesPerSecond: Double?

    public init(
        storage: SmeltCAMWeightStorageProfile,
        contextLength: Int,
        stateBytesPerElement: Int = 2,
        exactCheckpointBytes: UInt64? = nil,
        memoryLimitBytes: UInt64? = nil,
        sustainedMemoryBandwidthBytesPerSecond: Double? = nil
    ) {
        self.storage = storage
        self.contextLength = max(contextLength, 1)
        self.stateBytesPerElement = max(stateBytesPerElement, 1)
        self.exactCheckpointBytes = exactCheckpointBytes
        self.memoryLimitBytes = memoryLimitBytes
        self.sustainedMemoryBandwidthBytesPerSecond = sustainedMemoryBandwidthBytesPerSecond
    }
}

public struct SmeltCAMStaticAttentionRoleCost: Codable, Equatable, Sendable {
    public let role: String
    public let layerCount: Int
    public let attendedTokensPerLayer: Int
    public let residentParametersPerLayer: UInt64
    public let kvStateBytes: UInt64
    public let oneTokenKVReadBytes: UInt64
    public let oneTokenKVWriteBytes: UInt64
    public let oneTokenAttentionArithmeticOperations: UInt64
}

public struct SmeltCAMStaticCostReport: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let moduleID: String
    public let moduleSemanticSHA256: String
    public let blockID: String
    public let phase: String
    public let storage: SmeltCAMWeightStorageProfile
    public let contextLength: Int
    public let bytesPerStoredParameter: Double
    public let exactCheckpointBytes: UInt64?
    public let declaredResidentParameters: UInt64?
    public let declaredActiveParameters: UInt64?
    public let derivedResidentParameters: UInt64
    public let derivedActiveParameters: UInt64
    public let residentWeightBytes: UInt64
    public let oneTokenActiveWeightReadBytes: UInt64
    public let kvStateBytes: UInt64
    public let shortConvolutionStateBytes: UInt64
    public let totalPersistentBytes: UInt64
    public let memoryLimitBytes: UInt64?
    public let fitsMemoryLimit: Bool?
    public let oneTokenKVReadBytes: UInt64
    public let oneTokenKVWriteBytes: UInt64
    public let oneTokenArithmeticOperations: UInt64
    public let bandwidthLowerBoundSecondsPerToken: Double?
    public let bandwidthUpperBoundTokensPerSecond: Double?
    public let attentionRoles: [SmeltCAMStaticAttentionRoleCost]
    public let assumptions: [String]

    public func encodeJSON(prettyPrinted: Bool = true) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = prettyPrinted
            ? [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            : [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }
}

public enum SmeltCAMStaticCostModelError: Error, CustomStringConvertible {
    case missingBlock(String)
    case missingTransformerShape(String)
    case incompleteTransformerShape(String)
    case unsupportedLayerPattern(String)

    public var description: String {
        switch self {
        case .missingBlock(let id):
            return "CAM static cost: block '\(id)' is missing"
        case .missingTransformerShape(let id):
            return "CAM static cost: block '\(id)' has no transformer shape"
        case .incompleteTransformerShape(let detail):
            return "CAM static cost: incomplete transformer shape: \(detail)"
        case .unsupportedLayerPattern(let detail):
            return "CAM static cost: unsupported layer pattern: \(detail)"
        }
    }
}

/// Deterministic architecture-level accounting directly from typed CAM IR.
///
/// This model intentionally does not predict dispatch count or GPU time. Those
/// depend on the frozen physical plan and remain the responsibility of
/// `SmeltFrozenIRCostModel`. This earlier layer answers whether a topology and
/// storage choice can fit, and the minimum logical work/traffic it entails.
public enum SmeltCAMStaticCostModel {
    public static func report(
        module: SmeltCAMIR,
        blockID: String = "trunk",
        scenario: SmeltCAMStaticCostScenario
    ) throws -> SmeltCAMStaticCostReport {
        let module = try module.validated()
        guard let block = module.blocks.first(where: { $0.id == blockID }) else {
            throw SmeltCAMStaticCostModelError.missingBlock(blockID)
        }
        guard let transformer = block.shape.transformer else {
            throw SmeltCAMStaticCostModelError.missingTransformerShape(blockID)
        }
        guard let hidden = transformer.hiddenSize,
              let layers = transformer.layers
        else {
            throw SmeltCAMStaticCostModelError.incompleteTransformerShape(
                "hidden-size and layers are required"
            )
        }

        let expandedRoles = try expand(layers)
        let attentionByRole = Dictionary(
            uniqueKeysWithValues: (transformer.attentionByRole ?? []).map {
                ($0.role, $0.attention)
            }
        )
        var roleAccumulators: [SmeltCAMIR.LayerRole: RoleAccumulator] = [:]
        var derivedResident: UInt64 = 0
        var derivedActive: UInt64 = 0
        var shortConvolutionState: UInt64 = 0
        var attentionArithmetic: UInt64 = 0

        if let vocab = transformer.vocab {
            let embedding = product(vocab.size, hidden)
            derivedResident += embedding
            derivedActive += embedding
            if !vocab.tiedHead {
                derivedResident += embedding
                derivedActive += embedding
            }
        }
        if transformer.norm != nil {
            derivedResident += UInt64(hidden)
            derivedActive += UInt64(hidden)
        }

        for (layerIndex, role) in expandedRoles.enumerated() {
            let attention = attentionByRole[role] ?? transformer.attention
            var layerSharedParameters = UInt64(0)
            if let attention {
                let projectionParameters = attentionParameterCount(
                    hidden: hidden,
                    attention: attention
                )
                layerSharedParameters += projectionParameters

                let attendedTokens = min(
                    scenario.contextLength,
                    attention.window ?? scenario.contextLength
                )
                let kvBytes = product(
                    2,
                    attention.kvHeads,
                    attention.headDim,
                    attendedTokens,
                    scenario.stateBytesPerElement
                )
                let kvWrite = product(
                    2,
                    attention.kvHeads,
                    attention.headDim,
                    scenario.stateBytesPerElement
                )
                let roleArithmetic = product(
                    4,
                    attention.qHeads,
                    attention.headDim,
                    attendedTokens
                )
                attentionArithmetic += roleArithmetic
                var accumulator = roleAccumulators[role] ?? RoleAccumulator()
                accumulator.layerCount += 1
                accumulator.attendedTokensPerLayer = attendedTokens
                accumulator.residentParametersPerLayer = projectionParameters
                accumulator.kvStateBytes += kvBytes
                accumulator.kvReadBytes += kvBytes
                accumulator.kvWriteBytes += kvWrite
                accumulator.attentionArithmetic += roleArithmetic
                roleAccumulators[role] = accumulator

                shortConvolutionState += shortConvolutionStateBytes(
                    transformer.shortConvolutions ?? [],
                    hidden: hidden,
                    attention: attention,
                    bytesPerElement: scenario.stateBytesPerElement
                )
            }

            if transformer.norm != nil {
                // Attention and FFN pre-norm weights.
                layerSharedParameters += product(2, hidden)
            }

            let leadingDense = transformer.denseLayerCount ?? 0
            if layerIndex < leadingDense || transformer.expert == nil {
                if let ffn = transformer.ffn {
                    layerSharedParameters += product(3, hidden, ffn.dim)
                }
                derivedResident += layerSharedParameters
                derivedActive += layerSharedParameters
            } else if let expert = transformer.expert,
                      let router = transformer.router {
                let sharedExperts = router.sharedExperts ?? 0
                let perExpert = product(3, hidden, expert.ffn.dim)
                let routerParameters = product(hidden, router.experts + sharedExperts)
                    + UInt64(router.scoreCorrectionBias == true ? router.experts : 0)
                    + UInt64(router.globalScale == true ? 1 : 0)
                layerSharedParameters += routerParameters
                derivedResident += layerSharedParameters
                    + perExpert * UInt64(router.experts + sharedExperts)
                derivedActive += layerSharedParameters
                    + perExpert * UInt64(router.topK + sharedExperts)
            } else {
                derivedResident += layerSharedParameters
                derivedActive += layerSharedParameters
            }
        }

        let requirements = Dictionary(
            uniqueKeysWithValues: block.shape.requirements.compactMap { requirement in
                requirement.value.map { (requirement.key, $0) }
            }
        )
        let declaredResident = parseParameterCount(requirements["total-parameters"])
        let declaredActive = parseParameterCount(requirements["active-parameters"])
        let storageParameterBase = declaredResident ?? derivedResident
        let bytesPerParameter: Double
        if let exact = scenario.exactCheckpointBytes, storageParameterBase > 0 {
            bytesPerParameter = Double(exact) / Double(storageParameterBase)
        } else {
            bytesPerParameter = scenario.storage.fallbackBytesPerParameter
        }
        let residentBytes = scenario.exactCheckpointBytes
            ?? roundedBytes(parameters: storageParameterBase, bytesPerParameter: bytesPerParameter)
        let activeParameterBase = declaredActive ?? derivedActive
        let activeWeightBytes = roundedBytes(
            parameters: activeParameterBase,
            bytesPerParameter: bytesPerParameter
        )
        let kvState = roleAccumulators.values.reduce(UInt64(0)) { $0 + $1.kvStateBytes }
        let kvRead = roleAccumulators.values.reduce(UInt64(0)) { $0 + $1.kvReadBytes }
        let kvWrite = roleAccumulators.values.reduce(UInt64(0)) { $0 + $1.kvWriteBytes }
        let totalPersistent = residentBytes + kvState + shortConvolutionState
        let memoryFits = scenario.memoryLimitBytes.map { totalPersistent <= $0 }

        // Every active matrix weight participates in at least one multiply-add;
        // attention score/value work is additional to that projection/FFN bill.
        let arithmetic = activeParameterBase * 2 + attentionArithmetic
        let bytesPerToken = activeWeightBytes + kvRead + kvWrite
        let lowerBound = scenario.sustainedMemoryBandwidthBytesPerSecond.flatMap { bandwidth in
            bandwidth > 0 ? Double(bytesPerToken) / bandwidth : nil
        }

        var assumptions = [
            "Architecture bill only; no physical dispatch count or GPU-time claim.",
            "One-token decode at the stated populated context; sliding attention is capped by its authored window.",
            "KV and short-convolution state use \(scenario.stateBytesPerElement) bytes per element.",
            "Active weight traffic assumes each selected dense/expert weight is logically read once per token.",
            "Arithmetic counts multiply and add separately and excludes elementwise/reduction details not dimensioned in CAM.",
        ]
        if scenario.exactCheckpointBytes == nil {
            assumptions.append(
                "Resident storage is estimated from \(scenario.storage.rawValue) logical bytes per parameter."
            )
        } else {
            assumptions.append(
                "Resident storage uses the caller-supplied exact checkpoint-index total; effective bytes per parameter are derived from it."
            )
        }
        if declaredResident != nil || declaredActive != nil {
            assumptions.append(
                "Declared total/active parameter counts drive storage and streaming traffic; independently derived counts remain visible for audit."
            )
        }

        let roleReports = roleAccumulators.map { role, value in
            SmeltCAMStaticAttentionRoleCost(
                role: role.rawValue,
                layerCount: value.layerCount,
                attendedTokensPerLayer: value.attendedTokensPerLayer,
                residentParametersPerLayer: value.residentParametersPerLayer,
                kvStateBytes: value.kvStateBytes,
                oneTokenKVReadBytes: value.kvReadBytes,
                oneTokenKVWriteBytes: value.kvWriteBytes,
                oneTokenAttentionArithmeticOperations: value.attentionArithmetic
            )
        }.sorted { $0.role < $1.role }

        return SmeltCAMStaticCostReport(
            schemaVersion: 1,
            moduleID: module.module.id,
            moduleSemanticSHA256: try module.semanticSHA256(),
            blockID: blockID,
            phase: "one-token-decode-at-populated-context",
            storage: scenario.storage,
            contextLength: scenario.contextLength,
            bytesPerStoredParameter: bytesPerParameter,
            exactCheckpointBytes: scenario.exactCheckpointBytes,
            declaredResidentParameters: declaredResident,
            declaredActiveParameters: declaredActive,
            derivedResidentParameters: derivedResident,
            derivedActiveParameters: derivedActive,
            residentWeightBytes: residentBytes,
            oneTokenActiveWeightReadBytes: activeWeightBytes,
            kvStateBytes: kvState,
            shortConvolutionStateBytes: shortConvolutionState,
            totalPersistentBytes: totalPersistent,
            memoryLimitBytes: scenario.memoryLimitBytes,
            fitsMemoryLimit: memoryFits,
            oneTokenKVReadBytes: kvRead,
            oneTokenKVWriteBytes: kvWrite,
            oneTokenArithmeticOperations: arithmetic,
            bandwidthLowerBoundSecondsPerToken: lowerBound,
            bandwidthUpperBoundTokensPerSecond: lowerBound.flatMap { $0 > 0 ? 1 / $0 : nil },
            attentionRoles: roleReports,
            assumptions: assumptions
        )
    }

    private struct RoleAccumulator {
        var layerCount = 0
        var attendedTokensPerLayer = 0
        var residentParametersPerLayer: UInt64 = 0
        var kvStateBytes: UInt64 = 0
        var kvReadBytes: UInt64 = 0
        var kvWriteBytes: UInt64 = 0
        var attentionArithmetic: UInt64 = 0
    }

    private static func expand(
        _ pattern: SmeltCAMIR.LayerPattern
    ) throws -> [SmeltCAMIR.LayerRole] {
        if let repeatCount = pattern.repeatCount, !pattern.roles.isEmpty {
            return Array(repeating: pattern.roles, count: repeatCount).flatMap { $0 }
        }
        if let count = pattern.count {
            if pattern.roles.isEmpty {
                return Array(repeating: .attention, count: count)
            }
            guard pattern.roles.count == 1 else {
                throw SmeltCAMStaticCostModelError.unsupportedLayerPattern(
                    "count with multiple roles has no authored ordering"
                )
            }
            return Array(repeating: pattern.roles[0], count: count)
        }
        throw SmeltCAMStaticCostModelError.unsupportedLayerPattern(
            "expected count or repeated role sequence"
        )
    }

    private static func attentionParameterCount(
        hidden: Int,
        attention: SmeltCAMIR.AttentionShape
    ) -> UInt64 {
        let qWidth = attention.qHeads * attention.headDim
        let kvWidth = attention.kvHeads * attention.headDim
        var parameters = product(hidden, qWidth)
            + product(2, hidden, kvWidth)
            + product(qWidth, hidden)
        if attention.qkNorm != nil {
            parameters += UInt64(qWidth + kvWidth)
        }
        if let relative = attention.relativePosition {
            parameters += product(hidden, attention.qHeads, relative.projectionDim)
            parameters += product(relative.projectionDim, relative.extent)
        }
        return parameters
    }

    private static func shortConvolutionStateBytes(
        _ convolutions: [SmeltCAMIR.ShortConvolutionShape],
        hidden: Int,
        attention: SmeltCAMIR.AttentionShape,
        bytesPerElement: Int
    ) -> UInt64 {
        convolutions.reduce(UInt64(0)) { total, convolution in
            let width: Int
            switch convolution.site {
            case .attentionKey, .attentionValue:
                width = attention.kvHeads * attention.headDim
            case .attentionBranchOutput, .ffnBranchOutput:
                width = hidden
            }
            return total + product(
                max(convolution.kernelSize - 1, 0),
                width,
                bytesPerElement
            )
        }
    }

    private static func parseParameterCount(_ value: String?) -> UInt64? {
        guard let value else { return nil }
        let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()
        let multiplier: Double
        let number: Substring
        if normalized.hasSuffix("B") {
            multiplier = 1_000_000_000
            number = normalized.dropLast()
        } else if normalized.hasSuffix("M") {
            multiplier = 1_000_000
            number = normalized.dropLast()
        } else {
            multiplier = 1
            number = Substring(normalized)
        }
        guard let parsed = Double(number), parsed >= 0 else { return nil }
        return UInt64((parsed * multiplier).rounded())
    }

    private static func roundedBytes(
        parameters: UInt64,
        bytesPerParameter: Double
    ) -> UInt64 {
        UInt64((Double(parameters) * bytesPerParameter).rounded(.up))
    }

    private static func product(_ values: Int...) -> UInt64 {
        values.reduce(UInt64(1)) { $0 * UInt64(max($1, 0)) }
    }
}
