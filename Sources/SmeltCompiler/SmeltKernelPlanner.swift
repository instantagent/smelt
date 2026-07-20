// Model-derived kernel planning shared by source generation and weight-layout
// planning. The registry defines legal kernel capabilities; this planner maps
// an IR's graph, shapes, and quantization policy to concrete consumers.

import SmeltSchema

struct SmeltPlannedKernelWeight: Sendable, Equatable, Hashable {
    let weightName: String
    let role: SmeltKernelWeightRole
}

struct SmeltPlannedKernelUse: Sendable, Equatable {
    let consumerID: String
    let capability: SmeltKernelCapability
    let weights: [SmeltPlannedKernelWeight]
    let kind: SmeltKernelConsumerKind?

    init(
        consumerID: String,
        capability: SmeltKernelCapability,
        weights: [SmeltPlannedKernelWeight],
        kind: SmeltKernelConsumerKind? = nil
    ) {
        self.consumerID = consumerID
        self.capability = capability
        self.weights = weights
        self.kind = kind
    }
}

struct SmeltPlannedKernelCandidate: Sendable, Equatable {
    let consumerID: String
    let operation: SmeltKernelOperationPattern
    let shape: SmeltKernelShape
    let weights: [SmeltPlannedKernelWeight]
    let kind: SmeltKernelConsumerKind?

    init(
        consumerID: String,
        operation: SmeltKernelOperationPattern,
        shape: SmeltKernelShape,
        weights: [SmeltPlannedKernelWeight],
        kind: SmeltKernelConsumerKind? = nil
    ) {
        self.consumerID = consumerID
        self.operation = operation
        self.shape = shape
        self.weights = weights
        self.kind = kind
    }
}

struct SmeltUnsupportedKernelCandidate: Sendable, Equatable {
    let candidate: SmeltPlannedKernelCandidate
    let reason: String
}

struct SmeltPlannedKernelRoute: Sendable, Equatable {
    let candidate: SmeltPlannedKernelCandidate
    let capability: SmeltKernelCapability

    var pipelineNameOverride: String {
        capability.id
    }

    func launchGeometry(
        operation: SmeltKernelOperationPattern,
        expectedShape: SmeltKernelShape? = nil
    ) -> SmeltPlannedKernelLaunchGeometry? {
        guard candidate.operation == operation,
              capability.operation == operation,
              capability.shape == candidate.shape,
              expectedShape.map({ $0 == candidate.shape }) ?? true
        else {
            return nil
        }
        return SmeltPlannedKernelLaunchGeometry(capability: capability)
    }

    func affineMatvecResidualAddLaunchGeometry(
        expectedShape: SmeltKernelShape? = nil
    ) -> SmeltPlannedKernelLaunchGeometry? {
        launchGeometry(operation: .affineMatvecResidualAdd, expectedShape: expectedShape)
    }

    func fusedDualAffineMatvecResidualAddLaunchGeometry(
        expectedShape: SmeltKernelShape? = nil
    ) -> SmeltPlannedKernelLaunchGeometry? {
        launchGeometry(
            operation: .fusedDualAffineMatvecResidualAdd,
            expectedShape: expectedShape
        )
    }

    func affineMatvecPrefillFullLaunchGeometry(
        expectedShape: SmeltKernelShape? = nil
    ) -> SmeltPlannedKernelLaunchGeometry? {
        launchGeometry(operation: .affineMatvecPrefillFull, expectedShape: expectedShape)
    }

    func fusedGateUpSwigluPrefillFullLaunchGeometry(
        expectedShape: SmeltKernelShape? = nil
    ) -> SmeltPlannedKernelLaunchGeometry? {
        launchGeometry(operation: .fusedGateUpSwigluPrefillFull, expectedShape: expectedShape)
    }

    func affineMatvecPrefillSmallBatchLaunchGeometry(
        expectedShape: SmeltKernelShape? = nil
    ) -> SmeltPlannedKernelLaunchGeometry? {
        launchGeometry(operation: .affineMatvecPrefillSmallBatch, expectedShape: expectedShape)
    }

    func fusedGateUpSwigluPrefillSmallBatchLaunchGeometry(
        expectedShape: SmeltKernelShape? = nil
    ) -> SmeltPlannedKernelLaunchGeometry? {
        launchGeometry(
            operation: .fusedGateUpSwigluPrefillSmallBatch,
            expectedShape: expectedShape
        )
    }

    var generatedBufferBindingCount: Int? {
        switch capability.operation {
        case .affineMatvecResidualAdd:
            return 6
        case .fusedDualAffineMatvecResidualAdd:
            return 11
        case .affineMatvecPrefillFull:
            return 5
        case .fusedGateUpSwigluPrefillFull:
            return 8
        case .affineMatvecPrefillSmallBatch:
            return 5
        case .fusedDualAffineMatvecPrefillSmallBatch:
            return 9
        case .fusedGateUpSwigluPrefillSmallBatch:
            return 8
        case .affineVerifyArgmaxPrefill:
            return 5
        case .verifyArgmaxReduce:
            return 2
        case .affineStorageRead, .signedStorageRead:
            return nil
        }
    }

    var generatedConstantCount: Int? {
        switch capability.operation {
        case .affineMatvecResidualAdd,
             .fusedDualAffineMatvecResidualAdd:
            return 0
        case .affineMatvecPrefillFull,
             .fusedGateUpSwigluPrefillFull,
             .affineMatvecPrefillSmallBatch,
             .fusedDualAffineMatvecPrefillSmallBatch,
             .fusedGateUpSwigluPrefillSmallBatch:
            return 1
        case .affineVerifyArgmaxPrefill:
            return 2
        case .verifyArgmaxReduce:
            return 1
        case .affineStorageRead, .signedStorageRead:
            return nil
        }
    }
}

struct SmeltPlannedWeightUse: Sendable, Equatable {
    let consumerID: String
    let weightName: String
    let capability: SmeltKernelCapability
    let weightRole: SmeltKernelWeightRole
    let acceptedLayouts: [SmeltWeightStorageKind]
    let consumerKind: SmeltWeightConsumerKind?

    init(
        consumerID: String,
        weightName: String,
        capability: SmeltKernelCapability,
        weightRole: SmeltKernelWeightRole,
        acceptedLayouts: [SmeltWeightStorageKind],
        consumerKind: SmeltWeightConsumerKind? = nil
    ) {
        self.consumerID = consumerID
        self.weightName = weightName
        self.capability = capability
        self.weightRole = weightRole
        self.acceptedLayouts = acceptedLayouts
        self.consumerKind = consumerKind
    }
}

struct SmeltGeneratedPipelineUseAuthorizationFailure: Sendable, Equatable {
    let pipelineName: String
    let consumerID: String
    let plannedCapabilityName: String?
}

enum SmeltWeightConsumerKind: Sendable, Equatable, Hashable {
    case kernel(SmeltKernelConsumerKind)
    case storageRead

    var reportName: String {
        switch self {
        case .kernel(let kind):
            return kind.rawValue
        case .storageRead:
            return "storageRead"
        }
    }
}

struct SmeltPlannedKernelLaunchGeometry: Sendable, Equatable {
    let rowTile: Int
    let batchTile: Int?
    let threadgroupWidth: Int

    init?(capability: SmeltKernelCapability) {
        guard let rowTile = capability.rowTile,
              let threadgroupWidth = capability.threadgroupWidth
        else {
            return nil
        }
        self.rowTile = rowTile
        self.batchTile = capability.batchTile
        self.threadgroupWidth = threadgroupWidth
    }

    func gridWidth(rows: Int) -> Int {
        (rows + rowTile - 1) / rowTile
    }

    func gridHeight(batchSize: Int) -> Int? {
        guard let batchTile else { return nil }
        return (batchSize + batchTile - 1) / batchTile
    }
}

public struct SmeltKernelPlan: Sendable, Equatable {
    public static let empty = SmeltKernelPlan(generatedUses: [])

    let generatedUses: [SmeltPlannedKernelUse]
    let generatedCandidateCount: Int
    let unsupportedGeneratedCandidateCount: Int
    let unsupportedGeneratedCandidates: [SmeltUnsupportedKernelCandidate]
    let generatedCapabilities: [SmeltKernelCapability]
    let generatedCapabilitiesByName: [String: SmeltKernelCapability]
    let generatedIndex: SmeltKernelPlanIndex

    init(
        generatedUses: [SmeltPlannedKernelUse],
        generatedCandidateCount: Int? = nil,
        unsupportedGeneratedCandidateCount: Int? = nil,
        unsupportedGeneratedCandidates: [SmeltUnsupportedKernelCandidate] = []
    ) {
        self.generatedUses = generatedUses
        self.unsupportedGeneratedCandidates = unsupportedGeneratedCandidates
        self.unsupportedGeneratedCandidateCount =
            unsupportedGeneratedCandidateCount ?? unsupportedGeneratedCandidates.count
        self.generatedCandidateCount =
            generatedCandidateCount ?? generatedUses.count + self.unsupportedGeneratedCandidateCount
        self.generatedCapabilities = SmeltKernelPlanner.uniqueCapabilities(
            from: generatedUses
        )
        self.generatedCapabilitiesByName = Dictionary(
            uniqueKeysWithValues: self.generatedCapabilities.map { ($0.id, $0) }
        )
        self.generatedIndex = SmeltKernelPlanIndex(uses: generatedUses)
    }

    var isEmpty: Bool {
        generatedUses.isEmpty
    }

    var generatedCapabilityNames: [String] {
        generatedCapabilities.map(\.id)
    }

    var emittedGeneratedCapabilities: [SmeltKernelCapability] {
        generatedCapabilities.filter(\.requiresPackageLocalGeneratedSource)
    }

    var emittedGeneratedCapabilityNames: [String] {
        emittedGeneratedCapabilities.map(\.id)
    }

    var generatedCapabilityNameSet: Set<String> {
        Set(generatedCapabilityNames)
    }

    var emittedGeneratedCapabilityNameSet: Set<String> {
        Set(emittedGeneratedCapabilityNames)
    }

    var plannedWeightUses: [SmeltPlannedWeightUse] {
        generatedUses.flatMap { plannedUse in
            plannedUse.weights.compactMap { weight in
                guard let requirement = plannedUse.capability.weightRequirements.first(
                    where: { $0.role == weight.role }
                ) else {
                    return nil
                }
                return SmeltPlannedWeightUse(
                    consumerID: plannedUse.consumerID,
                    weightName: weight.weightName,
                    capability: plannedUse.capability,
                    weightRole: weight.role,
                    acceptedLayouts: requirement.acceptedLayouts,
                    consumerKind: plannedUse.kind.map(SmeltWeightConsumerKind.kernel)
                )
            }
        }
    }

    func containsGeneratedCapability(named name: String) -> Bool {
        generatedCapabilityNameSet.contains(name)
    }

    func generatedCapability(named name: String) -> SmeltKernelCapability? {
        generatedCapabilitiesByName[name]
    }

    private static func catalogRequiresPackageLocalGeneratedSource(named name: String) -> Bool {
        SmeltKernelCatalog.pipelineIndex(named: name) == nil
    }

    func requiresPackageLocalGeneratedSource(named name: String) -> Bool {
        if let capability = generatedCapability(named: name) {
            return capability.requiresPackageLocalGeneratedSource
        }
        return Self.catalogRequiresPackageLocalGeneratedSource(named: name)
    }

    func packageLocalGeneratedPipelineNames(_ names: [String]) -> [String] {
        names.filter(requiresPackageLocalGeneratedSource(named:))
    }

    func unplannedGeneratedPipelineNames(_ names: [String]) -> [String] {
        packageLocalGeneratedPipelineNames(names)
            .filter { !containsGeneratedCapability(named: $0) }
    }

    func generatedPipelineUsesMissingPlannedCandidates(
        _ uses: [SmeltNamedPipelineUse]
    ) -> [String] {
        uses
            .filter { requiresPackageLocalGeneratedSource(named: $0.name) }
            .filter { $0.plannedKernelCandidate == nil }
            .map(\.name)
    }

    func unauthorizedGeneratedPipelineUses(
        _ uses: [SmeltNamedPipelineUse]
    ) -> [SmeltGeneratedPipelineUseAuthorizationFailure] {
        uses.compactMap { use -> SmeltGeneratedPipelineUseAuthorizationFailure? in
            guard let candidate = use.plannedKernelCandidate else {
                return nil
            }
            guard let route = route(for: candidate) else {
                return SmeltGeneratedPipelineUseAuthorizationFailure(
                    pipelineName: use.name,
                    consumerID: candidate.consumerID,
                    plannedCapabilityName: nil
                )
            }
            guard route.capability.id == use.name else {
                return SmeltGeneratedPipelineUseAuthorizationFailure(
                    pipelineName: use.name,
                    consumerID: candidate.consumerID,
                    plannedCapabilityName: route.capability.id
                )
            }
            return nil
        }
    }

    func route(for candidate: SmeltPlannedKernelCandidate) -> SmeltPlannedKernelRoute? {
        guard let use = generatedIndex.use(
            consumerID: candidate.consumerID,
            operation: candidate.operation,
            shape: candidate.shape,
            weights: candidate.weights
        ) else {
            return nil
        }
        return SmeltPlannedKernelRoute(
            candidate: candidate,
            capability: use.capability
        )
    }

    func route(
        kind: SmeltKernelConsumerKind,
        context: SmeltKernelLayerConsumerContext
    ) -> SmeltPlannedKernelRoute? {
        guard let candidate = SmeltKernelConsumerNaming.candidate(
            kind: kind,
            context: context
        ) else {
            return nil
        }
        return route(for: candidate)
    }

    func route(
        kind: SmeltKernelConsumerKind,
        context: SmeltKernelLayerConsumerContext,
        operation: SmeltKernelOperationPattern
    ) -> SmeltPlannedKernelRoute? {
        guard let base = SmeltKernelConsumerNaming.candidate(
            kind: kind,
            context: context
        ) else {
            return nil
        }
        return route(for: SmeltPlannedKernelCandidate(
            consumerID: base.consumerID,
            operation: operation,
            shape: base.shape,
            weights: base.weights,
            kind: base.kind
        ))
    }

}

extension SmeltKernelPlan {
    func plannedKernelConsumerReports() -> [SmeltPlannedKernelConsumerReport] {
        generatedUses.map { use in
            SmeltPlannedKernelConsumerReport(
                consumerID: use.consumerID,
                consumerKind: use.kind?.rawValue,
                capabilityName: use.capability.id,
                phase: use.capability.phase.rawValue,
                operation: use.capability.operation.rawValue,
                rows: use.capability.shape.rows,
                cols: use.capability.shape.cols,
                groupSize: use.capability.shape.groupSize,
                weights: use.weights.map { weight in
                    SmeltPlannedKernelWeightReport(
                        weightName: weight.weightName,
                        role: weight.role.rawValue
                    )
                }
            )
        }
    }

    func generatedKernelCapabilityReports(
        storageKindID: (SmeltWeightStorageKind) -> String
    ) -> [SmeltGeneratedKernelCapabilityReport] {
        generatedCapabilities.map { capability in
            SmeltGeneratedKernelCapabilityReport(
                capabilityName: capability.id,
                phase: capability.phase.rawValue,
                operation: capability.operation.rawValue,
                rows: capability.shape.rows,
                cols: capability.shape.cols,
                groupSize: capability.shape.groupSize,
                sourceKind: capability.source.rawValue,
                emittedGeneratedSource: capability.requiresPackageLocalGeneratedSource,
                sourceTemplate: capability.sourceTemplate?.rawValue,
                weightRequirements: capability.weightRequirements.map { requirement in
                    SmeltGeneratedKernelWeightRequirementReport(
                        role: requirement.role.rawValue,
                        acceptedLayouts: requirement.acceptedLayouts.map(storageKindID)
                    )
                },
                rowTile: capability.rowTile,
                batchTile: capability.batchTile,
                threadgroupWidth: capability.threadgroupWidth
            )
        }
    }

    func unsupportedKernelCandidateReports() -> [SmeltUnsupportedKernelCandidateReport] {
        unsupportedGeneratedCandidates.map { unsupported in
            let candidate = unsupported.candidate
            return SmeltUnsupportedKernelCandidateReport(
                consumerID: candidate.consumerID,
                consumerKind: candidate.kind?.rawValue,
                phase: candidate.operation.plannedPhase.rawValue,
                operation: candidate.operation.rawValue,
                rows: candidate.shape.rows,
                cols: candidate.shape.cols,
                groupSize: candidate.shape.groupSize,
                weights: candidate.weights.map { weight in
                    SmeltPlannedKernelWeightReport(
                        weightName: weight.weightName,
                        role: weight.role.rawValue
                    )
                },
                reason: unsupported.reason
            )
        }
    }
}

struct SmeltKernelPlanIndex: Sendable, Equatable {
    private let usesByConsumerID: [String: [SmeltPlannedKernelUse]]

    init(uses: [SmeltPlannedKernelUse]) {
        self.usesByConsumerID = Dictionary(grouping: uses, by: \.consumerID)
    }

    func use(
        consumerID: String,
        operation: SmeltKernelOperationPattern,
        shape: SmeltKernelShape,
        weights: [SmeltPlannedKernelWeight]
    ) -> SmeltPlannedKernelUse? {
        usesByConsumerID[consumerID]?.first {
            $0.capability.operation == operation
                && $0.capability.shape == shape
                && $0.weights == weights
        }
    }

}

enum SmeltKernelPlanner {
    static func plan(for ir: SmeltModelIR) -> SmeltKernelPlan {
        guard ir.compilation.generatedKernels == .auto else { return .empty }
        guard ir.quantization.strategy == .affineU4 else { return .empty }

        let candidates = plannedGeneratedKernelCandidates(for: ir)
            .filter { ir.compilation.allowsGeneratedKernelConsumer(kind: $0.kind) }
        var uses: [SmeltPlannedKernelUse] = []
        var unsupported: [SmeltUnsupportedKernelCandidate] = []
        for candidate in candidates {
            if let use = plannedUse(from: candidate) {
                uses.append(use)
            } else {
                unsupported.append(SmeltUnsupportedKernelCandidate(
                    candidate: candidate,
                    reason: unsupportedReason(for: candidate)
                ))
            }
        }
        return SmeltKernelPlan(
            generatedUses: uses,
            generatedCandidateCount: candidates.count,
            unsupportedGeneratedCandidates: unsupported
        )
    }

    private static func plannedUse(
        from candidate: SmeltPlannedKernelCandidate
    ) -> SmeltPlannedKernelUse? {
        guard let capability = SmeltKernelCapabilityRegistry.generatedCapability(
            operation: candidate.operation,
            shape: candidate.shape
        ) else {
            return nil
        }
        let plannedRoles = Set(candidate.weights.map(\.role))
        guard capability.weightRequirements.allSatisfy({
            plannedRoles.contains($0.role)
        }) else {
            return nil
        }
        return SmeltPlannedKernelUse(
            consumerID: candidate.consumerID,
            capability: capability,
            weights: candidate.weights,
            kind: candidate.kind
        )
    }

    private static func unsupportedReason(
        for candidate: SmeltPlannedKernelCandidate
    ) -> String {
        guard let capability = SmeltKernelCapabilityRegistry.generatedCapability(
            operation: candidate.operation,
            shape: candidate.shape
        ) else {
            return "no_generated_capability"
        }
        let plannedRoles = Set(candidate.weights.map(\.role))
        let requiredRoles = Set(capability.weightRequirements.map(\.role))
        guard requiredRoles.isSubset(of: plannedRoles) else {
            return "missing_required_weight_role"
        }
        return "unsupported_generated_candidate"
    }

    private static func plannedGeneratedKernelCandidates(
        for ir: SmeltModelIR
    ) -> [SmeltPlannedKernelCandidate] {
        var candidates: [SmeltPlannedKernelCandidate] = []
        let groupSize = ir.quantization.groupSize

        for (layerIndex, layerType) in ir.layerPattern.expanded.enumerated() {
            let context = SmeltKernelLayerConsumerContext(
                ir: ir,
                layerIndex: layerIndex,
                layerType: layerType,
                groupSize: groupSize
            )
            candidates.append(contentsOf: SmeltKernelConsumerNaming.candidates(for: context))
        }

        let needsSmallBatchVerify = ir.prefill?.verifyArgmax == true
            || ir.prefill?.emitAllLogits == true
        guard needsSmallBatchVerify else { return candidates }

        let verifyCandidates = candidates.compactMap {
            candidate -> SmeltPlannedKernelCandidate? in
            let operation: SmeltKernelOperationPattern
            switch candidate.operation {
            case .affineMatvecPrefillFull:
                operation = .affineMatvecPrefillSmallBatch
            case .fusedGateUpSwigluPrefillFull:
                operation = .fusedGateUpSwigluPrefillSmallBatch
            default:
                return nil
            }
            return SmeltPlannedKernelCandidate(
                consumerID: candidate.consumerID,
                operation: operation,
                shape: candidate.shape,
                weights: candidate.weights,
                kind: candidate.kind
            )
        }
        var allCandidates = candidates + verifyCandidates
        allCandidates.append(contentsOf: smallBatchAffineProjectionCandidates(
            for: ir,
            groupSize: groupSize
        ))
        if ir.prefill?.verifyArgmax == true {
            let lmHeadShape = SmeltKernelShape(
                rows: ir.config.vocabSize,
                cols: ir.config.hiddenSize,
                groupSize: groupSize
            )
            allCandidates.append(SmeltPlannedKernelCandidate(
                consumerID: "lm_head.verify.argmax",
                operation: .affineVerifyArgmaxPrefill,
                shape: lmHeadShape,
                weights: [
                    SmeltPlannedKernelWeight(
                        weightName: ir.config.tiedLMHead
                            ? SmeltCanonicalTensorNames.embedTokens
                            : "lm_head_weight",
                        role: .affine
                    ),
                ]
            ))
            allCandidates.append(SmeltPlannedKernelCandidate(
                consumerID: "lm_head.verify.argmax.reduce",
                operation: .verifyArgmaxReduce,
                shape: SmeltKernelShape(
                    rows: ir.config.vocabSize,
                    cols: 0,
                    groupSize: 0
                ),
                weights: []
            ))
        }
        return allCandidates
    }

    /// Projections that are independently emitted for every prefill position.
    /// Registering them from the structural IR lets verification batches reuse
    /// each dequantized weight tile without any model-name or package special
    /// case. Fused dual projections are deliberately absent: they need their
    /// own capability rather than silently splitting an existing fusion.
    private static func smallBatchAffineProjectionCandidates(
        for ir: SmeltModelIR,
        groupSize: Int
    ) -> [SmeltPlannedKernelCandidate] {
        let hidden = ir.config.hiddenSize
        var candidates: [SmeltPlannedKernelCandidate] = []

        for (layerIndex, layerType) in ir.layerPattern.expanded.enumerated() {
            if layerType == .delta, let delta = ir.config.delta {
                candidates.append(SmeltKernelConsumerNaming.affinePrefillSmallBatchCandidate(
                    weightName: SmeltKernelConsumerNaming.deltaQKVWeight(
                        layerIndex: layerIndex
                    ),
                    rows: delta.qkvDim,
                    cols: hidden,
                    groupSize: groupSize
                ))
                candidates.append(SmeltKernelConsumerNaming.affinePrefillSmallBatchCandidate(
                    weightName: SmeltKernelConsumerNaming.deltaZWeight(
                        layerIndex: layerIndex
                    ),
                    rows: delta.zDim,
                    cols: hidden,
                    groupSize: groupSize
                ))
                candidates.append(SmeltKernelConsumerNaming.affinePrefillSmallBatchCandidate(
                    weightName: SmeltKernelConsumerNaming.deltaOutputWeight(
                        layerIndex: layerIndex
                    ),
                    rows: hidden,
                    cols: delta.zDim,
                    groupSize: groupSize
                ))
                candidates.append(
                    SmeltKernelConsumerNaming.fusedDualAffinePrefillSmallBatchCandidate(
                        firstWeightName: SmeltKernelConsumerNaming.deltaBWeight(
                            layerIndex: layerIndex
                        ),
                        secondWeightName: SmeltKernelConsumerNaming.deltaAWeight(
                            layerIndex: layerIndex
                        ),
                        rows: delta.numHeads,
                        cols: hidden,
                        groupSize: groupSize
                    )
                )
                continue
            }

            guard let attention = ir.config.attentionConfig(for: layerType) else {
                continue
            }
            candidates.append(SmeltKernelConsumerNaming.affinePrefillSmallBatchCandidate(
                weightName: SmeltKernelConsumerNaming.qProjWeight(
                    layerIndex: layerIndex
                ),
                rows: attention.qProjDim,
                cols: hidden,
                groupSize: groupSize
            ))
            candidates.append(SmeltKernelConsumerNaming.affinePrefillSmallBatchCandidate(
                weightName: SmeltKernelConsumerNaming.oProjWeight(
                    layerIndex: layerIndex
                ),
                rows: hidden,
                cols: attention.qHeads * attention.headDim,
                groupSize: groupSize
            ))
            if !attention.externalKV,
               !ir.isKVSharedLayer(layerIndex),
               attention.kProjDim == attention.vProjDim
            {
                candidates.append(
                    SmeltKernelConsumerNaming.fusedDualAffinePrefillSmallBatchCandidate(
                        firstWeightName: SmeltKernelConsumerNaming.kProjWeight(
                            layerIndex: layerIndex
                        ),
                        secondWeightName: SmeltKernelConsumerNaming.vProjWeight(
                            layerIndex: layerIndex
                        ),
                        rows: attention.kProjDim,
                        cols: hidden,
                        groupSize: groupSize
                    )
                )
            }
        }

        return candidates
    }

    fileprivate static func uniqueCapabilities(
        from uses: [SmeltPlannedKernelUse]
    ) -> [SmeltKernelCapability] {
        var emittedIDs: Set<String> = []
        var capabilities: [SmeltKernelCapability] = []

        for use in uses {
            guard emittedIDs.insert(use.capability.id).inserted else { continue }
            capabilities.append(use.capability)
        }

        return capabilities
    }
}

extension SmeltKernelOperationPattern {
    var plannedPhase: SmeltKernelPhase {
        switch self {
        case .affineMatvecResidualAdd, .fusedDualAffineMatvecResidualAdd:
            return .decode
        case .affineMatvecPrefillFull,
             .fusedGateUpSwigluPrefillFull,
             .affineMatvecPrefillSmallBatch,
             .fusedDualAffineMatvecPrefillSmallBatch,
             .fusedGateUpSwigluPrefillSmallBatch,
             .affineVerifyArgmaxPrefill,
             .verifyArgmaxReduce:
            return .prefill
        case .affineStorageRead, .signedStorageRead:
            return .storage
        }
    }
}
