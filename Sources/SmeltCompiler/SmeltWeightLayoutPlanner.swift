// Planned weight consumers for kernel-planned layout decisions.
//
// This layer is the compiler boundary between logical weights and physical
// storage: derive consumers from the kernel plan, choose one legal storage
// layout per logical weight, then hand the resulting entries to packers.

import SmeltSchema

struct SmeltWeightStorageDecision: Sendable, Equatable {
    let weightName: String
    let currentLayout: SmeltWeightStorageKind
    let selectedLayout: SmeltWeightStorageKind
    let uses: [SmeltPlannedWeightUse]
    let requiresDuplicateLayout: Bool
}

struct SmeltWeightStorageIssue: Sendable, Equatable {
    enum Kind: Sendable, Equatable {
        case missingWeightEntry
        case unsupportedCurrentLayout
        case currentLayoutRejected

        var reportName: String {
            switch self {
            case .missingWeightEntry:
                return "missingWeightEntry"
            case .unsupportedCurrentLayout:
                return "unsupportedCurrentLayout"
            case .currentLayoutRejected:
                return "currentLayoutRejected"
            }
        }
    }

    let weightName: String
    let kind: Kind
    let consumers: [SmeltPlannedWeightStorageDecisionConsumerReport]

    var consumerIDs: [String] {
        consumers.map(\.consumerID)
    }

    init(
        weightName: String,
        kind: Kind,
        consumers: [SmeltPlannedWeightStorageDecisionConsumerReport]
    ) {
        self.weightName = weightName
        self.kind = kind
        self.consumers = consumers
    }

    init(
        weightName: String,
        kind: Kind,
        uses: [SmeltPlannedWeightUse]
    ) {
        self.init(
            weightName: weightName,
            kind: kind,
            consumers: uses.map(Self.consumerReport)
        )
    }

    init(
        weightName: String,
        kind: Kind,
        consumerIDs: [String]
    ) {
        self.init(
            weightName: weightName,
            kind: kind,
            consumers: consumerIDs.map {
                SmeltPlannedWeightStorageDecisionConsumerReport(consumerID: $0)
            }
        )
    }

    private static func consumerReport(
        _ use: SmeltPlannedWeightUse
    ) -> SmeltPlannedWeightStorageDecisionConsumerReport {
        SmeltPlannedWeightStorageDecisionConsumerReport(
            consumerID: use.consumerID,
            consumerKind: use.consumerKind?.reportName
        )
    }
}

enum SmeltWeightStoragePlanValidationFailure: Sendable, Equatable {
    case illegalStorage([SmeltWeightStorageIssue])
    case duplicatePhysicalStorage([String])
}

struct SmeltWeightStoragePlan: Sendable, Equatable {
    let plannedUses: [SmeltPlannedWeightUse]
    let decisions: [SmeltWeightStorageDecision]
    let issues: [SmeltWeightStorageIssue]

    var isMemoryNeutral: Bool {
        validationFailure(policy: .memoryNeutral) == nil
    }

    var duplicateLayoutWeightNames: [String] {
        decisions
            .filter(\.requiresDuplicateLayout)
            .map(\.weightName)
    }

    func decision(for weightName: String) -> SmeltWeightStorageDecision? {
        decisions.first { $0.weightName == weightName }
    }

    func validationFailure(
        policy: SmeltWeightLayoutPolicy
    ) -> SmeltWeightStoragePlanValidationFailure? {
        guard issues.isEmpty else {
            return .illegalStorage(issues)
        }
        switch policy {
        case .memoryNeutral:
            let duplicates = duplicateLayoutWeightNames
            return duplicates.isEmpty ? nil : .duplicatePhysicalStorage(duplicates)
        }
    }

    init(
        plannedUses: [SmeltPlannedWeightUse],
        decisions: [SmeltWeightStorageDecision],
        issues: [SmeltWeightStorageIssue]
    ) {
        self.plannedUses = plannedUses
        self.decisions = decisions
        self.issues = issues
    }
}

extension SmeltWeightStoragePlan {
    var plannedWeightNames: [String] {
        Set(plannedUses.map(\.weightName)).sorted()
    }

    var plannedWeightConsumerIDs: [String] {
        Set(plannedUses.map(\.consumerID)).sorted()
    }

    var storageDecisionNames: [String] {
        decisions.map(\.weightName)
    }

    var duplicateLayoutCount: Int {
        duplicateLayoutWeightNames.count
    }

    var issueNames: [String] {
        issues.map(\.weightName).sorted()
    }

    func issueReports() -> [SmeltPlannedWeightStorageIssueReport] {
        issues.map { issue in
            SmeltPlannedWeightStorageIssueReport(
                weightName: issue.weightName,
                kind: issue.kind.reportName,
                consumers: issue.consumers
            )
        }
    }

    func plannedWeightConsumerReports(
        storageKindID: (SmeltWeightStorageKind) -> String
    ) -> [SmeltPlannedWeightConsumerReport] {
        plannedUses.map { use in
            SmeltPlannedWeightConsumerReport(
                weightName: use.weightName,
                consumerID: use.consumerID,
                consumerKind: use.consumerKind?.reportName,
                capabilityName: use.capability.id,
                weightRole: use.weightRole.rawValue,
                acceptedLayouts: use.acceptedLayouts.map(storageKindID)
            )
        }
    }

    func storageDecisionReports(
        storageKindID: (SmeltWeightStorageKind) -> String
    ) -> [SmeltPlannedWeightStorageDecisionReport] {
        decisions.map { decision in
            SmeltPlannedWeightStorageDecisionReport(
                weightName: decision.weightName,
                currentLayout: storageKindID(decision.currentLayout),
                selectedLayout: storageKindID(decision.selectedLayout),
                consumers: decision.uses.map { use in
                    SmeltPlannedWeightStorageDecisionConsumerReport(
                        consumerID: use.consumerID,
                        consumerKind: use.consumerKind?.reportName
                    )
                },
                requiresDuplicateLayout: decision.requiresDuplicateLayout
            )
        }
    }
}

struct SmeltPlannedWeightLayout: Sendable {
    let entries: [SmeltWeightEntry]
    let storagePlan: SmeltWeightStoragePlan
}

enum SmeltWeightLayoutPlanner {
    static func plannedWeightUses(
        for ir: SmeltModelIR,
        kernelPlan: SmeltKernelPlan? = nil
    ) -> [SmeltPlannedWeightUse] {
        let kernelPlan = kernelPlan ?? SmeltKernelPlanner.plan(for: ir)
        return plannedWeightUses(
            entries: SmeltWeightLayout.computeLayout(from: ir),
            kernelPlan: kernelPlan
        )
    }

    static func storagePlan(
        for ir: SmeltModelIR,
        entries: [SmeltWeightEntry],
        kernelPlan: SmeltKernelPlan? = nil
    ) -> SmeltWeightStoragePlan {
        let kernelPlan = kernelPlan ?? SmeltKernelPlanner.plan(for: ir)
        return storagePlan(entries: entries, kernelPlan: kernelPlan)
    }

    static func storagePlan(
        entries: [SmeltWeightEntry],
        kernelPlan: SmeltKernelPlan
    ) -> SmeltWeightStoragePlan {
        storagePlan(
            entries: entries,
            plannedUses: plannedWeightUses(entries: entries, kernelPlan: kernelPlan)
        )
    }

    static func plannedWeightUses(
        entries: [SmeltWeightEntry],
        kernelPlan: SmeltKernelPlan
    ) -> [SmeltPlannedWeightUse] {
        let kernelUses = kernelPlan.plannedWeightUses
        let kernelCoveredWeights = Set(kernelUses.map(\.weightName))
        let storageUses = entries.compactMap { entry -> SmeltPlannedWeightUse? in
            guard !kernelCoveredWeights.contains(entry.name),
                  let currentLayout = currentStorageKind(for: entry)
            else {
                return nil
            }
            switch currentLayout {
            case .affineU4RowMajor(let groupSize):
                let rows = entry.shape.first ?? 0
                let cols = entry.shape.dropFirst().first ?? 0
                let capability = SmeltKernelCapabilityRegistry.affineStorageRead(
                    rows: rows,
                    cols: cols,
                    groupSize: groupSize
                )
                return SmeltPlannedWeightUse(
                    consumerID: "\(entry.name).storage",
                    weightName: entry.name,
                    capability: capability,
                    weightRole: .affine,
                    acceptedLayouts: [.affineU4RowMajor(groupSize: groupSize)],
                    consumerKind: .storageRead
                )
            case let .signedRowMajor(format, groupSize):
                let rows = entry.shape.first ?? 0
                let cols = entry.shape.dropFirst().first ?? 0
                let capability = SmeltKernelCapabilityRegistry.signedStorageRead(
                    format: format,
                    rows: rows,
                    cols: cols,
                    groupSize: groupSize
                )
                return SmeltPlannedWeightUse(
                    consumerID: "\(entry.name).storage",
                    weightName: entry.name,
                    capability: capability,
                    weightRole: .signed,
                    acceptedLayouts: [.signedRowMajor(format: format, groupSize: groupSize)],
                    consumerKind: .storageRead
                )
            }
        }
        return kernelUses + storageUses
    }

    static func storagePlan(
        entries: [SmeltWeightEntry],
        plannedUses: [SmeltPlannedWeightUse]
    ) -> SmeltWeightStoragePlan {
        let usesByWeight = Dictionary(grouping: plannedUses, by: \.weightName)
        let entriesByName = Dictionary(uniqueKeysWithValues: entries.map { ($0.name, $0) })
        var decisions: [SmeltWeightStorageDecision] = []
        var issues: [SmeltWeightStorageIssue] = []

        for weightName in usesByWeight.keys.sorted() {
            let uses = usesByWeight[weightName] ?? []
            guard let entry = entriesByName[weightName] else {
                issues.append(SmeltWeightStorageIssue(
                    weightName: weightName,
                    kind: .missingWeightEntry,
                    uses: uses
                ))
                continue
            }
            guard let currentLayout = currentStorageKind(for: entry) else {
                issues.append(SmeltWeightStorageIssue(
                    weightName: weightName,
                    kind: .unsupportedCurrentLayout,
                    uses: uses
                ))
                continue
            }
            let rejectedConsumers = uses.filter {
                !$0.acceptedLayouts.contains(currentLayout)
            }
            guard rejectedConsumers.isEmpty else {
                issues.append(SmeltWeightStorageIssue(
                    weightName: weightName,
                    kind: .currentLayoutRejected,
                    uses: rejectedConsumers
                ))
                continue
            }
            decisions.append(SmeltWeightStorageDecision(
                weightName: weightName,
                currentLayout: currentLayout,
                selectedLayout: currentLayout,
                uses: uses,
                requiresDuplicateLayout: false
            ))
        }

        return SmeltWeightStoragePlan(
            plannedUses: plannedUses,
            decisions: decisions,
            issues: issues
        )
    }

    static func plannedLayout(
        for ir: SmeltModelIR,
        kernelPlan: SmeltKernelPlan? = nil
    ) -> SmeltPlannedWeightLayout {
        let kernelPlan = kernelPlan ?? SmeltKernelPlanner.plan(for: ir)
        let baseEntries = SmeltWeightLayout.computeLayout(from: ir)
        return plannedLayout(
            for: ir,
            entries: baseEntries,
            kernelPlan: kernelPlan
        )
    }

    static func plannedLayout(
        for ir: SmeltModelIR,
        entries baseEntries: [SmeltWeightEntry],
        kernelPlan: SmeltKernelPlan? = nil
    ) -> SmeltPlannedWeightLayout {
        let kernelPlan = kernelPlan ?? SmeltKernelPlanner.plan(for: ir)
        return plannedLayout(
            entries: projectionBankPackedEntries(baseEntries, for: ir),
            kernelPlan: kernelPlan
        )
    }

    static func plannedLayout(
        entries baseEntries: [SmeltWeightEntry],
        kernelPlan: SmeltKernelPlan
    ) -> SmeltPlannedWeightLayout {
        let storagePlan = storagePlan(entries: baseEntries, kernelPlan: kernelPlan)
        return SmeltPlannedWeightLayout(
            entries: entries(
                baseEntries,
                applying: storagePlan
            ),
            storagePlan: storagePlan
        )
    }

    private static func entries(
        _ baseEntries: [SmeltWeightEntry],
        applying storagePlan: SmeltWeightStoragePlan
    ) -> [SmeltWeightEntry] {
        let decisionsByWeight = Dictionary(uniqueKeysWithValues: storagePlan.decisions.map {
            ($0.weightName, $0)
        })
        return baseEntries.map { entry in
            guard let decision = decisionsByWeight[entry.name] else { return entry }
            switch decision.selectedLayout {
            case .affineU4RowMajor:
                // Current generated kernels accept the existing affine row-major
                // layout. Future storage kinds should materialize their entry
                // rewrite here, keeping packers behind this planned boundary.
                return entry
            case .signedRowMajor:
                return entry
            }
        }
    }

    /// Materialize CAM-authored common-input projection banks without adding
    /// bytes. Each member keeps its own manifest entry, so an unsupported
    /// backend can still issue ordinary independent matvecs. The optimized
    /// backend sees a contiguous code slab followed by a contiguous scale slab.
    private static func projectionBankPackedEntries(
        _ baseEntries: [SmeltWeightEntry],
        for ir: SmeltModelIR
    ) -> [SmeltWeightEntry] {
        guard !ir.config.projectionBanks.isEmpty else { return baseEntries }
        var result = baseEntries

        for (layerIndex, layerType) in ir.layerPattern.expanded.enumerated() {
            for bank in ir.config.projectionBanks {
                guard let names = projectionWeightNames(
                    for: bank,
                    layerIndex: layerIndex,
                    layerType: layerType
                ) else { continue }
                result = packProjectionBank(names: names, in: result)
            }
        }
        return result
    }

    private static func projectionWeightNames(
        for bank: SmeltProjectionBankConfig,
        layerIndex: Int,
        layerType: SmeltLayerType
    ) -> [String]? {
        switch bank.source {
        case .deltaInput:
            guard layerType == .delta else { return nil }
            let prefix = "layers_\(layerIndex)_linear_attn"
            return bank.outputs.compactMap { endpoint in
                switch endpoint {
                case .deltaQKV: return "\(prefix)_in_proj_qkv_weight"
                case .deltaZ: return "\(prefix)_in_proj_z_weight"
                case .deltaA: return "\(prefix)_in_proj_a_weight"
                case .deltaB: return "\(prefix)_in_proj_b_weight"
                case .deltaOut, .attentionQ, .attentionK, .attentionV, .attentionRelative,
                     .attentionOut, .ffnGate, .ffnUp, .ffnDown, .lmHead:
                    return nil
                }
            }
        case .deltaOutput:
            guard layerType == .delta else { return nil }
            let prefix = "layers_\(layerIndex)_linear_attn"
            return bank.outputs.compactMap { endpoint in
                switch endpoint {
                case .deltaOut: return "\(prefix)_out_proj_weight"
                case .deltaQKV, .deltaZ, .deltaA, .deltaB,
                     .attentionQ, .attentionK, .attentionV, .attentionRelative, .attentionOut,
                     .ffnGate, .ffnUp, .ffnDown, .lmHead:
                    return nil
                }
            }
        case .attentionInput:
            guard layerType.isAttentionFamily else { return nil }
            let prefix = "layers_\(layerIndex)_self_attn"
            return bank.outputs.compactMap { endpoint in
                switch endpoint {
                case .attentionQ: return "\(prefix)_q_proj_weight"
                case .attentionK: return "\(prefix)_k_proj_weight"
                case .attentionV: return "\(prefix)_v_proj_weight"
                case .attentionRelative: return "\(prefix)_r_proj_weight"
                case .deltaQKV, .deltaZ, .deltaA, .deltaB, .deltaOut,
                     .attentionOut, .ffnGate, .ffnUp, .ffnDown, .lmHead:
                    return nil
                }
            }
        case .attentionOutput:
            guard layerType.isAttentionFamily else { return nil }
            let prefix = "layers_\(layerIndex)_self_attn"
            return bank.outputs.compactMap { endpoint in
                switch endpoint {
                case .attentionOut: return "\(prefix)_o_proj_weight"
                case .deltaQKV, .deltaZ, .deltaA, .deltaB, .deltaOut,
                     .attentionQ, .attentionK, .attentionV, .attentionRelative,
                     .ffnGate, .ffnUp, .ffnDown, .lmHead:
                    return nil
                }
            }
        case .ffnInput:
            let prefix = "layers_\(layerIndex)_mlp"
            return bank.outputs.compactMap { endpoint in
                switch endpoint {
                case .ffnGate: return "\(prefix)_gate_proj_weight"
                case .ffnUp: return "\(prefix)_up_proj_weight"
                case .deltaQKV, .deltaZ, .deltaA, .deltaB, .deltaOut,
                     .attentionQ, .attentionK, .attentionV, .attentionRelative, .attentionOut,
                     .ffnDown, .lmHead:
                    return nil
                }
            }
        case .ffnIntermediate:
            let prefix = "layers_\(layerIndex)_mlp"
            return bank.outputs.compactMap { endpoint in
                switch endpoint {
                case .ffnDown: return "\(prefix)_down_proj_weight"
                case .deltaQKV, .deltaZ, .deltaA, .deltaB, .deltaOut,
                     .attentionQ, .attentionK, .attentionV, .attentionRelative, .attentionOut,
                     .ffnGate, .ffnUp, .lmHead:
                    return nil
                }
            }
        case .lmHeadInput:
            // The final projection has one member and therefore needs no
            // weight-bank repacking. Its activation view still lowers through
            // the ordinary signed representation/consumer path.
            return nil
        }
    }

    private static func packProjectionBank(
        names: [String],
        in entries: [SmeltWeightEntry]
    ) -> [SmeltWeightEntry] {
        guard (2...4).contains(names.count) else { return entries }
        let indexByName = Dictionary(uniqueKeysWithValues: entries.enumerated().map {
            ($0.element.name, $0.offset)
        })
        guard let firstIndex = indexByName[names[0]] else { return entries }
        let indices = names.compactMap { indexByName[$0] }
        guard indices.count == names.count,
              indices == Array(firstIndex..<(firstIndex + names.count))
        else { return entries }
        let members = indices.map { entries[$0] }
        guard let storageDType = members.first?.dtype,
              storageDType == .binary1 || storageDType == .ternary2,
              let cols = members.first?.paddedCols,
              members.allSatisfy({ entry in
                  entry.dtype == storageDType
                      && entry.groupSize == 128
                      && entry.paddedCols == cols
                      && entry.shape.count == 2
                      && entry.scalesOffset != nil
                      && entry.scalesSizeBytes != nil
              })
        else { return entries }

        let regionStart = members[0].offset
        let regionEnd = members.reduce(regionStart) { end, entry in
            max(end, entry.scalesOffset! + entry.scalesSizeBytes!)
        }
        var codeCursor = regionStart
        var replacements: [SmeltWeightEntry] = []
        for entry in members {
            replacements.append(entry.replacingSignedOffsets(
                codes: codeCursor,
                scales: 0
            ))
            codeCursor += entry.sizeBytes
        }
        var scaleCursor = (codeCursor + 127) & ~UInt64(127)
        for index in replacements.indices {
            replacements[index] = replacements[index].replacingSignedOffsets(
                codes: replacements[index].offset,
                scales: scaleCursor
            )
            scaleCursor += replacements[index].scalesSizeBytes!
        }
        guard scaleCursor <= regionEnd else { return entries }

        var result = entries
        for (index, replacement) in zip(indices, replacements) {
            result[index] = replacement
        }
        return result
    }

    private static func currentStorageKind(for entry: SmeltWeightEntry) -> SmeltWeightStorageKind? {
        switch entry.dtype {
        case .affineU4:
            guard let groupSize = entry.groupSize else { return nil }
            return .affineU4RowMajor(groupSize: groupSize)
        case .binary1:
            guard let groupSize = entry.groupSize else { return nil }
            return .signedRowMajor(format: .binary1, groupSize: groupSize)
        case .ternary2:
            guard let groupSize = entry.groupSize else { return nil }
            return .signedRowMajor(format: .ternary2, groupSize: groupSize)
        case .bf16, .fp16, .fp32, .int32, .raw, .turboQuantH, .u4Lut:
            return nil
        }
    }
}

private extension SmeltWeightEntry {
    func replacingSignedOffsets(codes: UInt64, scales: UInt64) -> SmeltWeightEntry {
        SmeltWeightEntry(
            name: name,
            offset: codes,
            sizeBytes: sizeBytes,
            shape: shape,
            dtype: dtype,
            groupSize: groupSize,
            lutOffset: lutOffset,
            lutSizeBytes: lutSizeBytes,
            packedRowStride: packedRowStride,
            paddedCols: paddedCols,
            scalesOffset: scales,
            scalesSizeBytes: scalesSizeBytes,
            biasesOffset: biasesOffset,
            biasesSizeBytes: biasesSizeBytes,
            codebookOffset: codebookOffset,
            codebookSizeBytes: codebookSizeBytes
        )
    }
}
