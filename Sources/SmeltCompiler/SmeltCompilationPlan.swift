// First-class compiler planning artifact.
//
// This is the object model-family setup should converge on: derive legal kernel
// uses from graph + shapes + quantization, then derive storage decisions from
// those consumers.

import SmeltSchema

struct SmeltCompilationPlan: Sendable, Equatable {
    let policy: SmeltCompilationConfig
    let bufferPlan: SmeltBufferPlan
    let kernelPlan: SmeltKernelPlan
    let plannedWeightEntries: [SmeltWeightEntry]
    let weightStoragePlan: SmeltWeightStoragePlan

    var generatedSourceFunctionNames: Set<String> {
        SmeltGeneratedKernelVariants.generatedSourceFunctionNames(kernelPlan: kernelPlan)
    }

    var generatedMetalSourceSuffix: String {
        SmeltGeneratedKernelVariants.lutMatvecSuffix(kernelPlan: kernelPlan)
    }

    func validateGeneratedPipelineNames(
        _ names: [String],
        context: String
    ) throws {
        let packageLocalNames = kernelPlan.packageLocalGeneratedPipelineNames(names)
        let unplanned = kernelPlan.unplannedGeneratedPipelineNames(names)
        guard unplanned.isEmpty else {
            throw SmeltCompilerError.unsupportedConfiguration(
                "\(context) emitted generated pipeline name(s) not present in "
                    + "the kernel plan: \(unplanned.sorted().joined(separator: ", "))"
            )
        }

        let missingSource = packageLocalNames.filter {
            !generatedSourceFunctionNames.contains($0)
        }
        guard missingSource.isEmpty else {
            throw SmeltCompilerError.unsupportedConfiguration(
                "\(context) emitted generated pipeline name(s) without "
                    + "generated Metal source: \(missingSource.sorted().joined(separator: ", "))"
            )
        }
    }

    func validateGeneratedPipelineUses(
        _ uses: [SmeltNamedPipelineUse],
        context: String
    ) throws {
        try validateGeneratedPipelineNames(uses.map(\.name), context: context)

        let missingCandidates = kernelPlan.generatedPipelineUsesMissingPlannedCandidates(uses)
        guard missingCandidates.isEmpty else {
            throw SmeltCompilerError.unsupportedConfiguration(
                "\(context) emitted generated pipeline name(s) without "
                    + "planned candidate metadata: "
                    + missingCandidates.sorted().joined(separator: ", ")
            )
        }

        let mismatches = kernelPlan.unauthorizedGeneratedPipelineUses(uses).map { failure in
            if let plannedCapabilityName = failure.plannedCapabilityName {
                return "\(failure.pipelineName) <- \(failure.consumerID) "
                    + "planned \(plannedCapabilityName)"
            }
            return "\(failure.pipelineName) <- \(failure.consumerID)"
        }
        guard mismatches.isEmpty else {
            throw SmeltCompilerError.unsupportedConfiguration(
                "\(context) emitted generated pipeline name(s) not authorized "
                    + "by their planned candidates: "
                    + mismatches.sorted().joined(separator: ", ")
            )
        }
    }

    var report: SmeltCompilationPlanReport {
        let plannedKernelConsumers = kernelPlan.plannedKernelConsumerReports()
        let plannedGeneratedKernelCapabilities =
            kernelPlan.generatedKernelCapabilityReports(storageKindID: Self.storageKindID)
        let plannedGeneratedKernelNames = kernelPlan.generatedCapabilityNames
        let emittedGeneratedKernelNames = kernelPlan.emittedGeneratedCapabilityNames
        return SmeltCompilationPlanReport(
            plannedKernelUses: plannedKernelConsumers.count,
            plannedKernelConsumers: plannedKernelConsumers,
            plannedKernelCandidates: kernelPlan.generatedCandidateCount,
            unsupportedKernelCandidates: kernelPlan.unsupportedGeneratedCandidateCount,
            unsupportedKernelCandidateRecords: kernelPlan.unsupportedKernelCandidateReports(),
            plannedBufferSlots: bufferPlan.slotCount,
            plannedActivationBytes: bufferPlan.totalActivationBytes,
            generatedKernels: plannedGeneratedKernelCapabilities.count,
            emittedGeneratedKernels: emittedGeneratedKernelNames.count,
            plannedGeneratedKernelCapabilities: plannedGeneratedKernelCapabilities,
            plannedGeneratedKernelNames: plannedGeneratedKernelNames,
            emittedGeneratedKernelNames: emittedGeneratedKernelNames,
            plannedWeightUses: weightStoragePlan.plannedUses.count,
            plannedWeightNames: weightStoragePlan.plannedWeightNames,
            plannedWeightConsumerIDs: weightStoragePlan.plannedWeightConsumerIDs,
            plannedWeightConsumers: weightStoragePlan.plannedWeightConsumerReports(
                storageKindID: Self.storageKindID
            ),
            plannedWeightStorageDecisions: weightStoragePlan.decisions.count,
            plannedWeightStorageDecisionNames: weightStoragePlan.storageDecisionNames,
            plannedWeightStorageDecisionRecords: weightStoragePlan.storageDecisionReports(
                storageKindID: Self.storageKindID
            ),
            duplicateWeightLayouts: weightStoragePlan.duplicateLayoutCount,
            weightStorageIssues: weightStoragePlan.issues.count,
            weightStorageIssueNames: weightStoragePlan.issueNames,
            weightStorageIssueRecords: weightStoragePlan.issueReports(),
            memoryNeutralWeightStorage: weightStoragePlan.isMemoryNeutral,
            kernelGeneration: policy.generatedKernels.rawValue,
            generatedKernelConsumerKinds: policy.generatedKernelConsumerKindNames,
            weightLayoutPolicy: policy.weightLayout.rawValue
        )
    }

    init(
        policy: SmeltCompilationConfig,
        bufferPlan: SmeltBufferPlan,
        kernelPlan: SmeltKernelPlan,
        plannedWeightLayout: SmeltPlannedWeightLayout
    ) {
        self.policy = policy
        self.bufferPlan = bufferPlan
        self.kernelPlan = kernelPlan
        self.plannedWeightEntries = plannedWeightLayout.entries
        self.weightStoragePlan = plannedWeightLayout.storagePlan
    }

    init(
        policy: SmeltCompilationConfig,
        bufferPlan: SmeltBufferPlan,
        kernelPlan: SmeltKernelPlan,
        plannedWeightEntries: [SmeltWeightEntry],
        weightStoragePlan: SmeltWeightStoragePlan
    ) {
        self.policy = policy
        self.bufferPlan = bufferPlan
        self.kernelPlan = kernelPlan
        self.plannedWeightEntries = plannedWeightEntries
        self.weightStoragePlan = weightStoragePlan
    }

    private static func storageKindID(_ kind: SmeltWeightStorageKind) -> String {
        switch kind {
        case .affineU4RowMajor(let groupSize):
            return "affine_u4_row_major_g\(groupSize)"
        case let .signedRowMajor(format, groupSize):
            return "\(format.rawValue)_row_major_g\(groupSize)"
        }
    }
}
