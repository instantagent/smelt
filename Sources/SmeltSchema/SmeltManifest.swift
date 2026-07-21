// SmeltManifest — Package manifest for a .smeltpkg artifact.
//
// SHARED between compiler and runtime. Lives in SmeltSchema so both can
// decode/encode the manifest without depending on each other.
//
// The manifest is the contract between the compiler and the runtime.
// Written as JSON by the compiler, loaded once by the runtime at package open.
// After loading, the manifest is read-only — no mutations, no ARC churn.

import Foundation

public enum SmeltManifestValidationError: Error, CustomStringConvertible, Equatable {
    case invalid(String)

    public var description: String {
        switch self {
        case .invalid(let detail):
            return "manifest: \(detail)"
        }
    }
}

// MARK: - Manifest (top-level)

/// The .smeltpkg manifest, serialized as manifest.json.
public struct SmeltManifest: Codable, Sendable {
    /// Manifest format version (for forward compat).
    public let version: Int

    /// Historical package label. Runtime command admission and CAM package
    /// construction use the block graph, not this field.
    public let kind: String?

    /// Schema-owned runtime policy stamp (for example "text-generation").
    /// Config-built package copies require it to match the CAM spec.
    public let architecture: String?

    /// The EXPLICIT headless-trunk marker (U3, the activation-dtype axis): true iff this package is a
    /// compiled headless trunk (embeddings-in / hidden-out ports, no token-embed / LM head) — the
    /// DenseTrunkEmitter codegen path and the trunk sidecar set it. The runtime
    /// keys trunk APIs off THIS marker instead of a port dtype, so trunk identity is
    /// declared, not re-derived from activations (a non-trunk package with fp32 ports can't slip
    /// through). Defaults nil → not a headless trunk.
    public let headlessTrunkABI: Bool?

    public static func isHeadlessTrunk(headlessTrunkABI: Bool?) -> Bool {
        headlessTrunkABI == true
    }

    /// Declared block composition (docs/block-spec-plan.md). Optional:
    /// pre-block packages lack it; when present its endpoint signature is
    /// the authoritative modality.
    public let blocks: SmeltBlockGraph?

    /// Declared drive loop (phases = command-buffer scopes, emission, stop
    /// conditions); optional for pre-block packages.
    public let loop: SmeltLoopSchedule?

    /// Model identity.
    public let modelName: String

    /// Global model config snapshot — runtime uses these for bounds checks.
    public let config: SmeltManifestConfig

    /// Legacy context policy field. Current dynamic runtimes ignore this and
    /// take context from the invocation/runtime configuration.
    public let context: SmeltContextManifest?

    /// SHA-256 checksums for all package files.
    public let checksums: SmeltManifestChecksums

    /// Build provenance and reproducibility fingerprinting for this package.
    public let buildProvenance: SmeltBuildProvenance?

    /// Device requirements.
    public let device: SmeltDeviceRequirements

    /// Weight file layout.
    public let weights: SmeltWeightManifest

    /// Buffer slot table — every pre-allocated buffer the runtime creates.
    public let buffers: SmeltBufferTable

    /// Pipeline table — kernel names in pipeline index order.
    /// Index in this array IS the integer pipeline index used in generated code.
    /// The runtime resolves names to MTLComputePipelineState at load time,
    /// then generated code uses only integer indices (no string lookups at dispatch).
    public let pipelines: [String]

    /// Dynamic slot layout — tells the runtime where per-layer state lives.
    public let slotLayout: SmeltSlotLayout

    /// Optional prefill configuration (nil if model doesn't use batch prefill).
    public let prefill: SmeltPrefillManifest?

    /// Inference parameters (EOS tokens, think-skip, max tokens).
    public let inference: SmeltInferenceManifest?

    /// Package-owned decode policy. Runnable text manifests must carry it.
    public let decode: SmeltPackageSpec.DecodePolicy?

    /// Optional package-owned validation and performance gate policy.
    public let validation: SmeltPackageSpec.Validation?

    /// Compile-time optimization summary for generated dispatch tables.
    public let optimizationReport: SmeltOptimizationReport?

    public init(
        version: Int = 1,
        modelName: String,
        config: SmeltManifestConfig,
        checksums: SmeltManifestChecksums,
        buildProvenance: SmeltBuildProvenance? = nil,
        device: SmeltDeviceRequirements,
        weights: SmeltWeightManifest,
        buffers: SmeltBufferTable,
        pipelines: [String],
        slotLayout: SmeltSlotLayout,
        prefill: SmeltPrefillManifest? = nil,
        inference: SmeltInferenceManifest? = nil,
        decode: SmeltPackageSpec.DecodePolicy? = nil,
        validation: SmeltPackageSpec.Validation? = nil,
        optimizationReport: SmeltOptimizationReport? = nil
    ) {
        self.init(
            version: version,
            modelName: modelName,
            config: config,
            context: nil,
            checksums: checksums,
            buildProvenance: buildProvenance,
            device: device,
            weights: weights,
            buffers: buffers,
            pipelines: pipelines,
            slotLayout: slotLayout,
            prefill: prefill,
            inference: inference,
            decode: decode,
            validation: validation,
            optimizationReport: optimizationReport
        )
    }

    public init(
        version: Int = 1,
        kind: String? = nil,
        architecture: String? = nil,
        headlessTrunkABI: Bool? = nil,
        blocks: SmeltBlockGraph? = nil,
        loop: SmeltLoopSchedule? = nil,
        modelName: String,
        config: SmeltManifestConfig,
        context: SmeltContextManifest?,
        checksums: SmeltManifestChecksums,
        buildProvenance: SmeltBuildProvenance? = nil,
        device: SmeltDeviceRequirements,
        weights: SmeltWeightManifest,
        buffers: SmeltBufferTable,
        pipelines: [String],
        slotLayout: SmeltSlotLayout,
        prefill: SmeltPrefillManifest? = nil,
        inference: SmeltInferenceManifest? = nil,
        decode: SmeltPackageSpec.DecodePolicy? = nil,
        validation: SmeltPackageSpec.Validation? = nil,
        optimizationReport: SmeltOptimizationReport? = nil
    ) {
        self.version = version
        self.kind = kind
        self.architecture = architecture
        self.headlessTrunkABI = headlessTrunkABI
        self.blocks = blocks
        self.loop = loop
        self.modelName = modelName
        self.config = config
        self.context = context
        self.checksums = checksums
        self.buildProvenance = buildProvenance
        self.device = device
        self.weights = weights
        self.buffers = buffers
        self.pipelines = pipelines
        self.slotLayout = slotLayout
        self.prefill = prefill
        self.inference = inference
        self.decode = decode
        self.validation = validation
        self.optimizationReport = optimizationReport
    }

    public init(
        version: Int = 1,
        kind: String? = nil,
        architecture: String? = nil,
        modelName: String,
        config: SmeltManifestConfig,
        checksums: SmeltManifestChecksums,
        buildProvenance: SmeltBuildProvenance? = nil,
        device: SmeltDeviceRequirements,
        weights: SmeltWeightManifest,
        buffers: SmeltBufferTable,
        pipelines: [String],
        slotLayout: SmeltSlotLayout,
        prefill: SmeltPrefillManifest? = nil,
        inference: SmeltInferenceManifest? = nil
    ) {
        self.init(
            version: version,
            kind: kind,
            architecture: architecture,
            modelName: modelName,
            config: config,
            context: nil,
            checksums: checksums,
            buildProvenance: buildProvenance,
            device: device,
            weights: weights,
            buffers: buffers,
            pipelines: pipelines,
            slotLayout: slotLayout,
            prefill: prefill,
            inference: inference,
            validation: nil,
            optimizationReport: nil
        )
    }

    public init(
        version: Int = 1,
        modelName: String,
        config: SmeltManifestConfig,
        context: SmeltContextManifest?,
        checksums: SmeltManifestChecksums,
        buildProvenance: SmeltBuildProvenance? = nil,
        device: SmeltDeviceRequirements,
        weights: SmeltWeightManifest,
        buffers: SmeltBufferTable,
        pipelines: [String],
        slotLayout: SmeltSlotLayout,
        prefill: SmeltPrefillManifest? = nil,
        inference: SmeltInferenceManifest? = nil
    ) {
        self.init(
            version: version,
            modelName: modelName,
            config: config,
            context: context,
            checksums: checksums,
            buildProvenance: buildProvenance,
            device: device,
            weights: weights,
            buffers: buffers,
            pipelines: pipelines,
            slotLayout: slotLayout,
            prefill: prefill,
            inference: inference,
            validation: nil,
            optimizationReport: nil
        )
    }
}

public struct SmeltContextManifest: Codable, Sendable, Equatable {
    public let defaultLimit: Int
    public let maxLimit: Int

    public init(defaultLimit: Int, maxLimit: Int) {
        self.defaultLimit = defaultLimit
        self.maxLimit = maxLimit
    }

    private enum CodingKeys: String, CodingKey {
        case defaultLimit = "default_limit"
        case maxLimit = "max_limit"
    }

    private enum AlternateCodingKeys: String, CodingKey {
        case defaultLimit
        case maxLimit
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let alternate = try decoder.container(keyedBy: AlternateCodingKeys.self)
        self.defaultLimit =
            try c.decodeIfPresent(Int.self, forKey: .defaultLimit)
            ?? alternate.decode(Int.self, forKey: .defaultLimit)
        self.maxLimit =
            try c.decodeIfPresent(Int.self, forKey: .maxLimit)
            ?? alternate.decode(Int.self, forKey: .maxLimit)
    }
}

public extension SmeltManifest {
    var defaultContextLimit: Int {
        config.staticContextCapacity
    }

    var maxContextLimit: Int {
        defaultContextLimit
    }
}

public struct SmeltOptimizationReport: Codable, Sendable, Equatable {
    public let decodeRewriteCounts: [String: Int]
    public let prefillRewriteCounts: [String: Int]
    public let decodeOpportunities: [SmeltFusionOpportunitySummary]
    public let prefillOpportunities: [SmeltFusionOpportunitySummary]
    public let compilationPlan: SmeltCompilationPlanReport?

    public init(
        decodeRewriteCounts: [String: Int] = [:],
        prefillRewriteCounts: [String: Int] = [:],
        decodeOpportunities: [SmeltFusionOpportunitySummary] = [],
        prefillOpportunities: [SmeltFusionOpportunitySummary] = [],
        compilationPlan: SmeltCompilationPlanReport? = nil
    ) {
        self.decodeRewriteCounts = decodeRewriteCounts
        self.prefillRewriteCounts = prefillRewriteCounts
        self.decodeOpportunities = decodeOpportunities
        self.prefillOpportunities = prefillOpportunities
        self.compilationPlan = compilationPlan
    }

    private enum CodingKeys: String, CodingKey {
        case decodeRewriteCounts
        case prefillRewriteCounts
        case decodeOpportunities
        case prefillOpportunities
        case compilationPlan
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        decodeRewriteCounts = try values.decodeIfPresent(
            [String: Int].self,
            forKey: .decodeRewriteCounts
        ) ?? [:]
        prefillRewriteCounts = try values.decodeIfPresent(
            [String: Int].self,
            forKey: .prefillRewriteCounts
        ) ?? [:]
        decodeOpportunities = try values.decodeIfPresent(
            [SmeltFusionOpportunitySummary].self,
            forKey: .decodeOpportunities
        ) ?? []
        prefillOpportunities = try values.decodeIfPresent(
            [SmeltFusionOpportunitySummary].self,
            forKey: .prefillOpportunities
        ) ?? []
        compilationPlan = try values.decodeIfPresent(
            SmeltCompilationPlanReport.self,
            forKey: .compilationPlan
        )
    }
}

public struct SmeltCompilationPlanReport: Codable, Sendable, Equatable {
    public let plannedKernelUses: Int
    public let plannedKernelConsumers: [SmeltPlannedKernelConsumerReport]
    public let plannedKernelCandidates: Int
    public let unsupportedKernelCandidates: Int
    public let unsupportedKernelCandidateRecords: [SmeltUnsupportedKernelCandidateReport]
    public let plannedBufferSlots: Int
    public let plannedActivationBytes: Int
    public let generatedKernels: Int
    public let emittedGeneratedKernels: Int
    public let plannedGeneratedKernelCapabilities: [SmeltGeneratedKernelCapabilityReport]
    public let plannedGeneratedKernelNames: [String]
    public let emittedGeneratedKernelNames: [String]
    public let plannedWeightUses: Int
    public let plannedWeightNames: [String]
    public let plannedWeightConsumerIDs: [String]
    public let plannedWeightConsumers: [SmeltPlannedWeightConsumerReport]
    public let plannedWeightStorageDecisions: Int
    public let plannedWeightStorageDecisionNames: [String]
    public let plannedWeightStorageDecisionRecords: [SmeltPlannedWeightStorageDecisionReport]
    public let duplicateWeightLayouts: Int
    public let weightStorageIssues: Int
    public let weightStorageIssueNames: [String]
    public let weightStorageIssueRecords: [SmeltPlannedWeightStorageIssueReport]
    public let memoryNeutralWeightStorage: Bool
    public let kernelGeneration: String
    public let generatedKernelConsumerKinds: [String]
    public let weightLayoutPolicy: String

    public init(
        plannedKernelUses: Int,
        plannedKernelConsumers: [SmeltPlannedKernelConsumerReport] = [],
        plannedKernelCandidates: Int = 0,
        unsupportedKernelCandidates: Int = 0,
        unsupportedKernelCandidateRecords: [SmeltUnsupportedKernelCandidateReport] = [],
        plannedBufferSlots: Int = 0,
        plannedActivationBytes: Int = 0,
        generatedKernels: Int,
        emittedGeneratedKernels: Int? = nil,
        plannedGeneratedKernelCapabilities: [SmeltGeneratedKernelCapabilityReport] = [],
        plannedGeneratedKernelNames: [String] = [],
        emittedGeneratedKernelNames: [String] = [],
        plannedWeightUses: Int,
        plannedWeightNames: [String] = [],
        plannedWeightConsumerIDs: [String] = [],
        plannedWeightConsumers: [SmeltPlannedWeightConsumerReport] = [],
        plannedWeightStorageDecisions: Int,
        plannedWeightStorageDecisionNames: [String] = [],
        plannedWeightStorageDecisionRecords: [SmeltPlannedWeightStorageDecisionReport] = [],
        duplicateWeightLayouts: Int,
        weightStorageIssues: Int,
        weightStorageIssueNames: [String] = [],
        weightStorageIssueRecords: [SmeltPlannedWeightStorageIssueReport] = [],
        memoryNeutralWeightStorage: Bool,
        kernelGeneration: String = "auto",
        generatedKernelConsumerKinds: [String],
        weightLayoutPolicy: String = "memory_neutral"
    ) {
        self.plannedKernelUses = plannedKernelUses
        self.plannedKernelConsumers = plannedKernelConsumers
        self.plannedKernelCandidates = plannedKernelCandidates
        self.unsupportedKernelCandidates = unsupportedKernelCandidates
        self.unsupportedKernelCandidateRecords = unsupportedKernelCandidateRecords
        self.plannedBufferSlots = plannedBufferSlots
        self.plannedActivationBytes = plannedActivationBytes
        self.generatedKernels = generatedKernels
        self.emittedGeneratedKernels = emittedGeneratedKernels ?? generatedKernels
        self.plannedGeneratedKernelCapabilities = plannedGeneratedKernelCapabilities
        self.plannedGeneratedKernelNames = plannedGeneratedKernelNames
        self.emittedGeneratedKernelNames = emittedGeneratedKernelNames
        self.plannedWeightUses = plannedWeightUses
        self.plannedWeightNames = plannedWeightNames
        self.plannedWeightConsumerIDs = plannedWeightConsumerIDs
        self.plannedWeightConsumers = plannedWeightConsumers
        self.plannedWeightStorageDecisions = plannedWeightStorageDecisions
        self.plannedWeightStorageDecisionNames = plannedWeightStorageDecisionNames
        self.plannedWeightStorageDecisionRecords = plannedWeightStorageDecisionRecords
        self.duplicateWeightLayouts = duplicateWeightLayouts
        self.weightStorageIssues = weightStorageIssues
        self.weightStorageIssueNames = weightStorageIssueNames
        self.weightStorageIssueRecords = weightStorageIssueRecords
        self.memoryNeutralWeightStorage = memoryNeutralWeightStorage
        self.kernelGeneration = kernelGeneration
        self.generatedKernelConsumerKinds = generatedKernelConsumerKinds
        self.weightLayoutPolicy = weightLayoutPolicy
    }

    private enum CodingKeys: String, CodingKey {
        case plannedKernelUses
        case plannedKernelConsumers
        case plannedKernelCandidates
        case unsupportedKernelCandidates
        case unsupportedKernelCandidateRecords
        case plannedBufferSlots
        case plannedActivationBytes
        case generatedKernels
        case emittedGeneratedKernels
        case plannedGeneratedKernelCapabilities
        case plannedGeneratedKernelNames
        case emittedGeneratedKernelNames
        case plannedWeightUses
        case plannedWeightNames
        case plannedWeightConsumerIDs
        case plannedWeightConsumers
        case plannedWeightStorageDecisions
        case plannedWeightStorageDecisionNames
        case plannedWeightStorageDecisionRecords
        case duplicateWeightLayouts
        case weightStorageIssues
        case weightStorageIssueNames
        case weightStorageIssueRecords
        case memoryNeutralWeightStorage
        case kernelGeneration
        case generatedKernelConsumerKinds
        case weightLayoutPolicy
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        plannedKernelUses = try values.decode(Int.self, forKey: .plannedKernelUses)
        plannedKernelConsumers = try values.decodeIfPresent(
            [SmeltPlannedKernelConsumerReport].self,
            forKey: .plannedKernelConsumers
        ) ?? []
        plannedKernelCandidates = try values.decodeIfPresent(
            Int.self,
            forKey: .plannedKernelCandidates
        ) ?? plannedKernelUses
        unsupportedKernelCandidates = try values.decodeIfPresent(
            Int.self,
            forKey: .unsupportedKernelCandidates
        ) ?? 0
        unsupportedKernelCandidateRecords = try values.decodeIfPresent(
            [SmeltUnsupportedKernelCandidateReport].self,
            forKey: .unsupportedKernelCandidateRecords
        ) ?? []
        plannedBufferSlots = try values.decodeIfPresent(
            Int.self,
            forKey: .plannedBufferSlots
        ) ?? 0
        plannedActivationBytes = try values.decodeIfPresent(
            Int.self,
            forKey: .plannedActivationBytes
        ) ?? 0
        generatedKernels = try values.decode(Int.self, forKey: .generatedKernels)
        emittedGeneratedKernels = try values.decodeIfPresent(
            Int.self,
            forKey: .emittedGeneratedKernels
        ) ?? generatedKernels
        plannedGeneratedKernelCapabilities = try values.decodeIfPresent(
            [SmeltGeneratedKernelCapabilityReport].self,
            forKey: .plannedGeneratedKernelCapabilities
        ) ?? []
        plannedGeneratedKernelNames = try values.decodeIfPresent(
            [String].self,
            forKey: .plannedGeneratedKernelNames
        ) ?? []
        emittedGeneratedKernelNames = try values.decodeIfPresent(
            [String].self,
            forKey: .emittedGeneratedKernelNames
        ) ?? []
        plannedWeightUses = try values.decode(Int.self, forKey: .plannedWeightUses)
        plannedWeightNames = try values.decodeIfPresent(
            [String].self,
            forKey: .plannedWeightNames
        ) ?? []
        plannedWeightConsumerIDs = try values.decodeIfPresent(
            [String].self,
            forKey: .plannedWeightConsumerIDs
        ) ?? []
        plannedWeightConsumers = try values.decodeIfPresent(
            [SmeltPlannedWeightConsumerReport].self,
            forKey: .plannedWeightConsumers
        ) ?? []
        plannedWeightStorageDecisions = try values.decode(
            Int.self,
            forKey: .plannedWeightStorageDecisions
        )
        plannedWeightStorageDecisionNames = try values.decodeIfPresent(
            [String].self,
            forKey: .plannedWeightStorageDecisionNames
        ) ?? []
        plannedWeightStorageDecisionRecords = try values.decodeIfPresent(
            [SmeltPlannedWeightStorageDecisionReport].self,
            forKey: .plannedWeightStorageDecisionRecords
        ) ?? []
        duplicateWeightLayouts = try values.decode(Int.self, forKey: .duplicateWeightLayouts)
        weightStorageIssues = try values.decode(Int.self, forKey: .weightStorageIssues)
        weightStorageIssueNames = try values.decodeIfPresent(
            [String].self,
            forKey: .weightStorageIssueNames
        ) ?? []
        weightStorageIssueRecords = try values.decodeIfPresent(
            [SmeltPlannedWeightStorageIssueReport].self,
            forKey: .weightStorageIssueRecords
        ) ?? []
        memoryNeutralWeightStorage = try values.decode(
            Bool.self,
            forKey: .memoryNeutralWeightStorage
        )
        kernelGeneration = try values.decodeIfPresent(
            String.self,
            forKey: .kernelGeneration
        ) ?? "auto"
        generatedKernelConsumerKinds = try values.decode(
            [String].self,
            forKey: .generatedKernelConsumerKinds
        )
        weightLayoutPolicy = try values.decodeIfPresent(
            String.self,
            forKey: .weightLayoutPolicy
        ) ?? "memory_neutral"
    }
}

public struct SmeltUnsupportedKernelCandidateReport: Codable, Sendable, Equatable {
    public let consumerID: String
    public let consumerKind: String?
    public let phase: String
    public let operation: String
    public let rows: Int
    public let cols: Int
    public let groupSize: Int
    public let weights: [SmeltPlannedKernelWeightReport]
    public let reason: String

    public init(
        consumerID: String,
        consumerKind: String? = nil,
        phase: String,
        operation: String,
        rows: Int,
        cols: Int,
        groupSize: Int,
        weights: [SmeltPlannedKernelWeightReport],
        reason: String
    ) {
        self.consumerID = consumerID
        self.consumerKind = consumerKind
        self.phase = phase
        self.operation = operation
        self.rows = rows
        self.cols = cols
        self.groupSize = groupSize
        self.weights = weights
        self.reason = reason
    }
}

public struct SmeltGeneratedKernelCapabilityReport: Codable, Sendable, Equatable {
    public let capabilityName: String
    public let phase: String
    public let operation: String
    public let rows: Int
    public let cols: Int
    public let groupSize: Int
    public let sourceKind: String
    public let emittedGeneratedSource: Bool
    public let sourceTemplate: String?
    public let weightRequirements: [SmeltGeneratedKernelWeightRequirementReport]
    public let rowTile: Int?
    public let batchTile: Int?
    public let threadgroupWidth: Int?

    public init(
        capabilityName: String,
        phase: String,
        operation: String,
        rows: Int,
        cols: Int,
        groupSize: Int,
        sourceKind: String,
        emittedGeneratedSource: Bool,
        sourceTemplate: String? = nil,
        weightRequirements: [SmeltGeneratedKernelWeightRequirementReport],
        rowTile: Int?,
        batchTile: Int?,
        threadgroupWidth: Int?
    ) {
        self.capabilityName = capabilityName
        self.phase = phase
        self.operation = operation
        self.rows = rows
        self.cols = cols
        self.groupSize = groupSize
        self.sourceKind = sourceKind
        self.emittedGeneratedSource = emittedGeneratedSource
        self.sourceTemplate = sourceTemplate
        self.weightRequirements = weightRequirements
        self.rowTile = rowTile
        self.batchTile = batchTile
        self.threadgroupWidth = threadgroupWidth
    }
}

public struct SmeltGeneratedKernelWeightRequirementReport: Codable, Sendable, Equatable {
    public let role: String
    public let acceptedLayouts: [String]

    public init(role: String, acceptedLayouts: [String]) {
        self.role = role
        self.acceptedLayouts = acceptedLayouts
    }
}

public struct SmeltPlannedKernelConsumerReport: Codable, Sendable, Equatable {
    public let consumerID: String
    public let consumerKind: String?
    public let capabilityName: String
    public let phase: String
    public let operation: String
    public let rows: Int
    public let cols: Int
    public let groupSize: Int
    public let weights: [SmeltPlannedKernelWeightReport]

    public init(
        consumerID: String,
        consumerKind: String? = nil,
        capabilityName: String,
        phase: String,
        operation: String,
        rows: Int,
        cols: Int,
        groupSize: Int,
        weights: [SmeltPlannedKernelWeightReport]
    ) {
        self.consumerID = consumerID
        self.consumerKind = consumerKind
        self.capabilityName = capabilityName
        self.phase = phase
        self.operation = operation
        self.rows = rows
        self.cols = cols
        self.groupSize = groupSize
        self.weights = weights
    }
}

public struct SmeltPlannedKernelWeightReport: Codable, Sendable, Equatable {
    public let weightName: String
    public let role: String

    public init(weightName: String, role: String) {
        self.weightName = weightName
        self.role = role
    }
}

public struct SmeltPlannedWeightConsumerReport: Codable, Sendable, Equatable {
    public let weightName: String
    public let consumerID: String
    public let consumerKind: String?
    public let capabilityName: String
    public let weightRole: String
    public let acceptedLayouts: [String]

    public init(
        weightName: String,
        consumerID: String,
        consumerKind: String? = nil,
        capabilityName: String,
        weightRole: String,
        acceptedLayouts: [String]
    ) {
        self.weightName = weightName
        self.consumerID = consumerID
        self.consumerKind = consumerKind
        self.capabilityName = capabilityName
        self.weightRole = weightRole
        self.acceptedLayouts = acceptedLayouts
    }
}

public struct SmeltPlannedWeightStorageDecisionReport: Codable, Sendable, Equatable {
    public let weightName: String
    public let currentLayout: String
    public let selectedLayout: String
    public let consumers: [SmeltPlannedWeightStorageDecisionConsumerReport]
    public let requiresDuplicateLayout: Bool

    public init(
        weightName: String,
        currentLayout: String,
        selectedLayout: String,
        consumers: [SmeltPlannedWeightStorageDecisionConsumerReport],
        requiresDuplicateLayout: Bool
    ) {
        self.weightName = weightName
        self.currentLayout = currentLayout
        self.selectedLayout = selectedLayout
        self.consumers = consumers
        self.requiresDuplicateLayout = requiresDuplicateLayout
    }
}

public struct SmeltPlannedWeightStorageDecisionConsumerReport: Codable, Sendable, Equatable {
    public let consumerID: String
    public let consumerKind: String?

    public init(
        consumerID: String,
        consumerKind: String? = nil
    ) {
        self.consumerID = consumerID
        self.consumerKind = consumerKind
    }
}

public struct SmeltPlannedWeightStorageIssueReport: Codable, Sendable, Equatable {
    public let weightName: String
    public let kind: String
    public let consumers: [SmeltPlannedWeightStorageDecisionConsumerReport]

    public init(
        weightName: String,
        kind: String,
        consumers: [SmeltPlannedWeightStorageDecisionConsumerReport]
    ) {
        self.weightName = weightName
        self.kind = kind
        self.consumers = consumers
    }
}

public struct SmeltFusionOpportunitySummary: Codable, Sendable, Equatable {
    public let pattern: String
    public let shape: String
    public let count: Int
    public let fusedKernelAvailable: Bool

    public init(
        pattern: String,
        shape: String,
        count: Int,
        fusedKernelAvailable: Bool
    ) {
        self.pattern = pattern
        self.shape = shape
        self.count = count
        self.fusedKernelAvailable = fusedKernelAvailable
    }
}

// MARK: - Slot layout (dynamic region metadata)

/// Tells the runtime where dynamic per-layer slots are located.
/// This is the contract between the buffer plan and the runtime loader.
public struct SmeltSlotLayout: Codable, Sendable {
    public let convStateBaseSlot: Int
    public let recStateBaseSlot: Int
    public let keyCacheBaseSlot: Int
    public let valCacheBaseSlot: Int
    public let ropeCosSlot: Int
    public let ropeSinSlot: Int
    public let ropeTablePairs: [SmeltRoPETablePairManifest]
    public let tokenIdSlot: Int
    public let positionSlot: Int
    /// Slot index for the monolithic weight buffer (mmap'd weights.bin).
    public let weightsSlot: Int

    public init(
        convStateBaseSlot: Int,
        recStateBaseSlot: Int,
        keyCacheBaseSlot: Int,
        valCacheBaseSlot: Int,
        ropeCosSlot: Int,
        ropeSinSlot: Int,
        ropeTablePairs: [SmeltRoPETablePairManifest] = [],
        tokenIdSlot: Int,
        positionSlot: Int,
        weightsSlot: Int
    ) {
        self.convStateBaseSlot = convStateBaseSlot
        self.recStateBaseSlot = recStateBaseSlot
        self.keyCacheBaseSlot = keyCacheBaseSlot
        self.valCacheBaseSlot = valCacheBaseSlot
        self.ropeCosSlot = ropeCosSlot
        self.ropeSinSlot = ropeSinSlot
        self.ropeTablePairs = ropeTablePairs
        self.tokenIdSlot = tokenIdSlot
        self.positionSlot = positionSlot
        self.weightsSlot = weightsSlot
    }

    public init(
        convStateBaseSlot: Int,
        recStateBaseSlot: Int,
        keyCacheBaseSlot: Int,
        valCacheBaseSlot: Int,
        ropeCosSlot: Int,
        ropeSinSlot: Int,
        tokenIdSlot: Int,
        positionSlot: Int,
        weightsSlot: Int
    ) {
        self.init(
            convStateBaseSlot: convStateBaseSlot,
            recStateBaseSlot: recStateBaseSlot,
            keyCacheBaseSlot: keyCacheBaseSlot,
            valCacheBaseSlot: valCacheBaseSlot,
            ropeCosSlot: ropeCosSlot,
            ropeSinSlot: ropeSinSlot,
            ropeTablePairs: [],
            tokenIdSlot: tokenIdSlot,
            positionSlot: positionSlot,
            weightsSlot: weightsSlot
        )
    }

    private enum CodingKeys: String, CodingKey {
        case convStateBaseSlot = "conv_state_base"
        case recStateBaseSlot = "rec_state_base"
        case keyCacheBaseSlot = "key_cache_base"
        case valCacheBaseSlot = "val_cache_base"
        case ropeCosSlot = "rope_cos"
        case ropeSinSlot = "rope_sin"
        case ropeTablePairs = "rope_table_pairs"
        case tokenIdSlot = "token_id"
        case positionSlot = "position"
        case weightsSlot = "weights"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.convStateBaseSlot = try container.decode(Int.self, forKey: .convStateBaseSlot)
        self.recStateBaseSlot = try container.decode(Int.self, forKey: .recStateBaseSlot)
        self.keyCacheBaseSlot = try container.decode(Int.self, forKey: .keyCacheBaseSlot)
        self.valCacheBaseSlot = try container.decode(Int.self, forKey: .valCacheBaseSlot)
        self.ropeCosSlot = try container.decode(Int.self, forKey: .ropeCosSlot)
        self.ropeSinSlot = try container.decode(Int.self, forKey: .ropeSinSlot)
        self.ropeTablePairs = try container.decodeIfPresent(
            [SmeltRoPETablePairManifest].self,
            forKey: .ropeTablePairs
        ) ?? []
        self.tokenIdSlot = try container.decode(Int.self, forKey: .tokenIdSlot)
        self.positionSlot = try container.decode(Int.self, forKey: .positionSlot)
        self.weightsSlot = try container.decode(Int.self, forKey: .weightsSlot)
    }
}

public struct SmeltRoPETablePairManifest: Codable, Sendable, Hashable {
    public let theta: Float
    public let dim: Int
    /// Denominator for rotary frequency computation. Nil means standard RoPE
    /// with frequencies based on `dim`.
    public let freqDim: Int?
    /// Optional frequency scaling applied before table materialization.
    public let scaling: SmeltRoPEScaling?
    /// RoPE rotation convention: "interleaved" for adjacent pairs, "split_half"
    /// for the Transformers rotate_half layout.
    public let layout: String
    public let cosSlot: Int
    public let sinSlot: Int

    public init(
        theta: Float,
        dim: Int,
        freqDim: Int? = nil,
        scaling: SmeltRoPEScaling? = nil,
        layout: String = "interleaved",
        cosSlot: Int,
        sinSlot: Int
    ) {
        self.theta = theta
        self.dim = dim
        self.freqDim = freqDim
        self.scaling = scaling
        self.layout = layout
        self.cosSlot = cosSlot
        self.sinSlot = sinSlot
    }

    private enum CodingKeys: String, CodingKey {
        case theta
        case dim
        case freqDim = "freq_dim"
        case scaling
        case layout
        case cosSlot = "cos_slot"
        case sinSlot = "sin_slot"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        theta = try container.decode(Float.self, forKey: .theta)
        dim = try container.decode(Int.self, forKey: .dim)
        freqDim = try container.decodeIfPresent(Int.self, forKey: .freqDim)
        scaling = try container.decodeIfPresent(SmeltRoPEScaling.self, forKey: .scaling)
        layout = try container.decodeIfPresent(String.self, forKey: .layout) ?? "interleaved"
        cosSlot = try container.decode(Int.self, forKey: .cosSlot)
        sinSlot = try container.decode(Int.self, forKey: .sinSlot)
    }
}

// MARK: - Config snapshot

/// Subset of config the runtime needs for dynamic bounds checks.
public struct SmeltManifestLayerPattern: Codable, Sendable, Equatable {
    public let pattern: [String]
    public let repeats: Int

    public init(pattern: [String], repeats: Int) {
        self.pattern = pattern
        self.repeats = repeats
    }
}

public struct SmeltManifestRoleAttentionConfig: Codable, Sendable, Equatable {
    public let qHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let qkNorm: Bool
    public let vNorm: Bool
    public let ropeTheta: Double
    public let ropeDim: Int
    public let slidingWindow: Int

    public init(
        qHeads: Int,
        kvHeads: Int,
        headDim: Int,
        qkNorm: Bool,
        vNorm: Bool,
        ropeTheta: Double,
        ropeDim: Int,
        slidingWindow: Int
    ) {
        self.qHeads = qHeads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.qkNorm = qkNorm
        self.vNorm = vNorm
        self.ropeTheta = ropeTheta
        self.ropeDim = ropeDim
        self.slidingWindow = slidingWindow
    }
}

public struct SmeltManifestPerLayerInputConfig: Codable, Sendable, Equatable {
    public let hiddenSize: Int
    public let vocabSize: Int

    public init(hiddenSize: Int, vocabSize: Int) {
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
    }
}

public struct SmeltManifestInputFusionConfig: Codable, Sendable, Equatable {
    public let sourceWidth: Int
    public let sourceCount: Int
    public let normalizeSources: Bool
    public let postProjectionWidth: Int?

    public init(
        sourceWidth: Int,
        sourceCount: Int,
        normalizeSources: Bool,
        postProjectionWidth: Int?
    ) {
        self.sourceWidth = sourceWidth
        self.sourceCount = sourceCount
        self.normalizeSources = normalizeSources
        self.postProjectionWidth = postProjectionWidth
    }
}

public struct SmeltManifestConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let numLayers: Int
    public let vocabSize: Int
    public let hiddenActivation: String?
    public let staticSeqCapacity: Int?
    public let ropeDim: Int
    public let numDeltaLayers: Int
    public let numAttnLayers: Int
    public let deltaNumHeads: Int
    public let deltaQKVDim: Int
    public let attnQProjDim: Int
    public let attnKProjDim: Int
    public let attnVProjDim: Int
    public let attnOutDim: Int
    public let ffnDim: Int
    public let blockTopology: String?
    public let layerPattern: SmeltManifestLayerPattern?
    public let attentionByRole: [String: SmeltManifestRoleAttentionConfig]
    public let perLayerInput: SmeltManifestPerLayerInputConfig?
    public let logitCap: Float?
    /// Trailing layer count whose K/V cache slot is allocated but
    /// never written — those layers cross-attend to an earlier
    /// non-shared layer's K/V instead. Consumers that read a
    /// last-layer K/V slot must walk back to the slot the target
    /// actually fills, since reading the shared layer's own slot
    /// would hand back zeros.
    ///
    /// Optional so we can distinguish a manifest predating this field
    /// (nil — "0 shared" and "unknown shared" diverge for any
    /// shared-K/V topology) from an explicit 0 (no sharing). Compilers
    /// since this field landed always populate it.
    public let sharedKVLayers: Int?
    public let turboQuantHPatterns: [String]
    public let inputFusion: SmeltManifestInputFusionConfig?

    public init(
        hiddenSize: Int,
        numLayers: Int,
        vocabSize: Int,
        hiddenActivation: String? = nil,
        staticSeqCapacity: Int? = nil,
        ropeDim: Int,
        numDeltaLayers: Int,
        numAttnLayers: Int,
        deltaNumHeads: Int = 0,
        deltaQKVDim: Int = 0,
        attnQProjDim: Int = 0,
        attnKProjDim: Int = 0,
        attnVProjDim: Int = 0,
        attnOutDim: Int = 0,
        ffnDim: Int = 0,
        blockTopology: String? = nil,
        layerPattern: SmeltManifestLayerPattern? = nil,
        attentionByRole: [String: SmeltManifestRoleAttentionConfig] = [:],
        perLayerInput: SmeltManifestPerLayerInputConfig? = nil,
        logitCap: Float? = nil,
        sharedKVLayers: Int? = nil,
        turboQuantHPatterns: [String] = [],
        inputFusion: SmeltManifestInputFusionConfig? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.vocabSize = vocabSize
        self.hiddenActivation = hiddenActivation
        self.staticSeqCapacity = staticSeqCapacity
        self.ropeDim = ropeDim
        self.numDeltaLayers = numDeltaLayers
        self.numAttnLayers = numAttnLayers
        self.deltaNumHeads = deltaNumHeads
        self.deltaQKVDim = deltaQKVDim
        self.attnQProjDim = attnQProjDim
        self.attnKProjDim = attnKProjDim
        self.attnVProjDim = attnVProjDim
        self.attnOutDim = attnOutDim
        self.ffnDim = ffnDim
        self.blockTopology = blockTopology
        self.layerPattern = layerPattern
        self.attentionByRole = attentionByRole
        self.perLayerInput = perLayerInput
        self.logitCap = logitCap
        self.sharedKVLayers = sharedKVLayers
        self.turboQuantHPatterns = turboQuantHPatterns
        self.inputFusion = inputFusion
    }

    private enum CodingKeys: String, CodingKey {
        case hiddenSize
        case numLayers
        case vocabSize
        case hiddenActivation
        case staticSeqCapacity
        case ropeDim
        case numDeltaLayers
        case numAttnLayers
        case deltaNumHeads
        case deltaQKVDim
        case attnQProjDim
        case attnKProjDim
        case attnVProjDim
        case attnOutDim
        case ffnDim
        case blockTopology
        case layerPattern
        case attentionByRole
        case perLayerInput
        case logitCap
        case sharedKVLayers
        case turboQuantHPatterns
        case inputFusion
    }

    private enum LegacyCodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numLayers = "num_layers"
        case vocabSize = "vocab_size"
        case staticSeqCapacity = "static_seq_capacity"
        case maxSeqLen = "max_seq_len"
        case ropeDim = "rope_dim"
        case numDeltaLayers = "num_delta_layers"
        case numAttnLayers = "num_attn_layers"
        case deltaNumHeads = "delta_num_heads"
        case deltaQKVDim = "delta_qkv_dim"
        case attnQProjDim = "attn_q_proj_dim"
        case attnKProjDim = "attn_k_proj_dim"
        case attnVProjDim = "attn_v_proj_dim"
        case attnOutDim = "attn_out_dim"
        case ffnDim = "ffn_dim"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let legacy = try decoder.container(keyedBy: LegacyCodingKeys.self)
        self.hiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .hiddenSize)
            ?? legacy.decode(Int.self, forKey: .hiddenSize)
        self.numLayers =
            try c.decodeIfPresent(Int.self, forKey: .numLayers)
            ?? legacy.decode(Int.self, forKey: .numLayers)
        self.vocabSize =
            try c.decodeIfPresent(Int.self, forKey: .vocabSize)
            ?? legacy.decode(Int.self, forKey: .vocabSize)
        self.hiddenActivation = try c.decodeIfPresent(String.self, forKey: .hiddenActivation)
        self.staticSeqCapacity =
            try c.decodeIfPresent(Int.self, forKey: .staticSeqCapacity)
            ?? legacy.decodeIfPresent(Int.self, forKey: .staticSeqCapacity)
            ?? legacy.decodeIfPresent(Int.self, forKey: .maxSeqLen)
        self.ropeDim =
            try c.decodeIfPresent(Int.self, forKey: .ropeDim)
            ?? legacy.decodeIfPresent(Int.self, forKey: .ropeDim)
            ?? 0
        self.numDeltaLayers =
            try c.decodeIfPresent(Int.self, forKey: .numDeltaLayers)
            ?? legacy.decode(Int.self, forKey: .numDeltaLayers)
        self.numAttnLayers =
            try c.decodeIfPresent(Int.self, forKey: .numAttnLayers)
            ?? legacy.decode(Int.self, forKey: .numAttnLayers)
        self.deltaNumHeads =
            try c.decodeIfPresent(Int.self, forKey: .deltaNumHeads)
            ?? legacy.decodeIfPresent(Int.self, forKey: .deltaNumHeads)
            ?? 0
        self.deltaQKVDim =
            try c.decodeIfPresent(Int.self, forKey: .deltaQKVDim)
            ?? legacy.decodeIfPresent(Int.self, forKey: .deltaQKVDim)
            ?? 0
        self.attnQProjDim =
            try c.decodeIfPresent(Int.self, forKey: .attnQProjDim)
            ?? legacy.decodeIfPresent(Int.self, forKey: .attnQProjDim)
            ?? 0
        self.attnKProjDim =
            try c.decodeIfPresent(Int.self, forKey: .attnKProjDim)
            ?? legacy.decodeIfPresent(Int.self, forKey: .attnKProjDim)
            ?? 0
        self.attnVProjDim =
            try c.decodeIfPresent(Int.self, forKey: .attnVProjDim)
            ?? legacy.decodeIfPresent(Int.self, forKey: .attnVProjDim)
            ?? 0
        self.attnOutDim =
            try c.decodeIfPresent(Int.self, forKey: .attnOutDim)
            ?? legacy.decodeIfPresent(Int.self, forKey: .attnOutDim)
            ?? 0
        self.ffnDim =
            try c.decodeIfPresent(Int.self, forKey: .ffnDim)
            ?? legacy.decodeIfPresent(Int.self, forKey: .ffnDim)
            ?? 0
        self.blockTopology = try c.decodeIfPresent(String.self, forKey: .blockTopology)
        self.layerPattern = try c.decodeIfPresent(
            SmeltManifestLayerPattern.self,
            forKey: .layerPattern
        )
        self.attentionByRole = try c.decodeIfPresent(
            [String: SmeltManifestRoleAttentionConfig].self,
            forKey: .attentionByRole
        ) ?? [:]
        self.perLayerInput = try c.decodeIfPresent(
            SmeltManifestPerLayerInputConfig.self,
            forKey: .perLayerInput
        )
        self.logitCap = try c.decodeIfPresent(Float.self, forKey: .logitCap)
        self.sharedKVLayers =
            try c.decodeIfPresent(Int.self, forKey: .sharedKVLayers)
        self.turboQuantHPatterns = try c.decodeIfPresent(
            [String].self,
            forKey: .turboQuantHPatterns
        ) ?? []
        self.inputFusion = try c.decodeIfPresent(
            SmeltManifestInputFusionConfig.self,
            forKey: .inputFusion
        )
    }

    public var staticContextCapacity: Int {
        max(staticSeqCapacity ?? 1, 1)
    }
}

// MARK: - Checksums

/// SHA-256 hex digests for integrity validation.
public struct SmeltManifestChecksums: Codable, Sendable {
    public let weightsBin: String
    public let metallib: String
    public let generatedSwift: String
    public let dispatchesBin: String
    public let prefillDispatchesBin: String?
    public let prefillVerifyArgmaxDispatchesBin: String?
    public let tokenizerJSON: String?

    public init(
        weightsBin: String,
        metallib: String,
        generatedSwift: String,
        dispatchesBin: String = "",
        prefillDispatchesBin: String? = nil,
        prefillVerifyArgmaxDispatchesBin: String? = nil,
        tokenizerJSON: String? = nil
    ) {
        self.weightsBin = weightsBin
        self.metallib = metallib
        self.generatedSwift = generatedSwift
        self.dispatchesBin = dispatchesBin
        self.prefillDispatchesBin = prefillDispatchesBin
        self.prefillVerifyArgmaxDispatchesBin = prefillVerifyArgmaxDispatchesBin
        self.tokenizerJSON = tokenizerJSON
    }

    private enum CodingKeys: String, CodingKey {
        case weightsBin = "weights_bin"
        case metallib
        case generatedSwift = "generated_swift"
        case dispatchesBin = "dispatches_bin"
        case prefillDispatchesBin = "prefill_dispatches_bin"
        case prefillVerifyArgmaxDispatchesBin = "prefill_verify_argmax_dispatches_bin"
        case tokenizerJSON = "tokenizer_json"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.weightsBin = try c.decodeIfPresent(String.self, forKey: .weightsBin) ?? ""
        self.metallib = try c.decodeIfPresent(String.self, forKey: .metallib) ?? ""
        self.generatedSwift = try c.decodeIfPresent(String.self, forKey: .generatedSwift) ?? ""
        self.dispatchesBin = try c.decodeIfPresent(String.self, forKey: .dispatchesBin) ?? ""
        self.prefillDispatchesBin = try c.decodeIfPresent(
            String.self,
            forKey: .prefillDispatchesBin
        )
        self.prefillVerifyArgmaxDispatchesBin = try c.decodeIfPresent(
            String.self,
            forKey: .prefillVerifyArgmaxDispatchesBin
        )
        self.tokenizerJSON = try c.decodeIfPresent(String.self, forKey: .tokenizerJSON)
    }
}

public struct SmeltBuildProvenance: Codable, Sendable {
    public let buildFingerprint: String
    public let weightsFingerprint: String
    public let specSHA256: String
    public let compilerSourcesSHA256: String
    /// Fingerprint of only the sources that can change the bytes written to
    /// weights.bin. Older manifests omit this and conservatively fall back to
    /// compilerSourcesSHA256 when deciding whether weights may be reused.
    public let weightBuilderSourcesSHA256: String?
    public let shaderSourcesSHA256: String
    public let resolvedOptions: SmeltResolvedBuildOptions

    public init(
        buildFingerprint: String,
        weightsFingerprint: String,
        specSHA256: String,
        compilerSourcesSHA256: String,
        weightBuilderSourcesSHA256: String? = nil,
        shaderSourcesSHA256: String,
        resolvedOptions: SmeltResolvedBuildOptions
    ) {
        self.buildFingerprint = buildFingerprint
        self.weightsFingerprint = weightsFingerprint
        self.specSHA256 = specSHA256
        self.compilerSourcesSHA256 = compilerSourcesSHA256
        self.weightBuilderSourcesSHA256 = weightBuilderSourcesSHA256
        self.shaderSourcesSHA256 = shaderSourcesSHA256
        self.resolvedOptions = resolvedOptions
    }
}

public struct SmeltResolvedBuildOptions: Codable, Sendable {
    public let layerPatternUnit: [String]
    public let layerPatternRepeats: Int
    public let quantizationStrategy: String
    public let groupSize: Int
    public let excludePatterns: [String]
    public let quantizeEmbedding: Bool
    public let loadingStrategy: String
    public let packing: String
    public let checkpointMap: String?
    public let prefillEngine: String?
    public let maxPrefillBatch: Int?
    public let prefillHandoffFamilies: [String]
    /// Whether the prefill kernel emits lm_head + logit_cap for
    /// every batch position (true) or only the last token (false).
    /// Targets that verify multiple positions need true so verify can
    /// read K+1 logit rows from one chunked-prefill call.
    public let prefillEmitAllLogits: Bool
    /// Whether the package includes an argmax-only prefill verifier table.
    public let prefillVerifyArgmax: Bool
    public let inferenceMaxTokens: Int
    public let eosTokens: [Int32]
    public let thinkToken: Int32?
    public let thinkEndToken: Int32?
    public let thinkSkipSuffix: Int32?
    public let tiedLMHead: Bool
    public let normMode: String
    public let traceMode: String
    /// Per-tensor TurboQuant-H opt-in glob list. Must be part of
    /// the reuse fingerprint: changing the list changes which
    /// tensors get re-encoded, which shifts subsequent offsets,
    /// so omitting it would let the build silently reuse a stale
    /// weights.bin on rebuild.
    public let turboQuantHPatterns: [String]
    /// Per-tensor preserve-native opt-in glob list (matched matvec
    /// projections kept at native bf16 instead of fp16-downcast; bf16-source
    /// only today, fp32-source deferred). Like `turboQuantHPatterns` it must
    /// be part of the reuse fingerprint: changing the list shifts the
    /// weights.bin layout, so omitting it would let the build silently reuse
    /// stale bytes on rebuild.
    public let preserveNativePatterns: [String]
    public let compilationGeneratedKernels: String
    public let compilationWeightLayout: String

    public init(
        layerPatternUnit: [String],
        layerPatternRepeats: Int,
        quantizationStrategy: String,
        groupSize: Int,
        excludePatterns: [String],
        quantizeEmbedding: Bool,
        loadingStrategy: String,
        packing: String,
        checkpointMap: String?,
        prefillEngine: String?,
        maxPrefillBatch: Int?,
        prefillHandoffFamilies: [String],
        prefillEmitAllLogits: Bool = false,
        prefillVerifyArgmax: Bool = false,
        inferenceMaxTokens: Int,
        eosTokens: [Int32],
        thinkToken: Int32?,
        thinkEndToken: Int32?,
        thinkSkipSuffix: Int32?,
        tiedLMHead: Bool,
        normMode: String = "weight",
        traceMode: String,
        turboQuantHPatterns: [String] = [],
        preserveNativePatterns: [String] = [],
        compilationGeneratedKernels: String = "auto",
        compilationWeightLayout: String = "memory_neutral"
    ) {
        self.layerPatternUnit = layerPatternUnit
        self.layerPatternRepeats = layerPatternRepeats
        self.quantizationStrategy = quantizationStrategy
        self.groupSize = groupSize
        self.excludePatterns = excludePatterns
        self.quantizeEmbedding = quantizeEmbedding
        self.loadingStrategy = loadingStrategy
        self.packing = packing
        self.checkpointMap = checkpointMap
        self.prefillEngine = prefillEngine
        self.maxPrefillBatch = maxPrefillBatch
        self.prefillHandoffFamilies = prefillHandoffFamilies
        self.prefillEmitAllLogits = prefillEmitAllLogits
        self.prefillVerifyArgmax = prefillVerifyArgmax
        self.inferenceMaxTokens = inferenceMaxTokens
        self.eosTokens = eosTokens
        self.thinkToken = thinkToken
        self.thinkEndToken = thinkEndToken
        self.thinkSkipSuffix = thinkSkipSuffix
        self.tiedLMHead = tiedLMHead
        self.normMode = normMode
        self.traceMode = traceMode
        self.turboQuantHPatterns = turboQuantHPatterns
        self.preserveNativePatterns = preserveNativePatterns
        self.compilationGeneratedKernels = compilationGeneratedKernels
        self.compilationWeightLayout = compilationWeightLayout
    }

    public init(
        layerPatternUnit: [String],
        layerPatternRepeats: Int,
        quantizationStrategy: String,
        groupSize: Int,
        excludePatterns: [String],
        quantizeEmbedding: Bool,
        loadingStrategy: String,
        packing: String,
        prefillEngine: String?,
        maxPrefillBatch: Int?,
        prefillHandoffFamilies: [String],
        prefillEmitAllLogits: Bool = false,
        prefillVerifyArgmax: Bool = false,
        inferenceMaxTokens: Int,
        eosTokens: [Int32],
        thinkToken: Int32?,
        thinkEndToken: Int32?,
        thinkSkipSuffix: Int32?,
        tiedLMHead: Bool,
        normMode: String = "weight",
        traceMode: String,
        turboQuantHPatterns: [String] = [],
        preserveNativePatterns: [String] = [],
        compilationGeneratedKernels: String = "auto",
        compilationWeightLayout: String = "memory_neutral"
    ) {
        self.init(
            layerPatternUnit: layerPatternUnit,
            layerPatternRepeats: layerPatternRepeats,
            quantizationStrategy: quantizationStrategy,
            groupSize: groupSize,
            excludePatterns: excludePatterns,
            quantizeEmbedding: quantizeEmbedding,
            loadingStrategy: loadingStrategy,
            packing: packing,
            checkpointMap: nil,
            prefillEngine: prefillEngine,
            maxPrefillBatch: maxPrefillBatch,
            prefillHandoffFamilies: prefillHandoffFamilies,
            prefillEmitAllLogits: prefillEmitAllLogits,
            prefillVerifyArgmax: prefillVerifyArgmax,
            inferenceMaxTokens: inferenceMaxTokens,
            eosTokens: eosTokens,
            thinkToken: thinkToken,
            thinkEndToken: thinkEndToken,
            thinkSkipSuffix: thinkSkipSuffix,
            tiedLMHead: tiedLMHead,
            normMode: normMode,
            traceMode: traceMode,
            turboQuantHPatterns: turboQuantHPatterns,
            preserveNativePatterns: preserveNativePatterns,
            compilationGeneratedKernels: compilationGeneratedKernels,
            compilationWeightLayout: compilationWeightLayout
        )
    }

    private enum CodingKeys: String, CodingKey {
        case layerPatternUnit
        case layerPatternRepeats
        case quantizationStrategy
        case groupSize
        case excludePatterns
        case quantizeEmbedding
        case loadingStrategy
        case packing
        case checkpointMap
        case prefillEngine
        case maxPrefillBatch
        case prefillHandoffFamilies
        case prefillEmitAllLogits
        case prefillVerifyArgmax
        case inferenceMaxTokens
        case eosTokens
        case thinkToken
        case thinkEndToken
        case thinkSkipSuffix
        case tiedLMHead
        case normMode
        case traceMode
        case turboQuantHPatterns
        case preserveNativePatterns
        case compilationGeneratedKernels
        case compilationWeightLayout
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        layerPatternUnit = try container.decode([String].self, forKey: .layerPatternUnit)
        layerPatternRepeats = try container.decode(Int.self, forKey: .layerPatternRepeats)
        quantizationStrategy = try container.decode(String.self, forKey: .quantizationStrategy)
        groupSize = try container.decode(Int.self, forKey: .groupSize)
        excludePatterns = try container.decode([String].self, forKey: .excludePatterns)
        quantizeEmbedding = try container.decode(Bool.self, forKey: .quantizeEmbedding)
        loadingStrategy = try container.decode(String.self, forKey: .loadingStrategy)
        packing = try container.decode(String.self, forKey: .packing)
        checkpointMap = try container.decodeIfPresent(String.self, forKey: .checkpointMap)
        prefillEngine = try container.decodeIfPresent(String.self, forKey: .prefillEngine)
        maxPrefillBatch = try container.decodeIfPresent(Int.self, forKey: .maxPrefillBatch)
        prefillHandoffFamilies = try container.decode([String].self, forKey: .prefillHandoffFamilies)
        prefillEmitAllLogits = try container.decodeIfPresent(Bool.self, forKey: .prefillEmitAllLogits) ?? false
        prefillVerifyArgmax = try container.decodeIfPresent(Bool.self, forKey: .prefillVerifyArgmax) ?? false
        inferenceMaxTokens = try container.decode(Int.self, forKey: .inferenceMaxTokens)
        eosTokens = try container.decode([Int32].self, forKey: .eosTokens)
        thinkToken = try container.decodeIfPresent(Int32.self, forKey: .thinkToken)
        thinkEndToken = try container.decodeIfPresent(Int32.self, forKey: .thinkEndToken)
        thinkSkipSuffix = try container.decodeIfPresent(Int32.self, forKey: .thinkSkipSuffix)
        tiedLMHead = try container.decode(Bool.self, forKey: .tiedLMHead)
        normMode = try container.decodeIfPresent(String.self, forKey: .normMode) ?? "weight"
        traceMode = try container.decodeIfPresent(String.self, forKey: .traceMode) ?? "full"
        turboQuantHPatterns = try container.decodeIfPresent(
            [String].self, forKey: .turboQuantHPatterns
        ) ?? []
        preserveNativePatterns = try container.decodeIfPresent(
            [String].self, forKey: .preserveNativePatterns
        ) ?? []
        compilationGeneratedKernels = try container.decodeIfPresent(
            String.self,
            forKey: .compilationGeneratedKernels
        ) ?? "auto"
        compilationWeightLayout = try container.decodeIfPresent(
            String.self,
            forKey: .compilationWeightLayout
        ) ?? "memory_neutral"
    }

    /// Materialised layer pattern: `unit` × `repeats`. The compiler stores
    /// the authoring representation (unit + repeat count); consumers that
    /// need per-layer family information walk the expanded list.
    public var expandedLayerPattern: [String] {
        Array(repeating: layerPatternUnit, count: layerPatternRepeats)
            .flatMap { $0 }
    }
}

// MARK: - Device requirements

/// Minimum device capabilities for this package.
public struct SmeltDeviceRequirements: Codable, Sendable {
    /// Minimum Metal GPU family.
    public let metalFamily: SmeltMetalFamily
    /// Minimum device memory in bytes for weight buffer + activations.
    public let minMemoryBytes: UInt64

    public init(metalFamily: SmeltMetalFamily, minMemoryBytes: UInt64) {
        self.metalFamily = metalFamily
        self.minMemoryBytes = minMemoryBytes
    }

    private enum CodingKeys: String, CodingKey {
        case metalFamily = "metal_family"
        case minMemoryBytes = "min_memory_bytes"
    }
}

/// Supported Metal GPU families.
public enum SmeltMetalFamily: String, Codable, Sendable {
    case apple7   // M1
    case apple8   // M2
    case apple9   // M3
    case apple10  // M4
}

// MARK: - Weight manifest

/// Describes the monolithic weights.bin layout.
public struct SmeltWeightManifest: Codable, Sendable {
    public let totalBytes: UInt64
    public let entries: [SmeltWeightEntry]

    public init(totalBytes: UInt64, entries: [SmeltWeightEntry]) {
        self.totalBytes = totalBytes
        self.entries = entries
    }

    private enum CodingKeys: String, CodingKey {
        case totalBytes = "total_bytes"
        case entries
    }
}

/// Element types for weight tensors.
public enum SmeltDType: String, Codable, Sendable, Equatable {
    case fp16
    case fp32
    /// BF16 weight storage: a raw two-byte copy of checkpoint BF16 bits,
    /// widened or consumed natively by the selected operation kernel.
    case bf16
    case int32
    case u4Lut = "u4_lut"
    case affineU4 = "affine_u4"
    /// Signed binary values {-scale,+scale}, one packed bit per weight.
    case binary1
    /// Signed ternary values {-scale,0,+scale}, stored as LSB-first semantic
    /// codes 0/1/2 with two packed bits per weight.
    case ternary2
    case turboQuantH = "turbo_quant_h"
    case raw

    /// Bytes per logical element for unpacked element types, or nil
    /// for packed/opaque formats (raw byte streams, sub-byte
    /// quantizations) where the byte count isn't a fixed
    /// per-element multiple. Centralising the lookup keeps every
    /// dtype-size site from having to enumerate the packed cases
    /// in lock-step with the enum.
    public var bytesPerElement: Int? {
        switch self {
        case .fp16: return MemoryLayout<Float16>.stride
        case .bf16: return 2
        case .fp32: return MemoryLayout<Float>.stride
        case .int32: return MemoryLayout<Int32>.stride
        case .raw, .u4Lut, .affineU4, .binary1, .ternary2, .turboQuantH:
            return nil
        }
    }
}

/// One weight tensor in weights.bin.
public struct SmeltWeightEntry: Codable, Sendable, Equatable {
    public let name: String
    public let offset: UInt64
    public let sizeBytes: UInt64
    public let shape: [Int]
    public let dtype: SmeltDType
    public let groupSize: Int?
    public let lutOffset: UInt64?
    public let lutSizeBytes: UInt64?
    public let packedRowStride: Int?
    public let paddedCols: Int?
    /// Byte offset of per-group FP16 scales (groupwise packed formats).
    public let scalesOffset: UInt64?
    /// Byte count of the scales region (groupwise packed formats).
    public let scalesSizeBytes: UInt64?
    /// Byte offset of per-group FP16 biases (affine_u4 only).
    public let biasesOffset: UInt64?
    /// Byte count of the biases region (affine_u4 only).
    public let biasesSizeBytes: UInt64?
    /// Byte offset of the per-group codebook (turbo_quant_h only).
    /// Codebook layout: [num_groups, 4] fp16.
    public let codebookOffset: UInt64?
    /// Byte count of the codebook region (turbo_quant_h only).
    public let codebookSizeBytes: UInt64?

    public init(
        name: String,
        offset: UInt64,
        sizeBytes: UInt64,
        shape: [Int],
        dtype: SmeltDType,
        groupSize: Int? = nil,
        lutOffset: UInt64? = nil,
        lutSizeBytes: UInt64? = nil,
        packedRowStride: Int? = nil,
        paddedCols: Int? = nil,
        scalesOffset: UInt64? = nil,
        scalesSizeBytes: UInt64? = nil,
        biasesOffset: UInt64? = nil,
        biasesSizeBytes: UInt64? = nil,
        codebookOffset: UInt64? = nil,
        codebookSizeBytes: UInt64? = nil
    ) {
        self.name = name
        self.offset = offset
        self.sizeBytes = sizeBytes
        self.shape = shape
        self.dtype = dtype
        self.groupSize = groupSize
        self.lutOffset = lutOffset
        self.lutSizeBytes = lutSizeBytes
        self.packedRowStride = packedRowStride
        self.paddedCols = paddedCols
        self.scalesOffset = scalesOffset
        self.scalesSizeBytes = scalesSizeBytes
        self.biasesOffset = biasesOffset
        self.biasesSizeBytes = biasesSizeBytes
        self.codebookOffset = codebookOffset
        self.codebookSizeBytes = codebookSizeBytes
    }

    private enum CodingKeys: String, CodingKey {
        case name, offset, shape, dtype
        case sizeBytes = "size_bytes"
        case groupSize = "group_size"
        case lutOffset = "lut_offset"
        case lutSizeBytes = "lut_size_bytes"
        case packedRowStride = "packed_row_stride"
        case paddedCols = "padded_cols"
        case scalesOffset = "scales_offset"
        case scalesSizeBytes = "scales_size_bytes"
        case biasesOffset = "biases_offset"
        case biasesSizeBytes = "biases_size_bytes"
        case codebookOffset = "codebook_offset"
        case codebookSizeBytes = "codebook_size_bytes"
    }
}

// MARK: - Buffer table

/// All pre-allocated buffers, in slot index order.
public struct SmeltBufferTable: Codable, Sendable {
    public let slots: [SmeltBufferSlot]

    public init(slots: [SmeltBufferSlot]) {
        self.slots = slots
    }
}

/// One buffer slot.
public struct SmeltBufferSlot: Codable, Sendable {
    public let index: Int
    public let name: String
    public let sizeBytes: Int
    public let dtype: SmeltDType
    public let shape: [Int]
    public let category: SmeltBufferCategory

    public init(
        index: Int,
        name: String,
        sizeBytes: Int,
        dtype: SmeltDType,
        shape: [Int] = [],
        category: SmeltBufferCategory
    ) {
        self.index = index
        self.name = name
        self.sizeBytes = sizeBytes
        self.dtype = dtype
        self.shape = shape
        self.category = category
    }

    private enum CodingKeys: String, CodingKey {
        case index, name, dtype, shape, category
        case sizeBytes = "size_bytes"
    }
}

/// Buffer categories.
public enum SmeltBufferCategory: String, Codable, Sendable, Equatable {
    case activation
    case state
    case weight
    case table
    case dynamic
}

// MARK: - JSON serialization

extension SmeltManifest {
    public func encodeJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(self)
    }

    public func encodePrettyJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(self)
    }

    public static func decode(from data: Data) throws -> SmeltManifest {
        let manifest = try JSONDecoder().decode(SmeltManifest.self, from: data)
        try manifest.validatePackageOwnedRuntimePolicy()
        return manifest
    }

    public func validatePackageOwnedRuntimePolicy() throws {
        if kind != nil {
            throw SmeltManifestValidationError.invalid(
                "text manifest must not declare root kind"
            )
        }
        if architecture != nil {
            throw SmeltManifestValidationError.invalid(
                "text manifest must not declare root architecture"
            )
        }
        if headlessTrunkABI == true {
            try validateHeadlessTrunkRuntimePolicy()
            return
        }
        try validateTextCAMGraphAndLoop()
        guard let inference else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation requires package-owned inference policy"
            )
        }
        guard !inference.eosTokens.isEmpty else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation requires inference.eos_tokens"
            )
        }
        guard let chatTemplate = inference.chatTemplate, !chatTemplate.isEmpty else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation requires inference.chat_template"
            )
        }
        guard SmeltPromptTemplateName.isKnownPromptTemplate(chatTemplate) else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation inference.chat_template '\(chatTemplate)' is not supported "
                    + "(available: \(SmeltPromptTemplateName.availablePromptTemplates))"
            )
        }
        guard inference.thinkingPolicy != nil else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation requires inference.thinking_policy"
            )
        }
        if let codec = inference.toolTranscriptCodec,
           !SmeltToolTranscriptCodecName.isKnown(codec) {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation inference.tool_transcript_codec '\(codec)' "
                    + "is not supported (available: "
                    + SmeltToolTranscriptCodecName.availableCodecs + ")"
            )
        }
        try validatePackageOwnedTextDecodePolicy(inference: inference)
        try validatePackageOwnedTextPerformanceProfile()
    }

    private func validateHeadlessTrunkRuntimePolicy() throws {
        guard blocks == nil, loop == nil, inference == nil, decode == nil, validation == nil else {
            throw SmeltManifestValidationError.invalid(
                "headless trunk manifest must not declare runnable text policy"
            )
        }
    }

    private func validatePackageOwnedTextDecodePolicy(
        inference: SmeltInferenceManifest
    ) throws {
        guard let decode else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation requires package-owned decode policy"
            )
        }
        guard decode.subSampler == nil else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation does not support decode.sub_sampler"
            )
        }
        guard let maxSteps = decode.maxSteps else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation requires decode.max_steps"
            )
        }
        guard maxSteps == inference.maxTokens else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation requires decode.max_steps to match inference.max_tokens"
            )
        }
        guard decode.durationSeconds == nil else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation does not support decode.duration_seconds"
            )
        }
        try validatePackageOwnedTextSampler(decode.sampler)
    }

    private func validatePackageOwnedTextSampler(
        _ sampler: SmeltPackageSpec.DecodePolicy.Sampler
    ) throws {
        if let temperature = sampler.temperature, temperature <= 0 {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation sampler temperature must be positive"
            )
        }
        if let topK = sampler.topK, topK <= 0 {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation sampler top_k must be positive"
            )
        }
        if let topP = sampler.topP, topP <= 0 || topP > 1 {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation sampler top_p must be in (0, 1]"
            )
        }
    }

    public func validateTextCAMGraphAndLoop() throws {
        guard let blocks, let loop else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation requires blocks and loop"
            )
        }
        do {
            try blocks.validate()
        } catch let err as SmeltBlockGraph.GraphError {
            throw SmeltManifestValidationError.invalid(err.description)
        }
        do {
            try loop.validate(against: blocks)
        } catch let err as SmeltLoopSchedule.ScheduleError {
            throw SmeltManifestValidationError.invalid(err.description)
        }
        try validateRunnableTextGraph(blocks)
        try validateRunnableTextLoop(loop, graph: blocks)
    }

    private func validateRunnableTextGraph(_ graph: SmeltBlockGraph) throws {
        guard let signature = graph.signature,
              signature.input == .text,
              signature.output == .text else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation requires text -> text block graph"
            )
        }
        guard let frontend = graph.blocks.first,
              frontend.role == .frontend,
              frontend.inputs == [.text],
              frontend.output == .tokens else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation requires text-to-token frontend"
            )
        }
        guard !textTokenFeedbackTrunks(in: graph).isEmpty else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation requires token-feedback logits trunk"
            )
        }
        guard !textOutputHeads(in: graph).isEmpty else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation requires logits-to-text head"
            )
        }
    }

    private func validateRunnableTextLoop(
        _ loop: SmeltLoopSchedule,
        graph: SmeltBlockGraph
    ) throws {
        guard case .perStep = loop.emission else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation requires per-step emission"
            )
        }
        guard loop.stop.contains(.eosToken), loop.stop.contains(.maxSteps),
              loop.stop.contains(.hostCancel) else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation requires eos-token, max-steps, and host-cancel stops"
            )
        }
        let trunkNames = textTokenFeedbackTrunks(in: graph).map(\.name)
        let headNames = textOutputHeads(in: graph).map(\.name)
        guard loop.perStep.contains(where: {
            phase($0, drivesAnyOf: trunkNames) && phase($0, drivesAnyOf: headNames)
        }) else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation requires per-step phase to drive token-feedback trunk and text head"
            )
        }
        if !loop.setup.isEmpty {
            guard loop.setup.contains(where: { phase($0, drivesAnyOf: trunkNames) }),
                  loop.setup.contains(where: { phase($0, drivesAnyOf: headNames) }) else {
                throw SmeltManifestValidationError.invalid(
                    "text CAM validation setup phases must drive token-feedback trunk and text head"
                )
            }
        }
    }

    private func textTokenFeedbackTrunks(
        in graph: SmeltBlockGraph
    ) -> [SmeltBlockGraph.Block] {
        graph.blocks.filter {
            $0.role == .trunk
                && $0.inputs.first == .tokens
                && $0.output == .logits
                && $0.feedback == .tokens
        }
    }

    private func textOutputHeads(in graph: SmeltBlockGraph) -> [SmeltBlockGraph.Block] {
        graph.blocks.filter {
            $0.role == .head
                && $0.inputs.first == .logits
                && $0.output == .text
        }
    }

    private func phase(
        _ phase: SmeltLoopSchedule.Phase,
        drivesAnyOf blockNames: [String]
    ) -> Bool {
        phase.blocks.contains { blockNames.contains($0) }
    }

    public func resolvedInferencePolicy() throws -> SmeltResolvedInferencePolicy {
        guard let inference else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation requires package-owned inference policy"
            )
        }
        return SmeltResolvedInferencePolicy(
            inference: inference,
            source: .package
        )
    }

    private func validatePackageOwnedTextPerformanceProfile() throws {
        let gate = SmeltPackagePerformanceGateID.textDecodePrefillStartup
        guard let validation else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation requires performance_profile"
            )
        }
        guard let profile = validation.performanceProfile else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation requires performance_profile"
            )
        }
        if let performanceGate = validation.performanceGate, performanceGate != profile.gate {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation performance_profile gate '\(profile.gate)' disagrees with "
                    + "performance_gate '\(performanceGate)'"
            )
        }
        guard profile.gate == gate else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation performance_profile gate must be '\(gate)', got "
                    + "'\(profile.gate)'"
            )
        }
        guard profile.command == .run else {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation performance_profile command must be 'run', got "
                    + "'\(profile.command.rawValue)'"
            )
        }
        for bound in profile.minBounds where !bound.min.isFinite || bound.min <= 0 {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation performance_profile min-bound '\(bound.metric)' "
                    + "must be positive and finite"
            )
        }
        for bound in profile.maxBounds where !bound.max.isFinite || bound.max <= 0 {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation performance_profile max-bound '\(bound.metric)' "
                    + "must be positive and finite"
            )
        }

        let canonical = SmeltPackagePerformanceProfiles.profile(
            for: gate,
            modelName: modelName
        )
        try requireTextValidationStrings(
            profile.requiredTraceLabels,
            include: canonical.requiredTraceLabels,
            field: "required trace label"
        )
        try requireTextValidationStrings(
            profile.requiredOutputMetrics,
            include: canonical.requiredOutputMetrics,
            field: "required output metric"
        )
        try requireTextValidationMinBounds(profile.minBounds, include: canonical.minBounds)
        try requireTextValidationMaxBounds(profile.maxBounds, include: canonical.maxBounds)
    }

    private func requireTextValidationStrings(
        _ actual: [String],
        include expected: [String],
        field: String
    ) throws {
        let actualSet = Set(actual)
        for value in expected where !actualSet.contains(value) {
            throw SmeltManifestValidationError.invalid(
                "text CAM validation performance_profile missing canonical "
                    + "\(field): \(value)"
            )
        }
    }

    private func requireTextValidationMinBounds(
        _ actual: [SmeltPackageSpec.Validation.PerformanceProfile.MinBound],
        include expected: [SmeltPackageSpec.Validation.PerformanceProfile.MinBound]
    ) throws {
        for bound in expected {
            // A package may declare a stricter (higher) floor than canonical, but
            // never a weaker one — mirror of the max-bound direction.
            guard actual.contains(where: {
                $0.metric == bound.metric && $0.unit == bound.unit && $0.min >= bound.min
            }) else {
                throw SmeltManifestValidationError.invalid(
                    "text CAM validation performance_profile missing canonical min-bound: "
                        + bound.metric
                )
            }
        }
    }

    private func requireTextValidationMaxBounds(
        _ actual: [SmeltPackageSpec.Validation.PerformanceProfile.Bound],
        include expected: [SmeltPackageSpec.Validation.PerformanceProfile.Bound]
    ) throws {
        for bound in expected {
            guard actual.contains(where: {
                $0.metric == bound.metric && $0.unit == bound.unit && $0.max <= bound.max
            }) else {
                throw SmeltManifestValidationError.invalid(
                    "text CAM validation performance_profile missing canonical max-bound: "
                        + bound.metric
                )
            }
        }
    }
}

// MARK: - Prefill manifest

/// Prefill configuration embedded in the package manifest.
/// Optional — nil if the model doesn't use batch prefill.
public struct SmeltPrefillManifest: Codable, Sendable {
    /// Prefill engine type (e.g. "coreml").
    public let engine: String
    /// Relative path to the compiled prefill model within .smeltpkg.
    public let modelPath: String
    /// Maximum token chunk size supported by this package's prefill path.
    public let maxBatchSize: Int
    /// Resolved state→slot handoff mappings.
    public let handoff: SmeltHandoffTable
    /// CoreML input/output tensor name contract.
    public let inputContract: SmeltPrefillInputContract

    public init(
        engine: String,
        modelPath: String,
        maxBatchSize: Int,
        handoff: SmeltHandoffTable,
        inputContract: SmeltPrefillInputContract
    ) {
        self.engine = engine
        self.modelPath = modelPath
        self.maxBatchSize = maxBatchSize
        self.handoff = handoff
        self.inputContract = inputContract
    }

    private enum CodingKeys: String, CodingKey {
        case engine
        case modelPath = "model_path"
        case maxBatchSize = "max_prefill_batch"
        case handoff
        case inputContract = "input_contract"
    }
}

/// Describes the CoreML prefill model's input/output tensor names.
/// Keeps exporter conventions out of runtime code.
public struct SmeltPrefillInputContract: Codable, Sendable {
    public let tokenInputName: String
    public let seqLenInputName: String
    public let startPosInputName: String
    public let ropeCosInputName: String?
    public let ropeSinInputName: String?
    public let maskInputName: String?
    public let logitsOutputName: String
    /// Pattern for state inputs: "{family}_{index}" e.g. "conv_state_0"
    public let stateInputPattern: String
    /// Pattern for state outputs: "{family}_{index}_out" e.g. "conv_state_0_out"
    public let stateOutputPattern: String

    public init(
        tokenInputName: String = "input_ids",
        seqLenInputName: String = "seq_len",
        startPosInputName: String = "start_pos",
        ropeCosInputName: String? = "rope_cos",
        ropeSinInputName: String? = "rope_sin",
        maskInputName: String? = "attn_mask",
        logitsOutputName: String = "logits",
        stateInputPattern: String = "{family}_{index}",
        stateOutputPattern: String = "{family}_{index}_out"
    ) {
        self.tokenInputName = tokenInputName
        self.seqLenInputName = seqLenInputName
        self.startPosInputName = startPosInputName
        self.ropeCosInputName = ropeCosInputName
        self.ropeSinInputName = ropeSinInputName
        self.maskInputName = maskInputName
        self.logitsOutputName = logitsOutputName
        self.stateInputPattern = stateInputPattern
        self.stateOutputPattern = stateOutputPattern
    }

    private enum CodingKeys: String, CodingKey {
        case tokenInputName = "token_input"
        case seqLenInputName = "seq_len_input"
        case startPosInputName = "start_pos_input"
        case ropeCosInputName = "rope_cos_input"
        case ropeSinInputName = "rope_sin_input"
        case maskInputName = "mask_input"
        case logitsOutputName = "logits_output"
        case stateInputPattern = "state_input_pattern"
        case stateOutputPattern = "state_output_pattern"
    }
}

// MARK: - Inference manifest

/// Whether a chat template that supports a thinking channel (e.g. Qwen's
/// `<think>…</think>`) opens that channel for generation. `disabled` pre-closes
/// it in the prompt so generation starts after `</think>` — the robust default
/// for quantized models, whose degraded distribution can't reliably navigate a
/// long thinking chain. `enabled` leaves it open. Templates without a think
/// channel ignore this. Under `enabled`, run and serve gate the runtime
/// think-skip on the policy so the trace flows at decode; it appears INLINE in
/// the output because the special `<think>`/`</think>` delimiters are stripped
/// by decode (reasoning-content separation is a follow-up). Downstream agent
/// consumers may force non-thinking regardless of policy when tool-call
/// constrained decode needs the call immediately. No shipped package
/// uses `enabled`; `disabled` is the verified default.
public enum SmeltThinkingPolicy: String, Codable, Sendable {
    case disabled
    case enabled
}

public struct SmeltResolvedInferencePolicy: Sendable {
    public enum Source: Equatable, Sendable {
        case package
    }

    public let inference: SmeltInferenceManifest
    public let source: Source

    public init(inference: SmeltInferenceManifest, source: Source) {
        self.inference = inference
        self.source = source
    }
}

public enum SmeltPromptTemplateName {
    public static let raw = ""
    public static let chatML = "chatml"
    /// Legacy combined prompt/tool capability accepted when opening packages
    /// authored before tool transcript codecs became an independent contract.
    /// New modules declare `chatml` plus `xml-function-parameters` separately.
    public static let chatMLXMLTools = "chatml-xml-tools"
    public static let headerTurns = "header-turns"
    public static let channelTurns = "channel-turns"

    public static let knownPromptTemplates = [
        chatML,
        chatMLXMLTools,
        headerTurns,
        channelTurns,
    ]

    public static let availablePromptTemplates =
        knownPromptTemplates.joined(separator: ", ")

    public static func isKnownPromptTemplate(_ name: String) -> Bool {
        knownPromptTemplates.contains(name)
    }

    public static func canonicalRoleTemplate(for name: String) -> String {
        name == chatMLXMLTools ? chatML : name
    }
}

/// Package-owned native tool transcript codecs. Prompt templates describe
/// role/turn framing; this independent capability describes how tool schemas,
/// assistant calls, and tool results are represented inside those turns.
public enum SmeltToolTranscriptCodecName {
    public static let xmlFunctionParameters = "xml-function-parameters"
    public static let channelCalls = "channel-calls"
    public static let inkling = "inkling"

    public static let knownCodecs = [
        xmlFunctionParameters,
        channelCalls,
        inkling,
    ]

    public static let availableCodecs = knownCodecs.joined(separator: ", ")

    public static func isKnown(_ name: String) -> Bool {
        knownCodecs.contains(name)
    }

    public static func inferredFromLegacyPromptTemplate(
        _ promptTemplate: String
    ) -> String? {
        promptTemplate == SmeltPromptTemplateName.chatMLXMLTools
            ? xmlFunctionParameters : nil
    }
}

/// Whether a captured prompt state can be restored at a shorter token prefix.
/// KV-only state is position-indexed; recurrent/convolutional state is an
/// opaque fold over the complete history and is valid only at its exact
/// captured position.
public enum SmeltPromptStateRestoreMode: String, Codable, Sendable, Equatable {
    case positionIndexed = "position-indexed"
    case exactPosition = "exact-position"

    public static func derive<S: Sequence>(
        fromPersistentStateNames names: S
    ) -> SmeltPromptStateRestoreMode where S.Element == String {
        Set(names) == ["kv-cache"] ? .positionIndexed : .exactPosition
    }
}

/// Runtime inference parameters serialized into the package manifest.
public struct SmeltInferenceManifest: Codable, Sendable {
    public let maxTokens: Int
    public let eosTokens: [Int32]
    public let thinkToken: Int32?
    public let thinkEndToken: Int32?
    /// Token to inject after </think> during think-skip (typically newline).
    public let thinkSkipSuffix: Int32?
    /// Prompt renderer packaged with the model (e.g. "chatml", "channel-turns").
    /// Both `smelt run` and `smelt serve` read this so prompt rendering is a
    /// package property, not a per-entry-point model-name guess. nil means raw
    /// prompt concatenation unless an explicit CLI template is supplied.
    public let chatTemplate: String?
    /// Thinking-channel policy for the chat template (see SmeltThinkingPolicy).
    /// nil → `.disabled`.
    public let thinkingPolicy: SmeltThinkingPolicy?
    /// Package-native tool transcript codec. nil means the package has no
    /// proven native tool transcript and API adapters must use their generic
    /// constrained-output fallback.
    public let toolTranscriptCodec: String?
    /// Restore semantics derived from the CAM trunk's declared persistent
    /// state families. New packages always carry this value; nil is accepted
    /// for older package manifests and resolved from their state buffer table.
    public let promptStateRestoreMode: SmeltPromptStateRestoreMode?

    public init(
        maxTokens: Int = 512,
        eosTokens: [Int32] = [],
        thinkToken: Int32? = nil,
        thinkEndToken: Int32? = nil,
        thinkSkipSuffix: Int32? = nil,
        chatTemplate: String? = nil,
        thinkingPolicy: SmeltThinkingPolicy? = nil,
        toolTranscriptCodec: String? = nil,
        promptStateRestoreMode: SmeltPromptStateRestoreMode? = nil
    ) {
        self.maxTokens = maxTokens
        self.eosTokens = eosTokens
        self.thinkToken = thinkToken
        self.thinkEndToken = thinkEndToken
        self.thinkSkipSuffix = thinkSkipSuffix
        self.chatTemplate = chatTemplate
        self.thinkingPolicy = thinkingPolicy
        self.toolTranscriptCodec = toolTranscriptCodec
        self.promptStateRestoreMode = promptStateRestoreMode
    }

    private enum CodingKeys: String, CodingKey {
        case maxTokens = "max_tokens"
        case eosTokens = "eos_tokens"
        case thinkToken = "think_token"
        case thinkEndToken = "think_end_token"
        case thinkSkipSuffix = "think_skip_suffix"
        case chatTemplate = "chat_template"
        case thinkingPolicy = "thinking_policy"
        case toolTranscriptCodec = "tool_transcript_codec"
        case promptStateRestoreMode = "prompt_state_restore_mode"
    }
}
