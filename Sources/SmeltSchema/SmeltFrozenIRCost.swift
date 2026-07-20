import Foundation

/// Stable, model-agnostic resource accounting for an already-frozen dispatch
/// table. These records describe what the package will execute; they do not
/// authorize compiler rewrites or stand in for measured performance evidence.
public enum SmeltFrozenIRStorageClass: String, Codable, CaseIterable, Sendable {
    case streamingWeight = "streaming_weight"
    case hotActivation = "hot_activation"
    case persistentState = "persistent_state"
    case lookupTable = "lookup_table"
    case runtimeScalar = "runtime_scalar"
    case unknown
}

public enum SmeltFrozenIRAccessKind: String, Codable, Sendable {
    case read
    case write
    case readWrite = "read_write"
    case unknown
}

public enum SmeltFrozenIROperationClass: String, Codable, CaseIterable, Sendable {
    case fp16Arithmetic = "fp16_arithmetic"
    case fp32Arithmetic = "fp32_arithmetic"
    case integerBitwise = "integer_bitwise"
    case transcendental
    case reduction
    case unknown
}

public enum SmeltFrozenIRSynchronizationClass: String, Codable, Sendable {
    case none
    case simdgroup
    case threadgroup
    case device
    case atomic
    case unknown
}

public struct SmeltFrozenIRGeometry: Codable, Equatable, Sendable {
    public let style: SmeltCostModelDispatchStyle
    public let grid: [Int]
    public let threadgroup: [Int]
    public let threadgroupCount: Int

    public init(
        style: SmeltCostModelDispatchStyle,
        grid: [Int],
        threadgroup: [Int],
        threadgroupCount: Int
    ) {
        self.style = style
        self.grid = grid
        self.threadgroup = threadgroup
        self.threadgroupCount = threadgroupCount
    }
}

public struct SmeltFrozenIRResourceAccess: Codable, Equatable, Sendable {
    public let bindingIndex: Int
    public let slotIndex: Int?
    public let resourceName: String
    public let storageClass: SmeltFrozenIRStorageClass
    public let access: SmeltFrozenIRAccessKind
    public let byteOffset: UInt64
    /// Logical bytes touched once by the kernel contract. This is deliberately
    /// distinct from cache-line traffic, replay, and physical DRAM bytes.
    public let logicalBytes: UInt64?
    public let unknownReason: String?

    public init(
        bindingIndex: Int,
        slotIndex: Int?,
        resourceName: String,
        storageClass: SmeltFrozenIRStorageClass,
        access: SmeltFrozenIRAccessKind,
        byteOffset: UInt64,
        logicalBytes: UInt64?,
        unknownReason: String? = nil
    ) {
        self.bindingIndex = bindingIndex
        self.slotIndex = slotIndex
        self.resourceName = resourceName
        self.storageClass = storageClass
        self.access = access
        self.byteOffset = byteOffset
        self.logicalBytes = logicalBytes
        self.unknownReason = unknownReason
    }
}

public struct SmeltFrozenIROperationBill: Codable, Equatable, Sendable {
    public let operationClass: SmeltFrozenIROperationClass
    /// Logical operations implied by the kernel contract, not an instruction
    /// count after Metal compilation.
    public let count: UInt64?
    public let unknownReason: String?

    public init(
        operationClass: SmeltFrozenIROperationClass,
        count: UInt64?,
        unknownReason: String? = nil
    ) {
        self.operationClass = operationClass
        self.count = count
        self.unknownReason = unknownReason
    }
}

public struct SmeltFrozenIRDispatchBill: Codable, Equatable, Sendable {
    public let recordIndex: Int
    public let dispatchOrdinal: Int?
    /// `swap` for a runtime record; otherwise the frozen Metal function name.
    public let pipeline: String
    public let executesGPU: Bool
    public let operationGroup: String?
    public let geometry: SmeltFrozenIRGeometry?
    public let resources: [SmeltFrozenIRResourceAccess]
    public let operations: [SmeltFrozenIROperationBill]
    public let synchronization: SmeltFrozenIRSynchronizationClass
    /// Device-visible temporary vectors avoided or introduced by this record.
    /// Register and threadgroup-local storage is intentionally excluded.
    public let intermediateMaterializationBytes: UInt64?
    public let hostRecordCount: Int
    public let calibratedMedianGPUUs: Double?
    public let calibratedP95GPUUs: Double?
    public let unknowns: [String]

    public init(
        recordIndex: Int,
        dispatchOrdinal: Int?,
        pipeline: String,
        executesGPU: Bool = true,
        operationGroup: String?,
        geometry: SmeltFrozenIRGeometry?,
        resources: [SmeltFrozenIRResourceAccess] = [],
        operations: [SmeltFrozenIROperationBill] = [],
        synchronization: SmeltFrozenIRSynchronizationClass = .none,
        intermediateMaterializationBytes: UInt64? = 0,
        hostRecordCount: Int = 1,
        calibratedMedianGPUUs: Double? = nil,
        calibratedP95GPUUs: Double? = nil,
        unknowns: [String] = []
    ) {
        self.recordIndex = recordIndex
        self.dispatchOrdinal = dispatchOrdinal
        self.pipeline = pipeline
        self.executesGPU = executesGPU
        self.operationGroup = operationGroup
        self.geometry = geometry
        self.resources = resources
        self.operations = operations
        self.synchronization = synchronization
        self.intermediateMaterializationBytes = intermediateMaterializationBytes
        self.hostRecordCount = hostRecordCount
        self.calibratedMedianGPUUs = calibratedMedianGPUUs
        self.calibratedP95GPUUs = calibratedP95GPUUs
        self.unknowns = unknowns
    }

    public var isDescribed: Bool {
        operationGroup != nil
            && geometry != nil
            && unknowns.isEmpty
            && resources.allSatisfy {
                $0.logicalBytes != nil
                    && $0.unknownReason == nil
                    && $0.access != .unknown
                    && $0.storageClass != .unknown
            }
            && operations.allSatisfy {
                $0.count != nil
                    && $0.unknownReason == nil
                    && $0.operationClass != .unknown
            }
            && synchronization != .unknown
            && intermediateMaterializationBytes != nil
    }
}

/// Ordered, already-resolved execution emitted by any frozen component. A
/// package dispatch table and a handwritten/MPS component both lower to this
/// same boundary before costing; the cost model never needs a model identity
/// switch.
public struct SmeltFrozenIRPlan: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let planID: String
    public let provenanceKey: String
    public let context: SmeltCostModelContext
    public let records: [SmeltFrozenIRDispatchBill]

    public init(
        schemaVersion: Int = 1,
        planID: String,
        provenanceKey: String,
        context: SmeltCostModelContext,
        records: [SmeltFrozenIRDispatchBill]
    ) {
        self.schemaVersion = schemaVersion
        self.planID = planID
        self.provenanceKey = provenanceKey
        self.context = context
        self.records = records
    }

    public func encodeJSON(prettyPrinted: Bool = true) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = prettyPrinted
            ? [.prettyPrinted, .sortedKeys]
            : [.sortedKeys]
        return try encoder.encode(self)
    }
}

/// One measured execution of a frozen component plan. The positional record
/// samples deliberately carry no model or operation names: those come from
/// the exact-provenance plan, and the generic cost layer rejects mismatches.
public struct SmeltFrozenIRExecutionSpan: Codable, Equatable, Sendable {
    public let recordIndices: [Int]
    public let gpuUs: Double

    public init(recordIndices: [Int], gpuUs: Double) {
        self.recordIndices = recordIndices
        self.gpuUs = max(gpuUs, 0)
    }
}

public struct SmeltFrozenIRExecutionProfile: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let provenanceKey: String
    public let context: SmeltCostModelContext
    public let deviceName: String?
    public let measurementMethod: String
    public let wholePlanGPUUs: Double
    /// Non-overlapping timing spans over one or more consecutive frozen
    /// records. Opaque external APIs can share a span when the device executes
    /// them without an observable stage boundary.
    public let spans: [SmeltFrozenIRExecutionSpan]

    public init(
        schemaVersion: Int = 1,
        provenanceKey: String,
        context: SmeltCostModelContext,
        deviceName: String? = nil,
        measurementMethod: String,
        wholePlanGPUUs: Double,
        spans: [SmeltFrozenIRExecutionSpan]
    ) {
        self.schemaVersion = schemaVersion
        self.provenanceKey = provenanceKey
        self.context = context
        self.deviceName = deviceName
        self.measurementMethod = measurementMethod
        self.wholePlanGPUUs = max(wholePlanGPUUs, 0)
        self.spans = spans
    }
}

public struct SmeltFrozenIRStorageTotal: Codable, Equatable, Sendable {
    public let storageClass: SmeltFrozenIRStorageClass
    public let readBytes: UInt64
    public let writeBytes: UInt64

    public init(
        storageClass: SmeltFrozenIRStorageClass,
        readBytes: UInt64,
        writeBytes: UInt64
    ) {
        self.storageClass = storageClass
        self.readBytes = readBytes
        self.writeBytes = writeBytes
    }
}

public struct SmeltFrozenIROperationTotal: Codable, Equatable, Sendable {
    public let operationClass: SmeltFrozenIROperationClass
    public let count: UInt64

    public init(operationClass: SmeltFrozenIROperationClass, count: UInt64) {
        self.operationClass = operationClass
        self.count = count
    }
}

public enum SmeltFrozenIRCalibrationStatus: String, Codable, Equatable, Sendable {
    case exactArtifactMatch = "exact_artifact_match"
    case absent
    case artifactMismatch = "artifact_mismatch"
    case contextMismatch = "context_mismatch"
    case legacyUnstructured = "legacy_unstructured"
}

public struct SmeltFrozenIRCostSummary: Codable, Equatable, Sendable {
    public let recordCount: Int
    public let dispatchCount: Int
    public let skippedDispatchCount: Int
    public let swapCount: Int
    public let describedDispatchCount: Int
    public let descriptorCoverageFraction: Double
    public let calibratedDispatchCount: Int
    public let calibratedMedianGPUUs: Double?
    public let describedCalibratedMedianGPUUs: Double?
    public let measuredGPUCoverageFraction: Double?
    public let measuredWholePlanMedianGPUUs: Double?
    public let additiveCalibrationErrorFraction: Double?
    public let instrumentedWholePlanMedianGPUUs: Double?
    public let instrumentedSpanMedianGPUUs: Double?
    public let calibrationReconciliationScale: Double?
    public let predictedHostRecordUs: Double?
    public let calibrationStatus: SmeltFrozenIRCalibrationStatus?
    public let storageTotals: [SmeltFrozenIRStorageTotal]
    public let operationTotals: [SmeltFrozenIROperationTotal]
    public let intermediateMaterializationBytes: UInt64?
    public let synchronizationCounts: [SmeltFrozenIRStringCount]
    public let unknownDispatches: [String]

    public init(
        recordCount: Int,
        dispatchCount: Int,
        skippedDispatchCount: Int,
        swapCount: Int,
        describedDispatchCount: Int,
        descriptorCoverageFraction: Double,
        calibratedDispatchCount: Int,
        calibratedMedianGPUUs: Double?,
        describedCalibratedMedianGPUUs: Double?,
        measuredGPUCoverageFraction: Double?,
        measuredWholePlanMedianGPUUs: Double?,
        additiveCalibrationErrorFraction: Double?,
        instrumentedWholePlanMedianGPUUs: Double? = nil,
        instrumentedSpanMedianGPUUs: Double? = nil,
        calibrationReconciliationScale: Double? = nil,
        predictedHostRecordUs: Double?,
        calibrationStatus: SmeltFrozenIRCalibrationStatus? = nil,
        storageTotals: [SmeltFrozenIRStorageTotal],
        operationTotals: [SmeltFrozenIROperationTotal],
        intermediateMaterializationBytes: UInt64?,
        synchronizationCounts: [SmeltFrozenIRStringCount],
        unknownDispatches: [String]
    ) {
        self.recordCount = recordCount
        self.dispatchCount = dispatchCount
        self.skippedDispatchCount = skippedDispatchCount
        self.swapCount = swapCount
        self.describedDispatchCount = describedDispatchCount
        self.descriptorCoverageFraction = descriptorCoverageFraction
        self.calibratedDispatchCount = calibratedDispatchCount
        self.calibratedMedianGPUUs = calibratedMedianGPUUs
        self.describedCalibratedMedianGPUUs = describedCalibratedMedianGPUUs
        self.measuredGPUCoverageFraction = measuredGPUCoverageFraction
        self.measuredWholePlanMedianGPUUs = measuredWholePlanMedianGPUUs
        self.additiveCalibrationErrorFraction = additiveCalibrationErrorFraction
        self.instrumentedWholePlanMedianGPUUs = instrumentedWholePlanMedianGPUUs
        self.instrumentedSpanMedianGPUUs = instrumentedSpanMedianGPUUs
        self.calibrationReconciliationScale = calibrationReconciliationScale
        self.predictedHostRecordUs = predictedHostRecordUs
        self.calibrationStatus = calibrationStatus
        self.storageTotals = storageTotals
        self.operationTotals = operationTotals
        self.intermediateMaterializationBytes = intermediateMaterializationBytes
        self.synchronizationCounts = synchronizationCounts
        self.unknownDispatches = unknownDispatches
    }
}

/// Dictionary-shaped values are represented as sorted arrays in the external
/// schema so byte-for-byte JSON output does not depend on hash iteration.
public struct SmeltFrozenIRStringCount: Codable, Equatable, Sendable {
    public let name: String
    public let count: Int

    public init(name: String, count: Int) {
        self.name = name
        self.count = count
    }
}

public struct SmeltFrozenIRCostReport: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let planID: String
    public let provenanceKey: String
    public let context: SmeltCostModelContext
    public let summary: SmeltFrozenIRCostSummary
    public let records: [SmeltFrozenIRDispatchBill]

    public init(
        schemaVersion: Int = 1,
        planID: String,
        provenanceKey: String,
        context: SmeltCostModelContext,
        summary: SmeltFrozenIRCostSummary,
        records: [SmeltFrozenIRDispatchBill]
    ) {
        self.schemaVersion = schemaVersion
        self.planID = planID
        self.provenanceKey = provenanceKey
        self.context = context
        self.summary = summary
        self.records = records
    }

    public func encodeJSON(prettyPrinted: Bool = true) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = prettyPrinted
            ? [.prettyPrinted, .sortedKeys]
            : [.sortedKeys]
        return try encoder.encode(self)
    }
}

public struct SmeltFrozenIRCostDelta: Codable, Equatable, Sendable {
    public let baselinePlanID: String
    public let candidatePlanID: String
    public let dispatchCountDelta: Int
    public let hostRecordCountDelta: Int
    public let logicalReadBytesDelta: Int64?
    public let logicalWriteBytesDelta: Int64?
    public let intermediateMaterializationBytesDelta: Int64?
    public let calibratedMedianGPUUsDelta: Double?
    public let unknowns: [String]

    public init(
        baselinePlanID: String,
        candidatePlanID: String,
        dispatchCountDelta: Int,
        hostRecordCountDelta: Int,
        logicalReadBytesDelta: Int64?,
        logicalWriteBytesDelta: Int64?,
        intermediateMaterializationBytesDelta: Int64?,
        calibratedMedianGPUUsDelta: Double?,
        unknowns: [String] = []
    ) {
        self.baselinePlanID = baselinePlanID
        self.candidatePlanID = candidatePlanID
        self.dispatchCountDelta = dispatchCountDelta
        self.hostRecordCountDelta = hostRecordCountDelta
        self.logicalReadBytesDelta = logicalReadBytesDelta
        self.logicalWriteBytesDelta = logicalWriteBytesDelta
        self.intermediateMaterializationBytesDelta = intermediateMaterializationBytesDelta
        self.calibratedMedianGPUUsDelta = calibratedMedianGPUUsDelta
        self.unknowns = unknowns
    }

    public func encodeJSON(prettyPrinted: Bool = true) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = prettyPrinted
            ? [.prettyPrinted, .sortedKeys]
            : [.sortedKeys]
        return try encoder.encode(self)
    }
}
