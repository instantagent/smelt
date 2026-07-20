import Foundation
import SmeltSchema

/// Small lowering surface for component-owned execution bricks. Components
/// describe their already-selected operations here; aggregation, calibration,
/// comparison, and admission remain generic.
public struct SmeltFrozenComponentPlanBuilder {
    public let planID: String
    public let provenanceKey: String
    public let context: SmeltCostModelContext
    public private(set) var records: [SmeltFrozenIRDispatchBill] = []

    public init(
        planID: String,
        provenanceKey: String,
        context: SmeltCostModelContext
    ) {
        self.planID = planID
        self.provenanceKey = provenanceKey
        self.context = context
    }

    public mutating func append(
        pipeline: String,
        operationGroup: String,
        logicalShape: [Int],
        resources: [SmeltFrozenIRResourceAccess],
        operations: [SmeltFrozenIROperationBill],
        synchronization: SmeltFrozenIRSynchronizationClass = .none,
        intermediateMaterializationBytes: UInt64 = 0,
        calibratedMedianGPUUs: Double? = nil,
        calibratedP95GPUUs: Double? = nil
    ) {
        let ordinal = records.count + 1
        records.append(SmeltFrozenIRDispatchBill(
            recordIndex: records.count,
            dispatchOrdinal: ordinal,
            pipeline: pipeline,
            operationGroup: operationGroup,
            // External execution APIs do not expose their internal Metal
            // launch geometry. Preserve the exact logical shape in `grid` and
            // make that opacity explicit with a zero internal group count.
            geometry: SmeltFrozenIRGeometry(
                style: .threadgroups,
                grid: logicalShape,
                threadgroup: [0, 0, 0],
                threadgroupCount: 0
            ),
            resources: resources,
            operations: operations,
            synchronization: synchronization,
            intermediateMaterializationBytes: intermediateMaterializationBytes,
            calibratedMedianGPUUs: calibratedMedianGPUUs,
            calibratedP95GPUUs: calibratedP95GPUUs
        ))
    }

    public func build() -> SmeltFrozenIRPlan {
        SmeltFrozenIRPlan(
            planID: planID,
            provenanceKey: provenanceKey,
            context: context,
            records: records
        )
    }

    public static func resource(
        bindingIndex: Int,
        name: String,
        storageClass: SmeltFrozenIRStorageClass,
        access: SmeltFrozenIRAccessKind,
        bytes: UInt64
    ) -> SmeltFrozenIRResourceAccess {
        SmeltFrozenIRResourceAccess(
            bindingIndex: bindingIndex,
            slotIndex: nil,
            resourceName: name,
            storageClass: storageClass,
            access: access,
            byteOffset: 0,
            logicalBytes: bytes
        )
    }

    public static func operation(
        _ operationClass: SmeltFrozenIROperationClass,
        count: UInt64
    ) -> SmeltFrozenIROperationBill {
        SmeltFrozenIROperationBill(operationClass: operationClass, count: count)
    }

    public static func product(_ values: Int...) -> UInt64 {
        values.reduce(UInt64(1)) { partial, value in
            let next = UInt64(max(value, 0))
            let (result, overflow) = partial.multipliedReportingOverflow(by: next)
            return overflow ? UInt64.max : result
        }
    }

    public static func bytes(elements: UInt64, stride: Int = 4) -> UInt64 {
        let (result, overflow) = elements.multipliedReportingOverflow(
            by: UInt64(max(stride, 0))
        )
        return overflow ? UInt64.max : result
    }
}
