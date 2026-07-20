import Foundation
import SmeltRuntime
import SmeltSchema

public enum SmeltFrozenIRCostModelError: Error, CustomStringConvertible {
    case invalidDispatchTable(String)
    case invalidPipelineIndex(record: Int, pipeline: Int)
    case unsupportedCalibrationTable(String)
    case emptyComponentProfile
    case componentProfileMismatch(String)

    public var description: String {
        switch self {
        case .invalidDispatchTable(let detail):
            return "frozen IR cost model: \(detail)"
        case .invalidPipelineIndex(let record, let pipeline):
            return "frozen IR cost model: record \(record) references missing pipeline \(pipeline)"
        case .unsupportedCalibrationTable(let table):
            return "frozen IR cost model: runtime calibration is unsupported for \(table)"
        case .emptyComponentProfile:
            return "frozen IR cost model: component calibration requires at least one profile"
        case .componentProfileMismatch(let detail):
            return "frozen IR cost model: component profile mismatch: \(detail)"
        }
    }
}

public enum SmeltFrozenIRDispatchTable: String, CaseIterable, Sendable {
    case decode = "dispatches.bin"
    case prefill = "prefill_dispatches.bin"
    case prefillVerifyArgmax = "prefill_verify_argmax_dispatches.bin"

    public var mode: SmeltCostModelMode {
        switch self {
        case .decode:
            return .decode
        case .prefill, .prefillVerifyArgmax:
            return .prefill
        }
    }

    public var planID: String {
        switch self {
        case .decode:
            return "package-decode"
        case .prefill:
            return "package-prefill"
        case .prefillVerifyArgmax:
            return "package-prefill-verify-argmax"
        }
    }

    public var cliName: String {
        switch self {
        case .decode: return "decode"
        case .prefill: return "prefill"
        case .prefillVerifyArgmax: return "prefill-verify-argmax"
        }
    }

    public init?(cliName: String) {
        guard let value = Self.allCases.first(where: {
            $0.cliName == cliName || $0.rawValue == cliName
        }) else { return nil }
        self = value
    }

    func checksum(in manifest: SmeltManifest) -> String? {
        let value: String?
        switch self {
        case .decode:
            value = manifest.checksums.dispatchesBin
        case .prefill:
            value = manifest.checksums.prefillDispatchesBin
        case .prefillVerifyArgmax:
            value = manifest.checksums.prefillVerifyArgmaxDispatchesBin
        }
        guard let value, !value.isEmpty else { return nil }
        return value
    }
}

/// Static resource accounting for the dispatch records that the runtime will
/// actually replay. Measured samples may annotate the bill, but never fill an
/// unknown static descriptor with a heuristic timing.
public struct SmeltFrozenIRCostModel {
    public let manifest: SmeltManifest
    public let records: [SmeltDispatchRecord]
    public let calibration: SmeltDeviceCostCalibration?
    public let dispatchTable: SmeltFrozenIRDispatchTable
    public let dispatchTableSHA256: String?

    public init(
        manifest: SmeltManifest,
        records: [SmeltDispatchRecord],
        calibration: SmeltDeviceCostCalibration? = nil,
        dispatchTable: SmeltFrozenIRDispatchTable = .decode,
        dispatchTableSHA256: String? = nil
    ) {
        self.manifest = manifest
        self.records = records
        self.calibration = calibration
        self.dispatchTable = dispatchTable
        self.dispatchTableSHA256 = dispatchTableSHA256
            ?? dispatchTable.checksum(in: manifest)
    }

    public static func load(
        packagePath: String,
        calibration: SmeltDeviceCostCalibration? = nil,
        dispatchTable: SmeltFrozenIRDispatchTable = .decode
    ) throws -> SmeltFrozenIRCostModel {
        let root = URL(fileURLWithPath: packagePath, isDirectory: true)
        let manifest = try SmeltManifest.decode(
            from: Data(contentsOf: root.appendingPathComponent("manifest.json"))
        )
        let data = try Data(contentsOf: root.appendingPathComponent(dispatchTable.rawValue))
        let stride = MemoryLayout<SmeltDispatchRecord>.stride
        guard data.count.isMultiple(of: stride) else {
            throw SmeltFrozenIRCostModelError.invalidDispatchTable(
                "\(dispatchTable.rawValue) size \(data.count) is not a multiple of record stride \(stride)"
            )
        }
        var records = [SmeltDispatchRecord](
            repeating: .empty(),
            count: data.count / stride
        )
        _ = records.withUnsafeMutableBytes { destination in
            data.copyBytes(to: destination)
        }
        return SmeltFrozenIRCostModel(
            manifest: manifest,
            records: records,
            calibration: calibration,
            dispatchTable: dispatchTable
        )
    }

    public func report(
        context: SmeltCostModelContext,
        planID: String? = nil
    ) throws -> SmeltFrozenIRCostReport {
        var bills: [SmeltFrozenIRDispatchBill] = []
        bills.reserveCapacity(records.count)
        var currentSlot = 0
        var alternateSlot = 1
        var dispatchOrdinal = 0
        let calibrationStatus = calibrationStatus(context: context)
        let usableCalibration = calibrationStatus == .exactArtifactMatch
            ? calibration : nil
        let sampleLookup = Dictionary(
            usableCalibration?.dispatchSamples.map { ($0.key, $0) } ?? [],
            uniquingKeysWith: { _, newer in newer }
        )

        for (recordIndex, record) in records.enumerated() {
            if record.opKind == SmeltDispatchRecord.opSwap {
                bills.append(SmeltFrozenIRDispatchBill(
                    recordIndex: recordIndex,
                    dispatchOrdinal: nil,
                    pipeline: "swap",
                    executesGPU: false,
                    operationGroup: "runtime.swap",
                    geometry: nil,
                    intermediateMaterializationBytes: 0
                ))
                swap(&currentSlot, &alternateSlot)
                continue
            }
            guard record.opKind == SmeltDispatchRecord.opDispatch else {
                bills.append(SmeltFrozenIRDispatchBill(
                    recordIndex: recordIndex,
                    dispatchOrdinal: nil,
                    pipeline: "runtime_record_\(record.opKind)",
                    executesGPU: false,
                    operationGroup: nil,
                    geometry: nil,
                    intermediateMaterializationBytes: nil,
                    unknowns: ["unknown runtime record kind \(record.opKind)"]
                ))
                continue
            }
            let pipelineIndex = Int(record.pipeline)
            guard pipelineIndex < manifest.pipelines.count else {
                throw SmeltFrozenIRCostModelError.invalidPipelineIndex(
                    record: recordIndex,
                    pipeline: pipelineIndex
                )
            }
            let pipeline = manifest.pipelines[pipelineIndex]
            dispatchOrdinal += 1
            guard Self.executes(record, context: context) else {
                bills.append(SmeltFrozenIRDispatchBill(
                    recordIndex: recordIndex,
                    dispatchOrdinal: dispatchOrdinal,
                    pipeline: pipeline,
                    executesGPU: false,
                    operationGroup: "runtime.skipped_guard",
                    geometry: nil,
                    intermediateMaterializationBytes: 0
                ))
                continue
            }
            let geometry = Self.geometry(for: record, context: context)
            let costKey = Self.dispatchCostKey(
                record: record,
                pipeline: pipeline,
                context: context
            )
            let sample = sampleLookup[costKey]
            let resolvedConstants = resolvedConstants(record, context: context)
            let description = Self.describe(
                pipeline: pipeline,
                record: record,
                geometry: geometry,
                constants: resolvedConstants
            )
            var unknowns = description.unknowns
            for operation in description.operations where operation.count == nil {
                unknowns.append(
                    operation.unknownReason
                        ?? "\(operation.operationClass.rawValue) operation count is unknown"
                )
            }
            var resources: [SmeltFrozenIRResourceAccess] = []
            resources.reserveCapacity(Int(record.bufferCount))

            for index in 0..<Int(record.bufferCount) {
                let binding = getBuffer(record, index: index)
                let resolvedSlot: Int
                switch binding.slot {
                case SmeltBufferRecord.slotCur:
                    resolvedSlot = currentSlot
                case SmeltBufferRecord.slotAlt:
                    resolvedSlot = alternateSlot
                default:
                    resolvedSlot = Int(binding.slot)
                }
                let offset = Self.resolvedOffset(binding, context: context)
                let role = description.roles[Int(binding.bindingIndex)]
                let resource = resolveResource(
                    bindingIndex: Int(binding.bindingIndex),
                    slotIndex: resolvedSlot,
                    offset: offset,
                    role: role,
                    operationGroup: description.group,
                    context: context
                )
                resources.append(resource)
                if let reason = resource.unknownReason {
                    unknowns.append("buffer \(binding.bindingIndex): \(reason)")
                }
            }
            resources.sort { lhs, rhs in
                if lhs.bindingIndex != rhs.bindingIndex {
                    return lhs.bindingIndex < rhs.bindingIndex
                }
                if lhs.slotIndex != rhs.slotIndex {
                    return (lhs.slotIndex ?? -1) < (rhs.slotIndex ?? -1)
                }
                return lhs.byteOffset < rhs.byteOffset
            }
            unknowns = Array(Set(unknowns)).sorted()
            bills.append(SmeltFrozenIRDispatchBill(
                recordIndex: recordIndex,
                dispatchOrdinal: dispatchOrdinal,
                pipeline: pipeline,
                operationGroup: description.group,
                geometry: geometry,
                resources: resources,
                operations: description.operations.sorted {
                    $0.operationClass.rawValue < $1.operationClass.rawValue
                },
                synchronization: description.synchronization,
                intermediateMaterializationBytes: description.materializationBytes,
                calibratedMedianGPUUs: sample?.medianGPUUs,
                calibratedP95GPUUs: sample?.p95GPUUs,
                unknowns: unknowns
            ))
        }

        let build = manifest.buildProvenance?.buildFingerprint ?? "unknown-build"
        let provenanceKey = usableCalibration.map(structuredProvenance)
            ?? "package:build=\(build):table=\(dispatchTable.rawValue):dispatches=\(dispatchTableSHA256 ?? "unknown-dispatches"):metallib=\(manifest.checksums.metallib):weights=\(manifest.checksums.weightsBin)"
        return SmeltFrozenIRCostReport(
            planID: planID ?? dispatchTable.planID,
            provenanceKey: provenanceKey,
            context: context,
            summary: Self.summarize(
                bills,
                calibration: usableCalibration,
                calibrationStatus: calibrationStatus
            ),
            records: bills
        )
    }

    /// Cost any component-owned frozen plan through the same report schema as
    /// a package dispatch table. Selection and operation ordering have already
    /// happened; this layer only joins exact-shape measurements and totals the
    /// declared resource/operation bill.
    public static func report(
        plan: SmeltFrozenIRPlan,
        calibration: SmeltDeviceCostCalibration? = nil
    ) -> SmeltFrozenIRCostReport {
        let calibrationStatus: SmeltFrozenIRCalibrationStatus
        let usableCalibration: SmeltDeviceCostCalibration?
        if let calibration {
            if calibration.provenanceKey != plan.provenanceKey {
                calibrationStatus = .artifactMismatch
                usableCalibration = nil
            } else if let context = calibration.context, context != plan.context {
                calibrationStatus = .contextMismatch
                usableCalibration = nil
            } else {
                calibrationStatus = .exactArtifactMatch
                usableCalibration = calibration
            }
        } else {
            calibrationStatus = .absent
            usableCalibration = nil
        }
        let samples = Dictionary(
            usableCalibration?.dispatchSamples.map { ($0.key, $0) } ?? [],
            uniquingKeysWith: { _, newer in newer }
        )
        let records = plan.records.map { record -> SmeltFrozenIRDispatchBill in
            let sample = costKey(record: record, context: plan.context).flatMap {
                samples[$0]
            }
            return SmeltFrozenIRDispatchBill(
                recordIndex: record.recordIndex,
                dispatchOrdinal: record.dispatchOrdinal,
                pipeline: record.pipeline,
                executesGPU: record.executesGPU,
                operationGroup: record.operationGroup,
                geometry: record.geometry,
                resources: record.resources,
                operations: record.operations,
                synchronization: record.synchronization,
                intermediateMaterializationBytes: record.intermediateMaterializationBytes,
                hostRecordCount: record.hostRecordCount,
                calibratedMedianGPUUs: sample?.medianGPUUs
                    ?? record.calibratedMedianGPUUs,
                calibratedP95GPUUs: sample?.p95GPUUs
                    ?? record.calibratedP95GPUUs,
                unknowns: record.unknowns
            )
        }
        return SmeltFrozenIRCostReport(
            schemaVersion: plan.schemaVersion,
            planID: plan.planID,
            provenanceKey: plan.provenanceKey,
            context: plan.context,
            summary: summarize(
                records,
                calibration: usableCalibration,
                calibrationStatus: calibrationStatus
            ),
            records: records
        )
    }

    public static func costKey(
        record: SmeltFrozenIRDispatchBill,
        context: SmeltCostModelContext
    ) -> SmeltDispatchCostKey? {
        guard record.executesGPU, let geometry = record.geometry else { return nil }
        func dimension(_ values: [Int], _ index: Int, fallback: Int) -> Int {
            values.indices.contains(index) ? values[index] : fallback
        }
        return SmeltDispatchCostKey(
            pipeline: record.pipeline,
            style: geometry.style,
            gridWidth: dimension(geometry.grid, 0, fallback: 1),
            gridHeight: dimension(geometry.grid, 1, fallback: 1),
            gridDepth: dimension(geometry.grid, 2, fallback: 1),
            threadgroupWidth: dimension(geometry.threadgroup, 0, fallback: 0),
            threadgroupHeight: dimension(geometry.threadgroup, 1, fallback: 0),
            threadgroupDepth: dimension(geometry.threadgroup, 2, fallback: 0),
            // A frozen component calibration describes this exact scheduled
            // occurrence, including cache warmth and neighboring operations.
            // Preserve its identity rather than smearing one shape median over
            // every layer that happens to share geometry.
            functionConstants: ["frozen_record=\(record.recordIndex)"],
            context: context
        )
    }

    /// Aggregate positional component measurements into the same exact-key
    /// calibration consumed by package dispatch tables. Component identity and
    /// operation semantics remain in the frozen plan; this join is generic.
    public static func calibration(
        plan: SmeltFrozenIRPlan,
        profiles: [SmeltFrozenIRExecutionProfile],
        cleanWholePlanGPUUs: [Double]? = nil,
        hostRecordUs: Double = 0
    ) throws -> SmeltDeviceCostCalibration {
        guard !profiles.isEmpty else {
            throw SmeltFrozenIRCostModelError.emptyComponentProfile
        }
        let recordsByIndex = Dictionary(
            uniqueKeysWithValues: plan.records.map { ($0.recordIndex, $0) }
        )
        let expectedIndices = Set(
            plan.records.filter(\.executesGPU).map(\.recordIndex)
        )
        let cleanValues = cleanWholePlanGPUUs ?? []
        guard cleanValues.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            throw SmeltFrozenIRCostModelError.componentProfileMismatch(
                "whole-plan GPU samples contain an invalid value"
            )
        }
        let cleanTarget = cleanValues.isEmpty ? nil : median(cleanValues)
        var valuesByKey: [SmeltDispatchCostKey: [Double]] = [:]
        var instrumentedSpanGPUUs: [Double] = []
        for (profileIndex, profile) in profiles.enumerated() {
            guard profile.provenanceKey == plan.provenanceKey else {
                throw SmeltFrozenIRCostModelError.componentProfileMismatch(
                    "profile \(profileIndex) provenance '\(profile.provenanceKey)'"
                        + " != plan provenance '\(plan.provenanceKey)'"
                )
            }
            guard profile.context == plan.context else {
                throw SmeltFrozenIRCostModelError.componentProfileMismatch(
                    "profile \(profileIndex) context does not match the frozen plan"
                )
            }
            let rawSpanGPUUs = profile.spans.reduce(0) { $0 + $1.gpuUs }
            instrumentedSpanGPUUs.append(rawSpanGPUUs)
            let reconciliationTarget = cleanTarget ?? profile.wholePlanGPUUs
            guard rawSpanGPUUs.isFinite, rawSpanGPUUs > 0,
                  reconciliationTarget.isFinite, reconciliationTarget >= 0
            else {
                throw SmeltFrozenIRCostModelError.componentProfileMismatch(
                    "profile \(profileIndex) has invalid span/whole-plan timing"
                )
            }
            let reconciliationScale = reconciliationTarget / rawSpanGPUUs
            var seen: Set<Int> = []
            for (spanIndex, span) in profile.spans.enumerated() {
                guard !span.recordIndices.isEmpty,
                      span.gpuUs.isFinite,
                      span.gpuUs >= 0
                else {
                    throw SmeltFrozenIRCostModelError.componentProfileMismatch(
                        "profile \(profileIndex) span \(spanIndex) is invalid"
                    )
                }
                let spanRecords = try span.recordIndices.map { recordIndex in
                    guard seen.insert(recordIndex).inserted else {
                        throw SmeltFrozenIRCostModelError.componentProfileMismatch(
                            "profile \(profileIndex) repeats record \(recordIndex)"
                        )
                    }
                    guard let record = recordsByIndex[recordIndex], record.executesGPU else {
                        throw SmeltFrozenIRCostModelError.componentProfileMismatch(
                            "profile \(profileIndex) references absent/non-GPU record"
                                + " \(recordIndex)"
                        )
                    }
                    return record
                }
                let weights = spanRecords.map(staticApportionmentWeight)
                let weightTotal = weights.reduce(0, +)
                for (record, weight) in zip(spanRecords, weights) {
                    guard let key = costKey(record: record, context: plan.context) else {
                        continue
                    }
                    let apportioned = weightTotal > 0
                        ? span.gpuUs * weight / weightTotal * reconciliationScale
                        : span.gpuUs / Double(max(spanRecords.count, 1))
                            * reconciliationScale
                    valuesByKey[key, default: []].append(apportioned)
                }
            }
            guard seen == expectedIndices else {
                let missing = expectedIndices.subtracting(seen).sorted()
                let extra = seen.subtracting(expectedIndices).sorted()
                throw SmeltFrozenIRCostModelError.componentProfileMismatch(
                    "profile \(profileIndex) coverage mismatch; missing=\(missing)"
                        + " extra=\(extra)"
                )
            }
        }
        let samples = valuesByKey.map { key, values in
            SmeltDispatchCostSample(
                key: key,
                medianGPUUs: median(values),
                p95GPUUs: percentile95(values),
                sampleCount: values.count
            )
        }.sorted { $0.key.stableID < $1.key.stableID }
        let wholePlanValues = cleanWholePlanGPUUs ?? profiles.map(\.wholePlanGPUUs)
        let methods = Array(Set(profiles.map(\.measurementMethod))).sorted()
        let devices = Array(Set(profiles.compactMap(\.deviceName))).sorted()
        return SmeltDeviceCostCalibration(
            provenanceKey: plan.provenanceKey,
            deviceName: devices.count == 1 ? devices[0] : nil,
            measurementMethod: methods.joined(separator: "+")
                + "+multi-record-spans-apportioned-by-declared-work"
                + (cleanWholePlanGPUUs == nil
                    ? "+reconciled-to-profiled-whole-plan"
                    : "+reconciled-to-clean-whole-plan-command-buffer"),
            context: plan.context,
            dispatchTableName: plan.planID,
            hostRecordUs: hostRecordUs,
            wholePlanMedianGPUUs: median(wholePlanValues),
            wholePlanP95GPUUs: percentile95(wholePlanValues),
            instrumentedWholePlanMedianGPUUs: median(
                profiles.map(\.wholePlanGPUUs)
            ),
            instrumentedSpanMedianGPUUs: median(instrumentedSpanGPUUs),
            dispatchSamples: samples
        )
    }

    private static func staticApportionmentWeight(
        _ record: SmeltFrozenIRDispatchBill
    ) -> Double {
        var bytes: UInt64 = 0
        for resource in record.resources {
            guard let logicalBytes = resource.logicalBytes else { continue }
            let multiplier: UInt64 = resource.access == .readWrite ? 2 : 1
            bytes = add(bytes, logicalBytes.multipliedReportingOverflow(by: multiplier).overflow
                ? UInt64.max
                : logicalBytes * multiplier)
        }
        var operations: Double = 0
        for operation in record.operations {
            guard let count = operation.count else { continue }
            let multiplier: Double
            switch operation.operationClass {
            case .fp16Arithmetic, .fp32Arithmetic, .integerBitwise:
                multiplier = 1
            case .reduction:
                multiplier = 4
            case .transcendental:
                multiplier = 8
            case .unknown:
                multiplier = 1
            }
            operations += Double(count) * multiplier
        }
        return max(Double(bytes) + operations, 1)
    }

    /// Build exact-key calibration samples from the package runtime. Dispatch
    /// samples are isolated GPU timestamp measurements; the whole-plan timing
    /// is recorded alongside them so additive estimates can be audited rather
    /// than mistaken for final admission evidence.
    public func calibrate(
        runtime: SmeltRuntime,
        context: SmeltCostModelContext,
        warmup: Int = 3,
        iterations: Int = 20,
        provenanceKey: String? = nil
    ) throws -> SmeltDeviceCostCalibration {
        let safeWarmup = max(warmup, 0)
        let safeIterations = max(iterations, 1)
        let token: Int32 = 0
        let position = Int32(context.position)
        let prefillTokens = (0..<context.sequenceLength).map { Int32($0) }

        guard (context.mode == .decode && dispatchTable == .decode)
                || (context.mode == .prefill && dispatchTable == .prefill)
        else {
            throw SmeltFrozenIRCostModelError.unsupportedCalibrationTable(
                dispatchTable.rawValue
            )
        }

        for _ in 0..<safeWarmup {
            runtime.resetWorkingBuffers()
            switch context.mode {
            case .decode:
                _ = try runtime.profileDecodeStep(tokenId: token, position: position)
            case .prefill:
                _ = try runtime.profilePrefillStep(
                    tokenIds: prefillTokens,
                    startPos: position
                )
            }
        }
        var wholePlanGPUUs: [Double] = []
        var wholePlanEncodeUs: [Double] = []
        wholePlanGPUUs.reserveCapacity(safeIterations)
        wholePlanEncodeUs.reserveCapacity(safeIterations)
        for _ in 0..<safeIterations {
            runtime.resetWorkingBuffers()
            switch context.mode {
            case .decode:
                let sample = try runtime.profileDecodeStep(
                    tokenId: token,
                    position: position
                )
                wholePlanGPUUs.append(sample.pureGpuMs * 1_000)
                wholePlanEncodeUs.append(sample.cpuMs * 1_000)
            case .prefill:
                let sample = try runtime.profilePrefillStep(
                    tokenIds: prefillTokens,
                    startPos: position
                )
                wholePlanGPUUs.append(sample.pureGpuMs * 1_000)
                wholePlanEncodeUs.append(sample.cpuMs * 1_000)
            }
        }

        let staticReport = try report(context: context)
        var representatives: [SmeltDispatchCostKey: SmeltFrozenIRDispatchBill] = [:]
        for bill in staticReport.records where bill.executesGPU {
            guard bill.dispatchOrdinal != nil else { continue }
            let record = records[bill.recordIndex]
            let key = Self.dispatchCostKey(
                record: record,
                pipeline: bill.pipeline,
                context: context
            )
            if representatives[key] == nil {
                representatives[key] = bill
            }
        }

        var samples: [SmeltDispatchCostSample] = []
        samples.reserveCapacity(representatives.count)
        for (key, bill) in representatives.sorted(by: {
            $0.key.stableID < $1.key.stableID
        }) {
            guard let ordinal = bill.dispatchOrdinal else { continue }
            let profile: SmeltRuntime.DispatchBenchmarkProfile?
            switch context.mode {
            case .decode:
                profile = try runtime.benchmarkDecodeDispatch(
                    tokenId: token,
                    position: position,
                    dispatchOrdinal: ordinal,
                    warmup: safeWarmup,
                    iterations: safeIterations
                )
            case .prefill:
                profile = try runtime.benchmarkPrefillDispatch(
                    tokenIds: prefillTokens,
                    startPos: position,
                    dispatchOrdinal: ordinal,
                    warmup: safeWarmup,
                    iterations: safeIterations
                )
            }
            guard let profile else { continue }
            samples.append(SmeltDispatchCostSample(
                key: key,
                medianGPUUs: profile.medianGpuUs,
                p95GPUUs: profile.p95GpuUs,
                sampleCount: profile.samplesGpuUs.count
            ))
        }

        let deviceName = runtime.metalDevice.name
        let build = manifest.buildProvenance?.buildFingerprint
        let provenanceBuild = build ?? "unknown-build"
        let resolvedProvenance = provenanceKey
            ?? "device=\(deviceName):build=\(provenanceBuild):table=\(dispatchTable.rawValue):dispatches=\(dispatchTableSHA256 ?? "unknown-dispatches"):metallib=\(manifest.checksums.metallib):weights=\(manifest.checksums.weightsBin)"
        let hostPerRecord = Self.median(wholePlanEncodeUs)
            / Double(max(staticReport.summary.recordCount, 1))
        return SmeltDeviceCostCalibration(
            provenanceKey: resolvedProvenance,
            deviceName: deviceName,
            measurementMethod: "isolated-dispatch-gpu-timestamps+whole-plan-command-buffer",
            context: context,
            buildFingerprint: build,
            dispatchTableName: dispatchTable.rawValue,
            dispatchesSHA256: dispatchTableSHA256,
            metallibSHA256: manifest.checksums.metallib,
            weightsSHA256: manifest.checksums.weightsBin,
            hostRecordUs: hostPerRecord,
            wholePlanMedianGPUUs: Self.median(wholePlanGPUUs),
            wholePlanP95GPUUs: Self.percentile95(wholePlanGPUUs),
            dispatchSamples: samples.sorted { $0.key.stableID < $1.key.stableID }
        )
    }

    public static func dispatchCostKey(
        record: SmeltDispatchRecord,
        pipeline: String,
        context: SmeltCostModelContext
    ) -> SmeltDispatchCostKey {
        let geometry = geometry(for: record, context: context)
        let constants = (0..<Int(record.constantCount)).map { index -> String in
            let constant = getConstant(record, index: index)
            return "\(constant.bindingIndex):k\(constant.kind):\(constant.value)"
        }.sorted()
        return SmeltDispatchCostKey(
            pipeline: pipeline,
            style: geometry.style,
            gridWidth: geometry.grid[0],
            gridHeight: geometry.grid[1],
            gridDepth: geometry.grid[2],
            threadgroupWidth: geometry.threadgroup[0],
            threadgroupHeight: geometry.threadgroup[1],
            threadgroupDepth: geometry.threadgroup[2],
            functionConstants: constants,
            context: context
        )
    }

    public static func delta(
        baseline: SmeltFrozenIRCostReport,
        candidate: SmeltFrozenIRCostReport
    ) -> SmeltFrozenIRCostDelta {
        let baselineRead = baseline.summary.storageTotals.reduce(UInt64(0)) {
            add($0, $1.readBytes)
        }
        let candidateRead = candidate.summary.storageTotals.reduce(UInt64(0)) {
            add($0, $1.readBytes)
        }
        let baselineWrite = baseline.summary.storageTotals.reduce(UInt64(0)) {
            add($0, $1.writeBytes)
        }
        let candidateWrite = candidate.summary.storageTotals.reduce(UInt64(0)) {
            add($0, $1.writeBytes)
        }
        var unknowns: [String] = []
        let baselineStaticComplete = baseline.summary.unknownDispatches.isEmpty
        let candidateStaticComplete = candidate.summary.unknownDispatches.isEmpty
        if !baselineStaticComplete {
            unknowns.append("baseline has undescribed dispatches")
        }
        if !candidateStaticComplete {
            unknowns.append("candidate has undescribed dispatches")
        }
        let contextsMatch = baseline.context == candidate.context
        if !contextsMatch {
            unknowns.append("execution contexts differ")
        }
        let calibrationsComplete =
            baseline.summary.calibratedDispatchCount == baseline.summary.dispatchCount
            && candidate.summary.calibratedDispatchCount == candidate.summary.dispatchCount
        let calibrationProvenanceMatches = baseline.provenanceKey == candidate.provenanceKey
        if (baseline.summary.calibratedDispatchCount > 0
                || candidate.summary.calibratedDispatchCount > 0)
            && !calibrationsComplete
        {
            unknowns.append("isolated calibration is incomplete")
        }
        if calibrationsComplete && !calibrationProvenanceMatches {
            unknowns.append("calibration provenance differs")
        }
        let staticDeltaKnown = baselineStaticComplete && candidateStaticComplete
            && contextsMatch
        let calibratedDeltaKnown = calibrationsComplete
            && calibrationProvenanceMatches && contextsMatch
        let baselineHostRecords = baseline.records.reduce(0) { $0 + $1.hostRecordCount }
        let candidateHostRecords = candidate.records.reduce(0) { $0 + $1.hostRecordCount }
        return SmeltFrozenIRCostDelta(
            baselinePlanID: baseline.planID,
            candidatePlanID: candidate.planID,
            dispatchCountDelta: candidate.summary.dispatchCount
                - baseline.summary.dispatchCount,
            hostRecordCountDelta: candidateHostRecords - baselineHostRecords,
            logicalReadBytesDelta: staticDeltaKnown
                ? signedDelta(candidateRead, baselineRead) : nil,
            logicalWriteBytesDelta: staticDeltaKnown
                ? signedDelta(candidateWrite, baselineWrite) : nil,
            intermediateMaterializationBytesDelta: optionalSignedDelta(
                candidate.summary.intermediateMaterializationBytes,
                baseline.summary.intermediateMaterializationBytes
            ),
            calibratedMedianGPUUsDelta: calibratedDeltaKnown
                ? optionalDelta(
                    candidate.summary.calibratedMedianGPUUs,
                    baseline.summary.calibratedMedianGPUUs
                ) : nil,
            unknowns: unknowns.sorted()
        )
    }

    public static func markdown(
        _ report: SmeltFrozenIRCostReport,
        topDispatches: Int = 12
    ) -> String {
        let summary = report.summary
        let percent = summary.descriptorCoverageFraction * 100
        var lines = [
            "## Frozen \(report.context.mode.rawValue) cost",
            "",
            "- Plan: `\(report.planID)`",
            "- Provenance: `\(report.provenanceKey)`",
            "- Records: \(summary.recordCount) (\(summary.dispatchCount) GPU dispatches, \(summary.skippedDispatchCount) guarded skips, \(summary.swapCount) swaps)",
            String(format: "- Descriptor coverage: %d/%d (%.1f%%)", summary.describedDispatchCount, summary.dispatchCount, percent),
        ]
        if let status = summary.calibrationStatus {
            lines.append("- Calibration artifact status: `\(status.rawValue)`")
        }
        if let measuredCoverage = summary.measuredGPUCoverageFraction,
           let calibrated = summary.calibratedMedianGPUUs
        {
            lines.append(String(
                format: "- Isolated-sample coverage: %.1f%% of %.1f additive us",
                measuredCoverage * 100,
                calibrated
            ))
        } else {
            lines.append("- Isolated-sample coverage: unknown (no matching calibration)")
        }
        if let wholePlan = summary.measuredWholePlanMedianGPUUs {
            lines.append(String(format: "- Measured whole-plan median GPU: %.1f us", wholePlan))
        } else {
            lines.append("- Measured whole-plan median GPU: unknown")
        }
        if let instrumented = summary.instrumentedWholePlanMedianGPUUs {
            lines.append(String(
                format: "- Counter-instrumented whole-plan median GPU: %.1f us",
                instrumented
            ))
        }
        if let scale = summary.calibrationReconciliationScale {
            lines.append(String(
                format: "- Counter-to-clean reconciliation scale: %.6f",
                scale
            ))
        }
        if let error = summary.additiveCalibrationErrorFraction {
            lines.append(String(
                format: "- Additive-vs-whole-plan error: %+.1f%% (diagnostic; not an admission estimate)",
                error * 100
            ))
        }
        if let host = summary.predictedHostRecordUs {
            lines.append(String(format: "- Host recording estimate: %.1f us", host))
        } else {
            lines.append("- Host recording estimate: unknown")
        }
        lines.append("")
        lines.append("### Logical bytes")
        lines.append("")
        lines.append("| Storage | Read | Write |")
        lines.append("|---|---:|---:|")
        for total in summary.storageTotals where total.readBytes > 0 || total.writeBytes > 0 {
            lines.append("| \(total.storageClass.rawValue) | \(total.readBytes) | \(total.writeBytes) |")
        }

        struct GroupTotal {
            var dispatches = 0
            var streamingReadBytes: UInt64 = 0
            var stateReadWriteBytes: UInt64 = 0
            var materializationBytes: UInt64 = 0
            var additiveGPUUs: Double = 0
            var calibratedDispatches = 0
        }
        var groupTotals: [String: GroupTotal] = [:]
        for bill in report.records where bill.executesGPU {
            let group = bill.operationGroup ?? "unknown"
            var total = groupTotals[group] ?? GroupTotal()
            total.dispatches += 1
            for resource in bill.resources {
                guard let bytes = resource.logicalBytes else { continue }
                if resource.storageClass == .streamingWeight,
                   resource.access == .read || resource.access == .readWrite
                {
                    total.streamingReadBytes = add(total.streamingReadBytes, bytes)
                }
                if resource.storageClass == .persistentState {
                    switch resource.access {
                    case .read, .write:
                        total.stateReadWriteBytes = add(total.stateReadWriteBytes, bytes)
                    case .readWrite:
                        total.stateReadWriteBytes = add(
                            total.stateReadWriteBytes,
                            add(bytes, bytes)
                        )
                    case .unknown:
                        break
                    }
                }
            }
            if let bytes = bill.intermediateMaterializationBytes {
                total.materializationBytes = add(total.materializationBytes, bytes)
            }
            if let timing = bill.calibratedMedianGPUUs {
                total.additiveGPUUs += timing
                total.calibratedDispatches += 1
            }
            groupTotals[group] = total
        }
        if !groupTotals.isEmpty {
            lines.append("")
            lines.append("### Operation-group bill")
            lines.append("")
            lines.append("| Group | Dispatches | Streaming read bytes | State R+W bytes | Materialized bytes | Additive GPU us |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for (group, total) in groupTotals.sorted(by: { lhs, rhs in
                if lhs.value.additiveGPUUs != rhs.value.additiveGPUUs {
                    return lhs.value.additiveGPUUs > rhs.value.additiveGPUUs
                }
                if lhs.value.streamingReadBytes != rhs.value.streamingReadBytes {
                    return lhs.value.streamingReadBytes > rhs.value.streamingReadBytes
                }
                return lhs.key < rhs.key
            }) {
                let timing = total.calibratedDispatches == total.dispatches
                    ? String(format: "%.1f", total.additiveGPUUs)
                    : "unknown"
                lines.append(
                    "| \(group) | \(total.dispatches) | \(total.streamingReadBytes) | \(total.stateReadWriteBytes) | \(total.materializationBytes) | \(timing) |"
                )
            }
        }

        let costly = report.records
            .filter(\.executesGPU)
            .sorted {
                let lhs = $0.calibratedMedianGPUUs ?? -1
                let rhs = $1.calibratedMedianGPUUs ?? -1
                if lhs != rhs { return lhs > rhs }
                return ($0.dispatchOrdinal ?? 0) < ($1.dispatchOrdinal ?? 0)
            }
            .prefix(max(topDispatches, 0))
        if !costly.isEmpty {
            lines.append("")
            lines.append("### Dispatches")
            lines.append("")
            lines.append("| Ordinal | Pipeline | Group | Median GPU us | Unknowns |")
            lines.append("|---:|---|---|---:|---|")
            for bill in costly {
                let timing = bill.calibratedMedianGPUUs.map {
                    String(format: "%.2f", $0)
                } ?? "unknown"
                lines.append(
                    "| \(bill.dispatchOrdinal ?? -1) | `\(bill.pipeline)` | \(bill.operationGroup ?? "unknown") | \(timing) | \(bill.unknowns.joined(separator: "; ")) |"
                )
            }
        }
        if !summary.unknownDispatches.isEmpty {
            lines.append("")
            lines.append("Unknown pipeline descriptors: " + summary.unknownDispatches.joined(separator: ", "))
        }
        lines.append("")
        return lines.joined(separator: "\n")
    }

    public static func markdown(_ delta: SmeltFrozenIRCostDelta) -> String {
        func signed(_ value: Int) -> String {
            value >= 0 ? "+\(value)" : "\(value)"
        }
        func signed(_ value: Int64?) -> String {
            guard let value else { return "unknown" }
            return value >= 0 ? "+\(value)" : "\(value)"
        }
        func signed(_ value: Double?) -> String {
            guard let value else { return "unknown" }
            return String(format: "%+.2f", value)
        }
        var lines = [
            "## Frozen cost delta",
            "",
            "Candidate `\(delta.candidatePlanID)` minus baseline `\(delta.baselinePlanID)`:",
            "",
            "- GPU dispatches: \(signed(delta.dispatchCountDelta))",
            "- Host records: \(signed(delta.hostRecordCountDelta))",
            "- Logical reads: \(signed(delta.logicalReadBytesDelta)) bytes",
            "- Logical writes: \(signed(delta.logicalWriteBytesDelta)) bytes",
            "- Intermediate materialization: \(signed(delta.intermediateMaterializationBytesDelta)) bytes",
            "- Additive calibrated median GPU: \(signed(delta.calibratedMedianGPUUsDelta)) us",
        ]
        if !delta.unknowns.isEmpty {
            lines.append("- Uncertainty: " + delta.unknowns.joined(separator: "; "))
        }
        lines.append("")
        return lines.joined(separator: "\n")
    }

    private struct ResourceRole {
        let access: SmeltFrozenIRAccessKind
        let logicalBytes: UInt64?
        /// True when the launch geometry deterministically replays one unique
        /// resource region across independent tiles. In that case logical
        /// traffic may exceed the backing region's unique byte span.
        let permitsRepeatedTraffic: Bool

        init(
            access: SmeltFrozenIRAccessKind,
            logicalBytes: UInt64?,
            permitsRepeatedTraffic: Bool = false
        ) {
            self.access = access
            self.logicalBytes = logicalBytes
            self.permitsRepeatedTraffic = permitsRepeatedTraffic
        }
    }

    private struct KernelDescription {
        let group: String?
        let roles: [Int: ResourceRole]
        let operations: [SmeltFrozenIROperationBill]
        let synchronization: SmeltFrozenIRSynchronizationClass
        let materializationBytes: UInt64?
        let unknowns: [String]

        static func unknown(_ pipeline: String) -> KernelDescription {
            KernelDescription(
                group: nil,
                roles: [:],
                operations: [SmeltFrozenIROperationBill(
                    operationClass: .unknown,
                    count: nil,
                    unknownReason: "missing descriptor for \(pipeline)"
                )],
                synchronization: .unknown,
                materializationBytes: nil,
                unknowns: ["missing operation-group descriptor"]
            )
        }
    }

    private struct WeightRegion {
        let start: UInt64
        let size: UInt64
        let name: String
    }

    private func resolveResource(
        bindingIndex: Int,
        slotIndex: Int,
        offset: UInt64,
        role: ResourceRole?,
        operationGroup: String?,
        context: SmeltCostModelContext
    ) -> SmeltFrozenIRResourceAccess {
        guard let slot = manifest.buffers.slots.first(where: { $0.index == slotIndex }) else {
            return SmeltFrozenIRResourceAccess(
                bindingIndex: bindingIndex,
                slotIndex: slotIndex,
                resourceName: "slot_\(slotIndex)",
                storageClass: .unknown,
                access: role?.access ?? .unknown,
                byteOffset: offset,
                logicalBytes: role?.logicalBytes,
                unknownReason: "slot is absent from manifest buffer table"
            )
        }
        let storageClass = Self.storageClass(
            slot: slot,
            layout: manifest.slotLayout
        )
        var name = slot.name
        var regionBytes: UInt64?
        var reason: String?
        if slotIndex == manifest.slotLayout.weightsSlot {
            if let span = resolveWeightSpan(
                offset: offset,
                requestedBytes: role?.logicalBytes
            ) {
                name = span.name
                regionBytes = span.availableBytes
            } else {
                reason = "weight offset \(offset) is absent from the weight manifest"
            }
        } else {
            let bufferBytes = logicalBufferSize(slot: slot, context: context)
            if offset < bufferBytes {
                regionBytes = bufferBytes - offset
            } else if bufferBytes > 0 {
                reason = "offset \(offset) exceeds \(slot.name) size \(bufferBytes)"
            }
        }
        let bytes = role?.logicalBytes ?? (operationGroup == nil ? nil : regionBytes)
        if operationGroup != nil, role == nil {
            reason = reason ?? "descriptor does not declare buffer \(bindingIndex)"
        }
        if operationGroup != nil, role != nil, role?.logicalBytes == nil {
            reason = reason ?? "descriptor logical span is unknown"
        }
        if let bytes, let regionBytes, bytes > regionBytes,
           role?.permitsRepeatedTraffic != true
        {
            reason = reason ?? "descriptor span \(bytes) exceeds available region \(regionBytes)"
        }
        return SmeltFrozenIRResourceAccess(
            bindingIndex: bindingIndex,
            slotIndex: slotIndex,
            resourceName: name,
            storageClass: storageClass,
            access: role?.access ?? .unknown,
            byteOffset: offset,
            logicalBytes: bytes,
            unknownReason: reason
        )
    }

    /// Dynamic-context buffer shapes use a zero extent in the manifest. The
    /// runtime replaces that extent with request context capacity before
    /// executing the frozen table. Reconstruct the minimum capacity required
    /// by this exact decode context so bounds checks do not mistake valid
    /// position-relative cache/table accesses for out-of-range reads.
    private func logicalBufferSize(
        slot: SmeltBufferSlot,
        context: SmeltCostModelContext
    ) -> UInt64 {
        guard slot.shape.contains(0), let bytesPerElement = slot.dtype.bytesPerElement else {
            return UInt64(max(slot.sizeBytes, 0))
        }
        let contextExtent = UInt64(max(context.sequenceLength, context.position + 1, 1))
        var elements: UInt64 = 1
        for extent in slot.shape {
            let resolved = extent == 0 ? contextExtent : UInt64(max(extent, 0))
            guard let product = Self.mul(elements, resolved) else {
                return UInt64.max
            }
            elements = product
        }
        guard let dynamicBytes = Self.mul(elements, UInt64(bytesPerElement)) else {
            return UInt64.max
        }
        return max(dynamicBytes, UInt64(max(slot.sizeBytes, 0)))
    }

    private func weightRegions() -> [WeightRegion] {
        var regions: [WeightRegion] = []
        regions.reserveCapacity(manifest.weights.entries.count * 4)
        for entry in manifest.weights.entries {
            regions.append(WeightRegion(
                start: entry.offset,
                size: entry.sizeBytes,
                name: entry.name
            ))
            if let start = entry.scalesOffset, let size = entry.scalesSizeBytes {
                regions.append(WeightRegion(
                    start: start,
                    size: size,
                    name: entry.name + ".scales"
                ))
            }
            if let start = entry.biasesOffset, let size = entry.biasesSizeBytes {
                regions.append(WeightRegion(
                    start: start,
                    size: size,
                    name: entry.name + ".biases"
                ))
            }
            if let start = entry.lutOffset, let size = entry.lutSizeBytes {
                regions.append(WeightRegion(
                    start: start,
                    size: size,
                    name: entry.name + ".lut"
                ))
            }
            if let start = entry.codebookOffset, let size = entry.codebookSizeBytes {
                regions.append(WeightRegion(
                    start: start,
                    size: size,
                    name: entry.name + ".codebook"
                ))
            }
        }
        return regions.sorted {
            if $0.start != $1.start { return $0.start < $1.start }
            return $0.name < $1.name
        }
    }

    private func resolveWeightSpan(
        offset: UInt64,
        requestedBytes: UInt64?
    ) -> (name: String, availableBytes: UInt64)? {
        let regions = weightRegions()
        guard var index = regions.firstIndex(where: {
            offset >= $0.start && offset < Self.add($0.start, $0.size)
        }) else { return nil }
        let first = regions[index]
        var cursor = offset
        var available = first.size - (offset - first.start)
        var names = [first.name]
        let requested = requestedBytes ?? available
        while available < requested {
            cursor = Self.add(offset, available)
            index += 1
            guard index < regions.count, regions[index].start == cursor else {
                break
            }
            available = Self.add(available, regions[index].size)
            names.append(regions[index].name)
        }
        let name = names.count == 1
            ? names[0]
            : names[0] + "+\(names.count - 1)_contiguous_regions"
        return (name, available)
    }

    public static func summarize(
        _ bills: [SmeltFrozenIRDispatchBill],
        calibration: SmeltDeviceCostCalibration?,
        calibrationStatus: SmeltFrozenIRCalibrationStatus
    ) -> SmeltFrozenIRCostSummary {
        let dispatches = bills.filter(\.executesGPU)
        let described = dispatches.filter(\.isDescribed)
        var storage: [SmeltFrozenIRStorageClass: (read: UInt64, write: UInt64)] = [:]
        var operations: [SmeltFrozenIROperationClass: UInt64] = [:]
        var synchronization: [String: Int] = [:]
        var materialization: UInt64 = 0
        var materializationComplete = true
        for bill in bills {
            synchronization[bill.synchronization.rawValue, default: 0] += 1
            if let bytes = bill.intermediateMaterializationBytes {
                materialization = Self.add(materialization, bytes)
            } else {
                materializationComplete = false
            }
            for resource in bill.resources {
                guard let bytes = resource.logicalBytes else { continue }
                var total = storage[resource.storageClass] ?? (0, 0)
                switch resource.access {
                case .read:
                    total.read = Self.add(total.read, bytes)
                case .write:
                    total.write = Self.add(total.write, bytes)
                case .readWrite:
                    total.read = Self.add(total.read, bytes)
                    total.write = Self.add(total.write, bytes)
                case .unknown:
                    break
                }
                storage[resource.storageClass] = total
            }
            for operation in bill.operations {
                if let count = operation.count {
                    operations[operation.operationClass] = Self.add(
                        operations[operation.operationClass, default: 0],
                        count
                    )
                }
            }
        }
        let calibrated = dispatches.compactMap(\.calibratedMedianGPUUs)
        let calibratedTotal = calibrated.isEmpty ? nil : calibrated.reduce(0, +)
        let describedCalibrated = described.compactMap(\.calibratedMedianGPUUs)
        let describedCalibratedTotal = describedCalibrated.isEmpty
            ? nil : describedCalibrated.reduce(0, +)
        let measuredCoverage: Double?
        if let total = calibratedTotal, total > 0 {
            measuredCoverage = (describedCalibratedTotal ?? 0) / total
        } else {
            measuredCoverage = nil
        }
        let wholePlan = calibration?.wholePlanMedianGPUUs
        let instrumentedSpan = calibration?.instrumentedSpanMedianGPUUs
        let reconciliationScale: Double?
        if let wholePlan, let instrumentedSpan, instrumentedSpan > 0 {
            reconciliationScale = wholePlan / instrumentedSpan
        } else {
            reconciliationScale = nil
        }
        let additiveError: Double?
        if let calibratedTotal, let wholePlan, wholePlan > 0 {
            additiveError = (calibratedTotal - wholePlan) / wholePlan
        } else {
            additiveError = nil
        }
        return SmeltFrozenIRCostSummary(
            recordCount: bills.count,
            dispatchCount: dispatches.count,
            skippedDispatchCount: bills.filter {
                $0.dispatchOrdinal != nil && !$0.executesGPU
            }.count,
            swapCount: bills.filter { $0.pipeline == "swap" }.count,
            describedDispatchCount: described.count,
            descriptorCoverageFraction: dispatches.isEmpty
                ? 1 : Double(described.count) / Double(dispatches.count),
            calibratedDispatchCount: calibrated.count,
            calibratedMedianGPUUs: calibratedTotal,
            describedCalibratedMedianGPUUs: describedCalibratedTotal,
            measuredGPUCoverageFraction: measuredCoverage,
            measuredWholePlanMedianGPUUs: wholePlan,
            additiveCalibrationErrorFraction: additiveError,
            instrumentedWholePlanMedianGPUUs:
                calibration?.instrumentedWholePlanMedianGPUUs,
            instrumentedSpanMedianGPUUs: instrumentedSpan,
            calibrationReconciliationScale: reconciliationScale,
            predictedHostRecordUs: calibration.map {
                Double(bills.reduce(0) { $0 + $1.hostRecordCount }) * $0.hostRecordUs
            },
            calibrationStatus: calibrationStatus,
            storageTotals: SmeltFrozenIRStorageClass.allCases.compactMap { kind in
                guard let value = storage[kind] else { return nil }
                return SmeltFrozenIRStorageTotal(
                    storageClass: kind,
                    readBytes: value.read,
                    writeBytes: value.write
                )
            },
            operationTotals: SmeltFrozenIROperationClass.allCases.compactMap { kind in
                guard let count = operations[kind] else { return nil }
                return SmeltFrozenIROperationTotal(operationClass: kind, count: count)
            },
            intermediateMaterializationBytes: materializationComplete
                ? materialization : nil,
            synchronizationCounts: synchronization.keys.sorted().map {
                SmeltFrozenIRStringCount(name: $0, count: synchronization[$0] ?? 0)
            },
            unknownDispatches: Array(Set(
                bills.filter {
                    !$0.unknowns.isEmpty || ($0.executesGPU && !$0.isDescribed)
                }.map(\.pipeline)
            )).sorted()
        )
    }

    private func calibrationStatus(
        context: SmeltCostModelContext
    ) -> SmeltFrozenIRCalibrationStatus {
        guard let calibration else { return .absent }
        guard let deviceName = calibration.deviceName, !deviceName.isEmpty,
              let measurementMethod = calibration.measurementMethod,
              !measurementMethod.isEmpty,
              let calibrationContext = calibration.context,
              let dispatches = calibration.dispatchesSHA256,
              let metallib = calibration.metallibSHA256,
              let weights = calibration.weightsSHA256
        else {
            return .legacyUnstructured
        }
        guard calibrationContext == context else {
            return .contextMismatch
        }
        if let expectedBuild = manifest.buildProvenance?.buildFingerprint {
            guard calibration.buildFingerprint == expectedBuild else {
                return .artifactMismatch
            }
        }
        let recordedTable = calibration.dispatchTableName ?? SmeltFrozenIRDispatchTable.decode.rawValue
        guard recordedTable == dispatchTable.rawValue,
              dispatches == dispatchTableSHA256,
              metallib == manifest.checksums.metallib,
              weights == manifest.checksums.weightsBin
        else {
            return .artifactMismatch
        }
        return .exactArtifactMatch
    }

    private func structuredProvenance(
        _ calibration: SmeltDeviceCostCalibration
    ) -> String {
        let device = calibration.deviceName ?? "unknown-device"
        let build = calibration.buildFingerprint ?? "unknown-build"
        let table = calibration.dispatchTableName ?? SmeltFrozenIRDispatchTable.decode.rawValue
        let dispatches = calibration.dispatchesSHA256 ?? "unknown-dispatches"
        let metallib = calibration.metallibSHA256 ?? "unknown-metallib"
        let weights = calibration.weightsSHA256 ?? "unknown-weights"
        return "device=\(device):build=\(build):table=\(table):dispatches=\(dispatches):metallib=\(metallib):weights=\(weights)"
    }

    private static func describe(
        pipeline: String,
        record: SmeltDispatchRecord,
        geometry: SmeltFrozenIRGeometry,
        constants: [Int: UInt64]
    ) -> KernelDescription {
        if pipeline.contains("signed_") && pipeline.contains("bitplane_")
            && pipeline.contains("gate_up_swiglu")
        {
            return signedGateUp(pipeline: pipeline, constants: constants)
        }
        if pipeline.contains("signed_") && pipeline.contains("bitplane_")
            && pipeline.contains("bank4_matvec")
        {
            return signedBank4(pipeline: pipeline, constants: constants)
        }
        if pipeline.contains("signed_") && pipeline.contains("bitplane_")
            && pipeline.contains("matvec")
        {
            return signedMatvec(pipeline: pipeline, constants: constants)
        }
        if pipeline.hasPrefix("signed_ternary_affine_")
            && pipeline.contains("qmm")
        {
            return signedTernaryAffineQMM(constants: constants)
        }
        if pipeline.hasPrefix("signed_ternary_affine_")
            && pipeline.contains("bank4_matvec")
        {
            return signedTernaryAffineBank4(
                constants: constants, geometry: geometry)
        }
        if pipeline.hasPrefix("signed_ternary_affine_")
            && pipeline.contains("gate_up_swiglu")
        {
            return signedTernaryAffineGateUp(
                constants: constants, geometry: geometry)
        }
        if pipeline.hasPrefix("signed_ternary_affine_")
            && pipeline.contains("matvec")
        {
            return signedTernaryAffineMatvec(
                pipeline: pipeline, constants: constants, geometry: geometry)
        }
        if pipeline.hasPrefix("signed_") && pipeline.contains("gate_up_swiglu") {
            return signedDirectGateUp(pipeline: pipeline, constants: constants)
        }
        if pipeline.hasPrefix("signed_") && pipeline.contains("matvec") {
            return signedDirectMatvec(pipeline: pipeline, constants: constants)
        }
        if pipeline.contains("signed_") && pipeline.contains("embedding_gather") {
            return signedEmbedding(
                pipeline: pipeline,
                constants: constants,
                geometry: geometry
            )
        }
        if pipeline.hasPrefix("rms_norm_gated_d128_signed_activation_bitplanes_") {
            return gatedRMSActivation(pipeline: pipeline, constants: constants, geometry: geometry)
        }
        if pipeline.contains("signed_activation_bitplanes_") {
            return signedActivation(
                pipeline: pipeline, constants: constants, geometry: geometry)
        }
        if pipeline == "residual_add_rms_norm_scale_only_precise" {
            return residualNormScale(constants: constants)
        }
        if pipeline.hasPrefix("deltanet_recurrence_mlx_decode")
            || pipeline.hasPrefix("deltanet_recurrence_mlx_prefill")
        {
            return deltaNetRecurrence(
                pipeline: pipeline,
                constants: constants,
                geometry: geometry
            )
        }
        if pipeline.hasPrefix("conv1d_update_silu") {
            return conv1d(
                pipeline: pipeline, constants: constants, geometry: geometry)
        }
        if pipeline.hasPrefix("rope_kv_cache_update") {
            return kvCache(constants: constants, rope: true)
        }
        if pipeline.hasPrefix("kv_cache_update") {
            return kvCache(constants: constants, rope: false)
        }
        if pipeline.hasPrefix("attention_decode") {
            return attention(pipeline: pipeline, constants: constants, geometry: geometry)
        }
        if pipeline.hasPrefix("attention_prefill") {
            return attentionPrefill(constants: constants, geometry: geometry)
        }
        if pipeline == "rope_and_kv_cache_prefill" {
            return ropeAndKVPrefill(constants: constants, geometry: geometry)
        }
        if pipeline.hasPrefix("per_head_rms_norm") {
            return perHeadRMSNorm(
                pipeline: pipeline,
                constants: constants,
                geometry: geometry
            )
        }
        if pipeline == "rms_norm_gated_d128_batched" {
            return gatedRMSNormBatched(constants: constants, geometry: geometry)
        }
        if pipeline == "rms_norm_gated_d128" {
            return gatedRMSNorm(geometry: geometry)
        }
        if pipeline.hasPrefix("rms_norm_scale_only") {
            return rmsScaleOnly(constants: constants)
        }
        if pipeline.hasPrefix("rms_norm_1pw") {
            return rmsNorm(
                constants: constants,
                pipeline: pipeline,
                geometry: geometry
            )
        }
        if pipeline == "rms_scale_qk" {
            return rmsScaleQK(constants: constants, geometry: geometry)
        }
        if pipeline == "apply_rope" {
            return applyRoPE(constants: constants)
        }
        if pipeline == "argmax_fp16_partials" {
            return argmaxPartials(constants: constants)
        }
        if pipeline == "argmax_key_reduce" {
            return argmaxReduce(constants: constants)
        }
        if pipeline == "gate_split" {
            return gateSplit(constants: constants)
        }
        if pipeline == "elementwise_add" {
            return elementwiseAdd(constants: constants, geometry: geometry)
        }
        if pipeline == "elementwise_mul" {
            return elementwiseMultiply(constants: constants, geometry: geometry)
        }
        if pipeline == "sigmoid_kernel" {
            return sigmoid(constants: constants, geometry: geometry)
        }
        if pipeline == "sigmoid_mul" {
            return sigmoidMultiply(constants: constants, geometry: geometry)
        }
        if pipeline == "swiglu_fused" || pipeline == "geglu_fused" {
            return gatedLinearUnit(pipeline: pipeline, constants: constants)
        }
        return .unknown(pipeline)
    }

    private static func signedMatvec(
        pipeline: String,
        constants: [Int: UInt64]
    ) -> KernelDescription {
        let addedResidual = pipeline.contains("matvec_add")
        let rowsIndex = addedResidual ? 6 : 5
        let colsIndex = addedResidual ? 7 : 6
        guard let rows = constants[rowsIndex], let cols = constants[colsIndex],
              let planes = planeCount(pipeline) else {
            return incomplete("signed.bitplane_matvec", "missing rows, cols, or plane count")
        }
        let groups = cols / 128
        let ternary = pipeline.contains("ternary")
        let batched = pipeline.contains("batched")
        let batch = batched ? (constants[7] ?? 1) : 1
        let batchTile = UInt64(
            embeddedDimension(pipeline, marker: "_b") ?? 1)
        let weightReplays = batched
            ? (batch + batchTile - 1) / batchTile
            : 1
        let codeBytes = mul(rows, cols / (ternary ? 4 : 8))
        let scaleBytes = mul(rows, groups, 2)
        var roles: [Int: ResourceRole] = [
            0: ResourceRole(
                access: .read,
                logicalBytes: mul(weightReplays, codeBytes ?? 0),
                permitsRepeatedTraffic: weightReplays > 1
            ),
            1: ResourceRole(
                access: .read,
                logicalBytes: mul(weightReplays, scaleBytes ?? 0),
                permitsRepeatedTraffic: weightReplays > 1
            ),
            2: ResourceRole(
                access: .read,
                logicalBytes: mul(batch, groups, UInt64(planes), 16)
            ),
            3: ResourceRole(
                access: .read, logicalBytes: mul(batch, groups, 2)),
            4: ResourceRole(
                access: .write, logicalBytes: mul(batch, rows, 2)),
        ]
        if addedResidual {
            roles[5] = ResourceRole(access: .read, logicalBytes: mul(rows, 2))
        }
        return KernelDescription(
            group: batched
                ? "signed.bitplane_matvec_batched_b\(batchTile)"
                : "signed.bitplane_matvec",
            roles: roles,
            operations: [
                operation(
                    .integerBitwise,
                    mul(batch, rows, cols, UInt64(max(planes - 1, 1)))
                ),
                operation(.fp32Arithmetic, mul(batch, rows, groups, 3)),
                operation(.reduction, mul(batch, rows, groups)),
            ],
            synchronization: .simdgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func signedGateUp(
        pipeline: String,
        constants: [Int: UInt64]
    ) -> KernelDescription {
        guard let rows = constants[7], let cols = constants[8],
              let planes = planeCount(pipeline) else {
            return incomplete("signed.bitplane_gate_up_swiglu", "missing rows, cols, or plane count")
        }
        let groups = cols / 128
        let ternary = pipeline.contains("ternary")
        let codeBytes = mul(rows, cols / (ternary ? 4 : 8))
        let scaleBytes = mul(rows, groups, 2)
        return KernelDescription(
            group: "signed.bitplane_gate_up_swiglu",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: codeBytes),
                1: ResourceRole(access: .read, logicalBytes: scaleBytes),
                2: ResourceRole(access: .read, logicalBytes: codeBytes),
                3: ResourceRole(access: .read, logicalBytes: scaleBytes),
                4: ResourceRole(access: .read, logicalBytes: mul(groups, UInt64(planes), 16)),
                5: ResourceRole(access: .read, logicalBytes: mul(groups, 2)),
                6: ResourceRole(access: .write, logicalBytes: mul(rows, 2)),
            ],
            operations: [
                operation(.integerBitwise, mul(rows, cols, 2, UInt64(max(planes - 1, 1)))),
                operation(.fp32Arithmetic, mul(rows, groups, 6)),
                operation(.transcendental, rows),
                operation(.reduction, mul(rows, groups, 2)),
            ],
            synchronization: .simdgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func signedBank4(
        pipeline: String,
        constants: [Int: UInt64]
    ) -> KernelDescription {
        guard let r0 = constants[8], let r1 = constants[9],
              let r2 = constants[10], let r3 = constants[11],
              let cols = constants[12], let planes = planeCount(pipeline) else {
            return incomplete("signed.bitplane_bank4_matvec", "missing rows, cols, or plane count")
        }
        let rows = add(add(r0, r1), add(r2, r3))
        let groups = cols / 128
        let ternary = pipeline.contains("ternary")
        return KernelDescription(
            group: "signed.bitplane_bank4_matvec",
            roles: [
                0: ResourceRole(
                    access: .read,
                    logicalBytes: mul(rows, cols / (ternary ? 4 : 8))),
                1: ResourceRole(access: .read, logicalBytes: mul(rows, groups, 2)),
                2: ResourceRole(access: .read, logicalBytes: mul(groups, UInt64(planes), 16)),
                3: ResourceRole(access: .read, logicalBytes: mul(groups, 2)),
                4: ResourceRole(access: .write, logicalBytes: mul(r0, 2)),
                5: ResourceRole(access: .write, logicalBytes: mul(r1, 2)),
                6: ResourceRole(access: .write, logicalBytes: mul(r2, 2)),
                7: ResourceRole(access: .write, logicalBytes: mul(r3, 2)),
            ],
            operations: [
                operation(.integerBitwise, mul(rows, cols, UInt64(max(planes - 1, 1)))),
                operation(.fp32Arithmetic, mul(rows, groups, 3)),
                operation(.reduction, mul(rows, groups)),
            ],
            synchronization: .simdgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func signedActivation(
        pipeline: String,
        constants: [Int: UInt64],
        geometry: SmeltFrozenIRGeometry? = nil
    ) -> KernelDescription {
        guard let planes = planeCount(pipeline) else {
            return incomplete("signed.activation_bitplanes", "missing plane count")
        }
        let normScale = pipeline.hasPrefix("norm_scale_")
        let fusedTwoInput = pipeline.hasPrefix("sigmoid_mul_")
            || pipeline.hasPrefix("swiglu_")
        let colsIndex = normScale ? 5 : (fusedTwoInput ? 4 : 3)
        guard let cols = constants[colsIndex] else {
            return incomplete("signed.activation_bitplanes", "missing cols")
        }
        let groups = cols / 128
        let batch = pipeline.contains("batched")
            ? UInt64(max(geometry?.grid[1] ?? 1, 1))
            : 1
        let planeBytes = mul(batch, groups, UInt64(planes), 16)
        let scaleBytes = mul(batch, groups, 2)
        let roles: [Int: ResourceRole]
        if normScale {
            roles = [
                0: ResourceRole(access: .read, logicalBytes: 4),
                1: ResourceRole(access: .read, logicalBytes: mul(cols, 2)),
                2: ResourceRole(access: .read, logicalBytes: mul(cols, 2)),
                3: ResourceRole(access: .write, logicalBytes: planeBytes),
                4: ResourceRole(access: .write, logicalBytes: scaleBytes),
            ]
        } else if fusedTwoInput {
            roles = [
                0: ResourceRole(
                    access: .read, logicalBytes: mul(batch, cols, 2)),
                1: ResourceRole(access: .read, logicalBytes: mul(cols, 2)),
                2: ResourceRole(access: .write, logicalBytes: planeBytes),
                3: ResourceRole(access: .write, logicalBytes: scaleBytes),
            ]
        } else {
            roles = [
                0: ResourceRole(
                    access: .read, logicalBytes: mul(batch, cols, 2)),
                1: ResourceRole(access: .write, logicalBytes: planeBytes),
                2: ResourceRole(access: .write, logicalBytes: scaleBytes),
            ]
        }
        return KernelDescription(
            group: pipeline.contains("batched")
                ? "signed.activation_bitplanes_batched"
                : "signed.activation_bitplanes",
            roles: roles,
            operations: [
                operation(
                    .fp32Arithmetic, mul(batch, cols, UInt64(planes + 4))),
                operation(
                    .integerBitwise, mul(batch, cols, UInt64(planes))),
                operation(.reduction, mul(batch, cols)),
                operation(
                    .transcendental, fusedTwoInput ? mul(batch, cols) : 0),
            ],
            synchronization: .simdgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func signedEmbedding(
        pipeline: String,
        constants: [Int: UInt64],
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        guard let hidden = constants[4] else {
            return incomplete("signed.embedding_gather", "missing hidden size")
        }
        let ternary = pipeline.contains("ternary")
        let batch = UInt64(max(geometry.grid[1], 1))
        return KernelDescription(
            group: "signed.embedding_gather",
            roles: [
                0: ResourceRole(
                    access: .read,
                    logicalBytes: mul(batch, hidden / (ternary ? 4 : 8))),
                1: ResourceRole(access: .read, logicalBytes: mul(batch, hidden / 128, 2)),
                2: ResourceRole(access: .read, logicalBytes: mul(batch, 4)),
                3: ResourceRole(access: .write, logicalBytes: mul(batch, hidden, 2)),
            ],
            operations: [operation(.fp16Arithmetic, mul(batch, hidden))],
            synchronization: .none,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func signedDirectMatvec(
        pipeline: String,
        constants: [Int: UInt64]
    ) -> KernelDescription {
        let addedResidual = pipeline.contains("matvec_add")
        let rowsIndex = addedResidual ? 5 : 4
        let colsIndex = addedResidual ? 6 : 5
        guard let rows = constants[rowsIndex], let cols = constants[colsIndex] else {
            return incomplete("signed.direct_matvec", "missing rows or cols")
        }
        let ternary = pipeline.contains("ternary")
        let groups = cols / 128
        let batch = pipeline.contains("batched") ? (constants[6] ?? 1) : 1
        let weightReplays = pipeline.contains("batched") ? (batch + 3) / 4 : 1
        var roles: [Int: ResourceRole] = [
            0: ResourceRole(
                access: .read,
                logicalBytes: mul(
                    weightReplays, rows, cols / (ternary ? 4 : 8)),
                permitsRepeatedTraffic: weightReplays > 1
            ),
            1: ResourceRole(
                access: .read,
                logicalBytes: mul(weightReplays, rows, groups, 2),
                permitsRepeatedTraffic: weightReplays > 1
            ),
            2: ResourceRole(access: .read, logicalBytes: mul(batch, cols, 2)),
            3: ResourceRole(access: .write, logicalBytes: mul(batch, rows, 2)),
        ]
        if addedResidual {
            roles[4] = ResourceRole(access: .read, logicalBytes: mul(rows, 2))
        }
        return KernelDescription(
            group: pipeline.contains("batched")
                ? "signed.direct_matvec_batched"
                : "signed.direct_matvec",
            roles: roles,
            operations: [
                operation(.fp16Arithmetic, mul(batch, rows, cols, 2)),
                operation(.reduction, mul(batch, rows, cols)),
            ],
            synchronization: .simdgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    /// Resource and arithmetic contract for the MLX-compatible ternary affine
    /// QMV family. Unlike the ordinary signed kernels, buffers 1 and 2 are two
    /// independently bound reads of the same scale region so the precise Metal
    /// expression retains the source affine scale/bias tree.
    private static func signedTernaryAffineMatvec(
        pipeline: String,
        constants: [Int: UInt64],
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        let addedResidual = pipeline.contains("matvec_add")
        let rowsIndex = addedResidual ? 6 : 5
        let colsIndex = addedResidual ? 7 : 6
        guard let rows = constants[rowsIndex], let cols = constants[colsIndex] else {
            return incomplete("signed.ternary_affine_matvec", "missing rows or cols")
        }
        let batch = UInt64(max(geometry.grid[1], 1))
        // The precise QMV grid has one independent threadgroup row per
        // activation row, so every row physically replays the weight stream.
        let weightReplays = batch
        let groups = cols / 128
        var roles: [Int: ResourceRole] = [
            0: ResourceRole(
                access: .read,
                logicalBytes: mul(weightReplays, rows, cols / 4),
                permitsRepeatedTraffic: weightReplays > 1
            ),
            1: ResourceRole(
                access: .read,
                logicalBytes: mul(weightReplays, rows, groups, 2),
                permitsRepeatedTraffic: weightReplays > 1
            ),
            2: ResourceRole(
                access: .read,
                logicalBytes: mul(weightReplays, rows, groups, 2),
                permitsRepeatedTraffic: weightReplays > 1
            ),
            3: ResourceRole(access: .read, logicalBytes: mul(batch, cols, 2)),
            4: ResourceRole(access: .write, logicalBytes: mul(batch, rows, 2)),
        ]
        if addedResidual {
            roles[5] = ResourceRole(
                access: .read, logicalBytes: mul(batch, rows, 2))
        }
        return KernelDescription(
            group: batch > 1 ? "signed.ternary_affine_matvec_batched"
                : "signed.ternary_affine_matvec",
            roles: roles,
            operations: [
                operation(.integerBitwise, mul(batch, rows, cols)),
                operation(.fp32Arithmetic, mul(batch, rows, cols, 2)),
                operation(.reduction, mul(batch, rows, cols)),
            ],
            synchronization: .simdgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func signedTernaryAffineQMM(
        constants: [Int: UInt64]
    ) -> KernelDescription {
        guard let rows = constants[5], let cols = constants[6],
              let batch = constants[7]
        else {
            return incomplete("signed.ternary_affine_qmm", "missing rows, cols, or batch")
        }
        let groups = cols / 128
        let weightReplays = (batch + 31) / 32
        return KernelDescription(
            group: "signed.ternary_affine_qmm",
            roles: [
                0: ResourceRole(
                    access: .read,
                    logicalBytes: mul(weightReplays, rows, cols / 4),
                    permitsRepeatedTraffic: weightReplays > 1
                ),
                1: ResourceRole(
                    access: .read,
                    logicalBytes: mul(weightReplays, rows, groups, 2),
                    permitsRepeatedTraffic: weightReplays > 1
                ),
                2: ResourceRole(
                    access: .read,
                    logicalBytes: mul(weightReplays, rows, groups, 2),
                    permitsRepeatedTraffic: weightReplays > 1
                ),
                3: ResourceRole(access: .read, logicalBytes: mul(batch, cols, 2)),
                4: ResourceRole(access: .write, logicalBytes: mul(batch, rows, 2)),
            ],
            operations: [
                operation(.integerBitwise, mul(weightReplays, rows, cols)),
                operation(.fp16Arithmetic, mul(batch, rows, cols, 2)),
                operation(.reduction, mul(batch, rows, cols)),
            ],
            synchronization: .threadgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func signedTernaryAffineBank4(
        constants: [Int: UInt64],
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        guard let r0 = constants[8], let r1 = constants[9],
              let r2 = constants[10], let r3 = constants[11],
              let cols = constants[12]
        else {
            return incomplete("signed.ternary_affine_bank4", "missing rows or cols")
        }
        let rows = add(add(r0, r1), add(r2, r3))
        let groups = cols / 128
        let batch = UInt64(max(geometry.grid[1], 1))
        return KernelDescription(
            group: "signed.ternary_affine_bank4_matvec",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: mul(rows, cols / 4)),
                1: ResourceRole(access: .read, logicalBytes: mul(rows, groups, 2)),
                2: ResourceRole(access: .read, logicalBytes: mul(rows, groups, 2)),
                3: ResourceRole(access: .read, logicalBytes: mul(batch, cols, 2)),
                4: ResourceRole(access: .write, logicalBytes: mul(batch, r0, 2)),
                5: ResourceRole(access: .write, logicalBytes: mul(batch, r1, 2)),
                6: ResourceRole(access: .write, logicalBytes: mul(batch, r2, 2)),
                7: ResourceRole(access: .write, logicalBytes: mul(batch, r3, 2)),
            ],
            operations: [
                operation(.integerBitwise, mul(batch, rows, cols)),
                operation(.fp32Arithmetic, mul(batch, rows, cols, 2)),
                operation(.reduction, mul(batch, rows, cols)),
            ],
            synchronization: .simdgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func signedTernaryAffineGateUp(
        constants: [Int: UInt64],
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        guard let rows = constants[8], let cols = constants[9] else {
            return incomplete("signed.ternary_affine_gate_up_swiglu", "missing rows or cols")
        }
        let groups = cols / 128
        let batch = UInt64(max(geometry.grid[1], 1))
        let codes = mul(rows, cols / 4)
        let scales = mul(rows, groups, 2)
        return KernelDescription(
            group: "signed.ternary_affine_gate_up_swiglu",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: codes),
                1: ResourceRole(access: .read, logicalBytes: scales),
                2: ResourceRole(access: .read, logicalBytes: scales),
                3: ResourceRole(access: .read, logicalBytes: codes),
                4: ResourceRole(access: .read, logicalBytes: scales),
                5: ResourceRole(access: .read, logicalBytes: scales),
                6: ResourceRole(access: .read, logicalBytes: mul(batch, cols, 2)),
                7: ResourceRole(access: .write, logicalBytes: mul(batch, rows, 2)),
            ],
            operations: [
                operation(.integerBitwise, mul(batch, rows, cols, 2)),
                operation(.fp32Arithmetic, mul(batch, rows, cols, 4)),
                operation(.transcendental, mul(batch, rows)),
                operation(.reduction, mul(batch, rows, cols, 2)),
            ],
            synchronization: .simdgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func signedDirectGateUp(
        pipeline: String,
        constants: [Int: UInt64]
    ) -> KernelDescription {
        guard let rows = constants[6], let cols = constants[7] else {
            return incomplete("signed.direct_gate_up_swiglu", "missing rows or cols")
        }
        let ternary = pipeline.contains("ternary")
        let groups = cols / 128
        let codes = mul(rows, cols / (ternary ? 4 : 8))
        let scales = mul(rows, groups, 2)
        let batch = pipeline.contains("batched") ? (constants[8] ?? 1) : 1
        let weightReplays = pipeline.contains("batched") ? (batch + 3) / 4 : 1
        return KernelDescription(
            group: pipeline.contains("batched")
                ? "signed.direct_gate_up_swiglu_batched"
                : "signed.direct_gate_up_swiglu",
            roles: [
                0: ResourceRole(
                    access: .read,
                    logicalBytes: mul(weightReplays, codes ?? 0),
                    permitsRepeatedTraffic: weightReplays > 1
                ),
                1: ResourceRole(
                    access: .read,
                    logicalBytes: mul(weightReplays, scales ?? 0),
                    permitsRepeatedTraffic: weightReplays > 1
                ),
                2: ResourceRole(
                    access: .read,
                    logicalBytes: mul(weightReplays, codes ?? 0),
                    permitsRepeatedTraffic: weightReplays > 1
                ),
                3: ResourceRole(
                    access: .read,
                    logicalBytes: mul(weightReplays, scales ?? 0),
                    permitsRepeatedTraffic: weightReplays > 1
                ),
                4: ResourceRole(access: .read, logicalBytes: mul(batch, cols, 2)),
                5: ResourceRole(access: .write, logicalBytes: mul(batch, rows, 2)),
            ],
            operations: [
                operation(.fp16Arithmetic, mul(batch, rows, cols, 4)),
                operation(.transcendental, mul(batch, rows)),
                operation(.reduction, mul(batch, rows, cols, 2)),
            ],
            synchronization: .simdgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func gatedRMSActivation(
        pipeline: String,
        constants: [Int: UInt64],
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        guard let dim = constants[5], let planes = planeCount(pipeline) else {
            return incomplete("norm.gated_signed_activation", "missing dimension or plane count")
        }
        let heads = UInt64(max(geometry.grid[0], 1))
        let values = mul(heads, dim)
        return KernelDescription(
            group: "norm.gated_signed_activation",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: mul(values ?? 0, 2)),
                1: ResourceRole(access: .read, logicalBytes: mul(values ?? 0, 2)),
                2: ResourceRole(access: .read, logicalBytes: mul(dim, 2)),
                3: ResourceRole(access: .write, logicalBytes: mul(heads, UInt64(planes), 16)),
                4: ResourceRole(access: .write, logicalBytes: mul(heads, 2)),
            ],
            operations: [
                operation(.fp32Arithmetic, mul(values ?? 0, UInt64(planes + 5))),
                operation(.integerBitwise, mul(values ?? 0, UInt64(planes))),
                operation(.transcendental, values),
                operation(.reduction, mul(values ?? 0, 2)),
            ],
            synchronization: .threadgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func residualNormScale(
        constants: [Int: UInt64]
    ) -> KernelDescription {
        guard let dim = constants[4] else {
            return incomplete("residual.rms_scale", "missing dimension")
        }
        return KernelDescription(
            group: "residual.rms_scale",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: mul(dim, 2)),
                1: ResourceRole(access: .read, logicalBytes: mul(dim, 2)),
                2: ResourceRole(access: .write, logicalBytes: mul(dim, 2)),
                3: ResourceRole(access: .write, logicalBytes: 4),
            ],
            operations: [
                operation(.fp16Arithmetic, dim),
                operation(.fp32Arithmetic, mul(dim, 2)),
                operation(.reduction, dim),
                operation(.transcendental, 1),
            ],
            synchronization: .threadgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func deltaNetRecurrence(
        pipeline: String,
        constants: [Int: UInt64],
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        let prefill = pipeline.contains("_prefill")
        let embeddedHeadDimension = embeddedDimension(
            pipeline, marker: "_d"
        ).map(UInt64.init)
        let specialized = embeddedHeadDimension != nil
        let dimension = embeddedHeadDimension
            ?? constants[7]
            ?? UInt64(max(geometry.grid[1], 1))
        let valueHeads = constants[prefill ? 8 : 9]
            ?? embeddedDimension(pipeline, marker: "_h").map(UInt64.init)
            ?? UInt64(max(geometry.grid[2], 1))
        let qkHeads = constants[prefill ? 9 : 10]
            ?? embeddedDimension(pipeline, marker: "_qk").map(UInt64.init)
            ?? valueHeads
        let batch = prefill
            ? (constants[specialized ? 7 : 10] ?? 1)
            : 1
        let stateBytes = mul(valueHeads, dimension, dimension, 2)
        return KernelDescription(
            group: "state.deltanet_recurrence",
            roles: [
                0: ResourceRole(access: .readWrite, logicalBytes: stateBytes),
                1: ResourceRole(access: .read, logicalBytes: mul(batch, add(mul(qkHeads, 2) ?? 0, valueHeads), dimension, 2)),
                2: ResourceRole(access: .read, logicalBytes: mul(batch, valueHeads, 2)),
                3: ResourceRole(access: .read, logicalBytes: mul(batch, valueHeads, 2)),
                4: ResourceRole(access: .read, logicalBytes: mul(valueHeads, 2)),
                5: ResourceRole(access: .read, logicalBytes: mul(valueHeads, 2)),
                6: ResourceRole(access: .write, logicalBytes: mul(batch, valueHeads, dimension, 2)),
            ],
            operations: [
                operation(.fp32Arithmetic, mul(batch, valueHeads, dimension, dimension, 6)),
                operation(.transcendental, mul(batch, valueHeads, 3)),
                operation(.reduction, mul(batch, valueHeads, dimension, dimension, 2)),
            ],
            synchronization: .threadgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func conv1d(
        pipeline: String,
        constants: [Int: UInt64],
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        let channels = constants[4] ?? UInt64(max(geometry.grid[0], 1))
        let batch = constants[3] ?? 1
        guard let kernelSize = constants[5] else {
            return incomplete("state.conv1d_update_silu", "missing kernel size")
        }
        var roles: [Int: ResourceRole] = [
            0: ResourceRole(access: .readWrite, logicalBytes: mul(channels, kernelSize, 2)),
            1: ResourceRole(access: .readWrite, logicalBytes: mul(batch, channels, 2)),
            2: ResourceRole(
                access: .read,
                logicalBytes: mul(batch, channels, kernelSize, 2),
                permitsRepeatedTraffic: batch > 1
            ),
        ]
        if !pipeline.contains("prefill") {
            roles[3] = ResourceRole(
                access: .write, logicalBytes: mul(channels, 2))
        }
        return KernelDescription(
            group: "state.conv1d_update_silu",
            roles: roles,
            operations: [
                operation(.fp32Arithmetic, mul(batch, channels, kernelSize, 2)),
                operation(.transcendental, mul(batch, channels)),
            ],
            synchronization: .none,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func kvCache(
        constants: [Int: UInt64],
        rope: Bool
    ) -> KernelDescription {
        let base = rope ? 4 : 2
        guard let dim = constants[base + 1], let heads = constants[base + 3] else {
            return incomplete("state.kv_cache_update", "missing head dimension or count")
        }
        let vectorBytes = mul(heads, dim, 2)
        var roles: [Int: ResourceRole] = [
            0: ResourceRole(access: .write, logicalBytes: vectorBytes),
            1: ResourceRole(access: .read, logicalBytes: vectorBytes),
        ]
        if rope {
            let ropeDim = constants[8] ?? dim
            roles[2] = ResourceRole(access: .read, logicalBytes: mul(ropeDim, 2))
            roles[3] = ResourceRole(access: .read, logicalBytes: mul(ropeDim, 2))
        }
        return KernelDescription(
            group: rope ? "state.rope_kv_cache_update" : "state.kv_cache_update",
            roles: roles,
            operations: [operation(.fp32Arithmetic, rope ? mul(vectorBytes ?? 0, 2) : 0)],
            synchronization: .none,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func attention(
        pipeline: String,
        constants: [Int: UInt64],
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        if let dim = embeddedDimension(pipeline, marker: "_d"),
           let queryHeads = embeddedDimension(pipeline, marker: "_h"),
           let kvHeads = embeddedDimension(pipeline, marker: "_kv")
        {
            guard let sequence = constants[3] else {
                return incomplete("attention.decode", "missing sequence length")
            }
            let d = UInt64(dim)
            let qh = UInt64(queryHeads)
            let kvh = UInt64(kvHeads)
            return KernelDescription(
                group: "attention.decode",
                roles: [
                    0: ResourceRole(access: .readWrite, logicalBytes: mul(qh, d, 2)),
                    1: ResourceRole(access: .read, logicalBytes: mul(kvh, sequence, d, 2)),
                    2: ResourceRole(access: .read, logicalBytes: mul(kvh, sequence, d, 2)),
                ],
                operations: [
                    operation(.fp32Arithmetic, mul(qh, sequence, d, 4)),
                    operation(.transcendental, mul(qh, sequence)),
                    operation(.reduction, mul(qh, sequence, 2)),
                ],
                synchronization: .threadgroup,
                materializationBytes: 0,
                unknowns: []
            )
        }
        guard let dim = constants[5], let sequence = constants[7],
              let kvHeads = constants[8] else {
            return incomplete("attention.decode", "missing dimension, sequence length, or KV heads")
        }
        let queryHeads = UInt64(max(geometry.grid[0], 1))
        return KernelDescription(
            group: "attention.decode",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: mul(queryHeads, dim, 2)),
                1: ResourceRole(access: .read, logicalBytes: mul(kvHeads, sequence, dim, 2)),
                2: ResourceRole(access: .read, logicalBytes: mul(kvHeads, sequence, dim, 2)),
                3: ResourceRole(access: .read, logicalBytes: 0),
                4: ResourceRole(access: .write, logicalBytes: mul(queryHeads, dim, 2)),
            ],
            operations: [
                operation(.fp32Arithmetic, mul(queryHeads, sequence, dim, 4)),
                operation(.transcendental, mul(queryHeads, sequence)),
                operation(.reduction, mul(queryHeads, sequence, 2)),
            ],
            synchronization: .threadgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func attentionPrefill(
        constants: [Int: UInt64],
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        guard let dim = constants[4], let batch = constants[5],
              let start = constants[6], let kvHeads = constants[8]
        else {
            return incomplete(
                "attention.prefill",
                "missing dimension, sequence length, start position, or KV heads"
            )
        }
        let queryHeads = UInt64(max(geometry.grid[0], 1))
        let slidingWindow = constants[10] ?? 0
        var activePositions: UInt64 = 0
        for position in 0..<batch {
            let causal = start + position + 1
            let active = slidingWindow > 0 ? min(causal, slidingWindow) : causal
            activePositions = add(activePositions, active)
        }
        let queryValues = mul(batch, queryHeads, dim)
        let cacheValues = mul(queryHeads, activePositions, dim)
        return KernelDescription(
            group: "attention.prefill",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: mul(queryValues ?? 0, 2)),
                1: ResourceRole(
                    access: .read,
                    logicalBytes: mul(cacheValues ?? 0, 2),
                    permitsRepeatedTraffic: true
                ),
                2: ResourceRole(
                    access: .read,
                    logicalBytes: mul(cacheValues ?? 0, 2),
                    permitsRepeatedTraffic: true
                ),
                3: ResourceRole(access: .write, logicalBytes: mul(queryValues ?? 0, 2)),
            ],
            operations: [
                operation(.fp32Arithmetic, mul(queryHeads, activePositions, dim, 4)),
                operation(.transcendental, mul(queryHeads, activePositions)),
                operation(.reduction, mul(queryHeads, activePositions, 2)),
            ],
            synchronization: .threadgroup,
            materializationBytes: 0,
            unknowns: kvHeads > queryHeads
                ? ["KV head count exceeds query head count"] : []
        )
    }

    private static func ropeAndKVPrefill(
        constants: [Int: UInt64],
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        guard let dim = constants[7], let ropeDim = constants[8],
              let queryHeads = constants[9], let kvHeads = constants[10],
              let batch = constants[11]
        else {
            return incomplete(
                "position.rope_kv_prefill",
                "missing dimension, rope dimension, head counts, or sequence length"
            )
        }
        let queryBytes = mul(batch, queryHeads, dim, 2)
        let kvBytes = mul(batch, kvHeads, dim, 2)
        let tableBytes = mul(batch, ropeDim, 2)
        return KernelDescription(
            group: "position.rope_kv_prefill",
            roles: [
                0: ResourceRole(access: .readWrite, logicalBytes: queryBytes),
                1: ResourceRole(access: .readWrite, logicalBytes: kvBytes),
                2: ResourceRole(access: .read, logicalBytes: kvBytes),
                3: ResourceRole(access: .read, logicalBytes: tableBytes),
                4: ResourceRole(access: .read, logicalBytes: tableBytes),
                5: ResourceRole(access: .write, logicalBytes: kvBytes),
                6: ResourceRole(access: .write, logicalBytes: kvBytes),
            ],
            operations: [
                operation(
                    .fp32Arithmetic,
                    mul(batch, add(queryHeads, kvHeads), ropeDim, 4)
                ),
            ],
            synchronization: .none,
            materializationBytes: 0,
            unknowns: UInt64(max(geometry.grid[0], 1)) == batch
                ? [] : ["sequence geometry disagrees with sequence constant"]
        )
    }

    private static func gatedRMSNormBatched(
        constants: [Int: UInt64],
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        guard let heads = constants[4] else {
            return incomplete("norm.gated_batched", "missing head count")
        }
        let batch = UInt64(max(geometry.grid[1], 1))
        let dim: UInt64 = 128
        let values = mul(batch, heads, dim)
        return KernelDescription(
            group: "norm.gated_batched",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: mul(values ?? 0, 2)),
                1: ResourceRole(access: .read, logicalBytes: mul(values ?? 0, 2)),
                2: ResourceRole(
                    access: .read,
                    logicalBytes: mul(batch, heads, dim, 2),
                    permitsRepeatedTraffic: batch * heads > 1
                ),
                3: ResourceRole(access: .write, logicalBytes: mul(values ?? 0, 2)),
            ],
            operations: [
                operation(.fp32Arithmetic, mul(values ?? 0, 5)),
                operation(.transcendental, values),
                operation(.reduction, values),
            ],
            synchronization: .threadgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func gatedRMSNorm(
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        let heads = UInt64(max(geometry.grid[0], 1))
        let dim: UInt64 = 128
        let values = mul(heads, dim)
        return KernelDescription(
            group: "norm.gated",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: mul(values ?? 0, 2)),
                1: ResourceRole(access: .read, logicalBytes: mul(values ?? 0, 2)),
                2: ResourceRole(
                    access: .read,
                    logicalBytes: mul(heads, dim, 2),
                    permitsRepeatedTraffic: heads > 1
                ),
                3: ResourceRole(access: .write, logicalBytes: mul(values ?? 0, 2)),
            ],
            operations: [
                operation(.fp32Arithmetic, mul(values ?? 0, 5)),
                operation(.transcendental, values),
                operation(.reduction, values),
            ],
            synchronization: .threadgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func rmsScaleOnly(
        constants: [Int: UInt64]
    ) -> KernelDescription {
        guard let dim = constants[2] else {
            return incomplete("norm.rms_scale_only", "missing dimension")
        }
        return KernelDescription(
            group: "norm.rms_scale_only",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: mul(dim, 2)),
                1: ResourceRole(access: .write, logicalBytes: 4),
            ],
            operations: [
                operation(.fp32Arithmetic, mul(dim, 2)),
                operation(.reduction, dim),
                operation(.transcendental, 1),
            ],
            synchronization: .threadgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func rmsNorm(
        constants: [Int: UInt64],
        pipeline: String,
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        guard let dim = constants[3] ?? embeddedDimension(pipeline, marker: "_d").map(UInt64.init) else {
            return incomplete("norm.rms", "missing dimension")
        }
        let batch = pipeline.contains("batched")
            ? UInt64(max(geometry.grid[0], 1)) : 1
        let values = mul(batch, dim)
        return KernelDescription(
            group: "norm.rms",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: mul(values ?? 0, 2)),
                1: ResourceRole(
                    access: .read,
                    logicalBytes: mul(batch, dim, 2),
                    permitsRepeatedTraffic: batch > 1
                ),
                2: ResourceRole(access: .write, logicalBytes: mul(values ?? 0, 2)),
            ],
            operations: [
                operation(.fp32Arithmetic, mul(values ?? 0, 4)),
                operation(.reduction, values),
                operation(.transcendental, batch),
            ],
            synchronization: .threadgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func rmsScaleQK(
        constants: [Int: UInt64],
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        guard let dim = constants[1], let heads = constants[4] else {
            return incomplete("norm.rms_scale_qk", "missing dimension or head count")
        }
        let batch = UInt64(max(geometry.grid[1], 1))
        let values = mul(batch, heads, 2, dim)
        return KernelDescription(
            group: "norm.rms_scale_qk",
            roles: [
                0: ResourceRole(access: .readWrite, logicalBytes: mul(values ?? 0, 2)),
            ],
            operations: [
                operation(.fp32Arithmetic, mul(values ?? 0, 4)),
                operation(.reduction, values),
                operation(.transcendental, mul(batch, heads, 2)),
            ],
            synchronization: .threadgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func applyRoPE(
        constants: [Int: UInt64]
    ) -> KernelDescription {
        guard let dim = constants[3], let ropeDim = constants[4],
              let heads = constants[5] else {
            return incomplete("position.rope", "missing dimension, rope dimension, or heads")
        }
        return KernelDescription(
            group: "position.rope",
            roles: [
                0: ResourceRole(access: .readWrite, logicalBytes: mul(heads, dim, 2)),
                1: ResourceRole(access: .read, logicalBytes: mul(ropeDim, 2)),
                2: ResourceRole(access: .read, logicalBytes: mul(ropeDim, 2)),
            ],
            operations: [operation(.fp32Arithmetic, mul(heads, ropeDim, 3))],
            synchronization: .none,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func argmaxPartials(
        constants: [Int: UInt64]
    ) -> KernelDescription {
        guard let vocab = constants[2], let chunk = constants[3] else {
            return incomplete("reduction.argmax_partials", "missing vocabulary or chunk size")
        }
        let partials = (vocab + chunk - 1) / max(chunk, 1)
        return KernelDescription(
            group: "reduction.argmax_partials",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: mul(vocab, 2)),
                1: ResourceRole(access: .write, logicalBytes: mul(partials, 8)),
            ],
            operations: [operation(.reduction, vocab)],
            synchronization: .threadgroup,
            materializationBytes: mul(partials, 8),
            unknowns: []
        )
    }

    private static func argmaxReduce(
        constants: [Int: UInt64]
    ) -> KernelDescription {
        guard let partials = constants[2] else {
            return incomplete("reduction.argmax_final", "missing partial count")
        }
        return KernelDescription(
            group: "reduction.argmax_final",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: mul(partials, 8)),
                1: ResourceRole(access: .write, logicalBytes: 4),
            ],
            operations: [operation(.reduction, partials)],
            synchronization: .threadgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func perHeadRMSNorm(
        pipeline: String,
        constants: [Int: UInt64],
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        let batched = pipeline.contains("batched")
        let dimIndex = batched ? 4 : 3
        guard let dim = constants[dimIndex] else {
            return incomplete("norm.per_head_rms", "missing head dimension")
        }
        let heads = UInt64(max(geometry.grid[0], 1))
        let batch = batched ? UInt64(max(geometry.grid[1], 1)) : 1
        let values = mul(batch, heads, dim)
        return KernelDescription(
            group: "norm.per_head_rms",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: mul(values ?? 0, 2)),
                1: ResourceRole(
                    access: .read,
                    logicalBytes: mul(batch, heads, dim, 2),
                    permitsRepeatedTraffic: batch * heads > 1
                ),
                2: ResourceRole(access: .write, logicalBytes: mul(values ?? 0, 2)),
            ],
            operations: [
                operation(.fp32Arithmetic, mul(values ?? 0, 4)),
                operation(.reduction, values),
                operation(.transcendental, mul(batch, heads)),
            ],
            synchronization: .threadgroup,
            materializationBytes: 0,
            unknowns: []
        )
    }

    private static func gateSplit(
        constants: [Int: UInt64]
    ) -> KernelDescription {
        guard let heads = constants[3], let dim = constants[4] else {
            return incomplete("activation.gate_split", "missing head count or dimension")
        }
        let values = mul(heads, dim)
        return KernelDescription(
            group: "activation.gate_split",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: mul(values ?? 0, 4)),
                1: ResourceRole(access: .write, logicalBytes: mul(values ?? 0, 2)),
                2: ResourceRole(access: .write, logicalBytes: mul(values ?? 0, 2)),
            ],
            operations: [],
            synchronization: .none,
            materializationBytes: mul(values ?? 0, 4),
            unknowns: []
        )
    }

    private static func elementwiseAdd(
        constants: [Int: UInt64],
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        let count = constants.values.first ?? UInt64(max(geometry.grid[0], 1))
        return KernelDescription(
            group: "activation.elementwise_add",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: mul(count, 2)),
                1: ResourceRole(access: .read, logicalBytes: mul(count, 2)),
                2: ResourceRole(access: .write, logicalBytes: mul(count, 2)),
            ],
            operations: [operation(.fp16Arithmetic, count)],
            synchronization: .none,
            materializationBytes: mul(count, 2),
            unknowns: []
        )
    }

    private static func elementwiseMultiply(
        constants: [Int: UInt64],
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        let count = constants.values.first ?? UInt64(max(geometry.grid[0], 1))
        return KernelDescription(
            group: "activation.elementwise_multiply",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: mul(count, 2)),
                1: ResourceRole(access: .read, logicalBytes: mul(count, 2)),
                2: ResourceRole(access: .write, logicalBytes: mul(count, 2)),
            ],
            operations: [operation(.fp16Arithmetic, count)],
            synchronization: .none,
            materializationBytes: mul(count, 2),
            unknowns: []
        )
    }

    private static func sigmoid(
        constants: [Int: UInt64],
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        let count = constants.values.first ?? UInt64(max(geometry.grid[0], 1))
        return KernelDescription(
            group: "activation.sigmoid",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: mul(count, 2)),
                1: ResourceRole(access: .write, logicalBytes: mul(count, 2)),
            ],
            operations: [
                operation(.fp32Arithmetic, mul(count, 2)),
                operation(.transcendental, count),
            ],
            synchronization: .none,
            materializationBytes: mul(count, 2),
            unknowns: []
        )
    }

    private static func sigmoidMultiply(
        constants: [Int: UInt64],
        geometry: SmeltFrozenIRGeometry
    ) -> KernelDescription {
        let count = constants.values.first ?? UInt64(max(geometry.grid[0], 1))
        return KernelDescription(
            group: "activation.sigmoid_multiply",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: mul(count, 2)),
                1: ResourceRole(access: .read, logicalBytes: mul(count, 2)),
                2: ResourceRole(access: .write, logicalBytes: mul(count, 2)),
            ],
            operations: [
                operation(.fp32Arithmetic, mul(count, 3)),
                operation(.transcendental, count),
            ],
            synchronization: .none,
            materializationBytes: mul(count, 2),
            unknowns: []
        )
    }

    private static func gatedLinearUnit(
        pipeline: String,
        constants: [Int: UInt64]
    ) -> KernelDescription {
        guard let count = constants[3] else {
            return incomplete("activation.gated_linear_unit", "missing element count")
        }
        let geglu = pipeline == "geglu_fused"
        return KernelDescription(
            group: geglu ? "activation.geglu" : "activation.swiglu",
            roles: [
                0: ResourceRole(access: .read, logicalBytes: mul(count, 2)),
                1: ResourceRole(access: .read, logicalBytes: mul(count, 2)),
                2: ResourceRole(access: .write, logicalBytes: mul(count, 2)),
            ],
            operations: [
                operation(.fp32Arithmetic, mul(count, geglu ? 9 : 4)),
                operation(.transcendental, count),
            ],
            synchronization: .none,
            materializationBytes: mul(count, 2),
            unknowns: []
        )
    }

    private static func incomplete(
        _ group: String,
        _ reason: String
    ) -> KernelDescription {
        KernelDescription(
            group: group,
            roles: [:],
            operations: [operation(.unknown, nil, reason: reason)],
            synchronization: .unknown,
            materializationBytes: nil,
            unknowns: [reason]
        )
    }

    private static func operation(
        _ kind: SmeltFrozenIROperationClass,
        _ count: UInt64?,
        reason: String? = nil
    ) -> SmeltFrozenIROperationBill {
        SmeltFrozenIROperationBill(
            operationClass: kind,
            count: count,
            unknownReason: reason
        )
    }

    private static func storageClass(
        slot: SmeltBufferSlot,
        layout: SmeltSlotLayout
    ) -> SmeltFrozenIRStorageClass {
        if slot.index == layout.weightsSlot || slot.category == .weight {
            return .streamingWeight
        }
        if slot.index == layout.tokenIdSlot || slot.index == layout.positionSlot {
            return .runtimeScalar
        }
        switch slot.category {
        case .activation:
            return .hotActivation
        case .state, .dynamic:
            return .persistentState
        case .table:
            return .lookupTable
        case .weight:
            return .streamingWeight
        }
    }

    private static func executes(
        _ record: SmeltDispatchRecord,
        context: SmeltCostModelContext
    ) -> Bool {
        if record.minSeqLen > 0 && context.sequenceLength < Int(record.minSeqLen) {
            return false
        }
        for index in 0..<Int(record.constantCount) {
            let constant = getConstant(record, index: index)
            switch constant.kind {
            case SmeltConstantRecord.kindSeqLenModLiteralSkipIfZero:
                let divisor = max(Int(constant.value), 1)
                if context.sequenceLength.isMultiple(of: divisor) { return false }
            case SmeltConstantRecord.kindPositionPlus1LessThanLiteralSkipIfFalse:
                if context.position + 1 >= Int(constant.value) { return false }
            case SmeltConstantRecord.kindPositionPlus1GreaterEqualLiteralSkipIfFalse:
                if context.position + 1 < Int(constant.value) { return false }
            case SmeltConstantRecord.kindSeqLenLessThanLiteralSkipIfFalse:
                if context.sequenceLength >= Int(constant.value) { return false }
            default:
                break
            }
        }
        return true
    }

    private static func geometry(
        for record: SmeltDispatchRecord,
        context: SmeltCostModelContext
    ) -> SmeltFrozenIRGeometry {
        let grid = [
            resolveGrid(record.gridW, kind: record.gridWKind, context: context),
            resolveGrid(record.gridH, kind: record.gridHKind, context: context),
            resolveGrid(record.gridD, kind: record.gridDKind, context: context),
        ]
        let threadgroup = [
            max(Int(record.tgW), 1),
            max(Int(record.tgH), 1),
            max(Int(record.tgD), 1),
        ]
        let style: SmeltCostModelDispatchStyle = record.dispatchStyle
            == SmeltDispatchRecord.styleThreads ? .threads : .threadgroups
        let count: Int
        switch style {
        case .threadgroups:
            count = grid.reduce(1, *)
        case .threads:
            count = zip(grid, threadgroup).reduce(1) { product, pair in
                product * ((pair.0 + pair.1 - 1) / pair.1)
            }
        }
        return SmeltFrozenIRGeometry(
            style: style,
            grid: grid,
            threadgroup: threadgroup,
            threadgroupCount: count
        )
    }

    private static func resolveGrid(
        _ literal: UInt32,
        kind: UInt8,
        context: SmeltCostModelContext
    ) -> Int {
        let sequence = max(context.sequenceLength, 1)
        switch kind {
        case SmeltDispatchRecord.gridSeqLen:
            return sequence
        case SmeltDispatchRecord.gridSeqLenMulLiteral:
            return sequence * max(Int(literal), 1)
        case SmeltDispatchRecord.gridSeqLenCeilDivLiteral:
            let divisor = max(Int(literal & 0x7FFF_FFFF), 1)
            return (literal & 0x8000_0000) != 0
                ? sequence / divisor
                : (sequence + divisor - 1) / divisor
        default:
            return max(Int(literal), 1)
        }
    }

    private func resolvedConstants(
        _ record: SmeltDispatchRecord,
        context: SmeltCostModelContext
    ) -> [Int: UInt64] {
        var result: [Int: UInt64] = [:]
        for index in 0..<Int(record.constantCount) {
            let constant = getConstant(record, index: index)
            let value: UInt64
            switch constant.kind {
            case SmeltConstantRecord.kindPosition:
                value = UInt64(context.position)
            case SmeltConstantRecord.kindPositionPlus1,
                 SmeltConstantRecord.kindPositionPlus1LessThanLiteralSkipIfFalse,
                 SmeltConstantRecord.kindPositionPlus1GreaterEqualLiteralSkipIfFalse:
                value = UInt64(context.position + 1)
            case SmeltConstantRecord.kindSeqLen:
                value = UInt64(context.sequenceLength)
            case SmeltConstantRecord.kindSeqLenLessThanLiteralSkipIfFalse:
                value = UInt64(context.sequenceLength)
            case SmeltConstantRecord.kindStartPos:
                value = UInt64(context.position)
            case SmeltConstantRecord.kindStartPosPlusLiteral:
                value = UInt64(context.position) + UInt64(constant.value)
            case SmeltConstantRecord.kindSeqLenMulLiteral:
                value = UInt64(context.sequenceLength) * UInt64(constant.value)
            case SmeltConstantRecord.kindSeqLenModLiteral,
                 SmeltConstantRecord.kindSeqLenModLiteralSkipIfZero:
                value = UInt64(context.sequenceLength % max(Int(constant.value), 1))
            case SmeltConstantRecord.kindCacheSeqCapacity:
                value = UInt64(manifest.config.staticContextCapacity)
            default:
                value = UInt64(constant.value)
            }
            if constant.bindingIndex != UInt8.max {
                result[Int(constant.bindingIndex)] = value
            }
        }
        return result
    }

    private static func resolvedOffset(
        _ binding: SmeltBufferRecord,
        context: SmeltCostModelContext
    ) -> UInt64 {
        switch binding.offsetKind {
        case 1:
            return UInt64(context.position) * binding.offset
        case 2:
            let stride = binding.offset & 0xFFFF_FFFF
            let base = binding.offset >> 32
            return UInt64(context.position) * stride + base
        case 3:
            return UInt64(max(context.sequenceLength - 1, 0)) * binding.offset
        case 4:
            let stride = binding.offset & 0xFFFF_FFFF
            let divisor = max(binding.offset >> 32, 1)
            return UInt64(context.sequenceLength) / divisor * stride
        default:
            return binding.offset
        }
    }

    private static func planeCount(_ pipeline: String) -> Int? {
        (2...8).first { pipeline.contains("_i\($0)_") }
    }

    private static func embeddedDimension(
        _ pipeline: String,
        marker: String
    ) -> Int? {
        let key = marker.trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        for component in pipeline.split(separator: "_") where component.hasPrefix(key) {
            let suffix = component.dropFirst(key.count)
            guard !suffix.isEmpty, suffix.allSatisfy(\.isNumber) else { continue }
            return Int(suffix)
        }
        return nil
    }

    private static func mul(_ values: UInt64...) -> UInt64? {
        var result: UInt64 = 1
        for value in values {
            let next = result.multipliedReportingOverflow(by: value)
            if next.overflow { return nil }
            result = next.partialValue
        }
        return result
    }

    private static func add(_ lhs: UInt64, _ rhs: UInt64) -> UInt64 {
        let result = lhs.addingReportingOverflow(rhs)
        return result.overflow ? UInt64.max : result.partialValue
    }

    private static func signedDelta(_ candidate: UInt64, _ baseline: UInt64) -> Int64? {
        if candidate >= baseline {
            let delta = candidate - baseline
            return delta <= UInt64(Int64.max) ? Int64(delta) : nil
        }
        let delta = baseline - candidate
        return delta <= UInt64(Int64.max) ? -Int64(delta) : nil
    }

    private static func optionalSignedDelta(
        _ candidate: UInt64?,
        _ baseline: UInt64?
    ) -> Int64? {
        guard let candidate, let baseline else { return nil }
        return signedDelta(candidate, baseline)
    }

    private static func optionalDelta(
        _ candidate: Double?,
        _ baseline: Double?
    ) -> Double? {
        guard let candidate, let baseline else { return nil }
        return candidate - baseline
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[middle - 1] + sorted[middle]) / 2
        }
        return sorted[middle]
    }

    private static func percentile95(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let index = min(
            max(Int(ceil(Double(sorted.count) * 0.95)) - 1, 0),
            sorted.count - 1
        )
        return sorted[index]
    }
}
