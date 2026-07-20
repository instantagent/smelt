import Foundation
import SmeltSchema

public struct SmeltCostedGraphPlan: Sendable {
    public let id: String
    public let ops: [SmeltIROp]

    public init(id: String, ops: [SmeltIROp]) {
        self.id = id
        self.ops = ops
    }
}

public struct SmeltGraphCostModel: Sendable {
    public let calibration: SmeltDeviceCostCalibration
    public let decisionPolicy: SmeltPlanDecisionPolicy

    public init(
        calibration: SmeltDeviceCostCalibration,
        decisionPolicy: SmeltPlanDecisionPolicy = SmeltPlanDecisionPolicy()
    ) {
        self.calibration = calibration
        self.decisionPolicy = decisionPolicy
    }

    /// Prices a complete graph plan. Missing samples produce an explicitly
    /// incomplete estimate; there is no heuristic fallback that can be
    /// mistaken for evidence.
    public func estimate(
        _ plan: SmeltCostedGraphPlan,
        context: SmeltCostModelContext
    ) -> SmeltPlanCostEstimate {
        var lookup: [SmeltDispatchCostKey: SmeltDispatchCostSample] = [:]
        for sample in calibration.dispatchSamples {
            lookup[sample.key] = sample
        }

        var recordCount = 0
        var dispatchCount = 0
        var swapCount = 0
        var totalThreadgroups = 0
        var singleThreadgroupDispatches = 0
        var maxThreadgroupsPerDispatch = 0
        var pipelines: Set<String> = []
        var calibratedDispatches = 0
        var missing: Set<String> = []
        var medianGPUUs = 0.0
        var p95GPUUs = 0.0

        for op in plan.ops {
            switch op {
            case .traceMarker:
                continue
            case .swap:
                recordCount += 1
                swapCount += 1
            case .dispatch(let dispatch):
                guard let key = Self.dispatchCostKey(for: dispatch, context: context) else {
                    continue
                }
                recordCount += 1
                dispatchCount += 1
                pipelines.insert(key.pipeline)
                let groups = Self.threadgroupCount(for: key)
                totalThreadgroups += groups
                if groups == 1 {
                    singleThreadgroupDispatches += 1
                }
                maxThreadgroupsPerDispatch = max(maxThreadgroupsPerDispatch, groups)

                if let sample = lookup[key] {
                    calibratedDispatches += 1
                    medianGPUUs += sample.medianGPUUs
                    p95GPUUs += sample.p95GPUUs
                } else {
                    missing.insert(key.stableID)
                }
            }
        }

        let missingKeys = missing.sorted()
        let complete = missingKeys.isEmpty && dispatchCount > 0
        return SmeltPlanCostEstimate(
            planID: plan.id,
            structure: SmeltPlanStructuralCost(
                recordCount: recordCount,
                dispatchCount: dispatchCount,
                swapCount: swapCount,
                totalThreadgroups: totalThreadgroups,
                singleThreadgroupDispatches: singleThreadgroupDispatches,
                maxThreadgroupsPerDispatch: maxThreadgroupsPerDispatch,
                distinctPipelines: pipelines.count
            ),
            calibratedDispatches: calibratedDispatches,
            missingCalibrationKeys: missingKeys,
            predictedMedianGPUUs: complete ? medianGPUUs : nil,
            predictedP95GPUUs: complete ? p95GPUUs : nil,
            predictedHostUs: Double(recordCount) * calibration.hostRecordUs
        )
    }

    /// Compare complete frozen plans before benchmarking. Exact calibration
    /// p95 tails form a conservative, additive error band; a delta inside the
    /// band stays too close to call rather than becoming a guessed win.
    public func predict(
        baseline: SmeltCostedGraphPlan,
        candidate: SmeltCostedGraphPlan,
        context: SmeltCostModelContext
    ) -> SmeltPlanCostPrediction {
        let baselineEstimate = estimate(baseline, context: context)
        let candidateEstimate = estimate(candidate, context: context)
        var missingSet = Set(
            baselineEstimate.missingCalibrationKeys
                + candidateEstimate.missingCalibrationKeys
        )
        if let calibrationContext = calibration.context,
           calibrationContext != context
        {
            missingSet.insert(
                "calibration-context:\(calibrationContext.mode.rawValue):"
                    + "s\(calibrationContext.sequenceLength):"
                    + "p\(calibrationContext.position)"
            )
        }
        let missing = Array(missingSet).sorted()
        guard missing.isEmpty,
              let baselineMedian = baselineEstimate.predictedMedianTotalUs,
              let candidateMedian = candidateEstimate.predictedMedianTotalUs,
              let baselineGPU = baselineEstimate.predictedMedianGPUUs,
              let candidateGPU = candidateEstimate.predictedMedianGPUUs,
              let baselineP95 = baselineEstimate.predictedP95GPUUs,
              let candidateP95 = candidateEstimate.predictedP95GPUUs
        else {
            return SmeltPlanCostPrediction(
                baseline: baselineEstimate,
                candidate: candidateEstimate,
                predictedMedianTotalDeltaUs: nil,
                errorBandUs: nil,
                verdict: .unknown,
                missingCalibrationKeys: missing
            )
        }

        let delta = candidateMedian - baselineMedian
        var errorBand = max(baselineP95 - baselineGPU, 0)
            + max(candidateP95 - candidateGPU, 0)
        if calibration.context == context,
           let wholePlanMedian = calibration.wholePlanMedianGPUUs,
           let wholePlanP95 = calibration.wholePlanP95GPUUs
        {
            errorBand += max(wholePlanP95 - wholePlanMedian, 0)
        }
        let verdict: SmeltPlanCostPredictionVerdict
        if abs(delta) <= errorBand {
            verdict = .tooCloseToCall
        } else if delta < 0 {
            verdict = .candidateFaster
        } else {
            verdict = .baselineFaster
        }
        return SmeltPlanCostPrediction(
            baseline: baselineEstimate,
            candidate: candidateEstimate,
            predictedMedianTotalDeltaUs: delta,
            errorBandUs: errorBand,
            verdict: verdict
        )
    }

    /// Final selection uses paired whole-plan timings, not the sum of isolated
    /// kernel samples. The baseline remains selected for every ambiguous or
    /// invalid result.
    public func decide(
        measurement: SmeltPairedPlanMeasurement,
        expectedContext: SmeltCostModelContext? = nil
    ) -> SmeltPlanCostDecision {
        SmeltPlanCostDecider(
            provenanceKey: calibration.provenanceKey,
            policy: decisionPolicy
        ).decide(
            measurement: measurement,
            expectedContext: expectedContext
        )
    }

    public static func dispatchCostKey(
        for dispatch: SmeltDispatch,
        context: SmeltCostModelContext
    ) -> SmeltDispatchCostKey? {
        if let minimum = dispatch.minSeqLen,
           context.sequenceLength < minimum
        {
            return nil
        }
        if let maximum = dispatch.maxSeqLenExclusive,
           context.sequenceLength >= maximum
        {
            return nil
        }
        let positionPlusOne = context.position + 1
        if let minimum = dispatch.minPositionPlus1,
           positionPlusOne < minimum
        {
            return nil
        }
        if let maximum = dispatch.maxPositionPlus1Exclusive,
           positionPlusOne >= maximum
        {
            return nil
        }

        let pipeline = dispatch.pipelineNameOverride
            ?? SmeltKernelCatalog.signature(for: dispatch.pipeline).metalFunctionName
        let constants = dispatch.constants
            .sorted { $0.bindingIndex < $1.bindingIndex }
            .map {
                "\($0.bindingIndex):\($0.type.typeName):\($0.expression)"
            }

        switch dispatch.dispatch {
        case let .threads(width, height, depth, tgWidth, tgHeight, tgDepth):
            return SmeltDispatchCostKey(
                pipeline: pipeline,
                style: .threads,
                gridWidth: resolve(
                    dispatch.dynamicGridW,
                    fallback: width,
                    sequenceLength: context.sequenceLength
                ),
                gridHeight: resolve(
                    dispatch.dynamicGridH,
                    fallback: height,
                    sequenceLength: context.sequenceLength
                ),
                gridDepth: resolve(
                    dispatch.dynamicGridD,
                    fallback: depth,
                    sequenceLength: context.sequenceLength
                ),
                threadgroupWidth: tgWidth,
                threadgroupHeight: tgHeight,
                threadgroupDepth: tgDepth,
                functionConstants: constants,
                context: context
            )
        case let .threadgroups(width, height, depth, tgWidth, tgHeight, tgDepth):
            return SmeltDispatchCostKey(
                pipeline: pipeline,
                style: .threadgroups,
                gridWidth: resolve(
                    dispatch.dynamicGridW,
                    fallback: width,
                    sequenceLength: context.sequenceLength
                ),
                gridHeight: resolve(
                    dispatch.dynamicGridH,
                    fallback: height,
                    sequenceLength: context.sequenceLength
                ),
                gridDepth: resolve(
                    dispatch.dynamicGridD,
                    fallback: depth,
                    sequenceLength: context.sequenceLength
                ),
                threadgroupWidth: tgWidth,
                threadgroupHeight: tgHeight,
                threadgroupDepth: tgDepth,
                functionConstants: constants,
                context: context
            )
        }
    }

    private static func resolve(
        _ dynamic: SmeltDynamicGridDimension?,
        fallback: Int,
        sequenceLength: Int
    ) -> Int {
        guard let dynamic else { return max(fallback, 1) }
        switch dynamic {
        case .seqLen:
            return sequenceLength
        case .seqLenMul(let factor):
            return sequenceLength * factor
        case .seqLenCeilDiv(let divisor):
            let safeDivisor = max(divisor, 1)
            return (sequenceLength + safeDivisor - 1) / safeDivisor
        case .seqLenFloorDiv(let divisor):
            return sequenceLength / max(divisor, 1)
        }
    }

    private static func threadgroupCount(for key: SmeltDispatchCostKey) -> Int {
        switch key.style {
        case .threadgroups:
            return key.gridWidth * key.gridHeight * key.gridDepth
        case .threads:
            return ceilDiv(key.gridWidth, key.threadgroupWidth)
                * ceilDiv(key.gridHeight, key.threadgroupHeight)
                * ceilDiv(key.gridDepth, key.threadgroupDepth)
        }
    }

    private static func ceilDiv(_ value: Int, _ divisor: Int) -> Int {
        (value + max(divisor, 1) - 1) / max(divisor, 1)
    }

}
