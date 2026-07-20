import Foundation

/// Persistent inputs and outputs for device-calibrated graph-plan costing.
/// These live in SmeltSchema so compiler, runtime probes, and package reports
/// exchange one format without depending on one another's implementation.
public enum SmeltCostModelMode: String, Codable, Sendable {
    case decode
    case prefill
}

public struct SmeltCostModelContext: Codable, Hashable, Sendable {
    public let mode: SmeltCostModelMode
    public let sequenceLength: Int
    public let position: Int

    public init(
        mode: SmeltCostModelMode,
        sequenceLength: Int,
        position: Int = 0
    ) {
        self.mode = mode
        self.sequenceLength = max(sequenceLength, 1)
        self.position = max(position, 0)
    }
}

public enum SmeltCostModelDispatchStyle: String, Codable, Sendable {
    case threads
    case threadgroups
}

public struct SmeltDispatchCostKey: Codable, Hashable, Sendable {
    public let pipeline: String
    public let style: SmeltCostModelDispatchStyle
    public let gridWidth: Int
    public let gridHeight: Int
    public let gridDepth: Int
    public let threadgroupWidth: Int
    public let threadgroupHeight: Int
    public let threadgroupDepth: Int
    public let functionConstants: [String]
    public let context: SmeltCostModelContext

    public init(
        pipeline: String,
        style: SmeltCostModelDispatchStyle,
        gridWidth: Int,
        gridHeight: Int,
        gridDepth: Int,
        threadgroupWidth: Int,
        threadgroupHeight: Int,
        threadgroupDepth: Int,
        functionConstants: [String] = [],
        context: SmeltCostModelContext
    ) {
        self.pipeline = pipeline
        self.style = style
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.gridDepth = gridDepth
        self.threadgroupWidth = threadgroupWidth
        self.threadgroupHeight = threadgroupHeight
        self.threadgroupDepth = threadgroupDepth
        self.functionConstants = functionConstants
        self.context = context
    }

    public var stableID: String {
        let constants = functionConstants.joined(separator: ",")
        return "\(context.mode.rawValue):s\(context.sequenceLength):p\(context.position):"
            + "\(pipeline):\(style.rawValue):"
            + "g\(gridWidth)x\(gridHeight)x\(gridDepth):"
            + "tg\(threadgroupWidth)x\(threadgroupHeight)x\(threadgroupDepth):"
            + constants
    }
}

public struct SmeltDispatchCostSample: Codable, Equatable, Sendable {
    public let key: SmeltDispatchCostKey
    public let medianGPUUs: Double
    public let p95GPUUs: Double
    public let sampleCount: Int

    public init(
        key: SmeltDispatchCostKey,
        medianGPUUs: Double,
        p95GPUUs: Double,
        sampleCount: Int
    ) {
        self.key = key
        self.medianGPUUs = medianGPUUs
        self.p95GPUUs = p95GPUUs
        self.sampleCount = sampleCount
    }
}

public struct SmeltDeviceCostCalibration: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let provenanceKey: String
    public let deviceName: String?
    public let measurementMethod: String?
    public let context: SmeltCostModelContext?
    public let buildFingerprint: String?
    /// Package-relative frozen dispatch table measured by this calibration.
    /// Older decode-only calibrations omit it and are interpreted as
    /// `dispatches.bin` by the consumer.
    public let dispatchTableName: String?
    public let dispatchesSHA256: String?
    public let metallibSHA256: String?
    public let weightsSHA256: String?
    public let hostRecordUs: Double
    public let wholePlanMedianGPUUs: Double?
    public let wholePlanP95GPUUs: Double?
    /// Instrumented command-buffer and summed-span timing before proportions
    /// are reconciled to the clean route. Opaque GPU intervals can overlap even
    /// when whole-plan timing barely changes; retaining both makes that
    /// non-additivity auditable instead of hiding it in a fitted factor.
    public let instrumentedWholePlanMedianGPUUs: Double?
    public let instrumentedSpanMedianGPUUs: Double?
    public let dispatchSamples: [SmeltDispatchCostSample]

    public init(
        schemaVersion: Int = 1,
        provenanceKey: String,
        deviceName: String? = nil,
        measurementMethod: String? = nil,
        context: SmeltCostModelContext? = nil,
        buildFingerprint: String? = nil,
        dispatchTableName: String? = nil,
        dispatchesSHA256: String? = nil,
        metallibSHA256: String? = nil,
        weightsSHA256: String? = nil,
        hostRecordUs: Double = 0,
        wholePlanMedianGPUUs: Double? = nil,
        wholePlanP95GPUUs: Double? = nil,
        instrumentedWholePlanMedianGPUUs: Double? = nil,
        instrumentedSpanMedianGPUUs: Double? = nil,
        dispatchSamples: [SmeltDispatchCostSample]
    ) {
        self.schemaVersion = schemaVersion
        self.provenanceKey = provenanceKey
        self.deviceName = deviceName
        self.measurementMethod = measurementMethod
        self.context = context
        self.buildFingerprint = buildFingerprint
        self.dispatchTableName = dispatchTableName
        self.dispatchesSHA256 = dispatchesSHA256
        self.metallibSHA256 = metallibSHA256
        self.weightsSHA256 = weightsSHA256
        self.hostRecordUs = max(hostRecordUs, 0)
        self.wholePlanMedianGPUUs = wholePlanMedianGPUUs
        self.wholePlanP95GPUUs = wholePlanP95GPUUs
        self.instrumentedWholePlanMedianGPUUs = instrumentedWholePlanMedianGPUUs
        self.instrumentedSpanMedianGPUUs = instrumentedSpanMedianGPUUs
        self.dispatchSamples = dispatchSamples
    }
}

public struct SmeltPlanStructuralCost: Codable, Equatable, Sendable {
    public let recordCount: Int
    public let dispatchCount: Int
    public let swapCount: Int
    public let totalThreadgroups: Int
    public let singleThreadgroupDispatches: Int
    public let maxThreadgroupsPerDispatch: Int
    public let distinctPipelines: Int

    public init(
        recordCount: Int,
        dispatchCount: Int,
        swapCount: Int,
        totalThreadgroups: Int,
        singleThreadgroupDispatches: Int,
        maxThreadgroupsPerDispatch: Int,
        distinctPipelines: Int
    ) {
        self.recordCount = recordCount
        self.dispatchCount = dispatchCount
        self.swapCount = swapCount
        self.totalThreadgroups = totalThreadgroups
        self.singleThreadgroupDispatches = singleThreadgroupDispatches
        self.maxThreadgroupsPerDispatch = maxThreadgroupsPerDispatch
        self.distinctPipelines = distinctPipelines
    }
}

public struct SmeltPlanCostEstimate: Codable, Equatable, Sendable {
    public let planID: String
    public let structure: SmeltPlanStructuralCost
    public let calibratedDispatches: Int
    public let missingCalibrationKeys: [String]
    public let predictedMedianGPUUs: Double?
    public let predictedP95GPUUs: Double?
    public let predictedHostUs: Double

    public init(
        planID: String,
        structure: SmeltPlanStructuralCost,
        calibratedDispatches: Int,
        missingCalibrationKeys: [String],
        predictedMedianGPUUs: Double?,
        predictedP95GPUUs: Double?,
        predictedHostUs: Double
    ) {
        self.planID = planID
        self.structure = structure
        self.calibratedDispatches = calibratedDispatches
        self.missingCalibrationKeys = missingCalibrationKeys
        self.predictedMedianGPUUs = predictedMedianGPUUs
        self.predictedP95GPUUs = predictedP95GPUUs
        self.predictedHostUs = predictedHostUs
    }

    public var hasCompleteCalibration: Bool {
        missingCalibrationKeys.isEmpty && predictedMedianGPUUs != nil
    }

    public var predictedMedianTotalUs: Double? {
        predictedMedianGPUUs.map { $0 + predictedHostUs }
    }
}

public enum SmeltPlanCostPredictionVerdict: String, Codable, Equatable, Sendable {
    case baselineFaster
    case candidateFaster
    case tooCloseToCall
    case unknown
}

/// Pre-measurement comparison of two complete plans. The prediction band is
/// derived from exact-key calibration tails; it never substitutes for paired
/// whole-plan admission evidence.
public struct SmeltPlanCostPrediction: Codable, Equatable, Sendable {
    public let baseline: SmeltPlanCostEstimate
    public let candidate: SmeltPlanCostEstimate
    /// Candidate minus baseline, including host-record cost.
    public let predictedMedianTotalDeltaUs: Double?
    public let errorBandUs: Double?
    public let verdict: SmeltPlanCostPredictionVerdict
    public let missingCalibrationKeys: [String]

    public init(
        baseline: SmeltPlanCostEstimate,
        candidate: SmeltPlanCostEstimate,
        predictedMedianTotalDeltaUs: Double?,
        errorBandUs: Double?,
        verdict: SmeltPlanCostPredictionVerdict,
        missingCalibrationKeys: [String] = []
    ) {
        self.baseline = baseline
        self.candidate = candidate
        self.predictedMedianTotalDeltaUs = predictedMedianTotalDeltaUs
        self.errorBandUs = errorBandUs
        self.verdict = verdict
        self.missingCalibrationKeys = missingCalibrationKeys.sorted()
    }
}

public enum SmeltPlanSampleOrder: String, Codable, Hashable, Sendable {
    case baselineFirst
    case candidateFirst
}

public struct SmeltPairedPlanSample: Codable, Equatable, Sendable {
    public let order: SmeltPlanSampleOrder
    public let baselineGPUUs: Double
    public let candidateGPUUs: Double
    public let baselineWallUs: Double?
    public let candidateWallUs: Double?

    public init(
        order: SmeltPlanSampleOrder,
        baselineGPUUs: Double,
        candidateGPUUs: Double,
        baselineWallUs: Double? = nil,
        candidateWallUs: Double? = nil
    ) {
        self.order = order
        self.baselineGPUUs = baselineGPUUs
        self.candidateGPUUs = candidateGPUUs
        self.baselineWallUs = baselineWallUs
        self.candidateWallUs = candidateWallUs
    }
}

public struct SmeltPlanParityEvidence: Codable, Equatable, Sendable {
    public let checkedSteps: Int
    public let valuesPerStep: Int
    public let firstDivergenceStep: Int?
    public let firstDivergenceValue: Int?
    public let maximumAbsoluteDifference: Double

    public init(
        checkedSteps: Int,
        valuesPerStep: Int,
        firstDivergenceStep: Int? = nil,
        firstDivergenceValue: Int? = nil,
        maximumAbsoluteDifference: Double = 0
    ) {
        self.checkedSteps = checkedSteps
        self.valuesPerStep = valuesPerStep
        self.firstDivergenceStep = firstDivergenceStep
        self.firstDivergenceValue = firstDivergenceValue
        self.maximumAbsoluteDifference = maximumAbsoluteDifference
    }
}

public struct SmeltPairedPlanMeasurement: Codable, Equatable, Sendable {
    public let provenanceKey: String
    public let context: SmeltCostModelContext
    public let baselinePlanID: String
    public let candidatePlanID: String
    public let exactOutputMatch: Bool
    public let samples: [SmeltPairedPlanSample]
    public let baselineTableSHA256: String?
    public let candidateTableSHA256: String?
    public let baselineStructure: SmeltPlanStructuralCost?
    public let candidateStructure: SmeltPlanStructuralCost?
    public let parity: SmeltPlanParityEvidence?

    public init(
        provenanceKey: String,
        context: SmeltCostModelContext,
        baselinePlanID: String,
        candidatePlanID: String,
        exactOutputMatch: Bool,
        samples: [SmeltPairedPlanSample],
        baselineTableSHA256: String? = nil,
        candidateTableSHA256: String? = nil,
        parity: SmeltPlanParityEvidence? = nil
    ) {
        self.init(
            provenanceKey: provenanceKey,
            context: context,
            baselinePlanID: baselinePlanID,
            candidatePlanID: candidatePlanID,
            exactOutputMatch: exactOutputMatch,
            samples: samples,
            baselineTableSHA256: baselineTableSHA256,
            candidateTableSHA256: candidateTableSHA256,
            baselineStructure: nil,
            candidateStructure: nil,
            parity: parity
        )
    }

    public init(
        provenanceKey: String,
        context: SmeltCostModelContext,
        baselinePlanID: String,
        candidatePlanID: String,
        exactOutputMatch: Bool,
        samples: [SmeltPairedPlanSample],
        baselineTableSHA256: String? = nil,
        candidateTableSHA256: String? = nil,
        baselineStructure: SmeltPlanStructuralCost?,
        candidateStructure: SmeltPlanStructuralCost?,
        parity: SmeltPlanParityEvidence? = nil
    ) {
        self.provenanceKey = provenanceKey
        self.context = context
        self.baselinePlanID = baselinePlanID
        self.candidatePlanID = candidatePlanID
        self.exactOutputMatch = exactOutputMatch
        self.samples = samples
        self.baselineTableSHA256 = baselineTableSHA256
        self.candidateTableSHA256 = candidateTableSHA256
        self.baselineStructure = baselineStructure
        self.candidateStructure = candidateStructure
        self.parity = parity
    }
}

public struct SmeltPlanDecisionPolicy: Codable, Equatable, Sendable {
    public let minimumPairs: Int
    public let minimumWinFraction: Double
    public let minimumRelativeImprovement: Double
    public let minimumAbsoluteImprovementUs: Double
    public let maximumWallRegressionFraction: Double
    public let maximumWallRegressionUs: Double
    public let maximumPlanRelativeMAD: Double
    public let minimumDeltaSignalToNoise: Double
    public let requireBothOrders: Bool

    public init(
        minimumPairs: Int = 9,
        minimumWinFraction: Double = 0.8,
        minimumRelativeImprovement: Double = 0.005,
        minimumAbsoluteImprovementUs: Double = 10,
        maximumWallRegressionFraction: Double = 0.005,
        maximumWallRegressionUs: Double = 50,
        maximumPlanRelativeMAD: Double = 0.03,
        minimumDeltaSignalToNoise: Double = 3,
        requireBothOrders: Bool = true
    ) {
        self.minimumPairs = max(minimumPairs, 1)
        self.minimumWinFraction = min(max(minimumWinFraction, 0.5), 1)
        self.minimumRelativeImprovement = max(minimumRelativeImprovement, 0)
        self.minimumAbsoluteImprovementUs = max(minimumAbsoluteImprovementUs, 0)
        self.maximumWallRegressionFraction = max(maximumWallRegressionFraction, 0)
        self.maximumWallRegressionUs = max(maximumWallRegressionUs, 0)
        self.maximumPlanRelativeMAD = max(maximumPlanRelativeMAD, 0)
        self.minimumDeltaSignalToNoise = max(minimumDeltaSignalToNoise, 0)
        self.requireBothOrders = requireBothOrders
    }
}

public enum SmeltPlanDecisionReason: String, Codable, Sendable {
    case candidateFaster
    case baselineFaster
    case exactnessFailed
    case insufficientPairs
    case orderBalanceMissing
    case provenanceMismatch
    case contextMismatch
    case identicalPlans
    case unstableMeasurement
    case wallTimeRegression
    case withinNoise
}

public struct SmeltPlanCostDecision: Codable, Equatable, Sendable {
    public let selectedPlanID: String
    public let reason: SmeltPlanDecisionReason
    /// Positive means the candidate was faster.
    public let medianCandidateImprovementUs: Double
    public let medianCandidateImprovementFraction: Double
    public let candidateWinFraction: Double
    public let pairedDeltaMADUs: Double
    public let baselineRelativeMAD: Double
    public let candidateRelativeMAD: Double

    public init(
        selectedPlanID: String,
        reason: SmeltPlanDecisionReason,
        medianCandidateImprovementUs: Double,
        medianCandidateImprovementFraction: Double,
        candidateWinFraction: Double,
        pairedDeltaMADUs: Double = 0,
        baselineRelativeMAD: Double = 0,
        candidateRelativeMAD: Double = 0
    ) {
        self.selectedPlanID = selectedPlanID
        self.reason = reason
        self.medianCandidateImprovementUs = medianCandidateImprovementUs
        self.medianCandidateImprovementFraction = medianCandidateImprovementFraction
        self.candidateWinFraction = candidateWinFraction
        self.pairedDeltaMADUs = pairedDeltaMADUs
        self.baselineRelativeMAD = baselineRelativeMAD
        self.candidateRelativeMAD = candidateRelativeMAD
    }
}

public struct SmeltPlanComparisonReport: Codable, Equatable, Sendable {
    public let measurement: SmeltPairedPlanMeasurement
    public let decisionPolicy: SmeltPlanDecisionPolicy
    public let decision: SmeltPlanCostDecision

    public init(
        measurement: SmeltPairedPlanMeasurement,
        decisionPolicy: SmeltPlanDecisionPolicy,
        decision: SmeltPlanCostDecision
    ) {
        self.measurement = measurement
        self.decisionPolicy = decisionPolicy
        self.decision = decision
    }
}

/// Deterministic policy for turning exact, interleaved whole-plan evidence
/// into a selection. Ambiguous or invalid evidence always retains baseline.
public struct SmeltPlanCostDecider: Sendable {
    public let provenanceKey: String
    public let policy: SmeltPlanDecisionPolicy

    public init(
        provenanceKey: String,
        policy: SmeltPlanDecisionPolicy = SmeltPlanDecisionPolicy()
    ) {
        self.provenanceKey = provenanceKey
        self.policy = policy
    }

    public func decide(
        measurement: SmeltPairedPlanMeasurement,
        expectedContext: SmeltCostModelContext? = nil
    ) -> SmeltPlanCostDecision {
        let baseline = measurement.baselinePlanID
        let candidate = measurement.candidatePlanID
        guard measurement.provenanceKey == provenanceKey else {
            return conservativeDecision(
                baseline,
                reason: .provenanceMismatch,
                measurement: measurement
            )
        }
        guard expectedContext.map({ $0 == measurement.context }) ?? true else {
            return conservativeDecision(
                baseline,
                reason: .contextMismatch,
                measurement: measurement
            )
        }
        guard measurement.exactOutputMatch else {
            return conservativeDecision(
                baseline,
                reason: .exactnessFailed,
                measurement: measurement
            )
        }
        if let baselineHash = measurement.baselineTableSHA256,
           let candidateHash = measurement.candidateTableSHA256,
           baselineHash == candidateHash
        {
            return conservativeDecision(
                baseline,
                reason: .identicalPlans,
                measurement: measurement
            )
        }
        guard measurement.samples.count >= policy.minimumPairs else {
            return conservativeDecision(
                baseline,
                reason: .insufficientPairs,
                measurement: measurement
            )
        }
        if policy.requireBothOrders {
            let orders = Set(measurement.samples.map(\.order))
            guard orders == Set([.baselineFirst, .candidateFirst]) else {
                return conservativeDecision(
                    baseline,
                    reason: .orderBalanceMissing,
                    measurement: measurement
                )
            }
        }

        let metrics = Self.metrics(measurement)
        let medianImprovement = metrics.medianImprovement
        let medianBaseline = metrics.medianBaseline
        let medianCandidate = metrics.medianCandidate
        let improvementFraction = metrics.improvementFraction
        let candidateWinFraction = metrics.candidateWinFraction
        if metrics.baselineRelativeMAD > policy.maximumPlanRelativeMAD
            || metrics.candidateRelativeMAD > policy.maximumPlanRelativeMAD
        {
            return conservativeDecision(
                baseline,
                reason: .unstableMeasurement,
                measurement: measurement
            )
        }
        let requiredCandidateImprovement = max(
            max(
                policy.minimumAbsoluteImprovementUs,
                medianBaseline * policy.minimumRelativeImprovement
            ),
            metrics.pairedDeltaMADUs * policy.minimumDeltaSignalToNoise
        )
        if medianImprovement >= requiredCandidateImprovement,
           candidateWinFraction >= policy.minimumWinFraction
        {
            let baselineWall = measurement.samples.compactMap(\.baselineWallUs)
            let candidateWall = measurement.samples.compactMap(\.candidateWallUs)
            if baselineWall.count == measurement.samples.count,
               candidateWall.count == measurement.samples.count
            {
                let medianBaselineWall = Self.median(baselineWall)
                let medianCandidateWall = Self.median(candidateWall)
                let allowedWallRegression = max(
                    policy.maximumWallRegressionUs,
                    medianBaselineWall * policy.maximumWallRegressionFraction
                )
                if medianCandidateWall - medianBaselineWall > allowedWallRegression {
                    return SmeltPlanCostDecision(
                        selectedPlanID: baseline,
                        reason: .wallTimeRegression,
                        medianCandidateImprovementUs: medianImprovement,
                        medianCandidateImprovementFraction: improvementFraction,
                        candidateWinFraction: candidateWinFraction,
                        pairedDeltaMADUs: metrics.pairedDeltaMADUs,
                        baselineRelativeMAD: metrics.baselineRelativeMAD,
                        candidateRelativeMAD: metrics.candidateRelativeMAD
                    )
                }
            }
            return SmeltPlanCostDecision(
                selectedPlanID: candidate,
                reason: .candidateFaster,
                medianCandidateImprovementUs: medianImprovement,
                medianCandidateImprovementFraction: improvementFraction,
                candidateWinFraction: candidateWinFraction,
                pairedDeltaMADUs: metrics.pairedDeltaMADUs,
                baselineRelativeMAD: metrics.baselineRelativeMAD,
                candidateRelativeMAD: metrics.candidateRelativeMAD
            )
        }

        let baselineWinFraction = Double(
            measurement.samples.filter {
                $0.baselineGPUUs < $0.candidateGPUUs
            }.count
        ) / Double(measurement.samples.count)
        let requiredBaselineImprovement = max(
            max(
                policy.minimumAbsoluteImprovementUs,
                medianCandidate * policy.minimumRelativeImprovement
            ),
            metrics.pairedDeltaMADUs * policy.minimumDeltaSignalToNoise
        )
        if -medianImprovement >= requiredBaselineImprovement,
           baselineWinFraction >= policy.minimumWinFraction
        {
            return SmeltPlanCostDecision(
                selectedPlanID: baseline,
                reason: .baselineFaster,
                medianCandidateImprovementUs: medianImprovement,
                medianCandidateImprovementFraction: improvementFraction,
                candidateWinFraction: candidateWinFraction,
                pairedDeltaMADUs: metrics.pairedDeltaMADUs,
                baselineRelativeMAD: metrics.baselineRelativeMAD,
                candidateRelativeMAD: metrics.candidateRelativeMAD
            )
        }

        return SmeltPlanCostDecision(
            selectedPlanID: baseline,
            reason: .withinNoise,
            medianCandidateImprovementUs: medianImprovement,
            medianCandidateImprovementFraction: improvementFraction,
            candidateWinFraction: candidateWinFraction,
            pairedDeltaMADUs: metrics.pairedDeltaMADUs,
            baselineRelativeMAD: metrics.baselineRelativeMAD,
            candidateRelativeMAD: metrics.candidateRelativeMAD
        )
    }

    private func conservativeDecision(
        _ baseline: String,
        reason: SmeltPlanDecisionReason,
        measurement: SmeltPairedPlanMeasurement
    ) -> SmeltPlanCostDecision {
        let metrics = Self.metrics(measurement)
        return SmeltPlanCostDecision(
            selectedPlanID: baseline,
            reason: reason,
            medianCandidateImprovementUs: metrics.medianImprovement,
            medianCandidateImprovementFraction: metrics.improvementFraction,
            candidateWinFraction: metrics.candidateWinFraction,
            pairedDeltaMADUs: metrics.pairedDeltaMADUs,
            baselineRelativeMAD: metrics.baselineRelativeMAD,
            candidateRelativeMAD: metrics.candidateRelativeMAD
        )
    }

    private static func metrics(
        _ measurement: SmeltPairedPlanMeasurement
    ) -> (
        medianImprovement: Double,
        medianBaseline: Double,
        medianCandidate: Double,
        improvementFraction: Double,
        candidateWinFraction: Double,
        pairedDeltaMADUs: Double,
        baselineRelativeMAD: Double,
        candidateRelativeMAD: Double
    ) {
        let improvements = measurement.samples.map {
            $0.baselineGPUUs - $0.candidateGPUUs
        }
        let medianImprovement = median(improvements)
        let baselineSamples = measurement.samples.map(\.baselineGPUUs)
        let candidateSamples = measurement.samples.map(\.candidateGPUUs)
        let medianBaseline = median(baselineSamples)
        let medianCandidate = median(candidateSamples)
        return (
            medianImprovement,
            medianBaseline,
            medianCandidate,
            medianBaseline > 0 ? medianImprovement / medianBaseline : 0,
            improvements.isEmpty
                ? 0
                : Double(improvements.filter { $0 > 0 }.count)
                    / Double(improvements.count),
            medianAbsoluteDeviation(improvements) * 1.4826,
            medianBaseline > 0
                ? medianAbsoluteDeviation(baselineSamples) * 1.4826 / medianBaseline
                : 0,
            medianCandidate > 0
                ? medianAbsoluteDeviation(candidateSamples) * 1.4826 / medianCandidate
                : 0
        )
    }

    private static func medianAbsoluteDeviation(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let center = median(values)
        return median(values.map { abs($0 - center) })
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
}
