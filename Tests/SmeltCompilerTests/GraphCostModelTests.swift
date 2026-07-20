import Foundation
import Testing

@testable import SmeltCompiler
import SmeltSchema

@Suite("Smelt graph cost model")
struct GraphCostModelTests {
    private let context = SmeltCostModelContext(
        mode: .decode,
        sequenceLength: 1,
        position: 0
    )

    @Test("A locally faster fusion loses when it displaces a better whole-plan fusion")
    func completePlanBeatsLocalRewrite() throws {
        let baseline = SmeltCostedGraphPlan(id: "residual_norm", ops: [
            dispatch("matvec", groups: 128),
            dispatch("residual_norm", groups: 1, threads: 1_024),
            dispatch("activation_view", groups: 40),
        ])
        let candidate = SmeltCostedGraphPlan(id: "matvec_residual", ops: [
            dispatch("matvec_residual", groups: 128),
            dispatch("norm", groups: 1, threads: 1_024),
            dispatch("activation_view", groups: 40),
        ])
        let model = makeModel(
            plans: [baseline, candidate],
            costs: [
                "matvec": 60,
                "residual_norm": 30,
                "matvec_residual": 55,
                "norm": 40,
                "activation_view": 20,
            ]
        )

        let baselineEstimate = model.estimate(baseline, context: context)
        let candidateEstimate = model.estimate(candidate, context: context)

        #expect(baselineEstimate.predictedMedianGPUUs == 110)
        #expect(candidateEstimate.predictedMedianGPUUs == 115)
        // Looking only at matvec + residual would have called 55 < 60 + 10
        // a win. The complete plan exposes the displaced residual+norm win.
        let baselineGPUUs = try #require(baselineEstimate.predictedMedianGPUUs)
        let candidateGPUUs = try #require(candidateEstimate.predictedMedianGPUUs)
        #expect(baselineGPUUs < candidateGPUUs)
        let prediction = model.predict(
            baseline: baseline,
            candidate: candidate,
            context: context
        )
        #expect(prediction.verdict == .tooCloseToCall)
        #expect(prediction.predictedMedianTotalDeltaUs == 5)
        #expect(prediction.errorBandUs == 11.25)
    }

    @Test("Dispatch removal is not treated as a speedup when reduction fan-out is slower")
    func reductionFanoutIsMeasured() throws {
        let staged = SmeltCostedGraphPlan(id: "staged", ops: [
            dispatch("norm_reduction", groups: 1, threads: 1_024),
            dispatch("activation_view", groups: 40),
        ])
        let cooperative = SmeltCostedGraphPlan(id: "cooperative", ops: [
            dispatch("norm_reduction_activation", groups: 1, threads: 1_024),
        ])
        let model = makeModel(
            plans: [staged, cooperative],
            costs: [
                "norm_reduction": 35,
                "activation_view": 20,
                "norm_reduction_activation": 65,
            ]
        )

        let stagedEstimate = model.estimate(staged, context: context)
        let cooperativeEstimate = model.estimate(cooperative, context: context)

        #expect(stagedEstimate.structure.dispatchCount == 2)
        #expect(cooperativeEstimate.structure.dispatchCount == 1)
        #expect(stagedEstimate.structure.maxThreadgroupsPerDispatch == 40)
        #expect(cooperativeEstimate.structure.maxThreadgroupsPerDispatch == 1)
        #expect(stagedEstimate.predictedMedianGPUUs == 55)
        #expect(cooperativeEstimate.predictedMedianGPUUs == 65)
        let prediction = model.predict(
            baseline: staged,
            candidate: cooperative,
            context: context
        )
        #expect(prediction.verdict == .baselineFaster)
        #expect(prediction.predictedMedianTotalDeltaUs == 10)
        #expect(prediction.errorBandUs == 6)
    }

    @Test("Missing exact geometry calibration cannot produce a prediction")
    func missingCalibrationIsExplicit() {
        let plan = SmeltCostedGraphPlan(id: "uncalibrated", ops: [
            dispatch("new_kernel", groups: 7),
        ])
        let model = SmeltGraphCostModel(
            calibration: SmeltDeviceCostCalibration(
                provenanceKey: "m2-max:test",
                dispatchSamples: []
            )
        )

        let estimate = model.estimate(plan, context: context)

        #expect(!estimate.hasCompleteCalibration)
        #expect(estimate.predictedMedianGPUUs == nil)
        #expect(estimate.missingCalibrationKeys.count == 1)
        #expect(estimate.missingCalibrationKeys[0].contains("new_kernel"))
        let prediction = model.predict(
            baseline: plan,
            candidate: plan,
            context: context
        )
        #expect(prediction.verdict == .unknown)
        #expect(prediction.predictedMedianTotalDeltaUs == nil)
        #expect(prediction.missingCalibrationKeys.count == 1)
    }

    @Test("A calibration from another execution context cannot predict")
    func mismatchedCalibrationContextIsUnknown() {
        let kernel = dispatch("kernel", groups: 1)
        let plan = SmeltCostedGraphPlan(id: "same-kernel", ops: [
            kernel,
        ])
        let otherContext = SmeltCostModelContext(
            mode: .decode,
            sequenceLength: 1,
            position: 16
        )
        guard case .dispatch(let dispatch) = kernel,
              let key = SmeltGraphCostModel.dispatchCostKey(
                for: dispatch,
                context: context
              )
        else {
            Issue.record("expected a dispatch cost key")
            return
        }
        let model = SmeltGraphCostModel(
            calibration: SmeltDeviceCostCalibration(
                provenanceKey: "other-context",
                context: otherContext,
                dispatchSamples: [SmeltDispatchCostSample(
                    key: key,
                    medianGPUUs: 1,
                    p95GPUUs: 2,
                    sampleCount: 20
                )]
            )
        )

        let prediction = model.predict(
            baseline: plan,
            candidate: plan,
            context: context
        )
        #expect(prediction.verdict == .unknown)
        #expect(prediction.missingCalibrationKeys.contains {
            $0.hasPrefix("calibration-context:")
        })
    }

    @Test("Interleaved paired evidence selects a material exact speedup")
    func pairedEvidenceSelectsCandidate() {
        let model = SmeltGraphCostModel(
            calibration: SmeltDeviceCostCalibration(
                provenanceKey: "m2-max:metallib:package",
                dispatchSamples: []
            )
        )
        let samples = (0..<10).map { index in
            SmeltPairedPlanSample(
                order: index.isMultiple(of: 2) ? .baselineFirst : .candidateFirst,
                baselineGPUUs: 25_500 + Double(index % 3) * 10,
                candidateGPUUs: 25_180 + Double(index % 4) * 10
            )
        }

        let decision = model.decide(measurement: SmeltPairedPlanMeasurement(
            provenanceKey: "m2-max:metallib:package",
            context: context,
            baselinePlanID: "baseline",
            candidatePlanID: "candidate",
            exactOutputMatch: true,
            samples: samples
        ))

        #expect(decision.selectedPlanID == "candidate")
        #expect(decision.reason == .candidateFaster)
        #expect(decision.medianCandidateImprovementFraction > 0.005)
        #expect(decision.candidateWinFraction == 1)
    }

    @Test("Sub-noise paired result retains the stable baseline")
    func pairedEvidenceRejectsNoise() {
        let model = SmeltGraphCostModel(
            calibration: SmeltDeviceCostCalibration(
                provenanceKey: "m2-max:metallib:package",
                dispatchSamples: []
            )
        )
        let samples = (0..<10).map { index in
            SmeltPairedPlanSample(
                order: index.isMultiple(of: 2) ? .baselineFirst : .candidateFirst,
                baselineGPUUs: 25_500 + Double(index % 3) * 15,
                candidateGPUUs: 25_470 + Double((index + 1) % 3) * 15
            )
        }

        let decision = model.decide(measurement: SmeltPairedPlanMeasurement(
            provenanceKey: "m2-max:metallib:package",
            context: context,
            baselinePlanID: "baseline",
            candidatePlanID: "candidate",
            exactOutputMatch: true,
            samples: samples
        ))

        #expect(decision.selectedPlanID == "baseline")
        #expect(decision.reason == .withinNoise)
    }

    @Test("GPU win with a host-path regression retains baseline")
    func wallTimeRegressionBlocksCandidate() {
        let model = SmeltGraphCostModel(
            calibration: SmeltDeviceCostCalibration(
                provenanceKey: "m2-max:metallib:package",
                dispatchSamples: []
            )
        )
        let samples = (0..<10).map { index in
            SmeltPairedPlanSample(
                order: index.isMultiple(of: 2) ? .baselineFirst : .candidateFirst,
                baselineGPUUs: 25_500,
                candidateGPUUs: 25_100,
                baselineWallUs: 26_000,
                candidateWallUs: 26_500
            )
        }

        let decision = model.decide(measurement: SmeltPairedPlanMeasurement(
            provenanceKey: "m2-max:metallib:package",
            context: context,
            baselinePlanID: "baseline",
            candidatePlanID: "candidate",
            exactOutputMatch: true,
            samples: samples
        ))

        #expect(decision.selectedPlanID == "baseline")
        #expect(decision.reason == .wallTimeRegression)
    }

    @Test("Identical dispatch tables cannot win through timing noise")
    func identicalPlansRetainBaseline() {
        let model = SmeltGraphCostModel(
            calibration: SmeltDeviceCostCalibration(
                provenanceKey: "m2-max:metallib:package",
                dispatchSamples: []
            )
        )
        let measurement = SmeltPairedPlanMeasurement(
            provenanceKey: "m2-max:metallib:package",
            context: context,
            baselinePlanID: "baseline",
            candidatePlanID: "candidate",
            exactOutputMatch: true,
            samples: [
                SmeltPairedPlanSample(
                    order: .baselineFirst,
                    baselineGPUUs: 25_600,
                    candidateGPUUs: 24_900
                ),
            ],
            baselineTableSHA256: "same",
            candidateTableSHA256: "same"
        )

        let decision = model.decide(measurement: measurement)

        #expect(decision.selectedPlanID == "baseline")
        #expect(decision.reason == .identicalPlans)
        #expect(decision.medianCandidateImprovementUs == 700)
    }

    @Test("High-variance contended measurements cannot select a candidate")
    func unstableMeasurementsRetainBaseline() {
        let model = SmeltGraphCostModel(
            calibration: SmeltDeviceCostCalibration(
                provenanceKey: "m2-max:metallib:package",
                dispatchSamples: []
            )
        )
        let samples = (0..<10).map { index in
            let baseline = index.isMultiple(of: 2) ? 20_000.0 : 30_000.0
            return SmeltPairedPlanSample(
                order: index.isMultiple(of: 2) ? .baselineFirst : .candidateFirst,
                baselineGPUUs: baseline,
                candidateGPUUs: baseline - 500,
                baselineWallUs: baseline + 500,
                candidateWallUs: baseline
            )
        }
        let measurement = SmeltPairedPlanMeasurement(
            provenanceKey: "m2-max:metallib:package",
            context: context,
            baselinePlanID: "baseline",
            candidatePlanID: "candidate",
            exactOutputMatch: true,
            samples: samples
        )

        let decision = model.decide(measurement: measurement)

        #expect(decision.selectedPlanID == "baseline")
        #expect(decision.reason == .unstableMeasurement)
        #expect(decision.baselineRelativeMAD > 0.03)
    }

    @Test("Recorded Bonsai experiments prove structural proxies can point backward")
    func recordedBonsaiExperiments() throws {
        struct Fixture: Decodable {
            struct Observation: Decodable {
                let id: String
                let baselineRecords: Int
                let candidateRecords: Int
                let baselinePureGPUms: Double
                let candidatePureGPUms: Double
                let exact: Bool
                let expectedPrediction: SmeltPlanCostPredictionVerdict
            }
            let observations: [Observation]
        }

        let url = try #require(
            Bundle.module.url(
                forResource: "bonsai-ternary-cost-observations",
                withExtension: "json",
                subdirectory: "Fixtures"
            )
        )
        let fixture = try JSONDecoder().decode(
            Fixture.self,
            from: Data(contentsOf: url)
        )
        let cooperative = try #require(fixture.observations.first {
            $0.id == "residual_norm_activation_cooperative"
        })
        let localFusion = try #require(fixture.observations.first {
            $0.id == "matvec_residual_displaces_residual_norm"
        })

        #expect(cooperative.exact)
        #expect(cooperative.candidateRecords < cooperative.baselineRecords)
        #expect(cooperative.candidatePureGPUms > cooperative.baselinePureGPUms)
        #expect(localFusion.expectedPrediction == .tooCloseToCall)
        #expect(cooperative.expectedPrediction == .baselineFaster)

        let localBaseline = SmeltCostedGraphPlan(id: "residual_norm", ops: [
            dispatch("matvec", groups: 128),
            dispatch("residual_norm", groups: 1, threads: 1_024),
            dispatch("activation_view", groups: 40),
        ])
        let localCandidate = SmeltCostedGraphPlan(id: "matvec_residual", ops: [
            dispatch("matvec_residual", groups: 128),
            dispatch("norm", groups: 1, threads: 1_024),
            dispatch("activation_view", groups: 40),
        ])
        let localModel = makeModel(
            plans: [localBaseline, localCandidate],
            costs: [
                "matvec": 60, "residual_norm": 30,
                "matvec_residual": 55, "norm": 40,
                "activation_view": 20,
            ]
        )
        #expect(localModel.predict(
            baseline: localBaseline,
            candidate: localCandidate,
            context: context
        ).verdict == localFusion.expectedPrediction)

        let staged = SmeltCostedGraphPlan(id: "staged", ops: [
            dispatch("norm_reduction", groups: 1, threads: 1_024),
            dispatch("activation_view", groups: 40),
        ])
        let cooperativePlan = SmeltCostedGraphPlan(id: "cooperative", ops: [
            dispatch("norm_reduction_activation", groups: 1, threads: 1_024),
        ])
        let cooperativeModel = makeModel(
            plans: [staged, cooperativePlan],
            costs: [
                "norm_reduction": 35,
                "activation_view": 20,
                "norm_reduction_activation": 65,
            ]
        )
        #expect(cooperativeModel.predict(
            baseline: staged,
            candidate: cooperativePlan,
            context: context
        ).verdict == cooperative.expectedPrediction)
    }

    private func makeModel(
        plans: [SmeltCostedGraphPlan],
        costs: [String: Double]
    ) -> SmeltGraphCostModel {
        var samplesByKey: [SmeltDispatchCostKey: SmeltDispatchCostSample] = [:]
        for plan in plans {
            for op in plan.ops {
                guard case .dispatch(let dispatch) = op,
                      let key = SmeltGraphCostModel.dispatchCostKey(
                          for: dispatch,
                          context: context
                      ),
                      let cost = costs[key.pipeline]
                else {
                    continue
                }
                samplesByKey[key] = SmeltDispatchCostSample(
                    key: key,
                    medianGPUUs: cost,
                    p95GPUUs: cost * 1.05,
                    sampleCount: 20
                )
            }
        }
        return SmeltGraphCostModel(
            calibration: SmeltDeviceCostCalibration(
                provenanceKey: "m2-max:test",
                dispatchSamples: Array(samplesByKey.values)
            )
        )
    }

    private func dispatch(
        _ name: String,
        groups: Int,
        threads: Int = 64
    ) -> SmeltIROp {
        .dispatch(SmeltDispatch(
            pipeline: .elementwiseAdd,
            pipelineNameOverride: name,
            buffers: [],
            constants: [],
            dispatch: .threadgroups(
                width: groups,
                height: 1,
                depth: 1,
                tgWidth: threads,
                tgHeight: 1,
                tgDepth: 1
            )
        ))
    }
}
