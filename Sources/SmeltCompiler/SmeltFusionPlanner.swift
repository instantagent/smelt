// SmeltFusionPlanner - Selects legal fused kernel routes for compiler emitters.
//
// The auto policy mirrors the current tuned shape registry exactly. Generated
// model-shape wrappers are only selected from an explicit SmeltKernelPlan.

import Foundation

enum SmeltFusionPolicy: String, Sendable {
    case auto
}

enum SmeltFusionKind: Sendable, Equatable {
    case specializedKernel
    case genericKernel
    case dispatchCoalescing
    case staged
}

struct SmeltFusionPlanner: Sendable {
    static let auto = SmeltFusionPlanner(policy: .auto)
    static let verifySmallBatch = SmeltFusionPlanner(
        policy: .auto,
        preferVerifySmallBatchKernels: true
    )

    let policy: SmeltFusionPolicy
    let preferVerifySmallBatchKernels: Bool
    private let kernelPlan: SmeltKernelPlan?

    init(
        policy: SmeltFusionPolicy = .auto,
        preferVerifySmallBatchKernels: Bool = false,
        kernelPlan: SmeltKernelPlan? = nil
    ) {
        self.policy = policy
        self.preferVerifySmallBatchKernels = preferVerifySmallBatchKernels
        self.kernelPlan = kernelPlan
    }

    static func planned(
        kernelPlan: SmeltKernelPlan,
        preferVerifySmallBatchKernels: Bool = false
    ) -> SmeltFusionPlanner {
        return SmeltFusionPlanner(
            policy: .auto,
            preferVerifySmallBatchKernels: preferVerifySmallBatchKernels,
            kernelPlan: kernelPlan
        )
    }

    func rewrite(window: ArraySlice<SmeltIROp>) -> SmeltFusionRewrite? {
        switch policy {
        case .auto:
            return bestRewrite(
                window: window,
                capabilities: Self.kernelCapabilities(respectDecodeCostModel: true)
            )
        }
    }

    func sourceVisibleRewrite(window: ArraySlice<SmeltIROp>) -> SmeltFusionRewrite? {
        switch policy {
        case .auto:
            return bestRewrite(
                window: window,
                capabilities: Self.kernelCapabilities(respectDecodeCostModel: false)
            )
        }
    }

    func decodeAffineMatvec(
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> SmeltAffineMatvecRoute {
        let pipeline = SmeltKernelShapeRegistry.decodeAffineDecodePipeline(
            rows: rows,
            cols: cols,
            groupSize: groupSize
        )
        return affineRoute(for: pipeline)
    }

    func unrolledPrefillAffineMatvec(
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> SmeltAffineMatvecRoute {
        let pipeline = SmeltKernelShapeRegistry.decodeAffinePipeline(
            rows: rows,
            cols: cols,
            groupSize: groupSize
        )
        return affineRoute(for: pipeline)
    }

    func prefillAffineFull(
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> SmeltPrefillTiledRoute? {
        guard let pipeline = SmeltKernelShapeRegistry.prefillAffineFullPipeline(
            rows: rows,
            cols: cols,
            groupSize: groupSize
        ) else {
            return nil
        }
        return SmeltPrefillTiledRoute(
            pipeline: pipeline,
            kind: .specializedKernel,
            rowTile: SmeltKernelShapeRegistry.prefillAffineFullRowTile(pipeline),
            batchTile: SmeltKernelShapeRegistry.prefillAffineFullBatchTile(pipeline),
            threadgroupWidth: SmeltKernelShapeRegistry.prefillAffineFullThreads(pipeline)
        )
    }

    func prefillAffineBatched(
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> SmeltPrefillTiledRoute? {
        if preferVerifySmallBatchKernels,
           let pipeline = SmeltKernelShapeRegistry.prefillAffineSmallBatchPipeline(
               rows: rows,
               cols: cols,
               groupSize: groupSize
           )
        {
            return SmeltPrefillTiledRoute(
                pipeline: pipeline,
                kind: .specializedKernel,
                rowTile: SmeltKernelShapeRegistry.prefillAffineSmallBatchRowTile(pipeline),
                batchTile: SmeltKernelShapeRegistry.prefillAffineSmallBatchBatchTile(pipeline),
                threadgroupWidth: SmeltKernelShapeRegistry.prefillAffineSmallBatchThreads(pipeline)
            )
        }
        guard let pipeline = SmeltKernelShapeRegistry.prefillAffineBatchedPipeline(
            rows: rows,
            cols: cols,
            groupSize: groupSize
        ) else {
            return nil
        }
        return SmeltPrefillTiledRoute(
            pipeline: pipeline,
            kind: .specializedKernel,
            rowTile: 8,
            batchTile: SmeltKernelShapeRegistry.prefillAffineBatchedBatchTile(pipeline),
            threadgroupWidth: 64
        )
    }

    func decodeDualAffineMatvec(
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> SmeltOptionalPipelineRoute {
        let pipeline = SmeltKernelShapeRegistry.decodeDualAffinePipeline(
            rows: rows,
            cols: cols,
            groupSize: groupSize
        )
        return optionalRoute(for: pipeline)
    }

    func prefillDualAffineMatvec(
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> SmeltOptionalPipelineRoute {
        let pipeline = SmeltKernelShapeRegistry.prefillDualAffinePipeline(
            rows: rows,
            cols: cols,
            groupSize: groupSize
        )
        return optionalRoute(for: pipeline)
    }

    func decodeFusedSwiGLU(
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> SmeltDecodeFusedFFNRoute {
        let pipeline = SmeltKernelShapeRegistry.decodeFusedFFNPipeline(
            rows: rows,
            cols: cols,
            groupSize: groupSize
        )
        return fusedFFNRoute(for: pipeline)
    }

    func decodeFusedGeGLU(
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> SmeltDecodeFusedFFNRoute {
        let pipeline = SmeltKernelShapeRegistry.decodeFusedGeGLUPipeline(
            rows: rows,
            cols: cols,
            groupSize: groupSize
        )
        return fusedFFNRoute(for: pipeline)
    }

    func prefillFusedGateUpFull(
        rows: Int,
        cols: Int,
        groupSize: Int,
        activation: SmeltActivation
    ) -> SmeltPrefillTiledRoute? {
        guard let pipeline = SmeltKernelShapeRegistry.prefillFusedGateUpFullPipeline(
            rows: rows,
            cols: cols,
            groupSize: groupSize,
            activation: activation,
            preferVerifySmallBatch: preferVerifySmallBatchKernels
        ) else {
            return nil
        }
        return SmeltPrefillTiledRoute(
            pipeline: pipeline,
            kind: .specializedKernel,
            rowTile: SmeltKernelShapeRegistry.prefillFusedGateUpFullRowTile(pipeline),
            batchTile: SmeltKernelShapeRegistry.prefillFusedGateUpFullBatchTile(pipeline),
            threadgroupWidth: SmeltKernelShapeRegistry.prefillFusedGateUpFullThreads(pipeline)
        )
    }

    func decodeNormScaleAffine(
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> SmeltDecodeNormScaleRoute? {
        guard let pipeline = SmeltKernelShapeRegistry.decodeNormScaleAffinePipeline(
            rows: rows,
            cols: cols,
            groupSize: groupSize
        ) else {
            return nil
        }
        let rowTile = SmeltKernelShapeRegistry.decodeNormScaleAffineUsesRows4(pipeline)
            ? 4
            : 8
        return SmeltDecodeNormScaleRoute(
            pipeline: pipeline,
            kind: .specializedKernel,
            rowTile: rowTile
        )
    }

    func decodeNormScaleGeGLU(
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> SmeltDecodeNormScaleRoute? {
        guard let pipeline = SmeltKernelShapeRegistry.decodeNormScaleGeGLUPipeline(
            rows: rows,
            cols: cols,
            groupSize: groupSize
        ) else {
            return nil
        }
        return SmeltDecodeNormScaleRoute(
            pipeline: pipeline,
            kind: .specializedKernel,
            rowTile: 4
        )
    }

    func decodeRMSNorm(dim: Int, eps: Float) -> SmeltRMSNormRoute {
        let pipeline = SmeltKernelShapeRegistry.decodeRmsNormPipeline(dim: dim, eps: eps)
        return SmeltRMSNormRoute(
            pipeline: pipeline,
            kind: fusionKind(for: pipeline),
            threadgroupWidth: SmeltKernelShapeRegistry.rmsNormThreads(pipeline)
        )
    }

    func decodeRMSNormVariableInput(dim: Int, eps: Float) -> SmeltRMSNormRoute {
        let pipeline = SmeltKernelShapeRegistry.decodeRmsNormPipeline(dim: dim, eps: eps)
        let threadgroupWidth: Int?
        switch pipeline {
        case .rmsNorm1PWD1024:
            threadgroupWidth = 128
        case .rmsNorm1PWD2560:
            threadgroupWidth = 320
        case .some(_):
            threadgroupWidth = 256
        case nil:
            threadgroupWidth = nil
        }
        return SmeltRMSNormRoute(
            pipeline: pipeline,
            kind: fusionKind(for: pipeline),
            threadgroupWidth: threadgroupWidth
        )
    }

    func prefillRMSNorm(dim: Int, eps: Float) -> SmeltRMSNormRoute {
        let pipeline = SmeltKernelShapeRegistry.batchedRmsNormPipeline(dim: dim, eps: eps)
        return SmeltRMSNormRoute(
            pipeline: pipeline,
            kind: fusionKind(for: pipeline),
            threadgroupWidth: SmeltKernelShapeRegistry.rmsNormThreads(pipeline)
        )
    }

    private func affineRoute(for pipeline: SmeltPipeline?) -> SmeltAffineMatvecRoute {
        SmeltAffineMatvecRoute(
            pipeline: pipeline,
            kind: fusionKind(for: pipeline),
            rowTile: SmeltKernelShapeRegistry.decodeAffineUsesRows4(pipeline) ? 4 : 8
        )
    }

    private func fusedFFNRoute(for pipeline: SmeltPipeline?) -> SmeltDecodeFusedFFNRoute {
        SmeltDecodeFusedFFNRoute(
            pipeline: pipeline,
            kind: fusionKind(for: pipeline),
            rowTile: SmeltKernelShapeRegistry.decodeFusedFFNUsesRows4(pipeline) ? 4 : 8,
            threadgroupWidth: SmeltKernelShapeRegistry.decodeFusedFFNThreads(pipeline)
        )
    }

    private func optionalRoute(for pipeline: SmeltPipeline?) -> SmeltOptionalPipelineRoute {
        SmeltOptionalPipelineRoute(
            pipeline: pipeline,
            kind: fusionKind(for: pipeline)
        )
    }

    private func fusionKind(for pipeline: SmeltPipeline?) -> SmeltFusionKind {
        pipeline == nil ? .genericKernel : .specializedKernel
    }

    private struct SmeltFusionKernelCapability: Sendable {
        let rule: SmeltFusionRule
        let minWindowSize: Int
        let maxWindowSize: Int
        let priority: Int
        let match: @Sendable (SmeltFusionPlanner, ArraySlice<SmeltIROp>) -> SmeltFusionRewrite?
    }

    private static func kernelCapabilities(
        respectDecodeCostModel: Bool
    ) -> [SmeltFusionKernelCapability] {
        [
            SmeltFusionKernelCapability(
                rule: .contiguousL2Normalize,
                minWindowSize: 2,
                maxWindowSize: 2,
                priority: 10,
                match: { planner, window in
                    planner.l2NormalizeContiguousRewrite(window: window)
                }
            ),
            SmeltFusionKernelCapability(
                rule: .matvecResidualAdd,
                minWindowSize: 2,
                maxWindowSize: 4,
                priority: 20,
                match: { planner, window in
                    planner.matvecResidualAddRewrite(window: window)
                }
            ),
            SmeltFusionKernelCapability(
                rule: .rmsNormResidualAdd,
                minWindowSize: 2,
                maxWindowSize: 2,
                priority: 25,
                match: { planner, window in
                    planner.rmsNormResidualAddRewrite(window: window)
                }
            ),
            SmeltFusionKernelCapability(
                rule: .rmsNormResidualAddScalarWeight,
                minWindowSize: 2,
                maxWindowSize: 2,
                priority: 28,
                match: { planner, window in
                    planner.rmsNormResidualAddScalarWeightRewrite(window: window)
                }
            ),
            SmeltFusionKernelCapability(
                rule: .cooperativeNormScaleConsumer,
                minWindowSize: 2,
                maxWindowSize: 8,
                priority: 30,
                match: { planner, window in
                    planner.cooperativeNormScaleConsumerRewrite(
                        window: window,
                        respectDecodeCostModel: respectDecodeCostModel
                    )
                }
            ),
            SmeltFusionKernelCapability(
                rule: .normActivationView,
                minWindowSize: 2,
                maxWindowSize: 4,
                priority: 31,
                match: { planner, window in
                    planner.normActivationViewRewrite(window: window)
                }
            ),
            SmeltFusionKernelCapability(
                rule: .residualAddNormActivationView,
                minWindowSize: 3,
                maxWindowSize: 4,
                priority: 32,
                match: { planner, window in
                    planner.residualAddNormActivationViewRewrite(window: window)
                }
            ),
            SmeltFusionKernelCapability(
                rule: .gatedNormActivationView,
                minWindowSize: 2,
                maxWindowSize: 4,
                priority: 32,
                match: { planner, window in
                    planner.gatedNormActivationViewRewrite(window: window)
                }
            ),
            SmeltFusionKernelCapability(
                rule: .sigmoidMulActivationView,
                minWindowSize: 2,
                maxWindowSize: 4,
                priority: 33,
                match: { planner, window in
                    planner.sigmoidMulActivationViewRewrite(window: window)
                }
            ),
            SmeltFusionKernelCapability(
                rule: .swigluActivationView,
                minWindowSize: 2,
                maxWindowSize: 4,
                priority: 34,
                match: { planner, window in
                    planner.swigluActivationViewRewrite(window: window)
                }
            ),
            SmeltFusionKernelCapability(
                rule: .fusedNormRopeKVPrefill,
                minWindowSize: 2,
                maxWindowSize: 8,
                priority: 35,
                match: { planner, window in
                    planner.fusedNormRopeKVPrefillRewrite(window: window)
                }
            ),
            SmeltFusionKernelCapability(
                rule: .dualMatvecActivation,
                minWindowSize: 3,
                maxWindowSize: 3,
                priority: 40,
                match: { planner, window in
                    planner.dualMatvecActivationRewrite(window: window)
                }
            ),
        ]
    }

    private func bestRewrite(
        window: ArraySlice<SmeltIROp>,
        capabilities: [SmeltFusionKernelCapability]
    ) -> SmeltFusionRewrite? {
        var best: (capability: SmeltFusionKernelCapability, rewrite: SmeltFusionRewrite)?

        for capability in capabilities where window.count >= capability.minWindowSize {
            let boundedWindow = window.prefix(
                min(window.count, capability.maxWindowSize)
            )
            guard let rewrite = capability.match(self, boundedWindow),
                  rewrite.rule == capability.rule,
                  rewrite.consumedOpCount > 0,
                  rewrite.consumedOpCount <= boundedWindow.count
            else {
                continue
            }

            guard let current = best else {
                best = (capability, rewrite)
                continue
            }

            if isBetterRewrite(
                candidate: rewrite,
                candidateCapability: capability,
                current: current.rewrite,
                currentCapability: current.capability
            ) {
                best = (capability, rewrite)
            }
        }

        return best?.rewrite
    }

    private func isBetterRewrite(
        candidate: SmeltFusionRewrite,
        candidateCapability: SmeltFusionKernelCapability,
        current: SmeltFusionRewrite,
        currentCapability: SmeltFusionKernelCapability
    ) -> Bool {
        if candidate.consumedOpCount != current.consumedOpCount {
            return candidate.consumedOpCount > current.consumedOpCount
        }
        let candidateKindRank = fusionKindRank(candidate.kind)
        let currentKindRank = fusionKindRank(current.kind)
        if candidateKindRank != currentKindRank {
            return candidateKindRank > currentKindRank
        }
        return candidateCapability.priority > currentCapability.priority
    }

    private func fusionKindRank(_ kind: SmeltFusionKind) -> Int {
        switch kind {
        case .specializedKernel:
            return 3
        case .genericKernel:
            return 2
        case .dispatchCoalescing:
            return 1
        case .staged:
            return 0
        }
    }

    private func l2NormalizeContiguousRewrite(
        window: ArraySlice<SmeltIROp>
    ) -> SmeltFusionRewrite? {
        guard window.count >= 2,
              case .dispatch(let a) = window[window.startIndex],
              case .dispatch(let b) = window[window.index(after: window.startIndex)],
              a.buffers.count == 1,
              b.buffers.count == 1
        else {
            return nil
        }

        let isGenericL2 = a.pipeline == .l2Normalize && b.pipeline == .l2Normalize
        let isD128L2 = a.pipeline == .l2NormalizeD128 && b.pipeline == .l2NormalizeD128
        guard isGenericL2 || isD128L2 else { return nil }

        let aBuffer = a.buffers[0]
        let bBuffer = b.buffers[0]
        guard aBuffer.slot == bBuffer.slot,
              aBuffer.offsetExpression == nil,
              bBuffer.offsetExpression == nil,
              aBuffer.offsetKind == 0,
              bBuffer.offsetKind == 0
        else {
            return nil
        }

        let headDim: UInt64
        if isGenericL2 {
            guard a.constants.count == 2,
                  b.constants.count == 2,
                  a.constants[0].expression == b.constants[0].expression,
                  a.constants[1].expression == b.constants[1].expression,
                  let parsedHeadDim = UInt64(a.constants[0].expression)
            else {
                return nil
            }
            headDim = parsedHeadDim
        } else {
            guard a.constants.isEmpty, b.constants.isEmpty else { return nil }
            headDim = 128
        }

        guard case .threadgroups(let aw, _, _, let atw, let ath, let atd) = a.dispatch,
              case .threadgroups(let bw, _, _, let btw, let bth, let btd) = b.dispatch,
              atw == btw,
              ath == bth,
              atd == btd
        else {
            return nil
        }

        let expectedBOffset = aBuffer.byteOffset + UInt64(aw) * headDim * 2
        guard bBuffer.byteOffset == expectedBOffset else { return nil }

        let fused = SmeltDispatch(
            pipeline: a.pipeline,
            buffers: [aBuffer],
            constants: a.constants,
            dispatch: .threadgroups(
                width: aw + bw,
                height: 1,
                depth: 1,
                tgWidth: atw,
                tgHeight: ath,
                tgDepth: atd
            ),
            comment: "L2 normalize Q+K (fused)"
        )

        return SmeltFusionRewrite(
            consumedOpCount: 2,
            producedOps: [.dispatch(fused)],
            kind: .dispatchCoalescing,
            rule: .contiguousL2Normalize
        )
    }

    private struct CooperativeNormScaleRule {
        let producerPipeline: SmeltPipeline?
        let consumerPipeline: SmeltPipeline
        let scaledPipeline: SmeltPipeline
        let consumerInputIndex: Int
        let writesNormOutput: Bool
        // Some decode shapes have a legal cooperative norm-scale kernel, but
        // profiling shows materializing the norm is faster because the scaled
        // consumer would otherwise redo the norm transform for many row tiles.
        let allowDecodeCooperativeScale: Bool
        let producerBufferIndicesBeforeOutput: [Int]

        init(
            producerPipeline: SmeltPipeline? = nil,
            consumerPipeline: SmeltPipeline,
            scaledPipeline: SmeltPipeline,
            consumerInputIndex: Int,
            writesNormOutput: Bool,
            allowDecodeCooperativeScale: Bool = true,
            producerBufferIndicesBeforeOutput: [Int] = []
        ) {
            self.producerPipeline = producerPipeline
            self.consumerPipeline = consumerPipeline
            self.scaledPipeline = scaledPipeline
            self.consumerInputIndex = consumerInputIndex
            self.writesNormOutput = writesNormOutput
            self.allowDecodeCooperativeScale = allowDecodeCooperativeScale
            self.producerBufferIndicesBeforeOutput = producerBufferIndicesBeforeOutput
        }
    }

    private struct CooperativeNormParameters {
        let dim: String
        let eps: String
        let scalePipeline: SmeltPipeline
        let scaleTgWidth: Int
        let normOutputIndex: Int
        let requiresProducerSpecificRule: Bool

        init(
            dim: String,
            eps: String,
            scalePipeline: SmeltPipeline,
            scaleTgWidth: Int,
            normOutputIndex: Int = 2,
            requiresProducerSpecificRule: Bool = false
        ) {
            self.dim = dim
            self.eps = eps
            self.scalePipeline = scalePipeline
            self.scaleTgWidth = scaleTgWidth
            self.normOutputIndex = normOutputIndex
            self.requiresProducerSpecificRule = requiresProducerSpecificRule
        }
    }

    private static let cooperativeNormScaleRules: [CooperativeNormScaleRule] = [
        CooperativeNormScaleRule(
            consumerPipeline: .fusedAffineGateUpSwigluC1024R3584G64BatchedFull,
            scaledPipeline: .normScaleAffineGateUpSwigluC1024R3584G64BatchedFull,
            consumerInputIndex: 6,
            writesNormOutput: true
        ),
        CooperativeNormScaleRule(
            consumerPipeline: .fusedAffineGateUpSwigluC2048R6144G64BatchedFull,
            scaledPipeline: .normScaleAffineGateUpSwigluC2048R6144G64BatchedFull,
            consumerInputIndex: 6,
            writesNormOutput: true
        ),
        CooperativeNormScaleRule(
            consumerPipeline: .fusedAffineGateUpSwigluC2048R8192G64BatchedFull,
            scaledPipeline: .normScaleAffineGateUpSwigluC2048R8192G64BatchedFull,
            consumerInputIndex: 6,
            writesNormOutput: true
        ),
        CooperativeNormScaleRule(
            consumerPipeline: .fusedAffineGateUpSwigluC2560R9216G64BatchedFull,
            scaledPipeline: .normScaleAffineGateUpSwigluC2560R9216G64BatchedFull,
            consumerInputIndex: 6,
            writesNormOutput: true
        ),
        CooperativeNormScaleRule(
            consumerPipeline: .fusedAffineGateUpSwigluC3072R8192G64BatchedFull,
            scaledPipeline: .normScaleAffineGateUpSwigluC3072R8192G64BatchedFull,
            consumerInputIndex: 6,
            writesNormOutput: true
        ),
        CooperativeNormScaleRule(
            consumerPipeline: .fusedAffineGateUpGeGLUC2560R10240G128Rows4,
            scaledPipeline: .normScaleAffineGateUpGeGLUC2560R10240G128Rows4,
            consumerInputIndex: 6,
            writesNormOutput: false,
            allowDecodeCooperativeScale: false
        ),
        CooperativeNormScaleRule(
            consumerPipeline: .fusedAffineGateUpGeGLUC2560R10240G128BatchedFull,
            scaledPipeline: .normScaleAffineGateUpGeGLUC2560R10240G128BatchedFull,
            consumerInputIndex: 6,
            writesNormOutput: true
        ),
        // c2560/r2048 (sliding Q) and c2560/r4096 (global Q) cooperative
        // fusion rules excluded: bench showed +5 ms verify from the
        // per-TG norm-scale redundancy at those row counts. The r=10240
        // FFN rule above stays — its larger row count amortizes the
        // saved norm pass past the per-TG cost.
        //
        // c256/r2048 (draft-model GeGLU) cooperative rule excluded: with
        // writesNormOutput=false the scaled kernel skips materializing
        // slot 8, so a trace marker over the pre-FFN norm would read
        // stale contents in full/stripped-markers mode. Bench was
        // neutral on the draft model anyway, so safer to keep the
        // marker-correct unfused path.
        CooperativeNormScaleRule(
            consumerPipeline: .affineMatvecC1024R6144G64BatchedFull,
            scaledPipeline: .normScaleAffineMatvecC1024R6144G64BatchedFull,
            consumerInputIndex: 3,
            writesNormOutput: true
        ),
        CooperativeNormScaleRule(
            consumerPipeline: .affineMatvecC1024R4096G64BatchedFull,
            scaledPipeline: .normScaleAffineMatvecC1024R4096G64BatchedFull,
            consumerInputIndex: 3,
            writesNormOutput: true
        ),
        CooperativeNormScaleRule(
            consumerPipeline: .affineMatvecC2048R2048G64BatchedFull,
            scaledPipeline: .normScaleAffineMatvecC2048R2048G64BatchedFull,
            consumerInputIndex: 3,
            writesNormOutput: true
        ),
        CooperativeNormScaleRule(
            consumerPipeline: .affineMatvecC2048R6144G64BatchedFull,
            scaledPipeline: .normScaleAffineMatvecC2048R6144G64BatchedFull,
            consumerInputIndex: 3,
            writesNormOutput: true
        ),
        CooperativeNormScaleRule(
            consumerPipeline: .affineMatvecC2048R4096G64BatchedFull,
            scaledPipeline: .normScaleAffineMatvecC2048R4096G64BatchedFull,
            consumerInputIndex: 3,
            writesNormOutput: true
        ),
        CooperativeNormScaleRule(
            consumerPipeline: .affineMatvecC2560R8192G64BatchedFull,
            scaledPipeline: .normScaleAffineMatvecC2560R8192G64BatchedFull,
            consumerInputIndex: 3,
            writesNormOutput: true
        ),
        CooperativeNormScaleRule(
            consumerPipeline: .affineMatvecC3072R3072G64BatchedFull,
            scaledPipeline: .normScaleAffineMatvecC3072R3072G64BatchedFull,
            consumerInputIndex: 3,
            writesNormOutput: true
        ),
        CooperativeNormScaleRule(
            consumerPipeline: .affineMatvecC2560R10240G128Batched,
            scaledPipeline: .normScaleAffineMatvecC2560R10240G128Batched,
            consumerInputIndex: 3,
            writesNormOutput: true
        ),
        CooperativeNormScaleRule(
            consumerPipeline: .affineMatvecC1536R2048G128Rows4,
            scaledPipeline: .normScaleAffineMatvecC1536R2048G128Rows4,
            consumerInputIndex: 3,
            writesNormOutput: true
        ),
        CooperativeNormScaleRule(
            consumerPipeline: .affineMatvecC1536R12288G128Rows4,
            scaledPipeline: .normScaleAffineMatvecC1536R12288G128Rows4,
            consumerInputIndex: 3,
            writesNormOutput: true
        ),
        CooperativeNormScaleRule(
            consumerPipeline: .affineMatvecC1536R262144G128Rows4,
            scaledPipeline: .normScaleAffineMatvecC1536R262144G128Rows4,
            consumerInputIndex: 3,
            writesNormOutput: true
        ),
        CooperativeNormScaleRule(
            consumerPipeline: .affineMatvecC256R1024G128Rows4,
            scaledPipeline: .normScaleAffineMatvecC256R1024G128Rows4,
            consumerInputIndex: 3,
            writesNormOutput: true
        ),
        CooperativeNormScaleRule(
            consumerPipeline: .affineMatvecC256R2048G128Rows4,
            scaledPipeline: .normScaleAffineMatvecC256R2048G128Rows4,
            consumerInputIndex: 3,
            writesNormOutput: true
        ),
        CooperativeNormScaleRule(
            producerPipeline: .rmsNorm1PWD2560Add,
            consumerPipeline: .affineMatvecC2560R256G128Rows4,
            scaledPipeline: .normAddScaleAffineMatvecC2560R256G128Rows4,
            consumerInputIndex: 3,
            writesNormOutput: true,
            allowDecodeCooperativeScale: false,
            producerBufferIndicesBeforeOutput: [2]
        ),
    ]

    private func cooperativeNormScaleParameters(
        for norm: SmeltDispatch
    ) -> CooperativeNormParameters? {
        switch norm.pipeline {
        case .rmsNorm1PWD1024Batched:
            return CooperativeNormParameters(
                dim: "1024",
                eps: "1e-6",
                scalePipeline: .rmsNormScaleOnlyD1024Batched,
                scaleTgWidth: 128
            )
        case .rmsNorm1PWD2560Batched:
            return CooperativeNormParameters(
                dim: "2560",
                eps: "1e-6",
                scalePipeline: .rmsNormScaleOnlyD2560Batched,
                scaleTgWidth: 1024
            )
        case .rmsNorm1PWBatched where norm.constants.count >= 2
            && norm.constants[0].expression == "2048"
            && constantExpression(norm.constants[1].expression, equals: 1e-6):
            return CooperativeNormParameters(
                dim: "2048",
                eps: "1e-6",
                scalePipeline: .rmsNormScaleOnlyD2048Batched,
                scaleTgWidth: 1024
            )
        case .rmsNorm1PWBatched where norm.constants.count >= 2
            && norm.constants[0].expression == "2048"
            && constantExpression(norm.constants[1].expression, equals: 1e-5):
            return CooperativeNormParameters(
                dim: "2048",
                eps: "1e-5",
                scalePipeline: .rmsNormScaleOnlyD2048Eps1e5Batched,
                scaleTgWidth: 1024
            )
        case .rmsNorm1PWBatched where norm.constants.count >= 2
            && norm.constants[0].expression == "2560"
            && constantExpression(norm.constants[1].expression, equals: 1e-6):
            return CooperativeNormParameters(
                dim: "2560",
                eps: "1e-6",
                scalePipeline: .rmsNormScaleOnlyD2560Batched,
                scaleTgWidth: 1024
            )
        case .rmsNorm1PWBatched where norm.constants.count >= 2
            && norm.constants[0].expression == "3072"
            && constantExpression(norm.constants[1].expression, equals: 1e-5):
            return CooperativeNormParameters(
                dim: "3072",
                eps: "1e-5",
                scalePipeline: .rmsNormScaleOnlyD3072Eps1e5Batched,
                scaleTgWidth: 1024
            )
        case .rmsNorm1PW:
            guard norm.constants.count >= 2,
                  let dim = Int(norm.constants[0].expression),
                  Float(norm.constants[1].expression) != nil
            else { return nil }
            let scaleTgWidth = (min(max(dim, 32), 1_024) / 32) * 32
            return CooperativeNormParameters(
                dim: norm.constants[0].expression,
                eps: norm.constants[1].expression,
                scalePipeline: .rmsNormScaleOnly,
                scaleTgWidth: scaleTgWidth
            )
        case .rmsNorm1PWD1536:
            return CooperativeNormParameters(
                dim: "1536",
                eps: "1e-6",
                scalePipeline: .rmsNormScaleOnlyD1536,
                scaleTgWidth: 192
            )
        case .rmsNorm1PWD2560:
            return CooperativeNormParameters(
                dim: "2560",
                eps: "1e-6",
                scalePipeline: .rmsNormScaleOnlyD2560,
                scaleTgWidth: 320
            )
        case .rmsNorm1PWD2560Add:
            return CooperativeNormParameters(
                dim: "2560",
                eps: "1e-6",
                scalePipeline: .rmsNormScaleOnlyD2560,
                scaleTgWidth: 320,
                normOutputIndex: 3,
                requiresProducerSpecificRule: true
            )
        default:
            return nil
        }
    }

    private func constantExpression(_ expression: String, equals expected: Float) -> Bool {
        guard let value = Float(expression) else { return false }
        return value == expected
    }

    /// Keep a generic RMSNorm result in the graph-owned signed activation
    /// representation. The fused producer is selected solely from dispatch
    /// contracts, so binary and ternary consumers inherit it without a
    /// model-family route.
    private func normActivationViewRewrite(
        window: ArraySlice<SmeltIROp>
    ) -> SmeltFusionRewrite? {
        guard window.count >= 2,
              case .dispatch(let norm) = window[window.startIndex],
              norm.pipeline == .rmsNorm1PW,
              norm.pipelineNameOverride == nil,
              norm.buffers.count == 3,
              norm.constants.count == 2,
              let cols = Int(norm.constants[0].expression),
              cols.isMultiple(of: 128),
              Float(norm.constants[1].expression) != nil,
              case let .threadgroups(1, 1, 1, normTGWidth, 1, 1) = norm.dispatch,
              normTGWidth.isMultiple(of: 32),
              normTGWidth <= 1_024,
              norm.dynamicGridW == nil,
              norm.dynamicGridH == nil,
              norm.dynamicGridD == nil
        else { return nil }

        let viewIndex = window.index(after: window.startIndex)
        // This fusion deliberately elides the materialized norm output, so
        // trace modes that observe it retain the staged graph.
        if viewIndex < window.endIndex,
           case .traceMarker = window[viewIndex]
        {
            return nil
        }
        guard viewIndex < window.endIndex,
              case .dispatch(let view) = window[viewIndex],
              view.pipelineNameOverride == nil,
              view.buffers.count == 3,
              view.constants.count == 1,
              view.constants[0].expression == norm.constants[0].expression,
              sameBinding(norm.buffers[2], view.buffers[0]),
              norm.minSeqLen == view.minSeqLen,
              norm.maxSeqLenExclusive == view.maxSeqLenExclusive,
              norm.minPositionPlus1 == view.minPositionPlus1,
              norm.maxPositionPlus1Exclusive == view.maxPositionPlus1Exclusive,
              case let .threadgroups(viewWidth, 1, 1, 32, 1, 1) = view.dispatch,
              viewWidth == cols / 128,
              view.dynamicGridW == nil,
              view.dynamicGridH == nil,
              view.dynamicGridD == nil
        else { return nil }

        let scaledViewPipeline: SmeltPipeline
        switch view.pipeline {
        case .signedActivationBitplanesI2G128:
            scaledViewPipeline = .normScaleSignedActivationBitplanesI2G128
        case .signedActivationBitplanesI3G128:
            scaledViewPipeline = .normScaleSignedActivationBitplanesI3G128
        case .signedActivationBitplanesI4G128:
            scaledViewPipeline = .normScaleSignedActivationBitplanesI4G128
        case .signedActivationBitplanesI5G128:
            scaledViewPipeline = .normScaleSignedActivationBitplanesI5G128
        case .signedActivationBitplanesI6G128:
            scaledViewPipeline = .normScaleSignedActivationBitplanesI6G128
        default:
            return nil
        }

        let normScaleSlot = SmeltFixedSlot.normScaleScratch.rawValue
        let scale = SmeltDispatch(
            pipeline: .rmsNormScaleOnlyPrecise,
            buffers: [
                rebind(norm.buffers[0], index: 0),
                SmeltBufferBinding(slot: normScaleSlot, index: 1),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: norm.constants[0].expression,
                    type: norm.constants[0].type,
                    index: 2
                ),
                SmeltConstantBinding(
                    expression: norm.constants[1].expression,
                    type: norm.constants[1].type,
                    index: 3
                ),
            ],
            dispatch: norm.dispatch,
            comment: "Precise RMS norm scale",
            minSeqLen: norm.minSeqLen,
            maxSeqLenExclusive: norm.maxSeqLenExclusive,
            minPositionPlus1: norm.minPositionPlus1,
            maxPositionPlus1Exclusive: norm.maxPositionPlus1Exclusive
        )
        let scaledView = SmeltDispatch(
            pipeline: scaledViewPipeline,
            buffers: [
                SmeltBufferBinding(slot: normScaleSlot, index: 0),
                rebind(norm.buffers[0], index: 1),
                rebind(norm.buffers[1], index: 2),
                rebind(view.buffers[1], index: 3),
                rebind(view.buffers[2], index: 4),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: view.constants[0].expression,
                    type: view.constants[0].type,
                    index: 5
                ),
            ],
            dispatch: view.dispatch,
            comment: "Norm scale -> signed activation view",
            minSeqLen: view.minSeqLen,
            maxSeqLenExclusive: view.maxSeqLenExclusive,
            minPositionPlus1: view.minPositionPlus1,
            maxPositionPlus1Exclusive: view.maxPositionPlus1Exclusive
        )
        return SmeltFusionRewrite(
            consumedOpCount: window.distance(
                from: window.startIndex,
                to: window.index(after: viewIndex)),
            producedOps: [.dispatch(scale), .dispatch(scaledView)],
            kind: .genericKernel,
            rule: .normActivationView
        )
    }

    /// Fold the residual boundary into the precise scale producer selected by
    /// the graph-owned norm activation-view route. The rewrite is stated only
    /// in terms of dispatch/buffer contracts, so every weight family inherits
    /// it without changing its projection kernel.
    private func residualAddNormActivationViewRewrite(
        window: ArraySlice<SmeltIROp>
    ) -> SmeltFusionRewrite? {
        guard window.count >= 3,
              case .dispatch(let add) = window[window.startIndex],
              add.pipeline == .elementwiseAdd,
              add.pipelineNameOverride == nil,
              add.buffers.count == 3,
              add.constants.count == 1,
              let cols = Int(add.constants[0].expression),
              case let .threads(addWidth, 1, 1, _, 1, 1) = add.dispatch,
              addWidth == cols,
              add.dynamicGridW == nil,
              add.dynamicGridH == nil,
              add.dynamicGridD == nil
        else { return nil }

        var normIndex = window.index(after: window.startIndex)
        var preservedSwap = false
        if normIndex < window.endIndex,
           case .swap = window[normIndex]
        {
            preservedSwap = true
            normIndex = window.index(after: normIndex)
        }
        guard normIndex < window.endIndex else { return nil }
        guard case .dispatch(let norm) = window[normIndex],
              norm.pipeline == .rmsNorm1PW,
              norm.pipelineNameOverride == nil,
              norm.buffers.count == 3,
              norm.constants.count == 2,
              norm.constants[0].expression == add.constants[0].expression,
              Float(norm.constants[1].expression) != nil,
              (preservedSwap
                  ? sameBindingAcrossSwap(add.buffers[2], norm.buffers[0])
                  : sameBinding(add.buffers[2], norm.buffers[0])),
              case let .threadgroups(1, 1, 1, normTGWidth, 1, 1) = norm.dispatch,
              normTGWidth.isMultiple(of: 32),
              normTGWidth <= 1_024,
              norm.dynamicGridW == nil,
              norm.dynamicGridH == nil,
              norm.dynamicGridD == nil,
              add.minSeqLen == norm.minSeqLen,
              add.maxSeqLenExclusive == norm.maxSeqLenExclusive,
              add.minPositionPlus1 == norm.minPositionPlus1,
              add.maxPositionPlus1Exclusive == norm.maxPositionPlus1Exclusive
        else { return nil }

        let viewIndex = window.index(after: normIndex)
        guard viewIndex < window.endIndex,
              case .dispatch(let view) = window[viewIndex],
              view.pipelineNameOverride == nil,
              view.buffers.count == 3,
              view.constants.count == 1,
              view.constants[0].expression == norm.constants[0].expression,
              sameBinding(norm.buffers[2], view.buffers[0]),
              norm.minSeqLen == view.minSeqLen,
              norm.maxSeqLenExclusive == view.maxSeqLenExclusive,
              norm.minPositionPlus1 == view.minPositionPlus1,
              norm.maxPositionPlus1Exclusive == view.maxPositionPlus1Exclusive,
              case let .threadgroups(viewWidth, 1, 1, 32, 1, 1) = view.dispatch,
              viewWidth == cols / 128,
              view.dynamicGridW == nil,
              view.dynamicGridH == nil,
              view.dynamicGridD == nil
        else { return nil }

        let scaledViewPipeline: SmeltPipeline
        switch view.pipeline {
        case .signedActivationBitplanesI2G128:
            scaledViewPipeline = .normScaleSignedActivationBitplanesI2G128
        case .signedActivationBitplanesI3G128:
            scaledViewPipeline = .normScaleSignedActivationBitplanesI3G128
        case .signedActivationBitplanesI4G128:
            scaledViewPipeline = .normScaleSignedActivationBitplanesI4G128
        case .signedActivationBitplanesI5G128:
            scaledViewPipeline = .normScaleSignedActivationBitplanesI5G128
        case .signedActivationBitplanesI6G128:
            scaledViewPipeline = .normScaleSignedActivationBitplanesI6G128
        default:
            return nil
        }

        let normScaleSlot = SmeltFixedSlot.normScaleScratch.rawValue
        let addScale = SmeltDispatch(
            pipeline: .residualAddRMSNormScaleOnlyPrecise,
            buffers: [
                rebind(add.buffers[0], index: 0),
                rebind(add.buffers[1], index: 1),
                rebind(add.buffers[2], index: 2),
                SmeltBufferBinding(slot: normScaleSlot, index: 3),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: norm.constants[0].expression,
                    type: norm.constants[0].type,
                    index: 4
                ),
                SmeltConstantBinding(
                    expression: norm.constants[1].expression,
                    type: norm.constants[1].type,
                    index: 5
                ),
            ],
            dispatch: norm.dispatch,
            comment: "Residual add -> precise RMS norm scale",
            minSeqLen: norm.minSeqLen,
            maxSeqLenExclusive: norm.maxSeqLenExclusive,
            minPositionPlus1: norm.minPositionPlus1,
            maxPositionPlus1Exclusive: norm.maxPositionPlus1Exclusive
        )
        let scaledView = SmeltDispatch(
            pipeline: scaledViewPipeline,
            buffers: [
                SmeltBufferBinding(slot: normScaleSlot, index: 0),
                rebind(norm.buffers[0], index: 1),
                rebind(norm.buffers[1], index: 2),
                rebind(view.buffers[1], index: 3),
                rebind(view.buffers[2], index: 4),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: view.constants[0].expression,
                    type: view.constants[0].type,
                    index: 5
                ),
            ],
            dispatch: view.dispatch,
            comment: "Norm scale -> signed activation view",
            minSeqLen: view.minSeqLen,
            maxSeqLenExclusive: view.maxSeqLenExclusive,
            minPositionPlus1: view.minPositionPlus1,
            maxPositionPlus1Exclusive: view.maxPositionPlus1Exclusive
        )
        let producedOps: [SmeltIROp] = preservedSwap
            ? [.dispatch(addScale), .swap, .dispatch(scaledView)]
            : [.dispatch(addScale), .dispatch(scaledView)]
        return SmeltFusionRewrite(
            consumedOpCount: window.distance(
                from: window.startIndex,
                to: window.index(after: viewIndex)
            ),
            producedOps: producedOps,
            kind: .genericKernel,
            rule: .residualAddNormActivationView
        )
    }

    /// Elide a materialized gated-norm output when its declared consumer is an
    /// exactly compatible activation representation.  This rule is stated only
    /// in terms of producer/consumer contracts and launch shapes; any model that
    /// emits the same graph inherits it.
    private func gatedNormActivationViewRewrite(
        window: ArraySlice<SmeltIROp>
    ) -> SmeltFusionRewrite? {
        guard window.count >= 2,
              case .dispatch(let norm) = window[window.startIndex],
              norm.pipeline == .rmsNormGated || norm.pipeline == .rmsNormGatedD128,
              norm.pipelineNameOverride == nil,
              norm.buffers.count == 4,
              norm.dynamicGridW == nil,
              norm.dynamicGridH == nil,
              norm.dynamicGridD == nil
        else {
            return nil
        }
        let headDim: String
        let eps: String
        switch norm.pipeline {
        case .rmsNormGated:
            guard norm.constants.count == 2,
                  norm.constants[0].expression == "128",
                  constantExpression(norm.constants[1].expression, equals: 1e-6)
            else { return nil }
            headDim = norm.constants[0].expression
            eps = norm.constants[1].expression
        case .rmsNormGatedD128:
            guard norm.constants.isEmpty else { return nil }
            headDim = "128"
            eps = "1e-6"
        default:
            return nil
        }

        var viewIndex = window.index(after: window.startIndex)
        var deferredMarkers: [SmeltIROp] = []
        while viewIndex < window.endIndex {
            guard case .traceMarker = window[viewIndex] else { break }
            deferredMarkers.append(window[viewIndex])
            viewIndex = window.index(after: viewIndex)
        }
        guard viewIndex < window.endIndex,
              case .dispatch(let view) = window[viewIndex],
              view.pipeline == .signedActivationBitplanesI6G128,
              view.pipelineNameOverride == nil,
              view.buffers.count == 3,
              view.constants.count == 1,
              sameBinding(norm.buffers[3], view.buffers[0]),
              norm.minSeqLen == view.minSeqLen,
              norm.maxSeqLenExclusive == view.maxSeqLenExclusive,
              norm.minPositionPlus1 == view.minPositionPlus1,
              norm.maxPositionPlus1Exclusive == view.maxPositionPlus1Exclusive,
              case let .threadgroups(normWidth, 1, 1, normTGWidth, 1, 1) = norm.dispatch,
              (normTGWidth == 32 || normTGWidth == 128),
              case let .threadgroups(viewWidth, 1, 1, 32, 1, 1) = view.dispatch,
              let cols = Int(view.constants[0].expression),
              cols == normWidth * 128,
              viewWidth == normWidth,
              view.dynamicGridW == nil,
              view.dynamicGridH == nil,
              view.dynamicGridD == nil
        else { return nil }

        if !deferredMarkers.isEmpty {
            guard case .fixed(let outputSlot) = norm.buffers[3].slot,
                  deferredMarkers.allSatisfy({ marker in
                      guard case .traceMarker(_, let markerSlot) = marker else {
                          return false
                      }
                      return markerSlot == outputSlot
                  })
            else { return nil }
        }

        let fusedBuffers = [
            rebind(norm.buffers[0], index: 0),
            rebind(norm.buffers[1], index: 1),
            rebind(norm.buffers[2], index: 2),
            rebind(view.buffers[1], index: 3),
            rebind(view.buffers[2], index: 4),
        ]
        let fused = SmeltDispatch(
            pipeline: .rmsNormGatedD128SignedActivationBitplanesI6G128,
            buffers: fusedBuffers,
            constants: [
                SmeltConstantBinding(
                    expression: headDim,
                    type: .uint32,
                    index: 5
                ),
                SmeltConstantBinding(
                    expression: eps,
                    type: .float32,
                    index: 6
                ),
            ],
            dispatch: norm.dispatch,
            comment: "Gated RMS norm -> signed i6/g128 activation view",
            minSeqLen: norm.minSeqLen,
            maxSeqLenExclusive: norm.maxSeqLenExclusive,
            minPositionPlus1: norm.minPositionPlus1,
            maxPositionPlus1Exclusive: norm.maxPositionPlus1Exclusive
        )
        var producedOps: [SmeltIROp] = [.dispatch(fused)]
        // Trace mode must audit the exact production kernel above. Replay the
        // original producer only to rematerialize the eliminated observed edge.
        if !deferredMarkers.isEmpty {
            producedOps.append(.dispatch(norm))
        }
        producedOps.append(contentsOf: deferredMarkers)
        return SmeltFusionRewrite(
            consumedOpCount: window.distance(
                from: window.startIndex,
                to: window.index(after: viewIndex)
            ),
            producedOps: producedOps,
            kind: .specializedKernel,
            rule: .gatedNormActivationView
        )
    }

    /// Fuse an elementwise attention gate directly into its declared signed
    /// activation representation. Selection is based on the graph edge and
    /// representation contract, so any compatible model family inherits it.
    private func sigmoidMulActivationViewRewrite(
        window: ArraySlice<SmeltIROp>
    ) -> SmeltFusionRewrite? {
        guard window.count >= 2,
              case .dispatch(let producer) = window[window.startIndex],
              producer.pipeline == .sigmoidMul,
              producer.pipelineNameOverride == nil,
              producer.buffers.count == 3,
              producer.constants.count == 1,
              producer.dynamicGridW == nil,
              producer.dynamicGridH == nil,
              producer.dynamicGridD == nil,
              case let .threads(producerWidth, 1, 1, _, 1, 1) = producer.dispatch,
              let cols = Int(producer.constants[0].expression),
              cols == producerWidth,
              cols > 0,
              cols.isMultiple(of: 128)
        else { return nil }

        var viewIndex = window.index(after: window.startIndex)
        var deferredMarkers: [SmeltIROp] = []
        while viewIndex < window.endIndex {
            guard case .traceMarker = window[viewIndex] else { break }
            deferredMarkers.append(window[viewIndex])
            viewIndex = window.index(after: viewIndex)
        }
        guard viewIndex < window.endIndex,
              case .dispatch(let view) = window[viewIndex],
              view.pipeline == .signedActivationBitplanesI6G128,
              view.pipelineNameOverride == nil,
              view.buffers.count == 3,
              view.constants.count == 1,
              sameBinding(producer.buffers[2], view.buffers[0]),
              producer.minSeqLen == view.minSeqLen,
              producer.maxSeqLenExclusive == view.maxSeqLenExclusive,
              producer.minPositionPlus1 == view.minPositionPlus1,
              producer.maxPositionPlus1Exclusive == view.maxPositionPlus1Exclusive,
              case let .threadgroups(viewWidth, 1, 1, 32, 1, 1) = view.dispatch,
              view.constants[0].expression == producer.constants[0].expression,
              viewWidth == cols / 128,
              view.dynamicGridW == nil,
              view.dynamicGridH == nil,
              view.dynamicGridD == nil
        else { return nil }

        if !deferredMarkers.isEmpty {
            guard case .fixed(let outputSlot) = producer.buffers[2].slot,
                  deferredMarkers.allSatisfy({ marker in
                      guard case .traceMarker(_, let markerSlot) = marker else {
                          return false
                      }
                      return markerSlot == outputSlot
                  })
            else { return nil }
        }

        let fused = SmeltDispatch(
            pipeline: .sigmoidMulSignedActivationBitplanesI6G128,
            buffers: [
                rebind(producer.buffers[0], index: 0),
                rebind(producer.buffers[1], index: 1),
                rebind(view.buffers[1], index: 2),
                rebind(view.buffers[2], index: 3),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: producer.constants[0].expression,
                    type: producer.constants[0].type,
                    index: 4
                ),
            ],
            dispatch: view.dispatch,
            comment: "Sigmoid multiply -> signed i6/g128 activation view",
            minSeqLen: producer.minSeqLen,
            maxSeqLenExclusive: producer.maxSeqLenExclusive,
            minPositionPlus1: producer.minPositionPlus1,
            maxPositionPlus1Exclusive: producer.maxPositionPlus1Exclusive
        )
        var producedOps: [SmeltIROp] = [.dispatch(fused)]
        // Audit the exact production producer, replaying only the eliminated
        // materialized edge when a full trace observes it.
        if !deferredMarkers.isEmpty {
            producedOps.append(.dispatch(producer))
        }
        producedOps.append(contentsOf: deferredMarkers)
        return SmeltFusionRewrite(
            consumedOpCount: window.distance(
                from: window.startIndex,
                to: window.index(after: viewIndex)
            ),
            producedOps: producedOps,
            kind: .specializedKernel,
            rule: .sigmoidMulActivationView
        )
    }

    /// Elide a materialized SwiGLU vector when the graph immediately consumes
    /// it as an i5/g128 signed activation representation. This rule is weight-
    /// family agnostic: binary and ternary projections inherit it through the
    /// same CAM edge and activation-view contract.
    private func swigluActivationViewRewrite(
        window: ArraySlice<SmeltIROp>
    ) -> SmeltFusionRewrite? {
        guard window.count >= 2,
              case .dispatch(let producer) = window[window.startIndex],
              producer.pipeline == .swigluFused,
              producer.pipelineNameOverride == nil,
              producer.buffers.count == 3,
              producer.constants.count == 1,
              producer.dynamicGridW == nil,
              producer.dynamicGridH == nil,
              producer.dynamicGridD == nil,
              case let .threads(producerWidth, 1, 1, _, 1, 1) = producer.dispatch,
              let cols = Int(producer.constants[0].expression),
              cols == producerWidth,
              cols > 0,
              cols.isMultiple(of: 128)
        else { return nil }

        var viewIndex = window.index(after: window.startIndex)
        var deferredMarkers: [SmeltIROp] = []
        while viewIndex < window.endIndex {
            guard case .traceMarker = window[viewIndex] else { break }
            deferredMarkers.append(window[viewIndex])
            viewIndex = window.index(after: viewIndex)
        }
        guard viewIndex < window.endIndex,
              case .dispatch(let view) = window[viewIndex],
              view.pipeline == .signedActivationBitplanesI5G128,
              view.pipelineNameOverride == nil,
              view.buffers.count == 3,
              view.constants.count == 1,
              sameBinding(producer.buffers[2], view.buffers[0]),
              producer.minSeqLen == view.minSeqLen,
              producer.maxSeqLenExclusive == view.maxSeqLenExclusive,
              producer.minPositionPlus1 == view.minPositionPlus1,
              producer.maxPositionPlus1Exclusive == view.maxPositionPlus1Exclusive,
              case let .threadgroups(viewWidth, 1, 1, 32, 1, 1) = view.dispatch,
              view.constants[0].expression == producer.constants[0].expression,
              viewWidth == cols / 128,
              view.dynamicGridW == nil,
              view.dynamicGridH == nil,
              view.dynamicGridD == nil
        else { return nil }

        if !deferredMarkers.isEmpty {
            guard case .fixed(let outputSlot) = producer.buffers[2].slot,
                  deferredMarkers.allSatisfy({ marker in
                      guard case .traceMarker(_, let markerSlot) = marker else {
                          return false
                      }
                      return markerSlot == outputSlot
                  })
            else { return nil }
        }

        let fused = SmeltDispatch(
            pipeline: .swigluSignedActivationBitplanesI5G128,
            buffers: [
                rebind(producer.buffers[0], index: 0),
                rebind(producer.buffers[1], index: 1),
                rebind(view.buffers[1], index: 2),
                rebind(view.buffers[2], index: 3),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: producer.constants[0].expression,
                    type: producer.constants[0].type,
                    index: 4),
            ],
            dispatch: view.dispatch,
            comment: "SwiGLU -> signed i5/g128 activation view",
            minSeqLen: producer.minSeqLen,
            maxSeqLenExclusive: producer.maxSeqLenExclusive,
            minPositionPlus1: producer.minPositionPlus1,
            maxPositionPlus1Exclusive: producer.maxPositionPlus1Exclusive
        )
        var producedOps: [SmeltIROp] = [.dispatch(fused)]
        if !deferredMarkers.isEmpty {
            producedOps.append(.dispatch(producer))
        }
        producedOps.append(contentsOf: deferredMarkers)
        return SmeltFusionRewrite(
            consumedOpCount: window.distance(
                from: window.startIndex,
                to: window.index(after: viewIndex)),
            producedOps: producedOps,
            kind: .specializedKernel,
            rule: .swigluActivationView
        )
    }

    private func cooperativeNormScaleConsumerRewrite(
        window: ArraySlice<SmeltIROp>,
        respectDecodeCostModel: Bool = true
    ) -> SmeltFusionRewrite? {
        guard window.count >= 2,
              case .dispatch(let norm) = window[window.startIndex],
              let normParams = cooperativeNormScaleParameters(for: norm)
        else {
            return nil
        }

        var consumerIndex = window.index(after: window.startIndex)
        var deferredMarkers: [SmeltIROp] = []
        while consumerIndex < window.endIndex {
            guard case .traceMarker = window[consumerIndex] else { break }
            deferredMarkers.append(window[consumerIndex])
            consumerIndex = window.index(after: consumerIndex)
        }

        guard consumerIndex < window.endIndex,
              case .dispatch(let consumer) = window[consumerIndex]
        else {
            return nil
        }

        guard norm.buffers.indices.contains(0),
              norm.buffers.indices.contains(1),
              norm.buffers.indices.contains(normParams.normOutputIndex)
        else {
            return nil
        }

        let normInput = norm.buffers[0]
        let normWeight = norm.buffers[1]
        let normOutput = norm.buffers[normParams.normOutputIndex]

        guard let rule = Self.cooperativeNormScaleRules.first(where: {
            $0.consumerPipeline == consumer.pipeline
                && ($0.producerPipeline == nil || $0.producerPipeline == norm.pipeline)
                && (!normParams.requiresProducerSpecificRule || $0.producerPipeline == norm.pipeline)
        }),
              consumer.buffers.indices.contains(rule.consumerInputIndex),
              rule.producerBufferIndicesBeforeOutput.allSatisfy({ norm.buffers.indices.contains($0) }),
              sameBinding(consumer.buffers[rule.consumerInputIndex], normOutput),
              norm.minSeqLen == consumer.minSeqLen,
              norm.maxSeqLenExclusive == consumer.maxSeqLenExclusive,
              norm.minPositionPlus1 == consumer.minPositionPlus1,
              norm.maxPositionPlus1Exclusive == consumer.maxPositionPlus1Exclusive
        else {
            return nil
        }

        // A trace marker over the original norm output must still observe a
        // materialized buffer. Output-eliding producer fusion is therefore a
        // stripped-path optimization only.
        guard rule.writesNormOutput || deferredMarkers.isEmpty else {
            return nil
        }

        if respectDecodeCostModel,
           !rule.allowDecodeCooperativeScale,
           isDecodeScaleOnlyPipeline(normParams.scalePipeline)
        {
            return nil
        }

        let normScaleSlot = SmeltFixedSlot.normScaleScratch.rawValue
        let scaleDispatch = SmeltDispatch(
            pipeline: normParams.scalePipeline,
            buffers: [
                rebind(normInput, index: 0),
                SmeltBufferBinding(slot: normScaleSlot, index: 1),
            ],
            constants: normParams.scalePipeline == .rmsNormScaleOnly ? [
                SmeltConstantBinding(
                    expression: normParams.dim,
                    type: .uint32,
                    index: 2
                ),
                SmeltConstantBinding(
                    expression: normParams.eps,
                    type: .float32,
                    index: 3
                ),
            ] : [],
            dispatch: .threadgroups(
                width: 1,
                height: 1,
                depth: 1,
                tgWidth: normParams.scaleTgWidth,
                tgHeight: 1,
                tgDepth: 1
            ),
            comment: "RMS norm scale only",
            dynamicGridW: norm.dynamicGridW,
            dynamicGridH: norm.dynamicGridH,
            dynamicGridD: norm.dynamicGridD,
            minSeqLen: norm.minSeqLen,
            maxSeqLenExclusive: norm.maxSeqLenExclusive,
            minPositionPlus1: norm.minPositionPlus1,
            maxPositionPlus1Exclusive: norm.maxPositionPlus1Exclusive
        )

        var scaledBuffers: [SmeltBufferBinding] = [
            SmeltBufferBinding(slot: normScaleSlot, index: 0),
            rebind(normInput, index: 1),
            rebind(normWeight, index: 2),
        ]

        var nextIndex = 3
        for bufferIndex in rule.producerBufferIndicesBeforeOutput {
            scaledBuffers.append(rebind(norm.buffers[bufferIndex], index: nextIndex))
            nextIndex += 1
        }

        if rule.writesNormOutput {
            scaledBuffers.append(rebind(normOutput, index: nextIndex))
            nextIndex += 1
        }

        for (consumerBufferIndex, buffer) in consumer.buffers.enumerated() {
            if consumerBufferIndex == rule.consumerInputIndex { continue }
            scaledBuffers.append(rebind(buffer, index: nextIndex))
            nextIndex += 1
        }

        let scaledConstants = consumer.constants.map { constant in
            SmeltConstantBinding(
                expression: constant.expression,
                type: constant.type,
                index: nextIndex
            )
        }

        let scaledConsumer = SmeltDispatch(
            pipeline: rule.scaledPipeline,
            buffers: scaledBuffers,
            constants: scaledConstants,
            dispatch: consumer.dispatch,
            comment: "Norm-scaled " + (consumer.comment ?? "consumer"),
            fcCols: consumer.fcCols,
            fcGroupSize: consumer.fcGroupSize,
            dynamicGridW: consumer.dynamicGridW,
            dynamicGridH: consumer.dynamicGridH,
            dynamicGridD: consumer.dynamicGridD,
            minSeqLen: consumer.minSeqLen,
            maxSeqLenExclusive: consumer.maxSeqLenExclusive,
            minPositionPlus1: consumer.minPositionPlus1,
            maxPositionPlus1Exclusive: consumer.maxPositionPlus1Exclusive
        )

        var producedOps: [SmeltIROp] = [
            .dispatch(scaleDispatch),
            .dispatch(scaledConsumer),
        ]
        producedOps.append(contentsOf: deferredMarkers)

        return SmeltFusionRewrite(
            consumedOpCount: window.distance(
                from: window.startIndex,
                to: window.index(after: consumerIndex)
            ),
            producedOps: producedOps,
            kind: .specializedKernel,
            rule: .cooperativeNormScaleConsumer
        )
    }

    private func isDecodeScaleOnlyPipeline(_ pipeline: SmeltPipeline) -> Bool {
        switch pipeline {
        case .rmsNormScaleOnly,
             .rmsNormScaleOnlyD1536,
             .rmsNormScaleOnlyD2560:
            return true
        default:
            return false
        }
    }

    private func fusedNormRopeKVPrefillRewrite(
        window: ArraySlice<SmeltIROp>
    ) -> SmeltFusionRewrite? {
        guard window.count >= 2,
              case .dispatch(let norm) = window[window.startIndex]
        else {
            return nil
        }

        var ropeIndex = window.index(after: window.startIndex)
        var deferredMarkers: [SmeltIROp] = []
        while ropeIndex < window.endIndex {
            guard case .traceMarker = window[ropeIndex] else { break }
            deferredMarkers.append(window[ropeIndex])
            ropeIndex = window.index(after: ropeIndex)
        }

        guard ropeIndex < window.endIndex,
              case .dispatch(let rope) = window[ropeIndex],
              rope.pipeline == .ropeAndKvCachePrefill,
              norm.constants.count == 3,
              rope.buffers.count == 7,
              rope.constants.count == 8,
              norm.dynamicGridW == nil,
              norm.dynamicGridH == .seqLen,
              norm.dynamicGridD == nil,
              rope.dynamicGridW == .seqLen,
              rope.dynamicGridH == nil,
              rope.dynamicGridD == nil,
              norm.minSeqLen == rope.minSeqLen,
              norm.maxSeqLenExclusive == rope.maxSeqLenExclusive,
              norm.minPositionPlus1 == rope.minPositionPlus1,
              norm.maxPositionPlus1Exclusive == rope.maxPositionPlus1Exclusive
        else {
            return nil
        }

        let fusedBuffers: [SmeltBufferBinding]
        let commentPrefix: String
        switch norm.pipeline {
        case .perHeadRmsNormNoScaleBatched:
            guard norm.buffers.count == 1,
                  sameBinding(norm.buffers[0], rope.buffers[2]),
                  norm.constants[0].expression == rope.constants[3].expression,
                  norm.constants[1].expression == rope.constants[0].expression
            else {
                return nil
            }
            fusedBuffers = rope.buffers
            commentPrefix = "Fused V norm + "

        case .perHeadRmsNormBatched:
            guard norm.buffers.count == 3,
                  sameBinding(norm.buffers[2], rope.buffers[0]),
                  norm.constants[0].expression == rope.constants[2].expression,
                  norm.constants[1].expression == rope.constants[0].expression,
                  rope.constants[3].expression == "0"
            else {
                return nil
            }
            var buffers = rope.buffers
            buffers[1] = rebind(norm.buffers[0], index: 1)
            buffers[2] = rebind(norm.buffers[1], index: 2)
            fusedBuffers = buffers
            commentPrefix = "Fused Q norm + "

        default:
            return nil
        }

        let eps = norm.constants[2]
        let fusedConstants = rope.constants + [
            SmeltConstantBinding(
                expression: eps.expression,
                type: eps.type,
                index: 15
            ),
        ]

        let fused = SmeltDispatch(
            pipeline: .fusedNormRopeAndKvCachePrefill,
            buffers: fusedBuffers,
            constants: fusedConstants,
            dispatch: rope.dispatch,
            comment: commentPrefix + (rope.comment ?? "RoPE + KV cache"),
            dynamicGridW: rope.dynamicGridW,
            dynamicGridH: rope.dynamicGridH,
            dynamicGridD: rope.dynamicGridD,
            minSeqLen: rope.minSeqLen,
            maxSeqLenExclusive: rope.maxSeqLenExclusive,
            minPositionPlus1: rope.minPositionPlus1,
            maxPositionPlus1Exclusive: rope.maxPositionPlus1Exclusive
        )

        var producedOps: [SmeltIROp] = [.dispatch(fused)]
        producedOps.append(contentsOf: deferredMarkers)

        return SmeltFusionRewrite(
            consumedOpCount: window.distance(
                from: window.startIndex,
                to: window.index(after: ropeIndex)
            ),
            producedOps: producedOps,
            kind: .specializedKernel,
            rule: .fusedNormRopeKVPrefill
        )
    }

    private func rmsNormResidualAddRewrite(
        window: ArraySlice<SmeltIROp>
    ) -> SmeltFusionRewrite? {
        guard window.count >= 2,
              case .dispatch(let norm) = window[window.startIndex],
              case .dispatch(let add) = window[window.index(after: window.startIndex)],
              norm.buffers.count == 3,
              add.pipeline == .elementwiseAdd,
              add.buffers.count == 3,
              add.constants.count == 1,
              norm.minSeqLen == add.minSeqLen,
              norm.maxSeqLenExclusive == add.maxSeqLenExclusive,
              norm.minPositionPlus1 == add.minPositionPlus1,
              norm.maxPositionPlus1Exclusive == add.maxPositionPlus1Exclusive
        else {
            return nil
        }

        let fusedPipeline: SmeltPipeline
        let dispatch: SmeltDispatchStyle
        let dynamicGridW: SmeltDynamicGridDimension?
        let dynamicGridH: SmeltDynamicGridDimension?
        let dynamicGridD: SmeltDynamicGridDimension?
        switch norm.pipeline {
        case .rmsNorm1PW
            where norm.constants.count == 2
                && norm.constants[0].expression == "256"
                && constantExpression(norm.constants[1].expression, equals: 1e-6)
                && add.constants[0].expression == "256"
                && norm.dynamicGridW == nil
                && norm.dynamicGridH == nil
                && norm.dynamicGridD == nil
                && add.dynamicGridW == nil
                && add.dynamicGridH == nil
                && add.dynamicGridD == nil:
            fusedPipeline = .rmsNorm1PWD256Add
            dispatch = .threadgroups(
                width: 1,
                height: 1,
                depth: 1,
                tgWidth: 256,
                tgHeight: 1,
                tgDepth: 1
            )
            dynamicGridW = nil
            dynamicGridH = nil
            dynamicGridD = nil
        case .rmsNorm1PWD1536
            where norm.constants.isEmpty
                && add.constants[0].expression == "1536"
                && norm.dynamicGridW == nil
                && norm.dynamicGridH == nil
                && norm.dynamicGridD == nil
                && add.dynamicGridW == nil
                && add.dynamicGridH == nil
                && add.dynamicGridD == nil:
            fusedPipeline = .rmsNorm1PWD1536Add
            dispatch = .threadgroups(
                width: 1,
                height: 1,
                depth: 1,
                tgWidth: 192,
                tgHeight: 1,
                tgDepth: 1
            )
            dynamicGridW = nil
            dynamicGridH = nil
            dynamicGridD = nil
        case .rmsNorm1PWD2560
            where norm.constants.isEmpty
                && add.constants[0].expression == "2560"
                && norm.dynamicGridW == nil
                && norm.dynamicGridH == nil
                && norm.dynamicGridD == nil
                && add.dynamicGridW == nil
                && add.dynamicGridH == nil
                && add.dynamicGridD == nil:
            fusedPipeline = .rmsNorm1PWD2560Add
            dispatch = .threadgroups(
                width: 1,
                height: 1,
                depth: 1,
                tgWidth: 320,
                tgHeight: 1,
                tgDepth: 1
            )
            dynamicGridW = nil
            dynamicGridH = nil
            dynamicGridD = nil
        case .rmsNorm1PWD1536Batched
            where norm.constants.isEmpty
                && add.constants[0].expression == "__seqLen__*1536"
                && norm.dynamicGridW == .seqLen
                && norm.dynamicGridH == nil
                && norm.dynamicGridD == nil
                && add.dynamicGridW == .seqLenMul(1536)
                && add.dynamicGridH == nil
                && add.dynamicGridD == nil:
            fusedPipeline = .rmsNorm1PWD1536AddBatched
            dispatch = norm.dispatch
            dynamicGridW = norm.dynamicGridW
            dynamicGridH = norm.dynamicGridH
            dynamicGridD = norm.dynamicGridD
        case .rmsNorm1PWD2560Batched
            where norm.constants.isEmpty
                && add.constants[0].expression == "__seqLen__*2560"
                && norm.dynamicGridW == .seqLen
                && norm.dynamicGridH == nil
                && norm.dynamicGridD == nil
                && add.dynamicGridW == .seqLenMul(2560)
                && add.dynamicGridH == nil
                && add.dynamicGridD == nil:
            fusedPipeline = .rmsNorm1PWD2560AddBatched
            dispatch = norm.dispatch
            dynamicGridW = norm.dynamicGridW
            dynamicGridH = norm.dynamicGridH
            dynamicGridD = norm.dynamicGridD
        default:
            return nil
        }

        let normInput = norm.buffers[0]
        let normWeight = norm.buffers[1]
        let normOutput = norm.buffers[2]
        let addInputA = add.buffers[0]
        let addInputB = add.buffers[1]
        let addOutput = add.buffers[2]

        let addAMatchesNormOutput = sameBinding(addInputA, normOutput)
        let addBMatchesNormOutput = sameBinding(addInputB, normOutput)
        let residual: SmeltBufferBinding
        if addAMatchesNormOutput && !addBMatchesNormOutput {
            residual = addInputB
        } else if addBMatchesNormOutput && !addAMatchesNormOutput {
            residual = addInputA
        } else {
            return nil
        }

        guard [normInput, normWeight, residual, addOutput].allSatisfy({
            $0.offsetExpression == nil && $0.offsetKind == 0
        }) else {
            return nil
        }

        let fused = SmeltDispatch(
            pipeline: fusedPipeline,
            buffers: [
                rebind(normInput, index: 0),
                rebind(normWeight, index: 1),
                rebind(residual, index: 2),
                rebind(addOutput, index: 3),
            ],
            constants: [],
            dispatch: dispatch,
            comment: (norm.comment ?? "RMS norm") + " + residual (fused)",
            dynamicGridW: dynamicGridW,
            dynamicGridH: dynamicGridH,
            dynamicGridD: dynamicGridD,
            minSeqLen: norm.minSeqLen,
            maxSeqLenExclusive: norm.maxSeqLenExclusive,
            minPositionPlus1: norm.minPositionPlus1,
            maxPositionPlus1Exclusive: norm.maxPositionPlus1Exclusive
        )

        return SmeltFusionRewrite(
            consumedOpCount: 2,
            producedOps: [.dispatch(fused)],
            kind: .specializedKernel,
            rule: .rmsNormResidualAdd
        )
    }

    private func rmsNormResidualAddScalarWeightRewrite(
        window: ArraySlice<SmeltIROp>
    ) -> SmeltFusionRewrite? {
        guard window.count >= 2,
              case .dispatch(let normAdd) = window[window.startIndex],
              case .dispatch(let scalarMul) = window[window.index(after: window.startIndex)],
              normAdd.pipeline == .rmsNorm1PWD256Add,
              normAdd.buffers.count == 4,
              normAdd.constants.isEmpty,
              scalarMul.pipeline == .scalarMulWeight,
              scalarMul.buffers.count == 3,
              scalarMul.constants.count == 1,
              scalarMul.constants[0].expression == "256",
              sameBinding(scalarMul.buffers[0], normAdd.buffers[3]),
              normAdd.dynamicGridW == nil,
              normAdd.dynamicGridH == nil,
              normAdd.dynamicGridD == nil,
              scalarMul.dynamicGridW == nil,
              scalarMul.dynamicGridH == nil,
              scalarMul.dynamicGridD == nil,
              normAdd.minSeqLen == scalarMul.minSeqLen,
              normAdd.maxSeqLenExclusive == scalarMul.maxSeqLenExclusive,
              normAdd.minPositionPlus1 == scalarMul.minPositionPlus1,
              normAdd.maxPositionPlus1Exclusive == scalarMul.maxPositionPlus1Exclusive
        else {
            return nil
        }

        let input = normAdd.buffers[0]
        let normWeight = normAdd.buffers[1]
        let residual = normAdd.buffers[2]
        let scalarWeight = scalarMul.buffers[1]
        let output = scalarMul.buffers[2]

        guard [input, normWeight, residual, scalarWeight, output].allSatisfy({
            $0.offsetExpression == nil && $0.offsetKind == 0
        }) else {
            return nil
        }

        let fused = SmeltDispatch(
            pipeline: .rmsNorm1PWD256AddScalarWeight,
            buffers: [
                rebind(input, index: 0),
                rebind(normWeight, index: 1),
                rebind(residual, index: 2),
                rebind(scalarWeight, index: 3),
                rebind(output, index: 4),
            ],
            constants: [],
            dispatch: .threadgroups(
                width: 1,
                height: 1,
                depth: 1,
                tgWidth: 256,
                tgHeight: 1,
                tgDepth: 1
            ),
            comment: (normAdd.comment ?? "RMS norm + residual") + " + layer scalar (fused)",
            minSeqLen: normAdd.minSeqLen,
            maxSeqLenExclusive: normAdd.maxSeqLenExclusive,
            minPositionPlus1: normAdd.minPositionPlus1,
            maxPositionPlus1Exclusive: normAdd.maxPositionPlus1Exclusive
        )

        return SmeltFusionRewrite(
            consumedOpCount: 2,
            producedOps: [.dispatch(fused)],
            kind: .specializedKernel,
            rule: .rmsNormResidualAddScalarWeight
        )
    }

    private func dualMatvecActivationRewrite(
        window: ArraySlice<SmeltIROp>
    ) -> SmeltFusionRewrite? {
        guard window.count >= 3 else { return nil }

        let firstIndex = window.startIndex
        let secondIndex = window.index(after: firstIndex)
        let activationIndex = window.index(after: secondIndex)
        guard case .dispatch(let first) = window[firstIndex],
              case .dispatch(let second) = window[secondIndex],
              case .dispatch(let activation) = window[activationIndex],
              let fusedPipeline = SmeltKernelShapeRegistry.prefillDualMatvecActivationPipeline(
                  first: first.pipeline,
                  second: second.pipeline,
                  activation: activation.pipeline
              ),
              first.buffers.count == 5,
              second.buffers.count == 5,
              activation.buffers.count == 3,
              first.constants.count == 1,
              second.constants.count == 1,
              activation.constants.count == 1,
              first.constants[0].expression == second.constants[0].expression,
              sameBinding(first.buffers[3], second.buffers[3]),
              sameBinding(first.buffers[4], activation.buffers[0]),
              sameBinding(second.buffers[4], activation.buffers[1]),
              first.minSeqLen == second.minSeqLen,
              first.minSeqLen == activation.minSeqLen,
              first.maxSeqLenExclusive == second.maxSeqLenExclusive,
              first.maxSeqLenExclusive == activation.maxSeqLenExclusive,
              first.minPositionPlus1 == second.minPositionPlus1,
              first.minPositionPlus1 == activation.minPositionPlus1,
              first.maxPositionPlus1Exclusive == second.maxPositionPlus1Exclusive,
              first.maxPositionPlus1Exclusive == activation.maxPositionPlus1Exclusive,
              first.dynamicGridW == nil,
              second.dynamicGridW == nil,
              first.dynamicGridH == .seqLenCeilDiv(8),
              second.dynamicGridH == first.dynamicGridH,
              first.dynamicGridD == second.dynamicGridD,
              case .threadgroups(
                  let gridW,
                  let gridH,
                  let gridD,
                  _,
                  let tgH,
                  let tgD
              ) = first.dispatch
        else {
            return nil
        }

        let fused = SmeltDispatch(
            pipeline: fusedPipeline,
            buffers: [
                rebind(first.buffers[0], index: 0),
                rebind(first.buffers[1], index: 1),
                rebind(first.buffers[2], index: 2),
                rebind(second.buffers[0], index: 3),
                rebind(second.buffers[1], index: 4),
                rebind(second.buffers[2], index: 5),
                rebind(first.buffers[3], index: 6),
                rebind(activation.buffers[2], index: 7),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: first.constants[0].expression,
                    type: first.constants[0].type,
                    index: 8
                ),
            ],
            dispatch: .threadgroups(
                width: gridW,
                height: (gridH * 8 + 2) / 3,
                depth: gridD,
                tgWidth: 64,
                tgHeight: tgH,
                tgDepth: tgD
            ),
            comment: "Fused " + (activation.comment ?? "dual matvec activation"),
            dynamicGridW: first.dynamicGridW,
            dynamicGridH: .seqLenCeilDiv(3),
            dynamicGridD: first.dynamicGridD,
            minSeqLen: first.minSeqLen,
            maxSeqLenExclusive: first.maxSeqLenExclusive,
            minPositionPlus1: first.minPositionPlus1,
            maxPositionPlus1Exclusive: first.maxPositionPlus1Exclusive
        )

        return SmeltFusionRewrite(
            consumedOpCount: 3,
            producedOps: [.dispatch(fused)],
            kind: .specializedKernel,
            rule: .dualMatvecActivation
        )
    }

    private struct MatvecResidualAddRule {
        let outputBufferIndex: Int
        let residualBindingIndex: Int
        let genericFusedPipeline: SmeltPipeline
        let genericNumRowsConstIndex: Int
    }

    private struct FixedAffineShape {
        let rows: Int
        let cols: Int
        let groupSize: Int
    }

    private struct GeneratedAffineResidualAddRoute {
        let name: String
        let shape: FixedAffineShape
        let launchGeometry: SmeltPlannedKernelLaunchGeometry
    }

    private struct BatchedFullAffineShape {
        let rows: Int
        let batchTile: Int
    }

    private static let matvecResidualAddRules: [SmeltPipeline: MatvecResidualAddRule] = [
        .fusedLutMatvec: MatvecResidualAddRule(
            outputBufferIndex: 3,
            residualBindingIndex: 4,
            genericFusedPipeline: .fusedLutMatvecAdd,
            genericNumRowsConstIndex: 5
        ),
        .affineMatvec: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .fusedAffineMatvecAdd,
            genericNumRowsConstIndex: 6
        ),
        .signedBinaryMatvecG128Rows8: MatvecResidualAddRule(
            outputBufferIndex: 3,
            residualBindingIndex: 4,
            genericFusedPipeline: .signedBinaryMatvecAddG128Rows8,
            genericNumRowsConstIndex: 5
        ),
        .signedTernaryAffineMatvecG128Rows8: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .signedTernaryAffineMatvecAddG128Rows8,
            genericNumRowsConstIndex: 6
        ),
        .affineMatvecC2048R2048G64: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .fusedAffineMatvecAdd,
            genericNumRowsConstIndex: 6
        ),
        .affineMatvecC6144R2048G64: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .fusedAffineMatvecAdd,
            genericNumRowsConstIndex: 6
        ),
        .affineMatvecC2048R1024G64Rows4: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .fusedAffineMatvecAdd,
            genericNumRowsConstIndex: 6
        ),
        .affineMatvecC3584R1024G64Rows4: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .fusedAffineMatvecAdd,
            genericNumRowsConstIndex: 6
        ),
        .affineMatvecC2048R1536G128Rows4: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .fusedAffineMatvecAdd,
            genericNumRowsConstIndex: 6
        ),
        .affineMatvecC4096R1536G128: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .fusedAffineMatvecAdd,
            genericNumRowsConstIndex: 6
        ),
        .affineMatvecC6144R1536G128Rows4: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .fusedAffineMatvecAdd,
            genericNumRowsConstIndex: 6
        ),
        .affineMatvecC12288R1536G128Rows4: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .fusedAffineMatvecAdd,
            genericNumRowsConstIndex: 6
        ),
        .affineMatvecC2048R1024G64BatchedFull: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .fusedAffineMatvecAdd,
            genericNumRowsConstIndex: 6
        ),
        .affineMatvecC3584R1024G64BatchedFull: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .fusedAffineMatvecAdd,
            genericNumRowsConstIndex: 6
        ),
        .affineMatvecC2048R2048G64BatchedFull: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .fusedAffineMatvecAdd,
            genericNumRowsConstIndex: 6
        ),
        .affineMatvecC6144R2048G64BatchedFull: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .fusedAffineMatvecAdd,
            genericNumRowsConstIndex: 6
        ),
        .affineMatvecC8192R2048G64BatchedFull: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .fusedAffineMatvecAdd,
            genericNumRowsConstIndex: 6
        ),
        .affineMatvecC4096R2560G64BatchedFull: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .fusedAffineMatvecAdd,
            genericNumRowsConstIndex: 6
        ),
        .affineMatvecC9216R2560G64BatchedFull: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .fusedAffineMatvecAdd,
            genericNumRowsConstIndex: 6
        ),
        .affineMatvecC3072R3072G64BatchedFull: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .fusedAffineMatvecAdd,
            genericNumRowsConstIndex: 6
        ),
        .affineMatvecC8192R3072G64BatchedFull: MatvecResidualAddRule(
            outputBufferIndex: 4,
            residualBindingIndex: 5,
            genericFusedPipeline: .fusedAffineMatvecAdd,
            genericNumRowsConstIndex: 6
        ),
    ]

    private static func carriesRowsAndColsConstants(_ pipeline: SmeltPipeline) -> Bool {
        switch pipeline {
        case .signedBinaryMatvecG128Rows8,
             .signedTernaryAffineMatvecG128Rows8,
             .signedBinaryBitplaneI4MatvecG128Rows8,
             .signedBinaryBitplaneI3MatvecG128Rows8,
             .signedBinaryBitplaneI2MatvecG128Rows8,
             .signedBinaryBitplaneI5MatvecG128Rows8,
             .signedBinaryBitplaneI6MatvecG128Rows8:
            return true
        default:
            return false
        }
    }

    private func matvecResidualAddRewrite(
        window: ArraySlice<SmeltIROp>
    ) -> SmeltFusionRewrite? {
        guard window.count >= 2,
              case .dispatch(let matvec) = window[window.startIndex],
              let rule = Self.matvecResidualAddRules[matvec.pipeline]
        else {
            return nil
        }
        guard matvec.pipelineNameOverride == nil else {
            return nil
        }
        var addIndex = window.index(after: window.startIndex)
        var deferredMarkers: [SmeltIROp] = []
        while addIndex < window.endIndex {
            guard case .traceMarker = window[addIndex] else { break }
            deferredMarkers.append(window[addIndex])
            addIndex = window.index(after: addIndex)
        }

        guard addIndex < window.endIndex,
              case .dispatch(let add) = window[addIndex],
              add.pipeline == .elementwiseAdd,
              add.buffers.count == 3,
              add.constants.count == 1
        else {
            return nil
        }

        guard matvec.minSeqLen == add.minSeqLen,
              matvec.maxSeqLenExclusive == add.maxSeqLenExclusive,
              matvec.minPositionPlus1 == add.minPositionPlus1,
              matvec.maxPositionPlus1Exclusive == add.maxPositionPlus1Exclusive
        else {
            return nil
        }

        let fixedShape = fixedAffineShape(for: matvec.pipeline)
        let batchedFullShape = batchedFullAffineShape(for: matvec.pipeline)

        let numRowsExpr: String
        let addCountExpr: String
        if let batchedFullShape {
            guard matvec.constants.count == 1,
                  matvec.constants[0].expression == "__seqLen__",
                  matvec.dynamicGridW == nil,
                  matvec.dynamicGridH == .seqLenCeilDiv(batchedFullShape.batchTile),
                  matvec.dynamicGridD == nil,
                  add.dynamicGridW == .seqLenMul(batchedFullShape.rows),
                  add.dynamicGridH == nil,
                  add.dynamicGridD == nil
            else {
                return nil
            }
            numRowsExpr = "\(batchedFullShape.rows)"
            addCountExpr = "__seqLen__*\(batchedFullShape.rows)"
        } else if matvec.pipeline == .affineMatvecC2048R2048G64
                    || matvec.pipeline == .affineMatvecC6144R2048G64 {
            guard matvec.dynamicGridW == nil,
                  matvec.dynamicGridH == nil,
                  matvec.dynamicGridD == nil,
                  add.dynamicGridW == nil,
                  add.dynamicGridH == nil,
                  add.dynamicGridD == nil
            else {
                return nil
            }
            numRowsExpr = "2048"
            addCountExpr = numRowsExpr
        } else if let fixedShape {
            guard matvec.dynamicGridW == nil,
                  matvec.dynamicGridH == nil,
                  matvec.dynamicGridD == nil,
                  add.dynamicGridW == nil,
                  add.dynamicGridH == nil,
                  add.dynamicGridD == nil
            else {
                return nil
            }
            numRowsExpr = "\(fixedShape.rows)"
            addCountExpr = numRowsExpr
        } else {
            guard matvec.constants.count == 1
                    || (Self.carriesRowsAndColsConstants(matvec.pipeline)
                        && matvec.constants.count == 2)
            else { return nil }
            numRowsExpr = matvec.constants[0].expression
            addCountExpr = numRowsExpr
        }

        let matvecOut = matvec.buffers[rule.outputBufferIndex]
        let addInA = add.buffers[0]
        let addInB = add.buffers[1]
        let addOut = add.buffers[2]

        guard deferredMarkers.isEmpty || !sameBinding(addOut, matvecOut) else {
            return nil
        }

        let addAMatchesMatvecOutput = sameBinding(addInA, matvecOut)
        let addBMatchesMatvecOutput = sameBinding(addInB, matvecOut)
        let residual: SmeltBufferBinding
        if addBMatchesMatvecOutput && !addAMatchesMatvecOutput {
            residual = addInA
        } else if addAMatchesMatvecOutput && !addBMatchesMatvecOutput {
            residual = addInB
        } else {
            return nil
        }

        guard addCountExpr == add.constants[0].expression else { return nil }

        let specializedFusedPipeline =
            r1536AffineResidualAddPipeline(for: matvec.pipeline)
            ?? qwenAffineResidualAddPipeline(for: matvec, numRowsExpr: numRowsExpr)
            ?? batchedFullAffineResidualAddPipeline(for: matvec.pipeline)
        let generatedFusedRoute = specializedFusedPipeline == nil
            ? generatedAffineResidualAddRoute(for: matvec, numRowsExpr: numRowsExpr)
            : nil
        let isBatchedFullFused = batchedFullShape != nil && specializedFusedPipeline != nil
        guard deferredMarkers.isEmpty || isBatchedFullFused || generatedFusedRoute != nil else {
            return nil
        }

        var fusedBuffers: [SmeltBufferBinding] = []
        for (idx, buffer) in matvec.buffers.enumerated() {
            if idx == rule.outputBufferIndex && !isBatchedFullFused {
                fusedBuffers.append(rebind(addOut, index: idx))
            } else {
                fusedBuffers.append(rebind(buffer, index: idx))
            }
        }
        fusedBuffers.append(rebind(residual, index: rule.residualBindingIndex))
        if isBatchedFullFused {
            fusedBuffers.append(rebind(addOut, index: 6))
        }

        let fusedPipeline = specializedFusedPipeline ?? rule.genericFusedPipeline
        let fusedConstants: [SmeltConstantBinding]
        if isBatchedFullFused {
            fusedConstants = [
                SmeltConstantBinding(
                    expression: matvec.constants[0].expression,
                    type: matvec.constants[0].type,
                    index: 7
                ),
            ]
        } else if generatedFusedRoute != nil {
            fusedConstants = []
        } else if Self.carriesRowsAndColsConstants(matvec.pipeline) {
            fusedConstants = [
                SmeltConstantBinding(
                    expression: matvec.constants[0].expression,
                    type: matvec.constants[0].type,
                    index: rule.genericNumRowsConstIndex
                ),
                SmeltConstantBinding(
                    expression: matvec.constants[1].expression,
                    type: matvec.constants[1].type,
                    index: rule.genericNumRowsConstIndex + 1
                ),
            ]
        } else if specializedFusedPipeline == nil {
            fusedConstants = [
                SmeltConstantBinding(
                    expression: numRowsExpr,
                    type: .uint32,
                    index: rule.genericNumRowsConstIndex
                ),
            ]
        } else {
            fusedConstants = []
        }

        let fusedDispatch: SmeltDispatchStyle
        let fusedFCCols: Int?
        let fusedFCGroupSize: Int?
        if specializedFusedPipeline != nil {
            fusedDispatch = matvec.dispatch
            fusedFCCols = nil
            fusedFCGroupSize = nil
        } else if let generatedFusedRoute {
            fusedDispatch = .threadgroups(
                width: generatedFusedRoute.launchGeometry.gridWidth(
                    rows: generatedFusedRoute.shape.rows
                ),
                height: 1,
                depth: 1,
                tgWidth: generatedFusedRoute.launchGeometry.threadgroupWidth,
                tgHeight: 1,
                tgDepth: 1
            )
            fusedFCCols = nil
            fusedFCGroupSize = nil
        } else if let fixedShape {
            fusedDispatch = .threadgroups(
                width: (fixedShape.rows + 7) / 8,
                height: 1,
                depth: 1,
                tgWidth: 64,
                tgHeight: 1,
                tgDepth: 1
            )
            fusedFCCols = fixedShape.cols
            fusedFCGroupSize = fixedShape.groupSize
        } else {
            fusedDispatch = matvec.dispatch
            fusedFCCols = matvec.fcCols
            fusedFCGroupSize = matvec.fcGroupSize
        }

        let fusedComment = (matvec.comment ?? "Matvec") + " + residual (fused)"
        let fused: SmeltDispatch
        if generatedFusedRoute != nil, let plannedKernelCandidate = matvec.plannedKernelCandidate {
            fused = SmeltDispatch(
                pipeline: fusedPipeline,
                pipelineNameOverride: generatedFusedRoute?.name,
                buffers: fusedBuffers,
                constants: fusedConstants,
                dispatch: fusedDispatch,
                comment: fusedComment,
                plannedKernelCandidate: plannedKernelCandidate,
                fcCols: fusedFCCols,
                fcGroupSize: fusedFCGroupSize,
                dynamicGridW: matvec.dynamicGridW,
                dynamicGridH: matvec.dynamicGridH,
                dynamicGridD: matvec.dynamicGridD,
                minSeqLen: matvec.minSeqLen,
                maxSeqLenExclusive: matvec.maxSeqLenExclusive,
                minPositionPlus1: matvec.minPositionPlus1,
                maxPositionPlus1Exclusive: matvec.maxPositionPlus1Exclusive
            )
        } else {
            fused = SmeltDispatch(
                pipeline: fusedPipeline,
                pipelineNameOverride: generatedFusedRoute?.name,
                buffers: fusedBuffers,
                constants: fusedConstants,
                dispatch: fusedDispatch,
                comment: fusedComment,
                fcCols: fusedFCCols,
                fcGroupSize: fusedFCGroupSize,
                dynamicGridW: matvec.dynamicGridW,
                dynamicGridH: matvec.dynamicGridH,
                dynamicGridD: matvec.dynamicGridD,
                minSeqLen: matvec.minSeqLen,
                maxSeqLenExclusive: matvec.maxSeqLenExclusive,
                minPositionPlus1: matvec.minPositionPlus1,
                maxPositionPlus1Exclusive: matvec.maxPositionPlus1Exclusive
            )
        }

        var fusedOps: [SmeltIROp] = [.dispatch(fused)]
        if let batchedFullShape,
           batchedFullShape.batchTile == 16,
           let batch8Pipeline = batch8AffineResidualAddPipeline(for: matvec.pipeline),
           matvec.minSeqLen == nil,
           matvec.maxSeqLenExclusive == nil,
           matvec.minPositionPlus1 == nil,
           matvec.maxPositionPlus1Exclusive == nil
        {
            func guardedFused(
                pipeline: SmeltPipeline,
                dispatch: SmeltDispatchStyle,
                dynamicGridH: SmeltDynamicGridDimension,
                minSeqLen: Int? = nil,
                maxSeqLenExclusive: Int? = nil,
                commentSuffix: String
            ) -> SmeltDispatch {
                SmeltDispatch(
                    pipeline: pipeline,
                    buffers: fusedBuffers,
                    constants: fusedConstants,
                    dispatch: dispatch,
                    comment: fusedComment + commentSuffix,
                    dynamicGridW: matvec.dynamicGridW,
                    dynamicGridH: dynamicGridH,
                    dynamicGridD: matvec.dynamicGridD,
                    minSeqLen: minSeqLen,
                    maxSeqLenExclusive: maxSeqLenExclusive
                )
            }

            let batch16Below = guardedFused(
                pipeline: fusedPipeline,
                dispatch: fusedDispatch,
                dynamicGridH: .seqLenCeilDiv(16),
                maxSeqLenExclusive: 8,
                commentSuffix: " [batch<8,b16]"
            )
            let batch8Exact = guardedFused(
                pipeline: batch8Pipeline,
                dispatch: .threadgroups(
                    width: (batchedFullShape.rows + 31) / 32,
                    height: 1,
                    depth: 1,
                    tgWidth: 64,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                dynamicGridH: .seqLenCeilDiv(8),
                minSeqLen: 8,
                maxSeqLenExclusive: 9,
                commentSuffix: " [batch=8,b8]"
            )
            let batch16Above = guardedFused(
                pipeline: fusedPipeline,
                dispatch: fusedDispatch,
                dynamicGridH: .seqLenCeilDiv(16),
                minSeqLen: 9,
                commentSuffix: " [batch>=9,b16]"
            )
            fusedOps = [
                .dispatch(batch16Below),
                .dispatch(batch8Exact),
                .dispatch(batch16Above),
            ]
        }

        return SmeltFusionRewrite(
            consumedOpCount: window.distance(
                from: window.startIndex,
                to: window.index(after: addIndex)
            ),
            producedOps: fusedOps + deferredMarkers,
            kind: specializedFusedPipeline == nil && generatedFusedRoute == nil
                ? .genericKernel
                : .specializedKernel,
            rule: .matvecResidualAdd
        )
    }

    private func generatedAffineResidualAddRoute(
        for matvec: SmeltDispatch,
        numRowsExpr: String
    ) -> GeneratedAffineResidualAddRoute? {
        guard let kernelPlan,
              let rows = Int(numRowsExpr)
        else {
            return nil
        }

        let shape: SmeltKernelShape
        if matvec.pipeline == .affineMatvec,
           let cols = matvec.fcCols,
           let groupSize = matvec.fcGroupSize {
            shape = SmeltKernelShape(rows: rows, cols: cols, groupSize: groupSize)
        } else if let fixedShape = fixedAffineShape(for: matvec.pipeline),
                  fixedShape.rows == rows {
            shape = SmeltKernelShape(
                rows: fixedShape.rows,
                cols: fixedShape.cols,
                groupSize: fixedShape.groupSize
            )
        } else {
            return nil
        }

        guard let candidate = matvec.plannedKernelCandidate,
              candidate.operation == .affineMatvecResidualAdd,
              candidate.shape == shape,
              let route = kernelPlan.route(for: candidate),
              let launchGeometry = route.affineMatvecResidualAddLaunchGeometry()
        else {
            return nil
        }
        let capability = route.capability

        return GeneratedAffineResidualAddRoute(
            name: capability.id,
            shape: FixedAffineShape(
                rows: capability.shape.rows,
                cols: capability.shape.cols,
                groupSize: capability.shape.groupSize
            ),
            launchGeometry: launchGeometry
        )
    }

    private func fixedAffineShape(for pipeline: SmeltPipeline) -> FixedAffineShape? {
        switch pipeline {
        case .affineMatvecC2048R1024G64Rows4:
            return FixedAffineShape(rows: 1_024, cols: 2_048, groupSize: 64)
        case .affineMatvecC3584R1024G64Rows4:
            return FixedAffineShape(rows: 1_024, cols: 3_584, groupSize: 64)
        case .affineMatvecC2048R1536G128Rows4:
            return FixedAffineShape(rows: 1_536, cols: 2_048, groupSize: 128)
        case .affineMatvecC4096R1536G128:
            return FixedAffineShape(rows: 1_536, cols: 4_096, groupSize: 128)
        case .affineMatvecC6144R1536G128Rows4:
            return FixedAffineShape(rows: 1_536, cols: 6_144, groupSize: 128)
        case .affineMatvecC12288R1536G128Rows4:
            return FixedAffineShape(rows: 1_536, cols: 12_288, groupSize: 128)
        default:
            return nil
        }
    }

    private func batchedFullAffineShape(for pipeline: SmeltPipeline) -> BatchedFullAffineShape? {
        switch pipeline {
        case .affineMatvecC2048R1024G64BatchedFull,
             .affineMatvecC3584R1024G64BatchedFull:
            return BatchedFullAffineShape(rows: 1_024, batchTile: 16)
        case .affineMatvecC2048R2048G64BatchedFull,
             .affineMatvecC6144R2048G64BatchedFull,
             .affineMatvecC8192R2048G64BatchedFull:
            return BatchedFullAffineShape(rows: 2_048, batchTile: 16)
        case .affineMatvecC4096R2560G64BatchedFull,
             .affineMatvecC9216R2560G64BatchedFull:
            return BatchedFullAffineShape(rows: 2_560, batchTile: 16)
        case .affineMatvecC3072R3072G64BatchedFull,
             .affineMatvecC8192R3072G64BatchedFull:
            return BatchedFullAffineShape(rows: 3_072, batchTile: 16)
        default:
            return nil
        }
    }

    private func qwenAffineResidualAddPipeline(
        for matvec: SmeltDispatch,
        numRowsExpr: String
    ) -> SmeltPipeline? {
        if numRowsExpr == "1024",
           matvec.pipeline == .affineMatvecC2048R1024G64Rows4
        {
            return .fusedAffineMatvecAddC2048R1024G64Rows4
        }

        if numRowsExpr != "2048" { return nil }

        if matvec.pipeline == .affineMatvecC2048R2048G64 {
            return .fusedAffineMatvecAddC2048R2048G64
        }

        if matvec.pipeline == .affineMatvecC6144R2048G64 {
            return .fusedAffineMatvecAddC6144R2048G64
        }

        guard matvec.pipeline == .affineMatvec,
              matvec.fcGroupSize == 64
        else {
            return nil
        }

        switch matvec.fcCols {
        case 2_048: return .fusedAffineMatvecAddC2048R2048G64
        case 6_144: return .fusedAffineMatvecAddC6144R2048G64
        default: return nil
        }
    }

    private func r1536AffineResidualAddPipeline(for pipeline: SmeltPipeline) -> SmeltPipeline? {
        switch pipeline {
        case .affineMatvecC2048R1536G128Rows4:
            return .fusedAffineMatvecAddC2048R1536G128Rows4
        case .affineMatvecC4096R1536G128:
            return .fusedAffineMatvecAddC4096R1536G128
        case .affineMatvecC6144R1536G128Rows4:
            return .fusedAffineMatvecAddC6144R1536G128Rows4
        case .affineMatvecC12288R1536G128Rows4:
            return .fusedAffineMatvecAddC12288R1536G128Rows4
        default:
            return nil
        }
    }

    private func batchedFullAffineResidualAddPipeline(for pipeline: SmeltPipeline) -> SmeltPipeline? {
        switch pipeline {
        case .affineMatvecC2048R1024G64BatchedFull:
            return .fusedAffineMatvecAddC2048R1024G64BatchedFull
        case .affineMatvecC3584R1024G64BatchedFull:
            return .fusedAffineMatvecAddC3584R1024G64BatchedFull
        case .affineMatvecC2048R2048G64BatchedFull:
            return .fusedAffineMatvecAddC2048R2048G64BatchedFull
        case .affineMatvecC6144R2048G64BatchedFull:
            return .fusedAffineMatvecAddC6144R2048G64BatchedFull
        case .affineMatvecC8192R2048G64BatchedFull:
            return .fusedAffineMatvecAddC8192R2048G64BatchedFull
        case .affineMatvecC4096R2560G64BatchedFull:
            return .fusedAffineMatvecAddC4096R2560G64BatchedFull
        case .affineMatvecC9216R2560G64BatchedFull:
            return .fusedAffineMatvecAddC9216R2560G64BatchedFull
        case .affineMatvecC3072R3072G64BatchedFull:
            return .fusedAffineMatvecAddC3072R3072G64BatchedFull
        case .affineMatvecC8192R3072G64BatchedFull:
            return .fusedAffineMatvecAddC8192R3072G64BatchedFull
        default:
            return nil
        }
    }

    private func batch8AffineResidualAddPipeline(for pipeline: SmeltPipeline) -> SmeltPipeline? {
        switch pipeline {
        case .affineMatvecC2048R1024G64BatchedFull:
            return .fusedAffineMatvecAddC2048R1024G64BatchedFullB8
        case .affineMatvecC3584R1024G64BatchedFull:
            return .fusedAffineMatvecAddC3584R1024G64BatchedFullB8
        default:
            return nil
        }
    }

    private func rebind(_ binding: SmeltBufferBinding, index: Int) -> SmeltBufferBinding {
        switch binding.slot {
        case .fixed(let slot):
            if let offsetExpression = binding.offsetExpression {
                return SmeltBufferBinding(slot: slot, offsetExpression: offsetExpression, index: index)
            }
            if binding.offsetKind != 0 {
                return SmeltBufferBinding(
                    slot: slot,
                    offset: binding.byteOffset,
                    offsetKind: binding.offsetKind,
                    index: index
                )
            }
            return SmeltBufferBinding(slot: slot, offset: binding.byteOffset, index: index)
        case .variable(let name):
            return SmeltBufferBinding(
                variableSlot: name,
                offset: binding.byteOffset,
                index: index
            )
        }
    }

    private func sameBinding(
        _ lhs: SmeltBufferBinding,
        _ rhs: SmeltBufferBinding
    ) -> Bool {
        lhs.slot == rhs.slot
            && lhs.byteOffset == rhs.byteOffset
            && lhs.offsetKind == rhs.offsetKind
            && lhs.offsetExpression == rhs.offsetExpression
    }

    private func sameBindingAcrossSwap(
        _ before: SmeltBufferBinding,
        _ after: SmeltBufferBinding
    ) -> Bool {
        guard before.byteOffset == after.byteOffset,
              before.offsetKind == after.offsetKind,
              before.offsetExpression == after.offsetExpression
        else { return false }
        switch (before.slot, after.slot) {
        case (.variable("cur"), .variable("alt")),
             (.variable("alt"), .variable("cur")):
            return true
        default:
            return false
        }
    }
}

enum SmeltFusionRule: String, Sendable, Equatable, Hashable {
    case contiguousL2Normalize
    case cooperativeNormScaleConsumer
    case dualMatvecActivation
    case fusedNormRopeKVPrefill
    case gatedNormActivationView
    case matvecResidualAdd
    case normActivationView
    case residualAddNormActivationView
    case normScaleDualAffineConsumer
    case rmsNormResidualAdd
    case rmsNormResidualAddScalarWeight
    case sigmoidMulActivationView
    case swigluActivationView
}

struct SmeltFusionRewrite: Sendable {
    let consumedOpCount: Int
    let producedOps: [SmeltIROp]
    let kind: SmeltFusionKind
    let rule: SmeltFusionRule
}

struct SmeltOptionalPipelineRoute: Sendable {
    let pipeline: SmeltPipeline?
    let kind: SmeltFusionKind
}

struct SmeltAffineMatvecRoute: Sendable {
    let pipeline: SmeltPipeline?
    let kind: SmeltFusionKind
    let rowTile: Int
}

struct SmeltDecodeFusedFFNRoute: Sendable {
    let pipeline: SmeltPipeline?
    let kind: SmeltFusionKind
    let rowTile: Int
    let threadgroupWidth: Int
}

struct SmeltPrefillTiledRoute: Sendable {
    let pipeline: SmeltPipeline
    let kind: SmeltFusionKind
    let rowTile: Int
    let batchTile: Int
    let threadgroupWidth: Int
}

struct SmeltDecodeNormScaleRoute: Sendable {
    let pipeline: SmeltPipeline
    let kind: SmeltFusionKind
    let rowTile: Int
}

struct SmeltRMSNormRoute: Sendable {
    let pipeline: SmeltPipeline?
    let kind: SmeltFusionKind
    let threadgroupWidth: Int?
}
