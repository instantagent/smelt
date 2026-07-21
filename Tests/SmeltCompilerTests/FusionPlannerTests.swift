import Foundation
import Testing

@testable import SmeltCompiler
import SmeltSchema

@Suite("Fusion Planner")
struct FusionPlannerTests {

    @Test("Decode affine route carries tuned pipeline and row tile")
    func decodeAffineRouteCarriesTunedPipelineAndRowTile() {
        let route = SmeltFusionPlanner.auto.decodeAffineMatvec(
            rows: 2_048,
            cols: 1_536,
            groupSize: 128
        )

        #expect(route.pipeline == .affineMatvecC1536R2048G128Rows4)
        #expect(route.kind == .specializedKernel)
        #expect(route.rowTile == 4)
    }

    @Test("Unsupported affine route falls back to generic kernel")
    func unsupportedAffineRouteFallsBackToGenericKernel() {
        let route = SmeltFusionPlanner.auto.decodeAffineMatvec(
            rows: 123,
            cols: 456,
            groupSize: 64
        )

        #expect(route.pipeline == nil)
        #expect(route.kind == .genericKernel)
    }

    @Test("Prefill affine full route carries QMM geometry")
    func prefillAffineFullRouteCarriesQMMGeometry() {
        let route = SmeltFusionPlanner.auto.prefillAffineFull(
            rows: 2_048,
            cols: 8_192,
            groupSize: 64
        )

        #expect(route?.pipeline == SmeltPipeline.affineMatvecC8192R2048G64BatchedFull)
        #expect(route?.kind == .specializedKernel)
        #expect(route?.rowTile == 32)
        #expect(route?.batchTile == 16)
        #expect(route?.threadgroupWidth == 128)
    }

    @Test("Verify-small-batch affine route uses SG4 hot projections")
    func verifySmallBatchAffineRouteUsesSG4HotProjections() {
        let autoQRoute = SmeltFusionPlanner.auto.prefillAffineBatched(
            rows: 2_048,
            cols: 2_560,
            groupSize: 128
        )
        let verifyQRoute = SmeltFusionPlanner.verifySmallBatch.prefillAffineBatched(
            rows: 2_048,
            cols: 2_560,
            groupSize: 128
        )
        let autoORoute = SmeltFusionPlanner.auto.prefillAffineBatched(
            rows: 2_560,
            cols: 2_048,
            groupSize: 128
        )
        let verifyORoute = SmeltFusionPlanner.verifySmallBatch.prefillAffineBatched(
            rows: 2_560,
            cols: 2_048,
            groupSize: 128
        )
        let autoGateUpRoute = SmeltFusionPlanner.auto.prefillAffineBatched(
            rows: 10_240,
            cols: 2_560,
            groupSize: 128
        )
        let verifyGateUpRoute = SmeltFusionPlanner.verifySmallBatch.prefillAffineBatched(
            rows: 10_240,
            cols: 2_560,
            groupSize: 128
        )
        let autoDownRoute = SmeltFusionPlanner.auto.prefillAffineBatched(
            rows: 2_560,
            cols: 10_240,
            groupSize: 128
        )
        let verifyDownRoute = SmeltFusionPlanner.verifySmallBatch.prefillAffineBatched(
            rows: 2_560,
            cols: 10_240,
            groupSize: 128
        )

        #expect(autoQRoute?.pipeline == .affineMatvecC2560R2048G128Batched)
        #expect(autoQRoute?.rowTile == 8)
        #expect(autoQRoute?.batchTile == 8)
        #expect(autoQRoute?.threadgroupWidth == 64)
        #expect(verifyQRoute?.pipeline == .affineMatvecC2560R2048G128BatchedTile3)
        #expect(verifyQRoute?.rowTile == 8)
        #expect(verifyQRoute?.batchTile == 3)
        #expect(verifyQRoute?.threadgroupWidth == 64)

        #expect(autoORoute?.pipeline == .affineMatvecC2048R2560G128Batched)
        #expect(autoORoute?.rowTile == 8)
        #expect(autoORoute?.batchTile == 8)
        #expect(autoORoute?.threadgroupWidth == 64)
        #expect(verifyORoute?.pipeline == .affineMatvecC2048R2560G128BatchedTile3)
        #expect(verifyORoute?.rowTile == 8)
        #expect(verifyORoute?.batchTile == 3)
        #expect(verifyORoute?.threadgroupWidth == 64)

        #expect(autoGateUpRoute?.pipeline == .affineMatvecC2560R10240G128Batched)
        #expect(autoGateUpRoute?.rowTile == 8)
        #expect(autoGateUpRoute?.batchTile == 8)
        #expect(autoGateUpRoute?.threadgroupWidth == 64)
        #expect(verifyGateUpRoute?.pipeline == .affineMatvecC2560R10240G128BatchedExtB4)
        #expect(verifyGateUpRoute?.rowTile == 8)
        #expect(verifyGateUpRoute?.batchTile == 4)
        #expect(verifyGateUpRoute?.threadgroupWidth == 64)

        #expect(autoDownRoute?.pipeline == .affineMatvecC10240R2560G128BatchedTile4)
        #expect(autoDownRoute?.rowTile == 8)
        #expect(autoDownRoute?.batchTile == 4)
        #expect(autoDownRoute?.threadgroupWidth == 64)
        #expect(verifyDownRoute?.pipeline == .affineMatvecC10240R2560G128BatchedTile3)
        #expect(verifyDownRoute?.rowTile == 8)
        #expect(verifyDownRoute?.batchTile == 3)
        #expect(verifyDownRoute?.threadgroupWidth == 64)
    }

    @Test("Prefill gate-up route is activation-configured")
    func prefillGateUpRouteIsActivationConfigured() {
        let route = SmeltFusionPlanner.auto.prefillFusedGateUpFull(
            rows: 10_240,
            cols: 2_560,
            groupSize: 128,
            activation: .geglu
        )

        #expect(route?.pipeline == SmeltPipeline.fusedAffineGateUpGeGLUC2560R10240G128BatchedFull)
        #expect(route?.kind == .specializedKernel)
        #expect(route?.rowTile == 32)
        #expect(route?.batchTile == 16)
        #expect(route?.threadgroupWidth == 128)
    }

    @Test("Verify-small-batch routes fused gate-up to BT3 kernel")
    func verifySmallBatchRoutesFusedGateUpToBT3Kernel() {
        let route = SmeltFusionPlanner.verifySmallBatch.prefillFusedGateUpFull(
            rows: 10_240,
            cols: 2_560,
            groupSize: 128,
            activation: .geglu
        )

        #expect(route?.pipeline == .fusedAffineGateUpGeGLUC2560R10240G128Batched)
        #expect(route?.kind == .specializedKernel)
        #expect(route?.rowTile == 8)
        #expect(route?.batchTile == 3)
        #expect(route?.threadgroupWidth == 64)
    }

    @Test("Variable-slot RMS norm route preserves current threadgroup width")
    func variableSlotRMSNormRoutePreservesCurrentThreadgroupWidth() {
        let regular = SmeltFusionPlanner.auto.decodeRMSNorm(dim: 1_536, eps: 1e-6)
        let variable = SmeltFusionPlanner.auto.decodeRMSNormVariableInput(
            dim: 1_536,
            eps: 1e-6
        )

        #expect(regular.pipeline == .rmsNorm1PWD1536)
        #expect(regular.kind == .specializedKernel)
        #expect(regular.threadgroupWidth == 192)
        #expect(variable.pipeline == .rmsNorm1PWD1536)
        #expect(variable.kind == .specializedKernel)
        #expect(variable.threadgroupWidth == 256)
    }

    @Test("Rewrite registry coalesces contiguous L2 normalize dispatches")
    func rewriteRegistryCoalescesContiguousL2NormalizeDispatches() {
        let rewrite = SmeltFusionPlanner.auto.rewrite(window: [
            .dispatch(SmeltDispatch(
                pipeline: .l2NormalizeD128,
                buffers: [
                    SmeltBufferBinding(slot: 2, offset: 0, index: 0),
                ],
                constants: [],
                dispatch: .threadgroups(
                    width: 8,
                    height: 1,
                    depth: 1,
                    tgWidth: 128,
                    tgHeight: 1,
                    tgDepth: 1
                )
            )),
            .dispatch(SmeltDispatch(
                pipeline: .l2NormalizeD128,
                buffers: [
                    SmeltBufferBinding(slot: 2, offset: 2_048, index: 0),
                ],
                constants: [],
                dispatch: .threadgroups(
                    width: 8,
                    height: 1,
                    depth: 1,
                    tgWidth: 128,
                    tgHeight: 1,
                    tgDepth: 1
                )
            )),
        ][...])

        #expect(rewrite?.consumedOpCount == 2)
        #expect(rewrite?.kind == .dispatchCoalescing)
        #expect(rewrite?.rule == .contiguousL2Normalize)
        guard case .dispatch(let fused)? = rewrite?.producedOps.first,
              case .threadgroups(let width, _, _, let tgWidth, _, _) = fused.dispatch
        else {
            Issue.record("Expected coalesced L2 dispatch rewrite")
            return
        }
        #expect(fused.pipeline == .l2NormalizeD128)
        #expect(width == 16)
        #expect(tgWidth == 128)
    }

    @Test("Residual-add rewrite selects specialized fused affine kernel")
    func residualAddRewriteSelectsSpecializedFusedAffineKernel() {
        let rewrite = SmeltFusionPlanner.auto.rewrite(window: [
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvec,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 13, index: 3),
                    SmeltBufferBinding(slot: 14, index: 4),
                ],
                constants: [
                    SmeltConstantBinding(expression: "2048", type: .uint32, index: 5),
                ],
                dispatch: .threadgroups(
                    width: 256,
                    height: 1,
                    depth: 1,
                    tgWidth: 64,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                comment: "FFN down projection",
                fcCols: 6_144,
                fcGroupSize: 64
            )),
            .dispatch(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 14, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "2048", type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: 2_048,
                    height: 1,
                    depth: 1,
                    tgWidth: 1_024,
                    tgHeight: 1,
                    tgDepth: 1
                )
            )),
        ][...])

        #expect(rewrite?.consumedOpCount == 2)
        #expect(rewrite?.kind == .specializedKernel)
        #expect(rewrite?.rule == .matvecResidualAdd)
        guard case .dispatch(let fused)? = rewrite?.producedOps.first else {
            Issue.record("Expected fused dispatch rewrite")
            return
        }
        #expect(fused.pipeline == .fusedAffineMatvecAddC6144R2048G64)
        #expect(fused.constants.isEmpty)
        #expect(fused.fcCols == nil)
        #expect(fused.fcGroupSize == nil)
    }

    @Test("Residual-add rewrite recognizes specialized Qwen 2B FFN down kernel")
    func residualAddRewriteRecognizesSpecializedQwen2BFFNDownKernel() {
        let rewrite = SmeltFusionPlanner.auto.rewrite(window: [
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvecC6144R2048G64,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 13, index: 3),
                    SmeltBufferBinding(slot: 14, index: 4),
                ],
                constants: [],
                dispatch: .threadgroups(
                    width: 256,
                    height: 1,
                    depth: 1,
                    tgWidth: 64,
                    tgHeight: 1,
                    tgDepth: 1
                )
            )),
            .dispatch(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 14, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "2048", type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: 2_048,
                    height: 1,
                    depth: 1,
                    tgWidth: 1_024,
                    tgHeight: 1,
                    tgDepth: 1
                )
            )),
        ][...])

        guard case .dispatch(let fused)? = rewrite?.producedOps.first else {
            Issue.record("Expected specialized fused dispatch rewrite")
            return
        }
        #expect(rewrite?.rule == .matvecResidualAdd)
        #expect(fused.pipeline == SmeltPipeline.fusedAffineMatvecAddC6144R2048G64)
        #expect(fused.constants.isEmpty)
    }

    @Test("Batched residual-add rewrite uses the measured B8 crossover only at batch eight")
    func batchedResidualAddRewriteUsesGuardedBatch8Crossover() throws {
        let rewrite = SmeltFusionPlanner.auto.rewrite(window: [
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvecC2048R1024G64BatchedFull,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 13, index: 3),
                    SmeltBufferBinding(slot: 14, index: 4),
                ],
                constants: [
                    SmeltConstantBinding(expression: "__seqLen__", type: .uint32, index: 5),
                ],
                dispatch: .threadgroups(
                    width: 32,
                    height: 16,
                    depth: 1,
                    tgWidth: 128,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                dynamicGridH: .seqLenCeilDiv(16)
            )),
            .dispatch(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 14, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", index: 2),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "__seqLen__*1024",
                        type: .uint32,
                        index: 3
                    ),
                ],
                dispatch: .threads(
                    width: 1_024,
                    height: 1,
                    depth: 1,
                    tgWidth: 1_024,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                dynamicGridW: .seqLenMul(1_024)
            )),
        ][...])

        let producedOps = try #require(rewrite?.producedOps)
        let dispatches: [SmeltDispatch] = producedOps.compactMap { op -> SmeltDispatch? in
            guard case .dispatch(let dispatch) = op else { return nil }
            return dispatch
        }
        #expect(dispatches.count == 3)

        let below = dispatches[0]
        #expect(below.pipeline == .fusedAffineMatvecAddC2048R1024G64BatchedFull)
        #expect(below.dynamicGridH == .seqLenCeilDiv(16))
        #expect(below.minSeqLen == nil)
        #expect(below.maxSeqLenExclusive == 8)
        #expect(below.toRecord().tgW == 128)

        let exact = dispatches[1]
        #expect(exact.pipeline == .fusedAffineMatvecAddC2048R1024G64BatchedFullB8)
        #expect(exact.dynamicGridH == .seqLenCeilDiv(8))
        #expect(exact.minSeqLen == 8)
        #expect(exact.maxSeqLenExclusive == 9)
        #expect(exact.toRecord().tgW == 64)

        let above = dispatches[2]
        #expect(above.pipeline == .fusedAffineMatvecAddC2048R1024G64BatchedFull)
        #expect(above.dynamicGridH == .seqLenCeilDiv(16))
        #expect(above.minSeqLen == 9)
        #expect(above.maxSeqLenExclusive == nil)
        #expect(above.toRecord().tgW == 128)
    }

    @Test("Residual-add rewrite falls back to generic fused matvec-add kernel")
    func residualAddRewriteFallsBackToGenericFusedMatvecAddKernel() {
        let rewrite = SmeltFusionPlanner.auto.rewrite(window: [
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvec,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 13, index: 3),
                    SmeltBufferBinding(slot: 14, index: 4),
                ],
                constants: [
                    SmeltConstantBinding(expression: "1234", type: .uint32, index: 5),
                ],
                dispatch: .threadgroups(
                    width: 155,
                    height: 1,
                    depth: 1,
                    tgWidth: 64,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                comment: "Generic projection",
                fcCols: 3_072,
                fcGroupSize: 64
            )),
            .dispatch(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 14, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "1234", type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: 1_234,
                    height: 1,
                    depth: 1,
                    tgWidth: 1_024,
                    tgHeight: 1,
                    tgDepth: 1
                )
            )),
        ][...])

        #expect(rewrite?.kind == .genericKernel)
        guard case .dispatch(let fused)? = rewrite?.producedOps.first else {
            Issue.record("Expected generic fused dispatch rewrite")
            return
        }
        #expect(fused.pipeline == .fusedAffineMatvecAdd)
        #expect(fused.constants.map(\.expression) == ["1234"])
        #expect(fused.fcCols == 3_072)
        #expect(fused.fcGroupSize == 64)
    }

    @Test("Auto residual-add rewrite does not invent generated affine route")
    func autoResidualAddRewriteDoesNotInventGeneratedAffineRoute() {
        let rewrite = SmeltFusionPlanner.auto.rewrite(window: [
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvec,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 13, index: 3),
                    SmeltBufferBinding(slot: 14, index: 4),
                ],
                constants: [
                    SmeltConstantBinding(expression: "2048", type: .uint32, index: 5),
                ],
                dispatch: .threadgroups(
                    width: 256,
                    height: 1,
                    depth: 1,
                    tgWidth: 64,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                comment: "Generated down projection",
                fcCols: 11_008,
                fcGroupSize: 64
            )),
            .dispatch(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 14, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "2048", type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: 2_048,
                    height: 1,
                    depth: 1,
                    tgWidth: 1_024,
                    tgHeight: 1,
                    tgDepth: 1
                )
            )),
        ][...])

        #expect(rewrite?.kind == .genericKernel)
        guard case .dispatch(let fused)? = rewrite?.producedOps.first else {
            Issue.record("Expected generic fallback dispatch rewrite")
            return
        }
        #expect(fused.pipeline == .fusedAffineMatvecAdd)
        #expect(fused.pipelineNameOverride == nil)
        #expect(fused.constants.map(\.expression) == ["2048"])
        #expect(fused.fcCols == 11_008)
        #expect(fused.fcGroupSize == 64)
    }

    @Test("Planned fusion only uses generated residual route for planned shapes")
    func plannedFusionOnlyUsesGeneratedResidualRouteForPlannedShapes() throws {
        let planner = SmeltFusionPlanner.planned(
            kernelPlan: SmeltKernelPlanner.plan(for: .qwen35_0_8B)
        )
        let layerContext = SmeltKernelLayerConsumerContext(
            ir: .qwen35_0_8B,
            layerIndex: 0,
            layerType: .delta
        )
        let downDecodeCandidate = try #require(SmeltKernelConsumerNaming.candidate(
            kind: .ffnDownResidualDecode,
            context: layerContext
        ))
        let rewrite = planner.rewrite(window: [
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvec,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 13, index: 3),
                    SmeltBufferBinding(slot: 14, index: 4),
                ],
                constants: [
                    SmeltConstantBinding(expression: "1024", type: .uint32, index: 5),
                ],
                dispatch: .threadgroups(
                    width: 128,
                    height: 1,
                    depth: 1,
                    tgWidth: 64,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                comment: "Planned down projection",
                plannedKernelCandidate: downDecodeCandidate,
                fcCols: 3_584,
                fcGroupSize: 64
            )),
            .dispatch(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 14, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "1024", type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: 1_024,
                    height: 1,
                    depth: 1,
                    tgWidth: 1_024,
                    tgHeight: 1,
                    tgDepth: 1
                )
            )),
        ][...])

        #expect(rewrite?.kind == .specializedKernel)
        guard case .dispatch(let fused)? = rewrite?.producedOps.first else {
            Issue.record("Expected planned generated fused dispatch rewrite")
            return
        }
        #expect(fused.pipelineNameOverride == "fused_affine_matvec_add_c3584_r1024_g64_rows4")
        #expect(fused.constants.isEmpty)
    }

    @Test("Planned residual fusion uses capability launch geometry")
    func plannedResidualFusionUsesCapabilityLaunchGeometry() {
        let candidate = SmeltPlannedKernelCandidate(
            consumerID: "geometry.down.residual.decode",
            operation: .affineMatvecResidualAdd,
            shape: SmeltKernelShape(rows: 512, cols: 1_024, groupSize: 64),
            weights: [
                SmeltPlannedKernelWeight(weightName: "geometry_down_weight", role: .affine),
            ]
        )
        let capability = SmeltKernelCapability(
            id: "fused_affine_matvec_add_c1024_r512_g64_geometry_probe",
            phase: .decode,
            operation: .affineMatvecResidualAdd,
            shape: candidate.shape,
            source: .packageLocalGenerated,
            weightRequirements: [
                SmeltKernelWeightRequirement(
                    role: .affine,
                    acceptedLayouts: [.affineU4RowMajor(groupSize: 64)]
                ),
            ],
            rowTile: 16,
            batchTile: nil,
            threadgroupWidth: 96
        )
        let planner = SmeltFusionPlanner.planned(
            kernelPlan: SmeltKernelPlan(generatedUses: [
                SmeltPlannedKernelUse(
                    consumerID: candidate.consumerID,
                    capability: capability,
                    weights: candidate.weights
                ),
            ])
        )

        let rewrite = planner.rewrite(window: [
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvec,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 13, index: 3),
                    SmeltBufferBinding(slot: 14, index: 4),
                ],
                constants: [
                    SmeltConstantBinding(expression: "512", type: .uint32, index: 5),
                ],
                dispatch: .threadgroups(
                    width: 64,
                    height: 1,
                    depth: 1,
                    tgWidth: 64,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                comment: "Geometry-probed down projection",
                plannedKernelCandidate: candidate,
                fcCols: 1_024,
                fcGroupSize: 64
            )),
            .dispatch(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 14, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "512", type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: 512,
                    height: 1,
                    depth: 1,
                    tgWidth: 512,
                    tgHeight: 1,
                    tgDepth: 1
                )
            )),
        ][...])

        guard case .dispatch(let fused)? = rewrite?.producedOps.first else {
            Issue.record("Expected planned generated fused dispatch rewrite")
            return
        }

        #expect(fused.pipelineNameOverride == capability.id)
        #expect(fused.constants.isEmpty)
        guard case .threadgroups(
            let width,
            let height,
            let depth,
            let tgWidth,
            let tgHeight,
            let tgDepth
        ) = fused.dispatch else {
            Issue.record("Expected planned generated fused threadgroup dispatch")
            return
        }
        #expect(width == 32)
        #expect(height == 1)
        #expect(depth == 1)
        #expect(tgWidth == 96)
        #expect(tgHeight == 1)
        #expect(tgDepth == 1)
    }

    @Test("Planned route launch geometry is route-owned")
    func plannedRouteLaunchGeometryIsRouteOwned() {
        let candidate = SmeltPlannedKernelCandidate(
            consumerID: "geometry.route-owned.decode",
            operation: .affineMatvecResidualAdd,
            shape: SmeltKernelShape(rows: 512, cols: 1_024, groupSize: 64),
            weights: [
                SmeltPlannedKernelWeight(weightName: "geometry_route_weight", role: .affine),
            ]
        )
        let capability = SmeltKernelCapability(
            id: "fused_affine_matvec_add_c1024_r512_g64_route_owned_probe",
            phase: .decode,
            operation: .affineMatvecResidualAdd,
            shape: candidate.shape,
            source: .packageLocalGenerated,
            weightRequirements: [
                SmeltKernelWeightRequirement(
                    role: .affine,
                    acceptedLayouts: [.affineU4RowMajor(groupSize: 64)]
                ),
            ],
            rowTile: 16,
            batchTile: nil,
            threadgroupWidth: 96
        )
        let route = SmeltPlannedKernelRoute(
            candidate: candidate,
            capability: capability
        )

        let geometry = route.affineMatvecResidualAddLaunchGeometry()
        #expect(geometry?.rowTile == 16)
        #expect(geometry?.threadgroupWidth == 96)
        #expect(geometry?.gridWidth(rows: candidate.shape.rows) == 32)
        #expect(
            route.affineMatvecResidualAddLaunchGeometry(expectedShape: candidate.shape) == geometry
        )
        #expect(route.affineMatvecResidualAddLaunchGeometry(
            expectedShape: SmeltKernelShape(rows: 768, cols: 1_024, groupSize: 64)
        ) == nil)

        let mismatchedCapability = SmeltKernelCapability(
            id: "fused_affine_matvec_add_c1024_r768_g64_route_owned_probe",
            phase: .decode,
            operation: .affineMatvecResidualAdd,
            shape: SmeltKernelShape(rows: 768, cols: 1_024, groupSize: 64),
            source: .packageLocalGenerated,
            weightRequirements: capability.weightRequirements,
            rowTile: 16,
            batchTile: nil,
            threadgroupWidth: 96
        )
        let mismatchedRoute = SmeltPlannedKernelRoute(
            candidate: candidate,
            capability: mismatchedCapability
        )

        #expect(mismatchedRoute.affineMatvecResidualAddLaunchGeometry() == nil)
    }

    @Test("Planned fusion requires matching generated residual consumer")
    func plannedFusionRequiresMatchingGeneratedResidualConsumer() throws {
        let planner = SmeltFusionPlanner.planned(
            kernelPlan: SmeltKernelPlanner.plan(for: .qwen35_0_8B)
        )
        let layerContext = SmeltKernelLayerConsumerContext(
            ir: .qwen35_0_8B,
            layerIndex: 0,
            layerType: .delta
        )
        let wrongConsumer = try #require(SmeltKernelConsumerNaming.candidate(
            kind: .ffnGateUpPrefill,
            context: layerContext
        ))
        let rewrite = planner.rewrite(window: [
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvec,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 13, index: 3),
                    SmeltBufferBinding(slot: 14, index: 4),
                ],
                constants: [
                    SmeltConstantBinding(expression: "1024", type: .uint32, index: 5),
                ],
                dispatch: .threadgroups(
                    width: 128,
                    height: 1,
                    depth: 1,
                    tgWidth: 64,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                comment: "Wrong planned consumer",
                plannedKernelCandidate: wrongConsumer,
                fcCols: 3_584,
                fcGroupSize: 64
            )),
            .dispatch(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 14, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "1024", type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: 1_024,
                    height: 1,
                    depth: 1,
                    tgWidth: 1_024,
                    tgHeight: 1,
                    tgDepth: 1
                )
            )),
        ][...])

        #expect(rewrite?.kind == .genericKernel)
        guard case .dispatch(let fused)? = rewrite?.producedOps.first else {
            Issue.record("Expected generic fallback dispatch rewrite")
            return
        }
        #expect(fused.pipeline == .fusedAffineMatvecAdd)
        #expect(fused.pipelineNameOverride == nil)
        #expect(fused.constants.map(\.expression) == ["1024"])
        #expect(fused.fcCols == 3_584)
        #expect(fused.fcGroupSize == 64)
    }

    @Test("Planned fusion falls back for unplanned generated residual shapes")
    func plannedFusionFallsBackForUnplannedGeneratedResidualShapes() {
        let planner = SmeltFusionPlanner.planned(
            kernelPlan: SmeltKernelPlanner.plan(for: .qwen35_0_8B)
        )
        let rewrite = planner.rewrite(window: [
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvec,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 13, index: 3),
                    SmeltBufferBinding(slot: 14, index: 4),
                ],
                constants: [
                    SmeltConstantBinding(expression: "2048", type: .uint32, index: 5),
                ],
                dispatch: .threadgroups(
                    width: 256,
                    height: 1,
                    depth: 1,
                    tgWidth: 64,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                comment: "Unplanned down projection",
                fcCols: 11_008,
                fcGroupSize: 64
            )),
            .dispatch(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 14, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "2048", type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: 2_048,
                    height: 1,
                    depth: 1,
                    tgWidth: 1_024,
                    tgHeight: 1,
                    tgDepth: 1
                )
            )),
        ][...])

        #expect(rewrite?.kind == .genericKernel)
        guard case .dispatch(let fused)? = rewrite?.producedOps.first else {
            Issue.record("Expected generic fallback dispatch rewrite")
            return
        }
        #expect(fused.pipeline == .fusedAffineMatvecAdd)
        #expect(fused.pipelineNameOverride == nil)
        #expect(fused.constants.map(\.expression) == ["2048"])
        #expect(fused.fcCols == 11_008)
        #expect(fused.fcGroupSize == 64)
    }

    @Test("Generated affine residual capability carries layout contract")
    func generatedAffineResidualCapabilityCarriesLayoutContract() throws {
        let capability = try #require(
            SmeltKernelCapabilityRegistry.generatedCapability(
                operation: .affineMatvecResidualAdd,
                shape: SmeltKernelShape(rows: 2_048, cols: 11_008, groupSize: 64)
            )
        )

        #expect(capability.id == "fused_affine_matvec_add_c11008_r2048_g64_rows4")
        #expect(capability.phase == .decode)
        #expect(capability.operation == .affineMatvecResidualAdd)
        #expect(capability.shape == SmeltKernelShape(rows: 2_048, cols: 11_008, groupSize: 64))
        #expect(capability.rowTile == 4)
        #expect(capability.threadgroupWidth == 64)
        #expect(capability.weightRequirements == [
            SmeltKernelWeightRequirement(
                role: .affine,
                acceptedLayouts: [.affineU4RowMajor(groupSize: 64)]
            ),
        ])
    }

    @Test("Generated affine prefill capability carries layout contract")
    func generatedAffinePrefillCapabilityCarriesLayoutContract() throws {
        let capability = try #require(
            SmeltKernelCapabilityRegistry.generatedCapability(
                operation: .affineMatvecPrefillFull,
                shape: SmeltKernelShape(rows: 1_024, cols: 3_584, groupSize: 64)
            )
        )

        #expect(capability.id == "affine_matvec_c3584_r1024_g64_batched_full")
        #expect(capability.phase == .prefill)
        #expect(capability.operation == .affineMatvecPrefillFull)
        #expect(capability.shape == SmeltKernelShape(rows: 1_024, cols: 3_584, groupSize: 64))
        #expect(capability.rowTile == 32)
        #expect(capability.batchTile == 16)
        #expect(capability.threadgroupWidth == 128)
        #expect(capability.weightRequirements == [
            SmeltKernelWeightRequirement(
                role: .affine,
                acceptedLayouts: [.affineU4RowMajor(groupSize: 64)]
            ),
        ])
    }

    @Test("Generated fused gate-up prefill capability carries layout contract")
    func generatedFusedGateUpPrefillCapabilityCarriesLayoutContract() throws {
        let capability = try #require(
            SmeltKernelCapabilityRegistry.generatedCapability(
                operation: .fusedGateUpSwigluPrefillFull,
                shape: SmeltKernelShape(rows: 3_584, cols: 1_024, groupSize: 64)
            )
        )

        #expect(capability.id == "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full")
        #expect(capability.phase == .prefill)
        #expect(capability.operation == .fusedGateUpSwigluPrefillFull)
        #expect(capability.shape == SmeltKernelShape(rows: 3_584, cols: 1_024, groupSize: 64))
        #expect(capability.rowTile == 32)
        #expect(capability.batchTile == 16)
        #expect(capability.threadgroupWidth == 128)
        #expect(capability.weightRequirements == [
            SmeltKernelWeightRequirement(
                role: .gate,
                acceptedLayouts: [.affineU4RowMajor(groupSize: 64)]
            ),
            SmeltKernelWeightRequirement(
                role: .up,
                acceptedLayouts: [.affineU4RowMajor(groupSize: 64)]
            ),
        ])
    }

    @Test("Generated capability registry resolves operation and shape")
    func generatedCapabilityRegistryResolvesOperationAndShape() throws {
        let residual = try #require(SmeltKernelCapabilityRegistry.generatedCapability(
            operation: .affineMatvecResidualAdd,
            shape: SmeltKernelShape(rows: 1_024, cols: 3_584, groupSize: 64)
        ))
        #expect(residual.id == "fused_affine_matvec_add_c3584_r1024_g64_rows4")

        let prefill = try #require(SmeltKernelCapabilityRegistry.generatedCapability(
            operation: .affineMatvecPrefillFull,
            shape: SmeltKernelShape(rows: 1_024, cols: 3_584, groupSize: 64)
        ))
        #expect(prefill.id == "affine_matvec_c3584_r1024_g64_batched_full")

        let gateUp = try #require(SmeltKernelCapabilityRegistry.generatedCapability(
            operation: .fusedGateUpSwigluPrefillFull,
            shape: SmeltKernelShape(rows: 3_584, cols: 1_024, groupSize: 64)
        ))
        #expect(gateUp.id == "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full")

        #expect(SmeltKernelCapabilityRegistry.generatedCapability(
            operation: .affineMatvecResidualAdd,
            shape: SmeltKernelShape(rows: 1_023, cols: 3_584, groupSize: 64)
        ) == nil)
    }

    @Test("Kernel planner exposes model-derived capabilities and consumers")
    func kernelPlannerExposesModelDerivedCapabilitiesAndConsumers() throws {
        let kernelPlan = SmeltKernelPlanner.plan(for: .qwen35_0_8B)
        let plannedUses = kernelPlan.generatedUses

        #expect(plannedUses.count == 78)
        #expect(kernelPlan.generatedCandidateCount == 78)
        #expect(kernelPlan.unsupportedGeneratedCandidateCount == 0)
        #expect(plannedUses.filter { $0.capability.phase == .decode }.count == 30)
        #expect(plannedUses.filter { $0.capability.phase == .prefill }.count == 48)
        #expect(!plannedUses.contains {
            $0.weights.contains { $0.weightName.contains("_q_proj_weight") }
        })

        let layer0GateUp = try #require(plannedUses.first {
            $0.consumerID == "layers_0_mlp.gate_up.prefill"
        })
        #expect(layer0GateUp.weights == [
            SmeltPlannedKernelWeight(
                weightName: "layers_0_mlp_gate_proj_weight",
                role: .gate
            ),
            SmeltPlannedKernelWeight(
                weightName: "layers_0_mlp_up_proj_weight",
                role: .up
            ),
        ])

        let capabilities = kernelPlan.generatedCapabilities
        let names = Set(capabilities.map(\.id))

        #expect(names.contains("fused_affine_matvec_add_c3584_r1024_g64_rows4"))
        #expect(names.contains("fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full"))
        #expect(names.contains("affine_matvec_c3584_r1024_g64_batched_full"))
        #expect(kernelPlan.generatedCapabilityNameSet == names)
        #expect(kernelPlan.emittedGeneratedCapabilityNameSet == [
            "fused_affine_matvec_add_c3584_r1024_g64_rows4",
        ])
        #expect(kernelPlan.emittedGeneratedCapabilities.allSatisfy {
            $0.requiresPackageLocalGeneratedSource
        })

        let gateUp = try #require(capabilities.first {
            $0.operation == .fusedGateUpSwigluPrefillFull
                && $0.shape == SmeltKernelShape(rows: 3_584, cols: 1_024, groupSize: 64)
        })
        #expect(gateUp.weightRequirements.map(\.role) == [.gate, .up])
        #expect(gateUp.weightRequirements.allSatisfy {
            $0.acceptedLayouts == [.affineU4RowMajor(groupSize: 64)]
        })

        let unsupportedBase = SmeltModelIR.qwen35_0_8B
        let unsupportedIR = SmeltModelIR(
            modelName: unsupportedBase.modelName,
            config: unsupportedBase.config,
            layerPattern: unsupportedBase.layerPattern,
            quantization: SmeltQuantizationConfig(
                strategy: unsupportedBase.quantization.strategy,
                groupSize: 48,
                excludePatterns: unsupportedBase.quantization.excludePatterns,
                quantizeEmbedding: unsupportedBase.quantization.quantizeEmbedding,
                turboQuantHPatterns: unsupportedBase.quantization.turboQuantHPatterns,
                preserveNativePatterns: unsupportedBase.quantization.preserveNativePatterns
            ),
            loading: unsupportedBase.loading,
            compilation: unsupportedBase.compilation,
            prefill: unsupportedBase.prefill,
            inference: unsupportedBase.inference
        )
        let unsupportedPlan = SmeltKernelPlanner.plan(for: unsupportedIR)

        #expect(unsupportedPlan.generatedUses.isEmpty)
        #expect(unsupportedPlan.generatedCandidateCount == 78)
        #expect(
            unsupportedPlan.unsupportedGeneratedCandidateCount
                == unsupportedPlan.generatedCandidateCount
        )
        let unsupportedReports = unsupportedPlan.unsupportedKernelCandidateReports()
        #expect(unsupportedReports.count == 78)
        #expect(unsupportedReports.allSatisfy { $0.reason == "no_generated_capability" })
        #expect(unsupportedReports.contains(SmeltUnsupportedKernelCandidateReport(
            consumerID: "layers_0_mlp.down_proj.residual.decode",
            consumerKind: "ffnDownResidualDecode",
            phase: "decode",
            operation: "affineMatvecResidualAdd",
            rows: 1_024,
            cols: 3_584,
            groupSize: 48,
            weights: [
                SmeltPlannedKernelWeightReport(
                    weightName: "layers_0_mlp_down_proj_weight",
                    role: "affine"
                ),
            ],
            reason: "no_generated_capability"
        )))
    }

    @Test("Kernel planner honors disabled generated-kernel policy")
    func kernelPlannerHonorsDisabledGeneratedKernelPolicy() throws {
        let base = SmeltModelIR.qwen35_0_8B
        let ir = SmeltModelIR(
            modelName: base.modelName,
            config: base.config,
            layerPattern: base.layerPattern,
            quantization: base.quantization,
            loading: base.loading,
            compilation: SmeltCompilationConfig(generatedKernels: .disabled),
            prefill: base.prefill,
            inference: base.inference
        )
        let layout = SmeltWeightLayout.computeLayout(from: ir)
        let plan = try SmeltCompiler.planCompilation(ir: ir, weightLayout: layout)

        #expect(plan.kernelPlan.isEmpty)
        #expect(plan.weightStoragePlan.plannedUses.count == 187)
        #expect(plan.weightStoragePlan.decisions.count == 187)
        #expect(plan.report.kernelGeneration == "disabled")
        #expect(plan.report.generatedKernelConsumerKinds == [])
        #expect(plan.report.weightLayoutPolicy == "memory_neutral")
        #expect(plan.generatedSourceFunctionNames.isEmpty)
        #expect(plan.generatedMetalSourceSuffix.isEmpty)
    }

    @Test("Kernel planner honors generated consumer kind allow-list")
    func kernelPlannerHonorsGeneratedConsumerKindAllowList() throws {
        let base = SmeltModelIR.qwen35_0_8B
        let ir = SmeltModelIR(
            modelName: base.modelName,
            config: base.config,
            layerPattern: base.layerPattern,
            quantization: base.quantization,
            loading: base.loading,
            compilation: SmeltCompilationConfig(
                generatedKernelConsumerKinds: [.ffnDownPrefill]
            ),
            prefill: base.prefill,
            inference: base.inference
        )
        let layout = SmeltWeightLayout.computeLayout(from: ir)
        let plan = try SmeltCompiler.planCompilation(ir: ir, weightLayout: layout)
        let report = plan.report

        #expect(report.kernelGeneration == "auto")
        #expect(report.generatedKernelConsumerKinds == ["ffnDownPrefill"])
        #expect(report.plannedKernelCandidates == 24)
        #expect(report.plannedKernelUses == 24)
        #expect(report.unsupportedKernelCandidates == 0)
        #expect(report.generatedKernels == 1)
        #expect(report.emittedGeneratedKernels == 0)
        #expect(report.plannedGeneratedKernelNames == [
            "affine_matvec_c3584_r1024_g64_batched_full",
        ])
        #expect(Set(report.plannedKernelConsumers.compactMap(\.consumerKind)) == [
            "ffnDownPrefill",
        ])
        #expect(Set(report.plannedWeightConsumers.compactMap(\.consumerKind)) == [
            "ffnDownPrefill",
            "storageRead",
        ])
        let plannedKernelConsumerIDs = Set(report.plannedKernelConsumers.map(\.consumerID))
        #expect(plannedKernelConsumerIDs.contains(
            "layers_0_mlp.down_proj.prefill"
        ))
        #expect(!plannedKernelConsumerIDs.contains(
            "layers_0_mlp.down_proj.residual.decode"
        ))
        #expect(!plannedKernelConsumerIDs.contains(
            "layers_0_mlp.gate_up.prefill"
        ))
    }

    @Test("Kernel plan index resolves exact planned consumers")
    func kernelPlanIndexResolvesExactPlannedConsumers() throws {
        let plan = SmeltKernelPlanner.plan(for: .qwen35_0_8B)
        let layerContext = SmeltKernelLayerConsumerContext(
            ir: .qwen35_0_8B,
            layerIndex: 0,
            layerType: .delta
        )
        let downCandidate = try #require(SmeltKernelConsumerNaming.candidate(
            kind: .ffnDownPrefill,
            context: layerContext
        ))

        let downRoute = try #require(plan.route(for: downCandidate))
        #expect(downRoute.candidate == downCandidate)
        #expect(downRoute.capability.id == "affine_matvec_c3584_r1024_g64_batched_full")
        #expect(plan.route(
            kind: .ffnDownPrefill,
            context: layerContext
        ) == downRoute)
        #expect(plan.route(
            kind: .ffnDownResidualDecode,
            context: layerContext
        )?.capability.id == "fused_affine_matvec_add_c3584_r1024_g64_rows4")
        let wrongGroupContext = SmeltKernelLayerConsumerContext(
            ir: .qwen35_0_8B,
            layerIndex: 0,
            layerType: .delta,
            groupSize: 128
        )
        #expect(plan.route(
            kind: .ffnDownPrefill,
            context: wrongGroupContext
        ) == nil)
        #expect(downCandidate.consumerID == "layers_0_mlp.down_proj.prefill")
        #expect(downCandidate.operation == .affineMatvecPrefillFull)
        #expect(downCandidate.shape == SmeltKernelShape(
            rows: 1_024,
            cols: 3_584,
            groupSize: 64
        ))
        #expect(downCandidate.weights == [
            SmeltPlannedKernelWeight(
                weightName: "layers_0_mlp_down_proj_weight",
                role: .affine
            ),
        ])

        let wrongRoleCandidate = SmeltPlannedKernelCandidate(
            consumerID: "layers_0_mlp.down_proj.prefill",
            operation: .affineMatvecPrefillFull,
            shape: SmeltKernelShape(rows: 1_024, cols: 3_584, groupSize: 64),
            weights: [
                SmeltPlannedKernelWeight(
                    weightName: "layers_0_mlp_down_proj_weight",
                    role: .gate
                ),
            ]
        )
        #expect(plan.route(for: wrongRoleCandidate) == nil)
    }

    @Test("Layer consumer context derives kernel candidates from graph facts")
    func layerConsumerContextDerivesKernelCandidatesFromGraphFacts() {
        let attention = SmeltAttentionConfig(
            qHeads: 16,
            kvHeads: 2,
            headDim: 128,
            gatedQ: false,
            qkvBias: true
        )
        let context = SmeltKernelLayerConsumerContext(
            layerIndex: 0,
            hiddenSize: 2_048,
            ffnDim: 11_008,
            groupSize: 64,
            blockTopology: .standard,
            attention: attention,
            attentionHasOwnKV: true,
            prefillEngine: "metal",
            ffnActivation: .swiglu
        )

        #expect(SmeltKernelConsumerNaming.candidateKinds(for: context) == [
            .qProjBiasDecode,
            .kProjBiasDecode,
            .vProjBiasDecode,
            .kvProjBiasDecode,
            .attentionOutputResidualDecode,
            .ffnDownResidualDecode,
            .ffnGateUpPrefill,
            .ffnDownPrefill,
        ])
        let candidates = SmeltKernelConsumerNaming.candidates(for: context)
        #expect(candidates.map(\.kind) == [
            .qProjBiasDecode,
            .kProjBiasDecode,
            .vProjBiasDecode,
            .kvProjBiasDecode,
            .attentionOutputResidualDecode,
            .ffnDownResidualDecode,
            .ffnGateUpPrefill,
            .ffnDownPrefill,
        ])
        #expect(candidates.map(\.shape) == [
            SmeltKernelShape(rows: 2_048, cols: 2_048, groupSize: 64),
            SmeltKernelShape(rows: 256, cols: 2_048, groupSize: 64),
            SmeltKernelShape(rows: 256, cols: 2_048, groupSize: 64),
            SmeltKernelShape(rows: 256, cols: 2_048, groupSize: 64),
            SmeltKernelShape(rows: 2_048, cols: 2_048, groupSize: 64),
            SmeltKernelShape(rows: 2_048, cols: 11_008, groupSize: 64),
            SmeltKernelShape(rows: 11_008, cols: 2_048, groupSize: 64),
            SmeltKernelShape(rows: 2_048, cols: 11_008, groupSize: 64),
        ])
    }

    @Test("Layer consumer context omits unavailable generated consumers")
    func layerConsumerContextOmitsUnavailableGeneratedConsumers() {
        let context = SmeltKernelLayerConsumerContext(
            layerIndex: 0,
            hiddenSize: 2_048,
            ffnDim: 11_008,
            groupSize: 64,
            blockTopology: .standard,
            attention: SmeltAttentionConfig(
                qHeads: 16,
                kvHeads: 2,
                headDim: 128,
                gatedQ: false,
                qkvBias: true,
                externalKV: true
            ),
            attentionHasOwnKV: false,
            prefillEngine: nil,
            ffnActivation: .geglu
        )

        #expect(SmeltKernelConsumerNaming.candidateKinds(for: context) == [
            .qProjBiasDecode,
            .attentionOutputResidualDecode,
            .ffnDownResidualDecode,
        ])
        #expect(SmeltKernelConsumerNaming.candidate(
            kind: .kProjBiasDecode,
            context: context
        ) == nil)
        #expect(SmeltKernelConsumerNaming.candidate(
            kind: .ffnDownPrefill,
            context: context
        ) == nil)
    }

    @Test("Weight layout planner maps capabilities to concrete weight consumers")
    func weightLayoutPlannerMapsCapabilitiesToConcreteWeightConsumers() throws {
        let kernelPlan = SmeltKernelPlanner.plan(for: .qwen35_0_8B)
        let generatedUses = kernelPlan.plannedWeightUses
        let uses = SmeltWeightLayoutPlanner.plannedWeightUses(
            for: .qwen35_0_8B,
            kernelPlan: kernelPlan
        )

        #expect(generatedUses.count == 102)
        #expect(generatedUses.filter { $0.capability.phase == .decode }.count == 30)
        #expect(generatedUses.filter { $0.capability.phase == .prefill }.count == 72)
        #expect(!generatedUses.contains { $0.weightName.contains("_q_proj_weight") })

        #expect(uses.count == 211)
        #expect(uses.filter { $0.capability.phase == .decode }.count == 30)
        #expect(uses.filter { $0.capability.phase == .prefill }.count == 72)
        #expect(uses.filter { $0.capability.phase == .storage }.count == 109)

        let layer0Down = uses.filter {
            $0.weightName == "layers_0_mlp_down_proj_weight"
        }
        #expect(Set(layer0Down.map(\.consumerID)) == [
            "layers_0_mlp.down_proj.residual.decode",
            "layers_0_mlp.down_proj.prefill",
        ])
        #expect(Set(layer0Down.map(\.capability.id)) == [
            "fused_affine_matvec_add_c3584_r1024_g64_rows4",
            "affine_matvec_c3584_r1024_g64_batched_full",
        ])
        #expect(layer0Down.allSatisfy {
            $0.acceptedLayouts == [.affineU4RowMajor(groupSize: 64)]
        })

        let layer0GateUp = uses.filter {
            $0.consumerID == "layers_0_mlp.gate_up.prefill"
        }
        #expect(Set(layer0GateUp.map(\.weightName)) == [
            "layers_0_mlp_gate_proj_weight",
            "layers_0_mlp_up_proj_weight",
        ])
        #expect(Set(layer0GateUp.map(\.weightRole)) == [.gate, .up])
        #expect(layer0GateUp.allSatisfy {
            $0.capability.id == "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full"
        })

        let attentionOutputUses = uses.filter {
            $0.weightName.hasSuffix("_self_attn_o_proj_weight")
        }
        #expect(attentionOutputUses.count == 6)
        #expect(attentionOutputUses.allSatisfy {
            $0.capability.operation == .affineMatvecResidualAdd
        })

        let qProjStorage = try #require(uses.first {
            $0.weightName == "layers_3_self_attn_q_proj_weight"
        })
        #expect(qProjStorage.consumerID == "layers_3_self_attn_q_proj_weight.storage")
        #expect(qProjStorage.consumerKind == .storageRead)
        #expect(qProjStorage.capability.phase == .storage)
        #expect(qProjStorage.capability.operation == .affineStorageRead)
        #expect(qProjStorage.acceptedLayouts == [.affineU4RowMajor(groupSize: 64)])
    }

    @Test("Weight layout planner skips non-affine generated consumers")
    func weightLayoutPlannerSkipsNonAffineGeneratedConsumers() {
        #expect(SmeltKernelPlanner.plan(for: .qwen35_2B).plannedWeightUses.isEmpty)
        #expect(SmeltWeightLayoutPlanner.plannedWeightUses(for: .qwen35_2B).isEmpty)
    }

    @Test("Weight storage plan chooses one memory-neutral layout per planned weight")
    func weightStoragePlanChoosesOneMemoryNeutralLayoutPerPlannedWeight() throws {
        let layout = SmeltWeightLayout.computeLayout(from: .qwen35_0_8B)
        let kernelPlan = SmeltKernelPlanner.plan(for: .qwen35_0_8B)
        let plan = SmeltWeightLayoutPlanner.storagePlan(for: .qwen35_0_8B, entries: layout)
        let planFromKernelPlan = SmeltWeightLayoutPlanner.storagePlan(
            entries: layout,
            kernelPlan: kernelPlan
        )

        #expect(planFromKernelPlan == plan)
        #expect(plan.isMemoryNeutral)
        #expect(plan.validationFailure(policy: .memoryNeutral) == nil)
        #expect(plan.duplicateLayoutWeightNames.isEmpty)
        #expect(plan.issues.isEmpty)
        #expect(plan.decisions.count == 187)
        #expect(plan.decisions.allSatisfy { !$0.requiresDuplicateLayout })
        #expect(plan.decisions.allSatisfy {
            $0.currentLayout == $0.selectedLayout
        })

        let down = try #require(plan.decision(for: "layers_0_mlp_down_proj_weight"))
        #expect(down.selectedLayout == .affineU4RowMajor(groupSize: 64))
        #expect(Set(down.uses.map(\.consumerID)) == [
            "layers_0_mlp.down_proj.residual.decode",
            "layers_0_mlp.down_proj.prefill",
        ])

        let gate = try #require(plan.decision(for: "layers_0_mlp_gate_proj_weight"))
        #expect(gate.uses.map(\.weightRole) == [.gate])
        #expect(gate.selectedLayout == .affineU4RowMajor(groupSize: 64))

        let qProj = try #require(plan.decision(for: "layers_3_self_attn_q_proj_weight"))
        #expect(qProj.uses.map(\.consumerID) == [
            "layers_3_self_attn_q_proj_weight.storage",
        ])
        #expect(qProj.selectedLayout == .affineU4RowMajor(groupSize: 64))
    }

    @Test("Planned weight layout carries selected storage decisions")
    func plannedWeightLayoutCarriesSelectedStorageDecisions() throws {
        let kernelPlan = SmeltKernelPlanner.plan(for: .qwen35_0_8B)
        let rawLayout = SmeltWeightLayout.computeLayout(from: .qwen35_0_8B)
        let plannedLayout = SmeltWeightLayoutPlanner.plannedLayout(
            entries: rawLayout,
            kernelPlan: kernelPlan
        )

        #expect(plannedLayout.entries.map(\.name) == rawLayout.map(\.name))
        #expect(plannedLayout.entries.map(\.offset) == rawLayout.map(\.offset))
        #expect(plannedLayout.entries.map(\.dtype) == rawLayout.map(\.dtype))
        #expect(plannedLayout.storagePlan.isMemoryNeutral)
        #expect(plannedLayout.storagePlan.decisions.count == 187)

        let downDecision = try #require(
            plannedLayout.storagePlan.decision(for: "layers_0_mlp_down_proj_weight")
        )
        #expect(downDecision.selectedLayout == .affineU4RowMajor(groupSize: 64))
        #expect(Set(downDecision.uses.map(\.consumerID)) == [
            "layers_0_mlp.down_proj.residual.decode",
            "layers_0_mlp.down_proj.prefill",
        ])

        let downEntry = try #require(plannedLayout.entries.first {
            $0.name == "layers_0_mlp_down_proj_weight"
        })
        #expect(downEntry.dtype == .affineU4)
        #expect(downEntry.groupSize == 64)
    }

    @Test("Weight storage plan reports missing planned weight entries")
    func weightStoragePlanReportsMissingPlannedWeightEntries() {
        let layout = SmeltWeightLayout.computeLayout(from: .qwen35_0_8B)
            .filter { $0.name != "layers_0_mlp_down_proj_weight" }
        let plan = SmeltWeightLayoutPlanner.storagePlan(for: .qwen35_0_8B, entries: layout)

        #expect(!plan.isMemoryNeutral)
        #expect(plan.issues.contains(SmeltWeightStorageIssue(
            weightName: "layers_0_mlp_down_proj_weight",
            kind: .missingWeightEntry,
            consumers: [
                SmeltPlannedWeightStorageDecisionConsumerReport(
                    consumerID: "layers_0_mlp.down_proj.residual.decode",
                    consumerKind: "ffnDownResidualDecode"
                ),
                SmeltPlannedWeightStorageDecisionConsumerReport(
                    consumerID: "layers_0_mlp.down_proj.prefill",
                    consumerKind: "ffnDownPrefill"
                ),
            ]
        )))
        #expect(plan.issueReports().contains(SmeltPlannedWeightStorageIssueReport(
            weightName: "layers_0_mlp_down_proj_weight",
            kind: "missingWeightEntry",
            consumers: [
                SmeltPlannedWeightStorageDecisionConsumerReport(
                    consumerID: "layers_0_mlp.down_proj.residual.decode",
                    consumerKind: "ffnDownResidualDecode"
                ),
                SmeltPlannedWeightStorageDecisionConsumerReport(
                    consumerID: "layers_0_mlp.down_proj.prefill",
                    consumerKind: "ffnDownPrefill"
                ),
            ]
        )))
        #expect(plan.validationFailure(policy: .memoryNeutral) == .illegalStorage(plan.issues))

        let plannedLayout = SmeltWeightLayoutPlanner.plannedLayout(
            entries: layout,
            kernelPlan: SmeltKernelPlanner.plan(for: .qwen35_0_8B)
        )
        #expect(plannedLayout.entries.map(\.name) == layout.map(\.name))
        #expect(plannedLayout.storagePlan.issues == plan.issues)
    }

    @Test("Weight storage plan reports duplicate layout policy failure")
    func weightStoragePlanReportsDuplicateLayoutPolicyFailure() {
        let storagePlan = SmeltWeightStoragePlan(
            plannedUses: [],
            decisions: [
                SmeltWeightStorageDecision(
                    weightName: "duplicate_probe_weight",
                    currentLayout: .affineU4RowMajor(groupSize: 64),
                    selectedLayout: .affineU4RowMajor(groupSize: 64),
                    uses: [],
                    requiresDuplicateLayout: true
                ),
            ],
            issues: []
        )

        #expect(!storagePlan.isMemoryNeutral)
        #expect(storagePlan.duplicateLayoutWeightNames == ["duplicate_probe_weight"])
        #expect(storagePlan.validationFailure(policy: .memoryNeutral) == .duplicatePhysicalStorage([
            "duplicate_probe_weight",
        ]))
    }

    @Test("Compiler accepts memory-neutral kernel-planned storage")
    func compilerAcceptsMemoryNeutralKernelPlannedStorage() throws {
        let layout = SmeltWeightLayout.computeLayout(from: .qwen35_0_8B)
        let plan = try SmeltCompiler.validateWeightStoragePlan(
            ir: .qwen35_0_8B,
            weightLayout: layout
        )

        #expect(plan.isMemoryNeutral)
        #expect(plan.decisions.count == 187)
    }

    @Test("Compiler planning returns shared kernel and weight storage plan")
    func compilerPlanningReturnsSharedKernelAndWeightStoragePlan() throws {
        let layout = SmeltWeightLayout.computeLayout(from: .qwen35_0_8B)
        let plan = try SmeltCompiler.planCompilation(
            ir: .qwen35_0_8B,
            weightLayout: layout
        )
        let irPlan = try SmeltCompiler.planCompilation(ir: .qwen35_0_8B)
        let storagePlan = try SmeltCompiler.validateWeightStoragePlan(
            ir: .qwen35_0_8B,
            weightLayout: layout
        )

        #expect(plan.bufferPlan == buildBufferPlan(from: .qwen35_0_8B))
        #expect(irPlan.bufferPlan == plan.bufferPlan)
        #expect(plan.kernelPlan.generatedUses.count == 78)
        #expect(plan.kernelPlan.generatedCapabilities.count == 4)
        #expect(plan.plannedWeightEntries == layout)
        #expect(irPlan.plannedWeightEntries == plan.plannedWeightEntries)
        #expect(storagePlan == plan.weightStoragePlan)
        #expect(plan.weightStoragePlan.plannedUses.count == 211)
        #expect(plan.weightStoragePlan.isMemoryNeutral)
        #expect(plan.weightStoragePlan.decisions.count == 187)

        let report = plan.report
        #expect(report.plannedBufferSlots == plan.bufferPlan.slotCount)
        #expect(report.plannedActivationBytes == plan.bufferPlan.totalActivationBytes)
        #expect(report.plannedKernelCandidates == 78)
        #expect(report.unsupportedKernelCandidates == 0)
        #expect(report.unsupportedKernelCandidateRecords.isEmpty)
        #expect(report.plannedKernelUses == 78)
        #expect(report.plannedKernelConsumers.count == 78)
        #expect(Set(report.plannedKernelConsumers.compactMap(\.consumerKind)) == [
            "attentionOutputResidualDecode",
            "ffnDownResidualDecode",
            "ffnGateUpPrefill",
            "ffnDownPrefill",
        ])
        #expect(report.plannedKernelConsumers.contains(SmeltPlannedKernelConsumerReport(
            consumerID: "layers_0_mlp.down_proj.residual.decode",
            consumerKind: "ffnDownResidualDecode",
            capabilityName: "fused_affine_matvec_add_c3584_r1024_g64_rows4",
            phase: "decode",
            operation: "affineMatvecResidualAdd",
            rows: 1_024,
            cols: 3_584,
            groupSize: 64,
            weights: [
                SmeltPlannedKernelWeightReport(
                    weightName: "layers_0_mlp_down_proj_weight",
                    role: "affine"
                ),
            ]
        )))
        #expect(report.plannedKernelConsumers.contains(SmeltPlannedKernelConsumerReport(
            consumerID: "layers_0_mlp.gate_up.prefill",
            consumerKind: "ffnGateUpPrefill",
            capabilityName: "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full",
            phase: "prefill",
            operation: "fusedGateUpSwigluPrefillFull",
            rows: 3_584,
            cols: 1_024,
            groupSize: 64,
            weights: [
                SmeltPlannedKernelWeightReport(
                    weightName: "layers_0_mlp_gate_proj_weight",
                    role: "gate"
                ),
                SmeltPlannedKernelWeightReport(
                    weightName: "layers_0_mlp_up_proj_weight",
                    role: "up"
                ),
            ]
        )))
        #expect(report.generatedKernels == 4)
        #expect(report.emittedGeneratedKernels == 1)
        #expect(report.plannedGeneratedKernelCapabilities.count == 4)
        #expect(report.plannedGeneratedKernelCapabilities.contains(
            SmeltGeneratedKernelCapabilityReport(
                capabilityName: "fused_affine_matvec_add_c3584_r1024_g64_rows4",
                phase: "decode",
                operation: "affineMatvecResidualAdd",
                rows: 1_024,
                cols: 3_584,
                groupSize: 64,
                sourceKind: "package_local_generated",
                emittedGeneratedSource: true,
                sourceTemplate: "affineMatvecResidualAddFixedRows4",
                weightRequirements: [
                    SmeltGeneratedKernelWeightRequirementReport(
                        role: "affine",
                        acceptedLayouts: ["affine_u4_row_major_g64"]
                    ),
                ],
                rowTile: 4,
                batchTile: nil,
                threadgroupWidth: 64
            )
        ))
        #expect(report.plannedGeneratedKernelCapabilities.contains(
            SmeltGeneratedKernelCapabilityReport(
                capabilityName: "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full",
                phase: "prefill",
                operation: "fusedGateUpSwigluPrefillFull",
                rows: 3_584,
                cols: 1_024,
                groupSize: 64,
                sourceKind: "built_in_catalog",
                emittedGeneratedSource: false,
                sourceTemplate: "fusedGateUpSwigluPrefillFull",
                weightRequirements: [
                    SmeltGeneratedKernelWeightRequirementReport(
                        role: "gate",
                        acceptedLayouts: ["affine_u4_row_major_g64"]
                    ),
                    SmeltGeneratedKernelWeightRequirementReport(
                        role: "up",
                        acceptedLayouts: ["affine_u4_row_major_g64"]
                    ),
                ],
                rowTile: 32,
                batchTile: 16,
                threadgroupWidth: 128
            )
        ))
        #expect(report.plannedGeneratedKernelNames == [
            "fused_affine_matvec_add_c3584_r1024_g64_rows4",
            "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full",
            "affine_matvec_c3584_r1024_g64_batched_full",
            "fused_affine_matvec_add_c2048_r1024_g64_rows4",
        ])
        #expect(report.emittedGeneratedKernelNames == [
            "fused_affine_matvec_add_c3584_r1024_g64_rows4",
        ])
        #expect(report.plannedWeightUses == 211)
        #expect(report.plannedWeightNames.count == 187)
        #expect(report.plannedWeightNames.contains("layers_0_mlp_down_proj_weight"))
        #expect(report.plannedWeightNames.contains("layers_0_mlp_gate_proj_weight"))
        #expect(report.plannedWeightNames.contains("layers_3_self_attn_q_proj_weight"))
        #expect(report.plannedWeightConsumerIDs.count == 187)
        #expect(report.plannedWeightConsumerIDs.contains("layers_0_mlp.down_proj.residual.decode"))
        #expect(report.plannedWeightConsumerIDs.contains("layers_0_mlp.gate_up.prefill"))
        #expect(report.plannedWeightConsumerIDs.contains("layers_3_self_attn_q_proj_weight.storage"))
        #expect(report.plannedWeightConsumers.count == 211)
        #expect(Set(report.plannedWeightConsumers.compactMap(\.consumerKind)) == [
            "attentionOutputResidualDecode",
            "ffnDownResidualDecode",
            "ffnGateUpPrefill",
            "ffnDownPrefill",
            "storageRead",
        ])
        #expect(report.plannedWeightConsumers.contains(SmeltPlannedWeightConsumerReport(
            weightName: "layers_0_mlp_down_proj_weight",
            consumerID: "layers_0_mlp.down_proj.residual.decode",
            consumerKind: "ffnDownResidualDecode",
            capabilityName: "fused_affine_matvec_add_c3584_r1024_g64_rows4",
            weightRole: "affine",
            acceptedLayouts: ["affine_u4_row_major_g64"]
        )))
        #expect(report.plannedWeightConsumers.contains(SmeltPlannedWeightConsumerReport(
            weightName: "layers_0_mlp_gate_proj_weight",
            consumerID: "layers_0_mlp.gate_up.prefill",
            consumerKind: "ffnGateUpPrefill",
            capabilityName: "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full",
            weightRole: "gate",
            acceptedLayouts: ["affine_u4_row_major_g64"]
        )))
        #expect(report.plannedWeightConsumers.contains(SmeltPlannedWeightConsumerReport(
            weightName: "layers_0_mlp_up_proj_weight",
            consumerID: "layers_0_mlp.gate_up.prefill",
            consumerKind: "ffnGateUpPrefill",
            capabilityName: "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full",
            weightRole: "up",
            acceptedLayouts: ["affine_u4_row_major_g64"]
        )))
        #expect(report.plannedWeightConsumers.contains(SmeltPlannedWeightConsumerReport(
            weightName: "layers_3_self_attn_q_proj_weight",
            consumerID: "layers_3_self_attn_q_proj_weight.storage",
            consumerKind: "storageRead",
            capabilityName: "affine_u4_row_major_storage_r4096_c1024_g64",
            weightRole: "affine",
            acceptedLayouts: ["affine_u4_row_major_g64"]
        )))
        #expect(report.plannedWeightStorageDecisions == 187)
        #expect(report.plannedWeightStorageDecisionNames == report.plannedWeightNames)
        #expect(report.plannedWeightStorageDecisionRecords.count == 187)
        #expect(report.plannedWeightStorageDecisionRecords.contains(
            SmeltPlannedWeightStorageDecisionReport(
                weightName: "layers_0_mlp_down_proj_weight",
                currentLayout: "affine_u4_row_major_g64",
                selectedLayout: "affine_u4_row_major_g64",
                consumers: [
                    SmeltPlannedWeightStorageDecisionConsumerReport(
                        consumerID: "layers_0_mlp.down_proj.residual.decode",
                        consumerKind: "ffnDownResidualDecode"
                    ),
                    SmeltPlannedWeightStorageDecisionConsumerReport(
                        consumerID: "layers_0_mlp.down_proj.prefill",
                        consumerKind: "ffnDownPrefill"
                    ),
                ],
                requiresDuplicateLayout: false
            )
        ))
        #expect(report.plannedWeightStorageDecisionRecords.contains(
            SmeltPlannedWeightStorageDecisionReport(
                weightName: "layers_3_self_attn_q_proj_weight",
                currentLayout: "affine_u4_row_major_g64",
                selectedLayout: "affine_u4_row_major_g64",
                consumers: [
                    SmeltPlannedWeightStorageDecisionConsumerReport(
                        consumerID: "layers_3_self_attn_q_proj_weight.storage",
                        consumerKind: "storageRead"
                    ),
                ],
                requiresDuplicateLayout: false
            )
        ))
        #expect(report.duplicateWeightLayouts == 0)
        #expect(report.weightStorageIssues == 0)
        #expect(report.weightStorageIssueNames.isEmpty)
        #expect(report.memoryNeutralWeightStorage)
        #expect(report.kernelGeneration == "auto")
        #expect(report.generatedKernelConsumerKinds == [
            "qProjBiasDecode",
            "kProjBiasDecode",
            "vProjBiasDecode",
            "kvProjBiasDecode",
            "attentionOutputResidualDecode",
            "ffnDownResidualDecode",
            "ffnGateUpPrefill",
            "ffnDownPrefill",
        ])
        #expect(report.weightLayoutPolicy == "memory_neutral")

        let down = try #require(
            plan.weightStoragePlan.decision(for: "layers_0_mlp_down_proj_weight")
        )
        #expect(Set(down.uses.map(\.capability.id)) == [
            "fused_affine_matvec_add_c3584_r1024_g64_rows4",
            "affine_matvec_c3584_r1024_g64_batched_full",
        ])
    }

    @Test("Compiler exposes compilation plan report without package build")
    func compilerExposesCompilationPlanReportWithoutPackageBuild() throws {
        let report = try SmeltCompiler.compilationPlanReport(ir: .qwen35_0_8B)
        let bufferPlan = buildBufferPlan(from: .qwen35_0_8B)

        #expect(report.plannedBufferSlots == bufferPlan.slotCount)
        #expect(report.plannedActivationBytes == bufferPlan.totalActivationBytes)
        #expect(report.plannedKernelCandidates == 78)
        #expect(report.unsupportedKernelCandidates == 0)
        #expect(report.unsupportedKernelCandidateRecords.isEmpty)
        #expect(report.plannedKernelUses == 78)
        #expect(report.generatedKernels == 4)
        #expect(report.emittedGeneratedKernels == 1)
        #expect(report.plannedWeightUses == 211)
        #expect(report.plannedWeightStorageDecisions == 187)
        #expect(report.memoryNeutralWeightStorage)
        #expect(report.plannedWeightConsumerIDs.contains(
            "layers_0_mlp.down_proj.residual.decode"
        ))
        #expect(report.plannedWeightConsumerIDs.contains(
            "layers_3_self_attn_q_proj_weight.storage"
        ))

        let encoded = try JSONEncoder().encode(report)
        let decoded = try JSONDecoder().decode(
            SmeltCompilationPlanReport.self,
            from: encoded
        )
        #expect(decoded == report)
    }

    @Test("Compiler accepts generated pipeline names only from kernel plan")
    func compilerAcceptsGeneratedPipelineNamesOnlyFromKernelPlan() throws {
        let layout = SmeltWeightLayout.computeLayout(from: .qwen35_0_8B)
        let compilationPlan = try SmeltCompiler.planCompilation(
            ir: .qwen35_0_8B,
            weightLayout: layout
        )
        let plan = compilationPlan.kernelPlan
        let sourceNames = compilationPlan.generatedSourceFunctionNames
        #expect(sourceNames.contains("fused_affine_matvec_add_c3584_r1024_g64_rows4"))
        #expect(!sourceNames.contains("fused_affine_matvec_add_c2048_r1024_g64_rows4"))
        #expect(sourceNames.count == 1)
        #expect(!sourceNames.contains("affine_matvec_c3584_r1024_g64_batched_full"))
        #expect(!sourceNames.contains("fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full"))
        #expect(compilationPlan.generatedMetalSourceSuffix.contains(
            "kernel void fused_affine_matvec_add_c3584_r1024_g64_rows4("
        ))

        try compilationPlan.validateGeneratedPipelineNames(
            ["fused_affine_matvec_add_c3584_r1024_g64_rows4"],
            context: "test"
        )
        let layerContext = SmeltKernelLayerConsumerContext(
            ir: .qwen35_0_8B,
            layerIndex: 0,
            layerType: .delta
        )
        let downDecodeCandidate = try #require(SmeltKernelConsumerNaming.candidate(
            kind: .ffnDownResidualDecode,
            context: layerContext
        ))
        try compilationPlan.validateGeneratedPipelineUses(
            [
                SmeltNamedPipelineUse(
                    name: "fused_affine_matvec_add_c3584_r1024_g64_rows4",
                    plannedKernelCandidate: downDecodeCandidate
                ),
            ],
            context: "test"
        )

        #expect(plan.unplannedGeneratedPipelineNames([
            "fused_affine_matvec_add_c3584_r1024_g64_rows4",
            "fused_affine_matvec_add_c11008_r2048_g64_rows4",
        ]) == ["fused_affine_matvec_add_c11008_r2048_g64_rows4"])
        #expect(plan.generatedPipelineUsesMissingPlannedCandidates([
            SmeltNamedPipelineUse(
                name: "fused_affine_matvec_add_c3584_r1024_g64_rows4",
                plannedKernelCandidate: nil
            ),
        ]) == ["fused_affine_matvec_add_c3584_r1024_g64_rows4"])
        #expect(plan.unauthorizedGeneratedPipelineUses([
            SmeltNamedPipelineUse(
                name: "fused_affine_matvec_add_c2048_r1024_g64_rows4",
                plannedKernelCandidate: downDecodeCandidate
            ),
        ]) == [
            SmeltGeneratedPipelineUseAuthorizationFailure(
                pipelineName: "fused_affine_matvec_add_c2048_r1024_g64_rows4",
                consumerID: "layers_0_mlp.down_proj.residual.decode",
                plannedCapabilityName: "fused_affine_matvec_add_c3584_r1024_g64_rows4"
            ),
        ])

        try compilationPlan.validateGeneratedPipelineNames(
            ["affine_matvec_c3584_r1024_g64_batched_full"],
            context: "test"
        )
        let downPrefillCandidate = try #require(SmeltKernelConsumerNaming.candidate(
            kind: .ffnDownPrefill,
            context: layerContext
        ))
        try compilationPlan.validateGeneratedPipelineUses(
            [
                SmeltNamedPipelineUse(
                    name: "affine_matvec_c3584_r1024_g64_batched_full",
                    plannedKernelCandidate: downPrefillCandidate
                ),
            ],
            context: "test"
        )
        #expect(plan.packageLocalGeneratedPipelineNames([
            "affine_matvec_c3584_r1024_g64_batched_full",
        ]) == [])

        #expect(throws: SmeltCompilerError.self) {
            try compilationPlan.validateGeneratedPipelineNames(
                ["fused_affine_matvec_add_c11008_r2048_g64_rows4"],
                context: "test"
            )
        }

        #expect(throws: SmeltCompilerError.self) {
            try compilationPlan.validateGeneratedPipelineUses(
                [
                    SmeltNamedPipelineUse(
                        name: "fused_affine_matvec_add_c3584_r1024_g64_rows4",
                        plannedKernelCandidate: nil
                    ),
                ],
                context: "test"
            )
        }

        #expect(throws: SmeltCompilerError.self) {
            try compilationPlan.validateGeneratedPipelineUses(
                [
                    SmeltNamedPipelineUse(
                        name: "fused_affine_matvec_add_c2048_r1024_g64_rows4",
                        plannedKernelCandidate: downDecodeCandidate
                    ),
                ],
                context: "test"
            )
        }

        let gateUpPrefillCandidate = try #require(SmeltKernelConsumerNaming.candidate(
            kind: .ffnGateUpPrefill,
            context: layerContext
        ))
        #expect(plan.unauthorizedGeneratedPipelineUses([
            SmeltNamedPipelineUse(
                name: "affine_matvec_c3584_r1024_g64_batched_full",
                plannedKernelCandidate: gateUpPrefillCandidate
            ),
        ]) == [
            SmeltGeneratedPipelineUseAuthorizationFailure(
                pipelineName: "affine_matvec_c3584_r1024_g64_batched_full",
                consumerID: "layers_0_mlp.gate_up.prefill",
                plannedCapabilityName: "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full"
            ),
        ])
    }

    @Test("Generated pipeline validation trusts planned capability source")
    func generatedPipelineValidationTrustsPlannedCapabilitySource() {
        let shape = SmeltKernelShape(rows: 1_024, cols: 3_584, groupSize: 64)
        let candidate = SmeltPlannedKernelCandidate(
            consumerID: "source_ownership.prefill",
            operation: .affineMatvecPrefillFull,
            shape: shape,
            weights: [
                SmeltPlannedKernelWeight(
                    weightName: "source_ownership_weight",
                    role: .affine
                ),
            ]
        )
        let catalogNamedCapability = SmeltKernelCapability(
            id: "affine_matvec_c3584_r1024_g64_batched_full",
            phase: .prefill,
            operation: .affineMatvecPrefillFull,
            shape: shape,
            source: .packageLocalGenerated,
            weightRequirements: [
                SmeltKernelWeightRequirement(
                    role: .affine,
                    acceptedLayouts: [.affineU4RowMajor(groupSize: 64)]
                ),
            ],
            rowTile: 32,
            batchTile: 16,
            threadgroupWidth: 128
        )
        let plan = SmeltKernelPlan(generatedUses: [
            SmeltPlannedKernelUse(
                consumerID: candidate.consumerID,
                capability: catalogNamedCapability,
                weights: candidate.weights
            ),
        ])

        #expect(plan.emittedGeneratedCapabilityNames == [
            "affine_matvec_c3584_r1024_g64_batched_full",
        ])
        #expect(plan.packageLocalGeneratedPipelineNames([
            "affine_matvec_c3584_r1024_g64_batched_full",
        ]) == [
            "affine_matvec_c3584_r1024_g64_batched_full",
        ])
        #expect(plan.generatedPipelineUsesMissingPlannedCandidates([
            SmeltNamedPipelineUse(
                name: "affine_matvec_c3584_r1024_g64_batched_full",
                plannedKernelCandidate: nil
            ),
        ]) == [
            "affine_matvec_c3584_r1024_g64_batched_full",
        ])
        #expect(plan.unauthorizedGeneratedPipelineUses([
            SmeltNamedPipelineUse(
                name: "affine_matvec_c3584_r1024_g64_batched_full",
                plannedKernelCandidate: candidate
            ),
        ]).isEmpty)
    }

    @Test("Compiler rejects illegal kernel-planned storage before codegen")
    func compilerRejectsIllegalKernelPlannedStorageBeforeCodegen() {
        let layout = SmeltWeightLayout.computeLayout(from: .qwen35_0_8B)
            .filter { $0.name != "layers_0_mlp_down_proj_weight" }

        #expect(throws: SmeltCompilerError.self) {
            _ = try SmeltCompiler.validateWeightStoragePlan(
                ir: .qwen35_0_8B,
                weightLayout: layout
            )
        }

        #expect(throws: SmeltCompilerError.self) {
            _ = try TopLevelEmitter.generate(
                ir: .qwen35_0_8B,
                plan: buildBufferPlan(from: .qwen35_0_8B),
                weightLayout: layout
            )
        }
    }

    @Test("Prefill no-scale V norm rewrites into fused RoPE KV cache")
    func prefillNoScaleVNormRewritesIntoFusedRopeKVCache() {
        let rewrite = SmeltFusionPlanner.auto.rewrite(window: [
            .dispatch(SmeltDispatch(
                pipeline: .perHeadRmsNormNoScaleBatched,
                buffers: [
                    SmeltBufferBinding(slot: 10, index: 0),
                ],
                constants: [
                    SmeltConstantBinding(expression: "1", type: .uint32, index: 1),
                    SmeltConstantBinding(expression: "512", type: .uint32, index: 2),
                    SmeltConstantBinding(expression: "1e-6", type: .float32, index: 3),
                ],
                dispatch: .threadgroups(
                    width: 1,
                    height: 16,
                    depth: 1,
                    tgWidth: 256,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                dynamicGridH: .seqLen
            )),
            .dispatch(SmeltDispatch(
                pipeline: .ropeAndKvCachePrefill,
                buffers: [
                    SmeltBufferBinding(slot: 8, index: 0),
                    SmeltBufferBinding(slot: 9, index: 1),
                    SmeltBufferBinding(slot: 10, index: 2),
                    SmeltBufferBinding(slot: 11, index: 3),
                    SmeltBufferBinding(slot: 12, index: 4),
                    SmeltBufferBinding(slot: 20, index: 5),
                    SmeltBufferBinding(slot: 21, index: 6),
                ],
                constants: [
                    SmeltConstantBinding(expression: "512", type: .uint32, index: 7),
                    SmeltConstantBinding(expression: "512", type: .uint32, index: 8),
                    SmeltConstantBinding(expression: "8", type: .uint32, index: 9),
                    SmeltConstantBinding(expression: "1", type: .uint32, index: 10),
                    SmeltConstantBinding(expression: "__seqLen__", type: .uint32, index: 11),
                    SmeltConstantBinding(expression: "__startPos__", type: .uint32, index: 12),
                    SmeltConstantBinding(expression: "cacheSeqCapacity", type: .uint32, index: 13),
                    SmeltConstantBinding(expression: "1", type: .uint32, index: 14),
                ],
                dispatch: .threadgroups(
                    width: 16,
                    height: 8,
                    depth: 1,
                    tgWidth: 256,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                dynamicGridW: .seqLen
            )),
        ][...])

        #expect(rewrite?.consumedOpCount == 2)
        #expect(rewrite?.kind == .specializedKernel)
        #expect(rewrite?.rule == .fusedNormRopeKVPrefill)
        guard case .dispatch(let fused)? = rewrite?.producedOps.first else {
            Issue.record("Expected fused norm RoPE KV dispatch rewrite")
            return
        }
        #expect(fused.pipeline == .fusedNormRopeAndKvCachePrefill)
        #expect(fused.buffers.count == 7)
        #expect(fused.constants.count == 9)
        #expect(fused.constants.last?.expression == "1e-6")
        #expect(fused.constants.last?.bindingIndex == 15)
        #expect(fused.dynamicGridW == .seqLen)
    }

    @Test("Prefill scaled Q norm rewrites shared-KV RoPE dispatch")
    func prefillScaledQNormRewritesSharedKVRopeDispatch() {
        let rewrite = SmeltFusionPlanner.auto.rewrite(window: [
            .dispatch(SmeltDispatch(
                pipeline: .perHeadRmsNormBatched,
                buffers: [
                    SmeltBufferBinding(slot: 30, index: 0),
                    SmeltBufferBinding(slot: 31, index: 1),
                    SmeltBufferBinding(slot: 8, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "8", type: .uint32, index: 3),
                    SmeltConstantBinding(expression: "256", type: .uint32, index: 4),
                    SmeltConstantBinding(expression: "1e-6", type: .float32, index: 5),
                ],
                dispatch: .threadgroups(
                    width: 8,
                    height: 1,
                    depth: 1,
                    tgWidth: 256,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                dynamicGridH: .seqLen
            )),
            .traceMarker(label: "L0.q_norm", bufferSlot: 8),
            .dispatch(SmeltDispatch(
                pipeline: .ropeAndKvCachePrefill,
                buffers: [
                    SmeltBufferBinding(slot: 8, index: 0),
                    SmeltBufferBinding(slot: 9, index: 1),
                    SmeltBufferBinding(slot: 10, index: 2),
                    SmeltBufferBinding(slot: 11, index: 3),
                    SmeltBufferBinding(slot: 12, index: 4),
                    SmeltBufferBinding(slot: 20, index: 5),
                    SmeltBufferBinding(slot: 21, index: 6),
                ],
                constants: [
                    SmeltConstantBinding(expression: "256", type: .uint32, index: 7),
                    SmeltConstantBinding(expression: "256", type: .uint32, index: 8),
                    SmeltConstantBinding(expression: "8", type: .uint32, index: 9),
                    SmeltConstantBinding(expression: "0", type: .uint32, index: 10),
                    SmeltConstantBinding(expression: "__seqLen__", type: .uint32, index: 11),
                    SmeltConstantBinding(expression: "__startPos__", type: .uint32, index: 12),
                    SmeltConstantBinding(expression: "cacheSeqCapacity", type: .uint32, index: 13),
                    SmeltConstantBinding(expression: "1", type: .uint32, index: 14),
                ],
                dispatch: .threadgroups(
                    width: 8,
                    height: 8,
                    depth: 1,
                    tgWidth: 256,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                dynamicGridW: .seqLen
            )),
        ][...])

        #expect(rewrite?.consumedOpCount == 3)
        #expect(rewrite?.kind == .specializedKernel)
        #expect(rewrite?.rule == .fusedNormRopeKVPrefill)
        #expect(rewrite?.producedOps.count == 2)
        guard case .dispatch(let fused)? = rewrite?.producedOps.first,
              case .traceMarker(let label, let bufferSlot)? = rewrite?.producedOps.last
        else {
            Issue.record("Expected fused dispatch followed by preserved trace marker")
            return
        }

        #expect(fused.pipeline == .fusedNormRopeAndKvCachePrefill)
        #expect(fused.buffers.count == 7)
        #expect(fused.buffers[0].slot == .fixed(8))
        #expect(fused.buffers[1].slot == .fixed(30))
        #expect(fused.buffers[2].slot == .fixed(31))
        #expect(fused.constants.count == 9)
        #expect(fused.constants[3].expression == "0")
        #expect(fused.constants.last?.expression == "1e-6")
        #expect(fused.constants.last?.bindingIndex == 15)
        #expect(fused.dynamicGridW == .seqLen)
        #expect(label == "L0.q_norm")
        #expect(bufferSlot == 8)
    }

    @Test("Generated wrapper source uses planned capability geometry")
    func generatedWrapperSourceUsesPlannedCapabilityGeometry() throws {
        let downCapability = try #require(
            SmeltKernelCapabilityRegistry.generatedCapability(
                operation: .affineMatvecPrefillFull,
                shape: SmeltKernelShape(rows: 1_024, cols: 11_008, groupSize: 64)
            )
        )
        let gateUpCapability = try #require(
            SmeltKernelCapabilityRegistry.generatedCapability(
                operation: .fusedGateUpSwigluPrefillFull,
                shape: SmeltKernelShape(rows: 11_008, cols: 1_024, groupSize: 64)
            )
        )
        let decodeCapability = try #require(
            SmeltKernelCapabilityRegistry.generatedCapability(
                operation: .affineMatvecResidualAdd,
                shape: SmeltKernelShape(rows: 1_024, cols: 11_008, groupSize: 64)
            )
        )
        let plan = SmeltKernelPlan(generatedUses: [
            SmeltPlannedKernelUse(
                consumerID: "test.down.prefill",
                capability: downCapability,
                weights: [
                    SmeltPlannedKernelWeight(weightName: "down_weight", role: .affine),
                ]
            ),
            SmeltPlannedKernelUse(
                consumerID: "test.gate_up.prefill",
                capability: gateUpCapability,
                weights: [
                    SmeltPlannedKernelWeight(weightName: "gate_weight", role: .gate),
                    SmeltPlannedKernelWeight(weightName: "up_weight", role: .up),
                ]
            ),
            SmeltPlannedKernelUse(
                consumerID: "test.down.decode",
                capability: decodeCapability,
                weights: [
                    SmeltPlannedKernelWeight(weightName: "down_weight", role: .affine),
                ]
            ),
        ])

        let source = SmeltGeneratedKernelVariants.lutMatvecSuffix(kernelPlan: plan)

        #expect(source.contains(
            "// planned geometry: row_tile=32 batch_tile=16 threadgroup_width=128"
        ))
        #expect(source.contains("threadgroup half Xs[16 * (32 + 8)];"))
        #expect(source.contains("affine_matvec_fixed_batched_full<1024, 11008, 64, 16>"))
        #expect(source.contains(
            "agent_fused_affine_gate_up_qmm_fixed_batched_full<11008, 1024, 64, 16>"
        ))
        #expect(source.contains(
            "// planned geometry: row_tile=4 batch_tile=nil threadgroup_width=64"
        ))
        #expect(source.contains("fused_affine_matvec_add_fixed_rows4<1024, 11008, 64>"))
    }

    @Test("Generated wrapper source takes geometry from planned capability")
    func generatedWrapperSourceTakesGeometryFromPlannedCapability() throws {
        let capability = SmeltKernelCapability(
            id: "affine_matvec_c1024_r512_g64_batched_full_geometry_probe",
            phase: .prefill,
            operation: .affineMatvecPrefillFull,
            shape: SmeltKernelShape(rows: 512, cols: 1_024, groupSize: 64),
            source: .packageLocalGenerated,
            sourceTemplate: .affineMatvecPrefillFull,
            weightRequirements: [
                SmeltKernelWeightRequirement(
                    role: .affine,
                    acceptedLayouts: [.affineU4RowMajor(groupSize: 64)]
                ),
            ],
            rowTile: 24,
            batchTile: 7,
            threadgroupWidth: 96
        )
        let plan = SmeltKernelPlan(generatedUses: [
            SmeltPlannedKernelUse(
                consumerID: "geometry.probe.prefill",
                capability: capability,
                weights: [
                    SmeltPlannedKernelWeight(weightName: "probe_weight", role: .affine),
                ]
            ),
        ])

        let source = SmeltGeneratedKernelVariants.lutMatvecSuffix(kernelPlan: plan)

        #expect(source.contains(
            "// planned geometry: row_tile=24 batch_tile=7 threadgroup_width=96"
        ))
        #expect(source.contains("threadgroup half Xs[7 * (24 + 8)];"))
        #expect(source.contains("affine_matvec_fixed_batched_full<512, 1024, 64, 7>"))
    }
}
