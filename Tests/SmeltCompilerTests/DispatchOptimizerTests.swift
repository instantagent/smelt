import Testing

@testable import SmeltCompiler

@Suite("SmeltDispatchOptimizer")
struct DispatchOptimizerTests {

    @Test("Default optimizer applies the planner registry across fusion rules")
    func defaultOptimizerAppliesPlannerRegistryAcrossFusionRules() {
        var ops: [SmeltIROp] = [
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
        ]

        let stats = SmeltDispatchOptimizer.optimize(&ops, planner: .auto)

        #expect(ops.count == 2)
        guard case .dispatch(let l2) = ops[0],
              case .threadgroups(let l2Width, _, _, _, _, _) = l2.dispatch,
              case .dispatch(let residual) = ops[1]
        else {
            Issue.record("Expected coalesced L2 and fused residual dispatches")
            return
        }
        #expect(l2.pipeline == .l2NormalizeD128)
        #expect(l2Width == 16)
        #expect(residual.pipeline == .fusedAffineMatvecAddC6144R2048G64)
        #expect(stats.rewriteCounts["contiguousL2Normalize"] == 1)
        #expect(stats.rewriteCounts["matvecResidualAdd"] == 1)
        #expect(stats.opportunities.contains(where: {
            $0.pattern == "contiguousL2Normalize"
                && $0.count == 1
                && $0.fusedKernelAvailable
        }))
        #expect(stats.opportunities.contains(where: {
            $0.pattern == "matvecResidualAdd"
                && $0.count == 1
                && $0.fusedKernelAvailable
        }))
    }

    @Test("Optimizer reports but skips costly residual-add decode Q projection cooperative norm")
    func optimizerReportsButSkipsCostlyResidualAddQProjectionCooperativeNorm() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .rmsNorm1PWD2560Add,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 30, offset: 1_024, index: 1),
                    SmeltBufferBinding(variableSlot: "residual", index: 2),
                    SmeltBufferBinding(slot: 8, index: 3),
                ],
                constants: [],
                dispatch: .threadgroups(width: 1, height: 1, depth: 1, tgWidth: 320, tgHeight: 1, tgDepth: 1),
                comment: "attention norm + residual"
            )),
            .traceMarker(label: "L0.attn_norm_add", bufferSlot: 8),
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvecC2560R256G128Rows4,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 8, index: 3),
                    SmeltBufferBinding(slot: 11, index: 4),
                ],
                constants: [],
                dispatch: .threadgroups(width: 64, height: 1, depth: 1, tgWidth: 64, tgHeight: 1, tgDepth: 1),
                comment: "Q projection"
            )),
        ]

        let stats = SmeltDispatchOptimizer.optimize(&ops, planner: .auto)

        #expect(ops.count == 3)
        #expect(stats.rewriteCounts[SmeltFusionRule.cooperativeNormScaleConsumer.rawValue] == nil)
        #expect(stats.opportunities.contains {
            $0.pattern == "normConsumer"
                && $0.shape == "rms_norm_1pw_d2560_add->affine_matvec_c2560_r256_g128_rows4"
                && $0.fusedKernelAvailable
        })
        guard case .dispatch(let normAdd) = ops[0],
              case .traceMarker(let label, let bufferSlot) = ops[1],
              case .dispatch(let qProjection) = ops[2]
        else {
            Issue.record("Expected original norm-add, trace marker, and Q projection")
            return
        }

        #expect(normAdd.pipeline == .rmsNorm1PWD2560Add)
        #expect(label == "L0.attn_norm_add")
        #expect(bufferSlot == 8)
        #expect(qProjection.pipeline == .affineMatvecC2560R256G128Rows4)
    }

    @Test("Optimizer reports but skips costly E4B decode GeGLU cooperative norm")
    func optimizerReportsButSkipsCostlyE4BDecodeGeGLUCooperativeNorm() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .rmsNorm1PWD2560,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 30, offset: 1_024, index: 1),
                    SmeltBufferBinding(slot: 8, index: 2),
                ],
                constants: [],
                dispatch: .threadgroups(width: 1, height: 1, depth: 1, tgWidth: 320, tgHeight: 1, tgDepth: 1),
                comment: "Pre-feedforward layernorm"
            )),
            .traceMarker(label: "L0.pre_ffn_norm", bufferSlot: 8),
            .dispatch(SmeltDispatch(
                pipeline: .fusedAffineGateUpGeGLUC2560R10240G128Rows4,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 30, offset: 40, index: 3),
                    SmeltBufferBinding(slot: 30, offset: 50, index: 4),
                    SmeltBufferBinding(slot: 30, offset: 60, index: 5),
                    SmeltBufferBinding(slot: 8, index: 6),
                    SmeltBufferBinding(slot: 13, index: 7),
                ],
                constants: [],
                dispatch: .threadgroups(width: 2_560, height: 1, depth: 1, tgWidth: 64, tgHeight: 1, tgDepth: 1),
                comment: "FFN fused gate+up+GeGLU"
            )),
        ]

        let stats = SmeltDispatchOptimizer.optimize(&ops, planner: .auto)

        #expect(ops.count == 3)
        #expect(stats.rewriteCounts[SmeltFusionRule.cooperativeNormScaleConsumer.rawValue] == nil)
        #expect(stats.opportunities.contains {
            $0.pattern == "normConsumer"
                && $0.shape == "rms_norm_1pw_d2560->fused_affine_gate_up_geglu_c2560_r10240_g128_rows4"
                && $0.fusedKernelAvailable
        })
        guard case .dispatch(let norm) = ops[0],
              case .traceMarker(let label, let bufferSlot) = ops[1],
              case .dispatch(let geglu) = ops[2]
        else {
            Issue.record("Expected original norm, trace marker, and GeGLU dispatch")
            return
        }

        #expect(norm.pipeline == .rmsNorm1PWD2560)
        #expect(label == "L0.pre_ffn_norm")
        #expect(bufferSlot == 8)
        #expect(geglu.pipeline == .fusedAffineGateUpGeGLUC2560R10240G128Rows4)
    }

    @Test("Optimizer applies c256 affine cooperative norm routes")
    func optimizerAppliesC256AffineCooperativeNorm() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .rmsNorm1PW,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 30, offset: 1_024, index: 1),
                    SmeltBufferBinding(slot: 8, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "256", type: .uint32, index: 3),
                    SmeltConstantBinding(expression: "1e-6", type: .float32, index: 4),
                ],
                dispatch: .threadgroups(width: 1, height: 1, depth: 1, tgWidth: 256, tgHeight: 1, tgDepth: 1),
                comment: "Input layernorm"
            )),
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvecC256R1024G128Rows4,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 8, index: 3),
                    SmeltBufferBinding(slot: 13, index: 4),
                ],
                constants: [],
                dispatch: .threadgroups(width: 256, height: 1, depth: 1, tgWidth: 64, tgHeight: 1, tgDepth: 1),
                comment: "Q projection"
            )),
        ]

        let stats = SmeltDispatchOptimizer.optimize(&ops, planner: .auto)

        #expect(stats.rewriteCounts[SmeltFusionRule.cooperativeNormScaleConsumer.rawValue] == 1)
        #expect(stats.opportunities.contains {
            $0.pattern == "normConsumer"
                && $0.shape == "rms_norm_1pw->affine_matvec_c256_r1024_g128_rows4"
                && $0.fusedKernelAvailable
        })
        guard case .dispatch(let scaleOnly) = ops[0],
              case .dispatch(let scaledAffine) = ops[1]
        else {
            Issue.record("Expected scale-only dispatch followed by norm-scaled c256 affine")
            return
        }

        #expect(scaleOnly.pipeline == .rmsNormScaleOnly)
        #expect(scaleOnly.constants.map(\.expression) == ["256", "1e-6"])
        #expect(scaledAffine.pipeline == .normScaleAffineMatvecC256R1024G128Rows4)
        #expect(scaledAffine.buffers[3].slot == .fixed(SmeltFixedSlot.normOutBuf.rawValue))
    }

    @Test("Optimizer reports prefill norm-consumer opportunities without rewriting them")
    func optimizerReportsPrefillNormConsumerOpportunity() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .rmsNorm1PWD1536Batched,
                buffers: [
                    SmeltBufferBinding(slot: 1, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 1234, index: 1),
                    SmeltBufferBinding(slot: 8, index: 2),
                ],
                constants: [],
                dispatch: .threadgroups(width: 1, height: 1, depth: 1, tgWidth: 192, tgHeight: 1, tgDepth: 1)
            )),
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvecC1536R2048G128Batched,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 8, index: 3),
                    SmeltBufferBinding(slot: 20, index: 4),
                ],
                constants: [],
                dispatch: .threadgroups(width: 512, height: 1, depth: 1, tgWidth: 64, tgHeight: 1, tgDepth: 1)
            )),
        ]

        let stats = SmeltDispatchOptimizer.optimize(&ops, planner: .auto)

        #expect(ops.count == 2)
        #expect(stats.rewriteCounts.isEmpty)
        let hasPrefillNormConsumerOpportunity = stats.opportunities.contains { opportunity in
            let matchesPattern = opportunity.pattern == "normConsumer"
            let matchesShape = opportunity.shape
                == "rms_norm_1pw_d1536_batched->affine_matvec_c1536_r2048_g128_batched"
            let matchesCount = opportunity.count == 1
            return matchesPattern
                && matchesShape
                && matchesCount
                && !opportunity.fusedKernelAvailable
        }
        #expect(hasPrefillNormConsumerOpportunity)
    }

    @Test("Optimizer fuses d1536 RMS norm followed by residual add")
    func optimizerFusesD1536RMSNormResidualAdd() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .rmsNorm1PWD1536,
                buffers: [
                    SmeltBufferBinding(slot: 8, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 1234, index: 1),
                    SmeltBufferBinding(slot: 9, index: 2),
                ],
                constants: [],
                dispatch: .threadgroups(
                    width: 1,
                    height: 1,
                    depth: 1,
                    tgWidth: 192,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                minSeqLen: 7
            )),
            .dispatch(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(variableSlot: "alt", offset: 18_432, index: 0),
                    SmeltBufferBinding(slot: 9, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", offset: 18_432, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "1536", type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: 1_536,
                    height: 1,
                    depth: 1,
                    tgWidth: 1_024,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                minSeqLen: 7
            )),
        ]

        let stats = SmeltDispatchOptimizer.optimize(&ops, planner: .auto)

        #expect(ops.count == 1)
        #expect(stats.rewriteCounts["rmsNormResidualAdd"] == 1)
        guard case .dispatch(let fused) = ops[0] else {
            Issue.record("Expected fused RMSNorm+add dispatch")
            return
        }
        #expect(fused.pipeline == .rmsNorm1PWD1536Add)
        #expect(fused.buffers.count == 4)
        #expect(fused.buffers[2].slot == .variable("alt"))
        #expect(fused.buffers[2].byteOffset == 18_432)
        #expect(fused.buffers[3].slot == .variable("alt"))
        #expect(fused.buffers[3].byteOffset == 18_432)
        #expect(fused.minSeqLen == 7)
    }

    @Test("Optimizer fuses generic d256 RMS norm followed by residual add")
    func optimizerFusesGenericD256RMSNormResidualAdd() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .rmsNorm1PW,
                buffers: [
                    SmeltBufferBinding(slot: 8, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 1234, index: 1),
                    SmeltBufferBinding(slot: 9, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "256", type: .uint32, index: 3),
                    SmeltConstantBinding(expression: "1e-6", type: .float32, index: 4),
                ],
                dispatch: .threadgroups(
                    width: 1,
                    height: 1,
                    depth: 1,
                    tgWidth: 256,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                minSeqLen: 7
            )),
            .dispatch(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 9, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "256", type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: 256,
                    height: 1,
                    depth: 1,
                    tgWidth: 256,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                minSeqLen: 7
            )),
        ]

        let stats = SmeltDispatchOptimizer.optimize(&ops, planner: .auto)

        #expect(ops.count == 1)
        #expect(stats.rewriteCounts["rmsNormResidualAdd"] == 1)
        guard case .dispatch(let fused) = ops[0] else {
            Issue.record("Expected fused d256 RMSNorm+add dispatch")
            return
        }
        #expect(fused.pipeline == .rmsNorm1PWD256Add)
        #expect(fused.buffers.count == 4)
        if case let .threadgroups(width, height, depth, tgWidth, tgHeight, tgDepth) = fused.dispatch {
            #expect(width == 1)
            #expect(height == 1)
            #expect(depth == 1)
            #expect(tgWidth == 256)
            #expect(tgHeight == 1)
            #expect(tgDepth == 1)
        } else {
            Issue.record("Expected threadgroup dispatch for fused d256 RMSNorm+add")
        }
        #expect(fused.minSeqLen == 7)
    }

    @Test("Optimizer fuses d256 RMS norm residual add followed by layer scalar")
    func optimizerFusesD256RMSNormResidualAddScalarWeight() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .rmsNorm1PWD256Add,
                buffers: [
                    SmeltBufferBinding(slot: 8, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 1234, index: 1),
                    SmeltBufferBinding(variableSlot: "cur", index: 2),
                    SmeltBufferBinding(variableSlot: "alt", index: 3),
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
                minSeqLen: 7
            )),
            .dispatch(SmeltDispatch(
                pipeline: .scalarMulWeight,
                buffers: [
                    SmeltBufferBinding(variableSlot: "alt", index: 0),
                    SmeltBufferBinding(slot: 30, offset: 5678, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "256", type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: 256,
                    height: 1,
                    depth: 1,
                    tgWidth: 256,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                minSeqLen: 7
            )),
        ]

        let stats = SmeltDispatchOptimizer.optimize(&ops, planner: .auto)

        #expect(ops.count == 1)
        #expect(stats.rewriteCounts[SmeltFusionRule.rmsNormResidualAddScalarWeight.rawValue] == 1)
        guard case .dispatch(let fused) = ops[0] else {
            Issue.record("Expected fused d256 RMSNorm+add+scalar dispatch")
            return
        }
        #expect(fused.pipeline == .rmsNorm1PWD256AddScalarWeight)
        #expect(fused.buffers.count == 5)
        if case let .threadgroups(width, height, depth, tgWidth, tgHeight, tgDepth) = fused.dispatch {
            #expect(width == 1)
            #expect(height == 1)
            #expect(depth == 1)
            #expect(tgWidth == 256)
            #expect(tgHeight == 1)
            #expect(tgDepth == 1)
        } else {
            Issue.record("Expected threadgroup dispatch for fused d256 RMSNorm+add+scalar")
        }
        #expect(fused.minSeqLen == 7)
    }

    @Test("Optimizer fuses batched d1536 RMS norm followed by residual add")
    func optimizerFusesBatchedD1536RMSNormResidualAdd() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .rmsNorm1PWD1536Batched,
                buffers: [
                    SmeltBufferBinding(slot: 8, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 1234, index: 1),
                    SmeltBufferBinding(slot: 9, index: 2),
                ],
                constants: [],
                dispatch: .threadgroups(
                    width: 64,
                    height: 1,
                    depth: 1,
                    tgWidth: 192,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                dynamicGridW: .seqLen
            )),
            .dispatch(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 9, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "__seqLen__*1536", type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: 1_536,
                    height: 1,
                    depth: 1,
                    tgWidth: 1_024,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                dynamicGridW: .seqLenMul(1_536)
            )),
        ]

        let stats = SmeltDispatchOptimizer.optimize(&ops, planner: .auto)

        #expect(ops.count == 1)
        #expect(stats.rewriteCounts["rmsNormResidualAdd"] == 1)
        guard case .dispatch(let fused) = ops[0] else {
            Issue.record("Expected fused batched RMSNorm+add dispatch")
            return
        }
        #expect(fused.pipeline == .rmsNorm1PWD1536AddBatched)
        #expect(fused.buffers.count == 4)
        #expect(fused.buffers[2].slot == .variable("cur"))
        #expect(fused.buffers[3].slot == .variable("alt"))
        #expect(fused.dynamicGridW == .seqLen)
    }

    @Test("Cooperative norm fusion rewrites decode norm into rows4 norm-scaled affine")
    func cooperativeNormFusionRewritesDecodeNorm() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .rmsNorm1PWD1536,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 30, offset: 1234, index: 1),
                    SmeltBufferBinding(slot: 8, index: 2),
                ],
                constants: [],
                dispatch: .threadgroups(width: 1, height: 1, depth: 1, tgWidth: 192, tgHeight: 1, tgDepth: 1),
                comment: "Input layernorm"
            )),
            .traceMarker(label: "L0.input_norm", bufferSlot: 8),
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvecC1536R2048G128Rows4,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 8, index: 3),
                    SmeltBufferBinding(slot: 20, index: 4),
                ],
                constants: [],
                dispatch: .threadgroups(width: 512, height: 1, depth: 1, tgWidth: 64, tgHeight: 1, tgDepth: 1),
                comment: "Q projection"
            )),
        ]

        CooperativeNormFusionPass().run(&ops)

        #expect(ops.count == 3)
        guard case .dispatch(let scaleOnly) = ops[0],
              case .dispatch(let scaledConsumer) = ops[1],
              case .traceMarker(let label, let bufferSlot) = ops[2]
        else {
            Issue.record("Expected fused dispatches followed by preserved trace marker")
            return
        }

        #expect(scaleOnly.pipeline == .rmsNormScaleOnlyD1536)
        #expect(scaleOnly.buffers.count == 2)
        #expect(scaleOnly.constants.isEmpty)

        #expect(scaledConsumer.pipeline == .normScaleAffineMatvecC1536R2048G128Rows4)
        #expect(scaledConsumer.buffers.count == 8)
        #expect(scaledConsumer.buffers[0].slot == .fixed(SmeltFixedSlot.normScaleScratch.rawValue))
        #expect(scaledConsumer.buffers[1].slot == .variable("cur"))
        #expect(scaledConsumer.buffers[2].slot == .fixed(30))
        #expect(scaledConsumer.buffers[3].slot == .fixed(8))
        #expect(scaledConsumer.constants.isEmpty)
        #expect(scaledConsumer.fcCols == nil)
        #expect(scaledConsumer.fcGroupSize == nil)
        #expect(label == "L0.input_norm")
        #expect(bufferSlot == 8)
    }

    @Test("Cooperative norm fusion rewrites batched QMM FFN with dynamic prefill grids")
    func cooperativeNormFusionRewritesBatchedQMMGateUp() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .rmsNorm1PWBatched,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 30, offset: 1234, index: 1),
                    SmeltBufferBinding(slot: 8, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "2048", type: .uint32, index: 3),
                    SmeltConstantBinding(expression: "1e-6", type: .float32, index: 4),
                ],
                dispatch: .threadgroups(width: 1, height: 1, depth: 1, tgWidth: 256, tgHeight: 1, tgDepth: 1),
                comment: "FFN norm",
                dynamicGridW: .seqLen
            )),
            .traceMarker(label: "L0.ffn_norm", bufferSlot: 8),
            .dispatch(SmeltDispatch(
                pipeline: .fusedAffineGateUpSwigluC2048R8192G64BatchedFull,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 30, offset: 40, index: 3),
                    SmeltBufferBinding(slot: 30, offset: 50, index: 4),
                    SmeltBufferBinding(slot: 30, offset: 60, index: 5),
                    SmeltBufferBinding(slot: 8, index: 6),
                    SmeltBufferBinding(slot: 13, index: 7),
                ],
                constants: [
                    SmeltConstantBinding(expression: "__seqLen__", type: .uint32, index: 8),
                ],
                dispatch: .threadgroups(width: 256, height: 1, depth: 1, tgWidth: 128, tgHeight: 1, tgDepth: 1),
                comment: "FFN gate/up",
                dynamicGridH: .seqLenCeilDiv(16)
            )),
        ]

        CooperativeNormFusionPass().run(&ops)

        #expect(ops.count == 3)
        guard case .dispatch(let scaleOnly) = ops[0],
              case .dispatch(let scaledConsumer) = ops[1],
              case .traceMarker(let label, let bufferSlot) = ops[2]
        else {
            Issue.record("Expected norm-scale prefill dispatches followed by preserved trace marker")
            return
        }

        #expect(scaleOnly.pipeline == .rmsNormScaleOnlyD2048Batched)
        #expect(scaleOnly.dynamicGridW == .seqLen)
        if case let .threadgroups(width, height, depth, tgWidth, tgHeight, tgDepth) = scaleOnly.dispatch {
            #expect(width == 1)
            #expect(height == 1)
            #expect(depth == 1)
            #expect(tgWidth == 1024)
            #expect(tgHeight == 1)
            #expect(tgDepth == 1)
        } else {
            Issue.record("Expected scale-only dispatch to use threadgroups")
        }

        #expect(scaledConsumer.pipeline == .normScaleAffineGateUpSwigluC2048R8192G64BatchedFull)
        #expect(scaledConsumer.buffers.count == 11)
        #expect(scaledConsumer.buffers[0].slot == .fixed(SmeltFixedSlot.normScaleScratch.rawValue))
        #expect(scaledConsumer.buffers[1].slot == .variable("cur"))
        #expect(scaledConsumer.buffers[2].slot == .fixed(30))
        #expect(scaledConsumer.buffers[3].slot == .fixed(8))
        #expect(scaledConsumer.buffers[10].slot == .fixed(13))
        #expect(scaledConsumer.constants.count == 1)
        #expect(scaledConsumer.constants[0].bindingIndex == 11)
        #expect(scaledConsumer.dynamicGridH == .seqLenCeilDiv(16))
        #expect(label == "L0.ffn_norm")
        #expect(bufferSlot == 8)
    }

    @Test("Cooperative norm fusion rewrites E4B batched FFN matvec")
    func cooperativeNormFusionRewritesE4BBatchedFFNMatvec() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .rmsNorm1PWD2560Batched,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 30, offset: 1234, index: 1),
                    SmeltBufferBinding(slot: 8, index: 2),
                ],
                constants: [],
                dispatch: .threadgroups(width: 1, height: 1, depth: 1, tgWidth: 320, tgHeight: 1, tgDepth: 1),
                comment: "FFN norm",
                dynamicGridW: .seqLen
            )),
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvecC2560R10240G128Batched,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 8, index: 3),
                    SmeltBufferBinding(slot: 13, index: 4),
                ],
                constants: [
                    SmeltConstantBinding(expression: "__seqLen__", type: .uint32, index: 5),
                ],
                dispatch: .threadgroups(width: 1280, height: 1, depth: 1, tgWidth: 64, tgHeight: 1, tgDepth: 1),
                comment: "FFN gate",
                dynamicGridH: .seqLenCeilDiv(8)
            )),
        ]

        CooperativeNormFusionPass().run(&ops)

        #expect(ops.count == 2)
        guard case .dispatch(let scaleOnly) = ops[0],
              case .dispatch(let scaledConsumer) = ops[1]
        else {
            Issue.record("Expected scale-only and norm-scale affine dispatches")
            return
        }

        #expect(scaleOnly.pipeline == .rmsNormScaleOnlyD2560Batched)
        #expect(scaleOnly.dynamicGridW == .seqLen)
        if case let .threadgroups(width, height, depth, tgWidth, tgHeight, tgDepth) = scaleOnly.dispatch {
            #expect(width == 1)
            #expect(height == 1)
            #expect(depth == 1)
            #expect(tgWidth == 1024)
            #expect(tgHeight == 1)
            #expect(tgDepth == 1)
        } else {
            Issue.record("Expected scale-only dispatch to use threadgroups")
        }

        #expect(scaledConsumer.pipeline == .normScaleAffineMatvecC2560R10240G128Batched)
        #expect(scaledConsumer.buffers.count == 8)
        #expect(scaledConsumer.buffers[0].slot == .fixed(SmeltFixedSlot.normScaleScratch.rawValue))
        #expect(scaledConsumer.buffers[1].slot == .variable("cur"))
        #expect(scaledConsumer.buffers[2].slot == .fixed(30))
        #expect(scaledConsumer.buffers[3].slot == .fixed(8))
        #expect(scaledConsumer.buffers[7].slot == .fixed(13))
        #expect(scaledConsumer.constants.count == 1)
        #expect(scaledConsumer.constants[0].bindingIndex == 8)
        #expect(scaledConsumer.dynamicGridH == .seqLenCeilDiv(8))
    }

    @Test("Default optimizer rewrites E4B prefill K V dual projection through norm scale")
    func defaultOptimizerRewritesE4BPrefillKVDualProjection() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .rmsNorm1PWD2560Batched,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 30, offset: 1234, index: 1),
                    SmeltBufferBinding(slot: 8, index: 2),
                ],
                constants: [],
                dispatch: .threadgroups(width: 1, height: 1, depth: 1, tgWidth: 1024, tgHeight: 1, tgDepth: 1),
                comment: "input norm",
                dynamicGridW: .seqLen
            )),
            .traceMarker(label: "L0.input_norm", bufferSlot: 8),
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvecC2560R2048G128Batched,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 8, index: 3),
                    SmeltBufferBinding(slot: 20, index: 4),
                ],
                constants: [
                    SmeltConstantBinding(expression: "__seqLen__", type: .uint32, index: 5),
                ],
                dispatch: .threadgroups(width: 256, height: 1, depth: 1, tgWidth: 64, tgHeight: 1, tgDepth: 1),
                comment: "Q projection",
                dynamicGridH: .seqLenCeilDiv(8)
            )),
            .traceMarker(label: "L0.q_proj", bufferSlot: 20),
            .dispatch(SmeltDispatch(
                pipeline: .fusedDualAffineMatvecC2560R512G128Batched,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 40, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 50, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 60, index: 2),
                    SmeltBufferBinding(slot: 30, offset: 70, index: 3),
                    SmeltBufferBinding(slot: 30, offset: 80, index: 4),
                    SmeltBufferBinding(slot: 30, offset: 90, index: 5),
                    SmeltBufferBinding(slot: 8, index: 6),
                    SmeltBufferBinding(slot: 21, index: 7),
                    SmeltBufferBinding(slot: 22, index: 8),
                ],
                constants: [],
                dispatch: .threadgroups(width: 64, height: 1, depth: 1, tgWidth: 64, tgHeight: 1, tgDepth: 1),
                comment: "K+V projection",
                dynamicGridH: .seqLen
            )),
        ]

        let stats = SmeltDispatchOptimizer.optimize(&ops, planner: .auto)

        #expect(stats.rewriteCounts[SmeltFusionRule.normScaleDualAffineConsumer.rawValue] == 1)
        #expect(ops.count == 6)
        guard case .dispatch(let norm) = ops[0],
              case .traceMarker(let normLabel, let normBufferSlot) = ops[1],
              case .dispatch(let qProjection) = ops[2],
              case .traceMarker(let qLabel, let qBufferSlot) = ops[3],
              case .dispatch(let scaleOnly) = ops[4],
              case .dispatch(let fusedKV) = ops[5]
        else {
            Issue.record("Expected norm, Q, inserted scale-only, and norm-scale K+V")
            return
        }

        #expect(norm.pipeline == .rmsNorm1PWD2560Batched)
        #expect(normLabel == "L0.input_norm")
        #expect(normBufferSlot == 8)
        #expect(qProjection.pipeline == .affineMatvecC2560R2048G128Batched)
        #expect(qLabel == "L0.q_proj")
        #expect(qBufferSlot == 20)

        #expect(scaleOnly.pipeline == .rmsNormScaleOnlyD2560Batched)
        #expect(scaleOnly.buffers[0].slot == .variable("cur"))
        #expect(scaleOnly.buffers[1].slot == .fixed(SmeltFixedSlot.normScaleScratch.rawValue))
        #expect(scaleOnly.dynamicGridW == .seqLen)

        #expect(fusedKV.pipeline == .normScaleFusedDualAffineMatvecC2560R512G128Batched)
        #expect(fusedKV.buffers.count == 11)
        #expect(fusedKV.buffers[0].slot == .fixed(SmeltFixedSlot.normScaleScratch.rawValue))
        #expect(fusedKV.buffers[1].slot == .variable("cur"))
        #expect(fusedKV.buffers[2].slot == .fixed(30))
        #expect(fusedKV.buffers[3].slot == .fixed(30))
        #expect(fusedKV.buffers[9].slot == .fixed(21))
        #expect(fusedKV.buffers[10].slot == .fixed(22))
        #expect(fusedKV.constants.isEmpty)
        #expect(fusedKV.dynamicGridH == .seqLen)
    }

    @Test("Default optimizer preserves overlapping E4B batched dual FFN fusion")
    func defaultOptimizerPreservesOverlappingE4BDualFFNFusion() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .rmsNorm1PWD2560Batched,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 30, offset: 1234, index: 1),
                    SmeltBufferBinding(slot: 8, index: 2),
                ],
                constants: [],
                dispatch: .threadgroups(width: 1, height: 1, depth: 1, tgWidth: 320, tgHeight: 1, tgDepth: 1),
                comment: "FFN norm",
                dynamicGridW: .seqLen
            )),
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvecC2560R10240G128Batched,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 8, index: 3),
                    SmeltBufferBinding(slot: 13, index: 4),
                ],
                constants: [
                    SmeltConstantBinding(expression: "__seqLen__", type: .uint32, index: 5),
                ],
                dispatch: .threadgroups(width: 1280, height: 1, depth: 1, tgWidth: 64, tgHeight: 1, tgDepth: 1),
                comment: "FFN gate",
                dynamicGridH: .seqLenCeilDiv(8)
            )),
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvecC2560R10240G128Batched,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 40, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 50, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 60, index: 2),
                    SmeltBufferBinding(slot: 8, index: 3),
                    SmeltBufferBinding(slot: 14, index: 4),
                ],
                constants: [
                    SmeltConstantBinding(expression: "__seqLen__", type: .uint32, index: 5),
                ],
                dispatch: .threadgroups(width: 1280, height: 1, depth: 1, tgWidth: 64, tgHeight: 1, tgDepth: 1),
                comment: "FFN up",
                dynamicGridH: .seqLenCeilDiv(8)
            )),
            .dispatch(SmeltDispatch(
                pipeline: .gegluFused,
                buffers: [
                    SmeltBufferBinding(slot: 13, index: 0),
                    SmeltBufferBinding(slot: 14, index: 1),
                    SmeltBufferBinding(slot: 15, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "__seqLen__*10240", type: .uint32, index: 3),
                ],
                dispatch: .threadgroups(width: 10240, height: 1, depth: 1, tgWidth: 256, tgHeight: 1, tgDepth: 1),
                comment: "GeGLU",
                dynamicGridW: .seqLenMul(10240)
            )),
        ]

        let stats = SmeltDispatchOptimizer.optimize(&ops, planner: .auto)

        #expect(stats.rewriteCounts[SmeltFusionRule.dualMatvecActivation.rawValue] == 1)
        #expect(stats.rewriteCounts[SmeltFusionRule.cooperativeNormScaleConsumer.rawValue] == nil)
        #expect(stats.opportunities.contains {
            $0.pattern == "normConsumer"
                && $0.shape == "rms_norm_1pw_d2560_batched->affine_matvec_c2560_r10240_g128_batched"
                && $0.fusedKernelAvailable
        })
        #expect(ops.count == 2)
        guard case .dispatch(let norm) = ops[0],
              case .dispatch(let fused) = ops[1]
        else {
            Issue.record("Expected norm followed by fused dual FFN")
            return
        }
        #expect(norm.pipeline == .rmsNorm1PWD2560Batched)
        #expect(fused.pipeline == .fusedAffineGateUpGeGLUC2560R10240G128Batched)
    }

    @Test("Optimizer leaves specialized decode norm alone when the next dispatch does not consume it")
    func cooperativeNormFusionSkipsNonMatchingConsumer() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .rmsNorm1PWD2048,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 30, offset: 1234, index: 1),
                    SmeltBufferBinding(slot: 8, index: 2),
                ],
                constants: [],
                dispatch: .threadgroups(width: 1, height: 1, depth: 1, tgWidth: 256, tgHeight: 1, tgDepth: 1)
            )),
            .dispatch(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(slot: 0, index: 0),
                    SmeltBufferBinding(slot: 1, index: 1),
                    SmeltBufferBinding(slot: 2, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "2048", type: .uint32, index: 3),
                ],
                dispatch: .threads(width: 2048, height: 1, depth: 1, tgWidth: 1024, tgHeight: 1, tgDepth: 1)
            )),
        ]

        SmeltDispatchOptimizer.optimize(&ops)

        #expect(ops.count == 2)
        guard case .dispatch(let norm) = ops[0] else {
            Issue.record("Expected first op to remain a dispatch")
            return
        }
        #expect(norm.pipeline == .rmsNorm1PWD2048)
    }

    @Test("Residual-add fusion selects Qwen specialized fused affine down projection")
    func residualFusionSelectsQwenSpecializedFfnDownKernel() {
        var ops: [SmeltIROp] = [
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
                dispatch: .threadgroups(width: 256, height: 1, depth: 1, tgWidth: 64, tgHeight: 1, tgDepth: 1),
                comment: "FFN down projection",
                fcCols: 6144,
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
                dispatch: .threads(width: 2048, height: 1, depth: 1, tgWidth: 1024, tgHeight: 1, tgDepth: 1)
            )),
        ]

        FuseMatvecResidualAddPass().run(&ops)

        #expect(ops.count == 1)
        guard case .dispatch(let fused) = ops[0] else {
            Issue.record("Expected fused dispatch")
            return
        }

        #expect(fused.pipeline == .fusedAffineMatvecAddC6144R2048G64)
        #expect(fused.constants.isEmpty)
        #expect(fused.fcCols == nil)
        #expect(fused.fcGroupSize == nil)
    }

    @Test("Residual-add fusion selects exact ternary affine brick generically")
    func residualFusionSelectsExactTernaryAffineKernel() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .signedTernaryAffineMatvecG128Rows8,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 2),
                    SmeltBufferBinding(slot: 13, index: 3),
                    SmeltBufferBinding(slot: 14, index: 4),
                ],
                constants: [
                    SmeltConstantBinding(expression: "5120", type: .uint32, index: 5),
                    SmeltConstantBinding(expression: "17408", type: .uint32, index: 6),
                ],
                dispatch: .threadgroups(
                    width: 640, height: 1, depth: 1,
                    tgWidth: 64, tgHeight: 1, tgDepth: 1)
            )),
            .dispatch(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 14, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "5120", type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: 5120, height: 1, depth: 1,
                    tgWidth: 1024, tgHeight: 1, tgDepth: 1)
            )),
        ]

        FuseMatvecResidualAddPass().run(&ops)

        #expect(ops.count == 1)
        guard case .dispatch(let fused) = ops[0] else {
            Issue.record("Expected exact ternary residual fusion")
            return
        }
        #expect(fused.pipeline == .signedTernaryAffineMatvecAddG128Rows8)
        #expect(fused.buffers.count == 6)
        #expect(fused.buffers[4].slot == .variable("alt"))
        #expect(fused.buffers[5].slot == .variable("cur"))
        #expect(fused.constants.map(\.expression) == ["5120", "17408"])
    }

    @Test("Residual-add fusion selects Qwen specialized fused affine output projection")
    func residualFusionSelectsQwenSpecializedOutputKernel() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvecC2048R2048G64,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 9, index: 3),
                    SmeltBufferBinding(slot: 8, index: 4),
                ],
                constants: [],
                dispatch: .threadgroups(width: 256, height: 1, depth: 1, tgWidth: 64, tgHeight: 1, tgDepth: 1),
                comment: "Output projection"
            )),
            .dispatch(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 8, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "2048", type: .uint32, index: 3),
                ],
                dispatch: .threads(width: 2048, height: 1, depth: 1, tgWidth: 1024, tgHeight: 1, tgDepth: 1)
            )),
        ]

        FuseMatvecResidualAddPass().run(&ops)

        #expect(ops.count == 1)
        guard case .dispatch(let fused) = ops[0] else {
            Issue.record("Expected fused dispatch")
            return
        }

        #expect(fused.pipeline == .fusedAffineMatvecAddC2048R2048G64)
        #expect(fused.constants.isEmpty)
    }

    @Test("Residual-add fusion selects batched full QMM affine kernels")
    func residualFusionSelectsBatchedFullQMMKernel() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvecC8192R2048G64BatchedFull,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 9, index: 3),
                    SmeltBufferBinding(slot: 8, index: 4),
                ],
                constants: [
                    SmeltConstantBinding(expression: "__seqLen__", type: .uint32, index: 5),
                ],
                dispatch: .threadgroups(
                    width: 64,
                    height: 1,
                    depth: 1,
                    tgWidth: 128,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                dynamicGridH: .seqLenCeilDiv(16)
            )),
            .traceMarker(label: "L0.ffn_down", bufferSlot: 8),
            .dispatch(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 8, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "__seqLen__*2048", type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: 2_048,
                    height: 1,
                    depth: 1,
                    tgWidth: 1_024,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                dynamicGridW: .seqLenMul(2_048)
            )),
        ]

        FuseMatvecResidualAddPass().run(&ops)

        #expect(ops.count == 2)
        guard ops.count == 2 else { return }
        guard case .dispatch(let fused) = ops[0] else {
            Issue.record("Expected fused dispatch")
            return
        }
        guard case .traceMarker(let label, let bufferSlot) = ops[1] else {
            Issue.record("Expected trace marker to survive after fused dispatch")
            return
        }

        #expect(fused.pipeline == .fusedAffineMatvecAddC8192R2048G64BatchedFull)
        #expect(fused.buffers.count == 7)
        #expect(fused.buffers[4].slot == .fixed(8))
        #expect(fused.buffers[5].slot == .variable("cur"))
        #expect(fused.buffers[6].slot == .variable("alt"))
        #expect(fused.constants.count == 1)
        #expect(fused.constants[0].expression == "__seqLen__")
        #expect(fused.constants[0].bindingIndex == 7)
        #expect(fused.dynamicGridH == .seqLenCeilDiv(16))
        #expect(fused.dynamicGridW == nil)
        #expect(label == "L0.ffn_down")
        #expect(bufferSlot == 8)
    }

    @Test("Residual-add fusion preserves generated named affine kernels")
    func residualFusionPreservesGeneratedNamedAffineKernel() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvec,
                pipelineNameOverride: "affine_matvec_c11008_r2048_g64_batched_full",
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 9, index: 3),
                    SmeltBufferBinding(slot: 8, index: 4),
                ],
                constants: [
                    SmeltConstantBinding(expression: "__seqLen__", type: .uint32, index: 5),
                ],
                dispatch: .threadgroups(
                    width: 64,
                    height: 1,
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
                    SmeltBufferBinding(slot: 8, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "__seqLen__*2048", type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: 2_048,
                    height: 1,
                    depth: 1,
                    tgWidth: 1_024,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                dynamicGridW: .seqLenMul(2_048)
            )),
        ]

        FuseMatvecResidualAddPass().run(&ops)

        #expect(ops.count == 2)
        guard case .dispatch(let generated) = ops[0],
              case .dispatch(let add) = ops[1]
        else {
            Issue.record("Expected generated affine and residual add to remain separate")
            return
        }

        #expect(generated.pipeline == .affineMatvec)
        #expect(generated.pipelineNameOverride == "affine_matvec_c11008_r2048_g64_batched_full")
        #expect(add.pipeline == .elementwiseAdd)
    }

    @Test("Batched full residual fusion does not cross trace marker when output aliases marker buffer")
    func residualFusionDoesNotCrossTraceMarkerWhenOutputAliasesMatvecBuffer() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvecC8192R2048G64BatchedFull,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 9, index: 3),
                    SmeltBufferBinding(slot: 8, index: 4),
                ],
                constants: [
                    SmeltConstantBinding(expression: "__seqLen__", type: .uint32, index: 5),
                ],
                dispatch: .threadgroups(
                    width: 64,
                    height: 1,
                    depth: 1,
                    tgWidth: 128,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                dynamicGridH: .seqLenCeilDiv(16)
            )),
            .traceMarker(label: "L0.ffn_down", bufferSlot: 8),
            .dispatch(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 8, index: 1),
                    SmeltBufferBinding(slot: 8, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "__seqLen__*2048", type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: 2_048,
                    height: 1,
                    depth: 1,
                    tgWidth: 1_024,
                    tgHeight: 1,
                    tgDepth: 1
                ),
                dynamicGridW: .seqLenMul(2_048)
            )),
        ]

        FuseMatvecResidualAddPass().run(&ops)

        #expect(ops.count == 3)
        guard case .dispatch(let matvec) = ops[0],
              case .traceMarker(let label, let bufferSlot) = ops[1],
              case .dispatch(let add) = ops[2]
        else {
            Issue.record("Expected unfused matvec, trace marker, and add")
            return
        }

        #expect(matvec.pipeline == .affineMatvecC8192R2048G64BatchedFull)
        #expect(add.pipeline == .elementwiseAdd)
        #expect(label == "L0.ffn_down")
        #expect(bufferSlot == 8)
    }

    @Test("Residual-add fusion selects rows4 specialized output kernel")
    func residualFusionSelectsRows4SpecializedOutputKernel() {
        var ops: [SmeltIROp] = [
            .dispatch(SmeltDispatch(
                pipeline: .affineMatvecC2048R1536G128Rows4,
                buffers: [
                    SmeltBufferBinding(slot: 30, offset: 10, index: 0),
                    SmeltBufferBinding(slot: 30, offset: 20, index: 1),
                    SmeltBufferBinding(slot: 30, offset: 30, index: 2),
                    SmeltBufferBinding(slot: 9, index: 3),
                    SmeltBufferBinding(slot: 8, index: 4),
                ],
                constants: [],
                dispatch: .threadgroups(width: 384, height: 1, depth: 1, tgWidth: 64, tgHeight: 1, tgDepth: 1),
                comment: "output projection"
            )),
            .dispatch(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(variableSlot: "cur", index: 0),
                    SmeltBufferBinding(slot: 8, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "1536", type: .uint32, index: 3),
                ],
                dispatch: .threads(width: 1536, height: 1, depth: 1, tgWidth: 1024, tgHeight: 1, tgDepth: 1)
            )),
        ]

        FuseMatvecResidualAddPass().run(&ops)

        #expect(ops.count == 1)
        guard case .dispatch(let fused) = ops[0] else {
            Issue.record("Expected fused residual-add dispatch")
            return
        }

        #expect(fused.pipeline == .fusedAffineMatvecAddC2048R1536G128Rows4)
        #expect(fused.constants.isEmpty)
        #expect(fused.fcCols == nil)
        #expect(fused.fcGroupSize == nil)
    }
}
