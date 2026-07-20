import Foundation
import Testing
@testable import SmeltCompiler
@testable import SmeltSchema

// MARK: - PrefillEmitter tests

/// Helper: Qwen 3.5 2B IR with Metal prefill engine.
private func qwen35MetalPrefill(batchSize: Int = 64) -> SmeltModelIR {
    SmeltModelIR(
        modelName: SmeltModelIR.qwen35_2B.modelName,
        config: SmeltModelIR.qwen35_2B.config,
        layerPattern: SmeltModelIR.qwen35_2B.layerPattern,
        quantization: SmeltModelIR.qwen35_2B.quantization,
        loading: SmeltModelIR.qwen35_2B.loading,
        prefill: SmeltPrefillConfig(
            engine: "metal",
            modelPath: "",
            cachePath: "cache",
            maxBatchSize: batchSize,
            handoffFamilies: ["conv_state", "rec_state", "key_cache", "value_cache", "rope"]
        )
    )
}

private func qwen35MetalVerifyArgmax(
    batchSize: Int = 64,
    verifyTokenCapacity: Int? = nil
) -> SmeltModelIR {
    let source = qwen35MetalPrefill(batchSize: batchSize)
    return SmeltModelIR(
        modelName: source.modelName,
        config: source.config,
        layerPattern: source.layerPattern,
        quantization: SmeltQuantizationConfig(
            strategy: .binary1,
            groupSize: 128,
            excludePatterns: [
                "conv1d_weight", "A_log", "dt_bias", "*_norm_weight",
            ],
            quantizeEmbedding: true
        ),
        loading: source.loading,
        prefill: SmeltPrefillConfig(
            engine: "metal",
            modelPath: "",
            cachePath: "cache",
            maxBatchSize: batchSize,
            handoffFamilies: [
                "conv_state", "rec_state", "key_cache", "value_cache", "rope",
            ],
            verifyArgmax: true,
            verifyTokenCapacity: verifyTokenCapacity
        )
    )
}

private func qwen35MetalPrefillAffineEmbedding(batchSize: Int = 64) -> SmeltModelIR {
    let base = qwen35MetalPrefill(batchSize: batchSize)
    return SmeltModelIR(
        modelName: base.modelName,
        config: base.config,
        layerPattern: base.layerPattern,
        quantization: SmeltQuantizationConfig(
            strategy: .affineU4,
            groupSize: 64,
            excludePatterns: ["conv1d_weight", "A_log", "dt_bias", "*_norm_weight"],
            quantizeEmbedding: true
        ),
        loading: base.loading,
        prefill: base.prefill
    )
}

private func qwen35MetalPrefill4B(batchSize: Int = 64) -> SmeltModelIR {
    SmeltModelIR(
        modelName: SmeltModelIR.qwen35_4B.modelName,
        config: SmeltModelIR.qwen35_4B.config,
        layerPattern: SmeltModelIR.qwen35_4B.layerPattern,
        quantization: SmeltModelIR.qwen35_4B.quantization,
        loading: SmeltModelIR.qwen35_4B.loading,
        prefill: SmeltPrefillConfig(
            engine: "metal",
            modelPath: "",
            cachePath: "cache",
            maxBatchSize: batchSize,
            handoffFamilies: ["conv_state", "rec_state", "key_cache", "value_cache", "rope"]
        )
    )
}

private func qwen35MetalPrefill0808(batchSize: Int = 64) -> SmeltModelIR {
    SmeltModelIR(
        modelName: SmeltModelIR.qwen35_0_8B.modelName,
        config: SmeltModelIR.qwen35_0_8B.config,
        layerPattern: SmeltModelIR.qwen35_0_8B.layerPattern,
        quantization: SmeltModelIR.qwen35_0_8B.quantization,
        loading: SmeltModelIR.qwen35_0_8B.loading,
        prefill: SmeltPrefillConfig(
            engine: "metal",
            modelPath: "",
            cachePath: "cache",
            maxBatchSize: batchSize,
            handoffFamilies: ["conv_state", "rec_state", "key_cache", "value_cache", "rope"]
        )
    )
}

@Test func mlxVectorAttentionAdmissionUsesShapeAndGQANotFamily() {
    #expect(PrefillEmitter.mlxVectorAttentionMaxQueryLength(
        headDim: 256, gqaRatio: 6
    ) == 5)
    #expect(PrefillEmitter.mlxVectorAttentionMaxQueryLength(
        headDim: 256, gqaRatio: 4
    ) == 8)
    #expect(PrefillEmitter.mlxVectorAttentionMaxQueryLength(
        headDim: 96, gqaRatio: 8
    ) == 4)
    #expect(PrefillEmitter.mlxVectorAttentionMaxQueryLength(
        headDim: 80, gqaRatio: 1
    ) == nil)
    #expect(PrefillEmitter.mlxVectorAttentionMaxQueryLength(
        headDim: 128, gqaRatio: 33
    ) == nil)
}

@Test func prefillEmitterProducesDispatchRecords() throws {
    let ir = qwen35MetalPrefill()
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try PrefillEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )

    // Should produce a non-trivial number of dispatch records
    let records = result.dispatchRecords
    #expect(records.count > 300, "Expected >300 prefill dispatch records, got \(records.count)")
    #expect(result.optimizationStats.rewriteCounts.isEmpty)
    #expect(!result.optimizationStats.opportunities.isEmpty)
    fputs("  Prefill dispatch table: \(records.count) records\n", stderr)
}

@Test func verifyArgmaxCheckpointsRecurrentStateWithoutPromptBatchExplosion() throws {
    let ir = qwen35MetalVerifyArgmax(
        batchSize: 256,
        verifyTokenCapacity: 32
    )
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let convHistory = plan.slots.filter {
        $0.name.hasPrefix("convStateHistory_")
    }
    let recHistory = plan.slots.filter {
        $0.name.hasPrefix("recStateHistory_")
    }
    #expect(convHistory.count == ir.numDeltaLayers)
    #expect(recHistory.count == ir.numDeltaLayers)
    #expect(convHistory.allSatisfy { $0.shape.first == 33 })
    #expect(recHistory.allSatisfy { $0.shape.first == 33 })

    let result = try PrefillEmitter.generate(
        ir: ir,
        plan: plan,
        weightLayout: layout,
        lmHeadMode: .verifyArgmaxOnly
    )
    let dispatches = result.dispatchRecords.filter {
        $0.opKind == SmeltDispatchRecord.opDispatch
    }
    let convPipeline = UInt16(
        SmeltPipeline.conv1dUpdateSiluPrefillCheckpoint.rawValue
    )
    let recPipeline = UInt16(
        SmeltPipeline.deltanetRecurrenceMlxPrefillCheckpoint.rawValue
    )
    let convRecords = dispatches.filter { $0.pipeline == convPipeline }
    let recRecords = dispatches.filter { $0.pipeline == recPipeline }
    #expect(convRecords.count == ir.numDeltaLayers)
    #expect(recRecords.count == ir.numDeltaLayers)

    let firstConv = try #require(convRecords.first)
    let firstRec = try #require(recRecords.first)
    #expect(firstConv.bufferCount == 4)
    #expect(firstRec.bufferCount == 8)
    #expect(getBuffer(firstConv, index: 3).bindingIndex == 6)
    #expect(getBuffer(firstConv, index: 3).slot == Int16(convHistory[0].index))
    #expect(getBuffer(firstRec, index: 7).bindingIndex == 11)
    #expect(getBuffer(firstRec, index: 7).slot == Int16(recHistory[0].index))

    let promptResult = try PrefillEmitter.generate(
        ir: ir,
        plan: plan,
        weightLayout: layout
    )
    #expect(promptResult.dispatchRecords.allSatisfy {
        $0.opKind != SmeltDispatchRecord.opDispatch
            || ($0.pipeline != convPipeline && $0.pipeline != recPipeline)
    })
}

@Test func prefillEmitterRecordCountMatchesExpected() throws {
    let B = 64
    let ir = qwen35MetalPrefill(batchSize: B)
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try PrefillEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )

    let records = result.dispatchRecords
    let dispatches = records.filter { $0.opKind == SmeltDispatchRecord.opDispatch }
    let swaps = records.filter { $0.opKind == SmeltDispatchRecord.opSwap }

    fputs(
        "  Prefill: \(dispatches.count) dispatches + \(swaps.count) swaps"
            + " = \(records.count) total\n",
        stderr
    )

    // 24 layers × 2 swaps per layer = 48 swaps (same as decode)
    #expect(swaps.count == 48, "Expected 48 swaps, got \(swaps.count)")

    // With split specialized recurrence + fused RoPE/KV: still comfortably sub-600.
    #expect(dispatches.count > 300, "Too few dispatches: \(dispatches.count)")
    #expect(dispatches.count < 600, "Too many dispatches: \(dispatches.count)")
}

@Test func prefillEmitterLMHeadUsesMatvecNotMatmul() throws {
    let ir = qwen35MetalPrefill()
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try PrefillEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )

    let records = result.dispatchRecords
    // The last few records should be: LM head (matvec) + argmax. Large
    // vocabs lower argmax to partials + key-reduce, so anchor on the
    // first argmax dispatch rather than assuming the legacy single pass.
    let argmaxPipeline = UInt16(SmeltPipeline.argmaxFP16.rawValue)
    let argmaxPartialsPipeline = UInt16(SmeltPipeline.argmaxFP16Partials.rawValue)
    guard let argmaxIdx = records.lastIndex(where: {
        ($0.pipeline == argmaxPipeline || $0.pipeline == argmaxPartialsPipeline)
            && $0.opKind == SmeltDispatchRecord.opDispatch
    }) else {
        Issue.record("No argmax dispatch found in prefill table")
        return
    }

    // The dispatch right before argmax should be the LM head
    let lmHeadIdx = argmaxIdx - 1
    let lmHead = records[lmHeadIdx]

    // Should be fused_lut_matvec (decode kernel), NOT fused_lut_matmul (batched)
    let matvecPipeline = UInt16(SmeltPipeline.fusedLutMatvec.rawValue)
    let matmulPipeline = UInt16(SmeltPipeline.fusedLutMatmul.rawValue)

    #expect(
        lmHead.pipeline == matvecPipeline || lmHead.pipeline == UInt16(SmeltPipeline.fp16Matvec.rawValue),
        "LM head should use matvec (not matmul), got pipeline \(lmHead.pipeline)"
    )
    #expect(
        lmHead.pipeline != matmulPipeline,
        "LM head must NOT use batched matmul (wastes 30MB logits buffer)"
    )
}

@Test func prefillEmitterUsesCorrectBatchedPipelines() throws {
    let ir = qwen35MetalPrefill()
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try PrefillEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )

    let records = result.dispatchRecords
    let dispatches = records.filter { $0.opKind == SmeltDispatchRecord.opDispatch }

    // Count pipeline usage
    var pipelineCounts: [UInt16: Int] = [:]
    for rec in dispatches {
        pipelineCounts[rec.pipeline, default: 0] += 1
    }

    // Should use batched kernels
    let matmulPipeline = UInt16(SmeltPipeline.fusedLutMatmul.rawValue)
    let batchedNormPipeline = UInt16(SmeltPipeline.rmsNorm1PWBatched.rawValue)
    let embGatherPipeline = UInt16(SmeltPipeline.embeddingGatherBatched.rawValue)
    let attnVectorPipeline = UInt16(
        SmeltPipeline.attentionPrefillSDPAVectorD256.rawValue
    )
    let attnFallbackPipeline = UInt16(
        SmeltPipeline.attentionPrefillMLXFallbackD256.rawValue
    )
    let perHeadNormBatchedPipeline = UInt16(SmeltPipeline.perHeadRmsNorm1PWBatched.rawValue)

    #expect(
        (pipelineCounts[matmulPipeline] ?? 0) > 0,
        "Should use fused_lut_matmul for batched projections"
    )
    #expect(
        (pipelineCounts[batchedNormPipeline] ?? 0) > 0,
        "Should use rms_norm_1pw_batched"
    )
    #expect(
        (pipelineCounts[embGatherPipeline] ?? 0) > 0,
        "Should use embedding_gather_batched"
    )
    #expect(
        (pipelineCounts[attnVectorPipeline] ?? 0) == 6,
        "Should have one guarded MLX vector attention route per attn layer"
    )
    #expect(
        (pipelineCounts[attnFallbackPipeline] ?? 0) == 6,
        "Should have one guarded MLX fallback attention route per attn layer"
    )
    #expect(
        (pipelineCounts[perHeadNormBatchedPipeline] ?? 0) == 12,
        "Should have 12 batched per-head RMS norm dispatches (Q and K for 6 attention layers)"
    )

    // DeltaNet prefill composes conv, the shared Q/K scale brick, and the
    // shape-specialized recurrence without family-specific normalization.
    let fusedRecPipeline = UInt16(SmeltPipeline.deltanetRecurrencePrefill.rawValue)
    let convPrefillPipeline = UInt16(SmeltPipeline.conv1dUpdateSilu6144x4Prefill.rawValue)
    let qkNormPrefillPipeline = UInt16(SmeltPipeline.rmsScaleQK.rawValue)
    let qNormPrefillPipeline = UInt16(SmeltPipeline.l2NormalizeQD128C6144H16Prefill.rawValue)
    let kNormPrefillPipeline = UInt16(SmeltPipeline.l2NormalizeKD128C6144H16Prefill.rawValue)
    let mlxRecPrefillPipeline = UInt16(SmeltPipeline.deltanetRecurrenceMlxPrefillD128H16.rawValue)
    #expect(
        (pipelineCounts[convPrefillPipeline] ?? 0) == 18,
        "Should have 18 specialized conv1d prefill dispatches"
    )
    #expect(
        (pipelineCounts[qkNormPrefillPipeline] ?? 0) == 18,
        "Should have 18 fused Q/K RMS-scale prefill dispatches"
    )
    #expect(
        (pipelineCounts[qNormPrefillPipeline] ?? 0) == 0
            && (pipelineCounts[kNormPrefillPipeline] ?? 0) == 0,
        "Should not fall back to split shape-specific Q/K normalization"
    )
    #expect(
        (pipelineCounts[mlxRecPrefillPipeline] ?? 0) == 18,
        "Should have 18 MLX-style tiled recurrence prefill dispatches"
    )
    #expect(
        (pipelineCounts[fusedRecPipeline] ?? 0) == 0,
        "Qwen prefill should not use the generic fused recurrence dispatch"
    )

    // No generic unrolled conv1d or state_decay kernels on the optimized path.
    let conv1dPipeline = UInt16(SmeltPipeline.conv1dUpdateSilu.rawValue)
    let l2NormalizePipeline = UInt16(SmeltPipeline.l2Normalize.rawValue)
    let stateDecayPipeline = UInt16(SmeltPipeline.stateDecay.rawValue)
    #expect(
        (pipelineCounts[conv1dPipeline] ?? 0) == 0,
        "Should have 0 generic conv1d dispatches on the specialized path"
    )
    #expect(
        (pipelineCounts[l2NormalizePipeline] ?? 0) == 0,
        "Should have 0 generic l2_normalize dispatches on the specialized path"
    )
    #expect(
        (pipelineCounts[stateDecayPipeline] ?? 0) == 0,
        "Should have 0 state_decay dispatches"
    )

    fputs("  Pipeline usage:\n", stderr)
    for (pipe, count) in pipelineCounts.sorted(by: { $0.key < $1.key }) {
        let name = Int(pipe) < SmeltKernelCatalog.pipelineNames.count
            ? SmeltKernelCatalog.pipelineNames[Int(pipe)]
            : "pipeline_\(pipe)"
        fputs("    \(name): \(count)\n", stderr)
    }
}

@Test func prefillEmitterUsesQMMAffinePipelinesForAffineQwen() throws {
    let ir = qwen35MetalPrefillAffineEmbedding()
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try PrefillEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )

    let dispatches = result.dispatchRecords.filter { $0.opKind == SmeltDispatchRecord.opDispatch }
    var pipelineCounts: [UInt16: Int] = [:]
    for rec in dispatches {
        pipelineCounts[rec.pipeline, default: 0] += 1
    }

    let swigluPipeline = UInt16(SmeltPipeline.swigluFused.rawValue)
    let qmmAffine2048Pipeline = UInt16(SmeltPipeline.affineMatvecC2048R2048G64BatchedFull.rawValue)
    let qmmAffine6144Pipeline = UInt16(SmeltPipeline.affineMatvecC2048R6144G64BatchedFull.rawValue)
    let qmmAffine4096Pipeline = UInt16(SmeltPipeline.affineMatvecC2048R4096G64BatchedFull.rawValue)
    let qmmAffineDownPipeline = UInt16(SmeltPipeline.affineMatvecC6144R2048G64BatchedFull.rawValue)
    let qmmFusedResidual2048Pipeline = UInt16(SmeltPipeline.fusedAffineMatvecAddC2048R2048G64BatchedFull.rawValue)
    let qmmFusedResidualDownPipeline = UInt16(SmeltPipeline.fusedAffineMatvecAddC6144R2048G64BatchedFull.rawValue)
    let qmmFusedFFNPipeline = UInt16(SmeltPipeline.fusedAffineGateUpSwigluC2048R6144G64BatchedFull.rawValue)
    let normScaleAffine6144Pipeline = UInt16(SmeltPipeline.normScaleAffineMatvecC2048R6144G64BatchedFull.rawValue)
    let normScaleAffine4096Pipeline = UInt16(SmeltPipeline.normScaleAffineMatvecC2048R4096G64BatchedFull.rawValue)
    let normScaleFusedFFNPipeline = UInt16(SmeltPipeline.normScaleAffineGateUpSwigluC2048R6144G64BatchedFull.rawValue)
    let scalarAffine2048Pipeline = UInt16(SmeltPipeline.affineMatvecC2048R2048G64Batched.rawValue)
    let scalarAffine6144Pipeline = UInt16(SmeltPipeline.affineMatvecC2048R6144G64Batched.rawValue)
    let scalarAffine4096Pipeline = UInt16(SmeltPipeline.affineMatvecC2048R4096G64Batched.rawValue)
    let scalarAffineDownPipeline = UInt16(SmeltPipeline.affineMatvecC6144R2048G64Batched.rawValue)

    #expect((pipelineCounts[qmmAffine2048Pipeline] ?? 0) > 0, "Canonical affine Qwen prefill should emit qmm 2048x2048 affine kernels")
    #expect((pipelineCounts[normScaleAffine6144Pipeline] ?? 0) > 0, "Canonical affine Qwen prefill should emit norm-scale qmm 2048x6144 affine kernels")
    #expect((pipelineCounts[normScaleAffine4096Pipeline] ?? 0) > 0, "Canonical affine Qwen prefill should emit norm-scale qmm 2048x4096 affine kernels")
    #expect((pipelineCounts[qmmAffine6144Pipeline] ?? 0) == 0, "Canonical affine Qwen prefill should consume staged qmm 2048x6144 affine kernels into norm-scale fusion")
    #expect((pipelineCounts[qmmAffine4096Pipeline] ?? 0) == 0, "Canonical affine Qwen prefill should consume staged qmm 2048x4096 affine kernels into norm-scale fusion")
    #expect((pipelineCounts[qmmAffineDownPipeline] ?? 0) == 0, "Canonical affine Qwen prefill should fuse qmm 6144x2048 residual kernels")
    #expect((pipelineCounts[qmmFusedResidual2048Pipeline] ?? 0) > 0, "Canonical affine Qwen prefill should emit fused qmm 2048x2048 residual kernels")
    #expect((pipelineCounts[qmmFusedResidualDownPipeline] ?? 0) > 0, "Canonical affine Qwen prefill should emit fused qmm 6144x2048 residual kernels")
    #expect((pipelineCounts[normScaleFusedFFNPipeline] ?? 0) > 0, "Canonical affine Qwen prefill should emit the norm-scale fused qmm FFN pipeline")
    #expect((pipelineCounts[qmmFusedFFNPipeline] ?? 0) == 0, "Canonical affine Qwen prefill should consume staged fused qmm FFN kernels into norm-scale fusion")
    #expect((pipelineCounts[swigluPipeline] ?? 0) == 0, "Canonical affine Qwen prefill should not emit split SwiGLU for the qmm FFN path")
    #expect((pipelineCounts[scalarAffine2048Pipeline] ?? 0) == 0, "Canonical affine Qwen prefill should not emit scalar batched 2048x2048 affine kernels")
    #expect((pipelineCounts[scalarAffine6144Pipeline] ?? 0) == 0, "Canonical affine Qwen prefill should not emit scalar batched 2048x6144 affine kernels")
    #expect((pipelineCounts[scalarAffine4096Pipeline] ?? 0) == 0, "Canonical affine Qwen prefill should not emit scalar batched 2048x4096 affine kernels")
    #expect((pipelineCounts[scalarAffineDownPipeline] ?? 0) == 0, "Canonical affine Qwen prefill should not emit scalar batched 6144x2048 affine kernels")
}

@Test func prefillEmitterUsesSpecializedRecurrenceForQwen4B() throws {
    let ir = qwen35MetalPrefill4B()
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try PrefillEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )

    let dispatches = result.dispatchRecords.filter { $0.opKind == SmeltDispatchRecord.opDispatch }
    var pipelineCounts: [UInt16: Int] = [:]
    for rec in dispatches {
        pipelineCounts[rec.pipeline, default: 0] += 1
    }

    let genericConvPrefill = UInt16(SmeltPipeline.conv1dUpdateSiluPrefill.rawValue)
    let qkNormPrefill = UInt16(SmeltPipeline.rmsScaleQK.rawValue)
    let genericQNormPrefill = UInt16(SmeltPipeline.l2NormalizeQPrefill.rawValue)
    let genericKNormPrefill = UInt16(SmeltPipeline.l2NormalizeKPrefill.rawValue)
    let genericRecPrefill = UInt16(SmeltPipeline.deltanetRecurrenceMlxPrefill.rawValue)
    let oldFusedRecPrefill = UInt16(SmeltPipeline.deltanetRecurrencePrefill.rawValue)
    let qwenConvPrefill = UInt16(SmeltPipeline.conv1dUpdateSilu6144x4Prefill.rawValue)
    let qwenRecPrefill = UInt16(SmeltPipeline.deltanetRecurrenceMlxPrefillD128H16.rawValue)
    let qwen4BRecPrefill = UInt16(SmeltPipeline.deltanetRecurrenceMlxPrefillD128H32QK16.rawValue)
    let qwen4BFusedAttnResidual = UInt16(SmeltPipeline.fusedAffineMatvecAddC4096R2560G64BatchedFull.rawValue)
    let qwen4BFusedFFNResidual = UInt16(SmeltPipeline.fusedAffineMatvecAddC9216R2560G64BatchedFull.rawValue)

    #expect((pipelineCounts[genericConvPrefill] ?? 0) == 24, "Qwen 4B should emit one generic conv prefill dispatch per delta layer")
    #expect((pipelineCounts[qkNormPrefill] ?? 0) == 24, "Qwen 4B should emit one fused Q/K RMS-scale prefill dispatch per delta layer")
    #expect((pipelineCounts[genericQNormPrefill] ?? 0) == 0, "Qwen 4B should not emit split generic Q norm prefill dispatches")
    #expect((pipelineCounts[genericKNormPrefill] ?? 0) == 0, "Qwen 4B should not emit split generic K norm prefill dispatches")
    #expect((pipelineCounts[qwen4BRecPrefill] ?? 0) == 24, "Qwen 4B should emit one specialized tiled recurrence prefill dispatch per delta layer")
    #expect((pipelineCounts[qwen4BFusedAttnResidual] ?? 0) > 0, "Qwen 4B should emit fused qmm attention residual kernels")
    #expect((pipelineCounts[qwen4BFusedFFNResidual] ?? 0) > 0, "Qwen 4B should emit fused qmm FFN residual kernels")
    #expect((pipelineCounts[genericRecPrefill] ?? 0) == 0, "Qwen 4B should not emit the generic tiled recurrence prefill dispatch")
    #expect((pipelineCounts[oldFusedRecPrefill] ?? 0) == 0, "Qwen 4B should not fall back to the old fused recurrence prefill kernel")
    #expect((pipelineCounts[qwenConvPrefill] ?? 0) == 0, "Qwen 4B should not use the 2B/0.8B conv prefill specialization")
    #expect((pipelineCounts[qwenRecPrefill] ?? 0) == 0, "Qwen 4B should not use the 2B/0.8B recurrence prefill specialization")
}

@Test func prefillEmitterUsesSpecializedBatchedDeltaNormForQwen0808() throws {
    let ir = qwen35MetalPrefill0808()
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try PrefillEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )

    let dispatches = result.dispatchRecords.filter { $0.opKind == SmeltDispatchRecord.opDispatch }
    var pipelineCounts: [UInt16: Int] = [:]
    for rec in dispatches {
        pipelineCounts[rec.pipeline, default: 0] += 1
    }

    let specialized = UInt16(SmeltPipeline.rmsNormGatedD128Batched.rawValue)
    let generic = UInt16(SmeltPipeline.rmsNormGated.rawValue)

    #expect(
        (pipelineCounts[specialized] ?? 0) == 18,
        "Qwen 0.8B should emit one specialized batched gated norm per Delta layer"
    )
    #expect(
        (pipelineCounts[generic] ?? 0) == 0,
        "Qwen 0.8B should not emit the generic gated norm on the Delta prefill path"
    )
}

@Test func prefillEmitterUsesAffineEmbeddingGatherBatchedForAffineEmbeddings() throws {
    let ir = qwen35MetalPrefillAffineEmbedding()
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try PrefillEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )

    let dispatches = result.dispatchRecords.filter { $0.opKind == SmeltDispatchRecord.opDispatch }
    var pipelineCounts: [UInt16: Int] = [:]
    for rec in dispatches {
        pipelineCounts[rec.pipeline, default: 0] += 1
    }

    let batchedAffineEmbedding = UInt16(SmeltPipeline.affineEmbeddingGatherBatched.rawValue)
    let singleAffineEmbedding = UInt16(SmeltPipeline.affineEmbeddingGather.rawValue)
    let fp16Embedding = UInt16(SmeltPipeline.embeddingGatherBatched.rawValue)

    #expect(
        (pipelineCounts[batchedAffineEmbedding] ?? 0) == 1,
        "Should emit one dynamic affine embedding gather dispatch"
    )
    #expect(
        (pipelineCounts[singleAffineEmbedding] ?? 0) == 0,
        "Should not unroll affine embedding gather one token at a time"
    )
    #expect(
        (pipelineCounts[fp16Embedding] ?? 0) == 0,
        "Affine-quantized embeddings should not use FP16 batched gather"
    )
}

@Test func prefillEmitterRejectsNonMetalEngine() {
    let ir = SmeltModelIR.qwen35_2B  // engine = "coreml"
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    #expect(throws: PrefillEmitterError.self) {
        try PrefillEmitter.generate(ir: ir, plan: plan, weightLayout: layout)
    }
}

@Test func prefillEmitterHasOutputProjectionPerLayer() throws {
    // This test catches the bug where the DeltaNet output projection
    // was missing from the prefill path. Every layer (delta + attention)
    // must have an output projection in both decode and prefill.
    let ir = qwen35MetalPrefill()
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    // Decode path
    _ = try TopLevelEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )

    // Prefill path
    let prefillResult = try PrefillEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )
    let prefillDispatches = prefillResult.dispatchRecords.filter {
        $0.opKind == SmeltDispatchRecord.opDispatch
    }

    let matmulPipe = UInt16(SmeltPipeline.fusedLutMatmul.rawValue)
    let prefillMatmuls = prefillDispatches.filter { $0.pipeline == matmulPipe }.count

    fputs("  Prefill matmuls: \(prefillMatmuls)\n", stderr)

    // Each DeltaNet layer: 5 batched matmuls (QKV, Z, B, A, out_proj)
    // Each Attention layer: 4 batched matmuls (Q, K, V, O)
    // FFN per layer: 3 batched matmuls (gate, up, down)
    // Total: 18*(5+3) + 6*(4+3) = 144 + 42 = 186
    #expect(prefillMatmuls == 186,
        "Expected 186 batched matmuls, got \(prefillMatmuls)")
}

@Test func prefillEmitterSmallBatchSize() throws {
    // B=4 should work and produce a much smaller table
    let ir = qwen35MetalPrefill(batchSize: 4)
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try PrefillEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )

    let records = result.dispatchRecords
    fputs("  Prefill B=4: \(records.count) records\n", stderr)

    // Should be much smaller than B=64
    #expect(records.count < 600, "B=4 should produce <600 records, got \(records.count)")
    #expect(records.count > 300, "B=4 should produce >300 records, got \(records.count)")
}

// MARK: - external_kv prefill compile smoke

private func parseExternalKVPrefillIR() throws -> SmeltModelIR {
    SmeltModelIR(
        modelName: "test/model",
        config: SmeltConfig(
            hiddenSize: 256, numLayers: 4, vocabSize: 262_144,
            staticSeqCapacity: 256, ropeDim: 256, rmsEps: 1e-6,
            attention: SmeltAttentionConfig(
                qHeads: 4, kvHeads: 1, headDim: 256, gatedQ: false,
                qkNorm: true, externalKV: true),
            ffn: SmeltFFNConfig(dim: 2048, activation: .swiglu)),
        layerPattern: SmeltLayerPattern(unit: [.attention], repeats: 4),
        quantization: SmeltQuantizationConfig(strategy: .lutU4, groupSize: 16, excludePatterns: []),
        loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic),
        prefill: SmeltPrefillConfig(
            engine: "metal", modelPath: "", cachePath: "",
            maxBatchSize: 16, handoffFamilies: ["key_cache", "value_cache", "rope"]))
}

@Test func prefillEmitterCompilesExternalKV() throws {
    // Smoke regression for the codex P1 caught at unit 2b1: a
    // external-KV spec with `external_kv true` on its attention
    // layers must compile cleanly through the prefill emitter
    // without trapping on missing K/V projection or K-norm weights.
    // PrefillEmitter signals "skip K branch" to the fused
    // ropeAndKvCachePrefill kernel via kvHeadsForRopeKV=0; this test
    // confirms the gate fires on externalKV layers.
    let ir = try parseExternalKVPrefillIR()
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try PrefillEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )

    let records = result.dispatchRecords
    let pipelines = result.pipelineNames
    #expect(!records.isEmpty)

    // The fused kernel is the primary K-side pipeline in prefill;
    // when externalKV is set we still dispatch it (Q RoPE rides
    // through with kv_heads=0 internally), so its index must appear
    // in the dispatched records.
    let fusedIndices = Set(pipelines.enumerated().compactMap { index, name in
        name.contains("rope_and_kv_cache_prefill") ? index : nil
    })
    #expect(
        !fusedIndices.isEmpty,
        "fused RoPE/KV-cache pipeline should be in the catalog"
    )
    let dispatched = records.contains { fusedIndices.contains(Int($0.pipeline)) }
    #expect(
        dispatched,
        "a compatible fused-rope kernel should be dispatched for externalKV Q RoPE"
    )

    // The standalone kv_cache_update pipeline lives in the kernel
    // catalog (decode path uses it) but PrefillEmitter never
    // dispatches it directly — and especially not for externalKV.
    if let kvUpdateIdx = pipelines.firstIndex(where: { $0.contains("kv_cache_update") }) {
        let dispatched = records.contains { Int($0.pipeline) == kvUpdateIdx }
        #expect(!dispatched, "external-KV layers must not dispatch kv_cache_update")
    }
}

// MARK: - GPTQ capture points

@Test func gptqCapturePointsCoverExactlyTheInScopeProjections() throws {
    let ir = qwen35MetalPrefillAffineEmbedding()
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try PrefillEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout, traceMode: .full
    )

    // The emitted capture points must be exactly the in-scope affine_u4 projections
    // (attn q/k/v/o + MLP gate/up/down) — no linear_attn/delta, norms, or embeddings,
    // and none missed across the batched / fused / unrolled emit paths.
    let capturedNames = Set(result.gptqCapturePoints.map(\.weightName))
    let expected = Set(SmeltGPTQScope.inResolvedScope(layout).map(\.name))
    #expect(!expected.isEmpty)
    #expect(capturedNames == expected,
            "missing \(expected.subtracting(capturedNames)); extra \(capturedNames.subtracting(expected))")
    // One point per weight (fused K+V / gate+up are captured under each name), with a
    // sane input dim, a real buffer slot, and a boundary inside the dispatch stream.
    #expect(result.gptqCapturePoints.count == capturedNames.count)
    for p in result.gptqCapturePoints {
        #expect(p.k > 0)
        #expect(p.inputSlot >= 0)
        #expect(p.dispatchCount >= 1 && p.dispatchCount <= result.dispatchRecords.count)
    }
}

@Test func gptqCapturePointsAbsentWithoutTraceMarkers() throws {
    // Capture points ride the trace-marker stream, so a stripped build emits none.
    let ir = qwen35MetalPrefillAffineEmbedding()
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let result = try PrefillEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout, traceMode: .stripped
    )
    #expect(result.gptqCapturePoints.isEmpty)
}
