// PrefillEmitter — Generates the prefill dispatch table for Metal-native prefill.
//
// Mirrors TopLevelEmitter but produces batched dispatches for prefill.
// Parallelizable ops use batched kernels (matmul, batched norms).
// Some fallback ops are still unrolled to the max prefill batch, but the
// generated records carry runtime seqLen guards so shorter chunks skip the tail.
//
// The default LM head uses fused_lut_matvec on the last token's hidden state
// (not batched matmul) to avoid allocating a [B, vocabSize] logits buffer.

import Foundation

/// Generates the prefill dispatch table for Metal-native prefill.
public struct PrefillEmitter {
    public enum LMHeadMode: Sendable, Equatable {
        case standard
        case verifyArgmaxOnly
    }

    /// MLX's Metal `sdpa_vector` admission rule expressed as a shape-only
    /// policy. The query-length/GQA product limit is easy to miss: D=256 does
    /// not imply that every length through eight uses the vector kernel.
    /// Returning nil means this topology must use the fallback path even for a
    /// single query row. Model identity is deliberately absent.
    static func mlxVectorAttentionMaxQueryLength(
        headDim: Int,
        gqaRatio: Int
    ) -> Int? {
        guard [64, 96, 128, 256].contains(headDim), gqaRatio > 0 else {
            return nil
        }
        let gqaLimitedLength = 32 / gqaRatio
        guard gqaLimitedLength > 0 else { return nil }
        return min(8, gqaLimitedLength)
    }

    /// Result of prefill code generation: binary dispatch table only.
    public struct GenerateResult {
        /// Binary dispatch records for prefill_dispatches.bin.
        public let dispatchRecords: [SmeltDispatchRecord]
        /// Optional debug boundaries for tracing prefill.
        public let traceMarkers: [SmeltTraceMarker]
        /// Where GPTQ calibration reads each in-scope projection's activation input.
        public let gptqCapturePoints: [SmeltGPTQCapturePoint]
        /// Full pipeline name table corresponding to `dispatchRecords`.
        public let pipelineNames: [String]
        /// Package-local concrete Metal function names appended to the manifest.
        public let namedPipelineNames: [String]
        /// Final named pipeline dispatches with planned route metadata when available.
        let namedPipelineUses: [SmeltNamedPipelineUse]
        /// Rewrites applied while optimizing the prefill dispatch IR.
        public let optimizationStats: SmeltOptimizationStats
    }

    static func generate(
        ir: SmeltModelIR,
        compilationPlan: SmeltCompilationPlan,
        traceMode: SmeltTraceMode = .full,
        lmHeadMode: LMHeadMode = .standard
    ) throws -> GenerateResult {
        try generate(
            ir: ir,
            plan: compilationPlan.bufferPlan,
            compilationPlan: compilationPlan,
            traceMode: traceMode,
            lmHeadMode: lmHeadMode
        )
    }

    static func generate(
        ir: SmeltModelIR,
        plan bufferPlan: SmeltBufferPlan,
        compilationPlan: SmeltCompilationPlan,
        traceMode: SmeltTraceMode = .full,
        lmHeadMode: LMHeadMode = .standard
    ) throws -> GenerateResult {
        try generatePlanned(
            ir: ir,
            plan: bufferPlan,
            plannedWeightLayout: compilationPlan.plannedWeightEntries,
            traceMode: traceMode,
            lmHeadMode: lmHeadMode,
            kernelPlan: compilationPlan.kernelPlan
        )
    }

    /// Generate the prefill dispatch table from the model IR.
    ///
    /// The table is driven by the DSL: layer pattern, dimensions, and
    /// quantization config all come from the IR. No model-specific
    /// hardcoding — works for any architecture the DSL can express.
    public static func generate(
        ir: SmeltModelIR,
        plan: SmeltBufferPlan,
        weightLayout: [SmeltWeightEntry],
        traceMode: SmeltTraceMode = .full,
        lmHeadMode: LMHeadMode = .standard,
        kernelPlan: SmeltKernelPlan? = nil
    ) throws -> GenerateResult {
        let compilationPlan = try SmeltCompiler.planCompilation(
            ir: ir,
            weightLayout: weightLayout,
            kernelPlan: kernelPlan
        )
        return try generate(
            ir: ir,
            plan: plan,
            compilationPlan: compilationPlan,
            traceMode: traceMode,
            lmHeadMode: lmHeadMode
        )
    }

    private static func generatePlanned(
        ir: SmeltModelIR,
        plan: SmeltBufferPlan,
        plannedWeightLayout weightLayout: [SmeltWeightEntry],
        traceMode: SmeltTraceMode,
        lmHeadMode: LMHeadMode,
        kernelPlan: SmeltKernelPlan
    ) throws -> GenerateResult {
        guard let prefill = ir.prefill, prefill.engine == "metal" else {
            throw PrefillEmitterError.notConfigured
        }

        // dense trunk ABI (W2): the prefill counterpart of DenseTrunkEmitter — the
        // unfused M>1 f32/bf16w sequence, embeddings-in / hidden-out. The fp16
        // path below is untouched. (LM-head modes don't apply: the trunk has no
        // head; the dense prefill is always the standard, headless table.)
        if ir.config.portTopology == .embeddingsInHiddenOut {
            return try DenseTrunkPrefillEmitter.generate(
                ir: ir, plan: plan, weightLayout: weightLayout)
        }

        let cfg = ir.config
        let hidden = cfg.hiddenSize
        let B = prefill.maxBatchSize
        let weightsSlot = SmeltFixedSlot.weights.rawValue
        let groupSize = ir.quantization.groupSize
        let fp16 = 2
        let weightEntries = Dictionary(
            uniqueKeysWithValues: weightLayout.map { ($0.name, $0) }
        )

        // Verify-style prefill (emit_all_logits with the
        // standard LM head) prefers the same B≤5-tuned matvec kernels
        // the verify-argmax planner uses: one batched prefill over K+1
        // positions per round.
        //
        // SMELT_PREFILL_VERIFY_USE_AUTO_ROUTING=1 reverts emit_all_logits
        // back to .auto for A/B comparison. Does NOT affect the
        // verifyArgmaxOnly path (which has no .auto kernels).
        let useAuto =
            ProcessInfo.processInfo.environment[
                "SMELT_PREFILL_VERIFY_USE_AUTO_ROUTING"
            ] == "1"
        let useSmallBatch = lmHeadMode == .verifyArgmaxOnly
            || (prefill.emitAllLogits && !useAuto)
        let effectiveKernelPlan = useSmallBatch
            ? .empty
            : kernelPlan
        let fusionPlanner: SmeltFusionPlanner = useSmallBatch
            ? .verifySmallBatch
            : .auto
        var emitter = SmeltCodeEmitter(
            indent: 0,
            traceMarkersEnabled: traceMode.recordsTraceMarkers,
            fusionPlanner: fusionPlanner
        )

        // --- Batched embedding gather ---
        let embedEntry = try requireWeight(
            SmeltCanonicalTensorNames.embedTokens, from: weightEntries
        )
        // Branch on the ENTRY dtype (manifest truth, Trap #3); group size
        // resolves from the entry first (per-tensor group sizes are possible).
        let embedGroupSize = embedEntry.groupSize ?? groupSize
        if embedEntry.dtype == .u4Lut,
           let lutOff = embedEntry.lutOffset
        {
            // No batched LUT gather kernel yet — unroll B individual gathers.
            // Each writes to a different offset in the [B, hidden] output buffer.
            for b in 0..<B {
                let tokenIdOff = UInt64(b * 4)  // Int32 offset into tokenIdsBatch
                let outputOff = UInt64(b * hidden * fp16)
                _ = try emitter.emit(SmeltDispatch(
                    pipeline: .lutEmbeddingGather,
                    buffers: [
                        SmeltBufferBinding(
                            slot: weightsSlot, offset: embedEntry.offset, index: 0
                        ),
                        SmeltBufferBinding(
                            slot: weightsSlot, offset: lutOff, index: 1
                        ),
                        SmeltBufferBinding(
                            slot: plan.tokenIdsBatchSlot, offset: tokenIdOff, index: 2
                        ),
                        SmeltBufferBinding(
                            slot: SmeltFixedSlot.hiddenA.rawValue,
                            offset: outputOff, index: 3
                        ),
                    ],
                    constants: [
                        SmeltConstantBinding(
                            expression: "\(hidden)", type: .uint32, index: 4
                        ),
                        SmeltConstantBinding(
                            expression: "\(embedGroupSize)", type: .uint32, index: 5
                        ),
                    ],
                    dispatch: .threads(
                        width: hidden / 2, height: 1, depth: 1,
                        tgWidth: min(hidden / 2, 256), tgHeight: 1, tgDepth: 1
                    ),
                    comment: "LUT embedding gather [b=\(b)]",
                    minSeqLen: b + 1
                ))
            }
        } else if embedEntry.dtype == .affineU4,
                  let scalesOff = embedEntry.scalesOffset,
                  let biasesOff = embedEntry.biasesOffset
        {
            _ = try emitter.emit(SmeltDispatch(
                pipeline: .affineEmbeddingGatherBatched,
                buffers: [
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: embedEntry.offset, index: 0
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: scalesOff, index: 1
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: biasesOff, index: 2
                    ),
                    SmeltBufferBinding(
                        slot: plan.tokenIdsBatchSlot, index: 3
                    ),
                    SmeltBufferBinding(
                        slot: SmeltFixedSlot.hiddenA.rawValue, index: 4
                    ),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "\(hidden)", type: .uint32, index: 5
                    ),
                    SmeltConstantBinding(
                        expression: "__seqLen__", type: .uint32, index: 6
                    ),
                    SmeltConstantBinding(
                        expression: "\(embedGroupSize)", type: .uint32, index: 7
                    ),
                ],
                dispatch: .threads(
                    width: hidden / 2, height: B, depth: 1,
                    tgWidth: min(hidden / 2, 256), tgHeight: 1, tgDepth: 1
                ),
                comment: "Batched affine embedding gather",
                dynamicGridH: .seqLen
            ))
        } else if embedEntry.dtype == .binary1 || embedEntry.dtype == .ternary2 {
            _ = try emitter.emitSignedEmbeddingGather(
                weightEntry: embedEntry,
                weightsSlot: weightsSlot,
                tokenIdSlot: plan.tokenIdsBatchSlot,
                outputSlot: SmeltFixedSlot.hiddenA.rawValue,
                hiddenSize: hidden,
                batchSize: B,
                dynamicGridH: .seqLen,
                comment: "Batched native signed embedding gather"
            )
        } else if embedEntry.dtype == .turboQuantH,
                  let codebookOff = embedEntry.codebookOffset,
                  let codesPerRow = embedEntry.packedRowStride
        {
            // No batched TQH gather kernel — unroll B single-token
            // gathers (matches the LUT path above).
            for b in 0..<B {
                _ = try emitter.emitTurboQuantHEmbeddingGather(
                    codesSlot: weightsSlot, codesOffset: embedEntry.offset,
                    codebookSlot: weightsSlot, codebookOffset: codebookOff,
                    tokenIdSlot: plan.tokenIdsBatchSlot,
                    tokenIdOffset: UInt64(b * 4),
                    outputSlot: SmeltFixedSlot.hiddenA.rawValue,
                    outputOffset: UInt64(b * hidden * fp16),
                    hiddenSize: hidden, codesPerRow: codesPerRow,
                    minSeqLen: b + 1,
                    comment: "TurboQuant-H embedding gather [b=\(b)]"
                )
            }
        } else {
            _ = try emitter.emit(SmeltDispatch(
                pipeline: .embeddingGatherBatched,
                buffers: [
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: embedEntry.offset, index: 0
                    ),
                    SmeltBufferBinding(
                        slot: plan.tokenIdsBatchSlot, index: 1
                    ),
                    SmeltBufferBinding(
                        slot: SmeltFixedSlot.hiddenA.rawValue, index: 2
                    ),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "\(hidden)", type: .uint32, index: 3
                    ),
                    SmeltConstantBinding(
                        expression: "__seqLen__", type: .uint32, index: 4
                    ),
                ],
                dispatch: .threads(
                    width: hidden, height: B, depth: 1,
                    tgWidth: min(hidden, 256), tgHeight: 1, tgDepth: 1
                ),
                comment: "Batched embedding gather",
                dynamicGridH: .seqLen
            ))
        }
        emitter.recordTraceMarker(
            label: "embed_out",
            bufferSlot: SmeltFixedSlot.hiddenA.rawValue
        )

        if cfg.hiddenSizePerLayerInput > 0 {
            try emitPerLayerInputsSetupBatched(
                config: cfg,
                weightEntries: weightEntries,
                weightsSlot: weightsSlot,
                tokenIdsBatchSlot: plan.tokenIdsBatchSlot,
                groupSize: groupSize,
                quantizeEmbedding: ir.quantization.quantizeEmbedding,
                batchSize: B,
                emitter: &emitter
            )
        }

        // --- Layer loop (driven by DSL layer pattern) ---
        let layers = ir.layerPattern.expanded
        var deltaIdx = 0
        var attnIdx = 0

        let normOutSlot = SmeltFixedSlot.normOutBuf.rawValue

        for (layerIndex, layerType) in layers.enumerated() {
            // Input layernorm (batched)
            let inputNormName = "layers_\(layerIndex)_input_layernorm_weight"
            try emitBatchedRMSNorm(
                emitter: &emitter,
                inputSlot: .variable("cur"),
                weightSlot: weightsSlot,
                weightOffset: try requireWeight(inputNormName, from: weightEntries).offset,
                outputSlot: normOutSlot,
                dim: hidden, eps: cfg.rmsEps, batchSize: B,
                comment: "L\(layerIndex) input norm (batched)"
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).input_norm",
                bufferSlot: SmeltFixedSlot.normOutBuf.rawValue
            )

            // Layer-specific dispatch
            switch layerType {
            case .delta:
                try emitPrefillDeltaLayer(
                    layerIndex: layerIndex,
                    deltaIndex: deltaIdx,
                    config: cfg,
                    plan: plan,
                    weightEntries: weightEntries,
                    weightsSlot: weightsSlot,
                    groupSize: groupSize,
                    batchSize: B,
                    checkpointRecurrentState: lmHeadMode == .verifyArgmaxOnly,
                    emitter: &emitter
                )
                deltaIdx += 1

            case .attention:
                let attn = try requireAttentionConfig(for: layerType, from: cfg)
                let ropeSlots = try resolveRoPESlots(
                    for: attn,
                    layerType: layerType,
                    config: cfg,
                    plan: plan
                )
                try emitPrefillAttentionLayer(
                    layerIndex: layerIndex,
                    attnIndex: attnIdx,
                    layerType: layerType,
                    config: cfg,
                    plan: plan,
                    weightEntries: weightEntries,
                    weightsSlot: weightsSlot,
                    groupSize: groupSize,
                    batchSize: B,
                    emitter: &emitter,
                    attnOverride: attn,
                    ropeCosSlotOverride: ropeSlots.cos,
                    ropeSinSlotOverride: ropeSlots.sin,
                    sharedKVSourceAttnIndex: ir.kvSharedSourceAttentionIndex(forGlobalLayerIndex: layerIndex)
                )
                attnIdx += 1

            case .sliding, .global:
                let attn = try requireAttentionConfig(for: layerType, from: cfg)
                let ropeSlots = try resolveRoPESlots(
                    for: attn,
                    layerType: layerType,
                    config: cfg,
                    plan: plan
                )
                try emitPrefillAttentionLayer(
                    layerIndex: layerIndex,
                    attnIndex: attnIdx,
                    layerType: layerType,
                    config: cfg,
                    plan: plan,
                    weightEntries: weightEntries,
                    weightsSlot: weightsSlot,
                    groupSize: groupSize,
                    batchSize: B,
                    emitter: &emitter,
                    attnOverride: attn,
                    ropeCosSlotOverride: ropeSlots.cos,
                    ropeSinSlotOverride: ropeSlots.sin,
                    sharedKVSourceAttnIndex: ir.kvSharedSourceAttentionIndex(forGlobalLayerIndex: layerIndex)
                )
                attnIdx += 1
            }

            if cfg.blockTopology == .standard {
                // Residual add (batched: B * hidden elements)
                try emitDynamicElementwiseAddVar(
                    emitter: &emitter,
                    inputAVar: "cur",
                    inputBSlot: normOutSlot,
                    outputVar: "alt",
                    elemsPerToken: hidden,
                    comment: "L\(layerIndex) residual (batched)"
                )
                emitter.recordSwap()
                emitter.recordTraceMarker(
                    label: "L\(layerIndex).mid",
                    bufferSlot: SmeltFixedSlot.hiddenB.rawValue
                )

                // Post-attention layernorm (batched)
                let postNormName = "layers_\(layerIndex)_post_attention_layernorm_weight"
                try emitBatchedRMSNorm(
                    emitter: &emitter,
                    inputSlot: .variable("cur"),
                    weightSlot: weightsSlot,
                    weightOffset: try requireWeight(postNormName, from: weightEntries).offset,
                    outputSlot: normOutSlot,
                    dim: hidden, eps: cfg.rmsEps, batchSize: B,
                    comment: "L\(layerIndex) post-attn norm (batched)"
                )
                emitter.recordTraceMarker(
                    label: "L\(layerIndex).post_norm",
                    bufferSlot: SmeltFixedSlot.normOutBuf.rawValue
                )

                try emitBatchedFFN(
                    layerIndex: layerIndex,
                    config: cfg,
                    plan: plan,
                    weightEntries: weightEntries,
                    weightsSlot: weightsSlot,
                    groupSize: groupSize,
                    batchSize: B,
                    ffnDownFp32Slot: plan.ffnDownFp32Slot,
                    kernelPlan: effectiveKernelPlan,
                    emitter: &emitter
                )

                // FFN residual add (batched)
                try emitDynamicElementwiseAddVar(
                    emitter: &emitter,
                    inputAVar: "cur",
                    inputBSlot: SmeltFixedSlot.ffnDownBuf.rawValue,
                    outputVar: "alt",
                    elemsPerToken: hidden,
                    comment: "L\(layerIndex) FFN residual (batched)"
                )
            } else {
                let postAttentionNormName = "layers_\(layerIndex)_post_attention_layernorm_weight"
                let postAttentionWeightOffset = try requireWeight(
                    postAttentionNormName, from: weightEntries
                ).offset
                let postAttentionFused: Bool
                if traceMode.usesStrippedOptimizations {
                    postAttentionFused = try emitBatchedRMSNorm1PWAddVarIfPossible(
                        emitter: &emitter,
                        inputSlot: normOutSlot,
                        weightSlot: weightsSlot,
                        weightOffset: postAttentionWeightOffset,
                        residualSlotVar: "cur",
                        outputSlotVar: "alt",
                        dim: hidden, eps: cfg.rmsEps, batchSize: B,
                        comment: "L\(layerIndex) post-attn norm + residual (batched, fused)"
                    )
                } else {
                    postAttentionFused = false
                }
                if !postAttentionFused {
                    try emitBatchedRMSNorm(
                        emitter: &emitter,
                        inputSlot: .fixed(normOutSlot),
                        weightSlot: weightsSlot,
                        weightOffset: postAttentionWeightOffset,
                        outputSlot: SmeltFixedSlot.ffnDownBuf.rawValue,
                        dim: hidden, eps: cfg.rmsEps, batchSize: B,
                        comment: "L\(layerIndex) post-attn norm (batched)"
                    )
                    emitter.recordTraceMarker(
                        label: "L\(layerIndex).post_attn_norm",
                        bufferSlot: SmeltFixedSlot.ffnDownBuf.rawValue
                    )

                    try emitDynamicElementwiseAddVar(
                        emitter: &emitter,
                        inputAVar: "cur",
                        inputBSlot: SmeltFixedSlot.ffnDownBuf.rawValue,
                        outputVar: "alt",
                        elemsPerToken: hidden,
                        comment: "L\(layerIndex) residual (batched)"
                    )
                }
                emitter.recordSwap()
                emitter.recordTraceMarker(
                    label: "L\(layerIndex).mid",
                    bufferSlot: SmeltFixedSlot.hiddenB.rawValue
                )

                let preFeedforwardNormName =
                    "layers_\(layerIndex)_pre_feedforward_layernorm_weight"
                try emitBatchedRMSNorm(
                    emitter: &emitter,
                    inputSlot: .variable("cur"),
                    weightSlot: weightsSlot,
                    weightOffset: try requireWeight(preFeedforwardNormName, from: weightEntries).offset,
                    outputSlot: normOutSlot,
                    dim: hidden, eps: cfg.rmsEps, batchSize: B,
                    comment: "L\(layerIndex) pre-ffn norm (batched)"
                )
                emitter.recordTraceMarker(
                    label: "L\(layerIndex).pre_ffn_norm",
                    bufferSlot: SmeltFixedSlot.normOutBuf.rawValue
                )

                try emitBatchedFFN(
                    layerIndex: layerIndex,
                    config: cfg,
                    plan: plan,
                    weightEntries: weightEntries,
                    weightsSlot: weightsSlot,
                    groupSize: groupSize,
                    batchSize: B,
                    ffnDownFp32Slot: plan.ffnDownFp32Slot,
                    kernelPlan: effectiveKernelPlan,
                    emitter: &emitter
                )

                let postFeedforwardNormName =
                    "layers_\(layerIndex)_post_feedforward_layernorm_weight"
                let postFeedforwardWeightOffset = try requireWeight(
                    postFeedforwardNormName, from: weightEntries
                ).offset
                let downEntryForPostNorm = try requireWeight(
                    SmeltKernelConsumerNaming.ffnDownWeight(layerIndex: layerIndex),
                    from: weightEntries
                )
                let postFeedforwardInputIsFP32 = plan.ffnDownFp32Slot >= 0
                    && downEntryForPostNorm.dtype == .fp16
                let postFeedforwardFused: Bool
                if !postFeedforwardInputIsFP32, traceMode.usesStrippedOptimizations {
                    postFeedforwardFused = try emitBatchedRMSNorm1PWAddVarIfPossible(
                        emitter: &emitter,
                        inputSlot: SmeltFixedSlot.ffnDownBuf.rawValue,
                        weightSlot: weightsSlot,
                        weightOffset: postFeedforwardWeightOffset,
                        residualSlotVar: "cur",
                        outputSlotVar: "alt",
                        dim: hidden, eps: cfg.rmsEps, batchSize: B,
                        comment: "L\(layerIndex) post-ffn norm + residual (batched, fused)"
                    )
                } else {
                    postFeedforwardFused = false
                }
                if !postFeedforwardFused {
                    if postFeedforwardInputIsFP32 {
                        try emitBatchedRMSNormFromFP32(
                            emitter: &emitter,
                            inputSlot: plan.ffnDownFp32Slot,
                            weightSlot: weightsSlot,
                            weightOffset: postFeedforwardWeightOffset,
                            outputSlot: normOutSlot,
                            dim: hidden, eps: cfg.rmsEps, batchSize: B,
                            comment: "L\(layerIndex) post-ffn norm from FP32 (batched)"
                        )
                    } else {
                        try emitBatchedRMSNorm(
                            emitter: &emitter,
                            inputSlot: .fixed(SmeltFixedSlot.ffnDownBuf.rawValue),
                            weightSlot: weightsSlot,
                            weightOffset: postFeedforwardWeightOffset,
                            outputSlot: normOutSlot,
                            dim: hidden, eps: cfg.rmsEps, batchSize: B,
                            comment: "L\(layerIndex) post-ffn norm (batched)"
                        )
                    }
                    emitter.recordTraceMarker(
                        label: "L\(layerIndex).post_ffn_norm",
                        bufferSlot: SmeltFixedSlot.normOutBuf.rawValue
                    )

                    try emitDynamicElementwiseAddVar(
                        emitter: &emitter,
                        inputAVar: "cur",
                        inputBSlot: normOutSlot,
                        outputVar: "alt",
                        elemsPerToken: hidden,
                        comment: "L\(layerIndex) FFN residual (batched)"
                    )
                }
                emitter.recordTraceMarker(
                    label: "L\(layerIndex).pre_per_layer_branch",
                    bufferSlot: SmeltFixedSlot.hiddenA.rawValue
                )

                if cfg.hiddenSizePerLayerInput > 0 {
                    try emitPerLayerResidualBranchBatched(
                        layerIndex: layerIndex,
                        config: cfg,
                        weightEntries: weightEntries,
                        weightsSlot: weightsSlot,
                        groupSize: groupSize,
                        batchSize: B,
                        emitter: &emitter
                    )
                }
            }

            emitter.recordSwap()
            emitter.recordTraceMarker(
                label: "L\(layerIndex).out",
                bufferSlot: SmeltFixedSlot.hiddenA.rawValue
            )
        }

        // --- Final norm (batched) ---
        let finalNormOffset = try requireWeight(
            "norm_weight", from: weightEntries
        ).offset
        try emitBatchedRMSNorm(
            emitter: &emitter,
            inputSlot: .variable("cur"),
            weightSlot: weightsSlot,
            weightOffset: finalNormOffset,
            outputSlot: normOutSlot,
            dim: hidden, eps: cfg.rmsEps, batchSize: B,
            comment: "Final norm (batched)"
        )
        emitter.recordTraceMarker(
            label: "final_norm",
            bufferSlot: SmeltFixedSlot.normOutBuf.rawValue
        )

        // --- LM Head ---
        // Default: one matvec on the last-token slice. With
        // prefill.emit_all_logits=true: B per-position matvecs
        // into a [B, vocab] logitsBuf, with argmax skipped (use
        // SmeltRuntime.prefillAllLogits to read the rows).
        let lmHeadEntry = cfg.tiedLMHead
            ? embedEntry
            : try requireWeight("lm_head_weight", from: weightEntries)
        let emitAllLogits = prefill.emitAllLogits && lmHeadMode == .standard

        if lmHeadMode == .verifyArgmaxOnly {
            try emitVerifyArgmaxLMHead(
                weightEntry: lmHeadEntry,
                weightsSlot: weightsSlot,
                inputSlot: normOutSlot,
                outputSlot: SmeltFixedSlot.argmaxBuf.rawValue,
                rows: cfg.vocabSize,
                cols: hidden,
                groupSize: groupSize,
                batchSize: B,
                logitCap: cfg.logitCap ?? 0,
                emitter: &emitter,
                comment: "Verify LM head argmax"
            )
        } else if emitAllLogits {
            try emitBatchedMatmul(
                weightEntry: lmHeadEntry,
                weightsSlot: weightsSlot,
                inputSlot: normOutSlot,
                outputSlot: SmeltFixedSlot.logitsBuf.rawValue,
                rows: cfg.vocabSize, cols: hidden, groupSize: groupSize,
                batchSize: B,
                emitter: &emitter,
                comment: "LM head (batched)"
            )
            if let logitCap = cfg.logitCap {
                // Per-position cap, gated by `minSeqLen: pos + 1`
                // so positions ≥ runtime seqLen don't waste compute
                // capping garbage rows. Capping vocab*B in one
                // dispatch would do tens of millions of unused
                // tanh ops per verify call.
                let outStride = UInt64(cfg.vocabSize * fp16)
                for pos in 0 ..< B {
                    let off = UInt64(pos) * outStride
                    _ = try emitter.emit(SmeltDispatch(
                        pipeline: .logitCap,
                        buffers: [
                            SmeltBufferBinding(
                                slot: SmeltFixedSlot.logitsBuf.rawValue,
                                offset: off, index: 0
                            ),
                            SmeltBufferBinding(
                                slot: SmeltFixedSlot.logitsBuf.rawValue,
                                offset: off, index: 1
                            ),
                        ],
                        constants: [
                            SmeltConstantBinding(
                                expression: "\(cfg.vocabSize)",
                                type: .uint32, index: 2
                            ),
                            SmeltConstantBinding(
                                expression: "\(logitCap)",
                                type: .float32, index: 3
                            ),
                        ],
                        dispatch: .threads(
                            width: cfg.vocabSize, height: 1, depth: 1,
                            tgWidth: min(cfg.vocabSize, 1024),
                            tgHeight: 1, tgDepth: 1
                        ),
                        comment: "Logit cap pos \(pos)",
                        minSeqLen: pos + 1
                    ))
                }
            }
        } else {
            try emitLMHeadLastToken(
                weightEntry: lmHeadEntry,
                weightsSlot: weightsSlot,
                inputSlot: normOutSlot,
                outputSlot: SmeltFixedSlot.logitsBuf.rawValue,
                rows: cfg.vocabSize, cols: hidden, groupSize: groupSize,
                hiddenStride: hidden * fp16,
                emitter: &emitter,
                comment: cfg.tiedLMHead
                    ? "LM head (last token, tied)"
                    : "LM head (last token, separate)"
            )

            if let logitCap = cfg.logitCap {
                _ = try emitter.emitLogitCap(
                    inputSlot: SmeltFixedSlot.logitsBuf.rawValue,
                    outputSlot: SmeltFixedSlot.logitsBuf.rawValue,
                    count: cfg.vocabSize,
                    cap: logitCap,
                    comment: "Logit capping"
                )
            }

            _ = try emitter.emitArgmax(
                inputSlot: SmeltFixedSlot.logitsBuf.rawValue,
                outputSlot: SmeltFixedSlot.argmaxBuf.rawValue,
                count: cfg.vocabSize,
                comment: "Argmax"
            )
        }

        // Run the same reportable optimization pipeline used by decode. Today
        // this is intentionally a no-op for prefill rewrites, but it records
        // real fusion opportunities before binary lowering.
        emitter.optimizeAndRebuildRecords()

        return GenerateResult(
            dispatchRecords: emitter.dispatchRecords,
            traceMarkers: emitter.buildTraceMarkers(),
            gptqCapturePoints: emitter.buildCapturePoints(),
            pipelineNames: emitter.buildPipelineNames(),
            namedPipelineNames: emitter.namedPipelines,
            namedPipelineUses: emitter.namedPipelineUses,
            optimizationStats: emitter.optimizationStats
        )
    }

    // MARK: - Batched helpers

    /// Slot reference: either a fixed slot or a variable name (cur/alt).
    private enum SlotRef {
        case fixed(Int)
        case variable(String)
    }

    private static func seqLenExpr(_ scale: Int = 1) -> String {
        scale == 1 ? "__seqLen__" : "__seqLen__*\(scale)"
    }

    private static func seqLenGrid(_ scale: Int = 1) -> SmeltDynamicGridDimension {
        scale == 1 ? .seqLen : .seqLenMul(scale)
    }

    private static func emitDynamicElementwiseAddVar(
        emitter: inout SmeltCodeEmitter,
        inputAVar: String,
        inputBSlot: Int,
        outputVar: String,
        elemsPerToken: Int,
        comment: String? = nil
    ) throws {
        _ = try emitter.emit(SmeltDispatch(
            pipeline: .elementwiseAdd,
            buffers: [
                SmeltBufferBinding(variableSlot: inputAVar, index: 0),
                SmeltBufferBinding(slot: inputBSlot, index: 1),
                SmeltBufferBinding(variableSlot: outputVar, index: 2),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: seqLenExpr(elemsPerToken),
                    type: .uint32,
                    index: 3
                )
            ],
            dispatch: .threads(
                width: elemsPerToken, height: 1, depth: 1,
                tgWidth: min(max(elemsPerToken, 1), 1024),
                tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            dynamicGridW: seqLenGrid(elemsPerToken)
        ))
    }

    private static func emitDynamicElementwiseAdd(
        emitter: inout SmeltCodeEmitter,
        inputASlot: Int,
        inputBSlot: Int,
        outputSlot: Int,
        elemsPerToken: Int,
        comment: String? = nil
    ) throws {
        _ = try emitter.emit(SmeltDispatch(
            pipeline: .elementwiseAdd,
            buffers: [
                SmeltBufferBinding(slot: inputASlot, index: 0),
                SmeltBufferBinding(slot: inputBSlot, index: 1),
                SmeltBufferBinding(slot: outputSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: seqLenExpr(elemsPerToken),
                    type: .uint32,
                    index: 3
                )
            ],
            dispatch: .threads(
                width: elemsPerToken, height: 1, depth: 1,
                tgWidth: min(max(elemsPerToken, 1), 1024),
                tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            dynamicGridW: seqLenGrid(elemsPerToken)
        ))
    }

    private static func emitDynamicScalarMul(
        emitter: inout SmeltCodeEmitter,
        inputSlot: Int,
        outputSlot: Int,
        scalar: Float,
        elemsPerToken: Int,
        comment: String? = nil
    ) throws {
        _ = try emitter.emit(SmeltDispatch(
            pipeline: .scalarMul,
            buffers: [
                SmeltBufferBinding(slot: inputSlot, index: 0),
                SmeltBufferBinding(slot: outputSlot, index: 1),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(scalar)", type: .float32, index: 2),
                SmeltConstantBinding(
                    expression: seqLenExpr(elemsPerToken),
                    type: .uint32,
                    index: 3
                ),
            ],
            dispatch: .threads(
                width: elemsPerToken, height: 1, depth: 1,
                tgWidth: min(max(elemsPerToken, 1), 1024),
                tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            dynamicGridW: seqLenGrid(elemsPerToken)
        ))
    }

    private static func emitDynamicScalarMulWeightVar(
        emitter: inout SmeltCodeEmitter,
        inputVar: String,
        weightSlot: Int,
        weightOffset: UInt64,
        outputVar: String,
        elemsPerToken: Int,
        comment: String? = nil
    ) throws {
        _ = try emitter.emit(SmeltDispatch(
            pipeline: .scalarMulWeight,
            buffers: [
                SmeltBufferBinding(variableSlot: inputVar, index: 0),
                SmeltBufferBinding(slot: weightSlot, offset: weightOffset, index: 1),
                SmeltBufferBinding(variableSlot: outputVar, index: 2),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: seqLenExpr(elemsPerToken),
                    type: .uint32,
                    index: 3
                ),
            ],
            dispatch: .threads(
                width: elemsPerToken, height: 1, depth: 1,
                tgWidth: min(max(elemsPerToken, 1), 1024),
                tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            dynamicGridW: seqLenGrid(elemsPerToken)
        ))
    }

    /// Emit batched RMS norm.
    private static func emitBatchedRMSNorm(
        emitter: inout SmeltCodeEmitter,
        inputSlot: SlotRef,
        weightSlot: Int, weightOffset: UInt64,
        outputSlot: Int,
        dim: Int, eps: Float, batchSize: Int,
        comment: String? = nil
    ) throws {
        let route = emitter.fusionPlanner.prefillRMSNorm(
            dim: dim,
            eps: eps
        )
        let specializedPipeline = route.pipeline
        let specializedThreads = route.threadgroupWidth
        let inputBinding: SmeltBufferBinding
        switch inputSlot {
        case .fixed(let slot):
            inputBinding = SmeltBufferBinding(slot: slot, index: 0)
        case .variable(let name):
            inputBinding = SmeltBufferBinding(
                slot: .variable(name), index: 0
            )
        }

        _ = try emitter.emit(SmeltDispatch(
            pipeline: specializedPipeline ?? .rmsNorm1PWBatched,
            buffers: [
                inputBinding,
                SmeltBufferBinding(
                    slot: weightSlot, offset: weightOffset, index: 1
                ),
                SmeltBufferBinding(slot: outputSlot, index: 2),
            ],
            constants: specializedPipeline == nil ? [
                SmeltConstantBinding(
                    expression: "\(dim)", type: .uint32, index: 3
                ),
                SmeltConstantBinding(
                    expression: "\(eps)", type: .float32, index: 4
                ),
            ] : [],
            dispatch: .threadgroups(
                width: batchSize, height: 1, depth: 1,
                tgWidth: specializedThreads ?? min(dim, 1024),
                tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            dynamicGridW: .seqLen
        ))
    }

    /// Returns true and emits a fused dispatch; on false the caller
    /// must fall back to separate norm + add.
    private static func emitBatchedRMSNorm1PWAddVarIfPossible(
        emitter: inout SmeltCodeEmitter,
        inputSlot: Int,
        weightSlot: Int, weightOffset: UInt64,
        residualSlotVar: String,
        outputSlotVar: String,
        dim: Int, eps: Float, batchSize: Int,
        comment: String? = nil
    ) throws -> Bool {
        let pipeline: SmeltPipeline
        let tgWidth: Int
        switch dim {
        case 2_560:
            pipeline = .rmsNorm1PWD2560AddBatched
            tgWidth = 320
        default:
            return false
        }
        guard eps == 1e-6 else { return false }
        _ = try emitter.emit(SmeltDispatch(
            pipeline: pipeline,
            buffers: [
                SmeltBufferBinding(slot: inputSlot, index: 0),
                SmeltBufferBinding(
                    slot: weightSlot, offset: weightOffset, index: 1
                ),
                SmeltBufferBinding(
                    variableSlot: residualSlotVar, index: 2
                ),
                SmeltBufferBinding(
                    variableSlot: outputSlotVar, index: 3
                ),
            ],
            constants: [],
            dispatch: .threadgroups(
                width: batchSize, height: 1, depth: 1,
                tgWidth: tgWidth, tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            dynamicGridW: .seqLen
        ))
        return true
    }

    private static func emitBatchedRMSNormFromFP32(
        emitter: inout SmeltCodeEmitter,
        inputSlot: Int,
        weightSlot: Int, weightOffset: UInt64,
        outputSlot: Int,
        dim: Int, eps: Float, batchSize: Int,
        comment: String? = nil
    ) throws {
        _ = try emitter.emit(SmeltDispatch(
            pipeline: .rmsNorm1PWFromFP32Batched,
            buffers: [
                SmeltBufferBinding(slot: inputSlot, index: 0),
                SmeltBufferBinding(slot: weightSlot, offset: weightOffset, index: 1),
                SmeltBufferBinding(slot: outputSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "\(dim)", type: .uint32, index: 3
                ),
                SmeltConstantBinding(
                    expression: "\(eps)", type: .float32, index: 4
                ),
            ],
            dispatch: .threadgroups(
                width: batchSize, height: 1, depth: 1,
                tgWidth: min(dim, 1024), tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            dynamicGridW: .seqLen
        ))
    }

    private static func emitBatchedFP16MatvecFP32Out(
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlot: Int, outputSlot: Int,
        rows: Int, cols: Int,
        batchSize: Int,
        emitter: inout SmeltCodeEmitter,
        comment: String? = nil
    ) throws {
        for b in 0..<batchSize {
            _ = try emitter.emit(SmeltDispatch(
                pipeline: .fp16MatvecFP32Out,
                buffers: [
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: weightEntry.offset, index: 0
                    ),
                    SmeltBufferBinding(
                        slot: inputSlot,
                        offset: UInt64(b * cols * 2),
                        index: 1
                    ),
                    SmeltBufferBinding(
                        slot: outputSlot,
                        offset: UInt64(b * rows * 4),
                        index: 2
                    ),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "\(cols)", type: .uint32, index: 3
                    ),
                ],
                dispatch: .threadgroups(
                    width: rows, height: 1, depth: 1,
                    tgWidth: 256, tgHeight: 1, tgDepth: 1
                ),
                comment: comment.map { "\($0) [b=\(b)]" },
                minSeqLen: b + 1
            ))
        }
    }

    /// Emit batched matmul (dispatches fused_lut_matmul or fp16_matvec×B).
    /// The exhaustive `else` for a family-routed matvec helper. matvecFamily already threw
    /// `.missing` for the bf16/fp32 holes (knownMissing), and each family branch's metadata guard
    /// caught malformed quant entries — so reaching here means a registered family this site has
    /// no lowering for. A loud throw, never a silent fp16 fallback.
    private static func throwUnroutableMatvecFamily(
        _ site: String, _ family: MatvecKernelTable.Family, _ weightEntry: SmeltWeightEntry
    ) throws {
        throw SmeltEmitError.unsupported(
            detail: "\(site): no matvec for family \(family) (weight '\(weightEntry.name)')")
    }

    private static func emitBatchedMatmul(
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlot: Int, outputSlot: Int,
        rows: Int, cols: Int, groupSize callerGroupSize: Int,
        batchSize: Int,
        emitter: inout SmeltCodeEmitter,
        comment: String? = nil,
        generatedFullRoute: SmeltPlannedKernelRoute? = nil
    ) throws {
        // Entry-resolved group size (per-tensor group sizes are possible).
        let groupSize = weightEntry.groupSize ?? callerGroupSize
        // Family via the ONE gateway (no inline dtype routing): bf16/fp32 weights and malformed
        // quant metadata are LOUD here instead of the old silent fp16 fallback (the final else).
        // The batched lowering below is unchanged — byte-identical for valid packages.
        let family = try emitter.matvecFamily(weightEntry, shape: .gemm, output: .fp16, slot: .fixed)
        if family == .lutU4, let lutOff = weightEntry.lutOffset {
            _ = try emitter.emit(SmeltDispatch(
                pipeline: .fusedLutMatmul,
                buffers: [
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: weightEntry.offset, index: 0
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: lutOff, index: 1
                    ),
                    SmeltBufferBinding(slot: inputSlot, index: 2),
                    SmeltBufferBinding(slot: outputSlot, index: 3),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "\(cols)", type: .uint32, index: 4
                    ),
                    SmeltConstantBinding(
                        expression: "\(groupSize)", type: .uint32, index: 5
                    ),
                    SmeltConstantBinding(
                        expression: "\(rows)", type: .uint32, index: 6
                    ),
                ],
                dispatch: .threadgroups(
                    width: rows, height: batchSize, depth: 1,
                    tgWidth: 256, tgHeight: 1, tgDepth: 1
                ),
                comment: comment,
                dynamicGridH: .seqLen
            ))
        } else if family == .affineU4,
                  let scalesOff = weightEntry.scalesOffset,
                  let biasesOff = weightEntry.biasesOffset
        {
            let batchedRoute = emitter.fusionPlanner.prefillAffineFull(
                rows: rows,
                cols: cols,
                groupSize: groupSize
            )
            if let batchedRoute {
                let batchedPipeline = batchedRoute.pipeline
                let rowTile = batchedRoute.rowTile
                let batchTile = batchedRoute.batchTile
                let tgWidth = batchedRoute.threadgroupWidth
                _ = try emitter.emit(SmeltDispatch(
                    pipeline: batchedPipeline,
                    buffers: [
                        SmeltBufferBinding(
                            slot: weightsSlot, offset: weightEntry.offset, index: 0
                        ),
                        SmeltBufferBinding(
                            slot: weightsSlot, offset: scalesOff, index: 1
                        ),
                        SmeltBufferBinding(
                            slot: weightsSlot, offset: biasesOff, index: 2
                        ),
                        SmeltBufferBinding(
                            slot: inputSlot, index: 3
                        ),
                        SmeltBufferBinding(slot: outputSlot, index: 4),
                    ],
                    constants: [
                        SmeltConstantBinding(
                            expression: "__seqLen__",
                            type: .uint32,
                            index: 5
                        )
                    ],
                    dispatch: .threadgroups(
                        width: (rows + rowTile - 1) / rowTile,
                        height: (batchSize + batchTile - 1) / batchTile,
                        depth: 1,
                        tgWidth: tgWidth, tgHeight: 1, tgDepth: 1
                    ),
                    comment: comment,
                    dynamicGridH: .seqLenCeilDiv(batchTile)
                ))
            } else if let generatedFullRoute,
                      let geometry = generatedFullRoute.affineMatvecPrefillFullLaunchGeometry(
                        expectedShape: SmeltKernelShape(
                            rows: rows,
                            cols: cols,
                            groupSize: groupSize
                        )
                      ),
                      let batchTile = geometry.batchTile
            {
                _ = try emitter.emit(SmeltDispatch(
                    pipeline: .affineMatvec,
                    plannedKernelRoute: generatedFullRoute,
                    buffers: [
                        SmeltBufferBinding(
                            slot: weightsSlot, offset: weightEntry.offset, index: 0
                        ),
                        SmeltBufferBinding(
                            slot: weightsSlot, offset: scalesOff, index: 1
                        ),
                        SmeltBufferBinding(
                            slot: weightsSlot, offset: biasesOff, index: 2
                        ),
                        SmeltBufferBinding(slot: inputSlot, index: 3),
                        SmeltBufferBinding(slot: outputSlot, index: 4),
                    ],
                    constants: [
                        SmeltConstantBinding(
                            expression: "__seqLen__",
                            type: .uint32,
                            index: 5
                        )
                    ],
                    dispatch: .threadgroups(
                        width: geometry.gridWidth(rows: rows),
                        height: geometry.gridHeight(batchSize: batchSize) ?? 1,
                        depth: 1,
                        tgWidth: geometry.threadgroupWidth, tgHeight: 1, tgDepth: 1
                    ),
                    comment: comment,
                    dynamicGridH: .seqLenCeilDiv(batchTile)
                ))
            } else if let batchedRoute = emitter.fusionPlanner.prefillAffineBatched(
                rows: rows,
                cols: cols,
                groupSize: groupSize
            ) {
                let batchedPipeline = batchedRoute.pipeline
                let rowTile = batchedRoute.rowTile
                let batchTile = batchedRoute.batchTile
                let tgWidth = batchedRoute.threadgroupWidth
                _ = try emitter.emit(SmeltDispatch(
                    pipeline: batchedPipeline,
                    buffers: [
                        SmeltBufferBinding(
                            slot: weightsSlot, offset: weightEntry.offset, index: 0
                        ),
                        SmeltBufferBinding(
                            slot: weightsSlot, offset: scalesOff, index: 1
                        ),
                        SmeltBufferBinding(
                            slot: weightsSlot, offset: biasesOff, index: 2
                        ),
                        SmeltBufferBinding(
                            slot: inputSlot, index: 3
                        ),
                        SmeltBufferBinding(slot: outputSlot, index: 4),
                    ],
                    constants: [
                        SmeltConstantBinding(
                            expression: "__seqLen__",
                            type: .uint32,
                            index: 5
                        )
                    ],
                    dispatch: .threadgroups(
                        width: (rows + rowTile - 1) / rowTile,
                        height: (batchSize + batchTile - 1) / batchTile,
                        depth: 1,
                        tgWidth: tgWidth, tgHeight: 1, tgDepth: 1
                    ),
                    comment: comment,
                    dynamicGridH: .seqLenCeilDiv(batchTile)
                ))
            } else {
                let route = emitter.fusionPlanner.unrolledPrefillAffineMatvec(
                    rows: rows,
                    cols: cols,
                    groupSize: groupSize
                )
                let specializedPipeline = route.pipeline
                for b in 0..<batchSize {
                    _ = try emitter.emit(SmeltDispatch(
                        pipeline: specializedPipeline ?? .affineMatvec,
                        buffers: [
                            SmeltBufferBinding(
                                slot: weightsSlot, offset: weightEntry.offset, index: 0
                            ),
                            SmeltBufferBinding(
                                slot: weightsSlot, offset: scalesOff, index: 1
                            ),
                            SmeltBufferBinding(
                                slot: weightsSlot, offset: biasesOff, index: 2
                            ),
                            SmeltBufferBinding(
                                slot: inputSlot,
                                offset: UInt64(b * cols * 2),
                                index: 3
                            ),
                            SmeltBufferBinding(
                                slot: outputSlot,
                                offset: UInt64(b * rows * 2),
                                index: 4
                            ),
                        ],
                        constants: specializedPipeline == nil ? [
                            SmeltConstantBinding(
                                expression: "\(rows)", type: .uint32, index: 5
                            ),
                        ] : [],
                        dispatch: .threadgroups(
                            width: (rows + route.rowTile - 1) / route.rowTile,
                            height: 1, depth: 1,
                            tgWidth: 64, tgHeight: 1, tgDepth: 1
                        ),
                        comment: comment.map { "\($0) [b=\(b)]" },
                        fcCols: specializedPipeline == nil ? cols : nil,
                        fcGroupSize: specializedPipeline == nil ? groupSize : nil,
                        minSeqLen: b + 1
                    ))
                }
            }
        } else if family == .binary1 || family == .ternary2 {
            _ = try emitter.emitSignedMatvec(
                weightEntry: weightEntry,
                weightsSlot: weightsSlot,
                inputBinding: SmeltBufferBinding(slot: inputSlot, index: 2),
                outputBinding: SmeltBufferBinding(slot: outputSlot, index: 3),
                rows: rows,
                cols: cols,
                batchSize: batchSize,
                dynamicGridH: .seqLen,
                comment: comment
            )
        } else if family == .tqh,
                  let codebookOff = weightEntry.codebookOffset,
                  let codesPerRow = weightEntry.packedRowStride {
            _ = try emitter.emitTQHMatvec(
                codesSlot: weightsSlot, codesOffset: weightEntry.offset,
                codebookSlot: weightsSlot, codebookOffset: codebookOff,
                inputSlot: inputSlot,
                xHatScratchSlot: SmeltFixedSlot.tqhMatvecXHatBuf.rawValue,
                outputSlot: outputSlot,
                rows: rows, cols: cols, codesPerRow: codesPerRow,
                batchSize: batchSize,
                dynamicGridH: .seqLen,
                comment: comment
            )
        } else if case let .dense(dt) = family {
            // Dense fp16-act prefill: dispatch B individual gemv matvecs (no batched dense matmul
            // kernel — rare path, only LM head + bf16/fp32-direct projections). .fp16 → fp16_matvec,
            // .bf16/.fp32 → the U2 fp16_matvec_{bf16,fp32}w kernels (gateway authorizes only these).
            let pipeline = try SmeltCodeEmitter.fp16DenseMatvecPipeline(dt)
            for b in 0..<batchSize {
                _ = try emitter.emit(SmeltDispatch(
                    pipeline: pipeline,
                    buffers: [
                        SmeltBufferBinding(
                            slot: weightsSlot, offset: weightEntry.offset, index: 0
                        ),
                        SmeltBufferBinding(
                            slot: inputSlot,
                            offset: UInt64(b * cols * 2),
                            index: 1
                        ),
                        SmeltBufferBinding(
                            slot: outputSlot,
                            offset: UInt64(b * rows * 2),
                            index: 2
                        ),
                    ],
                    constants: [
                        SmeltConstantBinding(
                            expression: "\(cols)", type: .uint32, index: 3
                        ),
                    ],
                    dispatch: .threadgroups(
                        width: rows, height: 1, depth: 1,
                        tgWidth: 256, tgHeight: 1, tgDepth: 1
                    ),
                    comment: comment.map { "\($0) [b=\(b)]" },
                    minSeqLen: b + 1
                ))
            }
        } else {
            try throwUnroutableMatvecFamily("emitBatchedMatmul", family, weightEntry)
        }
        // GPTQ capture point: after this projection's dispatch(es), inputSlot holds
        // the [seqLen, cols] activation. Records only in-scope affine_u4 projections.
        recordGPTQCapturePoint(weightEntry, inputSlot: inputSlot, k: cols, emitter: &emitter)
    }

    /// Record a GPTQ capture point when `weightEntry` is an in-scope affine_u4
    /// projection (attn q/k/v/o, MLP gate/up/down). The calibrator reads `inputSlot`
    /// as `[seqLen, k]` to accumulate this weight's activation Hessian.
    ///
    /// Only dispatched projections get a capture point. A cross-layer shared-KV
    /// layer (`sharedKVSourceAttnIndex != nil`) skips its own K/V dispatch, so its
    /// K/V weights — still affine_u4 and in `SmeltGPTQScope` — get none; calibration
    /// then can't produce their blocks and the build's coverage check fails loudly.
    /// No in-scope model uses shared KV today; handling it (scope-exclude the
    /// undispatched weights) belongs with the calibration driver.
    private static func recordGPTQCapturePoint(
        _ weightEntry: SmeltWeightEntry, inputSlot: Int, k: Int,
        emitter: inout SmeltCodeEmitter
    ) {
        guard SmeltGPTQScope.isResolvedInScope(weightEntry) else { return }
        emitter.recordCapturePoint(weightName: weightEntry.name, inputSlot: inputSlot, k: k)
    }

    private static func emitBatchedAffineMatmul(
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputBinding: SmeltBufferBinding,
        outputSlot: Int,
        rows: Int,
        cols: Int,
        route: SmeltPrefillTiledRoute,
        batchSize: Int,
        emitter: inout SmeltCodeEmitter,
        comment: String? = nil
    ) throws {
        guard weightEntry.dtype == .affineU4,
              let scalesOff = weightEntry.scalesOffset,
              let biasesOff = weightEntry.biasesOffset
        else {
            throw SmeltEmitError.missingConfig(
                detail: "Batched affine matmul requires affine_u4 weights"
            )
        }

        _ = try emitter.emit(SmeltDispatch(
            pipeline: route.pipeline,
            buffers: [
                SmeltBufferBinding(
                    slot: weightsSlot, offset: weightEntry.offset, index: 0
                ),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: scalesOff, index: 1
                ),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: biasesOff, index: 2
                ),
                inputBinding,
                SmeltBufferBinding(slot: outputSlot, index: 4),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "__seqLen__", type: .uint32, index: 5
                )
            ],
            dispatch: .threadgroups(
                width: (rows + route.rowTile - 1) / route.rowTile,
                height: (batchSize + route.batchTile - 1) / route.batchTile,
                depth: 1,
                tgWidth: route.threadgroupWidth, tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            dynamicGridH: .seqLenCeilDiv(route.batchTile)
        ))
    }

    private static func emitUnrolledFusedDualAffineMatvec(
        weightEntry1: SmeltWeightEntry,
        weightEntry2: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlot: Int,
        outputSlot1: Int,
        outputSlot2: Int,
        rows: Int,
        cols: Int,
        groupSize callerGroupSize: Int,
        batchSize: Int,
        generatedFullRoute: SmeltPlannedKernelRoute? = nil,
        emitter: inout SmeltCodeEmitter,
        comment: String? = nil
    ) throws {
        guard weightEntry1.dtype == .affineU4,
              let scales1 = weightEntry1.scalesOffset,
              let biases1 = weightEntry1.biasesOffset,
              weightEntry2.dtype == .affineU4,
              let scales2 = weightEntry2.scalesOffset,
              let biases2 = weightEntry2.biasesOffset
        else {
            throw PrefillEmitterError.unsupported(
                "fused dual affine prefill requires affine_u4 weights"
            )
        }
        // Entry-resolved group size (per-tensor group sizes are possible); the
        // fused kernel bakes ONE group size, so the pair must agree.
        let groupSize = weightEntry1.groupSize ?? callerGroupSize
        guard (weightEntry2.groupSize ?? callerGroupSize) == groupSize else {
            throw PrefillEmitterError.unsupported(
                "fused dual affine prefill requires matching group sizes "
                    + "(\(weightEntry1.name): \(groupSize), "
                    + "\(weightEntry2.name): \(weightEntry2.groupSize ?? callerGroupSize))"
            )
        }

        let batchedSpecializedRoute = emitter.fusionPlanner.prefillDualAffineMatvec(
            rows: rows,
            cols: cols,
            groupSize: groupSize
        )
        let batchedSpecializedPipeline = batchedSpecializedRoute.pipeline

        if let batchedSpecializedPipeline {
            _ = try emitter.emit(SmeltDispatch(
                pipeline: batchedSpecializedPipeline,
                buffers: [
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: weightEntry1.offset, index: 0
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: scales1, index: 1
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: biases1, index: 2
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: weightEntry2.offset, index: 3
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: scales2, index: 4
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: biases2, index: 5
                    ),
                    SmeltBufferBinding(slot: inputSlot, index: 6),
                    SmeltBufferBinding(slot: outputSlot1, index: 7),
                    SmeltBufferBinding(slot: outputSlot2, index: 8),
                ],
                constants: [],
                dispatch: .threadgroups(
                    width: (rows + 7) / 8, height: batchSize, depth: 1,
                    tgWidth: 64, tgHeight: 1, tgDepth: 1
                ),
                comment: comment,
                dynamicGridH: .seqLen
            ))
            // Fused K+V share one input slot — capture each weight at this boundary.
            recordGPTQCapturePoint(weightEntry1, inputSlot: inputSlot, k: cols, emitter: &emitter)
            recordGPTQCapturePoint(weightEntry2, inputSlot: inputSlot, k: cols, emitter: &emitter)
            return
        }

        for b in 0..<batchSize {
            _ = try emitter.emit(SmeltDispatch(
                pipeline: .fusedDualAffineMatvec,
                buffers: [
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: weightEntry1.offset, index: 0
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: scales1, index: 1
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: biases1, index: 2
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: weightEntry2.offset, index: 3
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: scales2, index: 4
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: biases2, index: 5
                    ),
                    SmeltBufferBinding(
                        slot: inputSlot, offset: UInt64(b * cols * 2), index: 6
                    ),
                    SmeltBufferBinding(
                        slot: outputSlot1, offset: UInt64(b * rows * 2), index: 7
                    ),
                    SmeltBufferBinding(
                        slot: outputSlot2, offset: UInt64(b * rows * 2), index: 8
                    ),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "\(rows)", type: .uint32, index: 9
                    ),
                ],
                dispatch: .threadgroups(
                    width: (rows + 7) / 8, height: 1, depth: 1,
                    tgWidth: 64, tgHeight: 1, tgDepth: 1
                ),
                comment: comment.map { "\($0) [b=\(b)]" },
                fcCols: cols,
                fcGroupSize: groupSize,
                minSeqLen: b + 1
            ))
        }
        // Fused K+V share one input slot — capture each weight at the same boundary.
        recordGPTQCapturePoint(weightEntry1, inputSlot: inputSlot, k: cols, emitter: &emitter)
        recordGPTQCapturePoint(weightEntry2, inputSlot: inputSlot, k: cols, emitter: &emitter)
    }

    private static func emitUnrolledFusedAffineGateUpSwiglu(
        gateEntry: SmeltWeightEntry,
        upEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlot: Int,
        outputSlot: Int,
        rows: Int,
        cols: Int,
        groupSize callerGroupSize: Int,
        batchSize: Int,
        generatedFullRoute: SmeltPlannedKernelRoute? = nil,
        emitter: inout SmeltCodeEmitter,
        comment: String? = nil
    ) throws {
        // Entry-resolved group size; callers verify gate/up agreement, this
        // re-derives defensively (per-tensor group sizes are possible).
        let groupSize = gateEntry.groupSize ?? callerGroupSize
        guard gateEntry.dtype == .affineU4,
              let gateScales = gateEntry.scalesOffset,
              let gateBiases = gateEntry.biasesOffset,
              upEntry.dtype == .affineU4,
              let upScales = upEntry.scalesOffset,
              let upBiases = upEntry.biasesOffset
        else {
            throw PrefillEmitterError.unsupported(
                "fused affine gate/up prefill requires affine_u4 weights"
            )
        }

        if let generatedFullRoute,
           let geometry = generatedFullRoute.fusedGateUpSwigluPrefillFullLaunchGeometry(
                expectedShape: SmeltKernelShape(
                    rows: rows,
                    cols: cols,
                    groupSize: groupSize
                )
           ),
           let batchTile = geometry.batchTile
        {
            _ = try emitter.emit(SmeltDispatch(
                pipeline: .fusedAffineGateUpSwiglu,
                plannedKernelRoute: generatedFullRoute,
                buffers: [
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: gateEntry.offset, index: 0
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: gateScales, index: 1
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: gateBiases, index: 2
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: upEntry.offset, index: 3
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: upScales, index: 4
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: upBiases, index: 5
                    ),
                    SmeltBufferBinding(slot: inputSlot, index: 6),
                    SmeltBufferBinding(slot: outputSlot, index: 7),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "__seqLen__", type: .uint32, index: 8
                    )
                ],
                dispatch: .threadgroups(
                    width: geometry.gridWidth(rows: rows),
                    height: geometry.gridHeight(batchSize: batchSize) ?? 1,
                    depth: 1,
                    tgWidth: geometry.threadgroupWidth, tgHeight: 1, tgDepth: 1
                ),
                comment: comment,
                dynamicGridH: .seqLenCeilDiv(batchTile)
            ))
            recordGPTQCapturePoint(gateEntry, inputSlot: inputSlot, k: cols, emitter: &emitter)
            recordGPTQCapturePoint(upEntry, inputSlot: inputSlot, k: cols, emitter: &emitter)
            return
        }

        let specializedPipeline: SmeltPipeline? = nil

        for b in 0..<batchSize {
            _ = try emitter.emit(SmeltDispatch(
                pipeline: specializedPipeline ?? .fusedAffineGateUpSwiglu,
                buffers: [
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: gateEntry.offset, index: 0
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: gateScales, index: 1
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: gateBiases, index: 2
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: upEntry.offset, index: 3
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: upScales, index: 4
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: upBiases, index: 5
                    ),
                    SmeltBufferBinding(
                        slot: inputSlot, offset: UInt64(b * cols * 2), index: 6
                    ),
                    SmeltBufferBinding(
                        slot: outputSlot, offset: UInt64(b * rows * 2), index: 7
                    ),
                ],
                constants: specializedPipeline == nil ? [
                    SmeltConstantBinding(
                        expression: "\(rows)", type: .uint32, index: 8
                    ),
                ] : [],
                dispatch: .threadgroups(
                    width: (rows + 7) / 8, height: 1, depth: 1,
                    tgWidth: 64, tgHeight: 1, tgDepth: 1
                ),
                comment: comment.map { "\($0) [b=\(b)]" },
                fcCols: specializedPipeline == nil ? cols : nil,
                fcGroupSize: specializedPipeline == nil ? groupSize : nil,
                minSeqLen: b + 1
            ))
        }
        // Fused gate+up share one input slot — capture each weight at the same boundary.
        recordGPTQCapturePoint(gateEntry, inputSlot: inputSlot, k: cols, emitter: &emitter)
        recordGPTQCapturePoint(upEntry, inputSlot: inputSlot, k: cols, emitter: &emitter)
    }

    /// Emit LM head on the last token only (fused_lut_matvec with seqLen-1 offset).
    private static func emitLMHeadLastToken(
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlot: Int, outputSlot: Int,
        rows: Int, cols: Int, groupSize callerGroupSize: Int,
        hiddenStride: Int,
        emitter: inout SmeltCodeEmitter,
        comment: String? = nil
    ) throws {
        // Entry-resolved group size (per-tensor group sizes are possible; a tied
        // Q4_K embedding may be g32 while the spec global is 64).
        let groupSize = weightEntry.groupSize ?? callerGroupSize
        // The input buffer offset is (seqLen-1)*hiddenStride, which is
        // runtime-dynamic. Use offsetKind=3 in the buffer record.
        if weightEntry.dtype == .u4Lut, let lutOff = weightEntry.lutOffset {
            _ = try emitter.emit(SmeltDispatch(
                pipeline: .fusedLutMatvec,
                buffers: [
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: weightEntry.offset, index: 0
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: lutOff, index: 1
                    ),
                    SmeltBufferBinding(
                        slot: inputSlot,
                        offsetExpression: "__seqLenMinus1__ * \(hiddenStride)",
                        index: 2
                    ),
                    SmeltBufferBinding(slot: outputSlot, index: 3),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "\(rows)", type: .uint32, index: 4
                    ),
                ],
                dispatch: .threadgroups(
                    width: (rows + 7) / 8, height: 1, depth: 1,
                    tgWidth: 64, tgHeight: 1, tgDepth: 1
                ),
                comment: comment,
                fcCols: cols,
                fcGroupSize: groupSize
            ))
        } else if weightEntry.dtype == .affineU4,
                  let scalesOff = weightEntry.scalesOffset,
                  let biasesOff = weightEntry.biasesOffset
        {
            let route = emitter.fusionPlanner.unrolledPrefillAffineMatvec(
                rows: rows,
                cols: cols,
                groupSize: groupSize
            )
            let specializedPipeline = route.pipeline
            _ = try emitter.emit(SmeltDispatch(
                pipeline: specializedPipeline ?? .affineMatvec,
                buffers: [
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: weightEntry.offset, index: 0
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: scalesOff, index: 1
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: biasesOff, index: 2
                    ),
                    SmeltBufferBinding(
                        slot: inputSlot,
                        offsetExpression: "__seqLenMinus1__ * \(hiddenStride)",
                        index: 3
                    ),
                    SmeltBufferBinding(slot: outputSlot, index: 4),
                ],
                constants: specializedPipeline == nil ? [
                    SmeltConstantBinding(
                        expression: "\(rows)", type: .uint32, index: 5
                    ),
                ] : [],
                dispatch: .threadgroups(
                    width: (rows + route.rowTile - 1) / route.rowTile,
                    height: 1, depth: 1,
                    tgWidth: 64, tgHeight: 1, tgDepth: 1
                ),
                comment: comment,
                fcCols: specializedPipeline == nil ? cols : nil,
                fcGroupSize: specializedPipeline == nil ? groupSize : nil
            ))
        } else if weightEntry.dtype == .binary1 || weightEntry.dtype == .ternary2 {
            _ = try emitter.emitSignedMatvec(
                weightEntry: weightEntry,
                weightsSlot: weightsSlot,
                inputBinding: SmeltBufferBinding(
                    slot: inputSlot,
                    offsetExpression: "__seqLenMinus1__ * \(hiddenStride)",
                    index: 2
                ),
                outputBinding: SmeltBufferBinding(slot: outputSlot, index: 3),
                rows: rows,
                cols: cols,
                comment: comment
            )
        } else if weightEntry.dtype == .turboQuantH,
                  let codebookOff = weightEntry.codebookOffset,
                  let codesPerRow = weightEntry.packedRowStride {
            _ = try emitter.emitTQHMatvec(
                codesSlot: weightsSlot, codesOffset: weightEntry.offset,
                codebookSlot: weightsSlot, codebookOffset: codebookOff,
                inputSlot: inputSlot,
                inputOffsetExpression: "__seqLenMinus1__ * \(hiddenStride)",
                xHatScratchSlot: SmeltFixedSlot.tqhMatvecXHatBuf.rawValue,
                outputSlot: outputSlot,
                rows: rows, cols: cols, codesPerRow: codesPerRow,
                comment: comment
            )
        } else {
            // Dense LM head: .fp16 → fp16_matvec, .bf16/.fp32 → the U2 fp16_matvec_{bf16,fp32}w
            // kernels (an untied bf16 lm_head_weight would otherwise silently mis-bind as fp16);
            // int32/raw throw loudly. fp16DenseMatvecPipeline is the same selector the gateway uses.
            _ = try emitter.emit(SmeltDispatch(
                pipeline: try SmeltCodeEmitter.fp16DenseMatvecPipeline(weightEntry.dtype),
                buffers: [
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: weightEntry.offset, index: 0
                    ),
                    SmeltBufferBinding(
                        slot: inputSlot,
                        offsetExpression: "__seqLenMinus1__ * \(hiddenStride)",
                        index: 1
                    ),
                    SmeltBufferBinding(slot: outputSlot, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "\(cols)", type: .uint32, index: 3
                    ),
                ],
                dispatch: .threadgroups(
                    width: rows, height: 1, depth: 1,
                    tgWidth: 256, tgHeight: 1, tgDepth: 1
                ),
                comment: comment
            ))
        }
    }

    /// LM head at a fixed batch position. Mirrors `emitLMHeadLastToken`
    /// but with compile-time-constant input/output offsets — used by
    /// the per-position emit-all-logits path to write each batch
    /// row's logits into its own slice of the widened logitsBuf.
    /// `pos` is the batch index (0-based); `minSeqLen = pos + 1`
    /// gates the dispatch so it runs only when seqLen > pos —
    /// otherwise positions ≥ runtime seqLen would read normOutBuf
    /// past the dynamically-shrunk capacity (UB on the GPU).
    private static func emitLMHeadAtPosition(
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlot: Int, inputOffset: UInt64,
        outputSlot: Int, outputOffset: UInt64,
        rows: Int, cols: Int, groupSize callerGroupSize: Int,
        pos: Int,
        emitter: inout SmeltCodeEmitter,
        comment: String? = nil
    ) throws {
        let groupSize = weightEntry.groupSize ?? callerGroupSize
        if weightEntry.dtype == .u4Lut, let lutOff = weightEntry.lutOffset {
            _ = try emitter.emit(SmeltDispatch(
                pipeline: .fusedLutMatvec,
                buffers: [
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: weightEntry.offset, index: 0
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: lutOff, index: 1
                    ),
                    SmeltBufferBinding(
                        slot: inputSlot, offset: inputOffset, index: 2
                    ),
                    SmeltBufferBinding(
                        slot: outputSlot, offset: outputOffset, index: 3
                    ),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "\(rows)", type: .uint32, index: 4
                    ),
                ],
                dispatch: .threadgroups(
                    width: (rows + 7) / 8, height: 1, depth: 1,
                    tgWidth: 64, tgHeight: 1, tgDepth: 1
                ),
                comment: comment,
                fcCols: cols,
                fcGroupSize: groupSize,
                minSeqLen: pos + 1
            ))
        } else if weightEntry.dtype == .affineU4,
                  let scalesOff = weightEntry.scalesOffset,
                  let biasesOff = weightEntry.biasesOffset
        {
            let route = emitter.fusionPlanner.unrolledPrefillAffineMatvec(
                rows: rows, cols: cols, groupSize: groupSize
            )
            let specializedPipeline = route.pipeline
            _ = try emitter.emit(SmeltDispatch(
                pipeline: specializedPipeline ?? .affineMatvec,
                buffers: [
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: weightEntry.offset, index: 0
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: scalesOff, index: 1
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: biasesOff, index: 2
                    ),
                    SmeltBufferBinding(
                        slot: inputSlot, offset: inputOffset, index: 3
                    ),
                    SmeltBufferBinding(
                        slot: outputSlot, offset: outputOffset, index: 4
                    ),
                ],
                constants: specializedPipeline == nil ? [
                    SmeltConstantBinding(
                        expression: "\(rows)", type: .uint32, index: 5
                    ),
                ] : [],
                dispatch: .threadgroups(
                    width: (rows + route.rowTile - 1) / route.rowTile,
                    height: 1, depth: 1,
                    tgWidth: 64, tgHeight: 1, tgDepth: 1
                ),
                comment: comment,
                fcCols: specializedPipeline == nil ? cols : nil,
                fcGroupSize: specializedPipeline == nil ? groupSize : nil,
                minSeqLen: pos + 1
            ))
        } else if weightEntry.dtype == .binary1 || weightEntry.dtype == .ternary2 {
            _ = try emitter.emitSignedMatvec(
                weightEntry: weightEntry,
                weightsSlot: weightsSlot,
                inputBinding: SmeltBufferBinding(
                    slot: inputSlot, offset: inputOffset, index: 2),
                outputBinding: SmeltBufferBinding(
                    slot: outputSlot, offset: outputOffset, index: 3),
                rows: rows,
                cols: cols,
                minSeqLen: pos + 1,
                comment: comment
            )
        } else if weightEntry.dtype == .turboQuantH,
                  let codebookOff = weightEntry.codebookOffset,
                  let codesPerRow = weightEntry.packedRowStride {
            _ = try emitter.emitTQHMatvec(
                codesSlot: weightsSlot, codesOffset: weightEntry.offset,
                codebookSlot: weightsSlot, codebookOffset: codebookOff,
                inputSlot: inputSlot, inputOffset: inputOffset,
                xHatScratchSlot: SmeltFixedSlot.tqhMatvecXHatBuf.rawValue,
                outputSlot: outputSlot, outputOffset: outputOffset,
                rows: rows, cols: cols, codesPerRow: codesPerRow,
                minSeqLen: pos + 1,
                comment: comment
            )
        } else {
            // Dense LM head (see emitLMHeadLastToken): .fp16 → fp16_matvec, .bf16/.fp32 → the U2
            // fp16_matvec_{bf16,fp32}w kernels; int32/raw throw loudly. Same selector as the gateway.
            _ = try emitter.emit(SmeltDispatch(
                pipeline: try SmeltCodeEmitter.fp16DenseMatvecPipeline(weightEntry.dtype),
                buffers: [
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: weightEntry.offset, index: 0
                    ),
                    SmeltBufferBinding(
                        slot: inputSlot, offset: inputOffset, index: 1
                    ),
                    SmeltBufferBinding(
                        slot: outputSlot, offset: outputOffset, index: 2
                    ),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "\(cols)", type: .uint32, index: 3
                    ),
                ],
                dispatch: .threadgroups(
                    width: rows, height: 1, depth: 1,
                    tgWidth: 256, tgHeight: 1, tgDepth: 1
                ),
                comment: comment,
                minSeqLen: pos + 1
            ))
        }
    }

    private static func emitVerifyArgmaxLMHead(
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlot: Int,
        outputSlot: Int,
        rows: Int,
        cols: Int,
        groupSize callerGroupSize: Int,
        batchSize: Int,
        logitCap: Float,
        emitter: inout SmeltCodeEmitter,
        comment: String? = nil
    ) throws {
        let groupSize = weightEntry.groupSize ?? callerGroupSize
        if weightEntry.dtype == .binary1 || weightEntry.dtype == .ternary2 {
            let logitsSlot = SmeltFixedSlot.logitsBuf.rawValue
            _ = try emitter.emitSignedMatvec(
                weightEntry: weightEntry,
                weightsSlot: weightsSlot,
                inputBinding: SmeltBufferBinding(slot: inputSlot, index: 2),
                outputBinding: SmeltBufferBinding(slot: logitsSlot, index: 3),
                rows: rows,
                cols: cols,
                batchSize: batchSize,
                dynamicGridH: .seqLen,
                comment: comment
            )

            let logitsRowBytes = UInt64(rows * MemoryLayout<Float16>.stride)
            for position in 0..<batchSize {
                let logitsOffset = UInt64(position) * logitsRowBytes
                if logitCap != 0 {
                    _ = try emitter.emit(SmeltDispatch(
                        pipeline: .logitCap,
                        buffers: [
                            SmeltBufferBinding(
                                slot: logitsSlot,
                                offset: logitsOffset,
                                index: 0
                            ),
                            SmeltBufferBinding(
                                slot: logitsSlot,
                                offset: logitsOffset,
                                index: 1
                            ),
                        ],
                        constants: [
                            SmeltConstantBinding(
                                expression: "\(rows)",
                                type: .uint32,
                                index: 2
                            ),
                            SmeltConstantBinding(
                                expression: "\(logitCap)",
                                type: .float32,
                                index: 3
                            ),
                        ],
                        dispatch: .threads(
                            width: rows,
                            height: 1,
                            depth: 1,
                            tgWidth: min(rows, 1_024),
                            tgHeight: 1,
                            tgDepth: 1
                        ),
                        comment: "Cap signed verify LM head [b=\(position)]",
                        minSeqLen: position + 1
                    ))
                }
                _ = try emitter.emitArgmax(
                    inputSlot: logitsSlot,
                    outputSlot: outputSlot,
                    count: rows,
                    inputOffset: logitsOffset,
                    outputOffset: UInt64(
                        position * MemoryLayout<Int32>.stride
                    ),
                    minSeqLen: position + 1,
                    comment: "Signed verify LM head argmax [b=\(position)]"
                )
            }
            return
        }
        guard weightEntry.dtype == .affineU4,
              let scalesOff = weightEntry.scalesOffset,
              let biasesOff = weightEntry.biasesOffset
        else {
            throw PrefillEmitterError.unsupported(
                "verify argmax LM head requires affine_u4 weights"
            )
        }
        guard let pipeline = SmeltKernelShapeRegistry.prefillVerifyArgmaxPipeline(
            rows: rows,
            cols: cols,
            groupSize: groupSize
        ) else {
            throw PrefillEmitterError.unsupported(
                "verify argmax LM head has no fused kernel for "
                + "rows=\(rows), cols=\(cols), groupSize=\(groupSize)"
            )
        }
        guard let reducePipeline = SmeltKernelShapeRegistry
            .prefillVerifyArgmaxReducePipeline(rows: rows)
        else {
            throw PrefillEmitterError.unsupported(
                "verify argmax LM head has no reducer for rows=\(rows)"
            )
        }

        let rowTile = 8
        let batchTile = 4
        let partialSlot = SmeltFixedSlot.logitsBuf.rawValue
        _ = try emitter.emit(SmeltDispatch(
            pipeline: pipeline,
            buffers: [
                SmeltBufferBinding(
                    slot: weightsSlot, offset: weightEntry.offset, index: 0
                ),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: scalesOff, index: 1
                ),
                SmeltBufferBinding(
                    slot: weightsSlot, offset: biasesOff, index: 2
                ),
                SmeltBufferBinding(slot: inputSlot, index: 3),
                SmeltBufferBinding(slot: partialSlot, index: 4),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "__seqLen__",
                    type: .uint32,
                    index: 5
                ),
                SmeltConstantBinding(
                    expression: "\(logitCap)",
                    type: .float32,
                    index: 6
                ),
            ],
            dispatch: .threadgroups(
                width: (rows + rowTile - 1) / rowTile,
                height: (batchSize + batchTile - 1) / batchTile,
                depth: 1,
                tgWidth: 64, tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            dynamicGridH: .seqLenCeilDiv(batchTile)
        ))

        _ = try emitter.emit(SmeltDispatch(
            pipeline: reducePipeline,
            buffers: [
                SmeltBufferBinding(slot: partialSlot, index: 0),
                SmeltBufferBinding(slot: outputSlot, index: 1),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "__seqLen__",
                    type: .uint32,
                    index: 2
                )
            ],
            dispatch: .threadgroups(
                width: batchSize, height: 1, depth: 1,
                tgWidth: 256, tgHeight: 1, tgDepth: 1
            ),
            comment: "Reduce verify LM head argmax",
            dynamicGridW: .seqLen
        ))
    }

    private static func emitBatchedPerHeadNorm(
        emitter: inout SmeltCodeEmitter,
        inputSlot: Int,
        weightSlot: Int,
        weightOffset: UInt64,
        outputSlot: Int,
        numHeads: Int,
        headDim: Int,
        eps: Float,
        batchSize: Int,
        comment: String? = nil
    ) throws {
        _ = try emitter.emit(SmeltDispatch(
            pipeline: .perHeadRmsNormBatched,
            buffers: [
                SmeltBufferBinding(slot: inputSlot, index: 0),
                SmeltBufferBinding(slot: weightSlot, offset: weightOffset, index: 1),
                SmeltBufferBinding(slot: outputSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(numHeads)", type: .uint32, index: 3),
                SmeltConstantBinding(expression: "\(headDim)", type: .uint32, index: 4),
                SmeltConstantBinding(expression: "\(eps)", type: .float32, index: 5),
            ],
            dispatch: .threadgroups(
                width: numHeads, height: batchSize, depth: 1,
                tgWidth: AttentionPlugin.directWeightNormThreadgroupWidth(headDim: headDim),
                tgHeight: 1,
                tgDepth: 1
            ),
            comment: comment,
            dynamicGridH: .seqLen
        ))
    }

    private static func emitPerLayerInputsSetupBatched(
        config: SmeltConfig,
        weightEntries: [String: SmeltWeightEntry],
        weightsSlot: Int,
        tokenIdsBatchSlot: Int,
        groupSize: Int,
        quantizeEmbedding: Bool,
        batchSize: Int,
        emitter: inout SmeltCodeEmitter
    ) throws {
        let hidden = config.hiddenSize
        let perLayerHidden = config.hiddenSizePerLayerInput
        let perLayerTotalDim = config.numLayers * perLayerHidden
        let perLayerInputsSlot = SmeltFixedSlot.perLayerInputsBuf.rawValue
        let perLayerScratchSlot = SmeltFixedSlot.perLayerScratchBuf.rawValue
        let fp16 = 2
        let perLayerEmbedEntry = try requireWeight(
            SmeltCanonicalTensorNames.embedTokensPerLayer,
            from: weightEntries
        )

        if quantizeEmbedding, perLayerEmbedEntry.dtype == .u4Lut, let lutOff = perLayerEmbedEntry.lutOffset {
            for b in 0..<batchSize {
                let tokenIdOff = UInt64(b * 4)
                let outputOff = UInt64(b * perLayerTotalDim * fp16)
                _ = try emitter.emit(SmeltDispatch(
                    pipeline: .lutEmbeddingGather,
                    buffers: [
                        SmeltBufferBinding(slot: weightsSlot, offset: perLayerEmbedEntry.offset, index: 0),
                        SmeltBufferBinding(slot: weightsSlot, offset: lutOff, index: 1),
                        SmeltBufferBinding(slot: tokenIdsBatchSlot, offset: tokenIdOff, index: 2),
                        SmeltBufferBinding(slot: perLayerInputsSlot, offset: outputOff, index: 3),
                    ],
                    constants: [
                        SmeltConstantBinding(expression: "\(perLayerTotalDim)", type: .uint32, index: 4),
                        SmeltConstantBinding(expression: "\(groupSize)", type: .uint32, index: 5),
                    ],
                    dispatch: .threads(
                        width: max(perLayerTotalDim / 2, 1), height: 1, depth: 1,
                        tgWidth: min(max(perLayerTotalDim / 2, 1), 256), tgHeight: 1, tgDepth: 1
                    ),
                    comment: "Per-layer token inputs [b=\(b)]",
                    minSeqLen: b + 1
                ))
            }
        } else if quantizeEmbedding,
                  perLayerEmbedEntry.dtype == .affineU4,
                  let scalesOff = perLayerEmbedEntry.scalesOffset,
                  let biasesOff = perLayerEmbedEntry.biasesOffset {
            _ = try emitter.emit(SmeltDispatch(
                pipeline: .affineEmbeddingGatherBatched,
                buffers: [
                    SmeltBufferBinding(slot: weightsSlot, offset: perLayerEmbedEntry.offset, index: 0),
                    SmeltBufferBinding(slot: weightsSlot, offset: scalesOff, index: 1),
                    SmeltBufferBinding(slot: weightsSlot, offset: biasesOff, index: 2),
                    SmeltBufferBinding(slot: tokenIdsBatchSlot, index: 3),
                    SmeltBufferBinding(slot: perLayerInputsSlot, index: 4),
                ],
                constants: [
                    SmeltConstantBinding(expression: "\(perLayerTotalDim)", type: .uint32, index: 5),
                    SmeltConstantBinding(expression: "__seqLen__", type: .uint32, index: 6),
                    SmeltConstantBinding(expression: "\(groupSize)", type: .uint32, index: 7),
                ],
                dispatch: .threads(
                    width: max(perLayerTotalDim / 2, 1), height: batchSize, depth: 1,
                    tgWidth: min(max(perLayerTotalDim / 2, 1), 256), tgHeight: 1, tgDepth: 1
                ),
                comment: "Per-layer token inputs (batched affine gather)",
                dynamicGridH: .seqLen
            ))
        } else if perLayerEmbedEntry.dtype == .binary1
                    || perLayerEmbedEntry.dtype == .ternary2 {
            _ = try emitter.emitSignedEmbeddingGather(
                weightEntry: perLayerEmbedEntry,
                weightsSlot: weightsSlot,
                tokenIdSlot: tokenIdsBatchSlot,
                outputSlot: perLayerInputsSlot,
                hiddenSize: perLayerTotalDim,
                batchSize: batchSize,
                dynamicGridH: .seqLen,
                comment: "Per-layer token inputs (batched native signed gather)"
            )
        } else if perLayerEmbedEntry.dtype == .turboQuantH,
                  let codebookOff = perLayerEmbedEntry.codebookOffset,
                  let codesPerRow = perLayerEmbedEntry.packedRowStride {
            // No batched TQH gather kernel — unroll B single-token
            // gathers (matches the LUT and embed_tokens patterns).
            for b in 0..<batchSize {
                _ = try emitter.emitTurboQuantHEmbeddingGather(
                    codesSlot: weightsSlot, codesOffset: perLayerEmbedEntry.offset,
                    codebookSlot: weightsSlot, codebookOffset: codebookOff,
                    tokenIdSlot: tokenIdsBatchSlot,
                    tokenIdOffset: UInt64(b * 4),
                    outputSlot: perLayerInputsSlot,
                    outputOffset: UInt64(b * perLayerTotalDim * fp16),
                    hiddenSize: perLayerTotalDim, codesPerRow: codesPerRow,
                    minSeqLen: b + 1,
                    comment: "Per-layer token inputs (turbo_quant_h gather) [b=\(b)]"
                )
            }
        } else {
            _ = try emitter.emit(SmeltDispatch(
                pipeline: .embeddingGatherBatched,
                buffers: [
                    SmeltBufferBinding(slot: weightsSlot, offset: perLayerEmbedEntry.offset, index: 0),
                    SmeltBufferBinding(slot: tokenIdsBatchSlot, index: 1),
                    SmeltBufferBinding(slot: perLayerInputsSlot, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "\(perLayerTotalDim)", type: .uint32, index: 3),
                    SmeltConstantBinding(expression: "__seqLen__", type: .uint32, index: 4),
                ],
                dispatch: .threads(
                    width: perLayerTotalDim, height: batchSize, depth: 1,
                    tgWidth: min(perLayerTotalDim, 256), tgHeight: 1, tgDepth: 1
                ),
                comment: "Per-layer token inputs (batched)",
                dynamicGridH: .seqLen
            ))
        }
        try emitDynamicScalarMul(
            emitter: &emitter,
            inputSlot: perLayerInputsSlot,
            outputSlot: perLayerInputsSlot,
            scalar: Float(perLayerHidden).squareRoot(),
            elemsPerToken: perLayerTotalDim,
            comment: "Per-layer token input scale (batched)"
        )

        try emitBatchedMatmul(
            weightEntry: try requireWeight("per_layer_model_projection_weight", from: weightEntries),
            weightsSlot: weightsSlot,
            inputSlot: SmeltFixedSlot.hiddenA.rawValue,
            outputSlot: perLayerScratchSlot,
            rows: perLayerTotalDim, cols: hidden, groupSize: groupSize,
            batchSize: batchSize,
            emitter: &emitter,
            comment: "Per-layer model projection (batched)"
        )
        try emitDynamicScalarMul(
            emitter: &emitter,
            inputSlot: perLayerScratchSlot,
            outputSlot: perLayerScratchSlot,
            scalar: 1.0 / Float(hidden).squareRoot(),
            elemsPerToken: perLayerTotalDim,
            comment: "Per-layer model projection scale (batched)"
        )
        try emitBatchedPerHeadNorm(
            emitter: &emitter,
            inputSlot: perLayerScratchSlot,
            weightSlot: weightsSlot,
            weightOffset: try requireWeight("per_layer_projection_norm_weight", from: weightEntries).offset,
            outputSlot: perLayerScratchSlot,
            numHeads: config.numLayers,
            headDim: perLayerHidden,
            eps: config.rmsEps,
            batchSize: batchSize,
            comment: "Per-layer projection RMS norm (batched)"
        )
        try emitDynamicElementwiseAdd(
            emitter: &emitter,
            inputASlot: perLayerScratchSlot,
            inputBSlot: perLayerInputsSlot,
            outputSlot: perLayerInputsSlot,
            elemsPerToken: perLayerTotalDim,
            comment: "Combine per-layer inputs (batched)"
        )
        try emitDynamicScalarMul(
            emitter: &emitter,
            inputSlot: perLayerInputsSlot,
            outputSlot: perLayerInputsSlot,
            scalar: 0.70710677,
            elemsPerToken: perLayerTotalDim,
            comment: "Per-layer input scale (batched)"
        )
        emitter.recordTraceMarker(
            label: "per_layer_inputs",
            bufferSlot: perLayerInputsSlot
        )
    }

    /// Emit batched FFN activation/projections.
    private static func emitBatchedFFN(
        layerIndex: Int,
        config: SmeltConfig,
        plan: SmeltBufferPlan,
        weightEntries: [String: SmeltWeightEntry],
        weightsSlot: Int,
        groupSize: Int,
        batchSize: Int,
        ffnDownFp32Slot: Int,
        kernelPlan: SmeltKernelPlan,
        emitter: inout SmeltCodeEmitter
    ) throws {
        let hidden = config.hiddenSize
        let ffnDim = config.ffnDim(for: layerIndex)
        let normOutSlot = SmeltFixedSlot.normOutBuf.rawValue

        let gateEntry = try requireWeight(
            SmeltKernelConsumerNaming.ffnGateWeight(layerIndex: layerIndex),
            from: weightEntries
        )
        let upEntry = try requireWeight(
            SmeltKernelConsumerNaming.ffnUpWeight(layerIndex: layerIndex),
            from: weightEntries
        )

        let gateScales = gateEntry.scalesOffset
        let gateBiases = gateEntry.biasesOffset
        let upScales = upEntry.scalesOffset
        let upBiases = upEntry.biasesOffset
        // Entry-resolved group size (per-tensor group sizes are possible); fused
        // gate+up paths require both entries to agree.
        let gateGroupSize = gateEntry.groupSize ?? groupSize
        let gateUpGroupsAgree = gateGroupSize == (upEntry.groupSize ?? groupSize)
        let plannedGateUpRoute = kernelPlan.route(
            kind: .ffnGateUpPrefill,
            context: SmeltKernelLayerConsumerContext(
                config: config,
                layerIndex: layerIndex,
                groupSize: gateGroupSize,
                prefillEngine: "metal"
            )
        )
        let fullFFNRoute = gateUpGroupsAgree
            ? emitter.fusionPlanner.prefillFusedGateUpFull(
                rows: ffnDim,
                cols: hidden,
                groupSize: gateGroupSize,
                activation: config.ffn.activation
            )
            : nil

        // Family via the ONE gateway. The fused full FFN kernel is used iff both weights authorize
        // the affine gate+up family for this activation; fullFFNRoute gates kernel availability for
        // the shape. fp16/bf16/mismatched → probe nil → unfused fallback.
        let gateUpFusion: MatvecKernelTable.Fusion =
            config.ffn.activation == .geglu ? .gateUpGeGLU : .gateUpSwiglu
        let gateUpFamily = emitter.bothFusedFamily(
            gateEntry, upEntry, shape: .gemm, fusion: gateUpFusion)
        let useFullAffineFFNPath =
            gateUpFamily == .affineU4
            && gateScales != nil
            && gateBiases != nil
            && upScales != nil
            && upBiases != nil
            && fullFFNRoute != nil
        let ffnInputBank = config.projectionBank(
            source: .ffnInput, containing: [.ffnGate, .ffnUp])

        if useFullAffineFFNPath {
            let route = fullFFNRoute!
            let rowTile = route.rowTile
            let batchTile = route.batchTile
            let tgWidth = route.threadgroupWidth
            let activationName = config.ffn.activation == .geglu ? "GeGLU" : "SwiGLU"
            _ = try emitter.emit(SmeltDispatch(
                pipeline: route.pipeline,
                buffers: [
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: gateEntry.offset, index: 0
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: gateScales!, index: 1
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: gateBiases!, index: 2
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: upEntry.offset, index: 3
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: upScales!, index: 4
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: upBiases!, index: 5
                    ),
                    SmeltBufferBinding(slot: normOutSlot, index: 6),
                    SmeltBufferBinding(slot: SmeltFixedSlot.ffnIntBuf.rawValue, index: 7),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "__seqLen__",
                        type: .uint32,
                        index: 8
                    )
                ],
                dispatch: .threadgroups(
                    width: (ffnDim + rowTile - 1) / rowTile,
                    height: (batchSize + batchTile - 1) / batchTile,
                    depth: 1,
                    tgWidth: tgWidth, tgHeight: 1, tgDepth: 1
                ),
                comment: "L\(layerIndex) FFN gate+up+\(activationName) (batched qmm)",
                dynamicGridH: .seqLenCeilDiv(batchTile)
            ))
            emitter.recordTraceMarker(
                label: "L\(layerIndex).ffn_int",
                bufferSlot: SmeltFixedSlot.ffnIntBuf.rawValue
            )
            // Fused gate+up share one input slot — capture each weight at this boundary.
            recordGPTQCapturePoint(gateEntry, inputSlot: normOutSlot, k: hidden, emitter: &emitter)
            recordGPTQCapturePoint(upEntry, inputSlot: normOutSlot, k: hidden, emitter: &emitter)
        } else if config.ffn.activation == .swiglu,
                  gateUpFamily == .binary1,
                  gateUpGroupsAgree,
                  let view = ffnInputBank?.activationView,
                  ffnInputBank?.usesActivationView(at: layerIndex) == true,
                  plan.projectionActivationPlanesSlot >= 0,
                  plan.projectionActivationScalesSlot >= 0,
                  let bitplaneLines = try emitter
                    .emitSignedBitplaneProjectionBankBatchedIfPossible(
                        view: view,
                        members: [
                            (gateEntry, SmeltFixedSlot.ffnGateBuf.rawValue, ffnDim),
                            (upEntry, SmeltFixedSlot.ffnUpBuf.rawValue, ffnDim),
                        ],
                        weightsSlot: weightsSlot,
                        inputSlot: normOutSlot,
                        planesSlot: plan.projectionActivationPlanesSlot,
                        activationScalesSlot: plan.projectionActivationScalesSlot,
                        cols: hidden,
                        batchSize: batchSize,
                        producerComment: "L\(layerIndex) CAM FFN-input view (batched)",
                        projectionComment: "L\(layerIndex) FFN gate/up (bit-GEMM B4)"
                    )
        {
            _ = bitplaneLines
            _ = try emitter.emit(SmeltDispatch(
                pipeline: .swigluFused,
                buffers: [
                    SmeltBufferBinding(
                        slot: SmeltFixedSlot.ffnGateBuf.rawValue, index: 0),
                    SmeltBufferBinding(
                        slot: SmeltFixedSlot.ffnUpBuf.rawValue, index: 1),
                    SmeltBufferBinding(
                        slot: SmeltFixedSlot.ffnIntBuf.rawValue, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: seqLenExpr(ffnDim), type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: ffnDim * batchSize, height: 1, depth: 1,
                    tgWidth: min(ffnDim * batchSize, 1024),
                    tgHeight: 1, tgDepth: 1
                ),
                comment: "L\(layerIndex) SwiGLU (batched activation-view bank)",
                dynamicGridW: seqLenGrid(ffnDim)
            ))
            emitter.recordTraceMarker(
                label: "L\(layerIndex).ffn_int",
                bufferSlot: SmeltFixedSlot.ffnIntBuf.rawValue
            )
            recordGPTQCapturePoint(
                gateEntry, inputSlot: normOutSlot, k: hidden, emitter: &emitter)
            recordGPTQCapturePoint(
                upEntry, inputSlot: normOutSlot, k: hidden, emitter: &emitter)
        } else if config.ffn.activation == .swiglu,
                  gateUpFamily == .binary1,
                  gateUpGroupsAgree
        {
            _ = try emitter.emitSignedBinaryGateUpSwiglu(
                gateEntry: gateEntry,
                upEntry: upEntry,
                weightsSlot: weightsSlot,
                inputBinding: SmeltBufferBinding(slot: normOutSlot, index: 4),
                outputBinding: SmeltBufferBinding(
                    slot: SmeltFixedSlot.ffnIntBuf.rawValue, index: 5),
                rows: ffnDim,
                cols: hidden,
                batchSize: batchSize,
                dynamicGridH: .seqLen,
                comment: "L\(layerIndex) FFN gate+up+SwiGLU (binary g128 batched)"
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).ffn_int",
                bufferSlot: SmeltFixedSlot.ffnIntBuf.rawValue
            )
            recordGPTQCapturePoint(
                gateEntry, inputSlot: normOutSlot, k: hidden, emitter: &emitter)
            recordGPTQCapturePoint(
                upEntry, inputSlot: normOutSlot, k: hidden, emitter: &emitter)
        } else if config.ffn.activation == .swiglu,
                  gateUpFamily == .affineU4,
                  gateUpGroupsAgree
        {
            try emitUnrolledFusedAffineGateUpSwiglu(
                gateEntry: gateEntry,
                upEntry: upEntry,
                weightsSlot: weightsSlot,
                inputSlot: normOutSlot,
                outputSlot: SmeltFixedSlot.ffnIntBuf.rawValue,
                rows: ffnDim,
                cols: hidden,
                groupSize: gateGroupSize,
                batchSize: batchSize,
                generatedFullRoute: plannedGateUpRoute,
                emitter: &emitter,
                comment: "L\(layerIndex) FFN gate+up+SwiGLU (batched)"
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).ffn_int",
                bufferSlot: SmeltFixedSlot.ffnIntBuf.rawValue
            )
        } else {

            // Gate projection (batched matmul)
            try emitBatchedMatmul(
                weightEntry: gateEntry,
                weightsSlot: weightsSlot,
                inputSlot: normOutSlot,
                outputSlot: SmeltFixedSlot.ffnGateBuf.rawValue,
                rows: ffnDim, cols: hidden, groupSize: groupSize,
                batchSize: batchSize,
                emitter: &emitter,
                comment: "L\(layerIndex) FFN gate (batched)"
            )

            // Up projection (batched matmul)
            try emitBatchedMatmul(
                weightEntry: upEntry,
                weightsSlot: weightsSlot,
                inputSlot: normOutSlot,
                outputSlot: SmeltFixedSlot.ffnUpBuf.rawValue,
                rows: ffnDim, cols: hidden, groupSize: groupSize,
                batchSize: batchSize,
                emitter: &emitter,
                comment: "L\(layerIndex) FFN up (batched)"
            )

            let activationPipeline: SmeltPipeline =
                config.ffn.activation == .geglu ? .gegluFused : .swigluFused
            let activationName = config.ffn.activation == .geglu ? "GeGLU" : "SwiGLU"

            _ = try emitter.emit(SmeltDispatch(
                pipeline: activationPipeline,
                buffers: [
                    SmeltBufferBinding(
                        slot: SmeltFixedSlot.ffnGateBuf.rawValue, index: 0
                    ),
                    SmeltBufferBinding(
                        slot: SmeltFixedSlot.ffnUpBuf.rawValue, index: 1
                    ),
                    SmeltBufferBinding(
                        slot: SmeltFixedSlot.ffnIntBuf.rawValue, index: 2
                    ),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: seqLenExpr(ffnDim), type: .uint32, index: 3
                    ),
                ],
                dispatch: .threads(
                    width: ffnDim * batchSize, height: 1, depth: 1,
                    tgWidth: min(ffnDim * batchSize, 1024), tgHeight: 1, tgDepth: 1
                ),
                comment: "L\(layerIndex) \(activationName) (batched)",
                dynamicGridW: seqLenGrid(ffnDim)
            ))
            emitter.recordTraceMarker(
                label: "L\(layerIndex).ffn_int",
                bufferSlot: SmeltFixedSlot.ffnIntBuf.rawValue
            )
        }

        // Down projection: activations can exceed FP16 range before the following RMSNorm.
        let downEntry = try requireWeight(
            SmeltKernelConsumerNaming.ffnDownWeight(layerIndex: layerIndex),
            from: weightEntries
        )
        let downGroupSize = downEntry.groupSize ?? groupSize
        let plannedDownRoute = kernelPlan.route(
            kind: .ffnDownPrefill,
            context: SmeltKernelLayerConsumerContext(
                config: config,
                layerIndex: layerIndex,
                groupSize: downGroupSize,
                prefillEngine: "metal"
            )
        )
        let downOutputSlot = ffnDownFp32Slot >= 0 && downEntry.dtype == .fp16
            ? ffnDownFp32Slot
            : SmeltFixedSlot.ffnDownBuf.rawValue
        if ffnDownFp32Slot >= 0 && downEntry.dtype == .fp16 {
            try emitBatchedFP16MatvecFP32Out(
                weightEntry: downEntry,
                weightsSlot: weightsSlot,
                inputSlot: SmeltFixedSlot.ffnIntBuf.rawValue,
                outputSlot: downOutputSlot,
                rows: hidden, cols: ffnDim,
                batchSize: batchSize,
                emitter: &emitter,
                comment: "L\(layerIndex) FFN down FP32 (batched)"
            )
        } else {
            let ffnDownBank = config.projectionBank(
                source: .ffnIntermediate, containing: [.ffnDown])
            let usedDownActivationView: Bool
            if let view = ffnDownBank?.activationView,
               ffnDownBank?.usesActivationView(at: layerIndex) == true,
               plan.projectionActivationPlanesSlot >= 0,
               plan.projectionActivationScalesSlot >= 0,
               let lines = try emitter
                .emitSignedBitplaneProjectionBankBatchedIfPossible(
                    view: view,
                    members: [(downEntry, downOutputSlot, hidden)],
                    weightsSlot: weightsSlot,
                    inputSlot: SmeltFixedSlot.ffnIntBuf.rawValue,
                    planesSlot: plan.projectionActivationPlanesSlot,
                    activationScalesSlot: plan.projectionActivationScalesSlot,
                    cols: ffnDim,
                    batchSize: batchSize,
                    producerComment: "L\(layerIndex) CAM FFN-intermediate view (batched)",
                    projectionComment: "L\(layerIndex) FFN down (bit-GEMM B4)"
                ) {
                _ = lines
                usedDownActivationView = true
            } else {
                usedDownActivationView = false
            }
            if !usedDownActivationView {
                try emitBatchedMatmul(
                    weightEntry: downEntry,
                    weightsSlot: weightsSlot,
                    inputSlot: SmeltFixedSlot.ffnIntBuf.rawValue,
                    outputSlot: downOutputSlot,
                    rows: hidden, cols: ffnDim, groupSize: groupSize,
                    batchSize: batchSize,
                    emitter: &emitter,
                    comment: "L\(layerIndex) FFN down (batched)",
                    generatedFullRoute: plannedDownRoute
                )
            }
        }
        emitter.recordTraceMarker(
            label: "L\(layerIndex).ffn_down",
            bufferSlot: downOutputSlot
        )
    }

    private static func emitMatvecVarSlice(
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlotVar: String,
        inputOffset: UInt64,
        outputSlot: Int,
        rows: Int,
        cols: Int,
        groupSize: Int,
        emitter: inout SmeltCodeEmitter,
        comment: String? = nil,
        minSeqLen: Int? = nil
    ) throws {
        // Family via the ONE gateway (no inline dtype routing). Variable-input-slot gemv:
        // tqh-variable is notMeaningful (no consumer), so matvecFamily throws .notMeaningful here
        // (its old explicit PrefillEmitterError branch is dropped); malformed quant is LOUD too.
        let family = try emitter.matvecFamily(weightEntry, shape: .gemv, output: .fp16, slot: .variable)
        if family == .lutU4, let lutOff = weightEntry.lutOffset {
            _ = try emitter.emit(SmeltDispatch(
                pipeline: .fusedLutMatvec,
                buffers: [
                    SmeltBufferBinding(slot: weightsSlot, offset: weightEntry.offset, index: 0),
                    SmeltBufferBinding(slot: weightsSlot, offset: lutOff, index: 1),
                    SmeltBufferBinding(variableSlot: inputSlotVar, offset: inputOffset, index: 2),
                    SmeltBufferBinding(slot: outputSlot, index: 3),
                ],
                constants: [
                    SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 4),
                ],
                dispatch: .threadgroups(
                    width: (rows + 7) / 8, height: 1, depth: 1,
                    tgWidth: 64, tgHeight: 1, tgDepth: 1
                ),
                comment: comment,
                fcCols: cols,
                fcGroupSize: groupSize,
                minSeqLen: minSeqLen
            ))
        } else if family == .affineU4,
                  let scalesOff = weightEntry.scalesOffset,
                  let biasesOff = weightEntry.biasesOffset {
            let route = emitter.fusionPlanner.decodeAffineMatvec(
                rows: rows,
                cols: cols,
                groupSize: groupSize
            )
            let specializedPipeline = route.pipeline
            _ = try emitter.emit(SmeltDispatch(
                pipeline: specializedPipeline ?? .affineMatvec,
                buffers: [
                    SmeltBufferBinding(slot: weightsSlot, offset: weightEntry.offset, index: 0),
                    SmeltBufferBinding(slot: weightsSlot, offset: scalesOff, index: 1),
                    SmeltBufferBinding(slot: weightsSlot, offset: biasesOff, index: 2),
                    SmeltBufferBinding(variableSlot: inputSlotVar, offset: inputOffset, index: 3),
                    SmeltBufferBinding(slot: outputSlot, index: 4),
                ],
                constants: specializedPipeline == nil ? [
                    SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 5),
                ] : [],
                dispatch: .threadgroups(
                    width: (rows + route.rowTile - 1) / route.rowTile,
                    height: 1, depth: 1,
                    tgWidth: 64, tgHeight: 1, tgDepth: 1
                ),
                comment: comment,
                fcCols: specializedPipeline == nil ? cols : nil,
                fcGroupSize: specializedPipeline == nil ? groupSize : nil,
                minSeqLen: minSeqLen
            ))
        } else if case let .dense(dt) = family {
            // fp16-act dense gemv: .fp16 → fp16_matvec, .bf16/.fp32 → the U2
            // fp16_matvec_{bf16,fp32}w kernels (same dispatch; only the weight load differs).
            _ = try emitter.emit(SmeltDispatch(
                pipeline: try SmeltCodeEmitter.fp16DenseMatvecPipeline(dt),
                buffers: [
                    SmeltBufferBinding(slot: weightsSlot, offset: weightEntry.offset, index: 0),
                    SmeltBufferBinding(variableSlot: inputSlotVar, offset: inputOffset, index: 1),
                    SmeltBufferBinding(slot: outputSlot, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "\(cols)", type: .uint32, index: 3),
                ],
                dispatch: .threadgroups(
                    width: rows, height: 1, depth: 1,
                    tgWidth: 256, tgHeight: 1, tgDepth: 1
                ),
                comment: comment,
                minSeqLen: minSeqLen
            ))
        } else {
            try throwUnroutableMatvecFamily("emitMatvecVarSlice", family, weightEntry)
        }
    }

    private static func emitMatvecFixed(
        weightEntry: SmeltWeightEntry,
        weightsSlot: Int,
        inputSlot: Int,
        outputSlot: Int,
        rows: Int,
        cols: Int,
        groupSize: Int,
        emitter: inout SmeltCodeEmitter,
        comment: String? = nil,
        minSeqLen: Int? = nil
    ) throws {
        // Family via the ONE gateway (no inline dtype routing). Fixed-input-slot gemv: tqh-fixed
        // is registered (the tied LM head); bf16/fp32 and malformed quant entries are LOUD.
        let family = try emitter.matvecFamily(weightEntry, shape: .gemv, output: .fp16, slot: .fixed)
        if family == .lutU4, let lutOff = weightEntry.lutOffset {
            _ = try emitter.emit(SmeltDispatch(
                pipeline: .fusedLutMatvec,
                buffers: [
                    SmeltBufferBinding(slot: weightsSlot, offset: weightEntry.offset, index: 0),
                    SmeltBufferBinding(slot: weightsSlot, offset: lutOff, index: 1),
                    SmeltBufferBinding(slot: inputSlot, index: 2),
                    SmeltBufferBinding(slot: outputSlot, index: 3),
                ],
                constants: [
                    SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 4),
                ],
                dispatch: .threadgroups(
                    width: (rows + 7) / 8, height: 1, depth: 1,
                    tgWidth: 64, tgHeight: 1, tgDepth: 1
                ),
                comment: comment,
                fcCols: cols,
                fcGroupSize: groupSize,
                minSeqLen: minSeqLen
            ))
        } else if family == .affineU4,
                  let scalesOff = weightEntry.scalesOffset,
                  let biasesOff = weightEntry.biasesOffset {
            let route = emitter.fusionPlanner.decodeAffineMatvec(
                rows: rows,
                cols: cols,
                groupSize: groupSize
            )
            let specializedPipeline = route.pipeline
            _ = try emitter.emit(SmeltDispatch(
                pipeline: specializedPipeline ?? .affineMatvec,
                buffers: [
                    SmeltBufferBinding(slot: weightsSlot, offset: weightEntry.offset, index: 0),
                    SmeltBufferBinding(slot: weightsSlot, offset: scalesOff, index: 1),
                    SmeltBufferBinding(slot: weightsSlot, offset: biasesOff, index: 2),
                    SmeltBufferBinding(slot: inputSlot, index: 3),
                    SmeltBufferBinding(slot: outputSlot, index: 4),
                ],
                constants: specializedPipeline == nil ? [
                    SmeltConstantBinding(expression: "\(rows)", type: .uint32, index: 5),
                ] : [],
                dispatch: .threadgroups(
                    width: (rows + route.rowTile - 1) / route.rowTile,
                    height: 1, depth: 1,
                    tgWidth: 64, tgHeight: 1, tgDepth: 1
                ),
                comment: comment,
                fcCols: specializedPipeline == nil ? cols : nil,
                fcGroupSize: specializedPipeline == nil ? groupSize : nil,
                minSeqLen: minSeqLen
            ))
        } else if family == .tqh,
                  let codebookOff = weightEntry.codebookOffset,
                  let codesPerRow = weightEntry.packedRowStride {
            _ = try emitter.emitTQHMatvec(
                codesSlot: weightsSlot, codesOffset: weightEntry.offset,
                codebookSlot: weightsSlot, codebookOffset: codebookOff,
                inputSlot: inputSlot,
                xHatScratchSlot: SmeltFixedSlot.tqhMatvecXHatBuf.rawValue,
                outputSlot: outputSlot,
                rows: rows, cols: cols, codesPerRow: codesPerRow,
                minSeqLen: minSeqLen,
                comment: comment
            )
        } else if case let .dense(dt) = family {
            // fp16-act dense gemv: .fp16 → fp16_matvec, .bf16/.fp32 → the U2
            // fp16_matvec_{bf16,fp32}w kernels (same dispatch; only the weight load differs).
            _ = try emitter.emit(SmeltDispatch(
                pipeline: try SmeltCodeEmitter.fp16DenseMatvecPipeline(dt),
                buffers: [
                    SmeltBufferBinding(slot: weightsSlot, offset: weightEntry.offset, index: 0),
                    SmeltBufferBinding(slot: inputSlot, index: 1),
                    SmeltBufferBinding(slot: outputSlot, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "\(cols)", type: .uint32, index: 3),
                ],
                dispatch: .threadgroups(
                    width: rows, height: 1, depth: 1,
                    tgWidth: 256, tgHeight: 1, tgDepth: 1
                ),
                comment: comment,
                minSeqLen: minSeqLen
            ))
        } else {
            try throwUnroutableMatvecFamily("emitMatvecFixed", family, weightEntry)
        }
    }

    private static func emitRMSNormFixed(
        inputSlot: Int,
        weightSlot: Int,
        weightOffset: UInt64,
        outputSlot: Int,
        dim: Int,
        eps: Float,
        emitter: inout SmeltCodeEmitter,
        comment: String? = nil,
        minSeqLen: Int? = nil
    ) throws {
        let route = emitter.fusionPlanner.decodeRMSNorm(
            dim: dim,
            eps: eps
        )
        let specializedPipeline = route.pipeline
        let specializedThreads = route.threadgroupWidth
        _ = try emitter.emit(SmeltDispatch(
            pipeline: specializedPipeline ?? .rmsNorm1PW,
            buffers: [
                SmeltBufferBinding(slot: inputSlot, index: 0),
                SmeltBufferBinding(slot: weightSlot, offset: weightOffset, index: 1),
                SmeltBufferBinding(slot: outputSlot, index: 2),
            ],
            constants: specializedPipeline == nil ? [
                SmeltConstantBinding(expression: "\(dim)", type: .uint32, index: 3),
                SmeltConstantBinding(expression: "\(eps)", type: .float32, index: 4),
            ] : [],
            dispatch: .threadgroups(
                width: 1, height: 1, depth: 1,
                tgWidth: specializedThreads ?? min(dim, 1024),
                tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            minSeqLen: minSeqLen
        ))
    }

    private static func emitPerLayerResidualBranchBatched(
        layerIndex: Int,
        config: SmeltConfig,
        weightEntries: [String: SmeltWeightEntry],
        weightsSlot: Int,
        groupSize: Int,
        batchSize: Int,
        emitter: inout SmeltCodeEmitter
    ) throws {
        let hidden = config.hiddenSize
        let perLayerHidden = config.hiddenSizePerLayerInput
        let perLayerTotalDim = config.numLayers * perLayerHidden
        let fp16 = 2
        let perLayerOffset = UInt64(layerIndex * perLayerHidden * fp16)
        let gateEntry = try requireWeight(
            "layers_\(layerIndex)_per_layer_input_gate_weight",
            from: weightEntries
        )
        let projectionEntry = try requireWeight(
            "layers_\(layerIndex)_per_layer_projection_weight",
            from: weightEntries
        )
        let postPerLayerNormEntry = try requireWeight(
            "layers_\(layerIndex)_post_per_layer_input_norm_weight",
            from: weightEntries
        )

        // Family via the ONE gateway: the batched-affine per-layer path needs BOTH the gate and
        // projection weights to authorize the affine family (.none — these are two SEPARATE
        // batched-affine matmuls with a GeGLU between, not one fused kernel). Non-affine /
        // mismatched → nil → the scalar fallback below (itself gateway-routed).
        let perLayerFamily = emitter.bothFusedFamily(
            gateEntry, projectionEntry, shape: .gemm, fusion: .none)
        if perLayerFamily == .affineU4,
           gateEntry.scalesOffset != nil,
           gateEntry.biasesOffset != nil,
           projectionEntry.scalesOffset != nil,
           projectionEntry.biasesOffset != nil,
           let gateRoute = emitter.fusionPlanner.prefillAffineBatched(
               rows: perLayerHidden,
               cols: hidden,
               groupSize: groupSize
           ),
           let projectionRoute = emitter.fusionPlanner.prefillAffineBatched(
               rows: hidden,
               cols: perLayerHidden,
               groupSize: groupSize
           )
        {
            try emitBatchedAffineMatmul(
                weightEntry: gateEntry,
                weightsSlot: weightsSlot,
                inputBinding: SmeltBufferBinding(variableSlot: "alt", index: 3),
                outputSlot: SmeltFixedSlot.ffnGateBuf.rawValue,
                rows: perLayerHidden,
                cols: hidden,
                route: gateRoute,
                batchSize: batchSize,
                emitter: &emitter,
                comment: "L\(layerIndex) per-layer input gate (batched)"
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).per_layer_gate",
                bufferSlot: SmeltFixedSlot.ffnGateBuf.rawValue
            )

            _ = try emitter.emit(SmeltDispatch(
                pipeline: .gegluFusedStridedBatched,
                buffers: [
                    SmeltBufferBinding(slot: SmeltFixedSlot.ffnGateBuf.rawValue, index: 0),
                    SmeltBufferBinding(
                        slot: SmeltFixedSlot.perLayerInputsBuf.rawValue,
                        offset: perLayerOffset,
                        index: 1
                    ),
                    SmeltBufferBinding(slot: SmeltFixedSlot.ffnIntBuf.rawValue, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "\(perLayerHidden)", type: .uint32, index: 3
                    ),
                    SmeltConstantBinding(
                        expression: "\(perLayerTotalDim)", type: .uint32, index: 4
                    ),
                ],
                dispatch: .threads(
                    width: perLayerHidden, height: batchSize, depth: 1,
                    tgWidth: min(perLayerHidden, 1024), tgHeight: 1, tgDepth: 1
                ),
                comment: "L\(layerIndex) per-layer GeGLU (batched)",
                dynamicGridH: .seqLen
            ))
            emitter.recordTraceMarker(
                label: "L\(layerIndex).per_layer_int",
                bufferSlot: SmeltFixedSlot.ffnIntBuf.rawValue
            )

            try emitBatchedAffineMatmul(
                weightEntry: projectionEntry,
                weightsSlot: weightsSlot,
                inputBinding: SmeltBufferBinding(
                    slot: SmeltFixedSlot.ffnIntBuf.rawValue,
                    index: 3
                ),
                outputSlot: SmeltFixedSlot.normOutBuf.rawValue,
                rows: hidden,
                cols: perLayerHidden,
                route: projectionRoute,
                batchSize: batchSize,
                emitter: &emitter,
                comment: "L\(layerIndex) per-layer projection (batched)"
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).per_layer_proj",
                bufferSlot: SmeltFixedSlot.normOutBuf.rawValue
            )

            try emitBatchedRMSNorm(
                emitter: &emitter,
                inputSlot: .fixed(SmeltFixedSlot.normOutBuf.rawValue),
                weightSlot: weightsSlot,
                weightOffset: postPerLayerNormEntry.offset,
                outputSlot: SmeltFixedSlot.residualBuf.rawValue,
                dim: hidden,
                eps: config.rmsEps,
                batchSize: batchSize,
                comment: "L\(layerIndex) post per-layer input norm (batched)"
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).post_per_layer_norm",
                bufferSlot: SmeltFixedSlot.residualBuf.rawValue
            )

            try emitDynamicElementwiseAddVar(
                emitter: &emitter,
                inputAVar: "alt",
                inputBSlot: SmeltFixedSlot.residualBuf.rawValue,
                outputVar: "alt",
                elemsPerToken: hidden,
                comment: "L\(layerIndex) per-layer residual add (batched)"
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).post_per_layer_residual",
                bufferSlot: SmeltFixedSlot.hiddenA.rawValue
            )
            return
        }

        for b in 0..<batchSize {
            let hiddenOffset = UInt64(b * hidden * fp16)
            let perLayerTokenOffset = UInt64(b * perLayerTotalDim * fp16) + perLayerOffset

            try emitMatvecVarSlice(
                weightEntry: gateEntry,
                weightsSlot: weightsSlot,
                inputSlotVar: "alt",
                inputOffset: hiddenOffset,
                outputSlot: SmeltFixedSlot.ffnGateBuf.rawValue,
                rows: perLayerHidden,
                cols: hidden,
                groupSize: groupSize,
                emitter: &emitter,
                comment: "L\(layerIndex) per-layer input gate [b=\(b)]",
                minSeqLen: b + 1
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).per_layer_gate",
                bufferSlot: SmeltFixedSlot.ffnGateBuf.rawValue
            )

            _ = try emitter.emit(SmeltDispatch(
                pipeline: .gegluFused,
                buffers: [
                    SmeltBufferBinding(slot: SmeltFixedSlot.ffnGateBuf.rawValue, index: 0),
                    SmeltBufferBinding(slot: SmeltFixedSlot.perLayerInputsBuf.rawValue, offset: perLayerTokenOffset, index: 1),
                    SmeltBufferBinding(slot: SmeltFixedSlot.ffnIntBuf.rawValue, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "\(perLayerHidden)", type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: perLayerHidden, height: 1, depth: 1,
                    tgWidth: min(perLayerHidden, 1024), tgHeight: 1, tgDepth: 1
                ),
                comment: "L\(layerIndex) per-layer GeGLU [b=\(b)]",
                minSeqLen: b + 1
            ))
            emitter.recordTraceMarker(
                label: "L\(layerIndex).per_layer_int",
                bufferSlot: SmeltFixedSlot.ffnIntBuf.rawValue
            )

            try emitMatvecFixed(
                weightEntry: projectionEntry,
                weightsSlot: weightsSlot,
                inputSlot: SmeltFixedSlot.ffnIntBuf.rawValue,
                outputSlot: SmeltFixedSlot.normOutBuf.rawValue,
                rows: hidden,
                cols: perLayerHidden,
                groupSize: groupSize,
                emitter: &emitter,
                comment: "L\(layerIndex) per-layer projection [b=\(b)]",
                minSeqLen: b + 1
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).per_layer_proj",
                bufferSlot: SmeltFixedSlot.normOutBuf.rawValue
            )

            try emitRMSNormFixed(
                inputSlot: SmeltFixedSlot.normOutBuf.rawValue,
                weightSlot: weightsSlot,
                weightOffset: postPerLayerNormEntry.offset,
                outputSlot: SmeltFixedSlot.residualBuf.rawValue,
                dim: hidden,
                eps: config.rmsEps,
                emitter: &emitter,
                comment: "L\(layerIndex) post per-layer input norm [b=\(b)]",
                minSeqLen: b + 1
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).post_per_layer_norm",
                bufferSlot: SmeltFixedSlot.residualBuf.rawValue
            )

            _ = try emitter.emit(SmeltDispatch(
                pipeline: .elementwiseAdd,
                buffers: [
                    SmeltBufferBinding(variableSlot: "alt", offset: hiddenOffset, index: 0),
                    SmeltBufferBinding(slot: SmeltFixedSlot.residualBuf.rawValue, index: 1),
                    SmeltBufferBinding(variableSlot: "alt", offset: hiddenOffset, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(expression: "\(hidden)", type: .uint32, index: 3),
                ],
                dispatch: .threads(
                    width: hidden, height: 1, depth: 1,
                    tgWidth: min(hidden, 1024), tgHeight: 1, tgDepth: 1
                ),
                comment: "L\(layerIndex) per-layer residual add [b=\(b)]",
                minSeqLen: b + 1
            ))
            emitter.recordTraceMarker(
                label: "L\(layerIndex).post_per_layer_residual",
                bufferSlot: SmeltFixedSlot.hiddenA.rawValue
            )
        }
    }

    // MARK: - DeltaNet prefill layer

    /// Emit one DeltaNet layer: batched projections, unrolled sequential recurrence.
    private static func emitPrefillDeltaLayer(
        layerIndex: Int,
        deltaIndex: Int,
        config: SmeltConfig,
        plan: SmeltBufferPlan,
        weightEntries: [String: SmeltWeightEntry],
        weightsSlot: Int,
        groupSize: Int,
        batchSize: Int,
        checkpointRecurrentState: Bool,
        emitter: inout SmeltCodeEmitter
    ) throws {
        let delta = config.delta!
        let hidden = config.hiddenSize
        let valueDim = delta.zDim
        let qkHeads = delta.qkHeads
        let prefix = "layers_\(layerIndex)_linear_attn"

        let qkvSlot = SmeltFixedSlot.qkvBuf.rawValue
        let zSlot = SmeltFixedSlot.zBuf.rawValue
        let aSlot = SmeltFixedSlot.aBuf.rawValue
        let bSlot = SmeltFixedSlot.bBuf.rawValue
        let normOutSlot = SmeltFixedSlot.normOutBuf.rawValue

        let qkvEntry = try requireWeight(
            "\(prefix)_in_proj_qkv_weight", from: weightEntries)
        let zEntry = try requireWeight(
            "\(prefix)_in_proj_z_weight", from: weightEntries)
        let bEntry = try requireWeight(
            "\(prefix)_in_proj_b_weight", from: weightEntries)
        let aEntry = try requireWeight(
            "\(prefix)_in_proj_a_weight", from: weightEntries)
        let deltaInputBank = config.projectionBank(
            source: .deltaInput,
            containing: [.deltaQKV, .deltaZ, .deltaA, .deltaB]
        )
        let usedActivationView: Bool
        if let view = deltaInputBank?.activationView,
           deltaInputBank?.usesActivationView(at: layerIndex) == true,
           plan.projectionActivationPlanesSlot >= 0,
           plan.projectionActivationScalesSlot >= 0,
           let lines = try emitter.emitSignedBitplaneProjectionBankBatchedIfPossible(
               view: view,
               members: [
                   (qkvEntry, qkvSlot, delta.qkvDim),
                   (zEntry, zSlot, valueDim),
                   (bEntry, bSlot, delta.numHeads),
                   (aEntry, aSlot, delta.numHeads),
               ],
               weightsSlot: weightsSlot,
               inputSlot: normOutSlot,
               planesSlot: plan.projectionActivationPlanesSlot,
               activationScalesSlot: plan.projectionActivationScalesSlot,
               cols: hidden,
               batchSize: batchSize,
               producerComment: "L\(layerIndex) CAM delta-input view (batched)",
               projectionComment: "L\(layerIndex) CAM delta-input projection (bit-GEMM B4)"
           ) {
            _ = lines
            usedActivationView = true
        } else {
            usedActivationView = false
        }

        if !usedActivationView {
            // Batched QKV projection
            try emitBatchedMatmul(
                weightEntry: qkvEntry,
                weightsSlot: weightsSlot,
                inputSlot: normOutSlot,
                outputSlot: qkvSlot,
                rows: delta.qkvDim, cols: hidden, groupSize: groupSize,
                batchSize: batchSize,
                emitter: &emitter,
                comment: "L\(layerIndex) QKV proj (batched)"
            )
        }
        emitter.recordTraceMarker(
            label: "L\(layerIndex).delta_qkv",
            bufferSlot: qkvSlot
        )

        if !usedActivationView {
            // Batched Z projection
            try emitBatchedMatmul(
                weightEntry: zEntry,
                weightsSlot: weightsSlot,
                inputSlot: normOutSlot,
                outputSlot: zSlot,
                rows: valueDim, cols: hidden, groupSize: groupSize,
                batchSize: batchSize,
                emitter: &emitter,
                comment: "L\(layerIndex) Z proj (batched)"
            )
        }

        // Family via the ONE gateway: fused batched dual iff both affine_u4.
        let abFamily = emitter.bothFusedFamily(
            bEntry, aEntry, shape: .gemm, fusion: .dualMatvec)
        if usedActivationView {
            emitter.recordTraceMarker(
                label: "L\(layerIndex).delta_b",
                bufferSlot: bSlot
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).delta_a",
                bufferSlot: aSlot
            )
        } else if abFamily == .affineU4 {
            try emitUnrolledFusedDualAffineMatvec(
                weightEntry1: bEntry,
                weightEntry2: aEntry,
                weightsSlot: weightsSlot,
                inputSlot: normOutSlot,
                outputSlot1: bSlot,
                outputSlot2: aSlot,
                rows: delta.numHeads,
                cols: hidden,
                groupSize: groupSize,
                batchSize: batchSize,
                emitter: &emitter,
                comment: "L\(layerIndex) B+A proj (batched)"
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).delta_b",
                bufferSlot: bSlot
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).delta_a",
                bufferSlot: aSlot
            )
        } else {
            // Batched B projection (reference dispatches B before A)
            try emitBatchedMatmul(
                weightEntry: bEntry,
                weightsSlot: weightsSlot,
                inputSlot: normOutSlot,
                outputSlot: bSlot,
                rows: delta.numHeads, cols: hidden, groupSize: groupSize,
                batchSize: batchSize,
                emitter: &emitter,
                comment: "L\(layerIndex) B proj (batched)"
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).delta_b",
                bufferSlot: bSlot
            )

            // Batched A projection
            try emitBatchedMatmul(
                weightEntry: aEntry,
                weightsSlot: weightsSlot,
                inputSlot: normOutSlot,
                outputSlot: aSlot,
                rows: delta.numHeads, cols: hidden, groupSize: groupSize,
                batchSize: batchSize,
                emitter: &emitter,
                comment: "L\(layerIndex) A proj (batched)"
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).delta_a",
                bufferSlot: aSlot
            )
        }
        let useD128H16Specialization =
            delta.qkvDim == 6_144
            && delta.convKernel == 4
            && delta.headDim == 128
            && delta.numHeads == 16
        let useD128H32QK16Specialization =
            delta.qkvDim == 8_192
            && delta.convKernel == 4
            && delta.headDim == 128
            && delta.numHeads == 32
            && qkHeads == 16
        let useD128H48QK16Specialization =
            delta.qkvDim == 10_240
            && delta.convKernel == 4
            && delta.headDim == 128
            && delta.numHeads == 48
            && qkHeads == 16
        let qkNormThreads = max(32, ((delta.headDim + 127) / 128) * 32)

        // --- DeltaNet prompt recurrence ---
        let convStateSlot = plan.convStateBaseSlot + deltaIndex
        let recStateSlot = plan.recStateBaseSlot + deltaIndex
        let convHistorySlot = plan.slots.first {
            $0.name == "convStateHistory_\(deltaIndex)"
        }?.index
        let recHistorySlot = plan.slots.first {
            $0.name == "recStateHistory_\(deltaIndex)"
        }?.index
        if checkpointRecurrentState,
           convHistorySlot == nil || recHistorySlot == nil {
            throw PrefillEmitterError.unsupported(
                "verify-argmax recurrent checkpoint slots missing for "
                    + "DeltaNet layer \(deltaIndex)"
            )
        }
        let convWeightOffset = try requireWeight(
            "\(prefix)_conv1d_weight", from: weightEntries
        ).offset
        let aLogOffset = try requireWeight(
            "\(prefix)_A_log", from: weightEntries
        ).offset
        let dtBiasOffset = try requireWeight(
            "\(prefix)_dt_bias", from: weightEntries
        ).offset

        let recOutSlot = SmeltFixedSlot.recOutBuf.rawValue

        if useD128H16Specialization && !checkpointRecurrentState {
            _ = try emitter.emit(SmeltDispatch(
                pipeline: .conv1dUpdateSilu6144x4Prefill,
                buffers: [
                    SmeltBufferBinding(slot: convStateSlot, index: 0),
                    SmeltBufferBinding(slot: qkvSlot, index: 1),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: convWeightOffset, index: 2
                    ),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "__seqLen__", type: .uint32, index: 3
                    ),
                ],
                dispatch: .threads(
                    width: delta.qkvDim, height: 1, depth: 1,
                    tgWidth: 256, tgHeight: 1, tgDepth: 1
                ),
                comment: "L\(layerIndex) QKV conv1d update + SiLU (prefill)"
            ))
            emitter.recordTraceMarker(
                label: "L\(layerIndex).delta_conv_raw",
                bufferSlot: qkvSlot
            )

            _ = try emitter.emit(SmeltDispatch(
                pipeline: .rmsScaleQK,
                buffers: [
                    SmeltBufferBinding(slot: qkvSlot, index: 0),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "\(delta.headDim)", type: .uint32, index: 1
                    ),
                    SmeltConstantBinding(
                        expression: "1e-6", type: .float32, index: 2
                    ),
                    SmeltConstantBinding(
                        expression: "\(delta.qkvDim)", type: .uint32, index: 3
                    ),
                    SmeltConstantBinding(
                        expression: "\(qkHeads)", type: .uint32, index: 4
                    ),
                ],
                dispatch: .threadgroups(
                    width: 2 * qkHeads, height: batchSize, depth: 1,
                    tgWidth: qkNormThreads, tgHeight: 1, tgDepth: 1
                ),
                comment: "L\(layerIndex) RMS scale Q+K (prefill)",
                dynamicGridH: .seqLen
            ))
            emitter.recordTraceMarker(
                label: "L\(layerIndex).delta_conv",
                bufferSlot: qkvSlot
            )

            _ = try emitter.emit(SmeltDispatch(
                pipeline: .deltanetRecurrenceMlxPrefillD128H16,
                buffers: [
                    SmeltBufferBinding(slot: recStateSlot, index: 0),
                    SmeltBufferBinding(slot: qkvSlot, index: 1),
                    SmeltBufferBinding(slot: bSlot, index: 2),
                    SmeltBufferBinding(slot: aSlot, index: 3),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: aLogOffset, index: 4
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: dtBiasOffset, index: 5
                    ),
                    SmeltBufferBinding(slot: recOutSlot, index: 6),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "__seqLen__", type: .uint32, index: 7
                    ),
                ],
                dispatch: .threads(
                    width: 32, height: delta.headDim, depth: delta.numHeads,
                    tgWidth: 32, tgHeight: 4, tgDepth: 1
                ),
                comment: "L\(layerIndex) recurrence core (MLX-style tiled prefill)"
            ))
            emitter.recordTraceMarker(
                label: "L\(layerIndex).delta_core",
                bufferSlot: recOutSlot
            )
        } else {
            var convBuffers = [
                    SmeltBufferBinding(slot: convStateSlot, index: 0),
                    SmeltBufferBinding(slot: qkvSlot, index: 1),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: convWeightOffset, index: 2
                    ),
                ]
            if let convHistorySlot, checkpointRecurrentState {
                convBuffers.append(SmeltBufferBinding(
                    slot: convHistorySlot,
                    index: 6
                ))
            }
            _ = try emitter.emit(SmeltDispatch(
                pipeline: checkpointRecurrentState
                    ? .conv1dUpdateSiluPrefillCheckpoint
                    : .conv1dUpdateSiluPrefill,
                buffers: convBuffers,
                constants: [
                    SmeltConstantBinding(
                        expression: "__seqLen__", type: .uint32, index: 3
                    ),
                    SmeltConstantBinding(
                        expression: "\(delta.qkvDim)", type: .uint32, index: 4
                    ),
                    SmeltConstantBinding(
                        expression: "\(delta.convKernel)", type: .uint32, index: 5
                    ),
                ],
                dispatch: .threads(
                    width: delta.qkvDim, height: 1, depth: 1,
                    tgWidth: 256, tgHeight: 1, tgDepth: 1
                ),
                comment: "L\(layerIndex) QKV conv1d update + SiLU (prefill)"
            ))
            emitter.recordTraceMarker(
                label: "L\(layerIndex).delta_conv_raw",
                bufferSlot: qkvSlot
            )

            _ = try emitter.emit(SmeltDispatch(
                pipeline: .rmsScaleQK,
                buffers: [
                    SmeltBufferBinding(slot: qkvSlot, index: 0),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "\(delta.headDim)", type: .uint32, index: 1
                    ),
                    SmeltConstantBinding(
                        expression: "1e-6", type: .float32, index: 2
                    ),
                    SmeltConstantBinding(
                        expression: "\(delta.qkvDim)", type: .uint32, index: 3
                    ),
                    SmeltConstantBinding(
                        expression: "\(qkHeads)", type: .uint32, index: 4
                    ),
                ],
                dispatch: .threadgroups(
                    width: 2 * qkHeads, height: batchSize, depth: 1,
                    tgWidth: qkNormThreads, tgHeight: 1, tgDepth: 1
                ),
                comment: "L\(layerIndex) RMS scale Q+K (prefill)",
                dynamicGridH: .seqLen
            ))
            emitter.recordTraceMarker(
                label: "L\(layerIndex).delta_conv",
                bufferSlot: qkvSlot
            )

            let useSpecializedRecurrence =
                (useD128H32QK16Specialization
                    || useD128H48QK16Specialization)
                && !checkpointRecurrentState
            var recurrenceBuffers = [
                    SmeltBufferBinding(slot: recStateSlot, index: 0),
                    SmeltBufferBinding(slot: qkvSlot, index: 1),
                    SmeltBufferBinding(slot: bSlot, index: 2),
                    SmeltBufferBinding(slot: aSlot, index: 3),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: aLogOffset, index: 4
                    ),
                    SmeltBufferBinding(
                        slot: weightsSlot, offset: dtBiasOffset, index: 5
                    ),
                    SmeltBufferBinding(slot: recOutSlot, index: 6),
                ]
            if let recHistorySlot, checkpointRecurrentState {
                recurrenceBuffers.append(SmeltBufferBinding(
                    slot: recHistorySlot,
                    index: 11
                ))
            }
            _ = try emitter.emit(SmeltDispatch(
                pipeline: checkpointRecurrentState
                    ? .deltanetRecurrenceMlxPrefillCheckpoint
                    : useD128H32QK16Specialization
                        ? .deltanetRecurrenceMlxPrefillD128H32QK16
                        : useD128H48QK16Specialization
                            ? .deltanetRecurrenceMlxPrefillD128H48QK16
                            : .deltanetRecurrenceMlxPrefill,
                buffers: recurrenceBuffers,
                constants: useSpecializedRecurrence ? [
                    SmeltConstantBinding(
                        expression: "__seqLen__", type: .uint32, index: 7
                    ),
                ] : [
                    SmeltConstantBinding(
                        expression: "\(delta.headDim)", type: .uint32, index: 7
                    ),
                    SmeltConstantBinding(
                        expression: "\(delta.numHeads)", type: .uint32, index: 8
                    ),
                    SmeltConstantBinding(
                        expression: "\(qkHeads)", type: .uint32, index: 9
                    ),
                    SmeltConstantBinding(
                        expression: "__seqLen__", type: .uint32, index: 10
                    ),
                ],
                dispatch: .threads(
                    width: 32, height: delta.headDim, depth: delta.numHeads,
                    tgWidth: 32, tgHeight: 4, tgDepth: 1
                ),
                comment: "L\(layerIndex) recurrence core (MLX-style tiled prefill)"
            ))
            emitter.recordTraceMarker(
                label: "L\(layerIndex).delta_core",
                bufferSlot: recOutSlot
            )
        }

        // Batched gated RMS norm: recOutBuf * sigmoid(zBuf) → recOutBuf
        let normWeightOffset = try requireWeight(
            "\(prefix)_norm_weight", from: weightEntries
        ).offset
        let useSpecializedBatchedGatedNorm = delta.headDim == 128 && config.rmsEps == 1e-6
        _ = try emitter.emit(SmeltDispatch(
            pipeline: useSpecializedBatchedGatedNorm ? .rmsNormGatedD128Batched : .rmsNormGated,
            buffers: [
                SmeltBufferBinding(slot: recOutSlot, index: 0),
                SmeltBufferBinding(slot: zSlot, index: 1),
                SmeltBufferBinding(
                    slot: weightsSlot,
                    offset: normWeightOffset,
                    index: 2
                ),
                SmeltBufferBinding(slot: recOutSlot, index: 3),
            ],
            constants: useSpecializedBatchedGatedNorm ? [
                SmeltConstantBinding(
                    expression: "\(delta.numHeads)", type: .uint32, index: 4
                ),
            ] : [
                SmeltConstantBinding(
                    expression: "\(delta.headDim)", type: .uint32, index: 4
                ),
                SmeltConstantBinding(
                    expression: "\(config.rmsEps)", type: .float32, index: 5
                ),
            ],
            dispatch: .threadgroups(
                width: useSpecializedBatchedGatedNorm ? delta.numHeads : delta.numHeads * batchSize,
                height: useSpecializedBatchedGatedNorm ? batchSize : 1,
                depth: 1,
                tgWidth: useSpecializedBatchedGatedNorm ? 32 : min(delta.headDim, 256),
                tgHeight: 1, tgDepth: 1
            ),
            comment: "L\(layerIndex) gated norm (batched)",
            dynamicGridW: useSpecializedBatchedGatedNorm ? nil : seqLenGrid(delta.numHeads),
            dynamicGridH: useSpecializedBatchedGatedNorm ? .seqLen : nil
        ))
        emitter.recordTraceMarker(
            label: "L\(layerIndex).delta_rec",
            bufferSlot: recOutSlot
        )

        // Batched output projection: recOutBuf → normOutBuf
        let outEntry = try requireWeight(
            "\(prefix)_out_proj_weight", from: weightEntries)
        let deltaOutputBank = config.projectionBank(
            source: .deltaOutput, containing: [.deltaOut])
        let usedOutputActivationView: Bool
        if let view = deltaOutputBank?.activationView,
           deltaOutputBank?.usesActivationView(at: layerIndex) == true,
           plan.projectionActivationPlanesSlot >= 0,
           plan.projectionActivationScalesSlot >= 0,
           let lines = try emitter.emitSignedBitplaneProjectionBankBatchedIfPossible(
               view: view,
               members: [(outEntry, normOutSlot, hidden)],
               weightsSlot: weightsSlot,
               inputSlot: recOutSlot,
               planesSlot: plan.projectionActivationPlanesSlot,
               activationScalesSlot: plan.projectionActivationScalesSlot,
               cols: valueDim,
               batchSize: batchSize,
               producerComment: "L\(layerIndex) CAM delta-output view (batched)",
               projectionComment: "L\(layerIndex) out projection (bit-GEMM B4)"
           ) {
            _ = lines
            usedOutputActivationView = true
        } else {
            usedOutputActivationView = false
        }
        if !usedOutputActivationView {
            try emitBatchedMatmul(
                weightEntry: outEntry,
                weightsSlot: weightsSlot,
                inputSlot: recOutSlot,
                outputSlot: normOutSlot,
                rows: hidden, cols: valueDim, groupSize: groupSize,
                batchSize: batchSize,
                emitter: &emitter,
                comment: "L\(layerIndex) out proj (batched)"
            )
        }
        emitter.recordTraceMarker(
            label: "L\(layerIndex).delta_out",
            bufferSlot: normOutSlot
        )
    }

    private static func emitBatchedProjectionBiasAdd(
        biasEntry: SmeltWeightEntry,
        weightsSlot: Int,
        outputSlot: Int,
        rows: Int,
        batchSize: Int,
        emitter: inout SmeltCodeEmitter,
        comment: String
    ) throws {
        _ = try emitter.emitBatchedBiasAdd(
            inputSlot: outputSlot,
            biasSlot: weightsSlot,
            biasOffset: biasEntry.offset,
            outputSlot: outputSlot,
            rows: rows,
            batchSize: batchSize,
            comment: comment
        )
    }

    // MARK: - Attention prefill layer

    /// Emit one attention layer: batched projections + causal attention.
    private static func emitPrefillAttentionLayer(
        layerIndex: Int,
        attnIndex: Int,
        layerType: SmeltLayerType,
        config: SmeltConfig,
        plan: SmeltBufferPlan,
        weightEntries: [String: SmeltWeightEntry],
        weightsSlot: Int,
        groupSize: Int,
        batchSize: Int,
        emitter: inout SmeltCodeEmitter,
        attnOverride: SmeltAttentionConfig? = nil,
        ropeCosSlotOverride: Int? = nil,
        ropeSinSlotOverride: Int? = nil,
        sharedKVSourceAttnIndex: Int? = nil
    ) throws {
        guard let attn = attnOverride ?? config.attentionConfig(for: layerType) else {
            throw PrefillEmitterError.unsupported(
                "layer type '\(layerType.rawValue)' requires attention config for Metal prefill"
            )
        }
        let hidden = config.hiddenSize
        let prefix = "layers_\(layerIndex)_self_attn"
        let normOutSlot = SmeltFixedSlot.normOutBuf.rawValue

        let qSlot = SmeltFixedSlot.attnQBuf.rawValue
        let kSlot = SmeltFixedSlot.attnKBuf.rawValue
        let vSlot = SmeltFixedSlot.attnVBuf.rawValue
        let outSlot = SmeltFixedSlot.attnOutBuf.rawValue
        let keyCacheSourceAttnIndex = sharedKVSourceAttnIndex ?? attnIndex
        let attentionHasOwnKV = sharedKVSourceAttnIndex == nil && !attn.externalKV
        let qEntry = try requireWeight(
            SmeltKernelConsumerNaming.qProjWeight(layerIndex: layerIndex),
            from: weightEntries
        )
        let kEntry = attentionHasOwnKV
            ? try requireWeight(
                SmeltKernelConsumerNaming.kProjWeight(layerIndex: layerIndex),
                from: weightEntries
            )
            : nil
        let vEntry = attentionHasOwnKV
            ? try requireWeight(
                SmeltKernelConsumerNaming.vProjWeight(layerIndex: layerIndex),
                from: weightEntries
            )
            : nil
        let attentionInputBank = config.projectionBank(
            source: .attentionInput,
            containing: [.attentionQ, .attentionK, .attentionV]
        )
        let usedInputActivationView: Bool
        if !attn.qkvBias,
           let kEntry,
           let vEntry,
           let view = attentionInputBank?.activationView,
           attentionInputBank?.usesActivationView(at: layerIndex) == true,
           plan.projectionActivationPlanesSlot >= 0,
           plan.projectionActivationScalesSlot >= 0,
           let lines = try emitter.emitSignedBitplaneProjectionBankBatchedIfPossible(
               view: view,
               members: [
                   (qEntry, qSlot, attn.qProjDim),
                   (kEntry, kSlot, attn.kProjDim),
                   (vEntry, vSlot, attn.vProjDim),
               ],
               weightsSlot: weightsSlot,
               inputSlot: normOutSlot,
               planesSlot: plan.projectionActivationPlanesSlot,
               activationScalesSlot: plan.projectionActivationScalesSlot,
               cols: hidden,
               batchSize: batchSize,
               producerComment: "L\(layerIndex) CAM attention-input view (batched)",
               projectionComment: "L\(layerIndex) CAM attention-input projection (bit-GEMM B4)"
           ) {
            _ = lines
            usedInputActivationView = true
        } else {
            usedInputActivationView = false
        }

        // Batched Q projection
        if !usedInputActivationView {
            try emitBatchedMatmul(
                weightEntry: qEntry,
                weightsSlot: weightsSlot,
                inputSlot: normOutSlot,
                outputSlot: qSlot,
                rows: attn.qProjDim, cols: hidden, groupSize: groupSize,
                batchSize: batchSize,
                emitter: &emitter,
                comment: "L\(layerIndex) Q proj (batched)"
            )
        }
        if attn.qkvBias && !usedInputActivationView {
            try emitBatchedProjectionBiasAdd(
                biasEntry: try requireWeight("\(prefix)_q_proj_bias", from: weightEntries),
                weightsSlot: weightsSlot,
                outputSlot: qSlot,
                rows: attn.qProjDim,
                batchSize: batchSize,
                emitter: &emitter,
                comment: "L\(layerIndex) Q projection bias (batched)"
            )
        }
        emitter.recordTraceMarker(
            label: "L\(layerIndex).q_proj",
            bufferSlot: qSlot
        )

        if attentionHasOwnKV, let kEntry, let vEntry {
            if usedInputActivationView {
                emitter.recordTraceMarker(
                    label: "L\(layerIndex).k_proj",
                    bufferSlot: kSlot
                )
                emitter.recordTraceMarker(
                    label: "L\(layerIndex).v_proj",
                    bufferSlot: vSlot
                )
            } else {
            // Family via the ONE gateway: fused batched K+V iff both affine_u4.
            let kvFamily = emitter.bothFusedFamily(kEntry, vEntry, shape: .gemm, fusion: .dualMatvec)
            if attn.kProjDim == attn.vProjDim, kvFamily == .affineU4 {
                try emitUnrolledFusedDualAffineMatvec(
                    weightEntry1: kEntry,
                    weightEntry2: vEntry,
                    weightsSlot: weightsSlot,
                    inputSlot: normOutSlot,
                    outputSlot1: kSlot,
                    outputSlot2: vSlot,
                    rows: attn.kProjDim,
                    cols: hidden,
                    groupSize: groupSize,
                    batchSize: batchSize,
                    emitter: &emitter,
                    comment: "L\(layerIndex) K+V proj (batched)"
                )
                if attn.qkvBias {
                    try emitBatchedProjectionBiasAdd(
                        biasEntry: try requireWeight("\(prefix)_k_proj_bias", from: weightEntries),
                        weightsSlot: weightsSlot,
                        outputSlot: kSlot,
                        rows: attn.kProjDim,
                        batchSize: batchSize,
                        emitter: &emitter,
                        comment: "L\(layerIndex) K projection bias (batched)"
                    )
                    try emitBatchedProjectionBiasAdd(
                        biasEntry: try requireWeight("\(prefix)_v_proj_bias", from: weightEntries),
                        weightsSlot: weightsSlot,
                        outputSlot: vSlot,
                        rows: attn.vProjDim,
                        batchSize: batchSize,
                        emitter: &emitter,
                        comment: "L\(layerIndex) V projection bias (batched)"
                    )
                }
                emitter.recordTraceMarker(
                    label: "L\(layerIndex).k_proj",
                    bufferSlot: kSlot
                )
                emitter.recordTraceMarker(
                    label: "L\(layerIndex).v_proj",
                    bufferSlot: vSlot
                )
            } else {
                // Batched K projection
                try emitBatchedMatmul(
                    weightEntry: kEntry,
                    weightsSlot: weightsSlot,
                    inputSlot: normOutSlot,
                    outputSlot: kSlot,
                    rows: attn.kProjDim, cols: hidden, groupSize: groupSize,
                    batchSize: batchSize,
                    emitter: &emitter,
                    comment: "L\(layerIndex) K proj (batched)"
                )
                if attn.qkvBias {
                    try emitBatchedProjectionBiasAdd(
                        biasEntry: try requireWeight("\(prefix)_k_proj_bias", from: weightEntries),
                        weightsSlot: weightsSlot,
                        outputSlot: kSlot,
                        rows: attn.kProjDim,
                        batchSize: batchSize,
                        emitter: &emitter,
                        comment: "L\(layerIndex) K projection bias (batched)"
                    )
                }
                emitter.recordTraceMarker(
                    label: "L\(layerIndex).k_proj",
                    bufferSlot: kSlot
                )

                // Batched V projection
                try emitBatchedMatmul(
                    weightEntry: vEntry,
                    weightsSlot: weightsSlot,
                    inputSlot: normOutSlot,
                    outputSlot: vSlot,
                    rows: attn.vProjDim, cols: hidden, groupSize: groupSize,
                    batchSize: batchSize,
                    emitter: &emitter,
                    comment: "L\(layerIndex) V proj (batched)"
                )
                if attn.qkvBias {
                    try emitBatchedProjectionBiasAdd(
                        biasEntry: try requireWeight("\(prefix)_v_proj_bias", from: weightEntries),
                        weightsSlot: weightsSlot,
                        outputSlot: vSlot,
                        rows: attn.vProjDim,
                        batchSize: batchSize,
                        emitter: &emitter,
                        comment: "L\(layerIndex) V projection bias (batched)"
                    )
                }
                emitter.recordTraceMarker(
                    label: "L\(layerIndex).v_proj",
                    bufferSlot: vSlot
                )
            }
            }
        } else if attn.externalKV {
            // External-KV layers consume target-supplied K/V via the
            // attnKBuf / attnVBuf slots; no projection needed here.
            _ = emitter.emitComment(
                "L\(layerIndex) external KV: K/V supplied by target package"
            )
        } else {
            _ = emitter.emitComment(
                "L\(layerIndex) reuse KV cache from attention layer \(keyCacheSourceAttnIndex)"
            )
        }

        // Gate split (if gated Q)
        if attn.gatedQ {
            _ = try emitter.emit(SmeltDispatch(
                pipeline: .gateSplit,
                buffers: [
                    SmeltBufferBinding(slot: qSlot, index: 0),
                    SmeltBufferBinding(slot: outSlot, index: 1),
                    SmeltBufferBinding(
                        slot: SmeltFixedSlot.attnGateBuf.rawValue, index: 2
                    ),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: seqLenExpr(attn.qHeads), type: .uint32, index: 3
                    ),
                    SmeltConstantBinding(
                        expression: "\(attn.headDim)", type: .uint32, index: 4
                    ),
                ],
                dispatch: .threads(
                    width: attn.headDim,
                    height: attn.qHeads * batchSize,
                    depth: 1,
                    tgWidth: min(attn.headDim, 256),
                    tgHeight: 1, tgDepth: 1
                ),
                comment: "L\(layerIndex) gate split (batched)",
                dynamicGridH: seqLenGrid(attn.qHeads)
            ))
            emitter.recordTraceMarker(
                label: "L\(layerIndex).attn_gate_raw",
                bufferSlot: SmeltFixedSlot.attnGateBuf.rawValue
            )
        }

        // Q source: after gate split (if gated) or direct Q buffer
        let qNormSrc = attn.gatedQ ? outSlot : qSlot

        if attn.qkNorm {
            let qNormWeight = try requireWeight(
                "\(prefix)_q_norm_weight", from: weightEntries
            )
            let qkNormPipeline: SmeltPipeline =
                switch attn.qkNormMode {
                case .onePlusWeight: .perHeadRmsNorm1PWBatched
                case .weight: .perHeadRmsNormBatched
                }
            let qkNormThreadgroupWidth = attn.qkNormMode == .weight
                ? AttentionPlugin.directWeightNormThreadgroupWidth(headDim: attn.headDim)
                : min(attn.headDim, 256)

            _ = try emitter.emit(SmeltDispatch(
                pipeline: qkNormPipeline,
                buffers: [
                    SmeltBufferBinding(slot: qNormSrc, index: 0),
                    SmeltBufferBinding(slot: weightsSlot, offset: qNormWeight.offset, index: 1),
                    SmeltBufferBinding(slot: qNormSrc, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "\(attn.qHeads)", type: .uint32, index: 3
                    ),
                    SmeltConstantBinding(
                        expression: "\(attn.headDim)", type: .uint32, index: 4
                    ),
                    SmeltConstantBinding(
                        expression: "\(config.rmsEps)", type: .float32, index: 5
                    ),
                ],
                dispatch: .threadgroups(
                    width: attn.qHeads, height: batchSize, depth: 1,
                    tgWidth: qkNormThreadgroupWidth, tgHeight: 1, tgDepth: 1
                ),
                comment: "L\(layerIndex) Q per-head RMS norm (batched)",
                dynamicGridH: .seqLen
            ))
            emitter.recordTraceMarker(
                label: "L\(layerIndex).q_post_norm",
                bufferSlot: qNormSrc
            )

            if sharedKVSourceAttnIndex == nil, !attn.externalKV {
                let kNormWeight = try requireWeight(
                    "\(prefix)_k_norm_weight", from: weightEntries
                )
                _ = try emitter.emit(SmeltDispatch(
                    pipeline: qkNormPipeline,
                    buffers: [
                        SmeltBufferBinding(slot: kSlot, index: 0),
                        SmeltBufferBinding(slot: weightsSlot, offset: kNormWeight.offset, index: 1),
                        SmeltBufferBinding(slot: kSlot, index: 2),
                    ],
                    constants: [
                        SmeltConstantBinding(
                            expression: "\(attn.kvHeads)", type: .uint32, index: 3
                        ),
                        SmeltConstantBinding(
                            expression: "\(attn.headDim)", type: .uint32, index: 4
                        ),
                        SmeltConstantBinding(
                            expression: "\(config.rmsEps)", type: .float32, index: 5
                        ),
                    ],
                    dispatch: .threadgroups(
                        width: attn.kvHeads, height: batchSize, depth: 1,
                        tgWidth: qkNormThreadgroupWidth, tgHeight: 1, tgDepth: 1
                    ),
                    comment: "L\(layerIndex) K per-head RMS norm (batched)",
                    dynamicGridH: .seqLen
                ))
                emitter.recordTraceMarker(
                    label: "L\(layerIndex).k_post_norm",
                    bufferSlot: kSlot
                )
            }
        }
        if attn.vNorm, sharedKVSourceAttnIndex == nil, !attn.externalKV {
            _ = try emitter.emit(SmeltDispatch(
                pipeline: .perHeadRmsNormNoScaleBatched,
                buffers: [
                    SmeltBufferBinding(slot: vSlot, index: 0),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "\(attn.kvHeads)", type: .uint32, index: 1
                    ),
                    SmeltConstantBinding(
                        expression: "\(attn.headDim)", type: .uint32, index: 2
                    ),
                    SmeltConstantBinding(
                        expression: "\(config.rmsEps)", type: .float32, index: 3
                    ),
                ],
                dispatch: .threadgroups(
                    width: attn.kvHeads, height: batchSize, depth: 1,
                    tgWidth: min(attn.headDim, 256), tgHeight: 1, tgDepth: 1
                ),
                comment: "L\(layerIndex) V per-head RMS norm (batched)",
                dynamicGridH: .seqLen
            ))
            emitter.recordTraceMarker(
                label: "L\(layerIndex).v_post_norm",
                bufferSlot: vSlot
            )
        }

        // Fused RoPE + KV cache write (1 dispatch replaces 4*B unrolled)
        let keyCacheSlot = plan.keyCacheBaseSlot + keyCacheSourceAttnIndex
        let valCacheSlot = plan.valCacheBaseSlot + keyCacheSourceAttnIndex
        let ropeDim = attn.effectiveRopeDim(default: config.ropeDim)
        let ropeLayout = AttentionPlugin.ropeLayoutConstant(
            config: config,
            attn: attn,
            headDim: attn.headDim,
            ropeDim: ropeDim
        )
        let ropeCosSlot = ropeCosSlotOverride ?? plan.ropeCosSlot
        let ropeSinSlot = ropeSinSlotOverride ?? plan.ropeSinSlot

        // kvHeadsForRopeKV=0 tells the fused kernel to skip the K
        // RoPE + KV-cache write half; we set it to 0 for shared-KV
        // layers (target writes via the source layer) and for
        // external-KV layers (target package writes from outside).
        let kvHeadsForRopeKV =
            (sharedKVSourceAttnIndex == nil && !attn.externalKV)
            ? attn.kvHeads : 0
        let useAnalyticRope = attn.ropeScaling == nil
        let ropePipeline: SmeltPipeline = useAnalyticRope
            ? .ropeAndKvCachePrefillAnalytic
            : .ropeAndKvCachePrefill
        let ropeBuffers: [SmeltBufferBinding] = useAnalyticRope
            ? [
                SmeltBufferBinding(slot: qNormSrc, index: 0),
                SmeltBufferBinding(slot: kSlot, index: 1),
                SmeltBufferBinding(slot: vSlot, index: 2),
                SmeltBufferBinding(slot: keyCacheSlot, index: 3),
                SmeltBufferBinding(slot: valCacheSlot, index: 4),
            ]
            : [
                SmeltBufferBinding(slot: qNormSrc, index: 0),
                SmeltBufferBinding(slot: kSlot, index: 1),
                SmeltBufferBinding(slot: vSlot, index: 2),
                SmeltBufferBinding(slot: ropeCosSlot, index: 3),
                SmeltBufferBinding(slot: ropeSinSlot, index: 4),
                SmeltBufferBinding(slot: keyCacheSlot, index: 5),
                SmeltBufferBinding(slot: valCacheSlot, index: 6),
            ]
        let ropeConstants: [SmeltConstantBinding] = useAnalyticRope
            ? [
                SmeltConstantBinding(
                    expression: "\(attn.headDim | (ropeDim << 16))",
                    type: .uint32, index: 5
                ),
                SmeltConstantBinding(
                    expression: "\(attn.qHeads | (kvHeadsForRopeKV << 16))",
                    type: .uint32, index: 6
                ),
                SmeltConstantBinding(
                    expression: "__seqLen__", type: .uint32, index: 7
                ),
                SmeltConstantBinding(
                    expression: "__startPos__", type: .uint32, index: 8
                ),
                SmeltConstantBinding(
                    expression: "cacheSeqCapacity", type: .uint32, index: 9
                ),
                SmeltConstantBinding(
                    expression: "\(ropeLayout)", type: .uint32, index: 10
                ),
                SmeltConstantBinding(
                    expression: "\(log2(attn.ropeTheta))", type: .float32, index: 11
                ),
            ]
            : [
                SmeltConstantBinding(
                    expression: "\(attn.headDim)", type: .uint32, index: 7
                ),
                SmeltConstantBinding(
                    expression: "\(ropeDim)", type: .uint32, index: 8
                ),
                SmeltConstantBinding(
                    expression: "\(attn.qHeads)", type: .uint32, index: 9
                ),
                SmeltConstantBinding(
                    expression: "\(kvHeadsForRopeKV)", type: .uint32, index: 10
                ),
                SmeltConstantBinding(
                    expression: "__seqLen__", type: .uint32, index: 11
                ),
                SmeltConstantBinding(
                    expression: "__startPos__", type: .uint32, index: 12
                ),
                SmeltConstantBinding(
                    expression: "cacheSeqCapacity", type: .uint32, index: 13
                ),
                SmeltConstantBinding(
                    expression: "\(ropeLayout)", type: .uint32, index: 14
                ),
            ]
        _ = try emitter.emit(SmeltDispatch(
            pipeline: ropePipeline,
            buffers: ropeBuffers,
            constants: ropeConstants,
            dispatch: .threadgroups(
                width: batchSize,
                height: max(attn.qHeads, kvHeadsForRopeKV),
                depth: 1,
                tgWidth: min(attn.headDim, 256),
                tgHeight: 1, tgDepth: 1
            ),
            comment: kvHeadsForRopeKV == 0
                ? (attn.externalKV
                    ? "L\(layerIndex) fused Q RoPE (external KV)"
                    : "L\(layerIndex) fused Q RoPE (shared KV)")
                : "L\(layerIndex) fused RoPE + KV cache",
            dynamicGridW: .seqLen
        ))
        emitter.recordTraceMarker(
            label: "L\(layerIndex).q_post_rope",
            bufferSlot: qNormSrc
        )
        if sharedKVSourceAttnIndex == nil, !attn.externalKV {
            emitter.recordTraceMarker(
                label: "L\(layerIndex).k_post_rope",
                bufferSlot: kSlot
            )
        }

        // Batched causal attention — layout [B, numHeads, headDim]
        var attentionConstants = [
            SmeltConstantBinding(
                expression: "\(attn.headDim)", type: .uint32, index: 4
            ),
            SmeltConstantBinding(
                expression: "__seqLen__", type: .uint32, index: 5
            ),
            SmeltConstantBinding(
                expression: "__startPos__", type: .uint32, index: 6
            ),
            SmeltConstantBinding(
                expression: "cacheSeqCapacity", type: .uint32, index: 7
            ),
            SmeltConstantBinding(
                expression: "\(attn.kvHeads)", type: .uint32, index: 8
            ),
            SmeltConstantBinding(
                expression: "\(attn.effectiveScoreScale(blockTopology: config.blockTopology))", type: .float32, index: 9
            ),
            SmeltConstantBinding(
                expression: "\(attn.slidingWindow)", type: .uint32, index: 10
            ),
        ]
        if let softcap = config.attnLogitCap {
            attentionConstants.append(
                SmeltConstantBinding(
                    expression: "\(softcap)", type: .float32, index: 11
                )
            )
        }
        let useD256GQA4AttentionPrefillGeometry =
            attn.headDim == 256
            && attn.qHeads / attn.kvHeads == 4
            && ((attn.qHeads == 8 && attn.kvHeads == 2)
                || (attn.qHeads == 16 && attn.kvHeads == 4))
        let useD64H32KV8AttentionPrefillGeometry =
            config.attnLogitCap == nil
            && attn.slidingWindow == 0
            && attn.headDim == 64
            && attn.qHeads == 32
            && attn.kvHeads == 8
        let useMlxD256AttentionTopology =
            config.attnLogitCap == nil
            && attn.slidingWindow == 0
            && attn.headDim == 256
            && attn.kvHeads > 0
            && attn.qHeads.isMultiple(of: attn.kvHeads)
        let mlxVectorMaxQueryLength = Self.mlxVectorAttentionMaxQueryLength(
            headDim: attn.headDim,
            gqaRatio: attn.qHeads / max(attn.kvHeads, 1)
        )
        let attentionOutSlot = qNormSrc == outSlot ? qSlot : outSlot
        let attentionBuffers = [
            SmeltBufferBinding(slot: qNormSrc, index: 0),
            SmeltBufferBinding(slot: keyCacheSlot, index: 1),
            SmeltBufferBinding(slot: valCacheSlot, index: 2),
            SmeltBufferBinding(slot: attentionOutSlot, index: 3),
        ]
        if useMlxD256AttentionTopology {
            if let mlxVectorMaxQueryLength {
                _ = try emitter.emit(SmeltDispatch(
                    pipeline: .attentionPrefillSDPAVectorD256,
                    buffers: attentionBuffers,
                    constants: attentionConstants,
                    dispatch: .threadgroups(
                        width: attn.qHeads, height: batchSize, depth: 1,
                        tgWidth: 1024, tgHeight: 1, tgDepth: 1
                    ),
                    comment: "L\(layerIndex) MLX vector attention topology (batched)",
                    dynamicGridH: .seqLen,
                    maxSeqLenExclusive: mlxVectorMaxQueryLength + 1
                ))
            }
            _ = try emitter.emit(SmeltDispatch(
                pipeline: .attentionPrefillMLXFallbackD256,
                buffers: attentionBuffers,
                constants: attentionConstants,
                dispatch: .threadgroups(
                    width: attn.qHeads, height: batchSize, depth: 1,
                    tgWidth: 1024, tgHeight: 1, tgDepth: 1
                ),
                comment: "L\(layerIndex) MLX matmul-softmax-matmul attention topology (batched)",
                dynamicGridH: .seqLen,
                minSeqLen: mlxVectorMaxQueryLength.map { $0 + 1 }
            ))
        } else {
            _ = try emitter.emit(SmeltDispatch(
                pipeline: config.attnLogitCap == nil
                    ? .attentionPrefill
                    : .attentionPrefillSoftcap,
                buffers: attentionBuffers,
                constants: attentionConstants,
                dispatch: .threadgroups(
                    width: attn.qHeads, height: batchSize, depth: 1,
                    tgWidth: (useD256GQA4AttentionPrefillGeometry
                        || useD64H32KV8AttentionPrefillGeometry)
                        ? 64
                        : 256,
                    tgHeight: 1, tgDepth: 1
                ),
                comment: config.attnLogitCap == nil
                    ? "L\(layerIndex) causal attention (batched)"
                    : "L\(layerIndex) causal attention with score softcapping (batched)",
                dynamicGridH: .seqLen
            ))
        }
        emitter.recordTraceMarker(
            label: "L\(layerIndex).attn_raw",
            bufferSlot: attentionOutSlot
        )

        // Gate multiply (if gated Q)
        if attn.gatedQ {
            let gateSlot = SmeltFixedSlot.attnGateBuf.rawValue
            let totalElems = attn.qHeads * attn.headDim * batchSize
            _ = try emitter.emit(SmeltDispatch(
                pipeline: .sigmoidMul,
                buffers: [
                    SmeltBufferBinding(slot: attentionOutSlot, index: 0),
                    SmeltBufferBinding(slot: gateSlot, index: 1),
                    SmeltBufferBinding(slot: outSlot, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: seqLenExpr(attn.qHeads * attn.headDim),
                        type: .uint32,
                        index: 3
                    ),
                ],
                dispatch: .threads(
                    width: totalElems, height: 1, depth: 1,
                    tgWidth: min(totalElems, 1024), tgHeight: 1, tgDepth: 1
                ),
                comment: "L\(layerIndex) sigmoid gate multiply (batched)",
                dynamicGridW: seqLenGrid(attn.qHeads * attn.headDim)
            ))
        }
        let attentionContextSlot = attn.gatedQ ? outSlot : attentionOutSlot
        emitter.recordTraceMarker(
            label: "L\(layerIndex).attn_ctx",
            bufferSlot: attentionContextSlot
        )

        // Output projection (batched matmul)
        let outputEntry = try requireWeight(
            SmeltKernelConsumerNaming.oProjWeight(layerIndex: layerIndex),
            from: weightEntries
        )
        let attentionOutputBank = config.projectionBank(
            source: .attentionOutput, containing: [.attentionOut])
        let usedOutputActivationView: Bool
        if let view = attentionOutputBank?.activationView,
           attentionOutputBank?.usesActivationView(at: layerIndex) == true,
           plan.projectionActivationPlanesSlot >= 0,
           plan.projectionActivationScalesSlot >= 0,
           let lines = try emitter.emitSignedBitplaneProjectionBankBatchedIfPossible(
               view: view,
               members: [(outputEntry, normOutSlot, hidden)],
               weightsSlot: weightsSlot,
               inputSlot: attentionContextSlot,
               planesSlot: plan.projectionActivationPlanesSlot,
               activationScalesSlot: plan.projectionActivationScalesSlot,
               cols: attn.qHeads * attn.headDim,
               batchSize: batchSize,
               producerComment: "L\(layerIndex) CAM attention-output view (batched)",
               projectionComment: "L\(layerIndex) O projection (bit-GEMM B4)"
           ) {
            _ = lines
            usedOutputActivationView = true
        } else {
            usedOutputActivationView = false
        }
        if !usedOutputActivationView {
            try emitBatchedMatmul(
                weightEntry: outputEntry,
                weightsSlot: weightsSlot,
                inputSlot: attentionContextSlot,
                outputSlot: normOutSlot,
                rows: hidden, cols: attn.qHeads * attn.headDim,
                groupSize: groupSize,
                batchSize: batchSize,
                emitter: &emitter,
                comment: "L\(layerIndex) O proj (batched)"
            )
        }
        emitter.recordTraceMarker(
            label: "L\(layerIndex).attn_out",
            bufferSlot: SmeltFixedSlot.normOutBuf.rawValue
        )
    }

    private static func resolveRoPESlots(
        for attn: SmeltAttentionConfig,
        layerType: SmeltLayerType,
        config: SmeltConfig,
        plan: SmeltBufferPlan
    ) throws -> (cos: Int, sin: Int) {
        let effectiveDim = attn.effectiveRopeDim(default: config.ropeDim)
        let freqDim: Int?
        if layerType != .attention, effectiveDim < attn.headDim {
            freqDim = attn.headDim
        } else {
            freqDim = nil
        }
        let params = SmeltRoPEParams(
            theta: attn.ropeTheta,
            dim: effectiveDim,
            freqDim: freqDim,
            scaling: attn.ropeScaling,
            layout: attn.effectiveRoPETableLayout(blockTopology: config.blockTopology)
        )
        if let pair = plan.ropeTablePairs.first(where: { $0.params == params }) {
            return (pair.cosSlot, pair.sinSlot)
        }
        if plan.ropeTablePairs.count <= 1 {
            return (plan.ropeCosSlot, plan.ropeSinSlot)
        }
        let freqDimText = freqDim.map(String.init) ?? "nil"
        throw PrefillEmitterError.unsupported(
            "No RoPE table pair allocated for theta=\(attn.ropeTheta), dim=\(effectiveDim), freq_dim=\(freqDimText)"
        )
    }

    private static func requireAttentionConfig(
        for layerType: SmeltLayerType,
        from config: SmeltConfig
    ) throws -> SmeltAttentionConfig {
        guard let attn = config.attentionConfig(for: layerType) else {
            throw PrefillEmitterError.unsupported(
                "layer type '\(layerType.rawValue)' requires attention config for Metal prefill"
            )
        }
        return attn
    }
}

// MARK: - Errors

public enum PrefillEmitterError: Error, CustomStringConvertible {
    case notConfigured
    case unsupported(String)

    public var description: String {
        switch self {
        case .notConfigured:
            return "Metal prefill not configured (engine != \"metal\")"
        case .unsupported(let detail):
            return "Unsupported prefill config: \(detail)"
        }
    }
}
