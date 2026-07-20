// AttentionPlugin — Emits dispatch code for one attention layer.
//
// Uses SmeltCodeEmitter to generate ~14 Metal dispatch calls per layer:
// Q/K/V projections, gate_split, 2x per-head RMS norm, 2x RoPE,
// 2x KV cache update, attention_decode, sigmoid_mul (fused), O projection.
//
// RoPE and KV cache dispatches are position-dependent — they emit code
// that references the `position` variable in the generated function scope.
// This means some constant expressions are dynamic (e.g. "position + 1").

import Foundation
import SmeltSchema

/// Emits Swift dispatch code for one attention layer.
public struct AttentionPlugin {
    // The compact short-context kernel is bit-exact for the one-key identity
    // case. Once there is more than one key, use the MLX-topology SDPA brick:
    // the compact kernel's dot/softmax reduction order can move fp16 outputs
    // by one ULP even when Q, K, and V are all exact.
    private static let mlxVectorDirectIdentityMaxSeqLenExclusive = 2
    private static let mlxVectorTwoPassMinSeqLen = 1024

    /// Emit dispatch code for a single attention layer.
    ///
    /// - Parameters:
    ///   - layerIndex: Global layer index in the expanded graph.
    ///   - attnIndex: Attention-local index used to select its KV cache.
    ///   - config: Model config (must have attention != nil).
    ///   - plan: Buffer plan for slot indices.
    ///   - weightEntries: Weight layout entries keyed by name.
    ///   - weightsSlot: Slot index of the monolithic weight buffer.
    ///   - emitter: Code emitter (mutating — tracks dispatch counter).
    /// - Returns: Array of Swift source lines.
    public static func emitLayer(
        layerIndex: Int,
        attnIndex: Int,
        layerType: SmeltLayerType = .attention,
        config: SmeltConfig,
        plan: SmeltBufferPlan,
        weightEntries: [String: SmeltWeightEntry],
        weightsSlot: Int,
        groupSize: Int,
        kernelPlan: SmeltKernelPlan? = nil,
        emitter: inout SmeltCodeEmitter,
        attnOverride: SmeltAttentionConfig? = nil,
        ropeCosSlotOverride: Int? = nil,
        ropeSinSlotOverride: Int? = nil,
        sharedKVSourceAttnIndex: Int? = nil,
        traceMode: SmeltTraceMode = .full
    ) throws -> [String] {
        guard let attn = attnOverride ?? config.attentionConfig(for: layerType) else {
            throw SmeltEmitError.missingConfig(detail: "Attention plugin requires attention config")
        }

        let prefix = "layers_\(layerIndex)_self_attn"
        let hidden = config.hiddenSize
        let nQ = attn.qHeads
        let nKV = attn.kvHeads
        let headDim = attn.headDim
        let ropeDim = attn.effectiveRopeDim(default: config.ropeDim)
        let ropeLayout = ropeLayoutConstant(
            config: config,
            attn: attn,
            headDim: headDim,
            ropeDim: ropeDim
        )
        let fp16 = 2

        // Slot indices
        let attnQSlot = SmeltFixedSlot.attnQBuf.rawValue
        let attnKSlot = SmeltFixedSlot.attnKBuf.rawValue
        let attnVSlot = SmeltFixedSlot.attnVBuf.rawValue
        let attnOutSlot = SmeltFixedSlot.attnOutBuf.rawValue
        let attnGateSlot = SmeltFixedSlot.attnGateBuf.rawValue
        let attnMaskSlot = SmeltFixedSlot.attnMaskBuf.rawValue
        let normOutSlot = SmeltFixedSlot.normOutBuf.rawValue
        let sharedKVSourceAttnIndex = sharedKVSourceAttnIndex
        let keyCacheSourceAttnIndex = sharedKVSourceAttnIndex ?? attnIndex
        let keyCacheSlot = plan.keyCacheBaseSlot + keyCacheSourceAttnIndex
        let valCacheSlot = plan.valCacheBaseSlot + keyCacheSourceAttnIndex
        let ropeCosSlot = ropeCosSlotOverride ?? plan.ropeCosSlot
        let ropeSinSlot = ropeSinSlotOverride ?? plan.ropeSinSlot
        let attentionHasOwnKV = sharedKVSourceAttnIndex == nil && !attn.externalKV

        func plannedRoute(
            _ kind: SmeltKernelConsumerKind,
            groupSize entryGroupSize: Int
        ) -> SmeltPlannedKernelRoute? {
            kernelPlan?.route(
                kind: kind,
                context: SmeltKernelLayerConsumerContext(
                    config: config,
                    layerIndex: layerIndex,
                    groupSize: entryGroupSize,
                    attention: attn,
                    attentionHasOwnKV: attentionHasOwnKV
                )
            )
        }

        // Use requireWeight() for safe weight lookup

        var lines: [String] = []

        // --- 1. Q projection: normOutBuf → attention query buffer ---
        let qProjRows = attn.gatedQ ? nQ * headDim * 2 : nQ * headDim
        let qProjOutputSlot = attn.gatedQ ? attnQSlot : attnOutSlot
        let qProjName = SmeltKernelConsumerNaming.qProjWeight(layerIndex: layerIndex)
        let qProjEntry = try requireWeight(qProjName, from: weightEntries)
        var packedAttentionProjectionLines: [String]?
        if attentionHasOwnKV,
           !attn.qkvBias,
           let bank = config.projectionBank(
               source: .attentionInput,
               containing: [.attentionQ, .attentionK, .attentionV]
           ) {
            let kEntry = try requireWeight(
                SmeltKernelConsumerNaming.kProjWeight(layerIndex: layerIndex),
                from: weightEntries
            )
            let vEntry = try requireWeight(
                SmeltKernelConsumerNaming.vProjWeight(layerIndex: layerIndex),
                from: weightEntries
            )
            let kvRows = nKV * headDim
            let membersByEndpoint: [SmeltCAMIR.ProjectionEndpoint: (
                entry: SmeltWeightEntry, outputSlot: Int, rows: Int
            )] = [
                .attentionQ: (qProjEntry, qProjOutputSlot, qProjRows),
                .attentionK: (kEntry, attnKSlot, kvRows),
                .attentionV: (vEntry, attnVSlot, kvRows),
            ]
            let bankMembers = bank.outputs.compactMap { membersByEndpoint[$0] }
            if let activationView = bank.activationView,
               bank.usesActivationView(at: layerIndex),
               plan.projectionActivationPlanesSlot >= 0,
               plan.projectionActivationScalesSlot >= 0 {
                packedAttentionProjectionLines = try emitter
                    .emitSignedBitplaneProjectionBankIfPossible(
                        view: activationView,
                        members: bankMembers,
                        weightsSlot: weightsSlot,
                        planesSlot: plan.projectionActivationPlanesSlot,
                        activationScalesSlot: plan.projectionActivationScalesSlot,
                        cols: hidden,
                        comment: "CAM attention-input low-bit projection bank"
                    )
            }
            if packedAttentionProjectionLines == nil {
                packedAttentionProjectionLines = try emitter
                    .emitSignedPackedProjectionBankIfPossible(
                        members: bankMembers,
                        weightsSlot: weightsSlot,
                        inputSlot: normOutSlot,
                        cols: hidden,
                        comment: "CAM attention-input projection bank"
                    )
            }
        }
        if let packedAttentionProjectionLines {
            lines += packedAttentionProjectionLines
            emitter.recordTraceMarker(
                label: "L\(layerIndex).q_proj",
                bufferSlot: qProjOutputSlot
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).k_proj",
                bufferSlot: SmeltFixedSlot.attnKBuf.rawValue
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).v_proj",
                bufferSlot: SmeltFixedSlot.attnVBuf.rawValue
            )
        } else {
        if attn.qkvBias {
            lines += try emitter.emitMatvec(
                weightEntry: qProjEntry, weightsSlot: weightsSlot,
                inputSlot: normOutSlot, outputSlot: qProjOutputSlot,
                rows: qProjRows, cols: hidden, groupSize: groupSize,
                comment: attn.gatedQ ? "Q projection (includes gate)" : "Q projection",
                plannedKernelRoute: plannedRoute(
                    .qProjBiasDecode,
                    groupSize: qProjEntry.groupSize ?? groupSize
                )
            )
        } else {
            lines += try emitter.emitMatvec(
                weightEntry: qProjEntry, weightsSlot: weightsSlot,
                inputSlot: normOutSlot, outputSlot: qProjOutputSlot,
                rows: qProjRows, cols: hidden, groupSize: groupSize,
                comment: attn.gatedQ ? "Q projection (includes gate)" : "Q projection"
            )
        }
        if attn.qkvBias {
            let qBias = try requireWeight("\(prefix)_q_proj_bias", from: weightEntries)
            lines += try emitter.emitElementwiseAddWithOffsets(
                inputASlot: qProjOutputSlot, inputAOffset: 0,
                inputBSlot: weightsSlot, inputBOffset: qBias.offset,
                outputSlot: qProjOutputSlot, outputOffset: 0,
                count: qProjRows,
                comment: attn.gatedQ ? "Q projection bias (includes gate)" : "Q projection bias"
            )
        }
        emitter.recordTraceMarker(
            label: "L\(layerIndex).q_proj",
            bufferSlot: qProjOutputSlot
        )

        if sharedKVSourceAttnIndex == nil, !attn.externalKV {
            let kProjName = SmeltKernelConsumerNaming.kProjWeight(layerIndex: layerIndex)
            let vProjName = SmeltKernelConsumerNaming.vProjWeight(layerIndex: layerIndex)
            let kProjEntry = try requireWeight(kProjName, from: weightEntries)
            let vProjEntry = try requireWeight(vProjName, from: weightEntries)
            let kvRows = nKV * headDim
            let kGroupSize = kProjEntry.groupSize ?? groupSize
            let vGroupSize = vProjEntry.groupSize ?? groupSize
            if attn.qkvBias,
               kGroupSize == vGroupSize,
               let kvRoute = plannedRoute(.kvProjBiasDecode, groupSize: kGroupSize)
            {
                let kBias = try requireWeight("\(prefix)_k_proj_bias", from: weightEntries)
                let vBias = try requireWeight("\(prefix)_v_proj_bias", from: weightEntries)
                lines += try emitter.emitFusedDualAffineMatvecAdd(
                    firstEntry: kProjEntry,
                    secondEntry: vProjEntry,
                    weightsSlot: weightsSlot,
                    inputSlot: normOutSlot,
                    firstOutputSlot: attnKSlot,
                    secondOutputSlot: attnVSlot,
                    firstResidualOffset: kBias.offset,
                    secondResidualOffset: vBias.offset,
                    rows: kvRows,
                    cols: hidden,
                    groupSize: kGroupSize,
                    comment: "K/V projection",
                    plannedKernelRoute: kvRoute
                )
            } else {
                // --- 2. K projection: normOutBuf → attnKBuf [nKV * headDim] ---
                if attn.qkvBias {
                    lines += try emitter.emitMatvec(
                        weightEntry: kProjEntry, weightsSlot: weightsSlot,
                        inputSlot: normOutSlot, outputSlot: attnKSlot,
                        rows: kvRows, cols: hidden, groupSize: groupSize,
                        comment: "K projection",
                        plannedKernelRoute: plannedRoute(
                            .kProjBiasDecode,
                            groupSize: kGroupSize
                        )
                    )
                } else {
                    lines += try emitter.emitMatvec(
                        weightEntry: kProjEntry, weightsSlot: weightsSlot,
                        inputSlot: normOutSlot, outputSlot: attnKSlot,
                        rows: kvRows, cols: hidden, groupSize: groupSize,
                        comment: "K projection"
                    )
                }
                if attn.qkvBias {
                    let kBias = try requireWeight("\(prefix)_k_proj_bias", from: weightEntries)
                    lines += try emitter.emitElementwiseAddWithOffsets(
                        inputASlot: attnKSlot, inputAOffset: 0,
                        inputBSlot: weightsSlot, inputBOffset: kBias.offset,
                        outputSlot: attnKSlot, outputOffset: 0,
                        count: kvRows,
                        comment: "K projection bias"
                    )
                }

                // --- 3. V projection: normOutBuf → attnVBuf [nKV * headDim] ---
                if attn.qkvBias {
                    lines += try emitter.emitMatvec(
                        weightEntry: vProjEntry, weightsSlot: weightsSlot,
                        inputSlot: normOutSlot, outputSlot: attnVSlot,
                        rows: kvRows, cols: hidden, groupSize: groupSize,
                        comment: "V projection",
                        plannedKernelRoute: plannedRoute(
                            .vProjBiasDecode,
                            groupSize: vGroupSize
                        )
                    )
                } else {
                    lines += try emitter.emitMatvec(
                        weightEntry: vProjEntry, weightsSlot: weightsSlot,
                        inputSlot: normOutSlot, outputSlot: attnVSlot,
                        rows: kvRows, cols: hidden, groupSize: groupSize,
                        comment: "V projection"
                    )
                }
                if attn.qkvBias {
                    let vBias = try requireWeight("\(prefix)_v_proj_bias", from: weightEntries)
                    lines += try emitter.emitElementwiseAddWithOffsets(
                        inputASlot: attnVSlot, inputAOffset: 0,
                        inputBSlot: weightsSlot, inputBOffset: vBias.offset,
                        outputSlot: attnVSlot, outputOffset: 0,
                        count: kvRows,
                        comment: "V projection bias"
                    )
                }
            }

            emitter.recordTraceMarker(
                label: "L\(layerIndex).k_proj",
                bufferSlot: SmeltFixedSlot.attnKBuf.rawValue
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).v_proj",
                bufferSlot: SmeltFixedSlot.attnVBuf.rawValue
            )
        } else if attn.externalKV {
            // External-KV layers consume target-supplied K/V via the
            // attnKBuf / attnVBuf slots; no projection needed here. The
            // runtime is responsible for filling those slots before
            // dispatch.
            lines.append(emitter.emitComment(
                "External KV: K/V supplied by target package, no projection"
            ))
        } else {
            lines.append(emitter.emitComment("Shared KV reuse"))
        }
        }

        // --- 4. Gate split (only when gatedQ) ---
        if attn.gatedQ {
        lines += try emitter.emit(SmeltDispatch(
            pipeline: .gateSplit,
            buffers: [
                SmeltBufferBinding(slot: attnQSlot, index: 0),
                SmeltBufferBinding(slot: attnOutSlot, index: 1),
                SmeltBufferBinding(slot: attnGateSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(nQ)", type: .uint32, index: 3),
                SmeltConstantBinding(expression: "\(headDim)", type: .uint32, index: 4),
            ],
            dispatch: .threads(
                width: headDim, height: nQ, depth: 1,
                tgWidth: min(headDim, 256), tgHeight: 1, tgDepth: 1
            ),
            comment: "Gate split"
        ))
        }

        let ropeByteStride = ropeDim * fp16
        let ropeBaseLog2 = log2(attn.ropeTheta)
        let ropeMathMode = attn.ropeScaling == nil ? 1 : 0

        // --- 5-7. Optional Q/K/V per-head RMS norms ---
        if attn.qkNorm {
            lines += try emitPerHeadNorm(
                emitter: &emitter, dataSlot: attnOutSlot, numHeads: nQ,
                weightName: "\(prefix)_q_norm_weight", weightsSlot: weightsSlot,
                weightEntries: weightEntries, headDim: headDim, eps: config.rmsEps,
                mode: attn.qkNormMode,
                comment: "Q per-head RMS norm"
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).q_post_norm",
                bufferSlot: SmeltFixedSlot.attnOutBuf.rawValue
            )
            if sharedKVSourceAttnIndex == nil, !attn.externalKV {
                lines += try emitPerHeadNorm(
                    emitter: &emitter, dataSlot: attnKSlot, numHeads: nKV,
                    weightName: "\(prefix)_k_norm_weight", weightsSlot: weightsSlot,
                    weightEntries: weightEntries, headDim: headDim, eps: config.rmsEps,
                    mode: attn.qkNormMode,
                    comment: "K per-head RMS norm"
                )
                emitter.recordTraceMarker(
                    label: "L\(layerIndex).k_post_norm",
                    bufferSlot: SmeltFixedSlot.attnKBuf.rawValue
                )
            }
        }
        if attn.vNorm, sharedKVSourceAttnIndex == nil, !attn.externalKV {
            lines += try emitPerHeadNormNoScale(
                emitter: &emitter, dataSlot: attnVSlot, numHeads: nKV,
                headDim: headDim, eps: config.rmsEps,
                comment: "V per-head RMS norm"
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).v_post_norm",
                bufferSlot: SmeltFixedSlot.attnVBuf.rawValue
            )
        }

        // --- 7-8. RoPE on Q and K ---
        // RoPE needs position-dependent cos/sin offsets. The generated code
        // references `position` (Int32 parameter of encodeDecodeStep).
        // We emit raw lines that compute the byte offset at runtime.
        do {
            // 7. RoPE on Q
            lines.append(emitter.emitComment("RoPE on Q"))
            lines += try emitRoPE(
                emitter: &emitter,
                uniqueId: "l\(layerIndex)q",
                dataSlot: attnOutSlot,
                ropeCosSlot: ropeCosSlot,
                ropeSinSlot: ropeSinSlot,
                numHeads: nQ, headDim: headDim, ropeDim: ropeDim,
                ropeByteStride: ropeByteStride,
                ropeLayout: ropeLayout,
                ropeBaseLog2: ropeBaseLog2,
                ropeMathMode: ropeMathMode
            )
            emitter.recordTraceMarker(
                label: "L\(layerIndex).q_post_rope",
                bufferSlot: SmeltFixedSlot.attnOutBuf.rawValue
            )

            if sharedKVSourceAttnIndex == nil, !attn.externalKV {
                if !traceMode.usesStrippedOptimizations {
                    // 8. RoPE on K
                    lines.append(emitter.emitComment("RoPE on K"))
                    lines += try emitRoPE(
                        emitter: &emitter,
                        uniqueId: "l\(layerIndex)k",
                        dataSlot: attnKSlot,
                        ropeCosSlot: ropeCosSlot,
                        ropeSinSlot: ropeSinSlot,
                        numHeads: nKV, headDim: headDim, ropeDim: ropeDim,
                        ropeByteStride: ropeByteStride,
                        ropeLayout: ropeLayout,
                        ropeBaseLog2: ropeBaseLog2,
                        ropeMathMode: ropeMathMode
                    )
                    emitter.recordTraceMarker(
                        label: "L\(layerIndex).k_post_rope",
                        bufferSlot: SmeltFixedSlot.attnKBuf.rawValue
                    )

                    // --- 9. KV cache updates (position-dependent) ---
                    lines += try emitKVCacheUpdate(
                        emitter: &emitter,
                        cacheSlot: keyCacheSlot,
                        newKVSlot: attnKSlot,
                        headDim: headDim, numHeads: nKV,
                        comment: "K cache update"
                    )
                } else {
                    // Stripped-marker packages prefer the fast dispatch graph
                    // over this intermediate K-buffer marker.
                    lines += try emitRopeKVCacheUpdate(
                        emitter: &emitter,
                        cacheSlot: keyCacheSlot,
                        newKVSlot: attnKSlot,
                        ropeCosSlot: ropeCosSlot,
                        ropeSinSlot: ropeSinSlot,
                        ropeByteStride: ropeByteStride,
                        headDim: headDim,
                        ropeDim: ropeDim,
                        numHeads: nKV,
                        ropeLayout: ropeLayout,
                        ropeBaseLog2: ropeBaseLog2,
                        ropeMathMode: ropeMathMode,
                        comment: "Fused RoPE + K cache update"
                    )
                }

                lines += try emitKVCacheUpdate(
                    emitter: &emitter,
                    cacheSlot: valCacheSlot,
                    newKVSlot: attnVSlot,
                    headDim: headDim, numHeads: nKV,
                    comment: "V cache update"
                )
            } else if attn.externalKV {
                // External-KV layers receive target's last-layer K/V
                // pre-RoPE'd and pre-cached. The runtime fills
                // keyCacheSlot / valCacheSlot directly from the
                // target's cache before dispatching this layer, so
                // K-RoPE and the per-token cache-write path are skipped.
                lines.append(emitter.emitComment(
                    "External KV: K/V cache pre-populated by target package"
                ))
            } else if let sharedKVSourceAttnIndex {
                lines.append(
                    emitter.emitComment("Reuse KV cache from attention layer \(sharedKVSourceAttnIndex)")
                )
            }

            // --- 10. Attention decode (position-dependent seqLen) ---
            lines += try emitAttentionDecode(
                emitter: &emitter,
                querySlot: attnOutSlot,
                keyCacheSlot: keyCacheSlot,
                valCacheSlot: valCacheSlot,
                maskSlot: attnMaskSlot,
                outputSlot: attnOutSlot,
                hidden: hidden,
                nQ: nQ, nKV: nKV, headDim: headDim,
                gqaRatio: attn.gqaRatio,
                slidingWindow: attn.slidingWindow,
                scale: attn.effectiveScoreScale(blockTopology: config.blockTopology),
                softcap: config.attnLogitCap,
                mlxVectorPartialsSlot: plan.slots.first {
                    $0.name == "mlxAttentionPartialsD256B128"
                }?.index,
                mlxVectorStatsSlot: plan.slots.first {
                    $0.name == "mlxAttentionStatsD256B128"
                }?.index
            )
        }
        emitter.recordTraceMarker(
            label: "L\(layerIndex).attn_raw",
            bufferSlot: SmeltFixedSlot.attnOutBuf.rawValue
        )

        // --- 11. Gate: out = attnOut * sigmoid(gate) ---
        if attn.gatedQ {
        let attnOutDim = nQ * headDim
        lines += try emitter.emit(SmeltDispatch(
            pipeline: .sigmoidMul,
            buffers: [
                SmeltBufferBinding(slot: attnOutSlot, index: 0),
                SmeltBufferBinding(slot: attnGateSlot, index: 1),
                SmeltBufferBinding(slot: attnOutSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(attnOutDim)", type: .uint32, index: 3),
            ],
            dispatch: .threads(
                width: attnOutDim, height: 1, depth: 1,
                tgWidth: min(attnOutDim, 1024), tgHeight: 1, tgDepth: 1
            ),
            comment: "Sigmoid gate multiply (fused)"
        ))
        }  // end gatedQ gate

        emitter.recordTraceMarker(
            label: "L\(layerIndex).attn_ctx",
            bufferSlot: SmeltFixedSlot.attnOutBuf.rawValue
        )

        // --- 12. O projection: attnOutBuf → normOutBuf ---
        let oProjName = SmeltKernelConsumerNaming.oProjWeight(layerIndex: layerIndex)
        let oProjEntry = try requireWeight(oProjName, from: weightEntries)
        let oProjRoute = config.blockTopology == .standard
            ? plannedRoute(
                .attentionOutputResidualDecode,
                groupSize: oProjEntry.groupSize ?? groupSize
            )
            : nil
        let outputBank = config.projectionBank(
            source: .attentionOutput,
            containing: [.attentionOut]
        )
        if let activationView = outputBank?.activationView,
           outputBank?.usesActivationView(at: layerIndex) == true,
           plan.projectionActivationPlanesSlot >= 0,
           plan.projectionActivationScalesSlot >= 0,
           let lowBitOutputLines = try emitter.emitSignedBitplaneProjectionIfPossible(
               view: activationView,
               weightEntry: oProjEntry,
               weightsSlot: weightsSlot,
               inputSlot: attnOutSlot,
               planesSlot: plan.projectionActivationPlanesSlot,
               activationScalesSlot: plan.projectionActivationScalesSlot,
               outputSlot: normOutSlot,
               rows: hidden,
               cols: nQ * headDim,
               producerComment: "CAM attention-output activation view",
               projectionComment: "CAM attention-output low-bit projection"
           ) {
            lines += lowBitOutputLines
        } else {
            lines += try emitter.emitMatvec(
                weightEntry: oProjEntry, weightsSlot: weightsSlot,
                inputSlot: attnOutSlot, outputSlot: normOutSlot,
                rows: hidden, cols: nQ * headDim, groupSize: groupSize,
                comment: "O projection",
                plannedKernelRoute: oProjRoute
            )
        }
        if !traceMode.usesStrippedOptimizations {
            emitter.recordTraceMarker(
                label: "L\(layerIndex).attn_out",
                bufferSlot: SmeltFixedSlot.normOutBuf.rawValue
            )
        }

        return lines
    }

    // MARK: - Position-dependent dispatch helpers

    /// RoPE layout constant consumed by Metal kernels.
    /// 0 = adjacent pairs, 1 = split-half over the rotary span,
    /// 2 = proportional RoPE where the active rotary prefix pairs with
    /// the midpoint of the full head.
    static func ropeLayoutConstant(
        config: SmeltConfig,
        attn: SmeltAttentionConfig,
        headDim: Int,
        ropeDim: Int
    ) -> Int {
        attn.effectiveRoPELayoutConstant(blockTopology: config.blockTopology, ropeDim: ropeDim)
    }

    /// Emit RoPE dispatch. Uses raw lines for position-dependent cos/sin offsets.
    private static func emitRoPE(
        emitter: inout SmeltCodeEmitter,
        uniqueId: String,
        dataSlot: Int,
        ropeCosSlot: Int, ropeSinSlot: Int,
        numHeads: Int, headDim: Int, ropeDim: Int,
        ropeByteStride: Int,
        ropeLayout: Int,
        ropeBaseLog2: Float,
        ropeMathMode: Int,
        minPositionPlus1: Int? = nil,
        maxPositionPlus1Exclusive: Int? = nil
    ) throws -> [String] {
        // RoPE cos/sin offset = position * ropeDim * 2 (FP16 bytes).
        // Keep the offset expression inline so the binary dispatch encoder can
        // recover the stride for runtime position-dependent buffer binding.
        let offsetExpression = "Int(position) * \(ropeByteStride)"
        var lines: [String] = []

        lines += try emitter.emit(SmeltDispatch(
            pipeline: .applyRope,
            buffers: [
                SmeltBufferBinding(slot: dataSlot, index: 0),
                SmeltBufferBinding(
                    slot: ropeCosSlot, offsetExpression: offsetExpression, index: 1
                ),
                SmeltBufferBinding(
                    slot: ropeSinSlot, offsetExpression: offsetExpression, index: 2
                ),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(headDim)", type: .uint32, index: 3),
                SmeltConstantBinding(expression: "\(ropeDim)", type: .uint32, index: 4),
                SmeltConstantBinding(expression: "\(numHeads)", type: .uint32, index: 5),
                SmeltConstantBinding(expression: "\(ropeLayout)", type: .uint32, index: 6),
                SmeltConstantBinding(
                    expression: "UInt32(position)", type: .uint32, index: 7
                ),
                SmeltConstantBinding(expression: "\(ropeBaseLog2)", type: .float32, index: 8),
                SmeltConstantBinding(expression: "\(ropeMathMode)", type: .uint32, index: 9),
            ],
            dispatch: .threads(
                width: numHeads * headDim, height: 1, depth: 1,
                tgWidth: min(numHeads * headDim, 1024), tgHeight: 1, tgDepth: 1
            ),
            minPositionPlus1: minPositionPlus1,
            maxPositionPlus1Exclusive: maxPositionPlus1Exclusive
        ))

        return lines
    }

    /// Emit KV cache update dispatch (position-dependent).
    private static func emitKVCacheUpdate(
        emitter: inout SmeltCodeEmitter,
        cacheSlot: Int, newKVSlot: Int,
        headDim: Int, numHeads: Int,
        comment: String,
        minPositionPlus1: Int? = nil,
        maxPositionPlus1Exclusive: Int? = nil
    ) throws -> [String] {
        // position is a runtime value — emit as "UInt32(position)"
        try emitter.emit(SmeltDispatch(
            pipeline: .kvCacheUpdate,
            buffers: [
                SmeltBufferBinding(slot: cacheSlot, index: 0),
                SmeltBufferBinding(slot: newKVSlot, index: 1),
            ],
            constants: [
                SmeltConstantBinding(expression: "cacheSeqCapacity", type: .uint32, index: 2),
                SmeltConstantBinding(expression: "\(headDim)", type: .uint32, index: 3),
                SmeltConstantBinding(
                    expression: "UInt32(position)", type: .uint32, index: 4
                ),
                SmeltConstantBinding(expression: "\(numHeads)", type: .uint32, index: 5),
            ],
            dispatch: .threads(
                width: numHeads * headDim, height: 1, depth: 1,
                tgWidth: min(numHeads * headDim, 1024), tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            minPositionPlus1: minPositionPlus1,
            maxPositionPlus1Exclusive: maxPositionPlus1Exclusive
        ))
    }

    /// Emit fused K RoPE and cache write. This writes the rotated K vector
    /// straight to cache and intentionally leaves the source K buffer unchanged.
    private static func emitRopeKVCacheUpdate(
        emitter: inout SmeltCodeEmitter,
        cacheSlot: Int, newKVSlot: Int,
        ropeCosSlot: Int, ropeSinSlot: Int,
        ropeByteStride: Int,
        headDim: Int, ropeDim: Int, numHeads: Int,
        ropeLayout: Int,
        ropeBaseLog2: Float,
        ropeMathMode: Int,
        comment: String,
        minPositionPlus1: Int? = nil,
        maxPositionPlus1Exclusive: Int? = nil
    ) throws -> [String] {
        let offsetExpression = "Int(position) * \(ropeByteStride)"
        return try emitter.emit(SmeltDispatch(
            pipeline: .ropeKVCacheUpdate,
            buffers: [
                SmeltBufferBinding(slot: cacheSlot, index: 0),
                SmeltBufferBinding(slot: newKVSlot, index: 1),
                SmeltBufferBinding(
                    slot: ropeCosSlot, offsetExpression: offsetExpression, index: 2
                ),
                SmeltBufferBinding(
                    slot: ropeSinSlot, offsetExpression: offsetExpression, index: 3
                ),
            ],
            constants: [
                SmeltConstantBinding(expression: "cacheSeqCapacity", type: .uint32, index: 4),
                SmeltConstantBinding(expression: "\(headDim)", type: .uint32, index: 5),
                SmeltConstantBinding(
                    expression: "UInt32(position)", type: .uint32, index: 6
                ),
                SmeltConstantBinding(expression: "\(numHeads)", type: .uint32, index: 7),
                SmeltConstantBinding(expression: "\(ropeDim)", type: .uint32, index: 8),
                SmeltConstantBinding(expression: "\(ropeLayout)", type: .uint32, index: 9),
                SmeltConstantBinding(expression: "\(ropeBaseLog2)", type: .float32, index: 10),
                SmeltConstantBinding(expression: "\(ropeMathMode)", type: .uint32, index: 11),
            ],
            dispatch: .threads(
                width: numHeads * headDim, height: 1, depth: 1,
                tgWidth: min(numHeads * headDim, 1024), tgHeight: 1, tgDepth: 1
            ),
            comment: comment,
            minPositionPlus1: minPositionPlus1,
            maxPositionPlus1Exclusive: maxPositionPlus1Exclusive
        ))
    }

    /// Emit attention decode dispatch (position-dependent seqLen).
    private static func emitAttentionDecode(
        emitter: inout SmeltCodeEmitter,
        querySlot: Int, keyCacheSlot: Int, valCacheSlot: Int,
        maskSlot: Int, outputSlot: Int,
        hidden: Int,
        nQ: Int, nKV: Int, headDim: Int,
        gqaRatio: Int, slidingWindow: Int, scale: Float, softcap: Float?,
        mlxVectorPartialsSlot: Int? = nil,
        mlxVectorStatsSlot: Int? = nil,
        minPositionPlus1: Int? = nil,
        maxPositionPlus1Exclusive: Int? = nil
    ) throws -> [String] {
        let supportsMLXVectorD256 =
            softcap == nil
            && querySlot == outputSlot
            && headDim == 256
            && nKV > 0
            && nQ.isMultiple(of: nKV)
            && gqaRatio == nQ / nKV
            && gqaRatio <= 32
            && slidingWindow == 0
            && abs(scale - 0.0625) < 0.0001
            && mlxVectorPartialsSlot != nil
            && mlxVectorStatsSlot != nil
        let compactIdentityPipeline: SmeltPipeline? = switch (nQ, nKV) {
        case (8, 2): .attentionDecodeD256H8KV2
        case (16, 4): .attentionDecodeD256H16KV4
        case (24, 4): .attentionDecodeD256H24KV4
        default: nil
        }
        let useD128H16KV2SDPASpecialization =
            softcap == nil
            &&
            querySlot == outputSlot
            && nQ == 16
            && nKV == 2
            && headDim == 128
            && gqaRatio == 8
            && slidingWindow == 0
            && abs(scale - Float(1.0 / sqrt(128.0))) < 0.0001

        // MLX routing is a property of attention semantics and topology. It
        // applies to every compatible package regardless of model family.
        // seqLen = position + 1 (runtime expression).
        if supportsMLXVectorD256,
           let mlxVectorPartialsSlot,
           let mlxVectorStatsSlot
        {
            let requestedMin = minPositionPlus1 ?? 0
            let onePassLimit = Self.mlxVectorTwoPassMinSeqLen
            var lines: [String] = []

            if let compactIdentityPipeline {
                let directMax = min(
                    Self.mlxVectorDirectIdentityMaxSeqLenExclusive,
                    maxPositionPlus1Exclusive
                        ?? Self.mlxVectorDirectIdentityMaxSeqLenExclusive
                )
                if requestedMin < directMax {
                    lines += try emitter.emit(SmeltDispatch(
                        pipeline: compactIdentityPipeline,
                        buffers: [
                            SmeltBufferBinding(slot: querySlot, index: 0),
                            SmeltBufferBinding(slot: keyCacheSlot, index: 1),
                            SmeltBufferBinding(slot: valCacheSlot, index: 2),
                        ],
                        constants: [
                            SmeltConstantBinding(
                                expression: "UInt32(position + 1)", type: .uint32, index: 3
                            ),
                            SmeltConstantBinding(
                                expression: "cacheSeqCapacity", type: .uint32, index: 4
                            ),
                        ],
                        dispatch: .threadgroups(
                            width: nQ, height: 1, depth: 1,
                            tgWidth: 64, tgHeight: 1, tgDepth: 1
                        ),
                        comment: "Attention decode (compact D256 identity topology)",
                        minPositionPlus1: minPositionPlus1,
                        maxPositionPlus1Exclusive: directMax
                    ))
                }
            }

            let vectorMin = max(
                requestedMin,
                compactIdentityPipeline == nil
                    ? 0
                    : Self.mlxVectorDirectIdentityMaxSeqLenExclusive
            )
            let vectorMax = min(
                onePassLimit,
                maxPositionPlus1Exclusive ?? onePassLimit
            )
            if vectorMin < vectorMax {
                lines += try emitter.emit(SmeltDispatch(
                    pipeline: .attentionDecodeMLXVectorD256,
                    buffers: [
                        SmeltBufferBinding(slot: querySlot, index: 0),
                        SmeltBufferBinding(slot: keyCacheSlot, index: 1),
                        SmeltBufferBinding(slot: valCacheSlot, index: 2),
                    ],
                    constants: [
                        SmeltConstantBinding(
                            expression: "UInt32(position + 1)", type: .uint32, index: 3
                        ),
                        SmeltConstantBinding(
                            expression: "cacheSeqCapacity", type: .uint32, index: 4
                        ),
                        SmeltConstantBinding(
                            expression: "\(nQ)", type: .uint32, index: 5
                        ),
                        SmeltConstantBinding(
                            expression: "\(nKV)", type: .uint32, index: 6
                        ),
                    ],
                    dispatch: .threadgroups(
                        width: nQ, height: 1, depth: 1,
                        tgWidth: 1024, tgHeight: 1, tgDepth: 1
                    ),
                    comment: "Attention decode (MLX D256 vector topology)",
                    minPositionPlus1: vectorMin,
                    maxPositionPlus1Exclusive: vectorMax
                ))
            }

            let twoPassMin = max(onePassLimit, requestedMin)
            if maxPositionPlus1Exclusive.map({ twoPassMin < $0 }) ?? true {
                lines += try emitter.emit(SmeltDispatch(
                    pipeline: .attentionDecodeMLXVector2Pass1D256B128,
                    buffers: [
                        SmeltBufferBinding(slot: querySlot, index: 0),
                        SmeltBufferBinding(slot: keyCacheSlot, index: 1),
                        SmeltBufferBinding(slot: valCacheSlot, index: 2),
                        SmeltBufferBinding(slot: mlxVectorPartialsSlot, index: 3),
                        SmeltBufferBinding(slot: mlxVectorStatsSlot, index: 4),
                    ],
                    constants: [
                        SmeltConstantBinding(
                            expression: "UInt32(position + 1)", type: .uint32, index: 5
                        ),
                        SmeltConstantBinding(
                            expression: "cacheSeqCapacity", type: .uint32, index: 6
                        ),
                        SmeltConstantBinding(
                            expression: "\(nQ)", type: .uint32, index: 7
                        ),
                        SmeltConstantBinding(
                            expression: "\(nKV)", type: .uint32, index: 8
                        ),
                    ],
                    dispatch: .threadgroups(
                        width: nKV, height: 1, depth: 128,
                        tgWidth: 32, tgHeight: gqaRatio, tgDepth: 1
                    ),
                    comment: "Attention decode (MLX D256 two-pass partials, B128)",
                    minPositionPlus1: twoPassMin,
                    maxPositionPlus1Exclusive: maxPositionPlus1Exclusive
                ))
                lines += try emitter.emit(SmeltDispatch(
                    pipeline: .attentionDecodeMLXVector2Pass2D256B128,
                    buffers: [
                        SmeltBufferBinding(slot: mlxVectorPartialsSlot, index: 0),
                        SmeltBufferBinding(slot: mlxVectorStatsSlot, index: 1),
                        SmeltBufferBinding(slot: outputSlot, index: 2),
                    ],
                    constants: [
                        SmeltConstantBinding(
                            expression: "\(nQ)", type: .uint32, index: 3
                        ),
                    ],
                    dispatch: .threadgroups(
                        width: nQ, height: 1, depth: 1,
                        tgWidth: 1024, tgHeight: 1, tgDepth: 1
                    ),
                    comment: "Attention decode (MLX D256 two-pass reduction, B128)",
                    minPositionPlus1: twoPassMin,
                    maxPositionPlus1Exclusive: maxPositionPlus1Exclusive
                ))
            }
            return lines
        }

        if useD128H16KV2SDPASpecialization {
            return try emitter.emit(SmeltDispatch(
                pipeline: .attentionDecodeD128H16KV2SDPA,
                buffers: [
                    SmeltBufferBinding(slot: querySlot, index: 0),
                    SmeltBufferBinding(slot: keyCacheSlot, index: 1),
                    SmeltBufferBinding(slot: valCacheSlot, index: 2),
                ],
                constants: [
                    SmeltConstantBinding(
                        expression: "UInt32(position + 1)", type: .uint32, index: 3
                    ),
                    SmeltConstantBinding(
                        expression: "cacheSeqCapacity", type: .uint32, index: 4
                    ),
                ],
                dispatch: .threadgroups(
                    width: nQ, height: 1, depth: 1,
                    tgWidth: 1024, tgHeight: 1, tgDepth: 1
                ),
                comment: "Attention decode (D128 H16 KV2 SDPA specialization)",
                minPositionPlus1: minPositionPlus1,
                maxPositionPlus1Exclusive: maxPositionPlus1Exclusive
            ))
        }

        var constants = [
            SmeltConstantBinding(expression: "\(headDim)", type: .uint32, index: 5),
            SmeltConstantBinding(expression: "cacheSeqCapacity", type: .uint32, index: 6),
            SmeltConstantBinding(
                expression: "UInt32(position + 1)", type: .uint32, index: 7
            ),
            SmeltConstantBinding(expression: "\(nKV)", type: .uint32, index: 8),
            SmeltConstantBinding(expression: "\(scale)", type: .float32, index: 9),
            SmeltConstantBinding(
                expression: "\(slidingWindow)", type: .uint32, index: 10
            ),
        ]
        if let softcap {
            constants.append(
                SmeltConstantBinding(expression: "\(softcap)", type: .float32, index: 11)
            )
        }

        return try emitter.emit(SmeltDispatch(
            pipeline: softcap == nil ? .attentionDecode : .attentionDecodeSoftcap,
            buffers: [
                SmeltBufferBinding(slot: querySlot, index: 0),
                SmeltBufferBinding(slot: keyCacheSlot, index: 1),
                SmeltBufferBinding(slot: valCacheSlot, index: 2),
                SmeltBufferBinding(slot: maskSlot, index: 3),
                SmeltBufferBinding(slot: outputSlot, index: 4),
            ],
            constants: constants,
            dispatch: .threadgroups(
                width: nQ, height: 1, depth: 1,
                tgWidth: 256, tgHeight: 1, tgDepth: 1
            ),
            comment: softcap == nil ? "Attention decode" : "Attention decode with score softcapping",
            minPositionPlus1: minPositionPlus1,
            maxPositionPlus1Exclusive: maxPositionPlus1Exclusive
        ))
    }

    // MARK: - Helpers

    /// Emit a per-head RMS norm dispatch (in-place on dataSlot).
    static func emitPerHeadNorm(
        emitter: inout SmeltCodeEmitter,
        dataSlot: Int, numHeads: Int,
        weightName: String, weightsSlot: Int,
        weightEntries: [String: SmeltWeightEntry],
        headDim: Int, eps: Float,
        mode: SmeltNormMode,
        comment: String
    ) throws -> [String] {
        let threadgroupWidth = mode == .weight
            ? directWeightNormThreadgroupWidth(headDim: headDim)
            : min(headDim, 256)
        return try emitter.emit(SmeltDispatch(
            pipeline: mode == .onePlusWeight ? .perHeadRmsNorm1PW : .perHeadRmsNorm,
            buffers: [
                SmeltBufferBinding(slot: dataSlot, index: 0),
                SmeltBufferBinding(
                    slot: weightsSlot,
                    offset: try requireWeight(weightName, from: weightEntries).offset,
                    index: 1
                ),
                SmeltBufferBinding(slot: dataSlot, index: 2),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "\(headDim)", type: .uint32, index: 3
                ),
                SmeltConstantBinding(
                    expression: "\(eps)", type: .float32, index: 4
                ),
            ],
            dispatch: .threadgroups(
                width: numHeads, height: 1, depth: 1,
                tgWidth: threadgroupWidth, tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }

    /// MLX's direct-weight RMSNorm consumes four contiguous values per thread
    /// and rounds the thread count up to whole SIMD groups. The geometry is a
    /// property of this norm implementation, not of any model family.
    static func directWeightNormThreadgroupWidth(headDim: Int) -> Int {
        let valueThreads = (headDim + 3) / 4
        return max(32, ((valueThreads + 31) / 32) * 32)
    }

    /// Emit a scale-less per-head RMS norm dispatch (in-place on dataSlot).
    static func emitPerHeadNormNoScale(
        emitter: inout SmeltCodeEmitter,
        dataSlot: Int, numHeads: Int,
        headDim: Int, eps: Float,
        comment: String
    ) throws -> [String] {
        try emitter.emit(SmeltDispatch(
            pipeline: .perHeadRmsNormNoScale,
            buffers: [
                SmeltBufferBinding(slot: dataSlot, index: 0),
            ],
            constants: [
                SmeltConstantBinding(
                    expression: "\(headDim)", type: .uint32, index: 1
                ),
                SmeltConstantBinding(
                    expression: "\(eps)", type: .float32, index: 2
                ),
            ],
            dispatch: .threadgroups(
                width: numHeads, height: 1, depth: 1,
                tgWidth: min(headDim, 256), tgHeight: 1, tgDepth: 1
            ),
            comment: comment
        ))
    }
}
