// DeltaNetPlugin — Emits dispatch code for one DeltaNet layer.
//
// Uses SmeltCodeEmitter to generate the split DeltaNet decode path:
// 3 projections (QKV, Z, fused A+B), conv1d update, Q/K normalization,
// MLX-style tiled recurrence core, gated RMS norm, output projection.
//
// All slot indices and weight offsets are resolved at compile time.
// The generated code uses only integer indices — no string lookups.

import SmeltSchema

/// Emits Swift dispatch code for one DeltaNet layer.
public struct DeltaNetPlugin {

    /// Emit dispatch code for a single DeltaNet layer.
    ///
    /// - Parameters:
    ///   - layerIndex: Global layer index (0-23 for Qwen 3.5 2B).
    ///   - deltaIndex: DeltaNet-local index (0-17 for Qwen 3.5 2B).
    ///   - config: Model config (must have delta != nil).
    ///   - plan: Buffer plan for slot indices.
    ///   - weightEntries: Weight layout entries keyed by name.
    ///   - weightsSlot: Slot index of the monolithic weight buffer.
    ///   - emitter: Code emitter (mutating — tracks dispatch counter).
    /// - Returns: Array of Swift source lines.
    public static func emitLayer(
        layerIndex: Int,
        deltaIndex: Int,
        config: SmeltConfig,
        plan: SmeltBufferPlan,
        weightEntries: [String: SmeltWeightEntry],
        weightsSlot: Int,
        groupSize: Int,
        traceMode: SmeltTraceMode = .full,
        emitter: inout SmeltCodeEmitter
    ) throws -> [String] {
        guard let delta = config.delta else {
            throw SmeltEmitError.missingConfig(detail: "DeltaNet plugin requires delta config")
        }

        let prefix = "layers_\(layerIndex)_linear_attn"
        let hidden = config.hiddenSize
        let numHeads = delta.numHeads
        let qkHeads = delta.qkHeads
        let headDim = delta.headDim
        let qkvDim = delta.qkvDim
        let valueDim = delta.zDim
        let convKernel = delta.convKernel
        let useD128H16Specialization =
            qkvDim == 6144 && convKernel == 4 && headDim == 128 && numHeads == 16
        let useD128H32QK16Specialization =
            qkvDim == 8192 && convKernel == 4 && headDim == 128 && numHeads == 32 && qkHeads == 16
        let useD128H48QK16Specialization =
            qkvDim == 10240 && convKernel == 4 && headDim == 128
                && numHeads == 48 && qkHeads == 16

        // Slot indices
        let qkvSlot = SmeltFixedSlot.qkvBuf.rawValue
        let zSlot = SmeltFixedSlot.zBuf.rawValue
        let aSlot = SmeltFixedSlot.aBuf.rawValue
        let bSlot = SmeltFixedSlot.bBuf.rawValue
        let normOutSlot = SmeltFixedSlot.normOutBuf.rawValue
        let recOutSlot = SmeltFixedSlot.recOutBuf.rawValue
        let convStateSlot = plan.convStateBaseSlot + deltaIndex
        let recStateSlot = plan.recStateBaseSlot + deltaIndex

        var lines: [String] = []

        let qkvName = "\(prefix)_in_proj_qkv_weight"
        let zName = "\(prefix)_in_proj_z_weight"
        let aName = "\(prefix)_in_proj_a_weight"
        let bName = "\(prefix)_in_proj_b_weight"
        let qkvEntry = try requireWeight(qkvName, from: weightEntries)
        let zEntry = try requireWeight(zName, from: weightEntries)
        let aEntry = try requireWeight(aName, from: weightEntries)
        let bEntry = try requireWeight(bName, from: weightEntries)
        let deltaBank = config.projectionBank(
            source: .deltaInput,
            containing: [.deltaQKV, .deltaZ, .deltaA, .deltaB]
        )
        let membersByEndpoint: [SmeltCAMIR.ProjectionEndpoint: (
            entry: SmeltWeightEntry, outputSlot: Int, rows: Int
        )] = [
            .deltaQKV: (qkvEntry, qkvSlot, qkvDim),
            .deltaZ: (zEntry, zSlot, valueDim),
            .deltaA: (aEntry, aSlot, numHeads),
            .deltaB: (bEntry, bSlot, numHeads),
        ]
        let packedMembers = deltaBank?.outputs.compactMap { membersByEndpoint[$0] } ?? []
        var packedProjectionLines: [String]?
        if let activationView = deltaBank?.activationView,
           deltaBank?.usesActivationView(at: layerIndex) == true,
           plan.projectionActivationPlanesSlot >= 0,
           plan.projectionActivationScalesSlot >= 0 {
            packedProjectionLines = try emitter
                .emitSignedBitplaneProjectionBankIfPossible(
                    view: activationView,
                    members: packedMembers,
                    weightsSlot: weightsSlot,
                    planesSlot: plan.projectionActivationPlanesSlot,
                    activationScalesSlot: plan.projectionActivationScalesSlot,
                    cols: hidden,
                    comment: "CAM delta-input low-bit projection bank"
                )
        }
        if packedProjectionLines == nil {
            packedProjectionLines = try emitter
                .emitSignedPackedProjectionBankIfPossible(
                    members: packedMembers,
                    weightsSlot: weightsSlot,
                    inputSlot: normOutSlot,
                    cols: hidden,
                    comment: "CAM delta-input projection bank"
                )
        }
        if let packedProjectionLines {
            lines += packedProjectionLines
        } else {
            // Ordinary graph lowering remains available for every storage
            // format and for packages authored before projection banks.
            lines += try emitter.emitMatvec(
                weightEntry: qkvEntry, weightsSlot: weightsSlot,
                inputSlot: normOutSlot, outputSlot: qkvSlot,
                rows: qkvDim, cols: hidden, groupSize: groupSize,
                comment: "QKV projection"
            )
            lines += try emitter.emitMatvec(
                weightEntry: zEntry, weightsSlot: weightsSlot,
                inputSlot: normOutSlot, outputSlot: zSlot,
                rows: valueDim, cols: hidden, groupSize: groupSize,
                comment: "Z projection"
            )
            // Family via the ONE gateway: the fused dual kernel is used iff BOTH weights authorize the
            // same fused-dual family (u4_lut/affine_u4); anything else (fp16, bf16/fp32, mismatched)
            // falls to the separate-matvec else, itself gateway-routed.
            let abFamily = emitter.bothFusedFamily(
                aEntry, bEntry, shape: .gemv, fusion: .dualMatvec)
            if abFamily == .lutU4,
               let aLutOff = aEntry.lutOffset,
               let bLutOff = bEntry.lutOffset {
                lines += try emitter.emitFusedDualLutMatvec(
                    w1IndicesSlot: weightsSlot, w1IndicesOffset: aEntry.offset,
                    w1LutSlot: weightsSlot, w1LutOffset: aLutOff,
                    w2IndicesSlot: weightsSlot, w2IndicesOffset: bEntry.offset,
                    w2LutSlot: weightsSlot, w2LutOffset: bLutOff,
                    inputSlot: normOutSlot, output1Slot: aSlot, output2Slot: bSlot,
                    rows: numHeads, cols: hidden, groupSize: groupSize,
                    comment: "Fused A+B projection"
                )
            } else if abFamily == .affineU4,
                      let aScales = aEntry.scalesOffset,
                      let aBiases = aEntry.biasesOffset,
                      let bScales = bEntry.scalesOffset,
                      let bBiases = bEntry.biasesOffset {
                lines += try emitter.emitFusedDualAffineMatvec(
                    w1WeightsSlot: weightsSlot, w1WeightsOffset: aEntry.offset,
                    w1ScalesSlot: weightsSlot, w1ScalesOffset: aScales,
                    w1BiasesSlot: weightsSlot, w1BiasesOffset: aBiases,
                    w2WeightsSlot: weightsSlot, w2WeightsOffset: bEntry.offset,
                    w2ScalesSlot: weightsSlot, w2ScalesOffset: bScales,
                    w2BiasesSlot: weightsSlot, w2BiasesOffset: bBiases,
                    inputSlot: normOutSlot, output1Slot: aSlot, output2Slot: bSlot,
                    rows: numHeads, cols: hidden, groupSize: groupSize,
                    comment: "Fused A+B projection (affine)"
                )
            } else {
                lines += try emitter.emitMatvec(
                    weightEntry: bEntry, weightsSlot: weightsSlot,
                    inputSlot: normOutSlot, outputSlot: bSlot,
                    rows: numHeads, cols: hidden, groupSize: groupSize,
                    comment: "B (beta) projection"
                )
                lines += try emitter.emitMatvec(
                    weightEntry: aEntry, weightsSlot: weightsSlot,
                    inputSlot: normOutSlot, outputSlot: aSlot,
                    rows: numHeads, cols: hidden, groupSize: groupSize,
                    comment: "A (decay) projection"
                )
            }
        }
        emitter.recordTraceMarker(
            label: "L\(layerIndex).delta_qkv",
            bufferSlot: qkvSlot
        )
        emitter.recordTraceMarker(
            label: "L\(layerIndex).delta_b",
            bufferSlot: bSlot
        )
        emitter.recordTraceMarker(
            label: "L\(layerIndex).delta_a",
            bufferSlot: aSlot
        )

        // --- 4. Split DeltaNet decode path ---
        let convWeightName = "\(prefix)_conv1d_weight"
        let aLogName = "\(prefix)_A_log"
        let dtBiasName = "\(prefix)_dt_bias"
        let normWeightName = "\(prefix)_norm_weight"

        lines += try emitter.emit(SmeltDispatch(
            pipeline: .conv1dUpdateSilu,
            buffers: [
                SmeltBufferBinding(slot: convStateSlot, index: 0),
                SmeltBufferBinding(slot: qkvSlot, index: 1),
                SmeltBufferBinding(
                    slot: weightsSlot,
                    offset: try requireWeight(convWeightName, from: weightEntries).offset,
                    index: 2
                ),
                SmeltBufferBinding(slot: qkvSlot, index: 3),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(qkvDim)", type: .uint32, index: 4),
                SmeltConstantBinding(expression: "\(convKernel)", type: .uint32, index: 5),
            ],
            dispatch: .threads(
                width: qkvDim, height: 1, depth: 1,
                tgWidth: 256, tgHeight: 1, tgDepth: 1
            ),
            comment: "DeltaNet conv1d update + SiLU"
        ))

        let normTgWidth = min(headDim, 256)
        let qkNormThreads = max(32, ((headDim + 127) / 128) * 32)
        lines += try emitter.emit(SmeltDispatch(
            pipeline: .rmsScaleQK,
            buffers: [
                SmeltBufferBinding(slot: qkvSlot, index: 0),
            ],
            constants: [
                SmeltConstantBinding(expression: "\(headDim)", type: .uint32, index: 1),
                SmeltConstantBinding(expression: "1e-6", type: .float32, index: 2),
                SmeltConstantBinding(expression: "\(qkvDim)", type: .uint32, index: 3),
                SmeltConstantBinding(expression: "\(qkHeads)", type: .uint32, index: 4),
            ],
            dispatch: .threadgroups(
                width: 2 * qkHeads, height: 1, depth: 1,
                tgWidth: qkNormThreads, tgHeight: 1, tgDepth: 1
            ),
            comment: "DeltaNet RMS scale Q+K"
        ))
        emitter.recordTraceMarker(
            label: "L\(layerIndex).delta_conv",
            bufferSlot: qkvSlot
        )

        lines += try emitter.emit(SmeltDispatch(
            pipeline: useD128H16Specialization
                ? .deltanetRecurrenceMlxDecodeD128H16
                : (useD128H32QK16Specialization
                    ? .deltanetRecurrenceMlxDecodeD128H32QK16
                    : (useD128H48QK16Specialization
                        ? .deltanetRecurrenceMlxDecodeD128H48QK16
                        : .deltanetRecurrenceMlxDecode)),
            buffers: [
                SmeltBufferBinding(slot: recStateSlot, index: 0),
                SmeltBufferBinding(slot: qkvSlot, index: 1),
                SmeltBufferBinding(slot: bSlot, index: 2),
                SmeltBufferBinding(slot: aSlot, index: 3),
                SmeltBufferBinding(
                    slot: weightsSlot,
                    offset: try requireWeight(aLogName, from: weightEntries).offset,
                    index: 4
                ),
                SmeltBufferBinding(
                    slot: weightsSlot,
                    offset: try requireWeight(dtBiasName, from: weightEntries).offset,
                    index: 5
                ),
                SmeltBufferBinding(slot: recOutSlot, index: 6),
            ],
            constants: (
                useD128H16Specialization
                    || useD128H32QK16Specialization
                    || useD128H48QK16Specialization
            ) ? [] : [
                SmeltConstantBinding(expression: "\(headDim)", type: .uint32, index: 7),
                SmeltConstantBinding(expression: "\(delta.headScale)", type: .float32, index: 8),
                SmeltConstantBinding(expression: "\(numHeads)", type: .uint32, index: 9),
                SmeltConstantBinding(expression: "\(qkHeads)", type: .uint32, index: 10),
            ],
            dispatch: .threads(
                width: 32, height: headDim, depth: numHeads,
                tgWidth: 32, tgHeight: 4, tgDepth: 1
            ),
            comment: "DeltaNet recurrence core (MLX-style tiled decode)"
        ))
        emitter.recordTraceMarker(
            label: "L\(layerIndex).delta_core",
            bufferSlot: recOutSlot
        )

        let useD128GatedNorm = headDim == 128 && config.rmsEps == 1e-6
        lines += try emitter.emit(SmeltDispatch(
            pipeline: useD128GatedNorm ? .rmsNormGatedD128 : .rmsNormGated,
            buffers: [
                SmeltBufferBinding(slot: recOutSlot, index: 0),
                SmeltBufferBinding(slot: zSlot, index: 1),
                SmeltBufferBinding(
                    slot: weightsSlot,
                    offset: try requireWeight(normWeightName, from: weightEntries).offset,
                    index: 2
                ),
                SmeltBufferBinding(slot: recOutSlot, index: 3),
            ],
            constants: useD128GatedNorm ? [] : [
                SmeltConstantBinding(expression: "\(headDim)", type: .uint32, index: 4),
                SmeltConstantBinding(expression: "\(config.rmsEps)", type: .float32, index: 5),
            ],
            dispatch: .threadgroups(
                width: numHeads, height: 1, depth: 1,
                tgWidth: useD128GatedNorm ? 32 : normTgWidth,
                tgHeight: 1, tgDepth: 1
            ),
            comment: "DeltaNet gated RMS norm"
        ))
        emitter.recordTraceMarker(
            label: "L\(layerIndex).delta_rec",
            bufferSlot: recOutSlot
        )

        // --- 5. Output projection: recOutBuf → normOutBuf ---
        let outProjName = "\(prefix)_out_proj_weight"
        let outProjEntry = try requireWeight(outProjName, from: weightEntries)
        let outBank = config.projectionBank(
            source: .deltaOutput,
            containing: [.deltaOut]
        )
        if let activationView = outBank?.activationView,
           outBank?.usesActivationView(at: layerIndex) == true,
           plan.projectionActivationPlanesSlot >= 0,
           plan.projectionActivationScalesSlot >= 0,
           let lowBitOutputLines = try emitter.emitSignedBitplaneProjectionIfPossible(
               view: activationView,
               weightEntry: outProjEntry,
               weightsSlot: weightsSlot,
               inputSlot: recOutSlot,
               planesSlot: plan.projectionActivationPlanesSlot,
               activationScalesSlot: plan.projectionActivationScalesSlot,
               outputSlot: normOutSlot,
               rows: hidden,
               cols: valueDim,
               producerComment: "CAM delta-output activation view",
               projectionComment: "CAM delta-output low-bit projection"
           ) {
            lines += lowBitOutputLines
        } else {
            lines += try emitter.emitMatvec(
                weightEntry: outProjEntry, weightsSlot: weightsSlot,
                inputSlot: recOutSlot, outputSlot: normOutSlot,
                rows: hidden, cols: valueDim, groupSize: groupSize,
                comment: "Output projection"
            )
        }
        if !traceMode.usesStrippedOptimizations {
            emitter.recordTraceMarker(
                label: "L\(layerIndex).delta_out",
                bufferSlot: normOutSlot
            )
        }

        return lines
    }
}
