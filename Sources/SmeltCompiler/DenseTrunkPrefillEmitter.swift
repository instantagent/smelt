// DenseTrunkPrefillEmitter — compiled dense headless-trunk prefill lowering.
// This is the M>1 counterpart of DenseTrunkEmitter. Activation storage is an
// IR axis; BF16 and FP32 share topology and bindings while operation kernels
// supply the dtype-specific implementation.
//
// Ports (identical to decode): `embeddings` input (hiddenA holds [seqLen, hidden]
// embeddings, written by the caller; no token gather) and `hidden` output (the
// final norm lands in normOutBuf; no LM head, no argmax, no selection).
//
// Prefill expands the same layer graph into batched GEMM and cached-attention
// bricks. Two swaps per layer keep cur == hiddenA at layer entry.
//
// Bit-exactness notes (the hand path runs slightly different kernels; each is a
// documented bit-identical equivalent of what we emit):
//   - matmul: the hand routes bf16 M>1 to gemm_tn_bf16w_f32 (first choice); we
//     emit gemm_bf16w_f32. The catalog documents both as bit-identical per (n,m)
//     (their codes==gen_codes gate depends on it); gemm has the simpler runtime
//     grid (height = seqLen, no TG_M padding).
//   - attention: the hand runs causal_gqa_attn_simd_f32 (its first choice); we emit
//     causal_gqa_attn_cached_f32, its start-aware sibling, byte-identical at
//     startPos 0 (the single-chunk W2 gate) and the cross-chunk path for startPos>0.
//   - head-norm / rope in place: the hand uses fresh transient buffers, but each
//     kernel thread reads then writes only its own lane(s), so in-place is the
//     same computation, same bytes.
//   - residual: the hand sets has_scale=1 with an all-ones scale; we set
//     has_scale=0 (out = residual + x). 1.0*x == x in fp32, so bit-identical.
//
// Runtime shape: frames (= seqLen) is a runtime value, so grids use the prefill
// dynamic-grid kinds (.seqLen / .seqLenMul) and constants use __seqLen__; KV/rope
// rows are startPos-strided (offsetKind 1) — offset 0 at startPos 0. The attention
// is cross-chunk: causal_gqa_attn_cached_f32 takes __startPos__ and attends absolute
// cache rows [0, startPos+t], so this table runs at any startPos (B3.2b). The chunked
// harness (SmeltRuntime.prefillTrunkChunked) drives it per chunk over a long prompt.
//
// Dtype discipline (the Phase-12 lesson): projections are a valid kernel lego in
// bf16/fp16/fp32/affine_u4 (legality via the ONE gateway, MatvecKernelTable.select), norms
// fp32 — a dtype with no matvec kernel throws; there is no silent fallback.

import Foundation
import SmeltSchema

public enum DenseTrunkPrefillEmitter {

    /// Generate the prefill dispatch table for a dense headless trunk.
    public static func generate(
        ir: SmeltModelIR,
        plan: SmeltBufferPlan,
        weightLayout: [SmeltWeightEntry]
    ) throws -> PrefillEmitter.GenerateResult {
        let cfg = ir.config
        guard let attn = cfg.attentionConfigs[.attention] else {
            throw DenseTrunkEmitterError.unsupported("no attention config")
        }
        // This table emits causal_gqa_attn_cached_f32 (the start-aware cross-chunk
        // attention, B3.2b), whose threadgroup score buffer is sized 2048 — a chunk
        // whose absolute causal length (startPos+chunkLen) passes that reads out of
        // bounds. A SINGLE chunk at startPos 0 has causal length == chunkLen, so the
        // per-chunk bound is the kernel's 2048 cap; the chunked harness additionally
        // bounds TOTAL seqLen ≤ 2048 at runtime (the last chunk's causal length is
        // the full seqLen). The emitter owns the kernel choice, so it owns the
        // per-chunk bound (validateSmeltIR also guards).
        guard let prefill = ir.prefill else {
            throw DenseTrunkEmitterError.unsupported("the dense prefill table requires a prefill config")
        }
        // Batch in 1...2048 (validateSmeltIR enforces this upstream too; the
        // emitter is a public entry, so it owns the bound a direct caller could
        // otherwise violate). The buffer plan uses max_prefill_batch as the
        // activation-slab multiplier (≥1); the cached attention caps at 2048.
        guard (1...2048).contains(prefill.maxBatchSize) else {
            throw DenseTrunkEmitterError.unsupported(
                "max_prefill_batch \(prefill.maxBatchSize) is out of 1...2048 "
                + "(causal_gqa_attn_cached_f32's threadgroup score buffer caps at 2048)")
        }
        // The prefill table realises EXACTLY the same dense-topology shape as the
        // decode table (DenseTrunkEmitter), so it enforces the same guard set — any
        // knob validateSmeltIR accepts but this emitter does not realise must
        // throw loud, not silently mis-emit. Kept in lockstep with DenseTrunkEmitter.
        try DenseTrunkGuards.validate(cfg: cfg, attn: attn, layerPattern: ir.layerPattern)

        let weights = Dictionary(uniqueKeysWithValues: weightLayout.map { ($0.name, $0) })
        let weightsSlot = SmeltFixedSlot.weights.rawValue
        let hidden = cfg.hiddenSize
        let heads = attn.qHeads, kvHeads = attn.kvHeads, headDim = attn.headDim
        let qDim = heads * headDim, kvDim = kvHeads * headDim
        let inter = cfg.ffn.dim
        let eps = cfg.rmsEps
        let bf16Activation = cfg.activationDtype == .bf16
        let activationBytes = bf16Activation ? 2 : 4
        let kvRowStride = UInt64(kvDim * activationBytes)
        let ropeRowStride = UInt64(headDim * activationBytes)

        // Resolved through the shared DenseTrunkGuards resolver, in lockstep with
        // DenseTrunkEmitter.proj (one resolver, so decode and prefill structurally cannot
        // disagree on the supported projection dtypes — the Phase-12 dispatch-safety class).
        func proj(_ name: String) throws -> SmeltWeightEntry {
            guard let entry = weights[name] else { throw DenseTrunkEmitterError.missingWeight(name) }
            if bf16Activation, entry.dtype != .bf16 {
                throw DenseTrunkEmitterError.wrongDtype(
                    name: name, expected: "bf16 for a bf16 activation trunk", got: "\(entry.dtype)")
            }
            _ = try DenseTrunkGuards.resolveProjection(
                entry,
                activationDtype: cfg.activationDtype,
                shape: .gemm,
                name: name
            )
            return entry
        }
        func norm(_ name: String) throws -> SmeltWeightEntry {
            guard let entry = weights[name] else { throw DenseTrunkEmitterError.missingWeight(name) }
            if bf16Activation, entry.dtype != .bf16 {
                throw DenseTrunkEmitterError.wrongDtype(
                    name: name, expected: "bf16 for a bf16 activation trunk", got: "\(entry.dtype)")
            }
            guard entry.dtype == .fp32 || entry.dtype == .bf16 else {
                throw DenseTrunkEmitterError.wrongDtype(
                    name: name, expected: "fp32 or bf16", got: "\(entry.dtype)")
            }
            return entry
        }

        // Slots are sized [maxBatch, dim] in the declared activation storage.
        let normOut = SmeltFixedSlot.normOutBuf.rawValue
        let qBuf = SmeltFixedSlot.attnQBuf.rawValue
        let kBuf = SmeltFixedSlot.attnKBuf.rawValue
        let attnOut = SmeltFixedSlot.attnOutBuf.rawValue
        let gateBuf = SmeltFixedSlot.ffnGateBuf.rawValue
        let upBuf = SmeltFixedSlot.ffnUpBuf.rawValue
        let swigluOut = SmeltFixedSlot.ffnIntBuf.rawValue

        var records: [SmeltDispatchRecord] = []
        // GPTQ capture points (calibration, Phase 4 U2): `dispatchOps` is the count of NON-swap
        // dispatch ops emitted so far, matching how SmeltRuntime.captureGPTQActivations counts the
        // table (swaps excluded). The trunk prefill table is written verbatim — no optimizer pass
        // (Qwen3TTSTrunkSidecar) — so this emit-time count IS the runtime capture boundary.
        var dispatchOps = 0
        var capturePoints: [SmeltGPTQCapturePoint] = []
        func emit(_ dispatch: SmeltDispatch) { records.append(dispatch.toRecord()); dispatchOps += 1 }
        func emitSwap() {
            var rec = SmeltDispatchRecord.empty()
            rec.opKind = SmeltDispatchRecord.opSwap
            records.append(rec)
        }

        // Constant helpers.
        func u32(_ v: Int, _ i: Int) -> SmeltConstantBinding {
            SmeltConstantBinding(expression: "\(v)", type: .uint32, index: i)
        }
        func seqLen(_ i: Int) -> SmeltConstantBinding {
            SmeltConstantBinding(expression: "__seqLen__", type: .uint32, index: i)
        }
        func seqLenMul(_ k: Int, _ i: Int) -> SmeltConstantBinding {
            SmeltConstantBinding(expression: "__seqLen__*\(k)", type: .uint32, index: i)
        }
        func startPos(_ i: Int) -> SmeltConstantBinding {
            SmeltConstantBinding(expression: "__startPos__", type: .uint32, index: i)
        }
        func f32(_ v: Float, _ i: Int) -> SmeltConstantBinding {
            SmeltConstantBinding(expression: "\(v)", type: .float32, index: i)
        }
        let dummy = SmeltBufferBinding(slot: weightsSlot, offset: 0, index: 2)  // unread (has_bias/has_scale = 0)

        // --- per-kernel emitters (frames = runtime seqLen) ---

        // rms_norm_codec_f32: x@0 w@1 out@2 | frames@3 dim@4 eps@5 | threads seqLen*32 ÷ 32.
        func rmsNormCodec(x: SmeltBufferBinding, weight: SmeltWeightEntry, out: Int, comment: String) {
            emit(SmeltDispatch(
                pipeline: bf16Activation
                    ? .rmsNormCodecBF16
                    : (weight.dtype == .bf16 ? .rmsNormCodecBF16WF32 : .rmsNormCodecF32),
                buffers: [x,
                          SmeltBufferBinding(slot: weightsSlot, offset: weight.offset, index: 1),
                          SmeltBufferBinding(slot: out, index: 2)],
                constants: [seqLen(3), u32(hidden, 4), f32(eps, 5)],
                dispatch: .threads(width: 32, height: 1, depth: 1, tgWidth: 32, tgHeight: 1, tgDepth: 1),
                comment: comment, dynamicGridW: .seqLenMul(32)))
        }

        // Rebind a (fixed-slot) out binding at a different Metal index — gemm_u4_f32 puts out
        // at @5 (vs gemm_bf16w_f32's @3); preserves byteOffset + offsetKind (cache-row writes).
        func reindex(_ b: SmeltBufferBinding, _ idx: Int) -> SmeltBufferBinding {
            guard case .fixed(let s) = b.slot else {
                preconditionFailure("gemm out must be a fixed slot")
            }
            return SmeltBufferBinding(slot: s, offset: b.byteOffset, offsetKind: b.offsetKind, index: idx)
        }
        // Projection GEMM, routed by the weight's dtype (Phase 3). bf16 → gemm_bf16w_f32
        // (byte-identical); u4 → gemm_u4_f32 binding the 3 regions + group_size, mirroring the
        // hand matmulU4Dispatch M>1 path. `out` may be a cache-row binding (offsetKind 1).
        //   bf16: x@0 w@1 bias@2 out@3 | M@4 N@5 K@6 has_bias@7
        //   u4:   x@0 nibbles@1 scales@2 biases@3 bias@4 out@5 | M@6 N@7 K@8 has_bias@9 group@10
        func gemm(x: Int, w: SmeltWeightEntry, out: SmeltBufferBinding, n: Int, k: Int, comment: String) throws {
            let loc = try WeightLocator.resolve(w)
            switch loc.kind {
            case .affineU4(let groupSize):
                let r = loc.regions
                emit(SmeltDispatch(
                    pipeline: .gemmU4F32,
                    buffers: [SmeltBufferBinding(slot: x, index: 0),
                              SmeltBufferBinding(slot: weightsSlot, offset: r[0].offset, index: 1),
                              SmeltBufferBinding(slot: weightsSlot, offset: r[1].offset, index: 2),
                              SmeltBufferBinding(slot: weightsSlot, offset: r[2].offset, index: 3),
                              SmeltBufferBinding(slot: weightsSlot, index: 4),
                              reindex(out, 5)],
                    constants: [seqLen(6), u32(n, 7), u32(k, 8), u32(0, 9), u32(groupSize, 10)],
                    dispatch: .threads(width: n * 32, height: 1, depth: 1, tgWidth: 32, tgHeight: 1, tgDepth: 1),
                    comment: comment, dynamicGridH: .seqLen))
            case .dense(.bf16), .dense(.fp32), .dense(.fp16):
                // Single dense weight region — gemm_bf16w_f32 / gemm_f32 / gemm_f16w_f32 share the
                // SAME binding + NON-TN geometry (grid=(N*32, seqLen), tg=(32,1)); dtype picks the
                // kernel lego. Dummy bias binds weights (has_bias=0, unread).
                let pipe: SmeltPipeline = {
                    if bf16Activation { return .gemmBF16 }
                    switch loc.kind { case .dense(.fp16): return .gemmF16WF32
                    case .dense(.fp32): return .gemmF32; default: return .gemmBF16WF32 }
                }()
                emit(SmeltDispatch(
                    pipeline: pipe,
                    buffers: [SmeltBufferBinding(slot: x, index: 0),
                              SmeltBufferBinding(slot: weightsSlot, offset: w.offset, index: 1),
                              dummy,
                              out],
                    constants: [seqLen(4), u32(n, 5), u32(k, 6), u32(0, 7)],
                    dispatch: .threads(width: n * 32, height: 1, depth: 1, tgWidth: 32, tgHeight: 1, tgDepth: 1),
                    comment: comment, dynamicGridH: .seqLen))
            case .dense(let dt):
                throw DenseTrunkEmitterError.wrongDtype(name: w.name, expected: "bf16/fp16/fp32 or u4", got: "\(dt)")
            case .signed(let dt, _):
                throw DenseTrunkEmitterError.wrongDtype(
                    name: w.name, expected: "bf16/fp16/fp32 or affine_u4", got: dt.rawValue)
            }
            // GPTQ capture point (Phase 4 U2): the projection's dispatch just executed and did
            // NOT write its input slot `x`, so `x` still holds the [seqLen, k] FP32 activation —
            // read it to accumulate this weight's Hessian. The dense trunk emits gemm() only for the
            // 7 GPTQ-scope projections (q/k/v/o/gate/up/down), so every call is in scope. inputIsFloat16
            // = false: the trunk activation ports are FP32 (not the text affine-u4 fp16 ABI).
            capturePoints.append(SmeltGPTQCapturePoint(
                weightName: w.name, inputSlot: x, k: k, dispatchCount: dispatchOps, inputIsFloat16: false))
        }

        // rms_norm_head_f32 (in place): x@0 w@1 out@2 | frames@3 heads@4 headDim@5 eps@6
        //   | threads seqLen*nHeads*32 ÷ 32.
        func headNorm(buf: Int, weight: SmeltWeightEntry, nHeads: Int, comment: String) {
            emit(SmeltDispatch(
                pipeline: bf16Activation
                    ? .rmsNormHeadBF16
                    : (weight.dtype == .bf16 ? .rmsNormHeadBF16WF32 : .rmsNormHeadF32),
                buffers: [SmeltBufferBinding(slot: buf, index: 0),
                          SmeltBufferBinding(slot: weightsSlot, offset: weight.offset, index: 1),
                          SmeltBufferBinding(slot: buf, index: 2)],
                constants: [seqLen(3), u32(nHeads, 4), u32(headDim, 5), f32(eps, 6)],
                dispatch: .threads(width: 32, height: 1, depth: 1, tgWidth: 32, tgHeight: 1, tgDepth: 1),
                comment: comment, dynamicGridW: .seqLenMul(nHeads * 32)))
        }

        // rope_apply_f32: x@0 cos@1 sin@2 out@3 | frames@4 heads@5 headDim@6
        //   | threads seqLen*nHeads*(headDim/2) ÷ 256. cos/sin rows startPos-strided
        //   (offsetKind 1) so the kernel's local-t index reads absolute row startPos+t.
        func rope(x: Int, out: SmeltBufferBinding, nHeads: Int, comment: String) {
            emit(SmeltDispatch(
                pipeline: bf16Activation ? .ropeApplyBF16 : .ropeApplyF32,
                buffers: [SmeltBufferBinding(slot: x, index: 0),
                          SmeltBufferBinding(slot: plan.ropeCosSlot, offset: ropeRowStride, offsetKind: 1, index: 1),
                          SmeltBufferBinding(slot: plan.ropeSinSlot, offset: ropeRowStride, offsetKind: 1, index: 2),
                          out],
                constants: [seqLen(4), u32(nHeads, 5), u32(headDim, 6)],
                dispatch: .threads(width: 256, height: 1, depth: 1, tgWidth: 256, tgHeight: 1, tgDepth: 1),
                comment: comment, dynamicGridW: .seqLenMul(nHeads * (headDim / 2))))
        }

        // scale_residual_tc_f32 (has_scale=0 → out = residual + x): x@0 res@1 scale@2 out@3
        //   | channels@4 frames@5 has_scale@6 | threads seqLen × hidden, tg 32×1.
        func residual(x: Int, comment: String) {
            emit(SmeltDispatch(
                pipeline: bf16Activation ? .scaleResidualTCBF16 : .scaleResidualTCF32,
                buffers: [SmeltBufferBinding(slot: x, index: 0),
                          SmeltBufferBinding(slot: .variable("cur"), index: 1),
                          dummy,
                          SmeltBufferBinding(slot: .variable("alt"), index: 3)],
                constants: [u32(hidden, 4), seqLen(5), u32(0, 6)],
                dispatch: .threads(width: 1, height: hidden, depth: 1, tgWidth: 32, tgHeight: 1, tgDepth: 1),
                comment: comment, dynamicGridW: .seqLen))
            emitSwap()
        }

        for layer in 0..<cfg.numLayers {
            let p = "layers_\(layer)"
            let qW = try proj("\(p)_self_attn_q_proj_weight")
            let kW = try proj("\(p)_self_attn_k_proj_weight")
            let vW = try proj("\(p)_self_attn_v_proj_weight")
            let oW = try proj("\(p)_self_attn_o_proj_weight")
            let gateW = try proj("\(p)_mlp_gate_proj_weight")
            let upW = try proj("\(p)_mlp_up_proj_weight")
            let downW = try proj("\(p)_mlp_down_proj_weight")
            let inputNormW = try norm("\(p)_input_layernorm_weight")
            let postAttnNormW = try norm("\(p)_post_attention_layernorm_weight")
            let qNormW = try norm("\(p)_self_attn_q_norm_weight")
            let kNormW = try norm("\(p)_self_attn_k_norm_weight")
            let keyCache = plan.keyCacheBaseSlot + layer
            let valCache = plan.valCacheBaseSlot + layer

            rmsNormCodec(x: SmeltBufferBinding(slot: .variable("cur"), index: 0),
                         weight: inputNormW, out: normOut, comment: "L\(layer) input norm")

            try gemm(x: normOut, w: qW, out: SmeltBufferBinding(slot: qBuf, index: 3),
                 n: qDim, k: hidden, comment: "L\(layer) q proj")
            try gemm(x: normOut, w: kW, out: SmeltBufferBinding(slot: kBuf, index: 3),
                 n: kvDim, k: hidden, comment: "L\(layer) k proj")
            // V written directly into cache rows [startPos, startPos+frames).
            try gemm(x: normOut, w: vW,
                 out: SmeltBufferBinding(slot: valCache, offset: kvRowStride, offsetKind: 1, index: 3),
                 n: kvDim, k: hidden, comment: "L\(layer) v proj → cache rows")

            headNorm(buf: qBuf, weight: qNormW, nHeads: heads, comment: "L\(layer) q norm")
            headNorm(buf: kBuf, weight: kNormW, nHeads: kvHeads, comment: "L\(layer) k norm")
            rope(x: qBuf, out: SmeltBufferBinding(slot: qBuf, index: 3), nHeads: heads,
                 comment: "L\(layer) q rope")
            // Post-RoPE K written directly into cache rows.
            rope(x: kBuf,
                 out: SmeltBufferBinding(slot: keyCache, offset: kvRowStride, offsetKind: 1, index: 3),
                 nHeads: kvHeads, comment: "L\(layer) k rope → cache rows")

            // causal_gqa_attn_cached_f32: q@0 (chunk-local) kCache@1 vCache@2 (full)
            //   out@3 | frames@4 heads@5 kvHeads@6 headDim@7 startPos@8 | threads
            //   chunkLen*32 × heads, tg 32×1. Chunk-local query t attends absolute
            //   cache rows [0, startPos+t] — the cross-chunk prefill (B3.2b). At
            //   startPos 0 it is byte-identical to causal_gqa_attn_simd_f32 (W2).
            emit(SmeltDispatch(
                pipeline: bf16Activation
                    ? .causalGQAAttnCachedBF16
                    : .causalGQAAttnCachedF32,
                buffers: [SmeltBufferBinding(slot: qBuf, index: 0),
                          SmeltBufferBinding(slot: keyCache, index: 1),
                          SmeltBufferBinding(slot: valCache, index: 2),
                          SmeltBufferBinding(slot: attnOut, index: 3)],
                constants: [seqLen(4), u32(heads, 5), u32(kvHeads, 6), u32(headDim, 7), startPos(8)],
                dispatch: .threads(width: 32, height: heads, depth: 1, tgWidth: 32, tgHeight: 1, tgDepth: 1),
                comment: "L\(layer) causal GQA attention (cross-chunk)", dynamicGridW: .seqLenMul(32)))

            try gemm(x: attnOut, w: oW, out: SmeltBufferBinding(slot: normOut, index: 3),
                 n: hidden, k: qDim, comment: "L\(layer) o proj")
            residual(x: normOut, comment: "L\(layer) attn residual")

            rmsNormCodec(x: SmeltBufferBinding(slot: .variable("cur"), index: 0),
                         weight: postAttnNormW, out: normOut, comment: "L\(layer) post-attn norm")

            try gemm(x: normOut, w: gateW, out: SmeltBufferBinding(slot: gateBuf, index: 3),
                 n: inter, k: hidden, comment: "L\(layer) gate proj")
            try gemm(x: normOut, w: upW, out: SmeltBufferBinding(slot: upBuf, index: 3),
                 n: inter, k: hidden, comment: "L\(layer) up proj")
            // swiglu_f32: gate@0 up@1 out@2 | count@3 | threads seqLen*inter ÷ 32.
            emit(SmeltDispatch(
                pipeline: bf16Activation ? .swigluBF16 : .swigluF32,
                buffers: [SmeltBufferBinding(slot: gateBuf, index: 0),
                          SmeltBufferBinding(slot: upBuf, index: 1),
                          SmeltBufferBinding(slot: swigluOut, index: 2)],
                constants: [seqLenMul(inter, 3)],
                dispatch: .threads(width: 32, height: 1, depth: 1, tgWidth: 32, tgHeight: 1, tgDepth: 1),
                comment: "L\(layer) swiglu", dynamicGridW: .seqLenMul(inter)))
            try gemm(x: swigluOut, w: downW, out: SmeltBufferBinding(slot: normOut, index: 3),
                 n: hidden, k: inter, comment: "L\(layer) down proj")
            residual(x: normOut, comment: "L\(layer) FFN residual")
        }

        // Hidden-output port: final norm → normOutBuf. No LM head / argmax / selection.
        rmsNormCodec(x: SmeltBufferBinding(slot: .variable("cur"), index: 0),
                     weight: try norm("norm_weight"), out: normOut,
                     comment: "final norm → normOutBuf (hidden output)")

        return PrefillEmitter.GenerateResult(
            dispatchRecords: records,
            traceMarkers: [],
            gptqCapturePoints: capturePoints,
            pipelineNames: SmeltKernelCatalog.pipelineNames,
            namedPipelineNames: [],
            namedPipelineUses: [],
            optimizationStats: SmeltOptimizationStats())
    }
}
