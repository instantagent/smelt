// DenseTrunkEmitter — compiled dense headless-trunk decode lowering.
// Activation storage is an IR axis: BF16 and FP32 use the same graph, slots,
// bindings, and dispatch topology while selecting dtype variants of each
// operation kernel.
//
// Port shape: this table is born with the embeddings-in/hidden-out ports — `embeddings`
// input (hiddenA is pre-filled by the caller; no token gather; the tokenId
// argument is unused) and `hidden` output (the final norm lands in
// normOutBuf; no LM head, no argmax, no selection) — which sidesteps every
// token-in and terminal-head paths by design.
//
// Each layer is RMSNorm → Q/K/V projection → per-head norm + RoPE → GQA →
// output projection + residual → RMSNorm → gate/up + SwiGLU → down projection
// + residual. Two swaps keep cur == hiddenA at every layer entry.
//
// Dtype discipline (the Phase-12 lesson): projections are a valid kernel lego in
// bf16/fp16/fp32/affine_u4 where registered by MatvecKernelTable; a missing
// activation × weight × shape cell throws with no fallback.

import Foundation
import SmeltSchema

public enum DenseTrunkEmitterError: Error, CustomStringConvertible {
    case missingWeight(String)
    case wrongDtype(name: String, expected: String, got: String)
    case unsupported(String)

    public var description: String {
        switch self {
        case .missingWeight(let name):
            return "dense trunk: missing weight entry '\(name)'"
        case let .wrongDtype(name, expected, got):
            return "dense trunk: weight '\(name)' must be \(expected), got \(got) "
                + "(no silent fallback — repack the weight or fix the spec)"
        case .unsupported(let why):
            return "dense trunk: \(why)"
        }
    }
}

public enum DenseTrunkEmitter {

    /// Generate the decode table for a dense headless trunk.
    public static func generate(
        ir: SmeltModelIR,
        plan: SmeltBufferPlan,
        weightLayout: [SmeltWeightEntry]
    ) throws -> TopLevelEmitter.GenerateResult {
        let cfg = ir.config
        guard let attn = cfg.attentionConfigs[.attention] else {
            throw DenseTrunkEmitterError.unsupported("no attention config")
        }
        // Dense-topology guard set, shared with DenseTrunkPrefillEmitter so decode
        // and prefill can never disagree on the supported shape (the Phase-12
        // dispatch-safety lesson): every knob validateSmeltIR accepts but this
        // emitter does not realise throws loud, not silently mis-emits.
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

        // Projection entry, resolved through the shared DenseTrunkGuards resolver so
        // decode and prefill can never drift on which projection dtypes the trunk supports
        // (.dense(.bf16) → fused bf16 route, byte-identical; .affineU4 → unfused u4 route;
        // anything else is loud). resolveProjection validates the dtype; the layer loop
        // branches on the resolved kind to pick the route.
        func proj(_ name: String) throws -> SmeltWeightEntry {
            guard let entry = weights[name] else {
                throw DenseTrunkEmitterError.missingWeight(name)
            }
            if bf16Activation, entry.dtype != .bf16 {
                throw DenseTrunkEmitterError.wrongDtype(
                    name: name, expected: "bf16 for a bf16 activation trunk", got: "\(entry.dtype)")
            }
            _ = try DenseTrunkGuards.resolveProjection(
                entry,
                activationDtype: cfg.activationDtype,
                shape: .gemv,
                name: name
            )
            return entry
        }
        // Norm scales may use a wider or narrower storage dtype than the
        // activation graph. Dtype selects the otherwise-identical scale-load
        // kernel.
        func norm(_ name: String) throws -> SmeltWeightEntry {
            guard let entry = weights[name] else {
                throw DenseTrunkEmitterError.missingWeight(name)
            }
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

        let normOut = SmeltFixedSlot.normOutBuf.rawValue
        let qBuf = SmeltFixedSlot.attnQBuf.rawValue
        let kBuf = SmeltFixedSlot.attnKBuf.rawValue
        let qRoped = SmeltFixedSlot.attnOutBuf.rawValue
        let attnOut = SmeltFixedSlot.attnGateBuf.rawValue
        let ffnInt = SmeltFixedSlot.ffnIntBuf.rawValue
        let ropeStride = UInt64(headDim * activationBytes)
        let kvRowStride = kvDim * activationBytes

        func u32(_ value: Int, _ index: Int) -> SmeltConstantBinding {
            SmeltConstantBinding(expression: "\(value)", type: .uint32, index: index)
        }
        func threads1D(_ width: Int, tg: Int = 32) -> SmeltDispatchStyle {
            .threads(width: width, height: 1, depth: 1, tgWidth: tg, tgHeight: 1, tgDepth: 1)
        }
        // Bind a projection output: position-strided (a cache row) when `outExpr` is given, plain
        // otherwise. Shared by the unfused gemv emits (gemv_u4 @5, gemv_f32/f16w @3) so their
        // out-binding can't drift.
        func bindOut(_ slot: Int, _ outExpr: String?, index: Int) -> SmeltBufferBinding {
            outExpr.map { SmeltBufferBinding(slot: slot, offsetExpression: $0, index: index) }
                ?? SmeltBufferBinding(slot: slot, index: index)
        }

        var records: [SmeltDispatchRecord] = []
        func emit(_ dispatch: SmeltDispatch) { records.append(dispatch.toRecord()) }
        func emitSwap() {
            var rec = SmeltDispatchRecord.empty()
            rec.opKind = SmeltDispatchRecord.opSwap
            records.append(rec)
        }

        // rms_norm_codec_f32: x@0 w@1 out@2 | frames@3 dim@4 eps@5 | frames*32 ÷ 32
        func emitLayerNorm(input: SmeltSlotRef, weight: SmeltWeightEntry, comment: String) {
            emit(SmeltDispatch(
                pipeline: bf16Activation
                    ? .rmsNormCodecBF16
                    : (weight.dtype == .bf16 ? .rmsNormCodecBF16WF32 : .rmsNormCodecF32),
                buffers: [
                    SmeltBufferBinding(slot: input, index: 0),
                    SmeltBufferBinding(slot: weightsSlot, offset: weight.offset, index: 1),
                    SmeltBufferBinding(slot: normOut, index: 2),
                ],
                constants: [
                    u32(1, 3), u32(hidden, 4),
                    SmeltConstantBinding(expression: "\(eps)", type: .float32, index: 5),
                ],
                dispatch: threads1D(32),
                comment: comment
            ))
        }

        // --- Unfused u4 projection emits (Phase 3): mirror the hand u4 path exactly
        // (the staged dense decode oracle → matmulU4Dispatch / swigluDispatch /
        // scaleResidualTCDispatch), so a u4 compiled trunk is bit-exact vs hand.

        // gemv_u4_f32 (M=1): x@0 nibbles@1 scales@2 biases@3 bias@4(has_bias=0 unread) out@5
        //   | M=1@6 N@7 K@8 has_bias=0@9 group_size@10 | N*32 ÷ 32. Binds the 3 absolute
        // regions WeightLocator resolves; the dummy bias binds weights (large, valid, unread).
        func emitGemvU4(x: Int, entry: SmeltWeightEntry, outSlot: Int, outExpr: String?,
                        n: Int, k: Int, comment: String) throws {
            let loc = try WeightLocator.resolve(entry)
            guard case .affineU4(let groupSize) = loc.kind else {
                throw DenseTrunkEmitterError.unsupported("emitGemvU4 on non-u4 '\(entry.name)'")
            }
            let r = loc.regions   // [nibbles, scales, biases], absolute into the weights blob
            let outBinding = bindOut(outSlot, outExpr, index: 5)
            emit(SmeltDispatch(
                pipeline: .gemvU4F32,
                buffers: [
                    SmeltBufferBinding(slot: x, index: 0),
                    SmeltBufferBinding(slot: weightsSlot, offset: r[0].offset, index: 1),
                    SmeltBufferBinding(slot: weightsSlot, offset: r[1].offset, index: 2),
                    SmeltBufferBinding(slot: weightsSlot, offset: r[2].offset, index: 3),
                    SmeltBufferBinding(slot: weightsSlot, index: 4),
                    outBinding,
                ],
                constants: [u32(1, 6), u32(n, 7), u32(k, 8), u32(0, 9), u32(groupSize, 10)],
                dispatch: threads1D(n * 32),
                comment: comment
            ))
        }
        // scale_residual_tc_f32 (has_scale=0 → out = res + x): x@0 res(cur)@1 scale@2(unread)
        //   out(alt)@3 | channels@4 frames=1@5 has_scale=0@6 | grid (1, channels). + swap, so
        // `cur` holds the merged residual — the unfused twin of gemv_add's fused merge.
        func emitU4ResidualMerge(x: Int, comment: String) {
            emit(SmeltDispatch(
                pipeline: .scaleResidualTCF32,
                buffers: [
                    SmeltBufferBinding(slot: x, index: 0),
                    SmeltBufferBinding(slot: .variable("cur"), index: 1),
                    SmeltBufferBinding(slot: x, index: 2),
                    SmeltBufferBinding(slot: .variable("alt"), index: 3),
                ],
                constants: [u32(hidden, 4), u32(1, 5), u32(0, 6)],
                dispatch: .threads(width: 1, height: hidden, depth: 1,
                                   tgWidth: 1, tgHeight: 1, tgDepth: 1),
                comment: comment
            ))
            emitSwap()
        }
        // swiglu_f32: gate@0 up@1 out@2 | count@3 | count ÷ 32. out = silu(gate)*up.
        func emitSwiglu(gate: Int, up: Int, out: Int, count: Int, comment: String) {
            emit(SmeltDispatch(
                pipeline: .swigluF32,
                buffers: [
                    SmeltBufferBinding(slot: gate, index: 0),
                    SmeltBufferBinding(slot: up, index: 1),
                    SmeltBufferBinding(slot: out, index: 2),
                ],
                constants: [u32(count, 3)],
                dispatch: threads1D(count),
                comment: comment
            ))
        }
        // gemv_f32 / gemv_f16w_f32 (M=1, DENSE weight, no scale/bias): x@0 w@1 bias@2(has_bias=0
        //   unread) out@3 | M=1@4 N@5 K@6 has_bias=0@7 | N*32 ÷ 32. The unfused dense twin of
        // emitGemvU4 — mirrors matmulDispatch's f32/f16 route. Dummy bias binds weights (unread).
        func emitGemvDense(x: Int, entry: SmeltWeightEntry, dtype: SmeltDType, outSlot: Int,
                           outExpr: String?, n: Int, k: Int, comment: String) {
            precondition(dtype == .fp32 || dtype == .fp16, "emitGemvDense is fp32/fp16 only")
            let outBinding = bindOut(outSlot, outExpr, index: 3)
            emit(SmeltDispatch(
                pipeline: dtype == .fp16 ? .gemvF16WF32 : .gemvF32,
                buffers: [
                    SmeltBufferBinding(slot: x, index: 0),
                    SmeltBufferBinding(slot: weightsSlot, offset: entry.offset, index: 1),
                    SmeltBufferBinding(slot: weightsSlot, index: 2),
                    outBinding,
                ],
                constants: [u32(1, 4), u32(n, 5), u32(k, 6), u32(0, 7)],
                dispatch: threads1D(n * 32),
                comment: comment
            ))
        }
        // Unfused projection gemv dispatched by resolved kind: u4 → gemv_u4_f32 (3 regions);
        // dense fp32/fp16 → gemv_f32/gemv_f16w_f32 (1 region). bf16 never reaches here (it's fused),
        // and any other dense dtype is a loud throw here — explicit, not a downstream precondition trap.
        func emitProjGemv(x: Int, entry: SmeltWeightEntry, outSlot: Int, outExpr: String?,
                          n: Int, k: Int, comment: String) throws {
            switch try WeightLocator.resolve(entry).kind {
            case .affineU4:
                try emitGemvU4(x: x, entry: entry, outSlot: outSlot, outExpr: outExpr,
                               n: n, k: k, comment: comment)
            case .dense(.fp32):
                emitGemvDense(x: x, entry: entry, dtype: .fp32, outSlot: outSlot, outExpr: outExpr,
                              n: n, k: k, comment: comment)
            case .dense(.fp16):
                emitGemvDense(x: x, entry: entry, dtype: .fp16, outSlot: outSlot, outExpr: outExpr,
                              n: n, k: k, comment: comment)
            case .dense(let dt):
                throw DenseTrunkEmitterError.wrongDtype(
                    name: entry.name, expected: "fp32/fp16 unfused (bf16 is fused)", got: "\(dt)")
            case .signed(let dt, _):
                throw DenseTrunkEmitterError.wrongDtype(
                    name: entry.name, expected: "fp32/fp16/bf16 or affine_u4", got: dt.rawValue)
            }
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
            // The projection route is per-layer uniform — assert it (a within-layer dtype mix
            // would mis-bind through the kind-branched emit) and branch on the shared kind.
            let projKind = try DenseTrunkGuards.uniformProjectionKind([
                ("\(p)_self_attn_q_proj_weight", qW), ("\(p)_self_attn_k_proj_weight", kW),
                ("\(p)_self_attn_v_proj_weight", vW), ("\(p)_self_attn_o_proj_weight", oW),
                ("\(p)_mlp_gate_proj_weight", gateW), ("\(p)_mlp_up_proj_weight", upW),
                ("\(p)_mlp_down_proj_weight", downW),
            ], activationDtype: cfg.activationDtype, shape: .gemv)
            // bf16 → fused route; u4 / dense(fp32,fp16) → the unfused per-projection route (the
            // dense and u4 unfused paths share the same scaffold, differing only in the gemv kernel
            // emitProjGemv picks). Dtype selects a kernel lego, never gates the route's existence.
            let unfused: Bool = {
                switch projKind {
                case .affineU4: return true        // u4: per-projection gemv_u4_f32
                case .dense(.bf16): return false   // bf16: the fused multi-projection route
                case .dense: return true           // f32/f16: per-projection gemv_f32/gemv_f16w_f32
                case .signed: return true           // emitProjGemv rejects this fp32-only seam loudly
                }
            }()

            emitLayerNorm(input: .variable("cur"), weight: inputNormW,
                          comment: "L\(layer) input norm")

            let vIntoExpr = "Int(position) * \(kvRowStride)"
            if unfused {
                // Unfused qkv: 3 separate gemv (u4 → gemv_u4_f32, dense → gemv_f32/gemv_f16w_f32),
                // v → cache row, mirroring the hand else-branch (the staged dense oracle).
                try emitProjGemv(x: normOut, entry: qW, outSlot: qBuf, outExpr: nil,
                                 n: qDim, k: hidden, comment: "L\(layer) unfused q-proj")
                try emitProjGemv(x: normOut, entry: kW, outSlot: kBuf, outExpr: nil,
                                 n: kvDim, k: hidden, comment: "L\(layer) unfused k-proj")
                try emitProjGemv(x: normOut, entry: vW, outSlot: valCache, outExpr: vIntoExpr,
                                 n: kvDim, k: hidden, comment: "L\(layer) unfused v-proj → cache row")
            } else {
                // gemv_qkv_bf16w_f32: x@0 qW@1 kW@2 vW@3 outQ@4 outK@5 vInto@6
                //   | nq@7 nk@8 nv@9 K@10 | (nq+nk+nv)*32 ÷ 32. V lands in its
                // cache row directly (position-strided) — the hand path's shape.
                emit(SmeltDispatch(
                    pipeline: bf16Activation ? .gemvQKVBF16 : .gemvQKVBF16WF32,
                    buffers: [
                        SmeltBufferBinding(slot: normOut, index: 0),
                        SmeltBufferBinding(slot: weightsSlot, offset: qW.offset, index: 1),
                        SmeltBufferBinding(slot: weightsSlot, offset: kW.offset, index: 2),
                        SmeltBufferBinding(slot: weightsSlot, offset: vW.offset, index: 3),
                        SmeltBufferBinding(slot: qBuf, index: 4),
                        SmeltBufferBinding(slot: kBuf, index: 5),
                        SmeltBufferBinding(slot: valCache, offsetExpression: vIntoExpr, index: 6),
                    ],
                    constants: [u32(qDim, 7), u32(kvDim, 8), u32(kvDim, 9), u32(hidden, 10)],
                    dispatch: threads1D((qDim + kvDim + kvDim) * 32),
                    comment: "L\(layer) fused qkv (v → cache row)"
                ))
            }

            // head_norm_rope_f32: x@0 normW@1 cos@2 sin@3 out@4
            //   | heads@5 headDim@6 eps@7 | heads*32 ÷ 32
            // The fp32 rope tables (W0.1) bind position-strided rows.
            func headNormRope(x: Int, normW: SmeltWeightEntry, nHeads: Int,
                              out: Int, outExpr: String?, comment: String) {
                var bufs = [
                    SmeltBufferBinding(slot: x, index: 0),
                    SmeltBufferBinding(slot: weightsSlot, offset: normW.offset, index: 1),
                    SmeltBufferBinding(slot: plan.ropeCosSlot,
                                       offsetExpression: "Int(position) * \(ropeStride)", index: 2),
                    SmeltBufferBinding(slot: plan.ropeSinSlot,
                                       offsetExpression: "Int(position) * \(ropeStride)", index: 3),
                ]
                if let outExpr {
                    bufs.append(SmeltBufferBinding(slot: out, offsetExpression: outExpr, index: 4))
                } else {
                    bufs.append(SmeltBufferBinding(slot: out, index: 4))
                }
                emit(SmeltDispatch(
                    pipeline: bf16Activation
                        ? .headNormRopeBF16
                        : (normW.dtype == .bf16
                            ? .headNormRopeBF16WF32
                            : .headNormRopeF32),
                    buffers: bufs,
                    constants: [
                        u32(nHeads, 5), u32(headDim, 6),
                        SmeltConstantBinding(expression: "\(eps)", type: .float32, index: 7),
                    ],
                    dispatch: threads1D(nHeads * 32),
                    comment: comment
                ))
            }
            headNormRope(x: qBuf, normW: qNormW, nHeads: heads,
                         out: qRoped, outExpr: nil, comment: "L\(layer) q norm+rope")
            headNormRope(x: kBuf, normW: kNormW, nHeads: kvHeads,
                         out: keyCache, outExpr: "Int(position) * \(kvRowStride)",
                         comment: "L\(layer) k norm+rope → cache row")

            // decode_gqa_attn_f32: q@0 kCache@1 vCache@2 out@3
            //   | cacheLen@4 (= position+1) heads@5 kvHeads@6 headDim@7
            emit(SmeltDispatch(
                pipeline: bf16Activation ? .decodeGQAAttnBF16 : .decodeGQAAttnF32,
                buffers: [
                    SmeltBufferBinding(slot: qRoped, index: 0),
                    SmeltBufferBinding(slot: keyCache, index: 1),
                    SmeltBufferBinding(slot: valCache, index: 2),
                    SmeltBufferBinding(slot: attnOut, index: 3),
                ],
                constants: [
                    SmeltConstantBinding(expression: "UInt32(position + 1)",
                                         type: .uint32, index: 4),
                    u32(heads, 5), u32(kvHeads, 6), u32(headDim, 7),
                ],
                dispatch: threads1D(heads * 32),
                comment: "L\(layer) decode GQA attention"
            ))

            // gemv_add_bf16w_f32: x@0 w@1 res@2 out@3 | n@4 K@5 | n*32 ÷ 32
            func gemvAdd(x: Int, w: SmeltWeightEntry, n: Int, k: Int, comment: String) {
                emit(SmeltDispatch(
                    pipeline: bf16Activation ? .gemvAddBF16 : .gemvAddBF16WF32,
                    buffers: [
                        SmeltBufferBinding(slot: x, index: 0),
                        SmeltBufferBinding(slot: weightsSlot, offset: w.offset, index: 1),
                        SmeltBufferBinding(slot: .variable("cur"), index: 2),
                        SmeltBufferBinding(slot: .variable("alt"), index: 3),
                    ],
                    constants: [u32(n, 4), u32(k, 5)],
                    dispatch: threads1D(n * 32),
                    comment: comment
                ))
                emitSwap()
            }
            if unfused {
                // o-proj into normOut (free after qkv), then residual merge + swap.
                try emitProjGemv(x: attnOut, entry: oW, outSlot: normOut, outExpr: nil,
                                 n: hidden, k: qDim, comment: "L\(layer) unfused o-proj")
                emitU4ResidualMerge(x: normOut, comment: "L\(layer) unfused o-proj + residual")
            } else {
                gemvAdd(x: attnOut, w: oW, n: hidden, k: qDim,
                        comment: "L\(layer) o-proj + residual")
            }

            emitLayerNorm(input: .variable("cur"), weight: postAttnNormW,
                          comment: "L\(layer) post-attn norm")

            if unfused {
                // Unfused gate/up → swiglu. gate/up are inter-sized, so route them through
                // the inter-sized ffnGateBuf/ffnUpBuf (NOT the qDim-sized qBuf/kBuf).
                let gateBuf = SmeltFixedSlot.ffnGateBuf.rawValue
                let upBuf = SmeltFixedSlot.ffnUpBuf.rawValue
                try emitProjGemv(x: normOut, entry: gateW, outSlot: gateBuf, outExpr: nil,
                                 n: inter, k: hidden, comment: "L\(layer) unfused gate-proj")
                try emitProjGemv(x: normOut, entry: upW, outSlot: upBuf, outExpr: nil,
                                 n: inter, k: hidden, comment: "L\(layer) unfused up-proj")
                emitSwiglu(gate: gateBuf, up: upBuf, out: ffnInt, count: inter,
                           comment: "L\(layer) unfused swiglu")
                // down-proj into normOut (free after post-attn norm), then residual merge.
                try emitProjGemv(x: ffnInt, entry: downW, outSlot: normOut, outExpr: nil,
                                 n: hidden, k: inter, comment: "L\(layer) unfused down-proj")
                emitU4ResidualMerge(x: normOut, comment: "L\(layer) unfused down-proj + residual")
            } else {
                // gemv_gateup_swiglu_bf16w_f32: x@0 gateW@1 upW@2 out@3 | n@4 K@5
                emit(SmeltDispatch(
                    pipeline: bf16Activation
                        ? .gemvGateUpSwigluBF16
                        : .gemvGateUpSwigluBF16WF32,
                    buffers: [
                        SmeltBufferBinding(slot: normOut, index: 0),
                        SmeltBufferBinding(slot: weightsSlot, offset: gateW.offset, index: 1),
                        SmeltBufferBinding(slot: weightsSlot, offset: upW.offset, index: 2),
                        SmeltBufferBinding(slot: ffnInt, index: 3),
                    ],
                    constants: [u32(inter, 4), u32(hidden, 5)],
                    dispatch: threads1D(inter * 32),
                    comment: "L\(layer) fused gate/up + swiglu"
                ))
                gemvAdd(x: ffnInt, w: downW, n: hidden, k: inter,
                        comment: "L\(layer) down-proj + residual")
            }
        }

        // The hidden-output port: final norm → normOutBuf. No LM head, no
        // argmax, no selection — heads consume normOutBuf.
        emitLayerNorm(input: .variable("cur"), weight: try norm("norm_weight"),
                      comment: "final norm → normOutBuf (hidden output)")

        // The generated-Swift artifact is a documented stub: dense trunks are
        // packages are record-interpreted only.
        let source = """
        // --- Generated by smelt build (dense trunk). Do not edit. ---
        // This package's decode path is the binary dispatch table
        // (dispatches.bin) interpreted by SmeltRuntime; the dense trunk has
        // no generated-Swift encoder. Ports: embeddings in (hiddenA),
        // hidden out (normOutBuf).
        import Metal

        @inline(__always)
        internal func encodeDecodeStep(
            _ enc: MTLComputeCommandEncoder,
            _ p: UnsafeBufferPointer<MTLComputePipelineState>,
            _ b: UnsafeBufferPointer<MTLBuffer>,
            position: Int32,
            tokenId: Int32,
            cacheSeqCapacity: UInt32
        ) {
            preconditionFailure("dense trunk packages are record-interpreted only")
        }
        """

        return TopLevelEmitter.GenerateResult(
            source: source,
            dispatchRecords: records,
            traceMarkers: [],
            pipelineNames: SmeltKernelCatalog.pipelineNames,
            namedPipelineNames: [],
            namedPipelineUses: [],
            optimizationStats: SmeltOptimizationStats(),
            isHeadlessTrunkABI: true  // embeddings-in / hidden-out, no LM head (U3 marker)
        )
    }
}
