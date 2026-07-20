// Qwen3TTSCodecEmitter — the real `Sources/` codec emitter (C2-FULL, docs/codec-c2-full-plan.md).
// Lifts the byte-identical-gated C2-A/B/C test-helper record builders into a pure two-phase planner:
// `plan(frames:config:)` returns the WHOLE `decodeCodec` graph (RVQ front → output_proj → pre_conv →
// transpose → 8-layer pre_transformer → transpose → vocoder tail) as a flat `[SmeltDispatchRecord]`
// table plus, per buffer slot, a `CodecSlotDesc` telling the caller what to put there. The records are
// a pure function of `(frames, topology)`: weights flow through buffer SLOTS, never constants. The
// caller (`Qwen3TTSGPUCodec.decodeCodec`) realizes the slots → `[MTLBuffer]` and runs the table through
// `SmeltCodecRecordRunner`. It is the SOLE codec decode path — built bit-exact (same kernels/bindings/
// grids) with the original hand decode it replaced (C2/C3), which is now retired.
//
// The transposes are GPU `transpose_f32` records (the original hand decode used a CPU readback, illegal
// inside the single-encoder table); bit-exactness rests on `testTransposeF32MatchesCPU`. All codec matmuls run at
// M=frames>1, K%4==0, so they bake `gemm_tn_f32` — exactly what `matmulDispatch` selects.

import SmeltSchema

private typealias B = SmeltCodecRecordBuilder

/// What buffer must occupy a given slot in the realized `[MTLBuffer]`. The caller materializes these.
public enum CodecSlotDesc: Equatable {
    case weight(String)        // raw checkpoint tensor → bufF32(f32(name))
    case codes                 // [Int32] input codes, row-major [16, frames]; rvq_gather_sum idx 0
    case codebookFirst         // derived: embedding_sum / max(cluster_usage, 1e-5), rvq_first  (idx 1)
    case codebookRestFlat      // derived: all 15 rvq_rest codebookEmbeddings concatenated into one buffer (idx 2)
    case projConcat            // derived: interleaved concat(firstProj, restProj) [outputProjDim, 2*rvqDim, 1]
    case ropeCos               // derived host: ropeTables(frames, ptHeadDim).cos
    case ropeSin               // derived host: ropeTables(frames, ptHeadDim).sin
    case ones(Int)             // [1,1,…] residual scale (count floats)
    case scratch(Int)          // zeroed float buffer (count floats): channel_copy pad dest / intermediate / zero-bias
}

/// Fixed Qwen3-TTS-12Hz codec topology. Architectural constants are defaulted; the genuinely
/// checkpoint-derived dims (`restN`, `firstN`, `upsampleInter`) are caller-supplied and the realizer
/// validates the loaded tensor shapes against them.
public struct Qwen3TTSCodecConfig {
    public var rvqDim = 256
    public var restCount = 15
    public var restN: Int            // rest-codebook vocab (rvq_gather_sum constant idx 7)
    public var firstN: Int           // first-codebook vocab (codes OOB bound, realizer-side)
    public var outputProjDim = 512   // output_proj cOut (= 2*rvqDim)
    public var preConvDim = 1024     // pre_conv cOut (= pre_transformer latent)
    public var ptHidden = 512
    public var ptHeads = 16
    public var ptHeadDim = 64
    public var ptInter = 1024
    public var ptWindow = 72
    public var ptLayers = 8
    public var upsampleInter: [Int]  // convNeXt inter (pwconv1 out) per upsample block [2]
    public var conv0Out = 1536
    public var dacRates = [8, 5, 4, 3]
    public var dacDilations = [1, 3, 9]

    public init(restN: Int, firstN: Int, upsampleInter: [Int]) {
        self.restN = restN; self.firstN = firstN; self.upsampleInter = upsampleInter
    }

    public var ptLatent: Int { preConvDim }              // pre_transformer latent == pre_conv channels
    public var ptAttnDim: Int { ptHeads * ptHeadDim }    // 1024
    public var dacOutDims: [Int] { (0..<dacRates.count).map { conv0Out >> ($0 + 1) } }  // 768,384,192,96
}

/// The standalone codec record table + the slot descriptors the caller must realize.
public struct Qwen3TTSCodecPlan {
    public let records: [SmeltDispatchRecord]
    public let slots: [CodecSlotDesc]
    public let pipelineNames: [String]   // dense local index: record.pipeline → metal-name → pso(name)
    public let outputSlot: Int
    public let outputLength: Int         // == 1920 * frames
}

public enum Qwen3TTSCodecEmitter {

    // Dense local pipeline index. `pipelineNames[k.rawValue]` is the metal function name; the caller
    // builds `pipelines = pipelineNames.map { pso($0) }`.
    private enum K: Int, CaseIterable {
        case rvq = 0, chcopy, conv1d, transconv, lnct, gelu, sres, snake, clamp, transpose, rms, gemm, rope, attn, swiglu, srtc
    }
    public static let pipelineNames: [String] = [
        "rvq_gather_sum_f32", "channel_copy_f32", "conv1d_forward_f32", "conv_transpose1d_f32",
        "layer_norm_ct_f32", "gelu_f32", "scale_residual_f32", "snake_beta_f32", "clamp_f32",
        "transpose_f32", "rms_norm_codec_f32", "gemm_tn_f32", "rope_apply_f32",
        "sliding_attn_simd_f32", "swiglu_f32", "scale_residual_tc_f32",
    ]
    // gemm_tn tiling (matches matmulDispatch's gemmTNFeatures / gemmTNTileM).
    private static let gemmTNFeatures = 4
    private static let gemmTNTileM = 16

    /// Build the whole-codec record table for `frames` codec frames.
    public static func plan(frames: Int, config c: Qwen3TTSCodecConfig) -> Qwen3TTSCodecPlan {
        precondition(pipelineNames.count == K.allCases.count, "codec pipelineNames out of sync with K")

        var slots: [CodecSlotDesc] = []
        var records: [SmeltDispatchRecord] = []
        func add(_ d: CodecSlotDesc) -> Int { slots.append(d); return slots.count - 1 }
        func w(_ name: String) -> Int { add(.weight(name)) }
        func scratch(_ count: Int) -> Int { add(.scratch(count)) }

        // u32 constant list with ascending binding indices from `from`.
        func u32(_ vals: [Int], from idx: Int) -> [B.Const] {
            vals.enumerated().map { .u32(UInt32($0.element), idx + $0.offset) }
        }

        // ---- stride-1 causal conv geometry (the bit-exact length anchor) ----
        func convPad(_ lengthIn: Int, kernel: Int, dilation: Int) -> (padded: Int, pad: Int, out: Int) {
            let kEff = (kernel - 1) * dilation + 1
            let padding = kEff - 1
            let nFrames = Int(Double(lengthIn - kEff + padding).rounded(.up)) + 1
            let ideal = (nFrames - 1) + (kEff - padding)
            let padded = lengthIn + padding + max(0, ideal - lengthIn)
            return (padded, padding, (padded - kEff) + 1)
        }

        // ---- conv1dCausal as records: channel_copy (causal pad) + conv1d_forward ----
        func conv1dCausal(cIn: Int, lengthIn: Int, cOut: Int, kernel: Int, dilation: Int, groups: Int,
                          inSlot: Int, outSlot: Int, wSlot: Int, bSlot: Int) {
            let p = convPad(lengthIn, kernel: kernel, dilation: dilation)
            let padSlot = scratch(cIn * p.padded)
            records.append(B.threads(pipeline: K.chcopy.rawValue, buffers: [(inSlot, 0), (padSlot, 1)],
                constants: u32([cIn, lengthIn, p.padded, 0, p.pad, lengthIn], from: 2),
                grid: (lengthIn, cIn, 1), tg: (min(lengthIn, 32), 1, 1)))
            records.append(B.threads(pipeline: K.conv1d.rawValue, buffers: [(padSlot, 0), (wSlot, 1), (bSlot, 2), (outSlot, 3)],
                constants: [.u32(UInt32(cIn), 4), .u32(UInt32(cOut), 5), .u32(UInt32(kernel), 6), .u32(1, 7),
                            .u32(0, 8), .u32(UInt32(dilation), 9), .u32(UInt32(groups), 10),
                            .u32(UInt32(p.padded) | 0x8000_0000, 11)],
                grid: (p.out, cOut, 1), tg: (min(p.out, 32), 1, 1)))
        }

        // ---- transConv as records: conv_transpose1d (+ channel_copy RIGHT-trim when kernel>stride) ----
        // Returns the result slot + lengthOut.
        func transConv(cIn: Int, lengthIn: Int, cOut: Int, kernel: Int, stride: Int, inSlot: Int, wSlot: Int, bSlot: Int) -> (slot: Int, len: Int) {
            let fullLen = (lengthIn - 1) * stride + kernel
            let trim = max(0, kernel - stride)
            let lengthOut = fullLen - trim
            let fullSlot = scratch(cOut * fullLen)
            records.append(B.threads(pipeline: K.transconv.rawValue, buffers: [(inSlot, 0), (wSlot, 1), (bSlot, 2), (fullSlot, 3)],
                constants: u32([cIn, cOut, kernel, stride, 0, lengthIn], from: 4),
                grid: (fullLen, cOut, 1), tg: (min(fullLen, 32), 1, 1)))
            if trim == 0 { return (fullSlot, lengthOut) }
            let outSlot = scratch(cOut * lengthOut)
            records.append(B.threads(pipeline: K.chcopy.rawValue, buffers: [(fullSlot, 0), (outSlot, 1)],
                constants: u32([cOut, fullLen, lengthOut, 0, 0, lengthOut], from: 2),
                grid: (lengthOut, cOut, 1), tg: (min(lengthOut, 32), 1, 1)))
            return (outSlot, lengthOut)
        }

        // ---- convNeXt as records: dwconv(k7) → layer_norm_ct → pwconv1(k1) → gelu → pwconv2(k1) → scale_residual ----
        // weight names under `prefix`: dwconv.conv.weight/.bias, norm.weight/.bias, pwconv1.weight/.bias, pwconv2.weight/.bias, gamma.
        func convNeXtBlock(dim: Int, frames f: Int, inter: Int, prefix: String, inSlot: Int, outSlot: Int) {
            let tg = (min(f, 32), 1, 1)
            let dwW = w("\(prefix).dwconv.conv.weight"), dwB = w("\(prefix).dwconv.conv.bias")
            let nW = w("\(prefix).norm.weight"), nB = w("\(prefix).norm.bias")
            let pw1W = w("\(prefix).pwconv1.weight"), pw1B = w("\(prefix).pwconv1.bias")
            let pw2W = w("\(prefix).pwconv2.weight"), pw2B = w("\(prefix).pwconv2.bias")
            let gamma = w("\(prefix).gamma")
            let dw = scratch(dim * f)
            conv1dCausal(cIn: dim, lengthIn: f, cOut: dim, kernel: 7, dilation: 1, groups: dim, inSlot: inSlot, outSlot: dw, wSlot: dwW, bSlot: dwB)
            let normed = scratch(dim * f)
            records.append(B.threads(pipeline: K.lnct.rawValue, buffers: [(dw, 0), (nW, 1), (nB, 2), (normed, 3)],
                constants: [.u32(UInt32(dim), 4), .u32(UInt32(f), 5), .f32(1e-6, 6)], grid: (f, 1, 1), tg: tg))
            let h1pre = scratch(inter * f)
            conv1dCausal(cIn: dim, lengthIn: f, cOut: inter, kernel: 1, dilation: 1, groups: 1, inSlot: normed, outSlot: h1pre, wSlot: pw1W, bSlot: pw1B)
            let h1 = scratch(inter * f)
            records.append(B.threads(pipeline: K.gelu.rawValue, buffers: [(h1pre, 0), (h1, 1)],
                constants: [.u32(UInt32(inter * f), 2)], grid: (inter * f, 1, 1), tg: (32, 1, 1)))
            let pw2out = scratch(dim * f)
            conv1dCausal(cIn: inter, lengthIn: f, cOut: dim, kernel: 1, dilation: 1, groups: 1, inSlot: h1, outSlot: pw2out, wSlot: pw2W, bSlot: pw2B)
            records.append(B.threads(pipeline: K.sres.rawValue, buffers: [(pw2out, 0), (inSlot, 1), (gamma, 2), (outSlot, 3)],
                constants: u32([dim, f], from: 4), grid: (f, dim, 1), tg: tg))
        }

        // ---- DAC residual unit as records (snake → conv1 k7/dil → snake → conv2 k1 → scale_residual(ones)) ----
        func residualUnit(dim: Int, frames f: Int, dilation: Int, prefix: String, inSlot: Int, outSlot: Int) {
            let tg = (min(f, 32), 1, 1)
            let a1 = w("\(prefix).act1.alpha"), b1 = w("\(prefix).act1.beta")
            let c1W = w("\(prefix).conv1.conv.weight"), c1B = w("\(prefix).conv1.conv.bias")
            let a2 = w("\(prefix).act2.alpha"), b2 = w("\(prefix).act2.beta")
            let c2W = w("\(prefix).conv2.conv.weight"), c2B = w("\(prefix).conv2.conv.bias")
            let onesSlot = add(.ones(dim))
            let s1 = scratch(dim * f)
            records.append(B.threads(pipeline: K.snake.rawValue, buffers: [(inSlot, 0), (a1, 1), (b1, 2), (s1, 3)],
                constants: u32([dim, f], from: 4), grid: (f, dim, 1), tg: tg))
            let c1 = scratch(dim * f)
            conv1dCausal(cIn: dim, lengthIn: f, cOut: dim, kernel: 7, dilation: dilation, groups: 1, inSlot: s1, outSlot: c1, wSlot: c1W, bSlot: c1B)
            let s2 = scratch(dim * f)
            records.append(B.threads(pipeline: K.snake.rawValue, buffers: [(c1, 0), (a2, 1), (b2, 2), (s2, 3)],
                constants: u32([dim, f], from: 4), grid: (f, dim, 1), tg: tg))
            let c2 = scratch(dim * f)
            conv1dCausal(cIn: dim, lengthIn: f, cOut: dim, kernel: 1, dilation: 1, groups: 1, inSlot: s2, outSlot: c2, wSlot: c2W, bSlot: c2B)
            records.append(B.threads(pipeline: K.sres.rawValue, buffers: [(c2, 0), (inSlot, 1), (onesSlot, 2), (outSlot, 3)],
                constants: u32([dim, f], from: 4), grid: (f, dim, 1), tg: tg))
        }

        // ====================== RVQ FRONT ======================
        let codesSlot = add(.codes)
        let firstSlot = add(.codebookFirst)
        let restSlot = add(.codebookRestFlat)
        let qSlot = scratch(2 * c.rvqDim * frames)
        records.append(B.threads(pipeline: K.rvq.rawValue, buffers: [(codesSlot, 0), (firstSlot, 1), (restSlot, 2), (qSlot, 3)],
            constants: u32([c.rvqDim, frames, c.restCount, c.restN], from: 4),
            grid: (frames, c.rvqDim, 1), tg: (min(frames, 32), 1, 1)))

        // output_proj k1 (weight = projConcat, zero bias) → [outputProjDim, frames]
        let projSlot = add(.projConcat)
        let projBias = scratch(c.outputProjDim)   // zeros, read-only
        let h1Slot = scratch(c.outputProjDim * frames)
        conv1dCausal(cIn: 2 * c.rvqDim, lengthIn: frames, cOut: c.outputProjDim, kernel: 1, dilation: 1, groups: 1,
            inSlot: qSlot, outSlot: h1Slot, wSlot: projSlot, bSlot: projBias)

        // pre_conv k3 → [preConvDim, frames]
        let h2Slot = scratch(c.preConvDim * frames)
        conv1dCausal(cIn: c.outputProjDim, lengthIn: frames, cOut: c.preConvDim, kernel: 3, dilation: 1, groups: 1,
            inSlot: h1Slot, outSlot: h2Slot, wSlot: w("decoder.pre_conv.conv.weight"), bSlot: w("decoder.pre_conv.conv.bias"))

        // ====================== PRE_TRANSFORMER ======================
        // transpose CT→TC: [preConvDim, frames] → [frames, preConvDim]
        func transpose(_ inSlot: Int, rows: Int, cols: Int) -> Int {
            let o = scratch(rows * cols)
            records.append(B.threads(pipeline: K.transpose.rawValue, buffers: [(inSlot, 0), (o, 1)],
                constants: u32([rows, cols], from: 2), grid: (cols, rows, 1), tg: (min(cols, 32), 1, 1)))
            return o
        }
        let cosSlot = add(.ropeCos), sinSlot = add(.ropeSin)
        let zBias = scratch(max(c.ptAttnDim, c.ptInter))   // shared zero-bias for nil-bias gemms

        func rRms(_ xS: Int, _ name: String) -> Int {
            let o = scratch(frames * c.ptHidden)
            records.append(B.threads(pipeline: K.rms.rawValue, buffers: [(xS, 0), (w(name), 1), (o, 2)],
                constants: [.u32(UInt32(frames), 3), .u32(UInt32(c.ptHidden), 4), .f32(1e-5, 5)],
                grid: (frames * 32, 1, 1), tg: (32, 1, 1)))
            return o
        }
        func rMm(_ xS: Int, M: Int, Kdim: Int, _ name: String, bias: String?, N: Int) -> Int {
            let o = scratch(M * N)
            let nGroups = (N + gemmTNFeatures - 1) / gemmTNFeatures
            let mRows = (M + gemmTNTileM - 1) / gemmTNTileM * gemmTNTileM
            let biasSlot = bias.map { w($0) } ?? zBias
            records.append(B.threads(pipeline: K.gemm.rawValue, buffers: [(xS, 0), (w(name), 1), (biasSlot, 2), (o, 3)],
                constants: u32([M, N, Kdim, bias == nil ? 0 : 1], from: 4),
                grid: (nGroups * 32, mRows, 1), tg: (32, gemmTNTileM, 1)))
            return o
        }
        func rRope(_ xS: Int) -> Int {
            let o = scratch(frames * c.ptHeads * c.ptHeadDim)
            let total = frames * c.ptHeads * (c.ptHeadDim / 2)
            records.append(B.threads(pipeline: K.rope.rawValue, buffers: [(xS, 0), (cosSlot, 1), (sinSlot, 2), (o, 3)],
                constants: u32([frames, c.ptHeads, c.ptHeadDim], from: 4), grid: (total, 1, 1), tg: (min(total, 256), 1, 1)))
            return o
        }
        func rAttn(_ qS: Int, _ kS: Int, _ vS: Int) -> Int {
            let o = scratch(frames * c.ptHeads * c.ptHeadDim)
            records.append(B.threads(pipeline: K.attn.rawValue, buffers: [(qS, 0), (kS, 1), (vS, 2), (o, 3)],
                constants: u32([frames, c.ptHeads, c.ptHeadDim, c.ptWindow], from: 4),
                grid: (frames * 32, c.ptHeads, 1), tg: (32, 1, 1)))
            return o
        }
        func rSwi(_ gS: Int, _ uS: Int) -> Int {
            let o = scratch(frames * c.ptInter)
            records.append(B.threads(pipeline: K.swiglu.rawValue, buffers: [(gS, 0), (uS, 1), (o, 2)],
                constants: u32([frames * c.ptInter], from: 3), grid: (frames * c.ptInter, 1, 1), tg: (32, 1, 1)))
            return o
        }
        func rSrtc(_ xS: Int, _ resS: Int, _ name: String) -> Int {
            let o = scratch(frames * c.ptHidden)
            records.append(B.threads(pipeline: K.srtc.rawValue, buffers: [(xS, 0), (resS, 1), (w(name), 2), (o, 3)],
                constants: u32([c.ptHidden, frames, 1], from: 4), grid: (frames, c.ptHidden, 1), tg: (min(frames, 32), 1, 1)))
            return o
        }
        func layer(_ h0: Int, _ i: Int) -> Int {
            let p = "decoder.pre_transformer.layers.\(i)."
            let nrm = rRms(h0, p + "input_layernorm.weight")
            let q = rRope(rMm(nrm, M: frames, Kdim: c.ptHidden, p + "self_attn.q_proj.weight", bias: nil, N: c.ptAttnDim))
            let k = rRope(rMm(nrm, M: frames, Kdim: c.ptHidden, p + "self_attn.k_proj.weight", bias: nil, N: c.ptAttnDim))
            let v = rMm(nrm, M: frames, Kdim: c.ptHidden, p + "self_attn.v_proj.weight", bias: nil, N: c.ptAttnDim)
            let ar = rAttn(q, k, v)
            let h1 = rSrtc(rMm(ar, M: frames, Kdim: c.ptAttnDim, p + "self_attn.o_proj.weight", bias: nil, N: c.ptHidden), h0, p + "self_attn_layer_scale.scale")
            let nrm2 = rRms(h1, p + "post_attention_layernorm.weight")
            let act = rSwi(rMm(nrm2, M: frames, Kdim: c.ptHidden, p + "mlp.gate_proj.weight", bias: nil, N: c.ptInter),
                           rMm(nrm2, M: frames, Kdim: c.ptHidden, p + "mlp.up_proj.weight", bias: nil, N: c.ptInter))
            return rSrtc(rMm(act, M: frames, Kdim: c.ptInter, p + "mlp.down_proj.weight", bias: nil, N: c.ptHidden), h1, p + "mlp_layer_scale.scale")
        }

        let ptIn = transpose(h2Slot, rows: c.ptLatent, cols: frames)               // CT→TC
        var hs = rMm(ptIn, M: frames, Kdim: c.ptLatent, "decoder.pre_transformer.input_proj.weight", bias: "decoder.pre_transformer.input_proj.bias", N: c.ptHidden)
        for i in 0..<c.ptLayers { hs = layer(hs, i) }
        hs = rMm(rRms(hs, "decoder.pre_transformer.norm.weight"), M: frames, Kdim: c.ptHidden,
                 "decoder.pre_transformer.output_proj.weight", bias: "decoder.pre_transformer.output_proj.bias", N: c.ptLatent)
        var hSlot = transpose(hs, rows: frames, cols: c.ptLatent)                  // TC→CT  → [preConvDim, frames]
        var dim = c.ptLatent
        var len = frames

        // ====================== VOCODER TAIL ======================
        func snakeRec(_ inS: Int, _ aS: Int, _ bS: Int, _ outS: Int, _ d: Int, _ l: Int) {
            records.append(B.threads(pipeline: K.snake.rawValue, buffers: [(inS, 0), (aS, 1), (bS, 2), (outS, 3)],
                constants: u32([d, l], from: 4), grid: (l, d, 1), tg: (min(l, 32), 1, 1)))
        }

        // 2 upsample blocks: transConv k2 s2 (trim 0) + convNeXt
        for bi in 0..<c.upsampleInter.count {
            let (tcSlot, tl) = transConv(cIn: dim, lengthIn: len, cOut: dim, kernel: 2, stride: 2,
                inSlot: hSlot, wSlot: w("decoder.upsample.\(bi).0.conv.weight"), bSlot: w("decoder.upsample.\(bi).0.conv.bias"))
            let cnOut = scratch(dim * tl)
            convNeXtBlock(dim: dim, frames: tl, inter: c.upsampleInter[bi], prefix: "decoder.upsample.\(bi).1", inSlot: tcSlot, outSlot: cnOut)
            hSlot = cnOut; len = tl
        }

        // conv0 k7 → conv0Out
        let conv0Slot = scratch(c.conv0Out * len)
        conv1dCausal(cIn: dim, lengthIn: len, cOut: c.conv0Out, kernel: 7, dilation: 1, groups: 1,
            inSlot: hSlot, outSlot: conv0Slot, wSlot: w("decoder.decoder.0.conv.weight"), bSlot: w("decoder.decoder.0.conv.bias"))
        hSlot = conv0Slot; dim = c.conv0Out

        // 4 DAC blocks
        for i in 0..<c.dacRates.count {
            let p = "decoder.decoder.\(i + 1).block"
            let snakeOut = scratch(dim * len)
            snakeRec(hSlot, w("\(p).0.alpha"), w("\(p).0.beta"), snakeOut, dim, len)
            hSlot = snakeOut
            let rate = c.dacRates[i], oD = c.dacOutDims[i]
            let (tcSlot, tl) = transConv(cIn: dim, lengthIn: len, cOut: oD, kernel: 2 * rate, stride: rate,
                inSlot: hSlot, wSlot: w("\(p).1.conv.weight"), bSlot: w("\(p).1.conv.bias"))
            hSlot = tcSlot; dim = oD; len = tl
            for ri in 0..<c.dacDilations.count {
                let ruOut = scratch(dim * len)
                residualUnit(dim: dim, frames: len, dilation: c.dacDilations[ri], prefix: "\(p).\(ri + 2)", inSlot: hSlot, outSlot: ruOut)
                hSlot = ruOut
            }
        }

        // final snake → final conv k7 (→ 1ch) → clamp[-1,1]
        let fsOut = scratch(dim * len)
        snakeRec(hSlot, w("decoder.decoder.5.alpha"), w("decoder.decoder.5.beta"), fsOut, dim, len)
        hSlot = fsOut
        let convFSlot = scratch(len)
        conv1dCausal(cIn: dim, lengthIn: len, cOut: 1, kernel: 7, dilation: 1, groups: 1,
            inSlot: hSlot, outSlot: convFSlot, wSlot: w("decoder.decoder.6.conv.weight"), bSlot: w("decoder.decoder.6.conv.bias"))
        hSlot = convFSlot
        let clampOut = scratch(len)
        records.append(B.threads(pipeline: K.clamp.rawValue, buffers: [(hSlot, 0), (clampOut, 1)],
            constants: [.u32(UInt32(len), 2), .f32(-1, 3), .f32(1, 4)], grid: (len, 1, 1), tg: (32, 1, 1)))

        return Qwen3TTSCodecPlan(records: records, slots: slots, pipelineNames: pipelineNames,
                                 outputSlot: clampOut, outputLength: len)
    }
}
