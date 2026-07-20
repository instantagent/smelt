// Qwen3TTSCodecStreamEmitter — the per-chunk record emitter for the STREAMING codec
// (docs/codec-c3-streaming-plan.md). The offline `Qwen3TTSCodecEmitter` compiled the fixed-shape
// `decodeCodec`; this compiles the chunked, stateful `Qwen3TTSCodecStream`. The decisive design is
// PER-CHUNK LITERAL RE-EMIT: once `(m, t0)` are fixed for a `decode()` call, EVERY length / grid /
// offset is known, so the chunk is "offline-static" in the C2 sense — emit a FRESH literal record
// table per chunk and run it through the UNCHANGED `SmeltCodecRecordRunner` (no dynamic kinds).
//
// Two crux problems the offline emitter never faced, both resolved here as plain records:
//   - Stateful: persistent cross-chunk caches (`preConv` / per-layer `ptInput` / `cn` / `d0` / `tc` /
//     `ru` / `c6`) are `.cache` SLOTS bound to the stream's own MTLBuffers — the concat (read old ‖
//     new → scratch) and update (scratch → cache) are ordinary `channel_copy_f32` records; within one
//     serial encoder, read-then-write ordering + Metal hazard tracking preserve the offline semantics
//     and the bytes persist to the next `decode()`.
//   - Frame-major `blitCopy` (illegal inside a compute encoder) → `channel_copy_f32` with channels=1
//     (a one-row strided copy degenerates to a contiguous copy — byte-for-byte the blit's movement).
//
// Matmul routing mirrors the resident `matmulDispatch` SELECTION per chunk M: `gemv_f32` at M==1 (the
// 1-frame first chunk → `sLen==1`), `gemm_tn_f32` at M>1 (K%4==0 always holds). The M==1 gemv record
// path is NEW (offline `frames` was always >1). Every grid/constant/binding is copied from the
// resident leaf dispatch it replaces, so the compiled chunk is bit-identical to that prior leaf decode.

import SmeltSchema

private typealias B = SmeltCodecRecordBuilder

/// Which persistent cross-chunk cache buffer a `.cache` slot binds to. The realizer (on the stream,
/// which owns the buffers) maps each id → its held MTLBuffer; the same id is read by a concat and
/// written by the matching update, so the emitter resolves it to ONE slot index per chunk.
public enum Qwen3TTSCodecCacheID: Hashable {
    case preConv, d0, c6
    case ptInput(Int)      // pre_transformer layer i input cache (≤71 rows, frame-major)
    case cn(Int)           // upsample block b ConvNeXt dwconv context (6 cols)
    case tc(Int)           // DAC stage i transposed-conv 1-col input history
    case ru(Int, Int)      // DAC stage i residual unit ri dilation context
}

/// What buffer occupies a streaming-codec slot. The stream's realizer materializes these against its
/// resident weights, pre-derived constants, persistent caches, and fresh per-chunk scratch.
public enum Qwen3TTSCodecStreamSlotDesc: Equatable {
    case weight(String)                       // codec weight → gpu.codecStreamWeight(name) (f32 raw; bf16/f16 widened)
    case codes                                // per-chunk [Int32] codebook-major [16, m]
    case codebookFirst                        // stream's derived rvq_first embedding
    case codebookRestFlat                     // stream's derived 15-rvq_rest embeddings, concatenated
    case projConcat                           // stream's derived output_proj concat
    case projZeroBias                         // stream's derived output_proj zero bias
    case ones                                 // stream's residual-unit unscaled-add ones
    case ropeCosSlice(start: Int, len: Int)   // cosT[start*headDim ..< (start+len)*headDim]
    case ropeSinSlice(start: Int, len: Int)
    case cache(Qwen3TTSCodecCacheID)
    case scratch(Int)                         // fresh zeroed per-chunk intermediate (count floats)
}

/// One chunk's record table + the slots the stream must realize.
public struct Qwen3TTSCodecStreamPlan {
    public let records: [SmeltDispatchRecord]
    public let slots: [Qwen3TTSCodecStreamSlotDesc]
    public let pipelineNames: [String]
    public let outputSlot: Int
    public let outputLength: Int               // == 1920 * m
}

public enum Qwen3TTSCodecStreamEmitter {

    // Dense local pipeline index (a superset of the offline emitter's — adds `gemv_f32` for M==1).
    private enum K: Int, CaseIterable {
        case rvq = 0, chcopy, conv1d, transconv, lnct, gelu, sres, snake, clamp, transpose, rms,
             gemm, gemv, rope, attn, swiglu, srtc
    }
    public static let pipelineNames: [String] = [
        "rvq_gather_sum_f32", "channel_copy_f32", "conv1d_forward_f32", "conv_transpose1d_f32",
        "layer_norm_ct_f32", "gelu_f32", "scale_residual_f32", "snake_beta_f32", "clamp_f32",
        "transpose_f32", "rms_norm_codec_f32", "gemm_tn_f32", "gemv_f32", "rope_apply_f32",
        "sliding_attn_simd_f32", "swiglu_f32", "scale_residual_tc_f32",
    ]
    // gemm_tn tiling (matches matmulDispatch's gemmTNFeatures / gemmTNTileM).
    private static let gemmTNFeatures = 4
    private static let gemmTNTileM = 16

    /// Build the record table for ONE `decode(m, t0)` chunk. `t0` is the global frames already
    /// consumed (== `framesDecoded`); `firstChunk == (t0 == 0)`.
    public static func plan(m: Int, t0: Int, config c: Qwen3TTSCodecConfig) -> Qwen3TTSCodecStreamPlan {
        precondition(pipelineNames.count == K.allCases.count, "stream pipelineNames out of sync with K")
        precondition(m > 0, "chunk must carry ≥1 frame")
        let isFirst = t0 == 0
        let window = c.ptWindow
        let hist = min(window - 1, t0)
        let sLen = hist + m
        let start = t0 - hist
        let H = c.ptHidden                          // 512: pre_transformer row width (frame-major rows)

        var slots: [Qwen3TTSCodecStreamSlotDesc] = []
        var records: [SmeltDispatchRecord] = []
        var cacheSlots: [Qwen3TTSCodecCacheID: Int] = [:]
        func add(_ d: Qwen3TTSCodecStreamSlotDesc) -> Int { slots.append(d); return slots.count - 1 }
        func scratch(_ n: Int) -> Int { add(.scratch(n)) }
        func w(_ name: String) -> Int { add(.weight(name)) }
        func cache(_ id: Qwen3TTSCodecCacheID) -> Int {
            if let s = cacheSlots[id] { return s }
            let s = add(.cache(id)); cacheSlots[id] = s; return s
        }

        func u32(_ vals: [Int], from idx: Int) -> [B.Const] {
            vals.enumerated().map { .u32(UInt32($0.element), idx + $0.offset) }
        }

        // Shared slots (one each, referenced throughout). zBias backs every nil-bias gemv/gemm_tn
        // (has_bias=0 → never read); the rope slice is the WHOLE [hist‖new] global-position window.
        let zBias = scratch(max(c.ptAttnDim, c.ptInter))
        let cosSlot = add(.ropeCosSlice(start: start, len: sLen))
        let sinSlot = add(.ropeSinSlice(start: start, len: sLen))
        let onesSlot = add(.ones)

        // ---- leaf record emitters (each mirrors the resident dispatch it replaces) ----

        // channel_copy_f32: per-channel strided copy (concat/update/emit-window). dst pre-zeroed.
        func chcopy(src: Int, dst: Int, channels: Int, srcLen: Int, dstLen: Int,
                    srcOff: Int, dstOff: Int, copyLen: Int) {
            guard copyLen > 0 else { return }
            records.append(B.threads(pipeline: K.chcopy.rawValue, buffers: [(src, 0), (dst, 1)],
                constants: u32([channels, srcLen, dstLen, srcOff, dstOff, copyLen], from: 2),
                grid: (copyLen, channels, 1), tg: (min(copyLen, 32), 1, 1)))
        }
        // Frame-major contiguous copy (the blit replacement): channels=1, srcLen/dstLen unread (c=0).
        func flatCopy(src: Int, dst: Int, srcOff: Int, dstOff: Int, copyLen: Int) {
            chcopy(src: src, dst: dst, channels: 1, srcLen: srcOff + copyLen, dstLen: dstOff + copyLen,
                   srcOff: srcOff, dstOff: dstOff, copyLen: copyLen)
        }
        // CT-layout cache concat: scratch s = [cache ‖ new]; returns s.
        func concatCT(_ id: Qwen3TTSCodecCacheID, cacheLen: Int, new: Int, newLen: Int, channels: Int) -> Int {
            let s = scratch(channels * (cacheLen + newLen))
            chcopy(src: cache(id), dst: s, channels: channels, srcLen: cacheLen, dstLen: cacheLen + newLen,
                   srcOff: 0, dstOff: 0, copyLen: cacheLen)
            chcopy(src: new, dst: s, channels: channels, srcLen: newLen, dstLen: cacheLen + newLen,
                   srcOff: 0, dstOff: cacheLen, copyLen: newLen)
            return s
        }
        func updateCT(_ id: Qwen3TTSCodecCacheID, from s: Int, channels: Int, sLen: Int, keep: Int) {
            chcopy(src: s, dst: cache(id), channels: channels, srcLen: sLen, dstLen: keep,
                   srcOff: sLen - keep, dstOff: 0, copyLen: keep)
        }
        // conv1dValid: conv1d_forward over a context-prefixed input (no pad). lengthOut = lengthIn-kEff+1.
        func conv1dValid(_ inSlot: Int, cIn: Int, lengthIn: Int, wSlot: Int, bSlot: Int,
                         cOut: Int, kernel: Int, dilation: Int, groups: Int) -> (Int, Int) {
            let kEff = (kernel - 1) * dilation + 1
            let lengthOut = lengthIn - kEff + 1
            let outSlot = scratch(cOut * lengthOut)
            records.append(B.threads(pipeline: K.conv1d.rawValue,
                buffers: [(inSlot, 0), (wSlot, 1), (bSlot, 2), (outSlot, 3)],
                constants: [.u32(UInt32(cIn), 4), .u32(UInt32(cOut), 5), .u32(UInt32(kernel), 6), .u32(1, 7),
                            .u32(0, 8), .u32(UInt32(dilation), 9), .u32(UInt32(groups), 10),
                            .u32(UInt32(lengthIn) | 0x8000_0000, 11)],
                grid: (lengthOut, cOut, 1), tg: (min(lengthOut, 32), 1, 1)))
            return (outSlot, lengthOut)
        }
        // conv_transpose1d FULL output (no trim) — the stream owns its emit window.
        func transConvRaw(_ inSlot: Int, cIn: Int, lengthIn: Int, wSlot: Int, bSlot: Int,
                          cOut: Int, kernel: Int, stride: Int) -> (Int, Int) {
            let fullLen = (lengthIn - 1) * stride + kernel
            let outSlot = scratch(cOut * fullLen)
            records.append(B.threads(pipeline: K.transconv.rawValue,
                buffers: [(inSlot, 0), (wSlot, 1), (bSlot, 2), (outSlot, 3)],
                constants: u32([cIn, cOut, kernel, stride, 0, lengthIn], from: 4),
                grid: (fullLen, cOut, 1), tg: (min(fullLen, 32), 1, 1)))
            return (outSlot, fullLen)
        }
        // matmul: M==1 → gemv_f32, M>1 → gemm_tn_f32 (K%4==0). biasSlot nil → zBias + has_bias=0.
        func matmul(_ inSlot: Int, M: Int, Kdim: Int, wSlot: Int, biasSlot: Int?, N: Int) -> Int {
            let outSlot = scratch(M * N)
            let hb = biasSlot == nil ? 0 : 1
            let bSlot = biasSlot ?? zBias
            if M == 1 {
                records.append(B.threads(pipeline: K.gemv.rawValue,
                    buffers: [(inSlot, 0), (wSlot, 1), (bSlot, 2), (outSlot, 3)],
                    constants: u32([1, N, Kdim, hb], from: 4),
                    grid: (N * 32, 1, 1), tg: (32, 1, 1)))
            } else {
                let nGroups = (N + gemmTNFeatures - 1) / gemmTNFeatures
                let mRows = (M + gemmTNTileM - 1) / gemmTNTileM * gemmTNTileM
                records.append(B.threads(pipeline: K.gemm.rawValue,
                    buffers: [(inSlot, 0), (wSlot, 1), (bSlot, 2), (outSlot, 3)],
                    constants: u32([M, N, Kdim, hb], from: 4),
                    grid: (nGroups * 32, mRows, 1), tg: (32, gemmTNTileM, 1)))
            }
            return outSlot
        }
        func rms(_ inSlot: Int, frames: Int, dim: Int, wSlot: Int) -> Int {
            let o = scratch(frames * dim)
            records.append(B.threads(pipeline: K.rms.rawValue, buffers: [(inSlot, 0), (wSlot, 1), (o, 2)],
                constants: [.u32(UInt32(frames), 3), .u32(UInt32(dim), 4), .f32(1e-5, 5)],
                grid: (frames * 32, 1, 1), tg: (32, 1, 1)))
            return o
        }
        func rope(_ inSlot: Int, frames: Int, heads: Int, headDim: Int) -> Int {
            let o = scratch(frames * heads * headDim)
            let total = frames * heads * (headDim / 2)
            records.append(B.threads(pipeline: K.rope.rawValue,
                buffers: [(inSlot, 0), (cosSlot, 1), (sinSlot, 2), (o, 3)],
                constants: u32([frames, heads, headDim], from: 4),
                grid: (total, 1, 1), tg: (min(total, 256), 1, 1)))
            return o
        }
        func attn(_ q: Int, _ k: Int, _ v: Int, frames: Int, heads: Int, headDim: Int, win: Int) -> Int {
            let o = scratch(frames * heads * headDim)
            records.append(B.threads(pipeline: K.attn.rawValue, buffers: [(q, 0), (k, 1), (v, 2), (o, 3)],
                constants: u32([frames, heads, headDim, win], from: 4),
                grid: (frames * 32, heads, 1), tg: (32, 1, 1)))
            return o
        }
        func swiglu(_ g: Int, _ u: Int, count: Int) -> Int {
            let o = scratch(count)
            records.append(B.threads(pipeline: K.swiglu.rawValue, buffers: [(g, 0), (u, 1), (o, 2)],
                constants: u32([count], from: 3), grid: (count, 1, 1), tg: (32, 1, 1)))
            return o
        }
        func srtc(_ x: Int, res: Int, scale: Int, channels: Int, frames: Int) -> Int {
            let o = scratch(frames * channels)
            records.append(B.threads(pipeline: K.srtc.rawValue, buffers: [(x, 0), (res, 1), (scale, 2), (o, 3)],
                constants: u32([channels, frames, 1], from: 4),
                grid: (frames, channels, 1), tg: (min(frames, 32), 1, 1)))
            return o
        }
        func snake(_ x: Int, dim: Int, frames: Int, alpha: Int, beta: Int) -> Int {
            let o = scratch(dim * frames)
            records.append(B.threads(pipeline: K.snake.rawValue, buffers: [(x, 0), (alpha, 1), (beta, 2), (o, 3)],
                constants: u32([dim, frames], from: 4), grid: (frames, dim, 1), tg: (min(frames, 32), 1, 1)))
            return o
        }
        func sres(_ x: Int, res: Int, scale: Int, channels: Int, frames: Int) -> Int {
            let o = scratch(channels * frames)
            records.append(B.threads(pipeline: K.sres.rawValue, buffers: [(x, 0), (res, 1), (scale, 2), (o, 3)],
                constants: u32([channels, frames], from: 4), grid: (frames, channels, 1), tg: (min(frames, 32), 1, 1)))
            return o
        }
        func lnct(_ x: Int, dim: Int, frames: Int, wSlot: Int, bSlot: Int) -> Int {
            let o = scratch(dim * frames)
            records.append(B.threads(pipeline: K.lnct.rawValue, buffers: [(x, 0), (wSlot, 1), (bSlot, 2), (o, 3)],
                constants: [.u32(UInt32(dim), 4), .u32(UInt32(frames), 5), .f32(1e-6, 6)],
                grid: (frames, 1, 1), tg: (min(frames, 32), 1, 1)))
            return o
        }
        func gelu(_ x: Int, count: Int) -> Int {
            let o = scratch(count)
            records.append(B.threads(pipeline: K.gelu.rawValue, buffers: [(x, 0), (o, 1)],
                constants: u32([count], from: 2), grid: (count, 1, 1), tg: (32, 1, 1)))
            return o
        }
        func transpose(_ inSlot: Int, rows: Int, cols: Int) -> Int {
            let o = scratch(rows * cols)
            records.append(B.threads(pipeline: K.transpose.rawValue, buffers: [(inSlot, 0), (o, 1)],
                constants: u32([rows, cols], from: 2), grid: (cols, rows, 1), tg: (min(cols, 32), 1, 1)))
            return o
        }

        // ====================== STAGE A: RVQ → output_proj → pre_conv ======================
        let codesSlot = add(.codes)
        let fSlot = add(.codebookFirst), rSlot = add(.codebookRestFlat)
        let qSlot = scratch(2 * c.rvqDim * m)
        records.append(B.threads(pipeline: K.rvq.rawValue,
            buffers: [(codesSlot, 0), (fSlot, 1), (rSlot, 2), (qSlot, 3)],
            constants: u32([c.rvqDim, m, c.restCount, c.restN], from: 4),
            grid: (m, c.rvqDim, 1), tg: (min(m, 32), 1, 1)))

        let projSlot = add(.projConcat), projBias = add(.projZeroBias)
        let (h0Slot, _) = conv1dValid(qSlot, cIn: 2 * c.rvqDim, lengthIn: m, wSlot: projSlot, bSlot: projBias,
                                      cOut: c.outputProjDim, kernel: 1, dilation: 1, groups: 1)
        let sPre = concatCT(.preConv, cacheLen: 2, new: h0Slot, newLen: m, channels: c.outputProjDim)
        let (preOut, _) = conv1dValid(sPre, cIn: c.outputProjDim, lengthIn: m + 2,
                                      wSlot: w("decoder.pre_conv.conv.weight"), bSlot: w("decoder.pre_conv.conv.bias"),
                                      cOut: c.preConvDim, kernel: 3, dilation: 1, groups: 1)
        updateCT(.preConv, from: sPre, channels: c.outputProjDim, sLen: m + 2, keep: 2)

        // ====================== STAGE B: pre_transformer over [hist ‖ new] ======================
        let ptIn = transpose(preOut, rows: c.preConvDim, cols: m)      // [preConvDim, m] → [m, preConvDim]
        var newRows = matmul(ptIn, M: m, Kdim: c.preConvDim,
                             wSlot: w("decoder.pre_transformer.input_proj.weight"),
                             biasSlot: w("decoder.pre_transformer.input_proj.bias"), N: H)
        var newRowsSrcOff = 0                                          // floats: newRows' trailing-m rows start here
        for i in 0..<c.ptLayers {
            let p = "decoder.pre_transformer.layers.\(i)."
            // s = [cache_i rows ‖ new rows] (frame-major ⇒ contiguous flat copies).
            let s = scratch(sLen * H)
            if hist > 0 { flatCopy(src: cache(.ptInput(i)), dst: s, srcOff: 0, dstOff: 0, copyLen: hist * H) }
            flatCopy(src: newRows, dst: s, srcOff: newRowsSrcOff, dstOff: hist * H, copyLen: m * H)
            let normed = rms(s, frames: sLen, dim: H, wSlot: w(p + "input_layernorm.weight"))
            let q = rope(matmul(normed, M: sLen, Kdim: H, wSlot: w(p + "self_attn.q_proj.weight"), biasSlot: nil, N: c.ptAttnDim),
                         frames: sLen, heads: c.ptHeads, headDim: c.ptHeadDim)
            let k = rope(matmul(normed, M: sLen, Kdim: H, wSlot: w(p + "self_attn.k_proj.weight"), biasSlot: nil, N: c.ptAttnDim),
                         frames: sLen, heads: c.ptHeads, headDim: c.ptHeadDim)
            let v = matmul(normed, M: sLen, Kdim: H, wSlot: w(p + "self_attn.v_proj.weight"), biasSlot: nil, N: c.ptAttnDim)
            let a = attn(q, k, v, frames: sLen, heads: c.ptHeads, headDim: c.ptHeadDim, win: window)
            let proj = matmul(a, M: sLen, Kdim: c.ptAttnDim, wSlot: w(p + "self_attn.o_proj.weight"), biasSlot: nil, N: H)
            var hS = srtc(proj, res: s, scale: w(p + "self_attn_layer_scale.scale"), channels: H, frames: sLen)
            let normed2 = rms(hS, frames: sLen, dim: H, wSlot: w(p + "post_attention_layernorm.weight"))
            let gate = matmul(normed2, M: sLen, Kdim: H, wSlot: w(p + "mlp.gate_proj.weight"), biasSlot: nil, N: c.ptInter)
            let up = matmul(normed2, M: sLen, Kdim: H, wSlot: w(p + "mlp.up_proj.weight"), biasSlot: nil, N: c.ptInter)
            let act = swiglu(gate, up, count: sLen * c.ptInter)
            let down = matmul(act, M: sLen, Kdim: c.ptInter, wSlot: w(p + "mlp.down_proj.weight"), biasSlot: nil, N: H)
            hS = srtc(down, res: hS, scale: w(p + "mlp_layer_scale.scale"), channels: H, frames: sLen)
            // Cache the trailing min(71, sLen) INPUT rows of s (NOT the output) for the next chunk.
            let keep = min(window - 1, sLen)
            flatCopy(src: s, dst: cache(.ptInput(i)), srcOff: (sLen - keep) * H, dstOff: 0, copyLen: keep * H)
            newRows = hS
            newRowsSrcOff = hist * H
        }
        // Final norm + output_proj are frame-local: contiguous trailing-m rows.
        let newOut = scratch(m * H)
        flatCopy(src: newRows, dst: newOut, srcOff: newRowsSrcOff, dstOff: 0, copyLen: m * H)
        let normedF = rms(newOut, frames: m, dim: H, wSlot: w("decoder.pre_transformer.norm.weight"))
        let ptOut = matmul(normedF, M: m, Kdim: H,
                           wSlot: w("decoder.pre_transformer.output_proj.weight"),
                           biasSlot: w("decoder.pre_transformer.output_proj.bias"), N: c.ptLatent)
        var hSlot = transpose(ptOut, rows: m, cols: c.ptLatent)        // [m, preConvDim] → [preConvDim, m]
        var dim = c.ptLatent
        var n = m

        // ====================== STAGE C: upsample ×2 + DAC + clamp ======================
        for b in 0..<c.upsampleInter.count {
            let pre = "decoder.upsample.\(b)"
            let (tc, tl) = transConvRaw(hSlot, cIn: dim, lengthIn: n, wSlot: w("\(pre).0.conv.weight"),
                                        bSlot: w("\(pre).0.conv.bias"), cOut: dim, kernel: 2, stride: 2)
            let s = concatCT(.cn(b), cacheLen: 6, new: tc, newLen: tl, channels: dim)
            let (dw, _) = conv1dValid(s, cIn: dim, lengthIn: tl + 6, wSlot: w("\(pre).1.dwconv.conv.weight"),
                                      bSlot: w("\(pre).1.dwconv.conv.bias"), cOut: dim, kernel: 7, dilation: 1, groups: dim)
            let normed = lnct(dw, dim: dim, frames: tl, wSlot: w("\(pre).1.norm.weight"), bSlot: w("\(pre).1.norm.bias"))
            let inter = c.upsampleInter[b]
            let (h1pre, _) = conv1dValid(normed, cIn: dim, lengthIn: tl, wSlot: w("\(pre).1.pwconv1.weight"),
                                         bSlot: w("\(pre).1.pwconv1.bias"), cOut: inter, kernel: 1, dilation: 1, groups: 1)
            let g = gelu(h1pre, count: inter * tl)
            let (pw2out, _) = conv1dValid(g, cIn: inter, lengthIn: tl, wSlot: w("\(pre).1.pwconv2.weight"),
                                          bSlot: w("\(pre).1.pwconv2.bias"), cOut: dim, kernel: 1, dilation: 1, groups: 1)
            updateCT(.cn(b), from: s, channels: dim, sLen: tl + 6, keep: 6)
            hSlot = sres(pw2out, res: tc, scale: w("\(pre).1.gamma"), channels: dim, frames: tl)
            n = tl
        }
        // DAC decoder.0 conv k=7.
        let sD0 = concatCT(.d0, cacheLen: 6, new: hSlot, newLen: n, channels: c.preConvDim)
        let (d0Out, _) = conv1dValid(sD0, cIn: c.preConvDim, lengthIn: n + 6, wSlot: w("decoder.decoder.0.conv.weight"),
                                     bSlot: w("decoder.decoder.0.conv.bias"), cOut: c.conv0Out, kernel: 7, dilation: 1, groups: 1)
        updateCT(.d0, from: sD0, channels: c.preConvDim, sLen: n + 6, keep: 6)
        hSlot = d0Out; dim = c.conv0Out
        for i in 0..<c.dacRates.count {
            let p = "decoder.decoder.\(i + 1).block"
            let sx = snake(hSlot, dim: dim, frames: n, alpha: w("\(p).0.alpha"), beta: w("\(p).0.beta"))
            let rate = c.dacRates[i], oD = c.dacOutDims[i]
            let histCols = isFirst ? 0 : 1
            let tcIn = histCols > 0 ? concatCT(.tc(i), cacheLen: 1, new: sx, newLen: n, channels: dim) : sx
            let (raw, rawLen) = transConvRaw(tcIn, cIn: dim, lengthIn: n + histCols, wSlot: w("\(p).1.conv.weight"),
                                             bSlot: w("\(p).1.conv.bias"), cOut: oD, kernel: 2 * rate, stride: rate)
            let outLen = n * rate
            let y = scratch(oD * outLen)
            chcopy(src: raw, dst: y, channels: oD, srcLen: rawLen, dstLen: outLen,
                   srcOff: histCols > 0 ? rate : 0, dstOff: 0, copyLen: outLen)
            // Cache the LAST post-snake input col (uses the PRE-transconv n/dim).
            chcopy(src: sx, dst: cache(.tc(i)), channels: dim, srcLen: n, dstLen: 1, srcOff: n - 1, dstOff: 0, copyLen: 1)
            hSlot = y; n = outLen; dim = oD
            for ri in 0..<c.dacDilations.count {
                let rp = "\(p).\(ri + 2)"
                let dilation = c.dacDilations[ri]
                let ctx = 6 * dilation
                let s = concatCT(.ru(i, ri), cacheLen: ctx, new: hSlot, newLen: n, channels: dim)
                let s1 = snake(s, dim: dim, frames: n + ctx, alpha: w("\(rp).act1.alpha"), beta: w("\(rp).act1.beta"))
                let (c1, _) = conv1dValid(s1, cIn: dim, lengthIn: n + ctx, wSlot: w("\(rp).conv1.conv.weight"),
                                          bSlot: w("\(rp).conv1.conv.bias"), cOut: dim, kernel: 7, dilation: dilation, groups: 1)
                let s2 = snake(c1, dim: dim, frames: n, alpha: w("\(rp).act2.alpha"), beta: w("\(rp).act2.beta"))
                let (c2, _) = conv1dValid(s2, cIn: dim, lengthIn: n, wSlot: w("\(rp).conv2.conv.weight"),
                                          bSlot: w("\(rp).conv2.conv.bias"), cOut: dim, kernel: 1, dilation: 1, groups: 1)
                updateCT(.ru(i, ri), from: s, channels: dim, sLen: n + ctx, keep: ctx)
                hSlot = sres(c2, res: hSlot, scale: onesSlot, channels: dim, frames: n)
            }
        }
        let sxF = snake(hSlot, dim: dim, frames: n, alpha: w("decoder.decoder.5.alpha"), beta: w("decoder.decoder.5.beta"))
        let sC6 = concatCT(.c6, cacheLen: 6, new: sxF, newLen: n, channels: dim)
        let (wav, _) = conv1dValid(sC6, cIn: dim, lengthIn: n + 6, wSlot: w("decoder.decoder.6.conv.weight"),
                                   bSlot: w("decoder.decoder.6.conv.bias"), cOut: 1, kernel: 7, dilation: 1, groups: 1)
        updateCT(.c6, from: sC6, channels: dim, sLen: n + 6, keep: 6)
        let clamped = scratch(n)
        records.append(B.threads(pipeline: K.clamp.rawValue, buffers: [(wav, 0), (clamped, 1)],
            constants: [.u32(UInt32(n), 2), .f32(-1, 3), .f32(1, 4)], grid: (n, 1, 1), tg: (32, 1, 1)))

        return Qwen3TTSCodecStreamPlan(records: records, slots: slots, pipelineNames: pipelineNames,
                                       outputSlot: clamped, outputLength: n)
    }
}
