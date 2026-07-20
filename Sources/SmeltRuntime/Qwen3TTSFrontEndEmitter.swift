// Qwen3TTSFrontEndEmitter — Phase 1a-ii: the TTS dual-track prefill front-end as a flat record table,
// run through the generic SmeltCodecRecordRunner (the same runtime-emit seam C2/C3 use for the codec),
// replacing 1a-i's host-issued matmulDispatch/siluDispatch/scaleResidualTCDispatch calls. Once seqLen is
// fixed the front-end is offline-static (literal grids/constants), so a fresh literal table is emitted
// per request — no dynamic kinds, no sidecar table, no IR.
//
// BIT-IDENTICAL to the host frontEndPrefillHiddenA by construction: every kernel + binding + grid +
// constant is copied from the resident leaf dispatch it replaces (gemm_tn_f32 / matmul_f32 mirror
// matmulDispatch's M>1 K%4 selection; silu_f32 mirrors siluDispatch; scale_residual_tc_f32 has_scale=0
// mirrors the add-only merge). The resident weights are the SAME bytes bufF32 would upload (fc1/fc2 are
// fp32 in every build; text_embedding bf16 widens in-kernel identically to weightRow's bits<<16; codec
// is fp32), and gather_rows reproduces weightRow's per-row gather — so the produced hiddenA is byte-for-
// byte equal to the host path (the byte-identical gate proves it).

import SmeltSchema

private typealias B = SmeltCodecRecordBuilder

/// What buffer occupies a front-end record slot. The realizer (the session / gpu, which owns the
/// resident weights + the request's id buffers + hiddenA + scratch) materializes these by slot index.
public enum Qwen3TTSFrontEndSlotDesc: Equatable {
    case textEmbed                 // talker.model.text_embedding.weight (bf16 in a bf16 build, else f32)
    case codecEmbed                // talker.model.codec_embedding.weight (fp32)
    case fc1W, fc1B, fc2W, fc2B     // talker.text_projection.linear_fc{1,2}.{weight,bias} (fp32)
    case textIds                   // [seqLen] Int32, host-written from layout().textId
    case codecIds                  // [seqLen] Int32, host-written from layout().codecId (-1 ⇒ zero row)
    case hiddenA                   // the trunk's hiddenA port (output)
    case scratch(Int)              // fresh per-request intermediate (count floats; fully written, no zero)
}

/// One request's front-end record table + the slots the realizer must materialize.
public struct Qwen3TTSFrontEndPlan {
    public let records: [SmeltDispatchRecord]
    public let slots: [Qwen3TTSFrontEndSlotDesc]
    public let pipelineNames: [String]
    public let hiddenASlot: Int
}

public enum Qwen3TTSFrontEndEmitter {

    // Dense local pipeline index (positional, matching SmeltCodecRecordRunner's record.pipeline lookup).
    private enum K: Int, CaseIterable {
        case gatherF32 = 0, gatherBF16, gemmTN, matmul, silu, srtc
    }
    public static let pipelineNames: [String] = [
        "gather_rows_f32", "gather_rows_bf16w_f32", "gemm_tn_f32", "matmul_f32", "silu_f32",
        "scale_residual_tc_f32",
    ]
    // gemm_tn tiling (matches matmulDispatch's gemmTNFeatures / gemmTNTileM).
    private static let gemmTNFeatures = 4
    private static let gemmTNTileM = 16

    /// Build the front-end record table for a `seqLen`-row prefill. `textEmbedBF16` picks the bf16-widen
    /// gather (bf16 build) vs the fp32 gather (f32 build); u4 text_embedding is unsupported here (stays on
    /// the host weightRow path). Dims come from talkerShape (textEmbedDim → projInter → hidden).
    public static func plan(seqLen: Int, textEmbedDim: Int, projInter: Int, hidden: Int,
                            textEmbedBF16: Bool) -> Qwen3TTSFrontEndPlan {
        precondition(pipelineNames.count == K.allCases.count, "front-end pipelineNames out of sync with K")
        precondition(seqLen > 1, "front-end seqLen must be > 1 (gemm_tn M>1 path)")
        precondition(textEmbedDim > 0 && projInter > 0 && hidden > 0, "front-end dims must be positive")

        // Fixed slots 0..8; scratch appended after.
        var slots: [Qwen3TTSFrontEndSlotDesc] = [
            .textEmbed, .codecEmbed, .fc1W, .fc1B, .fc2W, .fc2B, .textIds, .codecIds, .hiddenA]
        let textEmbed = 0, codecEmbed = 1, fc1W = 2, fc1B = 3, fc2W = 4, fc2B = 5
        let textIds = 6, codecIds = 7, hiddenA = 8
        func scratch(_ count: Int) -> Int { slots.append(.scratch(count)); return slots.count - 1 }
        var records: [SmeltDispatchRecord] = []

        // gather_rows_{f32,bf16w_f32}: table@0 ids@1 out@2 | n@3 dim@4 | flat grid n*dim.
        func gatherRows(table: Int, ids: Int, out: Int, n: Int, dim: Int, bf16: Bool) {
            records.append(B.threads(pipeline: (bf16 ? K.gatherBF16 : K.gatherF32).rawValue,
                buffers: [(table, 0), (ids, 1), (out, 2)],
                constants: [.u32(UInt32(n), 3), .u32(UInt32(dim), 4)],
                grid: (n * dim, 1, 1), tg: (min(n * dim, 256), 1, 1)))
        }
        // Biased matmul, mirroring matmulDispatch's M>1 selection: K%4==0 → gemm_tn_f32, else matmul_f32.
        // x@0 w@1 bias@2 out@3 | M@4 N@5 K@6 has_bias@7. Always biased (fc1/fc2 carry biases).
        func matmul(_ inSlot: Int, M: Int, Kdim: Int, wSlot: Int, biasSlot: Int, N: Int) -> Int {
            let out = scratch(M * N)
            let consts: [B.Const] = [.u32(UInt32(M), 4), .u32(UInt32(N), 5), .u32(UInt32(Kdim), 6), .u32(1, 7)]
            if Kdim % 4 == 0 {
                let nGroups = (N + gemmTNFeatures - 1) / gemmTNFeatures
                let mRows = (M + gemmTNTileM - 1) / gemmTNTileM * gemmTNTileM
                records.append(B.threads(pipeline: K.gemmTN.rawValue,
                    buffers: [(inSlot, 0), (wSlot, 1), (biasSlot, 2), (out, 3)], constants: consts,
                    grid: (nGroups * 32, mRows, 1), tg: (32, gemmTNTileM, 1)))
            } else {
                records.append(B.threads(pipeline: K.matmul.rawValue,
                    buffers: [(inSlot, 0), (wSlot, 1), (biasSlot, 2), (out, 3)], constants: consts,
                    grid: (N, M, 1), tg: (min(N, 32), 1, 1)))
            }
            return out
        }

        // Row-aligned dual track (mirrors frontEndPrefillHiddenA):
        // text gather → fc2(silu(fc1)) → projected; codec gather; hiddenA = projected + codec.
        let gatheredText = scratch(seqLen * textEmbedDim)
        gatherRows(table: textEmbed, ids: textIds, out: gatheredText, n: seqLen, dim: textEmbedDim, bf16: textEmbedBF16)
        let h1 = matmul(gatheredText, M: seqLen, Kdim: textEmbedDim, wSlot: fc1W, biasSlot: fc1B, N: projInter)
        let act = scratch(seqLen * projInter)
        records.append(B.threads(pipeline: K.silu.rawValue, buffers: [(h1, 0), (act, 1)],
            constants: [.u32(UInt32(seqLen * projInter), 2)],
            grid: (seqLen * projInter, 1, 1), tg: (min(seqLen * projInter, 256), 1, 1)))
        let projected = matmul(act, M: seqLen, Kdim: projInter, wSlot: fc2W, biasSlot: fc2B, N: hidden)
        let gatheredCodec = scratch(seqLen * hidden)
        gatherRows(table: codecEmbed, ids: codecIds, out: gatheredCodec, n: seqLen, dim: hidden, bf16: false)
        // scale_residual_tc has_scale=0 → out = residual + x. x@0=codec res@1=projected scale@2(unread)
        // out@3=hiddenA | channels@4 frames@5 has_scale@6 | grid frames×channels.
        records.append(B.threads(pipeline: K.srtc.rawValue,
            buffers: [(gatheredCodec, 0), (projected, 1), (gatheredCodec, 2), (hiddenA, 3)],
            constants: [.u32(UInt32(hidden), 4), .u32(UInt32(seqLen), 5), .u32(0, 6)],
            grid: (seqLen, hidden, 1), tg: (min(seqLen, 32), 1, 1)))

        return Qwen3TTSFrontEndPlan(records: records, slots: slots, pipelineNames: pipelineNames, hiddenASlot: hiddenA)
    }
}
