// Qwen3TTSMTP — the MTP code predictor (CPU reference for gating). The 5-layer
// transformer is op-for-op identical to the talker decoder layer (per-head q/k
// RMSNorm, GQA 16/8, rotate_half RoPE, silu MLP), differing only in dims and in
// using standard 1D RoPE (rope_scaling null, theta 1e6) instead of mRoPE — so it
// reuses Qwen3TTSTalker.forward rather than duplicating the layer math.

import Foundation

public enum Qwen3TTSMTP {

    /// small_to_mtp_projection: talker hidden (2048) -> MTP hidden (1024), nn.Linear with bias.
    public static func projection(_ x: [Float], rows: Int, weight: [Float], bias: [Float],
                                  inDim: Int = 2048, outDim: Int = 1024) -> [Float] {
        Qwen3TTSTalker.linearBias(x, rows: rows, inF: inDim, outF: outDim, weight, bias)
    }

    /// The 5-layer MTP transformer math, reusing the talker decoder forward at MTP dims with
    /// standard 1D RoPE (positions 0..<frames, theta 1e6) — full causal, no KV cache. Given an
    /// identical projected input sequence + positions this equals the real KV-cached decode (the
    /// cache is only an optimization). It does NOT cover the per-frame 15-pass orchestration that
    /// produces those inputs (codec_embedding per residual step, generation_steps, lm_head
    /// selection) — that is a separate, not-yet-ported unit.
    public static func transformer(inputsEmbeds: [Float], frames: Int, w: Qwen3TTSTalker.Weights) -> [Float] {
        let (cos, sin) = Qwen3TTSTalker.ropeCosSin(positions: (0..<frames).map(Float.init))
        return Qwen3TTSTalker.forward(
            inputsEmbeds: inputsEmbeds, frames: frames, cos: cos, sin: sin, w: w,
            hidden: 1024, heads: 16, kvHeads: 8, headDim: 128, inter: 3072)
    }

    /// lm_head[g]: MTP hidden (1024) -> residual codebook-(g+1) logits (vocab 2048), no bias.
    public static func lmHead(_ hidden: [Float], rows: Int, weight: [Float],
                              hiddenDim: Int = 1024, vocab: Int = 2048) -> [Float] {
        Qwen3TTSTalker.linearBias(hidden, rows: rows, inF: hiddenDim, outF: vocab, weight, nil)
    }

    /// Per-frame residual logits, teacher-forced (forward_sub_talker_finetune). Assembles the
    /// `groups`-position sequence — pos 0 = talker hidden, pos 1 = talker codec_embedding[code0],
    /// pos 1+i = MTP codec_embedding[i-1][code_i] for i=1..groups-2 — then projects, runs the 5L
    /// transformer, and applies lm_head[i-1] at position i for i=1..groups-1. Returns
    /// [groups-1, vocab]. Covers the per-frame logit path only — the residual-codebook
    /// AR sampling / KV-cache wrapper that drives the 15 sub-passes is separate decode infra.
    public static func subTalkerLogits(
        talkerHidden: [Float], codecIds: [Int],
        talkerCodecEmb: [Float], mtpCodecEmbs: [[Float]],
        projW: [Float], projB: [Float],
        transformerW: Qwen3TTSTalker.Weights, lmHeads: [[Float]]
    ) -> [Float] {
        let talkerDim = 2048, mtpHidden = 1024, vocab = 2048, groups = 16
        var ie = [Float](repeating: 0, count: groups * talkerDim)
        for d in 0..<talkerDim { ie[d] = talkerHidden[d] }                       // pos 0
        let c0 = codecIds[0]
        for d in 0..<talkerDim { ie[talkerDim + d] = talkerCodecEmb[c0 * talkerDim + d] }  // pos 1
        for i in 1...(groups - 2) {                                              // pos 1+i = 2..groups-1
            let ci = codecIds[i], table = mtpCodecEmbs[i - 1], base = (1 + i) * talkerDim
            for d in 0..<talkerDim { ie[base + d] = table[ci * talkerDim + d] }
        }

        let proj = projection(ie, rows: groups, weight: projW, bias: projB)
        let hid = transformer(inputsEmbeds: proj, frames: groups, w: transformerW)

        var out = [Float](repeating: 0, count: (groups - 1) * vocab)
        for i in 1...(groups - 1) {                                             // head i-1 reads position i
            let row = Array(hid[(i * mtpHidden)..<((i + 1) * mtpHidden)])
            let logits = lmHead(row, rows: 1, weight: lmHeads[i - 1])
            for v in 0..<vocab { out[(i - 1) * vocab + v] = logits[v] }
        }
        return out
    }
}
