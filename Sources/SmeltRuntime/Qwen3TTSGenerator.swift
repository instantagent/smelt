// Qwen3TTSGenerator — the autoregressive talker+MTP decode glue (CPU reference).
// Each frame: the talker emits codebook-0 (argmax of codecHead); the MTP emits the 15
// residual codebooks; the next frame's talker input is the additive sum of all 16
// codebook embeddings plus tts_pad (the dual-track feedback). Codebook-0 uses the
// talker's codec_embedding [3072,*]; the 15 residuals use the MTP's codec_embedding
// tables [2048,*]. Positions are sequential (mRoPE stays 1D through decode).

import Foundation

public enum Qwen3TTSGenerator {

    /// Single source of truth for the codebook-0 sampler constants, so the CPU `applyCb0Processors`
    /// and the GPU `cb0_argmax_f32` path can't drift (a divergence would silently break codes==gen_codes).
    public enum Cb0Config {
        public static let repetitionPenalty: Float = 1.05
        public static let suppressFrom = 2048
        public static let minNewTokens = 2
    }

    /// Next-frame talker input: sum of the frame's 16 codebook embeddings + tts_pad.
    /// `codes16` = [cb0, cb1, …, cb15]; `mtpCodecEmbs` = the 15 MTP tables (cb1..cb15).
    public static func nextFrameInput(
        codes16: [Int], talkerCodecEmb: [Float], mtpCodecEmbs: [[Float]],
        ttsPadEmbed: [Float], dim: Int = 2048
    ) -> [Float] {
        precondition(codes16.count == 16, "expected 16 codebooks, got \(codes16.count)")
        precondition(mtpCodecEmbs.count == 15, "expected 15 MTP codec_embedding tables")
        precondition(ttsPadEmbed.count == dim, "ttsPadEmbed must be dim-wide")
        let cb0 = codes16[0]
        precondition(cb0 >= 0 && (cb0 + 1) * dim <= talkerCodecEmb.count, "cb0 \(cb0) out of talker table")
        var out = [Float](repeating: 0, count: dim)
        for d in 0..<dim { out[d] = talkerCodecEmb[cb0 * dim + d] + ttsPadEmbed[d] }
        for i in 0..<15 {
            let id = codes16[i + 1], table = mtpCodecEmbs[i]
            precondition(id >= 0 && (id + 1) * dim <= table.count, "cb\(i + 1) id \(id) out of MTP table \(i)")
            for d in 0..<dim { out[d] += table[id * dim + d] }
        }
        return out
    }

    /// All weights the talker+MTP greedy decode needs. Codebook-0 comes from the talker;
    /// the 15 residuals from the MTP. `mtpCodecEmbs` holds the 15 MTP codec_embedding tables.
    public struct Weights {
        public var talker: Qwen3TTSTalker.Weights
        public var codecHeadW: [Float]
        public var talkerCodecEmb: [Float]
        public var mtp: Qwen3TTSTalker.Weights
        public var mtpProjW: [Float], mtpProjB: [Float]
        public var mtpLmHeads: [[Float]]
        public var mtpCodecEmbs: [[Float]]
        public var ttsPadEmbed: [Float]
        public var codecEosId: Int
        public init(talker: Qwen3TTSTalker.Weights, codecHeadW: [Float], talkerCodecEmb: [Float],
                    mtp: Qwen3TTSTalker.Weights, mtpProjW: [Float], mtpProjB: [Float],
                    mtpLmHeads: [[Float]], mtpCodecEmbs: [[Float]], ttsPadEmbed: [Float], codecEosId: Int) {
            self.talker = talker; self.codecHeadW = codecHeadW; self.talkerCodecEmb = talkerCodecEmb
            self.mtp = mtp; self.mtpProjW = mtpProjW; self.mtpProjB = mtpProjB
            self.mtpLmHeads = mtpLmHeads; self.mtpCodecEmbs = mtpCodecEmbs
            self.ttsPadEmbed = ttsPadEmbed; self.codecEosId = codecEosId
        }
    }

    private static func argmax(_ v: ArraySlice<Float>) -> Int {
        var best = v.startIndex, mx = -Float.greatestFiniteMagnitude
        for i in v.indices where v[i] > mx { mx = v[i]; best = i }
        return best - v.startIndex
    }

    /// The codebook-0 logits processors the real talker.generate applies (do_sample=False):
    ///   - suppress_tokens: the special-token range [suppressFrom, vocab) except codec_eos,
    ///     so cb0 is constrained to an audio code (0..suppressFrom-1) or eos.
    ///   - repetition_penalty: scale the logit of every already-emitted cb0 by 1/penalty
    ///     (if positive) or *penalty (if negative).
    ///   - min_new_tokens: suppress eos for the first `minNewTokens` frames.
    /// `frame` is the 0-based index of the frame being produced; `history` the prior cb0s.
    public static func applyCb0Processors(
        _ logits: [Float], history: [Int], frame: Int,
        repetitionPenalty: Float = Cb0Config.repetitionPenalty, suppressFrom: Int = Cb0Config.suppressFrom,
        vocab: Int = 3072, codecEosId: Int = 2150, minNewTokens: Int = Cb0Config.minNewTokens
    ) -> [Float] {
        var l = logits
        for t in Set(history) where t >= 0 && t < l.count {
            l[t] = l[t] > 0 ? l[t] / repetitionPenalty : l[t] * repetitionPenalty
        }
        for t in suppressFrom..<min(vocab, l.count) where t != codecEosId { l[t] = -.infinity }
        if frame < minNewTokens, codecEosId < l.count { l[codecEosId] = -.infinity }
        return l
    }

    /// Greedy MTP within one frame: given the talker's hidden for this frame and cb0,
    /// produce cb1..cb15 by 15 sequential argmax sub-passes (each codebook conditions on
    /// the prior). Mirrors code_predictor.generate over [past_hidden, cb0_embed] + 14 steps.
    public static func mtpGenerate(talkerHidden: [Float], cb0: Int, w: Weights, dim: Int = 2048) -> [Int] {
        var seq = talkerHidden                                 // pos 0: talker hidden
        seq += Array(w.talkerCodecEmb[(cb0 * dim)..<((cb0 + 1) * dim)])  // pos 1: cb0 embed
        var residuals: [Int] = []
        for gs in 0..<15 {
            let rows = seq.count / dim
            let proj = Qwen3TTSMTP.projection(seq, rows: rows, weight: w.mtpProjW, bias: w.mtpProjB)
            let hidden = Qwen3TTSMTP.transformer(inputsEmbeds: proj, frames: rows, w: w.mtp)
            let last = Array(hidden[((rows - 1) * 1024)..<(rows * 1024)])
            let logits = Qwen3TTSMTP.lmHead(last, rows: 1, weight: w.mtpLmHeads[gs])
            let cb = argmax(logits[logits.startIndex..<logits.endIndex])
            residuals.append(cb)
            if gs < 14 {
                let table = w.mtpCodecEmbs[gs]
                seq += Array(table[(cb * dim)..<((cb + 1) * dim)])
            }
        }
        return residuals
    }

    /// Free-running greedy generation: from the assembled prefill inputs_embeds, emit up to
    /// `maxFrames` frames of 16 codebooks, stopping when cb0 == codecEosId. Full-recompute
    /// talker forward each step (correctness reference; a KV cache is the perf path).
    /// Returns codes as [16][nFrames] codebook-major — exactly the `codesKT` shape
    /// Qwen3TTSCodec.decodeCodec consumes, so generate→decode chains without a transpose.
    public static func generate(prefill: [Float], prefillLen: Int, w: Weights, maxFrames: Int,
                                dim: Int = 2048) -> [[Int]] {
        guard maxFrames > 0 else { return Array(repeating: [], count: 16) }
        precondition(prefillLen > 0 && prefill.count >= prefillLen * dim, "generate: invalid prefill")
        var frames: [[Int]] = []          // frame-major during the loop; transposed at return
        var cb0History: [Int] = []
        // Prefill once into the KV cache; each frame is then a single-position decode step.
        var cache = Qwen3TTSTalker.KVCache(layers: w.talker.layers.count)
        let prefillHidden = Qwen3TTSTalker.forwardStep(
            newEmbeds: prefill, newFrames: prefillLen, startPos: 0, cache: &cache, w: w.talker)
        var lastHidden = Array(prefillHidden[((prefillLen - 1) * dim)..<(prefillLen * dim)])
        for frame in 0..<maxFrames {
            let rawLogits = Qwen3TTSTalker.codecHead(hidden: lastHidden, frames: 1, weight: w.codecHeadW)
            let logits = applyCb0Processors(rawLogits, history: cb0History, frame: frame, codecEosId: w.codecEosId)
            let cb0 = argmax(logits[logits.startIndex..<logits.endIndex])
            if cb0 == w.codecEosId { break }
            cb0History.append(cb0)
            let codes16 = [cb0] + mtpGenerate(talkerHidden: lastHidden, cb0: cb0, w: w)
            frames.append(codes16)
            guard frame < maxFrames - 1 else { break }  // skip the unused next-position decode
            let nextInput = nextFrameInput(codes16: codes16, talkerCodecEmb: w.talkerCodecEmb,
                                           mtpCodecEmbs: w.mtpCodecEmbs, ttsPadEmbed: w.ttsPadEmbed)
            lastHidden = Qwen3TTSTalker.forwardStep(
                newEmbeds: nextInput, newFrames: 1, startPos: cache.length, cache: &cache, w: w.talker)
        }
        // Transpose frame-major [nFrames][16] -> codebook-major [16][nFrames].
        var codes = [[Int]](repeating: [Int](repeating: 0, count: frames.count), count: 16)
        for (k, frame) in frames.enumerated() { for c in 0..<16 { codes[c][k] = frame[c] } }
        return codes
    }
}
