// Qwen3TTSTalker — the 28-layer Qwen3 talker decoder forward (CPU reference for
// gating against the real model). Differs from the codec pre_transformer: per-head
// q/k RMSNorm (head_dim only), GQA (16 query heads, 8 KV heads), no layer_scale,
// and mRoPE — which collapses to standard 1D RoPE here (the 3 position axes are
// identical for text/audio), so the gate feeds the reference cos/sin directly.

import Foundation
import Accelerate

public enum Qwen3TTSTalker {

    public struct Layer {
        public var inputNorm: [Float], postAttnNorm: [Float]   // [hidden]
        public var qProj: [Float], kProj: [Float], vProj: [Float], oProj: [Float]
        public var qNorm: [Float], kNorm: [Float]              // [headDim]
        public var gateProj: [Float], upProj: [Float], downProj: [Float]
        public init(inputNorm: [Float], postAttnNorm: [Float], qProj: [Float], kProj: [Float],
                    vProj: [Float], oProj: [Float], qNorm: [Float], kNorm: [Float],
                    gateProj: [Float], upProj: [Float], downProj: [Float]) {
            self.inputNorm = inputNorm; self.postAttnNorm = postAttnNorm
            self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.oProj = oProj
            self.qNorm = qNorm; self.kNorm = kNorm
            self.gateProj = gateProj; self.upProj = upProj; self.downProj = downProj
        }
    }

    public struct Weights {
        public var normW: [Float]
        public var layers: [Layer]
        public init(normW: [Float], layers: [Layer]) { self.normW = normW; self.layers = layers }
    }

    /// Codec head: talker hidden [T, hidden] -> codebook-0 logits [T, vocab].
    /// `weight` is row-major [vocab, hidden] (nn.Linear, no bias).
    public static func codecHead(hidden: [Float], frames: Int, weight: [Float],
                                 hiddenDim: Int = 2048, vocab: Int = 3072) -> [Float] {
        linear(hidden, frames: frames, inF: hiddenDim, outF: vocab, weight)
    }

    /// text_projection (Qwen3TTSTalkerResizeMLP): fc2(silu(fc1(x))), both nn.Linear
    /// with bias. Lifts text_embedding rows (text_hidden) into the talker hidden
    /// space during the dual-track input-embedding assembly.
    public static func textProjection(
        _ x: [Float], rows: Int, fc1W: [Float], fc1B: [Float], fc2W: [Float], fc2B: [Float],
        inDim: Int = 2048, interDim: Int = 2048, outDim: Int = 2048
    ) -> [Float] {
        var h = linearBias(x, rows: rows, inF: inDim, outF: interDim, fc1W, fc1B)
        for i in 0..<h.count { let g = h[i]; h[i] = g / (1 + expf(-g)) }
        return linearBias(h, rows: rows, inF: interDim, outF: outDim, fc2W, fc2B)
    }

    /// y[rows,outF] = x[rows,inF] · wᵀ + b, with w row-major [outF,inF] (nn.Linear).
    /// BLAS-backed (cblas_sgemm, C = x · wᵀ); bias added per row afterward.
    static func linearBias(_ x: [Float], rows: Int, inF: Int, outF: Int,
                           _ w: [Float], _ b: [Float]?) -> [Float] {
        precondition(x.count >= rows * inF && w.count >= outF * inF, "linearBias: undersized x/w")
        precondition(b == nil || b!.count >= outF, "linearBias: undersized bias")
        var y = [Float](repeating: 0, count: rows * outF)
        guard rows > 0 && outF > 0 && inF > 0 else {
            if let b = b { for t in 0..<rows { for o in 0..<outF { y[t * outF + o] = b[o] } } }
            return y
        }
        x.withUnsafeBufferPointer { xp in
            w.withUnsafeBufferPointer { wp in
                y.withUnsafeMutableBufferPointer { yp in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                                Int32(rows), Int32(outF), Int32(inF),
                                1, xp.baseAddress, Int32(inF),
                                wp.baseAddress, Int32(inF),
                                0, yp.baseAddress, Int32(outF))
                }
            }
        }
        if let b = b {
            for t in 0..<rows { for o in 0..<outF { y[t * outF + o] += b[o] } }
        }
        return y
    }

    private static func linear(_ x: [Float], frames: Int, inF: Int, outF: Int, _ w: [Float]) -> [Float] {
        linearBias(x, rows: frames, inF: inF, outF: outF, w, nil)
    }

    private static func rmsNorm(_ x: [Float], frames: Int, dim: Int, _ w: [Float], eps: Float) -> [Float] {
        var y = [Float](repeating: 0, count: frames * dim)
        for t in 0..<frames {
            let b = t * dim
            var ms: Float = 0
            for i in 0..<dim { ms += x[b + i] * x[b + i] }
            let inv = 1.0 / (ms / Float(dim) + eps).squareRoot()
            for i in 0..<dim { y[b + i] = x[b + i] * inv * w[i] }
        }
        return y
    }

    /// Per-head RMSNorm over head_dim, in place on a [T, nHeads*headDim] buffer.
    static func headRMSNorm(_ x: inout [Float], frames: Int, heads: Int, headDim: Int,
                            _ w: [Float], eps: Float) {
        for t in 0..<frames {
            for h in 0..<heads {
                let b = t * heads * headDim + h * headDim
                var ms: Float = 0
                for d in 0..<headDim { ms += x[b + d] * x[b + d] }
                let inv = 1.0 / (ms / Float(headDim) + eps).squareRoot()
                for d in 0..<headDim { x[b + d] = x[b + d] * inv * w[d] }
            }
        }
    }

    /// Derive RoPE cos/sin [T, headDim] for `positions`. The talker's mRoPE uses three
    /// position axes, but they are identical for text/audio prefill (cache_position
    /// broadcast to all 3), so it collapses to standard 1D RoPE here. rope_type
    /// "default" => attention_scaling 1.0; emb = cat(freqs, freqs) (rotate_half pairing).
    public static func ropeCosSin(positions: [Float], headDim: Int = 128, theta: Float = 1e6)
        -> (cos: [Float], sin: [Float]) {
        let half = headDim / 2
        var invFreq = [Float](repeating: 0, count: half)
        for i in 0..<half { invFreq[i] = 1.0 / powf(theta, Float(2 * i) / Float(headDim)) }
        let t = positions.count
        var cos = [Float](repeating: 0, count: t * headDim)
        var sin = [Float](repeating: 0, count: t * headDim)
        for p in 0..<t {
            let base = p * headDim
            for i in 0..<half {
                let f = positions[p] * invFreq[i]
                let c = cosf(f), s = sinf(f)
                cos[base + i] = c; cos[base + i + half] = c
                sin[base + i] = s; sin[base + i + half] = s
            }
        }
        return (cos, sin)
    }

    /// Apply RoPE (rotate_half) in place to a [T, heads*headDim] buffer using cos/sin [T, headDim].
    private static func applyRope(_ x: inout [Float], frames: Int, heads: Int, headDim: Int,
                                  cos: [Float], sin: [Float]) {
        let half = headDim / 2
        for t in 0..<frames {
            let cb = t * headDim
            for h in 0..<heads {
                let base = t * heads * headDim + h * headDim
                for j in 0..<half {
                    let a = x[base + j], b = x[base + j + half]
                    x[base + j] = a * cos[cb + j] - b * sin[cb + j]
                    x[base + j + half] = b * cos[cb + j + half] + a * sin[cb + j + half]
                }
            }
        }
    }

    /// Talker forward over a prefill sequence. `inputsEmbeds` is [T, hidden]; `cos`/`sin`
    /// are [T, headDim] (mRoPE collapsed to 1D). Returns the final-norm hidden [T, hidden].
    /// Kept as the cache-free reference that `forwardStep`'s KV-cache parity gate checks
    /// against — the per-layer math here and in `forwardStep` must stay in lockstep.
    public static func forward(
        inputsEmbeds: [Float], frames: Int, cos: [Float], sin: [Float], w: Weights,
        hidden: Int = 2048, heads: Int = 16, kvHeads: Int = 8, headDim: Int = 128,
        inter: Int = 6144, eps: Float = 1e-6
    ) -> [Float] {
        let qDim = heads * headDim       // 2048
        let kvDim = kvHeads * headDim    // 1024
        let scaling = 1.0 / Float(headDim).squareRoot()
        let group = heads / kvHeads      // 2 (GQA)

        var h = inputsEmbeds
        for layer in w.layers {
            let normed = rmsNorm(h, frames: frames, dim: hidden, layer.inputNorm, eps: eps)
            var q = linear(normed, frames: frames, inF: hidden, outF: qDim, layer.qProj)
            var k = linear(normed, frames: frames, inF: hidden, outF: kvDim, layer.kProj)
            let v = linear(normed, frames: frames, inF: hidden, outF: kvDim, layer.vProj)

            headRMSNorm(&q, frames: frames, heads: heads, headDim: headDim, layer.qNorm, eps: eps)
            headRMSNorm(&k, frames: frames, heads: kvHeads, headDim: headDim, layer.kNorm, eps: eps)
            applyRope(&q, frames: frames, heads: heads, headDim: headDim, cos: cos, sin: sin)
            applyRope(&k, frames: frames, heads: kvHeads, headDim: headDim, cos: cos, sin: sin)

            var attnOut = [Float](repeating: 0, count: frames * qDim)
            for qh in 0..<heads {
                let kvh = qh / group
                for t in 0..<frames {
                    var scores = [Float](repeating: 0, count: t + 1)
                    var mx: Float = -.greatestFiniteMagnitude
                    let qb = t * qDim + qh * headDim
                    for s in 0...t {
                        var dot: Float = 0
                        let kb = s * kvDim + kvh * headDim
                        for d in 0..<headDim { dot += q[qb + d] * k[kb + d] }
                        let sc = dot * scaling
                        scores[s] = sc; if sc > mx { mx = sc }
                    }
                    var denom: Float = 0
                    for s in 0...t { scores[s] = expf(scores[s] - mx); denom += scores[s] }
                    let ob = t * qDim + qh * headDim
                    for s in 0...t {
                        let wgt = scores[s] / denom
                        let vb = s * kvDim + kvh * headDim
                        for d in 0..<headDim { attnOut[ob + d] += wgt * v[vb + d] }
                    }
                }
            }
            let proj = linear(attnOut, frames: frames, inF: qDim, outF: hidden, layer.oProj)
            for i in 0..<h.count { h[i] += proj[i] }

            let normed2 = rmsNorm(h, frames: frames, dim: hidden, layer.postAttnNorm, eps: eps)
            let gate = linear(normed2, frames: frames, inF: hidden, outF: inter, layer.gateProj)
            let up = linear(normed2, frames: frames, inF: hidden, outF: inter, layer.upProj)
            var act = [Float](repeating: 0, count: frames * inter)
            for i in 0..<act.count { let g = gate[i]; act[i] = (g / (1 + expf(-g))) * up[i] }
            let down = linear(act, frames: frames, inF: inter, outF: hidden, layer.downProj)
            for i in 0..<h.count { h[i] += down[i] }
        }
        return rmsNorm(h, frames: frames, dim: hidden, w.normW, eps: eps)
    }

    /// Per-layer KV cache: post-RoPE K and post-`v_proj` V for the positions seen so far,
    /// flat `[pos * kvDim]` per layer. Incremental decode reuses these instead of recomputing.
    public struct KVCache {
        var k: [[Float]], v: [[Float]], length: Int
        public init(layers: Int) {
            k = Array(repeating: [], count: layers); v = Array(repeating: [], count: layers); length = 0
        }
    }

    /// Incremental forward: process `newFrames` rows of `newEmbeds` at absolute positions
    /// [startPos, startPos+newFrames), extending `cache` (which holds positions [0,startPos)).
    /// Returns the final-norm hidden for the new rows [newFrames, hidden] — bit-equivalent (to
    /// fp reassociation) to the matching rows of `forward` over the full [0,startPos+newFrames)
    /// sequence. RoPE is applied at absolute positions; cached K is post-RoPE, V post-v_proj.
    public static func forwardStep(
        newEmbeds: [Float], newFrames: Int, startPos: Int, cache: inout KVCache, w: Weights,
        hidden: Int = 2048, heads: Int = 16, kvHeads: Int = 8, headDim: Int = 128,
        inter: Int = 6144, eps: Float = 1e-6
    ) -> [Float] {
        let qDim = heads * headDim, kvDim = kvHeads * headDim
        let scaling = 1.0 / Float(headDim).squareRoot()
        let group = heads / kvHeads
        precondition(startPos == cache.length, "cache holds \(cache.length) positions, step starts at \(startPos)")
        let (cos, sin) = ropeCosSin(positions: (startPos..<(startPos + newFrames)).map(Float.init), headDim: headDim)

        var h = newEmbeds
        for (li, layer) in w.layers.enumerated() {
            let normed = rmsNorm(h, frames: newFrames, dim: hidden, layer.inputNorm, eps: eps)
            var q = linear(normed, frames: newFrames, inF: hidden, outF: qDim, layer.qProj)
            var k = linear(normed, frames: newFrames, inF: hidden, outF: kvDim, layer.kProj)
            let v = linear(normed, frames: newFrames, inF: hidden, outF: kvDim, layer.vProj)
            headRMSNorm(&q, frames: newFrames, heads: heads, headDim: headDim, layer.qNorm, eps: eps)
            headRMSNorm(&k, frames: newFrames, heads: kvHeads, headDim: headDim, layer.kNorm, eps: eps)
            applyRope(&q, frames: newFrames, heads: heads, headDim: headDim, cos: cos, sin: sin)
            applyRope(&k, frames: newFrames, heads: kvHeads, headDim: headDim, cos: cos, sin: sin)
            cache.k[li].append(contentsOf: k)   // now holds positions [0, startPos+newFrames)
            cache.v[li].append(contentsOf: v)
            let ck = cache.k[li], cv = cache.v[li]

            var attnOut = [Float](repeating: 0, count: newFrames * qDim)
            for qh in 0..<heads {
                let kvh = qh / group
                for t in 0..<newFrames {
                    let absPos = startPos + t
                    var scores = [Float](repeating: 0, count: absPos + 1)
                    var mx: Float = -.greatestFiniteMagnitude
                    let qb = t * qDim + qh * headDim
                    for s in 0...absPos {
                        var dot: Float = 0
                        let kb = s * kvDim + kvh * headDim
                        for d in 0..<headDim { dot += q[qb + d] * ck[kb + d] }
                        let sc = dot * scaling
                        scores[s] = sc; if sc > mx { mx = sc }
                    }
                    var denom: Float = 0
                    for s in 0...absPos { scores[s] = expf(scores[s] - mx); denom += scores[s] }
                    let ob = t * qDim + qh * headDim
                    for s in 0...absPos {
                        let wgt = scores[s] / denom
                        let vb = s * kvDim + kvh * headDim
                        for d in 0..<headDim { attnOut[ob + d] += wgt * cv[vb + d] }
                    }
                }
            }
            let proj = linear(attnOut, frames: newFrames, inF: qDim, outF: hidden, layer.oProj)
            for i in 0..<h.count { h[i] += proj[i] }

            let normed2 = rmsNorm(h, frames: newFrames, dim: hidden, layer.postAttnNorm, eps: eps)
            let gate = linear(normed2, frames: newFrames, inF: hidden, outF: inter, layer.gateProj)
            let up = linear(normed2, frames: newFrames, inF: hidden, outF: inter, layer.upProj)
            var act = [Float](repeating: 0, count: newFrames * inter)
            for i in 0..<act.count { let g = gate[i]; act[i] = (g / (1 + expf(-g))) * up[i] }
            let down = linear(act, frames: newFrames, inF: inter, outF: hidden, layer.downProj)
            for i in 0..<h.count { h[i] += down[i] }
        }
        cache.length = startPos + newFrames
        return rmsNorm(h, frames: newFrames, dim: hidden, w.normW, eps: eps)
    }
}
