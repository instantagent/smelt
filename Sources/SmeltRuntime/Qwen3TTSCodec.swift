// Qwen3TTSCodec — codec-decoder primitives for the Qwen3-TTS-12Hz port.
//
// Stage 1 (this file): RVQ dequantization (codes -> 512-d latent). Matches the
// upstream SplitResidualVectorQuantizer.decode: a semantic group (1 codebook) and
// an acoustic group (15 codebooks summed), each at working dim 256 then projected
// to 512 via an output_proj Conv1d(256->512, kernel=1, no bias); the two groups are
// summed. Codebooks are stored as accumulators (embedding_sum / cluster_usage) and
// recovered by division (framework Trap #5: never reconstruct, divide-at-load).

import Foundation

public enum Qwen3TTSCodec {

    /// Causal Conv1d matching `Qwen3TTSTokenizerV2CausalConvNet`: left-pad by
    /// `padding = (kernel-1)*dilation+1 - stride`, right-pad by `extra` (to round the
    /// frame count up), then a valid conv. Input/output row-major [C, T]; weight is
    /// row-major [outChannels, inChannels/groups, kernel]. Returns ([outChannels, Tout], Tout).
    public static func causalConv1d(
        input: [Float], inChannels: Int, lengthIn: Int,
        weight: [Float], bias: [Float]?,
        outChannels: Int, kernel: Int, stride: Int = 1, dilation: Int = 1, groups: Int = 1
    ) -> (out: [Float], lengthOut: Int) {
        let kEff = (kernel - 1) * dilation + 1
        let padding = kEff - stride
        let nFrames = Int((Double(lengthIn - kEff + padding) / Double(stride)).rounded(.up)) + 1
        let idealLength = (nFrames - 1) * stride + (kEff - padding)
        let extra = max(0, idealLength - lengthIn)
        let paddedLen = lengthIn + padding + extra

        var x = [Float](repeating: 0, count: inChannels * paddedLen)
        for c in 0..<inChannels {
            let dst = c * paddedLen + padding
            let src = c * lengthIn
            for t in 0..<lengthIn { x[dst + t] = input[src + t] }
        }

        let lengthOut = (paddedLen - kEff) / stride + 1
        let gsIn = inChannels / groups
        let gsOut = outChannels / groups
        var out = [Float](repeating: 0, count: outChannels * lengthOut)
        for oc in 0..<outChannels {
            let g = oc / gsOut
            let wOC = oc * gsIn * kernel
            for ol in 0..<lengthOut {
                var acc: Float = bias?[oc] ?? 0
                for ic in 0..<gsIn {
                    let xBase = (g * gsIn + ic) * paddedLen
                    let wBase = wOC + ic * kernel
                    for k in 0..<kernel {
                        acc += weight[wBase + k] * x[xBase + ol * stride + k * dilation]
                    }
                }
                out[oc * lengthOut + ol] = acc
            }
        }
        return (out, lengthOut)
    }

    /// Recover a codebook embedding table from stored accumulators.
    /// `embeddingSum` is row-major [codebookSize, dim]; `clusterUsage` is [codebookSize].
    /// Returns row-major [codebookSize, dim].
    public static func codebookEmbedding(
        embeddingSum: [Float], clusterUsage: [Float], dim: Int, eps: Float = 1e-5
    ) -> [Float] {
        let n = clusterUsage.count
        var emb = [Float](repeating: 0, count: n * dim)
        for i in 0..<n {
            let denom = max(clusterUsage[i], eps)
            let base = i * dim
            for d in 0..<dim { emb[base + d] = embeddingSum[base + d] / denom }
        }
        return emb
    }

    // MARK: - full codec decode (codes -> 24kHz waveform)

    public struct CodecWeights {
        public var firstEmb: [Float], restEmb: [[Float]], firstProj: [Float], restProj: [Float]
        public var preConvW: [Float], preConvB: [Float]
        public var preTransformer: PreTransformerWeights
        public var upsampleBlocks: [UpsampleBlock]
        public var decoderW: DecoderW
        public init(firstEmb: [Float], restEmb: [[Float]], firstProj: [Float], restProj: [Float],
                    preConvW: [Float], preConvB: [Float], preTransformer: PreTransformerWeights,
                    upsampleBlocks: [UpsampleBlock], decoderW: DecoderW) {
            self.firstEmb = firstEmb; self.restEmb = restEmb
            self.firstProj = firstProj; self.restProj = restProj
            self.preConvW = preConvW; self.preConvB = preConvB
            self.preTransformer = preTransformer; self.upsampleBlocks = upsampleBlocks
            self.decoderW = decoderW
        }
    }

    /// Decode 16-codebook codes [16][T] to a 24kHz waveform [-1,1] (length T * 1920).
    public static func decodeCodec(codesKT: [[Int32]], w: CodecWeights) -> [Float] {
        let frames = codesKT[0].count
        let latent = rvqDequantize(codesKT: codesKT, firstEmb: w.firstEmb, restEmb: w.restEmb,
                                   firstProj: w.firstProj, restProj: w.restProj)  // [512, T]
        let pc = causalConv1d(input: latent, inChannels: 512, lengthIn: frames,
                              weight: w.preConvW, bias: w.preConvB, outChannels: 1024, kernel: 3).out  // [1024, T]
        // transpose [1024, T] -> [T, 1024] for the transformer.
        var ptIn = [Float](repeating: 0, count: frames * 1024)
        for c in 0..<1024 { for t in 0..<frames { ptIn[t * 1024 + c] = pc[c * frames + t] } }
        let pt = preTransformer(input: ptIn, frames: frames, w: w.preTransformer)  // [T, 1024]
        // permute [T, 1024] -> [1024, T] for upsample.
        var upIn = [Float](repeating: 0, count: 1024 * frames)
        for t in 0..<frames { for c in 0..<1024 { upIn[c * frames + t] = pt[t * 1024 + c] } }
        let (up, upLen) = upsample(input: upIn, dim: 1024, lengthIn: frames, blocks: w.upsampleBlocks)
        return decoder(input: up, latent: 1024, frames: upLen, w: w.decoderW)
    }

    // MARK: - decoder (DAC-style snake/conv upsampling stack -> waveform)

    public struct SnakeBetaW {
        public var alpha: [Float], beta: [Float]
        public init(alpha: [Float], beta: [Float]) { self.alpha = alpha; self.beta = beta }
    }
    public struct ResUnitW {
        public var act1: SnakeBetaW, act2: SnakeBetaW
        public var conv1W: [Float], conv1B: [Float], conv2W: [Float], conv2B: [Float], dilation: Int
        public init(act1: SnakeBetaW, conv1W: [Float], conv1B: [Float], act2: SnakeBetaW,
                    conv2W: [Float], conv2B: [Float], dilation: Int) {
            self.act1 = act1; self.conv1W = conv1W; self.conv1B = conv1B
            self.act2 = act2; self.conv2W = conv2W; self.conv2B = conv2B; self.dilation = dilation
        }
    }
    public struct DecoderBlockW {
        public var snake: SnakeBetaW, transW: [Float], transB: [Float], rate: Int
        public var resUnits: [ResUnitW], inDim: Int, outDim: Int
        public init(snake: SnakeBetaW, transW: [Float], transB: [Float], rate: Int,
                    resUnits: [ResUnitW], inDim: Int, outDim: Int) {
            self.snake = snake; self.transW = transW; self.transB = transB; self.rate = rate
            self.resUnits = resUnits; self.inDim = inDim; self.outDim = outDim
        }
    }
    public struct DecoderW {
        public var conv0W: [Float], conv0B: [Float], blocks: [DecoderBlockW]
        public var finalSnake: SnakeBetaW, finalConvW: [Float], finalConvB: [Float]
        public init(conv0W: [Float], conv0B: [Float], blocks: [DecoderBlockW],
                    finalSnake: SnakeBetaW, finalConvW: [Float], finalConvB: [Float]) {
            self.conv0W = conv0W; self.conv0B = conv0B; self.blocks = blocks
            self.finalSnake = finalSnake; self.finalConvW = finalConvW; self.finalConvB = finalConvB
        }
    }

    /// SnakeBeta on [dim, T]: x + 1/(exp(beta)+1e-9) * sin(x*exp(alpha))^2 (per-channel).
    public static func snakeBeta(_ x: [Float], dim: Int, frames: Int, _ w: SnakeBetaW) -> [Float] {
        var out = x
        for c in 0..<dim {
            let a = expf(w.alpha[c])
            let invB = 1.0 / (expf(w.beta[c]) + 1e-9)
            let base = c * frames
            for t in 0..<frames {
                let v = x[base + t]
                let s = sinf(v * a)
                out[base + t] = v + invB * s * s
            }
        }
        return out
    }

    static func residualUnit(_ x: [Float], dim: Int, frames: Int, _ w: ResUnitW) -> [Float] {
        var h = snakeBeta(x, dim: dim, frames: frames, w.act1)
        h = causalConv1d(input: h, inChannels: dim, lengthIn: frames, weight: w.conv1W, bias: w.conv1B,
                         outChannels: dim, kernel: 7, dilation: w.dilation).out
        h = snakeBeta(h, dim: dim, frames: frames, w.act2)
        h = causalConv1d(input: h, inChannels: dim, lengthIn: frames, weight: w.conv2W, bias: w.conv2B,
                         outChannels: dim, kernel: 1).out
        for i in 0..<h.count { h[i] += x[i] }
        return h
    }

    /// Full decoder: input [latent, T] -> waveform [T * prod(rates)] in [-1, 1].
    public static func decoder(input: [Float], latent: Int, frames: Int, w: DecoderW) -> [Float] {
        var (h, len) = causalConv1d(input: input, inChannels: latent, lengthIn: frames,
                                    weight: w.conv0W, bias: w.conv0B, outChannels: 1536, kernel: 7)
        var dim = 1536
        for b in w.blocks {
            h = snakeBeta(h, dim: b.inDim, frames: len, b.snake)
            let tc = causalTransConv1d(input: h, cIn: b.inDim, lengthIn: len, weight: b.transW, bias: b.transB,
                                       cOut: b.outDim, kernel: 2 * b.rate, stride: b.rate)
            h = tc.out; len = tc.lengthOut; dim = b.outDim
            for ru in b.resUnits { h = residualUnit(h, dim: dim, frames: len, ru) }
        }
        h = snakeBeta(h, dim: dim, frames: len, w.finalSnake)
        let wav = causalConv1d(input: h, inChannels: dim, lengthIn: len, weight: w.finalConvW,
                               bias: w.finalConvB, outChannels: 1, kernel: 7).out
        return wav.map { max(-1, min(1, $0)) }
    }

    // MARK: - upsample (2x [causal transposed conv + ConvNeXt block])

    /// ConvTranspose1d (padding=0) then trim right by (kernel-stride), matching
    /// Qwen3TTSTokenizerV2CausalTransConvNet. weight row-major [C_in, C_out, K].
    public static func causalTransConv1d(
        input: [Float], cIn: Int, lengthIn: Int,
        weight: [Float], bias: [Float]?, cOut: Int, kernel: Int, stride: Int
    ) -> (out: [Float], lengthOut: Int) {
        let fullLen = (lengthIn - 1) * stride + kernel
        var out = [Float](repeating: 0, count: cOut * fullLen)
        if let b = bias { for oc in 0..<cOut { for t in 0..<fullLen { out[oc * fullLen + t] = b[oc] } } }
        for ic in 0..<cIn {
            for i in 0..<lengthIn {
                let x = input[ic * lengthIn + i]
                let wic = ic * cOut * kernel
                for oc in 0..<cOut {
                    let wb = wic + oc * kernel
                    let ob = oc * fullLen + i * stride
                    for k in 0..<kernel { out[ob + k] += weight[wb + k] * x }
                }
            }
        }
        let trim = kernel - stride
        if trim > 0 {
            let lengthOut = fullLen - trim
            var t = [Float](repeating: 0, count: cOut * lengthOut)
            for oc in 0..<cOut { for j in 0..<lengthOut { t[oc * lengthOut + j] = out[oc * fullLen + j] } }
            return (t, lengthOut)
        }
        return (out, fullLen)
    }

    public struct ConvNeXt {
        public var dwconvW: [Float], dwconvB: [Float], normW: [Float], normB: [Float]
        public var pw1W: [Float], pw1B: [Float], pw2W: [Float], pw2B: [Float], gamma: [Float]
        public init(dwconvW: [Float], dwconvB: [Float], normW: [Float], normB: [Float],
                    pw1W: [Float], pw1B: [Float], pw2W: [Float], pw2B: [Float], gamma: [Float]) {
            self.dwconvW = dwconvW; self.dwconvB = dwconvB; self.normW = normW; self.normB = normB
            self.pw1W = pw1W; self.pw1B = pw1B; self.pw2W = pw2W; self.pw2B = pw2B; self.gamma = gamma
        }
    }

    public struct UpsampleBlock {
        public var transW: [Float], transB: [Float], convNext: ConvNeXt
        public init(transW: [Float], transB: [Float], convNext: ConvNeXt) {
            self.transW = transW; self.transB = transB; self.convNext = convNext
        }
    }

    static func gelu(_ x: Float) -> Float { 0.5 * x * (1 + erff(x / 1.4142135)) }

    /// ConvNeXt block on [dim, T] (residual). dwconv is depthwise causal k=7; norm is
    /// LayerNorm over channels (eps 1e-6); pwconv1/2 are C->4C->C with GELU; gamma scales.
    /// LayerNorm over the channel dim per frame, on [dim, frames] (C,T) storage: for each t,
    /// normalize across c (mean/var over channels), then scale/shift by normW[c]/normB[c].
    public static func layerNormCT(_ x: [Float], dim: Int, frames: Int,
                                   normW: [Float], normB: [Float], eps: Float = 1e-6) -> [Float] {
        var out = [Float](repeating: 0, count: dim * frames)
        for t in 0..<frames {
            var mean: Float = 0
            for c in 0..<dim { mean += x[c * frames + t] }
            mean /= Float(dim)
            var varr: Float = 0
            for c in 0..<dim { let d = x[c * frames + t] - mean; varr += d * d }
            varr /= Float(dim)
            let inv = 1.0 / (varr + eps).squareRoot()
            for c in 0..<dim { out[c * frames + t] = (x[c * frames + t] - mean) * inv * normW[c] + normB[c] }
        }
        return out
    }

    public static func convNeXtBlock(input: [Float], dim: Int, frames: Int, w: ConvNeXt) -> [Float] {
        let (dw, _) = causalConv1d(
            input: input, inChannels: dim, lengthIn: frames,
            weight: w.dwconvW, bias: w.dwconvB, outChannels: dim, kernel: 7, groups: dim)
        let inter = w.pw1B.count  // 4*dim
        let normed = layerNormCT(dw, dim: dim, frames: frames, normW: w.normW, normB: w.normB)
        var out = input
        var vec = [Float](repeating: 0, count: dim)
        var h1 = [Float](repeating: 0, count: inter)
        for t in 0..<frames {
            for c in 0..<dim { vec[c] = normed[c * frames + t] }
            // pwconv1 -> GELU -> pwconv2 -> gamma, residual.
            for o in 0..<inter {
                var acc = w.pw1B[o]
                let wb = o * dim
                for c in 0..<dim { acc += w.pw1W[wb + c] * vec[c] }
                h1[o] = gelu(acc)
            }
            for c in 0..<dim {
                var acc = w.pw2B[c]
                let wb = c * inter
                for o in 0..<inter { acc += w.pw2W[wb + o] * h1[o] }
                out[c * frames + t] += w.gamma[c] * acc
            }
        }
        return out
    }

    /// Full upsample stage: input [dim, T] -> [dim, T * prod(strides)].
    public static func upsample(input: [Float], dim: Int, lengthIn: Int, blocks: [UpsampleBlock]) -> (out: [Float], lengthOut: Int) {
        var h = input
        var len = lengthIn
        for b in blocks {
            let (t, tl) = causalTransConv1d(
                input: h, cIn: dim, lengthIn: len, weight: b.transW, bias: b.transB,
                cOut: dim, kernel: 2, stride: 2)
            h = convNeXtBlock(input: t, dim: dim, frames: tl, w: b.convNext)
            len = tl
        }
        return (h, len)
    }

    // MARK: - pre_transformer (8-layer Qwen-style decoder over the 12.5Hz frames)

    public struct PreTransformerLayer {
        public var inputNorm: [Float], postAttnNorm: [Float]      // [H]
        public var qProj, kProj, vProj, oProj: [Float]            // [attnDim,H],[attnDim,H],[attnDim,H],[H,attnDim]
        public var attnScale: [Float], mlpScale: [Float]          // [H]
        public var gateProj, upProj, downProj: [Float]            // [inter,H],[inter,H],[H,inter]
        public init(inputNorm: [Float], postAttnNorm: [Float], qProj: [Float], kProj: [Float],
                    vProj: [Float], oProj: [Float], attnScale: [Float], mlpScale: [Float],
                    gateProj: [Float], upProj: [Float], downProj: [Float]) {
            self.inputNorm = inputNorm; self.postAttnNorm = postAttnNorm
            self.qProj = qProj; self.kProj = kProj; self.vProj = vProj; self.oProj = oProj
            self.attnScale = attnScale; self.mlpScale = mlpScale
            self.gateProj = gateProj; self.upProj = upProj; self.downProj = downProj
        }
    }

    public struct PreTransformerWeights {
        public var inputProjW: [Float], inputProjB: [Float]       // [H,latent],[H]
        public var outputProjW: [Float], outputProjB: [Float]     // [latent,H],[latent]
        public var normW: [Float]                                 // [H]
        public var layers: [PreTransformerLayer]
        public init(inputProjW: [Float], inputProjB: [Float], outputProjW: [Float],
                    outputProjB: [Float], normW: [Float], layers: [PreTransformerLayer]) {
            self.inputProjW = inputProjW; self.inputProjB = inputProjB
            self.outputProjW = outputProjW; self.outputProjB = outputProjB
            self.normW = normW; self.layers = layers
        }
    }

    /// y[t,o] = bias[o] + sum_i W[o,i]*x[t,i].  x:[T,inF], W:[outF,inF].
    private static func linear(_ x: [Float], frames: Int, inF: Int, outF: Int,
                               _ w: [Float], _ bias: [Float]?) -> [Float] {
        var y = [Float](repeating: 0, count: frames * outF)
        for t in 0..<frames {
            let xb = t * inF
            for o in 0..<outF {
                var acc: Float = bias?[o] ?? 0
                let wb = o * inF
                for i in 0..<inF { acc += w[wb + i] * x[xb + i] }
                y[t * outF + o] = acc
            }
        }
        return y
    }

    /// RMSNorm per row [T,H] (fp32, population mean of squares).
    static func rmsNorm(_ x: [Float], frames: Int, dim: Int, _ w: [Float], eps: Float = 1e-5) -> [Float] {
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

    /// Full pre_transformer forward. `input` is row-major [T, latent]; returns [T, latent].
    public static func preTransformer(
        input: [Float], frames: Int, w: PreTransformerWeights,
        hidden: Int = 512, latent: Int = 1024, heads: Int = 16, headDim: Int = 64,
        inter: Int = 1024, theta: Float = 10000, eps: Float = 1e-5, slidingWindow: Int = 72
    ) -> [Float] {
        let attnDim = heads * headDim
        let scaling = 1.0 / Float(headDim).squareRoot()

        // RoPE tables [T, headDim] (cat(freqs,freqs) layout).
        let half = headDim / 2
        var cosT = [Float](repeating: 0, count: frames * headDim)
        var sinT = [Float](repeating: 0, count: frames * headDim)
        for t in 0..<frames {
            for i in 0..<half {
                let invFreq = 1.0 / powf(theta, Float(2 * i) / Float(headDim))
                let ang = Float(t) * invFreq
                let c = cosf(ang), s = sinf(ang)
                cosT[t * headDim + i] = c; cosT[t * headDim + i + half] = c
                sinT[t * headDim + i] = s; sinT[t * headDim + i + half] = s
            }
        }

        var h = linear(input, frames: frames, inF: latent, outF: hidden, w.inputProjW, w.inputProjB)

        for layer in w.layers {
            // Attention block.
            let normed = rmsNorm(h, frames: frames, dim: hidden, layer.inputNorm, eps: eps)
            var q = linear(normed, frames: frames, inF: hidden, outF: attnDim, layer.qProj, nil)
            var k = linear(normed, frames: frames, inF: hidden, outF: attnDim, layer.kProj, nil)
            let v = linear(normed, frames: frames, inF: hidden, outF: attnDim, layer.vProj, nil)
            // Apply RoPE per head to q,k.
            for arr in 0..<2 {
                for t in 0..<frames {
                    for hd in 0..<heads {
                        let base = t * attnDim + hd * headDim
                        for j in 0..<half {
                            let cj = cosT[t * headDim + j], sj = sinT[t * headDim + j]
                            if arr == 0 {
                                let a = q[base + j], b = q[base + j + half]
                                q[base + j] = a * cj - b * sj
                                q[base + j + half] = b * cosT[t * headDim + j + half] + a * sinT[t * headDim + j + half]
                            } else {
                                let a = k[base + j], b = k[base + j + half]
                                k[base + j] = a * cj - b * sj
                                k[base + j + half] = b * cosT[t * headDim + j + half] + a * sinT[t * headDim + j + half]
                            }
                        }
                    }
                }
            }
            // Causal attention per head -> attnOut [T, attnDim].
            var attnOut = [Float](repeating: 0, count: frames * attnDim)
            for hd in 0..<heads {
                for t in 0..<frames {
                    // Sliding-window causal attention: query t attends to keys in
                    // (t - slidingWindow, t]. For frames <= slidingWindow this equals
                    // full causal (so the T=15 reference can't distinguish them).
                    let lo = max(0, t - slidingWindow + 1)
                    var scores = [Float](repeating: 0, count: t - lo + 1)
                    var mx: Float = -.greatestFiniteMagnitude
                    for s in lo...t {
                        var dot: Float = 0
                        let qb = t * attnDim + hd * headDim, kb = s * attnDim + hd * headDim
                        for d in 0..<headDim { dot += q[qb + d] * k[kb + d] }
                        let sc = dot * scaling
                        scores[s - lo] = sc; if sc > mx { mx = sc }
                    }
                    var denom: Float = 0
                    for s in lo...t { scores[s - lo] = expf(scores[s - lo] - mx); denom += scores[s - lo] }
                    let ob = t * attnDim + hd * headDim
                    for s in lo...t {
                        let wgt = scores[s - lo] / denom
                        let vb = s * attnDim + hd * headDim
                        for d in 0..<headDim { attnOut[ob + d] += wgt * v[vb + d] }
                    }
                }
            }
            let proj = linear(attnOut, frames: frames, inF: attnDim, outF: hidden, layer.oProj, nil)
            for t in 0..<frames {
                for i in 0..<hidden { h[t * hidden + i] += layer.attnScale[i] * proj[t * hidden + i] }
            }

            // MLP block (SwiGLU).
            let normed2 = rmsNorm(h, frames: frames, dim: hidden, layer.postAttnNorm, eps: eps)
            let gate = linear(normed2, frames: frames, inF: hidden, outF: inter, layer.gateProj, nil)
            let up = linear(normed2, frames: frames, inF: hidden, outF: inter, layer.upProj, nil)
            var act = [Float](repeating: 0, count: frames * inter)
            for i in 0..<act.count {
                let g = gate[i]
                act[i] = (g / (1 + expf(-g))) * up[i]  // silu(gate) * up
            }
            let down = linear(act, frames: frames, inF: inter, outF: hidden, layer.downProj, nil)
            for t in 0..<frames {
                for i in 0..<hidden { h[t * hidden + i] += layer.mlpScale[i] * down[t * hidden + i] }
            }
        }

        let normed = rmsNorm(h, frames: frames, dim: hidden, w.normW, eps: eps)
        return linear(normed, frames: frames, inF: hidden, outF: latent, w.outputProjW, w.outputProjB)
    }

    /// RVQ dequantize: `codesKT` is [16][T] (codebook 0 = semantic, 1...15 = acoustic).
    /// Embeddings are row-major [codebookSize, dim]; output_proj weights are row-major
    /// [outDim, dim] (the Conv1d kernel=1 axis is trivial). Returns row-major [outDim, T].
    public static func rvqDequantize(
        codesKT: [[Int32]],
        firstEmb: [Float], restEmb: [[Float]],
        firstProj: [Float], restProj: [Float],
        dim: Int = 256, outDim: Int = 512
    ) -> [Float] {
        precondition(codesKT.count == 1 + restEmb.count,
                     "expected \(1 + restEmb.count) codebooks, got \(codesKT.count)")
        let frames = codesKT[0].count
        let firstSize = firstEmb.count / dim
        precondition(codesKT.allSatisfy { $0.count == frames }, "ragged codes (unequal frame counts)")
        precondition(codesKT[0].allSatisfy { $0 >= 0 && Int($0) < firstSize },
                     "semantic code out of range [0,\(firstSize)) — strip special tokens before decode")
        for k in 0..<restEmb.count {
            let n = restEmb[k].count / dim
            precondition(codesKT[k + 1].allSatisfy { $0 >= 0 && Int($0) < n },
                         "acoustic code \(k) out of range [0,\(n))")
        }

        // Semantic group: single codebook lookup -> [dim, T].
        var qFirst = [Float](repeating: 0, count: dim * frames)
        for t in 0..<frames {
            let code = Int(codesKT[0][t]) * dim
            for d in 0..<dim { qFirst[d * frames + t] = firstEmb[code + d] }
        }

        // Acoustic group: residual sum over 15 codebooks -> [dim, T].
        var qRest = [Float](repeating: 0, count: dim * frames)
        for k in 0..<restEmb.count {
            let emb = restEmb[k]
            let codes = codesKT[k + 1]
            for t in 0..<frames {
                let code = Int(codes[t]) * dim
                for d in 0..<dim { qRest[d * frames + t] += emb[code + d] }
            }
        }

        // output_proj for each group (Conv1d k=1 == matmul [outDim, dim] @ [dim, T]), summed.
        var out = [Float](repeating: 0, count: outDim * frames)
        for o in 0..<outDim {
            let wBase = o * dim
            for t in 0..<frames {
                var acc: Float = 0
                for d in 0..<dim {
                    acc += firstProj[wBase + d] * qFirst[d * frames + t]
                    acc += restProj[wBase + d] * qRest[d * frames + t]
                }
                out[o * frames + t] = acc
            }
        }
        return out
    }
}
