import Accelerate
import Foundation

public struct WhisperTinyConstants {
    public static let dModel = 384
    public static let heads = 6
    public static let headDim = 64
    public static let ffnDim = 1536
    public static let layerCount = 4
    public static let sourceLength = 1500
    public static let maxTargetPositions = 448
    public static let vocabSize = 51865

    public static let startOfTranscript = 50258
    public static let english = 50259
    public static let transcribe = 50359
    public static let noTimestamps = 50363
    public static let endOfText = 50257

    public static let forcedPrefix = [
        startOfTranscript,
        english,
        transcribe,
        noTimestamps
    ]

    public static let alignmentHeads: [(layer: Int, head: Int)] = [
        (2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)
    ]
}

public struct WhisperTinyGreedyResult: Sendable {
    public let tokens: [Int]
    public let alignmentRows: [[Float]]
}

public struct WhisperTinyStepResult: Sendable {
    public let logits: [Float]
    public let alignmentRow: [Float]
}

public final class WhisperTinyCPUDecoder {
    public struct LayerWeights {
        let selfAttnLnW: [Float]
        let selfAttnLnB: [Float]
        let selfQW: [Float]
        let selfQB: [Float]
        let selfKW: [Float]
        let selfVW: [Float]
        let selfVB: [Float]
        let selfOW: [Float]
        let selfOB: [Float]

        let crossAttnLnW: [Float]
        let crossAttnLnB: [Float]
        let crossQW: [Float]
        let crossQB: [Float]
        let crossKW: [Float]
        let crossVW: [Float]
        let crossVB: [Float]
        let crossOW: [Float]
        let crossOB: [Float]

        let finalLnW: [Float]
        let finalLnB: [Float]
        let fc1W: [Float]
        let fc1B: [Float]
        let fc2W: [Float]
        let fc2B: [Float]
    }

    public struct CrossAttentionCache: Sendable {
        let keysByLayer: [[Float]]
        let valuesByLayer: [[Float]]
    }

    public struct State: Sendable {
        let crossAttentionCache: CrossAttentionCache
        var selfKeyCache: [[Float]]
        var selfValueCache: [[Float]]
        public var position: Int
    }

    private let tokEmbW: [Float]
    private let posEmbW: [Float]
    private let finalLnW: [Float]
    private let finalLnB: [Float]
    private let lmHeadW: [Float]
    private let layerWeights: [LayerWeights]

    public init(weightsPath: String? = nil) throws {
        let resolvedWeightsPath = try weightsPath ?? WhisperTinyReferenceAssets.weightsPath()
        let loader = try SafetensorsLoader(paths: [resolvedWeightsPath])

        func load(_ name: String) throws -> [Float] {
            guard let info = loader.tensor(named: name) else {
                throw DecoderError.weightNotFound(name)
            }

            let count = info.shape.reduce(1, *)
            var floats = [Float](repeating: 0, count: count)
            let ptr = loader.tensorData(info)

            switch info.dtype {
            case "F32":
                let src = ptr.bindMemory(to: Float.self, capacity: count)
                floats.withUnsafeMutableBufferPointer { dst in
                    dst.baseAddress!.initialize(from: src, count: count)
                }
            case "F16":
                let src = ptr.bindMemory(to: Float16.self, capacity: count)
                for i in 0..<count {
                    floats[i] = Float(src[i])
                }
            default:
                throw DecoderError.unsupportedDType(name: name, dtype: info.dtype)
            }

            return floats
        }

        self.tokEmbW = try load("model.decoder.embed_tokens.weight")
        self.posEmbW = try load("model.decoder.embed_positions.weight")
        self.finalLnW = try load("model.decoder.layer_norm.weight")
        self.finalLnB = try load("model.decoder.layer_norm.bias")
        self.lmHeadW = tokEmbW

        var layers: [LayerWeights] = []
        layers.reserveCapacity(WhisperTinyConstants.layerCount)
        for i in 0..<WhisperTinyConstants.layerCount {
            let prefix = "model.decoder.layers.\(i)"
            layers.append(
                LayerWeights(
                    selfAttnLnW: try load("\(prefix).self_attn_layer_norm.weight"),
                    selfAttnLnB: try load("\(prefix).self_attn_layer_norm.bias"),
                    selfQW: try load("\(prefix).self_attn.q_proj.weight"),
                    selfQB: try load("\(prefix).self_attn.q_proj.bias"),
                    selfKW: try load("\(prefix).self_attn.k_proj.weight"),
                    selfVW: try load("\(prefix).self_attn.v_proj.weight"),
                    selfVB: try load("\(prefix).self_attn.v_proj.bias"),
                    selfOW: try load("\(prefix).self_attn.out_proj.weight"),
                    selfOB: try load("\(prefix).self_attn.out_proj.bias"),
                    crossAttnLnW: try load("\(prefix).encoder_attn_layer_norm.weight"),
                    crossAttnLnB: try load("\(prefix).encoder_attn_layer_norm.bias"),
                    crossQW: try load("\(prefix).encoder_attn.q_proj.weight"),
                    crossQB: try load("\(prefix).encoder_attn.q_proj.bias"),
                    crossKW: try load("\(prefix).encoder_attn.k_proj.weight"),
                    crossVW: try load("\(prefix).encoder_attn.v_proj.weight"),
                    crossVB: try load("\(prefix).encoder_attn.v_proj.bias"),
                    crossOW: try load("\(prefix).encoder_attn.out_proj.weight"),
                    crossOB: try load("\(prefix).encoder_attn.out_proj.bias"),
                    finalLnW: try load("\(prefix).final_layer_norm.weight"),
                    finalLnB: try load("\(prefix).final_layer_norm.bias"),
                    fc1W: try load("\(prefix).fc1.weight"),
                    fc1B: try load("\(prefix).fc1.bias"),
                    fc2W: try load("\(prefix).fc2.weight"),
                    fc2B: try load("\(prefix).fc2.bias")
                )
            )
        }
        self.layerWeights = layers
    }

    public func loadReferenceEncoderOutput() throws -> [Float] {
        try WhisperTinyReferenceAssets.loadReference(name: "encoder_output").data
    }

    public func prepareCrossAttentionCache(
        encoderOutput: [Float],
        sourceLength: Int = WhisperTinyConstants.sourceLength
    ) -> CrossAttentionCache {
        let dm = WhisperTinyConstants.dModel
        precondition(encoderOutput.count == sourceLength * dm)

        var keysByLayer: [[Float]] = []
        var valuesByLayer: [[Float]] = []
        keysByLayer.reserveCapacity(WhisperTinyConstants.layerCount)
        valuesByLayer.reserveCapacity(WhisperTinyConstants.layerCount)

        for i in 0..<WhisperTinyConstants.layerCount {
            let layer = layerWeights[i]
            keysByLayer.append(
                linear(
                    encoderOutput,
                    weight: layer.crossKW,
                    bias: nil,
                    rows: sourceLength,
                    cols: dm,
                    outCols: dm
                )
            )
            valuesByLayer.append(
                linear(
                    encoderOutput,
                    weight: layer.crossVW,
                    bias: layer.crossVB,
                    rows: sourceLength,
                    cols: dm,
                    outCols: dm
                )
            )
        }

        return CrossAttentionCache(keysByLayer: keysByLayer, valuesByLayer: valuesByLayer)
    }

    public func makeState(
        encoderOutput: [Float],
        sourceLength: Int = WhisperTinyConstants.sourceLength
    ) -> State {
        let dm = WhisperTinyConstants.dModel
        let layerCount = WhisperTinyConstants.layerCount
        let maxSeq = WhisperTinyConstants.maxTargetPositions

        return State(
            crossAttentionCache: prepareCrossAttentionCache(
                encoderOutput: encoderOutput,
                sourceLength: sourceLength
            ),
            selfKeyCache: Array(
                repeating: [Float](repeating: 0, count: maxSeq * dm),
                count: layerCount
            ),
            selfValueCache: Array(
                repeating: [Float](repeating: 0, count: maxSeq * dm),
                count: layerCount
            ),
            position: 0
        )
    }

    public func decodeStep(
        token: Int,
        state: inout State,
        sourceLength: Int = WhisperTinyConstants.sourceLength
    ) -> WhisperTinyStepResult {
        let dm = WhisperTinyConstants.dModel
        let heads = WhisperTinyConstants.heads
        let headDim = WhisperTinyConstants.headDim
        let ffnDim = WhisperTinyConstants.ffnDim
        let vocabSize = WhisperTinyConstants.vocabSize
        let layerCount = WhisperTinyConstants.layerCount
        let scale = 1.0 / sqrt(Float(headDim))
        let pos = state.position

        precondition(pos < WhisperTinyConstants.maxTargetPositions)

        var hidden = [Float](repeating: 0, count: dm)
        let tokBase = token * dm
        let posBase = pos * dm
        for d in 0..<dm {
            hidden[d] = tokEmbW[tokBase + d] + posEmbW[posBase + d]
        }

        var crossAttnWeightsByLayer = Array(repeating: [Float](), count: layerCount)

        for layerIndex in 0..<layerCount {
            let layer = layerWeights[layerIndex]

            let normed = layerNorm(
                hidden,
                weight: layer.selfAttnLnW,
                bias: layer.selfAttnLnB,
                dim: dm
            )

            var q = linear(
                normed,
                weight: layer.selfQW,
                bias: layer.selfQB,
                rows: 1,
                cols: dm,
                outCols: dm
            )
            for i in 0..<q.count {
                q[i] *= scale
            }

            let kStep = linear(
                normed,
                weight: layer.selfKW,
                bias: nil,
                rows: 1,
                cols: dm,
                outCols: dm
            )
            let vStep = linear(
                normed,
                weight: layer.selfVW,
                bias: layer.selfVB,
                rows: 1,
                cols: dm,
                outCols: dm
            )

            for d in 0..<dm {
                state.selfKeyCache[layerIndex][pos * dm + d] = kStep[d]
                state.selfValueCache[layerIndex][pos * dm + d] = vStep[d]
            }

            let saOut = selfAttentionSingleStep(
                query: q,
                keyCache: state.selfKeyCache[layerIndex],
                valueCache: state.selfValueCache[layerIndex],
                cacheLength: pos + 1,
                heads: heads,
                headDim: headDim
            )

            let saProjOut = linear(
                saOut,
                weight: layer.selfOW,
                bias: layer.selfOB,
                rows: 1,
                cols: dm,
                outCols: dm
            )
            hidden = addVec(hidden, saProjOut)

            let caNormed = layerNorm(
                hidden,
                weight: layer.crossAttnLnW,
                bias: layer.crossAttnLnB,
                dim: dm
            )

            var caQ = linear(
                caNormed,
                weight: layer.crossQW,
                bias: layer.crossQB,
                rows: 1,
                cols: dm,
                outCols: dm
            )
            for i in 0..<caQ.count {
                caQ[i] *= scale
            }

            let (caOut, caWeights) = crossAttentionSingleStep(
                query: caQ,
                key: state.crossAttentionCache.keysByLayer[layerIndex],
                value: state.crossAttentionCache.valuesByLayer[layerIndex],
                sourceLength: sourceLength,
                heads: heads,
                headDim: headDim
            )
            crossAttnWeightsByLayer[layerIndex] = caWeights

            let caProjOut = linear(
                caOut,
                weight: layer.crossOW,
                bias: layer.crossOB,
                rows: 1,
                cols: dm,
                outCols: dm
            )
            hidden = addVec(hidden, caProjOut)

            let ffnNormed = layerNorm(
                hidden,
                weight: layer.finalLnW,
                bias: layer.finalLnB,
                dim: dm
            )
            var fc1 = linear(
                ffnNormed,
                weight: layer.fc1W,
                bias: layer.fc1B,
                rows: 1,
                cols: dm,
                outCols: ffnDim
            )
            fc1 = gelu(fc1)
            let fc2 = linear(
                fc1,
                weight: layer.fc2W,
                bias: layer.fc2B,
                rows: 1,
                cols: ffnDim,
                outCols: dm
            )
            hidden = addVec(hidden, fc2)
        }

        let alignmentRow = alignmentRow(
            crossAttnWeightsByLayer: crossAttnWeightsByLayer,
            sourceLength: sourceLength
        )

        let finalHidden = layerNorm(
            hidden,
            weight: finalLnW,
            bias: finalLnB,
            dim: dm
        )
        let logits = linear(
            finalHidden,
            weight: lmHeadW,
            bias: nil,
            rows: 1,
            cols: dm,
            outCols: vocabSize
        )

        state.position += 1
        return WhisperTinyStepResult(logits: logits, alignmentRow: alignmentRow)
    }

    public func greedyDecode(
        encoderOutput: [Float],
        sourceLength: Int = WhisperTinyConstants.sourceLength,
        initialPrompt: [Int] = WhisperTinyConstants.forcedPrefix,
        maxSteps: Int = WhisperTinyConstants.maxTargetPositions
    ) -> WhisperTinyGreedyResult {
        precondition(!initialPrompt.isEmpty)
        var state = makeState(encoderOutput: encoderOutput, sourceLength: sourceLength)
        var tokens: [Int] = []
        var alignmentRows: [[Float]] = []
        var sampledToken: Int?

        for step in 0..<maxSteps {
            let currentToken: Int
            if step < initialPrompt.count {
                currentToken = initialPrompt[step]
            } else if let sampledToken {
                currentToken = sampledToken
            } else {
                break
            }

            tokens.append(currentToken)

            let stepResult = decodeStep(
                token: currentToken,
                state: &state,
                sourceLength: sourceLength
            )
            alignmentRows.append(stepResult.alignmentRow)

            let nextToken = argmax(stepResult.logits)
            sampledToken = nextToken

            if step >= initialPrompt.count - 1, nextToken == WhisperTinyConstants.endOfText {
                tokens.append(nextToken)
                break
            }
        }

        return WhisperTinyGreedyResult(tokens: tokens, alignmentRows: alignmentRows)
    }

    private func alignmentRow(
        crossAttnWeightsByLayer: [[Float]],
        sourceLength: Int
    ) -> [Float] {
        var stepAlign = [Float](repeating: 0, count: sourceLength)
        for (layerIndex, headIndex) in WhisperTinyConstants.alignmentHeads {
            let weights = crossAttnWeightsByLayer[layerIndex]
            let base = headIndex * sourceLength
            for j in 0..<sourceLength {
                stepAlign[j] += weights[base + j]
            }
        }
        let scale = 1.0 / Float(WhisperTinyConstants.alignmentHeads.count)
        for i in 0..<stepAlign.count {
            stepAlign[i] *= scale
        }
        return stepAlign
    }
}

private func layerNorm(
    _ x: [Float],
    weight: [Float],
    bias: [Float],
    dim: Int
) -> [Float] {
    precondition(x.count % dim == 0)
    let rows = x.count / dim
    var out = [Float](repeating: 0, count: x.count)
    for row in 0..<rows {
        let offset = row * dim
        var mean: Float = 0
        for i in 0..<dim {
            mean += x[offset + i]
        }
        mean /= Float(dim)

        var variance: Float = 0
        for i in 0..<dim {
            let dx = x[offset + i] - mean
            variance += dx * dx
        }
        variance /= Float(dim)
        let scale = 1.0 / sqrt(variance + 1e-5)

        for i in 0..<dim {
            out[offset + i] = (x[offset + i] - mean) * scale * weight[i] + bias[i]
        }
    }
    return out
}

private func linear(
    _ x: [Float],
    weight: [Float],
    bias: [Float]?,
    rows: Int,
    cols: Int,
    outCols: Int
) -> [Float] {
    var out = [Float](repeating: 0, count: rows * outCols)

    x.withUnsafeBufferPointer { xBuffer in
        weight.withUnsafeBufferPointer { weightBuffer in
            out.withUnsafeMutableBufferPointer { outBuffer in
                let xBase = xBuffer.baseAddress!
                let wBase = weightBuffer.baseAddress!
                let outBase = outBuffer.baseAddress!

                if rows == 1 {
                    cblas_sgemv(
                        CblasRowMajor,
                        CblasNoTrans,
                        Int32(outCols),
                        Int32(cols),
                        1,
                        wBase,
                        Int32(cols),
                        xBase,
                        1,
                        0,
                        outBase,
                        1
                    )
                } else {
                    cblas_sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasTrans,
                        Int32(rows),
                        Int32(outCols),
                        Int32(cols),
                        1,
                        xBase,
                        Int32(cols),
                        wBase,
                        Int32(cols),
                        0,
                        outBase,
                        Int32(outCols)
                    )
                }
            }
        }
    }

    if let bias {
        for row in 0..<rows {
            let base = row * outCols
            for outCol in 0..<outCols {
                out[base + outCol] += bias[outCol]
            }
        }
    }
    return out
}

private func gelu(_ x: [Float]) -> [Float] {
    x.map { value in
        0.5 * value * (1.0 + Float(erf(Double(value) / sqrt(2.0))))
    }
}

private func addVec(_ a: [Float], _ b: [Float]) -> [Float] {
    precondition(a.count == b.count)
    var out = [Float](repeating: 0, count: a.count)
    for i in 0..<a.count {
        out[i] = a[i] + b[i]
    }
    return out
}

private func selfAttentionSingleStep(
    query: [Float],
    keyCache: [Float],
    valueCache: [Float],
    cacheLength: Int,
    heads: Int,
    headDim: Int
) -> [Float] {
    let dModel = heads * headDim
    var output = [Float](repeating: 0, count: dModel)

    for head in 0..<heads {
        var scores = [Float](repeating: 0, count: cacheLength)
        var maxScore = -Float.infinity
        for position in 0..<cacheLength {
            var dot: Float = 0
            for d in 0..<headDim {
                dot += query[head * headDim + d] * keyCache[position * dModel + head * headDim + d]
            }
            scores[position] = dot
            maxScore = max(maxScore, dot)
        }

        var sumExp: Float = 0
        for position in 0..<cacheLength {
            scores[position] = exp(scores[position] - maxScore)
            sumExp += scores[position]
        }
        for position in 0..<cacheLength {
            scores[position] /= sumExp
        }

        for d in 0..<headDim {
            var value: Float = 0
            for position in 0..<cacheLength {
                value += scores[position] * valueCache[position * dModel + head * headDim + d]
            }
            output[head * headDim + d] = value
        }
    }

    return output
}

private func crossAttentionSingleStep(
    query: [Float],
    key: [Float],
    value: [Float],
    sourceLength: Int,
    heads: Int,
    headDim: Int
) -> (out: [Float], weights: [Float]) {
    let dModel = heads * headDim
    var output = [Float](repeating: 0, count: dModel)
    var allWeights = [Float](repeating: 0, count: heads * sourceLength)

    for head in 0..<heads {
        var scores = [Float](repeating: 0, count: sourceLength)
        var maxScore = -Float.infinity
        for sourceIndex in 0..<sourceLength {
            var dot: Float = 0
            for d in 0..<headDim {
                dot += query[head * headDim + d] * key[sourceIndex * dModel + head * headDim + d]
            }
            scores[sourceIndex] = dot
            maxScore = max(maxScore, dot)
        }

        var sumExp: Float = 0
        for sourceIndex in 0..<sourceLength {
            scores[sourceIndex] = exp(scores[sourceIndex] - maxScore)
            sumExp += scores[sourceIndex]
        }
        for sourceIndex in 0..<sourceLength {
            scores[sourceIndex] /= sumExp
            allWeights[head * sourceLength + sourceIndex] = scores[sourceIndex]
        }

        for d in 0..<headDim {
            var valueOut: Float = 0
            for sourceIndex in 0..<sourceLength {
                valueOut += scores[sourceIndex] * value[sourceIndex * dModel + head * headDim + d]
            }
            output[head * headDim + d] = valueOut
        }
    }

    return (output, allWeights)
}

private func argmax(_ values: [Float]) -> Int {
    var bestIndex = 0
    var bestValue = values[0]
    for i in 1..<values.count {
        if values[i] > bestValue {
            bestValue = values[i]
            bestIndex = i
        }
    }
    return bestIndex
}

public enum DecoderError: Error, CustomStringConvertible {
    case weightNotFound(String)
    case unsupportedDType(name: String, dtype: String)

    public var description: String {
        switch self {
        case let .weightNotFound(name):
            return "Weight not found: \(name)"
        case let .unsupportedDType(name, dtype):
            return "Unsupported dtype '\(dtype)' for tensor \(name)"
        }
    }
}
