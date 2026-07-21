import XCTest
@testable import WhisperTinyDecoderSpike

@MainActor
final class WhisperTinyDecoderSpikeTests: XCTestCase {
    private let expectedTokens: [Int] = [
        50258, 50259, 50359, 50363, 5135, 11, 820, 321, 3126, 411,
        1320, 3061, 293, 1507, 30, 823, 300, 321, 600, 658,
        1101, 3547, 13, 50257
    ]

    private static let sharedRun = Result<WhisperTinyGreedyResult, Error> {
        let decoder = try WhisperTinyCPUDecoder()
        let encoderOutput = try decoder.loadReferenceEncoderOutput()
        return decoder.greedyDecode(
            encoderOutput: encoderOutput,
            maxSteps: 24
        )
    }

    private func requireReferenceAssets() throws {
        do {
            _ = try WhisperTinyReferenceAssets.repoRoot()
        } catch ReferenceAssetError.repoRootNotFound {
            throw XCTSkip("tools/whisper_ref is not present in this checkout")
        }
    }

    private func referenceRun() throws -> WhisperTinyGreedyResult {
        try requireReferenceAssets()
        return try Self.sharedRun.get()
    }

    func testGreedyDecodeMatchesReferenceTokens() throws {
        let result = try referenceRun()
        XCTAssertEqual(result.tokens, expectedTokens)
    }

    func testAlignmentWeightsMatchReference() throws {
        let result = try referenceRun()

        let (reference, shape) = try WhisperTinyReferenceAssets.loadReference(name: "alignment_weights_correct")
        let ourCount = result.alignmentRows.count
        let refCount = shape[0]
        let offset = ourCount - refCount
        XCTAssertGreaterThanOrEqual(offset, 0)

        let flattened = Array(result.alignmentRows[offset...]).flatMap { $0 }
        XCTAssertEqual(flattened.count, reference.count)

        let cosine = cosineSim(flattened, reference)
        let maxAbs = maxAbsDiff(flattened, reference)
        XCTAssertGreaterThanOrEqual(cosine, 0.999)
        XCTAssertLessThanOrEqual(maxAbs, 0.01)
    }

    func testStepApiMatchesGreedyDecode() throws {
        try requireReferenceAssets()
        let decoder = try WhisperTinyCPUDecoder()
        let encoderOutput = try decoder.loadReferenceEncoderOutput()
        var state = decoder.makeState(encoderOutput: encoderOutput)

        var tokens: [Int] = []
        var alignmentRows: [[Float]] = []
        var sampledToken: Int?

        for step in 0..<expectedTokens.count {
            let token: Int
            if step < WhisperTinyConstants.forcedPrefix.count {
                token = WhisperTinyConstants.forcedPrefix[step]
            } else if let sampledToken {
                token = sampledToken
            } else {
                XCTFail("Missing sampled token for decode step")
                return
            }

            tokens.append(token)
            let result = decoder.decodeStep(token: token, state: &state)
            alignmentRows.append(result.alignmentRow)
            sampledToken = argmax(result.logits)

            if step >= WhisperTinyConstants.forcedPrefix.count - 1,
               sampledToken == WhisperTinyConstants.endOfText {
                tokens.append(WhisperTinyConstants.endOfText)
                break
            }
        }

        let sharedRun = try referenceRun()
        XCTAssertEqual(tokens, sharedRun.tokens)

        let flattened = alignmentRows.flatMap { $0 }
        let sharedFlattened = sharedRun.alignmentRows.flatMap { $0 }
        XCTAssertEqual(flattened.count, sharedFlattened.count)
        XCTAssertLessThanOrEqual(maxAbsDiff(flattened, sharedFlattened), 1e-6)
    }
}

private func cosineSim(_ a: [Float], _ b: [Float]) -> Float {
    precondition(a.count == b.count)
    var dot: Float = 0
    var normA: Float = 0
    var normB: Float = 0
    for i in 0..<a.count {
        dot += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    return dot / (sqrt(normA) * sqrt(normB) + 1e-10)
}

private func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
    precondition(a.count == b.count)
    var maxDiff: Float = 0
    for i in 0..<a.count {
        maxDiff = max(maxDiff, abs(a[i] - b[i]))
    }
    return maxDiff
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
