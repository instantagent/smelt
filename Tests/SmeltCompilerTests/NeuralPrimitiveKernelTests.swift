// NeuralPrimitiveKernelTests — exactness gates for shared rig model
// preprocessing and normalization primitives.

import Foundation
import Metal
import XCTest

@testable import SmeltCompiler

final class NeuralPrimitiveKernelTests: XCTestCase {
    private var device: MTLDevice!
    private var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        try? XCTSkipIf(device == nil, "No Metal device available")
        queue = device?.makeCommandQueue()
    }

    private func pipeline(
        shaderFile: String,
        functionName: String
    ) throws -> MTLComputePipelineState {
        guard let source = loadMetalShaderSource(shaderFile) else {
            throw XCTSkip("Shader source not found: \(shaderFile)")
        }
        let library = try device.makeLibrary(source: source, options: nil)
        let function = try XCTUnwrap(
            library.makeFunction(name: functionName),
            "Metal function not found: \(functionName)"
        )
        return try device.makeComputePipelineState(function: function)
    }

    private func deterministicValues(count: Int, seed: Int) -> [Float] {
        (0..<count).map { index in
            let angle = Float(index &* 53 &+ seed &* 97) * 0.0091
            return sin(angle) * 0.73 + cos(angle * 0.29) * 0.17
        }
    }

    private func readFloat(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(
            UnsafeBufferPointer(
                start: buffer.contents().bindMemory(to: Float.self, capacity: count),
                count: count
            )
        )
    }

    private func encodeLayerNorm(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        input: MTLBuffer,
        weight: MTLBuffer,
        bias: MTLBuffer,
        output: MTLBuffer,
        rows: Int,
        dim: Int
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weight, offset: 0, index: 1)
        encoder.setBuffer(bias, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var rows = UInt32(rows)
        var dim = UInt32(dim)
        var epsilon: Float = 1e-5
        encoder.setBytes(&rows, length: 4, index: 4)
        encoder.setBytes(&dim, length: 4, index: 5)
        encoder.setBytes(&epsilon, length: 4, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(rows), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
    }

    func testLayerNormIsBitExactWithRetainedVisionPath() throws {
        let generic = try pipeline(
            shaderFile: "neural_primitives_f32.metal",
            functionName: "layer_norm_rows_f32"
        )
        let oracle = try pipeline(
            shaderFile: "qwen35_vision.metal",
            functionName: "qwen35_vision_layer_norm_f32"
        )
        let rows = 3

        for dim in [512, 768, 896] {
            let count = rows * dim
            let inputValues = deterministicValues(count: count, seed: 13 + dim)
            let weightValues = deterministicValues(count: dim, seed: 29 + dim).map { $0 + 1.1 }
            let biasValues = deterministicValues(count: dim, seed: 43 + dim).map { $0 * 0.1 }
            let input = try makeSharedBuffer(device: device, inputValues)
            let weight = try makeSharedBuffer(device: device, weightValues)
            let bias = try makeSharedBuffer(device: device, biasValues)
            let actualBuffer = try makeSharedBuffer(device: device, count: count, of: Float.self)
            let expectedBuffer = try makeSharedBuffer(device: device, count: count, of: Float.self)

            try runOnGPU(queue: queue) { encoder in
                encodeLayerNorm(
                    encoder,
                    pipeline: generic,
                    input: input,
                    weight: weight,
                    bias: bias,
                    output: actualBuffer,
                    rows: rows,
                    dim: dim
                )
                encodeLayerNorm(
                    encoder,
                    pipeline: oracle,
                    input: input,
                    weight: weight,
                    bias: bias,
                    output: expectedBuffer,
                    rows: rows,
                    dim: dim
                )
            }

            let actual = readFloat(actualBuffer, count: count)
            let expected = readFloat(expectedBuffer, count: count)
            XCTAssertTrue(actual.allSatisfy(\.isFinite))
            XCTAssertGreaterThan(actual.map(abs).max() ?? 0, 1e-3)
            for index in 0..<count {
                XCTAssertEqual(
                    actual[index].bitPattern,
                    expected[index].bitPattern,
                    "LayerNorm byte divergence at dim \(dim), scalar \(index)"
                )
            }
        }
    }

    func testFourierEmbeddingMatchesSourceFormula() throws {
        let embedding = try pipeline(
            shaderFile: "neural_primitives_f32.metal",
            functionName: "fourier_position_embedding_f32"
        )
        let rows = 4
        let inputDim = 3
        let numFreqs = 8
        let inputValues: [Float] = [
            -0.91, -0.37, 0.13,
            0.0, 0.25, 0.75,
            0.99, -0.51, 0.42,
            0.17, -0.83, 0.61,
        ]

        for (includePi, usePMPE) in [(false, false), (true, false), (true, true)] {
            let outputDim = inputDim * (numFreqs * 2 + 1)
            let input = try makeSharedBuffer(device: device, inputValues)
            let output = try makeSharedBuffer(
                device: device,
                count: rows * outputDim,
                of: Float.self
            )
            try runOnGPU(queue: queue) { encoder in
                encoder.setComputePipelineState(embedding)
                encoder.setBuffer(input, offset: 0, index: 0)
                encoder.setBuffer(output, offset: 0, index: 1)
                var rowsValue = UInt32(rows)
                var inputDimValue = UInt32(inputDim)
                var numFreqsValue = UInt32(numFreqs)
                var includeInputValue: UInt32 = 1
                var includePiValue: UInt32 = includePi ? 1 : 0
                var usePMPEValue: UInt32 = usePMPE ? 1 : 0
                var inputStrideValue = UInt32(inputDim)
                encoder.setBytes(&rowsValue, length: 4, index: 2)
                encoder.setBytes(&inputDimValue, length: 4, index: 3)
                encoder.setBytes(&numFreqsValue, length: 4, index: 4)
                encoder.setBytes(&includeInputValue, length: 4, index: 5)
                encoder.setBytes(&includePiValue, length: 4, index: 6)
                encoder.setBytes(&usePMPEValue, length: 4, index: 7)
                encoder.setBytes(&inputStrideValue, length: 4, index: 8)
                encoder.dispatchThreads(
                    MTLSize(width: outputDim, height: rows, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
                )
            }
            let actual = readFloat(output, count: rows * outputDim)
            let expected = cpuFourierEmbedding(
                input: inputValues,
                rows: rows,
                inputDim: inputDim,
                numFreqs: numFreqs,
                includePi: includePi,
                usePMPE: usePMPE
            )
            var maximumDifference: Float = 0
            for index in 0..<actual.count {
                maximumDifference = max(maximumDifference, abs(actual[index] - expected[index]))
            }
            XCTAssertTrue(actual.allSatisfy(\.isFinite))
            XCTAssertGreaterThan(actual.map(abs).max() ?? 0, 1e-3)
            // Metal and the host libm use different correctly-bounded trig
            // implementations. The observed worst case is 1.91e-5 at the
            // highest (128*pi) frequency, so this is a numerical agreement
            // gate, not an internal bit-identity claim.
            XCTAssertLessThan(
                maximumDifference,
                3e-5,
                "Fourier mismatch for includePi=\(includePi), usePMPE=\(usePMPE)"
            )
        }
    }

    func testFSQDecodeIsExhaustivelyExact() throws {
        let decode = try pipeline(
            shaderFile: "neural_primitives_f32.metal",
            functionName: "fsq_base8x5_decode_f32"
        )
        let count = 32_768
        let indices = (0..<count).map(UInt32.init)
        let input = try makeSharedBuffer(device: device, indices)
        let output = try makeSharedBuffer(device: device, count: count * 5, of: Float.self)
        try runOnGPU(queue: queue) { encoder in
            encoder.setComputePipelineState(decode)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(output, offset: 0, index: 1)
            var countValue = UInt32(count)
            encoder.setBytes(&countValue, length: 4, index: 2)
            encoder.dispatchThreads(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
        }
        let actual = readFloat(output, count: count * 5)
        for index in 0..<count {
            var basis = 1
            for level in 0..<5 {
                let digit = (index / basis) % 8
                let expected = (Float(digit) - 4) * 0.25
                XCTAssertEqual(
                    actual[index * 5 + level].bitPattern,
                    expected.bitPattern,
                    "FSQ divergence at index \(index), level \(level)"
                )
                basis *= 8
            }
        }
    }

    func testAppendStridedPointNormalsIsExact() throws {
        let append = try pipeline(
            shaderFile: "neural_primitives_f32.metal",
            functionName: "append_strided_features_f32"
        )
        let rows = 3
        let baseDim = 5
        let featureStride = 6
        let featureOffset = 3
        let featureCount = 3
        let baseValues = deterministicValues(count: rows * baseDim, seed: 151)
        let pointValues = deterministicValues(count: rows * featureStride, seed: 163)
        let base = try makeSharedBuffer(device: device, baseValues)
        let points = try makeSharedBuffer(device: device, pointValues)
        let outputDim = baseDim + featureCount
        let output = try makeSharedBuffer(
            device: device,
            count: rows * outputDim,
            of: Float.self
        )
        try runOnGPU(queue: queue) { encoder in
            encoder.setComputePipelineState(append)
            encoder.setBuffer(base, offset: 0, index: 0)
            encoder.setBuffer(points, offset: 0, index: 1)
            encoder.setBuffer(output, offset: 0, index: 2)
            var rowsValue = UInt32(rows)
            var baseDimValue = UInt32(baseDim)
            var featureStrideValue = UInt32(featureStride)
            var featureOffsetValue = UInt32(featureOffset)
            var featureCountValue = UInt32(featureCount)
            encoder.setBytes(&rowsValue, length: 4, index: 3)
            encoder.setBytes(&baseDimValue, length: 4, index: 4)
            encoder.setBytes(&featureStrideValue, length: 4, index: 5)
            encoder.setBytes(&featureOffsetValue, length: 4, index: 6)
            encoder.setBytes(&featureCountValue, length: 4, index: 7)
            encoder.dispatchThreads(
                MTLSize(width: outputDim, height: rows, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
            )
        }
        let actual = readFloat(output, count: rows * outputDim)
        for row in 0..<rows {
            for column in 0..<baseDim {
                XCTAssertEqual(
                    actual[row * outputDim + column].bitPattern,
                    baseValues[row * baseDim + column].bitPattern
                )
            }
            for feature in 0..<featureCount {
                XCTAssertEqual(
                    actual[row * outputDim + baseDim + feature].bitPattern,
                    pointValues[row * featureStride + featureOffset + feature].bitPattern
                )
            }
        }
    }

    private func cpuFourierEmbedding(
        input: [Float],
        rows: Int,
        inputDim: Int,
        numFreqs: Int,
        includePi: Bool,
        usePMPE: Bool
    ) -> [Float] {
        let outputDim = inputDim * (numFreqs * 2 + 1)
        var output = [Float](repeating: 0, count: rows * outputDim)
        for row in 0..<rows {
            let inputBase = row * inputDim
            let outputBase = row * outputDim
            for coordinate in 0..<inputDim {
                output[outputBase + coordinate] = input[inputBase + coordinate]
            }
            for coordinate in 0..<inputDim {
                for frequencyIndex in 0..<numFreqs {
                    let flattened = coordinate * numFreqs + frequencyIndex
                    let x = input[inputBase + coordinate]
                    var frequency = Float(1 << frequencyIndex)
                    if includePi { frequency *= Float.pi }
                    let angle = x * frequency
                    var sine = sin(angle)
                    var cosine = cos(angle)
                    if usePMPE {
                        let fraction = Float(frequencyIndex + 1) / Float(numFreqs)
                        let phase = (pow(Float(numFreqs), 1 - fraction) + fraction)
                            * (2 * Float.pi)
                        let phaseAngle = x * (0.5 * Float.pi) + phase
                        sine += sin(phaseAngle)
                        cosine += cos(phaseAngle)
                    }
                    output[outputBase + inputDim + flattened] = sine
                    output[outputBase + inputDim + inputDim * numFreqs + flattened] = cosine
                }
            }
        }
        return output
    }
}
