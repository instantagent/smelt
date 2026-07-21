// NoncausalAttentionKernelTests — bring-up gate for the shared Michelangelo /
// SkinVAE non-causal attention primitive.

import Foundation
import Metal
import XCTest

@testable import SmeltCompiler

final class NoncausalAttentionKernelTests: XCTestCase {
    private struct AttentionFixtureMetadata: Decodable {
        let queryTokens: Int
        let keyValueTokens: Int
        let heads: Int
        let headDimension: Int
    }

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
            let angle = Float(index &* 37 &+ seed &* 101) * 0.017
            return sin(angle) * 0.43 + cos(angle * 0.37) * 0.19
        }
    }

    private func read(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(
            UnsafeBufferPointer(
                start: buffer.contents().bindMemory(to: Float.self, capacity: count),
                count: count
            )
        )
    }

    private func encodeGeneric(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        q: MTLBuffer,
        k: MTLBuffer,
        v: MTLBuffer,
        output: MTLBuffer,
        queryTokens: Int,
        keyValueTokens: Int,
        heads: Int,
        headDim: Int
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(k, offset: 0, index: 1)
        encoder.setBuffer(v, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var queryTokens = UInt32(queryTokens)
        var keyValueTokens = UInt32(keyValueTokens)
        var heads = UInt32(heads)
        var headDim = UInt32(headDim)
        encoder.setBytes(&queryTokens, length: 4, index: 4)
        encoder.setBytes(&keyValueTokens, length: 4, index: 5)
        encoder.setBytes(&heads, length: 4, index: 6)
        encoder.setBytes(&headDim, length: 4, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(queryTokens), height: Int(heads), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    private func encodeQ8(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        q: MTLBuffer,
        k: MTLBuffer,
        v: MTLBuffer,
        output: MTLBuffer,
        queryTokens: Int,
        keyValueTokens: Int,
        heads: Int,
        headDim: Int
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(k, offset: 0, index: 1)
        encoder.setBuffer(v, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var queryTokens = UInt32(queryTokens)
        var keyValueTokens = UInt32(keyValueTokens)
        var heads = UInt32(heads)
        var headDim = UInt32(headDim)
        encoder.setBytes(&queryTokens, length: 4, index: 4)
        encoder.setBytes(&keyValueTokens, length: 4, index: 5)
        encoder.setBytes(&heads, length: 4, index: 6)
        encoder.setBytes(&headDim, length: 4, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (Int(queryTokens) + 7) / 8,
                height: Int(heads),
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
    }

    private func encodeQ16(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        q: MTLBuffer,
        k: MTLBuffer,
        v: MTLBuffer,
        output: MTLBuffer,
        queryTokens: Int,
        keyValueTokens: Int,
        heads: Int,
        headDim: Int
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(k, offset: 0, index: 1)
        encoder.setBuffer(v, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var queryTokens = UInt32(queryTokens)
        var keyValueTokens = UInt32(keyValueTokens)
        var heads = UInt32(heads)
        var headDim = UInt32(headDim)
        encoder.setBytes(&queryTokens, length: 4, index: 4)
        encoder.setBytes(&keyValueTokens, length: 4, index: 5)
        encoder.setBytes(&heads, length: 4, index: 6)
        encoder.setBytes(&headDim, length: 4, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (Int(queryTokens) + 15) / 16,
                height: Int(heads),
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1)
        )
    }

    private func encodeUpdate(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        q: MTLBuffer,
        k: MTLBuffer,
        v: MTLBuffer,
        accumulator: MTLBuffer,
        maximumState: MTLBuffer,
        denominatorState: MTLBuffer,
        queryTokens: Int,
        keyValueTokens: Int,
        heads: Int,
        headDim: Int,
        sourceStart: Int,
        sourceCount: Int,
        finalize: Bool
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(k, offset: 0, index: 1)
        encoder.setBuffer(v, offset: 0, index: 2)
        encoder.setBuffer(accumulator, offset: 0, index: 3)
        encoder.setBuffer(maximumState, offset: 0, index: 4)
        encoder.setBuffer(denominatorState, offset: 0, index: 5)
        var queryTokens = UInt32(queryTokens)
        var keyValueTokens = UInt32(keyValueTokens)
        var heads = UInt32(heads)
        var headDim = UInt32(headDim)
        var sourceStart = UInt32(sourceStart)
        var sourceCount = UInt32(sourceCount)
        var finalize: UInt32 = finalize ? 1 : 0
        encoder.setBytes(&queryTokens, length: 4, index: 6)
        encoder.setBytes(&keyValueTokens, length: 4, index: 7)
        encoder.setBytes(&heads, length: 4, index: 8)
        encoder.setBytes(&headDim, length: 4, index: 9)
        encoder.setBytes(&sourceStart, length: 4, index: 10)
        encoder.setBytes(&sourceCount, length: 4, index: 11)
        encoder.setBytes(&finalize, length: 4, index: 12)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(queryTokens), height: Int(heads), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    private func encodeVisionOracle(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        q: MTLBuffer,
        k: MTLBuffer,
        v: MTLBuffer,
        chunkStart: MTLBuffer,
        chunkEnd: MTLBuffer,
        output: MTLBuffer,
        tokens: Int,
        heads: Int,
        headDim: Int
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(k, offset: 0, index: 1)
        encoder.setBuffer(v, offset: 0, index: 2)
        encoder.setBuffer(chunkStart, offset: 0, index: 3)
        encoder.setBuffer(chunkEnd, offset: 0, index: 4)
        encoder.setBuffer(output, offset: 0, index: 5)
        var tokens = UInt32(tokens)
        var heads = UInt32(heads)
        var headDim = UInt32(headDim)
        encoder.setBytes(&tokens, length: 4, index: 6)
        encoder.setBytes(&heads, length: 4, index: 7)
        encoder.setBytes(&headDim, length: 4, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(tokens), height: Int(heads), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    func testEqualLengthPathIsBitExactWithVisionAttention() throws {
        let generic = try pipeline(
            shaderFile: "neural_primitives_f32.metal",
            functionName: "noncausal_attention_f32"
        )
        let oracle = try pipeline(
            shaderFile: "qwen35_vision.metal",
            functionName: "qwen35_vision_attention_f32"
        )
        let heads = 8
        let headDim = 64

        for tokens in [1, 31, 32, 33, 387, 388, 389, 511, 512, 513] {
            let count = tokens * heads * headDim
            let q = try makeSharedBuffer(
                device: device,
                deterministicValues(count: count, seed: 11 + tokens)
            )
            let k = try makeSharedBuffer(
                device: device,
                deterministicValues(count: count, seed: 23 + tokens)
            )
            let v = try makeSharedBuffer(
                device: device,
                deterministicValues(count: count, seed: 47 + tokens)
            )
            let starts = try makeSharedBuffer(device: device, [UInt32](repeating: 0, count: tokens))
            let ends = try makeSharedBuffer(
                device: device,
                [UInt32](repeating: UInt32(tokens), count: tokens)
            )
            let genericOutput = try makeSharedBuffer(device: device, count: count, of: Float.self)
            let oracleOutput = try makeSharedBuffer(device: device, count: count, of: Float.self)

            try runOnGPU(queue: queue) { encoder in
                encodeGeneric(
                    encoder,
                    pipeline: generic,
                    q: q,
                    k: k,
                    v: v,
                    output: genericOutput,
                    queryTokens: tokens,
                    keyValueTokens: tokens,
                    heads: heads,
                    headDim: headDim
                )
                encodeVisionOracle(
                    encoder,
                    pipeline: oracle,
                    q: q,
                    k: k,
                    v: v,
                    chunkStart: starts,
                    chunkEnd: ends,
                    output: oracleOutput,
                    tokens: tokens,
                    heads: heads,
                    headDim: headDim
                )
            }

            let actual = read(genericOutput, count: count)
            let expected = read(oracleOutput, count: count)
            XCTAssertTrue(actual.allSatisfy(\.isFinite))
            XCTAssertGreaterThan(actual.map(abs).max() ?? 0, 1e-3)
            for index in 0..<count {
                XCTAssertEqual(
                    actual[index].bitPattern,
                    expected[index].bitPattern,
                    "byte divergence at token count \(tokens), scalar \(index)"
                )
            }
        }
    }

    func testIndependentQueryAndKeyValueLengthsMatchCPU() throws {
        let attention = try pipeline(
            shaderFile: "neural_primitives_f32.metal",
            functionName: "noncausal_attention_f32"
        )
        let queryTokens = 3
        let keyValueTokens = 5
        let heads = 2
        let headDim = 64
        let hidden = heads * headDim
        let qValues = deterministicValues(count: queryTokens * hidden, seed: 71)
        let kValues = deterministicValues(count: keyValueTokens * hidden, seed: 89)
        let vValues = deterministicValues(count: keyValueTokens * hidden, seed: 107)
        let q = try makeSharedBuffer(device: device, qValues)
        let k = try makeSharedBuffer(device: device, kValues)
        let v = try makeSharedBuffer(device: device, vValues)
        let output = try makeSharedBuffer(
            device: device,
            count: queryTokens * hidden,
            of: Float.self
        )

        try runOnGPU(queue: queue) { encoder in
            encodeGeneric(
                encoder,
                pipeline: attention,
                q: q,
                k: k,
                v: v,
                output: output,
                queryTokens: queryTokens,
                keyValueTokens: keyValueTokens,
                heads: heads,
                headDim: headDim
            )
        }

        let actual = read(output, count: queryTokens * hidden)
        let expected = cpuAttention(
            q: qValues,
            k: kValues,
            v: vValues,
            queryTokens: queryTokens,
            keyValueTokens: keyValueTokens,
            heads: heads,
            headDim: headDim
        )
        XCTAssertTrue(actual.allSatisfy(\.isFinite))
        XCTAssertGreaterThan(actual.map(abs).max() ?? 0, 1e-3)
        var maximumDifference: Float = 0
        for index in 0..<actual.count {
            maximumDifference = max(maximumDifference, abs(actual[index] - expected[index]))
        }
        XCTAssertLessThan(maximumDifference, 2e-5)
    }

    func testEightQueryTileIsBitExactWithIndependentQueries() throws {
        let independent = try pipeline(
            shaderFile: "neural_primitives_f32.metal",
            functionName: "noncausal_attention_f32"
        )
        let q8 = try pipeline(
            shaderFile: "neural_primitives_f32.metal",
            functionName: "noncausal_attention_q8_f32"
        )
        let heads = 3
        let headDim = 64
        let hidden = heads * headDim
        for shape in [
            (queries: 8, keys: 5),
            (queries: 9, keys: 33),
            (queries: 33, keys: 388),
            (queries: 512, keys: 17),
        ] {
            let q = try makeSharedBuffer(
                device: device,
                deterministicValues(
                    count: shape.queries * hidden,
                    seed: 131 + shape.queries
                )
            )
            let k = try makeSharedBuffer(
                device: device,
                deterministicValues(
                    count: shape.keys * hidden,
                    seed: 173 + shape.keys
                )
            )
            let v = try makeSharedBuffer(
                device: device,
                deterministicValues(
                    count: shape.keys * hidden,
                    seed: 211 + shape.keys
                )
            )
            let expected = try makeSharedBuffer(
                device: device,
                count: shape.queries * hidden,
                of: Float.self
            )
            let actual = try makeSharedBuffer(
                device: device,
                count: shape.queries * hidden,
                of: Float.self
            )
            try runOnGPU(queue: queue) { encoder in
                encodeGeneric(
                    encoder,
                    pipeline: independent,
                    q: q,
                    k: k,
                    v: v,
                    output: expected,
                    queryTokens: shape.queries,
                    keyValueTokens: shape.keys,
                    heads: heads,
                    headDim: headDim
                )
                encodeQ8(
                    encoder,
                    pipeline: q8,
                    q: q,
                    k: k,
                    v: v,
                    output: actual,
                    queryTokens: shape.queries,
                    keyValueTokens: shape.keys,
                    heads: heads,
                    headDim: headDim
                )
            }
            let expectedValues = read(
                expected,
                count: shape.queries * hidden
            )
            let actualValues = read(
                actual,
                count: shape.queries * hidden
            )
            for index in expectedValues.indices {
                XCTAssertEqual(
                    actualValues[index].bitPattern,
                    expectedValues[index].bitPattern,
                    "Q=\(shape.queries) KV=\(shape.keys) index=\(index)"
                )
            }
        }
    }

    func testEightQueryTileProductionStress() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["SMELT_Q8_ATTENTION_STRESS"] == "1",
            "set SMELT_Q8_ATTENTION_STRESS=1 for the production-shape stress"
        )
        let independent = try pipeline(
            shaderFile: "neural_primitives_f32.metal",
            functionName: "noncausal_attention_f32"
        )
        let q8 = try pipeline(
            shaderFile: "neural_primitives_f32.metal",
            functionName: "noncausal_attention_q8_f32"
        )
        let repeatCount =
            Int(
                ProcessInfo.processInfo.environment["SMELT_Q8_ATTENTION_STRESS_REPEATS"]
                    ?? "128"
            ) ?? 128
        XCTAssertGreaterThan(repeatCount, 1)
        let heads = 12
        let headDim = 64
        let hidden = heads * headDim
        for shape in [
            (queries: 388, keys: 388),
            (queries: 512, keys: 388),
        ] {
            let q = try makeSharedBuffer(
                device: device,
                deterministicValues(
                    count: shape.queries * hidden,
                    seed: 307 + shape.queries
                )
            )
            let k = try makeSharedBuffer(
                device: device,
                deterministicValues(
                    count: shape.keys * hidden,
                    seed: 401 + shape.keys
                )
            )
            let v = try makeSharedBuffer(
                device: device,
                deterministicValues(
                    count: shape.keys * hidden,
                    seed: 503 + shape.keys
                )
            )
            let expected = try makeSharedBuffer(
                device: device,
                count: shape.queries * hidden,
                of: Float.self
            )
            let actual = try makeSharedBuffer(
                device: device,
                count: shape.queries * hidden,
                of: Float.self
            )
            try runOnGPU(queue: queue) { encoder in
                encodeGeneric(
                    encoder,
                    pipeline: independent,
                    q: q,
                    k: k,
                    v: v,
                    output: expected,
                    queryTokens: shape.queries,
                    keyValueTokens: shape.keys,
                    heads: heads,
                    headDim: headDim
                )
            }
            let expectedValues = read(
                expected,
                count: shape.queries * hidden
            )
            for repetition in 0..<repeatCount {
                try runOnGPU(queue: queue) { encoder in
                    encodeQ8(
                        encoder,
                        pipeline: q8,
                        q: q,
                        k: k,
                        v: v,
                        output: actual,
                        queryTokens: shape.queries,
                        keyValueTokens: shape.keys,
                        heads: heads,
                        headDim: headDim
                    )
                }
                let actualValues = read(
                    actual,
                    count: shape.queries * hidden
                )
                for index in expectedValues.indices {
                    XCTAssertEqual(
                        actualValues[index].bitPattern,
                        expectedValues[index].bitPattern,
                        "repetition=\(repetition) Q=\(shape.queries) "
                            + "KV=\(shape.keys) index=\(index)"
                    )
                }
            }
            print(
                "Q8_ATTENTION_STRESS queries=\(shape.queries) "
                    + "keys=\(shape.keys) repeats=\(repeatCount) mismatches=0"
            )
        }
    }

    func testSixteenQueryTileIsBitExactWithIndependentQueries() throws {
        let independent = try pipeline(
            shaderFile: "neural_primitives_f32.metal",
            functionName: "noncausal_attention_f32"
        )
        let q16 = try pipeline(
            shaderFile: "neural_primitives_f32.metal",
            functionName: "noncausal_attention_q16_f32"
        )
        let heads = 3
        let headDim = 64
        let hidden = heads * headDim
        for shape in [
            (queries: 16, keys: 5),
            (queries: 17, keys: 33),
            (queries: 388, keys: 388),
            (queries: 512, keys: 17),
        ] {
            let q = try makeSharedBuffer(
                device: device,
                deterministicValues(
                    count: shape.queries * hidden,
                    seed: 631 + shape.queries
                )
            )
            let k = try makeSharedBuffer(
                device: device,
                deterministicValues(
                    count: shape.keys * hidden,
                    seed: 673 + shape.keys
                )
            )
            let v = try makeSharedBuffer(
                device: device,
                deterministicValues(
                    count: shape.keys * hidden,
                    seed: 711 + shape.keys
                )
            )
            let expected = try makeSharedBuffer(
                device: device,
                count: shape.queries * hidden,
                of: Float.self
            )
            let actual = try makeSharedBuffer(
                device: device,
                count: shape.queries * hidden,
                of: Float.self
            )
            try runOnGPU(queue: queue) { encoder in
                encodeGeneric(
                    encoder,
                    pipeline: independent,
                    q: q,
                    k: k,
                    v: v,
                    output: expected,
                    queryTokens: shape.queries,
                    keyValueTokens: shape.keys,
                    heads: heads,
                    headDim: headDim
                )
                encodeQ16(
                    encoder,
                    pipeline: q16,
                    q: q,
                    k: k,
                    v: v,
                    output: actual,
                    queryTokens: shape.queries,
                    keyValueTokens: shape.keys,
                    heads: heads,
                    headDim: headDim
                )
            }
            let expectedValues = read(
                expected,
                count: shape.queries * hidden
            )
            let actualValues = read(
                actual,
                count: shape.queries * hidden
            )
            for index in expectedValues.indices {
                XCTAssertEqual(
                    actualValues[index].bitPattern,
                    expectedValues[index].bitPattern,
                    "Q=\(shape.queries) KV=\(shape.keys) index=\(index)"
                )
            }
        }
    }

    func testEightQueryTileExactRuntimeInputStress() throws {
        guard
            let fixturePath = ProcessInfo.processInfo.environment[
                "SMELT_Q8_ATTENTION_FIXTURE"
            ]
        else {
            throw XCTSkip("set SMELT_Q8_ATTENTION_FIXTURE to an exact Q/K/V capture")
        }
        let root = URL(fileURLWithPath: fixturePath, isDirectory: true)
        let metadata = try JSONDecoder().decode(
            AttentionFixtureMetadata.self,
            from: Data(contentsOf: root.appendingPathComponent("metadata.json"))
        )
        let qValues = try rawFloats(root.appendingPathComponent("q.f32"))
        let kValues = try rawFloats(root.appendingPathComponent("k.f32"))
        let vValues = try rawFloats(root.appendingPathComponent("v.f32"))
        let hidden = metadata.heads * metadata.headDimension
        XCTAssertEqual(qValues.count, metadata.queryTokens * hidden)
        XCTAssertEqual(kValues.count, metadata.keyValueTokens * hidden)
        XCTAssertEqual(vValues.count, metadata.keyValueTokens * hidden)

        let independent = try pipeline(
            shaderFile: "neural_primitives_f32.metal",
            functionName: "noncausal_attention_f32"
        )
        let q8 = try pipeline(
            shaderFile: "neural_primitives_f32.metal",
            functionName: "noncausal_attention_q8_f32"
        )
        let repeatCount =
            Int(
                ProcessInfo.processInfo.environment["SMELT_Q8_ATTENTION_STRESS_REPEATS"]
                    ?? "512"
            ) ?? 512
        XCTAssertGreaterThan(repeatCount, 1)
        let q = try makeSharedBuffer(device: device, qValues)
        let k = try makeSharedBuffer(device: device, kValues)
        let v = try makeSharedBuffer(device: device, vValues)
        let expected = try makeSharedBuffer(
            device: device,
            count: qValues.count,
            of: Float.self
        )
        let actual = try makeSharedBuffer(
            device: device,
            count: qValues.count,
            of: Float.self
        )
        try runOnGPU(queue: queue) { encoder in
            encodeGeneric(
                encoder,
                pipeline: independent,
                q: q,
                k: k,
                v: v,
                output: expected,
                queryTokens: metadata.queryTokens,
                keyValueTokens: metadata.keyValueTokens,
                heads: metadata.heads,
                headDim: metadata.headDimension
            )
        }
        let expectedValues = read(expected, count: qValues.count)
        for repetition in 0..<repeatCount {
            try runOnGPU(queue: queue) { encoder in
                encodeQ8(
                    encoder,
                    pipeline: q8,
                    q: q,
                    k: k,
                    v: v,
                    output: actual,
                    queryTokens: metadata.queryTokens,
                    keyValueTokens: metadata.keyValueTokens,
                    heads: metadata.heads,
                    headDim: metadata.headDimension
                )
            }
            let actualValues = read(actual, count: qValues.count)
            for index in expectedValues.indices {
                XCTAssertEqual(
                    actualValues[index].bitPattern,
                    expectedValues[index].bitPattern,
                    "repetition=\(repetition) index=\(index)"
                )
            }
        }
        print(
            "Q8_EXACT_INPUT_STRESS queries=\(metadata.queryTokens) "
                + "keys=\(metadata.keyValueTokens) repeats=\(repeatCount) "
                + "mismatches=0"
        )
    }

    private func rawFloats(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        XCTAssertTrue(data.count.isMultiple(of: MemoryLayout<Float>.stride))
        return data.withUnsafeBytes { bytes in
            Array(bytes.bindMemory(to: Float.self))
        }
    }

    func testStatefulKeyChunksAreBitExactWithMonolithicAttention() throws {
        let monolithic = try pipeline(
            shaderFile: "neural_primitives_f32.metal",
            functionName: "noncausal_attention_f32"
        )
        let update = try pipeline(
            shaderFile: "neural_primitives_f32.metal",
            functionName: "noncausal_attention_update_f32"
        )
        let queryTokens = 3
        let heads = 2
        let headDim = 64
        let hidden = heads * headDim

        for (keyValueTokens, chunkSize) in [(5, 2), (2_051, 1_024)] {
            let q = try makeSharedBuffer(
                device: device,
                deterministicValues(
                    count: queryTokens * hidden,
                    seed: 131 + keyValueTokens
                )
            )
            let k = try makeSharedBuffer(
                device: device,
                deterministicValues(
                    count: keyValueTokens * hidden,
                    seed: 149 + keyValueTokens
                )
            )
            let v = try makeSharedBuffer(
                device: device,
                deterministicValues(
                    count: keyValueTokens * hidden,
                    seed: 167 + keyValueTokens
                )
            )
            let expected = try makeSharedBuffer(
                device: device,
                count: queryTokens * hidden,
                of: Float.self
            )
            let actual = try makeSharedBuffer(
                device: device,
                [Float](repeating: 0, count: queryTokens * hidden)
            )
            let maximumState = try makeSharedBuffer(
                device: device,
                [Float](repeating: -.infinity, count: queryTokens * heads)
            )
            let denominatorState = try makeSharedBuffer(
                device: device,
                [Float](repeating: 0, count: queryTokens * heads)
            )

            try runOnGPU(queue: queue) { encoder in
                encodeGeneric(
                    encoder,
                    pipeline: monolithic,
                    q: q,
                    k: k,
                    v: v,
                    output: expected,
                    queryTokens: queryTokens,
                    keyValueTokens: keyValueTokens,
                    heads: heads,
                    headDim: headDim
                )
            }
            var sourceStart = 0
            while sourceStart < keyValueTokens {
                let sourceCount = min(chunkSize, keyValueTokens - sourceStart)
                try runOnGPU(queue: queue) { encoder in
                    encodeUpdate(
                        encoder,
                        pipeline: update,
                        q: q,
                        k: k,
                        v: v,
                        accumulator: actual,
                        maximumState: maximumState,
                        denominatorState: denominatorState,
                        queryTokens: queryTokens,
                        keyValueTokens: keyValueTokens,
                        heads: heads,
                        headDim: headDim,
                        sourceStart: sourceStart,
                        sourceCount: sourceCount,
                        finalize: sourceStart + sourceCount == keyValueTokens
                    )
                }
                sourceStart += sourceCount
            }

            let expectedValues = read(expected, count: queryTokens * hidden)
            let actualValues = read(actual, count: queryTokens * hidden)
            for index in actualValues.indices {
                XCTAssertEqual(
                    actualValues[index].bitPattern,
                    expectedValues[index].bitPattern,
                    "chunked attention diverged at K=\(keyValueTokens), scalar \(index)"
                )
            }
        }
    }

    private func cpuAttention(
        q: [Float],
        k: [Float],
        v: [Float],
        queryTokens: Int,
        keyValueTokens: Int,
        heads: Int,
        headDim: Int
    ) -> [Float] {
        let hidden = heads * headDim
        let scale = 1 / Float(headDim).squareRoot()
        var output = [Float](repeating: 0, count: queryTokens * hidden)
        for query in 0..<queryTokens {
            for head in 0..<heads {
                let queryBase = query * hidden + head * headDim
                var scores = [Float](repeating: 0, count: keyValueTokens)
                for source in 0..<keyValueTokens {
                    let sourceBase = source * hidden + head * headDim
                    for dimension in 0..<headDim {
                        scores[source] += q[queryBase + dimension] * k[sourceBase + dimension]
                    }
                    scores[source] *= scale
                }
                let maximum = scores.max() ?? -.infinity
                var denominator: Float = 0
                for source in 0..<keyValueTokens {
                    scores[source] = exp(scores[source] - maximum)
                    denominator += scores[source]
                }
                for dimension in 0..<headDim {
                    var accumulator: Float = 0
                    for source in 0..<keyValueTokens {
                        let sourceBase = source * hidden + head * headDim
                        accumulator += scores[source] * v[sourceBase + dimension]
                    }
                    output[queryBase + dimension] = accumulator / denominator
                }
            }
        }
        return output
    }
}
