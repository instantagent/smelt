import Metal
import XCTest

@testable import SmeltCompiler

final class DenseBF16WeightKernelTests: XCTestCase {
    private var device: MTLDevice!
    private var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        try? XCTSkipIf(device == nil, "No Metal device available")
        queue = device?.makeCommandQueue()
    }

    private func pipeline(_ file: String, _ name: String) throws -> MTLComputePipelineState {
        guard let source = loadMetalShaderSource(file) else {
            throw XCTSkip("Shader source not found: \(file)")
        }
        let library = try device.makeLibrary(source: source, options: nil)
        let function = try XCTUnwrap(library.makeFunction(name: name))
        return try device.makeComputePipelineState(function: function)
    }

    private func bf16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let rounded = bits &+ (0x7fff &+ ((bits >> 16) & 1))
        return UInt16(truncatingIfNeeded: rounded >> 16)
    }

    private func float(_ bf16: UInt16) -> Float {
        Float(bitPattern: UInt32(bf16) << 16)
    }

    private func fixture(count: Int, seed: Int) -> [Float] {
        (0..<count).map { index in
            sin(Float(index &* 43 &+ seed &* 79) * 0.011) * 0.51
        }
    }

    private func encode(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        input: MTLBuffer,
        weight: MTLBuffer,
        bias: MTLBuffer,
        output: MTLBuffer,
        rows: Int,
        outDim: Int,
        inDim: Int,
        rowsPerThreadgroup: Int = 1
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weight, offset: 0, index: 1)
        encoder.setBuffer(bias, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var rows = UInt32(rows)
        var outDim = UInt32(outDim)
        var inDim = UInt32(inDim)
        var hasBias: UInt32 = 1
        encoder.setBytes(&rows, length: 4, index: 4)
        encoder.setBytes(&outDim, length: 4, index: 5)
        encoder.setBytes(&inDim, length: 4, index: 6)
        encoder.setBytes(&hasBias, length: 4, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: Int(outDim),
                height: (Int(rows) + rowsPerThreadgroup - 1)
                    / rowsPerThreadgroup,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(
                width: 32 * rowsPerThreadgroup,
                height: 1,
                depth: 1
            )
        )
    }

    private func encodeEpilogue(
        _ encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        input: MTLBuffer,
        weight: MTLBuffer,
        bias: MTLBuffer,
        residual: MTLBuffer,
        output: MTLBuffer,
        rows: Int,
        outDim: Int,
        inDim: Int,
        epilogue: UInt32
    ) {
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weight, offset: 0, index: 1)
        encoder.setBuffer(bias, offset: 0, index: 2)
        encoder.setBuffer(residual, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        var rows = UInt32(rows)
        var outDim = UInt32(outDim)
        var inDim = UInt32(inDim)
        var hasBias: UInt32 = 1
        var epilogue = epilogue
        encoder.setBytes(&rows, length: 4, index: 5)
        encoder.setBytes(&outDim, length: 4, index: 6)
        encoder.setBytes(&inDim, length: 4, index: 7)
        encoder.setBytes(&hasBias, length: 4, index: 8)
        encoder.setBytes(&epilogue, length: 4, index: 9)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: Int(outDim),
                height: (Int(rows) + 7) / 8,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
    }

    func testDivisibleWidthIsBitExactWithRetainedGEMM() throws {
        let dense = try pipeline("neural_primitives_f32.metal", "dense_bf16w_f32")
        let retained = try pipeline("gemm_bf16w_f32.metal", "gemm_bf16w_f32")
        let rows = 3
        let outDim = 7
        let inDim = 64
        let inputValues = fixture(count: rows * inDim, seed: 11)
        let weightValues = fixture(count: outDim * inDim, seed: 23).map(bf16)
        let biasValues = fixture(count: outDim, seed: 37).map(bf16)
        let biasFloat = biasValues.map(float)
        let input = try makeSharedBuffer(device: device, inputValues)
        let weight = try makeSharedBuffer(device: device, weightValues)
        let biasBF16 = try makeSharedBuffer(device: device, biasValues)
        let biasF32 = try makeSharedBuffer(device: device, biasFloat)
        let actualBuffer = try makeSharedBuffer(
            device: device,
            count: rows * outDim,
            of: Float.self
        )
        let expectedBuffer = try makeSharedBuffer(
            device: device,
            count: rows * outDim,
            of: Float.self
        )
        try runOnGPU(queue: queue) { encoder in
            encode(
                encoder,
                pipeline: dense,
                input: input,
                weight: weight,
                bias: biasBF16,
                output: actualBuffer,
                rows: rows,
                outDim: outDim,
                inDim: inDim
            )
            encode(
                encoder,
                pipeline: retained,
                input: input,
                weight: weight,
                bias: biasF32,
                output: expectedBuffer,
                rows: rows,
                outDim: outDim,
                inDim: inDim
            )
        }
        let actual = read(actualBuffer, count: rows * outDim)
        let expected = read(expectedBuffer, count: rows * outDim)
        for index in 0..<actual.count {
            XCTAssertEqual(actual[index].bitPattern, expected[index].bitPattern)
        }
    }

    func testRowTilesAreBitExactWithIndependentRows() throws {
        let independent = try pipeline(
            "neural_primitives_f32.metal",
            "dense_bf16w_f32"
        )
        let rows4 = try pipeline(
            "neural_primitives_f32.metal",
            "dense_bf16w_f32_rows4"
        )
        let rows8 = try pipeline(
            "neural_primitives_f32.metal",
            "dense_bf16w_f32_rows8"
        )
        let cases = [
            (rows: 5, outDim: 7, inDim: 5),
            (rows: 7, outDim: 9, inDim: 54),
            (rows: 9, outDim: 11, inDim: 64),
            (rows: 5, outDim: 13, inDim: 768),
            (rows: 5, outDim: 7, inDim: 3_072),
        ]
        for item in cases {
            let inputValues = fixture(
                count: item.rows * item.inDim,
                seed: 101 + item.inDim
            )
            let weightValues = fixture(
                count: item.outDim * item.inDim,
                seed: 211 + item.inDim
            ).map(bf16)
            let biasValues = fixture(
                count: item.outDim,
                seed: 307 + item.inDim
            ).map(bf16)
            let input = try makeSharedBuffer(device: device, inputValues)
            let weight = try makeSharedBuffer(device: device, weightValues)
            let bias = try makeSharedBuffer(device: device, biasValues)
            let expected = try makeSharedBuffer(
                device: device,
                count: item.rows * item.outDim,
                of: Float.self
            )
            let actual = try makeSharedBuffer(
                device: device,
                count: item.rows * item.outDim,
                of: Float.self
            )
            let actualRows8 = try makeSharedBuffer(
                device: device,
                count: item.rows * item.outDim,
                of: Float.self
            )
            try runOnGPU(queue: queue) { encoder in
                encode(
                    encoder,
                    pipeline: independent,
                    input: input,
                    weight: weight,
                    bias: bias,
                    output: expected,
                    rows: item.rows,
                    outDim: item.outDim,
                    inDim: item.inDim
                )
                encode(
                    encoder,
                    pipeline: rows4,
                    input: input,
                    weight: weight,
                    bias: bias,
                    output: actual,
                    rows: item.rows,
                    outDim: item.outDim,
                    inDim: item.inDim,
                    rowsPerThreadgroup: 4
                )
                encode(
                    encoder,
                    pipeline: rows8,
                    input: input,
                    weight: weight,
                    bias: bias,
                    output: actualRows8,
                    rows: item.rows,
                    outDim: item.outDim,
                    inDim: item.inDim,
                    rowsPerThreadgroup: 8
                )
            }
            let expectedValues = read(
                expected,
                count: item.rows * item.outDim
            )
            let actualValues = read(
                actual,
                count: item.rows * item.outDim
            )
            let actualRows8Values = read(
                actualRows8,
                count: item.rows * item.outDim
            )
            for index in expectedValues.indices {
                XCTAssertEqual(
                    actualValues[index].bitPattern,
                    expectedValues[index].bitPattern,
                    "rows=\(item.rows) N=\(item.outDim) K=\(item.inDim) "
                        + "index=\(index)"
                )
                XCTAssertEqual(
                    actualRows8Values[index].bitPattern,
                    expectedValues[index].bitPattern,
                    "rows8 rows=\(item.rows) N=\(item.outDim) "
                        + "K=\(item.inDim) index=\(index)"
                )
            }
        }
    }

    func testRows8EpiloguesAreBitExactWithStagedKernels() throws {
        let dense = try pipeline(
            "neural_primitives_f32.metal",
            "dense_bf16w_f32_rows8"
        )
        let epilogue = try pipeline(
            "neural_primitives_f32.metal",
            "dense_bf16w_f32_rows8_epilogue"
        )
        let gelu = try pipeline("activations_f32.metal", "gelu_f32")
        let add = try pipeline(
            "neural_primitives_f32.metal",
            "add_rows_f32"
        )
        let rows = 9
        let outDim = 13
        for inDim in [64, 768, 3_072] {
            let count = rows * outDim
            let input = try makeSharedBuffer(
                device: device,
                fixture(count: rows * inDim, seed: 401 + inDim)
            )
            let weight = try makeSharedBuffer(
                device: device,
                fixture(count: outDim * inDim, seed: 503 + inDim).map(bf16)
            )
            let bias = try makeSharedBuffer(
                device: device,
                fixture(count: outDim, seed: 607 + inDim).map(bf16)
            )
            let residual = try makeSharedBuffer(
                device: device,
                fixture(count: count, seed: 701 + inDim)
            )
            let stagedDense = try makeSharedBuffer(
                device: device,
                count: count,
                of: Float.self
            )
            let expectedGELU = try makeSharedBuffer(
                device: device,
                count: count,
                of: Float.self
            )
            let expectedAdd = try makeSharedBuffer(
                device: device,
                count: count,
                of: Float.self
            )
            let actualGELU = try makeSharedBuffer(
                device: device,
                count: count,
                of: Float.self
            )
            let actualAdd = try makeSharedBuffer(
                device: device,
                count: count,
                of: Float.self
            )
            try runOnGPU(queue: queue) { encoder in
                encode(
                    encoder,
                    pipeline: dense,
                    input: input,
                    weight: weight,
                    bias: bias,
                    output: stagedDense,
                    rows: rows,
                    outDim: outDim,
                    inDim: inDim,
                    rowsPerThreadgroup: 8
                )
                encoder.setComputePipelineState(gelu)
                encoder.setBuffer(stagedDense, offset: 0, index: 0)
                encoder.setBuffer(expectedGELU, offset: 0, index: 1)
                var count32 = UInt32(count)
                encoder.setBytes(&count32, length: 4, index: 2)
                encoder.dispatchThreads(
                    MTLSize(width: count, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(
                        width: 256,
                        height: 1,
                        depth: 1
                    )
                )
                encoder.setComputePipelineState(add)
                encoder.setBuffer(residual, offset: 0, index: 0)
                encoder.setBuffer(stagedDense, offset: 0, index: 1)
                encoder.setBuffer(expectedAdd, offset: 0, index: 2)
                encoder.setBytes(&count32, length: 4, index: 3)
                encoder.dispatchThreads(
                    MTLSize(width: count, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(
                        width: 256,
                        height: 1,
                        depth: 1
                    )
                )
                encodeEpilogue(
                    encoder,
                    pipeline: epilogue,
                    input: input,
                    weight: weight,
                    bias: bias,
                    residual: residual,
                    output: actualGELU,
                    rows: rows,
                    outDim: outDim,
                    inDim: inDim,
                    epilogue: 1
                )
                encodeEpilogue(
                    encoder,
                    pipeline: epilogue,
                    input: input,
                    weight: weight,
                    bias: bias,
                    residual: residual,
                    output: actualAdd,
                    rows: rows,
                    outDim: outDim,
                    inDim: inDim,
                    epilogue: 2
                )
            }
            let expectedGELUValues = read(expectedGELU, count: count)
            let expectedAddValues = read(expectedAdd, count: count)
            let actualGELUValues = read(actualGELU, count: count)
            let actualAddValues = read(actualAdd, count: count)
            for index in 0..<count {
                XCTAssertEqual(
                    actualGELUValues[index].bitPattern,
                    expectedGELUValues[index].bitPattern,
                    "gelu K=\(inDim) index=\(index)"
                )
                XCTAssertEqual(
                    actualAddValues[index].bitPattern,
                    expectedAddValues[index].bitPattern,
                    "add K=\(inDim) index=\(index)"
                )
            }
        }
    }

    func testTailWidthsMatchScalarOracle() throws {
        let dense = try pipeline("neural_primitives_f32.metal", "dense_bf16w_f32")
        let rows = 3
        let outDim = 7
        for inDim in [5, 54] {
            let inputValues = fixture(count: rows * inDim, seed: 41 + inDim)
            let weightValues = fixture(count: outDim * inDim, seed: 59 + inDim).map(bf16)
            let biasValues = fixture(count: outDim, seed: 71 + inDim).map(bf16)
            let input = try makeSharedBuffer(device: device, inputValues)
            let weight = try makeSharedBuffer(device: device, weightValues)
            let bias = try makeSharedBuffer(device: device, biasValues)
            let output = try makeSharedBuffer(
                device: device,
                count: rows * outDim,
                of: Float.self
            )
            try runOnGPU(queue: queue) { encoder in
                encode(
                    encoder,
                    pipeline: dense,
                    input: input,
                    weight: weight,
                    bias: bias,
                    output: output,
                    rows: rows,
                    outDim: outDim,
                    inDim: inDim
                )
            }
            let actual = read(output, count: rows * outDim)
            var maximumDifference: Float = 0
            for row in 0..<rows {
                for column in 0..<outDim {
                    var expected = float(biasValues[column])
                    for inner in 0..<inDim {
                        expected += inputValues[row * inDim + inner]
                            * float(weightValues[column * inDim + inner])
                    }
                    maximumDifference = max(
                        maximumDifference,
                        abs(actual[row * outDim + column] - expected)
                    )
                }
            }
            XCTAssertTrue(actual.allSatisfy(\.isFinite))
            XCTAssertGreaterThan(actual.map(abs).max() ?? 0, 1e-3)
            XCTAssertLessThan(maximumDifference, 2e-5, "K=\(inDim)")
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
}
