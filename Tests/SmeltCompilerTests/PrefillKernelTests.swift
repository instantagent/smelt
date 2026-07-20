// PrefillKernelTests — Kernel equivalence tests for Metal GPU prefill.
//
// Each test runs the decode kernel B times to produce a reference output,
// then runs the batched kernel once and compares. Tests skip when the
// requested pipeline cannot be compiled from the test metallib.
//
// Coverage:
// 1. fused_lut_matmul — dominates compute
// 2. rms_norm_1pw_batched — every layer, twice
// 3. embedding_gather_batched — one-shot, simple
// 4. attention_prefill — most complex

import CryptoKit
import Metal
import XCTest

@testable import SmeltCompiler
@testable import SmeltSchema

final class PrefillKernelTests: XCTestCase {

    private var device: MTLDevice!
    private var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        try? XCTSkipIf(device == nil, "No Metal device available")
        queue = device.makeCommandQueue()
    }

    // MARK: - Helpers

    func testDirectWeightPerHeadRmsNormMatchesMLX0311BitExactly() throws {
        let pipeline = try XCTUnwrap(makePipeline(
            shaderFile: "attention_fused.metal",
            functionName: "per_head_rms_norm"
        ))
        let headDim = 256
        let numHeads = 2
        let count = headDim * numHeads
        let input = (0..<count).map {
            Float16(Float(($0 * 37) % 257 - 128) / 64.0)
        }
        let weight = (0..<headDim).map {
            Float16(0.5 + Float(($0 * 13) % 97) / 128.0)
        }
        let inputBuffer = try makeSharedBuffer(device: device, input)
        let weightBuffer = try makeSharedBuffer(device: device, weight)
        let outputBuffer = try makeSharedBuffer(device: device, count: count, of: Float16.self)

        let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var headDimValue = UInt32(headDim)
        var epsilon: Float = 1e-6
        encoder.setBytes(&headDimValue, length: 4, index: 3)
        encoder.setBytes(&epsilon, length: 4, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: numHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: AttentionPlugin.directWeightNormThreadgroupWidth(headDim: headDim),
                height: 1,
                depth: 1
            )
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        let bytes = Data(bytes: outputBuffer.contents(), count: count * 2)
        let digest = SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
        XCTAssertEqual(
            digest,
            "a6f8621053c7f8ed2f457125c0701f22da30e3eba136ed78eaa58cab4fe375e5",
            "oracle: mlx.core.fast.rms_norm, MLX v0.31.1, fp16 [2, 256]"
        )
    }

    func testGatedAttentionSigmoidMultiplyMatchesMLX0311BitExactly() throws {
        let pipeline = try XCTUnwrap(makePipeline(
            shaderFile: "activations_precise.metal",
            functionName: "sigmoid_mul"
        ))
        let count = 6_144
        let attention = (0..<count).map {
            Float16(Float(($0 * 37) % 257 - 128) / 64.0)
        }
        let gate = (0..<count).map {
            Float16(Float(($0 * 19) % 193 - 96) / 32.0)
        }
        let attentionBuffer = try makeSharedBuffer(device: device, attention)
        let gateBuffer = try makeSharedBuffer(device: device, gate)
        let outputBuffer = try makeSharedBuffer(device: device, count: count, of: Float16.self)

        let commandBuffer = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(commandBuffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(attentionBuffer, offset: 0, index: 0)
        encoder.setBuffer(gateBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var countValue = UInt32(count)
        encoder.setBytes(&countValue, length: 4, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        XCTAssertNil(commandBuffer.error)

        let bytes = Data(bytes: outputBuffer.contents(), count: count * 2)
        let digest = SHA256.hash(data: bytes)
            .map { String(format: "%02x", $0) }
            .joined()
        XCTAssertEqual(
            digest,
            "c99d1a64145b1ce70864585932c79138f69f7cd231b19fe1d622b2f49883db26",
            "oracle: a * mlx.core.sigmoid(b), MLX v0.31.1, fp16 [6144]"
        )
    }

    private func loadShaderSource(_ filename: String) -> String? {
        loadMetalShaderSource(filename)
    }

    private func makePipeline(
        shaderFile: String,
        functionName: String,
        cols: UInt32? = nil,
        groupSize: UInt32? = nil
    ) -> MTLComputePipelineState? {
        guard let source = loadShaderSource(shaderFile),
              let lib = try? device.makeLibrary(source: source, options: nil)
        else { return nil }

        let fn: MTLFunction?
        if let cols, let groupSize {
            let constants = MTLFunctionConstantValues()
            var colsVal = cols
            var gsVal = groupSize
            constants.setConstantValue(&colsVal, type: .uint, index: 0)
            constants.setConstantValue(&gsVal, type: .uint, index: 1)
            fn = try? lib.makeFunction(name: functionName, constantValues: constants)
        } else {
            fn = lib.makeFunction(name: functionName)
        }

        guard let fn else { return nil }
        return try? device.makeComputePipelineState(function: fn)
    }

    private func makePipelines(
        shaderFile: String,
        functionNames: [String]
    ) -> [String: MTLComputePipelineState]? {
        guard let source = loadShaderSource(shaderFile),
              let library = try? device.makeLibrary(source: source, options: nil)
        else { return nil }

        var result: [String: MTLComputePipelineState] = [:]
        for name in functionNames {
            guard let function = library.makeFunction(name: name),
                  let pipeline = try? device.makeComputePipelineState(function: function)
            else { return nil }
            result[name] = pipeline
        }
        return result
    }

    private func runBatchedQMM(
        pipeline: MTLComputePipelineState,
        buffers: [Int: MTLBuffer],
        actualBatchIndex: Int,
        actualBatch: Int,
        rows: Int
    ) throws {
        let commandBuffer = queue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        var actualBatchValue = UInt32(actualBatch)
        encoder.setBytes(&actualBatchValue, length: 4, index: actualBatchIndex)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (rows + 31) / 32,
                height: (actualBatch + 15) / 16,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
    }

    private func firstHalfBitMismatch(
        _ lhs: MTLBuffer,
        _ rhs: MTLBuffer,
        count: Int
    ) -> Int? {
        let lhsBits = lhs.contents().bindMemory(to: UInt16.self, capacity: count)
        let rhsBits = rhs.contents().bindMemory(to: UInt16.self, capacity: count)
        for index in 0..<count where lhsBits[index] != rhsBits[index] {
            return index
        }
        return nil
    }

    private func makeAffineTestData(
        rows: Int,
        cols: Int,
        groupSize: Int,
        seed: Int = 0
    ) -> (
        weights: [UInt8],
        scales: [Float16],
        biases: [Float16],
        input: [Float16],
        residual: [Float16]
    ) {
        let weightBytes = rows * cols / 2
        let sbCount = rows * cols / groupSize

        var weights = [UInt8](repeating: 0, count: weightBytes)
        var scales = [Float16](repeating: 0, count: sbCount)
        var biases = [Float16](repeating: 0, count: sbCount)
        var input = [Float16](repeating: 0, count: cols)
        var residual = [Float16](repeating: 0, count: rows)

        let weightSeed = seed &* 131
        let scaleSeed = Float(seed &* 7919)
        let biasSeed = Float(seed &* 6869)
        let inputSeed = Float(seed &* 6151)
        let residualSeed = Float(seed &* 4817)

        for i in 0..<weightBytes {
            weights[i] = UInt8(truncatingIfNeeded: i &* 17 &+ 23 &+ weightSeed)
        }
        for i in 0..<sbCount {
            scales[i] = Float16(
                0.01 + 0.04 * (0.5 + 0.5 * sin((Float(i) + scaleSeed) * 0.0071))
            )
            biases[i] = Float16(cos((Float(i) + biasSeed) * 0.0113) * 0.03)
        }
        for i in 0..<cols {
            input[i] = Float16(sin((Float(i) + inputSeed) * 0.0137) * 0.5)
        }
        for i in 0..<rows {
            residual[i] = Float16(cos((Float(i) + residualSeed) * 0.0091) * 0.25)
        }

        return (weights, scales, biases, input, residual)
    }

    private func referenceAffineRow(
        weights: [UInt8],
        scales: [Float16],
        biases: [Float16],
        input: ArraySlice<Float16>,
        row: Int,
        cols: Int,
        groupSize: Int
    ) -> Float {
        let groupsPerRow = cols / groupSize
        let rowWeightBase = row * (cols / 2)
        let rowSBBase = row * groupsPerRow
        var acc: Float = 0

        for g in 0..<groupsPerRow {
            let scale = Float(scales[rowSBBase + g])
            let bias = Float(biases[rowSBBase + g])
            let colBase = g * groupSize
            var dot: Float = 0
            var xsum: Float = 0

            for i in 0..<groupSize {
                let col = colBase + i
                let byte = weights[rowWeightBase + (col / 2)]
                let nibble = (col & 1) == 0 ? (byte & 0x0F) : (byte >> 4)
                let x = Float(input[input.startIndex + col])
                dot += Float(nibble) * x
                xsum += x
            }

            acc += scale * dot + bias * xsum
        }

        return Float(Float16(acc))
    }

    private func referenceAffineMatvecCPU(
        weights: [UInt8],
        scales: [Float16],
        biases: [Float16],
        inputData: [Float16],
        rows: Int,
        cols: Int,
        groupSize: Int,
        batchSize: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: batchSize * rows)
        for b in 0..<batchSize {
            let inputSlice = inputData[(b * cols)..<((b + 1) * cols)]
            for r in 0..<rows {
                out[b * rows + r] = referenceAffineRow(
                    weights: weights,
                    scales: scales,
                    biases: biases,
                    input: inputSlice,
                    row: r,
                    cols: cols,
                    groupSize: groupSize
                )
            }
        }
        return out
    }

    private func referenceAffineRowFloatInput(
        weights: [UInt8],
        scales: [Float16],
        biases: [Float16],
        input: [Float],
        row: Int,
        cols: Int,
        groupSize: Int
    ) -> Float {
        let groupsPerRow = cols / groupSize
        let rowWeightBase = row * (cols / 2)
        let rowSBBase = row * groupsPerRow
        var acc: Float = 0

        for g in 0..<groupsPerRow {
            let scale = Float(scales[rowSBBase + g])
            let bias = Float(biases[rowSBBase + g])
            let colBase = g * groupSize
            var dot: Float = 0
            var xsum: Float = 0

            for i in 0..<groupSize {
                let col = colBase + i
                let byte = weights[rowWeightBase + (col / 2)]
                let nibble = (col & 1) == 0 ? (byte & 0x0F) : (byte >> 4)
                let x = input[col]
                dot += Float(nibble) * x
                xsum += x
            }

            acc += scale * dot + bias * xsum
        }

        return Float(Float16(acc))
    }

    private func referenceAffineMatvecCPUFloatInput(
        weights: [UInt8],
        scales: [Float16],
        biases: [Float16],
        inputData: [Float],
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: rows)
        for r in 0..<rows {
            out[r] = referenceAffineRowFloatInput(
                weights: weights,
                scales: scales,
                biases: biases,
                input: inputData,
                row: r,
                cols: cols,
                groupSize: groupSize
            )
        }
        return out
    }

    private func referenceFusedAffineGateUpSwigluCPU(
        gateWeights: [UInt8],
        gateScales: [Float16],
        gateBiases: [Float16],
        upWeights: [UInt8],
        upScales: [Float16],
        upBiases: [Float16],
        inputData: [Float16],
        rows: Int,
        cols: Int,
        groupSize: Int,
        batchSize: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: batchSize * rows)
        for b in 0..<batchSize {
            let inputSlice = inputData[(b * cols)..<((b + 1) * cols)]
            for r in 0..<rows {
                let gate = referenceAffineRow(
                    weights: gateWeights,
                    scales: gateScales,
                    biases: gateBiases,
                    input: inputSlice,
                    row: r,
                    cols: cols,
                    groupSize: groupSize
                )
                let up = referenceAffineRow(
                    weights: upWeights,
                    scales: upScales,
                    biases: upBiases,
                    input: inputSlice,
                    row: r,
                    cols: cols,
                    groupSize: groupSize
                )
                let silu = gate / (1.0 + Foundation.exp(-gate))
                out[b * rows + r] = Float(Float16(silu * up))
            }
        }
        return out
    }

    private func referenceFusedAffineGateUpGeGLUCPU(
        gateWeights: [UInt8],
        gateScales: [Float16],
        gateBiases: [Float16],
        upWeights: [UInt8],
        upScales: [Float16],
        upBiases: [Float16],
        inputData: [Float16],
        rows: Int,
        cols: Int,
        groupSize: Int,
        batchSize: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: batchSize * rows)
        for b in 0..<batchSize {
            let inputSlice = inputData[(b * cols)..<((b + 1) * cols)]
            for r in 0..<rows {
                let gate = referenceAffineRow(
                    weights: gateWeights,
                    scales: gateScales,
                    biases: gateBiases,
                    input: inputSlice,
                    row: r,
                    cols: cols,
                    groupSize: groupSize
                )
                let up = referenceAffineRow(
                    weights: upWeights,
                    scales: upScales,
                    biases: upBiases,
                    input: inputSlice,
                    row: r,
                    cols: cols,
                    groupSize: groupSize
                )
                let gate3 = gate * gate * gate
                var inner = 0.7978845608 * (gate + 0.044715 * gate3)
                inner = min(20.0, max(-20.0, inner))
                let gelu = 0.5 * gate * (1.0 + Foundation.tanh(inner))
                let product = min(65504.0, max(-65504.0, gelu * up))
                out[b * rows + r] = Float(Float16(product))
            }
        }
        return out
    }

    private func referenceFusedAffineGateUpGeGLUCPUFloatInput(
        gateWeights: [UInt8],
        gateScales: [Float16],
        gateBiases: [Float16],
        upWeights: [UInt8],
        upScales: [Float16],
        upBiases: [Float16],
        inputData: [Float],
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: rows)
        for r in 0..<rows {
            let gate = referenceAffineRowFloatInput(
                weights: gateWeights,
                scales: gateScales,
                biases: gateBiases,
                input: inputData,
                row: r,
                cols: cols,
                groupSize: groupSize
            )
            let up = referenceAffineRowFloatInput(
                weights: upWeights,
                scales: upScales,
                biases: upBiases,
                input: inputData,
                row: r,
                cols: cols,
                groupSize: groupSize
            )
            let gate3 = gate * gate * gate
            var inner = 0.7978845608 * (gate + 0.044715 * gate3)
            inner = min(20.0, max(-20.0, inner))
            let gelu = 0.5 * gate * (1.0 + Foundation.tanh(inner))
            let product = min(65504.0, max(-65504.0, gelu * up))
            out[r] = Float(Float16(product))
        }
        return out
    }

    private func referenceRoundedAndUnroundedNormInput(
        input: [Float16],
        weight: [Float16]
    ) -> (rounded: [Float16], unrounded: [Float]) {
        let dim = input.count
        var sumSq: Float = 0
        for i in 0..<dim {
            let x = Float(input[i])
            sumSq += x * x
        }
        let rs = 1.0 / sqrt(sumSq / Float(dim) + 1e-6)

        var rounded = [Float16](repeating: 0, count: dim)
        var unrounded = [Float](repeating: 0, count: dim)
        for i in 0..<dim {
            let normalized = Float(input[i]) * rs * (1.0 + Float(weight[i]))
            rounded[i] = Float16(normalized)
            unrounded[i] = normalized
        }
        return (rounded, unrounded)
    }

    private func referenceFusedDualAffineMatvecCPU(
        w1Weights: [UInt8],
        w1Scales: [Float16],
        w1Biases: [Float16],
        w2Weights: [UInt8],
        w2Scales: [Float16],
        w2Biases: [Float16],
        inputData: [Float16],
        rows: Int,
        cols: Int,
        groupSize: Int,
        batchSize: Int
    ) -> (out1: [Float], out2: [Float]) {
        var out1 = [Float](repeating: 0, count: batchSize * rows)
        var out2 = [Float](repeating: 0, count: batchSize * rows)
        for b in 0..<batchSize {
            let inputSlice = inputData[(b * cols)..<((b + 1) * cols)]
            for r in 0..<rows {
                out1[b * rows + r] = referenceAffineRow(
                    weights: w1Weights,
                    scales: w1Scales,
                    biases: w1Biases,
                    input: inputSlice,
                    row: r,
                    cols: cols,
                    groupSize: groupSize
                )
                out2[b * rows + r] = referenceAffineRow(
                    weights: w2Weights,
                    scales: w2Scales,
                    biases: w2Biases,
                    input: inputSlice,
                    row: r,
                    cols: cols,
                    groupSize: groupSize
                )
            }
        }
        return (out1, out2)
    }

    private func referenceRmsNormGatedCPU(
        input: [Float16],
        gate: [Float16],
        weight: [Float16],
        headDim: Int,
        batchHeads: Int,
        eps: Float = 1e-6
    ) -> [Float] {
        var out = [Float](repeating: 0, count: batchHeads * headDim)
        for bh in 0..<batchHeads {
            let base = bh * headDim
            var sumSq: Float = 0
            for i in 0..<headDim {
                let x = Float(input[base + i])
                sumSq += x * x
            }
            let rs = 1.0 / sqrt(sumSq / Float(headDim) + eps)
            for i in 0..<headDim {
                let x = Float(input[base + i])
                let g = Float(gate[base + i])
                let w = Float(weight[i])
                let silu = g / (1.0 + Foundation.exp(-g))
                out[base + i] = Float(Float16(w * (x * rs) * silu))
            }
        }
        return out
    }

    // MARK: - LUT Matmul (Priority 1)

    /// Core equivalence: fused_lut_matmul output == B iterations of fused_lut_matvec.
    func testLutMatmul_MatchesIteratedMatvec() throws {
        let rows = 32
        let cols = 64
        let groupSize = 16
        let batchSize = 4

        let matvecPipeline = makePipeline(
            shaderFile: "lut_matvec.metal", functionName: "fused_lut_matvec",
            cols: UInt32(cols), groupSize: UInt32(groupSize)
        )
        try XCTSkipIf(matvecPipeline == nil, "Could not compile decode matvec shader")

        let matmulPipeline = makePipeline(
            shaderFile: "prefill_matmul.metal", functionName: "fused_lut_matmul"
        )
        try XCTSkipIf(matmulPipeline == nil, "Could not compile fused_lut_matmul pipeline")

        // Quantize weights
        let (indicesBuf, lutBuf, _) = try quantizeTestWeights(
            rows: rows, cols: cols, groupSize: groupSize
        )

        // Generate B input vectors
        var inputData = [Float16](repeating: 0, count: batchSize * cols)
        for b in 0..<batchSize {
            for c in 0..<cols {
                inputData[b * cols + c] = Float16(
                    cos(Float(b * cols + c) * 0.07) * 0.5
                )
            }
        }

        // Reference: run fused_lut_matvec B times
        let reference = try runIteratedMatvec(
            pipeline: matvecPipeline!, indices: indicesBuf, lut: lutBuf,
            inputData: inputData, rows: rows, cols: cols,
            groupSize: groupSize, batchSize: batchSize
        )

        // Test: run fused_lut_matmul once
        let inputBuf = device.makeBuffer(
            bytes: inputData, length: inputData.count * 2,
            options: .storageModeShared
        )!
        let outputBuf = device.makeBuffer(
            length: batchSize * rows * 2, options: .storageModeShared
        )!
        memset(outputBuf.contents(), 0, batchSize * rows * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(matmulPipeline!)
        enc.setBuffer(indicesBuf, offset: 0, index: 0)
        enc.setBuffer(lutBuf, offset: 0, index: 1)
        enc.setBuffer(inputBuf, offset: 0, index: 2)
        enc.setBuffer(outputBuf, offset: 0, index: 3)
        var colsVal = UInt32(cols)
        enc.setBytes(&colsVal, length: 4, index: 4)
        var gsVal = UInt32(groupSize)
        enc.setBytes(&gsVal, length: 4, index: 5)
        var rowsVal = UInt32(rows)
        enc.setBytes(&rowsVal, length: 4, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: batchSize, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error, "Metal error: \(cmdBuf.error?.localizedDescription ?? "")")

        // Compare
        let outPtr = outputBuf.contents().bindMemory(
            to: Float16.self, capacity: batchSize * rows
        )
        var maxDiff: Float = 0
        for i in 0..<(batchSize * rows) {
            let diff = abs(Float(outPtr[i]) - reference[i])
            if diff > maxDiff { maxDiff = diff }
        }

        fputs(
            "  lut_matmul vs iterated matvec [\(rows),\(cols)] B=\(batchSize):"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.02, "Batched matmul diverges from iterated matvec")
    }

    /// Larger dimensions to stress the kernel's tiling and reduction.
    func testLutMatmul_LargerDims() throws {
        let rows = 128
        let cols = 256
        let groupSize = 16
        let batchSize = 8

        let matvecPipeline = makePipeline(
            shaderFile: "lut_matvec.metal", functionName: "fused_lut_matvec",
            cols: UInt32(cols), groupSize: UInt32(groupSize)
        )
        try XCTSkipIf(matvecPipeline == nil, "Could not compile decode matvec shader")

        let matmulPipeline = makePipeline(
            shaderFile: "prefill_matmul.metal", functionName: "fused_lut_matmul"
        )
        try XCTSkipIf(matmulPipeline == nil, "Could not compile fused_lut_matmul pipeline")

        let (indicesBuf, lutBuf, _) = try quantizeTestWeights(
            rows: rows, cols: cols, groupSize: groupSize
        )

        var inputData = [Float16](repeating: 0, count: batchSize * cols)
        for i in 0..<inputData.count {
            inputData[i] = Float16(sin(Float(i) * 0.03) * 0.3)
        }

        let reference = try runIteratedMatvec(
            pipeline: matvecPipeline!, indices: indicesBuf, lut: lutBuf,
            inputData: inputData, rows: rows, cols: cols,
            groupSize: groupSize, batchSize: batchSize
        )

        let inputBuf = device.makeBuffer(
            bytes: inputData, length: inputData.count * 2,
            options: .storageModeShared
        )!
        let outputBuf = device.makeBuffer(
            length: batchSize * rows * 2, options: .storageModeShared
        )!
        memset(outputBuf.contents(), 0, batchSize * rows * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(matmulPipeline!)
        enc.setBuffer(indicesBuf, offset: 0, index: 0)
        enc.setBuffer(lutBuf, offset: 0, index: 1)
        enc.setBuffer(inputBuf, offset: 0, index: 2)
        enc.setBuffer(outputBuf, offset: 0, index: 3)
        var colsVal = UInt32(cols)
        enc.setBytes(&colsVal, length: 4, index: 4)
        var gsVal = UInt32(groupSize)
        enc.setBytes(&gsVal, length: 4, index: 5)
        var rowsVal = UInt32(rows)
        enc.setBytes(&rowsVal, length: 4, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: batchSize, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = outputBuf.contents().bindMemory(
            to: Float16.self, capacity: batchSize * rows
        )
        var maxDiff: Float = 0
        for i in 0..<(batchSize * rows) {
            let diff = abs(Float(outPtr[i]) - reference[i])
            if diff > maxDiff { maxDiff = diff }
        }

        fputs(
            "  lut_matmul larger [\(rows),\(cols)] B=\(batchSize):"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.02)
    }

    /// B=1 must exactly match a single matvec (same reduction path).
    func testLutMatmul_BatchSize1() throws {
        let rows = 32
        let cols = 64
        let groupSize = 16

        let matvecPipeline = makePipeline(
            shaderFile: "lut_matvec.metal", functionName: "fused_lut_matvec",
            cols: UInt32(cols), groupSize: UInt32(groupSize)
        )
        try XCTSkipIf(matvecPipeline == nil, "Could not compile decode matvec shader")

        let matmulPipeline = makePipeline(
            shaderFile: "prefill_matmul.metal", functionName: "fused_lut_matmul"
        )
        try XCTSkipIf(matmulPipeline == nil, "Could not compile fused_lut_matmul pipeline")

        let (indicesBuf, lutBuf, _) = try quantizeTestWeights(
            rows: rows, cols: cols, groupSize: groupSize
        )

        var inputData = [Float16](repeating: 0, count: cols)
        for c in 0..<cols {
            inputData[c] = Float16(cos(Float(c) * 0.1) * 0.5)
        }

        // Reference: single matvec
        let reference = try runIteratedMatvec(
            pipeline: matvecPipeline!, indices: indicesBuf, lut: lutBuf,
            inputData: inputData, rows: rows, cols: cols,
            groupSize: groupSize, batchSize: 1
        )

        // Batched with B=1
        let inputBuf = device.makeBuffer(
            bytes: inputData, length: cols * 2, options: .storageModeShared
        )!
        let outputBuf = device.makeBuffer(
            length: rows * 2, options: .storageModeShared
        )!
        memset(outputBuf.contents(), 0, rows * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(matmulPipeline!)
        enc.setBuffer(indicesBuf, offset: 0, index: 0)
        enc.setBuffer(lutBuf, offset: 0, index: 1)
        enc.setBuffer(inputBuf, offset: 0, index: 2)
        enc.setBuffer(outputBuf, offset: 0, index: 3)
        var colsVal = UInt32(cols)
        enc.setBytes(&colsVal, length: 4, index: 4)
        var gsVal = UInt32(groupSize)
        enc.setBytes(&gsVal, length: 4, index: 5)
        var rowsVal = UInt32(rows)
        enc.setBytes(&rowsVal, length: 4, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = outputBuf.contents().bindMemory(to: Float16.self, capacity: rows)
        var maxDiff: Float = 0
        for i in 0..<rows {
            let diff = abs(Float(outPtr[i]) - reference[i])
            if diff > maxDiff { maxDiff = diff }
        }

        fputs(
            "  lut_matmul B=1 vs single matvec: max diff = "
                + "\(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.001, "B=1 matmul should match single matvec tightly")
    }

    // MARK: - RMS Norm Batched (Priority 2)

    /// Batched RMS norm matches B individual norms.
    func testRmsNormBatched_MatchesIteratedNorm() throws {
        let normPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw"
        )
        try XCTSkipIf(normPipeline == nil, "Could not compile decode norm shader")

        let batchedPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw_batched"
        )
        try XCTSkipIf(batchedPipeline == nil, "Could not compile rms_norm_1pw_batched pipeline")

        let dim = 128
        let batchSize = 8
        let eps: Float = 1e-6

        // Generate input [B, dim] and weight [dim]
        var inputData = [Float16](repeating: 0, count: batchSize * dim)
        for i in 0..<inputData.count {
            inputData[i] = Float16(sin(Float(i) * 0.1) * 2.0)
        }
        var weightData = [Float16](repeating: 0, count: dim)
        for i in 0..<dim {
            weightData[i] = Float16(cos(Float(i) * 0.05) * 0.01)
        }

        let weightBuf = device.makeBuffer(
            bytes: weightData, length: dim * 2, options: .storageModeShared
        )!

        // Reference: run rms_norm_1pw B times
        var reference = [Float](repeating: 0, count: batchSize * dim)
        let singleInputBuf = device.makeBuffer(
            length: dim * 2, options: .storageModeShared
        )!
        let singleOutputBuf = device.makeBuffer(
            length: dim * 2, options: .storageModeShared
        )!

        for b in 0..<batchSize {
            let slice = Array(inputData[b * dim..<(b + 1) * dim])
            memcpy(singleInputBuf.contents(), slice, dim * 2)
            memset(singleOutputBuf.contents(), 0, dim * 2)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(normPipeline!)
            enc.setBuffer(singleInputBuf, offset: 0, index: 0)
            enc.setBuffer(weightBuf, offset: 0, index: 1)
            enc.setBuffer(singleOutputBuf, offset: 0, index: 2)
            var dimVal = UInt32(dim)
            enc.setBytes(&dimVal, length: 4, index: 3)
            var epsVal = eps
            enc.setBytes(&epsVal, length: 4, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(dim, 1024), height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let ptr = singleOutputBuf.contents().bindMemory(
                to: Float16.self, capacity: dim
            )
            for d in 0..<dim {
                reference[b * dim + d] = Float(ptr[d])
            }
        }

        // Test: single batched dispatch
        let batchInputBuf = device.makeBuffer(
            bytes: inputData, length: inputData.count * 2,
            options: .storageModeShared
        )!
        let batchOutputBuf = device.makeBuffer(
            length: batchSize * dim * 2, options: .storageModeShared
        )!
        memset(batchOutputBuf.contents(), 0, batchSize * dim * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchedPipeline!)
        enc.setBuffer(batchInputBuf, offset: 0, index: 0)
        enc.setBuffer(weightBuf, offset: 0, index: 1)
        enc.setBuffer(batchOutputBuf, offset: 0, index: 2)
        var dimVal = UInt32(dim)
        enc.setBytes(&dimVal, length: 4, index: 3)
        var epsVal = eps
        enc.setBytes(&epsVal, length: 4, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: batchSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(dim, 1024), height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = batchOutputBuf.contents().bindMemory(
            to: Float16.self, capacity: batchSize * dim
        )
        var maxDiff: Float = 0
        for i in 0..<(batchSize * dim) {
            let diff = abs(Float(outPtr[i]) - reference[i])
            if diff > maxDiff { maxDiff = diff }
        }

        fputs(
            "  rms_norm batched vs iterated dim=\(dim) B=\(batchSize):"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 1e-3, "Batched norm diverges from iterated norm")
    }

    /// Production hidden dimension exercises multiple contiguous reduction tiles.
    func testRmsNormBatched_RealDim() throws {
        let normPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw"
        )
        try XCTSkipIf(normPipeline == nil, "Could not compile decode norm shader")

        let batchedPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw_batched"
        )
        try XCTSkipIf(batchedPipeline == nil, "Could not compile rms_norm_1pw_batched pipeline")

        let dim = 5120
        let batchSize = 3
        let eps: Float = 1e-6

        var inputData = [Float16](repeating: 0, count: batchSize * dim)
        for i in 0..<inputData.count {
            inputData[i] = Float16(sin(Float(i) * 0.003) * 1.5)
        }
        var weightData = [Float16](repeating: 0, count: dim)
        for i in 0..<dim {
            weightData[i] = Float16(cos(Float(i) * 0.01) * 0.02)
        }

        let weightBuf = device.makeBuffer(
            bytes: weightData, length: dim * 2, options: .storageModeShared
        )!

        // Reference: iterated
        var reference = [Float16](repeating: 0, count: batchSize * dim)
        let singleInputBuf = device.makeBuffer(
            length: dim * 2, options: .storageModeShared
        )!
        let singleOutputBuf = device.makeBuffer(
            length: dim * 2, options: .storageModeShared
        )!

        for b in 0..<batchSize {
            let slice = Array(inputData[b * dim..<(b + 1) * dim])
            memcpy(singleInputBuf.contents(), slice, dim * 2)
            memset(singleOutputBuf.contents(), 0, dim * 2)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(normPipeline!)
            enc.setBuffer(singleInputBuf, offset: 0, index: 0)
            enc.setBuffer(weightBuf, offset: 0, index: 1)
            enc.setBuffer(singleOutputBuf, offset: 0, index: 2)
            var dimVal = UInt32(dim)
            enc.setBytes(&dimVal, length: 4, index: 3)
            var epsVal = eps
            enc.setBytes(&epsVal, length: 4, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            let ptr = singleOutputBuf.contents().bindMemory(
                to: Float16.self, capacity: dim
            )
            for d in 0..<dim {
                reference[b * dim + d] = ptr[d]
            }
        }

        // Test: batched
        let batchInputBuf = device.makeBuffer(
            bytes: inputData, length: inputData.count * 2,
            options: .storageModeShared
        )!
        let batchOutputBuf = device.makeBuffer(
            length: batchSize * dim * 2, options: .storageModeShared
        )!
        memset(batchOutputBuf.contents(), 0, batchSize * dim * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchedPipeline!)
        enc.setBuffer(batchInputBuf, offset: 0, index: 0)
        enc.setBuffer(weightBuf, offset: 0, index: 1)
        enc.setBuffer(batchOutputBuf, offset: 0, index: 2)
        var dimVal = UInt32(dim)
        enc.setBytes(&dimVal, length: 4, index: 3)
        var epsVal = eps
        enc.setBytes(&epsVal, length: 4, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: batchSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = batchOutputBuf.contents().bindMemory(
            to: Float16.self, capacity: batchSize * dim
        )
        var maxDiff: Float = 0
        var mismatches = 0
        for i in 0..<(batchSize * dim) {
            let diff = abs(Float(outPtr[i]) - Float(reference[i]))
            if diff > maxDiff { maxDiff = diff }
            if outPtr[i].bitPattern != reference[i].bitPattern { mismatches += 1 }
        }

        fputs(
            "  rms_norm batched real dim=\(dim) B=\(batchSize):"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertEqual(mismatches, 0, "Batched norm must preserve single-row arithmetic")
    }

    func testPerHeadRmsNormBatched_MatchesIteratedNorm() throws {
        let singlePipeline = makePipeline(
            shaderFile: "attention_fused.metal", functionName: "per_head_rms_norm"
        )
        try XCTSkipIf(singlePipeline == nil, "Could not compile per_head_rms_norm")

        let batchedPipeline = makePipeline(
            shaderFile: "attention_fused.metal", functionName: "per_head_rms_norm_batched"
        )
        try XCTSkipIf(batchedPipeline == nil, "Could not compile per_head_rms_norm_batched")

        let numHeads = 4
        let headDim = 64
        let batchSize = 5
        let elemsPerBatch = numHeads * headDim
        let eps: Float = 1e-6

        var input = [Float16](repeating: 0, count: batchSize * elemsPerBatch)
        var weight = [Float16](repeating: 0, count: headDim)
        for i in 0..<input.count {
            input[i] = Float16(sin(Float(i) * 0.017) * 0.7)
        }
        for i in 0..<weight.count {
            weight[i] = Float16(0.9 + 0.1 * cos(Float(i) * 0.031))
        }

        let weightBuf = device.makeBuffer(
            bytes: weight,
            length: weight.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!
        let singleInputBuf = device.makeBuffer(
            length: elemsPerBatch * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!
        let singleOutputBuf = device.makeBuffer(
            length: elemsPerBatch * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!

        var reference = [Float](repeating: 0, count: batchSize * elemsPerBatch)
        for b in 0..<batchSize {
            let slice = Array(input[(b * elemsPerBatch)..<((b + 1) * elemsPerBatch)])
            memcpy(singleInputBuf.contents(), slice, elemsPerBatch * MemoryLayout<Float16>.stride)
            memset(singleOutputBuf.contents(), 0, elemsPerBatch * MemoryLayout<Float16>.stride)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(singlePipeline!)
            enc.setBuffer(singleInputBuf, offset: 0, index: 0)
            enc.setBuffer(weightBuf, offset: 0, index: 1)
            enc.setBuffer(singleOutputBuf, offset: 0, index: 2)
            var headDimVal = UInt32(headDim)
            enc.setBytes(&headDimVal, length: 4, index: 3)
            var epsVal = eps
            enc.setBytes(&epsVal, length: 4, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: numHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(headDim, 256), height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let ptr = singleOutputBuf.contents().bindMemory(
                to: Float16.self, capacity: elemsPerBatch
            )
            for i in 0..<elemsPerBatch {
                reference[b * elemsPerBatch + i] = Float(ptr[i])
            }
        }

        let batchInputBuf = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!
        let batchOutputBuf = device.makeBuffer(
            length: input.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!
        memset(batchOutputBuf.contents(), 0, input.count * MemoryLayout<Float16>.stride)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchedPipeline!)
        enc.setBuffer(batchInputBuf, offset: 0, index: 0)
        enc.setBuffer(weightBuf, offset: 0, index: 1)
        enc.setBuffer(batchOutputBuf, offset: 0, index: 2)
        var numHeadsVal = UInt32(numHeads)
        enc.setBytes(&numHeadsVal, length: 4, index: 3)
        var headDimVal = UInt32(headDim)
        enc.setBytes(&headDimVal, length: 4, index: 4)
        var epsVal = eps
        enc.setBytes(&epsVal, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: numHeads, height: batchSize, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(headDim, 256), height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = batchOutputBuf.contents().bindMemory(
            to: Float16.self, capacity: input.count
        )
        var maxDiff: Float = 0
        for i in 0..<input.count {
            maxDiff = max(maxDiff, abs(Float(outPtr[i]) - reference[i]))
        }

        fputs(
            "  per_head_rms_norm_batched vs iterated B=\(batchSize):"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 1e-3)
    }

    func testPerHeadRmsNormNoScaleBatched_MatchesIteratedNorm() throws {
        let singlePipeline = makePipeline(
            shaderFile: "attention_fused.metal", functionName: "per_head_rms_norm_noscale"
        )
        try XCTSkipIf(singlePipeline == nil, "Could not compile per_head_rms_norm_noscale")

        let batchedPipeline = makePipeline(
            shaderFile: "attention_fused.metal", functionName: "per_head_rms_norm_noscale_batched"
        )
        try XCTSkipIf(batchedPipeline == nil, "Could not compile per_head_rms_norm_noscale_batched")

        let numHeads = 2
        let headDim = 64
        let batchSize = 6
        let elemsPerBatch = numHeads * headDim
        let eps: Float = 1e-6

        var input = [Float16](repeating: 0, count: batchSize * elemsPerBatch)
        for i in 0..<input.count {
            input[i] = Float16(cos(Float(i) * 0.023) * 0.8)
        }

        let singleInputBuf = device.makeBuffer(
            length: elemsPerBatch * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!

        var reference = [Float](repeating: 0, count: batchSize * elemsPerBatch)
        for b in 0..<batchSize {
            let slice = Array(input[(b * elemsPerBatch)..<((b + 1) * elemsPerBatch)])
            memcpy(singleInputBuf.contents(), slice, elemsPerBatch * MemoryLayout<Float16>.stride)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(singlePipeline!)
            enc.setBuffer(singleInputBuf, offset: 0, index: 0)
            var headDimVal = UInt32(headDim)
            enc.setBytes(&headDimVal, length: 4, index: 1)
            var epsVal = eps
            enc.setBytes(&epsVal, length: 4, index: 2)
            enc.dispatchThreadgroups(
                MTLSize(width: numHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(headDim, 256), height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let ptr = singleInputBuf.contents().bindMemory(
                to: Float16.self, capacity: elemsPerBatch
            )
            for i in 0..<elemsPerBatch {
                reference[b * elemsPerBatch + i] = Float(ptr[i])
            }
        }

        let batchInputBuf = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchedPipeline!)
        enc.setBuffer(batchInputBuf, offset: 0, index: 0)
        var numHeadsVal = UInt32(numHeads)
        enc.setBytes(&numHeadsVal, length: 4, index: 1)
        var headDimVal = UInt32(headDim)
        enc.setBytes(&headDimVal, length: 4, index: 2)
        var epsVal = eps
        enc.setBytes(&epsVal, length: 4, index: 3)
        enc.dispatchThreadgroups(
            MTLSize(width: numHeads, height: batchSize, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(headDim, 256), height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = batchInputBuf.contents().bindMemory(
            to: Float16.self, capacity: input.count
        )
        var maxDiff: Float = 0
        for i in 0..<input.count {
            maxDiff = max(maxDiff, abs(Float(outPtr[i]) - reference[i]))
        }

        fputs(
            "  per_head_rms_norm_noscale_batched vs iterated B=\(batchSize):"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 1e-3)
    }

    func testPerHeadRmsNormThreadgroupWidthDiagnostics() throws {
        let weightedPipeline = makePipeline(
            shaderFile: "attention_fused.metal", functionName: "per_head_rms_norm"
        )
        try XCTSkipIf(weightedPipeline == nil, "Could not compile per_head_rms_norm")

        let noScalePipeline = makePipeline(
            shaderFile: "attention_fused.metal", functionName: "per_head_rms_norm_noscale"
        )
        try XCTSkipIf(noScalePipeline == nil, "Could not compile per_head_rms_norm_noscale")

        let headDims = [256, 512]
        let numHeads = 2
        let eps: Float = 1e-6

        for headDim in headDims {
            let elemCount = numHeads * headDim
            var input = [Float16](repeating: 0, count: elemCount)
            var weight = [Float16](repeating: 0, count: headDim)
            for i in 0..<elemCount {
                input[i] = Float16(sin(Float(i) * 0.0191) * 1.7 + cos(Float(i) * 0.0073) * 0.6)
            }
            for i in 0..<headDim {
                weight[i] = Float16(0.15 * cos(Float(i) * 0.0217))
            }

            let weightBuf = device.makeBuffer(
                bytes: weight,
                length: headDim * MemoryLayout<Float16>.stride,
                options: .storageModeShared
            )!

            func maxWeightedDiff(inPlace: Bool) throws -> Float {
                let refBuf = device.makeBuffer(
                    bytes: input,
                    length: elemCount * MemoryLayout<Float16>.stride,
                    options: .storageModeShared
                )!
                let altBuf = device.makeBuffer(
                    bytes: input,
                    length: elemCount * MemoryLayout<Float16>.stride,
                    options: .storageModeShared
                )!
                let refOut = inPlace ? refBuf : device.makeBuffer(
                    length: elemCount * MemoryLayout<Float16>.stride,
                    options: .storageModeShared
                )!
                let altOut = inPlace ? altBuf : device.makeBuffer(
                    length: elemCount * MemoryLayout<Float16>.stride,
                    options: .storageModeShared
                )!
                if !inPlace {
                    memset(refOut.contents(), 0, elemCount * MemoryLayout<Float16>.stride)
                    memset(altOut.contents(), 0, elemCount * MemoryLayout<Float16>.stride)
                }

                var headDimVal = UInt32(headDim)
                var epsVal = eps

                let cmdBuf1 = queue.makeCommandBuffer()!
                let enc1 = cmdBuf1.makeComputeCommandEncoder()!
                enc1.setComputePipelineState(weightedPipeline!)
                enc1.setBuffer(refBuf, offset: 0, index: 0)
                enc1.setBuffer(weightBuf, offset: 0, index: 1)
                enc1.setBuffer(refOut, offset: 0, index: 2)
                enc1.setBytes(&headDimVal, length: 4, index: 3)
                enc1.setBytes(&epsVal, length: 4, index: 4)
                enc1.dispatchThreadgroups(
                    MTLSize(width: numHeads, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(headDim, 256), height: 1, depth: 1)
                )
                enc1.endEncoding()
                cmdBuf1.commit()
                cmdBuf1.waitUntilCompleted()
                XCTAssertNil(cmdBuf1.error)

                let cmdBuf2 = queue.makeCommandBuffer()!
                let enc2 = cmdBuf2.makeComputeCommandEncoder()!
                enc2.setComputePipelineState(weightedPipeline!)
                enc2.setBuffer(altBuf, offset: 0, index: 0)
                enc2.setBuffer(weightBuf, offset: 0, index: 1)
                enc2.setBuffer(altOut, offset: 0, index: 2)
                enc2.setBytes(&headDimVal, length: 4, index: 3)
                enc2.setBytes(&epsVal, length: 4, index: 4)
                enc2.dispatchThreadgroups(
                    MTLSize(width: numHeads, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
                )
                enc2.endEncoding()
                cmdBuf2.commit()
                cmdBuf2.waitUntilCompleted()
                XCTAssertNil(cmdBuf2.error)

                let refPtr = refOut.contents().bindMemory(to: Float16.self, capacity: elemCount)
                let altPtr = altOut.contents().bindMemory(to: Float16.self, capacity: elemCount)
                var maxDiff: Float = 0
                for i in 0..<elemCount {
                    let diff = abs(Float(refPtr[i]) - Float(altPtr[i]))
                    if diff > maxDiff { maxDiff = diff }
                }
                return maxDiff
            }

            func maxNoScaleDiff(inPlace: Bool) throws -> Float {
                let refBuf = device.makeBuffer(
                    bytes: input,
                    length: elemCount * MemoryLayout<Float16>.stride,
                    options: .storageModeShared
                )!
                let altBuf = device.makeBuffer(
                    bytes: input,
                    length: elemCount * MemoryLayout<Float16>.stride,
                    options: .storageModeShared
                )!

                var headDimVal = UInt32(headDim)
                var epsVal = eps

                let cmdBuf1 = queue.makeCommandBuffer()!
                let enc1 = cmdBuf1.makeComputeCommandEncoder()!
                enc1.setComputePipelineState(noScalePipeline!)
                enc1.setBuffer(refBuf, offset: 0, index: 0)
                enc1.setBytes(&headDimVal, length: 4, index: 1)
                enc1.setBytes(&epsVal, length: 4, index: 2)
                enc1.dispatchThreadgroups(
                    MTLSize(width: numHeads, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(headDim, 256), height: 1, depth: 1)
                )
                enc1.endEncoding()
                cmdBuf1.commit()
                cmdBuf1.waitUntilCompleted()
                XCTAssertNil(cmdBuf1.error)

                let cmdBuf2 = queue.makeCommandBuffer()!
                let enc2 = cmdBuf2.makeComputeCommandEncoder()!
                enc2.setComputePipelineState(noScalePipeline!)
                enc2.setBuffer(altBuf, offset: 0, index: 0)
                enc2.setBytes(&headDimVal, length: 4, index: 1)
                enc2.setBytes(&epsVal, length: 4, index: 2)
                enc2.dispatchThreadgroups(
                    MTLSize(width: numHeads, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
                )
                enc2.endEncoding()
                cmdBuf2.commit()
                cmdBuf2.waitUntilCompleted()
                XCTAssertNil(cmdBuf2.error)

                let refPtr = refBuf.contents().bindMemory(to: Float16.self, capacity: elemCount)
                let altPtr = altBuf.contents().bindMemory(to: Float16.self, capacity: elemCount)
                var maxDiff: Float = 0
                for i in 0..<elemCount {
                    let diff = abs(Float(refPtr[i]) - Float(altPtr[i]))
                    if diff > maxDiff { maxDiff = diff }
                }
                return maxDiff
            }

            let weightedOutOfPlace = try maxWeightedDiff(inPlace: false)
            let weightedInPlace = try maxWeightedDiff(inPlace: true)
            let noScaleInPlace = try maxNoScaleDiff(inPlace: true)

            fputs(
                "  per_head_rms_norm diagnostics headDim=\(headDim):"
                    + " weighted oop = \(String(format: "%.6f", weightedOutOfPlace))"
                    + ", weighted in-place = \(String(format: "%.6f", weightedInPlace))"
                    + ", noscale in-place = \(String(format: "%.6f", noScaleInPlace))\n",
                stderr
            )

            XCTAssertEqual(weightedOutOfPlace, 0, accuracy: 0.000001)
        }
    }

    func testRmsNormGatedBatched_Qwen0808Shape_MatchesCPUReference() throws {
        let pipeline = makePipeline(
            shaderFile: "norms.metal",
            functionName: "rms_norm_gated"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile rms_norm_gated")

        let headDim = 128
        let numHeads = 16
        let batchSize = 18
        let batchHeads = batchSize * numHeads
        let eps: Float = 1e-6

        var input = [Float16](repeating: 0, count: batchHeads * headDim)
        var gate = [Float16](repeating: 0, count: batchHeads * headDim)
        var weight = [Float16](repeating: 0, count: headDim)

        for i in 0..<input.count {
            input[i] = Float16(sin(Float(i) * 0.0131) * 0.2)
            gate[i] = Float16(cos(Float(i) * 0.0097) * 2.0)
        }
        for i in 0..<weight.count {
            weight[i] = Float16(0.9 + 0.2 * sin(Float(i) * 0.021))
        }

        let reference = referenceRmsNormGatedCPU(
            input: input,
            gate: gate,
            weight: weight,
            headDim: headDim,
            batchHeads: batchHeads,
            eps: eps
        )

        let inputBuf = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!
        let gateBuf = device.makeBuffer(
            bytes: gate,
            length: gate.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!
        let weightBuf = device.makeBuffer(
            bytes: weight,
            length: weight.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!
        let outputBuf = device.makeBuffer(
            length: input.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!
        memset(outputBuf.contents(), 0, input.count * MemoryLayout<Float16>.stride)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline!)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(gateBuf, offset: 0, index: 1)
        enc.setBuffer(weightBuf, offset: 0, index: 2)
        enc.setBuffer(outputBuf, offset: 0, index: 3)
        var headDimVal = UInt32(headDim)
        enc.setBytes(&headDimVal, length: 4, index: 4)
        var epsVal = eps
        enc.setBytes(&epsVal, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: batchHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(headDim, 256), height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error, "Metal error: \(cmdBuf.error?.localizedDescription ?? "")")

        let outPtr = outputBuf.contents().bindMemory(
            to: Float16.self,
            capacity: input.count
        )
        var maxDiff: Float = 0
        for i in 0..<input.count {
            let diff = abs(Float(outPtr[i]) - reference[i])
            if diff > maxDiff { maxDiff = diff }
        }

        fputs(
            "  rms_norm_gated batched qwen0.8b shape B=\(batchSize):"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 1e-3, "rms_norm_gated diverges from CPU reference")
    }

    func testRmsNormGatedD128Batched_Qwen0808Shape_MatchesCPUReference() throws {
        let pipeline = makePipeline(
            shaderFile: "norms.metal",
            functionName: "rms_norm_gated_d128_batched"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile rms_norm_gated_d128_batched")

        let headDim = 128
        let numHeads = 16
        let batchSize = 65
        let batchHeads = batchSize * numHeads
        let eps: Float = 1e-6

        var input = [Float16](repeating: 0, count: batchHeads * headDim)
        var gate = [Float16](repeating: 0, count: batchHeads * headDim)
        var weight = [Float16](repeating: 0, count: headDim)

        for i in 0..<input.count {
            input[i] = Float16(sin(Float(i) * 0.0131) * 0.2)
            gate[i] = Float16(cos(Float(i) * 0.0097) * 2.0)
        }
        for i in 0..<weight.count {
            weight[i] = Float16(0.9 + 0.2 * sin(Float(i) * 0.021))
        }

        let reference = referenceRmsNormGatedCPU(
            input: input,
            gate: gate,
            weight: weight,
            headDim: headDim,
            batchHeads: batchHeads,
            eps: eps
        )

        let inputBuf = device.makeBuffer(
            bytes: input,
            length: input.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!
        let gateBuf = device.makeBuffer(
            bytes: gate,
            length: gate.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!
        let weightBuf = device.makeBuffer(
            bytes: weight,
            length: weight.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!
        let outputBuf = device.makeBuffer(
            length: input.count * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        )!
        memset(outputBuf.contents(), 0, input.count * MemoryLayout<Float16>.stride)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline!)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(gateBuf, offset: 0, index: 1)
        enc.setBuffer(weightBuf, offset: 0, index: 2)
        enc.setBuffer(outputBuf, offset: 0, index: 3)
        var numHeadsVal = UInt32(numHeads)
        enc.setBytes(&numHeadsVal, length: 4, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: numHeads, height: batchSize, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error, "Metal error: \(cmdBuf.error?.localizedDescription ?? "")")

        let outPtr = outputBuf.contents().bindMemory(
            to: Float16.self,
            capacity: input.count
        )
        var maxDiff: Float = 0
        for i in 0..<input.count {
            let diff = abs(Float(outPtr[i]) - reference[i])
            if diff > maxDiff { maxDiff = diff }
        }

        fputs(
            "  rms_norm_gated_d128_batched qwen0.8b shape B=\(batchSize):"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.0021, "rms_norm_gated_d128_batched diverges from CPU reference")
    }

    func testRmsNormGatedD128Batched_MatchesDecodeKernelAtLargeBatchBoundaries() throws {
        let decodePipeline = makePipeline(
            shaderFile: "norms.metal",
            functionName: "rms_norm_gated_d128"
        )
        try XCTSkipIf(decodePipeline == nil, "Could not compile rms_norm_gated_d128")

        let batchedPipeline = makePipeline(
            shaderFile: "norms.metal",
            functionName: "rms_norm_gated_d128_batched"
        )
        try XCTSkipIf(
            batchedPipeline == nil,
            "Could not compile rms_norm_gated_d128_batched"
        )

        let headDim = 128
        let numHeads = 16

        for batchSize in [64, 65] {
            let batchHeads = batchSize * numHeads
            var input = [Float16](repeating: 0, count: batchHeads * headDim)
            var gate = [Float16](repeating: 0, count: batchHeads * headDim)
            var weight = [Float16](repeating: 0, count: headDim)

            for i in 0..<input.count {
                input[i] = Float16(sin(Float(i) * 0.0131) * 0.2)
                gate[i] = Float16(cos(Float(i) * 0.0097) * 2.0)
            }
            for i in 0..<weight.count {
                weight[i] = Float16(0.9 + 0.2 * sin(Float(i) * 0.021))
            }

            let weightBuf = device.makeBuffer(
                bytes: weight,
                length: weight.count * MemoryLayout<Float16>.stride,
                options: .storageModeShared
            )!
            let batchedInputBuf = device.makeBuffer(
                bytes: input,
                length: input.count * MemoryLayout<Float16>.stride,
                options: .storageModeShared
            )!
            let batchedGateBuf = device.makeBuffer(
                bytes: gate,
                length: gate.count * MemoryLayout<Float16>.stride,
                options: .storageModeShared
            )!
            let batchedOutputBuf = device.makeBuffer(
                length: input.count * MemoryLayout<Float16>.stride,
                options: .storageModeShared
            )!
            memset(batchedOutputBuf.contents(), 0, input.count * MemoryLayout<Float16>.stride)

            let batchedCmd = queue.makeCommandBuffer()!
            let batchedEnc = batchedCmd.makeComputeCommandEncoder()!
            batchedEnc.setComputePipelineState(batchedPipeline!)
            batchedEnc.setBuffer(batchedInputBuf, offset: 0, index: 0)
            batchedEnc.setBuffer(batchedGateBuf, offset: 0, index: 1)
            batchedEnc.setBuffer(weightBuf, offset: 0, index: 2)
            batchedEnc.setBuffer(batchedOutputBuf, offset: 0, index: 3)
            var numHeadsVal = UInt32(numHeads)
            batchedEnc.setBytes(&numHeadsVal, length: 4, index: 4)
            batchedEnc.dispatchThreadgroups(
                MTLSize(width: numHeads, height: batchSize, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
            batchedEnc.endEncoding()
            batchedCmd.commit()
            batchedCmd.waitUntilCompleted()
            XCTAssertNil(batchedCmd.error)

            var reference = [Float16](repeating: 0, count: input.count)
            for batch in 0..<batchSize {
                let base = batch * numHeads * headDim
                let decodeInputBuf = device.makeBuffer(
                    bytes: Array(input[base..<(base + numHeads * headDim)]),
                    length: numHeads * headDim * MemoryLayout<Float16>.stride,
                    options: .storageModeShared
                )!
                let decodeGateBuf = device.makeBuffer(
                    bytes: Array(gate[base..<(base + numHeads * headDim)]),
                    length: numHeads * headDim * MemoryLayout<Float16>.stride,
                    options: .storageModeShared
                )!
                let decodeOutputBuf = device.makeBuffer(
                    length: numHeads * headDim * MemoryLayout<Float16>.stride,
                    options: .storageModeShared
                )!
                memset(
                    decodeOutputBuf.contents(),
                    0,
                    numHeads * headDim * MemoryLayout<Float16>.stride
                )

                let cmdBuf = queue.makeCommandBuffer()!
                let enc = cmdBuf.makeComputeCommandEncoder()!
                enc.setComputePipelineState(decodePipeline!)
                enc.setBuffer(decodeInputBuf, offset: 0, index: 0)
                enc.setBuffer(decodeGateBuf, offset: 0, index: 1)
                enc.setBuffer(weightBuf, offset: 0, index: 2)
                enc.setBuffer(decodeOutputBuf, offset: 0, index: 3)
                enc.dispatchThreadgroups(
                    MTLSize(width: numHeads, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
                XCTAssertNil(cmdBuf.error)

                let ptr = decodeOutputBuf.contents().bindMemory(
                    to: Float16.self,
                    capacity: numHeads * headDim
                )
                for i in 0..<(numHeads * headDim) {
                    reference[base + i] = ptr[i]
                }
            }

            let outPtr = batchedOutputBuf.contents().bindMemory(
                to: Float16.self,
                capacity: input.count
            )
            var maxDiff: Float = 0
            for i in 0..<input.count {
                let diff = abs(Float(outPtr[i]) - Float(reference[i]))
                if diff > maxDiff { maxDiff = diff }
            }

            fputs(
                "  rms_norm_gated_d128_batched vs decode B=\(batchSize):"
                    + " max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(
                maxDiff,
                0.0021,
                "rms_norm_gated_d128_batched diverges from rms_norm_gated_d128"
            )
        }
    }

    func testRmsNormBatched_RealDim_LargeBatchBoundaries() throws {
        let normPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw"
        )
        try XCTSkipIf(normPipeline == nil, "Could not compile decode rms_norm_1pw shader")

        let batchedPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw_batched"
        )
        try XCTSkipIf(batchedPipeline == nil, "Could not compile rms_norm_1pw_batched pipeline")

        let dim = 2048
        let eps: Float = 1e-6
        var weightData = [Float16](repeating: 0, count: dim)
        for i in 0..<dim {
            weightData[i] = Float16(cos(Float(i) * 0.0027) * 0.04)
        }
        let weightBuf = device.makeBuffer(
            bytes: weightData, length: dim * 2, options: .storageModeShared
        )!

        for batchSize in [64, 65] {
            var inputData = [Float16](repeating: 0, count: batchSize * dim)
            for b in 0..<batchSize {
                for d in 0..<dim {
                    inputData[b * dim + d] = Float16(sin(Float((b + 3) * (d + 1)) * 0.0011) * 1.3)
                }
            }

            var reference = [Float](repeating: 0, count: batchSize * dim)
            for b in 0..<batchSize {
                let singleInputBuf = device.makeBuffer(
                    bytes: Array(inputData[(b * dim)..<((b + 1) * dim)]),
                    length: dim * 2,
                    options: .storageModeShared
                )!
                let singleOutputBuf = device.makeBuffer(
                    length: dim * 2, options: .storageModeShared
                )!
                memset(singleOutputBuf.contents(), 0, dim * 2)

                let cmdBuf = queue.makeCommandBuffer()!
                let enc = cmdBuf.makeComputeCommandEncoder()!
                enc.setComputePipelineState(normPipeline!)
                enc.setBuffer(singleInputBuf, offset: 0, index: 0)
                enc.setBuffer(weightBuf, offset: 0, index: 1)
                enc.setBuffer(singleOutputBuf, offset: 0, index: 2)
                var dimVal = UInt32(dim)
                enc.setBytes(&dimVal, length: 4, index: 3)
                var epsVal = eps
                enc.setBytes(&epsVal, length: 4, index: 4)
                enc.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)
                )
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
                XCTAssertNil(cmdBuf.error)

                let ptr = singleOutputBuf.contents().bindMemory(
                    to: Float16.self, capacity: dim
                )
                for d in 0..<dim {
                    reference[b * dim + d] = Float(ptr[d])
                }
            }

            let batchInputBuf = device.makeBuffer(
                bytes: inputData, length: inputData.count * 2,
                options: .storageModeShared
            )!
            let batchOutputBuf = device.makeBuffer(
                length: batchSize * dim * 2, options: .storageModeShared
            )!
            memset(batchOutputBuf.contents(), 0, batchSize * dim * 2)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(batchedPipeline!)
            enc.setBuffer(batchInputBuf, offset: 0, index: 0)
            enc.setBuffer(weightBuf, offset: 0, index: 1)
            enc.setBuffer(batchOutputBuf, offset: 0, index: 2)
            var dimVal = UInt32(dim)
            enc.setBytes(&dimVal, length: 4, index: 3)
            var epsVal = eps
            enc.setBytes(&epsVal, length: 4, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: batchSize, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let outPtr = batchOutputBuf.contents().bindMemory(
                to: Float16.self, capacity: batchSize * dim
            )
            var maxDiff: Float = 0
            for i in 0..<(batchSize * dim) {
                maxDiff = max(maxDiff, abs(Float(outPtr[i]) - reference[i]))
            }

            fputs(
                "  rms_norm batched real dim=\(dim) B=\(batchSize):"
                    + " max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 1e-3)
        }
    }

    func testRmsNormD1024Batched_MatchesDecodeSpecializationAtLargeBatchBoundaries() throws {
        let decodePipeline = makePipeline(
            shaderFile: "norms.metal",
            functionName: "rms_norm_1pw_d1024"
        )
        try XCTSkipIf(decodePipeline == nil, "Could not compile rms_norm_1pw_d1024")

        let batchedPipeline = makePipeline(
            shaderFile: "norms.metal",
            functionName: "rms_norm_1pw_d1024_batched"
        )
        try XCTSkipIf(batchedPipeline == nil, "Could not compile rms_norm_1pw_d1024_batched")

        let dim = 1024
        let eps: Float = 1e-6

        for batchSize in [64, 65] {
            var inputData = [Float16](repeating: 0, count: batchSize * dim)
            for i in 0..<inputData.count {
                inputData[i] = Float16(sin(Float(i) * 0.003) * 1.5)
            }
            var weightData = [Float16](repeating: 0, count: dim)
            for i in 0..<dim {
                weightData[i] = Float16(cos(Float(i) * 0.01) * 0.02)
            }

            let weightBuf = device.makeBuffer(
                bytes: weightData,
                length: dim * 2,
                options: .storageModeShared
            )!

            var reference = [Float16](repeating: 0, count: batchSize * dim)
            for batch in 0..<batchSize {
                let slice = Array(inputData[(batch * dim)..<((batch + 1) * dim)])
                let inputBuf = device.makeBuffer(
                    bytes: slice,
                    length: dim * 2,
                    options: .storageModeShared
                )!
                let outputBuf = device.makeBuffer(
                    length: dim * 2,
                    options: .storageModeShared
                )!
                memset(outputBuf.contents(), 0, dim * 2)

                let cmdBuf = queue.makeCommandBuffer()!
                let enc = cmdBuf.makeComputeCommandEncoder()!
                enc.setComputePipelineState(decodePipeline!)
                enc.setBuffer(inputBuf, offset: 0, index: 0)
                enc.setBuffer(weightBuf, offset: 0, index: 1)
                enc.setBuffer(outputBuf, offset: 0, index: 2)
                enc.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
                )
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
                XCTAssertNil(cmdBuf.error)

                let ptr = outputBuf.contents().bindMemory(to: Float16.self, capacity: dim)
                for i in 0..<dim {
                    reference[batch * dim + i] = ptr[i]
                }
            }

            let batchInputBuf = device.makeBuffer(
                bytes: inputData,
                length: inputData.count * 2,
                options: .storageModeShared
            )!
            let batchOutputBuf = device.makeBuffer(
                length: batchSize * dim * 2,
                options: .storageModeShared
            )!
            memset(batchOutputBuf.contents(), 0, batchSize * dim * 2)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(batchedPipeline!)
            enc.setBuffer(batchInputBuf, offset: 0, index: 0)
            enc.setBuffer(weightBuf, offset: 0, index: 1)
            enc.setBuffer(batchOutputBuf, offset: 0, index: 2)
            var dimVal = UInt32(dim)
            enc.setBytes(&dimVal, length: 4, index: 3)
            var epsVal = eps
            enc.setBytes(&epsVal, length: 4, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: batchSize, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let outPtr = batchOutputBuf.contents().bindMemory(
                to: Float16.self,
                capacity: batchSize * dim
            )
            var maxDiff: Float = 0
            for i in 0..<(batchSize * dim) {
                maxDiff = max(maxDiff, abs(Float(outPtr[i]) - Float(reference[i])))
            }

            fputs(
                "  rms_norm_1pw_d1024_batched vs d1024 B=\(batchSize):"
                    + " max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.0001, "Specialized batched RMSNorm diverges from d1024 specialization")
        }
    }

    // MARK: - Embedding Gather Batched (Priority 3)

    /// Batched FP16 embedding gather — must be exact match (pure lookup).
    func testEmbeddingGatherBatched_ExactMatch() throws {
        let gatherPipeline = makePipeline(
            shaderFile: "lut_matvec.metal", functionName: "embedding_gather"
        )
        try XCTSkipIf(gatherPipeline == nil, "Could not compile decode embedding shader")

        let batchedPipeline = makePipeline(
            shaderFile: "lut_matvec.metal", functionName: "embedding_gather_batched"
        )
        try XCTSkipIf(batchedPipeline == nil, "Could not compile embedding_gather_batched pipeline")

        let vocab = 256
        let hidden = 64
        let batchSize = 8
        let tokenIds: [Int32] = [3, 100, 0, 255, 42, 7, 200, 128]

        // Build embedding table
        var table = [Float16](repeating: 0, count: vocab * hidden)
        for i in 0..<table.count {
            table[i] = Float16(Float(i) * 0.001)
        }
        let tableBuf = device.makeBuffer(
            bytes: table, length: table.count * 2, options: .storageModeShared
        )!

        // Reference: gather one at a time
        var reference = [Float16](repeating: 0, count: batchSize * hidden)
        let tokenBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
        let singleOutputBuf = device.makeBuffer(
            length: hidden * 2, options: .storageModeShared
        )!

        for b in 0..<batchSize {
            tokenBuf.contents().storeBytes(of: tokenIds[b], as: Int32.self)
            memset(singleOutputBuf.contents(), 0, hidden * 2)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(gatherPipeline!)
            enc.setBuffer(tableBuf, offset: 0, index: 0)
            enc.setBuffer(tokenBuf, offset: 0, index: 1)
            enc.setBuffer(singleOutputBuf, offset: 0, index: 2)
            var hiddenVal = UInt32(hidden)
            enc.setBytes(&hiddenVal, length: 4, index: 3)
            enc.dispatchThreads(
                MTLSize(width: hidden, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(hidden, 256), height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            let ptr = singleOutputBuf.contents().bindMemory(
                to: Float16.self, capacity: hidden
            )
            for d in 0..<hidden {
                reference[b * hidden + d] = ptr[d]
            }
        }

        // Test: batched gather
        let tokenIdsBuf = device.makeBuffer(
            bytes: tokenIds, length: batchSize * 4, options: .storageModeShared
        )!
        let batchOutputBuf = device.makeBuffer(
            length: batchSize * hidden * 2, options: .storageModeShared
        )!
        memset(batchOutputBuf.contents(), 0, batchSize * hidden * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchedPipeline!)
        enc.setBuffer(tableBuf, offset: 0, index: 0)
        enc.setBuffer(tokenIdsBuf, offset: 0, index: 1)
        enc.setBuffer(batchOutputBuf, offset: 0, index: 2)
        var hiddenVal = UInt32(hidden)
        enc.setBytes(&hiddenVal, length: 4, index: 3)
        var bsVal = UInt32(batchSize)
        enc.setBytes(&bsVal, length: 4, index: 4)
        enc.dispatchThreads(
            MTLSize(width: hidden, height: batchSize, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(hidden, 256), height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        // Exact match — no arithmetic, pure lookup
        let outPtr = batchOutputBuf.contents().bindMemory(
            to: Float16.self, capacity: batchSize * hidden
        )
        var mismatches = 0
        for i in 0..<(batchSize * hidden) {
            if outPtr[i] != reference[i] { mismatches += 1 }
        }

        fputs(
            "  embedding_gather_batched exact match:"
                + " \(mismatches)/\(batchSize * hidden) mismatches\n",
            stderr
        )
        XCTAssertEqual(mismatches, 0, "Batched gather must exactly match iterated gather")
    }

    func testAffineEmbeddingGatherBatched_ExactMatch() throws {
        let gatherPipeline = makePipeline(
            shaderFile: "lut_matvec.metal", functionName: "affine_embedding_gather"
        )
        try XCTSkipIf(gatherPipeline == nil, "Could not compile affine embedding shader")

        let batchedPipeline = makePipeline(
            shaderFile: "lut_matvec.metal", functionName: "affine_embedding_gather_batched"
        )
        try XCTSkipIf(
            batchedPipeline == nil,
            "Could not compile affine_embedding_gather_batched pipeline"
        )

        let vocab = 256
        let hidden = 64
        let groupSize = 16
        let batchSize = 8
        let tokenIds: [Int32] = [3, 100, 0, 255, 42, 7, 200, 128]
        let affine = makeAffineTestData(rows: vocab, cols: hidden, groupSize: groupSize)

        let weightsBuf = device.makeBuffer(
            bytes: affine.weights, length: affine.weights.count, options: .storageModeShared
        )!
        let scalesBuf = device.makeBuffer(
            bytes: affine.scales, length: affine.scales.count * 2, options: .storageModeShared
        )!
        let biasesBuf = device.makeBuffer(
            bytes: affine.biases, length: affine.biases.count * 2, options: .storageModeShared
        )!

        var reference = [Float16](repeating: 0, count: batchSize * hidden)
        let tokenBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
        let singleOutputBuf = device.makeBuffer(
            length: hidden * 2, options: .storageModeShared
        )!

        for b in 0..<batchSize {
            tokenBuf.contents().storeBytes(of: tokenIds[b], as: Int32.self)
            memset(singleOutputBuf.contents(), 0, hidden * 2)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(gatherPipeline!)
            enc.setBuffer(weightsBuf, offset: 0, index: 0)
            enc.setBuffer(scalesBuf, offset: 0, index: 1)
            enc.setBuffer(biasesBuf, offset: 0, index: 2)
            enc.setBuffer(tokenBuf, offset: 0, index: 3)
            enc.setBuffer(singleOutputBuf, offset: 0, index: 4)
            var hiddenVal = UInt32(hidden)
            enc.setBytes(&hiddenVal, length: 4, index: 5)
            var groupSizeVal = UInt32(groupSize)
            enc.setBytes(&groupSizeVal, length: 4, index: 6)
            enc.dispatchThreads(
                MTLSize(width: hidden / 2, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(hidden / 2, 256), height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            let ptr = singleOutputBuf.contents().bindMemory(
                to: Float16.self, capacity: hidden
            )
            for d in 0..<hidden {
                reference[b * hidden + d] = ptr[d]
            }
        }

        let tokenIdsBuf = device.makeBuffer(
            bytes: tokenIds, length: batchSize * 4, options: .storageModeShared
        )!
        let batchOutputBuf = device.makeBuffer(
            length: batchSize * hidden * 2, options: .storageModeShared
        )!
        memset(batchOutputBuf.contents(), 0, batchSize * hidden * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchedPipeline!)
        enc.setBuffer(weightsBuf, offset: 0, index: 0)
        enc.setBuffer(scalesBuf, offset: 0, index: 1)
        enc.setBuffer(biasesBuf, offset: 0, index: 2)
        enc.setBuffer(tokenIdsBuf, offset: 0, index: 3)
        enc.setBuffer(batchOutputBuf, offset: 0, index: 4)
        var hiddenVal = UInt32(hidden)
        enc.setBytes(&hiddenVal, length: 4, index: 5)
        var batchSizeVal = UInt32(batchSize)
        enc.setBytes(&batchSizeVal, length: 4, index: 6)
        var groupSizeVal = UInt32(groupSize)
        enc.setBytes(&groupSizeVal, length: 4, index: 7)
        enc.dispatchThreads(
            MTLSize(width: hidden / 2, height: batchSize, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(hidden / 2, 256), height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = batchOutputBuf.contents().bindMemory(
            to: Float16.self, capacity: batchSize * hidden
        )
        var mismatches = 0
        for i in 0..<(batchSize * hidden) {
            if outPtr[i] != reference[i] { mismatches += 1 }
        }

        fputs(
            "  affine_embedding_gather_batched exact match:"
                + " \(mismatches)/\(batchSize * hidden) mismatches\n",
            stderr
        )
        XCTAssertEqual(mismatches, 0, "Batched affine gather must exactly match iterated gather")
    }

    // MARK: - Attention Prefill (Priority 4)
    // Layout: [B, numHeads, headDim] — batch is outermost dimension.
    // This matches the natural output of fused_lut_matmul.

    /// Batched causal attention matches incrementally-built decode attention.
    func testAttentionPrefill_MatchesIncrementalDecode() throws {
        let decodePipeline = makePipeline(
            shaderFile: "attention.metal", functionName: "attention_decode"
        )
        try XCTSkipIf(decodePipeline == nil, "Could not compile decode attention shader")

        let prefillPipeline = makePipeline(
            shaderFile: "prefill_attention.metal", functionName: "attention_prefill"
        )
        try XCTSkipIf(prefillPipeline == nil, "Could not compile attention_prefill pipeline")

        let numQHeads = 4
        let numKVHeads = 2
        let headDim = 32
        let batchSize = 8
        let startPos = 3
        let scale = 1.0 / sqrt(Float(headDim))

        let totalQ = batchSize * numQHeads * headDim
        let totalKV = batchSize * numKVHeads * headDim
        let prefixKV = startPos * numKVHeads * headDim
        var qData = [Float16](repeating: 0, count: totalQ)
        var kData = [Float16](repeating: 0, count: totalKV)
        var vData = [Float16](repeating: 0, count: totalKV)
        var prefixKData = [Float16](repeating: 0, count: prefixKV)
        var prefixVData = [Float16](repeating: 0, count: prefixKV)
        for i in 0..<totalQ { qData[i] = Float16(sin(Float(i) * 0.037) * 0.5) }
        for i in 0..<totalKV { kData[i] = Float16(cos(Float(i) * 0.053) * 0.5) }
        for i in 0..<totalKV { vData[i] = Float16(sin(Float(i) * 0.071) * 0.5) }
        for i in 0..<prefixKV { prefixKData[i] = Float16(cos(Float(i) * 0.041) * 0.45) }
        for i in 0..<prefixKV { prefixVData[i] = Float16(sin(Float(i) * 0.067) * 0.4) }

        let maxSeq = startPos + batchSize
        let cacheSize = numKVHeads * maxSeq * headDim
        let kCacheBuf = device.makeBuffer(length: cacheSize * 2, options: .storageModeShared)!
        let vCacheBuf = device.makeBuffer(length: cacheSize * 2, options: .storageModeShared)!
        memset(kCacheBuf.contents(), 0, cacheSize * 2)
        memset(vCacheBuf.contents(), 0, cacheSize * 2)

        let maskBuf = device.makeBuffer(length: maxSeq * 2, options: .storageModeShared)!
        let queryBuf = device.makeBuffer(length: numQHeads * headDim * 2, options: .storageModeShared)!
        let decodeOutBuf = device.makeBuffer(length: numQHeads * headDim * 2, options: .storageModeShared)!

        var reference = [Float](repeating: 0, count: totalQ)
        let kCachePtr = kCacheBuf.contents().bindMemory(to: Float16.self, capacity: cacheSize)
        let vCachePtr = vCacheBuf.contents().bindMemory(to: Float16.self, capacity: cacheSize)

        for pos in 0..<startPos {
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let cacheIdx = h * maxSeq * headDim + pos * headDim + d
                    let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                    kCachePtr[cacheIdx] = prefixKData[srcIdx]
                    vCachePtr[cacheIdx] = prefixVData[srcIdx]
                }
            }
        }

        for pos in 0..<batchSize {
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let cacheIdx = h * maxSeq * headDim + (startPos + pos) * headDim + d
                    let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                    kCachePtr[cacheIdx] = kData[srcIdx]
                    vCachePtr[cacheIdx] = vData[srcIdx]
                }
            }

            let qPtr = queryBuf.contents().bindMemory(to: Float16.self, capacity: numQHeads * headDim)
            let qSrcBase = pos * numQHeads * headDim
            for i in 0..<(numQHeads * headDim) {
                qPtr[i] = qData[qSrcBase + i]
            }

            let maskPtr = maskBuf.contents().bindMemory(to: Float16.self, capacity: maxSeq)
            for i in 0..<maxSeq {
                maskPtr[i] = i <= startPos + pos ? Float16(0) : Float16(-10000)
            }

            memset(decodeOutBuf.contents(), 0, numQHeads * headDim * 2)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(decodePipeline!)
            enc.setBuffer(queryBuf, offset: 0, index: 0)
            enc.setBuffer(kCacheBuf, offset: 0, index: 1)
            enc.setBuffer(vCacheBuf, offset: 0, index: 2)
            enc.setBuffer(maskBuf, offset: 0, index: 3)
            enc.setBuffer(decodeOutBuf, offset: 0, index: 4)
            var hdVal = UInt32(headDim); enc.setBytes(&hdVal, length: 4, index: 5)
            var msVal = UInt32(maxSeq); enc.setBytes(&msVal, length: 4, index: 6)
            var slVal = UInt32(startPos + pos + 1); enc.setBytes(&slVal, length: 4, index: 7)
            var kvhVal = UInt32(numKVHeads); enc.setBytes(&kvhVal, length: 4, index: 8)
            var scaleVal = scale; enc.setBytes(&scaleVal, length: 4, index: 9)
            var windowVal: UInt32 = 0; enc.setBytes(&windowVal, length: 4, index: 10)
            enc.dispatchThreadgroups(
                MTLSize(width: numQHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let outPtr = decodeOutBuf.contents().bindMemory(to: Float16.self, capacity: numQHeads * headDim)
            let refBase = pos * numQHeads * headDim
            for i in 0..<(numQHeads * headDim) {
                reference[refBase + i] = Float(outPtr[i])
            }
        }

        let qBuf = device.makeBuffer(bytes: qData, length: totalQ * 2, options: .storageModeShared)!
        let prefillOutBuf = device.makeBuffer(length: totalQ * 2, options: .storageModeShared)!
        memset(prefillOutBuf.contents(), 0, totalQ * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(prefillPipeline!)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kCacheBuf, offset: 0, index: 1)
        enc.setBuffer(vCacheBuf, offset: 0, index: 2)
        enc.setBuffer(prefillOutBuf, offset: 0, index: 3)
        var hdVal = UInt32(headDim); enc.setBytes(&hdVal, length: 4, index: 4)
        var slVal = UInt32(batchSize); enc.setBytes(&slVal, length: 4, index: 5)
        var startPosVal = UInt32(startPos); enc.setBytes(&startPosVal, length: 4, index: 6)
        var cacheSeqVal = UInt32(maxSeq); enc.setBytes(&cacheSeqVal, length: 4, index: 7)
        var kvhVal = UInt32(numKVHeads); enc.setBytes(&kvhVal, length: 4, index: 8)
        var scaleVal = scale; enc.setBytes(&scaleVal, length: 4, index: 9)
        var windowVal: UInt32 = 0; enc.setBytes(&windowVal, length: 4, index: 10)
        enc.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: batchSize, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = prefillOutBuf.contents().bindMemory(to: Float16.self, capacity: totalQ)
        var maxDiff: Float = 0
        for i in 0..<totalQ {
            let diff = abs(Float(outPtr[i]) - reference[i])
            if diff > maxDiff { maxDiff = diff }
        }

        fputs(
            "  attention_prefill vs incremental decode startPos=\(startPos), B=\(batchSize):"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.05, "Prefill attention diverges from incremental decode")
    }

    func testAttentionPrefill_LlamaHead64MatchesCPUReference() throws {
        let prefillPipeline = makePipeline(
            shaderFile: "prefill_attention.metal", functionName: "attention_prefill"
        )
        try XCTSkipIf(prefillPipeline == nil, "Could not compile attention_prefill pipeline")

        let numQHeads = 32
        let numKVHeads = 8
        let headDim = 64
        let gqaRatio = numQHeads / numKVHeads
        let batchSize = 6
        let startPos = 2
        let scale: Float = 1.0 / sqrt(Float(headDim))

        let totalQ = batchSize * numQHeads * headDim
        let totalKV = batchSize * numKVHeads * headDim
        let prefixKV = startPos * numKVHeads * headDim
        var qData = [Float16](repeating: 0, count: totalQ)
        var kData = [Float16](repeating: 0, count: totalKV)
        var vData = [Float16](repeating: 0, count: totalKV)
        var prefixKData = [Float16](repeating: 0, count: prefixKV)
        var prefixVData = [Float16](repeating: 0, count: prefixKV)
        for i in 0..<totalQ { qData[i] = Float16(sin(Float(i) * 0.019) * 0.35) }
        for i in 0..<totalKV { kData[i] = Float16(cos(Float(i) * 0.023) * 0.4) }
        for i in 0..<totalKV { vData[i] = Float16(sin(Float(i) * 0.029) * 0.45) }
        for i in 0..<prefixKV { prefixKData[i] = Float16(cos(Float(i) * 0.031) * 0.3) }
        for i in 0..<prefixKV { prefixVData[i] = Float16(sin(Float(i) * 0.037) * 0.25) }

        let maxSeq = startPos + batchSize
        let cacheElems = numKVHeads * maxSeq * headDim
        var keyCache = [Float16](repeating: 0, count: cacheElems)
        var valCache = [Float16](repeating: 0, count: cacheElems)

        for pos in 0..<startPos {
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let cacheIdx = h * maxSeq * headDim + pos * headDim + d
                    let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                    keyCache[cacheIdx] = prefixKData[srcIdx]
                    valCache[cacheIdx] = prefixVData[srcIdx]
                }
            }
        }
        for pos in 0..<batchSize {
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let cacheIdx = h * maxSeq * headDim + (startPos + pos) * headDim + d
                    let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                    keyCache[cacheIdx] = kData[srcIdx]
                    valCache[cacheIdx] = vData[srcIdx]
                }
            }
        }

        var reference = [Float](repeating: 0, count: totalQ)
        for pos in 0..<batchSize {
            let causalLen = startPos + pos + 1
            for qHead in 0..<numQHeads {
                let kvHead = qHead / gqaRatio
                var scores = [Float](repeating: 0, count: causalLen)
                var maxScore = -Float.infinity
                for s in 0..<causalLen {
                    var dot: Float = 0
                    let qBase = pos * numQHeads * headDim + qHead * headDim
                    let kBase = kvHead * maxSeq * headDim + s * headDim
                    for d in 0..<headDim {
                        dot += Float(qData[qBase + d]) * Float(keyCache[kBase + d])
                    }
                    let score = dot * scale
                    scores[s] = score
                    maxScore = max(maxScore, score)
                }
                var sumExp: Float = 0
                for s in 0..<causalLen {
                    scores[s] = exp(scores[s] - maxScore)
                    sumExp += scores[s]
                }
                let invSum = 1.0 / sumExp
                for d in 0..<headDim {
                    var acc: Float = 0
                    for s in 0..<causalLen {
                        let vBase = kvHead * maxSeq * headDim + s * headDim
                        acc += scores[s] * invSum * Float(valCache[vBase + d])
                    }
                    let outIdx = pos * numQHeads * headDim + qHead * headDim + d
                    reference[outIdx] = acc
                }
            }
        }

        let qBuf = device.makeBuffer(bytes: qData, length: totalQ * 2, options: .storageModeShared)!
        let kCacheBuf = device.makeBuffer(bytes: keyCache, length: cacheElems * 2, options: .storageModeShared)!
        let vCacheBuf = device.makeBuffer(bytes: valCache, length: cacheElems * 2, options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: totalQ * 2, options: .storageModeShared)!
        memset(outputBuf.contents(), 0, totalQ * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(prefillPipeline!)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kCacheBuf, offset: 0, index: 1)
        enc.setBuffer(vCacheBuf, offset: 0, index: 2)
        enc.setBuffer(outputBuf, offset: 0, index: 3)
        var headDimVal = UInt32(headDim); enc.setBytes(&headDimVal, length: 4, index: 4)
        var seqLenVal = UInt32(batchSize); enc.setBytes(&seqLenVal, length: 4, index: 5)
        var startPosVal = UInt32(startPos); enc.setBytes(&startPosVal, length: 4, index: 6)
        var cacheSeqVal = UInt32(maxSeq); enc.setBytes(&cacheSeqVal, length: 4, index: 7)
        var kvHeadsVal = UInt32(numKVHeads); enc.setBytes(&kvHeadsVal, length: 4, index: 8)
        var scaleVal = scale; enc.setBytes(&scaleVal, length: 4, index: 9)
        var windowVal: UInt32 = 0; enc.setBytes(&windowVal, length: 4, index: 10)
        enc.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: batchSize, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = outputBuf.contents().bindMemory(to: Float16.self, capacity: totalQ)
        var maxDiff: Float = 0
        for i in 0..<totalQ {
            maxDiff = max(maxDiff, abs(Float(outPtr[i]) - reference[i]))
        }

        fputs(
            "  attention_prefill llama head64 vs cpu:"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.05)
    }

    func testAttentionPrefillSoftcap_MatchesCPUReference() throws {
        let prefillPipeline = makePipeline(
            shaderFile: "prefill_attention.metal", functionName: "attention_prefill_softcap"
        )
        try XCTSkipIf(prefillPipeline == nil, "Could not compile attention_prefill_softcap pipeline")

        let numQHeads = 4
        let numKVHeads = 2
        let headDim = 32
        let gqaRatio = numQHeads / numKVHeads
        let batchSize = 6
        let startPos = 2
        let scale: Float = 1.0 / sqrt(Float(headDim))
        let softcap: Float = 5.0

        let totalQ = batchSize * numQHeads * headDim
        let totalKV = batchSize * numKVHeads * headDim
        let prefixKV = startPos * numKVHeads * headDim
        var qData = [Float16](repeating: 0, count: totalQ)
        var kData = [Float16](repeating: 0, count: totalKV)
        var vData = [Float16](repeating: 0, count: totalKV)
        var prefixKData = [Float16](repeating: 0, count: prefixKV)
        var prefixVData = [Float16](repeating: 0, count: prefixKV)
        for i in 0..<totalQ { qData[i] = Float16(sin(Float(i) * 0.031) * 0.4) }
        for i in 0..<totalKV { kData[i] = Float16(cos(Float(i) * 0.047) * 0.45) }
        for i in 0..<totalKV { vData[i] = Float16(sin(Float(i) * 0.059) * 0.5) }
        for i in 0..<prefixKV { prefixKData[i] = Float16(cos(Float(i) * 0.041) * 0.35) }
        for i in 0..<prefixKV { prefixVData[i] = Float16(sin(Float(i) * 0.071) * 0.3) }

        let maxSeq = startPos + batchSize
        let cacheElems = numKVHeads * maxSeq * headDim
        var keyCache = [Float16](repeating: 0, count: cacheElems)
        var valCache = [Float16](repeating: 0, count: cacheElems)

        for pos in 0..<startPos {
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let cacheIdx = h * maxSeq * headDim + pos * headDim + d
                    let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                    keyCache[cacheIdx] = prefixKData[srcIdx]
                    valCache[cacheIdx] = prefixVData[srcIdx]
                }
            }
        }
        for pos in 0..<batchSize {
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let cacheIdx = h * maxSeq * headDim + (startPos + pos) * headDim + d
                    let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                    keyCache[cacheIdx] = kData[srcIdx]
                    valCache[cacheIdx] = vData[srcIdx]
                }
            }
        }

        var reference = [Float](repeating: 0, count: totalQ)
        for pos in 0..<batchSize {
            let causalLen = startPos + pos + 1
            for qHead in 0..<numQHeads {
                let kvHead = qHead / gqaRatio
                var scores = [Float](repeating: 0, count: causalLen)
                var maxScore = -Float.infinity
                for s in 0..<causalLen {
                    var dot: Float = 0
                    let qBase = pos * numQHeads * headDim + qHead * headDim
                    let kBase = kvHead * maxSeq * headDim + s * headDim
                    for d in 0..<headDim {
                        dot += Float(qData[qBase + d]) * Float(keyCache[kBase + d])
                    }
                    let score = softcap * tanh((dot * scale) / softcap)
                    scores[s] = score
                    maxScore = max(maxScore, score)
                }
                var sumExp: Float = 0
                for s in 0..<causalLen {
                    scores[s] = exp(scores[s] - maxScore)
                    sumExp += scores[s]
                }
                let invSum = 1.0 / sumExp
                for d in 0..<headDim {
                    var acc: Float = 0
                    for s in 0..<causalLen {
                        let vBase = kvHead * maxSeq * headDim + s * headDim
                        acc += scores[s] * invSum * Float(valCache[vBase + d])
                    }
                    let outIdx = pos * numQHeads * headDim + qHead * headDim + d
                    reference[outIdx] = acc
                }
            }
        }

        let qBuf = device.makeBuffer(bytes: qData, length: totalQ * 2, options: .storageModeShared)!
        let kCacheBuf = device.makeBuffer(bytes: keyCache, length: cacheElems * 2, options: .storageModeShared)!
        let vCacheBuf = device.makeBuffer(bytes: valCache, length: cacheElems * 2, options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: totalQ * 2, options: .storageModeShared)!
        memset(outputBuf.contents(), 0, totalQ * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(prefillPipeline!)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kCacheBuf, offset: 0, index: 1)
        enc.setBuffer(vCacheBuf, offset: 0, index: 2)
        enc.setBuffer(outputBuf, offset: 0, index: 3)
        var headDimVal = UInt32(headDim); enc.setBytes(&headDimVal, length: 4, index: 4)
        var seqLenVal = UInt32(batchSize); enc.setBytes(&seqLenVal, length: 4, index: 5)
        var startPosVal = UInt32(startPos); enc.setBytes(&startPosVal, length: 4, index: 6)
        var cacheSeqVal = UInt32(maxSeq); enc.setBytes(&cacheSeqVal, length: 4, index: 7)
        var kvHeadsVal = UInt32(numKVHeads); enc.setBytes(&kvHeadsVal, length: 4, index: 8)
        var scaleVal = scale; enc.setBytes(&scaleVal, length: 4, index: 9)
        var windowVal: UInt32 = 0; enc.setBytes(&windowVal, length: 4, index: 10)
        var softcapVal = softcap; enc.setBytes(&softcapVal, length: 4, index: 11)
        enc.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: batchSize, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = outputBuf.contents().bindMemory(to: Float16.self, capacity: totalQ)
        var maxDiff: Float = 0
        for i in 0..<totalQ {
            maxDiff = max(maxDiff, abs(Float(outPtr[i]) - reference[i]))
        }

        fputs(
            "  attention_prefill_softcap vs cpu:"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.05)
    }

    func testAttentionPrefillSoftcap_GlobalShapeMatchesCPUReference() throws {
        let prefillPipeline = makePipeline(
            shaderFile: "prefill_attention.metal", functionName: "attention_prefill_softcap"
        )
        try XCTSkipIf(prefillPipeline == nil, "Could not compile attention_prefill_softcap pipeline")

        let numQHeads = 8
        let numKVHeads = 1
        let headDim = 512
        let gqaRatio = numQHeads / numKVHeads
        let batchSize = 2
        let startPos = 300
        let scale: Float = 1.0 / sqrt(Float(headDim))
        let softcap: Float = 30.0

        let totalQ = batchSize * numQHeads * headDim
        let totalKV = batchSize * numKVHeads * headDim
        let prefixKV = startPos * numKVHeads * headDim
        var qData = [Float16](repeating: 0, count: totalQ)
        var kData = [Float16](repeating: 0, count: totalKV)
        var vData = [Float16](repeating: 0, count: totalKV)
        var prefixKData = [Float16](repeating: 0, count: prefixKV)
        var prefixVData = [Float16](repeating: 0, count: prefixKV)
        for i in 0..<totalQ { qData[i] = Float16(sin(Float(i) * 0.007) * 0.4) }
        for i in 0..<totalKV { kData[i] = Float16(cos(Float(i) * 0.011) * 0.45) }
        for i in 0..<totalKV { vData[i] = Float16(sin(Float(i) * 0.013) * 0.5) }
        for i in 0..<prefixKV { prefixKData[i] = Float16(cos(Float(i) * 0.005) * 0.35) }
        for i in 0..<prefixKV { prefixVData[i] = Float16(sin(Float(i) * 0.009) * 0.3) }

        let maxSeq = startPos + batchSize
        let cacheElems = numKVHeads * maxSeq * headDim
        var keyCache = [Float16](repeating: 0, count: cacheElems)
        var valCache = [Float16](repeating: 0, count: cacheElems)

        for pos in 0..<startPos {
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let cacheIdx = h * maxSeq * headDim + pos * headDim + d
                    let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                    keyCache[cacheIdx] = prefixKData[srcIdx]
                    valCache[cacheIdx] = prefixVData[srcIdx]
                }
            }
        }
        for pos in 0..<batchSize {
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let cacheIdx = h * maxSeq * headDim + (startPos + pos) * headDim + d
                    let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                    keyCache[cacheIdx] = kData[srcIdx]
                    valCache[cacheIdx] = vData[srcIdx]
                }
            }
        }

        var reference = [Float](repeating: 0, count: totalQ)
        for pos in 0..<batchSize {
            let causalLen = startPos + pos + 1
            for qHead in 0..<numQHeads {
                let kvHead = qHead / gqaRatio
                var scores = [Float](repeating: 0, count: causalLen)
                var maxScore = -Float.infinity
                for s in 0..<causalLen {
                    var dot: Float = 0
                    let qBase = pos * numQHeads * headDim + qHead * headDim
                    let kBase = kvHead * maxSeq * headDim + s * headDim
                    for d in 0..<headDim {
                        dot += Float(qData[qBase + d]) * Float(keyCache[kBase + d])
                    }
                    let score = softcap * tanh((dot * scale) / softcap)
                    scores[s] = score
                    maxScore = max(maxScore, score)
                }
                var sumExp: Float = 0
                for s in 0..<causalLen {
                    scores[s] = exp(scores[s] - maxScore)
                    sumExp += scores[s]
                }
                let invSum = 1.0 / sumExp
                for d in 0..<headDim {
                    var acc: Float = 0
                    for s in 0..<causalLen {
                        let vBase = kvHead * maxSeq * headDim + s * headDim
                        acc += scores[s] * invSum * Float(valCache[vBase + d])
                    }
                    let outIdx = pos * numQHeads * headDim + qHead * headDim + d
                    reference[outIdx] = acc
                }
            }
        }

        let qBuf = device.makeBuffer(bytes: qData, length: totalQ * 2, options: .storageModeShared)!
        let kCacheBuf = device.makeBuffer(bytes: keyCache, length: cacheElems * 2, options: .storageModeShared)!
        let vCacheBuf = device.makeBuffer(bytes: valCache, length: cacheElems * 2, options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: totalQ * 2, options: .storageModeShared)!
        memset(outputBuf.contents(), 0, totalQ * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(prefillPipeline!)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kCacheBuf, offset: 0, index: 1)
        enc.setBuffer(vCacheBuf, offset: 0, index: 2)
        enc.setBuffer(outputBuf, offset: 0, index: 3)
        var headDimVal = UInt32(headDim); enc.setBytes(&headDimVal, length: 4, index: 4)
        var seqLenVal = UInt32(batchSize); enc.setBytes(&seqLenVal, length: 4, index: 5)
        var startPosVal = UInt32(startPos); enc.setBytes(&startPosVal, length: 4, index: 6)
        var cacheSeqVal = UInt32(maxSeq); enc.setBytes(&cacheSeqVal, length: 4, index: 7)
        var kvHeadsVal = UInt32(numKVHeads); enc.setBytes(&kvHeadsVal, length: 4, index: 8)
        var scaleVal = scale; enc.setBytes(&scaleVal, length: 4, index: 9)
        var windowVal: UInt32 = 0; enc.setBytes(&windowVal, length: 4, index: 10)
        var softcapVal = softcap; enc.setBytes(&softcapVal, length: 4, index: 11)
        enc.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: batchSize, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = outputBuf.contents().bindMemory(to: Float16.self, capacity: totalQ)
        var maxDiff: Float = 0
        for i in 0..<totalQ {
            maxDiff = max(maxDiff, abs(Float(outPtr[i]) - reference[i]))
        }

        fputs(
            "  attention_prefill_softcap global:"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.05)
    }

    func testAttentionPrefillSlidingWindow_MatchesCPUReference() throws {
        let prefillPipeline = makePipeline(
            shaderFile: "prefill_attention.metal", functionName: "attention_prefill"
        )
        try XCTSkipIf(prefillPipeline == nil, "Could not compile attention_prefill pipeline")

        let numQHeads = 4
        let numKVHeads = 2
        let headDim = 32
        let gqaRatio = numQHeads / numKVHeads
        let batchSize = 6
        let startPos = 5
        let slidingWindow = 4
        let scale: Float = 1.0 / sqrt(Float(headDim))

        let totalQ = batchSize * numQHeads * headDim
        let totalKV = batchSize * numKVHeads * headDim
        let prefixKV = startPos * numKVHeads * headDim
        var qData = [Float16](repeating: 0, count: totalQ)
        var kData = [Float16](repeating: 0, count: totalKV)
        var vData = [Float16](repeating: 0, count: totalKV)
        var prefixKData = [Float16](repeating: 0, count: prefixKV)
        var prefixVData = [Float16](repeating: 0, count: prefixKV)
        for i in 0..<totalQ { qData[i] = Float16(sin(Float(i) * 0.031) * 0.4) }
        for i in 0..<totalKV { kData[i] = Float16(cos(Float(i) * 0.047) * 0.45) }
        for i in 0..<totalKV { vData[i] = Float16(sin(Float(i) * 0.059) * 0.5) }
        for i in 0..<prefixKV { prefixKData[i] = Float16(cos(Float(i) * 0.041) * 0.35) }
        for i in 0..<prefixKV { prefixVData[i] = Float16(sin(Float(i) * 0.071) * 0.3) }

        let maxSeq = startPos + batchSize
        let cacheElems = numKVHeads * maxSeq * headDim
        var keyCache = [Float16](repeating: 0, count: cacheElems)
        var valCache = [Float16](repeating: 0, count: cacheElems)

        for pos in 0..<startPos {
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let cacheIdx = h * maxSeq * headDim + pos * headDim + d
                    let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                    keyCache[cacheIdx] = prefixKData[srcIdx]
                    valCache[cacheIdx] = prefixVData[srcIdx]
                }
            }
        }
        for pos in 0..<batchSize {
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let cacheIdx = h * maxSeq * headDim + (startPos + pos) * headDim + d
                    let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                    keyCache[cacheIdx] = kData[srcIdx]
                    valCache[cacheIdx] = vData[srcIdx]
                }
            }
        }

        var reference = [Float](repeating: 0, count: totalQ)
        for pos in 0..<batchSize {
            let causalLen = startPos + pos + 1
            let seqStart = max(0, causalLen - slidingWindow)
            for qHead in 0..<numQHeads {
                let kvHead = qHead / gqaRatio
                var scores = [Float](repeating: 0, count: causalLen)
                var maxScore = -Float.infinity
                for s in seqStart..<causalLen {
                    var dot: Float = 0
                    let qBase = pos * numQHeads * headDim + qHead * headDim
                    let kBase = kvHead * maxSeq * headDim + s * headDim
                    for d in 0..<headDim {
                        dot += Float(qData[qBase + d]) * Float(keyCache[kBase + d])
                    }
                    scores[s] = dot * scale
                    maxScore = max(maxScore, scores[s])
                }
                var sumExp: Float = 0
                for s in seqStart..<causalLen {
                    scores[s] = exp(scores[s] - maxScore)
                    sumExp += scores[s]
                }
                let invSum = 1.0 / sumExp
                for d in 0..<headDim {
                    var acc: Float = 0
                    for s in seqStart..<causalLen {
                        let vBase = kvHead * maxSeq * headDim + s * headDim
                        acc += scores[s] * invSum * Float(valCache[vBase + d])
                    }
                    let outIdx = pos * numQHeads * headDim + qHead * headDim + d
                    reference[outIdx] = acc
                }
            }
        }

        let qBuf = device.makeBuffer(bytes: qData, length: totalQ * 2, options: .storageModeShared)!
        let kCacheBuf = device.makeBuffer(bytes: keyCache, length: cacheElems * 2, options: .storageModeShared)!
        let vCacheBuf = device.makeBuffer(bytes: valCache, length: cacheElems * 2, options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: totalQ * 2, options: .storageModeShared)!
        memset(outputBuf.contents(), 0, totalQ * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(prefillPipeline!)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kCacheBuf, offset: 0, index: 1)
        enc.setBuffer(vCacheBuf, offset: 0, index: 2)
        enc.setBuffer(outputBuf, offset: 0, index: 3)
        var headDimVal = UInt32(headDim); enc.setBytes(&headDimVal, length: 4, index: 4)
        var seqLenVal = UInt32(batchSize); enc.setBytes(&seqLenVal, length: 4, index: 5)
        var startPosVal = UInt32(startPos); enc.setBytes(&startPosVal, length: 4, index: 6)
        var cacheSeqVal = UInt32(maxSeq); enc.setBytes(&cacheSeqVal, length: 4, index: 7)
        var kvHeadsVal = UInt32(numKVHeads); enc.setBytes(&kvHeadsVal, length: 4, index: 8)
        var scaleVal = scale; enc.setBytes(&scaleVal, length: 4, index: 9)
        var windowVal = UInt32(slidingWindow); enc.setBytes(&windowVal, length: 4, index: 10)
        enc.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: batchSize, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = outputBuf.contents().bindMemory(to: Float16.self, capacity: totalQ)
        var maxDiff: Float = 0
        for i in 0..<totalQ {
            maxDiff = max(maxDiff, abs(Float(outPtr[i]) - reference[i]))
        }

        fputs(
            "  attention_prefill sliding window vs cpu:"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.05)
    }

    func testAttentionPrefillSoftcap_SlidingWindow512MatchesCPUReference() throws {
        let prefillPipeline = makePipeline(
            shaderFile: "prefill_attention.metal", functionName: "attention_prefill_softcap"
        )
        try XCTSkipIf(prefillPipeline == nil, "Could not compile attention_prefill_softcap pipeline")

        let numQHeads = 8
        let numKVHeads = 1
        let headDim = 256
        let gqaRatio = numQHeads / numKVHeads
        let batchSize = 2
        let startPos = 600
        let slidingWindow = 512
        let scale: Float = 1.0 / sqrt(Float(headDim))
        let softcap: Float = 30.0

        let totalQ = batchSize * numQHeads * headDim
        let totalKV = batchSize * numKVHeads * headDim
        let prefixKV = startPos * numKVHeads * headDim
        var qData = [Float16](repeating: 0, count: totalQ)
        var kData = [Float16](repeating: 0, count: totalKV)
        var vData = [Float16](repeating: 0, count: totalKV)
        var prefixKData = [Float16](repeating: 0, count: prefixKV)
        var prefixVData = [Float16](repeating: 0, count: prefixKV)
        for i in 0..<totalQ { qData[i] = Float16(sin(Float(i) * 0.009) * 0.4) }
        for i in 0..<totalKV { kData[i] = Float16(cos(Float(i) * 0.013) * 0.45) }
        for i in 0..<totalKV { vData[i] = Float16(sin(Float(i) * 0.017) * 0.5) }
        for i in 0..<prefixKV { prefixKData[i] = Float16(cos(Float(i) * 0.004) * 0.35) }
        for i in 0..<prefixKV { prefixVData[i] = Float16(sin(Float(i) * 0.006) * 0.3) }

        let maxSeq = startPos + batchSize
        let cacheElems = numKVHeads * maxSeq * headDim
        var keyCache = [Float16](repeating: 0, count: cacheElems)
        var valCache = [Float16](repeating: 0, count: cacheElems)

        for pos in 0..<startPos {
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let cacheIdx = h * maxSeq * headDim + pos * headDim + d
                    let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                    keyCache[cacheIdx] = prefixKData[srcIdx]
                    valCache[cacheIdx] = prefixVData[srcIdx]
                }
            }
        }
        for pos in 0..<batchSize {
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let cacheIdx = h * maxSeq * headDim + (startPos + pos) * headDim + d
                    let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                    keyCache[cacheIdx] = kData[srcIdx]
                    valCache[cacheIdx] = vData[srcIdx]
                }
            }
        }

        var reference = [Float](repeating: 0, count: totalQ)
        for pos in 0..<batchSize {
            let causalLen = startPos + pos + 1
            let seqStart = max(0, causalLen - slidingWindow)
            for qHead in 0..<numQHeads {
                let kvHead = qHead / gqaRatio
                var scores = [Float](repeating: 0, count: causalLen)
                var maxScore = -Float.infinity
                for s in seqStart..<causalLen {
                    var dot: Float = 0
                    let qBase = pos * numQHeads * headDim + qHead * headDim
                    let kBase = kvHead * maxSeq * headDim + s * headDim
                    for d in 0..<headDim {
                        dot += Float(qData[qBase + d]) * Float(keyCache[kBase + d])
                    }
                    let score = softcap * tanh((dot * scale) / softcap)
                    scores[s] = score
                    maxScore = max(maxScore, score)
                }
                var sumExp: Float = 0
                for s in seqStart..<causalLen {
                    scores[s] = exp(scores[s] - maxScore)
                    sumExp += scores[s]
                }
                let invSum = 1.0 / sumExp
                for d in 0..<headDim {
                    var acc: Float = 0
                    for s in seqStart..<causalLen {
                        let vBase = kvHead * maxSeq * headDim + s * headDim
                        acc += scores[s] * invSum * Float(valCache[vBase + d])
                    }
                    let outIdx = pos * numQHeads * headDim + qHead * headDim + d
                    reference[outIdx] = acc
                }
            }
        }

        let qBuf = device.makeBuffer(bytes: qData, length: totalQ * 2, options: .storageModeShared)!
        let kCacheBuf = device.makeBuffer(bytes: keyCache, length: cacheElems * 2, options: .storageModeShared)!
        let vCacheBuf = device.makeBuffer(bytes: valCache, length: cacheElems * 2, options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: totalQ * 2, options: .storageModeShared)!
        memset(outputBuf.contents(), 0, totalQ * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(prefillPipeline!)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kCacheBuf, offset: 0, index: 1)
        enc.setBuffer(vCacheBuf, offset: 0, index: 2)
        enc.setBuffer(outputBuf, offset: 0, index: 3)
        var headDimVal = UInt32(headDim); enc.setBytes(&headDimVal, length: 4, index: 4)
        var seqLenVal = UInt32(batchSize); enc.setBytes(&seqLenVal, length: 4, index: 5)
        var startPosVal = UInt32(startPos); enc.setBytes(&startPosVal, length: 4, index: 6)
        var cacheSeqVal = UInt32(maxSeq); enc.setBytes(&cacheSeqVal, length: 4, index: 7)
        var kvHeadsVal = UInt32(numKVHeads); enc.setBytes(&kvHeadsVal, length: 4, index: 8)
        var scaleVal = scale; enc.setBytes(&scaleVal, length: 4, index: 9)
        var windowVal = UInt32(slidingWindow); enc.setBytes(&windowVal, length: 4, index: 10)
        var softcapVal = softcap; enc.setBytes(&softcapVal, length: 4, index: 11)
        enc.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: batchSize, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = outputBuf.contents().bindMemory(to: Float16.self, capacity: totalQ)
        var maxDiff: Float = 0
        for i in 0..<totalQ {
            maxDiff = max(maxDiff, abs(Float(outPtr[i]) - reference[i]))
        }

        fputs(
            "  attention_prefill_softcap sliding-512:"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.05)
    }

    func testAttentionPrefill_LongContextMatchesCPUReference() throws {
        let prefillPipeline = makePipeline(
            shaderFile: "prefill_attention.metal", functionName: "attention_prefill"
        )
        try XCTSkipIf(prefillPipeline == nil, "Could not compile attention_prefill pipeline")

        let numQHeads = 4
        let numKVHeads = 2
        let headDim = 32
        let gqaRatio = numQHeads / numKVHeads
        let batchSize = 4
        let startPos = 300
        let scale: Float = 1.0 / sqrt(Float(headDim))

        let totalQ = batchSize * numQHeads * headDim
        let totalKV = batchSize * numKVHeads * headDim
        let prefixKV = startPos * numKVHeads * headDim
        var qData = [Float16](repeating: 0, count: totalQ)
        var kData = [Float16](repeating: 0, count: totalKV)
        var vData = [Float16](repeating: 0, count: totalKV)
        var prefixKData = [Float16](repeating: 0, count: prefixKV)
        var prefixVData = [Float16](repeating: 0, count: prefixKV)
        for i in 0..<totalQ { qData[i] = Float16(sin(Float(i) * 0.013) * 0.4) }
        for i in 0..<totalKV { kData[i] = Float16(cos(Float(i) * 0.019) * 0.45) }
        for i in 0..<totalKV { vData[i] = Float16(sin(Float(i) * 0.023) * 0.5) }
        for i in 0..<prefixKV { prefixKData[i] = Float16(cos(Float(i) * 0.007) * 0.35) }
        for i in 0..<prefixKV { prefixVData[i] = Float16(sin(Float(i) * 0.009) * 0.3) }

        let maxSeq = startPos + batchSize
        let cacheElems = numKVHeads * maxSeq * headDim
        var keyCache = [Float16](repeating: 0, count: cacheElems)
        var valCache = [Float16](repeating: 0, count: cacheElems)

        for pos in 0..<startPos {
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let cacheIdx = h * maxSeq * headDim + pos * headDim + d
                    let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                    keyCache[cacheIdx] = prefixKData[srcIdx]
                    valCache[cacheIdx] = prefixVData[srcIdx]
                }
            }
        }
        for pos in 0..<batchSize {
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let cacheIdx = h * maxSeq * headDim + (startPos + pos) * headDim + d
                    let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                    keyCache[cacheIdx] = kData[srcIdx]
                    valCache[cacheIdx] = vData[srcIdx]
                }
            }
        }

        var reference = [Float](repeating: 0, count: totalQ)
        for pos in 0..<batchSize {
            let causalLen = startPos + pos + 1
            for qHead in 0..<numQHeads {
                let kvHead = qHead / gqaRatio
                var scores = [Float](repeating: 0, count: causalLen)
                var maxScore = -Float.infinity
                for s in 0..<causalLen {
                    var dot: Float = 0
                    let qBase = pos * numQHeads * headDim + qHead * headDim
                    let kBase = kvHead * maxSeq * headDim + s * headDim
                    for d in 0..<headDim {
                        dot += Float(qData[qBase + d]) * Float(keyCache[kBase + d])
                    }
                    scores[s] = dot * scale
                    maxScore = max(maxScore, scores[s])
                }
                var sumExp: Float = 0
                for s in 0..<causalLen {
                    scores[s] = exp(scores[s] - maxScore)
                    sumExp += scores[s]
                }
                let invSum = 1.0 / sumExp
                for d in 0..<headDim {
                    var acc: Float = 0
                    for s in 0..<causalLen {
                        let vBase = kvHead * maxSeq * headDim + s * headDim
                        acc += scores[s] * invSum * Float(valCache[vBase + d])
                    }
                    let outIdx = pos * numQHeads * headDim + qHead * headDim + d
                    reference[outIdx] = acc
                }
            }
        }

        let qBuf = device.makeBuffer(bytes: qData, length: totalQ * 2, options: .storageModeShared)!
        let kCacheBuf = device.makeBuffer(bytes: keyCache, length: cacheElems * 2, options: .storageModeShared)!
        let vCacheBuf = device.makeBuffer(bytes: valCache, length: cacheElems * 2, options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: totalQ * 2, options: .storageModeShared)!
        memset(outputBuf.contents(), 0, totalQ * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(prefillPipeline!)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kCacheBuf, offset: 0, index: 1)
        enc.setBuffer(vCacheBuf, offset: 0, index: 2)
        enc.setBuffer(outputBuf, offset: 0, index: 3)
        var headDimVal = UInt32(headDim); enc.setBytes(&headDimVal, length: 4, index: 4)
        var seqLenVal = UInt32(batchSize); enc.setBytes(&seqLenVal, length: 4, index: 5)
        var startPosVal = UInt32(startPos); enc.setBytes(&startPosVal, length: 4, index: 6)
        var cacheSeqVal = UInt32(maxSeq); enc.setBytes(&cacheSeqVal, length: 4, index: 7)
        var kvHeadsVal = UInt32(numKVHeads); enc.setBytes(&kvHeadsVal, length: 4, index: 8)
        var scaleVal = scale; enc.setBytes(&scaleVal, length: 4, index: 9)
        var windowVal: UInt32 = 0; enc.setBytes(&windowVal, length: 4, index: 10)
        enc.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: batchSize, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = outputBuf.contents().bindMemory(to: Float16.self, capacity: totalQ)
        var maxDiff: Float = 0
        for i in 0..<totalQ {
            maxDiff = max(maxDiff, abs(Float(outPtr[i]) - reference[i]))
        }

        fputs(
            "  attention_prefill long context vs cpu:"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.05)
    }

    /// B=1 must match single decode step.
    func testAttentionPrefill_BatchSize1() throws {
        let decodePipeline = makePipeline(
            shaderFile: "attention.metal", functionName: "attention_decode"
        )
        try XCTSkipIf(decodePipeline == nil, "Could not compile decode attention shader")

        let prefillPipeline = makePipeline(
            shaderFile: "prefill_attention.metal", functionName: "attention_prefill"
        )
        try XCTSkipIf(prefillPipeline == nil, "Could not compile attention_prefill pipeline")

        let numQHeads = 4
        let numKVHeads = 2
        let headDim = 32
        let startPos = 2
        let scale = 1.0 / sqrt(Float(headDim))

        var qData = [Float16](repeating: 0, count: numQHeads * headDim)
        var kData = [Float16](repeating: 0, count: numKVHeads * headDim)
        var vData = [Float16](repeating: 0, count: numKVHeads * headDim)
        for i in 0..<qData.count { qData[i] = Float16(sin(Float(i) * 0.1) * 0.5) }
        for i in 0..<kData.count { kData[i] = Float16(cos(Float(i) * 0.2) * 0.5) }
        for i in 0..<vData.count { vData[i] = Float16(sin(Float(i) * 0.3) * 0.5) }

        let totalSeq = startPos + 1
        let cacheElems = numKVHeads * totalSeq * headDim
        var prefixKData = [Float16](repeating: 0, count: startPos * numKVHeads * headDim)
        var prefixVData = [Float16](repeating: 0, count: startPos * numKVHeads * headDim)
        for i in 0..<prefixKData.count { prefixKData[i] = Float16(cos(Float(i) * 0.11) * 0.35) }
        for i in 0..<prefixVData.count { prefixVData[i] = Float16(sin(Float(i) * 0.13) * 0.3) }

        let kCacheBuf = device.makeBuffer(length: cacheElems * 2, options: .storageModeShared)!
        let vCacheBuf = device.makeBuffer(length: cacheElems * 2, options: .storageModeShared)!
        memset(kCacheBuf.contents(), 0, cacheElems * 2)
        memset(vCacheBuf.contents(), 0, cacheElems * 2)
        let kCachePtr = kCacheBuf.contents().bindMemory(to: Float16.self, capacity: cacheElems)
        let vCachePtr = vCacheBuf.contents().bindMemory(to: Float16.self, capacity: cacheElems)
        for pos in 0..<startPos {
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let cacheIdx = h * totalSeq * headDim + pos * headDim + d
                    let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                    kCachePtr[cacheIdx] = prefixKData[srcIdx]
                    vCachePtr[cacheIdx] = prefixVData[srcIdx]
                }
            }
        }
        for h in 0..<numKVHeads {
            for d in 0..<headDim {
                let cacheIdx = h * totalSeq * headDim + startPos * headDim + d
                let srcIdx = h * headDim + d
                kCachePtr[cacheIdx] = kData[srcIdx]
                vCachePtr[cacheIdx] = vData[srcIdx]
            }
        }

        let maskBuf = device.makeBuffer(length: totalSeq * 2, options: .storageModeShared)!
        let maskPtr = maskBuf.contents().bindMemory(to: Float16.self, capacity: totalSeq)
        for i in 0..<totalSeq {
            maskPtr[i] = i <= startPos ? Float16(0) : Float16(-10000)
        }

        let queryBuf = device.makeBuffer(bytes: qData, length: qData.count * 2, options: .storageModeShared)!
        let decodeOutBuf = device.makeBuffer(length: numQHeads * headDim * 2, options: .storageModeShared)!
        memset(decodeOutBuf.contents(), 0, numQHeads * headDim * 2)

        let cmdBuf1 = queue.makeCommandBuffer()!
        let enc1 = cmdBuf1.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(decodePipeline!)
        enc1.setBuffer(queryBuf, offset: 0, index: 0)
        enc1.setBuffer(kCacheBuf, offset: 0, index: 1)
        enc1.setBuffer(vCacheBuf, offset: 0, index: 2)
        enc1.setBuffer(maskBuf, offset: 0, index: 3)
        enc1.setBuffer(decodeOutBuf, offset: 0, index: 4)
        var hdVal = UInt32(headDim); enc1.setBytes(&hdVal, length: 4, index: 5)
        var msVal = UInt32(totalSeq); enc1.setBytes(&msVal, length: 4, index: 6)
        var slVal = UInt32(totalSeq); enc1.setBytes(&slVal, length: 4, index: 7)
        var kvhVal = UInt32(numKVHeads); enc1.setBytes(&kvhVal, length: 4, index: 8)
        var scaleVal = scale; enc1.setBytes(&scaleVal, length: 4, index: 9)
        var windowVal: UInt32 = 0; enc1.setBytes(&windowVal, length: 4, index: 10)
        enc1.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()

        var reference = [Float](repeating: 0, count: numQHeads * headDim)
        let refPtr = decodeOutBuf.contents().bindMemory(to: Float16.self, capacity: numQHeads * headDim)
        for i in 0..<reference.count { reference[i] = Float(refPtr[i]) }

        let qBuf = device.makeBuffer(bytes: qData, length: qData.count * 2, options: .storageModeShared)!
        let prefillOutBuf = device.makeBuffer(length: numQHeads * headDim * 2, options: .storageModeShared)!
        memset(prefillOutBuf.contents(), 0, numQHeads * headDim * 2)

        let cmdBuf2 = queue.makeCommandBuffer()!
        let enc2 = cmdBuf2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(prefillPipeline!)
        enc2.setBuffer(qBuf, offset: 0, index: 0)
        enc2.setBuffer(kCacheBuf, offset: 0, index: 1)
        enc2.setBuffer(vCacheBuf, offset: 0, index: 2)
        enc2.setBuffer(prefillOutBuf, offset: 0, index: 3)
        hdVal = UInt32(headDim); enc2.setBytes(&hdVal, length: 4, index: 4)
        slVal = UInt32(1); enc2.setBytes(&slVal, length: 4, index: 5)
        var startPosVal = UInt32(startPos); enc2.setBytes(&startPosVal, length: 4, index: 6)
        var cacheSeqVal = UInt32(totalSeq); enc2.setBytes(&cacheSeqVal, length: 4, index: 7)
        kvhVal = UInt32(numKVHeads); enc2.setBytes(&kvhVal, length: 4, index: 8)
        scaleVal = scale; enc2.setBytes(&scaleVal, length: 4, index: 9)
        windowVal = 0; enc2.setBytes(&windowVal, length: 4, index: 10)
        enc2.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc2.endEncoding()
        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()
        XCTAssertNil(cmdBuf2.error)

        let outPtr = prefillOutBuf.contents().bindMemory(to: Float16.self, capacity: numQHeads * headDim)
        var maxDiff: Float = 0
        for i in 0..<(numQHeads * headDim) {
            let diff = abs(Float(outPtr[i]) - reference[i])
            if diff > maxDiff { maxDiff = diff }
        }

        fputs("  attention_prefill startPos=\(startPos), B=1 vs decode: max diff = \(String(format: "%.6f", maxDiff))\n", stderr)
        XCTAssertLessThan(maxDiff, 0.001, "B=1 prefill should tightly match decode")
    }

    func testAttentionPrefill_LargeBatchBoundaries() throws {
        let decodePipeline = makePipeline(
            shaderFile: "attention.metal", functionName: "attention_decode"
        )
        try XCTSkipIf(decodePipeline == nil, "Could not compile decode attention shader")

        let prefillPipeline = makePipeline(
            shaderFile: "prefill_attention.metal", functionName: "attention_prefill"
        )
        try XCTSkipIf(prefillPipeline == nil, "Could not compile attention_prefill pipeline")

        let numQHeads = 4
        let numKVHeads = 2
        let headDim = 32
        let scale = Float(1.0 / sqrt(Float(headDim)))
        let startPos = 4

        for batchSize in [64, 65] {
            let totalQ = batchSize * numQHeads * headDim
            let totalKV = batchSize * numKVHeads * headDim
            let prefixKV = startPos * numKVHeads * headDim
            var qData = [Float16](repeating: 0, count: totalQ)
            var kData = [Float16](repeating: 0, count: totalKV)
            var vData = [Float16](repeating: 0, count: totalKV)
            var prefixKData = [Float16](repeating: 0, count: prefixKV)
            var prefixVData = [Float16](repeating: 0, count: prefixKV)
            for i in 0..<totalQ { qData[i] = Float16(sin(Float(i) * 0.011) * 0.45) }
            for i in 0..<totalKV { kData[i] = Float16(cos(Float(i) * 0.017) * 0.5) }
            for i in 0..<totalKV { vData[i] = Float16(sin(Float(i) * 0.023) * 0.4) }
            for i in 0..<prefixKV { prefixKData[i] = Float16(cos(Float(i) * 0.019) * 0.43) }
            for i in 0..<prefixKV { prefixVData[i] = Float16(sin(Float(i) * 0.029) * 0.38) }

            let maxSeq = startPos + batchSize
            let cacheSize = numKVHeads * maxSeq * headDim
            let kCacheBuf = device.makeBuffer(length: cacheSize * 2, options: .storageModeShared)!
            let vCacheBuf = device.makeBuffer(length: cacheSize * 2, options: .storageModeShared)!
            memset(kCacheBuf.contents(), 0, cacheSize * 2)
            memset(vCacheBuf.contents(), 0, cacheSize * 2)

            let maskBuf = device.makeBuffer(length: maxSeq * 2, options: .storageModeShared)!
            let queryBuf = device.makeBuffer(length: numQHeads * headDim * 2, options: .storageModeShared)!
            let decodeOutBuf = device.makeBuffer(length: numQHeads * headDim * 2, options: .storageModeShared)!

            var reference = [Float](repeating: 0, count: totalQ)
            let kCachePtr = kCacheBuf.contents().bindMemory(to: Float16.self, capacity: cacheSize)
            let vCachePtr = vCacheBuf.contents().bindMemory(to: Float16.self, capacity: cacheSize)

            for pos in 0..<startPos {
                for h in 0..<numKVHeads {
                    for d in 0..<headDim {
                        let cacheIdx = h * maxSeq * headDim + pos * headDim + d
                        let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                        kCachePtr[cacheIdx] = prefixKData[srcIdx]
                        vCachePtr[cacheIdx] = prefixVData[srcIdx]
                    }
                }
            }

            for pos in 0..<batchSize {
                for h in 0..<numKVHeads {
                    for d in 0..<headDim {
                        let cacheIdx = h * maxSeq * headDim + (startPos + pos) * headDim + d
                        let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                        kCachePtr[cacheIdx] = kData[srcIdx]
                        vCachePtr[cacheIdx] = vData[srcIdx]
                    }
                }

                let qPtr = queryBuf.contents().bindMemory(to: Float16.self, capacity: numQHeads * headDim)
                let qSrcBase = pos * numQHeads * headDim
                for i in 0..<(numQHeads * headDim) {
                    qPtr[i] = qData[qSrcBase + i]
                }

                let maskPtr = maskBuf.contents().bindMemory(to: Float16.self, capacity: maxSeq)
                for i in 0..<maxSeq {
                    maskPtr[i] = i <= startPos + pos ? Float16(0) : Float16(-10000)
                }

                memset(decodeOutBuf.contents(), 0, numQHeads * headDim * 2)

                let cmdBuf = queue.makeCommandBuffer()!
                let enc = cmdBuf.makeComputeCommandEncoder()!
                enc.setComputePipelineState(decodePipeline!)
                enc.setBuffer(queryBuf, offset: 0, index: 0)
                enc.setBuffer(kCacheBuf, offset: 0, index: 1)
                enc.setBuffer(vCacheBuf, offset: 0, index: 2)
                enc.setBuffer(maskBuf, offset: 0, index: 3)
                enc.setBuffer(decodeOutBuf, offset: 0, index: 4)
                var hdVal = UInt32(headDim); enc.setBytes(&hdVal, length: 4, index: 5)
                var msVal = UInt32(maxSeq); enc.setBytes(&msVal, length: 4, index: 6)
                var slVal = UInt32(startPos + pos + 1); enc.setBytes(&slVal, length: 4, index: 7)
                var kvhVal = UInt32(numKVHeads); enc.setBytes(&kvhVal, length: 4, index: 8)
                var scaleVal = scale; enc.setBytes(&scaleVal, length: 4, index: 9)
                var windowVal: UInt32 = 0; enc.setBytes(&windowVal, length: 4, index: 10)
                enc.dispatchThreadgroups(
                    MTLSize(width: numQHeads, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                )
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
                XCTAssertNil(cmdBuf.error)

                let outPtr = decodeOutBuf.contents().bindMemory(to: Float16.self, capacity: numQHeads * headDim)
                let refBase = pos * numQHeads * headDim
                for i in 0..<(numQHeads * headDim) {
                    reference[refBase + i] = Float(outPtr[i])
                }
            }

            let qBuf = device.makeBuffer(bytes: qData, length: totalQ * 2, options: .storageModeShared)!
            let prefillOutBuf = device.makeBuffer(length: totalQ * 2, options: .storageModeShared)!
            memset(prefillOutBuf.contents(), 0, totalQ * 2)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(prefillPipeline!)
            enc.setBuffer(qBuf, offset: 0, index: 0)
            enc.setBuffer(kCacheBuf, offset: 0, index: 1)
            enc.setBuffer(vCacheBuf, offset: 0, index: 2)
            enc.setBuffer(prefillOutBuf, offset: 0, index: 3)
            var hdVal = UInt32(headDim); enc.setBytes(&hdVal, length: 4, index: 4)
            var slVal = UInt32(batchSize); enc.setBytes(&slVal, length: 4, index: 5)
            var startPosVal = UInt32(startPos); enc.setBytes(&startPosVal, length: 4, index: 6)
            var cacheSeqVal = UInt32(maxSeq); enc.setBytes(&cacheSeqVal, length: 4, index: 7)
            var kvhVal = UInt32(numKVHeads); enc.setBytes(&kvhVal, length: 4, index: 8)
            var scaleVal = scale; enc.setBytes(&scaleVal, length: 4, index: 9)
            var windowVal: UInt32 = 0; enc.setBytes(&windowVal, length: 4, index: 10)
            enc.dispatchThreadgroups(
                MTLSize(width: numQHeads, height: batchSize, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let outPtr = prefillOutBuf.contents().bindMemory(to: Float16.self, capacity: totalQ)
            var maxDiff: Float = 0
            for i in 0..<totalQ {
                maxDiff = max(maxDiff, abs(Float(outPtr[i]) - reference[i]))
            }

            fputs(
                "  attention_prefill large batch startPos=\(startPos), B=\(batchSize):"
                    + " max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.05, "Prefill attention diverges at batch boundary \(batchSize)")
        }
    }

    func testRopeAndKvCachePrefill_LargeBatchBoundaries() throws {
        let pipeline = makePipeline(
            shaderFile: "prefill_rope_kv.metal", functionName: "rope_and_kv_cache_prefill"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile rope_and_kv_cache_prefill")

        let qHeads = 4
        let kvHeads = 2
        let headDim = 32
        let ropeDim = 16
        let cacheSeqCapacity = 128
        let startPos = 0

        func applyRopePairwise(_ data: inout [Float16], base: Int, cosRow: [Float16], sinRow: [Float16]) {
            for pair in 0..<(ropeDim / 2) {
                let d0 = pair * 2
                let d1 = d0 + 1
                let x0 = Float(data[base + d0])
                let x1 = Float(data[base + d1])
                let c0 = Float(cosRow[d0])
                let s0 = Float(sinRow[d0])
                let c1 = Float(cosRow[d1])
                let s1 = Float(sinRow[d1])
                data[base + d0] = Float16(x0 * c0 - x1 * s0)
                data[base + d1] = Float16(x1 * c1 + x0 * s1)
            }
        }

        var cosTable = [Float16](repeating: 0, count: cacheSeqCapacity * ropeDim)
        var sinTable = [Float16](repeating: 0, count: cacheSeqCapacity * ropeDim)
        for pos in 0..<cacheSeqCapacity {
            for d in 0..<ropeDim {
                cosTable[pos * ropeDim + d] = Float16(cos(Float(pos * ropeDim + d) * 0.013))
                sinTable[pos * ropeDim + d] = Float16(sin(Float(pos * ropeDim + d) * 0.017))
            }
        }

        for batchSize in [64, 65] {
            let qCount = batchSize * qHeads * headDim
            let kvCount = batchSize * kvHeads * headDim
            var queries = [Float16](repeating: 0, count: qCount)
            var keys = [Float16](repeating: 0, count: kvCount)
            var values = [Float16](repeating: 0, count: kvCount)
            for i in 0..<qCount { queries[i] = Float16(sin(Float(i) * 0.009) * 0.4) }
            for i in 0..<kvCount { keys[i] = Float16(cos(Float(i) * 0.015) * 0.45) }
            for i in 0..<kvCount { values[i] = Float16(sin(Float(i) * 0.021) * 0.35) }

            var refQueries = queries
            var refKeys = keys
            var refKeyCache = [Float16](repeating: 0, count: kvHeads * cacheSeqCapacity * headDim)
            var refValCache = [Float16](repeating: 0, count: kvHeads * cacheSeqCapacity * headDim)

            for pos in 0..<batchSize {
                let cosRow = Array(cosTable[(startPos + pos) * ropeDim..<((startPos + pos + 1) * ropeDim)])
                let sinRow = Array(sinTable[(startPos + pos) * ropeDim..<((startPos + pos + 1) * ropeDim)])

                for head in 0..<qHeads {
                    let qBase = pos * qHeads * headDim + head * headDim
                    applyRopePairwise(&refQueries, base: qBase, cosRow: cosRow, sinRow: sinRow)
                }

                for head in 0..<kvHeads {
                    let kBase = pos * kvHeads * headDim + head * headDim
                    applyRopePairwise(&refKeys, base: kBase, cosRow: cosRow, sinRow: sinRow)

                    let cacheBase = head * cacheSeqCapacity * headDim + (startPos + pos) * headDim
                    for d in 0..<headDim {
                        refKeyCache[cacheBase + d] = refKeys[kBase + d]
                        refValCache[cacheBase + d] = values[kBase + d]
                    }
                }
            }

            let qBuf = device.makeBuffer(bytes: queries, length: qCount * 2, options: .storageModeShared)!
            let kBuf = device.makeBuffer(bytes: keys, length: kvCount * 2, options: .storageModeShared)!
            let vBuf = device.makeBuffer(bytes: values, length: kvCount * 2, options: .storageModeShared)!
            let cosBuf = device.makeBuffer(bytes: cosTable, length: cosTable.count * 2, options: .storageModeShared)!
            let sinBuf = device.makeBuffer(bytes: sinTable, length: sinTable.count * 2, options: .storageModeShared)!
            let keyCacheBuf = device.makeBuffer(length: refKeyCache.count * 2, options: .storageModeShared)!
            let valCacheBuf = device.makeBuffer(length: refValCache.count * 2, options: .storageModeShared)!
            memset(keyCacheBuf.contents(), 0, refKeyCache.count * 2)
            memset(valCacheBuf.contents(), 0, refValCache.count * 2)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline!)
            enc.setBuffer(qBuf, offset: 0, index: 0)
            enc.setBuffer(kBuf, offset: 0, index: 1)
            enc.setBuffer(vBuf, offset: 0, index: 2)
            enc.setBuffer(cosBuf, offset: 0, index: 3)
            enc.setBuffer(sinBuf, offset: 0, index: 4)
            enc.setBuffer(keyCacheBuf, offset: 0, index: 5)
            enc.setBuffer(valCacheBuf, offset: 0, index: 6)
            var headDimVal = UInt32(headDim); enc.setBytes(&headDimVal, length: 4, index: 7)
            var ropeDimVal = UInt32(ropeDim); enc.setBytes(&ropeDimVal, length: 4, index: 8)
            var qHeadsVal = UInt32(qHeads); enc.setBytes(&qHeadsVal, length: 4, index: 9)
            var kvHeadsVal = UInt32(kvHeads); enc.setBytes(&kvHeadsVal, length: 4, index: 10)
            var seqLenVal = UInt32(batchSize); enc.setBytes(&seqLenVal, length: 4, index: 11)
            var startPosVal = UInt32(startPos); enc.setBytes(&startPosVal, length: 4, index: 12)
            var cacheSeqCapacityVal = UInt32(cacheSeqCapacity); enc.setBytes(&cacheSeqCapacityVal, length: 4, index: 13)
            var ropeLayoutVal = UInt32(0); enc.setBytes(&ropeLayoutVal, length: 4, index: 14)
            enc.dispatchThreadgroups(
                MTLSize(width: batchSize, height: max(qHeads, kvHeads), depth: 1),
                threadsPerThreadgroup: MTLSize(width: headDim, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let qPtr = qBuf.contents().bindMemory(to: Float16.self, capacity: qCount)
            let kPtr = kBuf.contents().bindMemory(to: Float16.self, capacity: kvCount)
            let keyCachePtr = keyCacheBuf.contents().bindMemory(to: Float16.self, capacity: refKeyCache.count)
            let valCachePtr = valCacheBuf.contents().bindMemory(to: Float16.self, capacity: refValCache.count)

            var maxDiff: Float = 0
            for i in 0..<qCount {
                maxDiff = max(maxDiff, abs(Float(qPtr[i]) - Float(refQueries[i])))
            }
            for i in 0..<kvCount {
                maxDiff = max(maxDiff, abs(Float(kPtr[i]) - Float(refKeys[i])))
            }
            for i in 0..<refKeyCache.count {
                maxDiff = max(maxDiff, abs(Float(keyCachePtr[i]) - Float(refKeyCache[i])))
                maxDiff = max(maxDiff, abs(Float(valCachePtr[i]) - Float(refValCache[i])))
            }

            fputs(
                "  rope_and_kv_cache_prefill batch=\(batchSize):"
                    + " max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.002, "RoPE/KV prefill diverges at batch boundary \(batchSize)")
        }
    }

    func testAnalyticRopeAndKvCachePrefillIsBitExactWithDecodeBrick() throws {
        let prefill = makePipeline(
            shaderFile: "prefill_rope_kv_precise.metal",
            functionName: "rope_and_kv_cache_prefill_analytic"
        )
        let decode = makePipeline(
            shaderFile: "attention.metal", functionName: "apply_rope"
        )
        try XCTSkipIf(prefill == nil || decode == nil, "Could not compile analytic RoPE kernels")

        let qHeads = 2
        let kvHeads = 1
        let headDim = 256
        let ropeDim = 256
        let batchSize = 3
        let cacheSeqCapacity = 8
        let ropeLayout = 1
        let baseLog2 = log2(Float(10_000_000))
        let qCount = batchSize * qHeads * headDim
        let kvCount = batchSize * kvHeads * headDim
        let queries = (0..<qCount).map {
            Float16(sin(Float($0 * 17 + 3)) * 1.25)
        }
        let keys = (0..<kvCount).map {
            Float16(cos(Float($0 * 13 + 5)) * 0.875)
        }
        let values = (0..<kvCount).map {
            Float16(sin(Float($0 * 7 + 11)) * 0.625)
        }

        let qActual = device.makeBuffer(
            bytes: queries, length: qCount * 2, options: .storageModeShared)!
        let kActual = device.makeBuffer(
            bytes: keys, length: kvCount * 2, options: .storageModeShared)!
        let vBuffer = device.makeBuffer(
            bytes: values, length: kvCount * 2, options: .storageModeShared)!
        let qReference = device.makeBuffer(
            bytes: queries, length: qCount * 2, options: .storageModeShared)!
        let kReference = device.makeBuffer(
            bytes: keys, length: kvCount * 2, options: .storageModeShared)!
        let dummyTable = device.makeBuffer(
            length: ropeDim * 2, options: .storageModeShared)!
        let cacheCount = kvHeads * cacheSeqCapacity * headDim
        let keyCache = device.makeBuffer(
            length: cacheCount * 2, options: .storageModeShared)!
        let valCache = device.makeBuffer(
            length: cacheCount * 2, options: .storageModeShared)!
        memset(keyCache.contents(), 0, cacheCount * 2)
        memset(valCache.contents(), 0, cacheCount * 2)

        try runOnGPU(queue: queue) { enc in
            enc.setComputePipelineState(decode!)
            for position in 0..<batchSize {
                var headDimValue = UInt32(headDim)
                var ropeDimValue = UInt32(ropeDim)
                var layoutValue = UInt32(ropeLayout)
                var positionValue = UInt32(position)
                var baseValue = baseLog2
                var mathMode: UInt32 = 1

                enc.setBuffer(
                    qReference,
                    offset: position * qHeads * headDim * 2,
                    index: 0
                )
                enc.setBuffer(dummyTable, offset: 0, index: 1)
                enc.setBuffer(dummyTable, offset: 0, index: 2)
                var headsValue = UInt32(qHeads)
                enc.setBytes(&headDimValue, length: 4, index: 3)
                enc.setBytes(&ropeDimValue, length: 4, index: 4)
                enc.setBytes(&headsValue, length: 4, index: 5)
                enc.setBytes(&layoutValue, length: 4, index: 6)
                enc.setBytes(&positionValue, length: 4, index: 7)
                enc.setBytes(&baseValue, length: 4, index: 8)
                enc.setBytes(&mathMode, length: 4, index: 9)
                enc.dispatchThreads(
                    MTLSize(width: qHeads * headDim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1)
                )

                enc.setBuffer(
                    kReference,
                    offset: position * kvHeads * headDim * 2,
                    index: 0
                )
                headsValue = UInt32(kvHeads)
                enc.setBytes(&headsValue, length: 4, index: 5)
                enc.dispatchThreads(
                    MTLSize(width: kvHeads * headDim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                )
            }
        }

        try runOnGPU(queue: queue) { enc in
            enc.setComputePipelineState(prefill!)
            enc.setBuffer(qActual, offset: 0, index: 0)
            enc.setBuffer(kActual, offset: 0, index: 1)
            enc.setBuffer(vBuffer, offset: 0, index: 2)
            enc.setBuffer(keyCache, offset: 0, index: 3)
            enc.setBuffer(valCache, offset: 0, index: 4)
            var packedDims = UInt32(headDim | (ropeDim << 16))
            var packedHeads = UInt32(qHeads | (kvHeads << 16))
            var seqLen = UInt32(batchSize)
            var startPos: UInt32 = 0
            var cacheCapacity = UInt32(cacheSeqCapacity)
            var layout = UInt32(ropeLayout)
            var base = baseLog2
            enc.setBytes(&packedDims, length: 4, index: 5)
            enc.setBytes(&packedHeads, length: 4, index: 6)
            enc.setBytes(&seqLen, length: 4, index: 7)
            enc.setBytes(&startPos, length: 4, index: 8)
            enc.setBytes(&cacheCapacity, length: 4, index: 9)
            enc.setBytes(&layout, length: 4, index: 10)
            enc.setBytes(&base, length: 4, index: 11)
            enc.dispatchThreadgroups(
                MTLSize(width: batchSize, height: qHeads, depth: 1),
                threadsPerThreadgroup: MTLSize(width: headDim, height: 1, depth: 1)
            )
        }

        let qGot = qActual.contents().bindMemory(to: Float16.self, capacity: qCount)
        let qWant = qReference.contents().bindMemory(to: Float16.self, capacity: qCount)
        XCTAssertEqual(
            (0..<qCount).map { qGot[$0].bitPattern },
            (0..<qCount).map { qWant[$0].bitPattern }
        )
        let kGot = kActual.contents().bindMemory(to: Float16.self, capacity: kvCount)
        let kWant = kReference.contents().bindMemory(to: Float16.self, capacity: kvCount)
        XCTAssertEqual(
            (0..<kvCount).map { kGot[$0].bitPattern },
            (0..<kvCount).map { kWant[$0].bitPattern }
        )
        let keyCacheValues = keyCache.contents().bindMemory(
            to: Float16.self, capacity: cacheCount)
        let valCacheValues = valCache.contents().bindMemory(
            to: Float16.self, capacity: cacheCount)
        for position in 0..<batchSize {
            for dim in 0..<headDim {
                let cacheIndex = position * headDim + dim
                let rowIndex = position * headDim + dim
                XCTAssertEqual(
                    keyCacheValues[cacheIndex].bitPattern,
                    kWant[rowIndex].bitPattern
                )
                XCTAssertEqual(
                    valCacheValues[cacheIndex].bitPattern,
                    values[rowIndex].bitPattern
                )
            }
        }
    }

    func testRopeAndKvCachePrefill_ProportionalSplitHalfLayout() throws {
        let pipeline = makePipeline(
            shaderFile: "prefill_rope_kv.metal", functionName: "rope_and_kv_cache_prefill"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile rope_and_kv_cache_prefill")

        let qHeads = 1
        let kvHeads = 1
        let headDim = 16
        let ropeDim = 4
        let cacheSeqCapacity = 1
        let batchSize = 1

        func applyProportionalRope(
            _ data: inout [Float16],
            base: Int,
            cosRow: [Float16],
            sinRow: [Float16]
        ) {
            for pair in 0..<(ropeDim / 2) {
                let d0 = pair
                let d1 = pair + headDim / 2
                let c0Index = pair
                let c1Index = pair + ropeDim / 2
                let x0 = Float(data[base + d0])
                let x1 = Float(data[base + d1])
                let c0 = Float(cosRow[c0Index])
                let s0 = Float(sinRow[c0Index])
                let c1 = Float(cosRow[c1Index])
                let s1 = Float(sinRow[c1Index])
                data[base + d0] = Float16(x0 * c0 - x1 * s0)
                data[base + d1] = Float16(x1 * c1 + x0 * s1)
            }
        }

        let queries = (0..<(qHeads * headDim)).map { Float16(Float($0 + 1) * 0.05) }
        let keys = (0..<(kvHeads * headDim)).map { Float16(Float($0 + 3) * -0.04) }
        let values = (0..<(kvHeads * headDim)).map { Float16(Float($0 + 5) * 0.03) }
        let cosTable = [Float16(0.8), Float16(0.7), Float16(0.8), Float16(0.7)]
        let sinTable = [Float16(0.6), Float16(0.5), Float16(0.6), Float16(0.5)]

        var refQueries = queries
        var refKeys = keys
        var refKeyCache = [Float16](repeating: 0, count: kvHeads * cacheSeqCapacity * headDim)
        var refValCache = [Float16](repeating: 0, count: kvHeads * cacheSeqCapacity * headDim)
        applyProportionalRope(&refQueries, base: 0, cosRow: cosTable, sinRow: sinTable)
        applyProportionalRope(&refKeys, base: 0, cosRow: cosTable, sinRow: sinTable)
        for d in 0..<headDim {
            refKeyCache[d] = refKeys[d]
            refValCache[d] = values[d]
        }

        let qBuf = device.makeBuffer(bytes: queries, length: queries.count * 2, options: .storageModeShared)!
        let kBuf = device.makeBuffer(bytes: keys, length: keys.count * 2, options: .storageModeShared)!
        let vBuf = device.makeBuffer(bytes: values, length: values.count * 2, options: .storageModeShared)!
        let cosBuf = device.makeBuffer(bytes: cosTable, length: cosTable.count * 2, options: .storageModeShared)!
        let sinBuf = device.makeBuffer(bytes: sinTable, length: sinTable.count * 2, options: .storageModeShared)!
        let keyCacheBuf = device.makeBuffer(length: refKeyCache.count * 2, options: .storageModeShared)!
        let valCacheBuf = device.makeBuffer(length: refValCache.count * 2, options: .storageModeShared)!
        memset(keyCacheBuf.contents(), 0, refKeyCache.count * 2)
        memset(valCacheBuf.contents(), 0, refValCache.count * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline!)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kBuf, offset: 0, index: 1)
        enc.setBuffer(vBuf, offset: 0, index: 2)
        enc.setBuffer(cosBuf, offset: 0, index: 3)
        enc.setBuffer(sinBuf, offset: 0, index: 4)
        enc.setBuffer(keyCacheBuf, offset: 0, index: 5)
        enc.setBuffer(valCacheBuf, offset: 0, index: 6)
        var headDimVal = UInt32(headDim); enc.setBytes(&headDimVal, length: 4, index: 7)
        var ropeDimVal = UInt32(ropeDim); enc.setBytes(&ropeDimVal, length: 4, index: 8)
        var qHeadsVal = UInt32(qHeads); enc.setBytes(&qHeadsVal, length: 4, index: 9)
        var kvHeadsVal = UInt32(kvHeads); enc.setBytes(&kvHeadsVal, length: 4, index: 10)
        var seqLenVal = UInt32(batchSize); enc.setBytes(&seqLenVal, length: 4, index: 11)
        var startPosVal = UInt32(0); enc.setBytes(&startPosVal, length: 4, index: 12)
        var cacheSeqVal = UInt32(cacheSeqCapacity); enc.setBytes(&cacheSeqVal, length: 4, index: 13)
        var ropeLayoutVal = UInt32(2); enc.setBytes(&ropeLayoutVal, length: 4, index: 14)
        enc.dispatchThreadgroups(
            MTLSize(width: batchSize, height: max(qHeads, kvHeads), depth: 1),
            threadsPerThreadgroup: MTLSize(width: headDim, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let qPtr = qBuf.contents().bindMemory(to: Float16.self, capacity: queries.count)
        let kPtr = kBuf.contents().bindMemory(to: Float16.self, capacity: keys.count)
        let keyCachePtr = keyCacheBuf.contents().bindMemory(to: Float16.self, capacity: refKeyCache.count)
        let valCachePtr = valCacheBuf.contents().bindMemory(to: Float16.self, capacity: refValCache.count)
        for i in 0..<queries.count {
            XCTAssertEqual(qPtr[i], refQueries[i])
        }
        for i in 0..<keys.count {
            XCTAssertEqual(kPtr[i], refKeys[i])
        }
        for i in 0..<refKeyCache.count {
            XCTAssertEqual(keyCachePtr[i], refKeyCache[i])
            XCTAssertEqual(valCachePtr[i], refValCache[i])
        }
    }

    func testAttentionPrefill_QwenShapeContinuationMatchesIncrementalDecode() throws {
        let prefillPipeline = makePipeline(
            shaderFile: "prefill_attention.metal", functionName: "attention_prefill"
        )
        let decodePipeline = makePipeline(
            shaderFile: "attention.metal", functionName: "attention_decode_d256_h8_kv2"
        )
        try XCTSkipIf(prefillPipeline == nil || decodePipeline == nil, "Could not compile Qwen attention kernels")

        let numQHeads = 8
        let numKVHeads = 2
        let headDim = 256
        let scale = Float(1.0 / sqrt(Float(headDim)))
        let startPos = 128
        let batchSize = 32
        let totalQ = batchSize * numQHeads * headDim
        let totalKV = batchSize * numKVHeads * headDim
        let prefixKV = startPos * numKVHeads * headDim

        var qData = [Float16](repeating: 0, count: totalQ)
        var kData = [Float16](repeating: 0, count: totalKV)
        var vData = [Float16](repeating: 0, count: totalKV)
        var prefixKData = [Float16](repeating: 0, count: prefixKV)
        var prefixVData = [Float16](repeating: 0, count: prefixKV)
        for i in 0..<totalQ { qData[i] = Float16(sin(Float(i) * 0.0017) * 0.45) }
        for i in 0..<totalKV { kData[i] = Float16(cos(Float(i) * 0.0021) * 0.5) }
        for i in 0..<totalKV { vData[i] = Float16(sin(Float(i) * 0.0029) * 0.4) }
        for i in 0..<prefixKV { prefixKData[i] = Float16(cos(Float(i) * 0.0019) * 0.43) }
        for i in 0..<prefixKV { prefixVData[i] = Float16(sin(Float(i) * 0.0027) * 0.38) }

        let maxSeq = startPos + batchSize
        let cacheSize = numKVHeads * maxSeq * headDim
        let kCacheBuf = device.makeBuffer(length: cacheSize * 2, options: .storageModeShared)!
        let vCacheBuf = device.makeBuffer(length: cacheSize * 2, options: .storageModeShared)!
        memset(kCacheBuf.contents(), 0, cacheSize * 2)
        memset(vCacheBuf.contents(), 0, cacheSize * 2)

        let queryBuf = device.makeBuffer(length: numQHeads * headDim * 2, options: .storageModeShared)!
        let decodeOutBuf = device.makeBuffer(length: numQHeads * headDim * 2, options: .storageModeShared)!

        var reference = [Float](repeating: 0, count: totalQ)
        let kCachePtr = kCacheBuf.contents().bindMemory(to: Float16.self, capacity: cacheSize)
        let vCachePtr = vCacheBuf.contents().bindMemory(to: Float16.self, capacity: cacheSize)

        for pos in 0..<startPos {
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let cacheIdx = h * maxSeq * headDim + pos * headDim + d
                    let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                    kCachePtr[cacheIdx] = prefixKData[srcIdx]
                    vCachePtr[cacheIdx] = prefixVData[srcIdx]
                }
            }
        }

        for pos in 0..<batchSize {
            for h in 0..<numKVHeads {
                for d in 0..<headDim {
                    let cacheIdx = h * maxSeq * headDim + (startPos + pos) * headDim + d
                    let srcIdx = pos * numKVHeads * headDim + h * headDim + d
                    kCachePtr[cacheIdx] = kData[srcIdx]
                    vCachePtr[cacheIdx] = vData[srcIdx]
                }
            }

            let qPtr = queryBuf.contents().bindMemory(to: Float16.self, capacity: numQHeads * headDim)
            let qSrcBase = pos * numQHeads * headDim
            for i in 0..<(numQHeads * headDim) {
                qPtr[i] = qData[qSrcBase + i]
            }

            memset(decodeOutBuf.contents(), 0, numQHeads * headDim * 2)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(decodePipeline!)
            enc.setBuffer(queryBuf, offset: 0, index: 0)
            enc.setBuffer(kCacheBuf, offset: 0, index: 1)
            enc.setBuffer(vCacheBuf, offset: 0, index: 2)
            enc.setBuffer(decodeOutBuf, offset: 0, index: 3)
            var hdVal = UInt32(headDim); enc.setBytes(&hdVal, length: 4, index: 4)
            var seqLenVal = UInt32(startPos + pos + 1); enc.setBytes(&seqLenVal, length: 4, index: 5)
            var kvhVal = UInt32(numKVHeads); enc.setBytes(&kvhVal, length: 4, index: 6)
            enc.dispatchThreadgroups(
                MTLSize(width: numQHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let outPtr = decodeOutBuf.contents().bindMemory(to: Float16.self, capacity: numQHeads * headDim)
            let refBase = pos * numQHeads * headDim
            for i in 0..<(numQHeads * headDim) {
                reference[refBase + i] = Float(outPtr[i])
            }
        }

        let qBuf = device.makeBuffer(bytes: qData, length: totalQ * 2, options: .storageModeShared)!
        let prefillOutBuf = device.makeBuffer(length: totalQ * 2, options: .storageModeShared)!
        memset(prefillOutBuf.contents(), 0, totalQ * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(prefillPipeline!)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kCacheBuf, offset: 0, index: 1)
        enc.setBuffer(vCacheBuf, offset: 0, index: 2)
        enc.setBuffer(prefillOutBuf, offset: 0, index: 3)
        var hdVal = UInt32(headDim); enc.setBytes(&hdVal, length: 4, index: 4)
        var chunkLenVal = UInt32(batchSize); enc.setBytes(&chunkLenVal, length: 4, index: 5)
        var startPosVal = UInt32(startPos); enc.setBytes(&startPosVal, length: 4, index: 6)
        var cacheSeqVal = UInt32(maxSeq); enc.setBytes(&cacheSeqVal, length: 4, index: 7)
        var kvhVal = UInt32(numKVHeads); enc.setBytes(&kvhVal, length: 4, index: 8)
        var scaleVal = scale; enc.setBytes(&scaleVal, length: 4, index: 9)
        var windowVal: UInt32 = 0; enc.setBytes(&windowVal, length: 4, index: 10)
        enc.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: batchSize, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = prefillOutBuf.contents().bindMemory(to: Float16.self, capacity: totalQ)
        var maxDiff: Float = 0
        for i in 0..<totalQ {
            maxDiff = max(maxDiff, abs(Float(outPtr[i]) - reference[i]))
        }

        fputs(
            "  attention_prefill qwen startPos=\(startPos), B=\(batchSize):"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.05, "Qwen-shape prefill attention diverges")
    }

    func testRopeAndKvCachePrefill_QwenShapeContinuationMatchesReference() throws {
        let pipeline = makePipeline(
            shaderFile: "prefill_rope_kv.metal", functionName: "rope_and_kv_cache_prefill"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile rope_and_kv_cache_prefill")

        let qHeads = 8
        let kvHeads = 2
        let headDim = 256
        let ropeDim = 64
        let cacheSeqCapacity = 256
        let startPos = 128
        let batchSize = 32

        func applyRopePairwise(_ data: inout [Float16], base: Int, cosRow: ArraySlice<Float16>, sinRow: ArraySlice<Float16>) {
            for pair in 0..<(ropeDim / 2) {
                let d0 = pair * 2
                let d1 = d0 + 1
                let x0 = Float(data[base + d0])
                let x1 = Float(data[base + d1])
                let c0 = Float(cosRow[cosRow.startIndex + d0])
                let s0 = Float(sinRow[sinRow.startIndex + d0])
                let c1 = Float(cosRow[cosRow.startIndex + d1])
                let s1 = Float(sinRow[sinRow.startIndex + d1])
                data[base + d0] = Float16(x0 * c0 - x1 * s0)
                data[base + d1] = Float16(x1 * c1 + x0 * s1)
            }
        }

        let qCount = batchSize * qHeads * headDim
        let kvCount = batchSize * kvHeads * headDim
        var queries = [Float16](repeating: 0, count: qCount)
        var keys = [Float16](repeating: 0, count: kvCount)
        var values = [Float16](repeating: 0, count: kvCount)
        for i in 0..<qCount { queries[i] = Float16(sin(Float(i) * 0.0019) * 0.4) }
        for i in 0..<kvCount { keys[i] = Float16(cos(Float(i) * 0.0023) * 0.45) }
        for i in 0..<kvCount { values[i] = Float16(sin(Float(i) * 0.0031) * 0.35) }

        var refQueries = queries
        var refKeys = keys
        var refKeyCache = [Float16](repeating: 0, count: kvHeads * cacheSeqCapacity * headDim)
        var refValCache = [Float16](repeating: 0, count: kvHeads * cacheSeqCapacity * headDim)

        var cosTable = [Float16](repeating: 0, count: cacheSeqCapacity * ropeDim)
        var sinTable = [Float16](repeating: 0, count: cacheSeqCapacity * ropeDim)
        for pos in 0..<cacheSeqCapacity {
            for d in 0..<ropeDim {
                cosTable[pos * ropeDim + d] = Float16(cos(Float(pos * ropeDim + d) * 0.0011))
                sinTable[pos * ropeDim + d] = Float16(sin(Float(pos * ropeDim + d) * 0.0013))
            }
        }

        for pos in 0..<batchSize {
            let cosRow = cosTable[(startPos + pos) * ropeDim..<((startPos + pos + 1) * ropeDim)]
            let sinRow = sinTable[(startPos + pos) * ropeDim..<((startPos + pos + 1) * ropeDim)]

            for head in 0..<qHeads {
                let qBase = pos * qHeads * headDim + head * headDim
                applyRopePairwise(&refQueries, base: qBase, cosRow: cosRow, sinRow: sinRow)
            }

            for head in 0..<kvHeads {
                let kBase = pos * kvHeads * headDim + head * headDim
                applyRopePairwise(&refKeys, base: kBase, cosRow: cosRow, sinRow: sinRow)

                let cacheBase = head * cacheSeqCapacity * headDim + (startPos + pos) * headDim
                for d in 0..<headDim {
                    refKeyCache[cacheBase + d] = refKeys[kBase + d]
                    refValCache[cacheBase + d] = values[kBase + d]
                }
            }
        }

        let qBuf = device.makeBuffer(bytes: queries, length: qCount * 2, options: .storageModeShared)!
        let kBuf = device.makeBuffer(bytes: keys, length: kvCount * 2, options: .storageModeShared)!
        let vBuf = device.makeBuffer(bytes: values, length: kvCount * 2, options: .storageModeShared)!
        let cosBuf = device.makeBuffer(bytes: cosTable, length: cosTable.count * 2, options: .storageModeShared)!
        let sinBuf = device.makeBuffer(bytes: sinTable, length: sinTable.count * 2, options: .storageModeShared)!
        let keyCacheBuf = device.makeBuffer(length: refKeyCache.count * 2, options: .storageModeShared)!
        let valCacheBuf = device.makeBuffer(length: refValCache.count * 2, options: .storageModeShared)!
        memset(keyCacheBuf.contents(), 0, refKeyCache.count * 2)
        memset(valCacheBuf.contents(), 0, refValCache.count * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline!)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kBuf, offset: 0, index: 1)
        enc.setBuffer(vBuf, offset: 0, index: 2)
        enc.setBuffer(cosBuf, offset: 0, index: 3)
        enc.setBuffer(sinBuf, offset: 0, index: 4)
        enc.setBuffer(keyCacheBuf, offset: 0, index: 5)
        enc.setBuffer(valCacheBuf, offset: 0, index: 6)
        var headDimVal = UInt32(headDim); enc.setBytes(&headDimVal, length: 4, index: 7)
        var ropeDimVal = UInt32(ropeDim); enc.setBytes(&ropeDimVal, length: 4, index: 8)
        var qHeadsVal = UInt32(qHeads); enc.setBytes(&qHeadsVal, length: 4, index: 9)
        var kvHeadsVal = UInt32(kvHeads); enc.setBytes(&kvHeadsVal, length: 4, index: 10)
        var seqLenVal = UInt32(batchSize); enc.setBytes(&seqLenVal, length: 4, index: 11)
        var startPosVal = UInt32(startPos); enc.setBytes(&startPosVal, length: 4, index: 12)
        var cacheSeqVal = UInt32(cacheSeqCapacity); enc.setBytes(&cacheSeqVal, length: 4, index: 13)
        var ropeLayoutVal = UInt32(0); enc.setBytes(&ropeLayoutVal, length: 4, index: 14)
        enc.dispatchThreadgroups(
            MTLSize(width: batchSize, height: max(qHeads, kvHeads), depth: 1),
            threadsPerThreadgroup: MTLSize(width: headDim, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let qPtr = qBuf.contents().bindMemory(to: Float16.self, capacity: qCount)
        let kPtr = kBuf.contents().bindMemory(to: Float16.self, capacity: kvCount)
        let keyCachePtr = keyCacheBuf.contents().bindMemory(to: Float16.self, capacity: refKeyCache.count)
        let valCachePtr = valCacheBuf.contents().bindMemory(to: Float16.self, capacity: refValCache.count)

        var maxDiff: Float = 0
        var queryDiff: Float = 0
        var keyDiff: Float = 0
        var keyCacheDiff: Float = 0
        var valCacheDiff: Float = 0
        for i in 0..<qCount {
            queryDiff = max(queryDiff, abs(Float(qPtr[i]) - Float(refQueries[i])))
        }
        for i in 0..<kvCount {
            keyDiff = max(keyDiff, abs(Float(kPtr[i]) - Float(refKeys[i])))
        }
        for i in 0..<refKeyCache.count {
            keyCacheDiff = max(keyCacheDiff, abs(Float(keyCachePtr[i]) - Float(refKeyCache[i])))
            valCacheDiff = max(valCacheDiff, abs(Float(valCachePtr[i]) - Float(refValCache[i])))
        }
        maxDiff = max(queryDiff, keyDiff, keyCacheDiff, valCacheDiff)

        fputs(
            "  rope_and_kv_cache_prefill qwen startPos=\(startPos), B=\(batchSize):"
                + " q=\(String(format: "%.6f", queryDiff))"
                + " k=\(String(format: "%.6f", keyDiff))"
                + " keyCache=\(String(format: "%.6f", keyCacheDiff))"
                + " valCache=\(String(format: "%.6f", valCacheDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.01, "Qwen-shape RoPE/KV prefill diverges")
    }

    func testDeltaNetRecurrenceMlxPrefill_LargeBatchBoundaries() throws {
        let pipeline = makePipeline(
            shaderFile: "prefill_recurrence.metal",
            functionName: "deltanet_recurrence_mlx_prefill_d128_h16"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile deltanet recurrence prefill kernel")

        let headDim = 128
        let numHeads = 16
        let hidden = headDim * numHeads
        let channels = hidden * 3
        let headScale = Float(1.0 / sqrt(Float(headDim)))

        func normalizeQK(_ qkv: inout [Float16], seqLen: Int) {
            for pos in 0..<seqLen {
                let base = pos * channels
                for head in 0..<numHeads {
                    let qBase = base + head * headDim
                    let kBase = base + hidden + head * headDim
                    var qNorm: Float = 0
                    var kNorm: Float = 0
                    for d in 0..<headDim {
                        let q = Float(qkv[qBase + d])
                        let k = Float(qkv[kBase + d])
                        qNorm += q * q
                        kNorm += k * k
                    }
                    // The explicit Q/K graph brick preserves L2-normalized K,
                    // while Q carries the attention head scale into the edge.
                    let qScale = headScale / max(sqrt(qNorm), 1e-6)
                    let kScale = 1.0 / max(sqrt(kNorm), 1e-6)
                    for d in 0..<headDim {
                        qkv[qBase + d] = Float16(Float(qkv[qBase + d]) * qScale)
                        qkv[kBase + d] = Float16(Float(qkv[kBase + d]) * kScale)
                    }
                }
            }
        }

        func softplus(_ x: Float) -> Float {
            x > 20 ? x : log1p(exp(x))
        }

        let extendedLargeBatchCoverage = ProcessInfo.processInfo.environment["SMELT_RUN_QWEN_SMOKE"] == "1"
        let seqLens = extendedLargeBatchCoverage ? [64, 65, 192, 255, 256] : [64, 65]

        for seqLen in seqLens {
            let stateCount = numHeads * headDim * headDim
            let qkvCount = seqLen * channels
            let projCount = seqLen * numHeads
            let outCount = seqLen * hidden

            var state = [Float16](repeating: 0, count: stateCount)
            var qkv = [Float16](repeating: 0, count: qkvCount)
            var bProj = [Float16](repeating: 0, count: projCount)
            var aProj = [Float16](repeating: 0, count: projCount)
            var aLog = [Float16](repeating: 0, count: numHeads)
            var dtBias = [Float16](repeating: 0, count: numHeads)

            for i in 0..<stateCount { state[i] = Float16(sin(Float(i) * 0.0007) * 0.03) }
            for i in 0..<qkvCount { qkv[i] = Float16(sin(Float(i) * 0.0019) * 0.4) }
            for i in 0..<projCount {
                bProj[i] = Float16(cos(Float(i) * 0.017) * 0.9)
                aProj[i] = Float16(sin(Float(i) * 0.013) * 0.7)
            }
            for i in 0..<numHeads {
                aLog[i] = Float16(-1.2 + Float(i) * 0.03)
                dtBias[i] = Float16(-0.2 + Float(i) * 0.02)
            }
            normalizeQK(&qkv, seqLen: seqLen)

            var refState = state.map(Float.init)
            let refQKV = qkv.map(Float.init)
            let refBProj = bProj.map(Float.init)
            let refAProj = aProj.map(Float.init)
            let refALog = aLog.map(Float.init)
            let refDtBias = dtBias.map(Float.init)
            var refOutput = [Float](repeating: 0, count: outCount)

            for pos in 0..<seqLen {
                let qkvBase = pos * channels
                let projBase = pos * numHeads
                for head in 0..<numHeads {
                    let qBase = qkvBase + head * headDim
                    let kBase = qkvBase + hidden + head * headDim
                    let vBase = qkvBase + 2 * hidden + head * headDim

                    let beta = 1.0 / (1.0 + exp(-refBProj[projBase + head]))
                    let decay = exp(-exp(refALog[head]) * softplus(refAProj[projBase + head] + refDtBias[head]))

                    for dv in 0..<headDim {
                        let stateBase = (head * headDim + dv) * headDim
                        var kvMem: Float = 0
                        for dk in 0..<headDim {
                            refState[stateBase + dk] *= decay
                            kvMem += refState[stateBase + dk] * refQKV[kBase + dk]
                        }
                        let delta = (refQKV[vBase + dv] - kvMem) * beta
                        var out: Float = 0
                        for dk in 0..<headDim {
                            refState[stateBase + dk] += refQKV[kBase + dk] * delta
                            out += refState[stateBase + dk] * refQKV[qBase + dk]
                        }
                        refOutput[pos * hidden + head * headDim + dv] = out
                    }
                }
            }

            let stateBuf = device.makeBuffer(bytes: state, length: stateCount * 2, options: .storageModeShared)!
            let qkvBuf = device.makeBuffer(bytes: qkv, length: qkvCount * 2, options: .storageModeShared)!
            let bProjBuf = device.makeBuffer(bytes: bProj, length: projCount * 2, options: .storageModeShared)!
            let aProjBuf = device.makeBuffer(bytes: aProj, length: projCount * 2, options: .storageModeShared)!
            let aLogBuf = device.makeBuffer(bytes: aLog, length: numHeads * 2, options: .storageModeShared)!
            let dtBiasBuf = device.makeBuffer(bytes: dtBias, length: numHeads * 2, options: .storageModeShared)!
            let outBuf = device.makeBuffer(length: outCount * 2, options: .storageModeShared)!
            memset(outBuf.contents(), 0, outCount * 2)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline!)
            enc.setBuffer(stateBuf, offset: 0, index: 0)
            enc.setBuffer(qkvBuf, offset: 0, index: 1)
            enc.setBuffer(bProjBuf, offset: 0, index: 2)
            enc.setBuffer(aProjBuf, offset: 0, index: 3)
            enc.setBuffer(aLogBuf, offset: 0, index: 4)
            enc.setBuffer(dtBiasBuf, offset: 0, index: 5)
            enc.setBuffer(outBuf, offset: 0, index: 6)
            var seqLenVal = UInt32(seqLen)
            enc.setBytes(&seqLenVal, length: 4, index: 7)
            enc.dispatchThreads(
                MTLSize(width: 32, height: headDim, depth: numHeads),
                threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let outPtr = outBuf.contents().bindMemory(to: Float16.self, capacity: outCount)
            let statePtr = stateBuf.contents().bindMemory(to: Float16.self, capacity: stateCount)
            var maxDiff: Float = 0
            for i in 0..<outCount {
                maxDiff = max(maxDiff, abs(Float(outPtr[i]) - refOutput[i]))
            }
            for i in 0..<stateCount {
                maxDiff = max(maxDiff, abs(Float(statePtr[i]) - refState[i]))
            }

            fputs(
                "  deltanet_recurrence_mlx_prefill_d128_h16 seqLen=\(seqLen):"
                    + " max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.03, "DeltaNet recurrence diverges at seqLen \(seqLen)")
        }
    }

    func testDeltaNetRecurrenceMlxDecode_GenericAndSpecializedMatchReference() throws {
        let genericPipeline = makePipeline(
            shaderFile: "recurrence.metal",
            functionName: "deltanet_recurrence_mlx_decode"
        )
        try XCTSkipIf(genericPipeline == nil, "Could not compile generic decode recurrence kernel")

        let specializedPipeline = makePipeline(
            shaderFile: "recurrence.metal",
            functionName: "deltanet_recurrence_mlx_decode_d128_h16"
        )
        try XCTSkipIf(
            specializedPipeline == nil,
            "Could not compile specialized decode recurrence kernel"
        )

        let headDim = 128
        let numHeads = 16
        let hidden = headDim * numHeads
        let channels = hidden * 3
        let headScale = Float(1.0 / sqrt(Float(headDim)))

        func normalizeQK(_ qkv: inout [Float16]) {
            for head in 0..<numHeads {
                let qBase = head * headDim
                let kBase = hidden + head * headDim
                var qNorm: Float = 0
                var kNorm: Float = 0
                for d in 0..<headDim {
                    let q = Float(qkv[qBase + d])
                    let k = Float(qkv[kBase + d])
                    qNorm += q * q
                    kNorm += k * k
                }
                // The recurrence consumes the already scaled Q graph edge.
                let qScale = headScale / max(sqrt(qNorm), 1e-6)
                let kScale = 1.0 / max(sqrt(kNorm), 1e-6)
                for d in 0..<headDim {
                    qkv[qBase + d] = Float16(Float(qkv[qBase + d]) * qScale)
                    qkv[kBase + d] = Float16(Float(qkv[kBase + d]) * kScale)
                }
            }
        }

        func softplus(_ x: Float) -> Float {
            x > 20 ? x : log1p(exp(x))
        }

        let stateCount = numHeads * headDim * headDim
        let projCount = numHeads
        let outCount = hidden

        var state = [Float16](repeating: 0, count: stateCount)
        var qkv = [Float16](repeating: 0, count: channels)
        var bProj = [Float16](repeating: 0, count: projCount)
        var aProj = [Float16](repeating: 0, count: projCount)
        var aLog = [Float16](repeating: 0, count: numHeads)
        var dtBias = [Float16](repeating: 0, count: numHeads)

        for i in 0..<stateCount { state[i] = Float16(sin(Float(i) * 0.0007) * 0.03) }
        for i in 0..<channels { qkv[i] = Float16(sin(Float(i) * 0.0019) * 0.4) }
        for i in 0..<projCount {
            bProj[i] = Float16(cos(Float(i) * 0.017) * 0.9)
            aProj[i] = Float16(sin(Float(i) * 0.013) * 0.7)
        }
        for i in 0..<numHeads {
            aLog[i] = Float16(-1.2 + Float(i) * 0.03)
            dtBias[i] = Float16(-0.2 + Float(i) * 0.02)
        }
        normalizeQK(&qkv)

        var refState = state.map(Float.init)
        let refQKV = qkv.map(Float.init)
        let refBProj = bProj.map(Float.init)
        let refAProj = aProj.map(Float.init)
        let refALog = aLog.map(Float.init)
        let refDtBias = dtBias.map(Float.init)
        var refOutput = [Float](repeating: 0, count: outCount)

        for head in 0..<numHeads {
            let qBase = head * headDim
            let kBase = hidden + head * headDim
            let vBase = 2 * hidden + head * headDim
            let beta = 1.0 / (1.0 + exp(-refBProj[head]))
            let decay = exp(-exp(refALog[head]) * softplus(refAProj[head] + refDtBias[head]))

            for dv in 0..<headDim {
                let stateBase = (head * headDim + dv) * headDim
                var kvMem: Float = 0
                for dk in 0..<headDim {
                    refState[stateBase + dk] *= decay
                    kvMem += refState[stateBase + dk] * refQKV[kBase + dk]
                }
                let delta = (refQKV[vBase + dv] - kvMem) * beta
                var out: Float = 0
                for dk in 0..<headDim {
                    refState[stateBase + dk] += refQKV[kBase + dk] * delta
                    out += refState[stateBase + dk] * refQKV[qBase + dk]
                }
                refOutput[head * headDim + dv] = out
            }
        }

        func runDecodeKernel(
            pipeline: MTLComputePipelineState,
            label: String,
            useGenericConstants: Bool
        ) throws -> Float {
            let stateBuf = device.makeBuffer(
                bytes: state,
                length: stateCount * 2,
                options: .storageModeShared
            )!
            let qkvBuf = device.makeBuffer(
                bytes: qkv,
                length: channels * 2,
                options: .storageModeShared
            )!
            let bProjBuf = device.makeBuffer(
                bytes: bProj,
                length: projCount * 2,
                options: .storageModeShared
            )!
            let aProjBuf = device.makeBuffer(
                bytes: aProj,
                length: projCount * 2,
                options: .storageModeShared
            )!
            let aLogBuf = device.makeBuffer(
                bytes: aLog,
                length: numHeads * 2,
                options: .storageModeShared
            )!
            let dtBiasBuf = device.makeBuffer(
                bytes: dtBias,
                length: numHeads * 2,
                options: .storageModeShared
            )!
            let outBuf = device.makeBuffer(length: outCount * 2, options: .storageModeShared)!
            memset(outBuf.contents(), 0, outCount * 2)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(stateBuf, offset: 0, index: 0)
            enc.setBuffer(qkvBuf, offset: 0, index: 1)
            enc.setBuffer(bProjBuf, offset: 0, index: 2)
            enc.setBuffer(aProjBuf, offset: 0, index: 3)
            enc.setBuffer(aLogBuf, offset: 0, index: 4)
            enc.setBuffer(dtBiasBuf, offset: 0, index: 5)
            enc.setBuffer(outBuf, offset: 0, index: 6)
            if useGenericConstants {
                var headDimVal = UInt32(headDim)
                var headScaleVal = headScale
                var numHeadsVal = UInt32(numHeads)
                var qkHeadsVal = UInt32(numHeads)
                enc.setBytes(&headDimVal, length: 4, index: 7)
                enc.setBytes(&headScaleVal, length: 4, index: 8)
                enc.setBytes(&numHeadsVal, length: 4, index: 9)
                enc.setBytes(&qkHeadsVal, length: 4, index: 10)
            }
            enc.dispatchThreads(
                MTLSize(width: 32, height: headDim, depth: numHeads),
                threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let outPtr = outBuf.contents().bindMemory(to: Float16.self, capacity: outCount)
            let statePtr = stateBuf.contents().bindMemory(to: Float16.self, capacity: stateCount)
            var maxDiff: Float = 0
            for i in 0..<outCount {
                maxDiff = max(maxDiff, abs(Float(outPtr[i]) - refOutput[i]))
            }
            for i in 0..<stateCount {
                maxDiff = max(maxDiff, abs(Float(statePtr[i]) - refState[i]))
            }

            fputs(
                "  \(label) max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            return maxDiff
        }

        let genericDiff = try runDecodeKernel(
            pipeline: genericPipeline!,
            label: "deltanet_recurrence_mlx_decode",
            useGenericConstants: true
        )
        let specializedDiff = try runDecodeKernel(
            pipeline: specializedPipeline!,
            label: "deltanet_recurrence_mlx_decode_d128_h16",
            useGenericConstants: false
        )

        XCTAssertLessThan(genericDiff, 0.06, "Generic decode recurrence diverges from reference")
        XCTAssertLessThan(
            specializedDiff,
            0.03,
            "Specialized decode recurrence diverges from reference"
        )
    }

    func testDeltaNetRecurrenceMlxDecode_D128H48QK16IsBitExactToGeneric() throws {
        let names = [
            "deltanet_recurrence_mlx_decode",
            "deltanet_recurrence_mlx_decode_d128_h48_qk16",
        ]
        let pipelines = makePipelines(shaderFile: "recurrence.metal", functionNames: names)
        try XCTSkipIf(pipelines == nil, "Could not compile decode recurrence kernels")

        let headDim = 128
        let valueHeads = 48
        let qkHeads = 16
        let qkvCount = (2 * qkHeads + valueHeads) * headDim
        let stateCount = valueHeads * headDim * headDim
        let outputCount = valueHeads * headDim

        var state = [Float16](repeating: 0, count: stateCount)
        var qkv = [Float16](repeating: 0, count: qkvCount)
        var bProj = [Float16](repeating: 0, count: valueHeads)
        var aProj = [Float16](repeating: 0, count: valueHeads)
        var aLog = [Float16](repeating: 0, count: valueHeads)
        var dtBias = [Float16](repeating: 0, count: valueHeads)
        for index in state.indices {
            state[index] = Float16(sin(Float(index) * 0.000_31) * 0.04)
        }
        for index in qkv.indices {
            qkv[index] = Float16(cos(Float(index) * 0.001_7) * 0.35)
        }
        for head in 0..<valueHeads {
            bProj[head] = Float16(sin(Float(head) * 0.17) * 0.8)
            aProj[head] = Float16(cos(Float(head) * 0.11) * 0.6)
            aLog[head] = Float16(-1.4 + Float(head) * 0.009)
            dtBias[head] = Float16(-0.3 + Float(head) * 0.007)
        }

        func run(_ pipeline: MTLComputePipelineState, generic: Bool) throws
            -> (state: MTLBuffer, output: MTLBuffer)
        {
            let stateBuffer = device.makeBuffer(
                bytes: state, length: stateCount * 2, options: .storageModeShared)!
            let qkvBuffer = device.makeBuffer(
                bytes: qkv, length: qkvCount * 2, options: .storageModeShared)!
            let bBuffer = device.makeBuffer(
                bytes: bProj, length: valueHeads * 2, options: .storageModeShared)!
            let aBuffer = device.makeBuffer(
                bytes: aProj, length: valueHeads * 2, options: .storageModeShared)!
            let aLogBuffer = device.makeBuffer(
                bytes: aLog, length: valueHeads * 2, options: .storageModeShared)!
            let dtBiasBuffer = device.makeBuffer(
                bytes: dtBias, length: valueHeads * 2, options: .storageModeShared)!
            let outputBuffer = device.makeBuffer(
                length: outputCount * 2, options: .storageModeShared)!
            memset(outputBuffer.contents(), 0, outputCount * 2)

            let commandBuffer = queue.makeCommandBuffer()!
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(pipeline)
            for (index, buffer) in [
                stateBuffer, qkvBuffer, bBuffer, aBuffer, aLogBuffer,
                dtBiasBuffer, outputBuffer,
            ].enumerated() {
                encoder.setBuffer(buffer, offset: 0, index: index)
            }
            if generic {
                var headDimValue = UInt32(headDim)
                var headScale = Float(1.0 / sqrt(Float(headDim)))
                var valueHeadsValue = UInt32(valueHeads)
                var qkHeadsValue = UInt32(qkHeads)
                encoder.setBytes(&headDimValue, length: 4, index: 7)
                encoder.setBytes(&headScale, length: 4, index: 8)
                encoder.setBytes(&valueHeadsValue, length: 4, index: 9)
                encoder.setBytes(&qkHeadsValue, length: 4, index: 10)
            }
            encoder.dispatchThreads(
                MTLSize(width: 32, height: headDim, depth: valueHeads),
                threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
            )
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error { throw error }
            return (stateBuffer, outputBuffer)
        }

        let generic = try run(pipelines![names[0]]!, generic: true)
        let specialized = try run(pipelines![names[1]]!, generic: false)
        XCTAssertNil(
            firstHalfBitMismatch(generic.output, specialized.output, count: outputCount),
            "Specialized recurrence output must be bit-exact"
        )
        XCTAssertNil(
            firstHalfBitMismatch(generic.state, specialized.state, count: stateCount),
            "Specialized recurrence state must be bit-exact"
        )
    }

    func testDeltaNetRecurrenceMlxDecode_SequentialReplayMatchesReference() throws {
        let genericPipeline = makePipeline(
            shaderFile: "recurrence.metal",
            functionName: "deltanet_recurrence_mlx_decode"
        )
        try XCTSkipIf(genericPipeline == nil, "Could not compile generic decode recurrence kernel")

        let specializedPipeline = makePipeline(
            shaderFile: "recurrence.metal",
            functionName: "deltanet_recurrence_mlx_decode_d128_h16"
        )
        try XCTSkipIf(
            specializedPipeline == nil,
            "Could not compile specialized decode recurrence kernel"
        )

        let headDim = 128
        let numHeads = 16
        let hidden = headDim * numHeads
        let channels = hidden * 3
        let seqLen = 64
        let headScale = Float(1.0 / sqrt(Float(headDim)))

        func normalizeQK(_ qkv: inout [Float16], seqLen: Int) {
            for pos in 0..<seqLen {
                let base = pos * channels
                for head in 0..<numHeads {
                    let qBase = base + head * headDim
                    let kBase = base + hidden + head * headDim
                    var qNorm: Float = 0
                    var kNorm: Float = 0
                    for d in 0..<headDim {
                        let q = Float(qkv[qBase + d])
                        let k = Float(qkv[kBase + d])
                        qNorm += q * q
                        kNorm += k * k
                    }
                    // The recurrence consumes the already scaled Q graph edge.
                    let qScale = headScale / max(sqrt(qNorm), 1e-6)
                    let kScale = 1.0 / max(sqrt(kNorm), 1e-6)
                    for d in 0..<headDim {
                        qkv[qBase + d] = Float16(Float(qkv[qBase + d]) * qScale)
                        qkv[kBase + d] = Float16(Float(qkv[kBase + d]) * kScale)
                    }
                }
            }
        }

        func softplus(_ x: Float) -> Float {
            x > 20 ? x : log1p(exp(x))
        }

        let stateCount = numHeads * headDim * headDim
        let qkvCount = seqLen * channels
        let projCount = seqLen * numHeads
        let outCount = seqLen * hidden

        var state = [Float16](repeating: 0, count: stateCount)
        var qkv = [Float16](repeating: 0, count: qkvCount)
        var bProj = [Float16](repeating: 0, count: projCount)
        var aProj = [Float16](repeating: 0, count: projCount)
        var aLog = [Float16](repeating: 0, count: numHeads)
        var dtBias = [Float16](repeating: 0, count: numHeads)

        for i in 0..<stateCount { state[i] = Float16(sin(Float(i) * 0.0007) * 0.03) }
        for i in 0..<qkvCount { qkv[i] = Float16(sin(Float(i) * 0.0019) * 0.4) }
        for i in 0..<projCount {
            bProj[i] = Float16(cos(Float(i) * 0.017) * 0.9)
            aProj[i] = Float16(sin(Float(i) * 0.013) * 0.7)
        }
        for i in 0..<numHeads {
            aLog[i] = Float16(-1.2 + Float(i) * 0.03)
            dtBias[i] = Float16(-0.2 + Float(i) * 0.02)
        }
        normalizeQK(&qkv, seqLen: seqLen)

        var refState = state.map(Float.init)
        let refQKV = qkv.map(Float.init)
        let refBProj = bProj.map(Float.init)
        let refAProj = aProj.map(Float.init)
        let refALog = aLog.map(Float.init)
        let refDtBias = dtBias.map(Float.init)
        var refOutput = [Float](repeating: 0, count: outCount)

        for pos in 0..<seqLen {
            let qkvBase = pos * channels
            let projBase = pos * numHeads
            for head in 0..<numHeads {
                let qBase = qkvBase + head * headDim
                let kBase = qkvBase + hidden + head * headDim
                let vBase = qkvBase + 2 * hidden + head * headDim
                let beta = 1.0 / (1.0 + exp(-refBProj[projBase + head]))
                let decay = exp(
                    -exp(refALog[head])
                        * softplus(refAProj[projBase + head] + refDtBias[head])
                )

                for dv in 0..<headDim {
                    let stateBase = (head * headDim + dv) * headDim
                    var kvMem: Float = 0
                    for dk in 0..<headDim {
                        refState[stateBase + dk] *= decay
                        kvMem += refState[stateBase + dk] * refQKV[kBase + dk]
                    }
                    let delta = (refQKV[vBase + dv] - kvMem) * beta
                    var out: Float = 0
                    for dk in 0..<headDim {
                        refState[stateBase + dk] += refQKV[kBase + dk] * delta
                        out += refState[stateBase + dk] * refQKV[qBase + dk]
                    }
                    refOutput[pos * hidden + head * headDim + dv] = out
                }
            }
        }

        func runSequentialDecodeReplay(
            pipeline: MTLComputePipelineState,
            label: String,
            useGenericConstants: Bool
        ) throws -> Float {
            let stateBuf = device.makeBuffer(
                bytes: state,
                length: stateCount * 2,
                options: .storageModeShared
            )!
            let qkvBuf = device.makeBuffer(length: channels * 2, options: .storageModeShared)!
            let bProjBuf = device.makeBuffer(length: numHeads * 2, options: .storageModeShared)!
            let aProjBuf = device.makeBuffer(length: numHeads * 2, options: .storageModeShared)!
            let aLogBuf = device.makeBuffer(
                bytes: aLog,
                length: numHeads * 2,
                options: .storageModeShared
            )!
            let dtBiasBuf = device.makeBuffer(
                bytes: dtBias,
                length: numHeads * 2,
                options: .storageModeShared
            )!
            let outBuf = device.makeBuffer(length: hidden * 2, options: .storageModeShared)!

            var replayOutput = [Float](repeating: 0, count: outCount)
            for pos in 0..<seqLen {
                let qkvBase = pos * channels
                let projBase = pos * numHeads
                let qkvSlice = Array(qkv[qkvBase..<(qkvBase + channels)])
                let bSlice = Array(bProj[projBase..<(projBase + numHeads)])
                let aSlice = Array(aProj[projBase..<(projBase + numHeads)])
                memcpy(qkvBuf.contents(), qkvSlice, channels * 2)
                memcpy(bProjBuf.contents(), bSlice, numHeads * 2)
                memcpy(aProjBuf.contents(), aSlice, numHeads * 2)
                memset(outBuf.contents(), 0, hidden * 2)

                let cmdBuf = queue.makeCommandBuffer()!
                let enc = cmdBuf.makeComputeCommandEncoder()!
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(stateBuf, offset: 0, index: 0)
                enc.setBuffer(qkvBuf, offset: 0, index: 1)
                enc.setBuffer(bProjBuf, offset: 0, index: 2)
                enc.setBuffer(aProjBuf, offset: 0, index: 3)
                enc.setBuffer(aLogBuf, offset: 0, index: 4)
                enc.setBuffer(dtBiasBuf, offset: 0, index: 5)
                enc.setBuffer(outBuf, offset: 0, index: 6)
                if useGenericConstants {
                    var headDimVal = UInt32(headDim)
                    var headScaleVal = headScale
                    var numHeadsVal = UInt32(numHeads)
                    var qkHeadsVal = UInt32(numHeads)
                    enc.setBytes(&headDimVal, length: 4, index: 7)
                    enc.setBytes(&headScaleVal, length: 4, index: 8)
                    enc.setBytes(&numHeadsVal, length: 4, index: 9)
                    enc.setBytes(&qkHeadsVal, length: 4, index: 10)
                }
                enc.dispatchThreads(
                    MTLSize(width: 32, height: headDim, depth: numHeads),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
                )
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
                XCTAssertNil(cmdBuf.error)

                let outPtr = outBuf.contents().bindMemory(to: Float16.self, capacity: hidden)
                for i in 0..<hidden {
                    replayOutput[pos * hidden + i] = Float(outPtr[i])
                }
            }

            let statePtr = stateBuf.contents().bindMemory(to: Float16.self, capacity: stateCount)
            var maxDiff: Float = 0
            for i in 0..<outCount {
                maxDiff = max(maxDiff, abs(replayOutput[i] - refOutput[i]))
            }
            for i in 0..<stateCount {
                maxDiff = max(maxDiff, abs(Float(statePtr[i]) - refState[i]))
            }

            fputs(
                "  \(label) sequential replay seqLen=\(seqLen):"
                    + " max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            return maxDiff
        }

        let genericDiff = try runSequentialDecodeReplay(
            pipeline: genericPipeline!,
            label: "deltanet_recurrence_mlx_decode",
            useGenericConstants: true
        )
        XCTAssertLessThan(
            genericDiff,
            0.05,
            "Generic decode recurrence diverges during sequential replay"
        )

        let specializedDiff = try runSequentialDecodeReplay(
            pipeline: specializedPipeline!,
            label: "deltanet_recurrence_mlx_decode_d128_h16",
            useGenericConstants: false
        )
        XCTAssertLessThan(
            specializedDiff,
            0.05,
            "Specialized decode recurrence diverges during sequential replay"
        )
    }

    func testDeltaNetRecurrenceMlxPrefillD128H48QK16IsBitExactToGeneric() throws {
        let names = [
            "deltanet_recurrence_mlx_prefill",
            "deltanet_recurrence_mlx_prefill_d128_h48_qk16",
        ]
        let pipelines = makePipelines(
            shaderFile: "prefill_recurrence.metal",
            functionNames: names
        )
        try XCTSkipIf(pipelines == nil, "Could not compile prefill recurrence kernels")

        let headDim = 128
        let valueHeads = 48
        let qkHeads = 16
        let channels = (2 * qkHeads + valueHeads) * headDim
        let stateCount = valueHeads * headDim * headDim

        for seqLen in [1, 65] {
            let qkvCount = seqLen * channels
            let projCount = seqLen * valueHeads
            let outputCount = seqLen * valueHeads * headDim
            let state = (0..<stateCount).map {
                Float16(sin(Float($0) * 0.000_31) * 0.04)
            }
            let qkv = (0..<qkvCount).map {
                Float16(cos(Float($0) * 0.001_7) * 0.35)
            }
            let bProj = (0..<projCount).map {
                Float16(sin(Float($0) * 0.017) * 0.8)
            }
            let aProj = (0..<projCount).map {
                Float16(cos(Float($0) * 0.011) * 0.6)
            }
            let aLog = (0..<valueHeads).map {
                Float16(-1.4 + Float($0) * 0.009)
            }
            let dtBias = (0..<valueHeads).map {
                Float16(-0.3 + Float($0) * 0.007)
            }

            func run(_ pipeline: MTLComputePipelineState, generic: Bool) throws
                -> (state: MTLBuffer, output: MTLBuffer)
            {
                let buffers = [
                    device.makeBuffer(bytes: state, length: stateCount * 2, options: .storageModeShared)!,
                    device.makeBuffer(bytes: qkv, length: qkvCount * 2, options: .storageModeShared)!,
                    device.makeBuffer(bytes: bProj, length: projCount * 2, options: .storageModeShared)!,
                    device.makeBuffer(bytes: aProj, length: projCount * 2, options: .storageModeShared)!,
                    device.makeBuffer(bytes: aLog, length: valueHeads * 2, options: .storageModeShared)!,
                    device.makeBuffer(bytes: dtBias, length: valueHeads * 2, options: .storageModeShared)!,
                    device.makeBuffer(length: outputCount * 2, options: .storageModeShared)!,
                ]
                memset(buffers[6].contents(), 0, outputCount * 2)

                let commandBuffer = queue.makeCommandBuffer()!
                let encoder = commandBuffer.makeComputeCommandEncoder()!
                encoder.setComputePipelineState(pipeline)
                for (index, buffer) in buffers.enumerated() {
                    encoder.setBuffer(buffer, offset: 0, index: index)
                }
                var seqLenValue = UInt32(seqLen)
                if generic {
                    var headDimValue = UInt32(headDim)
                    var valueHeadsValue = UInt32(valueHeads)
                    var qkHeadsValue = UInt32(qkHeads)
                    encoder.setBytes(&headDimValue, length: 4, index: 7)
                    encoder.setBytes(&valueHeadsValue, length: 4, index: 8)
                    encoder.setBytes(&qkHeadsValue, length: 4, index: 9)
                    encoder.setBytes(&seqLenValue, length: 4, index: 10)
                } else {
                    encoder.setBytes(&seqLenValue, length: 4, index: 7)
                }
                encoder.dispatchThreads(
                    MTLSize(width: 32, height: headDim, depth: valueHeads),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
                )
                encoder.endEncoding()
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                if let error = commandBuffer.error { throw error }
                return (buffers[0], buffers[6])
            }

            let generic = try run(pipelines![names[0]]!, generic: true)
            let specialized = try run(pipelines![names[1]]!, generic: false)
            XCTAssertNil(
                firstHalfBitMismatch(generic.output, specialized.output, count: outputCount),
                "H48 prefill output must be bit-exact at seqLen \(seqLen)"
            )
            XCTAssertNil(
                firstHalfBitMismatch(generic.state, specialized.state, count: stateCount),
                "H48 prefill state must be bit-exact at seqLen \(seqLen)"
            )
        }
    }

    func testDeltaNetGenericRecurrenceSingleTokenPrefillIsBitExactWithDecode() throws {
        let prefillPipeline = makePipeline(
            shaderFile: "prefill_recurrence.metal",
            functionName: "deltanet_recurrence_mlx_prefill"
        )
        try XCTSkipIf(prefillPipeline == nil, "Could not compile prefill recurrence kernel")

        let decodePipeline = makePipeline(
            shaderFile: "recurrence.metal",
            functionName: "deltanet_recurrence_mlx_decode"
        )
        try XCTSkipIf(
            decodePipeline == nil,
            "Could not compile specialized decode recurrence kernel"
        )

        let headDim = 128
        let numHeads = 3
        let qkHeads = 1
        let hidden = headDim * numHeads
        let channels = (2 * qkHeads + numHeads) * headDim

        func normalizeQK(_ qkv: inout [Float16], seqLen: Int) {
            for pos in 0..<seqLen {
                let base = pos * channels
                for head in 0..<qkHeads {
                    let qBase = base + head * headDim
                    let kBase = base + qkHeads * headDim + head * headDim
                    var qNorm: Float = 0
                    var kNorm: Float = 0
                    for d in 0..<headDim {
                        let q = Float(qkv[qBase + d])
                        let k = Float(qkv[kBase + d])
                        qNorm += q * q
                        kNorm += k * k
                    }
                    let qScale = 1.0 / max(sqrt(qNorm), 1e-6)
                    let kScale = 1.0 / max(sqrt(kNorm), 1e-6)
                    for d in 0..<headDim {
                        qkv[qBase + d] = Float16(Float(qkv[qBase + d]) * qScale)
                        qkv[kBase + d] = Float16(Float(qkv[kBase + d]) * kScale)
                    }
                }
            }
        }

        // A one-token prefill invocation and decode invocation have the same
        // persistent FP16 entry/exit boundary. Multi-token prefill does not:
        // MLX intentionally carries recurrent state in FP32 registers across
        // the chunk and rounds it only on the final state write.
        for seqLen in [1] {
            let stateCount = numHeads * headDim * headDim
            let qkvCount = seqLen * channels
            let projCount = seqLen * numHeads
            let outCount = seqLen * hidden

            var state = [Float16](repeating: 0, count: stateCount)
            var qkv = [Float16](repeating: 0, count: qkvCount)
            var bProj = [Float16](repeating: 0, count: projCount)
            var aProj = [Float16](repeating: 0, count: projCount)
            var aLog = [Float16](repeating: 0, count: numHeads)
            var dtBias = [Float16](repeating: 0, count: numHeads)

            for i in 0..<stateCount { state[i] = Float16(sin(Float(i) * 0.0007) * 0.03) }
            for i in 0..<qkvCount { qkv[i] = Float16(sin(Float(i) * 0.0019) * 0.4) }
            for i in 0..<projCount {
                bProj[i] = Float16(cos(Float(i) * 0.017) * 0.9)
                aProj[i] = Float16(sin(Float(i) * 0.013) * 0.7)
            }
            for i in 0..<numHeads {
                aLog[i] = Float16(-1.2 + Float(i) * 0.03)
                dtBias[i] = Float16(-0.2 + Float(i) * 0.02)
            }
            normalizeQK(&qkv, seqLen: seqLen)

            let prefillStateBuf = device.makeBuffer(
                bytes: state,
                length: stateCount * 2,
                options: .storageModeShared
            )!
            let prefillQkvBuf = device.makeBuffer(
                bytes: qkv,
                length: qkvCount * 2,
                options: .storageModeShared
            )!
            let prefillBProjBuf = device.makeBuffer(
                bytes: bProj,
                length: projCount * 2,
                options: .storageModeShared
            )!
            let prefillAProjBuf = device.makeBuffer(
                bytes: aProj,
                length: projCount * 2,
                options: .storageModeShared
            )!
            let aLogBuf = device.makeBuffer(
                bytes: aLog,
                length: numHeads * 2,
                options: .storageModeShared
            )!
            let dtBiasBuf = device.makeBuffer(
                bytes: dtBias,
                length: numHeads * 2,
                options: .storageModeShared
            )!
            let prefillOutBuf = device.makeBuffer(
                length: outCount * 2,
                options: .storageModeShared
            )!
            memset(prefillOutBuf.contents(), 0, outCount * 2)

            let prefillCmd = queue.makeCommandBuffer()!
            let prefillEnc = prefillCmd.makeComputeCommandEncoder()!
            prefillEnc.setComputePipelineState(prefillPipeline!)
            prefillEnc.setBuffer(prefillStateBuf, offset: 0, index: 0)
            prefillEnc.setBuffer(prefillQkvBuf, offset: 0, index: 1)
            prefillEnc.setBuffer(prefillBProjBuf, offset: 0, index: 2)
            prefillEnc.setBuffer(prefillAProjBuf, offset: 0, index: 3)
            prefillEnc.setBuffer(aLogBuf, offset: 0, index: 4)
            prefillEnc.setBuffer(dtBiasBuf, offset: 0, index: 5)
            prefillEnc.setBuffer(prefillOutBuf, offset: 0, index: 6)
            var headDimVal = UInt32(headDim)
            var numHeadsVal = UInt32(numHeads)
            var qkHeadsVal = UInt32(qkHeads)
            var seqLenVal = UInt32(seqLen)
            prefillEnc.setBytes(&headDimVal, length: 4, index: 7)
            prefillEnc.setBytes(&numHeadsVal, length: 4, index: 8)
            prefillEnc.setBytes(&qkHeadsVal, length: 4, index: 9)
            prefillEnc.setBytes(&seqLenVal, length: 4, index: 10)
            prefillEnc.dispatchThreads(
                MTLSize(width: 32, height: headDim, depth: numHeads),
                threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
            )
            prefillEnc.endEncoding()
            prefillCmd.commit()
            prefillCmd.waitUntilCompleted()
            XCTAssertNil(prefillCmd.error)

            let decodeStateBuf = device.makeBuffer(
                bytes: state,
                length: stateCount * 2,
                options: .storageModeShared
            )!
            let decodeQkvBuf = device.makeBuffer(length: channels * 2, options: .storageModeShared)!
            let decodeBProjBuf = device.makeBuffer(length: numHeads * 2, options: .storageModeShared)!
            let decodeAProjBuf = device.makeBuffer(length: numHeads * 2, options: .storageModeShared)!
            let decodeOutBuf = device.makeBuffer(length: hidden * 2, options: .storageModeShared)!

            var replayOutput = [Float16](repeating: 0, count: outCount)
            for pos in 0..<seqLen {
                let qkvBase = pos * channels
                let projBase = pos * numHeads
                let qkvSlice = Array(qkv[qkvBase..<(qkvBase + channels)])
                let bSlice = Array(bProj[projBase..<(projBase + numHeads)])
                let aSlice = Array(aProj[projBase..<(projBase + numHeads)])
                // memcpy returns its destination pointer; discard it so the
                // closure's trailing expression doesn't leak through withUnsafeBytes.
                _ = qkvSlice.withUnsafeBytes { src in
                    memcpy(decodeQkvBuf.contents(), src.baseAddress!, channels * 2)
                }
                _ = bSlice.withUnsafeBytes { src in
                    memcpy(decodeBProjBuf.contents(), src.baseAddress!, numHeads * 2)
                }
                _ = aSlice.withUnsafeBytes { src in
                    memcpy(decodeAProjBuf.contents(), src.baseAddress!, numHeads * 2)
                }
                memset(decodeOutBuf.contents(), 0, hidden * 2)

                let cmdBuf = queue.makeCommandBuffer()!
                let enc = cmdBuf.makeComputeCommandEncoder()!
                enc.setComputePipelineState(decodePipeline!)
                enc.setBuffer(decodeStateBuf, offset: 0, index: 0)
                enc.setBuffer(decodeQkvBuf, offset: 0, index: 1)
                enc.setBuffer(decodeBProjBuf, offset: 0, index: 2)
                enc.setBuffer(decodeAProjBuf, offset: 0, index: 3)
                enc.setBuffer(aLogBuf, offset: 0, index: 4)
                enc.setBuffer(dtBiasBuf, offset: 0, index: 5)
                enc.setBuffer(decodeOutBuf, offset: 0, index: 6)
                var headDimVal = UInt32(headDim)
                var headScaleVal = Float(1.0 / sqrt(Float(headDim)))
                var numHeadsVal = UInt32(numHeads)
                var qkHeadsVal = UInt32(qkHeads)
                enc.setBytes(&headDimVal, length: 4, index: 7)
                enc.setBytes(&headScaleVal, length: 4, index: 8)
                enc.setBytes(&numHeadsVal, length: 4, index: 9)
                enc.setBytes(&qkHeadsVal, length: 4, index: 10)
                enc.dispatchThreads(
                    MTLSize(width: 32, height: headDim, depth: numHeads),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
                )
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
                XCTAssertNil(cmdBuf.error)

                let ptr = decodeOutBuf.contents().bindMemory(to: Float16.self, capacity: hidden)
                for i in 0..<hidden {
                    replayOutput[pos * hidden + i] = ptr[i]
                }
            }

            let prefillOutPtr = prefillOutBuf.contents().bindMemory(to: Float16.self, capacity: outCount)
            let prefillStatePtr = prefillStateBuf.contents().bindMemory(to: Float16.self, capacity: stateCount)
            let decodeStatePtr = decodeStateBuf.contents().bindMemory(to: Float16.self, capacity: stateCount)
            var maxDiff: Float = 0
            var mismatchCount = 0
            for i in 0..<outCount {
                maxDiff = max(maxDiff, abs(Float(prefillOutPtr[i]) - Float(replayOutput[i])))
                if prefillOutPtr[i].bitPattern != replayOutput[i].bitPattern {
                    mismatchCount += 1
                }
            }
            for i in 0..<stateCount {
                maxDiff = max(maxDiff, abs(Float(prefillStatePtr[i]) - Float(decodeStatePtr[i])))
                if prefillStatePtr[i].bitPattern != decodeStatePtr[i].bitPattern {
                    mismatchCount += 1
                }
            }

            fputs(
                "  generic prefill recurrence vs decode replay seqLen=\(seqLen):"
                    + " mismatches=\(mismatchCount)"
                    + " max diff=\(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertEqual(
                mismatchCount,
                0,
                "Single-token prefill recurrence is not bit-exact with decode"
            )
        }
    }

    func testConv1dUpdateSiluPrefill_LargeBatchBoundaries() throws {
        let pipeline = makePipeline(
            shaderFile: "conv1d.metal",
            functionName: "conv1d_update_silu_c6144_k4_prefill"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile conv1d prefill kernel")

        let channels = 6144
        let kernelSize = 4

        let extendedLargeBatchCoverage = ProcessInfo.processInfo.environment["SMELT_RUN_QWEN_SMOKE"] == "1"
        let seqLens = extendedLargeBatchCoverage ? [64, 65, 192, 255, 256] : [64, 65]

        for seqLen in seqLens {
            let stateCount = channels * kernelSize
            let qkvCount = seqLen * channels
            var state = [Float16](repeating: 0, count: stateCount)
            var qkv = [Float16](repeating: 0, count: qkvCount)
            var weights = [Float16](repeating: 0, count: stateCount)

            for i in 0..<stateCount { state[i] = Float16(sin(Float(i) * 0.0013) * 0.05) }
            for i in 0..<qkvCount { qkv[i] = Float16(cos(Float(i) * 0.0021) * 0.45) }
            for i in 0..<stateCount { weights[i] = Float16(sin(Float(i) * 0.0017) * 0.2) }

            var refState = state.map(Float.init)
            var refQkv = qkv.map(Float.init)
            for pos in 0..<seqLen {
                let rowBase = pos * channels
                for ch in 0..<channels {
                    let s = ch * kernelSize
                    let x = refQkv[rowBase + ch]
                    refState[s + 0] = refState[s + 1]
                    refState[s + 1] = refState[s + 2]
                    refState[s + 2] = refState[s + 3]
                    refState[s + 3] = x
                    let acc =
                        refState[s + 0] * Float(weights[s + 0]) +
                        refState[s + 1] * Float(weights[s + 1]) +
                        refState[s + 2] * Float(weights[s + 2]) +
                        refState[s + 3] * Float(weights[s + 3])
                    refQkv[rowBase + ch] = acc / (1.0 + exp(-acc))
                }
            }

            let stateBuf = device.makeBuffer(bytes: state, length: stateCount * 2, options: .storageModeShared)!
            let qkvBuf = device.makeBuffer(bytes: qkv, length: qkvCount * 2, options: .storageModeShared)!
            let weightBuf = device.makeBuffer(bytes: weights, length: stateCount * 2, options: .storageModeShared)!

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline!)
            enc.setBuffer(stateBuf, offset: 0, index: 0)
            enc.setBuffer(qkvBuf, offset: 0, index: 1)
            enc.setBuffer(weightBuf, offset: 0, index: 2)
            var seqLenVal = UInt32(seqLen)
            enc.setBytes(&seqLenVal, length: 4, index: 3)
            enc.dispatchThreads(
                MTLSize(width: channels, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let statePtr = stateBuf.contents().bindMemory(to: Float16.self, capacity: stateCount)
            let qkvPtr = qkvBuf.contents().bindMemory(to: Float16.self, capacity: qkvCount)
            var maxDiff: Float = 0
            for i in 0..<stateCount {
                maxDiff = max(maxDiff, abs(Float(statePtr[i]) - refState[i]))
            }
            for i in 0..<qkvCount {
                maxDiff = max(maxDiff, abs(Float(qkvPtr[i]) - refQkv[i]))
            }

            fputs(
                "  conv1d_update_silu_c6144_k4_prefill seqLen=\(seqLen):"
                    + " max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.002, "Conv1d prefill diverges at seqLen \(seqLen)")
        }
    }

    func testConv1dUpdateSiluFusedQKL2_MatchesStagedBits() throws {
        let conv = makePipeline(shaderFile: "conv1d.metal", functionName: "conv1d_update_silu_c6144_k4")
        let l2 = makePipeline(shaderFile: "norms.metal", functionName: "l2_normalize_d128")
        let fused = makePipeline(
            shaderFile: "conv1d.metal",
            functionName: "conv1d_update_silu_l2_qk_c6144_k4_d128_h16"
        )
        try XCTSkipIf(conv == nil || l2 == nil || fused == nil, "Could not compile fused conv/L2 pipelines")

        let channels = 6_144
        let stateCount = channels * 4
        let state = (0..<stateCount).map { Float16(sin(Float($0 + 3) * 0.0013) * 0.25) }
        let qkv = (0..<channels).map { Float16(cos(Float($0 + 7) * 0.0031) * 0.5) }
        let weights = (0..<stateCount).map { Float16(sin(Float($0 + 13) * 0.0027) * 0.2) }

        let stagedState = device.makeBuffer(bytes: state, length: stateCount * 2, options: .storageModeShared)!
        let fusedState = device.makeBuffer(bytes: state, length: stateCount * 2, options: .storageModeShared)!
        let stagedQkv = device.makeBuffer(bytes: qkv, length: channels * 2, options: .storageModeShared)!
        let fusedQkv = device.makeBuffer(bytes: qkv, length: channels * 2, options: .storageModeShared)!
        let weightBuf = device.makeBuffer(bytes: weights, length: stateCount * 2, options: .storageModeShared)!

        let stagedCmd = queue.makeCommandBuffer()!
        let stagedEnc = stagedCmd.makeComputeCommandEncoder()!
        stagedEnc.setComputePipelineState(conv!)
        stagedEnc.setBuffer(stagedState, offset: 0, index: 0)
        stagedEnc.setBuffer(stagedQkv, offset: 0, index: 1)
        stagedEnc.setBuffer(weightBuf, offset: 0, index: 2)
        stagedEnc.setBuffer(stagedQkv, offset: 0, index: 3)
        stagedEnc.dispatchThreads(
            MTLSize(width: channels, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        stagedEnc.setComputePipelineState(l2!)
        stagedEnc.setBuffer(stagedQkv, offset: 0, index: 0)
        stagedEnc.dispatchThreadgroups(
            MTLSize(width: 16, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        stagedEnc.setBuffer(stagedQkv, offset: 16 * 128 * 2, index: 0)
        stagedEnc.dispatchThreadgroups(
            MTLSize(width: 16, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        stagedEnc.endEncoding()
        stagedCmd.commit()
        stagedCmd.waitUntilCompleted()
        XCTAssertNil(stagedCmd.error)

        let fusedCmd = queue.makeCommandBuffer()!
        let fusedEnc = fusedCmd.makeComputeCommandEncoder()!
        fusedEnc.setComputePipelineState(fused!)
        fusedEnc.setBuffer(fusedState, offset: 0, index: 0)
        fusedEnc.setBuffer(fusedQkv, offset: 0, index: 1)
        fusedEnc.setBuffer(weightBuf, offset: 0, index: 2)
        fusedEnc.setBuffer(fusedQkv, offset: 0, index: 3)
        fusedEnc.dispatchThreadgroups(
            MTLSize(width: 48, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        fusedEnc.endEncoding()
        fusedCmd.commit()
        fusedCmd.waitUntilCompleted()
        XCTAssertNil(fusedCmd.error)

        let stagedStateBits = stagedState.contents().bindMemory(to: UInt16.self, capacity: stateCount)
        let fusedStateBits = fusedState.contents().bindMemory(to: UInt16.self, capacity: stateCount)
        let stagedQkvBits = stagedQkv.contents().bindMemory(to: UInt16.self, capacity: channels)
        let fusedQkvBits = fusedQkv.contents().bindMemory(to: UInt16.self, capacity: channels)
        for i in 0..<stateCount { XCTAssertEqual(fusedStateBits[i], stagedStateBits[i], "state bit mismatch at \(i)") }
        for i in 0..<channels { XCTAssertEqual(fusedQkvBits[i], stagedQkvBits[i], "qkv bit mismatch at \(i)") }
    }

    func testL2NormalizePrefill_LargeBatchBoundaries() throws {
        let qPipeline = makePipeline(
            shaderFile: "norms.metal",
            functionName: "l2_normalize_q_d128_c6144_h16_prefill"
        )
        let kPipeline = makePipeline(
            shaderFile: "norms.metal",
            functionName: "l2_normalize_k_d128_c6144_h16_prefill"
        )
        try XCTSkipIf(qPipeline == nil || kPipeline == nil, "Could not compile Q/K l2 prefill kernels")

        let headDim = 128
        let numHeads = 16
        let hidden = 6144
        let kBase = numHeads * headDim

        let extendedLargeBatchCoverage = ProcessInfo.processInfo.environment["SMELT_RUN_QWEN_SMOKE"] == "1"
        let seqLens = extendedLargeBatchCoverage ? [64, 65, 192, 255, 256] : [64, 65]

        for seqLen in seqLens {
            let total = seqLen * hidden
            var qInput = [Float16](repeating: 0, count: total)
            var kInput = [Float16](repeating: 0, count: total)
            for i in 0..<total {
                qInput[i] = Float16(sin(Float(i) * 0.0023) * 0.5)
                kInput[i] = Float16(cos(Float(i) * 0.0029) * 0.45)
            }

            var qRef = qInput.map(Float.init)
            var kRef = kInput.map(Float.init)
            for pos in 0..<seqLen {
                let rowBase = pos * hidden
                for head in 0..<numHeads {
                    let qOff = rowBase + head * headDim
                    let kOff = rowBase + kBase + head * headDim

                    var qSum: Float = 0
                    var kSum: Float = 0
                    for d in 0..<headDim {
                        qSum += qRef[qOff + d] * qRef[qOff + d]
                        kSum += kRef[kOff + d] * kRef[kOff + d]
                    }
                    let qScale = 1.0 / max(sqrt(qSum), 1e-6)
                    let kScale = 1.0 / max(sqrt(kSum), 1e-6)
                    for d in 0..<headDim {
                        qRef[qOff + d] *= qScale
                        kRef[kOff + d] *= kScale
                    }
                }
            }

            let qBuf = device.makeBuffer(bytes: qInput, length: total * 2, options: .storageModeShared)!
            let kBuf = device.makeBuffer(bytes: kInput, length: total * 2, options: .storageModeShared)!

            let qCmd = queue.makeCommandBuffer()!
            let qEnc = qCmd.makeComputeCommandEncoder()!
            qEnc.setComputePipelineState(qPipeline!)
            qEnc.setBuffer(qBuf, offset: 0, index: 0)
            qEnc.dispatchThreadgroups(
                MTLSize(width: numHeads, height: seqLen, depth: 1),
                threadsPerThreadgroup: MTLSize(width: headDim, height: 1, depth: 1)
            )
            qEnc.endEncoding()
            qCmd.commit()
            qCmd.waitUntilCompleted()
            XCTAssertNil(qCmd.error)

            let kCmd = queue.makeCommandBuffer()!
            let kEnc = kCmd.makeComputeCommandEncoder()!
            kEnc.setComputePipelineState(kPipeline!)
            kEnc.setBuffer(kBuf, offset: 0, index: 0)
            kEnc.dispatchThreadgroups(
                MTLSize(width: numHeads, height: seqLen, depth: 1),
                threadsPerThreadgroup: MTLSize(width: headDim, height: 1, depth: 1)
            )
            kEnc.endEncoding()
            kCmd.commit()
            kCmd.waitUntilCompleted()
            XCTAssertNil(kCmd.error)

            let qPtr = qBuf.contents().bindMemory(to: Float16.self, capacity: total)
            let kPtr = kBuf.contents().bindMemory(to: Float16.self, capacity: total)
            var maxDiff: Float = 0
            for i in 0..<total {
                maxDiff = max(maxDiff, abs(Float(qPtr[i]) - qRef[i]))
                maxDiff = max(maxDiff, abs(Float(kPtr[i]) - kRef[i]))
            }

            fputs(
                "  l2_normalize_*_prefill seqLen=\(seqLen):"
                    + " max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.002, "L2 prefill normalization diverges at seqLen \(seqLen)")
        }
    }

    func testL2NormalizePrefill_MatchesSequentialDecodeAtLargeBatchBoundaries() throws {
        let decodePipeline = makePipeline(
            shaderFile: "norms.metal",
            functionName: "l2_normalize_d128"
        )
        let qPrefillPipeline = makePipeline(
            shaderFile: "norms.metal",
            functionName: "l2_normalize_q_d128_c6144_h16_prefill"
        )
        let kPrefillPipeline = makePipeline(
            shaderFile: "norms.metal",
            functionName: "l2_normalize_k_d128_c6144_h16_prefill"
        )
        try XCTSkipIf(
            decodePipeline == nil || qPrefillPipeline == nil || kPrefillPipeline == nil,
            "Could not compile decode/prefill L2 kernels"
        )

        let headDim = 128
        let numHeads = 16
        let hidden = 6144
        let qWidth = numHeads * headDim
        let kBase = qWidth

        for seqLen in [64, 65] {
            let total = seqLen * hidden
            var qkv = [Float16](repeating: 0, count: total)
            for i in 0..<total {
                qkv[i] = Float16(sin(Float(i) * 0.0023) * 0.5)
            }

            let prefillQBuf = device.makeBuffer(
                bytes: qkv,
                length: total * 2,
                options: .storageModeShared
            )!
            let prefillKBuf = device.makeBuffer(
                bytes: qkv,
                length: total * 2,
                options: .storageModeShared
            )!

            let qCmd = queue.makeCommandBuffer()!
            let qEnc = qCmd.makeComputeCommandEncoder()!
            qEnc.setComputePipelineState(qPrefillPipeline!)
            qEnc.setBuffer(prefillQBuf, offset: 0, index: 0)
            qEnc.dispatchThreadgroups(
                MTLSize(width: numHeads, height: seqLen, depth: 1),
                threadsPerThreadgroup: MTLSize(width: headDim, height: 1, depth: 1)
            )
            qEnc.endEncoding()
            qCmd.commit()
            qCmd.waitUntilCompleted()
            XCTAssertNil(qCmd.error)

            let kCmd = queue.makeCommandBuffer()!
            let kEnc = kCmd.makeComputeCommandEncoder()!
            kEnc.setComputePipelineState(kPrefillPipeline!)
            kEnc.setBuffer(prefillKBuf, offset: 0, index: 0)
            kEnc.dispatchThreadgroups(
                MTLSize(width: numHeads, height: seqLen, depth: 1),
                threadsPerThreadgroup: MTLSize(width: headDim, height: 1, depth: 1)
            )
            kEnc.endEncoding()
            kCmd.commit()
            kCmd.waitUntilCompleted()
            XCTAssertNil(kCmd.error)

            var decodeQReference = qkv
            var decodeKReference = qkv
            for pos in 0..<seqLen {
                let qSlice = Array(
                    decodeQReference[(pos * hidden)..<(pos * hidden + qWidth)]
                )
                let qBuf = device.makeBuffer(
                    bytes: qSlice,
                    length: qWidth * 2,
                    options: .storageModeShared
                )!
                let qStep = queue.makeCommandBuffer()!
                let qStepEnc = qStep.makeComputeCommandEncoder()!
                qStepEnc.setComputePipelineState(decodePipeline!)
                qStepEnc.setBuffer(qBuf, offset: 0, index: 0)
                qStepEnc.dispatchThreadgroups(
                    MTLSize(width: numHeads, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: headDim, height: 1, depth: 1)
                )
                qStepEnc.endEncoding()
                qStep.commit()
                qStep.waitUntilCompleted()
                XCTAssertNil(qStep.error)
                let qPtr = qBuf.contents().bindMemory(to: Float16.self, capacity: qWidth)
                for i in 0..<qWidth {
                    decodeQReference[pos * hidden + i] = qPtr[i]
                }

                let kSlice = Array(
                    decodeKReference[(pos * hidden + kBase)..<(pos * hidden + kBase + qWidth)]
                )
                let kBuf = device.makeBuffer(
                    bytes: kSlice,
                    length: qWidth * 2,
                    options: .storageModeShared
                )!
                let kStep = queue.makeCommandBuffer()!
                let kStepEnc = kStep.makeComputeCommandEncoder()!
                kStepEnc.setComputePipelineState(decodePipeline!)
                kStepEnc.setBuffer(kBuf, offset: 0, index: 0)
                kStepEnc.dispatchThreadgroups(
                    MTLSize(width: numHeads, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: headDim, height: 1, depth: 1)
                )
                kStepEnc.endEncoding()
                kStep.commit()
                kStep.waitUntilCompleted()
                XCTAssertNil(kStep.error)
                let kPtr = kBuf.contents().bindMemory(to: Float16.self, capacity: qWidth)
                for i in 0..<qWidth {
                    decodeKReference[pos * hidden + kBase + i] = kPtr[i]
                }
            }

            let prefillQPtr = prefillQBuf.contents().bindMemory(to: Float16.self, capacity: total)
            let prefillKPtr = prefillKBuf.contents().bindMemory(to: Float16.self, capacity: total)
            var maxQDiff: Float = 0
            var maxKDiff: Float = 0
            for pos in 0..<seqLen {
                let rowBase = pos * hidden
                for i in 0..<qWidth {
                    maxQDiff = max(
                        maxQDiff,
                        abs(Float(prefillQPtr[rowBase + i]) - Float(decodeQReference[rowBase + i]))
                    )
                    maxKDiff = max(
                        maxKDiff,
                        abs(Float(prefillKPtr[rowBase + kBase + i]) - Float(decodeKReference[rowBase + kBase + i]))
                    )
                }
            }

            fputs(
                "  l2_normalize_*_prefill vs decode seqLen=\(seqLen):"
                    + " q=\(String(format: "%.6f", maxQDiff))"
                    + " k=\(String(format: "%.6f", maxKDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxQDiff, 0.0001, "Q prefill L2 diverges from decode at seqLen \(seqLen)")
            XCTAssertLessThan(maxKDiff, 0.0001, "K prefill L2 diverges from decode at seqLen \(seqLen)")
        }
    }

    func testAttentionDecodeQwenSpecialization_MatchesGeneric() throws {
        let genericPipeline = makePipeline(
            shaderFile: "attention.metal", functionName: "attention_decode"
        )
        try XCTSkipIf(genericPipeline == nil, "Could not compile generic attention shader")

        let specializedPipeline = makePipeline(
            shaderFile: "attention.metal", functionName: "attention_decode_d256_h8_kv2"
        )
        try XCTSkipIf(
            specializedPipeline == nil,
            "Could not compile specialized Qwen attention shader"
        )

        let numQHeads = 8
        let numKVHeads = 2
        let headDim = 256
        let maxSeq = 256
        let seqLen = 37
        let scale = Float(1.0 / sqrt(Float(headDim)))

        let qCount = numQHeads * headDim
        let kvCount = numKVHeads * maxSeq * headDim
        var qData = [Float16](repeating: 0, count: qCount)
        var kData = [Float16](repeating: 0, count: kvCount)
        var vData = [Float16](repeating: 0, count: kvCount)
        var maskData = [Float16](repeating: 0, count: maxSeq)

        for i in 0..<qCount { qData[i] = Float16(sin(Float(i) * 0.013) * 0.5) }
        for i in 0..<kvCount { kData[i] = Float16(cos(Float(i) * 0.017) * 0.5) }
        for i in 0..<kvCount { vData[i] = Float16(sin(Float(i) * 0.019) * 0.5) }
        for i in 0..<maxSeq { maskData[i] = i < seqLen ? 0 : Float16(-10000) }

        let qGeneric = device.makeBuffer(
            bytes: qData, length: qCount * 2, options: .storageModeShared
        )!
        let qSpecialized = device.makeBuffer(
            bytes: qData, length: qCount * 2, options: .storageModeShared
        )!
        let kBuf = device.makeBuffer(
            bytes: kData, length: kvCount * 2, options: .storageModeShared
        )!
        let vBuf = device.makeBuffer(
            bytes: vData, length: kvCount * 2, options: .storageModeShared
        )!
        let maskBuf = device.makeBuffer(
            bytes: maskData, length: maxSeq * 2, options: .storageModeShared
        )!
        let genericOut = device.makeBuffer(
            length: qCount * 2, options: .storageModeShared
        )!
        memset(genericOut.contents(), 0, qCount * 2)

        let cmdBuf1 = queue.makeCommandBuffer()!
        let enc1 = cmdBuf1.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(genericPipeline!)
        enc1.setBuffer(qGeneric, offset: 0, index: 0)
        enc1.setBuffer(kBuf, offset: 0, index: 1)
        enc1.setBuffer(vBuf, offset: 0, index: 2)
        enc1.setBuffer(maskBuf, offset: 0, index: 3)
        enc1.setBuffer(genericOut, offset: 0, index: 4)
        var hdVal = UInt32(headDim); enc1.setBytes(&hdVal, length: 4, index: 5)
        var msVal = UInt32(maxSeq); enc1.setBytes(&msVal, length: 4, index: 6)
        var slVal = UInt32(seqLen); enc1.setBytes(&slVal, length: 4, index: 7)
        var kvhVal = UInt32(numKVHeads); enc1.setBytes(&kvhVal, length: 4, index: 8)
        var scaleVal = scale; enc1.setBytes(&scaleVal, length: 4, index: 9)
        var windowVal: UInt32 = 0; enc1.setBytes(&windowVal, length: 4, index: 10)
        enc1.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()
        XCTAssertNil(cmdBuf1.error)

        let cmdBuf2 = queue.makeCommandBuffer()!
        let enc2 = cmdBuf2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(specializedPipeline!)
        enc2.setBuffer(qSpecialized, offset: 0, index: 0)
        enc2.setBuffer(kBuf, offset: 0, index: 1)
        enc2.setBuffer(vBuf, offset: 0, index: 2)
        slVal = UInt32(seqLen); enc2.setBytes(&slVal, length: 4, index: 3)
        msVal = UInt32(maxSeq); enc2.setBytes(&msVal, length: 4, index: 4)
        enc2.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        enc2.endEncoding()
        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()
        XCTAssertNil(cmdBuf2.error)

        let genericPtr = genericOut.contents().bindMemory(to: Float16.self, capacity: qCount)
        let specializedPtr = qSpecialized.contents().bindMemory(to: Float16.self, capacity: qCount)
        var maxDiff: Float = 0
        for i in 0..<qCount {
            let diff = abs(Float(genericPtr[i]) - Float(specializedPtr[i]))
            if diff > maxDiff { maxDiff = diff }
        }

        fputs(
            "  attention_decode_d256_h8_kv2 vs generic:"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.02, "Specialized decode attention diverges from generic")
    }

    func testAttentionDecodeSoftcap_MatchesCPUReference() throws {
        let pipeline = makePipeline(
            shaderFile: "attention.metal", functionName: "attention_decode_softcap"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile softcap attention shader")

        let numQHeads = 4
        let numKVHeads = 2
        let headDim = 32
        let maxSeq = 64
        let seqLen = 17
        let gqaRatio = numQHeads / numKVHeads
        let scale = Float(1.0 / sqrt(Float(headDim)))
        let softcap: Float = 5.0

        let qCount = numQHeads * headDim
        let kvCount = numKVHeads * maxSeq * headDim
        var qData = [Float16](repeating: 0, count: qCount)
        var kData = [Float16](repeating: 0, count: kvCount)
        var vData = [Float16](repeating: 0, count: kvCount)
        var maskData = [Float16](repeating: 0, count: maxSeq)

        for i in 0..<qCount { qData[i] = Float16(sin(Float(i) * 0.011) * 0.5) }
        for i in 0..<kvCount { kData[i] = Float16(cos(Float(i) * 0.017) * 0.5) }
        for i in 0..<kvCount { vData[i] = Float16(sin(Float(i) * 0.023) * 0.5) }
        for i in 0..<maxSeq { maskData[i] = i < seqLen ? 0 : Float16(-10000) }

        let qBuf = device.makeBuffer(
            bytes: qData, length: qCount * 2, options: .storageModeShared
        )!
        let kBuf = device.makeBuffer(
            bytes: kData, length: kvCount * 2, options: .storageModeShared
        )!
        let vBuf = device.makeBuffer(
            bytes: vData, length: kvCount * 2, options: .storageModeShared
        )!
        let maskBuf = device.makeBuffer(
            bytes: maskData, length: maxSeq * 2, options: .storageModeShared
        )!
        let outBuf = device.makeBuffer(length: qCount * 2, options: .storageModeShared)!
        memset(outBuf.contents(), 0, qCount * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline!)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kBuf, offset: 0, index: 1)
        enc.setBuffer(vBuf, offset: 0, index: 2)
        enc.setBuffer(maskBuf, offset: 0, index: 3)
        enc.setBuffer(outBuf, offset: 0, index: 4)
        var hdVal = UInt32(headDim); enc.setBytes(&hdVal, length: 4, index: 5)
        var msVal = UInt32(maxSeq); enc.setBytes(&msVal, length: 4, index: 6)
        var slVal = UInt32(seqLen); enc.setBytes(&slVal, length: 4, index: 7)
        var kvhVal = UInt32(numKVHeads); enc.setBytes(&kvhVal, length: 4, index: 8)
        var scaleVal = scale; enc.setBytes(&scaleVal, length: 4, index: 9)
        var windowVal: UInt32 = 0; enc.setBytes(&windowVal, length: 4, index: 10)
        var softcapVal = softcap; enc.setBytes(&softcapVal, length: 4, index: 11)
        enc.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = outBuf.contents().bindMemory(to: Float16.self, capacity: qCount)
        var maxDiff: Float = 0
        for qHead in 0..<numQHeads {
            let kvHead = qHead / gqaRatio
            var scores = [Float](repeating: 0, count: seqLen)
            for s in 0..<seqLen {
                var dot: Float = 0
                for d in 0..<headDim {
                    let q = Float(qData[qHead * headDim + d])
                    let k = Float(kData[kvHead * maxSeq * headDim + s * headDim + d])
                    dot += q * k
                }
                let scaled = dot * scale
                scores[s] = softcap * tanh(scaled / softcap)
            }
            let maxScore = scores.max() ?? 0
            var exps = scores.map { exp($0 - maxScore) }
            let sumExp = exps.reduce(0, +)
            exps = exps.map { $0 / sumExp }
            for d in 0..<headDim {
                var expected: Float = 0
                for s in 0..<seqLen {
                    let v = Float(vData[kvHead * maxSeq * headDim + s * headDim + d])
                    expected += exps[s] * v
                }
                let actual = Float(outPtr[qHead * headDim + d])
                maxDiff = max(maxDiff, abs(actual - expected))
            }
        }

        fputs(
            "  attention_decode_softcap vs cpu:"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.01, "Softcap decode attention diverges from CPU reference")
    }

    func testAttentionDecodeSoftcap_AllowsAliasedQueryAndOutput() throws {
        let pipeline = makePipeline(
            shaderFile: "attention.metal", functionName: "attention_decode_softcap"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile softcap attention shader")

        let numQHeads = 4
        let numKVHeads = 2
        let headDim = 512
        let maxSeq = 64
        let seqLen = 17
        let gqaRatio = numQHeads / numKVHeads
        let scale = Float(1.0 / sqrt(Float(headDim)))
        let softcap: Float = 5.0

        let qCount = numQHeads * headDim
        let kvCount = numKVHeads * maxSeq * headDim
        var qData = [Float16](repeating: 0, count: qCount)
        var kData = [Float16](repeating: 0, count: kvCount)
        var vData = [Float16](repeating: 0, count: kvCount)
        var maskData = [Float16](repeating: 0, count: maxSeq)

        for i in 0..<qCount { qData[i] = Float16(sin(Float(i) * 0.011) * 0.5) }
        for i in 0..<kvCount { kData[i] = Float16(cos(Float(i) * 0.017) * 0.5) }
        for i in 0..<kvCount { vData[i] = Float16(sin(Float(i) * 0.023) * 0.5) }
        for i in 0..<maxSeq { maskData[i] = i < seqLen ? 0 : Float16(-10000) }

        let qBuf = device.makeBuffer(
            bytes: qData, length: qCount * 2, options: .storageModeShared
        )!
        let kBuf = device.makeBuffer(
            bytes: kData, length: kvCount * 2, options: .storageModeShared
        )!
        let vBuf = device.makeBuffer(
            bytes: vData, length: kvCount * 2, options: .storageModeShared
        )!
        let maskBuf = device.makeBuffer(
            bytes: maskData, length: maxSeq * 2, options: .storageModeShared
        )!

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline!)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kBuf, offset: 0, index: 1)
        enc.setBuffer(vBuf, offset: 0, index: 2)
        enc.setBuffer(maskBuf, offset: 0, index: 3)
        enc.setBuffer(qBuf, offset: 0, index: 4)
        var hdVal = UInt32(headDim); enc.setBytes(&hdVal, length: 4, index: 5)
        var msVal = UInt32(maxSeq); enc.setBytes(&msVal, length: 4, index: 6)
        var slVal = UInt32(seqLen); enc.setBytes(&slVal, length: 4, index: 7)
        var kvhVal = UInt32(numKVHeads); enc.setBytes(&kvhVal, length: 4, index: 8)
        var scaleVal = scale; enc.setBytes(&scaleVal, length: 4, index: 9)
        var windowVal: UInt32 = 0; enc.setBytes(&windowVal, length: 4, index: 10)
        var softcapVal = softcap; enc.setBytes(&softcapVal, length: 4, index: 11)
        enc.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = qBuf.contents().bindMemory(to: Float16.self, capacity: qCount)
        var maxDiff: Float = 0
        for qHead in 0..<numQHeads {
            let kvHead = qHead / gqaRatio
            var scores = [Float](repeating: 0, count: seqLen)
            for s in 0..<seqLen {
                var dot: Float = 0
                for d in 0..<headDim {
                    let q = Float(qData[qHead * headDim + d])
                    let k = Float(kData[kvHead * maxSeq * headDim + s * headDim + d])
                    dot += q * k
                }
                let scaled = dot * scale
                scores[s] = softcap * tanh(scaled / softcap)
            }
            let maxScore = scores.max() ?? 0
            var exps = scores.map { exp($0 - maxScore) }
            let sumExp = exps.reduce(0, +)
            exps = exps.map { $0 / sumExp }
            for d in 0..<headDim {
                var expected: Float = 0
                for s in 0..<seqLen {
                    let v = Float(vData[kvHead * maxSeq * headDim + s * headDim + d])
                    expected += exps[s] * v
                }
                let actual = Float(outPtr[qHead * headDim + d])
                maxDiff = max(maxDiff, abs(actual - expected))
            }
        }

        XCTAssertLessThan(
            maxDiff,
            0.01,
            "Softcap decode attention diverges when query and output alias"
        )
    }

    func testAttentionDecodeSlidingWindow_MatchesCPUReference() throws {
        let pipeline = makePipeline(
            shaderFile: "attention.metal", functionName: "attention_decode"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile generic attention shader")

        let numQHeads = 4
        let numKVHeads = 2
        let headDim = 32
        let maxSeq = 64
        let seqLen = 17
        let slidingWindow = 8
        let gqaRatio = numQHeads / numKVHeads
        let scale = Float(1.0 / sqrt(Float(headDim)))

        let qCount = numQHeads * headDim
        let kvCount = numKVHeads * maxSeq * headDim
        var qData = [Float16](repeating: 0, count: qCount)
        var kData = [Float16](repeating: 0, count: kvCount)
        var vData = [Float16](repeating: 0, count: kvCount)
        var maskData = [Float16](repeating: 0, count: maxSeq)

        for i in 0..<qCount { qData[i] = Float16(sin(Float(i) * 0.011) * 0.5) }
        for i in 0..<kvCount { kData[i] = Float16(cos(Float(i) * 0.017) * 0.5) }
        for i in 0..<kvCount { vData[i] = Float16(sin(Float(i) * 0.023) * 0.5) }
        for i in 0..<maxSeq { maskData[i] = i < seqLen ? 0 : Float16(-10000) }

        let qBuf = device.makeBuffer(bytes: qData, length: qCount * 2, options: .storageModeShared)!
        let kBuf = device.makeBuffer(bytes: kData, length: kvCount * 2, options: .storageModeShared)!
        let vBuf = device.makeBuffer(bytes: vData, length: kvCount * 2, options: .storageModeShared)!
        let maskBuf = device.makeBuffer(bytes: maskData, length: maxSeq * 2, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: qCount * 2, options: .storageModeShared)!
        memset(outBuf.contents(), 0, qCount * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline!)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kBuf, offset: 0, index: 1)
        enc.setBuffer(vBuf, offset: 0, index: 2)
        enc.setBuffer(maskBuf, offset: 0, index: 3)
        enc.setBuffer(outBuf, offset: 0, index: 4)
        var hdVal = UInt32(headDim); enc.setBytes(&hdVal, length: 4, index: 5)
        var msVal = UInt32(maxSeq); enc.setBytes(&msVal, length: 4, index: 6)
        var slVal = UInt32(seqLen); enc.setBytes(&slVal, length: 4, index: 7)
        var kvhVal = UInt32(numKVHeads); enc.setBytes(&kvhVal, length: 4, index: 8)
        var scaleVal = scale; enc.setBytes(&scaleVal, length: 4, index: 9)
        var windowVal = UInt32(slidingWindow); enc.setBytes(&windowVal, length: 4, index: 10)
        enc.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = outBuf.contents().bindMemory(to: Float16.self, capacity: qCount)
        var maxDiff: Float = 0
        let seqStart = max(0, seqLen - slidingWindow)
        for qHead in 0..<numQHeads {
            let kvHead = qHead / gqaRatio
            var scores = [Float](repeating: 0, count: seqLen)
            var maxScore = -Float.infinity
            for s in seqStart..<seqLen {
                var dot: Float = 0
                for d in 0..<headDim {
                    let q = Float(qData[qHead * headDim + d])
                    let k = Float(kData[kvHead * maxSeq * headDim + s * headDim + d])
                    dot += q * k
                }
                scores[s] = dot * scale
                maxScore = max(maxScore, scores[s])
            }
            var exps = [Float](repeating: 0, count: seqLen)
            var sumExp: Float = 0
            for s in seqStart..<seqLen {
                exps[s] = exp(scores[s] - maxScore)
                sumExp += exps[s]
            }
            let invSum = 1.0 / sumExp
            for d in 0..<headDim {
                var expected: Float = 0
                for s in seqStart..<seqLen {
                    let v = Float(vData[kvHead * maxSeq * headDim + s * headDim + d])
                    expected += exps[s] * invSum * v
                }
                let actual = Float(outPtr[qHead * headDim + d])
                maxDiff = max(maxDiff, abs(actual - expected))
            }
        }

        fputs(
            "  attention_decode sliding window vs cpu:"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.01, "Sliding-window decode attention diverges from CPU reference")
    }

    func testAttentionDecodeSlidingWindow_AllowsAliasedQueryAndOutput() throws {
        let pipeline = makePipeline(
            shaderFile: "attention.metal", functionName: "attention_decode"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile generic attention shader")

        let numQHeads = 4
        let numKVHeads = 2
        let headDim = 512
        let maxSeq = 64
        let seqLen = 17
        let slidingWindow = 8
        let gqaRatio = numQHeads / numKVHeads
        let scale = Float(1.0 / sqrt(Float(headDim)))

        let qCount = numQHeads * headDim
        let kvCount = numKVHeads * maxSeq * headDim
        var qData = [Float16](repeating: 0, count: qCount)
        var kData = [Float16](repeating: 0, count: kvCount)
        var vData = [Float16](repeating: 0, count: kvCount)
        var maskData = [Float16](repeating: 0, count: maxSeq)

        for i in 0..<qCount { qData[i] = Float16(sin(Float(i) * 0.011) * 0.5) }
        for i in 0..<kvCount { kData[i] = Float16(cos(Float(i) * 0.017) * 0.5) }
        for i in 0..<kvCount { vData[i] = Float16(sin(Float(i) * 0.023) * 0.5) }
        for i in 0..<maxSeq { maskData[i] = i < seqLen ? 0 : Float16(-10000) }

        let qBuf = device.makeBuffer(
            bytes: qData, length: qCount * 2, options: .storageModeShared
        )!
        let kBuf = device.makeBuffer(
            bytes: kData, length: kvCount * 2, options: .storageModeShared
        )!
        let vBuf = device.makeBuffer(
            bytes: vData, length: kvCount * 2, options: .storageModeShared
        )!
        let maskBuf = device.makeBuffer(
            bytes: maskData, length: maxSeq * 2, options: .storageModeShared
        )!

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline!)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kBuf, offset: 0, index: 1)
        enc.setBuffer(vBuf, offset: 0, index: 2)
        enc.setBuffer(maskBuf, offset: 0, index: 3)
        enc.setBuffer(qBuf, offset: 0, index: 4)
        var hdVal = UInt32(headDim); enc.setBytes(&hdVal, length: 4, index: 5)
        var msVal = UInt32(maxSeq); enc.setBytes(&msVal, length: 4, index: 6)
        var slVal = UInt32(seqLen); enc.setBytes(&slVal, length: 4, index: 7)
        var kvhVal = UInt32(numKVHeads); enc.setBytes(&kvhVal, length: 4, index: 8)
        var scaleVal = scale; enc.setBytes(&scaleVal, length: 4, index: 9)
        var windowVal = UInt32(slidingWindow); enc.setBytes(&windowVal, length: 4, index: 10)
        enc.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = qBuf.contents().bindMemory(to: Float16.self, capacity: qCount)
        let seqStart = max(0, seqLen - slidingWindow)
        var maxDiff: Float = 0
        for qHead in 0..<numQHeads {
            let kvHead = qHead / gqaRatio
            var scores = [Float](repeating: 0, count: seqLen)
            for s in seqStart..<seqLen {
                var dot: Float = 0
                for d in 0..<headDim {
                    let q = Float(qData[qHead * headDim + d])
                    let k = Float(kData[kvHead * maxSeq * headDim + s * headDim + d])
                    dot += q * k
                }
                scores[s] = dot * scale
            }
            let maxScore = scores[seqStart..<seqLen].max() ?? 0
            var sumExp: Float = 0
            for s in seqStart..<seqLen {
                scores[s] = exp(scores[s] - maxScore)
                sumExp += scores[s]
            }
            for d in 0..<headDim {
                var expected: Float = 0
                for s in seqStart..<seqLen {
                    let v = Float(vData[kvHead * maxSeq * headDim + s * headDim + d])
                    expected += (scores[s] / sumExp) * v
                }
                let actual = Float(outPtr[qHead * headDim + d])
                maxDiff = max(maxDiff, abs(actual - expected))
            }
        }

        XCTAssertLessThan(
            maxDiff,
            0.01,
            "Generic decode attention diverges when query and output alias"
        )
    }

    func testAttentionDecode_LongContextMatchesCPUReference() throws {
        let pipeline = makePipeline(
            shaderFile: "attention.metal", functionName: "attention_decode"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile generic attention shader")

        let numQHeads = 4
        let numKVHeads = 2
        let headDim = 32
        let maxSeq = 384
        let seqLen = 320
        let gqaRatio = numQHeads / numKVHeads
        let scale = Float(1.0 / sqrt(Float(headDim)))

        let qCount = numQHeads * headDim
        let kvCount = numKVHeads * maxSeq * headDim
        var qData = [Float16](repeating: 0, count: qCount)
        var kData = [Float16](repeating: 0, count: kvCount)
        var vData = [Float16](repeating: 0, count: kvCount)
        var maskData = [Float16](repeating: 0, count: maxSeq)

        for i in 0..<qCount { qData[i] = Float16(sin(Float(i) * 0.011) * 0.5) }
        for i in 0..<kvCount { kData[i] = Float16(cos(Float(i) * 0.017) * 0.5) }
        for i in 0..<kvCount { vData[i] = Float16(sin(Float(i) * 0.023) * 0.5) }
        for i in 0..<maxSeq { maskData[i] = i < seqLen ? 0 : Float16(-10000) }

        let qBuf = device.makeBuffer(bytes: qData, length: qCount * 2, options: .storageModeShared)!
        let kBuf = device.makeBuffer(bytes: kData, length: kvCount * 2, options: .storageModeShared)!
        let vBuf = device.makeBuffer(bytes: vData, length: kvCount * 2, options: .storageModeShared)!
        let maskBuf = device.makeBuffer(bytes: maskData, length: maxSeq * 2, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: qCount * 2, options: .storageModeShared)!
        memset(outBuf.contents(), 0, qCount * 2)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline!)
        enc.setBuffer(qBuf, offset: 0, index: 0)
        enc.setBuffer(kBuf, offset: 0, index: 1)
        enc.setBuffer(vBuf, offset: 0, index: 2)
        enc.setBuffer(maskBuf, offset: 0, index: 3)
        enc.setBuffer(outBuf, offset: 0, index: 4)
        var hdVal = UInt32(headDim); enc.setBytes(&hdVal, length: 4, index: 5)
        var msVal = UInt32(maxSeq); enc.setBytes(&msVal, length: 4, index: 6)
        var slVal = UInt32(seqLen); enc.setBytes(&slVal, length: 4, index: 7)
        var kvhVal = UInt32(numKVHeads); enc.setBytes(&kvhVal, length: 4, index: 8)
        var scaleVal = scale; enc.setBytes(&scaleVal, length: 4, index: 9)
        var windowVal: UInt32 = 0; enc.setBytes(&windowVal, length: 4, index: 10)
        enc.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = outBuf.contents().bindMemory(to: Float16.self, capacity: qCount)
        var maxDiff: Float = 0
        for qHead in 0..<numQHeads {
            let kvHead = qHead / gqaRatio
            var scores = [Float](repeating: 0, count: seqLen)
            var maxScore = -Float.infinity
            for s in 0..<seqLen {
                var dot: Float = 0
                for d in 0..<headDim {
                    let q = Float(qData[qHead * headDim + d])
                    let k = Float(kData[kvHead * maxSeq * headDim + s * headDim + d])
                    dot += q * k
                }
                scores[s] = dot * scale
                maxScore = max(maxScore, scores[s])
            }
            var exps = [Float](repeating: 0, count: seqLen)
            var sumExp: Float = 0
            for s in 0..<seqLen {
                exps[s] = exp(scores[s] - maxScore)
                sumExp += exps[s]
            }
            let invSum = 1.0 / sumExp
            for d in 0..<headDim {
                var expected: Float = 0
                for s in 0..<seqLen {
                    let v = Float(vData[kvHead * maxSeq * headDim + s * headDim + d])
                    expected += exps[s] * invSum * v
                }
                let actual = Float(outPtr[qHead * headDim + d])
                maxDiff = max(maxDiff, abs(actual - expected))
            }
        }

        fputs(
            "  attention_decode long context vs cpu:"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.01, "Long-context decode attention diverges from CPU reference")
    }

    func testAttentionDecodeQwenSDPASpecialization_MatchesGeneric() throws {
        let genericPipeline = makePipeline(
            shaderFile: "attention.metal", functionName: "attention_decode"
        )
        try XCTSkipIf(genericPipeline == nil, "Could not compile generic attention shader")

        let specializedPipeline = makePipeline(
            shaderFile: "attention.metal", functionName: "attention_decode_d256_h8_kv2_sdpa"
        )
        try XCTSkipIf(
            specializedPipeline == nil,
            "Could not compile SDPA decode attention specialization"
        )

        let numQHeads = 8
        let numKVHeads = 2
        let headDim = 256
        let maxSeq = 256
        let seqLen = 192
        let scale = Float(1.0 / sqrt(Float(headDim)))

        let qCount = numQHeads * headDim
        let kvCount = numKVHeads * maxSeq * headDim
        var qData = [Float16](repeating: 0, count: qCount)
        var kData = [Float16](repeating: 0, count: kvCount)
        var vData = [Float16](repeating: 0, count: kvCount)
        var maskData = [Float16](repeating: 0, count: maxSeq)

        for i in 0..<qCount { qData[i] = Float16(sin(Float(i) * 0.013) * 0.5) }
        for i in 0..<kvCount { kData[i] = Float16(cos(Float(i) * 0.017) * 0.5) }
        for i in 0..<kvCount { vData[i] = Float16(sin(Float(i) * 0.019) * 0.5) }
        for i in 0..<maxSeq { maskData[i] = i < seqLen ? 0 : Float16(-10000) }

        let qGeneric = device.makeBuffer(
            bytes: qData, length: qCount * 2, options: .storageModeShared
        )!
        let qSpecialized = device.makeBuffer(
            bytes: qData, length: qCount * 2, options: .storageModeShared
        )!
        let kBuf = device.makeBuffer(
            bytes: kData, length: kvCount * 2, options: .storageModeShared
        )!
        let vBuf = device.makeBuffer(
            bytes: vData, length: kvCount * 2, options: .storageModeShared
        )!
        let maskBuf = device.makeBuffer(
            bytes: maskData, length: maxSeq * 2, options: .storageModeShared
        )!
        let genericOut = device.makeBuffer(
            length: qCount * 2, options: .storageModeShared
        )!
        memset(genericOut.contents(), 0, qCount * 2)

        let cmdBuf1 = queue.makeCommandBuffer()!
        let enc1 = cmdBuf1.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(genericPipeline!)
        enc1.setBuffer(qGeneric, offset: 0, index: 0)
        enc1.setBuffer(kBuf, offset: 0, index: 1)
        enc1.setBuffer(vBuf, offset: 0, index: 2)
        enc1.setBuffer(maskBuf, offset: 0, index: 3)
        enc1.setBuffer(genericOut, offset: 0, index: 4)
        var hdVal = UInt32(headDim); enc1.setBytes(&hdVal, length: 4, index: 5)
        var msVal = UInt32(maxSeq); enc1.setBytes(&msVal, length: 4, index: 6)
        var slVal = UInt32(seqLen); enc1.setBytes(&slVal, length: 4, index: 7)
        var kvhVal = UInt32(numKVHeads); enc1.setBytes(&kvhVal, length: 4, index: 8)
        var scaleVal = scale; enc1.setBytes(&scaleVal, length: 4, index: 9)
        var windowVal: UInt32 = 0; enc1.setBytes(&windowVal, length: 4, index: 10)
        enc1.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()
        XCTAssertNil(cmdBuf1.error)

        let cmdBuf2 = queue.makeCommandBuffer()!
        let enc2 = cmdBuf2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(specializedPipeline!)
        enc2.setBuffer(qSpecialized, offset: 0, index: 0)
        enc2.setBuffer(kBuf, offset: 0, index: 1)
        enc2.setBuffer(vBuf, offset: 0, index: 2)
        slVal = UInt32(seqLen); enc2.setBytes(&slVal, length: 4, index: 3)
        msVal = UInt32(maxSeq); enc2.setBytes(&msVal, length: 4, index: 4)
        enc2.dispatchThreadgroups(
            MTLSize(width: numQHeads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)
        )
        enc2.endEncoding()
        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()
        XCTAssertNil(cmdBuf2.error)

        let genericPtr = genericOut.contents().bindMemory(to: Float16.self, capacity: qCount)
        let specializedPtr = qSpecialized.contents().bindMemory(to: Float16.self, capacity: qCount)
        var maxDiff: Float = 0
        for i in 0..<qCount {
            let diff = abs(Float(genericPtr[i]) - Float(specializedPtr[i]))
            if diff > maxDiff { maxDiff = diff }
        }

        fputs(
            "  attention_decode_d256_h8_kv2_sdpa vs generic:"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.02, "SDPA decode attention diverges from generic")
    }

    func testAttentionDecodeD256H24KV4SDPA_MatchesGenericAcrossRouteBoundaries() throws {
        let pipelines = makePipelines(
            shaderFile: "attention.metal",
            functionNames: ["attention_decode", "attention_decode_d256_h24_kv4_sdpa"]
        )
        let genericPipeline = try XCTUnwrap(pipelines?["attention_decode"])
        let specializedPipeline = try XCTUnwrap(
            pipelines?["attention_decode_d256_h24_kv4_sdpa"]
        )

        let numQHeads = 24
        let numKVHeads = 4
        let gqaRatio = numQHeads / numKVHeads
        let headDim = 256
        let cacheCapacity = 384
        let scale = Float(1.0 / sqrt(Float(headDim)))
        let qCount = numQHeads * headDim
        let kvCount = numKVHeads * cacheCapacity * headDim

        var qData = [Float16](repeating: 0, count: qCount)
        var kData = [Float16](repeating: 0, count: kvCount)
        var vData = [Float16](repeating: 0, count: kvCount)
        for i in 0..<qCount { qData[i] = Float16(sin(Float(i) * 0.013) * 0.5) }
        for i in 0..<kvCount { kData[i] = Float16(cos(Float(i) * 0.017) * 0.5) }
        for i in 0..<kvCount { vData[i] = Float16(sin(Float(i) * 0.019) * 0.5) }

        let kBuffer = device.makeBuffer(
            bytes: kData, length: kvCount * 2, options: .storageModeShared
        )!
        let vBuffer = device.makeBuffer(
            bytes: vData, length: kvCount * 2, options: .storageModeShared
        )!

        for seqLen in [63, 64, 255, 256, 257, cacheCapacity] {
            var maskData = [Float16](repeating: -10_000, count: cacheCapacity)
            for i in 0..<seqLen { maskData[i] = 0 }
            let maskBuffer = device.makeBuffer(
                bytes: maskData, length: cacheCapacity * 2, options: .storageModeShared
            )!
            let genericQuery = device.makeBuffer(
                bytes: qData, length: qCount * 2, options: .storageModeShared
            )!
            let specializedQuery = device.makeBuffer(
                bytes: qData, length: qCount * 2, options: .storageModeShared
            )!
            let genericOutput = device.makeBuffer(
                length: qCount * 2, options: .storageModeShared
            )!

            let genericCommand = queue.makeCommandBuffer()!
            let genericEncoder = genericCommand.makeComputeCommandEncoder()!
            genericEncoder.setComputePipelineState(genericPipeline)
            genericEncoder.setBuffer(genericQuery, offset: 0, index: 0)
            genericEncoder.setBuffer(kBuffer, offset: 0, index: 1)
            genericEncoder.setBuffer(vBuffer, offset: 0, index: 2)
            genericEncoder.setBuffer(maskBuffer, offset: 0, index: 3)
            genericEncoder.setBuffer(genericOutput, offset: 0, index: 4)
            var headDimValue = UInt32(headDim)
            var capacityValue = UInt32(cacheCapacity)
            var seqLenValue = UInt32(seqLen)
            var kvHeadsValue = UInt32(numKVHeads)
            var scaleValue = scale
            var windowValue: UInt32 = 0
            genericEncoder.setBytes(&headDimValue, length: 4, index: 5)
            genericEncoder.setBytes(&capacityValue, length: 4, index: 6)
            genericEncoder.setBytes(&seqLenValue, length: 4, index: 7)
            genericEncoder.setBytes(&kvHeadsValue, length: 4, index: 8)
            genericEncoder.setBytes(&scaleValue, length: 4, index: 9)
            genericEncoder.setBytes(&windowValue, length: 4, index: 10)
            genericEncoder.dispatchThreadgroups(
                MTLSize(width: numQHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            genericEncoder.endEncoding()
            genericCommand.commit()
            genericCommand.waitUntilCompleted()
            XCTAssertNil(genericCommand.error)

            let specializedCommand = queue.makeCommandBuffer()!
            let specializedEncoder = specializedCommand.makeComputeCommandEncoder()!
            specializedEncoder.setComputePipelineState(specializedPipeline)
            specializedEncoder.setBuffer(specializedQuery, offset: 0, index: 0)
            specializedEncoder.setBuffer(kBuffer, offset: 0, index: 1)
            specializedEncoder.setBuffer(vBuffer, offset: 0, index: 2)
            seqLenValue = UInt32(seqLen)
            capacityValue = UInt32(cacheCapacity)
            specializedEncoder.setBytes(&seqLenValue, length: 4, index: 3)
            specializedEncoder.setBytes(&capacityValue, length: 4, index: 4)
            specializedEncoder.dispatchThreadgroups(
                MTLSize(width: numQHeads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)
            )
            specializedEncoder.endEncoding()
            specializedCommand.commit()
            specializedCommand.waitUntilCompleted()
            XCTAssertNil(specializedCommand.error)

            let generic = genericOutput.contents().bindMemory(
                to: Float16.self, capacity: qCount
            )
            let specialized = specializedQuery.contents().bindMemory(
                to: Float16.self, capacity: qCount
            )
            var maxDiff: Float = 0
            for qHead in 0..<numQHeads {
                XCTAssertEqual(qHead / gqaRatio, min(qHead / gqaRatio, numKVHeads - 1))
                for d in 0..<headDim {
                    let i = qHead * headDim + d
                    maxDiff = max(maxDiff, abs(Float(generic[i]) - Float(specialized[i])))
                }
            }
            XCTAssertLessThan(
                maxDiff,
                0.02,
                "H24/KV4 SDPA diverges from generic at sequence length \(seqLen)"
            )
        }
    }

    func testRmsNormDecodeD2048_MatchesGeneric() throws {
        let genericPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw"
        )
        try XCTSkipIf(genericPipeline == nil, "Could not compile generic RMSNorm shader")

        let specializedPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw_d2048"
        )
        try XCTSkipIf(
            specializedPipeline == nil,
            "Could not compile specialized d2048 RMSNorm shader"
        )

        let dim = 2048
        let eps: Float = 1e-6
        var inputData = [Float16](repeating: 0, count: dim)
        var weightData = [Float16](repeating: 0, count: dim)
        for i in 0..<dim {
            inputData[i] = Float16(sin(Float(i) * 0.0031) * 1.7)
            weightData[i] = Float16(cos(Float(i) * 0.0097) * 0.02)
        }

        let inputBuf = device.makeBuffer(
            bytes: inputData, length: dim * 2, options: .storageModeShared
        )!
        let weightBuf = device.makeBuffer(
            bytes: weightData, length: dim * 2, options: .storageModeShared
        )!
        let genericOut = device.makeBuffer(length: dim * 2, options: .storageModeShared)!
        let specializedOut = device.makeBuffer(length: dim * 2, options: .storageModeShared)!
        memset(genericOut.contents(), 0, dim * 2)
        memset(specializedOut.contents(), 0, dim * 2)

        let cmdBuf1 = queue.makeCommandBuffer()!
        let enc1 = cmdBuf1.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(genericPipeline!)
        enc1.setBuffer(inputBuf, offset: 0, index: 0)
        enc1.setBuffer(weightBuf, offset: 0, index: 1)
        enc1.setBuffer(genericOut, offset: 0, index: 2)
        var dimVal = UInt32(dim); enc1.setBytes(&dimVal, length: 4, index: 3)
        var epsVal = eps; enc1.setBytes(&epsVal, length: 4, index: 4)
        enc1.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)
        )
        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()
        XCTAssertNil(cmdBuf1.error)

        let cmdBuf2 = queue.makeCommandBuffer()!
        let enc2 = cmdBuf2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(specializedPipeline!)
        enc2.setBuffer(inputBuf, offset: 0, index: 0)
        enc2.setBuffer(weightBuf, offset: 0, index: 1)
        enc2.setBuffer(specializedOut, offset: 0, index: 2)
        enc2.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc2.endEncoding()
        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()
        XCTAssertNil(cmdBuf2.error)

        let genericPtr = genericOut.contents().bindMemory(to: Float16.self, capacity: dim)
        let specializedPtr = specializedOut.contents().bindMemory(to: Float16.self, capacity: dim)
        var maxDiff: Float = 0
        for i in 0..<dim {
            let diff = abs(Float(genericPtr[i]) - Float(specializedPtr[i]))
            if diff > maxDiff { maxDiff = diff }
        }

        fputs(
            "  rms_norm_1pw_d2048 vs generic:"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.002, "Specialized d2048 RMSNorm diverges from generic")
    }

    func testRmsNormDecodeD2560_MatchesGeneric() throws {
        let genericPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw"
        )
        try XCTSkipIf(genericPipeline == nil, "Could not compile generic RMSNorm shader")

        let specializedPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw_d2560"
        )
        try XCTSkipIf(
            specializedPipeline == nil,
            "Could not compile specialized d2560 RMSNorm shader"
        )

        let dim = 2560
        let eps: Float = 1e-6
        var inputData = [Float16](repeating: 0, count: dim)
        var weightData = [Float16](repeating: 0, count: dim)
        for i in 0..<dim {
            inputData[i] = Float16(sin(Float(i) * 0.0027) * 1.9)
            weightData[i] = Float16(cos(Float(i) * 0.0083) * 0.02)
        }

        let inputBuf = device.makeBuffer(
            bytes: inputData, length: dim * 2, options: .storageModeShared
        )!
        let weightBuf = device.makeBuffer(
            bytes: weightData, length: dim * 2, options: .storageModeShared
        )!
        let genericOut = device.makeBuffer(length: dim * 2, options: .storageModeShared)!
        let specializedOut = device.makeBuffer(length: dim * 2, options: .storageModeShared)!
        memset(genericOut.contents(), 0, dim * 2)
        memset(specializedOut.contents(), 0, dim * 2)

        let cmdBuf1 = queue.makeCommandBuffer()!
        let enc1 = cmdBuf1.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(genericPipeline!)
        enc1.setBuffer(inputBuf, offset: 0, index: 0)
        enc1.setBuffer(weightBuf, offset: 0, index: 1)
        enc1.setBuffer(genericOut, offset: 0, index: 2)
        var dimVal = UInt32(dim); enc1.setBytes(&dimVal, length: 4, index: 3)
        var epsVal = eps; enc1.setBytes(&epsVal, length: 4, index: 4)
        enc1.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)
        )
        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()
        XCTAssertNil(cmdBuf1.error)

        let cmdBuf2 = queue.makeCommandBuffer()!
        let enc2 = cmdBuf2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(specializedPipeline!)
        enc2.setBuffer(inputBuf, offset: 0, index: 0)
        enc2.setBuffer(weightBuf, offset: 0, index: 1)
        enc2.setBuffer(specializedOut, offset: 0, index: 2)
        enc2.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 320, height: 1, depth: 1)
        )
        enc2.endEncoding()
        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()
        XCTAssertNil(cmdBuf2.error)

        let genericPtr = genericOut.contents().bindMemory(to: Float16.self, capacity: dim)
        let specializedPtr = specializedOut.contents().bindMemory(to: Float16.self, capacity: dim)
        var maxDiff: Float = 0
        for i in 0..<dim {
            let diff = abs(Float(genericPtr[i]) - Float(specializedPtr[i]))
            if diff > maxDiff { maxDiff = diff }
        }

        fputs(
            "  rms_norm_1pw_d2560 vs generic:"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.002, "Specialized d2560 RMSNorm diverges from generic")
    }

    func testRmsNormDecodeD1536Add_MatchesGenericPlusResidual() throws {
        let referenceNormPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw_d1536"
        )
        try XCTSkipIf(
            referenceNormPipeline == nil,
            "Could not compile specialized d1536 RMSNorm shader"
        )

        let addPipeline = makePipeline(
            shaderFile: "activations.metal", functionName: "elementwise_add"
        )
        try XCTSkipIf(addPipeline == nil, "Could not compile elementwise add shader")

        let specializedPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw_d1536_add"
        )
        try XCTSkipIf(
            specializedPipeline == nil,
            "Could not compile specialized d1536 fused RMSNorm+add shader"
        )

        let dim = 1536
        var inputData = [Float16](repeating: 0, count: dim)
        var weightData = [Float16](repeating: 0, count: dim)
        var residualData = [Float16](repeating: 0, count: dim)
        for i in 0..<dim {
            inputData[i] = Float16(sin(Float(i) * 0.0043) * 1.4)
            weightData[i] = Float16(cos(Float(i) * 0.0101) * 0.03)
            residualData[i] = Float16(sin(Float(i) * 0.0067 + 0.5) * 0.8)
        }

        let inputBuf = device.makeBuffer(
            bytes: inputData, length: dim * 2, options: .storageModeShared
        )!
        let weightBuf = device.makeBuffer(
            bytes: weightData, length: dim * 2, options: .storageModeShared
        )!
        let residualBuf = device.makeBuffer(
            bytes: residualData, length: dim * 2, options: .storageModeShared
        )!
        let genericOut = device.makeBuffer(length: dim * 2, options: .storageModeShared)!
        let referenceOut = device.makeBuffer(length: dim * 2, options: .storageModeShared)!
        let specializedOut = device.makeBuffer(length: dim * 2, options: .storageModeShared)!
        let aliasedOut = device.makeBuffer(
            bytes: residualData, length: dim * 2, options: .storageModeShared
        )!
        memset(genericOut.contents(), 0, dim * 2)
        memset(referenceOut.contents(), 0, dim * 2)
        memset(specializedOut.contents(), 0, dim * 2)

        let cmdBuf1 = queue.makeCommandBuffer()!
        let enc1 = cmdBuf1.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(referenceNormPipeline!)
        enc1.setBuffer(inputBuf, offset: 0, index: 0)
        enc1.setBuffer(weightBuf, offset: 0, index: 1)
        enc1.setBuffer(genericOut, offset: 0, index: 2)
        enc1.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 192, height: 1, depth: 1)
        )
        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()
        XCTAssertNil(cmdBuf1.error)

        let cmdBuf2 = queue.makeCommandBuffer()!
        let enc2 = cmdBuf2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(addPipeline!)
        enc2.setBuffer(genericOut, offset: 0, index: 0)
        enc2.setBuffer(residualBuf, offset: 0, index: 1)
        enc2.setBuffer(referenceOut, offset: 0, index: 2)
        var countVal = UInt32(dim); enc2.setBytes(&countVal, length: 4, index: 3)
        let width = addPipeline!.threadExecutionWidth
        let threadsPerGrid = MTLSize(width: dim, height: 1, depth: 1)
        let threadsPerGroup = MTLSize(width: min(width, dim), height: 1, depth: 1)
        enc2.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerGroup)
        enc2.endEncoding()
        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()
        XCTAssertNil(cmdBuf2.error)

        let cmdBuf3 = queue.makeCommandBuffer()!
        let enc3 = cmdBuf3.makeComputeCommandEncoder()!
        enc3.setComputePipelineState(specializedPipeline!)
        enc3.setBuffer(inputBuf, offset: 0, index: 0)
        enc3.setBuffer(weightBuf, offset: 0, index: 1)
        enc3.setBuffer(residualBuf, offset: 0, index: 2)
        enc3.setBuffer(specializedOut, offset: 0, index: 3)
        enc3.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 192, height: 1, depth: 1)
        )
        enc3.endEncoding()
        cmdBuf3.commit()
        cmdBuf3.waitUntilCompleted()
        XCTAssertNil(cmdBuf3.error)

        let cmdBuf4 = queue.makeCommandBuffer()!
        let enc4 = cmdBuf4.makeComputeCommandEncoder()!
        enc4.setComputePipelineState(specializedPipeline!)
        enc4.setBuffer(inputBuf, offset: 0, index: 0)
        enc4.setBuffer(weightBuf, offset: 0, index: 1)
        enc4.setBuffer(aliasedOut, offset: 0, index: 2)
        enc4.setBuffer(aliasedOut, offset: 0, index: 3)
        enc4.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 192, height: 1, depth: 1)
        )
        enc4.endEncoding()
        cmdBuf4.commit()
        cmdBuf4.waitUntilCompleted()
        XCTAssertNil(cmdBuf4.error)

        let referencePtr = referenceOut.contents().bindMemory(to: Float16.self, capacity: dim)
        let specializedPtr = specializedOut.contents().bindMemory(to: Float16.self, capacity: dim)
        let aliasedPtr = aliasedOut.contents().bindMemory(to: Float16.self, capacity: dim)
        var maxDiff: Float = 0
        var maxAliasedDiff: Float = 0
        for i in 0..<dim {
            let diff = abs(Float(referencePtr[i]) - Float(specializedPtr[i]))
            if diff > maxDiff { maxDiff = diff }
            let aliasedDiff = abs(Float(referencePtr[i]) - Float(aliasedPtr[i]))
            if aliasedDiff > maxAliasedDiff { maxAliasedDiff = aliasedDiff }
        }

        let msg =
            "  rms_norm_1pw_d1536_add vs d1536+elementwise_add:"
                + " max diff = \(String(format: "%.6f", maxDiff))"
                + ", aliased max diff = \(String(format: "%.6f", maxAliasedDiff))\n"
        fputs(msg, stderr)
        XCTAssertLessThan(
            maxDiff,
            0.002,
            "Specialized d1536 fused RMSNorm+add diverges from generic reference"
        )
        XCTAssertLessThan(
            maxAliasedDiff,
            0.002,
            "Specialized d1536 fused RMSNorm+add diverges when residual/output alias"
        )
    }

    func testRmsNormDecodeD256Add_MatchesGenericPlusResidual() throws {
        let referenceNormPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw"
        )
        try XCTSkipIf(referenceNormPipeline == nil, "Could not compile generic RMSNorm shader")

        let addPipeline = makePipeline(
            shaderFile: "activations.metal", functionName: "elementwise_add"
        )
        try XCTSkipIf(addPipeline == nil, "Could not compile elementwise add shader")

        let fusedPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw_d256_add"
        )
        try XCTSkipIf(
            fusedPipeline == nil,
            "Could not compile specialized d256 fused RMSNorm+add shader"
        )

        let dim = 256
        var inputData = [Float16](repeating: 0, count: dim)
        var weightData = [Float16](repeating: 0, count: dim)
        var residualData = [Float16](repeating: 0, count: dim)
        for i in 0..<dim {
            inputData[i] = Float16(sin(Float(i) * 0.019) * 1.1)
            weightData[i] = Float16(cos(Float(i) * 0.017) * 0.025)
            residualData[i] = Float16(sin(Float(i) * 0.013 + 0.3) * 0.7)
        }

        let inputBuf = device.makeBuffer(bytes: inputData, length: dim * 2, options: .storageModeShared)!
        let weightBuf = device.makeBuffer(bytes: weightData, length: dim * 2, options: .storageModeShared)!
        let residualBuf = device.makeBuffer(bytes: residualData, length: dim * 2, options: .storageModeShared)!
        let normOut = device.makeBuffer(length: dim * 2, options: .storageModeShared)!
        let referenceOut = device.makeBuffer(length: dim * 2, options: .storageModeShared)!
        let fusedOut = device.makeBuffer(length: dim * 2, options: .storageModeShared)!
        let aliasedOut = device.makeBuffer(bytes: residualData, length: dim * 2, options: .storageModeShared)!
        memset(normOut.contents(), 0, dim * 2)
        memset(referenceOut.contents(), 0, dim * 2)
        memset(fusedOut.contents(), 0, dim * 2)

        let cmdBuf1 = queue.makeCommandBuffer()!
        let enc1 = cmdBuf1.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(referenceNormPipeline!)
        enc1.setBuffer(inputBuf, offset: 0, index: 0)
        enc1.setBuffer(weightBuf, offset: 0, index: 1)
        enc1.setBuffer(normOut, offset: 0, index: 2)
        var dimVal = UInt32(dim); enc1.setBytes(&dimVal, length: 4, index: 3)
        var epsVal: Float = 1e-6; enc1.setBytes(&epsVal, length: 4, index: 4)
        enc1.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()
        XCTAssertNil(cmdBuf1.error)

        let cmdBuf2 = queue.makeCommandBuffer()!
        let enc2 = cmdBuf2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(addPipeline!)
        enc2.setBuffer(normOut, offset: 0, index: 0)
        enc2.setBuffer(residualBuf, offset: 0, index: 1)
        enc2.setBuffer(referenceOut, offset: 0, index: 2)
        var countVal = UInt32(dim); enc2.setBytes(&countVal, length: 4, index: 3)
        enc2.dispatchThreads(
            MTLSize(width: dim, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc2.endEncoding()
        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()
        XCTAssertNil(cmdBuf2.error)

        let cmdBuf3 = queue.makeCommandBuffer()!
        let enc3 = cmdBuf3.makeComputeCommandEncoder()!
        enc3.setComputePipelineState(fusedPipeline!)
        enc3.setBuffer(inputBuf, offset: 0, index: 0)
        enc3.setBuffer(weightBuf, offset: 0, index: 1)
        enc3.setBuffer(residualBuf, offset: 0, index: 2)
        enc3.setBuffer(fusedOut, offset: 0, index: 3)
        enc3.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc3.endEncoding()
        cmdBuf3.commit()
        cmdBuf3.waitUntilCompleted()
        XCTAssertNil(cmdBuf3.error)

        let cmdBuf4 = queue.makeCommandBuffer()!
        let enc4 = cmdBuf4.makeComputeCommandEncoder()!
        enc4.setComputePipelineState(fusedPipeline!)
        enc4.setBuffer(inputBuf, offset: 0, index: 0)
        enc4.setBuffer(weightBuf, offset: 0, index: 1)
        enc4.setBuffer(aliasedOut, offset: 0, index: 2)
        enc4.setBuffer(aliasedOut, offset: 0, index: 3)
        enc4.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc4.endEncoding()
        cmdBuf4.commit()
        cmdBuf4.waitUntilCompleted()
        XCTAssertNil(cmdBuf4.error)

        let referencePtr = referenceOut.contents().bindMemory(to: Float16.self, capacity: dim)
        let fusedPtr = fusedOut.contents().bindMemory(to: Float16.self, capacity: dim)
        let aliasedPtr = aliasedOut.contents().bindMemory(to: Float16.self, capacity: dim)
        var maxDiff: Float = 0
        var maxAliasedDiff: Float = 0
        for i in 0..<dim {
            let diff = abs(Float(referencePtr[i]) - Float(fusedPtr[i]))
            if diff > maxDiff { maxDiff = diff }
            let aliasedDiff = abs(Float(referencePtr[i]) - Float(aliasedPtr[i]))
            if aliasedDiff > maxAliasedDiff { maxAliasedDiff = aliasedDiff }
        }

        fputs(
            "  rms_norm_1pw_d256_add vs generic+elementwise_add:"
                + " max diff = \(String(format: "%.6f", maxDiff))"
                + ", aliased max diff = \(String(format: "%.6f", maxAliasedDiff))\n",
            stderr
        )
        // The fused kernel changes the reduction/rounding boundary. Its
        // output may differ from the staged Float16 path by one ULP (1/512)
        // while preserving the same numerical contract.
        XCTAssertEqual(maxDiff, 0, accuracy: 0.002, "Specialized d256 fused RMSNorm+add diverges from generic reference")
        XCTAssertEqual(maxAliasedDiff, 0, accuracy: 0.002, "Specialized d256 fused RMSNorm+add diverges when residual/output alias")
    }

    func testRmsNormDecodeD256AddScalarWeight_MatchesStagedReference() throws {
        let referenceNormPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw"
        )
        try XCTSkipIf(referenceNormPipeline == nil, "Could not compile generic RMSNorm shader")

        let addPipeline = makePipeline(
            shaderFile: "activations.metal", functionName: "elementwise_add"
        )
        try XCTSkipIf(addPipeline == nil, "Could not compile elementwise add shader")

        let scalarPipeline = makePipeline(
            shaderFile: "activations.metal", functionName: "scalar_mul_weight"
        )
        try XCTSkipIf(scalarPipeline == nil, "Could not compile scalar weight multiply shader")

        let fusedPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw_d256_add_scalar_weight"
        )
        try XCTSkipIf(
            fusedPipeline == nil,
            "Could not compile specialized d256 fused RMSNorm+add+scalar shader"
        )

        let dim = 256
        var inputData = [Float16](repeating: 0, count: dim)
        var weightData = [Float16](repeating: 0, count: dim)
        var residualData = [Float16](repeating: 0, count: dim)
        let scalarData = [Float16](repeating: Float16(0.8125), count: 1)
        for i in 0..<dim {
            inputData[i] = Float16(sin(Float(i) * 0.021) * 1.3)
            weightData[i] = Float16(cos(Float(i) * 0.011) * 0.031)
            residualData[i] = Float16(sin(Float(i) * 0.015 + 0.7) * 0.5)
        }

        let inputBuf = device.makeBuffer(bytes: inputData, length: dim * 2, options: .storageModeShared)!
        let weightBuf = device.makeBuffer(bytes: weightData, length: dim * 2, options: .storageModeShared)!
        let residualBuf = device.makeBuffer(bytes: residualData, length: dim * 2, options: .storageModeShared)!
        let scalarBuf = device.makeBuffer(bytes: scalarData, length: 2, options: .storageModeShared)!
        let normOut = device.makeBuffer(length: dim * 2, options: .storageModeShared)!
        let addOut = device.makeBuffer(length: dim * 2, options: .storageModeShared)!
        let referenceOut = device.makeBuffer(length: dim * 2, options: .storageModeShared)!
        let fusedOut = device.makeBuffer(length: dim * 2, options: .storageModeShared)!
        let aliasedOut = device.makeBuffer(bytes: residualData, length: dim * 2, options: .storageModeShared)!
        memset(normOut.contents(), 0, dim * 2)
        memset(addOut.contents(), 0, dim * 2)
        memset(referenceOut.contents(), 0, dim * 2)
        memset(fusedOut.contents(), 0, dim * 2)

        let cmdBuf1 = queue.makeCommandBuffer()!
        let enc1 = cmdBuf1.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(referenceNormPipeline!)
        enc1.setBuffer(inputBuf, offset: 0, index: 0)
        enc1.setBuffer(weightBuf, offset: 0, index: 1)
        enc1.setBuffer(normOut, offset: 0, index: 2)
        var dimVal = UInt32(dim); enc1.setBytes(&dimVal, length: 4, index: 3)
        var epsVal: Float = 1e-6; enc1.setBytes(&epsVal, length: 4, index: 4)
        enc1.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()
        XCTAssertNil(cmdBuf1.error)

        let cmdBuf2 = queue.makeCommandBuffer()!
        let enc2 = cmdBuf2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(addPipeline!)
        enc2.setBuffer(normOut, offset: 0, index: 0)
        enc2.setBuffer(residualBuf, offset: 0, index: 1)
        enc2.setBuffer(addOut, offset: 0, index: 2)
        var countVal = UInt32(dim); enc2.setBytes(&countVal, length: 4, index: 3)
        enc2.dispatchThreads(
            MTLSize(width: dim, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc2.endEncoding()
        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()
        XCTAssertNil(cmdBuf2.error)

        let cmdBuf3 = queue.makeCommandBuffer()!
        let enc3 = cmdBuf3.makeComputeCommandEncoder()!
        enc3.setComputePipelineState(scalarPipeline!)
        enc3.setBuffer(addOut, offset: 0, index: 0)
        enc3.setBuffer(scalarBuf, offset: 0, index: 1)
        enc3.setBuffer(referenceOut, offset: 0, index: 2)
        var scalarCountVal = UInt32(dim); enc3.setBytes(&scalarCountVal, length: 4, index: 3)
        enc3.dispatchThreads(
            MTLSize(width: dim, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc3.endEncoding()
        cmdBuf3.commit()
        cmdBuf3.waitUntilCompleted()
        XCTAssertNil(cmdBuf3.error)

        let cmdBuf4 = queue.makeCommandBuffer()!
        let enc4 = cmdBuf4.makeComputeCommandEncoder()!
        enc4.setComputePipelineState(fusedPipeline!)
        enc4.setBuffer(inputBuf, offset: 0, index: 0)
        enc4.setBuffer(weightBuf, offset: 0, index: 1)
        enc4.setBuffer(residualBuf, offset: 0, index: 2)
        enc4.setBuffer(scalarBuf, offset: 0, index: 3)
        enc4.setBuffer(fusedOut, offset: 0, index: 4)
        enc4.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc4.endEncoding()
        cmdBuf4.commit()
        cmdBuf4.waitUntilCompleted()
        XCTAssertNil(cmdBuf4.error)

        let cmdBuf5 = queue.makeCommandBuffer()!
        let enc5 = cmdBuf5.makeComputeCommandEncoder()!
        enc5.setComputePipelineState(fusedPipeline!)
        enc5.setBuffer(inputBuf, offset: 0, index: 0)
        enc5.setBuffer(weightBuf, offset: 0, index: 1)
        enc5.setBuffer(aliasedOut, offset: 0, index: 2)
        enc5.setBuffer(scalarBuf, offset: 0, index: 3)
        enc5.setBuffer(aliasedOut, offset: 0, index: 4)
        enc5.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        enc5.endEncoding()
        cmdBuf5.commit()
        cmdBuf5.waitUntilCompleted()
        XCTAssertNil(cmdBuf5.error)

        let referencePtr = referenceOut.contents().bindMemory(to: Float16.self, capacity: dim)
        let fusedPtr = fusedOut.contents().bindMemory(to: Float16.self, capacity: dim)
        let aliasedPtr = aliasedOut.contents().bindMemory(to: Float16.self, capacity: dim)
        var maxDiff: Float = 0
        var maxAliasedDiff: Float = 0
        for i in 0..<dim {
            let diff = abs(Float(referencePtr[i]) - Float(fusedPtr[i]))
            if diff > maxDiff { maxDiff = diff }
            let aliasedDiff = abs(Float(referencePtr[i]) - Float(aliasedPtr[i]))
            if aliasedDiff > maxAliasedDiff { maxAliasedDiff = aliasedDiff }
        }

        fputs(
            "  rms_norm_1pw_d256_add_scalar_weight vs staged reference:"
                + " max diff = \(String(format: "%.6f", maxDiff))"
                + ", aliased max diff = \(String(format: "%.6f", maxAliasedDiff))\n",
            stderr
        )
        XCTAssertEqual(maxDiff, 0, accuracy: 0.002, "Specialized d256 fused RMSNorm+add+scalar diverges from staged reference")
        XCTAssertEqual(maxAliasedDiff, 0, accuracy: 0.002, "Specialized d256 fused RMSNorm+add+scalar diverges when residual/output alias")
    }

    func testRmsNormD1536BatchedAdd_MatchesBatchedNormPlusResidual() throws {
        let referenceNormPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw_d1536_batched"
        )
        try XCTSkipIf(
            referenceNormPipeline == nil,
            "Could not compile specialized batched d1536 RMSNorm shader"
        )

        let addPipeline = makePipeline(
            shaderFile: "activations.metal", functionName: "elementwise_add"
        )
        try XCTSkipIf(addPipeline == nil, "Could not compile elementwise add shader")

        let fusedPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw_d1536_add_batched"
        )
        try XCTSkipIf(
            fusedPipeline == nil,
            "Could not compile specialized batched d1536 fused RMSNorm+add shader"
        )

        let dim = 1536
        let batch = 5
        let count = dim * batch
        var inputData = [Float16](repeating: 0, count: count)
        var weightData = [Float16](repeating: 0, count: dim)
        var residualData = [Float16](repeating: 0, count: count)
        for b in 0..<batch {
            for i in 0..<dim {
                let idx = b * dim + i
                inputData[idx] = Float16(sin(Float(idx) * 0.0031) * 1.3)
                residualData[idx] = Float16(cos(Float(idx) * 0.0057 + 0.2) * 0.7)
            }
        }
        for i in 0..<dim {
            weightData[i] = Float16(cos(Float(i) * 0.0097) * 0.025)
        }

        let inputBuf = device.makeBuffer(
            bytes: inputData, length: count * 2, options: .storageModeShared
        )!
        let weightBuf = device.makeBuffer(
            bytes: weightData, length: dim * 2, options: .storageModeShared
        )!
        let residualBuf = device.makeBuffer(
            bytes: residualData, length: count * 2, options: .storageModeShared
        )!
        let normOut = device.makeBuffer(length: count * 2, options: .storageModeShared)!
        let referenceOut = device.makeBuffer(length: count * 2, options: .storageModeShared)!
        let fusedOut = device.makeBuffer(length: count * 2, options: .storageModeShared)!
        let aliasedOut = device.makeBuffer(
            bytes: residualData, length: count * 2, options: .storageModeShared
        )!
        memset(normOut.contents(), 0, count * 2)
        memset(referenceOut.contents(), 0, count * 2)
        memset(fusedOut.contents(), 0, count * 2)

        let cmdBuf1 = queue.makeCommandBuffer()!
        let enc1 = cmdBuf1.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(referenceNormPipeline!)
        enc1.setBuffer(inputBuf, offset: 0, index: 0)
        enc1.setBuffer(weightBuf, offset: 0, index: 1)
        enc1.setBuffer(normOut, offset: 0, index: 2)
        enc1.dispatchThreadgroups(
            MTLSize(width: batch, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 192, height: 1, depth: 1)
        )
        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()
        XCTAssertNil(cmdBuf1.error)

        let cmdBuf2 = queue.makeCommandBuffer()!
        let enc2 = cmdBuf2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(addPipeline!)
        enc2.setBuffer(normOut, offset: 0, index: 0)
        enc2.setBuffer(residualBuf, offset: 0, index: 1)
        enc2.setBuffer(referenceOut, offset: 0, index: 2)
        var countVal = UInt32(count)
        enc2.setBytes(&countVal, length: 4, index: 3)
        let width = addPipeline!.threadExecutionWidth
        enc2.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(width, count), height: 1, depth: 1)
        )
        enc2.endEncoding()
        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()
        XCTAssertNil(cmdBuf2.error)

        let cmdBuf3 = queue.makeCommandBuffer()!
        let enc3 = cmdBuf3.makeComputeCommandEncoder()!
        enc3.setComputePipelineState(fusedPipeline!)
        enc3.setBuffer(inputBuf, offset: 0, index: 0)
        enc3.setBuffer(weightBuf, offset: 0, index: 1)
        enc3.setBuffer(residualBuf, offset: 0, index: 2)
        enc3.setBuffer(fusedOut, offset: 0, index: 3)
        enc3.dispatchThreadgroups(
            MTLSize(width: batch, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 192, height: 1, depth: 1)
        )
        enc3.endEncoding()
        cmdBuf3.commit()
        cmdBuf3.waitUntilCompleted()
        XCTAssertNil(cmdBuf3.error)

        let cmdBuf4 = queue.makeCommandBuffer()!
        let enc4 = cmdBuf4.makeComputeCommandEncoder()!
        enc4.setComputePipelineState(fusedPipeline!)
        enc4.setBuffer(inputBuf, offset: 0, index: 0)
        enc4.setBuffer(weightBuf, offset: 0, index: 1)
        enc4.setBuffer(aliasedOut, offset: 0, index: 2)
        enc4.setBuffer(aliasedOut, offset: 0, index: 3)
        enc4.dispatchThreadgroups(
            MTLSize(width: batch, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 192, height: 1, depth: 1)
        )
        enc4.endEncoding()
        cmdBuf4.commit()
        cmdBuf4.waitUntilCompleted()
        XCTAssertNil(cmdBuf4.error)

        let referencePtr = referenceOut.contents().bindMemory(to: Float16.self, capacity: count)
        let fusedPtr = fusedOut.contents().bindMemory(to: Float16.self, capacity: count)
        let aliasedPtr = aliasedOut.contents().bindMemory(to: Float16.self, capacity: count)
        var maxDiff: Float = 0
        var maxAliasedDiff: Float = 0
        for i in 0..<count {
            let diff = abs(Float(referencePtr[i]) - Float(fusedPtr[i]))
            if diff > maxDiff { maxDiff = diff }
            let aliasedDiff = abs(Float(referencePtr[i]) - Float(aliasedPtr[i]))
            if aliasedDiff > maxAliasedDiff { maxAliasedDiff = aliasedDiff }
        }

        fputs(
            "  rms_norm_1pw_d1536_add_batched vs batched d1536+elementwise_add:"
                + " max diff = \(String(format: "%.6f", maxDiff))"
                + ", aliased max diff = \(String(format: "%.6f", maxAliasedDiff))\n",
            stderr
        )
        XCTAssertLessThan(
            maxDiff,
            0.002,
            "Specialized batched d1536 fused RMSNorm+add diverges from staged reference"
        )
        XCTAssertLessThan(
            maxAliasedDiff,
            0.002,
            "Specialized batched d1536 fused RMSNorm+add diverges when residual/output alias"
        )
    }

    func testRmsNormDecodeD1536Add_WithZeroResidualMatchesNormExactly() throws {
        let normPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw_d1536"
        )
        try XCTSkipIf(normPipeline == nil, "Could not compile specialized d1536 RMSNorm shader")

        let fusedPipeline = makePipeline(
            shaderFile: "norms.metal", functionName: "rms_norm_1pw_d1536_add"
        )
        try XCTSkipIf(fusedPipeline == nil, "Could not compile specialized d1536 fused RMSNorm+add shader")

        let dim = 1536
        var inputData = [Float16](repeating: 0, count: dim)
        var weightData = [Float16](repeating: 0, count: dim)
        let residualData = [Float16](repeating: 0, count: dim)
        for i in 0..<dim {
            inputData[i] = Float16(sin(Float(i) * 0.0043) * 1.4)
            weightData[i] = Float16(cos(Float(i) * 0.0101) * 0.03)
        }

        let inputBuf = device.makeBuffer(bytes: inputData, length: dim * 2, options: .storageModeShared)!
        let weightBuf = device.makeBuffer(bytes: weightData, length: dim * 2, options: .storageModeShared)!
        let residualBuf = device.makeBuffer(bytes: residualData, length: dim * 2, options: .storageModeShared)!
        let normOut = device.makeBuffer(length: dim * 2, options: .storageModeShared)!
        let fusedOut = device.makeBuffer(length: dim * 2, options: .storageModeShared)!
        memset(normOut.contents(), 0, dim * 2)
        memset(fusedOut.contents(), 0, dim * 2)

        let cmdBuf1 = queue.makeCommandBuffer()!
        let enc1 = cmdBuf1.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(normPipeline!)
        enc1.setBuffer(inputBuf, offset: 0, index: 0)
        enc1.setBuffer(weightBuf, offset: 0, index: 1)
        enc1.setBuffer(normOut, offset: 0, index: 2)
        enc1.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 192, height: 1, depth: 1)
        )
        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()
        XCTAssertNil(cmdBuf1.error)

        let cmdBuf2 = queue.makeCommandBuffer()!
        let enc2 = cmdBuf2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(fusedPipeline!)
        enc2.setBuffer(inputBuf, offset: 0, index: 0)
        enc2.setBuffer(weightBuf, offset: 0, index: 1)
        enc2.setBuffer(residualBuf, offset: 0, index: 2)
        enc2.setBuffer(fusedOut, offset: 0, index: 3)
        enc2.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 192, height: 1, depth: 1)
        )
        enc2.endEncoding()
        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()
        XCTAssertNil(cmdBuf2.error)

        let normPtr = normOut.contents().bindMemory(to: Float16.self, capacity: dim)
        let fusedPtr = fusedOut.contents().bindMemory(to: Float16.self, capacity: dim)
        var maxDiff: Float = 0
        var firstMismatch: Int? = nil
        for i in 0..<dim {
            let diff = abs(Float(normPtr[i]) - Float(fusedPtr[i]))
            if diff > maxDiff { maxDiff = diff }
            if diff > 0 && firstMismatch == nil { firstMismatch = i }
        }

        if let i = firstMismatch {
            fputs(
                "  rms_norm_1pw_d1536_add zero residual first mismatch[\(i)]: norm=\(normPtr[i]) fused=\(fusedPtr[i])\n",
                stderr
            )
        }
        let zeroResidualMsg =
            "  rms_norm_1pw_d1536_add zero residual vs norm:"
                + " max diff = \(String(format: "%.6f", maxDiff))\n"
        fputs(zeroResidualMsg, stderr)
        XCTAssertEqual(maxDiff, 0, accuracy: 0.000001, "Specialized d1536 fused RMSNorm+add diverges from norm-only path with zero residual")
    }

    func testFusedAffineGateUpSwigluQwenSpecialization_MatchesGeneric() throws {
        let cases: [(rows: Int, cols: Int, functionName: String, rowTile: Int, tgWidth: Int)] = [
            (3584, 1024, "fused_affine_gate_up_swiglu_c1024_r3584_g64_rows4", 4, 64),
            (6144, 2048, "fused_affine_gate_up_swiglu_c2048_r6144_g64", 4, 64),
        ]

        for (rows, cols, functionName, rowTile, tgWidth) in cases {
            let genericPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: "fused_affine_gate_up_swiglu",
                cols: UInt32(cols),
                groupSize: 64
            )
            try XCTSkipIf(genericPipeline == nil, "Could not compile generic fused affine FFN shader")

            let specializedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: functionName
            )
            try XCTSkipIf(
                specializedPipeline == nil,
                "Could not compile specialized fused affine FFN shader"
            )

            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64)
            let gateW = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let gateS = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let gateB = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let upW = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let upS = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let upB = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let inputBuf = device.makeBuffer(bytes: data.input, length: cols * 2, options: .storageModeShared)!
            let genericOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            let specializedOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            memset(genericOut.contents(), 0, rows * 2)
            memset(specializedOut.contents(), 0, rows * 2)

            let cmdBuf1 = queue.makeCommandBuffer()!
            let enc1 = cmdBuf1.makeComputeCommandEncoder()!
            enc1.setComputePipelineState(genericPipeline!)
            enc1.setBuffer(gateW, offset: 0, index: 0)
            enc1.setBuffer(gateS, offset: 0, index: 1)
            enc1.setBuffer(gateB, offset: 0, index: 2)
            enc1.setBuffer(upW, offset: 0, index: 3)
            enc1.setBuffer(upS, offset: 0, index: 4)
            enc1.setBuffer(upB, offset: 0, index: 5)
            enc1.setBuffer(inputBuf, offset: 0, index: 6)
            enc1.setBuffer(genericOut, offset: 0, index: 7)
            var rowsVal = UInt32(rows); enc1.setBytes(&rowsVal, length: 4, index: 8)
            enc1.dispatchThreadgroups(
                MTLSize(width: rows / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc1.endEncoding()
            cmdBuf1.commit()
            cmdBuf1.waitUntilCompleted()
            XCTAssertNil(cmdBuf1.error)

            let cmdBuf2 = queue.makeCommandBuffer()!
            let enc2 = cmdBuf2.makeComputeCommandEncoder()!
            enc2.setComputePipelineState(specializedPipeline!)
            enc2.setBuffer(gateW, offset: 0, index: 0)
            enc2.setBuffer(gateS, offset: 0, index: 1)
            enc2.setBuffer(gateB, offset: 0, index: 2)
            enc2.setBuffer(upW, offset: 0, index: 3)
            enc2.setBuffer(upS, offset: 0, index: 4)
            enc2.setBuffer(upB, offset: 0, index: 5)
            enc2.setBuffer(inputBuf, offset: 0, index: 6)
            enc2.setBuffer(specializedOut, offset: 0, index: 7)
            enc2.dispatchThreadgroups(
                MTLSize(width: (rows + rowTile - 1) / rowTile, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1)
            )
            enc2.endEncoding()
            cmdBuf2.commit()
            cmdBuf2.waitUntilCompleted()
            XCTAssertNil(cmdBuf2.error)

            let genericPtr = genericOut.contents().bindMemory(to: Float16.self, capacity: rows)
            let specializedPtr = specializedOut.contents().bindMemory(to: Float16.self, capacity: rows)
            var maxDiff: Float = 0
            for i in 0..<rows {
                let diff = abs(Float(genericPtr[i]) - Float(specializedPtr[i]))
                if diff > maxDiff { maxDiff = diff }
            }

            fputs(
                "  \(functionName) vs generic:"
                    + " max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.002, "\(functionName) diverges from generic")
        }
    }

    func testFusedAffineGateUpSwigluVectorCache_MatchesNoCacheBits() throws {
        let cases: [(rows: Int, cols: Int, cached: String, reference: String)] = [
            (
                3_584, 1_024,
                "fused_affine_gate_up_swiglu_c1024_r3584_g64_rows4",
                "fused_affine_gate_up_swiglu_c1024_r3584_g64_rows4_nocache_reference"
            ),
            (
                6_144, 2_048,
                "fused_affine_gate_up_swiglu_c2048_r6144_g64",
                "fused_affine_gate_up_swiglu_c2048_r6144_g64_rows4_nocache_reference"
            ),
        ]
        let functionNames = cases.flatMap { [$0.cached, $0.reference] }
        guard let pipelines = makePipelines(
            shaderFile: "lut_matvec.metal",
            functionNames: functionNames
        ) else {
            throw XCTSkip("Could not compile vector-cache gate/up parity pipelines")
        }

        for (rows, cols, cachedName, referenceName) in cases {
            let gate = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 61)
            let up = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 67)
            let gateW = device.makeBuffer(bytes: gate.weights, length: gate.weights.count, options: .storageModeShared)!
            let gateS = device.makeBuffer(bytes: gate.scales, length: gate.scales.count * 2, options: .storageModeShared)!
            let gateB = device.makeBuffer(bytes: gate.biases, length: gate.biases.count * 2, options: .storageModeShared)!
            let upW = device.makeBuffer(bytes: up.weights, length: up.weights.count, options: .storageModeShared)!
            let upS = device.makeBuffer(bytes: up.scales, length: up.scales.count * 2, options: .storageModeShared)!
            let upB = device.makeBuffer(bytes: up.biases, length: up.biases.count * 2, options: .storageModeShared)!
            let input = device.makeBuffer(bytes: gate.input, length: cols * 2, options: .storageModeShared)!
            let cachedOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            let referenceOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!

            func run(_ pipeline: MTLComputePipelineState, output: MTLBuffer) throws {
                memset(output.contents(), 0, rows * 2)
                let commandBuffer = queue.makeCommandBuffer()!
                let encoder = commandBuffer.makeComputeCommandEncoder()!
                encoder.setComputePipelineState(pipeline)
                for (index, buffer) in [
                    0: gateW, 1: gateS, 2: gateB,
                    3: upW, 4: upS, 5: upB,
                    6: input, 7: output,
                ] {
                    encoder.setBuffer(buffer, offset: 0, index: index)
                }
                encoder.dispatchThreadgroups(
                    MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
                )
                encoder.endEncoding()
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
                if let error = commandBuffer.error { throw error }
            }

            try run(pipelines[referenceName]!, output: referenceOut)
            try run(pipelines[cachedName]!, output: cachedOut)
            let mismatch = firstHalfBitMismatch(referenceOut, cachedOut, count: rows)
            XCTAssertNil(
                mismatch,
                "\(cachedName) changed FP16 bits at row \(mismatch.map(String.init) ?? "<none>")"
            )
        }
    }

    func testGenericFusedAffineGateUpGeGLUShapes_MatchesIterated() throws {
        let cases: [(rows: Int, cols: Int)] = [
            (6144, 1536),
            (12288, 1536),
        ]

        for (rows, cols) in cases {
            let pipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: "fused_affine_gate_up_geglu",
                cols: UInt32(cols),
                groupSize: 128
            )
            try XCTSkipIf(pipeline == nil, "Could not compile generic fused affine GeGLU shader")

            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 128)
            let gateW = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let gateS = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let gateB = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let upW = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let upS = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let upB = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!

            let gpu = try runIteratedFusedAffineGateUpGeGLU(
                pipeline: pipeline!,
                gateWeights: gateW,
                gateScales: gateS,
                gateBiases: gateB,
                upWeights: upW,
                upScales: upS,
                upBiases: upB,
                inputData: data.input,
                rows: rows,
                cols: cols,
                batchSize: 1
            )
            let reference = try runIteratedAffineGateUpGeGLUReference(
                gateWeights: gateW,
                gateScales: gateS,
                gateBiases: gateB,
                upWeights: upW,
                upScales: upS,
                upBiases: upB,
                inputData: data.input,
                rows: rows,
                cols: cols,
                groupSize: 128,
                batchSize: 1
            )

            var maxDiff: Float = 0
            for i in 0..<rows {
                let diff = abs(gpu[i] - reference[i])
                if diff > maxDiff { maxDiff = diff }
            }

            fputs(
                "  fused_affine_gate_up_geglu rows=\(rows) cols=\(cols) vs iterated:"
                    + " max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(
                maxDiff,
                0.002,
                "generic fused_affine_gate_up_geglu diverges from iterated reference"
            )
        }
    }

    func testFusedAffineGateUpGeGLUSpecialization_MatchesGeneric() throws {
        let cases: [(rows: Int, functionName: String)] = [
            (6144, "fused_affine_gate_up_geglu_c1536_r6144_g128_rows4"),
            (12288, "fused_affine_gate_up_geglu_c1536_r12288_g128_rows4"),
            (6144, "fused_affine_gate_up_geglu_c1536_r6144_g128_rows8"),
            (12288, "fused_affine_gate_up_geglu_c1536_r12288_g128_rows8"),
        ]

        for (rows, functionName) in cases {
            let genericPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: "fused_affine_gate_up_geglu",
                cols: 1536,
                groupSize: 128
            )
            try XCTSkipIf(genericPipeline == nil, "Could not compile generic fused affine GeGLU shader")

            let specializedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: functionName
            )
            try XCTSkipIf(specializedPipeline == nil, "Could not compile \(functionName)")

            let cols = 1536
            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 128)
            let gateW = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let gateS = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let gateB = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let upW = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let upS = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let upB = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let inputBuf = device.makeBuffer(bytes: data.input, length: cols * 2, options: .storageModeShared)!
            let genericOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            let specializedOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            memset(genericOut.contents(), 0, rows * 2)
            memset(specializedOut.contents(), 0, rows * 2)

            let cmdBuf1 = queue.makeCommandBuffer()!
            let enc1 = cmdBuf1.makeComputeCommandEncoder()!
            enc1.setComputePipelineState(genericPipeline!)
            enc1.setBuffer(gateW, offset: 0, index: 0)
            enc1.setBuffer(gateS, offset: 0, index: 1)
            enc1.setBuffer(gateB, offset: 0, index: 2)
            enc1.setBuffer(upW, offset: 0, index: 3)
            enc1.setBuffer(upS, offset: 0, index: 4)
            enc1.setBuffer(upB, offset: 0, index: 5)
            enc1.setBuffer(inputBuf, offset: 0, index: 6)
            enc1.setBuffer(genericOut, offset: 0, index: 7)
            var rowsVal = UInt32(rows); enc1.setBytes(&rowsVal, length: 4, index: 8)
            enc1.dispatchThreadgroups(
                MTLSize(width: rows / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc1.endEncoding()
            cmdBuf1.commit()
            cmdBuf1.waitUntilCompleted()
            XCTAssertNil(cmdBuf1.error)

            let cmdBuf2 = queue.makeCommandBuffer()!
            let enc2 = cmdBuf2.makeComputeCommandEncoder()!
            enc2.setComputePipelineState(specializedPipeline!)
            enc2.setBuffer(gateW, offset: 0, index: 0)
            enc2.setBuffer(gateS, offset: 0, index: 1)
            enc2.setBuffer(gateB, offset: 0, index: 2)
            enc2.setBuffer(upW, offset: 0, index: 3)
            enc2.setBuffer(upS, offset: 0, index: 4)
            enc2.setBuffer(upB, offset: 0, index: 5)
            enc2.setBuffer(inputBuf, offset: 0, index: 6)
            enc2.setBuffer(specializedOut, offset: 0, index: 7)
            enc2.dispatchThreadgroups(
                MTLSize(width: rows / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc2.endEncoding()
            cmdBuf2.commit()
            cmdBuf2.waitUntilCompleted()
            XCTAssertNil(cmdBuf2.error)

            let genericPtr = genericOut.contents().bindMemory(to: Float16.self, capacity: rows)
            let specializedPtr = specializedOut.contents().bindMemory(to: Float16.self, capacity: rows)
            var maxDiff: Float = 0
            for i in 0..<rows {
                let diff = abs(Float(genericPtr[i]) - Float(specializedPtr[i]))
                if diff > maxDiff { maxDiff = diff }
            }

            fputs(
                "  \(functionName) vs generic:"
                    + " max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.002, "\(functionName) diverges from generic")
        }
    }

    func testFusedAffineMatvecAddSpecializations_MatchStagedAffinePlusResidual() throws {
        let addPipeline = makePipeline(
            shaderFile: "activations.metal",
            functionName: "elementwise_add"
        )
        try XCTSkipIf(addPipeline == nil, "Could not compile elementwise_add")

        let cases: [(cols: Int, staged: String, fused: String)] = [
            (2048, "affine_matvec_c2048_r2048_g64", "fused_affine_matvec_add_c2048_r2048_g64"),
            (6144, "affine_matvec_c6144_r2048_g64", "fused_affine_matvec_add_c6144_r2048_g64"),
        ]

        for (cols, stagedName, fusedName) in cases {
            if fusedName == "fused_affine_matvec_add_c2048_r2048_g64" {
                XCTExpectFailure(
                    "Qwen c2048_r2048_g64 fused residual path is still not exact against staged qmm affine + elementwise_add"
                )
            }

            let stagedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: stagedName
            )
            try XCTSkipIf(stagedPipeline == nil, "Could not compile \(stagedName)")

            let fusedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: fusedName
            )
            try XCTSkipIf(fusedPipeline == nil, "Could not compile \(fusedName)")

            let rows = 2048
            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64)
            let weightsBuf = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let scalesBuf = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let biasesBuf = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let inputBuf = device.makeBuffer(bytes: data.input, length: cols * 2, options: .storageModeShared)!
            let residualBuf = device.makeBuffer(bytes: data.residual, length: rows * 2, options: .storageModeShared)!
            let stagedMatvecOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            let stagedOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            let fusedOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            memset(stagedMatvecOut.contents(), 0, rows * 2)
            memset(stagedOut.contents(), 0, rows * 2)
            memset(fusedOut.contents(), 0, rows * 2)

            let stagedCmdBuf = queue.makeCommandBuffer()!
            let stagedEnc = stagedCmdBuf.makeComputeCommandEncoder()!
            stagedEnc.setComputePipelineState(stagedPipeline!)
            stagedEnc.setBuffer(weightsBuf, offset: 0, index: 0)
            stagedEnc.setBuffer(scalesBuf, offset: 0, index: 1)
            stagedEnc.setBuffer(biasesBuf, offset: 0, index: 2)
            stagedEnc.setBuffer(inputBuf, offset: 0, index: 3)
            stagedEnc.setBuffer(stagedMatvecOut, offset: 0, index: 4)
            stagedEnc.dispatchThreadgroups(
                MTLSize(width: rows / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            stagedEnc.endEncoding()

            let addEnc = stagedCmdBuf.makeComputeCommandEncoder()!
            addEnc.setComputePipelineState(addPipeline!)
            addEnc.setBuffer(stagedMatvecOut, offset: 0, index: 0)
            addEnc.setBuffer(residualBuf, offset: 0, index: 1)
            addEnc.setBuffer(stagedOut, offset: 0, index: 2)
            var rowsVal = UInt32(rows)
            addEnc.setBytes(&rowsVal, length: 4, index: 3)
            let addTgWidth = min(addPipeline!.maxTotalThreadsPerThreadgroup, rows)
            addEnc.dispatchThreads(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: addTgWidth, height: 1, depth: 1)
            )
            addEnc.endEncoding()
            stagedCmdBuf.commit()
            stagedCmdBuf.waitUntilCompleted()
            XCTAssertNil(stagedCmdBuf.error)

            let fusedCmdBuf = queue.makeCommandBuffer()!
            let fusedEnc = fusedCmdBuf.makeComputeCommandEncoder()!
            fusedEnc.setComputePipelineState(fusedPipeline!)
            fusedEnc.setBuffer(weightsBuf, offset: 0, index: 0)
            fusedEnc.setBuffer(scalesBuf, offset: 0, index: 1)
            fusedEnc.setBuffer(biasesBuf, offset: 0, index: 2)
            fusedEnc.setBuffer(inputBuf, offset: 0, index: 3)
            fusedEnc.setBuffer(fusedOut, offset: 0, index: 4)
            fusedEnc.setBuffer(residualBuf, offset: 0, index: 5)
            fusedEnc.dispatchThreadgroups(
                MTLSize(width: rows / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            fusedEnc.endEncoding()
            fusedCmdBuf.commit()
            fusedCmdBuf.waitUntilCompleted()
            XCTAssertNil(fusedCmdBuf.error)

            let stagedPtr = stagedOut.contents().bindMemory(to: Float16.self, capacity: rows)
            let fusedPtr = fusedOut.contents().bindMemory(to: Float16.self, capacity: rows)
            var maxDiff: Float = 0
            for i in 0..<rows {
                let diff = abs(Float(stagedPtr[i]) - Float(fusedPtr[i]))
                if diff > maxDiff { maxDiff = diff }
            }

            fputs(
                "  \(fusedName) vs staged affine+add: max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertEqual(maxDiff, 0, accuracy: 0.000001, "\(fusedName) diverges from staged affine+add")
        }
    }

    func testFusedAffineMatvecAddDecode_MatchesStagedAffinePlusResidual() throws {
        let addPipeline = makePipeline(
            shaderFile: "activations.metal",
            functionName: "elementwise_add"
        )
        try XCTSkipIf(addPipeline == nil, "Could not compile elementwise_add")

        let cases: [(rows: Int, cols: Int)] = [
            (1536, 2048),
            (1536, 4096),
            (1536, 6144),
            (1536, 12288),
        ]

        XCTExpectFailure(
            "Generic fused_affine_matvec_add is not yet exact against staged affine_matvec + elementwise_add for g128 decode shapes"
        )

        for (rows, cols) in cases {
            let stagedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: "affine_matvec",
                cols: UInt32(cols),
                groupSize: 128
            )
            try XCTSkipIf(stagedPipeline == nil, "Could not compile affine_matvec for cols=\(cols)")

            let fusedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: "fused_affine_matvec_add",
                cols: UInt32(cols),
                groupSize: 128
            )
            try XCTSkipIf(fusedPipeline == nil, "Could not compile fused_affine_matvec_add for cols=\(cols)")

            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 128, seed: 29)
            let weightsBuf = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let scalesBuf = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let biasesBuf = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let inputBuf = device.makeBuffer(bytes: data.input, length: cols * 2, options: .storageModeShared)!
            let residualBuf = device.makeBuffer(bytes: data.residual, length: rows * 2, options: .storageModeShared)!
            let stagedMatvecOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            let stagedOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            let fusedOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            memset(stagedMatvecOut.contents(), 0, rows * 2)
            memset(stagedOut.contents(), 0, rows * 2)
            memset(fusedOut.contents(), 0, rows * 2)

            let stagedCmdBuf = queue.makeCommandBuffer()!
            let stagedEnc = stagedCmdBuf.makeComputeCommandEncoder()!
            stagedEnc.setComputePipelineState(stagedPipeline!)
            stagedEnc.setBuffer(weightsBuf, offset: 0, index: 0)
            stagedEnc.setBuffer(scalesBuf, offset: 0, index: 1)
            stagedEnc.setBuffer(biasesBuf, offset: 0, index: 2)
            stagedEnc.setBuffer(inputBuf, offset: 0, index: 3)
            stagedEnc.setBuffer(stagedMatvecOut, offset: 0, index: 4)
            var rowsVal = UInt32(rows); stagedEnc.setBytes(&rowsVal, length: 4, index: 5)
            stagedEnc.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            stagedEnc.endEncoding()

            let addEnc = stagedCmdBuf.makeComputeCommandEncoder()!
            addEnc.setComputePipelineState(addPipeline!)
            addEnc.setBuffer(stagedMatvecOut, offset: 0, index: 0)
            addEnc.setBuffer(residualBuf, offset: 0, index: 1)
            addEnc.setBuffer(stagedOut, offset: 0, index: 2)
            addEnc.setBytes(&rowsVal, length: 4, index: 3)
            let addTgWidth = min(addPipeline!.maxTotalThreadsPerThreadgroup, rows)
            addEnc.dispatchThreads(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: addTgWidth, height: 1, depth: 1)
            )
            addEnc.endEncoding()
            stagedCmdBuf.commit()
            stagedCmdBuf.waitUntilCompleted()
            XCTAssertNil(stagedCmdBuf.error)

            let fusedCmdBuf = queue.makeCommandBuffer()!
            let fusedEnc = fusedCmdBuf.makeComputeCommandEncoder()!
            fusedEnc.setComputePipelineState(fusedPipeline!)
            fusedEnc.setBuffer(weightsBuf, offset: 0, index: 0)
            fusedEnc.setBuffer(scalesBuf, offset: 0, index: 1)
            fusedEnc.setBuffer(biasesBuf, offset: 0, index: 2)
            fusedEnc.setBuffer(inputBuf, offset: 0, index: 3)
            fusedEnc.setBuffer(fusedOut, offset: 0, index: 4)
            fusedEnc.setBuffer(residualBuf, offset: 0, index: 5)
            fusedEnc.setBytes(&rowsVal, length: 4, index: 6)
            fusedEnc.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            fusedEnc.endEncoding()
            fusedCmdBuf.commit()
            fusedCmdBuf.waitUntilCompleted()
            XCTAssertNil(fusedCmdBuf.error)

            let stagedPtr = stagedOut.contents().bindMemory(to: Float16.self, capacity: rows)
            let fusedPtr = fusedOut.contents().bindMemory(to: Float16.self, capacity: rows)
            var maxDiff: Float = 0
            for i in 0..<rows {
                let diff = abs(Float(stagedPtr[i]) - Float(fusedPtr[i]))
                if diff > maxDiff { maxDiff = diff }
            }

            fputs(
                "  fused_affine_matvec_add cols=\(cols) vs staged affine+add:"
                    + " max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertEqual(maxDiff, 0, accuracy: 0.000001, "cols=\(cols) diverges from staged affine+add")
        }
    }

    func testFusedAffineMatvecAddDecodeSpecializations_MatchStagedSpecializedAffinePlusResidual() throws {
        let addPipeline = makePipeline(
            shaderFile: "activations.metal",
            functionName: "elementwise_add"
        )
        try XCTSkipIf(addPipeline == nil, "Could not compile elementwise_add")

        let cases: [(rows: Int, cols: Int, staged: String, fused: String, width: Int)] = [
            (1536, 2048, "affine_matvec_c2048_r1536_g128_rows4", "fused_affine_matvec_add_c2048_r1536_g128_rows4", 384),
            (1536, 4096, "affine_matvec_c4096_r1536_g128", "fused_affine_matvec_add_c4096_r1536_g128", 192),
            (1536, 6144, "affine_matvec_c6144_r1536_g128_rows4", "fused_affine_matvec_add_c6144_r1536_g128_rows4", 384),
            (1536, 12288, "affine_matvec_c12288_r1536_g128_rows4", "fused_affine_matvec_add_c12288_r1536_g128_rows4", 384),
        ]

        for (rows, cols, stagedName, fusedName, width) in cases {
            let stagedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: stagedName
            )
            try XCTSkipIf(stagedPipeline == nil, "Could not compile \(stagedName)")

            let fusedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: fusedName
            )
            try XCTSkipIf(fusedPipeline == nil, "Could not compile \(fusedName)")

            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 128, seed: 31)
            let weightsBuf = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let scalesBuf = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let biasesBuf = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let inputBuf = device.makeBuffer(bytes: data.input, length: cols * 2, options: .storageModeShared)!
            let residualBuf = device.makeBuffer(bytes: data.residual, length: rows * 2, options: .storageModeShared)!
            let stagedMatvecOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            let stagedOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            let fusedOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            memset(stagedMatvecOut.contents(), 0, rows * 2)
            memset(stagedOut.contents(), 0, rows * 2)
            memset(fusedOut.contents(), 0, rows * 2)

            let stagedCmdBuf = queue.makeCommandBuffer()!
            let stagedEnc = stagedCmdBuf.makeComputeCommandEncoder()!
            stagedEnc.setComputePipelineState(stagedPipeline!)
            stagedEnc.setBuffer(weightsBuf, offset: 0, index: 0)
            stagedEnc.setBuffer(scalesBuf, offset: 0, index: 1)
            stagedEnc.setBuffer(biasesBuf, offset: 0, index: 2)
            stagedEnc.setBuffer(inputBuf, offset: 0, index: 3)
            stagedEnc.setBuffer(stagedMatvecOut, offset: 0, index: 4)
            stagedEnc.dispatchThreadgroups(
                MTLSize(width: width, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            stagedEnc.endEncoding()

            let addEnc = stagedCmdBuf.makeComputeCommandEncoder()!
            addEnc.setComputePipelineState(addPipeline!)
            addEnc.setBuffer(stagedMatvecOut, offset: 0, index: 0)
            addEnc.setBuffer(residualBuf, offset: 0, index: 1)
            addEnc.setBuffer(stagedOut, offset: 0, index: 2)
            var rowsVal = UInt32(rows)
            addEnc.setBytes(&rowsVal, length: 4, index: 3)
            let addTgWidth = min(addPipeline!.maxTotalThreadsPerThreadgroup, rows)
            addEnc.dispatchThreads(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: addTgWidth, height: 1, depth: 1)
            )
            addEnc.endEncoding()
            stagedCmdBuf.commit()
            stagedCmdBuf.waitUntilCompleted()
            XCTAssertNil(stagedCmdBuf.error)

            let fusedCmdBuf = queue.makeCommandBuffer()!
            let fusedEnc = fusedCmdBuf.makeComputeCommandEncoder()!
            fusedEnc.setComputePipelineState(fusedPipeline!)
            fusedEnc.setBuffer(weightsBuf, offset: 0, index: 0)
            fusedEnc.setBuffer(scalesBuf, offset: 0, index: 1)
            fusedEnc.setBuffer(biasesBuf, offset: 0, index: 2)
            fusedEnc.setBuffer(inputBuf, offset: 0, index: 3)
            fusedEnc.setBuffer(fusedOut, offset: 0, index: 4)
            fusedEnc.setBuffer(residualBuf, offset: 0, index: 5)
            fusedEnc.dispatchThreadgroups(
                MTLSize(width: width, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            fusedEnc.endEncoding()
            fusedCmdBuf.commit()
            fusedCmdBuf.waitUntilCompleted()
            XCTAssertNil(fusedCmdBuf.error)

            let stagedPtr = stagedOut.contents().bindMemory(to: Float16.self, capacity: rows)
            let fusedPtr = fusedOut.contents().bindMemory(to: Float16.self, capacity: rows)
            var maxDiff: Float = 0
            for i in 0..<rows {
                let diff = abs(Float(stagedPtr[i]) - Float(fusedPtr[i]))
                if diff > maxDiff { maxDiff = diff }
            }

            fputs(
                "  \(fusedName) vs staged specialized affine+add:"
                    + " max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertEqual(maxDiff, 0, accuracy: 0.000001, "\(fusedName) diverges from staged specialized affine+add")
        }
    }

    func testFusedAffineMatvecAddDecodeSpecializations_WithZeroResidualMatchSpecializedAffine() throws {
        let cases: [(rows: Int, cols: Int, staged: String, fused: String, width: Int)] = [
            (1536, 2048, "affine_matvec_c2048_r1536_g128_rows4", "fused_affine_matvec_add_c2048_r1536_g128_rows4", 384),
            (1536, 4096, "affine_matvec_c4096_r1536_g128", "fused_affine_matvec_add_c4096_r1536_g128", 192),
            (1536, 6144, "affine_matvec_c6144_r1536_g128_rows4", "fused_affine_matvec_add_c6144_r1536_g128_rows4", 384),
            (1536, 12288, "affine_matvec_c12288_r1536_g128_rows4", "fused_affine_matvec_add_c12288_r1536_g128_rows4", 384),
        ]

        for (rows, cols, stagedName, fusedName, width) in cases {
            let stagedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: stagedName
            )
            try XCTSkipIf(stagedPipeline == nil, "Could not compile \(stagedName)")

            let fusedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: fusedName
            )
            try XCTSkipIf(fusedPipeline == nil, "Could not compile \(fusedName)")

            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 128, seed: 37)
            let zeroResidual = [Float16](repeating: 0, count: rows)
            let weightsBuf = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let scalesBuf = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let biasesBuf = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let inputBuf = device.makeBuffer(bytes: data.input, length: cols * 2, options: .storageModeShared)!
            let residualBuf = device.makeBuffer(bytes: zeroResidual, length: rows * 2, options: .storageModeShared)!
            let stagedOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            let fusedOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            memset(stagedOut.contents(), 0, rows * 2)
            memset(fusedOut.contents(), 0, rows * 2)

            let stagedCmdBuf = queue.makeCommandBuffer()!
            let stagedEnc = stagedCmdBuf.makeComputeCommandEncoder()!
            stagedEnc.setComputePipelineState(stagedPipeline!)
            stagedEnc.setBuffer(weightsBuf, offset: 0, index: 0)
            stagedEnc.setBuffer(scalesBuf, offset: 0, index: 1)
            stagedEnc.setBuffer(biasesBuf, offset: 0, index: 2)
            stagedEnc.setBuffer(inputBuf, offset: 0, index: 3)
            stagedEnc.setBuffer(stagedOut, offset: 0, index: 4)
            stagedEnc.dispatchThreadgroups(
                MTLSize(width: width, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            stagedEnc.endEncoding()
            stagedCmdBuf.commit()
            stagedCmdBuf.waitUntilCompleted()
            XCTAssertNil(stagedCmdBuf.error)

            let fusedCmdBuf = queue.makeCommandBuffer()!
            let fusedEnc = fusedCmdBuf.makeComputeCommandEncoder()!
            fusedEnc.setComputePipelineState(fusedPipeline!)
            fusedEnc.setBuffer(weightsBuf, offset: 0, index: 0)
            fusedEnc.setBuffer(scalesBuf, offset: 0, index: 1)
            fusedEnc.setBuffer(biasesBuf, offset: 0, index: 2)
            fusedEnc.setBuffer(inputBuf, offset: 0, index: 3)
            fusedEnc.setBuffer(fusedOut, offset: 0, index: 4)
            fusedEnc.setBuffer(residualBuf, offset: 0, index: 5)
            fusedEnc.dispatchThreadgroups(
                MTLSize(width: width, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            fusedEnc.endEncoding()
            fusedCmdBuf.commit()
            fusedCmdBuf.waitUntilCompleted()
            XCTAssertNil(fusedCmdBuf.error)

            let stagedPtr = stagedOut.contents().bindMemory(to: Float16.self, capacity: rows)
            let fusedPtr = fusedOut.contents().bindMemory(to: Float16.self, capacity: rows)
            var maxDiff: Float = 0
            for i in 0..<rows {
                let diff = abs(Float(stagedPtr[i]) - Float(fusedPtr[i]))
                if diff > maxDiff { maxDiff = diff }
            }

            fputs(
                "  \(fusedName) zero residual vs specialized affine:"
                    + " max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertEqual(maxDiff, 0, accuracy: 0.000001, "\(fusedName) diverges with zero residual")
        }
    }

    func testAffineMatvecQwenSpecializations_MatchGeneric() throws {
        let cases: [(rows: Int, cols: Int, functionName: String)] = [
            (2048, 2048, "affine_matvec_c2048_r2048_g64"),
            (6144, 2048, "affine_matvec_c2048_r6144_g64"),
            (4096, 2048, "affine_matvec_c2048_r4096_g64"),
            (512, 2048, "affine_matvec_c2048_r512_g64"),
            (2048, 6144, "affine_matvec_c6144_r2048_g64"),
            (2048, 1024, "affine_matvec_c1024_r2048_g64_rows4"),
            (6144, 1024, "affine_matvec_c1024_r6144_g64_rows4"),
            (1024, 2048, "affine_matvec_c2048_r1024_g64_rows4"),
            (1024, 3584, "affine_matvec_c3584_r1024_g64_rows4"),
            (248320, 1024, "affine_matvec_c1024_r248320_g64_rows4"),
            (151936, 2048, "affine_matvec_c2048_r151936_g64_rows8"),
        ]

        for (rows, cols, functionName) in cases {
            let specializedWidth = (
                functionName == "affine_matvec_c2048_r2048_g64"
                || functionName.hasSuffix("_rows4")
            )
                ? (rows + 3) / 4
                : (rows + 7) / 8
            let genericPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: "affine_matvec",
                cols: UInt32(cols),
                groupSize: 64
            )
            try XCTSkipIf(genericPipeline == nil, "Could not compile generic affine matvec shader")

            let specializedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: functionName
            )
            try XCTSkipIf(specializedPipeline == nil, "Could not compile \(functionName)")

            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64)
            let weightsBuf = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let scalesBuf = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let biasesBuf = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let inputBuf = device.makeBuffer(bytes: data.input, length: cols * 2, options: .storageModeShared)!
            let genericOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            let specializedOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            memset(genericOut.contents(), 0, rows * 2)
            memset(specializedOut.contents(), 0, rows * 2)

            let cmdBuf1 = queue.makeCommandBuffer()!
            let enc1 = cmdBuf1.makeComputeCommandEncoder()!
            enc1.setComputePipelineState(genericPipeline!)
            enc1.setBuffer(weightsBuf, offset: 0, index: 0)
            enc1.setBuffer(scalesBuf, offset: 0, index: 1)
            enc1.setBuffer(biasesBuf, offset: 0, index: 2)
            enc1.setBuffer(inputBuf, offset: 0, index: 3)
            enc1.setBuffer(genericOut, offset: 0, index: 4)
            var rowsVal = UInt32(rows); enc1.setBytes(&rowsVal, length: 4, index: 5)
            enc1.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc1.endEncoding()
            cmdBuf1.commit()
            cmdBuf1.waitUntilCompleted()
            XCTAssertNil(cmdBuf1.error)

            let cmdBuf2 = queue.makeCommandBuffer()!
            let enc2 = cmdBuf2.makeComputeCommandEncoder()!
            enc2.setComputePipelineState(specializedPipeline!)
            enc2.setBuffer(weightsBuf, offset: 0, index: 0)
            enc2.setBuffer(scalesBuf, offset: 0, index: 1)
            enc2.setBuffer(biasesBuf, offset: 0, index: 2)
            enc2.setBuffer(inputBuf, offset: 0, index: 3)
            enc2.setBuffer(specializedOut, offset: 0, index: 4)
            enc2.dispatchThreadgroups(
                MTLSize(width: specializedWidth, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc2.endEncoding()
            cmdBuf2.commit()
            cmdBuf2.waitUntilCompleted()
            XCTAssertNil(cmdBuf2.error)

            let genericPtr = genericOut.contents().bindMemory(to: Float16.self, capacity: rows)
            let specializedPtr = specializedOut.contents().bindMemory(to: Float16.self, capacity: rows)
            var maxDiff: Float = 0
            for i in 0..<rows {
                let diff = abs(Float(genericPtr[i]) - Float(specializedPtr[i]))
                if diff > maxDiff { maxDiff = diff }
            }

            fputs(
                "  \(functionName) vs generic: max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.002, "\(functionName) diverges from generic")
            if functionName == "affine_matvec_c1024_r248320_g64_rows4" {
                let mismatch = firstHalfBitMismatch(genericOut, specializedOut, count: rows)
                XCTAssertNil(
                    mismatch,
                    "Qwen 0.8B rows4 LM head changed FP16 bits at row "
                        + (mismatch.map(String.init) ?? "<none>")
                )
            }
        }
    }

    func testAffineMatvecDecodeSpecializations_MatchGeneric() throws {
        let cases: [(rows: Int, cols: Int, functionName: String)] = [
            (2048, 1536, "affine_matvec_c1536_r2048_g128_rows4"),
            (256, 1536, "affine_matvec_c1536_r256_g128_rows4"),
            (256, 1536, "affine_matvec_c1536_r256_g128_rows8"),
            (6144, 1536, "affine_matvec_c1536_r6144_g128_rows4"),
            (12288, 1536, "affine_matvec_c1536_r12288_g128_rows4"),
            (262144, 1536, "affine_matvec_c1536_r262144_g128_rows4"),
            (262144, 1536, "affine_matvec_c1536_r262144_g128_rows8"),
            (1536, 256, "affine_matvec_c256_r1536_g128_rows4"),
            (1536, 2048, "affine_matvec_c2048_r1536_g128_rows4"),
            (1536, 6144, "affine_matvec_c6144_r1536_g128_rows4"),
            (1536, 12288, "affine_matvec_c12288_r1536_g128_rows4"),
        ]

        for (rows, cols, functionName) in cases {
            let genericPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: "affine_matvec",
                cols: UInt32(cols),
                groupSize: 128
            )
            try XCTSkipIf(genericPipeline == nil, "Could not compile generic affine matvec shader")

            let specializedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: functionName
            )
            try XCTSkipIf(specializedPipeline == nil, "Could not compile \(functionName)")

            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 128)
            let weightsBuf = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let scalesBuf = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let biasesBuf = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let inputBuf = device.makeBuffer(bytes: data.input, length: cols * 2, options: .storageModeShared)!
            let genericOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            let specializedOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            memset(genericOut.contents(), 0, rows * 2)
            memset(specializedOut.contents(), 0, rows * 2)

            let cmdBuf1 = queue.makeCommandBuffer()!
            let enc1 = cmdBuf1.makeComputeCommandEncoder()!
            enc1.setComputePipelineState(genericPipeline!)
            enc1.setBuffer(weightsBuf, offset: 0, index: 0)
            enc1.setBuffer(scalesBuf, offset: 0, index: 1)
            enc1.setBuffer(biasesBuf, offset: 0, index: 2)
            enc1.setBuffer(inputBuf, offset: 0, index: 3)
            enc1.setBuffer(genericOut, offset: 0, index: 4)
            var rowsVal = UInt32(rows); enc1.setBytes(&rowsVal, length: 4, index: 5)
            enc1.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc1.endEncoding()
            cmdBuf1.commit()
            cmdBuf1.waitUntilCompleted()
            XCTAssertNil(cmdBuf1.error)

            let cmdBuf2 = queue.makeCommandBuffer()!
            let enc2 = cmdBuf2.makeComputeCommandEncoder()!
            enc2.setComputePipelineState(specializedPipeline!)
            enc2.setBuffer(weightsBuf, offset: 0, index: 0)
            enc2.setBuffer(scalesBuf, offset: 0, index: 1)
            enc2.setBuffer(biasesBuf, offset: 0, index: 2)
            enc2.setBuffer(inputBuf, offset: 0, index: 3)
            enc2.setBuffer(specializedOut, offset: 0, index: 4)
            let specializedWidth = functionName.contains("rows8")
                ? (rows + 7) / 8
                : (rows + 3) / 4
            enc2.dispatchThreadgroups(
                MTLSize(width: specializedWidth, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc2.endEncoding()
            cmdBuf2.commit()
            cmdBuf2.waitUntilCompleted()
            XCTAssertNil(cmdBuf2.error)

            let genericPtr = genericOut.contents().bindMemory(to: Float16.self, capacity: rows)
            let specializedPtr = specializedOut.contents().bindMemory(to: Float16.self, capacity: rows)
            var maxDiff: Float = 0
            for i in 0..<rows {
                let diff = abs(Float(genericPtr[i]) - Float(specializedPtr[i]))
                if diff > maxDiff { maxDiff = diff }
            }

            fputs(
                "  \(functionName) vs generic: max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.002, "\(functionName) diverges from generic")
        }
    }

    func testAffineMatvecDecodeSpecialization_InputOffsetMatchesDirectBuffer() throws {
        let rows = 2048
        let cols = 1536
        let functionName = "affine_matvec_c1536_r2048_g128_rows4"
        let specializedPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: functionName
        )
        try XCTSkipIf(specializedPipeline == nil, "Could not compile \(functionName)")

        let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 128, seed: 211)
        let weightsBuf = device.makeBuffer(
            bytes: data.weights,
            length: data.weights.count,
            options: .storageModeShared
        )!
        let scalesBuf = device.makeBuffer(
            bytes: data.scales,
            length: data.scales.count * 2,
            options: .storageModeShared
        )!
        let biasesBuf = device.makeBuffer(
            bytes: data.biases,
            length: data.biases.count * 2,
            options: .storageModeShared
        )!
        let directInputBuf = device.makeBuffer(
            bytes: data.input,
            length: cols * 2,
            options: .storageModeShared
        )!

        let inputPadding = 64
        let paddedInputCount = cols + inputPadding * 2
        let paddedInputBuf = device.makeBuffer(
            length: paddedInputCount * 2,
            options: .storageModeShared
        )!
        memset(paddedInputBuf.contents(), 0, paddedInputCount * 2)
        let paddedInputPtr = paddedInputBuf.contents().bindMemory(
            to: Float16.self,
            capacity: paddedInputCount
        )
        for i in 0..<cols {
            paddedInputPtr[inputPadding + i] = data.input[i]
        }

        let directOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
        let offsetOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
        memset(directOut.contents(), 0, rows * 2)
        memset(offsetOut.contents(), 0, rows * 2)

        let cmdBuf1 = queue.makeCommandBuffer()!
        let enc1 = cmdBuf1.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(specializedPipeline!)
        enc1.setBuffer(weightsBuf, offset: 0, index: 0)
        enc1.setBuffer(scalesBuf, offset: 0, index: 1)
        enc1.setBuffer(biasesBuf, offset: 0, index: 2)
        enc1.setBuffer(directInputBuf, offset: 0, index: 3)
        enc1.setBuffer(directOut, offset: 0, index: 4)
        enc1.dispatchThreadgroups(
            MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()
        XCTAssertNil(cmdBuf1.error)

        let cmdBuf2 = queue.makeCommandBuffer()!
        let enc2 = cmdBuf2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(specializedPipeline!)
        enc2.setBuffer(weightsBuf, offset: 0, index: 0)
        enc2.setBuffer(scalesBuf, offset: 0, index: 1)
        enc2.setBuffer(biasesBuf, offset: 0, index: 2)
        enc2.setBuffer(paddedInputBuf, offset: inputPadding * 2, index: 3)
        enc2.setBuffer(offsetOut, offset: 0, index: 4)
        enc2.dispatchThreadgroups(
            MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        enc2.endEncoding()
        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()
        XCTAssertNil(cmdBuf2.error)

        let directPtr = directOut.contents().bindMemory(to: Float16.self, capacity: rows)
        let offsetPtr = offsetOut.contents().bindMemory(to: Float16.self, capacity: rows)
        var maxDiff: Float = 0
        for i in 0..<rows {
            let diff = abs(Float(directPtr[i]) - Float(offsetPtr[i]))
            if diff > maxDiff { maxDiff = diff }
        }

        fputs(
            "  \(functionName) direct vs offset input: max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(
            maxDiff,
            0.002,
            "\(functionName) changes when the same input is bound through an offset slice"
        )
    }

    func testNormScaleAffineMatvecDecodeSpecializations_MatchGeneric() throws {
        let cases: [(rows: Int, cols: Int, functionName: String)] = [
            (2048, 1536, "norm_scale_affine_matvec_c1536_r2048_g128_rows4"),
            (12288, 1536, "norm_scale_affine_matvec_c1536_r12288_g128_rows4"),
            (262144, 1536, "norm_scale_affine_matvec_c1536_r262144_g128_rows4"),
            (262144, 1536, "norm_scale_affine_matvec_c1536_r262144_g128_rows8"),
            (1024, 256, "norm_scale_affine_matvec_c256_r1024_g128_rows4"),
            (2048, 256, "norm_scale_affine_matvec_c256_r2048_g128_rows4"),
        ]

        for (rows, cols, functionName) in cases {
            let genericPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: "norm_scale_affine_matvec",
                cols: UInt32(cols),
                groupSize: 128
            )
            try XCTSkipIf(genericPipeline == nil, "Could not compile generic norm-scale affine shader")

            let specializedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: functionName
            )
            try XCTSkipIf(specializedPipeline == nil, "Could not compile \(functionName)")

            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 128, seed: 17)
            var normWeight = [Float16](repeating: 0, count: cols)
            for i in 0..<cols {
                normWeight[i] = Float16(cos(Float(i) * 0.017) * 0.125)
            }
            var scaleValue: Float = 0.8125

            let scaleBuf = device.makeBuffer(bytes: &scaleValue, length: MemoryLayout<Float>.size, options: .storageModeShared)!
            let normInputBuf = device.makeBuffer(bytes: data.input, length: cols * 2, options: .storageModeShared)!
            let normWeightBuf = device.makeBuffer(bytes: normWeight, length: cols * 2, options: .storageModeShared)!
            let weightsBuf = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let scalesBuf = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let biasesBuf = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let genericNormOut = device.makeBuffer(length: cols * 2, options: .storageModeShared)!
            let specializedNormOut = device.makeBuffer(length: cols * 2, options: .storageModeShared)!
            let genericOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            let specializedOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            memset(genericNormOut.contents(), 0, cols * 2)
            memset(specializedNormOut.contents(), 0, cols * 2)
            memset(genericOut.contents(), 0, rows * 2)
            memset(specializedOut.contents(), 0, rows * 2)

            let cmdBuf1 = queue.makeCommandBuffer()!
            let enc1 = cmdBuf1.makeComputeCommandEncoder()!
            enc1.setComputePipelineState(genericPipeline!)
            enc1.setBuffer(scaleBuf, offset: 0, index: 0)
            enc1.setBuffer(normInputBuf, offset: 0, index: 1)
            enc1.setBuffer(normWeightBuf, offset: 0, index: 2)
            enc1.setBuffer(genericNormOut, offset: 0, index: 3)
            enc1.setBuffer(weightsBuf, offset: 0, index: 4)
            enc1.setBuffer(scalesBuf, offset: 0, index: 5)
            enc1.setBuffer(biasesBuf, offset: 0, index: 6)
            enc1.setBuffer(genericOut, offset: 0, index: 7)
            var rowsVal = UInt32(rows); enc1.setBytes(&rowsVal, length: 4, index: 8)
            enc1.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc1.endEncoding()
            cmdBuf1.commit()
            cmdBuf1.waitUntilCompleted()
            XCTAssertNil(cmdBuf1.error)

            let cmdBuf2 = queue.makeCommandBuffer()!
            let enc2 = cmdBuf2.makeComputeCommandEncoder()!
            enc2.setComputePipelineState(specializedPipeline!)
            enc2.setBuffer(scaleBuf, offset: 0, index: 0)
            enc2.setBuffer(normInputBuf, offset: 0, index: 1)
            enc2.setBuffer(normWeightBuf, offset: 0, index: 2)
            enc2.setBuffer(specializedNormOut, offset: 0, index: 3)
            enc2.setBuffer(weightsBuf, offset: 0, index: 4)
            enc2.setBuffer(scalesBuf, offset: 0, index: 5)
            enc2.setBuffer(biasesBuf, offset: 0, index: 6)
            enc2.setBuffer(specializedOut, offset: 0, index: 7)
            let specializedWidth = functionName.contains("rows8")
                ? (rows + 7) / 8
                : (rows + 3) / 4
            enc2.dispatchThreadgroups(
                MTLSize(width: specializedWidth, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc2.endEncoding()
            cmdBuf2.commit()
            cmdBuf2.waitUntilCompleted()
            XCTAssertNil(cmdBuf2.error)

            let genericNormPtr = genericNormOut.contents().bindMemory(to: Float16.self, capacity: cols)
            let specializedNormPtr = specializedNormOut.contents().bindMemory(to: Float16.self, capacity: cols)
            var maxNormDiff: Float = 0
            for i in 0..<cols {
                let diff = abs(Float(genericNormPtr[i]) - Float(specializedNormPtr[i]))
                if diff > maxNormDiff { maxNormDiff = diff }
            }

            let genericPtr = genericOut.contents().bindMemory(to: Float16.self, capacity: rows)
            let specializedPtr = specializedOut.contents().bindMemory(to: Float16.self, capacity: rows)
            var maxDiff: Float = 0
            for i in 0..<rows {
                let diff = abs(Float(genericPtr[i]) - Float(specializedPtr[i]))
                if diff > maxDiff { maxDiff = diff }
            }

            fputs(
                "  \(functionName) vs generic: max out diff = \(String(format: "%.6f", maxDiff)), max norm diff = \(String(format: "%.6f", maxNormDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.002, "\(functionName) output diverges from generic")
            XCTAssertLessThan(maxNormDiff, 0.002, "\(functionName) norm output diverges from generic")
        }
    }

    func testNormScaleAffineMatvecDecodeSpecializations_MatchStagedNormPlusAffine() throws {
        let cases: [(rows: Int, affineFn: String, fusedFn: String)] = [
            (2048, "affine_matvec_c1536_r2048_g128_rows4", "norm_scale_affine_matvec_c1536_r2048_g128_rows4"),
            (12288, "affine_matvec_c1536_r12288_g128_rows4", "norm_scale_affine_matvec_c1536_r12288_g128_rows4"),
            (262144, "affine_matvec_c1536_r262144_g128_rows4", "norm_scale_affine_matvec_c1536_r262144_g128_rows4"),
            (262144, "affine_matvec_c1536_r262144_g128_rows8", "norm_scale_affine_matvec_c1536_r262144_g128_rows8"),
        ]

        let normPipeline = makePipeline(
            shaderFile: "norms.metal",
            functionName: "rms_norm_1pw_d1536"
        )
        try XCTSkipIf(normPipeline == nil, "Could not compile specialized d1536 RMSNorm shader")

        for (rows, affineFn, fusedFn) in cases {
            let stagedAffinePipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: affineFn
            )
            try XCTSkipIf(stagedAffinePipeline == nil, "Could not compile \(affineFn)")

            let fusedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: fusedFn
            )
            try XCTSkipIf(fusedPipeline == nil, "Could not compile \(fusedFn)")

            let cols = 1536
            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 128, seed: 71)
            var normWeight = [Float16](repeating: 0, count: cols)
            for i in 0..<cols {
                normWeight[i] = Float16(cos(Float(i) * 0.0173) * 0.125)
            }

            let inputBuf = device.makeBuffer(bytes: data.input, length: cols * 2, options: .storageModeShared)!
            let normWeightBuf = device.makeBuffer(bytes: normWeight, length: cols * 2, options: .storageModeShared)!
            let weightsBuf = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let scalesBuf = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let biasesBuf = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let stagedNormOut = device.makeBuffer(length: cols * 2, options: .storageModeShared)!
            let fusedNormOut = device.makeBuffer(length: cols * 2, options: .storageModeShared)!
            let stagedOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            let fusedOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            memset(stagedNormOut.contents(), 0, cols * 2)
            memset(fusedNormOut.contents(), 0, cols * 2)
            memset(stagedOut.contents(), 0, rows * 2)
            memset(fusedOut.contents(), 0, rows * 2)

            let cmdBuf1 = queue.makeCommandBuffer()!
            let enc1 = cmdBuf1.makeComputeCommandEncoder()!
            enc1.setComputePipelineState(normPipeline!)
            enc1.setBuffer(inputBuf, offset: 0, index: 0)
            enc1.setBuffer(normWeightBuf, offset: 0, index: 1)
            enc1.setBuffer(stagedNormOut, offset: 0, index: 2)
            enc1.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 192, height: 1, depth: 1)
            )
            enc1.endEncoding()
            cmdBuf1.commit()
            cmdBuf1.waitUntilCompleted()
            XCTAssertNil(cmdBuf1.error)

            let cmdBuf2 = queue.makeCommandBuffer()!
            let enc2 = cmdBuf2.makeComputeCommandEncoder()!
            enc2.setComputePipelineState(stagedAffinePipeline!)
            enc2.setBuffer(weightsBuf, offset: 0, index: 0)
            enc2.setBuffer(scalesBuf, offset: 0, index: 1)
            enc2.setBuffer(biasesBuf, offset: 0, index: 2)
            enc2.setBuffer(stagedNormOut, offset: 0, index: 3)
            enc2.setBuffer(stagedOut, offset: 0, index: 4)
            let fusedWidth = fusedFn.contains("rows8")
                ? (rows + 7) / 8
                : (rows + 3) / 4
            enc2.dispatchThreadgroups(
                MTLSize(width: fusedWidth, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc2.endEncoding()
            cmdBuf2.commit()
            cmdBuf2.waitUntilCompleted()
            XCTAssertNil(cmdBuf2.error)

            var scaleValue: Float = 0
            let scaleBuf = device.makeBuffer(
                bytes: &scaleValue,
                length: MemoryLayout<Float>.size,
                options: .storageModeShared
            )!

            let cmdBuf3 = queue.makeCommandBuffer()!
            let enc3 = cmdBuf3.makeComputeCommandEncoder()!
            enc3.setComputePipelineState(makePipeline(shaderFile: "norms.metal", functionName: "rms_norm_scale_only_d1536")!)
            enc3.setBuffer(inputBuf, offset: 0, index: 0)
            enc3.setBuffer(scaleBuf, offset: 0, index: 1)
            enc3.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 192, height: 1, depth: 1)
            )
            enc3.endEncoding()
            cmdBuf3.commit()
            cmdBuf3.waitUntilCompleted()
            XCTAssertNil(cmdBuf3.error)

            let cmdBuf4 = queue.makeCommandBuffer()!
            let enc4 = cmdBuf4.makeComputeCommandEncoder()!
            enc4.setComputePipelineState(fusedPipeline!)
            enc4.setBuffer(scaleBuf, offset: 0, index: 0)
            enc4.setBuffer(inputBuf, offset: 0, index: 1)
            enc4.setBuffer(normWeightBuf, offset: 0, index: 2)
            enc4.setBuffer(fusedNormOut, offset: 0, index: 3)
            enc4.setBuffer(weightsBuf, offset: 0, index: 4)
            enc4.setBuffer(scalesBuf, offset: 0, index: 5)
            enc4.setBuffer(biasesBuf, offset: 0, index: 6)
            enc4.setBuffer(fusedOut, offset: 0, index: 7)
            enc4.dispatchThreadgroups(
                MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc4.endEncoding()
            cmdBuf4.commit()
            cmdBuf4.waitUntilCompleted()
            XCTAssertNil(cmdBuf4.error)

            let stagedNormPtr = stagedNormOut.contents().bindMemory(to: Float16.self, capacity: cols)
            let fusedNormPtr = fusedNormOut.contents().bindMemory(to: Float16.self, capacity: cols)
            let stagedPtr = stagedOut.contents().bindMemory(to: Float16.self, capacity: rows)
            let fusedPtr = fusedOut.contents().bindMemory(to: Float16.self, capacity: rows)
            var maxNormDiff: Float = 0
            var maxOutDiff: Float = 0
            for i in 0..<cols {
                let diff = abs(Float(stagedNormPtr[i]) - Float(fusedNormPtr[i]))
                if diff > maxNormDiff { maxNormDiff = diff }
            }
            for i in 0..<rows {
                let diff = abs(Float(stagedPtr[i]) - Float(fusedPtr[i]))
                if diff > maxOutDiff { maxOutDiff = diff }
            }

            fputs(
                "  \(fusedFn) vs staged d1536+affine:"
                    + " max norm diff = \(String(format: "%.6f", maxNormDiff))"
                    + ", max out diff = \(String(format: "%.6f", maxOutDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxNormDiff, 0.002, "\(fusedFn) norm output diverges from staged d1536 RMSNorm")
            XCTAssertLessThan(maxOutDiff, 0.002, "\(fusedFn) output diverges from staged d1536 RMSNorm + affine")
        }
    }

    func testNormScaleAffineGateUpGeGLUDecodeSpecializations_MatchStagedNormPlusGeGLU() throws {
        let cases: [(rows: Int, gateFn: String, upFn: String, fusedFn: String)] = [
            (6144, "affine_matvec_c1536_r6144_g128_rows4", "affine_matvec_c1536_r6144_g128_rows4", "norm_scale_affine_gate_up_geglu_c1536_r6144_g128_rows4"),
            (12288, "affine_matvec_c1536_r12288_g128_rows4", "affine_matvec_c1536_r12288_g128_rows4", "norm_scale_affine_gate_up_geglu_c1536_r12288_g128_rows4"),
        ]

        let normPipeline = makePipeline(
            shaderFile: "norms.metal",
            functionName: "rms_norm_1pw_d1536"
        )
        let scaleOnlyPipeline = makePipeline(
            shaderFile: "norms.metal",
            functionName: "rms_norm_scale_only_d1536"
        )
        let gegluPipeline = makePipeline(
            shaderFile: "activations.metal",
            functionName: "geglu_fused"
        )
        try XCTSkipIf(normPipeline == nil, "Could not compile specialized d1536 RMSNorm shader")
        try XCTSkipIf(scaleOnlyPipeline == nil, "Could not compile specialized d1536 RMS scale-only shader")
        try XCTSkipIf(gegluPipeline == nil, "Could not compile GeGLU activation shader")

        for (rows, gateFn, upFn, fusedFn) in cases {
            let gatePipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: gateFn
            )
            let upPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: upFn
            )
            let fusedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: fusedFn
            )
            try XCTSkipIf(gatePipeline == nil, "Could not compile \(gateFn)")
            try XCTSkipIf(upPipeline == nil, "Could not compile \(upFn)")
            try XCTSkipIf(fusedPipeline == nil, "Could not compile \(fusedFn)")

            let cols = 1536
            let gateData = makeAffineTestData(rows: rows, cols: cols, groupSize: 128, seed: 111)
            let upData = makeAffineTestData(rows: rows, cols: cols, groupSize: 128, seed: 113)
            var normWeight = [Float16](repeating: 0, count: cols)
            for i in 0..<cols {
                normWeight[i] = Float16(cos(Float(i) * 0.0173) * 0.125)
            }

            let inputBuf = device.makeBuffer(bytes: gateData.input, length: cols * 2, options: .storageModeShared)!
            let normWeightBuf = device.makeBuffer(bytes: normWeight, length: cols * 2, options: .storageModeShared)!
            let gateWeightsBuf = device.makeBuffer(bytes: gateData.weights, length: gateData.weights.count, options: .storageModeShared)!
            let gateScalesBuf = device.makeBuffer(bytes: gateData.scales, length: gateData.scales.count * 2, options: .storageModeShared)!
            let gateBiasesBuf = device.makeBuffer(bytes: gateData.biases, length: gateData.biases.count * 2, options: .storageModeShared)!
            let upWeightsBuf = device.makeBuffer(bytes: upData.weights, length: upData.weights.count, options: .storageModeShared)!
            let upScalesBuf = device.makeBuffer(bytes: upData.scales, length: upData.scales.count * 2, options: .storageModeShared)!
            let upBiasesBuf = device.makeBuffer(bytes: upData.biases, length: upData.biases.count * 2, options: .storageModeShared)!
            let stagedNormOut = device.makeBuffer(length: cols * 2, options: .storageModeShared)!
            let stagedGateOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            let stagedUpOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            let stagedOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            let fusedOut = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
            memset(stagedNormOut.contents(), 0, cols * 2)
            memset(stagedGateOut.contents(), 0, rows * 2)
            memset(stagedUpOut.contents(), 0, rows * 2)
            memset(stagedOut.contents(), 0, rows * 2)
            memset(fusedOut.contents(), 0, rows * 2)

            let cmdBuf1 = queue.makeCommandBuffer()!
            let enc1 = cmdBuf1.makeComputeCommandEncoder()!
            enc1.setComputePipelineState(normPipeline!)
            enc1.setBuffer(inputBuf, offset: 0, index: 0)
            enc1.setBuffer(normWeightBuf, offset: 0, index: 1)
            enc1.setBuffer(stagedNormOut, offset: 0, index: 2)
            enc1.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 192, height: 1, depth: 1)
            )
            enc1.endEncoding()
            cmdBuf1.commit()
            cmdBuf1.waitUntilCompleted()
            XCTAssertNil(cmdBuf1.error)

            let cmdBuf2 = queue.makeCommandBuffer()!
            let enc2 = cmdBuf2.makeComputeCommandEncoder()!
            enc2.setComputePipelineState(gatePipeline!)
            enc2.setBuffer(gateWeightsBuf, offset: 0, index: 0)
            enc2.setBuffer(gateScalesBuf, offset: 0, index: 1)
            enc2.setBuffer(gateBiasesBuf, offset: 0, index: 2)
            enc2.setBuffer(stagedNormOut, offset: 0, index: 3)
            enc2.setBuffer(stagedGateOut, offset: 0, index: 4)
            enc2.dispatchThreadgroups(
                MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc2.endEncoding()
            cmdBuf2.commit()
            cmdBuf2.waitUntilCompleted()
            XCTAssertNil(cmdBuf2.error)

            let cmdBuf3 = queue.makeCommandBuffer()!
            let enc3 = cmdBuf3.makeComputeCommandEncoder()!
            enc3.setComputePipelineState(upPipeline!)
            enc3.setBuffer(upWeightsBuf, offset: 0, index: 0)
            enc3.setBuffer(upScalesBuf, offset: 0, index: 1)
            enc3.setBuffer(upBiasesBuf, offset: 0, index: 2)
            enc3.setBuffer(stagedNormOut, offset: 0, index: 3)
            enc3.setBuffer(stagedUpOut, offset: 0, index: 4)
            enc3.dispatchThreadgroups(
                MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc3.endEncoding()
            cmdBuf3.commit()
            cmdBuf3.waitUntilCompleted()
            XCTAssertNil(cmdBuf3.error)

            let cmdBuf4 = queue.makeCommandBuffer()!
            let enc4 = cmdBuf4.makeComputeCommandEncoder()!
            enc4.setComputePipelineState(gegluPipeline!)
            enc4.setBuffer(stagedGateOut, offset: 0, index: 0)
            enc4.setBuffer(stagedUpOut, offset: 0, index: 1)
            enc4.setBuffer(stagedOut, offset: 0, index: 2)
            var rowsVal = UInt32(rows)
            enc4.setBytes(&rowsVal, length: 4, index: 3)
            enc4.dispatchThreads(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(rows, 1024), height: 1, depth: 1)
            )
            enc4.endEncoding()
            cmdBuf4.commit()
            cmdBuf4.waitUntilCompleted()
            XCTAssertNil(cmdBuf4.error)

            var scaleValue: Float = 0
            let scaleBuf = device.makeBuffer(
                bytes: &scaleValue,
                length: MemoryLayout<Float>.size,
                options: .storageModeShared
            )!
            let cmdBuf5 = queue.makeCommandBuffer()!
            let enc5 = cmdBuf5.makeComputeCommandEncoder()!
            enc5.setComputePipelineState(scaleOnlyPipeline!)
            enc5.setBuffer(inputBuf, offset: 0, index: 0)
            enc5.setBuffer(scaleBuf, offset: 0, index: 1)
            enc5.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 192, height: 1, depth: 1)
            )
            enc5.endEncoding()
            cmdBuf5.commit()
            cmdBuf5.waitUntilCompleted()
            XCTAssertNil(cmdBuf5.error)

            let cmdBuf6 = queue.makeCommandBuffer()!
            let enc6 = cmdBuf6.makeComputeCommandEncoder()!
            enc6.setComputePipelineState(fusedPipeline!)
            enc6.setBuffer(scaleBuf, offset: 0, index: 0)
            enc6.setBuffer(inputBuf, offset: 0, index: 1)
            enc6.setBuffer(normWeightBuf, offset: 0, index: 2)
            enc6.setBuffer(gateWeightsBuf, offset: 0, index: 3)
            enc6.setBuffer(gateScalesBuf, offset: 0, index: 4)
            enc6.setBuffer(gateBiasesBuf, offset: 0, index: 5)
            enc6.setBuffer(upWeightsBuf, offset: 0, index: 6)
            enc6.setBuffer(upScalesBuf, offset: 0, index: 7)
            enc6.setBuffer(upBiasesBuf, offset: 0, index: 8)
            enc6.setBuffer(fusedOut, offset: 0, index: 9)
            enc6.dispatchThreadgroups(
                MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc6.endEncoding()
            cmdBuf6.commit()
            cmdBuf6.waitUntilCompleted()
            XCTAssertNil(cmdBuf6.error)

            let stagedPtr = stagedOut.contents().bindMemory(to: Float16.self, capacity: rows)
            let fusedPtr = fusedOut.contents().bindMemory(to: Float16.self, capacity: rows)
            var maxOutDiff: Float = 0
            for i in 0..<rows {
                let diff = abs(Float(stagedPtr[i]) - Float(fusedPtr[i]))
                if diff > maxOutDiff { maxOutDiff = diff }
            }

            fputs(
                "  \(fusedFn) vs staged d1536+GeGLU:"
                    + " max out diff = \(String(format: "%.6f", maxOutDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxOutDiff, 0.002, "\(fusedFn) output diverges from staged d1536 RMSNorm + GeGLU")
        }
    }

    func testNormScaleQMMBatchedFullSpecializations_MatchStagedNormPlusQMM() throws {
        let cols = 2048
        let affineRows = 2048
        let gateRows = 8192
        let batchSize = 19
        let batchTile = 16

        let normPipeline = makePipeline(shaderFile: "norms.metal", functionName: "rms_norm_1pw_batched")
        let scaleOnlyPipeline = makePipeline(shaderFile: "norms.metal", functionName: "rms_norm_scale_only_d2048_batched")
        let affinePipeline = makePipeline(shaderFile: "lut_matvec.metal", functionName: "affine_matvec_c2048_r2048_g64_batched_full")
        let normScaleAffinePipeline = makePipeline(shaderFile: "lut_matvec.metal", functionName: "norm_scale_affine_matvec_c2048_r2048_g64_batched_full")
        let gatePipeline = makePipeline(shaderFile: "lut_matvec.metal", functionName: "fused_affine_gate_up_swiglu_c2048_r8192_g64_batched_full")
        let normScaleGatePipeline = makePipeline(shaderFile: "lut_matvec.metal", functionName: "norm_scale_affine_gate_up_swiglu_c2048_r8192_g64_batched_full")
        try XCTSkipIf(normPipeline == nil, "Could not compile batched RMSNorm shader")
        try XCTSkipIf(scaleOnlyPipeline == nil, "Could not compile d2048 scale-only shader")
        try XCTSkipIf(affinePipeline == nil, "Could not compile staged affine QMM shader")
        try XCTSkipIf(normScaleAffinePipeline == nil, "Could not compile norm-scale affine QMM shader")
        try XCTSkipIf(gatePipeline == nil, "Could not compile staged fused gate/up QMM shader")
        try XCTSkipIf(normScaleGatePipeline == nil, "Could not compile norm-scale fused gate/up QMM shader")

        let affineData = makeAffineTestData(rows: affineRows, cols: cols, groupSize: 64, seed: 121)
        let gateData = makeAffineTestData(rows: gateRows, cols: cols, groupSize: 64, seed: 123)
        let upData = makeAffineTestData(rows: gateRows, cols: cols, groupSize: 64, seed: 125)
        var inputData = [Float16](repeating: 0, count: batchSize * cols)
        for b in 0..<batchSize {
            for c in 0..<cols {
                inputData[b * cols + c] = Float16(sin(Float((b + 5) * (c + 11)) * 0.003) * 1.25)
            }
        }
        var normWeight = [Float16](repeating: 0, count: cols)
        for i in 0..<cols {
            normWeight[i] = Float16(cos(Float(i) * 0.009) * 0.05)
        }

        let inputBuf = device.makeBuffer(bytes: inputData, length: inputData.count * 2, options: .storageModeShared)!
        let normWeightBuf = device.makeBuffer(bytes: normWeight, length: normWeight.count * 2, options: .storageModeShared)!
        let scaleBuf = device.makeBuffer(length: batchSize * MemoryLayout<Float>.size, options: .storageModeShared)!
        let stagedNormOut = device.makeBuffer(length: batchSize * cols * 2, options: .storageModeShared)!
        let affineNormOut = device.makeBuffer(length: batchSize * cols * 2, options: .storageModeShared)!
        let gateNormOut = device.makeBuffer(length: batchSize * cols * 2, options: .storageModeShared)!
        let stagedAffineOut = device.makeBuffer(length: batchSize * affineRows * 2, options: .storageModeShared)!
        let fusedAffineOut = device.makeBuffer(length: batchSize * affineRows * 2, options: .storageModeShared)!
        let stagedGateOut = device.makeBuffer(length: batchSize * gateRows * 2, options: .storageModeShared)!
        let fusedGateOut = device.makeBuffer(length: batchSize * gateRows * 2, options: .storageModeShared)!
        let affineWeights = device.makeBuffer(bytes: affineData.weights, length: affineData.weights.count, options: .storageModeShared)!
        let affineScales = device.makeBuffer(bytes: affineData.scales, length: affineData.scales.count * 2, options: .storageModeShared)!
        let affineBiases = device.makeBuffer(bytes: affineData.biases, length: affineData.biases.count * 2, options: .storageModeShared)!
        let gateWeights = device.makeBuffer(bytes: gateData.weights, length: gateData.weights.count, options: .storageModeShared)!
        let gateScales = device.makeBuffer(bytes: gateData.scales, length: gateData.scales.count * 2, options: .storageModeShared)!
        let gateBiases = device.makeBuffer(bytes: gateData.biases, length: gateData.biases.count * 2, options: .storageModeShared)!
        let upWeights = device.makeBuffer(bytes: upData.weights, length: upData.weights.count, options: .storageModeShared)!
        let upScales = device.makeBuffer(bytes: upData.scales, length: upData.scales.count * 2, options: .storageModeShared)!
        let upBiases = device.makeBuffer(bytes: upData.biases, length: upData.biases.count * 2, options: .storageModeShared)!

        memset(scaleBuf.contents(), 0, batchSize * MemoryLayout<Float>.size)
        memset(stagedNormOut.contents(), 0, batchSize * cols * 2)
        memset(affineNormOut.contents(), 0, batchSize * cols * 2)
        memset(gateNormOut.contents(), 0, batchSize * cols * 2)
        memset(stagedAffineOut.contents(), 0, batchSize * affineRows * 2)
        memset(fusedAffineOut.contents(), 0, batchSize * affineRows * 2)
        memset(stagedGateOut.contents(), 0, batchSize * gateRows * 2)
        memset(fusedGateOut.contents(), 0, batchSize * gateRows * 2)

        let normCmd = queue.makeCommandBuffer()!
        let normEnc = normCmd.makeComputeCommandEncoder()!
        normEnc.setComputePipelineState(normPipeline!)
        normEnc.setBuffer(inputBuf, offset: 0, index: 0)
        normEnc.setBuffer(normWeightBuf, offset: 0, index: 1)
        normEnc.setBuffer(stagedNormOut, offset: 0, index: 2)
        var dimVal = UInt32(cols)
        var epsVal: Float = 1e-6
        normEnc.setBytes(&dimVal, length: 4, index: 3)
        normEnc.setBytes(&epsVal, length: 4, index: 4)
        normEnc.dispatchThreadgroups(
            MTLSize(width: batchSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)
        )
        normEnc.endEncoding()
        normCmd.commit()
        normCmd.waitUntilCompleted()
        XCTAssertNil(normCmd.error)

        let scaleCmd = queue.makeCommandBuffer()!
        let scaleEnc = scaleCmd.makeComputeCommandEncoder()!
        scaleEnc.setComputePipelineState(scaleOnlyPipeline!)
        scaleEnc.setBuffer(inputBuf, offset: 0, index: 0)
        scaleEnc.setBuffer(scaleBuf, offset: 0, index: 1)
        scaleEnc.dispatchThreadgroups(
            MTLSize(width: batchSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)
        )
        scaleEnc.endEncoding()
        scaleCmd.commit()
        scaleCmd.waitUntilCompleted()
        XCTAssertNil(scaleCmd.error)

        let stagedAffineCmd = queue.makeCommandBuffer()!
        let stagedAffineEnc = stagedAffineCmd.makeComputeCommandEncoder()!
        stagedAffineEnc.setComputePipelineState(affinePipeline!)
        stagedAffineEnc.setBuffer(affineWeights, offset: 0, index: 0)
        stagedAffineEnc.setBuffer(affineScales, offset: 0, index: 1)
        stagedAffineEnc.setBuffer(affineBiases, offset: 0, index: 2)
        stagedAffineEnc.setBuffer(stagedNormOut, offset: 0, index: 3)
        stagedAffineEnc.setBuffer(stagedAffineOut, offset: 0, index: 4)
        var actualBatch = UInt32(batchSize)
        stagedAffineEnc.setBytes(&actualBatch, length: 4, index: 5)
        stagedAffineEnc.dispatchThreadgroups(
            MTLSize(width: affineRows / 32, height: (batchSize + batchTile - 1) / batchTile, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        stagedAffineEnc.endEncoding()
        stagedAffineCmd.commit()
        stagedAffineCmd.waitUntilCompleted()
        XCTAssertNil(stagedAffineCmd.error)

        let fusedAffineCmd = queue.makeCommandBuffer()!
        let fusedAffineEnc = fusedAffineCmd.makeComputeCommandEncoder()!
        fusedAffineEnc.setComputePipelineState(normScaleAffinePipeline!)
        fusedAffineEnc.setBuffer(scaleBuf, offset: 0, index: 0)
        fusedAffineEnc.setBuffer(inputBuf, offset: 0, index: 1)
        fusedAffineEnc.setBuffer(normWeightBuf, offset: 0, index: 2)
        fusedAffineEnc.setBuffer(affineNormOut, offset: 0, index: 3)
        fusedAffineEnc.setBuffer(affineWeights, offset: 0, index: 4)
        fusedAffineEnc.setBuffer(affineScales, offset: 0, index: 5)
        fusedAffineEnc.setBuffer(affineBiases, offset: 0, index: 6)
        fusedAffineEnc.setBuffer(fusedAffineOut, offset: 0, index: 7)
        fusedAffineEnc.setBytes(&actualBatch, length: 4, index: 8)
        fusedAffineEnc.dispatchThreadgroups(
            MTLSize(width: affineRows / 32, height: (batchSize + batchTile - 1) / batchTile, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        fusedAffineEnc.endEncoding()
        fusedAffineCmd.commit()
        fusedAffineCmd.waitUntilCompleted()
        XCTAssertNil(fusedAffineCmd.error)

        let stagedGateCmd = queue.makeCommandBuffer()!
        let stagedGateEnc = stagedGateCmd.makeComputeCommandEncoder()!
        stagedGateEnc.setComputePipelineState(gatePipeline!)
        stagedGateEnc.setBuffer(gateWeights, offset: 0, index: 0)
        stagedGateEnc.setBuffer(gateScales, offset: 0, index: 1)
        stagedGateEnc.setBuffer(gateBiases, offset: 0, index: 2)
        stagedGateEnc.setBuffer(upWeights, offset: 0, index: 3)
        stagedGateEnc.setBuffer(upScales, offset: 0, index: 4)
        stagedGateEnc.setBuffer(upBiases, offset: 0, index: 5)
        stagedGateEnc.setBuffer(stagedNormOut, offset: 0, index: 6)
        stagedGateEnc.setBuffer(stagedGateOut, offset: 0, index: 7)
        stagedGateEnc.setBytes(&actualBatch, length: 4, index: 8)
        stagedGateEnc.dispatchThreadgroups(
            MTLSize(width: gateRows / 32, height: (batchSize + batchTile - 1) / batchTile, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        stagedGateEnc.endEncoding()
        stagedGateCmd.commit()
        stagedGateCmd.waitUntilCompleted()
        XCTAssertNil(stagedGateCmd.error)

        let fusedGateCmd = queue.makeCommandBuffer()!
        let fusedGateEnc = fusedGateCmd.makeComputeCommandEncoder()!
        fusedGateEnc.setComputePipelineState(normScaleGatePipeline!)
        fusedGateEnc.setBuffer(scaleBuf, offset: 0, index: 0)
        fusedGateEnc.setBuffer(inputBuf, offset: 0, index: 1)
        fusedGateEnc.setBuffer(normWeightBuf, offset: 0, index: 2)
        fusedGateEnc.setBuffer(gateNormOut, offset: 0, index: 3)
        fusedGateEnc.setBuffer(gateWeights, offset: 0, index: 4)
        fusedGateEnc.setBuffer(gateScales, offset: 0, index: 5)
        fusedGateEnc.setBuffer(gateBiases, offset: 0, index: 6)
        fusedGateEnc.setBuffer(upWeights, offset: 0, index: 7)
        fusedGateEnc.setBuffer(upScales, offset: 0, index: 8)
        fusedGateEnc.setBuffer(upBiases, offset: 0, index: 9)
        fusedGateEnc.setBuffer(fusedGateOut, offset: 0, index: 10)
        fusedGateEnc.setBytes(&actualBatch, length: 4, index: 11)
        fusedGateEnc.dispatchThreadgroups(
            MTLSize(width: gateRows / 32, height: (batchSize + batchTile - 1) / batchTile, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        fusedGateEnc.endEncoding()
        fusedGateCmd.commit()
        fusedGateCmd.waitUntilCompleted()
        XCTAssertNil(fusedGateCmd.error)

        let stagedNormPtr = stagedNormOut.contents().bindMemory(to: Float16.self, capacity: batchSize * cols)
        let affineNormPtr = affineNormOut.contents().bindMemory(to: Float16.self, capacity: batchSize * cols)
        let gateNormPtr = gateNormOut.contents().bindMemory(to: Float16.self, capacity: batchSize * cols)
        var maxAffineNormDiff: Float = 0
        var maxGateNormDiff: Float = 0
        for i in 0..<(batchSize * cols) {
            maxAffineNormDiff = max(maxAffineNormDiff, abs(Float(stagedNormPtr[i]) - Float(affineNormPtr[i])))
            maxGateNormDiff = max(maxGateNormDiff, abs(Float(stagedNormPtr[i]) - Float(gateNormPtr[i])))
        }

        let stagedAffinePtr = stagedAffineOut.contents().bindMemory(to: Float16.self, capacity: batchSize * affineRows)
        let fusedAffinePtr = fusedAffineOut.contents().bindMemory(to: Float16.self, capacity: batchSize * affineRows)
        var maxAffineOutDiff: Float = 0
        for i in 0..<(batchSize * affineRows) {
            maxAffineOutDiff = max(maxAffineOutDiff, abs(Float(stagedAffinePtr[i]) - Float(fusedAffinePtr[i])))
        }

        let stagedGatePtr = stagedGateOut.contents().bindMemory(to: Float16.self, capacity: batchSize * gateRows)
        let fusedGatePtr = fusedGateOut.contents().bindMemory(to: Float16.self, capacity: batchSize * gateRows)
        var maxGateOutDiff: Float = 0
        for i in 0..<(batchSize * gateRows) {
            maxGateOutDiff = max(maxGateOutDiff, abs(Float(stagedGatePtr[i]) - Float(fusedGatePtr[i])))
        }

        fputs(
            "  norm-scale QMM batched full vs staged:"
                + " affine norm diff = \(String(format: "%.6f", maxAffineNormDiff))"
                + ", affine out diff = \(String(format: "%.6f", maxAffineOutDiff))"
                + ", gate norm diff = \(String(format: "%.6f", maxGateNormDiff))"
                + ", gate out diff = \(String(format: "%.6f", maxGateOutDiff))\n",
            stderr
        )
        // Fused norm-scale consumers change Float16 rounding boundaries.
        // These budgets pin the observed one-ULP norm/affine drift and the
        // amplified gate-up rounding without pretending the paths are exact.
        XCTAssertLessThan(maxAffineNormDiff, 0.002, "Norm-scale affine side-effect norm diverges")
        XCTAssertLessThan(maxAffineOutDiff, 0.02, "Norm-scale affine QMM output diverges")
        XCTAssertLessThan(maxGateNormDiff, 0.002, "Norm-scale gate/up side-effect norm diverges")
        XCTAssertLessThan(maxGateOutDiff, 0.5, "Norm-scale gate/up QMM output diverges")
    }

    func testNormBoundaryRequiresRoundedHalfInputForAffineAndGeGLUConsumers() throws {
        let cols = 1536
        let groupSize = 128
        let affineRows = 2048
        let gegluRows = 6144

        let affineData = makeAffineTestData(rows: affineRows, cols: cols, groupSize: groupSize, seed: 83)
        let gateData = makeAffineTestData(rows: gegluRows, cols: cols, groupSize: groupSize, seed: 89)
        let upData = makeAffineTestData(rows: gegluRows, cols: cols, groupSize: groupSize, seed: 97)

        var normWeight = [Float16](repeating: 0, count: cols)
        for i in 0..<cols {
            normWeight[i] = Float16(cos(Float(i) * 0.0173) * 0.125)
        }

        let normInput = referenceRoundedAndUnroundedNormInput(
            input: affineData.input,
            weight: normWeight
        )

        let affineRounded = referenceAffineMatvecCPU(
            weights: affineData.weights,
            scales: affineData.scales,
            biases: affineData.biases,
            inputData: normInput.rounded,
            rows: affineRows,
            cols: cols,
            groupSize: groupSize,
            batchSize: 1
        )
        let affineUnrounded = referenceAffineMatvecCPUFloatInput(
            weights: affineData.weights,
            scales: affineData.scales,
            biases: affineData.biases,
            inputData: normInput.unrounded,
            rows: affineRows,
            cols: cols,
            groupSize: groupSize
        )

        let gegluRounded = referenceFusedAffineGateUpGeGLUCPU(
            gateWeights: gateData.weights,
            gateScales: gateData.scales,
            gateBiases: gateData.biases,
            upWeights: upData.weights,
            upScales: upData.scales,
            upBiases: upData.biases,
            inputData: normInput.rounded,
            rows: gegluRows,
            cols: cols,
            groupSize: groupSize,
            batchSize: 1
        )
        let gegluUnrounded = referenceFusedAffineGateUpGeGLUCPUFloatInput(
            gateWeights: gateData.weights,
            gateScales: gateData.scales,
            gateBiases: gateData.biases,
            upWeights: upData.weights,
            upScales: upData.scales,
            upBiases: upData.biases,
            inputData: normInput.unrounded,
            rows: gegluRows,
            cols: cols,
            groupSize: groupSize
        )

        var affineMaxDiff: Float = 0
        var gegluMaxDiff: Float = 0
        for i in 0..<affineRows {
            let diff = abs(affineRounded[i] - affineUnrounded[i])
            if diff > affineMaxDiff { affineMaxDiff = diff }
        }
        for i in 0..<gegluRows {
            let diff = abs(gegluRounded[i] - gegluUnrounded[i])
            if diff > gegluMaxDiff { gegluMaxDiff = diff }
        }

        fputs(
            "  norm boundary rounded-vs-unrounded input:"
                + " affine max diff = \(String(format: "%.6f", affineMaxDiff))"
                + ", geglu max diff = \(String(format: "%.6f", gegluMaxDiff))\n",
            stderr
        )
        XCTAssertGreaterThan(
            affineMaxDiff,
            0.001,
            "Skipping the staged half round-trip at the norm boundary should change affine consumer output"
        )
        XCTAssertGreaterThan(
            gegluMaxDiff,
            0.001,
            "Skipping the staged half round-trip at the norm boundary should change GeGLU consumer output"
        )
    }

    func testAffineMatvecQwenBatchedSpecializations_MatchIterated() throws {
        let cases: [(rows: Int, cols: Int, functionName: String, batchTile: Int)] = [
            (2048, 2048, "affine_matvec_c2048_r2048_g64_batched", 8),
            (6144, 2048, "affine_matvec_c2048_r6144_g64_batched", 8),
            (4096, 2048, "affine_matvec_c2048_r4096_g64_batched", 8),
            (512, 2048, "affine_matvec_c2048_r512_g64_batched", 8),
            (2048, 6144, "affine_matvec_c6144_r2048_g64_batched", 8),
        ]
        let batchSize = 4

        for (rows, cols, functionName, batchTile) in cases {
            let genericPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: "affine_matvec",
                cols: UInt32(cols),
                groupSize: 64
            )
            try XCTSkipIf(genericPipeline == nil, "Could not compile generic affine matvec shader")

            let batchedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: functionName
            )
            try XCTSkipIf(batchedPipeline == nil, "Could not compile \(functionName)")

            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 1)
            var inputData = [Float16](repeating: 0, count: batchSize * cols)
            for b in 0..<batchSize {
                for c in 0..<cols {
                    inputData[b * cols + c] = Float16(sin(Float((b + 1) * (c + 3)) * 0.013))
                }
            }

            let weightsBuf = device.makeBuffer(
                bytes: data.weights, length: data.weights.count, options: .storageModeShared
            )!
            let scalesBuf = device.makeBuffer(
                bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared
            )!
            let biasesBuf = device.makeBuffer(
                bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared
            )!
            let inputBuf = device.makeBuffer(
                bytes: inputData, length: inputData.count * 2, options: .storageModeShared
            )!
            let batchedOut = device.makeBuffer(
                length: batchSize * rows * 2, options: .storageModeShared
            )!
            memset(batchedOut.contents(), 0, batchSize * rows * 2)

            let reference = try runIteratedAffineMatvec(
                pipeline: genericPipeline!,
                weights: weightsBuf,
                scales: scalesBuf,
                biases: biasesBuf,
                inputData: inputData,
                rows: rows,
                cols: cols,
                batchSize: batchSize
            )

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(batchedPipeline!)
            enc.setBuffer(weightsBuf, offset: 0, index: 0)
            enc.setBuffer(scalesBuf, offset: 0, index: 1)
            enc.setBuffer(biasesBuf, offset: 0, index: 2)
            enc.setBuffer(inputBuf, offset: 0, index: 3)
            enc.setBuffer(batchedOut, offset: 0, index: 4)
            var actualBatch = UInt32(batchSize)
            enc.setBytes(&actualBatch, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(
                    width: (rows + 7) / 8,
                    height: (batchSize + batchTile - 1) / batchTile,
                    depth: 1
                ),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let outPtr = batchedOut.contents().bindMemory(
                to: Float16.self, capacity: batchSize * rows
            )
            var maxDiff: Float = 0
            for i in 0..<(batchSize * rows) {
                let diff = abs(Float(outPtr[i]) - reference[i])
                if diff > maxDiff { maxDiff = diff }
            }

            fputs(
                "  \(functionName) vs iterated: max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            // The qmm full path reorders accumulation through simdgroup_matrix tiles,
            // so it is not expected to be nearly bit-identical to the scalar reference.
            XCTAssertLessThan(maxDiff, 0.07, "\(functionName) diverges from iterated affine_matvec")
        }
    }

    func testAffineMatvecSmallBatchSG4_MatchesScalarBatchedBits() throws {
        let rows = 2048
        let cols = 2560
        let groupSize = 128
        let batchSize = 5

        let referencePipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "affine_matvec_c2560_r2048_g128_batched"
        )
        try XCTSkipIf(referencePipeline == nil, "Could not compile scalar batched affine")

        let smallBatchPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "affine_matvec_c2560_r2048_g128_batched_sg4_bt5"
        )
        try XCTSkipIf(smallBatchPipeline == nil, "Could not compile SG4 small-batch affine")

        let data = makeAffineTestData(rows: rows, cols: cols, groupSize: groupSize, seed: 71)
        var inputData = [Float16](repeating: 0, count: batchSize * cols)
        for b in 0..<batchSize {
            for c in 0..<cols {
                let value = sin(Float((b + 2) * (c + 5)) * 0.011)
                inputData[b * cols + c] = Float16(value)
            }
        }

        let weightsBuf = device.makeBuffer(
            bytes: data.weights, length: data.weights.count, options: .storageModeShared
        )!
        let scalesBuf = device.makeBuffer(
            bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared
        )!
        let biasesBuf = device.makeBuffer(
            bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared
        )!
        let inputBuf = device.makeBuffer(
            bytes: inputData, length: inputData.count * 2, options: .storageModeShared
        )!
        let referenceOut = device.makeBuffer(
            length: batchSize * rows * 2, options: .storageModeShared
        )!
        let smallBatchOut = device.makeBuffer(
            length: batchSize * rows * 2, options: .storageModeShared
        )!
        memset(referenceOut.contents(), 0, batchSize * rows * 2)
        memset(smallBatchOut.contents(), 0, batchSize * rows * 2)

        var actualBatch = UInt32(batchSize)

        do {
            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(referencePipeline!)
            enc.setBuffer(weightsBuf, offset: 0, index: 0)
            enc.setBuffer(scalesBuf, offset: 0, index: 1)
            enc.setBuffer(biasesBuf, offset: 0, index: 2)
            enc.setBuffer(inputBuf, offset: 0, index: 3)
            enc.setBuffer(referenceOut, offset: 0, index: 4)
            enc.setBytes(&actualBatch, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error {
                throw error
            }
        }

        do {
            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(smallBatchPipeline!)
            enc.setBuffer(weightsBuf, offset: 0, index: 0)
            enc.setBuffer(scalesBuf, offset: 0, index: 1)
            enc.setBuffer(biasesBuf, offset: 0, index: 2)
            enc.setBuffer(inputBuf, offset: 0, index: 3)
            enc.setBuffer(smallBatchOut, offset: 0, index: 4)
            enc.setBytes(&actualBatch, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: (rows + 15) / 16, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error {
                throw error
            }
        }

        let referenceBits = referenceOut.contents().bindMemory(
            to: UInt16.self, capacity: batchSize * rows
        )
        let smallBatchBits = smallBatchOut.contents().bindMemory(
            to: UInt16.self, capacity: batchSize * rows
        )
        var firstMismatch: Int?
        for i in 0..<(batchSize * rows) {
            if referenceBits[i] != smallBatchBits[i] {
                firstMismatch = i
                break
            }
        }

        if let firstMismatch {
            XCTFail(
                "SG4 small-batch affine differs at output \(firstMismatch): " +
                    "reference=0x\(String(referenceBits[firstMismatch], radix: 16)) " +
                    "sg4=0x\(String(smallBatchBits[firstMismatch], radix: 16))"
            )
        }
    }

    func testAffineMatvecOProjectionSmallBatchSG4_MatchesScalarBatchedBits() throws {
        let rows = 2560
        let cols = 2048
        let groupSize = 128
        let batchSize = 5

        let referencePipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "affine_matvec_c2048_r2560_g128_batched"
        )
        try XCTSkipIf(referencePipeline == nil, "Could not compile scalar O batched affine")

        let smallBatchPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "affine_matvec_c2048_r2560_g128_batched_sg4_bt5"
        )
        try XCTSkipIf(smallBatchPipeline == nil, "Could not compile O SG4 small-batch affine")

        let data = makeAffineTestData(rows: rows, cols: cols, groupSize: groupSize, seed: 73)
        var inputData = [Float16](repeating: 0, count: batchSize * cols)
        for b in 0..<batchSize {
            for c in 0..<cols {
                let value = sin(Float((b + 3) * (c + 7)) * 0.013)
                inputData[b * cols + c] = Float16(value)
            }
        }

        let weightsBuf = device.makeBuffer(
            bytes: data.weights, length: data.weights.count, options: .storageModeShared
        )!
        let scalesBuf = device.makeBuffer(
            bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared
        )!
        let biasesBuf = device.makeBuffer(
            bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared
        )!
        let inputBuf = device.makeBuffer(
            bytes: inputData, length: inputData.count * 2, options: .storageModeShared
        )!
        let referenceOut = device.makeBuffer(
            length: batchSize * rows * 2, options: .storageModeShared
        )!
        let smallBatchOut = device.makeBuffer(
            length: batchSize * rows * 2, options: .storageModeShared
        )!
        memset(referenceOut.contents(), 0, batchSize * rows * 2)
        memset(smallBatchOut.contents(), 0, batchSize * rows * 2)

        var actualBatch = UInt32(batchSize)

        do {
            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(referencePipeline!)
            enc.setBuffer(weightsBuf, offset: 0, index: 0)
            enc.setBuffer(scalesBuf, offset: 0, index: 1)
            enc.setBuffer(biasesBuf, offset: 0, index: 2)
            enc.setBuffer(inputBuf, offset: 0, index: 3)
            enc.setBuffer(referenceOut, offset: 0, index: 4)
            enc.setBytes(&actualBatch, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error {
                throw error
            }
        }

        do {
            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(smallBatchPipeline!)
            enc.setBuffer(weightsBuf, offset: 0, index: 0)
            enc.setBuffer(scalesBuf, offset: 0, index: 1)
            enc.setBuffer(biasesBuf, offset: 0, index: 2)
            enc.setBuffer(inputBuf, offset: 0, index: 3)
            enc.setBuffer(smallBatchOut, offset: 0, index: 4)
            enc.setBytes(&actualBatch, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: (rows + 15) / 16, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error {
                throw error
            }
        }

        let referenceBits = referenceOut.contents().bindMemory(
            to: UInt16.self, capacity: batchSize * rows
        )
        let smallBatchBits = smallBatchOut.contents().bindMemory(
            to: UInt16.self, capacity: batchSize * rows
        )
        var firstMismatch: Int?
        for i in 0..<(batchSize * rows) {
            if referenceBits[i] != smallBatchBits[i] {
                firstMismatch = i
                break
            }
        }

        if let firstMismatch {
            XCTFail(
                "SG4 O small-batch affine differs at output \(firstMismatch): " +
                    "reference=0x\(String(referenceBits[firstMismatch], radix: 16)) " +
                    "sg4=0x\(String(smallBatchBits[firstMismatch], radix: 16))"
            )
        }
    }

    func testAffineMatvecFFNDownSmallBatchSG4_MatchesTile4BatchedBits() throws {
        let rows = 2560
        let cols = 10240
        let groupSize = 128
        let batchSize = 5

        let referencePipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "affine_matvec_c10240_r2560_g128_batched_tile4"
        )
        try XCTSkipIf(referencePipeline == nil, "Could not compile FFN down tile4 affine")

        let smallBatchPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "affine_matvec_c10240_r2560_g128_batched_sg4_bt5"
        )
        try XCTSkipIf(smallBatchPipeline == nil, "Could not compile FFN down SG4 affine")

        let data = makeAffineTestData(rows: rows, cols: cols, groupSize: groupSize, seed: 79)
        var inputData = [Float16](repeating: 0, count: batchSize * cols)
        for b in 0..<batchSize {
            for c in 0..<cols {
                let value = sin(Float((b + 5) * (c + 11)) * 0.005)
                inputData[b * cols + c] = Float16(value)
            }
        }

        let weightsBuf = device.makeBuffer(
            bytes: data.weights, length: data.weights.count, options: .storageModeShared
        )!
        let scalesBuf = device.makeBuffer(
            bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared
        )!
        let biasesBuf = device.makeBuffer(
            bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared
        )!
        let inputBuf = device.makeBuffer(
            bytes: inputData, length: inputData.count * 2, options: .storageModeShared
        )!
        let referenceOut = device.makeBuffer(
            length: batchSize * rows * 2, options: .storageModeShared
        )!
        let smallBatchOut = device.makeBuffer(
            length: batchSize * rows * 2, options: .storageModeShared
        )!
        memset(referenceOut.contents(), 0, batchSize * rows * 2)
        memset(smallBatchOut.contents(), 0, batchSize * rows * 2)

        var actualBatch = UInt32(batchSize)

        do {
            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(referencePipeline!)
            enc.setBuffer(weightsBuf, offset: 0, index: 0)
            enc.setBuffer(scalesBuf, offset: 0, index: 1)
            enc.setBuffer(biasesBuf, offset: 0, index: 2)
            enc.setBuffer(inputBuf, offset: 0, index: 3)
            enc.setBuffer(referenceOut, offset: 0, index: 4)
            enc.setBytes(&actualBatch, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: (batchSize + 3) / 4, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error {
                throw error
            }
        }

        do {
            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(smallBatchPipeline!)
            enc.setBuffer(weightsBuf, offset: 0, index: 0)
            enc.setBuffer(scalesBuf, offset: 0, index: 1)
            enc.setBuffer(biasesBuf, offset: 0, index: 2)
            enc.setBuffer(inputBuf, offset: 0, index: 3)
            enc.setBuffer(smallBatchOut, offset: 0, index: 4)
            enc.setBytes(&actualBatch, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: (rows + 15) / 16, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error {
                throw error
            }
        }

        let referenceBits = referenceOut.contents().bindMemory(
            to: UInt16.self, capacity: batchSize * rows
        )
        let smallBatchBits = smallBatchOut.contents().bindMemory(
            to: UInt16.self, capacity: batchSize * rows
        )
        var firstMismatch: Int?
        for i in 0..<(batchSize * rows) {
            if referenceBits[i] != smallBatchBits[i] {
                firstMismatch = i
                break
            }
        }

        if let firstMismatch {
            XCTFail(
                "SG4 FFN down small-batch affine differs at output \(firstMismatch): " +
                    "reference=0x\(String(referenceBits[firstMismatch], radix: 16)) " +
                    "sg4=0x\(String(smallBatchBits[firstMismatch], radix: 16))"
            )
        }
    }

    func testAffineMatvecFFNGateUpSmallBatchSG4_MatchesScalarBatchedBits() throws {
        let rows = 10240
        let cols = 2560
        let groupSize = 128
        let batchSize = 5

        let referencePipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "affine_matvec_c2560_r10240_g128_batched"
        )
        try XCTSkipIf(referencePipeline == nil, "Could not compile FFN gate/up affine")

        let smallBatchPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "affine_matvec_c2560_r10240_g128_batched_sg4_bt5"
        )
        try XCTSkipIf(smallBatchPipeline == nil, "Could not compile FFN gate/up SG4 affine")

        let data = makeAffineTestData(rows: rows, cols: cols, groupSize: groupSize, seed: 83)
        var inputData = [Float16](repeating: 0, count: batchSize * cols)
        for b in 0..<batchSize {
            for c in 0..<cols {
                let value = sin(Float((b + 7) * (c + 13)) * 0.011)
                inputData[b * cols + c] = Float16(value)
            }
        }

        let weightsBuf = device.makeBuffer(
            bytes: data.weights, length: data.weights.count, options: .storageModeShared
        )!
        let scalesBuf = device.makeBuffer(
            bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared
        )!
        let biasesBuf = device.makeBuffer(
            bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared
        )!
        let inputBuf = device.makeBuffer(
            bytes: inputData, length: inputData.count * 2, options: .storageModeShared
        )!
        let referenceOut = device.makeBuffer(
            length: batchSize * rows * 2, options: .storageModeShared
        )!
        let smallBatchOut = device.makeBuffer(
            length: batchSize * rows * 2, options: .storageModeShared
        )!
        memset(referenceOut.contents(), 0, batchSize * rows * 2)
        memset(smallBatchOut.contents(), 0, batchSize * rows * 2)

        var actualBatch = UInt32(batchSize)

        do {
            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(referencePipeline!)
            enc.setBuffer(weightsBuf, offset: 0, index: 0)
            enc.setBuffer(scalesBuf, offset: 0, index: 1)
            enc.setBuffer(biasesBuf, offset: 0, index: 2)
            enc.setBuffer(inputBuf, offset: 0, index: 3)
            enc.setBuffer(referenceOut, offset: 0, index: 4)
            enc.setBytes(&actualBatch, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error {
                throw error
            }
        }

        do {
            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(smallBatchPipeline!)
            enc.setBuffer(weightsBuf, offset: 0, index: 0)
            enc.setBuffer(scalesBuf, offset: 0, index: 1)
            enc.setBuffer(biasesBuf, offset: 0, index: 2)
            enc.setBuffer(inputBuf, offset: 0, index: 3)
            enc.setBuffer(smallBatchOut, offset: 0, index: 4)
            enc.setBytes(&actualBatch, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: (rows + 15) / 16, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error {
                throw error
            }
        }

        let referenceBits = referenceOut.contents().bindMemory(
            to: UInt16.self, capacity: batchSize * rows
        )
        let smallBatchBits = smallBatchOut.contents().bindMemory(
            to: UInt16.self, capacity: batchSize * rows
        )
        var firstMismatch: Int?
        for i in 0..<(batchSize * rows) {
            if referenceBits[i] != smallBatchBits[i] {
                firstMismatch = i
                break
            }
        }

        if let firstMismatch {
            XCTFail(
                "SG4 FFN gate/up small-batch affine differs at output \(firstMismatch): " +
                    "reference=0x\(String(referenceBits[firstMismatch], radix: 16)) " +
                    "sg4=0x\(String(smallBatchBits[firstMismatch], radix: 16))"
            )
        }
    }

    func testAffineMatvecQwenBatchedSpecializations_LargeBatchBoundaries() throws {
        let slow = ProcessInfo.processInfo.environment["SMELT_RUN_SLOW_TESTS"] == "1"
        try XCTSkipUnless(slow, "Large-batch prefill affine boundary coverage is opt-in")

        let cases: [(rows: Int, cols: Int, functionName: String, batchTile: Int)] = [
            (2048, 2048, "affine_matvec_c2048_r2048_g64_batched", 8),
            (6144, 2048, "affine_matvec_c2048_r6144_g64_batched", 8),
            (2048, 6144, "affine_matvec_c6144_r2048_g64_batched", 8),
        ]

        for (rows, cols, functionName, batchTile) in cases {
            let genericPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: "affine_matvec",
                cols: UInt32(cols),
                groupSize: 64
            )
            try XCTSkipIf(genericPipeline == nil, "Could not compile generic affine matvec shader")

            let batchedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: functionName
            )
            try XCTSkipIf(batchedPipeline == nil, "Could not compile \(functionName)")

            for batchSize in [64, 65] {
                let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64)
                var inputData = [Float16](repeating: 0, count: batchSize * cols)
                for b in 0..<batchSize {
                    for c in 0..<cols {
                        inputData[b * cols + c] = Float16(sin(Float((b + 1) * (c + 5)) * 0.007))
                    }
                }

                let weightsBuf = device.makeBuffer(
                    bytes: data.weights, length: data.weights.count, options: .storageModeShared
                )!
                let scalesBuf = device.makeBuffer(
                    bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared
                )!
                let biasesBuf = device.makeBuffer(
                    bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared
                )!
                let inputBuf = device.makeBuffer(
                    bytes: inputData, length: inputData.count * 2, options: .storageModeShared
                )!
                let batchedOut = device.makeBuffer(
                    length: batchSize * rows * 2, options: .storageModeShared
                )!
                memset(batchedOut.contents(), 0, batchSize * rows * 2)

                let reference = try runIteratedAffineMatvec(
                    pipeline: genericPipeline!,
                    weights: weightsBuf,
                    scales: scalesBuf,
                    biases: biasesBuf,
                    inputData: inputData,
                    rows: rows,
                    cols: cols,
                    batchSize: batchSize
                )

                let cmdBuf = queue.makeCommandBuffer()!
                let enc = cmdBuf.makeComputeCommandEncoder()!
                enc.setComputePipelineState(batchedPipeline!)
                enc.setBuffer(weightsBuf, offset: 0, index: 0)
                enc.setBuffer(scalesBuf, offset: 0, index: 1)
                enc.setBuffer(biasesBuf, offset: 0, index: 2)
                enc.setBuffer(inputBuf, offset: 0, index: 3)
                enc.setBuffer(batchedOut, offset: 0, index: 4)
                var actualBatch = UInt32(batchSize)
                enc.setBytes(&actualBatch, length: 4, index: 5)
                enc.dispatchThreadgroups(
                    MTLSize(
                        width: (rows + 7) / 8,
                        height: (batchSize + batchTile - 1) / batchTile,
                        depth: 1
                    ),
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
                )
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
                XCTAssertNil(cmdBuf.error)

                let outPtr = batchedOut.contents().bindMemory(
                    to: Float16.self, capacity: batchSize * rows
                )
                var maxDiff: Float = 0
                for i in 0..<(batchSize * rows) {
                    maxDiff = max(maxDiff, abs(Float(outPtr[i]) - reference[i]))
                }

                fputs(
                    "  \(functionName) batch=\(batchSize):"
                        + " max diff = \(String(format: "%.6f", maxDiff))\n",
                    stderr
                )
                XCTAssertLessThan(maxDiff, 0.04, "\(functionName) diverges at batch boundary \(batchSize)")
            }
        }
    }

    func testAffineMatvecQwenBatchedFullSpecializations_MatchIterated() throws {
        let cases: [(rows: Int, cols: Int, functionName: String)] = [
            (2048, 1024, "affine_matvec_c1024_r2048_g64_batched_full"),
            (3584, 1024, "affine_matvec_c1024_r3584_g64_batched_full"),
            (4096, 1024, "affine_matvec_c1024_r4096_g64_batched_full"),
            (512, 1024, "affine_matvec_c1024_r512_g64_batched_full"),
            (6144, 1024, "affine_matvec_c1024_r6144_g64_batched_full"),
            (1024, 2048, "affine_matvec_c2048_r1024_g64_batched_full"),
            (1024, 3584, "affine_matvec_c3584_r1024_g64_batched_full"),
            (2048, 2048, "affine_matvec_c2048_r2048_g64_batched_full"),
            (6144, 2048, "affine_matvec_c2048_r6144_g64_batched_full"),
            (4096, 2048, "affine_matvec_c2048_r4096_g64_batched_full"),
            (512, 2048, "affine_matvec_c2048_r512_g64_batched_full"),
            (2048, 6144, "affine_matvec_c6144_r2048_g64_batched_full"),
            (2048, 8192, "affine_matvec_c8192_r2048_g64_batched_full"),
        ]
        let batchSize = 16
        for (rows, cols, functionName) in cases {
            let usesQMMFull = true
            let rowTile = usesQMMFull ? 32 : 8
            let batchTile = usesQMMFull ? 16 : 8
            let tgWidth = usesQMMFull ? 128 : 64
            let genericPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: "affine_matvec",
                cols: UInt32(cols),
                groupSize: 64
            )
            try XCTSkipIf(genericPipeline == nil, "Could not compile generic affine matvec shader")

            let fullPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: functionName
            )
            try XCTSkipIf(fullPipeline == nil, "Could not compile \(functionName)")

            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 31)
            var inputData = [Float16](repeating: 0, count: batchSize * cols)
            for b in 0..<batchSize {
                for c in 0..<cols {
                    inputData[b * cols + c] = Float16(cos(Float((b + 3) * (c + 7)) * 0.011))
                }
            }

            let weightsBuf = device.makeBuffer(
                bytes: data.weights, length: data.weights.count, options: .storageModeShared
            )!
            let scalesBuf = device.makeBuffer(
                bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared
            )!
            let biasesBuf = device.makeBuffer(
                bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared
            )!
            let inputBuf = device.makeBuffer(
                bytes: inputData, length: inputData.count * 2, options: .storageModeShared
            )!
            let outBuf = device.makeBuffer(length: batchSize * rows * 2, options: .storageModeShared)!
            memset(outBuf.contents(), 0, batchSize * rows * 2)

            let reference = try runIteratedAffineMatvec(
                pipeline: genericPipeline!,
                weights: weightsBuf,
                scales: scalesBuf,
                biases: biasesBuf,
                inputData: inputData,
                rows: rows,
                cols: cols,
                batchSize: batchSize
            )

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(fullPipeline!)
            enc.setBuffer(weightsBuf, offset: 0, index: 0)
            enc.setBuffer(scalesBuf, offset: 0, index: 1)
            enc.setBuffer(biasesBuf, offset: 0, index: 2)
            enc.setBuffer(inputBuf, offset: 0, index: 3)
            enc.setBuffer(outBuf, offset: 0, index: 4)
            var actualBatch = UInt32(batchSize)
            enc.setBytes(&actualBatch, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(
                    width: (rows + rowTile - 1) / rowTile,
                    height: (batchSize + batchTile - 1) / batchTile,
                    depth: 1
                ),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let outPtr = outBuf.contents().bindMemory(to: Float16.self, capacity: batchSize * rows)
            var maxDiff: Float = 0
            for i in 0..<(batchSize * rows) {
                let diff = abs(Float(outPtr[i]) - reference[i])
                if diff > maxDiff { maxDiff = diff }
            }

            fputs(
                "  \(functionName) vs iterated: max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            // The qmm full path reorders accumulation through simdgroup_matrix tiles,
            // so it is not expected to be nearly bit-identical to the scalar reference.
            XCTAssertLessThan(maxDiff, 0.07, "\(functionName) diverges from iterated affine_matvec")
        }
    }

    func testAffineMatvecQwenBatchedFullNoPadding_MatchesCurrentBits() throws {
        let cases: [(rows: Int, cols: Int, current: String, candidate: String)] = [
            (
                2048, 1024,
                "affine_matvec_c1024_r2048_g64_batched_full_padded_reference",
                "affine_matvec_c1024_r2048_g64_batched_full"
            ),
            (
                2048, 2048,
                "affine_matvec_c2048_r2048_g64_batched_full_padded_reference",
                "affine_matvec_c2048_r2048_g64_batched_full"
            ),
        ]
        let batchSizes = [1, 15, 16, 17, 64, 65]
        let maxBatch = batchSizes.max()!

        for (rows, cols, currentName, candidateName) in cases {
            let currentPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: currentName
            )
            try XCTSkipIf(currentPipeline == nil, "Could not compile \(currentName)")

            let candidatePipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: candidateName
            )
            try XCTSkipIf(candidatePipeline == nil, "Could not compile \(candidateName)")

            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 53)
            var inputData = [Float16](repeating: 0, count: maxBatch * cols)
            for b in 0..<maxBatch {
                for c in 0..<cols {
                    inputData[b * cols + c] = Float16(
                        sin(Float((b + 17) * (c + 29)) * 0.0053)
                    )
                }
            }

            let weightsBuf = device.makeBuffer(
                bytes: data.weights, length: data.weights.count, options: .storageModeShared
            )!
            let scalesBuf = device.makeBuffer(
                bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared
            )!
            let biasesBuf = device.makeBuffer(
                bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared
            )!
            let inputBuf = device.makeBuffer(
                bytes: inputData, length: inputData.count * 2, options: .storageModeShared
            )!
            let outputBytes = maxBatch * rows * 2
            let currentOut = device.makeBuffer(length: outputBytes, options: .storageModeShared)!
            let candidateOut = device.makeBuffer(length: outputBytes, options: .storageModeShared)!

            for batchSize in batchSizes {
                memset(currentOut.contents(), 0xA5, outputBytes)
                memset(candidateOut.contents(), 0x5A, outputBytes)

                func encode(_ pipeline: MTLComputePipelineState, output: MTLBuffer) throws {
                    let commandBuffer = queue.makeCommandBuffer()!
                    let encoder = commandBuffer.makeComputeCommandEncoder()!
                    encoder.setComputePipelineState(pipeline)
                    encoder.setBuffer(weightsBuf, offset: 0, index: 0)
                    encoder.setBuffer(scalesBuf, offset: 0, index: 1)
                    encoder.setBuffer(biasesBuf, offset: 0, index: 2)
                    encoder.setBuffer(inputBuf, offset: 0, index: 3)
                    encoder.setBuffer(output, offset: 0, index: 4)
                    var actualBatch = UInt32(batchSize)
                    encoder.setBytes(&actualBatch, length: 4, index: 5)
                    encoder.dispatchThreadgroups(
                        MTLSize(
                            width: (rows + 31) / 32,
                            height: (batchSize + 15) / 16,
                            depth: 1
                        ),
                        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
                    )
                    encoder.endEncoding()
                    commandBuffer.commit()
                    commandBuffer.waitUntilCompleted()
                    XCTAssertNil(commandBuffer.error)
                }

                try encode(currentPipeline!, output: currentOut)
                try encode(candidatePipeline!, output: candidateOut)

                let elementCount = batchSize * rows
                let currentBits = currentOut.contents().bindMemory(
                    to: UInt16.self, capacity: elementCount
                )
                let candidateBits = candidateOut.contents().bindMemory(
                    to: UInt16.self, capacity: elementCount
                )
                var firstMismatch: Int?
                for index in 0..<elementCount where currentBits[index] != candidateBits[index] {
                    firstMismatch = index
                    break
                }

                XCTAssertNil(
                    firstMismatch,
                    "\(candidateName) changed FP16 bits at batch \(batchSize), index \(firstMismatch ?? -1)"
                )
            }
        }
    }

    func testFusedAffineGateUpQwenBatchedFullNoPadding_MatchesCurrentBits() throws {
        let cases: [(rows: Int, cols: Int, current: String, candidate: String)] = [
            (
                3584, 1024,
                "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full_padded_reference",
                "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full"
            ),
            (
                6144, 2048,
                "fused_affine_gate_up_swiglu_c2048_r6144_g64_batched_full_padded_reference",
                "fused_affine_gate_up_swiglu_c2048_r6144_g64_batched_full"
            ),
        ]
        let names = cases.flatMap { [$0.current, $0.candidate] }
        let pipelines = makePipelines(shaderFile: "lut_matvec.metal", functionNames: names)
        try XCTSkipIf(pipelines == nil, "Could not compile Qwen gate/up padding parity pipelines")

        let batchSizes = [1, 17, 64, 65]
        let maxBatch = batchSizes.max()!
        for (rows, cols, currentName, candidateName) in cases {
            let gate = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 59)
            let up = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 61)
            var inputData = [Float16](repeating: 0, count: maxBatch * cols)
            for b in 0..<maxBatch {
                for c in 0..<cols {
                    inputData[b * cols + c] = Float16(
                        cos(Float((b + 19) * (c + 31)) * 0.0047)
                    )
                }
            }

            let gateW = device.makeBuffer(bytes: gate.weights, length: gate.weights.count, options: .storageModeShared)!
            let gateS = device.makeBuffer(bytes: gate.scales, length: gate.scales.count * 2, options: .storageModeShared)!
            let gateB = device.makeBuffer(bytes: gate.biases, length: gate.biases.count * 2, options: .storageModeShared)!
            let upW = device.makeBuffer(bytes: up.weights, length: up.weights.count, options: .storageModeShared)!
            let upS = device.makeBuffer(bytes: up.scales, length: up.scales.count * 2, options: .storageModeShared)!
            let upB = device.makeBuffer(bytes: up.biases, length: up.biases.count * 2, options: .storageModeShared)!
            let input = device.makeBuffer(bytes: inputData, length: inputData.count * 2, options: .storageModeShared)!
            let outputBytes = maxBatch * rows * 2
            let currentOut = device.makeBuffer(length: outputBytes, options: .storageModeShared)!
            let candidateOut = device.makeBuffer(length: outputBytes, options: .storageModeShared)!

            for batchSize in batchSizes {
                memset(currentOut.contents(), 0xA5, outputBytes)
                memset(candidateOut.contents(), 0x5A, outputBytes)
                let common = [0: gateW, 1: gateS, 2: gateB, 3: upW, 4: upS, 5: upB, 6: input]
                try runBatchedQMM(
                    pipeline: pipelines![currentName]!,
                    buffers: common.merging([7: currentOut]) { _, new in new },
                    actualBatchIndex: 8,
                    actualBatch: batchSize,
                    rows: rows
                )
                try runBatchedQMM(
                    pipeline: pipelines![candidateName]!,
                    buffers: common.merging([7: candidateOut]) { _, new in new },
                    actualBatchIndex: 8,
                    actualBatch: batchSize,
                    rows: rows
                )

                let mismatch = firstHalfBitMismatch(
                    currentOut, candidateOut, count: batchSize * rows
                )
                XCTAssertNil(
                    mismatch,
                    "\(candidateName) changed FP16 bits at batch \(batchSize), index \(mismatch ?? -1)"
                )
            }
        }
    }

    func testNormScaleQwenQMMNoPadding_MatchesCurrentBits() throws {
        let cases: [(
            cols: Int,
            affineRows: Int,
            gateRows: Int,
            affineCurrent: String,
            affineCandidate: String,
            gateCurrent: String,
            gateCandidate: String
        )] = [
            (
                1024,
                6144,
                3584,
                "norm_scale_affine_matvec_c1024_r6144_g64_batched_full_padded_reference",
                "norm_scale_affine_matvec_c1024_r6144_g64_batched_full",
                "norm_scale_affine_gate_up_swiglu_c1024_r3584_g64_batched_full_padded_reference",
                "norm_scale_affine_gate_up_swiglu_c1024_r3584_g64_batched_full"
            ),
            (
                2048,
                6144,
                6144,
                "norm_scale_affine_matvec_c2048_r6144_g64_batched_full_padded_reference",
                "norm_scale_affine_matvec_c2048_r6144_g64_batched_full",
                "norm_scale_affine_gate_up_swiglu_c2048_r6144_g64_batched_full_padded_reference",
                "norm_scale_affine_gate_up_swiglu_c2048_r6144_g64_batched_full"
            ),
        ]
        let names = cases.flatMap {
            [$0.affineCurrent, $0.affineCandidate, $0.gateCurrent, $0.gateCandidate]
        }
        let pipelines = makePipelines(shaderFile: "lut_matvec.metal", functionNames: names)
        try XCTSkipIf(pipelines == nil, "Could not compile Qwen norm-scale QMM padding parity pipelines")

        let batchSizes = [1, 17, 65]
        let maxBatch = batchSizes.max()!
        for testCase in cases {
            let cols = testCase.cols
            let affineRows = testCase.affineRows
            let gateRows = testCase.gateRows
            let affine = makeAffineTestData(rows: affineRows, cols: cols, groupSize: 64, seed: 67)
            let gate = makeAffineTestData(rows: gateRows, cols: cols, groupSize: 64, seed: 71)
            let up = makeAffineTestData(rows: gateRows, cols: cols, groupSize: 64, seed: 73)

            var inputData = [Float16](repeating: 0, count: maxBatch * cols)
            var normWeightData = [Float16](repeating: 0, count: cols)
            var scaleData = [Float](repeating: 0, count: maxBatch)
            for b in 0..<maxBatch {
                scaleData[b] = 0.7 + Float(b % 7) * 0.03125
                for c in 0..<cols {
                    inputData[b * cols + c] = Float16(
                        sin(Float((b + 23) * (c + 37)) * 0.0039)
                    )
                }
            }
            for c in 0..<cols {
                normWeightData[c] = Float16(cos(Float(c + 41) * 0.0071) * 0.075)
            }

            let scale = device.makeBuffer(bytes: scaleData, length: scaleData.count * 4, options: .storageModeShared)!
            let input = device.makeBuffer(bytes: inputData, length: inputData.count * 2, options: .storageModeShared)!
            let normWeight = device.makeBuffer(bytes: normWeightData, length: normWeightData.count * 2, options: .storageModeShared)!
            let affineW = device.makeBuffer(bytes: affine.weights, length: affine.weights.count, options: .storageModeShared)!
            let affineS = device.makeBuffer(bytes: affine.scales, length: affine.scales.count * 2, options: .storageModeShared)!
            let affineB = device.makeBuffer(bytes: affine.biases, length: affine.biases.count * 2, options: .storageModeShared)!
            let gateW = device.makeBuffer(bytes: gate.weights, length: gate.weights.count, options: .storageModeShared)!
            let gateS = device.makeBuffer(bytes: gate.scales, length: gate.scales.count * 2, options: .storageModeShared)!
            let gateB = device.makeBuffer(bytes: gate.biases, length: gate.biases.count * 2, options: .storageModeShared)!
            let upW = device.makeBuffer(bytes: up.weights, length: up.weights.count, options: .storageModeShared)!
            let upS = device.makeBuffer(bytes: up.scales, length: up.scales.count * 2, options: .storageModeShared)!
            let upB = device.makeBuffer(bytes: up.biases, length: up.biases.count * 2, options: .storageModeShared)!

            let normBytes = maxBatch * cols * 2
            let affineBytes = maxBatch * affineRows * 2
            let gateBytes = maxBatch * gateRows * 2
            let affineCurrentNorm = device.makeBuffer(length: normBytes, options: .storageModeShared)!
            let affineCandidateNorm = device.makeBuffer(length: normBytes, options: .storageModeShared)!
            let affineCurrentOut = device.makeBuffer(length: affineBytes, options: .storageModeShared)!
            let affineCandidateOut = device.makeBuffer(length: affineBytes, options: .storageModeShared)!
            let gateCurrentNorm = device.makeBuffer(length: normBytes, options: .storageModeShared)!
            let gateCandidateNorm = device.makeBuffer(length: normBytes, options: .storageModeShared)!
            let gateCurrentOut = device.makeBuffer(length: gateBytes, options: .storageModeShared)!
            let gateCandidateOut = device.makeBuffer(length: gateBytes, options: .storageModeShared)!

            for batchSize in batchSizes {
                memset(affineCurrentNorm.contents(), 0xA5, normBytes)
                memset(affineCandidateNorm.contents(), 0x5A, normBytes)
                memset(affineCurrentOut.contents(), 0xA5, affineBytes)
                memset(affineCandidateOut.contents(), 0x5A, affineBytes)
                try runBatchedQMM(
                    pipeline: pipelines![testCase.affineCurrent]!,
                    buffers: [
                        0: scale, 1: input, 2: normWeight, 3: affineCurrentNorm,
                        4: affineW, 5: affineS, 6: affineB, 7: affineCurrentOut,
                    ],
                    actualBatchIndex: 8,
                    actualBatch: batchSize,
                    rows: affineRows
                )
                try runBatchedQMM(
                    pipeline: pipelines![testCase.affineCandidate]!,
                    buffers: [
                        0: scale, 1: input, 2: normWeight, 3: affineCandidateNorm,
                        4: affineW, 5: affineS, 6: affineB, 7: affineCandidateOut,
                    ],
                    actualBatchIndex: 8,
                    actualBatch: batchSize,
                    rows: affineRows
                )
                XCTAssertNil(
                    firstHalfBitMismatch(
                        affineCurrentNorm, affineCandidateNorm, count: batchSize * cols
                    ),
                    "\(testCase.affineCandidate) changed norm FP16 bits at batch \(batchSize)"
                )
                XCTAssertNil(
                    firstHalfBitMismatch(
                        affineCurrentOut, affineCandidateOut, count: batchSize * affineRows
                    ),
                    "\(testCase.affineCandidate) changed output FP16 bits at batch \(batchSize)"
                )

                memset(gateCurrentNorm.contents(), 0xA5, normBytes)
                memset(gateCandidateNorm.contents(), 0x5A, normBytes)
                memset(gateCurrentOut.contents(), 0xA5, gateBytes)
                memset(gateCandidateOut.contents(), 0x5A, gateBytes)
                try runBatchedQMM(
                    pipeline: pipelines![testCase.gateCurrent]!,
                    buffers: [
                        0: scale, 1: input, 2: normWeight, 3: gateCurrentNorm,
                        4: gateW, 5: gateS, 6: gateB,
                        7: upW, 8: upS, 9: upB, 10: gateCurrentOut,
                    ],
                    actualBatchIndex: 11,
                    actualBatch: batchSize,
                    rows: gateRows
                )
                try runBatchedQMM(
                    pipeline: pipelines![testCase.gateCandidate]!,
                    buffers: [
                        0: scale, 1: input, 2: normWeight, 3: gateCandidateNorm,
                        4: gateW, 5: gateS, 6: gateB,
                        7: upW, 8: upS, 9: upB, 10: gateCandidateOut,
                    ],
                    actualBatchIndex: 11,
                    actualBatch: batchSize,
                    rows: gateRows
                )
                XCTAssertNil(
                    firstHalfBitMismatch(
                        gateCurrentNorm, gateCandidateNorm, count: batchSize * cols
                    ),
                    "\(testCase.gateCandidate) changed norm FP16 bits at batch \(batchSize)"
                )
                XCTAssertNil(
                    firstHalfBitMismatch(
                        gateCurrentOut, gateCandidateOut, count: batchSize * gateRows
                    ),
                    "\(testCase.gateCandidate) changed output FP16 bits at batch \(batchSize)"
                )
            }
        }
    }

    func testFusedAffineMatvecAddBatchedFullSpecializations_MatchStagedFullPlusResidual() throws {
        let addPipeline = makePipeline(
            shaderFile: "activations.metal",
            functionName: "elementwise_add"
        )
        try XCTSkipIf(addPipeline == nil, "Could not compile elementwise_add")

        let cases: [(rows: Int, cols: Int, staged: String, fused: String)] = [
            (
                1024, 2048,
                "affine_matvec_c2048_r1024_g64_batched_full",
                "fused_affine_matvec_add_c2048_r1024_g64_batched_full"
            ),
            (
                1024, 3584,
                "affine_matvec_c3584_r1024_g64_batched_full",
                "fused_affine_matvec_add_c3584_r1024_g64_batched_full"
            ),
            (
                2048, 2048,
                "affine_matvec_c2048_r2048_g64_batched_full",
                "fused_affine_matvec_add_c2048_r2048_g64_batched_full"
            ),
            (
                2048, 6144,
                "affine_matvec_c6144_r2048_g64_batched_full",
                "fused_affine_matvec_add_c6144_r2048_g64_batched_full"
            ),
            (
                2048, 8192,
                "affine_matvec_c8192_r2048_g64_batched_full",
                "fused_affine_matvec_add_c8192_r2048_g64_batched_full"
            ),
            (
                2560, 4096,
                "affine_matvec_c4096_r2560_g64_batched_full",
                "fused_affine_matvec_add_c4096_r2560_g64_batched_full"
            ),
            (
                2560, 9216,
                "affine_matvec_c9216_r2560_g64_batched_full",
                "fused_affine_matvec_add_c9216_r2560_g64_batched_full"
            ),
        ]

        let batchSize = 19
        for (rows, cols, stagedName, fusedName) in cases {
            let stagedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: stagedName
            )
            try XCTSkipIf(stagedPipeline == nil, "Could not compile \(stagedName)")

            let fusedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: fusedName
            )
            try XCTSkipIf(fusedPipeline == nil, "Could not compile \(fusedName)")

            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 47)
            var inputData = [Float16](repeating: 0, count: batchSize * cols)
            var residualData = [Float16](repeating: 0, count: batchSize * rows)
            for b in 0..<batchSize {
                for c in 0..<cols {
                    inputData[b * cols + c] = Float16(sin(Float((b + 5) * (c + 13)) * 0.006))
                }
                for r in 0..<rows {
                    residualData[b * rows + r] = Float16(cos(Float((b + 7) * (r + 11)) * 0.004) * 0.35)
                }
            }

            let weightsBuf = device.makeBuffer(
                bytes: data.weights, length: data.weights.count, options: .storageModeShared
            )!
            let scalesBuf = device.makeBuffer(
                bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared
            )!
            let biasesBuf = device.makeBuffer(
                bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared
            )!
            let inputBuf = device.makeBuffer(
                bytes: inputData, length: inputData.count * 2, options: .storageModeShared
            )!
            let residualBuf = device.makeBuffer(
                bytes: residualData, length: residualData.count * 2, options: .storageModeShared
            )!
            let stagedMatvecOut = device.makeBuffer(
                length: batchSize * rows * 2, options: .storageModeShared
            )!
            let stagedOut = device.makeBuffer(
                length: batchSize * rows * 2, options: .storageModeShared
            )!
            let fusedMatvecOut = device.makeBuffer(
                length: batchSize * rows * 2, options: .storageModeShared
            )!
            let fusedOut = device.makeBuffer(
                length: batchSize * rows * 2, options: .storageModeShared
            )!
            let aliasedMatvecOut = device.makeBuffer(
                length: batchSize * rows * 2, options: .storageModeShared
            )!
            let aliasedOut = device.makeBuffer(
                bytes: residualData, length: residualData.count * 2, options: .storageModeShared
            )!
            memset(stagedMatvecOut.contents(), 0, batchSize * rows * 2)
            memset(stagedOut.contents(), 0, batchSize * rows * 2)
            memset(fusedMatvecOut.contents(), 0, batchSize * rows * 2)
            memset(fusedOut.contents(), 0, batchSize * rows * 2)
            memset(aliasedMatvecOut.contents(), 0, batchSize * rows * 2)

            let rowTile = 32
            let batchTile = 16
            let tgWidth = 128
            let grid = MTLSize(
                width: (rows + rowTile - 1) / rowTile,
                height: (batchSize + batchTile - 1) / batchTile,
                depth: 1
            )
            let threads = MTLSize(width: tgWidth, height: 1, depth: 1)

            let stagedCmdBuf = queue.makeCommandBuffer()!
            let stagedEnc = stagedCmdBuf.makeComputeCommandEncoder()!
            stagedEnc.setComputePipelineState(stagedPipeline!)
            stagedEnc.setBuffer(weightsBuf, offset: 0, index: 0)
            stagedEnc.setBuffer(scalesBuf, offset: 0, index: 1)
            stagedEnc.setBuffer(biasesBuf, offset: 0, index: 2)
            stagedEnc.setBuffer(inputBuf, offset: 0, index: 3)
            stagedEnc.setBuffer(stagedMatvecOut, offset: 0, index: 4)
            var actualBatch = UInt32(batchSize)
            stagedEnc.setBytes(&actualBatch, length: 4, index: 5)
            stagedEnc.dispatchThreadgroups(grid, threadsPerThreadgroup: threads)
            stagedEnc.endEncoding()

            let addEnc = stagedCmdBuf.makeComputeCommandEncoder()!
            addEnc.setComputePipelineState(addPipeline!)
            addEnc.setBuffer(stagedMatvecOut, offset: 0, index: 0)
            addEnc.setBuffer(residualBuf, offset: 0, index: 1)
            addEnc.setBuffer(stagedOut, offset: 0, index: 2)
            var countVal = UInt32(batchSize * rows)
            addEnc.setBytes(&countVal, length: 4, index: 3)
            addEnc.dispatchThreads(
                MTLSize(width: batchSize * rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1_024, height: 1, depth: 1)
            )
            addEnc.endEncoding()
            stagedCmdBuf.commit()
            stagedCmdBuf.waitUntilCompleted()
            XCTAssertNil(stagedCmdBuf.error)

            let fusedCmdBuf = queue.makeCommandBuffer()!
            let fusedEnc = fusedCmdBuf.makeComputeCommandEncoder()!
            fusedEnc.setComputePipelineState(fusedPipeline!)
            fusedEnc.setBuffer(weightsBuf, offset: 0, index: 0)
            fusedEnc.setBuffer(scalesBuf, offset: 0, index: 1)
            fusedEnc.setBuffer(biasesBuf, offset: 0, index: 2)
            fusedEnc.setBuffer(inputBuf, offset: 0, index: 3)
            fusedEnc.setBuffer(fusedMatvecOut, offset: 0, index: 4)
            fusedEnc.setBuffer(residualBuf, offset: 0, index: 5)
            fusedEnc.setBuffer(fusedOut, offset: 0, index: 6)
            fusedEnc.setBytes(&actualBatch, length: 4, index: 7)
            fusedEnc.dispatchThreadgroups(grid, threadsPerThreadgroup: threads)
            fusedEnc.endEncoding()
            fusedCmdBuf.commit()
            fusedCmdBuf.waitUntilCompleted()
            XCTAssertNil(fusedCmdBuf.error)

            let aliasCmdBuf = queue.makeCommandBuffer()!
            let aliasEnc = aliasCmdBuf.makeComputeCommandEncoder()!
            aliasEnc.setComputePipelineState(fusedPipeline!)
            aliasEnc.setBuffer(weightsBuf, offset: 0, index: 0)
            aliasEnc.setBuffer(scalesBuf, offset: 0, index: 1)
            aliasEnc.setBuffer(biasesBuf, offset: 0, index: 2)
            aliasEnc.setBuffer(inputBuf, offset: 0, index: 3)
            aliasEnc.setBuffer(aliasedMatvecOut, offset: 0, index: 4)
            aliasEnc.setBuffer(aliasedOut, offset: 0, index: 5)
            aliasEnc.setBuffer(aliasedOut, offset: 0, index: 6)
            aliasEnc.setBytes(&actualBatch, length: 4, index: 7)
            aliasEnc.dispatchThreadgroups(grid, threadsPerThreadgroup: threads)
            aliasEnc.endEncoding()
            aliasCmdBuf.commit()
            aliasCmdBuf.waitUntilCompleted()
            XCTAssertNil(aliasCmdBuf.error)

            let stagedPtr = stagedOut.contents().bindMemory(to: Float16.self, capacity: batchSize * rows)
            let stagedMatvecPtr = stagedMatvecOut.contents().bindMemory(to: Float16.self, capacity: batchSize * rows)
            let fusedMatvecPtr = fusedMatvecOut.contents().bindMemory(to: Float16.self, capacity: batchSize * rows)
            let fusedPtr = fusedOut.contents().bindMemory(to: Float16.self, capacity: batchSize * rows)
            let aliasedPtr = aliasedOut.contents().bindMemory(to: Float16.self, capacity: batchSize * rows)
            var maxMatvecDiff: Float = 0
            var maxDiff: Float = 0
            var maxAliasedDiff: Float = 0
            for i in 0..<(batchSize * rows) {
                let matvecDiff = abs(Float(stagedMatvecPtr[i]) - Float(fusedMatvecPtr[i]))
                if matvecDiff > maxMatvecDiff { maxMatvecDiff = matvecDiff }
                let diff = abs(Float(stagedPtr[i]) - Float(fusedPtr[i]))
                if diff > maxDiff { maxDiff = diff }
                let aliasedDiff = abs(Float(stagedPtr[i]) - Float(aliasedPtr[i]))
                if aliasedDiff > maxAliasedDiff { maxAliasedDiff = aliasedDiff }
            }

            fputs(
                "  \(fusedName) vs staged full+add:"
                    + " matvec max diff = \(String(format: "%.6f", maxMatvecDiff))"
                    + " max diff = \(String(format: "%.6f", maxDiff))"
                    + ", aliased max diff = \(String(format: "%.6f", maxAliasedDiff))\n",
                stderr
            )
            XCTAssertEqual(
                maxMatvecDiff,
                0,
                accuracy: 0.000001,
                "\(fusedName) intermediate diverges from staged full"
            )
            XCTAssertEqual(maxDiff, 0, accuracy: 0.000001, "\(fusedName) diverges from staged full+add")
            XCTAssertEqual(
                maxAliasedDiff,
                0,
                accuracy: 0.000001,
                "\(fusedName) diverges when residual/output alias"
            )
        }
    }

    func testAffineMatvecQwenBatchedFullSpecialization_LargeBatchBoundaries() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["SMELT_RUN_SLOW_TESTS"] == "1",
            "Large-batch prefill boundary coverage is opt-in"
        )

        let rows = 2048
        let cols = 2048
        let functionName = "affine_matvec_c2048_r2048_g64_batched_full"
        let referencePipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "affine_matvec_c2048_r2048_g64_batched"
        )
        try XCTSkipIf(referencePipeline == nil, "Could not compile scalar batched affine matvec shader")

        let fullPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: functionName
        )
        try XCTSkipIf(fullPipeline == nil, "Could not compile \(functionName)")

        for batchSize in [64, 65] {
            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 41)
            var inputData = [Float16](repeating: 0, count: batchSize * cols)
            for b in 0..<batchSize {
                for c in 0..<cols {
                    inputData[b * cols + c] = Float16(cos(Float((b + 11) * (c + 5)) * 0.007))
                }
            }

            let weightsBuf = device.makeBuffer(
                bytes: data.weights, length: data.weights.count, options: .storageModeShared
            )!
            let scalesBuf = device.makeBuffer(
                bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared
            )!
            let biasesBuf = device.makeBuffer(
                bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared
            )!
            let inputBuf = device.makeBuffer(
                bytes: inputData, length: inputData.count * 2, options: .storageModeShared
            )!
            let outBuf = device.makeBuffer(length: batchSize * rows * 2, options: .storageModeShared)!
            memset(outBuf.contents(), 0, batchSize * rows * 2)

            let refBuf = device.makeBuffer(length: batchSize * rows * 2, options: .storageModeShared)!
            memset(refBuf.contents(), 0, batchSize * rows * 2)

            let refCmdBuf = queue.makeCommandBuffer()!
            let refEnc = refCmdBuf.makeComputeCommandEncoder()!
            refEnc.setComputePipelineState(referencePipeline!)
            refEnc.setBuffer(weightsBuf, offset: 0, index: 0)
            refEnc.setBuffer(scalesBuf, offset: 0, index: 1)
            refEnc.setBuffer(biasesBuf, offset: 0, index: 2)
            refEnc.setBuffer(inputBuf, offset: 0, index: 3)
            refEnc.setBuffer(refBuf, offset: 0, index: 4)
            var refActualBatch = UInt32(batchSize)
            refEnc.setBytes(&refActualBatch, length: 4, index: 5)
            refEnc.dispatchThreadgroups(
                MTLSize(
                    width: rows / 8,
                    height: (batchSize + 7) / 8,
                    depth: 1
                ),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            refEnc.endEncoding()
            refCmdBuf.commit()
            refCmdBuf.waitUntilCompleted()
            XCTAssertNil(refCmdBuf.error)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(fullPipeline!)
            enc.setBuffer(weightsBuf, offset: 0, index: 0)
            enc.setBuffer(scalesBuf, offset: 0, index: 1)
            enc.setBuffer(biasesBuf, offset: 0, index: 2)
            enc.setBuffer(inputBuf, offset: 0, index: 3)
            enc.setBuffer(outBuf, offset: 0, index: 4)
            var actualBatch = UInt32(batchSize)
            enc.setBytes(&actualBatch, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(
                    width: rows / 32,
                    height: (batchSize + 15) / 16,
                    depth: 1
                ),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let outPtr = outBuf.contents().bindMemory(to: Float16.self, capacity: batchSize * rows)
            let refPtr = refBuf.contents().bindMemory(to: Float16.self, capacity: batchSize * rows)
            var maxDiff: Float = 0
            var worstBatch = 0
            for i in 0..<(batchSize * rows) {
                let diff = abs(Float(outPtr[i]) - Float(refPtr[i]))
                if diff > maxDiff {
                    maxDiff = diff
                    worstBatch = i / rows
                }
            }

            fputs(
                "  \(functionName) batch=\(batchSize): max diff = \(String(format: "%.6f", maxDiff))"
                    + " worstBatch=\(worstBatch)\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.07, "\(functionName) diverges from scalar batched affine matvec at batch \(batchSize)")
        }
    }

    func testAffineMatvecQwen0808BatchedFullSpecializations_LargeBatchBoundaries() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["SMELT_RUN_QWEN_SMOKE"] == "1",
            "0.8B large-batch prefill boundary coverage is opt-in"
        )

        let cases: [(rows: Int, cols: Int, functionName: String)] = [
            (2048, 1024, "affine_matvec_c1024_r2048_g64_batched_full"),
            (6144, 1024, "affine_matvec_c1024_r6144_g64_batched_full"),
            (1024, 2048, "affine_matvec_c2048_r1024_g64_batched_full"),
            (3584, 1024, "affine_matvec_c1024_r3584_g64_batched_full"),
            (1024, 3584, "affine_matvec_c3584_r1024_g64_batched_full"),
        ]

        for (rows, cols, functionName) in cases {
            let genericPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: "affine_matvec",
                cols: UInt32(cols),
                groupSize: 64
            )
            try XCTSkipIf(genericPipeline == nil, "Could not compile generic affine matvec shader")

            let fullPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: functionName
            )
            try XCTSkipIf(fullPipeline == nil, "Could not compile \(functionName)")

            for batchSize in [64, 65] {
                let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64)
                var inputData = [Float16](repeating: 0, count: batchSize * cols)
                for b in 0..<batchSize {
                    for c in 0..<cols {
                        inputData[b * cols + c] = Float16(cos(Float((b + 11) * (c + 5)) * 0.007))
                    }
                }

                let weightsBuf = device.makeBuffer(
                    bytes: data.weights, length: data.weights.count, options: .storageModeShared
                )!
                let scalesBuf = device.makeBuffer(
                    bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared
                )!
                let biasesBuf = device.makeBuffer(
                    bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared
                )!
                let inputBuf = device.makeBuffer(
                    bytes: inputData, length: inputData.count * 2, options: .storageModeShared
                )!
                let outBuf = device.makeBuffer(length: batchSize * rows * 2, options: .storageModeShared)!
                memset(outBuf.contents(), 0, batchSize * rows * 2)

                let reference = try runIteratedAffineMatvec(
                    pipeline: genericPipeline!,
                    weights: weightsBuf,
                    scales: scalesBuf,
                    biases: biasesBuf,
                    inputData: inputData,
                    rows: rows,
                    cols: cols,
                    batchSize: batchSize
                )

                let cmdBuf = queue.makeCommandBuffer()!
                let enc = cmdBuf.makeComputeCommandEncoder()!
                enc.setComputePipelineState(fullPipeline!)
                enc.setBuffer(weightsBuf, offset: 0, index: 0)
                enc.setBuffer(scalesBuf, offset: 0, index: 1)
                enc.setBuffer(biasesBuf, offset: 0, index: 2)
                enc.setBuffer(inputBuf, offset: 0, index: 3)
                enc.setBuffer(outBuf, offset: 0, index: 4)
                var actualBatch = UInt32(batchSize)
                enc.setBytes(&actualBatch, length: 4, index: 5)
                enc.dispatchThreadgroups(
                    MTLSize(
                        width: (rows + 31) / 32,
                        height: (batchSize + 15) / 16,
                        depth: 1
                    ),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
                )
                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()
                XCTAssertNil(cmdBuf.error)

                let outPtr = outBuf.contents().bindMemory(to: Float16.self, capacity: batchSize * rows)
                var maxDiff: Float = 0
                for i in 0..<(batchSize * rows) {
                    maxDiff = max(maxDiff, abs(Float(outPtr[i]) - reference[i]))
                }

                fputs(
                    "  \(functionName) batch=\(batchSize): max diff = \(String(format: "%.6f", maxDiff))\n",
                    stderr
                )
                XCTAssertLessThan(maxDiff, 0.07, "\(functionName) diverges at batch boundary \(batchSize)")
            }
        }
    }

    func testFusedAffineGateUpSwigluQwenQMMSpecialization_MatchesIterated() throws {
        let cases: [(rows: Int, cols: Int, functionName: String)] = [
            (3584, 1024, "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full"),
            (6144, 2048, "fused_affine_gate_up_swiglu_c2048_r6144_g64_batched_full"),
            (9216, 2560, "fused_affine_gate_up_swiglu_c2560_r9216_g64_batched_full"),
        ]

        for (rows, cols, functionName) in cases {
            let genericPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: "fused_affine_gate_up_swiglu",
                cols: UInt32(cols),
                groupSize: 64
            )
            try XCTSkipIf(genericPipeline == nil, "Could not compile generic fused affine FFN shader")

            let batchedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: functionName
            )
            try XCTSkipIf(
                batchedPipeline == nil,
                "Could not compile \(functionName)"
            )

            let batchSize = 2
            let usesQMMFull = true
            let batchTile = usesQMMFull ? 16 : 8
            let rowTile = usesQMMFull ? 32 : 8
            let tgWidth = usesQMMFull ? 128 : 64
            let gate = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 51)
            let up = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 52)
            var inputData = [Float16](repeating: 0, count: batchSize * cols)
            for b in 0..<batchSize {
                for c in 0..<cols {
                    inputData[b * cols + c] = Float16(cos(Float((b + 5) * (c + 1)) * 0.009))
                }
            }

            let gateW = device.makeBuffer(bytes: gate.weights, length: gate.weights.count, options: .storageModeShared)!
            let gateS = device.makeBuffer(bytes: gate.scales, length: gate.scales.count * 2, options: .storageModeShared)!
            let gateB = device.makeBuffer(bytes: gate.biases, length: gate.biases.count * 2, options: .storageModeShared)!
            let upW = device.makeBuffer(bytes: up.weights, length: up.weights.count, options: .storageModeShared)!
            let upS = device.makeBuffer(bytes: up.scales, length: up.scales.count * 2, options: .storageModeShared)!
            let upB = device.makeBuffer(bytes: up.biases, length: up.biases.count * 2, options: .storageModeShared)!
            let inputBuf = device.makeBuffer(bytes: inputData, length: inputData.count * 2, options: .storageModeShared)!
            let batchedOut = device.makeBuffer(length: batchSize * rows * 2, options: .storageModeShared)!
            memset(batchedOut.contents(), 0, batchSize * rows * 2)

            let reference = try runIteratedFusedAffineGateUpSwiglu(
                pipeline: genericPipeline!,
                gateWeights: gateW,
                gateScales: gateS,
                gateBiases: gateB,
                upWeights: upW,
                upScales: upS,
                upBiases: upB,
                inputData: inputData,
                rows: rows,
                cols: cols,
                batchSize: batchSize
            )

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(batchedPipeline!)
            enc.setBuffer(gateW, offset: 0, index: 0)
            enc.setBuffer(gateS, offset: 0, index: 1)
            enc.setBuffer(gateB, offset: 0, index: 2)
            enc.setBuffer(upW, offset: 0, index: 3)
            enc.setBuffer(upS, offset: 0, index: 4)
            enc.setBuffer(upB, offset: 0, index: 5)
            enc.setBuffer(inputBuf, offset: 0, index: 6)
            enc.setBuffer(batchedOut, offset: 0, index: 7)
            var actualBatch = UInt32(batchSize)
            enc.setBytes(&actualBatch, length: 4, index: 8)
            enc.dispatchThreadgroups(
                MTLSize(
                    width: rows / rowTile,
                    height: (batchSize + batchTile - 1) / batchTile,
                    depth: 1
                ),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let outPtr = batchedOut.contents().bindMemory(
                to: Float16.self, capacity: batchSize * rows
            )
            var maxDiff: Float = 0
            for i in 0..<(batchSize * rows) {
                let diff = abs(Float(outPtr[i]) - reference[i])
                if diff > maxDiff { maxDiff = diff }
            }

            fputs(
                "  \(functionName) vs iterated: max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.04, "\(functionName) diverges from iterated generic")
        }
    }

    func testFusedAffineGateUpSwigluQwenBatchedSpecialization_LargeBatchBoundaries() throws {
        let slow = ProcessInfo.processInfo.environment["SMELT_RUN_SLOW_TESTS"] == "1"
        try XCTSkipUnless(slow, "Large-batch prefill FFN boundary coverage is opt-in")

        let genericPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "fused_affine_gate_up_swiglu",
            cols: 2048,
            groupSize: 64
        )
        try XCTSkipIf(genericPipeline == nil, "Could not compile generic fused affine shader")

        let batchedPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "fused_affine_gate_up_swiglu_c2048_r6144_g64_batched"
        )
        try XCTSkipIf(batchedPipeline == nil, "Could not compile batched fused affine shader")

        let rows = 6144
        let cols = 2048
        let batchTile = 2
        let rowTile = 4

        for batchSize in [64, 65] {
            let gate = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 51)
            let up = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 52)
            var inputData = [Float16](repeating: 0, count: batchSize * cols)
            for b in 0..<batchSize {
                for c in 0..<cols {
                    inputData[b * cols + c] = Float16(cos(Float((b + 9) * (c + 1)) * 0.005))
                }
            }

            let gateW = device.makeBuffer(bytes: gate.weights, length: gate.weights.count, options: .storageModeShared)!
            let gateS = device.makeBuffer(bytes: gate.scales, length: gate.scales.count * 2, options: .storageModeShared)!
            let gateB = device.makeBuffer(bytes: gate.biases, length: gate.biases.count * 2, options: .storageModeShared)!
            let upW = device.makeBuffer(bytes: up.weights, length: up.weights.count, options: .storageModeShared)!
            let upS = device.makeBuffer(bytes: up.scales, length: up.scales.count * 2, options: .storageModeShared)!
            let upB = device.makeBuffer(bytes: up.biases, length: up.biases.count * 2, options: .storageModeShared)!
            let inputBuf = device.makeBuffer(bytes: inputData, length: inputData.count * 2, options: .storageModeShared)!
            let batchedOut = device.makeBuffer(length: batchSize * rows * 2, options: .storageModeShared)!
            memset(batchedOut.contents(), 0, batchSize * rows * 2)

            let reference = try runIteratedFusedAffineGateUpSwiglu(
                pipeline: genericPipeline!,
                gateWeights: gateW,
                gateScales: gateS,
                gateBiases: gateB,
                upWeights: upW,
                upScales: upS,
                upBiases: upB,
                inputData: inputData,
                rows: rows,
                cols: cols,
                batchSize: batchSize
            )

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(batchedPipeline!)
            enc.setBuffer(gateW, offset: 0, index: 0)
            enc.setBuffer(gateS, offset: 0, index: 1)
            enc.setBuffer(gateB, offset: 0, index: 2)
            enc.setBuffer(upW, offset: 0, index: 3)
            enc.setBuffer(upS, offset: 0, index: 4)
            enc.setBuffer(upB, offset: 0, index: 5)
            enc.setBuffer(inputBuf, offset: 0, index: 6)
            enc.setBuffer(batchedOut, offset: 0, index: 7)
            var actualBatch = UInt32(batchSize)
            enc.setBytes(&actualBatch, length: 4, index: 8)
            enc.dispatchThreadgroups(
                MTLSize(
                    width: (rows + rowTile - 1) / rowTile,
                    height: (batchSize + batchTile - 1) / batchTile,
                    depth: 1
                ),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let outPtr = batchedOut.contents().bindMemory(to: Float16.self, capacity: batchSize * rows)
            var maxDiff: Float = 0
            for i in 0..<(batchSize * rows) {
                maxDiff = max(maxDiff, abs(Float(outPtr[i]) - reference[i]))
            }

            fputs(
                "  fused_affine_gate_up_swiglu_c2048_r6144_g64_batched batch=\(batchSize):"
                    + " max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.04, "Scalar batched FFN diverges at batch boundary \(batchSize)")
        }
    }

    func testFusedAffineGateUpSwigluQwenBatchedFullSpecialization_MatchesIterated() throws {
        let genericPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "fused_affine_gate_up_swiglu",
            cols: 2048,
            groupSize: 64
        )
        try XCTSkipIf(genericPipeline == nil, "Could not compile generic fused affine FFN shader")

        let fullPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "fused_affine_gate_up_swiglu_c2048_r6144_g64_batched_full"
        )
        try XCTSkipIf(
            fullPipeline == nil,
            "Could not compile full batched fused affine FFN shader"
        )

        let rows = 6144
        let cols = 2048
        let batchSize = 16
        let batchTile = 16
        let gate = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 11)
        let up = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 12)
        var inputData = [Float16](repeating: 0, count: batchSize * cols)
        for b in 0..<batchSize {
            for c in 0..<cols {
                inputData[b * cols + c] = Float16(cos(Float((b + 9) * (c + 4)) * 0.008))
            }
        }

        let gateW = device.makeBuffer(bytes: gate.weights, length: gate.weights.count, options: .storageModeShared)!
        let gateS = device.makeBuffer(bytes: gate.scales, length: gate.scales.count * 2, options: .storageModeShared)!
        let gateB = device.makeBuffer(bytes: gate.biases, length: gate.biases.count * 2, options: .storageModeShared)!
        let upW = device.makeBuffer(bytes: up.weights, length: up.weights.count, options: .storageModeShared)!
        let upS = device.makeBuffer(bytes: up.scales, length: up.scales.count * 2, options: .storageModeShared)!
        let upB = device.makeBuffer(bytes: up.biases, length: up.biases.count * 2, options: .storageModeShared)!
        let inputBuf = device.makeBuffer(bytes: inputData, length: inputData.count * 2, options: .storageModeShared)!
        let batchedOut = device.makeBuffer(length: batchSize * rows * 2, options: .storageModeShared)!
        memset(batchedOut.contents(), 0, batchSize * rows * 2)

        let reference = try runIteratedFusedAffineGateUpSwiglu(
            pipeline: genericPipeline!,
            gateWeights: gateW,
            gateScales: gateS,
            gateBiases: gateB,
            upWeights: upW,
            upScales: upS,
            upBiases: upB,
            inputData: inputData,
            rows: rows,
            cols: cols,
            batchSize: batchSize
        )

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fullPipeline!)
        enc.setBuffer(gateW, offset: 0, index: 0)
        enc.setBuffer(gateS, offset: 0, index: 1)
        enc.setBuffer(gateB, offset: 0, index: 2)
        enc.setBuffer(upW, offset: 0, index: 3)
        enc.setBuffer(upS, offset: 0, index: 4)
        enc.setBuffer(upB, offset: 0, index: 5)
        enc.setBuffer(inputBuf, offset: 0, index: 6)
        enc.setBuffer(batchedOut, offset: 0, index: 7)
        var actualBatch = UInt32(batchSize)
        enc.setBytes(&actualBatch, length: 4, index: 8)
        enc.dispatchThreadgroups(
            MTLSize(
                width: rows / 32,
                height: (batchSize + batchTile - 1) / batchTile,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = batchedOut.contents().bindMemory(
            to: Float16.self, capacity: batchSize * rows
        )
        var maxDiff: Float = 0
        for i in 0..<(batchSize * rows) {
            let diff = abs(Float(outPtr[i]) - reference[i])
            if diff > maxDiff { maxDiff = diff }
        }

        fputs(
            "  fused_affine_gate_up_swiglu_c2048_r6144_g64_batched_full vs iterated:"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        // The qmm full path reorders accumulation through simdgroup_matrix tiles and
        // applies SwiGLU after the reordered accumulation, so adversarial synthetic
        // gate/up weights can diverge materially from the scalar reference even though
        // real-model parity is covered by package-level verify/smoke tests.
        XCTAssertLessThan(maxDiff, 0.3, "Full batched fused affine FFN diverges from iterated generic")
    }

    func testFusedAffineGateUpSwigluLlamaBatchedFullSpecialization_MatchesIterated() throws {
        let rows = 8192
        let cols = 2048
        let batchSize = 8
        let batchTile = 16
        let functionName = "fused_affine_gate_up_swiglu_c2048_r8192_g64_batched_full"
        let genericPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "fused_affine_gate_up_swiglu",
            cols: UInt32(cols),
            groupSize: 64
        )
        try XCTSkipIf(genericPipeline == nil, "Could not compile generic fused affine FFN shader")

        let fullPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: functionName
        )
        try XCTSkipIf(
            fullPipeline == nil,
            "Could not compile full batched fused affine FFN shader"
        )

        let gate = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 21)
        let up = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 22)
        var inputData = [Float16](repeating: 0, count: batchSize * cols)
        for b in 0..<batchSize {
            for c in 0..<cols {
                inputData[b * cols + c] = Float16(cos(Float((b + 5) * (c + 11)) * 0.006))
            }
        }

        let gateW = device.makeBuffer(bytes: gate.weights, length: gate.weights.count, options: .storageModeShared)!
        let gateS = device.makeBuffer(bytes: gate.scales, length: gate.scales.count * 2, options: .storageModeShared)!
        let gateB = device.makeBuffer(bytes: gate.biases, length: gate.biases.count * 2, options: .storageModeShared)!
        let upW = device.makeBuffer(bytes: up.weights, length: up.weights.count, options: .storageModeShared)!
        let upS = device.makeBuffer(bytes: up.scales, length: up.scales.count * 2, options: .storageModeShared)!
        let upB = device.makeBuffer(bytes: up.biases, length: up.biases.count * 2, options: .storageModeShared)!
        let inputBuf = device.makeBuffer(bytes: inputData, length: inputData.count * 2, options: .storageModeShared)!
        let batchedOut = device.makeBuffer(length: batchSize * rows * 2, options: .storageModeShared)!
        memset(batchedOut.contents(), 0, batchSize * rows * 2)

        let reference = try runIteratedFusedAffineGateUpSwiglu(
            pipeline: genericPipeline!,
            gateWeights: gateW,
            gateScales: gateS,
            gateBiases: gateB,
            upWeights: upW,
            upScales: upS,
            upBiases: upB,
            inputData: inputData,
            rows: rows,
            cols: cols,
            batchSize: batchSize
        )

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fullPipeline!)
        enc.setBuffer(gateW, offset: 0, index: 0)
        enc.setBuffer(gateS, offset: 0, index: 1)
        enc.setBuffer(gateB, offset: 0, index: 2)
        enc.setBuffer(upW, offset: 0, index: 3)
        enc.setBuffer(upS, offset: 0, index: 4)
        enc.setBuffer(upB, offset: 0, index: 5)
        enc.setBuffer(inputBuf, offset: 0, index: 6)
        enc.setBuffer(batchedOut, offset: 0, index: 7)
        var actualBatch = UInt32(batchSize)
        enc.setBytes(&actualBatch, length: 4, index: 8)
        enc.dispatchThreadgroups(
            MTLSize(
                width: rows / 32,
                height: (batchSize + batchTile - 1) / batchTile,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = batchedOut.contents().bindMemory(
            to: Float16.self, capacity: batchSize * rows
        )
        var maxDiff: Float = 0
        for i in 0..<(batchSize * rows) {
            let diff = abs(Float(outPtr[i]) - reference[i])
            if diff > maxDiff { maxDiff = diff }
        }

        fputs(
            "  \(functionName) vs iterated:"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.3, "Llama full batched fused affine FFN diverges from iterated generic")
    }

    func testFusedAffineGateUpSwigluQwenQMMSpecialization_HandlesShortBatchTail() throws {
        let genericPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "fused_affine_gate_up_swiglu",
            cols: 2048,
            groupSize: 64
        )
        try XCTSkipIf(genericPipeline == nil, "Could not compile generic fused affine FFN shader")

        let batchedPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "fused_affine_gate_up_swiglu_c2048_r6144_g64_batched_full"
        )
        try XCTSkipIf(
            batchedPipeline == nil,
            "Could not compile qmm fused affine FFN shader"
        )

        let rows = 6144
        let cols = 2048
        let batchSize = 1
        let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64)
        var inputData = [Float16](repeating: 0, count: batchSize * cols)
        for b in 0..<batchSize {
            for c in 0..<cols {
                inputData[b * cols + c] = Float16(sin(Float((b + 7) * (c + 3)) * 0.011))
            }
        }

        let gateW = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
        let gateS = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
        let gateB = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
        let upW = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
        let upS = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
        let upB = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
        let inputBuf = device.makeBuffer(bytes: inputData, length: inputData.count * 2, options: .storageModeShared)!
        let batchedOut = device.makeBuffer(length: batchSize * rows * 2, options: .storageModeShared)!
        memset(batchedOut.contents(), 0, batchSize * rows * 2)

        let reference = try runIteratedFusedAffineGateUpSwiglu(
            pipeline: genericPipeline!,
            gateWeights: gateW,
            gateScales: gateS,
            gateBiases: gateB,
            upWeights: upW,
            upScales: upS,
            upBiases: upB,
            inputData: inputData,
            rows: rows,
            cols: cols,
            batchSize: batchSize
        )

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchedPipeline!)
        enc.setBuffer(gateW, offset: 0, index: 0)
        enc.setBuffer(gateS, offset: 0, index: 1)
        enc.setBuffer(gateB, offset: 0, index: 2)
        enc.setBuffer(upW, offset: 0, index: 3)
        enc.setBuffer(upS, offset: 0, index: 4)
        enc.setBuffer(upB, offset: 0, index: 5)
        enc.setBuffer(inputBuf, offset: 0, index: 6)
        enc.setBuffer(batchedOut, offset: 0, index: 7)
        var actualBatch = UInt32(batchSize)
        enc.setBytes(&actualBatch, length: 4, index: 8)
        enc.dispatchThreadgroups(
            MTLSize(
                width: rows / 32,
                height: 1,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = batchedOut.contents().bindMemory(
            to: Float16.self, capacity: batchSize * rows
        )
        var maxDiff: Float = 0
        for i in 0..<(batchSize * rows) {
            let diff = abs(Float(outPtr[i]) - reference[i])
            if diff > maxDiff { maxDiff = diff }
        }

        fputs(
            "  fused_affine_gate_up_swiglu_c2048_r6144_g64_batched_full tail:"
                + " max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.04, "QMM fused affine FFN tail diverges from iterated generic")
    }

    func testFusedAffineGateUpSwigluQwenQMMSpecialization_LargeBatchBoundaries() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["SMELT_RUN_EXPERIMENTAL_QMM_FFN_TESTS"] == "1",
            "Experimental fused qmm FFN coverage is opt-in"
        )

        let referencePipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "fused_affine_gate_up_swiglu_c2048_r6144_g64_batched"
        )
        try XCTSkipIf(referencePipeline == nil, "Could not compile scalar batched fused affine FFN shader")

        let batchedPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "fused_affine_gate_up_swiglu_c2048_r6144_g64_batched_full"
        )
        try XCTSkipIf(
            batchedPipeline == nil,
            "Could not compile qmm fused affine FFN shader"
        )

        let rows = 6144
        let cols = 2048
        let rowTile = 32
        let batchTile = 16
        let tgWidth = 128
        for batchSize in [64, 65] {
            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64)
            var upWeightsData = data.weights
            var upScalesData = data.scales
            var upBiasesData = data.biases
            for i in 0..<upWeightsData.count {
                let byte = upWeightsData[i]
                upWeightsData[i] = ((byte & 0x0F) << 4) | ((byte & 0xF0) >> 4)
            }
            for i in 0..<upScalesData.count {
                upScalesData[i] = Float16(Float(upScalesData[i]) * (i % 2 == 0 ? 0.9375 : 1.0625))
                upBiasesData[i] = Float16(Float(upBiasesData[i]) + (i % 3 == 0 ? 0.03125 : -0.015625))
            }
            var inputData = [Float16](repeating: 0, count: batchSize * cols)
            for b in 0..<batchSize {
                for c in 0..<cols {
                    inputData[b * cols + c] = Float16(sin(Float((b + 13) * (c + 2)) * 0.006))
                }
            }

            let gateW = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let gateS = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let gateB = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let upW = device.makeBuffer(bytes: upWeightsData, length: upWeightsData.count, options: .storageModeShared)!
            let upS = device.makeBuffer(bytes: upScalesData, length: upScalesData.count * 2, options: .storageModeShared)!
            let upB = device.makeBuffer(bytes: upBiasesData, length: upBiasesData.count * 2, options: .storageModeShared)!
            let inputBuf = device.makeBuffer(bytes: inputData, length: inputData.count * 2, options: .storageModeShared)!
            let batchedOut = device.makeBuffer(length: batchSize * rows * 2, options: .storageModeShared)!
            memset(batchedOut.contents(), 0, batchSize * rows * 2)

            let refOut = device.makeBuffer(length: batchSize * rows * 2, options: .storageModeShared)!
            memset(refOut.contents(), 0, batchSize * rows * 2)

            let refCmdBuf = queue.makeCommandBuffer()!
            let refEnc = refCmdBuf.makeComputeCommandEncoder()!
            refEnc.setComputePipelineState(referencePipeline!)
            refEnc.setBuffer(gateW, offset: 0, index: 0)
            refEnc.setBuffer(gateS, offset: 0, index: 1)
            refEnc.setBuffer(gateB, offset: 0, index: 2)
            refEnc.setBuffer(upW, offset: 0, index: 3)
            refEnc.setBuffer(upS, offset: 0, index: 4)
            refEnc.setBuffer(upB, offset: 0, index: 5)
            refEnc.setBuffer(inputBuf, offset: 0, index: 6)
            refEnc.setBuffer(refOut, offset: 0, index: 7)
            var refActualBatch = UInt32(batchSize)
            refEnc.setBytes(&refActualBatch, length: 4, index: 8)
            refEnc.dispatchThreadgroups(
                MTLSize(
                    width: rows / 8,
                    height: (batchSize + 1) / 2,
                    depth: 1
                ),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            refEnc.endEncoding()
            refCmdBuf.commit()
            refCmdBuf.waitUntilCompleted()
            XCTAssertNil(refCmdBuf.error)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(batchedPipeline!)
            enc.setBuffer(gateW, offset: 0, index: 0)
            enc.setBuffer(gateS, offset: 0, index: 1)
            enc.setBuffer(gateB, offset: 0, index: 2)
            enc.setBuffer(upW, offset: 0, index: 3)
            enc.setBuffer(upS, offset: 0, index: 4)
            enc.setBuffer(upB, offset: 0, index: 5)
            enc.setBuffer(inputBuf, offset: 0, index: 6)
            enc.setBuffer(batchedOut, offset: 0, index: 7)
            var actualBatch = UInt32(batchSize)
            enc.setBytes(&actualBatch, length: 4, index: 8)
            enc.dispatchThreadgroups(
                MTLSize(
                    width: rows / rowTile,
                    height: (batchSize + batchTile - 1) / batchTile,
                    depth: 1
                ),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let outPtr = batchedOut.contents().bindMemory(
                to: Float16.self, capacity: batchSize * rows
            )
            let refPtr = refOut.contents().bindMemory(
                to: Float16.self, capacity: batchSize * rows
            )
            var maxDiff: Float = 0
            var worstBatch = 0
            var worstRow = 0
            var worstRef: Float = 0
            var worstOut: Float = 0
            var maxRefAbs: Float = 0
            for i in 0..<(batchSize * rows) {
                let outVal = Float(outPtr[i])
                let refVal = Float(refPtr[i])
                let diff = abs(outVal - refVal)
                maxRefAbs = max(maxRefAbs, abs(refVal))
                if diff > maxDiff {
                    maxDiff = diff
                    worstBatch = i / rows
                    worstRow = i % rows
                    worstRef = refVal
                    worstOut = outVal
                }
            }
            let relDiff = maxDiff / max(maxRefAbs, 1)

            fputs(
                "  fused_affine_gate_up_swiglu_c2048_r6144_g64_batched_full batch=\(batchSize):"
                    + " max diff = \(String(format: "%.6f", maxDiff))"
                    + " rel = \(String(format: "%.6f", relDiff))"
                    + " ref = \(String(format: "%.6f", worstRef))"
                    + " out = \(String(format: "%.6f", worstOut))"
                    + " worstBatch=\(worstBatch)"
                    + " worstRow=\(worstRow)\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.04, "QMM fused affine FFN diverges from scalar batched path at batch \(batchSize)")
        }
    }

    func testFusedAffineGateUpSwigluQwen0808BatchedFull_LargeBatchBoundaries() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["SMELT_RUN_QWEN_SMOKE"] == "1",
            "0.8B large-batch prefill FFN boundary coverage is opt-in"
        )

        let genericPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "fused_affine_gate_up_swiglu",
            cols: 1024,
            groupSize: 64
        )
        try XCTSkipIf(genericPipeline == nil, "Could not compile generic fused affine FFN shader")

        let fullPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full"
        )
        try XCTSkipIf(fullPipeline == nil, "Could not compile 0.8B full batched fused affine FFN shader")

        let rows = 3584
        let cols = 1024
        for batchSize in [64, 65] {
            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64)
            var inputData = [Float16](repeating: 0, count: batchSize * cols)
            for b in 0..<batchSize {
                for c in 0..<cols {
                    inputData[b * cols + c] = Float16(sin(Float((b + 17) * (c + 3)) * 0.008))
                }
            }

            let gateW = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let gateS = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let gateB = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let upW = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let upS = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let upB = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let inputBuf = device.makeBuffer(bytes: inputData, length: inputData.count * 2, options: .storageModeShared)!
            let batchedOut = device.makeBuffer(length: batchSize * rows * 2, options: .storageModeShared)!
            memset(batchedOut.contents(), 0, batchSize * rows * 2)

            let reference = try runIteratedFusedAffineGateUpSwiglu(
                pipeline: genericPipeline!,
                gateWeights: gateW,
                gateScales: gateS,
                gateBiases: gateB,
                upWeights: upW,
                upScales: upS,
                upBiases: upB,
                inputData: inputData,
                rows: rows,
                cols: cols,
                batchSize: batchSize
            )

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(fullPipeline!)
            enc.setBuffer(gateW, offset: 0, index: 0)
            enc.setBuffer(gateS, offset: 0, index: 1)
            enc.setBuffer(gateB, offset: 0, index: 2)
            enc.setBuffer(upW, offset: 0, index: 3)
            enc.setBuffer(upS, offset: 0, index: 4)
            enc.setBuffer(upB, offset: 0, index: 5)
            enc.setBuffer(inputBuf, offset: 0, index: 6)
            enc.setBuffer(batchedOut, offset: 0, index: 7)
            var actualBatch = UInt32(batchSize)
            enc.setBytes(&actualBatch, length: 4, index: 8)
            enc.dispatchThreadgroups(
                MTLSize(
                    width: (rows + 31) / 32,
                    height: (batchSize + 15) / 16,
                    depth: 1
                ),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let outPtr = batchedOut.contents().bindMemory(
                to: Float16.self, capacity: batchSize * rows
            )
            var maxDiff: Float = 0
            for i in 0..<(batchSize * rows) {
                maxDiff = max(maxDiff, abs(Float(outPtr[i]) - reference[i]))
            }

            fputs(
                "  fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full batch=\(batchSize):"
                    + " max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.04, "0.8B full batched fused affine FFN diverges at batch boundary \(batchSize)")
        }
    }

    func testSwiGLUFused_LargeBatchBoundaries() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["SMELT_RUN_SLOW_TESTS"] == "1",
            "Large-batch prefill boundary coverage is opt-in"
        )

        let pipeline = makePipeline(
            shaderFile: "activations_precise.metal",
            functionName: "swiglu_fused"
        )
        try XCTSkipIf(pipeline == nil, "Could not compile swiglu_fused")

        let dim = 6144
        for batchSize in [64, 65, 255, 256] {
            let count = batchSize * dim
            var gateData = [Float16](repeating: 0, count: count)
            var upData = [Float16](repeating: 0, count: count)
            var refData = [Float16](repeating: 0, count: count)

            for i in 0..<count {
                let g = sin(Float(i + 17) * 0.0017) * 6.0
                let u = cos(Float(i + 31) * 0.0013) * 5.0
                gateData[i] = Float16(g)
                upData[i] = Float16(u)
                let silu = g / (1.0 + exp(-g))
                refData[i] = Float16(silu * u)
            }

            let gateBuf = device.makeBuffer(bytes: gateData, length: count * 2, options: .storageModeShared)!
            let upBuf = device.makeBuffer(bytes: upData, length: count * 2, options: .storageModeShared)!
            let outBuf = device.makeBuffer(length: count * 2, options: .storageModeShared)!
            memset(outBuf.contents(), 0, count * 2)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline!)
            enc.setBuffer(gateBuf, offset: 0, index: 0)
            enc.setBuffer(upBuf, offset: 0, index: 1)
            enc.setBuffer(outBuf, offset: 0, index: 2)
            var countU32 = UInt32(count)
            enc.setBytes(&countU32, length: 4, index: 3)
            enc.dispatchThreads(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let outPtr = outBuf.contents().bindMemory(to: Float16.self, capacity: count)
            var maxDiff: Float = 0
            for i in 0..<count {
                maxDiff = max(maxDiff, abs(Float(outPtr[i]) - Float(refData[i])))
            }

            fputs(
                "  swiglu_fused batch=\(batchSize): max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.04, "swiglu_fused diverges at batch \(batchSize)")
        }
    }

    func testGeGLUFusedStridedBatched_MatchesPerBatchGeGLUBits() throws {
        let referencePipeline = makePipeline(
            shaderFile: "activations.metal",
            functionName: "geglu_fused"
        )
        let stridedPipeline = makePipeline(
            shaderFile: "activations.metal",
            functionName: "geglu_fused_strided_batched"
        )
        try XCTSkipIf(referencePipeline == nil, "Could not compile geglu_fused")
        try XCTSkipIf(stridedPipeline == nil, "Could not compile geglu_fused_strided_batched")

        let batchSize = 5
        let count = 256
        let upStride = 42 * count
        let layerOffset = 17 * count
        var gateData = [Float16](repeating: 0, count: batchSize * count)
        var upData = [Float16](repeating: 0, count: batchSize * upStride)

        for b in 0..<batchSize {
            for i in 0..<count {
                gateData[b * count + i] = Float16(sin(Float((b + 1) * (i + 3)) * 0.017) * 4.0)
                upData[b * upStride + layerOffset + i] =
                    Float16(cos(Float((b + 5) * (i + 7)) * 0.013) * 3.0)
            }
        }

        let gateBuf = device.makeBuffer(
            bytes: gateData, length: gateData.count * 2, options: .storageModeShared
        )!
        let upBuf = device.makeBuffer(
            bytes: upData, length: upData.count * 2, options: .storageModeShared
        )!
        let referenceOut = device.makeBuffer(
            length: batchSize * count * 2, options: .storageModeShared
        )!
        let stridedOut = device.makeBuffer(
            length: batchSize * count * 2, options: .storageModeShared
        )!
        memset(referenceOut.contents(), 0, batchSize * count * 2)
        memset(stridedOut.contents(), 0, batchSize * count * 2)

        for b in 0..<batchSize {
            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(referencePipeline!)
            enc.setBuffer(gateBuf, offset: b * count * 2, index: 0)
            enc.setBuffer(upBuf, offset: (b * upStride + layerOffset) * 2, index: 1)
            enc.setBuffer(referenceOut, offset: b * count * 2, index: 2)
            var countU32 = UInt32(count)
            enc.setBytes(&countU32, length: 4, index: 3)
            enc.dispatchThreads(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)
        }

        do {
            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(stridedPipeline!)
            enc.setBuffer(gateBuf, offset: 0, index: 0)
            enc.setBuffer(upBuf, offset: layerOffset * 2, index: 1)
            enc.setBuffer(stridedOut, offset: 0, index: 2)
            var countU32 = UInt32(count)
            var upStrideU32 = UInt32(upStride)
            enc.setBytes(&countU32, length: 4, index: 3)
            enc.setBytes(&upStrideU32, length: 4, index: 4)
            enc.dispatchThreads(
                MTLSize(width: count, height: batchSize, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)
        }

        let referenceBits = referenceOut.contents().bindMemory(
            to: UInt16.self, capacity: batchSize * count
        )
        let stridedBits = stridedOut.contents().bindMemory(
            to: UInt16.self, capacity: batchSize * count
        )
        var firstMismatch: Int?
        for i in 0..<(batchSize * count) {
            if referenceBits[i] != stridedBits[i] {
                firstMismatch = i
                break
            }
        }

        if let firstMismatch {
            XCTFail(
                "Strided batched GeGLU differs at output \(firstMismatch): " +
                    "reference=0x\(String(referenceBits[firstMismatch], radix: 16)) " +
                    "strided=0x\(String(stridedBits[firstMismatch], radix: 16))"
            )
        }
    }

    func testQwenQMMBatchedSpecializations_HandleShortBatchTail() throws {
        let rows = 2048
        let cols = 2048
        let batchSize = 3
        let batchTile = 16

        let genericAffine = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "affine_matvec",
            cols: UInt32(cols),
            groupSize: 64
        )
        let specializedAffine = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "affine_matvec_c2048_r2048_g64_batched_full"
        )
        try XCTSkipIf(genericAffine == nil || specializedAffine == nil, "Could not compile qmm affine specialization")

        let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64)
        var inputData = [Float16](repeating: 0, count: batchSize * cols)
        for b in 0..<batchSize {
            for c in 0..<cols {
                inputData[b * cols + c] = Float16(sin(Float((b + 2) * (c + 5)) * 0.007))
            }
        }

        let weightsBuf = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
        let scalesBuf = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
        let biasesBuf = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
        let inputBuf = device.makeBuffer(bytes: inputData, length: inputData.count * 2, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: batchSize * rows * 2, options: .storageModeShared)!
        memset(outBuf.contents(), 0, batchSize * rows * 2)

        let reference = try runIteratedAffineMatvec(
            pipeline: genericAffine!,
            weights: weightsBuf,
            scales: scalesBuf,
            biases: biasesBuf,
            inputData: inputData,
            rows: rows,
            cols: cols,
            batchSize: batchSize
        )

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(specializedAffine!)
        enc.setBuffer(weightsBuf, offset: 0, index: 0)
        enc.setBuffer(scalesBuf, offset: 0, index: 1)
        enc.setBuffer(biasesBuf, offset: 0, index: 2)
        enc.setBuffer(inputBuf, offset: 0, index: 3)
        enc.setBuffer(outBuf, offset: 0, index: 4)
        var actualBatch = UInt32(batchSize)
        enc.setBytes(&actualBatch, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: (rows + 31) / 32, height: (batchSize + batchTile - 1) / batchTile, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        XCTAssertNil(cmdBuf.error)

        let outPtr = outBuf.contents().bindMemory(to: Float16.self, capacity: batchSize * rows)
        var maxDiff: Float = 0
        for i in 0..<(batchSize * rows) {
            maxDiff = max(maxDiff, abs(Float(outPtr[i]) - reference[i]))
        }
        XCTAssertLessThan(maxDiff, 0.07, "QMM affine tail handling diverges from iterated path")
    }

    func testFusedDualAffineMatvecQwenBatchedSpecialization_MatchesIterated() throws {
        let cases: [(rows: Int, cols: Int, functionName: String)] = [
            (16, 1024, "fused_dual_affine_matvec_c1024_r16_g64_batched"),
            (512, 1024, "fused_dual_affine_matvec_c1024_r512_g64_batched"),
            (16, 2048, "fused_dual_affine_matvec_c2048_r16_g64_batched"),
            (512, 2048, "fused_dual_affine_matvec_c2048_r512_g64_batched"),
            (32, 2560, "fused_dual_affine_matvec_c2560_r32_g64_batched"),
            (1024, 2560, "fused_dual_affine_matvec_c2560_r1024_g64_batched"),
        ]

        for (rows, cols, functionName) in cases {
            let genericPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: "fused_dual_affine_matvec",
                cols: UInt32(cols),
                groupSize: 64
            )
            try XCTSkipIf(genericPipeline == nil, "Could not compile generic fused dual affine shader")

            let batchedPipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: functionName
            )
            try XCTSkipIf(
                batchedPipeline == nil,
                "Could not compile \(functionName)"
            )

            let batchSize = 4
            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64)
            var inputData = [Float16](repeating: 0, count: batchSize * cols)
            for b in 0..<batchSize {
                for c in 0..<cols {
                    inputData[b * cols + c] = Float16(sin(Float((b + 7) * (c + 11)) * 0.006))
                }
            }

            let w1 = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let s1 = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let b1 = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let w2 = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let s2 = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let b2 = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let inputBuf = device.makeBuffer(bytes: inputData, length: inputData.count * 2, options: .storageModeShared)!
            let out1 = device.makeBuffer(length: batchSize * rows * 2, options: .storageModeShared)!
            let out2 = device.makeBuffer(length: batchSize * rows * 2, options: .storageModeShared)!
            memset(out1.contents(), 0, batchSize * rows * 2)
            memset(out2.contents(), 0, batchSize * rows * 2)

            let reference = try runIteratedFusedDualAffineMatvec(
                pipeline: genericPipeline!,
                w1Weights: w1,
                w1Scales: s1,
                w1Biases: b1,
                w2Weights: w2,
                w2Scales: s2,
                w2Biases: b2,
                inputData: inputData,
                rows: rows,
                cols: cols,
                batchSize: batchSize
            )

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(batchedPipeline!)
            enc.setBuffer(w1, offset: 0, index: 0)
            enc.setBuffer(s1, offset: 0, index: 1)
            enc.setBuffer(b1, offset: 0, index: 2)
            enc.setBuffer(w2, offset: 0, index: 3)
            enc.setBuffer(s2, offset: 0, index: 4)
            enc.setBuffer(b2, offset: 0, index: 5)
            enc.setBuffer(inputBuf, offset: 0, index: 6)
            enc.setBuffer(out1, offset: 0, index: 7)
            enc.setBuffer(out2, offset: 0, index: 8)
            enc.dispatchThreadgroups(
                MTLSize(width: rows / 8, height: batchSize, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let out1Ptr = out1.contents().bindMemory(to: Float16.self, capacity: batchSize * rows)
            let out2Ptr = out2.contents().bindMemory(to: Float16.self, capacity: batchSize * rows)
            var maxDiff: Float = 0
            for i in 0..<(batchSize * rows) {
                let diff1 = abs(Float(out1Ptr[i]) - reference.out1[i])
                let diff2 = abs(Float(out2Ptr[i]) - reference.out2[i])
                maxDiff = max(maxDiff, diff1, diff2)
            }

            fputs(
                "  \(functionName) vs iterated: max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.002, "\(functionName) diverges from iterated generic")
        }
    }

    func testFusedDualAffineMatvecQwenDecodeSpecialization_MatchesGeneric() throws {
        let genericPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "fused_dual_affine_matvec",
            cols: 2048,
            groupSize: 64
        )
        try XCTSkipIf(genericPipeline == nil, "Could not compile generic fused dual affine shader")

        let specializedPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "fused_dual_affine_matvec_c2048_r16_g64"
        )
        try XCTSkipIf(
            specializedPipeline == nil,
            "Could not compile fused_dual_affine_matvec_c2048_r16_g64"
        )

        let rows = 16
        let cols = 2048
        let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64)
        let w1 = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
        let s1 = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
        let b1 = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
        let w2 = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
        let s2 = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
        let b2 = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
        let input = (0..<cols).map { Float16(cos(Float($0 + 3) * 0.011)) }
        let inputBuf = device.makeBuffer(bytes: input, length: input.count * 2, options: .storageModeShared)!
        let out1Generic = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
        let out2Generic = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
        let out1Specialized = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
        let out2Specialized = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
        memset(out1Generic.contents(), 0, rows * 2)
        memset(out2Generic.contents(), 0, rows * 2)
        memset(out1Specialized.contents(), 0, rows * 2)
        memset(out2Specialized.contents(), 0, rows * 2)

        let cmdBuf1 = queue.makeCommandBuffer()!
        let enc1 = cmdBuf1.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(genericPipeline!)
        enc1.setBuffer(w1, offset: 0, index: 0)
        enc1.setBuffer(s1, offset: 0, index: 1)
        enc1.setBuffer(b1, offset: 0, index: 2)
        enc1.setBuffer(w2, offset: 0, index: 3)
        enc1.setBuffer(s2, offset: 0, index: 4)
        enc1.setBuffer(b2, offset: 0, index: 5)
        enc1.setBuffer(inputBuf, offset: 0, index: 6)
        enc1.setBuffer(out1Generic, offset: 0, index: 7)
        enc1.setBuffer(out2Generic, offset: 0, index: 8)
        var rowsVal = UInt32(rows)
        enc1.setBytes(&rowsVal, length: 4, index: 9)
        enc1.dispatchThreadgroups(
            MTLSize(width: rows / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        enc1.endEncoding()
        cmdBuf1.commit()
        cmdBuf1.waitUntilCompleted()
        XCTAssertNil(cmdBuf1.error)

        let cmdBuf2 = queue.makeCommandBuffer()!
        let enc2 = cmdBuf2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(specializedPipeline!)
        enc2.setBuffer(w1, offset: 0, index: 0)
        enc2.setBuffer(s1, offset: 0, index: 1)
        enc2.setBuffer(b1, offset: 0, index: 2)
        enc2.setBuffer(w2, offset: 0, index: 3)
        enc2.setBuffer(s2, offset: 0, index: 4)
        enc2.setBuffer(b2, offset: 0, index: 5)
        enc2.setBuffer(inputBuf, offset: 0, index: 6)
        enc2.setBuffer(out1Specialized, offset: 0, index: 7)
        enc2.setBuffer(out2Specialized, offset: 0, index: 8)
        enc2.dispatchThreadgroups(
            MTLSize(width: rows / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        enc2.endEncoding()
        cmdBuf2.commit()
        cmdBuf2.waitUntilCompleted()
        XCTAssertNil(cmdBuf2.error)

        let out1GenericPtr = out1Generic.contents().bindMemory(to: Float16.self, capacity: rows)
        let out2GenericPtr = out2Generic.contents().bindMemory(to: Float16.self, capacity: rows)
        let out1SpecializedPtr = out1Specialized.contents().bindMemory(to: Float16.self, capacity: rows)
        let out2SpecializedPtr = out2Specialized.contents().bindMemory(to: Float16.self, capacity: rows)
        var maxDiff: Float = 0
        for i in 0..<rows {
            maxDiff = max(maxDiff, abs(Float(out1GenericPtr[i]) - Float(out1SpecializedPtr[i])))
            maxDiff = max(maxDiff, abs(Float(out2GenericPtr[i]) - Float(out2SpecializedPtr[i])))
        }

        fputs(
            "  fused_dual_affine_matvec_c2048_r16_g64 vs generic: max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(maxDiff, 0.002, "fused_dual_affine_matvec_c2048_r16_g64 diverges from generic")
    }

    func testFusedDualAffineMatvecQwenBatchedSpecialization_LargeBatchBoundaries() throws {
        let functionName = "fused_dual_affine_matvec_c2048_r16_g64_batched"
        let genericPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "fused_dual_affine_matvec",
            cols: 2048,
            groupSize: 64
        )
        try XCTSkipIf(genericPipeline == nil, "Could not compile generic fused dual affine shader")

        let batchedPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: functionName
        )
        try XCTSkipIf(batchedPipeline == nil, "Could not compile \(functionName)")

        let rows = 16
        let cols = 2048
        for batchSize in [64, 65] {
            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64)
            var inputData = [Float16](repeating: 0, count: batchSize * cols)
            for b in 0..<batchSize {
                for c in 0..<cols {
                    inputData[b * cols + c] = Float16(sin(Float((b + 5) * (c + 3)) * 0.0047))
                }
            }

            let w1 = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let s1 = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let b1 = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let w2 = device.makeBuffer(bytes: data.weights, length: data.weights.count, options: .storageModeShared)!
            let s2 = device.makeBuffer(bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared)!
            let b2 = device.makeBuffer(bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared)!
            let inputBuf = device.makeBuffer(bytes: inputData, length: inputData.count * 2, options: .storageModeShared)!
            let out1 = device.makeBuffer(length: batchSize * rows * 2, options: .storageModeShared)!
            let out2 = device.makeBuffer(length: batchSize * rows * 2, options: .storageModeShared)!
            memset(out1.contents(), 0, batchSize * rows * 2)
            memset(out2.contents(), 0, batchSize * rows * 2)

            let reference = try runIteratedFusedDualAffineMatvec(
                pipeline: genericPipeline!,
                w1Weights: w1,
                w1Scales: s1,
                w1Biases: b1,
                w2Weights: w2,
                w2Scales: s2,
                w2Biases: b2,
                inputData: inputData,
                rows: rows,
                cols: cols,
                batchSize: batchSize
            )

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(batchedPipeline!)
            enc.setBuffer(w1, offset: 0, index: 0)
            enc.setBuffer(s1, offset: 0, index: 1)
            enc.setBuffer(b1, offset: 0, index: 2)
            enc.setBuffer(w2, offset: 0, index: 3)
            enc.setBuffer(s2, offset: 0, index: 4)
            enc.setBuffer(b2, offset: 0, index: 5)
            enc.setBuffer(inputBuf, offset: 0, index: 6)
            enc.setBuffer(out1, offset: 0, index: 7)
            enc.setBuffer(out2, offset: 0, index: 8)
            enc.dispatchThreadgroups(
                MTLSize(width: rows / 8, height: batchSize, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            XCTAssertNil(cmdBuf.error)

            let out1Ptr = out1.contents().bindMemory(to: Float16.self, capacity: batchSize * rows)
            let out2Ptr = out2.contents().bindMemory(to: Float16.self, capacity: batchSize * rows)
            var maxDiff: Float = 0
            for i in 0..<(batchSize * rows) {
                maxDiff = max(maxDiff, abs(Float(out1Ptr[i]) - reference.out1[i]))
                maxDiff = max(maxDiff, abs(Float(out2Ptr[i]) - reference.out2[i]))
            }

            fputs(
                "  \(functionName) batch=\(batchSize):"
                    + " max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(maxDiff, 0.002, "\(functionName) diverges at batch boundary \(batchSize)")
        }
    }

    func testGenericAffineMatvecQwen0808Shapes_MatchCPUReference() throws {
        let cases: [(rows: Int, cols: Int)] = [
            (4096, 1024),
            (2048, 1024),
            (512, 1024),
            (1024, 2048),
            (1024, 3584),
        ]
        let batchSize = 2

        for (rows, cols) in cases {
            let pipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: "affine_matvec",
                cols: UInt32(cols),
                groupSize: 64
            )
            try XCTSkipIf(
                pipeline == nil,
                "Could not compile generic affine matvec shader for cols=\(cols)"
            )

            let data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64)
            var inputData = [Float16](repeating: 0, count: batchSize * cols)
            for b in 0..<batchSize {
                for c in 0..<cols {
                    inputData[b * cols + c] = Float16(
                        sin(Float((b + 13) * (c + 17)) * 0.0043)
                    )
                }
            }

            let weightsBuf = device.makeBuffer(
                bytes: data.weights, length: data.weights.count, options: .storageModeShared
            )!
            let scalesBuf = device.makeBuffer(
                bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared
            )!
            let biasesBuf = device.makeBuffer(
                bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared
            )!

            let gpu = try runIteratedAffineMatvec(
                pipeline: pipeline!,
                weights: weightsBuf,
                scales: scalesBuf,
                biases: biasesBuf,
                inputData: inputData,
                rows: rows,
                cols: cols,
                batchSize: batchSize
            )
            let cpu = referenceAffineMatvecCPU(
                weights: data.weights,
                scales: data.scales,
                biases: data.biases,
                inputData: inputData,
                rows: rows,
                cols: cols,
                groupSize: 64,
                batchSize: batchSize
            )

            var maxDiff: Float = 0
            for i in 0..<(batchSize * rows) {
                maxDiff = max(maxDiff, abs(gpu[i] - cpu[i]))
            }

            fputs(
                "  affine_matvec rows=\(rows) cols=\(cols) vs CPU: "
                    + "max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(
                maxDiff,
                0.02,
                "generic affine_matvec rows=\(rows) cols=\(cols) diverges from CPU reference"
            )
        }
    }

    func testGenericFusedAffineGateUpSwigluQwen0808Shape_MatchesCPUReference() throws {
        let rows = 3584
        let cols = 1024
        let batchSize = 2

        let pipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "fused_affine_gate_up_swiglu",
            cols: UInt32(cols),
            groupSize: 64
        )
        try XCTSkipIf(pipeline == nil, "Could not compile generic fused affine FFN shader")

        let gate = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 41)
        let up = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 42)
        var inputData = [Float16](repeating: 0, count: batchSize * cols)
        for b in 0..<batchSize {
            for c in 0..<cols {
                inputData[b * cols + c] = Float16(
                    cos(Float((b + 19) * (c + 7)) * 0.0061)
                )
            }
        }

        let gateW = device.makeBuffer(
            bytes: gate.weights, length: gate.weights.count, options: .storageModeShared
        )!
        let gateS = device.makeBuffer(
            bytes: gate.scales, length: gate.scales.count * 2, options: .storageModeShared
        )!
        let gateB = device.makeBuffer(
            bytes: gate.biases, length: gate.biases.count * 2, options: .storageModeShared
        )!
        let upW = device.makeBuffer(
            bytes: up.weights, length: up.weights.count, options: .storageModeShared
        )!
        let upS = device.makeBuffer(
            bytes: up.scales, length: up.scales.count * 2, options: .storageModeShared
        )!
        let upB = device.makeBuffer(
            bytes: up.biases, length: up.biases.count * 2, options: .storageModeShared
        )!

        let gpu = try runIteratedFusedAffineGateUpSwiglu(
            pipeline: pipeline!,
            gateWeights: gateW,
            gateScales: gateS,
            gateBiases: gateB,
            upWeights: upW,
            upScales: upS,
            upBiases: upB,
            inputData: inputData,
            rows: rows,
            cols: cols,
            batchSize: batchSize
        )
        let cpu = referenceFusedAffineGateUpSwigluCPU(
            gateWeights: gate.weights,
            gateScales: gate.scales,
            gateBiases: gate.biases,
            upWeights: up.weights,
            upScales: up.scales,
            upBiases: up.biases,
            inputData: inputData,
            rows: rows,
            cols: cols,
            groupSize: 64,
            batchSize: batchSize
        )

        var maxDiff: Float = 0
        for i in 0..<(batchSize * rows) {
            maxDiff = max(maxDiff, abs(gpu[i] - cpu[i]))
        }

        fputs(
            "  fused_affine_gate_up_swiglu rows=\(rows) cols=\(cols) vs CPU: "
                + "max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(
            maxDiff,
            0.02,
            "generic fused_affine_gate_up_swiglu diverges from CPU reference"
        )
    }

    func testGenericFusedDualAffineMatvecQwen0808Shape_MatchesCPUReference() throws {
        let rows = 16
        let cols = 1024
        let batchSize = 3

        let pipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "fused_dual_affine_matvec",
            cols: UInt32(cols),
            groupSize: 64
        )
        try XCTSkipIf(pipeline == nil, "Could not compile generic fused dual affine shader")

        let w1Data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64)
        let w2Data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64)
        var inputData = [Float16](repeating: 0, count: batchSize * cols)
        for b in 0..<batchSize {
            for c in 0..<cols {
                inputData[b * cols + c] = Float16(
                    sin(Float((b + 23) * (c + 9)) * 0.0087)
                )
            }
        }

        let w1 = device.makeBuffer(
            bytes: w1Data.weights, length: w1Data.weights.count, options: .storageModeShared
        )!
        let s1 = device.makeBuffer(
            bytes: w1Data.scales, length: w1Data.scales.count * 2, options: .storageModeShared
        )!
        let b1 = device.makeBuffer(
            bytes: w1Data.biases, length: w1Data.biases.count * 2, options: .storageModeShared
        )!
        let w2 = device.makeBuffer(
            bytes: w2Data.weights, length: w2Data.weights.count, options: .storageModeShared
        )!
        let s2 = device.makeBuffer(
            bytes: w2Data.scales, length: w2Data.scales.count * 2, options: .storageModeShared
        )!
        let b2 = device.makeBuffer(
            bytes: w2Data.biases, length: w2Data.biases.count * 2, options: .storageModeShared
        )!

        let gpu = try runIteratedFusedDualAffineMatvec(
            pipeline: pipeline!,
            w1Weights: w1,
            w1Scales: s1,
            w1Biases: b1,
            w2Weights: w2,
            w2Scales: s2,
            w2Biases: b2,
            inputData: inputData,
            rows: rows,
            cols: cols,
            batchSize: batchSize
        )
        let cpu = referenceFusedDualAffineMatvecCPU(
            w1Weights: w1Data.weights,
            w1Scales: w1Data.scales,
            w1Biases: w1Data.biases,
            w2Weights: w2Data.weights,
            w2Scales: w2Data.scales,
            w2Biases: w2Data.biases,
            inputData: inputData,
            rows: rows,
            cols: cols,
            groupSize: 64,
            batchSize: batchSize
        )

        var maxDiff: Float = 0
        for i in 0..<(batchSize * rows) {
            maxDiff = max(maxDiff, abs(gpu.out1[i] - cpu.out1[i]))
            maxDiff = max(maxDiff, abs(gpu.out2[i] - cpu.out2[i]))
        }

        fputs(
            "  fused_dual_affine_matvec rows=\(rows) cols=\(cols) vs CPU: "
                + "max diff = \(String(format: "%.6f", maxDiff))\n",
            stderr
        )
        XCTAssertLessThan(
            maxDiff,
            0.02,
            "generic fused_dual_affine_matvec diverges from CPU reference"
        )
    }

    func testFusedDualAffineMatvecQwen0808Rows4_MatchesGenericBits() throws {
        let rows = 16
        let cols = 1024
        let generic = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "fused_dual_affine_matvec",
            cols: UInt32(cols),
            groupSize: 64
        )
        let rows4 = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "fused_dual_affine_matvec_c1024_r16_g64_rows4"
        )
        try XCTSkipIf(generic == nil || rows4 == nil, "Could not compile Qwen 0.8B dual-affine pipelines")

        let w1Data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 71)
        let w2Data = makeAffineTestData(rows: rows, cols: cols, groupSize: 64, seed: 72)
        let input = (0..<cols).map { Float16(sin(Float($0 + 11) * 0.0137)) }
        let w1 = device.makeBuffer(bytes: w1Data.weights, length: w1Data.weights.count, options: .storageModeShared)!
        let s1 = device.makeBuffer(bytes: w1Data.scales, length: w1Data.scales.count * 2, options: .storageModeShared)!
        let b1 = device.makeBuffer(bytes: w1Data.biases, length: w1Data.biases.count * 2, options: .storageModeShared)!
        let w2 = device.makeBuffer(bytes: w2Data.weights, length: w2Data.weights.count, options: .storageModeShared)!
        let s2 = device.makeBuffer(bytes: w2Data.scales, length: w2Data.scales.count * 2, options: .storageModeShared)!
        let b2 = device.makeBuffer(bytes: w2Data.biases, length: w2Data.biases.count * 2, options: .storageModeShared)!

        let expected = try runIteratedFusedDualAffineMatvec(
            pipeline: generic!, w1Weights: w1, w1Scales: s1, w1Biases: b1,
            w2Weights: w2, w2Scales: s2, w2Biases: b2,
            inputData: input, rows: rows, cols: cols, batchSize: 1
        )
        let actual = try runIteratedFusedDualAffineMatvec(
            pipeline: rows4!, w1Weights: w1, w1Scales: s1, w1Biases: b1,
            w2Weights: w2, w2Scales: s2, w2Biases: b2,
            inputData: input, rows: rows, cols: cols, batchSize: 1,
            rowTile: 4, setRowsBuffer: false
        )

        XCTAssertEqual(actual.out1, expected.out1, "rows4 output1 changed FP16 bits")
        XCTAssertEqual(actual.out2, expected.out2, "rows4 output2 changed FP16 bits")
    }

    func testGenericFusedDualAffineMatvecShapes_MatchCPUReference() throws {
        let cases: [(rows: Int, cols: Int)] = [
            (256, 1536),
            (512, 1536),
        ]
        let batchSize = 2

        for (rows, cols) in cases {
            let pipeline = makePipeline(
                shaderFile: "lut_matvec.metal",
                functionName: "fused_dual_affine_matvec",
                cols: UInt32(cols),
                groupSize: 128
            )
            try XCTSkipIf(pipeline == nil, "Could not compile generic fused dual affine shader")

            let w1Data = makeAffineTestData(rows: rows, cols: cols, groupSize: 128, seed: 61)
            let w2Data = makeAffineTestData(rows: rows, cols: cols, groupSize: 128, seed: 62)
            var inputData = [Float16](repeating: 0, count: batchSize * cols)
            for b in 0..<batchSize {
                for c in 0..<cols {
                    inputData[b * cols + c] = Float16(
                        sin(Float((b + 29) * (c + 13)) * 0.0069)
                    )
                }
            }

            let w1 = device.makeBuffer(
                bytes: w1Data.weights, length: w1Data.weights.count, options: .storageModeShared
            )!
            let s1 = device.makeBuffer(
                bytes: w1Data.scales, length: w1Data.scales.count * 2, options: .storageModeShared
            )!
            let b1 = device.makeBuffer(
                bytes: w1Data.biases, length: w1Data.biases.count * 2, options: .storageModeShared
            )!
            let w2 = device.makeBuffer(
                bytes: w2Data.weights, length: w2Data.weights.count, options: .storageModeShared
            )!
            let s2 = device.makeBuffer(
                bytes: w2Data.scales, length: w2Data.scales.count * 2, options: .storageModeShared
            )!
            let b2 = device.makeBuffer(
                bytes: w2Data.biases, length: w2Data.biases.count * 2, options: .storageModeShared
            )!

            let gpu = try runIteratedFusedDualAffineMatvec(
                pipeline: pipeline!,
                w1Weights: w1,
                w1Scales: s1,
                w1Biases: b1,
                w2Weights: w2,
                w2Scales: s2,
                w2Biases: b2,
                inputData: inputData,
                rows: rows,
                cols: cols,
                batchSize: batchSize
            )
            let cpu = referenceFusedDualAffineMatvecCPU(
                w1Weights: w1Data.weights,
                w1Scales: w1Data.scales,
                w1Biases: w1Data.biases,
                w2Weights: w2Data.weights,
                w2Scales: w2Data.scales,
                w2Biases: w2Data.biases,
                inputData: inputData,
                rows: rows,
                cols: cols,
                groupSize: 128,
                batchSize: batchSize
            )

            var maxDiff: Float = 0
            for i in 0..<(batchSize * rows) {
                maxDiff = max(maxDiff, abs(gpu.out1[i] - cpu.out1[i]))
                maxDiff = max(maxDiff, abs(gpu.out2[i] - cpu.out2[i]))
            }

            fputs(
                "  fused_dual_affine_matvec rows=\(rows) cols=\(cols) vs CPU: "
                    + "max diff = \(String(format: "%.6f", maxDiff))\n",
                stderr
            )
            XCTAssertLessThan(
                maxDiff,
                0.02,
                "generic fused_dual_affine_matvec rows=\(rows) cols=\(cols) diverges from CPU reference"
            )
        }
    }

    // MARK: - Matmul test helpers

    /// Quantize a test weight matrix and return (indices buffer, LUT buffer, numGroups).
    private func quantizeTestWeights(
        rows: Int, cols: Int, groupSize: Int
    ) throws -> (MTLBuffer, MTLBuffer, Int) {
        let count = rows * cols
        var weightFP16 = [Float16](repeating: 0, count: count)
        for idx in 0..<count {
            weightFP16[idx] = Float16(sin(Float(idx) * 0.05) * 0.1)
        }

        let outputPath = NSTemporaryDirectory()
            + "prefill_test_\(ProcessInfo.processInfo.globallyUniqueString).bin"
        defer { try? FileManager.default.removeItem(atPath: outputPath) }

        let config = SmeltQuantizationConfig(
            strategy: .lutU4, groupSize: groupSize, excludePatterns: []
        )
        let entries = try weightFP16.withUnsafeBytes { raw in
            try SmeltQuantizer.quantize(
                tensors: [(
                    runtimeName: "test_weight",
                    data: raw.baseAddress! as UnsafeRawPointer,
                    byteCount: count * 2,
                    shape: [rows, cols],
                    dtype: "F16"
                )],
                config: config,
                outputPath: outputPath,
                device: device
            )
        }

        let entry = entries[0]
        let fileData = try Data(contentsOf: URL(fileURLWithPath: outputPath))
        let numGroups = rows / groupSize

        let indicesBuf = fileData.withUnsafeBytes { raw in
            device.makeBuffer(
                bytes: raw.baseAddress!.advanced(by: Int(entry.offset)),
                length: count / 2,
                options: .storageModeShared
            )!
        }
        let lutBuf = fileData.withUnsafeBytes { raw in
            device.makeBuffer(
                bytes: raw.baseAddress!.advanced(by: Int(entry.lutOffset!)),
                length: numGroups * 16 * 2,
                options: .storageModeShared
            )!
        }

        return (indicesBuf, lutBuf, numGroups)
    }

    /// Run fused_lut_matvec B times, once per input vector. Returns [B * rows] reference.
    private func runIteratedMatvec(
        pipeline: MTLComputePipelineState,
        indices: MTLBuffer, lut: MTLBuffer,
        inputData: [Float16],
        rows: Int, cols: Int, groupSize: Int, batchSize: Int
    ) throws -> [Float] {
        var reference = [Float](repeating: 0, count: batchSize * rows)
        let singleInputBuf = device.makeBuffer(
            length: cols * 2, options: .storageModeShared
        )!
        let singleOutputBuf = device.makeBuffer(
            length: rows * 2, options: .storageModeShared
        )!

        for b in 0..<batchSize {
            let slice = Array(inputData[b * cols..<(b + 1) * cols])
            memcpy(singleInputBuf.contents(), slice, cols * 2)
            memset(singleOutputBuf.contents(), 0, rows * 2)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(indices, offset: 0, index: 0)
            enc.setBuffer(lut, offset: 0, index: 1)
            enc.setBuffer(singleInputBuf, offset: 0, index: 2)
            enc.setBuffer(singleOutputBuf, offset: 0, index: 3)
            var colsVal = UInt32(cols)
            enc.setBytes(&colsVal, length: 4, index: 4)
            var gsVal = UInt32(groupSize)
            enc.setBytes(&gsVal, length: 4, index: 5)
            var nrVal = UInt32(rows)
            enc.setBytes(&nrVal, length: 4, index: 6)
            enc.dispatchThreadgroups(
                MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            if let error = cmdBuf.error {
                throw error
            }

            let ptr = singleOutputBuf.contents().bindMemory(
                to: Float16.self, capacity: rows
            )
            for r in 0..<rows {
                reference[b * rows + r] = Float(ptr[r])
            }
        }

        return reference
    }

    private func runIteratedAffineMatvec(
        pipeline: MTLComputePipelineState,
        weights: MTLBuffer,
        scales: MTLBuffer,
        biases: MTLBuffer,
        inputData: [Float16],
        rows: Int,
        cols: Int,
        batchSize: Int
    ) throws -> [Float] {
        var reference = [Float](repeating: 0, count: batchSize * rows)
        let singleInputBuf = device.makeBuffer(length: cols * 2, options: .storageModeShared)!
        let singleOutputBuf = device.makeBuffer(length: rows * 2, options: .storageModeShared)!

        for b in 0..<batchSize {
            let slice = Array(inputData[b * cols..<(b + 1) * cols])
            memcpy(singleInputBuf.contents(), slice, cols * 2)
            memset(singleOutputBuf.contents(), 0, rows * 2)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(weights, offset: 0, index: 0)
            enc.setBuffer(scales, offset: 0, index: 1)
            enc.setBuffer(biases, offset: 0, index: 2)
            enc.setBuffer(singleInputBuf, offset: 0, index: 3)
            enc.setBuffer(singleOutputBuf, offset: 0, index: 4)
            var rowsVal = UInt32(rows)
            enc.setBytes(&rowsVal, length: 4, index: 5)
            enc.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error { throw error }

            let ptr = singleOutputBuf.contents().bindMemory(to: Float16.self, capacity: rows)
            for r in 0..<rows {
                reference[b * rows + r] = Float(ptr[r])
            }
        }

        return reference
    }

    private func runIteratedAffineGateUpGeGLUReference(
        gateWeights: MTLBuffer,
        gateScales: MTLBuffer,
        gateBiases: MTLBuffer,
        upWeights: MTLBuffer,
        upScales: MTLBuffer,
        upBiases: MTLBuffer,
        inputData: [Float16],
        rows: Int,
        cols: Int,
        groupSize: Int,
        batchSize: Int
    ) throws -> [Float] {
        let affinePipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "affine_matvec",
            cols: UInt32(cols),
            groupSize: UInt32(groupSize)
        )
        XCTAssertNotNil(affinePipeline)
        let gegluPipeline = makePipeline(
            shaderFile: "activations.metal",
            functionName: "geglu_fused"
        )
        XCTAssertNotNil(gegluPipeline)

        var reference = [Float](repeating: 0, count: batchSize * rows)
        let singleInputBuf = device.makeBuffer(length: cols * 2, options: .storageModeShared)!
        let gateBuf = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
        let upBuf = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: rows * 2, options: .storageModeShared)!

        for b in 0..<batchSize {
            let slice = Array(inputData[b * cols..<(b + 1) * cols])
            memcpy(singleInputBuf.contents(), slice, cols * 2)
            memset(gateBuf.contents(), 0, rows * 2)
            memset(upBuf.contents(), 0, rows * 2)
            memset(outputBuf.contents(), 0, rows * 2)

            let cmdBuf = queue.makeCommandBuffer()!

            do {
                let enc = cmdBuf.makeComputeCommandEncoder()!
                enc.setComputePipelineState(affinePipeline!)
                enc.setBuffer(gateWeights, offset: 0, index: 0)
                enc.setBuffer(gateScales, offset: 0, index: 1)
                enc.setBuffer(gateBiases, offset: 0, index: 2)
                enc.setBuffer(singleInputBuf, offset: 0, index: 3)
                enc.setBuffer(gateBuf, offset: 0, index: 4)
                var rowsVal = UInt32(rows)
                enc.setBytes(&rowsVal, length: 4, index: 5)
                enc.dispatchThreadgroups(
                    MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
                )
                enc.endEncoding()
            }

            do {
                let enc = cmdBuf.makeComputeCommandEncoder()!
                enc.setComputePipelineState(affinePipeline!)
                enc.setBuffer(upWeights, offset: 0, index: 0)
                enc.setBuffer(upScales, offset: 0, index: 1)
                enc.setBuffer(upBiases, offset: 0, index: 2)
                enc.setBuffer(singleInputBuf, offset: 0, index: 3)
                enc.setBuffer(upBuf, offset: 0, index: 4)
                var rowsVal = UInt32(rows)
                enc.setBytes(&rowsVal, length: 4, index: 5)
                enc.dispatchThreadgroups(
                    MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
                )
                enc.endEncoding()
            }

            do {
                let enc = cmdBuf.makeComputeCommandEncoder()!
                enc.setComputePipelineState(gegluPipeline!)
                enc.setBuffer(gateBuf, offset: 0, index: 0)
                enc.setBuffer(upBuf, offset: 0, index: 1)
                enc.setBuffer(outputBuf, offset: 0, index: 2)
                var rowsVal = UInt32(rows)
                enc.setBytes(&rowsVal, length: 4, index: 3)
                enc.dispatchThreads(
                    MTLSize(width: rows, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(rows, 1024), height: 1, depth: 1)
                )
                enc.endEncoding()
            }

            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error { throw error }

            let ptr = outputBuf.contents().bindMemory(to: Float16.self, capacity: rows)
            for r in 0..<rows {
                reference[b * rows + r] = Float(ptr[r])
            }
        }

        return reference
    }

    private func runIteratedFusedAffineGateUpSwiglu(
        pipeline: MTLComputePipelineState,
        gateWeights: MTLBuffer,
        gateScales: MTLBuffer,
        gateBiases: MTLBuffer,
        upWeights: MTLBuffer,
        upScales: MTLBuffer,
        upBiases: MTLBuffer,
        inputData: [Float16],
        rows: Int,
        cols: Int,
        batchSize: Int
    ) throws -> [Float] {
        var reference = [Float](repeating: 0, count: batchSize * rows)
        let singleInputBuf = device.makeBuffer(length: cols * 2, options: .storageModeShared)!
        let singleOutputBuf = device.makeBuffer(length: rows * 2, options: .storageModeShared)!

        for b in 0..<batchSize {
            let slice = Array(inputData[b * cols..<(b + 1) * cols])
            memcpy(singleInputBuf.contents(), slice, cols * 2)
            memset(singleOutputBuf.contents(), 0, rows * 2)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(gateWeights, offset: 0, index: 0)
            enc.setBuffer(gateScales, offset: 0, index: 1)
            enc.setBuffer(gateBiases, offset: 0, index: 2)
            enc.setBuffer(upWeights, offset: 0, index: 3)
            enc.setBuffer(upScales, offset: 0, index: 4)
            enc.setBuffer(upBiases, offset: 0, index: 5)
            enc.setBuffer(singleInputBuf, offset: 0, index: 6)
            enc.setBuffer(singleOutputBuf, offset: 0, index: 7)
            var rowsVal = UInt32(rows)
            enc.setBytes(&rowsVal, length: 4, index: 8)
            enc.dispatchThreadgroups(
                MTLSize(width: rows / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error { throw error }

            let ptr = singleOutputBuf.contents().bindMemory(to: Float16.self, capacity: rows)
            for r in 0..<rows {
                reference[b * rows + r] = Float(ptr[r])
            }
        }

        return reference
    }

    private func runIteratedFusedAffineGateUpGeGLU(
        pipeline: MTLComputePipelineState,
        gateWeights: MTLBuffer,
        gateScales: MTLBuffer,
        gateBiases: MTLBuffer,
        upWeights: MTLBuffer,
        upScales: MTLBuffer,
        upBiases: MTLBuffer,
        inputData: [Float16],
        rows: Int,
        cols: Int,
        batchSize: Int
    ) throws -> [Float] {
        var reference = [Float](repeating: 0, count: batchSize * rows)
        let singleInputBuf = device.makeBuffer(length: cols * 2, options: .storageModeShared)!
        let singleOutputBuf = device.makeBuffer(length: rows * 2, options: .storageModeShared)!

        for b in 0..<batchSize {
            let slice = Array(inputData[b * cols..<(b + 1) * cols])
            memcpy(singleInputBuf.contents(), slice, cols * 2)
            memset(singleOutputBuf.contents(), 0, rows * 2)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(gateWeights, offset: 0, index: 0)
            enc.setBuffer(gateScales, offset: 0, index: 1)
            enc.setBuffer(gateBiases, offset: 0, index: 2)
            enc.setBuffer(upWeights, offset: 0, index: 3)
            enc.setBuffer(upScales, offset: 0, index: 4)
            enc.setBuffer(upBiases, offset: 0, index: 5)
            enc.setBuffer(singleInputBuf, offset: 0, index: 6)
            enc.setBuffer(singleOutputBuf, offset: 0, index: 7)
            var rowsVal = UInt32(rows)
            enc.setBytes(&rowsVal, length: 4, index: 8)
            enc.dispatchThreadgroups(
                MTLSize(width: rows / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error { throw error }

            let ptr = singleOutputBuf.contents().bindMemory(to: Float16.self, capacity: rows)
            for r in 0..<rows {
                reference[b * rows + r] = Float(ptr[r])
            }
        }

        return reference
    }

    private func runIteratedFusedDualAffineMatvec(
        pipeline: MTLComputePipelineState,
        w1Weights: MTLBuffer,
        w1Scales: MTLBuffer,
        w1Biases: MTLBuffer,
        w2Weights: MTLBuffer,
        w2Scales: MTLBuffer,
        w2Biases: MTLBuffer,
        inputData: [Float16],
        rows: Int,
        cols: Int,
        batchSize: Int,
        rowTile: Int = 8,
        setRowsBuffer: Bool = true
    ) throws -> (out1: [Float], out2: [Float]) {
        var ref1 = [Float](repeating: 0, count: batchSize * rows)
        var ref2 = [Float](repeating: 0, count: batchSize * rows)
        let singleInputBuf = device.makeBuffer(length: cols * 2, options: .storageModeShared)!
        let singleOut1 = device.makeBuffer(length: rows * 2, options: .storageModeShared)!
        let singleOut2 = device.makeBuffer(length: rows * 2, options: .storageModeShared)!

        for b in 0..<batchSize {
            let slice = Array(inputData[b * cols..<(b + 1) * cols])
            memcpy(singleInputBuf.contents(), slice, cols * 2)
            memset(singleOut1.contents(), 0, rows * 2)
            memset(singleOut2.contents(), 0, rows * 2)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(w1Weights, offset: 0, index: 0)
            enc.setBuffer(w1Scales, offset: 0, index: 1)
            enc.setBuffer(w1Biases, offset: 0, index: 2)
            enc.setBuffer(w2Weights, offset: 0, index: 3)
            enc.setBuffer(w2Scales, offset: 0, index: 4)
            enc.setBuffer(w2Biases, offset: 0, index: 5)
            enc.setBuffer(singleInputBuf, offset: 0, index: 6)
            enc.setBuffer(singleOut1, offset: 0, index: 7)
            enc.setBuffer(singleOut2, offset: 0, index: 8)
            if setRowsBuffer {
                var rowsVal = UInt32(rows)
                enc.setBytes(&rowsVal, length: 4, index: 9)
            }
            enc.dispatchThreadgroups(
                MTLSize(width: (rows + rowTile - 1) / rowTile, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error { throw error }

            let ptr1 = singleOut1.contents().bindMemory(to: Float16.self, capacity: rows)
            let ptr2 = singleOut2.contents().bindMemory(to: Float16.self, capacity: rows)
            for r in 0..<rows {
                ref1[b * rows + r] = Float(ptr1[r])
                ref2[b * rows + r] = Float(ptr2[r])
            }
        }

        return (ref1, ref2)
    }

    func testE4BLMHeadVerifyArgmaxMatchesFullLogitsWhenEnabled() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["SMELT_RUN_LM_HEAD_ARGMAX_PARITY"] == "1",
            "Full E4B LM-head argmax parity is opt-in"
        )

        guard let fullPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "affine_matvec_c2560_r262144_g128_rows8"
        ), let argmaxPipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "affine_matvec_argmax_c2560_r262144_g128_batched"
        ), let reducePipeline = makePipeline(
            shaderFile: "lut_matvec.metal",
            functionName: "lm_head_argmax_reduce_r262144"
        ) else {
            throw XCTSkip("LM-head parity pipelines unavailable on this Metal device")
        }

        let rows = 262_144
        let cols = 2_560
        let groupSize = 128
        let batchSize = 3
        let data = makeAffineTestData(
            rows: rows,
            cols: cols,
            groupSize: groupSize,
            seed: 77
        )
        var inputData = [Float16](repeating: 0, count: batchSize * cols)
        for b in 0..<batchSize {
            for c in 0..<cols {
                inputData[b * cols + c] = Float16(
                    sin((Float(c) + Float(b * 97)) * 0.0097) * 0.45
                )
            }
        }

        let weightsBuf = device.makeBuffer(
            bytes: data.weights, length: data.weights.count, options: .storageModeShared
        )!
        let scalesBuf = device.makeBuffer(
            bytes: data.scales, length: data.scales.count * 2, options: .storageModeShared
        )!
        let biasesBuf = device.makeBuffer(
            bytes: data.biases, length: data.biases.count * 2, options: .storageModeShared
        )!
        let inputBuf = device.makeBuffer(
            bytes: inputData, length: inputData.count * 2, options: .storageModeShared
        )!
        let logitsBuf = device.makeBuffer(
            length: batchSize * rows * 2, options: .storageModeShared
        )!
        let partialBuf = device.makeBuffer(
            length: batchSize * (rows / 8) * MemoryLayout<UInt64>.stride,
            options: .storageModeShared
        )!
        let argmaxBuf = device.makeBuffer(
            length: batchSize * MemoryLayout<Int32>.stride,
            options: .storageModeShared
        )!
        memset(logitsBuf.contents(), 0, batchSize * rows * 2)
        memset(partialBuf.contents(), 0, batchSize * (rows / 8) * MemoryLayout<UInt64>.stride)
        memset(argmaxBuf.contents(), 0, batchSize * MemoryLayout<Int32>.stride)

        for b in 0..<batchSize {
            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(fullPipeline)
            enc.setBuffer(weightsBuf, offset: 0, index: 0)
            enc.setBuffer(scalesBuf, offset: 0, index: 1)
            enc.setBuffer(biasesBuf, offset: 0, index: 2)
            enc.setBuffer(inputBuf, offset: b * cols * 2, index: 3)
            enc.setBuffer(logitsBuf, offset: b * rows * 2, index: 4)
            enc.dispatchThreadgroups(
                MTLSize(width: rows / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error { throw error }
        }

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(argmaxPipeline)
        enc.setBuffer(weightsBuf, offset: 0, index: 0)
        enc.setBuffer(scalesBuf, offset: 0, index: 1)
        enc.setBuffer(biasesBuf, offset: 0, index: 2)
        enc.setBuffer(inputBuf, offset: 0, index: 3)
        enc.setBuffer(partialBuf, offset: 0, index: 4)
        var actualBatch = UInt32(batchSize)
        enc.setBytes(&actualBatch, length: 4, index: 5)
        var logitCap: Float = 0
        enc.setBytes(&logitCap, length: 4, index: 6)
        enc.dispatchThreadgroups(
            MTLSize(width: rows / 8, height: (batchSize + 3) / 4, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw error }

        let reduceCmdBuf = queue.makeCommandBuffer()!
        let reduceEnc = reduceCmdBuf.makeComputeCommandEncoder()!
        reduceEnc.setComputePipelineState(reducePipeline)
        reduceEnc.setBuffer(partialBuf, offset: 0, index: 0)
        reduceEnc.setBuffer(argmaxBuf, offset: 0, index: 1)
        reduceEnc.setBytes(&actualBatch, length: 4, index: 2)
        reduceEnc.dispatchThreadgroups(
            MTLSize(width: batchSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        reduceEnc.endEncoding()
        reduceCmdBuf.commit()
        reduceCmdBuf.waitUntilCompleted()
        if let error = reduceCmdBuf.error { throw error }

        let logitsPtr = logitsBuf.contents().bindMemory(
            to: Float16.self,
            capacity: batchSize * rows
        )
        let argmaxPtr = argmaxBuf.contents().bindMemory(
            to: Int32.self,
            capacity: batchSize
        )
        for b in 0..<batchSize {
            var expected = 0
            var expectedValue = logitsPtr[b * rows]
            for row in 1..<rows {
                let value = logitsPtr[b * rows + row]
                if Float(value) > Float(expectedValue) {
                    expected = row
                    expectedValue = value
                }
            }
            let actual = Int(argmaxPtr[b])
            XCTAssertEqual(actual, expected, "batch \(b)")
        }
    }
}
