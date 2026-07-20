// MatvecKernelTests — Kernel-level correctness test for fused_lut_matvec.
//
// Creates a small weight matrix, quantizes it, runs the Metal kernel,
// and compares output to a CPU reference. This isolates whether the
// kernel correctly reads packed u4+LUT format.

import Metal
import XCTest

@testable import SmeltCompiler
@testable import SmeltSchema

final class MatvecKernelTests: XCTestCase {

    private var device: MTLDevice!
    private var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        try? XCTSkipIf(device == nil, "No Metal device available")
        queue = device.makeCommandQueue()
    }

    private func makeLUTMatvecPipeline(cols: UInt32, groupSize: UInt32) -> MTLComputePipelineState? {
        guard let source = loadMetalShaderSource("lut_matvec.metal"),
              let lib = try? device.makeLibrary(source: source, options: nil)
        else { return nil }

        let constants = MTLFunctionConstantValues()
        var colsVal = cols
        var groupSizeVal = groupSize
        constants.setConstantValue(&colsVal, type: .uint, index: 0)
        constants.setConstantValue(&groupSizeVal, type: .uint, index: 1)
        guard let fn = try? lib.makeFunction(name: "fused_lut_matvec", constantValues: constants) else {
            return nil
        }
        return try? device.makeComputePipelineState(function: fn)
    }

    // MARK: - Direct kernel test

    /// Create known data, pack it manually, run the kernel, compare to CPU matmul.
    func testFusedLutMatvec_SmallMatrix() throws {
        let rows = 4
        let cols = 8
        let groupSize = 4  // one group for all 4 rows
        let halfCols = cols / 2
        guard let pipeline = makeLUTMatvecPipeline(cols: UInt32(cols), groupSize: UInt32(groupSize)) else {
            throw XCTSkip("Could not compile lut_matvec shader")
        }

        // Create a simple weight matrix (FP16 values)
        // W = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        //      [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        //      [0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5],
        //      [0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9]]
        let weightFP32: [[Float]] = [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5],
            [0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9],
        ]

        // Input vector
        let inputFP32: [Float] = [1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.5, 0.5]

        // CPU reference: W @ x
        var expectedOutput = [Float](repeating: 0, count: rows)
        for row in 0..<rows {
            for col in 0..<cols {
                expectedOutput[row] += weightFP32[row][col] * inputFP32[col]
            }
        }
        fputs("  CPU reference output: \(expectedOutput)\n", stderr)

        // Now quantize the weight: KMeans with k=16 on all 4×8=32 values
        let allValues = weightFP32.flatMap { $0 }

        // For this test, use exact centroids (the unique values in the data)
        let uniqueVals = Array(Set(allValues)).sorted()
        // Pad to 16 centroids
        var lut = [Float16](repeating: 0, count: 16)
        for idx in 0..<min(uniqueVals.count, 16) {
            lut[idx] = Float16(uniqueVals[idx])
        }

        // Assign each element to nearest centroid
        var indices = [UInt8](repeating: 0, count: rows * cols)
        for idx in 0..<allValues.count {
            var best = 0
            var bestDist: Float = .infinity
            for cidx in 0..<16 {
                let dist = abs(allValues[idx] - Float(lut[cidx]))
                if dist < bestDist { bestDist = dist; best = cidx }
            }
            indices[idx] = UInt8(best)
        }

        // Pack into u4: low nibble = even index, high nibble = odd index
        var packed = [UInt8](repeating: 0, count: rows * halfCols)
        for row in 0..<rows {
            for col in 0..<halfCols {
                let lo = indices[row * cols + col * 2]
                let hi = indices[row * cols + col * 2 + 1]
                packed[row * halfCols + col] = (hi << 4) | lo
            }
        }

        // CPU reference with quantized values (should be close to expectedOutput)
        var quantizedOutput = [Float](repeating: 0, count: rows)
        for row in 0..<rows {
            for col in 0..<halfCols {
                let byte = packed[row * halfCols + col]
                let loVal = Float(lut[Int(byte & 0x0F)])
                let hiVal = Float(lut[Int(byte >> 4)])
                quantizedOutput[row] += loVal * inputFP32[col * 2]
                quantizedOutput[row] += hiVal * inputFP32[col * 2 + 1]
            }
        }
        fputs("  CPU quantized output: \(quantizedOutput)\n", stderr)

        // Create Metal buffers
        let indicesBuf = device.makeBuffer(
            bytes: packed, length: packed.count, options: .storageModeShared
        )!
        let lutBuf = device.makeBuffer(
            bytes: lut, length: lut.count * 2, options: .storageModeShared
        )!
        let inputFP16 = inputFP32.map { Float16($0) }
        let inputBuf = device.makeBuffer(
            bytes: inputFP16, length: inputFP16.count * 2, options: .storageModeShared
        )!
        let outputBuf = device.makeBuffer(
            length: rows * 2, options: .storageModeShared
        )!
        memset(outputBuf.contents(), 0, rows * 2)

        // Dispatch
        guard let cmdBuf = queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder()
        else {
            XCTFail("Failed to create command buffer")
            return
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(indicesBuf, offset: 0, index: 0)
        enc.setBuffer(lutBuf, offset: 0, index: 1)
        enc.setBuffer(inputBuf, offset: 0, index: 2)
        enc.setBuffer(outputBuf, offset: 0, index: 3)
        var nrVal = UInt32(rows)
        enc.setBytes(&nrVal, length: 4, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            XCTFail("Metal error: \(error)")
            return
        }

        // Read GPU output
        let gpuPtr = outputBuf.contents().bindMemory(to: Float16.self, capacity: rows)
        var gpuOutput = [Float](repeating: 0, count: rows)
        for idx in 0..<rows { gpuOutput[idx] = Float(gpuPtr[idx]) }

        fputs("  GPU kernel output:    \(gpuOutput)\n", stderr)

        // Compare GPU to CPU quantized reference
        for row in 0..<rows {
            let diff = abs(gpuOutput[row] - quantizedOutput[row])
            XCTAssertLessThan(
                diff, 0.01,
                "Row \(row): GPU=\(gpuOutput[row]) CPU=\(quantizedOutput[row]) diff=\(diff)"
            )
        }
    }

    /// Test with the actual SmeltQuantizer output — quantize a small tensor,
    /// then run fused_lut_matvec and compare to CPU dequant+matmul.
    func testFusedLutMatvec_WithAgentQuantizer() throws {
        let rows = 32
        let cols = 64
        let groupSize = 16
        let halfCols = cols / 2
        let count = rows * cols
        guard let pipeline = makeLUTMatvecPipeline(cols: UInt32(cols), groupSize: UInt32(groupSize)) else {
            throw XCTSkip("Could not compile lut_matvec shader")
        }

        // Generate weight-like data
        var weightFP16 = [Float16](repeating: 0, count: count)
        for idx in 0..<count {
            weightFP16[idx] = Float16(sin(Float(idx) * 0.05) * 0.1)
        }

        // Generate input vector
        var inputFP16 = [Float16](repeating: 0, count: cols)
        for idx in 0..<cols {
            inputFP16[idx] = Float16(cos(Float(idx) * 0.1) * 0.5)
        }

        // Quantize with SmeltQuantizer
        let outputPath = NSTemporaryDirectory()
            + "matvec_test_\(ProcessInfo.processInfo.globallyUniqueString).bin"
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

        // CPU dequant + matmul reference
        var cpuOutput = [Float](repeating: 0, count: rows)
        fileData.withUnsafeBytes { raw in
            let packed = raw.baseAddress!.advanced(by: Int(entry.offset))
                .bindMemory(to: UInt8.self, capacity: count / 2)
            let lutBase = raw.baseAddress!.advanced(by: Int(entry.lutOffset!))
                .bindMemory(to: Float16.self, capacity: numGroups * 16)

            for row in 0..<rows {
                let group = row / groupSize
                let groupLUT = lutBase + group * 16
                var acc: Float = 0
                for col in 0..<halfCols {
                    let byte = packed[row * halfCols + col]
                    let loVal = Float(groupLUT[Int(byte & 0x0F)])
                    let hiVal = Float(groupLUT[Int(byte >> 4)])
                    acc += loVal * Float(inputFP16[col * 2])
                    acc += hiVal * Float(inputFP16[col * 2 + 1])
                }
                cpuOutput[row] = acc
            }
        }

        // GPU kernel
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
        let inputBuf = device.makeBuffer(
            bytes: inputFP16, length: cols * 2, options: .storageModeShared
        )!
        let outputBuf = device.makeBuffer(
            length: rows * 2, options: .storageModeShared
        )!
        memset(outputBuf.contents(), 0, rows * 2)

        guard let cmdBuf = queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder()
        else {
            XCTFail("Failed to create command buffer")
            return
        }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(indicesBuf, offset: 0, index: 0)
        enc.setBuffer(lutBuf, offset: 0, index: 1)
        enc.setBuffer(inputBuf, offset: 0, index: 2)
        enc.setBuffer(outputBuf, offset: 0, index: 3)
        var nrVal = UInt32(rows)
        enc.setBytes(&nrVal, length: 4, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        XCTAssertNil(cmdBuf.error, "Metal error: \(cmdBuf.error?.localizedDescription ?? "")")

        // Compare
        let gpuPtr = outputBuf.contents().bindMemory(to: Float16.self, capacity: rows)
        var maxDiff: Float = 0
        var mismatches = 0
        for row in 0..<rows {
            let gpuVal = Float(gpuPtr[row])
            let cpuVal = cpuOutput[row]
            let diff = abs(gpuVal - cpuVal)
            if diff > maxDiff { maxDiff = diff }
            if diff > 0.01 { mismatches += 1 }
        }

        fputs(
            "  SmeltQuantizer → fused_lut_matvec test [\(rows), \(cols)]:\n"
                + "    Max diff: \(String(format: "%.6f", maxDiff))\n"
                + "    Mismatches (>0.01): \(mismatches)/\(rows)\n"
                + "    GPU[0..3]: \(Float(gpuPtr[0])) \(Float(gpuPtr[1]))"
                + " \(Float(gpuPtr[2])) \(Float(gpuPtr[3]))\n"
                + "    CPU[0..3]: \(cpuOutput[0]) \(cpuOutput[1])"
                + " \(cpuOutput[2]) \(cpuOutput[3])\n",
            stderr
        )

        XCTAssertEqual(
            mismatches, 0,
            "GPU and CPU outputs differ on \(mismatches)/\(rows) rows (max diff \(maxDiff))"
        )
    }
}
