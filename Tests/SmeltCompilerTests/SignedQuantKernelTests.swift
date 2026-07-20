import Metal
import XCTest

@testable import SmeltCompiler

final class SignedQuantKernelTests: XCTestCase {
    private func packedCodes(
        rows: Int,
        cols: Int,
        bits: Int,
        code: (Int, Int) -> UInt8
    ) -> [UInt8] {
        let rowBytes = cols * bits / 8
        var bytes = [UInt8](repeating: 0, count: rows * rowBytes)
        for row in 0..<rows {
            for col in 0..<cols {
                let bit = col * bits
                bytes[row * rowBytes + bit / 8] |= code(row, col) << UInt8(bit & 7)
            }
        }
        return bytes
    }

    /// Native `.ternary2` is the source-neutral LSB-first two-bit spelling.
    /// Test fixtures use semantic codes so they exercise the same package ABI
    /// as the checkpoint adapter, CPU codec, writer, and Metal consumers.
    private func nativeTernaryCodes(
        rows: Int,
        cols: Int,
        code: (Int, Int) -> UInt8
    ) -> [UInt8] {
        packedCodes(rows: rows, cols: cols, bits: 2, code: code)
    }

    private func run(
        function: String,
        codes: [UInt8],
        scales: [Float16],
        input: [Float16],
        rows: Int,
        cols: Int,
        batches: Int,
        batchTile: Int = 1,
        rowTile: Int = 8
    ) throws -> [Float16] {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let pipeline = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant.metal",
            functionName: function
        ))
        let output = try makeSharedBuffer(
            device: device, count: rows * batches, of: Float16.self)
        let codeBuffer = try makeSharedBuffer(device: device, codes)
        let scaleBuffer = try makeSharedBuffer(device: device, scales)
        let inputBuffer = try makeSharedBuffer(device: device, input)
        try runOnGPU(queue: queue) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(codeBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(inputBuffer, offset: 0, index: 2)
            encoder.setBuffer(output, offset: 0, index: 3)
            var rowCount = UInt32(rows)
            var colCount = UInt32(cols)
            encoder.setBytes(&rowCount, length: 4, index: 4)
            encoder.setBytes(&colCount, length: 4, index: 5)
            if batchTile > 1 {
                var actualBatch = UInt32(batches)
                encoder.setBytes(&actualBatch, length: 4, index: 6)
            }
            encoder.dispatchThreadgroups(
                MTLSize(
                    width: (rows + rowTile - 1) / rowTile,
                    height: (batches + batchTile - 1) / batchTile,
                    depth: 1
                ),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }
        return Array(UnsafeBufferPointer(
            start: output.contents().assumingMemoryBound(to: Float16.self),
            count: rows * batches
        ))
    }

    func testPreciseTernaryAffineMatvecMatchesMLXQMVFastBitExactly() throws {
        let rows = 9
        let cols = 512
        let groups = cols / 128
        let codes = nativeTernaryCodes(rows: rows, cols: cols) {
            UInt8(($0 * 13 + $1 * 7) % 3)
        }
        let scales = (0..<(rows * groups)).map {
            Float16(0.03125 * Float(($0 % 5) + 1))
        }
        let input = (0..<cols).map {
            Float16(Float(($0 * 17 + 3) % 31 - 15) / 16)
        }
        // Generated with MLX v0.31.1 affine_qmv_fast<half, 128, 2> using
        // the fixtures above. Bit patterns are the contract: a tolerance
        // would hide the exact reduction-order bug this test guards.
        let mlxExpected: [UInt16] = [
            0xb190, 0x3468, 0xb0b0, 0x2e20, 0xa580,
            0x3370, 0xb518, 0x3498, 0xb170,
        ]

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let pipeline = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant_precise.metal",
            functionName: "signed_ternary_affine_matvec_g128_rows8"
        ))
        let codeBuffer = try makeSharedBuffer(device: device, codes)
        let scaleBuffer = try makeSharedBuffer(device: device, scales)
        let inputBuffer = try makeSharedBuffer(device: device, input)
        let output = try makeSharedBuffer(
            device: device, count: rows, of: Float16.self)
        try runOnGPU(queue: queue) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(codeBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 2)
            encoder.setBuffer(inputBuffer, offset: 0, index: 3)
            encoder.setBuffer(output, offset: 0, index: 4)
            var rowCount = UInt32(rows)
            var colCount = UInt32(cols)
            encoder.setBytes(&rowCount, length: 4, index: 5)
            encoder.setBytes(&colCount, length: 4, index: 6)
            encoder.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }
        let got = UnsafeBufferPointer(
            start: output.contents().assumingMemoryBound(to: Float16.self),
            count: rows
        ).map(\.bitPattern)
        XCTAssertEqual(got, mlxExpected)
    }

    func testPreciseTernaryAffineMatvecPreservesMLXQMVBitsForEveryBatchRow() throws {
        let rows = 9
        // Production-width fixture: ten 512-column QMV blocks catches reduction
        // topology changes that the original one-block oracle cannot see.
        let cols = 5_120
        let batches = 3
        let groups = cols / 128
        let codes = nativeTernaryCodes(rows: rows, cols: cols) {
            UInt8(($0 * 13 + $1 * 7) % 3)
        }
        let scales = (0..<(rows * groups)).map {
            Float16(0.00390625 * Float(($0 % 5) + 1))
        }
        let input = (0..<(batches * cols)).map {
            Float16(Float(($0 * 17 + 3) % 31 - 15) / 16)
        }
        // Generated with MLX v0.31.1 affine_qmv_fast<half, 128, 2>, M=3.
        let mlxExpected: [UInt16] = [
            0xa160, 0x9dc0, 0x2420, 0xa160, 0x9dc0, 0x2420, 0xa160, 0x9dc0, 0x2420,
            0x2520, 0xa650, 0x1cc0, 0x2520, 0xa650, 0x1cc0, 0x2520, 0xa650, 0x1cc0,
            0x2280, 0x1700, 0xa360, 0x2280, 0x1700, 0xa360, 0x2280, 0x1700, 0xa360,
        ]

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let codeBuffer = try makeSharedBuffer(device: device, codes)
        let scaleBuffer = try makeSharedBuffer(device: device, scales)
        let inputBuffer = try makeSharedBuffer(device: device, input)
        let pipeline = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant_precise.metal",
            functionName: "signed_ternary_affine_matvec_g128_rows8"
        ))
        let output = try makeSharedBuffer(
            device: device, count: rows * batches, of: Float16.self)
        try runOnGPU(queue: queue) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(codeBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 2)
            encoder.setBuffer(inputBuffer, offset: 0, index: 3)
            encoder.setBuffer(output, offset: 0, index: 4)
            var rowCount = UInt32(rows)
            var colCount = UInt32(cols)
            encoder.setBytes(&rowCount, length: 4, index: 5)
            encoder.setBytes(&colCount, length: 4, index: 6)
            encoder.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: batches, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }
        let got = UnsafeBufferPointer(
            start: output.contents().assumingMemoryBound(to: Float16.self),
            count: rows * batches
        ).map(\.bitPattern)
        XCTAssertEqual(got, mlxExpected)
    }

    func testPreciseTernaryAffineGateUpSwiGLUMatchesStagedPathBitExactly() throws {
        let rows = 9
        let cols = 5_120
        let groups = cols / 128
        let gateCodes = nativeTernaryCodes(rows: rows, cols: cols) {
            UInt8(($0 * 13 + $1 * 7) % 3)
        }
        let upCodes = nativeTernaryCodes(rows: rows, cols: cols) {
            UInt8(($0 * 5 + $1 * 11 + 1) % 3)
        }
        let gateScales = (0..<(rows * groups)).map {
            Float16(0.00390625 * Float(($0 % 5) + 1))
        }
        let upScales = (0..<(rows * groups)).map {
            Float16(0.0029296875 * Float(($0 % 7) + 1))
        }
        let input = (0..<cols).map {
            Float16(Float(($0 * 17 + 3) % 31 - 15) / 16)
        }

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let matvec = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant_precise.metal",
            functionName: "signed_ternary_affine_matvec_g128_rows8"
        ))
        let swiglu = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "activations_precise.metal",
            functionName: "swiglu_fused"
        ))
        let fused = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant_precise.metal",
            functionName: "signed_ternary_affine_gate_up_swiglu_g128_rows8"
        ))
        let gateCodeBuffer = try makeSharedBuffer(device: device, gateCodes)
        let gateScaleBuffer = try makeSharedBuffer(device: device, gateScales)
        let upCodeBuffer = try makeSharedBuffer(device: device, upCodes)
        let upScaleBuffer = try makeSharedBuffer(device: device, upScales)
        let inputBuffer = try makeSharedBuffer(device: device, input)
        let stagedGate = try makeSharedBuffer(
            device: device, count: rows, of: Float16.self)
        let stagedUp = try makeSharedBuffer(
            device: device, count: rows, of: Float16.self)
        let stagedOutput = try makeSharedBuffer(
            device: device, count: rows, of: Float16.self)
        let fusedOutput = try makeSharedBuffer(
            device: device, count: rows, of: Float16.self)

        func encodeMatvec(
            _ encoder: MTLComputeCommandEncoder,
            codes: MTLBuffer,
            scales: MTLBuffer,
            output: MTLBuffer
        ) {
            encoder.setComputePipelineState(matvec)
            encoder.setBuffer(codes, offset: 0, index: 0)
            encoder.setBuffer(scales, offset: 0, index: 1)
            encoder.setBuffer(scales, offset: 0, index: 2)
            encoder.setBuffer(inputBuffer, offset: 0, index: 3)
            encoder.setBuffer(output, offset: 0, index: 4)
            var rowCount = UInt32(rows)
            var colCount = UInt32(cols)
            encoder.setBytes(&rowCount, length: 4, index: 5)
            encoder.setBytes(&colCount, length: 4, index: 6)
            encoder.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }

        try runOnGPU(queue: queue) { encoder in
            encodeMatvec(
                encoder, codes: gateCodeBuffer, scales: gateScaleBuffer,
                output: stagedGate)
            encodeMatvec(
                encoder, codes: upCodeBuffer, scales: upScaleBuffer,
                output: stagedUp)

            encoder.setComputePipelineState(swiglu)
            encoder.setBuffer(stagedGate, offset: 0, index: 0)
            encoder.setBuffer(stagedUp, offset: 0, index: 1)
            encoder.setBuffer(stagedOutput, offset: 0, index: 2)
            var rowCount = UInt32(rows)
            encoder.setBytes(&rowCount, length: 4, index: 3)
            encoder.dispatchThreads(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: rows, height: 1, depth: 1)
            )

            encoder.setComputePipelineState(fused)
            encoder.setBuffer(gateCodeBuffer, offset: 0, index: 0)
            encoder.setBuffer(gateScaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(gateScaleBuffer, offset: 0, index: 2)
            encoder.setBuffer(upCodeBuffer, offset: 0, index: 3)
            encoder.setBuffer(upScaleBuffer, offset: 0, index: 4)
            encoder.setBuffer(upScaleBuffer, offset: 0, index: 5)
            encoder.setBuffer(inputBuffer, offset: 0, index: 6)
            encoder.setBuffer(fusedOutput, offset: 0, index: 7)
            encoder.setBytes(&rowCount, length: 4, index: 8)
            var colCount = UInt32(cols)
            encoder.setBytes(&colCount, length: 4, index: 9)
            encoder.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }

        let stagedBits = UnsafeBufferPointer(
            start: stagedOutput.contents().assumingMemoryBound(to: Float16.self),
            count: rows
        ).map(\.bitPattern)
        let fusedBits = UnsafeBufferPointer(
            start: fusedOutput.contents().assumingMemoryBound(to: Float16.self),
            count: rows
        ).map(\.bitPattern)
        XCTAssertEqual(fusedBits, stagedBits)
    }

    func testPreciseTernaryAffineResidualAddMatchesStagedPathBitExactly() throws {
        let rows = 9
        let cols = 5_120
        let groups = cols / 128
        let codes = nativeTernaryCodes(rows: rows, cols: cols) {
            UInt8(($0 * 13 + $1 * 7) % 3)
        }
        let scales = (0..<(rows * groups)).map {
            Float16(0.00390625 * Float(($0 % 5) + 1))
        }
        let input = (0..<cols).map {
            Float16(Float(($0 * 17 + 3) % 31 - 15) / 16)
        }
        let residual = (0..<rows).map {
            Float16(Float(($0 * 11 + 5) % 17 - 8) / 8)
        }

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let matvec = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant_precise.metal",
            functionName: "signed_ternary_affine_matvec_g128_rows8"
        ))
        let add = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "activations.metal",
            functionName: "elementwise_add"
        ))
        let fused = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant_precise.metal",
            functionName: "signed_ternary_affine_matvec_add_g128_rows8"
        ))
        let codeBuffer = try makeSharedBuffer(device: device, codes)
        let scaleBuffer = try makeSharedBuffer(device: device, scales)
        let inputBuffer = try makeSharedBuffer(device: device, input)
        let residualBuffer = try makeSharedBuffer(device: device, residual)
        let stagedProjection = try makeSharedBuffer(
            device: device, count: rows, of: Float16.self)
        let stagedOutput = try makeSharedBuffer(
            device: device, count: rows, of: Float16.self)
        let fusedOutput = try makeSharedBuffer(
            device: device, count: rows, of: Float16.self)

        try runOnGPU(queue: queue) { encoder in
            encoder.setComputePipelineState(matvec)
            encoder.setBuffer(codeBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 2)
            encoder.setBuffer(inputBuffer, offset: 0, index: 3)
            encoder.setBuffer(stagedProjection, offset: 0, index: 4)
            var rowCount = UInt32(rows)
            var colCount = UInt32(cols)
            encoder.setBytes(&rowCount, length: 4, index: 5)
            encoder.setBytes(&colCount, length: 4, index: 6)
            encoder.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )

            encoder.setComputePipelineState(add)
            encoder.setBuffer(stagedProjection, offset: 0, index: 0)
            encoder.setBuffer(residualBuffer, offset: 0, index: 1)
            encoder.setBuffer(stagedOutput, offset: 0, index: 2)
            encoder.setBytes(&rowCount, length: 4, index: 3)
            encoder.dispatchThreads(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: rows, height: 1, depth: 1)
            )

            encoder.setComputePipelineState(fused)
            encoder.setBuffer(codeBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 2)
            encoder.setBuffer(inputBuffer, offset: 0, index: 3)
            encoder.setBuffer(fusedOutput, offset: 0, index: 4)
            encoder.setBuffer(residualBuffer, offset: 0, index: 5)
            encoder.setBytes(&rowCount, length: 4, index: 6)
            encoder.setBytes(&colCount, length: 4, index: 7)
            encoder.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }

        let stagedBits = UnsafeBufferPointer(
            start: stagedOutput.contents().assumingMemoryBound(to: Float16.self),
            count: rows
        ).map(\.bitPattern)
        let fusedBits = UnsafeBufferPointer(
            start: fusedOutput.contents().assumingMemoryBound(to: Float16.self),
            count: rows
        ).map(\.bitPattern)
        XCTAssertEqual(fusedBits, stagedBits)
    }

    func testPreciseTernaryAffineProjectionBankMatchesIndependentRowsBitExactly() throws {
        let rowCounts = [3, 5, 2]
        let totalRows = rowCounts.reduce(0, +)
        let cols = 5_120
        let groups = cols / 128
        let rowBytes = cols / 4
        let codes = nativeTernaryCodes(rows: totalRows, cols: cols) {
            UInt8(($0 * 13 + $1 * 7) % 3)
        }
        let scales = (0..<(totalRows * groups)).map {
            Float16(0.00390625 * Float(($0 % 5) + 1))
        }
        let input = (0..<cols).map {
            Float16(Float(($0 * 17 + 3) % 31 - 15) / 16)
        }

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let matvec = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant_precise.metal",
            functionName: "signed_ternary_affine_matvec_g128_rows8"
        ))
        let bank = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant_precise.metal",
            functionName: "signed_ternary_affine_bank4_matvec_g128_rows8"
        ))
        let codeBuffer = try makeSharedBuffer(device: device, codes)
        let scaleBuffer = try makeSharedBuffer(device: device, scales)
        let inputBuffer = try makeSharedBuffer(device: device, input)
        let references = try rowCounts.map {
            try makeSharedBuffer(device: device, count: $0, of: Float16.self)
        }
        let bankOutputs = try rowCounts.map {
            try makeSharedBuffer(device: device, count: $0, of: Float16.self)
        }

        try runOnGPU(queue: queue) { encoder in
            var rowBase = 0
            for index in rowCounts.indices {
                encoder.setComputePipelineState(matvec)
                encoder.setBuffer(
                    codeBuffer, offset: rowBase * rowBytes, index: 0)
                encoder.setBuffer(
                    scaleBuffer,
                    offset: rowBase * groups * MemoryLayout<Float16>.stride,
                    index: 1)
                encoder.setBuffer(
                    scaleBuffer,
                    offset: rowBase * groups * MemoryLayout<Float16>.stride,
                    index: 2)
                encoder.setBuffer(inputBuffer, offset: 0, index: 3)
                encoder.setBuffer(references[index], offset: 0, index: 4)
                var rows = UInt32(rowCounts[index])
                var colCount = UInt32(cols)
                encoder.setBytes(&rows, length: 4, index: 5)
                encoder.setBytes(&colCount, length: 4, index: 6)
                encoder.dispatchThreadgroups(
                    MTLSize(
                        width: (rowCounts[index] + 7) / 8,
                        height: 1,
                        depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
                )
                rowBase += rowCounts[index]
            }

            encoder.setComputePipelineState(bank)
            encoder.setBuffer(codeBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 2)
            encoder.setBuffer(inputBuffer, offset: 0, index: 3)
            encoder.setBuffer(bankOutputs[0], offset: 0, index: 4)
            encoder.setBuffer(bankOutputs[1], offset: 0, index: 5)
            encoder.setBuffer(bankOutputs[2], offset: 0, index: 6)
            encoder.setBuffer(bankOutputs[2], offset: 0, index: 7)
            var rows0 = UInt32(rowCounts[0])
            var rows1 = UInt32(rowCounts[1])
            var rows2 = UInt32(rowCounts[2])
            var rows3: UInt32 = 0
            var colCount = UInt32(cols)
            encoder.setBytes(&rows0, length: 4, index: 8)
            encoder.setBytes(&rows1, length: 4, index: 9)
            encoder.setBytes(&rows2, length: 4, index: 10)
            encoder.setBytes(&rows3, length: 4, index: 11)
            encoder.setBytes(&colCount, length: 4, index: 12)
            encoder.dispatchThreadgroups(
                MTLSize(width: (totalRows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }

        for index in rowCounts.indices {
            let expected = UnsafeBufferPointer(
                start: references[index].contents().assumingMemoryBound(
                    to: Float16.self),
                count: rowCounts[index]
            ).map(\.bitPattern)
            let got = UnsafeBufferPointer(
                start: bankOutputs[index].contents().assumingMemoryBound(
                    to: Float16.self),
                count: rowCounts[index]
            ).map(\.bitPattern)
            XCTAssertEqual(got, expected, "bank member \(index)")
        }
    }

    func testPreciseTernaryAffineQMMMatchesMLXByteLaneDequantizationBitExactly() throws {
        let rows = 8
        let cols = 256
        // M=14 is MLX's QMM admission boundary for this small projection on
        // the canonical GPU. Keeping more than one activation row also guards
        // every row position owned by the first BM32 tile.
        let batches = 14
        let groups = cols / 128
        let codes = nativeTernaryCodes(rows: rows, cols: cols) {
            UInt8(($0 * 13 + $1 * 7) % 3)
        }
        let scaleBits: [UInt16] = [0x1b94, 0x2057, 0x22ab, 0x2491, 0x25f3]
        let scales = (0..<(rows * groups)).map {
            Float16(bitPattern: scaleBits[$0 % scaleBits.count])
        }
        let input = (0..<(batches * cols)).map {
            Float16(Float(($0 * 17 + 3) % 31 - 15) / 16)
        }
        // Generated with MLX v0.31.1 affine_qmm_t<half, 128, 2>, using the
        // fixtures above and biases=-scales. These bits specifically catch the
        // tempting but incorrect `scale * unpackedCode` dequantization: MLX
        // rounds scale/{4,16,64} independently for byte lanes 1...3.
        let mlxExpected: [UInt16] = [
            0x18a2, 0x28ac, 0xa7c2, 0x80e8, 0x2a5a, 0xa304, 0x9751, 0x2b36,
            0x25e8, 0xa6e3, 0xa70c, 0x2884, 0xa85a, 0x9f84, 0x2a2d, 0x24d1,
            0x9d80, 0xa212, 0x2530, 0x9d3d, 0xa463, 0x1e55, 0x9d8c, 0xa968,
            0x203d, 0xa659, 0xa5a1, 0x230c, 0xa80d, 0x1d80, 0x24f6, 0x2141,
            0xa198, 0xa0fe, 0x2c13, 0xa490, 0xa390, 0x20f0, 0xa663, 0xaa81,
            0xa304, 0x29f2, 0x234d, 0xa631, 0x2bc7, 0x9fc5, 0xa872, 0x241c,
            0x1f4a, 0x203b, 0xa72b, 0x21b0, 0x21fc, 0xa04d, 0x23d6, 0x2606,
            0x1c72, 0x2328, 0x2278, 0x1cde, 0x24a2, 0xa1c9, 0x1dac, 0x0d74,
            0x2412, 0xa631, 0xac51, 0x2800, 0xa7ee, 0x99d9, 0x29eb, 0x2001,
            0x9abe, 0xa998, 0x267e, 0x8db2, 0xab91, 0x2491, 0x17d2, 0xabbc,
            0xa2c7, 0x2263, 0x255e, 0xa5ec, 0x2434, 0x1ff6, 0xa83c, 0x1e98,
            0xa087, 0x24a8, 0x243e, 0xa2b0, 0x25d8, 0x98b1, 0xa483, 0xa44b,
            0x98d7, 0x2919, 0x223c, 0xa16f, 0x2ad4, 0xa254, 0xa4a0, 0x297e,
            0x20b6, 0x16bc, 0xac59, 0x2411, 0x1c5e, 0x9c02, 0x25cb, 0x2a74,
        ]

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let pipeline = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant_precise.metal",
            functionName: "signed_ternary_affine_qmm_g128_bm32_bn32_bk32"
        ))
        let codeBuffer = try makeSharedBuffer(device: device, codes)
        let scaleBuffer = try makeSharedBuffer(device: device, scales)
        let inputBuffer = try makeSharedBuffer(device: device, input)
        let output = try makeSharedBuffer(
            device: device, count: rows * batches, of: Float16.self)
        try runOnGPU(queue: queue) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(codeBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 2)
            encoder.setBuffer(inputBuffer, offset: 0, index: 3)
            encoder.setBuffer(output, offset: 0, index: 4)
            var rowCount = UInt32(rows)
            var colCount = UInt32(cols)
            var batchCount = UInt32(batches)
            encoder.setBytes(&rowCount, length: 4, index: 5)
            encoder.setBytes(&colCount, length: 4, index: 6)
            encoder.setBytes(&batchCount, length: 4, index: 7)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
            )
        }
        let got = UnsafeBufferPointer(
            start: output.contents().assumingMemoryBound(to: Float16.self),
            count: rows * batches
        ).map(\.bitPattern)
        XCTAssertEqual(got, mlxExpected)
    }

    func testBatchTiledSignedMatvecsAreBitExactWithLegacyBatchPath() throws {
        let rows = 9
        let cols = 256
        let batches = 5
        let groups = cols / 128
        let scales = (0..<(rows * groups)).map {
            Float16(0.01171875 * Float(($0 % 7) + 1))
        }
        let input = (0..<(batches * cols)).map {
            Float16(sin(Float($0 * 17 + 3)) * 0.75)
        }
        let cases: [(legacy: String, batched: String, bits: Int)] = [
            (
                "signed_binary_matvec_g128_rows8",
                "signed_binary_matvec_g128_rows8_batched_b4",
                1
            ),
            (
                "signed_ternary_matvec_g128_rows8",
                "signed_ternary_matvec_g128_rows8_batched_b4",
                2
            ),
        ]

        for item in cases {
            let semanticCode: (Int, Int) -> UInt8 = { row, col in
                item.bits == 1
                    ? UInt8((row * 13 + col * 7) & 1)
                    : UInt8((row * 13 + col * 7) % 3)
            }
            let codes = item.bits == 1
                ? packedCodes(rows: rows, cols: cols, bits: 1, code: semanticCode)
                : nativeTernaryCodes(rows: rows, cols: cols, code: semanticCode)
            let legacy = try run(
                function: item.legacy,
                codes: codes,
                scales: scales,
                input: input,
                rows: rows,
                cols: cols,
                batches: batches,
                rowTile: item.bits == 2 ? 2 : 8
            )
            let batched = try run(
                function: item.batched,
                codes: codes,
                scales: scales,
                input: input,
                rows: rows,
                cols: cols,
                batches: batches,
                batchTile: 4
            )
            XCTAssertEqual(
                batched.map(\.bitPattern),
                legacy.map(\.bitPattern),
                item.batched
            )
        }
    }

    func testBatchedBinaryBitplaneGEMMIsBitExactWithIndependentRows() throws {
        let rows = 9
        let cols = 256
        let batches = 5
        let groups = cols / 128
        let codes = packedCodes(rows: rows, cols: cols, bits: 1) {
            UInt8(($0 * 13 + $1 * 7) & 1)
        }
        let weightScales = (0..<(rows * groups)).map {
            Float16(0.001953125 * Float(($0 % 7) + 1))
        }
        let input = (0..<(batches * cols)).map {
            Float16(sin(Float($0 * 17 + 3)) * 0.75)
        }

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let codeBuffer = try makeSharedBuffer(device: device, codes)
        let weightScaleBuffer = try makeSharedBuffer(device: device, weightScales)
        let inputBuffer = try makeSharedBuffer(device: device, input)

        for bitCount in 2...6 {
            let singleBuilder = try XCTUnwrap(makeComputePipeline(
                device: device,
                shaderFile: "signed_quant.metal",
                functionName: "signed_activation_bitplanes_i\(bitCount)_g128"
            ))
            let batchedBuilder = try XCTUnwrap(makeComputePipeline(
                device: device,
                shaderFile: "signed_quant.metal",
                functionName: "signed_activation_bitplanes_i\(bitCount)_g128_batched"
            ))
            let singleConsumer = try XCTUnwrap(makeComputePipeline(
                device: device,
                shaderFile: "signed_quant.metal",
                functionName: "signed_binary_bitplane_i\(bitCount)_matvec_g128_rows8"
            ))
            let variants = [("b4", 4, 64), ("b8", 8, 128), ("b16", 16, 256)]
            let batchedConsumers = try variants.map { variant in
                try XCTUnwrap(makeComputePipeline(
                    device: device,
                    shaderFile: "signed_quant.metal",
                    functionName: "signed_binary_bitplane_i\(bitCount)_matvec_g128_rows8_batched_\(variant.0)"
                ))
            }
            let planeWordsPerBatch = groups * bitCount * 4
            let singlePlanes = try makeSharedBuffer(
                device: device,
                count: batches * planeWordsPerBatch,
                of: UInt32.self
            )
            let batchedPlanes = try makeSharedBuffer(
                device: device,
                count: batches * planeWordsPerBatch,
                of: UInt32.self
            )
            let singleScales = try makeSharedBuffer(
                device: device,
                count: batches * groups,
                of: Float16.self
            )
            let batchedScales = try makeSharedBuffer(
                device: device,
                count: batches * groups,
                of: Float16.self
            )
            let singleOutput = try makeSharedBuffer(
                device: device, count: batches * rows, of: Float16.self)
            let batchedOutputs = try variants.map { _ in
                try makeSharedBuffer(
                    device: device, count: batches * rows, of: Float16.self)
            }

            try runOnGPU(queue: queue) { encoder in
                for batch in 0..<batches {
                    encoder.setComputePipelineState(singleBuilder)
                    encoder.setBuffer(
                        inputBuffer, offset: batch * cols * 2, index: 0)
                    encoder.setBuffer(
                        singlePlanes,
                        offset: batch * planeWordsPerBatch * 4,
                        index: 1
                    )
                    encoder.setBuffer(
                        singleScales, offset: batch * groups * 2, index: 2)
                    var builderCols = UInt32(cols)
                    encoder.setBytes(&builderCols, length: 4, index: 3)
                    encoder.dispatchThreadgroups(
                        MTLSize(width: groups, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(
                            width: 32, height: 1, depth: 1)
                    )

                    encoder.setComputePipelineState(singleConsumer)
                    encoder.setBuffer(codeBuffer, offset: 0, index: 0)
                    encoder.setBuffer(weightScaleBuffer, offset: 0, index: 1)
                    encoder.setBuffer(
                        singlePlanes,
                        offset: batch * planeWordsPerBatch * 4,
                        index: 2
                    )
                    encoder.setBuffer(
                        singleScales, offset: batch * groups * 2, index: 3)
                    encoder.setBuffer(
                        singleOutput, offset: batch * rows * 2, index: 4)
                    var singleRows = UInt32(rows)
                    var singleCols = UInt32(cols)
                    encoder.setBytes(&singleRows, length: 4, index: 5)
                    encoder.setBytes(&singleCols, length: 4, index: 6)
                    encoder.dispatchThreadgroups(
                        MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(
                            width: 64, height: 1, depth: 1)
                    )
                }

                encoder.setComputePipelineState(batchedBuilder)
                encoder.setBuffer(inputBuffer, offset: 0, index: 0)
                encoder.setBuffer(batchedPlanes, offset: 0, index: 1)
                encoder.setBuffer(batchedScales, offset: 0, index: 2)
                var batchedBuilderCols = UInt32(cols)
                encoder.setBytes(&batchedBuilderCols, length: 4, index: 3)
                encoder.dispatchThreadgroups(
                    MTLSize(width: groups, height: batches, depth: 1),
                    threadsPerThreadgroup: MTLSize(
                        width: 32, height: 1, depth: 1)
                )

                for (index, variant) in variants.enumerated() {
                    encoder.setComputePipelineState(batchedConsumers[index])
                    encoder.setBuffer(codeBuffer, offset: 0, index: 0)
                    encoder.setBuffer(weightScaleBuffer, offset: 0, index: 1)
                    encoder.setBuffer(batchedPlanes, offset: 0, index: 2)
                    encoder.setBuffer(batchedScales, offset: 0, index: 3)
                    encoder.setBuffer(batchedOutputs[index], offset: 0, index: 4)
                    var batchedRows = UInt32(rows)
                    var batchedCols = UInt32(cols)
                    var actualBatch = UInt32(batches)
                    encoder.setBytes(&batchedRows, length: 4, index: 5)
                    encoder.setBytes(&batchedCols, length: 4, index: 6)
                    encoder.setBytes(&actualBatch, length: 4, index: 7)
                    encoder.dispatchThreadgroups(
                        MTLSize(
                            width: (rows + 7) / 8,
                            height: (batches + variant.1 - 1) / variant.1,
                            depth: 1
                        ),
                        threadsPerThreadgroup: MTLSize(
                            width: variant.2, height: 1, depth: 1)
                    )
                }
            }

            let singlePlaneValues = UnsafeBufferPointer(
                start: singlePlanes.contents().assumingMemoryBound(to: UInt32.self),
                count: batches * planeWordsPerBatch
            )
            let batchedPlaneValues = UnsafeBufferPointer(
                start: batchedPlanes.contents().assumingMemoryBound(to: UInt32.self),
                count: batches * planeWordsPerBatch
            )
            XCTAssertEqual(
                Array(batchedPlaneValues), Array(singlePlaneValues),
                "i\(bitCount) producer"
            )
            let singleScaleValues = UnsafeBufferPointer(
                start: singleScales.contents().assumingMemoryBound(to: Float16.self),
                count: batches * groups
            )
            let batchedScaleValues = UnsafeBufferPointer(
                start: batchedScales.contents().assumingMemoryBound(to: Float16.self),
                count: batches * groups
            )
            XCTAssertEqual(
                batchedScaleValues.map(\.bitPattern),
                singleScaleValues.map(\.bitPattern),
                "i\(bitCount) scales"
            )
            let singleOutputValues = UnsafeBufferPointer(
                start: singleOutput.contents().assumingMemoryBound(to: Float16.self),
                count: batches * rows
            )
            for (index, variant) in variants.enumerated() {
                let batchedOutputValues = UnsafeBufferPointer(
                    start: batchedOutputs[index].contents()
                        .assumingMemoryBound(to: Float16.self),
                    count: batches * rows
                )
                XCTAssertEqual(
                    batchedOutputValues.map(\.bitPattern),
                    singleOutputValues.map(\.bitPattern),
                    "i\(bitCount) consumer \(variant.0)"
                )
            }
        }
    }

    func testBatchedBinaryBitplaneGEMMCanProfileProductionShape() throws {
        guard ProcessInfo.processInfo.environment["SMELT_SIGNED_BITGEMM_PROFILE"] == "1"
        else {
            throw XCTSkip("set SMELT_SIGNED_BITGEMM_PROFILE=1")
        }
        let rows = Int(
            ProcessInfo.processInfo.environment["SMELT_SIGNED_BITGEMM_ROWS"]
                ?? "17408")!
        let cols = Int(
            ProcessInfo.processInfo.environment["SMELT_SIGNED_BITGEMM_COLS"]
                ?? "5120")!
        let batches = Int(
            ProcessInfo.processInfo.environment["SMELT_SIGNED_BITGEMM_BATCH"]
                ?? "256")!
        let bitCount = Int(
            ProcessInfo.processInfo.environment["SMELT_SIGNED_BITGEMM_BITS"]
                ?? "4")!
        XCTAssertTrue(cols.isMultiple(of: 128))
        XCTAssertTrue((2...6).contains(bitCount))
        let groups = cols / 128
        let codes = [UInt8](repeating: 0x5a, count: rows * cols / 8)
        let weightScales = (0..<(rows * groups)).map {
            Float16(0.001953125 * Float(($0 % 7) + 1))
        }
        let input = (0..<(batches * cols)).map {
            Float16(sin(Float($0 * 17 + 3)) * 0.75)
        }

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let builder = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant.metal",
            functionName: "signed_activation_bitplanes_i\(bitCount)_g128_batched"
        ))
        let variants = [("b4", 4, 64), ("b8", 8, 128), ("b16", 16, 256)]
        let consumers = try variants.map { variant in
            try XCTUnwrap(makeComputePipeline(
                device: device,
                shaderFile: "signed_quant.metal",
                functionName: "signed_binary_bitplane_i\(bitCount)_matvec_g128_rows8_batched_\(variant.0)"
            ))
        }
        let codeBuffer = try makeSharedBuffer(device: device, codes)
        let weightScaleBuffer = try makeSharedBuffer(device: device, weightScales)
        let inputBuffer = try makeSharedBuffer(device: device, input)
        let planes = try makeSharedBuffer(
            device: device,
            count: batches * groups * bitCount * 4,
            of: UInt32.self
        )
        let activationScales = try makeSharedBuffer(
            device: device, count: batches * groups, of: Float16.self)
        let output = try makeSharedBuffer(
            device: device, count: batches * rows, of: Float16.self)

        func encodeBuilder(_ encoder: MTLComputeCommandEncoder) {
            encoder.setComputePipelineState(builder)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(planes, offset: 0, index: 1)
            encoder.setBuffer(activationScales, offset: 0, index: 2)
            var colCount = UInt32(cols)
            encoder.setBytes(&colCount, length: 4, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: groups, height: batches, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        }
        func encodeConsumer(
            _ encoder: MTLComputeCommandEncoder,
            pipeline: MTLComputePipelineState,
            variant: (String, Int, Int)
        ) {
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(codeBuffer, offset: 0, index: 0)
            encoder.setBuffer(weightScaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(planes, offset: 0, index: 2)
            encoder.setBuffer(activationScales, offset: 0, index: 3)
            encoder.setBuffer(output, offset: 0, index: 4)
            var rowCount = UInt32(rows)
            var colCount = UInt32(cols)
            var batchCount = UInt32(batches)
            encoder.setBytes(&rowCount, length: 4, index: 5)
            encoder.setBytes(&colCount, length: 4, index: 6)
            encoder.setBytes(&batchCount, length: 4, index: 7)
            encoder.dispatchThreadgroups(
                MTLSize(
                    width: (rows + 7) / 8,
                    height: (batches + variant.1 - 1) / variant.1,
                    depth: 1
                ),
                threadsPerThreadgroup: MTLSize(
                    width: variant.2, height: 1, depth: 1)
            )
        }
        try runOnGPU(queue: queue) { encodeBuilder($0) }

        func profile(
            repetitions: Int,
            encode: (MTLComputeCommandEncoder) -> Void
        ) throws -> Double {
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
            for _ in 0..<repetitions { encode(encoder) }
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            return (command.gpuEndTime - command.gpuStartTime) * 1e6
                / Double(repetitions)
        }

        let repetitions = 5
        let builderUS = try profile(repetitions: repetitions, encode: encodeBuilder)
        var timings: [String] = []
        for (index, variant) in variants.enumerated() {
            let elapsed = try profile(repetitions: repetitions) {
                encodeConsumer($0, pipeline: consumers[index], variant: variant)
            }
            timings.append(String(format: "%@=%.1fus", variant.0, elapsed))
        }
        print(
            "signed bit-GEMM i\(bitCount) rows=\(rows) cols=\(cols) "
                + "batch=\(batches) builder="
                + String(format: "%.1fus ", builderUS)
                + timings.joined(separator: " ")
        )
    }

    func testBinaryAndTernaryG128MatvecMatchCPUAcrossBatchAndRowTail() throws {
        let rows = 5
        let cols = 256
        let batches = 5
        let groups = cols / 128
        let scales = (0..<(rows * groups)).map {
            Float16(0.03125 * Float(($0 % 5) + 1))
        }
        let input = (0..<(batches * cols)).map {
            Float16(sin(Float($0 * 17 + 3)) * 0.75)
        }

        let cases: [(String, Int, (Int, Int) -> UInt8, (UInt8) -> Float)] = [
            (
                "signed_binary_matvec_g128_rows8",
                1,
                { row, col in UInt8((row * 13 + col * 7) & 1) },
                { $0 == 0 ? -1 : 1 }
            ),
            (
                "signed_ternary_matvec_g128_rows8",
                2,
                { row, col in UInt8((row * 13 + col * 7) % 3) },
                { Float(Int($0) - 1) }
            ),
        ]

        for (function, bits, code, semantic) in cases {
            let codes = bits == 1
                ? packedCodes(rows: rows, cols: cols, bits: bits, code: code)
                : nativeTernaryCodes(rows: rows, cols: cols, code: code)
            let got = try run(
                function: function,
                codes: codes,
                scales: scales,
                input: input,
                rows: rows,
                cols: cols,
                batches: batches,
                rowTile: bits == 2 ? 2 : 8
            )
            for batch in 0..<batches {
                for row in 0..<rows {
                    var expected: Float = 0
                    for col in 0..<cols {
                        expected += Float(input[batch * cols + col])
                            * semantic(code(row, col))
                            * Float(scales[row * groups + col / 128])
                    }
                    XCTAssertEqual(
                        Float(got[batch * rows + row]),
                        expected,
                        accuracy: max(0.02, abs(expected) * 0.01),
                        "\(function) batch \(batch) row \(row)"
                    )
                }
            }
        }
    }

    func testPackedFourMemberProjectionBankIsBitExactAtEveryBoundary() throws {
        let memberRows = [5, 3, 7, 2]
        let cols = 256
        let groups = cols / 128
        let input = (0..<cols).map { Float16(sin(Float($0 * 17 + 3)) * 0.75) }
        var memberCodes: [[UInt8]] = []
        var memberScales: [[Float16]] = []
        var expected: [[Float16]] = []
        for (member, rows) in memberRows.enumerated() {
            let memberID = member
            let code: (Int, Int) -> UInt8 = { row, col in
                let value = memberID * 19 + row * 13 + col * 7
                return UInt8(value & 1)
            }
            let codes = packedCodes(
                rows: rows, cols: cols, bits: 1, code: code)
            let scales: [Float16] = (0..<(rows * groups)).map { group in
                Float16(0.00390625 * Float((member + group) % 7 + 1))
            }
            memberCodes.append(codes)
            memberScales.append(scales)
            expected.append(try run(
                function: "signed_binary_matvec_g128_rows8",
                codes: codes,
                scales: scales,
                input: input,
                rows: rows,
                cols: cols,
                batches: 1
            ))
        }

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let pipeline = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant.metal",
            functionName: "signed_binary_packed_bank4_matvec_g128_rows8"
        ))
        let allCodes: [UInt8] = memberCodes.flatMap { $0 }
        let allScales: [Float16] = memberScales.flatMap { $0 }
        let codeBuffer = try makeSharedBuffer(device: device, allCodes)
        let scaleBuffer = try makeSharedBuffer(device: device, allScales)
        let inputBuffer = try makeSharedBuffer(device: device, input)
        let outputs = try memberRows.map {
            try makeSharedBuffer(device: device, count: $0, of: Float16.self)
        }
        try runOnGPU(queue: queue) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(codeBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(inputBuffer, offset: 0, index: 2)
            for (index, output) in outputs.enumerated() {
                encoder.setBuffer(output, offset: 0, index: index + 3)
            }
            var rows = memberRows.map(UInt32.init)
            for index in rows.indices {
                encoder.setBytes(&rows[index], length: 4, index: index + 7)
            }
            var colCount = UInt32(cols)
            encoder.setBytes(&colCount, length: 4, index: 11)
            encoder.dispatchThreadgroups(
                MTLSize(width: (memberRows.reduce(0, +) + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }

        for index in outputs.indices {
            let got = UnsafeBufferPointer(
                start: outputs[index].contents().assumingMemoryBound(to: Float16.self),
                count: memberRows[index]
            )
            XCTAssertEqual(got.map(\.bitPattern), expected[index].map(\.bitPattern))
        }
    }

    func testBinaryGateUpSwiGLUMatchesStagedSemanticsAcrossBatchAndRowTail() throws {
        let rows = 5
        let cols = 256
        let batches = 5
        let groups = cols / 128
        let gateCode: (Int, Int) -> UInt8 = {
            UInt8(($0 * 11 + $1 * 5) & 1)
        }
        let upCode: (Int, Int) -> UInt8 = {
            UInt8(($0 * 7 + $1 * 3 + 1) & 1)
        }
        let gateCodes = packedCodes(
            rows: rows, cols: cols, bits: 1, code: gateCode)
        let upCodes = packedCodes(
            rows: rows, cols: cols, bits: 1, code: upCode)
        let gateScales = (0..<(rows * groups)).map {
            Float16(0.015625 * Float(($0 % 5) + 1))
        }
        let upScales = (0..<(rows * groups)).map {
            Float16(0.01171875 * Float(($0 % 7) + 1))
        }
        let input = (0..<(batches * cols)).map {
            Float16(sin(Float($0 * 19 + 7)) * 0.5)
        }

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let gateCodeBuffer = try makeSharedBuffer(device: device, gateCodes)
        let gateScaleBuffer = try makeSharedBuffer(device: device, gateScales)
        let upCodeBuffer = try makeSharedBuffer(device: device, upCodes)
        let upScaleBuffer = try makeSharedBuffer(device: device, upScales)
        let inputBuffer = try makeSharedBuffer(device: device, input)
        func execute(
            functionName: String,
            batchTile: Int,
            threadgroupWidth: Int = 64
        ) throws -> [Float16] {
            let pipeline = try XCTUnwrap(makeComputePipeline(
                device: device,
                shaderFile: "signed_quant.metal",
                functionName: functionName
            ))
            let output = try makeSharedBuffer(
                device: device, count: rows * batches, of: Float16.self)
            try runOnGPU(queue: queue) { encoder in
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(gateCodeBuffer, offset: 0, index: 0)
                encoder.setBuffer(gateScaleBuffer, offset: 0, index: 1)
                encoder.setBuffer(upCodeBuffer, offset: 0, index: 2)
                encoder.setBuffer(upScaleBuffer, offset: 0, index: 3)
                encoder.setBuffer(inputBuffer, offset: 0, index: 4)
                encoder.setBuffer(output, offset: 0, index: 5)
                var rowCount = UInt32(rows)
                var colCount = UInt32(cols)
                encoder.setBytes(&rowCount, length: 4, index: 6)
                encoder.setBytes(&colCount, length: 4, index: 7)
                if batchTile > 1 {
                    var actualBatch = UInt32(batches)
                    encoder.setBytes(&actualBatch, length: 4, index: 8)
                }
                encoder.dispatchThreadgroups(
                    MTLSize(
                        width: (rows + 7) / 8,
                        height: (batches + batchTile - 1) / batchTile,
                        depth: 1
                    ),
                    threadsPerThreadgroup: MTLSize(
                        width: threadgroupWidth, height: 1, depth: 1)
                )
            }
            return Array(UnsafeBufferPointer(
                start: output.contents().assumingMemoryBound(to: Float16.self),
                count: rows * batches
            ))
        }
        let got = try execute(
            functionName: "signed_binary_gate_up_swiglu_g128_rows8",
            batchTile: 1
        )
        let batchTiled = try execute(
            functionName: "signed_binary_gate_up_swiglu_g128_rows8_batched_b4",
            batchTile: 4
        )
        XCTAssertEqual(batchTiled.map(\.bitPattern), got.map(\.bitPattern))

        for batch in 0..<batches {
            for row in 0..<rows {
                var gate: Float = 0
                var up: Float = 0
                for col in 0..<cols {
                    let x = Float(input[batch * cols + col])
                    gate += x * (gateCode(row, col) == 0 ? -1 : 1)
                        * Float(gateScales[row * groups + col / 128])
                    up += x * (upCode(row, col) == 0 ? -1 : 1)
                        * Float(upScales[row * groups + col / 128])
                }
                gate = Float(Float16(gate))
                up = Float(Float16(up))
                let expected = Float(Float16(max(
                    -65_504,
                    min(65_504, gate / (1 + exp(-gate)) * up)
                )))
                XCTAssertEqual(
                    Float(got[batch * rows + row]),
                    expected,
                    accuracy: max(0.03, abs(expected) * 0.015),
                    "batch \(batch) row \(row)"
                )
            }
        }
    }

    func testBinaryLowBitGateUpSwiGLUIsExactWithStagedViewAndCanProfile() throws {
        let profiling = ProcessInfo.processInfo.environment[
            "SMELT_SIGNED_GATE_PROFILE"
        ] == "1"
        let environment = ProcessInfo.processInfo.environment
        let bitCount = Int(environment["SMELT_SIGNED_GATE_BITS"] ?? "3")!
        XCTAssertTrue([3, 4, 5, 6].contains(bitCount))
        let rows = profiling
            ? Int(environment["SMELT_SIGNED_GATE_ROWS"] ?? "17408")!
            : 13
        let cols = profiling
            ? Int(environment["SMELT_SIGNED_GATE_COLS"] ?? "5120")!
            : 256
        let groups = cols / 128
        let planeWords = groups * bitCount * 4
        let gateCodes = [UInt8](
            repeating: 0xa5, count: rows * cols / 8)
        let upCodes = [UInt8](
            repeating: 0x3c, count: rows * cols / 8)
        let gateScales = (0..<(rows * groups)).map {
            Float16(0.001953125 * Float(($0 % 7) + 1))
        }
        let upScales = (0..<(rows * groups)).map {
            Float16(0.0029296875 * Float(($0 % 5) + 1))
        }
        let input = (0..<cols).map {
            Float16(sin(Float($0 * 17 + 3)) * (0.25 + Float($0 % 11) * 0.125))
        }

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let builder = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant.metal",
            functionName: "signed_activation_bitplanes_i\(bitCount)_g128"
        ))
        let stagedMatvec = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant.metal",
            functionName: "signed_binary_bitplane_i\(bitCount)_matvec_g128_rows8"
        ))
        let swiglu = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "activations_precise.metal",
            functionName: "swiglu_fused"
        ))
        let fused = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant.metal",
            functionName: "signed_binary_bitplane_i\(bitCount)_gate_up_swiglu_g128_rows8"
        ))
        let direct = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant.metal",
            functionName: "signed_binary_gate_up_swiglu_g128_rows8"
        ))

        let gateCodeBuffer = try makeSharedBuffer(device: device, gateCodes)
        let gateScaleBuffer = try makeSharedBuffer(device: device, gateScales)
        let upCodeBuffer = try makeSharedBuffer(device: device, upCodes)
        let upScaleBuffer = try makeSharedBuffer(device: device, upScales)
        let inputBuffer = try makeSharedBuffer(device: device, input)
        let planes = try makeSharedBuffer(
            device: device, count: planeWords, of: UInt32.self)
        let activationScales = try makeSharedBuffer(
            device: device, count: groups, of: Float16.self)
        let stagedGate = try makeSharedBuffer(
            device: device, count: rows, of: Float16.self)
        let stagedUp = try makeSharedBuffer(
            device: device, count: rows, of: Float16.self)
        let stagedOutput = try makeSharedBuffer(
            device: device, count: rows, of: Float16.self)
        let fusedOutput = try makeSharedBuffer(
            device: device, count: rows, of: Float16.self)
        let directOutput = try makeSharedBuffer(
            device: device, count: rows, of: Float16.self)

        func encodeBuilder(_ encoder: MTLComputeCommandEncoder) {
            encoder.setComputePipelineState(builder)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(planes, offset: 0, index: 1)
            encoder.setBuffer(activationScales, offset: 0, index: 2)
            var colCount = UInt32(cols)
            encoder.setBytes(&colCount, length: 4, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: groups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        }
        func encodeStagedMatvec(
            _ encoder: MTLComputeCommandEncoder,
            codes: MTLBuffer,
            scales: MTLBuffer,
            output: MTLBuffer
        ) {
            encoder.setComputePipelineState(stagedMatvec)
            encoder.setBuffer(codes, offset: 0, index: 0)
            encoder.setBuffer(scales, offset: 0, index: 1)
            encoder.setBuffer(planes, offset: 0, index: 2)
            encoder.setBuffer(activationScales, offset: 0, index: 3)
            encoder.setBuffer(output, offset: 0, index: 4)
            var rowCount = UInt32(rows)
            var colCount = UInt32(cols)
            encoder.setBytes(&rowCount, length: 4, index: 5)
            encoder.setBytes(&colCount, length: 4, index: 6)
            encoder.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }
        func encodeFused(_ encoder: MTLComputeCommandEncoder) {
            encoder.setComputePipelineState(fused)
            encoder.setBuffer(gateCodeBuffer, offset: 0, index: 0)
            encoder.setBuffer(gateScaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(upCodeBuffer, offset: 0, index: 2)
            encoder.setBuffer(upScaleBuffer, offset: 0, index: 3)
            encoder.setBuffer(planes, offset: 0, index: 4)
            encoder.setBuffer(activationScales, offset: 0, index: 5)
            encoder.setBuffer(fusedOutput, offset: 0, index: 6)
            var rowCount = UInt32(rows)
            var colCount = UInt32(cols)
            encoder.setBytes(&rowCount, length: 4, index: 7)
            encoder.setBytes(&colCount, length: 4, index: 8)
            encoder.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }
        func encodeDirect(_ encoder: MTLComputeCommandEncoder) {
            encoder.setComputePipelineState(direct)
            encoder.setBuffer(gateCodeBuffer, offset: 0, index: 0)
            encoder.setBuffer(gateScaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(upCodeBuffer, offset: 0, index: 2)
            encoder.setBuffer(upScaleBuffer, offset: 0, index: 3)
            encoder.setBuffer(inputBuffer, offset: 0, index: 4)
            encoder.setBuffer(directOutput, offset: 0, index: 5)
            var rowCount = UInt32(rows)
            var colCount = UInt32(cols)
            encoder.setBytes(&rowCount, length: 4, index: 6)
            encoder.setBytes(&colCount, length: 4, index: 7)
            encoder.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }

        try runOnGPU(queue: queue) { encoder in
            encodeBuilder(encoder)
            encodeStagedMatvec(
                encoder,
                codes: gateCodeBuffer,
                scales: gateScaleBuffer,
                output: stagedGate
            )
            encodeStagedMatvec(
                encoder,
                codes: upCodeBuffer,
                scales: upScaleBuffer,
                output: stagedUp
            )
            encoder.setComputePipelineState(swiglu)
            encoder.setBuffer(stagedGate, offset: 0, index: 0)
            encoder.setBuffer(stagedUp, offset: 0, index: 1)
            encoder.setBuffer(stagedOutput, offset: 0, index: 2)
            var rowCount = UInt32(rows)
            encoder.setBytes(&rowCount, length: 4, index: 3)
            encoder.dispatchThreads(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(rows, 1024), height: 1, depth: 1)
            )
            encodeFused(encoder)
        }
        let stagedValues = UnsafeBufferPointer(
            start: stagedOutput.contents().assumingMemoryBound(to: Float16.self),
            count: rows
        )
        let fusedValues = UnsafeBufferPointer(
            start: fusedOutput.contents().assumingMemoryBound(to: Float16.self),
            count: rows
        )
        XCTAssertEqual(fusedValues.map(\.bitPattern), stagedValues.map(\.bitPattern))

        guard profiling else { return }
        func profile(
            encode: (MTLComputeCommandEncoder) -> Void
        ) throws -> Double {
            let repetitions = 100
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
            for _ in 0..<repetitions { encode(encoder) }
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            if let error = command.error { throw error }
            return (command.gpuEndTime - command.gpuStartTime)
                * 1e6 / Double(repetitions)
        }
        var directSamples: [Double] = []
        var bitplaneSamples: [Double] = []
        for round in 0..<7 {
            let bitplaneEncode: (MTLComputeCommandEncoder) -> Void = { encoder in
                encodeBuilder(encoder)
                encodeFused(encoder)
            }
            if round.isMultiple(of: 2) {
                directSamples.append(try profile(encode: encodeDirect))
                bitplaneSamples.append(try profile(encode: bitplaneEncode))
            } else {
                bitplaneSamples.append(try profile(encode: bitplaneEncode))
                directSamples.append(try profile(encode: encodeDirect))
            }
        }
        let directUS = directSamples.sorted()[directSamples.count / 2]
        let bitplaneUS = bitplaneSamples.sorted()[bitplaneSamples.count / 2]
        print(String(
            format: "signed gate/up i%d rows=%d cols=%d direct_med=%.2fus builder+bitplane_med=%.2fus speedup=%.3fx",
            bitCount, rows, cols, directUS, bitplaneUS, directUS / bitplaneUS
        ))
    }

    func testBinaryMatvecAddIsBitExactWithStagedMatvecThenResidual() throws {
        let rows = 5
        let cols = 256
        let batches = 2
        let groups = cols / 128
        let codes = packedCodes(rows: rows, cols: cols, bits: 1) {
            UInt8(($0 * 13 + $1 * 7) & 1)
        }
        let scales = (0..<(rows * groups)).map {
            Float16(0.01953125 * Float(($0 % 5) + 1))
        }
        let input = (0..<(batches * cols)).map {
            Float16(cos(Float($0 * 11 + 5)) * 0.5)
        }
        let residual = (0..<(batches * rows)).map {
            Float16(sin(Float($0 * 23 + 2)) * 0.75)
        }
        let staged = try run(
            function: "signed_binary_matvec_g128_rows8",
            codes: codes,
            scales: scales,
            input: input,
            rows: rows,
            cols: cols,
            batches: batches
        )

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let pipeline = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant.metal",
            functionName: "signed_binary_matvec_add_g128_rows8"
        ))
        let output = try makeSharedBuffer(
            device: device, count: rows * batches, of: Float16.self)
        let codeBuffer = try makeSharedBuffer(device: device, codes)
        let scaleBuffer = try makeSharedBuffer(device: device, scales)
        let inputBuffer = try makeSharedBuffer(device: device, input)
        let residualBuffer = try makeSharedBuffer(device: device, residual)
        try runOnGPU(queue: queue) { encoder in
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(codeBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(inputBuffer, offset: 0, index: 2)
            encoder.setBuffer(output, offset: 0, index: 3)
            encoder.setBuffer(residualBuffer, offset: 0, index: 4)
            var rowCount = UInt32(rows)
            var colCount = UInt32(cols)
            encoder.setBytes(&rowCount, length: 4, index: 5)
            encoder.setBytes(&colCount, length: 4, index: 6)
            encoder.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: batches, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }
        let got = Array(UnsafeBufferPointer(
            start: output.contents().assumingMemoryBound(to: Float16.self),
            count: rows * batches
        ))
        let expected = zip(staged, residual).map {
            Float16(Float($0.0) + Float($0.1))
        }
        XCTAssertEqual(got.map(\.bitPattern), expected.map(\.bitPattern))
    }

    func testBinaryBitplaneActivationViewMatchesQuantizedCPUAndCanProfile() throws {
        let profiling = ProcessInfo.processInfo.environment["SMELT_SIGNED_BITPLANE_PROFILE"] == "1"
        let bitCount = Int(
            ProcessInfo.processInfo.environment["SMELT_SIGNED_BITPLANE_BITS"] ?? "4"
        )!
        XCTAssertTrue((2...6).contains(bitCount))
        let maxQuantized = bitCount == 2 ? 1 : (1 << (bitCount - 1)) - 1
        let rows = profiling
            ? Int(ProcessInfo.processInfo.environment["SMELT_SIGNED_BITPLANE_ROWS"] ?? "17408")!
            : 9
        let cols = profiling
            ? Int(ProcessInfo.processInfo.environment["SMELT_SIGNED_BITPLANE_COLS"] ?? "5120")!
            : 256
        let groups = cols / 128
        let codes = packedCodes(rows: rows, cols: cols, bits: 1) {
            UInt8(($0 * 13 + $1 * 7) & 1)
        }
        let scales = (0..<(rows * groups)).map {
            Float16(0.001953125 * Float(($0 % 7) + 1))
        }
        let input = (0..<cols).map {
            Float16(sin(Float($0 * 17 + 3)) * 0.75)
        }

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let builder = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant.metal",
            functionName: "signed_activation_bitplanes_i\(bitCount)_g128"
        ))
        let bitplaneMatvec = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant.metal",
            functionName: "signed_binary_bitplane_i\(bitCount)_matvec_g128_rows8"
        ))
        let directMatvec = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant.metal",
            functionName: "signed_binary_matvec_g128_rows8"
        ))
        let codeBuffer = try makeSharedBuffer(device: device, codes)
        let scaleBuffer = try makeSharedBuffer(device: device, scales)
        let inputBuffer = try makeSharedBuffer(device: device, input)
        let planes = try makeSharedBuffer(
            device: device, count: groups * bitCount * 4, of: UInt32.self)
        let activationScales = try makeSharedBuffer(
            device: device, count: groups, of: Float16.self)
        let bitplaneOutput = try makeSharedBuffer(
            device: device, count: rows, of: Float16.self)
        let directOutput = try makeSharedBuffer(
            device: device, count: rows, of: Float16.self)
        func encodeBuilder(_ encoder: MTLComputeCommandEncoder) {
            encoder.setComputePipelineState(builder)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(planes, offset: 0, index: 1)
            encoder.setBuffer(activationScales, offset: 0, index: 2)
            var colCount = UInt32(cols)
            encoder.setBytes(&colCount, length: 4, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: groups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        }
        func encodeBitplane(_ encoder: MTLComputeCommandEncoder) {
            encoder.setComputePipelineState(bitplaneMatvec)
            encoder.setBuffer(codeBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(planes, offset: 0, index: 2)
            encoder.setBuffer(activationScales, offset: 0, index: 3)
            encoder.setBuffer(bitplaneOutput, offset: 0, index: 4)
            var rowCount = UInt32(rows)
            var colCount = UInt32(cols)
            encoder.setBytes(&rowCount, length: 4, index: 5)
            encoder.setBytes(&colCount, length: 4, index: 6)
            encoder.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }
        func encodeDirect(_ encoder: MTLComputeCommandEncoder) {
            encoder.setComputePipelineState(directMatvec)
            encoder.setBuffer(codeBuffer, offset: 0, index: 0)
            encoder.setBuffer(scaleBuffer, offset: 0, index: 1)
            encoder.setBuffer(inputBuffer, offset: 0, index: 2)
            encoder.setBuffer(directOutput, offset: 0, index: 3)
            var rowCount = UInt32(rows)
            var colCount = UInt32(cols)
            encoder.setBytes(&rowCount, length: 4, index: 4)
            encoder.setBytes(&colCount, length: 4, index: 5)
            encoder.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }
        try runOnGPU(queue: queue) { encoder in
            encodeBuilder(encoder)
            encodeBitplane(encoder)
            encodeDirect(encoder)
        }

        let got = UnsafeBufferPointer(
            start: bitplaneOutput.contents().assumingMemoryBound(to: Float16.self),
            count: rows
        )
        let activationScaleValues = UnsafeBufferPointer(
            start: activationScales.contents().assumingMemoryBound(to: Float16.self),
            count: groups
        )
        let validationRows = profiling
            ? Array(Set([0, 1, 7, 8, rows / 2, rows - 1])).sorted()
            : Array(0..<rows)
        for row in validationRows {
            var expected: Float = 0
            for col in 0..<cols {
                let activationScale = Float(activationScaleValues[col / 128])
                let q = Int(round(max(
                    -Float(maxQuantized),
                    min(Float(maxQuantized), Float(input[col]) / activationScale)
                )))
                let sign: Float = ((row * 13 + col * 7) & 1) == 0 ? -1 : 1
                expected += Float(q) * activationScale * sign
                    * Float(scales[row * groups + col / 128])
            }
            XCTAssertEqual(
                Float(got[row]), expected,
                accuracy: max(0.02, abs(expected) * 0.01),
                "row \(row)"
            )
        }
        if !profiling {
            return
        }

        func profile(repetitions: Int, encode: (MTLComputeCommandEncoder) -> Void) throws -> Double {
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
            for _ in 0..<repetitions { encode(encoder) }
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            return (command.gpuEndTime - command.gpuStartTime) * 1e6 / Double(repetitions)
        }
        let repetitions = 100
        let buildUS = try profile(repetitions: repetitions, encode: encodeBuilder)
        let bitplaneUS = try profile(repetitions: repetitions, encode: encodeBitplane)
        let directUS = try profile(repetitions: repetitions, encode: encodeDirect)
        print(String(format: "signed bitplane i%d g128 rows=%d cols=%d build=%.2fus matvec=%.2fus total=%.2fus direct=%.2fus speedup=%.3fx", bitCount, rows, cols, buildUS, bitplaneUS, buildUS + bitplaneUS, directUS, directUS / (buildUS + bitplaneUS)))
    }

    func testTernaryBitplaneMatvecsMatchQuantizedCPUAcrossRowTail() throws {
        let rows = 9
        let cols = 256
        let groups = cols / 128
        let code: (Int, Int) -> UInt8 = {
            UInt8(($0 * 13 + $1 * 7) % 3)
        }
        let codes = nativeTernaryCodes(
            rows: rows, cols: cols, code: code)
        let weightScales = (0..<(rows * groups)).map {
            Float16(0.00390625 * Float(($0 % 7) + 1))
        }
        let input = (0..<cols).map {
            Float16(sin(Float($0 * 17 + 3)) * 0.75)
        }
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let codeBuffer = try makeSharedBuffer(device: device, codes)
        let weightScaleBuffer = try makeSharedBuffer(device: device, weightScales)
        let inputBuffer = try makeSharedBuffer(device: device, input)

        for bitCount in [4, 5, 6] {
            let maxQuantized = (1 << (bitCount - 1)) - 1
            let builder = try XCTUnwrap(makeComputePipeline(
                device: device,
                shaderFile: "signed_quant.metal",
                functionName: "signed_activation_bitplanes_i\(bitCount)_g128"
            ))
            let matvec = try XCTUnwrap(makeComputePipeline(
                device: device,
                shaderFile: "signed_quant.metal",
                functionName: "signed_ternary_bitplane_i\(bitCount)_matvec_g128_rows8"
            ))
            let wideMatvec = try XCTUnwrap(makeComputePipeline(
                device: device,
                shaderFile: "signed_quant.metal",
                functionName: "signed_ternary_bitplane_i\(bitCount)_matvec_g128_rows2_wide"
            ))
            let planes = try makeSharedBuffer(
                device: device,
                count: groups * bitCount * 4,
                of: UInt32.self
            )
            let activationScales = try makeSharedBuffer(
                device: device, count: groups, of: Float16.self)
            let output = try makeSharedBuffer(
                device: device, count: rows, of: Float16.self)
            let wideOutput = try makeSharedBuffer(
                device: device, count: rows, of: Float16.self)

            try runOnGPU(queue: queue) { encoder in
                encoder.setComputePipelineState(builder)
                encoder.setBuffer(inputBuffer, offset: 0, index: 0)
                encoder.setBuffer(planes, offset: 0, index: 1)
                encoder.setBuffer(activationScales, offset: 0, index: 2)
                var builderCols = UInt32(cols)
                encoder.setBytes(&builderCols, length: 4, index: 3)
                encoder.dispatchThreadgroups(
                    MTLSize(width: groups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )

                encoder.setComputePipelineState(matvec)
                encoder.setBuffer(codeBuffer, offset: 0, index: 0)
                encoder.setBuffer(weightScaleBuffer, offset: 0, index: 1)
                encoder.setBuffer(planes, offset: 0, index: 2)
                encoder.setBuffer(activationScales, offset: 0, index: 3)
                encoder.setBuffer(output, offset: 0, index: 4)
                var rowCount = UInt32(rows)
                var matvecCols = UInt32(cols)
                encoder.setBytes(&rowCount, length: 4, index: 5)
                encoder.setBytes(&matvecCols, length: 4, index: 6)
                encoder.dispatchThreadgroups(
                    MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
                )

                encoder.setComputePipelineState(wideMatvec)
                encoder.setBuffer(codeBuffer, offset: 0, index: 0)
                encoder.setBuffer(weightScaleBuffer, offset: 0, index: 1)
                encoder.setBuffer(planes, offset: 0, index: 2)
                encoder.setBuffer(activationScales, offset: 0, index: 3)
                encoder.setBuffer(wideOutput, offset: 0, index: 4)
                var wideRows = UInt32(rows)
                var wideCols = UInt32(cols)
                encoder.setBytes(&wideRows, length: 4, index: 5)
                encoder.setBytes(&wideCols, length: 4, index: 6)
                encoder.dispatchThreadgroups(
                    MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
                )
            }

            let scaleValues = UnsafeBufferPointer(
                start: activationScales.contents().assumingMemoryBound(to: Float16.self),
                count: groups
            )
            let outputValues = UnsafeBufferPointer(
                start: output.contents().assumingMemoryBound(to: Float16.self),
                count: rows
            )
            let wideOutputValues = UnsafeBufferPointer(
                start: wideOutput.contents().assumingMemoryBound(to: Float16.self),
                count: rows
            )
            XCTAssertEqual(
                wideOutputValues.map(\.bitPattern),
                outputValues.map(\.bitPattern),
                "i\(bitCount) wide-input geometry must preserve accumulation order"
            )
            for row in 0..<rows {
                var expected: Float = 0
                for col in 0..<cols {
                    let activationScale = Float(scaleValues[col / 128])
                    let q = Int(round(max(
                        -Float(maxQuantized),
                        min(Float(maxQuantized), Float(input[col]) / activationScale)
                    )))
                    expected += Float(q) * activationScale
                        * Float(Int(code(row, col)) - 1)
                        * Float(weightScales[row * groups + col / 128])
                }
                XCTAssertEqual(
                    Float(outputValues[row]), expected,
                    accuracy: max(0.02, abs(expected) * 0.01),
                    "i\(bitCount) row \(row)"
                )
            }
        }
    }

    func testTernaryProjectionBankIsBitExactWithIndependentConsumers() throws {
        let rowCounts = [3, 5, 4, 7]
        let cols = 256
        let groups = cols / 128
        let input = (0..<cols).map {
            Float16(sin(Float($0 * 17 + 3)) * 0.75)
        }
        var memberCodes: [[UInt8]] = []
        var memberScales: [[Float16]] = []
        for (member, rows) in rowCounts.enumerated() {
            let codes: [UInt8] = nativeTernaryCodes(
                rows: rows,
                cols: cols
            ) { row, col in
                UInt8((member * 11 + row * 13 + col * 7) % 3)
            }
            let scales: [Float16] = (0..<(rows * groups)).map { index in
                Float16(0.00390625 * Float((member * 3 + index) % 11 + 1))
            }
            memberCodes.append(codes)
            memberScales.append(scales)
        }

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let inputBuffer = try makeSharedBuffer(device: device, input)
        let codeBuffers = try memberCodes.map {
            try makeSharedBuffer(device: device, $0)
        }
        let scaleBuffers = try memberScales.map {
            try makeSharedBuffer(device: device, $0)
        }
        let bankCodes: [UInt8] = memberCodes.flatMap { $0 }
        let bankScales: [Float16] = memberScales.flatMap { $0 }
        let bankCodeBuffer = try makeSharedBuffer(device: device, bankCodes)
        let bankScaleBuffer = try makeSharedBuffer(device: device, bankScales)

        for bitCount in [4, 5, 6] {
            let builder = try XCTUnwrap(makeComputePipeline(
                device: device,
                shaderFile: "signed_quant.metal",
                functionName: "signed_activation_bitplanes_i\(bitCount)_g128"
            ))
            let independent = try XCTUnwrap(makeComputePipeline(
                device: device,
                shaderFile: "signed_quant.metal",
                functionName: "signed_ternary_bitplane_i\(bitCount)_matvec_g128_rows8"
            ))
            let bank = try XCTUnwrap(makeComputePipeline(
                device: device,
                shaderFile: "signed_quant.metal",
                functionName: "signed_ternary_bitplane_i\(bitCount)_bank4_matvec_g128_rows8"
            ))
            let planes = try makeSharedBuffer(
                device: device,
                count: groups * bitCount * 4,
                of: UInt32.self
            )
            let activationScales = try makeSharedBuffer(
                device: device, count: groups, of: Float16.self)
            let independentOutputs = try rowCounts.map {
                try makeSharedBuffer(device: device, count: $0, of: Float16.self)
            }
            let bankOutputs = try rowCounts.map {
                try makeSharedBuffer(device: device, count: $0, of: Float16.self)
            }

            try runOnGPU(queue: queue) { encoder in
                encoder.setComputePipelineState(builder)
                encoder.setBuffer(inputBuffer, offset: 0, index: 0)
                encoder.setBuffer(planes, offset: 0, index: 1)
                encoder.setBuffer(activationScales, offset: 0, index: 2)
                var builderCols = UInt32(cols)
                encoder.setBytes(&builderCols, length: 4, index: 3)
                encoder.dispatchThreadgroups(
                    MTLSize(width: groups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )

                encoder.setComputePipelineState(independent)
                for member in rowCounts.indices {
                    encoder.setBuffer(codeBuffers[member], offset: 0, index: 0)
                    encoder.setBuffer(scaleBuffers[member], offset: 0, index: 1)
                    encoder.setBuffer(planes, offset: 0, index: 2)
                    encoder.setBuffer(activationScales, offset: 0, index: 3)
                    encoder.setBuffer(independentOutputs[member], offset: 0, index: 4)
                    var rows = UInt32(rowCounts[member])
                    var matvecCols = UInt32(cols)
                    encoder.setBytes(&rows, length: 4, index: 5)
                    encoder.setBytes(&matvecCols, length: 4, index: 6)
                    encoder.dispatchThreadgroups(
                        MTLSize(width: (rowCounts[member] + 3) / 4, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
                    )
                }

                encoder.setComputePipelineState(bank)
                encoder.setBuffer(bankCodeBuffer, offset: 0, index: 0)
                encoder.setBuffer(bankScaleBuffer, offset: 0, index: 1)
                encoder.setBuffer(planes, offset: 0, index: 2)
                encoder.setBuffer(activationScales, offset: 0, index: 3)
                for member in rowCounts.indices {
                    encoder.setBuffer(bankOutputs[member], offset: 0, index: member + 4)
                    var rows = UInt32(rowCounts[member])
                    encoder.setBytes(&rows, length: 4, index: member + 8)
                }
                var bankCols = UInt32(cols)
                encoder.setBytes(&bankCols, length: 4, index: 12)
                encoder.dispatchThreadgroups(
                    MTLSize(
                        width: (rowCounts.reduce(0, +) + 3) / 4,
                        height: 1,
                        depth: 1
                    ),
                    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
                )
            }

            for member in rowCounts.indices {
                let expected = Array(UnsafeBufferPointer(
                    start: independentOutputs[member].contents()
                        .assumingMemoryBound(to: Float16.self),
                    count: rowCounts[member]
                ))
                let actual = Array(UnsafeBufferPointer(
                    start: bankOutputs[member].contents()
                        .assumingMemoryBound(to: Float16.self),
                    count: rowCounts[member]
                ))
                XCTAssertEqual(
                    actual, expected,
                    "i\(bitCount) member \(member) must preserve standalone rounding"
                )
            }
        }
    }

    func testTernaryGateUpSwiGLUIsBitExactWithStagedConsumers() throws {
        let rows = 7
        let cols = 256
        let groups = cols / 128
        let gateCodes = nativeTernaryCodes(rows: rows, cols: cols) {
            UInt8(($0 * 13 + $1 * 7) % 3)
        }
        let upCodes = nativeTernaryCodes(rows: rows, cols: cols) {
            UInt8(($0 * 5 + $1 * 11 + 1) % 3)
        }
        let gateScales = (0..<(rows * groups)).map {
            Float16(0.00390625 * Float(($0 % 7) + 1))
        }
        let upScales = (0..<(rows * groups)).map {
            Float16(0.005859375 * Float(($0 % 5) + 1))
        }
        let input = (0..<cols).map {
            Float16(sin(Float($0 * 17 + 3)) * 0.75)
        }

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let gateCodeBuffer = try makeSharedBuffer(device: device, gateCodes)
        let upCodeBuffer = try makeSharedBuffer(device: device, upCodes)
        let gateScaleBuffer = try makeSharedBuffer(device: device, gateScales)
        let upScaleBuffer = try makeSharedBuffer(device: device, upScales)
        let inputBuffer = try makeSharedBuffer(device: device, input)

        for bitCount in [4, 5, 6] {
            let builder = try XCTUnwrap(makeComputePipeline(
                device: device,
                shaderFile: "signed_quant.metal",
                functionName: "signed_activation_bitplanes_i\(bitCount)_g128"
            ))
            let matvec = try XCTUnwrap(makeComputePipeline(
                device: device,
                shaderFile: "signed_quant.metal",
                functionName: "signed_ternary_bitplane_i\(bitCount)_matvec_g128_rows8"
            ))
            let swiglu = try XCTUnwrap(makeComputePipeline(
                device: device,
                shaderFile: "activations_precise.metal",
                functionName: "swiglu_fused"
            ))
            let fused = try XCTUnwrap(makeComputePipeline(
                device: device,
                shaderFile: "signed_quant.metal",
                functionName: "signed_ternary_bitplane_i\(bitCount)_gate_up_swiglu_g128_rows8"
            ))
            let planes = try makeSharedBuffer(
                device: device,
                count: groups * bitCount * 4,
                of: UInt32.self
            )
            let activationScales = try makeSharedBuffer(
                device: device, count: groups, of: Float16.self)
            let stagedGate = try makeSharedBuffer(
                device: device, count: rows, of: Float16.self)
            let stagedUp = try makeSharedBuffer(
                device: device, count: rows, of: Float16.self)
            let stagedOutput = try makeSharedBuffer(
                device: device, count: rows, of: Float16.self)
            let fusedOutput = try makeSharedBuffer(
                device: device, count: rows, of: Float16.self)

            try runOnGPU(queue: queue) { encoder in
                encoder.setComputePipelineState(builder)
                encoder.setBuffer(inputBuffer, offset: 0, index: 0)
                encoder.setBuffer(planes, offset: 0, index: 1)
                encoder.setBuffer(activationScales, offset: 0, index: 2)
                var builderCols = UInt32(cols)
                encoder.setBytes(&builderCols, length: 4, index: 3)
                encoder.dispatchThreadgroups(
                    MTLSize(width: groups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )

                encoder.setComputePipelineState(matvec)
                for (codes, scales, output) in [
                    (gateCodeBuffer, gateScaleBuffer, stagedGate),
                    (upCodeBuffer, upScaleBuffer, stagedUp),
                ] {
                    encoder.setBuffer(codes, offset: 0, index: 0)
                    encoder.setBuffer(scales, offset: 0, index: 1)
                    encoder.setBuffer(planes, offset: 0, index: 2)
                    encoder.setBuffer(activationScales, offset: 0, index: 3)
                    encoder.setBuffer(output, offset: 0, index: 4)
                    var rowCount = UInt32(rows)
                    var matvecCols = UInt32(cols)
                    encoder.setBytes(&rowCount, length: 4, index: 5)
                    encoder.setBytes(&matvecCols, length: 4, index: 6)
                    encoder.dispatchThreadgroups(
                        MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
                    )
                }

                encoder.setComputePipelineState(swiglu)
                encoder.setBuffer(stagedGate, offset: 0, index: 0)
                encoder.setBuffer(stagedUp, offset: 0, index: 1)
                encoder.setBuffer(stagedOutput, offset: 0, index: 2)
                var outputCount = UInt32(rows)
                encoder.setBytes(&outputCount, length: 4, index: 3)
                encoder.dispatchThreads(
                    MTLSize(width: rows, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: rows, height: 1, depth: 1)
                )

                encoder.setComputePipelineState(fused)
                encoder.setBuffer(gateCodeBuffer, offset: 0, index: 0)
                encoder.setBuffer(gateScaleBuffer, offset: 0, index: 1)
                encoder.setBuffer(upCodeBuffer, offset: 0, index: 2)
                encoder.setBuffer(upScaleBuffer, offset: 0, index: 3)
                encoder.setBuffer(planes, offset: 0, index: 4)
                encoder.setBuffer(activationScales, offset: 0, index: 5)
                encoder.setBuffer(fusedOutput, offset: 0, index: 6)
                var fusedRows = UInt32(rows)
                var fusedCols = UInt32(cols)
                encoder.setBytes(&fusedRows, length: 4, index: 7)
                encoder.setBytes(&fusedCols, length: 4, index: 8)
                encoder.dispatchThreadgroups(
                    MTLSize(width: (rows + 3) / 4, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
                )
            }

            let staged = UnsafeBufferPointer(
                start: stagedOutput.contents().assumingMemoryBound(to: Float16.self),
                count: rows
            ).map(\.bitPattern)
            let actual = UnsafeBufferPointer(
                start: fusedOutput.contents().assumingMemoryBound(to: Float16.self),
                count: rows
            ).map(\.bitPattern)
            XCTAssertEqual(actual, staged, "i\(bitCount) must preserve staged rounding")
        }
    }

    func testRMSNormActivationViewsAreBitExactWithStagedGraph() throws {
        let cols = 5_120
        let groups = cols / 128
        let input = (0..<cols).map {
            let groupScale = 0.125 + Float(($0 / 128) % 13) * 0.375
            return Float16(sin(Float($0 * 17 + 3)) * groupScale)
        }
        let weight = (0..<cols).map {
            Float16(cos(Float($0 * 7 + 5)) * 0.25)
        }

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let norm = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "norms.metal",
            functionName: "rms_norm_1pw"
        ))
        let preciseScale = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "norms.metal",
            functionName: "rms_norm_scale_only_precise"
        ))
        let inputBuffer = try makeSharedBuffer(device: device, input)
        let weightBuffer = try makeSharedBuffer(device: device, weight)
        let normalized = try makeSharedBuffer(
            device: device, count: cols, of: Float16.self)
        let normScale = try makeSharedBuffer(
            device: device, count: 1, of: Float.self)

        for bitCount in [2, 3, 4, 5, 6] {
            let planeWords = groups * bitCount * 4
            let builder = try XCTUnwrap(makeComputePipeline(
                device: device,
                shaderFile: "signed_quant.metal",
                functionName: "signed_activation_bitplanes_i\(bitCount)_g128"
            ))
            let scaledView = try XCTUnwrap(makeComputePipeline(
                device: device,
                shaderFile: "signed_quant.metal",
                functionName: "norm_scale_signed_activation_bitplanes_i\(bitCount)_g128"
            ))
            let stagedPlanes = try makeSharedBuffer(
                device: device, count: planeWords, of: UInt32.self)
            let stagedScales = try makeSharedBuffer(
                device: device, count: groups, of: Float16.self)
            let fusedPlanes = try makeSharedBuffer(
                device: device, count: planeWords, of: UInt32.self)
            let fusedScales = try makeSharedBuffer(
                device: device, count: groups, of: Float16.self)

            try runOnGPU(queue: queue) { encoder in
                encoder.setComputePipelineState(norm)
                encoder.setBuffer(inputBuffer, offset: 0, index: 0)
                encoder.setBuffer(weightBuffer, offset: 0, index: 1)
                encoder.setBuffer(normalized, offset: 0, index: 2)
                var dimension = UInt32(cols)
                var eps: Float = 1e-6
                encoder.setBytes(&dimension, length: 4, index: 3)
                encoder.setBytes(&eps, length: 4, index: 4)
                encoder.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1_024, height: 1, depth: 1)
                )

                encoder.setComputePipelineState(builder)
                encoder.setBuffer(normalized, offset: 0, index: 0)
                encoder.setBuffer(stagedPlanes, offset: 0, index: 1)
                encoder.setBuffer(stagedScales, offset: 0, index: 2)
                encoder.setBytes(&dimension, length: 4, index: 3)
                encoder.dispatchThreadgroups(
                    MTLSize(width: groups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )

                encoder.setComputePipelineState(preciseScale)
                encoder.setBuffer(inputBuffer, offset: 0, index: 0)
                encoder.setBuffer(normScale, offset: 0, index: 1)
                encoder.setBytes(&dimension, length: 4, index: 2)
                encoder.setBytes(&eps, length: 4, index: 3)
                encoder.dispatchThreadgroups(
                    MTLSize(width: 1, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 1_024, height: 1, depth: 1)
                )

                encoder.setComputePipelineState(scaledView)
                encoder.setBuffer(normScale, offset: 0, index: 0)
                encoder.setBuffer(inputBuffer, offset: 0, index: 1)
                encoder.setBuffer(weightBuffer, offset: 0, index: 2)
                encoder.setBuffer(fusedPlanes, offset: 0, index: 3)
                encoder.setBuffer(fusedScales, offset: 0, index: 4)
                encoder.setBytes(&dimension, length: 4, index: 5)
                encoder.dispatchThreadgroups(
                    MTLSize(width: groups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
            }

            let stagedPlaneValues = UnsafeBufferPointer(
                start: stagedPlanes.contents().assumingMemoryBound(to: UInt32.self),
                count: planeWords
            )
            let fusedPlaneValues = UnsafeBufferPointer(
                start: fusedPlanes.contents().assumingMemoryBound(to: UInt32.self),
                count: planeWords
            )
            XCTAssertEqual(
                Array(fusedPlaneValues),
                Array(stagedPlaneValues),
                "i\(bitCount) planes"
            )
            let stagedScaleValues = UnsafeBufferPointer(
                start: stagedScales.contents().assumingMemoryBound(to: Float16.self),
                count: groups
            )
            let fusedScaleValues = UnsafeBufferPointer(
                start: fusedScales.contents().assumingMemoryBound(to: Float16.self),
                count: groups
            )
            XCTAssertEqual(
                fusedScaleValues.map(\.bitPattern),
                stagedScaleValues.map(\.bitPattern),
                "i\(bitCount) scales"
            )
        }
    }

    func testResidualAddPreciseNormScaleIsBitExactWithStagedGraph() throws {
        let cols = 256
        let first = (0..<cols).map {
            Float16(sin(Float($0 * 17 + 3)) * 1.5)
        }
        let second = (0..<cols).map {
            Float16(cos(Float($0 * 11 + 5)) * 0.75)
        }
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let add = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "activations.metal",
            functionName: "elementwise_add"
        ))
        let scale = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "norms.metal",
            functionName: "rms_norm_scale_only_precise"
        ))
        let fused = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "norms.metal",
            functionName: "residual_add_rms_norm_scale_only_precise"
        ))
        let firstBuffer = try makeSharedBuffer(device: device, first)
        let secondBuffer = try makeSharedBuffer(device: device, second)
        let stagedOutput = try makeSharedBuffer(
            device: device, count: cols, of: Float16.self)
        let fusedOutput = try makeSharedBuffer(
            device: device, count: cols, of: Float16.self)
        let stagedScale = try makeSharedBuffer(
            device: device, count: 1, of: Float.self)
        let fusedScale = try makeSharedBuffer(
            device: device, count: 1, of: Float.self)

        try runOnGPU(queue: queue) { encoder in
            encoder.setComputePipelineState(add)
            encoder.setBuffer(firstBuffer, offset: 0, index: 0)
            encoder.setBuffer(secondBuffer, offset: 0, index: 1)
            encoder.setBuffer(stagedOutput, offset: 0, index: 2)
            var addCount = UInt32(cols)
            encoder.setBytes(&addCount, length: 4, index: 3)
            encoder.dispatchThreads(
                MTLSize(width: cols, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: cols, height: 1, depth: 1)
            )

            encoder.setComputePipelineState(scale)
            encoder.setBuffer(stagedOutput, offset: 0, index: 0)
            encoder.setBuffer(stagedScale, offset: 0, index: 1)
            var scaleDim = UInt32(cols)
            var scaleEpsilon: Float = 1e-6
            encoder.setBytes(&scaleDim, length: 4, index: 2)
            encoder.setBytes(&scaleEpsilon, length: 4, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: cols, height: 1, depth: 1)
            )

            encoder.setComputePipelineState(fused)
            encoder.setBuffer(firstBuffer, offset: 0, index: 0)
            encoder.setBuffer(secondBuffer, offset: 0, index: 1)
            encoder.setBuffer(fusedOutput, offset: 0, index: 2)
            encoder.setBuffer(fusedScale, offset: 0, index: 3)
            var fusedDim = UInt32(cols)
            var fusedEpsilon: Float = 1e-6
            encoder.setBytes(&fusedDim, length: 4, index: 4)
            encoder.setBytes(&fusedEpsilon, length: 4, index: 5)
            encoder.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: cols, height: 1, depth: 1)
            )
        }

        let stagedValues = UnsafeBufferPointer(
            start: stagedOutput.contents().assumingMemoryBound(to: Float16.self),
            count: cols
        ).map(\.bitPattern)
        let fusedValues = UnsafeBufferPointer(
            start: fusedOutput.contents().assumingMemoryBound(to: Float16.self),
            count: cols
        ).map(\.bitPattern)
        XCTAssertEqual(fusedValues, stagedValues)
        let stagedScaleBits = stagedScale.contents()
            .assumingMemoryBound(to: Float.self).pointee.bitPattern
        let fusedScaleBits = fusedScale.contents()
            .assumingMemoryBound(to: Float.self).pointee.bitPattern
        XCTAssertEqual(fusedScaleBits, stagedScaleBits)
    }

    func testGatedRMSNormI6ProducerIsBitExactWithStagedGraph() throws {
        let heads = 48
        let headDim = 128
        let cols = heads * headDim
        let planeWords = heads * 6 * 4
        let input = (0..<cols).map {
            let headScale = 0.125 + Float(($0 / headDim) % 12) * 0.625
            return Float16(sin(Float($0 * 17 + 3)) * headScale)
        }
        let gate = (0..<cols).map {
            let headScale = 0.25 + Float(($0 / headDim) % 9) * 0.5
            return Float16(cos(Float($0 * 11 + 5)) * headScale)
        }
        let weight = (0..<headDim).map {
            Float16(0.5 + Float(($0 * 7) % 31) / 32.0)
        }

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let norm = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "norms_gated_precise.metal",
            functionName: "rms_norm_gated_d128"
        ))
        let builder = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant.metal",
            functionName: "signed_activation_bitplanes_i6_g128"
        ))
        let fused = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant.metal",
            functionName: "rms_norm_gated_d128_signed_activation_bitplanes_i6_g128"
        ))

        let inputBuffer = try makeSharedBuffer(device: device, input)
        let gateBuffer = try makeSharedBuffer(device: device, gate)
        let weightBuffer = try makeSharedBuffer(device: device, weight)
        let normalized = try makeSharedBuffer(
            device: device, count: cols, of: Float16.self)
        let stagedPlanes = try makeSharedBuffer(
            device: device, count: planeWords, of: UInt32.self)
        let stagedScales = try makeSharedBuffer(
            device: device, count: heads, of: Float16.self)
        let fusedPlanes = try makeSharedBuffer(
            device: device, count: planeWords, of: UInt32.self)
        let fusedScales = try makeSharedBuffer(
            device: device, count: heads, of: Float16.self)

        try runOnGPU(queue: queue) { encoder in
            encoder.setComputePipelineState(norm)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(gateBuffer, offset: 0, index: 1)
            encoder.setBuffer(weightBuffer, offset: 0, index: 2)
            encoder.setBuffer(normalized, offset: 0, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: heads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )

            encoder.setComputePipelineState(builder)
            encoder.setBuffer(normalized, offset: 0, index: 0)
            encoder.setBuffer(stagedPlanes, offset: 0, index: 1)
            encoder.setBuffer(stagedScales, offset: 0, index: 2)
            var colCount = UInt32(cols)
            encoder.setBytes(&colCount, length: 4, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: heads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )

            encoder.setComputePipelineState(fused)
            encoder.setBuffer(inputBuffer, offset: 0, index: 0)
            encoder.setBuffer(gateBuffer, offset: 0, index: 1)
            encoder.setBuffer(weightBuffer, offset: 0, index: 2)
            encoder.setBuffer(fusedPlanes, offset: 0, index: 3)
            encoder.setBuffer(fusedScales, offset: 0, index: 4)
            var fusedDimension = UInt32(headDim)
            var fusedEps: Float = 1e-6
            encoder.setBytes(&fusedDimension, length: 4, index: 5)
            encoder.setBytes(&fusedEps, length: 4, index: 6)
            encoder.dispatchThreadgroups(
                MTLSize(width: heads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )

        }

        let stagedPlaneValues = UnsafeBufferPointer(
            start: stagedPlanes.contents().assumingMemoryBound(to: UInt32.self),
            count: planeWords
        )
        let fusedPlaneValues = UnsafeBufferPointer(
            start: fusedPlanes.contents().assumingMemoryBound(to: UInt32.self),
            count: planeWords
        )
        XCTAssertEqual(Array(fusedPlaneValues), Array(stagedPlaneValues))

        let stagedScaleValues = UnsafeBufferPointer(
            start: stagedScales.contents().assumingMemoryBound(to: Float16.self),
            count: heads
        )
        let fusedScaleValues = UnsafeBufferPointer(
            start: fusedScales.contents().assumingMemoryBound(to: Float16.self),
            count: heads
        )
        XCTAssertEqual(
            fusedScaleValues.map(\.bitPattern),
            stagedScaleValues.map(\.bitPattern)
        )
    }

    func testSigmoidMulI6ProducerIsBitExactWithStagedGraph() throws {
        let groups = 48
        let cols = groups * 128
        let planeWords = groups * 6 * 4
        let inputA = (0..<cols).map {
            let groupScale = 0.125 + Float(($0 / 128) % 13) * 0.5
            return Float16(sin(Float($0 * 17 + 3)) * groupScale)
        }
        let inputB = (0..<cols).map {
            let groupScale = 0.25 + Float(($0 / 128) % 11) * 0.375
            return Float16(cos(Float($0 * 7 + 5)) * groupScale)
        }

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let sigmoidMul = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "activations_precise.metal",
            functionName: "sigmoid_mul"
        ))
        let builder = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant.metal",
            functionName: "signed_activation_bitplanes_i6_g128"
        ))
        let fused = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant.metal",
            functionName: "sigmoid_mul_signed_activation_bitplanes_i6_g128"
        ))

        let inputABuffer = try makeSharedBuffer(device: device, inputA)
        let inputBBuffer = try makeSharedBuffer(device: device, inputB)
        let gated = try makeSharedBuffer(
            device: device, count: cols, of: Float16.self)
        let stagedPlanes = try makeSharedBuffer(
            device: device, count: planeWords, of: UInt32.self)
        let stagedScales = try makeSharedBuffer(
            device: device, count: groups, of: Float16.self)
        let fusedPlanes = try makeSharedBuffer(
            device: device, count: planeWords, of: UInt32.self)
        let fusedScales = try makeSharedBuffer(
            device: device, count: groups, of: Float16.self)

        try runOnGPU(queue: queue) { encoder in
            encoder.setComputePipelineState(sigmoidMul)
            encoder.setBuffer(inputABuffer, offset: 0, index: 0)
            encoder.setBuffer(inputBBuffer, offset: 0, index: 1)
            encoder.setBuffer(gated, offset: 0, index: 2)
            var stagedCols = UInt32(cols)
            encoder.setBytes(&stagedCols, length: 4, index: 3)
            encoder.dispatchThreads(
                MTLSize(width: cols, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)
            )

            encoder.setComputePipelineState(builder)
            encoder.setBuffer(gated, offset: 0, index: 0)
            encoder.setBuffer(stagedPlanes, offset: 0, index: 1)
            encoder.setBuffer(stagedScales, offset: 0, index: 2)
            var builderCols = UInt32(cols)
            encoder.setBytes(&builderCols, length: 4, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: groups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )

            encoder.setComputePipelineState(fused)
            encoder.setBuffer(inputABuffer, offset: 0, index: 0)
            encoder.setBuffer(inputBBuffer, offset: 0, index: 1)
            encoder.setBuffer(fusedPlanes, offset: 0, index: 2)
            encoder.setBuffer(fusedScales, offset: 0, index: 3)
            var fusedCols = UInt32(cols)
            encoder.setBytes(&fusedCols, length: 4, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: groups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        }

        let stagedPlaneValues = UnsafeBufferPointer(
            start: stagedPlanes.contents().assumingMemoryBound(to: UInt32.self),
            count: planeWords
        )
        let fusedPlaneValues = UnsafeBufferPointer(
            start: fusedPlanes.contents().assumingMemoryBound(to: UInt32.self),
            count: planeWords
        )
        XCTAssertEqual(Array(fusedPlaneValues), Array(stagedPlaneValues))

        let stagedScaleValues = UnsafeBufferPointer(
            start: stagedScales.contents().assumingMemoryBound(to: Float16.self),
            count: groups
        )
        let fusedScaleValues = UnsafeBufferPointer(
            start: fusedScales.contents().assumingMemoryBound(to: Float16.self),
            count: groups
        )
        XCTAssertEqual(
            fusedScaleValues.map(\.bitPattern),
            stagedScaleValues.map(\.bitPattern)
        )
    }

    func testSwiGLUI5ProducerIsBitExactWithStagedGraph() throws {
        let groups = 136
        let cols = groups * 128
        let planeWords = groups * 5 * 4
        let gate = (0..<cols).map {
            let groupScale = 0.125 + Float(($0 / 128) % 13) * 0.5
            return Float16(sin(Float($0 * 17 + 3)) * groupScale)
        }
        let up = (0..<cols).map {
            let groupScale = 0.25 + Float(($0 / 128) % 11) * 0.375
            return Float16(cos(Float($0 * 7 + 5)) * groupScale)
        }

        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let swiglu = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "activations_precise.metal",
            functionName: "swiglu_fused"
        ))
        let builder = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant.metal",
            functionName: "signed_activation_bitplanes_i5_g128"
        ))
        let fused = try XCTUnwrap(makeComputePipeline(
            device: device,
            shaderFile: "signed_quant.metal",
            functionName: "swiglu_signed_activation_bitplanes_i5_g128"
        ))

        let gateBuffer = try makeSharedBuffer(device: device, gate)
        let upBuffer = try makeSharedBuffer(device: device, up)
        let activated = try makeSharedBuffer(
            device: device, count: cols, of: Float16.self)
        let stagedPlanes = try makeSharedBuffer(
            device: device, count: planeWords, of: UInt32.self)
        let stagedScales = try makeSharedBuffer(
            device: device, count: groups, of: Float16.self)
        let fusedPlanes = try makeSharedBuffer(
            device: device, count: planeWords, of: UInt32.self)
        let fusedScales = try makeSharedBuffer(
            device: device, count: groups, of: Float16.self)

        try runOnGPU(queue: queue) { encoder in
            encoder.setComputePipelineState(swiglu)
            encoder.setBuffer(gateBuffer, offset: 0, index: 0)
            encoder.setBuffer(upBuffer, offset: 0, index: 1)
            encoder.setBuffer(activated, offset: 0, index: 2)
            var stagedCols = UInt32(cols)
            encoder.setBytes(&stagedCols, length: 4, index: 3)
            encoder.dispatchThreads(
                MTLSize(width: cols, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)
            )

            encoder.setComputePipelineState(builder)
            encoder.setBuffer(activated, offset: 0, index: 0)
            encoder.setBuffer(stagedPlanes, offset: 0, index: 1)
            encoder.setBuffer(stagedScales, offset: 0, index: 2)
            var builderCols = UInt32(cols)
            encoder.setBytes(&builderCols, length: 4, index: 3)
            encoder.dispatchThreadgroups(
                MTLSize(width: groups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )

            encoder.setComputePipelineState(fused)
            encoder.setBuffer(gateBuffer, offset: 0, index: 0)
            encoder.setBuffer(upBuffer, offset: 0, index: 1)
            encoder.setBuffer(fusedPlanes, offset: 0, index: 2)
            encoder.setBuffer(fusedScales, offset: 0, index: 3)
            var fusedCols = UInt32(cols)
            encoder.setBytes(&fusedCols, length: 4, index: 4)
            encoder.dispatchThreadgroups(
                MTLSize(width: groups, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        }

        let stagedPlaneValues = UnsafeBufferPointer(
            start: stagedPlanes.contents().assumingMemoryBound(to: UInt32.self),
            count: planeWords
        )
        let fusedPlaneValues = UnsafeBufferPointer(
            start: fusedPlanes.contents().assumingMemoryBound(to: UInt32.self),
            count: planeWords
        )
        XCTAssertEqual(Array(fusedPlaneValues), Array(stagedPlaneValues))

        let stagedScaleValues = UnsafeBufferPointer(
            start: stagedScales.contents().assumingMemoryBound(to: Float16.self),
            count: groups
        )
        let fusedScaleValues = UnsafeBufferPointer(
            start: fusedScales.contents().assumingMemoryBound(to: Float16.self),
            count: groups
        )
        XCTAssertEqual(
            fusedScaleValues.map(\.bitPattern),
            stagedScaleValues.map(\.bitPattern)
        )
    }

    func testBinaryDualProjectionBankIsExactAndCanProfile() throws {
        let profiling = ProcessInfo.processInfo.environment["SMELT_SIGNED_DUAL_PROFILE"] == "1"
        let environment = ProcessInfo.processInfo.environment
        let firstRows = profiling
            ? Int(environment["SMELT_SIGNED_DUAL_FIRST_ROWS"] ?? "10240")!
            : 16
        let secondRows = profiling
            ? Int(environment["SMELT_SIGNED_DUAL_SECOND_ROWS"] ?? "6144")!
            : 8
        let cols = profiling
            ? Int(environment["SMELT_SIGNED_DUAL_COLS"] ?? "5120")!
            : 256
        let groups = cols / 128
        let firstCodes = profiling
            ? [UInt8](repeating: 0xa5, count: firstRows * cols / 8)
            : packedCodes(rows: firstRows, cols: cols, bits: 1) {
                UInt8(($0 * 13 + $1 * 7) & 1)
            }
        let secondCodes = profiling
            ? [UInt8](repeating: 0x3c, count: secondRows * cols / 8)
            : packedCodes(rows: secondRows, cols: cols, bits: 1) {
                UInt8(($0 * 11 + $1 * 5 + 1) & 1)
            }
        let firstScales = (0..<(firstRows * groups)).map {
            Float16(0.001953125 * Float(($0 % 7) + 1))
        }
        let secondScales = (0..<(secondRows * groups)).map {
            Float16(0.0029296875 * Float(($0 % 5) + 1))
        }
        let input = (0..<cols).map { Float16(sin(Float($0 * 17 + 3)) * 0.75) }
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        let dual = try XCTUnwrap(makeComputePipeline(
            device: device, shaderFile: "signed_quant.metal",
            functionName: "signed_binary_dual_matvec_g128_rows8"
        ))
        let virtual = try XCTUnwrap(makeComputePipeline(
            device: device, shaderFile: "signed_quant.metal",
            functionName: "signed_binary_virtual_bank_matvec_g128_rows8"
        ))
        let packed = try XCTUnwrap(makeComputePipeline(
            device: device, shaderFile: "signed_quant.metal",
            functionName: "signed_binary_packed_bank_matvec_g128_rows8"
        ))
        let direct = try XCTUnwrap(makeComputePipeline(
            device: device, shaderFile: "signed_quant.metal",
            functionName: "signed_binary_matvec_g128_rows8"
        ))
        let firstCodesBuffer = try makeSharedBuffer(device: device, firstCodes)
        let secondCodesBuffer = try makeSharedBuffer(device: device, secondCodes)
        let firstScalesBuffer = try makeSharedBuffer(device: device, firstScales)
        let secondScalesBuffer = try makeSharedBuffer(device: device, secondScales)
        let inputBuffer = try makeSharedBuffer(device: device, input)
        let packedCodesBuffer = try makeSharedBuffer(
            device: device, firstCodes + secondCodes)
        let packedScalesBuffer = try makeSharedBuffer(
            device: device, firstScales + secondScales)
        let firstOutput = try makeSharedBuffer(
            device: device, count: firstRows, of: Float16.self)
        let secondOutput = try makeSharedBuffer(
            device: device, count: secondRows, of: Float16.self)
        let firstVirtualOutput = try makeSharedBuffer(
            device: device, count: firstRows, of: Float16.self)
        let secondVirtualOutput = try makeSharedBuffer(
            device: device, count: secondRows, of: Float16.self)
        let firstPackedOutput = try makeSharedBuffer(
            device: device, count: firstRows, of: Float16.self)
        let secondPackedOutput = try makeSharedBuffer(
            device: device, count: secondRows, of: Float16.self)
        let firstReference = try makeSharedBuffer(
            device: device, count: firstRows, of: Float16.self)
        let secondReference = try makeSharedBuffer(
            device: device, count: secondRows, of: Float16.self)

        func encodeDual(_ encoder: MTLComputeCommandEncoder) {
            encoder.setComputePipelineState(dual)
            encoder.setBuffer(firstCodesBuffer, offset: 0, index: 0)
            encoder.setBuffer(firstScalesBuffer, offset: 0, index: 1)
            encoder.setBuffer(secondCodesBuffer, offset: 0, index: 2)
            encoder.setBuffer(secondScalesBuffer, offset: 0, index: 3)
            encoder.setBuffer(inputBuffer, offset: 0, index: 4)
            encoder.setBuffer(firstOutput, offset: 0, index: 5)
            encoder.setBuffer(secondOutput, offset: 0, index: 6)
            var first = UInt32(firstRows)
            var second = UInt32(secondRows)
            var colCount = UInt32(cols)
            encoder.setBytes(&first, length: 4, index: 7)
            encoder.setBytes(&second, length: 4, index: 8)
            encoder.setBytes(&colCount, length: 4, index: 9)
            encoder.dispatchThreadgroups(
                MTLSize(width: (max(firstRows, secondRows) + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }
        func encodeVirtual(_ encoder: MTLComputeCommandEncoder) {
            encoder.setComputePipelineState(virtual)
            encoder.setBuffer(firstCodesBuffer, offset: 0, index: 0)
            encoder.setBuffer(firstScalesBuffer, offset: 0, index: 1)
            encoder.setBuffer(secondCodesBuffer, offset: 0, index: 2)
            encoder.setBuffer(secondScalesBuffer, offset: 0, index: 3)
            encoder.setBuffer(inputBuffer, offset: 0, index: 4)
            encoder.setBuffer(firstVirtualOutput, offset: 0, index: 5)
            encoder.setBuffer(secondVirtualOutput, offset: 0, index: 6)
            var first = UInt32(firstRows)
            var second = UInt32(secondRows)
            var colCount = UInt32(cols)
            encoder.setBytes(&first, length: 4, index: 7)
            encoder.setBytes(&second, length: 4, index: 8)
            encoder.setBytes(&colCount, length: 4, index: 9)
            encoder.dispatchThreadgroups(
                MTLSize(width: (firstRows + secondRows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }
        func encodePacked(_ encoder: MTLComputeCommandEncoder) {
            encoder.setComputePipelineState(packed)
            encoder.setBuffer(packedCodesBuffer, offset: 0, index: 0)
            encoder.setBuffer(packedScalesBuffer, offset: 0, index: 1)
            encoder.setBuffer(inputBuffer, offset: 0, index: 2)
            encoder.setBuffer(firstPackedOutput, offset: 0, index: 3)
            encoder.setBuffer(secondPackedOutput, offset: 0, index: 4)
            var first = UInt32(firstRows)
            var second = UInt32(secondRows)
            var colCount = UInt32(cols)
            encoder.setBytes(&first, length: 4, index: 5)
            encoder.setBytes(&second, length: 4, index: 6)
            encoder.setBytes(&colCount, length: 4, index: 7)
            encoder.dispatchThreadgroups(
                MTLSize(width: (firstRows + secondRows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }
        func encodeDirect(
            _ encoder: MTLComputeCommandEncoder,
            codes: MTLBuffer,
            scales: MTLBuffer,
            output: MTLBuffer,
            rows: Int
        ) {
            encoder.setComputePipelineState(direct)
            encoder.setBuffer(codes, offset: 0, index: 0)
            encoder.setBuffer(scales, offset: 0, index: 1)
            encoder.setBuffer(inputBuffer, offset: 0, index: 2)
            encoder.setBuffer(output, offset: 0, index: 3)
            var rowCount = UInt32(rows)
            var colCount = UInt32(cols)
            encoder.setBytes(&rowCount, length: 4, index: 4)
            encoder.setBytes(&colCount, length: 4, index: 5)
            encoder.dispatchThreadgroups(
                MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
            )
        }
        try runOnGPU(queue: queue) { encoder in
            encodeDual(encoder)
            encodeVirtual(encoder)
            encodePacked(encoder)
            encodeDirect(
                encoder, codes: firstCodesBuffer, scales: firstScalesBuffer,
                output: firstReference, rows: firstRows)
            encodeDirect(
                encoder, codes: secondCodesBuffer, scales: secondScalesBuffer,
                output: secondReference, rows: secondRows)
        }
        let firstGot = UnsafeBufferPointer(
            start: firstOutput.contents().assumingMemoryBound(to: Float16.self),
            count: firstRows)
        let secondGot = UnsafeBufferPointer(
            start: secondOutput.contents().assumingMemoryBound(to: Float16.self),
            count: secondRows)
        let firstExpected = UnsafeBufferPointer(
            start: firstReference.contents().assumingMemoryBound(to: Float16.self),
            count: firstRows)
        let secondExpected = UnsafeBufferPointer(
            start: secondReference.contents().assumingMemoryBound(to: Float16.self),
            count: secondRows)
        XCTAssertEqual(firstGot.map(\.bitPattern), firstExpected.map(\.bitPattern))
        XCTAssertEqual(secondGot.map(\.bitPattern), secondExpected.map(\.bitPattern))
        let firstVirtual = UnsafeBufferPointer(
            start: firstVirtualOutput.contents().assumingMemoryBound(to: Float16.self),
            count: firstRows)
        let secondVirtual = UnsafeBufferPointer(
            start: secondVirtualOutput.contents().assumingMemoryBound(to: Float16.self),
            count: secondRows)
        XCTAssertEqual(firstVirtual.map(\.bitPattern), firstExpected.map(\.bitPattern))
        XCTAssertEqual(secondVirtual.map(\.bitPattern), secondExpected.map(\.bitPattern))
        let firstPacked = UnsafeBufferPointer(
            start: firstPackedOutput.contents().assumingMemoryBound(to: Float16.self),
            count: firstRows)
        let secondPacked = UnsafeBufferPointer(
            start: secondPackedOutput.contents().assumingMemoryBound(to: Float16.self),
            count: secondRows)
        XCTAssertEqual(firstPacked.map(\.bitPattern), firstExpected.map(\.bitPattern))
        XCTAssertEqual(secondPacked.map(\.bitPattern), secondExpected.map(\.bitPattern))
        guard profiling else { return }

        func profile(
            repetitions: Int,
            encode: (MTLComputeCommandEncoder) -> Void
        ) throws -> Double {
            let command = try XCTUnwrap(queue.makeCommandBuffer())
            let encoder = try XCTUnwrap(command.makeComputeCommandEncoder())
            for _ in 0..<repetitions { encode(encoder) }
            encoder.endEncoding()
            command.commit()
            command.waitUntilCompleted()
            return (command.gpuEndTime - command.gpuStartTime) * 1e6
                / Double(repetitions)
        }
        let repetitions = 100
        let dualUS = try profile(repetitions: repetitions, encode: encodeDual)
        let virtualUS = try profile(repetitions: repetitions, encode: encodeVirtual)
        let packedUS = try profile(repetitions: repetitions, encode: encodePacked)
        let firstUS = try profile(repetitions: repetitions) { encoder in
            encodeDirect(
                encoder, codes: firstCodesBuffer, scales: firstScalesBuffer,
                output: firstReference, rows: firstRows)
        }
        let secondUS = try profile(repetitions: repetitions) { encoder in
            encodeDirect(
                encoder, codes: secondCodesBuffer, scales: secondScalesBuffer,
                output: secondReference, rows: secondRows)
        }
        print(String(format: "signed bank rows=%d+%d cols=%d same-thread=%.2fus virtual=%.2fus packed=%.2fus separate=%.2fus (%.2f+%.2f) speedups same=%.3fx virtual=%.3fx packed=%.3fx", firstRows, secondRows, cols, dualUS, virtualUS, packedUS, firstUS + secondUS, firstUS, secondUS, (firstUS + secondUS) / dualUS, (firstUS + secondUS) / virtualUS, (firstUS + secondUS) / packedUS))
    }
}
