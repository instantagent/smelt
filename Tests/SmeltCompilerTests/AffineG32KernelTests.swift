// AffineG32KernelTests — GPU parity for group-size-32 affine routing
// (gemv routing parity at g32).
//
// Some paths produce affineU4 entries at group size 32 where every prior
// package used 64/128. These tests run the actual decode kernels
// (affine_matvec, fused_affine_gate_up_swiglu) with FC_GROUP_SIZE=32 against
// the CPU reference (SmeltAffineU4.dequantizeRow + dot), at shapes that
// exercise multiple groups per row and multiple row-tiles.

import Metal
import XCTest

@testable import SmeltCompiler
@testable import SmeltRuntime

final class AffineG32KernelTests: XCTestCase {
    private var device: MTLDevice!
    private var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        try? XCTSkipIf(device == nil, "No Metal device available")
        queue = device.makeCommandQueue()
    }

    private func makePipeline(
        _ name: String, cols: UInt32, groupSize: UInt32
    ) throws -> MTLComputePipelineState {
        guard let source = loadMetalShaderSource("lut_matvec.metal") else {
            throw XCTSkip("Could not load lut_matvec.metal")
        }
        let lib = try device.makeLibrary(source: source, options: nil)
        let constants = MTLFunctionConstantValues()
        var colsVal = cols
        var groupSizeVal = groupSize
        constants.setConstantValue(&colsVal, type: .uint, index: 0)
        constants.setConstantValue(&groupSizeVal, type: .uint, index: 1)
        // A PRESENT kernel that fails to build is a hard failure, never a
        // skip (framework §6).
        let fn = try lib.makeFunction(name: name, constantValues: constants)
        return try device.makeComputePipelineState(function: fn)
    }

    private func deterministicWeights(rows: Int, cols: Int, seed: Int) -> [Float] {
        (0..<(rows * cols)).map { i in
            Float((i &* 31 &+ seed &* 17) % 97) / 48.0 - 1.0
        }
    }

    private func buffer<T>(_ array: [T]) -> MTLBuffer {
        array.withUnsafeBytes {
            device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)!
        }
    }

    func testAffineMatvecG32MatchesCPU() throws {
        let rows = 20, cols = 128, groupSize = 32
        let pipeline = try makePipeline(
            "affine_matvec", cols: UInt32(cols), groupSize: UInt32(groupSize))

        let weights = deterministicWeights(rows: rows, cols: cols, seed: 7)
        let packed = SmeltAffineU4.quantize(
            weights, rows: rows, cols: cols, groupSize: groupSize)
        let dequant = SmeltAffineU4.dequantize(packed)
        let input = (0..<cols).map { Float16(Float($0 % 13) / 7.0 - 0.9) }

        var expected = [Float](repeating: 0, count: rows)
        for r in 0..<rows {
            for c in 0..<cols {
                expected[r] += dequant[r * cols + c] * Float(input[c])
            }
        }

        let outBuf = device.makeBuffer(length: rows * 2)!
        var numRows = UInt32(rows)
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(buffer(packed.nibbles), offset: 0, index: 0)
        enc.setBuffer(buffer(packed.scales), offset: 0, index: 1)
        enc.setBuffer(buffer(packed.biases), offset: 0, index: 2)
        enc.setBuffer(buffer(input), offset: 0, index: 3)
        enc.setBuffer(outBuf, offset: 0, index: 4)
        enc.setBytes(&numRows, length: 4, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let out = outBuf.contents().bindMemory(to: UInt16.self, capacity: rows)
        for r in 0..<rows {
            let got = Float(Float16(bitPattern: out[r]))
            XCTAssertEqual(
                got, expected[r], accuracy: max(0.05, abs(expected[r]) * 0.02),
                "row \(r)")
        }
    }

    func testFusedAffineGateUpSwigluG32MatchesCPU() throws {
        let rows = 16, cols = 96, groupSize = 32
        let pipeline = try makePipeline(
            "fused_affine_gate_up_swiglu",
            cols: UInt32(cols), groupSize: UInt32(groupSize))

        let gateW = deterministicWeights(rows: rows, cols: cols, seed: 3)
        let upW = deterministicWeights(rows: rows, cols: cols, seed: 11)
        let gate = SmeltAffineU4.quantize(gateW, rows: rows, cols: cols, groupSize: groupSize)
        let up = SmeltAffineU4.quantize(upW, rows: rows, cols: cols, groupSize: groupSize)
        let gateDq = SmeltAffineU4.dequantize(gate)
        let upDq = SmeltAffineU4.dequantize(up)
        let input = (0..<cols).map { Float16(Float(($0 % 11)) / 6.0 - 0.8) }

        var expected = [Float](repeating: 0, count: rows)
        for r in 0..<rows {
            var g: Float = 0, u: Float = 0
            for c in 0..<cols {
                g += gateDq[r * cols + c] * Float(input[c])
                u += upDq[r * cols + c] * Float(input[c])
            }
            let silu = g / (1 + exp(-g))
            expected[r] = silu * u
        }

        let outBuf = device.makeBuffer(length: rows * 2)!
        var numRows = UInt32(rows)
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(buffer(gate.nibbles), offset: 0, index: 0)
        enc.setBuffer(buffer(gate.scales), offset: 0, index: 1)
        enc.setBuffer(buffer(gate.biases), offset: 0, index: 2)
        enc.setBuffer(buffer(up.nibbles), offset: 0, index: 3)
        enc.setBuffer(buffer(up.scales), offset: 0, index: 4)
        enc.setBuffer(buffer(up.biases), offset: 0, index: 5)
        enc.setBuffer(buffer(input), offset: 0, index: 6)
        enc.setBuffer(outBuf, offset: 0, index: 7)
        enc.setBytes(&numRows, length: 4, index: 8)
        enc.dispatchThreadgroups(
            MTLSize(width: (rows + 7) / 8, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let out = outBuf.contents().bindMemory(to: UInt16.self, capacity: rows)
        for r in 0..<rows {
            let got = Float(Float16(bitPattern: out[r]))
            XCTAssertEqual(
                got, expected[r], accuracy: max(0.08, abs(expected[r]) * 0.03),
                "row \(r)")
        }
    }
}
