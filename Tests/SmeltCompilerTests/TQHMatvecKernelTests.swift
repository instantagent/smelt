// Numerical parity test for the tqh_matvec kernel pair against the
// CPU codec path. The two are algebraically non-identical:
//   - codec dequantizes W to fp16 row-by-row, then a downstream
//     matvec does fp32(fp16 W[r,c]) * fp32(x[c]) accumulated in fp32.
//   - kernel keeps fp32 through the matvec without ever rounding
//     individual W[r,c] to fp16.
// Decode α requires them to agree within fp16 ulp on the
// resulting logits — both cosine and max-abs-diff. Without these
// asserts, a missed `inv_sqrt_G` (which would scale every output
// by ~11.3× and preserve cosine ≈ 1.0) or a single-row sign flip
// would land green at unit test time but collapse α in the live
// build.

import Foundation
import Metal
import XCTest
@testable import SmeltCompiler
import SmeltSchema

final class TQHMatvecKernelTests: XCTestCase {
    private var device: MTLDevice!
    private var queue: MTLCommandQueue!

    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        try? XCTSkipIf(device == nil, "No Metal device available")
        queue = device.makeCommandQueue()
    }

    func testTQHMatvecMatchesCodecMatvec() throws {
        try runParityCheck(rows: 64, cols: 256, groupSize: 128)
    }

    func testTQHMatvecPartialFinalGroup() throws {
        try runParityCheck(rows: 32, cols: 130, groupSize: 128)
    }

    func testTQHMatvecFFNScaleCols() throws {
        // Wide FFN down_proj shape: W is [hidden, ffnDim] = [2560, 10240],
        // so the matvec input dim is cols=10240. Make sure the kernel
        // produces correct results at this scale.
        try runParityCheck(rows: 256, cols: 10240, groupSize: 128)
    }

    func testTQHMatvecBatchedMatchesCodecMatvec() throws {
        try runBatchedParityCheck(rows: 64, cols: 256, groupSize: 128, batchSize: 5)
    }

    func testTQHMatvecBatchedFFNScale() throws {
        try runBatchedParityCheck(rows: 256, cols: 10240, groupSize: 128, batchSize: 5)
    }

    private func runBatchedParityCheck(
        rows: Int, cols: Int, groupSize: Int, batchSize: Int
    ) throws {
        var rng = SplitMix64(seed: 42)
        var w = [Float16](repeating: 0, count: rows * cols)
        for k in 0 ..< rows * cols {
            let u1 = rng.nextUnitDouble()
            let u2 = rng.nextUnitDouble()
            let r = (-2.0 * Foundation.log(max(u1, 1e-12))).squareRoot()
            w[k] = Float16(r * Foundation.cos(2.0 * Double.pi * u2) * 0.1)
        }

        let (codes, codebook) = w.withUnsafeBufferPointer { buf in
            SmeltTurboQuantHQuantizer.quantize(
                weights: buf.baseAddress!,
                rows: rows, cols: cols, groupSize: groupSize, seed: 0
            )
        }
        let numGroups = SmeltTurboQuantHCodec.numGroups(cols: cols, groupSize: groupSize)
        let codesPerRow = ((numGroups * groupSize) + 3) / 4

        // Build B distinct inputs.
        var x = [Float16](repeating: 0, count: batchSize * cols)
        for b in 0 ..< batchSize {
            for i in 0 ..< cols {
                let phase = Float(i) * 0.0731 + Float(b) * 0.137
                x[b * cols + i] = Float16(Foundation.sin(phase) * 0.5)
            }
        }

        var yRef = [Float16](repeating: 0, count: batchSize * rows)
        let codebookBits = codebook.map { $0.bitPattern }
        var wRow = [UInt16](repeating: 0, count: cols)
        for r in 0 ..< rows {
            codes.withUnsafeBufferPointer { codeBuf in
                codebookBits.withUnsafeBufferPointer { cbBuf in
                    wRow.withUnsafeMutableBufferPointer { outBuf in
                        SmeltTurboQuantHCodec.dequantizeRowInto(
                            codes: codeBuf.baseAddress! + r * codesPerRow,
                            codebook: cbBuf.baseAddress!,
                            output: outBuf.baseAddress!,
                            cols: cols,
                            groupSize: groupSize
                        )
                    }
                }
            }
            for b in 0 ..< batchSize {
                var acc: Float = 0
                for c in 0 ..< cols {
                    let wf = Float(Float16(bitPattern: wRow[c]))
                    acc += wf * Float(x[b * cols + c])
                }
                yRef[b * rows + r] = Float16(acc)
            }
        }

        guard let preparePipeline = makeComputePipeline(
            device: device,
            shaderFile: "lut_matvec.metal",
            functionName: "tqh_matvec_prepare_input_batched"
        ),
        let matvecPipeline = makeComputePipeline(
            device: device,
            shaderFile: "lut_matvec.metal",
            functionName: "tqh_matvec_batched"
        ) else {
            throw XCTSkip("Failed to compile batched TQH matvec kernels")
        }

        let codesBuf = try makeSharedBuffer(device: device, codes)
        let codebookBuf = try makeSharedBuffer(device: device, codebook)
        let inputBuf = try makeSharedBuffer(device: device, x)
        let xHatBuf = try makeSharedBuffer(
            device: device,
            count: batchSize * numGroups * groupSize, of: Float.self
        )
        let outputBuf = try makeSharedBuffer(
            device: device, count: batchSize * rows, of: Float16.self
        )

        try runOnGPU(queue: queue) { encoder in
            encoder.setComputePipelineState(preparePipeline)
            encoder.setBuffer(inputBuf, offset: 0, index: 0)
            encoder.setBuffer(xHatBuf, offset: 0, index: 1)
            var colsVal = UInt32(cols)
            encoder.setBytes(&colsVal, length: 4, index: 2)
            encoder.dispatchThreads(
                MTLSize(width: numGroups * groupSize, height: batchSize, depth: 1),
                threadsPerThreadgroup: MTLSize(width: groupSize, height: 1, depth: 1)
            )

            encoder.setComputePipelineState(matvecPipeline)
            encoder.setBuffer(codesBuf, offset: 0, index: 0)
            encoder.setBuffer(codebookBuf, offset: 0, index: 1)
            encoder.setBuffer(xHatBuf, offset: 0, index: 2)
            encoder.setBuffer(outputBuf, offset: 0, index: 3)
            var numRowsVal = UInt32(rows)
            encoder.setBytes(&numRowsVal, length: 4, index: 4)
            encoder.setBytes(&colsVal, length: 4, index: 5)
            var cprVal = UInt32(codesPerRow)
            encoder.setBytes(&cprVal, length: 4, index: 6)
            let rowsPerTG = 16
            let threadsPerTG = 128
            let numTGs = (rows + rowsPerTG - 1) / rowsPerTG
            let dispatchWidth = numTGs * threadsPerTG
            encoder.dispatchThreads(
                MTLSize(width: dispatchWidth, height: batchSize, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerTG, height: 1, depth: 1)
            )
        }

        let yGPU = Array(UnsafeBufferPointer(
            start: outputBuf.contents().assumingMemoryBound(to: Float16.self),
            count: batchSize * rows
        ))

        var dot: Double = 0
        var na: Double = 0
        var nb: Double = 0
        var maxAbs: Float = 0
        for i in 0 ..< batchSize * rows {
            let a = Double(yRef[i])
            let b = Double(yGPU[i])
            dot += a * b
            na += a * a
            nb += b * b
            let d = abs(Float(yRef[i]) - Float(yGPU[i]))
            if d > maxAbs { maxAbs = d }
        }
        let cos = dot / ((na * nb).squareRoot() + 1e-12)
        let refMaxAbs = yRef.map { abs(Float($0)) }.max() ?? 0
        XCTAssertGreaterThan(
            cos, 0.9999,
            "[batched \(batchSize)x\(rows)x\(cols)] cosine \(cos), maxAbs \(maxAbs)"
        )
        XCTAssertLessThan(
            maxAbs, 0.05 * refMaxAbs,
            "[batched \(batchSize)x\(rows)x\(cols)] maxAbs \(maxAbs) > 5% of refMaxAbs \(refMaxAbs)"
        )
    }

    private func runParityCheck(rows: Int, cols: Int, groupSize: Int) throws {
        var rng = SplitMix64(seed: 42)
        var w = [Float16](repeating: 0, count: rows * cols)
        for k in 0 ..< rows * cols {
            let u1 = rng.nextUnitDouble()
            let u2 = rng.nextUnitDouble()
            let r = (-2.0 * Foundation.log(max(u1, 1e-12))).squareRoot()
            w[k] = Float16(r * Foundation.cos(2.0 * Double.pi * u2) * 0.1)
        }

        let (codes, codebook) = w.withUnsafeBufferPointer { buf in
            SmeltTurboQuantHQuantizer.quantize(
                weights: buf.baseAddress!,
                rows: rows, cols: cols, groupSize: groupSize,
                seed: 0
            )
        }
        let numGroups = SmeltTurboQuantHCodec.numGroups(cols: cols, groupSize: groupSize)
        let codesPerRow = ((numGroups * groupSize) + 3) / 4

        var x = [Float16](repeating: 0, count: cols)
        for i in 0 ..< cols {
            x[i] = Float16(Foundation.sin(Float(i) * 0.0731) * 0.5)
        }

        var yRef = [Float16](repeating: 0, count: rows)
        let codebookBits = codebook.map { $0.bitPattern }
        var wRow = [UInt16](repeating: 0, count: cols)
        for r in 0 ..< rows {
            codes.withUnsafeBufferPointer { codeBuf in
                codebookBits.withUnsafeBufferPointer { cbBuf in
                    wRow.withUnsafeMutableBufferPointer { outBuf in
                        SmeltTurboQuantHCodec.dequantizeRowInto(
                            codes: codeBuf.baseAddress! + r * codesPerRow,
                            codebook: cbBuf.baseAddress!,
                            output: outBuf.baseAddress!,
                            cols: cols,
                            groupSize: groupSize
                        )
                    }
                }
            }
            var acc: Float = 0
            for c in 0 ..< cols {
                let wf = Float(Float16(bitPattern: wRow[c]))
                acc += wf * Float(x[c])
            }
            yRef[r] = Float16(acc)
        }

        guard let preparePipeline = makeComputePipeline(
            device: device,
            shaderFile: "lut_matvec.metal",
            functionName: "tqh_matvec_prepare_input"
        ),
        let matvecPipeline = makeComputePipeline(
            device: device,
            shaderFile: "lut_matvec.metal",
            functionName: "tqh_matvec"
        ) else {
            throw XCTSkip("Failed to compile TQH matvec kernels")
        }

        let codesBuf = try makeSharedBuffer(device: device, codes)
        let codebookBuf = try makeSharedBuffer(device: device, codebook)
        let inputBuf = try makeSharedBuffer(device: device, x)
        let xHatBuf = try makeSharedBuffer(
            device: device, count: numGroups * groupSize, of: Float.self
        )
        let outputBuf = try makeSharedBuffer(
            device: device, count: rows, of: Float16.self
        )

        try runOnGPU(queue: queue) { encoder in
            encoder.setComputePipelineState(preparePipeline)
            encoder.setBuffer(inputBuf, offset: 0, index: 0)
            encoder.setBuffer(xHatBuf, offset: 0, index: 1)
            var colsVal = UInt32(cols)
            encoder.setBytes(&colsVal, length: 4, index: 2)
            encoder.dispatchThreads(
                MTLSize(width: numGroups * groupSize, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: groupSize, height: 1, depth: 1)
            )

            encoder.setComputePipelineState(matvecPipeline)
            encoder.setBuffer(codesBuf, offset: 0, index: 0)
            encoder.setBuffer(codebookBuf, offset: 0, index: 1)
            encoder.setBuffer(xHatBuf, offset: 0, index: 2)
            encoder.setBuffer(outputBuf, offset: 0, index: 3)
            var numRowsVal = UInt32(rows)
            encoder.setBytes(&numRowsVal, length: 4, index: 4)
            encoder.setBytes(&colsVal, length: 4, index: 5)
            var cprVal = UInt32(codesPerRow)
            encoder.setBytes(&cprVal, length: 4, index: 6)
            let rowsPerTG = 16
            let threadsPerTG = 128
            let numTGs = (rows + rowsPerTG - 1) / rowsPerTG
            let dispatchWidth = numTGs * threadsPerTG
            encoder.dispatchThreads(
                MTLSize(width: dispatchWidth, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threadsPerTG, height: 1, depth: 1)
            )
        }

        let yGPU = Array(UnsafeBufferPointer(
            start: outputBuf.contents().assumingMemoryBound(to: Float16.self),
            count: rows
        ))

        var dot: Double = 0
        var na: Double = 0
        var nb: Double = 0
        var maxAbs: Float = 0
        for r in 0 ..< rows {
            let a = Double(yRef[r])
            let b = Double(yGPU[r])
            dot += a * b
            na += a * a
            nb += b * b
            let d = abs(Float(yRef[r]) - Float(yGPU[r]))
            if d > maxAbs { maxAbs = d }
        }
        let cos = dot / ((na * nb).squareRoot() + 1e-12)
        let refMaxAbs = yRef.map { abs(Float($0)) }.max() ?? 0
        XCTAssertGreaterThan(
            cos, 0.9999,
            "[\(rows)x\(cols)] cosine \(cos), maxAbs \(maxAbs)"
        )
        // Magnitude check: catches missed inv_sqrt_G or per-row sign
        // flips that preserve direction. Threshold = 0.05 * refMaxAbs,
        // i.e. 5% of the reference's largest fp16 logit magnitude.
        XCTAssertLessThan(
            maxAbs, 0.05 * refMaxAbs,
            "[\(rows)x\(cols)] maxAbs \(maxAbs) > 5% of refMaxAbs \(refMaxAbs)"
        )
    }
}
