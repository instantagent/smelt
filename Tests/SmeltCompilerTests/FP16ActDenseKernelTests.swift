// FP16ActDenseKernelTests — U2 of docs/dtype-building-blocks-plan.md: the hand-written fp16-
// ACTIVATION dense matvec kernels for bf16 / fp32 WEIGHTS, gated vs a CPU reference.
//
// These fill the fp16-act × {bf16,fp32} dense HOLES the gateway tracks in knownMissing — a bf16
// (the checkpoint's native dtype) or fp32 projection weight becomes a kernel LEGO in the fp16-act
// LLM path. The kernels are the fp16_matvec SHAPE (fp16 input + output, threadgroup reduction,
// half clamp) with the weight load swapped; only the reduction ORDER differs from a sequential CPU
// sum, so the gate is per-element ULP-bounded — NOT bit-exact (fp16_matvec's simd_sum + threadgroup
// partials are GPU-deterministic but not a portable CPU contract, plan U2 gate note).
//
// Coverage (codex U2a review, commit 11bf6ed → SHIP-WITH-CHANGES):
//   - SIGNED inputs/weights, not just positive [0,1] — real post-norm activations are signed, so a
//     sign/cancellation bug must be exercised.
//   - The PRODUCTION launch geometry: emitFP16Matvec dispatches tgWidth=256 while MATVEC_TPG=64
//     (SmeltCodeEmitter.swift:1629, lut_matvec.metal:17). At 256 threads, simdgroups 0-1 (threads
//     0-63, stride 64) tile all of cols4; threads 64-255 redo subsets into simdgroups 2-7 which the
//     final `s < MATVEC_TPG/32` reduction DISCARDS. Correct-but-wasteful — but only true for cols4 >
//     64 (cols > 256), so a large-C @ tgWidth=256 case is what proves it for these kernels (the
//     decode tests at tgWidth=64 never reach the discard path).
//   - A PRINCIPLED per-element tolerance: ≤2 fp16-output ULP + an fp32 reduction-reorder bound, NOT
//     a flat maxAbs smoke threshold (which was ~4 ULP, looser than the <0.5 ULP observed).

import Metal
import XCTest

@testable import SmeltCompiler

final class FP16ActDenseKernelTests: XCTestCase {
    /// Deterministic values. `signed` swings [-1,1] (real activations); otherwise positive-biased
    /// [0,1] so a row dot has a clear magnitude that clears the zero-tripwire.
    private func values(_ n: Int, seed: Int, signed: Bool) -> [Float] {
        (0..<n).map { i in
            let s = Float(sin(Double((i &* 12_347) &+ seed &* 7919) * 0.001))
            return signed ? s : 0.5 + 0.5 * s
        }
    }

    /// bf16 = the top 16 bits of an fp32 (mantissa truncated); the GPU widens it back exactly, so
    /// the CPU reference reads the SAME stored value via the inverse widen.
    private func bf16Bits(_ f: Float) -> UInt16 { UInt16(truncatingIfNeeded: f.bitPattern >> 16) }
    private func widenBF16(_ b: UInt16) -> Float { Float(bitPattern: UInt32(b) << 16) }

    /// fp16 unit-in-the-last-place at magnitude `v` (10 mantissa bits → ULP = 2^(exp-10)).
    private func ulpFP16(_ v: Float) -> Float {
        let a = abs(v)
        guard a > 0, a.isFinite else { return Float(Float16.leastNonzeroMagnitude) }
        return exp2(floor(log2(a)) - 10)
    }

    /// One threadgroup-per-row dispatch of a fp16-act dense matvec, returning the half output rows.
    private func runMatvec(
        device: MTLDevice, queue: MTLCommandQueue, functionName: String,
        weightBuf: MTLBuffer, inputBuf: MTLBuffer, rows: Int, cols: Int, tgWidth: Int
    ) throws -> [Float16] {
        let pipeline = try XCTUnwrap(
            makeComputePipeline(device: device, shaderFile: "lut_matvec.metal",
                                functionName: functionName))
        let outputBuf = try makeSharedBuffer(device: device, count: rows, of: Float16.self)
        try runOnGPU(queue: queue) { enc in
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(weightBuf, offset: 0, index: 0)
            enc.setBuffer(inputBuf, offset: 0, index: 1)
            enc.setBuffer(outputBuf, offset: 0, index: 2)
            var c = UInt32(cols)
            enc.setBytes(&c, length: 4, index: 3)
            enc.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1))
        }
        return Array(UnsafeBufferPointer(
            start: outputBuf.contents().assumingMemoryBound(to: Float16.self), count: rows))
    }

    /// GPU vs a CPU float-accumulate reference, per-element tolerance = 2 fp16-output ULP + an fp32
    /// reduction-reorder bound (C·2⁻²³·maxAbs). Tight enough to flag a wrong weight dtype (reading
    /// bf16 bytes as half), loose enough to absorb the GPU tree-reduction order.
    private func assertMatches(
        gpu: [Float16], widenWeight: (Int) -> Float, x: [Float16], rows: Int, cols: Int,
        label: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        var cpu = [Float](repeating: 0, count: rows)
        for r in 0..<rows {
            var total: Float = 0
            for c in 0..<cols { total += widenWeight(r * cols + c) * Float(x[c]) }
            cpu[r] = max(-65504, min(65504, total))
        }
        let maxAbs = cpu.map(abs).max() ?? 1
        XCTAssertGreaterThan(maxAbs, 1e-3, "\(label): output is all ~zero — kernel may not have run",
                             file: file, line: line)
        let reorder = Float(cols) * 0x1p-23 * maxAbs
        for r in 0..<rows {
            let tol = 2 * ulpFP16(cpu[r]) + reorder
            let diff = abs(Float(gpu[r]) - cpu[r])
            XCTAssertLessThanOrEqual(
                diff, tol,
                "\(label): row \(r) diff \(diff) > tol \(tol) (gpu \(Float(gpu[r])) cpu \(cpu[r]))",
                file: file, line: line)
        }
    }

    // (rows, cols, tgWidth, signed). cols=258 → 64 float4 chunks + a 2-element scalar tail @ the
    // decode geometry (tgWidth 64). cols=1028 @ tgWidth 256 → cols4=257 > 64, the production launch
    // that exercises the simdgroup-2..7 discard path. Signed everywhere except one positive baseline.
    private let cases: [(rows: Int, cols: Int, tg: Int, signed: Bool)] = [
        (9, 258, 64, false),
        (9, 258, 64, true),
        (5, 1028, 256, true),
    ]

    func testFP16MatvecBF16WMatchesCPUReference() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        for (i, tc) in cases.enumerated() {
            let wF = values(tc.rows * tc.cols, seed: 100 + i, signed: tc.signed)
            let wBits = wF.map(bf16Bits)
            let xH = values(tc.cols, seed: 200 + i, signed: tc.signed).map { Float16($0) }
            let weightBuf = try makeSharedBuffer(device: device, wBits)
            let inputBuf = try makeSharedBuffer(device: device, xH)
            let gpu = try runMatvec(device: device, queue: queue, functionName: "fp16_matvec_bf16w",
                                    weightBuf: weightBuf, inputBuf: inputBuf,
                                    rows: tc.rows, cols: tc.cols, tgWidth: tc.tg)
            assertMatches(gpu: gpu, widenWeight: { widenBF16(wBits[$0]) }, x: xH,
                          rows: tc.rows, cols: tc.cols,
                          label: "fp16_matvec_bf16w[\(tc.rows),\(tc.cols)]@tg\(tc.tg)")
        }
    }

    func testFP16MatvecFP32WMatchesCPUReference() throws {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let queue = try XCTUnwrap(device.makeCommandQueue())
        for (i, tc) in cases.enumerated() {
            let wF = values(tc.rows * tc.cols, seed: 300 + i, signed: tc.signed)
            let xH = values(tc.cols, seed: 400 + i, signed: tc.signed).map { Float16($0) }
            let weightBuf = try makeSharedBuffer(device: device, wF)
            let inputBuf = try makeSharedBuffer(device: device, xH)
            let gpu = try runMatvec(device: device, queue: queue, functionName: "fp16_matvec_fp32w",
                                    weightBuf: weightBuf, inputBuf: inputBuf,
                                    rows: tc.rows, cols: tc.cols, tgWidth: tc.tg)
            assertMatches(gpu: gpu, widenWeight: { wF[$0] }, x: xH,
                          rows: tc.rows, cols: tc.cols,
                          label: "fp16_matvec_fp32w[\(tc.rows),\(tc.cols)]@tg\(tc.tg)")
        }
    }
}
