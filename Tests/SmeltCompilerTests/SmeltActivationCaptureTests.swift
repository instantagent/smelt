import Foundation
import Metal
import Testing
@testable import SmeltRuntime

// The GPTQ capture math is model-independent: per-channel E[x²] and the activation Hessian
// H = Σ x xᵀ are pure functions of the captured rows. These check that against a plain CPU
// reference, and that the FP16 ingestion path matches FP32 on the same values.

private func device() -> MTLDevice? { MTLCreateSystemDefaultDevice() }

private func fp32Buffer(_ rows: [Float], _ dev: MTLDevice) throws -> MTLBuffer {
    try makeSharedBuffer(device: dev, rows)
}

private func fp16Buffer(_ rows: [Float], _ dev: MTLDevice) throws -> MTLBuffer {
    try makeSharedBuffer(device: dev, rows.map { Float16($0).bitPattern })
}

/// Reference full symmetric Hessian H[i,j] = Σ_row x[row,i]·x[row,j] over `m` rows of width `k`.
private func referenceHessian(_ rows: [Float], m: Int, k: Int) -> [Float] {
    var h = [Float](repeating: 0, count: k * k)
    for r in 0..<m {
        let base = r * k
        for i in 0..<k {
            for j in 0..<k { h[i * k + j] += rows[base + i] * rows[base + j] }
        }
    }
    return h
}

@Test func captureImatrixMatchesMeanSquare() throws {
    guard let dev = device() else { return }
    let (m, k) = (4, 5)
    let rows: [Float] = (0..<(m * k)).map { Float($0 % 7) - 3.0 }
    let cap = SmeltActivationCapture()
    cap.accumulate("w", try fp32Buffer(rows, dev), m: m, k: k)

    let imp = try #require(cap.importance("w"))
    #expect(cap.calibrationRows("w") == m)
    for c in 0..<k {
        var s: Float = 0
        for r in 0..<m { let v = rows[r * k + c]; s += v * v }
        #expect(abs(imp[c] - s / Float(m)) < 1e-5)
    }
}

@Test func captureHessianMatchesCPUReference() throws {
    guard let dev = device() else { return }
    let (m, k) = (6, 8)
    let rows: [Float] = (0..<(m * k)).map { sinf(Float($0) * 0.37) }
    let cap = SmeltActivationCapture()
    cap.captureHessian = true
    cap.accumulate("w", try fp32Buffer(rows, dev), m: m, k: k)

    let h = try #require(cap.hessian("w"))
    #expect(h.count == k * k)
    let ref = referenceHessian(rows, m: m, k: k)
    // ssyrk reduction order differs from the naive triple loop — compare with a small tolerance.
    var maxErr: Float = 0
    for i in 0..<(k * k) { maxErr = max(maxErr, abs(h[i] - ref[i])) }
    #expect(maxErr < 1e-4, "max |H - ref| = \(maxErr)")
    // hessian() must hand off a FULL symmetric matrix (GPTQ reads the lower triangle).
    for i in 0..<k { for j in 0..<k { #expect(abs(h[i * k + j] - h[j * k + i]) < 1e-6) } }
}

@Test func captureFlushesAcrossManyRows() throws {
    guard let dev = device() else { return }
    // > hessFlushRows (128) so the batched ssyrk flush path runs mid-accumulation, then again on handoff.
    let (m, k) = (300, 4)
    let rows: [Float] = (0..<(m * k)).map { cosf(Float($0) * 0.11) }
    let cap = SmeltActivationCapture()
    cap.captureHessian = true
    // Feed in two chunks to exercise pending-buffer carryover across calls.
    cap.accumulate("w", try fp32Buffer(Array(rows[0..<(150 * k)]), dev), m: 150, k: k)
    cap.accumulate("w", try fp32Buffer(Array(rows[(150 * k)...]), dev), m: 150, k: k)

    let h = try #require(cap.hessian("w"))
    let ref = referenceHessian(rows, m: m, k: k)
    var maxRel: Float = 0
    for i in 0..<(k * k) { maxRel = max(maxRel, abs(h[i] - ref[i]) / (abs(ref[i]) + 1e-3)) }
    #expect(maxRel < 1e-3, "max rel err = \(maxRel)")
}

@Test func captureFP16PathMatchesFP32OnHalfValues() throws {
    guard let dev = device() else { return }
    let (m, k) = (5, 6)
    // Both paths see the SAME fp16-rounded inputs: the fp32 path gets pre-rounded values, the fp16
    // path gets the raw half bits — so the captured stats must agree.
    let raw: [Float] = (0..<(m * k)).map { Float($0) * 0.013 - 0.2 }
    let rounded = raw.map { Float(Float16($0)) }

    let ref = SmeltActivationCapture(); ref.captureHessian = true
    ref.accumulate("w", try fp32Buffer(rounded, dev), m: m, k: k)

    let f16 = SmeltActivationCapture(); f16.captureHessian = true
    f16.accumulate("w", try fp16Buffer(raw, dev), m: m, k: k, inputIsFloat16: true)

    let hRef = try #require(ref.hessian("w"))
    let hF16 = try #require(f16.hessian("w"))
    for i in 0..<(k * k) { #expect(abs(hRef[i] - hF16[i]) < 1e-5) }
    let impRef = try #require(ref.importance("w"))
    let impF16 = try #require(f16.importance("w"))
    for c in 0..<k { #expect(abs(impRef[c] - impF16[c]) < 1e-6) }
}

@Test func captureHessianNamesRestrictsHessianButNotImatrix() throws {
    guard let dev = device() else { return }
    let (m, k) = (3, 4)
    let rows: [Float] = (0..<(m * k)).map { Float($0) }
    let cap = SmeltActivationCapture()
    cap.captureHessian = true
    cap.captureHessianNames = ["wanted"]
    cap.accumulate("wanted", try fp32Buffer(rows, dev), m: m, k: k)
    cap.accumulate("other", try fp32Buffer(rows, dev), m: m, k: k)

    #expect(cap.hessian("wanted") != nil)
    #expect(cap.hessian("other") == nil)        // out of the capture set — no full Hessian
    #expect(cap.importance("wanted") != nil)
    #expect(cap.importance("other") != nil)     // imatrix diagonal still captured for all
}
