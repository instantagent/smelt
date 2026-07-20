import XCTest
@testable import SmeltCompiler

/// Round-trip + layout tests for the shared affine-u4 numerics (SmeltAffineU4), the one
/// implementation the LLM quantizer and the Qwen3-TTS package builder both use.
final class SmeltAffineU4Tests: XCTestCase {

    /// Deterministic gaussian-ish weights (no Foundation RNG) via a small LCG.
    private func sampleWeights(_ n: Int, scale: Float) -> [Float] {
        var state: UInt64 = 0x9E3779B97F4A7C15
        func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            // top 24 bits → [0,1)
            return Float(state >> 40) / Float(1 << 24)
        }
        // Box–Muller for an approximately normal distribution.
        var out = [Float](repeating: 0, count: n)
        var i = 0
        while i < n {
            let u1 = max(next(), 1e-7), u2 = next()
            let r = (-2.0 * Foundation.log(u1)).squareRoot()
            out[i] = r * Foundation.cos(2 * .pi * u2) * scale
            if i + 1 < n { out[i + 1] = r * Foundation.sin(2 * .pi * u2) * scale }
            i += 2
        }
        return out
    }

    private func relL2(_ a: [Float], _ b: [Float]) -> Float {
        var diff: Float = 0, nb: Float = 0
        for i in 0..<a.count { let d = a[i] - b[i]; diff += d * d; nb += b[i] * b[i] }
        return diff.squareRoot() / nb.squareRoot()
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<a.count { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (na.squareRoot() * nb.squareRoot())
    }

    /// 4-bit affine on weight-like gaussian data should round-trip with small error — the
    /// quality floor the end-to-end TTS gate inherits.
    func testRoundTripGaussianNearFloor() {
        let rows = 64, cols = 2048, g = 64
        let w = sampleWeights(rows * cols, scale: 0.02)
        let packed = SmeltAffineU4.quantize(w, rows: rows, cols: cols, groupSize: g)
        let deq = SmeltAffineU4.dequantize(packed)
        XCTAssertEqual(deq.count, w.count)
        let rl2 = relL2(deq, w), cos = cosine(deq, w)
        // Group-wise 4-bit min/max on *pure gaussian* weights: relL2 ~9%, cosine ~0.996 (outliers
        // stretch the min/max range). This is the per-tensor WEIGHT floor; end-to-end output cosine
        // runs higher because the matmul averages zero-mean per-weight errors over K.
        XCTAssertLessThan(rl2, 0.10, "relL2 \(rl2) too high for 4-bit affine")
        XCTAssertGreaterThan(cos, 0.995, "cosine \(cos) too low for 4-bit affine")
    }

    /// Smaller group sizes strictly reduce error (more scales/biases per row).
    func testSmallerGroupLowersError() {
        let rows = 32, cols = 1024
        let w = sampleWeights(rows * cols, scale: 0.05)
        let e64 = relL2(SmeltAffineU4.dequantize(SmeltAffineU4.quantize(w, rows: rows, cols: cols, groupSize: 64)), w)
        let e32 = relL2(SmeltAffineU4.dequantize(SmeltAffineU4.quantize(w, rows: rows, cols: cols, groupSize: 32)), w)
        XCTAssertLessThanOrEqual(e32, e64 + 1e-6, "g=32 (\(e32)) should not exceed g=64 (\(e64))")
    }

    /// Exact nibble packing/layout: a single group [0..15] maps to nibbles 0..15, low-first.
    func testExactPackingLayout() {
        let cols = 16, g = 16
        let w = (0..<cols).map { Float($0) }   // min 0, max 15 → scale 1, bias 0, nibble == value
        let packed = SmeltAffineU4.quantize(w, rows: 1, cols: cols, groupSize: g)
        XCTAssertEqual(packed.nibbles.count, 8)
        for byteIdx in 0..<8 {
            let lo = packed.nibbles[byteIdx] & 0x0F
            let hi = packed.nibbles[byteIdx] >> 4
            XCTAssertEqual(Int(lo), 2 * byteIdx, "low nibble of byte \(byteIdx)")
            XCTAssertEqual(Int(hi), 2 * byteIdx + 1, "high nibble of byte \(byteIdx)")
        }
        // scale=1, bias=0 → exact dequant
        XCTAssertEqual(SmeltAffineU4.dequantize(packed), w)
    }

    /// MSE-optimal clip is a strict improvement over min/max: clamping a few outliers buys finer bulk
    /// resolution, so reconstruction relL2 never gets worse and typically improves on heavy-tailed data.
    func testMSEClipBeatsMinMax() {
        let rows = 64, cols = 1024, g = 64
        // Heavy-tailed: gaussian bulk + occasional large outliers (where min/max wastes range).
        var w = sampleWeights(rows * cols, scale: 0.02)
        var st: UInt64 = 0xDEADBEEF
        for i in 0..<w.count {
            st = st &* 6364136223846793005 &+ 1
            if st >> 60 == 0 { w[i] = (Int(st >> 32) & 1 == 0 ? 1 : -1) * 0.5 }   // ~6% outliers at ±0.5
        }
        let eMinMax = relL2(SmeltAffineU4.dequantize(SmeltAffineU4.quantize(w, rows: rows, cols: cols, groupSize: g, clip: .minMax)), w)
        let eMse = relL2(SmeltAffineU4.dequantize(SmeltAffineU4.quantize(w, rows: rows, cols: cols, groupSize: g, clip: .mseOptimal)), w)
        // The c=1.0 candidate IS min/max, so mseOptimal can only improve. The synthetic gain is modest
        // (~1%); the real gain is end-to-end logit fidelity on the checkpoint, not weight relL2 here.
        XCTAssertLessThanOrEqual(eMse, eMinMax + 1e-6, "MSE-optimal (\(eMse)) must not exceed min/max (\(eMinMax))")
        XCTAssertLessThan(eMse, eMinMax, "MSE-optimal should improve on heavy-tailed data (\(eMse) vs \(eMinMax))")
    }

    /// Activation-weighted clip minimizes Σ h_k·(w−q)² (output error), so on data where the high-error
    /// channels are also the high-importance ones, the weighted reconstruction error drops vs min/max.
    func testActivationWeightedClipLowersWeightedError() {
        let rows = 32, cols = 256, g = 64
        let w = sampleWeights(rows * cols, scale: 0.02)
        // Importance concentrated on a few channels; the quantizer should protect those.
        var imp = [Float](repeating: 0.1, count: cols)
        for k in stride(from: 0, to: cols, by: 17) { imp[k] = 10 }
        func weightedErr(_ deq: [Float]) -> Float {
            var e: Float = 0
            for r in 0..<rows { for k in 0..<cols { let d = deq[r * cols + k] - w[r * cols + k]; e += imp[k] * d * d } }
            return e
        }
        let mm = weightedErr(SmeltAffineU4.dequantize(SmeltAffineU4.quantize(w, rows: rows, cols: cols, groupSize: g, clip: .minMax)))
        let aw = weightedErr(SmeltAffineU4.dequantize(SmeltAffineU4.quantize(w, rows: rows, cols: cols, groupSize: g, clip: .mseOptimal, importance: imp)))
        XCTAssertLessThan(aw, mm, "activation-weighted clip should lower the importance-weighted error (\(aw) vs \(mm))")
    }

    /// A constant group (range < 1e-12) uses scale=1, bias=value → reconstructs exactly.
    func testConstantGroup() {
        let cols = 64
        let w = [Float](repeating: 0.123, count: cols)
        let packed = SmeltAffineU4.quantize(w, rows: 1, cols: cols, groupSize: 64)
        let deq = SmeltAffineU4.dequantize(packed)
        // bias = 0.123 stored as fp16; nibble*1+bias with nibble=0 → fp16(0.123).
        let expected = Float(Float16(0.123))
        for v in deq { XCTAssertEqual(v, expected, accuracy: 1e-6) }
    }
}
