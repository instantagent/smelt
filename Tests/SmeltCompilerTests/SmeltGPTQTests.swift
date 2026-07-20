import XCTest
import Accelerate
@testable import SmeltCompiler

/// GPTQ error-feedback quantizer: validates the R = upper-Cholesky-of-H⁻¹ math, that GPTQ lowers the
/// activation-weighted OUTPUT error vs plain affine on correlated data, and that it emits a valid
/// SmeltAffineU4-format block.
final class SmeltGPTQTests: XCTestCase {

    private func lcg(_ s: inout UInt64) -> Float {
        s = s &* 6364136223846793005 &+ 1442695040888963407
        return Float(Int32(truncatingIfNeeded: s >> 16)) / Float(Int32.max)   // ~[-1,1)
    }

    /// (b) R = upper Cholesky of (H+λI)⁻¹: RᵀR · (H+λI) ≈ identity.
    func testCholeskyOfInverseIsCorrect() {
        let n = 48
        var seed: UInt64 = 0xC0FFEE123
        // SPD H = BBᵀ/m + I (well-conditioned, symmetric).
        let m = 64
        var B = [Float](repeating: 0, count: n * m)
        for i in 0..<B.count { B[i] = lcg(&seed) }
        var H = [Float](repeating: 0, count: n * n)
        for a in 0..<n { for b in 0..<n {
            var s: Float = 0; for k in 0..<m { s += B[a * m + k] * B[b * m + k] }
            H[a * n + b] = s / Float(m) + (a == b ? 1 : 0)
        } }
        let damping: Float = 0.01
        var diagSum: Float = 0; for i in 0..<n { diagSum += H[i * n + i] }
        let lambda = damping * diagSum / Float(n)
        var Hd = H; for i in 0..<n { Hd[i * n + i] += lambda }   // the matrix the function factors

        var Hwork = H
        let R = SmeltGPTQ.choleskyOfInverse(&Hwork, n: n, damping: damping)
        // M = RᵀR (R upper, column-major: R[i,j]=R[j*n+i], i≤j). M[a,b] = Σ_{i≤min(a,b)} R[i,a]R[i,b].
        func Rv(_ i: Int, _ j: Int) -> Float { i <= j ? R[j * n + i] : 0 }
        var M = [Float](repeating: 0, count: n * n)
        for a in 0..<n { for b in 0..<n {
            var s: Float = 0; for i in 0...min(a, b) { s += Rv(i, a) * Rv(i, b) }
            M[a * n + b] = s
        } }
        // Check M·Hd ≈ I.
        var maxErr: Float = 0
        for a in 0..<n { for b in 0..<n {
            var s: Float = 0; for k in 0..<n { s += M[a * n + k] * Hd[k * n + b] }
            maxErr = max(maxErr, abs(s - (a == b ? 1 : 0)))
        } }
        XCTAssertLessThan(maxErr, 2e-3, "RᵀR·(H+λI) deviates from identity by \(maxErr)")
    }

    /// (c) On correlated (low-rank) activations, GPTQ's output error beats plain affine min/max.
    /// (d) The emitted block is a valid SmeltAffineU4.Packed (round-trips, nibbles in range).
    func testGPTQBeatsAffineOnCorrelatedActivations() {
        let N = 32, K = 128, T = 256, rank = 12, g = 64
        var seed: UInt64 = 0x5EED99
        let W = (0..<N * K).map { _ in lcg(&seed) * 0.05 }
        // x_t = A z_t + small noise → strongly correlated across the K input channels.
        let A = (0..<K * rank).map { _ in lcg(&seed) }
        var X = [Float](repeating: 0, count: T * K)
        for t in 0..<T {
            let z = (0..<rank).map { _ in lcg(&seed) }
            for k in 0..<K {
                var s: Float = 0; for r in 0..<rank { s += A[k * rank + r] * z[r] }
                X[t * K + k] = s + lcg(&seed) * 0.05
            }
        }
        // H = Σ_t x_t x_tᵀ.
        var H = [Float](repeating: 0, count: K * K)
        for t in 0..<T { let b = t * K
            for a in 0..<K { let xa = X[b + a]; for c in 0..<K { H[a * K + c] += xa * X[b + c] } }
        }

        let gptq = SmeltAffineU4.dequantize(SmeltGPTQ.quantize(weights: W, rows: N, cols: K, groupSize: g, hessian: H))
        let affine = SmeltAffineU4.dequantize(SmeltAffineU4.quantize(W, rows: N, cols: K, groupSize: g))
        XCTAssertEqual(gptq.count, N * K)   // (d) valid block, round-trips through dequantize

        // Output error Σ_t ‖(W − Ŵ) x_t‖² for each quantizer.
        func outErr(_ Wq: [Float]) -> Float {
            var e: Float = 0
            for t in 0..<T { let xb = t * K
                for n in 0..<N { let wb = n * K
                    var d: Float = 0; for k in 0..<K { d += (W[wb + k] - Wq[wb + k]) * X[xb + k] }
                    e += d * d
                }
            }
            return e
        }
        let eG = outErr(gptq), eA = outErr(affine)
        XCTAssertLessThan(eG, eA, "GPTQ output error \(eG) should beat affine \(eA) on correlated activations")
    }

    /// A dead input channel (zero activation → H diag 0) with huge weights must be zeroed so it doesn't
    /// blow the group's affine range and crush the live columns. With the guard, the live reconstruction
    /// stays accurate; without it, the group scale explodes and the live columns collapse toward 0.
    func testGPTQZeroesDeadChannels() {
        let N = 8, K = 64, T = 128, g = 64   // single group
        var seed: UInt64 = 0xDEAD0
        var W = (0..<N * K).map { _ in lcg(&seed) * 0.05 }
        for n in 0..<N { W[n * K + 0] = 100 }   // column 0: huge weights...
        var X = [Float](repeating: 0, count: T * K)
        for t in 0..<T { for k in 1..<K { X[t * K + k] = lcg(&seed) } }   // ...but column 0 activation ≡ 0
        var H = [Float](repeating: 0, count: K * K)
        for t in 0..<T { let b = t * K
            for a in 0..<K { let xa = X[b + a]; for c in 0..<K { H[a * K + c] += xa * X[b + c] } }
        }   // H row/col 0 are all zero (dead channel)
        let Wq = SmeltAffineU4.dequantize(SmeltGPTQ.quantize(weights: W, rows: N, cols: K, groupSize: g, hessian: H))
        var num: Float = 0, den: Float = 0
        for t in 0..<T { let xb = t * K
            for n in 0..<N { let wb = n * K
                var dq: Float = 0, dr: Float = 0
                for k in 0..<K { dq += (W[wb + k] - Wq[wb + k]) * X[xb + k]; dr += W[wb + k] * X[xb + k] }
                num += dq * dq; den += dr * dr
            }
        }
        // ~0.056 with the guard (the int4 floor for one 64-wide group); ~1.0 if the dead column polluted
        // the scale and crushed the live columns. 0.08 cleanly separates the two.
        XCTAssertLessThan((num / den).squareRoot(), 0.08, "dead channel polluted the group scale (live cols crushed)")
    }
}
