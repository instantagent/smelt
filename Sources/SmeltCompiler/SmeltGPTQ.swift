// SmeltGPTQ — GPTQ error-feedback weight quantization to the SAME u4 block format SmeltAffineU4
// produces (per-(output-row, input-group) int4 nibbles + fp16 scale/bias). Build-time CPU only; the
// runtime gemv_u4 kernel is unchanged. GPTQ (Frantar et al., arXiv:2210.17323): quantize input
// columns left-to-right and, after each column, compensate the not-yet-quantized columns for the
// rounding error using the activation Hessian H = E[xxᵀ] — minimizing the layer's OUTPUT error, not
// weight error. Natural column order (no act-order) so the emitted block stays runtime-neutral.
//
// W is [N=out, K=in] row-major; the gemv reduces over K, groups run along K. Per-group affine scale is
// computed (min/max) at group entry from the current (post-prior-feedback) W and FROZEN for the group,
// matching SmeltAffineU4 so the block is byte-identical. The nibble uses the full-precision scale/bias
// and the error is measured against the kernel's fp16 dequant (nibble·fp16(scale)+fp16(bias)), so GPTQ
// optimizes exactly what the runtime computes.

import Foundation
import Accelerate

public enum SmeltGPTQ {

    /// Quantize `weights` [rows×cols] row-major fp32 to a u4 block using GPTQ with Hessian `hessian`
    /// [cols×cols] (symmetric, = Σ x xᵀ over calibration). `damping` (× mean diag) regularizes H.
    public static func quantize(weights: [Float], rows: Int, cols: Int, groupSize: Int,
                                hessian: [Float], damping: Float = 0.01) -> SmeltAffineU4.Packed {
        precondition(weights.count == rows * cols, "weights.count != rows*cols")
        precondition(hessian.count == cols * cols, "hessian.count != cols*cols")
        let N = rows, K = cols
        let groups = SmeltAffineU4.numGroups(cols: K, groupSize: groupSize)
        let rowStride = SmeltAffineU4.packedRowStride(cols: K)

        // Working copy of W (mutated by feedback) and H (factored in place).
        var W = weights
        var H = hessian

        // R = the UPPER Cholesky factor of H⁻¹ (the GPTQ compensation object: H⁻¹ = RᵀR). Sequence on
        // the symmetric H (row-major == column-major): chol(H)→inv→chol(inv). On failure (singular even
        // after damping) fall back to a diagonal R (→ plain per-group affine, no cross-column feedback).
        let R = choleskyOfInverse(&H, n: K, damping: damping)

        var nibbles = [UInt8](repeating: 0, count: N * rowStride)
        var scales = [UInt16](repeating: 0, count: N * groups)
        var biases = [UInt16](repeating: 0, count: N * groups)

        // Dead input channels (no calibration signal → zero output contribution) must not widen a
        // group's affine min/max and hurt its live columns — zero them up front (matches IST-DASLab).
        // Use the ORIGINAL (pre-damping) Hessian diagonal to detect them.
        for k in 0..<K where hessian[k * K + k] == 0 { for n in 0..<N { W[n * K + k] = 0 } }

        // Per-(row,group) fp16-roundtripped scale/bias for the group being quantized — the nibble is
        // chosen AND the error measured on the kernel's actual fp16 dequant grid.
        var scaleF16 = [Float](repeating: 1, count: N)
        var biasF16 = [Float](repeating: 0, count: N)
        var err = [Float](repeating: 0, count: N)

        W.withUnsafeMutableBufferPointer { w in
            R.withUnsafeBufferPointer { r in
                for j in 0..<K {
                    let g = j / groupSize
                    if j % groupSize == 0 {
                        // Freeze the group's per-row scale/bias from the current W columns [j, gEnd).
                        let gEnd = min(j + groupSize, K)
                        for n in 0..<N {
                            var mn = Float.infinity, mx = -Float.infinity
                            let base = n * K
                            for c in j..<gEnd { let v = w[base + c]; if v < mn { mn = v }; if v > mx { mx = v } }
                            let p = SmeltAffineU4.affineParams(lo: mn, hi: mx)
                            scaleF16[n] = Float(Float16(bitPattern: p.scaleBits)); biasF16[n] = Float(Float16(bitPattern: p.biasBits))
                            scales[n * groups + g] = p.scaleBits
                            biases[n * groups + g] = p.biasBits
                        }
                    }
                    // Quantize column j (all rows), record the error against the kernel's fp16 dequant.
                    let rjj = r[j * K + j]                       // R[j,j] (upper, column-major index)
                    for n in 0..<N {
                        let v = w[n * K + j]
                        let inv = scaleF16[n] > 0 ? 1 / scaleF16[n] : 0   // fp16 scale can underflow to 0
                        let nib = min(max(((v - biasF16[n]) * inv).rounded(), 0), 15)
                        let nibble = UInt8(nib)
                        let byteIdx = n * rowStride + (j >> 1)
                        if j & 1 == 0 { nibbles[byteIdx] = nibble } else { nibbles[byteIdx] |= nibble << 4 }
                        let wq = nib * scaleF16[n] + biasF16[n]
                        err[n] = rjj != 0 ? (v - wq) / rjj : 0
                    }
                    // Feedback: W[:, j+1:] −= err ⊗ R[j, j+1:]  (rank-1, all rows; R[j,k]=r[k*K+j], k>j).
                    if j + 1 < K {
                        err.withUnsafeBufferPointer { e in
                            cblas_sger(CblasRowMajor, Int32(N), Int32(K - j - 1), -1.0,
                                       e.baseAddress!, 1,
                                       r.baseAddress! + ((j + 1) * K + j), Int32(K),
                                       w.baseAddress! + (j + 1), Int32(K))
                        }
                    }
                }
            }
        }
        return SmeltAffineU4.Packed(nibbles: nibbles, scales: scales, biases: biases,
                                    rows: N, cols: K, groupSize: groupSize)
    }

    /// In-place: `h` (n×n symmetric) → returns R, the UPPER Cholesky factor of (h + damping·meanDiag·I)⁻¹
    /// as a flat column-major n×n (R[i,j] at j*n+i, valid for i≤j). Returns a diagonal R (1/√diag) if the
    /// factorization fails — degrades GPTQ to per-group affine rather than producing garbage.
    static func choleskyOfInverse(_ h: inout [Float], n: Int, damping: Float) -> [Float] {
        var diagSum: Float = 0
        for i in 0..<n { diagSum += h[i * n + i] }
        let lambda = damping * (diagSum / Float(n))
        for i in 0..<n {
            h[i * n + i] += lambda
            if h[i * n + i] <= 0 { h[i * n + i] = lambda > 0 ? lambda : 1 }   // dead/negative channel guard
        }
        var uplo: CChar = 0x55   // 'U'
        // Smelt intentionally uses the LP64 Accelerate ABI. Spell the LAPACK
        // integer width directly so this public package also compiles for
        // consumers that cannot inherit unsafe Clang importer flags.
        var nn = Int32(n), lda = Int32(n), info = Int32(0)
        h.withUnsafeMutableBufferPointer { p in
            spotrf_(&uplo, &nn, p.baseAddress!, &lda, &info)          // H = UᵀU
            if info == 0 { spotri_(&uplo, &nn, p.baseAddress!, &lda, &info) }  // → H⁻¹ (symmetric)
            if info == 0 { spotrf_(&uplo, &nn, p.baseAddress!, &lda, &info) }  // H⁻¹ = RᵀR, R upper
        }
        if info != 0 {
            // Factorization failed even after damping — fall back to a diagonal R, which zeroes the
            // off-diagonal feedback terms so GPTQ degrades to plain per-group affine. Only the zero
            // off-diagonals matter for correctness; R[j,j] merely scales a per-column error that is then
            // never propagated, so use the identity (and don't read the LAPACK-clobbered `h`).
            var diag = [Float](repeating: 0, count: n * n)
            for i in 0..<n { diag[i * n + i] = 1 }
            return diag
        }
        return h
    }
}
