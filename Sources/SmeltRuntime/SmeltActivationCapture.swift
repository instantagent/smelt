// SmeltActivationCapture — model-agnostic activation statistics for quantization calibration.
//
// Accumulates, per projection weight: the per-input-channel sum-of-squares (the imatrix diagonal)
// and, opt-in, the full activation Hessian H = Σ x xᵀ that SmeltGPTQ consumes. Both are pure
// functions of the captured input rows, so the type carries no model coupling.
//
// Input rows arrive either as FP32 (the TTS codec's matmul inputs) or FP16 (the general runtime's
// post-norm activations); `inputIsFloat16` selects the widening. The FP32 path reads the buffer in
// place, bit-identically to the original TTS capture.

import Foundation
import Metal
import Accelerate

public final class SmeltActivationCapture {
    private var sumSq: [String: [Float]] = [:]
    private var rowCount: [String: Int] = [:]
    /// Opt-in full Hessian `H = Σ x xᵀ` per weight, for SmeltGPTQ. ssyrk fills only the row-major upper
    /// triangle while accumulating; the accessors mirror it to a full symmetric matrix on handoff (GPTQ's
    /// LAPACK path reads the other triangle). Off by default: H is [K,K] (down_proj K=6144 → 151 MB each).
    public var captureHessian = false
    /// When set, restrict the full-Hessian capture to these weight names (the imatrix diagonal is still
    /// captured for all). Bounds memory when only a few linears are measured — capturing every projection
    /// at once is many GB. `nil` (default) captures all weights `captureHessian` sees.
    public var captureHessianNames: Set<String>?
    private var hess: [String: [Float]] = [:]
    /// Buffered activation rows ([rows, k] row-major) awaiting a batched ssyrk flush into `hess`. A rank-1
    /// ssyrk per M=1 decode token is bandwidth-bound — it streams the whole [k,k] H (151 MB for K=6144)
    /// per token. Buffering ~`hessFlushRows` rows and flushing as ONE ssyrk(m=rows) cuts that H traffic
    /// by that factor (the dominant calibration cost); the accumulated XᵀX is the same up to BLAS
    /// reduction order (a few fp ulps, immaterial to the Cholesky / logit-cosine consumers).
    private var hessPending: [String: [Float]] = [:]
    private let hessFlushRows = 128
    public init() {}

    /// Accumulate `xBuf` ([m, k] row-major) per-channel sum-of-squares under `name` (the imatrix
    /// diagonal), and — when `captureHessian` — buffer the rows for the batched `XᵀX` Hessian. Rows are
    /// FP16 when `inputIsFloat16` (the general runtime's post-norm activations), else FP32.
    func accumulate(_ name: String, _ xBuf: MTLBuffer, m: Int, k: Int, inputIsFloat16: Bool = false) {
        if inputIsFloat16 {
            let src = xBuf.contents().bindMemory(to: UInt16.self, capacity: m * k)
            var widened = [Float](repeating: 0, count: m * k)
            for i in 0..<(m * k) { widened[i] = Float(Float16(bitPattern: src[i])) }
            widened.withUnsafeBufferPointer { accumulateRows(name, $0.baseAddress!, m: m, k: k) }
        } else {
            let x = xBuf.contents().bindMemory(to: Float.self, capacity: m * k)
            accumulateRows(name, x, m: m, k: k)
        }
    }

    /// Core accumulation over `m` row-major FP32 rows of width `k`.
    private func accumulateRows(_ name: String, _ x: UnsafePointer<Float>, m: Int, k: Int) {
        if sumSq[name] == nil { sumSq[name] = [Float](repeating: 0, count: k) }
        sumSq[name]!.withUnsafeMutableBufferPointer { acc in
            for mm in 0..<m {
                let base = mm * k
                for kk in 0..<k { let v = x[base + kk]; acc[kk] += v * v }
            }
        }
        rowCount[name, default: 0] += m
        if captureHessian, captureHessianNames?.contains(name) ?? true {
            // Eager-allocate hess[name] (before any flush) so hessians()/flushHessian can rely on
            // hess.keys ⊇ hessPending.keys — a captured weight is never pending-only.
            if hess[name] == nil { hess[name] = [Float](repeating: 0, count: k * k) }
            hessPending[name, default: []].append(contentsOf: UnsafeBufferPointer(start: x, count: m * k))
            if hessPending[name]!.count >= hessFlushRows * k { flushHessian(name) }
        }
    }

    /// Flush `name`'s buffered rows into `hess` as one `ssyrk` (upper triangle): C += XᵀX over the
    /// buffered [rows, k] X. `C[i,j] = Σ_row x[row,i]·x[row,j]`, i≤j; X is row-major (lda=k). `k` is the
    /// weight's input dim, recovered from the imatrix entry (always present alongside a captured Hessian).
    private func flushHessian(_ name: String) {
        guard let pend = hessPending[name], !pend.isEmpty, let k = sumSq[name]?.count else { return }
        let rows = pend.count / k
        hess[name]!.withUnsafeMutableBufferPointer { h in
            pend.withUnsafeBufferPointer { p in
                cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans,
                            Int32(k), Int32(rows), 1.0, p.baseAddress!, Int32(k), 1.0, h.baseAddress!, Int32(k))
            }
        }
        hessPending[name] = []
    }

    /// Accumulated `H = Σ x xᵀ` [k,k] for `name` as a FULL symmetric matrix, or nil if uncaptured. Pass to
    /// SmeltGPTQ as the activation Hessian. ssyrk fills only the row-major upper triangle, but GPTQ's
    /// `spotrf_('U')` reads it column-major (= the row-major lower triangle) and treats H as fully
    /// symmetric, so mirror upper→lower here before handing off. Flushes any buffered rows first.
    public func hessian(_ name: String) -> [Float]? {
        flushHessian(name)
        guard var h = hess[name], let k = sumSq[name]?.count else { return nil }
        for i in 0..<k { for j in (i + 1)..<k { h[j * k + i] = h[i * k + j] } }
        return h
    }

    /// E[x_k²] per input channel for `name`, or nil if `name` was never seen.
    public func importance(_ name: String) -> [Float]? {
        guard let s = sumSq[name], let r = rowCount[name], r > 0 else { return nil }
        let inv = 1.0 / Float(r)
        return s.map { $0 * inv }
    }

    /// All weight names seen during calibration.
    public var capturedNames: [String] { Array(sumSq.keys) }

    /// Calibration rows (input tokens) accumulated for `name`. The generic rank of `H = Σxxᵀ` is
    /// min(this, K), so it bounds the Hessian rank — the GPTQ de-risk signal for high-K weights.
    public func calibrationRows(_ name: String) -> Int { rowCount[name] ?? 0 }

    /// All captured Hessians {name: full symmetric [k,k]}.
    public func hessians() -> [String: [Float]] {
        var out: [String: [Float]] = [:]
        for name in hess.keys { out[name] = hessian(name) }
        return out
    }

    /// The full imatrix {name: E[x²]} for every captured weight.
    public func imatrix() -> [String: [Float]] {
        var m: [String: [Float]] = [:]
        for n in sumSq.keys { if let imp = importance(n) { m[n] = imp } }
        return m
    }
}

/// Historical name for the capture type, retained where the Qwen3-TTS codec path constructs it.
public typealias Qwen3TTSActivationCapture = SmeltActivationCapture
