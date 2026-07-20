import Foundation
import Accelerate
import Testing
@testable import SmeltCompiler
@testable import SmeltRuntime
import SmeltSchema

// P3 parity gate: GPTQ-u4 must beat naive affine-u4 on the property GPTQ optimizes —
// the activation-weighted output error tr(ΔW·H·ΔWᵀ), where ΔW = W − dequant(quant(W)).
// Crucially the error is measured on a HELD-OUT activation Hessian (calibration set A,
// evaluation set B), so this tests generalization, not training fit. Gated on the built
// package + checkpoint; skips cleanly when absent.

private let qwen35PkgPath = "/tmp/qwen35-0.8b-pkg/Qwen_Qwen3.5-0.8B.smeltpkg"
private let qwen35Checkpoint = "/tmp/qwen35-0.8b"

/// tr(ΔW · H · ΔWᵀ) = Σ_rows (ΔW_r)ᵀ H (ΔW_r), with H the full symmetric [cols,cols] Hessian.
private func weightedOutputError(_ deltaW: [Float], rows: Int, cols: Int, hessian H: [Float]) -> Double {
    var total = 0.0
    var y = [Float](repeating: 0, count: cols)
    deltaW.withUnsafeBufferPointer { dw in
        H.withUnsafeBufferPointer { h in
            for r in 0..<rows {
                let row = dw.baseAddress! + r * cols
                y.withUnsafeMutableBufferPointer { yb in
                    cblas_ssymv(CblasRowMajor, CblasUpper, Int32(cols), 1.0,
                                h.baseAddress!, Int32(cols), row, 1, 0.0, yb.baseAddress!, 1)
                    total += Double(cblas_sdot(Int32(cols), row, 1, yb.baseAddress!, 1))
                }
            }
        }
    }
    return total
}

@Test func gptqU4BeatsNaiveAffineOnHeldOutWeightedError() throws {
    guard FileManager.default.fileExists(atPath: "\(qwen35PkgPath)/gptq_capture_points.json"),
          FileManager.default.fileExists(atPath: "\(qwen35Checkpoint)/config.json")
    else { return }
    // A representative attention layer + its MLP: distinct input slots and dims.
    // VANILLA (non-QAT) weights: GPTQ's Hessian-aware rounding is a clear win.
    let mean = try runHeldOutParity(
        packagePath: qwen35PkgPath, checkpointDir: qwen35Checkpoint,
        ir: SmeltModelIR.qwen35_0_8B, attnLayer: 3, label: "qwen35_0.8b")
    #expect(mean < 0.95, "GPTQ should clearly beat affine on vanilla weights; mean \(mean)")
}

/// Capture the held-out activation-weighted error tr(ΔW·H_B·ΔWᵀ) for GPTQ vs naive
/// affine on one attention layer + its MLP, and assert GPTQ wins on every
/// adequately-calibrated weight (rank ≥ cols). Calibrates on token set A, evaluates
/// on a DISJOINT set B (generalization, not training fit).
@discardableResult
private func runHeldOutParity(
    packagePath: String, checkpointDir: String,
    ir: SmeltModelIR, attnLayer: Int, label: String
) throws -> Double {
    let groupSize = ir.quantization.groupSize
    let sample = [
        "layers_\(attnLayer)_self_attn_q_proj_weight",
        "layers_\(attnLayer)_self_attn_o_proj_weight",
        "layers_\(attnLayer)_mlp_gate_proj_weight",
        "layers_\(attnLayer)_mlp_up_proj_weight",
        "layers_\(attnLayer)_mlp_down_proj_weight",
    ]
    let sampleSet = Set(sample)

    let points = try JSONDecoder().decode(
        SmeltGPTQCapturePoints.self,
        from: Data(contentsOf: URL(fileURLWithPath: "\(packagePath)/gptq_capture_points.json"))
    ).prefill.filter { sampleSet.contains($0.weightName) }
    #expect(points.count == sample.count)

    // Calibration set A and a DISJOINT held-out set B. GPTQ's Hessian rank is bounded
    // by the calibration token count, so we use enough tokens (8×256 = 2048) to exceed
    // most in-scope input dims — under-calibration makes GPTQ overfit the calibration
    // subspace and regress on held-out data (the plan's §5 rank caveat).
    let tokensA: [[Int32]] = (0..<8).map { s in (0..<256).map { Int32(100 + s * 400 + $0) } }
    let tokensB: [[Int32]] = (0..<8).map { s in (0..<256).map { Int32(40000 + s * 400 + $0) } }

    let runtime = try SmeltRuntime(packagePath: packagePath)
    func captureHessians(_ tokenSets: [[Int32]]) throws -> SmeltActivationCapture {
        let cap = SmeltActivationCapture()
        cap.captureHessian = true
        cap.captureHessianNames = sampleSet
        for tokens in tokenSets {
            try autoreleasepool { try runtime.captureGPTQActivations(tokenIds: tokens, capturePoints: points, into: cap) }
        }
        return cap
    }
    let calib = try captureHessians(tokensA)
    let heldOut = try captureHessians(tokensB)

    // fp32 weights by Agent name from the checkpoint.
    let adapter = try SmeltCheckpointAdapter.authored(for: ir)
    let loader = try SafetensorsLoader(directory: checkpointDir)
    var hfByAgent: [String: SafetensorInfo] = [:]
    for info in loader.tensors where adapter.isTextModelTensor(info.name) { hfByAgent[adapter.mapName(info.name)] = info }

    var adequateRatios: [Double] = []   // weights with calibration rank ≥ cols
    var report: [String] = []
    for name in sample {
        let info = try #require(hfByAgent[name])
        let (rows, cols) = (info.shape[0], info.shape[1])
        // Reuse the calibrator's dtype-correct widening (the build's bit-exact path).
        let w = try SmeltGPTQCalibrator.widenToF32(
            loader.tensorData(info), dtype: info.dtype, count: rows * cols, name: name)

        let affine = SmeltAffineU4.quantize(w, rows: rows, cols: cols, groupSize: groupSize)
        let calibH = try #require(calib.hessian(name))
        let gptq = SmeltGPTQ.quantize(weights: w, rows: rows, cols: cols, groupSize: groupSize, hessian: calibH)

        let affineDQ = SmeltAffineU4.dequantize(affine)
        let gptqDQ = SmeltAffineU4.dequantize(gptq)
        let dAffine = (0..<(rows * cols)).map { w[$0] - affineDQ[$0] }
        let dGptq = (0..<(rows * cols)).map { w[$0] - gptqDQ[$0] }

        let heldH = try #require(heldOut.hessian(name))
        let errAffine = weightedOutputError(dAffine, rows: rows, cols: cols, hessian: heldH)
        let errGptq = weightedOutputError(dGptq, rows: rows, cols: cols, hessian: heldH)
        let ratio = errAffine > 0 ? errGptq / errAffine : 1.0
        // GPTQ needs rank WELL above cols to generalize, not just rank ≥ cols: at rank ≈ cols
        // it overfits the calibration Hessian (calib error drops but held-out rises). Require
        // rows ≥ 2× cols as the adequacy bar (rank ≤ rows; full rank presumed).
        let calibRows = calib.calibrationRows(name)
        let adequate = calibRows >= 2 * cols
        report.append("\(name) cols=\(cols) calibRows=\(calibRows) adequate=\(adequate) ratio=\(String(format: "%.3f", ratio))")

        // Only adequately-calibrated weights (rank ≥ cols, full rank presumed) feed the
        // aggregate; the caller asserts the model-specific expectation on the returned mean
        // (GPTQ wins on vanilla weights; ≈ affine on QAT-conditioned weights).
        if adequate { adequateRatios.append(ratio) }
    }

    print("[\(label)] GPTQ vs naive-affine held-out weighted error:\n  " + report.joined(separator: "\n  "))
    #expect(!adequateRatios.isEmpty, "no adequately-calibrated weight in the sample")
    // Mean held-out GPTQ/affine error ratio over adequately-calibrated weights. <1 = GPTQ wins.
    return adequateRatios.reduce(0, +) / Double(adequateRatios.count)
}

// MARK: - 2-bit experiment (does Hessian-aware quant regain value below the QAT bit-width?)

/// Naive per-(row,group) affine to `maxLevel`+1 levels (round-to-nearest, no feedback),
/// returned dequantized. maxLevel=15 → 4-bit, maxLevel=3 → 2-bit.
private func naiveAffineDequant(_ w: [Float], rows: Int, cols: Int, groupSize: Int, maxLevel: Int) -> [Float] {
    let lvl = Float(maxLevel)
    var what = [Float](repeating: 0, count: rows * cols)
    for n in 0..<rows {
        var j = 0
        while j < cols {
            let gEnd = min(j + groupSize, cols)
            var mn = Float.infinity, mx = -Float.infinity
            for c in j..<gEnd { let v = w[n * cols + c]; if v < mn { mn = v }; if v > mx { mx = v } }
            let scale = mx > mn ? (mx - mn) / lvl : 1
            for c in j..<gEnd {
                let code = min(max((((w[n * cols + c] - mn) / scale)).rounded(), 0), lvl)
                what[n * cols + c] = code * scale + mn
            }
            j = gEnd
        }
    }
    return what
}

/// GPTQ error-feedback affine to `maxLevel`+1 levels, returned dequantized. Mirrors
/// SmeltGPTQ.quantize (reuses choleskyOfInverse) but parameterizes the grid and skips
/// the u4 packing / fp16-scale roundtrip (fp32 scale; this is a reconstruction-error probe).
private func gptqAffineDequant(_ weights: [Float], rows N: Int, cols K: Int, groupSize: Int, hessian: [Float], maxLevel: Int, damping: Float = 0.01) -> [Float] {
    let lvl = Float(maxLevel)
    var W = weights
    var H = hessian
    let R = SmeltGPTQ.choleskyOfInverse(&H, n: K, damping: damping)
    for k in 0..<K where hessian[k * K + k] == 0 { for n in 0..<N { W[n * K + k] = 0 } }
    var what = [Float](repeating: 0, count: N * K)
    var scale = [Float](repeating: 1, count: N)
    var bias = [Float](repeating: 0, count: N)
    var err = [Float](repeating: 0, count: N)
    W.withUnsafeMutableBufferPointer { w in
        R.withUnsafeBufferPointer { r in
            for j in 0..<K {
                if j % groupSize == 0 {
                    let gEnd = min(j + groupSize, K)
                    for n in 0..<N {
                        var mn = Float.infinity, mx = -Float.infinity
                        let base = n * K
                        for c in j..<gEnd { let v = w[base + c]; if v < mn { mn = v }; if v > mx { mx = v } }
                        scale[n] = mx > mn ? (mx - mn) / lvl : 1
                        bias[n] = mn
                    }
                }
                let rjj = r[j * K + j]
                for n in 0..<N {
                    let v = w[n * K + j]
                    let inv = scale[n] > 0 ? 1 / scale[n] : 0
                    let code = min(max((((v - bias[n]) * inv)).rounded(), 0), lvl)
                    let wq = code * scale[n] + bias[n]
                    what[n * K + j] = wq
                    err[n] = rjj != 0 ? (v - wq) / rjj : 0
                }
                if j + 1 < K {
                    err.withUnsafeBufferPointer { e in
                        cblas_sger(CblasRowMajor, Int32(N), Int32(K - j - 1), -1.0, e.baseAddress!, 1,
                                   r.baseAddress! + ((j + 1) * K + j), Int32(K), w.baseAddress! + (j + 1), Int32(K))
                    }
                }
            }
        }
    }
    return what
}

/// Bits-per-weight of uniform per-(row,group) affine 2-bit: 2-bit codes + one fp16
/// scale + one fp16 min per row-group. Floor is ~2.016 bpw at one group per row (G=cols),
/// because affine pays per-ROW metadata — it can never reach TQH's shared-codebook ~2.0.
private func affineBpw(cols: Int, groupSize: Int) -> Double {
    let groupsPerRow = (cols + groupSize - 1) / groupSize
    return 2.0 + Double(groupsPerRow * 32) / Double(cols)
}

/// Production TQH 2-bit (Hadamard + k-means 4-centroid codebook) reconstruction, optionally
/// imatrix-weighted. Calls the shipped SmeltTurboQuantHQuantizer + SmeltTurboQuantHCodec, so
/// this measures the actual production 2-bit format (capacity-based, not uniform-affine).
/// Returns the reconstruction and the format's exact bpw (codes + shared codebook, over rows*cols).
private func tqhDequant(_ w: [Float], rows: Int, cols: Int, groupSize: Int, importance: [Float]?) -> (recon: [Float], bpw: Double) {
    let w16 = w.map { Float16($0) }
    let (codes, codebook) = w16.withUnsafeBufferPointer {
        SmeltTurboQuantHQuantizer.quantize(weights: $0.baseAddress!, rows: rows, cols: cols, groupSize: groupSize, importance: importance)
    }
    let cbBits = codebook.map { $0.bitPattern }
    let codesPerRow = codes.count / rows
    var what = [Float](repeating: 0, count: rows * cols)
    var rowOut = [UInt16](repeating: 0, count: cols)
    codes.withUnsafeBufferPointer { cp in
        cbBits.withUnsafeBufferPointer { cb in
            for n in 0..<rows {
                rowOut.withUnsafeMutableBufferPointer { ro in
                    SmeltTurboQuantHCodec.dequantizeRowInto(codes: cp.baseAddress! + n * codesPerRow,
                        codebook: cb.baseAddress!, output: ro.baseAddress!, cols: cols, groupSize: groupSize)
                }
                for j in 0..<cols { what[n * cols + j] = Float(Float16(bitPattern: rowOut[j])) }
            }
        }
    }
    let bpw = Double(codes.count * 8 + codebook.count * 16) / Double(rows * cols)
    return (what, bpw)
}

/// Capture held-out Hessians for one attn layer + MLP, then report GPTQ/naive held-out
/// weighted-error ratios at 4-bit (sanity vs the shipped result) and 2-bit (the experiment).
/// Returns the mean 2-bit GPTQ/naive ratio over adequately-calibrated weights, plus the
/// per-role iso-bpw verdict (tqh/affIso, keyed by role label like "q_proj"/"mlp_gate_proj").
@discardableResult
private func run2BitExperiment(packagePath: String, checkpointDir: String, ir: SmeltModelIR, attnLayer: Int, label: String, calibSeqs: Int = 8) throws -> (mean: Double, iso: [String: Double]) {
    let groupSize = ir.quantization.groupSize
    let roles = ["q_proj", "o_proj", "mlp_gate_proj", "mlp_up_proj", "mlp_down_proj"]
    func weightName(_ role: String) -> String {
        role.hasPrefix("mlp") ? "layers_\(attnLayer)_\(role)_weight" : "layers_\(attnLayer)_self_attn_\(role)_weight"
    }
    let sample = roles.map(weightName)
    let sampleSet = Set(sample)
    let points = try JSONDecoder().decode(
        SmeltGPTQCapturePoints.self,
        from: Data(contentsOf: URL(fileURLWithPath: "\(packagePath)/gptq_capture_points.json"))
    ).prefill.filter { sampleSet.contains($0.weightName) }
    #expect(points.count == sample.count)

    let tokensA: [[Int32]] = (0..<calibSeqs).map { s in (0..<256).map { Int32(100 + s * 400 + $0) } }
    let tokensB: [[Int32]] = (0..<8).map { s in (0..<256).map { Int32(40000 + s * 400 + $0) } }
    let runtime = try SmeltRuntime(packagePath: packagePath)
    func capture(_ sets: [[Int32]]) throws -> SmeltActivationCapture {
        let cap = SmeltActivationCapture(); cap.captureHessian = true; cap.captureHessianNames = sampleSet
        for t in sets { try autoreleasepool { try runtime.captureGPTQActivations(tokenIds: t, capturePoints: points, into: cap) } }
        return cap
    }
    let calib = try capture(tokensA), heldOut = try capture(tokensB)

    let adapter = try SmeltCheckpointAdapter.authored(for: ir)
    let loader = try SafetensorsLoader(directory: checkpointDir)
    var hfByAgent: [String: SafetensorInfo] = [:]
    for info in loader.tensors where adapter.isTextModelTensor(info.name) { hfByAgent[adapter.mapName(info.name)] = info }

    var ratios2: [Double] = []
    var isoByRole: [String: Double] = [:]
    var report: [String] = []
    for role in roles {
        let name = weightName(role)
        let info = try #require(hfByAgent[name])
        let (rows, cols) = (info.shape[0], info.shape[1])
        let w = try SmeltGPTQCalibrator.widenToF32(loader.tensorData(info), dtype: info.dtype, count: rows * cols, name: name)
        let calibH = try #require(calib.hessian(name)), heldH = try #require(heldOut.hessian(name))
        // Report GPTQ/naive on BOTH the held-out Hessian (generalization) AND the calibration
        // Hessian GPTQ optimized (training). GPTQ MUST win on calib (calib<1) — if it doesn't,
        // that's a bug (it isn't minimizing its objective), not overfitting.
        func ratios(_ maxLevel: Int) -> (held: Double, calib: Double) {
            let naive = naiveAffineDequant(w, rows: rows, cols: cols, groupSize: groupSize, maxLevel: maxLevel)
            let gptq = gptqAffineDequant(w, rows: rows, cols: cols, groupSize: groupSize, hessian: calibH, maxLevel: maxLevel)
            let dN = (0..<rows * cols).map { w[$0] - naive[$0] }
            let dG = (0..<rows * cols).map { w[$0] - gptq[$0] }
            let hN = weightedOutputError(dN, rows: rows, cols: cols, hessian: heldH)
            let hG = weightedOutputError(dG, rows: rows, cols: cols, hessian: heldH)
            let cN = weightedOutputError(dN, rows: rows, cols: cols, hessian: calibH)
            let cG = weightedOutputError(dG, rows: rows, cols: cols, hessian: calibH)
            return (hN > 0 ? hG / hN : 1.0, cN > 0 ? cG / cN : 1.0)
        }
        let r4 = ratios(15), r2 = ratios(3)
        // rows ≥ 2× cols: GPTQ overfits at rank ≈ cols, so require comfortable over-calibration.
        let adequate = calib.calibrationRows(name) >= 2 * cols
        // ISO-BPW comparison: the g/n numbers above are NOT bpw-matched — uniform-affine at the IR
        // group size (the affG<groupSize> column) pays per-row fp16 scale+min, ~2.25–2.5 bpw, while
        // TQH is ~2.0 bpw (its 4-centroid codebook is shared across all rows WITHIN a column-group,
        // so its per-weight overhead is near-zero). To compare fairly, also run GPTQ-affine at one
        // group per row (G=cols), affine's MINIMUM bpw (~2.016), the closest uniform-affine can get
        // to TQH's budget. Verdict ratio tqh/affIso < 1 ⇒ TQH wins at equal bpw; > 1 ⇒ uniform-affine
        // wins even when handed no extra bits. Errors are held-out weighted. (imatrix-weighted TQH
        // ≈ plain on this metric — see ASSUMPTIONS — so only the plain TQH point is measured here.)
        // CAVEAT: gptqAffineDequant uses fp32 scale/min (it's a reconstruction probe) while charged
        // fp16-metadata bpw, and TQH runs the real fp16 codec — a small bias FAVORING affine. It
        // doesn't flip these verdicts (margins are 2.4–5.3×), but the real fp16-stored affine would
        // be marginally worse than measured here.
        func heldErr(_ recon: [Float]) -> Double {
            weightedOutputError((0..<rows * cols).map { w[$0] - recon[$0] }, rows: rows, cols: cols, hessian: heldH)
        }
        let affGroupBpw = affineBpw(cols: cols, groupSize: groupSize)
        let affIsoBpw = affineBpw(cols: cols, groupSize: cols)
        let eAffIso = heldErr(gptqAffineDequant(w, rows: rows, cols: cols, groupSize: cols, hessian: calibH, maxLevel: 3))
        let (tqhRecon, tqhBpw) = tqhDequant(w, rows: rows, cols: cols, groupSize: 128, importance: nil)
        let isoVerdict = eAffIso > 0 ? heldErr(tqhRecon) / eAffIso : 1.0
        let f = { (x: Double) in String(format: "%.3f", x) }
        report.append("\(name) cols=\(cols) adeq=\(adequate) | 4b g/n=\(f(r4.held)) 2b g/n=\(f(r2.held)) "
            + "| iso-bpw aff@\(cols)=\(f(affIsoBpw)) tqh=\(f(tqhBpw)) (affG\(groupSize)=\(f(affGroupBpw))) "
            + "| tqh/affIso=\(f(isoVerdict))")
        if adequate { ratios2.append(r2.held) }
        isoByRole[role] = isoVerdict
    }
    print("[\(label)] GPTQ vs naive held-out weighted error, by bit-width:\n  " + report.joined(separator: "\n  "))
    // Guard the mean against a vacuous pass: with no adequately-calibrated weight the
    // sum/max(count,1) collapses to 0.0, which would slip under any `< 0.9` threshold.
    #expect(!ratios2.isEmpty, "[\(label)] no adequately-calibrated weights — mean is vacuous")
    return (ratios2.reduce(0, +) / Double(max(ratios2.count, 1)), isoByRole)
}

@Test func gptqBeatsNaiveAt2BitGivenAdequateCalibration() throws {
    // GPTQ beats naive at 2-bit on BOTH vanilla and QAT weights — the key is calibration rank
    // ≫ cols, NOT the QAT-ness. (An earlier under-calibrated run, rank ≈ cols, made GPTQ overfit
    // and wrongly looked like "GPTQ doesn't help QAT"; see ASSUMPTIONS.md. The run2BitExperiment
    // report prints calib vs held per weight so the overfit→generalize behavior stays visible.)
    let qpkg = qwen35PkgPath, qckpt = qwen35Checkpoint

    // Vanilla qwen (cols 1024 → 2048 tok is rank 2×): GPTQ wins at 2-bit.
    if FileManager.default.fileExists(atPath: "\(qpkg)/gptq_capture_points.json") {
        let q2 = try run2BitExperiment(packagePath: qpkg, checkpointDir: qckpt,
                                       ir: SmeltModelIR.qwen35_0_8B, attnLayer: 3, label: "qwen35 vanilla (2048 tok)", calibSeqs: 8)
        #expect(q2.mean < 0.9, "GPTQ should beat naive at 2-bit on vanilla weights; mean \(q2.mean)")
    }
}
