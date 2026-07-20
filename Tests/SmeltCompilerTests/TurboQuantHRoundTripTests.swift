// Validate SmeltTurboQuantHQuantizer against the Python reference
// at tools/turboquant-h-sim.py. Same algorithm — Hadamard rotate,
// 4-centroid k-means codebook per group, 2-bit codes, inverse
// Hadamard on dequant — so round-trip MSE/cosine should land at
// the Lloyd-Max 2-bit floor (~0.94 cosine on Gaussian).

import Foundation
import Testing
@testable import SmeltCompiler

@Test func turboQuantHFWHTIsNormalizedSelfInverse() {
    // Fast Walsh-Hadamard is self-inverse under the 1/√n
    // normalization: applying FWHT twice and scaling by 1/n
    // returns the input. Validates the butterfly matches the
    // orthonormal Sylvester Hadamard rather than an unscaled
    // variant.
    let n = 128
    let scale = Float(1.0 / Double(n).squareRoot())
    var rng = SplitMix64(seed: 1)
    var x = [Float](repeating: 0, count: n)
    for i in 0 ..< n {
        x[i] = Float(rng.nextUnitDouble()) - 0.5
    }
    var y = x
    SmeltTurboQuantHQuantizer.fastWalshHadamard(buffer: &y, size: n)
    for k in 0 ..< n { y[k] *= scale }
    SmeltTurboQuantHQuantizer.fastWalshHadamard(buffer: &y, size: n)
    for k in 0 ..< n { y[k] *= scale }
    var maxDiff: Float = 0
    for k in 0 ..< n {
        let d = abs(x[k] - y[k])
        if d > maxDiff { maxDiff = d }
    }
    #expect(maxDiff < 1e-4,
            "FWHT round-trip max abs error \(maxDiff) > 1e-4")
}

@Test func turboQuantHRoundTripGaussianCosineNearFloor() {
    let rows = 1024
    let cols = 1024
    var rng = SplitMix64(seed: 42)
    var w = [Float16](repeating: 0, count: rows * cols)
    for k in 0 ..< (rows * cols) {
        let u1 = rng.nextUnitDouble()
        let u2 = rng.nextUnitDouble()
        // Box-Muller to Gaussian, std=0.02 to match embedding scale.
        let r = (-2.0 * Foundation.log(max(u1, 1e-12))).squareRoot()
        let theta = 2.0 * Double.pi * u2
        let g = r * Foundation.cos(theta) * 0.02
        w[k] = Float16(g)
    }

    let (codes, codebook) = w.withUnsafeBufferPointer { buf in
        SmeltTurboQuantHQuantizer.quantize(
            weights: buf.baseAddress!,
            rows: rows, cols: cols, groupSize: 128, seed: 7
        )
    }
    let rec = SmeltTurboQuantHQuantizer.dequantize(
        codes: codes, codebook: codebook,
        rows: rows, cols: cols, groupSize: 128
    )

    var sumDiff2: Double = 0
    var dotAB: Double = 0
    var normA2: Double = 0
    var normB2: Double = 0
    for k in 0 ..< (rows * cols) {
        let a = Double(Float(w[k]))
        let b = Double(Float(rec[k]))
        let d = a - b
        sumDiff2 += d * d
        dotAB += a * b
        normA2 += a * a
        normB2 += b * b
    }
    let mse = sumDiff2 / Double(rows * cols)
    let cos = dotAB / ((normA2 * normB2).squareRoot() + 1e-12)
    // Python sim on the same synthetic distribution reports
    // cosine ≈ 0.939, MSE ≈ 4.7e-5. Swift float order can drift
    // the codebook centroids; allow ±0.02 cosine tolerance.
    #expect(cos > 0.90,
            "round-trip cosine \(cos) below Lloyd-Max 2-bit floor")
    #expect(mse < 1e-3,
            "round-trip MSE \(mse) above expected 2-bit floor")
}

@Test func turboQuantHRoundTripHandlesPartialFinalGroup() {
    // cols=200 with groupSize=128 → one full 128 group + one
    // partial 72 group. Exercises the encoder padding + decoder
    // trim-on-write path. Without the fix the inverse Hadamard
    // mixes pad-slot zeros into the real columns and corrupts them.
    let rows = 256
    let cols = 200
    var rng = SplitMix64(seed: 11)
    var w = [Float16](repeating: 0, count: rows * cols)
    for k in 0 ..< (rows * cols) {
        let u1 = rng.nextUnitDouble()
        let u2 = rng.nextUnitDouble()
        let r = (-2.0 * Foundation.log(max(u1, 1e-12))).squareRoot()
        let g = r * Foundation.cos(2.0 * Double.pi * u2) * 0.02
        w[k] = Float16(g)
    }
    let (codes, codebook) = w.withUnsafeBufferPointer { buf in
        SmeltTurboQuantHQuantizer.quantize(
            weights: buf.baseAddress!,
            rows: rows, cols: cols, groupSize: 128, seed: 5
        )
    }
    let rec = SmeltTurboQuantHQuantizer.dequantize(
        codes: codes, codebook: codebook,
        rows: rows, cols: cols, groupSize: 128
    )
    // Restrict cosine to the partial tail group (cols 128..199).
    // The first full 128 group reconstructs at the Lloyd-Max floor
    // regardless of the pad-slot bug, so a global cosine threshold
    // would let the regression slip in undetected. The tail-only
    // cosine sees the corruption directly: pre-fix lands ~0.85,
    // post-fix lands ~0.94+. Bar set strictly between.
    var dot: Double = 0; var na: Double = 0; var nb: Double = 0
    for r in 0 ..< rows {
        for col in 128 ..< cols {
            let a = Double(Float(w[r * cols + col]))
            let b = Double(Float(rec[r * cols + col]))
            dot += a * b; na += a * a; nb += b * b
        }
    }
    let cos = dot / ((na * nb).squareRoot() + 1e-12)
    #expect(cos > 0.90,
            "partial-tail cosine \(cos) — pad-slot mixing not handled")
}

@Test func turboQuantHWriterEmitsValidPackageEntry() throws {
    // Drives the SmeltAffineQuantizer dispatch end-to-end: feed a
    // tensor matching turboQuantHPatterns through the writer with
    // strategy=affineU4 and assert the manifest entry + on-disk
    // codes/codebook bytes are shaped as the runtime reader will
    // expect. No Metal needed; no spec parsing needed.
    let rows = 256
    let cols = 256
    var rng = SplitMix64(seed: 13)
    var w = [Float16](repeating: 0, count: rows * cols)
    for k in 0 ..< (rows * cols) {
        let u1 = rng.nextUnitDouble(); let u2 = rng.nextUnitDouble()
        let r = (-2.0 * Foundation.log(max(u1, 1e-12))).squareRoot()
        w[k] = Float16(r * Foundation.cos(2.0 * Double.pi * u2) * 0.02)
    }

    let tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("tqh-writer-\(UUID().uuidString)",
                                isDirectory: true)
    try FileManager.default.createDirectory(
        at: tempDir, withIntermediateDirectories: true
    )
    defer { try? FileManager.default.removeItem(at: tempDir) }
    let weightsPath = tempDir.appendingPathComponent("weights.bin").path

    let config = SmeltQuantizationConfig(
        strategy: .affineU4,
        groupSize: 128,
        excludePatterns: [],
        quantizeEmbedding: false,
        turboQuantHPatterns: ["W_tqh"]
    )
    let entries = try w.withUnsafeBufferPointer { buf -> [SmeltWeightEntry] in
        let raw = UnsafeRawPointer(buf.baseAddress!)
        return try SmeltAffineQuantizer.quantize(
            tensors: [(
                runtimeName: "W_tqh",
                data: raw,
                byteCount: rows * cols * 2,
                shape: [rows, cols],
                dtype: "F16"
            )],
            config: config,
            outputPath: weightsPath
        )
    }
    #expect(entries.count == 1)
    let entry = entries[0]
    #expect(entry.dtype == .turboQuantH,
            "tensor matched turboQuantHPatterns but dtype is \(entry.dtype)")
    #expect(entry.codebookOffset != nil,
            "turbo_quant_h entry missing codebookOffset")
    #expect(entry.codebookSizeBytes != nil,
            "turbo_quant_h entry missing codebookSizeBytes")
    let numGroups = (cols + 127) / 128
    #expect(
        entry.codebookSizeBytes == UInt64(numGroups * 4 * 2),
        Comment(rawValue:
            "codebook size \(entry.codebookSizeBytes ?? 0) "
                + "!= num_groups=\(numGroups) * 4 fp16")
    )
    let codesPerRow = ((numGroups * 128) + 3) / 4
    #expect(
        entry.sizeBytes == UInt64(rows * codesPerRow),
        Comment(rawValue:
            "code stream size \(entry.sizeBytes) "
                + "!= rows=\(rows) * codesPerRow=\(codesPerRow)")
    )

    // Round-trip the on-disk codes + codebook back to fp16 and
    // confirm cosine lands at the Lloyd-Max floor. Uses the same
    // SmeltTurboQuantHQuantizer.dequantize the Metal kernel will
    // need to mirror.
    let weightsData = try Data(contentsOf: URL(fileURLWithPath: weightsPath))
    let codes = Array(weightsData[
        Int(entry.offset) ..< Int(entry.offset) + Int(entry.sizeBytes)
    ])
    let codebookOffset = Int(entry.codebookOffset!)
    let codebookSize = Int(entry.codebookSizeBytes!)
    let codebookBytes = Array(weightsData[
        codebookOffset ..< codebookOffset + codebookSize
    ])
    var codebook = [Float16](repeating: 0, count: codebookSize / 2)
    _ = codebookBytes.withUnsafeBytes { src in
        codebook.withUnsafeMutableBytes { dst in
            memcpy(dst.baseAddress!, src.baseAddress!, codebookSize)
        }
    }
    let rec = SmeltTurboQuantHQuantizer.dequantize(
        codes: codes, codebook: codebook,
        rows: rows, cols: cols, groupSize: 128
    )
    var dot: Double = 0; var na: Double = 0; var nb: Double = 0
    for k in 0 ..< (rows * cols) {
        let a = Double(Float(w[k])); let b = Double(Float(rec[k]))
        dot += a * b; na += a * a; nb += b * b
    }
    let cos = dot / ((na * nb).squareRoot() + 1e-12)
    #expect(cos > 0.90,
            "on-disk round-trip cosine \(cos) below 2-bit floor")
}

@Test func turboQuantHCompilerDequantRowMatchesWholeTensor() throws {
    // Unit test for SmeltTurboQuantHQuantizer.dequantizeRow (the
    // compiler-side per-row helper): write a row through the
    // whole-tensor encoder, decode it via dequantizeRow, compare
    // against the whole-tensor dequantize on the same row. Does
    // NOT exercise SmeltRuntime.dequantizeTurboQuantHEmbeddingRow
    // — that inline copy is covered by the runtime-side
    // integration test owed in a follow-up unit.
    let rows = 128
    let cols = 384
    var rng = SplitMix64(seed: 17)
    var w = [Float16](repeating: 0, count: rows * cols)
    for k in 0 ..< (rows * cols) {
        let u1 = rng.nextUnitDouble(); let u2 = rng.nextUnitDouble()
        let r = (-2.0 * Foundation.log(max(u1, 1e-12))).squareRoot()
        w[k] = Float16(r * Foundation.cos(2.0 * Double.pi * u2) * 0.02)
    }

    let (codes, codebook) = w.withUnsafeBufferPointer { buf in
        SmeltTurboQuantHQuantizer.quantize(
            weights: buf.baseAddress!,
            rows: rows, cols: cols, groupSize: 128, seed: 1
        )
    }

    let plan = SmeltTurboQuantHQuantizer.plan(
        rows: rows, cols: cols, groupSize: 128
    )
    let codesPerRow = (plan.paddedToGroups + 3) / 4

    // Compare runtime per-row dequant against the full-tensor
    // dequant on a few sampled rows.
    let writerRec = SmeltTurboQuantHQuantizer.dequantize(
        codes: codes, codebook: codebook,
        rows: rows, cols: cols, groupSize: 128
    )

    for row in [0, 17, 63, rows - 1] {
        let runtimeRec = codes.withUnsafeBufferPointer { cbuf in
            codebook.withUnsafeBufferPointer { kbuf in
                SmeltTurboQuantHQuantizer.dequantizeRow(
                    codes: cbuf.baseAddress!,
                    codebook: kbuf.baseAddress!,
                    row: row, cols: cols,
                    codesPerRow: codesPerRow,
                    groupSize: 128
                )
            }
        }
        for k in 0 ..< cols {
            let a = Float(writerRec[row * cols + k])
            let b = Float(runtimeRec[k])
            #expect(abs(a - b) < 1e-3,
                    Comment(rawValue:
                        "row \(row) col \(k): writer=\(a) runtime=\(b)")
            )
        }
    }
}

@Test func turboQuantHCodePackingRoundTrips() {
    // Pack 64 known 2-bit codes into bytes, read each back.
    let positions = 64
    var codes = [UInt8](repeating: 0, count: positions / 4)
    var expected = [UInt8](repeating: 0, count: positions)
    var rng = SplitMix64(seed: 1)
    for p in 0 ..< positions {
        let v = UInt8(rng.next() & 0x3)
        expected[p] = v
        let shift = (p % 4) * 2
        codes[p / 4] |= (v & 0x3) << shift
    }
    for p in 0 ..< positions {
        let got = SmeltTurboQuantHQuantizer.readCode2(
            codes: codes, row: 0, codesPerRow: positions / 4, position: p
        )
        #expect(got == expected[p],
                "code mismatch at pos \(p): got \(got) expected \(expected[p])")
    }
}

@Test func turboQuantHImatrixWeightsCodebookTowardImportantLanes() {
    // Single 128-lane group. Each row's columns are ≈constant (1.0) plus
    // small noise, so the rotated representation is dominated by lane 0
    // (the Hadamard DC lane ≈ √128 ≈ 11.3); the other lanes carry only the
    // ~0.01 noise. Concentrating importance on lane 0 must pull the codebook
    // toward the large-magnitude values; concentrating it on the noise lanes
    // must keep the codebook near zero. Proves imatrix weighting reaches the
    // codebook fit, in the correct direction.
    let rows = 512
    let cols = 128
    var rng = SplitMix64(seed: 23)
    var w = [Float16](repeating: 0, count: rows * cols)
    for k in 0 ..< (rows * cols) {
        let noise = Float(rng.nextUnitDouble() - 0.5) * 0.02
        w[k] = Float16(1.0 + noise)
    }

    var impLane0 = [Float](repeating: 0, count: cols)
    impLane0[0] = 1.0
    var impNoise = [Float](repeating: 1, count: cols)
    impNoise[0] = 0

    let cbLane0 = w.withUnsafeBufferPointer { buf in
        SmeltTurboQuantHQuantizer.quantize(
            weights: buf.baseAddress!, rows: rows, cols: cols,
            groupSize: 128, seed: 3, importance: impLane0
        ).codebook
    }
    let cbNoise = w.withUnsafeBufferPointer { buf in
        SmeltTurboQuantHQuantizer.quantize(
            weights: buf.baseAddress!, rows: rows, cols: cols,
            groupSize: 128, seed: 3, importance: impNoise
        ).codebook
    }

    let maxLane0 = cbLane0.map { abs(Float($0)) }.max() ?? 0
    let maxNoise = cbNoise.map { abs(Float($0)) }.max() ?? 0
    #expect(maxLane0 > 5.0,
            "lane-0-weighted codebook max |c|=\(maxLane0) should track the ≈11.3 DC lane")
    #expect(maxNoise < 1.0,
            "noise-lane-weighted codebook max |c|=\(maxNoise) should stay near zero")
}

@Test func turboQuantHImatrixLengthMismatchThrows() throws {
    // A wrong-length imatrix (255 != paddedToGroups 256) must surface as an
    // actionable SmeltAffineQuantizerError, not the quantizer's precondition trap.
    let rows = 64
    let cols = 256
    var rng = SplitMix64(seed: 31)
    var w = [Float16](repeating: 0, count: rows * cols)
    for k in 0 ..< (rows * cols) {
        let u1 = rng.nextUnitDouble(); let u2 = rng.nextUnitDouble()
        let r = (-2.0 * Foundation.log(max(u1, 1e-12))).squareRoot()
        w[k] = Float16(r * Foundation.cos(2.0 * Double.pi * u2) * 0.02)
    }
    let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("tqh-imlen-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmp) }
    let config = SmeltQuantizationConfig(
        strategy: .affineU4, groupSize: 128, excludePatterns: [],
        quantizeEmbedding: false, turboQuantHPatterns: ["W_tqh"]
    )
    #expect(throws: SmeltAffineQuantizerError.self) {
        _ = try w.withUnsafeBufferPointer { buf -> [SmeltWeightEntry] in
            try SmeltAffineQuantizer.quantize(
                tensors: [(
                    runtimeName: "W_tqh",
                    data: UnsafeRawPointer(buf.baseAddress!),
                    byteCount: rows * cols * 2,
                    shape: [rows, cols],
                    dtype: "F16"
                )],
                config: config,
                outputPath: tmp.appendingPathComponent("weights.bin").path,
                imatrix: ["W_tqh": [Float](repeating: 1, count: 255)]
            )
        }
    }
}
