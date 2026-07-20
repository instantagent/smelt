// SmeltTurboQuantHQuantizer — CPU-only TurboQuant-H 2.125-bpw
// weight quantization for large 2-D tensors.
//
// Algorithm (per cactus-compute/cactus/blog/turboquant-h.md):
//
//   For each row in W: partition into P groups of G=128 contiguous
//   elements. Apply the normalized Sylvester Hadamard H_G to each
//   group in-place; H_G is self-inverse so dequant uses the same
//   butterfly. For each group index p ∈ [0, P), fit 4 centroids
//   via k-means over the rotated values across ALL rows for that
//   group. Encode every weight as the 2-bit index of its nearest
//   centroid. Dequant: scatter codes -> centroids -> inverse
//   Hadamard.
//
// Storage:
//   Codes:     [R, ceil(C * 2 bits / 8)] uint8, packed 4 per byte
//              from LSB up (positions 0..3 → bits 0..7).
//   Codebooks: [P, 4] fp16, one 4-centroid table per group.
//
// Effective bpw = 2 + 64/(R*G) ≈ 2 for R >> 16.
//
// Validated against the Python reference at
// tools/turboquant-h-sim.py — same Lloyd-Max 2-bit cosine
// floor (~0.939 on Gaussian, ~0.94 on real model weights).

import Foundation
import SmeltSchema

public struct SmeltTurboQuantHQuantizer {

    public struct Plan {
        public let rows: Int
        public let cols: Int
        public let groupSize: Int
        public let codesSizeBytes: UInt64
        public let codebookSizeBytes: UInt64

        public var numGroups: Int {
            SmeltTurboQuantHCodec.numGroups(cols: cols, groupSize: groupSize)
        }
        public var paddedToGroups: Int { numGroups * groupSize }
    }

    public static func plan(rows: Int, cols: Int, groupSize: Int) -> Plan {
        let numGroups = SmeltTurboQuantHCodec.numGroups(
            cols: cols, groupSize: groupSize
        )
        let padded = numGroups * groupSize
        // 2 bpw codes, rounded up to byte boundary per row. Pad each
        // row to padded-cols nibble positions (the partial final
        // group's tail slots get zero codes that won't affect
        // dequant after the trim).
        let codesPerRow = (padded + 3) / 4
        return Plan(
            rows: rows, cols: cols, groupSize: groupSize,
            codesSizeBytes: UInt64(rows * codesPerRow),
            codebookSizeBytes: UInt64(numGroups * 4 * 2)
        )
    }

    /// Quantize a [rows, cols] fp16 weight tensor.
    ///
    /// Returns (codesBytes, codebookBytes) suitable for direct
    /// writing into a Smelt package's weights.bin.
    public static func quantize(
        weights: UnsafePointer<Float16>,
        rows: Int,
        cols: Int,
        groupSize: Int = 128,
        seed: UInt64 = 0,
        codebookIters: Int = 16,
        codebookSampleCap: Int = 200_000,
        importance: [Float]? = nil
    ) -> (codes: [UInt8], codebook: [Float16]) {
        let plan = plan(rows: rows, cols: cols, groupSize: groupSize)
        let p = plan.numGroups
        let g = groupSize
        let padded = plan.paddedToGroups
        let codesPerRow = (padded + 3) / 4

        precondition(g > 0 && (g & (g - 1)) == 0,
                     "TurboQuant-H groupSize must be a power of two")
        if let importance {
            precondition(importance.count == padded,
                "TurboQuant-H importance length must equal paddedToGroups")
            // Finite + non-negative by construction (squared moments); enforce
            // it so a misused signed/inf input can't corrupt the weighted
            // seeding walk, drive wsum negative, or inject inf/NaN.
            precondition(importance.allSatisfy { $0.isFinite && $0 >= 0 },
                "TurboQuant-H importance must be finite and non-negative")
        }
        let hadamardScale = Float(1.0 / Double(g).squareRoot())

        var codebook = [Float16](repeating: 0, count: p * 4)
        var codes = [UInt8](repeating: 0, count: rows * codesPerRow)

        // Stream per group: rotate that group's column slice across
        // all rows into a per-group [rows, g] scratch buffer, fit the
        // codebook, encode every row's codes for that group, free the
        // scratch. Avoids materializing the full rows*padded float
        // surface (5.4GB+ on large PLI shapes).
        //
        // Per-group work is independent: each iteration reads its
        // own column slice of `weights`, writes to its own
        // `codebook[gIdx*4..]` slot, and writes to its own
        // disjoint byte ranges in `codes` (group g occupies bytes
        // `r*codesPerRow + g*(G/4) .. r*codesPerRow + (g+1)*(G/4)-1`
        // per row). Parallelize across CPU cores via
        // concurrentPerform — ~10× speedup on M5 Max's 12 cores
        // for FFN-shape matrices.
        // Each concurrentPerform task owns one group `gIdx` and writes
        // only into its own disjoint slices: `codebook[gIdx*4 ..< gIdx*4+4]`
        // and, per row, the byte range `r*codesPerRow + gIdx*(g/4) ..<
        // r*codesPerRow + (gIdx+1)*(g/4)` (g is a power of two ≥ 4, so group
        // boundaries land on whole-byte/4-position boundaries — no two tasks
        // touch the same code byte). `weights` is read-only. Disjoint writes
        // + read-only shared input make `nonisolated(unsafe)` sound here.
        nonisolated(unsafe) let weights = weights
        codebook.withUnsafeMutableBufferPointer { codebookBuf in
            codes.withUnsafeMutableBufferPointer { codesBuf in
                nonisolated(unsafe) let codebookBase = codebookBuf.baseAddress!
                nonisolated(unsafe) let codesBase = codesBuf.baseAddress!
                DispatchQueue.concurrentPerform(iterations: p) { gIdx in
                    var rotated = [Float](repeating: 0, count: rows * g)
                    let groupColBase = gIdx * g
                    rotateGroupAcrossRows(
                        dest: &rotated,
                        weights: weights,
                        rows: rows, cols: cols,
                        groupColBase: groupColBase,
                        groupSize: g,
                        hadamardScale: hadamardScale
                    )

                    // Per-lane importance for this group's G rotated lanes
                    // (importance is keyed by rotated lane, shared across rows).
                    let groupImportance = importance.map {
                        Array($0[gIdx * g ..< gIdx * g + g])
                    }
                    var rng = SplitMix64(seed: seed &+ UInt64(gIdx))
                    let centroids = fitCodebook(
                        rotated: rotated, rows: rows, groupSize: g,
                        sampleCap: codebookSampleCap, iters: codebookIters,
                        importance: groupImportance, rng: &rng
                    )
                    for k in 0 ..< 4 {
                        codebookBase[gIdx * 4 + k] = centroids[k]
                    }

                    encodeGroup(
                        rotated: rotated, codebook: centroids,
                        rows: rows, groupSize: g,
                        groupIndex: gIdx,
                        codes: codesBase, codesPerRow: codesPerRow
                    )
                }
            }
        }
        return (codes, codebook)
    }

    /// Dequantize ONE row directly from packed code bytes + a
    /// per-group codebook. Mirrors the runtime's per-row gather
    /// path: the caller hands in raw byte pointers into a Smelt
    /// package's weights.bin and gets [cols] fp16 back.
    ///
    /// codesPerRow = ((numGroups * groupSize) + 3) / 4 — same as
    /// SmeltWeightEntry.packedRowStride for a turboQuantH entry.
    public static func dequantizeRow(
        codes: UnsafePointer<UInt8>,
        codebook: UnsafePointer<Float16>,
        row: Int,
        cols: Int,
        codesPerRow: Int,
        groupSize: Int = 128
    ) -> [Float16] {
        let p = SmeltTurboQuantHCodec.numGroups(cols: cols, groupSize: groupSize)
        precondition(groupSize > 0 && (groupSize & (groupSize - 1)) == 0,
                     "TurboQuant-H groupSize must be a power of two")
        let hadamardScale = Float(1.0 / Double(groupSize).squareRoot())
        let rowBase = row * codesPerRow

        var out = [Float16](repeating: 0, count: cols)
        var groupBuf = [Float](repeating: 0, count: groupSize)
        for gIdx in 0 ..< p {
            for j in 0 ..< groupSize {
                let pos = gIdx * groupSize + j
                let byte = codes[rowBase + pos / 4]
                let shift = (pos % 4) * 2
                let code = Int((byte >> shift) & 0x3)
                groupBuf[j] = Float(codebook[gIdx * 4 + code])
            }
            fastWalshHadamard(buffer: &groupBuf, size: groupSize)
            for j in 0 ..< groupSize {
                let pos = gIdx * groupSize + j
                if pos < cols {
                    out[pos] = Float16(groupBuf[j] * hadamardScale)
                }
            }
        }
        return out
    }

    public static func dequantize(
        codes: [UInt8],
        codebook: [Float16],
        rows: Int,
        cols: Int,
        groupSize: Int = 128
    ) -> [Float16] {
        let plan = plan(rows: rows, cols: cols, groupSize: groupSize)
        let g = groupSize
        let padded = plan.paddedToGroups
        let p = plan.numGroups
        let codesPerRow = (padded + 3) / 4

        precondition(g > 0 && (g & (g - 1)) == 0,
                     "TurboQuant-H groupSize must be a power of two")
        let hadamardScale = Float(1.0 / Double(g).squareRoot())

        var out = [Float16](repeating: 0, count: rows * cols)
        var groupBuf = [Float](repeating: 0, count: g)
        for r in 0 ..< rows {
            for gIdx in 0 ..< p {
                // Always read the full G-wide group. Inverse Hadamard
                // mixes all G entries to produce each output, so the
                // encoder's pad-slot codes (driven by zero inputs at
                // encode time) are load-bearing on the inverse. Only
                // trim at the OUTPUT write site.
                for j in 0 ..< g {
                    let pos = gIdx * g + j
                    let code = readCode2(
                        codes: codes, row: r, codesPerRow: codesPerRow,
                        position: pos
                    )
                    groupBuf[j] = Float(codebook[gIdx * 4 + Int(code)])
                }
                fastWalshHadamard(buffer: &groupBuf, size: g)
                for j in 0 ..< g {
                    let pos = gIdx * g + j
                    if pos < cols {
                        out[r * cols + pos] = Float16(groupBuf[j] * hadamardScale)
                    }
                }
            }
        }
        return out
    }

    // MARK: - Hadamard

    /// In-place fast Walsh-Hadamard transform on a power-of-two
    /// buffer. Produces the UN-normalized Hadamard product H_n · x;
    /// the caller multiplies by 1/√n to land on the orthonormal form.
    /// O(n log n) — n × log2(n) adds/subs, no multiplies on the
    /// rotation itself, vs the explicit matrix-vector form which is
    /// O(n²) on the same shape (16K vs 896 ops at n=128).
    /// Forwards to SmeltTurboQuantHCodec.fastWalshHadamard so the
    /// runtime decode path and the compiler-side encode share one
    /// implementation.
    static func fastWalshHadamard(buffer: inout [Float], size n: Int) {
        SmeltTurboQuantHCodec.fastWalshHadamard(buffer: &buffer, size: n)
    }

    /// Rotate one group's column slice (cols `groupColBase ..
    /// groupColBase + groupSize` clamped to `cols`) across every row
    /// into `dest[r * groupSize .. r * groupSize + groupSize]`.
    /// Streaming form: avoids the rows × padded surface and reads
    /// the source mmap region exactly once.
    private static func rotateGroupAcrossRows(
        dest: inout [Float],
        weights: UnsafePointer<Float16>,
        rows: Int, cols: Int,
        groupColBase: Int, groupSize: Int,
        hadamardScale: Float
    ) {
        var scratch = [Float](repeating: 0, count: groupSize)
        for r in 0 ..< rows {
            for j in 0 ..< groupSize {
                let col = groupColBase + j
                scratch[j] = col < cols ? Float(weights[r * cols + col]) : 0
            }
            fastWalshHadamard(buffer: &scratch, size: groupSize)
            for j in 0 ..< groupSize {
                dest[r * groupSize + j] = scratch[j] * hadamardScale
            }
        }
    }

    // MARK: - Codebook fitting

    /// Fit a 4-centroid codebook over rotated values for ONE group,
    /// reading directly from the contiguous per-group rotated buffer
    /// (shape [rows * groupSize]).
    private static func fitCodebook(
        rotated: [Float], rows: Int, groupSize: Int,
        sampleCap: Int, iters: Int,
        importance: [Float]?, rng: inout SplitMix64
    ) -> [Float16] {
        // Normalize per-lane weights to mean 1. Centroids (Σλv/Σλ) and the
        // k-means++ probabilities depend only on weight ratios, so the fit is
        // unchanged, but it becomes invariant to the imatrix's absolute scale
        // and the degeneracy thresholds stay scale-stable.
        let importance = importance.map { imp -> [Float] in
            let sum = imp.reduce(0, +)
            guard sum > 0 else { return imp }
            let inv = Float(imp.count) / sum
            return imp.map { $0 * inv }
        }
        let totalValues = rows * groupSize
        let sampleSize = min(totalValues, sampleCap)
        var sample = [Float](repeating: 0, count: sampleSize)
        // Each rotated value at flat index k lives on lane `k % groupSize`;
        // its weight is that lane's importance (same for every row).
        var weights = importance == nil
            ? nil : [Float](repeating: 0, count: sampleSize)
        if sampleSize == totalValues {
            for k in 0 ..< totalValues {
                sample[k] = rotated[k]
                if let imp = importance { weights![k] = imp[k % groupSize] }
            }
        } else {
            for k in 0 ..< sampleSize {
                let pick = Int(rng.next() % UInt64(totalValues))
                sample[k] = rotated[pick]
                if let imp = importance { weights![k] = imp[pick % groupSize] }
            }
        }
        return fitCodebookScalar(
            values: sample, weights: weights, iters: iters, rng: &rng
        )
    }

    /// Lloyd-Max 2-bit fit. With `weights == nil` this is plain (unweighted)
    /// k-means. With per-value weights (importance) it minimizes the
    /// importance-weighted objective
    /// `Σ_i w_i (v_i − c)²`: weighted k-means++ seeding and weighted centroid
    /// means; nearest-centroid assignment is unchanged (w_i scales every
    /// candidate distance equally, so it does not affect the argmin).
    private static func fitCodebookScalar(
        values: [Float], weights: [Float]?, iters: Int, rng: inout SplitMix64
    ) -> [Float16] {
        var centroids = [Float](repeating: 0, count: 4)
        // First centroid: uniform (unweighted) or ∝ weight (weighted).
        if let weights {
            var total: Float = 0
            for w in weights { total += w }
            if total < 1e-12 {
                centroids[0] = values[Int(rng.next() % UInt64(values.count))]
            } else {
                let pick = Float(rng.nextUnitDouble()) * total
                var run: Float = 0
                var chosen = 0
                for i in 0 ..< weights.count {
                    run += weights[i]
                    if run >= pick { chosen = i; break }
                }
                centroids[0] = values[chosen]
            }
        } else {
            centroids[0] = values[Int(rng.next() % UInt64(values.count))]
        }
        for c in 1 ..< 4 {
            var score = [Float](repeating: 0, count: values.count)
            for i in 0 ..< values.count {
                var best: Float = .greatestFiniteMagnitude
                for k in 0 ..< c {
                    let d = values[i] - centroids[k]
                    let dd = d * d
                    if dd < best { best = dd }
                }
                score[i] = (weights?[i] ?? 1) * best
            }
            var total: Float = 0
            for s in score { total += s }
            if total < 1e-12 {
                centroids[c] = values[Int(rng.next() % UInt64(values.count))]
            } else {
                let pick = Float(rng.nextUnitDouble()) * total
                var run: Float = 0
                var chosen = 0
                for i in 0 ..< values.count {
                    run += score[i]
                    if run >= pick { chosen = i; break }
                }
                centroids[c] = values[chosen]
            }
        }

        for _ in 0 ..< iters {
            var sum = [Float](repeating: 0, count: 4)
            var wsum = [Float](repeating: 0, count: 4)
            for i in 0 ..< values.count {
                let v = values[i]
                var bestIdx = 0
                var bestDist: Float = .greatestFiniteMagnitude
                for k in 0 ..< 4 {
                    let d = abs(v - centroids[k])
                    if d < bestDist { bestDist = d; bestIdx = k }
                }
                let w = weights?[i] ?? 1
                sum[bestIdx] += w * v
                wsum[bestIdx] += w
            }
            var changed = false
            for k in 0 ..< 4 where wsum[k] > 0 {
                let m = sum[k] / wsum[k]
                if abs(m - centroids[k]) > 1e-6 { changed = true }
                centroids[k] = m
            }
            if !changed { break }
        }
        centroids.sort()
        return centroids.map { Float16($0) }
    }

    // MARK: - Code packing

    /// Encode one group's column slice across all rows. Codes pack
    /// 4-per-byte: position p in row r lives at bit-offset 2*(p % 4)
    /// of `codes[r * codesPerRow + p/4]`, LSB-first.
    private static func encodeGroup(
        rotated: [Float], codebook: [Float16],
        rows: Int, groupSize: Int,
        groupIndex: Int,
        codes: UnsafeMutablePointer<UInt8>, codesPerRow: Int
    ) {
        var cb = [Float](repeating: 0, count: 4)
        for k in 0 ..< 4 { cb[k] = Float(codebook[k]) }
        for r in 0 ..< rows {
            for j in 0 ..< groupSize {
                let v = rotated[r * groupSize + j]
                var bestK: UInt8 = 0
                var bestDist: Float = .greatestFiniteMagnitude
                for k in 0 ..< 4 {
                    let d = abs(v - cb[k])
                    if d < bestDist {
                        bestDist = d
                        bestK = UInt8(k)
                    }
                }
                let pos = groupIndex * groupSize + j
                let byte = r * codesPerRow + pos / 4
                let shift = (pos % 4) * 2
                codes[byte] |= (bestK & 0x3) << shift
            }
        }
    }

    static func readCode2(
        codes: [UInt8], row: Int, codesPerRow: Int, position: Int
    ) -> UInt8 {
        let byte = codes[row * codesPerRow + position / 4]
        let shift = (position % 4) * 2
        return (byte >> shift) & 0x3
    }
}

/// SplitMix64: small deterministic PRNG used for codebook init.
/// Avoids pulling Foundation's Darwin RNG which complicates parity
/// against the Python reference.
struct SplitMix64 {
    var state: UInt64

    init(seed: UInt64) { self.state = seed &+ 0x9E37_79B9_7F4A_7C15 }

    mutating func next() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func nextUnitDouble() -> Double {
        Double(next() >> 11) / Double(UInt64(1) << 53)
    }
}
