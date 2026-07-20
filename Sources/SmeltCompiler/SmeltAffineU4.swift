// SmeltAffineU4 — the shared per-row affine u4 (int4) quant/dequant numerics.
//
// One implementation of the group-wise affine scheme, used by the LLM quantizer
// (SmeltAffineQuantizer), the Qwen3-TTS package builder, and the kernel parity tests.
// Storage per row:
//   packed nibbles: (cols+1)/2 bytes  (2 per byte, LOW nibble = even column)
//   scales / biases: G = ceil(cols/groupSize) fp16 each
// Quantize:   scale = (max-min)/15, bias = min, nibble = round((v-bias)/scale) ∈ [0,15]
// Dequantize: v ≈ nibble · fp16(scale) + fp16(bias)   (scale/bias read back through fp16,
//             matching what the GPU kernel widens — so a parity test reproduces the kernel).

import Foundation

public enum SmeltAffineU4 {

    @inlinable
    public static func numGroups(cols: Int, groupSize: Int) -> Int {
        (cols + groupSize - 1) / groupSize
    }

    /// Bytes per packed row: 2 nibbles per byte.
    @inlinable
    public static func packedRowStride(cols: Int) -> Int { (cols + 1) / 2 }

    /// Affine quant params for an endpoint range [lo, hi]: the full-precision scale/bias used to compute
    /// the nibble, plus their fp16 bit patterns (what the block stores and the kernel widens).
    /// `scale = (hi−lo)/15`, `bias = lo`; a degenerate group (range < 1e-12) uses scale = 1. The one
    /// place this formula lives — shared by the affine path and SmeltGPTQ so they can't drift.
    @inlinable
    public static func affineParams(lo: Float, hi: Float) -> (scale: Float, bias: Float, scaleBits: UInt16, biasBits: UInt16) {
        let range = hi - lo
        let scale: Float = range < 1e-12 ? 1.0 : range / 15.0
        return (scale, lo, Float16(scale).bitPattern, Float16(lo).bitPattern)
    }

    /// Run `body` with a pointer into `imp` (or nil if absent), keeping the array alive for the call.
    @usableFromInline
    static func withOptionalImportance<R>(_ imp: [Float]?, _ body: (UnsafePointer<Float>?) -> R) -> R {
        if let imp { return imp.withUnsafeBufferPointer { body($0.baseAddress) } }
        return body(nil)
    }

    /// Quantize one row of `cols` floats into `packed` ((cols+1)/2 bytes) plus `scales`/`biases`
    /// (G fp16 bit patterns each). Buffers must be sized by the caller. Packed bytes are written
    /// in full (no read-modify-write), so the destination need not be pre-zeroed.
    /// How a group's quantization endpoints [lo, hi] are chosen.
    public enum ClipMode: Sendable {
        /// lo = group min, hi = group max — exact min/max affine (the LLM path; bit-stable).
        case minMax
        /// Shrink the [min, max] interval toward its center to minimize fp16-roundtrip reconstruction
        /// MSE: a few outliers clamp, but the bulk gets finer step size. A strict improvement over
        /// minMax (the c=1.0 candidate IS minMax), at build-time grid-search cost only.
        case mseOptimal
    }

    /// The endpoints [lo, hi] for one group under `clip`. mseOptimal grid-searches a shrink factor of
    /// the [min,max] interval, scoring each by reconstruction MSE through the SAME fp16 scale/bias the
    /// kernel reads (so the chosen endpoints minimize the error the runtime actually sees).
    @usableFromInline
    static func clipEndpoints(_ values: UnsafePointer<Float>, _ colStart: Int, _ colEnd: Int,
                              _ minVal: Float, _ maxVal: Float, _ clip: ClipMode,
                              _ importance: UnsafePointer<Float>?) -> (lo: Float, hi: Float) {
        switch clip {
        case .minMax:
            return (minVal, maxVal)
        case .mseOptimal:
            let half = (maxVal - minVal) / 2
            if half < 1e-12 { return (minVal, maxVal) }
            var bestLo = minVal, bestHi = maxVal, bestErr = Float.infinity
            // c = 1.0 (EXACTLY min/max) down to 0.4; finer shrink = clamp more outliers, finer bulk
            // steps. Shrinking from min/max (not center±c·half) keeps c=1.0 bit-identical to min/max, so
            // the search can only improve on it.
            for step in 0...8 {
                let c = 1.0 - Float(step) * 0.075
                let lo = minVal + (1 - c) * half, hi = maxVal - (1 - c) * half
                let scale = (hi - lo) / 15.0
                let scaleF = Float(Float16(scale)), loF = Float(Float16(lo))   // kernel reads fp16
                let inv = 1.0 / scale
                var err: Float = 0
                for k in colStart..<colEnd {
                    let v = values[k]
                    let nib = min(max(((v - lo) * inv).rounded(), 0.0), 15.0)
                    let d = v - (nib * scaleF + loF)
                    err += (importance?[k] ?? 1.0) * d * d
                }
                if err < bestErr { bestErr = err; bestLo = lo; bestHi = hi }
            }
            return (bestLo, bestHi)
        }
    }

    /// `importance` (length `cols`, per input channel `h_k = E[x_k²]`) activation-weights the mseOptimal
    /// endpoint search; nil = unweighted. Ignored for `.minMax`.
    public static func quantizeRow(
        values: UnsafePointer<Float>,
        cols: Int,
        groupSize: Int,
        clip: ClipMode = .minMax,
        importance: UnsafePointer<Float>? = nil,
        packed: UnsafeMutablePointer<UInt8>,
        scales: UnsafeMutablePointer<UInt16>,
        biases: UnsafeMutablePointer<UInt16>
    ) {
        let groups = numGroups(cols: cols, groupSize: groupSize)
        for g in 0..<groups {
            let colStart = g * groupSize
            let colEnd = min(colStart + groupSize, cols)

            var minVal: Float = .infinity
            var maxVal: Float = -.infinity
            for c in colStart..<colEnd {
                let v = values[c]
                if v < minVal { minVal = v }
                if v > maxVal { maxVal = v }
            }

            let (lo, hi) = clipEndpoints(values, colStart, colEnd, minVal, maxVal, clip, importance)
            let p = affineParams(lo: lo, hi: hi)
            let scale = p.scale, bias = p.bias
            scales[g] = p.scaleBits
            biases[g] = p.biasBits

            let invScale = 1.0 / scale
            for c in colStart..<colEnd {
                let normalized = (values[c] - bias) * invScale
                let clamped = min(max(normalized, 0.0), 15.0)
                let nibble = UInt8(clamped.rounded())
                let byteIdx = c >> 1
                if c & 1 == 0 {
                    packed[byteIdx] = nibble                      // low nibble, clears high
                } else {
                    packed[byteIdx] |= nibble << 4               // high nibble
                }
            }
        }
    }

    /// Dequantize one row back to `cols` floats, reading scale/bias through fp16 exactly as the
    /// GPU kernel does — the reference for kernel parity and for the CPU embedding-gather path
    /// (hot at runtime, so scale/bias are loaded once per group, not per element).
    public static func dequantizeRow(
        packed: UnsafePointer<UInt8>,
        scales: UnsafePointer<UInt16>,
        biases: UnsafePointer<UInt16>,
        cols: Int,
        groupSize: Int,
        out: UnsafeMutablePointer<Float>
    ) {
        let groups = numGroups(cols: cols, groupSize: groupSize)
        for g in 0..<groups {
            let colStart = g * groupSize
            let colEnd = min(colStart + groupSize, cols)
            let scale = Float(Float16(bitPattern: scales[g]))
            let bias = Float(Float16(bitPattern: biases[g]))
            for c in colStart..<colEnd {
                let byte = packed[c >> 1]
                let nibble = (c & 1 == 0) ? (byte & 0x0F) : (byte >> 4)
                out[c] = Float(nibble) * scale + bias
            }
        }
    }

    // MARK: - Whole-tensor convenience (tests / small tensors)

    public struct Packed {
        public let nibbles: [UInt8]   // rows * ((cols+1)/2)
        public let scales: [UInt16]   // rows * G  (fp16 bits)
        public let biases: [UInt16]   // rows * G  (fp16 bits)
        public let rows: Int
        public let cols: Int
        public let groupSize: Int

        @inlinable public var rowStride: Int { SmeltAffineU4.packedRowStride(cols: cols) }
        @inlinable public var groups: Int { SmeltAffineU4.numGroups(cols: cols, groupSize: groupSize) }
    }

    /// Quantize a row-major [rows×cols] tensor. Convenience over `quantizeRow`.
    public static func quantize(_ weights: [Float], rows: Int, cols: Int, groupSize: Int,
                                clip: ClipMode = .minMax, importance: [Float]? = nil) -> Packed {
        precondition(weights.count == rows * cols, "weights.count != rows*cols")
        precondition(importance == nil || importance!.count == cols, "importance.count != cols")
        let groups = numGroups(cols: cols, groupSize: groupSize)
        let rowStride = (cols + 1) / 2
        var nibbles = [UInt8](repeating: 0, count: rows * rowStride)
        var scales = [UInt16](repeating: 0, count: rows * groups)
        var biases = [UInt16](repeating: 0, count: rows * groups)
        weights.withUnsafeBufferPointer { w in
            nibbles.withUnsafeMutableBufferPointer { p in
                scales.withUnsafeMutableBufferPointer { s in
                    biases.withUnsafeMutableBufferPointer { b in
                        withOptionalImportance(importance) { imp in
                            for r in 0..<rows {
                                quantizeRow(
                                    values: w.baseAddress! + r * cols, cols: cols, groupSize: groupSize,
                                    clip: clip, importance: imp,
                                    packed: p.baseAddress! + r * rowStride,
                                    scales: s.baseAddress! + r * groups,
                                    biases: b.baseAddress! + r * groups)
                            }
                        }
                    }
                }
            }
        }
        return Packed(nibbles: nibbles, scales: scales, biases: biases,
                      rows: rows, cols: cols, groupSize: groupSize)
    }

    /// Dequantize a Packed tensor back to row-major [rows×cols] floats.
    public static func dequantize(_ p: Packed) -> [Float] {
        var out = [Float](repeating: 0, count: p.rows * p.cols)
        out.withUnsafeMutableBufferPointer { o in
            p.nibbles.withUnsafeBufferPointer { n in
                p.scales.withUnsafeBufferPointer { s in
                    p.biases.withUnsafeBufferPointer { b in
                        for r in 0..<p.rows {
                            dequantizeRow(
                                packed: n.baseAddress! + r * p.rowStride,
                                scales: s.baseAddress! + r * p.groups,
                                biases: b.baseAddress! + r * p.groups,
                                cols: p.cols, groupSize: p.groupSize,
                                out: o.baseAddress! + r * p.cols)
                        }
                    }
                }
            }
        }
        return out
    }
}
