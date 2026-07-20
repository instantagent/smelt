// Runtime-side TurboQuant-H embedding tests. Closes the drift
// window the compiler-side TurboQuantHRoundTripTests left open:
// SmeltRuntime's dequantizeTurboQuantHEmbeddingRow now forwards
// to SmeltTurboQuantHCodec.dequantizeRowInto, so a parity test
// against the shared codec confirms both legs land at the same
// bytes for the same on-disk input.

import Foundation
import Testing
@testable import SmeltCompiler
@testable import SmeltRuntime
import SmeltSchema

@Test func codecRowMatchesWriterOnAlignedCols() throws {
    // Encode a synthetic tensor via the compiler-side quantizer,
    // drive the shared codec per-row, compare to the writer's
    // whole-tensor dequant. The runtime call site
    // (SmeltRuntime.dequantizeTurboQuantHEmbeddingRow) is a thin
    // wrapper that forwards to this codec after manifest guards;
    // an integration test through embedToken still needs a built
    // package fixture — owed in a follow-up unit.
    let rows = 64
    let cols = 384
    var rng = SplitMix64(seed: 23)
    var w = [Float16](repeating: 0, count: rows * cols)
    for k in 0 ..< (rows * cols) {
        let u1 = rng.nextUnitDouble(); let u2 = rng.nextUnitDouble()
        let r = (-2.0 * Foundation.log(max(u1, 1e-12))).squareRoot()
        w[k] = Float16(r * Foundation.cos(2.0 * Double.pi * u2) * 0.02)
    }
    let (codes, codebook) = w.withUnsafeBufferPointer { buf in
        SmeltTurboQuantHQuantizer.quantize(
            weights: buf.baseAddress!,
            rows: rows, cols: cols, groupSize: 128, seed: 5
        )
    }
    let plan = SmeltTurboQuantHQuantizer.plan(
        rows: rows, cols: cols, groupSize: 128
    )
    let codesPerRow = (plan.paddedToGroups + 3) / 4

    for row in [0, 17, rows - 1] {
        let rowStart = row * codesPerRow
        let codecOut = codes.withUnsafeBufferPointer { cbuf in
            codebook.withUnsafeBufferPointer { kbuf in
                var dst = [UInt16](repeating: 0, count: cols)
                dst.withUnsafeMutableBufferPointer { obuf in
                    SmeltTurboQuantHCodec.dequantizeRowInto(
                        codes: cbuf.baseAddress!.advanced(by: rowStart),
                        codebook: UnsafeRawPointer(kbuf.baseAddress!)
                            .assumingMemoryBound(to: UInt16.self),
                        output: obuf.baseAddress!,
                        cols: cols, groupSize: 128
                    )
                }
                return dst
            }
        }
        let writerRec = SmeltTurboQuantHQuantizer.dequantize(
            codes: codes, codebook: codebook,
            rows: rows, cols: cols, groupSize: 128
        )
        for k in 0 ..< cols {
            let a = Float(Float16(bitPattern: codecOut[k]))
            let b = Float(writerRec[row * cols + k])
            #expect(abs(a - b) < 1e-3,
                    Comment(rawValue:
                        "row \(row) col \(k): codec=\(a) writer=\(b)"))
        }
    }
}

@Test func codecRowHandlesPartialFinalGroup() throws {
    // cols not a multiple of groupSize: codec must read all G
    // codes per group (pad slots are load-bearing on the inverse
    // Hadamard) but trim writes to `cols`. Covers the
    // `if pos < cols` write guard.
    let rows = 32
    let cols = 200  // one full 128 group + one partial 72 group
    var rng = SplitMix64(seed: 41)
    var w = [Float16](repeating: 0, count: rows * cols)
    for k in 0 ..< (rows * cols) {
        let u1 = rng.nextUnitDouble(); let u2 = rng.nextUnitDouble()
        let r = (-2.0 * Foundation.log(max(u1, 1e-12))).squareRoot()
        w[k] = Float16(r * Foundation.cos(2.0 * Double.pi * u2) * 0.02)
    }
    let (codes, codebook) = w.withUnsafeBufferPointer { buf in
        SmeltTurboQuantHQuantizer.quantize(
            weights: buf.baseAddress!,
            rows: rows, cols: cols, groupSize: 128, seed: 9
        )
    }
    let plan = SmeltTurboQuantHQuantizer.plan(
        rows: rows, cols: cols, groupSize: 128
    )
    let codesPerRow = (plan.paddedToGroups + 3) / 4
    let writerRec = SmeltTurboQuantHQuantizer.dequantize(
        codes: codes, codebook: codebook,
        rows: rows, cols: cols, groupSize: 128
    )
    for row in [0, 1, rows - 1] {
        let rowStart = row * codesPerRow
        var dst = [UInt16](repeating: 0xFFFF, count: cols + 8)
        codes.withUnsafeBufferPointer { cbuf in
            codebook.withUnsafeBufferPointer { kbuf in
                dst.withUnsafeMutableBufferPointer { obuf in
                    SmeltTurboQuantHCodec.dequantizeRowInto(
                        codes: cbuf.baseAddress!.advanced(by: rowStart),
                        codebook: UnsafeRawPointer(kbuf.baseAddress!)
                            .assumingMemoryBound(to: UInt16.self),
                        output: obuf.baseAddress!,
                        cols: cols, groupSize: 128
                    )
                }
            }
        }
        // Bytes past cols stay 0xFFFF — the codec must not over-write.
        for k in cols ..< (cols + 8) {
            #expect(dst[k] == 0xFFFF,
                    "codec wrote past cols at row \(row) col \(k)")
        }
        for k in 0 ..< cols {
            let a = Float(Float16(bitPattern: dst[k]))
            let b = Float(writerRec[row * cols + k])
            #expect(abs(a - b) < 1e-3,
                    Comment(rawValue:
                        "row \(row) col \(k): codec=\(a) writer=\(b)"))
        }
    }
}
