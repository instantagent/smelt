import Foundation
import Testing
@testable import SmeltCompiler
import SmeltSchema

// Exercises the general GPTQ build seam: SmeltAffineQuantizer.quantize(gptqBlocks:)
// writes precomputed affine_u4 blocks verbatim and enforces full coverage.

private let rows = 4
private let cols = 6
private let groupSize = 2

private func makeBlock(seed: Int) -> SmeltAffineU4.Packed {
    let rowStride = (cols + 1) / 2
    let groups = SmeltAffineU4.numGroups(cols: cols, groupSize: groupSize)
    return SmeltAffineU4.Packed(
        nibbles: (0..<rows * rowStride).map { UInt8(($0 + seed) & 0xFF) },
        scales: (0..<rows * groups).map { UInt16(truncatingIfNeeded: 1000 + seed * 100 + $0) },
        biases: (0..<rows * groups).map { UInt16(truncatingIfNeeded: 2000 + seed * 100 + $0) },
        rows: rows, cols: cols, groupSize: groupSize)
}

/// Projection weights (in scope) + one out-of-scope fp16 norm weight, with raw
/// F32 backing kept alive for the duration of `body`.
private func withSyntheticTensors(
    _ body: ([(runtimeName: String, data: UnsafeRawPointer, byteCount: Int, shape: [Int], dtype: String)]) throws -> Void
) rethrows {
    let projElems = rows * cols
    let q = UnsafeMutablePointer<Float>.allocate(capacity: projElems)
    let dn = UnsafeMutablePointer<Float>.allocate(capacity: projElems)
    let norm = UnsafeMutablePointer<Float>.allocate(capacity: cols)
    defer { q.deallocate(); dn.deallocate(); norm.deallocate() }
    q.update(repeating: 0.1, count: projElems)
    dn.update(repeating: 0.2, count: projElems)
    norm.update(repeating: 1.0, count: cols)
    let tensors: [(runtimeName: String, data: UnsafeRawPointer, byteCount: Int, shape: [Int], dtype: String)] = [
        (runtimeName: "layers_0_self_attn_q_proj_weight", data: UnsafeRawPointer(q),
         byteCount: projElems * 4, shape: [rows, cols], dtype: "F32"),
        (runtimeName: "layers_0_mlp_down_proj_weight", data: UnsafeRawPointer(dn),
         byteCount: projElems * 4, shape: [rows, cols], dtype: "F32"),
        (runtimeName: "final_norm_weight", data: UnsafeRawPointer(norm),
         byteCount: cols * 4, shape: [cols], dtype: "F32"),
    ]
    try body(tensors)
}

private let affineConfig = SmeltQuantizationConfig(
    strategy: .affineU4, groupSize: groupSize, excludePatterns: ["*_norm_weight"])

private func tempOutputPath() -> String {
    NSTemporaryDirectory() + "gptq-inject-\(ProcessInfo.processInfo.globallyUniqueString).bin"
}

@Test func gptqBlockInjectionWritesVerbatim() throws {
    let qBlock = makeBlock(seed: 1)
    let dnBlock = makeBlock(seed: 2)
    let out = tempOutputPath()
    defer { try? FileManager.default.removeItem(atPath: out) }

    try withSyntheticTensors { tensors in
        let entries = try SmeltAffineQuantizer.quantize(
            tensors: tensors, config: affineConfig, outputPath: out,
            gptqBlocks: [
                "layers_0_self_attn_q_proj_weight": qBlock,
                "layers_0_mlp_down_proj_weight": dnBlock,
            ])
        let bytes = try Data(contentsOf: URL(fileURLWithPath: out))

        for (name, block) in [
            ("layers_0_self_attn_q_proj_weight", qBlock),
            ("layers_0_mlp_down_proj_weight", dnBlock),
        ] {
            let e = try #require(entries.first { $0.name == name })
            #expect(e.dtype == .affineU4)
            // nibbles at offset, scales/biases (fp16 bits, little-endian) at their offsets
            #expect(Array(bytes[Int(e.offset)..<Int(e.offset) + block.nibbles.count]) == block.nibbles)
            #expect(readU16(bytes, at: Int(e.scalesOffset!), count: block.scales.count) == block.scales)
            #expect(readU16(bytes, at: Int(e.biasesOffset!), count: block.biases.count) == block.biases)
        }
    }
}

@Test func gptqInjectionRejectsMissingBlock() throws {
    let out = tempOutputPath()
    defer { try? FileManager.default.removeItem(atPath: out) }
    withSyntheticTensors { tensors in
        #expect(throws: SmeltAffineQuantizerError.self) {
            // down_proj block omitted → coverage gap.
            _ = try SmeltAffineQuantizer.quantize(
                tensors: tensors, config: affineConfig, outputPath: out,
                gptqBlocks: ["layers_0_self_attn_q_proj_weight": makeBlock(seed: 1)])
        }
    }
}

@Test func gptqInjectionRejectsOutOfScopeBlock() throws {
    let out = tempOutputPath()
    defer { try? FileManager.default.removeItem(atPath: out) }
    withSyntheticTensors { tensors in
        #expect(throws: SmeltAffineQuantizerError.self) {
            // A block for the fp16 norm weight, which is not in scope.
            _ = try SmeltAffineQuantizer.quantize(
                tensors: tensors, config: affineConfig, outputPath: out,
                gptqBlocks: [
                    "layers_0_self_attn_q_proj_weight": makeBlock(seed: 1),
                    "layers_0_mlp_down_proj_weight": makeBlock(seed: 2),
                    "final_norm_weight": makeBlock(seed: 3),
                ])
        }
    }
}

@Test func gptqInjectionRejectsShapeMismatch() throws {
    let out = tempOutputPath()
    defer { try? FileManager.default.removeItem(atPath: out) }
    let wrong = SmeltAffineU4.Packed(
        nibbles: [0, 0], scales: [0], biases: [0], rows: 1, cols: 2, groupSize: groupSize)
    withSyntheticTensors { tensors in
        #expect(throws: SmeltAffineQuantizerError.self) {
            _ = try SmeltAffineQuantizer.quantize(
                tensors: tensors, config: affineConfig, outputPath: out,
                gptqBlocks: [
                    "layers_0_self_attn_q_proj_weight": wrong,
                    "layers_0_mlp_down_proj_weight": makeBlock(seed: 2),
                ])
        }
    }
}

@Test func gptqBuildIgnoresStaleResume() throws {
    let out = tempOutputPath()
    let progress = out + ".progress"
    defer {
        try? FileManager.default.removeItem(atPath: out)
        try? FileManager.default.removeItem(atPath: progress)
    }
    let qName = "layers_0_self_attn_q_proj_weight"
    let dnName = "layers_0_mlp_down_proj_weight"
    try withSyntheticTensors { tensors in
        // First GPTQ build with one set of blocks.
        _ = try SmeltAffineQuantizer.quantize(
            tensors: tensors, config: affineConfig, outputPath: out,
            gptqBlocks: [qName: makeBlock(seed: 5), dnName: makeBlock(seed: 6)])
        // Simulate a stale resume marker (index past every tensor).
        try "99".write(toFile: progress, atomically: true, encoding: .utf8)
        // A second build with different blocks must ignore the stale .progress
        // and rewrite fresh — not skip the tensors and keep the first run's bytes.
        let fresh = makeBlock(seed: 9)
        let entries = try SmeltAffineQuantizer.quantize(
            tensors: tensors, config: affineConfig, outputPath: out,
            gptqBlocks: [qName: fresh, dnName: makeBlock(seed: 10)])
        let bytes = try Data(contentsOf: URL(fileURLWithPath: out))
        let e = try #require(entries.first { $0.name == qName })
        #expect(Array(bytes[Int(e.offset)..<Int(e.offset) + fresh.nibbles.count]) == fresh.nibbles)
    }
}

private func readU16(_ data: Data, at offset: Int, count: Int) -> [UInt16] {
    (0..<count).map { i in
        UInt16(data[offset + 2 * i]) | (UInt16(data[offset + 2 * i + 1]) << 8)
    }
}
