// SmeltFP32TrunkWriter — checkpoint weight materialization for dense
// embeddings-in/hidden-out trunks. The layout owns storage dtype; this writer
// copies or widens each dense entry according to that declaration without
// making activation family a capability gate.
//
// Dtype contract (matches the hand Qwen3TTSPackageBuilder, so the bytes are
// bit-identical to the source package):
//   • bf16 entry  ← BF16 source only, raw 2-byte copy (no narrowing — the
//     hand path requires bf16 source for bf16 specs; a non-bf16 projection
//     is a loud error, not a silent round-trip).
//   • fp32 entry  ← F32 source raw copy, or BF16/F16 source widened exactly
//     (Float(bitPattern: UInt32(bf16) << 16) — the codebase's canonical
//     widen; F16 via Float(Float16)).
// Anything else throws (the Phase-12 dispatch-safety lesson).

import Foundation
import SmeltSchema

enum SmeltFP32TrunkWriterError: Error, CustomStringConvertible {
    case missingTensor(String)
    case shapeMismatch(name: String, expected: [Int], got: [Int])
    case sourceByteMismatch(name: String, expected: Int, got: Int)
    case unsupportedEntryDtype(name: String, dtype: String)
    case unsupportedSourceDtype(name: String, entry: String, source: String)

    var description: String {
        switch self {
        case .missingTensor(let n):
            return "dense trunk writer: no source tensor for layout entry '\(n)'"
        case let .shapeMismatch(name, expected, got):
            return "dense trunk writer: '\(name)' shape \(got) != layout \(expected)"
        case let .sourceByteMismatch(name, expected, got):
            return "dense trunk writer: '\(name)' source has \(got) bytes, expected \(expected)"
        case let .unsupportedEntryDtype(name, dtype):
            return "dense trunk writer: layout entry '\(name)' has dtype \(dtype); "
                + "registered dense storage is bf16 or fp32"
        case let .unsupportedSourceDtype(name, entry, source):
            return "dense trunk writer: '\(name)' is a \(entry) entry but the "
                + "checkpoint provides \(source) (no silent conversion)"
        }
    }
}

enum SmeltFP32TrunkWriter {

    /// Writes `weights.bin` for a dense trunk from assembled checkpoint
    /// tensors, following the storage dtypes declared by `computeLayout`.
    /// Returns the manifest's weight entries.
    static func write(
        tensors: [(runtimeName: String, data: UnsafeRawPointer,
                   byteCount: Int, shape: [Int], dtype: String)],
        expectedLayout: [SmeltWeightEntry],
        outputPath: String
    ) throws -> [SmeltWeightEntry] {
        let sourceByName = Dictionary(
            tensors.map { ($0.runtimeName, $0) }, uniquingKeysWith: { a, _ in a })

        let totalSize = expectedLayout.reduce(UInt64(0)) {
            max($0, $1.offset + $1.sizeBytes)
        }
        var blob = Data(count: Int(totalSize))

        try blob.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            for entry in expectedLayout {
                guard let src = sourceByName[entry.name] else {
                    throw SmeltFP32TrunkWriterError.missingTensor(entry.name)
                }
                guard src.shape == entry.shape else {
                    throw SmeltFP32TrunkWriterError.shapeMismatch(
                        name: entry.name, expected: entry.shape, got: src.shape)
                }
                let elementCount = entry.shape.reduce(1, *)
                let dst = base.advanced(by: Int(entry.offset))

                switch entry.dtype {
                case .bf16:
                    // Raw 2-byte copy; bf16 source only (parity with the hand path).
                    guard src.dtype == "BF16" else {
                        throw SmeltFP32TrunkWriterError.unsupportedSourceDtype(
                            name: entry.name, entry: "bf16", source: src.dtype)
                    }
                    try requireSourceBytes(src, elementCount * 2)
                    memcpy(dst, src.data, elementCount * 2)

                case .fp32:
                    let out = dst.bindMemory(to: Float.self, capacity: elementCount)
                    switch src.dtype {
                    case "F32":
                        try requireSourceBytes(src, elementCount * 4)
                        memcpy(dst, src.data, elementCount * 4)
                    case "BF16":
                        // Checkpoint tensor offsets are NOT guaranteed 2-byte
                        // aligned (safetensors packs back-to-back); read
                        // unaligned, like the hand Qwen3TTSPackageBuilder.
                        try requireSourceBytes(src, elementCount * 2)
                        for i in 0..<elementCount {
                            let bits = src.data.loadUnaligned(
                                fromByteOffset: i * 2, as: UInt16.self)
                            out[i] = Float(bitPattern: UInt32(bits) << 16)
                        }
                    case "F16":
                        try requireSourceBytes(src, elementCount * 2)
                        for i in 0..<elementCount {
                            let bits = src.data.loadUnaligned(
                                fromByteOffset: i * 2, as: UInt16.self)
                            out[i] = Float(Float16(bitPattern: bits))
                        }
                    default:
                        throw SmeltFP32TrunkWriterError.unsupportedSourceDtype(
                            name: entry.name, entry: "fp32", source: src.dtype)
                    }

                default:
                    throw SmeltFP32TrunkWriterError.unsupportedEntryDtype(
                        name: entry.name, dtype: "\(entry.dtype)")
                }
            }
        }

        try blob.write(to: URL(fileURLWithPath: outputPath))
        return expectedLayout
    }

    private static func requireSourceBytes(
        _ src: (runtimeName: String, data: UnsafeRawPointer,
                byteCount: Int, shape: [Int], dtype: String),
        _ expected: Int
    ) throws {
        guard src.byteCount == expected else {
            throw SmeltFP32TrunkWriterError.sourceByteMismatch(
                name: src.runtimeName, expected: expected, got: src.byteCount)
        }
    }
}
