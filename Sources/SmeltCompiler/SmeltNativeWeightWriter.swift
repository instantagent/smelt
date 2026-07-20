// SmeltNativeWeightWriter — streams already-quantized semantic tensors and
// ordinary dense companions into a planned weights.bin layout.
//
// Container adapters own source spelling (for example MLX's
// weight/scales/biases triplet). This writer sees one logical signed tensor:
// packed codes plus canonical fp16 scales. It therefore works for any source
// which can satisfy SmeltPrequantizedTensorView, without a model-specific
// package path or a full-precision expansion.

// Dense entries share the same strict conversion contract used by the normal
// checkpoint path. Unsupported dtype combinations fail loudly.


import Darwin
import Foundation
import SmeltRuntime
import SmeltSchema

enum SmeltNativeWeightWriterError: Error, CustomStringConvertible {
    case duplicateSource(String)
    case sourceKindCollision(String)
    case missingSource(String)
    case unexpectedSource(String)
    case invalidSignedEntry(String, String)
    case signedFormatMismatch(name: String, expected: String, got: String)
    case shapeMismatch(name: String, expected: [Int], got: [Int])
    case sourceByteMismatch(name: String, expected: Int, got: Int)
    case unsupportedEntryDtype(name: String, dtype: String)
    case unsupportedSourceDtype(name: String, entry: String, source: String)
    case fileCreateFailed(String)
    case fileResizeFailed(String)
    case mmapFailed(String)
    case syncFailed(String)

    var description: String {
        switch self {
        case .duplicateSource(let name):
            return "native weight writer received duplicate source '\(name)'"
        case .sourceKindCollision(let name):
            return "native weight writer received both packed and dense sources for '\(name)'"
        case .missingSource(let name):
            return "native weight writer has no source for layout entry '\(name)'"
        case .unexpectedSource(let name):
            return "native weight writer source '\(name)' has no planned layout entry"
        case let .invalidSignedEntry(name, detail):
            return "native weight writer signed entry '\(name)' is invalid: \(detail)"
        case let .signedFormatMismatch(name, expected, got):
            return "native weight writer signed entry '\(name)' needs \(expected), got \(got)"
        case let .shapeMismatch(name, expected, got):
            return "native weight writer '\(name)' shape \(got) != layout \(expected)"
        case let .sourceByteMismatch(name, expected, got):
            return "native weight writer '\(name)' source has \(got) bytes, expected \(expected)"
        case let .unsupportedEntryDtype(name, dtype):
            return "native weight writer layout entry '\(name)' has unsupported dtype \(dtype)"
        case let .unsupportedSourceDtype(name, entry, source):
            return "native weight writer '\(name)' is a \(entry) entry but the checkpoint "
                + "provides \(source)"
        case .fileCreateFailed(let path):
            return "native weight writer could not create \(path)"
        case .fileResizeFailed(let path):
            return "native weight writer could not resize \(path)"
        case .mmapFailed(let path):
            return "native weight writer could not mmap \(path)"
        case .syncFailed(let path):
            return "native weight writer could not sync \(path)"
        }
    }
}

enum SmeltNativeWeightWriter {
    typealias DenseTensor = (
        runtimeName: String,
        data: UnsafeRawPointer,
        byteCount: Int,
        shape: [Int],
        dtype: String
    )

    struct PackedTensor {
        let runtimeName: String
        let view: SmeltPrequantizedTensorView
    }

    /// Write the exact planned layout without allocating a model-sized blob.
    /// The source objects which own `denseTensors` and `packedTensors` must
    /// remain alive for the duration of this call.
    static func write(
        packedTensors: [PackedTensor],
        denseTensors: [DenseTensor],
        expectedLayout: [SmeltWeightEntry],
        outputPath: String
    ) throws -> [SmeltWeightEntry] {
        let packedByName = try uniquePackedSources(packedTensors)
        let denseByName = try uniqueDenseSources(denseTensors)
        for name in packedByName.keys where denseByName[name] != nil {
            throw SmeltNativeWeightWriterError.sourceKindCollision(name)
        }

        let expectedNames = Set(expectedLayout.filter { $0.sizeBytes > 0 }.map(\.name))
        for name in packedByName.keys where !expectedNames.contains(name) {
            throw SmeltNativeWeightWriterError.unexpectedSource(name)
        }
        for name in denseByName.keys where !expectedNames.contains(name) {
            throw SmeltNativeWeightWriterError.unexpectedSource(name)
        }

        // Validate the complete package before creating a multi-gigabyte file.
        for entry in expectedLayout where entry.sizeBytes > 0 {
            switch entry.dtype {
            case .binary1, .ternary2:
                guard let packed = packedByName[entry.name] else {
                    throw SmeltNativeWeightWriterError.missingSource(entry.name)
                }
                try validate(packed: packed, against: entry)
            case .fp16, .bf16, .fp32, .int32:
                guard let dense = denseByName[entry.name] else {
                    throw SmeltNativeWeightWriterError.missingSource(entry.name)
                }
                try validate(dense: dense, against: entry)
            case .raw, .u4Lut, .affineU4, .turboQuantH:
                throw SmeltNativeWeightWriterError.unsupportedEntryDtype(
                    name: entry.name, dtype: entry.dtype.rawValue)
            }
        }

        let totalBytes = SmeltWeightManifestLoader.totalBytes(from: expectedLayout)
        guard totalBytes > 0, totalBytes <= UInt64(Int.max) else {
            throw SmeltNativeWeightWriterError.fileResizeFailed(outputPath)
        }

        let fd = open(outputPath, O_RDWR | O_CREAT | O_TRUNC, 0o644)
        guard fd >= 0 else {
            throw SmeltNativeWeightWriterError.fileCreateFailed(outputPath)
        }
        var completed = false
        defer {
            close(fd)
            if !completed {
                try? FileManager.default.removeItem(atPath: outputPath)
            }
        }
        guard ftruncate(fd, off_t(totalBytes)) == 0 else {
            throw SmeltNativeWeightWriterError.fileResizeFailed(outputPath)
        }
        let mapped = mmap(
            nil, Int(totalBytes), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        guard let output = mapped, output != MAP_FAILED else {
            throw SmeltNativeWeightWriterError.mmapFailed(outputPath)
        }
        defer { munmap(output, Int(totalBytes)) }

        for entry in expectedLayout where entry.sizeBytes > 0 {
            switch entry.dtype {
            case .binary1, .ternary2:
                try write(
                    packed: packedByName[entry.name]!, entry: entry, output: output)
            case .fp16, .bf16, .fp32, .int32:
                try write(
                    dense: denseByName[entry.name]!, entry: entry, output: output)
            case .raw, .u4Lut, .affineU4, .turboQuantH:
                // Rejected in the preflight above.
                preconditionFailure("unsupported entry escaped native writer preflight")
            }
        }

        guard msync(output, Int(totalBytes), MS_SYNC) == 0, fsync(fd) == 0 else {
            throw SmeltNativeWeightWriterError.syncFailed(outputPath)
        }
        completed = true
        return expectedLayout
    }

    private static func uniquePackedSources(
        _ tensors: [PackedTensor]
    ) throws -> [String: SmeltPrequantizedTensorView] {
        var result: [String: SmeltPrequantizedTensorView] = [:]
        for tensor in tensors {
            guard result.updateValue(tensor.view, forKey: tensor.runtimeName) == nil else {
                throw SmeltNativeWeightWriterError.duplicateSource(tensor.runtimeName)
            }
        }
        return result
    }

    private static func uniqueDenseSources(
        _ tensors: [DenseTensor]
    ) throws -> [String: DenseTensor] {
        var result: [String: DenseTensor] = [:]
        for tensor in tensors {
            guard result.updateValue(tensor, forKey: tensor.runtimeName) == nil else {
                throw SmeltNativeWeightWriterError.duplicateSource(tensor.runtimeName)
            }
        }
        return result
    }

    private static func validate(
        packed: SmeltPrequantizedTensorView,
        against entry: SmeltWeightEntry
    ) throws {
        let expectedFormat: SmeltSignedQuantFormat = entry.dtype == .binary1
            ? .binary1 : .ternary2
        let descriptor = packed.descriptor
        guard descriptor.format == expectedFormat else {
            throw SmeltNativeWeightWriterError.signedFormatMismatch(
                name: entry.name,
                expected: expectedFormat.rawValue,
                got: descriptor.format.rawValue)
        }
        guard entry.shape.count == 2,
              let groupSize = entry.groupSize,
              let rowStride = entry.packedRowStride,
              let paddedCols = entry.paddedCols,
              let scalesOffset = entry.scalesOffset,
              let scalesSize = entry.scalesSizeBytes
        else {
            throw SmeltNativeWeightWriterError.invalidSignedEntry(
                entry.name, "missing rank-2/group/stride/scale metadata")
        }
        guard descriptor.logicalShape == [entry.shape[0], paddedCols] else {
            throw SmeltNativeWeightWriterError.shapeMismatch(
                name: entry.name,
                expected: [entry.shape[0], paddedCols],
                got: descriptor.logicalShape)
        }
        guard descriptor.groupSize == groupSize,
              descriptor.packedRowStride == rowStride,
              descriptor.paddedCols == paddedCols,
              descriptor.codeByteCount == Int(entry.sizeBytes),
              descriptor.scaleCount * 2 == Int(scalesSize)
        else {
            throw SmeltNativeWeightWriterError.invalidSignedEntry(
                entry.name, "source geometry does not match planned packed regions")
        }
        guard scalesOffset >= entry.offset + entry.sizeBytes else {
            throw SmeltNativeWeightWriterError.invalidSignedEntry(
                entry.name, "scale region overlaps packed codes")
        }
    }

    private static func validate(
        dense: DenseTensor,
        against entry: SmeltWeightEntry
    ) throws {
        guard dense.shape == entry.shape else {
            throw SmeltNativeWeightWriterError.shapeMismatch(
                name: entry.name, expected: entry.shape, got: dense.shape)
        }
        let elements = try elementCount(entry.shape, name: entry.name)
        let sourceStride: Int
        switch dense.dtype {
        case "F16", "BF16": sourceStride = 2
        case "F32", "I32": sourceStride = 4
        default:
            throw SmeltNativeWeightWriterError.unsupportedSourceDtype(
                name: entry.name, entry: entry.dtype.rawValue, source: dense.dtype)
        }
        guard dense.byteCount == elements * sourceStride else {
            throw SmeltNativeWeightWriterError.sourceByteMismatch(
                name: entry.name, expected: elements * sourceStride, got: dense.byteCount)
        }
        switch entry.dtype {
        case .fp16:
            guard dense.dtype == "F16" || dense.dtype == "BF16" || dense.dtype == "F32" else {
                throw SmeltNativeWeightWriterError.unsupportedSourceDtype(
                    name: entry.name, entry: "fp16", source: dense.dtype)
            }
        case .bf16:
            guard dense.dtype == "BF16" else {
                throw SmeltNativeWeightWriterError.unsupportedSourceDtype(
                    name: entry.name, entry: "bf16", source: dense.dtype)
            }
        case .fp32:
            guard dense.dtype == "F16" || dense.dtype == "BF16" || dense.dtype == "F32" else {
                throw SmeltNativeWeightWriterError.unsupportedSourceDtype(
                    name: entry.name, entry: "fp32", source: dense.dtype)
            }
        case .int32:
            guard dense.dtype == "I32" else {
                throw SmeltNativeWeightWriterError.unsupportedSourceDtype(
                    name: entry.name, entry: "int32", source: dense.dtype)
            }
        case .raw, .u4Lut, .affineU4, .binary1, .ternary2, .turboQuantH:
            throw SmeltNativeWeightWriterError.unsupportedEntryDtype(
                name: entry.name, dtype: entry.dtype.rawValue)
        }
        guard entry.sizeBytes == UInt64(elements * (entry.dtype.bytesPerElement ?? 0)) else {
            throw SmeltNativeWeightWriterError.invalidSignedEntry(
                entry.name, "dense byte count does not match shape/dtype")
        }
    }

    private static func write(
        packed: SmeltPrequantizedTensorView,
        entry: SmeltWeightEntry,
        output: UnsafeMutableRawPointer
    ) throws {
        let codeDestination = output.advanced(by: Int(entry.offset))
        _ = packed.withCodes { codes in
            switch entry.dtype {
            case .binary1, .ternary2:
                memcpy(codeDestination, codes, Int(entry.sizeBytes))
            default:
                preconditionFailure("non-signed entry escaped native packed writer")
            }
        }
        let scalesOffset = Int(entry.scalesOffset!)
        for group in 0..<packed.descriptor.scaleCount {
            var bits = try packed.canonicalScaleBits(at: group)
            memcpy(output.advanced(by: scalesOffset + group * 2), &bits, 2)
        }
    }

    private static func write(
        dense: DenseTensor,
        entry: SmeltWeightEntry,
        output: UnsafeMutableRawPointer
    ) throws {
        let elements = try elementCount(entry.shape, name: entry.name)
        let destination = output.advanced(by: Int(entry.offset))
        switch entry.dtype {
        case .fp16:
            let out = destination.bindMemory(to: UInt16.self, capacity: elements)
            switch dense.dtype {
            case "F16":
                memcpy(destination, dense.data, elements * 2)
            case "BF16":
                for index in 0..<elements {
                    let bits = dense.data.loadUnaligned(
                        fromByteOffset: index * 2, as: UInt16.self)
                    out[index] = Float16(Float(bitPattern: UInt32(bits) << 16)).bitPattern
                }
            case "F32":
                for index in 0..<elements {
                    let value = dense.data.loadUnaligned(
                        fromByteOffset: index * 4, as: Float.self)
                    out[index] = Float16(value).bitPattern
                }
            default:
                preconditionFailure("unsupported fp16 source escaped preflight")
            }
        case .bf16, .int32:
            memcpy(destination, dense.data, Int(entry.sizeBytes))
        case .fp32:
            let out = destination.bindMemory(to: Float.self, capacity: elements)
            switch dense.dtype {
            case "F32":
                memcpy(destination, dense.data, elements * 4)
            case "BF16":
                for index in 0..<elements {
                    let bits = dense.data.loadUnaligned(
                        fromByteOffset: index * 2, as: UInt16.self)
                    out[index] = Float(bitPattern: UInt32(bits) << 16)
                }
            case "F16":
                for index in 0..<elements {
                    let bits = dense.data.loadUnaligned(
                        fromByteOffset: index * 2, as: UInt16.self)
                    out[index] = Float(Float16(bitPattern: bits))
                }
            default:
                preconditionFailure("unsupported fp32 source escaped preflight")
            }
        case .raw, .u4Lut, .affineU4, .binary1, .ternary2, .turboQuantH:
            preconditionFailure("unsupported dense entry escaped preflight")
        }
    }

    private static func elementCount(_ shape: [Int], name: String) throws -> Int {
        guard !shape.isEmpty else {
            throw SmeltNativeWeightWriterError.shapeMismatch(
                name: name, expected: [1], got: shape)
        }
        var count = 1
        for dimension in shape {
            guard dimension > 0 else {
                throw SmeltNativeWeightWriterError.shapeMismatch(
                    name: name, expected: shape.map { max(1, $0) }, got: shape)
            }
            let (next, overflow) = count.multipliedReportingOverflow(by: dimension)
            guard !overflow else {
                throw SmeltNativeWeightWriterError.sourceByteMismatch(
                    name: name, expected: Int.max, got: 0)
            }
            count = next
        }
        return count
    }
}
