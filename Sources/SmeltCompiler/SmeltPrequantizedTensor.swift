// SmeltPrequantizedTensor — source-neutral packed tensor contract plus the MLX
// safetensors triplet adapter.
//
// The adapter is the only layer that knows MLX spells one logical tensor as
// weight/scales/biases. Writers and runtimes consume the semantic descriptor:
// packed signed codes, canonical fp16 scales, shape, and group geometry.

import Foundation
import SmeltRuntime

private final class SmeltPrequantizedStorageLease {
    let loader: SafetensorsLoader

    init(loader: SafetensorsLoader) {
        self.loader = loader
    }
}

public struct SmeltPrequantizedTensorDescriptor: Equatable, Sendable {
    public let sourceName: String
    public let format: SmeltSignedQuantFormat
    public let logicalShape: [Int]
    public let groupSize: Int
    public let packedRowStride: Int
    public let paddedCols: Int
    public let codeByteCount: Int
    public let scaleCount: Int

    public init(
        sourceName: String,
        format: SmeltSignedQuantFormat,
        logicalShape: [Int],
        groupSize: Int,
        packedRowStride: Int,
        paddedCols: Int,
        codeByteCount: Int,
        scaleCount: Int
    ) {
        self.sourceName = sourceName
        self.format = format
        self.logicalShape = logicalShape
        self.groupSize = groupSize
        self.packedRowStride = packedRowStride
        self.paddedCols = paddedCols
        self.codeByteCount = codeByteCount
        self.scaleCount = scaleCount
    }
}

/// Zero-copy view over one prequantized tensor. The view retains the source
/// mapping and lends code bytes through `withCodes`.
public struct SmeltPrequantizedTensorView {
    public let descriptor: SmeltPrequantizedTensorDescriptor
    private let codePointer: UnsafeRawPointer
    private let mlxScales: UnsafeRawPointer
    private let mlxBiases: UnsafeRawPointer
    private let storageLease: SmeltPrequantizedStorageLease

    fileprivate init(
        descriptor: SmeltPrequantizedTensorDescriptor,
        codes: UnsafeRawPointer,
        mlxScales: UnsafeRawPointer,
        mlxBiases: UnsafeRawPointer,
        storageLease: SmeltPrequantizedStorageLease
    ) {
        self.descriptor = descriptor
        self.codePointer = codes
        self.mlxScales = mlxScales
        self.mlxBiases = mlxBiases
        self.storageLease = storageLease
    }

    /// Borrow the packed source bytes while retaining their mmap. Returning a
    /// naked pointer from the view allowed optimized Swift to release the
    /// loader before a streaming writer consumed it.
    public func withCodes<Result>(
        _ body: (UnsafeRawPointer) throws -> Result
    ) rethrows -> Result {
        try withExtendedLifetime(storageLease) {
            try body(codePointer)
        }
    }

    public func canonicalScaleBits(at group: Int) throws -> UInt16 {
        guard group >= 0, group < descriptor.scaleCount else {
            throw SmeltPrequantizedTensorError.invalidGeometry(
                descriptor.sourceName,
                "scale group \(group) outside 0..<\(descriptor.scaleCount)"
            )
        }
        return try withExtendedLifetime(storageLease) {
            let sourceScale = mlxScales.loadUnaligned(
                fromByteOffset: group * 2, as: UInt16.self)
            let bias = mlxBiases.loadUnaligned(
                fromByteOffset: group * 2, as: UInt16.self)
            return try SmeltSignedQuantCodec.canonicalMLXScaleBits(
                format: descriptor.format,
                sourceScaleBits: sourceScale,
                biasBits: bias,
                group: group
            )
        }
    }
}

public enum SmeltPrequantizedTensorError: Error, CustomStringConvertible {
    case duplicateTensor(String)
    case orphanCompanion(String)
    case missingCompanion(weight: String, companion: String)
    case invalidDType(String, expected: String, actual: String)
    case invalidGeometry(String, String)
    case invalidPayload(String, String)

    public var description: String {
        switch self {
        case .duplicateTensor(let name):
            return "prequantized checkpoint contains duplicate tensor '\(name)'"
        case .orphanCompanion(let name):
            return "prequantized checkpoint companion '\(name)' has no U32 .weight sibling"
        case let .missingCompanion(weight, companion):
            return "prequantized tensor '\(weight)' is missing companion '\(companion)'"
        case let .invalidDType(name, expected, actual):
            return "prequantized tensor '\(name)' needs dtype \(expected), got \(actual)"
        case let .invalidGeometry(name, detail):
            return "prequantized tensor '\(name)' has invalid geometry: \(detail)"
        case let .invalidPayload(name, detail):
            return "prequantized tensor '\(name)' has invalid payload: \(detail)"
        }
    }
}

/// MLX safetensors adapter. It owns the loader so all zero-copy tensor views
/// remain valid, and also conforms to CheckpointTensorSource for the dense
/// tensors which are not members of a packed triplet.
public struct SmeltPrequantizedSafetensors: CheckpointTensorSource {
    public let prequantizedTensors: [SmeltPrequantizedTensorView]
    public let checkpointTensors: [CheckpointTensorDescriptor]
    public let consumedTensorNames: Set<String>

    private let storageLease: SmeltPrequantizedStorageLease

    /// MLX conversion normalizes architecture-specific `(1 + delta)` norm
    /// tensors to the direct multiplicative weights consumed by MLX RMSNorm.
    /// Keep that container fact here so model/runtime code remains source-blind.
    public var normWeightSemantics: CheckpointNormWeightSemantics { .directWeight }

    public init(
        loader: SafetensorsLoader,
        format: SmeltSignedQuantFormat,
        groupSize: Int
    ) throws {
        guard groupSize > 0 else {
            throw SmeltPrequantizedTensorError.invalidGeometry(
                "checkpoint", "group size must be positive")
        }

        let storageLease = SmeltPrequantizedStorageLease(loader: loader)
        var byName: [String: SafetensorInfo] = [:]
        for info in loader.tensors {
            guard byName.updateValue(info, forKey: info.name) == nil else {
                throw SmeltPrequantizedTensorError.duplicateTensor(info.name)
            }
        }

        let packedWeights = loader.tensors.filter {
            $0.dtype == "U32" && $0.name.hasSuffix(".weight")
        }
        let packedWeightNames = Set(packedWeights.map(\.name))

        // A config-declared prequantized checkpoint must not silently ignore a
        // dangling MLX companion. Limit the check to the MLX suffix contract.
        for info in loader.tensors where info.name.hasSuffix(".scales")
            || info.name.hasSuffix(".biases")
        {
            let suffix = info.name.hasSuffix(".scales") ? ".scales" : ".biases"
            let base = String(info.name.dropLast(suffix.count))
            guard packedWeightNames.contains(base + ".weight") else {
                throw SmeltPrequantizedTensorError.orphanCompanion(info.name)
            }
        }

        var views: [SmeltPrequantizedTensorView] = []
        var consumed: Set<String> = []
        for weight in packedWeights.sorted(by: { $0.name < $1.name }) {
            let base = String(weight.name.dropLast(".weight".count))
            let scalesName = base + ".scales"
            let biasesName = base + ".biases"
            guard let scales = byName[scalesName] else {
                throw SmeltPrequantizedTensorError.missingCompanion(
                    weight: weight.name, companion: scalesName)
            }
            guard let biases = byName[biasesName] else {
                throw SmeltPrequantizedTensorError.missingCompanion(
                    weight: weight.name, companion: biasesName)
            }
            guard scales.dtype == "F16" else {
                throw SmeltPrequantizedTensorError.invalidDType(
                    scales.name, expected: "F16", actual: scales.dtype)
            }
            guard biases.dtype == "F16" else {
                throw SmeltPrequantizedTensorError.invalidDType(
                    biases.name, expected: "F16", actual: biases.dtype)
            }
            guard weight.shape.count == 2,
                  scales.shape.count == 2,
                  biases.shape == scales.shape,
                  weight.shape[0] == scales.shape[0]
            else {
                throw SmeltPrequantizedTensorError.invalidGeometry(
                    weight.name,
                    "expected weight [rows,packed_u32] and matching scales/biases [rows,groups]"
                )
            }

            let rows = weight.shape[0]
            let groupsPerRow = scales.shape[1]
            let (logicalCols, colsOverflow) = groupsPerRow.multipliedReportingOverflow(
                by: groupSize)
            guard rows > 0, groupsPerRow > 0, !colsOverflow else {
                throw SmeltPrequantizedTensorError.invalidGeometry(
                    weight.name, "row/column count is empty or overflows")
            }
            let packedRowBytes = try SmeltSignedQuantCodec.packedByteCount(
                logicalCount: logicalCols, format: format)
            // U32 storage may only add whole-word tail bytes. Bonsai's group
            // geometry is exact; rejecting bit padding keeps row semantics
            // identical across MLX and GGUF.
            guard packedRowBytes % 4 == 0,
                  weight.shape[1] == packedRowBytes / 4
            else {
                throw SmeltPrequantizedTensorError.invalidGeometry(
                    weight.name,
                    "U32 row width \(weight.shape[1]) does not encode "
                        + "\(logicalCols) \(format.rawValue) values exactly"
                )
            }
            let (codeBytes, codeOverflow) = rows.multipliedReportingOverflow(
                by: packedRowBytes)
            let (scaleCount, scaleOverflow) = rows.multipliedReportingOverflow(
                by: groupsPerRow)
            guard !codeOverflow, !scaleOverflow,
                  weight.byteCount == codeBytes,
                  scales.byteCount == scaleCount * 2,
                  biases.byteCount == scaleCount * 2
            else {
                throw SmeltPrequantizedTensorError.invalidGeometry(
                    weight.name, "payload byte counts do not match resolved geometry")
            }

            let codePointer = loader.tensorData(weight)
            do {
                try SmeltSignedQuantCodec.validateCodes(
                    codePointer,
                    byteCount: codeBytes,
                    logicalCount: rows * logicalCols,
                    format: format
                )
            } catch {
                throw SmeltPrequantizedTensorError.invalidPayload(
                    weight.name, String(describing: error))
            }

            let descriptor = SmeltPrequantizedTensorDescriptor(
                sourceName: weight.name,
                format: format,
                logicalShape: [rows, logicalCols],
                groupSize: groupSize,
                packedRowStride: packedRowBytes,
                paddedCols: logicalCols,
                codeByteCount: codeBytes,
                scaleCount: scaleCount
            )
            let view = SmeltPrequantizedTensorView(
                descriptor: descriptor,
                codes: codePointer,
                mlxScales: loader.tensorData(scales),
                mlxBiases: loader.tensorData(biases),
                storageLease: storageLease
            )
            // Prove every affine group before the bias plane can disappear.
            do {
                for group in 0..<scaleCount {
                    _ = try view.canonicalScaleBits(at: group)
                }
            } catch {
                throw SmeltPrequantizedTensorError.invalidPayload(
                    weight.name, String(describing: error))
            }
            views.append(view)
            consumed.formUnion([weight.name, scalesName, biasesName])
        }

        let dense = loader.checkpointTensors.filter {
            !consumed.contains($0.name)
        }
        self.storageLease = storageLease
        self.prequantizedTensors = views
        self.checkpointTensors = dense
        self.consumedTensorNames = consumed
    }

    public func checkpointTensorData(
        _ descriptor: CheckpointTensorDescriptor
    ) -> UnsafeRawPointer {
        storageLease.loader.checkpointTensorData(descriptor)
    }
}
