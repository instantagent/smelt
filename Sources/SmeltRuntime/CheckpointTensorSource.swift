// CheckpointTensorSource — common interface over checkpoint containers.
//
// The builder assembles its quantizer input (name → data/shape/dtype) from a
// source-format container. Safetensors is the HF path. Sources vend mmap'd or
// owned pointers; the caller keeps the source alive while pointers are in use.

import Foundation

/// How dense norm weights in a checkpoint container are encoded. Model-native
/// sources inherit the authored model contract; converted containers may have
/// normalized the same tensor to a direct multiplicative weight.
public enum CheckpointNormWeightSemantics: Sendable, Equatable {
    case modelDeclared
    case directWeight
    case onePlusDelta
}

/// One tensor a checkpoint container can vend.
public struct CheckpointTensorDescriptor {
    /// Position in the source's tensor list (used by `checkpointTensorData`).
    public let index: Int
    /// Source-domain name (HF naming; adapters map to smelt names).
    public let name: String
    /// Source dtype string: "F16", "BF16", "F32", "I32", ...
    public let dtype: String
    public let shape: [Int]
    public let byteCount: Int

    public init(index: Int, name: String, dtype: String, shape: [Int], byteCount: Int) {
        self.index = index
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.byteCount = byteCount
    }
}

/// A checkpoint container the builder can assemble quantizer input from.
public protocol CheckpointTensorSource {
    var checkpointTensors: [CheckpointTensorDescriptor] { get }
    var normWeightSemantics: CheckpointNormWeightSemantics { get }
    /// Raw data for a descriptor previously vended by `checkpointTensors`.
    /// Valid only while the source is alive.
    func checkpointTensorData(_ descriptor: CheckpointTensorDescriptor) -> UnsafeRawPointer
}

public extension CheckpointTensorSource {
    var normWeightSemantics: CheckpointNormWeightSemantics { .modelDeclared }
}

extension SafetensorsLoader: CheckpointTensorSource {
    public var checkpointTensors: [CheckpointTensorDescriptor] {
        tensors.enumerated().map { index, info in
            CheckpointTensorDescriptor(
                index: index,
                name: info.name,
                dtype: info.dtype,
                shape: info.shape,
                byteCount: info.byteCount
            )
        }
    }

    public func checkpointTensorData(
        _ descriptor: CheckpointTensorDescriptor
    ) -> UnsafeRawPointer {
        tensorData(tensors[descriptor.index])
    }
}
