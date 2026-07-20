import Foundation
import Metal

/// The two point-feature projections that start the rig model's neural graph.
public enum SmeltPointProjection: Sendable, Equatable {
    case mesh
    case condition

    fileprivate var weightName: String {
        switch self {
        case .mesh: return "mesh_encoder.encoder.input_proj.weight"
        case .condition: return "vae.model.cond_encoder.proj_in.weight"
        }
    }

    fileprivate var biasName: String {
        switch self {
        case .mesh: return "mesh_encoder.encoder.input_proj.bias"
        case .condition: return "vae.model.cond_encoder.proj_in.bias"
        }
    }

    fileprivate var outputDimension: Int {
        switch self {
        case .mesh: return 512
        case .condition: return 768
        }
    }

    fileprivate var includePi: UInt32 {
        switch self {
        case .mesh: return 0
        case .condition: return 1
        }
    }

    fileprivate var usePMPE: UInt32 {
        switch self {
        case .mesh: return 0
        case .condition: return 1
        }
    }
}

/// Staged production implementation of Fourier/PMPE + normal append + BF16
/// dense projection. This survives as the exactness oracle when these stages
/// are later fused into model record plans.
public final class SmeltPointProjectionRuntime {
    /// Keeps one dense projection dispatch below the practical Apple GPU
    /// command-buffer limit while preserving independent row semantics.
    static let maximumRowsPerCommandBuffer = 1_024

    private let artifact: SmeltRigArtifact
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let fourier: MTLComputePipelineState
    private let appendFeatures: MTLComputePipelineState
    private let dense: MTLComputePipelineState

    public init(artifact: SmeltRigArtifact, device: MTLDevice? = nil) throws {
        let device = try device ?? Self.requireDevice()
        guard let queue = device.makeCommandQueue() else {
            throw SmeltPointProjectionRuntimeError.commandQueueCreationFailed
        }
        let library = try artifact.makeLibrary(device: device)
        self.artifact = artifact
        self.device = device
        self.queue = queue
        fourier = try Self.pipeline(
            device: device,
            library: library,
            name: "fourier_position_embedding_f32"
        )
        appendFeatures = try Self.pipeline(
            device: device,
            library: library,
            name: "append_strided_features_f32"
        )
        dense = try Self.pipeline(device: device, library: library, name: "dense_bf16w_f32")
    }

    /// Projects interleaved `[x,y,z,nx,ny,nz]` rows through the authored
    /// rig model input layer. The returned row-major activations stay FP32.
    public func project(
        pointNormals: [Float],
        projection: SmeltPointProjection
    ) throws -> [Float] {
        guard !pointNormals.isEmpty, pointNormals.count.isMultiple(of: 6) else {
            throw SmeltPointProjectionRuntimeError.invalidPointNormals(pointNormals.count)
        }
        let rows = pointNormals.count / 6
        let fourierDim = 51
        let inputDim = 54
        let outputDim = projection.outputDimension
        let input = try makeBuffer(pointNormals, label: "rig.point-normals")
        let fourierOutput = try makeBuffer(
            count: rows * fourierDim,
            label: "rig.fourier"
        )
        let projectedInput = try makeBuffer(
            count: rows * inputDim,
            label: "rig.point-features"
        )
        let output = try makeBuffer(
            count: rows * outputDim,
            label: "rig.point-projection"
        )
        let weight = try artifact.makeWeightBuffer(
            device: device,
            tensorNamed: projection.weightName
        )
        let bias = try artifact.makeWeightBuffer(
            device: device,
            tensorNamed: projection.biasName
        )
        var rowStart = 0
        while rowStart < rows {
            let rowCount = min(
                Self.maximumRowsPerCommandBuffer,
                rows - rowStart
            )
            let commandBuffer = try require(
                queue.makeCommandBuffer(),
                .commandBufferCreationFailed
            )
            commandBuffer.label = "rig.point-projection.rows.\(rowStart)..<\(rowStart + rowCount)"
            let encoder = try require(
                commandBuffer.makeComputeCommandEncoder(),
                .commandEncoderCreationFailed
            )

            encoder.setComputePipelineState(fourier)
            encoder.setBuffer(
                input,
                offset: rowStart * 6 * MemoryLayout<Float>.stride,
                index: 0
            )
            encoder.setBuffer(
                fourierOutput,
                offset: rowStart * fourierDim * MemoryLayout<Float>.stride,
                index: 1
            )
            var rowsValue = UInt32(rowCount)
            var positionDimensions: UInt32 = 3
            var frequencies: UInt32 = 8
            var includeInput: UInt32 = 1
            var includePi = projection.includePi
            var usePMPE = projection.usePMPE
            var inputStride: UInt32 = 6
            encoder.setBytes(&rowsValue, length: 4, index: 2)
            encoder.setBytes(&positionDimensions, length: 4, index: 3)
            encoder.setBytes(&frequencies, length: 4, index: 4)
            encoder.setBytes(&includeInput, length: 4, index: 5)
            encoder.setBytes(&includePi, length: 4, index: 6)
            encoder.setBytes(&usePMPE, length: 4, index: 7)
            encoder.setBytes(&inputStride, length: 4, index: 8)
            encoder.dispatchThreads(
                MTLSize(width: fourierDim, height: rowCount, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
            )

            encoder.setComputePipelineState(appendFeatures)
            encoder.setBuffer(
                fourierOutput,
                offset: rowStart * fourierDim * MemoryLayout<Float>.stride,
                index: 0
            )
            encoder.setBuffer(
                input,
                offset: rowStart * 6 * MemoryLayout<Float>.stride,
                index: 1
            )
            encoder.setBuffer(
                projectedInput,
                offset: rowStart * inputDim * MemoryLayout<Float>.stride,
                index: 2
            )
            var baseDim = UInt32(fourierDim)
            var featureStride: UInt32 = 6
            var featureOffset: UInt32 = 3
            var featureCount: UInt32 = 3
            encoder.setBytes(&rowsValue, length: 4, index: 3)
            encoder.setBytes(&baseDim, length: 4, index: 4)
            encoder.setBytes(&featureStride, length: 4, index: 5)
            encoder.setBytes(&featureOffset, length: 4, index: 6)
            encoder.setBytes(&featureCount, length: 4, index: 7)
            encoder.dispatchThreads(
                MTLSize(width: inputDim, height: rowCount, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
            )

            encoder.setComputePipelineState(dense)
            encoder.setBuffer(
                projectedInput,
                offset: rowStart * inputDim * MemoryLayout<Float>.stride,
                index: 0
            )
            encoder.setBuffer(weight, offset: 0, index: 1)
            encoder.setBuffer(bias, offset: 0, index: 2)
            encoder.setBuffer(
                output,
                offset: rowStart * outputDim * MemoryLayout<Float>.stride,
                index: 3
            )
            var outputDimValue = UInt32(outputDim)
            var inputDimValue = UInt32(inputDim)
            var hasBias: UInt32 = 1
            encoder.setBytes(&rowsValue, length: 4, index: 4)
            encoder.setBytes(&outputDimValue, length: 4, index: 5)
            encoder.setBytes(&inputDimValue, length: 4, index: 6)
            encoder.setBytes(&hasBias, length: 4, index: 7)
            encoder.dispatchThreadgroups(
                MTLSize(width: outputDim, height: rowCount, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error {
                throw SmeltPointProjectionRuntimeError.gpuExecutionFailed(
                    "rows \(rowStart)..<\(rowStart + rowCount): \(error)"
                )
            }
            rowStart += rowCount
        }
        return Array(
            UnsafeBufferPointer(
                start: output.contents().bindMemory(
                    to: Float.self,
                    capacity: rows * outputDim
                ),
                count: rows * outputDim
            )
        )
    }

    private func makeBuffer(_ values: [Float], label: String) throws -> MTLBuffer {
        let buffer = try values.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress,
                  let buffer = device.makeBuffer(
                      bytes: base,
                      length: bytes.count,
                      options: .storageModeShared
                  )
            else {
                throw SmeltPointProjectionRuntimeError.bufferCreationFailed(label)
            }
            return buffer
        }
        buffer.label = label
        return buffer
    }

    private func makeBuffer(count: Int, label: String) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(
            length: count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw SmeltPointProjectionRuntimeError.bufferCreationFailed(label)
        }
        buffer.label = label
        return buffer
    }

    private static func requireDevice() throws -> MTLDevice {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw SmeltPointProjectionRuntimeError.metalUnavailable
        }
        return device
    }

    private static func pipeline(
        device: MTLDevice,
        library: MTLLibrary,
        name: String
    ) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw SmeltPointProjectionRuntimeError.pipelineMissing(name)
        }
        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            throw SmeltPointProjectionRuntimeError.pipelineCreationFailed(name, "\(error)")
        }
    }

    private func require<T>(
        _ value: T?,
        _ error: SmeltPointProjectionRuntimeError
    ) throws -> T {
        guard let value else { throw error }
        return value
    }
}

public enum SmeltPointProjectionRuntimeError: Error, Equatable {
    case metalUnavailable
    case commandQueueCreationFailed
    case commandBufferCreationFailed
    case commandEncoderCreationFailed
    case invalidPointNormals(Int)
    case bufferCreationFailed(String)
    case pipelineMissing(String)
    case pipelineCreationFailed(String, String)
    case gpuExecutionFailed(String)
}
