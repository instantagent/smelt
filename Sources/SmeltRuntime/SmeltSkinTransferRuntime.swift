import Metal

/// GPU implementation of the reusable neighbor-blend plus top-four skinning
/// brick. The kernel retains vertex-local source order and returns GLB-ready
/// four-lane rows.
public final class SmeltSkinTransferRuntime {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState

    public init(
        artifact: SmeltComponentArtifact,
        device: MTLDevice? = nil
    ) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else {
            throw SmeltSkinTransferRuntimeError.metalUnavailable
        }
        guard let queue = device.makeCommandQueue() else {
            throw SmeltSkinTransferRuntimeError.commandQueueCreationFailed
        }
        let library = try artifact.makeLibrary(device: device)
        guard let function = library.makeFunction(name: "skin_transfer_top4_f32") else {
            throw SmeltSkinTransferRuntimeError.pipelineMissing
        }
        self.device = device
        self.queue = queue
        pipeline = try device.makeComputePipelineState(function: function)
    }

    public func transfer(
        plan: SmeltSkinTransferPlan,
        sampledWeights: SmeltSkinWeights
    ) throws -> SmeltVertexSkin {
        guard sampledWeights.vertexCount == plan.querySourceIndices.count,
              sampledWeights.jointCount > 0,
              sampledWeights.jointCount <= Int(UInt16.max),
              sampledWeights.values.count
                == sampledWeights.vertexCount * sampledWeights.jointCount,
              plan.neighborOffsets.count == plan.vertexCount + 1,
              plan.neighborQueryRows.count == plan.neighborBlends.count
        else {
            throw SmeltSkinTransferRuntimeError.invalidInput
        }
        return try transfer(
            plan: plan,
            sampled: buffer(sampledWeights.values),
            queryCount: sampledWeights.vertexCount,
            jointCount: sampledWeights.jointCount,
            jointMajor: false
        )
    }

    func transfer(
        plan: SmeltSkinTransferPlan,
        sampledField: SmeltGPUSkinField
    ) throws -> SmeltVertexSkin {
        guard sampledField.vertexCount == plan.querySourceIndices.count,
              sampledField.jointCount > 0,
              sampledField.jointCount <= Int(UInt16.max),
              sampledField.jointMajorWeights.length
                >= sampledField.vertexCount * sampledField.jointCount
                    * MemoryLayout<Float>.stride,
              plan.neighborOffsets.count == plan.vertexCount + 1,
              plan.neighborQueryRows.count == plan.neighborBlends.count
        else {
            throw SmeltSkinTransferRuntimeError.invalidInput
        }
        return try transfer(
            plan: plan,
            sampled: sampledField.jointMajorWeights,
            queryCount: sampledField.vertexCount,
            jointCount: sampledField.jointCount,
            jointMajor: true
        )
    }

    private func transfer(
        plan: SmeltSkinTransferPlan,
        sampled: MTLBuffer,
        queryCount: Int,
        jointCount: Int,
        jointMajor: Bool
    ) throws -> SmeltVertexSkin {
        let offsets = try buffer(plan.neighborOffsets.map(UInt32.init))
        let rows = try buffer(plan.neighborQueryRows.map(UInt32.init))
        let blends = try buffer(plan.neighborBlends)
        let jointBytes = plan.vertexCount * 4 * MemoryLayout<UInt16>.stride
        let weightBytes = plan.vertexCount * 4 * MemoryLayout<Float>.stride
        guard let outputJoints = device.makeBuffer(length: jointBytes, options: .storageModeShared),
              let outputWeights = device.makeBuffer(
                  length: weightBytes,
                  options: .storageModeShared
              ),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw SmeltSkinTransferRuntimeError.bufferCreationFailed
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(sampled, offset: 0, index: 0)
        encoder.setBuffer(offsets, offset: 0, index: 1)
        encoder.setBuffer(rows, offset: 0, index: 2)
        encoder.setBuffer(blends, offset: 0, index: 3)
        encoder.setBuffer(outputJoints, offset: 0, index: 4)
        encoder.setBuffer(outputWeights, offset: 0, index: 5)
        var vertexCount = UInt32(plan.vertexCount)
        var jointCountValue = UInt32(jointCount)
        var queryCountValue = UInt32(queryCount)
        var jointMajorValue: UInt32 = jointMajor ? 1 : 0
        encoder.setBytes(&vertexCount, length: 4, index: 6)
        encoder.setBytes(&jointCountValue, length: 4, index: 7)
        encoder.setBytes(&queryCountValue, length: 4, index: 8)
        encoder.setBytes(&jointMajorValue, length: 4, index: 9)
        encoder.dispatchThreads(
            MTLSize(width: plan.vertexCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(pipeline.maxTotalThreadsPerThreadgroup, 256),
                height: 1,
                depth: 1
            )
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw SmeltSkinTransferRuntimeError.gpuExecutionFailed("\(error)")
        }
        let joints = Array(
            UnsafeBufferPointer(
                start: outputJoints.contents().bindMemory(
                    to: UInt16.self,
                    capacity: plan.vertexCount * 4
                ),
                count: plan.vertexCount * 4
            )
        )
        let weights = Array(
            UnsafeBufferPointer(
                start: outputWeights.contents().bindMemory(
                    to: Float.self,
                    capacity: plan.vertexCount * 4
                ),
                count: plan.vertexCount * 4
            )
        )
        return SmeltVertexSkin(
            vertexCount: plan.vertexCount,
            jointCount: jointCount,
            jointIndices: joints,
            weights: weights
        )
    }

    private func buffer<T>(_ values: [T]) throws -> MTLBuffer {
        try values.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress,
                  let buffer = device.makeBuffer(
                      bytes: base,
                      length: bytes.count,
                      options: .storageModeShared
                  )
            else {
                throw SmeltSkinTransferRuntimeError.bufferCreationFailed
            }
            return buffer
        }
    }
}

public enum SmeltSkinTransferRuntimeError: Error, Equatable {
    case metalUnavailable
    case commandQueueCreationFailed
    case pipelineMissing
    case bufferCreationFailed
    case invalidInput
    case gpuExecutionFailed(String)
}
