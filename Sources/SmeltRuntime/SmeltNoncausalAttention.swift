import Foundation
import Metal

/// Shared non-causal attention execution block. Short key/value sequences use
/// one dispatch; long sequences preserve the same online-softmax recurrence
/// across bounded command buffers.
final class SmeltNoncausalAttention {
    static let maximumKeysPerCommandBuffer = 1_024

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let monolithic: MTLComputePipelineState
    private let update: MTLComputePipelineState

    init(
        device: MTLDevice,
        queue: MTLCommandQueue,
        library: MTLLibrary
    ) throws {
        self.device = device
        self.queue = queue
        monolithic = try Self.pipeline(
            device: device,
            library: library,
            name: "noncausal_attention_f32"
        )
        update = try Self.pipeline(
            device: device,
            library: library,
            name: "noncausal_attention_update_f32"
        )
    }

    func encodeMonolithic(
        _ encoder: MTLComputeCommandEncoder,
        query: MTLBuffer,
        key: MTLBuffer,
        value: MTLBuffer,
        output: MTLBuffer,
        queryTokens: Int,
        keyValueTokens: Int,
        heads: Int,
        headDimension: Int
    ) {
        encoder.setComputePipelineState(monolithic)
        encoder.setBuffer(query, offset: 0, index: 0)
        encoder.setBuffer(key, offset: 0, index: 1)
        encoder.setBuffer(value, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var queryTokens = UInt32(queryTokens)
        var keyValueTokens = UInt32(keyValueTokens)
        var heads = UInt32(heads)
        var headDimension = UInt32(headDimension)
        encoder.setBytes(&queryTokens, length: 4, index: 4)
        encoder.setBytes(&keyValueTokens, length: 4, index: 5)
        encoder.setBytes(&heads, length: 4, index: 6)
        encoder.setBytes(&headDimension, length: 4, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(queryTokens), height: Int(heads), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    func run(
        query: MTLBuffer,
        key: MTLBuffer,
        value: MTLBuffer,
        output: MTLBuffer,
        queryTokens: Int,
        keyValueTokens: Int,
        heads: Int,
        headDimension: Int,
        label: String
    ) throws {
        guard queryTokens > 0, keyValueTokens > 0,
              heads > 0, headDimension > 0
        else {
            throw SmeltNoncausalAttentionError.invalidShape
        }
        if keyValueTokens <= Self.maximumKeysPerCommandBuffer {
            let commandBuffer = try require(
                queue.makeCommandBuffer(),
                .commandBufferCreationFailed
            )
            commandBuffer.label = label
            let encoder = try require(
                commandBuffer.makeComputeCommandEncoder(),
                .commandEncoderCreationFailed
            )
            encodeMonolithic(
                encoder,
                query: query,
                key: key,
                value: value,
                output: output,
                queryTokens: queryTokens,
                keyValueTokens: keyValueTokens,
                heads: heads,
                headDimension: headDimension
            )
            try complete(
                encoder: encoder,
                commandBuffer: commandBuffer,
                stage: label
            )
            return
        }

        let stateCount = queryTokens * heads
        let maximumState = try buffer(
            [Float](repeating: -.infinity, count: stateCount),
            label: "\(label).maximum"
        )
        let denominatorState = try buffer(
            [Float](repeating: 0, count: stateCount),
            label: "\(label).denominator"
        )
        output.contents().initializeMemory(
            as: UInt8.self,
            repeating: 0,
            count: output.length
        )
        var sourceStart = 0
        while sourceStart < keyValueTokens {
            let sourceCount = min(
                Self.maximumKeysPerCommandBuffer,
                keyValueTokens - sourceStart
            )
            let commandBuffer = try require(
                queue.makeCommandBuffer(),
                .commandBufferCreationFailed
            )
            let chunkLabel = "\(label).keys.\(sourceStart)..<\(sourceStart + sourceCount)"
            commandBuffer.label = chunkLabel
            let encoder = try require(
                commandBuffer.makeComputeCommandEncoder(),
                .commandEncoderCreationFailed
            )
            encodeUpdate(
                encoder,
                query: query,
                key: key,
                value: value,
                accumulator: output,
                maximumState: maximumState,
                denominatorState: denominatorState,
                queryTokens: queryTokens,
                keyValueTokens: keyValueTokens,
                heads: heads,
                headDimension: headDimension,
                sourceStart: sourceStart,
                sourceCount: sourceCount,
                finalize: sourceStart + sourceCount == keyValueTokens
            )
            try complete(
                encoder: encoder,
                commandBuffer: commandBuffer,
                stage: chunkLabel
            )
            sourceStart += sourceCount
        }
    }

    private func encodeUpdate(
        _ encoder: MTLComputeCommandEncoder,
        query: MTLBuffer,
        key: MTLBuffer,
        value: MTLBuffer,
        accumulator: MTLBuffer,
        maximumState: MTLBuffer,
        denominatorState: MTLBuffer,
        queryTokens: Int,
        keyValueTokens: Int,
        heads: Int,
        headDimension: Int,
        sourceStart: Int,
        sourceCount: Int,
        finalize: Bool
    ) {
        encoder.setComputePipelineState(update)
        encoder.setBuffer(query, offset: 0, index: 0)
        encoder.setBuffer(key, offset: 0, index: 1)
        encoder.setBuffer(value, offset: 0, index: 2)
        encoder.setBuffer(accumulator, offset: 0, index: 3)
        encoder.setBuffer(maximumState, offset: 0, index: 4)
        encoder.setBuffer(denominatorState, offset: 0, index: 5)
        var queryTokens = UInt32(queryTokens)
        var keyValueTokens = UInt32(keyValueTokens)
        var heads = UInt32(heads)
        var headDimension = UInt32(headDimension)
        var sourceStart = UInt32(sourceStart)
        var sourceCount = UInt32(sourceCount)
        var finalize: UInt32 = finalize ? 1 : 0
        encoder.setBytes(&queryTokens, length: 4, index: 6)
        encoder.setBytes(&keyValueTokens, length: 4, index: 7)
        encoder.setBytes(&heads, length: 4, index: 8)
        encoder.setBytes(&headDimension, length: 4, index: 9)
        encoder.setBytes(&sourceStart, length: 4, index: 10)
        encoder.setBytes(&sourceCount, length: 4, index: 11)
        encoder.setBytes(&finalize, length: 4, index: 12)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(queryTokens), height: Int(heads), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    private func buffer(_ values: [Float], label: String) throws -> MTLBuffer {
        let result = try values.withUnsafeBytes { bytes -> MTLBuffer in
            guard let base = bytes.baseAddress,
                  let buffer = device.makeBuffer(
                      bytes: base,
                      length: bytes.count,
                      options: .storageModeShared
                  )
            else {
                throw SmeltNoncausalAttentionError.bufferCreationFailed(label)
            }
            return buffer
        }
        result.label = label
        return result
    }

    private static func pipeline(
        device: MTLDevice,
        library: MTLLibrary,
        name: String
    ) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw SmeltNoncausalAttentionError.pipelineMissing(name)
        }
        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            throw SmeltNoncausalAttentionError.pipelineCreationFailed(
                name,
                "\(error)"
            )
        }
    }

    private func complete(
        encoder: MTLComputeCommandEncoder,
        commandBuffer: MTLCommandBuffer,
        stage: String
    ) throws {
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw SmeltNoncausalAttentionError.gpuExecutionFailed(
                stage,
                "\(error)"
            )
        }
    }

    private func require<T>(
        _ value: T?,
        _ error: SmeltNoncausalAttentionError
    ) throws -> T {
        guard let value else { throw error }
        return value
    }
}

enum SmeltNoncausalAttentionError: Error, Equatable {
    case invalidShape
    case commandBufferCreationFailed
    case commandEncoderCreationFailed
    case bufferCreationFailed(String)
    case pipelineMissing(String)
    case pipelineCreationFailed(String, String)
    case gpuExecutionFailed(String, String)
}
