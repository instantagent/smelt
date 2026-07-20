import Foundation
import Metal

/// SkinVAE condition-encoder output and its deterministic 384 source rows.
public struct SmeltSkinConditionEncoding: Sendable, Equatable {
    public let selectedSourceIndices: [Int]
    public let conditionTokens: [Float]

    public init(selectedSourceIndices: [Int], conditionTokens: [Float]) {
        self.selectedSourceIndices = selectedSourceIndices
        self.conditionTokens = conditionTokens
    }
}

/// Staged pure-Smelt SkinVAE condition encoder using the pinned real weights.
public final class SmeltSkinConditionEncoder {
    private static let width = 768
    private static let heads = 12
    private static let headDimension = 64
    private static let mlpWidth = 3_072
    private static let prefix = "vae.model.cond_encoder"

    private let artifact: SmeltRigArtifact
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipelines: [String: MTLComputePipelineState]
    private let attention: SmeltNoncausalAttention

    public init(artifact: SmeltRigArtifact, device: MTLDevice? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else {
            throw SmeltSkinConditionEncoderError.metalUnavailable
        }
        guard let queue = device.makeCommandQueue() else {
            throw SmeltSkinConditionEncoderError.commandQueueCreationFailed
        }
        let library = try artifact.makeLibrary(device: device)
        var pipelines: [String: MTLComputePipelineState] = [:]
        for name in [
            "layer_norm_rows_bf16w_f32",
            "dense_bf16w_f32",
            "repack_concatenated_head_parts_f32",
            "add_rows_f32",
            "gelu_f32",
        ] {
            guard let function = library.makeFunction(name: name) else {
                throw SmeltSkinConditionEncoderError.pipelineMissing(name)
            }
            do {
                pipelines[name] = try device.makeComputePipelineState(function: function)
            } catch {
                throw SmeltSkinConditionEncoderError.pipelineCreationFailed(
                    name,
                    "\(error)"
                )
            }
        }
        self.artifact = artifact
        self.device = device
        self.queue = queue
        self.pipelines = pipelines
        attention = try SmeltNoncausalAttention(
            device: device,
            queue: queue,
            library: library
        )
    }

    /// Reduced-shape first-divergence route through all three encoder blocks,
    /// final LayerNorm, and the 768→512 condition quantizer.
    public func encodeReduced(query: [Float], data: [Float]) throws -> [Float] {
        var hidden = try block(input: query, data: data, layer: 0)
        hidden = try block(input: hidden, data: nil, layer: 1)
        hidden = try block(input: hidden, data: nil, layer: 2)
        return try finish(input: hidden)
    }

    /// Executes the production 54,000-row / 384-token condition encoder.
    public func encode(pointNormals: [Float]) throws -> SmeltSkinConditionEncoding {
        let selection = try SmeltPointSelector.select(
            pointNormals: pointNormals,
            projection: .condition
        )
        let projection = try SmeltPointProjectionRuntime(
            artifact: artifact,
            device: device
        )
        let data = try projection.project(
            pointNormals: pointNormals,
            projection: .condition
        )
        let query = try projection.project(
            pointNormals: selection.pointNormals,
            projection: .condition
        )
        return SmeltSkinConditionEncoding(
            selectedSourceIndices: selection.sourceIndices,
            conditionTokens: try encodeReduced(query: query, data: data)
        )
    }

    private func block(input: [Float], data: [Float]?, layer: Int) throws -> [Float] {
        guard (0..<3).contains(layer), (layer == 0) == (data != nil) else {
            throw SmeltSkinConditionEncoderError.invalidBlock(layer)
        }
        guard !input.isEmpty, input.count.isMultiple(of: Self.width),
              input.allSatisfy(\.isFinite)
        else {
            throw SmeltSkinConditionEncoderError.invalidInputCount(input.count)
        }
        if let data {
            guard !data.isEmpty, data.count.isMultiple(of: Self.width),
                  data.allSatisfy(\.isFinite)
            else {
                throw SmeltSkinConditionEncoderError.invalidDataCount(data.count)
            }
        }
        let tokens = input.count / Self.width
        let dataTokens = (data?.count ?? input.count) / Self.width
        let prefix = "\(Self.prefix).blocks.\(layer)"
        let source = try buffer(input, label: "rig.vae.cond.\(layer).input")
        let attentionData = try data.map {
            try buffer($0, label: "rig.vae.cond.\(layer).data")
        } ?? source
        let normalizedQuery = try buffer(count: input.count, label: "rig.vae.cond.norm-q")
        let normalizedData = try buffer(
            count: dataTokens * Self.width,
            label: "rig.vae.cond.norm-data"
        )
        let rawQuery = try buffer(count: input.count, label: "rig.vae.cond.raw-q")
        let rawKey = try buffer(
            count: dataTokens * Self.width,
            label: "rig.vae.cond.raw-k"
        )
        let rawValue = try buffer(
            count: dataTokens * Self.width,
            label: "rig.vae.cond.raw-v"
        )
        let query = try buffer(count: input.count, label: "rig.vae.cond.q")
        let key = try buffer(count: dataTokens * Self.width, label: "rig.vae.cond.k")
        let value = try buffer(count: dataTokens * Self.width, label: "rig.vae.cond.v")
        let unused = try buffer(
            count: max(input.count, dataTokens * Self.width),
            label: "rig.vae.cond.unused"
        )
        let attended = try buffer(count: input.count, label: "rig.vae.cond.attended")
        let projectedAttention = try buffer(
            count: input.count,
            label: "rig.vae.cond.attn-proj"
        )
        let attentionResidual = try buffer(
            count: input.count,
            label: "rig.vae.cond.attn-residual"
        )
        let normalizedMLP = try buffer(count: input.count, label: "rig.vae.cond.mlp-norm")
        let expandedMLP = try buffer(
            count: tokens * Self.mlpWidth,
            label: "rig.vae.cond.mlp-expanded"
        )
        let activatedMLP = try buffer(
            count: tokens * Self.mlpWidth,
            label: "rig.vae.cond.mlp-gelu"
        )
        let projectedMLP = try buffer(count: input.count, label: "rig.vae.cond.mlp-proj")
        let output = try buffer(count: input.count, label: "rig.vae.cond.output")
        var commandBuffer = try require(
            queue.makeCommandBuffer(),
            .commandBufferCreationFailed
        )
        commandBuffer.label = "rig.condition.\(layer).pre-attention"
        var encoder = try require(
            commandBuffer.makeComputeCommandEncoder(),
            .commandEncoderCreationFailed
        )

        if layer == 0 {
            try encodeLayerNorm(
                encoder,
                input: source,
                output: normalizedQuery,
                rows: tokens,
                epsilon: 1e-5,
                weight: "\(prefix).norm2.weight",
                bias: "\(prefix).norm2.bias"
            )
            try encodeLayerNorm(
                encoder,
                input: attentionData,
                output: normalizedData,
                rows: dataTokens,
                epsilon: 1e-6,
                weight: "\(prefix).attn2.norm_cross.weight",
                bias: "\(prefix).attn2.norm_cross.bias"
            )
        } else {
            try encodeLayerNorm(
                encoder,
                input: source,
                output: normalizedQuery,
                rows: tokens,
                epsilon: 1e-5,
                weight: "\(prefix).norm1.weight",
                bias: "\(prefix).norm1.bias"
            )
        }
        let attentionPrefix = layer == 0 ? "\(prefix).attn2" : "\(prefix).attn1"
        try encodeDense(
            encoder,
            input: normalizedQuery,
            output: rawQuery,
            rows: tokens,
            inputDimension: Self.width,
            outputDimension: Self.width,
            weight: "\(attentionPrefix).to_q.weight",
            bias: nil
        )
        let keyValueInput = layer == 0 ? normalizedData : normalizedQuery
        try encodeDense(
            encoder,
            input: keyValueInput,
            output: rawKey,
            rows: dataTokens,
            inputDimension: Self.width,
            outputDimension: Self.width,
            weight: "\(attentionPrefix).to_k.weight",
            bias: nil
        )
        try encodeDense(
            encoder,
            input: keyValueInput,
            output: rawValue,
            rows: dataTokens,
            inputDimension: Self.width,
            outputDimension: Self.width,
            weight: "\(attentionPrefix).to_v.weight",
            bias: nil
        )
        if layer == 0 {
            try encodeRepack(
                encoder,
                inputs: (rawKey, rawValue, rawValue),
                outputs: (key, value, unused),
                tokens: dataTokens,
                parts: 2
            )
        } else {
            try encodeRepack(
                encoder,
                inputs: (rawQuery, rawKey, rawValue),
                outputs: (query, key, value),
                tokens: tokens,
                parts: 3
            )
        }
        try complete(
            encoder: encoder,
            commandBuffer: commandBuffer,
            stage: "condition block \(layer) pre-attention"
        )
        try attention.run(
            query: layer == 0 ? rawQuery : query,
            key: key,
            value: value,
            output: attended,
            queryTokens: tokens,
            keyValueTokens: dataTokens,
            heads: Self.heads,
            headDimension: Self.headDimension,
            label: "rig.condition.\(layer).attention"
        )
        commandBuffer = try require(
            queue.makeCommandBuffer(),
            .commandBufferCreationFailed
        )
        commandBuffer.label = "rig.condition.\(layer).post-attention"
        encoder = try require(
            commandBuffer.makeComputeCommandEncoder(),
            .commandEncoderCreationFailed
        )
        try encodeDense(
            encoder,
            input: attended,
            output: projectedAttention,
            rows: tokens,
            inputDimension: Self.width,
            outputDimension: Self.width,
            weight: "\(attentionPrefix).to_out.0.weight",
            bias: "\(attentionPrefix).to_out.0.bias"
        )
        try encodeAdd(
            encoder,
            lhs: source,
            rhs: projectedAttention,
            output: attentionResidual,
            count: input.count
        )
        try encodeLayerNorm(
            encoder,
            input: attentionResidual,
            output: normalizedMLP,
            rows: tokens,
            epsilon: 1e-5,
            weight: "\(prefix).norm3.weight",
            bias: "\(prefix).norm3.bias"
        )
        try encodeDense(
            encoder,
            input: normalizedMLP,
            output: expandedMLP,
            rows: tokens,
            inputDimension: Self.width,
            outputDimension: Self.mlpWidth,
            weight: "\(prefix).ff.net.0.proj.weight",
            bias: "\(prefix).ff.net.0.proj.bias"
        )
        try encodeGELU(
            encoder,
            input: expandedMLP,
            output: activatedMLP,
            count: tokens * Self.mlpWidth
        )
        try encodeDense(
            encoder,
            input: activatedMLP,
            output: projectedMLP,
            rows: tokens,
            inputDimension: Self.mlpWidth,
            outputDimension: Self.width,
            weight: "\(prefix).ff.net.2.weight",
            bias: "\(prefix).ff.net.2.bias"
        )
        try encodeAdd(
            encoder,
            lhs: attentionResidual,
            rhs: projectedMLP,
            output: output,
            count: input.count
        )
        try complete(
            encoder: encoder,
            commandBuffer: commandBuffer,
            stage: "condition block \(layer) post-attention"
        )
        return read(output, count: input.count)
    }

    private func finish(input: [Float]) throws -> [Float] {
        let tokens = input.count / Self.width
        let source = try buffer(input, label: "rig.vae.cond.finish-input")
        let normalized = try buffer(count: input.count, label: "rig.vae.cond.finish-norm")
        let output = try buffer(count: tokens * 512, label: "rig.vae.cond.finish-output")
        let commandBuffer = try require(
            queue.makeCommandBuffer(),
            .commandBufferCreationFailed
        )
        commandBuffer.label = "rig.condition.finish"
        let encoder = try require(
            commandBuffer.makeComputeCommandEncoder(),
            .commandEncoderCreationFailed
        )
        try encodeLayerNorm(
            encoder,
            input: source,
            output: normalized,
            rows: tokens,
            epsilon: 1e-5,
            weight: "\(Self.prefix).norm_out.weight",
            bias: "\(Self.prefix).norm_out.bias"
        )
        try encodeDense(
            encoder,
            input: normalized,
            output: output,
            rows: tokens,
            inputDimension: Self.width,
            outputDimension: 512,
            weight: "vae.model.cond_quant.weight",
            bias: "vae.model.cond_quant.bias"
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw SmeltSkinConditionEncoderError.gpuExecutionFailed(
                "condition finish: \(error)"
            )
        }
        return read(output, count: tokens * 512)
    }

    private func encodeLayerNorm(
        _ encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        output: MTLBuffer,
        rows: Int,
        epsilon: Float,
        weight: String,
        bias: String
    ) throws {
        encoder.setComputePipelineState(try pipeline("layer_norm_rows_bf16w_f32"))
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(try artifact.makeWeightBuffer(device: device, tensorNamed: weight), offset: 0, index: 1)
        encoder.setBuffer(try artifact.makeWeightBuffer(device: device, tensorNamed: bias), offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var rows = UInt32(rows)
        var dimension = UInt32(Self.width)
        var epsilon = epsilon
        encoder.setBytes(&rows, length: 4, index: 4)
        encoder.setBytes(&dimension, length: 4, index: 5)
        encoder.setBytes(&epsilon, length: 4, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(rows), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
    }

    private func encodeDense(
        _ encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        output: MTLBuffer,
        rows: Int,
        inputDimension: Int,
        outputDimension: Int,
        weight: String,
        bias: String?
    ) throws {
        let weightBuffer = try artifact.makeWeightBuffer(device: device, tensorNamed: weight)
        let biasBuffer = try bias.map {
            try artifact.makeWeightBuffer(device: device, tensorNamed: $0)
        } ?? weightBuffer
        encoder.setComputePipelineState(try pipeline("dense_bf16w_f32"))
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: 0, index: 1)
        encoder.setBuffer(biasBuffer, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var rows = UInt32(rows)
        var outputDimension = UInt32(outputDimension)
        var inputDimension = UInt32(inputDimension)
        var hasBias: UInt32 = bias == nil ? 0 : 1
        encoder.setBytes(&rows, length: 4, index: 4)
        encoder.setBytes(&outputDimension, length: 4, index: 5)
        encoder.setBytes(&inputDimension, length: 4, index: 6)
        encoder.setBytes(&hasBias, length: 4, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(outputDimension), height: Int(rows), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    private func encodeRepack(
        _ encoder: MTLComputeCommandEncoder,
        inputs: (MTLBuffer, MTLBuffer, MTLBuffer),
        outputs: (MTLBuffer, MTLBuffer, MTLBuffer),
        tokens: Int,
        parts: UInt32
    ) throws {
        encoder.setComputePipelineState(try pipeline("repack_concatenated_head_parts_f32"))
        encoder.setBuffer(inputs.0, offset: 0, index: 0)
        encoder.setBuffer(inputs.1, offset: 0, index: 1)
        encoder.setBuffer(inputs.2, offset: 0, index: 2)
        encoder.setBuffer(outputs.0, offset: 0, index: 3)
        encoder.setBuffer(outputs.1, offset: 0, index: 4)
        encoder.setBuffer(outputs.2, offset: 0, index: 5)
        var tokens = UInt32(tokens)
        var heads = UInt32(Self.heads)
        var headDimension = UInt32(Self.headDimension)
        var parts = parts
        encoder.setBytes(&tokens, length: 4, index: 6)
        encoder.setBytes(&heads, length: 4, index: 7)
        encoder.setBytes(&headDimension, length: 4, index: 8)
        encoder.setBytes(&parts, length: 4, index: 9)
        encoder.dispatchThreads(
            MTLSize(width: Self.width, height: Int(tokens), depth: Int(parts)),
            threadsPerThreadgroup: MTLSize(width: 32, height: 2, depth: 1)
        )
    }

    private func encodeAdd(
        _ encoder: MTLComputeCommandEncoder,
        lhs: MTLBuffer,
        rhs: MTLBuffer,
        output: MTLBuffer,
        count: Int
    ) throws {
        encoder.setComputePipelineState(try pipeline("add_rows_f32"))
        encoder.setBuffer(lhs, offset: 0, index: 0)
        encoder.setBuffer(rhs, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var count = UInt32(count)
        encoder.setBytes(&count, length: 4, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
    }

    private func encodeGELU(
        _ encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        output: MTLBuffer,
        count: Int
    ) throws {
        encoder.setComputePipelineState(try pipeline("gelu_f32"))
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        var count = UInt32(count)
        encoder.setBytes(&count, length: 4, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
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
                throw SmeltSkinConditionEncoderError.bufferCreationFailed(label)
            }
            return buffer
        }
        result.label = label
        return result
    }

    private func buffer(count: Int, label: String) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(
            length: count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw SmeltSkinConditionEncoderError.bufferCreationFailed(label)
        }
        buffer.label = label
        return buffer
    }

    private func read(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(
            UnsafeBufferPointer(
                start: buffer.contents().bindMemory(to: Float.self, capacity: count),
                count: count
            )
        )
    }

    private func pipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let pipeline = pipelines[name] else {
            throw SmeltSkinConditionEncoderError.pipelineMissing(name)
        }
        return pipeline
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
            throw SmeltSkinConditionEncoderError.gpuExecutionFailed(
                "\(stage): \(error)"
            )
        }
    }

    private func require<T>(
        _ value: T?,
        _ error: SmeltSkinConditionEncoderError
    ) throws -> T {
        guard let value else { throw error }
        return value
    }
}

public enum SmeltSkinConditionEncoderError: Error, Equatable {
    case metalUnavailable
    case commandQueueCreationFailed
    case commandBufferCreationFailed
    case commandEncoderCreationFailed
    case pipelineMissing(String)
    case pipelineCreationFailed(String, String)
    case bufferCreationFailed(String)
    case invalidBlock(Int)
    case invalidInputCount(Int)
    case invalidDataCount(Int)
    case gpuExecutionFailed(String)
}
