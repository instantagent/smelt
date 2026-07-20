import Foundation
import Metal

/// Production Michelangelo encoder result and the exact source rows selected
/// for its 512 query tokens.
public struct SmeltMeshEncoding: Sendable, Equatable {
    public let selectedSourceIndices: [Int]
    public let embeddings: [Float]

    public init(selectedSourceIndices: [Int], embeddings: [Float]) {
        self.selectedSourceIndices = selectedSourceIndices
        self.embeddings = embeddings
    }
}

/// Staged pure-Smelt execution of Michelangelo's first residual cross block.
/// Keeping this boundary independently callable is intentional: reference
/// captures can stop at every sub-operation before the full encoder is admitted.
public final class SmeltMeshEncoder {
    private static let width = 512
    private static let heads = 8
    private static let headDimension = 64
    private static let mlpWidth = 2_048
    private static let prefix = "mesh_encoder.encoder.cross_attn"

    private let artifact: SmeltRigArtifact
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipelines: [String: MTLComputePipelineState]
    private let attention: SmeltNoncausalAttention

    public init(artifact: SmeltRigArtifact, device: MTLDevice? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else {
            throw SmeltMeshEncoderError.metalUnavailable
        }
        guard let queue = device.makeCommandQueue() else {
            throw SmeltMeshEncoderError.commandQueueCreationFailed
        }
        let library = try artifact.makeLibrary(device: device)
        var pipelines: [String: MTLComputePipelineState] = [:]
        for name in [
            "layer_norm_rows_bf16w_f32",
            "dense_bf16w_f32",
            "extract_interleaved_head_part_f32",
            "add_rows_f32",
            "gelu_f32",
            "rms_norm_rows_bf16w_f32",
        ] {
            guard let function = library.makeFunction(name: name) else {
                throw SmeltMeshEncoderError.pipelineMissing(name)
            }
            do {
                pipelines[name] = try device.makeComputePipelineState(function: function)
            } catch {
                throw SmeltMeshEncoderError.pipelineCreationFailed(
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

    /// Runs one real checkpoint cross-attention + MLP residual block.
    /// Inputs and output are row-major FP32 tensors of width 512.
    public func crossBlock(query: [Float], data: [Float]) throws -> [Float] {
        guard !query.isEmpty, query.count.isMultiple(of: Self.width) else {
            throw SmeltMeshEncoderError.invalidQueryCount(query.count)
        }
        guard !data.isEmpty, data.count.isMultiple(of: Self.width) else {
            throw SmeltMeshEncoderError.invalidDataCount(data.count)
        }
        guard query.allSatisfy(\.isFinite), data.allSatisfy(\.isFinite) else {
            throw SmeltMeshEncoderError.nonFiniteInput
        }
        let queryTokens = query.count / Self.width
        let dataTokens = data.count / Self.width
        let queryInput = try buffer(query, label: "rig.mesh.cross.query")
        let dataInput = try buffer(data, label: "rig.mesh.cross.data")
        let normalizedQuery = try buffer(count: query.count, label: "rig.mesh.cross.qnorm")
        let normalizedData = try buffer(count: data.count, label: "rig.mesh.cross.dnorm")
        let projectedQuery = try buffer(count: query.count, label: "rig.mesh.cross.q")
        let projectedKV = try buffer(
            count: dataTokens * Self.width * 2,
            label: "rig.mesh.cross.kv"
        )
        let key = try buffer(count: data.count, label: "rig.mesh.cross.k")
        let value = try buffer(count: data.count, label: "rig.mesh.cross.v")
        let attended = try buffer(count: query.count, label: "rig.mesh.cross.attended")
        let projectedAttention = try buffer(
            count: query.count,
            label: "rig.mesh.cross.attn-proj"
        )
        let attentionResidual = try buffer(
            count: query.count,
            label: "rig.mesh.cross.attn-residual"
        )
        let normalizedMLP = try buffer(count: query.count, label: "rig.mesh.cross.mlp-norm")
        let expandedMLP = try buffer(
            count: queryTokens * Self.mlpWidth,
            label: "rig.mesh.cross.mlp-expanded"
        )
        let activatedMLP = try buffer(
            count: queryTokens * Self.mlpWidth,
            label: "rig.mesh.cross.mlp-gelu"
        )
        let projectedMLP = try buffer(count: query.count, label: "rig.mesh.cross.mlp-proj")
        let output = try buffer(count: query.count, label: "rig.mesh.cross.output")

        var commandBuffer = try require(
            queue.makeCommandBuffer(),
            .commandBufferCreationFailed
        )
        commandBuffer.label = "rig.mesh.cross-block.pre-attention"
        var encoder = try require(
            commandBuffer.makeComputeCommandEncoder(),
            .commandEncoderCreationFailed
        )
        try encodeLayerNorm(
            encoder,
            input: queryInput,
            output: normalizedQuery,
            rows: queryTokens,
            weight: "\(Self.prefix).ln_1.weight",
            bias: "\(Self.prefix).ln_1.bias"
        )
        try encodeLayerNorm(
            encoder,
            input: dataInput,
            output: normalizedData,
            rows: dataTokens,
            weight: "\(Self.prefix).ln_2.weight",
            bias: "\(Self.prefix).ln_2.bias"
        )
        try encodeDense(
            encoder,
            input: normalizedQuery,
            output: projectedQuery,
            rows: queryTokens,
            inputDimension: Self.width,
            outputDimension: Self.width,
            weight: "\(Self.prefix).attn.c_q.weight",
            bias: nil
        )
        try encodeDense(
            encoder,
            input: normalizedData,
            output: projectedKV,
            rows: dataTokens,
            inputDimension: Self.width,
            outputDimension: Self.width * 2,
            weight: "\(Self.prefix).attn.c_kv.weight",
            bias: nil
        )
        try encodeExtract(
            encoder,
            input: projectedKV,
            output: key,
            tokens: dataTokens,
            parts: 2,
            part: 0
        )
        try encodeExtract(
            encoder,
            input: projectedKV,
            output: value,
            tokens: dataTokens,
            parts: 2,
            part: 1
        )
        try complete(
            encoder: encoder,
            commandBuffer: commandBuffer,
            stage: "cross-block pre-attention"
        )
        try attention.run(
            query: projectedQuery,
            key: key,
            value: value,
            output: attended,
            queryTokens: queryTokens,
            keyValueTokens: dataTokens,
            heads: Self.heads,
            headDimension: Self.headDimension,
            label: "rig.mesh.cross-block.attention"
        )
        commandBuffer = try require(
            queue.makeCommandBuffer(),
            .commandBufferCreationFailed
        )
        commandBuffer.label = "rig.mesh.cross-block.post-attention"
        encoder = try require(
            commandBuffer.makeComputeCommandEncoder(),
            .commandEncoderCreationFailed
        )
        try encodeDense(
            encoder,
            input: attended,
            output: projectedAttention,
            rows: queryTokens,
            inputDimension: Self.width,
            outputDimension: Self.width,
            weight: "\(Self.prefix).attn.c_proj.weight",
            bias: "\(Self.prefix).attn.c_proj.bias"
        )
        try encodeAdd(
            encoder,
            lhs: queryInput,
            rhs: projectedAttention,
            output: attentionResidual,
            count: query.count
        )
        try encodeLayerNorm(
            encoder,
            input: attentionResidual,
            output: normalizedMLP,
            rows: queryTokens,
            weight: "\(Self.prefix).ln_3.weight",
            bias: "\(Self.prefix).ln_3.bias"
        )
        try encodeDense(
            encoder,
            input: normalizedMLP,
            output: expandedMLP,
            rows: queryTokens,
            inputDimension: Self.width,
            outputDimension: Self.mlpWidth,
            weight: "\(Self.prefix).mlp.c_fc.weight",
            bias: "\(Self.prefix).mlp.c_fc.bias"
        )
        try encodeGELU(
            encoder,
            input: expandedMLP,
            output: activatedMLP,
            count: queryTokens * Self.mlpWidth
        )
        try encodeDense(
            encoder,
            input: activatedMLP,
            output: projectedMLP,
            rows: queryTokens,
            inputDimension: Self.mlpWidth,
            outputDimension: Self.width,
            weight: "\(Self.prefix).mlp.c_proj.weight",
            bias: "\(Self.prefix).mlp.c_proj.bias"
        )
        try encodeAdd(
            encoder,
            lhs: attentionResidual,
            rhs: projectedMLP,
            output: output,
            count: query.count
        )
        try complete(
            encoder: encoder,
            commandBuffer: commandBuffer,
            stage: "cross-block post-attention"
        )
        return Array(
            UnsafeBufferPointer(
                start: output.contents().bindMemory(to: Float.self, capacity: query.count),
                count: query.count
            )
        )
    }

    /// Runs one of Michelangelo's eight real residual self-attention blocks.
    public func selfBlock(input: [Float], layer: Int) throws -> [Float] {
        guard (0..<8).contains(layer) else {
            throw SmeltMeshEncoderError.invalidSelfBlock(layer)
        }
        guard !input.isEmpty, input.count.isMultiple(of: Self.width) else {
            throw SmeltMeshEncoderError.invalidQueryCount(input.count)
        }
        guard input.allSatisfy(\.isFinite) else {
            throw SmeltMeshEncoderError.nonFiniteInput
        }
        let tokens = input.count / Self.width
        let prefix = "mesh_encoder.encoder.self_attn.resblocks.\(layer)"
        let source = try buffer(input, label: "rig.mesh.self.\(layer).input")
        let normalizedAttention = try buffer(
            count: input.count,
            label: "rig.mesh.self.\(layer).attn-norm"
        )
        let combinedQKV = try buffer(
            count: input.count * 3,
            label: "rig.mesh.self.\(layer).qkv"
        )
        let query = try buffer(count: input.count, label: "rig.mesh.self.\(layer).q")
        let key = try buffer(count: input.count, label: "rig.mesh.self.\(layer).k")
        let value = try buffer(count: input.count, label: "rig.mesh.self.\(layer).v")
        let attended = try buffer(
            count: input.count,
            label: "rig.mesh.self.\(layer).attended"
        )
        let projectedAttention = try buffer(
            count: input.count,
            label: "rig.mesh.self.\(layer).attn-proj"
        )
        let attentionResidual = try buffer(
            count: input.count,
            label: "rig.mesh.self.\(layer).attn-residual"
        )
        let normalizedMLP = try buffer(
            count: input.count,
            label: "rig.mesh.self.\(layer).mlp-norm"
        )
        let expandedMLP = try buffer(
            count: tokens * Self.mlpWidth,
            label: "rig.mesh.self.\(layer).mlp-expanded"
        )
        let activatedMLP = try buffer(
            count: tokens * Self.mlpWidth,
            label: "rig.mesh.self.\(layer).mlp-gelu"
        )
        let projectedMLP = try buffer(
            count: input.count,
            label: "rig.mesh.self.\(layer).mlp-proj"
        )
        let output = try buffer(count: input.count, label: "rig.mesh.self.\(layer).output")
        let commandBuffer = try require(
            queue.makeCommandBuffer(),
            .commandBufferCreationFailed
        )
        commandBuffer.label = "rig.mesh.self-block.\(layer)"
        let encoder = try require(
            commandBuffer.makeComputeCommandEncoder(),
            .commandEncoderCreationFailed
        )
        try encodeLayerNorm(
            encoder,
            input: source,
            output: normalizedAttention,
            rows: tokens,
            weight: "\(prefix).ln_1.weight",
            bias: "\(prefix).ln_1.bias"
        )
        try encodeDense(
            encoder,
            input: normalizedAttention,
            output: combinedQKV,
            rows: tokens,
            inputDimension: Self.width,
            outputDimension: Self.width * 3,
            weight: "\(prefix).attn.c_qkv.weight",
            bias: nil
        )
        try encodeExtract(
            encoder,
            input: combinedQKV,
            output: query,
            tokens: tokens,
            parts: 3,
            part: 0
        )
        try encodeExtract(
            encoder,
            input: combinedQKV,
            output: key,
            tokens: tokens,
            parts: 3,
            part: 1
        )
        try encodeExtract(
            encoder,
            input: combinedQKV,
            output: value,
            tokens: tokens,
            parts: 3,
            part: 2
        )
        try encodeAttention(
            encoder,
            query: query,
            key: key,
            value: value,
            output: attended,
            queryTokens: tokens,
            dataTokens: tokens
        )
        try encodeDense(
            encoder,
            input: attended,
            output: projectedAttention,
            rows: tokens,
            inputDimension: Self.width,
            outputDimension: Self.width,
            weight: "\(prefix).attn.c_proj.weight",
            bias: "\(prefix).attn.c_proj.bias"
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
            weight: "\(prefix).ln_2.weight",
            bias: "\(prefix).ln_2.bias"
        )
        try encodeDense(
            encoder,
            input: normalizedMLP,
            output: expandedMLP,
            rows: tokens,
            inputDimension: Self.width,
            outputDimension: Self.mlpWidth,
            weight: "\(prefix).mlp.c_fc.weight",
            bias: "\(prefix).mlp.c_fc.bias"
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
            weight: "\(prefix).mlp.c_proj.weight",
            bias: "\(prefix).mlp.c_proj.bias"
        )
        try encodeAdd(
            encoder,
            lhs: attentionResidual,
            rhs: projectedMLP,
            output: output,
            count: input.count
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw SmeltMeshEncoderError.gpuExecutionFailed(
                "self-block \(layer): \(error)"
            )
        }
        return read(output, count: input.count)
    }

    /// Applies Michelangelo's final LayerNorm and the rig model's 512→896 projection
    /// plus the source-default BF16-epsilon RMSNorm.
    public func finish(input: [Float]) throws -> [Float] {
        guard !input.isEmpty, input.count.isMultiple(of: Self.width) else {
            throw SmeltMeshEncoderError.invalidQueryCount(input.count)
        }
        guard input.allSatisfy(\.isFinite) else {
            throw SmeltMeshEncoderError.nonFiniteInput
        }
        let tokens = input.count / Self.width
        let outputWidth = 896
        let source = try buffer(input, label: "rig.mesh.finish.input")
        let normalized = try buffer(count: input.count, label: "rig.mesh.finish.norm")
        let projected = try buffer(
            count: tokens * outputWidth,
            label: "rig.mesh.finish.projected"
        )
        let output = try buffer(count: tokens * outputWidth, label: "rig.mesh.finish.output")
        let commandBuffer = try require(
            queue.makeCommandBuffer(),
            .commandBufferCreationFailed
        )
        commandBuffer.label = "rig.mesh.finish"
        let encoder = try require(
            commandBuffer.makeComputeCommandEncoder(),
            .commandEncoderCreationFailed
        )
        try encodeLayerNorm(
            encoder,
            input: source,
            output: normalized,
            rows: tokens,
            weight: "mesh_encoder.encoder.ln_post.weight",
            bias: "mesh_encoder.encoder.ln_post.bias"
        )
        try encodeDense(
            encoder,
            input: normalized,
            output: projected,
            rows: tokens,
            inputDimension: Self.width,
            outputDimension: outputWidth,
            weight: "output_proj.0.weight",
            bias: "output_proj.0.bias"
        )
        try encodeRMSNorm(
            encoder,
            input: projected,
            output: output,
            rows: tokens,
            dimension: outputWidth,
            weight: "output_proj.1.weight"
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw SmeltMeshEncoderError.gpuExecutionFailed(
                "finish: \(error)"
            )
        }
        return read(output, count: tokens * outputWidth)
    }

    /// Reduced-shape bring-up route through the entire real Michelangelo graph.
    /// Production uses the same functions with 512 query and 54,000 data rows.
    public func encodeReduced(query: [Float], data: [Float]) throws -> [Float] {
        var hidden = try crossBlock(query: query, data: data)
        for layer in 0..<8 {
            hidden = try selfBlock(input: hidden, layer: layer)
        }
        return try finish(input: hidden)
    }

    /// Executes the production 54,000-row Michelangelo path using deterministic
    /// seed-0 point selection and the package's authoritative BF16 weights.
    public func encode(pointNormals: [Float]) throws -> SmeltMeshEncoding {
        let selection = try SmeltPointSelector.select(
            pointNormals: pointNormals,
            projection: .mesh
        )
        let projection = try SmeltPointProjectionRuntime(
            artifact: artifact,
            device: device
        )
        let data = try projection.project(pointNormals: pointNormals, projection: .mesh)
        let query = try projection.project(
            pointNormals: selection.pointNormals,
            projection: .mesh
        )
        return SmeltMeshEncoding(
            selectedSourceIndices: selection.sourceIndices,
            embeddings: try encodeReduced(query: query, data: data)
        )
    }

    private func encodeLayerNorm(
        _ encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        output: MTLBuffer,
        rows: Int,
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
        var epsilon: Float = 1e-5
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

    private func encodeExtract(
        _ encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        output: MTLBuffer,
        tokens: Int,
        parts: UInt32,
        part: UInt32
    ) throws {
        encoder.setComputePipelineState(try pipeline("extract_interleaved_head_part_f32"))
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        var tokens = UInt32(tokens)
        var heads = UInt32(Self.heads)
        var headDimension = UInt32(Self.headDimension)
        var parts = parts
        var part = part
        encoder.setBytes(&tokens, length: 4, index: 2)
        encoder.setBytes(&heads, length: 4, index: 3)
        encoder.setBytes(&headDimension, length: 4, index: 4)
        encoder.setBytes(&parts, length: 4, index: 5)
        encoder.setBytes(&part, length: 4, index: 6)
        encoder.dispatchThreads(
            MTLSize(width: Self.width, height: Int(tokens), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
        )
    }

    private func encodeRMSNorm(
        _ encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        output: MTLBuffer,
        rows: Int,
        dimension: Int,
        weight: String
    ) throws {
        encoder.setComputePipelineState(try pipeline("rms_norm_rows_bf16w_f32"))
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(
            try artifact.makeWeightBuffer(device: device, tensorNamed: weight),
            offset: 0,
            index: 1
        )
        encoder.setBuffer(output, offset: 0, index: 2)
        var rows = UInt32(rows)
        var dimension = UInt32(dimension)
        var epsilon: Float = 0.0078125
        encoder.setBytes(&rows, length: 4, index: 3)
        encoder.setBytes(&dimension, length: 4, index: 4)
        encoder.setBytes(&epsilon, length: 4, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(rows), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
    }

    private func encodeAttention(
        _ encoder: MTLComputeCommandEncoder,
        query: MTLBuffer,
        key: MTLBuffer,
        value: MTLBuffer,
        output: MTLBuffer,
        queryTokens: Int,
        dataTokens: Int
    ) throws {
        attention.encodeMonolithic(
            encoder,
            query: query,
            key: key,
            value: value,
            output: output,
            queryTokens: queryTokens,
            keyValueTokens: dataTokens,
            heads: Self.heads,
            headDimension: Self.headDimension
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
                throw SmeltMeshEncoderError.bufferCreationFailed(label)
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
            throw SmeltMeshEncoderError.bufferCreationFailed(label)
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
            throw SmeltMeshEncoderError.pipelineMissing(name)
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
            throw SmeltMeshEncoderError.gpuExecutionFailed(
                "\(stage): \(error)"
            )
        }
    }

    private func require<T>(
        _ value: T?,
        _ error: SmeltMeshEncoderError
    ) throws -> T {
        guard let value else { throw error }
        return value
    }
}

public enum SmeltMeshEncoderError: Error, Equatable {
    case metalUnavailable
    case commandQueueCreationFailed
    case commandBufferCreationFailed
    case commandEncoderCreationFailed
    case pipelineMissing(String)
    case pipelineCreationFailed(String, String)
    case bufferCreationFailed(String)
    case invalidQueryCount(Int)
    case invalidDataCount(Int)
    case nonFiniteInput
    case invalidSelfBlock(Int)
    case gpuExecutionFailed(String)
}
