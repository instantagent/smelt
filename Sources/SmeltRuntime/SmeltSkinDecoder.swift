import Foundation
import Metal

/// Exactness capture for one staged SkinVAE decoder execution.
public struct SmeltSkinDecoderCapture: Sendable, Equatable {
    public let fsqCodes: [Float]
    public let postQuantOutput: [Float]
    public let selfLayerOutputs: [[Float]]
    public let projectedQueries: [Float]
    public let crossOutput: [Float]
    public let normalizedOutput: [Float]
    public let rawLogits: [Float]
    public let probabilities: [Float]

    public init(
        fsqCodes: [Float],
        postQuantOutput: [Float],
        selfLayerOutputs: [[Float]],
        projectedQueries: [Float],
        crossOutput: [Float],
        normalizedOutput: [Float],
        rawLogits: [Float],
        probabilities: [Float]
    ) {
        self.fsqCodes = fsqCodes
        self.postQuantOutput = postQuantOutput
        self.selfLayerOutputs = selfLayerOutputs
        self.projectedQueries = projectedQueries
        self.crossOutput = crossOutput
        self.normalizedOutput = normalizedOutput
        self.rawLogits = rawLogits
        self.probabilities = probabilities
    }
}

/// Vertex-major decoded skin weights with one column per generated joint.
public struct SmeltSkinWeights: Sendable, Equatable {
    public let vertexCount: Int
    public let jointCount: Int
    public let values: [Float]

    public init(vertexCount: Int, jointCount: Int, values: [Float]) {
        self.vertexCount = vertexCount
        self.jointCount = jointCount
        self.values = values
    }
}

struct SmeltGPUSkinField {
  let vertexCount: Int
  let jointCount: Int
  let jointMajorWeights: MTLBuffer
}

struct SmeltSkinDecoderKernelPolicy: Sendable, Equatable {
    static let optimized = SmeltSkinDecoderKernelPolicy(
        maximumDenseRowsPerThreadgroup: 8,
        fuseDenseEpilogue: true,
        selfAttentionQueryTile: 1,
        crossAttentionQueryTile: 16
    )

    let maximumDenseRowsPerThreadgroup: Int
    let fuseDenseEpilogue: Bool
    let selfAttentionQueryTile: Int
    let crossAttentionQueryTile: Int

    init(
        maximumDenseRowsPerThreadgroup: Int,
        fuseDenseEpilogue: Bool,
        attentionQueryTile: Int
    ) {
        self.init(
            maximumDenseRowsPerThreadgroup: maximumDenseRowsPerThreadgroup,
            fuseDenseEpilogue: fuseDenseEpilogue,
            selfAttentionQueryTile: attentionQueryTile,
            crossAttentionQueryTile: attentionQueryTile
        )
    }

    init(
        maximumDenseRowsPerThreadgroup: Int,
        fuseDenseEpilogue: Bool,
        selfAttentionQueryTile: Int,
        crossAttentionQueryTile: Int
    ) {
        precondition(
            [1, 4, 8].contains(maximumDenseRowsPerThreadgroup),
            "dense row tile must be 1, 4, or 8"
        )
        precondition(
            [1, 8].contains(selfAttentionQueryTile),
            "self-attention query tile must be 1 or 8"
        )
        precondition(
            [1, 8, 16].contains(crossAttentionQueryTile),
            "cross-attention query tile must be 1, 8, or 16"
        )
        self.maximumDenseRowsPerThreadgroup = maximumDenseRowsPerThreadgroup
        self.fuseDenseEpilogue = fuseDenseEpilogue
        self.selfAttentionQueryTile = selfAttentionQueryTile
        self.crossAttentionQueryTile = crossAttentionQueryTile
    }
}

/// Staged pure-Smelt SkinVAE decoder backed directly by the pinned BF16
/// checkpoint. The staged route is retained as the optimization oracle.
public final class SmeltSkinDecoder {
    private static let width = 768
    private static let latentWidth = 512
    private static let heads = 12
    private static let headDimension = 64
    private static let mlpWidth = 3_072
    private static let decoderPrefix = "vae.model.decoder"
  private static let fullQueryChunk = 2_048

    private struct PreparedCrossQuery {
        let source: MTLBuffer
        let rawQuery: MTLBuffer
        let rows: Int
    let rowOffset: Int
    }

    private struct PreparedCrossData {
        let key: MTLBuffer
        let value: MTLBuffer
        let rows: Int
    }

  private struct CrossDataWorkspace {
    let normalized: MTLBuffer
    let rawKey: MTLBuffer
    let rawValue: MTLBuffer
    let key: MTLBuffer
    let value: MTLBuffer
    let unused: MTLBuffer
  }

  private struct JointWorkspace {
    let levels: MTLBuffer
    let combined: MTLBuffer
    let selfBlock: SelfBlockWorkspace
    let crossData: CrossDataWorkspace
    let cross: CrossWorkspace
  }

    private struct PreparedLatentCache {
        let buffer: MTLBuffer
        let rows: Int
    }

    private struct AttentionCaptureRequest {
        let root: URL
        let jointOrdinal: Int
        let chunkOrdinal: Int
    }

    private struct CrossWorkspace {
        let attended: MTLBuffer
        let projectedAttention: MTLBuffer
        let attentionResidual: MTLBuffer
        let normalizedMLP: MTLBuffer
        let expandedMLP: MTLBuffer
        let activatedMLP: MTLBuffer
        let projectedMLP: MTLBuffer
        let output: MTLBuffer
        let normalizedOutput: MTLBuffer
        let rawLogits: MTLBuffer
        let probabilities: MTLBuffer
    }

    private struct SelfBlockWorkspace {
        let normalizedQuery: MTLBuffer
        let rawQuery: MTLBuffer
        let rawKey: MTLBuffer
        let rawValue: MTLBuffer
        let query: MTLBuffer
        let key: MTLBuffer
        let value: MTLBuffer
        let attended: MTLBuffer
        let projectedAttention: MTLBuffer
        let attentionResidual: MTLBuffer
        let normalizedMLP: MTLBuffer
        let expandedMLP: MTLBuffer
        let activatedMLP: MTLBuffer
        let projectedMLP: MTLBuffer
        let outputA: MTLBuffer
        let outputB: MTLBuffer
    }

    private enum DenseEpilogue: UInt32 {
        case gelu = 1
        case residualAdd = 2
    }

  private let artifact: SmeltComponentArtifact
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipelines: [String: MTLComputePipelineState]
    private let kernelPolicy: SmeltSkinDecoderKernelPolicy
    private let attentionCaptureRequest: AttentionCaptureRequest?
    private let weightBufferLock = NSLock()
    private var weightBuffers: [String: MTLBuffer] = [:]

    public convenience init(
    artifact: SmeltComponentArtifact,
        device: MTLDevice? = nil
    ) throws {
        try self.init(
            artifact: artifact,
            device: device,
            kernelPolicy: .optimized
        )
    }

    init(
    artifact: SmeltComponentArtifact,
        device: MTLDevice? = nil,
        kernelPolicy: SmeltSkinDecoderKernelPolicy
    ) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else {
            throw SmeltSkinDecoderError.metalUnavailable
        }
        guard let queue = device.makeCommandQueue() else {
            throw SmeltSkinDecoderError.commandQueueCreationFailed
        }
        let library = try artifact.makeLibrary(device: device)
        var pipelines: [String: MTLComputePipelineState] = [:]
        for name in [
            "layer_norm_rows_bf16w_f32",
            "dense_bf16w_f32",
            "dense_bf16w_f32_rows4",
            "dense_bf16w_f32_rows8",
            "dense_bf16w_f32_rows8_epilogue",
            "dense_bf16w_f32_rows8_cols2_epilogue",
            "repack_concatenated_head_parts_f32",
            "noncausal_attention_f32",
            "noncausal_attention_q8_f32",
            "noncausal_attention_q16_f32",
            "add_rows_f32",
            "gelu_f32",
            "fsq_base8x5_decode_f32",
            "fourier_position_embedding_f32",
            "pmpe_bf16_semantics_f32",
            "append_strided_features_f32",
            "sigmoid_f32",
        ] {
            guard let function = library.makeFunction(name: name) else {
                throw SmeltSkinDecoderError.pipelineMissing(name)
            }
            do {
                pipelines[name] = try device.makeComputePipelineState(function: function)
            } catch {
                throw SmeltSkinDecoderError.pipelineCreationFailed(
                    name,
                    "\(error)"
                )
            }
        }
        self.artifact = artifact
        self.device = device
        self.queue = queue
        self.pipelines = pipelines
        self.kernelPolicy = kernelPolicy
        self.attentionCaptureRequest = Self.makeAttentionCaptureRequest()
    }

    /// Runs every decoder boundary and returns its intermediate tensors. This
    /// is the first-divergence route used by reduced source fixtures.
    public func decodeReduced(
        indices: [UInt32],
        conditionTokens: [Float],
        pointNormals: [Float]
    ) throws -> SmeltSkinDecoderCapture {
        let prepared = try prepareLatents(
            indices: indices,
            conditionTokens: conditionTokens
        )
        let projectedQueries = try projectQueries(pointNormals: pointNormals)
        let cross = try transformerBlock(
            input: projectedQueries,
            data: prepared.cache,
            prefix: "\(Self.decoderPrefix).blocks.10"
        )
        let finish = try finish(input: cross)
        return SmeltSkinDecoderCapture(
            fsqCodes: prepared.fsqCodes,
            postQuantOutput: prepared.postQuantOutput,
            selfLayerOutputs: prepared.selfLayerOutputs,
            projectedQueries: projectedQueries,
            crossOutput: cross,
            normalizedOutput: finish.normalized,
            rawLogits: finish.rawLogits,
            probabilities: finish.probabilities
        )
    }

    /// Decodes one joint's four FSQ indices against the production condition
    /// tensor. Vertex queries are chunked because cross-attention is independent
    /// per query; this bounds working memory without changing reduction order.
    public func decode(
        indices: [UInt32],
        conditionTokens: [Float],
        pointNormals: [Float]
    ) throws -> [Float] {
        guard indices.count == 4 else {
            throw SmeltSkinDecoderError.invalidIndexCount(indices.count)
        }
        guard conditionTokens.count == 384 * Self.latentWidth else {
            throw SmeltSkinDecoderError.invalidConditionCount(
                conditionTokens.count
            )
        }
        try validatePointNormals(pointNormals)
        let queries = try prepareCrossQueries(pointNormals: pointNormals)
        let workspace = try makeCrossWorkspace(
            rows: queries.map(\.rows).max() ?? 0
        )
        let selfWorkspace = try makeSelfBlockWorkspace(
            tokens: conditionTokens.count / Self.latentWidth + indices.count
        )
        return try decode(
            indices: indices,
            conditionTokens: conditionTokens,
            preparedQueries: queries,
            workspace: workspace,
            selfWorkspace: selfWorkspace,
            jointOrdinal: nil
        )
    }

    private func decode(
        indices: [UInt32],
        conditionTokens: [Float],
        preparedQueries: [PreparedCrossQuery],
        workspace: CrossWorkspace,
        selfWorkspace: SelfBlockWorkspace,
        jointOrdinal: Int?
    ) throws -> [Float] {
        let prepared = try prepareLatentCache(
            indices: indices,
            conditionTokens: conditionTokens,
            workspace: selfWorkspace
        )
        let crossData = try prepareCrossData(
            input: prepared.buffer,
            tokens: prepared.rows
        )
        var probabilities: [Float] = []
        probabilities.reserveCapacity(
            preparedQueries.reduce(0) { $0 + $1.rows }
        )
        for (chunkOrdinal, query) in preparedQueries.enumerated() {
            try captureCrossAttentionInputsIfRequested(
                query: query,
                data: crossData,
                jointOrdinal: jointOrdinal,
                chunkOrdinal: chunkOrdinal
            )
            let chunk = try autoreleasepool {
                try applyPreparedCross(
                    query: query,
                    data: crossData,
                    workspace: workspace
                )
            }
            probabilities.append(contentsOf: chunk)
        }
        return probabilities
    }

    /// Decodes multiple joints and returns `[vertex, joint]` row-major weights.
  /// Prepared queries and GPU output storage are shared, but each joint must
  /// complete before its scratch workspace is reused. This preserves the
  /// per-joint bits produced by `decode` across long asynchronous workloads.
    public func decodeJoints(
        indicesByJoint: [[UInt32]],
        conditionTokens: [Float],
        pointNormals: [Float]
    ) throws -> SmeltSkinWeights {
    let field = try decodeJointField(
      indicesByJoint: indicesByJoint,
      conditionTokens: conditionTokens,
      pointNormals: pointNormals
    )
    let probabilities = read(
      field.jointMajorWeights,
      count: field.vertexCount * field.jointCount
    )
    var values = [Float](repeating: 0, count: probabilities.count)
    for joint in 0..<field.jointCount {
      for vertex in 0..<field.vertexCount {
        values[vertex * field.jointCount + joint] =
          probabilities[joint * field.vertexCount + vertex]
      }
    }
    return SmeltSkinWeights(
      vertexCount: field.vertexCount,
      jointCount: field.jointCount,
      values: values
    )
  }

  func decodeJointField(
    indicesByJoint: [[UInt32]],
    conditionTokens: [Float],
    pointNormals: [Float]
  ) throws -> SmeltGPUSkinField {
        guard !indicesByJoint.isEmpty else {
            throw SmeltSkinDecoderError.invalidJointCount(0)
        }
    guard indicesByJoint.allSatisfy({ $0.count == 4 }) else {
      throw SmeltSkinDecoderError.invalidIndices
    }
    guard conditionTokens.count == 384 * Self.latentWidth else {
      throw SmeltSkinDecoderError.invalidConditionCount(conditionTokens.count)
    }
        try validatePointNormals(pointNormals)
        let vertexCount = pointNormals.count / 6
        let jointCount = indicesByJoint.count
        let queries = try prepareCrossQueries(pointNormals: pointNormals)
    let tokens = conditionTokens.count / Self.latentWidth + 4
    let workspace = try makeJointWorkspace(
      tokens: tokens,
      maximumQueryRows: queries.map(\.rows).max() ?? 0,
      conditionTokens: conditionTokens
    )
    let jointMajor = try buffer(
      count: vertexCount * jointCount,
      label: "skinning.vae.decoder.joint-major-probabilities"
        )
        for (joint, indices) in indicesByJoint.enumerated() {
      let (commandBuffer, encoder) = try commandBufferAndEncoder()
      try encodeJoint(
        encoder,
                    indices: indices,
                    preparedQueries: queries,
                    workspace: workspace,
        output: jointMajor,
        outputRowOffset: joint * vertexCount,
                    jointOrdinal: joint
                )
      encoder.endEncoding()
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
      if let error = commandBuffer.error {
        throw SmeltSkinDecoderError.gpuExecutionFailed("\(error)")
            }
        }
    return SmeltGPUSkinField(
            vertexCount: vertexCount,
            jointCount: jointCount,
      jointMajorWeights: jointMajor
    )
  }

  private func makeJointWorkspace(
    tokens: Int,
    maximumQueryRows: Int,
    conditionTokens: [Float]
  ) throws -> JointWorkspace {
    var combined = [Float](repeating: 0, count: 4 * Self.latentWidth)
    combined.append(contentsOf: conditionTokens)
    return try JointWorkspace(
      levels: buffer(count: 4 * 5, label: "skinning.vae.decoder.fsq-levels"),
      combined: buffer(combined, label: "skinning.vae.decoder.combined-latents"),
      selfBlock: makeSelfBlockWorkspace(tokens: tokens),
      crossData: makeCrossDataWorkspace(tokens: tokens),
      cross: makeCrossWorkspace(rows: maximumQueryRows)
    )
  }

  private func encodeJoint(
    _ encoder: MTLComputeCommandEncoder,
    indices: [UInt32],
    preparedQueries: [PreparedCrossQuery],
    workspace: JointWorkspace,
    output: MTLBuffer,
    outputRowOffset: Int,
    jointOrdinal _: Int
  ) throws {
    guard indices.count == 4, indices.allSatisfy({ $0 < 32_768 }) else {
      throw SmeltSkinDecoderError.invalidIndices
    }
    let tokens = 388
    encoder.setComputePipelineState(try pipeline("fsq_base8x5_decode_f32"))
    indices.withUnsafeBytes { bytes in
      if let baseAddress = bytes.baseAddress {
        encoder.setBytes(baseAddress, length: bytes.count, index: 0)
      }
    }
    encoder.setBuffer(workspace.levels, offset: 0, index: 1)
    var indexCount = UInt32(indices.count)
    encoder.setBytes(&indexCount, length: 4, index: 2)
    encoder.dispatchThreads(
      MTLSize(width: indices.count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: indices.count, height: 1, depth: 1)
    )
    try encodeDense(
      encoder,
      input: workspace.levels,
      output: workspace.combined,
      rows: indices.count,
      inputDimension: 5,
      outputDimension: Self.latentWidth,
      weight: "vae.model.FSQ.project_out.weight",
      bias: "vae.model.FSQ.project_out.bias"
    )

    var hidden = workspace.selfBlock.outputA
    try encodeDense(
      encoder,
      input: workspace.combined,
      output: hidden,
      rows: tokens,
      inputDimension: Self.latentWidth,
      outputDimension: Self.width,
      weight: "vae.model.post_quant.weight",
      bias: "vae.model.post_quant.bias"
        )
    for layer in 0..<10 {
      let next =
        layer.isMultiple(of: 2)
        ? workspace.selfBlock.outputB
        : workspace.selfBlock.outputA
      try encodeTransformerSelfBlock(
        encoder,
        input: hidden,
        output: next,
        tokens: tokens,
        prefix: "\(Self.decoderPrefix).blocks.\(layer)",
        workspace: workspace.selfBlock
      )
      hidden = next
    }
    let crossData = try encodePreparedCrossData(
      encoder,
      input: hidden,
      tokens: tokens,
      workspace: workspace.crossData
    )
    for query in preparedQueries {
      try encodePreparedCross(
        encoder,
        query: query,
        data: crossData,
        workspace: workspace.cross,
        probabilities: output,
        probabilityRowOffset: outputRowOffset + query.rowOffset
      )
    }
    }

    private func captureCrossAttentionInputsIfRequested(
        query: PreparedCrossQuery,
        data: PreparedCrossData,
        jointOrdinal: Int?,
        chunkOrdinal: Int
    ) throws {
        guard let request = attentionCaptureRequest,
            jointOrdinal == request.jointOrdinal,
            chunkOrdinal == request.chunkOrdinal
        else {
            return
        }
        try FileManager.default.createDirectory(
            at: request.root,
            withIntermediateDirectories: true
        )
        try writeRawFloats(
            read(query.rawQuery, count: query.rows * Self.width),
            to: request.root.appendingPathComponent("q.f32")
        )
        try writeRawFloats(
            read(data.key, count: data.rows * Self.width),
            to: request.root.appendingPathComponent("k.f32")
        )
        try writeRawFloats(
            read(data.value, count: data.rows * Self.width),
            to: request.root.appendingPathComponent("v.f32")
        )
        let metadata: [String: Int] = [
            "queryTokens": query.rows,
            "keyValueTokens": data.rows,
            "heads": Self.heads,
            "headDimension": Self.headDimension,
            "jointOrdinal": request.jointOrdinal,
            "chunkOrdinal": request.chunkOrdinal,
        ]
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(metadata).write(
            to: request.root.appendingPathComponent("metadata.json"),
            options: .atomic
        )
    }

    private func writeRawFloats(_ values: [Float], to url: URL) throws {
        try values.withUnsafeBytes { bytes in
            try Data(bytes).write(to: url, options: .atomic)
        }
    }

    private static func makeAttentionCaptureRequest() -> AttentionCaptureRequest? {
        let environment = ProcessInfo.processInfo.environment
    guard let directory = environment["SMELT_SKINNING_ATTENTION_CAPTURE"],
            !directory.isEmpty,
      let jointText = environment["SMELT_SKINNING_ATTENTION_CAPTURE_JOINT"],
            let joint = Int(jointText),
      let chunkText = environment["SMELT_SKINNING_ATTENTION_CAPTURE_CHUNK"],
            let chunk = Int(chunkText)
        else {
            return nil
        }
        return AttentionCaptureRequest(
            root: URL(fileURLWithPath: directory, isDirectory: true),
            jointOrdinal: joint,
            chunkOrdinal: chunk
        )
    }

    private func prepareLatents(
        indices: [UInt32],
        conditionTokens: [Float]
    ) throws -> (
        fsqCodes: [Float],
        postQuantOutput: [Float],
        selfLayerOutputs: [[Float]],
        cache: [Float]
    ) {
        guard !indices.isEmpty, indices.allSatisfy({ $0 < 32_768 }) else {
            throw SmeltSkinDecoderError.invalidIndices
        }
        guard !conditionTokens.isEmpty,
            conditionTokens.count.isMultiple(of: Self.latentWidth),
            conditionTokens.allSatisfy(\.isFinite)
        else {
            throw SmeltSkinDecoderError.invalidConditionCount(
                conditionTokens.count
            )
        }
        let fsqCodes = try decodeFSQ(indices: indices)
        var combined = fsqCodes
        combined.append(contentsOf: conditionTokens)
        var hidden = try dense(
            input: combined,
            rows: combined.count / Self.latentWidth,
            inputDimension: Self.latentWidth,
            outputDimension: Self.width,
            weight: "vae.model.post_quant.weight",
            bias: "vae.model.post_quant.bias",
      label: "skinning.vae.decoder.post-quant"
        )
        let postQuantOutput = hidden
        var layers: [[Float]] = []
        layers.reserveCapacity(10)
        for layer in 0..<10 {
            hidden = try transformerBlock(
                input: hidden,
                data: nil,
                prefix: "\(Self.decoderPrefix).blocks.\(layer)"
            )
            layers.append(hidden)
        }
        return (fsqCodes, postQuantOutput, layers, hidden)
    }

    private func prepareLatentCache(
        indices: [UInt32],
        conditionTokens: [Float],
        workspace: SelfBlockWorkspace
    ) throws -> PreparedLatentCache {
        guard !indices.isEmpty, indices.allSatisfy({ $0 < 32_768 }) else {
            throw SmeltSkinDecoderError.invalidIndices
        }
        guard !conditionTokens.isEmpty,
            conditionTokens.count.isMultiple(of: Self.latentWidth),
            conditionTokens.allSatisfy(\.isFinite)
        else {
            throw SmeltSkinDecoderError.invalidConditionCount(
                conditionTokens.count
            )
        }
        let fsqCodes = try decodeFSQ(indices: indices)
        var combined = fsqCodes
        combined.append(contentsOf: conditionTokens)
        let tokens = combined.count / Self.latentWidth
        let combinedBuffer = try buffer(
            combined,
      label: "skinning.vae.decoder.post-quant.input"
        )
        var hidden = workspace.outputA
        let (commandBuffer, encoder) = try commandBufferAndEncoder()
        try encodeDense(
            encoder,
            input: combinedBuffer,
            output: hidden,
            rows: tokens,
            inputDimension: Self.latentWidth,
            outputDimension: Self.width,
            weight: "vae.model.post_quant.weight",
            bias: "vae.model.post_quant.bias"
        )
        try finish(commandBuffer: commandBuffer, encoder: encoder)
        for layer in 0..<10 {
            let output =
                layer.isMultiple(of: 2)
                ? workspace.outputB
                : workspace.outputA
            hidden = try transformerSelfBlock(
                input: hidden,
                output: output,
                tokens: tokens,
                prefix: "\(Self.decoderPrefix).blocks.\(layer)",
                workspace: workspace
            )
        }
        return PreparedLatentCache(buffer: hidden, rows: tokens)
    }

    private func decodeFSQ(indices: [UInt32]) throws -> [Float] {
    let input = try uintBuffer(indices, label: "skinning.vae.decoder.fsq-indices")
        let levels = try buffer(
            count: indices.count * 5,
      label: "skinning.vae.decoder.fsq-levels"
        )
        let output = try buffer(
            count: indices.count * Self.latentWidth,
      label: "skinning.vae.decoder.fsq-codes"
        )
        let (commandBuffer, encoder) = try commandBufferAndEncoder()
        encoder.setComputePipelineState(try pipeline("fsq_base8x5_decode_f32"))
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(levels, offset: 0, index: 1)
        var count = UInt32(indices.count)
        encoder.setBytes(&count, length: 4, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: indices.count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(indices.count, 256), height: 1, depth: 1)
        )
        try encodeDense(
            encoder,
            input: levels,
            output: output,
            rows: indices.count,
            inputDimension: 5,
            outputDimension: Self.latentWidth,
            weight: "vae.model.FSQ.project_out.weight",
            bias: "vae.model.FSQ.project_out.bias"
        )
        try finish(commandBuffer: commandBuffer, encoder: encoder)
        return read(output, count: indices.count * Self.latentWidth)
    }

    private func projectQueries(pointNormals: [Float]) throws -> [Float] {
        try validatePointNormals(pointNormals)
        let rows = pointNormals.count / 6
        // Upstream explicitly casts sampled decoder queries to `z.dtype`
        // (BF16) before PMPE. Preserve that orchestration boundary exactly,
        // while widening the rounded values for the FP32 Metal route.
        let roundedPointNormals = pointNormals.map(Self.widenedBF16)
        let input = try buffer(
            roundedPointNormals,
      label: "skinning.vae.decoder.point-normals"
        )
    let fourier = try buffer(count: rows * 51, label: "skinning.vae.decoder.pmpe")
    let features = try buffer(count: rows * 54, label: "skinning.vae.decoder.features")
        let projected = try buffer(
            count: rows * Self.width,
      label: "skinning.vae.decoder.projected-queries"
        )
        let (commandBuffer, encoder) = try commandBufferAndEncoder()
        encoder.setComputePipelineState(try pipeline("pmpe_bf16_semantics_f32"))
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(fourier, offset: 0, index: 1)
        var rowCount = UInt32(rows)
        var positionDimensions: UInt32 = 3
        var frequencies: UInt32 = 8
        var includeInput: UInt32 = 1
        var includePi: UInt32 = 1
        var usePMPE: UInt32 = 1
        var inputStride: UInt32 = 6
        encoder.setBytes(&rowCount, length: 4, index: 2)
        encoder.setBytes(&positionDimensions, length: 4, index: 3)
        encoder.setBytes(&frequencies, length: 4, index: 4)
        encoder.setBytes(&includeInput, length: 4, index: 5)
        encoder.setBytes(&includePi, length: 4, index: 6)
        encoder.setBytes(&usePMPE, length: 4, index: 7)
        encoder.setBytes(&inputStride, length: 4, index: 8)
        encoder.dispatchThreads(
            MTLSize(width: 51, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
        )

        encoder.setComputePipelineState(try pipeline("append_strided_features_f32"))
        encoder.setBuffer(fourier, offset: 0, index: 0)
        encoder.setBuffer(input, offset: 0, index: 1)
        encoder.setBuffer(features, offset: 0, index: 2)
        var baseDimension: UInt32 = 51
        var featureStride: UInt32 = 6
        var featureOffset: UInt32 = 3
        var featureCount: UInt32 = 3
        encoder.setBytes(&rowCount, length: 4, index: 3)
        encoder.setBytes(&baseDimension, length: 4, index: 4)
        encoder.setBytes(&featureStride, length: 4, index: 5)
        encoder.setBytes(&featureOffset, length: 4, index: 6)
        encoder.setBytes(&featureCount, length: 4, index: 7)
        encoder.dispatchThreads(
            MTLSize(width: 54, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
        )
        try encodeDense(
            encoder,
            input: features,
            output: projected,
            rows: rows,
            inputDimension: 54,
            outputDimension: Self.width,
            weight: "\(Self.decoderPrefix).proj_query.weight",
            bias: "\(Self.decoderPrefix).proj_query.bias"
        )
        try finish(commandBuffer: commandBuffer, encoder: encoder)
        return read(projected, count: rows * Self.width)
    }

    private func prepareCrossQueries(
        pointNormals: [Float]
    ) throws -> [PreparedCrossQuery] {
        try validatePointNormals(pointNormals)
        let totalRows = pointNormals.count / 6
        var result: [PreparedCrossQuery] = []
        result.reserveCapacity(
            (totalRows + Self.fullQueryChunk - 1) / Self.fullQueryChunk
        )
        var start = 0
        while start < totalRows {
            let count = min(Self.fullQueryChunk, totalRows - start)
            let pointStart = start * 6
            let pointEnd = pointStart + count * 6
            result.append(
                try autoreleasepool {
                    let source = try projectQueries(
                        pointNormals: Array(pointNormals[pointStart..<pointEnd])
                    )
                    let normalized = try layerNorm(
                        input: source,
                        rows: count,
                        epsilon: 1e-5,
                        weight: "\(Self.decoderPrefix).blocks.10.norm2.weight",
                        bias: "\(Self.decoderPrefix).blocks.10.norm2.bias",
            label: "skinning.vae.decoder.cross-query-norm"
                    )
                    let rawQuery = try dense(
                        input: normalized,
                        rows: count,
                        inputDimension: Self.width,
                        outputDimension: Self.width,
                        weight: "\(Self.decoderPrefix).blocks.10.attn2.to_q.weight",
                        bias: nil,
            label: "skinning.vae.decoder.cross-query"
                    )
                    return PreparedCrossQuery(
                        source: try buffer(
                            source,
              label: "skinning.vae.decoder.prepared-cross-source"
                        ),
                        rawQuery: try buffer(
                            rawQuery,
              label: "skinning.vae.decoder.prepared-cross-query"
                        ),
            rows: count,
            rowOffset: start
                    )
                }
            )
            start += count
        }
        return result
    }

    private func prepareCrossData(
        input: MTLBuffer,
        tokens: Int
    ) throws -> PreparedCrossData {
        let count = tokens * Self.width
        guard tokens > 0,
            input.length >= count * MemoryLayout<Float>.stride
        else {
            throw SmeltSkinDecoderError.invalidDataCount(count)
        }
    let workspace = try makeCrossDataWorkspace(tokens: tokens)
        let (commandBuffer, encoder) = try commandBufferAndEncoder()
    let result = try encodePreparedCrossData(
      encoder,
      input: input,
      tokens: tokens,
      workspace: workspace
    )
    try finish(commandBuffer: commandBuffer, encoder: encoder)
    return result
  }

  private func makeCrossDataWorkspace(tokens: Int) throws -> CrossDataWorkspace {
    let count = tokens * Self.width
    return try CrossDataWorkspace(
      normalized: buffer(count: count, label: "skinning.vae.decoder.cross-data-norm"),
      rawKey: buffer(count: count, label: "skinning.vae.decoder.raw-k"),
      rawValue: buffer(count: count, label: "skinning.vae.decoder.raw-v"),
      key: buffer(count: count, label: "skinning.vae.decoder.k"),
      value: buffer(count: count, label: "skinning.vae.decoder.v"),
      unused: buffer(count: count, label: "skinning.vae.decoder.unused")
    )
  }

  private func encodePreparedCrossData(
    _ encoder: MTLComputeCommandEncoder,
    input: MTLBuffer,
    tokens: Int,
    workspace: CrossDataWorkspace
  ) throws -> PreparedCrossData {
        try encodeLayerNorm(
            encoder,
            input: input,
      output: workspace.normalized,
            rows: tokens,
            epsilon: 1e-6,
            weight: "\(Self.decoderPrefix).blocks.10.attn2.norm_cross.weight",
            bias: "\(Self.decoderPrefix).blocks.10.attn2.norm_cross.bias"
        )
        try encodeDense(
            encoder,
      input: workspace.normalized,
      output: workspace.rawKey,
            rows: tokens,
            inputDimension: Self.width,
            outputDimension: Self.width,
            weight: "\(Self.decoderPrefix).blocks.10.attn2.to_k.weight",
            bias: nil
        )
        try encodeDense(
            encoder,
      input: workspace.normalized,
      output: workspace.rawValue,
            rows: tokens,
            inputDimension: Self.width,
            outputDimension: Self.width,
            weight: "\(Self.decoderPrefix).blocks.10.attn2.to_v.weight",
            bias: nil
        )
        try encodeRepack(
            encoder,
      inputs: (workspace.rawKey, workspace.rawValue, workspace.rawValue),
      outputs: (workspace.key, workspace.value, workspace.unused),
            tokens: tokens,
            parts: 2
        )
        return PreparedCrossData(
      key: workspace.key,
      value: workspace.value,
            rows: tokens
        )
    }

    private func applyPreparedCross(
        query: PreparedCrossQuery,
        data: PreparedCrossData,
        workspace: CrossWorkspace
    ) throws -> [Float] {
    let (commandBuffer, encoder) = try commandBufferAndEncoder()
    try encodePreparedCross(
      encoder,
      query: query,
      data: data,
      workspace: workspace,
      probabilities: workspace.probabilities,
      probabilityRowOffset: 0
    )
    try finish(commandBuffer: commandBuffer, encoder: encoder)
    return read(workspace.probabilities, count: query.rows)
  }

  private func encodePreparedCross(
    _ encoder: MTLComputeCommandEncoder,
    query: PreparedCrossQuery,
    data: PreparedCrossData,
    workspace: CrossWorkspace,
    probabilities: MTLBuffer,
    probabilityRowOffset: Int
  ) throws {
        let inputCount = query.rows * Self.width
        let dataCount = data.rows * Self.width
        guard query.rows > 0,
            query.source.length >= inputCount * MemoryLayout<Float>.stride,
            query.rawQuery.length >= inputCount * MemoryLayout<Float>.stride
        else {
            throw SmeltSkinDecoderError.invalidInputCount(inputCount)
        }
        guard data.rows > 0,
            data.key.length >= dataCount * MemoryLayout<Float>.stride,
            data.value.length >= dataCount * MemoryLayout<Float>.stride
        else {
            throw SmeltSkinDecoderError.invalidDataCount(dataCount)
        }
        let tokens = query.rows
        let dataTokens = data.rows
        guard workspace.attended.length >= inputCount * MemoryLayout<Float>.stride,
            workspace.expandedMLP.length
                >= tokens * Self.mlpWidth * MemoryLayout<Float>.stride,
      probabilities.length
        >= (probabilityRowOffset + tokens) * MemoryLayout<Float>.stride
        else {
            throw SmeltSkinDecoderError.invalidInputCount(inputCount)
        }
        try encodeAttention(
            encoder,
            query: query.rawQuery,
            key: data.key,
            value: data.value,
            output: workspace.attended,
            queryTokens: tokens,
            dataTokens: dataTokens,
            queryTile: kernelPolicy.crossAttentionQueryTile
        )
        let attentionPrefix = "\(Self.decoderPrefix).blocks.10.attn2"
        try encodeDenseEpilogue(
            encoder,
            input: workspace.attended,
            denseOutput: workspace.projectedAttention,
            output: workspace.attentionResidual,
            residual: query.source,
            rows: tokens,
            inputDimension: Self.width,
            outputDimension: Self.width,
            weight: "\(attentionPrefix).to_out.0.weight",
            bias: "\(attentionPrefix).to_out.0.bias",
            epilogue: .residualAdd
        )
        try encodeLayerNorm(
            encoder,
            input: workspace.attentionResidual,
            output: workspace.normalizedMLP,
            rows: tokens,
            epsilon: 1e-5,
            weight: "\(Self.decoderPrefix).blocks.10.norm3.weight",
            bias: "\(Self.decoderPrefix).blocks.10.norm3.bias"
        )
        try encodeDenseEpilogue(
            encoder,
            input: workspace.normalizedMLP,
            denseOutput: workspace.expandedMLP,
            output: workspace.activatedMLP,
            residual: nil,
            rows: tokens,
            inputDimension: Self.width,
            outputDimension: Self.mlpWidth,
            weight: "\(Self.decoderPrefix).blocks.10.ff.net.0.proj.weight",
            bias: "\(Self.decoderPrefix).blocks.10.ff.net.0.proj.bias",
            epilogue: .gelu
        )
        try encodeDenseEpilogue(
            encoder,
            input: workspace.activatedMLP,
            denseOutput: workspace.projectedMLP,
            output: workspace.output,
            residual: workspace.attentionResidual,
            rows: tokens,
            inputDimension: Self.mlpWidth,
            outputDimension: Self.width,
            weight: "\(Self.decoderPrefix).blocks.10.ff.net.2.weight",
            bias: "\(Self.decoderPrefix).blocks.10.ff.net.2.bias",
            epilogue: .residualAdd
        )
        try encodeLayerNorm(
            encoder,
            input: workspace.output,
            output: workspace.normalizedOutput,
            rows: tokens,
            epsilon: 1e-5,
            weight: "\(Self.decoderPrefix).norm_out.weight",
            bias: "\(Self.decoderPrefix).norm_out.bias"
        )
        try encodeDense(
            encoder,
            input: workspace.normalizedOutput,
            output: workspace.rawLogits,
            rows: tokens,
            inputDimension: Self.width,
            outputDimension: 1,
            weight: "\(Self.decoderPrefix).proj_out.weight",
            bias: "\(Self.decoderPrefix).proj_out.bias"
        )
        encoder.setComputePipelineState(try pipeline("sigmoid_f32"))
        encoder.setBuffer(workspace.rawLogits, offset: 0, index: 0)
    encoder.setBuffer(
      probabilities,
      offset: probabilityRowOffset * MemoryLayout<Float>.stride,
      index: 1
    )
        var probabilityCount = UInt32(tokens)
        encoder.setBytes(&probabilityCount, length: 4, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: tokens, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(tokens, 256),
                height: 1,
                depth: 1
            )
        )
    }

    private func makeCrossWorkspace(rows: Int) throws -> CrossWorkspace {
        guard rows > 0 else {
            throw SmeltSkinDecoderError.invalidInputCount(0)
        }
        let hiddenCount = rows * Self.width
        let expandedCount = rows * Self.mlpWidth
        return try CrossWorkspace(
            attended: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.attended"
            ),
            projectedAttention: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.attention-projection"
            ),
            attentionResidual: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.attention-residual"
            ),
            normalizedMLP: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.norm-mlp"
            ),
            expandedMLP: buffer(
                count: expandedCount,
        label: "skinning.vae.decoder.mlp-expanded"
            ),
            activatedMLP: buffer(
                count: expandedCount,
        label: "skinning.vae.decoder.mlp-gelu"
            ),
            projectedMLP: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.mlp-projection"
            ),
            output: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.block.output"
            ),
            normalizedOutput: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.finish-norm"
            ),
            rawLogits: buffer(
                count: rows,
        label: "skinning.vae.decoder.raw-logits"
            ),
            probabilities: buffer(
                count: rows,
        label: "skinning.vae.decoder.probabilities"
            )
        )
    }

    private func makeSelfBlockWorkspace(
        tokens: Int
    ) throws -> SelfBlockWorkspace {
        guard tokens > 0 else {
            throw SmeltSkinDecoderError.invalidInputCount(0)
        }
        let hiddenCount = tokens * Self.width
        let expandedCount = tokens * Self.mlpWidth
        return try SelfBlockWorkspace(
            normalizedQuery: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.norm-q"
            ),
            rawQuery: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.raw-q"
            ),
            rawKey: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.raw-k"
            ),
            rawValue: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.raw-v"
            ),
            query: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.q"
            ),
            key: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.k"
            ),
            value: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.v"
            ),
            attended: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.attended"
            ),
            projectedAttention: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.attention-projection"
            ),
            attentionResidual: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.attention-residual"
            ),
            normalizedMLP: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.norm-mlp"
            ),
            expandedMLP: buffer(
                count: expandedCount,
        label: "skinning.vae.decoder.mlp-expanded"
            ),
            activatedMLP: buffer(
                count: expandedCount,
        label: "skinning.vae.decoder.mlp-gelu"
            ),
            projectedMLP: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.mlp-projection"
            ),
            outputA: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.block.output-a"
            ),
            outputB: buffer(
                count: hiddenCount,
        label: "skinning.vae.decoder.block.output-b"
            )
        )
    }

    private func layerNorm(
        input: [Float],
        rows: Int,
        epsilon: Float,
        weight: String,
        bias: String,
        label: String
    ) throws -> [Float] {
        let source = try buffer(input, label: "\(label).input")
        let output = try buffer(count: input.count, label: "\(label).output")
        let (commandBuffer, encoder) = try commandBufferAndEncoder()
        try encodeLayerNorm(
            encoder,
            input: source,
            output: output,
            rows: rows,
            epsilon: epsilon,
            weight: weight,
            bias: bias
        )
        try finish(commandBuffer: commandBuffer, encoder: encoder)
        return read(output, count: input.count)
    }

    private func transformerSelfBlock(
        input: MTLBuffer,
        output: MTLBuffer,
        tokens: Int,
        prefix: String,
        workspace: SelfBlockWorkspace
    ) throws -> MTLBuffer {
    let (commandBuffer, encoder) = try commandBufferAndEncoder()
    try encodeTransformerSelfBlock(
      encoder,
      input: input,
      output: output,
      tokens: tokens,
      prefix: prefix,
      workspace: workspace
    )
    try finish(commandBuffer: commandBuffer, encoder: encoder)
    return output
  }

  private func encodeTransformerSelfBlock(
    _ encoder: MTLComputeCommandEncoder,
    input: MTLBuffer,
    output: MTLBuffer,
    tokens: Int,
    prefix: String,
    workspace: SelfBlockWorkspace
  ) throws {
        let count = tokens * Self.width
        guard tokens > 0,
            input.length >= count * MemoryLayout<Float>.stride
        else {
            throw SmeltSkinDecoderError.invalidInputCount(count)
        }
        guard
            workspace.normalizedQuery.length
                >= count * MemoryLayout<Float>.stride,
            workspace.expandedMLP.length
                >= tokens * Self.mlpWidth * MemoryLayout<Float>.stride,
            output.length >= count * MemoryLayout<Float>.stride
        else {
            throw SmeltSkinDecoderError.invalidInputCount(count)
        }
        let normalizedQuery = workspace.normalizedQuery
        let rawQuery = workspace.rawQuery
        let rawKey = workspace.rawKey
        let rawValue = workspace.rawValue
        let query = workspace.query
        let key = workspace.key
        let value = workspace.value
        let attended = workspace.attended
        let projectedAttention = workspace.projectedAttention
        let attentionResidual = workspace.attentionResidual
        let normalizedMLP = workspace.normalizedMLP
        let expandedMLP = workspace.expandedMLP
        let activatedMLP = workspace.activatedMLP
        let projectedMLP = workspace.projectedMLP
        try encodeLayerNorm(
            encoder,
            input: input,
            output: normalizedQuery,
            rows: tokens,
            epsilon: 1e-5,
            weight: "\(prefix).norm1.weight",
            bias: "\(prefix).norm1.bias"
        )
        let attentionPrefix = "\(prefix).attn1"
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
        try encodeDense(
            encoder,
            input: normalizedQuery,
            output: rawKey,
            rows: tokens,
            inputDimension: Self.width,
            outputDimension: Self.width,
            weight: "\(attentionPrefix).to_k.weight",
            bias: nil
        )
        try encodeDense(
            encoder,
            input: normalizedQuery,
            output: rawValue,
            rows: tokens,
            inputDimension: Self.width,
            outputDimension: Self.width,
            weight: "\(attentionPrefix).to_v.weight",
            bias: nil
        )
        try encodeRepack(
            encoder,
            inputs: (rawQuery, rawKey, rawValue),
            outputs: (query, key, value),
            tokens: tokens,
            parts: 3
        )
        try encodeAttention(
            encoder,
            query: query,
            key: key,
            value: value,
            output: attended,
            queryTokens: tokens,
            dataTokens: tokens,
            queryTile: kernelPolicy.selfAttentionQueryTile
        )
        try encodeDenseEpilogue(
            encoder,
            input: attended,
            denseOutput: projectedAttention,
            output: attentionResidual,
            residual: input,
            rows: tokens,
            inputDimension: Self.width,
            outputDimension: Self.width,
            weight: "\(attentionPrefix).to_out.0.weight",
            bias: "\(attentionPrefix).to_out.0.bias",
            epilogue: .residualAdd
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
        try encodeDenseEpilogue(
            encoder,
            input: normalizedMLP,
            denseOutput: expandedMLP,
            output: activatedMLP,
            residual: nil,
            rows: tokens,
            inputDimension: Self.width,
            outputDimension: Self.mlpWidth,
            weight: "\(prefix).ff.net.0.proj.weight",
            bias: "\(prefix).ff.net.0.proj.bias",
            epilogue: .gelu
        )
        try encodeDenseEpilogue(
            encoder,
            input: activatedMLP,
            denseOutput: projectedMLP,
            output: output,
            residual: attentionResidual,
            rows: tokens,
            inputDimension: Self.mlpWidth,
            outputDimension: Self.width,
            weight: "\(prefix).ff.net.2.weight",
            bias: "\(prefix).ff.net.2.bias",
            epilogue: .residualAdd
        )
    }

    private func transformerBlock(
        input: [Float],
        data: [Float]?,
        prefix: String
    ) throws -> [Float] {
        guard !input.isEmpty,
            input.count.isMultiple(of: Self.width),
            input.allSatisfy(\.isFinite)
        else {
            throw SmeltSkinDecoderError.invalidInputCount(input.count)
        }
        if let data {
            guard !data.isEmpty,
                data.count.isMultiple(of: Self.width),
                data.allSatisfy(\.isFinite)
            else {
                throw SmeltSkinDecoderError.invalidDataCount(data.count)
            }
        }
        let cross = data != nil
        let tokens = input.count / Self.width
        let dataTokens = (data?.count ?? input.count) / Self.width
    let source = try buffer(input, label: "skinning.vae.decoder.block.input")
        let attentionData =
            try data.map {
        try buffer($0, label: "skinning.vae.decoder.block.data")
            } ?? source
    let normalizedQuery = try buffer(count: input.count, label: "skinning.vae.decoder.norm-q")
        let normalizedData = try buffer(
            count: dataTokens * Self.width,
      label: "skinning.vae.decoder.norm-data"
    )
    let rawQuery = try buffer(count: input.count, label: "skinning.vae.decoder.raw-q")
    let rawKey = try buffer(count: dataTokens * Self.width, label: "skinning.vae.decoder.raw-k")
    let rawValue = try buffer(count: dataTokens * Self.width, label: "skinning.vae.decoder.raw-v")
    let query = try buffer(count: input.count, label: "skinning.vae.decoder.q")
    let key = try buffer(count: dataTokens * Self.width, label: "skinning.vae.decoder.k")
    let value = try buffer(count: dataTokens * Self.width, label: "skinning.vae.decoder.v")
        let unused = try buffer(
            count: max(input.count, dataTokens * Self.width),
      label: "skinning.vae.decoder.unused"
        )
    let attended = try buffer(count: input.count, label: "skinning.vae.decoder.attended")
        let projectedAttention = try buffer(
            count: input.count,
      label: "skinning.vae.decoder.attention-projection"
        )
        let attentionResidual = try buffer(
            count: input.count,
      label: "skinning.vae.decoder.attention-residual"
        )
    let normalizedMLP = try buffer(count: input.count, label: "skinning.vae.decoder.norm-mlp")
        let expandedMLP = try buffer(
            count: tokens * Self.mlpWidth,
      label: "skinning.vae.decoder.mlp-expanded"
        )
        let activatedMLP = try buffer(
            count: tokens * Self.mlpWidth,
      label: "skinning.vae.decoder.mlp-gelu"
        )
    let projectedMLP = try buffer(count: input.count, label: "skinning.vae.decoder.mlp-projection")
    let output = try buffer(count: input.count, label: "skinning.vae.decoder.block.output")
        let (commandBuffer, encoder) = try commandBufferAndEncoder()

        try encodeLayerNorm(
            encoder,
            input: source,
            output: normalizedQuery,
            rows: tokens,
            epsilon: 1e-5,
            weight: "\(prefix).\(cross ? "norm2" : "norm1").weight",
            bias: "\(prefix).\(cross ? "norm2" : "norm1").bias"
        )
        if cross {
            try encodeLayerNorm(
                encoder,
                input: attentionData,
                output: normalizedData,
                rows: dataTokens,
                epsilon: 1e-6,
                weight: "\(prefix).attn2.norm_cross.weight",
                bias: "\(prefix).attn2.norm_cross.bias"
            )
        }
        let attentionPrefix = "\(prefix).\(cross ? "attn2" : "attn1")"
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
        let keyValueInput = cross ? normalizedData : normalizedQuery
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
        if cross {
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
        try encodeAttention(
            encoder,
            query: cross ? rawQuery : query,
            key: key,
            value: value,
            output: attended,
            queryTokens: tokens,
            dataTokens: dataTokens,
            queryTile: cross
                ? kernelPolicy.crossAttentionQueryTile
                : kernelPolicy.selfAttentionQueryTile
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
        try finish(commandBuffer: commandBuffer, encoder: encoder)
        return read(output, count: input.count)
    }

    private func finish(input: [Float]) throws -> (
        normalized: [Float],
        rawLogits: [Float],
        probabilities: [Float]
    ) {
        let tokens = input.count / Self.width
    let source = try buffer(input, label: "skinning.vae.decoder.finish-input")
    let normalized = try buffer(count: input.count, label: "skinning.vae.decoder.finish-norm")
    let raw = try buffer(count: tokens, label: "skinning.vae.decoder.raw-logits")
    let probabilities = try buffer(count: tokens, label: "skinning.vae.decoder.probabilities")
        let (commandBuffer, encoder) = try commandBufferAndEncoder()
        try encodeLayerNorm(
            encoder,
            input: source,
            output: normalized,
            rows: tokens,
            epsilon: 1e-5,
            weight: "\(Self.decoderPrefix).norm_out.weight",
            bias: "\(Self.decoderPrefix).norm_out.bias"
        )
        try encodeDense(
            encoder,
            input: normalized,
            output: raw,
            rows: tokens,
            inputDimension: Self.width,
            outputDimension: 1,
            weight: "\(Self.decoderPrefix).proj_out.weight",
            bias: "\(Self.decoderPrefix).proj_out.bias"
        )
        encoder.setComputePipelineState(try pipeline("sigmoid_f32"))
        encoder.setBuffer(raw, offset: 0, index: 0)
        encoder.setBuffer(probabilities, offset: 0, index: 1)
        var count = UInt32(tokens)
        encoder.setBytes(&count, length: 4, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: tokens, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(tokens, 256), height: 1, depth: 1)
        )
        try finish(commandBuffer: commandBuffer, encoder: encoder)
        return (
            read(normalized, count: input.count),
            read(raw, count: tokens),
            read(probabilities, count: tokens)
        )
    }

    private func dense(
        input: [Float],
        rows: Int,
        inputDimension: Int,
        outputDimension: Int,
        weight: String,
        bias: String?,
        label: String
    ) throws -> [Float] {
        let source = try buffer(input, label: "\(label).input")
        let output = try buffer(count: rows * outputDimension, label: "\(label).output")
        let (commandBuffer, encoder) = try commandBufferAndEncoder()
        try encodeDense(
            encoder,
            input: source,
            output: output,
            rows: rows,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            weight: weight,
            bias: bias
        )
        try finish(commandBuffer: commandBuffer, encoder: encoder)
        return read(output, count: rows * outputDimension)
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
        encoder.setBuffer(try weightBuffer(weight), offset: 0, index: 1)
        encoder.setBuffer(try weightBuffer(bias), offset: 0, index: 2)
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
        let weight = try weightBuffer(weight)
        let biasBuffer = try bias.map(weightBuffer) ?? weight
        let rowsPerThreadgroup: Int
        if kernelPolicy.maximumDenseRowsPerThreadgroup >= 8,
            rows >= 8,
            inputDimension <= 3_072
        {
            rowsPerThreadgroup = 8
        } else if kernelPolicy.maximumDenseRowsPerThreadgroup >= 4,
            rows >= 4,
            inputDimension <= 3_072
        {
            rowsPerThreadgroup = 4
        } else {
            rowsPerThreadgroup = 1
        }
        let densePipeline: String
        switch rowsPerThreadgroup {
        case 8:
            densePipeline = "dense_bf16w_f32_rows8"
        case 4:
            densePipeline = "dense_bf16w_f32_rows4"
        default:
            densePipeline = "dense_bf16w_f32"
        }
        encoder.setComputePipelineState(
            try pipeline(densePipeline)
        )
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weight, offset: 0, index: 1)
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
            MTLSize(
                width: Int(outputDimension),
                height: (Int(rows) + rowsPerThreadgroup - 1)
                    / rowsPerThreadgroup,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(
                width: 32 * rowsPerThreadgroup,
                height: 1,
                depth: 1
            )
        )
    }

    private func encodeDenseEpilogue(
        _ encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        denseOutput: MTLBuffer,
        output: MTLBuffer,
        residual: MTLBuffer?,
        rows: Int,
        inputDimension: Int,
        outputDimension: Int,
        weight: String,
        bias: String?,
        epilogue: DenseEpilogue
    ) throws {
        guard kernelPolicy.fuseDenseEpilogue,
            rows >= 8,
            inputDimension <= 3_072
        else {
            try encodeDense(
                encoder,
                input: input,
                output: denseOutput,
                rows: rows,
                inputDimension: inputDimension,
                outputDimension: outputDimension,
                weight: weight,
                bias: bias
            )
            switch epilogue {
            case .gelu:
                try encodeGELU(
                    encoder,
                    input: denseOutput,
                    output: output,
                    count: rows * outputDimension
                )
            case .residualAdd:
                guard let residual else {
                    throw SmeltSkinDecoderError.invalidInputCount(0)
                }
                try encodeAdd(
                    encoder,
                    lhs: residual,
                    rhs: denseOutput,
                    output: output,
                    count: rows * outputDimension
                )
            }
            return
        }
        let weight = try weightBuffer(weight)
        let biasBuffer = try bias.map(weightBuffer) ?? weight
        let residualBuffer = residual ?? input
        let outputColumnsPerThreadgroup = inputDimension >= 2_048 ? 2 : 1
        let densePipeline =
            outputColumnsPerThreadgroup == 2
            ? "dense_bf16w_f32_rows8_cols2_epilogue"
            : "dense_bf16w_f32_rows8_epilogue"
        encoder.setComputePipelineState(
            try pipeline(densePipeline)
        )
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weight, offset: 0, index: 1)
        encoder.setBuffer(biasBuffer, offset: 0, index: 2)
        encoder.setBuffer(residualBuffer, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        var rows = UInt32(rows)
        var outputDimension = UInt32(outputDimension)
        var inputDimension = UInt32(inputDimension)
        var hasBias: UInt32 = bias == nil ? 0 : 1
        var epilogue = epilogue.rawValue
        encoder.setBytes(&rows, length: 4, index: 5)
        encoder.setBytes(&outputDimension, length: 4, index: 6)
        encoder.setBytes(&inputDimension, length: 4, index: 7)
        encoder.setBytes(&hasBias, length: 4, index: 8)
        encoder.setBytes(&epilogue, length: 4, index: 9)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (Int(outputDimension) + outputColumnsPerThreadgroup - 1)
                    / outputColumnsPerThreadgroup,
                height: (Int(rows) + 7) / 8,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
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

    private func encodeAttention(
        _ encoder: MTLComputeCommandEncoder,
        query: MTLBuffer,
        key: MTLBuffer,
        value: MTLBuffer,
        output: MTLBuffer,
        queryTokens: Int,
        dataTokens: Int,
        queryTile: Int
    ) throws {
        let tiledQueryTokens =
            queryTile >= 8 && Self.headDimension <= 64
            ? queryTokens - queryTokens % queryTile
            : 0
        if tiledQueryTokens > 0 {
            try encodeAttentionDispatch(
                encoder,
                query: query,
                key: key,
                value: value,
                output: output,
                queryOffset: 0,
                queryTokens: tiledQueryTokens,
                dataTokens: dataTokens,
                queryTile: queryTile
            )
        }
        if tiledQueryTokens < queryTokens {
            try encodeAttentionDispatch(
                encoder,
                query: query,
                key: key,
                value: value,
                output: output,
                queryOffset: tiledQueryTokens,
                queryTokens: queryTokens - tiledQueryTokens,
                dataTokens: dataTokens,
                queryTile: 1
            )
        }
    }

    private func encodeAttentionDispatch(
        _ encoder: MTLComputeCommandEncoder,
        query: MTLBuffer,
        key: MTLBuffer,
        value: MTLBuffer,
        output: MTLBuffer,
        queryOffset: Int,
        queryTokens: Int,
        dataTokens: Int,
        queryTile: Int
    ) throws {
        let pipelineName = switch queryTile {
        case 16: "noncausal_attention_q16_f32"
        case 8: "noncausal_attention_q8_f32"
        default: "noncausal_attention_f32"
        }
        encoder.setComputePipelineState(try pipeline(pipelineName))
        let byteOffset = queryOffset * Self.width * MemoryLayout<Float>.stride
        encoder.setBuffer(query, offset: byteOffset, index: 0)
        encoder.setBuffer(key, offset: 0, index: 1)
        encoder.setBuffer(value, offset: 0, index: 2)
        encoder.setBuffer(output, offset: byteOffset, index: 3)
        var queryTokens = UInt32(queryTokens)
        var dataTokens = UInt32(dataTokens)
        var heads = UInt32(Self.heads)
        var headDimension = UInt32(Self.headDimension)
        encoder.setBytes(&queryTokens, length: 4, index: 4)
        encoder.setBytes(&dataTokens, length: 4, index: 5)
        encoder.setBytes(&heads, length: 4, index: 6)
        encoder.setBytes(&headDimension, length: 4, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (Int(queryTokens) + queryTile - 1) / queryTile,
                height: Self.heads,
                depth: 1
            ),
            threadsPerThreadgroup: MTLSize(
                width: queryTile * 32,
                height: 1,
                depth: 1
            )
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

    private func commandBufferAndEncoder() throws -> (
        MTLCommandBuffer,
        MTLComputeCommandEncoder
    ) {
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw SmeltSkinDecoderError.commandBufferCreationFailed
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw SmeltSkinDecoderError.commandEncoderCreationFailed
        }
        return (commandBuffer, encoder)
    }

    private func finish(
        commandBuffer: MTLCommandBuffer,
        encoder: MTLComputeCommandEncoder
    ) throws {
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw SmeltSkinDecoderError.gpuExecutionFailed("\(error)")
        }
    }

    private func validatePointNormals(_ values: [Float]) throws {
        guard !values.isEmpty,
            values.count.isMultiple(of: 6),
            values.allSatisfy(\.isFinite)
        else {
            throw SmeltSkinDecoderError.invalidPointNormals(values.count)
        }
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
                throw SmeltSkinDecoderError.bufferCreationFailed(label)
            }
            return buffer
        }
        result.label = label
        return result
    }

    private func uintBuffer(_ values: [UInt32], label: String) throws -> MTLBuffer {
        let result = try values.withUnsafeBytes { bytes -> MTLBuffer in
            guard let base = bytes.baseAddress,
                let buffer = device.makeBuffer(
                    bytes: base,
                    length: bytes.count,
                    options: .storageModeShared
                )
            else {
                throw SmeltSkinDecoderError.bufferCreationFailed(label)
            }
            return buffer
        }
        result.label = label
        return result
    }

    private func buffer(count: Int, label: String) throws -> MTLBuffer {
        guard
            let buffer = device.makeBuffer(
                length: count * MemoryLayout<Float>.stride,
                options: .storageModeShared
            )
        else {
            throw SmeltSkinDecoderError.bufferCreationFailed(label)
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

    private func weightBuffer(_ name: String) throws -> MTLBuffer {
        weightBufferLock.lock()
        defer { weightBufferLock.unlock() }
        if let cached = weightBuffers[name] { return cached }
        let buffer = try artifact.makeWeightBuffer(
            device: device,
            tensorNamed: name
        )
        weightBuffers[name] = buffer
        return buffer
    }

    private func pipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let pipeline = pipelines[name] else {
            throw SmeltSkinDecoderError.pipelineMissing(name)
        }
        return pipeline
    }

    private static func widenedBF16(_ value: Float) -> Float {
        let bits = value.bitPattern
        let leastSignificantRetainedBit = (bits >> 16) & 1
        let rounded = bits &+ 0x7FFF &+ leastSignificantRetainedBit
        return Float(bitPattern: rounded & 0xFFFF_0000)
    }
}

public enum SmeltSkinDecoderError: Error, Equatable {
    case metalUnavailable
    case commandQueueCreationFailed
    case commandBufferCreationFailed
    case commandEncoderCreationFailed
    case pipelineMissing(String)
    case pipelineCreationFailed(String, String)
    case bufferCreationFailed(String)
    case invalidIndices
    case invalidIndexCount(Int)
    case invalidJointCount(Int)
    case invalidConditionCount(Int)
    case invalidInputCount(Int)
    case invalidDataCount(Int)
    case invalidPointNormals(Int)
    case gpuExecutionFailed(String)
}
