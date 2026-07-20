import Foundation
import Metal
import MetalPerformanceShaders

public enum SmeltQwen35VisionWeightDType: String, Sendable, Equatable {
    case f16 = "F16"
    case bf16 = "BF16"
}

/// The MPS backend is the production batched path. The Metal backend preserves
/// the original per-row reduction order for checkpoint bring-up and parity
/// localization; it is intentionally explicit rather than an ambient toggle.
public enum SmeltQwen35VisionGEMMBackend: Sendable, Equatable {
    case mps
    case referenceMetal
}

public enum SmeltQwen35VisionAttentionBackend: Sendable, Equatable {
    case reference
    case mpsStaged
}

public final class SmeltQwen35VisionWeights {
    public struct Matrix {
        public let buffer: MTLBuffer
        public let dtype: SmeltQwen35VisionWeightDType
        public let shape: [Int]
        let mpsBuffer: MTLBuffer
        let mpsDataType: MPSDataType
        let mpsElementBytes: Int
    }

    private let matrices: [String: Matrix]
    private let vectors: [String: MTLBuffer]

    public init(
        device: MTLDevice,
        checkpoint: any CheckpointTensorSource,
        plan: SmeltQwen35VisionCheckpointPlan
    ) throws {
        let descriptors = Dictionary(
            uniqueKeysWithValues: checkpoint.checkpointTensors.map { ($0.name, $0) }
        )
        var matrices: [String: Matrix] = [:]
        var vectors: [String: MTLBuffer] = [:]
        for tensor in plan.tensors {
            guard let descriptor = descriptors[tensor.name] else {
                throw SmeltQwen35VisionRuntimeError.missingWeight(tensor.name)
            }
            let source = checkpoint.checkpointTensorData(descriptor)
            if tensor.shape.count == 1 {
                let values = try Self.floatValues(
                    source: source,
                    dtype: tensor.dtype,
                    count: tensor.shape[0],
                    name: tensor.name
                )
                vectors[tensor.name] = try Self.buffer(
                    device: device,
                    values: values,
                    label: tensor.name
                )
            } else {
                guard let dtype = SmeltQwen35VisionWeightDType(rawValue: tensor.dtype) else {
                    throw SmeltQwen35VisionRuntimeError.unsupportedWeightDType(
                        tensor.name,
                        tensor.dtype
                    )
                }
                guard let buffer = device.makeBuffer(
                    bytes: source,
                    length: descriptor.byteCount,
                    options: .storageModeShared
                ) else {
                    throw SmeltQwen35VisionRuntimeError.bufferAllocationFailed(
                        tensor.name,
                        descriptor.byteCount
                    )
                }
                buffer.label = tensor.name
                let mps: (buffer: MTLBuffer, dataType: MPSDataType, elementBytes: Int)
                switch dtype {
                case .f16:
                    mps = (buffer, .float16, MemoryLayout<UInt16>.stride)
                case .bf16:
                    // MPSMatrixMultiplication's mixed BF16/F32 path is not
                    // numerically usable on the current Apple GPU stack. A
                    // BF16-to-F32 widen is exact and keeps artifact bytes and
                    // model precision unchanged while still enabling batched
                    // matrix execution.
                    let count = descriptor.byteCount / MemoryLayout<UInt16>.stride
                    let widenedBytes = count * MemoryLayout<Float>.stride
                    guard let widened = device.makeBuffer(
                        length: widenedBytes,
                        options: .storageModeShared
                    ) else {
                        throw SmeltQwen35VisionRuntimeError.bufferAllocationFailed(
                            "\(tensor.name).mps-f32",
                            widenedBytes
                        )
                    }
                    widened.label = "\(tensor.name).mps-f32"
                    let sourceValues = source.bindMemory(to: UInt16.self, capacity: count)
                    let destination = widened.contents().bindMemory(
                        to: Float.self,
                        capacity: count
                    )
                    for index in 0..<count {
                        destination[index] = Float(
                            bitPattern: UInt32(sourceValues[index]) << 16
                        )
                    }
                    mps = (widened, .float32, MemoryLayout<Float>.stride)
                }
                matrices[tensor.name] = Matrix(
                    buffer: buffer,
                    dtype: dtype,
                    shape: tensor.shape,
                    mpsBuffer: mps.buffer,
                    mpsDataType: mps.dataType,
                    mpsElementBytes: mps.elementBytes
                )
            }
        }
        self.matrices = matrices
        self.vectors = vectors
    }

    public func matrix(_ name: String) throws -> Matrix {
        guard let matrix = matrices[name] else {
            throw SmeltQwen35VisionRuntimeError.missingWeight(name)
        }
        return matrix
    }

    public func vector(_ name: String) throws -> MTLBuffer {
        guard let vector = vectors[name] else {
            throw SmeltQwen35VisionRuntimeError.missingWeight(name)
        }
        return vector
    }

    private static func floatValues(
        source: UnsafeRawPointer,
        dtype: String,
        count: Int,
        name: String
    ) throws -> [Float] {
        switch dtype {
        case "BF16":
            let values = source.bindMemory(to: UInt16.self, capacity: count)
            return (0..<count).map { Float(bitPattern: UInt32(values[$0]) << 16) }
        case "F16":
            let values = source.bindMemory(to: Float16.self, capacity: count)
            return (0..<count).map { Float(values[$0]) }
        default:
            throw SmeltQwen35VisionRuntimeError.unsupportedWeightDType(name, dtype)
        }
    }

    private static func buffer(
        device: MTLDevice,
        values: [Float],
        label: String
    ) throws -> MTLBuffer {
        let bytes = values.count * MemoryLayout<Float>.stride
        guard let buffer = values.withUnsafeBytes({ raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: bytes, options: .storageModeShared)
        }) else {
            throw SmeltQwen35VisionRuntimeError.bufferAllocationFailed(label, bytes)
        }
        buffer.label = label
        return buffer
    }
}

public final class SmeltQwen35VisionPipelines {
    let gemmBF16: MTLComputePipelineState
    let gemmF16: MTLComputePipelineState
    let softmaxRows: MTLComputePipelineState
    let add: MTLComputePipelineState
    let addBias: MTLComputePipelineState
    let layerNorm: MTLComputePipelineState
    let splitRoPE: MTLComputePipelineState
    let attention: MTLComputePipelineState
    let geluTanh: MTLComputePipelineState
    let checkFinite: MTLComputePipelineState

    public init(device: MTLDevice, shaderDirectory: String) throws {
        func library(_ file: String) throws -> MTLLibrary {
            let path = URL(fileURLWithPath: shaderDirectory).appendingPathComponent(file).path
            let source: String
            do {
                source = try String(contentsOfFile: path, encoding: .utf8)
            } catch {
                throw SmeltQwen35VisionRuntimeError.shaderCompilationFailed(
                    "\(file): \(error)"
                )
            }
            do {
                return try device.makeLibrary(source: source, options: nil)
            } catch {
                throw SmeltQwen35VisionRuntimeError.shaderCompilationFailed(
                    "\(file): \(error)"
                )
            }
        }

        let vision = try library("qwen35_vision.metal")
        let bf16 = try library("gemm_bf16w_f32.metal")
        let f16 = try library("gemm_f16w_f32.metal")
        gemmBF16 = try Self.pipeline(device: device, library: bf16, name: "gemm_bf16w_f32")
        gemmF16 = try Self.pipeline(device: device, library: f16, name: "gemm_f16w_f32")
        softmaxRows = try Self.pipeline(
            device: device,
            library: vision,
            name: "qwen35_vision_softmax_rows_f32"
        )
        add = try Self.pipeline(device: device, library: vision, name: "qwen35_vision_add_f32")
        addBias = try Self.pipeline(
            device: device,
            library: vision,
            name: "qwen35_vision_add_bias_rows_f32"
        )
        layerNorm = try Self.pipeline(
            device: device,
            library: vision,
            name: "qwen35_vision_layer_norm_f32"
        )
        splitRoPE = try Self.pipeline(
            device: device,
            library: vision,
            name: "qwen35_vision_split_rope_f32"
        )
        attention = try Self.pipeline(
            device: device,
            library: vision,
            name: "qwen35_vision_attention_f32"
        )
        geluTanh = try Self.pipeline(
            device: device,
            library: vision,
            name: "qwen35_vision_gelu_tanh_f32"
        )
        checkFinite = try Self.pipeline(
            device: device,
            library: vision,
            name: "qwen35_vision_check_finite_f32"
        )
    }

    public init(device: MTLDevice, library: MTLLibrary) throws {
        gemmBF16 = try Self.pipeline(device: device, library: library, name: "gemm_bf16w_f32")
        gemmF16 = try Self.pipeline(device: device, library: library, name: "gemm_f16w_f32")
        softmaxRows = try Self.pipeline(
            device: device,
            library: library,
            name: "qwen35_vision_softmax_rows_f32"
        )
        add = try Self.pipeline(device: device, library: library, name: "qwen35_vision_add_f32")
        addBias = try Self.pipeline(
            device: device,
            library: library,
            name: "qwen35_vision_add_bias_rows_f32"
        )
        layerNorm = try Self.pipeline(
            device: device,
            library: library,
            name: "qwen35_vision_layer_norm_f32"
        )
        splitRoPE = try Self.pipeline(
            device: device,
            library: library,
            name: "qwen35_vision_split_rope_f32"
        )
        attention = try Self.pipeline(
            device: device,
            library: library,
            name: "qwen35_vision_attention_f32"
        )
        geluTanh = try Self.pipeline(
            device: device,
            library: library,
            name: "qwen35_vision_gelu_tanh_f32"
        )
        checkFinite = try Self.pipeline(
            device: device,
            library: library,
            name: "qwen35_vision_check_finite_f32"
        )
    }

    private static func pipeline(
        device: MTLDevice,
        library: MTLLibrary,
        name: String
    ) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw SmeltQwen35VisionRuntimeError.pipelineMissing(name)
        }
        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            throw SmeltQwen35VisionRuntimeError.shaderCompilationFailed("\(name): \(error)")
        }
    }
}

public final class SmeltQwen35VisionRuntime {
    private struct MPSKernelKey: Hashable {
        let rows: Int
        let outputSize: Int
        let inputSize: Int
    }

    private struct MPSAttentionKernels {
        let queryKey: MPSMatrixMultiplication
        let probabilityValue: MPSMatrixMultiplication
    }

    public struct Grid: Sendable, Equatable {
        public let temporal: Int
        public let height: Int
        public let width: Int

        public init(temporal: Int, height: Int, width: Int) {
            self.temporal = temporal
            self.height = height
            self.width = width
        }

        public var patchCount: Int { temporal * height * width }
    }

    public struct Output {
        public let buffer: MTLBuffer
        public let tokenCount: Int
        public let hiddenSize: Int
        public let timing: Timing
        public let operationProfile: SmeltFrozenOperationTimingProfile?

        public func values() -> [Float] {
            let count = tokenCount * hiddenSize
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: pointer, count: count))
        }
    }

    public struct Timing: Sendable, Equatable {
        public let workspaceSeconds: Double
        public let patchCopySeconds: Double
        public let positionSeconds: Double
        public let setupSeconds: Double
        public let commandEncodingSeconds: Double
        public let commandExecutionSeconds: Double
        public let gpuSeconds: Double
        public let totalSeconds: Double
    }

    private final class Workspace {
        let patches: MTLBuffer
        let position: MTLBuffer
        let cosines: MTLBuffer
        let sines: MTLBuffer
        let chunkStart: MTLBuffer
        let chunkEnd: MTLBuffer
        let hiddenA: MTLBuffer
        let hiddenB: MTLBuffer
        let normed: MTLBuffer
        let qkv: MTLBuffer
        let q: MTLBuffer
        let k: MTLBuffer
        let v: MTLBuffer
        let attention: MTLBuffer
        let projection: MTLBuffer
        let intermediate: MTLBuffer
        let mergedIntermediate: MTLBuffer
        let output: MTLBuffer
        let attentionScores: MTLBuffer?

        init(
            device: MTLDevice,
            config: SmeltQwen35VisionConfig,
            tokenCount: Int,
            attentionScoreElements: Int?
        ) throws {
            let patchWidth = config.inChannels
                * config.temporalPatchSize
                * config.patchSize
                * config.patchSize
            let mergedTokens = tokenCount / (config.spatialMergeSize * config.spatialMergeSize)
            let mergedHidden = config.hiddenSize
                * config.spatialMergeSize
                * config.spatialMergeSize

            func make(_ count: Int, label: String, stride: Int = 4) throws -> MTLBuffer {
                let bytes = count * stride
                guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                    throw SmeltQwen35VisionRuntimeError.bufferAllocationFailed(label, bytes)
                }
                buffer.label = label
                return buffer
            }

            patches = try make(tokenCount * patchWidth, label: "qwen35.vision.patches")
            position = try make(tokenCount * config.hiddenSize, label: "qwen35.vision.position")
            cosines = try make(tokenCount * config.headDim, label: "qwen35.vision.cos")
            sines = try make(tokenCount * config.headDim, label: "qwen35.vision.sin")
            chunkStart = try make(tokenCount, label: "qwen35.vision.chunk-start", stride: 4)
            chunkEnd = try make(tokenCount, label: "qwen35.vision.chunk-end", stride: 4)
            hiddenA = try make(tokenCount * config.hiddenSize, label: "qwen35.vision.hidden-a")
            hiddenB = try make(tokenCount * config.hiddenSize, label: "qwen35.vision.hidden-b")
            normed = try make(tokenCount * config.hiddenSize, label: "qwen35.vision.normed")
            qkv = try make(tokenCount * config.hiddenSize * 3, label: "qwen35.vision.qkv")
            q = try make(tokenCount * config.hiddenSize, label: "qwen35.vision.q")
            k = try make(tokenCount * config.hiddenSize, label: "qwen35.vision.k")
            v = try make(tokenCount * config.hiddenSize, label: "qwen35.vision.v")
            attention = try make(tokenCount * config.hiddenSize, label: "qwen35.vision.attention")
            projection = try make(tokenCount * config.hiddenSize, label: "qwen35.vision.projection")
            intermediate = try make(
                tokenCount * config.intermediateSize,
                label: "qwen35.vision.intermediate"
            )
            mergedIntermediate = try make(
                mergedTokens * mergedHidden,
                label: "qwen35.vision.merger-intermediate"
            )
            output = try make(
                mergedTokens * config.outputHiddenSize,
                label: "qwen35.vision.output"
            )
            if let attentionScoreElements {
                attentionScores = device.makeBuffer(
                    length: attentionScoreElements * MemoryLayout<Float>.stride,
                    options: .storageModeShared
                )
                attentionScores?.label = "qwen35.vision.attention-scores"
            } else {
                attentionScores = nil
            }
        }
    }

    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let config: SmeltQwen35VisionConfig
    public let weights: SmeltQwen35VisionWeights
    public let pipelines: SmeltQwen35VisionPipelines
    public let gemmBackend: SmeltQwen35VisionGEMMBackend
    public let attentionBackend: SmeltQwen35VisionAttentionBackend

    private var mpsKernels: [MPSKernelKey: MPSMatrixMultiplication] = [:]
    private var mpsAttentionKernels: [Int: MPSAttentionKernels] = [:]
    private let mpsKernelLock = NSLock()

    public init(
        device: MTLDevice,
        queue: MTLCommandQueue,
        config: SmeltQwen35VisionConfig,
        weights: SmeltQwen35VisionWeights,
        pipelines: SmeltQwen35VisionPipelines,
        gemmBackend: SmeltQwen35VisionGEMMBackend = .mps,
        attentionBackend: SmeltQwen35VisionAttentionBackend = .mpsStaged
    ) {
        self.device = device
        self.queue = queue
        self.config = config
        self.weights = weights
        self.pipelines = pipelines
        self.gemmBackend = gemmBackend
        self.attentionBackend = attentionBackend
    }

    public func encode(
        patches: [Float],
        grids: [Grid],
        diagnoseNonFinite: Bool = false,
        profileFrozenOperations: Bool = false
    ) throws -> Output {
        let totalStart = ProcessInfo.processInfo.systemUptime
        guard !grids.isEmpty,
              grids.allSatisfy({
                  $0.temporal > 0 && $0.height > 0 && $0.width > 0
                    && $0.height % config.spatialMergeSize == 0
                    && $0.width % config.spatialMergeSize == 0
              })
        else {
            throw SmeltQwen35VisionRuntimeError.invalidGrid(grids)
        }
        let tokens = grids.reduce(0) { $0 + $1.patchCount }
        let mergeUnit = config.spatialMergeSize * config.spatialMergeSize
        guard tokens % mergeUnit == 0 else {
            throw SmeltQwen35VisionRuntimeError.invalidGrid(grids)
        }
        let patchWidth = config.inChannels
            * config.temporalPatchSize
            * config.patchSize
            * config.patchSize
        guard patches.count == tokens * patchWidth else {
            throw SmeltQwen35VisionRuntimeError.patchCountMismatch(
                expected: tokens * patchWidth,
                got: patches.count
            )
        }

        let largestAttentionChunk = grids.map { $0.height * $0.width }.max() ?? 0
        let scoreElementsResult = largestAttentionChunk.multipliedReportingOverflow(
            by: largestAttentionChunk
        )
        let scoreBytesResult = scoreElementsResult.partialValue.multipliedReportingOverflow(
            by: MemoryLayout<Float>.stride
        )
        let requestMPSStagedAttention = attentionBackend == .mpsStaged
            && !scoreElementsResult.overflow
            && !scoreBytesResult.overflow
            && scoreBytesResult.partialValue <= device.maxBufferLength
        let workspace = try Workspace(
            device: device,
            config: config,
            tokenCount: tokens,
            attentionScoreElements: requestMPSStagedAttention
                ? scoreElementsResult.partialValue
                : nil
        )
        let workspaceEnd = ProcessInfo.processInfo.systemUptime
        let useMPSStagedAttention = requestMPSStagedAttention
            && workspace.attentionScores != nil
        if profileFrozenOperations && diagnoseNonFinite {
            throw SmeltQwen35VisionRuntimeError.incompatibleDiagnosticsAndProfiling
        }
        let operationProfiler: SmeltMetalFrozenOperationProfiler?
        if profileFrozenOperations {
            guard gemmBackend == .mps, useMPSStagedAttention else {
                throw SmeltQwen35VisionRuntimeError.unsupportedProfiledRoute(
                    gemm: gemmBackend,
                    attention: attentionBackend
                )
            }
            let frozenPlan = try Self.frozenPlan(
                config: config,
                grids: grids,
                provenanceKey: "runtime-profile"
            )
            operationProfiler = try SmeltMetalFrozenOperationProfiler(
                device: device,
                recordCapacity: frozenPlan.records.count
            )
        } else {
            operationProfiler = nil
        }
        _ = patches.withUnsafeBytes { raw in
            memcpy(workspace.patches.contents(), raw.baseAddress!, raw.count)
        }
        let patchCopyEnd = ProcessInfo.processInfo.systemUptime
        try preparePositionInputs(workspace: workspace, grids: grids)
        let setupEnd = ProcessInfo.processInfo.systemUptime

        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw SmeltQwen35VisionRuntimeError.commandBufferUnavailable
        }
        commandBuffer.label = "qwen35.vision.encode"
        var encoder: MTLComputeCommandEncoder?
        func currentEncoder() throws -> MTLComputeCommandEncoder {
            if let encoder { return encoder }
            let created: MTLComputeCommandEncoder
            if let operationProfiler {
                created = try operationProfiler.makeComputeEncoder(
                    commandBuffer: commandBuffer,
                    label: "qwen35.vision.frozen-segment"
                )
            } else {
                guard let normal = commandBuffer.makeComputeCommandEncoder() else {
                    throw SmeltQwen35VisionRuntimeError.commandBufferUnavailable
                }
                created = normal
            }
            encoder = created
            return created
        }
        func endCurrentEncoder() {
            if let encoder {
                if let operationProfiler {
                    operationProfiler.endComputeEncoder(encoder)
                } else {
                    encoder.endEncoding()
                }
            }
            encoder = nil
        }
        func metal(
            _ label: String,
            _ body: (MTLComputeCommandEncoder) throws -> Void
        ) throws {
            if let operationProfiler {
                let current = try currentEncoder()
                try operationProfiler.recordMetalOperation(label: label)
                try body(current)
            } else {
                try body(currentEncoder())
            }
        }
        if let operationProfiler {
            try operationProfiler.encodeBoundaryMarker(
                commandBuffer: commandBuffer,
                buffer: workspace.patches,
                label: "qwen35.vision.profile-start"
            )
        }
        let diagnosticBuffer: MTLBuffer?
        if diagnoseNonFinite {
            guard let buffer = device.makeBuffer(length: 4, options: .storageModeShared) else {
                throw SmeltQwen35VisionRuntimeError.bufferAllocationFailed("finite-check", 4)
            }
            buffer.contents().storeBytes(of: UInt32.max, as: UInt32.self)
            diagnosticBuffer = buffer
        } else {
            diagnosticBuffer = nil
        }
        var diagnosticStages: [String] = []
        func check(_ buffer: MTLBuffer, _ count: Int, _ label: String) throws {
            guard let diagnosticBuffer else { return }
            let stage = diagnosticStages.count
            diagnosticStages.append(label)
            dispatchFiniteCheck(
                try currentEncoder(),
                input: buffer,
                diagnosticBuffer: diagnosticBuffer,
                count: count,
                stage: stage
            )
        }
        func gemm(
            label: String,
            input: MTLBuffer,
            matrix: SmeltQwen35VisionWeights.Matrix,
            bias: MTLBuffer,
            output: MTLBuffer,
            rows: Int,
            inputSize: Int,
            outputSize: Int
        ) throws {
            if gemmBackend == .referenceMetal {
                try metal(label) { encoder in
                    try dispatchGEMM(
                        encoder,
                        input: input,
                        matrix: matrix,
                        bias: bias,
                        output: output,
                        rows: rows,
                        inputSize: inputSize,
                        outputSize: outputSize
                    )
                }
                return
            }
            endCurrentEncoder()
            if let operationProfiler {
                try operationProfiler.encodeOpaqueOperation(label: label) {
                    try dispatchMPSGEMM(
                        commandBuffer,
                        input: input,
                        matrix: matrix,
                        output: output,
                        rows: rows,
                        inputSize: inputSize,
                        outputSize: outputSize
                    )
                }
            } else {
                try dispatchMPSGEMM(
                    commandBuffer,
                    input: input,
                    matrix: matrix,
                    output: output,
                    rows: rows,
                    inputSize: inputSize,
                    outputSize: outputSize
                )
            }
            try metal("\(label).bias") { encoder in
                dispatchAddBias(
                    encoder,
                    values: output,
                    bias: bias,
                    rows: rows,
                    columns: outputSize
                )
            }
        }
        func attention() throws {
            if useMPSStagedAttention {
                endCurrentEncoder()
                try dispatchMPSStagedAttention(
                    commandBuffer,
                    workspace: workspace,
                    grids: grids,
                    profiler: operationProfiler
                )
                if let operationProfiler {
                    try operationProfiler.encodeBoundaryMarker(
                        commandBuffer: commandBuffer,
                        buffer: workspace.attention,
                        label: "qwen35.vision.attention-end"
                    )
                }
            } else {
                try metal("attention.reference") { encoder in
                    dispatchAttention(encoder, workspace: workspace, tokens: tokens)
                }
            }
        }

        let patchWeight = try weights.matrix("model.visual.patch_embed.proj.weight")
        try gemm(
            label: "patch-embed",
            input: workspace.patches,
            matrix: patchWeight,
            bias: try weights.vector("model.visual.patch_embed.proj.bias"),
            output: workspace.hiddenA,
            rows: tokens,
            inputSize: patchWidth,
            outputSize: config.hiddenSize
        )
        try check(workspace.hiddenA, tokens * config.hiddenSize, "patch-embed")
        try metal("position-add") { encoder in
            dispatchAdd(
                encoder,
                lhs: workspace.hiddenA,
                rhs: workspace.position,
                output: workspace.hiddenB,
                count: tokens * config.hiddenSize
            )
        }
        try check(workspace.hiddenB, tokens * config.hiddenSize, "position-add")

        var residual = workspace.hiddenB
        var nextResidual = workspace.hiddenA
        for layer in 0..<config.layerCount {
            let prefix = "model.visual.blocks.\(layer)"
            let norm1Weight = try weights.vector("\(prefix).norm1.weight")
            let norm1Bias = try weights.vector("\(prefix).norm1.bias")
            try metal("layer-\(layer).norm1") { encoder in
                dispatchLayerNorm(
                    encoder,
                    input: residual,
                    weight: norm1Weight,
                    bias: norm1Bias,
                    output: workspace.normed,
                    rows: tokens,
                    dim: config.hiddenSize
                )
            }
            try check(workspace.normed, tokens * config.hiddenSize, "layer-\(layer).norm1")
            try gemm(
                label: "layer-\(layer).qkv",
                input: workspace.normed,
                matrix: try weights.matrix("\(prefix).attn.qkv.weight"),
                bias: try weights.vector("\(prefix).attn.qkv.bias"),
                output: workspace.qkv,
                rows: tokens,
                inputSize: config.hiddenSize,
                outputSize: config.hiddenSize * 3
            )
            try check(workspace.qkv, tokens * config.hiddenSize * 3, "layer-\(layer).qkv")
            try metal("layer-\(layer).qkv-rope-split") { encoder in
                dispatchSplitRoPE(encoder, workspace: workspace, tokens: tokens)
            }
            try check(workspace.q, tokens * config.hiddenSize, "layer-\(layer).q-rope")
            try check(workspace.k, tokens * config.hiddenSize, "layer-\(layer).k-rope")
            try check(workspace.v, tokens * config.hiddenSize, "layer-\(layer).v")
            try attention()
            try check(workspace.attention, tokens * config.hiddenSize, "layer-\(layer).attention")
            try gemm(
                label: "layer-\(layer).attention-proj",
                input: workspace.attention,
                matrix: try weights.matrix("\(prefix).attn.proj.weight"),
                bias: try weights.vector("\(prefix).attn.proj.bias"),
                output: workspace.projection,
                rows: tokens,
                inputSize: config.hiddenSize,
                outputSize: config.hiddenSize
            )
            try check(workspace.projection, tokens * config.hiddenSize, "layer-\(layer).attn-proj")
            try metal("layer-\(layer).attention-residual") { encoder in
                dispatchAdd(
                    encoder,
                    lhs: residual,
                    rhs: workspace.projection,
                    output: nextResidual,
                    count: tokens * config.hiddenSize
                )
            }
            try check(nextResidual, tokens * config.hiddenSize, "layer-\(layer).attn-residual")
            swap(&residual, &nextResidual)

            let norm2Weight = try weights.vector("\(prefix).norm2.weight")
            let norm2Bias = try weights.vector("\(prefix).norm2.bias")
            try metal("layer-\(layer).norm2") { encoder in
                dispatchLayerNorm(
                    encoder,
                    input: residual,
                    weight: norm2Weight,
                    bias: norm2Bias,
                    output: workspace.normed,
                    rows: tokens,
                    dim: config.hiddenSize
                )
            }
            try check(workspace.normed, tokens * config.hiddenSize, "layer-\(layer).norm2")
            try gemm(
                label: "layer-\(layer).fc1",
                input: workspace.normed,
                matrix: try weights.matrix("\(prefix).mlp.linear_fc1.weight"),
                bias: try weights.vector("\(prefix).mlp.linear_fc1.bias"),
                output: workspace.intermediate,
                rows: tokens,
                inputSize: config.hiddenSize,
                outputSize: config.intermediateSize
            )
            try check(workspace.intermediate, tokens * config.intermediateSize, "layer-\(layer).fc1")
            try metal("layer-\(layer).gelu") { encoder in
                dispatchGELU(
                    encoder,
                    buffer: workspace.intermediate,
                    count: tokens * config.intermediateSize
                )
            }
            try check(workspace.intermediate, tokens * config.intermediateSize, "layer-\(layer).gelu")
            try gemm(
                label: "layer-\(layer).fc2",
                input: workspace.intermediate,
                matrix: try weights.matrix("\(prefix).mlp.linear_fc2.weight"),
                bias: try weights.vector("\(prefix).mlp.linear_fc2.bias"),
                output: workspace.projection,
                rows: tokens,
                inputSize: config.intermediateSize,
                outputSize: config.hiddenSize
            )
            try check(workspace.projection, tokens * config.hiddenSize, "layer-\(layer).fc2")
            try metal("layer-\(layer).ffn-residual") { encoder in
                dispatchAdd(
                    encoder,
                    lhs: residual,
                    rhs: workspace.projection,
                    output: nextResidual,
                    count: tokens * config.hiddenSize
                )
            }
            try check(nextResidual, tokens * config.hiddenSize, "layer-\(layer).ffn-residual")
            swap(&residual, &nextResidual)
        }

        let mergerNormWeight = try weights.vector("model.visual.merger.norm.weight")
        let mergerNormBias = try weights.vector("model.visual.merger.norm.bias")
        try metal("merger.norm") { encoder in
            dispatchLayerNorm(
                encoder,
                input: residual,
                weight: mergerNormWeight,
                bias: mergerNormBias,
                output: workspace.normed,
                rows: tokens,
                dim: config.hiddenSize
            )
        }
        try check(workspace.normed, tokens * config.hiddenSize, "merger.norm")
        let mergedTokens = tokens / mergeUnit
        let mergedHidden = config.hiddenSize * mergeUnit
        try gemm(
            label: "merger.fc1",
            input: workspace.normed,
            matrix: try weights.matrix("model.visual.merger.linear_fc1.weight"),
            bias: try weights.vector("model.visual.merger.linear_fc1.bias"),
            output: workspace.mergedIntermediate,
            rows: mergedTokens,
            inputSize: mergedHidden,
            outputSize: mergedHidden
        )
        try check(
            workspace.mergedIntermediate,
            mergedTokens * mergedHidden,
            "merger.fc1"
        )
        try metal("merger.gelu") { encoder in
            dispatchGELU(
                encoder,
                buffer: workspace.mergedIntermediate,
                count: mergedTokens * mergedHidden
            )
        }
        try check(
            workspace.mergedIntermediate,
            mergedTokens * mergedHidden,
            "merger.gelu"
        )
        try gemm(
            label: "merger.fc2",
            input: workspace.mergedIntermediate,
            matrix: try weights.matrix("model.visual.merger.linear_fc2.weight"),
            bias: try weights.vector("model.visual.merger.linear_fc2.bias"),
            output: workspace.output,
            rows: mergedTokens,
            inputSize: mergedHidden,
            outputSize: config.outputHiddenSize
        )
        try check(
            workspace.output,
            mergedTokens * config.outputHiddenSize,
            "merger.fc2"
        )

        endCurrentEncoder()
        let encodingEnd = ProcessInfo.processInfo.systemUptime
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let completionEnd = ProcessInfo.processInfo.systemUptime
        if let error = commandBuffer.error { throw error }
        let operationProfile = try operationProfiler?.finish(
            commandBuffer: commandBuffer
        )
        if let diagnosticBuffer {
            let firstBad = diagnosticBuffer.contents().load(as: UInt32.self)
            if firstBad != UInt32.max {
                let index = Int(firstBad)
                let label = diagnosticStages.indices.contains(index)
                    ? diagnosticStages[index]
                    : "unknown-stage-\(index)"
                throw SmeltQwen35VisionRuntimeError.nonFiniteStage(label)
            }
        }
        return Output(
            buffer: workspace.output,
            tokenCount: mergedTokens,
            hiddenSize: config.outputHiddenSize,
            timing: Timing(
                workspaceSeconds: workspaceEnd - totalStart,
                patchCopySeconds: patchCopyEnd - workspaceEnd,
                positionSeconds: setupEnd - patchCopyEnd,
                setupSeconds: setupEnd - totalStart,
                commandEncodingSeconds: encodingEnd - setupEnd,
                commandExecutionSeconds: completionEnd - encodingEnd,
                gpuSeconds: max(commandBuffer.gpuEndTime - commandBuffer.gpuStartTime, 0),
                totalSeconds: completionEnd - totalStart
            ),
            operationProfile: operationProfile
        )
    }

    private func preparePositionInputs(workspace: Workspace, grids: [Grid]) throws {
        let table = try weights.matrix("model.visual.pos_embed.weight")
        guard table.shape == [config.positionEmbeddingCount, config.hiddenSize] else {
            throw SmeltQwen35VisionRuntimeError.invalidPositionTable(table.shape)
        }
        let side = Int(Double(config.positionEmbeddingCount).squareRoot())
        guard side * side == config.positionEmbeddingCount else {
            throw SmeltQwen35VisionRuntimeError.invalidPositionTable(table.shape)
        }
        let pos = workspace.position.contents().bindMemory(
            to: Float.self,
            capacity: grids.reduce(0) { $0 + $1.patchCount } * config.hiddenSize
        )
        let cosines = workspace.cosines.contents().bindMemory(
            to: Float.self,
            capacity: grids.reduce(0) { $0 + $1.patchCount } * config.headDim
        )
        let sines = workspace.sines.contents().bindMemory(
            to: Float.self,
            capacity: grids.reduce(0) { $0 + $1.patchCount } * config.headDim
        )
        let starts = workspace.chunkStart.contents().bindMemory(
            to: UInt32.self,
            capacity: grids.reduce(0) { $0 + $1.patchCount }
        )
        let ends = workspace.chunkEnd.contents().bindMemory(
            to: UInt32.self,
            capacity: grids.reduce(0) { $0 + $1.patchCount }
        )
        let merge = config.spatialMergeSize
        let frequencyCount = config.headDim / 4
        let tableF32 = table.dtype == .bf16
            ? table.mpsBuffer.contents().bindMemory(
                to: Float.self,
                capacity: config.positionEmbeddingCount * config.hiddenSize
            )
            : nil
        let tableF16 = table.dtype == .f16
            ? table.buffer.contents().bindMemory(
                to: Float16.self,
                capacity: config.positionEmbeddingCount * config.hiddenSize
            )
            : nil

        func writePosition(token: Int, indexes: [Int], factors: [Float]) {
            let output = pos + token * config.hiddenSize
            if let values = tableF32 {
                for channel in 0..<config.hiddenSize {
                    var value: Float = 0
                    for corner in 0..<4 {
                        value += factors[corner]
                            * values[indexes[corner] * config.hiddenSize + channel]
                    }
                    output[channel] = value
                }
            } else if let values = tableF16 {
                for channel in 0..<config.hiddenSize {
                    var value: Float = 0
                    for corner in 0..<4 {
                        value += factors[corner]
                            * Float(values[indexes[corner] * config.hiddenSize + channel])
                    }
                    output[channel] = value
                }
            }
        }

        var token = 0
        for grid in grids {
            let rowCoordinates = Self.interpolationCoordinates(count: grid.height, side: side)
            let columnCoordinates = Self.interpolationCoordinates(count: grid.width, side: side)
            for _ in 0..<grid.temporal {
                let chunkBegin = token
                let chunkEnd = chunkBegin + grid.height * grid.width
                for blockRow in 0..<(grid.height / merge) {
                    for blockColumn in 0..<(grid.width / merge) {
                        for innerRow in 0..<merge {
                            for innerColumn in 0..<merge {
                                let row = blockRow * merge + innerRow
                                let column = blockColumn * merge + innerColumn
                                let rc = rowCoordinates[row]
                                let cc = columnCoordinates[column]
                                let indexes = [
                                    rc.lower * side + cc.lower,
                                    rc.lower * side + cc.upper,
                                    rc.upper * side + cc.lower,
                                    rc.upper * side + cc.upper,
                                ]
                                let factors = [
                                    (1 - rc.fraction) * (1 - cc.fraction),
                                    (1 - rc.fraction) * cc.fraction,
                                    rc.fraction * (1 - cc.fraction),
                                    rc.fraction * cc.fraction,
                                ]
                                writePosition(
                                    token: token,
                                    indexes: indexes,
                                    factors: factors
                                )
                                for frequency in 0..<frequencyCount {
                                    let inverse = pow(
                                        10_000.0,
                                        -Float(2 * frequency) / Float(config.headDim / 2)
                                    )
                                    let rowAngle = Float(row) * inverse
                                    let columnAngle = Float(column) * inverse
                                    let first = token * config.headDim + frequency
                                    let second = first + frequencyCount
                                    let third = second + frequencyCount
                                    let fourth = third + frequencyCount
                                    cosines[first] = cos(rowAngle)
                                    sines[first] = sin(rowAngle)
                                    cosines[second] = cos(columnAngle)
                                    sines[second] = sin(columnAngle)
                                    cosines[third] = cosines[first]
                                    sines[third] = sines[first]
                                    cosines[fourth] = cosines[second]
                                    sines[fourth] = sines[second]
                                }
                                starts[token] = UInt32(chunkBegin)
                                ends[token] = UInt32(chunkEnd)
                                token += 1
                            }
                        }
                    }
                }
            }
        }
    }

    private struct InterpolationCoordinate {
        let lower: Int
        let upper: Int
        let fraction: Float
    }

    private static func interpolationCoordinates(
        count: Int,
        side: Int
    ) -> [InterpolationCoordinate] {
        if count == 1 {
            return [.init(lower: 0, upper: 0, fraction: 0)]
        }
        return (0..<count).map { index in
            let value = Float(index) * Float(side - 1) / Float(count - 1)
            let lower = Int(value.rounded(.down))
            return .init(
                lower: lower,
                upper: min(lower + 1, side - 1),
                fraction: value - Float(lower)
            )
        }
    }

    private func dispatchMPSGEMM(
        _ commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        matrix: SmeltQwen35VisionWeights.Matrix,
        output: MTLBuffer,
        rows: Int,
        inputSize: Int,
        outputSize: Int
    ) throws {
        guard matrix.shape.reduce(1, *) == outputSize * inputSize else {
            throw SmeltQwen35VisionRuntimeError.matrixShapeMismatch(
                matrix.shape,
                expected: [outputSize, inputSize]
            )
        }
        let inputDescriptor = MPSMatrixDescriptor(
            rows: rows,
            columns: inputSize,
            rowBytes: inputSize * MemoryLayout<Float>.stride,
            dataType: .float32
        )
        let weightDescriptor = MPSMatrixDescriptor(
            rows: outputSize,
            columns: inputSize,
            rowBytes: inputSize * matrix.mpsElementBytes,
            dataType: matrix.mpsDataType
        )
        let outputDescriptor = MPSMatrixDescriptor(
            rows: rows,
            columns: outputSize,
            rowBytes: outputSize * MemoryLayout<Float>.stride,
            dataType: .float32
        )
        let key = MPSKernelKey(
            rows: rows,
            outputSize: outputSize,
            inputSize: inputSize
        )
        mpsKernelLock.lock()
        let kernel: MPSMatrixMultiplication
        if let cached = mpsKernels[key] {
            kernel = cached
        } else {
            let created = MPSMatrixMultiplication(
                device: device,
                transposeLeft: false,
                transposeRight: true,
                resultRows: rows,
                resultColumns: outputSize,
                interiorColumns: inputSize,
                alpha: 1,
                beta: 0
            )
            created.options = .skipAPIValidation
            mpsKernels[key] = created
            kernel = created
        }
        mpsKernelLock.unlock()
        kernel.encode(
            commandBuffer: commandBuffer,
            leftMatrix: MPSMatrix(buffer: input, descriptor: inputDescriptor),
            rightMatrix: MPSMatrix(buffer: matrix.mpsBuffer, descriptor: weightDescriptor),
            resultMatrix: MPSMatrix(buffer: output, descriptor: outputDescriptor)
        )
    }

    private func dispatchMPSStagedAttention(
        _ commandBuffer: MTLCommandBuffer,
        workspace: Workspace,
        grids: [Grid],
        profiler: SmeltMetalFrozenOperationProfiler?
    ) throws {
        guard let scores = workspace.attentionScores else {
            throw SmeltQwen35VisionRuntimeError.commandBufferUnavailable
        }
        let hidden = config.headCount * config.headDim
        var chunkStart = 0
        for (segment, grid) in grids.enumerated() {
            let chunkTokens = grid.height * grid.width
            let activationDescriptor = MPSMatrixDescriptor(
                rows: chunkTokens,
                columns: config.headDim,
                rowBytes: hidden * MemoryLayout<Float>.stride,
                dataType: .float32
            )
            let scoreDescriptor = MPSMatrixDescriptor(
                rows: chunkTokens,
                columns: chunkTokens,
                rowBytes: chunkTokens * MemoryLayout<Float>.stride,
                dataType: .float32
            )

            mpsKernelLock.lock()
            let kernels: MPSAttentionKernels
            if let cached = mpsAttentionKernels[chunkTokens] {
                kernels = cached
            } else {
                let queryKey = MPSMatrixMultiplication(
                    device: device,
                    transposeLeft: false,
                    transposeRight: true,
                    resultRows: chunkTokens,
                    resultColumns: chunkTokens,
                    interiorColumns: config.headDim,
                    alpha: 1,
                    beta: 0
                )
                queryKey.options = .skipAPIValidation
                let probabilityValue = MPSMatrixMultiplication(
                    device: device,
                    transposeLeft: false,
                    transposeRight: false,
                    resultRows: chunkTokens,
                    resultColumns: config.headDim,
                    interiorColumns: chunkTokens,
                    alpha: 1,
                    beta: 0
                )
                probabilityValue.options = .skipAPIValidation
                let created = MPSAttentionKernels(
                    queryKey: queryKey,
                    probabilityValue: probabilityValue
                )
                mpsAttentionKernels[chunkTokens] = created
                kernels = created
            }
            mpsKernelLock.unlock()

            for frame in 0..<grid.temporal {
                for head in 0..<config.headCount {
                    let stem = "attention.s\(segment).f\(frame).h\(head)"
                    let offset = (chunkStart * hidden + head * config.headDim)
                        * MemoryLayout<Float>.stride
                    let encodeQueryKey = {
                        kernels.queryKey.encode(
                            commandBuffer: commandBuffer,
                            leftMatrix: MPSMatrix(
                                buffer: workspace.q,
                                offset: offset,
                                descriptor: activationDescriptor
                            ),
                            rightMatrix: MPSMatrix(
                                buffer: workspace.k,
                                offset: offset,
                                descriptor: activationDescriptor
                            ),
                            resultMatrix: MPSMatrix(
                                buffer: scores,
                                descriptor: scoreDescriptor
                            )
                        )
                    }
                    if let profiler {
                        try profiler.encodeOpaqueOperation(
                            label: "\(stem).qk",
                            encodeQueryKey
                        )
                        let softmaxEncoder = try profiler.makeComputeEncoder(
                            commandBuffer: commandBuffer,
                            label: "\(stem).softmax"
                        )
                        try profiler.recordMetalOperation(label: "\(stem).softmax")
                        dispatchSoftmaxRows(
                            softmaxEncoder,
                            scores: scores,
                            rows: chunkTokens,
                            columns: chunkTokens
                        )
                        profiler.endComputeEncoder(softmaxEncoder)
                    } else {
                        encodeQueryKey()
                        guard let softmaxEncoder = commandBuffer.makeComputeCommandEncoder() else {
                            throw SmeltQwen35VisionRuntimeError.commandBufferUnavailable
                        }
                        dispatchSoftmaxRows(
                            softmaxEncoder,
                            scores: scores,
                            rows: chunkTokens,
                            columns: chunkTokens
                        )
                        softmaxEncoder.endEncoding()
                    }
                    let encodeProbabilityValue = {
                        kernels.probabilityValue.encode(
                            commandBuffer: commandBuffer,
                            leftMatrix: MPSMatrix(
                                buffer: scores,
                                descriptor: scoreDescriptor
                            ),
                            rightMatrix: MPSMatrix(
                                buffer: workspace.v,
                                offset: offset,
                                descriptor: activationDescriptor
                            ),
                            resultMatrix: MPSMatrix(
                                buffer: workspace.attention,
                                offset: offset,
                                descriptor: activationDescriptor
                            )
                        )
                    }
                    if let profiler {
                        try profiler.encodeOpaqueOperation(
                            label: "\(stem).pv",
                            encodeProbabilityValue
                        )
                    } else {
                        encodeProbabilityValue()
                    }
                }
                chunkStart += chunkTokens
            }
        }
    }

    private func dispatchSoftmaxRows(
        _ encoder: MTLComputeCommandEncoder,
        scores: MTLBuffer,
        rows: Int,
        columns: Int
    ) {
        encoder.setComputePipelineState(pipelines.softmaxRows)
        encoder.setBuffer(scores, offset: 0, index: 0)
        var rows = UInt32(rows), columns = UInt32(columns)
        var scale = 1 / sqrt(Float(config.headDim))
        encoder.setBytes(&rows, length: 4, index: 1)
        encoder.setBytes(&columns, length: 4, index: 2)
        encoder.setBytes(&scale, length: 4, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(rows), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(Int(columns), 256), height: 1, depth: 1)
        )
    }

    private func dispatchAddBias(
        _ encoder: MTLComputeCommandEncoder,
        values: MTLBuffer,
        bias: MTLBuffer,
        rows: Int,
        columns: Int
    ) {
        encoder.setComputePipelineState(pipelines.addBias)
        encoder.setBuffer(values, offset: 0, index: 0)
        encoder.setBuffer(bias, offset: 0, index: 1)
        var rows = UInt32(rows), columns = UInt32(columns)
        encoder.setBytes(&rows, length: 4, index: 2)
        encoder.setBytes(&columns, length: 4, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: Int(columns), height: Int(rows), depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(Int(columns), 256), height: 1, depth: 1)
        )
    }

    private func dispatchGEMM(
        _ encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        matrix: SmeltQwen35VisionWeights.Matrix,
        bias: MTLBuffer,
        output: MTLBuffer,
        rows: Int,
        inputSize: Int,
        outputSize: Int
    ) throws {
        guard matrix.shape.reduce(1, *) == inputSize * outputSize else {
            throw SmeltQwen35VisionRuntimeError.matrixShapeMismatch(
                matrix.shape,
                expected: [outputSize, inputSize]
            )
        }
        let pipeline = matrix.dtype == .bf16 ? pipelines.gemmBF16 : pipelines.gemmF16
        guard pipeline.threadExecutionWidth == 32 else {
            throw SmeltQwen35VisionRuntimeError.unsupportedSIMDWidth(
                pipeline.threadExecutionWidth
            )
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(matrix.buffer, offset: 0, index: 1)
        encoder.setBuffer(bias, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var m = UInt32(rows), n = UInt32(outputSize), k = UInt32(inputSize), hasBias = UInt32(1)
        encoder.setBytes(&m, length: 4, index: 4)
        encoder.setBytes(&n, length: 4, index: 5)
        encoder.setBytes(&k, length: 4, index: 6)
        encoder.setBytes(&hasBias, length: 4, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: outputSize, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    private func dispatchAdd(
        _ encoder: MTLComputeCommandEncoder,
        lhs: MTLBuffer,
        rhs: MTLBuffer,
        output: MTLBuffer,
        count: Int
    ) {
        encoder.setComputePipelineState(pipelines.add)
        encoder.setBuffer(lhs, offset: 0, index: 0)
        encoder.setBuffer(rhs, offset: 0, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var count = UInt32(count)
        encoder.setBytes(&count, length: 4, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(Int(count), 256), height: 1, depth: 1)
        )
    }

    private func dispatchLayerNorm(
        _ encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        weight: MTLBuffer,
        bias: MTLBuffer,
        output: MTLBuffer,
        rows: Int,
        dim: Int
    ) {
        encoder.setComputePipelineState(pipelines.layerNorm)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weight, offset: 0, index: 1)
        encoder.setBuffer(bias, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var rows = UInt32(rows), dim = UInt32(dim), epsilon = config.layerNormEpsilon
        encoder.setBytes(&rows, length: 4, index: 4)
        encoder.setBytes(&dim, length: 4, index: 5)
        encoder.setBytes(&epsilon, length: 4, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(rows), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(Int(dim), 256), height: 1, depth: 1)
        )
    }

    private func dispatchSplitRoPE(
        _ encoder: MTLComputeCommandEncoder,
        workspace: Workspace,
        tokens: Int
    ) {
        encoder.setComputePipelineState(pipelines.splitRoPE)
        encoder.setBuffer(workspace.qkv, offset: 0, index: 0)
        encoder.setBuffer(workspace.cosines, offset: 0, index: 1)
        encoder.setBuffer(workspace.sines, offset: 0, index: 2)
        encoder.setBuffer(workspace.q, offset: 0, index: 3)
        encoder.setBuffer(workspace.k, offset: 0, index: 4)
        encoder.setBuffer(workspace.v, offset: 0, index: 5)
        var tokens = UInt32(tokens), heads = UInt32(config.headCount)
        var headDim = UInt32(config.headDim)
        encoder.setBytes(&tokens, length: 4, index: 6)
        encoder.setBytes(&heads, length: 4, index: 7)
        encoder.setBytes(&headDim, length: 4, index: 8)
        encoder.dispatchThreads(
            MTLSize(
                width: config.headDim / 2,
                height: config.headCount,
                depth: Int(tokens)
            ),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    private func dispatchAttention(
        _ encoder: MTLComputeCommandEncoder,
        workspace: Workspace,
        tokens: Int
    ) {
        encoder.setComputePipelineState(pipelines.attention)
        encoder.setBuffer(workspace.q, offset: 0, index: 0)
        encoder.setBuffer(workspace.k, offset: 0, index: 1)
        encoder.setBuffer(workspace.v, offset: 0, index: 2)
        encoder.setBuffer(workspace.chunkStart, offset: 0, index: 3)
        encoder.setBuffer(workspace.chunkEnd, offset: 0, index: 4)
        encoder.setBuffer(workspace.attention, offset: 0, index: 5)
        var tokens = UInt32(tokens), heads = UInt32(config.headCount)
        var headDim = UInt32(config.headDim)
        encoder.setBytes(&tokens, length: 4, index: 6)
        encoder.setBytes(&heads, length: 4, index: 7)
        encoder.setBytes(&headDim, length: 4, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: Int(tokens), height: config.headCount, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    private func dispatchGELU(
        _ encoder: MTLComputeCommandEncoder,
        buffer: MTLBuffer,
        count: Int
    ) {
        encoder.setComputePipelineState(pipelines.geluTanh)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.setBuffer(buffer, offset: 0, index: 1)
        var count = UInt32(count)
        encoder.setBytes(&count, length: 4, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(Int(count), 256), height: 1, depth: 1)
        )
    }

    private func dispatchFiniteCheck(
        _ encoder: MTLComputeCommandEncoder,
        input: MTLBuffer,
        diagnosticBuffer: MTLBuffer,
        count: Int,
        stage: Int
    ) {
        encoder.setComputePipelineState(pipelines.checkFinite)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(diagnosticBuffer, offset: 0, index: 1)
        var count = UInt32(count), stage = UInt32(stage)
        encoder.setBytes(&count, length: 4, index: 2)
        encoder.setBytes(&stage, length: 4, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(Int(count), 256), height: 1, depth: 1)
        )
    }
}

public enum SmeltQwen35VisionRuntimeError: Error, CustomStringConvertible {
    case missingWeight(String)
    case unsupportedWeightDType(String, String)
    case bufferAllocationFailed(String, Int)
    case shaderCompilationFailed(String)
    case pipelineMissing(String)
    case commandBufferUnavailable
    case invalidGrid([SmeltQwen35VisionRuntime.Grid])
    case patchCountMismatch(expected: Int, got: Int)
    case invalidPositionTable([Int])
    case matrixShapeMismatch([Int], expected: [Int])
    case unsupportedSIMDWidth(Int)
    case nonFiniteStage(String)
    case incompatibleDiagnosticsAndProfiling
    case unsupportedProfiledRoute(
        gemm: SmeltQwen35VisionGEMMBackend,
        attention: SmeltQwen35VisionAttentionBackend
    )
    case unsupportedFrozenPlanRoute(
        gemm: SmeltQwen35VisionGEMMBackend,
        attention: SmeltQwen35VisionAttentionBackend
    )

    public var description: String {
        switch self {
        case .missingWeight(let name):
            return "Qwen3.5 vision runtime: missing weight '\(name)'"
        case .unsupportedWeightDType(let name, let dtype):
            return "Qwen3.5 vision runtime: '\(name)' dtype '\(dtype)' is unsupported"
        case .bufferAllocationFailed(let label, let bytes):
            return "Qwen3.5 vision runtime: failed to allocate \(bytes) bytes for '\(label)'"
        case .shaderCompilationFailed(let detail):
            return "Qwen3.5 vision runtime shader compile failed: \(detail)"
        case .pipelineMissing(let name):
            return "Qwen3.5 vision runtime pipeline '\(name)' is missing"
        case .commandBufferUnavailable:
            return "Qwen3.5 vision runtime could not create a command buffer"
        case .invalidGrid(let grids):
            return "Qwen3.5 vision runtime received invalid grids \(grids)"
        case .patchCountMismatch(let expected, let got):
            return "Qwen3.5 vision runtime patch payload has \(got) values; expected \(expected)"
        case .invalidPositionTable(let shape):
            return "Qwen3.5 vision runtime position table shape \(shape) is invalid"
        case .matrixShapeMismatch(let got, let expected):
            return "Qwen3.5 vision runtime matrix shape \(got) != \(expected)"
        case .unsupportedSIMDWidth(let width):
            return "Qwen3.5 vision runtime requires SIMD width 32; got \(width)"
        case .nonFiniteStage(let stage):
            return "Qwen3.5 vision runtime first produced a non-finite value at '\(stage)'"
        case .incompatibleDiagnosticsAndProfiling:
            return "Qwen3.5 vision runtime cannot insert finite-check diagnostics into a frozen-operation profile"
        case .unsupportedProfiledRoute(let gemm, let attention):
            return "Qwen3.5 vision runtime profiling requires gemm=mps and attention=mps-staged; got \(gemm)/\(attention)"
        case .unsupportedFrozenPlanRoute(let gemm, let attention):
            return "Qwen3.5 vision frozen cost plans currently describe gemm=mps and attention=mps-staged; got \(gemm)/\(attention)"
        }
    }
}
