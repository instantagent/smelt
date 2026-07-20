import Foundation
import Metal

/// Checkpoint-backed skeleton-language model execution through Smelt's compiled
/// embeddings-in/hidden-out trunk plus the tied BF16 language-model head.
public final class SmeltSkeletonLanguageRuntime {
    public struct PrefillOutput: Sendable {
        public let hiddenStates: [Float]
        public let finalLogits: [Float]

        public init(hiddenStates: [Float], finalLogits: [Float]) {
            self.hiddenStates = hiddenStates
            self.finalLogits = finalLogits
        }
    }

    public struct DecodeOutput: Sendable {
        public let hiddenState: [Float]
        public let logits: [Float]

        public init(hiddenState: [Float], logits: [Float]) {
            self.hiddenState = hiddenState
            self.logits = logits
        }
    }

    public let artifact: SmeltRigArtifact
    public let trunk: SmeltRuntime
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let dense: MTLComputePipelineState
    private let gather: MTLComputePipelineState
    private let tokenEmbeddings: MTLBuffer
    private let languageModelHead: MTLBuffer

    public init(
        packagePath: String,
        device requestedDevice: MTLDevice? = nil,
        verifyPackage: Bool = true
    ) throws {
        guard let device = requestedDevice ?? MTLCreateSystemDefaultDevice() else {
            throw SmeltSkeletonLanguageRuntimeError.metalUnavailable
        }
        guard let queue = device.makeCommandQueue() else {
            throw SmeltSkeletonLanguageRuntimeError.commandQueueCreationFailed
        }
        let artifact = try SmeltRigArtifact(path: packagePath, verify: false)
        let trunk = try SmeltRuntime(
            packagePath: "\(packagePath)/\(artifact.manifest.files.languageTrunk)",
            device: device,
            verifyPackage: verifyPackage,
            contextLimit: artifact.manifest.configuration.languageMaximumPositions
        )
        let library = try artifact.makeLibrary(device: device)
        let dense = try Self.makePipeline(
            named: "dense_bf16",
            library: library,
            device: device
        )
        let gather = try Self.makePipeline(
            named: "gather_row_bf16",
            library: library,
            device: device
        )
        let embeddings = try artifact.makeWeightBuffer(
            device: device,
            tensorNamed: "transformer.model.embed_tokens.weight"
        )
        let head = try artifact.makeWeightBuffer(
            device: device,
            tensorNamed: "transformer.lm_head.weight"
        )
        self.artifact = artifact
        self.trunk = trunk
        self.device = device
        self.queue = queue
        self.dense = dense
        self.gather = gather
        tokenEmbeddings = embeddings
        languageModelHead = head
    }

    /// Gathers one tied BF16 token embedding and advances the compiled trunk's
    /// existing KV cache at the supplied absolute position.
    public func decodeTeacherForced(
        tokenID: Int,
        position: Int
    ) throws -> DecodeOutput {
        let configuration = artifact.manifest.configuration
        guard tokenID >= 0, tokenID < configuration.languageVocabularySize else {
            throw SmeltSkeletonLanguageRuntimeError.invalidTokenID(
                tokenID,
                vocabularySize: configuration.languageVocabularySize
            )
        }
        guard position >= 0, position < configuration.languageMaximumPositions else {
            throw SmeltSkeletonLanguageRuntimeError.invalidPosition(
                position,
                maximumPositions: configuration.languageMaximumPositions
            )
        }
        try trunk.ensureContextCapacity(position + 1)
        let hiddenInput = try trunk.portSlotBuffer("hiddenA")
        let token = try makeTokenBuffer(UInt32(tokenID))
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw SmeltSkeletonLanguageRuntimeError.commandEncodingFailed
        }
        encoder.setComputePipelineState(gather)
        encoder.setBuffer(tokenEmbeddings, offset: 0, index: 0)
        encoder.setBuffer(token, offset: 0, index: 1)
        encoder.setBuffer(hiddenInput, offset: 0, index: 2)
        var hiddenSize = UInt32(configuration.languageHiddenSize)
        var slot: UInt32 = 0
        encoder.setBytes(&hiddenSize, length: 4, index: 3)
        encoder.setBytes(&slot, length: 4, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: configuration.languageHiddenSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(
                    configuration.languageHiddenSize,
                    gather.maxTotalThreadsPerThreadgroup
                ),
                height: 1,
                depth: 1
            )
        )
        try trunk.encodeTrunkDecode(
            into: encoder,
            tokenId: Int32(tokenID),
            position: Int32(position)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw SmeltSkeletonLanguageRuntimeError.gpuExecutionFailed("\(error)")
        }
        let normalized = try trunk.portSlotBuffer("normOutBuf")
        let hidden = SmeltBF16.decode(
            normalized.contents().bindMemory(
                to: UInt16.self,
                capacity: configuration.languageHiddenSize
            ),
            count: configuration.languageHiddenSize
        )
        return DecodeOutput(
            hiddenState: hidden,
            logits: try projectLogits(hidden: hidden)
        )
    }

    /// Exactness probe that advances one teacher-forced decode while retaining
    /// every dense-trunk layer boundary.
    func decodeTeacherForcedCapturingLayers(
        tokenID: Int,
        position: Int
    ) throws -> (output: DecodeOutput, capture: SmeltTrunkLayerCapture) {
        try gatherTokenEmbeddingForCapture(tokenID: tokenID, position: position)
        let capture = try trunk.captureTrunkDecodeLayerOutputs(
            tokenId: Int32(tokenID),
            position: Int32(position)
        )
        let output = DecodeOutput(
            hiddenState: capture.finalHiddenStates,
            logits: try projectLogits(hidden: capture.finalHiddenStates)
        )
        return (output, capture)
    }

    /// Exactness probe that advances one teacher-forced decode while retaining
    /// requested dense-trunk dispatch boundaries.
    func decodeTeacherForcedCapturingDispatches(
        tokenID: Int,
        position: Int,
        requests: [SmeltTrunkDispatchCaptureRequest]
    ) throws -> [String: [Float]] {
        try gatherTokenEmbeddingForCapture(tokenID: tokenID, position: position)
        return try trunk.captureTrunkDecodeDispatchOutputs(
            tokenId: Int32(tokenID),
            position: Int32(position),
            requests: requests
        )
    }

    /// Runs a single compiled prefill and projects its final normalized hidden
    /// row through the tied BF16 head. Input embeddings cross the public API as
    /// FP32 values and narrow once at the trunk's declared BF16 storage port.
    public func prefill(
        embeddings: [Float],
        sequenceLength: Int
    ) throws -> PrefillOutput {
        let configuration = artifact.manifest.configuration
        guard sequenceLength > 0,
              embeddings.count == sequenceLength * configuration.languageHiddenSize
        else {
            throw SmeltSkeletonLanguageRuntimeError.invalidEmbeddingShape(
                count: embeddings.count,
                sequenceLength: sequenceLength,
                hiddenSize: configuration.languageHiddenSize
            )
        }
        let hidden = try trunk.prefillTrunk(
            embeddings: embeddings,
            seqLen: sequenceLength
        )
        let lastStart = (sequenceLength - 1) * configuration.languageHiddenSize
        let lastHidden = Array(
            hidden[lastStart..<(lastStart + configuration.languageHiddenSize)]
        )
        return PrefillOutput(
            hiddenStates: hidden,
            finalLogits: try projectLogits(hidden: lastHidden)
        )
    }

    /// Returns checkpoint-authored BF16 token rows widened exactly to FP32.
    /// No arithmetic or GPU round trip occurs at this boundary.
    public func embeddings(tokenIDs: [Int]) throws -> [Float] {
        let configuration = artifact.manifest.configuration
        guard tokenIDs.allSatisfy({
            $0 >= 0 && $0 < configuration.languageVocabularySize
        }) else {
            let invalid = tokenIDs.first {
                $0 < 0 || $0 >= configuration.languageVocabularySize
            } ?? -1
            throw SmeltSkeletonLanguageRuntimeError.invalidTokenID(
                invalid,
                vocabularySize: configuration.languageVocabularySize
            )
        }
        guard let descriptor = artifact.checkpointTensors.first(where: {
            $0.name == "transformer.model.embed_tokens.weight"
        }), descriptor.dtype == "BF16",
            descriptor.shape == [
                configuration.languageVocabularySize,
                configuration.languageHiddenSize,
            ]
        else {
            throw SmeltSkeletonLanguageRuntimeError.invalidEmbeddingTable
        }
        let source = artifact.checkpointTensorData(descriptor).bindMemory(
            to: UInt16.self,
            capacity: configuration.languageVocabularySize * configuration.languageHiddenSize
        )
        var result: [Float] = []
        result.reserveCapacity(tokenIDs.count * configuration.languageHiddenSize)
        for tokenID in tokenIDs {
            let row = tokenID * configuration.languageHiddenSize
            for column in 0..<configuration.languageHiddenSize {
                result.append(
                    SmeltBF16.decode(source[row + column])
                )
            }
        }
        return result
    }

    /// Applies the tied BF16 output projection to one normalized hidden row.
    public func projectLogits(hidden: [Float]) throws -> [Float] {
        let configuration = artifact.manifest.configuration
        guard hidden.count == configuration.languageHiddenSize else {
            throw SmeltSkeletonLanguageRuntimeError.invalidHiddenSize(
                expected: configuration.languageHiddenSize,
                got: hidden.count
            )
        }
        let input = try makeBF16Buffer(hidden, label: "rig.language.head.input")
        guard let output = device.makeBuffer(
            length: configuration.languageVocabularySize * MemoryLayout<UInt16>.stride,
            options: .storageModeShared
        ) else {
            throw SmeltSkeletonLanguageRuntimeError.bufferCreationFailed(
                "rig.language.head.output"
            )
        }
        output.label = "rig.language.head.output"
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw SmeltSkeletonLanguageRuntimeError.commandEncodingFailed
        }
        encoder.setComputePipelineState(dense)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(languageModelHead, offset: 0, index: 1)
        encoder.setBuffer(languageModelHead, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var rows: UInt32 = 1
        var outputDimension = UInt32(configuration.languageVocabularySize)
        var inputDimension = UInt32(configuration.languageHiddenSize)
        var hasBias: UInt32 = 0
        encoder.setBytes(&rows, length: 4, index: 4)
        encoder.setBytes(&outputDimension, length: 4, index: 5)
        encoder.setBytes(&inputDimension, length: 4, index: 6)
        encoder.setBytes(&hasBias, length: 4, index: 7)
        encoder.dispatchThreadgroups(
            MTLSize(width: configuration.languageVocabularySize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw SmeltSkeletonLanguageRuntimeError.gpuExecutionFailed("\(error)")
        }
        return SmeltBF16.decode(
            output.contents().bindMemory(
                to: UInt16.self,
                capacity: configuration.languageVocabularySize
            ),
            count: configuration.languageVocabularySize
        )
    }

    private func makeBF16Buffer(_ values: [Float], label: String) throws -> MTLBuffer {
        let encoded = SmeltBF16.encode(values)
        let buffer = try encoded.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress,
                  let buffer = device.makeBuffer(
                      bytes: base,
                      length: bytes.count,
                      options: .storageModeShared
                  )
            else {
                throw SmeltSkeletonLanguageRuntimeError.bufferCreationFailed(label)
            }
            return buffer
        }
        buffer.label = label
        return buffer
    }

    private func gatherTokenEmbeddingForCapture(
        tokenID: Int,
        position: Int
    ) throws {
        let configuration = artifact.manifest.configuration
        guard tokenID >= 0, tokenID < configuration.languageVocabularySize else {
            throw SmeltSkeletonLanguageRuntimeError.invalidTokenID(
                tokenID,
                vocabularySize: configuration.languageVocabularySize
            )
        }
        guard position >= 0, position < configuration.languageMaximumPositions else {
            throw SmeltSkeletonLanguageRuntimeError.invalidPosition(
                position,
                maximumPositions: configuration.languageMaximumPositions
            )
        }
        try trunk.ensureContextCapacity(position + 1)
        let hiddenInput = try trunk.portSlotBuffer("hiddenA")
        let token = try makeTokenBuffer(UInt32(tokenID))
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw SmeltSkeletonLanguageRuntimeError.commandEncodingFailed
        }
        encoder.setComputePipelineState(gather)
        encoder.setBuffer(tokenEmbeddings, offset: 0, index: 0)
        encoder.setBuffer(token, offset: 0, index: 1)
        encoder.setBuffer(hiddenInput, offset: 0, index: 2)
        var hiddenSize = UInt32(configuration.languageHiddenSize)
        var slot: UInt32 = 0
        encoder.setBytes(&hiddenSize, length: 4, index: 3)
        encoder.setBytes(&slot, length: 4, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: configuration.languageHiddenSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(
                    configuration.languageHiddenSize,
                    gather.maxTotalThreadsPerThreadgroup
                ),
                height: 1,
                depth: 1
            )
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw SmeltSkeletonLanguageRuntimeError.gpuExecutionFailed("\(error)")
        }
    }

    private func makeTokenBuffer(_ token: UInt32) throws -> MTLBuffer {
        var token = token
        guard let buffer = device.makeBuffer(
            bytes: &token,
            length: MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ) else {
            throw SmeltSkeletonLanguageRuntimeError.bufferCreationFailed(
                "rig.language.teacher-token"
            )
        }
        buffer.label = "rig.language.teacher-token"
        return buffer
    }

    private static func makePipeline(
        named name: String,
        library: MTLLibrary,
        device: MTLDevice
    ) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: name) else {
            throw SmeltSkeletonLanguageRuntimeError.pipelineMissing(name)
        }
        do {
            return try device.makeComputePipelineState(function: function)
        } catch {
            throw SmeltSkeletonLanguageRuntimeError.pipelineCreationFailed(name, "\(error)")
        }
    }
}

public enum SmeltSkeletonLanguageRuntimeError: Error, Equatable {
    case metalUnavailable
    case commandQueueCreationFailed
    case pipelineMissing(String)
    case pipelineCreationFailed(String, String)
    case invalidEmbeddingShape(count: Int, sequenceLength: Int, hiddenSize: Int)
    case invalidHiddenSize(expected: Int, got: Int)
    case invalidTokenID(Int, vocabularySize: Int)
    case invalidPosition(Int, maximumPositions: Int)
    case invalidEmbeddingTable
    case bufferCreationFailed(String)
    case commandEncodingFailed
    case gpuExecutionFailed(String)
}
