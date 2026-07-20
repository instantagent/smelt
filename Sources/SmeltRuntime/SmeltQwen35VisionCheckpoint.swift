import Foundation
import SmeltSchema

/// The dense vision-transformer shape derived from a selected multimodal
/// module region. The name reflects the source checkpoint adapter; selection
/// itself is structural and never switches on a module id or semantic hash.
public struct SmeltQwen35VisionConfig: Codable, Sendable, Equatable {
    public let hiddenSize: Int
    public let layerCount: Int
    public let headCount: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let layerNormEpsilon: Float
    public let inChannels: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let positionEmbeddingCount: Int
    public let spatialMergeSize: Int
    public let outputHiddenSize: Int
    public let activation: String

    public init(
        hiddenSize: Int,
        layerCount: Int,
        headCount: Int,
        headDim: Int,
        intermediateSize: Int,
        layerNormEpsilon: Float,
        inChannels: Int,
        patchSize: Int,
        temporalPatchSize: Int,
        positionEmbeddingCount: Int,
        spatialMergeSize: Int,
        outputHiddenSize: Int,
        activation: String
    ) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.headDim = headDim
        self.intermediateSize = intermediateSize
        self.layerNormEpsilon = layerNormEpsilon
        self.inChannels = inChannels
        self.patchSize = patchSize
        self.temporalPatchSize = temporalPatchSize
        self.positionEmbeddingCount = positionEmbeddingCount
        self.spatialMergeSize = spatialMergeSize
        self.outputHiddenSize = outputHiddenSize
        self.activation = activation
    }
}

public struct SmeltQwen35VisionCheckpointPlan: Sendable, Equatable {
    public struct ExpectedTensor: Sendable, Equatable {
        public let name: String
        public let shape: [Int]

        public init(name: String, shape: [Int]) {
            self.name = name
            self.shape = shape
        }
    }

    public struct Tensor: Sendable, Equatable {
        public let descriptorIndex: Int
        public let name: String
        public let dtype: String
        public let shape: [Int]

        public init(descriptorIndex: Int, name: String, dtype: String, shape: [Int]) {
            self.descriptorIndex = descriptorIndex
            self.name = name
            self.dtype = dtype
            self.shape = shape
        }
    }

    public let config: SmeltQwen35VisionConfig
    public let sourceID: String
    public let tensors: [Tensor]

    public init(module: SmeltCAMIR, checkpoint: any CheckpointTensorSource) throws {
        let descriptor = try SmeltCAMPackageDescriptor(from: module)
        let capabilities = try SmeltCAMPackageCapabilities(descriptor: descriptor)
        let decision = try capabilities.resolve(.runMultimodalText)
        let bindingPlan = try SmeltModuleRuntimeBindingPlan(
            capabilities: capabilities,
            decision: decision
        )
        let graph = bindingPlan.graph

        let encoderBlocks = graph.blocks.filter { $0.operatorName == "transformer-encoder" }
        let mergerBlocks = graph.blocks.filter { $0.operatorName == "adapter" }
        guard encoderBlocks.count == 1, mergerBlocks.count == 1,
              let encoderBlock = encoderBlocks.first,
              let mergerBlock = mergerBlocks.first
        else {
            throw SmeltQwen35VisionCheckpointError.unsupportedGraph(
                "multimodal region needs one transformer-encoder and one adapter block"
            )
        }

        let compiledBlockIDs = Set(
            graph.nodes.compactMap { node in
                node.implementation == "compiled" ? node.blockID : nil
            }
        )
        guard compiledBlockIDs.contains(encoderBlock.blockID),
              compiledBlockIDs.contains(mergerBlock.blockID)
        else {
            throw SmeltQwen35VisionCheckpointError.unsupportedGraph(
                "vision blocks are not both executable in the selected region"
            )
        }

        config = try Self.config(encoder: encoderBlock, merger: mergerBlock)
        let ownerIDs = Set([encoderBlock.blockID, mergerBlock.blockID])
        let bindings = graph.tensorBindings.filter { ownerIDs.contains($0.ownerBlockID) }
        guard !bindings.isEmpty else {
            throw SmeltQwen35VisionCheckpointError.unsupportedGraph(
                "selected vision blocks have no tensor ownership bindings"
            )
        }
        let sourceIDs = Set(bindings.map(\.sourceID))
        guard sourceIDs.count == 1, let sourceID = sourceIDs.first else {
            throw SmeltQwen35VisionCheckpointError.unsupportedGraph(
                "selected vision tensors must come from one checkpoint source"
            )
        }
        self.sourceID = sourceID

        let patterns = bindings.map { binding -> String in
            binding.tensorPattern.pattern
        }
        let selected = checkpoint.checkpointTensors.filter { tensor in
            patterns.contains { Self.glob($0, matches: tensor.name) }
        }
        let expected = Self.expectedTensors(config: config)
        let expectedByName = Dictionary(uniqueKeysWithValues: expected.map { ($0.name, $0) })
        let selectedByName = Dictionary(grouping: selected, by: \.name)

        let duplicateNames = selectedByName.filter { $0.value.count != 1 }.map(\.key).sorted()
        guard duplicateNames.isEmpty else {
            throw SmeltQwen35VisionCheckpointError.duplicateTensors(duplicateNames)
        }

        let selectedNames = Set(selectedByName.keys)
        let expectedNames = Set(expectedByName.keys)
        let missing = expectedNames.subtracting(selectedNames).sorted()
        let unexpected = selectedNames.subtracting(expectedNames).sorted()
        guard missing.isEmpty, unexpected.isEmpty else {
            throw SmeltQwen35VisionCheckpointError.coverage(
                missing: missing,
                unexpected: unexpected
            )
        }

        var resolved: [Tensor] = []
        resolved.reserveCapacity(expected.count)
        for expectedTensor in expected.sorted(by: { $0.name < $1.name }) {
            guard let descriptor = selectedByName[expectedTensor.name]?.first else {
                throw SmeltQwen35VisionCheckpointError.coverage(
                    missing: [expectedTensor.name],
                    unexpected: []
                )
            }
            guard descriptor.shape == expectedTensor.shape else {
                throw SmeltQwen35VisionCheckpointError.shapeMismatch(
                    expectedTensor.name,
                    expected: expectedTensor.shape,
                    got: descriptor.shape
                )
            }
            guard descriptor.dtype == "BF16" || descriptor.dtype == "F16" else {
                throw SmeltQwen35VisionCheckpointError.unsupportedDType(
                    expectedTensor.name,
                    descriptor.dtype
                )
            }
            resolved.append(
                Tensor(
                    descriptorIndex: descriptor.index,
                    name: descriptor.name,
                    dtype: descriptor.dtype,
                    shape: descriptor.shape
                )
            )
        }
        tensors = resolved
    }

    public static func expectedTensors(
        config c: SmeltQwen35VisionConfig
    ) -> [ExpectedTensor] {
        let hidden = c.hiddenSize
        let qkv = hidden * 3
        let intermediate = c.intermediateSize
        let merged = hidden * c.spatialMergeSize * c.spatialMergeSize
        var result: [ExpectedTensor] = [
            .init(
                name: "model.visual.patch_embed.proj.weight",
                shape: [hidden, c.inChannels, c.temporalPatchSize, c.patchSize, c.patchSize]
            ),
            .init(name: "model.visual.patch_embed.proj.bias", shape: [hidden]),
            .init(
                name: "model.visual.pos_embed.weight",
                shape: [c.positionEmbeddingCount, hidden]
            ),
        ]
        for layer in 0..<c.layerCount {
            let prefix = "model.visual.blocks.\(layer)"
            result.append(contentsOf: [
                .init(name: "\(prefix).norm1.weight", shape: [hidden]),
                .init(name: "\(prefix).norm1.bias", shape: [hidden]),
                .init(name: "\(prefix).attn.qkv.weight", shape: [qkv, hidden]),
                .init(name: "\(prefix).attn.qkv.bias", shape: [qkv]),
                .init(name: "\(prefix).attn.proj.weight", shape: [hidden, hidden]),
                .init(name: "\(prefix).attn.proj.bias", shape: [hidden]),
                .init(name: "\(prefix).norm2.weight", shape: [hidden]),
                .init(name: "\(prefix).norm2.bias", shape: [hidden]),
                .init(name: "\(prefix).mlp.linear_fc1.weight", shape: [intermediate, hidden]),
                .init(name: "\(prefix).mlp.linear_fc1.bias", shape: [intermediate]),
                .init(name: "\(prefix).mlp.linear_fc2.weight", shape: [hidden, intermediate]),
                .init(name: "\(prefix).mlp.linear_fc2.bias", shape: [hidden]),
            ])
        }
        result.append(contentsOf: [
            .init(name: "model.visual.merger.norm.weight", shape: [hidden]),
            .init(name: "model.visual.merger.norm.bias", shape: [hidden]),
            .init(name: "model.visual.merger.linear_fc1.weight", shape: [merged, merged]),
            .init(name: "model.visual.merger.linear_fc1.bias", shape: [merged]),
            .init(name: "model.visual.merger.linear_fc2.weight", shape: [c.outputHiddenSize, merged]),
            .init(name: "model.visual.merger.linear_fc2.bias", shape: [c.outputHiddenSize]),
        ])
        return result
    }

    private static func config(
        encoder: SmeltCAMPackageDescriptor.Block,
        merger: SmeltCAMPackageDescriptor.Block
    ) throws -> SmeltQwen35VisionConfig {
        guard let transformer = encoder.shape.transformer,
              let hidden = transformer.hiddenSize,
              let layers = transformer.layers,
              let attention = transformer.attention,
              let ffn = transformer.ffn,
              let norm = transformer.norm,
              attention.qHeads == attention.kvHeads,
              attention.qHeads * attention.headDim == hidden,
              norm.normType == "layer",
              let epsilonText = norm.eps,
              let epsilon = Float(epsilonText)
        else {
            throw SmeltQwen35VisionCheckpointError.unsupportedGraph(
                "vision encoder is not a dense pre-layernorm transformer"
            )
        }
        let layerCount: Int
        if let count = layers.count {
            layerCount = count
        } else if let repeatCount = layers.repeatCount, !layers.roles.isEmpty {
            layerCount = layers.roles.count * repeatCount
        } else {
            throw SmeltQwen35VisionCheckpointError.unsupportedGraph(
                "vision encoder layer count is unresolved"
            )
        }
        let encoderRequirements = try requirementMap(encoder.shape.requirements)
        let mergerRequirements = try requirementMap(merger.shape.requirements)
        let activation = try required("activation", in: encoderRequirements)
        let mergerActivation = try required("activation", in: mergerRequirements)
        guard activation == mergerActivation else {
            throw SmeltQwen35VisionCheckpointError.unsupportedGraph(
                "vision encoder and merger activations disagree"
            )
        }
        guard ffn.activation == "gelu" else {
            throw SmeltQwen35VisionCheckpointError.unsupportedGraph(
                "vision encoder FFN must declare gelu"
            )
        }
        return SmeltQwen35VisionConfig(
            hiddenSize: hidden,
            layerCount: layerCount,
            headCount: attention.qHeads,
            headDim: attention.headDim,
            intermediateSize: ffn.dim,
            layerNormEpsilon: epsilon,
            inChannels: try requiredInt("in-channels", in: encoderRequirements),
            patchSize: try requiredInt("patch-size", in: encoderRequirements),
            temporalPatchSize: try requiredInt("temporal-patch-size", in: encoderRequirements),
            positionEmbeddingCount: try requiredInt("position-embeddings", in: encoderRequirements),
            spatialMergeSize: try requiredInt("spatial-merge-size", in: encoderRequirements),
            outputHiddenSize: try requiredInt("out-hidden-size", in: mergerRequirements),
            activation: activation
        )
    }

    private static func requirementMap(
        _ requirements: [SmeltCAMPackageDescriptor.BlockRequirement]
    ) throws -> [String: String] {
        var result: [String: String] = [:]
        for requirement in requirements {
            guard result[requirement.key] == nil else {
                throw SmeltQwen35VisionCheckpointError.unsupportedGraph(
                    "duplicate block requirement '\(requirement.key)'"
                )
            }
            result[requirement.key] = requirement.value ?? ""
        }
        return result
    }

    private static func required(_ key: String, in values: [String: String]) throws -> String {
        guard let value = values[key] else {
            throw SmeltQwen35VisionCheckpointError.missingRequirement(key)
        }
        return value
    }

    private static func requiredInt(_ key: String, in values: [String: String]) throws -> Int {
        let value = try required(key, in: values)
        guard let result = Int(value), result > 0 else {
            throw SmeltQwen35VisionCheckpointError.invalidRequirement(key, value)
        }
        return result
    }

    private static func glob(_ pattern: String, matches value: String) -> Bool {
        if pattern == "*" { return true }
        if !pattern.contains("*") { return pattern == value }
        let parts = pattern.split(separator: "*", omittingEmptySubsequences: false).map(String.init)
        var remainder = value[...]
        if let first = parts.first, !first.isEmpty {
            guard remainder.hasPrefix(first) else { return false }
            remainder.removeFirst(first.count)
        }
        for part in parts.dropFirst().dropLast() where !part.isEmpty {
            guard let range = remainder.range(of: part) else { return false }
            remainder = remainder[range.upperBound...]
        }
        if let last = parts.last, !last.isEmpty {
            return remainder.hasSuffix(last)
        }
        return true
    }
}

public enum SmeltQwen35VisionCheckpointError: Error, CustomStringConvertible, Equatable {
    case unsupportedGraph(String)
    case missingRequirement(String)
    case invalidRequirement(String, String)
    case duplicateTensors([String])
    case coverage(missing: [String], unexpected: [String])
    case shapeMismatch(String, expected: [Int], got: [Int])
    case unsupportedDType(String, String)

    public var description: String {
        switch self {
        case .unsupportedGraph(let reason):
            return "Qwen3.5 vision checkpoint plan: \(reason)"
        case .missingRequirement(let key):
            return "Qwen3.5 vision checkpoint plan: missing block requirement '\(key)'"
        case .invalidRequirement(let key, let value):
            return "Qwen3.5 vision checkpoint plan: invalid block requirement '\(key)=\(value)'"
        case .duplicateTensors(let names):
            return "Qwen3.5 vision checkpoint plan: duplicate tensors \(names)"
        case .coverage(let missing, let unexpected):
            return "Qwen3.5 vision checkpoint plan: coverage mismatch; missing=\(missing), unexpected=\(unexpected)"
        case .shapeMismatch(let name, let expected, let got):
            return "Qwen3.5 vision checkpoint plan: '\(name)' shape \(got) != \(expected)"
        case .unsupportedDType(let name, let dtype):
            return "Qwen3.5 vision checkpoint plan: '\(name)' dtype '\(dtype)' is unsupported"
        }
    }
}
