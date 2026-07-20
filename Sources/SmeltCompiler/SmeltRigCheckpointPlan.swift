import SmeltRuntime
import Foundation

/// Exact carried-versus-training-only policy for the pinned rig model checkpoint.
public struct SmeltRigCheckpointPlan: Sendable {
    /// Whether a checkpoint tensor participates in rig inference.
    public enum Disposition: String, Sendable, Equatable {
        case carried
        case trainingOnly
    }

    /// Network surface that owns a tensor.
    public enum Component: String, Sendable, Equatable {
        case vae
        case meshEncoder
        case qwen
        case outputProjection
    }

    /// Immutable tensor contract derived from the pinned upstream source.
    public struct ExpectedTensor: Sendable, Equatable {
        public let name: String
        public let shape: [Int]
        public let component: Component
        public let disposition: Disposition

        public init(
            name: String,
            shape: [Int],
            component: Component,
            disposition: Disposition
        ) {
            self.name = name
            self.shape = shape
            self.component = component
            self.disposition = disposition
        }
    }

    /// Validated source tensor ready for zero-copy package copying.
    public struct Tensor: Sendable, Equatable {
        public let descriptorIndex: Int
        public let name: String
        public let shape: [Int]
        public let byteCount: Int
        public let storageKey: String
        public let storageOffset: Int
        public let component: Component
        public let disposition: Disposition
    }

    /// Every validated checkpoint tensor, including explicit training-only rows.
    public let tensors: [Tensor]
    /// The inference-active subset written to a rig model package.
    public var carriedTensors: [Tensor] {
        tensors.filter { $0.disposition == .carried }
    }
    /// Explicitly audited tensors omitted from rig inference packages.
    public var trainingOnlyTensors: [Tensor] {
        tensors.filter { $0.disposition == .trainingOnly }
    }

    /// Validates exact name, shape, BF16 dtype, dense byte length, and the tied
    /// Qwen embedding/head alias before exposing a carried tensor.
    public init(checkpoint: PyTorchCheckpointLoader) throws {
        let expected = Self.expectedTensors
        let sourceByName = Dictionary(grouping: checkpoint.checkpointTensors, by: \.name)
        let duplicates = sourceByName.filter { $0.value.count != 1 }.map(\.key).sorted()
        guard duplicates.isEmpty else {
            throw SmeltRigCheckpointPlanError.duplicateTensors(duplicates)
        }
        let expectedNames = Set(expected.map(\.name))
        let sourceNames = Set(sourceByName.keys)
        let missing = expectedNames.subtracting(sourceNames).sorted()
        let unexpected = sourceNames.subtracting(expectedNames).sorted()
        guard missing.isEmpty, unexpected.isEmpty else {
            throw SmeltRigCheckpointPlanError.coverage(
                missing: missing,
                unexpected: unexpected
            )
        }

        let sourceInfo = Dictionary(
            uniqueKeysWithValues: checkpoint.tensors.map { ($0.name, $0) }
        )
        var tensors: [Tensor] = []
        tensors.reserveCapacity(expected.count)
        for item in expected {
            guard let descriptor = sourceByName[item.name]?.first else {
                throw SmeltRigCheckpointPlanError.coverage(
                    missing: [item.name],
                    unexpected: []
                )
            }
            guard descriptor.dtype == "BF16" else {
                throw SmeltRigCheckpointPlanError.dtype(
                    item.name,
                    expected: "BF16",
                    got: descriptor.dtype
                )
            }
            guard descriptor.shape == item.shape else {
                throw SmeltRigCheckpointPlanError.shape(
                    item.name,
                    expected: item.shape,
                    got: descriptor.shape
                )
            }
            let expectedBytes = item.shape.reduce(2, *)
            guard descriptor.byteCount == expectedBytes else {
                throw SmeltRigCheckpointPlanError.byteCount(
                    item.name,
                    expected: expectedBytes,
                    got: descriptor.byteCount
                )
            }
            guard let info = sourceInfo[item.name] else {
                throw SmeltRigCheckpointPlanError.coverage(
                    missing: [item.name],
                    unexpected: []
                )
            }
            tensors.append(
                Tensor(
                    descriptorIndex: descriptor.index,
                    name: item.name,
                    shape: item.shape,
                    byteCount: descriptor.byteCount,
                    storageKey: info.storageKey,
                    storageOffset: info.storageOffset,
                    component: item.component,
                    disposition: item.disposition
                )
            )
        }

        guard let embedding = sourceInfo["transformer.model.embed_tokens.weight"],
              let head = sourceInfo["transformer.lm_head.weight"],
              embedding.storageKey == head.storageKey,
              embedding.storageOffset == head.storageOffset
        else {
            throw SmeltRigCheckpointPlanError.untiedLanguageModelHead
        }
        self.tensors = tensors
    }

    /// Full pinned checkpoint inventory. Training-only exclusions are encoded
    /// here beside active tensors so an upstream architecture change cannot be
    /// silently dropped by a prefix filter.
    public static let expectedTensors: [ExpectedTensor] = {
        var result: [ExpectedTensor] = []
        result += vaeEncoder(prefix: "vae.model.encoder", learned: true, carried: false)
        result += vaeEncoder(prefix: "vae.model.cond_encoder", learned: false, carried: true)
        result += vaeDecoder()
        result += [
            tensor("vae.model.cond_quant.weight", [512, 768], .vae, .carried),
            tensor("vae.model.cond_quant.bias", [512], .vae, .carried),
            tensor("vae.model.quant.weight", [512, 768], .vae, .trainingOnly),
            tensor("vae.model.quant.bias", [512], .vae, .trainingOnly),
            tensor("vae.model.post_quant.weight", [768, 512], .vae, .carried),
            tensor("vae.model.post_quant.bias", [768], .vae, .carried),
            tensor("vae.model.FSQ.project_in.weight", [5, 512], .vae, .trainingOnly),
            tensor("vae.model.FSQ.project_in.bias", [5], .vae, .trainingOnly),
            tensor("vae.model.FSQ.project_out.weight", [512, 5], .vae, .carried),
            tensor("vae.model.FSQ.project_out.bias", [512], .vae, .carried),
        ]
        result += meshEncoder()
        result += qwen()
        result += [
            tensor("output_proj.0.weight", [896, 512], .outputProjection, .carried),
            tensor("output_proj.0.bias", [896], .outputProjection, .carried),
            tensor("output_proj.1.weight", [896], .outputProjection, .carried),
        ]
        precondition(result.count == 672, "rig model expected inventory must contain 672 tensors")
        return result
    }()

    private static func tensor(
        _ name: String,
        _ shape: [Int],
        _ component: Component,
        _ disposition: Disposition
    ) -> ExpectedTensor {
        ExpectedTensor(
            name: name,
            shape: shape,
            component: component,
            disposition: disposition
        )
    }

    private static func pair(
        _ prefix: String,
        weight: [Int],
        bias: [Int],
        component: Component,
        disposition: Disposition
    ) -> [ExpectedTensor] {
        [
            tensor("\(prefix).weight", weight, component, disposition),
            tensor("\(prefix).bias", bias, component, disposition),
        ]
    }

    private static func vaeEncoder(
        prefix: String,
        learned: Bool,
        carried: Bool
    ) -> [ExpectedTensor] {
        let disposition: Disposition = carried ? .carried : .trainingOnly
        var result: [ExpectedTensor] = []
        if learned {
            result.append(tensor("\(prefix).learned_queries", [32, 768], .vae, disposition))
        }
        result += pair(
            "\(prefix).proj_in",
            weight: [768, learned ? 55 : 54],
            bias: [768],
            component: .vae,
            disposition: disposition
        )
        result += vaeCrossBlock("\(prefix).blocks.0", disposition: disposition)
        result += vaeSelfBlock("\(prefix).blocks.1", disposition: disposition)
        result += vaeSelfBlock("\(prefix).blocks.2", disposition: disposition)
        result += pair(
            "\(prefix).norm_out",
            weight: [768],
            bias: [768],
            component: .vae,
            disposition: disposition
        )
        return result
    }

    private static func vaeDecoder() -> [ExpectedTensor] {
        var result: [ExpectedTensor] = []
        for layer in 0..<10 {
            result += vaeSelfBlock("vae.model.decoder.blocks.\(layer)", disposition: .carried)
        }
        result += vaeCrossBlock("vae.model.decoder.blocks.10", disposition: .carried)
        result += pair(
            "vae.model.decoder.proj_query",
            weight: [768, 54],
            bias: [768],
            component: .vae,
            disposition: .carried
        )
        result += pair(
            "vae.model.decoder.norm_out",
            weight: [768],
            bias: [768],
            component: .vae,
            disposition: .carried
        )
        result += pair(
            "vae.model.decoder.proj_out",
            weight: [1, 768],
            bias: [1],
            component: .vae,
            disposition: .carried
        )
        return result
    }

    private static func vaeSelfBlock(
        _ prefix: String,
        disposition: Disposition
    ) -> [ExpectedTensor] {
        var result = pair(
            "\(prefix).norm1",
            weight: [768],
            bias: [768],
            component: .vae,
            disposition: disposition
        )
        for projection in ["to_q", "to_k", "to_v"] {
            result.append(
                tensor(
                    "\(prefix).attn1.\(projection).weight",
                    [768, 768],
                    .vae,
                    disposition
                )
            )
        }
        result += pair(
            "\(prefix).attn1.to_out.0",
            weight: [768, 768],
            bias: [768],
            component: .vae,
            disposition: disposition
        )
        result += vaeFeedForward(prefix, disposition: disposition)
        return result
    }

    private static func vaeCrossBlock(
        _ prefix: String,
        disposition: Disposition
    ) -> [ExpectedTensor] {
        var result = pair(
            "\(prefix).norm2",
            weight: [768],
            bias: [768],
            component: .vae,
            disposition: disposition
        )
        result += pair(
            "\(prefix).attn2.norm_cross",
            weight: [768],
            bias: [768],
            component: .vae,
            disposition: disposition
        )
        for projection in ["to_q", "to_k", "to_v"] {
            result.append(
                tensor(
                    "\(prefix).attn2.\(projection).weight",
                    [768, 768],
                    .vae,
                    disposition
                )
            )
        }
        result += pair(
            "\(prefix).attn2.to_out.0",
            weight: [768, 768],
            bias: [768],
            component: .vae,
            disposition: disposition
        )
        result += vaeFeedForward(prefix, disposition: disposition)
        return result
    }

    private static func vaeFeedForward(
        _ prefix: String,
        disposition: Disposition
    ) -> [ExpectedTensor] {
        var result = pair(
            "\(prefix).norm3",
            weight: [768],
            bias: [768],
            component: .vae,
            disposition: disposition
        )
        result += pair(
            "\(prefix).ff.net.0.proj",
            weight: [3_072, 768],
            bias: [3_072],
            component: .vae,
            disposition: disposition
        )
        result += pair(
            "\(prefix).ff.net.2",
            weight: [768, 3_072],
            bias: [768],
            component: .vae,
            disposition: disposition
        )
        return result
    }

    private static func meshEncoder() -> [ExpectedTensor] {
        let component = Component.meshEncoder
        let disposition = Disposition.carried
        let prefix = "mesh_encoder.encoder"
        var result = pair(
            "\(prefix).input_proj",
            weight: [512, 54],
            bias: [512],
            component: component,
            disposition: disposition
        )
        result += [
            tensor("\(prefix).cross_attn.attn.c_q.weight", [512, 512], component, disposition),
            tensor("\(prefix).cross_attn.attn.c_kv.weight", [1_024, 512], component, disposition),
        ]
        result += pair(
            "\(prefix).cross_attn.attn.c_proj",
            weight: [512, 512],
            bias: [512],
            component: component,
            disposition: disposition
        )
        for norm in ["ln_1", "ln_2"] {
            result += pair(
                "\(prefix).cross_attn.\(norm)",
                weight: [512],
                bias: [512],
                component: component,
                disposition: disposition
            )
        }
        result += pair(
            "\(prefix).cross_attn.mlp.c_fc",
            weight: [2_048, 512],
            bias: [2_048],
            component: component,
            disposition: disposition
        )
        result += pair(
            "\(prefix).cross_attn.mlp.c_proj",
            weight: [512, 2_048],
            bias: [512],
            component: component,
            disposition: disposition
        )
        result += pair(
            "\(prefix).cross_attn.ln_3",
            weight: [512],
            bias: [512],
            component: component,
            disposition: disposition
        )
        for layer in 0..<8 {
            let block = "\(prefix).self_attn.resblocks.\(layer)"
            result.append(
                tensor("\(block).attn.c_qkv.weight", [1_536, 512], component, disposition)
            )
            result += pair(
                "\(block).attn.c_proj",
                weight: [512, 512],
                bias: [512],
                component: component,
                disposition: disposition
            )
            result += pair(
                "\(block).ln_1",
                weight: [512],
                bias: [512],
                component: component,
                disposition: disposition
            )
            result += pair(
                "\(block).mlp.c_fc",
                weight: [2_048, 512],
                bias: [2_048],
                component: component,
                disposition: disposition
            )
            result += pair(
                "\(block).mlp.c_proj",
                weight: [512, 2_048],
                bias: [512],
                component: component,
                disposition: disposition
            )
            result += pair(
                "\(block).ln_2",
                weight: [512],
                bias: [512],
                component: component,
                disposition: disposition
            )
        }
        result += pair(
            "\(prefix).ln_post",
            weight: [512],
            bias: [512],
            component: component,
            disposition: disposition
        )
        return result
    }

    private static func qwen() -> [ExpectedTensor] {
        let component = Component.qwen
        let disposition = Disposition.carried
        var result = [
            tensor("transformer.model.embed_tokens.weight", [33_036, 896], component, disposition),
        ]
        for layer in 0..<28 {
            let prefix = "transformer.model.layers.\(layer)"
            result += [
                tensor("\(prefix).self_attn.q_proj.weight", [2_048, 896], component, disposition),
                tensor("\(prefix).self_attn.k_proj.weight", [1_024, 896], component, disposition),
                tensor("\(prefix).self_attn.v_proj.weight", [1_024, 896], component, disposition),
                tensor("\(prefix).self_attn.o_proj.weight", [896, 2_048], component, disposition),
                tensor("\(prefix).self_attn.q_norm.weight", [128], component, disposition),
                tensor("\(prefix).self_attn.k_norm.weight", [128], component, disposition),
                tensor("\(prefix).mlp.gate_proj.weight", [3_072, 896], component, disposition),
                tensor("\(prefix).mlp.up_proj.weight", [3_072, 896], component, disposition),
                tensor("\(prefix).mlp.down_proj.weight", [896, 3_072], component, disposition),
                tensor("\(prefix).input_layernorm.weight", [896], component, disposition),
                tensor("\(prefix).post_attention_layernorm.weight", [896], component, disposition),
            ]
        }
        result += [
            tensor("transformer.model.norm.weight", [896], component, disposition),
            tensor("transformer.lm_head.weight", [33_036, 896], component, disposition),
        ]
        return result
    }
}

/// Exact rig model checkpoint contract failures.
public enum SmeltRigCheckpointPlanError: Error, Equatable {
    case duplicateTensors([String])
    case coverage(missing: [String], unexpected: [String])
    case dtype(String, expected: String, got: String)
    case shape(String, expected: [Int], got: [Int])
    case byteCount(String, expected: Int, got: Int)
    case untiedLanguageModelHead
}
