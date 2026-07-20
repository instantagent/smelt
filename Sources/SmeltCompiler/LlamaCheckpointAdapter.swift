import Foundation
import SmeltSchema

public struct LlamaCheckpointAdapter {

    public static func mapName(_ hfName: String) -> String {
        var name = hfName
        for prefix in ["model.language_model.", "language_model.", "model."] {
            if name.hasPrefix(prefix) {
                name = String(name.dropFirst(prefix.count))
                break
            }
        }

        name = name.replacingOccurrences(of: ".", with: "_")

        if name == "embed_tokens_weight" {
            return SmeltCanonicalTensorNames.embedTokens
        }
        if name == "norm_weight" {
            return "norm_weight"
        }
        if name == "lm_head_weight" {
            return "lm_head_weight"
        }
        return name
    }

    public static func isTextModelTensor(_ hfName: String) -> Bool {
        let skipPrefixes = [
            "vision", "vision_tower", "multi_modal_projector", "multimodal_projector",
            "model.vision_tower", "model.vision", "model.multi_modal_projector",
            "model.mm_", "mm_",
        ]
        if skipPrefixes.contains(where: { hfName.hasPrefix($0) }) {
            return false
        }

        let keepPrefixes = [
            "model.language_model.",
            "language_model.",
            "model.layers.",
            "model.embed_tokens.",
            "model.norm.",
            "layers.",
            "embed_tokens.",
            "norm.",
            "lm_head.",
        ]
        return keepPrefixes.contains(where: { hfName.hasPrefix($0) })
    }

    public static func validateConfig(
        hfConfig: [String: Any],
        modelIR: SmeltModelIR
    ) throws {
        let cfg = (hfConfig["text_config"] as? [String: Any]) ?? hfConfig
        let model = modelIR.config

        if cfg["model_type"] != nil {
            try requireString("model_type", in: cfg, expected: "llama")
        }
        try requireInt("hidden_size", in: cfg, expected: model.hiddenSize)
        try requireInt("num_hidden_layers", in: cfg, expected: model.numLayers)
        try requireInt("vocab_size", in: cfg, expected: model.vocabSize)
        try requireFloat("rms_norm_eps", in: cfg, expected: model.rmsEps)
        try requireBool("tie_word_embeddings", in: cfg, expected: model.tiedLMHead)
        try requireInt("intermediate_size", in: cfg, expected: model.ffn.dim)

        if model.ffn.activation != .swiglu {
            throw LlamaAdapterError.configMismatch(
                "ffn.activation",
                expected: SmeltActivation.swiglu.rawValue,
                got: model.ffn.activation.rawValue
            )
        }
        if cfg["hidden_act"] != nil {
            try requireString("hidden_act", in: cfg, expected: "silu")
        }
        try requireOptionalBool("attention_bias", in: cfg, expected: false)
        try requireOptionalBool("mlp_bias", in: cfg, expected: false)

        if let maxPositionEmbeddings = cfg["max_position_embeddings"] as? Int,
           let staticSeqCapacity = model.staticSeqCapacity,
           staticSeqCapacity > maxPositionEmbeddings
        {
            throw LlamaAdapterError.configMismatch(
                "static_seq_capacity",
                expected: "<= \(maxPositionEmbeddings)",
                got: "\(staticSeqCapacity)"
            )
        }

        let attentionTypes = Set(modelIR.layerPattern.expanded.filter(\.isAttentionFamily))
        guard attentionTypes == Set([SmeltLayerType.attention]) else {
            throw LlamaAdapterError.configMismatch(
                "layer pattern",
                expected: "dense attn-only transformer",
                got: modelIR.layerPattern.expanded.map(\.rawValue).joined(separator: ",")
            )
        }
        let attentionConfigs = attentionTypes.compactMap { model.attentionConfig(for: $0) }

        guard let sharedQHeads = sharedInt(attentionConfigs.map(\.qHeads)) else {
            throw LlamaAdapterError.missingConfigKey("attention")
        }
        try requireInt("num_attention_heads", in: cfg, expected: sharedQHeads)

        guard let sharedKVHeads = sharedInt(attentionConfigs.map(\.kvHeads)) else {
            throw LlamaAdapterError.missingConfigKey("attention")
        }
        try requireInt("num_key_value_heads", in: cfg, expected: sharedKVHeads)

        guard let sharedHeadDim = sharedInt(attentionConfigs.map(\.headDim)) else {
            throw LlamaAdapterError.missingConfigKey("attention")
        }
        if cfg["head_dim"] != nil {
            try requireInt("head_dim", in: cfg, expected: sharedHeadDim)
        } else if model.hiddenSize / sharedQHeads != sharedHeadDim {
            throw LlamaAdapterError.configMismatch(
                "head_dim",
                expected: "\(model.hiddenSize / sharedQHeads)",
                got: "\(sharedHeadDim)"
            )
        }

        for attn in attentionConfigs {
            guard attn.gatedQ == false else {
                throw LlamaAdapterError.configMismatch(
                    "gated_q",
                    expected: "false",
                    got: "true"
                )
            }
            guard attn.qkNorm == false else {
                throw LlamaAdapterError.configMismatch(
                    "qk_norm",
                    expected: "false",
                    got: "true"
                )
            }
            guard attn.vNorm == false else {
                throw LlamaAdapterError.configMismatch(
                    "v_norm",
                    expected: "false",
                    got: "true"
                )
            }
            guard attn.effectiveRopeDim(default: model.ropeDim) == sharedHeadDim else {
                throw LlamaAdapterError.configMismatch(
                    "rope_dim",
                    expected: "\(sharedHeadDim)",
                    got: "\(attn.effectiveRopeDim(default: model.ropeDim))"
                )
            }
            guard attn.ropeLayout == .splitHalf else {
                throw LlamaAdapterError.configMismatch(
                    "rope_layout",
                    expected: SmeltRoPELayout.splitHalf.rawValue,
                    got: attn.ropeLayout?.rawValue ?? "nil"
                )
            }
            if cfg["rope_theta"] != nil {
                try requireFloat("rope_theta", in: cfg, expected: attn.ropeTheta)
            }
            try requireLlama3RoPEScaling(attn.ropeScaling, in: cfg)
        }
    }

    public static func isTiedWeight(_ runtimeName: String, config: SmeltConfig) -> Bool {
        config.tiedLMHead && runtimeName == "lm_head_weight"
    }

    private static func requireLlama3RoPEScaling(
        _ scaling: SmeltRoPEScaling?,
        in cfg: [String: Any]
    ) throws {
        guard let hfScaling = cfg["rope_scaling"] as? [String: Any] else {
            guard scaling == nil else {
                throw LlamaAdapterError.missingConfigKey("rope_scaling")
            }
            return
        }
        guard let scaling else {
            throw LlamaAdapterError.configMismatch(
                "rope_scaling",
                expected: String(describing: hfScaling),
                got: "nil"
            )
        }

        let ropeType = (hfScaling["rope_type"] as? String) ?? (hfScaling["type"] as? String)
        guard ropeType == scaling.type.rawValue else {
            throw LlamaAdapterError.configMismatch(
                "rope_scaling.rope_type",
                expected: scaling.type.rawValue,
                got: ropeType ?? "nil"
            )
        }
        try requireFloat("factor", in: hfScaling, expected: scaling.factor)
        try requireFloat("low_freq_factor", in: hfScaling, expected: scaling.lowFreqFactor)
        try requireFloat("high_freq_factor", in: hfScaling, expected: scaling.highFreqFactor)
        try requireInt(
            "original_max_position_embeddings",
            in: hfScaling,
            expected: scaling.originalMaxPositionEmbeddings
        )
    }

    private static func sharedInt(_ values: [Int]) -> Int? {
        guard let first = values.first else { return nil }
        return values.dropFirst().allSatisfy({ $0 == first }) ? first : nil
    }

    private static func requireInt(
        _ key: String,
        in cfg: [String: Any],
        expected: Int
    ) throws {
        guard let value = cfg[key] as? Int else {
            throw LlamaAdapterError.missingConfigKey(key)
        }
        guard value == expected else {
            throw LlamaAdapterError.configMismatch(key, expected: "\(expected)", got: "\(value)")
        }
    }

    private static func requireFloat(
        _ key: String,
        in cfg: [String: Any],
        expected: Float
    ) throws {
        let raw = cfg[key]
        let value: Float?
        switch raw {
        case let v as Float:
            value = v
        case let v as Double:
            value = Float(v)
        case let v as NSNumber:
            value = v.floatValue
        default:
            value = nil
        }
        guard let value else {
            throw LlamaAdapterError.missingConfigKey(key)
        }
        guard abs(value - expected) <= 1e-5 else {
            throw LlamaAdapterError.configMismatch(key, expected: "\(expected)", got: "\(value)")
        }
    }

    private static func requireBool(
        _ key: String,
        in cfg: [String: Any],
        expected: Bool
    ) throws {
        guard let value = cfg[key] as? Bool else {
            throw LlamaAdapterError.missingConfigKey(key)
        }
        guard value == expected else {
            throw LlamaAdapterError.configMismatch(key, expected: "\(expected)", got: "\(value)")
        }
    }

    private static func requireOptionalBool(
        _ key: String,
        in cfg: [String: Any],
        expected: Bool
    ) throws {
        guard let raw = cfg[key], !(raw is NSNull) else { return }
        guard let value = raw as? Bool else {
            throw LlamaAdapterError.missingConfigKey(key)
        }
        guard value == expected else {
            throw LlamaAdapterError.configMismatch(key, expected: "\(expected)", got: "\(value)")
        }
    }

    private static func requireString(
        _ key: String,
        in cfg: [String: Any],
        expected: String
    ) throws {
        guard let value = cfg[key] as? String else {
            throw LlamaAdapterError.missingConfigKey(key)
        }
        guard value == expected else {
            throw LlamaAdapterError.configMismatch(key, expected: expected, got: value)
        }
    }
}

public enum LlamaAdapterError: Error, CustomStringConvertible {
    case missingConfigKey(String)
    case configMismatch(String, expected: String, got: String)

    public var description: String {
        switch self {
        case let .missingConfigKey(key):
            return "Missing key '\(key)' in Llama HF config.json"
        case let .configMismatch(key, expected, got):
            return "Llama config mismatch: \(key) expected \(expected), got \(got)"
        }
    }
}
