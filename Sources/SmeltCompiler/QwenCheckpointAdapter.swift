// QwenCheckpointAdapter — Maps Qwen HuggingFace checkpoints to Smelt canonical schema.
//
// Handles: name mapping, config validation, tied weights.

import Foundation
import SmeltSchema

/// Maps Qwen HuggingFace tensor names to Smelt canonical names.
public struct QwenCheckpointAdapter {

    /// Map a HuggingFace tensor name to Smelt canonical name.
    ///
    /// Example: "model.layers.0.linear_attn.in_proj_qkv.weight"
    ///        → "layers_0_linear_attn_in_proj_qkv_weight"
    public static func mapName(_ hfName: String) -> String {
        var name = hfName
        // Conditional-generation checkpoints can wrap the same text model as
        // model.language_model.model.*, language_model.model.*, or model.*.
        // Peel only known container/module wrappers, repeatedly, so the
        // canonical text map remains independent of the outer modality graph.
        while true {
            let prefix = ["language_model.", "model."].first {
                name.hasPrefix($0)
            }
            guard let prefix else { break }
            name = String(name.dropFirst(prefix.count))
        }
        // Replace dots with underscores
        name = name.replacingOccurrences(of: ".", with: "_")
        // Special cases
        if name == "embed_tokens_weight" {
            name = SmeltCanonicalTensorNames.embedTokens
        }
        if name == "lm_head_weight" {
            name = "lm_head_weight"
        }
        // conv1d weight: shape [C, 1, K] → Smelt expects name conv1d_weight
        // The HF name is layers_N_linear_attn_conv1d_weight after dot replacement
        return name
    }

    /// Check if a tensor belongs to the text model (skip vision/MTP tensors).
    public static func isTextModelTensor(_ hfName: String) -> Bool {
        // Keep text under any supported language_model wrapper. Skip sibling
        // modality towers before name normalization.
        if hfName.hasPrefix("mtp.") { return false }
        if hfName.hasPrefix("model.visual") { return false }
        if hfName.hasPrefix("vision") { return false }
        return true
    }

    /// Validate HF config.json against Smelt IR config.
    public static func validateConfig(
        hfConfig: [String: Any],
        modelConfig: SmeltConfig
    ) throws {
        // Qwen 3.5 nests model config under "text_config"
        let cfg = hfConfig["text_config"] as? [String: Any] ?? hfConfig

        guard let hiddenSize = cfg["hidden_size"] as? Int else {
            throw QwenAdapterError.missingConfigKey("hidden_size")
        }
        guard hiddenSize == modelConfig.hiddenSize else {
            throw QwenAdapterError.configMismatch(
                "hidden_size", expected: modelConfig.hiddenSize, got: hiddenSize
            )
        }

        guard let numLayers = cfg["num_hidden_layers"] as? Int else {
            throw QwenAdapterError.missingConfigKey("num_hidden_layers")
        }
        guard numLayers == modelConfig.numLayers else {
            throw QwenAdapterError.configMismatch(
                "num_hidden_layers", expected: modelConfig.numLayers, got: numLayers
            )
        }

        guard let vocabSize = cfg["vocab_size"] as? Int else {
            throw QwenAdapterError.missingConfigKey("vocab_size")
        }
        guard vocabSize == modelConfig.vocabSize else {
            throw QwenAdapterError.configMismatch(
                "vocab_size", expected: modelConfig.vocabSize, got: vocabSize
            )
        }
    }

    /// Get the list of tensor names expected by Smelt for this model.
    /// Returns tuples of (runtimeName, expectedShape).
    public static func expectedTensors(
        from ir: SmeltModelIR
    ) -> [(runtimeName: String, shape: [Int])] {
        // Use the weight layout computation to get all expected names
        let layout = SmeltWeightLayout.computeLayout(from: ir)
        return layout.map { ($0.name, $0.shape) }
    }

    /// Check if a tensor is the tied embedding/LM head.
    public static func isTiedWeight(_ runtimeName: String, config: SmeltConfig) -> Bool {
        config.tiedLMHead && runtimeName == "lm_head_weight"
    }

    /// Get the HF tensor name that maps to a Smelt name.
    /// Reverse of mapName for lookup in safetensors.
    public static func reverseMapName(_ runtimeName: String) -> String {
        var name = runtimeName
        // Special cases
        if name == SmeltCanonicalTensorNames.embedTokens {
            return "model.embed_tokens.weight"
        }
        if name == "lm_head_weight" {
            return "lm_head.weight"
        }
        if name == "norm_weight" {
            return "model.norm.weight"
        }
        // General: underscores back to dots, add "model." prefix
        // layers_0_linear_attn_... → model.layers.0.linear_attn...
        // This is tricky because some underscores are part of names
        // Use regex-like approach: numbers after "layers_" become dots
        if name.hasPrefix("layers_") {
            name = "model." + name
            // Convert layers_N_ to layers.N.
            let pattern = "layers_(\\d+)_"
            if let range = name.range(
                of: pattern, options: .regularExpression
            ) {
                let match = String(name[range])
                let replacement = match
                    .replacingOccurrences(of: "_", with: ".")
                name = name.replacingCharacters(in: range, with: replacement)
            }
            // Remaining: replace specific known separators
            // input_layernorm_weight → input_layernorm.weight
            // post_attention_layernorm_weight → post_attention_layernorm.weight
            // linear_attn_in_proj_qkv_weight → linear_attn.in_proj_qkv.weight
            // This requires knowing the module boundary structure
            // For now: the last "_weight" becomes ".weight"
            if name.hasSuffix("_weight") {
                name = String(name.dropLast(7)) + ".weight"
            }
        }
        return name
    }
}

// MARK: - Errors

public enum QwenAdapterError: Error, CustomStringConvertible {
    case missingConfigKey(String)
    case configMismatch(String, expected: Int, got: Int)
    case tensorNotFound(String)

    public var description: String {
        switch self {
        case let .missingConfigKey(key):
            return "Missing key '\(key)' in HF config.json"
        case let .configMismatch(key, expected, got):
            return "Config mismatch: \(key) expected \(expected), got \(got)"
        case let .tensorNotFound(name):
            return "Tensor '\(name)' not found in checkpoint"
        }
    }
}
