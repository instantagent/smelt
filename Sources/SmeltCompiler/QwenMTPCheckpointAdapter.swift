// QwenMTPCheckpointAdapter — projects the checkpoint-owned `mtp` module onto
// the same canonical transformer tensors used by an ordinary Smelt package.
// Runtime composition is separate; this adapter contains no execution policy.

import Foundation

public struct QwenMTPCheckpointAdapter {
    public static func mapName(_ hfName: String) -> String {
        switch hfName {
        case "mtp.fc.weight":
            return "pre_projection_weight"
        case "mtp.pre_fc_norm_embedding.weight":
            return "input_fusion_norm_0_weight"
        case "mtp.pre_fc_norm_hidden.weight":
            return "input_fusion_norm_1_weight"
        case "mtp.norm.weight":
            return "norm_weight"
        case "lm_head.weight":
            return "lm_head_weight"
        default:
            if hfName.hasPrefix("mtp.layers.") {
                var name = String(hfName.dropFirst("mtp.".count))
                name = name.replacingOccurrences(of: ".", with: "_")
                return name
            }
            return hfName.replacingOccurrences(of: ".", with: "_")
        }
    }

    public static func isModuleTensor(_ hfName: String) -> Bool {
        hfName.hasPrefix("mtp.") || hfName == "lm_head.weight"
    }

    public static func validateConfig(
        hfConfig: [String: Any],
        modelConfig: SmeltConfig
    ) throws {
        let text = hfConfig["text_config"] as? [String: Any] ?? hfConfig
        try requireInt(
            "hidden_size", in: text, expected: modelConfig.hiddenSize
        )
        try requireInt(
            "vocab_size", in: text, expected: modelConfig.vocabSize
        )

        let mtpLayers = (text["mtp_num_hidden_layers"] as? Int)
            ?? (hfConfig["mtp_num_hidden_layers"] as? Int)
        guard let mtpLayers else {
            throw QwenAdapterError.missingConfigKey("mtp_num_hidden_layers")
        }
        guard mtpLayers == modelConfig.numLayers else {
            throw QwenAdapterError.configMismatch(
                "mtp_num_hidden_layers",
                expected: modelConfig.numLayers,
                got: mtpLayers
            )
        }

        guard let fusion = modelConfig.inputFusion,
              fusion.sourceCount == 2,
              fusion.sourceWidth == modelConfig.hiddenSize,
              fusion.normalizeSources,
              fusion.postProjectionWidth == nil
        else {
            throw SmeltCompilerError.unsupportedConfiguration(
                "hf.qwen-mtp requires two independently-normalized hidden-width input sources and no post projection"
            )
        }
    }

    private static func requireInt(
        _ key: String,
        in config: [String: Any],
        expected: Int
    ) throws {
        guard let actual = config[key] as? Int else {
            throw QwenAdapterError.missingConfigKey(key)
        }
        guard actual == expected else {
            throw QwenAdapterError.configMismatch(
                key, expected: expected, got: actual
            )
        }
    }
}
