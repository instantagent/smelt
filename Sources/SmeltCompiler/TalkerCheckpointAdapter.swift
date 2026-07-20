// TalkerCheckpointAdapter — Maps the Qwen3-TTS *talker trunk* checkpoint to the
// Smelt canonical schema for the generic dense-trunk build path (W5 of
// docs/talker-trunk-fit-audit.md).
//
// Unlike the full Qwen3-TTS package policy, which carries tensors from the CAM
// graph by verbatim checkpoint name, this adapter ingests ONLY the talker decoder trunk
// — the embeddings-in / hidden-out region
// the generic compiler compiles — and renames its tensors to the canonical
// `layers_N_...` schema that `SmeltWeightLayout.computeLayout` expects. The
// front-end embeddings (`codec_embedding`, `text_embedding`), the MTP
// (`talker.code_predictor.*`), the codec head (`talker.codec_head.*`), the text
// projection (`talker.text_projection.*`) and the speech-tokenizer decoder
// (`decoder.*`) are NOT trunk tensors and are filtered out.

import Foundation
import SmeltSchema

public struct TalkerCheckpointAdapter {

    /// The exact per-layer trunk module suffixes (the part after
    /// `talker.model.layers.<N>.`): attn projections, per-head qk-norm, MLP, the
    /// two layer norms — 11 tensors per layer.
    private static let trunkLayerSuffixes: Set<String> = [
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        "input_layernorm.weight", "post_attention_layernorm.weight",
    ]

    /// The talker trunk's tensors, and only those. An EXPLICIT allowlist, not a
    /// prefix match: `assembleCheckpointTensors` only errors on *missing* expected
    /// tensors (then iterates expectedLayout), so an extra kept tensor would be
    /// silently ignored — a `*.bias`, `rotary_emb.inv_freq`, or stray norm a future
    /// checkpoint ships must be rejected here, not waved through. Per-layer tensors
    /// must match a known module suffix; the only non-layer trunk tensor is the
    /// final `model.norm`. Front-end `codec_embedding`/`text_embedding` live under
    /// `talker.model.` too but have no `.layers.` segment, so they fall through.
    public static func isTrunkTensor(_ hfName: String) -> Bool {
        if hfName == "talker.model.norm.weight" { return true }
        let prefix = "talker.model.layers."
        guard hfName.hasPrefix(prefix) else { return false }
        // Strip `talker.model.layers.<N>.` → the module suffix.
        let afterPrefix = hfName.dropFirst(prefix.count)
        guard let dot = afterPrefix.firstIndex(of: ".") else { return false }
        let suffix = afterPrefix[afterPrefix.index(after: dot)...]
        return trunkLayerSuffixes.contains(String(suffix))
    }

    /// Canonical weights.bin / manifest key for a trunk tensor. Strips the
    /// `talker.model.` namespace and replaces dots with underscores, matching the
    /// `layers_N_self_attn_q_proj_weight` / `norm_weight` names computeLayout emits.
    /// Example: `talker.model.layers.0.self_attn.q_norm.weight`
    ///        → `layers_0_self_attn_q_norm_weight`.
    public static func mapName(_ hfName: String) -> String {
        if hfName == "talker.model.norm.weight" { return "norm_weight" }
        var name = hfName
        let prefix = "talker.model."
        if name.hasPrefix(prefix) { name = String(name.dropFirst(prefix.count)) }
        return name.replacingOccurrences(of: ".", with: "_")
    }

    /// Validate the qwen3_tts config.json against the trunk IR. The talker dims
    /// live under the nested `talker_config`; `rope_theta` and `rms_norm_eps` are
    /// kernel constants that would NOT surface as a shape mismatch downstream, so
    /// they are checked here (the rest of the dims are caught by per-tensor shape
    /// validation in assembleCheckpointTensors).
    public static func validateConfig(hfConfig: [String: Any], ir: SmeltModelIR) throws {
        guard let talker = hfConfig["talker_config"] as? [String: Any] else {
            throw TalkerAdapterError.missingConfigKey("talker_config")
        }

        // JSON numbers bridge to NSNumber regardless of integer vs decimal
        // spelling (1000000 vs 1000000.0), so read through NSNumber rather than
        // a strict `as? Int`/`as? Double` that would false-fail on either form.
        func number(_ key: String) throws -> NSNumber {
            guard let n = talker[key] as? NSNumber else {
                throw TalkerAdapterError.missingConfigKey("talker_config.\(key)")
            }
            return n
        }
        func expectInt(_ key: String, _ expected: Int) throws {
            // Int(exactly:) rejects a non-integral value (2048.9) rather than
            // letting intValue silently truncate it to a passing 2048.
            let n = try number(key)
            guard let got = Int(exactly: n.doubleValue) else {
                throw TalkerAdapterError.configMismatch(
                    "talker_config.\(key)", expected: "\(expected)",
                    got: "\(n.doubleValue) (non-integral)")
            }
            guard got == expected else {
                throw TalkerAdapterError.configMismatch(
                    "talker_config.\(key)", expected: "\(expected)", got: "\(got)")
            }
        }

        try expectInt("hidden_size", ir.config.hiddenSize)
        try expectInt("num_hidden_layers", ir.config.numLayers)
        try expectInt("vocab_size", ir.config.vocabSize)

        // rope_theta: compare as the IR's Float.
        guard let theta = ir.config.attention?.ropeTheta else {
            throw TalkerAdapterError.missingConfigKey("(IR) attention.ropeTheta")
        }
        let cfgTheta = Float(try number("rope_theta").doubleValue)
        guard cfgTheta == theta else {
            throw TalkerAdapterError.configMismatch(
                "talker_config.rope_theta", expected: "\(theta)", got: "\(cfgTheta)")
        }

        // rms_norm_eps → the IR stores Float rmsEps.
        let cfgEps = Float(try number("rms_norm_eps").doubleValue)
        guard cfgEps == ir.config.rmsEps else {
            throw TalkerAdapterError.configMismatch(
                "talker_config.rms_norm_eps",
                expected: "\(ir.config.rmsEps)", got: "\(cfgEps)")
        }
    }

    /// The trunk has no LM head (hidden-out port): nothing is tied.
    public static func isTiedWeight(_ runtimeName: String, config: SmeltConfig) -> Bool {
        false
    }
}

// MARK: - Errors

public enum TalkerAdapterError: Error, CustomStringConvertible {
    case missingConfigKey(String)
    case configMismatch(String, expected: String, got: String)

    public var description: String {
        switch self {
        case let .missingConfigKey(key):
            return "Missing key '\(key)' in Qwen3-TTS config.json"
        case let .configMismatch(key, expected, got):
            return "Talker config mismatch: \(key) expected \(expected), got \(got)"
        }
    }
}
