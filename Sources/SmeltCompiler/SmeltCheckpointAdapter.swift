// SmeltCheckpointAdapter — Selects the HF checkpoint mapping path from graph shape.

import Foundation

enum SmeltCheckpointAdapter: Equatable {
    case qwen
    case qwenMTP
    case llama
    /// Qwen3-TTS talker *trunk* (W5 of docs/talker-trunk-fit-audit.md): the
    /// embeddings-in / hidden-out decoder region, ingested through the generic
    /// dense-trunk build path. Selected from the explicit port topology; the full
    /// TTS package is hand-built and never reaches this adapter.
    case talker

    static func authored(for ir: SmeltModelIR) throws -> SmeltCheckpointAdapter {
        guard let map = ir.loading.checkpointMap else {
            throw SmeltCompilerError.unsupportedConfiguration(
                "HuggingFace checkpoint ingestion requires loading.checkpoint_map "
                    + "to name the tensor map; local weights.json/weights.bin builds "
                    + "do not need one"
            )
        }
        let adapter = SmeltCheckpointAdapter(checkpointMap: map)
        try adapter.validateAuthoredMapCompatibility(map, ir: ir)
        return adapter
    }

    private init(checkpointMap: SmeltCheckpointMap) {
        switch checkpointMap {
        case .qwenHF:
            self = .qwen
        case .qwenMTPHF:
            self = .qwenMTP
        case .llamaHF:
            self = .llama
        case .qwen3TTSTalkerTrunkHF:
            self = .talker
        }
    }

    private func validateAuthoredMapCompatibility(
        _ map: SmeltCheckpointMap,
        ir: SmeltModelIR
    ) throws {
        let config = ir.config
        let reason: String?
        switch self {
        case .qwen:
            let hasQwenAttentionMarker = config.delta != nil
                || config.attentionConfigs.values.contains {
                    $0.qkNorm || $0.gatedQ || $0.qkvBias
            }
            reason = config.blockTopology == .standard
                && config.portTopology == .tokenInLogitsOut
                && config.backboneHiddenSize == nil
                && hasQwenAttentionMarker
                ? nil
                : "expected a standard fp16/bf16 text graph with Qwen-style attention markers"
        case .qwenMTP:
            let fusion = config.inputFusion
            reason = config.blockTopology == .standard
                && config.portTopology == .tokenInLogitsOut
                && config.backboneHiddenSize == nil
                && fusion?.sourceCount == 2
                && fusion?.sourceWidth == config.hiddenSize
                && fusion?.normalizeSources == true
                && fusion?.postProjectionWidth == nil
                && !config.tiedLMHead
                ? nil
                : "expected an untied Qwen text stack with two normalized hidden-width input sources"
        case .llama:
            reason = config.blockTopology == .standard
                && config.backboneHiddenSize == nil
                && !config.attentionConfigs.isEmpty
                && config.attentionConfigs.values.allSatisfy({
                    !$0.qkNorm
                        && !$0.gatedQ
                        && !$0.qkvBias
                        && !$0.vNorm
                        && $0.ropeLayout == .splitHalf
                        && $0.ropeScaling?.type == .llama3
                })
                ? nil
                : "expected standard dense attention with split-half llama3 RoPE"
        case .talker:
            reason = config.portTopology == .embeddingsInHiddenOut
                ? nil
                : "expected an embeddings-in/hidden-out dense trunk"
        }
        if let reason {
            throw SmeltCompilerError.unsupportedConfiguration(
                "loading.checkpoint_map '\(map.rawValue)' does not match graph shape: \(reason)"
            )
        }
    }

    func validateConfig(hfConfig: [String: Any], ir: SmeltModelIR) throws {
        switch self {
        case .qwen:
            try QwenCheckpointAdapter.validateConfig(hfConfig: hfConfig, modelConfig: ir.config)
        case .qwenMTP:
            try QwenMTPCheckpointAdapter.validateConfig(
                hfConfig: hfConfig, modelConfig: ir.config)
        case .llama:
            try LlamaCheckpointAdapter.validateConfig(hfConfig: hfConfig, modelIR: ir)
        case .talker:
            try TalkerCheckpointAdapter.validateConfig(hfConfig: hfConfig, ir: ir)
        }
    }

    func isTextModelTensor(_ hfName: String) -> Bool {
        switch self {
        case .qwen:
            return QwenCheckpointAdapter.isTextModelTensor(hfName)
        case .qwenMTP:
            return QwenMTPCheckpointAdapter.isModuleTensor(hfName)
        case .llama:
            return LlamaCheckpointAdapter.isTextModelTensor(hfName)
        case .talker:
            return TalkerCheckpointAdapter.isTrunkTensor(hfName)
        }
    }

    func mapName(_ hfName: String) -> String {
        switch self {
        case .qwen:
            return QwenCheckpointAdapter.mapName(hfName)
        case .qwenMTP:
            return QwenMTPCheckpointAdapter.mapName(hfName)
        case .llama:
            return LlamaCheckpointAdapter.mapName(hfName)
        case .talker:
            return TalkerCheckpointAdapter.mapName(hfName)
        }
    }

    func isTiedWeight(_ runtimeName: String, config: SmeltConfig) -> Bool {
        switch self {
        case .qwen:
            return QwenCheckpointAdapter.isTiedWeight(runtimeName, config: config)
        case .qwenMTP:
            return false
        case .llama:
            return LlamaCheckpointAdapter.isTiedWeight(runtimeName, config: config)
        case .talker:
            return TalkerCheckpointAdapter.isTiedWeight(runtimeName, config: config)
        }
    }

    /// When true, mapped checkpoint tensors that the current layout cannot
    /// consume should be rejected instead of silently ignored. Qwen2-derived
    /// dense checkpoints can ship q/k/v bias tensors; Smelt's dense text path
    /// is still bias-free, so keeping those tensors and then iterating only the
    /// layout would build a non-equivalent model.
    var rejectsUnsupportedMappedBiasTensors: Bool {
        if case .qwen = self { return true }
        return false
    }

    /// When true, the kept-and-mapped checkpoint tensor set must EQUAL the trunk
    /// layout — every tensor the adapter admits has to land in `expectedLayout`,
    /// or the build throws. assembleCheckpointTensors otherwise only catches
    /// *missing* tensors and silently ignores extras; the talker trunk has a
    /// fixed, fully-known tensor set, so an admitted-but-unexpected tensor (a
    /// stray `*.bias`, an out-of-range `layers.999.*`, a non-numeric index) is a
    /// malformed checkpoint, not something to wave through. The single-stack LLM
    /// adapters (qwen/llama) deliberately keep-and-ignore extras, so
    /// this stays opt-in.
    var requiresExactTrunkCoverage: Bool {
        if case .talker = self { return true }
        return false
    }
}
