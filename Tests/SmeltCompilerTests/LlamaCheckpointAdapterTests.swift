import Foundation
import Testing
@testable import SmeltCompiler

private func loadLlamaIR() throws -> SmeltModelIR {
    let ir = FixtureModelIRs.llama_arch_1b
    try validateSmeltIR(ir)
    return ir
}

private func loadLlamaHFConfig(
    _ resource: String = "llama-arch-1b-hf-config"
) throws -> [String: Any] {
    let fixtureURL = Bundle.module.url(
        forResource: resource,
        withExtension: "json",
        subdirectory: "Fixtures"
    )!
    let data = try Data(contentsOf: fixtureURL)
    guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw NSError(domain: "LlamaCheckpointAdapterTests", code: 1)
    }
    return json
}

private func loadLlamaKeySample(
    _ resource: String = "llama-arch-1b-key-sample"
) throws -> [String] {
    let fixtureURL = Bundle.module.url(
        forResource: resource,
        withExtension: "txt",
        subdirectory: "Fixtures"
    )!
    return try String(contentsOf: fixtureURL, encoding: .utf8)
        .split(separator: "\n")
        .map(String.init)
        .filter { !$0.isEmpty }
}

@Test func llamaCheckpointAdapterMapsCanonicalTensorNames() {
    #expect(
        LlamaCheckpointAdapter.mapName("model.layers.0.self_attn.q_proj.weight")
            == "layers_0_self_attn_q_proj_weight"
    )
    #expect(LlamaCheckpointAdapter.mapName("model.embed_tokens.weight") == "embed_tokens")
    #expect(LlamaCheckpointAdapter.mapName("model.norm.weight") == "norm_weight")
    #expect(LlamaCheckpointAdapter.mapName("lm_head.weight") == "lm_head_weight")
}

@Test func llamaCheckpointAdapterFiltersNonTextTensors() {
    #expect(LlamaCheckpointAdapter.isTextModelTensor("model.layers.0.mlp.gate_proj.weight"))
    #expect(LlamaCheckpointAdapter.isTextModelTensor("model.embed_tokens.weight"))
    #expect(!LlamaCheckpointAdapter.isTextModelTensor("model.vision_tower.encoder.weight"))
    #expect(!LlamaCheckpointAdapter.isTextModelTensor("model.multi_modal_projector.weight"))
}

@Test func llamaCheckpointAdapterValidatesLlamaArchConfig() throws {
    let ir = try loadLlamaIR()
    let hfConfig = try loadLlamaHFConfig()
    try LlamaCheckpointAdapter.validateConfig(hfConfig: hfConfig, modelIR: ir)
}

@Test func llamaCheckpointAdapterRejectsMissingSplitHalfRoPELayout() throws {
    let ir = try loadLlamaIR()
    let attn = SmeltAttentionConfig(
        qHeads: 32,
        kvHeads: 8,
        headDim: 64,
        gatedQ: false,
        qkNorm: false,
        ropeTheta: 500_000,
        ropeDim: 64,
        ropeScaling: SmeltRoPEScaling(
            type: .llama3,
            factor: 32,
            lowFreqFactor: 1,
            highFreqFactor: 4,
            originalMaxPositionEmbeddings: 8192
        )
    )
    let config = SmeltConfig(
        hiddenSize: ir.config.hiddenSize,
        numLayers: ir.config.numLayers,
        vocabSize: ir.config.vocabSize,
        staticSeqCapacity: ir.config.staticSeqCapacity,
        ropeDim: ir.config.ropeDim,
        rmsEps: ir.config.rmsEps,
        attention: attn,
        ffn: ir.config.ffn,
        tiedLMHead: true
    )
    let badIR = SmeltModelIR(
        modelName: ir.modelName,
        config: config,
        layerPattern: ir.layerPattern,
        quantization: ir.quantization,
        loading: ir.loading,
        prefill: ir.prefill,
        inference: ir.inference
    )

    do {
        try LlamaCheckpointAdapter.validateConfig(hfConfig: try loadLlamaHFConfig(), modelIR: badIR)
        Issue.record("Expected Llama config mismatch on rope_layout")
    } catch let error as LlamaAdapterError {
        #expect(error.description.contains("rope_layout"))
        #expect(error.description.contains("split_half"))
    }
}

@Test func llamaCheckpointAdapterRejectsMismatchedRoPEScaling() throws {
    let ir = try loadLlamaIR()
    var hfConfig = try loadLlamaHFConfig()
    var scaling = hfConfig["rope_scaling"] as! [String: Any]
    scaling["factor"] = 16.0
    hfConfig["rope_scaling"] = scaling

    do {
        try LlamaCheckpointAdapter.validateConfig(hfConfig: hfConfig, modelIR: ir)
        Issue.record("Expected Llama config mismatch on rope scaling factor")
    } catch let error as LlamaAdapterError {
        #expect(error.description.contains("factor"))
        #expect(error.description.contains("32.0"))
        #expect(error.description.contains("16.0"))
    }
}

@Test func llamaCheckpointAdapterMapsOfficialTensorKeySample() throws {
    let keys = try loadLlamaKeySample()
    let mapped = Set(keys.filter(LlamaCheckpointAdapter.isTextModelTensor).map(LlamaCheckpointAdapter.mapName))

    #expect(mapped.contains("embed_tokens"))
    #expect(mapped.contains("layers_0_input_layernorm_weight"))
    #expect(mapped.contains("layers_0_self_attn_q_proj_weight"))
    #expect(mapped.contains("layers_0_self_attn_k_proj_weight"))
    #expect(mapped.contains("layers_0_self_attn_v_proj_weight"))
    #expect(mapped.contains("layers_0_self_attn_o_proj_weight"))
    #expect(mapped.contains("layers_0_mlp_gate_proj_weight"))
    #expect(mapped.contains("layers_0_mlp_up_proj_weight"))
    #expect(mapped.contains("layers_0_mlp_down_proj_weight"))
    #expect(mapped.contains("layers_15_post_attention_layernorm_weight"))
    #expect(mapped.contains("norm_weight"))
}

@Test func llamaCheckpointAdapterOfficialKeySampleMatchesExpectedLayoutNames() throws {
    let ir = try loadLlamaIR()
    let keys = try loadLlamaKeySample()
    let mapped = Set(keys.filter(LlamaCheckpointAdapter.isTextModelTensor).map(LlamaCheckpointAdapter.mapName))
    let expectedNames = Set(SmeltWeightLayout.computeLayout(from: ir).map(\.name))
    let relevantExpectedSubset: Set<String> = [
        "embed_tokens",
        "layers_0_input_layernorm_weight",
        "layers_0_self_attn_q_proj_weight",
        "layers_0_self_attn_k_proj_weight",
        "layers_0_self_attn_v_proj_weight",
        "layers_0_self_attn_o_proj_weight",
        "layers_0_post_attention_layernorm_weight",
        "layers_0_mlp_gate_proj_weight",
        "layers_0_mlp_up_proj_weight",
        "layers_0_mlp_down_proj_weight",
        "layers_15_self_attn_q_proj_weight",
        "layers_15_mlp_down_proj_weight",
        "norm_weight",
    ]

    #expect(!expectedNames.contains("lm_head_weight"))
    #expect(relevantExpectedSubset.isSubset(of: expectedNames))
    #expect(relevantExpectedSubset.isSubset(of: mapped))
}

@Test func agentCheckpointAdapterUsesAuthoredLlamaMap() throws {
    let ir = try loadLlamaIR()

    #expect(try SmeltCheckpointAdapter.authored(for: ir) == .llama)
}

@Test func llamaMapRequiresLlama3RoPEShape() throws {
    let ir = FixtureModelIRs.llama_arch_1b_noRoPEScaling

    do {
        _ = try SmeltCheckpointAdapter.authored(for: ir)
        Issue.record("Expected authored Llama map to reject non-llama3 RoPE shape")
    } catch {
        #expect("\(error)".contains("loading.checkpoint_map 'hf.llama'"))
    }
}
