import Testing
@testable import SmeltCompiler
import SmeltSchema

private func qwenMTPFixture() -> SmeltModelIR {
    SmeltModelIR(
        modelName: "test/qwen-mtp",
        config: SmeltConfig(
            hiddenSize: 64,
            numLayers: 1,
            vocabSize: 128,
            staticSeqCapacity: 32,
            ropeDim: 16,
            rmsEps: 1e-6,
            normMode: .onePlusWeight,
            attention: SmeltAttentionConfig(
                qHeads: 2,
                kvHeads: 1,
                headDim: 32,
                gatedQ: false,
                qkNorm: true,
                ropeTheta: 10_000_000
            ),
            ffn: SmeltFFNConfig(dim: 128, activation: .swiglu),
            tiedLMHead: false,
            inputFusion: SmeltInputFusionConfig(
                sourceWidth: 64,
                sourceCount: 2,
                normalizeSources: true
            )
        ),
        layerPattern: SmeltLayerPattern(unit: [.attention], repeats: 1),
        quantization: SmeltQuantizationConfig(
            strategy: .fp16,
            groupSize: 128,
            excludePatterns: []
        ),
        loading: SmeltLoadingConfig(
            strategy: .mmapPrefault,
            packing: .monolithic,
            checkpointMap: .qwenMTPHF
        )
    )
}

@Test func qwenMTPInputFusionParsesAndValidates() throws {
    let ir = qwenMTPFixture()
    try validateSmeltIR(ir)

    let fusion = try #require(ir.config.inputFusion)
    #expect(fusion.sourceWidth == 64)
    #expect(fusion.sourceCount == 2)
    #expect(fusion.normalizeSources)
    #expect(fusion.postProjectionWidth == nil)
    #expect(ir.config.backboneHiddenSize == nil)
    #expect(try SmeltCheckpointAdapter.authored(for: ir) == .qwenMTP)
}

@Test func qwenMTPLayoutIsAnOrdinaryLayerWithoutDeadEmbeddingOrPostProjection() throws {
    let ir = qwenMTPFixture()
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let byName = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })

    #expect(byName[SmeltCanonicalTensorNames.embedTokens] == nil)
    #expect(byName["input_fusion_norm_0_weight"]?.shape == [64])
    #expect(byName["input_fusion_norm_1_weight"]?.shape == [64])
    #expect(byName["pre_projection_weight"]?.shape == [64, 128])
    #expect(byName["post_projection_weight"] == nil)
    #expect(byName["layers_0_self_attn_q_proj_weight"]?.shape == [64, 64])
    #expect(byName["lm_head_weight"]?.shape == [128, 64])

    let plan = buildBufferPlan(from: ir)
    let hiddenB = try #require(plan.slots.first { $0.name == "hiddenB" })
    #expect(hiddenB.sizeBytes >= 128 * 2)
}

@Test func qwenMTPEmitterNormalizesBothSourcesBeforeOrdinaryStack() throws {
    let ir = qwenMTPFixture()
    try validateSmeltIR(ir)
    let result = try TopLevelEmitter.generate(
        ir: ir,
        plan: buildBufferPlan(from: ir),
        weightLayout: SmeltWeightLayout.computeLayout(from: ir)
    )

    #expect(!result.source.contains("Embedding →"))
    #expect(result.source.contains("Input fusion source 0 norm"))
    #expect(result.source.contains("Input fusion source 1 norm"))
    #expect(result.source.contains("Pre-projection (fused input → stack hidden)"))
    #expect(!result.source.contains("Post-projection"))
    #expect(result.source.contains("Layer 0 (attn)"))
    #expect(result.source.contains("LM head"))
}

@Test func qwenMTPCheckpointNamesProjectIntoCanonicalLayerSchema() throws {
    #expect(QwenMTPCheckpointAdapter.mapName("mtp.fc.weight") == "pre_projection_weight")
    #expect(
        QwenMTPCheckpointAdapter.mapName("mtp.pre_fc_norm_embedding.weight")
            == "input_fusion_norm_0_weight"
    )
    #expect(
        QwenMTPCheckpointAdapter.mapName("mtp.pre_fc_norm_hidden.weight")
            == "input_fusion_norm_1_weight"
    )
    #expect(
        QwenMTPCheckpointAdapter.mapName("mtp.layers.0.self_attn.q_proj.weight")
            == "layers_0_self_attn_q_proj_weight"
    )
    #expect(QwenMTPCheckpointAdapter.mapName("mtp.norm.weight") == "norm_weight")
    #expect(QwenMTPCheckpointAdapter.mapName("lm_head.weight") == "lm_head_weight")
    #expect(QwenMTPCheckpointAdapter.isModuleTensor("mtp.layers.0.mlp.up_proj.weight"))
    #expect(QwenMTPCheckpointAdapter.isModuleTensor("lm_head.weight"))
    #expect(!QwenMTPCheckpointAdapter.isModuleTensor("model.layers.0.mlp.up_proj.weight"))

    let ir = qwenMTPFixture()
    try QwenMTPCheckpointAdapter.validateConfig(
        hfConfig: [
            "text_config": [
                "hidden_size": 64,
                "vocab_size": 128,
                "mtp_num_hidden_layers": 1,
            ],
        ],
        modelConfig: ir.config
    )
}

@Test func manifestCarriesGenericInputFusionABI() throws {
    let ir = qwenMTPFixture()
    let manifestFusion = try #require(
        SmeltCompiler.manifestConfigSnapshot(from: ir).inputFusion
    )
    #expect(manifestFusion.sourceWidth == 64)
    #expect(manifestFusion.sourceCount == 2)
    #expect(manifestFusion.normalizeSources)
    #expect(manifestFusion.postProjectionWidth == nil)
}
