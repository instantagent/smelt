// FixtureModelIRs — programmatic `SmeltModelIR` constants that reproduce the
// retired model-spec fixtures field-for-field. These replaced the legacy DSL +
// parser: each constant here was proven byte-equal to the parsed fixture oracle
// (see the transient FixtureParityTests) before the parser and fixtures were
// deleted, then frozen as the golden. Downstream tests construct their IR from
// these instead of parsing text.
//
// The talker-trunk constants reuse the production `SmeltModelIR.denseTrunk`
// factory (the fp32-trunk ABI authority) rather than re-authoring the dims.

@testable import SmeltCompiler
import SmeltSchema

enum FixtureModelIRs {

    // MARK: - Qwen 3.5 text models

    /// `qwen35-2b` fixture — lut_u4 (group 16), CoreML prefill.
    static let qwen35_2B = SmeltModelIR(
        modelName: "Qwen/Qwen3.5-2B",
        config: SmeltConfig(
            hiddenSize: 2048,
            numLayers: 24,
            vocabSize: 248_320,
            staticSeqCapacity: 256,
            ropeDim: 64,
            rmsEps: 1e-6,
            normMode: .onePlusWeight,
            delta: SmeltDeltaConfig(
                numHeads: 16, headDim: 128, convKernel: 4,
                qkvDim: 6144, zDim: 2048, aDim: 16, bDim: 16
            ),
            attention: SmeltAttentionConfig(
                qHeads: 8, kvHeads: 2, headDim: 256,
                gatedQ: true, qkNorm: true, ropeTheta: 10_000_000
            ),
            ffn: SmeltFFNConfig(dim: 6144, activation: .swiglu),
            tiedLMHead: true
        ),
        layerPattern: SmeltLayerPattern(unit: [.delta, .delta, .delta, .attention], repeats: 6),
        quantization: SmeltQuantizationConfig(
            strategy: .lutU4, groupSize: 16,
            excludePatterns: ["conv1d_weight", "A_log", "dt_bias", "*_norm_weight"],
            quantizeEmbedding: true
        ),
        loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic, checkpointMap: .qwenHF),
        prefill: SmeltPrefillConfig(
            engine: "coreml",
            modelPath: "models/2b_batch_prefill_pal4/model.mlmodelc",
            cachePath: "models/2b_cache",
            maxBatchSize: 64,
            handoffFamilies: ["conv_state", "rec_state", "key_cache", "value_cache", "rope"]
        ),
        decode: SmeltDecodeConfig(
            policy: SmeltPackageSpec.DecodePolicy(sampler: .init(mode: .greedy), maxSteps: 512)
        ),
        inference: SmeltInferenceConfig(
            maxTokens: 512, maxTokensSource: .explicit,
            eosTokens: [248_044, 248_046], eosTokensSource: .explicit,
            chatTemplate: "chatml", thinkingPolicy: .disabled
        )
    )

    /// `qwen35-2b-affine-metalprefill` fixture — affine_u4 (group 64), Metal prefill.
    static let qwen35_2b_affine_metalprefill = SmeltModelIR(
        modelName: "Qwen/Qwen3.5-2B",
        config: SmeltConfig(
            hiddenSize: 2048,
            numLayers: 24,
            vocabSize: 248_320,
            staticSeqCapacity: 256,
            ropeDim: 64,
            rmsEps: 1e-6,
            normMode: .onePlusWeight,
            delta: SmeltDeltaConfig(
                numHeads: 16, headDim: 128, convKernel: 4,
                qkvDim: 6144, zDim: 2048, aDim: 16, bDim: 16
            ),
            attention: SmeltAttentionConfig(
                qHeads: 8, kvHeads: 2, headDim: 256,
                gatedQ: true, qkNorm: true, ropeTheta: 10_000_000
            ),
            ffn: SmeltFFNConfig(dim: 6144, activation: .swiglu),
            tiedLMHead: true
        ),
        layerPattern: SmeltLayerPattern(unit: [.delta, .delta, .delta, .attention], repeats: 6),
        quantization: SmeltQuantizationConfig(
            strategy: .affineU4, groupSize: 64,
            excludePatterns: ["conv1d_weight", "A_log", "dt_bias", "*_norm_weight"],
            quantizeEmbedding: true
        ),
        loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic, checkpointMap: .qwenHF),
        prefill: SmeltPrefillConfig(
            engine: "metal", modelPath: "", cachePath: "",
            maxBatchSize: 256,
            handoffFamilies: ["conv_state", "rec_state", "key_cache", "value_cache", "rope"]
        ),
        decode: SmeltDecodeConfig(
            policy: SmeltPackageSpec.DecodePolicy(sampler: .init(mode: .greedy), maxSteps: 512)
        ),
        inference: SmeltInferenceConfig(
            maxTokens: 512, maxTokensSource: .explicit,
            eosTokens: [248_044, 248_046], eosTokensSource: .explicit,
            chatTemplate: "chatml", thinkingPolicy: .disabled
        )
    )

    /// `qwen35-2b-affine` fixture — affine_u4 (group 64), CoreML prefill.
    /// `turboQuantH` / `preserveNative` reproduce the per-tensor quant globs the
    /// preserve_native / TQH gate tests injected into the fixture text.
    static func qwen35_2b_affine(
        turboQuantH: [String] = [],
        preserveNative: [String] = []
    ) -> SmeltModelIR {
        SmeltModelIR(
            modelName: "Qwen/Qwen3.5-2B",
            config: SmeltConfig(
                hiddenSize: 2048,
                numLayers: 24,
                vocabSize: 248_320,
                staticSeqCapacity: 256,
                ropeDim: 64,
                rmsEps: 1e-6,
                normMode: .onePlusWeight,
                delta: SmeltDeltaConfig(
                    numHeads: 16, headDim: 128, convKernel: 4,
                    qkvDim: 6144, zDim: 2048, aDim: 16, bDim: 16
                ),
                attention: SmeltAttentionConfig(
                    qHeads: 8, kvHeads: 2, headDim: 256,
                    gatedQ: true, qkNorm: true, ropeTheta: 10_000_000
                ),
                ffn: SmeltFFNConfig(dim: 6144, activation: .swiglu),
                tiedLMHead: true
            ),
            layerPattern: SmeltLayerPattern(unit: [.delta, .delta, .delta, .attention], repeats: 6),
            quantization: SmeltQuantizationConfig(
                strategy: .affineU4, groupSize: 64,
                excludePatterns: ["conv1d_weight", "A_log", "dt_bias", "*_norm_weight"],
                quantizeEmbedding: true,
                turboQuantHPatterns: turboQuantH,
                preserveNativePatterns: preserveNative
            ),
            loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic, checkpointMap: .qwenHF),
            prefill: SmeltPrefillConfig(
                engine: "coreml",
                modelPath: "models/2b_batch_prefill_pal4/model.mlmodelc",
                cachePath: "models/2b_cache",
                maxBatchSize: 64,
                handoffFamilies: ["conv_state", "rec_state", "key_cache", "value_cache", "rope"]
            ),
            decode: SmeltDecodeConfig(
                policy: SmeltPackageSpec.DecodePolicy(sampler: .init(mode: .greedy), maxSteps: 512)
            ),
            inference: SmeltInferenceConfig(
                maxTokens: 512, maxTokensSource: .explicit,
                eosTokens: [248_044, 248_046], eosTokensSource: .explicit,
                chatTemplate: "chatml", thinkingPolicy: .disabled
            )
        )
    }

    /// `qwen35-0.8b` fixture — affine_u4 (group 64), Metal prefill.
    static let qwen35_0_8B = SmeltModelIR(
        modelName: "Qwen/Qwen3.5-0.8B",
        config: SmeltConfig(
            hiddenSize: 1024,
            numLayers: 24,
            vocabSize: 248_320,
            staticSeqCapacity: 256,
            ropeDim: 64,
            rmsEps: 1e-6,
            normMode: .onePlusWeight,
            delta: SmeltDeltaConfig(
                numHeads: 16, headDim: 128, convKernel: 4,
                qkvDim: 6144, zDim: 2048, aDim: 16, bDim: 16
            ),
            attention: SmeltAttentionConfig(
                qHeads: 8, kvHeads: 2, headDim: 256,
                gatedQ: true, qkNorm: true, ropeTheta: 10_000_000
            ),
            ffn: SmeltFFNConfig(dim: 3584, activation: .swiglu),
            tiedLMHead: true
        ),
        layerPattern: SmeltLayerPattern(unit: [.delta, .delta, .delta, .attention], repeats: 6),
        quantization: SmeltQuantizationConfig(
            strategy: .affineU4, groupSize: 64,
            excludePatterns: ["conv1d_weight", "A_log", "dt_bias", "*_norm_weight"],
            quantizeEmbedding: true
        ),
        loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic, checkpointMap: .qwenHF),
        prefill: SmeltPrefillConfig(
            engine: "metal", modelPath: "", cachePath: "",
            maxBatchSize: 256,
            handoffFamilies: ["conv_state", "rec_state", "key_cache", "value_cache", "rope"]
        ),
        decode: SmeltDecodeConfig(
            policy: SmeltPackageSpec.DecodePolicy(sampler: .init(mode: .greedy), maxSteps: 512)
        ),
        inference: SmeltInferenceConfig(
            maxTokens: 512, maxTokensSource: .explicit,
            eosTokens: [248_044, 248_046], eosTokensSource: .explicit,
            chatTemplate: "chatml", thinkingPolicy: .disabled
        )
    )

    // MARK: - Llama dense text model

    /// `llama-arch-1b` fixture — dense Llama 3.2 1B Instruct, llama3 RoPE scaling.
    static let llama_arch_1b = llamaIR(
        ropeScaling: SmeltRoPEScaling(
            type: .llama3, factor: 32, lowFreqFactor: 1, highFreqFactor: 4,
            originalMaxPositionEmbeddings: 8192
        )
    )

    /// The same fixture with the `rope_scaling` block removed — the negative case
    /// for the authored-Llama-map gate (`llamaMapRequiresLlama3RoPEShape`).
    static let llama_arch_1b_noRoPEScaling = llamaIR(ropeScaling: nil)

    private static func llamaIR(ropeScaling: SmeltRoPEScaling?) -> SmeltModelIR {
        SmeltModelIR(
            modelName: "meta-llama/Llama-3.2-1B-Instruct",
            config: SmeltConfig(
                hiddenSize: 2048,
                numLayers: 16,
                vocabSize: 128_256,
                staticSeqCapacity: 4096,
                ropeDim: 64,
                rmsEps: 1e-5,
                normMode: .weight,
                attention: SmeltAttentionConfig(
                    qHeads: 32, kvHeads: 8, headDim: 64,
                    gatedQ: false, qkNorm: false,
                    ropeTheta: 500_000, ropeDim: 64, ropeLayout: .splitHalf,
                    ropeScaling: ropeScaling
                ),
                ffn: SmeltFFNConfig(dim: 8192, activation: .swiglu),
                tiedLMHead: true
            ),
            layerPattern: SmeltLayerPattern(unit: [.attention], repeats: 16),
            quantization: SmeltQuantizationConfig(
                strategy: .affineU4, groupSize: 64,
                excludePatterns: ["*_norm_weight"],
                quantizeEmbedding: true
            ),
            loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic, checkpointMap: .llamaHF),
            prefill: SmeltPrefillConfig(
                engine: "metal", modelPath: "", cachePath: "",
                maxBatchSize: 256,
                handoffFamilies: ["key_cache", "value_cache", "rope"]
            ),
            decode: SmeltDecodeConfig(
                policy: SmeltPackageSpec.DecodePolicy(sampler: .init(mode: .greedy), maxSteps: 512)
            ),
            inference: SmeltInferenceConfig(
                maxTokens: 512, maxTokensSource: .explicit,
                eosTokens: [128_001, 128_008, 128_009], eosTokensSource: .explicit,
                chatTemplate: "header-turns", thinkingPolicy: .disabled
            )
        )
    }

    // MARK: - fp32 talker-class trunks

    /// `f32-trunk` fixture — the fp32 dense trunk parity fixture (W0.3).
    static let f32_trunk = SmeltModelIR(
        modelName: "smelt-test/f32-trunk",
        config: SmeltConfig(
            hiddenSize: 256,
            numLayers: 2,
            vocabSize: 512,
            staticSeqCapacity: 64,
            ropeDim: 64,
            rmsEps: 1e-6,
            normMode: .weight,
            activationDtype: .fp32,
            portTopology: .embeddingsInHiddenOut,
            attention: SmeltAttentionConfig(
                qHeads: 4, kvHeads: 2, headDim: 64,
                gatedQ: false, qkNorm: true, qkNormMode: .weight,
                ropeTheta: 1_000_000, ropeLayout: .splitHalf
            ),
            ffn: SmeltFFNConfig(dim: 512, activation: .swiglu),
            tiedLMHead: false
        ),
        layerPattern: SmeltLayerPattern(unit: [.attention], repeats: 2),
        quantization: SmeltQuantizationConfig(
            strategy: .fp16, groupSize: 64, excludePatterns: [], quantizeEmbedding: false
        ),
        loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic),
        prefill: SmeltPrefillConfig(
            engine: "metal", modelPath: "", cachePath: "",
            maxBatchSize: 32,
            handoffFamilies: ["key_cache", "value_cache"]
        )
    )

    /// `talker-trunk` fixture — the batch-256 real-dims talker trunk.
    /// Reuses the production trunk ABI factory (dims match the fixture).
    static let talkerTrunk = SmeltModelIR.denseTrunk(
        modelName: "qwen3-tts-12hz-talker-trunk",
        hidden: 2048, numLayers: 28, vocab: 3072,
        heads: 16, kvHeads: 8, headDim: 128, inter: 6144,
        maxPrefillBatch: 256
    )

    /// `talker-trunk-smallbatch` fixture — identical but `max_prefill_batch 8`.
    static let talkerTrunkSmallBatch = SmeltModelIR.denseTrunk(
        modelName: "qwen3-tts-12hz-talker-trunk",
        hidden: 2048, numLayers: 28, vocab: 3072,
        heads: 16, kvHeads: 8, headDim: 128, inter: 6144,
        maxPrefillBatch: 8
    )

    /// The talker trunk with the fp32 activation ABI flipped to fp16 — the
    /// non-fp32 TTS pipeline that the `.talker` checkpoint map must reject.
    static let talkerTrunkFP16: SmeltModelIR = {
        let base = talkerTrunk
        let c = base.config
        return SmeltModelIR(
            modelName: base.modelName,
            config: SmeltConfig(
                hiddenSize: c.hiddenSize,
                numLayers: c.numLayers,
                vocabSize: c.vocabSize,
                staticSeqCapacity: c.staticSeqCapacity,
                ropeDim: c.ropeDim,
                rmsEps: c.rmsEps,
                normMode: c.normMode,
                activationDtype: .fp16,
                attentionConfigs: c.attentionConfigs,
                ffn: c.ffn,
                tiedLMHead: c.tiedLMHead
            ),
            layerPattern: base.layerPattern,
            quantization: base.quantization,
            loading: base.loading,
            prefill: base.prefill
        )
    }()
}
