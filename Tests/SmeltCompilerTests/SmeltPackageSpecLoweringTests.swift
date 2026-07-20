import Foundation
import Testing
@testable import SmeltCompiler
@testable import SmeltSchema

@Suite struct SmeltPackageSpecLoweringTests {

    private func camIR(_ name: String) throws -> SmeltCAMIR {
        registryModuleIR(name)
    }

    @Test func lowersCanonicalTextGenerationFixturesIntoValidCAMSpecs() throws {
        for (_, architecture, template) in [
            ("qwen35-2b", "text-generation", "chatml"),
        ] {
            let ir = FixtureModelIRs.qwen35_2B
            let spec = try SmeltPackageSpecLowering.textGeneration(from: ir)
            try spec.validate()
            let signature = SmeltPackageSpecLowering.signature(for: spec)

            #expect(signature.architecture == architecture)
            #expect(signature.blockRoutes == [
                "tokenizer:native:none",
                "trunk:compiled:baked-inline",
                "text-head:native:none",
            ])
            #expect(signature.setupPhases == ["prefill:trunk,text-head"])
            #expect(signature.perStepPhases == ["decode:trunk,text-head"])
            #expect(signature.emission == "per-step")
            #expect(signature.chatTemplate == template)
            #expect(signature.eosTokens == ir.inference.eosTokens)
            #expect(ir.decode.policySource == .explicit)
            #expect(spec.decode?.sampler.mode == ir.decode.policy?.sampler.mode)
            #expect(spec.decode?.maxSteps == ir.inference.maxTokens)
            #expect(spec.validation.performanceGate == SmeltPackagePerformanceGateID.textDecodePrefillStartup)
            #expect(spec.validation.performanceProfile?.gate == SmeltPackagePerformanceGateID.textDecodePrefillStartup)
            #expect(spec.validation.performanceProfile?.command == .run)
            #expect(spec.validation.performanceProfile?.requiredOutputMetrics
                == SmeltPackagePerformanceMetricName.textDecodePrefillStartupRequired)
            #expect(spec.validation.performanceProfile?.maxBounds == [
                .init(
                    metric: SmeltPackagePerformanceMetricName.traceFirstTokenMS,
                    max: SmeltPackagePerformanceBudget.qwen35TextTraceFirstTokenMaxMS,
                    unit: SmeltPackagePerformanceUnit.milliseconds
                ),
            ])
            #expect(signature.outputFiles.contains("manifest.json"))
            #expect(signature.outputFiles.contains("weights.bin"))
            #expect(signature.outputFiles.contains("model.metallib"))
            #expect(signature.outputFiles.contains("SmeltGenerated.swift"))
            #expect(signature.outputFiles.contains("dispatches.bin"))
            #expect(signature.outputFiles.contains("tokenizer.json"))
            #expect(signature.outputFiles.contains("tokenizer.bin"))
            #expect(spec.tensors.map(\.canonicalName) == SmeltWeightLayout.computeLayout(from: ir).map(\.name))

            let legacy = try SmeltPackageAssemblySignature(
                textManifest: legacyManifest(from: ir))
            #expect(signature == legacy)
        }
    }

    @Test func qwen35TextCAMParityProjectionMatchesLegacyPackagePlan() throws {
        let legacy = FixtureModelIRs.qwen35_2b_affine_metalprefill
        let cam = try camIR("qwen35_text.cam")
        let projection = try Qwen35TextLegacyParityOracle.project(cam: cam, legacy: legacy)
        let legacyPlan = try SmeltPackageResolvedPlan.resolve(
            SmeltPackageSpecLowering.textGeneration(from: legacy)
        )
        let expectedPackage = Qwen35TextPackageParitySignature(plan: legacyPlan)
        let expectedCAMPackageFiles = (expectedPackage.packageFiles + [
            SmeltCAMPackageDescriptor.packageFileName,
        ]).sorted()

        #expect(projection.package == expectedPackage)
        #expect(projection.semanticSHA256.count == 64)
        #expect(projection.exportABISHA256.count == 64)
        #expect(projection.modelLocator == legacy.modelName)
        #expect(projection.tokenizerLocator == "\(legacy.modelName)/tokenizer.json")
        #expect(projection.prefillSourcePath == nil)
        #expect(projection.prefillCachePath == nil)
        #expect(projection.prefillBatch == legacy.prefill?.maxBatchSize)
        #expect(projection.tensorExpansion == .patternOnly)
        #expect(projection.tensorPatterns == ["weights.* -> trunk.*"])
        #expect(projection.quantization == Qwen35TextQuantizationSignature(
            storageFormat: "affine-u4",
            groupSize: 64,
            quantizeEmbeddings: true,
            preservedPatterns: ["*_norm_weight", "A_log", "conv1d_weight", "dt_bias"].sorted()
        ))
        #expect(projection.compile == Qwen35TextCompileSignature(
            target: "metal",
            prefill: "metal batch 256",
            layout: "memory-neutral"
        ))
        #expect(projection.gates.startupElapsedMaxMS == 100)
        #expect(projection.gates.inventoryFiles == expectedCAMPackageFiles)
        #expect(projection.bridgeDefaults == [
            "attention.gated_q:true",
            "chat_template:chatml",
            "loading:mmap_prefault/monolithic",
            "sampler:greedy",
            "static_seq_capacity:256",
            "thinking_policy:disabled",
        ])
    }

    @Test func qwen35TextCAMParityProjectionRejectsNonQwenTextCAM() throws {
        let legacy = FixtureModelIRs.qwen35_2B
        let cam = try camIR("qwen35_fast.cam")

        #expect(throws: Qwen35TextLegacyParityError.self) {
            try Qwen35TextLegacyParityOracle.project(cam: cam, legacy: legacy)
        }
    }

    @Test func qwen35TextCAMParityProjectionRejectsSemanticDrift() throws {
        let legacy = FixtureModelIRs.qwen35_2B
        let cam = try camIR("qwen35_text.cam")

        #expect(throws: Qwen35TextLegacyParityError.self) {
            try Qwen35TextLegacyParityOracle.project(
                cam: copyCAMIR(cam, flows: replacingMaxSteps(in: cam.flows, with: 256)),
                legacy: legacy
            )
        }
        #expect(throws: Qwen35TextLegacyParityError.self) {
            try Qwen35TextLegacyParityOracle.project(
                cam: copyCAMIR(cam, quantization: replacingDefaultQuantGroup(in: cam.quantization, with: 32)),
                legacy: legacy
            )
        }
        #expect(throws: Qwen35TextLegacyParityError.self) {
            try Qwen35TextLegacyParityOracle.project(
                cam: copyCAMIR(cam, gates: removingInventoryFile("prefill_dispatches.bin", from: cam.gates)),
                legacy: legacy
            )
        }
    }

    @Test func lowererRecordsPrefillArtifactsWhenIRHasPrefill() throws {
        let ir = FixtureModelIRs.qwen35_2b_affine_metalprefill
        #expect(ir.prefill != nil)

        let spec = try SmeltPackageSpecLowering.textGeneration(from: ir)
        try spec.validate()
        let signature = SmeltPackageSpecLowering.signature(for: spec)

        #expect(signature.outputFiles.contains("prefill_dispatches.bin"))
        #expect(!signature.outputFiles.contains("prefill_verify_argmax_dispatches.bin"))
    }

    @Test func lowererRecordsVerifyArgmaxArtifactWhenEnabled() throws {
        let base = FixtureModelIRs.qwen35_2b_affine_metalprefill
        let prefill = SmeltPrefillConfig(
            engine: base.prefill!.engine,
            modelPath: base.prefill!.modelPath,
            cachePath: base.prefill!.cachePath,
            maxBatchSize: base.prefill!.maxBatchSize,
            handoffFamilies: base.prefill!.handoffFamilies,
            emitAllLogits: base.prefill!.emitAllLogits,
            verifyArgmax: true
        )
        let ir = SmeltModelIR(
            modelName: base.modelName,
            config: base.config,
            layerPattern: base.layerPattern,
            quantization: base.quantization,
            loading: base.loading,
            compilation: base.compilation,
            runtime: base.runtime,
            prefill: prefill,
            decode: base.decode,
            inference: base.inference
        )

        let spec = try SmeltPackageSpecLowering.textGeneration(from: ir)
        try spec.validate()
        let signature = SmeltPackageSpecLowering.signature(for: spec)

        #expect(signature.outputFiles.contains("prefill_dispatches.bin"))
        #expect(signature.outputFiles.contains("prefill_verify_argmax_dispatches.bin"))
    }

    @Test func lowersCompleteTextManifestIntoValidCAMSpec() throws {
        let ir = FixtureModelIRs.qwen35_2b_affine_metalprefill
        let manifest = legacyManifest(from: ir)

        let spec = try SmeltPackageSpecLowering.textGeneration(
            from: manifest,
            packageName: "qwen35-2b-affine-metalprefill.smeltpkg",
            sourcePath: "legacy-package"
        )
        try spec.validate()
        let plan = try SmeltPackageResolvedPlan.resolve(spec)
        let legacy = try SmeltPackageAssemblySignature(textManifest: manifest)

        #expect(SmeltPackageSpecLowering.signature(for: spec) == legacy)
        #expect(plan.sources.map(\.locator) == ["legacy-package"])
        #expect(plan.policy.mode == .textGeneration)
        #expect(plan.policy.inference?.chatTemplate == "chatml")
        #expect(plan.validationPerformanceGate == SmeltPackagePerformanceGateID.textDecodePrefillStartup)
        #expect(plan.packageFiles.map(\.path).contains("prefill_dispatches.bin"))
        #expect(plan.packageFiles.map(\.path).contains("tokenizer.bin"))
    }

    @Test func textManifestLoweringPreservesPackageOwnedDecodePolicy() throws {
        let ir = FixtureModelIRs.qwen35_2B
        let decode = SmeltPackageSpec.DecodePolicy(
            sampler: .init(mode: .sample, temperature: 0.8, topK: 40, topP: 0.9),
            maxSteps: ir.inference.maxTokens
        )
        let manifest = copyLegacyManifest(legacyManifest(from: ir), decode: decode)

        let spec = try SmeltPackageSpecLowering.textGeneration(from: manifest)

        #expect(spec.decode?.sampler.mode == .sample)
        #expect(spec.decode?.sampler.temperature == 0.8)
        #expect(spec.decode?.sampler.topK == 40)
        #expect(spec.decode?.sampler.topP == 0.9)
        #expect(spec.decode?.maxSteps == ir.inference.maxTokens)
    }

    @Test func textManifestLoweringRejectsMissingPackageOwnedDecodePolicy() throws {
        let ir = FixtureModelIRs.qwen35_2B
        let manifest = copyLegacyManifest(legacyManifest(from: ir), dropDecode: true)

        #expect(throws: SmeltPackageSpecLoweringError.self) {
            try SmeltPackageSpecLowering.textGeneration(from: manifest)
        }
    }

    @Test func textManifestLoweringUsesGraphInsteadOfModelName() throws {
        let ir = FixtureModelIRs.qwen35_2B
        let manifest = copyLegacyManifest(
            legacyManifest(from: ir),
            modelName: "acme/RenamedTextModel"
        )

        let spec = try SmeltPackageSpecLowering.textGeneration(from: manifest)
        let legacy = try SmeltPackageAssemblySignature(textManifest: manifest)

        #expect(spec.modelName == "acme/RenamedTextModel")
        #expect(spec.runtime.architecture == SmeltRuntimeGraphPolicy.textGeneration.rawValue)
        #expect(legacy.architecture == SmeltRuntimeGraphPolicy.textGeneration.rawValue)
    }

    @Test func textManifestLoweringRequiresCompleteRunnableInventory() throws {
        let ir = FixtureModelIRs.qwen35_2B
        let manifest = legacyManifest(from: ir)
        let checksums = SmeltManifestChecksums(
            weightsBin: manifest.checksums.weightsBin,
            metallib: manifest.checksums.metallib,
            generatedSwift: manifest.checksums.generatedSwift,
            dispatchesBin: manifest.checksums.dispatchesBin,
            prefillDispatchesBin: manifest.checksums.prefillDispatchesBin,
            prefillVerifyArgmaxDispatchesBin: manifest.checksums.prefillVerifyArgmaxDispatchesBin,
            tokenizerJSON: nil
        )
        let incomplete = copyLegacyManifest(manifest, checksums: checksums)

        #expect(throws: SmeltPackageSpecLoweringError.self) {
            try SmeltPackageSpecLowering.textGeneration(from: incomplete)
        }
    }

    @Test func lowererDerivesTextGenerationRuntimeFromGraphInsteadOfModelName() throws {
        let base = FixtureModelIRs.qwen35_2B
        let ir = SmeltModelIR(
            modelName: "acme/RenamedTextModel",
            config: base.config,
            layerPattern: base.layerPattern,
            quantization: base.quantization,
            loading: base.loading,
            compilation: base.compilation,
            runtime: base.runtime,
            prefill: base.prefill,
            decode: base.decode,
            inference: base.inference
        )

        let spec = try SmeltPackageSpecLowering.textGeneration(from: ir)
        let plan = try SmeltPackageResolvedPlan.resolve(spec)
        #expect(spec.runtime.architecture == SmeltRuntimeGraphPolicy.textGeneration.rawValue)
        #expect(plan.runtime.architecture == SmeltRuntimeGraphPolicy.textGeneration.rawValue)
    }

    @Test func lowererRejectsTextGenerationRuntimeArchitectureSelectorBeforeSpecValidation() throws {
        let base = FixtureModelIRs.qwen35_2B
        let ir = SmeltModelIR(
            modelName: base.modelName,
            config: base.config,
            layerPattern: base.layerPattern,
            quantization: base.quantization,
            loading: base.loading,
            compilation: base.compilation,
            runtime: SmeltRuntimePolicyConfig(
                architecture: SmeltRuntimeGraphPolicy.textGeneration.rawValue
            ),
            prefill: base.prefill,
            decode: base.decode,
            inference: base.inference
        )

        #expect(throws: SmeltPackageSpecLoweringError.self) {
            try SmeltPackageSpecLowering.textGeneration(from: ir)
        }

        let explicitSourceOnly = SmeltModelIR(
            modelName: base.modelName,
            config: base.config,
            layerPattern: base.layerPattern,
            quantization: base.quantization,
            loading: base.loading,
            compilation: base.compilation,
            runtime: SmeltRuntimePolicyConfig(
                architecture: nil,
                architectureSource: .explicit
            ),
            prefill: base.prefill,
            decode: base.decode,
            inference: base.inference
        )

        #expect(throws: SmeltPackageSpecLoweringError.self) {
            try SmeltPackageSpecLowering.textGeneration(from: explicitSourceOnly)
        }
    }

    @Test func lowererRequiresExplicitTextGenerationThinkingPolicyBeforeSpecValidation() throws {
        let base = FixtureModelIRs.qwen35_2B
        let inference = SmeltInferenceConfig(
            maxTokens: base.inference.maxTokens,
            maxTokensSource: base.inference.maxTokensSource,
            eosTokens: base.inference.eosTokens,
            eosTokensSource: base.inference.eosTokensSource,
            thinkToken: base.inference.thinkToken,
            thinkEndToken: base.inference.thinkEndToken,
            thinkSkipSuffix: base.inference.thinkSkipSuffix,
            chatTemplate: base.inference.chatTemplate,
            thinkingPolicy: nil
        )
        let ir = SmeltModelIR(
            modelName: base.modelName,
            config: base.config,
            layerPattern: base.layerPattern,
            quantization: base.quantization,
            loading: base.loading,
            compilation: base.compilation,
            runtime: base.runtime,
            prefill: base.prefill,
            decode: base.decode,
            inference: inference
        )

        #expect(throws: SmeltPackageSpecLoweringError.self) {
            try SmeltPackageSpecLowering.textGeneration(from: ir)
        }
    }

    @Test func lowererRequiresExplicitTextGenerationMaxTokensBeforeSpecValidation() throws {
        let base = FixtureModelIRs.qwen35_2B
        let inference = SmeltInferenceConfig(
            maxTokens: base.inference.maxTokens,
            maxTokensSource: .modelPreset,
            eosTokens: base.inference.eosTokens,
            eosTokensSource: base.inference.eosTokensSource,
            thinkToken: base.inference.thinkToken,
            thinkEndToken: base.inference.thinkEndToken,
            thinkSkipSuffix: base.inference.thinkSkipSuffix,
            chatTemplate: base.inference.chatTemplate,
            thinkingPolicy: base.inference.thinkingPolicy
        )
        let ir = SmeltModelIR(
            modelName: base.modelName,
            config: base.config,
            layerPattern: base.layerPattern,
            quantization: base.quantization,
            loading: base.loading,
            compilation: base.compilation,
            runtime: base.runtime,
            prefill: base.prefill,
            decode: base.decode,
            inference: inference
        )

        #expect(throws: SmeltPackageSpecLoweringError.self) {
            try SmeltPackageSpecLowering.textGeneration(from: ir)
        }
    }

    @Test func lowererRequiresExplicitTextGenerationEOSTokensBeforeSpecValidation() throws {
        let base = FixtureModelIRs.qwen35_2B
        let inference = SmeltInferenceConfig(
            maxTokens: base.inference.maxTokens,
            maxTokensSource: base.inference.maxTokensSource,
            eosTokens: base.inference.eosTokens,
            eosTokensSource: .modelPreset,
            thinkToken: base.inference.thinkToken,
            thinkEndToken: base.inference.thinkEndToken,
            thinkSkipSuffix: base.inference.thinkSkipSuffix,
            chatTemplate: base.inference.chatTemplate,
            thinkingPolicy: base.inference.thinkingPolicy
        )
        let ir = SmeltModelIR(
            modelName: base.modelName,
            config: base.config,
            layerPattern: base.layerPattern,
            quantization: base.quantization,
            loading: base.loading,
            compilation: base.compilation,
            runtime: base.runtime,
            prefill: base.prefill,
            decode: base.decode,
            inference: inference
        )

        #expect(throws: SmeltPackageSpecLoweringError.self) {
            try SmeltPackageSpecLowering.textGeneration(from: ir)
        }
    }

    @Test func lowererRequiresExplicitTextGenerationChatTemplateBeforeSpecValidation() throws {
        let base = FixtureModelIRs.qwen35_2B
        let inference = SmeltInferenceConfig(
            maxTokens: base.inference.maxTokens,
            maxTokensSource: base.inference.maxTokensSource,
            eosTokens: base.inference.eosTokens,
            eosTokensSource: base.inference.eosTokensSource,
            thinkToken: base.inference.thinkToken,
            thinkEndToken: base.inference.thinkEndToken,
            thinkSkipSuffix: base.inference.thinkSkipSuffix,
            chatTemplate: nil,
            thinkingPolicy: base.inference.thinkingPolicy
        )
        let ir = SmeltModelIR(
            modelName: base.modelName,
            config: base.config,
            layerPattern: base.layerPattern,
            quantization: base.quantization,
            loading: base.loading,
            compilation: base.compilation,
            runtime: base.runtime,
            prefill: base.prefill,
            decode: base.decode,
            inference: inference
        )

        #expect(throws: SmeltPackageSpecLoweringError.self) {
            try SmeltPackageSpecLowering.textGeneration(from: ir)
        }
    }

    @Test func lowererRequiresExplicitTextGenerationDecodePolicyBeforeSpecValidation() throws {
        let base = FixtureModelIRs.qwen35_2B
        let ir = SmeltModelIR(
            modelName: base.modelName,
            config: base.config,
            layerPattern: base.layerPattern,
            quantization: base.quantization,
            loading: base.loading,
            compilation: base.compilation,
            runtime: base.runtime,
            prefill: base.prefill,
            decode: SmeltDecodeConfig(
                policy: base.decode.policy,
                policySource: .modelPreset
            ),
            inference: base.inference
        )

        #expect(throws: SmeltPackageSpecLoweringError.self) {
            try SmeltPackageSpecLowering.textGeneration(from: ir)
        }
    }

    @Test func lowererRequiresTextGenerationDecodeMaxStepsBeforeSpecValidation() throws {
        let base = FixtureModelIRs.qwen35_2B
        let decode = SmeltDecodeConfig(
            policy: SmeltPackageSpec.DecodePolicy(
                sampler: base.decode.policy!.sampler
            )
        )
        let ir = SmeltModelIR(
            modelName: base.modelName,
            config: base.config,
            layerPattern: base.layerPattern,
            quantization: base.quantization,
            loading: base.loading,
            compilation: base.compilation,
            runtime: base.runtime,
            prefill: base.prefill,
            decode: decode,
            inference: base.inference
        )

        #expect(throws: SmeltPackageSpecLoweringError.self) {
            try SmeltPackageSpecLowering.textGeneration(from: ir)
        }
    }

    @Test func lowererRejectsTextGenerationDecodeMaxStepsDriftBeforeSpecValidation() throws {
        let base = FixtureModelIRs.qwen35_2B
        let decode = SmeltDecodeConfig(
            policy: SmeltPackageSpec.DecodePolicy(
                sampler: base.decode.policy!.sampler,
                maxSteps: base.inference.maxTokens + 1
            )
        )
        let ir = SmeltModelIR(
            modelName: base.modelName,
            config: base.config,
            layerPattern: base.layerPattern,
            quantization: base.quantization,
            loading: base.loading,
            compilation: base.compilation,
            runtime: base.runtime,
            prefill: base.prefill,
            decode: decode,
            inference: base.inference
        )

        #expect(throws: SmeltPackageSpecLoweringError.self) {
            try SmeltPackageSpecLowering.textGeneration(from: ir)
        }
    }

    @Test func lowersQwen3TTSSpecsIntoValidCAMSpec() throws {
        let profile = SmeltQwen3TTSPackageProfiles.runnable
        let decode = Qwen3TTSManifest.Decode(
            doSample: true,
            temperature: 0.7,
            topK: 12,
            subtalkerTemperature: 0.8,
            subtalkerTopK: 8
        )
        let specs = qwen3TTSRunnableSpecs(textEmbeddingDType: .bf16)
        let spec = try SmeltPackageSpecLowering.qwen3TTS(
            from: specs,
            packageName: "qwen3-tts.smeltpkg",
            sourcePath: "fixtures/qwen3-tts",
            eosTokens: profile.eosTokens,
            decode: decode,
            tensorBlocks: qwen3TTSTestTensorBlocks(for: specs),
            tensorSourceDTypes: qwen3TTSTestTensorSourceDTypes(for: specs),
            pipelines: ["qwen_kernel"],
            pageSize: 16
        )

        try spec.validate()
        let plan = try SmeltPackageResolvedPlan.resolve(spec)

        #expect(plan.runtime.architecture == profile.runtimeArchitecture)
        #expect(plan.policy.mode == .sidecarTextToCodecAudio)
        #expect(plan.validationPerformanceGate == profile.performanceGate)
        #expect(plan.validationPerformanceProfile?.requiredOutputMetrics
            == SmeltPackagePerformanceMetricName.qwen3TTSTTFARequired.sorted())
        #expect(plan.validationPerformanceProfile?.maxBounds == [
            .init(
                metric: SmeltPackagePerformanceMetricName.firstAudioSeconds,
                max: SmeltPackagePerformanceBudget.qwen3TTSFirstAudioMaxSeconds,
                unit: SmeltPackagePerformanceUnit.seconds
            ),
            .init(
                metric: SmeltPackagePerformanceMetricName.realTimeFactor,
                max: SmeltPackagePerformanceBudget.qwen3TTSRealTimeFactorMax,
                unit: SmeltPackagePerformanceUnit.realTimeFactor
            ),
            .init(
                metric: SmeltPackagePerformanceMetricName.ttfaSeconds,
                max: SmeltPackagePerformanceBudget.qwen3TTSTTFAMaxSeconds,
                unit: SmeltPackagePerformanceUnit.seconds
            ),
        ])
        #expect(plan.validationStructureProfile?.id == SmeltPackageStructureProfileID.qwen3TTSRunnable)
        #expect(plan.validationStructureProfile?.requiredPipelines == ["qwen_kernel"])
        #expect(plan.validationStructureProfile?.requiredFiles == [
            "config.json",
            "manifest.json",
            "merges.txt",
            "model.metallib",
            "tokenizer_config.json",
            "trunk",
            "trunk-mtp",
            "vocab.json",
            "weights.bin",
        ])
        #expect(plan.runtime.routes.map(\.signature) == [
            "tts-frontend:compiled:runtime-emit",
            "talker:compiled:baked-sidecar",
            "codec-head:native:none",
            "mtp-head:native:internal-sidecar",
            "codec-decoder:compiled:runtime-emit",
        ])
        #expect(plan.validationStructureProfile?.requiredRoutes == plan.runtime.routes.map(\.signature).sorted())
        #expect(plan.runtime.setupPhases == ["prefill:talker,codec-head"])
        #expect(plan.runtime.perStepPhases == [
            "mtp:mtp-head",
            "advance:talker,codec-head:feeds-next-step",
        ])
        #expect(plan.policy.mode == .sidecarTextToCodecAudio)
        #expect(plan.policy.tokenizer?.format == "byte-bpe")
        #expect(plan.policy.tokenizer?.files == profile.tokenizerFiles.sorted())
        #expect(plan.policy.inference?.eosTokens == profile.eosTokens)
        #expect(plan.policy.decode?.sampler.mode == .sample)
        #expect(abs((plan.policy.decode?.sampler.temperature ?? 0) - 0.7) < 0.0001)
        #expect(plan.policy.decode?.sampler.topK == 12)
        #expect(plan.policy.decode?.subSampler?.mode == .sample)
        #expect(abs((plan.policy.decode?.subSampler?.temperature ?? 0) - 0.8) < 0.0001)
        #expect(plan.policy.decode?.subSampler?.topK == 8)
        #expect(plan.packageFiles.map(\.path) == [
            "config.json",
            "manifest.json",
            "merges.txt",
            "model.metallib",
            "tokenizer_config.json",
            "trunk",
            "trunk-mtp",
            "vocab.json",
            "weights.bin",
        ])
        let owners = Dictionary(uniqueKeysWithValues: plan.tensors.map {
            ($0.canonicalName, $0.block)
        })
        #expect(owners["decoder.pre_conv.conv.weight"] == "codec-decoder")
        #expect(owners["talker.codec_head.weight"] == "codec-head")
        #expect(owners["talker.code_predictor.lm_head.0.weight"] == "mtp-head")
        #expect(owners["talker.code_predictor.model.layers.0.self_attn.q_proj.weight"] == "mtp-head")
        #expect(owners["talker.model.layers.0.self_attn.q_proj.weight"] == "talker")
        #expect(owners["talker.model.text_embedding.weight"] == "tts-frontend")
        #expect(owners["talker.text_projection.linear_fc1.weight"] == "tts-frontend")
        let sourceDTypes = Dictionary(uniqueKeysWithValues: plan.tensors.map {
            ($0.canonicalName, $0.sourceDType)
        })
        #expect(sourceDTypes["decoder.pre_conv.conv.weight"] == .f32)
        #expect(sourceDTypes["talker.codec_head.weight"] == .bf16)
        #expect(sourceDTypes["talker.model.text_embedding.weight"] == .bf16)
        #expect(plan.tensors.count == specs.count)
        #expect(plan.architectureConfigSignature.contains("\"sidecars\":[\"trunk\",\"trunk-mtp\"]"))
        #expect(plan.signature.lines.contains("validation-structure-required-pipeline:qwen_kernel"))
        #expect(plan.signature.lines.contains("validation-structure-required-route:talker:compiled:baked-sidecar"))
        let expectedTotalBytes = Qwen3TTSPackageBuilder.planLayout(
            specs.sorted { $0.name < $1.name },
            pageSize: 16
        ).totalBytes
        #expect(plan.architectureConfigSignature.contains("\"total_bytes\":\(expectedTotalBytes)"))
    }

    @Test func qwen3TTSLoweringDefaultsComeFromPackageProfile() throws {
        let profile = SmeltQwen3TTSPackageProfiles.runnable
        let specs = qwen3TTSRunnableSpecs(textEmbeddingDType: .bf16)
        let spec = try SmeltPackageSpecLowering.qwen3TTS(
            from: specs,
            sourcePath: "fixtures/qwen3-tts",
            tensorBlocks: qwen3TTSTestTensorBlocks(for: specs),
            tensorSourceDTypes: qwen3TTSTestTensorSourceDTypes(for: specs)
        )
        let plan = try SmeltPackageResolvedPlan.resolve(spec)

        #expect(spec.packageName == profile.packageName)
        #expect(spec.modelName == profile.modelName)
        #expect(spec.blocks == profile.compiledTalkerGraph)
        #expect(spec.loop == profile.loop)
        #expect(spec.runtime.architecture == profile.runtimeArchitecture)
        #expect(spec.sidecars.map(\.path) == profile.sidecarPaths)
        #expect(spec.sidecars.map(\.id) == profile.sidecars.map(\.id))
        #expect(spec.sidecars.map(\.kind) == profile.sidecars.map(\.kind))
        #expect(spec.tokenizer?.files == profile.tokenizerFiles)
        #expect(spec.inference?.maxTokens == profile.maxTokens)
        #expect(spec.inference?.eosTokens == profile.eosTokens)
        #expect(spec.validation.performanceGate == profile.performanceGate)
        #expect(spec.validation.structureProfile?.id == profile.structureProfileID)
        #expect(plan.policy.mode == .sidecarTextToCodecAudio)
        #expect(plan.policy.tokenizer?.files == profile.tokenizerFiles.sorted())
        #expect(plan.policy.inference?.maxTokens == profile.maxTokens)
        #expect(plan.policy.inference?.eosTokens == profile.eosTokens)
        #expect(plan.architectureConfigSignature.contains("\"architecture\":\"\(profile.runtimeArchitecture)\""))
        #expect(plan.architectureConfigSignature.contains("\"page_size\":\(profile.pageSize)"))
        #expect(plan.architectureConfigSignature.contains("\"sidecars\":[\"trunk\",\"trunk-mtp\"]"))
    }

    @Test func qwen3TTSLoweringRecordsNativeFrontEndWhenTextEmbeddingIsNotBF16() throws {
        let specs = qwen3TTSRunnableSpecs(textEmbeddingDType: .f32)
        let spec = try SmeltPackageSpecLowering.qwen3TTS(
            from: specs,
            sourcePath: "fixtures/qwen3-tts",
            tensorBlocks: qwen3TTSTestTensorBlocks(for: specs),
            tensorSourceDTypes: qwen3TTSTestTensorSourceDTypes(for: specs),
            pageSize: 16
        )
        try spec.validate()
        let plan = try SmeltPackageResolvedPlan.resolve(spec)

        #expect(plan.runtime.routes.first?.signature == "tts-frontend:native:none")
    }

    @Test func qwen3TTSLoweringRejectsInvalidSubtalkerSamplingPolicy() throws {
        let specs = qwen3TTSRunnableSpecs(textEmbeddingDType: .bf16)
        #expect(throws: SmeltPackageSpecLoweringError.self) {
            try SmeltPackageSpecLowering.qwen3TTS(
                from: specs,
                decode: Qwen3TTSManifest.Decode(
                    doSample: true,
                    temperature: 0.7,
                    topK: 12,
                    subtalkerTemperature: 0,
                    subtalkerTopK: 8
                ),
                tensorBlocks: qwen3TTSTestTensorBlocks(for: specs),
                tensorSourceDTypes: qwen3TTSTestTensorSourceDTypes(for: specs)
            )
        }
    }

    @Test func qwen3TTSLoweringRequiresExactTokenizerFileSet() throws {
        let specs = qwen3TTSRunnableSpecs(textEmbeddingDType: .bf16)
        #expect(throws: SmeltPackageSpecLoweringError.self) {
            try SmeltPackageSpecLowering.qwen3TTS(
                from: specs,
                tokenizerFiles: ["vocab.json"],
                tensorBlocks: qwen3TTSTestTensorBlocks(for: specs),
                tensorSourceDTypes: qwen3TTSTestTensorSourceDTypes(for: specs)
            )
        }
    }

    @Test func qwen3TTSLoweringRejectsCompiledTrunkConfigMismatch() throws {
        let temp = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temp) }
        let config = """
        {
          "talker_config": {
            "rms_norm_eps": 0.00001,
            "rope_theta": 1000000,
            "code_predictor_config": {
              "rms_norm_eps": 0.000001,
              "rope_theta": 1000000
            }
          }
        }
        """
        try Data(config.utf8).write(to: temp.appendingPathComponent("config.json"))

        let specs = qwen3TTSRunnableSpecs(textEmbeddingDType: .bf16)
        #expect(throws: SmeltPackageSpecLoweringError.self) {
            try SmeltPackageSpecLowering.qwen3TTS(
                from: specs,
                sourcePath: temp.path,
                tensorBlocks: qwen3TTSTestTensorBlocks(for: specs),
                tensorSourceDTypes: qwen3TTSTestTensorSourceDTypes(for: specs)
            )
        }
    }

    @Test func qwen3TTSLoweringRejectsCodecOnlyPackages() throws {
        let specs: [Qwen3TTSPackageBuilder.WeightSpec] = [
            .init(name: "decoder.pre_conv.conv.weight", shape: [2, 2]),
        ]
        #expect(throws: SmeltPackageSpecLoweringError.self) {
            try SmeltPackageSpecLowering.qwen3TTS(
                from: specs,
                sourcePath: "fixtures/qwen3-tts",
                tensorBlocks: qwen3TTSTestTensorBlocks(for: specs),
                tensorSourceDTypes: qwen3TTSTestTensorSourceDTypes(for: specs)
            )
        }
    }

    private func legacyManifest(from ir: SmeltModelIR) -> SmeltManifest {
        let prefillDispatches: String? = ir.prefill == nil ? nil : "prefill"
        let verifyArgmax: String? = ir.prefill?.verifyArgmax == true ? "verify" : nil
        return SmeltManifest(
            blocks: .tokenFeedbackText,
            loop: .tokenFeedbackText,
            modelName: ir.modelName,
            config: SmeltManifestConfig(
                hiddenSize: ir.config.hiddenSize,
                numLayers: ir.config.numLayers,
                vocabSize: ir.config.vocabSize,
                staticSeqCapacity: ir.config.staticSeqCapacity,
                ropeDim: ir.config.ropeDim,
                numDeltaLayers: ir.numDeltaLayers,
                numAttnLayers: ir.numAttnLayers,
                ffnDim: ir.config.ffn.dim,
                sharedKVLayers: ir.config.sharedKVLayers
            ),
            context: nil,
            checksums: SmeltManifestChecksums(
                weightsBin: "weights",
                metallib: "metallib",
                generatedSwift: "generated",
                dispatchesBin: "dispatches",
                prefillDispatchesBin: prefillDispatches,
                prefillVerifyArgmaxDispatchesBin: verifyArgmax,
                tokenizerJSON: "tokenizer"
            ),
            device: SmeltDeviceRequirements(
                metalFamily: .apple7,
                minMemoryBytes: 1
            ),
            weights: SmeltWeightManifest(
                totalBytes: 0,
                entries: SmeltWeightLayout.computeLayout(from: ir)
            ),
            buffers: SmeltBufferTable(slots: []),
            pipelines: [],
            slotLayout: SmeltSlotLayout(
                convStateBaseSlot: 0,
                recStateBaseSlot: 0,
                keyCacheBaseSlot: 0,
                valCacheBaseSlot: 0,
                ropeCosSlot: 0,
                ropeSinSlot: 0,
                tokenIdSlot: 0,
                positionSlot: 0,
                weightsSlot: 0
            ),
            inference: SmeltInferenceManifest(
                maxTokens: ir.inference.maxTokens,
                eosTokens: ir.inference.eosTokens,
                thinkToken: ir.inference.thinkToken,
                thinkEndToken: ir.inference.thinkEndToken,
                thinkSkipSuffix: ir.inference.thinkSkipSuffix,
                chatTemplate: ir.inference.chatTemplate,
                thinkingPolicy: ir.inference.thinkingPolicy
            ),
            decode: ir.decode.policy
        )
    }

    private func copyLegacyManifest(
        _ manifest: SmeltManifest,
        modelName: String? = nil,
        checksums: SmeltManifestChecksums? = nil,
        decode: SmeltPackageSpec.DecodePolicy? = nil,
        dropDecode: Bool = false
    ) -> SmeltManifest {
        SmeltManifest(
            version: manifest.version,
            kind: manifest.kind,
            headlessTrunkABI: manifest.headlessTrunkABI,
            blocks: manifest.blocks,
            loop: manifest.loop,
            modelName: modelName ?? manifest.modelName,
            config: manifest.config,
            context: manifest.context,
            checksums: checksums ?? manifest.checksums,
            buildProvenance: manifest.buildProvenance,
            device: manifest.device,
            weights: manifest.weights,
            buffers: manifest.buffers,
            pipelines: manifest.pipelines,
            slotLayout: manifest.slotLayout,
            prefill: manifest.prefill,
            inference: manifest.inference,
            decode: dropDecode ? nil : (decode ?? manifest.decode),
            optimizationReport: manifest.optimizationReport
        )
    }
}

private enum Qwen35TextLegacyParityError: Error, CustomStringConvertible, Equatable {
    case mismatch(String)

    var description: String {
        switch self {
        case .mismatch(let reason): return "qwen35_text parity: \(reason)"
        }
    }
}

private enum Qwen35TextTensorExpansion: String, Equatable {
    case patternOnly = "pattern-only"
}

private struct Qwen35TextQuantizationSignature: Equatable {
    let storageFormat: String
    let groupSize: Int
    let quantizeEmbeddings: Bool
    let preservedPatterns: [String]
}

private struct Qwen35TextCompileSignature: Equatable {
    let target: String
    let prefill: String
    let layout: String
}

private struct Qwen35TextGateSignature: Equatable {
    let startupElapsedMaxMS: Int
    let inventoryFiles: [String]
}

private struct Qwen35TextPackageParitySignature: Equatable {
    let modelName: String
    let sourceLocators: [String]
    let commands: [String]
    let routes: [String]
    let setupPhases: [String]
    let perStepPhases: [String]
    let emission: String
    let packageFiles: [String]
    let quantizationFormat: String?
    let quantizationGroupSize: Int?
    let maxTokens: Int?
    let eosTokens: [Int32]
    let chatTemplate: String?
    let thinkingPolicy: String?
    let decodeSampler: String?
    let decodeMaxSteps: Int?
    let validationPerformanceGate: String?
    let validationPerformanceCommand: String?
    let validationRequiredOutputMetrics: [String]
    let validationMaxBounds: [String]

    init(plan: SmeltPackageResolvedPlan) {
        modelName = plan.modelName
        sourceLocators = plan.sources.map { "\($0.id):\($0.locator):\($0.revision ?? "none")" }
        commands = plan.runtime.commands.map(\.rawValue)
        routes = plan.runtime.routes.map(\.signature)
        setupPhases = plan.runtime.setupPhases
        perStepPhases = plan.runtime.perStepPhases
        emission = plan.runtime.emission
        packageFiles = plan.packageFiles.map(\.path).sorted()
        quantizationFormat = plan.quantization?.format.rawValue
        quantizationGroupSize = plan.quantization?.groupSize
        maxTokens = plan.policy.inference?.maxTokens
        eosTokens = plan.policy.inference?.eosTokens ?? []
        chatTemplate = plan.policy.inference?.chatTemplate
        thinkingPolicy = plan.policy.inference?.thinkingPolicy?.rawValue
        decodeSampler = plan.policy.decode?.sampler.mode.rawValue
        decodeMaxSteps = plan.policy.decode?.maxSteps
        validationPerformanceGate = plan.validationPerformanceGate
        validationPerformanceCommand = plan.validationPerformanceProfile?.command.rawValue
        validationRequiredOutputMetrics = plan.validationPerformanceProfile?.requiredOutputMetrics ?? []
        validationMaxBounds = plan.validationPerformanceProfile?.maxBounds.map {
            "\($0.metric):\($0.max):\($0.unit)"
        } ?? []
    }
}

private struct Qwen35TextLegacyParityProjection: Equatable {
    let semanticSHA256: String
    let exportABISHA256: String
    let modelLocator: String
    let tokenizerLocator: String
    let prefillSourcePath: String?
    let prefillCachePath: String?
    let prefillBatch: Int
    let tensorExpansion: Qwen35TextTensorExpansion
    let tensorPatterns: [String]
    let quantization: Qwen35TextQuantizationSignature
    let compile: Qwen35TextCompileSignature
    let gates: Qwen35TextGateSignature
    let bridgeDefaults: [String]
    let package: Qwen35TextPackageParitySignature
}

private enum Qwen35TextLegacyParityOracle {
    static func project(
        cam rawCAM: SmeltCAMIR,
        legacy: SmeltModelIR
    ) throws -> Qwen35TextLegacyParityProjection {
        let cam = try rawCAM.validated()
        try validateModule(cam)
        try validateLegacyAnchor(legacy)
        try validateSources(cam, legacy: legacy)
        try validateExport(cam)
        try validateBlockShape(cam, legacy: legacy)
        try validateGraph(cam)
        let flowPolicy = try validateFlow(cam, legacy: legacy)
        let tensorPatterns = try validateTensorPattern(cam)
        let quantization = try validateQuantization(cam, legacy: legacy)
        let compile = try validateCompile(cam, legacy: legacy)

        let legacyPlan = try SmeltPackageResolvedPlan.resolve(
            SmeltPackageSpecLowering.textGeneration(from: legacy)
        )
        let package = Qwen35TextPackageParitySignature(plan: legacyPlan)
        let gates = try validateGates(
            cam,
            expectedPackageFiles: (package.packageFiles + [
                SmeltCAMPackageDescriptor.packageFileName,
            ]).sorted()
        )

        try require(package.routes == [
            "tokenizer:native:none",
            "trunk:compiled:baked-inline",
            "text-head:native:none",
        ], "legacy package routes drifted")
        try require(package.setupPhases == ["prefill:trunk,text-head"], "legacy setup phase drifted")
        try require(package.perStepPhases == ["decode:trunk,text-head"], "legacy decode phase drifted")
        try require(package.emission == "per-step", "legacy emission drifted")
        try require(package.eosTokens == flowPolicy.eosTokens, "CAM EOS policy does not match package")
        try require(package.maxTokens == flowPolicy.maxSteps, "CAM max-steps does not match inference")
        try require(package.decodeMaxSteps == flowPolicy.maxSteps, "CAM max-steps does not match decode")
        try require(package.quantizationFormat == "u4", "legacy package quantization format drifted")
        try require(package.quantizationGroupSize == quantization.groupSize, "quant group does not match package")
        try require(
            package.validationPerformanceGate == SmeltPackagePerformanceGateID.textDecodePrefillStartup,
            "legacy performance gate drifted"
        )

        return Qwen35TextLegacyParityProjection(
            semanticSHA256: try cam.semanticSHA256(),
            exportABISHA256: try cam.exportABISHA256(),
            modelLocator: source(cam, "weights").locator,
            tokenizerLocator: source(cam, "tokenizer").locator,
            prefillSourcePath: nil,
            prefillCachePath: nil,
            prefillBatch: legacy.prefill!.maxBatchSize,
            tensorExpansion: .patternOnly,
            tensorPatterns: tensorPatterns,
            quantization: quantization,
            compile: compile,
            gates: gates,
            bridgeDefaults: bridgeDefaults(from: legacy),
            package: package
        )
    }

    private struct FlowPolicy {
        let eosTokens: [Int32]
        let maxSteps: Int
    }

    private static func validateModule(_ cam: SmeltCAMIR) throws {
        try require(cam.schemaVersion == SmeltCAMIR.currentSchemaVersion, "schema version drifted")
        try require(cam.module.id == "qwen35_text", "module id must be qwen35_text")
        try require(cam.imports.isEmpty, "qwen35_text parity does not allow imports")
        try require(cam.capabilities == ["bake.prompt-prefix", "run.generate"], "capabilities drifted")
        try require(cam.backendConstraints == [.init("target", "metal")], "backend constraints drifted")
        try require(cam.sourceQuantization.isEmpty, "source quantization is not part of this bridge")
        try require(cam.artifacts.isEmpty, "artifact roles are not part of this bridge")
    }

    private static func validateLegacyAnchor(_ legacy: SmeltModelIR) throws {
        try require(legacy.modelName == "Qwen/Qwen3.5-2B", "legacy anchor model drifted")
        try require(legacy.prefill != nil, "legacy prefill missing")
        try require(legacy.runtime.architecture == nil, "runtime architecture selector drifted in")
        try require(legacy.runtime.architectureSource == .defaultValue, "runtime architecture source drifted")
        try require(legacy.config.compiledSeqCapacity == 256, "static sequence capacity drifted")
        try require(legacy.config.attention?.gatedQ == true, "Qwen gated-q default drifted")
        try require(legacy.loading.strategy == .mmapPrefault, "loading strategy drifted")
        try require(legacy.loading.packing == .monolithic, "loading packing drifted")
        try require(legacy.decode.policy?.sampler.mode == .greedy, "decode sampler drifted")
        try require(legacy.inference.chatTemplate == "chatml", "chat template drifted")
        try require(legacy.inference.thinkingPolicy == .disabled, "thinking policy drifted")
    }

    private static func validateSources(_ cam: SmeltCAMIR, legacy: SmeltModelIR) throws {
        let signatures = cam.sources.map {
            "\($0.id):\($0.kind):\($0.locator):\($0.revision ?? "none")"
        }.sorted()
        try require(signatures == [
            "tokenizer:hf-file:\(legacy.modelName)/tokenizer.json:none",
            "weights:hf:\(legacy.modelName):main",
        ], "source set drifted")
    }

    private static func validateExport(_ cam: SmeltCAMIR) throws {
        try require(cam.exports == [
            .init(
                id: "generate",
                inputs: [.init(name: "prompt", type: .init("text", attributes: ["encoding": "utf8"]))],
                outputs: [.init(name: "text", type: .init("text", attributes: ["encoding": "utf8"]))],
                capabilities: ["bake.prompt-prefix", "run.generate"],
                gates: ["startup"]
            ),
        ], "export ABI drifted")
        try require(cam.exportBindings == [.init(export: "generate", flow: "generate")], "export binding drifted")
    }

    private static func validateBlockShape(_ cam: SmeltCAMIR, legacy: SmeltModelIR) throws {
        let block = try exactlyOne(cam.blocks, "block")
        try require(block.id == "trunk", "block id drifted")
        try require(block.operatorName == .transformer, "block operator drifted")
        let shape = try requireValue(block.shape.transformer, "trunk transformer shape missing")
        let delta = try requireValue(shape.delta, "delta shape missing")
        let attention = try requireValue(shape.attention, "attention shape missing")
        let ffn = try requireValue(shape.ffn, "ffn shape missing")
        let norm = try requireValue(shape.norm, "norm shape missing")
        let vocab = try requireValue(shape.vocab, "vocab shape missing")
        let legacyDelta = try requireValue(legacy.config.delta, "legacy delta shape missing")
        let legacyAttention = try requireValue(legacy.config.attention, "legacy attention shape missing")

        try require(shape.hiddenSize == legacy.config.hiddenSize, "hidden size drifted")
        try require(shape.layers?.roles == [.delta, .delta, .delta, .attention], "layer roles drifted")
        try require(shape.layers?.repeatCount == legacy.layerPattern.repeats, "layer repeat count drifted")
        try require((shape.layers?.roles.count ?? 0) * (shape.layers?.repeatCount ?? 0) == legacy.config.numLayers, "layer count drifted")
        try require(delta.heads == legacyDelta.numHeads, "delta heads drifted")
        try require(delta.headDim == legacyDelta.headDim, "delta head dim drifted")
        try require(delta.convKernel == legacyDelta.convKernel, "delta conv kernel drifted")
        try require(delta.projections == [
            "qkv": legacyDelta.qkvDim,
            "z": legacyDelta.zDim,
            "a": legacyDelta.aDim,
            "b": legacyDelta.bDim,
        ], "delta projections drifted")
        try require(attention.qHeads == legacyAttention.qHeads, "attention q heads drifted")
        try require(attention.kvHeads == legacyAttention.kvHeads, "attention kv heads drifted")
        try require(attention.headDim == legacyAttention.headDim, "attention head dim drifted")
        try require(attention.rope == .init(kind: .neox, theta: Int(legacyAttention.ropeTheta)), "attention rope drifted")
        try require(attention.qkNorm == .rms, "attention qk norm drifted")
        try require(ffn.dim == legacy.config.ffn.dim, "ffn dim drifted")
        try require(ffn.activation.rawValue == legacy.config.ffn.activation.rawValue, "ffn activation drifted")
        try require(norm.kind == .rms, "norm kind drifted")
        try require(norm.eps == "1e-6", "norm epsilon drifted")
        try require(norm.mode == .onePlusWeight, "norm mode drifted")
        try require(vocab.size == legacy.config.vocabSize, "vocab size drifted")
        try require(vocab.tiedHead == legacy.config.tiedLMHead, "tied head drifted")
    }

    private static func validateGraph(_ cam: SmeltCAMIR) throws {
        let nodes = Dictionary(uniqueKeysWithValues: cam.graphNodes.map { ($0.id, $0) })
        try require(Set(nodes.keys) == ["detokenizer", "sampler", "tokenizer", "trunk"], "graph node set drifted")
        try validateNode(
            nodes["tokenizer"],
            implementation: .native,
            block: nil,
            source: nil,
            inputs: ["prompt"],
            outputs: ["tokens"],
            annotations: [
                "assistant-prelude=preclosed-think",
                "prompt-format=chatml",
                "tag=text-tokenizer",
                "thinking-policy=disabled",
                "tool-format=xml-function-parameters",
            ]
        )
        try validateNode(
            nodes["trunk"],
            implementation: .compiled,
            block: "trunk",
            source: nil,
            inputs: ["tokens"],
            outputs: ["hidden"],
            annotations: [
                "artifact=baked-inline",
                "feedback=tokens",
                "state=kv-cache,conv-state,rec-state",
            ]
        )
        try validateNode(
            nodes["sampler"],
            implementation: .native,
            block: nil,
            source: nil,
            inputs: ["hidden"],
            outputs: ["tokens"],
            annotations: ["state=sampler", "tag=sampler"]
        )
        try validateNode(
            nodes["detokenizer"],
            implementation: .native,
            block: nil,
            source: nil,
            inputs: ["tokens"],
            outputs: ["text"],
            annotations: ["tag=text-detokenizer"]
        )
        try require(Set(cam.graphEdges.map(edgeSignature)) == Set([
            "value:hidden->node:sampler.hidden:hidden",
            "value:tokens->node:trunk.tokens:tokens",
            "value:tokens_2->node:detokenizer.tokens:tokens",
            "input:prompt->node:tokenizer.prompt:text",
            "node:detokenizer.text->output:text:text",
            "node:sampler.tokens->value:tokens_2:tokens",
            "node:tokenizer.tokens->value:tokens:tokens",
            "node:trunk.hidden->value:hidden:hidden",
        ]), "graph edges drifted")
        try require(Set(cam.feedbackEdges.map(feedbackSignature)) == Set([
            "node:sampler.tokens->node:trunk.tokens",
        ]), "feedback edges drifted")
    }

    private static func validateFlow(
        _ cam: SmeltCAMIR,
        legacy: SmeltModelIR
    ) throws -> FlowPolicy {
        let flow = try exactlyOne(cam.flows, "flow")
        try require(flow.id == "generate", "flow id drifted")
        try require(flow.phases == [
            .init(role: .setup, calls: [.node("tokenizer")]),
            .init(role: .step, label: "decode", calls: [.node("trunk"), .node("sampler")]),
        ], "flow phases drifted")
        try require(flow.emit == [.node("detokenizer", "text")], "flow emit drifted")

        let eosTokens = flow.stop.compactMap { stop -> Int32? in
            stop.kind == .eosToken ? stop.value.map(Int32.init) : nil
        }.sorted()
        let maxSteps = try requireValue(
            flow.stop.first { $0.kind == .maxSteps }?.value,
            "flow max-steps missing"
        )
        try require(flow.stop.contains(.init(kind: .hostCancel)), "host cancel stop missing")
        try require(eosTokens == legacy.inference.eosTokens.sorted(), "flow EOS tokens drifted")
        try require(maxSteps == legacy.inference.maxTokens, "flow max-steps drifted")
        return FlowPolicy(eosTokens: eosTokens, maxSteps: maxSteps)
    }

    private static func validateTensorPattern(_ cam: SmeltCAMIR) throws -> [String] {
        let tensor = try exactlyOne(cam.tensors, "tensor map")
        try require(tensor.source == "weights", "tensor source drifted")
        try require(tensor.selector.source == "weights", "tensor selector source drifted")
        try require(tensor.selector.pattern == "*", "tensor selector pattern drifted")
        try require(tensor.owner == "trunk", "tensor owner drifted")
        try require(tensor.target.block == "trunk", "tensor target block drifted")
        try require(tensor.target.selector == "*", "tensor target selector drifted")
        return ["weights.* -> trunk.*"]
    }

    private static func validateQuantization(
        _ cam: SmeltCAMIR,
        legacy: SmeltModelIR
    ) throws -> Qwen35TextQuantizationSignature {
        try require(legacy.quantization.strategy == .affineU4, "legacy quantization strategy drifted")
        try require(legacy.quantization.groupSize == 64, "legacy quantization group drifted")
        try require(legacy.quantization.quantizeEmbedding, "legacy embedding quantization drifted")

        let defaultRule = try exactlyOne(
            cam.quantization.filter { $0.action == .default },
            "default quant rule"
        )
        try require(defaultRule.selector == .init("*", source: "weights"), "default quant selector drifted")
        try require(defaultRule.storage == .init(format: .affineU4, groupSize: 64), "default quant storage drifted")
        try require(defaultRule.resolution == .declaredTensor, "default quant resolution drifted")
        try require(defaultRule.source == nil, "default quant source drifted")

        let embeddingRule = try exactlyOne(
            cam.quantization.filter { $0.action == .quantize },
            "embedding quant rule"
        )
        try require(embeddingRule.selector == .init("embeddings", source: "weights"), "embedding quant selector drifted")
        try require(embeddingRule.source == "weights", "embedding quant source drifted")
        try require(embeddingRule.storage == nil, "embedding quant storage drifted")
        try require(embeddingRule.priority == 10, "embedding quant priority drifted")
        try require(embeddingRule.resolution == .sourceDeferred, "embedding quant resolution drifted")

        let preserves = cam.quantization.filter { $0.action == .preserve }
        let preservedPatterns = preserves.map(\.selector.pattern).sorted()
        try require(preservedPatterns == legacy.quantization.excludePatterns.sorted(), "preserve quant patterns drifted")
        try require(preserves.allSatisfy {
            $0.selector.source == "weights"
                && $0.source == "weights"
                && $0.storage == nil
                && $0.resolution == .sourceDeferred
        }, "preserve quant rule shape drifted")
        try require(cam.quantization.count == 1 + 1 + preserves.count, "unsupported extra quant rules")

        return Qwen35TextQuantizationSignature(
            storageFormat: defaultRule.storage!.format.rawValue,
            groupSize: defaultRule.storage!.groupSize!,
            quantizeEmbeddings: true,
            preservedPatterns: preservedPatterns
        )
    }

    private static func validateCompile(
        _ cam: SmeltCAMIR,
        legacy: SmeltModelIR
    ) throws -> Qwen35TextCompileSignature {
        let target = try uniqueConstraintValue(cam.compile, key: "target")
        let prefill = try uniqueConstraintValue(cam.compile, key: "prefill")
        let layout = try uniqueConstraintValue(cam.compile, key: "layout")
        try require(target == "metal", "compile target drifted")
        try require(!cam.compile.contains { $0.key == "package-spec" }, "package-spec selector returned")
        try require(prefill == "metal batch \(legacy.prefill!.maxBatchSize)", "compile prefill drifted")
        try require(layout == "memory-neutral", "compile layout drifted")
        return Qwen35TextCompileSignature(
            target: target,
            prefill: prefill,
            layout: layout
        )
    }

    private static func validateGates(
        _ cam: SmeltCAMIR,
        expectedPackageFiles: [String]
    ) throws -> Qwen35TextGateSignature {
        let startup = try requireValue(cam.gates.first { $0.id == "startup" }, "startup gate missing")
        try require(startup.from == .init(kind: .flowAccepted, flow: "generate"), "startup gate from drifted")
        try require(startup.to == .init(
            kind: .emit,
            flow: "generate",
            endpoint: .moduleOutput("text"),
            predicates: [.init(subject: "tokens", relation: .greaterThanOrEqual, value: "1")]
        ), "startup gate to drifted")
        try require(startup.requirements == [
            .init(subject: "elapsed", relation: .lessThanOrEqual, value: "115", unit: "ms"),
        ], "startup requirement drifted")

        let inventory = try requireValue(cam.gates.first { $0.id == "inventory" }, "inventory gate missing")
        let requirement = try exactlyOne(
            inventory.requirements.filter { $0.subject == "package-files" },
            "inventory package-files requirement"
        )
        try require(requirement.subject == "package-files", "inventory subject drifted")
        try require(requirement.relation == .include, "inventory relation drifted")
        try require(requirement.unit == nil, "inventory unit drifted")
        let inventoryFiles = requirement.value.split(separator: ",").map(String.init).sorted()
        try require(inventoryFiles == expectedPackageFiles, "inventory files do not match package files")

        let prefill = try requireValue(cam.gates.first { $0.id == "prefill" }, "prefill gate missing")
        try require(prefill.requirements == [
            .init(subject: "prefill-batch", relation: .lessThanOrEqual, value: "256"),
        ], "prefill requirement drifted")

        let decode = try requireValue(cam.gates.first { $0.id == "decode" }, "decode gate missing")
        try require(decode.requirements == [
            .init(subject: "decode-output.tokens", relation: .greaterThanOrEqual, value: "1"),
        ], "decode requirement drifted")

        try require(Set(cam.gates.map(\.id)) == [
            "startup",
            "prefill",
            "decode",
            "inventory",
        ], "unsupported gates")
        return Qwen35TextGateSignature(
            startupElapsedMaxMS: 100,
            inventoryFiles: inventoryFiles
        )
    }

    private static func bridgeDefaults(from legacy: SmeltModelIR) -> [String] {
        [
            "attention.gated_q:\(legacy.config.attention?.gatedQ == true)",
            "chat_template:\(legacy.inference.chatTemplate ?? "none")",
            "loading:\(legacy.loading.strategy.rawValue)/\(legacy.loading.packing.rawValue)",
            "sampler:\(legacy.decode.policy?.sampler.mode.rawValue ?? "none")",
            "static_seq_capacity:\(legacy.config.compiledSeqCapacity)",
            "thinking_policy:\(legacy.inference.thinkingPolicy?.rawValue ?? "none")",
        ]
    }

    private static func validateNode(
        _ node: SmeltCAMIR.GraphNode?,
        implementation: SmeltCAMIR.GraphImplementation,
        block: String?,
        source: String?,
        inputs: [String],
        outputs: [String],
        annotations: [String]
    ) throws {
        let node = try requireValue(node, "graph node missing")
        try require(node.implementation == implementation, "node \(node.id) implementation drifted")
        try require(node.block == block, "node \(node.id) block drifted")
        try require(node.source == source, "node \(node.id) source drifted")
        try require(node.imported == nil, "node \(node.id) unexpectedly imports")
        try require(node.inputs.map(\.name).sorted() == inputs.sorted(), "node \(node.id) inputs drifted")
        try require(node.outputs.map(\.name).sorted() == outputs.sorted(), "node \(node.id) outputs drifted")
        try require(node.annotations.map { "\($0.key)=\($0.value)" }.sorted() == annotations.sorted(), "node \(node.id) annotations drifted")
    }

    private static func source(_ cam: SmeltCAMIR, _ id: String) -> SmeltCAMIR.Source {
        cam.sources.first { $0.id == id }!
    }

    private static func edgeSignature(_ edge: SmeltCAMIR.GraphEdge) -> String {
        "\(endpointSignature(edge.from))->\(endpointSignature(edge.to)):\(edge.type?.name ?? "none")"
    }

    private static func feedbackSignature(_ edge: SmeltCAMIR.FeedbackEdge) -> String {
        "\(endpointSignature(edge.from))->\(endpointSignature(edge.to))"
    }

    private static func endpointSignature(_ endpoint: SmeltCAMIR.EndpointRef) -> String {
        switch endpoint.kind {
        case .moduleInput: return "input:\(endpoint.name ?? "")"
        case .moduleOutput: return "output:\(endpoint.name ?? "")"
        case .graphValue: return "value:\(endpoint.name ?? "")"
        case .nodePort: return "node:\(endpoint.node ?? "").\(endpoint.port ?? "")"
        case .importedPort:
            return "import:\(endpoint.importAlias ?? "").\(endpoint.export ?? "").\(endpoint.port ?? "")"
        }
    }

    private static func uniqueConstraintValue(
        _ constraints: [SmeltCAMIR.Constraint],
        key: String
    ) throws -> String {
        try exactlyOne(constraints.filter { $0.key == key }, "\(key) constraint").value
    }

    private static func exactlyOne<T>(_ values: [T], _ label: String) throws -> T {
        guard values.count == 1, let value = values.first else {
            throw Qwen35TextLegacyParityError.mismatch("\(label) expected once, found \(values.count)")
        }
        return value
    }

    private static func requireValue<T>(_ value: T?, _ label: String) throws -> T {
        guard let value else {
            throw Qwen35TextLegacyParityError.mismatch(label)
        }
        return value
    }

    private static func require(_ condition: Bool, _ label: String) throws {
        guard condition else {
            throw Qwen35TextLegacyParityError.mismatch(label)
        }
    }
}

private func copyCAMIR(
    _ ir: SmeltCAMIR,
    module: SmeltCAMIR.Module? = nil,
    exports: [SmeltCAMIR.Export]? = nil,
    exportBindings: [SmeltCAMIR.ExportBinding]? = nil,
    sources: [SmeltCAMIR.Source]? = nil,
    blocks: [SmeltCAMIR.Block]? = nil,
    graphNodes: [SmeltCAMIR.GraphNode]? = nil,
    graphEdges: [SmeltCAMIR.GraphEdge]? = nil,
    feedbackEdges: [SmeltCAMIR.FeedbackEdge]? = nil,
    flows: [SmeltCAMIR.Flow]? = nil,
    capabilities: [String]? = nil,
    backendConstraints: [SmeltCAMIR.Constraint]? = nil,
    tensors: [SmeltCAMIR.TensorMap]? = nil,
    quantization: [SmeltCAMIR.QuantRule]? = nil,
    sourceQuantization: [SmeltCAMIR.SourceQuantizationRule]? = nil,
    compile: [SmeltCAMIR.Constraint]? = nil,
    artifacts: [SmeltCAMIR.ArtifactRole]? = nil,
    gates: [SmeltCAMIR.Gate]? = nil
) -> SmeltCAMIR {
    SmeltCAMIR(
        schemaVersion: ir.schemaVersion,
        module: module ?? ir.module,
        imports: ir.imports,
        exports: exports ?? ir.exports,
        exportBindings: exportBindings ?? ir.exportBindings,
        sources: sources ?? ir.sources,
        blocks: blocks ?? ir.blocks,
        graphNodes: graphNodes ?? ir.graphNodes,
        graphEdges: graphEdges ?? ir.graphEdges,
        feedbackEdges: feedbackEdges ?? ir.feedbackEdges,
        flows: flows ?? ir.flows,
        capabilities: capabilities ?? ir.capabilities,
        backendConstraints: backendConstraints ?? ir.backendConstraints,
        tensors: tensors ?? ir.tensors,
        quantization: quantization ?? ir.quantization,
        sourceQuantization: sourceQuantization ?? ir.sourceQuantization,
        compile: compile ?? ir.compile,
        artifacts: artifacts ?? ir.artifacts,
        gates: gates ?? ir.gates
    )
}

private func replacingMaxSteps(
    in flows: [SmeltCAMIR.Flow],
    with value: Int
) -> [SmeltCAMIR.Flow] {
    flows.map { flow in
        guard flow.id == "generate" else { return flow }
        return SmeltCAMIR.Flow(
            id: flow.id,
            phases: flow.phases,
            emit: flow.emit,
            stop: flow.stop.map {
                $0.kind == .maxSteps ? .init(kind: .maxSteps, value: value) : $0
            }
        )
    }
}

private func replacingDefaultQuantGroup(
    in rules: [SmeltCAMIR.QuantRule],
    with value: Int
) -> [SmeltCAMIR.QuantRule] {
    rules.map { rule in
        guard rule.action == .default, let storage = rule.storage else { return rule }
        return SmeltCAMIR.QuantRule(
            selector: rule.selector,
            action: rule.action,
            storage: .init(
                format: storage.format,
                groupSize: value,
                computeDType: storage.computeDType
            ),
            source: rule.source,
            priority: rule.priority,
            calibration: rule.calibration,
            resolution: rule.resolution
        )
    }
}

private func removingInventoryFile(
    _ file: String,
    from gates: [SmeltCAMIR.Gate]
) -> [SmeltCAMIR.Gate] {
    gates.map { gate in
        guard gate.id == "inventory" else { return gate }
        return SmeltCAMIR.Gate(
            id: gate.id,
            from: gate.from,
            to: gate.to,
            requirements: gate.requirements.map { requirement in
                guard requirement.subject == "package-files" else { return requirement }
                let files = requirement.value
                    .split(separator: ",")
                    .map(String.init)
                    .filter { $0 != file }
                    .joined(separator: ",")
                return SmeltCAMIR.Comparison(
                    subject: requirement.subject,
                    relation: requirement.relation,
                    value: files,
                    unit: requirement.unit
                )
            },
            evidence: gate.evidence
        )
    }
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        "smelt-cam-lowering-\(UUID().uuidString)",
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
    return url
}

private func qwen3TTSRunnableSpecs(
    textEmbeddingDType: Qwen3TTSPackageBuilder.WeightDType
) -> [Qwen3TTSPackageBuilder.WeightSpec] {
    let hidden = 64
    let headDim = 32
    let qDim = 64
    let kvDim = 32
    let inter = 128
    let vocab = 128

    var specs: [Qwen3TTSPackageBuilder.WeightSpec] = [
        .init(name: "talker.model.text_embedding.weight", shape: [vocab, hidden], dtype: textEmbeddingDType),
        .init(name: "talker.text_projection.linear_fc1.weight", shape: [hidden, hidden]),
        .init(name: "talker.text_projection.linear_fc1.bias", shape: [hidden]),
        .init(name: "talker.text_projection.linear_fc2.weight", shape: [hidden, hidden]),
        .init(name: "talker.text_projection.linear_fc2.bias", shape: [hidden]),
        .init(name: "talker.model.codec_embedding.weight", shape: [vocab, hidden]),
        .init(name: "talker.codec_head.weight", shape: [vocab, hidden], dtype: .bf16),
        .init(name: "talker.code_predictor.lm_head.0.weight", shape: [vocab, hidden], dtype: .bf16),
        .init(name: "talker.code_predictor.model.codec_embedding.0.weight", shape: [vocab, hidden], dtype: .bf16),
        .init(name: "talker.code_predictor.small_to_mtp_projection.weight", shape: [hidden, hidden], dtype: .bf16),
        .init(name: "talker.code_predictor.small_to_mtp_projection.bias", shape: [hidden]),
        .init(name: "decoder.pre_conv.conv.weight", shape: [2, 2]),
    ]

    appendQwen3TTSTrunkLayerSpecs(
        to: &specs,
        prefix: "talker.model.",
        hidden: hidden,
        headDim: headDim,
        qDim: qDim,
        kvDim: kvDim,
        inter: inter
    )
    appendQwen3TTSTrunkLayerSpecs(
        to: &specs,
        prefix: "talker.code_predictor.model.",
        hidden: hidden,
        headDim: headDim,
        qDim: qDim,
        kvDim: kvDim,
        inter: inter
    )
    return specs
}

private func appendQwen3TTSTrunkLayerSpecs(
    to specs: inout [Qwen3TTSPackageBuilder.WeightSpec],
    prefix: String,
    hidden: Int,
    headDim: Int,
    qDim: Int,
    kvDim: Int,
    inter: Int
) {
    let layer = "\(prefix)layers.0"
    specs.append(contentsOf: [
        .init(name: "\(layer).self_attn.q_proj.weight", shape: [qDim, hidden], dtype: .bf16),
        .init(name: "\(layer).self_attn.k_proj.weight", shape: [kvDim, hidden], dtype: .bf16),
        .init(name: "\(layer).self_attn.v_proj.weight", shape: [kvDim, hidden], dtype: .bf16),
        .init(name: "\(layer).self_attn.o_proj.weight", shape: [hidden, qDim], dtype: .bf16),
        .init(name: "\(layer).mlp.gate_proj.weight", shape: [inter, hidden], dtype: .bf16),
        .init(name: "\(layer).mlp.up_proj.weight", shape: [inter, hidden], dtype: .bf16),
        .init(name: "\(layer).mlp.down_proj.weight", shape: [hidden, inter], dtype: .bf16),
        .init(name: "\(layer).input_layernorm.weight", shape: [hidden]),
        .init(name: "\(layer).post_attention_layernorm.weight", shape: [hidden]),
        .init(name: "\(layer).self_attn.q_norm.weight", shape: [headDim]),
        .init(name: "\(layer).self_attn.k_norm.weight", shape: [headDim]),
        .init(name: "\(prefix)norm.weight", shape: [hidden]),
    ])
}
