import Foundation
import Testing
@testable import SmeltSchema

@Suite struct SmeltPackageSpecTests {

    private func routes(for graph: SmeltBlockGraph) -> [SmeltPackageSpec.RuntimeDescriptor.BlockRoute] {
        graph.runtimeRoutes
    }

    private var allCommands: [SmeltPackageSpec.RuntimeDescriptor.Command] {
        SmeltPackageSpec.RuntimeDescriptor.Command.allCases
    }

    private var source: SmeltPackageSpec.Source {
        .init(id: "weights", kind: .huggingFace, repo: "org/model", revision: "main")
    }

    private func tensor(block: String) -> SmeltPackageSpec.TensorMap {
        .init(
            source: "weights",
            name: "\(block).weight",
            canonicalName: "\(block).weight",
            block: block,
            sourceDType: .f16,
            storedDType: .f16,
            shape: [2, 2]
        )
    }

    private func outputFiles(_ files: [String]) -> SmeltPackageSpec.PackageFileSet {
        .init(files: ["manifest.json"] + files)
    }

    private func sidecarSignatures(_ sidecars: [SmeltPackageSpec.Sidecar]) -> [String] {
        sidecars.map { "\($0.id)|\($0.path)|\($0.kind)" }
    }

    private func llmSpec(
        modelName: String = "qwen",
        runtime: SmeltPackageSpec.RuntimeDescriptor? = nil,
        sources: [SmeltPackageSpec.Source]? = nil,
        artifacts: [SmeltPackageSpec.Artifact]? = nil,
        inference: SmeltInferenceManifest? = SmeltInferenceManifest(
            maxTokens: 32,
            eosTokens: [1],
            chatTemplate: "chatml",
            thinkingPolicy: .disabled
        ),
        quantization: SmeltPackageSpec.QuantizationPlan? = nil,
        sidecars: [SmeltPackageSpec.Sidecar] = [],
        packageFiles: SmeltPackageSpec.PackageFileSet? = nil,
        validation: SmeltPackageSpec.Validation = .init()
    ) -> SmeltPackageSpec {
        let graph = SmeltBlockGraph.tokenFeedbackText
        return SmeltPackageSpec(
            packageName: "qwen.cam",
            modelName: modelName,
            sources: sources ?? [source],
            blocks: graph,
            loop: .tokenFeedbackText,
            runtime: runtime ?? .forGraph(
                architecture: "text-generation",
                commands: allCommands,
                graph: graph
            ),
            architectureConfig: .object(["hidden_size": .int(2)]),
            tensors: [tensor(block: "trunk")],
            quantization: quantization,
            sidecars: sidecars,
            artifacts: artifacts ?? [.init(id: "weights", path: "weights.bin", role: "weights")],
            outputFiles: packageFiles ?? outputFiles(
                ["weights.bin", "tokenizer.json"] + sidecars.map(\.path)
            ),
            tokenizer: .init(format: "tokenizer-json", files: ["tokenizer.json"]),
            inference: inference,
            decode: .init(sampler: .init(mode: .greedy), maxSteps: 32),
            validation: validation
        )
    }

    private func qwenTTSSpec(
        quantization: SmeltPackageSpec.QuantizationPlan? = .init(format: .u4, groupSize: 32),
        sidecars: [SmeltPackageSpec.Sidecar] = SmeltQwen3TTSPackageProfiles.runnable.sidecars
    ) -> SmeltPackageSpec {
        let graph = SmeltBlockGraph.qwen3TTSCompiledTrunkNativeFrontEnd
        let profile = SmeltQwen3TTSPackageProfiles.runnable
        return SmeltPackageSpec(
            packageName: profile.packageName,
            modelName: profile.modelName,
            sources: [source],
            blocks: graph,
            loop: profile.loop,
            runtime: .forGraph(
                architecture: profile.runtimeArchitecture,
                commands: allCommands,
                graph: graph
            ),
            architectureConfig: .object(["page_size": .int(profile.pageSize)]),
            tensors: [tensor(block: "talker")],
            quantization: quantization,
            sidecars: sidecars,
            artifacts: [.init(id: "weights", path: "weights.bin", role: "weights")],
            outputFiles: outputFiles(["weights.bin", "tokenizer.json"]
                + sidecars.map(\.path)),
            tokenizer: .init(format: "byte-bpe", files: ["tokenizer.json"]),
            inference: .init(maxTokens: profile.maxTokens, eosTokens: profile.eosTokens),
            decode: .init(
                sampler: .init(mode: .sample, temperature: 0.8, topK: 20),
                maxSteps: profile.maxTokens
            )
        )
    }

    @Test func validFamilySpecsValidate() throws {
        try llmSpec().validate()
        try qwenTTSSpec().validate()
    }

    @Test func rejectsUnsafeLocalSourcePath() {
        let unsafe = SmeltPackageSpec.Source(
            id: "weights",
            kind: .localDirectory,
            path: "../escape"
        )
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(sources: [unsafe]).validate()
        }
    }

    @Test func runtimeArchitectureStampsArePublicSchemaData() {
        #expect(SmeltQwen3TTSPackageProfiles.runnable.runtimeArchitecture
            == SmeltRuntimeGraphPolicy.sidecarTextToCodecAudio.rawValue)
    }

    @Test func runtimeDescriptorRoutesAreGraphAuthored() {
        #expect(routes(for: .tokenFeedbackText).map(\.signature) == [
            "tokenizer:native:none",
            "trunk:compiled:baked-inline",
            "text-head:native:none",
        ])
        #expect(routes(for: .qwen3TTSCompiledTrunkNativeFrontEnd).map(\.signature) == [
            "tts-frontend:native:none",
            "talker:compiled:baked-sidecar",
            "codec-head:native:none",
            "mtp-head:native:internal-sidecar",
            "codec-decoder:compiled:runtime-emit",
        ])
    }

    @Test func loopScheduleSignaturesArePublicSchemaData() {
        #expect(SmeltLoopSchedule.tokenFeedbackText.setupSignatures == [
            "prefill:trunk,text-head",
        ])
        #expect(SmeltLoopSchedule.tokenFeedbackText.perStepSignatures == [
            "decode:trunk,text-head",
        ])
        #expect(SmeltLoopSchedule.tokenFeedbackText.emissionSignature == "per-step")
        #expect(SmeltLoopSchedule.qwen3TTS.setupSignatures == [
            "prefill:talker,codec-head",
        ])
        #expect(SmeltLoopSchedule.qwen3TTS.perStepSignatures == [
            "mtp:mtp-head",
            "advance:talker,codec-head:feeds-next-step",
        ])
        #expect(
            SmeltLoopSchedule.qwen3TTS.emissionSignature == "chunked:1:1:double:codec-decoder"
        )
    }

    @Test func rejectsMissingRuntimeRoute() {
        let graph = SmeltBlockGraph.tokenFeedbackText
        let badRuntime = SmeltPackageSpec.RuntimeDescriptor(
            architecture: "text-generation",
            commands: allCommands,
            routes: Array(routes(for: graph).dropLast())
        )
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(runtime: badRuntime).validate()
        }
    }

    @Test func rejectsDuplicateRuntimeRoute() {
        let graph = SmeltBlockGraph.tokenFeedbackText
        let graphRoutes = routes(for: graph)
        let badRuntime = SmeltPackageSpec.RuntimeDescriptor(
            architecture: "text-generation",
            commands: allCommands,
            routes: graphRoutes + [graphRoutes[0]]
        )
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(runtime: badRuntime).validate()
        }
    }

    @Test func rejectsRouteThatDisagreesWithGraph() {
        let graph = SmeltBlockGraph.tokenFeedbackText
        let badRuntime = SmeltPackageSpec.RuntimeDescriptor(
            architecture: "text-generation",
            commands: allCommands,
            routes: graph.blocks.map {
                if $0.name == "trunk" {
                    return .init(block: $0.name, impl: $0.impl, delivery: nil)
                }
                return .init(block: $0.name, impl: $0.impl, delivery: $0.compiledDelivery)
            }
        )
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(runtime: badRuntime).validate()
        }
    }

    @Test func rejectsRuntimeRouteForUnknownBlock() {
        let graph = SmeltBlockGraph.tokenFeedbackText
        let badRuntime = SmeltPackageSpec.RuntimeDescriptor(
            architecture: "text-generation",
            commands: allCommands,
            routes: routes(for: graph) + [
                .init(block: "ghost", impl: .native, delivery: nil)
            ]
        )
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(runtime: badRuntime).validate()
        }
    }

    @Test func rejectsMissingResolvedLLMPolicy() {
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(inference: nil).validate()
        }
        let noTemplate = SmeltInferenceManifest(maxTokens: 32, eosTokens: [1])
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(inference: noTemplate).validate()
        }
    }

    @Test func rejectsUnsupportedLLMChatTemplateBeforeRuntime() {
        let unknownTemplate = SmeltInferenceManifest(
            maxTokens: 32,
            eosTokens: [1],
            chatTemplate: "future-template",
            thinkingPolicy: .disabled
        )
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(inference: unknownTemplate).validate()
        }
    }

    @Test func rejectsMissingLLMThinkingPolicyBeforeRuntime() {
        let noThinkingPolicy = SmeltInferenceManifest(
            maxTokens: 32,
            eosTokens: [1],
            chatTemplate: "chatml"
        )
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(inference: noThinkingPolicy).validate()
        }
    }

    @Test func rejectsPackageFilePrefixConflicts() {
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(packageFiles: outputFiles([
                "weights.bin",
                "tokenizer.json",
                "manifest.json/child",
            ])).validate()
        }
    }

    @Test func allowsCaptureArtifactsInsideSidecarDirectories() throws {
        let calibration = SmeltPackageSpec.QuantizationPlan.Calibration(
            corpus: .init(
                source: "weights",
                path: "calibration/prompts.txt",
                renderPolicy: "prompt-lines"
            ),
            captureArtifacts: [
                .init(
                    id: "talker-captures",
                    path: "trunk/gptq_capture_points.json",
                    role: "activation-capture"
                ),
            ]
        )
        try qwenTTSSpec(
            quantization: .init(format: .u4, groupSize: 32, calibration: calibration)
        ).validate()
    }

    @Test func validatesKnownPerformanceGateProfiles() throws {
        try llmSpec(validation: SmeltPackagePerformanceProfiles.validation(
            parityFixture: "qwen",
            performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
        )).validate()
    }

    @Test func validatesStructuredPerformanceProfile() throws {
        let profile = SmeltPackagePerformanceProfiles.profile(
            for: SmeltPackagePerformanceGateID.textDecodePrefillStartup
        )
        #expect(profile.requiredTraceLabels == SmeltPackagePerformanceTraceLabel
            .textDecodePrefillStartupRequired)
        #expect(profile.requiredOutputMetrics == SmeltPackagePerformanceMetricName
            .textDecodePrefillStartupRequired)
        #expect(profile.maxBounds == [
            .init(
                metric: SmeltPackagePerformanceMetricName.traceFirstTokenMS,
                max: SmeltPackagePerformanceBudget.textTraceFirstTokenMaxMS,
                unit: SmeltPackagePerformanceUnit.milliseconds
            ),
        ])
        try llmSpec(validation: .init(
            parityFixture: "qwen",
            performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
            performanceProfile: profile
        )).validate()
    }

    @Test func performanceBudgetsPinUserFacingGates() {
        #expect(SmeltPackagePerformanceBudget.textTraceFirstTokenMaxMS == 100)
        #expect(SmeltPackagePerformanceBudget.qwen35TextTraceFirstTokenMaxMS == 115)
        #expect(SmeltPackagePerformanceBudget.textTraceFirstTokenMaxMS(
            forModelName: "Qwen/Qwen3.5-0.8B"
        ) == 100)
        #expect(SmeltPackagePerformanceBudget.textTraceFirstTokenMaxMS(
            forModelName: "Qwen/Qwen3.5-2B"
        ) == 115)
        #expect(SmeltPackagePerformanceBudget.qwen3TTSFirstAudioMaxSeconds == 0.50)
        #expect(SmeltPackagePerformanceBudget.qwen3TTSTTFAMaxSeconds == 0.50)
        #expect(SmeltPackagePerformanceBudget.qwen3TTSRealTimeFactorMax == 1.20)
        #expect(SmeltPackagePerformanceUnit.realTimeFactor == "x")
        #expect(SmeltPackagePerformanceBudget.qwen35FastDecodeMinTPS == 280)
        #expect(SmeltPackagePerformanceBudget.qwen35FastPrefill64MinTPS == 2280)
        #expect(SmeltPackagePerformanceBudget.qwen35FastPrefill256MinTPS == 2600)
        #expect(SmeltPackagePerformanceBudget.qwen35TextDecodeMinTPS == 170)
        #expect(SmeltPackagePerformanceBudget.qwen35TextPrefill64MinTPS == 820)
        #expect(SmeltPackagePerformanceBudget.qwen35TextPrefill256MinTPS == 850)
        #expect(SmeltPackagePerformanceBudget.textDecodeMinTPS(forModelName: "Qwen/Qwen3.5-0.8B") == 280)
        #expect(SmeltPackagePerformanceBudget.textPrefill64MinTPS(forModelName: "Qwen/Qwen3.5-0.8B") == 2280)
        #expect(SmeltPackagePerformanceBudget.textPrefill256MinTPS(forModelName: "Qwen/Qwen3.5-0.8B") == 2600)
        #expect(SmeltPackagePerformanceBudget.textDecodeMinTPS(forModelName: "Qwen/Qwen3.5-2B") == 170)
        #expect(SmeltPackagePerformanceBudget.textPrefill64MinTPS(forModelName: "Qwen/Qwen3.5-2B") == 820)
        #expect(SmeltPackagePerformanceBudget.textPrefill256MinTPS(forModelName: "Qwen/Qwen3.5-2B") == 850)
        #expect(SmeltPackagePerformanceBudget.textDecodeMinTPS(forModelName: "qwen") == nil)
        #expect(SmeltPackagePerformanceUnit.milliseconds == "ms")
        #expect(SmeltPackagePerformanceUnit.seconds == "s")
        #expect(SmeltPackagePerformanceUnit.tokensPerSecond == "tok/s")
    }

    @Test func pinsCanonicalTTSPerformanceProfile() {
        let profile = SmeltPackagePerformanceProfiles.profile(
            for: SmeltPackagePerformanceGateID.qwen3TTSTTFA
        )
        #expect(profile.command == .bench)
        #expect(profile.requiredOutputMetrics == [
            SmeltPackagePerformanceMetricName.firstAudioSeconds,
            SmeltPackagePerformanceMetricName.ttfaSeconds,
            SmeltPackagePerformanceMetricName.realTimeFactor,
        ])
        #expect(profile.minBounds.isEmpty)
        #expect(profile.maxBounds == [
            .init(
                metric: SmeltPackagePerformanceMetricName.firstAudioSeconds,
                max: SmeltPackagePerformanceBudget.qwen3TTSFirstAudioMaxSeconds,
                unit: SmeltPackagePerformanceUnit.seconds
            ),
            .init(
                metric: SmeltPackagePerformanceMetricName.ttfaSeconds,
                max: SmeltPackagePerformanceBudget.qwen3TTSTTFAMaxSeconds,
                unit: SmeltPackagePerformanceUnit.seconds
            ),
            .init(
                metric: SmeltPackagePerformanceMetricName.realTimeFactor,
                max: SmeltPackagePerformanceBudget.qwen3TTSRealTimeFactorMax,
                unit: SmeltPackagePerformanceUnit.realTimeFactor
            ),
        ])
    }

    @Test func pinsTextModelSpecificFloorProfiles() {
        let fast = SmeltPackagePerformanceProfiles.profile(
            for: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
            modelName: "Qwen/Qwen3.5-0.8B"
        )
        #expect(fast.minBounds == [
            .init(
                metric: SmeltPackagePerformanceMetricName.decodeTokensPerSecond,
                min: SmeltPackagePerformanceBudget.qwen35FastDecodeMinTPS,
                unit: SmeltPackagePerformanceUnit.tokensPerSecond
            ),
            .init(
                metric: SmeltPackagePerformanceMetricName.prefill64TokensPerSecond,
                min: SmeltPackagePerformanceBudget.qwen35FastPrefill64MinTPS,
                unit: SmeltPackagePerformanceUnit.tokensPerSecond
            ),
            .init(
                metric: SmeltPackagePerformanceMetricName.prefill256TokensPerSecond,
                min: SmeltPackagePerformanceBudget.qwen35FastPrefill256MinTPS,
                unit: SmeltPackagePerformanceUnit.tokensPerSecond
            ),
        ])
        let text = SmeltPackagePerformanceProfiles.profile(
            for: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
            modelName: "Qwen/Qwen3.5-2B"
        )
        #expect(text.minBounds == [
            .init(
                metric: SmeltPackagePerformanceMetricName.decodeTokensPerSecond,
                min: SmeltPackagePerformanceBudget.qwen35TextDecodeMinTPS,
                unit: SmeltPackagePerformanceUnit.tokensPerSecond
            ),
            .init(
                metric: SmeltPackagePerformanceMetricName.prefill64TokensPerSecond,
                min: SmeltPackagePerformanceBudget.qwen35TextPrefill64MinTPS,
                unit: SmeltPackagePerformanceUnit.tokensPerSecond
            ),
            .init(
                metric: SmeltPackagePerformanceMetricName.prefill256TokensPerSecond,
                min: SmeltPackagePerformanceBudget.qwen35TextPrefill256MinTPS,
                unit: SmeltPackagePerformanceUnit.tokensPerSecond
            ),
        ])
        #expect(SmeltPackagePerformanceProfiles.profile(
            for: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
            modelName: "qwen"
        ).minBounds.isEmpty)
    }

    @Test func validatesQwen35TextModelSpecificStartupProfile() throws {
        let modelName = "Qwen/Qwen3.5-2B"
        try llmSpec(
            modelName: modelName,
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: modelName,
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                modelName: modelName
            )
        ).validate()
    }

    @Test func rejectsGateWithoutCanonicalPerformanceProfile() {
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(validation: .init(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )).validate()
        }
    }

    @Test func rejectsStaleKnownPerformanceProfile() {
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(validation: .init(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                performanceProfile: .init(
                    gate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                    command: .run,
                    requiredTraceLabels: [SmeltPackagePerformanceTraceLabel.execToMainDyld],
                    requiredOutputMetrics: [SmeltPackagePerformanceMetricName.traceFirstTokenMS],
                    maxBounds: [
                        .init(
                            metric: SmeltPackagePerformanceMetricName.traceFirstTokenMS,
                            max: SmeltPackagePerformanceBudget.textTraceFirstTokenMaxMS,
                            unit: SmeltPackagePerformanceUnit.milliseconds
                        ),
                    ]
                )
            )).validate()
        }
    }

    @Test func rejectsWeakerTextFloorThanCanonical() {
        let modelName = "Qwen/Qwen3.5-2B"
        let canonical = SmeltPackagePerformanceProfiles.profile(
            for: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
            modelName: modelName
        )
        let weakened = SmeltPackageSpec.Validation.PerformanceProfile(
            gate: canonical.gate,
            command: canonical.command,
            requiredTraceLabels: canonical.requiredTraceLabels,
            requiredOutputMetrics: canonical.requiredOutputMetrics,
            minBounds: canonical.minBounds.map { bound in
                bound.metric == SmeltPackagePerformanceMetricName.decodeTokensPerSecond
                    ? SmeltPackageSpec.Validation.PerformanceProfile.MinBound(
                        metric: bound.metric,
                        min: bound.min - 1,
                        unit: bound.unit
                    )
                    : bound
            },
            maxBounds: canonical.maxBounds
        )
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(
                modelName: modelName,
                validation: .init(
                    performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                    performanceProfile: weakened
                )
            ).validate()
        }
    }

    @Test func validatesQuantizationCalibrationPolicy() throws {
        try llmSpec(quantization: .init(
            format: .gptq,
            groupSize: 128,
            calibration: Self.quantizationCalibration()
        )).validate()
    }

    @Test func validatesStructuredStructureProfile() throws {
        try llmSpec(validation: .init(
            structureProfile: .init(
                id: SmeltPackageStructureProfileID.qwen3TTSRunnable,
                requiredPipelines: ["qwen_kernel"],
                forbiddenPipelines: ["fallback_kernel"],
                requiredFiles: ["manifest.json", "weights.bin"],
                forbiddenFiles: ["debug.trace"],
                requiredRoutes: ["talker:compiled:baked-sidecar"]
            )
        )).validate()
    }

    @Test func qwen3TTSRunnableStructureProfileIsPublicSchemaData() throws {
        let graph = SmeltBlockGraph.qwen3TTSCompiledTalker
        let profile = SmeltPackageStructureProfiles.qwen3TTSRunnable(
            pipelines: ["qwen_kernel"],
            tokenizerFiles: Qwen3TTSManifest.requiredTokenizerFiles,
            graph: graph
        )

        #expect(profile.id == SmeltPackageStructureProfileID.qwen3TTSRunnable)
        #expect(profile.requiredPipelines == ["qwen_kernel"])
        #expect(profile.requiredFiles == SmeltPackageStructureProfiles.qwen3TTSRunnableBaseFiles
            + Qwen3TTSManifest.requiredTokenizerFiles)
        #expect(profile.requiredRoutes == graph.runtimeRoutes
            .map(\.signature))
        try llmSpec(validation: .init(structureProfile: profile)).validate()
    }

    @Test func qwen3TTSPackageProfileIsPublicSchemaData() throws {
        let profile = SmeltQwen3TTSPackageProfiles.runnable
        let structureProfile = profile.structureProfile(
            pipelines: ["qwen_kernel"],
            graph: profile.compiledTalkerGraph
        )

        #expect(profile.packageName == "qwen3-tts-12hz.smeltpkg")
        #expect(profile.modelName == "qwen3-tts-12hz")
        #expect(profile.runtimeArchitecture == SmeltRuntimeGraphPolicy.sidecarTextToCodecAudio.rawValue)
        #expect(profile.baseGraph == .qwen3TTS)
        #expect(profile.compiledTalkerGraph == .qwen3TTSCompiledTalker)
        #expect(profile.compiledTrunkNativeFrontEndGraph == .qwen3TTSCompiledTrunkNativeFrontEnd)
        #expect(profile.supportedBlockGraphs == [
            .qwen3TTS,
            .qwen3TTSCompiledTalker,
            .qwen3TTSCompiledTrunkNativeFrontEnd,
        ])
        #expect(profile.graph(textEmbeddingIsBF16: true) == profile.compiledTalkerGraph)
        #expect(profile.graph(textEmbeddingIsBF16: false) == profile.compiledTrunkNativeFrontEndGraph)
        #expect(profile.loop == .qwen3TTS)
        #expect(profile.tokenizerFiles == Qwen3TTSManifest.requiredTokenizerFiles)
        #expect(sidecarSignatures(profile.sidecars) == sidecarSignatures(
            SmeltPackageSidecarProfiles.qwen3TTSRunnableHeadlessTrunks.map(\.sidecar)
        ))
        #expect(profile.sidecarPaths == ["trunk", "trunk-mtp"])
        #expect(profile.pageSize == 16_384)
        #expect(profile.maxTokens == 2048)
        #expect(profile.eosTokens == [2150])
        #expect(profile.performanceGate == SmeltPackagePerformanceGateID.qwen3TTSTTFA)
        #expect(profile.structureProfileID == SmeltPackageStructureProfileID.qwen3TTSRunnable)
        #expect(structureProfile.id == profile.structureProfileID)
        #expect(structureProfile.requiredFiles == SmeltPackageStructureProfiles.qwen3TTSRunnableBaseFiles
            + profile.tokenizerFiles)
        #expect(SmeltQwen3TTSPackageProfiles.all == [profile])
    }

    @Test func qwen3TTSSidecarProfilesArePublicSchemaData() {
        let trunks = SmeltPackageSidecarProfiles.qwen3TTSRunnableHeadlessTrunks

        #expect(SmeltPackageSidecarKind.known == [SmeltPackageSidecarKind.compiledTrunk])
        #expect(trunks.map(\.id) == ["talker-trunk", "mtp-trunk"])
        #expect(trunks.map(\.path) == ["trunk", "trunk-mtp"])
        #expect(trunks.map(\.kind) == [
            SmeltPackageSidecarKind.compiledTrunk,
            SmeltPackageSidecarKind.compiledTrunk,
        ])
        #expect(trunks.map(\.modelName) == [
            "qwen3-tts-12hz-talker-trunk",
            "qwen3-tts-12hz-mtp-trunk",
        ])
        #expect(SmeltPackageSidecarProfiles.qwen3TTSRunnableSidecarPaths
            == ["trunk", "trunk-mtp"])
        #expect(trunks.map(\.sidecar).map(\.path)
            == SmeltPackageSidecarProfiles.qwen3TTSRunnableSidecarPaths)
    }

    @Test func sidecarAudioSpecRequiresGraphDeclaredSidecarCount() throws {
        let expected = SmeltPackageSidecarProfiles.qwen3TTSRunnableHeadlessTrunks
            .map(\.sidecar)

        #expect(throws: SmeltPackageSpecError.self) {
            try qwenTTSSpec(sidecars: Array(expected.prefix(1))).validate()
        }
        try qwenTTSSpec(sidecars: [
            expected[0],
            .init(id: "custom-mtp-trunk", path: "custom-mtp", kind: expected[1].kind),
        ]).validate()
    }

    @Test func validatesKnownGenericSidecarKind() throws {
        try llmSpec(sidecars: [
            .init(id: "trunk-sidecar", path: "trunk", kind: SmeltPackageSidecarKind.compiledTrunk),
        ]).validate()
    }

    @Test func rejectsUnknownGenericSidecarKind() {
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(sidecars: [
                .init(id: "trunk-sidecar", path: "trunk", kind: "compiled-mtp"),
            ]).validate()
        }
    }

    @Test func rejectsInvalidQuantizationCalibrationPolicy() {
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(quantization: .init(
                format: .f16,
                calibration: Self.quantizationCalibration()
            )).validate()
        }
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(quantization: .init(
                format: .gptq,
                groupSize: 128,
                calibration: .init()
            )).validate()
        }
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(quantization: .init(
                format: .gptq,
                groupSize: 128,
                calibration: .init(
                    corpus: .init(
                        source: "missing",
                        path: "calibration/prompts.jsonl",
                        renderPolicy: "chat-template"
                    )
                )
            )).validate()
        }
    }

    @Test func rejectsInvalidStructureProfile() {
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(validation: .init(
                structureProfile: .init(id: "custom.structure", requiredFiles: ["manifest.json"])
            )).validate()
        }
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(validation: .init(
                structureProfile: .init(id: SmeltPackageStructureProfileID.qwen3TTSRunnable)
            )).validate()
        }
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(validation: .init(
                structureProfile: .init(
                    id: SmeltPackageStructureProfileID.qwen3TTSRunnable,
                    requiredFiles: ["../manifest.json"]
                )
            )).validate()
        }
    }

    @Test func rejectsUnknownPerformanceGateID() {
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(validation: .init(
                parityFixture: "qwen",
                performanceGate: "custom.release"
            )).validate()
        }
    }

    @Test func rejectsMismatchedPerformanceProfileGate() {
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(validation: .init(
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                performanceProfile: .init(
                    gate: SmeltPackagePerformanceGateID.qwen3TTSTTFA,
                    command: .run
                )
            )).validate()
        }
    }

    @Test func rejectsInvalidPerformanceProfileBounds() {
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(validation: .init(
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                performanceProfile: .init(
                    gate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                    command: .run,
                    minBounds: [
                        .init(
                            metric: SmeltPackagePerformanceMetricName.decodeTokensPerSecond,
                            min: 0,
                            unit: SmeltPackagePerformanceUnit.tokensPerSecond
                        ),
                    ]
                )
            )).validate()
        }
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(validation: .init(
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                performanceProfile: .init(
                    gate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                    command: .run,
                    maxBounds: [
                        .init(
                            metric: SmeltPackagePerformanceMetricName.traceFirstTokenMS,
                            max: 0,
                            unit: SmeltPackagePerformanceUnit.milliseconds
                        ),
                    ]
                )
            )).validate()
        }
    }

    @Test func qwen3TTSManifestRejectsWrongValidationGate() {
        let manifest = Qwen3TTSManifest(
            version: 1,
            blocks: .qwen3TTSCompiledTrunkNativeFrontEnd,
            loop: .qwen3TTS,
            modelName: "qwen3-tts",
            pageSize: 16,
            pipelines: [],
            eosTokens: [2150],
            totalBytes: 16,
            weights: [],
            validation: .init(
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                performanceProfile: .init(
                    gate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                    command: .bench
                )
            )
        )
        #expect(throws: SmeltPackageSpecError.self) {
            try manifest.validateQwen3TTSValidation()
        }
    }

    @Test func qwen3TTSManifestRejectsRemovedKindField() {
        let json = """
        {
          "version": 1,
          "kind": "tts",
          "blocks": {
            "kind": "qwen3-tts",
            "trunk": "compiled",
            "frontend": "native"
          },
          "loop": {
            "kind": "qwen3-tts"
          },
          "modelName": "qwen3-tts",
          "pageSize": 16,
          "pipelines": [],
          "eosTokens": [2150],
          "totalBytes": 16,
          "weights": []
        }
        """

        #expect(throws: DecodingError.self) {
            try Qwen3TTSManifest.decode(from: Data(json.utf8))
        }
    }

    @Test func qwen3TTSManifestRejectsRemovedArchitectureField() {
        let json = """
        {
          "version": 1,
          "architecture": "\(SmeltRuntimeGraphPolicy.sidecarTextToCodecAudio.rawValue)",
          "blocks": {
            "kind": "qwen3-tts",
            "trunk": "compiled",
            "frontend": "native"
          },
          "loop": {
            "kind": "qwen3-tts"
          },
          "modelName": "qwen3-tts",
          "pageSize": 16,
          "pipelines": [],
          "eosTokens": [2150],
          "totalBytes": 16,
          "weights": []
        }
        """

        #expect(throws: DecodingError.self) {
            try Qwen3TTSManifest.decode(from: Data(json.utf8))
        }
    }

    @Test func qwen3TTSManifestAcceptsCodecDecoderGraphAndLoop() throws {
        let manifest = Self.qwen3TTSManifest(
            blocks: .qwen3TTSCodecDecoder,
            loop: .qwen3TTSCodecDecoder,
            decode: nil,
            validation: nil
        )

        try manifest.validateQwen3TTSValidation()

        let wrongLoop = Self.qwen3TTSManifest(
            blocks: .qwen3TTSCodecDecoder,
            loop: .qwen3TTS,
            decode: nil,
            validation: nil
        )
        #expect(throws: SmeltPackageSpecError.self) {
            try wrongLoop.validateQwen3TTSValidation()
        }
    }

    @Test func qwen3TTSManifestAllowsMissingDecodePolicyWithoutTTFAGate() throws {
        let manifest = Self.qwen3TTSManifest(
            decode: nil,
            validation: nil
        )

        try manifest.validateQwen3TTSValidation()
    }

    @Test func qwen3TTSManifestRejectsMissingGraphAndLoop() throws {
        let manifest = Self.qwen3TTSManifest(
            blocks: nil,
            loop: nil,
            validation: nil
        )

        #expect(throws: SmeltPackageSpecError.self) {
            try manifest.validateQwen3TTSValidation()
        }
    }

    @Test func qwen3TTSManifestRejectsPartialStampedGraphAndLoop() {
        let graphOnly = Self.qwen3TTSManifest(loop: nil, validation: nil)
        #expect(throws: SmeltPackageSpecError.self) {
            try graphOnly.validateQwen3TTSValidation()
        }

        let loopOnly = Self.qwen3TTSManifest(blocks: nil, validation: nil)
        #expect(throws: SmeltPackageSpecError.self) {
            try loopOnly.validateQwen3TTSValidation()
        }
    }

    @Test func qwen3TTSManifestRejectsStampedGraphDrift() {
        let manifest = Self.qwen3TTSManifest(
            blocks: .tokenFeedbackText,
            validation: nil
        )

        #expect(throws: SmeltPackageSpecError.self) {
            try manifest.validateQwen3TTSValidation()
        }
    }

    @Test func qwen3TTSManifestRejectsStampedLoopDrift() {
        let loop = SmeltLoopSchedule(
            setup: [SmeltLoopSchedule.Phase(name: "prefill", blocks: ["talker"])],
            perStep: [SmeltLoopSchedule.Phase(name: "decode", blocks: ["talker"])],
            emission: .chunked(first: 1, max: 1, growth: .double, via: "codec-decoder"),
            stop: [.eosToken, .maxSteps]
        )
        let manifest = Self.qwen3TTSManifest(loop: loop, validation: nil)

        #expect(throws: SmeltPackageSpecError.self) {
            try manifest.validateQwen3TTSValidation()
        }
    }

    @Test func qwen3TTSManifestRequiresPerformanceProfileForTTFAGate() {
        let manifest = Self.qwen3TTSManifest(
            validation: .init(
                performanceGate: SmeltPackagePerformanceGateID.qwen3TTSTTFA
            )
        )

        #expect(throws: SmeltPackageSpecError.self) {
            try manifest.validateQwen3TTSValidation()
        }
    }

    @Test func qwen3TTSManifestRejectsStaleTTFAPerformanceProfile() {
        let manifest = Self.qwen3TTSManifest(
            validation: .init(
                performanceGate: SmeltPackagePerformanceGateID.qwen3TTSTTFA,
                performanceProfile: .init(
                    gate: SmeltPackagePerformanceGateID.qwen3TTSTTFA,
                    command: .bench,
                    requiredOutputMetrics: [
                        SmeltPackagePerformanceMetricName.ttfaSeconds,
                    ],
                    maxBounds: [
                        .init(
                            metric: SmeltPackagePerformanceMetricName.ttfaSeconds,
                            max: SmeltPackagePerformanceBudget.qwen3TTSTTFAMaxSeconds,
                            unit: SmeltPackagePerformanceUnit.seconds
                        ),
                    ]
                )
            )
        )

        #expect(throws: SmeltPackageSpecError.self) {
            try manifest.validateQwen3TTSValidation()
        }
    }

    @Test func qwen3TTSManifestRejectsRTFBoundWithSecondsUnit() {
        let manifest = Self.qwen3TTSManifest(
            validation: .init(
                performanceGate: SmeltPackagePerformanceGateID.qwen3TTSTTFA,
                performanceProfile: .init(
                    gate: SmeltPackagePerformanceGateID.qwen3TTSTTFA,
                    command: .bench,
                    requiredOutputMetrics: SmeltPackagePerformanceMetricName.qwen3TTSTTFARequired,
                    maxBounds: [
                        .init(
                            metric: SmeltPackagePerformanceMetricName.firstAudioSeconds,
                            max: SmeltPackagePerformanceBudget.qwen3TTSFirstAudioMaxSeconds,
                            unit: SmeltPackagePerformanceUnit.seconds
                        ),
                        .init(
                            metric: SmeltPackagePerformanceMetricName.ttfaSeconds,
                            max: SmeltPackagePerformanceBudget.qwen3TTSTTFAMaxSeconds,
                            unit: SmeltPackagePerformanceUnit.seconds
                        ),
                        .init(
                            metric: SmeltPackagePerformanceMetricName.realTimeFactor,
                            max: SmeltPackagePerformanceBudget.qwen3TTSRealTimeFactorMax,
                            unit: SmeltPackagePerformanceUnit.seconds
                        ),
                    ]
                )
            )
        )

        #expect(throws: SmeltPackageSpecError.self) {
            try manifest.validateQwen3TTSValidation()
        }
    }

    @Test func qwen3TTSManifestRequiresDecodePolicyForTTFAGate() {
        let manifest = Self.qwen3TTSManifest(
            decode: nil,
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "qwen3-tts",
                performanceGate: SmeltPackagePerformanceGateID.qwen3TTSTTFA
            )
        )

        #expect(throws: SmeltPackageSpecError.self) {
            try manifest.validateQwen3TTSValidation()
        }
    }

    @Test func qwen3TTSManifestRejectsInvalidDecodePolicyForTTFAGate() {
        let manifest = Self.qwen3TTSManifest(
            decode: .init(
                doSample: true,
                temperature: 0,
                topK: 8,
                subtalkerTemperature: 0.8,
                subtalkerTopK: 8
            ),
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "qwen3-tts",
                performanceGate: SmeltPackagePerformanceGateID.qwen3TTSTTFA
            )
        )

        #expect(throws: SmeltPackageSpecError.self) {
            try manifest.validateQwen3TTSValidation()
        }
    }

    @Test func qwen3TTSManifestRejectsWrongStructureProfile() {
        let manifest = Qwen3TTSManifest(
            version: 1,
            blocks: .qwen3TTSCompiledTrunkNativeFrontEnd,
            loop: .qwen3TTS,
            modelName: "qwen3-tts",
            pageSize: 16,
            pipelines: [],
            eosTokens: [2150],
            totalBytes: 16,
            weights: [],
            validation: .init(
                structureProfile: .init(id: "custom.structure", requiredFiles: ["manifest.json"])
            )
        )
        #expect(throws: SmeltPackageSpecError.self) {
            try manifest.validateQwen3TTSValidation()
        }
    }

    @Test func qwen3TTSManifestChecksStructureProfilePipelinesAndRoutes() throws {
        let graph = SmeltBlockGraph.qwen3TTSCompiledTrunkNativeFrontEnd
        let manifest = Self.qwen3TTSManifest(
            blocks: graph,
            pipelines: ["qwen_kernel"],
            structureProfile: .init(
                id: SmeltPackageStructureProfileID.qwen3TTSRunnable,
                requiredPipelines: ["qwen_kernel"],
                forbiddenPipelines: ["fallback_kernel"],
                requiredRoutes: [
                    "tts-frontend:native:none",
                    "talker:compiled:baked-sidecar",
                ]
            )
        )

        try manifest.validateQwen3TTSValidation()

        let missingPipeline = Self.qwen3TTSManifest(
            blocks: graph,
            pipelines: [],
            structureProfile: .init(
                id: SmeltPackageStructureProfileID.qwen3TTSRunnable,
                requiredPipelines: ["qwen_kernel"]
            )
        )
        #expect(throws: SmeltPackageSpecError.self) {
            try missingPipeline.validateQwen3TTSValidation()
        }

        let missingRoute = Self.qwen3TTSManifest(
            blocks: graph,
            pipelines: ["qwen_kernel"],
            structureProfile: .init(
                id: SmeltPackageStructureProfileID.qwen3TTSRunnable,
                requiredRoutes: ["talker:compiled:runtime-emit"]
            )
        )
        #expect(throws: SmeltPackageSpecError.self) {
            try missingRoute.validateQwen3TTSValidation()
        }
    }

    @Test func qwen3TTSManifestChecksStructureProfilePackageFiles() throws {
        let package = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: package) }
        try Data("{}".utf8).write(to: package.appendingPathComponent("manifest.json"))

        let manifest = Self.qwen3TTSManifest(
            structureProfile: .init(
                id: SmeltPackageStructureProfileID.qwen3TTSRunnable,
                requiredFiles: ["manifest.json"],
                forbiddenFiles: ["debug.trace"]
            )
        )
        try manifest.validateQwen3TTSValidation(packagePath: package.path)

        let missingFile = Self.qwen3TTSManifest(
            structureProfile: .init(
                id: SmeltPackageStructureProfileID.qwen3TTSRunnable,
                requiredFiles: ["trunk"]
            )
        )
        #expect(throws: SmeltPackageSpecError.self) {
            try missingFile.validateQwen3TTSValidation(packagePath: package.path)
        }

        let unsafeFile = Self.qwen3TTSManifest(
            structureProfile: .init(
                id: SmeltPackageStructureProfileID.qwen3TTSRunnable,
                requiredFiles: ["../manifest.json"]
            )
        )
        #expect(throws: SmeltPackageSpecError.self) {
            try unsafeFile.validateQwen3TTSValidation(packagePath: package.path)
        }
    }

    @Test func rejectsUnsafePackageOutputPaths() {
        let badArtifact = SmeltPackageSpec.Artifact(
            id: "weights",
            path: "../weights.bin",
            role: "weights"
        )
        #expect(throws: SmeltPackageSpecError.self) {
            try llmSpec(artifacts: [badArtifact]).validate()
        }
    }

    private static func quantizationCalibration() -> SmeltPackageSpec.QuantizationPlan.Calibration {
        .init(
            corpus: .init(
                source: "weights",
                path: "calibration/prompts.jsonl",
                renderPolicy: "chat-template",
                maxSamples: 128,
                maxTokens: 2048
            ),
            captureArtifacts: [
                .init(
                    id: "prefill-captures",
                    path: "gptq_capture_points.json",
                    role: "activation-capture"
                ),
            ],
            sideInputs: [
                .init(id: "imatrix", path: "calibration/imatrix.dat", role: "imatrix"),
            ],
            stagedPackages: [
                .init(
                    id: "bf16-prefill",
                    path: "calibration/bf16-prefill.smeltpkg",
                    role: "activation-source"
                ),
            ],
            resourceBounds: .init(
                maxSamples: 128,
                maxTokens: 2048,
                maxLayersPerPass: 4,
                maxBytes: 1_000_000_000
            ),
            equalityGates: [
                .init(
                    id: "packed-regions",
                    candidate: "weights.bin",
                    reference: "calibration/reference.weights.bin",
                    metric: "sha256"
                ),
            ]
        )
    }

    private static func qwen3TTSManifest(
        blocks: SmeltBlockGraph? = SmeltBlockGraph.qwen3TTSCompiledTrunkNativeFrontEnd,
        loop: SmeltLoopSchedule? = .qwen3TTS,
        pipelines: [String] = [],
        decode: Qwen3TTSManifest.Decode? = .init(
            doSample: false,
            temperature: 1.0,
            topK: 50,
            subtalkerTemperature: 1.0,
            subtalkerTopK: 50
        ),
        validation: SmeltPackageSpec.Validation? = nil,
        structureProfile: SmeltPackageSpec.Validation.StructureProfile? = nil
    ) -> Qwen3TTSManifest {
        Qwen3TTSManifest(
            version: 1,
            blocks: blocks,
            loop: loop,
            modelName: "qwen3-tts",
            pageSize: 16,
            pipelines: pipelines,
            eosTokens: [2150],
            totalBytes: 16,
            weights: [],
            decode: decode,
            validation: validation ?? structureProfile.map {
                SmeltPackageSpec.Validation(structureProfile: $0)
            }
        )
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(
            "smelt-package-spec-\(UUID().uuidString)",
            isDirectory: true
        )
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
