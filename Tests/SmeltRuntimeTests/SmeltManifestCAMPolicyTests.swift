import XCTest
import SmeltSchema

final class SmeltManifestCAMPolicyTests: XCTestCase {
    func testTextManifestRejectsMissingInferenceWithoutGateBeforeRuntime() throws {
        let manifest = makeManifest(inference: nil, decode: Self.decode)

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation requires package-owned inference policy"
                ),
                String(describing: error)
            )
        }
    }

    func testTextManifestRejectsGraphOnlyTopologyWithoutGateBeforeRuntime() throws {
        let manifest = makeManifest(inference: Self.inference, decode: Self.decode, loop: nil)

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation requires blocks and loop"
                ),
                String(describing: error)
            )
        }
    }

    func testTextManifestRejectsRootKindBeforeRuntime() throws {
        let manifest = makeManifest(
            kind: "llm",
            inference: Self.inference,
            decode: Self.decode,
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains("must not declare root kind"),
                String(describing: error)
            )
        }
    }

    func testTextManifestRejectsRootArchitectureBeforeRuntime() throws {
        let manifest = makeManifest(
            architecture: "text-generation",
            inference: Self.inference,
            decode: Self.decode,
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains("must not declare root architecture"),
                String(describing: error)
            )
        }
    }

    func testHeadlessTrunkRejectsRunnableTextPolicyBeforeRuntime() throws {
        let manifest = makeManifest(
            headlessTrunkABI: true,
            inference: Self.inference,
            decode: nil,
            blocks: nil,
            loop: nil,
            validation: nil
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "headless trunk manifest must not declare runnable text policy"
                ),
                String(describing: error)
            )
        }
    }

    func testHeadlessTrunkRejectsRunnableTextTopologyBeforeRuntime() throws {
        let manifest = makeManifest(
            headlessTrunkABI: true,
            inference: nil,
            decode: nil,
            blocks: .tokenFeedbackText,
            loop: .tokenFeedbackText,
            validation: nil
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "headless trunk manifest must not declare runnable text policy"
                ),
                String(describing: error)
            )
        }
    }

    func testCAMGatedTextManifestRequiresInferencePolicyBeforeDecodeFallback() throws {
        let manifest = makeManifest(
            inference: nil,
            decode: Self.decode,
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation requires package-owned inference policy"
                ),
                String(describing: error)
            )
        }
    }

    func testCAMGatedTextManifestRequiresGraphAndLoop() throws {
        let manifest = makeManifest(
            inference: Self.inference,
            decode: Self.decode,
            loop: nil,
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation requires blocks and loop"
                ),
                String(describing: error)
            )
        }
    }

    func testCAMGatedTextManifestRejectsStructurallyValidWrongLoop() throws {
        let wrongLoop = SmeltLoopSchedule(
            setup: [],
            perStep: [SmeltLoopSchedule.Phase(name: "decode", blocks: ["tokenizer"])],
            emission: .perStep,
            stop: [.eosToken, .maxSteps, .hostCancel]
        )
        let manifest = makeManifest(
            inference: Self.inference,
            decode: Self.decode,
            loop: wrongLoop,
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation requires per-step phase to drive token-feedback trunk and text head"
                ),
                String(describing: error)
            )
        }
    }

    func testCAMGatedTextManifestRejectsLoopDrift() throws {
        let driftedLoop = SmeltLoopSchedule(
            setup: [SmeltLoopSchedule.Phase(name: "prefill", blocks: ["trunk", "text-head"])],
            perStep: [SmeltLoopSchedule.Phase(name: "decode", blocks: ["missing-block"])],
            emission: .perStep,
            stop: [.eosToken, .maxSteps, .hostCancel]
        )
        let manifest = makeManifest(
            inference: Self.inference,
            decode: Self.decode,
            loop: driftedLoop,
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "phase 'decode' drives unknown block 'missing-block'"
                ),
                String(describing: error)
            )
        }
    }

    func testPredecodedCAMGatedTextManifestStillRequiresInferencePolicy() throws {
        let manifest = makeManifest(
            inference: nil,
            decode: Self.decode,
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )
        let predecoded = try JSONDecoder().decode(
            SmeltManifest.self,
            from: manifest.encodeJSON()
        )

        XCTAssertThrowsError(try predecoded.validatePackageOwnedRuntimePolicy()) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation requires package-owned inference policy"
                ),
                String(describing: error)
            )
        }
    }

    func testPredecodedTextManifestRejectsRootArchitectureBeforeRuntime() throws {
        let manifest = makeManifest(
            architecture: "text-generation",
            inference: Self.inference,
            decode: Self.decode,
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )
        let predecoded = try JSONDecoder().decode(
            SmeltManifest.self,
            from: manifest.encodeJSON()
        )

        XCTAssertThrowsError(try predecoded.validatePackageOwnedRuntimePolicy()) { error in
            XCTAssertTrue(
                String(describing: error).contains("must not declare root architecture"),
                String(describing: error)
            )
        }
    }

    func testCAMGatedTextManifestRequiresChatTemplateBeforeDecodeFallback() throws {
        let manifest = makeManifest(
            inference: .init(maxTokens: 32, eosTokens: [1]),
            decode: Self.decode,
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation requires inference.chat_template"
                ),
                String(describing: error)
            )
        }
    }

    func testCAMGatedTextManifestRejectsUnknownChatTemplateBeforeRuntime() throws {
        let manifest = makeManifest(
            inference: .init(maxTokens: 32, eosTokens: [1], chatTemplate: "future-template"),
            decode: Self.decode,
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation inference.chat_template 'future-template' is not supported"
                ),
                String(describing: error)
            )
        }
    }

    func testCAMGatedTextManifestRejectsUnknownToolTranscriptCodec() throws {
        let manifest = makeManifest(
            inference: .init(
                maxTokens: 32,
                eosTokens: [1],
                chatTemplate: "chatml",
                thinkingPolicy: .disabled,
                toolTranscriptCodec: "future-tools"
            ),
            decode: Self.decode
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "inference.tool_transcript_codec 'future-tools' is not supported"
                ),
                String(describing: error)
            )
        }
    }

    func testCAMGatedTextManifestRequiresThinkingPolicyBeforeRuntime() throws {
        let manifest = makeManifest(
            inference: .init(maxTokens: 32, eosTokens: [1], chatTemplate: "chatml"),
            decode: Self.decode,
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation requires inference.thinking_policy"
                ),
                String(describing: error)
            )
        }
    }

    func testCAMGatedTextManifestRequiresDecodePolicyBeforeRuntime() throws {
        let manifest = makeManifest(
            inference: Self.inference,
            decode: nil,
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation requires package-owned decode policy"
                ),
                String(describing: error)
            )
        }
    }

    func testCAMGatedTextManifestRequiresDecodeMaxStepsBeforeRuntime() throws {
        let manifest = makeManifest(
            inference: Self.inference,
            decode: .init(sampler: .init(mode: .greedy)),
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation requires decode.max_steps"
                ),
                String(describing: error)
            )
        }
    }

    func testCAMGatedTextManifestRejectsDecodeMaxStepsDriftBeforeRuntime() throws {
        let manifest = makeManifest(
            inference: Self.inference,
            decode: .init(sampler: .init(mode: .greedy), maxSteps: Self.inference.maxTokens + 1),
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation requires decode.max_steps to match inference.max_tokens"
                ),
                String(describing: error)
            )
        }
    }

    func testCAMGatedTextManifestRequiresEOSTokensBeforeDecodeFallback() throws {
        let manifest = makeManifest(
            inference: .init(maxTokens: 32, eosTokens: [], chatTemplate: "chatml"),
            decode: Self.decode,
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation requires inference.eos_tokens"
                ),
                String(describing: error)
            )
        }
    }

    func testCAMGatedTextManifestRequiresCanonicalPerformanceProfile() throws {
        let manifest = makeManifest(
            inference: Self.inference,
            decode: Self.decode,
            validation: .init(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation requires performance_profile"
                ),
                String(describing: error)
            )
        }
    }

    func testCAMGatedTextManifestRejectsStalePerformanceProfile() throws {
        let manifest = makeManifest(
            inference: Self.inference,
            decode: Self.decode,
            validation: .init(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                performanceProfile: .init(
                    gate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                    command: .run,
                    requiredTraceLabels: SmeltPackagePerformanceTraceLabel
                        .textDecodePrefillStartupRequired,
                    requiredOutputMetrics: [SmeltPackagePerformanceMetricName.traceFirstTokenMS],
                    maxBounds: [
                        .init(
                            metric: SmeltPackagePerformanceMetricName.traceFirstTokenMS,
                            max: SmeltPackagePerformanceBudget.textTraceFirstTokenMaxMS,
                            unit: SmeltPackagePerformanceUnit.milliseconds
                        ),
                    ]
                )
            )
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation performance_profile missing canonical required output metric"
                ),
                String(describing: error)
            )
        }
    }

    func testCAMGatedTextManifestRejectsLoosenedStartupBound() throws {
        let canonical = SmeltPackagePerformanceProfiles.profile(
            for: SmeltPackagePerformanceGateID.textDecodePrefillStartup
        )
        let loose = SmeltPackageSpec.Validation.PerformanceProfile(
            gate: canonical.gate,
            command: canonical.command,
            requiredTraceLabels: canonical.requiredTraceLabels,
            requiredOutputMetrics: canonical.requiredOutputMetrics,
            minBounds: canonical.minBounds,
            maxBounds: [
                .init(
                    metric: SmeltPackagePerformanceMetricName.traceFirstTokenMS,
                    max: SmeltPackagePerformanceBudget.textTraceFirstTokenMaxMS + 20,
                    unit: SmeltPackagePerformanceUnit.milliseconds
                ),
            ]
        )
        let manifest = makeManifest(
            inference: Self.inference,
            decode: Self.decode,
            validation: .init(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                performanceProfile: loose
            )
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation performance_profile missing canonical max-bound: trace_first_token_ms"
                ),
                String(describing: error)
            )
        }
    }

    func testCAMGatedQwen35TextManifestAcceptsSanctionedStartupBound() throws {
        let modelName = "Qwen/Qwen3.5-2B"
        let manifest = makeManifest(
            inference: Self.inference,
            decode: Self.decode,
            modelName: modelName,
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: modelName,
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                modelName: modelName
            )
        )

        XCTAssertNoThrow(try SmeltManifest.decode(from: manifest.encodeJSON()))
    }

    func testCAMGatedQwen35FastManifestRejectsTextStartupBound() throws {
        let modelName = "Qwen/Qwen3.5-0.8B"
        let relaxed = SmeltPackagePerformanceProfiles.profile(
            for: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
            modelName: "Qwen/Qwen3.5-2B"
        )
        let manifest = makeManifest(
            inference: Self.inference,
            decode: Self.decode,
            modelName: modelName,
            validation: .init(
                parityFixture: modelName,
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                performanceProfile: relaxed
            )
        )

        XCTAssertThrowsError(try SmeltManifest.decode(from: manifest.encodeJSON())) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation performance_profile missing canonical min-bound: decode_tokens_per_second"
                ),
                String(describing: error)
            )
        }
    }

    func testCAMGatedTextManifestCarriesResolvedInferencePolicy() throws {
        let manifest = makeManifest(
            inference: Self.inference,
            decode: Self.decode,
            validation: SmeltPackagePerformanceProfiles.validation(
                parityFixture: "qwen",
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup
            )
        )

        let decoded = try SmeltManifest.decode(from: manifest.encodeJSON())

        XCTAssertEqual(decoded.inference?.eosTokens, [1])
        XCTAssertEqual(decoded.inference?.chatTemplate, "chatml")
        XCTAssertEqual(decoded.inference?.thinkingPolicy, .disabled)
        XCTAssertEqual(decoded.decode?.sampler.mode, .greedy)
        XCTAssertEqual(decoded.decode?.maxSteps, Self.inference.maxTokens)
    }

    func testResolvedInferencePolicyPrefersPackagePolicy() throws {
        let packagePolicy = SmeltInferenceManifest(
            maxTokens: 7,
            eosTokens: [42],
            chatTemplate: "package-template",
            thinkingPolicy: .enabled
        )
        let manifest = makeManifest(
            inference: packagePolicy,
            modelName: "qwen"
        )

        let resolved = try manifest.resolvedInferencePolicy()

        XCTAssertEqual(resolved.source, .package)
        XCTAssertEqual(resolved.inference.maxTokens, 7)
        XCTAssertEqual(resolved.inference.eosTokens, [42])
        XCTAssertEqual(resolved.inference.chatTemplate, "package-template")
        XCTAssertEqual(resolved.inference.thinkingPolicy, .enabled)
    }

    func testResolvedInferencePolicyDoesNotDerivePackagePolicyFromModelName() throws {
        let manifest = makeManifest(inference: nil, modelName: "Qwen/Qwen3.5-0.8B")

        XCTAssertThrowsError(try manifest.resolvedInferencePolicy()) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation requires package-owned inference policy"
                ),
                String(describing: error)
            )
        }
    }

    func testResolvedInferencePolicyRejectsMissingPackagePolicy() throws {
        let manifest = makeManifest(inference: nil, modelName: "future/unknown-family")

        XCTAssertThrowsError(try manifest.resolvedInferencePolicy()) { error in
            XCTAssertTrue(
                String(describing: error).contains(
                    "text CAM validation requires package-owned inference policy"
                ),
                String(describing: error)
            )
        }
    }

    func testKnownPromptTemplateNamesAreSchemaOwned() throws {
        XCTAssertEqual(SmeltPromptTemplateName.raw, "")
        XCTAssertEqual(SmeltPromptTemplateName.headerTurns, "header-turns")
        XCTAssertEqual(SmeltPromptTemplateName.channelTurns, "channel-turns")
        XCTAssertEqual(SmeltPromptTemplateName.chatML, "chatml")
        XCTAssertEqual(
            SmeltPromptTemplateName.chatMLXMLTools, "chatml-xml-tools"
        )
        XCTAssertEqual(
            SmeltPromptTemplateName.availablePromptTemplates,
            "chatml, chatml-xml-tools, header-turns, channel-turns"
        )
        XCTAssertTrue(SmeltPromptTemplateName.isKnownPromptTemplate("chatml"))
        XCTAssertTrue(
            SmeltPromptTemplateName.isKnownPromptTemplate("chatml-xml-tools")
        )
        XCTAssertFalse(SmeltPromptTemplateName.isKnownPromptTemplate("future-template"))
        XCTAssertEqual(
            SmeltPromptTemplateName.canonicalRoleTemplate(for: "chatml-xml-tools"),
            "chatml"
        )
    }

    func testKnownToolTranscriptCodecNamesAreSchemaOwned() throws {
        XCTAssertEqual(
            SmeltToolTranscriptCodecName.xmlFunctionParameters,
            "xml-function-parameters"
        )
        XCTAssertEqual(SmeltToolTranscriptCodecName.channelCalls, "channel-calls")
        XCTAssertEqual(SmeltToolTranscriptCodecName.inkling, "inkling")
        XCTAssertEqual(
            SmeltToolTranscriptCodecName.availableCodecs,
            "xml-function-parameters, channel-calls, inkling"
        )
        XCTAssertTrue(
            SmeltToolTranscriptCodecName.isKnown("xml-function-parameters")
        )
        XCTAssertFalse(SmeltToolTranscriptCodecName.isKnown("future-tools"))
        XCTAssertEqual(
            SmeltToolTranscriptCodecName.inferredFromLegacyPromptTemplate(
                "chatml-xml-tools"
            ),
            "xml-function-parameters"
        )
        XCTAssertNil(
            SmeltToolTranscriptCodecName.inferredFromLegacyPromptTemplate("chatml")
        )
    }

    func testPromptStateRestoreModeDerivesFromPersistentStateSemantics() {
        XCTAssertEqual(
            SmeltPromptStateRestoreMode.derive(
                fromPersistentStateNames: ["kv-cache"]
            ),
            .positionIndexed
        )
        for opaqueState in [
            "conv-state", "rec-state", "short-convolution-state", "future-state",
        ] {
            XCTAssertEqual(
                SmeltPromptStateRestoreMode.derive(
                    fromPersistentStateNames: ["kv-cache", opaqueState]
                ),
                .exactPosition,
                opaqueState
            )
        }
    }

    private static let inference = SmeltInferenceManifest(
        maxTokens: 32,
        eosTokens: [1],
        chatTemplate: "chatml",
        thinkingPolicy: .disabled
    )

    private static let decode = SmeltPackageSpec.DecodePolicy(
        sampler: .init(mode: .greedy),
        maxSteps: inference.maxTokens
    )

    private func makeManifest(
        kind: String? = nil,
        architecture: String? = nil,
        headlessTrunkABI: Bool? = nil,
        inference: SmeltInferenceManifest?,
        decode: SmeltPackageSpec.DecodePolicy? = nil,
        modelName: String = "qwen",
        blocks: SmeltBlockGraph? = .tokenFeedbackText,
        loop: SmeltLoopSchedule? = .tokenFeedbackText,
        validation: SmeltPackageSpec.Validation? = nil
    ) -> SmeltManifest {
        SmeltManifest(
            kind: kind,
            architecture: architecture,
            headlessTrunkABI: headlessTrunkABI,
            blocks: blocks,
            loop: loop,
            modelName: modelName,
            config: SmeltManifestConfig(
                hiddenSize: 1,
                numLayers: 1,
                vocabSize: 1,
                staticSeqCapacity: 1,
                ropeDim: 1,
                numDeltaLayers: 0,
                numAttnLayers: 0,
                ffnDim: 1
            ),
            context: nil,
            checksums: SmeltManifestChecksums(
                weightsBin: "",
                metallib: "",
                generatedSwift: "",
                dispatchesBin: ""
            ),
            device: SmeltDeviceRequirements(
                metalFamily: .apple7,
                minMemoryBytes: 1
            ),
            weights: SmeltWeightManifest(totalBytes: 0, entries: []),
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
            inference: inference,
            decode: decode,
            validation: validation
        )
    }
}
