import Foundation
import Testing
@testable import SmeltCompiler
@testable import SmeltSchema

@Suite struct SmeltCAMCheckedPackageProjectionTests {
    private static let syntheticQwen3TTSProjectedPackageSpecSHA256 =
        "b8b48cecd28ad529abeca763eae06bf25290cd99f36d069142fc9cd8338f362f"

    @Test func attentionQKNormWeightSemanticsAreIndependentOfResidualNorm() throws {
        let ir = try SmeltCAMCheckedPackageProjector.sourceModelIR(
            cam: registryModuleIR("bonsai_27b_ternary")
        )

        #expect(ir.config.normMode == .onePlusWeight)
        #expect(ir.config.attention?.qkNormMode == .weight)
        #expect(ir.config.attention?.ropeLayout == .splitHalf)
    }

    @Test func projectsCheckedTextCAMToPackageSpec() throws {
        let cam = registryModuleIR("qwen35_text")
        let expected = try CAMModuleCompletionMatrix.module(fixture: "qwen35_text.module.json")
        let expectedProjection = try #require(expected.packageProjection)
        let artifactRoot = "artifacts/qwen35-2b-qmm16x128/Qwen_Qwen3.5-2B-build-source.smeltpkg"
        let projection = try SmeltCAMCheckedPackageProjector.project(
            cam: cam,
            artifactRoot: artifactRoot
        )
        let plan = try SmeltPackageResolvedPlan.resolve(projection.spec)

        #expect(projection.packageProjectionID == expectedProjection.id)
        #expect(projection.packageProjectionVersion == expectedProjection.version)
        #expect(projection.camSemanticSHA256 == expected.semanticSHA256)
        #expect(projection.exportABISHA256 == expected.exportABISHA256)
        #expect(projection.descriptorVersion == SmeltCAMPackageDescriptor.currentDescriptorVersion)
        #expect(
            projection.descriptorGraphSignatureSHA256
                == expected.descriptorGraphSignatureSHA256
        )
        #expect(
            projection.projectedPackageSpecSHA256
                == expectedProjection.projectedPackageSpecSHA256
        )
        #expect(plan.sources.map { "\($0.id):\($0.kind.rawValue):\($0.locator)" } == [
            "package:local-directory:\(artifactRoot)",
        ])
        #expect(projection.packageFiles == expectedProjection.packageFiles)
        #expect(plan.runtime.routes.map(\.signature) == [
            "tokenizer:native:none",
            "trunk:compiled:baked-inline",
            "text-head:native:none",
        ])
        #expect(projection.spec.runtime.architecture == SmeltRuntimeGraphPolicy.textGeneration.rawValue)
        #expect(plan.runtime.architecture == SmeltRuntimeGraphPolicy.textGeneration.rawValue)
        #expect(projection.spec.loop.setupSignatures == ["prefill:trunk,text-head"])
        #expect(projection.spec.loop.perStepSignatures == ["decode:trunk,text-head"])
        #expect(projection.spec.loop.emissionSignature == "per-step")
        #expect(projection.spec.loop.stop == [.eosToken, .maxSteps, .hostCancel])
        #expect(plan.runtime.setupPhases == ["prefill:trunk,text-head"])
        #expect(plan.runtime.perStepPhases == ["decode:trunk,text-head"])
        #expect(plan.runtime.emission == "per-step")
        #expect(plan.policy.inference?.maxTokens == 512)
        #expect(plan.policy.inference?.eosTokens == [248044, 248046])
        #expect(plan.policy.inference?.chatTemplate == SmeltPromptTemplateName.chatML)
        #expect(plan.policy.inference?.thinkingPolicy == .disabled)
        #expect(plan.policy.inference?.promptStateRestoreMode == .exactPosition)
        #expect(plan.policy.decode?.sampler.mode == .greedy)
        #expect(plan.policy.decode?.maxSteps == 512)
        #expect(plan.validationPerformanceProfile?.maxBounds == [
            .init(
                metric: SmeltPackagePerformanceMetricName.traceFirstTokenMS,
                max: SmeltPackagePerformanceBudget.qwen35TextTraceFirstTokenMaxMS,
                unit: SmeltPackagePerformanceUnit.milliseconds
            ),
        ])
        try Self.assertLoadingCheckpointMap(projection.spec, expected: "hf.qwen", label: "qwen35_text")
    }

    @Test func rejectsCheckedTextProjectionWithoutAuthoredCheckpointMap() throws {
        let cam = try mutatedModuleIR(registryModuleIR("qwen35_text")) { object in
            try Self.mutateSource(&object, id: "weights", label: "missing checkpoint map") { source in
                guard source.removeValue(forKey: "checkpointMap") != nil else {
                    throw Self.mutationFailure("missing checkpoint map: weights source has no checkpointMap")
                }
            }
        }
        try Self.expectProjectionRejectedWithoutProfileMiss(
            cam,
            label: "missing checkpoint map",
            requiredFragments: ["weights source checkpoint map missing"]
        )
    }

    @Test func rejectsCheckedTextProjectionWithUnsupportedCheckpointMap() throws {
        let cam = try mutatedModuleIR(registryModuleIR("qwen35_text")) { object in
            try Self.mutateSource(&object, id: "weights", label: "unsupported checkpoint map") { source in
                try Self.setValue(
                    &source,
                    path: ["checkpointMap"],
                    to: "hf.not-real",
                    label: "unsupported checkpoint map"
                )
            }
        }
        try Self.expectProjectionRejectedWithoutProfileMiss(
            cam,
            label: "unsupported checkpoint map",
            requiredFragments: ["weights source checkpoint map 'hf.not-real' is unsupported"]
        )
    }

    @Test func projectsCheckedFastCAMToPrefillPackageSpec() throws {
        let cam = registryModuleIR("qwen35_fast")
        let expected = try CAMModuleCompletionMatrix.module(fixture: "qwen35_fast.module.json")
        let expectedProjection = try #require(expected.packageProjection)
        let artifactRoot = "artifacts/qwen35-0.8b-qmm16x128/Qwen_Qwen3.5-0.8B-build-source.smeltpkg"
        let projection = try SmeltCAMCheckedPackageProjector.project(
            cam: cam,
            artifactRoot: artifactRoot
        )
        let plan = try SmeltPackageResolvedPlan.resolve(projection.spec)

        #expect(projection.packageProjectionID == expectedProjection.id)
        #expect(projection.packageProjectionVersion == expectedProjection.version)
        #expect(projection.camSemanticSHA256 == expected.semanticSHA256)
        #expect(projection.exportABISHA256 == expected.exportABISHA256)
        #expect(projection.descriptorVersion == SmeltCAMPackageDescriptor.currentDescriptorVersion)
        #expect(
            projection.descriptorGraphSignatureSHA256
                == expected.descriptorGraphSignatureSHA256
        )
        #expect(
            projection.projectedPackageSpecSHA256
                == expectedProjection.projectedPackageSpecSHA256
        )
        #expect(plan.sources.map { "\($0.id):\($0.kind.rawValue):\($0.locator)" } == [
            "package:local-directory:\(artifactRoot)",
        ])
        #expect(projection.packageFiles == expectedProjection.packageFiles)
        #expect(projection.packageFiles.contains("prefill_dispatches.bin"))
        #expect(plan.runtime.routes.map(\.signature) == [
            "tokenizer:native:none",
            "trunk:compiled:baked-inline",
            "text-head:native:none",
        ])
        #expect(projection.spec.runtime.architecture == SmeltRuntimeGraphPolicy.textGeneration.rawValue)
        #expect(plan.runtime.architecture == SmeltRuntimeGraphPolicy.textGeneration.rawValue)
        #expect(projection.spec.loop.setupSignatures == ["prefill:trunk,text-head"])
        #expect(projection.spec.loop.perStepSignatures == ["decode:trunk,text-head"])
        #expect(projection.spec.loop.emissionSignature == "per-step")
        #expect(projection.spec.loop.stop == [.eosToken, .maxSteps, .hostCancel])
        #expect(plan.runtime.setupPhases == ["prefill:trunk,text-head"])
        #expect(plan.runtime.perStepPhases == ["decode:trunk,text-head"])
        #expect(plan.runtime.emission == "per-step")
        #expect(plan.quantization?.format == .u4)
        #expect(plan.quantization?.groupSize == 64)
        #expect(plan.validationParityFixture == "Qwen/Qwen3.5-0.8B")
        #expect(plan.validationPerformanceGate == SmeltPackagePerformanceGateID.textDecodePrefillStartup)
        #expect(
            plan.validationPerformanceProfile?.gate
                == SmeltPackagePerformanceGateID.textDecodePrefillStartup
        )
        #expect(plan.validationPerformanceProfile?.maxBounds == [
            .init(
                metric: SmeltPackagePerformanceMetricName.traceFirstTokenMS,
                max: SmeltPackagePerformanceBudget.textTraceFirstTokenMaxMS,
                unit: SmeltPackagePerformanceUnit.milliseconds
            ),
        ])
        #expect(plan.policy.inference?.maxTokens == 512)
        #expect(plan.policy.inference?.eosTokens == [248044, 248046])
        #expect(plan.policy.inference?.chatTemplate == SmeltPromptTemplateName.chatML)
        #expect(plan.policy.inference?.thinkingPolicy == .disabled)
        #expect(plan.policy.decode?.sampler.mode == .greedy)
        #expect(plan.policy.decode?.maxSteps == 512)
        try Self.assertLoadingCheckpointMap(projection.spec, expected: "hf.qwen", label: "qwen35_fast")
    }

    @Test func checkedProjectionParsesAllLogitsPrefillCompileToken() throws {
        let plain = try SmeltCAMCheckedPackageProjector.parsePrefillCompileConstraint(
            "metal batch 64",
            engine: "metal"
        )
        #expect(plain.batch == 64)
        #expect(plain.emitAllLogits == false)
        #expect(plain.verifyTokenCapacity == 0)

        let allLogits = try SmeltCAMCheckedPackageProjector.parsePrefillCompileConstraint(
            "metal all-logits batch 64",
            engine: "metal"
        )
        #expect(allLogits.batch == 64)
        #expect(allLogits.emitAllLogits == true)

        let transactional = try SmeltCAMCheckedPackageProjector.parsePrefillCompileConstraint(
            "metal verify-argmax batch 64 transaction 32",
            engine: "metal"
        )
        #expect(transactional.batch == 64)
        #expect(transactional.verifyArgmax == true)
        #expect(transactional.verifyTokenCapacity == 32)
    }

    @Test func projectsCheckedQwen3TTSCAMToDerivedAudioPackageSpec() throws {
        let artifactRoot = URL(fileURLWithPath: "/tmp/smelt-qwen3-tts-package", isDirectory: true)
        try? FileManager.default.removeItem(at: artifactRoot)
        defer { try? FileManager.default.removeItem(at: artifactRoot) }
        try Self.writeQwen3TTSPayloads(to: artifactRoot)

        let cam = registryModuleIR("qwen3_tts")
        #expect(cam.tensors.map {
            "\($0.selector.pattern)->\($0.target.block).\($0.target.selector)"
        }.sorted() == [
            "talker.model.text_embedding.weight->tts-frontend.text_embedding",
            "talker.text_projection.*->tts-frontend.text_projection.*",
            "talker.model.codec_embedding.weight->talker.codec_embedding",
            "talker.model.layers.*->talker.layers.*",
            "talker.model.norm.weight->talker.norm",
            "talker.codec_head.*->codec-head.*",
            "talker.code_predictor.*->mtp-head.*",
            "decoder.*->codec-decoder.*",
        ].sorted())
        let expected = try CAMModuleCompletionMatrix.module(fixture: "qwen3_tts.module.json")
        let expectedProjection = try #require(expected.packageProjection)
        let projection = try SmeltCAMCheckedPackageProjector.project(
            cam: cam,
            artifactRoot: artifactRoot.path
        )
        let plan = try SmeltPackageResolvedPlan.resolve(projection.spec)

        #expect(projection.packageProjectionID == expectedProjection.id)
        #expect(projection.packageProjectionVersion == expectedProjection.version)
        #expect(projection.buildCommandCovered == true)
        #expect(projection.camSemanticSHA256 == expected.semanticSHA256)
        #expect(projection.exportABISHA256 == expected.exportABISHA256)
        #expect(projection.descriptorVersion == SmeltCAMPackageDescriptor.currentDescriptorVersion)
        #expect(
            projection.descriptorGraphSignatureSHA256
                == expected.descriptorGraphSignatureSHA256
        )
        #expect(
            projection.projectedPackageSpecSHA256
                == Self.syntheticQwen3TTSProjectedPackageSpecSHA256
        )
        #expect(projection.projectedPackageSpecSHA256 != expectedProjection.projectedPackageSpecSHA256)
        #expect(projection.packageFiles == expectedProjection.packageFiles)
        #expect(projection.spec.runtime.architecture == SmeltRuntimeGraphPolicy.sidecarTextToCodecAudio.rawValue)
        #expect(projection.spec.loop == .qwen3TTS)
        #expect(plan.runtime.architecture == SmeltRuntimeGraphPolicy.sidecarTextToCodecAudio.rawValue)
        #expect(plan.runtime.routes.map(\.signature) == [
            "tts-frontend:native:none",
            "talker:compiled:baked-sidecar",
            "codec-head:native:none",
            "mtp-head:native:internal-sidecar",
            "codec-decoder:compiled:runtime-emit",
        ])
        #expect(plan.runtime.setupPhases == ["prefill:talker,codec-head"])
        #expect(plan.runtime.perStepPhases == [
            "mtp:mtp-head",
            "advance:talker,codec-head:feeds-next-step",
        ])
        #expect(plan.runtime.emission == "chunked:1:1:double:codec-decoder")
        #expect(plan.quantization?.format == .u4)
        #expect(plan.quantization?.groupSize == 128)
        #expect(plan.policy.mode == .sidecarTextToCodecAudio)
        #expect(plan.policy.inference?.maxTokens == 2_048)
        #expect(plan.policy.inference?.eosTokens == [2_150])
        #expect(plan.validationPerformanceGate == SmeltPackagePerformanceGateID.qwen3TTSTTFA)
        #expect(plan.validationPerformanceProfile?.gate == SmeltPackagePerformanceGateID.qwen3TTSTTFA)

        let tensorBlocks = Set(plan.tensors.map(\.block))
        #expect(tensorBlocks.isSuperset(of: [
            "codec-decoder",
            "codec-head",
            "mtp-head",
            "talker",
            "tts-frontend",
        ]))
        let sourceDTypes = Dictionary(uniqueKeysWithValues: plan.tensors.map {
            ($0.canonicalName, $0.sourceDType)
        })
        #expect(sourceDTypes["decoder.pre_conv.conv.weight"] == .f32)
        #expect(sourceDTypes["talker.codec_head.weight"] == .bf16)
        #expect(sourceDTypes["talker.code_predictor.lm_head.0.weight"] == .bf16)
        #expect(plan.sources.map { "\($0.id):\($0.kind.rawValue):\($0.locator)" } == [
            "qwen3-tts-source:local-directory:/tmp/smelt-qwen3-tts-package",
        ])
    }

    @Test func qwen3TTSManifestKindStringIsRejectedByCheckedProjection() throws {
        let artifactRoot = URL(fileURLWithPath: "/tmp/smelt-qwen3-tts-kind-string-drift", isDirectory: true)
        try? FileManager.default.removeItem(at: artifactRoot)
        defer { try? FileManager.default.removeItem(at: artifactRoot) }
        try Self.writeQwen3TTSPayloads(to: artifactRoot)
        let manifestURL = artifactRoot.appendingPathComponent("manifest.json")
        var raw = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as? [String: Any]
        )
        raw["kind"] = "not-runtime-authority"
        try JSONSerialization.data(withJSONObject: raw, options: [.sortedKeys])
            .write(to: manifestURL)

        do {
            _ = try SmeltCAMCheckedPackageProjector.project(
                cam: registryModuleIR("qwen3_tts"),
                artifactRoot: artifactRoot.path
            )
            Issue.record("expected removed Qwen3-TTS manifest kind field to fail")
        } catch {
            #expect(
                String(describing: error).contains("qwen3-tts manifest kind is no longer supported"),
                "\(error)"
            )
        }
    }

    @Test func rejectsCheckedQwen3TTSManifestQuantDriftAfterProfileSelection() throws {
        let artifactRoot = URL(fileURLWithPath: "/tmp/smelt-qwen3-tts-quant-drift", isDirectory: true)
        try? FileManager.default.removeItem(at: artifactRoot)
        defer { try? FileManager.default.removeItem(at: artifactRoot) }
        try Self.writeQwen3TTSPayloads(to: artifactRoot, u4GroupSize: 64)

        do {
            _ = try SmeltCAMCheckedPackageProjector.project(
                cam: registryModuleIR("qwen3_tts"),
                artifactRoot: artifactRoot.path
            )
            Issue.record("expected qwen-tts manifest quant drift to fail")
        } catch {
            #expect(Self.errorText(error).contains("u4 group drifted"))
        }
    }

    @Test func rejectsCheckedQwen3TTSMissingDecodeAfterProfileSelection() throws {
        let artifactRoot = URL(fileURLWithPath: "/tmp/smelt-qwen3-tts-decode-drift", isDirectory: true)
        try? FileManager.default.removeItem(at: artifactRoot)
        defer { try? FileManager.default.removeItem(at: artifactRoot) }
        try Self.writeQwen3TTSPayloads(to: artifactRoot, decode: nil)

        do {
            _ = try SmeltCAMCheckedPackageProjector.project(
                cam: registryModuleIR("qwen3_tts"),
                artifactRoot: artifactRoot.path
            )
            Issue.record("expected qwen-tts manifest decode drift to fail")
        } catch {
            #expect(Self.errorText(error).contains("decode"))
        }
    }

    @Test func rejectsCheckedQwen3TTSCheckpointMapDriftAfterProfileSelection() throws {
        let base = registryModuleIR("qwen3_tts")
        let cases: [(String, IRMutation?, (URL) throws -> Void, String)] = [
            (
                "map without manifest weights",
                { object in
                    try Self.mutateTensorMap(
                        &object,
                        pattern: "talker.text_projection.*",
                        label: "map without manifest weights"
                    ) { map in
                        try Self.setValue(
                            &map,
                            path: ["selector", "pattern"],
                            to: "frontend.*",
                            label: "map without manifest weights"
                        )
                    }
                },
                { root in
                    try Self.writeQwen3TTSPayloads(to: root)
                },
                "audio tensor map frontend.* matched no manifest weights"
            ),
            (
                "wrong target owner",
                { object in
                    try Self.mutateTensorMap(
                        &object,
                        pattern: "talker.model.text_embedding.weight",
                        label: "wrong target owner"
                    ) { map in
                        try Self.setValue(&map, path: ["target", "block"], to: "talker", label: "wrong target owner")
                        try Self.setValue(&map, path: ["owner"], to: "talker", label: "wrong target owner")
                    }
                },
                { root in
                    try Self.writeQwen3TTSPayloads(to: root)
                },
                "audio tts-frontend tensor evidence talker.model.text_embedding.weight maps to talker"
            ),
            (
                "codec head wrong target owner",
                { object in
                    try Self.mutateTensorMap(
                        &object,
                        pattern: "talker.codec_head.*",
                        label: "codec head wrong target owner"
                    ) { map in
                        map["owner"] = "talker"
                        map["target"] = ["block": "talker", "selector": "codec_head.*"]
                    }
                },
                { root in
                    try Self.writeQwen3TTSPayloads(to: root)
                },
                "audio tensor partitions do not cover every block"
            ),
            (
                "norm wrong target owner",
                { object in
                    try Self.mutateTensorMap(
                        &object,
                        pattern: "talker.model.norm.weight",
                        label: "norm wrong target owner"
                    ) { map in
                        try Self.setValue(&map, path: ["target", "block"], to: "codec-head", label: "norm wrong target owner")
                        try Self.setValue(&map, path: ["owner"], to: "codec-head", label: "norm wrong target owner")
                    }
                },
                { root in
                    try Self.writeQwen3TTSPayloads(to: root)
                },
                "audio talker tensor evidence talker.model.norm.weight maps to codec-head"
            ),
            (
                "missing exact frontend weight",
                nil,
                { root in
                    try Self.writeQwen3TTSPayloads(to: root) { manifest in
                        Self.qwen3TTSManifest(
                            manifest,
                            replacingWeights: manifest.weights.filter {
                                $0.name != "talker.model.text_embedding.weight"
                            }
                        )
                    }
                },
                "audio tensor map talker.model.text_embedding.weight matched no manifest weights"
            ),
            (
                "narrowed tensor evidence",
                { object in
                    try Self.setBlockRequirement(
                        &object,
                        block: "talker",
                        key: "tensor-evidence",
                        from: "talker.model.layers.*",
                        to: "talker.model.layers.*.self_attn.q_proj.weight",
                        label: "narrowed tensor evidence"
                    )
                },
                { root in
                    try Self.writeQwen3TTSPayloads(to: root)
                },
                "has no tensor evidence for talker"
            ),
            (
                "missing residual lm head",
                nil,
                { root in
                    try Self.writeQwen3TTSPayloads(to: root) { manifest in
                        Self.qwen3TTSManifest(
                            manifest,
                            replacingWeights: manifest.weights.filter {
                                $0.name != "talker.code_predictor.lm_head.14.weight"
                            }
                        )
                    }
                },
                "audio residual lm_head.14 missing from manifest"
            ),
            (
                "out of range residual lm head",
                nil,
                { root in
                    try Self.writeQwen3TTSPayloads(to: root) { manifest in
                        Self.qwen3TTSManifest(
                            manifest,
                            replacingWeights: manifest.weights.map { entry in
                                entry.name == "talker.code_predictor.lm_head.14.weight"
                                    ? Self.qwen3TTSRenamedEntry(
                                        entry,
                                        name: "talker.code_predictor.lm_head.15.weight"
                                    )
                                    : entry
                            }
                        )
                    }
                },
                "audio residual lm_head.14 missing from manifest"
            ),
            (
                "out of range residual codec embedding",
                nil,
                { root in
                    try Self.writeQwen3TTSPayloads(to: root) { manifest in
                        Self.qwen3TTSManifest(
                            manifest,
                            replacingWeights: manifest.weights.map { entry in
                                entry.name == "talker.code_predictor.model.codec_embedding.14.weight"
                                    ? Self.qwen3TTSRenamedEntry(
                                        entry,
                                        name: "talker.code_predictor.model.codec_embedding.15.weight"
                                    )
                                    : entry
                            }
                        )
                    }
                },
                "audio residual codec_embedding.14 missing from manifest"
            ),
            (
                "missing preserve policy",
                { object in
                    try Self.removeElement(&object, "quantization", label: "missing preserve policy", where: {
                        ($0["action"] as? String) == "preserve"
                            && Self.quantSelectorPattern($0) == "talker.text_projection.*"
                    })
                },
                { root in
                    try Self.writeQwen3TTSPayloads(to: root)
                },
                "audio non-u4 weight talker.text_projection.linear_fc1.bias is not covered by CAM preserve policy"
            ),
            (
                "u4 preserved weight",
                nil,
                { root in
                    try Self.writeQwen3TTSPayloads(to: root) { manifest in
                        Self.qwen3TTSManifest(
                            manifest,
                            replacingWeights: manifest.weights.map { entry in
                                entry.name == "talker.text_projection.linear_fc1.weight"
                                    ? Self.qwen3TTSEntry(entry, dtype: .u4, groupSize: 128)
                                    : entry
                            }
                        )
                    }
                },
                "audio preserved weight talker.text_projection.linear_fc1.weight is stored u4"
            ),
            (
                "all weights preserved without default u4",
                { object in
                    // `preserve "*"` as the parser lowers it: specificity 0 and
                    // declared-tensor resolution ("*" matches the audio maps'
                    // whole-block target selectors).
                    try Self.appendElement(&object, "quantization", label: "all weights preserved without default u4", [
                        "action": "preserve",
                        "priority": 0,
                        "resolution": "declared-tensor",
                        "selector": ["pattern": "*", "source": "weights"],
                        "source": "weights",
                    ])
                },
                { root in
                    try Self.writeQwen3TTSPayloads(to: root) { manifest in
                        Self.qwen3TTSManifest(
                            manifest,
                            replacingWeights: manifest.weights.map { entry in
                                entry.dtype == Qwen3TTSPackageBuilder.WeightDType.u4.rawValue
                                    ? Self.qwen3TTSEntry(entry, dtype: .f32)
                                    : entry
                            }
                        )
                    }
                },
                "audio manifest has no default affine-u4 weights"
            ),
            (
                "unmapped manifest weight",
                nil,
                { root in
                    try Self.writeQwen3TTSPayloads(to: root) { manifest in
                        let template = try #require(manifest.weights.first)
                        let orphan = Qwen3TTSManifest.Entry(
                            name: "orphan.weight",
                            offset: template.offset,
                            byteLength: template.byteLength,
                            shape: template.shape,
                            dtype: template.dtype,
                            groupSize: template.groupSize,
                            scaleOffset: template.scaleOffset,
                            scaleByteLength: template.scaleByteLength,
                            biasOffset: template.biasOffset,
                            biasByteLength: template.biasByteLength
                        )
                        return Self.qwen3TTSManifest(
                            manifest,
                            replacingWeights: manifest.weights + [orphan]
                        )
                    }
                },
                "audio manifest weight orphan.weight must match exactly one CAM tensor map"
            ),
        ]

        for (label, mutation, write, expectedError) in cases {
            let artifactRoot = URL(
                fileURLWithPath: "/tmp/smelt-qwen3-tts-checkpoint-\(label.replacingOccurrences(of: " ", with: "-"))",
                isDirectory: true
            )
            try? FileManager.default.removeItem(at: artifactRoot)
            defer { try? FileManager.default.removeItem(at: artifactRoot) }
            try write(artifactRoot)

            let cam = try mutation.map { try mutatedModuleIR(base, $0) } ?? base
            do {
                _ = try SmeltCAMCheckedPackageProjector.project(
                    cam: cam,
                    artifactRoot: artifactRoot.path
                )
                Issue.record("expected qwen-tts checkpoint map drift to fail for \(label)")
            } catch {
                #expect(Self.errorText(error).contains(expectedError), "\(label): \(Self.errorText(error))")
            }
        }
    }

    @Test func projectsSupportedAndRejectsUnsupportedCheckedQwen3TTSDrift() throws {
        let base = registryModuleIR("qwen3_tts")
        let baselineRoot = URL(fileURLWithPath: "/tmp/smelt-qwen3-tts-drift-baseline", isDirectory: true)
        try? FileManager.default.removeItem(at: baselineRoot)
        defer { try? FileManager.default.removeItem(at: baselineRoot) }
        try Self.writeQwen3TTSPayloads(to: baselineRoot)
        let baseline = try SmeltCAMCheckedPackageProjector.project(
            cam: base,
            artifactRoot: baselineRoot.path
        )

        let supportedCases: [(label: String, mutation: IRMutation, u4GroupSize: Int, projectionID: String)] = [
            (
                "source locator",
                { object in
                    try Self.setSourceLocator(
                        &object,
                        id: "weights",
                        to: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice-drift",
                        label: "source locator"
                    )
                },
                128,
                "streaming-text-to-24khz-audio-derived-manifest-affine-u4-g128-sidecars"
            ),
            (
                "tokenizer locator",
                { object in
                    try Self.setSourceLocator(
                        &object,
                        id: "tokenizer",
                        to: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/tokenizer-drift.json",
                        label: "tokenizer locator"
                    )
                },
                128,
                "streaming-text-to-24khz-audio-derived-manifest-affine-u4-g128-sidecars"
            ),
            (
                "quant group",
                { object in
                    try Self.mutateDefaultQuantRule(&object, label: "quant group") { rule in
                        try Self.setValue(&rule, path: ["storage", "groupSize"], to: 64, label: "quant group")
                    }
                },
                64,
                "streaming-text-to-24khz-audio-derived-manifest-affine-u4-g64-sidecars"
            ),
        ]
        let rejectedCases: [(String, IRMutation)] = [
            (
                "optional audio output",
                { object in
                    try Self.mutateElement(&object, "exports", label: "optional audio output", where: { _ in true }) { export in
                        try Self.mutateElement(&export, "outputs", label: "optional audio output", where: {
                            ($0["name"] as? String) == "audio"
                        }) { output in
                            output["optional"] = true
                        }
                    }
                }
            ),
            (
                "unused graph node",
                { object in
                    try Self.addUnusedGraphNode(&object, consuming: "text", label: "unused graph node")
                }
            ),
            (
                "extra tensor binding",
                { object in
                    try Self.appendTensorMap(
                        &object,
                        pattern: "talker.extra",
                        targetBlock: "talker",
                        targetSelector: "extra",
                        label: "extra tensor binding"
                    )
                }
            ),
            (
                "flow order",
                { object in
                    try Self.mutateFlowPhase(&object, label: "flow order", where: {
                        ($0["label"] as? String) == "generate"
                    }) { phase in
                        phase["calls"] = Self.nodeCalls(["talker", "mtp-head", "codec-head"])
                    }
                }
            ),
            (
                "block max frames",
                { object in
                    // `require max-frames 2048` appears in BOTH the talker and
                    // mtp-head blocks; the whole-source text edit drifted both.
                    try Self.setBlockRequirement(&object, block: "talker", key: "max-frames", to: "1024", label: "block max frames")
                    try Self.setBlockRequirement(&object, block: "mtp-head", key: "max-frames", to: "1024", label: "block max frames")
                }
            ),
            (
                "codec source dtype",
                { object in
                    try Self.setBlockRequirement(
                        &object,
                        block: "codec-decoder",
                        key: "source-dtype",
                        from: "f32",
                        to: "bf16",
                        label: "codec source dtype"
                    )
                }
            ),
            (
                "residual codebooks",
                { object in
                    try Self.setBlockRequirement(&object, block: "mtp-head", key: "residual-codebooks", to: "8", label: "residual codebooks")
                }
            ),
            (
                "coordinated residual codebooks",
                { object in
                    try Self.setBlockRequirement(&object, block: "mtp-head", key: "codebooks", to: "8", label: "coordinated residual codebooks")
                    try Self.setBlockRequirement(&object, block: "mtp-head", key: "residual-codebooks", to: "8", label: "coordinated residual codebooks")
                }
            ),
            (
                "feedback edge",
                { object in
                    try Self.setFeedbackEdgeSourceToGraphValue(&object, named: "codec_token", label: "feedback edge")
                }
            ),
        ]

        for testCase in supportedCases {
            let artifactRoot = URL(
                fileURLWithPath: "/tmp/smelt-qwen3-tts-\(testCase.label.replacingOccurrences(of: " ", with: "-"))",
                isDirectory: true
            )
            try? FileManager.default.removeItem(at: artifactRoot)
            defer { try? FileManager.default.removeItem(at: artifactRoot) }
            try Self.writeQwen3TTSPayloads(to: artifactRoot, u4GroupSize: testCase.u4GroupSize)

            let cam = try mutatedModuleIR(base, testCase.mutation)
            let projection = try SmeltCAMCheckedPackageProjector.project(
                cam: cam,
                artifactRoot: artifactRoot.path
            )
            let plan = try SmeltPackageResolvedPlan.resolve(projection.spec)
            #expect(
                projection.camSemanticSHA256 != baseline.camSemanticSHA256,
                "\(testCase.label): CAM semantic hash did not change"
            )
            #expect(
                projection.packageProjectionID == testCase.projectionID,
                "\(testCase.label): projection id drifted"
            )
            #expect(plan.quantization?.groupSize == testCase.u4GroupSize)
            #expect(projection.packageFiles == baseline.packageFiles)
        }

        for (label, mutation) in rejectedCases {
            let artifactRoot = URL(
                fileURLWithPath: "/tmp/smelt-qwen3-tts-reject-\(label.replacingOccurrences(of: " ", with: "-"))",
                isDirectory: true
            )
            try? FileManager.default.removeItem(at: artifactRoot)
            defer { try? FileManager.default.removeItem(at: artifactRoot) }
            try Self.writeQwen3TTSPayloads(to: artifactRoot)

            let cam = try mutatedModuleIR(base, mutation)
            try Self.expectProjectionRejectedWithoutProfileMiss(
                cam,
                label: label,
                artifactRoot: artifactRoot.path
            )
        }
    }

    @Test func checkedProjectedAndBuildCoveredCompletionModulesAreExplicit() {
        #expect(CAMModuleCompletionMatrix.checkedProjectedFixtureNames == [
            "qwen35_text.module.json",
            "qwen35_fast.module.json",
            "qwen3_tts.module.json",
        ])
        #expect(CAMModuleCompletionMatrix.buildCommandCoveredFixtureNames == [
            "qwen35_text.module.json",
            "qwen35_fast.module.json",
            "qwen3_tts.module.json",
        ])
    }

    @Test func featureAdmissionAcceptsCheckedCompletionModules() throws {
        for name in CAMModuleCompletionMatrix.checkedProjectedFixtureNames {
            let descriptor = try SmeltCAMPackageDescriptor(
                from: registryModuleIR(name)
            )
            let admission = SmeltCAMFeatureAdmission(descriptor: descriptor)
            #expect(
                admission.unsupportedFeatureSet.isEmpty,
                "\(name): \(admission.unsupportedDiagnostic)"
            )
        }
    }

    @Test func featureAdmissionRejectsDS4UsingStructuralCodes() throws {
        let descriptor = try SmeltCAMPackageDescriptor(
            from: registryModuleIR("ds4_heavy_quant")
        )
        let admission = SmeltCAMFeatureAdmission(descriptor: descriptor)

        #expect(admission.schema == SmeltCAMFeatureAdmission.currentSchema)
        #expect(admission.stage == SmeltCAMFeatureAdmission.preBridgeStage)
        #expect(admission.unsupportedFeatureSet == [
            "compile.generated-kernels",
            "compile.memory-bound",
            "gate.quant-quality",
            "quant.calibration.gptq",
            "quant.storage.gptq",
            "transformer.moe.expert",
            "transformer.moe.router",
            "transformer.rope.yarn",
        ])
        let repeated = SmeltCAMFeatureAdmission(descriptor: descriptor)
        #expect(admission.requiredFeatureSet.count == 8)
        #expect(admission.requiredObligations.count == 10)
        #expect(admission.requiredObligationIDs == repeated.requiredObligationIDs)
        #expect(admission.unsupportedObligationIDs == admission.requiredObligationIDs)
        #expect(Set(admission.requiredObligationIDs).count == admission.requiredObligationIDs.count)
        for id in admission.requiredObligationIDs {
            #expect(Self.isLowercaseSHA256(id), Comment(rawValue: id))
        }

        let gptqStorage = admission.requiredObligations.filter {
            $0.code == "quant.storage.gptq"
        }
        #expect(gptqStorage.count == 2)
        #expect(gptqStorage.map { $0.parameters["action"] ?? "" }.sorted() == [
            "default",
            "store",
        ])
        #expect(gptqStorage.map { $0.parameters["group_size"] ?? "" } == [
            "128",
            "128",
        ])
        #expect(gptqStorage.contains {
            $0.parameters["pattern"] == "model.layers.*.mlp.experts.*"
        })

        let yarn = try #require(admission.requiredObligations.first {
            $0.code == "transformer.rope.yarn"
        })
        #expect(yarn.parameters["theta"] == "1000000")
        #expect(yarn.parameters["q_heads"] == "128")
        #expect(yarn.parameters["kv_heads"] == "8")

        let router = try #require(admission.requiredObligations.first {
            $0.code == "transformer.moe.router"
        })
        #expect(router.parameters["top_k"] == "8")
        #expect(router.parameters["experts"] == "256")

        let expert = try #require(admission.requiredObligations.first {
            $0.code == "transformer.moe.expert"
        })
        #expect(expert.parameters["ffn_dim"] == "18432")
        #expect(expert.parameters["activation"] == "swiglu")
        #expect(expert.parameters["experts"] == "256")

        let calibration = try #require(admission.requiredObligations.first {
            $0.code == "quant.calibration.gptq"
        })
        #expect(calibration.parameters["corpus_source"] == "calibration_prompts")
        #expect(Self.isLowercaseSHA256(calibration.parameters["corpus_path_sha256"] ?? ""))
        #expect(calibration.parameters["max_tokens"] == "4096")
        #expect(calibration.parameters["captures"] == "trunk.attention,trunk.experts")
        #expect(calibration.parameters["layers_per_pass"] == "1")

        let quantQuality = admission.requiredObligations.filter {
            $0.code == "gate.quant-quality"
        }
        #expect(quantQuality.count == 2)
        #expect(Set(quantQuality.map { $0.parameters["subject"] ?? "" }) == [
            "calibration.gptq.rank",
            "perplexity.delta",
        ])

        let diagnostic = admission.unsupportedDiagnostic
        for fragment in [
            "transformer.rope.yarn",
            "theta=1000000",
            "uses Yarn RoPE",
            "transformer.moe.router",
            "top_k=8",
            "transformer.moe.expert",
            "ffn_dim=18432",
            "uses MoE router",
            "uses MoE expert",
            "quant.storage.gptq",
            "pattern=model.layers.*.mlp.experts.*",
            "group_size=128",
            "uses GPTQ tensor storage",
            "quant.calibration.gptq",
            "corpus_path_sha256=",
            "declares GPTQ calibration artifacts",
            "compile.generated-kernels",
            "value=auto",
            "requires generated kernels",
            "compile.memory-bound",
            "value=peak <= 48 GiB",
            "declares peak memory requirement",
            "gate.quant-quality",
            "declares quant-quality gate subjects",
        ] {
            #expect(diagnostic.contains(fragment), Comment(rawValue: fragment))
        }
        for banned in [
            "ds4",
            "DS4",
            "deepseek",
            "qwen",
            "moduleID",
            "modelName",
            "family",
            "semantic hash",
            "export ABI hash",
            "descriptor graph signature",
            ".cam",
            "deepseek-ai",
            "calibration/ds4-prompts.jsonl",
        ] {
            #expect(!diagnostic.contains(banned), Comment(rawValue: diagnostic))
        }
    }

    @Test func featureAdmissionConsumptionUsesCanonicalObligationIDsOnly() throws {
        let descriptor = try SmeltCAMPackageDescriptor(
            from: registryModuleIR("ds4_heavy_quant")
        )
        let admission = SmeltCAMFeatureAdmission(descriptor: descriptor)
        let defaultGPTQ = try #require(admission.requiredObligations.first {
            $0.code == "quant.storage.gptq" && $0.parameters["action"] == "default"
        })

        let codeOnly = SmeltCAMFeatureAdmission(
            descriptor: descriptor,
            consumedFeatureSet: ["quant.storage.gptq"]
        )
        #expect(codeOnly.unsupportedFeatures.filter { $0.code == "quant.storage.gptq" }.count == 2)
        #expect(codeOnly.unsupportedFeatureSet.contains("quant.storage.gptq"))

        let consumedOne = SmeltCAMFeatureAdmission(
            descriptor: descriptor,
            consumedFeatureSet: ["quant.storage.gptq"],
            consumedObligationIDs: [defaultGPTQ.canonicalID]
        )
        #expect(consumedOne.consumedFeatureSet == ["quant.storage.gptq"])
        #expect(consumedOne.consumedObligationIDs == [defaultGPTQ.canonicalID])
        #expect(!consumedOne.unsupportedObligationIDs.contains(defaultGPTQ.canonicalID))
        #expect(consumedOne.unsupportedFeatures.filter { $0.code == "quant.storage.gptq" }.count == 1)
        #expect(consumedOne.unsupportedFeatureSet.contains("quant.storage.gptq"))
    }

    @Test func featureAdmissionTypedObligationIDsTrackDS4ParameterDrift() throws {
        let base = registryModuleIR("ds4_heavy_quant")
        let baseline = try Self.featureAdmission(from: base)
        let expertPattern = "model.layers.*.mlp.experts.*"
        let expertPatternShard = "model.layers.*.mlp.experts.0.*"

        func changed(
            _ baseline: ObligationIdentity,
            _ drifted: ObligationIdentity
        ) -> ObligationDriftExpectation {
            .init(baseline: baseline, drifted: drifted)
        }

        func rope(
            qHeads: String = "128",
            kvHeads: String = "8",
            headDim: String = "128",
            theta: String = "1000000"
        ) -> ObligationIdentity {
            .init(
                "transformer.rope.yarn",
                parameters: [
                    "q_heads": qHeads,
                    "kv_heads": kvHeads,
                    "head_dim": headDim,
                    "theta": theta,
                ]
            )
        }

        func router(topK: String = "8", experts: String = "256") -> ObligationIdentity {
            .init(
                "transformer.moe.router",
                parameters: ["top_k": topK, "experts": experts]
            )
        }

        func expert(
            dim: String = "18432",
            activation: String = "swiglu",
            experts: String = "256"
        ) -> ObligationIdentity {
            .init(
                "transformer.moe.expert",
                parameters: [
                    "ffn_dim": dim,
                    "activation": activation,
                    "experts": experts,
                ]
            )
        }

        func gptqStorage(
            action: String,
            pattern: String,
            groupSize: String? = nil
        ) -> ObligationIdentity {
            var parameters = ["action": action, "pattern": pattern]
            if let groupSize {
                parameters["group_size"] = groupSize
            }
            return .init("quant.storage.gptq", parameters: parameters)
        }

        func expertGPTQStorage(pattern: String = expertPattern) -> ObligationIdentity {
            gptqStorage(action: "store", pattern: pattern)
        }

        func gptqCalibration(_ parameters: [String: String] = [:]) -> ObligationIdentity {
            var merged = [
                "action": "store",
                "pattern": expertPattern,
            ]
            for (key, value) in parameters {
                merged[key] = value
            }
            return .init("quant.calibration.gptq", parameters: merged)
        }

        func quantQualityGate(subject: String, value: String? = nil) -> ObligationIdentity {
            var parameters = ["subject": subject]
            if let value {
                parameters["value"] = value
            }
            return .init("gate.quant-quality", parameters: parameters)
        }

        func compileRequirement(_ code: String, key: String, value: String) -> ObligationIdentity {
            .init(code, parameters: ["key": key, "value": value])
        }

        let cases: [ObligationDriftCase] = [
            .init(
                label: "rope q-heads",
                mutation: { object in
                    try Self.mutateTransformer(&object, label: "rope q-heads") { transformer in
                        try Self.setValue(&transformer, path: ["attention", "qHeads"], to: 127, label: "rope q-heads")
                    }
                },
                changed: [changed(rope(), rope(qHeads: "127"))]
            ),
            .init(
                label: "rope kv-heads",
                mutation: { object in
                    try Self.mutateTransformer(&object, label: "rope kv-heads") { transformer in
                        try Self.setValue(&transformer, path: ["attention", "kvHeads"], to: 7, label: "rope kv-heads")
                    }
                },
                changed: [changed(rope(), rope(kvHeads: "7"))]
            ),
            .init(
                label: "rope head-dim",
                mutation: { object in
                    try Self.mutateTransformer(&object, label: "rope head-dim") { transformer in
                        try Self.setValue(&transformer, path: ["attention", "headDim"], to: 64, label: "rope head-dim")
                    }
                },
                changed: [changed(rope(), rope(headDim: "64"))]
            ),
            .init(
                label: "rope theta",
                mutation: { object in
                    try Self.mutateTransformer(&object, label: "rope theta") { transformer in
                        try Self.setValue(&transformer, path: ["attention", "rope", "theta"], to: 999999, label: "rope theta")
                    }
                },
                changed: [changed(rope(), rope(theta: "999999"))]
            ),
            .init(
                label: "router top-k",
                mutation: { object in
                    try Self.mutateTransformer(&object, label: "router top-k") { transformer in
                        try Self.setValue(&transformer, path: ["router", "topK"], to: 4, label: "router top-k")
                    }
                },
                changed: [changed(router(), router(topK: "4"))]
            ),
            .init(
                label: "router expert count",
                mutation: { object in
                    try Self.mutateTransformer(&object, label: "router expert count") { transformer in
                        try Self.setValue(&transformer, path: ["router", "experts"], to: 128, label: "router expert count")
                    }
                },
                changed: [
                    changed(router(), router(experts: "128")),
                    changed(expert(), expert(experts: "128")),
                ]
            ),
            .init(
                label: "expert dim",
                mutation: { object in
                    try Self.mutateTransformer(&object, label: "expert dim") { transformer in
                        try Self.setValue(&transformer, path: ["expert", "ffn", "dim"], to: 18433, label: "expert dim")
                    }
                },
                changed: [changed(expert(), expert(dim: "18433"))]
            ),
            .init(
                label: "expert activation",
                mutation: { object in
                    try Self.mutateTransformer(&object, label: "expert activation") { transformer in
                        try Self.setValue(&transformer, path: ["expert", "ffn", "activation"], to: "geglu", label: "expert activation")
                    }
                },
                changed: [changed(expert(), expert(activation: "geglu"))]
            ),
            .init(
                label: "default gptq group",
                mutation: { object in
                    try Self.mutateDefaultQuantRule(&object, label: "default gptq group") { rule in
                        try Self.setValue(&rule, path: ["storage", "groupSize"], to: 64, label: "default gptq group")
                    }
                },
                changed: [
                    changed(
                        gptqStorage(action: "default", pattern: "*", groupSize: "128"),
                        gptqStorage(action: "default", pattern: "*", groupSize: "64")
                    ),
                ],
                stable: [expertGPTQStorage()]
            ),
            .init(
                label: "expert gptq tensor pattern",
                mutation: { object in
                    // The old text edit hit both occurrences of the pattern:
                    // the tensor map AND the gptq store rule selector.
                    try Self.mutateTensorMap(&object, pattern: expertPattern, label: "expert gptq tensor pattern") { map in
                        try Self.setValue(&map, path: ["selector", "pattern"], to: expertPatternShard, label: "expert gptq tensor pattern")
                    }
                    try Self.setQuantRulePattern(
                        &object,
                        action: "store",
                        from: expertPattern,
                        to: expertPatternShard,
                        label: "expert gptq tensor pattern"
                    )
                },
                changed: [
                    changed(expertGPTQStorage(), expertGPTQStorage(pattern: expertPatternShard)),
                    changed(gptqCalibration(), gptqCalibration(["pattern": expertPatternShard])),
                ]
            ),
            .init(
                label: "calibration corpus source",
                mutation: { object in
                    // Both occurrences: the file source declaration and the
                    // calibration corpus reference.
                    try Self.mutateSource(&object, id: "calibration_prompts", label: "calibration corpus source") { source in
                        try Self.setValue(&source, path: ["id"], to: "calibration_alt_prompts", label: "calibration corpus source")
                    }
                    try Self.mutateGPTQCalibration(&object, label: "calibration corpus source") { calibration in
                        try Self.setValue(&calibration, path: ["corpus", "source"], to: "calibration_alt_prompts", label: "calibration corpus source")
                    }
                },
                changed: [
                    changed(
                        gptqCalibration(["corpus_source": "calibration_prompts"]),
                        gptqCalibration(["corpus_source": "calibration_alt_prompts"])
                    ),
                ]
            ),
            .init(
                label: "calibration corpus path",
                mutation: { object in
                    // Both occurrences: the file source locator and the
                    // calibration corpus path.
                    try Self.setSourceLocator(
                        &object,
                        id: "calibration_prompts",
                        to: "calibration/ds4-alt-prompts.jsonl",
                        label: "calibration corpus path"
                    )
                    try Self.mutateGPTQCalibration(&object, label: "calibration corpus path") { calibration in
                        try Self.setValue(&calibration, path: ["corpus", "path"], to: "calibration/ds4-alt-prompts.jsonl", label: "calibration corpus path")
                    }
                },
                changed: [
                    changed(gptqCalibration(), gptqCalibration()),
                ]
            ),
            .init(
                label: "calibration tokens",
                mutation: { object in
                    try Self.mutateGPTQCalibration(&object, label: "calibration tokens") { calibration in
                        try Self.setValue(&calibration, path: ["corpus", "maxTokens"], to: 2048, label: "calibration tokens")
                    }
                },
                changed: [
                    changed(
                        gptqCalibration(["max_tokens": "4096"]),
                        gptqCalibration(["max_tokens": "2048"])
                    ),
                ]
            ),
            .init(
                label: "calibration captures",
                mutation: { object in
                    try Self.mutateGPTQCalibration(&object, label: "calibration captures") { calibration in
                        try Self.setValue(&calibration, path: ["captures"], to: ["trunk.attention"], label: "calibration captures")
                    }
                },
                changed: [
                    changed(
                        gptqCalibration(["captures": "trunk.attention,trunk.experts"]),
                        gptqCalibration(["captures": "trunk.attention"])
                    ),
                ]
            ),
            .init(
                label: "calibration layers",
                mutation: { object in
                    try Self.mutateGPTQCalibration(&object, label: "calibration layers") { calibration in
                        try Self.setValue(&calibration, path: ["layersPerPass"], to: 2, label: "calibration layers")
                    }
                },
                changed: [
                    changed(
                        gptqCalibration(["layers_per_pass": "1"]),
                        gptqCalibration(["layers_per_pass": "2"])
                    ),
                ]
            ),
            .init(
                label: "calibration cosine",
                mutation: { object in
                    try Self.mutateGPTQCalibration(&object, label: "calibration cosine") { calibration in
                        try Self.mutateElement(&calibration, "requirements", label: "calibration cosine", where: {
                            ($0["subject"] as? String) == "cosine"
                        }) { requirement in
                            requirement["value"] = "0.990"
                        }
                    }
                },
                changed: [
                    changed(gptqCalibration(), gptqCalibration()),
                ]
            ),
            .init(
                label: "quant quality rank",
                mutation: { object in
                    try Self.setGateRequirementValue(
                        &object,
                        gate: "quant_quality",
                        subject: "calibration.gptq.rank",
                        to: "64",
                        label: "quant quality rank"
                    )
                },
                changed: [
                    changed(
                        quantQualityGate(subject: "calibration.gptq.rank", value: "128"),
                        quantQualityGate(subject: "calibration.gptq.rank", value: "64")
                    ),
                ],
                stable: [
                    quantQualityGate(subject: "perplexity.delta"),
                ]
            ),
            .init(
                label: "quant quality perplexity",
                mutation: { object in
                    try Self.setGateRequirementValue(
                        &object,
                        gate: "quant_quality",
                        subject: "perplexity.delta",
                        to: "0.10",
                        label: "quant quality perplexity"
                    )
                },
                changed: [
                    changed(
                        quantQualityGate(subject: "perplexity.delta", value: "0.05"),
                        quantQualityGate(subject: "perplexity.delta", value: "0.10")
                    ),
                ],
                stable: [
                    quantQualityGate(subject: "calibration.gptq.rank"),
                ]
            ),
            .init(
                label: "generated kernels mode",
                mutation: { object in
                    try Self.setCompileValue(&object, key: "generated-kernels", to: "manual", label: "generated kernels mode")
                },
                changed: [
                    changed(
                        compileRequirement(
                            "compile.generated-kernels",
                            key: "generated-kernels",
                            value: "auto"
                        ),
                        compileRequirement(
                            "compile.generated-kernels",
                            key: "generated-kernels",
                            value: "manual"
                        )
                    ),
                ]
            ),
            .init(
                label: "memory bound",
                mutation: { object in
                    try Self.setCompileValue(&object, key: "memory", to: "peak <= 47 GiB", label: "memory bound")
                },
                changed: [
                    changed(
                        compileRequirement(
                            "compile.memory-bound",
                            key: "memory",
                            value: "peak <= 48 GiB"
                        ),
                        compileRequirement(
                            "compile.memory-bound",
                            key: "memory",
                            value: "peak <= 47 GiB"
                        )
                    ),
                ]
            ),
        ]

        for testCase in cases {
            let drifted = try Self.featureAdmission(
                from: mutatedModuleIR(base, testCase.mutation)
            )
            #expect(
                drifted.requiredFeatureSet == baseline.requiredFeatureSet,
                "\(testCase.label): \(drifted.requiredFeatureSet)"
            )
            #expect(
                drifted.unsupportedFeatureSet == baseline.unsupportedFeatureSet,
                "\(testCase.label): \(drifted.unsupportedFeatureSet)"
            )
            #expect(
                drifted.requiredObligations.count == baseline.requiredObligations.count,
                "\(testCase.label): \(drifted.requiredObligations.map(\.checkSummary))"
            )
            #expect(
                Self.obligationCountsByCode(in: drifted) == Self.obligationCountsByCode(in: baseline),
                "\(testCase.label): \(Self.obligationCountsByCode(in: drifted))"
            )
            #expect(
                drifted.requiredObligationIDs != baseline.requiredObligationIDs,
                "\(testCase.label): required obligation IDs did not move"
            )

            for expectation in testCase.changed {
                let baselineObligation = try Self.obligation(
                    expectation.baseline,
                    in: baseline,
                    label: "\(testCase.label) baseline"
                )
                let driftedObligation = try Self.obligation(
                    expectation.drifted,
                    in: drifted,
                    label: "\(testCase.label) drifted"
                )
                #expect(
                    driftedObligation.canonicalID != baselineObligation.canonicalID,
                    "\(testCase.label): \(expectation.baseline.summary)"
                )
            }

            for identity in testCase.stable {
                let baselineObligation = try Self.obligation(
                    identity,
                    in: baseline,
                    label: "\(testCase.label) stable baseline"
                )
                let driftedObligation = try Self.obligation(
                    identity,
                    in: drifted,
                    label: "\(testCase.label) stable drifted"
                )
                #expect(
                    driftedObligation.canonicalID == baselineObligation.canonicalID,
                    "\(testCase.label): stable obligation moved \(identity.summary)"
                )
            }
        }
    }

    @Test func completionModulesWithoutCheckedProjectionRejectAtProfileSelection() throws {
        for name in CAMModuleCompletionMatrix.unprojectedFixtureNames {
            let cam = registryModuleIR(name)
            try Self.expectNoCheckedPackageProjectionProfile(cam, label: name)
        }
    }

    @Test func rejectsDS4AfterDescriptorKeepsHeavyQuantContracts() throws {
        let cam = registryModuleIR("ds4_heavy_quant")
        let descriptor = try SmeltCAMPackageDescriptor(from: cam)

        #expect(descriptor.moduleID == "ds4_heavy_quant")
        let defaultRule = try #require(descriptor.quantization.first { $0.action == "default" })
        #expect(defaultRule.storage?.storageFormat == "gptq")
        #expect(defaultRule.storage?.groupSize == 128)
        #expect(defaultRule.calibration == nil)

        let embeddingStore = try #require(
            descriptor.quantization.first { $0.tensorPattern.pattern == "model.embed_tokens.*" }
        )
        #expect(embeddingStore.action == "store")
        #expect(embeddingStore.storage?.storageFormat == "turbo-quant-h")
        #expect(embeddingStore.storage?.groupSize == 128)
        #expect(embeddingStore.calibration == nil)

        let expertStore = try #require(
            descriptor.quantization.first { $0.tensorPattern.pattern == "model.layers.*.mlp.experts.*" }
        )
        #expect(expertStore.action == "store")
        #expect(expertStore.storage?.storageFormat == "gptq")
        #expect(expertStore.storage?.groupSize == 128)
        #expect(expertStore.calibration?.method == "gptq")
        #expect(expertStore.calibration?.corpus.sourceID == "calibration_prompts")
        #expect(expertStore.calibration?.captures == ["trunk.attention", "trunk.experts"])
        #expect(expertStore.calibration?.layersPerPass == 1)

        let quantQuality = try #require(descriptor.gateContracts.first { $0.gateID == "quant_quality" })
        #expect(quantQuality.requirements.map { $0.subject }.sorted() == [
            "calibration.gptq.rank",
            "perplexity.delta",
        ])

        try Self.expectUnsupportedProjectionFeatures(cam, label: "heavy quant")
    }

    @Test func heavyQuantProjectionDiagnosticIsStructuralNotIdentityBased() throws {
        let label = "renamed heavy quant"
        let cam = try mutatedModuleIR(registryModuleIR("ds4_heavy_quant")) { object in
            object["module"] = ["id": "heavy_quant_probe"]
            try Self.setSourceLocator(&object, id: "weights", to: "example/heavy-quant-probe", label: label)
            try Self.setSourceLocator(
                &object,
                id: "calibration_prompts",
                to: "calibration/heavy-prompts.jsonl",
                label: label
            )
            try Self.mutateGPTQCalibration(&object, label: label) { calibration in
                try Self.setValue(&calibration, path: ["corpus", "path"], to: "calibration/heavy-prompts.jsonl", label: label)
            }
        }
        let error = try Self.expectUnsupportedProjectionFeatures(cam, label: label)

        for banned in [
            "ds4",
            "DS4",
            "deepseek",
            "qwen",
            "moduleID",
            "modelName",
            "family",
            "semantic hash",
            "export ABI hash",
            "descriptor graph signature",
            ".cam",
        ] {
            #expect(!error.contains(banned), Comment(rawValue: error))
        }
    }

    @Test func featureAdmissionRejectsUnknownCompileRequirementBeforeDerivedProjection() throws {
        let cam = try mutatedModuleIR(registryModuleIR("qwen35_text")) { object in
            try Self.appendCompileConstraint(
                &object,
                key: "tensor-teleport",
                value: "enabled",
                label: "unknown compile requirement"
            )
        }
        let descriptor = try SmeltCAMPackageDescriptor(from: cam)
        let admission = SmeltCAMFeatureAdmission(descriptor: descriptor)

        #expect(admission.unsupportedFeatureSet == ["compile.unclassified"])
        #expect(admission.unsupportedDiagnostic.contains("compile.unclassified"))
        #expect(admission.unsupportedDiagnostic.contains("tensor-teleport"))

        try Self.expectUnsupportedProjectionFeatures(
            cam,
            label: "unknown compile requirement",
            requiredFragments: [
                "unsupported CAM package projection features",
                "compile.unclassified",
                "declares unclassified compile requirement",
                "tensor-teleport",
            ]
        )
    }

    @Test func featureAdmissionRejectsSingletonUnsupportedStructuralFeaturesBeforeProjection() throws {
        let base = registryModuleIR("qwen35_text")
        let cases: [(label: String, mutation: IRMutation, features: [String])] = [
            (
                label: "yarn rope",
                mutation: { object in
                    try Self.mutateTransformer(&object, label: "yarn rope") { transformer in
                        try Self.setValue(&transformer, path: ["attention", "rope", "kind"], to: "yarn", label: "yarn rope")
                    }
                },
                features: ["transformer.rope.yarn"]
            ),
            (
                label: "moe router",
                mutation: { object in
                    // `router top-k 2 experts 4` added to the trunk transformer.
                    try Self.mutateTransformer(&object, label: "moe router") { transformer in
                        transformer["router"] = ["topK": 2, "experts": 4]
                    }
                },
                features: ["transformer.moe.router"]
            ),
            (
                label: "moe expert",
                mutation: { object in
                    // `expert ffn dim 6144 activation swiglu` added to the trunk.
                    try Self.mutateTransformer(&object, label: "moe expert") { transformer in
                        transformer["expert"] = ["ffn": ["dim": 6144, "activation": "swiglu"] as [String: Any]]
                    }
                },
                features: ["transformer.moe.expert"]
            ),
            (
                label: "gptq storage",
                mutation: { object in
                    // `default affine-u4 group 64` -> `default gptq group 128`.
                    try Self.mutateDefaultQuantRule(&object, label: "gptq storage") { rule in
                        try Self.setValue(&rule, path: ["storage"], to: ["format": "gptq", "groupSize": 128] as [String: Any], label: "gptq storage")
                    }
                },
                features: ["quant.storage.gptq"]
            ),
            (
                label: "generated kernels",
                mutation: { object in
                    try Self.appendCompileConstraint(&object, key: "generated-kernels", value: "auto", label: "generated kernels")
                },
                features: ["compile.generated-kernels"]
            ),
            (
                label: "memory bound",
                mutation: { object in
                    try Self.appendCompileConstraint(&object, key: "memory", value: "peak <= 48 GiB", label: "memory bound")
                },
                features: ["compile.memory-bound"]
            ),
            (
                label: "quant quality",
                mutation: { object in
                    // `gate quant_quality: require perplexity delta <= 0.05`.
                    try Self.appendElement(&object, "gates", label: "quant quality", [
                        "evidence": [[String: Any]](),
                        "id": "quant_quality",
                        "measurements": [[String: Any]](),
                        "requirements": [
                            ["relation": "<=", "subject": "perplexity.delta", "value": "0.05"],
                        ],
                    ])
                },
                features: ["gate.quant-quality"]
            ),
        ]

        for testCase in cases {
            let cam = try mutatedModuleIR(base, testCase.mutation)
            let descriptor = try SmeltCAMPackageDescriptor(from: cam)
            let admission = SmeltCAMFeatureAdmission(descriptor: descriptor)

            #expect(
                Set(admission.unsupportedFeatureSet) == Set(testCase.features),
                "\(testCase.label): \(admission.unsupportedDiagnostic)"
            )
            try Self.expectUnsupportedProjectionFeatures(
                cam,
                label: testCase.label,
                requiredFragments: ["unsupported CAM package projection features"] + testCase.features
            )
        }
    }

    @Test func featureAdmissionCanIsolateGPTQCalibrationAfterStorageObligationIsConsumed() throws {
        let label = "isolate gptq calibration"
        let cam = try mutatedModuleIR(registryModuleIR("qwen35_text")) { object in
            // `source calibration_prompts: file "calibration/qwen-prompts.jsonl"`.
            try Self.appendElement(&object, "sources", label: label, [
                "id": "calibration_prompts",
                "kind": "file",
                "locator": "calibration/qwen-prompts.jsonl",
            ])
            // `store "model.layers.*.mlp.down_proj.*" as gptq group 128` plus
            // the `calibrate gptq:` block the parser attaches to non-default
            // gptq store rules (priority = pattern specificity 28; the pattern
            // matches no declared tensor map, so resolution stays deferred).
            try Self.appendElement(&object, "quantization", label: label, [
                "action": "store",
                "calibration": [
                    "captures": ["trunk.attention"],
                    "corpus": [
                        "maxTokens": 4096,
                        "path": "calibration/qwen-prompts.jsonl",
                        "source": "calibration_prompts",
                    ] as [String: Any],
                    "layersPerPass": 1,
                    "method": "gptq",
                    "requirements": [
                        ["relation": ">=", "subject": "cosine", "value": "0.995"],
                    ],
                ] as [String: Any],
                "priority": 28,
                "resolution": "source-deferred",
                "selector": ["pattern": "model.layers.*.mlp.down_proj.*", "source": "weights"],
                "source": "weights",
                "storage": ["format": "gptq", "groupSize": 128] as [String: Any],
            ])
        }
        let descriptor = try SmeltCAMPackageDescriptor(from: cam)
        let admission = SmeltCAMFeatureAdmission(descriptor: descriptor)
        let storageIDs = admission.requiredObligations
            .filter { $0.code == "quant.storage.gptq" }
            .map(\.canonicalID)
        let isolated = SmeltCAMFeatureAdmission(
            descriptor: descriptor,
            consumedObligationIDs: storageIDs
        )

        #expect(admission.unsupportedFeatureSet == [
            "quant.calibration.gptq",
            "quant.storage.gptq",
        ])
        #expect(isolated.unsupportedFeatureSet == ["quant.calibration.gptq"])
    }

    @Test func heavyQuantProjectionDiagnosticTracksStructuralFactRemoval() throws {
        let base = registryModuleIR("ds4_heavy_quant")
        let cases: [(label: String, mutation: IRMutation, absent: [String], present: [String])] = [
            (
                label: "rope",
                mutation: { object in
                    try Self.mutateTransformer(&object, label: "rope") { transformer in
                        try Self.setValue(&transformer, path: ["attention", "rope", "kind"], to: "neox", label: "rope")
                    }
                },
                absent: ["transformer.rope.yarn"],
                present: ["transformer.moe.router", "quant.calibration.gptq"]
            ),
            (
                label: "moe",
                mutation: { object in
                    try Self.mutateTransformer(&object, label: "moe") { transformer in
                        guard transformer.removeValue(forKey: "router") != nil,
                              transformer.removeValue(forKey: "expert") != nil
                        else {
                            throw Self.mutationFailure("moe: trunk transformer has no router/expert facts")
                        }
                    }
                },
                absent: ["transformer.moe.router", "transformer.moe.expert"],
                present: ["transformer.rope.yarn", "quant.storage.gptq"]
            ),
            (
                label: "generated kernels",
                mutation: { object in
                    try Self.removeCompileConstraint(&object, key: "generated-kernels", label: "generated kernels")
                },
                absent: ["compile.generated-kernels"],
                present: ["compile.memory-bound", "gate.quant-quality"]
            ),
            (
                label: "memory bound",
                mutation: { object in
                    try Self.removeCompileConstraint(&object, key: "memory", label: "memory bound")
                },
                absent: ["compile.memory-bound"],
                present: ["compile.generated-kernels", "quant.storage.gptq"]
            ),
            (
                label: "quant quality",
                mutation: { object in
                    try Self.removeElement(&object, "gates", label: "quant quality", where: {
                        ($0["id"] as? String) == "quant_quality"
                    })
                },
                absent: ["gate.quant-quality"],
                present: ["quant.calibration.gptq", "transformer.rope.yarn"]
            ),
        ]

        for testCase in cases {
            let cam = try mutatedModuleIR(base, testCase.mutation)
            let error = try Self.expectUnsupportedProjectionFeatures(
                cam,
                label: testCase.label,
                requiredFragments: ["unsupported CAM package projection features"] + testCase.present
            )
            for fragment in testCase.absent {
                #expect(!error.contains(fragment), "\(testCase.label): \(error)")
            }
            for fragment in testCase.present {
                #expect(error.contains(fragment), "\(testCase.label): \(error)")
            }
        }
    }

    @Test func rejectsSemanticDriftBeforePackageProjection() throws {
        let cam = try mutatedModuleIR(registryModuleIR("qwen35_text")) { object in
            // `stop: eos-token ... | max-steps 512 | host-cancel` collapsed to
            // `stop: max-steps 1 | host-cancel`.
            try Self.mutateFlow(&object, label: "semantic drift") { flow in
                guard flow["stop"] != nil else {
                    throw Self.mutationFailure("semantic drift: flow has no stop conditions")
                }
                flow["stop"] = [
                    ["kind": "host-cancel"],
                    ["kind": "max-steps", "value": 1] as [String: Any],
                ]
            }
        }

        try Self.expectProjectionRejectedWithoutProfileMiss(
            cam,
            label: "semantic drift",
            requiredFragments: ["package loop stop condition drifted"]
        )
    }

    @Test func projectsSupportedAndRejectsUnsupportedCheckedTextDrift() throws {
        let base = registryModuleIR("qwen35_text")
        let baseline = try SmeltCAMCheckedPackageProjector.project(
            cam: base,
            artifactRoot: "/tmp/source"
        )
        let supportedCases: [(String, IRMutation)] = [
            (
                "source locator",
                { object in
                    try Self.setSourceLocator(&object, id: "weights", to: "Qwen/Qwen3.5-2B-drift", label: "source locator")
                }
            ),
            (
                "quant group",
                { object in
                    try Self.mutateDefaultQuantRule(&object, label: "quant group") { rule in
                        try Self.setValue(&rule, path: ["storage", "groupSize"], to: 32, label: "quant group")
                    }
                }
            ),
            (
                "prefill batch",
                { object in
                    try Self.setCompileValue(&object, key: "prefill", to: "metal batch 128", label: "prefill batch")
                }
            ),
            (
                "thinking policy",
                { object in
                    try Self.setNodeAnnotation(&object, node: "tokenizer", key: "thinking-policy", to: "enabled", label: "thinking policy")
                }
            ),
        ]
        let rejectedCases: [(String, IRMutation)] = [
            (
                "optional export input",
                { object in
                    // `in style: text utf8 optional` added to the export.
                    try Self.mutateElement(&object, "exports", label: "optional export input", where: { _ in true }) { export in
                        guard var inputs = export["inputs"] as? [[String: Any]] else {
                            throw Self.mutationFailure("optional export input: export has no inputs")
                        }
                        inputs.append([
                            "name": "style",
                            "optional": true,
                            "type": ["name": "text", "attributes": ["encoding": "utf8"]] as [String: Any],
                        ])
                        export["inputs"] = inputs
                    }
                }
            ),
            (
                "unused graph node",
                { object in
                    try Self.addUnusedGraphNode(&object, consuming: "prompt", label: "unused graph node")
                }
            ),
            (
                "extra tensor binding",
                { object in
                    try Self.appendTensorMap(
                        &object,
                        pattern: "embed_tokens",
                        targetBlock: "trunk",
                        targetSelector: "embed_tokens",
                        label: "extra tensor binding"
                    )
                }
            ),
            (
                "graph signature",
                { object in
                    try Self.setNodeAnnotation(&object, node: "detokenizer", key: "tag", to: "text-renderer", label: "graph signature")
                }
            ),
            (
                "inventory",
                { object in
                    try Self.replaceInGateRequirementValues(
                        &object,
                        of: ",tokenizer.json,tokenizer.bin,module.json",
                        with: ",tokenizer.json,module.json",
                        expectedCount: 1,
                        label: "inventory"
                    )
                }
            ),
            (
                "flow step label",
                { object in
                    try Self.mutateFlowPhase(&object, label: "flow step label", where: {
                        ($0["label"] as? String) == "decode"
                    }) { phase in
                        phase["label"] = "sample"
                    }
                }
            ),
            (
                "flow step order",
                { object in
                    try Self.mutateFlowPhase(&object, label: "flow step order", where: {
                        ($0["label"] as? String) == "decode"
                    }) { phase in
                        phase["calls"] = Self.nodeCalls(["sampler", "trunk"])
                    }
                }
            ),
            (
                "flow emit",
                { object in
                    try Self.mutateFlow(&object, label: "flow emit") { flow in
                        guard flow["emit"] != nil else {
                            throw Self.mutationFailure("flow emit: flow has no emit")
                        }
                        flow["emit"] = [["kind": "nodePort", "node": "sampler", "port": "tokens"]]
                    }
                }
            ),
            (
                "sampler annotation",
                { object in
                    try Self.setNodeAnnotation(&object, node: "sampler", key: "tag", to: "chooser", label: "sampler annotation")
                }
            ),
            (
                "prompt format",
                { object in
                    try Self.setNodeAnnotation(&object, node: "tokenizer", key: "prompt-format", to: "raw", label: "prompt format")
                }
            ),
            (
                "capability route",
                { object in
                    try Self.replaceCapability(&object, "run.generate", with: "run.chat", label: "capability route")
                }
            ),
            (
                "feedback edge",
                { object in
                    try Self.setFeedbackEdgeSourceToGraphValue(&object, named: "tokens", label: "feedback edge")
                }
            ),
            (
                "extra scheduled node",
                { object in
                    try Self.mutateFlowPhase(&object, label: "extra scheduled node", where: {
                        ($0["label"] as? String) == "decode"
                    }) { phase in
                        phase["calls"] = Self.nodeCalls(["trunk", "sampler", "detokenizer"])
                    }
                }
            ),
            (
                "missing prefill compile",
                { object in
                    try Self.removeCompileConstraint(&object, key: "prefill", label: "missing prefill compile")
                }
            ),
        ]

        for (label, mutation) in supportedCases {
            let cam = try mutatedModuleIR(base, mutation)
            try Self.expectSupportedCAMConfigDriftProjects(cam, baseline: baseline, label: label)
        }
        for (label, mutation) in rejectedCases {
            let cam = try mutatedModuleIR(base, mutation)
            try Self.expectProjectionRejectedWithoutProfileMiss(cam, label: label)
        }
    }

    @Test func projectsSupportedAndRejectsUnsupportedCheckedFastDrift() throws {
        let base = registryModuleIR("qwen35_fast")
        let baseline = try SmeltCAMCheckedPackageProjector.project(
            cam: base,
            artifactRoot: "/tmp/source"
        )
        let supportedCases: [(String, IRMutation)] = [
            (
                "source locator",
                { object in
                    try Self.setSourceLocator(&object, id: "weights", to: "Qwen/Qwen3.5-0.8B-drift", label: "source locator")
                }
            ),
            (
                "tokenizer locator",
                { object in
                    try Self.setSourceLocator(&object, id: "tokenizer", to: "Qwen/Qwen3.5-0.8B/tokenizer-drift.json", label: "tokenizer locator")
                }
            ),
            (
                "hidden size",
                { object in
                    try Self.mutateTransformer(&object, label: "hidden size") { transformer in
                        try Self.setValue(&transformer, path: ["hiddenSize"], to: 2048, label: "hidden size")
                    }
                }
            ),
            (
                "ffn dim",
                { object in
                    try Self.mutateTransformer(&object, label: "ffn dim") { transformer in
                        try Self.setValue(&transformer, path: ["ffn", "dim"], to: 4096, label: "ffn dim")
                    }
                }
            ),
            (
                "quant format",
                { object in
                    try Self.mutateDefaultQuantRule(&object, label: "quant format") { rule in
                        try Self.setValue(&rule, path: ["storage", "format"], to: "lut-u4", label: "quant format")
                    }
                }
            ),
            (
                "quant group",
                { object in
                    try Self.mutateDefaultQuantRule(&object, label: "quant group") { rule in
                        try Self.setValue(&rule, path: ["storage", "groupSize"], to: 128, label: "quant group")
                    }
                }
            ),
            (
                "max steps",
                { object in
                    try Self.mutateFlow(&object, label: "max steps") { flow in
                        try Self.mutateElement(&flow, "stop", label: "max steps", where: {
                            ($0["kind"] as? String) == "max-steps"
                        }) { stop in
                            stop["value"] = 256
                        }
                    }
                }
            ),
        ]
        let rejectedCases: [(String, IRMutation)] = [
            (
                "embedding quantization",
                { object in
                    try Self.setQuantRulePattern(&object, action: "quantize", from: "embeddings", to: "lm_head", label: "embedding quantization")
                }
            ),
            (
                "prefill compile",
                { object in
                    try Self.removeCompileConstraint(&object, key: "prefill", label: "prefill compile")
                }
            ),
            (
                "prompt format",
                { object in
                    try Self.setNodeAnnotation(&object, node: "tokenizer", key: "prompt-format", to: "raw", label: "prompt format")
                }
            ),
            (
                "capability route",
                { object in
                    try Self.replaceCapability(&object, "run.generate", with: "run.chat", label: "capability route")
                }
            ),
            (
                "inventory",
                { object in
                    try Self.replaceInGateRequirementValues(
                        &object,
                        of: ",dispatches.bin,prefill_dispatches.bin,tokenizer.json",
                        with: ",dispatches.bin,tokenizer.json",
                        expectedCount: 1,
                        label: "inventory"
                    )
                }
            ),
        ]

        for (label, mutation) in supportedCases {
            let cam = try mutatedModuleIR(base, mutation)
            try Self.expectSupportedCAMConfigDriftProjects(cam, baseline: baseline, label: label)
        }
        for (label, mutation) in rejectedCases {
            let cam = try mutatedModuleIR(base, mutation)
            try Self.expectProjectionRejectedWithoutProfileMiss(cam, label: label)
        }
    }

    @Test func projectsSupportedAndRejectsUnsupportedCheckedReasonerDrift() throws {
        let base = registryModuleIR("qwen35_reasoner")
        let baseline = try SmeltCAMCheckedPackageProjector.project(
            cam: base,
            artifactRoot: "/tmp/source"
        )
        let supportedCases: [(String, IRMutation)] = [
            (
                "source locator",
                { object in
                    try Self.setSourceLocator(&object, id: "weights", to: "Qwen/Qwen3.5-4B-drift", label: "source locator")
                }
            ),
            (
                "tokenizer locator",
                { object in
                    try Self.setSourceLocator(&object, id: "tokenizer", to: "Qwen/Qwen3.5-4B/tokenizer-drift.json", label: "tokenizer locator")
                }
            ),
            (
                "prompt builder template",
                { object in
                    try Self.setNodeAnnotation(&object, node: "prompt_builder", key: "template", to: "raw-review", label: "prompt builder template")
                }
            ),
            (
                "layer pattern",
                { object in
                    try Self.mutateTransformer(&object, label: "layer pattern") { transformer in
                        try Self.setValue(
                            &transformer,
                            path: ["layers", "roles"],
                            to: ["delta", "delta", "attention", "delta"],
                            label: "layer pattern"
                        )
                    }
                }
            ),
            (
                "attention heads",
                { object in
                    try Self.mutateTransformer(&object, label: "attention heads") { transformer in
                        try Self.setValue(&transformer, path: ["attention", "kvHeads"], to: 2, label: "attention heads")
                    }
                }
            ),
            (
                "rope theta",
                { object in
                    try Self.mutateTransformer(&object, label: "rope theta") { transformer in
                        try Self.setValue(&transformer, path: ["attention", "rope", "theta"], to: 1_000_000, label: "rope theta")
                    }
                }
            ),
            (
                "ffn dim",
                { object in
                    try Self.mutateTransformer(&object, label: "ffn dim") { transformer in
                        try Self.setValue(&transformer, path: ["ffn", "dim"], to: 8192, label: "ffn dim")
                    }
                }
            ),
            (
                "quant format",
                { object in
                    try Self.mutateDefaultQuantRule(&object, label: "quant format") { rule in
                        try Self.setValue(&rule, path: ["storage", "format"], to: "lut-u4", label: "quant format")
                    }
                }
            ),
            (
                "max steps",
                { object in
                    try Self.mutateFlow(&object, label: "max steps") { flow in
                        try Self.mutateElement(&flow, "stop", label: "max steps", where: {
                            ($0["kind"] as? String) == "max-steps"
                        }) { stop in
                            stop["value"] = 768
                        }
                    }
                }
            ),
        ]
        let rejectedCases: [(String, IRMutation)] = [
            (
                "prompt builder candidate input",
                { object in
                    // `candidate -> prompt_builder.candidate` retargeted to a
                    // `prompt_builder.answer` port: the edge destination and the
                    // node's declared input port both take the new name.
                    try Self.mutateElement(&object, "graphEdges", label: "prompt builder candidate input", where: {
                        (($0["to"] as? [String: Any])?["node"] as? String) == "prompt_builder"
                            && (($0["to"] as? [String: Any])?["port"] as? String) == "candidate"
                    }) { edge in
                        edge["to"] = ["kind": "nodePort", "node": "prompt_builder", "port": "answer"]
                    }
                    try Self.mutateElement(&object, "graphNodes", label: "prompt builder candidate input", where: {
                        ($0["id"] as? String) == "prompt_builder"
                    }) { node in
                        try Self.mutateElement(&node, "inputs", label: "prompt builder candidate input", where: {
                            ($0["name"] as? String) == "candidate"
                        }) { input in
                            input["name"] = "answer"
                        }
                    }
                }
            ),
            (
                "prompt builder tokenizer edge",
                { object in
                    // `review_prompt -> tokenizer(...)` becomes `context ->
                    // tokenizer(...)`: the tokenizer consumes the module input
                    // directly, bypassing the prompt builder's output value.
                    let textType: [String: Any] = ["name": "text", "attributes": ["encoding": "utf8"]]
                    try Self.mutateElement(&object, "graphEdges", label: "prompt builder tokenizer edge", where: {
                        (($0["to"] as? [String: Any])?["node"] as? String) == "tokenizer"
                            && (($0["to"] as? [String: Any])?["port"] as? String) == "review_prompt"
                    }) { edge in
                        edge["from"] = ["kind": "moduleInput", "name": "context"]
                        edge["to"] = ["kind": "nodePort", "node": "tokenizer", "port": "context"]
                        edge["type"] = textType
                    }
                    try Self.mutateElement(&object, "graphNodes", label: "prompt builder tokenizer edge", where: {
                        ($0["id"] as? String) == "tokenizer"
                    }) { node in
                        try Self.mutateElement(&node, "inputs", label: "prompt builder tokenizer edge", where: {
                            ($0["name"] as? String) == "review_prompt"
                        }) { input in
                            input["name"] = "context"
                            input["type"] = textType
                        }
                    }
                }
            ),
            (
                "setup order",
                { object in
                    try Self.mutateFlowPhase(&object, label: "setup order", where: {
                        ($0["role"] as? String) == "setup"
                    }) { phase in
                        phase["calls"] = Self.nodeCalls(["tokenizer", "prompt_builder"])
                    }
                }
            ),
            (
                "delta projection",
                { object in
                    try Self.mutateTransformer(&object, label: "delta projection") { transformer in
                        try Self.setValue(&transformer, path: ["delta", "projections", "z"], to: 2048, label: "delta projection")
                    }
                }
            ),
            (
                "embedding quantization",
                { object in
                    try Self.setQuantRulePattern(&object, action: "quantize", from: "embeddings", to: "lm_head", label: "embedding quantization")
                }
            ),
            (
                "inventory",
                { object in
                    try Self.replaceInGateRequirementValues(
                        &object,
                        of: ",dispatches.bin,prefill_dispatches.bin,tokenizer.json",
                        with: ",dispatches.bin,tokenizer.json",
                        expectedCount: 1,
                        label: "inventory"
                    )
                }
            ),
        ]

        for (label, mutation) in supportedCases {
            let cam = try mutatedModuleIR(base, mutation)
            try Self.expectSupportedCAMConfigDriftProjects(cam, baseline: baseline, label: label)
        }
        for (label, mutation) in rejectedCases {
            let cam = try mutatedModuleIR(base, mutation)
            try Self.expectProjectionRejectedWithoutProfileMiss(cam, label: label)
        }
    }

    @Test func requiresExplicitArtifactRoot() throws {
        let cam = registryModuleIR("qwen35_text")

        #expect(throws: SmeltCAMCheckedPackageProjectionError.self) {
            _ = try SmeltCAMCheckedPackageProjector.project(cam: cam, artifactRoot: "")
        }
    }

    @Test func checkedProjectionFactsAreRegistryOwned() throws {
        let source = try String(
            contentsOf: Self.repoRoot
                .appendingPathComponent("Sources", isDirectory: true)
                .appendingPathComponent("SmeltCompiler", isDirectory: true)
                .appendingPathComponent("SmeltCAMCheckedPackageProjection.swift"),
            encoding: .utf8
        )

        #expect(!source.contains("acceptedSemanticSHA256"))
        #expect(!source.contains("acceptedExportABISHA256"))
        #expect(!source.contains("acceptedDescriptorGraphSignatureSHA256"))
        #expect(!source.contains(#"locator: "Qwen/Qwen3.5-2B""#))
        #expect(!source.contains(#"locator: "Qwen/Qwen3.5-2B/tokenizer.json""#))
        #expect(!source.contains(#"locator: "Qwen/Qwen3.5-0.8B""#))
        #expect(!source.contains(#"locator: "Qwen/Qwen3.5-0.8B/tokenizer.json""#))
        #expect(!source.contains(#"locator: "Qwen/Qwen3.5-4B""#))
        #expect(!source.contains(#"locator: "Qwen/Qwen3.5-4B/tokenizer.json""#))
        #expect(
            Self.occurrences(
                of: #"locator: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice""#,
                in: source
            ) == 0
        )
        #expect(
            Self.occurrences(
                of: #"locator: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/tokenizer.json""#,
                in: source
            ) == 0
        )
        #expect(!source.contains("runtimeArchitecture: SmeltPackageRuntimeArchitecture.qwen"))
        #expect(!source.contains("chatTemplate: SmeltPromptTemplateName.chatML"))
        #expect(!source.contains("chatTemplate: SmeltPromptTemplateName.channelTurns"))
        #expect(!source.contains("SmeltPackageRuntimeArchitecture.qwen"))
        #expect(!source.contains("SmeltPromptTemplateName.chatML"))
        #expect(!source.contains("SmeltPromptTemplateName.channelTurns"))
        #expect(
            source.contains(
                "private static let checkedDerivedAudioPackageProjectionProfiles: "
                    + "[CheckedDerivedAudioPackageProjectionProfile] = []"
            )
        )
        #expect(source.contains("derivedTextProjectionProfile("))
        #expect(source.contains("derivedTextProjectionID("))
        #expect(source.contains("derivedTextQuantization(from:"))
        #expect(source.contains("derivedTextAssembly(from:"))
        #expect(source.contains("derivedAudioProjectionProfile("))
        #expect(source.contains("derivedAudioGraph("))
        #expect(source.contains("derivedAudioQuantizationProfile("))
        #expect(source.contains("packageInventoryFiles(from:"))
        #expect(source.contains("primaryModelSource(from:"))
        #expect(source.contains("defaultStorage: .init(format: storage.format, groupSize: groupSize)"))
        #expect(source.contains("maxBatchSize: parsed.batch"))
        #expect(!source.contains("CheckedProjectionPackageSpecCompatibility"))
        #expect(!source.contains("CheckedProjectionPackageSpecBridge"))
        #expect(source.contains("CheckedProjectionTextPolicyBridge"))
        #expect(source.contains("textPolicyBridge("))
        #expect(!source.contains("lowererRuntimeArchitecture"))
        #expect(!source.contains(#"annotation("lowerer-runtime""#))
        #expect(!source.contains(#"annotation("chat-template""#))
        #expect(source.contains("Temporary text lowerer adapter"))
        #expect(!source.contains("packageSpecBridge(from:"))
        #expect(!source.contains("packageSpecRuntimeArchitecture"))
        #expect(!source.contains("CheckedProjectionRuntimeAdapter"))
        #expect(!source.contains("RuntimeAdapter"))
        #expect(!source.contains("runtimeAdapter"))
        #expect(!source.contains("packageRuntimeArchitecture"))
        #expect(!source.contains("let loop: SmeltLoopSchedule"))
        #expect(!source.contains("contract.loop"))
        #expect(!source.contains(".tokenFeedbackText"))
        #expect(source.contains("textLowererCompatibilityLoop(from:"))
        #expect(!source.contains("SmeltModelIR.qwen35_0_8B"))
        #expect(!source.contains("SmeltModelIR.qwen35_4B"))
        #expect(!source.contains(#"contract.id == "qwen35_fast""#))
        #expect(!source.contains(#"contract.id != "qwen35_fast""#))
        #expect(!source.contains(#"id: "qwen35_text""#))
        #expect(!source.contains(#"id: "qwen35_fast""#))
        #expect(!source.contains(#"id: "qwen35_reasoner""#))
        #expect(source.contains(#""text-to-text-transformer""#))
        #expect(source.contains(#""two-text-transformer""#))
        #expect(source.contains(#""prefill-decode""#))
        #expect(source.contains(#""decode-only""#))
        #expect(source.contains(#""tqh-embeddings""#))
        #expect(source.contains(#""projection-bias""#))
        #expect(source.contains(#""prompt-adapter""#))
        #expect(
            source.contains(
                #""streaming-text-to-\("#
            )
        )
        #expect(source.contains(".explicitCAMShape("))
        #expect(source.contains(".roleAttentionCAMShape"))
        #expect(!source.contains(".checkedDerivedShape("))
        #expect(source.contains("checkedPackageProjectionProfiles"))
        #expect(source.contains("checkedPackageProjectionProfile("))

        let selectorBody = try Self.slice(
            source,
            from: "private static func checkedPackageProjectionProfile",
            through: "private static func derivedAudioProjectionProfile"
        )
        for banned in [
            "SmeltPackageRuntimeArchitecture",
            "SmeltPromptTemplateName",
            "chatTemplate",
            "thinkingPolicy",
            "locator",
            "modelName",
            "moduleID",
            "fixture",
            ".cam",
            "draft-judge",
            "reasoner",
            "google",
            "WeiboAI",
            #"contract.id =="#,
            #"contract.id !="#,
        ] {
            #expect(!selectorBody.contains(banned), Comment(rawValue: banned))
        }

        let textDerivationBody = try Self.slice(
            source,
            from: "private static func derivedTextProjectionProfile",
            through: "private static func checkedPackageProjectionProfile"
        )
        for required in [
            "primaryModelSource(from:",
            "packageInventoryFiles(from:",
            "derivedTextPolicyBridge(from:",
            "derivedTextQuantization(from:",
            "derivedTextPrefillPolicy(from:",
            "derivedTextAssembly(from:",
            "derivedTextProjectionID(",
        ] {
            #expect(textDerivationBody.contains(required), Comment(rawValue: required))
        }
        for banned in [
            "79b4c0a181c578a8a08aaf02a4eb46cddaea0b427e043e6fd4f91bd938040fd0",
            "b53a72b8058fcd4f9f999a3d6e2c516e00282efbb9258b1eb9b9babe21a9659d",
            "41f548686c492dfc58fdaa6cccd2ea801f55ec49c86c311ae82fd5ec5a9b0610",
            "65f52025f8da627a5e41bc7c51f97c631603c414e9cd2ea9924e3d21f88f4afb",
            "009e265813438429d7fde5d0fdf5d834d04d0b9954e564f7e865b23e99f5b0e4",
            "Qwen/Qwen3.5",
            "WeiboAI",
            #"contract.id =="#,
            #"contract.id !="#,
            ".cam",
        ] {
            #expect(!textDerivationBody.contains(banned), Comment(rawValue: banned))
        }

        let audioDerivationBody = try Self.slice(
            source,
            from: "private static func derivedAudioProjectionProfile",
            through: "private static func modelIR"
        )
        for required in [
            "packageInventoryFiles(from:",
            "primaryModelSource(from:",
            "derivedAudioGraph(",
            "derivedAudioTensorMaps(",
            "derivedAudioQuantizationProfile(",
            "derivedAudioProjectionID(",
        ] {
            #expect(audioDerivationBody.contains(required), Comment(rawValue: required))
        }
        for banned in [
            "59e764fb436823239195e78542b86e093c79b42cab2a3df9c238e38abc55447a",
            "691518abe0d2984d67f8c4eaaa1173c44d7163eba13a46d67201e5b05e86c8dc",
            "3d41d53467492c0f78d91a49dda6e4e8ae12a93ac9bec24cb2be21bfacb7b062",
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            #"contract.id =="#,
            #"contract.id !="#,
            ".cam",
        ] {
            #expect(!audioDerivationBody.contains(banned), Comment(rawValue: banned))
        }

        #expect(source.contains("SmeltCAMFeatureAdmission"))
        #expect(!source.contains("private static func unsupportedProjectionFeatures"))
        #expect(!source.contains("UnsupportedProjectionFeature"))
        let featureAdmissionBody = try Self.slice(
            source,
            from: "private static func profileMissDiagnostic",
            through: "private static func derivedTextProjectionProfile"
        )
        for banned in [
            "ds4",
            "DS4",
            "deepseek",
            "moduleID",
            "modelName",
            "family",
            "locator",
        ] {
            #expect(!featureAdmissionBody.contains(banned), Comment(rawValue: banned))
        }

        let contractTable = try Self.slice(
            source,
            from: "private static let checkedPackageProjectionProfiles",
            through: "public static func project"
        )
        #expect(!contractTable.contains("loop:"))
        #expect(!contractTable.contains("SmeltLoopSchedule("))
        for banned in [
            "lowererRuntimeArchitecture",
            "SmeltPackageRuntimeArchitecture",
            "SmeltPromptTemplateName",
            "chatTemplate",
            "thinkingPolicy",
            "chatml",
            "chat-template",
            "lowerer-runtime",
        ] {
            #expect(!contractTable.contains(banned), Comment(rawValue: banned))
        }

        let loopProjectionBody = try Self.slice(
            source,
            from: "private static func textLowererCompatibilityLoop",
            through: "private static func textLowererCompatibilityBlocks"
        )
        for banned in [
            #"contract.id =="#,
            #"contract.id !="#,
            ".tokenFeedbackText",
            "SmeltLoopSchedule.tokenFeedbackText",
        ] {
            #expect(!loopProjectionBody.contains(banned), Comment(rawValue: banned))
        }
    }

    private static func expectNoCheckedPackageProjectionProfile(
        _ cam: SmeltCAMIR,
        label: String
    ) throws {
        let description = try expectProjectionError(cam, label: label)
        if !description.isEmpty {
            #expect(
                description.contains("no checked package projection profile"),
                "\(label): \(description)"
            )
        }
    }

    private static func expectProjectionRejectedWithoutProfileMiss(
        _ cam: SmeltCAMIR,
        label: String,
        artifactRoot: String = "/tmp/source",
        requiredFragments: [String] = []
    ) throws {
        do {
            _ = try SmeltCAMCheckedPackageProjector.project(
                cam: cam,
                artifactRoot: artifactRoot
            )
            Issue.record("expected projection rejection for \(label)")
        } catch {
            let description = projectionErrorDescription(error)
            #expect(
                !description.contains("no checked package projection profile"),
                "\(label): \(description)"
            )
            for fragment in requiredFragments {
                #expect(description.contains(fragment), "\(label): \(description)")
            }
        }
    }

    private static func expectSupportedCAMConfigDriftProjects(
        _ cam: SmeltCAMIR,
        baseline: SmeltCAMCheckedPackageProjection,
        label: String
    ) throws {
        let projection = try SmeltCAMCheckedPackageProjector.project(
            cam: cam,
            artifactRoot: "/tmp/source"
        )
        #expect(
            projection.camSemanticSHA256 != baseline.camSemanticSHA256,
            "\(label): CAM semantic hash did not change"
        )
        #expect(!projection.packageProjectionID.isEmpty, "\(label): projection id missing")
        try assertTextProjectionMatchesCAM(projection, cam: cam, label: label)
    }

    private static func assertTextProjectionMatchesCAM(
        _ projection: SmeltCAMCheckedPackageProjection,
        cam: SmeltCAMIR,
        label: String
    ) throws {
        let plan = try SmeltPackageResolvedPlan.resolve(projection.spec)
        let modelSourceIDs = Set(cam.tensors.map(\.source))
        #expect(modelSourceIDs.count == 1, "\(label): expected one model tensor source")
        let modelSourceID = try #require(modelSourceIDs.first)
        let modelSource = try #require(cam.sources.first { $0.id == modelSourceID })
        #expect(cam.flows.count == 1, "\(label): expected one text flow")
        let flow = try #require(cam.flows.first)
        let maxSteps = try #require(flow.stop.first { $0.kind == .maxSteps }?.value)
        let eosTokens = flow.stop
            .filter { $0.kind == .eosToken }
            .compactMap(\.value)
            .map(Int32.init)
            .sorted()
        let defaultQuant = try #require(cam.quantization.first { $0.action == .default })
        let defaultStorage = try #require(defaultQuant.storage)
        let expectedQuantFormat = try expectedTextDType(defaultStorage.format)

        #expect(plan.modelName == modelSource.locator, "\(label): model source locator not projected")
        #expect(
            plan.validationParityFixture == modelSource.locator,
            "\(label): validation fixture not projected"
        )
        #expect(
            plan.sources.map { "\($0.id):\($0.kind.rawValue):\($0.locator)" } == [
                "package:local-directory:/tmp/source",
            ],
            "\(label): package source root not projected"
        )
        #expect(
            plan.quantization?.format == expectedQuantFormat,
            "\(label): quant format not projected"
        )
        #expect(
            plan.quantization?.groupSize == defaultStorage.groupSize,
            "\(label): quant group not projected"
        )
        #expect(plan.policy.inference?.maxTokens == maxSteps, "\(label): max tokens not projected")
        #expect(plan.policy.decode?.maxSteps == maxSteps, "\(label): decode max steps not projected")
        #expect(plan.policy.inference?.eosTokens == eosTokens, "\(label): EOS tokens not projected")
        #expect(
            projection.spec.outputFiles.files.sorted() == projection.packageFiles,
            "\(label): package inventory not projected"
        )
        if let checkpointMap = modelSource.checkpointMap {
            let architecture = try Self.object(
                projection.spec.architectureConfig,
                label: "\(label): architecture_config"
            )
            let loading = try Self.object(
                try Self.value("loading", in: architecture),
                label: "\(label): loading"
            )
            #expect(
                try Self.string("checkpoint_map", in: loading) == checkpointMap,
                "\(label): checkpoint map not projected"
            )
        }
    }

    private static func expectedTextDType(
        _ format: SmeltCAMIR.QuantStorageFormat
    ) throws -> SmeltPackageSpec.TensorDType {
        switch format {
        case .affineU4, .lutU4:
            return .u4
        case .fp16:
            return .f16
        case .bf16:
            return .bf16
        case .gptq:
            return .gptq
        case .turboQuantH:
            return .turboQuantH
        case .binary1:
            return .binary1
        case .ternary2:
            return .ternary2
        }
    }

    @discardableResult
    private static func expectProjectionError(
        _ cam: SmeltCAMIR,
        label: String
    ) throws -> String {
        do {
            _ = try SmeltCAMCheckedPackageProjector.project(
                cam: cam,
                artifactRoot: "/tmp/source"
            )
            Issue.record("expected projection error for \(label)")
            return ""
        } catch let error as SmeltCAMCheckedPackageProjectionError {
            return error.description
        } catch {
            Issue.record("expected projection error for \(label), got \(error)")
            return ""
        }
    }

    private static func projectionErrorDescription(_ error: Error) -> String {
        if let error = error as? SmeltCAMCheckedPackageProjectionError {
            return error.description
        }
        return String(describing: error)
    }

    @discardableResult
    private static func expectUnsupportedProjectionFeatures(
        _ cam: SmeltCAMIR,
        label: String,
        requiredFragments: [String] = [
            "unsupported CAM package projection features",
            "transformer.rope.yarn",
            "uses Yarn RoPE",
            "transformer.moe.router",
            "transformer.moe.expert",
            "uses MoE router",
            "uses MoE expert",
            "quant.storage.gptq",
            "uses GPTQ tensor storage",
            "quant.calibration.gptq",
            "declares GPTQ calibration artifacts",
            "compile.generated-kernels",
            "requires generated kernels",
            "compile.memory-bound",
            "declares peak memory requirement",
            "gate.quant-quality",
            "declares quant-quality gate subjects",
        ]
    ) throws -> String {
        do {
            _ = try SmeltCAMCheckedPackageProjector.project(
                cam: cam,
                artifactRoot: "/tmp/source"
            )
            Issue.record("expected unsupported projection features for \(label)")
            return ""
        } catch let error as SmeltCAMCheckedPackageProjectionError {
            let description = error.description
            for fragment in requiredFragments {
                #expect(description.contains(fragment), "\(label): \(description)")
            }
            #expect(
                !description.contains("no checked package projection profile"),
                "\(label): \(description)"
            )
            return description
        } catch {
            Issue.record("expected unsupported projection error for \(label), got \(error)")
            return ""
        }
    }

    private static func occurrences(of needle: String, in haystack: String) -> Int {
        haystack.components(separatedBy: needle).count - 1
    }

    private static func slice(_ source: String, from start: String, through end: String) throws -> String {
        guard let startRange = source.range(of: start),
              let endRange = source[startRange.upperBound...].range(of: end)
        else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "missing source slice \(start) through \(end)"
            )
        }
        return String(source[startRange.lowerBound..<endRange.lowerBound])
    }

    private static func object(
        _ value: SmeltPackageSpecValue,
        label: String
    ) throws -> [String: SmeltPackageSpecValue] {
        guard case .object(let object) = value else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported("\(label) is not an object")
        }
        return object
    }

    private static func assertLoadingCheckpointMap(
        _ spec: SmeltPackageSpec,
        expected: String,
        label: String
    ) throws {
        let architecture = try Self.object(
            spec.architectureConfig,
            label: "\(label): architecture_config"
        )
        let loading = try Self.object(
            try Self.value("loading", in: architecture),
            label: "\(label): loading"
        )
        #expect(
            try Self.string("checkpoint_map", in: loading) == expected,
            "\(label): checkpoint map not projected"
        )
    }

    private static func int(
        _ key: String,
        in object: [String: SmeltPackageSpecValue]
    ) throws -> Int {
        let value = try value(key, in: object)
        if case .int(let int) = value {
            return int
        }
        if case .number(let number) = value, number.rounded() == number {
            return Int(number)
        }
        throw SmeltCAMCheckedPackageProjectionError.unsupported("\(key) is not an integer")
    }

    private static func string(
        _ key: String,
        in object: [String: SmeltPackageSpecValue]
    ) throws -> String {
        guard case .string(let string) = try value(key, in: object) else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported("\(key) is not a string")
        }
        return string
    }

    private static func stringArray(
        _ key: String,
        in object: [String: SmeltPackageSpecValue]
    ) throws -> [String] {
        guard case .array(let values) = try value(key, in: object) else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported("\(key) is not an array")
        }
        return try values.map { value in
            guard case .string(let string) = value else {
                throw SmeltCAMCheckedPackageProjectionError.unsupported("\(key) contains a non-string")
            }
            return string
        }
    }

    private static func value(
        _ key: String,
        in object: [String: SmeltPackageSpecValue]
    ) throws -> SmeltPackageSpecValue {
        guard let value = object[key] else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported("\(key) is missing")
        }
        return value
    }

    private static func writeQwen3TTSPayloads(
        to directory: URL,
        u4GroupSize: Int = 128,
        decode: Qwen3TTSManifest.Decode? = .init(
            doSample: false,
            temperature: 1,
            topK: 50,
            subtalkerTemperature: 1,
            subtalkerTopK: 50
        ),
        mutateManifest: ((Qwen3TTSManifest) throws -> Qwen3TTSManifest)? = nil
    ) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let profile = SmeltQwen3TTSPackageProfiles.runnable
        let specs = qwen3TTSRunnableSpecs(u4GroupSize: u4GroupSize)
        let pageSize = profile.pageSize
        let orderedSpecs = specs.sorted { $0.name < $1.name }
        let layout = Qwen3TTSPackageBuilder.planLayout(orderedSpecs, pageSize: pageSize)
        let pipelines = ["qwen3_tts_test_kernel"]
        let graph = profile.graph(textEmbeddingIsBF16: false)
        let validation = SmeltPackagePerformanceProfiles.validation(
            parityFixture: profile.modelName,
            performanceGate: profile.performanceGate,
            structureProfile: profile.structureProfile(pipelines: pipelines, graph: graph)
        )
        var manifest = Qwen3TTSManifest(
            version: 1,
            blocks: graph,
            loop: profile.loop,
            modelName: profile.modelName,
            pageSize: pageSize,
            pipelines: pipelines,
            eosTokens: profile.eosTokens,
            totalBytes: layout.totalBytes,
            weights: layout.entries,
            tokenizerFiles: profile.tokenizerFiles,
            decode: decode,
            validation: validation
        )
        if let mutateManifest {
            manifest = try mutateManifest(manifest)
        }
        try manifest.encoded().write(to: directory.appendingPathComponent("manifest.json"))
        try Data(repeating: 0, count: Int(layout.totalBytes)).write(
            to: directory.appendingPathComponent("weights.bin")
        )
        try Data("metallib".utf8).write(to: directory.appendingPathComponent("model.metallib"))
        for tokenizerFile in profile.tokenizerFiles {
            try Data("tokenizer".utf8).write(to: directory.appendingPathComponent(tokenizerFile))
        }
        for sidecarPath in profile.sidecarPaths {
            try FileManager.default.createDirectory(
                at: directory.appendingPathComponent(sidecarPath, isDirectory: true),
                withIntermediateDirectories: true
            )
        }
    }

    private static func qwen3TTSRunnableSpecs(
        u4GroupSize: Int
    ) -> [Qwen3TTSPackageBuilder.WeightSpec] {
        let hidden = 64
        let headDim = 32
        let qDim = 64
        let kvDim = 32
        let inter = 128
        let vocab = 128

        var specs: [Qwen3TTSPackageBuilder.WeightSpec] = [
            qwen3TTSU4Weight(
                name: "talker.model.text_embedding.weight",
                shape: [vocab, hidden],
                groupSize: u4GroupSize
            ),
            .init(
                name: "talker.text_projection.linear_fc1.weight",
                shape: [hidden, hidden]
            ),
            .init(name: "talker.text_projection.linear_fc1.bias", shape: [hidden]),
            .init(
                name: "talker.text_projection.linear_fc2.weight",
                shape: [hidden, hidden]
            ),
            .init(name: "talker.text_projection.linear_fc2.bias", shape: [hidden]),
            .init(
                name: "talker.model.codec_embedding.weight",
                shape: [vocab, hidden]
            ),
            .init(
                name: "talker.codec_head.weight",
                shape: [vocab, hidden],
                dtype: .bf16
            ),
            .init(
                name: "talker.code_predictor.small_to_mtp_projection.weight",
                shape: [hidden, hidden],
                dtype: .bf16
            ),
            .init(name: "talker.code_predictor.small_to_mtp_projection.bias", shape: [hidden]),
            .init(name: "decoder.pre_conv.conv.weight", shape: [2, 2], dtype: .bf16),
            .init(name: "decoder.quantizer.rvq_first.output_proj.weight", shape: [2, 2], dtype: .bf16),
            .init(name: "decoder.pre_transformer.input_proj.weight", shape: [2, 2], dtype: .bf16),
            .init(name: "decoder.upsample.0.0.conv.weight", shape: [2, 2], dtype: .bf16),
            .init(name: "decoder.decoder.6.conv.weight", shape: [2, 2], dtype: .bf16),
        ]
        for index in 0..<15 {
            specs.append(
                .init(
                    name: "talker.code_predictor.lm_head.\(index).weight",
                    shape: [vocab, hidden],
                    dtype: .bf16
                )
            )
            specs.append(
                .init(
                    name: "talker.code_predictor.model.codec_embedding.\(index).weight",
                    shape: [vocab, hidden],
                    dtype: .bf16
                )
            )
        }

        appendQwen3TTSTrunkLayerSpecs(
            to: &specs,
            prefix: "talker.model.",
            hidden: hidden,
            headDim: headDim,
            qDim: qDim,
            kvDim: kvDim,
            inter: inter,
            groupSize: u4GroupSize
        )
        appendQwen3TTSTrunkLayerSpecs(
            to: &specs,
            prefix: "talker.code_predictor.model.",
            hidden: hidden,
            headDim: headDim,
            qDim: qDim,
            kvDim: kvDim,
            inter: inter,
            groupSize: u4GroupSize
        )
        return specs
    }

    private static func qwen3TTSManifest(
        _ manifest: Qwen3TTSManifest,
        replacingWeights weights: [Qwen3TTSManifest.Entry]
    ) -> Qwen3TTSManifest {
        Qwen3TTSManifest(
            version: manifest.version,
            blocks: manifest.blocks,
            loop: manifest.loop,
            modelName: manifest.modelName,
            pageSize: manifest.pageSize,
            pipelines: manifest.pipelines,
            eosTokens: manifest.eosTokens,
            totalBytes: manifest.totalBytes,
            weights: weights,
            tokenizerFiles: manifest.tokenizerFiles,
            decode: manifest.decode,
            validation: manifest.validation
        )
    }

    private static func qwen3TTSEntry(
        _ entry: Qwen3TTSManifest.Entry,
        dtype: Qwen3TTSPackageBuilder.WeightDType,
        groupSize: Int? = nil
    ) -> Qwen3TTSManifest.Entry {
        Qwen3TTSManifest.Entry(
            name: entry.name,
            offset: entry.offset,
            byteLength: entry.byteLength,
            shape: entry.shape,
            dtype: dtype.rawValue,
            groupSize: groupSize,
            scaleOffset: entry.scaleOffset,
            scaleByteLength: entry.scaleByteLength,
            biasOffset: entry.biasOffset,
            biasByteLength: entry.biasByteLength
        )
    }

    private static func qwen3TTSRenamedEntry(
        _ entry: Qwen3TTSManifest.Entry,
        name: String
    ) -> Qwen3TTSManifest.Entry {
        Qwen3TTSManifest.Entry(
            name: name,
            offset: entry.offset,
            byteLength: entry.byteLength,
            shape: entry.shape,
            dtype: entry.dtype,
            groupSize: entry.groupSize,
            scaleOffset: entry.scaleOffset,
            scaleByteLength: entry.scaleByteLength,
            biasOffset: entry.biasOffset,
            biasByteLength: entry.biasByteLength
        )
    }

    private static func appendQwen3TTSTrunkLayerSpecs(
        to specs: inout [Qwen3TTSPackageBuilder.WeightSpec],
        prefix: String,
        hidden: Int,
        headDim: Int,
        qDim: Int,
        kvDim: Int,
        inter: Int,
        groupSize: Int
    ) {
        let layer = "\(prefix)layers.0"
        specs.append(contentsOf: [
            qwen3TTSU4Weight(
                name: "\(layer).self_attn.q_proj.weight",
                shape: [qDim, hidden],
                groupSize: groupSize
            ),
            qwen3TTSU4Weight(
                name: "\(layer).self_attn.k_proj.weight",
                shape: [kvDim, hidden],
                groupSize: groupSize
            ),
            qwen3TTSU4Weight(
                name: "\(layer).self_attn.v_proj.weight",
                shape: [kvDim, hidden],
                groupSize: groupSize
            ),
            qwen3TTSU4Weight(
                name: "\(layer).self_attn.o_proj.weight",
                shape: [hidden, qDim],
                groupSize: groupSize
            ),
            qwen3TTSU4Weight(
                name: "\(layer).mlp.gate_proj.weight",
                shape: [inter, hidden],
                groupSize: groupSize
            ),
            qwen3TTSU4Weight(
                name: "\(layer).mlp.up_proj.weight",
                shape: [inter, hidden],
                groupSize: groupSize
            ),
            qwen3TTSU4Weight(
                name: "\(layer).mlp.down_proj.weight",
                shape: [hidden, inter],
                groupSize: groupSize
            ),
            .init(name: "\(layer).input_layernorm.weight", shape: [hidden]),
            .init(name: "\(layer).post_attention_layernorm.weight", shape: [hidden]),
            .init(name: "\(layer).self_attn.q_norm.weight", shape: [headDim]),
            .init(name: "\(layer).self_attn.k_norm.weight", shape: [headDim]),
            .init(name: "\(prefix)norm.weight", shape: [hidden]),
        ])
    }

    private static func qwen3TTSU4Weight(
        name: String,
        shape: [Int],
        groupSize: Int
    ) -> Qwen3TTSPackageBuilder.WeightSpec {
        .init(name: name, shape: shape, dtype: .u4, groupSize: groupSize)
    }

    private static func errorText(_ error: Error) -> String {
        return String(describing: error)
    }

    // MARK: - Module-IR drift mutation helpers
    //
    // The drift suites edit the authored module IR the way the retired `.cam`
    // text edits did: each mutation targets the exact lowered field the old
    // text replacement changed, then `mutatedModuleIR` re-decodes and
    // re-validates the result. Helpers throw when the targeted field is
    // missing, so a renamed fixture field fails loudly instead of letting a
    // no-op mutation pass vacuously (the `expectedCount` guard equivalent).

    private typealias IRMutation = (inout [String: Any]) throws -> Void

    private static func mutationFailure(_ message: String) -> Error {
        SmeltCAMCheckedPackageProjectionError.unsupported(message)
    }

    private static func mutateElement(
        _ container: inout [String: Any],
        _ key: String,
        label: String,
        where predicate: ([String: Any]) -> Bool,
        _ transform: (inout [String: Any]) throws -> Void
    ) throws {
        guard var elements = container[key] as? [[String: Any]] else {
            throw mutationFailure("\(label): missing '\(key)' array")
        }
        let matches = elements.indices.filter { predicate(elements[$0]) }
        guard matches.count == 1 else {
            throw mutationFailure("\(label): expected one '\(key)' match, got \(matches.count)")
        }
        try transform(&elements[matches[0]])
        container[key] = elements
    }

    private static func appendElement(
        _ container: inout [String: Any],
        _ key: String,
        label: String,
        _ element: [String: Any]
    ) throws {
        guard var elements = container[key] as? [[String: Any]] else {
            throw mutationFailure("\(label): missing '\(key)' array")
        }
        elements.append(element)
        container[key] = elements
    }

    private static func removeElement(
        _ container: inout [String: Any],
        _ key: String,
        label: String,
        where predicate: ([String: Any]) -> Bool
    ) throws {
        guard var elements = container[key] as? [[String: Any]] else {
            throw mutationFailure("\(label): missing '\(key)' array")
        }
        let matches = elements.indices.filter { predicate(elements[$0]) }
        guard matches.count == 1 else {
            throw mutationFailure("\(label): expected one '\(key)' match, got \(matches.count)")
        }
        elements.remove(at: matches[0])
        container[key] = elements
    }

    private static func updateObject(
        _ container: inout [String: Any],
        path: [String],
        label: String,
        _ transform: (inout [String: Any]) throws -> Void
    ) throws {
        guard let key = path.first else {
            try transform(&container)
            return
        }
        guard var child = container[key] as? [String: Any] else {
            throw mutationFailure("\(label): missing object '\(key)'")
        }
        try updateObject(&child, path: Array(path.dropFirst()), label: label, transform)
        container[key] = child
    }

    /// Overwrite an EXISTING leaf at `path` (last element is the leaf key);
    /// throws if any path component or the leaf itself is absent, so typo'd
    /// keys can't silently produce an unmutated IR.
    private static func setValue(
        _ container: inout [String: Any],
        path: [String],
        to value: Any,
        label: String
    ) throws {
        guard let leaf = path.last else {
            throw mutationFailure("\(label): empty mutation path")
        }
        try updateObject(&container, path: Array(path.dropLast()), label: label) { parent in
            guard parent[leaf] != nil else {
                throw Self.mutationFailure("\(label): missing leaf '\(leaf)'")
            }
            parent[leaf] = value
        }
    }

    private static func mutateSource(
        _ object: inout [String: Any],
        id: String,
        label: String,
        _ transform: (inout [String: Any]) throws -> Void
    ) throws {
        try mutateElement(&object, "sources", label: label, where: { ($0["id"] as? String) == id }, transform)
    }

    private static func setSourceLocator(
        _ object: inout [String: Any],
        id: String,
        to locator: String,
        label: String
    ) throws {
        try mutateSource(&object, id: id, label: label) { source in
            try Self.setValue(&source, path: ["locator"], to: locator, label: label)
        }
    }

    private static func mutateTransformer(
        _ object: inout [String: Any],
        block: String = "trunk",
        label: String,
        _ transform: (inout [String: Any]) throws -> Void
    ) throws {
        try mutateElement(&object, "blocks", label: label, where: { ($0["id"] as? String) == block }) { blockObject in
            try Self.updateObject(&blockObject, path: ["shape", "transformer"], label: label, transform)
        }
    }

    private static func mutateAttentionRole(
        _ object: inout [String: Any],
        role: String,
        label: String,
        _ transform: (inout [String: Any]) throws -> Void
    ) throws {
        try mutateTransformer(&object, label: label) { transformer in
            try Self.mutateElement(&transformer, "attentionByRole", label: label, where: {
                ($0["role"] as? String) == role
            }) { entry in
                try Self.updateObject(&entry, path: ["attention"], label: label, transform)
            }
        }
    }

    private static func setBlockRequirement(
        _ object: inout [String: Any],
        block: String,
        key: String,
        from oldValue: String? = nil,
        to newValue: String,
        label: String
    ) throws {
        try mutateElement(&object, "blocks", label: label, where: { ($0["id"] as? String) == block }) { blockObject in
            try Self.updateObject(&blockObject, path: ["shape"], label: label) { shape in
                try Self.mutateElement(&shape, "requirements", label: label, where: { requirement in
                    (requirement["key"] as? String) == key
                        && (oldValue == nil || (requirement["value"] as? String) == oldValue)
                }) { requirement in
                    requirement["value"] = newValue
                }
            }
        }
    }

    private static func setNodeAnnotation(
        _ object: inout [String: Any],
        node: String,
        key: String,
        to value: String,
        label: String
    ) throws {
        try mutateElement(&object, "graphNodes", label: label, where: { ($0["id"] as? String) == node }) { nodeObject in
            try Self.mutateElement(&nodeObject, "annotations", label: label, where: {
                ($0["key"] as? String) == key
            }) { annotation in
                annotation["value"] = value
            }
        }
    }

    private static func mutateFlow(
        _ object: inout [String: Any],
        label: String,
        _ transform: (inout [String: Any]) throws -> Void
    ) throws {
        try mutateElement(&object, "flows", label: label, where: { _ in true }, transform)
    }

    private static func mutateFlowPhase(
        _ object: inout [String: Any],
        label: String,
        where predicate: ([String: Any]) -> Bool,
        _ transform: (inout [String: Any]) throws -> Void
    ) throws {
        try mutateFlow(&object, label: label) { flow in
            try Self.mutateElement(&flow, "phases", label: label, where: predicate, transform)
        }
    }

    private static func nodeCalls(_ nodes: [String]) -> [[String: Any]] {
        nodes.map { ["kind": "node", "node": $0] }
    }

    private static func quantSelectorPattern(_ rule: [String: Any]) -> String? {
        (rule["selector"] as? [String: Any])?["pattern"] as? String
    }

    private static func mutateQuantRule(
        _ object: inout [String: Any],
        label: String,
        where predicate: ([String: Any]) -> Bool,
        _ transform: (inout [String: Any]) throws -> Void
    ) throws {
        try mutateElement(&object, "quantization", label: label, where: predicate, transform)
    }

    private static func mutateDefaultQuantRule(
        _ object: inout [String: Any],
        label: String,
        _ transform: (inout [String: Any]) throws -> Void
    ) throws {
        try mutateQuantRule(&object, label: label, where: { ($0["action"] as? String) == "default" }, transform)
    }

    private static func setQuantRulePattern(
        _ object: inout [String: Any],
        action: String,
        from oldPattern: String,
        to newPattern: String,
        label: String
    ) throws {
        try mutateQuantRule(&object, label: label, where: {
            ($0["action"] as? String) == action && Self.quantSelectorPattern($0) == oldPattern
        }) { rule in
            try Self.setValue(&rule, path: ["selector", "pattern"], to: newPattern, label: label)
        }
        try refreshQuantRuleDerivedFields(&object, label: label)
    }

    /// Mirror of the grammar parser's `quantRule` derivation: a non-default
    /// rule's `priority` is the selector pattern's non-wildcard character
    /// count and its `resolution` is `declared-tensor` exactly when the
    /// pattern matches a declared tensor-map selector or target. Re-run after
    /// any mutation that touches quant selector patterns or tensor maps so
    /// the mutated IR stays faithful to what reparsing the drifted `.cam`
    /// text produced (these fields feed obligation canonical IDs).
    private static func refreshQuantRuleDerivedFields(
        _ object: inout [String: Any],
        label: String
    ) throws {
        guard let tensors = object["tensors"] as? [[String: Any]] else {
            throw mutationFailure("\(label): missing 'tensors' array")
        }
        var exactPatterns: Set<String> = []
        for tensor in tensors {
            if let pattern = (tensor["selector"] as? [String: Any])?["pattern"] as? String {
                exactPatterns.insert(pattern)
            }
            if let selector = (tensor["target"] as? [String: Any])?["selector"] as? String {
                exactPatterns.insert(selector)
            }
        }
        guard var rules = object["quantization"] as? [[String: Any]] else {
            throw mutationFailure("\(label): missing 'quantization' array")
        }
        for index in rules.indices where (rules[index]["action"] as? String) != "default" {
            guard let pattern = quantSelectorPattern(rules[index]) else {
                throw mutationFailure("\(label): quant rule without selector pattern")
            }
            rules[index]["priority"] = pattern.filter { $0 != "*" }.count
            rules[index]["resolution"] = exactPatterns.contains(pattern)
                ? "declared-tensor" : "source-deferred"
        }
        object["quantization"] = rules
    }

    private static func mutateGPTQCalibration(
        _ object: inout [String: Any],
        label: String,
        _ transform: (inout [String: Any]) throws -> Void
    ) throws {
        try mutateQuantRule(&object, label: label, where: { $0["calibration"] != nil }) { rule in
            try Self.updateObject(&rule, path: ["calibration"], label: label, transform)
        }
    }

    private static func mutateTensorMap(
        _ object: inout [String: Any],
        pattern: String,
        label: String,
        _ transform: (inout [String: Any]) throws -> Void
    ) throws {
        try mutateElement(&object, "tensors", label: label, where: {
            (($0["selector"] as? [String: Any])?["pattern"] as? String) == pattern
        }, transform)
        try refreshQuantRuleDerivedFields(&object, label: label)
    }

    private static func appendTensorMap(
        _ object: inout [String: Any],
        pattern: String,
        targetBlock: String,
        targetSelector: String,
        label: String
    ) throws {
        try appendElement(&object, "tensors", label: label, [
            "owner": targetBlock,
            "selector": ["pattern": pattern, "source": "weights"],
            "source": "weights",
            "target": ["block": targetBlock, "selector": targetSelector],
        ])
        try refreshQuantRuleDerivedFields(&object, label: label)
    }

    private static func setCompileValue(
        _ object: inout [String: Any],
        key: String,
        to value: String,
        label: String
    ) throws {
        try mutateElement(&object, "compile", label: label, where: { ($0["key"] as? String) == key }) {
            $0["value"] = value
        }
    }

    private static func appendCompileConstraint(
        _ object: inout [String: Any],
        key: String,
        value: String,
        label: String
    ) throws {
        try appendElement(&object, "compile", label: label, ["key": key, "value": value])
    }

    private static func removeCompileConstraint(
        _ object: inout [String: Any],
        key: String,
        label: String
    ) throws {
        try removeElement(&object, "compile", label: label, where: { ($0["key"] as? String) == key })
    }

    private static func mutateGate(
        _ object: inout [String: Any],
        id: String,
        label: String,
        _ transform: (inout [String: Any]) throws -> Void
    ) throws {
        try mutateElement(&object, "gates", label: label, where: { ($0["id"] as? String) == id }, transform)
    }

    private static func setGateRequirementValue(
        _ object: inout [String: Any],
        gate: String,
        subject: String,
        to value: String,
        label: String
    ) throws {
        try mutateGate(&object, id: gate, label: label) { gateObject in
            try Self.mutateElement(&gateObject, "requirements", label: label, where: {
                ($0["subject"] as? String) == subject
            }) { requirement in
                requirement["value"] = value
            }
        }
    }

    /// The gate-value equivalent of the old whole-source `replacingOccurrences`
    /// for inventory CSV drifts: replaces `old` inside every gate requirement
    /// value and requires the total occurrence count to match, exactly like
    /// the retired text edit's blast radius (e.g. a model's CSV appears in
    /// two gates and both drift).
    private static func replaceInGateRequirementValues(
        _ object: inout [String: Any],
        of old: String,
        with new: String,
        expectedCount: Int,
        label: String
    ) throws {
        guard var gates = object["gates"] as? [[String: Any]] else {
            throw mutationFailure("\(label): missing 'gates' array")
        }
        var replaced = 0
        for gateIndex in gates.indices {
            guard var requirements = gates[gateIndex]["requirements"] as? [[String: Any]] else {
                continue
            }
            for index in requirements.indices {
                guard let value = requirements[index]["value"] as? String, value.contains(old) else {
                    continue
                }
                replaced += value.components(separatedBy: old).count - 1
                requirements[index]["value"] = value.replacingOccurrences(of: old, with: new)
            }
            gates[gateIndex]["requirements"] = requirements
        }
        guard replaced == expectedCount else {
            throw mutationFailure("\(label): expected \(expectedCount) occurrences of '\(old)', got \(replaced)")
        }
        object["gates"] = gates
    }

    /// A `capability` line drift touches both lowered surfaces the parser
    /// derives from it: the export's capability list and the module-level
    /// sorted union.
    private static func replaceCapability(
        _ object: inout [String: Any],
        _ old: String,
        with new: String,
        label: String
    ) throws {
        guard var capabilities = object["capabilities"] as? [String],
              let index = capabilities.firstIndex(of: old)
        else {
            throw mutationFailure("\(label): missing module capability '\(old)'")
        }
        capabilities[index] = new
        object["capabilities"] = capabilities.sorted()
        try mutateElement(&object, "exports", label: label, where: {
            (($0["capabilities"] as? [String]) ?? []).contains(old)
        }) { export in
            guard var exportCapabilities = export["capabilities"] as? [String],
                  let exportIndex = exportCapabilities.firstIndex(of: old)
            else {
                throw Self.mutationFailure("\(label): missing export capability '\(old)'")
            }
            exportCapabilities[exportIndex] = new
            export["capabilities"] = exportCapabilities.sorted()
        }
    }

    private static func setFeedbackEdgeSourceToGraphValue(
        _ object: inout [String: Any],
        named name: String,
        label: String
    ) throws {
        try mutateElement(&object, "feedbackEdges", label: label, where: { _ in true }) { edge in
            guard edge["from"] != nil else {
                throw Self.mutationFailure("\(label): feedback edge missing 'from'")
            }
            edge["from"] = ["kind": "graphValue", "name": name]
        }
    }

    /// The `<input> -> ignored(native; tag ignored) -> ignored_text` graph
    /// line as the parser lowers it: one unused native node consuming the
    /// module text input, plus its two edges (module input into the node,
    /// node output into a dangling graph value).
    private static func addUnusedGraphNode(
        _ object: inout [String: Any],
        consuming input: String,
        label: String
    ) throws {
        let inputType: [String: Any] = ["name": "text", "attributes": ["encoding": "utf8"]]
        let outputType: [String: Any] = ["name": "ignored_text", "attributes": [String: Any]()]
        try appendElement(&object, "graphNodes", label: label, [
            "annotations": [["key": "tag", "value": "ignored"]],
            "id": "ignored",
            "implementation": "native",
            "inputs": [["name": input, "optional": false, "type": inputType] as [String: Any]],
            "outputs": [["name": "ignored_text", "optional": false, "type": outputType] as [String: Any]],
        ])
        try appendElement(&object, "graphEdges", label: label, [
            "from": ["kind": "moduleInput", "name": input],
            "to": ["kind": "nodePort", "node": "ignored", "port": input],
            "type": inputType,
        ])
        try appendElement(&object, "graphEdges", label: label, [
            "from": ["kind": "nodePort", "node": "ignored", "port": "ignored_text"],
            "to": ["kind": "graphValue", "name": "ignored_text"],
            "type": outputType,
        ])
    }

    private static func isLowercaseSHA256(_ value: String) -> Bool {
        let allowed = Set("0123456789abcdef")
        return value.count == 64 && value.allSatisfy { allowed.contains($0) }
    }

    private static func featureAdmission(from cam: SmeltCAMIR) throws -> SmeltCAMFeatureAdmission {
        try SmeltCAMFeatureAdmission(descriptor: SmeltCAMPackageDescriptor(
            from: cam
        ))
    }

    private struct ObligationIdentity: Sendable {
        let code: String
        let parameters: [String: String]

        init(_ code: String, parameters: [String: String] = [:]) {
            self.code = code
            self.parameters = parameters
        }

        var summary: String {
            let parameterText = parameters
                .map { "\($0.key)=\($0.value)" }
                .sorted()
                .joined(separator: ",")
            return parameterText.isEmpty ? code : "\(code) {\(parameterText)}"
        }
    }

    private struct ObligationDriftExpectation: Sendable {
        let baseline: ObligationIdentity
        let drifted: ObligationIdentity
    }

    private struct ObligationDriftCase {
        let label: String
        let mutation: IRMutation
        let changed: [ObligationDriftExpectation]
        let stable: [ObligationIdentity]

        init(
            label: String,
            mutation: @escaping IRMutation,
            changed: [ObligationDriftExpectation],
            stable: [ObligationIdentity] = []
        ) {
            self.label = label
            self.mutation = mutation
            self.changed = changed
            self.stable = stable
        }
    }

    private static func obligation(
        _ identity: ObligationIdentity,
        in admission: SmeltCAMFeatureAdmission,
        label: String
    ) throws -> SmeltCAMFeatureAdmission.FeatureRequirement {
        let matches = admission.requiredObligations.filter { obligation in
            obligation.code == identity.code
                && identity.parameters.allSatisfy {
                    obligation.parameters[$0.key] == $0.value
                }
        }
        #expect(
            matches.count == 1,
            "\(label): expected one \(identity.summary), got \(matches.map(\.checkSummary))"
        )
        guard matches.count == 1 else {
            throw SmeltCAMCheckedPackageProjectionError.unsupported(
                "\(label): obligation identity mismatch for \(identity.summary)"
            )
        }
        return matches[0]
    }

    private static func obligationCountsByCode(
        in admission: SmeltCAMFeatureAdmission
    ) -> [String: Int] {
        var counts: [String: Int] = [:]
        for obligation in admission.requiredObligations {
            counts[obligation.code, default: 0] += 1
        }
        return counts
    }

    private static func obligationIDs(
        _ code: String,
        in admission: SmeltCAMFeatureAdmission
    ) -> Set<String> {
        Set(admission.requiredObligations.filter { $0.code == code }.map(\.canonicalID))
    }

    private static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }
}
