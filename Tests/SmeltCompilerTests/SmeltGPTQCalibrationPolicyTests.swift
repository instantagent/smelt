import Foundation
import Testing
@testable import SmeltCompiler
import SmeltSchema

@Suite struct SmeltGPTQCalibrationPolicyTests {
    @Test func resolvesQwen3TTSPromptLinePolicyFromPackageSpecJSON() throws {
        let data = try JSONEncoder().encode(Self.qwenTTSSpec(calibration: Self.calibration(
            maxSamples: 3,
            maxLayersPerPass: 4
        )))

        let policy = try SmeltGPTQCalibrationPolicy.qwen3TTSPromptLines(
            fromPackageSpecJSON: data,
            defaultLayersPerPass: 12
        )

        #expect(policy.corpusPath == "calibration/prompts.txt")
        #expect(policy.maxSamples == 3)
        #expect(policy.layersPerPass == 4)
        #expect(policy.captureArtifactPaths == [
            "trunk-mtp/gptq_capture_points.json",
            "trunk/gptq_capture_points.json",
        ])
    }

    @Test func capsPolicyLayersPerPassAtRunnerDefault() throws {
        let policy = try SmeltGPTQCalibrationPolicy.qwen3TTSPromptLines(
            from: Self.qwenTTSSpec(calibration: Self.calibration(maxLayersPerPass: 64)),
            defaultLayersPerPass: 12
        )

        #expect(policy.layersPerPass == 12)
    }

    @Test func rejectsUnsupportedQwen3TTSCalibrationFields() throws {
        #expect(throws: SmeltGPTQCalibrationPolicyError.self) {
            try SmeltGPTQCalibrationPolicy.qwen3TTSPromptLines(
                from: Self.qwenTTSSpec(calibration: Self.calibration(renderPolicy: "chat-template")),
                defaultLayersPerPass: 12
            )
        }
        #expect(throws: SmeltGPTQCalibrationPolicyError.self) {
            try SmeltGPTQCalibrationPolicy.qwen3TTSPromptLines(
                from: Self.qwenTTSSpec(calibration: Self.calibration(maxTokens: 256)),
                defaultLayersPerPass: 12
            )
        }
        #expect(throws: SmeltGPTQCalibrationPolicyError.self) {
            try SmeltGPTQCalibrationPolicy.qwen3TTSPromptLines(
                from: Self.qwenTTSSpec(calibration: Self.calibration(maxBytes: 1_000_000)),
                defaultLayersPerPass: 12
            )
        }
        #expect(throws: SmeltGPTQCalibrationPolicyError.self) {
            try SmeltGPTQCalibrationPolicy.qwen3TTSPromptLines(
                from: Self.qwenTTSSpec(calibration: Self.calibration(sideInputs: [
                    .init(id: "imatrix", path: "calibration/imatrix.dat", role: "imatrix"),
                ])),
                defaultLayersPerPass: 12
            )
        }
    }

    @Test func rejectsQwen3TTSCalibrationWithoutActivationCaptureArtifacts() throws {
        #expect(throws: SmeltGPTQCalibrationPolicyError.self) {
            try SmeltGPTQCalibrationPolicy.qwen3TTSPromptLines(
                from: Self.qwenTTSSpec(calibration: Self.calibration(captureArtifacts: [])),
                defaultLayersPerPass: 12
            )
        }
    }

    @Test func rejectsUnsupportedQwen3TTSCalibrationCaptureArtifactRoles() throws {
        #expect(throws: SmeltGPTQCalibrationPolicyError.self) {
            try SmeltGPTQCalibrationPolicy.qwen3TTSPromptLines(
                from: Self.qwenTTSSpec(calibration: Self.calibration(captureArtifacts: [
                    .init(
                        id: "text-sidecar",
                        path: "trunk/text_sidecar.json",
                        role: "text-sidecar"
                    ),
                ])),
                defaultLayersPerPass: 12
            )
        }
    }

    @Test func rejectsPackageSpecWithoutCalibrationPolicy() throws {
        #expect(throws: SmeltGPTQCalibrationPolicyError.self) {
            try SmeltGPTQCalibrationPolicy.qwen3TTSPromptLines(
                from: Self.qwenTTSSpec(calibration: nil),
                defaultLayersPerPass: 12
            )
        }
    }

    @Test func resolvesLLMImatrixSideInputPolicyFromPackageSpecJSON() throws {
        let data = try JSONEncoder().encode(Self.llmSpec(calibration: Self.llmImatrixCalibration()))

        let policy = try SmeltLLMImatrixCalibrationPolicy.fromPackageSpecJSON(data)

        #expect(policy.imatrixPath == "calibration/qwen.smeltim")
    }

    @Test func rejectsLLMImatrixPolicyWithoutExactlyOneImatrixInput() throws {
        #expect(throws: SmeltLLMImatrixCalibrationPolicyError.self) {
            try SmeltLLMImatrixCalibrationPolicy.from(Self.llmSpec(
                calibration: Self.llmImatrixCalibration(sideInputs: [
                    .init(id: "wrong", path: "calibration/sidecar.dat", role: "activation-stats"),
                ])
            ))
        }

        #expect(throws: SmeltLLMImatrixCalibrationPolicyError.self) {
            try SmeltLLMImatrixCalibrationPolicy.from(Self.llmSpec(
                calibration: Self.llmImatrixCalibration(sideInputs: [
                    .init(id: "a", path: "calibration/a.smeltim", role: "imatrix"),
                    .init(id: "b", path: "calibration/b.smeltim", role: "imatrix"),
                ])
            ))
        }
    }

    @Test func rejectsUnsupportedLLMImatrixCalibrationFields() throws {
        #expect(throws: SmeltLLMImatrixCalibrationPolicyError.self) {
            try SmeltLLMImatrixCalibrationPolicy.from(Self.llmSpec(
                calibration: Self.llmImatrixCalibration(
                    corpus: .init(
                        source: "weights",
                        path: "calibration/corpus.txt",
                        renderPolicy: "chat-template"
                    )
                )
            ))
        }

        #expect(throws: SmeltLLMImatrixCalibrationPolicyError.self) {
            try SmeltLLMImatrixCalibrationPolicy.from(Self.llmSpec(
                calibration: Self.llmImatrixCalibration(captureArtifacts: [
                    .init(id: "captures", path: "gptq_capture_points.json", role: "activation-capture"),
                ])
            ))
        }

        #expect(throws: SmeltLLMImatrixCalibrationPolicyError.self) {
            try SmeltLLMImatrixCalibrationPolicy.from(Self.llmSpec(
                calibration: Self.llmImatrixCalibration(
                    resourceBounds: .init(maxBytes: 1_000_000)
                )
            ))
        }

        #expect(throws: SmeltLLMImatrixCalibrationPolicyError.self) {
            try SmeltLLMImatrixCalibrationPolicy.from(Self.llmSpec(
                calibration: Self.llmImatrixCalibration(equalityGates: [
                    .init(
                        id: "heldout",
                        candidate: "candidate.json",
                        reference: "reference.json",
                        metric: "argmax"
                    ),
                ])
            ))
        }
    }

    @Test func resolvesTextRuntimeGPTQTokenIDPolicyFromPackageSpecJSON() throws {
        let data = try JSONEncoder().encode(Self.llmSpec(calibration: Self.llmRuntimeGPTQCalibration(
            maxSamples: 8,
            maxTokens: 9,
            resourceMaxSamples: 2,
            resourceMaxTokens: 3,
            maxLayersPerPass: 5
        )))

        let policy = try SmeltRuntimeGPTQCalibrationPolicy.textTokenIDLines(
            fromPackageSpecJSON: data,
            defaultLayersPerPass: 12
        )

        #expect(policy.tokenIDsPath == "calibration/tokens.txt")
        #expect(policy.capturePointsPath == "gptq_capture_points.json")
        #expect(policy.maxSamples == 2)
        #expect(policy.maxTokens == 3)
        #expect(policy.layersPerPass == 5)
    }

    @Test func loadsTextRuntimeGPTQTokenIDCorpusRelativeToPolicy() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-gptq-policy-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let calibrationDir = root.appendingPathComponent("calibration")
        try FileManager.default.createDirectory(
            at: calibrationDir,
            withIntermediateDirectories: true
        )
        try Data("1 2 3 4\n5,6,7\n".utf8).write(
            to: calibrationDir.appendingPathComponent("tokens.txt")
        )
        let specPath = root.appendingPathComponent("package-spec.json")
        let spec = Self.llmSpec(calibration: Self.llmRuntimeGPTQCalibration(
            maxSamples: 1,
            maxTokens: 3,
            maxLayersPerPass: 4
        ))
        try JSONEncoder().encode(spec).write(to: specPath)

        let inputs = try SmeltRuntimeGPTQCalibrationPolicy.tokenIDLinesFromPackageSpecJSON(
            at: specPath.path,
            defaultLayersPerPass: 12
        )

        #expect(inputs.policy.layersPerPass == 4)
        #expect(inputs.tokenIDs == [[1, 2, 3]])
    }

    @Test func rejectsUnsupportedTextRuntimeGPTQCalibrationFields() throws {
        #expect(throws: SmeltGPTQCalibrationPolicyError.self) {
            try SmeltRuntimeGPTQCalibrationPolicy.textTokenIDLines(
                from: Self.llmSpec(calibration: Self.llmRuntimeGPTQCalibration(
                    renderPolicy: "chat-template"
                )),
                defaultLayersPerPass: 12
            )
        }
        #expect(throws: SmeltGPTQCalibrationPolicyError.self) {
            try SmeltRuntimeGPTQCalibrationPolicy.textTokenIDLines(
                from: Self.llmSpec(calibration: Self.llmRuntimeGPTQCalibration(
                    captureArtifacts: []
                )),
                defaultLayersPerPass: 12
            )
        }
        #expect(throws: SmeltGPTQCalibrationPolicyError.self) {
            try SmeltRuntimeGPTQCalibrationPolicy.textTokenIDLines(
                from: Self.llmSpec(calibration: Self.llmRuntimeGPTQCalibration(
                    sideInputs: [
                        .init(id: "imatrix", path: "calibration/qwen.smeltim", role: "imatrix"),
                    ]
                )),
                defaultLayersPerPass: 12
            )
        }
    }

    private static func calibration(
        renderPolicy: String = SmeltGPTQCalibrationPolicy.qwen3TTSPromptLinesRenderPolicy,
        maxSamples: Int? = nil,
        maxTokens: Int? = nil,
        maxLayersPerPass: Int? = nil,
        maxBytes: UInt64? = nil,
        captureArtifacts: [SmeltPackageSpec.QuantizationPlan.Calibration.Artifact] = [
            .init(
                id: "talker-captures",
                path: "trunk/gptq_capture_points.json",
                role: SmeltGPTQCalibrationPolicy.activationCaptureRole
            ),
            .init(
                id: "mtp-captures",
                path: "trunk-mtp/gptq_capture_points.json",
                role: SmeltGPTQCalibrationPolicy.activationCaptureRole
            ),
        ],
        sideInputs: [SmeltPackageSpec.QuantizationPlan.Calibration.Artifact] = []
    ) -> SmeltPackageSpec.QuantizationPlan.Calibration {
        .init(
            corpus: .init(
                source: "weights",
                path: "calibration/prompts.txt",
                renderPolicy: renderPolicy,
                maxSamples: maxSamples,
                maxTokens: maxTokens
            ),
            captureArtifacts: captureArtifacts,
            sideInputs: sideInputs,
            resourceBounds: .init(
                maxSamples: maxSamples,
                maxLayersPerPass: maxLayersPerPass,
                maxBytes: maxBytes
            )
        )
    }

    private static func qwenTTSSpec(
        calibration: SmeltPackageSpec.QuantizationPlan.Calibration?
    ) -> SmeltPackageSpec {
        let graph = SmeltBlockGraph.qwen3TTSCompiledTrunkNativeFrontEnd
        return SmeltPackageSpec(
            packageName: "qwen3-tts.cam",
            modelName: "qwen3-tts",
            sources: [
                .init(id: "weights", kind: .localDirectory, path: "checkpoint")
            ],
            blocks: graph,
            loop: .qwen3TTS,
            runtime: .forGraph(
                architecture: SmeltRuntimeGraphPolicy.sidecarTextToCodecAudio.rawValue,
                commands: SmeltPackageSpec.RuntimeDescriptor.Command.allCases,
                graph: graph
            ),
            architectureConfig: .object(["page_size": .int(16_384)]),
            tensors: [
                .init(
                    source: "weights",
                    name: "talker.model.layers.0.self_attn.q_proj.weight",
                    canonicalName: "talker.model.layers.0.self_attn.q_proj.weight",
                    block: "talker",
                    sourceDType: .bf16,
                    storedDType: .u4,
                    shape: [2, 2]
                )
            ],
            quantization: .init(format: .u4, groupSize: 64, calibration: calibration),
            sidecars: SmeltPackageSidecarProfiles.qwen3TTSRunnableHeadlessTrunks
                .map(\.sidecar),
            artifacts: [
                .init(id: "weights", path: "weights.bin", role: "weights"),
            ],
            outputFiles: .init(files: [
                "manifest.json",
                "merges.txt",
                "tokenizer.json",
                "trunk",
                "trunk-mtp",
                "weights.bin",
            ]),
            tokenizer: .init(format: "byte-bpe", files: ["merges.txt", "tokenizer.json"]),
            inference: .init(maxTokens: 2048, eosTokens: [2150]),
            decode: .init(
                sampler: .init(mode: .sample, temperature: 0.7, topK: 12),
                maxSteps: 2048
            )
        )
    }

    private static func llmImatrixCalibration(
        corpus: SmeltPackageSpec.QuantizationPlan.Calibration.Corpus? = nil,
        captureArtifacts: [SmeltPackageSpec.QuantizationPlan.Calibration.Artifact] = [],
        sideInputs: [SmeltPackageSpec.QuantizationPlan.Calibration.Artifact] = [
            .init(id: "imatrix", path: "calibration/qwen.smeltim", role: "imatrix"),
        ],
        stagedPackages: [SmeltPackageSpec.QuantizationPlan.Calibration.Artifact] = [],
        resourceBounds: SmeltPackageSpec.QuantizationPlan.Calibration.ResourceBounds? = nil,
        equalityGates: [SmeltPackageSpec.QuantizationPlan.Calibration.EqualityGate] = []
    ) -> SmeltPackageSpec.QuantizationPlan.Calibration {
        .init(
            corpus: corpus,
            captureArtifacts: captureArtifacts,
            sideInputs: sideInputs,
            stagedPackages: stagedPackages,
            resourceBounds: resourceBounds,
            equalityGates: equalityGates
        )
    }

    private static func llmRuntimeGPTQCalibration(
        renderPolicy: String = SmeltRuntimeGPTQCalibrationPolicy.textTokenIDLinesRenderPolicy,
        maxSamples: Int? = nil,
        maxTokens: Int? = nil,
        resourceMaxSamples: Int? = nil,
        resourceMaxTokens: Int? = nil,
        maxLayersPerPass: Int? = nil,
        captureArtifacts: [SmeltPackageSpec.QuantizationPlan.Calibration.Artifact] = [
            .init(id: "captures", path: "gptq_capture_points.json", role: "activation-capture"),
        ],
        sideInputs: [SmeltPackageSpec.QuantizationPlan.Calibration.Artifact] = []
    ) -> SmeltPackageSpec.QuantizationPlan.Calibration {
        .init(
            corpus: .init(
                source: "weights",
                path: "calibration/tokens.txt",
                renderPolicy: renderPolicy,
                maxSamples: maxSamples,
                maxTokens: maxTokens
            ),
            captureArtifacts: captureArtifacts,
            sideInputs: sideInputs,
            resourceBounds: .init(
                maxSamples: resourceMaxSamples,
                maxTokens: resourceMaxTokens,
                maxLayersPerPass: maxLayersPerPass
            )
        )
    }

    private static func llmSpec(
        calibration: SmeltPackageSpec.QuantizationPlan.Calibration?
    ) -> SmeltPackageSpec {
        let graph = SmeltBlockGraph.tokenFeedbackText
        return SmeltPackageSpec(
            packageName: "qwen.cam",
            modelName: "qwen",
            sources: [
                .init(id: "weights", kind: .localDirectory, path: "checkpoint")
            ],
            blocks: graph,
            loop: .tokenFeedbackText,
            runtime: .forGraph(
                architecture: "text-generation",
                commands: SmeltPackageSpec.RuntimeDescriptor.Command.allCases,
                graph: graph
            ),
            architectureConfig: .object(["hidden_size": .int(2)]),
            tensors: [
                .init(
                    source: "weights",
                    name: "layers.0.mlp.down_proj.weight",
                    canonicalName: "layers_0_down_proj_weight",
                    block: "trunk",
                    sourceDType: .bf16,
                    storedDType: .u4,
                    shape: [2, 2]
                )
            ],
            quantization: .init(format: .u4, groupSize: 128, calibration: calibration),
            artifacts: [
                .init(id: "weights", path: "weights.bin", role: "weights"),
                .init(id: "generated", path: "SmeltGenerated.swift", role: "generated-swift"),
            ],
            outputFiles: .init(files: [
                "manifest.json",
                "SmeltGenerated.swift",
                "tokenizer.json",
                "weights.bin",
            ]),
            tokenizer: .init(format: "tokenizer-json", files: ["tokenizer.json"]),
            inference: .init(
                maxTokens: 32,
                eosTokens: [1],
                chatTemplate: "channel-turns",
                thinkingPolicy: .disabled
            ),
            decode: .init(sampler: .init(mode: .greedy), maxSteps: 32)
        )
    }
}
