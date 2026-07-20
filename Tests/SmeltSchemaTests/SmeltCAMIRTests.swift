import Foundation
import Testing
@testable import SmeltSchema

@Suite struct SmeltCAMIRTests {

    @Test func canonicalJSONAndHashAreStable() throws {
        let ir = try Self.fixture().validated()
        let firstJSON = try ir.canonicalJSONData()
        let secondJSON = try ir.canonicalJSONData()
        #expect(firstJSON == secondJSON)
        #expect(String(data: firstJSON, encoding: .utf8)?.contains("qwen3_tts") == true)
        #expect(try ir.semanticSHA256() == Self.fixture().semanticSHA256())
    }

    @Test func unorderedDeclarationsCanonicalizeToSameBytes() throws {
        let canonical = try Self.fixture().canonicalJSONData()
        let shuffled = Self.fixture(
            imports: [Self.voiceImport],
            sources: [Self.normSource, Self.tokenizerSource, Self.weightsSource],
            capabilities: ["run.stream", "run.generate"],
            graphNodes: [Self.detokenizerNode, Self.trunkNode, Self.samplerNode, Self.tokenizerNode],
            graphEdges: [Self.hiddenToSampler, Self.trunkToHidden, Self.promptToTokenizer],
            quantization: [Self.normQuant, Self.defaultQuant]
        )
        #expect(try shuffled.canonicalJSONData() == canonical)
        #expect(try shuffled.semanticSHA256() == Self.fixture().semanticSHA256())
    }

    @Test func sourceIntegrityPinsRoundTripThroughDescriptor() throws {
        let digest = String(repeating: "a", count: 64)
        let pinned = SmeltCAMIR.Source(
            id: "weights",
            kind: "hf-file",
            locator: "repo/model.safetensors.index.json",
            revision: "exact-revision",
            checkpointMap: "hf.fixture",
            sha256: digest,
            byteCount: 1_234
        )
        let ir = try Self.fixture(sources: [pinned, Self.tokenizerSource, Self.normSource]).validated()
        let decoded = try JSONDecoder().decode(
            SmeltCAMIR.self,
            from: ir.canonicalJSONData()
        )
        let descriptor = try SmeltCAMPackageDescriptor(from: decoded)
        let source = try #require(descriptor.sourceReferences.first { $0.sourceID == "weights" })

        #expect(source.sha256 == digest)
        #expect(source.byteCount == 1_234)
    }

    @Test func invalidSourceIntegrityPinIsRejected() {
        let invalid = SmeltCAMIR.Source(
            id: "weights",
            kind: "hf",
            locator: "repo",
            sha256: "not-a-digest"
        )
        let ir = Self.fixture(sources: [invalid, Self.tokenizerSource, Self.normSource])
        #expect(throws: SmeltCAMIRError.self) {
            _ = try ir.validated()
        }
    }

    @Test func flowPhasesRemainOrderSensitive() throws {
        let flow = SmeltCAMIR.Flow(
            id: Self.generateFlow.id,
            phases: Self.generateFlow.phases.reversed(),
            emit: Self.generateFlow.emit,
            stop: Self.generateFlow.stop
        )
        let reordered = Self.fixture(flows: [flow])
        #expect(try reordered.semanticSHA256() != Self.fixture().semanticSHA256())
    }

    @Test func exportABIHashIgnoresInternalGraphChanges() throws {
        let baseline = try Self.fixture().exportABISHA256()
        let changedInternals = Self.fixture(
            graphNodes: [
                Self.tokenizerNode,
                Self.samplerNode,
                Self.detokenizerNode,
                .init(
                    id: "trunk",
                    implementation: .compiled,
                    block: "trunk",
                    inputs: Self.trunkNode.inputs,
                    outputs: Self.trunkNode.outputs,
                    annotations: [.init("artifact", "sidecar")]
                ),
            ],
            graphEdges: [Self.promptToTokenizer, Self.trunkToHidden, Self.hiddenToSampler]
        )
        #expect(try changedInternals.exportABISHA256() == baseline)

        let changedABI = Self.fixture(exports: [
            .init(
                id: "generate",
                inputs: Self.generateExport.inputs,
                outputs: [
                    .init(name: "tokens", type: .init("tokens")),
                ],
                capabilities: Self.generateExport.capabilities,
                gates: Self.generateExport.gates
            ),
        ])
        #expect(try changedABI.exportABISHA256() != baseline)
    }

    @Test func importedABIHashIsSelfVerifying() {
        let stale = SmeltCAMIR.Import(
            alias: "voice",
            moduleID: "qwen3_tts",
            canonicalURI: "file://elsewhere/qwen3_tts.cam",
            irSHA256: String(repeating: "3", count: 64),
            exportABISHA256: String(repeating: "4", count: 64),
            exportABI: [Self.synthExport]
        )
        let ir = Self.fixture(imports: [stale])
        #expect(throws: SmeltCAMIRError.self) {
            _ = try ir.validated()
        }
    }

    @Test func importedCanonicalURIDoesNotAffectSemanticHash() throws {
        let first = Self.voiceImport
        let second = SmeltCAMIR.Import(
            alias: first.alias,
            moduleID: first.moduleID,
            canonicalURI: "file://tmp/renamed/qwen3_tts.cam",
            irSHA256: first.irSHA256,
            exportABISHA256: first.exportABISHA256,
            exportABI: first.exportABI,
            parameters: first.parameters
        )
        #expect(first != second)
        #expect(
            try Self.fixture(imports: [first]).semanticSHA256()
                == Self.fixture(imports: [second]).semanticSHA256()
        )
    }

    @Test func importedGraphNodesMirrorReferencedExportABI() throws {
        let reordered = Self.importedSynthNode(inputs: [Self.speakerPort, Self.textPort])
        _ = try Self.importedNodeIR(node: reordered).validated()

        let missingOptional = Self.importedSynthNode(inputs: [Self.textPort])
        try Self.expectValidationFailure(
            Self.importedNodeIR(node: missingOptional),
            contains: "missing input port 'speaker'"
        )

        let extra = Self.importedSynthNode(inputs: [
            Self.speakerPort,
            Self.textPort,
            .init(name: "style", type: .init("voice-style"), optional: true),
        ])
        try Self.expectValidationFailure(
            Self.importedNodeIR(node: extra),
            contains: "extra input port 'style'"
        )

        let optionalDrift = Self.importedSynthNode(inputs: [
            .init(name: "speaker", type: .init("voice-id")),
            Self.textPort,
        ])
        try Self.expectValidationFailure(
            Self.importedNodeIR(node: optionalDrift),
            contains: "expected optional=true, got optional=false"
        )

        let typeDrift = Self.importedSynthNode(outputs: [
            .init(name: "audio", type: .init("pcm", attributes: ["dtype": "f32", "rate": "16khz"])),
        ])
        try Self.expectValidationFailure(
            Self.importedNodeIR(node: typeDrift),
            contains: "expected pcm[dtype=f32,rate=24khz], got pcm[dtype=f32,rate=16khz]"
        )
    }

    @Test func importedABIsRejectDuplicateExportsAndPortsBeforeLookup() throws {
        let duplicateExports = SmeltCAMIR.Import(
            alias: "voice",
            moduleID: "qwen3_tts",
            irSHA256: String(repeating: "3", count: 64),
            exportABISHA256: try SmeltCAMIR.exportABISHA256(
                for: [Self.speakerSynthExport, Self.speakerSynthExport]
            ),
            exportABI: [Self.speakerSynthExport, Self.speakerSynthExport]
        )
        try Self.expectValidationFailure(
            Self.importedNodeIR(imports: [duplicateExports]),
            contains: "duplicate or empty export ABI for import voice"
        )

        let duplicatePortsExport = SmeltCAMIR.Export(
            id: "synth",
            inputs: [Self.textPort, Self.textPort],
            outputs: [Self.audio24Port]
        )
        let duplicatePortsImport = SmeltCAMIR.Import(
            alias: "voice",
            moduleID: "qwen3_tts",
            irSHA256: String(repeating: "3", count: 64),
            exportABISHA256: try SmeltCAMIR.exportABISHA256(for: [duplicatePortsExport]),
            exportABI: [duplicatePortsExport]
        )
        try Self.expectValidationFailure(
            Self.importedNodeIR(imports: [duplicatePortsImport]),
            contains: "duplicate or empty input port for imported export synth"
        )
    }

    @Test func compiledNodesRequireDeclaredBlocks() {
        let ir = Self.fixture(graphNodes: [
            .init(
                id: "trunk",
                implementation: .compiled,
                block: "missing",
                inputs: Self.trunkNode.inputs,
                outputs: Self.trunkNode.outputs
            ),
            Self.tokenizerNode,
            Self.samplerNode,
            Self.detokenizerNode,
        ])
        #expect(throws: SmeltCAMIRError.self) {
            _ = try ir.validated()
        }
    }

    @Test func graphEdgesValidateEndpointDirectionAndPorts() {
        let bad = Self.fixture(graphEdges: [
            .init(from: .node("trunk", "tokens"), to: .graphValue("hidden")),
        ])
        #expect(throws: SmeltCAMIRError.self) {
            _ = try bad.validated()
        }
    }

    @Test func graphEdgesRejectDeclaredValueTypeDrift() throws {
        let badFamily = Self.fixture(graphEdges: [
            .init(
                from: .moduleInput("prompt"),
                to: .node("tokenizer", "prompt"),
                type: .init("pcm", attributes: ["dtype": "f32", "rate": "24khz"])
            ),
        ])
        try Self.expectValidationFailure(
            badFamily,
            contains: "declares value type pcm"
        )

        let badAttribute = Self.fixture(graphEdges: [
            .init(
                from: .moduleInput("prompt"),
                to: .node("tokenizer", "prompt"),
                type: .init("text", attributes: ["encoding": "utf16"])
            ),
        ])
        try Self.expectValidationFailure(
            badAttribute,
            contains: "declares value type text[encoding=utf16]"
        )
    }

    @Test func graphValuesCarryProducerTypesAcrossEdges() throws {
        let bad = Self.fixture(graphEdges: [
            .init(
                from: .moduleInput("prompt"),
                to: .graphValue("prompt_text"),
                type: .init("text", attributes: ["encoding": "utf8"])
            ),
            .init(
                from: .graphValue("prompt_text"),
                to: .node("trunk", "tokens"),
                type: .init("tokens")
            ),
        ])
        try Self.expectValidationFailure(
            bad,
            contains: "graph edge graphValue:prompt_text->nodePort:trunk.tokens"
        )
    }

    @Test func flowCallsValidateGraphNodesAndImportedExports() {
        let bad = Self.fixture(flows: [
            .init(
                id: "generate",
                phases: [
                    .init(role: .step, label: "decode", calls: [.node("missing")]),
                ],
                emit: [.moduleOutput("text")],
                stop: [.init(kind: .hostCancel)]
            ),
        ])
        #expect(throws: SmeltCAMIRError.self) {
            _ = try bad.validated()
        }
    }

    @Test func structuredGateRequirementsAreValidated() throws {
        let gate = try Self.fixture().validated().gates.first { $0.id == "startup" }
        #expect(gate?.from?.kind == .flowAccepted)
        #expect(gate?.to?.predicates == [
            .init(subject: "tokens", relation: .greaterThanOrEqual, value: "1"),
        ])
        #expect(gate?.requirements == [
            .init(subject: "elapsed", relation: .lessThanOrEqual, value: "100", unit: "ms"),
        ])

        let bad = Self.fixture(gates: [
            .init(
                id: "startup",
                requirements: [.init(subject: "elapsed", relation: .lessThanOrEqual, value: "100", unit: "ms")]
            ),
        ])
        #expect(throws: SmeltCAMIRError.self) {
            _ = try bad.validated()
        }
    }

    @Test func ds4QuantCalibrationIsStructuredAndCanonical() throws {
        let ir = try Self.ds4Fixture().validated()
        let calibrationRule = try #require(
            ir.quantization.first { $0.calibration?.method == .gptq }
        )
        #expect(calibrationRule.calibration?.corpus.source == "calibration_prompts")
        #expect(calibrationRule.calibration?.captures == ["trunk.attention", "trunk.experts"])
        #expect(calibrationRule.calibration?.layersPerPass == 1)
        #expect(calibrationRule.calibration?.requirements == [
            .init(subject: "cosine", relation: .greaterThanOrEqual, value: "0.995"),
        ])
        #expect(calibrationRule.storage?.format == .gptq)
        #expect(calibrationRule.storage?.groupSize == 128)
    }

    @Test func defaultQuantRuleMayOverlapSpecificOverrides() throws {
        let ir = Self.fixture(quantization: [
            .init(
                selector: .init("*"),
                action: .default,
                storage: .init(format: .lutU4, groupSize: 16)
            ),
            .init(selector: .init("*_norm_weight"), action: .preserve),
        ])
        _ = try ir.validated()
    }

    @Test func ambiguousNonDefaultQuantOverlapsNeedPriority() {
        let ir = Self.fixture(quantization: [
            Self.normQuant,
            .init(
                selector: .init("*_norm_weight"),
                action: .store,
                storage: .init(format: .fp16)
            ),
        ])
        #expect(throws: SmeltCAMIRError.self) {
            _ = try ir.validated()
        }
    }

    @Test func quantRulesMustTargetDeclaredTensors() {
        let ir = Self.fixture(quantization: [
            Self.defaultQuant,
            .init(selector: .init("missing.*"), action: .preserve),
        ])
        #expect(throws: SmeltCAMIRError.self) {
            _ = try ir.validated()
        }
    }

    @Test func diagnosticSpansDoNotAffectSemanticHash() throws {
        let ir = Self.fixture()
        let first = SmeltCAMIRDiagnostics(spans: [
            "module": .init(file: "Examples/CAM/qwen35_text.cam", line: 1, column: 1),
        ])
        let second = SmeltCAMIRDiagnostics(spans: [
            "module": .init(file: "tmp/reformatted.cam", line: 99, column: 4),
        ])
        #expect(first != second)
        #expect(try ir.semanticSHA256() == Self.fixture().semanticSHA256())
    }

    @Test func graphTagsAreAnnotationsNotHandlerSelectors() throws {
        let node = try Self.fixture().validated().graphNodes.first { $0.id == "trunk" }
        #expect(node?.annotations.contains(.init("tag", "decode-core")) == true)
        #expect(node?.implementation == .compiled)
        #expect(node?.block == "trunk")
    }

    @Test func canonicalJSONDoesNotPreserveRawGateOrShapeMiniLanguages() throws {
        let json = try #require(String(data: try Self.fixture().canonicalJSONData(), encoding: .utf8))
        #expect(!json.contains("elapsed <= 100 ms"))
        #expect(!json.contains("[delta,delta,delta,attn]*6"))
        #expect(!json.contains("sampler.tokens"))
    }

    @Test func transformerDecoderOperatorIsStructured() throws {
        let decoder = SmeltCAMIR.Block(
            id: "decoder",
            operatorName: .transformerDecoder,
            shape: .init(
                derivation: .init(source: "weights"),
                transformer: .init()
            )
        )
        let ir = try Self.fixture(blocks: [Self.trunkBlock, decoder]).validated()
        let lowered = try #require(ir.blocks.first { $0.id == "decoder" })
        #expect(lowered.operatorName == .transformerDecoder)
    }

    @Test func sourceBackedCompiledNodesDoNotNeedFakeBlocks() throws {
        let prefillSource = SmeltCAMIR.Source(
            id: "prefill",
            kind: "file",
            locator: "models/2b_batch_prefill_pal4/model.mlmodelc"
        )
        let prefillNode = SmeltCAMIR.GraphNode(
            id: "prefill",
            implementation: .compiled,
            source: "prefill",
            inputs: [.init(name: "tokens", type: .init("tokens"))],
            outputs: [.init(name: "prefill_state", type: .init("state"))],
            annotations: [.init("artifact", "sidecar"), .init("engine", "coreml")]
        )
        let ir = Self.fixture(
            sources: [Self.weightsSource, Self.tokenizerSource, Self.normSource, prefillSource],
            graphNodes: [
                Self.tokenizerNode,
                prefillNode,
                Self.trunkNode,
                Self.samplerNode,
                Self.detokenizerNode,
            ],
            graphEdges: [
                Self.promptToTokenizer,
                .init(from: .node("tokenizer", "tokens"), to: .node("prefill", "tokens")),
                .init(from: .node("prefill", "prefill_state"), to: .node("trunk", "prefill_state")),
                Self.trunkToHidden,
                Self.hiddenToSampler,
            ],
            flows: [
                .init(
                    id: "generate",
                    phases: [
                        .init(role: .setup, calls: [.node("tokenizer"), .node("prefill")]),
                        .init(role: .step, label: "decode", calls: [.node("trunk"), .node("sampler")]),
                    ],
                    emit: [.moduleOutput("text")],
                    stop: Self.generateFlow.stop
                ),
            ]
        )
        _ = try ir.validated()

        let bothBlockAndSource = Self.fixture(
            sources: [Self.weightsSource, Self.tokenizerSource, Self.normSource, prefillSource],
            graphNodes: [
                .init(
                    id: "bad",
                    implementation: .compiled,
                    block: "trunk",
                    source: "prefill",
                    inputs: [.init(name: "tokens", type: .init("tokens"))],
                    outputs: [.init(name: "hidden", type: .init("hidden"))]
                ),
                Self.tokenizerNode,
                Self.samplerNode,
                Self.detokenizerNode,
            ]
        )
        #expect(throws: SmeltCAMIRError.self) {
            _ = try bothBlockAndSource.validated()
        }
    }

    @Test func roleSpecificAttentionIsStructured() throws {
        let block = SmeltCAMIR.Block(
            id: "trunk",
            operatorName: .transformer,
            shape: .init(transformer: .init(
                hiddenSize: 2560,
                layers: .init(roles: [.sliding, .sliding, .sliding, .global], repeatCount: 7),
                attentionByRole: [
                    .init(
                        role: .global,
                        attention: .init(
                            qHeads: 8,
                            kvHeads: 2,
                            headDim: 512,
                            rope: .init(kind: .neox, theta: 1_000_000),
                            qkNorm: .rms,
                            qkNormMode: .weight,
                            vNorm: .rms
                        )
                    ),
                    .init(
                        role: .sliding,
                        attention: .init(
                            qHeads: 8,
                            kvHeads: 2,
                            headDim: 256,
                            rope: .init(kind: .neox, theta: 10_000),
                            qkNorm: .rms,
                            vNorm: .rms,
                            window: 512
                        )
                    ),
                ],
                ffn: .init(dim: 10_240, activation: .geglu),
                norm: .init(kind: .rms, eps: "1e-6", mode: .weight),
                vocab: .init(size: 262_144, tiedHead: true),
                perLayerInput: .init(hiddenSize: 256, vocabSize: 262_144),
                sharedKVLayers: 18,
                logitCap: "30"
            ))
        )
        let ir = try Self.fixture(blocks: [block]).validated()
        let shape = try #require(ir.blocks.first?.shape.transformer)
        #expect(shape.attentionByRole?.map(\.role) == [.global, .sliding])
        #expect(shape.attentionByRole?.first?.attention.qkNormMode == .weight)
        #expect(shape.perLayerInput?.hiddenSize == 256)
        #expect(shape.sharedKVLayers == 18)
        #expect(shape.logitCap == "30")
    }

    @Test func projectionActivationViewLayerSpansAreGenericAndValidated() throws {
        func fixture(
            spans: [SmeltCAMIR.ActivationViewLayerSpan]?,
            view: SmeltCAMIR.ProjectionActivationView? = .signedBitplanesI4
        ) -> SmeltCAMIR {
            let block = SmeltCAMIR.Block(
                id: "trunk",
                operatorName: .transformer,
                shape: .init(transformer: .init(
                    hiddenSize: 512,
                    layers: .init(roles: [.delta, .attention], repeatCount: 4),
                    projectionBanks: [
                        .init(
                            id: "delta-input",
                            source: .deltaInput,
                            outputs: [.deltaQKV, .deltaZ, .deltaA, .deltaB],
                            activationView: view,
                            activationViewLayerSpans: spans
                        ),
                    ]
                ))
            )
            return Self.fixture(blocks: [block])
        }

        let first = fixture(spans: [
            .init(start: 5, count: 3),
            .init(start: 1, count: 2),
        ])
        let reordered = fixture(spans: [
            .init(start: 1, count: 2),
            .init(start: 5, count: 3),
        ])
        _ = try first.validated()
        #expect(try first.semanticSHA256() == reordered.semanticSHA256())

        #expect(throws: SmeltCAMIRError.self) {
            _ = try fixture(spans: [
                .init(start: 1, count: 3),
                .init(start: 3, count: 2),
            ]).validated()
        }
        #expect(throws: SmeltCAMIRError.self) {
            _ = try fixture(spans: [.init(start: 7, count: 2)]).validated()
        }
        #expect(throws: SmeltCAMIRError.self) {
            _ = try fixture(spans: [.init(start: .max, count: 2)]).validated()
        }
        #expect(throws: SmeltCAMIRError.self) {
            _ = try fixture(
                spans: [.init(start: 1, count: 1)],
                view: nil
            ).validated()
        }
    }

    @Test func sourceDeferredQuantIsExplicit() throws {
        let ir = Self.fixture(
            tensors: [
                .init(
                    source: "weights",
                    selector: .init("*", source: "weights"),
                    target: .init(block: "trunk", selector: "*"),
                    owner: "trunk"
                ),
            ],
            quantization: [
                Self.defaultQuant,
                .init(
                    selector: .init("embed_tokens", source: "weights"),
                    action: .store,
                    storage: .init(format: .turboQuantH, groupSize: 128),
                    source: "weights",
                    resolution: .sourceDeferred
                ),
            ]
        )
        let validated = try ir.validated()
        let store = try #require(validated.quantization.first { $0.action == .store })
        #expect(store.resolution == .sourceDeferred)

        let notDeferred = Self.fixture(
            tensors: ir.tensors,
            quantization: [
                Self.defaultQuant,
                .init(
                    selector: .init("embed_tokens", source: "weights"),
                    action: .store,
                    storage: .init(format: .turboQuantH, groupSize: 128)
                ),
            ]
        )
        #expect(throws: SmeltCAMIRError.self) {
            _ = try notDeferred.validated()
        }
    }

    @Test func sourceQuantizationPolicyIsStructured() throws {
        let ir = Self.fixture(sourceQuantization: [
            .init(
                source: "weights",
                sourceDTypes: ["q4_k", "q4_0", "q4_1"],
                action: .preserve,
                storage: .init(format: .affineU4, groupSize: 32)
            ),
            .init(
                source: "weights",
                sourceDTypes: ["q8_0", "iq4_xs", "q5_k", "q6_k"],
                action: .dequant,
                targetDType: "f16",
                evidence: [
                    .init(
                        kind: .dequantIdentity,
                        source: "weights",
                        sourceDTypes: ["q4_k", "q4_0", "q4_1"]
                    ),
                ]
            ),
        ])
        let rules = try ir.validated().sourceQuantization
        #expect(rules.map(\.sourceDTypes) == [
            ["iq4_xs", "q5_k", "q6_k", "q8_0"],
            ["q4_0", "q4_1", "q4_k"],
        ])
        #expect(rules[0].targetDType == "f16")
        #expect(rules[1].storage?.format == .affineU4)
    }

    @Test func evidenceRequirementsAreTypedContracts() throws {
        let gate = SmeltCAMIR.Gate(
            id: "storage",
            requirements: [],
            evidence: [
                .init(
                    kind: .tensorStoredAs,
                    tensor: "embed_tokens",
                    storage: .init(format: .turboQuantH, groupSize: 128)
                ),
                .init(kind: .sourceSHA256Recorded, source: "weights"),
            ]
        )
        let validated = try Self.fixture(gates: [Self.startupGate, gate]).validated()
        let evidence = try #require(validated.gates.first { $0.id == "storage" }?.evidence)
        #expect(evidence.map(\.kind) == [.sourceSHA256Recorded, .tensorStoredAs])

        let bad = Self.fixture(gates: [
            .init(
                id: "bad",
                requirements: [],
                evidence: [.init(kind: .dequantIdentity, sourceDTypes: ["q4_0"])]
            ),
        ])
        #expect(throws: SmeltCAMIRError.self) {
            _ = try bad.validated()
        }
    }

    private static func fixture(
        imports: [SmeltCAMIR.Import] = [voiceImport],
        exports: [SmeltCAMIR.Export] = [generateExport],
        sources: [SmeltCAMIR.Source] = [weightsSource, tokenizerSource, normSource],
        blocks: [SmeltCAMIR.Block] = [trunkBlock],
        capabilities: [String] = ["run.generate", "run.stream"],
        graphNodes: [SmeltCAMIR.GraphNode] = [
            tokenizerNode,
            trunkNode,
            samplerNode,
            detokenizerNode,
        ],
        graphEdges: [SmeltCAMIR.GraphEdge] = [
            promptToTokenizer,
            trunkToHidden,
            hiddenToSampler,
        ],
        flows: [SmeltCAMIR.Flow] = [generateFlow],
        tensors: [SmeltCAMIR.TensorMap] = [
            .init(
                source: "weights",
                selector: .init("*", source: "weights"),
                target: .init(block: "trunk", selector: "*"),
                owner: "trunk"
            ),
            .init(
                source: "weights",
                selector: .init("*_norm_weight", source: "weights"),
                target: .init(block: "trunk", selector: "*_norm_weight"),
                owner: "trunk"
            ),
        ],
        quantization: [SmeltCAMIR.QuantRule] = [defaultQuant, normQuant],
        sourceQuantization: [SmeltCAMIR.SourceQuantizationRule] = [],
        gates: [SmeltCAMIR.Gate] = [startupGate]
    ) -> SmeltCAMIR {
        SmeltCAMIR(
            module: .init(id: "qwen35_text"),
            imports: imports,
            exports: exports,
            exportBindings: [.init(export: "generate", flow: "generate")],
            sources: sources,
            blocks: blocks,
            graphNodes: graphNodes,
            graphEdges: graphEdges,
            feedbackEdges: [
                .init(from: .node("sampler", "tokens"), to: .node("trunk", "tokens")),
            ],
            flows: flows,
            capabilities: capabilities,
            backendConstraints: [.init("target", "metal")],
            tensors: tensors,
            quantization: quantization,
            sourceQuantization: sourceQuantization,
            compile: [.init("layout", "memory-neutral"), .init("target", "metal")],
            artifacts: [.init(id: "weights", role: "weights")],
            gates: gates
        )
    }

    private static func expectValidationFailure(
        _ ir: SmeltCAMIR,
        contains expected: String
    ) throws {
        do {
            _ = try ir.validated()
            #expect(Bool(false), "expected CAM IR validation failure")
        } catch let error as SmeltCAMIRError {
            #expect(
                error.description.contains(expected),
                "expected '\(expected)', got '\(error.description)'"
            )
        } catch {
            #expect(Bool(false), "unexpected error \(error)")
        }
    }

    private static func importedNodeIR(
        imports: [SmeltCAMIR.Import] = [speakerVoiceImport],
        node: SmeltCAMIR.GraphNode = importedSynthNode()
    ) -> SmeltCAMIR {
        SmeltCAMIR(
            module: .init(id: "imported_node_parent"),
            imports: imports,
            exports: [.init(id: "run", inputs: [], outputs: [])],
            exportBindings: [.init(export: "run", flow: "run")],
            blocks: [],
            graphNodes: [node],
            graphEdges: [],
            flows: [
                .init(
                    id: "run",
                    phases: [],
                    emit: [],
                    stop: [.init(kind: .hostCancel)]
                ),
            ]
        )
    }

    private static func importedSynthNode(
        inputs: [SmeltCAMIR.Port] = [textPort, speakerPort],
        outputs: [SmeltCAMIR.Port] = [audio24Port]
    ) -> SmeltCAMIR.GraphNode {
        SmeltCAMIR.GraphNode(
            id: "voice_node",
            implementation: .imported,
            imported: .init(alias: "voice", export: "synth"),
            inputs: inputs,
            outputs: outputs
        )
    }

    private static func ds4Fixture() -> SmeltCAMIR {
        SmeltCAMIR(
            module: .init(id: "ds4_heavy_quant"),
            imports: [],
            exports: [
                .init(
                    id: "generate",
                    inputs: [.init(name: "prompt", type: .init("text", attributes: ["encoding": "utf8"]))],
                    outputs: [.init(name: "text", type: .init("text", attributes: ["encoding": "utf8"]))],
                    capabilities: ["run.generate"],
                    gates: ["startup", "quant_quality"]
                ),
            ],
            exportBindings: [.init(export: "generate", flow: "generate")],
            sources: [
                .init(id: "weights", kind: "hf", locator: "deepseek-ai/DS4", revision: "main"),
                .init(id: "calibration_prompts", kind: "file", locator: "calibration/ds4-prompts.jsonl"),
            ],
            blocks: [
                .init(
                    id: "trunk",
                    operatorName: .transformer,
                    shape: .init(transformer: .init(
                        hiddenSize: 7168,
                        layers: .init(count: 61),
                        attention: .init(
                            qHeads: 128,
                            kvHeads: 8,
                            headDim: 128,
                            rope: .init(kind: .yarn, theta: 1_000_000),
                            qkNorm: .rms
                        ),
                        router: .init(topK: 8, experts: 256),
                        expert: .init(ffn: .init(dim: 18_432, activation: .swiglu)),
                        norm: .init(kind: .rms, eps: "1e-6", mode: .weight),
                        vocab: .init(size: 163_840, tiedHead: false)
                    ))
                ),
            ],
            graphNodes: [
                tokenizerNode,
                .init(
                    id: "trunk",
                    implementation: .compiled,
                    block: "trunk",
                    inputs: [.init(name: "tokens", type: .init("tokens"))],
                    outputs: [.init(name: "hidden", type: .init("hidden"))]
                ),
                samplerNode,
                detokenizerNode,
            ],
            graphEdges: [
                .init(
                    from: .moduleInput("prompt"),
                    to: .node("tokenizer", "prompt"),
                    type: .init("text", attributes: ["encoding": "utf8"])
                ),
                .init(from: .node("tokenizer", "tokens"), to: .node("trunk", "tokens"), type: .init("tokens")),
                .init(from: .node("trunk", "hidden"), to: .graphValue("hidden"), type: .init("hidden")),
                .init(from: .graphValue("hidden"), to: .node("sampler", "hidden"), type: .init("hidden")),
                .init(from: .node("sampler", "tokens"), to: .node("detokenizer", "tokens"), type: .init("tokens")),
                .init(
                    from: .node("detokenizer", "text"),
                    to: .moduleOutput("text"),
                    type: .init("text", attributes: ["encoding": "utf8"])
                ),
            ],
            feedbackEdges: [],
            flows: [
                .init(
                    id: "generate",
                    phases: [
                        .init(role: .setup, calls: [.node("tokenizer")]),
                        .init(
                            role: .step,
                            label: "decode",
                            calls: [.node("trunk"), .node("sampler"), .node("detokenizer")]
                        ),
                    ],
                    emit: [.moduleOutput("text")],
                    stop: [.init(kind: .maxSteps, value: 1024), .init(kind: .hostCancel)]
                ),
            ],
            capabilities: ["run.generate"],
            tensors: [
                .init(
                    source: "weights",
                    selector: .init("model.embed_tokens.*", source: "weights"),
                    target: .init(block: "trunk", selector: "embeddings.*"),
                    owner: "trunk"
                ),
                .init(
                    source: "weights",
                    selector: .init("model.layers.*.mlp.experts.*", source: "weights"),
                    target: .init(block: "trunk", selector: "experts.*"),
                    owner: "trunk"
                ),
                .init(
                    source: "weights",
                    selector: .init("*_norm_weight", source: "weights"),
                    target: .init(block: "trunk", selector: "*_norm_weight"),
                    owner: "trunk"
                ),
            ],
            quantization: [
                .init(
                    selector: .init("*", source: "weights"),
                    action: .default,
                    storage: .init(format: .gptq, groupSize: 128)
                ),
                .init(selector: .init("*_norm_weight", source: "weights"), action: .preserve),
                .init(
                    selector: .init("model.embed_tokens.*", source: "weights"),
                    action: .store,
                    storage: .init(format: .turboQuantH, groupSize: 128),
                    priority: 10
                ),
                .init(
                    selector: .init("model.layers.*.mlp.experts.*", source: "weights"),
                    action: .store,
                    storage: .init(format: .gptq, groupSize: 128),
                    priority: 20,
                    calibration: .init(
                        method: .gptq,
                        corpus: .init(
                            source: "calibration_prompts",
                            path: "calibration/ds4-prompts.jsonl",
                            maxTokens: 4096
                        ),
                        captures: ["trunk.experts", "trunk.attention"],
                        layersPerPass: 1,
                        requirements: [
                            .init(subject: "cosine", relation: .greaterThanOrEqual, value: "0.995"),
                        ]
                    )
                ),
            ],
            compile: [.init("target", "metal"), .init("layout", "memory-neutral")],
            artifacts: [.init(id: "weights", role: "weights")],
            gates: [
                startupGate,
                .init(
                    id: "quant_quality",
                    requirements: [
                        .init(subject: "perplexity.delta", relation: .lessThanOrEqual, value: "0.05"),
                    ]
                ),
            ]
        )
    }

    private static let generateExport = SmeltCAMIR.Export(
        id: "generate",
        inputs: [
            .init(name: "prompt", type: .init("text", attributes: ["encoding": "utf8"])),
        ],
        outputs: [
            .init(name: "text", type: .init("text", attributes: ["encoding": "utf8"])),
        ],
        capabilities: ["run.generate"],
        gates: ["startup"]
    )

    private static let synthExport = SmeltCAMIR.Export(
        id: "synth",
        inputs: [.init(name: "text", type: .init("text", attributes: ["encoding": "utf8"]))],
        outputs: [.init(name: "audio", type: .init("pcm", attributes: ["dtype": "f32", "rate": "24khz"]))],
        capabilities: ["run.synthesize", "run.stream"]
    )

    private static let textPort = SmeltCAMIR.Port(
        name: "text",
        type: .init("text", attributes: ["encoding": "utf8"])
    )

    private static let speakerPort = SmeltCAMIR.Port(
        name: "speaker",
        type: .init("voice-id"),
        optional: true
    )

    private static let audio24Port = SmeltCAMIR.Port(
        name: "audio",
        type: .init("pcm", attributes: ["dtype": "f32", "rate": "24khz"])
    )

    private static let speakerSynthExport = SmeltCAMIR.Export(
        id: "synth",
        inputs: [textPort, speakerPort],
        outputs: [audio24Port],
        capabilities: ["run.synthesize", "run.stream"]
    )

    private static let speakerVoiceImport = SmeltCAMIR.Import(
        alias: "voice",
        moduleID: "qwen3_tts",
        irSHA256: String(repeating: "3", count: 64),
        exportABISHA256: try! SmeltCAMIR.exportABISHA256(for: [speakerSynthExport]),
        exportABI: [speakerSynthExport]
    )

    private static let voiceImport = SmeltCAMIR.Import(
        alias: "voice",
        moduleID: "qwen3_tts",
        canonicalURI: "file://Examples/CAM/qwen3_tts.cam",
        irSHA256: String(repeating: "3", count: 64),
        exportABISHA256: try! SmeltCAMIR.exportABISHA256(for: [synthExport]),
        exportABI: [synthExport],
        parameters: ["speaker": "default"]
    )

    private static let weightsSource = SmeltCAMIR.Source(
        id: "weights",
        kind: "hf",
        locator: "Qwen/Qwen3.5-2B",
        revision: "main"
    )

    private static let tokenizerSource = SmeltCAMIR.Source(
        id: "tokenizer",
        kind: "hf-file",
        locator: "Qwen/Qwen3.5-2B/tokenizer.json"
    )

    private static let normSource = SmeltCAMIR.Source(
        id: "norms",
        kind: "derived",
        locator: "weights"
    )

    private static let trunkBlock = SmeltCAMIR.Block(
        id: "trunk",
        operatorName: .transformer,
        shape: .init(transformer: .init(
            hiddenSize: 2048,
            layers: .init(roles: [.delta, .delta, .delta, .attention], repeatCount: 6),
            delta: .init(
                heads: 16,
                headDim: 128,
                convKernel: 4,
                projections: ["qkv": 6144, "z": 2048, "a": 16, "b": 16]
            ),
            attention: .init(
                qHeads: 8,
                kvHeads: 2,
                headDim: 256,
                rope: .init(kind: .neox, theta: 10_000_000),
                qkNorm: .rms
            ),
            ffn: .init(dim: 6144, activation: .swiglu),
            norm: .init(kind: .rms, eps: "1e-6", mode: .onePlusWeight),
            vocab: .init(size: 248_320, tiedHead: true)
        ))
    )

    private static let tokenizerNode = SmeltCAMIR.GraphNode(
        id: "tokenizer",
        implementation: .native,
        inputs: [.init(name: "prompt", type: .init("text", attributes: ["encoding": "utf8"]))],
        outputs: [.init(name: "tokens", type: .init("tokens"))],
        annotations: [.init("tag", "text-tokenizer")]
    )

    private static let trunkNode = SmeltCAMIR.GraphNode(
        id: "trunk",
        implementation: .compiled,
        block: "trunk",
        inputs: [
            .init(name: "tokens", type: .init("tokens")),
            .init(name: "prefill_state", type: .init("state"), optional: true),
        ],
        outputs: [.init(name: "hidden", type: .init("hidden", attributes: ["dtype": "f16", "dim": "2048"]))],
        annotations: [.init("artifact", "baked-inline"), .init("tag", "decode-core")]
    )

    private static let samplerNode = SmeltCAMIR.GraphNode(
        id: "sampler",
        implementation: .native,
        inputs: [.init(name: "hidden", type: .init("hidden"))],
        outputs: [.init(name: "tokens", type: .init("tokens"))],
        annotations: [.init("tag", "sampler")]
    )

    private static let detokenizerNode = SmeltCAMIR.GraphNode(
        id: "detokenizer",
        implementation: .native,
        inputs: [.init(name: "tokens", type: .init("tokens"))],
        outputs: [.init(name: "text", type: .init("text", attributes: ["encoding": "utf8"]))],
        annotations: [.init("tag", "text-detokenizer")]
    )

    private static let promptToTokenizer = SmeltCAMIR.GraphEdge(
        from: .moduleInput("prompt"),
        to: .node("tokenizer", "prompt"),
        type: .init("text", attributes: ["encoding": "utf8"])
    )

    private static let trunkToHidden = SmeltCAMIR.GraphEdge(
        from: .node("trunk", "hidden"),
        to: .graphValue("hidden"),
        type: .init("hidden")
    )

    private static let hiddenToSampler = SmeltCAMIR.GraphEdge(
        from: .graphValue("hidden"),
        to: .node("sampler", "hidden"),
        type: .init("hidden")
    )

    private static let generateFlow = SmeltCAMIR.Flow(
        id: "generate",
        phases: [
            .init(role: .setup, calls: [.node("tokenizer")]),
            .init(role: .step, label: "decode", calls: [.node("trunk"), .node("sampler")]),
        ],
        emit: [.moduleOutput("text")],
        stop: [
            .init(kind: .hostCancel),
            .init(kind: .eosToken, value: 248_044),
            .init(kind: .maxSteps, value: 512),
        ]
    )

    private static let startupGate = SmeltCAMIR.Gate(
        id: "startup",
        from: .init(kind: .flowAccepted, flow: "generate"),
        to: .init(
            kind: .emit,
            flow: "generate",
            endpoint: .moduleOutput("text"),
            predicates: [
                .init(subject: "tokens", relation: .greaterThanOrEqual, value: "1"),
            ]
        ),
        requirements: [
            .init(subject: "elapsed", relation: .lessThanOrEqual, value: "100", unit: "ms"),
        ]
    )

    private static let defaultQuant = SmeltCAMIR.QuantRule(
        selector: .init("*", source: "weights"),
        action: .default,
        storage: .init(format: .lutU4, groupSize: 16)
    )

    private static let normQuant = SmeltCAMIR.QuantRule(
        selector: .init("*_norm_weight", source: "weights"),
        action: .preserve,
        priority: 10
    )

}
