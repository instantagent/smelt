import SmeltModuleAuthoring

// Qwen3.5 text-generation family. text/fast/4b share a single-input scaffold;
// reasoner is a two-input judge with a prompt_builder adapter front end. The
// trunk shape, quant rules, compile constraints, constant nodes, and gates are
// shared. Everything is authored as the parser's fully-lowered output — the
// quant-rule priorities (10/12/13/5/7), flattened top-level capabilities, and
// graph-value dedup names (tokens / tokens_2) — held to byte parity by
// ModuleAuthoringParityTests.

// MARK: - Public definitions

func qwen35Text() -> SmeltCAMIR {
    qwen35TextGen(
        id: "qwen35_text", repo: "Qwen/Qwen3.5-2B", elapsedMs: "115", preparePromptPrefix: true,
        toolTranscriptCodec: SmeltToolTranscriptCodecName.xmlFunctionParameters,
        trunk: qwen35Trunk(
            hiddenSize: 2048, ffnDim: 6144,
            deltaHeads: 16, deltaProjections: ["qkv": 6144, "z": 2048, "a": 16, "b": 16],
            qHeads: 8, kvHeads: 2, repeatCount: 6
        )
    )
}

func qwen35Fast() -> SmeltCAMIR {
    qwen35TextGen(
        id: "qwen35_fast", repo: "Qwen/Qwen3.5-0.8B", elapsedMs: "100", preparePromptPrefix: false,
        toolTranscriptCodec: SmeltToolTranscriptCodecName.xmlFunctionParameters,
        trunk: qwen35Trunk(
            hiddenSize: 1024, ffnDim: 3584,
            deltaHeads: 16, deltaProjections: ["qkv": 6144, "z": 2048, "a": 16, "b": 16],
            qHeads: 8, kvHeads: 2, repeatCount: 6
        )
    )
}

func qwen36TwentySevenB() -> SmeltCAMIR {
    qwen35TextGen(
        id: "qwen36_27b", repo: "Qwen/Qwen3.6-27B", elapsedMs: "2000",
        preparePromptPrefix: true, verifyArgmax: true,
        toolTranscriptCodec: SmeltToolTranscriptCodecName.xmlFunctionParameters,
        trunk: qwen35Trunk(
            hiddenSize: 5_120, ffnDim: 17_408,
            deltaHeads: 48,
            deltaProjections: ["qkv": 10_240, "z": 6_144, "a": 48, "b": 48],
            qHeads: 24, kvHeads: 4, repeatCount: 16, tiedHead: false
        )
    )
}

/// Qwen 3.6's checkpoint-owned next-token predictor. The only unusual brick
/// is the generic two-source input fusion declared on the transformer block;
/// after fusion this is an ordinary one-layer Qwen decoder package.
func qwen36TwentySevenBMTP() -> SmeltCAMIR {
    qwen35TextGen(
        id: "qwen36_27b_mtp",
        repo: "Qwen/Qwen3.6-27B",
        elapsedMs: "500",
        preparePromptPrefix: false,
        trunk: qwen36MTPTrunk(),
        quantization: qwen36MTPQuantRules(),
        checkpointMap: "hf.qwen-mtp",
        compile: qwen35DecodeOnlyCompile()
    )
}

func bonsaiTwentySevenBBinary() -> SmeltCAMIR {
    bonsaiTwentySevenB(
        id: "bonsai_27b_binary",
        repo: "prism-ml/Bonsai-27B-mlx-1bit",
        revision: "151444121fc0b30e64787911d17cc27b6c088aa0",
        format: .binary1
    )
}

func bonsaiTwentySevenBTernary() -> SmeltCAMIR {
    bonsaiTwentySevenB(
        id: "bonsai_27b_ternary",
        repo: "prism-ml/Ternary-Bonsai-27B-mlx-2bit",
        revision: "bec2f91c06d064684042f9f55b6733ffc77e99e9",
        format: .ternary2
    )
}

private func bonsaiTwentySevenB(
    id: String,
    repo: String,
    revision: String,
    format: IR.QuantStorageFormat
) -> SmeltCAMIR {
    qwen35TextGen(
        id: id,
        repo: repo,
        revision: revision,
        elapsedMs: "2000",
        preparePromptPrefix: true,
        verifyArgmax: true,
        verifyTokenCapacity: format == .ternary2 ? 32 : nil,
        promptFormat: SmeltPromptTemplateName.chatML,
        toolTranscriptCodec: SmeltToolTranscriptCodecName.xmlFunctionParameters,
        trunk: qwen35Trunk(
            hiddenSize: 5_120,
            ffnDim: 17_408,
            deltaHeads: 48,
            deltaProjections: ["qkv": 10_240, "z": 6_144, "a": 48, "b": 48],
            qHeads: 24,
            kvHeads: 4,
            repeatCount: 16,
            tiedHead: false,
            qkNormMode: .weight,
            projectionActivationView: format == .binary1
                ? .signedBitplanesI3 : nil,
            ffnInputActivationView: format == .binary1
                ? .signedBitplanesI4 : nil,
            ffnInputActivationViewLayerSpans: format == .binary1
                ? [
                    IR.ActivationViewLayerSpan(start: 5, count: 4),
                    IR.ActivationViewLayerSpan(start: 10, count: 54),
                ] : nil,
            deltaOutputActivationView: format == .binary1
                ? .signedBitplanesI6 : nil,
            attentionOutputActivationView: format == .binary1
                ? .signedBitplanesI6 : nil,
            ffnIntermediateActivationView: format == .binary1
                ? .signedBitplanesI4 : nil
        ),
        quantization: qwen35QuantRules(format: format, groupSize: 128)
    )
}

func qwen35FourB() -> SmeltCAMIR {
    let textCapabilities = ["prepare.prompt-prefix", "run.generate"]
    let multimodalCapabilities = ["prepare.prompt-prefix", "run.generate.multimodal"]
    let trunk = qwen35Trunk(
        hiddenSize: 2560, ffnDim: 9216,
        deltaHeads: 32, deltaProjections: ["qkv": 8192, "z": 4096, "a": 32, "b": 32],
        qHeads: 16, kvHeads: 4, repeatCount: 8
    )
    return SmeltCAMIR(
        module: IR.Module(id: "qwen35_4b"),
        exports: [
            IR.Export(
                id: "generate",
                inputs: [port("prompt", textType())],
                outputs: [port("text", textType())],
                capabilities: textCapabilities,
                gates: ["startup"]
            ),
            IR.Export(
                id: "generate_multimodal",
                inputs: [
                    port("prompt", textType()),
                    port("media", qwen35MediaType()),
                ],
                outputs: [port("text", textType())],
                capabilities: multimodalCapabilities,
                gates: ["multimodal-startup"]
            ),
        ],
        exportBindings: [
            IR.ExportBinding(export: "generate", flow: "generate"),
            IR.ExportBinding(export: "generate_multimodal", flow: "generate_multimodal"),
        ],
        sources: qwen35Sources(repo: "Qwen/Qwen3.5-4B"),
        blocks: [trunk, qwen35VisionEncoder(), qwen35VisionMerger()],
        graphNodes: [
            tokenizerNode(
                input: port("prompt", textType()),
                toolTranscriptCodec: SmeltToolTranscriptCodecName.xmlFunctionParameters
            ),
            trunkNode(), samplerNode(), detokenizerNode(),
        ]
            + qwen35MultimodalNodes(),
        graphEdges: [
            IR.GraphEdge(
                from: .moduleInput("prompt"),
                to: .node("tokenizer", "prompt"),
                type: textType()
            ),
        ] + qwen35BackEdges() + qwen35MultimodalEdges(),
        feedbackEdges: [
            qwen35FeedbackEdge(),
            IR.FeedbackEdge(
                from: .node("multimodal-sampler", "tokens"),
                to: .node("multimodal-trunk", "tokens")
            ),
        ],
        flows: [
            qwen35Flow(id: "generate", setupCalls: [.node("tokenizer")]),
            qwen35MultimodalFlow(),
        ],
        capabilities: Array(Set(textCapabilities + multimodalCapabilities)),
        backendConstraints: qwen35BackendConstraints(),
        tensors: qwen35MultimodalTensors(),
        quantization: qwen35QuantRules(),
        compile: qwen35Compile() + [IR.Constraint("vision", "metal")],
        artifacts: [
            IR.ArtifactRole(
                id: "vision-component",
                role: "compiled-component",
                required: true
            ),
        ],
        gates: qwen35Gates(flow: "generate", elapsedMs: "210") + [qwen35MultimodalGate()]
    )
}

func qwen35Reasoner() -> SmeltCAMIR {
    let caps = ["prepare.prompt-prefix", "run.generate"]
    let frontNodes: [IR.GraphNode] = [
        IR.GraphNode(
            id: "prompt_builder",
            implementation: .adapter,
            inputs: [port("candidate", textType()), port("context", textType())],
            outputs: [port("review_prompt", bareType("review_prompt"))],
            annotations: [annot("template", "draft-judge")]
        ),
        tokenizerNode(input: port("review_prompt", bareType("review_prompt"))),
    ]
    let frontEdges: [IR.GraphEdge] = [
        IR.GraphEdge(from: .moduleInput("candidate"), to: .node("prompt_builder", "candidate"), type: textType()),
        IR.GraphEdge(from: .moduleInput("context"), to: .node("prompt_builder", "context"), type: textType()),
        IR.GraphEdge(from: .node("prompt_builder", "review_prompt"), to: .graphValue("review_prompt"), type: bareType("review_prompt")),
        IR.GraphEdge(from: .graphValue("review_prompt"), to: .node("tokenizer", "review_prompt"), type: bareType("review_prompt")),
    ]
    return SmeltCAMIR(
        module: IR.Module(id: "qwen35_reasoner"),
        exports: [
            IR.Export(
                id: "review",
                inputs: [port("candidate", textType()), port("context", textType())],
                outputs: [port("text", textType())],
                capabilities: caps,
                gates: ["startup"]
            ),
        ],
        exportBindings: [IR.ExportBinding(export: "review", flow: "review")],
        sources: qwen35Sources(repo: "Qwen/Qwen3.5-4B"),
        blocks: [qwen35Trunk(
            hiddenSize: 2560, ffnDim: 9216,
            deltaHeads: 32, deltaProjections: ["qkv": 8192, "z": 4096, "a": 32, "b": 32],
            qHeads: 16, kvHeads: 4, repeatCount: 8
        )],
        graphNodes: frontNodes + [trunkNode(), samplerNode(), detokenizerNode()],
        graphEdges: frontEdges + qwen35BackEdges(),
        feedbackEdges: [qwen35FeedbackEdge()],
        flows: [qwen35Flow(id: "review", setupCalls: [.node("prompt_builder"), .node("tokenizer")])],
        capabilities: caps,
        backendConstraints: qwen35BackendConstraints(),
        tensors: qwen35Tensors(),
        quantization: qwen35QuantRules(),
        compile: qwen35Compile(),
        gates: qwen35Gates(flow: "review", elapsedMs: "150")
    )
}

// MARK: - text/fast shared builder

private func qwen35TextGen(
    id: String, repo: String, revision: String = "main",
    elapsedMs: String, preparePromptPrefix: Bool,
    verifyArgmax: Bool = false,
    verifyTokenCapacity: Int? = nil,
    promptFormat: String = SmeltPromptTemplateName.chatML,
    toolTranscriptCodec: String? = nil,
    trunk: IR.Block,
    quantization: [IR.QuantRule] = qwen35QuantRules(),
    checkpointMap: String = "hf.qwen",
    compile: [IR.Constraint]? = nil
) -> SmeltCAMIR {
    let caps = preparePromptPrefix ? ["prepare.prompt-prefix", "run.generate"] : ["run.generate"]
    let includePrefill = compile?.contains { $0.key == "prefill" } ?? true
    let frontEdges: [IR.GraphEdge] = [
        IR.GraphEdge(from: .moduleInput("prompt"), to: .node("tokenizer", "prompt"), type: textType()),
    ]
    return SmeltCAMIR(
        module: IR.Module(id: id),
        exports: [
            IR.Export(
                id: "generate",
                inputs: [port("prompt", textType())],
                outputs: [port("text", textType())],
                capabilities: caps,
                gates: ["startup"]
            ),
        ],
        exportBindings: [IR.ExportBinding(export: "generate", flow: "generate")],
        sources: qwen35Sources(
            repo: repo,
            revision: revision,
            checkpointMap: checkpointMap
        ),
        blocks: [trunk],
        graphNodes: [
            tokenizerNode(
                input: port("prompt", textType()),
                promptFormat: promptFormat,
                toolTranscriptCodec: toolTranscriptCodec
            ),
            trunkNode(), samplerNode(), detokenizerNode(),
        ],
        graphEdges: frontEdges + qwen35BackEdges(),
        feedbackEdges: [qwen35FeedbackEdge()],
        flows: [qwen35Flow(id: "generate", setupCalls: [.node("tokenizer")])],
        capabilities: caps,
        backendConstraints: qwen35BackendConstraints(),
        tensors: qwen35Tensors(),
        quantization: quantization,
        compile: compile ?? qwen35Compile(
            verifyArgmax: verifyArgmax,
            verifyTokenCapacity: verifyTokenCapacity
        ),
        gates: qwen35Gates(
            flow: "generate",
            elapsedMs: elapsedMs,
            verifyArgmax: verifyArgmax,
            includePrefill: includePrefill
        )
    )
}

// MARK: - Qwen3.5 4B multimodal region

private func qwen35MediaType() -> IR.TypeRef {
    // The current executable component implements still images. Video is an
    // upstream model capability, not yet a runtime/component capability.
    IR.TypeRef("media", attributes: ["kinds": "image"])
}

private func qwen35VisionEncoder() -> IR.Block {
    IR.Block(
        id: "vision-encoder",
        operatorName: .transformerEncoder,
        shape: IR.BlockShape(
            transformer: IR.TransformerShape(
                hiddenSize: 1_024,
                layers: IR.LayerPattern(roles: [.global], repeatCount: 24),
                attention: IR.AttentionShape(qHeads: 16, kvHeads: 16, headDim: 64),
                ffn: IR.FFNShape(dim: 4_096, activation: .gelu),
                norm: IR.NormShape(kind: .layer, eps: "1e-6", mode: .weight)
            ),
            requirements: [
                IR.BlockRequirement("activation", "gelu-pytorch-tanh"),
                IR.BlockRequirement("attention-bias"),
                IR.BlockRequirement("in-channels", "3"),
                IR.BlockRequirement("patch-size", "16"),
                IR.BlockRequirement("temporal-patch-size", "2"),
                IR.BlockRequirement("position-embeddings", "2304"),
                IR.BlockRequirement("spatial-merge-size", "2"),
            ]
        )
    )
}

private func qwen35VisionMerger() -> IR.Block {
    IR.Block(
        id: "vision-merger",
        operatorName: .adapter,
        shape: IR.BlockShape(requirements: [
            IR.BlockRequirement("activation", "gelu-pytorch-tanh"),
            IR.BlockRequirement("hidden-size", "1024"),
            IR.BlockRequirement("out-hidden-size", "2560"),
            IR.BlockRequirement("spatial-merge-size", "2"),
        ])
    )
}

private func qwen35MultimodalNodes() -> [IR.GraphNode] {
    [
        IR.GraphNode(
            id: "multimodal-tokenizer",
            implementation: .native,
            inputs: [port("prompt", textType())],
            outputs: [port("tokens", bareType("tokens"))],
            annotations: [
                annot("assistant-prelude", "preclosed-think"),
                annot("prompt-format", "chatml-multimodal"),
                annot("tag", "text-tokenizer"),
                annot("thinking-policy", "disabled"),
                annot("tool-format", SmeltToolTranscriptCodecName.xmlFunctionParameters),
                annot("vision-end-token", "248054"),
                annot("vision-start-token", "248053"),
                annot("image-token", "248056"),
                annot("video-token", "248057"),
            ]
        ),
        IR.GraphNode(
            id: "media-preprocessor",
            implementation: .native,
            inputs: [port("media", qwen35MediaType())],
            outputs: [
                port("patches", IR.TypeRef("tensor", attributes: ["dtype": "f32"])),
                port("grid_thw", IR.TypeRef("shape", attributes: ["axes": "t,h,w"])),
            ],
            annotations: [
                annot("image-mean", "0.5,0.5,0.5"),
                annot("image-std", "0.5,0.5,0.5"),
                annot("max-pixels", "16777216"),
                annot("min-pixels", "65536"),
                annot("patch-size", "16"),
                annot("resample", "bicubic"),
                annot("spatial-merge-size", "2"),
                annot("tag", "media-preprocessor"),
                annot("temporal-patch-size", "2"),
            ]
        ),
        IR.GraphNode(
            id: "vision-encoder",
            implementation: .compiled,
            block: "vision-encoder",
            inputs: [
                port("patches", IR.TypeRef("tensor", attributes: ["dtype": "f32"])),
                port("grid_thw", IR.TypeRef("shape", attributes: ["axes": "t,h,w"])),
            ],
            outputs: [
                port(
                    "vision_hidden",
                    IR.TypeRef("hidden", attributes: ["dim": "1024", "dtype": "f16"])
                ),
            ],
            annotations: [
                annot("artifact", "vision-component"),
                annot("tag", "vision-encoder"),
            ]
        ),
        IR.GraphNode(
            id: "vision-merger",
            implementation: .compiled,
            block: "vision-merger",
            inputs: [
                port(
                    "vision_hidden",
                    IR.TypeRef("hidden", attributes: ["dim": "1024", "dtype": "f16"])
                ),
                port("grid_thw", IR.TypeRef("shape", attributes: ["axes": "t,h,w"])),
            ],
            outputs: [
                port(
                    "visual_embeddings",
                    IR.TypeRef("embeddings", attributes: ["dim": "2560", "dtype": "f16"])
                ),
            ],
            annotations: [
                annot("artifact", "vision-component"),
                annot("tag", "vision-merger"),
            ]
        ),
        IR.GraphNode(
            id: "visual-token-fusion",
            implementation: .native,
            inputs: [
                port("tokens", bareType("tokens")),
                port(
                    "visual_embeddings",
                    IR.TypeRef("embeddings", attributes: ["dim": "2560", "dtype": "f16"])
                ),
                port("grid_thw", IR.TypeRef("shape", attributes: ["axes": "t,h,w"])),
            ],
            outputs: [port("prefill_state", bareType("state"))],
            annotations: [
                annot("deepstack-visual-indexes", "none"),
                annot("mrope-interleaved", "true"),
                annot("mrope-section", "11,11,10"),
                annot("tag", "visual-token-fusion"),
            ]
        ),
        IR.GraphNode(
            id: "multimodal-trunk",
            implementation: .compiled,
            block: "trunk",
            inputs: [
                port("prefill_state", bareType("state")),
                port("tokens", bareType("tokens")),
            ],
            outputs: [port("hidden", bareType("hidden"))],
            annotations: [
                annot("artifact", "compiled-inline"),
                annot("feedback", "tokens"),
                annot("state", "kv-cache,conv-state,rec-state"),
                annot("tag", "decode-core"),
            ]
        ),
        IR.GraphNode(
            id: "multimodal-sampler",
            implementation: .native,
            inputs: [port("hidden", bareType("hidden"))],
            outputs: [port("tokens", bareType("tokens"))],
            annotations: [annot("state", "sampler"), annot("tag", "sampler")]
        ),
        IR.GraphNode(
            id: "multimodal-detokenizer",
            implementation: .native,
            inputs: [port("tokens", bareType("tokens"))],
            outputs: [port("text", textType())],
            annotations: [annot("tag", "text-detokenizer")]
        ),
    ]
}

private func qwen35MultimodalEdges() -> [IR.GraphEdge] {
    [
        IR.GraphEdge(
            from: .moduleInput("prompt"),
            to: .node("multimodal-tokenizer", "prompt"),
            type: textType()
        ),
        IR.GraphEdge(
            from: .moduleInput("media"),
            to: .node("media-preprocessor", "media"),
            type: qwen35MediaType()
        ),
        IR.GraphEdge(
            from: .node("media-preprocessor", "patches"),
            to: .node("vision-encoder", "patches"),
            type: IR.TypeRef("tensor", attributes: ["dtype": "f32"])
        ),
        IR.GraphEdge(
            from: .node("media-preprocessor", "grid_thw"),
            to: .node("vision-encoder", "grid_thw"),
            type: IR.TypeRef("shape", attributes: ["axes": "t,h,w"])
        ),
        IR.GraphEdge(
            from: .node("media-preprocessor", "grid_thw"),
            to: .node("vision-merger", "grid_thw"),
            type: IR.TypeRef("shape", attributes: ["axes": "t,h,w"])
        ),
        IR.GraphEdge(
            from: .node("media-preprocessor", "grid_thw"),
            to: .node("visual-token-fusion", "grid_thw"),
            type: IR.TypeRef("shape", attributes: ["axes": "t,h,w"])
        ),
        IR.GraphEdge(
            from: .node("vision-encoder", "vision_hidden"),
            to: .node("vision-merger", "vision_hidden"),
            type: IR.TypeRef("hidden", attributes: ["dim": "1024", "dtype": "f16"])
        ),
        IR.GraphEdge(
            from: .node("vision-merger", "visual_embeddings"),
            to: .node("visual-token-fusion", "visual_embeddings"),
            type: IR.TypeRef("embeddings", attributes: ["dim": "2560", "dtype": "f16"])
        ),
        IR.GraphEdge(
            from: .node("multimodal-tokenizer", "tokens"),
            to: .node("visual-token-fusion", "tokens"),
            type: bareType("tokens")
        ),
        IR.GraphEdge(
            from: .node("visual-token-fusion", "prefill_state"),
            to: .node("multimodal-trunk", "prefill_state"),
            type: bareType("state")
        ),
        IR.GraphEdge(
            from: .node("multimodal-trunk", "hidden"),
            to: .node("multimodal-sampler", "hidden"),
            type: bareType("hidden")
        ),
        IR.GraphEdge(
            from: .node("multimodal-sampler", "tokens"),
            to: .node("multimodal-detokenizer", "tokens"),
            type: bareType("tokens")
        ),
        IR.GraphEdge(
            from: .node("multimodal-detokenizer", "text"),
            to: .moduleOutput("text"),
            type: textType()
        ),
    ]
}

private func qwen35MultimodalFlow() -> IR.Flow {
    IR.Flow(
        id: "generate_multimodal",
        phases: [
            IR.FlowPhase(
                role: .setup,
                calls: [
                    .node("multimodal-tokenizer"),
                    .node("media-preprocessor"),
                    .node("vision-encoder"),
                    .node("vision-merger"),
                    .node("visual-token-fusion"),
                ]
            ),
            IR.FlowPhase(
                role: .step,
                label: "decode",
                calls: [.node("multimodal-trunk"), .node("multimodal-sampler")]
            ),
        ],
        emit: [.node("multimodal-detokenizer", "text")],
        stop: [
            IR.StopCondition(kind: .eosToken, value: 248_044),
            IR.StopCondition(kind: .eosToken, value: 248_046),
            IR.StopCondition(kind: .hostCancel),
            IR.StopCondition(kind: .maxSteps, value: 512),
        ]
    )
}

private func qwen35MultimodalTensors() -> [IR.TensorMap] {
    [
        IR.TensorMap(
            source: "weights",
            selector: IR.TensorSelector("model.language_model.*", source: "weights"),
            target: IR.TensorTarget(block: "trunk", selector: "*"),
            owner: "trunk"
        ),
        IR.TensorMap(
            source: "weights",
            selector: IR.TensorSelector("model.visual.patch_embed.*", source: "weights"),
            target: IR.TensorTarget(block: "vision-encoder", selector: "patch_embed.*"),
            owner: "vision-encoder"
        ),
        IR.TensorMap(
            source: "weights",
            selector: IR.TensorSelector("model.visual.pos_embed.*", source: "weights"),
            target: IR.TensorTarget(block: "vision-encoder", selector: "pos_embed.*"),
            owner: "vision-encoder"
        ),
        IR.TensorMap(
            source: "weights",
            selector: IR.TensorSelector("model.visual.blocks.*", source: "weights"),
            target: IR.TensorTarget(block: "vision-encoder", selector: "blocks.*"),
            owner: "vision-encoder"
        ),
        IR.TensorMap(
            source: "weights",
            selector: IR.TensorSelector("model.visual.merger.*", source: "weights"),
            target: IR.TensorTarget(block: "vision-merger", selector: "*"),
            owner: "vision-merger"
        ),
    ]
}

private func qwen35MultimodalGate() -> IR.Gate {
    IR.Gate(
        id: "multimodal-startup",
        from: IR.GateEvent(kind: .flowAccepted, flow: "generate_multimodal"),
        to: IR.GateEvent(
            kind: .emit,
            flow: "generate_multimodal",
            endpoint: .moduleOutput("text"),
            predicates: [
                IR.Comparison(subject: "tokens", relation: .greaterThanOrEqual, value: "1"),
            ]
        ),
        requirements: []
    )
}

// MARK: - Shared pieces

private func qwen35Trunk(
    hiddenSize: Int, ffnDim: Int, deltaHeads: Int, deltaProjections: [String: Int],
    qHeads: Int, kvHeads: Int, repeatCount: Int, tiedHead: Bool = true,
    qkNormMode: IR.NormMode? = nil,
    projectionActivationView: IR.ProjectionActivationView? = nil,
    ffnInputActivationView: IR.ProjectionActivationView? = nil,
    ffnInputActivationViewLayerSpans: [IR.ActivationViewLayerSpan]? = nil,
    deltaOutputActivationView: IR.ProjectionActivationView? = nil,
    attentionOutputActivationView: IR.ProjectionActivationView? = nil,
    ffnIntermediateActivationView: IR.ProjectionActivationView? = nil,
    lmHeadActivationView: IR.ProjectionActivationView? = nil
) -> IR.Block {
    IR.Block(
        id: "trunk",
        operatorName: .transformer,
        shape: IR.BlockShape(
            transformer: IR.TransformerShape(
                hiddenSize: hiddenSize,
                layers: IR.LayerPattern(roles: [.delta, .delta, .delta, .attention], repeatCount: repeatCount),
                delta: IR.DeltaShape(heads: deltaHeads, headDim: 128, convKernel: 4, projections: deltaProjections),
                attention: IR.AttentionShape(
                    qHeads: qHeads, kvHeads: kvHeads, headDim: 256,
                    rope: IR.RopeShape(kind: .neox, theta: 10_000_000),
                    qkNorm: .rms,
                    qkNormMode: qkNormMode
                ),
                ffn: IR.FFNShape(dim: ffnDim, activation: .swiglu),
                norm: IR.NormShape(kind: .rms, eps: "1e-6", mode: .onePlusWeight),
                vocab: IR.VocabShape(size: 248_320, tiedHead: tiedHead),
                projectionBanks: [
                    IR.ProjectionBank(
                        id: "delta-input",
                        source: .deltaInput,
                        outputs: [.deltaQKV, .deltaZ, .deltaA, .deltaB],
                        activationView: projectionActivationView
                    ),
                    IR.ProjectionBank(
                        id: "attention-input",
                        source: .attentionInput,
                        outputs: [.attentionQ, .attentionK, .attentionV],
                        activationView: projectionActivationView
                    ),
                    IR.ProjectionBank(
                        id: "delta-output",
                        source: .deltaOutput,
                        outputs: [.deltaOut],
                        activationView: deltaOutputActivationView
                    ),
                    IR.ProjectionBank(
                        id: "attention-output",
                        source: .attentionOutput,
                        outputs: [.attentionOut],
                        activationView: attentionOutputActivationView
                    ),
                    IR.ProjectionBank(
                        id: "ffn-input",
                        source: .ffnInput,
                        outputs: [.ffnGate, .ffnUp],
                        activationView: ffnInputActivationView,
                        activationViewLayerSpans: ffnInputActivationViewLayerSpans
                    ),
                    IR.ProjectionBank(
                        id: "ffn-intermediate",
                        source: .ffnIntermediate,
                        outputs: [.ffnDown],
                        activationView: ffnIntermediateActivationView
                    ),
                ] + (lmHeadActivationView.map { activationView in [
                    IR.ProjectionBank(
                        id: "lm-head-input",
                        source: .lmHeadInput,
                        outputs: [.lmHead],
                        activationView: activationView
                    ),
                ] } ?? [])
            )
        )
    )
}

private func qwen36MTPTrunk() -> IR.Block {
    IR.Block(
        id: "trunk",
        operatorName: .transformer,
        shape: IR.BlockShape(
            transformer: IR.TransformerShape(
                hiddenSize: 5_120,
                layers: IR.LayerPattern(roles: [.attention], repeatCount: 1),
                attention: IR.AttentionShape(
                    qHeads: 24,
                    kvHeads: 4,
                    headDim: 256,
                    rope: IR.RopeShape(kind: .neox, theta: 10_000_000),
                    qkNorm: .rms
                ),
                ffn: IR.FFNShape(dim: 17_408, activation: .swiglu),
                norm: IR.NormShape(kind: .rms, eps: "1e-6", mode: .onePlusWeight),
                vocab: IR.VocabShape(size: 248_320, tiedHead: false),
                projectionBanks: [
                    IR.ProjectionBank(
                        id: "attention-input",
                        source: .attentionInput,
                        outputs: [.attentionQ, .attentionK, .attentionV]
                    ),
                    IR.ProjectionBank(
                        id: "ffn-input",
                        source: .ffnInput,
                        outputs: [.ffnGate, .ffnUp]
                    ),
                ]
            ),
            requirements: [
                IR.BlockRequirement("static-seq-capacity", "8192"),
                IR.BlockRequirement("rope-dim", "64"),
                IR.BlockRequirement("gated-q", "true"),
                IR.BlockRequirement("input-fusion-source-width", "5120"),
                IR.BlockRequirement("input-fusion-source-count", "2"),
                IR.BlockRequirement("input-fusion-normalize-sources", "true"),
            ]
        )
    )
}

private func tokenizerNode(
    input: IR.Port,
    promptFormat: String = SmeltPromptTemplateName.chatML,
    toolTranscriptCodec: String? = nil
) -> IR.GraphNode {
    IR.GraphNode(
        id: "tokenizer",
        implementation: .native,
        inputs: [input],
        outputs: [port("tokens", bareType("tokens"))],
        annotations: [
            annot("assistant-prelude", "preclosed-think"),
            annot("prompt-format", promptFormat),
            annot("tag", "text-tokenizer"),
            annot("thinking-policy", "disabled"),
        ] + (toolTranscriptCodec.map { [annot("tool-format", $0)] } ?? [])
    )
}

private func trunkNode() -> IR.GraphNode {
    IR.GraphNode(
        id: "trunk",
        implementation: .compiled,
        block: "trunk",
        inputs: [port("tokens", bareType("tokens"))],
        outputs: [port("hidden", bareType("hidden"))],
        annotations: [
            annot("artifact", "compiled-inline"),
            annot("feedback", "tokens"),
            annot("state", "kv-cache,conv-state,rec-state"),
        ]
    )
}

private func samplerNode() -> IR.GraphNode {
    IR.GraphNode(
        id: "sampler",
        implementation: .native,
        inputs: [port("hidden", bareType("hidden"))],
        outputs: [port("tokens", bareType("tokens"))],
        annotations: [annot("state", "sampler"), annot("tag", "sampler")]
    )
}

private func detokenizerNode() -> IR.GraphNode {
    IR.GraphNode(
        id: "detokenizer",
        implementation: .native,
        inputs: [port("tokens", bareType("tokens"))],
        outputs: [port("text", textType())],
        annotations: [annot("tag", "text-detokenizer")]
    )
}

/// The tokenizer→trunk→sampler→detokenizer→output edges, shared by all three.
private func qwen35BackEdges() -> [IR.GraphEdge] {
    [
        IR.GraphEdge(from: .node("tokenizer", "tokens"), to: .graphValue("tokens"), type: bareType("tokens")),
        IR.GraphEdge(from: .graphValue("tokens"), to: .node("trunk", "tokens"), type: bareType("tokens")),
        IR.GraphEdge(from: .node("trunk", "hidden"), to: .graphValue("hidden"), type: bareType("hidden")),
        IR.GraphEdge(from: .graphValue("hidden"), to: .node("sampler", "hidden"), type: bareType("hidden")),
        IR.GraphEdge(from: .node("sampler", "tokens"), to: .graphValue("tokens_2"), type: bareType("tokens")),
        IR.GraphEdge(from: .graphValue("tokens_2"), to: .node("detokenizer", "tokens"), type: bareType("tokens")),
        IR.GraphEdge(from: .node("detokenizer", "text"), to: .moduleOutput("text"), type: textType()),
    ]
}

private func qwen35FeedbackEdge() -> IR.FeedbackEdge {
    IR.FeedbackEdge(from: .node("sampler", "tokens"), to: .node("trunk", "tokens"))
}

private func qwen35Flow(id: String, setupCalls: [IR.FlowCall]) -> IR.Flow {
    IR.Flow(
        id: id,
        phases: [
            IR.FlowPhase(role: .setup, calls: setupCalls),
            IR.FlowPhase(role: .step, label: "decode", calls: [.node("trunk"), .node("sampler")]),
        ],
        emit: [.node("detokenizer", "text")],
        stop: [
            IR.StopCondition(kind: .eosToken, value: 248_044),
            IR.StopCondition(kind: .eosToken, value: 248_046),
            IR.StopCondition(kind: .hostCancel),
            IR.StopCondition(kind: .maxSteps, value: 512),
        ]
    )
}

private func qwen35Gates(
    flow: String,
    elapsedMs: String,
    verifyArgmax: Bool = false,
    includePrefill: Bool = true
) -> [IR.Gate] {
    let inventoryFiles = "manifest.json,weights.bin,model.metallib,SmeltGenerated.swift,dispatches.bin"
        + (includePrefill ? ",prefill_dispatches.bin" : "")
        + (includePrefill && verifyArgmax ? ",prefill_verify_argmax_dispatches.bin" : "")
        + ",tokenizer.json,tokenizer.bin,module.json"
    var gates = [
        IR.Gate(
            id: "startup",
            from: IR.GateEvent(kind: .flowAccepted, flow: flow),
            to: IR.GateEvent(
                kind: .emit,
                flow: flow,
                endpoint: .moduleOutput("text"),
                predicates: [IR.Comparison(subject: "tokens", relation: .greaterThanOrEqual, value: "1")]
            ),
            requirements: [IR.Comparison(subject: "elapsed", relation: .lessThanOrEqual, value: elapsedMs, unit: "ms")],
            measurements: [IR.GateMeasurement(subject: "elapsed", processMode: .cold, cacheState: .cold, occurrence: .first)]
        ),
        IR.Gate(id: "decode", requirements: [IR.Comparison(subject: "decode-output.tokens", relation: .greaterThanOrEqual, value: "1")]),
        IR.Gate(
            id: "inventory",
            requirements: [
                IR.Comparison(
                    subject: "package-files",
                    relation: .include,
                    value: inventoryFiles
                ),
                IR.Comparison(
                    subject: "release-surface-ids",
                    relation: .include,
                    value: includePrefill
                        ? "gate.startup,gate.prefill,gate.decode,release.verify"
                        : "gate.startup,gate.decode,release.verify"
                ),
            ]
        ),
    ]
    if includePrefill {
        gates.insert(
            IR.Gate(
                id: "prefill",
                requirements: [
                    IR.Comparison(
                        subject: "prefill-batch",
                        relation: .lessThanOrEqual,
                        value: "256"
                    ),
                ]
            ),
            at: 1
        )
    }
    return gates
}

private func qwen35Sources(
    repo: String,
    revision: String = "main",
    checkpointMap: String = "hf.qwen"
) -> [IR.Source] {
    [
        IR.Source(
            id: "weights",
            kind: "hf",
            locator: repo,
            revision: revision,
            checkpointMap: checkpointMap
        ),
        IR.Source(
            id: "tokenizer",
            kind: "hf-file",
            locator: "\(repo)/tokenizer.json",
            revision: revision == "main" ? nil : revision
        ),
    ]
}

private func qwen36MTPQuantRules() -> [IR.QuantRule] {
    [
        IR.QuantRule(
            selector: IR.TensorSelector("*_norm_weight", source: "weights"),
            action: .preserve,
            source: "weights",
            priority: 12,
            resolution: .sourceDeferred
        ),
        IR.QuantRule(
            selector: IR.TensorSelector("pre_projection_weight", source: "weights"),
            action: .preserve,
            source: "weights",
            priority: 13,
            resolution: .sourceDeferred
        ),
        IR.QuantRule(
            selector: IR.TensorSelector("*", source: "weights"),
            action: .default,
            storage: IR.QuantStorage(format: .affineU4, groupSize: 64)
        ),
    ]
}

private func qwen35Tensors() -> [IR.TensorMap] {
    [
        IR.TensorMap(
            source: "weights",
            selector: IR.TensorSelector("*", source: "weights"),
            target: IR.TensorTarget(block: "trunk", selector: "*"),
            owner: "trunk"
        ),
    ]
}

private func qwen35QuantRules(
    format: IR.QuantStorageFormat = .affineU4,
    groupSize: Int = 64
) -> [IR.QuantRule] {
    [
        IR.QuantRule(selector: IR.TensorSelector("embeddings", source: "weights"), action: .quantize, source: "weights", priority: 10, resolution: .sourceDeferred),
        IR.QuantRule(selector: IR.TensorSelector("*_norm_weight", source: "weights"), action: .preserve, source: "weights", priority: 12, resolution: .sourceDeferred),
        IR.QuantRule(selector: IR.TensorSelector("conv1d_weight", source: "weights"), action: .preserve, source: "weights", priority: 13, resolution: .sourceDeferred),
        IR.QuantRule(selector: IR.TensorSelector("A_log", source: "weights"), action: .preserve, source: "weights", priority: 5, resolution: .sourceDeferred),
        IR.QuantRule(selector: IR.TensorSelector("dt_bias", source: "weights"), action: .preserve, source: "weights", priority: 7, resolution: .sourceDeferred),
        IR.QuantRule(selector: IR.TensorSelector("*", source: "weights"), action: .default, storage: IR.QuantStorage(format: format, groupSize: groupSize)),
    ]
}

private func qwen35Compile(
    verifyArgmax: Bool = false,
    verifyTokenCapacity: Int? = nil
) -> [IR.Constraint] {
    let prefill = if verifyArgmax {
        "metal verify-argmax batch 256"
            + (verifyTokenCapacity.map { " transaction \($0)" } ?? "")
    } else {
        "metal batch 256"
    }
    return [
        IR.Constraint("target", "metal"),
        IR.Constraint("prefill", prefill),
        IR.Constraint("layout", "memory-neutral"),
    ]
}

private func qwen35DecodeOnlyCompile() -> [IR.Constraint] {
    [
        IR.Constraint("target", "metal"),
        IR.Constraint("layout", "memory-neutral"),
    ]
}

private func qwen35BackendConstraints() -> [IR.Constraint] {
    [IR.Constraint("target", "metal")]
}
