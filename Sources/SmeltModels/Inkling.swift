import SmeltModuleAuthoring

// Inkling is authored from the pinned upstream metadata rather than inferred
// from a downloaded checkpoint. The 975B checkpoint is intentionally not a
// package-build prerequisite: the index is enough to validate inventory and
// to construct full-shape frozen cost reports, while reduced fixtures exercise
// the exact topology.

private let inklingRevision = "e4aa5ee880fbb0d2c1a93b1e4a39f2d4b97eb28a"
private let inklingNVFP4Revision = "1fa46988f638221367b5fdeee4e86d5c9882ae23"
private let inklingTransformersRevision = "28596623762cb409bb1c9234f04bfb1269b1ece1"

func inkling() -> SmeltCAMIR {
    let textCapabilities = ["run.generate", "run.generate.tools"]
    let multimodalCapabilities = [
        "run.generate",
        "run.generate.audio-input",
        "run.generate.image-input",
        "run.generate.multimodal",
        "run.generate.tools",
    ]

    return SmeltCAMIR(
        module: IR.Module(id: "inkling"),
        exports: [
            IR.Export(
                id: "generate",
                inputs: inklingRenderedInputs(),
                outputs: [inklingPort("text", inklingType("text"))],
                capabilities: textCapabilities
            ),
            IR.Export(
                id: "generate_multimodal",
                inputs: inklingRenderedInputs() + [
                    inklingPort("media", inklingMediaType()),
                ],
                outputs: [inklingPort("text", inklingType("text"))],
                capabilities: multimodalCapabilities
            ),
        ],
        exportBindings: [
            IR.ExportBinding(export: "generate", flow: "generate"),
            IR.ExportBinding(export: "generate_multimodal", flow: "generate_multimodal"),
        ],
        sources: inklingSources(),
        blocks: [
            inklingTrunk(),
            inklingImagePatchEncoder(),
            inklingAudioEncoder(),
        ],
        graphNodes: inklingTextNodes() + inklingMultimodalNodes(),
        graphEdges: inklingTextEdges() + inklingMultimodalEdges(),
        feedbackEdges: [
            IR.FeedbackEdge(from: .node("sampler", "tokens"), to: .node("trunk", "tokens")),
            IR.FeedbackEdge(
                from: .node("multimodal-sampler", "tokens"),
                to: .node("multimodal-trunk", "tokens")
            ),
        ],
        flows: [inklingTextFlow(), inklingMultimodalFlow()],
        capabilities: Array(Set(textCapabilities + multimodalCapabilities)),
        backendConstraints: [IR.Constraint("target", "metal")],
        tensors: inklingTensorMaps(),
        quantization: inklingQuantization(),
        compile: [
            IR.Constraint("target", "metal"),
            IR.Constraint("frozen-ir", "required"),
            IR.Constraint("layout", "model-agnostic multi-buffer"),
            IR.Constraint("prefill", "metal tiled full-attention"),
        ],
        artifacts: [
            IR.ArtifactRole(id: "image-patch-component", role: "compiled-component", required: true),
            IR.ArtifactRole(id: "audio-embedding-component", role: "compiled-component", required: true),
        ]
    )
}

/// The checkpoint-owned multi-token predictor is a separate module because it
/// has an independent weight file and lifecycle. It consumes two ordinary
/// hidden-state sources and returns hidden states in the base model space; the
/// base unembedding remains an independently connectable brick.
func inklingMTP() -> SmeltCAMIR {
    let hidden = inklingHiddenType()
    return SmeltCAMIR(
        module: IR.Module(id: "inkling_mtp"),
        exports: [
            IR.Export(
                id: "predict_hidden",
                inputs: [
                    inklingPort("base_hidden", hidden),
                    inklingPort("token_hidden", hidden),
                ],
                outputs: [inklingPort("predicted_hidden", hidden)],
                capabilities: ["run.predict-future-hidden"]
            ),
        ],
        exportBindings: [IR.ExportBinding(export: "predict_hidden", flow: "predict_hidden")],
        sources: inklingSources(),
        blocks: [inklingMTPBlock()],
        graphNodes: [
            IR.GraphNode(
                id: "mtp",
                implementation: .compiled,
                block: "mtp",
                inputs: [
                    inklingPort("base_hidden", hidden),
                    inklingPort("token_hidden", hidden),
                ],
                outputs: [inklingPort("predicted_hidden", hidden)],
                annotations: [
                    inklingAnnotation("artifact", "baked-inline"),
                    inklingAnnotation("tag", "multi-token-predictor"),
                ]
            ),
        ],
        graphEdges: [
            IR.GraphEdge(
                from: .moduleInput("base_hidden"),
                to: .node("mtp", "base_hidden"),
                type: hidden
            ),
            IR.GraphEdge(
                from: .moduleInput("token_hidden"),
                to: .node("mtp", "token_hidden"),
                type: hidden
            ),
            IR.GraphEdge(
                from: .node("mtp", "predicted_hidden"),
                to: .moduleOutput("predicted_hidden"),
                type: hidden
            ),
        ],
        flows: [
            IR.Flow(
                id: "predict_hidden",
                phases: [IR.FlowPhase(role: .step, calls: [.node("mtp")])],
                emit: [.node("mtp", "predicted_hidden")],
                stop: []
            ),
        ],
        capabilities: ["run.predict-future-hidden"],
        backendConstraints: [IR.Constraint("target", "metal")],
        tensors: [
            IR.TensorMap(
                source: "weights",
                selector: IR.TensorSelector("model.mtp.*", source: "weights"),
                target: IR.TensorTarget(block: "mtp", selector: "*"),
                owner: "mtp"
            ),
        ],
        quantization: inklingQuantization(),
        compile: [
            IR.Constraint("target", "metal"),
            IR.Constraint("frozen-ir", "required"),
            IR.Constraint("layout", "model-agnostic multi-buffer"),
        ]
    )
}

// MARK: - Exact architecture bricks

private func inklingTrunk() -> IR.Block {
    IR.Block(
        id: "trunk",
        operatorName: .transformer,
        shape: IR.BlockShape(
            transformer: IR.TransformerShape(
                hiddenSize: 6_144,
                layers: IR.LayerPattern(
                    roles: [.sliding, .sliding, .sliding, .sliding, .sliding, .global],
                    repeatCount: 11
                ),
                attentionByRole: inklingAttentionByRole(),
                ffn: IR.FFNShape(dim: 24_576, activation: .swiglu),
                router: IR.RouterShape(
                    topK: 6,
                    experts: 256,
                    sharedExperts: 2,
                    activation: .sigmoid,
                    normalization: .selectedAndShared,
                    scoreCorrectionBias: true,
                    routeScale: "8",
                    globalScale: true,
                    sharedExpertSink: true
                ),
                expert: IR.ExpertShape(
                    ffn: IR.FFNShape(dim: 3_072, activation: .swiglu)
                ),
                norm: IR.NormShape(kind: .rms, eps: "1e-6", mode: .weight),
                vocab: IR.VocabShape(size: 201_024, tiedHead: false),
                projectionBanks: inklingProjectionBanks(includeLMHead: true),
                denseLayerCount: 2,
                shortConvolutions: inklingShortConvolutions()
            ),
            requirements: [
                IR.BlockRequirement("active-parameters", "41B"),
                IR.BlockRequirement("attention-log-scaling-compute", "f32"),
                IR.BlockRequirement("attention-relative-bias-compute", "f32"),
                IR.BlockRequirement("embedding-norm", "rms"),
                IR.BlockRequirement("max-context", "1048576"),
                IR.BlockRequirement("moe-shared-output-accumulation", "f32"),
                IR.BlockRequirement("padded-vocab", "201024"),
                IR.BlockRequirement("short-convolution-compute", "f32"),
                IR.BlockRequirement("total-parameters", "975B"),
                IR.BlockRequirement("unpadded-output-vocab", "200058"),
            ]
        )
    )
}

private func inklingMTPBlock() -> IR.Block {
    IR.Block(
        id: "mtp",
        operatorName: .transformer,
        shape: IR.BlockShape(
            transformer: IR.TransformerShape(
                hiddenSize: 6_144,
                layers: IR.LayerPattern(
                    roles: [.sliding, .global, .sliding, .global, .sliding, .sliding, .sliding, .sliding],
                    repeatCount: 1
                ),
                attentionByRole: inklingAttentionByRole(),
                ffn: IR.FFNShape(dim: 24_576, activation: .swiglu),
                norm: IR.NormShape(kind: .rms, eps: "1e-6", mode: .weight),
                projectionBanks: inklingProjectionBanks(includeLMHead: false),
                shortConvolutions: inklingShortConvolutions()
            ),
            requirements: [
                IR.BlockRequirement("input-fusion-source-count", "2"),
                IR.BlockRequirement("input-fusion-source-width", "6144"),
                IR.BlockRequirement("input-projection-shape", "6144x12288"),
                IR.BlockRequirement("output-unembedding", "base-module"),
                IR.BlockRequirement("predictor-count", "8"),
                IR.BlockRequirement("short-convolution-compute", "f32"),
            ]
        )
    )
}

private func inklingAttentionByRole() -> [IR.RoleAttentionShape] {
    [
        IR.RoleAttentionShape(
            role: .sliding,
            attention: IR.AttentionShape(
                qHeads: 64,
                kvHeads: 16,
                headDim: 128,
                relativePosition: IR.RelativePositionShape(
                    projectionDim: 16,
                    extent: 512
                ),
                scaling: .inverseHeadDim,
                qkNorm: .rms,
                window: 512
            )
        ),
        IR.RoleAttentionShape(
            role: .global,
            attention: IR.AttentionShape(
                qHeads: 64,
                kvHeads: 8,
                headDim: 128,
                relativePosition: IR.RelativePositionShape(
                    projectionDim: 16,
                    extent: 1_024,
                    logScalingFloor: 128_000,
                    logScalingAlpha: "0.1"
                ),
                scaling: .inverseHeadDim,
                qkNorm: .rms
            )
        ),
    ]
}

private func inklingProjectionBanks(includeLMHead: Bool) -> [IR.ProjectionBank] {
    [
        IR.ProjectionBank(
            id: "attention-input",
            source: .attentionInput,
            outputs: [.attentionQ, .attentionK, .attentionV, .attentionRelative]
        ),
        IR.ProjectionBank(
            id: "attention-output",
            source: .attentionOutput,
            outputs: [.attentionOut]
        ),
        IR.ProjectionBank(
            id: "ffn-input",
            source: .ffnInput,
            outputs: [.ffnGate, .ffnUp]
        ),
        IR.ProjectionBank(
            id: "ffn-intermediate",
            source: .ffnIntermediate,
            outputs: [.ffnDown]
        ),
    ] + (includeLMHead ? [
        IR.ProjectionBank(
            id: "lm-head-input",
            source: .lmHeadInput,
            outputs: [.lmHead]
        ),
    ] : [])
}

private func inklingShortConvolutions() -> [IR.ShortConvolutionShape] {
    [
        IR.ShortConvolutionShape(site: .attentionKey, kernelSize: 4, residual: .addInput),
        IR.ShortConvolutionShape(site: .attentionValue, kernelSize: 4, residual: .addInput),
        IR.ShortConvolutionShape(site: .attentionBranchOutput, kernelSize: 4, residual: .addInput),
        IR.ShortConvolutionShape(site: .ffnBranchOutput, kernelSize: 4, residual: .addInput),
    ]
}

private func inklingImagePatchEncoder() -> IR.Block {
    IR.Block(
        id: "image-patch-encoder",
        operatorName: .patchEncoder,
        shape: IR.BlockShape(requirements: [
            IR.BlockRequirement("channels", "3"),
            IR.BlockRequirement("fold-plan", "1x5x5,1x2x2,1x4x4,2x1x1"),
            IR.BlockRequirement("hidden-size", "6144"),
            IR.BlockRequirement("layer-0", "75x128+rmsnorm128"),
            IR.BlockRequirement("layer-1", "512x320+rmsnorm320"),
            IR.BlockRequirement("layer-2", "5120x4800+rmsnorm4800"),
            IR.BlockRequirement("layer-3", "9600x6144+rmsnorm6144"),
            IR.BlockRequirement("layers", "4"),
            IR.BlockRequirement("patch-size", "40"),
            IR.BlockRequirement("temporal-patch-size", "2"),
        ])
    )
}

private func inklingAudioEncoder() -> IR.Block {
    IR.Block(
        id: "audio-encoder",
        operatorName: .discreteAudioEncoder,
        shape: IR.BlockShape(requirements: [
            IR.BlockRequirement("encoding", "dMel"),
            IR.BlockRequirement("embedding-shape", "1280x6144"),
            IR.BlockRequirement("fft-size", "1600"),
            IR.BlockRequirement("hidden-size", "6144"),
            IR.BlockRequirement("hop-size", "800"),
            IR.BlockRequirement("mel-bins", "80"),
            IR.BlockRequirement("sample-rate", "16000"),
            IR.BlockRequirement("token-duration-ms", "50"),
            IR.BlockRequirement("values-per-token", "16"),
            IR.BlockRequirement("window-size", "1600"),
        ])
    )
}

// MARK: - Graph

private func inklingRenderedInputs() -> [IR.Port] {
    [
        inklingPort("prompt", inklingType("text")),
        inklingPort("tools", inklingType("tools"), optional: true),
        inklingPort("reasoning_effort", inklingReasoningEffortType(), optional: true),
    ]
}

private func inklingTextNodes() -> [IR.GraphNode] {
    [
        inklingRendererNode(id: "renderer"),
        inklingTokenizerNode(id: "tokenizer"),
        inklingTrunkNode(id: "trunk", prefill: false),
        inklingSamplerNode(id: "sampler"),
        inklingDetokenizerNode(id: "detokenizer"),
    ]
}

private func inklingTextEdges() -> [IR.GraphEdge] {
    let rendered = inklingType("rendered-prompt")
    let tokens = inklingType("tokens")
    let hidden = inklingHiddenType()
    let text = inklingType("text")
    return [
        IR.GraphEdge(from: .moduleInput("prompt"), to: .node("renderer", "prompt"), type: text),
        IR.GraphEdge(from: .moduleInput("tools"), to: .node("renderer", "tools"), type: inklingType("tools")),
        IR.GraphEdge(
            from: .moduleInput("reasoning_effort"),
            to: .node("renderer", "reasoning_effort"),
            type: inklingReasoningEffortType()
        ),
        IR.GraphEdge(
            from: .node("renderer", "rendered_prompt"),
            to: .node("tokenizer", "rendered_prompt"),
            type: rendered
        ),
        IR.GraphEdge(from: .node("tokenizer", "tokens"), to: .node("trunk", "tokens"), type: tokens),
        IR.GraphEdge(from: .node("trunk", "hidden"), to: .node("sampler", "hidden"), type: hidden),
        IR.GraphEdge(from: .node("sampler", "tokens"), to: .node("detokenizer", "tokens"), type: tokens),
        IR.GraphEdge(from: .node("detokenizer", "text"), to: .moduleOutput("text"), type: text),
    ]
}

private func inklingMultimodalNodes() -> [IR.GraphNode] {
    let patchTensor = inklingType("tensor", ["dtype": "f32", "layout": "patch-major"])
    let audioTokens = inklingType("tensor", ["dtype": "u16", "encoding": "dMel"])
    let optionalEmbeddings = inklingType("embeddings", ["dim": "6144", "optional": "true"])
    return [
        inklingRendererNode(id: "multimodal-renderer"),
        inklingTokenizerNode(id: "multimodal-tokenizer"),
        IR.GraphNode(
            id: "media-preprocessor",
            implementation: .native,
            inputs: [inklingPort("media", inklingMediaType())],
            outputs: [
                inklingPort("image_patches", patchTensor, optional: true),
                inklingPort("audio_tokens", audioTokens, optional: true),
            ],
            annotations: [
                inklingAnnotation("audio-hop-size", "800"),
                inklingAnnotation("audio-mel-bins", "80"),
                inklingAnnotation("audio-sample-rate", "16000"),
                inklingAnnotation("image-mean", "0.48145466,0.4578275,0.40821073"),
                inklingAnnotation("image-patch-size", "40"),
                inklingAnnotation("image-resample", "bicubic"),
                inklingAnnotation("image-std", "0.26862954,0.26130258,0.27577711"),
                inklingAnnotation("tag", "media-preprocessor"),
            ]
        ),
        IR.GraphNode(
            id: "image-patch-encoder",
            implementation: .compiled,
            block: "image-patch-encoder",
            inputs: [inklingPort("image_patches", patchTensor, optional: true)],
            outputs: [inklingPort("image_embeddings", optionalEmbeddings, optional: true)],
            annotations: [
                inklingAnnotation("artifact", "image-patch-component"),
                inklingAnnotation("tag", "patch-encoder"),
            ]
        ),
        IR.GraphNode(
            id: "audio-encoder",
            implementation: .compiled,
            block: "audio-encoder",
            inputs: [inklingPort("audio_tokens", audioTokens, optional: true)],
            outputs: [inklingPort("audio_embeddings", optionalEmbeddings, optional: true)],
            annotations: [
                inklingAnnotation("artifact", "audio-embedding-component"),
                inklingAnnotation("tag", "discrete-audio-encoder"),
            ]
        ),
        IR.GraphNode(
            id: "multimodal-token-fusion",
            implementation: .native,
            inputs: [
                inklingPort("tokens", inklingType("tokens")),
                inklingPort("image_embeddings", optionalEmbeddings, optional: true),
                inklingPort("audio_embeddings", optionalEmbeddings, optional: true),
            ],
            outputs: [inklingPort("prefill_state", inklingType("state"))],
            annotations: [
                inklingAnnotation("audio-token-id", "200053"),
                inklingAnnotation("image-token-id", "200054"),
                inklingAnnotation("tag", "multimodal-token-fusion"),
            ]
        ),
        inklingTrunkNode(id: "multimodal-trunk", prefill: true),
        inklingSamplerNode(id: "multimodal-sampler"),
        inklingDetokenizerNode(id: "multimodal-detokenizer"),
    ]
}

private func inklingMultimodalEdges() -> [IR.GraphEdge] {
    let text = inklingType("text")
    let tools = inklingType("tools")
    let tokens = inklingType("tokens")
    let hidden = inklingHiddenType()
    let patchTensor = inklingType("tensor", ["dtype": "f32", "layout": "patch-major"])
    let audioTokens = inklingType("tensor", ["dtype": "u16", "encoding": "dMel"])
    let optionalEmbeddings = inklingType("embeddings", ["dim": "6144", "optional": "true"])
    return [
        IR.GraphEdge(from: .moduleInput("prompt"), to: .node("multimodal-renderer", "prompt"), type: text),
        IR.GraphEdge(from: .moduleInput("tools"), to: .node("multimodal-renderer", "tools"), type: tools),
        IR.GraphEdge(
            from: .moduleInput("reasoning_effort"),
            to: .node("multimodal-renderer", "reasoning_effort"),
            type: inklingReasoningEffortType()
        ),
        IR.GraphEdge(
            from: .node("multimodal-renderer", "rendered_prompt"),
            to: .node("multimodal-tokenizer", "rendered_prompt"),
            type: inklingType("rendered-prompt")
        ),
        IR.GraphEdge(from: .moduleInput("media"), to: .node("media-preprocessor", "media"), type: inklingMediaType()),
        IR.GraphEdge(
            from: .node("media-preprocessor", "image_patches"),
            to: .node("image-patch-encoder", "image_patches"),
            type: patchTensor
        ),
        IR.GraphEdge(
            from: .node("media-preprocessor", "audio_tokens"),
            to: .node("audio-encoder", "audio_tokens"),
            type: audioTokens
        ),
        IR.GraphEdge(
            from: .node("multimodal-tokenizer", "tokens"),
            to: .node("multimodal-token-fusion", "tokens"),
            type: tokens
        ),
        IR.GraphEdge(
            from: .node("image-patch-encoder", "image_embeddings"),
            to: .node("multimodal-token-fusion", "image_embeddings"),
            type: optionalEmbeddings
        ),
        IR.GraphEdge(
            from: .node("audio-encoder", "audio_embeddings"),
            to: .node("multimodal-token-fusion", "audio_embeddings"),
            type: optionalEmbeddings
        ),
        IR.GraphEdge(
            from: .node("multimodal-token-fusion", "prefill_state"),
            to: .node("multimodal-trunk", "prefill_state"),
            type: inklingType("state")
        ),
        IR.GraphEdge(from: .node("multimodal-trunk", "hidden"), to: .node("multimodal-sampler", "hidden"), type: hidden),
        IR.GraphEdge(from: .node("multimodal-sampler", "tokens"), to: .node("multimodal-detokenizer", "tokens"), type: tokens),
        IR.GraphEdge(from: .node("multimodal-detokenizer", "text"), to: .moduleOutput("text"), type: text),
    ]
}

private func inklingRendererNode(id: String) -> IR.GraphNode {
    IR.GraphNode(
        id: id,
        implementation: .native,
        inputs: inklingRenderedInputs(),
        outputs: [inklingPort("rendered_prompt", inklingType("rendered-prompt"))],
        annotations: [
            inklingAnnotation("assistant-prelude", "thinking"),
            inklingAnnotation("default-reasoning-effort", "0.9"),
            inklingAnnotation("prompt-format", "inkling-renderer"),
            inklingAnnotation("reasoning-effort-map", "none=0,minimal=0.1,low=0.2,medium=0.7,high=0.9,max=0.99"),
            inklingAnnotation("reasoning-effort-range", "0...0.99"),
            inklingAnnotation("tag", "prompt-renderer"),
            inklingAnnotation("tool-format", SmeltToolTranscriptCodecName.inkling),
        ]
    )
}

private func inklingTokenizerNode(id: String) -> IR.GraphNode {
    IR.GraphNode(
        id: id,
        implementation: .native,
        inputs: [inklingPort("rendered_prompt", inklingType("rendered-prompt"))],
        outputs: [inklingPort("tokens", inklingType("tokens"))],
        annotations: [inklingAnnotation("tag", "text-tokenizer")]
    )
}

private func inklingTrunkNode(id: String, prefill: Bool) -> IR.GraphNode {
    IR.GraphNode(
        id: id,
        implementation: .compiled,
        block: "trunk",
        inputs: (prefill ? [inklingPort("prefill_state", inklingType("state"))] : []) + [
            inklingPort("tokens", inklingType("tokens")),
        ],
        outputs: [inklingPort("hidden", inklingHiddenType())],
        annotations: [
            inklingAnnotation("artifact", "baked-inline"),
            inklingAnnotation("feedback", "tokens"),
            inklingAnnotation("state", "kv-cache,short-convolution-state"),
            inklingAnnotation("tag", "decode-core"),
        ]
    )
}

private func inklingSamplerNode(id: String) -> IR.GraphNode {
    IR.GraphNode(
        id: id,
        implementation: .native,
        inputs: [inklingPort("hidden", inklingHiddenType())],
        outputs: [inklingPort("tokens", inklingType("tokens"))],
        annotations: [
            inklingAnnotation("output-vocab", "200058"),
            inklingAnnotation("state", "sampler"),
            inklingAnnotation("tag", "sampler"),
        ]
    )
}

private func inklingDetokenizerNode(id: String) -> IR.GraphNode {
    IR.GraphNode(
        id: id,
        implementation: .native,
        inputs: [inklingPort("tokens", inklingType("tokens"))],
        outputs: [inklingPort("text", inklingType("text"))],
        annotations: [inklingAnnotation("tag", "text-detokenizer")]
    )
}

private func inklingTextFlow() -> IR.Flow {
    IR.Flow(
        id: "generate",
        phases: [
            IR.FlowPhase(role: .setup, calls: [.node("renderer"), .node("tokenizer")]),
            IR.FlowPhase(role: .step, label: "decode", calls: [.node("trunk"), .node("sampler")]),
        ],
        emit: [.node("detokenizer", "text")],
        stop: inklingStopConditions()
    )
}

private func inklingMultimodalFlow() -> IR.Flow {
    IR.Flow(
        id: "generate_multimodal",
        phases: [
            IR.FlowPhase(
                role: .setup,
                calls: [
                    .node("multimodal-renderer"),
                    .node("multimodal-tokenizer"),
                    .node("media-preprocessor"),
                    .node("image-patch-encoder"),
                    .node("audio-encoder"),
                    .node("multimodal-token-fusion"),
                ]
            ),
            IR.FlowPhase(
                role: .step,
                label: "decode",
                calls: [.node("multimodal-trunk"), .node("multimodal-sampler")]
            ),
        ],
        emit: [.node("multimodal-detokenizer", "text")],
        stop: inklingStopConditions()
    )
}

private func inklingStopConditions() -> [IR.StopCondition] {
    [
        IR.StopCondition(kind: .eosToken, value: 200_006),
        IR.StopCondition(kind: .hostCancel),
        IR.StopCondition(kind: .maxSteps, value: 4_096),
    ]
}

// MARK: - Sources, tensors, and policy

private func inklingSources() -> [IR.Source] {
    let repo = "thinkingmachines/Inkling"
    return [
        IR.Source(
            id: "weights",
            kind: "hf",
            locator: repo,
            revision: inklingRevision
        ),
        IR.Source(
            id: "weights-nvfp4",
            kind: "hf",
            locator: repo,
            revision: inklingNVFP4Revision
        ),
        IR.Source(
            id: "config",
            kind: "hf-file",
            locator: "\(repo)/config.json",
            revision: inklingRevision,
            sha256: "58720f145bcecef9a7ab2b419ab346e7c634af8d2f3e7362e900d00f789ea46c",
            byteCount: 2_415
        ),
        IR.Source(
            id: "checkpoint-index",
            kind: "hf-file",
            locator: "\(repo)/model.safetensors.index.json",
            revision: inklingRevision,
            sha256: "6bdebc2a928b1be96e1666b40704a4222ee0c764c2611247bb8ad4d485ea9a97",
            byteCount: 128_600
        ),
        IR.Source(
            id: "checkpoint-index-nvfp4",
            kind: "hf-file",
            locator: "\(repo)/model.safetensors.index.json",
            revision: inklingNVFP4Revision,
            sha256: "23e5784d569500563c2603920d0d2e7240caab5daa627bb5928e3d092fc6e560"
        ),
        IR.Source(
            id: "processor",
            kind: "hf-file",
            locator: "\(repo)/processor_config.json",
            revision: inklingRevision,
            sha256: "b4a3962ea5f7ec39f40b5cf14e57ce99776c3dcce4756a110f7a169809e3a04c",
            byteCount: 1_110
        ),
        IR.Source(id: "tokenizer", kind: "hf-file", locator: "\(repo)/tokenizer.json", revision: inklingRevision),
        IR.Source(
            id: "tokenizer-config",
            kind: "hf-file",
            locator: "\(repo)/tokenizer_config.json",
            revision: inklingRevision,
            sha256: "2e36c9748a2081abb935b2e745ee22e82efa32589c2500df7e5bc0f93145cd77",
            byteCount: 12_111
        ),
        IR.Source(
            id: "special-tokens",
            kind: "hf-file",
            locator: "\(repo)/special_tokens_map.json",
            revision: inklingRevision,
            sha256: "abc97715b4b3b30eb65ea6895afd7b529c32bd10c28901f9ddae7edc39723b0f",
            byteCount: 517
        ),
        IR.Source(
            id: "renderer",
            kind: "hf-file",
            locator: "\(repo)/chat_template.jinja",
            revision: inklingRevision,
            sha256: "0aa1aa0c729d90176dcaa00c440c8faffca2957ffb2cc4b79456ee6d02bcf43b",
            byteCount: 6_294
        ),
        IR.Source(
            id: "reference",
            kind: "git",
            locator: "huggingface/transformers/src/transformers/models/inkling",
            revision: inklingTransformersRevision
        ),
    ]
}

private func inklingTensorMaps() -> [IR.TensorMap] {
    [
        IR.TensorMap(
            source: "weights",
            selector: IR.TensorSelector("model.llm.*", source: "weights"),
            target: IR.TensorTarget(block: "trunk", selector: "*"),
            owner: "trunk"
        ),
        IR.TensorMap(
            source: "weights",
            selector: IR.TensorSelector("model.embed*", source: "weights"),
            target: IR.TensorTarget(block: "trunk", selector: "embedding.*"),
            owner: "trunk"
        ),
        IR.TensorMap(
            source: "weights",
            selector: IR.TensorSelector("model.norm.*", source: "weights"),
            target: IR.TensorTarget(block: "trunk", selector: "norm.*"),
            owner: "trunk"
        ),
        IR.TensorMap(
            source: "weights",
            selector: IR.TensorSelector("model.unembed.*", source: "weights"),
            target: IR.TensorTarget(block: "trunk", selector: "lm_head.*"),
            owner: "trunk"
        ),
        IR.TensorMap(
            source: "weights",
            selector: IR.TensorSelector("model.visual.*", source: "weights"),
            target: IR.TensorTarget(block: "image-patch-encoder", selector: "*"),
            owner: "image-patch-encoder"
        ),
        IR.TensorMap(
            source: "weights",
            selector: IR.TensorSelector("model.audio.*", source: "weights"),
            target: IR.TensorTarget(block: "audio-encoder", selector: "*"),
            owner: "audio-encoder"
        ),
    ]
}

private func inklingQuantization() -> [IR.QuantRule] {
    [
        IR.QuantRule(
            selector: IR.TensorSelector("*norm*", source: "weights"),
            action: .preserve,
            source: "weights",
            priority: 20,
            resolution: .sourceDeferred
        ),
        IR.QuantRule(
            selector: IR.TensorSelector("*sconv*", source: "weights"),
            action: .preserve,
            source: "weights",
            priority: 20,
            resolution: .sourceDeferred
        ),
        IR.QuantRule(
            selector: IR.TensorSelector("*relative*", source: "weights"),
            action: .preserve,
            source: "weights",
            priority: 20,
            resolution: .sourceDeferred
        ),
        IR.QuantRule(
            selector: IR.TensorSelector("*", source: "weights"),
            action: .default,
            storage: IR.QuantStorage(format: .affineU4, groupSize: 64)
        ),
    ]
}

private func inklingMediaType() -> IR.TypeRef {
    inklingType("media", ["kinds": "audio,image"])
}

private func inklingHiddenType() -> IR.TypeRef {
    inklingType("hidden", ["dim": "6144", "dtype": "bf16"])
}

private func inklingReasoningEffortType() -> IR.TypeRef {
    inklingType("reasoning-effort", [
        "default": "0.9",
        "max": "0.99",
        "min": "0",
        "named": "none,minimal,low,medium,high,max",
    ])
}

private func inklingType(_ name: String, _ attributes: [String: String] = [:]) -> IR.TypeRef {
    if name == "text", attributes["encoding"] == nil {
        return IR.TypeRef(name, attributes: attributes.merging(["encoding": "utf8"]) { current, _ in current })
    }
    return IR.TypeRef(name, attributes: attributes)
}

private func inklingPort(_ name: String, _ type: IR.TypeRef, optional: Bool = false) -> IR.Port {
    IR.Port(name: name, type: type, optional: optional)
}

private func inklingAnnotation(_ key: String, _ value: String) -> IR.Constraint {
    IR.Constraint(key, value)
}
