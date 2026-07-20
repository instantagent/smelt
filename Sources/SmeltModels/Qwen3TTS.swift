import SmeltModuleAuthoring

// Qwen3-TTS-12Hz: the most complex fixture. A five-block streaming TTS pipeline
// (tts-frontend -> talker -> codec-head -> mtp-head -> codec-decoder) with
// codec-token feedback and a text->44.1kHz-style audio sidecar. Every block
// derives its shape from the weights; the codec/front-end booleans
// (streaming/speaker-conditioning) and the synthesized graph-value ports
// (prompt_hidden, talker_hidden, cb0_token, codec_token) are reproduced exactly
// from the parser's lowering. Quant priorities (8/11/18/23/35/37/46/51) and the
// declared-tensor vs source-deferred resolutions match the specificity +
// tensor-match derivation. Held to byte parity by ModuleAuthoringParityTests.

func qwen3TTS() -> SmeltCAMIR {
    let caps = ["run.synthesize", "run.stream", "bake.voice-defaults"]
    return SmeltCAMIR(
        module: IR.Module(id: "qwen3_tts"),
        exports: [
            IR.Export(
                id: "synth",
                inputs: [port("text", textType()), port("speaker", bareType("voice-id"), optional: true)],
                outputs: [port("audio", pcmType())],
                capabilities: caps,
                gates: ["startup"]
            ),
        ],
        exportBindings: [IR.ExportBinding(export: "synth", flow: "synth")],
        sources: [
            IR.Source(
                id: "weights",
                kind: "hf",
                locator: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                revision: "main",
                checkpointMap: "hf.qwen3-tts-talker-trunk"
            ),
            IR.Source(id: "tokenizer", kind: "hf-file", locator: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice/tokenizer.json"),
            IR.Source(id: "stream_parity", kind: "file", locator: "evidence/streaming-parity.json"),
        ],
        blocks: ttsBlocks(),
        graphNodes: ttsNodes(),
        graphEdges: ttsEdges(),
        feedbackEdges: [IR.FeedbackEdge(from: .node("mtp-head", "codec_token"), to: .node("talker", "codec_token"))],
        flows: [ttsFlow()],
        capabilities: caps,
        backendConstraints: [IR.Constraint("target", "metal")],
        tensors: [
            tensorMap(pattern: "talker.model.text_embedding.weight", block: "tts-frontend", target: "text_embedding"),
            tensorMap(pattern: "talker.text_projection.*", block: "tts-frontend", target: "text_projection.*"),
            tensorMap(pattern: "talker.model.codec_embedding.weight", block: "talker", target: "codec_embedding"),
            tensorMap(pattern: "talker.model.layers.*", block: "talker", target: "layers.*"),
            tensorMap(pattern: "talker.model.norm.weight", block: "talker", target: "norm"),
            tensorMap(pattern: "talker.codec_head.*", block: "codec-head", target: "*"),
            tensorMap(pattern: "talker.code_predictor.*", block: "mtp-head", target: "*"),
            tensorMap(pattern: "decoder.*", block: "codec-decoder", target: "*"),
        ],
        quantization: [
            ttsPreserve("decoder.*", priority: 8, resolution: .declaredTensor),
            ttsPreserve("*norm.weight", priority: 11, resolution: .sourceDeferred),
            ttsPreserve("talker.codec_head.*", priority: 18, resolution: .declaredTensor),
            ttsPreserve("talker.text_projection.*", priority: 23, resolution: .declaredTensor),
            ttsPreserve("talker.model.codec_embedding.weight", priority: 35, resolution: .declaredTensor),
            ttsPreserve("talker.code_predictor.lm_head.*.weight", priority: 37, resolution: .sourceDeferred),
            ttsPreserve("talker.code_predictor.small_to_mtp_projection.*", priority: 46, resolution: .sourceDeferred),
            ttsPreserve("talker.code_predictor.model.codec_embedding.*.weight", priority: 51, resolution: .sourceDeferred),
            IR.QuantRule(selector: IR.TensorSelector("*", source: "weights"), action: .default, storage: IR.QuantStorage(format: .affineU4, groupSize: 128)),
        ],
        compile: [
            IR.Constraint("target", "metal"),
            IR.Constraint("layout", "memory-neutral"),
        ],
        gates: ttsGates()
    )
}

// MARK: - Blocks

private func ttsBlocks() -> [IR.Block] {
    [
        IR.Block(
            id: "tts-frontend",
            operatorName: .ttsFrontend,
            shape: IR.BlockShape(
                derivation: IR.ShapeDerivation(source: "weights"),
                frontend: IR.FrontendShape(speakerConditioning: true),
                requirements: [
                    IR.BlockRequirement("speaker-conditioning", optional: true),
                    IR.BlockRequirement("source-dtype", "bf16"),
                    IR.BlockRequirement("shape-evidence", "config.json"),
                    tensorEvidence("talker.model.text_embedding.weight"),
                    tensorEvidence("talker.text_projection.*"),
                ]
            )
        ),
        IR.Block(
            id: "talker",
            operatorName: .transformer,
            shape: IR.BlockShape(
                derivation: IR.ShapeDerivation(source: "weights"),
                transformer: IR.TransformerShape(),
                requirements: [
                    IR.BlockRequirement("codec-feedback"),
                    IR.BlockRequirement("max-frames", "2048"),
                    IR.BlockRequirement("codec-eos", "2150"),
                    IR.BlockRequirement("source-dtype", "bf16"),
                    tensorEvidence("talker.model.codec_embedding.weight"),
                    tensorEvidence("talker.model.layers.*"),
                    tensorEvidence("talker.model.norm.weight"),
                ]
            )
        ),
        IR.Block(
            id: "codec-head",
            operatorName: .codecHead,
            shape: IR.BlockShape(
                derivation: IR.ShapeDerivation(source: "weights"),
                requirements: [
                    IR.BlockRequirement("sampler"),
                    IR.BlockRequirement("source-dtype", "bf16"),
                    tensorEvidence("talker.codec_head.*"),
                ]
            )
        ),
        IR.Block(
            id: "mtp-head",
            operatorName: .transformer,
            shape: IR.BlockShape(
                derivation: IR.ShapeDerivation(source: "weights"),
                transformer: IR.TransformerShape(),
                requirements: [
                    IR.BlockRequirement("codebooks", "16"),
                    IR.BlockRequirement("residual-codebooks", "16"),
                    IR.BlockRequirement("max-frames", "2048"),
                    IR.BlockRequirement("source-dtype", "bf16"),
                    tensorEvidence("talker.code_predictor.*"),
                ]
            )
        ),
        IR.Block(
            id: "codec-decoder",
            operatorName: .codecDecoder,
            shape: IR.BlockShape(
                derivation: IR.ShapeDerivation(source: "weights"),
                codecDecoder: IR.CodecDecoderShape(streaming: true),
                requirements: [
                    IR.BlockRequirement("streaming"),
                    IR.BlockRequirement("audio-rate", "24khz"),
                    IR.BlockRequirement("audio-format", "pcm-f32"),
                    IR.BlockRequirement("source-dtype", "f32"),
                    tensorEvidence("decoder.*"),
                    tensorEvidence("decoder.quantizer.*"),
                    tensorEvidence("decoder.pre_conv.*"),
                    tensorEvidence("decoder.pre_transformer.*"),
                    tensorEvidence("decoder.upsample.*"),
                    tensorEvidence("decoder.decoder.*"),
                ]
            )
        ),
    ]
}

// MARK: - Graph

private func ttsNodes() -> [IR.GraphNode] {
    [
        IR.GraphNode(
            id: "tts-frontend",
            implementation: .native,
            inputs: [port("text", textType())],
            outputs: [port("prompt_hidden", bareType("prompt_hidden"))],
            annotations: [annot("speaker", "optional")]
        ),
        IR.GraphNode(
            id: "talker",
            implementation: .compiled,
            block: "talker",
            inputs: [port("codec_token", bareType("codec_token")), port("prompt_hidden", bareType("prompt_hidden"))],
            outputs: [port("talker_hidden", bareType("talker_hidden"))],
            annotations: [annot("artifact", "sidecar"), annot("feedback", "codec_token"), annot("state", "kv-cache")]
        ),
        IR.GraphNode(
            id: "codec-head",
            implementation: .native,
            inputs: [port("talker_hidden", bareType("talker_hidden"))],
            outputs: [port("cb0_token", bareType("cb0_token"))],
            annotations: [annot("state", "sampler")]
        ),
        IR.GraphNode(
            id: "mtp-head",
            implementation: .compiled,
            block: "mtp-head",
            inputs: [port("cb0_token", bareType("cb0_token")), port("talker_hidden", bareType("talker_hidden"))],
            outputs: [port("codec_token", bareType("codec_token"))],
            annotations: [annot("artifact", "sidecar"), annot("codebooks", "16")]
        ),
        IR.GraphNode(
            id: "codec-decoder",
            implementation: .compiled,
            block: "codec-decoder",
            inputs: [port("codec_token", bareType("codec_token"))],
            outputs: [port("audio", pcmType())],
            annotations: [annot("artifact", "baked-inline"), annot("streaming", "true")]
        ),
    ]
}

private func ttsEdges() -> [IR.GraphEdge] {
    [
        IR.GraphEdge(from: .moduleInput("text"), to: .node("tts-frontend", "text"), type: textType()),
        IR.GraphEdge(from: .node("tts-frontend", "prompt_hidden"), to: .graphValue("prompt_hidden"), type: bareType("prompt_hidden")),
        IR.GraphEdge(from: .graphValue("prompt_hidden"), to: .node("talker", "prompt_hidden"), type: bareType("prompt_hidden")),
        IR.GraphEdge(from: .node("talker", "talker_hidden"), to: .graphValue("talker_hidden"), type: bareType("talker_hidden")),
        IR.GraphEdge(from: .graphValue("talker_hidden"), to: .node("codec-head", "talker_hidden"), type: bareType("talker_hidden")),
        IR.GraphEdge(from: .node("codec-head", "cb0_token"), to: .graphValue("cb0_token"), type: bareType("cb0_token")),
        IR.GraphEdge(from: .graphValue("talker_hidden"), to: .node("mtp-head", "talker_hidden"), type: bareType("talker_hidden")),
        IR.GraphEdge(from: .node("mtp-head", "codec_token"), to: .graphValue("codec_token"), type: bareType("codec_token")),
        IR.GraphEdge(from: .graphValue("cb0_token"), to: .node("mtp-head", "cb0_token"), type: bareType("cb0_token")),
        IR.GraphEdge(from: .graphValue("codec_token"), to: .node("codec-decoder", "codec_token"), type: bareType("codec_token")),
        IR.GraphEdge(from: .node("codec-decoder", "audio"), to: .moduleOutput("audio"), type: pcmType()),
    ]
}

private func ttsFlow() -> IR.Flow {
    IR.Flow(
        id: "synth",
        phases: [
            IR.FlowPhase(role: .setup, calls: [.node("tts-frontend"), .node("talker"), .node("codec-head")]),
            IR.FlowPhase(role: .step, label: "generate", calls: [.node("mtp-head"), .node("talker"), .node("codec-head")]),
            IR.FlowPhase(role: .step, label: "render", calls: [.node("codec-decoder")]),
        ],
        emit: [.node("codec-decoder", "audio")],
        stop: [
            IR.StopCondition(kind: .codecEOS),
            IR.StopCondition(kind: .maxFrames, value: 2048),
            IR.StopCondition(kind: .hostCancel),
        ]
    )
}

// MARK: - Gates

private func ttsGates() -> [IR.Gate] {
    [
        IR.Gate(
            id: "startup",
            from: IR.GateEvent(kind: .flowAccepted, flow: "synth"),
            to: IR.GateEvent(
                kind: .emit,
                flow: "synth",
                endpoint: .moduleOutput("audio"),
                predicates: [
                    IR.Comparison(subject: "duration", relation: .greaterThanOrEqual, value: "20", unit: "ms"),
                    IR.Comparison(subject: "format", relation: .equal, value: "pcm f32 24khz"),
                ]
            ),
            requirements: [IR.Comparison(subject: "elapsed", relation: .lessThanOrEqual, value: "400", unit: "ms")],
            measurements: [IR.GateMeasurement(subject: "elapsed", processMode: .cold, cacheState: .cold, occurrence: .first)]
        ),
        IR.Gate(
            id: "audio_contract",
            requirements: [
                IR.Comparison(subject: "audio-rate", relation: .equal, value: "24khz"),
                IR.Comparison(subject: "package-files", relation: .include, value: "manifest.json,weights.bin,model.metallib,trunk,trunk-mtp,vocab.json,merges.txt,tokenizer_config.json,config.json,module.json"),
                IR.Comparison(subject: "release-surface-ids", relation: .include, value: "gate.startup-audio,bake.voice-defaults,gate.audio-contract,correctness.stream-parity,release.verify"),
            ]
        ),
        IR.Gate(
            id: "streaming_parity",
            requirements: [],
            evidence: [IR.EvidenceRequirement(kind: .sourceSHA256Recorded, source: "stream_parity")]
        ),
    ]
}

// MARK: - Local sugar

private func pcmType() -> IR.TypeRef { IR.TypeRef("pcm", attributes: ["dtype": "f32", "rate": "24khz"]) }

private func tensorEvidence(_ value: String) -> IR.BlockRequirement { IR.BlockRequirement("tensor-evidence", value) }

private func ttsPreserve(_ pattern: String, priority: Int, resolution: IR.QuantResolution) -> IR.QuantRule {
    IR.QuantRule(selector: IR.TensorSelector(pattern, source: "weights"), action: .preserve, source: "weights", priority: priority, resolution: resolution)
}
