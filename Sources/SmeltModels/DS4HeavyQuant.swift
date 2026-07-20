import SmeltModuleAuthoring

// DS4 heavy-quant: the fail-closed NEGATIVE CANARY. A MoE (router + experts)
// trunk stored GPTQ with a calibration corpus — features Smelt admits as typed
// UNSUPPORTED obligations. Byte parity here proves the ported IR reproduces the
// exact unsupported surface (GPTQ storage, MoE router/experts, calibration).
// Quant priorities (12/19/25/26) and declared-tensor vs source-deferred
// resolutions reproduce the parser's specificity + tensor-match derivation.

func ds4HeavyQuant() -> SmeltCAMIR {
    let caps = ["run.generate"]
    return SmeltCAMIR(
        module: IR.Module(id: "ds4_heavy_quant"),
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
        sources: [
            IR.Source(id: "weights", kind: "hf", locator: "deepseek-ai/DS4", revision: "main"),
            IR.Source(id: "calibration_prompts", kind: "file", locator: "calibration/ds4-prompts.jsonl"),
        ],
        blocks: [ds4Trunk()],
        graphNodes: llmTextGenNodes(tokenizer: plainTokenizerAnnotations, trunkState: "kv-cache"),
        graphEdges: llmTextGenEdges(),
        feedbackEdges: [llmTextGenFeedback()],
        flows: [llmGenerateFlow(stop: [
            IR.StopCondition(kind: .eosToken),
            IR.StopCondition(kind: .maxSteps, value: 1024),
            IR.StopCondition(kind: .hostCancel),
        ])],
        capabilities: caps,
        backendConstraints: [IR.Constraint("target", "metal")],
        tensors: [
            tensorMap(pattern: "model.embed_tokens.*", block: "trunk", target: "embeddings.*"),
            tensorMap(pattern: "model.layers.*.self_attn.*", block: "trunk", target: "attention.*"),
            tensorMap(pattern: "model.layers.*.mlp.router.*", block: "trunk", target: "router.*"),
            tensorMap(pattern: "model.layers.*.mlp.experts.*", block: "trunk", target: "experts.*"),
            tensorMap(pattern: "lm_head.*", block: "trunk", target: "head.*"),
        ],
        quantization: [
            IR.QuantRule(selector: IR.TensorSelector("*_norm_weight", source: "weights"), action: .preserve, source: "weights", priority: 12, resolution: .sourceDeferred),
            IR.QuantRule(selector: IR.TensorSelector("model.layers.*.mlp.router.*", source: "weights"), action: .preserve, source: "weights", priority: 25, resolution: .declaredTensor),
            IR.QuantRule(selector: IR.TensorSelector("model.embed_tokens.*", source: "weights"), action: .store, storage: IR.QuantStorage(format: .turboQuantH, groupSize: 128), source: "weights", priority: 19, resolution: .declaredTensor),
            IR.QuantRule(
                selector: IR.TensorSelector("model.layers.*.mlp.experts.*", source: "weights"),
                action: .store,
                storage: IR.QuantStorage(format: .gptq, groupSize: 128),
                source: "weights",
                priority: 26,
                calibration: ds4Calibration(),
                resolution: .declaredTensor
            ),
            IR.QuantRule(selector: IR.TensorSelector("*", source: "weights"), action: .default, storage: IR.QuantStorage(format: .gptq, groupSize: 128)),
        ],
        compile: [
            IR.Constraint("target", "metal"),
            IR.Constraint("generated-kernels", "auto"),
            IR.Constraint("prefill", "metal batch 128"),
            IR.Constraint("layout", "memory-neutral"),
            IR.Constraint("memory", "peak <= 48 GiB"),
        ],
        gates: [
            startupTimingGate(elapsedMs: "250", measured: false),
            IR.Gate(
                id: "quant_quality",
                requirements: [
                    IR.Comparison(subject: "calibration.gptq.rank", relation: .greaterThanOrEqual, value: "128"),
                    IR.Comparison(subject: "perplexity.delta", relation: .lessThanOrEqual, value: "0.05"),
                ]
            ),
        ]
    )
}

private func ds4Trunk() -> IR.Block {
    IR.Block(
        id: "trunk",
        operatorName: .transformer,
        shape: IR.BlockShape(
            transformer: IR.TransformerShape(
                hiddenSize: 7168,
                layers: IR.LayerPattern(count: 61),
                attention: IR.AttentionShape(
                    qHeads: 128, kvHeads: 8, headDim: 128,
                    rope: IR.RopeShape(kind: .yarn, theta: 1_000_000),
                    qkNorm: .rms
                ),
                router: IR.RouterShape(topK: 8, experts: 256),
                expert: IR.ExpertShape(ffn: IR.FFNShape(dim: 18432, activation: .swiglu)),
                norm: IR.NormShape(kind: .rms, eps: "1e-6", mode: .weight),
                vocab: IR.VocabShape(size: 163_840, tiedHead: false)
            )
        )
    )
}

private func ds4Calibration() -> IR.QuantCalibration {
    IR.QuantCalibration(
        method: .gptq,
        corpus: IR.QuantCalibrationCorpus(source: "calibration_prompts", path: "calibration/ds4-prompts.jsonl", maxTokens: 4096),
        captures: ["trunk.attention", "trunk.experts"],
        layersPerPass: 1,
        requirements: [IR.Comparison(subject: "cosine", relation: .greaterThanOrEqual, value: "0.995")]
    )
}
