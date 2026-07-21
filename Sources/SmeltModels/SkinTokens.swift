import SmeltModuleAuthoring

private struct SkinTokensTensorInventory {
    enum Disposition: String {
        case carried
        case trainingOnly
    }

    enum Component: String {
        case vae
        case meshEncoder
        case qwen
        case outputProjection
    }

    struct ExpectedTensor {
        let name: String
        let shape: [Int]
        let component: Component
        let disposition: Disposition

        init(
            name: String,
            shape: [Int],
            component: Component,
            disposition: Disposition
        ) {
            self.name = name
            self.shape = shape
            self.component = component
            self.disposition = disposition
        }
    }

    /// Full pinned checkpoint inventory. Training-only exclusions are encoded
    /// here beside active tensors so an upstream architecture change cannot be
    /// silently dropped by a prefix filter.
    public static let expectedTensors: [ExpectedTensor] = {
        var result: [ExpectedTensor] = []
        result += vaeEncoder(prefix: "vae.model.encoder", learned: true, carried: false)
        result += vaeEncoder(prefix: "vae.model.cond_encoder", learned: false, carried: true)
        result += vaeDecoder()
        result += [
            tensor("vae.model.cond_quant.weight", [512, 768], .vae, .carried),
            tensor("vae.model.cond_quant.bias", [512], .vae, .carried),
            tensor("vae.model.quant.weight", [512, 768], .vae, .trainingOnly),
            tensor("vae.model.quant.bias", [512], .vae, .trainingOnly),
            tensor("vae.model.post_quant.weight", [768, 512], .vae, .carried),
            tensor("vae.model.post_quant.bias", [768], .vae, .carried),
            tensor("vae.model.FSQ.project_in.weight", [5, 512], .vae, .trainingOnly),
            tensor("vae.model.FSQ.project_in.bias", [5], .vae, .trainingOnly),
            tensor("vae.model.FSQ.project_out.weight", [512, 5], .vae, .carried),
            tensor("vae.model.FSQ.project_out.bias", [512], .vae, .carried),
        ]
        result += meshEncoder()
        result += qwen()
        result += [
            tensor("output_proj.0.weight", [896, 512], .outputProjection, .carried),
            tensor("output_proj.0.bias", [896], .outputProjection, .carried),
            tensor("output_proj.1.weight", [896], .outputProjection, .carried),
        ]
        precondition(result.count == 672, "skinning component expected inventory must contain 672 tensors")
        return result
    }()

    private static func tensor(
        _ name: String,
        _ shape: [Int],
        _ component: Component,
        _ disposition: Disposition
    ) -> ExpectedTensor {
        ExpectedTensor(
            name: name,
            shape: shape,
            component: component,
            disposition: disposition
        )
    }

    private static func pair(
        _ prefix: String,
        weight: [Int],
        bias: [Int],
        component: Component,
        disposition: Disposition
    ) -> [ExpectedTensor] {
        [
            tensor("\(prefix).weight", weight, component, disposition),
            tensor("\(prefix).bias", bias, component, disposition),
        ]
    }

    private static func vaeEncoder(
        prefix: String,
        learned: Bool,
        carried: Bool
    ) -> [ExpectedTensor] {
        let disposition: Disposition = carried ? .carried : .trainingOnly
        var result: [ExpectedTensor] = []
        if learned {
            result.append(tensor("\(prefix).learned_queries", [32, 768], .vae, disposition))
        }
        result += pair(
            "\(prefix).proj_in",
            weight: [768, learned ? 55 : 54],
            bias: [768],
            component: .vae,
            disposition: disposition
        )
        result += vaeCrossBlock("\(prefix).blocks.0", disposition: disposition)
        result += vaeSelfBlock("\(prefix).blocks.1", disposition: disposition)
        result += vaeSelfBlock("\(prefix).blocks.2", disposition: disposition)
        result += pair(
            "\(prefix).norm_out",
            weight: [768],
            bias: [768],
            component: .vae,
            disposition: disposition
        )
        return result
    }

    private static func vaeDecoder() -> [ExpectedTensor] {
        var result: [ExpectedTensor] = []
        for layer in 0..<10 {
            result += vaeSelfBlock("vae.model.decoder.blocks.\(layer)", disposition: .carried)
        }
        result += vaeCrossBlock("vae.model.decoder.blocks.10", disposition: .carried)
        result += pair(
            "vae.model.decoder.proj_query",
            weight: [768, 54],
            bias: [768],
            component: .vae,
            disposition: .carried
        )
        result += pair(
            "vae.model.decoder.norm_out",
            weight: [768],
            bias: [768],
            component: .vae,
            disposition: .carried
        )
        result += pair(
            "vae.model.decoder.proj_out",
            weight: [1, 768],
            bias: [1],
            component: .vae,
            disposition: .carried
        )
        return result
    }

    private static func vaeSelfBlock(
        _ prefix: String,
        disposition: Disposition
    ) -> [ExpectedTensor] {
        var result = pair(
            "\(prefix).norm1",
            weight: [768],
            bias: [768],
            component: .vae,
            disposition: disposition
        )
        for projection in ["to_q", "to_k", "to_v"] {
            result.append(
                tensor(
                    "\(prefix).attn1.\(projection).weight",
                    [768, 768],
                    .vae,
                    disposition
                )
            )
        }
        result += pair(
            "\(prefix).attn1.to_out.0",
            weight: [768, 768],
            bias: [768],
            component: .vae,
            disposition: disposition
        )
        result += vaeFeedForward(prefix, disposition: disposition)
        return result
    }

    private static func vaeCrossBlock(
        _ prefix: String,
        disposition: Disposition
    ) -> [ExpectedTensor] {
        var result = pair(
            "\(prefix).norm2",
            weight: [768],
            bias: [768],
            component: .vae,
            disposition: disposition
        )
        result += pair(
            "\(prefix).attn2.norm_cross",
            weight: [768],
            bias: [768],
            component: .vae,
            disposition: disposition
        )
        for projection in ["to_q", "to_k", "to_v"] {
            result.append(
                tensor(
                    "\(prefix).attn2.\(projection).weight",
                    [768, 768],
                    .vae,
                    disposition
                )
            )
        }
        result += pair(
            "\(prefix).attn2.to_out.0",
            weight: [768, 768],
            bias: [768],
            component: .vae,
            disposition: disposition
        )
        result += vaeFeedForward(prefix, disposition: disposition)
        return result
    }

    private static func vaeFeedForward(
        _ prefix: String,
        disposition: Disposition
    ) -> [ExpectedTensor] {
        var result = pair(
            "\(prefix).norm3",
            weight: [768],
            bias: [768],
            component: .vae,
            disposition: disposition
        )
        result += pair(
            "\(prefix).ff.net.0.proj",
            weight: [3_072, 768],
            bias: [3_072],
            component: .vae,
            disposition: disposition
        )
        result += pair(
            "\(prefix).ff.net.2",
            weight: [768, 3_072],
            bias: [768],
            component: .vae,
            disposition: disposition
        )
        return result
    }

    private static func meshEncoder() -> [ExpectedTensor] {
        let component = Component.meshEncoder
        let disposition = Disposition.carried
        let prefix = "mesh_encoder.encoder"
        var result = pair(
            "\(prefix).input_proj",
            weight: [512, 54],
            bias: [512],
            component: component,
            disposition: disposition
        )
        result += [
            tensor("\(prefix).cross_attn.attn.c_q.weight", [512, 512], component, disposition),
            tensor("\(prefix).cross_attn.attn.c_kv.weight", [1_024, 512], component, disposition),
        ]
        result += pair(
            "\(prefix).cross_attn.attn.c_proj",
            weight: [512, 512],
            bias: [512],
            component: component,
            disposition: disposition
        )
        for norm in ["ln_1", "ln_2"] {
            result += pair(
                "\(prefix).cross_attn.\(norm)",
                weight: [512],
                bias: [512],
                component: component,
                disposition: disposition
            )
        }
        result += pair(
            "\(prefix).cross_attn.mlp.c_fc",
            weight: [2_048, 512],
            bias: [2_048],
            component: component,
            disposition: disposition
        )
        result += pair(
            "\(prefix).cross_attn.mlp.c_proj",
            weight: [512, 2_048],
            bias: [512],
            component: component,
            disposition: disposition
        )
        result += pair(
            "\(prefix).cross_attn.ln_3",
            weight: [512],
            bias: [512],
            component: component,
            disposition: disposition
        )
        for layer in 0..<8 {
            let block = "\(prefix).self_attn.resblocks.\(layer)"
            result.append(
                tensor("\(block).attn.c_qkv.weight", [1_536, 512], component, disposition)
            )
            result += pair(
                "\(block).attn.c_proj",
                weight: [512, 512],
                bias: [512],
                component: component,
                disposition: disposition
            )
            result += pair(
                "\(block).ln_1",
                weight: [512],
                bias: [512],
                component: component,
                disposition: disposition
            )
            result += pair(
                "\(block).mlp.c_fc",
                weight: [2_048, 512],
                bias: [2_048],
                component: component,
                disposition: disposition
            )
            result += pair(
                "\(block).mlp.c_proj",
                weight: [512, 2_048],
                bias: [512],
                component: component,
                disposition: disposition
            )
            result += pair(
                "\(block).ln_2",
                weight: [512],
                bias: [512],
                component: component,
                disposition: disposition
            )
        }
        result += pair(
            "\(prefix).ln_post",
            weight: [512],
            bias: [512],
            component: component,
            disposition: disposition
        )
        return result
    }

    private static func qwen() -> [ExpectedTensor] {
        let component = Component.qwen
        let disposition = Disposition.carried
        var result = [
            tensor("transformer.model.embed_tokens.weight", [33_036, 896], component, disposition),
        ]
        for layer in 0..<28 {
            let prefix = "transformer.model.layers.\(layer)"
            result += [
                tensor("\(prefix).self_attn.q_proj.weight", [2_048, 896], component, disposition),
                tensor("\(prefix).self_attn.k_proj.weight", [1_024, 896], component, disposition),
                tensor("\(prefix).self_attn.v_proj.weight", [1_024, 896], component, disposition),
                tensor("\(prefix).self_attn.o_proj.weight", [896, 2_048], component, disposition),
                tensor("\(prefix).self_attn.q_norm.weight", [128], component, disposition),
                tensor("\(prefix).self_attn.k_norm.weight", [128], component, disposition),
                tensor("\(prefix).mlp.gate_proj.weight", [3_072, 896], component, disposition),
                tensor("\(prefix).mlp.up_proj.weight", [3_072, 896], component, disposition),
                tensor("\(prefix).mlp.down_proj.weight", [896, 3_072], component, disposition),
                tensor("\(prefix).input_layernorm.weight", [896], component, disposition),
                tensor("\(prefix).post_attention_layernorm.weight", [896], component, disposition),
            ]
        }
        result += [
            tensor("transformer.model.norm.weight", [896], component, disposition),
            tensor("transformer.lm_head.weight", [33_036, 896], component, disposition),
        ]
        return result
    }
}

func skinTokensArticulation() -> SmeltCAMIR {
    let glb = IR.TypeRef("artifact", attributes: ["media-type": "model/gltf-binary"])
    let mesh = bareType("triangle-mesh")
    let sampled = bareType("sampled-surface")
    let meshEncoding = bareType("mesh-encoding")
    let condition = bareType("condition-encoding")
    let generation = bareType("skeleton-generation")
    let queries = bareType("selected-surface-queries")
    let transferPlan = bareType("skin-transfer-plan")
    let skinField = bareType("gpu-skin-field")
    let vertexSkin = bareType("vertex-skin")
    let run = SmeltPackageRunContract(
        export: "transform",
        entrypoint: "transform",
        input: .init(
            flag: "input",
            mediaTypes: ["model/gltf-binary"],
            fileExtensions: ["glb"],
            help: "Source triangle mesh"
        ),
        output: .init(
            flag: "output",
            mediaTypes: ["model/gltf-binary"],
            fileExtensions: ["glb"],
            help: "Generated skinned mesh"
        ),
        options: [
            .init(
                flag: "skeleton-tokens",
                value: .string,
                help: "Comma-separated skeleton prefix tokens"
            ),
            .init(
                flag: "sampling-seed",
                value: .unsignedInteger,
                defaultValue: "0",
                help: "Surface point-sampling seed"
            ),
            .init(
                flag: "sample-seed",
                value: .unsignedInteger,
                help: "Enable sampled skeleton decoding with this seed"
            ),
            .init(
                flag: "beam-count",
                value: .positiveInteger,
                defaultValue: "10",
                help: "Sampled decoding beam width"
            ),
        ]
    )
    let nodes: [IR.GraphNode] = [
        nativeNode("glb.decode", inputs: [port("input", glb)], outputs: [port("mesh", mesh)]),
        nativeNode(
            "surface.sample",
            inputs: [port("mesh", mesh)],
            outputs: [port("sampled", sampled)]
        ),
        nativeNode(
            "mesh.encode",
            inputs: [port("sampled", sampled)],
            outputs: [port("encoding", meshEncoding)]
        ),
        nativeNode(
            "condition.encode",
            inputs: [port("sampled", sampled)],
            outputs: [port("condition", condition)]
        ),
        nativeNode(
            "sequence.generate",
            inputs: [port("encoding", meshEncoding)],
            outputs: [port("generation", generation)],
            annotations: [annot("sidecar", "language-trunk")]
        ),
        nativeNode(
            "skin.neighbors",
            inputs: [port("mesh", mesh), port("sampled", sampled)],
            outputs: [port("queries", queries), port("plan", transferPlan)]
        ),
        nativeNode(
            "skin.decode",
            inputs: [
                port("generation", generation),
                port("condition", condition),
                port("queries", queries),
            ],
            outputs: [port("skin", skinField)]
        ),
        nativeNode(
            "skin.transfer",
            inputs: [port("plan", transferPlan), port("skin", skinField)],
            outputs: [port("vertex-skin", vertexSkin)]
        ),
        nativeNode(
            "glb.encode",
            inputs: [
                port("mesh", mesh),
                port("sampled", sampled),
                port("generation", generation),
                port("vertex-skin", vertexSkin),
            ],
            outputs: [port("output", glb)]
        ),
    ]
    return SmeltCAMIR(
        module: .init(id: "skintokens_articulation"),
        run: run,
        exports: [
            .init(
                id: "transform",
                inputs: [port("input", glb)],
                outputs: [port("output", glb)],
                capabilities: ["run.transform"]
            ),
        ],
        exportBindings: [.init(export: "transform", flow: "transform")],
        sources: [
            .init(
                id: "checkpoint",
                kind: "pytorch-checkpoint",
                locator: "huggingface://VAST-AI/SkinTokens/experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt",
                revision: "79736cad0fd84de384d5eede659b4ebd24effe33",
                checkpointMap: "identity",
                sha256: "f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692"
            ),
        ],
        blocks: skinTokensBlocks(),
        graphNodes: nodes,
        graphEdges: skinTokensEdges(
            glb: glb,
            mesh: mesh,
            sampled: sampled,
            meshEncoding: meshEncoding,
            condition: condition,
            generation: generation,
            queries: queries,
            transferPlan: transferPlan,
            skinField: skinField,
            vertexSkin: vertexSkin
        ),
        flows: [
            .init(
                id: "transform",
                phases: [
                    .init(
                        role: .setup,
                        calls: nodes.map { .node($0.id, entrypoint: $0.id) }
                    ),
                ],
                emit: [.node("glb.encode", "output")],
                stop: []
            ),
        ],
        capabilities: ["run.transform"],
        backendConstraints: [annot("device", "metal")],
        tensors: skinTokensTensorMaps(),
        compile: skinTokensCompileRequirements(),
        artifacts: [
            .init(id: "language-trunk", role: "compiled-trunk"),
        ]
    )
}

private func nativeNode(
    _ id: String,
    inputs: [IR.Port],
    outputs: [IR.Port],
    annotations: [IR.Constraint] = []
) -> IR.GraphNode {
    .init(
        id: id,
        implementation: .native,
        inputs: inputs,
        outputs: outputs,
        annotations: annotations
    )
}

private func skinTokensBlocks() -> [IR.Block] {
    [
        .init(
            id: "skin-vae",
            operatorName: .adapter,
            shape: .init(
                derivation: .init(source: "checkpoint", authority: "cam"),
                requirements: [
                    .init("sampled-point-count", "54000"),
                    .init("condition-token-count", "384"),
                    .init("tokens-per-joint", "4"),
                    .init("width", "768"),
                    .init("fsq-levels", "8,8,8,8,8"),
                ]
            )
        ),
        .init(
            id: "mesh-encoder",
            operatorName: .adapter,
            shape: .init(
                derivation: .init(source: "checkpoint", authority: "cam"),
                requirements: [
                    .init("sampled-point-count", "54000"),
                    .init("token-count", "512"),
                    .init("width", "512"),
                ]
            )
        ),
        .init(
            id: "language",
            operatorName: .transformer,
            shape: .init(
                derivation: .init(source: "checkpoint", authority: "cam"),
                transformer: .init(
                    hiddenSize: 896,
                    layers: .init(roles: [.attention], repeatCount: 28),
                    attention: .init(
                        qHeads: 16,
                        kvHeads: 8,
                        headDim: 128,
                        rope: .init(kind: .neox, theta: 1_000_000),
                        qkNorm: .rms
                    ),
                    ffn: .init(dim: 3_072, activation: .swiglu),
                    norm: .init(kind: .rms, eps: "1e-6"),
                    vocab: .init(size: 33_036, tiedHead: true)
                ),
                requirements: [
                    .init("static-seq-capacity", "3192"),
                    .init("sidecar", "language-trunk"),
                    .init("compiled-model-name", "qwen3-dense-trunk"),
                    .init("activation-dtype", "bf16"),
                    .init("weight-dtype", "bf16"),
                ]
            )
        ),
    ]
}

private func skinTokensTensorMaps() -> [IR.TensorMap] {
    SkinTokensTensorInventory.expectedTensors.enumerated().map { ordinal, tensor in
        let owner: String
        switch tensor.component {
        case .vae: owner = "skin-vae"
        case .meshEncoder: owner = "mesh-encoder"
        case .qwen, .outputProjection: owner = "language"
        }
        let alias = [
            "transformer.model.embed_tokens.weight",
            "transformer.lm_head.weight",
        ].contains(tensor.name) ? "language-token-embedding" : nil
        return .init(
            source: "checkpoint",
            selector: .init(tensor.name, source: "checkpoint"),
            target: .init(
                block: owner,
                selector: skinTokensTargetName(tensor.name)
            ),
            owner: owner,
            shape: tensor.shape,
            sourceDType: "BF16",
            disposition: tensor.disposition == .carried ? .carried : .trainingOnly,
            storageAlias: alias,
            layoutOrdinal: ordinal
        )
    }
}

private func skinTokensTargetName(_ source: String) -> String {
    guard source.hasPrefix("transformer.model.layers.") else {
        switch source {
        case "transformer.model.norm.weight": return "norm_weight"
        case "transformer.model.embed_tokens.weight": return "token_embedding_weight"
        case "transformer.lm_head.weight": return "lm_head_weight"
        default: return source
        }
    }
    let fields = source.split(separator: ".")
    guard fields.count >= 6, let layer = Int(fields[3]) else { return source }
    return "layers_\(layer)_" + fields.dropFirst(4).joined(separator: "_")
}

private func skinTokensEdges(
    glb: IR.TypeRef,
    mesh: IR.TypeRef,
    sampled: IR.TypeRef,
    meshEncoding: IR.TypeRef,
    condition: IR.TypeRef,
    generation: IR.TypeRef,
    queries: IR.TypeRef,
    transferPlan: IR.TypeRef,
    skinField: IR.TypeRef,
    vertexSkin: IR.TypeRef
) -> [IR.GraphEdge] {
    [
        .init(from: .moduleInput("input"), to: .node("glb.decode", "input"), type: glb),
        .init(from: .node("glb.decode", "mesh"), to: .node("surface.sample", "mesh"), type: mesh),
        .init(from: .node("glb.decode", "mesh"), to: .node("skin.neighbors", "mesh"), type: mesh),
        .init(from: .node("glb.decode", "mesh"), to: .node("glb.encode", "mesh"), type: mesh),
        .init(from: .node("surface.sample", "sampled"), to: .node("mesh.encode", "sampled"), type: sampled),
        .init(from: .node("surface.sample", "sampled"), to: .node("condition.encode", "sampled"), type: sampled),
        .init(from: .node("surface.sample", "sampled"), to: .node("skin.neighbors", "sampled"), type: sampled),
        .init(from: .node("surface.sample", "sampled"), to: .node("glb.encode", "sampled"), type: sampled),
        .init(from: .node("mesh.encode", "encoding"), to: .node("sequence.generate", "encoding"), type: meshEncoding),
        .init(from: .node("sequence.generate", "generation"), to: .node("skin.decode", "generation"), type: generation),
        .init(from: .node("sequence.generate", "generation"), to: .node("glb.encode", "generation"), type: generation),
        .init(from: .node("condition.encode", "condition"), to: .node("skin.decode", "condition"), type: condition),
        .init(from: .node("skin.neighbors", "queries"), to: .node("skin.decode", "queries"), type: queries),
        .init(from: .node("skin.neighbors", "plan"), to: .node("skin.transfer", "plan"), type: transferPlan),
        .init(from: .node("skin.decode", "skin"), to: .node("skin.transfer", "skin"), type: skinField),
        .init(from: .node("skin.transfer", "vertex-skin"), to: .node("glb.encode", "vertex-skin"), type: vertexSkin),
        .init(from: .node("glb.encode", "output"), to: .moduleOutput("output"), type: glb),
    ]
}

private func skinTokensCompileRequirements() -> [IR.Constraint] {
    let shaders = [
        "activations_f32.metal",
        "causal_gqa_attn_simd_f32.metal",
        "decode_gqa_attn_f32.metal",
        "dense_trunk_bf16_precise.metal",
        "gemm_bf16w_f32.metal",
        "gemv_add_bf16w_f32.metal",
        "gemv_gateup_swiglu_bf16w_f32.metal",
        "gemv_qkv_bf16w_f32.metal",
        "gather_row_bf16w_f32.metal",
        "head_norm_rope_f32.metal",
        "rms_norm_codec_f32.metal",
        "rms_norm_head_f32.metal",
        "rope_apply_f32.metal",
        "scale_residual_f32.metal",
        "skin_transfer_precise.metal",
        "neural_primitives_f32.metal",
    ]
    let pipelines = [
        "noncausal_attention_f32", "noncausal_attention_q8_f32",
        "noncausal_attention_q16_f32",
        "noncausal_attention_update_f32", "layer_norm_rows_f32",
        "fourier_position_embedding_f32", "pmpe_bf16_semantics_f32",
        "append_strided_features_f32", "fsq_base8x5_decode_f32", "add_rows_f32",
        "sigmoid_f32", "dense_bf16w_f32", "dense_bf16w_f32_rows4",
        "dense_bf16w_f32_rows8", "dense_bf16w_f32_rows8_epilogue",
        "dense_bf16w_f32_rows8_cols2_epilogue", "gelu_f32",
        "layer_norm_rows_bf16w_f32", "extract_interleaved_head_part_f32",
        "rms_norm_rows_bf16w_f32", "repack_concatenated_head_parts_f32",
        "rms_norm_codec_bf16w_f32", "rms_norm_head_bf16w_f32",
        "head_norm_rope_bf16w_f32", "gather_row_bf16w_f32", "gemv_qkv_bf16w_f32",
        "decode_gqa_attn_f32", "gemv_add_bf16w_f32", "gemv_gateup_swiglu_bf16w_f32",
        "gemm_bf16w_f32", "rope_apply_f32", "causal_gqa_attn_cached_f32",
        "scale_residual_tc_f32", "swiglu_f32", "skin_transfer_top4_f32",
        "rms_norm_codec_bf16",
        "rms_norm_head_bf16", "head_norm_rope_bf16", "gemv_qkv_bf16",
        "gemv_add_bf16", "gemv_gateup_swiglu_bf16", "decode_gqa_attn_bf16",
        "gemm_bf16", "rope_apply_bf16", "causal_gqa_attn_cached_bf16",
        "scale_residual_tc_bf16", "swiglu_bf16", "gather_row_bf16", "dense_bf16",
    ]
    return indexedRequirements("shader", shaders)
        + indexedRequirements("pipeline", pipelines)
}

private func indexedRequirements(_ key: String, _ values: [String]) -> [IR.Constraint] {
    values.enumerated().map { index, value in
        let digits = String(index)
        let suffix = String(repeating: "0", count: max(0, 3 - digits.count)) + digits
        return annot("\(key).\(suffix)", value)
    }
}
