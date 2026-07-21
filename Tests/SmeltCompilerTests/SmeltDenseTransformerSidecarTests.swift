import SmeltModels
import SmeltSchema
import Testing

@testable import SmeltCompiler

@Suite("Smelt dense-transformer sidecar")
struct SmeltDenseTransformerSidecarTests {
    private func manifest(
        dtype: SmeltDType = .bf16,
        mutate: (inout [SmeltComponentPackageManifest.Tensor]) -> Void = { _ in }
    ) throws -> SmeltComponentPackageManifest {
        var tensors: [SmeltComponentPackageManifest.Tensor] = []
        var cursor: UInt64 = 0
        func add(_ source: String, target: String, shape: [Int], offset: UInt64? = nil) {
            let byteCount = UInt64(shape.reduce(1, *) * (dtype.bytesPerElement ?? 0))
            let tensorOffset = offset ?? cursor
            tensors.append(
                .init(
                    name: source,
                    target: target,
                    dtype: dtype.rawValue.uppercased(),
                    shape: shape,
                    offset: tensorOffset,
                    byteCount: byteCount,
                    allocationByteCount: byteCount,
                    storageID: "storage:\(tensorOffset)",
                    owner: "language"
                )
            )
            if offset == nil { cursor += byteCount }
        }

        let embeddingOffset = cursor
        add(
            "transformer.model.embed_tokens.weight",
            target: "token_embedding_weight",
            shape: [33_036, 896]
        )
        for layer in 0..<28 {
            let source = "transformer.model.layers.\(layer)"
            let target = "layers_\(layer)"
            let entries: [(String, String, [Int])] = [
                ("self_attn.q_proj.weight", "self_attn_q_proj_weight", [2_048, 896]),
                ("self_attn.k_proj.weight", "self_attn_k_proj_weight", [1_024, 896]),
                ("self_attn.v_proj.weight", "self_attn_v_proj_weight", [1_024, 896]),
                ("self_attn.o_proj.weight", "self_attn_o_proj_weight", [896, 2_048]),
                ("mlp.gate_proj.weight", "mlp_gate_proj_weight", [3_072, 896]),
                ("mlp.up_proj.weight", "mlp_up_proj_weight", [3_072, 896]),
                ("mlp.down_proj.weight", "mlp_down_proj_weight", [896, 3_072]),
                ("input_layernorm.weight", "input_layernorm_weight", [896]),
                ("post_attention_layernorm.weight", "post_attention_layernorm_weight", [896]),
                ("self_attn.q_norm.weight", "self_attn_q_norm_weight", [128]),
                ("self_attn.k_norm.weight", "self_attn_k_norm_weight", [128]),
            ]
            for entry in entries {
                add(
                    "\(source).\(entry.0)",
                    target: "\(target)_\(entry.1)",
                    shape: entry.2
                )
            }
        }
        add(
            "transformer.model.norm.weight",
            target: "norm_weight",
            shape: [896]
        )
        add(
            "transformer.lm_head.weight",
            target: "lm_head_weight",
            shape: [33_036, 896],
            offset: embeddingOffset
        )
        mutate(&tensors)
        let module = try #require(SmeltModels.definition(id: "skintokens_articulation"))
        return SmeltComponentPackageManifest(
            moduleID: module.module.id,
            camSemanticSHA256: String(repeating: "0", count: 64),
            run: try #require(module.run),
            sources: [
                .init(
                    id: "checkpoint",
                    kind: "pytorch-checkpoint",
                    locator: "test",
                    sha256: String(repeating: "0", count: 64)
                ),
            ],
            pageSize: 4_096,
            totalBytes: max(cursor, 1),
            pipelines: ["test"],
            tensors: tensors,
            omittedTensors: [],
            checksums: .init(
                weightsSHA256: String(repeating: "0", count: 64),
                metallibSHA256: String(repeating: "0", count: 64)
            )
        )
    }

    private func ir(activationDtype: SmeltDType = .bf16) throws -> SmeltModelIR {
        let ir = SmeltModelIR.denseTrunk(
            modelName: "dense-transformer-test",
            hidden: 896,
            numLayers: 28,
            vocab: 33_036,
            heads: 16,
            kvHeads: 8,
            headDim: 128,
            inter: 3_072,
            maxPrefillBatch: 1_024,
            staticSeqCapacity: 3_192,
            activationDtype: activationDtype
        )
        return ir
    }

    @Test("All trunk tensors map through canonical CAM targets as BF16")
    func mapsCanonicalLayout() throws {
        let manifest = try manifest()
        let layout = try SmeltDenseTransformerSidecar.weightLayout(
            manifest: manifest,
            ir: ir(),
            weightDtype: .bf16
        )
        #expect(layout.count == 28 * 11 + 1)
        #expect(layout.allSatisfy { $0.dtype == .bf16 })
        #expect(layout.allSatisfy { $0.sizeBytes == UInt64($0.shape.reduce(1, *) * 2) })
        let source = Dictionary(uniqueKeysWithValues: manifest.tensors.map { ($0.target, $0) })
        let query = try #require(
            layout.first { $0.name == "layers_17_self_attn_q_proj_weight" }
        )
        #expect(query.offset == source["layers_17_self_attn_q_proj_weight"]?.offset)
        let finalNorm = try #require(layout.first { $0.name == "norm_weight" })
        #expect(finalNorm.offset == source["norm_weight"]?.offset)
    }

    @Test("BF16 entries select only BF16 transformer kernels")
    func emitsBF16NormRoutes() throws {
        let ir = try ir()
        let layout = try SmeltDenseTransformerSidecar.weightLayout(
            manifest: manifest(),
            ir: ir,
            weightDtype: .bf16
        )
        let compilation = try SmeltCompiler.planCompilation(ir: ir, weightLayout: layout)
        let decode = try TopLevelEmitter.generate(ir: ir, compilationPlan: compilation)
        let prefill = try PrefillEmitter.generate(ir: ir, compilationPlan: compilation)
        let decodePipelines = Set(decode.dispatchRecords.map(\.pipeline))
        let prefillPipelines = Set(prefill.dispatchRecords.map(\.pipeline))
        #expect(decodePipelines.contains(UInt16(SmeltPipeline.rmsNormCodecBF16.rawValue)))
        #expect(decodePipelines.contains(UInt16(SmeltPipeline.decodeGQAAttnBF16.rawValue)))
        #expect(!decodePipelines.contains(UInt16(SmeltPipeline.rmsNormCodecF32.rawValue)))
        #expect(prefillPipelines.contains(UInt16(SmeltPipeline.gemmBF16.rawValue)))
        #expect(prefillPipelines.contains(UInt16(SmeltPipeline.causalGQAAttnCachedBF16.rawValue)))
        #expect(!prefillPipelines.contains(UInt16(SmeltPipeline.rmsNormCodecF32.rawValue)))
        #expect(ir.config.staticSeqCapacity == 3_192)
    }

    @Test("Deep canonical-target corruption fails before emission")
    func rejectsDeepTensorCorruption() throws {
        let missingTarget = "layers_27_self_attn_k_norm_weight"
        let missing = try manifest { tensors in
            tensors.removeAll { $0.target == missingTarget }
        }
        #expect(
            throws: SmeltDenseTransformerSidecar.SidecarError.missingTensor(missingTarget)
        ) {
            _ = try SmeltDenseTransformerSidecar.weightLayout(
                manifest: missing,
                ir: ir(),
                weightDtype: .bf16
            )
        }

        let malformed = try manifest { tensors in
            guard let index = tensors.firstIndex(where: { $0.target == missingTarget }) else {
                return
            }
            let original = tensors[index]
            tensors[index] = .init(
                name: original.name,
                target: original.target,
                dtype: original.dtype,
                shape: [127],
                offset: original.offset,
                byteCount: original.byteCount,
                allocationByteCount: original.allocationByteCount,
                storageID: original.storageID,
                owner: original.owner
            )
        }
        #expect(throws: SmeltDenseTransformerSidecar.SidecarError.self) {
            _ = try SmeltDenseTransformerSidecar.weightLayout(
                manifest: malformed,
                ir: ir(),
                weightDtype: .bf16
            )
        }
    }

    @Test("FP16, BF16, and FP32 use one scalar-policy layout path")
    func scalarStorageIsParameterized() throws {
        for dtype in [SmeltDType.fp16, .bf16, .fp32] {
            let ir = try ir(activationDtype: dtype)
            let layout = try SmeltDenseTransformerSidecar.weightLayout(
                manifest: manifest(dtype: dtype),
                ir: ir,
                weightDtype: dtype
            )
            #expect(ir.config.activationDtype == dtype)
            #expect(layout.allSatisfy { $0.dtype == dtype })
            #expect(
                layout.allSatisfy {
                    $0.sizeBytes
                        == UInt64($0.shape.reduce(1, *) * (dtype.bytesPerElement ?? 0))
                }
            )
        }
        #expect(throws: Error.self) {
            try validateSmeltIR(ir(activationDtype: .fp16))
        }
    }
}
