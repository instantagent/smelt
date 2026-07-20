import Testing

@testable import SmeltCompiler
import SmeltSchema

@Suite("Smelt Qwen dense-trunk sidecar")
struct SmeltQwenDenseTrunkSidecarTests {
    private func manifest(
        mutate: (inout [SmeltRigPackageManifest.Tensor]) -> Void = { _ in }
    ) -> SmeltRigPackageManifest {
        var tensors: [SmeltRigPackageManifest.Tensor] = []
        var cursor: UInt64 = 0
        func add(_ name: String, _ shape: [Int], offset: UInt64? = nil) {
            let byteCount = UInt64(shape.reduce(1, *) * 2)
            let tensorOffset = offset ?? cursor
            tensors.append(
                .init(
                    name: name,
                    dtype: "BF16",
                    shape: shape,
                    offset: tensorOffset,
                    byteCount: byteCount,
                    allocationByteCount: byteCount,
                    storageID: "storage:\(tensorOffset)",
                    component: "qwen"
                )
            )
            if offset == nil { cursor += byteCount }
        }

        let embeddingOffset = cursor
        add("transformer.model.embed_tokens.weight", [33_036, 896])
        for layer in 0..<28 {
            let prefix = "transformer.model.layers.\(layer)"
            add("\(prefix).self_attn.q_proj.weight", [2_048, 896])
            add("\(prefix).self_attn.k_proj.weight", [1_024, 896])
            add("\(prefix).self_attn.v_proj.weight", [1_024, 896])
            add("\(prefix).self_attn.o_proj.weight", [896, 2_048])
            add("\(prefix).mlp.gate_proj.weight", [3_072, 896])
            add("\(prefix).mlp.up_proj.weight", [3_072, 896])
            add("\(prefix).mlp.down_proj.weight", [896, 3_072])
            add("\(prefix).input_layernorm.weight", [896])
            add("\(prefix).post_attention_layernorm.weight", [896])
            add("\(prefix).self_attn.q_norm.weight", [128])
            add("\(prefix).self_attn.k_norm.weight", [128])
        }
        add("transformer.model.norm.weight", [896])
        add("transformer.lm_head.weight", [33_036, 896], offset: embeddingOffset)
        mutate(&tensors)
        return SmeltRigPackageManifest(
            source: .init(
                repository: "test",
                commit: "test",
                huggingFaceRevision: "test",
                checkpointSHA256: String(repeating: "0", count: 64)
            ),
            pageSize: 4_096,
            totalBytes: max(cursor, 1),
            pipelines: ["test"],
            tensors: tensors,
            omittedTrainingTensors: [],
            checksums: .init(
                weightsSHA256: String(repeating: "0", count: 64),
                metallibSHA256: String(repeating: "0", count: 64)
            )
        )
    }

    private func ir() throws -> SmeltModelIR {
        let config = SmeltRigPackageManifest.Configuration.pinned
        let ir = SmeltModelIR.denseTrunk(
            modelName: "tokenrig-qwen3-trunk-test",
            hidden: config.languageHiddenSize,
            numLayers: config.languageLayerCount,
            vocab: config.languageVocabularySize,
            heads: config.languageQueryHeads,
            kvHeads: config.languageKeyValueHeads,
            headDim: config.languageHeadDim,
            inter: config.languageIntermediateSize,
            maxPrefillBatch: 1_024,
            staticSeqCapacity: config.languageMaximumPositions,
            activationDtype: .bf16
        )
        try validateSmeltIR(ir)
        return ir
    }

    @Test("All trunk tensors map directly as BF16")
    func mapsCanonicalLayout() throws {
        let manifest = manifest()
        let layout = try SmeltQwenDenseTrunkSidecar.weightLayout(
            manifest: manifest,
            ir: ir()
        )
        #expect(layout.count == 28 * 11 + 1)
        #expect(layout.allSatisfy { $0.dtype == .bf16 })
        #expect(layout.allSatisfy { $0.sizeBytes == UInt64($0.shape.reduce(1, *) * 2) })
        let source = Dictionary(uniqueKeysWithValues: manifest.tensors.map { ($0.name, $0) })
        let query = try #require(
            layout.first { $0.name == "layers_17_self_attn_q_proj_weight" }
        )
        #expect(query.offset == source["transformer.model.layers.17.self_attn.q_proj.weight"]?.offset)
        let finalNorm = try #require(layout.first { $0.name == "norm_weight" })
        #expect(finalNorm.offset == source["transformer.model.norm.weight"]?.offset)
    }

    @Test("BF16 norm entries select only BF16 norm kernels")
    func emitsBF16NormRoutes() throws {
        let ir = try ir()
        let layout = try SmeltQwenDenseTrunkSidecar.weightLayout(
            manifest: manifest(),
            ir: ir
        )
        let compilation = try SmeltCompiler.planCompilation(
            ir: ir,
            weightLayout: layout
        )
        let decode = try TopLevelEmitter.generate(
            ir: ir,
            compilationPlan: compilation
        )
        let prefill = try PrefillEmitter.generate(
            ir: ir,
            compilationPlan: compilation
        )
        let decodePipelines = Set(decode.dispatchRecords.map(\.pipeline))
        let prefillPipelines = Set(prefill.dispatchRecords.map(\.pipeline))
        #expect(ir.config.activationDtype == .bf16)
        #expect(decodePipelines.contains(UInt16(SmeltPipeline.rmsNormCodecBF16.rawValue)))
        #expect(decodePipelines.contains(UInt16(SmeltPipeline.headNormRopeBF16.rawValue)))
        #expect(decodePipelines.contains(UInt16(SmeltPipeline.gemvQKVBF16.rawValue)))
        #expect(decodePipelines.contains(UInt16(SmeltPipeline.decodeGQAAttnBF16.rawValue)))
        #expect(decodePipelines.contains(UInt16(SmeltPipeline.gemvAddBF16.rawValue)))
        #expect(decodePipelines.contains(UInt16(SmeltPipeline.gemvGateUpSwigluBF16.rawValue)))
        #expect(!decodePipelines.contains(UInt16(SmeltPipeline.rmsNormCodecF32.rawValue)))
        #expect(!decodePipelines.contains(UInt16(SmeltPipeline.headNormRopeF32.rawValue)))
        #expect(prefillPipelines.contains(UInt16(SmeltPipeline.rmsNormCodecBF16.rawValue)))
        #expect(prefillPipelines.contains(UInt16(SmeltPipeline.rmsNormHeadBF16.rawValue)))
        #expect(prefillPipelines.contains(UInt16(SmeltPipeline.gemmBF16.rawValue)))
        #expect(prefillPipelines.contains(UInt16(SmeltPipeline.ropeApplyBF16.rawValue)))
        #expect(prefillPipelines.contains(UInt16(SmeltPipeline.causalGQAAttnCachedBF16.rawValue)))
        #expect(prefillPipelines.contains(UInt16(SmeltPipeline.scaleResidualTCBF16.rawValue)))
        #expect(prefillPipelines.contains(UInt16(SmeltPipeline.swigluBF16.rawValue)))
        #expect(!prefillPipelines.contains(UInt16(SmeltPipeline.rmsNormCodecF32.rawValue)))
        #expect(!prefillPipelines.contains(UInt16(SmeltPipeline.rmsNormHeadF32.rawValue)))
        #expect(ir.config.staticSeqCapacity == 3_192)
    }

    @Test("Deep tensor corruption fails before emission")
    func rejectsDeepTensorCorruption() throws {
        let missingName = "transformer.model.layers.27.self_attn.k_norm.weight"
        let missing = manifest { tensors in
            tensors.removeAll { $0.name == missingName }
        }
        #expect(throws: SmeltQwenDenseTrunkSidecar.SidecarError.missingTensor(missingName)) {
            _ = try SmeltQwenDenseTrunkSidecar.weightLayout(manifest: missing, ir: ir())
        }

        let malformed = manifest { tensors in
            guard let index = tensors.firstIndex(where: { $0.name == missingName }) else { return }
            let original = tensors[index]
            tensors[index] = .init(
                name: original.name,
                dtype: original.dtype,
                shape: [127],
                offset: original.offset,
                byteCount: original.byteCount,
                allocationByteCount: original.allocationByteCount,
                storageID: original.storageID,
                component: original.component
            )
        }
        #expect(throws: SmeltQwenDenseTrunkSidecar.SidecarError.self) {
            _ = try SmeltQwenDenseTrunkSidecar.weightLayout(manifest: malformed, ir: ir())
        }
    }
}
