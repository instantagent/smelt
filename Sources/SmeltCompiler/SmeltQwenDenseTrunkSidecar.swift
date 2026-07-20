import SmeltSchema
import Foundation

/// Compiles the rig model's shape-invariant Qwen3 block into a standard Smelt
/// embeddings-in/hidden-out sidecar while sharing the parent package's exact
/// checkpoint bytes and Metal library.
public enum SmeltQwenDenseTrunkSidecar {
    public enum SidecarError: Error, Equatable, CustomStringConvertible {
        case missingTensor(String)
        case shapeMismatch(name: String, expected: [Int], got: [Int])
        case dtypeMismatch(name: String, expected: String, got: String)
        case logicalByteCount(name: String, expected: UInt64, got: UInt64)
        case missingSharedFile(String)
        case invalidSharedLink(String)

        public var description: String {
            switch self {
            case .missingTensor(let name):
                return "rig model Qwen sidecar is missing tensor '\(name)'"
            case let .shapeMismatch(name, expected, got):
                return "rig model Qwen tensor '\(name)' shape \(got) != \(expected)"
            case let .dtypeMismatch(name, expected, got):
                return "rig model Qwen tensor '\(name)' dtype \(got) != \(expected)"
            case let .logicalByteCount(name, expected, got):
                return "rig model Qwen tensor '\(name)' has \(got) bytes, expected \(expected)"
            case .missingSharedFile(let path):
                return "rig model Qwen sidecar shared file is missing: \(path)"
            case .invalidSharedLink(let why):
                return "rig model Qwen sidecar has an invalid shared link: \(why)"
            }
        }
    }

    /// Fully preflighted sidecar state. No filesystem mutation occurs until
    /// both compiler tables and the complete weight mapping have been built.
    public struct PreparedTrunk {
        let ir: SmeltModelIR
        let plan: SmeltBufferPlan
        let layout: [SmeltWeightEntry]
        let decode: TopLevelEmitter.GenerateResult
        let prefill: PrefillEmitter.GenerateResult
        let totalBytes: UInt64
        let directoryName: String
    }

    /// Builds the compiler IR and both dispatch tables against direct offsets in
    /// the parent rig model weight blob.
    public static func prepare(
        manifest: SmeltRigPackageManifest,
        maxPrefillBatch: Int = 2_048
    ) throws -> PreparedTrunk {
        try manifest.validate()
        let config = manifest.configuration
        let ir = SmeltModelIR.denseTrunk(
            modelName: "qwen3-dense-trunk",
            hidden: config.languageHiddenSize,
            numLayers: config.languageLayerCount,
            vocab: config.languageVocabularySize,
            heads: config.languageQueryHeads,
            kvHeads: config.languageKeyValueHeads,
            headDim: config.languageHeadDim,
            inter: config.languageIntermediateSize,
            maxPrefillBatch: maxPrefillBatch,
            staticSeqCapacity: config.languageMaximumPositions,
            activationDtype: .bf16
        )
        try validateSmeltIR(ir)
        let layout = try weightLayout(manifest: manifest, ir: ir)
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
        return PreparedTrunk(
            ir: ir,
            plan: compilation.bufferPlan,
            layout: layout,
            decode: decode,
            prefill: prefill,
            totalBytes: manifest.totalBytes,
            directoryName: manifest.files.languageTrunk
        )
    }

    /// Maps all 309 trunk-consumed tensors to the canonical names expected by
    /// the shared dense-trunk emitters. Every entry remains BF16.
    static func weightLayout(
        manifest: SmeltRigPackageManifest,
        ir: SmeltModelIR
    ) throws -> [SmeltWeightEntry] {
        let tensors = Dictionary(
            uniqueKeysWithValues: manifest.tensors.map { ($0.name, $0) }
        )
        guard let attention = ir.config.attention else {
            throw SidecarError.missingTensor("Qwen attention configuration")
        }
        let hidden = ir.config.hiddenSize
        let queryWidth = attention.qHeads * attention.headDim
        let keyValueWidth = attention.kvHeads * attention.headDim
        let intermediate = ir.config.ffn.dim
        let headDim = attention.headDim

        let perLayer: [(canonical: String, source: String, shape: [Int])] = [
            ("_self_attn_q_proj_weight", ".self_attn.q_proj.weight", [queryWidth, hidden]),
            ("_self_attn_k_proj_weight", ".self_attn.k_proj.weight", [keyValueWidth, hidden]),
            ("_self_attn_v_proj_weight", ".self_attn.v_proj.weight", [keyValueWidth, hidden]),
            ("_self_attn_o_proj_weight", ".self_attn.o_proj.weight", [hidden, queryWidth]),
            ("_mlp_gate_proj_weight", ".mlp.gate_proj.weight", [intermediate, hidden]),
            ("_mlp_up_proj_weight", ".mlp.up_proj.weight", [intermediate, hidden]),
            ("_mlp_down_proj_weight", ".mlp.down_proj.weight", [hidden, intermediate]),
            ("_input_layernorm_weight", ".input_layernorm.weight", [hidden]),
            ("_post_attention_layernorm_weight", ".post_attention_layernorm.weight", [hidden]),
            ("_self_attn_q_norm_weight", ".self_attn.q_norm.weight", [headDim]),
            ("_self_attn_k_norm_weight", ".self_attn.k_norm.weight", [headDim]),
        ]
        var specifications: [(canonical: String, source: String, shape: [Int])] = []
        specifications.reserveCapacity(ir.config.numLayers * perLayer.count + 1)
        for layer in 0..<ir.config.numLayers {
            let sourcePrefix = "transformer.model.layers.\(layer)"
            for entry in perLayer {
                specifications.append(
                    (
                        "layers_\(layer)\(entry.canonical)",
                        "\(sourcePrefix)\(entry.source)",
                        entry.shape
                    )
                )
            }
        }
        specifications.append(
            ("norm_weight", "transformer.model.norm.weight", [hidden])
        )

        return try specifications.map { specification in
            guard let tensor = tensors[specification.source] else {
                throw SidecarError.missingTensor(specification.source)
            }
            guard tensor.shape == specification.shape else {
                throw SidecarError.shapeMismatch(
                    name: specification.source,
                    expected: specification.shape,
                    got: tensor.shape
                )
            }
            guard tensor.dtype == "BF16" else {
                throw SidecarError.dtypeMismatch(
                    name: specification.source,
                    expected: "BF16",
                    got: tensor.dtype
                )
            }
            let expectedBytes = UInt64(specification.shape.reduce(1, *) * 2)
            guard tensor.byteCount == expectedBytes else {
                throw SidecarError.logicalByteCount(
                    name: specification.source,
                    expected: expectedBytes,
                    got: tensor.byteCount
                )
            }
            return SmeltWeightEntry(
                name: specification.canonical,
                offset: tensor.offset,
                sizeBytes: tensor.byteCount,
                shape: tensor.shape,
                dtype: .bf16
            )
        }
    }

    /// Atomically commits a preflighted trunk into the temporary parent package.
    public static func commit(
        _ prepared: PreparedTrunk,
        intoPackage packagePath: String
    ) throws {
        let fileManager = FileManager.default
        let trunkPath = "\(packagePath)/\(prepared.directoryName)"
        let temporaryPath = "\(packagePath)/.\(prepared.directoryName).tmp"
        if fileManager.fileExists(atPath: temporaryPath) {
            try fileManager.removeItem(atPath: temporaryPath)
        }
        try fileManager.createDirectory(
            atPath: temporaryPath,
            withIntermediateDirectories: true
        )
        var committed = false
        defer {
            if !committed {
                try? fileManager.removeItem(atPath: temporaryPath)
            }
        }

        try SmeltCompiler.writeDispatchTable(
            prepared.decode.dispatchRecords,
            to: "\(temporaryPath)/dispatches.bin"
        )
        try SmeltCompiler.writeDispatchTable(
            prepared.prefill.dispatchRecords,
            to: "\(temporaryPath)/prefill_dispatches.bin"
        )
        if !prepared.prefill.gptqCapturePoints.isEmpty {
            let data = try SmeltGPTQCapturePoints(
                prefill: prepared.prefill.gptqCapturePoints
            ).canonicalJSONData()
            try data.write(
                to: URL(fileURLWithPath: "\(temporaryPath)/gptq_capture_points.json")
            )
        }
        try prepared.decode.source.write(
            toFile: "\(temporaryPath)/SmeltGenerated.swift",
            atomically: true,
            encoding: .utf8
        )
        try linkShared(
            name: "weights.bin",
            sidecarPath: temporaryPath,
            packagePath: packagePath
        )
        try linkShared(
            name: "model.metallib",
            sidecarPath: temporaryPath,
            packagePath: packagePath
        )

        let handoff = SmeltHandoffResolver.resolve(
            families: ["key_cache", "value_cache"],
            ir: prepared.ir,
            plan: prepared.plan
        )
        let checksums = try SmeltCompiler.computeManifestChecksums(
            pkgPath: temporaryPath,
            weightsPath: "\(temporaryPath)/weights.bin",
            metallibPath: "\(temporaryPath)/model.metallib",
            generatedSwiftPath: "\(temporaryPath)/SmeltGenerated.swift",
            dispatchesPath: "\(temporaryPath)/dispatches.bin",
            prefillDispatchesPath: "\(temporaryPath)/prefill_dispatches.bin",
            prefillVerifyArgmaxDispatchesPath: nil
        )
        let manifest = SmeltManifest(
            kind: nil,
            headlessTrunkABI: true,
            blocks: nil,
            loop: nil,
            modelName: prepared.ir.modelName,
            config: SmeltCompiler.manifestConfigSnapshot(from: prepared.ir),
            context: nil,
            checksums: checksums,
            buildProvenance: nil,
            device: SmeltDeviceRequirements(
                metalFamily: .apple7,
                minMemoryBytes: UInt64(prepared.plan.totalActivationBytes)
            ),
            weights: SmeltWeightManifest(
                totalBytes: prepared.totalBytes,
                entries: prepared.layout
            ),
            buffers: prepared.plan.toBufferTable(),
            pipelines: SmeltKernelCatalog.pipelineNames,
            slotLayout: prepared.plan.toSlotLayout(),
            prefill: SmeltPrefillManifest(
                engine: "metal",
                modelPath: "prefill.mlmodelc",
                maxBatchSize: prepared.ir.prefill?.maxBatchSize ?? 1_024,
                handoff: handoff,
                inputContract: SmeltPrefillInputContract()
            ),
            inference: nil,
            optimizationReport: nil
        )
        try manifest.encodePrettyJSON().write(
            to: URL(fileURLWithPath: "\(temporaryPath)/manifest.json")
        )

        if fileManager.fileExists(atPath: trunkPath) {
            try fileManager.removeItem(atPath: trunkPath)
        }
        try fileManager.moveItem(atPath: temporaryPath, toPath: trunkPath)
        committed = true
    }

    private static func linkShared(
        name: String,
        sidecarPath: String,
        packagePath: String
    ) throws {
        let fileManager = FileManager.default
        let target = "\(packagePath)/\(name)"
        guard fileManager.fileExists(atPath: target) else {
            throw SidecarError.missingSharedFile(target)
        }
        let link = "\(sidecarPath)/\(name)"
        try fileManager.createSymbolicLink(
            atPath: link,
            withDestinationPath: "../\(name)"
        )
        let resolved = URL(fileURLWithPath: link).resolvingSymlinksInPath().path
        let expected = URL(fileURLWithPath: target).resolvingSymlinksInPath().path
        guard resolved == expected else {
            throw SidecarError.invalidSharedLink(
                "\(name) resolves to \(resolved), expected \(expected)"
            )
        }
    }
}
