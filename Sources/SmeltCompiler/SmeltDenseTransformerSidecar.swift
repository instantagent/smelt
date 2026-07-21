import SmeltSchema
import Foundation

/// Compiles a CAM-declared dense transformer block into a standard Smelt
/// embeddings-in/hidden-out sidecar while sharing its parent package artifacts.
public enum SmeltDenseTransformerSidecar {
    public enum SidecarError: Error, Equatable, CustomStringConvertible {
        case missingTensor(String)
        case shapeMismatch(name: String, expected: [Int], got: [Int])
        case dtypeMismatch(name: String, expected: String, got: String)
        case logicalByteCount(name: String, expected: UInt64, got: UInt64)
        case missingSharedFile(String)
        case invalidSharedLink(String)
        case missingScalarPolicy(String)
        case unsupportedScalarPolicy(String, String)

        public var description: String {
            switch self {
            case .missingTensor(let name):
                return "dense transformer sidecar is missing tensor '\(name)'"
            case let .shapeMismatch(name, expected, got):
                return "dense transformer tensor '\(name)' shape \(got) != \(expected)"
            case let .dtypeMismatch(name, expected, got):
                return "dense transformer tensor '\(name)' dtype \(got) != \(expected)"
            case let .logicalByteCount(name, expected, got):
                return "dense transformer tensor '\(name)' has \(got) bytes, expected \(expected)"
            case .missingSharedFile(let path):
                return "dense transformer sidecar shared file is missing: \(path)"
            case .invalidSharedLink(let why):
                return "dense transformer sidecar has an invalid shared link: \(why)"
            case .missingScalarPolicy(let key):
                return "dense transformer sidecar is missing scalar policy '\(key)'"
            case let .unsupportedScalarPolicy(key, value):
                return "dense transformer sidecar does not support \(key)='\(value)'"
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
    /// the parent component weight blob.
    public static func prepare(
        manifest: SmeltComponentPackageManifest,
        block: SmeltCAMIR.Block,
        directoryName: String,
        maxPrefillBatch: Int = 2_048
    ) throws -> PreparedTrunk {
        try manifest.validate()
        guard let config = block.shape.transformer,
              let hidden = config.hiddenSize,
              let layers = config.layers,
              let repeatCount = layers.repeatCount,
              let vocab = config.vocab,
              let attention = config.attention,
              let ffn = config.ffn,
              let staticCapacityText = block.shape.requirements.first(
                  where: { $0.key == "static-seq-capacity" }
              )?.value,
              let staticCapacity = Int(staticCapacityText)
        else {
            throw SidecarError.missingTensor("CAM dense transformer shape")
        }
        let layerCount = layers.roles.count * repeatCount
        let activationDtype = try scalarDtype(
            requirement: "activation-dtype",
            block: block
        )
        let weightDtype = try scalarDtype(
            requirement: "weight-dtype",
            block: block
        )
        let modelName = block.shape.requirements.first(
            where: { $0.key == "compiled-model-name" }
        )?.value ?? block.id
        let ir = SmeltModelIR.denseTrunk(
            modelName: modelName,
            hidden: hidden,
            numLayers: layerCount,
            vocab: vocab.size,
            heads: attention.qHeads,
            kvHeads: attention.kvHeads,
            headDim: attention.headDim,
            inter: ffn.dim,
            maxPrefillBatch: maxPrefillBatch,
            staticSeqCapacity: staticCapacity,
            activationDtype: activationDtype
        )
        try validateSmeltIR(ir)
        let layout = try weightLayout(
            manifest: manifest,
            ir: ir,
            weightDtype: weightDtype
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
        return PreparedTrunk(
            ir: ir,
            plan: compilation.bufferPlan,
            layout: layout,
            decode: decode,
            prefill: prefill,
            totalBytes: manifest.totalBytes,
            directoryName: directoryName
        )
    }

    /// Resolves the canonical dense-trunk tensor names authored by CAM.
    static func weightLayout(
        manifest: SmeltComponentPackageManifest,
        ir: SmeltModelIR,
        weightDtype: SmeltDType
    ) throws -> [SmeltWeightEntry] {
        let tensors = Dictionary(
            uniqueKeysWithValues: manifest.tensors.map { ($0.target, $0) }
        )
        guard let attention = ir.config.attention else {
            throw SidecarError.missingTensor("dense transformer attention configuration")
        }
        let hidden = ir.config.hiddenSize
        let queryWidth = attention.qHeads * attention.headDim
        let keyValueWidth = attention.kvHeads * attention.headDim
        let intermediate = ir.config.ffn.dim
        let headDim = attention.headDim

        let perLayer: [(suffix: String, shape: [Int])] = [
            ("_self_attn_q_proj_weight", [queryWidth, hidden]),
            ("_self_attn_k_proj_weight", [keyValueWidth, hidden]),
            ("_self_attn_v_proj_weight", [keyValueWidth, hidden]),
            ("_self_attn_o_proj_weight", [hidden, queryWidth]),
            ("_mlp_gate_proj_weight", [intermediate, hidden]),
            ("_mlp_up_proj_weight", [intermediate, hidden]),
            ("_mlp_down_proj_weight", [hidden, intermediate]),
            ("_input_layernorm_weight", [hidden]),
            ("_post_attention_layernorm_weight", [hidden]),
            ("_self_attn_q_norm_weight", [headDim]),
            ("_self_attn_k_norm_weight", [headDim]),
        ]
        var specifications: [(canonical: String, shape: [Int])] = []
        specifications.reserveCapacity(ir.config.numLayers * perLayer.count + 1)
        for layer in 0..<ir.config.numLayers {
            for entry in perLayer {
                specifications.append(
                    (
                        "layers_\(layer)\(entry.suffix)",
                        entry.shape
                    )
                )
            }
        }
        specifications.append(("norm_weight", [hidden]))

        return try specifications.map { specification in
            guard let tensor = tensors[specification.canonical] else {
                throw SidecarError.missingTensor(specification.canonical)
            }
            guard tensor.shape == specification.shape else {
                throw SidecarError.shapeMismatch(
                    name: specification.canonical,
                    expected: specification.shape,
                    got: tensor.shape
                )
            }
            guard SmeltDType(rawValue: tensor.dtype.lowercased()) == weightDtype else {
                throw SidecarError.dtypeMismatch(
                    name: specification.canonical,
                    expected: weightDtype.rawValue.uppercased(),
                    got: tensor.dtype
                )
            }
            guard let bytesPerElement = weightDtype.bytesPerElement else {
                throw SidecarError.unsupportedScalarPolicy(
                    "weight-dtype",
                    weightDtype.rawValue
                )
            }
            let expectedBytes = UInt64(
                specification.shape.reduce(1, *) * bytesPerElement
            )
            guard tensor.byteCount == expectedBytes else {
                throw SidecarError.logicalByteCount(
                    name: specification.canonical,
                    expected: expectedBytes,
                    got: tensor.byteCount
                )
            }
            return SmeltWeightEntry(
                name: specification.canonical,
                offset: tensor.offset,
                sizeBytes: tensor.byteCount,
                shape: tensor.shape,
                dtype: weightDtype
            )
        }
    }

    private static func scalarDtype(
        requirement key: String,
        block: SmeltCAMIR.Block
    ) throws -> SmeltDType {
        guard let value = block.shape.requirements.first(where: { $0.key == key })?.value else {
            throw SidecarError.missingScalarPolicy(key)
        }
        guard let dtype = SmeltDType(rawValue: value.lowercased()),
              [.fp16, .bf16, .fp32].contains(dtype)
        else {
            throw SidecarError.unsupportedScalarPolicy(key, value)
        }
        return dtype
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
