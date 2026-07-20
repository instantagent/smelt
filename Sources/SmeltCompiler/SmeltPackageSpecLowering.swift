// SmeltPackageSpecLowering — shadow-lower existing compiler inputs into CAM.
//
// This file does not write packages and does not affect runtime dispatch. It is
// the comparison bridge: current family-specific builders can produce the pure
// CAM spec they ought to be equivalent to, then tests can pin the resolved
// graph, loop, tensor map, files, and runtime policy before any default flips.

import Foundation
import SmeltSchema

public enum SmeltPackageSpecLoweringError: Error, CustomStringConvertible, Equatable {
    case unsupportedModel(String)

    public var description: String {
        switch self {
        case .unsupportedModel(let why): return "package spec lowering: \(why)"
        }
    }
}

public enum SmeltPackageSpecLowering {
    private typealias RuntimeCommand = SmeltPackageSpec.RuntimeDescriptor.Command

    private static let allRuntimeCommands: [RuntimeCommand] = RuntimeCommand.allCases
    public static func textGeneration(
        from ir: SmeltModelIR,
        packageName: String? = nil,
        source: SmeltPackageSpec.Source? = nil
    ) throws -> SmeltPackageSpec {
        try validateSmeltIR(ir)
        if let architecture = ir.runtime.architecture {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text-generation CAM lowering does not accept runtime.architecture '\(architecture)'"
            )
        }
        guard ir.runtime.architectureSource == .defaultValue else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text-generation CAM lowering does not accept runtime.architecture source '\(ir.runtime.architectureSource.rawValue)'"
            )
        }
        let architecture = textGenerationArchitecture()
        let source = source ?? SmeltPackageSpec.Source(
            id: "checkpoint",
            kind: .huggingFace,
            repo: ir.modelName
        )
        let weightLayout = SmeltWeightLayout.computeLayout(from: ir)
        let artifacts = textArtifacts(prefill: ir.prefill)
        let outputFiles = SmeltPackageSpec.PackageFileSet(
            files: ["manifest.json"] + artifacts.map(\.path) + ["tokenizer.json"]
        )
        guard let thinkingPolicy = ir.inference.thinkingPolicy else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text-generation CAM lowering requires inference.thinking_policy"
            )
        }
        guard ir.inference.maxTokensSource == .explicit else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text-generation CAM lowering requires explicit inference.max_tokens"
            )
        }
        guard ir.inference.eosTokensSource == .explicit else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text-generation CAM lowering requires explicit inference.eos_tokens"
            )
        }
        guard let chatTemplate = ir.inference.chatTemplate, !chatTemplate.isEmpty else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text-generation CAM lowering requires inference.chat_template"
            )
        }
        guard SmeltPromptTemplateName.isKnownPromptTemplate(chatTemplate) else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text-generation CAM lowering chat_template '\(chatTemplate)' is not supported"
            )
        }
        guard ir.decode.policySource == .explicit, let decode = ir.decode.policy else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text-generation CAM lowering requires explicit decode policy"
            )
        }
        guard decode.subSampler == nil else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text-generation CAM lowering does not support decode.sub_sampler"
            )
        }
        guard let maxSteps = decode.maxSteps else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text-generation CAM lowering requires decode.max_steps"
            )
        }
        guard maxSteps == ir.inference.maxTokens else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text-generation CAM lowering requires decode.max_steps to match inference.max_tokens"
            )
        }
        guard decode.durationSeconds == nil else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text-generation CAM lowering does not support decode.duration_seconds"
            )
        }

        let inference = SmeltInferenceManifest(
            maxTokens: ir.inference.maxTokens,
            eosTokens: ir.inference.eosTokens,
            thinkToken: ir.inference.thinkToken,
            thinkEndToken: ir.inference.thinkEndToken,
            thinkSkipSuffix: ir.inference.thinkSkipSuffix,
            chatTemplate: chatTemplate,
            thinkingPolicy: thinkingPolicy,
            toolTranscriptCodec: ir.inference.toolTranscriptCodec,
            promptStateRestoreMode: ir.inference.promptStateRestoreMode
        )

        return SmeltPackageSpec(
            packageName: packageName ?? defaultPackageName(for: ir.modelName),
            modelName: ir.modelName,
            sources: [source],
            blocks: .tokenFeedbackText,
            loop: .tokenFeedbackText,
            runtime: .forGraph(
                architecture: architecture,
                commands: allRuntimeCommands,
                graph: .tokenFeedbackText
            ),
            architectureConfig: architectureConfig(for: ir),
            tensors: weightLayout.map { tensorMap(from: $0, source: source.id) },
            quantization: quantizationPlan(for: ir.quantization),
            artifacts: artifacts,
            outputFiles: outputFiles,
            tokenizer: .init(format: "tokenizer-json", files: ["tokenizer.json"]),
            inference: inference,
            decode: decode,
            validation: validation(
                parityFixture: ir.modelName,
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                modelName: ir.modelName
            )
        )
    }

    public static func signature(for spec: SmeltPackageSpec) -> SmeltPackageAssemblySignature {
        SmeltPackageAssemblySignature(spec: spec)
    }

    public static func qwen3TTS(
        from specs: [Qwen3TTSPackageBuilder.WeightSpec],
        packageName: String? = nil,
        sourcePath: String = ".",
        modelName: String = SmeltQwen3TTSPackageProfiles.runnable.modelName,
        eosTokens: [Int32] = SmeltQwen3TTSPackageProfiles.runnable.eosTokens,
        tokenizerFiles: [String] = SmeltQwen3TTSPackageProfiles.runnable.tokenizerFiles,
        decode: Qwen3TTSManifest.Decode? = nil,
        tensorBlocks: [String: String],
        tensorSourceDTypes: [String: SmeltPackageSpec.TensorDType],
        pipelines: [String] = Qwen3TTSPackageBuilder.ttsPipelineNames,
        pageSize: Int = SmeltQwen3TTSPackageProfiles.runnable.pageSize
    ) throws -> SmeltPackageSpec {
        let profile = SmeltQwen3TTSPackageProfiles.runnable
        guard !specs.isEmpty else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "Qwen3-TTS CAM lowering needs at least one tensor"
            )
        }
        guard pageSize > 0, pageSize & (pageSize - 1) == 0 else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "Qwen3-TTS page size \(pageSize) must be a positive power of two"
            )
        }
        guard !tokenizerFiles.isEmpty else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "Qwen3-TTS CAM lowering needs tokenizer files"
            )
        }
        guard tokenizerFiles == profile.tokenizerFiles else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "Qwen3-TTS CAM lowering needs the exact package tokenizer file set: "
                    + profile.tokenizerFiles.joined(separator: ", ")
            )
        }
        guard !eosTokens.isEmpty else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "Qwen3-TTS CAM lowering needs codec EOS tokens"
            )
        }
        for spec in specs where spec.dtype == .u4 {
            try Qwen3TTSPackageBuilder.validateU4(spec)
        }
        do {
            try Qwen3TTSPackageBuilder.validateTrunkConfigConstants(sourcePath)
        } catch {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "Qwen3-TTS compiled trunk config preflight failed: \(error)"
            )
        }

        let orderedSpecs = specs.sorted { $0.name < $1.name }
        let layout = Qwen3TTSPackageBuilder.planLayout(
            orderedSpecs,
            pageSize: pageSize
        )
        guard layout.entries.contains(where: { $0.name.hasPrefix("talker.") }) else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "Qwen3-TTS CAM lowering only supports full runnable text-to-audio packages"
            )
        }
        guard Qwen3TTSPackageBuilder.shouldShipTrunks(layout.entries) else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "Qwen3-TTS CAM lowering needs both compiled trunk sidecars to be shippable"
            )
        }

        let textEmbedIsBF16 = layout.entries.first {
            $0.name == "talker.model.text_embedding.weight"
        }?.dtype == "bf16"
        let graph = profile.graph(textEmbeddingIsBF16: textEmbedIsBF16)
        try requireTensorBlocks(
            tensorBlocks,
            exactlyCovering: orderedSpecs.map(\.name),
            allowedBlocks: Set(graph.blocks.map(\.name))
        )
        try requireTensorSourceDTypes(
            tensorSourceDTypes,
            exactlyCovering: orderedSpecs.map(\.name)
        )
        let source = SmeltPackageSpec.Source(
            id: "qwen3-tts-source",
            kind: .localDirectory,
            path: sourcePath
        )
        let artifacts = qwen3TTSArtifacts(tokenizerFiles: tokenizerFiles)
        let sidecars = profile.sidecars
        let manifestForPreflight = Qwen3TTSManifest(
            version: 1,
            blocks: graph,
            loop: profile.loop,
            modelName: modelName,
            pageSize: pageSize,
            pipelines: pipelines,
            eosTokens: eosTokens,
            totalBytes: layout.totalBytes,
            weights: layout.entries,
            tokenizerFiles: tokenizerFiles,
            decode: decode
        )
        do {
            _ = try Qwen3TTSTrunkSidecar.prepare(manifest: manifestForPreflight, spec: .talker)
            _ = try Qwen3TTSTrunkSidecar.prepare(manifest: manifestForPreflight, spec: .mtp)
        } catch {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "Qwen3-TTS compiled trunk preflight failed: \(error)"
            )
        }

        return SmeltPackageSpec(
            packageName: packageName ?? (modelName == profile.modelName
                ? profile.packageName
                : defaultPackageName(for: modelName)),
            modelName: modelName,
            sources: [source],
            blocks: graph,
            loop: profile.loop,
            runtime: .forGraph(
                architecture: profile.runtimeArchitecture,
                commands: allRuntimeCommands,
                graph: graph
            ),
            architectureConfig: try qwen3TTSArchitectureConfig(
                runtimeArchitecture: profile.runtimeArchitecture,
                modelName: modelName,
                eosTokens: eosTokens,
                tokenizerFiles: tokenizerFiles,
                decode: decode,
                pipelines: pipelines,
                sidecarPaths: profile.sidecarPaths,
                pageSize: pageSize,
                totalBytes: layout.totalBytes,
                entries: layout.entries
            ),
            tensors: try orderedSpecs.map { spec in
                guard let block = tensorBlocks[spec.name] else {
                    throw SmeltPackageSpecLoweringError.unsupportedModel(
                        "Qwen3-TTS CAM tensor block map missing \(spec.name)"
                    )
                }
                guard let sourceDType = tensorSourceDTypes[spec.name] else {
                    throw SmeltPackageSpecLoweringError.unsupportedModel(
                        "Qwen3-TTS CAM tensor source dtype map missing \(spec.name)"
                    )
                }
                return tensorMap(
                    from: spec,
                    source: source.id,
                    block: block,
                    sourceDType: sourceDType
                )
            },
            quantization: try qwen3TTSQuantizationPlan(orderedSpecs),
            sidecars: sidecars,
            artifacts: artifacts,
            outputFiles: .init(files: qwen3TTSOutputFiles(
                artifacts: artifacts,
                sidecars: sidecars
            )),
            tokenizer: .init(format: "byte-bpe", files: tokenizerFiles),
            inference: .init(
                maxTokens: profile.maxTokens,
                eosTokens: eosTokens
            ),
            decode: try qwen3TTSDecodePolicy(decode),
            validation: validation(
                parityFixture: modelName,
                performanceGate: profile.performanceGate,
                structureProfile: profile.structureProfile(pipelines: pipelines, graph: graph)
            )
        )
    }

    public static func textGeneration(
        from manifest: SmeltManifest,
        packageName: String? = nil,
        sourcePath: String = "."
    ) throws -> SmeltPackageSpec {
        guard manifest.headlessTrunkABI != true else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "headless trunk manifests are not runnable text packages"
            )
        }
        guard let blocks = manifest.blocks, let loop = manifest.loop else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text manifest has no runnable block graph / loop"
            )
        }
        guard let tokenizerChecksum = manifest.checksums.tokenizerJSON,
              !tokenizerChecksum.isEmpty
        else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text manifest has no tokenizer.json checksum"
            )
        }

        let architecture = textGenerationArchitecture()
        let inference = try resolvedInferencePolicy(from: manifest)
        let decode = try resolvedDecodePolicy(from: manifest, inference: inference)
        let source = SmeltPackageSpec.Source(
            id: "legacy-package",
            kind: .localDirectory,
            path: sourcePath
        )
        let artifacts = textArtifacts(from: manifest.checksums)
        return SmeltPackageSpec(
            packageName: packageName ?? defaultPackageName(for: manifest.modelName),
            modelName: manifest.modelName,
            sources: [source],
            blocks: blocks,
            loop: loop,
            runtime: .forGraph(
                architecture: architecture,
                commands: allRuntimeCommands,
                graph: blocks
            ),
            architectureConfig: architectureConfig(for: manifest),
            tensors: manifest.weights.entries.map { tensorMap(from: $0, source: source.id) },
            artifacts: artifacts,
            outputFiles: .init(files: textOutputFiles(from: manifest.checksums)),
            tokenizer: .init(format: "tokenizer-json", files: ["tokenizer.json"]),
            inference: inference,
            decode: decode,
            validation: validation(
                parityFixture: manifest.modelName,
                performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                modelName: manifest.modelName
            )
        )
    }

    private static func validation(
        parityFixture: String?,
        performanceGate: String,
        modelName: String? = nil,
        structureProfile: SmeltPackageSpec.Validation.StructureProfile? = nil
    ) -> SmeltPackageSpec.Validation {
        SmeltPackagePerformanceProfiles.validation(
            parityFixture: parityFixture,
            performanceGate: performanceGate,
            modelName: modelName,
            structureProfile: structureProfile
        )
    }

    private static func textGenerationArchitecture() -> String {
        return SmeltRuntimeGraphPolicy.textGeneration.rawValue
    }

    private static func defaultPackageName(for modelName: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        let scalars = modelName.unicodeScalars.map {
            allowed.contains($0) ? Character($0) : "_"
        }
        return String(scalars) + ".smeltpkg"
    }

    private static func textArtifacts(
        prefill: SmeltPrefillConfig?
    ) -> [SmeltPackageSpec.Artifact] {
        var artifacts: [SmeltPackageSpec.Artifact] = [
            .init(id: "weights", path: "weights.bin", role: "weights"),
            .init(id: "metallib", path: "model.metallib", role: "metallib"),
            .init(id: "generated-swift", path: "SmeltGenerated.swift", role: "generated"),
            .init(id: "decode-dispatches", path: "dispatches.bin", role: "dispatch-table"),
            .init(
                id: "compiled-tokenizer",
                path: SmeltTokenizerPackageLayout.compiledFileName,
                role: "tokenizer-cache"
            ),
        ]
        artifacts.append(contentsOf: prefillPackageFiles(for: prefill).map {
            .init(id: $0.id, path: $0.path, role: "dispatch-table")
        })
        return artifacts
    }

    private static func textArtifacts(
        from checksums: SmeltManifestChecksums
    ) -> [SmeltPackageSpec.Artifact] {
        var artifacts = textBaseArtifacts()
        if checksums.tokenizerJSON != nil {
            artifacts.append(.init(
                id: "compiled-tokenizer",
                path: SmeltTokenizerPackageLayout.compiledFileName,
                role: "tokenizer-cache"
            ))
        }
        if checksums.prefillDispatchesBin != nil {
            artifacts.append(.init(
                id: "prefill-dispatches",
                path: "prefill_dispatches.bin",
                role: "dispatch-table"
            ))
        }
        if checksums.prefillVerifyArgmaxDispatchesBin != nil {
            artifacts.append(.init(
                id: "prefill-verify-argmax-dispatches",
                path: "prefill_verify_argmax_dispatches.bin",
                role: "dispatch-table"
            ))
        }
        return artifacts
    }

    private static func textBaseArtifacts() -> [SmeltPackageSpec.Artifact] {
        [
            .init(id: "weights", path: "weights.bin", role: "weights"),
            .init(id: "metallib", path: "model.metallib", role: "metallib"),
            .init(id: "generated-swift", path: "SmeltGenerated.swift", role: "generated"),
            .init(id: "decode-dispatches", path: "dispatches.bin", role: "dispatch-table"),
        ]
    }

    private static func textOutputFiles(
        from checksums: SmeltManifestChecksums
    ) -> [String] {
        var files = [
            "manifest.json",
            "weights.bin",
            "model.metallib",
            "SmeltGenerated.swift",
            "dispatches.bin",
        ]
        if checksums.tokenizerJSON != nil {
            files.append("tokenizer.json")
            files.append(SmeltTokenizerPackageLayout.compiledFileName)
        }
        if checksums.prefillDispatchesBin != nil {
            files.append("prefill_dispatches.bin")
        }
        if checksums.prefillVerifyArgmaxDispatchesBin != nil {
            files.append("prefill_verify_argmax_dispatches.bin")
        }
        return files
    }

    private static func artifactID(from path: String) -> String {
        path.map { char in
            char.isLetter || char.isNumber ? char : "-"
        }
        .reduce(into: "") { $0.append($1) }
        .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
    }

    private static func prefillPackageFiles(
        for prefill: SmeltPrefillConfig?
    ) -> [(id: String, path: String)] {
        guard let prefill else { return [] }
        var out = [(id: "prefill-dispatches", path: "prefill_dispatches.bin")]
        if prefill.verifyArgmax {
            out.append((
                id: "prefill-verify-argmax-dispatches",
                path: "prefill_verify_argmax_dispatches.bin"
            ))
        }
        return out
    }

    private static func tensorMap(
        from entry: SmeltWeightEntry,
        source: String
    ) -> SmeltPackageSpec.TensorMap {
        let dtype = tensorDType(from: entry.dtype)
        return SmeltPackageSpec.TensorMap(
            source: source,
            name: entry.name,
            canonicalName: entry.name,
            block: "trunk",
            sourceDType: dtype,
            storedDType: dtype,
            shape: entry.shape
        )
    }

    private static func tensorMap(
        from spec: Qwen3TTSPackageBuilder.WeightSpec,
        source: String,
        block: String,
        sourceDType: SmeltPackageSpec.TensorDType
    ) -> SmeltPackageSpec.TensorMap {
        let storedDType = tensorDType(from: spec.dtype)
        return SmeltPackageSpec.TensorMap(
            source: source,
            name: spec.name,
            canonicalName: spec.name,
            block: block,
            sourceDType: sourceDType,
            storedDType: storedDType,
            shape: spec.shape
        )
    }

    private static func tensorDType(from dtype: SmeltDType) -> SmeltPackageSpec.TensorDType {
        switch dtype {
        case .fp32: return .f32
        case .fp16: return .f16
        case .bf16: return .bf16
        case .u4Lut, .affineU4: return .u4
        case .binary1: return .binary1
        case .ternary2: return .ternary2
        case .turboQuantH: return .turboQuantH
        case .int32, .raw: return .raw
        }
    }

    private static func tensorDType(
        from dtype: Qwen3TTSPackageBuilder.WeightDType
    ) -> SmeltPackageSpec.TensorDType {
        switch dtype {
        case .f32: return .f32
        case .f16: return .f16
        case .bf16: return .bf16
        case .u4: return .u4
        }
    }

    private static func requireTensorBlocks(
        _ tensorBlocks: [String: String],
        exactlyCovering tensorNames: [String],
        allowedBlocks: Set<String>
    ) throws {
        let names = Set(tensorNames)
        let keys = Set(tensorBlocks.keys)
        guard keys == names else {
            let missing = names.subtracting(keys).sorted()
            let extra = keys.subtracting(names).sorted()
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "Qwen3-TTS CAM tensor block map drifted"
                    + (missing.isEmpty ? "" : " missing \(missing.joined(separator: ","))")
                    + (extra.isEmpty ? "" : " extra \(extra.joined(separator: ","))")
            )
        }
        let invalidBlocks = tensorBlocks.values.filter { !allowedBlocks.contains($0) }.sorted()
        guard invalidBlocks.isEmpty else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "Qwen3-TTS CAM tensor block map references unknown block "
                    + invalidBlocks.joined(separator: ",")
            )
        }
    }

    private static func requireTensorSourceDTypes(
        _ tensorSourceDTypes: [String: SmeltPackageSpec.TensorDType],
        exactlyCovering tensorNames: [String]
    ) throws {
        let names = Set(tensorNames)
        let keys = Set(tensorSourceDTypes.keys)
        guard keys == names else {
            let missing = names.subtracting(keys).sorted()
            let extra = keys.subtracting(names).sorted()
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "Qwen3-TTS CAM tensor source dtype map drifted"
                    + (missing.isEmpty ? "" : " missing \(missing.joined(separator: ","))")
                    + (extra.isEmpty ? "" : " extra \(extra.joined(separator: ","))")
            )
        }
        let invalidDTypes = tensorSourceDTypes.values
            .filter { ![.f32, .bf16].contains($0) }
            .map(\.rawValue)
            .sorted()
        guard invalidDTypes.isEmpty else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "Qwen3-TTS CAM tensor source dtype map references unsupported dtype "
                    + invalidDTypes.joined(separator: ",")
            )
        }
    }

    private static func qwen3TTSQuantizationPlan(
        _ specs: [Qwen3TTSPackageBuilder.WeightSpec]
    ) throws -> SmeltPackageSpec.QuantizationPlan? {
        let u4GroupSizes = Set(specs.compactMap { spec in
            spec.dtype == .u4 ? spec.groupSize : nil
        })
        guard !u4GroupSizes.isEmpty else { return nil }
        guard u4GroupSizes.count == 1, let groupSize = u4GroupSizes.first else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "Qwen3-TTS CAM lowering needs one resolved u4 group size"
            )
        }
        return .init(format: .u4, groupSize: groupSize)
    }

    private static func quantizationPlan(
        for quantization: SmeltQuantizationConfig
    ) -> SmeltPackageSpec.QuantizationPlan {
        let format: SmeltPackageSpec.TensorDType
        switch quantization.strategy {
        case .fp16: format = .f16
        case .lutU4, .affineU4: format = .u4
        case .binary1: format = .binary1
        case .ternary2: format = .ternary2
        case .turboQuantH: format = .turboQuantH
        }
        return .init(format: format, groupSize: quantization.groupSize)
    }

    private static func architectureConfig(for ir: SmeltModelIR) -> SmeltPackageSpecValue {
        var loading: [String: SmeltPackageSpecValue] = [
            "strategy": .string(ir.loading.strategy.rawValue),
            "packing": .string(ir.loading.packing.rawValue),
        ]
        if let checkpointMap = ir.loading.checkpointMap {
            loading["checkpoint_map"] = .string(checkpointMap.rawValue)
        }

        var object: [String: SmeltPackageSpecValue] = [
            "hidden_size": .int(ir.config.hiddenSize),
            "num_layers": .int(ir.config.numLayers),
            "vocab_size": .int(ir.config.vocabSize),
            "compiled_seq_capacity": .int(ir.config.compiledSeqCapacity),
            "rope_dim": .int(ir.config.ropeDim),
            "rms_eps": .number(Double(ir.config.rmsEps)),
            "norm_mode": .string(ir.config.normMode.rawValue),
            "activation_dtype": .string(ir.config.activationDtype.rawValue),
            "block_topology": .string(ir.config.blockTopology.rawValue),
            "tied_lm_head": .bool(ir.config.tiedLMHead),
            "layers": .object([
                "pattern": .array(ir.layerPattern.unit.map { .string($0.rawValue) }),
                "repeats": .int(ir.layerPattern.repeats),
            ]),
            "quantization": .object([
                "strategy": .string(ir.quantization.strategy.rawValue),
                "group_size": .int(ir.quantization.groupSize),
                "quantize_embedding": .bool(ir.quantization.quantizeEmbedding),
                "exclude": .array(ir.quantization.excludePatterns.map { .string($0) }),
                "turbo_quant_h": .array(ir.quantization.turboQuantHPatterns.map { .string($0) }),
                "preserve_native": .array(ir.quantization.preserveNativePatterns.map { .string($0) }),
            ]),
            "loading": .object(loading),
            "compilation": .object([
                "generated_kernels": .string(ir.compilation.generatedKernels.rawValue),
                "generated_kernel_consumers": .array(
                    ir.compilation.generatedKernelConsumerKindNames.map { .string($0) }
                ),
                "weight_layout": .string(ir.compilation.weightLayout.rawValue),
            ]),
        ]
        if let prefill = ir.prefill {
            object["prefill"] = .object([
                "engine": .string(prefill.engine),
                "model": .string(prefill.modelPath),
                "cache": .string(prefill.cachePath),
                "max_batch_size": .int(prefill.maxBatchSize),
                "handoff": .array(prefill.handoffFamilies.map { .string($0) }),
                "emit_all_logits": .bool(prefill.emitAllLogits),
                "verify_argmax": .bool(prefill.verifyArgmax),
            ])
        }
        if let logitCap = ir.config.logitCap {
            object["logit_cap"] = .number(Double(logitCap))
        }
        if let attnLogitCap = ir.config.attnLogitCap {
            object["attn_logit_cap"] = .number(Double(attnLogitCap))
        }
        if ir.config.sharedKVLayers > 0 {
            object["shared_kv_layers"] = .int(ir.config.sharedKVLayers)
        }
        if let sharedKVFFNDim = ir.config.sharedKVFFNDim {
            object["shared_kv_ffn_dim"] = .int(sharedKVFFNDim)
        }
        if let backboneHiddenSize = ir.config.backboneHiddenSize {
            object["backbone_hidden_size"] = .int(backboneHiddenSize)
        }
        if let fusion = ir.config.inputFusion {
            object["input_fusion_source_width"] = .int(fusion.sourceWidth)
            object["input_fusion_source_count"] = .int(fusion.sourceCount)
            object["input_fusion_normalize_sources"] = .bool(fusion.normalizeSources)
            if let postWidth = fusion.postProjectionWidth {
                object["input_fusion_post_projection_width"] = .int(postWidth)
            }
        }
        if let cluster = ir.config.clusterEmbedder {
            object["cluster_embedder"] = .object([
                "num_centroids": .int(cluster.numCentroids),
                "top_k": .int(cluster.topK),
            ])
        }
        return .object(object)
    }

    private static func architectureConfig(for manifest: SmeltManifest) -> SmeltPackageSpecValue {
        var object: [String: SmeltPackageSpecValue] = [
            "hidden_size": .int(manifest.config.hiddenSize),
            "num_layers": .int(manifest.config.numLayers),
            "vocab_size": .int(manifest.config.vocabSize),
            "static_seq_capacity": .int(manifest.config.staticContextCapacity),
            "rope_dim": .int(manifest.config.ropeDim),
            "num_delta_layers": .int(manifest.config.numDeltaLayers),
            "num_attn_layers": .int(manifest.config.numAttnLayers),
            "ffn_dim": .int(manifest.config.ffnDim),
            "pipelines": .array(manifest.pipelines.map { .string($0) }),
            "weight_total_bytes": .int(Int(manifest.weights.totalBytes)),
        ]
        if let options = manifest.buildProvenance?.resolvedOptions {
            var loading: [String: SmeltPackageSpecValue] = [
                "strategy": .string(options.loadingStrategy),
                "packing": .string(options.packing),
            ]
            if let checkpointMap = options.checkpointMap {
                loading["checkpoint_map"] = .string(checkpointMap)
            }
            object["loading"] = .object(loading)
        }
        if let hiddenActivation = manifest.config.hiddenActivation {
            object["hidden_activation"] = .string(hiddenActivation)
        }
        if let blockTopology = manifest.config.blockTopology {
            object["block_topology"] = .string(blockTopology)
        }
        if let layerPattern = manifest.config.layerPattern {
            object["layers"] = .object([
                "pattern": .array(layerPattern.pattern.map { .string($0) }),
                "repeats": .int(layerPattern.repeats),
            ])
        }
        if !manifest.config.attentionByRole.isEmpty {
            object["attention"] = .object(
                Dictionary(uniqueKeysWithValues: manifest.config.attentionByRole.map { role, attention in
                    (
                        role,
                        .object([
                            "q_heads": .int(attention.qHeads),
                            "kv_heads": .int(attention.kvHeads),
                            "head_dim": .int(attention.headDim),
                            "qk_norm": .bool(attention.qkNorm),
                            "v_norm": .bool(attention.vNorm),
                            "rope_theta": .number(attention.ropeTheta),
                            "rope_dim": .int(attention.ropeDim),
                            "sliding_window": .int(attention.slidingWindow),
                        ])
                    )
                })
            )
        }
        if let perLayerInput = manifest.config.perLayerInput {
            object["per_layer_input"] = .object([
                "hidden_size": .int(perLayerInput.hiddenSize),
                "vocab_size": .int(perLayerInput.vocabSize),
            ])
        }
        if let sharedKVLayers = manifest.config.sharedKVLayers, sharedKVLayers > 0 {
            object["shared_kv_layers"] = .int(sharedKVLayers)
        }
        if let logitCap = manifest.config.logitCap {
            object["logit_cap"] = .number(Double(logitCap))
        }
        if !manifest.config.turboQuantHPatterns.isEmpty {
            object["turbo_quant_h"] = .array(
                manifest.config.turboQuantHPatterns.map { .string($0) }
            )
        }
        if let prefill = manifest.prefill {
            object["prefill"] = .object([
                "engine": .string(prefill.engine),
                "model": .string(prefill.modelPath),
                "max_batch_size": .int(prefill.maxBatchSize),
                "handoff_entries": .int(prefill.handoff.entries.count),
            ])
        }
        return .object(object)
    }

    private static func resolvedInferencePolicy(
        from manifest: SmeltManifest
    ) throws -> SmeltInferenceManifest {
        guard let inference = manifest.inference else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text manifest has no inference policy"
            )
        }
        guard !inference.eosTokens.isEmpty else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text manifest has no eos tokens"
            )
        }
        return SmeltInferenceManifest(
            maxTokens: inference.maxTokens,
            eosTokens: inference.eosTokens,
            thinkToken: inference.thinkToken,
            thinkEndToken: inference.thinkEndToken,
            thinkSkipSuffix: inference.thinkSkipSuffix,
            chatTemplate: try resolvedTextChatTemplate(
                packageTemplate: inference.chatTemplate,
                modelName: manifest.modelName
            ),
            thinkingPolicy: inference.thinkingPolicy ?? .disabled,
            toolTranscriptCodec: inference.toolTranscriptCodec,
            promptStateRestoreMode: inference.promptStateRestoreMode
        )
    }

    private static func resolvedDecodePolicy(
        from manifest: SmeltManifest,
        inference: SmeltInferenceManifest
    ) throws -> SmeltPackageSpec.DecodePolicy {
        guard let decode = manifest.decode else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text manifest has no package-owned decode policy"
            )
        }
        guard decode.subSampler == nil else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text manifest decode policy cannot use sub_sampler"
            )
        }
        guard let maxSteps = decode.maxSteps else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text manifest decode policy has no max_steps"
            )
        }
        guard maxSteps == inference.maxTokens else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text manifest decode.max_steps does not match inference.max_tokens"
            )
        }
        guard decode.durationSeconds == nil else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text manifest decode policy cannot use duration_seconds"
            )
        }
        return decode
    }


    private static func qwen3TTSArtifacts(
        tokenizerFiles: [String]
    ) -> [SmeltPackageSpec.Artifact] {
        [
            .init(id: "weights", path: "weights.bin", role: "weights"),
            .init(id: "metallib", path: "model.metallib", role: "metallib"),
        ] + tokenizerFiles.map {
            .init(id: "tokenizer-\(artifactID(from: $0))", path: $0, role: "tokenizer")
        }
    }

    private static func qwen3TTSOutputFiles(
        artifacts: [SmeltPackageSpec.Artifact],
        sidecars: [SmeltPackageSpec.Sidecar]
    ) -> [String] {
        orderedUnique(
            ["manifest.json"] + artifacts.map(\.path) + sidecars.map(\.path)
        )
    }

    private static func qwen3TTSDecodePolicy(
        _ decode: Qwen3TTSManifest.Decode?
    ) throws -> SmeltPackageSpec.DecodePolicy {
        guard let decode else {
            return .init(sampler: .init(mode: .greedy))
        }
        if decode.doSample {
            guard decode.temperature > 0 else {
                throw SmeltPackageSpecLoweringError.unsupportedModel(
                    "Qwen3-TTS sampling temperature must be positive"
                )
            }
            guard decode.topK > 0 else {
                throw SmeltPackageSpecLoweringError.unsupportedModel(
                    "Qwen3-TTS sampling top_k must be positive"
                )
            }
            guard decode.subtalkerTemperature > 0 else {
                throw SmeltPackageSpecLoweringError.unsupportedModel(
                    "Qwen3-TTS subtalker sampling temperature must be positive"
                )
            }
            guard decode.subtalkerTopK > 0 else {
                throw SmeltPackageSpecLoweringError.unsupportedModel(
                    "Qwen3-TTS subtalker sampling top_k must be positive"
                )
            }
        }
        return .init(
            sampler: .init(
                mode: decode.doSample ? .sample : .greedy,
                temperature: decode.doSample ? Double(decode.temperature) : nil,
                topK: decode.doSample ? decode.topK : nil
            ),
            subSampler: decode.doSample ? .init(
                mode: .sample,
                temperature: Double(decode.subtalkerTemperature),
                topK: decode.subtalkerTopK
            ) : nil
        )
    }

    private static func qwen3TTSArchitectureConfig(
        runtimeArchitecture: String,
        modelName: String,
        eosTokens: [Int32],
        tokenizerFiles: [String],
        decode: Qwen3TTSManifest.Decode?,
        pipelines: [String],
        sidecarPaths: [String],
        pageSize: Int,
        totalBytes: UInt64,
        entries: [Qwen3TTSManifest.Entry]
    ) throws -> SmeltPackageSpecValue {
        guard let totalBytesInt = Int(exactly: totalBytes) else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "Qwen3-TTS total bytes \(totalBytes) do not fit in CAM integer config"
            )
        }
        var object: [String: SmeltPackageSpecValue] = [
            "architecture": .string(runtimeArchitecture),
            "model_name": .string(modelName),
            "page_size": .int(pageSize),
            "total_bytes": .int(totalBytesInt),
            "pipelines": .array(pipelines.map { .string($0) }),
            "eos_tokens": .array(eosTokens.map { .int(Int($0)) }),
            "tokenizer_files": .array(tokenizerFiles.map { .string($0) }),
            "sidecars": .array(sidecarPaths.map { .string($0) }),
            "weight_layout": try jsonValue(entries),
        ]
        if let decode {
            object["decode"] = try jsonValue(decode)
        }
        return .object(object)
    }

}

private func jsonValue<T: Encodable>(_ value: T) throws -> SmeltPackageSpecValue {
    let data = try JSONEncoder().encode(value)
    return try JSONDecoder().decode(SmeltPackageSpecValue.self, from: data)
}

private func orderedUnique(_ values: [String]) -> [String] {
    var seen = Set<String>()
    var out: [String] = []
    for value in values where seen.insert(value).inserted {
        out.append(value)
    }
    return out
}

public struct SmeltPackageAssemblySignature: Sendable, Equatable {
    public let architecture: String
    public let blockRoutes: [String]
    public let setupPhases: [String]
    public let perStepPhases: [String]
    public let emission: String
    public let tensorNames: [String]
    public let outputFiles: [String]
    public let chatTemplate: String?
    public let toolTranscriptCodec: String?
    public let promptStateRestoreMode: SmeltPromptStateRestoreMode?
    public let eosTokens: [Int32]

    public init(spec: SmeltPackageSpec) {
        architecture = spec.runtime.architecture
        blockRoutes = spec.runtime.routes.map(\.signature)
        setupPhases = spec.loop.setupSignatures
        perStepPhases = spec.loop.perStepSignatures
        emission = spec.loop.emissionSignature
        tensorNames = spec.tensors.map(\.canonicalName)
        outputFiles = spec.outputFiles.files.sorted()
        chatTemplate = spec.inference?.chatTemplate
        toolTranscriptCodec = spec.inference?.toolTranscriptCodec
        promptStateRestoreMode = spec.inference?.promptStateRestoreMode
        eosTokens = spec.inference?.eosTokens ?? []
    }

    public init(textManifest manifest: SmeltManifest) throws {
        guard let blocks = manifest.blocks, let loop = manifest.loop else {
            throw SmeltPackageSpecLoweringError.unsupportedModel(
                "text manifest has no runnable block graph / loop"
            )
        }
        architecture = SmeltRuntimeGraphPolicy.textGeneration.rawValue
        blockRoutes = blocks.runtimeRouteSignatures
        setupPhases = loop.setupSignatures
        perStepPhases = loop.perStepSignatures
        emission = loop.emissionSignature
        tensorNames = manifest.weights.entries.map(\.name)

        var files = textBasePackageFiles()
        if manifest.checksums.prefillDispatchesBin != nil {
            files.append("prefill_dispatches.bin")
        }
        if manifest.checksums.prefillVerifyArgmaxDispatchesBin != nil {
            files.append("prefill_verify_argmax_dispatches.bin")
        }
        if manifest.checksums.tokenizerJSON != nil {
            files.append("tokenizer.json")
            files.append(SmeltTokenizerPackageLayout.compiledFileName)
        }
        outputFiles = files.sorted()
        chatTemplate = try resolvedTextChatTemplate(
            packageTemplate: manifest.inference?.chatTemplate,
            modelName: manifest.modelName
        )
        toolTranscriptCodec = manifest.inference?.toolTranscriptCodec
        promptStateRestoreMode = manifest.inference?.promptStateRestoreMode
        eosTokens = manifest.inference?.eosTokens ?? []
    }
}

private func textBasePackageFiles() -> [String] {
    [
        "SmeltGenerated.swift",
        "dispatches.bin",
        "manifest.json",
        "model.metallib",
        "weights.bin",
    ]
}

/// CAM packages must carry the configured prompt template into lowered output.
private func resolvedTextChatTemplate(
    packageTemplate: String?,
    modelName: String
) throws -> String {
    guard let packageTemplate, !packageTemplate.isEmpty else {
        throw SmeltPackageSpecLoweringError.unsupportedModel(
            "text chat template for '\(modelName)' must be package-owned"
        )
    }
    return packageTemplate
}
