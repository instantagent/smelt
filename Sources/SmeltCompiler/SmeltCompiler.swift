// SmeltCompiler — Compiles an SmeltModelIR into a .smeltpkg artifact.
//
// The compiler reads a declarative model description, validates dimensions,
// processes weights, code-generates a model-specific Swift dispatch function,
// pre-compiles Metal shaders, and packages everything into a .smeltpkg.

import Foundation
import CryptoKit
import Metal
import SmeltRuntime
import SmeltSchema

public enum SmeltTraceMode: String, Sendable, CaseIterable {
    case full
    case stripped
    case strippedMarkers = "stripped-markers"

    var recordsTraceMarkers: Bool {
        switch self {
        case .full, .strippedMarkers:
            return true
        case .stripped:
            return false
        }
    }

    var usesStrippedOptimizations: Bool {
        switch self {
        case .full:
            return false
        case .stripped, .strippedMarkers:
            return true
        }
    }
}

/// Top-level compiler entry point.
public enum SmeltCompiler {

    final class OwnedCheckpointTensorBuffer {
        let pointer: UnsafeMutableRawPointer
        let byteCount: Int

        init(byteCount: Int, alignment: Int = MemoryLayout<Float>.alignment) {
            self.byteCount = byteCount
            self.pointer = .allocate(byteCount: byteCount, alignment: alignment)
        }

        deinit {
            pointer.deallocate()
        }
    }

    enum ExistingWeightsReuseValidation {
        case reuse([SmeltWeightEntry])
        case reuseWithWarning([SmeltWeightEntry], String)
        case rebuild(String)
    }

    private struct WeightsFingerprintInputs: Codable {
        let modelName: String
        let modelRevision: String?
        let config: SmeltManifestConfig
        let layerPatternUnit: [String]
        let layerPatternRepeats: Int
        let quantizationStrategy: String
        let groupSize: Int
        let excludePatterns: [String]
        let quantizeEmbedding: Bool
        // Checkpoint-map changes reinterpret source tensor names. Reusing
        // packed bytes across maps can silently bind the wrong tensors.
        let checkpointMap: String?
        let tiedLMHead: Bool
        let normMode: String
        let weightLayoutEntries: [String]
        // Adding / removing entries here flips the on-disk
        // weights.bin layout — must contribute to the fingerprint
        // so toggling TQH on/off rebuilds instead of reusing
        // stale bytes.
        let turboQuantHPatterns: [String]
        // Preserve-native opt-in glob list. Like turboQuantHPatterns,
        // adding / removing entries here flips the on-disk weights.bin
        // layout (native vs fp16-downcast projections), so it must
        // contribute to the fingerprint to rebuild instead of reusing
        // stale bytes.
        let preserveNativePatterns: [String]
        // Weight-layout policy is allowed to change physical storage.
        let compilationWeightLayout: String
        // Hash of the optional imatrix artifact: same spec + different imatrix
        // changes the codebook values, so it must rebuild rather than reuse.
        // Omitted from the JSON when nil (encodeIfPresent), so imatrix-free
        // packages keep their existing fingerprint.
        let imatrixFingerprint: String?
        // Hash of the injected GPTQ u4 blocks: same spec + different blocks
        // changes the quantized weights, so it must rebuild rather than reuse.
        // Omitted when nil, so non-GPTQ packages keep their existing fingerprint.
        let gptqFingerprint: String?
    }

    private struct PackageFingerprintInputs: Codable {
        let modelName: String
        let config: SmeltManifestConfig
        let resolvedOptions: SmeltResolvedBuildOptions
        let specSHA256: String
        let compilerSourcesSHA256: String
        let shaderSourcesSHA256: String
    }

    /// Source files that can change executable lowering but cannot map,
    /// normalize, encode, or lay out checkpoint tensors. Keeping this as an
    /// auditable classification prevents optimizer-only work from rewriting
    /// multi-gigabyte packed weights.
    static let weightByteIrrelevantSourcePaths: Set<String> = [
        "Sources/SmeltCompiler/SmeltCAMSourcePackageBuilder.swift",
        "Sources/SmeltCompiler/SmeltBufferPlan.swift",
        "Sources/SmeltCompiler/DenseTrunkEmitter.swift",
        "Sources/SmeltCompiler/DenseTrunkPrefillEmitter.swift",
        "Sources/SmeltCompiler/PrefillEmitter.swift",
        "Sources/SmeltCompiler/AttentionPlugin.swift",
        "Sources/SmeltCompiler/DeltaNetPlugin.swift",
        "Sources/SmeltCompiler/SmeltCodeEmitter.swift",
        "Sources/SmeltCompiler/SmeltDispatchOptimizer.swift",
        "Sources/SmeltCompiler/SmeltFusionPlanner.swift",
        "Sources/SmeltCompiler/SmeltGeneratedKernelVariants.swift",
        "Sources/SmeltCompiler/SmeltFrozenIRCostModel.swift",
        "Sources/SmeltCompiler/SmeltGraphCostModel.swift",
        "Sources/SmeltCompiler/SmeltKernelCapabilityRegistry.swift",
        "Sources/SmeltCompiler/SmeltKernelCatalog.swift",
        "Sources/SmeltCompiler/SmeltKernelConsumerNaming.swift",
        "Sources/SmeltCompiler/SmeltKernelPlanner.swift",
        "Sources/SmeltCompiler/SmeltKernelShapeRegistry.swift",
        "Sources/SmeltCompiler/SmeltMetalLibraryCompiler.swift",
        "Sources/SmeltCompiler/MatvecKernelTable.swift",
        "Sources/SmeltCompiler/TopLevelEmitter.swift",
        "Sources/SmeltRuntime/SmeltFrozenComponentPlanBuilder.swift",
        "Sources/SmeltRuntime/SmeltMetalFrozenOperationProfiler.swift",
        "Sources/SmeltRuntime/SmeltOptimizerReport.swift",
        "Sources/SmeltRuntime/SmeltQwen35VisionFrozenPlan.swift",
        "Sources/SmeltRuntime/SmeltSpeculativeDecode.swift",
        "Sources/SmeltRuntime/SmeltSuffixLookupDrafter.swift",
    ]

    static func sourceCanAffectPackedWeightBytes(_ path: String) -> Bool {
        !weightByteIrrelevantSourcePaths.contains(path)
    }

    /// Build result containing paths to all generated artifacts.
    public struct BuildResult {
        /// Path to the generated .smeltpkg directory.
        public let packagePath: String
        /// Path to the generated Swift dispatch file within the package.
        public let generatedSwiftPath: String
        /// Path to the compiled .metallib within the package.
        public let metallibPath: String
        /// Path to the manifest.json within the package.
        public let manifestPath: String
    }

    /// Build a .smeltpkg from an already assembled model IR.
    public static func build(
        ir: SmeltModelIR,
        inputName: String,
        sourceBaseDirectory: String = FileManager.default.currentDirectoryPath,
        outputDir: String,
        weightsDir: String? = nil,
        shaderDir: String,
        traceMode: SmeltTraceMode = .full,
        imatrixPath: String? = nil,
        gptqBlocks: [String: SmeltAffineU4.Packed]? = nil
    ) throws -> BuildResult {
        let fm = FileManager.default

        try validateSmeltIR(ir)
        try ensureCodegenSupport(for: ir)

        // Opt-in importance matrix for weighted TurboQuant-H codebooks; its
        // hash joins the weights fingerprint below.
        let imatrix = try imatrixPath.map { try SmeltImatrix.read(path: $0) }
        let imatrixFingerprint = try imatrixPath.map { try sha256Hex(ofFileAt: $0) }

        // Opt-in GPTQ u4 blocks injected in place of plain affine quantization.
        // GPTQ produces affine_u4 blocks, so it requires the affine_u4 strategy;
        // its hash joins the weights fingerprint below.
        if gptqBlocks != nil, ir.quantization.strategy != .affineU4 {
            throw SmeltCompilerError.unsupportedConfiguration(
                "gptqBlocks require the affine_u4 quantization strategy, got \(ir.quantization.strategy.rawValue)"
            )
        }
        if gptqBlocks != nil, let weightsDir,
           fm.fileExists(atPath: "\(weightsDir)/weights.json") {
            throw SmeltCompilerError.unsupportedConfiguration(
                "gptqBlocks cannot be injected into a pre-quantized weights.json checkpoint at \(weightsDir)"
            )
        }
        let gptqFingerprint = gptqBlocks.map { gptqBlocksFingerprint($0) }

        // --- Phase 2: Resolve the graph/kernel/weight compilation plan ---
        let expectedCompilationPlan = try planCompilation(ir: ir)
        let kernelPlan = expectedCompilationPlan.kernelPlan
        let manifestConfig = manifestConfigSnapshot(from: ir)
        let buildProvenance = try computeBuildProvenance(
            ir: ir,
            manifestConfig: manifestConfig,
            compilationPlan: expectedCompilationPlan,
            agentFile: inputName,
            shaderDir: shaderDir,
            traceMode: traceMode,
            imatrixFingerprint: imatrixFingerprint,
            gptqFingerprint: gptqFingerprint
        )

        // --- Phase 3: Resolve model name for package path ---
        let modelName = ir.modelName
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: " ", with: "_")
        let pkgPath = "\(outputDir)/\(modelName).smeltpkg"

        // Preserve existing weights.bin if present (skip 2-min re-quantization)
        let existingWeightsBin = "\(pkgPath)/weights.bin"
        let existingManifest = "\(pkgPath)/manifest.json"
        // An imatrix or GPTQ build must re-quantize from scratch: the reuse path
        // skips the fingerprint check for missing-provenance packages, and the
        // resume path keys only off weights.bin size — so a stale (unweighted, or
        // different-imatrix/GPTQ) weights.bin could be silently reused/resumed.
        // Drop prior artifacts before deciding, so such a build is always fresh.
        if imatrix != nil || gptqBlocks != nil, fm.fileExists(atPath: existingWeightsBin) {
            try removeExistingWeightArtifacts(pkgPath: pkgPath)
        }
        let hasExistingWeights = fm.fileExists(atPath: existingWeightsBin)

        if !fm.fileExists(atPath: pkgPath) {
            try fm.createDirectory(atPath: pkgPath, withIntermediateDirectories: true)
        }

        // --- Phase 4: Weight loading ---
        let baseWeightLayout: [SmeltWeightEntry]
        var hfCache: HFCacheEntry?
        var tokenizerAssets: HFTokenizerAssets?

        if hasExistingWeights, fm.fileExists(atPath: existingManifest) {
            switch try validateExistingWeightsForReuse(
                manifestPath: existingManifest,
                weightsPath: existingWeightsBin,
                buildProvenance: buildProvenance
            ) {
            case .reuse(let existingLayout):
                fputs("Reusing existing weights.bin (verified provenance match)\n", stderr)
                baseWeightLayout = existingLayout
            case .reuseWithWarning(let existingLayout, let warning)
                where ir.quantization.preserveNativePatterns.isEmpty:
                fputs("Warning: \(warning)\n", stderr)
                fputs("Reusing existing weights.bin without provenance verification\n", stderr)
                baseWeightLayout = existingLayout
            case .reuseWithWarning(_, let warning):
                // Never reuse UNVERIFIABLE bytes when the build's dtype layout depends on the exact
                // stored bytes: preserve_native is set — reusing fp16-downcast bytes
                // under a freshly-written bf16 manifest reads fp16 as bf16 garbage (the U2c
                // stale-reuse hazard). Force a rebuild.
                fputs(
                    "Not reusing unverifiable existing weights.bin "
                        + "(\(warning); provenance required for preserve_native builds)\n",
                    stderr
                )
                try removeExistingWeightArtifacts(pkgPath: pkgPath)
                baseWeightLayout = try buildWeightLayout(
                    ir: ir,
                    compilationPlan: expectedCompilationPlan,
                    weightsDir: weightsDir,
                    pkgPath: pkgPath,
                    imatrix: imatrix,
                    gptqBlocks: gptqBlocks,
                    hfCache: &hfCache
                )
            case .rebuild(let reason):
                fputs("Not reusing existing weights.bin: \(reason)\n", stderr)
                try removeExistingWeightArtifacts(pkgPath: pkgPath)
                baseWeightLayout = try buildWeightLayout(
                    ir: ir,
                    compilationPlan: expectedCompilationPlan,
                    weightsDir: weightsDir,
                    pkgPath: pkgPath,
                    imatrix: imatrix,
                    gptqBlocks: gptqBlocks,
                    hfCache: &hfCache
                )
            }
        } else if let weightsDir, fm.fileExists(atPath: "\(weightsDir)/weights.json") {
            // Local pre-quantized weights
            baseWeightLayout = try SmeltWeightManifestLoader.load(
                from: "\(weightsDir)/weights.json"
            )
            // Copy weights.bin to package (required)
            let srcPath = "\(weightsDir)/weights.bin"
            guard fm.fileExists(atPath: srcPath) else {
                throw SmeltCompilerError.noShaders("\(weightsDir)/weights.bin not found")
            }
            try fm.copyItem(atPath: srcPath, toPath: "\(pkgPath)/weights.bin")
        } else {
            baseWeightLayout = try buildWeightLayout(
                ir: ir,
                compilationPlan: expectedCompilationPlan,
                weightsDir: weightsDir,
                pkgPath: pkgPath,
                imatrix: imatrix,
                gptqBlocks: gptqBlocks,
                hfCache: &hfCache
            )
        }
        if let hfCache {
            tokenizerAssets = HFTokenizerAssets(
                tokenizerPath: hfCache.tokenizerPath,
                tokenizerConfigPath: hfCache.tokenizerConfigPath
            )
        } else {
            let packagedTokenizer = "\(pkgPath)/tokenizer.json"
            if !fm.fileExists(atPath: packagedTokenizer) {
                if let weightsDir,
                   fm.fileExists(atPath: "\(weightsDir)/tokenizer.json")
                {
                    let tokenizerConfig = "\(weightsDir)/tokenizer_config.json"
                    tokenizerAssets = HFTokenizerAssets(
                        tokenizerPath: "\(weightsDir)/tokenizer.json",
                        tokenizerConfigPath: fm.fileExists(atPath: tokenizerConfig)
                            ? tokenizerConfig
                            : nil
                    )
                } else {
                    tokenizerAssets = try HuggingFaceCache.resolveTokenizerAssets(
                        modelId: ir.modelName,
                        revision: ir.modelRevision ?? "main"
                    )
                }
            }
        }
        try removeStaleGeneratedPackageArtifacts(
            pkgPath: pkgPath,
            preserveTokenizerAssets: hfCache == nil
        )
        let compilationPlan = try planCompilation(
            ir: ir,
            weightLayout: baseWeightLayout,
            kernelPlan: kernelPlan
        )
        let weightLayout = compilationPlan.plannedWeightEntries
        let plan = compilationPlan.bufferPlan
        if !compilationPlan.kernelPlan.isEmpty {
            let report = compilationPlan.report
            fputs(
                "  Kernel plan: \(report.plannedKernelUses) planned uses, "
                    + "\(report.generatedKernels) generated capabilities, "
                    + "\(report.emittedGeneratedKernels) emitted generated kernels\n",
                stderr
            )
        }
        let weightStoragePlan = compilationPlan.weightStoragePlan
        if !weightStoragePlan.decisions.isEmpty {
            fputs(
                "  Weight storage plan: \(weightStoragePlan.decisions.count) planned weights, memory-neutral\n",
                stderr
            )
        }
        // --- Phase 5: Code generation + dispatch table ---
        let generateResult = try TopLevelEmitter.generate(
            ir: ir,
            compilationPlan: compilationPlan,
            traceMode: traceMode
        )
        try compilationPlan.validateGeneratedPipelineUses(
            generateResult.namedPipelineUses,
            context: "decode"
        )
        let generatedSource = generateResult.source
        let records = generateResult.dispatchRecords
        let decodeTraceMarkers = generateResult.traceMarkers
        var manifestPipelineNames = generateResult.pipelineNames
        fputs(
            "  Decode fusion rewrites: \(generateResult.optimizationStats.logSummary)\n",
            stderr
        )
        fputs(
            "  Decode fusion opportunities: \(generateResult.optimizationStats.opportunityLogSummary)\n",
            stderr
        )

        // --- Phase 5b: Prefill dispatch table (Metal engine) ---
        var prefillRecords: [SmeltDispatchRecord]?
        var prefillVerifyArgmaxRecords: [SmeltDispatchRecord]?
        var prefillTraceMarkers: [SmeltTraceMarker] = []
        var prefillGPTQCapturePoints: [SmeltGPTQCapturePoint] = []
        var prefillOptimizationStats = SmeltOptimizationStats()
        if let prefill = ir.prefill, prefill.engine == "metal" {
            let prefillResult = try PrefillEmitter.generate(
                ir: ir,
                compilationPlan: compilationPlan,
                traceMode: traceMode
            )
            try compilationPlan.validateGeneratedPipelineUses(
                prefillResult.namedPipelineUses,
                context: "prefill"
            )
            let (mergedPipelines, remappedPrefillRecords) = mergePipelineTables(
                baseNames: manifestPipelineNames,
                extraNames: prefillResult.pipelineNames,
                extraRecords: prefillResult.dispatchRecords
            )
            manifestPipelineNames = mergedPipelines
            prefillRecords = remappedPrefillRecords
            prefillTraceMarkers = prefillResult.traceMarkers
            prefillGPTQCapturePoints = prefillResult.gptqCapturePoints
            prefillOptimizationStats = prefillResult.optimizationStats
            fputs(
                "  Prefill dispatch table: \(prefillResult.dispatchRecords.count) ops\n",
                stderr
            )
            fputs(
                "  Prefill fusion rewrites: \(prefillOptimizationStats.logSummary)\n",
                stderr
            )
            fputs(
                "  Prefill fusion opportunities: \(prefillOptimizationStats.opportunityLogSummary)\n",
                stderr
            )
            if prefill.emitAllLogits || prefill.verifyArgmax {
                do {
                    let verifyArgmaxResult = try PrefillEmitter.generate(
                        ir: ir,
                        compilationPlan: compilationPlan,
                        traceMode: traceMode,
                        lmHeadMode: .verifyArgmaxOnly
                    )
                    try compilationPlan.validateGeneratedPipelineUses(
                        verifyArgmaxResult.namedPipelineUses,
                        context: "prefill verify argmax"
                    )
                    let (argmaxMerged, remappedArgmaxRecords) = mergePipelineTables(
                        baseNames: manifestPipelineNames,
                        extraNames: verifyArgmaxResult.pipelineNames,
                        extraRecords: verifyArgmaxResult.dispatchRecords
                    )
                    manifestPipelineNames = argmaxMerged
                    prefillVerifyArgmaxRecords = remappedArgmaxRecords
                    fputs(
                        "  Prefill verify argmax table: "
                            + "\(verifyArgmaxResult.dispatchRecords.count) ops\n",
                        stderr
                    )
                } catch PrefillEmitterError.unsupported(let detail) {
                    if prefill.verifyArgmax {
                        throw SmeltCompilerError.unsupportedConfiguration(
                            "requested prefill verify-argmax could not lower: \(detail)"
                        )
                    }
                    fputs(
                        "  Prefill verify argmax table skipped: \(detail)\n",
                        stderr
                    )
                }
            }
        }

        // --- Phase 6: Resolve handoff table ---
        var handoffTable: SmeltHandoffTable?
        if let prefill = ir.prefill {
            handoffTable = SmeltHandoffResolver.resolve(
                families: prefill.handoffFamilies,
                ir: ir,
                plan: plan
            )
        }

        // --- Phase 7: Compile Metal library ---
        let metallibDstPath = "\(pkgPath)/model.metallib"
        let generatedKernelSource = compilationPlan.generatedMetalSourceSuffix
        try compileMetalLibrary(
            shaderDir: shaderDir,
            outputPath: metallibDstPath,
            generatedLutMatvecSuffix: generatedKernelSource
        )

        // --- Phase 7b: Pre-compile pipeline binary archive ---
        // A cold system shader cache (first run on a machine) otherwise pays
        // full backend compilation for every pipeline at load. Best-effort:
        // archive misses at runtime just fall back to compilation.
        let archivePath = "\(pkgPath)/model.metalarchive"
        do {
            guard let archiveDevice = MTLCreateSystemDefaultDevice() else {
                throw SmeltCompilerError.metalCompileFailed("no Metal device")
            }
            let archiveLibrary = try archiveDevice.makeLibrary(
                URL: URL(fileURLWithPath: metallibDstPath)
            )
            let archive = try archiveDevice.makeBinaryArchive(
                descriptor: MTLBinaryArchiveDescriptor()
            )
            var archived = 0
            let samplerKernels = [
                "sample_temperature_fp16", "sample_temperature_gumbel_fp16"
            ]
            for entry in Set(manifestPipelineNames).sorted() + samplerKernels {
                guard let function = try? SmeltRuntime.pipelineFunction(
                    for: entry, library: archiveLibrary
                ) else { continue }
                let descriptor = MTLComputePipelineDescriptor()
                descriptor.computeFunction = function
                try archive.addComputePipelineFunctions(descriptor: descriptor)
                archived += 1
            }
            if fm.fileExists(atPath: archivePath) {
                try fm.removeItem(atPath: archivePath)
            }
            try archive.serialize(to: URL(fileURLWithPath: archivePath))
            fputs("  Pipeline archive: \(archived) pipelines\n", stderr)
        } catch {
            try? fm.removeItem(atPath: archivePath)
            fputs("  Pipeline archive skipped: \(error)\n", stderr)
        }

        // --- Phase 8: Write generated Swift + dispatch table ---
        let swiftPath = "\(pkgPath)/SmeltGenerated.swift"
        try generatedSource.write(
            toFile: swiftPath, atomically: true, encoding: .utf8
        )
        if !generatedKernelSource.isEmpty {
            try generatedKernelSource.write(
                toFile: "\(pkgPath)/SmeltGeneratedKernels.metal",
                atomically: true,
                encoding: .utf8
            )
        }

        // Write binary dispatch tables via the shared serializer (record layout can't drift).
        let dispatchPath = "\(pkgPath)/dispatches.bin"
        let dispatchBytes = try writeDispatchTable(records, to: dispatchPath)
        fputs(
            "  Dispatch table: \(records.count) ops, \(dispatchBytes / 1024) KB\n",
            stderr
        )

        // Write prefill dispatch table (if Metal prefill is enabled)
        if let prefillRecs = prefillRecords {
            let prefillPath = "\(pkgPath)/prefill_dispatches.bin"
            let prefillBytes = try writeDispatchTable(prefillRecs, to: prefillPath)
            fputs(
                "  Prefill table: \(prefillRecs.count) ops, \(prefillBytes / 1024) KB\n",
                stderr
            )
        }
        let verifyArgmaxPath = "\(pkgPath)/prefill_verify_argmax_dispatches.bin"
        if let verifyArgmaxRecs = prefillVerifyArgmaxRecords {
            let verifyArgmaxBytes = try writeDispatchTable(verifyArgmaxRecs, to: verifyArgmaxPath)
            fputs(
                "  Prefill verify argmax table: \(verifyArgmaxRecs.count) ops,"
                    + " \(verifyArgmaxBytes / 1024) KB\n",
                stderr
            )
        } else if fm.fileExists(atPath: verifyArgmaxPath) {
            try fm.removeItem(atPath: verifyArgmaxPath)
        }

        let traceMarkers = SmeltTraceMarkers(
            decode: decodeTraceMarkers,
            prefill: prefillTraceMarkers
        )
        let tracePath = "\(pkgPath)/trace_markers.json"
        let traceEncoder = JSONEncoder()
        traceEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let traceData = try traceEncoder.encode(traceMarkers)
        try traceData.write(to: URL(fileURLWithPath: tracePath))

        // GPTQ capture points — where calibration reads each in-scope projection's
        // activation input. Present only when the build records trace markers
        // (full trace mode), which a calibration build uses.
        let capturePointsPath = "\(pkgPath)/gptq_capture_points.json"
        if prefillGPTQCapturePoints.isEmpty {
            if fm.fileExists(atPath: capturePointsPath) { try fm.removeItem(atPath: capturePointsPath) }
        } else {
            let capturePoints = SmeltGPTQCapturePoints(prefill: prefillGPTQCapturePoints)
            let capData = try capturePoints.canonicalJSONData()
            try capData.write(to: URL(fileURLWithPath: capturePointsPath))
            fputs("  GPTQ capture points: \(prefillGPTQCapturePoints.count)\n", stderr)
        }

        // --- Phase 9: Copy prefill model + prompt cache from spec paths ---
        // Prefill asset paths are resolved relative to the source base directory.
        // Metal prefill doesn't need a CoreML model — skip the copy.
        if let prefill = ir.prefill {
            let prefillDst = "\(pkgPath)/prefill.mlmodelc"
            if prefill.engine != "metal", !prefill.modelPath.isEmpty, !fm.fileExists(atPath: prefillDst) {
                let resolved = resolvePath(prefill.modelPath, relativeTo: sourceBaseDirectory)
                if let resolved {
                    try fm.copyItem(atPath: resolved, toPath: prefillDst)
                    fputs("  Copied prefill model: \(resolved)\n", stderr)
                } else {
                    fputs(
                        "  Warning: prefill model not found: \(prefill.modelPath)\n",
                        stderr
                    )
                }
            }

            let cacheDst = "\(pkgPath)/cache"
            if !prefill.cachePath.isEmpty, !fm.fileExists(atPath: cacheDst) {
                let resolved = resolvePath(prefill.cachePath, relativeTo: sourceBaseDirectory)
                if let resolved {
                    try fm.copyItem(atPath: resolved, toPath: cacheDst)
                    fputs("  Copied prompt cache: \(resolved)\n", stderr)
                } else {
                    fputs(
                        "  Warning: prompt cache not found: \(prefill.cachePath)\n",
                        stderr
                    )
                }
            }
        }

        // --- Phase 10: Copy tokenizer assets (HF cache) ---
        if let tokenizerAssets {
            if let tokPath = tokenizerAssets.tokenizerPath {
                let tokenizerDst = "\(pkgPath)/tokenizer.json"
                try materializePackageFile(
                    sourcePath: tokPath,
                    destinationPath: tokenizerDst,
                    fileManager: fm
                )
            }
            if let tokConfigPath = tokenizerAssets.tokenizerConfigPath {
                // Extract special tokens
                if let data = try? Data(contentsOf: URL(fileURLWithPath: tokConfigPath)),
                   let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                {
                    var specialTokens: [String: Any] = [:]
                    for key in ["eos_token_id", "bos_token_id", "pad_token_id"] {
                        if let val = config[key] { specialTokens[key] = val }
                    }
                    if !specialTokens.isEmpty {
                        let stData = try JSONSerialization.data(
                            withJSONObject: specialTokens, options: .prettyPrinted
                        )
                        try stData.write(to: URL(fileURLWithPath: "\(pkgPath)/special_tokens.json"))
                    }
                }
            }
        }

        // --- Phase 10b: Compile tokenizer.json → tokenizer.bin ---
        // Always recompile: a stale tokenizer.bin from a previous build into
        // the same package directory would shadow the fresh tokenizer.json.
        let tokenizerJSONPath = "\(pkgPath)/tokenizer.json"
        let compiledTokenizerPath = "\(pkgPath)/\(SmeltTokenizer.compiledFileName)"
        if fm.fileExists(atPath: compiledTokenizerPath) {
            try fm.removeItem(atPath: compiledTokenizerPath)
        }
        if fm.fileExists(atPath: tokenizerJSONPath) {
            let tokenizer = try SmeltTokenizer(jsonPath: tokenizerJSONPath)
            try tokenizer.writeCompiledTokenizer(to: compiledTokenizerPath)
            fputs("  Compiled tokenizer: \(compiledTokenizerPath)\n", stderr)
        }

        // --- Phase 11: Write manifest with checksums + provenance ---
        let prefillDispatchPath = prefillRecords == nil ? nil : "\(pkgPath)/prefill_dispatches.bin"
        let prefillVerifyArgmaxDispatchPath = prefillVerifyArgmaxRecords == nil
            ? nil
            : "\(pkgPath)/prefill_verify_argmax_dispatches.bin"
        let checksums = try computeManifestChecksums(
            pkgPath: pkgPath,
            weightsPath: "\(pkgPath)/weights.bin",
            metallibPath: metallibDstPath,
            generatedSwiftPath: swiftPath,
            dispatchesPath: dispatchPath,
            prefillDispatchesPath: prefillDispatchPath,
            prefillVerifyArgmaxDispatchesPath: prefillVerifyArgmaxDispatchPath
        )
        // A headless trunk (embeddings-in / hidden-out, no LM head) is NOT a runnable
        // token-in/logits-out text generation package, so it must not stamp the
        // .tokenFeedbackText block graph/loop. Runtime front doors must not treat it
        // as runnable and sample an argmax the trunk
        // never wrote. Honest nil blocks/loop (matching the trunk sidecar); the headlessTrunkABI marker
        // + detect's rejection are the gate. (U3 honesty close, holistic-review blocker.)
        let isHeadlessTrunk = generateResult.isHeadlessTrunkABI
        let manifest = SmeltManifest(
            headlessTrunkABI: generateResult.isHeadlessTrunkABI,
            blocks: isHeadlessTrunk ? nil : .tokenFeedbackText,
            loop: isHeadlessTrunk ? nil : .tokenFeedbackText,
            modelName: ir.modelName,
            config: manifestConfig,
            context: nil,
            checksums: checksums,
            buildProvenance: buildProvenance,
            device: SmeltDeviceRequirements(
                metalFamily: .apple7,
                minMemoryBytes: UInt64(plan.totalActivationBytes)
            ),
            weights: SmeltWeightManifest(
                totalBytes: SmeltWeightManifestLoader.totalBytes(from: weightLayout),
                entries: weightLayout
            ),
            buffers: plan.toBufferTable(),
            pipelines: manifestPipelineNames,
            slotLayout: plan.toSlotLayout(),
            prefill: {
                guard let prefill = ir.prefill, let table = handoffTable else { return nil }
                return SmeltPrefillManifest(
                    engine: prefill.engine,
                    modelPath: "prefill.mlmodelc",
                    maxBatchSize: prefill.maxBatchSize,
                    handoff: table,
                    inputContract: SmeltPrefillInputContract()
                )
            }(),
            inference: {
                let inf = ir.inference
                // Emit the section if ANY inference property is set — not just
                // eosTokens — so a package that bakes only chat_template /
                // thinking_policy doesn't silently drop them from the manifest.
                guard !inf.eosTokens.isEmpty || inf.chatTemplate != nil
                    || inf.thinkingPolicy != nil else { return nil }
                return SmeltInferenceManifest(
                    maxTokens: inf.maxTokens,
                    eosTokens: inf.eosTokens,
                    thinkToken: inf.thinkToken,
                    thinkEndToken: inf.thinkEndToken,
                    thinkSkipSuffix: inf.thinkSkipSuffix,
                    chatTemplate: inf.chatTemplate,
                    thinkingPolicy: inf.thinkingPolicy,
                    toolTranscriptCodec: inf.toolTranscriptCodec,
                    promptStateRestoreMode: inf.promptStateRestoreMode
                )
            }(),
            decode: isHeadlessTrunk ? nil : ir.decode.policy,
            validation: isHeadlessTrunk
                ? nil
                : SmeltPackagePerformanceProfiles.validation(
                    parityFixture: ir.modelName,
                    performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                    modelName: ir.modelName
                ),
            optimizationReport: SmeltOptimizationReport(
                decodeRewriteCounts: generateResult.optimizationStats.rewriteCounts,
                prefillRewriteCounts: prefillOptimizationStats.rewriteCounts,
                decodeOpportunities: generateResult.optimizationStats.opportunities,
                prefillOpportunities: prefillOptimizationStats.opportunities,
                compilationPlan: compilationPlan.report
            )
        )

        let manifestDstPath = "\(pkgPath)/manifest.json"
        let manifestData = try manifest.encodePrettyJSON()
        try manifestData.write(to: URL(fileURLWithPath: manifestDstPath))

        return BuildResult(
            packagePath: pkgPath,
            generatedSwiftPath: swiftPath,
            metallibPath: metallibDstPath,
            manifestPath: manifestDstPath
        )
    }

    // MARK: - Path resolution

    /// Resolve a path relative to a base directory.
    /// Tries: absolute path, relative to base, relative to CWD.
    /// Returns the first path that exists, or nil.
    private static func resolvePath(
        _ path: String, relativeTo baseDir: String
    ) -> String? {
        let fm = FileManager.default
        // Absolute path
        if path.hasPrefix("/"), fm.fileExists(atPath: path) {
            return path
        }
        // Relative to the source base directory
        let relative = "\(baseDir)/\(path)"
        if fm.fileExists(atPath: relative) {
            return relative
        }
        // Relative to CWD (fallback)
        if fm.fileExists(atPath: path) {
            return path
        }
        return nil
    }

    // internal (not private): the Qwen3-TTS trunk sidecar reuses this to build its
    // co-resident manifest's config from the synthesised trunk IR.
    static func manifestConfigSnapshot(from ir: SmeltModelIR) -> SmeltManifestConfig {
        let attentionConfigs = ir.layerPattern.expanded.compactMap { ir.config.attentionConfig(for: $0) }
        let carriesExplicitBlockGraph = ir.config.blockTopology != .standard
        let attentionByRole: [String: SmeltManifestRoleAttentionConfig] = carriesExplicitBlockGraph
            ? Dictionary(uniqueKeysWithValues: ir.config.attentionConfigs.map { role, attention in
                (
                    role.rawValue,
                    SmeltManifestRoleAttentionConfig(
                        qHeads: attention.qHeads,
                        kvHeads: attention.kvHeads,
                        headDim: attention.headDim,
                        qkNorm: attention.qkNorm,
                        vNorm: attention.vNorm,
                        ropeTheta: Double(attention.ropeTheta),
                        ropeDim: attention.effectiveRopeDim(default: ir.config.ropeDim),
                        slidingWindow: attention.slidingWindow
                    )
                )
            })
            : [:]
        return SmeltManifestConfig(
            hiddenSize: ir.config.hiddenSize,
            numLayers: ir.config.numLayers,
            vocabSize: ir.config.vocabSize,
            hiddenActivation: ir.config.hiddenActivation?.rawValue,
            staticSeqCapacity: ir.config.staticSeqCapacity,
            ropeDim: ir.config.ropeDim,
            numDeltaLayers: ir.numDeltaLayers,
            numAttnLayers: ir.numAttnLayers,
            deltaNumHeads: ir.config.delta?.numHeads ?? 0,
            deltaQKVDim: ir.config.delta?.qkvDim ?? 0,
            attnQProjDim: attentionConfigs.map(\.qProjDim).max() ?? 0,
            attnKProjDim: attentionConfigs.map(\.kProjDim).max() ?? 0,
            attnVProjDim: attentionConfigs.map(\.vProjDim).max() ?? 0,
            attnOutDim: attentionConfigs.map { $0.qHeads * $0.headDim }.max() ?? 0,
            ffnDim: ir.config.maxFFNDim,
            blockTopology: carriesExplicitBlockGraph ? ir.config.blockTopology.rawValue : nil,
            layerPattern: carriesExplicitBlockGraph
                ? SmeltManifestLayerPattern(
                    pattern: ir.layerPattern.unit.map(\.rawValue),
                    repeats: ir.layerPattern.repeats
                )
                : nil,
            attentionByRole: attentionByRole,
            perLayerInput: ir.config.vocabSizePerLayerInput > 0 || ir.config.hiddenSizePerLayerInput > 0
                ? SmeltManifestPerLayerInputConfig(
                    hiddenSize: ir.config.hiddenSizePerLayerInput,
                    vocabSize: ir.config.vocabSizePerLayerInput
                )
                : nil,
            logitCap: ir.config.logitCap,
            sharedKVLayers: ir.config.sharedKVLayers,
            turboQuantHPatterns: ir.quantization.turboQuantHPatterns,
            inputFusion: ir.config.resolvedInputFusion.map {
                SmeltManifestInputFusionConfig(
                    sourceWidth: $0.sourceWidth,
                    sourceCount: $0.sourceCount,
                    normalizeSources: $0.normalizeSources,
                    postProjectionWidth: $0.postProjectionWidth
                )
            }
        )
    }

    static func ensureCodegenSupport(for ir: SmeltModelIR) throws {
        // Cluster-embedder gate lifted in unit 2c3: cluster.metal now
        // ships the cluster_sparse_lm_head kernel source. validateSmeltIR
        // still rejects the structurally-broken combinations
        // (quantized embedder, untied lm_head, non-divisible vocab,
        // top_k > num_centroids).
        if ir.config.sharedKVLayers > 0 {
            guard let firstShared = ir.firstKVSharedLayerIndex else {
                throw SmeltCompilerError.unsupportedConfiguration(
                    "shared KV layers require a valid trailing shared region"
                )
            }

            let sharedRegion = ir.layerPattern.expanded[firstShared...]
            if sharedRegion.contains(where: { !$0.isAttentionFamily }) {
                throw SmeltCompilerError.unsupportedConfiguration(
                    "shared KV layers currently require the trailing shared region to contain only attention-family layers"
                )
            }

            for layerIndex in firstShared..<ir.totalLayers {
                guard ir.kvSharedSourceGlobalLayerIndex(for: layerIndex) != nil else {
                    throw SmeltCompilerError.unsupportedConfiguration(
                        "shared KV layer \(layerIndex) has no earlier non-shared layer of the same attention family"
                    )
                }
            }
        }
        if ir.config.hiddenSizePerLayerInput > 0 || ir.config.vocabSizePerLayerInput > 0 {
            if ir.config.hiddenSizePerLayerInput <= 0 || ir.config.vocabSizePerLayerInput <= 0 {
                throw SmeltCompilerError.unsupportedConfiguration(
                    "staged per-layer residual inputs currently require both hidden_size_per_layer_input and vocab_size_per_layer_input to be positive"
                )
            }
            if ir.config.hiddenActivation != .geluPytorchTanh {
                throw SmeltCompilerError.unsupportedConfiguration(
                    "staged per-layer residual inputs currently require hidden_activation = gelu_pytorch_tanh"
                )
            }
            if ir.config.hiddenSizePerLayerInput > ir.config.maxFFNDim {
                throw SmeltCompilerError.unsupportedConfiguration(
                    "staged per-layer residual inputs currently require hidden_size_per_layer_input <= max FFN dim so existing FFN scratch buffers can be reused"
                )
            }
        }
        // TurboQuant-H validation has to live in the IR preflight
        // (not buildWeightLayout) because the weight-reuse branch
        // skips weight building entirely — a package with an
        // unsupported pattern (e.g. strategy=lutU4 + TQH, or
        // cluster_embedder + TQH-on-embed_tokens) would otherwise
        // sail past the gate on rebuilds and hit the broken
        // codegen path at runtime.
        if !ir.quantization.turboQuantHPatterns.isEmpty {
            // The TQH writer dispatch only lives inside
            // SmeltAffineQuantizer (via `quantizeTensorTurboQuantH`).
            // Strategy=.lutU4 / .fp16 routes through SmeltQuantizer,
            // which ignores turboQuantHPatterns — the embed_tokens
            // tensor would silently end up fp16 or u4_lut.
            guard ir.quantization.strategy == .affineU4 else {
                throw SmeltCompilerError.unsupportedRuntimeFeature(
                    "turbo_quant_h patterns require "
                    + "`strategy affine_u4`; current strategy is "
                    + "'\(ir.quantization.strategy.rawValue)' which "
                    + "routes through SmeltQuantizer (no TQH writer)"
                )
            }
            // Embed sites (decode `embed_tokens` and per-layer
            // `embed_tokens_per_layer`) route through
            // tqh_embedding_gather; FFN gate/up/down route through
            // emitMatvec's .turboQuantH branch (Unit 44) +
            // tqh_matvec kernels + the tqhMatvecXHatBuf scratch
            // slot (sized per Unit 50). Other consumers still need
            // wiring or scratch sizing — reject everything outside
            // this allowlist.
            let supportedPatterns: Set<String> = [
                SmeltCanonicalTensorNames.embedTokens,
                SmeltCanonicalTensorNames.embedTokensPerLayer,
                "layers_*_down_proj_weight",
                "layers_*_gate_proj_weight",
                "layers_*_up_proj_weight",
            ]
            // Each excluded pattern fails for a specific reason:
            //   - attn q/k/v/o proj: fused dual-matvec emitters
            //     have no .turboQuantH branch.
            //   - per_layer_input_gate_weight: emitMatvecVar /
            //     emitMatvecVarSlice have no .turboQuantH branch
            //     (they throw on TQH per Unit 50 audit fix).
            //   - untied lm_head_weight: just not allowlisted; the
            //     existing emitMatvec branch + scratch sizing
            //     already work for cols=hiddenSize.
            for pattern in ir.quantization.turboQuantHPatterns
            where !supportedPatterns.contains(pattern) {
                throw SmeltCompilerError.unsupportedRuntimeFeature(
                    "turbo_quant_h pattern '\(pattern)' is not in the "
                    + "supported allowlist. Supported today: "
                    + supportedPatterns.sorted().joined(separator: ", ")
                )
            }
            // cluster_embedder routes the LM head through a fused
            // clusterSparseLMHead kernel that reads lm_head_weight
            // (== embed_tokens entry on tied_lm_head packages) as
            // raw fp16. A TQH-encoded embed_tokens would be read as
            // garbage.
            if ir.config.clusterEmbedder != nil,
               ir.quantization.turboQuantHPatterns.contains(
                   SmeltCanonicalTensorNames.embedTokens
               )
            {
                throw SmeltCompilerError.unsupportedRuntimeFeature(
                    "turbo_quant_h pattern '"
                    + SmeltCanonicalTensorNames.embedTokens
                    + "' combined with cluster_embedder routes the "
                    + "TQH-encoded embed table through "
                    + "clusterSparseLMHead, which reads it as fp16. "
                    + "Drop "
                    + SmeltCanonicalTensorNames.embedTokens
                    + " from turbo_quant_h, or remove cluster_embedder."
                )
            }
        }
    }

    static func validateWeightStoragePlan(
        ir: SmeltModelIR,
        weightLayout: [SmeltWeightEntry],
        kernelPlan: SmeltKernelPlan? = nil
    ) throws -> SmeltWeightStoragePlan {
        try planCompilation(
            ir: ir,
            weightLayout: weightLayout,
            kernelPlan: kernelPlan
        ).weightStoragePlan
    }

    static func validateWeightStoragePlan(
        _ plan: SmeltWeightStoragePlan,
        policy: SmeltWeightLayoutPolicy
    ) throws {
        guard let failure = plan.validationFailure(policy: policy) else { return }

        switch failure {
        case .illegalStorage(let issues):
            throw SmeltCompilerError.unsupportedConfiguration(
                "kernel-planned weight layout has illegal storage: "
                    + issues.map(describeWeightStorageIssue).joined(separator: "; ")
            )
        case .duplicatePhysicalStorage(let duplicates):
            throw SmeltCompilerError.unsupportedConfiguration(
                "kernel-planned weight layout requires duplicate physical storage, "
                    + "but memory-neutral packaging is active: "
                    + duplicates.joined(separator: ", ")
            )
        }
    }

    /// Resolve the compiler's graph/kernel/weight-layout plan for an already parsed IR.
    public static func compilationPlanReport(
        ir: SmeltModelIR
    ) throws -> SmeltCompilationPlanReport {
        try planCompilation(ir: ir).report
    }

    static func planCompilation(
        ir: SmeltModelIR,
        kernelPlan: SmeltKernelPlan? = nil
    ) throws -> SmeltCompilationPlan {
        let bufferPlan = buildBufferPlan(from: ir)
        let kernelPlan = kernelPlan ?? SmeltKernelPlanner.plan(for: ir)
        let plannedWeightLayout = SmeltWeightLayoutPlanner.plannedLayout(
            for: ir,
            kernelPlan: kernelPlan
        )
        try validateWeightStoragePlan(
            plannedWeightLayout.storagePlan,
            policy: ir.compilation.weightLayout
        )
        return SmeltCompilationPlan(
            policy: ir.compilation,
            bufferPlan: bufferPlan,
            kernelPlan: kernelPlan,
            plannedWeightLayout: plannedWeightLayout
        )
    }

    static func planCompilation(
        ir: SmeltModelIR,
        weightLayout: [SmeltWeightEntry],
        kernelPlan: SmeltKernelPlan? = nil
    ) throws -> SmeltCompilationPlan {
        let bufferPlan = buildBufferPlan(from: ir)
        let kernelPlan = kernelPlan ?? SmeltKernelPlanner.plan(for: ir)
        let plannedWeightLayout = SmeltWeightLayoutPlanner.plannedLayout(
            entries: weightLayout,
            kernelPlan: kernelPlan
        )
        try validateWeightStoragePlan(
            plannedWeightLayout.storagePlan,
            policy: ir.compilation.weightLayout
        )
        return SmeltCompilationPlan(
            policy: ir.compilation,
            bufferPlan: bufferPlan,
            kernelPlan: kernelPlan,
            plannedWeightLayout: plannedWeightLayout
        )
    }

    private static func describeWeightStorageIssue(
        _ issue: SmeltWeightStorageIssue
    ) -> String {
        let consumers = issue.consumers.isEmpty
            ? "no planned consumers"
            : issue.consumers.map { consumer in
                let kind = consumer.consumerKind.map { " (\($0))" } ?? ""
                return consumer.consumerID + kind
            }.joined(separator: ", ")
        switch issue.kind {
        case .missingWeightEntry:
            return "\(issue.weightName) is missing for \(consumers)"
        case .unsupportedCurrentLayout:
            return "\(issue.weightName) has an unsupported current storage layout for \(consumers)"
        case .currentLayoutRejected:
            return "\(issue.weightName) current storage layout is rejected by \(consumers)"
        }
    }

    private static func resolvedBuildOptions(
        from ir: SmeltModelIR,
        traceMode: SmeltTraceMode
    ) -> SmeltResolvedBuildOptions {
        SmeltResolvedBuildOptions(
            layerPatternUnit: ir.layerPattern.unit.map(\.rawValue),
            layerPatternRepeats: ir.layerPattern.repeats,
            quantizationStrategy: ir.quantization.strategy.rawValue,
            groupSize: ir.quantization.groupSize,
            excludePatterns: ir.quantization.excludePatterns,
            quantizeEmbedding: ir.quantization.quantizeEmbedding,
            loadingStrategy: ir.loading.strategy.rawValue,
            packing: ir.loading.packing.rawValue,
            checkpointMap: ir.loading.checkpointMap?.rawValue,
            prefillEngine: ir.prefill?.engine,
            maxPrefillBatch: ir.prefill?.maxBatchSize,
            prefillHandoffFamilies: ir.prefill?.handoffFamilies ?? [],
            prefillEmitAllLogits: ir.prefill?.emitAllLogits ?? false,
            prefillVerifyArgmax: ir.prefill?.verifyArgmax ?? false,
            inferenceMaxTokens: ir.inference.maxTokens,
            eosTokens: ir.inference.eosTokens,
            thinkToken: ir.inference.thinkToken,
            thinkEndToken: ir.inference.thinkEndToken,
            thinkSkipSuffix: ir.inference.thinkSkipSuffix,
            tiedLMHead: ir.config.tiedLMHead,
            normMode: ir.config.normMode.rawValue,
            traceMode: traceMode.rawValue,
            turboQuantHPatterns: ir.quantization.turboQuantHPatterns,
            preserveNativePatterns: ir.quantization.preserveNativePatterns,
            compilationGeneratedKernels: ir.compilation.generatedKernels.rawValue,
            compilationWeightLayout: ir.compilation.weightLayout.rawValue
        )
    }

    /// The weights.bin reuse fingerprint. Internal (not private) so the
    /// stable/changed gates can exercise it directly: identical inputs must
    /// produce identical fingerprints (reuse), and any change to a quantization
    /// input hash must change it (rebuild instead of stale bytes).
    static func weightsFingerprintHex(
        ir: SmeltModelIR,
        manifestConfig: SmeltManifestConfig,
        weightLayoutEntries: [String],
        imatrixFingerprint: String?,
        gptqFingerprint: String?
    ) throws -> String {
        try sha256Hex(ofJSON: WeightsFingerprintInputs(
            modelName: ir.modelName,
            modelRevision: ir.modelRevision,
            config: manifestConfig,
            layerPatternUnit: ir.layerPattern.unit.map(\.rawValue),
            layerPatternRepeats: ir.layerPattern.repeats,
            quantizationStrategy: ir.quantization.strategy.rawValue,
            groupSize: ir.quantization.groupSize,
            excludePatterns: ir.quantization.excludePatterns,
            quantizeEmbedding: ir.quantization.quantizeEmbedding,
            checkpointMap: ir.loading.checkpointMap?.rawValue,
            tiedLMHead: ir.config.tiedLMHead,
            normMode: ir.config.normMode.rawValue,
            weightLayoutEntries: weightLayoutEntries,
            turboQuantHPatterns: ir.quantization.turboQuantHPatterns,
            preserveNativePatterns: ir.quantization.preserveNativePatterns,
            compilationWeightLayout: ir.compilation.weightLayout.rawValue,
            imatrixFingerprint: imatrixFingerprint,
            gptqFingerprint: gptqFingerprint
        ))
    }

    /// Internal for the fingerprint gates (see weightsFingerprintHex).
    static func manifestConfigSnapshotForTesting(from ir: SmeltModelIR) -> SmeltManifestConfig {
        manifestConfigSnapshot(from: ir)
    }

    private static func computeBuildProvenance(
        ir: SmeltModelIR,
        manifestConfig: SmeltManifestConfig,
        compilationPlan: SmeltCompilationPlan,
        agentFile: String,
        shaderDir: String,
        traceMode: SmeltTraceMode,
        imatrixFingerprint: String?,
        gptqFingerprint: String?
    ) throws -> SmeltBuildProvenance {
        let resolvedOptions = resolvedBuildOptions(from: ir, traceMode: traceMode)
        let specSHA256 = try sha256Hex(ofFileAt: agentFile)
        let compilerSourcesSHA256 = try sha256Hex(
            ofTreeRoots: [
                ("Sources/SmeltCompiler", Set(["swift"])),
                ("Sources/SmeltRuntime", Set(["swift"])),
                ("Sources/SmeltSchema", Set(["swift"])),
            ]
        )
        // These files lower an already-planned package into runtime dispatches
        // and kernels. They may change executable behavior, but do not map,
        // normalize, encode, or lay out checkpoint tensors. Keep this list
        // deliberately narrow: every other compiler/runtime source is treated
        // as capable of changing packed weight bytes. SmeltSchema declarations
        // are excluded as a root: resolved weight-affecting schema state is
        // already captured by the model/options/layout fingerprint below.
        let weightBuilderSourcesSHA256 = try sha256Hex(
            ofTreeRoots: [
                ("Sources/SmeltCompiler", Set(["swift"])),
                ("Sources/SmeltRuntime", Set(["swift"])),
            ],
            excludingPaths: weightByteIrrelevantSourcePaths
        )
        let shaderSourcesSHA256 = try sha256Hex(
            ofTreeRoots: [(shaderDir, Set(["metal", "h"]))]
        )
        let weightLayoutEntries = compilationPlan.plannedWeightEntries
            .map { entry in
                let shape = entry.shape.map(String.init).joined(separator: "x")
                return "\(entry.name):\(entry.dtype.rawValue):\(shape)"
            }

        let weightsFingerprint = try weightsFingerprintHex(
            ir: ir,
            manifestConfig: manifestConfig,
            weightLayoutEntries: weightLayoutEntries,
            imatrixFingerprint: imatrixFingerprint,
            gptqFingerprint: gptqFingerprint
        )
        let buildFingerprint = try sha256Hex(ofJSON: PackageFingerprintInputs(
            modelName: ir.modelName,
            config: manifestConfig,
            resolvedOptions: resolvedOptions,
            specSHA256: specSHA256,
            compilerSourcesSHA256: compilerSourcesSHA256,
            shaderSourcesSHA256: shaderSourcesSHA256
        ))

        return SmeltBuildProvenance(
            buildFingerprint: buildFingerprint,
            weightsFingerprint: weightsFingerprint,
            specSHA256: specSHA256,
            compilerSourcesSHA256: compilerSourcesSHA256,
            weightBuilderSourcesSHA256: weightBuilderSourcesSHA256,
            shaderSourcesSHA256: shaderSourcesSHA256,
            resolvedOptions: resolvedOptions
        )
    }

    static func validateExistingWeightsForReuse(
        manifestPath: String,
        weightsPath: String,
        buildProvenance: SmeltBuildProvenance
    ) throws -> ExistingWeightsReuseValidation {
        let manifestData = try Data(contentsOf: URL(fileURLWithPath: manifestPath))
        // This is a compiler-side reuse probe for a package we are about to
        // rewrite. Old packages may fail today's package-owned runtime policy
        // validation even though their weights/checksums are still usable.
        // Runtime loads and perf gates continue to use SmeltManifest.decode.
        let manifest = try JSONDecoder().decode(SmeltManifest.self, from: manifestData)
        let layout = manifest.weights.entries

        guard let existingProvenance = manifest.buildProvenance else {
            return .reuseWithWarning(layout, "existing package has no buildProvenance")
        }
        guard !manifest.checksums.weightsBin.isEmpty else {
            return .reuseWithWarning(layout, "existing package has no weights.bin checksum")
        }

        let actualWeightsChecksum = try sha256Hex(ofFileAt: weightsPath)
        guard actualWeightsChecksum == manifest.checksums.weightsBin else {
            return .rebuild(
                "weights.bin checksum mismatch (manifest \(manifest.checksums.weightsBin.prefix(12))..., actual \(actualWeightsChecksum.prefix(12))...)"
            )
        }

        // Legacy manifests did not distinguish code that writes weights from
        // code that only lowers dispatches. They retain the old conservative
        // behavior until rebuilt once with the granular fingerprint.
        if let existingWeightBuilderSources = existingProvenance.weightBuilderSourcesSHA256,
           let currentWeightBuilderSources = buildProvenance.weightBuilderSourcesSHA256
        {
            guard existingWeightBuilderSources == currentWeightBuilderSources else {
                return .rebuild("weight-building compiler/runtime sources changed")
            }
        } else {
            guard existingProvenance.compilerSourcesSHA256
                    == buildProvenance.compilerSourcesSHA256 else {
                return .rebuild("legacy compiler/runtime sources changed")
            }
        }

        guard existingProvenance.weightsFingerprint == buildProvenance.weightsFingerprint else {
            return .rebuild("weights were built with different quantization/model options")
        }

        return .reuse(layout)
    }

    private static func removeExistingWeightArtifacts(pkgPath: String) throws {
        let fm = FileManager.default
        for suffix in ["weights.bin", "weights.bin.progress"] {
            let path = "\(pkgPath)/\(suffix)"
            if fm.fileExists(atPath: path) {
                try fm.removeItem(atPath: path)
            }
        }
    }

    static func removeStaleGeneratedPackageArtifacts(
        pkgPath: String,
        preserveTokenizerAssets: Bool = false
    ) throws {
        let fm = FileManager.default
        var names = [
            "SmeltGenerated.swift",
            "SmeltGeneratedKernels.metal",
            "dispatches.bin",
            "prefill_dispatches.bin",
            "prefill_verify_argmax_dispatches.bin",
            "trace_markers.json",
            "gptq_capture_points.json",
            "manifest.json",
            "model.metallib",
            "model.metalarchive",
            "prefill.mlmodelc",
            "cache",
            SmeltBakeArtifacts.prefixMeta,
            SmeltBakeArtifacts.prefixSnapshot,
            SmeltBakeArtifacts.grammarMeta,
            SmeltBakeArtifacts.grammarTrie,
            SmeltPackageInterface.fileName,
            SmeltBakeManifest.fileName,
        ]
        if !preserveTokenizerAssets {
            names += [
                "tokenizer.json",
                "tokenizer.bin",
                "special_tokens.json",
            ]
        }
        for name in names {
            let path = "\(pkgPath)/\(name)"
            if packageArtifactExists(atPath: path, fileManager: fm) {
                try fm.removeItem(atPath: path)
            }
        }
    }

    private static func packageArtifactExists(atPath path: String, fileManager fm: FileManager) -> Bool {
        if fm.fileExists(atPath: path) {
            return true
        }
        return (try? fm.destinationOfSymbolicLink(atPath: path)) != nil
    }

    /// Copy file bytes into a package instead of copying the source directory
    /// entry. Hugging Face snapshots commonly expose files as relative
    /// symlinks into `../../blobs`; preserving that link in an unrelated
    /// package directory creates a dangling artifact that later inventory
    /// checks cannot distinguish from a missing file.
    static func materializePackageFile(
        sourcePath: String,
        destinationPath: String,
        fileManager fm: FileManager = .default
    ) throws {
        if packageArtifactExists(atPath: destinationPath, fileManager: fm) {
            try fm.removeItem(atPath: destinationPath)
        }
        let source = URL(fileURLWithPath: sourcePath)
            .resolvingSymlinksInPath().path
        try fm.copyItem(atPath: source, toPath: destinationPath)
    }

    static func missingRequiredCheckpointTensorNames(
        expectedLayout: [SmeltWeightEntry],
        availableMappedNames: Set<String>,
        adapter: SmeltCheckpointAdapter,
        config: SmeltConfig
    ) -> [String] {
        expectedLayout.compactMap { entry in
            if adapter.isTiedWeight(entry.name, config: config) {
                return nil
            }
            return availableMappedNames.contains(entry.name) ? nil : entry.name
        }
    }

    static func normalizedCheckpointShape(_ shape: [Int]) -> [Int] {
        // Conv1D checkpoints use both [channels, 1, kernel] and
        // [channels, kernel, 1]. Both are byte-contiguous views of the same
        // canonical [channels, kernel] tensor; flatten either singleton axis.
        if shape.count == 3 && (shape[1] == 1 || shape[2] == 1) {
            return [shape[0], shape[1] * shape[2]]
        }
        return shape
    }

    struct RemoteCheckpointPreflight {
        let probe: HFCheckpointProbeResult
        let tensorsByRuntimeName: [String: HFCheckpointTensorMetadata]
        let requiredTensors: [HFCheckpointTensorMetadata]
    }

    static func preflightRemoteCheckpoint(
        ir: SmeltModelIR,
        adapter: SmeltCheckpointAdapter,
        kernelPlan: SmeltKernelPlan? = nil
    ) throws -> RemoteCheckpointPreflight {
        let compilationPlan = try planCompilation(ir: ir, kernelPlan: kernelPlan)
        return try preflightRemoteCheckpoint(
            ir: ir,
            adapter: adapter,
            compilationPlan: compilationPlan
        )
    }

    static func preflightRemoteCheckpoint(
        ir: SmeltModelIR,
        adapter: SmeltCheckpointAdapter,
        compilationPlan: SmeltCompilationPlan
    ) throws -> RemoteCheckpointPreflight {
        let expectedLayout = compilationPlan.plannedWeightEntries
        let remoteProbe = try HuggingFaceCheckpointProbe.probe(
            modelId: ir.modelName,
            revision: ir.modelRevision ?? "main"
        )
        try adapter.validateConfig(hfConfig: remoteProbe.config, ir: ir)
        try validateSignedCheckpointConfiguration(hfConfig: remoteProbe.config, ir: ir)

        var remoteTensorsByRuntimeName: [String: HFCheckpointTensorMetadata] = [:]
        for tensor in remoteProbe.tensors {
            guard adapter.isTextModelTensor(tensor.name) else { continue }
            remoteTensorsByRuntimeName[adapter.mapName(tensor.name)] = tensor
        }

        let missingRequired = missingRequiredCheckpointTensorNames(
            expectedLayout: expectedLayout,
            availableMappedNames: Set(remoteTensorsByRuntimeName.keys),
            adapter: adapter,
            config: ir.config
        )
        if !missingRequired.isEmpty {
            let preview = missingRequired.prefix(8).joined(separator: ", ")
            throw SmeltCompilerError.unsupportedConfiguration(
                "checkpoint is missing \(missingRequired.count) required text tensors: \(preview)"
            )
        }

        try rejectUnsupportedMappedBiasTensorsIfNeeded(
            mappedNames: Set(remoteTensorsByRuntimeName.keys),
            expectedLayout: expectedLayout,
            adapter: adapter
        )

        for entry in expectedLayout {
            guard let remote = remoteTensorsByRuntimeName[entry.name] else {
                continue
            }
            if entry.dtype == .binary1 || entry.dtype == .ternary2 {
                try validateRemotePrequantizedTensor(
                    weight: remote,
                    allTensors: remoteProbe.tensors,
                    entry: entry
                )
            } else if let remoteShape = remote.shape {
                let normalizedShape = normalizedCheckpointShape(remoteShape)
                guard normalizedShape == entry.shape else {
                    throw SmeltCompilerError.unsupportedConfiguration(
                        "checkpoint tensor '\(remote.name)' mapped to '\(entry.name)' has shape \(normalizedShape), expected \(entry.shape)"
                    )
                }
            }
        }

        let requiredTensors: [HFCheckpointTensorMetadata]
        if signedQuantFormat(for: ir.quantization.strategy) != nil {
            // A logical packed tensor is an atomic source triplet. Request all
            // text-owned metadata so a single-file subset includes codes,
            // scales, and biases; sibling modality towers stay out.
            requiredTensors = remoteProbe.tensors.filter {
                adapter.isTextModelTensor($0.name)
            }
        } else {
            requiredTensors = expectedLayout.compactMap { entry in
                adapter.isTiedWeight(entry.name, config: ir.config)
                    ? nil : remoteTensorsByRuntimeName[entry.name]
            }
        }

        return RemoteCheckpointPreflight(
            probe: remoteProbe,
            tensorsByRuntimeName: remoteTensorsByRuntimeName,
            requiredTensors: requiredTensors
        )
    }

    static func signedQuantFormat(
        for strategy: SmeltQuantStrategy
    ) -> SmeltSignedQuantFormat? {
        switch strategy {
        case .binary1: return .binary1
        case .ternary2: return .ternary2
        case .fp16, .lutU4, .affineU4, .turboQuantH: return nil
        }
    }

    /// A signed strategy consumes already-quantized values. Prove that the
    /// checkpoint advertises the exact bit width and group geometry instead
    /// of interpreting arbitrary U32 tensors under the requested format.
    static func validateSignedCheckpointConfiguration(
        hfConfig: [String: Any],
        ir: SmeltModelIR
    ) throws {
        guard let format = signedQuantFormat(for: ir.quantization.strategy) else {
            return
        }
        guard let quantization = hfConfig["quantization"] as? [String: Any] else {
            throw SmeltCompilerError.unsupportedConfiguration(
                "\(format.rawValue) strategy requires checkpoint quantization metadata"
            )
        }
        let expectedBits = format.bitsPerCode
        guard let bits = (quantization["bits"] as? NSNumber)?.intValue,
              bits == expectedBits
        else {
            throw SmeltCompilerError.unsupportedConfiguration(
                "\(format.rawValue) strategy requires checkpoint quantization.bits = \(expectedBits)"
            )
        }
        guard let groupSize = (quantization["group_size"] as? NSNumber)?.intValue,
              groupSize == ir.quantization.groupSize
        else {
            throw SmeltCompilerError.unsupportedConfiguration(
                "\(format.rawValue) strategy requires checkpoint quantization.group_size = "
                    + "\(ir.quantization.groupSize)"
            )
        }
    }

    private static func validateRemotePrequantizedTensor(
        weight: HFCheckpointTensorMetadata,
        allTensors: [HFCheckpointTensorMetadata],
        entry: SmeltWeightEntry
    ) throws {
        guard weight.dtype == "U32",
              let weightShape = weight.shape,
              weightShape.count == 2,
              entry.shape.count == 2,
              let packedRowStride = entry.packedRowStride,
              packedRowStride % 4 == 0,
              let paddedCols = entry.paddedCols,
              let groupSize = entry.groupSize,
              paddedCols % groupSize == 0
        else {
            throw SmeltCompilerError.unsupportedConfiguration(
                "packed checkpoint tensor '\(weight.name)' lacks valid U32/layout geometry"
            )
        }
        let expectedWeightShape = [entry.shape[0], packedRowStride / 4]
        guard weightShape == expectedWeightShape else {
            throw SmeltCompilerError.unsupportedConfiguration(
                "packed checkpoint tensor '\(weight.name)' has shape \(weightShape), expected \(expectedWeightShape)"
            )
        }
        guard weight.name.hasSuffix(".weight") else {
            throw SmeltCompilerError.unsupportedConfiguration(
                "packed checkpoint tensor '\(weight.name)' does not use the atomic .weight contract"
            )
        }
        let base = String(weight.name.dropLast(".weight".count))
        let metadataByName = Dictionary(
            allTensors.map { ($0.name, $0) },
            uniquingKeysWith: { first, _ in first }
        )
        let expectedScaleShape = [entry.shape[0], paddedCols / groupSize]
        for suffix in [".scales", ".biases"] {
            let companionName = base + suffix
            guard let companion = metadataByName[companionName],
                  companion.dtype == "F16",
                  companion.shape == expectedScaleShape
            else {
                throw SmeltCompilerError.unsupportedConfiguration(
                    "packed checkpoint tensor '\(weight.name)' needs F16 companion "
                        + "'\(companionName)' with shape \(expectedScaleShape)"
                )
            }
        }
    }

    static func rejectUnsupportedMappedBiasTensorsIfNeeded(
        mappedNames: Set<String>,
        expectedLayout: [SmeltWeightEntry],
        adapter: SmeltCheckpointAdapter
    ) throws {
        guard adapter.rejectsUnsupportedMappedBiasTensors else { return }

        let expectedNames = Set(expectedLayout.map(\.name))
        let unsupportedBiases = mappedNames
            .subtracting(expectedNames)
            .filter { $0.hasSuffix("_bias") }
            .sorted()
        guard !unsupportedBiases.isEmpty else { return }

        let preview = unsupportedBiases.prefix(8).joined(separator: ", ")
        throw SmeltCompilerError.unsupportedConfiguration(
            "checkpoint contains \(unsupportedBiases.count) unsupported bias tensor(s) "
                + "that Smelt's dense text path cannot consume: \(preview)"
        )
    }

    static func preflightRemoteCheckpointMetadata(
        ir: SmeltModelIR,
        adapter: SmeltCheckpointAdapter,
        kernelPlan: SmeltKernelPlan? = nil
    ) throws -> [String: HFCheckpointTensorMetadata] {
        try preflightRemoteCheckpoint(
            ir: ir,
            adapter: adapter,
            kernelPlan: kernelPlan
        ).tensorsByRuntimeName
    }

    private static func buildWeightLayout(
        ir: SmeltModelIR,
        compilationPlan: SmeltCompilationPlan,
        weightsDir: String?,
        pkgPath: String,
        imatrix: [String: [Float]]?,
        gptqBlocks: [String: SmeltAffineU4.Packed]?,
        hfCache: inout HFCacheEntry?
    ) throws -> [SmeltWeightEntry] {
        let fm = FileManager.default
        let expectedLayout = compilationPlan.plannedWeightEntries

        if let weightsDir, fm.fileExists(atPath: "\(weightsDir)/weights.json") {
            let weightLayout = try SmeltWeightManifestLoader.load(
                from: "\(weightsDir)/weights.json"
            )
            let srcPath = "\(weightsDir)/weights.bin"
            guard fm.fileExists(atPath: srcPath) else {
                throw SmeltCompilerError.noShaders("\(weightsDir)/weights.bin not found")
            }
            try fm.copyItem(atPath: srcPath, toPath: "\(pkgPath)/weights.bin")
            return weightLayout
        }

        let adapter = try SmeltCheckpointAdapter.authored(for: ir)

        let cache: HFCacheEntry

        if let weightsDir,
           let localCache = try localHuggingFaceCheckpointCacheEntry(directory: weightsDir)
        {
            fputs("Loading local HuggingFace checkpoint from \(weightsDir)...\n", stderr)
            cache = localCache
        } else {
            fputs("Preflighting \(ir.modelName) checkpoint metadata...\n", stderr)
            let remotePreflight = try preflightRemoteCheckpoint(
                ir: ir,
                adapter: adapter,
                compilationPlan: compilationPlan
            )

            fputs("Downloading \(ir.modelName) from HuggingFace...\n", stderr)
            cache = try HuggingFaceCache.resolve(
                modelId: ir.modelName,
                revision: ir.modelRevision ?? "main",
                probe: remotePreflight.probe,
                requiredTensors: remotePreflight.requiredTensors
            )
        }
        hfCache = cache

        let configData = try Data(contentsOf: URL(fileURLWithPath: cache.configPath))
        if let configJson = try JSONSerialization.jsonObject(with: configData)
            as? [String: Any]
        {
            try adapter.validateConfig(hfConfig: configJson, ir: ir)
            try validateSignedCheckpointConfiguration(hfConfig: configJson, ir: ir)
        }

        let loader = try SafetensorsLoader(paths: cache.safetensorsPaths)
        if let signedFormat = signedQuantFormat(for: ir.quantization.strategy) {
            let source = try SmeltPrequantizedSafetensors(
                loader: loader,
                format: signedFormat,
                groupSize: ir.quantization.groupSize
            )
            let signedLayout = expectedLayout.filter {
                $0.dtype == .binary1 || $0.dtype == .ternary2
            }
            let denseLayout = expectedLayout.filter {
                $0.dtype != .binary1 && $0.dtype != .ternary2
            }
            let packedTensors = try assemblePrequantizedCheckpointTensors(
                source: source,
                adapter: adapter,
                expectedLayout: signedLayout,
                ir: ir
            )
            var denseTensors = try assembleCheckpointTensors(
                source: source,
                adapter: adapter,
                expectedLayout: denseLayout,
                ir: ir
            )
            let (compatibleTensors, ownedTensorBuffers) =
                try adjustedCheckpointTensorsForNormCompatibility(
                    denseTensors,
                    config: ir.config,
                    sourceSemantics: source.normWeightSemantics
                )
            denseTensors = compatibleTensors
            let (indexNarrowed, indexBuffers) =
                try adjustedCheckpointTensorsForIndexBuffers(
                    denseTensors,
                    config: ir.config
                )
            denseTensors = indexNarrowed
            let allOwnedBuffers = ownedTensorBuffers + indexBuffers

            fputs(
                "Writing \(packedTensors.count) native \(signedFormat.rawValue) tensors "
                    + "+ \(denseTensors.count) dense tensors...\n",
                stderr
            )
            return try withExtendedLifetime(source) {
                try withExtendedLifetime(allOwnedBuffers) {
                    try SmeltNativeWeightWriter.write(
                        packedTensors: packedTensors,
                        denseTensors: denseTensors,
                        expectedLayout: expectedLayout,
                        outputPath: "\(pkgPath)/weights.bin"
                    )
                }
            }
        }

        let device = MTLCreateSystemDefaultDevice()!
        var tensors = try assembleCheckpointTensors(
            source: loader,
            adapter: adapter,
            expectedLayout: expectedLayout,
            ir: ir
        )

        let (compatibleTensors, ownedTensorBuffers) =
            try adjustedCheckpointTensorsForNormCompatibility(
                tensors,
                config: ir.config
            )
        tensors = compatibleTensors

        let (indexNarrowed, indexBuffers) =
            try adjustedCheckpointTensorsForIndexBuffers(
                tensors,
                config: ir.config
            )
        tensors = indexNarrowed
        let allOwnedBuffers = ownedTensorBuffers + indexBuffers

        let hasDirectDenseStorage = expectedLayout.allSatisfy {
            $0.dtype == .bf16 || $0.dtype == .fp32
        }
        if ir.config.portTopology == .embeddingsInHiddenOut,
           hasDirectDenseStorage
        {
            // Dense ported trunk: the layout declares each storage family;
            // materialize it directly with no activation-family capability
            // branch inside the writer.
            fputs("Writing \(tensors.count) dense trunk tensors...\n", stderr)
            return try withExtendedLifetime(loader) {
                try withExtendedLifetime(allOwnedBuffers) {
                    try SmeltFP32TrunkWriter.write(
                        tensors: tensors,
                        expectedLayout: expectedLayout,
                        outputPath: "\(pkgPath)/weights.bin"
                    )
                }
            }
        }

        if ir.quantization.strategy == .affineU4 {
            fputs("Quantizing \(tensors.count) tensors (affine u4, CPU)...\n", stderr)
            return try withExtendedLifetime(loader) {
                try withExtendedLifetime(allOwnedBuffers) {
                    try SmeltAffineQuantizer.quantize(
                        tensors: tensors,
                        config: ir.quantization,
                        outputPath: "\(pkgPath)/weights.bin",
                        expectedLayout: expectedLayout,
                        imatrix: imatrix,
                        gptqBlocks: gptqBlocks,
                        activationDtype: ir.config.activationDtype
                    )
                }
            }
        }

        fputs("Quantizing \(tensors.count) tensors on Metal GPU...\n", stderr)
        return try withExtendedLifetime(loader) {
            try withExtendedLifetime(allOwnedBuffers) {
                try SmeltQuantizer.quantize(
                    tensors: tensors,
                    config: ir.quantization,
                    outputPath: "\(pkgPath)/weights.bin",
                    device: device,
                    activationDtype: ir.config.activationDtype
                )
            }
        }
    }

    /// Assemble the quantizer's input from any checkpoint container: map
    /// source-domain names through the adapter, check coverage against the
    /// expected layout, validate shapes, and vend (name, data, shape, dtype)
    /// tuples. Pointers are valid only while `source` is alive — callers wrap
    /// downstream use in `withExtendedLifetime(source)`.
    static func assembleCheckpointTensors(
        source: any CheckpointTensorSource,
        adapter: SmeltCheckpointAdapter,
        expectedLayout: [SmeltWeightEntry],
        ir: SmeltModelIR
    ) throws -> [(runtimeName: String, data: UnsafeRawPointer,
                  byteCount: Int, shape: [Int], dtype: String)] {
        var byRuntimeName: [String: CheckpointTensorDescriptor] = [:]
        for descriptor in source.checkpointTensors {
            guard adapter.isTextModelTensor(descriptor.name) else { continue }
            let mapped = adapter.mapName(descriptor.name)
            // Exact-coverage adapters require an injective map: a source that
            // vends two descriptors for the same canonical name (duplicate
            // shards, an arbitrary source) would silently collapse to one entry
            // and pass exact coverage while having dropped a tensor. Reject loud.
            if adapter.requiresExactTrunkCoverage, let prior = byRuntimeName[mapped] {
                throw SmeltCompilerError.unsupportedConfiguration(
                    "checkpoint maps two tensors to the same trunk name '\(mapped)' "
                    + "('\(prior.name)' and '\(descriptor.name)')"
                )
            }
            byRuntimeName[mapped] = descriptor
        }

        let missingRequired = missingRequiredCheckpointTensorNames(
            expectedLayout: expectedLayout,
            availableMappedNames: Set(byRuntimeName.keys),
            adapter: adapter,
            config: ir.config
        )
        if !missingRequired.isEmpty {
            let preview = missingRequired.prefix(8).joined(separator: ", ")
            throw SmeltCompilerError.unsupportedConfiguration(
                "checkpoint is missing \(missingRequired.count) required text tensors: \(preview)"
            )
        }

        try rejectUnsupportedMappedBiasTensorsIfNeeded(
            mappedNames: Set(byRuntimeName.keys),
            expectedLayout: expectedLayout,
            adapter: adapter
        )

        // Exact-coverage adapters (the talker trunk) forbid silently keeping a
        // mapped tensor that has no home in the layout — the loop below only
        // reads expectedLayout, so any extra would vanish without a trace.
        if adapter.requiresExactTrunkCoverage {
            let expectedNames = Set(expectedLayout.map(\.name))
            let extras = Set(byRuntimeName.keys).subtracting(expectedNames)
            if !extras.isEmpty {
                let preview = extras.sorted().prefix(8).joined(separator: ", ")
                throw SmeltCompilerError.unsupportedConfiguration(
                    "checkpoint has \(extras.count) tensor(s) the adapter mapped into the trunk but the layout does not expect: \(preview)"
                )
            }
        }

        var tensors: [(runtimeName: String, data: UnsafeRawPointer,
                       byteCount: Int, shape: [Int], dtype: String)] = []
        for entry in expectedLayout {
            guard let descriptor = byRuntimeName[entry.name] else {
                if adapter.isTiedWeight(entry.name, config: ir.config) {
                    continue
                }
                throw SmeltCompilerError.unsupportedConfiguration(
                    "checkpoint tensor '\(entry.name)' not found after adapter coverage check"
                )
            }
            let shape = normalizedCheckpointShape(descriptor.shape)
            guard shape == entry.shape else {
                throw SmeltCompilerError.unsupportedConfiguration(
                    "checkpoint tensor '\(descriptor.name)' mapped to '\(entry.name)' has shape \(shape), expected \(entry.shape)"
                )
            }
            tensors.append((
                runtimeName: entry.name,
                data: source.checkpointTensorData(descriptor),
                byteCount: descriptor.byteCount,
                shape: shape,
                dtype: descriptor.dtype
            ))
        }
        return tensors
    }

    /// Map one semantic packed tensor per source triplet through the same
    /// authored checkpoint adapter and exact layout coverage checks as dense
    /// tensors. Container spelling never reaches the writer or runtime.
    static func assemblePrequantizedCheckpointTensors(
        source: SmeltPrequantizedSafetensors,
        adapter: SmeltCheckpointAdapter,
        expectedLayout: [SmeltWeightEntry],
        ir: SmeltModelIR
    ) throws -> [SmeltNativeWeightWriter.PackedTensor] {
        var byRuntimeName: [String: SmeltPrequantizedTensorView] = [:]
        for view in source.prequantizedTensors {
            let sourceName = view.descriptor.sourceName
            guard adapter.isTextModelTensor(sourceName) else { continue }
            let mapped = adapter.mapName(sourceName)
            guard byRuntimeName.updateValue(view, forKey: mapped) == nil else {
                throw SmeltCompilerError.unsupportedConfiguration(
                    "checkpoint maps multiple packed tensors to '\(mapped)'"
                )
            }
        }

        let missingRequired = missingRequiredCheckpointTensorNames(
            expectedLayout: expectedLayout,
            availableMappedNames: Set(byRuntimeName.keys),
            adapter: adapter,
            config: ir.config
        )
        if !missingRequired.isEmpty {
            let preview = missingRequired.prefix(8).joined(separator: ", ")
            throw SmeltCompilerError.unsupportedConfiguration(
                "checkpoint is missing \(missingRequired.count) required packed text tensors: \(preview)"
            )
        }

        let expectedNames = Set(expectedLayout.map(\.name))
        let extras = Set(byRuntimeName.keys).subtracting(expectedNames)
        if !extras.isEmpty {
            let preview = extras.sorted().prefix(8).joined(separator: ", ")
            throw SmeltCompilerError.unsupportedConfiguration(
                "checkpoint has \(extras.count) packed text tensor(s) with no planned layout entry: \(preview)"
            )
        }

        return expectedLayout.compactMap { entry in
            byRuntimeName[entry.name].map {
                SmeltNativeWeightWriter.PackedTensor(runtimeName: entry.name, view: $0)
            }
        }
    }

    private static func localHuggingFaceCheckpointCacheEntry(
        directory: String
    ) throws -> HFCacheEntry? {
        let fm = FileManager.default
        let configPath = "\(directory)/config.json"
        guard fm.fileExists(atPath: configPath) else {
            return nil
        }

        let indexPath = "\(directory)/model.safetensors.index.json"
        let singlePath = "\(directory)/model.safetensors"
        let safetensorsPaths: [String]

        if fm.fileExists(atPath: indexPath) {
            let indexData = try Data(contentsOf: URL(fileURLWithPath: indexPath))
            guard let index = try JSONSerialization.jsonObject(with: indexData)
                as? [String: Any],
                  let weightMap = index["weight_map"] as? [String: String]
            else {
                throw SmeltCompilerError.unsupportedConfiguration(
                    "Invalid safetensors index at \(indexPath)"
                )
            }

            var seen = Set<String>()
            safetensorsPaths = try weightMap.values.sorted().compactMap { file in
                guard seen.insert(file).inserted else { return nil }
                let path = "\(directory)/\(file)"
                guard fm.fileExists(atPath: path) else {
                    throw SmeltCompilerError.unsupportedConfiguration(
                        "Local HuggingFace checkpoint is missing shard \(path)"
                    )
                }
                return path
            }
        } else if fm.fileExists(atPath: singlePath) {
            safetensorsPaths = [singlePath]
        } else {
            return nil
        }

        let tokenizerPath = "\(directory)/tokenizer.json"
        let tokenizerConfigPath = "\(directory)/tokenizer_config.json"
        return HFCacheEntry(
            directory: directory,
            safetensorsPaths: safetensorsPaths,
            configPath: configPath,
            tokenizerPath: fm.fileExists(atPath: tokenizerPath) ? tokenizerPath : nil,
            tokenizerConfigPath: fm.fileExists(atPath: tokenizerConfigPath)
                ? tokenizerConfigPath
                : nil
        )
    }

    static func shouldShiftNormWeightForCompatibility(
        runtimeName: String,
        config: SmeltConfig,
        sourceSemantics: CheckpointNormWeightSemantics = .modelDeclared
    ) -> Bool {
        // Dense ported trunks use weight-direct norm kernels — no (1+w), so
        // no checkpoint-time w−1 shift.
        guard config.portTopology == .tokenInLogitsOut else { return false }
        let sourceNormMode: SmeltNormMode
        switch sourceSemantics {
        case .modelDeclared:
            sourceNormMode = config.normMode
        case .directWeight:
            sourceNormMode = .weight
        case .onePlusDelta:
            sourceNormMode = .onePlusWeight
        }
        guard sourceNormMode == .weight else { return false }

        if runtimeName == "norm_weight" {
            return true
        }

        let compatibilitySuffixes = [
            "_input_layernorm_weight",
            "_post_attention_layernorm_weight",
            "_pre_feedforward_layernorm_weight",
            "_post_feedforward_layernorm_weight",
            "_post_per_layer_input_norm_weight",
        ]
        return compatibilitySuffixes.contains { runtimeName.hasSuffix($0) }
    }

    static func adjustedCheckpointTensorsForNormCompatibility(
        _ tensors: [(runtimeName: String, data: UnsafeRawPointer,
                     byteCount: Int, shape: [Int], dtype: String)],
        config: SmeltConfig,
        sourceSemantics: CheckpointNormWeightSemantics = .modelDeclared
    ) throws -> (
        tensors: [(runtimeName: String, data: UnsafeRawPointer,
                   byteCount: Int, shape: [Int], dtype: String)],
        ownedBuffers: [OwnedCheckpointTensorBuffer]
    ) {
        var adjusted: [(runtimeName: String, data: UnsafeRawPointer,
                        byteCount: Int, shape: [Int], dtype: String)] = []
        adjusted.reserveCapacity(tensors.count)

        var ownedBuffers: [OwnedCheckpointTensorBuffer] = []

        for tensor in tensors {
            guard shouldShiftNormWeightForCompatibility(
                runtimeName: tensor.runtimeName,
                config: config,
                sourceSemantics: sourceSemantics
            ) else {
                adjusted.append(tensor)
                continue
            }

            let buffer = try shiftedNormWeightBufferForOnePlusWeightCompatibility(
                source: tensor.data,
                shape: tensor.shape,
                dtype: tensor.dtype
            )
            ownedBuffers.append(buffer)
            adjusted.append((
                runtimeName: tensor.runtimeName,
                data: UnsafeRawPointer(buffer.pointer),
                byteCount: buffer.byteCount,
                shape: tensor.shape,
                dtype: "F32"
            ))
        }

        return (adjusted, ownedBuffers)
    }

    /// Convert assistant-only int64 index buffers (currently just
    /// `masked_embedding_token_ordering`) to int32 with bounds validation.
    /// HF ships int64; smelt's layout and the `cluster_sparse_lm_head`
    /// kernel read int32. Out-of-range source values would silently
    /// scramble the sparse lm_head — the kernel writes through the
    /// permuted index — so reject them here rather than later.
    static func adjustedCheckpointTensorsForIndexBuffers(
        _ tensors: [(runtimeName: String, data: UnsafeRawPointer,
                     byteCount: Int, shape: [Int], dtype: String)],
        config: SmeltConfig
    ) throws -> (
        tensors: [(runtimeName: String, data: UnsafeRawPointer,
                   byteCount: Int, shape: [Int], dtype: String)],
        ownedBuffers: [OwnedCheckpointTensorBuffer]
    ) {
        var adjusted: [(runtimeName: String, data: UnsafeRawPointer,
                        byteCount: Int, shape: [Int], dtype: String)] = []
        adjusted.reserveCapacity(tensors.count)
        var ownedBuffers: [OwnedCheckpointTensorBuffer] = []

        for tensor in tensors {
            guard tensor.runtimeName == SmeltCanonicalTensorNames.maskedEmbeddingTokenOrdering else {
                adjusted.append(tensor)
                continue
            }

            switch tensor.dtype {
            case "I32":
                adjusted.append(tensor)
            case "I64":
                let buffer = try downcastInt64IndexBufferToInt32(
                    name: tensor.runtimeName,
                    source: tensor.data,
                    shape: tensor.shape,
                    upperBoundExclusive: config.vocabSize
                )
                ownedBuffers.append(buffer)
                let elementCount = tensor.shape.reduce(1, *)
                adjusted.append((
                    runtimeName: tensor.runtimeName,
                    data: UnsafeRawPointer(buffer.pointer),
                    byteCount: elementCount * MemoryLayout<Int32>.stride,
                    shape: tensor.shape,
                    dtype: "I32"
                ))
            default:
                throw SmeltCompilerError.unsupportedConfiguration(
                    "tensor '\(tensor.runtimeName)' must be I32 or I64, got '\(tensor.dtype)'"
                )
            }
        }

        return (adjusted, ownedBuffers)
    }

    /// Down-cast a 1-D int64 index buffer to int32 with per-element
    /// bounds validation.
    private static func downcastInt64IndexBufferToInt32(
        name: String,
        source: UnsafeRawPointer,
        shape: [Int],
        upperBoundExclusive: Int
    ) throws -> OwnedCheckpointTensorBuffer {
        let elementCount = shape.reduce(1, *)
        let byteCount = elementCount * MemoryLayout<Int32>.stride
        let buffer = OwnedCheckpointTensorBuffer(byteCount: byteCount)
        let dst = buffer.pointer.bindMemory(to: Int32.self, capacity: elementCount)

        for idx in 0 ..< elementCount {
            var value: Int64 = 0
            memcpy(&value, source.advanced(by: idx * MemoryLayout<Int64>.stride), 8)
            guard value >= 0, value < Int64(upperBoundExclusive) else {
                throw SmeltCompilerError.unsupportedConfiguration(
                    "index buffer '\(name)' entry \(idx) = \(value) outside [0, \(upperBoundExclusive))"
                )
            }
            dst[idx] = Int32(value)
        }

        return buffer
    }

    static func shiftedNormWeightBufferForOnePlusWeightCompatibility(
        source: UnsafeRawPointer,
        shape: [Int],
        dtype: String
    ) throws -> OwnedCheckpointTensorBuffer {
        let elementCount = shape.reduce(1, *)
        let byteCount = elementCount * MemoryLayout<Float>.stride
        let buffer = OwnedCheckpointTensorBuffer(byteCount: byteCount)
        let dst = buffer.pointer.bindMemory(to: Float.self, capacity: elementCount)

        for idx in 0..<elementCount {
            dst[idx] = try decodeCheckpointFloat(
                source: source,
                index: idx,
                dtype: dtype
            ) - 1.0
        }

        return buffer
    }

    static func decodeCheckpointFloat(
        source: UnsafeRawPointer,
        index: Int,
        dtype: String
    ) throws -> Float {
        switch dtype {
        case "F32":
            var value: Float = 0
            memcpy(&value, source.advanced(by: index * 4), 4)
            return value
        case "F16":
            var bits: UInt16 = 0
            memcpy(&bits, source.advanced(by: index * 2), 2)
            return Float(Float16(bitPattern: bits))
        case "BF16":
            var bits: UInt16 = 0
            memcpy(&bits, source.advanced(by: index * 2), 2)
            return Float(bitPattern: UInt32(bits) << 16)
        default:
            throw SmeltCompilerError.unsupportedConfiguration(
                "unsupported dtype '\(dtype)' for norm compatibility transform"
            )
        }
    }

    /// Serialize a dispatch table to its on-disk `.bin` (a flat `SmeltDispatchRecord`
    /// array), returning the byte count for logging. The single serializer for the main
    /// build's decode/prefill/verify tables AND the Qwen3-TTS trunk sidecar, so the record
    /// layout can't drift between them.
    @discardableResult
    static func writeDispatchTable(_ records: [SmeltDispatchRecord], to path: String) throws -> Int {
        // SmeltDispatchRecord — and its SmeltBufferRecord sub-fields — carry interior
        // alignment padding (top-level bytes 1, 7, 10-11; the [4,8) gap in every
        // SmeltBufferRecord) that the memberwise initializers leave uninitialized.
        // `storeBytes(of: rec)` copies the whole struct verbatim, padding included, so
        // that stack garbage lands in dispatches.bin and the same package built twice
        // yields byte-different output (and a different manifest checksum). Instead we
        // build each record into the zero-filled `Data(count:)` buffer and write only
        // the meaningful field bytes in place, leaving every padding byte zero. The
        // on-disk layout is byte-for-byte identical for the meaningful bytes — the Metal
        // reader never reads padding as data — so this is a pure determinism fix, not an
        // ABI change.
        let stride = MemoryLayout<SmeltDispatchRecord>.stride
        var data = Data(count: records.count * stride)
        data.withUnsafeMutableBytes { raw in
            let table = raw.bindMemory(to: SmeltDispatchRecord.self)
            for (idx, rec) in records.enumerated() {
                copyDispatchRecordFields(rec, into: &table[idx])
            }
        }
        try data.write(to: URL(fileURLWithPath: path))
        return data.count
    }

    /// Copy every meaningful field of `src` (recursively into its buffer/constant
    /// sub-records) into an already zero-filled `dst`, touching only field bytes so the
    /// interior alignment padding stays zero. This keeps the serialized bytes
    /// deterministic across builds. Note: assigning a whole sub-record value
    /// (`dst.buf0 = src.buf0`) would copy that sub-record's padding too, so each leaf
    /// scalar is set individually.
    private static func copyDispatchRecordFields(
        _ src: SmeltDispatchRecord,
        into dst: inout SmeltDispatchRecord
    ) {
        dst.opKind = src.opKind
        dst.pipeline = src.pipeline
        dst.style = src.style
        dst.bufferCount = src.bufferCount
        dst.constantCount = src.constantCount
        dst.pad = src.pad
        dst.gridW = src.gridW
        dst.gridH = src.gridH
        dst.gridD = src.gridD
        dst.tgW = src.tgW
        dst.tgH = src.tgH
        dst.tgD = src.tgD
        copyBufferRecordFields(src.buf0, into: &dst.buf0)
        copyBufferRecordFields(src.buf1, into: &dst.buf1)
        copyBufferRecordFields(src.buf2, into: &dst.buf2)
        copyBufferRecordFields(src.buf3, into: &dst.buf3)
        copyBufferRecordFields(src.buf4, into: &dst.buf4)
        copyBufferRecordFields(src.buf5, into: &dst.buf5)
        copyBufferRecordFields(src.buf6, into: &dst.buf6)
        copyBufferRecordFields(src.buf7, into: &dst.buf7)
        copyBufferRecordFields(src.buf8, into: &dst.buf8)
        copyBufferRecordFields(src.buf9, into: &dst.buf9)
        copyBufferRecordFields(src.buf10, into: &dst.buf10)
        copyBufferRecordFields(src.buf11, into: &dst.buf11)
        copyBufferRecordFields(src.buf12, into: &dst.buf12)
        copyBufferRecordFields(src.buf13, into: &dst.buf13)
        copyBufferRecordFields(src.buf14, into: &dst.buf14)
        copyBufferRecordFields(src.buf15, into: &dst.buf15)
        copyConstantRecordFields(src.con0, into: &dst.con0)
        copyConstantRecordFields(src.con1, into: &dst.con1)
        copyConstantRecordFields(src.con2, into: &dst.con2)
        copyConstantRecordFields(src.con3, into: &dst.con3)
        copyConstantRecordFields(src.con4, into: &dst.con4)
        copyConstantRecordFields(src.con5, into: &dst.con5)
        copyConstantRecordFields(src.con6, into: &dst.con6)
        copyConstantRecordFields(src.con7, into: &dst.con7)
    }

    /// Padding: the [4,8) gap before `offset` stays zero because only field bytes
    /// (slot@0, bindingIndex@2, offsetKind@3, offset@8) are written.
    private static func copyBufferRecordFields(
        _ src: SmeltBufferRecord,
        into dst: inout SmeltBufferRecord
    ) {
        dst.slot = src.slot
        dst.bindingIndex = src.bindingIndex
        dst.offsetKind = src.offsetKind
        dst.offset = src.offset
    }

    /// SmeltConstantRecord has no implicit padding (its `pad` is an explicit field),
    /// but writing field-by-field keeps the serializer uniform.
    private static func copyConstantRecordFields(
        _ src: SmeltConstantRecord,
        into dst: inout SmeltConstantRecord
    ) {
        dst.kind = src.kind
        dst.bindingIndex = src.bindingIndex
        dst.pad = src.pad
        dst.value = src.value
    }

    // internal (not private): the Qwen3-TTS trunk sidecar reuses this to checksum its
    // co-resident artifacts (hashing the shared weights.bin/metallib symlink targets).
    static func computeManifestChecksums(
        pkgPath: String,
        weightsPath: String,
        metallibPath: String,
        generatedSwiftPath: String,
        dispatchesPath: String,
        prefillDispatchesPath: String?,
        prefillVerifyArgmaxDispatchesPath: String?
    ) throws -> SmeltManifestChecksums {
        let tokenizerPath = "\(pkgPath)/tokenizer.json"
        return SmeltManifestChecksums(
            weightsBin: try sha256Hex(ofFileAt: weightsPath),
            metallib: try sha256Hex(ofFileAt: metallibPath),
            generatedSwift: try sha256Hex(ofFileAt: generatedSwiftPath),
            dispatchesBin: try sha256Hex(ofFileAt: dispatchesPath),
            prefillDispatchesBin: try prefillDispatchesPath.map { try sha256Hex(ofFileAt: $0) },
            prefillVerifyArgmaxDispatchesBin: try prefillVerifyArgmaxDispatchesPath.map {
                try sha256Hex(ofFileAt: $0)
            },
            tokenizerJSON: FileManager.default.fileExists(atPath: tokenizerPath)
                ? try sha256Hex(ofFileAt: tokenizerPath)
                : nil
        )
    }

    private static func sha256Hex<T: Encodable>(ofJSON value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return sha256Hex(of: try encoder.encode(value))
    }

    private static func sha256Hex(of data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Deterministic fingerprint of injected GPTQ u4 blocks: hashes each block's
    /// name and packed bytes in name order, so different blocks rebuild instead
    /// of reusing stale weights. Each variable-length field is length-prefixed so
    /// the hash is injective — without delimiters, bytes could migrate across a
    /// field/block boundary and collide two distinct block sets onto one digest,
    /// silently reusing the wrong weights.
    static func gptqBlocksFingerprint(_ blocks: [String: SmeltAffineU4.Packed]) -> String {
        var hasher = SHA256()
        func updateLengthPrefixed(_ data: Data) {
            withUnsafeBytes(of: UInt64(data.count).littleEndian) { hasher.update(bufferPointer: $0) }
            hasher.update(data: data)
        }
        for name in blocks.keys.sorted() {
            let block = blocks[name]!
            updateLengthPrefixed(Data(name.utf8))
            updateLengthPrefixed(Data(block.nibbles))
            updateLengthPrefixed(block.scales.withUnsafeBytes { Data($0) })
            updateLengthPrefixed(block.biases.withUnsafeBytes { Data($0) })
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func sha256Hex(ofFileAt path: String) throws -> String {
        let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        defer { try? handle.close() }

        var hasher = SHA256()
        while true {
            let chunk = try handle.read(upToCount: 1 << 20) ?? Data()
            if chunk.isEmpty {
                break
            }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func sha256Hex(
        ofTreeRoots roots: [(path: String, allowedExtensions: Set<String>)],
        excludingPaths: Set<String> = []
    ) throws -> String {
        let fm = FileManager.default
        var hasher = SHA256()

        for (root, allowedExtensions) in roots.sorted(by: { $0.path < $1.path }) {
            guard let enumerator = fm.enumerator(atPath: root) else {
                throw SmeltCompilerError.noShaders("failed to enumerate \(root)")
            }
            var files: [String] = []
            for case let relPath as String in enumerator {
                let fullPath = "\(root)/\(relPath)"
                if excludingPaths.contains(fullPath) {
                    continue
                }
                var isDir: ObjCBool = false
                guard fm.fileExists(atPath: fullPath, isDirectory: &isDir), !isDir.boolValue else {
                    continue
                }
                if let ext = URL(fileURLWithPath: relPath).pathExtension.isEmpty
                    ? nil
                    : URL(fileURLWithPath: relPath).pathExtension,
                   !allowedExtensions.isEmpty,
                   !allowedExtensions.contains(ext)
                {
                    continue
                }
                files.append(relPath)
            }
            files.sort()

            hasher.update(data: Data(root.utf8))
            hasher.update(data: Data([0]))
            for relPath in files {
                hasher.update(data: Data(relPath.utf8))
                hasher.update(data: Data([0]))
                try updateHasher(&hasher, withFileAt: "\(root)/\(relPath)")
                hasher.update(data: Data([0]))
            }
        }

        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func updateHasher(_ hasher: inout SHA256, withFileAt path: String) throws {
        let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        defer { try? handle.close() }

        while true {
            let chunk = try handle.read(upToCount: 1 << 20) ?? Data()
            if chunk.isEmpty {
                break
            }
            hasher.update(data: chunk)
        }
    }

    /// Merge decode and prefill pipeline tables into one manifest array.
    /// Prefill records are remapped from their local pipeline indices into the
    /// unified manifest pipeline index space.
    private static func mergePipelineTables(
        baseNames: [String],
        extraNames: [String],
        extraRecords: [SmeltDispatchRecord]
    ) -> ([String], [SmeltDispatchRecord]) {
        var mergedNames = baseNames
        var nameToIndex = Dictionary(uniqueKeysWithValues: baseNames.enumerated().map { ($1, $0) })

        for name in extraNames where nameToIndex[name] == nil {
            nameToIndex[name] = mergedNames.count
            mergedNames.append(name)
        }

        let remappedRecords = extraRecords.map { record -> SmeltDispatchRecord in
            guard record.opKind == SmeltDispatchRecord.opDispatch else { return record }
            let localIndex = Int(record.pipeline)
            guard localIndex >= 0, localIndex < extraNames.count,
                  let mergedIndex = nameToIndex[extraNames[localIndex]]
            else {
                return record
            }
            var rec = record
            rec.pipeline = UInt16(mergedIndex)
            return rec
        }

        return (mergedNames, remappedRecords)
    }

}

// MARK: - Compiler errors

/// Errors from the Smelt compiler.
public enum SmeltCompilerError: Error, CustomStringConvertible {
    case noShaders(String)
    case metalCompileFailed(String)
    case metallibLinkFailed
    case unsupportedConfiguration(String)
    case unsupportedRuntimeFeature(String)

    public var description: String {
        switch self {
        case let .noShaders(dir):
            return "No .metal files found in \(dir)"
        case let .metalCompileFailed(file):
            return "Metal compilation failed for \(file)"
        case .metallibLinkFailed:
            return "metallib linking failed"
        case let .unsupportedConfiguration(detail):
            return "Unsupported compiler configuration: \(detail)"
        case let .unsupportedRuntimeFeature(detail):
            return "Feature not yet runtime-supported: \(detail)"
        }
    }
}
