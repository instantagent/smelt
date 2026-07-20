import Foundation
import Metal
import SmeltSchema
import SmeltRuntime
import SmeltCompiler
#if os(macOS)
import Darwin
#endif

private enum KernelLabError: Error, CustomStringConvertible {
    case missingMetallib(String)
    case missingDevice
    case missingCommandQueue
    case missingFunction(String)
    case missingPipeline(String)
    case missingBuffer(String)
    case missingSample(String)
    case missingShaderSource(String)
    case missingInsertionPoint(String)
    case missingPackagePath
    case missingPackagePipeline(String)
    case missingDispatchTable(String)
    case invalidPackage(String)
    case unsupportedPlatform(String)
    case metalCompileFailed(String)
    case unknownCase(String)

    var description: String {
        switch self {
        case .missingMetallib(let path):
            return "missing model.metallib at \(path)"
        case .missingDevice:
            return "Metal device unavailable"
        case .missingCommandQueue:
            return "Metal command queue unavailable"
        case .missingFunction(let name):
            return "kernel function not found in lab library: \(name)"
        case .missingPipeline(let name):
            return "pipeline not built: \(name)"
        case .missingBuffer(let name):
            return "buffer not allocated: \(name)"
        case .missingSample(let name):
            return "no timing samples collected for \(name)"
        case .missingShaderSource(let path):
            return "missing shader source at \(path)"
        case .missingInsertionPoint(let marker):
            return "could not insert lab declarations before marker: \(marker)"
        case .missingPackagePath:
            return "--library package requires a model.smeltpkg path"
        case .missingPackagePipeline(let name):
            return "package is missing required pipeline: \(name)"
        case .missingDispatchTable(let path):
            return "missing dispatch table at \(path)"
        case .invalidPackage(let detail):
            return "invalid package for kernel lab: \(detail)"
        case .unsupportedPlatform(let detail):
            return "unsupported platform for kernel lab: \(detail)"
        case .metalCompileFailed(let detail):
            return "source metallib compile failed: \(detail)"
        case .unknownCase(let name):
            return "unknown kernel-lab case: \(name)"
        }
    }
}

private enum KernelLabLibraryMode: String {
    case source
    case package
}

private enum KernelPackageReplayStateMode: String {
    case reset
    case primed
}

private enum KernelLabConstant {
    case uint32(UInt32)
    case float32(Float)

    func bind(to encoder: MTLComputeCommandEncoder, index: Int) {
        switch self {
        case .uint32(var value):
            encoder.setBytes(&value, length: MemoryLayout<UInt32>.stride, index: index)
        case .float32(var value):
            encoder.setBytes(&value, length: MemoryLayout<Float>.stride, index: index)
        }
    }
}

private enum KernelLabFunctionConstant {
    case uint32(UInt32)

    func bind(to values: MTLFunctionConstantValues, index: Int) {
        switch self {
        case .uint32(var value):
            values.setConstantValue(&value, type: .uint, index: index)
        }
    }
}

private enum KernelLabGrid {
    case threadgroups(width: Int, threadsPerThreadgroup: Int)
    case threadgroups2D(width: Int, height: Int, threadsPerThreadgroup: Int)
    case threads(width: Int, threadsPerThreadgroup: Int)
    case threads3D(
        width: Int,
        height: Int,
        depth: Int,
        threadsPerThreadgroupWidth: Int,
        threadsPerThreadgroupHeight: Int
    )
}

private struct KernelLabDispatch {
    let function: String
    let functionConstants: [Int: KernelLabFunctionConstant]
    let buffers: [Int: String]
    let constants: [Int: KernelLabConstant]
    let grid: KernelLabGrid

    init(
        function: String,
        functionConstants: [Int: KernelLabFunctionConstant] = [:],
        buffers: [Int: String],
        constants: [Int: KernelLabConstant],
        grid: KernelLabGrid
    ) {
        self.function = function
        self.functionConstants = functionConstants
        self.buffers = buffers
        self.constants = constants
        self.grid = grid
    }
}

private struct KernelLabVariant {
    let name: String
    let dispatches: [KernelLabDispatch]
}

private struct KernelLabCase {
    let name: String
    let buffers: [String: Int]
    let baseline: KernelLabVariant
    let candidate: KernelLabVariant
}

private struct KernelLabStats {
    let medianUs: Double
    let p95Us: Double
}

func runKernelLabCommand() {
    let usage = "Usage: smelt kernel-lab [model.smeltpkg] [--case all|package-replay|qwen08-*|vibe3b-gate-up-*] [--transform identity|unfuse-affine-residual-add|unfuse-signed-gate-up|specialize-attention-d256-h24-kv4|fuse-rope-kv-cache-update|fuse-kv-affine-pair|fuse-norm-scale-gate-up|fuse-lmhead-argmax|skip-matching-dispatches|substitute-pipeline] [--filter SUBSTRING] [--to-pipeline NAME] [--state reset|primed] [--library source|package] [--shader-dir DIR] [--iterations N] [--warmup N] [--position N]\n"
    if args.contains("--help") || args.contains("-h") {
        fputs(usage, stdout)
        exit(0)
    }
    guard args.count >= 2 else {
        fputs(usage, stderr)
        exit(1)
    }

    let packagePath = parseKernelLabPackagePath()
    let caseName = parseArg("--case", default: "all")
    let libraryMode = KernelLabLibraryMode(
        rawValue: parseArg("--library", default: "source")
    ) ?? .source
    let shaderDir = parseArg("--shader-dir", default: "Resources/Shaders")
    let iterations = max(1, Int(parseArg("--iterations", default: "100")) ?? 100)
    let warmup = max(0, Int(parseArg("--warmup", default: "10")) ?? 10)
    let position = Int32(parseArg("--position", default: "10")) ?? 10
    let transformName = parseArg("--transform", default: "identity")
    let filter = parseArg("--filter", default: "")
    let toPipeline = parseArg("--to-pipeline", default: "")
    let stateMode = KernelPackageReplayStateMode(
        rawValue: parseArg("--state", default: "reset")
    ) ?? .reset

    do {
        let construction = packagePath.map {
            requireCAMTextRuntimePlanOrExit(
                packagePath: $0,
                request: .kernelLabPackage,
                verb: "kernel-lab",
                requireAuthoredInventory: true
            )
        }
        if caseName == "package-replay" {
            guard let packagePath, let construction else {
                throw KernelLabError.missingPackagePath
            }
            let lab = try KernelPackageReplayLab(
                packagePath: packagePath,
                construction: construction,
                libraryMode: libraryMode,
                shaderDir: shaderDir
            )
            fputs(
                "Kernel lab: package=\(packagePath) replay=dispatches.bin transform=\(transformName) filter=\(filter.isEmpty ? "<all>" : filter) to=\(toPipeline.isEmpty ? "<none>" : toPipeline) state=\(stateMode.rawValue) iterations=\(iterations) warmup=\(warmup) position=\(position)\n\n",
                stderr
            )
            try lab.run(
                transformName: transformName,
                filter: filter,
                toPipeline: toPipeline,
                stateMode: stateMode,
                iterations: iterations,
                warmup: warmup,
                position: position
            )
            return
        }

        if libraryMode == .package && packagePath == nil {
            throw KernelLabError.missingPackagePath
        }
        let lab = try KernelLab(
            packagePath: packagePath,
            libraryMode: libraryMode,
            shaderDir: shaderDir
        )
        let cases = try labCases(named: caseName)
        fputs(
            "Kernel lab: package=\(packagePath ?? "<none>") library=\(libraryMode.rawValue) iterations=\(iterations) warmup=\(warmup)\n\n",
            stderr
        )
        for testCase in cases {
            try lab.run(testCase, iterations: iterations, warmup: warmup)
        }
    } catch {
        fputs("Kernel lab failed: \(error)\n", stderr)
        exit(1)
    }
}

private func parseKernelLabPackagePath() -> String? {
    let valueFlags: Set<String> = [
        "--case",
        "--library",
        "--shader-dir",
        "--iterations",
        "--warmup",
        "--position",
        "--transform",
        "--filter",
        "--state",
        "--to-pipeline",
    ]
    var index = 2
    while index < args.count {
        let token = args[index]
        if valueFlags.contains(token) {
            index += 2
            continue
        }
        if token.hasPrefix("--") {
            index += 1
            continue
        }
        return token
    }
    return nil
}

private struct KernelPackageReplayRewrite {
    let name: String
    let records: [SmeltDispatchRecord]
    let pipelineOverrides: [UInt16: MTLComputePipelineState]
    let transformedDispatches: Int
    let insertedDispatches: Int
}

private struct KernelPackageReplaySample {
    let token: Int32
    let totalUs: Double
    let pureGpuUs: Double
}

private struct KernelPackageReplayOutput {
    let token: Int32
    let logits: [UInt16]
}

private enum KernelPackageReplayVariant {
    case rewritten
    case original
    case packageTable
}

private struct KernelPackageReplayStats {
    let totalMedianUs: Double
    let totalP95Us: Double
    let pureGpuMedianUs: Double
    let pureGpuP95Us: Double
}

private final class KernelPackageReplayLab {
    private let packagePath: String
    private let libraryMode: KernelLabLibraryMode
    private let shaderDir: String
    private let manifest: SmeltManifest
    private let pipelineIndexByName: [String: UInt16]
    private let pipelineNames: [String]
    private let originalRecords: [SmeltDispatchRecord]
    private let runtime: SmeltRuntime
    private var sourceLibrary: MTLLibrary?
    private var sourceExtraFunctionNames: Set<String> = []
    private var sourcePipelineIndices: [String: UInt16] = [:]
    private var sourcePipelineOverrides: [UInt16: MTLComputePipelineState] = [:]

    init(
        packagePath: String,
        construction: CAMTextRuntimeConstruction,
        libraryMode: KernelLabLibraryMode,
        shaderDir: String
    ) throws {
        try construction.requirePackagePath(packagePath)
        self.packagePath = packagePath
        self.libraryMode = libraryMode
        self.shaderDir = shaderDir
        let manifestPath = URL(fileURLWithPath: packagePath)
            .appendingPathComponent("manifest.json")
        let manifestData = try Data(contentsOf: manifestPath)
        self.manifest = try SmeltManifest.decode(from: manifestData)
        self.pipelineNames = manifest.pipelines

        var indexByName: [String: UInt16] = [:]
        for (index, name) in manifest.pipelines.enumerated() {
            guard index <= Int(UInt16.max) else {
                throw KernelLabError.invalidPackage(
                    "pipeline index \(index) exceeds UInt16 dispatch table range"
                )
            }
            indexByName[name] = UInt16(index)
        }
        self.pipelineIndexByName = indexByName

        let dispatchPath = URL(fileURLWithPath: packagePath)
            .appendingPathComponent("dispatches.bin")
        self.originalRecords = try Self.readDispatchRecords(dispatchPath.path)
        self.runtime = try construction.makeRuntime(contextLimit: nil)
    }

    func run(
        transformName: String,
        filter: String,
        toPipeline: String,
        stateMode: KernelPackageReplayStateMode,
        iterations: Int,
        warmup: Int,
        position: Int32
    ) throws {
        let rewrite: KernelPackageReplayRewrite
        switch transformName {
        case "identity":
            rewrite = makeIdentityDispatchRecords()
        case "unfuse-affine-residual-add":
            rewrite = try makeUnfusedAffineResidualAddRecords(filter: filter)
        case "unfuse-signed-gate-up":
            rewrite = try makeUnfusedSignedGateUpRecords(filter: filter)
        case "specialize-attention-d256-h24-kv4":
            rewrite = try makeSpecializedAttentionD256H24KV4Records(filter: filter)
        case "fuse-rope-kv-cache-update":
            rewrite = try makeFusedRopeKVCacheUpdateRecords(filter: filter)
        case "fuse-kv-affine-pair":
            rewrite = try makeFusedKVAffinePairRecords(filter: filter)
        case "fuse-norm-scale-gate-up":
            rewrite = try makeFusedNormScaleGateUpRecords(filter: filter)
        case "fuse-lmhead-argmax":
            rewrite = try makeFusedLMHeadArgmaxRecords(filter: filter)
        case "skip-matching-dispatches":
            rewrite = try makeSkippedDispatchRecords(filter: filter)
        case "substitute-pipeline":
            rewrite = try makeSubstitutedPipelineRecords(
                filter: filter,
                toPipeline: toPipeline
            )
        default:
            throw KernelLabError.unknownCase("package-replay transform \(transformName)")
        }

        fputs("package-replay:\n", stderr)
        fputs("  transformed dispatches: \(rewrite.transformedDispatches)\n", stderr)
        fputs("  inserted dispatches:    \(rewrite.insertedDispatches)\n", stderr)
        fputs("  measurement order:     rotated\n", stderr)

        var baselineSamples: [KernelPackageReplaySample] = []
        var candidateSamples: [KernelPackageReplaySample] = []
        var packageTableSamples: [KernelPackageReplaySample] = []
        baselineSamples.reserveCapacity(iterations)
        candidateSamples.reserveCapacity(iterations)
        packageTableSamples.reserveCapacity(iterations)

        func runVariant(_ variant: KernelPackageReplayVariant) throws -> KernelPackageReplaySample {
            switch variant {
            case .rewritten:
                try runReplay(
                    records: rewrite.records,
                    position: position,
                    stateMode: stateMode,
                    pipelineOverrides: rewrite.pipelineOverrides
                )
            case .original:
                try runReplay(
                    records: originalRecords,
                    position: position,
                    stateMode: stateMode
                )
            case .packageTable:
                try runPackageTable(
                    position: position,
                    stateMode: stateMode
                )
            }
        }

        func captureOutput(
            _ variant: KernelPackageReplayVariant
        ) throws -> KernelPackageReplayOutput {
            let sample = try runVariant(variant)
            return KernelPackageReplayOutput(
                token: sample.token,
                logits: runtime.allLogitsHalf().map(\.bitPattern)
            )
        }

        func parityDescription(
            candidate: KernelPackageReplayOutput,
            reference: KernelPackageReplayOutput
        ) -> String {
            guard candidate.token == reference.token else {
                return "FAIL token candidate=\(candidate.token) reference=\(reference.token)"
            }
            guard candidate.logits.count == reference.logits.count else {
                return "FAIL logit count candidate=\(candidate.logits.count) reference=\(reference.logits.count)"
            }
            if let index = candidate.logits.indices.first(where: {
                candidate.logits[$0] != reference.logits[$0]
            }) {
                return "FAIL first fp16 logit \(index): candidate=0x\(String(candidate.logits[index], radix: 16)) reference=0x\(String(reference.logits[index], radix: 16))"
            }
            return "bit-exact token=\(candidate.token), fp16_logits=\(candidate.logits.count)"
        }

        let rewrittenOutput = try captureOutput(.rewritten)
        let originalOutput = try captureOutput(.original)
        let packageTableOutput = try captureOutput(.packageTable)
        fputs(
            "  output parity rewritten-vs-original: \(parityDescription(candidate: rewrittenOutput, reference: originalOutput))\n",
            stderr
        )
        fputs(
            "  output parity original-vs-package:  \(parityDescription(candidate: originalOutput, reference: packageTableOutput))\n",
            stderr
        )

        let variants: [KernelPackageReplayVariant] = [
            .rewritten,
            .original,
            .packageTable,
        ]
        func rotatedVariant(at offset: Int, slot: Int) -> KernelPackageReplayVariant {
            variants[(offset + slot) % variants.count]
        }

        for warmupIndex in 0..<warmup {
            for slot in variants.indices {
                _ = try runVariant(rotatedVariant(at: warmupIndex, slot: slot))
            }
        }

        for iteration in 0..<iterations {
            for slot in variants.indices {
                let variant = rotatedVariant(at: iteration, slot: slot)
                let sample = try runVariant(variant)
                switch variant {
                case .rewritten:
                    baselineSamples.append(sample)
                case .original:
                    candidateSamples.append(sample)
                case .packageTable:
                    packageTableSamples.append(sample)
                }
            }
        }

        let rewritten = try stats(name: rewrite.name, samples: baselineSamples)
        let original = try stats(name: "override-original", samples: candidateSamples)
        let packageTable = try stats(name: "package-table", samples: packageTableSamples)
        let rewrittenTotalDeltaUs = rewritten.totalMedianUs - original.totalMedianUs
        let rewrittenTotalDeltaPct = original.totalMedianUs == 0
            ? 0
            : rewrittenTotalDeltaUs / original.totalMedianUs * 100
        let rewrittenPureGpuDeltaUs = rewritten.pureGpuMedianUs - original.pureGpuMedianUs
        let rewrittenPureGpuDeltaPct = original.pureGpuMedianUs == 0
            ? 0
            : rewrittenPureGpuDeltaUs / original.pureGpuMedianUs * 100
        let rewriteTotalVerdict = rewrittenTotalDeltaUs < 0 ? "total faster" : "total slower"
        let rewritePureGpuVerdict = rewrittenPureGpuDeltaUs < 0 ? "pure GPU faster" : "pure GPU slower"
        let tableTotalDeltaUs = packageTable.totalMedianUs - original.totalMedianUs
        let tableTotalDeltaPct = original.totalMedianUs == 0
            ? 0
            : tableTotalDeltaUs / original.totalMedianUs * 100
        let tablePureGpuDeltaUs = packageTable.pureGpuMedianUs - original.pureGpuMedianUs
        let tablePureGpuDeltaPct = original.pureGpuMedianUs == 0
            ? 0
            : tablePureGpuDeltaUs / original.pureGpuMedianUs * 100

        fputs(
            "  rewritten \(rewrite.name): total \(formatUs(rewritten.totalMedianUs))us median, \(formatUs(rewritten.totalP95Us))us p95; pure GPU \(formatUs(rewritten.pureGpuMedianUs))us median, \(formatUs(rewritten.pureGpuP95Us))us p95\n",
            stderr
        )
        fputs(
            "  original  override-original: total \(formatUs(original.totalMedianUs))us median, \(formatUs(original.totalP95Us))us p95; pure GPU \(formatUs(original.pureGpuMedianUs))us median, \(formatUs(original.pureGpuP95Us))us p95\n",
            stderr
        )
        fputs(
            "  package   package-table:     total \(formatUs(packageTable.totalMedianUs))us median, \(formatUs(packageTable.totalP95Us))us p95; pure GPU \(formatUs(packageTable.pureGpuMedianUs))us median, \(formatUs(packageTable.pureGpuP95Us))us p95\n",
            stderr
        )
        fputs(
            "  net rewritten-vs-original: total \(signedUs(rewrittenTotalDeltaUs))us (\(signedPct(rewrittenTotalDeltaPct))%) \(rewriteTotalVerdict); pure GPU \(signedUs(rewrittenPureGpuDeltaUs))us (\(signedPct(rewrittenPureGpuDeltaPct))%) \(rewritePureGpuVerdict)\n\n",
            stderr
        )
        fputs(
            "  original-vs-package-table: total \(signedUs(tableTotalDeltaUs))us (\(signedPct(tableTotalDeltaPct))%); pure GPU \(signedUs(tablePureGpuDeltaUs))us (\(signedPct(tablePureGpuDeltaPct))%) package-table minus override-original\n\n",
            stderr
        )
    }

    private static func readDispatchRecords(_ path: String) throws -> [SmeltDispatchRecord] {
        guard FileManager.default.fileExists(atPath: path) else {
            throw KernelLabError.missingDispatchTable(path)
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let stride = MemoryLayout<SmeltDispatchRecord>.stride
        guard data.count % stride == 0 else {
            throw KernelLabError.invalidPackage(
                "dispatches.bin size \(data.count) is not a multiple of record stride \(stride)"
            )
        }
        var records = [SmeltDispatchRecord](
            repeating: .empty(),
            count: data.count / stride
        )
        _ = records.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst)
        }
        return records
    }

    private func makeIdentityDispatchRecords() -> KernelPackageReplayRewrite {
        KernelPackageReplayRewrite(
            name: "identity-replay",
            records: originalRecords,
            pipelineOverrides: [:],
            transformedDispatches: 0,
            insertedDispatches: 0
        )
    }

    private func runReplay(
        records: [SmeltDispatchRecord],
        position: Int32,
        stateMode: KernelPackageReplayStateMode,
        pipelineOverrides: [UInt16: MTLComputePipelineState] = [:]
    ) throws -> KernelPackageReplaySample {
        runtime.resetWorkingBuffers()
        if stateMode == .primed, position > 0 {
            for primePosition in 0..<position {
                _ = try runtime.profileDecodeStep(
                    tokenId: 0,
                    position: primePosition,
                    dispatchRecords: records,
                    pipelineOverrides: pipelineOverrides
                )
            }
        }
        let result = try runtime.profileDecodeStep(
            tokenId: 0,
            position: position,
            dispatchRecords: records,
            pipelineOverrides: pipelineOverrides
        )
        return KernelPackageReplaySample(
            token: result.token,
            totalUs: (result.cpuMs + result.gpuMs + result.readMs) * 1_000,
            pureGpuUs: result.pureGpuMs * 1_000
        )
    }

    private func runPackageTable(
        position: Int32,
        stateMode: KernelPackageReplayStateMode
    ) throws -> KernelPackageReplaySample {
        runtime.resetWorkingBuffers()
        if stateMode == .primed, position > 0 {
            for primePosition in 0..<position {
                _ = try runtime.decodeStep(tokenId: 0, position: primePosition)
            }
        }
        let result = try runtime.profileDecodeStep(tokenId: 0, position: position)
        return KernelPackageReplaySample(
            token: result.token,
            totalUs: (result.cpuMs + result.gpuMs + result.readMs) * 1_000,
            pureGpuUs: result.pureGpuMs * 1_000
        )
    }

    private func makeUnfusedAffineResidualAddRecords(
        filter: String
    ) throws -> KernelPackageReplayRewrite {
        guard let elementwiseAddPipeline = pipelineIndexByName["elementwise_add"] else {
            throw KernelLabError.missingPackagePipeline("elementwise_add")
        }

        var result: [SmeltDispatchRecord] = []
        result.reserveCapacity(originalRecords.count)
        var transformed = 0
        var inserted = 0

        for record in originalRecords {
            guard record.opKind == SmeltDispatchRecord.opDispatch,
                  Int(record.pipeline) < pipelineNames.count
            else {
                result.append(record)
                continue
            }

            let fusedName = pipelineNames[Int(record.pipeline)]
            guard matchesFilter(fusedName, filter: filter),
                  let affineName = unfusedAffinePipelineName(from: fusedName)
            else {
                result.append(record)
                continue
            }
            guard let affinePipeline = pipelineIndexByName[affineName] else {
                throw KernelLabError.missingPackagePipeline(affineName)
            }
            guard let rows = rowCount(from: fusedName, record: record) else {
                throw KernelLabError.invalidPackage(
                    "could not infer row count for \(fusedName)"
                )
            }

            let pair = try unfuseAffineResidualAdd(
                record,
                affinePipeline: affinePipeline,
                elementwiseAddPipeline: elementwiseAddPipeline,
                rows: rows
            )
            result.append(pair.affine)
            result.append(pair.add)
            transformed += 1
            inserted += 1
        }

        guard transformed > 0 else {
            throw KernelLabError.invalidPackage(
                "transform matched no fused affine residual-add dispatches"
                    + (filter.isEmpty ? "" : " for filter '\(filter)'")
            )
        }

        return KernelPackageReplayRewrite(
            name: "rewritten-unfused",
            records: result,
            pipelineOverrides: [:],
            transformedDispatches: transformed,
            insertedDispatches: inserted
        )
    }

    private func makeUnfusedSignedGateUpRecords(
        filter: String
    ) throws -> KernelPackageReplayRewrite {
        let fusedName = "signed_binary_gate_up_swiglu_g128_rows8"
        guard let matvecPipeline = pipelineIndexByName["signed_binary_matvec_g128_rows8"] else {
            throw KernelLabError.missingPackagePipeline(
                "signed_binary_matvec_g128_rows8"
            )
        }
        guard let swigluPipeline = pipelineIndexByName["swiglu_fused"] else {
            throw KernelLabError.missingPackagePipeline("swiglu_fused")
        }
        let gateScratch = try scratchBuffer(named: "ffnGateBuf")
        let upScratch = try scratchBuffer(named: "ffnUpBuf")

        var result: [SmeltDispatchRecord] = []
        result.reserveCapacity(originalRecords.count)
        var transformed = 0
        var inserted = 0

        for record in originalRecords {
            guard record.opKind == SmeltDispatchRecord.opDispatch,
                  Int(record.pipeline) < pipelineNames.count
            else {
                result.append(record)
                continue
            }

            let pipelineName = pipelineNames[Int(record.pipeline)]
            guard pipelineName == fusedName,
                  matchesFilter(pipelineName, filter: filter)
            else {
                result.append(record)
                continue
            }

            let staged = try unfuseSignedGateUp(
                record,
                matvecPipeline: matvecPipeline,
                swigluPipeline: swigluPipeline,
                gateScratch: gateScratch,
                upScratch: upScratch
            )
            result.append(staged.gate)
            result.append(staged.up)
            result.append(staged.swiglu)
            transformed += 1
            inserted += 2
        }

        guard transformed > 0 else {
            throw KernelLabError.invalidPackage(
                "transform matched no \(fusedName) dispatches"
                    + (filter.isEmpty ? "" : " for filter '\(filter)'")
            )
        }

        return KernelPackageReplayRewrite(
            name: "rewritten-unfused-signed-gate-up",
            records: result,
            pipelineOverrides: [:],
            transformedDispatches: transformed,
            insertedDispatches: inserted
        )
    }

    private func makeSpecializedAttentionD256H24KV4Records(
        filter: String
    ) throws -> KernelPackageReplayRewrite {
        let genericName = "attention_decode"
        let specializedName = "attention_decode_d256_h24_kv4"
        guard let genericPipeline = pipelineIndexByName[genericName] else {
            throw KernelLabError.missingPackagePipeline(genericName)
        }
        let specializedPipeline = try pipelineIndexByName[specializedName]
            ?? sourceOverlayPipelineIndex(named: specializedName)
        let specializedOverride = sourcePipelineOverrides[specializedPipeline]

        var result: [SmeltDispatchRecord] = []
        result.reserveCapacity(originalRecords.count)
        var transformed = 0

        for record in originalRecords {
            guard record.opKind == SmeltDispatchRecord.opDispatch,
                  Int(record.pipeline) < pipelineNames.count
            else {
                result.append(record)
                continue
            }

            let pipelineName = pipelineNames[Int(record.pipeline)]
            guard pipelineName == genericName,
                  matchesFilter(pipelineName, filter: filter),
                  record.gridW == 24,
                  let headDim = constant(binding: 5, in: record),
                  let cacheSeqCapacity = constant(binding: 6, in: record),
                  let seqLen = constant(binding: 7, in: record),
                  let kvHeads = constant(binding: 8, in: record),
                  let scale = constant(binding: 9, in: record),
                  let slidingWindow = constant(binding: 10, in: record),
                  headDim.kind == SmeltConstantRecord.kindLiteralU32,
                  headDim.value == 256,
                  kvHeads.kind == SmeltConstantRecord.kindLiteralU32,
                  kvHeads.value == 4,
                  scale.kind == SmeltConstantRecord.kindLiteralF32,
                  abs(Float(bitPattern: scale.value) - 0.0625) < 0.0001,
                  slidingWindow.kind == SmeltConstantRecord.kindLiteralU32,
                  slidingWindow.value == 0,
                  let query = buffer(binding: 0, in: record),
                  let keyCache = buffer(binding: 1, in: record),
                  let valueCache = buffer(binding: 2, in: record),
                  let output = buffer(binding: 4, in: record),
                  sameBufferLocation(query, output)
            else {
                result.append(record)
                continue
            }

            var specialized = record
            // Package replay validates pipeline indices against the authored
            // manifest. Keep the generic package slot and override only its
            // pipeline state when the specialization comes from source.
            specialized.pipeline = specializedOverride == nil
                ? specializedPipeline
                : genericPipeline
            specialized.bufferCount = 0
            specialized.constantCount = 0
            specialized.tgW = 64
            try appendBuffer(rebinding(query, to: 0), to: &specialized)
            try appendBuffer(rebinding(keyCache, to: 1), to: &specialized)
            try appendBuffer(rebinding(valueCache, to: 2), to: &specialized)
            try appendConstant(rebinding(seqLen, to: 3), to: &specialized)
            try appendConstant(rebinding(cacheSeqCapacity, to: 4), to: &specialized)
            result.append(specialized)
            transformed += 1
        }

        guard transformed > 0 else {
            throw KernelLabError.invalidPackage(
                "transform matched no D256 H24 KV4 in-place attention dispatches"
                    + (filter.isEmpty ? "" : " for filter '\(filter)'")
            )
        }

        return KernelPackageReplayRewrite(
            name: "rewritten-attention-d256-h24-kv4",
            records: result,
            pipelineOverrides: specializedOverride.map { [genericPipeline: $0] } ?? [:],
            transformedDispatches: transformed,
            insertedDispatches: 0
        )
    }

    private func makeFusedRopeKVCacheUpdateRecords(
        filter: String
    ) throws -> KernelPackageReplayRewrite {
        let fusedPipeline = try pipelineIndexByName["rope_kv_cache_update"]
            ?? sourceOverlayPipelineIndex(named: "rope_kv_cache_update")

        var result: [SmeltDispatchRecord] = []
        result.reserveCapacity(originalRecords.count)
        var transformed = 0
        var index = 0

        while index < originalRecords.count {
            let rope = originalRecords[index]
            guard index + 1 < originalRecords.count,
                  rope.opKind == SmeltDispatchRecord.opDispatch,
                  Int(rope.pipeline) < pipelineNames.count
            else {
                result.append(rope)
                index += 1
                continue
            }

            let cache = originalRecords[index + 1]
            guard cache.opKind == SmeltDispatchRecord.opDispatch,
                  Int(cache.pipeline) < pipelineNames.count
            else {
                result.append(rope)
                index += 1
                continue
            }

            let ropeName = pipelineNames[Int(rope.pipeline)]
            let cacheName = pipelineNames[Int(cache.pipeline)]
            let pairName = "\(ropeName)->\(cacheName)"
            guard ropeName == "apply_rope",
                  cacheName == "kv_cache_update",
                  matchesFilter(pairName, filter: filter)
                    || matchesFilter(ropeName, filter: filter)
                    || matchesFilter(cacheName, filter: filter)
            else {
                result.append(rope)
                index += 1
                continue
            }

            let fused = try fuseRopeKVCacheUpdate(
                rope: rope,
                cache: cache,
                fusedPipeline: fusedPipeline
            )
            result.append(fused)
            transformed += 1
            index += 2
        }

        guard transformed > 0 else {
            throw KernelLabError.invalidPackage(
                "transform matched no apply_rope->kv_cache_update dispatch pairs"
                    + (filter.isEmpty ? "" : " for filter '\(filter)'")
            )
        }

        return KernelPackageReplayRewrite(
            name: "rewritten-rope-kv-cache-update",
            records: result,
            pipelineOverrides: sourcePipelineOverrides,
            transformedDispatches: transformed,
            insertedDispatches: -transformed
        )
    }

    private func makeFusedKVAffinePairRecords(
        filter: String
    ) throws -> KernelPackageReplayRewrite {
        let fusedName = "fused_dual_affine_matvec_add_c2048_r256_g64_rows4"
        let fusedPipeline = try pipelineIndexByName[fusedName]
            ?? sourceOverlayPipelineIndex(named: fusedName)
        let kvPipelineName = "fused_affine_matvec_add_c2048_r256_g64_rows4"

        var result: [SmeltDispatchRecord] = []
        result.reserveCapacity(originalRecords.count)
        var transformed = 0
        var index = 0

        while index < originalRecords.count {
            let k = originalRecords[index]
            guard index + 1 < originalRecords.count,
                  k.opKind == SmeltDispatchRecord.opDispatch,
                  Int(k.pipeline) < pipelineNames.count
            else {
                result.append(k)
                index += 1
                continue
            }

            let v = originalRecords[index + 1]
            guard v.opKind == SmeltDispatchRecord.opDispatch,
                  Int(v.pipeline) < pipelineNames.count
            else {
                result.append(k)
                index += 1
                continue
            }

            let kName = pipelineNames[Int(k.pipeline)]
            let vName = pipelineNames[Int(v.pipeline)]
            let pairName = "\(kName)->\(vName)"
            guard kName == kvPipelineName,
                  vName == kvPipelineName,
                  matchesFilter(pairName, filter: filter)
                    || matchesFilter(fusedName, filter: filter)
                    || matchesFilter(kName, filter: filter)
            else {
                result.append(k)
                index += 1
                continue
            }

            let fused = try fuseKVAffinePair(k: k, v: v, fusedPipeline: fusedPipeline)
            result.append(fused)
            transformed += 1
            index += 2
        }

        guard transformed > 0 else {
            throw KernelLabError.invalidPackage(
                "transform matched no adjacent K/V affine pairs"
                    + (filter.isEmpty ? "" : " for filter '\(filter)'")
            )
        }

        return KernelPackageReplayRewrite(
            name: "rewritten-kv-affine-pair",
            records: result,
            pipelineOverrides: sourcePipelineOverrides,
            transformedDispatches: transformed,
            insertedDispatches: -transformed
        )
    }

    private func makeFusedNormScaleGateUpRecords(
        filter: String
    ) throws -> KernelPackageReplayRewrite {
        let scalePipeline = try pipelineIndexByName["rms_norm_scale_only"]
            ?? sourceOverlayPipelineIndex(named: "rms_norm_scale_only")
        let scaledPipelineName = "norm_scale_affine_gate_up_swiglu:2048:64"
        let scaledPipeline = try pipelineIndexByName[scaledPipelineName]
            ?? sourceOverlayPipelineIndex(named: scaledPipelineName)

        var result: [SmeltDispatchRecord] = []
        result.reserveCapacity(originalRecords.count)
        var transformed = 0
        var index = 0

        while index < originalRecords.count {
            let norm = originalRecords[index]
            guard index + 1 < originalRecords.count,
                  norm.opKind == SmeltDispatchRecord.opDispatch,
                  Int(norm.pipeline) < pipelineNames.count
            else {
                result.append(norm)
                index += 1
                continue
            }

            let gateUp = originalRecords[index + 1]
            guard gateUp.opKind == SmeltDispatchRecord.opDispatch,
                  Int(gateUp.pipeline) < pipelineNames.count
            else {
                result.append(norm)
                index += 1
                continue
            }

            let normName = pipelineNames[Int(norm.pipeline)]
            let gateUpName = pipelineNames[Int(gateUp.pipeline)]
            let pairName = "\(normName)->\(gateUpName)"
            guard normName == "rms_norm_1pw_d2048",
                  gateUpName == "fused_affine_gate_up_swiglu_c2048_r11008_g64",
                  matchesFilter(pairName, filter: filter)
                    || matchesFilter(normName, filter: filter)
                    || matchesFilter(gateUpName, filter: filter)
            else {
                result.append(norm)
                index += 1
                continue
            }

            let pair = try fuseNormScaleGateUp(
                norm: norm,
                gateUp: gateUp,
                scalePipeline: scalePipeline,
                scaledPipeline: scaledPipeline
            )
            result.append(pair.scale)
            result.append(pair.scaledGateUp)
            transformed += 1
            index += 2
        }

        guard transformed > 0 else {
            throw KernelLabError.invalidPackage(
                "transform matched no rms_norm_1pw_d2048->fused_affine_gate_up_swiglu_c2048_r11008_g64 dispatch pairs"
                    + (filter.isEmpty ? "" : " for filter '\(filter)'")
            )
        }

        return KernelPackageReplayRewrite(
            name: "rewritten-norm-scale-gate-up",
            records: result,
            pipelineOverrides: sourcePipelineOverrides,
            transformedDispatches: transformed,
            insertedDispatches: 0
        )
    }

    private func makeFusedLMHeadArgmaxRecords(
        filter: String
    ) throws -> KernelPackageReplayRewrite {
        var result: [SmeltDispatchRecord] = []
        result.reserveCapacity(originalRecords.count)
        var transformed = 0
        var index = 0

        while index < originalRecords.count {
            let lmHead = originalRecords[index]
            guard index + 2 < originalRecords.count,
                  lmHead.opKind == SmeltDispatchRecord.opDispatch,
                  Int(lmHead.pipeline) < pipelineNames.count
            else {
                result.append(lmHead)
                index += 1
                continue
            }

            let partials = originalRecords[index + 1]
            let reduce = originalRecords[index + 2]
            guard partials.opKind == SmeltDispatchRecord.opDispatch,
                  reduce.opKind == SmeltDispatchRecord.opDispatch,
                  Int(partials.pipeline) < pipelineNames.count,
                  Int(reduce.pipeline) < pipelineNames.count
            else {
                result.append(lmHead)
                index += 1
                continue
            }

            let lmHeadName = pipelineNames[Int(lmHead.pipeline)]
            let partialsName = pipelineNames[Int(partials.pipeline)]
            let reduceName = pipelineNames[Int(reduce.pipeline)]
            let trioName = "\(lmHeadName)->\(partialsName)->\(reduceName)"
            guard let shape = lmHeadShape(from: lmHeadName),
                  partialsName == "argmax_fp16_partials",
                  reduceName == "argmax_key_reduce",
                  matchesFilter(trioName, filter: filter)
                    || matchesFilter(lmHeadName, filter: filter)
                    || matchesFilter(partialsName, filter: filter)
                    || matchesFilter(reduceName, filter: filter)
            else {
                result.append(lmHead)
                index += 1
                continue
            }

            let fusedName = "affine_matvec_argmax_c\(shape.cols)_r\(shape.rows)_g\(shape.groupSize)_b1"
            let fusedPipeline = try pipelineIndexByName[fusedName]
                ?? sourceOverlayPipelineIndex(named: fusedName)
            let reduceFusedName = "lm_head_argmax_reduce_r\(shape.rows)"
            let reducePipeline = try pipelineIndexByName[reduceFusedName]
                ?? sourceOverlayPipelineIndex(named: reduceFusedName)

            let pair = try fuseLMHeadArgmax(
                lmHead: lmHead,
                partials: partials,
                reduce: reduce,
                fusedPipeline: fusedPipeline,
                reducePipeline: reducePipeline,
                shape: shape
            )
            result.append(pair.fused)
            result.append(pair.reduce)
            transformed += 1
            index += 3
        }

        guard transformed > 0 else {
            throw KernelLabError.invalidPackage(
                "transform matched no lm_head->argmax_fp16_partials->argmax_key_reduce dispatch trios"
                    + (filter.isEmpty ? "" : " for filter '\(filter)'")
            )
        }

        return KernelPackageReplayRewrite(
            name: "rewritten-lmhead-argmax",
            records: result,
            pipelineOverrides: sourcePipelineOverrides,
            transformedDispatches: transformed,
            insertedDispatches: -transformed
        )
    }

    private func makeSkippedDispatchRecords(
        filter: String
    ) throws -> KernelPackageReplayRewrite {
        guard !filter.isEmpty else {
            throw KernelLabError.invalidPackage(
                "skip-matching-dispatches requires --filter"
            )
        }

        var result: [SmeltDispatchRecord] = []
        result.reserveCapacity(originalRecords.count)
        var transformed = 0

        for record in originalRecords {
            guard record.opKind == SmeltDispatchRecord.opDispatch,
                  Int(record.pipeline) < pipelineNames.count
            else {
                result.append(record)
                continue
            }

            let pipelineName = pipelineNames[Int(record.pipeline)]
            if matchesFilter(pipelineName, filter: filter) {
                transformed += 1
                continue
            }
            result.append(record)
        }

        guard transformed > 0 else {
            throw KernelLabError.invalidPackage(
                "transform matched no dispatches for filter '\(filter)'"
            )
        }

        return KernelPackageReplayRewrite(
            name: "skipped-dispatches",
            records: result,
            pipelineOverrides: [:],
            transformedDispatches: transformed,
            insertedDispatches: 0
        )
    }

    private func makeSubstitutedPipelineRecords(
        filter: String,
        toPipeline: String
    ) throws -> KernelPackageReplayRewrite {
        guard !filter.isEmpty else {
            throw KernelLabError.invalidPackage(
                "substitute-pipeline requires --filter"
            )
        }
        guard !toPipeline.isEmpty else {
            throw KernelLabError.invalidPackage(
                "substitute-pipeline requires --to-pipeline"
            )
        }
        let packagedTargetPipeline: UInt16?
        var sourceTargetPipeline: MTLComputePipelineState?
        switch libraryMode {
        case .package:
            packagedTargetPipeline = pipelineIndexByName[toPipeline]
        case .source:
            // Source mode must exercise the current authored shader even when
            // the checked package already contains an entry point with the
            // same name. Otherwise an in-place kernel experiment silently
            // reuses the packaged PSO and reports a false identity result.
            packagedTargetPipeline = nil
            let overlayIndex = try sourceOverlayPipelineIndex(named: toPipeline)
            sourceTargetPipeline = sourcePipelineOverrides[overlayIndex]
        }

        var result: [SmeltDispatchRecord] = []
        result.reserveCapacity(originalRecords.count)
        var pipelineOverrides: [UInt16: MTLComputePipelineState] = [:]
        var transformed = 0

        for record in originalRecords {
            guard record.opKind == SmeltDispatchRecord.opDispatch,
                  Int(record.pipeline) < pipelineNames.count
            else {
                result.append(record)
                continue
            }

            let sourceName = pipelineNames[Int(record.pipeline)]
            guard matchesFilter(sourceName, filter: filter) else {
                result.append(record)
                continue
            }

            var substituted = record
            if let packagedTargetPipeline {
                substituted.pipeline = packagedTargetPipeline
            } else if let sourceTargetPipeline {
                // Runtime validation intentionally rejects dispatch indices
                // beyond the package manifest. Reuse the matched package slot
                // and override only its pipeline state in this replay.
                pipelineOverrides[record.pipeline] = sourceTargetPipeline
            } else {
                throw KernelLabError.missingPipeline(toPipeline)
            }
            if let rows = rowCount(from: sourceName, record: record),
               let targetTile = rowTile(from: toPipeline)
            {
                substituted.gridW = UInt32((rows + targetTile - 1) / targetTile)
            }
            if let targetThreadgroupWidth = threadgroupWidth(from: toPipeline) {
                substituted.tgW = UInt32(targetThreadgroupWidth)
            }
            result.append(substituted)
            transformed += 1
        }

        guard transformed > 0 else {
            throw KernelLabError.invalidPackage(
                "transform matched no dispatches for filter '\(filter)'"
            )
        }

        return KernelPackageReplayRewrite(
            name: "substituted-pipeline",
            records: result,
            pipelineOverrides: pipelineOverrides,
            transformedDispatches: transformed,
            insertedDispatches: 0
        )
    }

    private func sourceOverlayPipelineIndex(named functionName: String) throws -> UInt16 {
        guard libraryMode == .source else {
            throw KernelLabError.missingPackagePipeline(functionName)
        }
        if let existing = sourcePipelineIndices[functionName] {
            return existing
        }
        if sourceLibrary == nil {
            sourceExtraFunctionNames.insert(functionName)
            sourceLibrary = try KernelLab.makeSourceLibrary(
                device: runtime.metalDevice,
                shaderDir: shaderDir,
                extraFunctionNames: sourceExtraFunctionNames
            )
        }
        var function: MTLFunction?
        if let parsed = parseFunctionConstantPipelineName(functionName) {
            let constants = MTLFunctionConstantValues()
            var cols = parsed.cols
            var groupSize = parsed.groupSize
            constants.setConstantValue(&cols, type: .uint, index: 0)
            constants.setConstantValue(&groupSize, type: .uint, index: 1)
            function = try sourceLibrary?.makeFunction(
                name: parsed.name,
                constantValues: constants
            )
        } else {
            function = sourceLibrary?.makeFunction(name: functionName)
        }
        if function == nil,
           KernelLab.labDeclaration(for: functionName) != nil,
           !sourceExtraFunctionNames.contains(functionName)
        {
            sourceExtraFunctionNames.insert(functionName)
            sourceLibrary = try KernelLab.makeSourceLibrary(
                device: runtime.metalDevice,
                shaderDir: shaderDir,
                extraFunctionNames: sourceExtraFunctionNames
            )
            function = sourceLibrary?.makeFunction(name: functionName)
        }
        guard let function else {
            throw KernelLabError.missingFunction(functionName)
        }
        let nextIndex = pipelineNames.count + sourcePipelineIndices.count
        guard nextIndex <= Int(UInt16.max) else {
            throw KernelLabError.invalidPackage(
                "source overlay pipeline index \(nextIndex) exceeds UInt16 dispatch range"
            )
        }
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.label = "kernel-lab.source-overlay.\(functionName)"
        descriptor.computeFunction = function
        let pipeline = try runtime.metalDevice.makeComputePipelineState(
            descriptor: descriptor,
            options: [],
            reflection: nil
        )
        let index = UInt16(nextIndex)
        sourcePipelineIndices[functionName] = index
        sourcePipelineOverrides[index] = pipeline
        return index
    }

    private func parseFunctionConstantPipelineName(
        _ pipelineName: String
    ) -> (name: String, cols: UInt32, groupSize: UInt32)? {
        let parts = pipelineName.split(separator: ":")
        guard parts.count == 3,
              let cols = UInt32(parts[1]),
              let groupSize = UInt32(parts[2])
        else {
            return nil
        }
        return (String(parts[0]), cols, groupSize)
    }

    private func scratchBuffer(named name: String) throws -> SmeltBufferRecord {
        guard let slot = manifest.buffers.slots.first(where: { $0.name == name }) else {
            throw KernelLabError.invalidPackage(
                "package has no '\(name)' activation buffer for staged replay"
            )
        }
        guard slot.category == .activation, slot.dtype == .fp16 else {
            throw KernelLabError.invalidPackage(
                "package buffer '\(name)' must be an fp16 activation"
            )
        }
        guard slot.index >= 0, slot.index <= Int(Int16.max) else {
            throw KernelLabError.invalidPackage(
                "package buffer '\(name)' index \(slot.index) exceeds dispatch range"
            )
        }
        var buffer = SmeltBufferRecord.empty()
        buffer.slot = Int16(slot.index)
        return buffer
    }

    private func unfuseSignedGateUp(
        _ fused: SmeltDispatchRecord,
        matvecPipeline: UInt16,
        swigluPipeline: UInt16,
        gateScratch: SmeltBufferRecord,
        upScratch: SmeltBufferRecord
    ) throws -> (gate: SmeltDispatchRecord, up: SmeltDispatchRecord, swiglu: SmeltDispatchRecord) {
        guard let gateCodes = buffer(binding: 0, in: fused),
              let gateScales = buffer(binding: 1, in: fused),
              let upCodes = buffer(binding: 2, in: fused),
              let upScales = buffer(binding: 3, in: fused),
              let input = buffer(binding: 4, in: fused),
              let output = buffer(binding: 5, in: fused),
              let rows = constant(binding: 6, in: fused),
              let cols = constant(binding: 7, in: fused),
              rows.kind == SmeltConstantRecord.kindLiteralU32,
              cols.kind == SmeltConstantRecord.kindLiteralU32,
              rows.value > 0,
              cols.value > 0
        else {
            throw KernelLabError.invalidPackage(
                "signed gate/up record is missing literal shape or buffer bindings"
            )
        }

        let activationBytes = UInt64(rows.value) * UInt64(MemoryLayout<Float16>.stride)
        for scratch in [gateScratch, upScratch] {
            guard let slot = manifest.buffers.slots.first(where: {
                $0.index == Int(scratch.slot)
            }), UInt64(slot.sizeBytes) >= activationBytes else {
                throw KernelLabError.invalidPackage(
                    "signed gate/up staged replay scratch is smaller than \(activationBytes) bytes"
                )
            }
        }

        var gate = fused
        gate.pipeline = matvecPipeline
        gate.bufferCount = 0
        gate.constantCount = 0
        try appendBuffer(rebinding(gateCodes, to: 0), to: &gate)
        try appendBuffer(rebinding(gateScales, to: 1), to: &gate)
        try appendBuffer(rebinding(input, to: 2), to: &gate)
        try appendBuffer(rebinding(gateScratch, to: 3), to: &gate)
        try appendConstant(rebinding(rows, to: 4), to: &gate)
        try appendConstant(rebinding(cols, to: 5), to: &gate)

        var up = fused
        up.pipeline = matvecPipeline
        up.bufferCount = 0
        up.constantCount = 0
        try appendBuffer(rebinding(upCodes, to: 0), to: &up)
        try appendBuffer(rebinding(upScales, to: 1), to: &up)
        try appendBuffer(rebinding(input, to: 2), to: &up)
        try appendBuffer(rebinding(upScratch, to: 3), to: &up)
        try appendConstant(rebinding(rows, to: 4), to: &up)
        try appendConstant(rebinding(cols, to: 5), to: &up)

        var swiglu = SmeltDispatchRecord.empty()
        swiglu.opKind = SmeltDispatchRecord.opDispatch
        swiglu.pipeline = swigluPipeline
        swiglu.dispatchStyle = SmeltDispatchRecord.styleThreads
        swiglu.minSeqLen = fused.minSeqLen
        swiglu.gridW = rows.value
        swiglu.gridH = 1
        swiglu.gridD = 1
        swiglu.tgW = min(rows.value, 1_024)
        swiglu.tgH = 1
        swiglu.tgD = 1
        try appendBuffer(rebinding(gateScratch, to: 0), to: &swiglu)
        try appendBuffer(rebinding(upScratch, to: 1), to: &swiglu)
        try appendBuffer(rebinding(output, to: 2), to: &swiglu)
        try appendConstant(rebinding(rows, to: 3), to: &swiglu)

        return (gate, up, swiglu)
    }

    private func unfuseAffineResidualAdd(
        _ fused: SmeltDispatchRecord,
        affinePipeline: UInt16,
        elementwiseAddPipeline: UInt16,
        rows: Int
    ) throws -> (affine: SmeltDispatchRecord, add: SmeltDispatchRecord) {
        guard let weights = buffer(binding: 0, in: fused),
              let scales = buffer(binding: 1, in: fused),
              let biases = buffer(binding: 2, in: fused),
              let input = buffer(binding: 3, in: fused),
              let output = buffer(binding: 4, in: fused),
              let residual = buffer(binding: 5, in: fused)
        else {
            throw KernelLabError.invalidPackage(
                "fused affine residual-add record is missing expected bindings"
            )
        }
        if output.slot == residual.slot && output.offset == residual.offset {
            throw KernelLabError.invalidPackage(
                "cannot unfuse residual-add in place when output and residual share the same slot"
            )
        }

        let skipConstants = try skipOnlyConstants(in: fused)

        var affine = fused
        affine.pipeline = affinePipeline
        affine.bufferCount = 0
        affine.constantCount = 0
        try appendBuffer(weights, to: &affine)
        try appendBuffer(scales, to: &affine)
        try appendBuffer(biases, to: &affine)
        try appendBuffer(input, to: &affine)
        try appendBuffer(output, to: &affine)
        try appendConstants(skipConstants, to: &affine)

        var add = SmeltDispatchRecord.empty()
        add.opKind = SmeltDispatchRecord.opDispatch
        add.pipeline = elementwiseAddPipeline
        add.dispatchStyle = SmeltDispatchRecord.styleThreads
        add.minSeqLen = fused.minSeqLen
        add.gridW = UInt32(rows)
        add.gridH = 1
        add.gridD = 1
        add.tgW = UInt32(min(rows, 1_024))
        add.tgH = 1
        add.tgD = 1

        var matvecOut = output
        matvecOut.bindingIndex = 0
        var residualIn = residual
        residualIn.bindingIndex = 1
        var addOut = output
        addOut.bindingIndex = 2
        try appendBuffer(matvecOut, to: &add)
        try appendBuffer(residualIn, to: &add)
        try appendBuffer(addOut, to: &add)
        try appendConstants(skipConstants, to: &add)
        var count = SmeltConstantRecord.empty()
        count.kind = SmeltConstantRecord.kindLiteralU32
        count.bindingIndex = 3
        count.value = UInt32(rows)
        try appendConstant(count, to: &add)

        return (affine, add)
    }

    private func fuseNormScaleGateUp(
        norm: SmeltDispatchRecord,
        gateUp: SmeltDispatchRecord,
        scalePipeline: UInt16,
        scaledPipeline: UInt16
    ) throws -> (scale: SmeltDispatchRecord, scaledGateUp: SmeltDispatchRecord) {
        guard let normInput = buffer(binding: 0, in: norm),
              let normWeight = buffer(binding: 1, in: norm),
              let normOutput = buffer(binding: 2, in: norm),
              let gateWeights = buffer(binding: 0, in: gateUp),
              let gateScales = buffer(binding: 1, in: gateUp),
              let gateBiases = buffer(binding: 2, in: gateUp),
              let upWeights = buffer(binding: 3, in: gateUp),
              let upScales = buffer(binding: 4, in: gateUp),
              let upBiases = buffer(binding: 5, in: gateUp),
              let gateInput = buffer(binding: 6, in: gateUp),
              let output = buffer(binding: 7, in: gateUp)
        else {
            throw KernelLabError.invalidPackage(
                "norm-scale gate-up pair is missing expected bindings"
            )
        }
        guard sameBufferLocation(gateInput, normOutput) else {
            throw KernelLabError.invalidPackage(
                "norm-scale gate-up consumer input does not read norm output"
            )
        }

        var scaleScratch = normInput
        scaleScratch.slot = Int16(SmeltFixedSlot.normScaleScratch.rawValue)
        scaleScratch.offsetKind = 0
        scaleScratch.offset = 0

        var scale = norm
        scale.pipeline = scalePipeline
        scale.bufferCount = 0
        scale.constantCount = 0
        scale.dispatchStyle = SmeltDispatchRecord.styleThreadgroups
        scale.gridW = 1
        scale.gridH = 1
        scale.gridD = 1
        scale.tgW = norm.tgW
        scale.tgH = 1
        scale.tgD = 1
        try appendBuffer(rebinding(normInput, to: 0), to: &scale)
        try appendBuffer(rebinding(scaleScratch, to: 1), to: &scale)
        try appendConstant(literalU32(2_048, binding: 2), to: &scale)
        try appendConstant(literalF32(1e-6, binding: 3), to: &scale)

        var scaled = gateUp
        scaled.pipeline = scaledPipeline
        scaled.bufferCount = 0
        scaled.constantCount = 0
        try appendBuffer(rebinding(scaleScratch, to: 0), to: &scaled)
        try appendBuffer(rebinding(normInput, to: 1), to: &scaled)
        try appendBuffer(rebinding(normWeight, to: 2), to: &scaled)
        try appendBuffer(rebinding(gateWeights, to: 3), to: &scaled)
        try appendBuffer(rebinding(gateScales, to: 4), to: &scaled)
        try appendBuffer(rebinding(gateBiases, to: 5), to: &scaled)
        try appendBuffer(rebinding(upWeights, to: 6), to: &scaled)
        try appendBuffer(rebinding(upScales, to: 7), to: &scaled)
        try appendBuffer(rebinding(upBiases, to: 8), to: &scaled)
        try appendBuffer(rebinding(output, to: 9), to: &scaled)
        try appendConstant(literalU32(11_008, binding: 10), to: &scaled)

        return (scale, scaled)
    }

    private func fuseLMHeadArgmax(
        lmHead: SmeltDispatchRecord,
        partials: SmeltDispatchRecord,
        reduce: SmeltDispatchRecord,
        fusedPipeline: UInt16,
        reducePipeline: UInt16,
        shape: (rows: Int, cols: Int, groupSize: Int)
    ) throws -> (fused: SmeltDispatchRecord, reduce: SmeltDispatchRecord) {
        guard let weights = buffer(binding: 0, in: lmHead),
              let scales = buffer(binding: 1, in: lmHead),
              let biases = buffer(binding: 2, in: lmHead),
              let input = buffer(binding: 3, in: lmHead),
              let logits = buffer(binding: 4, in: lmHead),
              let argmaxInput = buffer(binding: 0, in: partials),
              let partialKeys = buffer(binding: 1, in: partials),
              let reduceInput = buffer(binding: 0, in: reduce),
              let argmaxOutput = buffer(binding: 1, in: reduce)
        else {
            throw KernelLabError.invalidPackage(
                "lm_head argmax trio is missing expected bindings"
            )
        }
        guard sameBufferLocation(logits, argmaxInput) else {
            throw KernelLabError.invalidPackage(
                "lm_head output does not feed argmax partials input"
            )
        }
        guard sameBufferLocation(partialKeys, reduceInput) else {
            throw KernelLabError.invalidPackage(
                "argmax partials output does not feed reduce input"
            )
        }
        if let vocabSize = constant(binding: 2, in: partials),
           vocabSize.kind == SmeltConstantRecord.kindLiteralU32,
           vocabSize.value != UInt32(shape.rows)
        {
            throw KernelLabError.invalidPackage(
                "argmax partials vocab \(vocabSize.value) does not match lm_head rows \(shape.rows)"
            )
        }

        var fused = lmHead
        fused.pipeline = fusedPipeline
        fused.bufferCount = 0
        fused.constantCount = 0
        fused.dispatchStyle = SmeltDispatchRecord.styleThreadgroups
        fused.gridW = UInt32((shape.rows + 7) / 8)
        fused.gridH = 1
        fused.gridD = 1
        fused.tgW = 64
        fused.tgH = 1
        fused.tgD = 1
        try appendBuffer(rebinding(weights, to: 0), to: &fused)
        try appendBuffer(rebinding(scales, to: 1), to: &fused)
        try appendBuffer(rebinding(biases, to: 2), to: &fused)
        try appendBuffer(rebinding(input, to: 3), to: &fused)
        try appendBuffer(rebinding(partialKeys, to: 4), to: &fused)
        try appendConstant(literalU32(1, binding: 5), to: &fused)
        try appendConstant(literalF32(0, binding: 6), to: &fused)

        var fusedReduce = reduce
        fusedReduce.pipeline = reducePipeline
        fusedReduce.bufferCount = 0
        fusedReduce.constantCount = 0
        fusedReduce.dispatchStyle = SmeltDispatchRecord.styleThreadgroups
        fusedReduce.gridW = 1
        fusedReduce.gridH = 1
        fusedReduce.gridD = 1
        fusedReduce.tgW = 256
        fusedReduce.tgH = 1
        fusedReduce.tgD = 1
        try appendBuffer(rebinding(partialKeys, to: 0), to: &fusedReduce)
        try appendBuffer(rebinding(argmaxOutput, to: 1), to: &fusedReduce)
        try appendConstant(literalU32(1, binding: 2), to: &fusedReduce)

        return (fused, fusedReduce)
    }

    private func fuseKVAffinePair(
        k: SmeltDispatchRecord,
        v: SmeltDispatchRecord,
        fusedPipeline: UInt16
    ) throws -> SmeltDispatchRecord {
        guard k.dispatchStyle == v.dispatchStyle,
              k.gridW == v.gridW,
              k.gridH == v.gridH,
              k.gridD == v.gridD,
              k.tgW == v.tgW,
              k.tgH == v.tgH,
              k.tgD == v.tgD,
              k.minSeqLen == v.minSeqLen
        else {
            throw KernelLabError.invalidPackage(
                "cannot fuse K/V affine pair with incompatible dispatch geometry"
            )
        }
        guard let kWeights = buffer(binding: 0, in: k),
              let kScales = buffer(binding: 1, in: k),
              let kBiases = buffer(binding: 2, in: k),
              let kInput = buffer(binding: 3, in: k),
              let kOutput = buffer(binding: 4, in: k),
              let kResidual = buffer(binding: 5, in: k),
              let vWeights = buffer(binding: 0, in: v),
              let vScales = buffer(binding: 1, in: v),
              let vBiases = buffer(binding: 2, in: v),
              let vInput = buffer(binding: 3, in: v),
              let vOutput = buffer(binding: 4, in: v),
              let vResidual = buffer(binding: 5, in: v),
              sameBufferLocation(kInput, vInput)
        else {
            throw KernelLabError.invalidPackage(
                "K/V affine pair is missing expected bindings"
            )
        }

        var fused = SmeltDispatchRecord.empty()
        fused.opKind = SmeltDispatchRecord.opDispatch
        fused.pipeline = fusedPipeline
        fused.dispatchStyle = k.dispatchStyle
        fused.minSeqLen = k.minSeqLen
        fused.gridW = k.gridW
        fused.gridH = k.gridH
        fused.gridD = k.gridD
        fused.tgW = k.tgW
        fused.tgH = k.tgH
        fused.tgD = k.tgD

        try appendBuffer(rebinding(kWeights, to: 0), to: &fused)
        try appendBuffer(rebinding(kScales, to: 1), to: &fused)
        try appendBuffer(rebinding(kBiases, to: 2), to: &fused)
        try appendBuffer(rebinding(vWeights, to: 3), to: &fused)
        try appendBuffer(rebinding(vScales, to: 4), to: &fused)
        try appendBuffer(rebinding(vBiases, to: 5), to: &fused)
        try appendBuffer(rebinding(kInput, to: 6), to: &fused)
        try appendBuffer(rebinding(kOutput, to: 7), to: &fused)
        try appendBuffer(rebinding(vOutput, to: 8), to: &fused)
        try appendBuffer(rebinding(kResidual, to: 9), to: &fused)
        try appendBuffer(rebinding(vResidual, to: 10), to: &fused)
        return fused
    }

    private func unfusedAffinePipelineName(from fusedName: String) -> String? {
        let generatedPrefix = "fused_affine_matvec_add_"
        if fusedName.hasPrefix(generatedPrefix) {
            return "affine_matvec_" + fusedName.dropFirst(generatedPrefix.count)
        }
        let genericPrefix = "fused_affine_matvec_add:"
        if fusedName.hasPrefix(genericPrefix) {
            return "affine_matvec:" + fusedName.dropFirst(genericPrefix.count)
        }
        return nil
    }

    private func lmHeadShape(
        from pipelineName: String
    ) -> (rows: Int, cols: Int, groupSize: Int)? {
        guard pipelineName.hasPrefix("affine_matvec_c"),
              let cols = intAfter("_c", in: pipelineName),
              let rows = intAfter("_r", in: pipelineName),
              let groupSize = intAfter("_g", in: pipelineName)
        else {
            return nil
        }
        return (rows: rows, cols: cols, groupSize: groupSize)
    }

    private func matchesFilter(_ pipelineName: String, filter: String) -> Bool {
        let parts = filter
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !parts.isEmpty else { return true }
        return parts.contains { pipelineName.contains($0) }
    }

    private func rowTile(from pipelineName: String) -> Int? {
        if let rowTile = intAfter("_rows", in: pipelineName) {
            return rowTile
        }
        if pipelineName.hasPrefix("affine_matvec_c")
            || pipelineName.hasPrefix("fused_affine_gate_up_")
            || pipelineName.hasPrefix("fused_affine_matvec_add_")
        {
            return 8
        }
        return nil
    }

    private func threadgroupWidth(from pipelineName: String) -> Int? {
        if pipelineName.range(of: #"_sg[0-9]+"#, options: .regularExpression) != nil {
            return 128
        }
        if let rowTile = rowTile(from: pipelineName) {
            if rowTile >= 32 {
                return 256
            }
            if rowTile >= 16 {
                return 128
            }
        }
        return nil
    }

    private func rowCount(
        from pipelineName: String,
        record: SmeltDispatchRecord
    ) -> Int? {
        if let rows = intAfter("_r", in: pipelineName) {
            return rows
        }
        if let rowTile = intAfter("_rows", in: pipelineName) {
            return Int(record.gridW) * rowTile
        }
        guard record.gridW > 0 else { return nil }
        return Int(record.gridW) * 8
    }

    private func intAfter(_ marker: String, in value: String) -> Int? {
        guard let range = value.range(of: marker) else { return nil }
        var digits = ""
        var index = range.upperBound
        while index < value.endIndex {
            let char = value[index]
            guard let digit = char.wholeNumberValue else { break }
            digits += String(digit)
            index = value.index(after: index)
        }
        return digits.isEmpty ? nil : Int(digits)
    }

    private func fuseRopeKVCacheUpdate(
        rope: SmeltDispatchRecord,
        cache: SmeltDispatchRecord,
        fusedPipeline: UInt16
    ) throws -> SmeltDispatchRecord {
        guard rope.dispatchStyle == cache.dispatchStyle,
              rope.gridW == cache.gridW,
              rope.gridH == cache.gridH,
              rope.gridD == cache.gridD,
              rope.tgW == cache.tgW,
              rope.tgH == cache.tgH,
              rope.tgD == cache.tgD,
              rope.minSeqLen == cache.minSeqLen
        else {
            throw KernelLabError.invalidPackage(
                "cannot fuse apply_rope->kv_cache_update with incompatible dispatch geometry"
            )
        }
        guard let ropeData = buffer(binding: 0, in: rope),
              let ropeCos = buffer(binding: 1, in: rope),
              let ropeSin = buffer(binding: 2, in: rope),
              let cacheOut = buffer(binding: 0, in: cache),
              let cacheInput = buffer(binding: 1, in: cache),
              sameBufferLocation(ropeData, cacheInput)
        else {
            throw KernelLabError.invalidPackage(
                "cannot fuse apply_rope->kv_cache_update with incompatible buffers"
            )
        }
        guard let ropeHeadDim = constant(binding: 3, in: rope),
              let ropeDim = constant(binding: 4, in: rope),
              let ropeNumHeads = constant(binding: 5, in: rope),
              let ropeLayout = constant(binding: 6, in: rope),
              let cacheSeqCapacity = constant(binding: 2, in: cache),
              let cacheHeadDim = constant(binding: 3, in: cache),
              let position = constant(binding: 4, in: cache),
              let cacheNumHeads = constant(binding: 5, in: cache),
              sameConstantValue(ropeHeadDim, cacheHeadDim),
              sameConstantValue(ropeNumHeads, cacheNumHeads)
        else {
            throw KernelLabError.invalidPackage(
                "cannot fuse apply_rope->kv_cache_update with incompatible constants"
            )
        }

        var fused = SmeltDispatchRecord.empty()
        fused.opKind = SmeltDispatchRecord.opDispatch
        fused.pipeline = fusedPipeline
        fused.dispatchStyle = cache.dispatchStyle
        fused.minSeqLen = cache.minSeqLen
        fused.gridW = cache.gridW
        fused.gridH = cache.gridH
        fused.gridD = cache.gridD
        fused.tgW = cache.tgW
        fused.tgH = cache.tgH
        fused.tgD = cache.tgD

        try appendBuffer(rebinding(cacheOut, to: 0), to: &fused)
        try appendBuffer(rebinding(cacheInput, to: 1), to: &fused)
        try appendBuffer(rebinding(ropeCos, to: 2), to: &fused)
        try appendBuffer(rebinding(ropeSin, to: 3), to: &fused)
        try appendConstant(rebinding(cacheSeqCapacity, to: 4), to: &fused)
        try appendConstant(rebinding(cacheHeadDim, to: 5), to: &fused)
        try appendConstant(rebinding(position, to: 6), to: &fused)
        try appendConstant(rebinding(cacheNumHeads, to: 7), to: &fused)
        try appendConstant(rebinding(ropeDim, to: 8), to: &fused)
        try appendConstant(rebinding(ropeLayout, to: 9), to: &fused)
        return fused
    }

    private func sameBufferLocation(
        _ lhs: SmeltBufferRecord,
        _ rhs: SmeltBufferRecord
    ) -> Bool {
        lhs.slot == rhs.slot
            && lhs.offsetKind == rhs.offsetKind
            && lhs.offset == rhs.offset
    }

    private func sameConstantValue(
        _ lhs: SmeltConstantRecord,
        _ rhs: SmeltConstantRecord
    ) -> Bool {
        lhs.kind == rhs.kind && lhs.value == rhs.value
    }

    private func buffer(
        binding: UInt8,
        in record: SmeltDispatchRecord
    ) -> SmeltBufferRecord? {
        for index in 0..<Int(record.bufferCount) {
            let candidate = getBuffer(record, index: index)
            if candidate.bindingIndex == binding {
                return candidate
            }
        }
        return nil
    }

    private func constant(
        binding: UInt8,
        in record: SmeltDispatchRecord
    ) -> SmeltConstantRecord? {
        for index in 0..<Int(record.constantCount) {
            let candidate = getConstant(record, index: index)
            if candidate.bindingIndex == binding {
                return candidate
            }
        }
        return nil
    }

    private func rebinding(
        _ buffer: SmeltBufferRecord,
        to bindingIndex: UInt8
    ) -> SmeltBufferRecord {
        var rebound = buffer
        rebound.bindingIndex = bindingIndex
        return rebound
    }

    private func rebinding(
        _ constant: SmeltConstantRecord,
        to bindingIndex: UInt8
    ) -> SmeltConstantRecord {
        var rebound = constant
        rebound.bindingIndex = bindingIndex
        return rebound
    }

    private func literalU32(_ value: UInt32, binding: UInt8) -> SmeltConstantRecord {
        var constant = SmeltConstantRecord.empty()
        constant.kind = SmeltConstantRecord.kindLiteralU32
        constant.bindingIndex = binding
        constant.value = value
        return constant
    }

    private func literalF32(_ value: Float, binding: UInt8) -> SmeltConstantRecord {
        var constant = SmeltConstantRecord.empty()
        constant.kind = SmeltConstantRecord.kindLiteralF32
        constant.bindingIndex = binding
        constant.value = value.bitPattern
        return constant
    }

    private func skipOnlyConstants(
        in record: SmeltDispatchRecord
    ) throws -> [SmeltConstantRecord] {
        var result: [SmeltConstantRecord] = []
        for index in 0..<Int(record.constantCount) {
            let constant = getConstant(record, index: index)
            guard constant.bindingIndex == UInt8.max else {
                throw KernelLabError.invalidPackage(
                    "cannot unfuse record with non-skip constant binding \(constant.bindingIndex)"
                )
            }
            result.append(constant)
        }
        return result
    }

    private func appendBuffer(
        _ buffer: SmeltBufferRecord,
        to record: inout SmeltDispatchRecord
    ) throws {
        let index = Int(record.bufferCount)
        guard index < agentMaxBuffersPerDispatch else {
            throw KernelLabError.invalidPackage("too many replay buffer bindings")
        }
        setBuffer(buffer, index: index, in: &record)
        record.bufferCount += 1
    }

    private func appendConstants(
        _ constants: [SmeltConstantRecord],
        to record: inout SmeltDispatchRecord
    ) throws {
        for constant in constants {
            try appendConstant(constant, to: &record)
        }
    }

    private func appendConstant(
        _ constant: SmeltConstantRecord,
        to record: inout SmeltDispatchRecord
    ) throws {
        let index = Int(record.constantCount)
        guard index < agentMaxConstantsPerDispatch else {
            throw KernelLabError.invalidPackage("too many replay constants")
        }
        setConstant(constant, index: index, in: &record)
        record.constantCount += 1
    }

    private func setBuffer(
        _ buffer: SmeltBufferRecord,
        index: Int,
        in record: inout SmeltDispatchRecord
    ) {
        switch index {
        case 0: record.buf0 = buffer
        case 1: record.buf1 = buffer
        case 2: record.buf2 = buffer
        case 3: record.buf3 = buffer
        case 4: record.buf4 = buffer
        case 5: record.buf5 = buffer
        case 6: record.buf6 = buffer
        case 7: record.buf7 = buffer
        case 8: record.buf8 = buffer
        case 9: record.buf9 = buffer
        case 10: record.buf10 = buffer
        case 11: record.buf11 = buffer
        case 12: record.buf12 = buffer
        case 13: record.buf13 = buffer
        case 14: record.buf14 = buffer
        case 15: record.buf15 = buffer
        default: break
        }
    }

    private func setConstant(
        _ constant: SmeltConstantRecord,
        index: Int,
        in record: inout SmeltDispatchRecord
    ) {
        switch index {
        case 0: record.con0 = constant
        case 1: record.con1 = constant
        case 2: record.con2 = constant
        case 3: record.con3 = constant
        case 4: record.con4 = constant
        case 5: record.con5 = constant
        case 6: record.con6 = constant
        case 7: record.con7 = constant
        default: break
        }
    }

    private func stats(
        name: String,
        samples: [KernelPackageReplaySample]
    ) throws -> KernelPackageReplayStats {
        guard !samples.isEmpty else {
            throw KernelLabError.missingSample(name)
        }
        let total = samples.map(\.totalUs).sorted()
        let pureGpu = samples.map(\.pureGpuUs).sorted()
        let p95Index = min(total.count - 1, Int(ceil(Double(total.count) * 0.95)) - 1)
        return KernelPackageReplayStats(
            totalMedianUs: total[total.count / 2],
            totalP95Us: total[p95Index],
            pureGpuMedianUs: pureGpu[pureGpu.count / 2],
            pureGpuP95Us: pureGpu[p95Index]
        )
    }

    private func formatUs(_ value: Double) -> String {
        String(format: "%7.2f", value)
    }

    private func signedUs(_ value: Double) -> String {
        String(format: "%+7.2f", value)
    }

    private func signedPct(_ value: Double) -> String {
        String(format: "%+5.1f", value)
    }
}

private final class KernelLab {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]

    init(
        packagePath: String?,
        libraryMode: KernelLabLibraryMode,
        shaderDir: String
    ) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw KernelLabError.missingDevice
        }
        guard let queue = device.makeCommandQueue() else {
            throw KernelLabError.missingCommandQueue
        }
        self.device = device
        self.queue = queue
        switch libraryMode {
        case .source:
            self.library = try Self.makeSourceLibrary(device: device, shaderDir: shaderDir)
        case .package:
            guard let packagePath else {
                throw KernelLabError.missingPackagePath
            }
            let metallibPath = URL(fileURLWithPath: packagePath)
                .appendingPathComponent("model.metallib")
                .path
            guard FileManager.default.fileExists(atPath: metallibPath) else {
                throw KernelLabError.missingMetallib(metallibPath)
            }
            self.library = try device.makeLibrary(URL: URL(fileURLWithPath: metallibPath))
        }
    }

    fileprivate static func makeSourceLibrary(
        device: MTLDevice,
        shaderDir: String,
        extraFunctionNames: Set<String> = []
    ) throws -> MTLLibrary {
        let root = URL(fileURLWithPath: shaderDir)
        let fileManager = FileManager.default
        let tempRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("smelt-kernel-lab-\(ProcessInfo.processInfo.globallyUniqueString)")
        try fileManager.createDirectory(at: tempRoot, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: tempRoot) }

        var matvec = try readSource(root.appendingPathComponent("lut_matvec.metal"))

        let marker = "#undef SMELT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4"
        guard let range = matvec.range(of: marker) else {
            throw KernelLabError.missingInsertionPoint(marker)
        }
        let authoredOverlay = !extraFunctionNames.isEmpty
            && extraFunctionNames.allSatisfy {
                authoredShaderFunction($0, root: root, fileManager: fileManager)
            }
        var labDeclarations = ""
        if !authoredOverlay {
            labDeclarations = Self.missingMacroDeclarations(
                Self.labMatvecDeclarations,
                from: matvec
            )
            if !matvec.contains("affine_matvec_c2048_r151936_g64_rows8") {
                labDeclarations += Self.labVibeLMHeadRows8Declaration
            }
            if !matvec.contains("fused_affine_gate_up_swiglu_c2048_r11008_g64_rows4_sg1") {
                labDeclarations += Self.labVibeGateUpRows4SG1Declaration
            }
        }
        for functionName in extraFunctionNames.sorted()
            where !matvec.contains(functionName)
                && !labDeclarations.contains(functionName)
        {
            if let declaration = labDeclaration(for: functionName) {
                labDeclarations += "\n" + declaration
            }
        }
        matvec.insert(contentsOf: labDeclarations, at: range.lowerBound)
        let labMatvec = tempRoot.appendingPathComponent("lut_matvec.metal")
        try matvec.write(to: labMatvec, atomically: true, encoding: .utf8)

        let moduleCache = tempRoot.appendingPathComponent("module-cache")
        try fileManager.createDirectory(at: moduleCache, withIntermediateDirectories: true)
        let sources = [
            root.appendingPathComponent("activations.metal"),
            root.appendingPathComponent("norms.metal"),
            root.appendingPathComponent("attention.metal"),
            root.appendingPathComponent("recurrence.metal"),
            root.appendingPathComponent("signed_quant.metal"),
            root.appendingPathComponent("signed_quant_precise.metal"),
            labMatvec,
        ]
        var airFiles: [URL] = []
        for source in sources {
            guard fileManager.fileExists(atPath: source.path) else {
                throw KernelLabError.missingShaderSource(source.path)
            }
            let air = tempRoot.appendingPathComponent(
                source.deletingPathExtension().lastPathComponent + ".air"
            )
            var compileArguments = [
                "-sdk", "macosx", "metal",
                "-fmodules-cache-path=\(moduleCache.path)",
                "-I", root.path,
            ]
            let file = source.lastPathComponent
            if file.hasSuffix("_precise.metal")
                || file == "attention.metal"
                || file == "norms.metal"
                || file == "recurrence.metal"
                || file == "signed_quant.metal"
            {
                compileArguments.append("-fno-fast-math")
            }
            compileArguments += ["-c", source.path, "-o", air.path]
            try runProcess("/usr/bin/xcrun", arguments: compileArguments)
            airFiles.append(air)
        }

        let metallib = tempRoot.appendingPathComponent("kernel-lab.metallib")
        try runProcess(
            "/usr/bin/xcrun",
            arguments: ["-sdk", "macosx", "metallib"]
                + airFiles.map(\.path)
                + ["-o", metallib.path]
        )
        return try device.makeLibrary(URL: metallib)
    }

    /// Kernel Lab carries a small set of generated declarations for shapes
    /// that are absent from older shader sources. Once a shape is promoted
    /// into the authored source, do not declare the same Metal entry point a
    /// second time. Keeping this source-driven prevents every promoted generic
    /// kernel from needing a new one-off exclusion in the lab harness.
    private static func missingMacroDeclarations(
        _ declarations: String,
        from authoredSource: String
    ) -> String {
        declarations.split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
            .filter { line in
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard trimmed.hasPrefix("SMELT_DECLARE_"),
                      let open = trimmed.firstIndex(of: "("),
                      let comma = trimmed[trimmed.index(after: open)...]
                        .firstIndex(of: ",")
                else {
                    return true
                }
                let functionName = trimmed[trimmed.index(after: open)..<comma]
                    .trimmingCharacters(in: .whitespaces)
                return !authoredSource.contains("(\(functionName),")
                    && !authoredSource.contains("kernel void \(functionName)(")
            }
            .joined(separator: "\n")
    }

    private static func authoredShaderFunction(
        _ functionName: String,
        root: URL,
        fileManager: FileManager
    ) -> Bool {
        guard let enumerator = fileManager.enumerator(
            at: root,
            includingPropertiesForKeys: nil
        ) else {
            return false
        }
        for case let url as URL in enumerator where url.pathExtension == "metal" {
            guard let source = try? String(contentsOf: url, encoding: .utf8) else {
                continue
            }
            if source.contains("kernel void \(functionName)(") {
                return true
            }
        }
        return false
    }

    fileprivate static func labDeclaration(for functionName: String) -> String? {
        if let shape = parseAffineMatvecArgmaxShape(functionName) {
            return "SMELT_DECLARE_AFFINE_MATVEC_ARGMAX_FIXED_BATCHED_GROUP(\(functionName), \(shape.rows), \(shape.cols), \(shape.groupSize), \(shape.batchTile))\n"
        }
        if let rows = parseLMHeadArgmaxReduceRows(functionName) {
            return "SMELT_DECLARE_LM_HEAD_ARGMAX_REDUCE(\(functionName), \(rows))\n"
        }
        if let shape = parseFusedDualAffineMatvecAddShape(functionName) {
            return fusedDualAffineMatvecAddRows4Declaration(functionName, shape)
        }
        guard let shape = parseGeneratedShape(functionName) else {
            return nil
        }
        switch shape.kind {
        case .affineMatvec:
            guard shape.variant == .standard else {
                return nil
            }
            switch shape.rowTile {
            case 4:
                return "SMELT_DECLARE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(\(functionName), \(shape.rows), \(shape.cols), \(shape.groupSize))\n"
            case 8:
                return "SMELT_DECLARE_AFFINE_MATVEC_FIXED_ROWS8_GROUP(\(functionName), \(shape.rows), \(shape.cols), \(shape.groupSize))\n"
            default:
                return nil
            }
        case .fusedAffineMatvecAdd:
            switch shape.rowTile {
            case 4:
                switch shape.variant {
                case .threadgroupInput:
                    return "SMELT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_TGINPUT_GROUP(\(functionName), \(shape.rows), \(shape.cols), \(shape.groupSize))\n"
                case .singleRowSimdgroup:
                    return "SMELT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_SG1_GROUP(\(functionName), \(shape.rows), \(shape.cols), \(shape.groupSize))\n"
                case .scaleBiasCache:
                    return "SMELT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_SBCACHE_GROUP(\(functionName), \(shape.rows), \(shape.cols), \(shape.groupSize))\n"
                case .standard:
                    return "SMELT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_GROUP(\(functionName), \(shape.rows), \(shape.cols), \(shape.groupSize))\n"
                case .exp2Activation:
                    return nil
                }
            case 8:
                return fusedAffineMatvecAddRows8Declaration(functionName, shape)
            case 16:
                return "SMELT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS16_GROUP(\(functionName), \(shape.rows), \(shape.cols), \(shape.groupSize))\n"
            case 32:
                return "SMELT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS32_GROUP(\(functionName), \(shape.rows), \(shape.cols), \(shape.groupSize))\n"
            default:
                return nil
            }
        case .fusedGateUpSwiglu:
            switch shape.rowTile {
            case 4:
                switch shape.variant {
                case .threadgroupInput:
                    return "SMELT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_ROWS4_TGINPUT_GROUP(\(functionName), \(shape.rows), \(shape.cols), \(shape.groupSize))\n"
                case .exp2Activation:
                    return "SMELT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_ROWS4_EXP2_GROUP(\(functionName), \(shape.rows), \(shape.cols), \(shape.groupSize))\n"
                case .scaleBiasCache:
                    return "SMELT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_ROWS4_SBCACHE_GROUP(\(functionName), \(shape.rows), \(shape.cols), \(shape.groupSize))\n"
                case .standard:
                    return fusedGateUpSwigluDeclaration(functionName, shape)
                case .singleRowSimdgroup:
                    return nil
                }
            case 8:
                return fusedGateUpSwigluDeclaration(functionName, shape)
            case 16:
                return "SMELT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_ROWS16_GROUP(\(functionName), \(shape.rows), \(shape.cols), \(shape.groupSize))\n"
            case 32:
                return "SMELT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_ROWS32_GROUP(\(functionName), \(shape.rows), \(shape.cols), \(shape.groupSize))\n"
            default:
                return nil
            }
        }
    }

    private enum GeneratedShapeKind {
        case affineMatvec
        case fusedAffineMatvecAdd
        case fusedGateUpSwiglu
    }

    private struct GeneratedShape {
        let kind: GeneratedShapeKind
        let cols: Int
        let rows: Int
        let groupSize: Int
        let rowTile: Int
        let variant: GeneratedShapeVariant
    }

    private enum GeneratedShapeVariant: String {
        case standard = ""
        case threadgroupInput = "tginput"
        case exp2Activation = "exp2"
        case singleRowSimdgroup = "sg1"
        case scaleBiasCache = "sbcache"

        static func parse(afterRows suffixParts: ArraySlice<String>) -> GeneratedShapeVariant? {
            guard let suffix = suffixParts.first else {
                return .standard
            }
            guard suffixParts.count == 1 else {
                return nil
            }
            return GeneratedShapeVariant(rawValue: suffix)
        }
    }

    private static func parseAffineMatvecArgmaxShape(
        _ functionName: String
    ) -> (rows: Int, cols: Int, groupSize: Int, batchTile: Int)? {
        guard functionName.hasPrefix("affine_matvec_argmax_c") else {
            return nil
        }
        let parts = functionName.split(separator: "_").map(String.init)
        guard let cols = intComponent("c", in: parts),
              let rows = intComponent("r", in: parts),
              let groupSize = intComponent("g", in: parts),
              let batchTile = intComponent("b", in: parts)
        else {
            return nil
        }
        return (
            rows: rows,
            cols: cols,
            groupSize: groupSize,
            batchTile: batchTile
        )
    }

    private static func parseFusedDualAffineMatvecAddShape(
        _ functionName: String
    ) -> (rows: Int, cols: Int, groupSize: Int)? {
        guard functionName.hasPrefix("fused_dual_affine_matvec_add_c"),
              functionName.hasSuffix("_rows4")
        else {
            return nil
        }
        let parts = functionName.split(separator: "_").map(String.init)
        guard let cols = intComponent("c", in: parts),
              let rows = intComponent("r", in: parts),
              let groupSize = intComponent("g", in: parts)
        else {
            return nil
        }
        return (rows: rows, cols: cols, groupSize: groupSize)
    }

    private static func parseLMHeadArgmaxReduceRows(
        _ functionName: String
    ) -> Int? {
        guard functionName.hasPrefix("lm_head_argmax_reduce_r") else {
            return nil
        }
        let parts = functionName.split(separator: "_").map(String.init)
        return intComponent("r", in: parts)
    }

    private static func parseGeneratedShape(_ functionName: String) -> GeneratedShape? {
        let kind: GeneratedShapeKind
        if functionName.hasPrefix("affine_matvec_c") {
            kind = .affineMatvec
        } else if functionName.hasPrefix("fused_affine_matvec_add_c") {
            kind = .fusedAffineMatvecAdd
        } else if functionName.hasPrefix("fused_affine_gate_up_swiglu_c") {
            kind = .fusedGateUpSwiglu
        } else {
            return nil
        }

        let parts = functionName.split(separator: "_").map(String.init)
        guard let rowPartIndex = parts.firstIndex(where: { $0.hasPrefix("rows") }),
              let rowTile = intComponent("rows", in: [parts[rowPartIndex]]),
              [4, 8, 16, 32].contains(rowTile),
              let variant = GeneratedShapeVariant.parse(afterRows: parts[(rowPartIndex + 1)...])
        else {
            return nil
        }
        guard variant == .standard || rowTile == 4 else {
            return nil
        }

        guard let cols = intComponent("c", in: parts),
              let rows = intComponent("r", in: parts),
              let groupSize = intComponent("g", in: parts)
        else {
            return nil
        }
        return GeneratedShape(
            kind: kind,
            cols: cols,
            rows: rows,
            groupSize: groupSize,
            rowTile: rowTile,
            variant: variant
        )
    }

    private static func intComponent(_ prefix: String, in parts: [String]) -> Int? {
        for part in parts where part.hasPrefix(prefix) {
            let digits = part.dropFirst(prefix.count)
            guard !digits.isEmpty, digits.allSatisfy(\.isNumber) else {
                continue
            }
            return Int(digits)
        }
        return nil
    }

    private static func fusedAffineMatvecAddRows8Declaration(
        _ functionName: String,
        _ shape: GeneratedShape
    ) -> String {
        """
        kernel void \(functionName)(
            device const uint8_t* weights  [[buffer(0)]],
            device const half*    scales   [[buffer(1)]],
            device const half*    biases   [[buffer(2)]],
            device const half*    input    [[buffer(3)]],
            device half*          output   [[buffer(4)]],
            device const half*    residual [[buffer(5)]],
            uint tgid       [[threadgroup_position_in_grid]],
            uint simd_lane  [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]]
        ) {
            fused_affine_matvec_add_fixed<\(shape.rows), \(shape.cols), \(shape.groupSize)>(
                weights, scales, biases, input, output, residual,
                tgid, simd_lane, simd_group
            );
        }

        """
    }

    private static func fusedDualAffineMatvecAddRows4Declaration(
        _ functionName: String,
        _ shape: (rows: Int, cols: Int, groupSize: Int)
    ) -> String {
        """
        kernel void \(functionName)(
            device const uint8_t* w1_weights [[buffer(0)]],
            device const half*    w1_scales  [[buffer(1)]],
            device const half*    w1_biases  [[buffer(2)]],
            device const uint8_t* w2_weights [[buffer(3)]],
            device const half*    w2_scales  [[buffer(4)]],
            device const half*    w2_biases  [[buffer(5)]],
            device const half*    input      [[buffer(6)]],
            device half*          output1    [[buffer(7)]],
            device half*          output2    [[buffer(8)]],
            device const half*    residual1  [[buffer(9)]],
            device const half*    residual2  [[buffer(10)]],
            uint tgid       [[threadgroup_position_in_grid]],
            uint simd_lane  [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]]
        ) {
            fused_dual_affine_matvec_add_fixed_rows4<\(shape.rows), \(shape.cols), \(shape.groupSize)>(
                w1_weights, w1_scales, w1_biases,
                w2_weights, w2_scales, w2_biases,
                input, output1, output2,
                residual1, residual2,
                tgid, simd_lane, simd_group
            );
        }

        """
    }

    private static func fusedGateUpSwigluDeclaration(
        _ functionName: String,
        _ shape: GeneratedShape
    ) -> String {
        let callee = shape.rowTile == 4
            ? "fused_affine_gate_up_swiglu_fixed_rows4"
            : "fused_affine_gate_up_swiglu_fixed"
        return """
        kernel void \(functionName)(
            device const uint8_t* gate_weights [[buffer(0)]],
            device const half*    gate_scales  [[buffer(1)]],
            device const half*    gate_biases  [[buffer(2)]],
            device const uint8_t* up_weights   [[buffer(3)]],
            device const half*    up_scales    [[buffer(4)]],
            device const half*    up_biases    [[buffer(5)]],
            device const half*    input        [[buffer(6)]],
            device half*          output       [[buffer(7)]],
            uint tgid       [[threadgroup_position_in_grid]],
            uint simd_lane  [[thread_index_in_simdgroup]],
            uint simd_group [[simdgroup_index_in_threadgroup]]
        ) {
            \(callee)<\(shape.rows), \(shape.cols), \(shape.groupSize)>(
                gate_weights, gate_scales, gate_biases,
                up_weights, up_scales, up_biases,
                input, output,
                tgid, simd_lane, simd_group
            );
        }

        """
    }

    private static func readSource(_ url: URL) throws -> String {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw KernelLabError.missingShaderSource(url.path)
        }
        return try String(contentsOf: url, encoding: .utf8)
    }

    private static func runProcess(_ executable: String, arguments: [String]) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(decoding: data, as: UTF8.self)
            throw KernelLabError.metalCompileFailed(
                ([executable] + arguments).joined(separator: " ") + "\n" + output
            )
        }
    }

    private static let labMatvecDeclarations = """

SMELT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_GROUP(fused_affine_matvec_add_c3584_r1024_g64_rows4, 1024, 3584, 64)
SMELT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_GROUP(fused_affine_matvec_add_c2048_r1024_g64_rows4, 1024, 2048, 64)
SMELT_DECLARE_NORM_SCALE_AFFINE_MATVEC_FIXED_ROWS4_GROUP(norm_scale_affine_matvec_c1024_r6144_g64_rows4, 6144, 1024, 64)
SMELT_DECLARE_AFFINE_MATVEC_FIXED_ROWS8_GROUP(affine_matvec_c1024_r2048_g64_rows8, 2048, 1024, 64)
SMELT_DECLARE_AFFINE_MATVEC_FIXED_ROWS8_GROUP(affine_matvec_c1024_r6144_g64_rows8, 6144, 1024, 64)
SMELT_DECLARE_AFFINE_MATVEC_FIXED_ROWS8_GROUP(affine_matvec_c2048_r1024_g64_rows8, 1024, 2048, 64)
SMELT_DECLARE_AFFINE_MATVEC_FIXED_ROWS8_GROUP(affine_matvec_c3584_r1024_g64_rows8, 1024, 3584, 64)
SMELT_DECLARE_AFFINE_MATVEC_FIXED_ROWS8_GROUP(affine_matvec_c1024_r248320_g64_rows8, 248320, 1024, 64)
SMELT_DECLARE_AFFINE_MATVEC_FIXED_ROWS8_GROUP(affine_matvec_c2048_r2048_g64_rows8, 2048, 2048, 64)
SMELT_DECLARE_AFFINE_MATVEC_FIXED_FULL(affine_matvec_c1024_r248320_g64_batched_full, 248320, 1024)
SMELT_DECLARE_AFFINE_MATVEC_ARGMAX_FIXED_BATCHED_GROUP(affine_matvec_argmax_c1024_r248320_g64_b1, 248320, 1024, 64, 1)
SMELT_DECLARE_AFFINE_MATVEC_ARGMAX_FIXED_BATCHED_GROUP(affine_matvec_argmax_c1024_r248320_g64_b4, 248320, 1024, 64, 4)
SMELT_DECLARE_LM_HEAD_ARGMAX_REDUCE(lm_head_argmax_reduce_r248320, 248320)
SMELT_DECLARE_AFFINE_MATVEC_ARGMAX_FIXED_BATCHED_GROUP(affine_matvec_argmax_c2048_r151936_g64_b1, 151936, 2048, 64, 1)
SMELT_DECLARE_LM_HEAD_ARGMAX_REDUCE(lm_head_argmax_reduce_r151936, 151936)
SMELT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_GROUP(fused_affine_matvec_add_c11008_r2048_g64_rows4, 2048, 11008, 64)
SMELT_DECLARE_FUSED_AFFINE_MATVEC_ADD_FIXED_ROWS4_GROUP(fused_affine_matvec_add_c2048_r256_g64_rows4, 256, 2048, 64)
kernel void fused_affine_matvec_add_c11008_r2048_g64_rows8(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    device const half*    residual [[buffer(5)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    fused_affine_matvec_add_fixed<2048, 11008, 64>(
        weights, scales, biases, input, output, residual,
        tgid, simd_lane, simd_group
    );
}
kernel void fused_affine_matvec_add_c2048_r256_g64_rows8(
    device const uint8_t* weights  [[buffer(0)]],
    device const half*    scales   [[buffer(1)]],
    device const half*    biases   [[buffer(2)]],
    device const half*    input    [[buffer(3)]],
    device half*          output   [[buffer(4)]],
    device const half*    residual [[buffer(5)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    fused_affine_matvec_add_fixed<256, 2048, 64>(
        weights, scales, biases, input, output, residual,
        tgid, simd_lane, simd_group
    );
}
kernel void fused_affine_gate_up_swiglu_c2048_r11008_g64_rows8(
    device const uint8_t* gate_weights [[buffer(0)]],
    device const half*    gate_scales  [[buffer(1)]],
    device const half*    gate_biases  [[buffer(2)]],
    device const uint8_t* up_weights   [[buffer(3)]],
    device const half*    up_scales    [[buffer(4)]],
    device const half*    up_biases    [[buffer(5)]],
    device const half*    input        [[buffer(6)]],
    device half*          output       [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    fused_affine_gate_up_swiglu_fixed<11008, 2048, 64>(
        gate_weights, gate_scales, gate_biases,
        up_weights, up_scales, up_biases,
        input, output,
        tgid, simd_lane, simd_group
    );
}
kernel void fused_affine_gate_up_swiglu_c2048_r11008_g64_rows8_sg2(
    device const uint8_t* gate_weights [[buffer(0)]],
    device const half*    gate_scales  [[buffer(1)]],
    device const half*    gate_biases  [[buffer(2)]],
    device const uint8_t* up_weights   [[buffer(3)]],
    device const half*    up_scales    [[buffer(4)]],
    device const half*    up_biases    [[buffer(5)]],
    device const half*    input        [[buffer(6)]],
    device half*          output       [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    fused_affine_gate_up_swiglu_fixed_rows8_sg2<11008, 2048, 64>(
        gate_weights, gate_scales, gate_biases,
        up_weights, up_scales, up_biases,
        input, output,
        tgid, simd_lane, simd_group
    );
}
kernel void fused_affine_gate_up_swiglu_c2048_r11008_g64_rows4_hacc(
    device const uint8_t* gate_weights [[buffer(0)]],
    device const half*    gate_scales  [[buffer(1)]],
    device const half*    gate_biases  [[buffer(2)]],
    device const uint8_t* up_weights   [[buffer(3)]],
    device const half*    up_scales    [[buffer(4)]],
    device const half*    up_biases    [[buffer(5)]],
    device const half*    input        [[buffer(6)]],
    device half*          output       [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    fused_affine_gate_up_swiglu_fixed_rows4_hacc<11008, 2048, 64>(
        gate_weights, gate_scales, gate_biases,
        up_weights, up_scales, up_biases,
        input, output,
        tgid, simd_lane, simd_group
    );
}
SMELT_DECLARE_FUSED_AFFINE_GATE_UP_FIXED_FULL(fused_affine_gate_up_swiglu_c2048_r11008_g64_batched_full, 11008, 2048)

"""

    private static let labVibeLMHeadRows8Declaration = """

SMELT_DECLARE_AFFINE_MATVEC_FIXED_ROWS8_GROUP(affine_matvec_c2048_r151936_g64_rows8, 151936, 2048, 64)

"""

    private static let labVibeGateUpRows4SG1Declaration = """

kernel void fused_affine_gate_up_swiglu_c2048_r11008_g64_rows4_sg1(
    device const uint8_t* gate_weights [[buffer(0)]],
    device const half*    gate_scales  [[buffer(1)]],
    device const half*    gate_biases  [[buffer(2)]],
    device const uint8_t* up_weights   [[buffer(3)]],
    device const half*    up_scales    [[buffer(4)]],
    device const half*    up_biases    [[buffer(5)]],
    device const half*    input        [[buffer(6)]],
    device half*          output       [[buffer(7)]],
    uint tgid       [[threadgroup_position_in_grid]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    fused_affine_gate_up_swiglu_fixed_rows4_sg1<11008, 2048, 64>(
        gate_weights, gate_scales, gate_biases,
        up_weights, up_scales, up_biases,
        input, output,
        tgid, simd_lane, simd_group
    );
}

"""

    func run(_ testCase: KernelLabCase, iterations: Int, warmup: Int) throws {
        let buffers = try makeBuffers(testCase.buffers)
        let dispatches = testCase.baseline.dispatches + testCase.candidate.dispatches
        for dispatch in dispatches {
            try makePipeline(dispatch)
        }

        for _ in 0..<warmup {
            _ = try runVariant(testCase.baseline, buffers: buffers)
            _ = try runVariant(testCase.candidate, buffers: buffers)
        }

        var baselineSamples: [Double] = []
        var candidateSamples: [Double] = []
        baselineSamples.reserveCapacity(iterations)
        candidateSamples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            baselineSamples.append(try runVariant(testCase.baseline, buffers: buffers))
            candidateSamples.append(try runVariant(testCase.candidate, buffers: buffers))
        }

        let baseline = try stats(name: testCase.baseline.name, samples: baselineSamples)
        let candidate = try stats(name: testCase.candidate.name, samples: candidateSamples)
        let deltaUs = candidate.medianUs - baseline.medianUs
        let deltaPct = baseline.medianUs == 0 ? 0 : deltaUs / baseline.medianUs * 100
        let verdict = deltaUs < 0 ? "candidate faster" : "candidate slower"

        fputs("\(testCase.name):\n", stderr)
        fputs(
            "  baseline  \(testCase.baseline.name): \(formatUs(baseline.medianUs))us median, \(formatUs(baseline.p95Us))us p95\n",
            stderr
        )
        fputs(
            "  candidate \(testCase.candidate.name): \(formatUs(candidate.medianUs))us median, \(formatUs(candidate.p95Us))us p95\n",
            stderr
        )
        fputs(
            "  delta: \(signedUs(deltaUs))us (\(signedPct(deltaPct))%) \(verdict)\n\n",
            stderr
        )
    }

    private func makeBuffers(_ specs: [String: Int]) throws -> [String: MTLBuffer] {
        var result: [String: MTLBuffer] = [:]
        for (name, bytes) in specs {
            guard let buffer = device.makeBuffer(
                length: max(bytes, 16),
                options: .storageModeShared
            ) else {
                throw KernelLabError.missingBuffer(name)
            }
            memset(buffer.contents(), 0, buffer.length)
            buffer.label = "kernel-lab.\(name)"
            result[name] = buffer
        }
        return result
    }

    private func makePipeline(_ dispatch: KernelLabDispatch) throws {
        let key = pipelineKey(dispatch)
        guard pipelines[key] == nil else { return }
        let function: MTLFunction?
        if dispatch.functionConstants.isEmpty {
            function = library.makeFunction(name: dispatch.function)
        } else {
            let values = MTLFunctionConstantValues()
            for (index, constant) in dispatch.functionConstants {
                constant.bind(to: values, index: index)
            }
            function = try library.makeFunction(
                name: dispatch.function,
                constantValues: values
            )
        }
        guard let function else {
            throw KernelLabError.missingFunction(dispatch.function)
        }
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.label = key
        descriptor.computeFunction = function
        pipelines[key] = try device.makeComputePipelineState(
            descriptor: descriptor,
            options: [],
            reflection: nil
        )
    }

    private func runVariant(
        _ variant: KernelLabVariant,
        buffers: [String: MTLBuffer]
    ) throws -> Double {
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw KernelLabError.missingCommandQueue
        }
        commandBuffer.label = "kernel-lab.\(variant.name)"

        for dispatch in variant.dispatches {
            guard let pipeline = pipelines[pipelineKey(dispatch)] else {
                throw KernelLabError.missingPipeline(dispatch.function)
            }
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw KernelLabError.missingCommandQueue
            }
            encoder.label = dispatch.function
            encoder.setComputePipelineState(pipeline)
            for (index, name) in dispatch.buffers {
                guard let buffer = buffers[name] else {
                    throw KernelLabError.missingBuffer(name)
                }
                encoder.setBuffer(buffer, offset: 0, index: index)
            }
            for (index, constant) in dispatch.constants {
                constant.bind(to: encoder, index: index)
            }
            switch dispatch.grid {
            case .threadgroups(let width, let threadsPerThreadgroup):
                encoder.dispatchThreadgroups(
                    MTLSize(width: width, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(
                        width: threadsPerThreadgroup,
                        height: 1,
                        depth: 1
                    )
                )
            case .threadgroups2D(let width, let height, let threadsPerThreadgroup):
                encoder.dispatchThreadgroups(
                    MTLSize(width: width, height: height, depth: 1),
                    threadsPerThreadgroup: MTLSize(
                        width: threadsPerThreadgroup,
                        height: 1,
                        depth: 1
                    )
                )
            case .threads(let width, let threadsPerThreadgroup):
                encoder.dispatchThreads(
                    MTLSize(width: width, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(
                        width: threadsPerThreadgroup,
                        height: 1,
                        depth: 1
                    )
                )
            case .threads3D(
                let width,
                let height,
                let depth,
                let threadsPerThreadgroupWidth,
                let threadsPerThreadgroupHeight
            ):
                encoder.dispatchThreads(
                    MTLSize(width: width, height: height, depth: depth),
                    threadsPerThreadgroup: MTLSize(
                        width: threadsPerThreadgroupWidth,
                        height: threadsPerThreadgroupHeight,
                        depth: 1
                    )
                )
            }
            encoder.endEncoding()
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw error
        }
        return (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1_000_000
    }

    private func stats(name: String, samples: [Double]) throws -> KernelLabStats {
        guard !samples.isEmpty else {
            throw KernelLabError.missingSample(name)
        }
        let sorted = samples.sorted()
        let median = sorted[sorted.count / 2]
        let p95Index = min(sorted.count - 1, Int(ceil(Double(sorted.count) * 0.95)) - 1)
        return KernelLabStats(medianUs: median, p95Us: sorted[p95Index])
    }

    private func pipelineKey(_ dispatch: KernelLabDispatch) -> String {
        guard !dispatch.functionConstants.isEmpty else {
            return dispatch.function
        }
        let suffix = dispatch.functionConstants
            .keys
            .sorted()
            .map { index -> String in
                switch dispatch.functionConstants[index]! {
                case .uint32(let value):
                    return "fc\(index)=\(value)"
                }
            }
            .joined(separator: ",")
        return "\(dispatch.function)[\(suffix)]"
    }

    private func formatUs(_ value: Double) -> String {
        String(format: "%7.2f", value)
    }

    private func signedUs(_ value: Double) -> String {
        String(format: "%+7.2f", value)
    }

    private func signedPct(_ value: Double) -> String {
        String(format: "%+5.1f", value)
    }
}

private func labCases(named name: String) throws -> [KernelLabCase] {
    let cases = [
        qwen08ResidualAddCase(),
        qwen08ResidualAddC2048Case(),
        qwen08DeltaTailC2048x18Case(),
        qwen08NormScaleC6144Case(),
        qwen08Rows8AffineCase(rows: 2_048, cols: 1_024),
        qwen08Rows8AffineCase(rows: 6_144, cols: 1_024),
        qwen08Rows8AffineCase(rows: 1_024, cols: 2_048),
        qwen08Rows8AffineCase(rows: 1_024, cols: 3_584),
        qwen08Rows8AffineCase(rows: 248_320, cols: 1_024),
        qwen08FullB1AffineCase(rows: 6_144, cols: 1_024),
        qwen08FullB1AffineCase(rows: 248_320, cols: 1_024),
        qwen08FullB1ResidualAddCase(),
        qwen08FullB1GateUpCase(),
        qwen08DeltaNetDecodeThreadgroupCase(rowsPerThreadgroup: 8),
        qwen08DeltaNetDecodeThreadgroupCase(rowsPerThreadgroup: 16),
        qwen08DeltaNetDecodeThreadgroupCase(rowsPerThreadgroup: 32),
        qwenQMMNoPadCase(rows: 2_048, cols: 1_024, batchSize: 64),
        qwenQMMNoPadCase(rows: 2_048, cols: 2_048, batchSize: 64),
        qwen08GenericGateUpCase(),
        qwen08NormScaleGateUpCase(),
        vibe3BGateUpRows8Case(),
        vibe3BGateUpRows4SG1Case(),
        vibe3BGateUpRows8SG2Case(),
        vibe3BGateUpRows4HalfAccumCase(),
        vibe3BGateUpFullB1Case(),
        vibe3BResidualAddRows8Case(rows: 2_048, cols: 11_008),
        vibe3BResidualAddRows8Case(rows: 256, cols: 2_048),
        vibe3BAffineRows8Case(rows: 2_048, cols: 2_048),
        vibe3BLMHeadFixedCase(),
        vibe3BLMHeadArgmaxCase(),
        qwen08LMHeadArgmaxCase(batchTile: 1),
        qwen08LMHeadArgmaxCase(batchTile: 4),
        qwen08ArgmaxSplitCase(chunkSize: 1_024),
        qwen08ArgmaxSplitCase(chunkSize: 2_048),
        qwen08ArgmaxSplitCase(chunkSize: 4_096),
    ]
    if name == "all" {
        return cases
    }
    guard let match = cases.first(where: { $0.name == name }) else {
        throw KernelLabError.unknownCase(name)
    }
    return [match]
}

private func qwen08DeltaNetDecodeThreadgroupCase(
    rowsPerThreadgroup: Int
) -> KernelLabCase {
    let headDim = 128
    let heads = 16
    return KernelLabCase(
        name: "qwen08-deltanet-decode-tgrows\(rowsPerThreadgroup)",
        buffers: [
            "state": halfBytes(heads * headDim * headDim),
            "qkv": halfBytes(3 * heads * headDim),
            "bProj": halfBytes(heads),
            "aProj": halfBytes(heads),
            "aLog": halfBytes(heads),
            "dtBias": halfBytes(heads),
            "output": halfBytes(heads * headDim),
        ],
        baseline: KernelLabVariant(
            name: "deltanet-tgrows4",
            dispatches: [
                KernelLabDispatch(
                    function: "deltanet_recurrence_mlx_decode_d128_h16",
                    buffers: [
                        0: "state", 1: "qkv", 2: "bProj", 3: "aProj",
                        4: "aLog", 5: "dtBias", 6: "output",
                    ],
                    constants: [:],
                    grid: .threads3D(
                        width: 32,
                        height: headDim,
                        depth: heads,
                        threadsPerThreadgroupWidth: 32,
                        threadsPerThreadgroupHeight: 4
                    )
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "deltanet-tgrows\(rowsPerThreadgroup)",
            dispatches: [
                KernelLabDispatch(
                    function: "deltanet_recurrence_mlx_decode_d128_h16",
                    buffers: [
                        0: "state", 1: "qkv", 2: "bProj", 3: "aProj",
                        4: "aLog", 5: "dtBias", 6: "output",
                    ],
                    constants: [:],
                    grid: .threads3D(
                        width: 32,
                        height: headDim,
                        depth: heads,
                        threadsPerThreadgroupWidth: 32,
                        threadsPerThreadgroupHeight: rowsPerThreadgroup
                    )
                ),
            ]
        )
    )
}

private func qwenQMMNoPadCase(
    rows: Int,
    cols: Int,
    batchSize: Int
) -> KernelLabCase {
    let group = 64
    let functionBase = "affine_matvec_c\(cols)_r\(rows)_g64_batched_full"
    return KernelLabCase(
        name: "qwen-qmm-nopad-c\(cols)-r\(rows)-b\(batchSize)",
        buffers: [
            "weights": packedU4Bytes(rows: rows, cols: cols),
            "scales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "biases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "input": halfBytes(cols * batchSize),
            "output": halfBytes(rows * batchSize),
        ],
        baseline: KernelLabVariant(
            name: "qmm-padded-reference",
            dispatches: [
                KernelLabDispatch(
                    function: functionBase + "_padded_reference",
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "output"],
                    constants: [5: .uint32(UInt32(batchSize))],
                    grid: .threadgroups2D(
                        width: (rows + 31) / 32,
                        height: (batchSize + 15) / 16,
                        threadsPerThreadgroup: 128
                    )
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "qmm-production-no-padding-stores",
            dispatches: [
                KernelLabDispatch(
                    function: functionBase,
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "output"],
                    constants: [5: .uint32(UInt32(batchSize))],
                    grid: .threadgroups2D(
                        width: (rows + 31) / 32,
                        height: (batchSize + 15) / 16,
                        threadsPerThreadgroup: 128
                    )
                ),
            ]
        )
    )
}

private func qwen08NormScaleGateUpCase() -> KernelLabCase {
    let rows = 3_584
    let cols = 1_024
    let group = 64
    return KernelLabCase(
        name: "qwen08-norm-scale-gate-up-c1024-r3584",
        buffers: [
            "normInput": halfBytes(cols),
            "normWeight": halfBytes(cols),
            "normOutput": halfBytes(cols),
            "normScale": MemoryLayout<Float>.stride,
            "gateWeights": packedU4Bytes(rows: rows, cols: cols),
            "gateScales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "gateBiases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "upWeights": packedU4Bytes(rows: rows, cols: cols),
            "upScales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "upBiases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "output": halfBytes(rows),
        ],
        baseline: KernelLabVariant(
            name: "rmsnorm+fixed-rows4-gate-up",
            dispatches: [
                KernelLabDispatch(
                    function: "rms_norm_1pw_d1024",
                    buffers: [0: "normInput", 1: "normWeight", 2: "normOutput"],
                    constants: [:],
                    grid: .threadgroups(width: 1, threadsPerThreadgroup: 128)
                ),
                KernelLabDispatch(
                    function: "fused_affine_gate_up_swiglu_c1024_r3584_g64_rows4",
                    buffers: [
                        0: "gateWeights",
                        1: "gateScales",
                        2: "gateBiases",
                        3: "upWeights",
                        4: "upScales",
                        5: "upBiases",
                        6: "normOutput",
                        7: "output",
                    ],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "scale-only+norm-scaled-generic-gate-up",
            dispatches: [
                KernelLabDispatch(
                    function: "rms_norm_scale_only",
                    buffers: [0: "normInput", 1: "normScale"],
                    constants: [2: .uint32(UInt32(cols)), 3: .float32(1e-6)],
                    grid: .threadgroups(width: 1, threadsPerThreadgroup: 128)
                ),
                KernelLabDispatch(
                    function: "norm_scale_affine_gate_up_swiglu",
                    functionConstants: [0: .uint32(UInt32(cols)), 1: .uint32(UInt32(group))],
                    buffers: [
                        0: "normScale",
                        1: "normInput",
                        2: "normWeight",
                        3: "gateWeights",
                        4: "gateScales",
                        5: "gateBiases",
                        6: "upWeights",
                        7: "upScales",
                        8: "upBiases",
                        9: "output",
                    ],
                    constants: [10: .uint32(UInt32(rows))],
                    grid: .threadgroups(width: rows / 8, threadsPerThreadgroup: 64)
                ),
            ]
        )
    )
}

private func qwen08ArgmaxSplitCase(chunkSize: Int) -> KernelLabCase {
    let vocab = 248_320
    let partials = (vocab + chunkSize - 1) / chunkSize
    return KernelLabCase(
        name: "qwen08-argmax-split-c\(chunkSize)",
        buffers: [
            "logits": halfBytes(vocab),
            "partials": partials * MemoryLayout<UInt32>.stride * 2,
            "argmax": MemoryLayout<Int32>.stride,
        ],
        baseline: KernelLabVariant(
            name: "argmax-single-tg",
            dispatches: [
                KernelLabDispatch(
                    function: "argmax_fp16",
                    buffers: [0: "logits", 1: "argmax"],
                    constants: [2: .uint32(UInt32(vocab))],
                    grid: .threadgroups(width: 1, threadsPerThreadgroup: 1_024)
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "argmax-split-c\(chunkSize)",
            dispatches: [
                KernelLabDispatch(
                    function: "argmax_fp16_partials",
                    buffers: [0: "logits", 1: "partials"],
                    constants: [
                        2: .uint32(UInt32(vocab)),
                        3: .uint32(UInt32(chunkSize)),
                    ],
                    grid: .threadgroups(width: partials, threadsPerThreadgroup: 256)
                ),
                KernelLabDispatch(
                    function: "argmax_key_reduce",
                    buffers: [0: "partials", 1: "argmax"],
                    constants: [2: .uint32(UInt32(partials))],
                    grid: .threadgroups(width: 1, threadsPerThreadgroup: 256)
                ),
            ]
        )
    )
}

private func qwen08LMHeadArgmaxCase(batchTile: Int) -> KernelLabCase {
    let rows = 248_320
    let cols = 1_024
    let group = 64
    let partials = rows / 8
    return KernelLabCase(
        name: "qwen08-lmhead-argmax-b\(batchTile)",
        buffers: [
            "weights": packedU4Bytes(rows: rows, cols: cols),
            "scales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "biases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "input": halfBytes(cols),
            "logits": halfBytes(rows),
            "partialKeys": partials * MemoryLayout<UInt32>.stride * 2,
            "argmax": MemoryLayout<Int32>.stride,
        ],
        baseline: KernelLabVariant(
            name: "lmhead+argmax",
            dispatches: [
                KernelLabDispatch(
                    function: "affine_matvec_c1024_r248320_g64",
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "logits"],
                    constants: [:],
                    grid: .threadgroups(width: rows / 8, threadsPerThreadgroup: 64)
                ),
                KernelLabDispatch(
                    function: "argmax_fp16",
                    buffers: [0: "logits", 1: "argmax"],
                    constants: [2: .uint32(UInt32(rows))],
                    grid: .threadgroups(width: 1, threadsPerThreadgroup: 1_024)
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "lmhead-argmax-fused-b\(batchTile)",
            dispatches: [
                KernelLabDispatch(
                    function: "affine_matvec_argmax_c1024_r248320_g64_b\(batchTile)",
                    buffers: [
                        0: "weights",
                        1: "scales",
                        2: "biases",
                        3: "input",
                        4: "partialKeys",
                    ],
                    constants: [5: .uint32(1), 6: .float32(0)],
                    grid: .threadgroups(width: rows / 8, threadsPerThreadgroup: 64)
                ),
                KernelLabDispatch(
                    function: "lm_head_argmax_reduce_r248320",
                    buffers: [0: "partialKeys", 1: "argmax"],
                    constants: [2: .uint32(1)],
                    grid: .threadgroups(width: 1, threadsPerThreadgroup: 256)
                ),
            ]
        )
    )
}

private func qwen08GenericGateUpCase() -> KernelLabCase {
    let rows = 3_584
    let cols = 1_024
    let group = 64
    return KernelLabCase(
        name: "qwen08-generic-gate-up-c1024-r3584",
        buffers: [
            "gateWeights": packedU4Bytes(rows: rows, cols: cols),
            "gateScales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "gateBiases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "upWeights": packedU4Bytes(rows: rows, cols: cols),
            "upScales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "upBiases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "input": halfBytes(cols),
            "output": halfBytes(rows),
        ],
        baseline: KernelLabVariant(
            name: "fixed-rows4-gate-up",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_gate_up_swiglu_c1024_r3584_g64_rows4",
                    buffers: [
                        0: "gateWeights",
                        1: "gateScales",
                        2: "gateBiases",
                        3: "upWeights",
                        4: "upScales",
                        5: "upBiases",
                        6: "input",
                        7: "output",
                    ],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "generic-fc-gate-up",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_gate_up_swiglu",
                    functionConstants: [0: .uint32(UInt32(cols)), 1: .uint32(UInt32(group))],
                    buffers: [
                        0: "gateWeights",
                        1: "gateScales",
                        2: "gateBiases",
                        3: "upWeights",
                        4: "upScales",
                        5: "upBiases",
                        6: "input",
                        7: "output",
                    ],
                    constants: [8: .uint32(UInt32(rows))],
                    grid: .threadgroups(width: rows / 8, threadsPerThreadgroup: 64)
                ),
            ]
        )
    )
}

private func vibe3BGateUpRows8Case() -> KernelLabCase {
    let rows = 11_008
    let cols = 2_048
    let group = 64
    return KernelLabCase(
        name: "vibe3b-gate-up-rows8-c2048-r11008",
        buffers: [
            "gateWeights": packedU4Bytes(rows: rows, cols: cols),
            "gateScales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "gateBiases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "upWeights": packedU4Bytes(rows: rows, cols: cols),
            "upScales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "upBiases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "input": halfBytes(cols),
            "output": halfBytes(rows),
        ],
        baseline: KernelLabVariant(
            name: "vibe3b-gate-up-rows4",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_gate_up_swiglu_c2048_r11008_g64",
                    buffers: [
                        0: "gateWeights",
                        1: "gateScales",
                        2: "gateBiases",
                        3: "upWeights",
                        4: "upScales",
                        5: "upBiases",
                        6: "input",
                        7: "output",
                    ],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "vibe3b-gate-up-rows8",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_gate_up_swiglu_c2048_r11008_g64_rows8",
                    buffers: [
                        0: "gateWeights",
                        1: "gateScales",
                        2: "gateBiases",
                        3: "upWeights",
                        4: "upScales",
                        5: "upBiases",
                        6: "input",
                        7: "output",
                    ],
                    constants: [:],
                    grid: .threadgroups(width: rows / 8, threadsPerThreadgroup: 64)
                ),
            ]
        )
    )
}

private func vibe3BGateUpRows4SG1Case() -> KernelLabCase {
    let rows = 11_008
    let cols = 2_048
    let group = 64
    return KernelLabCase(
        name: "vibe3b-gate-up-rows4-sg1-c2048-r11008",
        buffers: [
            "gateWeights": packedU4Bytes(rows: rows, cols: cols),
            "gateScales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "gateBiases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "upWeights": packedU4Bytes(rows: rows, cols: cols),
            "upScales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "upBiases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "input": halfBytes(cols),
            "output": halfBytes(rows),
        ],
        baseline: KernelLabVariant(
            name: "vibe3b-gate-up-rows4",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_gate_up_swiglu_c2048_r11008_g64",
                    buffers: [
                        0: "gateWeights",
                        1: "gateScales",
                        2: "gateBiases",
                        3: "upWeights",
                        4: "upScales",
                        5: "upBiases",
                        6: "input",
                        7: "output",
                    ],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "vibe3b-gate-up-rows4-sg1",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_gate_up_swiglu_c2048_r11008_g64_rows4_sg1",
                    buffers: [
                        0: "gateWeights",
                        1: "gateScales",
                        2: "gateBiases",
                        3: "upWeights",
                        4: "upScales",
                        5: "upBiases",
                        6: "input",
                        7: "output",
                    ],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 128)
                ),
            ]
        )
    )
}

private func vibe3BGateUpRows8SG2Case() -> KernelLabCase {
    let rows = 11_008
    let cols = 2_048
    let group = 64
    return KernelLabCase(
        name: "vibe3b-gate-up-rows8-sg2-c2048-r11008",
        buffers: [
            "gateWeights": packedU4Bytes(rows: rows, cols: cols),
            "gateScales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "gateBiases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "upWeights": packedU4Bytes(rows: rows, cols: cols),
            "upScales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "upBiases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "input": halfBytes(cols),
            "output": halfBytes(rows),
        ],
        baseline: KernelLabVariant(
            name: "vibe3b-gate-up-rows4",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_gate_up_swiglu_c2048_r11008_g64",
                    buffers: [
                        0: "gateWeights",
                        1: "gateScales",
                        2: "gateBiases",
                        3: "upWeights",
                        4: "upScales",
                        5: "upBiases",
                        6: "input",
                        7: "output",
                    ],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "vibe3b-gate-up-rows8-sg2",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_gate_up_swiglu_c2048_r11008_g64_rows8_sg2",
                    buffers: [
                        0: "gateWeights",
                        1: "gateScales",
                        2: "gateBiases",
                        3: "upWeights",
                        4: "upScales",
                        5: "upBiases",
                        6: "input",
                        7: "output",
                    ],
                    constants: [:],
                    grid: .threadgroups(width: rows / 8, threadsPerThreadgroup: 128)
                ),
            ]
        )
    )
}

private func vibe3BGateUpRows4HalfAccumCase() -> KernelLabCase {
    let rows = 11_008
    let cols = 2_048
    let group = 64
    return KernelLabCase(
        name: "vibe3b-gate-up-rows4-hacc-c2048-r11008",
        buffers: [
            "gateWeights": packedU4Bytes(rows: rows, cols: cols),
            "gateScales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "gateBiases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "upWeights": packedU4Bytes(rows: rows, cols: cols),
            "upScales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "upBiases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "input": halfBytes(cols),
            "output": halfBytes(rows),
        ],
        baseline: KernelLabVariant(
            name: "vibe3b-gate-up-rows4",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_gate_up_swiglu_c2048_r11008_g64",
                    buffers: [
                        0: "gateWeights",
                        1: "gateScales",
                        2: "gateBiases",
                        3: "upWeights",
                        4: "upScales",
                        5: "upBiases",
                        6: "input",
                        7: "output",
                    ],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "vibe3b-gate-up-rows4-hacc",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_gate_up_swiglu_c2048_r11008_g64_rows4_hacc",
                    buffers: [
                        0: "gateWeights",
                        1: "gateScales",
                        2: "gateBiases",
                        3: "upWeights",
                        4: "upScales",
                        5: "upBiases",
                        6: "input",
                        7: "output",
                    ],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
            ]
        )
    )
}

private func vibe3BGateUpFullB1Case() -> KernelLabCase {
    let rows = 11_008
    let cols = 2_048
    let group = 64
    return KernelLabCase(
        name: "vibe3b-gate-up-full-b1-c2048-r11008",
        buffers: [
            "gateWeights": packedU4Bytes(rows: rows, cols: cols),
            "gateScales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "gateBiases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "upWeights": packedU4Bytes(rows: rows, cols: cols),
            "upScales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "upBiases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "input": halfBytes(cols * 16),
            "output": halfBytes(rows * 16),
        ],
        baseline: KernelLabVariant(
            name: "vibe3b-gate-up-rows4",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_gate_up_swiglu_c2048_r11008_g64",
                    buffers: [
                        0: "gateWeights",
                        1: "gateScales",
                        2: "gateBiases",
                        3: "upWeights",
                        4: "upScales",
                        5: "upBiases",
                        6: "input",
                        7: "output",
                    ],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "vibe3b-gate-up-full-qmm-b1",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_gate_up_swiglu_c2048_r11008_g64_batched_full",
                    buffers: [
                        0: "gateWeights",
                        1: "gateScales",
                        2: "gateBiases",
                        3: "upWeights",
                        4: "upScales",
                        5: "upBiases",
                        6: "input",
                        7: "output",
                    ],
                    constants: [8: .uint32(1)],
                    grid: .threadgroups(width: (rows + 31) / 32, threadsPerThreadgroup: 128)
                ),
            ]
        )
    )
}

private func vibe3BResidualAddRows8Case(rows: Int, cols: Int) -> KernelLabCase {
    let group = 64
    return KernelLabCase(
        name: "vibe3b-residual-add-rows8-c\(cols)-r\(rows)",
        buffers: [
            "weights": packedU4Bytes(rows: rows, cols: cols),
            "scales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "biases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "input": halfBytes(cols),
            "residual": halfBytes(rows),
            "output": halfBytes(rows),
        ],
        baseline: KernelLabVariant(
            name: "vibe3b-residual-add-rows4",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_matvec_add_c\(cols)_r\(rows)_g64_rows4",
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "output", 5: "residual"],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "vibe3b-residual-add-rows8",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_matvec_add_c\(cols)_r\(rows)_g64_rows8",
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "output", 5: "residual"],
                    constants: [:],
                    grid: .threadgroups(width: rows / 8, threadsPerThreadgroup: 64)
                ),
            ]
        )
    )
}

private func vibe3BAffineRows8Case(rows: Int, cols: Int) -> KernelLabCase {
    let group = 64
    return KernelLabCase(
        name: "vibe3b-affine-rows8-c\(cols)-r\(rows)",
        buffers: [
            "weights": packedU4Bytes(rows: rows, cols: cols),
            "scales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "biases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "input": halfBytes(cols),
            "output": halfBytes(rows),
        ],
        baseline: KernelLabVariant(
            name: "vibe3b-affine-rows4",
            dispatches: [
                KernelLabDispatch(
                    function: "affine_matvec_c\(cols)_r\(rows)_g64",
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "output"],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "vibe3b-affine-rows8",
            dispatches: [
                KernelLabDispatch(
                    function: "affine_matvec_c\(cols)_r\(rows)_g64_rows8",
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "output"],
                    constants: [:],
                    grid: .threadgroups(width: rows / 8, threadsPerThreadgroup: 64)
                ),
            ]
        )
    )
}

private func vibe3BLMHeadFixedCase() -> KernelLabCase {
    let rows = 151_936
    let cols = 2_048
    let group = 64
    return KernelLabCase(
        name: "vibe3b-lmhead-fixed-c2048-r151936",
        buffers: [
            "weights": packedU4Bytes(rows: rows, cols: cols),
            "scales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "biases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "input": halfBytes(cols),
            "logits": halfBytes(rows),
        ],
        baseline: KernelLabVariant(
            name: "generic-affine",
            dispatches: [
                KernelLabDispatch(
                    function: "affine_matvec",
                    functionConstants: [0: .uint32(UInt32(cols)), 1: .uint32(UInt32(group))],
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "logits"],
                    constants: [5: .uint32(UInt32(rows))],
                    grid: .threadgroups(width: rows / 8, threadsPerThreadgroup: 64)
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "fixed-affine-rows8",
            dispatches: [
                KernelLabDispatch(
                    function: "affine_matvec_c2048_r151936_g64_rows8",
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "logits"],
                    constants: [:],
                    grid: .threadgroups(width: rows / 8, threadsPerThreadgroup: 64)
                ),
            ]
        )
    )
}

private func vibe3BLMHeadArgmaxCase() -> KernelLabCase {
    let rows = 151_936
    let cols = 2_048
    let group = 64
    let baselineChunk = 2_048
    let baselinePartials = (rows + baselineChunk - 1) / baselineChunk
    let fusedPartials = rows / 8
    return KernelLabCase(
        name: "vibe3b-lmhead-argmax-c2048-r151936",
        buffers: [
            "weights": packedU4Bytes(rows: rows, cols: cols),
            "scales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "biases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "input": halfBytes(cols),
            "logits": halfBytes(rows),
            "baselinePartials": baselinePartials * MemoryLayout<UInt32>.stride * 2,
            "fusedPartials": fusedPartials * MemoryLayout<UInt32>.stride * 2,
            "argmax": MemoryLayout<Int32>.stride,
        ],
        baseline: KernelLabVariant(
            name: "generic-lmhead+split-argmax",
            dispatches: [
                KernelLabDispatch(
                    function: "affine_matvec",
                    functionConstants: [0: .uint32(UInt32(cols)), 1: .uint32(UInt32(group))],
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "logits"],
                    constants: [5: .uint32(UInt32(rows))],
                    grid: .threadgroups(width: rows / 8, threadsPerThreadgroup: 64)
                ),
                KernelLabDispatch(
                    function: "argmax_fp16_partials",
                    buffers: [0: "logits", 1: "baselinePartials"],
                    constants: [
                        2: .uint32(UInt32(rows)),
                        3: .uint32(UInt32(baselineChunk)),
                    ],
                    grid: .threadgroups(width: baselinePartials, threadsPerThreadgroup: 256)
                ),
                KernelLabDispatch(
                    function: "argmax_key_reduce",
                    buffers: [0: "baselinePartials", 1: "argmax"],
                    constants: [2: .uint32(UInt32(baselinePartials))],
                    grid: .threadgroups(width: 1, threadsPerThreadgroup: 256)
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "lmhead-argmax-fused-b1",
            dispatches: [
                KernelLabDispatch(
                    function: "affine_matvec_argmax_c2048_r151936_g64_b1",
                    buffers: [
                        0: "weights",
                        1: "scales",
                        2: "biases",
                        3: "input",
                        4: "fusedPartials",
                    ],
                    constants: [5: .uint32(1), 6: .float32(0)],
                    grid: .threadgroups(width: fusedPartials, threadsPerThreadgroup: 64)
                ),
                KernelLabDispatch(
                    function: "lm_head_argmax_reduce_r151936",
                    buffers: [0: "fusedPartials", 1: "argmax"],
                    constants: [2: .uint32(1)],
                    grid: .threadgroups(width: 1, threadsPerThreadgroup: 256)
                ),
            ]
        )
    )
}

private func qwen08FullB1AffineCase(rows: Int, cols: Int) -> KernelLabCase {
    let group = 64
    return KernelLabCase(
        name: "qwen08-full-b1-c\(cols)-r\(rows)",
        buffers: [
            "weights": packedU4Bytes(rows: rows, cols: cols),
            "scales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "biases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "input": halfBytes(cols * 16),
            "output": halfBytes(rows * 16),
        ],
        baseline: KernelLabVariant(
            name: "decode-rows4",
            dispatches: [
                KernelLabDispatch(
                    function: "affine_matvec_c\(cols)_r\(rows)_g64_rows4",
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "output"],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "full-qmm-b1",
            dispatches: [
                KernelLabDispatch(
                    function: "affine_matvec_c\(cols)_r\(rows)_g64_batched_full",
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "output"],
                    constants: [5: .uint32(1)],
                    grid: .threadgroups(width: (rows + 31) / 32, threadsPerThreadgroup: 128)
                ),
            ]
        )
    )
}

private func qwen08FullB1ResidualAddCase() -> KernelLabCase {
    let rows = 1_024
    let cols = 3_584
    let group = 64
    return KernelLabCase(
        name: "qwen08-full-b1-residual-add-c3584-r1024",
        buffers: [
            "weights": packedU4Bytes(rows: rows, cols: cols),
            "scales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "biases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "input": halfBytes(cols * 16),
            "residual": halfBytes(rows * 16),
            "matvecOut": halfBytes(rows * 16),
            "output": halfBytes(rows * 16),
        ],
        baseline: KernelLabVariant(
            name: "decode-fused-rows4",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_matvec_add_c3584_r1024_g64_rows4",
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "output", 5: "residual"],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "full-qmm-b1-fused-add",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_matvec_add_c3584_r1024_g64_batched_full",
                    buffers: [
                        0: "weights",
                        1: "scales",
                        2: "biases",
                        3: "input",
                        4: "matvecOut",
                        5: "residual",
                        6: "output",
                    ],
                    constants: [7: .uint32(1)],
                    grid: .threadgroups(width: (rows + 31) / 32, threadsPerThreadgroup: 128)
                ),
            ]
        )
    )
}

private func qwen08FullB1GateUpCase() -> KernelLabCase {
    let rows = 3_584
    let cols = 1_024
    let group = 64
    return KernelLabCase(
        name: "qwen08-full-b1-gate-up-c1024-r3584",
        buffers: [
            "gateWeights": packedU4Bytes(rows: rows, cols: cols),
            "gateScales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "gateBiases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "upWeights": packedU4Bytes(rows: rows, cols: cols),
            "upScales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "upBiases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "input": halfBytes(cols * 16),
            "output": halfBytes(rows * 16),
        ],
        baseline: KernelLabVariant(
            name: "decode-gate-up-rows4",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_gate_up_swiglu_c1024_r3584_g64_rows4",
                    buffers: [
                        0: "gateWeights",
                        1: "gateScales",
                        2: "gateBiases",
                        3: "upWeights",
                        4: "upScales",
                        5: "upBiases",
                        6: "input",
                        7: "output",
                    ],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "full-qmm-b1-gate-up",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_gate_up_swiglu_c1024_r3584_g64_batched_full",
                    buffers: [
                        0: "gateWeights",
                        1: "gateScales",
                        2: "gateBiases",
                        3: "upWeights",
                        4: "upScales",
                        5: "upBiases",
                        6: "input",
                        7: "output",
                    ],
                    constants: [8: .uint32(1)],
                    grid: .threadgroups(width: (rows + 31) / 32, threadsPerThreadgroup: 128)
                ),
            ]
        )
    )
}

private func qwen08Rows8AffineCase(rows: Int, cols: Int) -> KernelLabCase {
    let group = 64
    let baselineIsCurrentRows8 = rows == 248_320 && cols == 1_024
    let baselineFunction = baselineIsCurrentRows8
        ? "affine_matvec_c\(cols)_r\(rows)_g64"
        : "affine_matvec_c\(cols)_r\(rows)_g64_rows4"
    let baselineTile = baselineIsCurrentRows8 ? 8 : 4
    let baselineName = baselineIsCurrentRows8 ? "current-affine" : "affine-rows4"
    return KernelLabCase(
        name: "qwen08-rows8-c\(cols)-r\(rows)",
        buffers: [
            "weights": packedU4Bytes(rows: rows, cols: cols),
            "scales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "biases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "input": halfBytes(cols),
            "output": halfBytes(rows),
        ],
        baseline: KernelLabVariant(
            name: baselineName,
            dispatches: [
                KernelLabDispatch(
                    function: baselineFunction,
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "output"],
                    constants: [:],
                    grid: .threadgroups(width: rows / baselineTile, threadsPerThreadgroup: 64)
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "affine-rows8",
            dispatches: [
                KernelLabDispatch(
                    function: "affine_matvec_c\(cols)_r\(rows)_g64_rows8",
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "output"],
                    constants: [:],
                    grid: .threadgroups(width: rows / 8, threadsPerThreadgroup: 64)
                ),
            ]
        )
    )
}

private func qwen08ResidualAddCase() -> KernelLabCase {
    let rows = 1_024
    let cols = 3_584
    let group = 64
    return KernelLabCase(
        name: "qwen08-residual-add",
        buffers: [
            "weights": packedU4Bytes(rows: rows, cols: cols),
            "scales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "biases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "input": halfBytes(cols),
            "residual": halfBytes(rows),
            "matvecOut": halfBytes(rows),
            "output": halfBytes(rows),
        ],
        baseline: KernelLabVariant(
            name: "rows4-matvec+add",
            dispatches: [
                KernelLabDispatch(
                    function: "affine_matvec_c3584_r1024_g64_rows4",
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "matvecOut"],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
                KernelLabDispatch(
                    function: "elementwise_add",
                    buffers: [0: "matvecOut", 1: "residual", 2: "output"],
                    constants: [3: .uint32(UInt32(rows))],
                    grid: .threads(width: rows, threadsPerThreadgroup: min(rows, 1024))
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "rows4-fused-matvec-add",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_matvec_add_c3584_r1024_g64_rows4",
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "output", 5: "residual"],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
            ]
        )
    )
}

private func qwen08ResidualAddC2048Case() -> KernelLabCase {
    let rows = 1_024
    let cols = 2_048
    let group = 64
    return KernelLabCase(
        name: "qwen08-residual-add-c2048",
        buffers: [
            "weights": packedU4Bytes(rows: rows, cols: cols),
            "scales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "biases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "input": halfBytes(cols),
            "residual": halfBytes(rows),
            "matvecOut": halfBytes(rows),
            "output": halfBytes(rows),
        ],
        baseline: KernelLabVariant(
            name: "rows4-matvec+add",
            dispatches: [
                KernelLabDispatch(
                    function: "affine_matvec_c2048_r1024_g64_rows4",
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "matvecOut"],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
                KernelLabDispatch(
                    function: "elementwise_add",
                    buffers: [0: "matvecOut", 1: "residual", 2: "output"],
                    constants: [3: .uint32(UInt32(rows))],
                    grid: .threads(width: rows, threadsPerThreadgroup: min(rows, 1024))
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "rows4-fused-matvec-add",
            dispatches: [
                KernelLabDispatch(
                    function: "fused_affine_matvec_add_c2048_r1024_g64_rows4",
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "input", 4: "output", 5: "residual"],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
            ]
        )
    )
}

private func qwen08DeltaTailC2048x18Case() -> KernelLabCase {
    let layers = 18
    let rows = 1_024
    let cols = 2_048
    let group = 64
    var baselineDispatches: [KernelLabDispatch] = []
    var candidateDispatches: [KernelLabDispatch] = []

    for layer in 0..<layers {
        let current = layer.isMultiple(of: 2) ? "hiddenA" : "hiddenB"
        let next = layer.isMultiple(of: 2) ? "hiddenB" : "hiddenA"
        let norm = KernelLabDispatch(
            function: "rms_norm_gated_d128",
            buffers: [0: "recOut", 1: "z", 2: "normWeight", 3: "recOut"],
            constants: [:],
            grid: .threadgroups(width: 16, threadsPerThreadgroup: 128)
        )
        baselineDispatches.append(norm)
        baselineDispatches.append(KernelLabDispatch(
            function: "affine_matvec_c2048_r1024_g64_rows4",
            buffers: [0: "weights", 1: "scales", 2: "biases", 3: "recOut", 4: "projOut"],
            constants: [:],
            grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
        ))
        baselineDispatches.append(KernelLabDispatch(
            function: "elementwise_add",
            buffers: [0: current, 1: "projOut", 2: next],
            constants: [3: .uint32(UInt32(rows))],
            grid: .threads(width: rows, threadsPerThreadgroup: min(rows, 1024))
        ))

        candidateDispatches.append(norm)
        candidateDispatches.append(KernelLabDispatch(
            function: "fused_affine_matvec_add_c2048_r1024_g64_rows4",
            buffers: [0: "weights", 1: "scales", 2: "biases", 3: "recOut", 4: next, 5: current],
            constants: [:],
            grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
        ))
    }

    return KernelLabCase(
        name: "qwen08-delta-tail-c2048-x18",
        buffers: [
            "weights": packedU4Bytes(rows: rows, cols: cols),
            "scales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "biases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "recOut": halfBytes(cols),
            "z": halfBytes(cols),
            "normWeight": halfBytes(128),
            "hiddenA": halfBytes(rows),
            "hiddenB": halfBytes(rows),
            "projOut": halfBytes(rows),
        ],
        baseline: KernelLabVariant(
            name: "18x-gated-norm+matvec+add",
            dispatches: baselineDispatches
        ),
        candidate: KernelLabVariant(
            name: "18x-gated-norm+fused-matvec-add",
            dispatches: candidateDispatches
        )
    )
}

private func qwen08NormScaleC6144Case() -> KernelLabCase {
    let rows = 6_144
    let cols = 1_024
    let group = 64
    return KernelLabCase(
        name: "qwen08-norm-scale-c6144",
        buffers: [
            "normInput": halfBytes(cols),
            "normWeight": halfBytes(cols),
            "normOutput": halfBytes(cols),
            "scale": MemoryLayout<Float>.stride,
            "weights": packedU4Bytes(rows: rows, cols: cols),
            "scales": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "biases": scaleBiasBytes(rows: rows, cols: cols, group: group),
            "output": halfBytes(rows),
        ],
        baseline: KernelLabVariant(
            name: "rms-norm+rows4-matvec",
            dispatches: [
                KernelLabDispatch(
                    function: "rms_norm_1pw_d1024",
                    buffers: [0: "normInput", 1: "normWeight", 2: "normOutput"],
                    constants: [:],
                    grid: .threadgroups(width: 1, threadsPerThreadgroup: 128)
                ),
                KernelLabDispatch(
                    function: "affine_matvec_c1024_r6144_g64_rows4",
                    buffers: [0: "weights", 1: "scales", 2: "biases", 3: "normOutput", 4: "output"],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
            ]
        ),
        candidate: KernelLabVariant(
            name: "scale-only+norm-scale-rows4-matvec",
            dispatches: [
                KernelLabDispatch(
                    function: "rms_norm_scale_only",
                    buffers: [0: "normInput", 1: "scale"],
                    constants: [2: .uint32(UInt32(cols)), 3: .float32(1e-6)],
                    grid: .threadgroups(width: 1, threadsPerThreadgroup: 1_024)
                ),
                KernelLabDispatch(
                    function: "norm_scale_affine_matvec_c1024_r6144_g64_rows4",
                    buffers: [
                        0: "scale",
                        1: "normInput",
                        2: "normWeight",
                        3: "normOutput",
                        4: "weights",
                        5: "scales",
                        6: "biases",
                        7: "output",
                    ],
                    constants: [:],
                    grid: .threadgroups(width: rows / 4, threadsPerThreadgroup: 64)
                ),
            ]
        )
    )
}

private func halfBytes(_ elements: Int) -> Int {
    elements * 2
}

private func packedU4Bytes(rows: Int, cols: Int) -> Int {
    rows * cols / 2
}

private func scaleBiasBytes(rows: Int, cols: Int, group: Int) -> Int {
    rows * (cols / group) * 2
}
