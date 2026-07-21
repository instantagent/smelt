import Foundation
import SmeltSchema

public struct SmeltOptimizerReportOptions: Sendable {
    public var verifyWeights: Bool
    public var topPipelines: Int
    public var maxAgentTasks: Int
    public var includeTimings: Bool
    public var timingIterations: Int
    public var timingDecodePosition: Int32
    public var timingPrefillTokens: [Int]
    public var timingTopKernels: Int
    public var planComparisons: [SmeltPlanComparisonReport]

    public init(
        verifyWeights: Bool = false,
        topPipelines: Int = 8,
        maxAgentTasks: Int = 8,
        includeTimings: Bool = false,
        timingIterations: Int = 1,
        timingDecodePosition: Int32 = 0,
        timingPrefillTokens: [Int] = [64],
        timingTopKernels: Int = 8,
        planComparisons: [SmeltPlanComparisonReport] = []
    ) {
        self.verifyWeights = verifyWeights
        self.topPipelines = topPipelines
        self.maxAgentTasks = maxAgentTasks
        self.includeTimings = includeTimings
        self.timingIterations = max(timingIterations, 1)
        self.timingDecodePosition = max(timingDecodePosition, 0)
        self.timingPrefillTokens = timingPrefillTokens.filter { $0 > 0 }
        self.timingTopKernels = max(timingTopKernels, 0)
        self.planComparisons = planComparisons
    }
}

public struct SmeltOptimizerTimingEvidence: Sendable {
    public let decode: SmeltOptimizerModeTiming?
    public let prefill: [SmeltOptimizerModeTiming]
    public let warnings: [String]

    public init(
        decode: SmeltOptimizerModeTiming?,
        prefill: [SmeltOptimizerModeTiming],
        warnings: [String] = []
    ) {
        self.decode = decode
        self.prefill = prefill
        self.warnings = warnings
    }

    public func preferredTiming(for mode: String) -> SmeltOptimizerModeTiming? {
        if mode == "decode" {
            return decode
        }
        if mode == "prefill" {
            return prefill.max { lhs, rhs in lhs.tokenCount < rhs.tokenCount }
        }
        return nil
    }
}

public struct SmeltOptimizerModeTiming: Sendable {
    public let mode: String
    public let iterations: Int
    public let tokenCount: Int
    public let position: Int32?
    public let totalGpuUs: Double
    public let kernels: [SmeltOptimizerKernelTiming]

    public init(
        mode: String,
        iterations: Int,
        tokenCount: Int,
        position: Int32?,
        totalGpuUs: Double,
        kernels: [SmeltOptimizerKernelTiming]
    ) {
        self.mode = mode
        self.iterations = iterations
        self.tokenCount = tokenCount
        self.position = position
        self.totalGpuUs = totalGpuUs
        self.kernels = kernels
    }
}

public struct SmeltOptimizerKernelTiming: Sendable {
    public let name: String
    public let dispatchCount: Int
    public let avgGpuUs: Double
    public let pctOfTotal: Double

    public init(
        name: String,
        dispatchCount: Int,
        avgGpuUs: Double,
        pctOfTotal: Double
    ) {
        self.name = name
        self.dispatchCount = dispatchCount
        self.avgGpuUs = avgGpuUs
        self.pctOfTotal = pctOfTotal
    }
}

public struct SmeltOptimizerCAMReportContext: Sendable, Equatable {
    public let camSemanticSHA256: String
    public let exportABISHA256: String
    public let descriptorGraphSignatureSHA256: String
    public let exportID: String
    public let flowID: String
    public let matchedGateIDs: [String]
    public let authoredCapabilities: [String]

    public init(
        camSemanticSHA256: String,
        exportABISHA256: String,
        descriptorGraphSignatureSHA256: String,
        exportID: String,
        flowID: String,
        matchedGateIDs: [String],
        authoredCapabilities: [String]
    ) {
        self.camSemanticSHA256 = camSemanticSHA256
        self.exportABISHA256 = exportABISHA256
        self.descriptorGraphSignatureSHA256 = descriptorGraphSignatureSHA256
        self.exportID = exportID
        self.flowID = flowID
        self.matchedGateIDs = matchedGateIDs
        self.authoredCapabilities = authoredCapabilities
    }
}

public enum SmeltOptimizerReportGenerator {
    public static func agentTasks(packagePath: String) throws -> [SmeltOptimizerAgentTask] {
        let manifest = try loadManifest(packagePath: packagePath)
        return agentTasks(from: manifest)
    }

    public static func agentTasks(from manifest: SmeltManifest) -> [SmeltOptimizerAgentTask] {
        SmeltOptimizerAgentTask.tasks(from: manifest)
    }

    public static func markdown(
        packagePath: String,
        options: SmeltOptimizerReportOptions = SmeltOptimizerReportOptions()
    ) throws -> String {
        try markdown(
            packagePath: packagePath,
            camContext: nil,
            options: options
        )
    }

    public static func markdown(
        packagePath: String,
        camContext: SmeltOptimizerCAMReportContext?,
        options: SmeltOptimizerReportOptions = SmeltOptimizerReportOptions()
    ) throws -> String {
        let manifest = try loadManifest(packagePath: packagePath)
        let integrity = try SmeltPackageIntegrity.verify(
            packagePath: packagePath,
            manifest: manifest,
            includeWeights: options.verifyWeights
        )
        let decodeReport = try SmeltPackageStructure.inspectDecode(packagePath: packagePath)
        let prefillReport = try SmeltPackageStructure.inspectPrefill(
            packagePath: packagePath,
            manifest: manifest
        )
        let verifyArgmaxReport = try SmeltPackageStructure.inspectPrefillVerifyArgmax(
            packagePath: packagePath
        )
        let timingEvidence = collectTimingEvidence(
            packagePath: packagePath,
            options: options
        )
        return markdown(
            packagePath: packagePath,
            manifest: manifest,
            integrity: integrity,
            decodeReport: decodeReport,
            prefillReport: prefillReport,
            verifyArgmaxReport: verifyArgmaxReport,
            timingEvidence: timingEvidence,
            camContext: camContext,
            options: options
        )
    }

    public static func writeMarkdown(
        packagePath: String,
        outputPath: String,
        options: SmeltOptimizerReportOptions = SmeltOptimizerReportOptions()
    ) throws {
        let report = try markdown(packagePath: packagePath, options: options)
        try report.write(toFile: outputPath, atomically: true, encoding: .utf8)
    }

    static func markdown(
        packagePath: String,
        manifest: SmeltManifest,
        integrity: SmeltPackageIntegrityReport?,
        decodeReport: SmeltDispatchStructureReport?,
        prefillReport: SmeltDispatchStructureReport?,
        verifyArgmaxReport: SmeltDispatchStructureReport? = nil,
        timingEvidence: SmeltOptimizerTimingEvidence? = nil,
        camContext: SmeltOptimizerCAMReportContext? = nil,
        options: SmeltOptimizerReportOptions
    ) -> String {
        let missingTasks = SmeltOptimizerAgentTask.tasks(from: manifest)

        var lines: [String] = []
        lines.append("# Smelt Optimizer Report")
        lines.append("")
        appendVerdict(
            to: &lines,
            manifest: manifest,
            camContext: camContext,
            missingTasks: missingTasks
        )
        appendPackage(
            to: &lines,
            packagePath: packagePath,
            manifest: manifest,
            camContext: camContext,
            integrity: integrity
        )
        appendStructureCoverage(
            to: &lines,
            decodeReport: decodeReport,
            prefillReport: prefillReport,
            topPipelines: options.topPipelines
        )
        appendMeasuredTimingEvidence(
            to: &lines,
            timingEvidence: timingEvidence,
            options: options
        )
        appendPlanCostEvidence(
            to: &lines,
            comparisons: options.planComparisons
        )
        appendCompilerOptimizationSummary(
            to: &lines,
            report: manifest.optimizationReport,
            vocabSize: manifest.config.vocabSize
        )
        appendCompilerStrategyDiagnosis(
            to: &lines,
            manifest: manifest,
            decodeReport: decodeReport,
            prefillReport: prefillReport,
            timingEvidence: timingEvidence,
            tasks: missingTasks,
            options: options
        )
        appendTimingBackedStructuralWorkQueue(
            to: &lines,
            timingEvidence: timingEvidence
        )
        appendAgentWorkQueue(
            to: &lines,
            tasks: missingTasks,
            maxTasks: options.maxAgentTasks
        )
        appendNotes(to: &lines)
        return lines.joined(separator: "\n") + "\n"
    }

    private static func loadManifest(packagePath: String) throws -> SmeltManifest {
        let manifestPath = "\(packagePath)/manifest.json"
        let data = try Data(contentsOf: URL(fileURLWithPath: manifestPath))
        return try SmeltManifest.decode(from: data)
    }

    private static func collectTimingEvidence(
        packagePath: String,
        options: SmeltOptimizerReportOptions
    ) -> SmeltOptimizerTimingEvidence? {
        guard options.includeTimings else { return nil }

        var warnings: [String] = []
        do {
            let runtime = try SmeltRuntime(packagePath: packagePath)
            let decode = try collectDecodeTiming(
                runtime: runtime,
                iterations: options.timingIterations,
                position: options.timingDecodePosition
            )

            var prefillTimings: [SmeltOptimizerModeTiming] = []
            if runtime.hasMetalPrefill {
                let requestedTokenCounts = options.timingPrefillTokens.isEmpty
                    ? [min(runtime.maxPrefillBatchSize, 64)]
                    : options.timingPrefillTokens
                for requestedTokenCount in requestedTokenCounts {
                    let tokenCount = min(requestedTokenCount, runtime.maxPrefillBatchSize)
                    if tokenCount != requestedTokenCount {
                        warnings.append(
                            "Requested prefill timing for \(requestedTokenCount) tokens, but the package max prefill batch is \(runtime.maxPrefillBatchSize); measured \(tokenCount) tokens instead."
                        )
                    }
                    guard tokenCount > 0 else { continue }
                    prefillTimings.append(
                        try collectPrefillTiming(
                            runtime: runtime,
                            iterations: options.timingIterations,
                            tokenCount: tokenCount
                        )
                    )
                }
            } else {
                warnings.append("Package has no Metal prefill dispatch table, so prefill timings were not collected.")
            }

            return SmeltOptimizerTimingEvidence(
                decode: decode,
                prefill: prefillTimings,
                warnings: warnings
            )
        } catch {
            return SmeltOptimizerTimingEvidence(
                decode: nil,
                prefill: [],
                warnings: ["Timing collection failed: \(error)."]
            )
        }
    }

    private static func collectDecodeTiming(
        runtime: SmeltRuntime,
        iterations: Int,
        position: Int32
    ) throws -> SmeltOptimizerModeTiming {
        var accumulated: [String: (totalUs: Double, count: Int, dispatches: Int)] = [:]
        for _ in 0..<max(iterations, 1) {
            let results = try runtime.profileKernels(tokenId: 0, position: position)
            for result in results {
                let previous = accumulated[result.name, default: (0, 0, result.dispatchCount)]
                accumulated[result.name] = (
                    previous.totalUs + result.totalGpuUs,
                    previous.count + 1,
                    result.dispatchCount
                )
            }
        }
        return makeModeTiming(
            mode: "decode",
            iterations: max(iterations, 1),
            tokenCount: 1,
            position: position,
            accumulated: accumulated
        )
    }

    private static func collectPrefillTiming(
        runtime: SmeltRuntime,
        iterations: Int,
        tokenCount: Int
    ) throws -> SmeltOptimizerModeTiming {
        let safeTokenCount = max(tokenCount, 1)
        let tokenIds = (0..<safeTokenCount).map { Int32($0) }

        runtime.resetWorkingBuffers()
        _ = try runtime.prefillStep(tokenIds: tokenIds, startPos: 0)
        runtime.resetWorkingBuffers()

        var accumulated: [String: (totalUs: Double, count: Int, dispatches: Int)] = [:]
        for _ in 0..<max(iterations, 1) {
            runtime.resetWorkingBuffers()
            let results = try runtime.profilePrefillKernels(tokenIds: tokenIds, startPos: 0)
            for result in results {
                let previous = accumulated[result.name, default: (0, 0, result.dispatchCount)]
                accumulated[result.name] = (
                    previous.totalUs + result.totalGpuUs,
                    previous.count + 1,
                    result.dispatchCount
                )
            }
        }

        return makeModeTiming(
            mode: "prefill",
            iterations: max(iterations, 1),
            tokenCount: safeTokenCount,
            position: nil,
            accumulated: accumulated
        )
    }

    private static func makeModeTiming(
        mode: String,
        iterations: Int,
        tokenCount: Int,
        position: Int32?,
        accumulated: [String: (totalUs: Double, count: Int, dispatches: Int)]
    ) -> SmeltOptimizerModeTiming {
        let grandTotal = accumulated.values.reduce(0.0) { partial, value in
            partial + (value.totalUs / Double(max(value.count, 1)))
        }
        let kernels = accumulated.map { name, value in
            let avgUs = value.totalUs / Double(max(value.count, 1))
            return SmeltOptimizerKernelTiming(
                name: name,
                dispatchCount: value.dispatches,
                avgGpuUs: avgUs,
                pctOfTotal: grandTotal > 0 ? (avgUs / grandTotal) * 100.0 : 0
            )
        }
        .sorted { lhs, rhs in
            if lhs.avgGpuUs != rhs.avgGpuUs {
                return lhs.avgGpuUs > rhs.avgGpuUs
            }
            return lhs.name < rhs.name
        }

        return SmeltOptimizerModeTiming(
            mode: mode,
            iterations: iterations,
            tokenCount: tokenCount,
            position: position,
            totalGpuUs: grandTotal,
            kernels: kernels
        )
    }

    private static func appendVerdict(
        to lines: inout [String],
        manifest: SmeltManifest,
        camContext: SmeltOptimizerCAMReportContext?,
        missingTasks: [SmeltOptimizerAgentTask]
    ) {
        lines.append("## Verdict")
        lines.append("")
        if camContext == nil {
            lines.append(
                "**Rebuild package?** Unknown. No CAM optimizer-report route was available, so this report cannot prove CAM gate admission for the package."
            )
        } else if manifest.optimizationReport == nil {
            lines.append(
                "**Rebuild package?** Probably. The package has CAM optimizer-report admission, but it does not contain compile-time optimizer metadata, so the compiler cannot hand agents a reliable missing-fusion queue."
            )
        } else {
            lines.append(
                "**Rebuild package?** No for CAM admission. This package exposes a CAM optimizer-report route and has optimizer metadata."
            )
        }

        if missingTasks.isEmpty {
            lines.append(
                "**Add kernels?** No missing fusion opportunities were recorded in the manifest."
            )
        } else {
            let totalSites = missingTasks.reduce(0) { $0 + $1.count }
            lines.append(
                "**Add kernels?** Yes. The compiler recorded \(missingTasks.count) missing fusion shapes covering \(totalSites) source sites; the work queue below is ordered by Apple Silicon impact score, not raw site count."
            )
        }
        if camContext == nil {
            lines.append(
                "**Evidence level.** Exact for checksums, dispatch tables, and stored compile-time optimizer opportunities; CAM route identity was not provided."
            )
        } else {
            lines.append(
                "**Evidence level.** Exact for checksums, dispatch tables, CAM route identity, and stored compile-time optimizer opportunities."
            )
        }
        lines.append("")
    }

    private static func appendPackage(
        to lines: inout [String],
        packagePath: String,
        manifest: SmeltManifest,
        camContext: SmeltOptimizerCAMReportContext?,
        integrity: SmeltPackageIntegrityReport?
    ) {
        lines.append("## Package")
        lines.append("")
        lines.append("- Package: `\(packagePath)`")
        lines.append("- Model: `\(manifest.modelName)`")
        lines.append("- Shape: hidden \(manifest.config.hiddenSize), FFN \(manifest.config.ffnDim), layers \(manifest.config.numLayers), vocab \(manifest.config.vocabSize)")
        if let camContext {
            lines.append("- CAM semantic SHA-256: `\(camContext.camSemanticSHA256)`")
            lines.append("- CAM export ABI SHA-256: `\(camContext.exportABISHA256)`")
            lines.append("- CAM graph signature SHA-256: `\(camContext.descriptorGraphSignatureSHA256)`")
            lines.append("- CAM optimizer route: export `\(camContext.exportID)`, flow `\(camContext.flowID)`")
            lines.append("- CAM matched gates: \(formatNameList(camContext.matchedGateIDs))")
            lines.append("- CAM authored capabilities: \(formatNameList(camContext.authoredCapabilities))")
        } else {
            lines.append("- CAM optimizer route: unavailable")
        }
        if let provenance = integrity?.buildProvenance ?? manifest.buildProvenance {
            lines.append("- Build fingerprint: `\(provenance.buildFingerprint)`")
            lines.append("- Weights fingerprint: `\(provenance.weightsFingerprint)`")
            lines.append("- Compiler sources SHA-256: `\(provenance.compilerSourcesSHA256)`")
            lines.append("- Shader sources SHA-256: `\(provenance.shaderSourcesSHA256)`")
        }
        if let integrity {
            let verified = integrity.verifiedFiles.map { "`\($0.name)`" }.joined(separator: ", ")
            let skipped = integrity.skippedFiles.map { "`\($0)`" }.joined(separator: ", ")
            lines.append("- Verified package files: \(verified.isEmpty ? "none" : verified)")
            if !skipped.isEmpty {
                lines.append("- Skipped package files: \(skipped)")
            }
        }
        lines.append("")
    }

    private static func appendStructureCoverage(
        to lines: inout [String],
        decodeReport: SmeltDispatchStructureReport?,
        prefillReport: SmeltDispatchStructureReport?,
        topPipelines: Int
    ) {
        lines.append("## Current Smelt Structure Coverage")
        lines.append("")
        appendStructureBlock(
            to: &lines,
            title: "Decode",
            kind: "decode",
            report: decodeReport,
            topPipelines: topPipelines
        )
        appendStructureBlock(
            to: &lines,
            title: "Prefill",
            kind: "prefill",
            report: prefillReport,
            topPipelines: topPipelines
        )
    }

    private static func appendStructureBlock(
        to lines: inout [String],
        title: String,
        kind: String,
        report: SmeltDispatchStructureReport?,
        topPipelines: Int
    ) {
        lines.append("### \(title)")
        lines.append("")
        guard let report else {
            lines.append("No \(kind) dispatch table was found.")
            lines.append("")
            return
        }
        lines.append(
            "The \(kind) table has \(report.totalRecords) records: \(report.dispatchCount) dispatches and \(report.swapCount) swaps."
        )
        lines.append(
            "CAM gate evaluation is route-contract driven; this section reports dispatch-table structure without manifest-family profile inference."
        )
        let top = report.pipelineUsages.prefix(max(topPipelines, 0))
        if !top.isEmpty {
            lines.append("Top active pipelines:")
            for usage in top {
                lines.append("- `\(usage.name)`: \(usage.dispatchCount)")
            }
        }
        lines.append("")
    }

    private static func appendMeasuredTimingEvidence(
        to lines: inout [String],
        timingEvidence: SmeltOptimizerTimingEvidence?,
        options: SmeltOptimizerReportOptions
    ) {
        lines.append("## Measured Timing Evidence")
        lines.append("")
        guard options.includeTimings else {
            lines.append(
                "Not collected. Run `smelt lab inspect cost <package> --profile-timings` to add profiling-mode per-kernel GPU timing to this report."
            )
            lines.append("")
            return
        }
        guard let timingEvidence else {
            lines.append("Timing collection was requested but no timing evidence was produced.")
            lines.append("")
            return
        }

        lines.append(
            "Timing source: profiling-mode GPU timestamps, averaged across \(options.timingIterations) iteration\(options.timingIterations == 1 ? "" : "s"). The absolute totals are diagnostic because each dispatch is profiled separately; the ranking and percentages are the evidence agents should act on."
        )
        if !timingEvidence.warnings.isEmpty {
            lines.append("Warnings:")
            for warning in timingEvidence.warnings {
                lines.append("- \(warning)")
            }
            lines.append("")
        }

        if let decode = timingEvidence.decode {
            appendTimingBlock(
                to: &lines,
                title: "Decode",
                timing: decode,
                maxKernels: options.timingTopKernels
            )
        } else {
            lines.append("### Decode")
            lines.append("")
            lines.append("No decode timing was collected.")
            lines.append("")
        }

        if timingEvidence.prefill.isEmpty {
            lines.append("### Prefill")
            lines.append("")
            lines.append("No prefill timing was collected.")
            lines.append("")
        } else {
            for timing in timingEvidence.prefill.sorted(by: { $0.tokenCount < $1.tokenCount }) {
                appendTimingBlock(
                    to: &lines,
                    title: "Prefill \(timing.tokenCount) Tokens",
                    timing: timing,
                    maxKernels: options.timingTopKernels
                )
            }
        }
    }

    private static func appendPlanCostEvidence(
        to lines: inout [String],
        comparisons: [SmeltPlanComparisonReport]
    ) {
        lines.append("## Graph Plan Cost Evidence")
        lines.append("")
        guard !comparisons.isEmpty else {
            lines.append(
                "No paired graph-plan evidence was supplied. Run `smelt lab probe compare-decode-plans` and pass its JSON with `--plan-comparison`."
            )
            lines.append("")
            return
        }

        for report in comparisons {
            let measurement = report.measurement
            let decision = SmeltPlanCostDecider(
                provenanceKey: measurement.provenanceKey,
                policy: report.decisionPolicy
            ).decide(measurement: measurement)
            let baselineMedian = medianCostSample(
                measurement.samples.map(\.baselineGPUUs)
            )
            let candidateMedian = medianCostSample(
                measurement.samples.map(\.candidateGPUUs)
            )
            lines.append(
                "### `\(measurement.baselinePlanID)` vs `\(measurement.candidatePlanID)`"
            )
            lines.append("")
            lines.append(
                "- Exact FP16 logits: \(measurement.exactOutputMatch ? "yes" : "no"); "
                    + "checked \(measurement.parity?.checkedSteps ?? 0) decode steps."
            )
            lines.append(
                "- Context: \(measurement.context.mode.rawValue), sequence length "
                    + "\(measurement.context.sequenceLength), position "
                    + "\(measurement.context.position)."
            )
            lines.append(
                "- Paired pure-GPU medians: baseline "
                    + "\(formatMicroseconds(baselineMedian)), candidate "
                    + "\(formatMicroseconds(candidateMedian)), "
                    + "\(measurement.samples.count) interleaved pairs."
            )
            if let baseline = measurement.baselineStructure,
               let candidate = measurement.candidateStructure
            {
                lines.append(
                    "- Complete-plan structure: records \(baseline.recordCount) → "
                        + "\(candidate.recordCount), dispatches \(baseline.dispatchCount) → "
                        + "\(candidate.dispatchCount), total threadgroups "
                        + "\(baseline.totalThreadgroups) → \(candidate.totalThreadgroups), "
                        + "max dispatch fan-out \(baseline.maxThreadgroupsPerDispatch) → "
                        + "\(candidate.maxThreadgroupsPerDispatch)."
                )
            }
            lines.append(
                "- Deterministic decision: `\(decision.selectedPlanID)` "
                    + "(`\(decision.reason.rawValue)`); candidate delta "
                    + "\(formatMicroseconds(decision.medianCandidateImprovementUs)), "
                    + "win fraction \(String(format: "%.1f", decision.candidateWinFraction * 100))%, "
                    + "paired-delta MAD \(formatMicroseconds(decision.pairedDeltaMADUs))."
            )
            if decision != report.decision {
                lines.append(
                    "- Warning: stored decision did not match a fresh policy evaluation; the recomputed decision above is authoritative."
                )
            }
            lines.append(
                "- Provenance key: `\(measurement.provenanceKey)`."
            )
            lines.append("")
        }
    }

    private static func medianCostSample(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[middle - 1] + sorted[middle]) / 2
        }
        return sorted[middle]
    }

    private static func appendTimingBlock(
        to lines: inout [String],
        title: String,
        timing: SmeltOptimizerModeTiming,
        maxKernels: Int
    ) {
        lines.append("### \(title)")
        lines.append("")
        if let position = timing.position {
            lines.append("- Context: position \(position), \(timing.iterations) iteration\(timing.iterations == 1 ? "" : "s").")
        } else {
            lines.append("- Context: \(timing.tokenCount) token\(timing.tokenCount == 1 ? "" : "s"), \(timing.iterations) iteration\(timing.iterations == 1 ? "" : "s").")
        }
        lines.append("- Total profiled GPU time: \(formatMicroseconds(timing.totalGpuUs)).")
        if timing.mode == "prefill", timing.tokenCount > 0, timing.totalGpuUs > 0 {
            let tokensPerSecond = Double(timing.tokenCount) / (timing.totalGpuUs / 1_000_000.0)
            lines.append("- Profile-mode throughput proxy: \(String(format: "%.1f", tokensPerSecond)) tok/s.")
        }
        let top = timing.kernels.prefix(max(maxKernels, 0))
        if top.isEmpty {
            lines.append("- Top measured kernels: none.")
        } else {
            lines.append("Top measured kernels:")
            for kernel in top {
                lines.append(
                    "- `\(kernel.name)`: \(kernel.dispatchCount)x, \(formatMicroseconds(kernel.avgGpuUs)), \(String(format: "%.1f", kernel.pctOfTotal))%"
                )
            }
        }
        lines.append("")
    }

    private static func appendCompilerOptimizationSummary(
        to lines: inout [String],
        report: SmeltOptimizationReport?,
        vocabSize: Int
    ) {
        lines.append("## Compiler Optimization Summary")
        lines.append("")
        guard let report else {
            lines.append(
                "The manifest does not include optimizer metadata. Rebuild the package with a current compiler to get deterministic fusion opportunities."
            )
            lines.append("")
            return
        }
        lines.append("Existing-kernel rewrites already applied:")
        lines.append("- Decode: \(formatCounts(report.decodeRewriteCounts))")
        lines.append("- Prefill: \(formatCounts(report.prefillRewriteCounts))")
        lines.append("")

        if let plan = report.compilationPlan {
            lines.append("Capability-planned compilation:")
            lines.append("- Kernel generation policy: `\(plan.kernelGeneration)`")
            lines.append(
                "- Generated kernel consumer kinds: "
                    + formatNameList(plan.generatedKernelConsumerKinds)
            )
            lines.append("- Weight layout policy: `\(plan.weightLayoutPolicy)`")
            lines.append("- Planned buffer slots: \(plan.plannedBufferSlots)")
            lines.append("- Planned activation bytes: \(plan.plannedActivationBytes)")
            lines.append(
                "- Generated kernel candidates: \(plan.plannedKernelUses) selected / "
                    + "\(plan.plannedKernelCandidates) total "
                    + "(\(plan.unsupportedKernelCandidates) unsupported)"
            )
            if !plan.unsupportedKernelCandidateRecords.isEmpty {
                lines.append(
                    "- Unsupported kernel candidate records: "
                        + formatUnsupportedKernelCandidateList(
                            plan.unsupportedKernelCandidateRecords
                        )
                )
            }
            lines.append("- Planned kernel uses: \(plan.plannedKernelUses)")
            if !plan.plannedKernelConsumers.isEmpty {
                lines.append(
                    "- Planned kernel consumer records: "
                        + formatKernelConsumerList(plan.plannedKernelConsumers)
                )
            }
            lines.append("- Planned generated capabilities: \(plan.generatedKernels)")
            lines.append("- Emitted generated kernels: \(plan.emittedGeneratedKernels)")
            if !plan.plannedGeneratedKernelCapabilities.isEmpty {
                lines.append(
                    "- Planned generated capability records: "
                        + formatGeneratedCapabilityList(plan.plannedGeneratedKernelCapabilities)
                )
            }
            if !plan.plannedGeneratedKernelNames.isEmpty {
                lines.append(
                    "- Planned generated capability names: "
                        + formatNameList(plan.plannedGeneratedKernelNames)
                )
            }
            if !plan.emittedGeneratedKernelNames.isEmpty {
                lines.append(
                    "- Emitted generated kernel names: "
                        + formatNameList(plan.emittedGeneratedKernelNames)
                )
            }
            lines.append("- Planned weight consumers: \(plan.plannedWeightUses)")
            if !plan.plannedWeightNames.isEmpty {
                lines.append(
                    "- Planned weight names: " + formatNameList(plan.plannedWeightNames)
                )
            }
            if !plan.plannedWeightConsumerIDs.isEmpty {
                lines.append(
                    "- Planned weight consumer IDs: "
                        + formatNameList(plan.plannedWeightConsumerIDs)
                )
            }
            if !plan.plannedWeightConsumers.isEmpty {
                lines.append(
                    "- Planned weight consumer records: "
                        + formatWeightConsumerList(plan.plannedWeightConsumers)
                )
            }
            lines.append("- Planned weight storage decisions: \(plan.plannedWeightStorageDecisions)")
            if !plan.plannedWeightStorageDecisionNames.isEmpty {
                lines.append(
                    "- Planned weight storage decision names: "
                        + formatNameList(plan.plannedWeightStorageDecisionNames)
                )
            }
            if !plan.plannedWeightStorageDecisionRecords.isEmpty {
                lines.append(
                    "- Planned weight storage decision records: "
                        + formatWeightStorageDecisionList(
                            plan.plannedWeightStorageDecisionRecords
                        )
                )
            }
            if !plan.weightStorageIssueRecords.isEmpty {
                lines.append(
                    "- Weight storage issue records: "
                        + formatWeightStorageIssueList(plan.weightStorageIssueRecords)
                )
            }
            if !plan.weightStorageIssueNames.isEmpty {
                lines.append(
                    "- Weight storage issue names: " + formatNameList(plan.weightStorageIssueNames)
                )
            }
            lines.append(
                "- Weight storage: "
                    + "\(plan.memoryNeutralWeightStorage ? "memory-neutral" : "requires tradeoff"), "
                    + "\(plan.duplicateWeightLayouts) duplicate layout"
                    + "\(plan.duplicateWeightLayouts == 1 ? "" : "s"), "
                    + "\(plan.weightStorageIssues) issue"
                    + "\(plan.weightStorageIssues == 1 ? "" : "s")"
            )
            lines.append("")
        }

        let available = modeTaggedOpportunities(from: report)
            .filter { $0.summary.fusedKernelAvailable }
            .sorted { lhs, rhs in
                sortOpportunity(lhs, rhs, vocabSize: vocabSize)
            }
        if available.isEmpty {
            lines.append("No available-but-source-visible fusion opportunities were recorded before rewriting.")
        } else {
            lines.append(
                "The compiler also recorded source opportunities for kernels that already exist. These are useful sanity checks that the optimizer saw the expected regions before lowering:"
            )
            for item in available.prefix(6) {
                lines.append(
                    "- \(item.mode): `\(item.summary.pattern)` at `\(item.summary.shape)` (\(item.summary.count)x)"
                )
            }
        }
        lines.append("")
    }

    private static func appendCompilerStrategyDiagnosis(
        to lines: inout [String],
        manifest: SmeltManifest,
        decodeReport: SmeltDispatchStructureReport?,
        prefillReport: SmeltDispatchStructureReport?,
        timingEvidence: SmeltOptimizerTimingEvidence?,
        tasks: [SmeltOptimizerAgentTask],
        options: SmeltOptimizerReportOptions
    ) {
        lines.append("## Compiler Strategy Diagnosis")
        lines.append("")
        lines.append("This section combines manifest shape, dispatch-table structure, recorded optimizer opportunities, and optional measured timings to separate local fusion work from structural kernel work.")
        lines.append("- Optimization goal: maximize throughput. No target is required for structural-vs-local decisions.")
        appendStrategyMode(
            to: &lines,
            title: "Decode",
            mode: "decode",
            report: decodeReport,
            timing: timingEvidence?.preferredTiming(for: "decode"),
            tasks: tasks,
            manifest: manifest
        )
        appendStrategyMode(
            to: &lines,
            title: "Prefill",
            mode: "prefill",
            report: prefillReport,
            timing: timingEvidence?.preferredTiming(for: "prefill"),
            tasks: tasks,
            manifest: manifest
        )
        lines.append("")
    }

    private static func appendStrategyMode(
        to lines: inout [String],
        title: String,
        mode: String,
        report: SmeltDispatchStructureReport?,
        timing: SmeltOptimizerModeTiming?,
        tasks: [SmeltOptimizerAgentTask],
        manifest: SmeltManifest
    ) {
        lines.append("")
        lines.append("### \(title)")
        lines.append("")
        guard let report else {
            lines.append("No \(mode) dispatch table was found, so the compiler cannot diagnose this path.")
            return
        }

        let hotspots = SmeltOptimizerKernelHotspot.hotspots(
            from: report,
            mode: mode,
            vocabSize: manifest.config.vocabSize
        )
        if hotspots.isEmpty {
            lines.append("- Static active-kernel hotspots: none detected.")
        } else {
            let total = hotspots.reduce(0.0) { $0 + $1.workProxy }
            let formatted = hotspots.prefix(4).map { hotspot in
                "`\(hotspot.name)` \(hotspot.dispatchCount)x, \(formatPercent(hotspot.workProxy, of: total)) of static QMM work proxy"
            }.joined(separator: "; ")
            lines.append("- Static active-kernel hotspots: \(formatted).")
        }

        if let timing, !timing.kernels.isEmpty {
            let formatted = timing.kernels.prefix(4).map { kernel in
                "`\(kernel.name)` \(kernel.dispatchCount)x, \(formatMicroseconds(kernel.avgGpuUs)), \(String(format: "%.1f", kernel.pctOfTotal))% measured GPU"
            }.joined(separator: "; ")
            lines.append("- Measured active-kernel hotspots: \(formatted).")
        } else {
            lines.append("- Measured active-kernel hotspots: not collected.")
        }

        let modeTasks = tasks.filter { $0.mode == mode }
        let totalScore = modeTasks.reduce(0) { $0 + $1.appleSiliconImpactScore }
        if let topTask = modeTasks.first {
            lines.append(
                "- Missing local-fusion queue: \(modeTasks.count) task\(modeTasks.count == 1 ? "" : "s"), total score \(totalScore); top task is `\(topTask.shape)` at score \(topTask.appleSiliconImpactScore)."
            )
        } else {
            lines.append("- Missing local-fusion queue: empty.")
        }

        lines.append("- Compiler judgment: \(strategyJudgment(mode: mode, tasks: modeTasks, hotspots: hotspots, timing: timing, manifest: manifest))")
        lines.append("- Next non-frontier action: \(nextCompilerAction(mode: mode, tasks: modeTasks, hotspots: hotspots, timing: timing))")
    }

    private static func strategyJudgment(
        mode: String,
        tasks: [SmeltOptimizerAgentTask],
        hotspots: [SmeltOptimizerKernelHotspot],
        timing: SmeltOptimizerModeTiming?,
        manifest: SmeltManifest
    ) -> String {
        let topHotspot = hotspots.first
        let topTask = tasks.first
        let goalPhrase = "The compiler's throughput objective"

        if let measured = timingBackedStructuralHotspot(timing) {
            return "\(goalPhrase) should trust the measured profile first: `\(measured.name)` accounts for \(String(format: "%.1f", measured.pctOfTotal))% of profiled GPU time, so agents should improve that active kernel or its tiling before chasing lower-impact boundary fusions."
        }

        if mode == "decode" {
            if topHotspot?.vocabScale == true {
                return "\(goalPhrase) probably requires improving or avoiding the vocab-scale final projection and other heavyweight active QMM kernels. The remaining local fusions are useful, but they are not a credible decode plan by themselves."
            }
            if let topTask, topTask.touchesVocabScale {
                return "\(goalPhrase) should prioritize the vocab projection fusion first because the compiler sees it as the largest remaining decode-local opportunity."
            }
            if shouldEscalateDecodeToStructural(tasks: tasks, hotspots: hotspots) {
                return "\(goalPhrase) should be treated as structural decode work: the active heavyweight QMM hotspot dominates the static work proxy and the remaining local-fusion queue is lower leverage."
            }
            return "\(goalPhrase) should continue through the local-fusion queue, then re-run the report to see whether active hotspots moved."
        }

        if let topTask, topTask.pattern == "normConsumer", topTask.shape.contains("_batched") {
            return "\(goalPhrase) can plausibly benefit from the remaining batched norm-consumer fusions, but those are still local traffic reductions. If they do not move throughput materially, switch to batched QMM tile geometry or fused FFN-generator work."
        }
        if hotspots.contains(where: { $0.gateActivation }) {
            return "\(goalPhrase) is dominated by batched FFN/QMM work. Local activation or norm fusions are good first moves, but the compiler should escalate to kernel-shape generation when the top local queue is exhausted."
        }
        return "\(goalPhrase) should start with the top local-fusion queue, then re-run this report to see whether active hotspots moved."
    }

    private static func nextCompilerAction(
        mode: String,
        tasks: [SmeltOptimizerAgentTask],
        hotspots: [SmeltOptimizerKernelHotspot],
        timing: SmeltOptimizerModeTiming?
    ) -> String {
        if let measured = timingBackedStructuralHotspot(timing) {
            return "emit a timing-backed structural work item for `\(measured.name)` and require the next agent to show before/after timing evidence for this same package."
        }
        let topTaskTouchesVocab = tasks.first?.touchesVocabScale ?? false
        if mode == "decode",
           let hotspot = hotspots.first,
           hotspot.vocabScale,
           !topTaskTouchesVocab {
            return "emit a structural decode work item for vocab-projection/QMM geometry instead of blindly selecting the next small boundary fusion."
        }
        if mode == "decode",
           let hotspot = hotspots.first,
           shouldEscalateDecodeToStructural(tasks: tasks, hotspots: hotspots) {
            return "emit a structural decode work item for `\(hotspot.name)` and adjacent active QMM geometry before spending another pass on lower-score boundary fusions."
        }
        if !tasks.isEmpty {
            return "attempt the top local-fusion task, then require a regenerated optimizer report and perf gate before advancing."
        }
        if let hotspot = hotspots.first {
            return "emit a structural kernel-generator work item for `\(hotspot.name)` because no source-level missing fusion remains for this mode."
        }
        return "collect a per-kernel profile for this exact package and feed the measured hotspot back into the optimizer report."
    }

    private static func timingBackedStructuralHotspot(
        _ timing: SmeltOptimizerModeTiming?
    ) -> SmeltOptimizerKernelTiming? {
        guard let timing else { return nil }
        return timing.kernels.first { kernel in
            kernel.pctOfTotal >= 20.0 && isStructuralTimingKernel(kernel.name)
        }
    }

    private static func isStructuralTimingKernel(_ name: String) -> Bool {
        name.contains("affine")
            || name.contains("matvec")
            || name.contains("matmul")
            || name.contains("gate_up")
            || name.contains("geglu")
            || name.contains("swiglu")
            || name.contains("attention")
    }

    private static func shouldEscalateDecodeToStructural(
        tasks: [SmeltOptimizerAgentTask],
        hotspots: [SmeltOptimizerKernelHotspot]
    ) -> Bool {
        guard let topHotspot = hotspots.first else { return false }
        let totalWork = hotspots.reduce(0.0) { $0 + $1.workProxy }
        let topShare = totalWork > 0 ? topHotspot.workProxy / totalWork : 0
        let topTaskScore = tasks.first?.appleSiliconImpactScore ?? 0
        let localQueueScore = tasks.reduce(0) { $0 + $1.appleSiliconImpactScore }
        let weakTopLocalTask = topTaskScore < 8_000
        let weakLocalQueue = localQueueScore < 18_000
        let heavyweightHotspot = topShare >= 0.35 || topHotspot.gateActivation || topHotspot.vocabScale
        return heavyweightHotspot && (weakTopLocalTask || weakLocalQueue)
    }

    private static func appendAgentWorkQueue(
        to lines: inout [String],
        tasks: [SmeltOptimizerAgentTask],
        maxTasks: Int
    ) {
        lines.append("## Agent Work Queue")
        lines.append("")
        guard !tasks.isEmpty else {
            lines.append("No missing fused-kernel tasks were recorded.")
            lines.append("")
            return
        }

        for (index, task) in tasks.prefix(max(maxTasks, 0)).enumerated() {
            lines.append(task.markdownCard(priority: index + 1))
        }
    }

    private static func appendTimingBackedStructuralWorkQueue(
        to lines: inout [String],
        timingEvidence: SmeltOptimizerTimingEvidence?
    ) {
        guard let timingEvidence else { return }
        let candidates = [
            timingEvidence.preferredTiming(for: "prefill"),
            timingEvidence.preferredTiming(for: "decode"),
        ]
        .compactMap { timing -> (SmeltOptimizerModeTiming, SmeltOptimizerKernelTiming)? in
            guard let timing, let kernel = timingBackedStructuralHotspot(timing) else {
                return nil
            }
            return (timing, kernel)
        }
        .sorted { lhs, rhs in
            if lhs.1.pctOfTotal != rhs.1.pctOfTotal {
                return lhs.1.pctOfTotal > rhs.1.pctOfTotal
            }
            return lhs.0.mode < rhs.0.mode
        }

        guard !candidates.isEmpty else { return }

        lines.append("## Timing-Backed Structural Work Queue")
        lines.append("")
        for (index, candidate) in candidates.enumerated() {
            let timing = candidate.0
            let kernel = candidate.1
            let titleMode = timing.mode == "prefill"
                ? "prefill \(timing.tokenCount)-token"
                : "decode"
            lines.append("### Structural Priority \(index + 1): \(titleMode) active-kernel optimization for `\(kernel.name)`")
            lines.append("")
            lines.append("Measured evidence: `\(kernel.name)` accounts for \(String(format: "%.1f", kernel.pctOfTotal))% of profiled \(timing.mode) GPU time (\(formatMicroseconds(kernel.avgGpuUs)), \(kernel.dispatchCount)x).")
            lines.append("")
            lines.append("Build: Improve this active kernel's implementation, tile geometry, dispatch shape, or family-independent kernel generator. Do not spend this task on unrelated boundary-fusion cleanup unless the profiling evidence shows it directly reduces this kernel's time.")
            lines.append("")
            lines.append("Why it matters: This is measured hot-path work, not just a static source-site count. A successful change should reduce the same measured kernel timing and improve the end-to-end CAM gate suite.")
            lines.append("")
            lines.append("Likely files: \(likelyStructuralFiles(for: kernel.name).map { "`\($0)`" }.joined(separator: ", ")).")
            lines.append("")
            lines.append("Done when: `smelt lab inspect cost --profile-timings` shows lower timing for this kernel on the same package shape, and the relevant CAM gate suite still passes parity, structure, and performance gates.")
            lines.append("")
        }
    }

    private static func likelyStructuralFiles(for kernelName: String) -> [String] {
        if kernelName.contains("attention") {
            return [
                "Resources/Shaders/attention.metal",
                "Sources/SmeltCompiler/AttentionPlugin.swift",
                "Sources/SmeltCompiler/SmeltKernelCatalog.swift",
            ]
        }
        if kernelName.contains("affine")
            || kernelName.contains("matvec")
            || kernelName.contains("matmul")
            || kernelName.contains("geglu")
            || kernelName.contains("swiglu")
            || kernelName.contains("gate_up") {
            return [
                "Resources/Shaders/lut_matvec.metal",
                "Sources/SmeltCompiler/SmeltKernelShapeRegistry.swift",
                "Sources/SmeltCompiler/SmeltKernelCatalog.swift",
            ]
        }
        return [
            "Resources/Shaders",
            "Sources/SmeltCompiler/SmeltKernelCatalog.swift",
            "Sources/SmeltRuntime/SmeltTextRuntime.swift",
        ]
    }

    private static func appendNotes(to lines: inout [String]) {
        lines.append("## Notes And Limits")
        lines.append("")
        lines.append("- This report is English-first on purpose: the task cards are intended to be read directly by implementation agents.")
        lines.append("- The missing-fusion queue comes from the compiler's source-level opportunity scanner stored in the package manifest. It is deterministic for a rebuilt package.")
        lines.append("- Apple Silicon priority scores are deterministic heuristics. They favor batched prefill, large matvec rows, GeGLU/SwiGLU staging removal, and KV-cache memory traffic because those are the paths most likely to move unified-memory bandwidth and dispatch overhead.")
        lines.append("- Timing evidence is opt-in because per-kernel profiling is slower than a normal run. When present, timing-backed structural recommendations should outrank static source-site counts.")
        lines.append("- A fresh perf gate is still required to prove end-to-end speedup; profiling-mode kernel timings identify where an agent should work next.")
        lines.append("- Rebuild advice is based on CAM optimizer-report admission plus stored compiler metadata. To detect every future kernel added after a package was built, packages will eventually need to store a replayable pre-lowering op trace or the CLI needs a source-hash comparison against the local compiler checkout.")
    }

    private static func formatCounts(_ counts: [String: Int]) -> String {
        guard !counts.isEmpty else { return "none" }
        return counts
            .sorted { lhs, rhs in lhs.key < rhs.key }
            .map { "`\($0.key)`=\($0.value)" }
            .joined(separator: ", ")
    }

    private static func formatNameList(_ names: [String], limit: Int = 16) -> String {
        guard !names.isEmpty else { return "none" }
        let prefix = names.prefix(max(limit, 0)).map { "`\($0)`" }.joined(separator: ", ")
        let remaining = names.count - min(names.count, max(limit, 0))
        guard remaining > 0 else { return prefix }
        return prefix + ", +\(remaining) more"
    }

    private static func formatKernelConsumerList(
        _ consumers: [SmeltPlannedKernelConsumerReport],
        limit: Int = 8
    ) -> String {
        guard !consumers.isEmpty else { return "none" }
        let boundedLimit = max(limit, 0)
        let prefix = consumers.prefix(boundedLimit).map { consumer in
            let kind = consumer.consumerKind.map { " \($0)" } ?? ""
            return "`\(consumer.consumerID)`->`\(consumer.capabilityName)`"
                + " \(consumer.phase)/\(consumer.operation)"
                + kind
                + " c\(consumer.cols)r\(consumer.rows)g\(consumer.groupSize)"
        }.joined(separator: ", ")
        let remaining = consumers.count - min(consumers.count, boundedLimit)
        guard remaining > 0 else { return prefix }
        return prefix + ", +\(remaining) more"
    }

    private static func formatGeneratedCapabilityList(
        _ capabilities: [SmeltGeneratedKernelCapabilityReport],
        limit: Int = 8
    ) -> String {
        guard !capabilities.isEmpty else { return "none" }
        let boundedLimit = max(limit, 0)
        let prefix = capabilities.prefix(boundedLimit).map { capability in
            return "`\(capability.capabilityName)`"
                + " \(capability.phase)/\(capability.operation)"
                + " c\(capability.cols)r\(capability.rows)g\(capability.groupSize)"
                + " \(capability.sourceKind)"
        }.joined(separator: ", ")
        let remaining = capabilities.count - min(capabilities.count, boundedLimit)
        guard remaining > 0 else { return prefix }
        return prefix + ", +\(remaining) more"
    }

    private static func formatUnsupportedKernelCandidateList(
        _ candidates: [SmeltUnsupportedKernelCandidateReport],
        limit: Int = 8
    ) -> String {
        guard !candidates.isEmpty else { return "none" }
        let boundedLimit = max(limit, 0)
        let prefix = candidates.prefix(boundedLimit).map { candidate in
            let weights = candidate.weights
                .map { "`\($0.role):\($0.weightName)`" }
                .joined(separator: ", ")
            let kind = candidate.consumerKind.map { " \($0)" } ?? ""
            return "`\(candidate.consumerID)`"
                + " \(candidate.phase)/\(candidate.operation)"
                + kind
                + " c\(candidate.cols)r\(candidate.rows)g\(candidate.groupSize)"
                + " \(candidate.reason)"
                + (weights.isEmpty ? "" : " via \(weights)")
        }.joined(separator: ", ")
        let remaining = candidates.count - min(candidates.count, boundedLimit)
        guard remaining > 0 else { return prefix }
        return prefix + ", +\(remaining) more"
    }

    private static func formatWeightConsumerList(
        _ consumers: [SmeltPlannedWeightConsumerReport],
        limit: Int = 8
    ) -> String {
        guard !consumers.isEmpty else { return "none" }
        let boundedLimit = max(limit, 0)
        let prefix = consumers.prefix(boundedLimit).map { consumer in
            let kind = consumer.consumerKind.map { " \($0)" } ?? ""
            return "`\(consumer.weightName)`->`\(consumer.consumerID)`"
                + " via `\(consumer.capabilityName)`/\(consumer.weightRole)"
                + kind
        }.joined(separator: ", ")
        let remaining = consumers.count - min(consumers.count, boundedLimit)
        guard remaining > 0 else { return prefix }
        return prefix + ", +\(remaining) more"
    }

    private static func formatWeightStorageDecisionList(
        _ decisions: [SmeltPlannedWeightStorageDecisionReport],
        limit: Int = 8
    ) -> String {
        guard !decisions.isEmpty else { return "none" }
        let boundedLimit = max(limit, 0)
        let prefix = decisions.prefix(boundedLimit).map { decision in
            let storage = decision.requiresDuplicateLayout ? "duplicate" : "single"
            return "`\(decision.weightName)` "
                + "`\(decision.currentLayout)`->`\(decision.selectedLayout)` "
                + "\(storage) for "
                + formatStorageDecisionConsumers(decision.consumers)
        }.joined(separator: ", ")
        let remaining = decisions.count - min(decisions.count, boundedLimit)
        guard remaining > 0 else { return prefix }
        return prefix + ", +\(remaining) more"
    }

    private static func formatStorageDecisionConsumers(
        _ consumers: [SmeltPlannedWeightStorageDecisionConsumerReport]
    ) -> String {
        guard !consumers.isEmpty else { return "none" }
        return consumers.map { consumer in
            let kind = consumer.consumerKind.map { " (\($0))" } ?? ""
            return "`\(consumer.consumerID)`" + kind
        }.joined(separator: ", ")
    }

    private static func formatWeightStorageIssueList(
        _ issues: [SmeltPlannedWeightStorageIssueReport],
        limit: Int = 8
    ) -> String {
        guard !issues.isEmpty else { return "none" }
        let boundedLimit = max(limit, 0)
        let prefix = issues.prefix(boundedLimit).map { issue in
            "`\(issue.weightName)` \(issue.kind) for "
                + formatStorageDecisionConsumers(issue.consumers)
        }.joined(separator: ", ")
        let remaining = issues.count - min(issues.count, boundedLimit)
        guard remaining > 0 else { return prefix }
        return prefix + ", +\(remaining) more"
    }

    private static func formatPercent(_ value: Double, of total: Double) -> String {
        guard total > 0 else { return "0%" }
        return String(format: "%.0f%%", (value / total) * 100.0)
    }

    private static func formatMicroseconds(_ value: Double) -> String {
        if value >= 1_000 {
            return String(format: "%.2f ms", value / 1_000.0)
        }
        return String(format: "%.0f us", value)
    }

    fileprivate static func modeTaggedOpportunities(
        from report: SmeltOptimizationReport
    ) -> [SmeltOptimizerModeOpportunity] {
        report.decodeOpportunities.map {
            SmeltOptimizerModeOpportunity(mode: "decode", summary: $0)
        } + report.prefillOpportunities.map {
            SmeltOptimizerModeOpportunity(mode: "prefill", summary: $0)
        }
    }

    fileprivate static func sortOpportunity(
        _ lhs: SmeltOptimizerModeOpportunity,
        _ rhs: SmeltOptimizerModeOpportunity,
        vocabSize: Int
    ) -> Bool {
        let lhsScore = SmeltOptimizerAppleSiliconScorer.score(
            mode: lhs.mode,
            pattern: lhs.summary.pattern,
            shape: lhs.summary.shape,
            count: lhs.summary.count,
            vocabSize: vocabSize
        )
        let rhsScore = SmeltOptimizerAppleSiliconScorer.score(
            mode: rhs.mode,
            pattern: rhs.summary.pattern,
            shape: rhs.summary.shape,
            count: rhs.summary.count,
            vocabSize: vocabSize
        )
        if lhsScore.value != rhsScore.value {
            return lhsScore.value > rhsScore.value
        }
        if lhs.summary.count != rhs.summary.count {
            return lhs.summary.count > rhs.summary.count
        }
        if lhs.mode != rhs.mode {
            return lhs.mode < rhs.mode
        }
        if lhs.summary.pattern != rhs.summary.pattern {
            return lhs.summary.pattern < rhs.summary.pattern
        }
        return lhs.summary.shape < rhs.summary.shape
    }
}

fileprivate struct SmeltOptimizerModeOpportunity {
    let mode: String
    let summary: SmeltFusionOpportunitySummary
}

private struct SmeltOptimizerAppleSiliconScore: Sendable {
    let value: Int
    let signals: [String]
    let touchesVocabScale: Bool
}

private struct SmeltOptimizerShapeTraits {
    let maxColumns: Int?
    let maxRows: Int?
    let batched: Bool
    let matvec: Bool
    let gateActivation: Bool
    let ropeKVPrefill: Bool
    let applyRope: Bool
    let elementwiseAdd: Bool

    static func parse(_ shape: String) -> SmeltOptimizerShapeTraits {
        SmeltOptimizerShapeTraits(
            maxColumns: maxTaggedValue("c", in: shape),
            maxRows: maxTaggedValue("r", in: shape),
            batched: shape.contains("_batched"),
            matvec: shape.contains("matvec"),
            gateActivation: shape.contains("geglu") || shape.contains("swiglu"),
            ropeKVPrefill: shape.contains("rope_and_kv_cache_prefill"),
            applyRope: shape.contains("apply_rope"),
            elementwiseAdd: shape.contains("elementwise_add")
        )
    }

    private static func maxTaggedValue(_ tag: Character, in shape: String) -> Int? {
        let tokens = shape.split { character in
            !character.isLetter && !character.isNumber
        }
        let values = tokens.compactMap { token -> Int? in
            guard token.first == tag else { return nil }
            return Int(token.dropFirst())
        }
        return values.max()
    }
}

private enum SmeltOptimizerAppleSiliconScorer {
    static func score(
        mode: String,
        pattern: String,
        shape: String,
        count: Int,
        vocabSize: Int
    ) -> SmeltOptimizerAppleSiliconScore {
        let traits = SmeltOptimizerShapeTraits.parse(shape)
        let touchesVocabScale = traits.maxRows.map {
            SmeltOptimizerVocabHeuristic.isVocabScale(rows: $0, vocabSize: vocabSize)
        } ?? false
        var value = max(count, 1) * 10
        var signals: [String] = ["\(count)x source sites"]

        if mode == "prefill" {
            value += count * 80
            signals.append("prefill batches stress unified-memory bandwidth")
        } else {
            value += count * 18
            signals.append("decode is scalar-token work")
        }

        switch pattern {
        case "dualMatvecActivation":
            value += count * 180
            signals.append("two matvec outputs feed one activation")
        case "normConsumer":
            value += count * 55
            signals.append("norm output can stay local to the consumer")
        default:
            value += count * 35
            signals.append("removes intermediate dispatch boundaries")
        }

        if traits.batched {
            value += count * 70
            signals.append("batched shape avoids repeated activation traffic")
        }
        if traits.gateActivation {
            value += count * 120
            signals.append("GeGLU/SwiGLU staging is high-traffic")
        }
        if traits.ropeKVPrefill {
            value += count * 70
            signals.append("RoPE plus KV-cache prefill touches cache memory")
        } else if traits.applyRope {
            value += count * (mode == "prefill" ? 45 : 18)
            signals.append("RoPE fusion removes a dispatch boundary")
        }

        if let rows = traits.maxRows {
            if touchesVocabScale {
                value += 3_000 + count * 250
                signals.append("vocab-scale row count r\(rows)")
            } else if rows >= 10_000 {
                value += count * 80
                signals.append("large matvec row count r\(rows)")
            } else if rows >= 4_096 {
                value += count * 45
                signals.append("medium matvec row count r\(rows)")
            } else if rows >= 2_048 {
                value += count * 30
                signals.append("matvec row count r\(rows)")
            }
        }

        if traits.matvec {
            let rowBand = min(max((traits.maxRows ?? 1_024) / 1_024, 1), 32)
            value += count * max(rowBand * 6, 12)
            if let columns = traits.maxColumns {
                signals.append("QMM consumer has c\(columns) inputs")
            } else {
                signals.append("QMM consumer is bandwidth-sensitive")
            }
        }

        if traits.elementwiseAdd {
            value -= count * 20
            signals.append("elementwise-only consumer is lower leverage")
        }
        if mode == "decode", !touchesVocabScale {
            value -= count * 20
            signals.append("decode-only fusions need benchmark proof")
        }

        return SmeltOptimizerAppleSiliconScore(
            value: max(value, 0),
            signals: Array(signals.prefix(5)),
            touchesVocabScale: touchesVocabScale
        )
    }
}

private enum SmeltOptimizerVocabHeuristic {
    static func isVocabScale(rows: Int, vocabSize: Int) -> Bool {
        guard vocabSize > 0 else { return false }
        if rows == vocabSize { return true }
        let lowerBound = max(32_768, Int(Double(vocabSize) * 0.90))
        let upperBound = Int(Double(vocabSize) * 1.10)
        return rows >= lowerBound && rows <= upperBound
    }
}

private struct SmeltOptimizerKernelHotspot {
    let name: String
    let dispatchCount: Int
    let workProxy: Double
    let vocabScale: Bool
    let gateActivation: Bool

    static func hotspots(
        from report: SmeltDispatchStructureReport,
        mode: String,
        vocabSize: Int
    ) -> [SmeltOptimizerKernelHotspot] {
        report.pipelineUsages.compactMap { usage in
            hotspot(
                name: usage.name,
                dispatchCount: usage.dispatchCount,
                mode: mode,
                vocabSize: vocabSize
            )
        }
        .sorted { lhs, rhs in
            if lhs.workProxy != rhs.workProxy {
                return lhs.workProxy > rhs.workProxy
            }
            if lhs.dispatchCount != rhs.dispatchCount {
                return lhs.dispatchCount > rhs.dispatchCount
            }
            return lhs.name < rhs.name
        }
    }

    private static func hotspot(
        name: String,
        dispatchCount: Int,
        mode: String,
        vocabSize: Int
    ) -> SmeltOptimizerKernelHotspot? {
        let traits = SmeltOptimizerShapeTraits.parse(name)
        let affineLike = name.contains("affine") || name.contains("matvec")
        guard affineLike, let rows = traits.maxRows, let columns = traits.maxColumns else {
            return nil
        }

        var multiplier = 1.0
        if traits.gateActivation || name.contains("gate_up") {
            multiplier *= 2.0
        }
        if mode == "prefill", traits.batched {
            multiplier *= 4.0
        }
        if name.contains("norm_scale") || name.contains("norm_add_scale") {
            multiplier *= 1.05
        }
        let work = Double(max(dispatchCount, 1))
            * Double(rows)
            * Double(columns)
            * multiplier

        return SmeltOptimizerKernelHotspot(
            name: name,
            dispatchCount: dispatchCount,
            workProxy: work,
            vocabScale: SmeltOptimizerVocabHeuristic.isVocabScale(
                rows: rows,
                vocabSize: vocabSize
            ),
            gateActivation: traits.gateActivation || name.contains("gate_up")
        )
    }
}

public struct SmeltOptimizerAgentTask: Sendable {
    public let id: String
    public let mode: String
    public let pattern: String
    public let shape: String
    public let count: Int
    public let appleSiliconImpactScore: Int
    public let scoringSignals: [String]
    public let title: String
    public let buildInstruction: String
    public let impact: String
    public let likelyFiles: [String]
    public let touchesVocabScale: Bool

    static func tasks(from manifest: SmeltManifest) -> [SmeltOptimizerAgentTask] {
        guard let report = manifest.optimizationReport else { return [] }
        return SmeltOptimizerReportGenerator
            .modeTaggedOpportunities(from: report)
            .filter { !$0.summary.fusedKernelAvailable }
            .sorted { lhs, rhs in
                SmeltOptimizerReportGenerator.sortOpportunity(
                    lhs,
                    rhs,
                    vocabSize: manifest.config.vocabSize
                )
            }
            .map { tagged in
                SmeltOptimizerAgentTask(
                    mode: tagged.mode,
                    summary: tagged.summary,
                    vocabSize: manifest.config.vocabSize
                )
            }
    }

    init(mode: String, summary: SmeltFusionOpportunitySummary, vocabSize: Int) {
        let score = SmeltOptimizerAppleSiliconScorer.score(
            mode: mode,
            pattern: summary.pattern,
            shape: summary.shape,
            count: summary.count,
            vocabSize: vocabSize
        )
        self.mode = mode
        self.pattern = summary.pattern
        self.shape = summary.shape
        self.count = summary.count
        self.appleSiliconImpactScore = score.value
        self.scoringSignals = score.signals
        self.touchesVocabScale = score.touchesVocabScale
        self.id = Self.makeID(mode: mode, pattern: summary.pattern, shape: summary.shape)
        self.title = Self.makeTitle(
            mode: mode,
            pattern: summary.pattern,
            shape: summary.shape,
            count: summary.count
        )
        self.buildInstruction = Self.makeBuildInstruction(
            mode: mode,
            pattern: summary.pattern,
            shape: summary.shape
        )
        self.impact = Self.makeImpact(
            mode: mode,
            pattern: summary.pattern,
            count: summary.count
        )
        self.likelyFiles = Self.makeLikelyFiles(pattern: summary.pattern, shape: summary.shape)
    }

    public func markdownCard(priority: Int) -> String {
        [
            "### Priority \(priority): \(title)",
            "",
            "Task ID: `\(id)`",
            "",
            "What the compiler sees: \(mode) has \(count) source site\(count == 1 ? "" : "s") of `\(pattern)` with shape `\(shape)`.",
            "",
            "Apple Silicon priority score: \(appleSiliconImpactScore).",
            "",
            "Scoring signals: \(scoringSignals.joined(separator: "; ")).",
            "",
            "Build: \(buildInstruction)",
            "",
            "Why it matters: \(impact)",
            "",
            "Likely files: \(likelyFiles.map { "`\($0)`" }.joined(separator: ", ")).",
            "",
            "Correctness gates: rebuild the package if package-affecting inputs changed, run `smelt lab inspect cost` again, and run the relevant CAM gate suite before trusting the speedup.",
            "",
            "Done when: this task ID disappears from the optimizer report and the CAM gate suite still passes parity, structure, and performance gates.",
            "",
        ].joined(separator: "\n")
    }

    private static func makeID(mode: String, pattern: String, shape: String) -> String {
        let raw = "\(mode)-\(pattern)-\(shape)"
        let scalars = raw.unicodeScalars.map { scalar -> Character in
            if CharacterSet.alphanumerics.contains(scalar) {
                return Character(String(scalar).lowercased())
            }
            return "-"
        }
        let collapsed = String(scalars)
            .split(separator: "-")
            .joined(separator: "-")
        return "fusion-\(String(collapsed.prefix(80)))-\(stableHash(raw).prefix(12))"
    }

    private static func stableHash(_ value: String) -> String {
        var hash: UInt64 = 14_695_981_039_346_656_037
        for byte in value.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1_099_511_628_211
        }
        return String(hash, radix: 16)
    }

    private static func makeTitle(
        mode: String,
        pattern: String,
        shape: String,
        count: Int
    ) -> String {
        "\(mode) \(pattern) fusion for `\(shape)` (\(count)x)"
    }

    private static func makeBuildInstruction(
        mode: String,
        pattern: String,
        shape: String
    ) -> String {
        let target = suggestedTargetKernel(pattern: pattern, shape: shape)
        if let target {
            return "Add and register a fused route that lowers this pattern to `\(target)` in \(mode). The route should be selected by the auto fusion policy, not by model-family special casing."
        }
        return "Add and register a fused route for this exact pattern in \(mode). The route should be selected by the auto fusion policy, not by model-family special casing."
    }

    private static func makeImpact(
        mode: String,
        pattern: String,
        count: Int
    ) -> String {
        switch pattern {
        case "dualMatvecActivation":
            return "This removes a gate/up activation staging region at each site. In prefill, that usually saves multiple memory traffic passes over batched activations, which is the main Apple Silicon bottleneck for these shapes."
        case "normConsumer":
            return "This removes a norm-to-consumer dispatch boundary at each site and keeps the normalized value local to the fused consumer path."
        default:
            return "This removes one or more dispatch boundaries at each recorded site and reduces intermediate buffer traffic."
        }
    }

    private static func makeLikelyFiles(pattern: String, shape: String) -> [String] {
        var files = [
            "Sources/SmeltCompiler/SmeltFusionPlanner.swift",
            "Sources/SmeltCompiler/SmeltKernelCatalog.swift",
            "Sources/SmeltCompiler/SmeltKernelShapeRegistry.swift",
        ]
        if shape.contains("rope_and_kv_cache_prefill") {
            files.append("Resources/Shaders/prefill_rope_kv.metal")
        } else if shape.contains("apply_rope") {
            files.append("Resources/Shaders/attention.metal")
        } else if shape.contains("attention") {
            files.append("Resources/Shaders/attention_fused.metal")
        } else if shape.contains("norm") && shape.contains("matvec") {
            files.append("Resources/Shaders/fused_norm_matvec.metal")
        } else if shape.contains("geglu") || shape.contains("swiglu") {
            files.append("Resources/Shaders/lut_matvec.metal")
            files.append("Resources/Shaders/activations.metal")
        } else {
            files.append("Resources/Shaders")
        }
        return files
    }

    private static func suggestedTargetKernel(pattern: String, shape: String) -> String? {
        guard pattern == "normConsumer" else { return nil }
        guard let arrow = shape.range(of: "->") else { return nil }
        let producer = String(shape[..<arrow.lowerBound])
        let consumer = String(shape[arrow.upperBound...])
        guard !producer.contains("_add") else {
            return nil
        }
        if consumer.hasPrefix("affine_matvec_") {
            return "norm_scale_\(consumer)"
        }
        if consumer.hasPrefix("fused_affine_gate_up_") {
            return consumer.replacingOccurrences(
                of: "fused_affine_gate_up_",
                with: "norm_scale_affine_gate_up_"
            )
        }
        if consumer == "apply_rope" || consumer == "rope_and_kv_cache_prefill" {
            return "fused_norm_\(consumer)"
        }
        return nil
    }
}
