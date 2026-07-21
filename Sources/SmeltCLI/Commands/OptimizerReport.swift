import Foundation
import SmeltCompiler
import SmeltRuntime
import SmeltSchema

func requireCAMOptimizerReportAdmissionOrExit(
    packagePath: String,
    verb: String
) -> SmeltOptimizerCAMReportContext {
    let capabilities = requireCAMPackageCapabilitiesOrExit(
        packagePath: packagePath,
        verb: verb
    )
    requireCAMPackageInventoryOrExit(
        capabilities,
        packagePath: packagePath,
        verb: verb,
        requireAuthoredInventory: true
    )
    let request = SmeltCAMCapabilityRequest.optimizerReport
    let decision: SmeltCAMPackageCapabilities.Decision
    do {
        decision = try capabilities.resolve(request)
    } catch SmeltCAMPackageCapabilitiesError.noMatchingExport {
        fputs("smelt \(verb): no CAM export satisfies optimizer report request\n", stderr)
        exit(1)
    } catch {
        fputs("smelt \(verb): \(error)\n", stderr)
        exit(1)
    }
    requireCAMCapabilityFilesOrExit(
        request.requiredPackageFiles,
        packagePath: packagePath,
        verb: verb
    )
    return SmeltOptimizerCAMReportContext(
        camSemanticSHA256: capabilities.camSemanticSHA256,
        exportABISHA256: capabilities.exportABISHA256,
        descriptorGraphSignatureSHA256: capabilities.descriptorGraphSignatureSHA256,
        exportID: decision.exportID,
        flowID: decision.flowID,
        matchedGateIDs: decision.matchedGateIDs.sorted(),
        authoredCapabilities: decision.authoredCapabilities.sorted()
    )
}

func runOptimizerReportCommand(_ args: [String]) {
    let optimizerReportUsage = "Usage: smelt lab inspect cost <model.smeltpkg> [--output FILE] [--verify-weights] [--top-pipelines N] [--max-smelt-tasks N] [--profile-timings] [--timing-iterations N] [--timing-decode-position N] [--timing-prefill-tokens N[,M...]] [--timing-top-kernels N] [--plan-comparison FILE] [--frozen-cost] [--frozen-cost-table decode|prefill|prefill-verify-argmax] [--frozen-cost-sequence-length N] [--frozen-cost-json FILE] [--frozen-cost-baseline FILE] [--frozen-cost-delta-json FILE] [--frozen-cost-calibration FILE | --frozen-cost-calibrate FILE] [--frozen-cost-calibration-warmup N] [--frozen-cost-calibration-iterations N] [--frozen-cost-position N] [--frozen-cost-top-dispatches N]\n"
    var packagePath: String?
    var outputPath: String?
    var verifyWeights = false
    var topPipelines = 8
    var maxAgentTasks = 8
    var profileTimings = false
    var timingIterations = 1
    var timingDecodePosition: Int32 = 0
    var timingPrefillTokens = [64]
    var timingTopKernels = 8
    var planComparisonPaths: [String] = []
    var includeFrozenCost = false
    var frozenCostJSONPath: String?
    var frozenCostBaselinePath: String?
    var frozenCostDeltaJSONPath: String?
    var frozenCostCalibrationPath: String?
    var frozenCostCalibrationOutputPath: String?
    var frozenCostCalibrationWarmup = 3
    var frozenCostCalibrationIterations = 20
    var frozenCostPosition = 0
    var frozenCostSequenceLength: Int?
    var frozenCostTable = SmeltFrozenIRDispatchTable.decode
    var frozenCostTopDispatches = 12
    var idx = 2
    while idx < args.count {
        let arg = args[idx]
        switch arg {
        case "--output":
            guard idx + 1 < args.count else {
                fputs("Error: --output requires a path\n", stderr)
                exit(1)
            }
            outputPath = args[idx + 1]
            idx += 2
        case "--verify-weights":
            verifyWeights = true
            idx += 1
        case "--profile-timings":
            profileTimings = true
            idx += 1
        case "--top-pipelines":
            guard idx + 1 < args.count, let parsed = Int(args[idx + 1]), parsed >= 0 else {
                fputs("Error: --top-pipelines requires a non-negative integer\n", stderr)
                exit(1)
            }
            topPipelines = parsed
            idx += 2
        case "--max-smelt-tasks":
            guard idx + 1 < args.count, let parsed = Int(args[idx + 1]), parsed >= 0 else {
                fputs("Error: --max-smelt-tasks requires a non-negative integer\n", stderr)
                exit(1)
            }
            maxAgentTasks = parsed
            idx += 2
        case "--timing-iterations":
            guard idx + 1 < args.count, let parsed = Int(args[idx + 1]), parsed > 0 else {
                fputs("Error: --timing-iterations requires a positive integer\n", stderr)
                exit(1)
            }
            timingIterations = parsed
            idx += 2
        case "--timing-decode-position":
            guard idx + 1 < args.count, let parsed = Int32(args[idx + 1]), parsed >= 0 else {
                fputs("Error: --timing-decode-position requires a non-negative integer\n", stderr)
                exit(1)
            }
            timingDecodePosition = parsed
            idx += 2
        case "--timing-prefill-tokens":
            guard idx + 1 < args.count else {
                fputs("Error: --timing-prefill-tokens requires a comma-separated positive integer list\n", stderr)
                exit(1)
            }
            let tokenParts = args[idx + 1].split(separator: ",")
            let parsed = tokenParts.map {
                Int($0.trimmingCharacters(in: .whitespacesAndNewlines))
            }
            guard !parsed.isEmpty, parsed.allSatisfy({ ($0 ?? 0) > 0 }) else {
                fputs("Error: --timing-prefill-tokens requires a comma-separated positive integer list\n", stderr)
                exit(1)
            }
            timingPrefillTokens = parsed.compactMap { $0 }
            idx += 2
        case "--timing-top-kernels":
            guard idx + 1 < args.count, let parsed = Int(args[idx + 1]), parsed >= 0 else {
                fputs("Error: --timing-top-kernels requires a non-negative integer\n", stderr)
                exit(1)
            }
            timingTopKernels = parsed
            idx += 2
        case "--plan-comparison":
            guard idx + 1 < args.count else {
                fputs("Error: --plan-comparison requires a JSON path\n", stderr)
                exit(1)
            }
            planComparisonPaths.append(args[idx + 1])
            idx += 2
        case "--frozen-cost":
            includeFrozenCost = true
            idx += 1
        case "--frozen-cost-table":
            guard idx + 1 < args.count,
                  let parsed = SmeltFrozenIRDispatchTable(cliName: args[idx + 1])
            else {
                fputs(
                    "Error: --frozen-cost-table requires decode, prefill, or prefill-verify-argmax\n",
                    stderr
                )
                exit(1)
            }
            includeFrozenCost = true
            frozenCostTable = parsed
            idx += 2
        case "--frozen-cost-sequence-length":
            guard idx + 1 < args.count,
                  let parsed = Int(args[idx + 1]), parsed > 0 else {
                fputs("Error: --frozen-cost-sequence-length requires a positive integer\n", stderr)
                exit(1)
            }
            includeFrozenCost = true
            frozenCostSequenceLength = parsed
            idx += 2
        case "--frozen-cost-json":
            guard idx + 1 < args.count else {
                fputs("Error: --frozen-cost-json requires a path\n", stderr)
                exit(1)
            }
            includeFrozenCost = true
            frozenCostJSONPath = args[idx + 1]
            idx += 2
        case "--frozen-cost-baseline":
            guard idx + 1 < args.count else {
                fputs("Error: --frozen-cost-baseline requires a report JSON path\n", stderr)
                exit(1)
            }
            includeFrozenCost = true
            frozenCostBaselinePath = args[idx + 1]
            idx += 2
        case "--frozen-cost-delta-json":
            guard idx + 1 < args.count else {
                fputs("Error: --frozen-cost-delta-json requires an output path\n", stderr)
                exit(1)
            }
            includeFrozenCost = true
            frozenCostDeltaJSONPath = args[idx + 1]
            idx += 2
        case "--frozen-cost-calibration":
            guard idx + 1 < args.count else {
                fputs("Error: --frozen-cost-calibration requires a JSON path\n", stderr)
                exit(1)
            }
            includeFrozenCost = true
            frozenCostCalibrationPath = args[idx + 1]
            idx += 2
        case "--frozen-cost-calibrate":
            guard idx + 1 < args.count else {
                fputs("Error: --frozen-cost-calibrate requires a JSON output path\n", stderr)
                exit(1)
            }
            includeFrozenCost = true
            frozenCostCalibrationOutputPath = args[idx + 1]
            idx += 2
        case "--frozen-cost-calibration-warmup":
            guard idx + 1 < args.count,
                  let parsed = Int(args[idx + 1]), parsed >= 0 else {
                fputs("Error: --frozen-cost-calibration-warmup requires a non-negative integer\n", stderr)
                exit(1)
            }
            includeFrozenCost = true
            frozenCostCalibrationWarmup = parsed
            idx += 2
        case "--frozen-cost-calibration-iterations":
            guard idx + 1 < args.count,
                  let parsed = Int(args[idx + 1]), parsed > 0 else {
                fputs("Error: --frozen-cost-calibration-iterations requires a positive integer\n", stderr)
                exit(1)
            }
            includeFrozenCost = true
            frozenCostCalibrationIterations = parsed
            idx += 2
        case "--frozen-cost-position":
            guard idx + 1 < args.count,
                  let parsed = Int(args[idx + 1]), parsed >= 0 else {
                fputs("Error: --frozen-cost-position requires a non-negative integer\n", stderr)
                exit(1)
            }
            includeFrozenCost = true
            frozenCostPosition = parsed
            idx += 2
        case "--frozen-cost-top-dispatches":
            guard idx + 1 < args.count,
                  let parsed = Int(args[idx + 1]), parsed >= 0 else {
                fputs("Error: --frozen-cost-top-dispatches requires a non-negative integer\n", stderr)
                exit(1)
            }
            includeFrozenCost = true
            frozenCostTopDispatches = parsed
            idx += 2
        default:
            if arg.hasPrefix("--") {
                fputs("Error: unknown optimizer-report option \(arg)\n", stderr)
                exit(1)
            }
            guard packagePath == nil else {
                fputs(optimizerReportUsage, stderr)
                exit(1)
            }
            packagePath = arg
            idx += 1
        }
    }
    guard let resolvedPackagePath = packagePath ?? inferPackagePathFromCWD() else {
        fputs(optimizerReportUsage, stderr)
        exit(1)
    }
    if frozenCostCalibrationPath != nil && frozenCostCalibrationOutputPath != nil {
        fputs("Error: use either --frozen-cost-calibration or --frozen-cost-calibrate, not both\n", stderr)
        exit(1)
    }
    if frozenCostDeltaJSONPath != nil && frozenCostBaselinePath == nil {
        fputs("Error: --frozen-cost-delta-json requires --frozen-cost-baseline\n", stderr)
        exit(1)
    }
    if frozenCostTable == .decode,
       let frozenCostSequenceLength,
       frozenCostSequenceLength != 1
    {
        fputs("Error: decode frozen cost requires --frozen-cost-sequence-length 1\n", stderr)
        exit(1)
    }
    if frozenCostTable == .prefillVerifyArgmax,
       frozenCostCalibrationOutputPath != nil
    {
        fputs(
            "Error: runtime calibration is currently supported for decode and normal prefill tables only\n",
            stderr
        )
        exit(1)
    }
    let camContext = requireCAMOptimizerReportAdmissionOrExit(
        packagePath: resolvedPackagePath,
        verb: "lab inspect cost"
    )

    do {
        let planComparisons = try planComparisonPaths.map { path in
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            return try JSONDecoder().decode(
                SmeltPlanComparisonReport.self,
                from: data
            )
        }
        let effectiveVerifyWeights = verifyWeights
            || frozenCostCalibrationOutputPath != nil
        var report = try SmeltOptimizerReportGenerator.markdown(
            packagePath: resolvedPackagePath,
            camContext: camContext,
            options: SmeltOptimizerReportOptions(
                verifyWeights: effectiveVerifyWeights,
                topPipelines: topPipelines,
                maxAgentTasks: maxAgentTasks,
                includeTimings: profileTimings,
                timingIterations: timingIterations,
                timingDecodePosition: timingDecodePosition,
                timingPrefillTokens: timingPrefillTokens,
                timingTopKernels: timingTopKernels,
                planComparisons: planComparisons
            )
        )
        if includeFrozenCost {
            let unloadedModel = try SmeltFrozenIRCostModel.load(
                packagePath: resolvedPackagePath,
                dispatchTable: frozenCostTable
            )
            var calibration: SmeltDeviceCostCalibration?
            if let path = frozenCostCalibrationPath {
                calibration = try JSONDecoder().decode(
                    SmeltDeviceCostCalibration.self,
                    from: Data(contentsOf: URL(fileURLWithPath: path))
                )
            } else {
                calibration = nil
            }
            let costContext = SmeltCostModelContext(
                mode: frozenCostTable.mode,
                sequenceLength: frozenCostSequenceLength
                    ?? (frozenCostTable.mode == .decode ? 1 : 256),
                position: frozenCostPosition
            )
            if let path = frozenCostCalibrationOutputPath {
                let calibrationRequest: SmeltCAMCapabilityRequest =
                    frozenCostTable.mode == .decode
                        ? .profileDecodeKernels
                        : .profilePrefillKernels
                let construction = requireCAMTextRuntimePlanOrExit(
                    packagePath: resolvedPackagePath,
                    request: calibrationRequest,
                    verb: "lab inspect cost",
                    requireAuthoredInventory: true
                )
                let runtime = try construction.makeRuntime(contextLimit: nil)
                calibration = try unloadedModel.calibrate(
                    runtime: runtime,
                    context: costContext,
                    warmup: frozenCostCalibrationWarmup,
                    iterations: frozenCostCalibrationIterations
                )
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                try encoder.encode(calibration).write(
                    to: URL(fileURLWithPath: path),
                    options: .atomic
                )
                fputs("Wrote frozen cost calibration: \(path)\n", stderr)
            }
            let costModel = SmeltFrozenIRCostModel(
                manifest: unloadedModel.manifest,
                records: unloadedModel.records,
                calibration: calibration,
                dispatchTable: unloadedModel.dispatchTable,
                dispatchTableSHA256: unloadedModel.dispatchTableSHA256
            )
            let frozenReport = try costModel.report(context: costContext)
            if !report.hasSuffix("\n") { report.append("\n") }
            report.append("\n")
            report.append(SmeltFrozenIRCostModel.markdown(
                frozenReport,
                topDispatches: frozenCostTopDispatches
            ))
            if let path = frozenCostBaselinePath {
                let baseline = try JSONDecoder().decode(
                    SmeltFrozenIRCostReport.self,
                    from: Data(contentsOf: URL(fileURLWithPath: path))
                )
                let delta = SmeltFrozenIRCostModel.delta(
                    baseline: baseline,
                    candidate: frozenReport
                )
                report.append("\n")
                report.append(SmeltFrozenIRCostModel.markdown(delta))
                if let output = frozenCostDeltaJSONPath {
                    try delta.encodeJSON().write(
                        to: URL(fileURLWithPath: output),
                        options: .atomic
                    )
                    fputs("Wrote frozen cost delta: \(output)\n", stderr)
                }
            }
            if let path = frozenCostJSONPath {
                try frozenReport.encodeJSON().write(
                    to: URL(fileURLWithPath: path),
                    options: .atomic
                )
                fputs("Wrote frozen cost report: \(path)\n", stderr)
            }
        }
        if let outputPath {
            try report.write(toFile: outputPath, atomically: true, encoding: .utf8)
            fputs("Wrote optimizer report: \(outputPath)\n", stderr)
        } else {
            print(report, terminator: "")
        }
    } catch {
        fputs("Optimizer report failed: \(error)\n", stderr)
        exit(1)
    }
}
