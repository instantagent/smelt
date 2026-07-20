import Foundation

func runBenchCommand() {
    guard args.count >= 3 else {
        fputs(
            "Usage: smelt bench <model.smeltpkg> [--iterations N] [--warmup N] [--fixed-position N] [--positions 0,32,64,...] [--trace-start N --trace-length N] [--min-tps N] [--max-p95-ms N]\n",
            stderr
        )
        exit(1)
    }
    let pkgPath = args[2]
    if args.contains("--gate-qwen35") {
        fputs(
            "smelt bench: --gate-qwen35 was removed; use module gate contracts or explicit --min-tps/--max-p95-ms\n",
            stderr
        )
        exit(1)
    }
    let iterations = Int(parseArg("--iterations", default: "100")) ?? 100
    let warmupIterations = Int(parseArg("--warmup", default: "5")) ?? 5
    let fixedPosition = Int32(parseArg("--fixed-position", default: ""))
    let sweepPositions = parseCSVNonNegativeInts(parseArg("--positions", default: ""))
    let traceStart = Int32(parseArg("--trace-start", default: ""))
    let traceLength = Int(parseArg("--trace-length", default: "0")) ?? 0
    let minTpsArg = Double(parseArg("--min-tps", default: ""))
    let maxP95MsArg = Double(parseArg("--max-p95-ms", default: ""))

    let construction = requireBenchTextRuntimePlanOrExit(packagePath: pkgPath)

    do {
        let minTps = minTpsArg
        let maxP95Ms = maxP95MsArg
        let hasTrace = traceStart != nil || traceLength > 0
        let hasGates = minTps != nil || maxP95Ms != nil
        if fixedPosition != nil && !sweepPositions.isEmpty {
            throw NSError(
                domain: "SmeltCLI",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Use either --fixed-position or --positions, not both"
                ]
            )
        }
        if hasTrace && (fixedPosition != nil || !sweepPositions.isEmpty) {
            throw NSError(
                domain: "SmeltCLI",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Use trace mode by itself, without --fixed-position or --positions"
                ]
            )
        }
        if hasTrace && (traceStart == nil || traceLength <= 0) {
            throw NSError(
                domain: "SmeltCLI",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Trace mode requires both --trace-start N and --trace-length N"
                ]
            )
        }
        if hasGates && (fixedPosition != nil || !sweepPositions.isEmpty || hasTrace) {
            throw NSError(
                domain: "SmeltCLI",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Decode performance gates apply only to the standard varying-position benchmark"
                ]
            )
        }

        if hasTrace, let traceStart {
            let samples = try traceDecode(
                packagePath: pkgPath,
                startPosition: traceStart,
                count: traceLength,
                warmupIterations: warmupIterations,
                construction: construction
            )
            printDecodeTrace(
                samples,
                startPosition: traceStart,
                warmupIterations: warmupIterations
            )
        } else if !sweepPositions.isEmpty {
            let runtime = try construction.makeRuntime(contextLimit: nil)
            let results = try sweepPositions.map { position in
                try benchmarkDecode(
                    packagePath: pkgPath,
                    iterations: iterations,
                    warmupIterations: warmupIterations,
                    fixedPosition: Int32(position),
                    construction: construction,
                    runtime: runtime
                )
            }
            printDecodePositionSweep(results, iterations: iterations)
        } else {
            let result = try benchmarkDecode(
                packagePath: pkgPath,
                iterations: iterations,
                warmupIterations: warmupIterations,
                fixedPosition: fixedPosition,
                construction: construction
            )
            printDecodeBenchmarkSummary(result, iterations: iterations)
            let failures = decodeGateFailures(
                result: result,
                minTps: minTps,
                maxP95Ms: maxP95Ms
            )
            if !failures.isEmpty {
                throw makeGateFailure(failures.joined(separator: "; "))
            }
        }
    } catch {
        fputs("Bench failed: \(error)\n", stderr)
        exit((error as NSError).domain == "SmeltBenchmarkGate" ? 2 : 1)
    }
}
