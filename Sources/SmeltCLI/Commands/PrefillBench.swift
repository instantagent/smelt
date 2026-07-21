import Foundation

func runPrefillBenchCommand(_ args: [String]) {
    guard args.count >= 3 else {
        fputs(
            "Usage: smelt lab bench prefill <model.smeltpkg> [--tokens N] [--iterations N] [--warmup N] [--min-tps N] [--max-p95-ms N] [--benchmark-settle-timeout N] [--benchmark-settle-interval N]\n",
            stderr
        )
        exit(1)
    }
    let pkgPath = args[2]
    if args.contains("--gate-qwen35") {
        fputs(
            "smelt lab bench prefill: --gate-qwen35 was removed; use module gate contracts or explicit --min-tps/--max-p95-ms\n",
            stderr
        )
        exit(1)
    }
    let construction = requireCAMTextRuntimePlanOrExit(
        packagePath: pkgPath,
        request: .benchPrefill,
        verb: "lab bench prefill"
    )
    let numTokens = Int(parseArg(args, "--tokens", default: "256")) ?? 256
    let iterations = Int(parseArg(args, "--iterations", default: "10")) ?? 10
    let warmupIterations = Int(parseArg(args, "--warmup", default: "2")) ?? 2
    let minTpsArg = Double(parseArg(args, "--min-tps", default: ""))
    let maxP95MsArg = Double(parseArg(args, "--max-p95-ms", default: ""))
    let settleConfig = parseBenchmarkSettleConfig()

    do {
        let result = try benchmarkPrefillWithSettle(
            packagePath: pkgPath,
            numTokens: numTokens,
            iterations: iterations,
            warmupIterations: warmupIterations,
            minTps: minTpsArg,
            maxP95Ms: maxP95MsArg,
            config: settleConfig,
            construction: construction
        )
        fputs(
            "Benchmarking Metal prefill: \(pkgPath) (\(iterations) iterations, \(numTokens) tokens)\n",
            stderr
        )
        fputs("Results:\n", stderr)
        fputs("  Warmup:       \(result.warmupIterations)\n", stderr)
        fputs("  Final token:  \(result.finalToken)\n", stderr)
        fputs(
            "  Wall time:    \(String(format: "%.1f", result.medianWallMs))ms/prefill"
                + "  (median of \(iterations))\n",
            stderr
        )
        fputs(
            "  P95:          \(String(format: "%.1f", result.p95WallMs))ms/prefill\n",
            stderr
        )
        fputs(
            "  Pure GPU med: \(String(format: "%.1f", result.medianPureGpuMs))ms/prefill\n",
            stderr
        )
        fputs(
            "  Pure GPU p95: \(String(format: "%.1f", result.p95PureGpuMs))ms/prefill\n",
            stderr
        )
        fputs(
            "  CPU med:      \(String(format: "%.3f", result.medianCpuMs))ms/prefill\n",
            stderr
        )
        fputs(
            "  CPU p95:      \(String(format: "%.3f", result.p95CpuMs))ms/prefill\n",
            stderr
        )
        fputs(
            "  Submit med:   \(String(format: "%.3f", result.medianSubmitGapMs))ms/prefill\n",
            stderr
        )
        fputs(
            "  Submit p95:   \(String(format: "%.3f", result.p95SubmitGapMs))ms/prefill\n",
            stderr
        )
        fputs(
            "  Read med:     \(String(format: "%.3f", result.medianReadMs))ms/prefill\n",
            stderr
        )
        fputs(
            "  Read p95:     \(String(format: "%.3f", result.p95ReadMs))ms/prefill\n",
            stderr
        )
        fputs(
            "  Tokens/sec:   \(String(format: "%.1f", result.tokensPerSecond))\n",
            stderr
        )
        let failures = prefillGateFailures(
            result: result,
            tokenCount: numTokens,
            minTps: minTpsArg,
            maxP95Ms: maxP95MsArg
        )
        if !failures.isEmpty {
            throw makeGateFailure(failures.joined(separator: "; "))
        }
    } catch {
        fputs("Prefill bench failed: \(error)\n", stderr)
        exit((error as NSError).domain == "SmeltBenchmarkGate" ? 2 : 1)
    }
}
