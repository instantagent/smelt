import Foundation
import SmeltRuntime
import SmeltSchema

/// Percentile (`p` ∈ `[0, 1]`) over an already-sorted array.
/// Caller sorts once and may invoke this multiple times.
func percentile(
    _ sortedValues: [Double],
    p: Double
) -> Double {
    guard !sortedValues.isEmpty else { return 0 }
    if sortedValues.count == 1 { return sortedValues[0] }
    let clamped = max(0, min(p, 1))
    let index = Int((Double(sortedValues.count - 1) * clamped).rounded(.up))
    return sortedValues[index]
}

struct DecodeBenchmarkResult {
    let warmupIterations: Int
    let medianTotalMs: Double
    let p95TotalMs: Double
    let medianPureGpuMs: Double
    let p95PureGpuMs: Double
    let medianCpuMs: Double
    let p95CpuMs: Double
    let medianSubmitGapMs: Double
    let p95SubmitGapMs: Double
    let medianReadMs: Double
    let p95ReadMs: Double
    let tokensPerSecond: Double
    let fixedPosition: Int32?
    let windowWallMs: Double?
    let windowTokensPerSecond: Double?
}

struct DecodeTraceSample {
    let position: Int32
    let totalMs: Double
    let pureGpuMs: Double
    let cpuMs: Double
    let submitGapMs: Double
    let readMs: Double
}

struct KernelTraceSample {
    let position: Int32
    let totalUs: Double
    let perKernelUs: [String: Double]
}

struct PrefillBenchmarkResult {
    let finalToken: Int32
    let warmupIterations: Int
    let medianWallMs: Double
    let p95WallMs: Double
    let medianPureGpuMs: Double
    let p95PureGpuMs: Double
    let medianCpuMs: Double
    let p95CpuMs: Double
    let medianSubmitGapMs: Double
    let p95SubmitGapMs: Double
    let medianReadMs: Double
    let p95ReadMs: Double
    let tokensPerSecond: Double
}

struct BenchmarkSettleConfig {
    let timeoutSeconds: Double
    let intervalSeconds: Double

    var isEnabled: Bool { timeoutSeconds > 0 }
}

struct PrefillParityResult {
    let tokenCount: Int
    let decodeToken: Int32
    let prefillToken: Int32
    let decodeContinuation: [Int32]
    let prefillContinuation: [Int32]

    var matches: Bool {
        decodeToken == prefillToken
            && decodeContinuation == prefillContinuation
    }
}

enum PrefillContinuationRoute: Equatable {
    case plain
    case suffixLookup(K: Int, maxNeedleLength: Int)

    var label: String {
        switch self {
        case .plain:
            return "plain"
        case .suffixLookup(let K, let maxNeedleLength):
            return "suffix(K=\(K),max-n=\(maxNeedleLength))"
        }
    }
}

func parseBenchmarkSettleConfig() -> BenchmarkSettleConfig {
    let timeout = Double(parseArg("--benchmark-settle-timeout", default: "")) ?? 0
    let interval = Double(parseArg("--benchmark-settle-interval", default: "")) ?? 15
    return BenchmarkSettleConfig(
        timeoutSeconds: max(timeout, 0),
        intervalSeconds: max(interval, 0.1)
    )
}

private func thermalStateDescription(_ state: ProcessInfo.ThermalState) -> String {
    switch state {
    case .nominal:
        return "nominal"
    case .fair:
        return "fair"
    case .serious:
        return "serious"
    case .critical:
        return "critical"
    @unknown default:
        return "unknown"
    }
}

func benchmarkPrefillWithSettle(
    packagePath: String,
    numTokens: Int,
    iterations: Int,
    warmupIterations: Int,
    minTps: Double?,
    maxP95Ms: Double?,
    config: BenchmarkSettleConfig,
    construction: CAMTextRuntimeConstruction? = nil
) throws -> PrefillBenchmarkResult {
    guard config.isEnabled, minTps != nil || maxP95Ms != nil else {
        return try benchmarkPrefill(
            packagePath: packagePath,
            numTokens: numTokens,
            iterations: iterations,
            warmupIterations: warmupIterations,
            construction: construction
        )
    }

    let start = CFAbsoluteTimeGetCurrent()
    let deadline = start + config.timeoutSeconds
    var attempts = 0

    while true {
        attempts += 1
        let result = try benchmarkPrefill(
            packagePath: packagePath,
            numTokens: numTokens,
            iterations: iterations,
            warmupIterations: warmupIterations,
            construction: construction
        )
        let failures = prefillGateFailures(
            result: result,
            tokenCount: numTokens,
            minTps: minTps,
            maxP95Ms: maxP95Ms
        )
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let thermal = thermalStateDescription(ProcessInfo.processInfo.thermalState)

        if failures.isEmpty {
            fputs(
                "  settle-\(numTokens): accepted attempt \(attempts),"
                    + " \(String(format: "%.1f", result.tokensPerSecond)) tok/s,"
                    + " p95=\(String(format: "%.1f", result.p95WallMs))ms,"
                    + " warmup=\(warmupIterations),"
                    + " thermal=\(thermal), waited=\(String(format: "%.1f", elapsed))s\n",
                stderr
            )
            return result
        }

        let remaining = deadline - CFAbsoluteTimeGetCurrent()
        if remaining <= 0 {
            fputs(
                "  settle-\(numTokens): timeout after \(String(format: "%.1f", elapsed))s;"
                    + " last attempt \(String(format: "%.1f", result.tokensPerSecond)) tok/s,"
                    + " p95=\(String(format: "%.1f", result.p95WallMs))ms,"
                    + " warmup=\(warmupIterations),"
                    + " thermal=\(thermal); using last benchmark\n",
                stderr
            )
            return result
        }

        let sleepSeconds = min(config.intervalSeconds, remaining)
        fputs(
            "  settle-\(numTokens): \(failures.joined(separator: "; "));"
                + " thermal=\(thermal); sleeping \(String(format: "%.1f", sleepSeconds))s\n",
            stderr
        )
        Thread.sleep(forTimeInterval: sleepSeconds)
    }
}

let decodeBenchmarkWrapPosition: Int32 = 200

func primeDecodeState(
    runtime: SmeltRuntime,
    targetPosition: Int32
) throws {
    // A positional benchmark knows its final cache extent up front. Reserve it
    // once so replay measures model execution instead of O(n²) one-token Metal
    // buffer growth/copies on an otherwise unprepared runtime.
    try runtime.ensureContextCapacity(Int(targetPosition) + 1)
    runtime.resetWorkingBuffers()
    guard targetPosition > 0 else { return }
    for position in 0..<targetPosition {
        _ = try runtime.decodeStep(tokenId: 0, position: position)
    }
}

@discardableResult
func warmSequentialDecode(
    runtime: SmeltRuntime,
    iterations: Int,
    wrapPosition: Int32 = decodeBenchmarkWrapPosition
) throws -> Int32 {
    runtime.resetWorkingBuffers()
    var nextPosition: Int32 = 0
    for _ in 0..<iterations {
        _ = try runtime.decodeStep(tokenId: 0, position: nextPosition)
        nextPosition += 1
        if nextPosition >= wrapPosition {
            runtime.resetWorkingBuffers()
            nextPosition = 0
        }
    }
    return nextPosition
}

func nextSequentialDecodePosition(
    runtime: SmeltRuntime,
    nextPosition: inout Int32,
    wrapPosition: Int32 = decodeBenchmarkWrapPosition
) -> Int32 {
    if nextPosition >= wrapPosition {
        runtime.resetWorkingBuffers()
        nextPosition = 0
    }
    let position = nextPosition
    nextPosition += 1
    return position
}

func benchmarkDecode(
    packagePath: String,
    iterations: Int,
    warmupIterations: Int = 5,
    fixedPosition: Int32? = nil,
    construction: CAMTextRuntimeConstruction? = nil,
    runtime providedRuntime: SmeltRuntime? = nil
) throws -> DecodeBenchmarkResult {
    try construction?.requirePackagePath(packagePath)
    let runtime: SmeltRuntime
    if let providedRuntime {
        runtime = providedRuntime
    } else {
        runtime = try construction?.makeRuntime(contextLimit: nil)
            ?? SmeltRuntime(packagePath: packagePath)
    }
    var nextPosition: Int32 = 0

    if let fixedPosition {
        for _ in 0..<warmupIterations {
            try primeDecodeState(runtime: runtime, targetPosition: fixedPosition)
            _ = try runtime.decodeStep(tokenId: 0, position: fixedPosition)
        }
    } else {
        nextPosition = try warmSequentialDecode(
            runtime: runtime,
            iterations: warmupIterations
        )
    }

    var totalMs: [Double] = []
    var pureGpuMs: [Double] = []
    var cpuMs: [Double] = []
    var submitGapMs: [Double] = []
    var readMs: [Double] = []
    let windowStart = DispatchTime.now()
    for _ in 0..<iterations {
        let position: Int32
        if let fixedPosition {
            try primeDecodeState(runtime: runtime, targetPosition: fixedPosition)
            position = fixedPosition
        } else {
            position = nextSequentialDecodePosition(
                runtime: runtime,
                nextPosition: &nextPosition
            )
        }
        let result = try runtime.profileDecodeStep(
            tokenId: 0,
            position: position
        )
        totalMs.append(result.cpuMs + result.gpuMs + result.readMs)
        pureGpuMs.append(result.pureGpuMs)
        cpuMs.append(result.cpuMs)
        submitGapMs.append(max(result.gpuMs - result.pureGpuMs, 0))
        readMs.append(result.readMs)
    }
    let windowElapsedMs =
        Double(DispatchTime.now().uptimeNanoseconds - windowStart.uptimeNanoseconds)
        / 1_000_000.0
    // Whole-window wall throughput is only meaningful for the varying-sequential
    // benchmark; fixed-position iterations replay a prefix inside the window.
    let windowWallMs: Double? = fixedPosition == nil ? windowElapsedMs : nil
    let windowTokensPerSecond: Double? =
        (fixedPosition == nil && windowElapsedMs > 0 && iterations > 0)
        ? Double(iterations) * 1_000.0 / windowElapsedMs
        : nil
    totalMs.sort()
    pureGpuMs.sort()
    cpuMs.sort()
    submitGapMs.sort()
    readMs.sort()
    let medianTotalMs = totalMs[iterations / 2]
    let p95TotalMs = percentile(totalMs, p: 0.95)
    let medianPureGpuMs = pureGpuMs[iterations / 2]
    let p95PureGpuMs = percentile(pureGpuMs, p: 0.95)
    let medianCpuMs = cpuMs[iterations / 2]
    let p95CpuMs = percentile(cpuMs, p: 0.95)
    let medianSubmitGapMs = submitGapMs[iterations / 2]
    let p95SubmitGapMs = percentile(submitGapMs, p: 0.95)
    let medianReadMs = readMs[iterations / 2]
    let p95ReadMs = percentile(readMs, p: 0.95)
    let tokensPerSecond = medianTotalMs > 0 ? 1_000.0 / medianTotalMs : 0
    return DecodeBenchmarkResult(
        warmupIterations: warmupIterations,
        medianTotalMs: medianTotalMs,
        p95TotalMs: p95TotalMs,
        medianPureGpuMs: medianPureGpuMs,
        p95PureGpuMs: p95PureGpuMs,
        medianCpuMs: medianCpuMs,
        p95CpuMs: p95CpuMs,
        medianSubmitGapMs: medianSubmitGapMs,
        p95SubmitGapMs: p95SubmitGapMs,
        medianReadMs: medianReadMs,
        p95ReadMs: p95ReadMs,
        tokensPerSecond: tokensPerSecond,
        fixedPosition: fixedPosition,
        windowWallMs: windowWallMs,
        windowTokensPerSecond: windowTokensPerSecond
    )
}

func traceDecode(
    packagePath: String,
    startPosition: Int32,
    count: Int,
    warmupIterations: Int = 1,
    construction: CAMTextRuntimeConstruction? = nil
) throws -> [DecodeTraceSample] {
    try construction?.requirePackagePath(packagePath)
    let runtime = try construction?.makeRuntime(contextLimit: nil)
        ?? SmeltRuntime(packagePath: packagePath)

    let endPosition = startPosition + Int32(count)
    // The trace continues beyond its primed prefix, so reserve the complete
    // measured range before either warmup or collection begins.
    try runtime.ensureContextCapacity(Int(endPosition))
    for _ in 0..<warmupIterations {
        try primeDecodeState(runtime: runtime, targetPosition: startPosition)
        for position in startPosition..<endPosition {
            _ = try runtime.decodeStep(tokenId: 0, position: position)
        }
    }

    try primeDecodeState(runtime: runtime, targetPosition: startPosition)
    var samples: [DecodeTraceSample] = []
    for position in startPosition..<endPosition {
        let result = try runtime.profileDecodeStep(tokenId: 0, position: position)
        samples.append(
            DecodeTraceSample(
                position: position,
                totalMs: result.cpuMs + result.gpuMs + result.readMs,
                pureGpuMs: result.pureGpuMs,
                cpuMs: result.cpuMs,
                submitGapMs: max(result.gpuMs - result.pureGpuMs, 0),
                readMs: result.readMs
            )
        )
    }
    return samples
}

func printDecodeBenchmarkSummary(
    _ result: DecodeBenchmarkResult,
    iterations: Int
) {
    fputs("Results:\n", stderr)
    fputs("  Warmup:       \(result.warmupIterations)\n", stderr)
    fputs("  Iterations:   \(iterations)\n", stderr)
    if let fixedPosition = result.fixedPosition {
        fputs("  Position:     fixed \(fixedPosition)\n", stderr)
    } else {
        fputs("  Position:     varying sequential (wrap at \(decodeBenchmarkWrapPosition))\n", stderr)
    }
    fputs(
        "  Median:       \(String(format: "%.2f", result.medianTotalMs))ms/tok"
            + "  (\(String(format: "%.1f", result.tokensPerSecond)) tok/s)\n",
        stderr
    )
    fputs(
        "  P95:          \(String(format: "%.2f", result.p95TotalMs))ms/tok\n",
        stderr
    )
    fputs(
        "  Pure GPU med: \(String(format: "%.2f", result.medianPureGpuMs))ms/tok\n",
        stderr
    )
    fputs(
        "  Pure GPU p95: \(String(format: "%.2f", result.p95PureGpuMs))ms/tok\n",
        stderr
    )
    fputs(
        "  CPU med:      \(String(format: "%.3f", result.medianCpuMs))ms/tok\n",
        stderr
    )
    fputs(
        "  CPU p95:      \(String(format: "%.3f", result.p95CpuMs))ms/tok\n",
        stderr
    )
    fputs(
        "  Submit med:   \(String(format: "%.3f", result.medianSubmitGapMs))ms/tok\n",
        stderr
    )
    fputs(
        "  Submit p95:   \(String(format: "%.3f", result.p95SubmitGapMs))ms/tok\n",
        stderr
    )
    fputs(
        "  Read med:     \(String(format: "%.3f", result.medianReadMs))ms/tok\n",
        stderr
    )
    fputs(
        "  Read p95:     \(String(format: "%.3f", result.p95ReadMs))ms/tok\n",
        stderr
    )
    if let windowWallMs = result.windowWallMs,
        let windowTps = result.windowTokensPerSecond {
        fputs(
            "  Window wall:  \(String(format: "%.1f", windowWallMs))ms"
                + "  (\(String(format: "%.1f", windowTps)) tok/s over \(iterations) steps)\n",
            stderr
        )
    }
}

func printDecodePositionSweep(
    _ results: [DecodeBenchmarkResult],
    iterations: Int
) {
    fputs("Position sweep:\n", stderr)
    let positions = results.compactMap(\.fixedPosition).map(String.init).joined(separator: ", ")
    fputs("  Warmup:       \(results.first?.warmupIterations ?? 0)\n", stderr)
    fputs("  Iterations:   \(iterations)\n", stderr)
    fputs("  Positions:    \(positions)\n", stderr)
    fputs("  Note:         fixed-position samples replay a zero-token prefix before each measured step\n\n", stderr)

    let header =
        "  Pos   Median    P95   GPU med  GPU p95  CPU med Submit med Submit p95"
    fputs(header + "\n", stderr)
    fputs("  " + String(repeating: "─", count: header.count - 2) + "\n", stderr)
    for result in results {
        let pos = result.fixedPosition ?? -1
        let line = String(
            format:
                "  %3d  %7.2f  %6.2f  %7.2f  %7.2f  %7.3f  %10.3f  %10.3f",
            pos,
            result.medianTotalMs,
            result.p95TotalMs,
            result.medianPureGpuMs,
            result.p95PureGpuMs,
            result.medianCpuMs,
            result.medianSubmitGapMs,
            result.p95SubmitGapMs
        )
        fputs(line + "\n", stderr)
    }
}

func printDecodeTrace(
    _ samples: [DecodeTraceSample],
    startPosition: Int32,
    warmupIterations: Int
) {
    fputs("Decode trace:\n", stderr)
    fputs("  Warmup:       \(warmupIterations)\n", stderr)
    fputs("  Start:        \(startPosition)\n", stderr)
    fputs("  Count:        \(samples.count)\n\n", stderr)

    let header = "  Pos   Total   GPU   CPU  Submit  Read"
    fputs(header + "\n", stderr)
    fputs("  " + String(repeating: "─", count: header.count - 2) + "\n", stderr)
    for sample in samples {
        let line = String(
            format: "  %3d  %6.2f %5.2f %5.3f %7.3f %5.3f",
            sample.position,
            sample.totalMs,
            sample.pureGpuMs,
            sample.cpuMs,
            sample.submitGapMs,
            sample.readMs
        )
        fputs(line + "\n", stderr)
    }

    let totals = samples.map(\.totalMs).sorted()
    let pureGpu = samples.map(\.pureGpuMs).sorted()
    let submit = samples.map(\.submitGapMs).sorted()
    fputs("  " + String(repeating: "─", count: header.count - 2) + "\n", stderr)
    fputs(
        String(
            format:
                "  med  %6.2f %5.2f   --- %7.3f   ---\n",
            totals[totals.count / 2],
            pureGpu[pureGpu.count / 2],
            submit[submit.count / 2]
        ),
        stderr
    )
    fputs(
        String(
            format:
                "  p95  %6.2f %5.2f   --- %7.3f   ---\n",
            percentile(totals, p: 0.95),
            percentile(pureGpu, p: 0.95),
            percentile(submit, p: 0.95)
        ),
        stderr
    )
}

func printKernelTrace(
    _ samples: [KernelTraceSample],
    startPosition: Int32
) {
    guard !samples.isEmpty else { return }

    var kernelSums: [String: Double] = [:]
    for sample in samples {
        for (name, us) in sample.perKernelUs {
            kernelSums[name, default: 0] += us
        }
    }
    let focusKernels = kernelSums
        .sorted { $0.value > $1.value }
        .prefix(6)
        .map(\.key)

    func label(for name: String) -> String {
        if name.hasPrefix("attention_decode") { return "attention" }
        if name.hasPrefix("deltanet_recurrence") { return "recurrence" }
        if name.hasPrefix("fused_affine_gate_up_swiglu") { return "ffn_gate_up" }
        if name.hasPrefix("affine_matvec_c6144_r2048") { return "down_proj" }
        if name.hasPrefix("affine_matvec_c2048_r6144") { return "up_proj" }
        if name.hasPrefix("fused_affine_matvec_add_c6144") { return "proj_add_6k" }
        return String(name.prefix(12))
    }

    fputs("Kernel trace:\n", stderr)
    fputs("  Start:        \(startPosition)\n", stderr)
    fputs("  Count:        \(samples.count)\n\n", stderr)

    let kernelLabels = focusKernels.map(label)
    let header = "  Pos   Total " + kernelLabels.map { $0.padding(toLength: 12, withPad: " ", startingAt: 0) }.joined(separator: " ")
    fputs(header + "\n", stderr)
    fputs("  " + String(repeating: "─", count: max(header.count - 2, 0)) + "\n", stderr)
    for sample in samples {
        var line = String(format: "  %3d  %6.0f", sample.position, sample.totalUs)
        for name in focusKernels {
            let us = sample.perKernelUs[name, default: 0]
            line += " " + String(format: "%12.0f", us)
        }
        fputs(line + "\n", stderr)
    }
}

func benchmarkPrefill(
    packagePath: String,
    numTokens: Int,
    iterations: Int,
    warmupIterations: Int = 2,
    construction: CAMTextRuntimeConstruction? = nil
) throws -> PrefillBenchmarkResult {
    try construction?.requirePackagePath(packagePath)
    let runtime = try construction?.makeRuntime(contextLimit: nil)
        ?? SmeltRuntime(packagePath: packagePath)
    guard runtime.hasMetalPrefill else {
        throw NSError(
            domain: "SmeltCLI",
            code: 1,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "Package does not have Metal prefill (engine != metal)"
            ]
        )
    }

    let tokenIds = (0..<numTokens).map { Int32($0) }
    let chunkSize = max(runtime.maxPrefillBatchSize, 1)

    func runPrefillPass() throws -> Int32 {
        var cur: Int32 = 0
        var start = 0
        while start < tokenIds.count {
            let end = min(start + chunkSize, tokenIds.count)
            cur = try runtime.prefillStep(
                tokenIds: Array(tokenIds[start..<end]),
                startPos: Int32(start)
            )
            start = end
        }
        return cur
    }

    func profilePrefillPass() throws -> (
        token: Int32, cpuMs: Double, gpuMs: Double, readMs: Double, pureGpuMs: Double
    ) {
        var cur: Int32 = 0
        var start = 0
        var cpuMs = 0.0
        var gpuMs = 0.0
        var readMs = 0.0
        var pureGpuMs = 0.0
        while start < tokenIds.count {
            let end = min(start + chunkSize, tokenIds.count)
            let result = try runtime.profilePrefillStep(
                tokenIds: Array(tokenIds[start..<end]),
                startPos: Int32(start)
            )
            cur = result.token
            cpuMs += result.cpuMs
            gpuMs += result.gpuMs
            readMs += result.readMs
            pureGpuMs += result.pureGpuMs
            start = end
        }
        return (
            token: cur,
            cpuMs: cpuMs,
            gpuMs: gpuMs,
            readMs: readMs,
            pureGpuMs: pureGpuMs
        )
    }

    for _ in 0..<warmupIterations {
        runtime.resetWorkingBuffers()
        _ = try runPrefillPass()
    }

    var wallTimesMs: [Double] = []
    var pureGpuMs: [Double] = []
    var cpuMs: [Double] = []
    var submitGapMs: [Double] = []
    var readMs: [Double] = []
    var finalToken: Int32 = 0
    for _ in 0..<iterations {
        runtime.resetWorkingBuffers()
        let start = CFAbsoluteTimeGetCurrent()
        let result = try profilePrefillPass()
        finalToken = result.token
        let elapsedMs = (CFAbsoluteTimeGetCurrent() - start) * 1_000.0
        wallTimesMs.append(elapsedMs)
        pureGpuMs.append(result.pureGpuMs)
        cpuMs.append(result.cpuMs)
        submitGapMs.append(max(result.gpuMs - result.pureGpuMs, 0))
        readMs.append(result.readMs)
    }

    wallTimesMs.sort()
    pureGpuMs.sort()
    cpuMs.sort()
    submitGapMs.sort()
    readMs.sort()
    let medianWallMs = wallTimesMs[iterations / 2]
    let p95WallMs = percentile(wallTimesMs, p: 0.95)
    let medianPureGpuMs = pureGpuMs[iterations / 2]
    let p95PureGpuMs = percentile(pureGpuMs, p: 0.95)
    let medianCpuMs = cpuMs[iterations / 2]
    let p95CpuMs = percentile(cpuMs, p: 0.95)
    let medianSubmitGapMs = submitGapMs[iterations / 2]
    let p95SubmitGapMs = percentile(submitGapMs, p: 0.95)
    let medianReadMs = readMs[iterations / 2]
    let p95ReadMs = percentile(readMs, p: 0.95)
    let tokensPerSecond = medianWallMs > 0
        ? (Double(numTokens) * 1_000.0) / medianWallMs
        : 0
    return PrefillBenchmarkResult(
        finalToken: finalToken,
        warmupIterations: warmupIterations,
        medianWallMs: medianWallMs,
        p95WallMs: p95WallMs,
        medianPureGpuMs: medianPureGpuMs,
        p95PureGpuMs: p95PureGpuMs,
        medianCpuMs: medianCpuMs,
        p95CpuMs: p95CpuMs,
        medianSubmitGapMs: medianSubmitGapMs,
        p95SubmitGapMs: p95SubmitGapMs,
        medianReadMs: medianReadMs,
        p95ReadMs: p95ReadMs,
        tokensPerSecond: tokensPerSecond
    )
}

func verifyPrefillParity(
    packagePath: String,
    tokenCounts: [Int],
    continuationTokenCount: Int = 0,
    continuationRoute: PrefillContinuationRoute = .plain,
    template: String,
    construction: CAMTextRuntimeConstruction? = nil
) throws -> [PrefillParityResult] {
    let checkpoints = Array(Set(tokenCounts.filter { $0 > 0 })).sorted()
    guard let maxTokenCount = checkpoints.last else { return [] }

    let (_, tokenPrefix) = try buildRepeatedVerifyPrompt(
        packagePath: packagePath,
        template: template,
        maxTokenCount: maxTokenCount,
        tokenizer: try construction?.makeTokenizer()
    )

    try construction?.requirePackagePath(packagePath)
    let runtime = try construction?.makeRuntime(contextLimit: nil)
        ?? SmeltRuntime(packagePath: packagePath)
    guard runtime.hasMetalPrefill else { return [] }

    // Checkpoint one continuous reference pass. Long-context gates commonly
    // request several increasing lengths (for example 5K and 10K); replaying
    // every prefix from zero makes their cost cumulative for no semantic gain.
    let checkpointSet = Set(checkpoints)
    var decodeTokens: [Int: Int32] = [:]
    var decodeContinuations: [Int: [Int32]] = [:]
    // A speculative round commits a whole verified block. The parity result
    // is clipped to `continuationTokenCount`, but the last round may legally
    // overshoot that count by up to K-1 tokens before the checkpoint is
    // restored. Reserve that physical target state up front as well.
    let continuationOverflow: Int
    switch continuationRoute {
    case .plain:
        continuationOverflow = 0
    case .suffixLookup(let K, _):
        continuationOverflow = K
    }
    let requestContextCapacity = maxTokenCount
        + continuationTokenCount
        + continuationOverflow
    // Plan the final KV/RoPE capacity once. Calling decodeStep against an
    // unplanned long prefix otherwise grows every context-scoped Metal buffer
    // one token at a time, causing quadratic allocation/copy churn and enough
    // transient unified-memory pressure for macOS to kill a 27B verification.
    // Include the post-prefix continuation for the same reason.
    try runtime.prepareForRequest(
        batchCapacity: 1,
        contextCapacity: requestContextCapacity
    )
    runtime.resetWorkingBuffers()
    for (position, tokenId) in tokenPrefix.prefix(maxTokenCount).enumerated() {
        let token = try runtime.decodeStep(
            tokenId: tokenId,
            position: Int32(position)
        )
        let tokenCount = position + 1
        if checkpointSet.contains(tokenCount) {
            decodeTokens[tokenCount] = token
            decodeContinuations[tokenCount] = try checkpointContinuation(
                runtime: runtime,
                route: .plain,
                prefixTokens: Array(tokenPrefix.prefix(tokenCount)),
                firstToken: token,
                tokenCount: continuationTokenCount,
                restoreAfter: tokenCount < maxTokenCount
            )
        }
    }

    // Split chunks at checkpoint boundaries so each recorded token is exactly
    // the same prefix endpoint as the sequential reference while still using
    // the package's normal maximum prefill batch everywhere else.
    var prefillTokens: [Int: Int32] = [:]
    var prefillContinuations: [Int: [Int32]] = [:]
    let chunkSize = max(runtime.maxPrefillBatchSize, 1)
    try runtime.prepareForRequest(
        batchCapacity: min(chunkSize, maxTokenCount),
        contextCapacity: requestContextCapacity
    )
    runtime.resetWorkingBuffers()
    var start = 0
    for checkpoint in checkpoints {
        var token: Int32 = 0
        while start < checkpoint {
            let end = min(start + chunkSize, checkpoint)
            token = try runtime.prefillStep(
                tokenIds: Array(tokenPrefix[start..<end]),
                startPos: Int32(start)
            )
            start = end
        }
        prefillTokens[checkpoint] = token
        prefillContinuations[checkpoint] = try checkpointContinuation(
            runtime: runtime,
            route: continuationRoute,
            prefixTokens: Array(tokenPrefix.prefix(checkpoint)),
            firstToken: token,
            tokenCount: continuationTokenCount,
            restoreAfter: checkpoint < maxTokenCount
        )
    }

    return tokenCounts.compactMap { tokenCount in
        guard tokenCount > 0,
              let decodeToken = decodeTokens[tokenCount],
              let prefillToken = prefillTokens[tokenCount]
        else { return nil }
        return PrefillParityResult(
            tokenCount: tokenCount,
            decodeToken: decodeToken,
            prefillToken: prefillToken,
            decodeContinuation: decodeContinuations[tokenCount] ?? [],
            prefillContinuation: prefillContinuations[tokenCount] ?? []
        )
    }
}

/// Generate a checkpoint continuation, then restore the exact prompt state if
/// the caller still has later checkpoints to consume. This makes 5K/10K gates
/// one continuous prefix pass without allowing the 5K generated tail to
/// contaminate the 10K prompt state.
private func checkpointContinuation(
    runtime: SmeltRuntime,
    route: PrefillContinuationRoute,
    prefixTokens: [Int32],
    firstToken: Int32,
    tokenCount: Int,
    restoreAfter: Bool
) throws -> [Int32] {
    guard tokenCount > 0 else { return [] }
    let snapshot = restoreAfter
        ? runtime.capturePromptSnapshot(
            capturedLength: prefixTokens.count,
            promptLength: prefixTokens.count,
            nextToken: firstToken
        )
        : nil

    let continuation: [Int32]
    switch route {
    case .plain:
        continuation = try generateParityContinuation(
            runtime: runtime,
            firstToken: firstToken,
            startPosition: prefixTokens.count,
            tokenCount: tokenCount
        )
    case .suffixLookup(let K, let maxNeedleLength):
        continuation = try generateSuffixParityContinuation(
            runtime: runtime,
            prefixTokens: prefixTokens,
            firstToken: firstToken,
            K: K,
            maxNeedleLength: maxNeedleLength,
            tokenCount: tokenCount
        )
    }

    if let snapshot {
        runtime.resetWorkingBuffers()
        try runtime.restorePromptSnapshot(snapshot)
    }
    return continuation
}

/// Continue from an already-consumed prefix. `firstToken` is the model output
/// at `startPosition`, so it is the first generated token and the next decode
/// consumes it. Comparing more than the endpoint catches state handoffs that
/// preserve one argmax but quickly drift into repetition.
private func generateParityContinuation(
    runtime: SmeltRuntime,
    firstToken: Int32?,
    startPosition: Int,
    tokenCount: Int
) throws -> [Int32] {
    guard tokenCount > 0, var token = firstToken else { return [] }
    var result: [Int32] = []
    result.reserveCapacity(tokenCount)
    for offset in 0..<tokenCount {
        result.append(token)
        if offset + 1 < tokenCount {
            token = try runtime.decodeStep(
                tokenId: token,
                position: Int32(startPosition + offset)
            )
        }
    }
    return result
}

/// Continue through the same history-only preflight + target-verification path
/// used by an interactive suffix drafter. The returned prefix is clipped to
/// the requested count, but the runtime is allowed to commit a complete
/// verified block internally before the checkpoint is restored.
private func generateSuffixParityContinuation(
    runtime: SmeltRuntime,
    prefixTokens: [Int32],
    firstToken: Int32,
    K: Int,
    maxNeedleLength: Int,
    tokenCount: Int
) throws -> [Int32] {
    guard tokenCount > 0, let lastToken = prefixTokens.last else { return [] }
    let speculative = try SmeltSpeculativeRuntime(
        target: runtime,
        drafter: SmeltSuffixLookupDrafter(maxNeedleLength: maxNeedleLength),
        K: K
    )
    speculative.adoptCurrentTargetLogits(argmax: firstToken)
    speculative.resetSuffixCache(promptTokens: prefixTokens)

    var generated: [Int32] = []
    generated.reserveCapacity(tokenCount + K)
    var liveToken = lastToken
    var livePosition = Int32(prefixTokens.count - 1)
    while generated.count < tokenCount {
        let result = try speculative.decodeStep(
            lastToken: liveToken,
            position: livePosition,
            selectionMode: .argmax
        )
        guard !result.committedTokens.isEmpty else {
            throw SmeltSpeculativeRuntimeError.invalidConfiguration(
                "suffix continuation made no progress"
            )
        }
        generated.append(contentsOf: result.committedTokens)
        speculative.recordGeneratedTokens(generated)
        liveToken = result.nextToken
        livePosition = result.nextPosition
    }
    return Array(generated.prefix(tokenCount))
}
