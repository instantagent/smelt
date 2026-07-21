import Foundation
import SmeltRuntime
import SmeltServe

func runBenchLogprobsCommand(_ args: [String]) {
    let (packagePath, _) = resolvePackagePath(arguments: args, usage: [
        "Usage: smelt lab bench logprobs <model.smeltpkg>",
        "       [--prompt-tokens N (default 150)]",
        "       [--iters N (default 5)]",
        "       [--topk N (default 1)]\n",
    ])
    let construction = requireCAMTextRuntimePlanOrExit(
        packagePath: packagePath,
        request: .benchPrefillLogprobs,
        verb: "lab bench logprobs"
    )

    let promptTokens = Int(parseArg(args, "--prompt-tokens", default: "150")) ?? 150
    let iters = Int(parseArg(args, "--iters", default: "5")) ?? 5
    let topK = Int(parseArg(args, "--topk", default: "1")) ?? 1

    do {
        try runBenchLogprobs(
            packagePath: packagePath,
            construction: construction,
            promptTokens: promptTokens,
            iters: iters,
            topK: topK
        )
    } catch {
        fputs("bench-logprobs failed: \(error)\n", stderr)
        exit(1)
    }
}

private func runBenchLogprobs(
    packagePath: String,
    construction: CAMTextRuntimeConstruction,
    promptTokens: Int,
    iters: Int,
    topK: Int
) throws {
    try construction.requirePackagePath(packagePath)
    try construction.requirePrefillAllLogitsCapacity(tokenCount: promptTokens)
    let runtime = try construction.makeRuntime(contextLimit: nil)
    guard runtime.canEmitPrefillLogits(for: promptTokens) else {
        fputs(
            "Package does not support prefillAllLogits at \(promptTokens) tokens "
            + "(emit_all_logits=\(runtime.supportsChunkedPrefillVerify), "
            + "numDeltaLayers=\(runtime.numDeltaLayers), "
            + "maxPrefillBatch=\(runtime.maxPrefillBatchSize))\n",
            stderr
        )
        exit(1)
    }
    let vocab = Int(runtime.vocabSize)

    // Synthetic prompt: pad with a common token id so the shape is realistic.
    // Real token content doesn't matter for kernel timing; the path is the same.
    let inputIds: [Int32] = Array(repeating: 100, count: promptTokens)

    fputs("--- bench-logprobs ---\n", stderr)
    fputs("package:        \(packagePath)\n", stderr)
    fputs("vocab:          \(vocab)\n", stderr)
    fputs("prompt tokens:  \(promptTokens)\n", stderr)
    fputs("iters:          \(iters)\n", stderr)
    fputs("topK:           \(topK)\n", stderr)
    fputs("------------------------------------------\n", stderr)

    try runtime.prepareForRequest(batchCapacity: 1, contextCapacity: promptTokens + 16)

    // Warmup (one untimed iteration so caches / JIT settle).
    _ = try runtime.prefillAllLogits(tokens: inputIds, startPos: 0)

    var prefillTimings: [Double] = []
    var computeAllTimings: [Double] = []
    var computeOneTiming: Double = 0

    for _ in 0 ..< iters {
        try runtime.prepareForRequest(batchCapacity: 1, contextCapacity: promptTokens + 16)

        let pfStart = CFAbsoluteTimeGetCurrent()
        let rows = try runtime.prefillAllLogits(tokens: inputIds, startPos: 0)
        let pfElapsed = CFAbsoluteTimeGetCurrent() - pfStart
        prefillTimings.append(pfElapsed)

        let cStart = CFAbsoluteTimeGetCurrent()
        var sink: Double = 0  // prevent compiler from dead-eliminating
        for i in 1 ..< rows.count {
            let entry = LogprobsCompute.computeFromLogits(
                logits: rows[i - 1],
                chosenToken: inputIds[i],
                topK: topK
            )
            sink += entry.logprob
        }
        let cElapsed = CFAbsoluteTimeGetCurrent() - cStart
        computeAllTimings.append(cElapsed)
        if computeOneTiming == 0, rows.count > 1 {
            // Measure a single computeFromLogits in isolation.
            let oneStart = CFAbsoluteTimeGetCurrent()
            _ = LogprobsCompute.computeFromLogits(
                logits: rows.last!,
                chosenToken: inputIds[0],
                topK: topK
            )
            computeOneTiming = CFAbsoluteTimeGetCurrent() - oneStart
        }
        _ = sink
    }

    let pfMean = prefillTimings.reduce(0, +) / Double(iters)
    let cMean  = computeAllTimings.reduce(0, +) / Double(iters)
    let total  = pfMean + cMean

    fputs(String(format: "prefillAllLogits:     %7.1f ms  (%5.1f ms/tok)\n",
                 pfMean * 1000, pfMean * 1000 / Double(promptTokens)), stderr)
    fputs(String(format: "computeFromLogits ×N: %7.1f ms  (%5.1f ms/row, %d rows)\n",
                 cMean * 1000, cMean * 1000 / Double(max(1, promptTokens - 1)),
                 promptTokens - 1), stderr)
    fputs(String(format: "computeFromLogits ×1: %7.1f ms\n",
                 computeOneTiming * 1000), stderr)
    fputs(String(format: "total per request:    %7.1f ms\n", total * 1000), stderr)
    fputs(String(format: "  prefill share:      %5.1f%%\n", pfMean / total * 100), stderr)
    fputs(String(format: "  compute  share:     %5.1f%%\n", cMean / total * 100), stderr)
    fputs("------------------------------------------\n", stderr)
}
