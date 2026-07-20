import Foundation
import SmeltRuntime
import SmeltServe
import SmeltSchema

let builtInVerifyPrompts: [String] = [
    "The capital of France is?",
    "Explain tail recursion in one short paragraph.",
    "List three metals used in electronics and what each is commonly used for.",
    "You are optimizing a quantized language model on Apple Silicon. Name two likely prompt-prefill bottlenecks and explain each one briefly.",
]

func buildRepeatedVerifyPrompt(
    packagePath: String,
    template: String,
    maxTokenCount: Int,
    tokenizer providedTokenizer: SmeltTokenizer? = nil
) throws -> (prompt: String, tokenIds: [Int32]) {
    guard maxTokenCount > 0 else { return ("", []) }

    let tokenizer = try providedTokenizer ?? SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
    var prompt = builtInVerifyPrompts.joined(separator: "\n\n")
    var tokenIds = try buildInputIds(
        prompt: prompt,
        tokenizer: tokenizer,
        template: template
    )
    while tokenIds.count < maxTokenCount {
        prompt += "\n\n" + builtInVerifyPrompts.joined(separator: "\n\n")
        tokenIds = try buildInputIds(
            prompt: prompt,
            tokenizer: tokenizer,
            template: template
        )
    }

    return (prompt, tokenIds)
}

func loadVerifyPrompts(path: String?) throws -> [String] {
    let candidatePath: String?
    if let path, !path.isEmpty {
        candidatePath = path
    } else if FileManager.default.fileExists(
        atPath: "tools/prompts/qwen_parity.txt"
    ) {
        candidatePath = "tools/prompts/qwen_parity.txt"
    } else {
        candidatePath = nil
    }

    guard let candidatePath else {
        return builtInVerifyPrompts
    }

    let text = try String(contentsOfFile: candidatePath, encoding: .utf8)
    let prompts = text
        .split(whereSeparator: \.isNewline)
        .map(String.init)
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty && !$0.hasPrefix("#") }
    return prompts.isEmpty ? builtInVerifyPrompts : prompts
}

func makeGateFailure(_ message: String) -> NSError {
    NSError(
        domain: "SmeltBenchmarkGate",
        code: 2,
        userInfo: [NSLocalizedDescriptionKey: message]
    )
}

func decodeGateFailures(
    result: DecodeBenchmarkResult,
    minTps: Double?,
    maxP95Ms: Double?
) -> [String] {
    var failures: [String] = []
    if let minTps, result.tokensPerSecond < minTps {
        failures.append(
            "decode tok/s \(String(format: "%.1f", result.tokensPerSecond)) < \(String(format: "%.1f", minTps))"
        )
    }
    if let maxP95Ms, result.p95TotalMs > maxP95Ms {
        failures.append(
            "decode p95 \(String(format: "%.2f", result.p95TotalMs))ms > \(String(format: "%.2f", maxP95Ms))ms"
        )
    }
    return failures
}

func prefillGateFailures(
    result: PrefillBenchmarkResult,
    tokenCount: Int,
    minTps: Double?,
    maxP95Ms: Double?
) -> [String] {
    var failures: [String] = []
    if let minTps, result.tokensPerSecond < minTps {
        failures.append(
            "prefill-\(tokenCount) tok/s \(String(format: "%.1f", result.tokensPerSecond)) < \(String(format: "%.1f", minTps))"
        )
    }
    if let maxP95Ms, result.p95WallMs > maxP95Ms {
        failures.append(
            "prefill-\(tokenCount) p95 \(String(format: "%.1f", result.p95WallMs))ms > \(String(format: "%.1f", maxP95Ms))ms"
        )
    }
    return failures
}

func defaultPrefillVerifyTokenCounts(
    maxPrefillBatchSize: Int
) -> [Int] {
    guard maxPrefillBatchSize > 0 else { return [] }
    let pivots = [64, 128, maxPrefillBatchSize]
    var counts = Set<Int>()
    for pivot in pivots where pivot > 0 {
        for delta in -1...1 {
            let count = pivot + delta
            if count > 0 && count <= maxPrefillBatchSize {
                counts.insert(count)
            }
        }
    }
    return counts.sorted()
}

func shouldUseModelGenerate(
    packagePath: String,
    manifest: SmeltManifest
) -> (Bool, String) {
    let fm = FileManager.default

    if fm.fileExists(atPath: "\(packagePath)/prefill_dispatches.bin") {
        return (true, "metal-prefill")
    }

    guard let prefill = manifest.prefill else {
        return (true, "no-prefill")
    }

    if prefill.engine == "metal" {
        return (false, "missing declared metal prefill table")
    }

    let prefillModelPath = "\(packagePath)/\(prefill.modelPath)"
    let cachePath = "\(packagePath)/cache"
    let hasModel = !prefill.modelPath.isEmpty && fm.fileExists(atPath: prefillModelPath)
    let hasCache = fm.fileExists(atPath: cachePath)

    if hasModel && hasCache {
        return (true, "coreml-prefill")
    }

    return (false, "missing declared prefill artifacts")
}

func printPackageIntegrityReport(
    _ report: SmeltPackageIntegrityReport,
    label: String
) {
    fputs("\(label) package integrity:\n", stderr)
    fputs("  verified files: \(report.verifiedFiles.count)\n", stderr)
    for file in report.verifiedFiles {
        fputs(
            "    \(file.name): \(file.actualSHA256.prefix(12))...\n",
            stderr
        )
    }
    if !report.skippedFiles.isEmpty {
        let skipped = report.skippedFiles.joined(separator: ", ")
        fputs(
            "  skipped files: \(skipped)\n",
            stderr
        )
    }
    if let provenance = report.buildProvenance {
        fputs(
            "  build fingerprint:   \(provenance.buildFingerprint.prefix(12))...\n",
            stderr
        )
        fputs(
            "  weights fingerprint: \(provenance.weightsFingerprint.prefix(12))...\n",
            stderr
        )
    } else {
        fputs("  build provenance: missing\n", stderr)
    }
    fputs("\n", stderr)
}

func printDispatchStructureReport(
    _ report: SmeltDispatchStructureReport,
    label: String,
    kind: String,
    topCount: Int = 8
) {
    fputs("\(label) \(kind) structure:\n", stderr)
    fputs(
        "  records: \(report.totalRecords) total = \(report.dispatchCount) dispatches + \(report.swapCount) swaps\n",
        stderr
    )
    if !report.pipelineUsages.isEmpty {
        fputs("  top pipelines:\n", stderr)
        for usage in report.pipelineUsages.prefix(topCount) {
            fputs(
                "    \(usage.dispatchCount)x \(usage.name)\n",
                stderr
            )
        }
    }
    fputs("\n", stderr)
}
