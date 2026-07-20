import Foundation
import SmeltRuntime

func loadTraceRecords(path: String) throws -> [SmeltTraceRecord] {
    let text = try String(contentsOfFile: path, encoding: .utf8)
    let decoder = JSONDecoder()
    return try text
        .split(whereSeparator: \.isNewline)
        .map { line in
            try decoder.decode(
                SmeltTraceRecord.self,
                from: Data(line.utf8)
            )
        }
}

/// Run replay against `tracePath` and optionally a `packagePath`.
/// On failure, if `failureBundleDir` is provided, write a structured
/// triage bundle there before re-throwing the error.
func replayTrace(
    tracePath: String,
    packagePath: String?,
    construction: CAMTextRuntimeConstruction? = nil,
    failureBundleDir: String? = nil
) throws {
    func writeBundleIfRequested(plan: SmeltReplayPlan?, actualTokens: [Int32]?, error: Error) {
        guard let dir = failureBundleDir else { return }
        do {
            let bundle = try SmeltReplay.writeFailureBundle(
                sourceTracePath: tracePath,
                plan: plan,
                actualTokens: actualTokens,
                error: error,
                directory: dir
            )
            fputs("Wrote failure bundle: \(bundle.directory)\n", stderr)
        } catch {
            fputs("Failed to write failure bundle: \(error)\n", stderr)
        }
    }

    let records: [SmeltTraceRecord]
    do {
        records = try loadTraceRecords(path: tracePath)
    } catch {
        writeBundleIfRequested(plan: nil, actualTokens: nil, error: error)
        throw error
    }

    let plan: SmeltReplayPlan
    do {
        plan = try SmeltReplay.validate(records: records)
    } catch {
        writeBundleIfRequested(plan: nil, actualTokens: nil, error: error)
        throw error
    }

    fputs("Trace ID: \(plan.traceId)\n", stderr)
    fputs("Events: \(plan.totalEvents)\n", stderr)
    fputs("Expected tokens: \(plan.expectedTokens.count)\n", stderr)
    fputs("Sampler: \(plan.sampler)\n", stderr)

    guard let packagePath else {
        fputs("No --package provided; summary only.\n", stderr)
        return
    }

    guard let construction else {
        throw CLIError("replay package execution requires module text construction")
    }
    try construction.requirePackagePath(packagePath)
    do {
        try SmeltReplay.validatePackage(
            plan: plan,
            packagePath: packagePath,
            camSemanticSHA256: construction.camSemanticSHA256,
            exportABISHA256: construction.exportABISHA256
        )
    } catch {
        writeBundleIfRequested(plan: plan, actualTokens: nil, error: error)
        throw error
    }

    let (manifest, _) = try construction.loadManifestConfig()
    let model = try construction.makeModel(contextLimit: nil, manifest: manifest)

    do {
        try SmeltReplay.validateContextLimit(
            plan: plan,
            modelMaxContextTokens: model.maxContextTokens
        )
    } catch {
        writeBundleIfRequested(plan: plan, actualTokens: nil, error: error)
        throw error
    }
    var emitted = 0
    let result = try model.generate(
        tokenIds: plan.contextTokenIds,
        selectionMode: .argmax
    ) { _ in
        emitted += 1
        return emitted < plan.expectedTokens.count
    }

    do {
        try SmeltReplay.compareTokens(
            actual: result.tokens, expected: plan.expectedTokens
        )
    } catch {
        writeBundleIfRequested(plan: plan, actualTokens: result.tokens, error: error)
        throw error
    }

    fputs(
        "Replay matched \(result.tokens.count) tokens. "
            + "prefill \(String(format: "%.1f", result.prefillTime * 1_000))ms, "
            + "generate \(String(format: "%.1f", result.generateTime * 1_000))ms.\n",
        stderr
    )
}
