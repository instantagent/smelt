// SmeltReplay — Trace replay validation as a runtime contract.
//
// Replay is a debugging product surface: take a recorded trace, validate
// that it can run against a model, and either produce the same tokens
// (success) or diagnose where it diverged (failure). The runtime
// exposes the validation pieces so callers (CLI, tests, future GUIs)
// can build on a typed contract rather than re-parsing free-form
// error strings.

import Foundation

/// Typed errors a replay can fail with. Mirrors the call sites in the
/// CLI's `replayTrace`; each case holds the data a caller needs
/// to render a useful diagnostic without re-parsing strings.
public enum SmeltReplayError: Error, LocalizedError, Equatable {
    case emptyTrace
    case missingContextTokens
    case unsupportedSampler(actual: String)
    case traceIdentityMismatch(field: String, expected: String?, actual: String?)
    case packageHashMismatch(traceHash: String, modelHash: String)
    case tokenizerHashMismatch(traceHash: String, modelHash: String?)
    case camSemanticHashMissing(modelHash: String)
    case camSemanticHashMismatch(traceHash: String, modelHash: String)
    case exportABIHashMissing(modelHash: String)
    case exportABIHashMismatch(traceHash: String, modelHash: String)
    case contextLimitMismatch(traceLimit: Int, modelLimit: Int)
    case tokenDivergence(at: Int, expected: Int32, actual: Int32)
    case lengthMismatch(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .emptyTrace:
            return "Trace is empty"
        case .missingContextTokens:
            return "Trace has no contextTokenIds — re-run with the current trace schema"
        case .unsupportedSampler(let actual):
            return "Replay currently supports argmax traces only, got \(actual)"
        case .traceIdentityMismatch(let field, let expected, let actual):
            return "Trace identity mismatch for \(field): expected \(expected ?? "<none>"), got \(actual ?? "<none>")"
        case .packageHashMismatch(let trace, let model):
            return "Package hash mismatch: trace recorded against \(trace), loaded model has \(model). Replay would diverge."
        case .tokenizerHashMismatch(let trace, let model):
            return "Tokenizer hash mismatch: trace recorded against \(trace), loaded model has \(model ?? "<no tokenizer>")"
        case .camSemanticHashMissing(let model):
            return "Trace does not record CAM semantic hash; loaded package has \(model). Replay would depend on unrecorded CAM semantics."
        case .camSemanticHashMismatch(let trace, let model):
            return "CAM semantic hash mismatch: trace recorded against \(trace), loaded package has \(model). Replay would diverge."
        case .exportABIHashMissing(let model):
            return "Trace does not record CAM export ABI hash; loaded package has \(model). Replay would depend on unrecorded CAM ABI."
        case .exportABIHashMismatch(let trace, let model):
            return "CAM export ABI hash mismatch: trace recorded against \(trace), loaded package has \(model). Replay would diverge."
        case .contextLimitMismatch(let trace, let model):
            return "Context limit mismatch: trace recorded with contextLimit=\(trace), loaded model has maxContextTokens=\(model). Decoding-policy decisions will diverge."
        case .tokenDivergence(let i, let expected, let actual):
            return "Replay diverged at token \(i): expected \(expected), got \(actual)"
        case .lengthMismatch(let expected, let actual):
            return "Replay length mismatch: expected \(expected) tokens, got \(actual)"
        }
    }
}

/// Pre-execution snapshot of what a trace says it needs and what it
/// expects to produce. Returned by `SmeltReplay.validate` — has been
/// proven internally consistent (non-empty, has context tokens,
/// sampler we can replay) but not yet validated against any model.
public struct SmeltReplayPlan: Equatable {
    public let traceId: String
    public let sessionId: String
    public let packageHash: String
    public let tokenizerHash: String?
    public let camSemanticSHA256: String?
    public let exportABISHA256: String?
    public let sampler: String
    public let contextTokenIds: [Int32]
    public let expectedTokens: [Int32]
    public let totalEvents: Int
    /// Context limit the trace was recorded against, if any
    /// `decodingPolicy.contextPressure` event captured one. Used by
    /// `validateContextLimit` — older traces (no decodingPolicy
    /// records) leave this nil and skip the check.
    public let recordedContextLimit: Int?

    public init(
        traceId: String,
        sessionId: String,
        packageHash: String,
        tokenizerHash: String?,
        camSemanticSHA256: String? = nil,
        exportABISHA256: String? = nil,
        sampler: String,
        contextTokenIds: [Int32],
        expectedTokens: [Int32],
        totalEvents: Int,
        recordedContextLimit: Int? = nil
    ) {
        self.traceId = traceId
        self.sessionId = sessionId
        self.packageHash = packageHash
        self.tokenizerHash = tokenizerHash
        self.camSemanticSHA256 = camSemanticSHA256
        self.exportABISHA256 = exportABISHA256
        self.sampler = sampler
        self.contextTokenIds = contextTokenIds
        self.expectedTokens = expectedTokens
        self.totalEvents = totalEvents
        self.recordedContextLimit = recordedContextLimit
    }
}

public enum SmeltReplay {
    /// Validate a trace's internal coherence and return an executable
    /// plan. Throws `SmeltReplayError.emptyTrace`,
    /// `.missingContextTokens`, or `.unsupportedSampler` when the
    /// trace can't be replayed regardless of which model is loaded.
    /// Pure — no model access, no file IO.
    public static func validate(
        records: [SmeltTraceRecord]
    ) throws -> SmeltReplayPlan {
        guard !records.isEmpty else {
            throw SmeltReplayError.emptyTrace
        }
        let start = records.first { $0.contextTokenIds != nil } ?? records[0]
        guard let contextTokenIds = start.contextTokenIds else {
            throw SmeltReplayError.missingContextTokens
        }
        guard start.sampler == "argmax" else {
            throw SmeltReplayError.unsupportedSampler(actual: start.sampler)
        }
        try validateRecordIdentity(records, against: start)
        let expected = records
            .filter {
                $0.eventType == SmeltEventType.textDelta.rawValue
                    && $0.tokenId != nil
            }
            .compactMap(\.tokenId)
        // Pull the first contextLimit captured anywhere in the trace.
        // We only need one — the limit doesn't change within a
        // generation, and old traces without decodingPolicy records
        // will leave this nil (skips validateContextLimit).
        let recordedContextLimit = records
            .lazy
            .compactMap { $0.decodingPolicy?.contextPressure?.contextLimit }
            .first
        return SmeltReplayPlan(
            traceId: start.traceId,
            sessionId: start.sessionId,
            packageHash: start.packageHash,
            tokenizerHash: start.tokenizerHash,
            camSemanticSHA256: start.camSemanticSHA256,
            exportABISHA256: start.exportABISHA256,
            sampler: start.sampler,
            contextTokenIds: contextTokenIds,
            expectedTokens: expected,
            totalEvents: records.count,
            recordedContextLimit: recordedContextLimit
        )
    }

    private static func validateRecordIdentity(
        _ records: [SmeltTraceRecord],
        against start: SmeltTraceRecord
    ) throws {
        for record in records {
            try requireRecordIdentity("traceId", record.traceId, start.traceId)
            try requireRecordIdentity("sessionId", record.sessionId, start.sessionId)
            try requireRecordIdentity("generationId", record.generationId, start.generationId)
            try requireRecordIdentity("packageHash", record.packageHash, start.packageHash)
            try requireRecordIdentity("tokenizerHash", record.tokenizerHash, start.tokenizerHash)
            try requireRecordIdentity(
                "camSemanticSHA256",
                record.camSemanticSHA256,
                start.camSemanticSHA256
            )
            try requireRecordIdentity(
                "exportABISHA256",
                record.exportABISHA256,
                start.exportABISHA256
            )
            try requireRecordIdentity("sampler", record.sampler, start.sampler)
        }
    }

    private static func requireRecordIdentity(
        _ field: String,
        _ actual: String?,
        _ expected: String?
    ) throws {
        guard actual == expected else {
            throw SmeltReplayError.traceIdentityMismatch(
                field: field,
                expected: expected,
                actual: actual
            )
        }
    }

    /// Compare a plan's recorded package hash against the model
    /// package at `packagePath`. Throws `.packageHashMismatch` or
    /// `.tokenizerHashMismatch` if either differs. A trace without a
    /// recorded tokenizerHash is allowed to load against any model
    /// (older traces predate the tokenizer-hash field).
    public static func validatePackage(
        plan: SmeltReplayPlan,
        packagePath: String,
        camSemanticSHA256: String? = nil,
        exportABISHA256: String? = nil
    ) throws {
        let actualPackageHash = try SmeltHash.packageHash(packagePath: packagePath)
        guard plan.packageHash == actualPackageHash else {
            throw SmeltReplayError.packageHashMismatch(
                traceHash: plan.packageHash,
                modelHash: actualPackageHash
            )
        }
        if let traceTokenizerHash = plan.tokenizerHash {
            let tokenizerPath = "\(packagePath)/tokenizer.json"
            let actualTokenizerHash: String?
            if FileManager.default.fileExists(atPath: tokenizerPath) {
                actualTokenizerHash = try SmeltHash.fileHash(path: tokenizerPath)
            } else {
                actualTokenizerHash = nil
            }
            guard traceTokenizerHash == actualTokenizerHash else {
                throw SmeltReplayError.tokenizerHashMismatch(
                    traceHash: traceTokenizerHash,
                    modelHash: actualTokenizerHash
                )
            }
        }
        if let camSemanticSHA256 {
            guard let traceCAMHash = plan.camSemanticSHA256 else {
                throw SmeltReplayError.camSemanticHashMissing(
                    modelHash: camSemanticSHA256
                )
            }
            guard traceCAMHash == camSemanticSHA256 else {
                throw SmeltReplayError.camSemanticHashMismatch(
                    traceHash: traceCAMHash,
                    modelHash: camSemanticSHA256
                )
            }
        }
        if let exportABISHA256 {
            guard let traceExportABIHash = plan.exportABISHA256 else {
                throw SmeltReplayError.exportABIHashMissing(
                    modelHash: exportABISHA256
                )
            }
            guard traceExportABIHash == exportABISHA256 else {
                throw SmeltReplayError.exportABIHashMismatch(
                    traceHash: traceExportABIHash,
                    modelHash: exportABISHA256
                )
            }
        }
    }

    /// Compare the trace's recorded context limit against a model's
    /// `maxContextTokens`. Decoding-policy decisions (e.g. the
    /// resolved max output tokens, the context-pressure band)
    /// depend on the limit, so a mismatch means the replay can't
    /// reproduce those decisions even when the package hash matches.
    /// Skipped silently when the trace didn't record a context
    /// limit (`recordedContextLimit == nil`) — older traces predate
    /// the decoding-policy field.
    public static func validateContextLimit(
        plan: SmeltReplayPlan,
        modelMaxContextTokens: Int
    ) throws {
        guard let traceLimit = plan.recordedContextLimit else { return }
        guard traceLimit == modelMaxContextTokens else {
            throw SmeltReplayError.contextLimitMismatch(
                traceLimit: traceLimit,
                modelLimit: modelMaxContextTokens
            )
        }
    }

    /// Diff the actual emitted tokens against the plan's expected
    /// tokens. Throws on first divergence or length mismatch. Used
    /// after `model.generate` returns to verify the replay reproduced
    /// the trace.
    public static func compareTokens(
        actual: [Int32],
        expected: [Int32]
    ) throws {
        let compareCount = min(actual.count, expected.count)
        for index in 0..<compareCount where actual[index] != expected[index] {
            throw SmeltReplayError.tokenDivergence(
                at: index,
                expected: expected[index],
                actual: actual[index]
            )
        }
        guard actual.count == expected.count else {
            throw SmeltReplayError.lengthMismatch(
                expected: expected.count, actual: actual.count
            )
        }
    }

    /// Write a structured failure bundle for offline triage. Bundle
    /// contains the original trace (`trace.jsonl`), a machine-readable
    /// `result.json` capturing what was attempted and what failed,
    /// and a human-readable `summary.txt`. Pre-model failures (empty
    /// trace, hash mismatch) write what they have — `plan` and
    /// `actualTokens` are both optional. Idempotent: overwrites
    /// existing files in the directory.
    public static func writeFailureBundle(
        sourceTracePath: String,
        plan: SmeltReplayPlan?,
        actualTokens: [Int32]?,
        error: Error,
        directory: String
    ) throws -> SmeltReplayBundle {
        let fm = FileManager.default
        try fm.createDirectory(
            atPath: directory, withIntermediateDirectories: true
        )

        let traceDest = "\(directory)/trace.jsonl"
        if fm.fileExists(atPath: traceDest) {
            try fm.removeItem(atPath: traceDest)
        }
        try fm.copyItem(atPath: sourceTracePath, toPath: traceDest)

        let result = SmeltReplayBundleResult(
            schemaVersion: 1,
            trace: plan.map(SmeltReplayBundleTrace.init(plan:)),
            actualTokens: actualTokens,
            error: SmeltReplayBundleError(error: error)
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let resultData = try encoder.encode(result)
        let resultPath = "\(directory)/result.json"
        try resultData.write(to: URL(fileURLWithPath: resultPath))

        let summaryPath = "\(directory)/summary.txt"
        let summary = renderBundleSummary(
            plan: plan, actualTokens: actualTokens, error: error
        )
        try summary.write(toFile: summaryPath, atomically: true, encoding: .utf8)

        return SmeltReplayBundle(
            directory: directory,
            tracePath: traceDest,
            resultPath: resultPath,
            summaryPath: summaryPath
        )
    }
}

/// Paths a `writeFailureBundle` call produced. Returned so the CLI
/// can print the directory or test code can inspect each artifact.
public struct SmeltReplayBundle: Equatable {
    public let directory: String
    public let tracePath: String
    public let resultPath: String
    public let summaryPath: String
}

/// `result.json` schema. Stable across patch versions; the
/// `schemaVersion` field gates future evolutions. Consumers
/// (CI dashboards, support tooling, re-run harnesses) decode this
/// and switch on `error.type`.
struct SmeltReplayBundleResult: Codable {
    let schemaVersion: Int
    let trace: SmeltReplayBundleTrace?
    let actualTokens: [Int32]?
    let error: SmeltReplayBundleError
}

struct SmeltReplayBundleTrace: Codable {
    let traceId: String
    let sessionId: String
    let packageHash: String
    let tokenizerHash: String?
    let camSemanticSHA256: String?
    let exportABISHA256: String?
    let sampler: String
    let totalEvents: Int
    let expectedTokenCount: Int

    init(plan: SmeltReplayPlan) {
        self.traceId = plan.traceId
        self.sessionId = plan.sessionId
        self.packageHash = plan.packageHash
        self.tokenizerHash = plan.tokenizerHash
        self.camSemanticSHA256 = plan.camSemanticSHA256
        self.exportABISHA256 = plan.exportABISHA256
        self.sampler = plan.sampler
        self.totalEvents = plan.totalEvents
        self.expectedTokenCount = plan.expectedTokens.count
    }
}

/// Stable error-type strings for bundle consumers. Keep these in
/// sync with `SmeltReplayError`'s cases — the strings are part of
/// the bundle's product surface.
struct SmeltReplayBundleError: Codable {
    let type: String
    let message: String
    let divergenceIndex: Int?
    let expectedToken: Int32?
    let actualToken: Int32?
    let expectedLength: Int?
    let actualLength: Int?
    let traceHash: String?
    let modelHash: String?

    init(error: Error) {
        self.message = (error as? LocalizedError)?.errorDescription
            ?? String(describing: error)
        if let replayError = error as? SmeltReplayError {
            switch replayError {
            case .emptyTrace:
                self.type = "emptyTrace"
                self.divergenceIndex = nil
                self.expectedToken = nil
                self.actualToken = nil
                self.expectedLength = nil
                self.actualLength = nil
                self.traceHash = nil
                self.modelHash = nil
            case .missingContextTokens:
                self.type = "missingContextTokens"
                self.divergenceIndex = nil
                self.expectedToken = nil
                self.actualToken = nil
                self.expectedLength = nil
                self.actualLength = nil
                self.traceHash = nil
                self.modelHash = nil
            case .unsupportedSampler:
                self.type = "unsupportedSampler"
                self.divergenceIndex = nil
                self.expectedToken = nil
                self.actualToken = nil
                self.expectedLength = nil
                self.actualLength = nil
                self.traceHash = nil
                self.modelHash = nil
            case .traceIdentityMismatch(_, let expected, let actual):
                self.type = "traceIdentityMismatch"
                self.divergenceIndex = nil
                self.expectedToken = nil
                self.actualToken = nil
                self.expectedLength = nil
                self.actualLength = nil
                self.traceHash = actual
                self.modelHash = expected
            case .packageHashMismatch(let trace, let model):
                self.type = "packageHashMismatch"
                self.divergenceIndex = nil
                self.expectedToken = nil
                self.actualToken = nil
                self.expectedLength = nil
                self.actualLength = nil
                self.traceHash = trace
                self.modelHash = model
            case .tokenizerHashMismatch(let trace, let model):
                self.type = "tokenizerHashMismatch"
                self.divergenceIndex = nil
                self.expectedToken = nil
                self.actualToken = nil
                self.expectedLength = nil
                self.actualLength = nil
                self.traceHash = trace
                self.modelHash = model
            case .camSemanticHashMissing(let model):
                self.type = "camSemanticHashMissing"
                self.divergenceIndex = nil
                self.expectedToken = nil
                self.actualToken = nil
                self.expectedLength = nil
                self.actualLength = nil
                self.traceHash = nil
                self.modelHash = model
            case .camSemanticHashMismatch(let trace, let model):
                self.type = "camSemanticHashMismatch"
                self.divergenceIndex = nil
                self.expectedToken = nil
                self.actualToken = nil
                self.expectedLength = nil
                self.actualLength = nil
                self.traceHash = trace
                self.modelHash = model
            case .exportABIHashMissing(let model):
                self.type = "exportABIHashMissing"
                self.divergenceIndex = nil
                self.expectedToken = nil
                self.actualToken = nil
                self.expectedLength = nil
                self.actualLength = nil
                self.traceHash = nil
                self.modelHash = model
            case .exportABIHashMismatch(let trace, let model):
                self.type = "exportABIHashMismatch"
                self.divergenceIndex = nil
                self.expectedToken = nil
                self.actualToken = nil
                self.expectedLength = nil
                self.actualLength = nil
                self.traceHash = trace
                self.modelHash = model
            case .contextLimitMismatch(let trace, let model):
                self.type = "contextLimitMismatch"
                self.divergenceIndex = nil
                self.expectedToken = nil
                self.actualToken = nil
                self.expectedLength = trace
                self.actualLength = model
                self.traceHash = nil
                self.modelHash = nil
            case .tokenDivergence(let at, let expected, let actual):
                self.type = "tokenDivergence"
                self.divergenceIndex = at
                self.expectedToken = expected
                self.actualToken = actual
                self.expectedLength = nil
                self.actualLength = nil
                self.traceHash = nil
                self.modelHash = nil
            case .lengthMismatch(let expected, let actual):
                self.type = "lengthMismatch"
                self.divergenceIndex = nil
                self.expectedToken = nil
                self.actualToken = nil
                self.expectedLength = expected
                self.actualLength = actual
                self.traceHash = nil
                self.modelHash = nil
            }
        } else {
            self.type = "unknown"
            self.divergenceIndex = nil
            self.expectedToken = nil
            self.actualToken = nil
            self.expectedLength = nil
            self.actualLength = nil
            self.traceHash = nil
            self.modelHash = nil
        }
    }
}

private func renderBundleSummary(
    plan: SmeltReplayPlan?,
    actualTokens: [Int32]?,
    error: Error
) -> String {
    var lines: [String] = []
    lines.append("Smelt replay failure bundle")
    lines.append("")
    if let plan {
        lines.append("Trace ID:        \(plan.traceId)")
        lines.append("Session ID:      \(plan.sessionId)")
        lines.append("Package hash:    \(plan.packageHash)")
        lines.append("Tokenizer hash:  \(plan.tokenizerHash ?? "(none)")")
        lines.append("CAM hash:        \(plan.camSemanticSHA256 ?? "(none)")")
        lines.append("Export ABI hash: \(plan.exportABISHA256 ?? "(none)")")
        lines.append("Sampler:         \(plan.sampler)")
        lines.append("Total events:    \(plan.totalEvents)")
        lines.append("Expected tokens: \(plan.expectedTokens.count)")
        if let limit = plan.recordedContextLimit {
            lines.append("Context limit:   \(limit)")
        }
    } else {
        lines.append("Trace did not validate; no plan available.")
    }
    if let actualTokens {
        lines.append("Actual tokens:   \(actualTokens.count)")
    }
    lines.append("")
    lines.append("Error: \((error as? LocalizedError)?.errorDescription ?? String(describing: error))")
    lines.append("")
    return lines.joined(separator: "\n")
}
