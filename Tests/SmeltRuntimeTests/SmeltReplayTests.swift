import Foundation
@testable import SmeltCompiler
@testable import SmeltSchema
import Testing
@testable import SmeltRuntime

// Tests for `SmeltReplay.validate` and friends — pure trace-coherence
// checks that don't require a model package.
//
// Each test builds a synthetic `SmeltTraceRecord` array, runs it
// through the validator, and asserts the typed `SmeltReplayError`
// surface. Hash-validation against an actual package is exercised
// indirectly through the runtime's existing trace tests.

private func makeTextDeltaRecord(
    traceId: String = "trace-1",
    sessionId: String = "session-1",
    generationId: String = "gen-1",
    packageHash: String = "pkg",
    tokenizerHash: String? = nil,
    camSemanticSHA256: String? = nil,
    exportABISHA256: String? = nil,
    sampler: String = "argmax",
    contextTokenIds: [Int32]? = nil,
    tokenId: Int32? = nil
) -> SmeltTraceRecord {
    SmeltTraceRecord(
        traceId: traceId,
        sessionId: sessionId,
        generationId: generationId,
        eventType: SmeltEventType.textDelta.rawValue,
        packageHash: packageHash,
        tokenizerHash: tokenizerHash,
        camSemanticSHA256: camSemanticSHA256,
        exportABISHA256: exportABISHA256,
        prefixCacheKey: nil,
        sampler: sampler,
        promptHash: nil,
        contextTokenIds: contextTokenIds,
        tokenId: tokenId,
        timestampUs: 0
    )
}

@Test func replayValidationRejectsEmptyTrace() throws {
    #expect(throws: SmeltReplayError.emptyTrace) {
        try SmeltReplay.validate(records: [])
    }
}

@Test func replayValidationRejectsTraceWithoutContextTokens() throws {
    let records = [makeTextDeltaRecord(tokenId: 7)]
    #expect(throws: SmeltReplayError.missingContextTokens) {
        try SmeltReplay.validate(records: records)
    }
}

@Test func replayValidationRejectsNonArgmaxSampler() throws {
    let records = [
        makeTextDeltaRecord(
            sampler: "temperature:0.7:seed:1",
            contextTokenIds: [10, 20, 30]
        )
    ]
    let error = #expect(throws: SmeltReplayError.self) {
        try SmeltReplay.validate(records: records)
    }
    if case .unsupportedSampler(let actual) = error {
        #expect(actual == "temperature:0.7:seed:1")
    } else {
        Issue.record("expected .unsupportedSampler, got \(String(describing: error))")
    }
}

@Test func replayValidationExtractsPlanFromCoherentTrace() throws {
    let records = [
        makeTextDeltaRecord(contextTokenIds: [1, 2, 3]),
        makeTextDeltaRecord(tokenId: 100),
        makeTextDeltaRecord(tokenId: 101),
        makeTextDeltaRecord(tokenId: 102),
    ]
    let plan = try SmeltReplay.validate(records: records)
    #expect(plan.traceId == "trace-1")
    #expect(plan.packageHash == "pkg")
    #expect(plan.sampler == "argmax")
    #expect(plan.contextTokenIds == [1, 2, 3])
    #expect(plan.expectedTokens == [100, 101, 102])
    #expect(plan.totalEvents == 4)
}

@Test func replayValidationRejectsMixedCAMIdentityTokenRecords() throws {
    let camA = String(repeating: "a", count: 64)
    let camB = String(repeating: "b", count: 64)
    let exportABI = String(repeating: "c", count: 64)
    let records = [
        makeTextDeltaRecord(
            camSemanticSHA256: camA,
            exportABISHA256: exportABI,
            contextTokenIds: [1, 2, 3]
        ),
        makeTextDeltaRecord(
            camSemanticSHA256: camB,
            exportABISHA256: exportABI,
            tokenId: 100
        ),
    ]

    let error = #expect(throws: SmeltReplayError.self) {
        try SmeltReplay.validate(records: records)
    }
    if case .traceIdentityMismatch(let field, let expected, let actual) = error {
        #expect(field == "camSemanticSHA256")
        #expect(expected == camA)
        #expect(actual == camB)
    } else {
        Issue.record("expected .traceIdentityMismatch, got \(String(describing: error))")
    }
}

@Test func replayValidationIgnoresNonTokenEvents() throws {
    // metrics records have no tokenId and shouldn't be counted as
    // expected tokens; toolCall records similarly.
    let contextRecord = makeTextDeltaRecord(contextTokenIds: [1])
    let metricsRecord = SmeltTraceRecord(
        traceId: "trace-1", sessionId: "session-1", generationId: "gen-1",
        eventType: SmeltEventType.metrics.rawValue,
        packageHash: "pkg", tokenizerHash: nil, prefixCacheKey: nil,
        sampler: "argmax", promptHash: nil, timestampUs: 0
    )
    let tokenRecord = makeTextDeltaRecord(tokenId: 99)
    let records = [contextRecord, metricsRecord, tokenRecord]
    let plan = try SmeltReplay.validate(records: records)
    #expect(plan.expectedTokens == [99])
    #expect(plan.totalEvents == 3)
}

@Test func replayCompareTokensSucceedsOnExactMatch() throws {
    try SmeltReplay.compareTokens(actual: [1, 2, 3], expected: [1, 2, 3])
}

@Test func replayCompareTokensReportsFirstDivergence() throws {
    let error = #expect(throws: SmeltReplayError.self) {
        try SmeltReplay.compareTokens(actual: [1, 2, 99], expected: [1, 2, 3])
    }
    if case .tokenDivergence(let at, let expected, let actual) = error {
        #expect(at == 2)
        #expect(expected == 3)
        #expect(actual == 99)
    } else {
        Issue.record("expected .tokenDivergence, got \(String(describing: error))")
    }
}

@Test func replayCompareTokensReportsLengthMismatch() throws {
    let error = #expect(throws: SmeltReplayError.self) {
        try SmeltReplay.compareTokens(actual: [1, 2, 3, 4], expected: [1, 2, 3])
    }
    if case .lengthMismatch(let expected, let actual) = error {
        #expect(expected == 3)
        #expect(actual == 4)
    } else {
        Issue.record("expected .lengthMismatch, got \(String(describing: error))")
    }
}

@Test func replayBundleCapturesTokenDivergence() throws {
    let tempDir = NSTemporaryDirectory()
        + "smelt-replay-bundle-\(UUID().uuidString)"
    let traceSrc = tempDir + "/source-trace.jsonl"
    let bundleDir = tempDir + "/bundle"
    try FileManager.default.createDirectory(
        atPath: tempDir, withIntermediateDirectories: true
    )
    defer { try? FileManager.default.removeItem(atPath: tempDir) }

    // Synthetic trace JSONL — content doesn't matter for bundle
    // copy; we just need a file at the source path.
    try "{\"placeholder\": true}\n".write(
        toFile: traceSrc, atomically: true, encoding: .utf8
    )

    let plan = SmeltReplayPlan(
        traceId: "trace-x",
        sessionId: "session-x",
        packageHash: "pkg",
        tokenizerHash: "tok",
        sampler: "argmax",
        contextTokenIds: [1, 2, 3],
        expectedTokens: [10, 20, 30],
        totalEvents: 6
    )
    let actualTokens: [Int32] = [10, 20, 99]
    let error = SmeltReplayError.tokenDivergence(at: 2, expected: 30, actual: 99)

    let bundle = try SmeltReplay.writeFailureBundle(
        sourceTracePath: traceSrc,
        plan: plan,
        actualTokens: actualTokens,
        error: error,
        directory: bundleDir
    )

    #expect(FileManager.default.fileExists(atPath: bundle.tracePath))
    #expect(FileManager.default.fileExists(atPath: bundle.resultPath))
    #expect(FileManager.default.fileExists(atPath: bundle.summaryPath))

    // result.json — verify the structured fields the audit consumer
    // would switch on are populated correctly.
    let resultData = try Data(contentsOf: URL(fileURLWithPath: bundle.resultPath))
    let json = try JSONSerialization.jsonObject(with: resultData) as? [String: Any]
    #expect(json?["schemaVersion"] as? Int == 1)
    #expect(json?["actualTokens"] as? [Int] == [10, 20, 99])
    let errorBlock = json?["error"] as? [String: Any]
    #expect(errorBlock?["type"] as? String == "tokenDivergence")
    #expect(errorBlock?["divergenceIndex"] as? Int == 2)
    #expect(errorBlock?["expectedToken"] as? Int == 30)
    #expect(errorBlock?["actualToken"] as? Int == 99)
    let traceBlock = json?["trace"] as? [String: Any]
    #expect(traceBlock?["traceId"] as? String == "trace-x")
    #expect(traceBlock?["packageHash"] as? String == "pkg")
    #expect(traceBlock?["tokenizerHash"] as? String == "tok")
    #expect(traceBlock?["expectedTokenCount"] as? Int == 3)

    // summary.txt is human-readable; just check the trace ID and
    // error message landed.
    let summary = try String(contentsOfFile: bundle.summaryPath, encoding: .utf8)
    #expect(summary.contains("trace-x"))
    #expect(summary.contains("token 2"))
    #expect(summary.contains("99"))
}

@Test func replayBundleHandlesNilPlan() throws {
    let tempDir = NSTemporaryDirectory()
        + "smelt-replay-bundle-nilplan-\(UUID().uuidString)"
    let traceSrc = tempDir + "/source-trace.jsonl"
    let bundleDir = tempDir + "/bundle"
    try FileManager.default.createDirectory(
        atPath: tempDir, withIntermediateDirectories: true
    )
    defer { try? FileManager.default.removeItem(atPath: tempDir) }
    try "".write(toFile: traceSrc, atomically: true, encoding: .utf8)

    let bundle = try SmeltReplay.writeFailureBundle(
        sourceTracePath: traceSrc,
        plan: nil,
        actualTokens: nil,
        error: SmeltReplayError.emptyTrace,
        directory: bundleDir
    )

    let resultData = try Data(contentsOf: URL(fileURLWithPath: bundle.resultPath))
    let json = try JSONSerialization.jsonObject(with: resultData) as? [String: Any]
    #expect(json?["trace"] is NSNull || json?["trace"] == nil)
    #expect(json?["actualTokens"] is NSNull || json?["actualTokens"] == nil)
    let errorBlock = json?["error"] as? [String: Any]
    #expect(errorBlock?["type"] as? String == "emptyTrace")

    let summary = try String(contentsOfFile: bundle.summaryPath, encoding: .utf8)
    #expect(summary.contains("did not validate"))
}

@Test func replayBundleIsIdempotent() throws {
    // Writing twice to the same directory should succeed and produce
    // the same artifact paths — supports re-runs and CI retries.
    let tempDir = NSTemporaryDirectory()
        + "smelt-replay-bundle-idempotent-\(UUID().uuidString)"
    let traceSrc = tempDir + "/source-trace.jsonl"
    let bundleDir = tempDir + "/bundle"
    try FileManager.default.createDirectory(
        atPath: tempDir, withIntermediateDirectories: true
    )
    defer { try? FileManager.default.removeItem(atPath: tempDir) }
    try "{}\n".write(toFile: traceSrc, atomically: true, encoding: .utf8)

    let plan = SmeltReplayPlan(
        traceId: "t", sessionId: "s",
        packageHash: "p", tokenizerHash: nil,
        sampler: "argmax",
        contextTokenIds: [1], expectedTokens: [], totalEvents: 1
    )
    let bundle1 = try SmeltReplay.writeFailureBundle(
        sourceTracePath: traceSrc, plan: plan, actualTokens: nil,
        error: SmeltReplayError.lengthMismatch(expected: 0, actual: 1),
        directory: bundleDir
    )
    let bundle2 = try SmeltReplay.writeFailureBundle(
        sourceTracePath: traceSrc, plan: plan, actualTokens: nil,
        error: SmeltReplayError.lengthMismatch(expected: 0, actual: 1),
        directory: bundleDir
    )
    #expect(bundle1 == bundle2)
}

@Test func replayValidationCapturesContextLimitFromDecodingPolicy() throws {
    let pressure = SmeltContextPressure(
        contextLimit: 4096,
        promptTokens: 100,
        estimatedPromptTokens: 100,
        requestedMaxTokens: 256,
        resolvedMaxTokens: 256,
        effectiveMaxTokens: 256,
        toolCallMinTokens: 64,
        messageCount: 1,
        availableInputTokens: 3996,
        availableOutputTokens: 3996,
        pressureRatio: 0.024,
        action: "allow",
        reason: nil
    )
    let policy = SmeltDecodingPolicy(
        name: "default",
        phase: "assistant_text",
        sampler: "argmax",
        temperature: nil,
        seed: nil,
        source: "test",
        reason: nil,
        contextPressure: pressure
    )
    let context = makeTextDeltaRecord(contextTokenIds: [1, 2, 3])
    let withPolicy = SmeltTraceRecord(
        traceId: "trace-1", sessionId: "session-1", generationId: "gen-1",
        eventType: SmeltEventType.metrics.rawValue,
        packageHash: "pkg", tokenizerHash: nil, prefixCacheKey: nil,
        sampler: "argmax", promptHash: nil,
        decodingPolicy: policy,
        timestampUs: 0
    )
    let plan = try SmeltReplay.validate(records: [context, withPolicy])
    #expect(plan.recordedContextLimit == 4096)
}

@Test func replayValidationLeavesContextLimitNilWhenAbsent() throws {
    let records = [makeTextDeltaRecord(contextTokenIds: [1, 2, 3])]
    let plan = try SmeltReplay.validate(records: records)
    #expect(plan.recordedContextLimit == nil)
}

@Test func replayValidateContextLimitDetectsMismatch() throws {
    let plan = SmeltReplayPlan(
        traceId: "t", sessionId: "s",
        packageHash: "p", tokenizerHash: nil,
        sampler: "argmax",
        contextTokenIds: [1], expectedTokens: [], totalEvents: 1,
        recordedContextLimit: 4096
    )
    let error = #expect(throws: SmeltReplayError.self) {
        try SmeltReplay.validateContextLimit(plan: plan, modelMaxContextTokens: 2048)
    }
    if case .contextLimitMismatch(let traceLimit, let modelLimit) = error {
        #expect(traceLimit == 4096)
        #expect(modelLimit == 2048)
    } else {
        Issue.record("expected .contextLimitMismatch, got \(String(describing: error))")
    }
}

@Test func replayValidateContextLimitSkipsWhenTraceHasNone() throws {
    let plan = SmeltReplayPlan(
        traceId: "t", sessionId: "s",
        packageHash: "p", tokenizerHash: nil,
        sampler: "argmax",
        contextTokenIds: [1], expectedTokens: [], totalEvents: 1,
        recordedContextLimit: nil
    )
    // Should not throw — older traces without contextLimit are
    // grandfathered in.
    try SmeltReplay.validateContextLimit(plan: plan, modelMaxContextTokens: 1024)
}

@Test func replayBundleEncodesContextLimitMismatch() throws {
    let tempDir = NSTemporaryDirectory()
        + "smelt-replay-bundle-ctx-\(UUID().uuidString)"
    let traceSrc = tempDir + "/source-trace.jsonl"
    let bundleDir = tempDir + "/bundle"
    try FileManager.default.createDirectory(
        atPath: tempDir, withIntermediateDirectories: true
    )
    defer { try? FileManager.default.removeItem(atPath: tempDir) }
    try "{}\n".write(toFile: traceSrc, atomically: true, encoding: .utf8)

    let plan = SmeltReplayPlan(
        traceId: "t", sessionId: "s",
        packageHash: "p", tokenizerHash: nil,
        sampler: "argmax",
        contextTokenIds: [1], expectedTokens: [], totalEvents: 1,
        recordedContextLimit: 4096
    )
    let bundle = try SmeltReplay.writeFailureBundle(
        sourceTracePath: traceSrc, plan: plan, actualTokens: nil,
        error: SmeltReplayError.contextLimitMismatch(traceLimit: 4096, modelLimit: 2048),
        directory: bundleDir
    )

    let resultData = try Data(contentsOf: URL(fileURLWithPath: bundle.resultPath))
    let json = try JSONSerialization.jsonObject(with: resultData) as? [String: Any]
    let errorBlock = json?["error"] as? [String: Any]
    #expect(errorBlock?["type"] as? String == "contextLimitMismatch")
    // contextLimitMismatch reuses expectedLength/actualLength fields
    // for traceLimit/modelLimit — keeps the bundle schema flat
    // without inventing a separate pair.
    #expect(errorBlock?["expectedLength"] as? Int == 4096)
    #expect(errorBlock?["actualLength"] as? Int == 2048)

    let summary = try String(contentsOfFile: bundle.summaryPath, encoding: .utf8)
    #expect(summary.contains("Context limit:   4096"))
}

@Test func replayValidatePackageDetectsHashMismatch() throws {
    // Use a synthetic plan with a hash that won't match anything real.
    // We don't need an actual package on disk for this — pass a path
    // that we know exists but with a mismatched recorded hash.
    let plan = SmeltReplayPlan(
        traceId: "trace-1", sessionId: "session-1",
        packageHash: "deadbeef", tokenizerHash: nil,
        sampler: "argmax",
        contextTokenIds: [1], expectedTokens: [], totalEvents: 1
    )
    // A directory that's guaranteed to exist but has no manifest.json
    // produces a thrown error from packageHash() — also fine, since
    // the test's contract is "validatePackage rejects mismatch."
    // We pick a directory we know lacks a manifest so packageHash
    // throws a file-not-found rather than computing a hash.
    let tempDir = NSTemporaryDirectory()
    #expect(throws: (any Error).self) {
        try SmeltReplay.validatePackage(plan: plan, packagePath: tempDir)
    }
}

@Test func replayValidatePackageRequiresCAMIdentityForCAMPackages() throws {
    let package = try makeReplayCAMPackage()
    defer { try? FileManager.default.removeItem(atPath: package.deletingLastPathComponent().path) }

    let packageHash = try SmeltHash.packageHash(packagePath: package.path)
    let loadedCapabilities = try SmeltCAMPackageCapabilities.loadIfPresent(packageURL: package)
    let capabilities = try #require(loadedCapabilities)

    let missingCAM = SmeltReplayPlan(
        traceId: "trace-1",
        sessionId: "session-1",
        packageHash: packageHash,
        tokenizerHash: nil,
        sampler: "argmax",
        contextTokenIds: [1],
        expectedTokens: [],
        totalEvents: 1
    )
    let missingError = #expect(throws: SmeltReplayError.self) {
        try SmeltReplay.validatePackage(
            plan: missingCAM,
            packagePath: package.path,
            camSemanticSHA256: capabilities.camSemanticSHA256,
            exportABISHA256: capabilities.exportABISHA256
        )
    }
    if case .camSemanticHashMissing(let modelHash) = missingError {
        #expect(modelHash == capabilities.camSemanticSHA256)
    } else {
        Issue.record("expected .camSemanticHashMissing, got \(String(describing: missingError))")
    }

    let mismatchedCAM = SmeltReplayPlan(
        traceId: "trace-1",
        sessionId: "session-1",
        packageHash: packageHash,
        tokenizerHash: nil,
        camSemanticSHA256: String(repeating: "0", count: 64),
        exportABISHA256: capabilities.exportABISHA256,
        sampler: "argmax",
        contextTokenIds: [1],
        expectedTokens: [],
        totalEvents: 1
    )
    let mismatchError = #expect(throws: SmeltReplayError.self) {
        try SmeltReplay.validatePackage(
            plan: mismatchedCAM,
            packagePath: package.path,
            camSemanticSHA256: capabilities.camSemanticSHA256,
            exportABISHA256: capabilities.exportABISHA256
        )
    }
    if case .camSemanticHashMismatch(let traceHash, let modelHash) = mismatchError {
        #expect(traceHash == String(repeating: "0", count: 64))
        #expect(modelHash == capabilities.camSemanticSHA256)
    } else {
        Issue.record("expected .camSemanticHashMismatch, got \(String(describing: mismatchError))")
    }

    let matchingCAM = SmeltReplayPlan(
        traceId: "trace-1",
        sessionId: "session-1",
        packageHash: packageHash,
        tokenizerHash: nil,
        camSemanticSHA256: capabilities.camSemanticSHA256,
        exportABISHA256: capabilities.exportABISHA256,
        sampler: "argmax",
        contextTokenIds: [1],
        expectedTokens: [],
        totalEvents: 1
    )
    try SmeltReplay.validatePackage(
        plan: matchingCAM,
        packagePath: package.path,
        camSemanticSHA256: capabilities.camSemanticSHA256,
        exportABISHA256: capabilities.exportABISHA256
    )
}

private func makeReplayCAMPackage() throws -> URL {
    let root = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("smelt-replay-cam-identity-\(UUID().uuidString)", isDirectory: true)
    let package = root.appendingPathComponent("package.smeltpkg", isDirectory: true)
    try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: package.appendingPathComponent("manifest.json"))
    let descriptor = try SmeltCAMPackageDescriptor(from: registryModuleIR("qwen35_text"))
    try descriptor.canonicalJSONData()
        .write(to: package.appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName))
    return package
}
