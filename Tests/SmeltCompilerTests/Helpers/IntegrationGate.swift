import Foundation
import Testing
@testable import SmeltCompiler

// Gate for heavy package-building integration tests. The fast push CI
// (tools/test-default.sh) sets SMELT_SKIP_INTEGRATION_TESTS so these tests are
// skipped — they build full .smeltpkg packages whose weights.bin accumulate and
// exhaust the runner's disk. They run in the manual `integration` workflow.

/// True in the fast push CI (env set by tools/test-default.sh); unset in the
/// integration workflow.
let skipIntegrationTests =
    ProcessInfo.processInfo.environment["SMELT_SKIP_INTEGRATION_TESTS"] != nil

enum IntegrationGateError: Error, CustomStringConvertible {
    case packageBuildInFastCI
    var description: String {
        "SmeltCompiler.build ran under SMELT_SKIP_INTEGRATION_TESTS — this test "
            + "builds a package, so mark it @Test(.requiresPackageBuild) (or, for "
            + "XCTest, guard with `try XCTSkipIf(skipIntegrationTests)`)."
    }
}

extension Trait where Self == ConditionTrait {
    /// Marks a Swift Testing test that builds a real `.smeltpkg` via
    /// `SmeltCompiler.build`. Skipped in the fast push CI; runs in the manual
    /// integration workflow.
    static var requiresPackageBuild: Self {
        .disabled(
            if: skipIntegrationTests,
            "package-build integration test; run via the integration workflow"
        )
    }
}

/// The sole entry point for building a `.smeltpkg` in tests: mirrors
/// `SmeltCompiler.build` and forwards to it, but first throws under the fast push
/// CI. A correctly-marked `.requiresPackageBuild` test is skipped before reaching
/// here, so a throw means an *unmarked* heavy test slipped through (the backstop).
/// Every test build must route through this — grep-enforced — so the skip-gate
/// can't be bypassed.
@discardableResult
func managedBuild(
    ir: SmeltModelIR,
    inputName: String,
    outputDir: String,
    weightsDir: String? = nil,
    shaderDir: String,
    traceMode: SmeltTraceMode = .full,
    imatrixPath: String? = nil,
    gptqBlocks: [String: SmeltAffineU4.Packed]? = nil
) throws -> SmeltCompiler.BuildResult {
    guard !skipIntegrationTests else { throw IntegrationGateError.packageBuildInFastCI }
    return try SmeltCompiler.build(
        ir: ir,
        inputName: inputName,
        outputDir: outputDir,
        weightsDir: weightsDir,
        shaderDir: shaderDir,
        traceMode: traceMode,
        imatrixPath: imatrixPath,
        gptqBlocks: gptqBlocks
    )
}

/// Delete the output package dir a `managedBuild` produced (the parent of
/// `packagePath`), plus any input weights-bundle dirs the caller owns. Call from
/// a func-scoped `defer` registered right after the build, so it fires at test
/// scope exit after every use of `result` — not at object deinit, which Swift
/// could trigger early. Never pass a shared/cached dir a test intends to reuse.
func cleanupManagedPackage(_ result: SmeltCompiler.BuildResult, inputDirs: [URL] = []) {
    try? FileManager.default.removeItem(
        at: URL(fileURLWithPath: result.packagePath).deletingLastPathComponent())
    for dir in inputDirs { try? FileManager.default.removeItem(at: dir) }
}
