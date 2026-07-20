// SmeltVerifier — Verification infrastructure for Smelt-generated code.
//
// Provides compile-time verification that the generated code matches
// expected dispatch counts, uses correct slot indices, and produces
// structurally valid Swift. Runtime verification (token-for-token match)
// requires real weights and GPU — handled by the E2E test script.

import Foundation

/// Compile-time verification results for a generated model.
public struct SmeltVerificationReport: Sendable {
    /// Total dispatch count in the generated function.
    public let totalDispatches: Int
    /// Number of pipeline state changes.
    public let pipelineStateChanges: Int
    /// Whether the generated code contains any string lookups (should be zero).
    public let hasStringLookups: Bool
    /// Whether all pipeline references are integer indices.
    public let allIntegerPipelines: Bool
    /// Whether the function opens and closes correctly.
    public let validFunctionStructure: Bool
    /// Whether cur/alt double-buffer swap count matches layer count.
    public let correctSwapCount: Int
    /// Expected swap count (2 per layer: after attention/delta + after FFN).
    public let expectedSwapCount: Int

    /// Whether dispatch count equals pipeline state changes (1:1 invariant).
    public var dispatchPipelineParity: Bool {
        totalDispatches == pipelineStateChanges
    }

    /// Overall pass/fail — includes all structural checks.
    public var passed: Bool {
        !hasStringLookups && allIntegerPipelines && validFunctionStructure
            && correctSwapCount == expectedSwapCount
            && dispatchPipelineParity
    }
}

/// Verify generated Swift source for structural correctness.
public struct SmeltVerifier {

    /// Verify a generated encodeDecodeStep function source.
    ///
    /// - Parameters:
    ///   - source: Complete generated Swift source string.
    ///   - expectedLayers: Number of model layers (24 for Qwen 3.5 2B).
    /// - Returns: Verification report.
    public static func verify(
        source: String,
        expectedLayers: Int
    ) -> SmeltVerificationReport {
        let lines = source.components(separatedBy: "\n")
        let codeLines = lines.filter {
            !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//")
        }

        let totalDispatches = codeLines.filter {
            $0.contains("dispatchThread")
        }.count

        let pipelineStateChanges = codeLines.filter {
            $0.contains("setComputePipelineState")
        }.count

        let hasStringLookups = codeLines.contains { $0.contains("[\"") }

        let allIntegerPipelines = codeLines.filter {
            $0.contains("setComputePipelineState")
        }.allSatisfy { $0.contains("p[") }

        let trimmed = source.trimmingCharacters(in: .whitespacesAndNewlines)
        let validFunctionStructure = trimmed.contains("func encodeDecodeStep(")
            && trimmed.hasSuffix("}")

        let correctSwapCount = codeLines.filter {
            $0.contains("swap(&cur, &alt)")
        }.count

        let expectedSwapCount = expectedLayers * 2

        return SmeltVerificationReport(
            totalDispatches: totalDispatches,
            pipelineStateChanges: pipelineStateChanges,
            hasStringLookups: hasStringLookups,
            allIntegerPipelines: allIntegerPipelines,
            validFunctionStructure: validFunctionStructure,
            correctSwapCount: correctSwapCount,
            expectedSwapCount: expectedSwapCount
        )
    }
}

/// Optimized decode dispatch budget for Qwen 3.5 2B.
public struct SmeltDispatchBudget {
    /// Expected dispatches per DeltaNet layer after optimization.
    /// The canonical path emits conv1d + SiLU followed by one joint Q/K RMS
    /// scale dispatch and no longer emits the speculative-decode surface.
    public static let deltaNetPerLayer = 8
    /// Expected dispatches per attention layer.
    /// The optimized decode artifact keeps the gated-Q split path, per-head Q/K RMS norms,
    /// fused RoPE/KV update, and the capability-routed D256 attention set:
    /// compact, vector, and two-pass continuation kernels. Guards make the
    /// alternatives mutually exclusive at runtime, but every frozen route is
    /// represented in the dispatch budget.
    public static let attentionPerLayer = 24
    /// Expected shared dispatches per layer after optimization.
    /// Source emits 6, but both residual adds are fused away in optimized records.
    public static let sharedPerLayer = 4
    /// Expected global dispatches (embed + final_norm + lm_head + split argmax).
    public static let global = 5

    /// Compute expected total for a given model.
    public static func expectedTotal(
        numDelta: Int, numAttn: Int, numLayers: Int
    ) -> Int {
        numDelta * deltaNetPerLayer
            + numAttn * attentionPerLayer
            + numLayers * sharedPerLayer
            + global
    }
}
