// SmeltModel — Public API for Smelt inference.
//
// Wraps SmeltRuntime + SmeltPrefillRunner into a single generate call.
// All inference parameters (EOS tokens, think-skip, max tokens) come from
// the package manifest, which is compiled from the model spec.
//
// Usage:
//   let model = try SmeltModel(package: "model.smeltpkg")
//   let result = try model.generate(tokenIds: inputTokens)
//   print(result.tokens)  // [Int32]
//
// Streaming:
//   for try await token in model.generateStream(tokenIds: inputTokens) {
//       print(token.id)
//   }
//
// Callback:
//   try model.generate(tokenIds: inputTokens) { token in
//       print(token.id)
//       return true  // false to stop early
//   }

import Foundation
import Metal
import SmeltSchema

/// A single generated token, yielded during streaming generation.
public struct SmeltToken: Sendable {
    /// The token ID.
    public let id: Int32
    /// Position in the sequence at decode time.
    public let position: Int
}

/// Result of a generate call.
public struct SmeltGenerateResult: Sendable {
    /// Generated token IDs (excluding EOS).
    public let tokens: [Int32]
    /// Wall-clock time for the generate phase (excluding prefill).
    public let generateTime: Double
    /// Tokens per second during generation.
    public let tokensPerSecond: Double
    /// Wall-clock time for prefill.
    public let prefillTime: Double
}

/// Result of a generation that also captures the resulting live model state.
///
/// The snapshot can be reused as the starting point for the next turn or forked
/// into another native session without replaying the whole transcript.
public struct SmeltStatefulGenerateResult: Sendable {
    public let result: SmeltGenerateResult
    public let snapshot: SmeltPromptSnapshot
}

public typealias SmeltAllowedTokenMaskProvider = () throws -> [UInt32]?

/// High-level inference API. Loads a .smeltpkg and generates tokens.
///
/// All CAM behavior is driven by the package manifest; runnable text packages
/// must carry package-owned inference and decode policy.
/// - EOS tokens, think-skip, max tokens from the resolved inference policy
/// - Prefill model and cache from the `prefill` block
/// - Decode kernels and weights from the package
public final class SmeltModel: @unchecked Sendable {
    private static let preferredMetalPrefillChunkSize = 128


    private let packagePath: String
    private let runtime: SmeltRuntime
    private let prefiller: SmeltPrefillRunner?
    private let inferenceConfig: SmeltInferenceManifest
    private let contextLimit: Int
    private let preparedPrefix: SmeltPreparedPrefix?

    private struct FinishedGeneration {
        let result: SmeltGenerateResult
        let lastToken: Int32?
        let lastTokenPosition: Int
    }

    /// Load a .smeltpkg and prepare for inference.
    ///
    /// - Parameter packagePath: Path to the .smeltpkg directory.
    /// - Throws: If the package is invalid or Metal device is unavailable.
    public init(
        package packagePath: String,
        contextLimit: Int? = nil,
        manifest: SmeltManifest? = nil
    ) throws {
        self.packagePath = packagePath
        self.runtime = try SmeltRuntime(
            packagePath: packagePath,
            contextLimit: contextLimit,
            manifest: manifest
        )

        let manifest = runtime.manifest

        guard manifest.headlessTrunkABI != true else {
            throw SmeltRuntimeError.invalidPackage(
                "SmeltModel requires a runnable text package, got headless trunk manifest"
            )
        }

        // Prefill is optional — decode-only packages skip it.
        // Metal prefill uses the dispatch table interpreter (no CoreML).
        if let prefillConfig = manifest.prefill {
            if prefillConfig.engine == "metal" {
                self.prefiller = nil  // Metal prefill handled via runtime.prefillStep()
            } else {
                self.prefiller = try runtime.makePrefillRunner()
            }
        } else {
            self.prefiller = nil
        }

        try manifest.validatePackageOwnedRuntimePolicy()

        self.inferenceConfig = try manifest.resolvedInferencePolicy().inference

        self.contextLimit = runtime.maxContextTokens

        self.preparedPrefix = try SmeltPreparedPrefix.load(packagePath: packagePath)
    }

    // MARK: - Prepared prompt prefix

    /// Token IDs of the package's prepared prompt prefix, when compiled into
    /// the package. Matching requests skip prefill by restoring that state.
    public var preparedPrefixTokenIds: [Int32]? { preparedPrefix?.tokenIds }
    public var preparedPrefixContinuation: SmeltPreparedPromptContinuation? {
        preparedPrefix?.continuation
    }

    private func preparedPrefixMatch(_ tokenIds: [Int32]) -> SmeltPreparedPrefix? {
        guard let prepared = preparedPrefix,
              tokenIds.count >= prepared.tokenIds.count,
              tokenIds.starts(with: prepared.tokenIds)
        else { return nil }
        return prepared
    }

    /// Maximum context length supported by this package.
    public var maxContextTokens: Int { contextLimit }

    /// Current runtime memory footprint, including live request-scoped capacity.
    public var memoryStats: SmeltRuntime.MemoryStats {
        runtime.memoryStats()
    }

    /// Prefill a reusable base prompt and capture the live recurrent/KV state.
    ///
    /// This is intended for fixed harness/system prompts that should be reused
    /// across many requests without paying the full prefill cost each time.
    public func captureBasePrompt(tokenIds: [Int32]) throws -> SmeltPromptSnapshot {
        try ensureInputFitsContext(tokenIds.count)
        guard !tokenIds.isEmpty else {
            return SmeltPromptSnapshot(
                promptLength: 0,
                nextToken: 0,
                byteCount: 0,
                capturedLength: 0,
                replayTokenIds: [],
                convStates: [],
                recStates: [],
                keyCaches: [],
                valueCaches: []
            )
        }

        let capturedLength = capturedSnapshotPrefixLength(for: tokenIds.count)
        let replayTokenIds = Array(tokenIds[capturedLength...])
        let capturedTokenIds = Array(tokenIds[..<capturedLength])
        let requestBatch = requestBatchCapacity(for: capturedTokenIds.count)
        let requestContext = max(capturedTokenIds.count, 1)
        try runtime.prepareForRequest(
            batchCapacity: requestBatch,
            contextCapacity: requestContext
        )
        defer { runtime.trimRequestBuffers() }
        runtime.resetWorkingBuffers()

        let (cur, pos) = try prefillPrompt(
            tokenIds: capturedTokenIds,
            startPos: 0,
            selectionMode: .argmax
        )
        return runtime.capturePromptSnapshot(
            capturedLength: pos,
            promptLength: tokenIds.count,
            nextToken: cur,
            replayTokenIds: replayTokenIds
        )
    }

    /// Capture a prepared state through the ordinary per-token execution path.
    /// This is an offline package-construction tool for runtimes whose batched
    /// prompt route is unavailable or has not earned the same numeric contract
    /// as decode. Runtime restore is identical regardless of capture mode.
    public func captureBasePromptSequential(
        tokenIds: [Int32]
    ) throws -> SmeltPromptSnapshot {
        try ensureInputFitsContext(tokenIds.count)
        guard !tokenIds.isEmpty else {
            return SmeltPromptSnapshot(
                promptLength: 0,
                nextToken: 0,
                byteCount: 0,
                capturedLength: 0,
                replayTokenIds: [],
                convStates: [],
                recStates: [],
                keyCaches: [],
                valueCaches: []
            )
        }
        try runtime.prepareForRequest(
            batchCapacity: 1,
            contextCapacity: tokenIds.count
        )
        defer { runtime.trimRequestBuffers() }
        runtime.resetWorkingBuffers()
        let diagnostics = ProcessInfo.processInfo.environment[
            "SMELT_PREPARED_PROMPT_DIAGNOSTICS"
        ] == "1"
        var nextToken: Int32 = 0
        for (position, tokenId) in tokenIds.enumerated() {
            let started = diagnostics ? CFAbsoluteTimeGetCurrent() : 0
            if diagnostics {
                fputs(
                    "prepared-prompt sequential position=\(position) phase=begin\n",
                    stderr
                )
            }
            nextToken = try runtime.decodeStep(
                tokenId: tokenId,
                position: Int32(position),
                selectionMode: .argmax
            )
            if diagnostics {
                let elapsedMS = (CFAbsoluteTimeGetCurrent() - started) * 1_000
                fputs(
                    "prepared-prompt sequential position=\(position) phase=complete "
                        + "elapsed_ms=\(String(format: "%.3f", elapsedMS))\n",
                    stderr
                )
            }
        }
        return runtime.capturePromptSnapshot(
            capturedLength: tokenIds.count,
            promptLength: tokenIds.count,
            nextToken: nextToken
        )
    }

    /// Generate using a previously captured base prompt snapshot plus a suffix.
    public func generate(
        from snapshot: SmeltPromptSnapshot,
        tokenIds suffixTokenIds: [Int32],
        selectionMode: SmeltSelectionMode = .argmax,
        allowedTokenMask: SmeltAllowedTokenMaskProvider? = nil
    ) throws -> SmeltGenerateResult {
        try generate(
            from: snapshot,
            tokenIds: suffixTokenIds,
            selectionMode: selectionMode,
            allowedTokenMask: allowedTokenMask
        ) { _ in true }
    }

    /// Generate tokens from prefilled input.
    ///
    /// Runs CoreML batch prefill (if configured), handles think-skip,
    /// then decodes on Metal GPU until EOS or max tokens.
    ///
    /// - Parameter tokenIds: Input token IDs (user content + suffix tokens).
    /// - Returns: Generated tokens and timing info.
    /// - Throws: If prefill fails.
    public func generate(
        tokenIds: [Int32],
        selectionMode: SmeltSelectionMode = .argmax,
        allowedTokenMask: SmeltAllowedTokenMaskProvider? = nil
    ) throws -> SmeltGenerateResult {
        try ensureInputFitsContext(tokenIds.count)
        guard !tokenIds.isEmpty else {
            return SmeltGenerateResult(
                tokens: [], generateTime: 0,
                tokensPerSecond: 0, prefillTime: 0
            )
        }
        if let prepared = preparedPrefixMatch(tokenIds) {
            return try generate(
                from: prepared.snapshot,
                tokenIds: Array(tokenIds[prepared.tokenIds.count...]),
                selectionMode: selectionMode,
                allowedTokenMask: allowedTokenMask
            )
        }

        let requestBatch = requestBatchCapacity(for: tokenIds.count)
        let requestContext = max(tokenIds.count, 1)
        try runtime.prepareForRequest(
            batchCapacity: requestBatch,
            contextCapacity: requestContext
        )
        defer { runtime.trimRequestBuffers() }
        runtime.resetWorkingBuffers()

        let prefillStart = CFAbsoluteTimeGetCurrent()
        let (cur, pos) = try prefillPrompt(
            tokenIds: tokenIds,
            startPos: 0,
            selectionMode: selectionMode,
            allowedTokenMask: allowedTokenMask
        )
        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart

        return try finishGeneration(
            cur: cur,
            pos: pos,
            prefillTime: prefillTime,
            selectionMode: selectionMode,
            allowedTokenMask: allowedTokenMask
        ) { _ in true }
    }

    // MARK: - Streaming generation

    /// Generate tokens with a per-token callback.
    ///
    /// Same pipeline as `generate()` — prefill, think-skip, decode —
    /// but calls `onToken` for each token as it decodes. The callback
    /// runs synchronously on the calling thread between GPU submissions.
    ///
    /// - Parameters:
    ///   - tokenIds: Input token IDs.
    ///   - onToken: Called for each generated token. Return `false` to stop early.
    /// - Returns: Timing info for the full generation.
    public func generate(
        tokenIds: [Int32],
        selectionMode: SmeltSelectionMode = .argmax,
        allowedTokenMask: SmeltAllowedTokenMaskProvider? = nil,
        onToken: (SmeltToken) -> Bool
    ) throws -> SmeltGenerateResult {
        try ensureInputFitsContext(tokenIds.count)
        guard !tokenIds.isEmpty else {
            return SmeltGenerateResult(
                tokens: [], generateTime: 0,
                tokensPerSecond: 0, prefillTime: 0
            )
        }
        if let prepared = preparedPrefixMatch(tokenIds) {
            return try generate(
                from: prepared.snapshot,
                tokenIds: Array(tokenIds[prepared.tokenIds.count...]),
                selectionMode: selectionMode,
                allowedTokenMask: allowedTokenMask,
                onToken: onToken
            )
        }

        let requestBatch = requestBatchCapacity(for: tokenIds.count)
        let requestContext = max(tokenIds.count, 1)
        try runtime.prepareForRequest(
            batchCapacity: requestBatch,
            contextCapacity: requestContext
        )
        defer { runtime.trimRequestBuffers() }
        runtime.resetWorkingBuffers()

        let prefillStart = CFAbsoluteTimeGetCurrent()
        let (cur, pos) = try prefillPrompt(
            tokenIds: tokenIds,
            startPos: 0,
            selectionMode: selectionMode,
            allowedTokenMask: allowedTokenMask
        )
        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart

        return try finishGeneration(
            cur: cur,
            pos: pos,
            prefillTime: prefillTime,
            selectionMode: selectionMode,
            allowedTokenMask: allowedTokenMask,
            onToken: onToken
        )
    }

    /// Generate using a previously captured snapshot and capture the updated
    /// state after the generated tokens.
    public func generateAndCapture(
        from snapshot: SmeltPromptSnapshot,
        tokenIds suffixTokenIds: [Int32],
        selectionMode: SmeltSelectionMode = .argmax,
        allowedTokenMask: SmeltAllowedTokenMaskProvider? = nil,
        suppressThinking: Bool? = nil,
        onToken: (SmeltToken) -> Bool
    ) throws -> SmeltStatefulGenerateResult {
        let continuationTokenIds = snapshot.replayTokenIds + suffixTokenIds
        try ensureInputFitsContext(snapshot.promptLength + suffixTokenIds.count)
        let requestBatch = requestBatchCapacity(for: continuationTokenIds.count)
        let requestContext = max(snapshot.promptLength + suffixTokenIds.count, 1)
        try runtime.prepareForRequest(
            batchCapacity: requestBatch,
            contextCapacity: requestContext
        )
        defer { runtime.trimRequestBuffers() }
        runtime.resetWorkingBuffers()
        try runtime.restorePromptSnapshot(snapshot)

        let prefillStart = CFAbsoluteTimeGetCurrent()
        let cur: Int32
        let pos: Int
        if continuationTokenIds.isEmpty {
            cur = snapshot.nextToken
            pos = snapshot.promptLength
        } else {
            let result = try prefillPrompt(
                tokenIds: continuationTokenIds,
                startPos: snapshot.capturedLength,
                selectionMode: selectionMode,
                allowedTokenMask: allowedTokenMask
            )
            cur = result.cur
            pos = result.pos
        }
        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart

        let finished = try finishGenerationDetailed(
            cur: cur,
            pos: pos,
            prefillTime: prefillTime,
            selectionMode: selectionMode,
            allowedTokenMask: allowedTokenMask,
            suppressThinking: suppressThinking,
            onToken: onToken
        )
        let captured = try captureSnapshotAfterGeneration(
            finished: finished,
            fallbackSnapshot: snapshot,
            selectionMode: selectionMode
        )
        return SmeltStatefulGenerateResult(
            result: finished.result,
            snapshot: captured
        )
    }

    /// Generate from fresh token IDs and capture the resulting state.
    public func generateAndCapture(
        tokenIds: [Int32],
        selectionMode: SmeltSelectionMode = .argmax,
        allowedTokenMask: SmeltAllowedTokenMaskProvider? = nil,
        suppressThinking: Bool? = nil,
        onToken: (SmeltToken) -> Bool
    ) throws -> SmeltStatefulGenerateResult {
        try ensureInputFitsContext(tokenIds.count)
        guard !tokenIds.isEmpty else {
            let empty = SmeltPromptSnapshot(
                promptLength: 0,
                nextToken: 0,
                byteCount: 0,
                capturedLength: 0,
                replayTokenIds: [],
                convStates: [],
                recStates: [],
                keyCaches: [],
                valueCaches: []
            )
            return SmeltStatefulGenerateResult(
                result: SmeltGenerateResult(
                    tokens: [],
                    generateTime: 0,
                    tokensPerSecond: 0,
                    prefillTime: 0
                ),
                snapshot: empty
            )
        }

        let requestBatch = requestBatchCapacity(for: tokenIds.count)
        let requestContext = max(tokenIds.count, 1)
        try runtime.prepareForRequest(
            batchCapacity: requestBatch,
            contextCapacity: requestContext
        )
        defer { runtime.trimRequestBuffers() }
        runtime.resetWorkingBuffers()

        let prefillStart = CFAbsoluteTimeGetCurrent()
        let (cur, pos) = try prefillPrompt(
            tokenIds: tokenIds,
            startPos: 0,
            selectionMode: selectionMode,
            allowedTokenMask: allowedTokenMask
        )
        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart

        let finished = try finishGenerationDetailed(
            cur: cur,
            pos: pos,
            prefillTime: prefillTime,
            selectionMode: selectionMode,
            allowedTokenMask: allowedTokenMask,
            suppressThinking: suppressThinking,
            onToken: onToken
        )
        let captured = try captureSnapshotAfterGeneration(
            finished: finished,
            fallbackSnapshot: nil,
            selectionMode: selectionMode
        )
        return SmeltStatefulGenerateResult(
            result: finished.result,
            snapshot: captured
        )
    }

    /// Generate with a callback from a restored base prompt snapshot.
    public func generate(
        from snapshot: SmeltPromptSnapshot,
        tokenIds suffixTokenIds: [Int32],
        selectionMode: SmeltSelectionMode = .argmax,
        allowedTokenMask: SmeltAllowedTokenMaskProvider? = nil,
        onToken: (SmeltToken) -> Bool
    ) throws -> SmeltGenerateResult {
        let continuationTokenIds = snapshot.replayTokenIds + suffixTokenIds
        try ensureInputFitsContext(snapshot.promptLength + suffixTokenIds.count)
        let requestBatch = requestBatchCapacity(for: continuationTokenIds.count)
        let requestContext = max(snapshot.promptLength + suffixTokenIds.count, 1)
        try runtime.prepareForRequest(
            batchCapacity: requestBatch,
            contextCapacity: requestContext
        )
        defer { runtime.trimRequestBuffers() }
        runtime.resetWorkingBuffers()
        try runtime.restorePromptSnapshot(snapshot)

        let prefillStart = CFAbsoluteTimeGetCurrent()
        let cur: Int32
        let pos: Int
        if continuationTokenIds.isEmpty {
            cur = snapshot.nextToken
            pos = snapshot.promptLength
        } else {
            let result = try prefillPrompt(
                tokenIds: continuationTokenIds,
                startPos: snapshot.capturedLength,
                selectionMode: selectionMode,
                allowedTokenMask: allowedTokenMask
            )
            cur = result.cur
            pos = result.pos
        }
        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart

        return try finishGeneration(
            cur: cur,
            pos: pos,
            prefillTime: prefillTime,
            selectionMode: selectionMode,
            allowedTokenMask: allowedTokenMask,
            onToken: onToken
        )
    }

    private func requestBatchCapacity(for tokenCount: Int) -> Int {
        runtime.hasMetalPrefill
            ? min(max(tokenCount, 1), max(runtime.maxPrefillBatchSize, 1))
            : 1
    }

    private func capturedSnapshotPrefixLength(for tokenCount: Int) -> Int {
        let chunkSize = metalPrefillChunkSize
        guard tokenCount > chunkSize else { return tokenCount }
        let aligned = (tokenCount / chunkSize) * chunkSize
        return max(aligned, chunkSize)
    }

    private func prefillPrompt(
        tokenIds: [Int32],
        startPos: Int,
        selectionMode: SmeltSelectionMode,
        allowedTokenMask: SmeltAllowedTokenMaskProvider? = nil
    ) throws -> (cur: Int32, pos: Int) {
        if runtime.hasMetalPrefill && runtime.maxPrefillBatchSize > 0 {
            return try runMetalPrefill(
                tokenIds: tokenIds,
                startPos: startPos,
                selectionMode: selectionMode,
                allowedTokenMask: allowedTokenMask
            )
        } else if let prefiller, startPos == 0 {
            if allowedTokenMask != nil {
                throw NSError(
                    domain: "SmeltModel",
                    code: 1,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Constrained sampling requires Metal prefill"
                    ]
                )
            }
            if !selectionMode.usesArgmaxFastPath {
                throw NSError(
                    domain: "SmeltModel",
                    code: 1,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Temperature sampling requires Metal prefill"
                    ]
                )
            }
            let prefillResult = try prefiller.runPrefill(tokenIds: tokenIds)
            return (prefillResult.firstToken, Int(prefillResult.position))
        } else {
            guard !tokenIds.isEmpty else { return (0, startPos) }
            var cur: Int32 = 0
            for (i, tid) in tokenIds.enumerated() {
                let isLast = i == tokenIds.count - 1
                cur = try runtime.decodeStep(
                    tokenId: tid,
                    position: Int32(startPos + i),
                    selectionMode: isLast ? selectionMode : .argmax,
                    allowedTokenMask: isLast ? allowedTokenMask : nil
                )
            }
            return (cur, startPos + tokenIds.count)
        }
    }

    private func finishGeneration(
        cur initialCur: Int32,
        pos initialPos: Int,
        prefillTime: Double,
        selectionMode: SmeltSelectionMode,
        allowedTokenMask: SmeltAllowedTokenMaskProvider? = nil,
        suppressThinking: Bool? = nil,
        onToken: (SmeltToken) -> Bool
    ) throws -> SmeltGenerateResult {
        try finishGenerationDetailed(
            cur: initialCur,
            pos: initialPos,
            prefillTime: prefillTime,
            selectionMode: selectionMode,
            allowedTokenMask: allowedTokenMask,
            suppressThinking: suppressThinking,
            onToken: onToken
        ).result
    }

    private func ensureInputFitsContext(_ tokenCount: Int) throws {
        guard tokenCount < contextLimit else {
            throw SmeltRuntimeError.inputExceedsContext(
                limit: contextLimit, requested: tokenCount
            )
        }
    }

    private func finishGenerationDetailed(
        cur initialCur: Int32,
        pos initialPos: Int,
        prefillTime: Double,
        selectionMode: SmeltSelectionMode,
        allowedTokenMask: SmeltAllowedTokenMaskProvider? = nil,
        suppressThinking: Bool? = nil,
        onToken: (SmeltToken) -> Bool
    ) throws -> FinishedGeneration {
        let maxTokens = inferenceConfig.maxTokens
        let eosTokens = Set(inferenceConfig.eosTokens)
        var cur = initialCur
        var pos = initialPos

        // Skip a leading <think> turn unless the package opted into thinking.
        // nil → derive from the package policy (the package-owned default);
        // downstream agent consumers pass `true` to force non-thinking for tool
        // decode. `.enabled` lets the trace flow inline (delimiters are stripped
        // by special-token decode — reasoning-content separation is a follow-up).
        let doSkip = suppressThinking ?? (inferenceConfig.thinkingPolicy != .enabled)
        if doSkip,
           let thinkTok = inferenceConfig.thinkToken,
           let thinkEnd = inferenceConfig.thinkEndToken,
           cur == thinkTok
        {
            _ = try runtime.decodeStep(tokenId: thinkEnd, position: Int32(pos))
            pos += 1
            if let suffix = inferenceConfig.thinkSkipSuffix {
                cur = try runtime.decodeStep(
                    tokenId: suffix,
                    position: Int32(pos),
                    selectionMode: selectionMode
                )
                pos += 1
            }
        }

        let genStart = CFAbsoluteTimeGetCurrent()
        var tokens: [Int32] = [cur]
        var lastToken: Int32? = cur
        var lastTokenPosition = pos
        var stopped = !onToken(SmeltToken(id: cur, position: pos))

        while !stopped && tokens.count < maxTokens && pos < contextLimit {
            cur = try runtime.decodeStep(
                tokenId: cur,
                position: Int32(pos),
                selectionMode: selectionMode,
                allowedTokenMask: allowedTokenMask
            )
            if eosTokens.contains(cur) { break }
            tokens.append(cur)
            pos += 1
            lastToken = cur
            lastTokenPosition = pos
            stopped = !onToken(SmeltToken(id: cur, position: pos))
        }

        let genTime = CFAbsoluteTimeGetCurrent() - genStart
        let tps = genTime > 0 ? Double(tokens.count) / genTime : 0

        return FinishedGeneration(
            result: SmeltGenerateResult(
                tokens: tokens,
                generateTime: genTime,
                tokensPerSecond: tps,
                prefillTime: prefillTime
            ),
            lastToken: lastToken,
            lastTokenPosition: lastTokenPosition
        )
    }

    private func captureSnapshotAfterGeneration(
        finished: FinishedGeneration,
        fallbackSnapshot: SmeltPromptSnapshot?,
        selectionMode: SmeltSelectionMode
    ) throws -> SmeltPromptSnapshot {
        guard let lastToken = finished.lastToken else {
            return fallbackSnapshot ?? SmeltPromptSnapshot(
                promptLength: 0,
                nextToken: 0,
                byteCount: 0,
                capturedLength: 0,
                replayTokenIds: [],
                convStates: [],
                recStates: [],
                keyCaches: [],
                valueCaches: []
            )
        }

        let position = min(max(finished.lastTokenPosition, 0), contextLimit - 1)
        let lookahead = try runtime.decodeStep(
            tokenId: lastToken,
            position: Int32(position),
            selectionMode: selectionMode
        )
        let capturedLength = min(position + 1, contextLimit)
        return runtime.capturePromptSnapshot(
            capturedLength: capturedLength,
            promptLength: capturedLength,
            nextToken: lookahead
        )
    }

    private func runMetalPrefill(
        tokenIds: [Int32],
        startPos: Int = 0,
        selectionMode: SmeltSelectionMode,
        allowedTokenMask: SmeltAllowedTokenMaskProvider? = nil
    ) throws -> (cur: Int32, pos: Int) {
        guard !tokenIds.isEmpty else { return (0, startPos) }

        let chunkSize = metalPrefillChunkSize
        var start = 0
        var cur: Int32 = 0

        while start < tokenIds.count {
            let end = min(start + chunkSize, tokenIds.count)
            let chunk = Array(tokenIds[start..<end])
            let isLast = end == tokenIds.count
            cur = try runtime.prefillStep(
                tokenIds: chunk,
                startPos: Int32(startPos + start),
                selectionMode: isLast ? selectionMode : .argmax,
                allowedTokenMask: isLast ? allowedTokenMask : nil
            )
            start = end
        }

        return (cur, startPos + tokenIds.count)
    }

    private var metalPrefillChunkSize: Int {
        min(
            max(runtime.maxPrefillBatchSize, 1),
            Self.preferredMetalPrefillChunkSize
        )
    }

    /// Stream tokens as an AsyncSequence.
    ///
    /// Usage:
    ///   for try await token in model.generateStream(tokenIds: input) {
    ///       print(token.id)
    ///   }
    ///
    /// Prefill errors are thrown on first iteration. The stream finishes
    /// on EOS or max tokens.
    public func generateStream(
        tokenIds: [Int32],
        selectionMode: SmeltSelectionMode = .argmax
    ) -> AsyncThrowingStream<SmeltToken, any Error> {
        AsyncThrowingStream { continuation in
            do {
                _ = try self.generate(
                    tokenIds: tokenIds,
                    selectionMode: selectionMode
                ) { token in
                    continuation.yield(token)
                    return true
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }
}
