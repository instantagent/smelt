import Foundation
import SmeltRuntime
import SmeltServe
import SmeltSchema

func runPromptDecodeOnly(
    packagePath: String,
    inputIds: [Int32],
    inference: SmeltInferenceManifest,
    contextLimit: Int?,
    maxTokens: Int,
    selectionMode: SmeltSelectionMode,
    construction: CAMTextRuntimeConstruction? = nil
) throws -> (tokens: [Int32], prefillTime: Double, generateTime: Double) {
    guard !inputIds.isEmpty else { return ([], 0, 0) }

    try construction?.requirePackagePath(packagePath)
    let runtime = try construction?.makeRuntime(contextLimit: contextLimit)
        ?? SmeltRuntime(packagePath: packagePath, contextLimit: contextLimit)
    guard inputIds.count < runtime.maxContextTokens else {
        throw SmeltRuntimeError.inputExceedsContext(
            limit: runtime.maxContextTokens,
            requested: inputIds.count
        )
    }
    var cur: Int32 = 0
    let prefillStart = CFAbsoluteTimeGetCurrent()
    for (position, tokenId) in inputIds.enumerated() {
        let isLast = position == inputIds.count - 1
        cur = try runtime.decodeStep(
            tokenId: tokenId,
            position: Int32(position),
            selectionMode: isLast ? selectionMode : .argmax
        )
    }
    let prefillElapsed = CFAbsoluteTimeGetCurrent() - prefillStart

    var pos = inputIds.count
    // Skip a leading <think> turn unless the package opted into thinking
    // (thinking_policy: enabled lets the trace flow).
    if inference.thinkingPolicy != .enabled,
       let thinkTok = inference.thinkToken,
       let thinkEnd = inference.thinkEndToken,
       cur == thinkTok
    {
        _ = try runtime.decodeStep(tokenId: thinkEnd, position: Int32(pos))
        pos += 1
        if let suffix = inference.thinkSkipSuffix {
            cur = try runtime.decodeStep(
                tokenId: suffix,
                position: Int32(pos),
                selectionMode: selectionMode
            )
            pos += 1
        }
    }

    let constructionLimit = construction?.effectiveMaxTokens(maxTokens) ?? maxTokens
    let limit = min(constructionLimit, inference.maxTokens, runtime.maxContextTokens - pos)
    let eosTokens = Set(inference.eosTokens)
    var generated: [Int32] = []
    let genStart = CFAbsoluteTimeGetCurrent()
    while generated.count < limit {
        if eosTokens.contains(cur) { break }
        generated.append(cur)
        cur = try runtime.decodeStep(
            tokenId: cur,
            position: Int32(pos),
            selectionMode: selectionMode
        )
        pos += 1
    }
    let genElapsed = CFAbsoluteTimeGetCurrent() - genStart

    return (generated, prefillElapsed, genElapsed)
}

private func formatLogitStats(_ logits: [Float]) -> String {
    let nanCount = logits.filter { $0.isNaN }.count
    let finite = logits.filter { $0.isFinite }
    let nonzeroFinite = finite.filter { $0 != 0 }
    let minVal = finite.min() ?? .nan
    let maxVal = finite.max() ?? .nan
    return "count=\(logits.count) nan=\(nanCount) finite=\(finite.count) "
        + "nonzeroFinite=\(nonzeroFinite.count) min=\(minVal) max=\(maxVal)"
}

func runPromptDebug(
    packagePath: String,
    prompt: String,
    inputIds: [Int32],
    tokenizer: SmeltTokenizer,
    template: String,
    mode: String,
    hiddenSize: Int,
    construction: CAMTextRuntimeConstruction? = nil
) throws {
    let contextLimit = try parsePositiveIntArg("--context-limit")
    try construction?.requirePackagePath(packagePath)
    let camDecodeOnlyDebug = construction?.debugRequiresDecodeOnly() ?? false
    let runtime = try construction?.makeRuntime(contextLimit: contextLimit)
        ?? SmeltRuntime(packagePath: packagePath, contextLimit: contextLimit)
    let stopAfterDispatch = Int(parseArg("--stop-after-dispatch", default: "")) ?? -1
    let debugTokenIndex = Int(parseArg("--debug-token-index", default: "0")) ?? 0
    let debugPrefixTokens = Int(parseArg("--debug-prefix-tokens", default: "")) ?? -1
    let traceVerifyTokenCount = Int(parseArg("--trace-verify-token-count", default: "")) ?? -1
    let traceLayers = args.contains("--trace-layers")
    let debugPrefill = args.contains("--debug-prefill")
    let forceDecode = args.contains("--force-decode") || camDecodeOnlyDebug
    if debugPrefill && camDecodeOnlyDebug {
        throw NSError(
            domain: "SmeltCLI",
            code: 1,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "--debug-prefill is unavailable for module decode-only flow"
            ]
        )
    }
    let dumpTraceLabel = parseArg("--dump-trace-label", default: "")
    let dumpTraceOccurrence = try parseTraceOccurrenceSelection(
        parseArg("--dump-trace-occurrence", default: "last")
    )
    let dumpSlot = Int(parseArg("--dump-slot", default: "")) ?? -1
    let dumpOffset = Int(parseArg("--dump-offset", default: "0")) ?? 0
    let dumpCount = Int(parseArg("--dump-count", default: "32")) ?? 32
    let activePrompt: String
    let activeInputIds: [Int32]
    if traceVerifyTokenCount > 0 {
        let verifyPrompt = try buildRepeatedVerifyPrompt(
            packagePath: packagePath,
            template: template,
            maxTokenCount: traceVerifyTokenCount,
            tokenizer: tokenizer
        )
        activePrompt = verifyPrompt.prompt
        activeInputIds = Array(verifyPrompt.tokenIds.prefix(traceVerifyTokenCount))
    } else if debugPrefixTokens > 0 {
        activePrompt = prompt
        activeInputIds = Array(inputIds.prefix(debugPrefixTokens))
    } else {
        activePrompt = prompt
        activeInputIds = inputIds
    }

    if traceLayers {
        try runLayerTraceDebug(
            packagePath: packagePath,
            prompt: activePrompt,
            inputIds: activeInputIds,
            tokenizer: tokenizer,
            contextLimit: contextLimit,
            construction: construction
        )
        return
    }

    if !dumpTraceLabel.isEmpty {
        let markers = try loadTraceMarkers(packagePath: packagePath)
        let marker: SmeltTraceMarker?
        let markerMatches: [SmeltTraceMarker]
        let mode: DebugTraceMode
        if forceDecode {
            markerMatches = traceMarkerMatches(
                label: dumpTraceLabel,
                markers: markers.decode
            )
            marker = selectTraceMarker(
                label: dumpTraceLabel,
                markers: markers.decode,
                occurrence: dumpTraceOccurrence
            )
            mode = .decode
        } else {
            markerMatches = traceMarkerMatches(
                label: dumpTraceLabel,
                markers: markers.prefill
            )
            marker = selectTraceMarker(
                label: dumpTraceLabel,
                markers: markers.prefill,
                occurrence: dumpTraceOccurrence
            )
            mode = .prefill
        }
        guard let marker else {
            throw NSError(
                domain: "SmeltCLI",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Trace label '\(dumpTraceLabel)' not found for \(forceDecode ? "decode" : "prefill") markers"
                ]
            )
        }

        let sample = try captureTraceSample(
            packagePath: packagePath,
            tokenIds: activeInputIds,
            marker: marker,
            mode: mode,
            usesBatchedRowOffset: !(mode == .prefill && markerMatches.count > 1),
            contextLimit: contextLimit,
            count: dumpCount,
            construction: construction
        )
        let values = sample.values.map { String(format: "%.4f", $0) }

        fputs("Loaded: \(packagePath)\n", stderr)
        fputs("Prompt: \(activePrompt)\n", stderr)
        fputs("Prompt tokens: \(activeInputIds.count)\n", stderr)
        fputs("Trace label: \(marker.label)\n", stderr)
        fputs("Trace mode: \(forceDecode ? "decode" : "prefill")\n", stderr)
        fputs("Trace slot: \(marker.bufferSlot)\n", stderr)
        fputs("Trace dispatch: \(marker.dispatchCount)\n", stderr)
        fputs(
            "Trace token: \(sample.token) (\(tokenizer.decode([sample.token])))\n",
            stderr
        )
        fputs("Trace values[0:\(values.count)]: \(values)\n", stderr)
        return
    }

    let firstToken: Int32

    if stopAfterDispatch > 0 {
        if debugPrefill {
            firstToken = try runtime.debugPrefillStep(
                tokenIds: activeInputIds,
                startPos: 0,
                maxDispatches: stopAfterDispatch
            )
        } else {
            guard debugTokenIndex >= 0, debugTokenIndex < activeInputIds.count else {
                throw NSError(
                    domain: "SmeltCLI",
                    code: 1,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "--debug-token-index \(debugTokenIndex) is out of range for \(activeInputIds.count) prompt tokens"
                    ]
                )
            }
            runtime.resetWorkingBuffers()
            if debugTokenIndex > 0 {
                for position in 0..<debugTokenIndex {
                    _ = try runtime.decodeStep(
                        tokenId: activeInputIds[position],
                        position: Int32(position)
                    )
                }
            }
            firstToken = try runtime.debugDecodeStep(
                tokenId: activeInputIds[debugTokenIndex],
                position: Int32(debugTokenIndex),
                maxDispatches: stopAfterDispatch
            )
        }
    } else if !forceDecode && runtime.hasMetalPrefill && runtime.maxPrefillBatchSize > 0 {
        var cur: Int32 = 0
        let chunkSize = max(runtime.maxPrefillBatchSize, 1)
        var start = 0
        while start < activeInputIds.count {
            let end = min(start + chunkSize, activeInputIds.count)
            cur = try runtime.prefillStep(
                tokenIds: Array(activeInputIds[start..<end]),
                startPos: Int32(start)
            )
            start = end
        }
        firstToken = cur
    } else {
        guard !activeInputIds.isEmpty else {
            fputs("No input IDs to debug.\n", stderr)
            return
        }
        var cur: Int32 = 0
        for (position, tokenId) in activeInputIds.enumerated() {
            cur = try runtime.decodeStep(tokenId: tokenId, position: Int32(position))
        }
        firstToken = cur
    }

    let logits = runtime.allLogits()
    let top5 = runtime.topKLogits(k: 5)
    let logitsPrefix = Array(logits.prefix(20)).map { String(format: "%.4f", $0) }
    let hiddenA = runtime.dumpSlot(0, count: 12)
    let hiddenB = runtime.dumpSlot(1, count: 12)
    let normOut = runtime.dumpSlot(8, count: 12)
    let hasBatchedRows = !forceDecode && activeInputIds.count > 1 && hiddenSize > 0
    let lastRowOffset = hasBatchedRows ? (activeInputIds.count - 1) * hiddenSize : 0
    let hiddenALast = hasBatchedRows
        ? runtime.dumpSlot(0, elementOffset: lastRowOffset, count: 12)
        : []
    let hiddenBLast = hasBatchedRows
        ? runtime.dumpSlot(1, elementOffset: lastRowOffset, count: 12)
        : []
    let normOutLast = hasBatchedRows
        ? runtime.dumpSlot(8, elementOffset: lastRowOffset, count: 12)
        : []
    let slotDump = dumpSlot >= 0
        ? runtime.dumpSlot(dumpSlot, elementOffset: dumpOffset, count: dumpCount)
        : []

    fputs("Loaded: \(packagePath)\n", stderr)
    fputs("Prompt: \(activePrompt)\n", stderr)
    fputs("Prompt tokens: \(activeInputIds.count)\n", stderr)
    fputs("Input IDs: \(activeInputIds)\n", stderr)
    let debugMode = forceDecode ? "forced-decode" : mode
    fputs("Prefill mode: \(debugMode)\n", stderr)
    if stopAfterDispatch > 0 {
        if debugPrefill {
            fputs("Debug stop: prefill dispatch \(stopAfterDispatch)\n", stderr)
            for line in runtime.prefillDispatchWindow(endingAt: stopAfterDispatch) {
                fputs("  \(line)\n", stderr)
            }
        } else {
            fputs(
                "Debug stop: dispatch \(stopAfterDispatch) on token index \(debugTokenIndex)"
                    + " (id \(activeInputIds[debugTokenIndex]))\n",
                stderr
            )
        }
    }
    fputs("First token: \(firstToken) (\(tokenizer.decode([firstToken])))\n", stderr)
    fputs("hiddenA[0:12]: \(hiddenA.map { String(format: "%.4f", $0) })\n", stderr)
    fputs("hiddenB[0:12]: \(hiddenB.map { String(format: "%.4f", $0) })\n", stderr)
    fputs("normOut[0:12]: \(normOut.map { String(format: "%.4f", $0) })\n", stderr)
    if hasBatchedRows {
        fputs(
            "hiddenA[last row][0:12]: \(hiddenALast.map { String(format: "%.4f", $0) })\n",
            stderr
        )
        fputs(
            "hiddenB[last row][0:12]: \(hiddenBLast.map { String(format: "%.4f", $0) })\n",
            stderr
        )
        fputs(
            "normOut[last row][0:12]: \(normOutLast.map { String(format: "%.4f", $0) })\n",
            stderr
        )
    }
    if dumpSlot >= 0 {
        fputs(
            "slot\(dumpSlot)[\(dumpOffset):\(dumpOffset + slotDump.count)]: "
                + "\(slotDump.map { String(format: "%.4f", $0) })\n",
            stderr
        )
    }
    fputs("Logit stats: \(formatLogitStats(logits))\n", stderr)
    fputs("Logits[0:20]: \(logitsPrefix)\n", stderr)
    fputs("Top-5 logits:\n", stderr)
    for (tok, val) in top5 {
        fputs("  \(tok) (\(tokenizer.decode([tok]))): \(String(format: "%.4f", val))\n", stderr)
    }
}

func runPrompt(
    packagePath: String,
    prompt: String,
    maxTokens: Int,
    template: String,
    selectionMode: SmeltSelectionMode,
    selectionDescription: String,
    contextLimit: Int?,
    systemPrompt: String = "",
    grammarBindings: [String: [String]] = [:],
    construction: CAMTextRuntimeConstruction? = nil
) throws {
    try construction?.requirePackagePath(packagePath)
    if args.contains("--debug") {
        let tokenizer = try construction?.makeTokenizer()
            ?? SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
        let (manifest, inference) = try construction?.loadManifestConfig()
            ?? loadInferenceConfig(packagePath: packagePath)
        let resolvedTemplate = try construction?.resolveTemplate(cliOverride: template)
            ?? resolveChatTemplate(
                cliOverride: template, packageTemplate: inference.chatTemplate)
        let renderedInput: (prompt: String, systemPrompt: String)
        if let construction {
            renderedInput = try construction.renderPrompt(
                prompt: prompt,
                systemPrompt: systemPrompt
            )
        } else {
            renderedInput = (prompt, systemPrompt)
        }
        var inputIds = try buildInputIds(
            prompt: renderedInput.prompt,
            tokenizer: tokenizer,
            template: resolvedTemplate,
            thinkingPolicy: resolvedThinkingPolicy(inference)
        )
        if !renderedInput.systemPrompt.isEmpty {
            let systemIds = try buildSystemIds(
                systemPrompt: renderedInput.systemPrompt,
                tokenizer: tokenizer,
                template: resolvedTemplate
            )
            inputIds = systemIds + inputIds
        }
        let mode: String
        if let construction {
            mode = try construction.runtimeModeDescription(manifest: manifest).mode
        } else {
            let (_, resolvedMode) = shouldUseModelGenerate(
                packagePath: packagePath,
                manifest: manifest
            )
            mode = resolvedMode
        }
        try runPromptDebug(
            packagePath: packagePath,
            prompt: prompt,
            inputIds: inputIds,
            tokenizer: tokenizer,
            template: resolvedTemplate,
            mode: mode,
            hiddenSize: manifest.config.hiddenSize,
            construction: construction
        )
        return
    }

    let result = try evaluatePrompt(
        packagePath: packagePath,
        prompt: prompt,
        maxTokens: maxTokens,
        template: template,
        selectionMode: selectionMode,
        selectionDescription: selectionDescription,
        contextLimit: contextLimit,
        systemPrompt: systemPrompt,
        grammarBindings: grammarBindings,
        construction: construction
    )

    printPromptRunResult(result, packagePath: packagePath)
}

func printPromptRunResult(_ result: PromptRunResult, packagePath: String) {
    fputs("Loaded: \(packagePath)\n", stderr)
    fputs("Prompt: \(result.prompt)\n", stderr)
    fputs("Prompt tokens: \(result.promptTokens)\n", stderr)
    fputs("Prefill mode: \(result.mode)\n", stderr)
    fputs("Selection: \(result.selection)\n", stderr)
    fputs("Completion:\n", stderr)
    // The completion is the program's output; everything else is
    // diagnostics. stdout/stderr separation is what lets a package
    // compose as a pipe stage: `... | smelt run pkg | jq .`
    print(result.completion)
    fflush(stdout)
    fputs("Generated token IDs: \(result.generated)\n", stderr)
    fputs(
        "Timing: prefill \(String(format: "%.1f", result.prefillElapsed * 1_000))ms, "
            + "generate \(String(format: "%.1f", result.genElapsed * 1_000))ms, "
            + "\(result.generated.isEmpty ? "0.0" : String(format: "%.1f", Double(result.generated.count) / result.genElapsed)) tok/s\n",
        stderr
    )
}

struct PromptRunResult {
    let prompt: String
    let promptTokens: Int
    let mode: String
    let selection: String
    let generated: [Int32]
    let completion: String
    let prefillElapsed: Double
    let genElapsed: Double
}

/// Loaded per-package state for `smelt run` evaluation. A linger worker
/// holds one of these across requests so repeat invocations skip the
/// tokenizer/model load.
struct RunContext {
    let packagePath: String
    let tokenizer: SmeltTokenizer
    let inference: SmeltInferenceManifest
    let chatTemplate: String?
    let modelName: String
    let mode: String
    let contextLimit: Int?
    let construction: CAMTextRuntimeConstruction?
    /// Non-nil when the package takes the SmeltModel.generate path.
    let model: SmeltModel?
    /// Background-compiled matcher for the package's compiled JSON-schema
    /// grammar, if any. The llguidance tokenizer build costs ~0.4s on a 250k
    /// vocab, so it overlaps model init and is joined only when a request
    /// needs the mask. Clones are taken fresh per request (a clone's state is
    /// consumed by decoding).
    final class GrammarBox: @unchecked Sendable {
        private let group = DispatchGroup()
        private var prototypeResult:
            Result<SmeltLLGuidanceMatcher, Error> =
                .failure(SmeltLLGuidanceError.matcherInitFailed("not built"))

        init(
            tokenizer: SmeltTokenizer,
            eosTokens: [Int32],
            jsonSchema: String,
            serializedTrie: Data?
        ) {
            nonisolated(unsafe) let tokenizerRef = tokenizer
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                self.prototypeResult = Result {
                    let llgTokenizer = try Self.makeLLGTokenizer(
                        tokenizer: tokenizerRef,
                        eosTokens: eosTokens,
                        serializedTrie: serializedTrie
                    )
                    return try SmeltLLGuidanceMatcher(
                        tokenizer: llgTokenizer,
                        jsonSchema: jsonSchema
                    )
                }
                self.group.leave()
            }
        }

        /// Prefer the trie compiled next to the grammar (a few-ms load) over
        /// re-building it from the vocabulary (~0.4s on a 250k vocab); a
        /// corrupt or stale trie falls back to the full build.
        private static func makeLLGTokenizer(
            tokenizer: SmeltTokenizer,
            eosTokens: [Int32],
            serializedTrie: Data?
        ) throws -> SmeltLLGuidanceTokenizer {
            if let serializedTrie {
                do {
                    let start = CFAbsoluteTimeGetCurrent()
                    let llgTokenizer = try SmeltLLGuidanceTokenizer(
                        tokenizer: tokenizer, serializedTrie: serializedTrie
                    )
                    if ProcessInfo.processInfo.environment["SMELT_STARTUP_TRACE"] == "1" {
                        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000
                        fputs(
                            "startup: \(String(format: "%+7.1fms", ms))  "
                                + "llg tokenizer from compiled trie\n",
                            stderr
                        )
                    }
                    return llgTokenizer
                } catch {
                    fputs(
                        "warning: compiled llguidance trie failed to load "
                            + "(\(error)); rebuilding from vocabulary\n",
                        stderr
                    )
                }
            }
            return try SmeltLLGuidanceTokenizer(
                tokenizer: tokenizer, eosTokens: eosTokens
            )
        }

        func freshMatcher() throws -> SmeltLLGuidanceMatcher {
            group.wait()
            return try prototypeResult.get().freshCopy()
        }
    }

    let grammarBox: GrammarBox?

    static func load(
        packagePath: String,
        contextLimit: Int?,
        grammarBindings: [String: [String]] = [:],
        construction: CAMTextRuntimeConstruction? = nil
    ) throws -> RunContext {
        try construction?.requirePackagePath(packagePath)
        let traceStartup =
            ProcessInfo.processInfo.environment["SMELT_STARTUP_TRACE"] == "1"
        func stamp(_ label: String, since start: Double) {
            guard traceStartup else { return }
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            fputs("startup: \(String(format: "%+7.1fms", elapsed))  \(label)\n", stderr)
        }

        let tokenizerStart = CFAbsoluteTimeGetCurrent()
        let tokenizer = try construction?.makeTokenizer()
            ?? SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
        stamp("tokenizer load", since: tokenizerStart)
        let (manifest, inference) = try construction?.loadManifestConfig()
            ?? loadInferenceConfig(packagePath: packagePath)
        let useModelGenerate: Bool
        let mode: String
        if let construction {
            let runtimeMode = try construction.runtimeModeDescription(manifest: manifest)
            useModelGenerate = runtimeMode.useModelGenerate
            mode = runtimeMode.mode
        } else {
            (useModelGenerate, mode) = shouldUseModelGenerate(
                packagePath: packagePath,
                manifest: manifest
            )
        }

        // Start the grammar build before model init so the two overlap.
        var grammarBox: GrammarBox?
        let compiledGrammar = try SmeltCompiledGrammar.load(packagePath: packagePath)
        if compiledGrammar == nil, !grammarBindings.isEmpty {
            throw NSError(
                domain: "SmeltCLI",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "--bind given but the package has no compiled grammar"
                ]
            )
        }
        if let compiledGrammar {
            if useModelGenerate {
                // Splice runtime bindings into the schema's "$bind:NAME"
                // slots before the matcher build. Fails closed: unbound
                // slots and unknown binding names are errors.
                let jsonSchema = try SmeltGrammarBinding.apply(
                    bindings: grammarBindings, toJSONSchema: compiledGrammar.jsonSchema
                )
                grammarBox = GrammarBox(
                    tokenizer: tokenizer,
                    eosTokens: inference.eosTokens,
                    jsonSchema: jsonSchema,
                    serializedTrie: compiledGrammar.serializedTrie
                )
            } else if !grammarBindings.isEmpty {
                throw NSError(
                    domain: "SmeltCLI",
                    code: 1,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "--bind given but the package uses the decode-only "
                            + "path, which cannot constrain output"
                    ]
                )
            } else {
                fputs(
                    "warning: compiled grammar ignored — package uses the "
                        + "decode-only path\n",
                    stderr
                )
            }
        }

        let model: SmeltModel?
        if useModelGenerate {
            let modelStart = CFAbsoluteTimeGetCurrent()
            model = try construction?.makeModel(
                contextLimit: contextLimit,
                manifest: manifest
            ) ?? SmeltModel(
                package: packagePath,
                contextLimit: contextLimit,
                manifest: manifest
            )
            stamp("SmeltModel init (total)", since: modelStart)
        } else {
            model = nil
        }

        return RunContext(
            packagePath: packagePath,
            tokenizer: tokenizer,
            inference: inference,
            chatTemplate: inference.chatTemplate,
            modelName: manifest.modelName,
            mode: mode,
            contextLimit: contextLimit,
            construction: construction,
            model: model,
            grammarBox: grammarBox
        )
    }

    func evaluate(
        prompt: String,
        maxTokens: Int,
        template: String,
        selectionMode: SmeltSelectionMode,
        selectionDescription: String,
        systemPrompt: String
    ) throws -> PromptRunResult {
        let resolvedTemplate = try construction?.resolveTemplate(cliOverride: template)
            ?? resolveChatTemplate(
                cliOverride: template, packageTemplate: chatTemplate)
        let thinkingPolicy = resolvedThinkingPolicy(inference)
        let renderedInput: (prompt: String, systemPrompt: String)
        if let construction {
            renderedInput = try construction.renderPrompt(
                prompt: prompt,
                systemPrompt: systemPrompt
            )
        } else {
            renderedInput = (prompt, systemPrompt)
        }
        var inputIds = try buildInputIds(
            prompt: renderedInput.prompt, tokenizer: tokenizer, template: resolvedTemplate,
            thinkingPolicy: thinkingPolicy)
        if !renderedInput.systemPrompt.isEmpty {
            let systemIds = try buildSystemIds(
                systemPrompt: renderedInput.systemPrompt,
                tokenizer: tokenizer,
                template: resolvedTemplate
            )
            inputIds = systemIds + inputIds
        } else if let prepared = model?.preparedPrefixTokenIds {
            // A prepared prompt prefix reuses its compiled prefill snapshot
            // whenever the request begins with the same exact token IDs.
            // --system/--system-file overrides it; SMELT_NO_PREPARED_PREFIX=1
            // disables it.
            inputIds = buildInputIdsApplyingPreparedPrefix(
                prompt: prompt,
                tokenizer: tokenizer,
                fullInputIds: inputIds,
                preparedPrefixTokenIds: prepared,
                continuation: model?.preparedPrefixContinuation,
                template: resolvedTemplate,
                thinkingPolicy: thinkingPolicy
            )
        }

        let generated: [Int32]
        let prefillElapsed: Double
        let genElapsed: Double

        if let model {
            guard inputIds.count < model.maxContextTokens else {
                throw SmeltRuntimeError.inputExceedsContext(
                    limit: model.maxContextTokens,
                    requested: inputIds.count
                )
            }
            // Compiled grammar: a fresh matcher clone masks each decode step;
            // generation stops as soon as the grammar reaches an accepting
            // state (the output is structurally complete). decodeStep
            // evaluates the mask closure between commit and GPU wait, so
            // the mask compute is hidden under the forward pass (consume
            // itself is a few microseconds and stays on the caller).
            let matcher = try grammarBox?.freshMatcher()
            var seen = 0
            let result = try model.generate(
                tokenIds: inputIds,
                selectionMode: selectionMode,
                allowedTokenMask: matcher.map { m in { try m.computeMask() } }
            ) { token in
                seen += 1
                if let matcher {
                    guard (try? matcher.consume(tokenIds: [token.id])) != nil
                    else { return false }
                    if matcher.isAccepting { return false }
                }
                return seen < (construction?.effectiveMaxTokens(maxTokens) ?? maxTokens)
            }
            generated = result.tokens
            prefillElapsed = result.prefillTime
            genElapsed = result.generateTime
        } else {
            let result = try runPromptDecodeOnly(
                packagePath: packagePath,
                inputIds: inputIds,
                inference: inference,
                contextLimit: contextLimit,
                maxTokens: maxTokens,
                selectionMode: selectionMode,
                construction: construction
            )
            generated = result.tokens
            prefillElapsed = result.prefillTime
            genElapsed = result.generateTime
        }

        let completion = tokenizer.decode(generated)
        return PromptRunResult(
            prompt: renderedInput.prompt,
            promptTokens: inputIds.count,
            mode: mode,
            selection: selectionDescription,
            generated: generated,
            completion: completion,
            prefillElapsed: prefillElapsed,
            genElapsed: genElapsed
        )
    }
}

func evaluatePrompt(
    packagePath: String,
    prompt: String,
    maxTokens: Int,
    template: String,
    selectionMode: SmeltSelectionMode = .argmax,
    selectionDescription: String = "argmax",
    contextLimit: Int? = nil,
    systemPrompt: String = "",
    grammarBindings: [String: [String]] = [:],
    construction: CAMTextRuntimeConstruction? = nil
) throws -> PromptRunResult {
    try RunContext.load(
        packagePath: packagePath,
        contextLimit: contextLimit,
        grammarBindings: grammarBindings,
        construction: construction
    )
        .evaluate(
            prompt: prompt,
            maxTokens: maxTokens,
            template: template,
            selectionMode: selectionMode,
            selectionDescription: selectionDescription,
            systemPrompt: systemPrompt
        )
}
