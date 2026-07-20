import Foundation
import CryptoKit
import SmeltRuntime
import SmeltSchema

// Text generation handler shared by executable and embedded consumers.

public final class SmeltServeHandler: @unchecked Sendable {
    private let runtime: SmeltRuntime
    private let tokenizer: SmeltTokenizer
    private let inference: SmeltInferenceManifest
    private let modelId: String
    private let packageIdentity: String
    private let template: String
    private let toolTranscriptCodec: SmeltNativeToolTranscriptCodec?
    private let bakedPromptPrefix: SmeltBakedPromptPrefix?
    private let preparedPromptStates: [SmeltPreparedPromptState]
    private let declaredInterface: SmeltPackageInterface?
    private let declaredDefaults: [String: SmeltPackageArgumentValue]
    private let packageGrammarPrototype: SmeltLLGuidanceMatcher?
    private let prefixCache: SmeltPromptStateCache?
    private let sessionStore: SmeltSessionRegistry?
    /// Prompt-format-owned special tokens that carry visible semantic bytes.
    /// The ordinary decoder suppresses every special token; native codecs opt
    /// back in only to their own envelope tokens.
    private let visibleGeneratedControlTokenText: [Int32: String]
    /// Some instruct models' native tool-call start token. Looked up by
    /// content name so other tokenizers leave this nil and the
    /// native-shape decoder is never triggered. Used to detect when
    /// the model emits its trained `<|tool_call>call:NAME{args}
    /// <tool_call|>` shape; the start token is `special: true` and
    /// so gets stripped from the decoded text, leaving the `call:`
    /// body as the only textual signal — token-ID detection is the
    /// only reliable signal.
    private let nativeToolCallStartTokenId: Int32?
    // Built once on first tool-bearing request and reused; the
    // llguidance vocabulary copy is multi-MB for large tokenizers,
    // and the tokenizer + eosTokens are immutable for the handler's
    // lifetime. The serve run loop is serial (StdioTransport.swift),
    // so plain var is safe.
    private var cachedLLGTokenizer: SmeltLLGuidanceTokenizer?

    // Small cache of pristine (never-consumed) tool matchers keyed on the
    // generated grammar. Keeping auto, required, and specific variants avoids
    // recompiling when a client changes policy between turns.
    // Lock-free for the same reason as cachedLLGTokenizer: the serve
    // loop is serial.
    private var cachedToolMatchers: [String: SmeltLLGuidanceMatcher] = [:]

    /// Thought-skip is suppressed under any constrained case to keep
    /// the matcher's state consistent with the emitted tokens. The
    /// strict case auto-terminates on matcher.isAccepting (the JSON
    /// object closed); the union case does not, because text responses
    /// remain open-ended until the model chooses EOS.
    fileprivate enum DecodeMode {
        case freeText(skipThought: Bool)
        case constrained(SmeltLLGuidanceMatcher)
        case constrainedUnion(SmeltLLGuidanceMatcher)

        var skipThought: Bool {
            if case .freeText(let s) = self { return s }
            return false
        }

        var matcher: SmeltLLGuidanceMatcher? {
            switch self {
            case .constrained(let m), .constrainedUnion(let m): return m
            case .freeText: return nil
            }
        }

        var autoTerminatesOnAccept: Bool {
            switch self {
            case .constrained: return true
            case .constrainedUnion, .freeText: return false
            }
        }

        var allowedTokenMaskCallback: (() throws -> [UInt32])? {
            guard let matcher else { return nil }
            return { try matcher.computeMask() }
        }
    }

    /// Mask-level "min new tokens" floor: clears EOS bits from the
    /// allowed-token mask until the decoded output has at least N
    /// non-whitespace characters. Applies in both constrained-union
    /// mode (matcher's mask, matcher's EOS IDs) and free-text mode
    /// (synthetic full mask, inference EOS IDs). Prevents an instruct model
    /// degenerating to a single-letter response + EOS — a known
    /// failure pattern under tool_choice:"auto" union grammar AND
    /// on the post-tool free-text turn after `lastNonSystemRoleIsTool`
    /// drops tools.
    ///
    /// This is the standard min-new-tokens / MinLengthLogitsProcessor
    /// pattern (HF Transformers, vLLM, TGI), applied here at the
    /// llguidance mask boundary so it composes cleanly with the
    /// matcher's existing constraints.
    /// Min-tokens EOS floor for FREE-TEXT mode only. Under a tool
    /// matcher (strict or union), the floor lives in the grammar —
    /// `larkUnion` encodes a 3-non-whitespace-char minimum in
    /// text_response, and strict mode auto-terminates on JSON close
    /// (the floor doesn't apply there). The mask-based gate would
    /// otherwise compose with the matcher's mask in ways that can
    /// collide (e.g. the matcher's only allowed tokens overlapping
    /// the cleared EOS set produces an all-zero mask the sampler
    /// can't pick from).
    private final class MinTokensEOSGate {
        private static let minimumSubstantiveCharacters = 3

        private let runtimeEOSTokens: [Int32]
        private let vocabSize: Int
        private var substantiveCharacters = 0

        init(
            runtimeEOSTokens: [Int32],
            vocabSize: Int
        ) {
            self.runtimeEOSTokens = runtimeEOSTokens
            self.vocabSize = vocabSize
        }

        var isActive: Bool {
            substantiveCharacters < Self.minimumSubstantiveCharacters
        }

        func computeMask() throws -> [UInt32]? {
            guard isActive else { return nil }
            let words = (vocabSize + 31) / 32
            var mask = Array(repeating: UInt32.max, count: words)
            let tail = vocabSize % 32
            if tail != 0 {
                mask[words - 1] = (UInt32(1) << tail) - 1
            }
            for token in runtimeEOSTokens where token >= 0 {
                Self.clear(token: UInt32(token), in: &mask)
            }
            return mask
        }

        func observe(decodedText: String) {
            guard isActive else { return }
            for character in decodedText where !character.isWhitespace {
                substantiveCharacters += 1
                if substantiveCharacters >= Self.minimumSubstantiveCharacters {
                    break
                }
            }
        }

        private static func clear(token: UInt32, in mask: inout [UInt32]) {
            let wordIndex = Int(token / 32)
            guard wordIndex < mask.count else { return }
            let bit = token % 32
            mask[wordIndex] &= ~(UInt32(1) << bit)
        }
    }

    public init(
        packagePath: String,
        runtime: SmeltRuntime,
        tokenizer: SmeltTokenizer,
        inference: SmeltInferenceManifest,
        modelId: String,
        template: String
    ) throws {
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.inference = inference
        self.modelId = modelId
        self.packageIdentity = try SmeltPackageIdentity.compute(packagePath: packagePath)
        self.template = template
        self.toolTranscriptCodec = try SmeltNativeToolTranscriptCodec.resolve(
            inference.toolTranscriptCodec
        )
        self.bakedPromptPrefix = try SmeltBakedPromptPrefix.load(
            packagePath: packagePath
        )
        var prepared = try SmeltPreparedPromptSet.load(
            packagePath: packagePath
        )?.states ?? []
        if let bakedPromptPrefix,
           !prepared.contains(where: { $0.id == "run/default" }) {
            prepared.append(SmeltPreparedPromptState(
                id: "run/default",
                tokenIds: bakedPromptPrefix.tokenIds,
                snapshot: bakedPromptPrefix.snapshot
            ))
        }
        self.preparedPromptStates = prepared
        let declaredInterface = try SmeltPackageInterface.load(packagePath: packagePath)
        self.declaredInterface = declaredInterface
        let declaredDefaults = try declaredInterface?.resolve(
            declaredRaw: [:]
        ) ?? [:]
        self.declaredDefaults = declaredDefaults
        self.packageGrammarPrototype = try Self.loadPackageGrammar(
            packagePath: packagePath,
            tokenizer: tokenizer,
            inference: inference,
            interface: declaredInterface,
            resolved: declaredDefaults,
            explicitBindings: [:]
        )
        self.prefixCache = Self.makePrefixCache(runtime: runtime)
        self.sessionStore = Self.makeSessionStore()
        if let toolTranscriptCodec {
            self.visibleGeneratedControlTokenText = Dictionary(
                uniqueKeysWithValues: toolTranscriptCodec
                    .visibleGeneratedControlTokens.compactMap { literal in
                        tokenizer.addedTokenId(for: literal).map {
                            (Int32($0), literal)
                        }
                    }
            )
        } else {
            self.visibleGeneratedControlTokenText = [:]
        }
        self.nativeToolCallStartTokenId = tokenizer
            .addedTokenId(for: "<|tool_call>")
            .map { Int32($0) }
    }

    private static func loadPackageGrammar(
        packagePath: String,
        tokenizer: SmeltTokenizer,
        inference: SmeltInferenceManifest,
        interface: SmeltPackageInterface?,
        resolved: [String: SmeltPackageArgumentValue],
        explicitBindings: [String: [String]]
    ) throws -> SmeltLLGuidanceMatcher? {
        let ignored = SmeltBakeManifest.ignoredFromEnv()
        guard !ignored.contains(.grammar) else { return nil }
        let marker = try SmeltBakeManifest.load(packagePath: packagePath)
        let grammar = marker?.declares(.grammar) == true
            ? try SmeltBakedGrammar.loadStrict(packagePath: packagePath)
            : SmeltBakedGrammar.load(packagePath: packagePath)
        guard let grammar else { return nil }

        var bindings: [String: [String]] = [:]
        for arg in interface?.args ?? [] {
            guard let target = arg.target,
                  target.hasPrefix(SmeltPackageInterface.bindTargetPrefix),
                  case .list(let values)? = resolved[arg.flag]
            else { continue }
            bindings[String(target.dropFirst(SmeltPackageInterface.bindTargetPrefix.count))] = values
        }
        for (name, values) in explicitBindings {
            bindings[name] = values
        }
        let schema = try SmeltGrammarBinding.apply(
            bindings: bindings,
            toJSONSchema: grammar.jsonSchema
        )
        let llgTokenizer: SmeltLLGuidanceTokenizer
        if let trie = grammar.serializedTrie,
           let restored = try? SmeltLLGuidanceTokenizer(
                tokenizer: tokenizer,
                serializedTrie: trie
           ) {
            llgTokenizer = restored
        } else {
            llgTokenizer = try SmeltLLGuidanceTokenizer(
                tokenizer: tokenizer,
                eosTokens: inference.eosTokens
            )
        }
        return try SmeltLLGuidanceMatcher(
            tokenizer: llgTokenizer,
            jsonSchema: schema
        )
    }

    private func applyingDeclaredPrompt(
        to messages: [OpenAIChatMessage]
    ) throws -> [OpenAIChatMessage] {
        guard let declaredInterface else { return messages }
        return try messages.map { message in
            guard message.role == .user else { return message }
            return OpenAIChatMessage(
                role: message.role,
                content: try declaredInterface.fillPrompt(
                    resolved: declaredDefaults,
                    input: message.content ?? ""
                ),
                name: message.name,
                toolCallId: message.toolCallId,
                toolCalls: message.toolCalls
            )
        }
    }

    /// Configure the in-memory session store for the `session_id`
    /// chat-completions extension. Returns nil when
    /// `SMELT_SERVE_SESSIONS=0` (disables the extension; the server
    /// then 404s any `session_id` request and `create_session: true`
    /// becomes a no-op). Defaults are sized for single-machine
    /// single-user (a handful of concurrent Pi tabs).
    private static func makeSessionStore() -> SmeltSessionRegistry? {
        let env = ProcessInfo.processInfo.environment
        if env["SMELT_SERVE_SESSIONS"] == "0" { return nil }
        let maxSessions = env["SMELT_SERVE_SESSIONS_MAX"]
            .flatMap(Int.init) ?? 16
        let idleSecs = env["SMELT_SERVE_SESSIONS_IDLE_SECS"]
            .flatMap(TimeInterval.init) ?? 1800
        return SmeltSessionRegistry(
            maxSessions: maxSessions,
            idleTimeoutSeconds: idleSecs
        )
    }

    /// Configure the LCP prefix cache from environment variables. Returns
    /// nil when `SMELT_SERVE_PREFIX_CACHE_MB=0` (cache disabled) or when a
    /// dense-attention runtime lacks the all-logits suffix-prefill primitive.
    /// Recurrent runtimes use exact-position checkpoints and sequentially
    /// evaluate only the appended suffix.
    private static func makePrefixCache(runtime: SmeltRuntime) -> SmeltPromptStateCache? {
        let env = ProcessInfo.processInfo.environment
        let mb = env["SMELT_SERVE_PREFIX_CACHE_MB"].flatMap(Int.init) ?? 2048
        guard mb > 0 else { return nil }
        guard runtime.promptStateRequiresExactPositionRestore
                || runtime.supportsChunkedPrefillVerify
        else { return nil }
        let minMatch = env["SMELT_SERVE_PREFIX_CACHE_MIN_MATCH"]
            .flatMap(Int.init) ?? 32
        let tailFresh = env["SMELT_SERVE_PREFIX_CACHE_TAIL_FRESH"]
            .flatMap(Int.init) ?? 32
        return SmeltPromptStateCache(
            maxBytes: mb * 1024 * 1024,
            minMatchTokens: minMatch,
            tailFreshTokens: tailFresh,
            requiresExactRestore: runtime.promptStateRequiresExactPositionRestore
        )
    }

    private func preparedPromptMatch(
        inputIds: [Int32],
        contract: String? = nil
    ) -> SmeltPreparedPromptState? {
        SmeltPreparedPromptSet(states: preparedPromptStates).longestMatch(
            tokenIds: inputIds,
            contract: contract
        )
    }

    /// Result of handling a request. `.complete` returns a buffered
    /// SmeltServeRawResponse that the serve loop writes via
    /// transport.write(); `.streamed` indicates the handler already
    /// emitted chunks via the transport's stream handle and ended
    /// the stream itself (the serve loop should skip transport.write).
    public enum HandlerResult: Sendable {
        case complete(SmeltServeRawResponse)
        case streamed
    }

    public func handle(
        _ raw: SmeltServeRawRequest,
        transport: any SmeltServeTransport
    ) async -> HandlerResult {
        switch (raw.method, raw.path) {
        case (.post, .chatCompletions):
            return await handleChatCompletions(
                raw.body, requestId: raw.id, transport: transport
            )
        case (.post, .completions):
            return await handleCompletions(
                raw.body, requestId: raw.id, transport: transport
            )
        case (.get, .models):
            return .complete(handleModels())
        case (.get, .chatCompletions), (.get, .completions),
             (.post, .models):
            return .complete(OpenAIJSON.errorResponse(
                status: 405, code: .methodNotAllowed,
                message: "Method not allowed for \(raw.path.rawValue)"
            ))
        case (_, .audioSpeech), (_, .audioVoices):
            return .complete(OpenAIJSON.errorResponse(
                status: 400, code: .invalidRequest,
                message: "\(raw.path.rawValue) requires a text-to-PCM package; this server is "
                    + "serving the text generation package '\(modelId)'"
            ))
        }
    }

    private func handleModels() -> SmeltServeRawResponse {
        let now = Int(Date().timeIntervalSince1970)
        let response = OpenAIModelsResponse(
            object: "list",
            data: [OpenAIModelEntry(
                id: modelId,
                object: "model",
                created: now,
                ownedBy: "smelt"
            )]
        )
        let body = try! OpenAIJSON.encode(response)
        return SmeltServeRawResponse(
            statusCode: 200,
            headers: ["X-Smelt-Package-Identity": packageIdentity],
            body: body
        )
    }

    private func handleChatCompletions(
        _ body: Data,
        requestId: SmeltServeRequestId,
        transport: any SmeltServeTransport
    ) async -> HandlerResult {
        let request: OpenAIChatCompletionsRequest
        do {
            request = try OpenAIJSON.decode(OpenAIChatCompletionsRequest.self, from: body)
        } catch {
            return .complete(OpenAIJSON.errorResponse(
                status: 400, code: .invalidRequest,
                message: "Failed to decode request: \(error)"
            ))
        }
        guard !request.messages.isEmpty else {
            return .complete(OpenAIJSON.errorResponse(
                status: 400, code: .invalidRequest,
                message: "messages must contain at least one entry"
            ))
        }
        if let reason = invalidSamplingParameters(
            temperature: request.temperature,
            topK: request.topK,
            topP: request.topP
        ) {
            return .complete(OpenAIJSON.errorResponse(
                status: 400, code: .invalidRequest, message: reason
            ))
        }
        if let unsupported = request.messages
            .flatMap({ $0.contentParts ?? [] })
            .first(where: {
                if case .unsupported = $0 { return true }
                return false
            })
        {
            return .complete(OpenAIJSON.errorResponse(
                status: 400, code: .invalidRequest,
                message: "Unsupported chat content part type '\(unsupported.typeName)'"
            ))
        }
        if request.messages
            .flatMap({ $0.contentParts ?? [] })
            .contains(where: {
                if case .imageURL = $0 { return true }
                return false
            })
        {
            return .complete(OpenAIJSON.errorResponse(
                status: 400, code: .invalidRequest,
                message: "image_url content requires a CAM multimodal serve executor"
            ))
        }

        let sessionIntent: SessionIntent
        switch resolveSessionIntent(request: request) {
        case .ok(let intent):
            sessionIntent = intent
        case .error(let response):
            return .complete(response)
        }

        let packageMessages: [OpenAIChatMessage]
        do {
            packageMessages = try applyingDeclaredPrompt(to: request.messages)
        } catch {
            return .complete(OpenAIJSON.errorResponse(
                status: 400,
                code: .invalidRequest,
                message: "Package interface could not resolve: \(error)"
            ))
        }

        var toolDescriptors: [SmeltToolDescriptor]?
        do {
            toolDescriptors = try extractActiveToolDescriptors(
                request: request,
                rawBody: body
            )
        } catch let error as ToolWiringError {
            return .complete(error.response)
        } catch {
            return .complete(OpenAIJSON.errorResponse(
                status: 400, code: .invalidRequest,
                message: "Tool validation failed: \(error)"
            ))
        }

        let useUnion = Self.toolMatcherUsesUnion(choice: request.toolChoice)
        let usesPackageNativeTools = toolTranscriptCodec != nil
        // Prompt formats without a package-native tool transcript cannot
        // reliably disambiguate the union grammar after a tool result. For
        // those formats only, default `tool_choice:"auto"` follow-up turns to
        // no matcher/tools prompt. Native transcript capabilities retain the
        // descriptors and support exact multi-step chains.
        //
        // The repeated-answer predicate AND the deterministic
        // replay below both read the same answer text — compute it
        // once and reuse the Optional<String> so the O(N) message
        // walk doesn't run twice per request.
        let cachedRepeatAnswer = repeatedAnsweredQuestionAnswer(
            packageMessages
        )
        if !usesPackageNativeTools && useUnion && toolDescriptors != nil
            && (lastNonSystemRoleIsTool(packageMessages)
                || cachedRepeatAnswer != nil)
        {
            toolDescriptors = nil
        }
        let effectiveMessages: [OpenAIChatMessage]
        if let descriptors = toolDescriptors, !usesPackageNativeTools {
            // Place the tools system message AFTER the caller's
            // existing system messages so the user's instructions are
            // anchored first. This prose mandate keeps OpenAI-style
            // auto tool prompts reliable under the existing JSON/union
            // matcher.
            let toolsMessage = OpenAIChatMessage(
                role: .system,
                content: renderToolsSystemMessage(
                    descriptors: descriptors,
                    choice: request.toolChoice,
                    useUnion: useUnion
                )
            )
            var merged: [OpenAIChatMessage] = []
            var inserted = false
            for message in packageMessages {
                merged.append(message)
                if !inserted && message.role == .system {
                    inserted = true
                    merged.append(toolsMessage)
                }
            }
            if !inserted {
                merged.insert(toolsMessage, at: 0)
            }
            effectiveMessages = merged
        } else {
            effectiveMessages = packageMessages
        }

        let inputIds: [Int32]
        do {
            inputIds = try buildChatCompletionsInputIdsApplyingBakedPrefix(
                messages: effectiveMessages,
                tokenizer: tokenizer,
                template: template,
                thinkingPolicy: resolvedThinkingPolicy(inference),
                bakedPrefix: bakedPromptPrefix,
                tools: usesPackageNativeTools ? toolDescriptors : nil,
                toolTranscriptCodec: toolTranscriptCodec
            )
        } catch {
            return .complete(OpenAIJSON.errorResponse(
                status: 400, code: .invalidRequest,
                message: "Failed to apply chat template: \(error)"
            ))
        }
        let preparedPrompt = preparedPromptMatch(
            inputIds: inputIds,
            contract: request.promptContract ?? "run/default"
        )
        let promptDiagnostics = ProcessInfo.processInfo.environment[
            "SMELT_SERVE_PROMPT_DIAGNOSTICS"
        ]
        if promptDiagnostics == "1" || promptDiagnostics == "ids" {
            var bytes = Data(capacity: inputIds.count * MemoryLayout<Int32>.size)
            for token in inputIds {
                var littleEndian = token.littleEndian
                withUnsafeBytes(of: &littleEndian) { bytes.append(contentsOf: $0) }
            }
            let digest = SHA256.hash(data: bytes).map {
                String(format: "%02x", $0)
            }.joined()
            fputs(
                "smelt serve prompt tokens=\(inputIds.count) int32le_sha256=\(digest)\n",
                stderr
            )
            if promptDiagnostics == "ids" {
                fputs(
                    "smelt serve prompt input_ids="
                        + inputIds.map(String.init).joined(separator: ",")
                        + "\n",
                    stderr
                )
            }
        }

        // Only reserve the think-channel budget when the skip will actually run
        // (thinking disabled); under .enabled no tokens are hidden-injected.
        let thoughtReserve =
            (inference.thinkToken != nil && inference.thinkingPolicy != .enabled) ? 2 : 0
        if inputIds.count + thoughtReserve >= runtime.maxContextTokens {
            return .complete(OpenAIJSON.errorResponse(
                status: 400, code: .contextLengthExceeded,
                message: "prompt is \(inputIds.count) tokens"
                    + (thoughtReserve > 0 ? " plus \(thoughtReserve)-token think-channel reserve" : "")
                    + "; context limit is \(runtime.maxContextTokens)"
            ))
        }

        let requestedMax = request.effectiveMaxTokens ?? inference.maxTokens
        let maxTokens = max(0, min(
            requestedMax,
            inference.maxTokens,
            runtime.maxContextTokens - inputIds.count - thoughtReserve
        ))

        // Deterministic answer paths bypass the model, so they can't
        // produce real per-token logprobs. When the caller asked for
        // logprobs, fall back to model generation to honor the
        // contract.
        let deterministicAnswer: String? =
            (useUnion && requestInvolvesTools(request)
                && request.logprobs != true)
                ? cachedRepeatAnswer
                : nil

        let selectionMode = resolveSelectionMode(
            temperature: request.temperature
                ?? preparedPrompt?.sampling?.temperature,
            seed: request.seed,
            topK: request.topK ?? preparedPrompt?.sampling?.topK,
            topP: request.topP ?? preparedPrompt?.sampling?.topP
        )

        // OpenAI: logprobs=true means include per-token logprobs;
        // top_logprobs (0..20) controls how many alternatives per token.
        let logprobsTopK: Int? = (request.logprobs == true)
            ? max(0, min(20, request.topLogprobs ?? 0))
            : nil

        let toolMatcher: SmeltLLGuidanceMatcher?
        do {
            toolMatcher = try toolDescriptors.map { descriptors in
                try buildToolMatcher(descriptors: descriptors, useUnion: useUnion)
            }
        } catch let error as ToolWiringError {
            return .complete(error.response)
        } catch {
            return .complete(OpenAIJSON.errorResponse(
                status: 500, code: .internalError,
                message: "Failed to build tool grammar matcher: \(error)"
            ))
        }
        if toolMatcher != nil && packageGrammarPrototype != nil {
            return .complete(OpenAIJSON.errorResponse(
                status: 400,
                code: .invalidRequest,
                message: "Request tools cannot replace this package's output schema"
            ))
        }
        let agentMatcher: SmeltLLGuidanceMatcher?
        do {
            agentMatcher = try toolMatcher == nil
                ? packageGrammarPrototype?.freshCopy()
                : nil
        } catch {
            return .complete(OpenAIJSON.errorResponse(
                status: 500,
                code: .internalError,
                message: "Failed to load the package output schema: \(error)"
            ))
        }

        // The old partial-LCP path bypasses tool transcripts because a
        // divergent suffix once exposed unsafe state trimming. Exact-position
        // recurrent checkpoints are safe for arbitrary transcript content:
        // tool schemas, calls, and responses are all token-visible identity.
        let skipPrefixCache = requestInvolvesTools(request)
            && prefixCache?.requiresExactRestore != true

        if let deterministicAnswer {
            let completionTokens = tokenizer.encode(deterministicAnswer).count
            let sessionId = commitSessionIntent(sessionIntent)
            // X-Smelt-Replay-Hit signals that the response was served
            // from the verbatim-resume cache rather than generated by
            // the model. Callers who want fresh generation (e.g. the
            // tool result diverged since the prior answer was cached)
            // can detect this and re-issue with a perturbed prompt or
            // `logprobs:true` (which already bypasses replay).
            var headers: [String: String] = ["X-Smelt-Replay-Hit": "1"]
            if let sessionId {
                headers["X-Smelt-Session-Id"] = sessionId
            }
            if request.stream == true {
                return await runDeterministicAnswerStream(
                    content: deterministicAnswer,
                    promptTokens: inputIds.count,
                    completionTokens: completionTokens,
                    includeUsage: request.streamOptions?.includeUsage ?? false,
                    extraHeaders: headers,
                    requestId: requestId,
                    transport: transport
                )
            }
            let response = OpenAIChatCompletionsResponse(
                id: OpenAIJSON.chatCompletionId(),
                object: "chat.completion",
                created: Int(Date().timeIntervalSince1970),
                model: modelId,
                choices: [OpenAIChoice(
                    index: 0,
                    message: OpenAIChatMessage(
                        role: .assistant,
                        content: deterministicAnswer
                    ),
                    finishReason: .stop,
                    logprobs: nil
                )],
                usage: OpenAIUsage(
                    promptTokens: inputIds.count,
                    completionTokens: completionTokens,
                    totalTokens: inputIds.count + completionTokens
                )
            )
            let body: Data
            do {
                body = try OpenAIJSON.encode(response)
            } catch {
                return .complete(OpenAIJSON.errorResponse(
                    status: 500, code: .internalError,
                    message: "Response encode failed: \(error)"
                ))
            }
            return .complete(SmeltServeRawResponse(
                statusCode: 200, headers: headers, body: body
            ))
        }

        if request.stream == true {
            // Allocate the session id BEFORE beginStream so it can ship
            // in the response headers (chunked-encoding sends headers
            // upfront, before the body). Once allocated we commit it
            // to the store immediately — there's no after-completion
            // hook on the streaming path that doesn't race with errors.
            let streamSessionId = commitSessionIntent(sessionIntent)
            let streamHeaders = streamSessionId
                .map { ["X-Smelt-Session-Id": $0] } ?? [:]
            let includeUsage = request.streamOptions?.includeUsage ?? false
            if let toolMatcher {
                // Stop sequences are suppressed under any active tool
                // matcher (strict OR union). The stop scanner operates
                // on decoded text in runGenerationStreamingCore — it
                // can't tell whether the model committed to the JSON
                // arm or the text arm, so a caller-supplied stop like
                // `path` will truncate inside JSON tool-call args and
                // produce a malformed envelope that fails decode. The
                // trade-off is that text-arm responses under
                // tool_choice:"auto" no longer honor `stop`; the
                // natural EOS (free-text end-of-turn) terminates them.
                return await runStreamingChatTools(
                    inputIds: inputIds,
                    preparedPrompt: preparedPrompt,
                    maxTokens: maxTokens,
                    selectionMode: selectionMode,
                    toolMatcher: toolMatcher,
                    useUnion: useUnion,
                    toolDescriptors: toolDescriptors ?? [],
                    logprobsTopK: logprobsTopK,
                    stopSequences: [],
                    includeUsage: includeUsage,
                    skipPrefixCache: skipPrefixCache,
                    extraHeaders: streamHeaders,
                    requestId: requestId,
                    transport: transport
                )
            }
            return await runStreamingChat(
                inputIds: inputIds,
                preparedPrompt: preparedPrompt,
                maxTokens: maxTokens,
                selectionMode: selectionMode,
                logprobsTopK: logprobsTopK,
                stopSequences: agentMatcher == nil
                    ? (request.stop?.sequences ?? []) : [],
                agentMatcher: agentMatcher,
                includeUsage: includeUsage,
                skipPrefixCache: skipPrefixCache,
                extraHeaders: streamHeaders,
                requestId: requestId,
                transport: transport
            )
        }

        let mode: DecodeMode
        if let matcher = toolMatcher {
            mode = useUnion ? .constrainedUnion(matcher) : .constrained(matcher)
        } else if let agentMatcher {
            mode = .constrained(agentMatcher)
        } else {
            mode = .freeText(skipThought: inference.thinkingPolicy != .enabled)
        }
        let result: ChatGenerationResult
        do {
            // Suppress stop sequences whenever a tool matcher is set
            // (strict OR union). The stop scanner in
            // runGenerationStreamingCore operates on decoded text
            // and can't distinguish JSON-arm bytes from text-arm
            // bytes — a caller stop like "path" would truncate
            // inside a JSON tool-call's args and the partial JSON
            // then fails decode. Text-arm union responses no longer
            // honor `stop`; natural EOS terminates them.
            result = try await runChatGeneration(
                inputIds: inputIds,
                preparedPrompt: preparedPrompt,
                maxTokens: maxTokens,
                selectionMode: selectionMode,
                logprobsTopK: logprobsTopK,
                stopSequences: toolMatcher != nil || agentMatcher != nil
                    ? [] : (request.stop?.sequences ?? []),
                mode: mode,
                skipPrefixCache: skipPrefixCache
            )
        } catch {
            return .complete(OpenAIJSON.errorResponse(
                status: 500, code: .internalError,
                message: "Generation failed: \(error)"
            ))
        }

        let chatLogprobs: OpenAILogprobs? = result.logprobs.map { entries in
            OpenAILogprobs(content: entries.map {
                LogprobsCompute.chatTokenLogprob($0, tokenizer: tokenizer)
            })
        }

        // Four terminal shapes:
        // 1. tool-call (JSON): strict matcher accepted OR union mode +
        //    first non-whitespace char is `{`. Decode the JSON
        //    `{"name":..,"arguments":..}` shape; failure =
        //    tool_call_failure envelope.
        // 2. tool-call (native): union mode + output starts with the
        //    native tool-call prefix `<|tool_call>` (the
        //    trained format leaks through the text arm).
        // 3. text: union mode + first significant char is neither `{`
        //    nor a native prefix, OR no matcher.
        // 4. strict-no-accept error: matcher set but never accepted.
        //
        // The first-significant-char (not literally first-char) check
        // matters because the union lark grammar's JSON arm allows the
        // leading whitespace JSON syntax itself permits, so the model
        // can legitimately emit e.g. `\n{"name":...}`. The text arm
        // requires its first non-whitespace to be non-`{`, so leading
        // whitespace alone never commits either arm.
        let shape: BufferedShape
        if let toolTranscriptCodec, toolMatcher != nil {
            if toolTranscriptCodec.containsToolCall(in: result.text) {
                shape = .native
            } else if useUnion {
                shape = .text
            } else {
                shape = .strictNoAccept
            }
        } else {
            shape = classifyBufferedShape(
                text: result.text,
                tokens: result.tokens,
                nativeToolCallStartTokenId: nativeToolCallStartTokenId,
                finishReason: result.finishReason,
                toolMatcherSet: toolMatcher != nil,
                useUnion: useUnion
            )
        }
        let message: OpenAIChatMessage
        var finishReason = result.finishReason
        switch shape {
        case .json:
            do {
                let call = try decodeGeneratedToolCall(result.text)
                message = OpenAIChatMessage(
                    role: .assistant, content: nil, toolCalls: [call]
                )
                finishReason = .toolCalls
            } catch {
                return .complete(OpenAIJSON.toolCallFailureResponse(
                    message: "Tool-call JSON did not decode: \(error) "
                        + "(finish_reason=\(result.finishReason.rawValue))",
                    partialJson: result.text,
                    generatedTokenIds: result.tokens
                ))
            }
        case .native:
            do {
                if toolTranscriptCodec != nil {
                    let decoded = try decodeNativeTranscriptToolCalls(
                        result.text,
                        descriptors: toolDescriptors ?? []
                    )
                    message = OpenAIChatMessage(
                        role: .assistant,
                        content: decoded.content,
                        toolCalls: decoded.calls
                    )
                } else {
                    let call = try decodeNativeToolCall(
                        result.text,
                        allowedToolNames: Set(
                            (toolDescriptors ?? []).map { $0.name }
                        )
                    )
                    message = OpenAIChatMessage(
                        role: .assistant, content: nil, toolCalls: [call]
                    )
                }
                finishReason = .toolCalls
            } catch {
                return .complete(OpenAIJSON.toolCallFailureResponse(
                    message: "Native tool-call did not decode: \(error) "
                        + "(finish_reason=\(result.finishReason.rawValue))",
                    partialJson: result.text,
                    generatedTokenIds: result.tokens
                ))
            }
        case .strictNoAccept:
            return .complete(OpenAIJSON.toolCallFailureResponse(
                message: "Tool-call generation did not complete: "
                    + "matcher did not accept within max_tokens "
                    + "(finish_reason=\(result.finishReason.rawValue))",
                partialJson: result.text,
                generatedTokenIds: result.tokens
            ))
        case .text:
            message = OpenAIChatMessage(
                role: .assistant, content: result.text
            )
        }

        let response = OpenAIChatCompletionsResponse(
            id: OpenAIJSON.chatCompletionId(),
            object: "chat.completion",
            created: Int(Date().timeIntervalSince1970),
            model: modelId,
            choices: [OpenAIChoice(
                index: 0,
                message: message,
                finishReason: finishReason,
                logprobs: chatLogprobs
            )],
            usage: OpenAIUsage(
                promptTokens: inputIds.count,
                // result.tokens may be one higher than the text
                // bytes the user sees when stop-sequence truncation
                // cuts mid-token. The completion-tokens field is
                // best-effort under that scenario; OpenAI's spec
                // doesn't require strict accuracy.
                completionTokens: result.tokens.count,
                totalTokens: inputIds.count + result.tokens.count
            )
        )
        let body = try! OpenAIJSON.encode(response)
        let sessionId = commitSessionIntent(sessionIntent)
        let headers = sessionId.map { ["X-Smelt-Session-Id": $0] } ?? [:]
        return .complete(SmeltServeRawResponse(
            statusCode: 200, headers: headers, body: body
        ))
    }

    /// Wraps an error response so the tool-wiring helpers
    /// (`extractActiveToolDescriptors`, `buildToolMatcher`) can
    /// surface 4xx validation failures distinctly from generic
    /// runtime/grammar build errors.
    private struct ToolWiringError: Error {
        let response: SmeltServeRawResponse
    }

    private enum SessionIntent {
        case stateless
        case create
        case resume(id: String)
    }

    private enum SessionIntentResult {
        case ok(SessionIntent)
        case error(SmeltServeRawResponse)
    }

    /// Validate the request's session fields against the live store.
    /// session_id wins when both are set; an unknown session_id is
    /// always a 404 (with `error.code = "session_not_found"`), the
    /// signal Pi uses to clear its persisted id and retry with
    /// `create_session: true`.
    ///
    /// Contract reminder: the caller MUST still send the full message
    /// history in every request. session_id is a cache-affinity hint
    /// (it pairs with the prefix cache via the stored inputIds) — it
    /// does NOT cause the server to substitute prior messages on its
    /// own. The store deliberately doesn't validate that the new
    /// inputIds extend the stored ones: branch/edit workflows can
    /// legitimately diverge, and the prefix cache's LCP handles
    /// partial-match prefills correctly either way.
    private func resolveSessionIntent(
        request: OpenAIChatCompletionsRequest
    ) -> SessionIntentResult {
        guard let store = sessionStore else {
            // Server disabled session_id support; treat all requests
            // as stateless. We do NOT 404 — callers that don't care
            // about sessions still want their generation to succeed.
            return .ok(.stateless)
        }
        if let id = request.sessionId {
            guard store.contains(id) else {
                return .error(OpenAIJSON.errorResponse(
                    status: 404, code: .sessionNotFound,
                    message: "session_id \"\(id)\" not found "
                        + "(evicted, expired, or never allocated)"
                ))
            }
            return .ok(.resume(id: id))
        }
        if request.createSession == true {
            return .ok(.create)
        }
        return .ok(.stateless)
    }

    /// Returned id is the value to echo in `X-Smelt-Session-Id`;
    /// nil means "omit the header" (stateless request or store
    /// disabled — callers must not synthesize an id of their own).
    private func commitSessionIntent(_ intent: SessionIntent) -> String? {
        guard let store = sessionStore else { return nil }
        switch intent {
        case .stateless:
            return nil
        case .create:
            let id = store.allocate()
            store.touch(id)
            return id
        case .resume(let id):
            store.touch(id)
            return id
        }
    }

    private func decodeGeneratedToolCall(
        _ jsonText: String
    ) throws -> OpenAIToolCall {
        let parsed = try OpenAIJSON.decode(
            SmeltGeneratedToolCall.self,
            from: Data(jsonText.utf8)
        )
        return OpenAIToolCall(
            id: SmeltToolCallID.next(),
            type: .function,
            function: OpenAIToolCallFunction(
                name: parsed.name,
                arguments: try SmeltJSON.canonicalString(parsed.arguments)
            )
        )
    }

    private func decodeNativeTranscriptToolCalls(
        _ text: String,
        descriptors: [SmeltToolDescriptor]
    ) throws -> (content: String?, calls: [OpenAIToolCall]) {
        guard let toolTranscriptCodec else {
            throw SmeltServeError("package has no native tool transcript codec")
        }
        let decoded = try toolTranscriptCodec.decode(text, tools: descriptors)
        return (
            decoded.leadingText,
            decoded.calls.map { call in
                OpenAIToolCall(
                    id: SmeltToolCallID.next(),
                    type: .function,
                    function: OpenAIToolCallFunction(
                        name: call.name,
                        arguments: call.argumentsJSON
                    )
                )
            }
        )
    }

    // decodeNativeToolCall, scanJSONObjectEnd,
    // NativeToolCallParseError, BufferedShape, and
    // classifyBufferedShape extracted to NativeToolCallDecode.swift.

    /// Extract the tool descriptors that should be active for this
    /// request. Returns nil only when no tools were sent OR
    /// `tool_choice` was `"none"`. The choice → matcher-shape mapping
    /// (auto/nil → union, required/specific → strict) lives in
    /// `toolMatcherUsesUnion`; this function just produces the
    /// descriptor list. `required`/`specific` with no tools 400s per
    /// OpenAI's contract; `specific` filters to the named function.
    private func extractActiveToolDescriptors(
        request: OpenAIChatCompletionsRequest,
        rawBody: Data
    ) throws -> [SmeltToolDescriptor]? {
        if request.toolChoice == .disabled { return nil }
        let tools = request.tools ?? []
        // tool_choice = required / specific with no tools must 400 per
        // OpenAI's contract; auto with no tools degrades to "no tools".
        if tools.isEmpty {
            switch request.toolChoice {
            case .required, .specific:
                throw ToolWiringError(response: OpenAIJSON.errorResponse(
                    status: 400, code: .invalidRequest,
                    message: "tool_choice requires a non-empty tools array"
                ))
            default:
                return nil
            }
        }

        let orderedTools = try Self.orderedToolJSON(from: rawBody)
        guard orderedTools.count == tools.count else {
            throw SmeltOrderedJSONError.invalid(
                "decoded and ordered tool counts disagree"
            )
        }

        let candidates: [(
            tool: OpenAIChatTool,
            schema: (parameterJSON: String?, descriptorJSON: String)
        )]
        if case .specific(let name) = request.toolChoice {
            candidates = tools.enumerated().compactMap { index, tool in
                guard tool.function.name == name else { return nil }
                return (tool, orderedTools[index])
            }
            if candidates.isEmpty {
                throw ToolWiringError(response: OpenAIJSON.errorResponse(
                    status: 400, code: .invalidRequest,
                    message: "tool_choice references unknown function "
                        + "\"\(name)\""
                ))
            }
        } else {
            candidates = tools.enumerated().map { index, tool in
                (tool, orderedTools[index])
            }
        }

        return try candidates.map { candidate in
            let tool = candidate.tool
            let schemaJSON: String
            if let params = tool.function.parameters {
                // OpenAI requires the parameters JSON schema to describe
                // an object; our decode-on-acceptance type also requires
                // that. Reject other top-level types at the boundary with
                // 400 instead of letting it 500 at decode time.
                if !Self.schemaDescribesObject(params) {
                    throw ToolWiringError(response: OpenAIJSON.errorResponse(
                        status: 400, code: .invalidRequest,
                        message: "tool \"\(tool.function.name)\" parameters "
                            + "must describe an object"
                    ))
                }
                guard let ordered = candidate.schema.parameterJSON else {
                    throw SmeltOrderedJSONError.invalid(
                        "tool \"\(tool.function.name)\" lost its parameters"
                    )
                }
                schemaJSON = ordered
            } else {
                schemaJSON = "{}"
            }
            return SmeltToolDescriptor(
                name: tool.function.name,
                schemaJSON: schemaJSON,
                description: tool.function.description,
                promptJSON: candidate.schema.descriptorJSON
            )
        }
    }

    /// Recover the source member order erased by `JSONDecoder`. The pinned
    /// Transformers template feeds each tool through Jinja `tojson`, where
    /// object-member order changes token IDs even though JSON semantics do not.
    /// Keeping that order is therefore part of faithful prompt adaptation, not
    /// a model-family special case.
    private static func orderedToolJSON(
        from body: Data
    ) throws -> [(parameterJSON: String?, descriptorJSON: String)] {
        let root = try SmeltOrderedJSONValue.parse(body)
        guard let tools = root.member(named: "tools") else { return [] }
        if case .null = tools { return [] }
        guard let values = tools.arrayValues else {
            throw SmeltOrderedJSONError.invalid("tools is not an array")
        }
        return try values.map { tool in
            guard let function = tool.member(named: "function") else {
                throw SmeltOrderedJSONError.invalid(
                    "tool is missing its function object"
                )
            }
            return (
                function.member(named: "parameters")?.compactJSON,
                tool.compactJSON
            )
        }
    }

    /// JSON Schema's `type` keyword may be a single string or an
    /// array of strings ("ANY of these"). Returns true ONLY when
    /// the schema unambiguously requires an object — either `type:
    /// "object"` or `type: ["object"]`. An elided `type` is
    /// permissive (matches any JSON value), and a union like
    /// `["object", "null"]` lets the matcher accept null; both
    /// would produce a non-object `arguments` that the decoder
    /// then rejects as a 500. Tighter here = 400 at boundary.
    private static func schemaDescribesObject(
        _ schema: SmeltJSONValue
    ) -> Bool {
        guard case .object(let obj) = schema else { return false }
        guard let typeField = obj["type"] else { return false }
        switch typeField {
        case .string(let s):
            return s == "object"
        case .array(let arr):
            return arr.count == 1
                && (arr.first.flatMap { value -> Bool? in
                    if case .string("object") = value { return true }
                    return nil
                } ?? false)
        default:
            return false
        }
    }

    private func buildToolMatcher(
        descriptors: [SmeltToolDescriptor],
        useUnion: Bool
    ) throws -> SmeltLLGuidanceMatcher {
        let diagnostics = ProcessInfo.processInfo.environment[
            "SMELT_SERVE_PREFIX_CACHE_DIAGNOSTICS"
        ] == "1"
        let matcherStart = CFAbsoluteTimeGetCurrent()
        let tokenizer = try sharedLLGuidanceTokenizer()
        do {
            // Generate the grammar (cheap Swift templating); the
            // expensive step is llg_new_matcher compiling it. Key the
            // cache on the grammar text so a repeated tool set skips
            // the compile. Hand each request a fresh clone of the
            // pristine prototype — a consumed matcher can't be rewound.
            let grammar: String
            let key: String
            if let toolTranscriptCodec {
                grammar = try toolTranscriptCodec.larkGrammar(
                    for: descriptors,
                    allowText: useUnion
                )
                key = toolTranscriptCodec.name + "\n" + grammar
            } else if useUnion {
                grammar = try SmeltToolGrammar.larkUnion(for: descriptors)
                key = "lark\n" + grammar
            } else {
                grammar = try SmeltToolGrammar.jsonSchema(for: descriptors)
                key = "json\n" + grammar
            }
            if let cached = cachedToolMatchers[key] {
                let copy = try cached.freshCopy()
                if diagnostics {
                    let ms = (CFAbsoluteTimeGetCurrent() - matcherStart) * 1_000
                    fputs(
                        "smelt serve llguidance matcher=clone ms="
                            + "\(String(format: "%.1f", ms))\n",
                        stderr
                    )
                }
                return copy
            }
            let built = (useUnion || toolTranscriptCodec != nil)
                ? try SmeltLLGuidanceMatcher(tokenizer: tokenizer, lark: grammar)
                : try SmeltLLGuidanceMatcher(tokenizer: tokenizer, jsonSchema: grammar)
            // Store a pristine clone as the prototype before `built`
            // is consumed by this request, so the next hit clones from
            // the initial state.
            if cachedToolMatchers.count >= 8,
               let evicted = cachedToolMatchers.keys.sorted().first {
                cachedToolMatchers.removeValue(forKey: evicted)
            }
            cachedToolMatchers[key] = try built.freshCopy()
            if diagnostics {
                let ms = (CFAbsoluteTimeGetCurrent() - matcherStart) * 1_000
                fputs(
                    "smelt serve llguidance matcher=compile ms="
                        + "\(String(format: "%.1f", ms))\n",
                    stderr
                )
            }
            return built
        } catch let error as SmeltToolGrammarError {
            // Per-tool schema canonicalization rejects non-object
            // JSON parameter shapes — surface as 400, not 500.
            throw ToolWiringError(response: OpenAIJSON.errorResponse(
                status: 400, code: .invalidRequest,
                message: "Invalid tool grammar: \(error)"
            ))
        }
    }

    /// OpenAI default when tools are present is `tool_choice:"auto"` —
    /// the model may emit a tool call OR free text. We honor that via
    /// the lark-union grammar. `required` and `specific` keep the
    /// strict json_schema constraint (must emit a call). `none` is
    /// handled upstream by `extractActiveToolDescriptors` returning
    /// nil (no matcher at all).
    private static func toolMatcherUsesUnion(
        choice: OpenAIToolChoice?
    ) -> Bool {
        switch choice {
        case nil, .some(.auto): return true
        case .some(.required), .some(.specific), .some(.disabled): return false
        }
    }

    // lastNonSystemRoleIsTool, requestInvolvesTools, and
    // repeatedAnsweredQuestionAnswer extracted to
    // ChatMessageHeuristics.swift.

    /// SSE error frame emitted before `[DONE]` when a tool-call turn
    /// fails (matcher refused to accept, JSON failed to parse mid-
    /// stream, or union JSON arm was truncated). pi-ai and the
    /// OpenAI SDK both surface `data: {"error": ...}` as a stream
    /// error event; the envelope carries partial_json +
    /// generated_token_ids so callers can record / replay.
    private static func writeToolCallFailureFrame(
        stream: SmeltServeStreamHandle,
        message: String,
        partialJson: String,
        generatedTokenIds: [Int32]
    ) async throws {
        let envelope = OpenAIJSON.toolCallFailureEnvelope(
            message: message,
            partialJson: partialJson,
            generatedTokenIds: generatedTokenIds
        )
        var frame = Data("data: ".utf8)
        frame.append(envelope)
        frame.append(Data("\n\n".utf8))
        try await stream.writeChunk(frame)
    }

    private func sharedLLGuidanceTokenizer() throws -> SmeltLLGuidanceTokenizer {
        if let cached = cachedLLGTokenizer { return cached }
        let new = try SmeltLLGuidanceTokenizer(
            tokenizer: tokenizer,
            eosTokens: inference.eosTokens
        )
        cachedLLGTokenizer = new
        return new
    }

    /// The matcher constrains the output shape; this message conveys
    /// the SEMANTIC information (which tools exist, what each one
    /// does, what arguments to fill) so the model can pick correctly.
    /// Format is generic across chat templates; per-template tool
    /// channels are selected by prompt-format capability, not model identity.
    ///
    /// `useUnion` toggles the prompt's mandate from "MUST invoke a
    /// tool" (strict mode) to "if a tool is the right next step,
    /// invoke it; otherwise reply with text" (union mode). The
    /// grammar enforces the shape; this just tells the model when a
    /// text reply is allowed — which it is on call → result → answer
    /// follow-up turns.
    private func renderToolsSystemMessage(
        descriptors: [SmeltToolDescriptor],
        choice: OpenAIToolChoice?,
        useUnion: Bool
    ) -> String {
        let mandate: String
        if useUnion {
            mandate = "When the user asks you to read, write, list, or "
                + "otherwise interact with a file, system, or external "
                + "resource that the tools below can reach, you MUST "
                + "invoke the appropriate tool by responding with a "
                + "JSON object of the exact shape "
                + "{\"name\": \"<tool_name>\", \"arguments\": {<arguments>}}. "
                + "Only respond with plain text when answering from "
                + "information already present in the conversation or "
                + "when the user's question does not require a tool."
        } else if case .specific(let name) = choice {
            mandate = "You MUST invoke the \"\(name)\" tool. Respond ONLY "
                + "with a JSON object matching the shape "
                + "{\"name\": \"<tool_name>\", \"arguments\": {<arguments>}}."
        } else if descriptors.count == 1 {
            mandate = "You MUST invoke the \"\(descriptors[0].name)\" tool. "
                + "Respond ONLY with a JSON object matching the shape "
                + "{\"name\": \"<tool_name>\", \"arguments\": {<arguments>}}."
        } else {
            mandate = "You MUST invoke exactly one of the tools below. "
                + "Respond ONLY with a JSON object matching the shape "
                + "{\"name\": \"<tool_name>\", \"arguments\": {<arguments>}}."
        }
        var lines: [String] = [
            "You have access to the tools listed below.",
            mandate,
            "",
            "Tools:"
        ]
        for descriptor in descriptors {
            var entry = "- \(descriptor.name)"
            if let desc = descriptor.description, !desc.isEmpty {
                entry += ": \(desc)"
            }
            entry += "\n  Arguments schema: \(descriptor.schemaJSON)"
            lines.append(entry)
        }
        return lines.joined(separator: "\n")
    }

    private func handleCompletions(
        _ body: Data,
        requestId: SmeltServeRequestId,
        transport: any SmeltServeTransport
    ) async -> HandlerResult {
        let request: OpenAICompletionsRequest
        do {
            request = try OpenAIJSON.decode(OpenAICompletionsRequest.self, from: body)
        } catch {
            return .complete(OpenAIJSON.errorResponse(
                status: 400, code: .invalidRequest,
                message: "Failed to decode request: \(error)"
            ))
        }
        if let reason = invalidSamplingParameters(
            temperature: request.temperature,
            topK: request.topK,
            topP: request.topP
        ) {
            return .complete(OpenAIJSON.errorResponse(
                status: 400, code: .invalidRequest, message: reason
            ))
        }

        // promptText is what response.text echoes when echo:true + the
        // prompt-side prefill path is NOT used. For pre-tokenized array
        // prompts, decode back to text. inputIds is what the model
        // actually consumes — directly from the array, or via
        // buildInputIds(template:"") for the string form.
        let promptText: String
        let inputIds: [Int32]
        switch request.prompt {
        case .text(let s):
            guard !s.isEmpty else {
                return .complete(OpenAIJSON.errorResponse(
                    status: 400, code: .invalidRequest,
                    message: "prompt must be a non-empty string"
                ))
            }
            promptText = s
            do {
                inputIds = try buildInputIds(
                    prompt: s,
                    tokenizer: tokenizer,
                    template: ""
                )
            } catch {
                return .complete(OpenAIJSON.errorResponse(
                    status: 400, code: .invalidRequest,
                    message: "Failed to tokenize prompt: \(error)"
                ))
            }
        case .tokens(let ids):
            guard !ids.isEmpty else {
                return .complete(OpenAIJSON.errorResponse(
                    status: 400, code: .invalidRequest,
                    message: "prompt array must be non-empty"
                ))
            }
            // Token IDs come from untrusted client input. Out-of-vocab
            // or negative values would crash decodeStep / prefillAllLogits
            // via precondition traps in the runtime — validate up front
            // so malformed requests return 400, not 500-or-worse.
            let vocab = Int32(runtime.vocabSize)
            if let bad = ids.first(where: { $0 < 0 || $0 >= vocab }) {
                return .complete(OpenAIJSON.errorResponse(
                    status: 400, code: .invalidRequest,
                    message: "prompt contains out-of-vocab token id \(bad) "
                        + "(vocab size \(runtime.vocabSize))"
                ))
            }
            inputIds = ids
            // Decode tokens back to text for echo:true responses on the
            // fallback path. The prompt-side path renders its own tokens
            // and doesn't read this.
            promptText = tokenizer.decode(ids)
        }

        if inputIds.count >= runtime.maxContextTokens {
            return .complete(OpenAIJSON.errorResponse(
                status: 400, code: .contextLengthExceeded,
                message: "prompt is \(inputIds.count) tokens; context limit is \(runtime.maxContextTokens)"
            ))
        }

        let requestedMax = request.effectiveMaxTokens ?? inference.maxTokens
        let maxTokens = max(0, min(
            requestedMax,
            inference.maxTokens,
            runtime.maxContextTokens - inputIds.count
        ))
        // OpenAI legacy: logprobs is an Int (top-K count, 0..5), not bool.
        let logprobsTopK: Int? = request.logprobs.map { max(0, min(5, $0)) }
        let includePromptLogprobs = (request.echo == true) && logprobsTopK != nil
        let n = max(1, min(16, request.n ?? 1))

        if request.stream == true {
            // echo with stream isn't supported — the prompt-side logits
            // path needs the full rows array which streaming explicitly
            // doesn't materialize. Match OpenAI's behavior (it rejects
            // the combination on the legacy endpoint).
            if request.echo == true {
                return .complete(OpenAIJSON.errorResponse(
                    status: 400, code: .invalidRequest,
                    message: "echo:true is incompatible with stream:true"
                ))
            }
            return await runStreamingCompletions(
                inputIds: inputIds,
                maxTokens: maxTokens,
                temperature: request.temperature,
                topK: request.topK,
                topP: request.topP,
                seed: request.seed,
                stopSequences: request.stop?.sequences ?? [],
                n: n,
                requestId: requestId,
                transport: transport
            )
        }

        var choices: [OpenAICompletionChoice] = []
        choices.reserveCapacity(n)
        var totalCompletionTokens = 0
        // When sampling is deterministic (temperature ≤ 0 / nil ⇒
        // argmax), all n generations would be byte-identical. Skip the
        // redundant work and clone the single result n times. Saves
        // ~165 ms × (n-1) on a typical 16-choice deterministic request.
        let isDeterministic = (request.temperature ?? 0) <= 0
        let effectiveN = isDeterministic ? 1 : n
        // Sequential, not parallel: each call to runCompletionsGeneration
        // mutates the shared runtime K/V buffers (prepareForRequest +
        // resetWorkingBuffers + chunkedPrefillAllLogits). Concurrent
        // calls would corrupt state. The prefix cache makes iteration
        // k > 0 cheap by hitting on iteration 0's stored prefix.
        for index in 0 ..< effectiveN {
            // Per-choice seed offset: when request.seed is provided,
            // derive a distinct sub-seed so the n samples diverge on
            // temperature > 0. Wrapping add avoids Int trap at max.
            let perChoiceSeed: Int? = request.seed.map { $0 &+ index }
            let selectionMode = resolveSelectionMode(
                temperature: request.temperature,
                seed: perChoiceSeed,
                topK: request.topK,
                topP: request.topP
            )
            let result: CompletionsGenerationResult
            do {
                result = try runCompletionsGeneration(
                    inputIds: inputIds,
                    maxTokens: maxTokens,
                    selectionMode: selectionMode,
                    stopSequences: request.stop?.sequences ?? [],
                    logprobsTopK: logprobsTopK,
                    includePromptLogprobs: includePromptLogprobs
                )
            } catch {
                return .complete(OpenAIJSON.errorResponse(
                    status: 500, code: .internalError,
                    message: "Generation failed: \(error)"
                ))
            }
            let text: String
            if request.echo == true && !result.textOffsetsIncludePrompt {
                text = promptText + result.text
            } else if request.echo == true {
                text = result.tokenChunks.prefix(inputIds.count).joined() + result.text
            } else {
                text = result.text
            }
            let completionLogprobs: OpenAICompletionLogprobs? = result.logprobs.map { entries in
                let offsets: [Int]
                if request.echo == true && !result.textOffsetsIncludePrompt {
                    let promptOffset = promptText.utf8.count
                    offsets = result.textOffsets.map { $0 + promptOffset }
                } else {
                    offsets = result.textOffsets
                }
                return LogprobsCompute.completionLogprobs(
                    entries,
                    tokenTexts: result.tokenChunks,
                    textOffsets: offsets,
                    tokenizer: tokenizer
                )
            }
            choices.append(OpenAICompletionChoice(
                index: index,
                text: text,
                finishReason: result.finishReason,
                logprobs: completionLogprobs
            ))
            totalCompletionTokens += result.tokens.count
        }

        // Deterministic-mode replication: when temperature ≤ 0 the
        // first iteration produced the only meaningful result; clone
        // it n-1 more times with adjusted index so the wire response
        // still has `choices` of length n per the OpenAI contract.
        // Match: usage.completion_tokens scales by the number of
        // logical completions emitted, not by how many we actually
        // ran. OpenAI's billing semantics count each completion in
        // the response.
        if isDeterministic, let template = choices.first {
            let perChoiceTokens = totalCompletionTokens
            while choices.count < n {
                choices.append(OpenAICompletionChoice(
                    index: choices.count,
                    text: template.text,
                    finishReason: template.finishReason,
                    logprobs: template.logprobs
                ))
                totalCompletionTokens += perChoiceTokens
            }
        }

        let response = OpenAICompletionsResponse(
            id: OpenAIJSON.completionId(),
            object: "text_completion",
            created: Int(Date().timeIntervalSince1970),
            model: modelId,
            choices: choices,
            usage: OpenAIUsage(
                promptTokens: inputIds.count,
                completionTokens: totalCompletionTokens,
                totalTokens: inputIds.count + totalCompletionTokens
            )
        )
        let bodyOut = try! OpenAIJSON.encode(response)
        return .complete(SmeltServeRawResponse(statusCode: 200, body: bodyOut))
    }

    private struct CompletionsGenerationResult {
        let tokens: [Int32]
        let text: String
        let finishReason: OpenAIFinishReason
        // nil entries are allowed: they represent the first prompt
        // token (BOS) on the echo:true + prompt-side-logprobs path,
        // which has no preceding distribution to compute logprob from.
        let logprobs: [LogprobEntry?]?
        let tokenChunks: [String]
        let textOffsets: [Int]
        /// True when textOffsets already index into the FULL response
        /// text (echo:true prompt-side path). False means offsets are
        /// relative to the generation-only text and the handler must
        /// shift them by promptOffset before emitting.
        let textOffsetsIncludePrompt: Bool
    }

    private func runCompletionsGeneration(
        inputIds: [Int32],
        maxTokens: Int,
        selectionMode: SmeltSelectionMode,
        stopSequences: [String],
        logprobsTopK: Int?,
        includePromptLogprobs: Bool
    ) throws -> CompletionsGenerationResult {
        try runtime.prepareForRequest(
            batchCapacity: 1,
            contextCapacity: inputIds.count + maxTokens
        )

        // Track per-position prompt logprobs + tokens + offsets when the
        // caller asked for echo+logprobs AND the package supports batched
        // prefill. The package's max_prefill_batch limit is per-dispatch,
        // not per-prompt — we chunk long prompts internally below. The
        // only hard requirements are the all-logits capability and a
        // positionally composable package state layout.
        let topK = logprobsTopK
        let promptLogprobsPath = includePromptLogprobs
            && topK != nil
            && runtime.supportsBatchedPromptPrefill

        var cur: Int32
        var promptLogprobs: [LogprobEntry?] = []
        var promptTokens: [String] = []
        var promptOffsets: [Int] = []
        // Carries the row that predicts the FIRST generation token out
        // of the prompt-side branch so the loop's first iteration can
        // compute its logprob from the CORRECT distribution. The
        // runtime's current logits buffer after a chunked prefill holds
        // the row 0 of the LAST chunk, not the row that predicts the
        // next continuation — so a direct compute(runtime:) on the
        // first generated token would read the wrong row.
        var firstGenerationLogitsRow: [Float16]? = nil

        var promptRunningOffset = 0
        // Track the cold-path post-prefill snapshot for caching.
        // Only set on a cache miss (cold path) when promptLogprobsPath
        // is taken AND prefixCache is enabled.
        var capturedSnapshotForCache: SmeltPromptSnapshot? = nil

        if promptLogprobsPath, let topK {
            let prefill = try runPrefillWithCache(
                inputIds: inputIds, wantFullRows: true
            )
            var rows = prefill.rows!
            let firstRowAbsolutePosition = prefill.firstRowAbsolutePosition
            capturedSnapshotForCache = prefill.capturedSnapshot
            // Build the prompt-side render (tokens + offsets) sequentially —
            // this is cheap (tokenizer lookups, no per-vocab scans).
            promptTokens.reserveCapacity(inputIds.count)
            promptOffsets.reserveCapacity(inputIds.count)
            for i in 0 ..< inputIds.count {
                let tokenText = LogprobsCompute.rawTokenString(
                    inputIds[i], tokenizer: tokenizer
                )
                promptTokens.append(tokenText)
                promptOffsets.append(promptRunningOffset)
                promptRunningOffset += tokenText.utf8.count
            }

            // Compute per-position logprobs in parallel. `rows` covers
            // absolute positions [firstRowAbsolutePosition, ...]. For
            // chosen-token at position i, the scoring distribution is
            // at position i-1, which lives at rows[i-1 - firstRowAbsolutePosition]
            // when that index is valid. Positions before
            // firstRowAbsolutePosition + 1 get nil (BOS on cold path,
            // or cached-stem positions on hit path).
            promptLogprobs = [LogprobEntry?](repeating: nil, count: inputIds.count)
            // The first position whose logprob is computable: we need
            // the row at i-1 to exist in `rows`, i.e., i - 1 >= firstRowAbsolutePosition.
            let firstScorablePosition = firstRowAbsolutePosition + 1
            if inputIds.count > firstScorablePosition {
                let immutableRows = rows
                let immutableInputIds = inputIds
                let baseAbs = firstRowAbsolutePosition
                let iterations = inputIds.count - firstScorablePosition
                promptLogprobs.withUnsafeMutableBufferPointer { buf in
                    // Each task writes a distinct index `i`, so the
                    // writes are disjoint and race-free. The buffer
                    // pointer is non-Sendable and `buf` is inout, so
                    // capture a plain base pointer and vouch for it
                    // with nonisolated(unsafe) to satisfy strict
                    // concurrency without a per-element lock.
                    nonisolated(unsafe) let base = buf.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: iterations) { task in
                        let i = firstScorablePosition + task
                        base[i] = LogprobsCompute.computeFromLogits(
                            logits: immutableRows[i - 1 - baseAbs],
                            chosenToken: immutableInputIds[i],
                            topK: topK
                        )
                    }
                }
            }
            // Drop middle rows now that the parallel section finished.
            // Each fp16 row is vocab × 2 bytes (~512 KB at vocab=262144).
            // For a 4096-token prompt that's ~2 GB held until end of
            // request — only rows.last is needed below.
            if rows.count > 1 {
                for i in 0 ..< rows.count - 1 {
                    rows[i] = []
                }
            }
            // withUnsafeBufferPointer's pointer is valid ONLY inside the
            // closure — call select inline rather than returning the
            // pointer (it would dangle as the temporary array gets freed).
            cur = rows.last!.withUnsafeBufferPointer { buf in
                SmeltLogitsSelector.select(
                    logits: buf,
                    position: Int32(inputIds.count - 1),
                    mode: selectionMode
                )
            }
            firstGenerationLogitsRow = rows.last
        } else {
            // Same per-request reset as runPrelude's sequential branch: this
            // endpoint must clear every prior persistent state family before
            // sequential prefill, or identical requests drift across a session.
            runtime.resetWorkingBuffers()
            if includePromptLogprobs {
                fputs(
                    "smelt serve: prompt logprobs unavailable for this package "
                    + "(missing all-logits prefill or a positionally composable state layout); "
                    + "echoing without prompt-side logprobs.\n",
                    stderr
                )
            }
            cur = -1
            for (position, tokenId) in inputIds.enumerated() {
                let isLast = position == inputIds.count - 1
                cur = try runtime.decodeStep(
                    tokenId: tokenId,
                    position: Int32(position),
                    selectionMode: isLast ? selectionMode : .argmax
                )
            }
        }
        var pos = inputIds.count

        var streamingDecoder = SmeltStreamingTokenDecoder(continuingExistingText: true)
        let eosTokens = Set(inference.eosTokens)
        let stopWindow = stopSequences.lazy.map(\.count).max() ?? 0
        var generated: [Int32] = []
        var accumulated = ""
        var finishReason: OpenAIFinishReason = .length
        var logprobs: [LogprobEntry?]? = (topK != nil) ? promptLogprobs : nil
        var tokenChunks: [String] = promptTokens
        var textOffsets: [Int] = promptOffsets
        // When the prompt-side path filled promptOffsets, generation
        // offsets must continue from end-of-prompt to stay aligned with
        // response.text. Without prompt-side, generation tracks its own
        // text starting at 0 and the handler shifts later if echo:true.
        let generationOffsetBase = promptLogprobsPath ? promptRunningOffset : 0

        while generated.count < maxTokens {
            if eosTokens.contains(cur) {
                finishReason = .stop
                break
            }
            generated.append(cur)
            let chunk = streamingDecoder.decode(tokenId: cur, tokenizer: tokenizer)
            let offsetBeforeChunk = generationOffsetBase + accumulated.utf8.count
            accumulated += chunk

            if let topK = logprobsTopK {
                logprobs?.append(consumeFirstGenerationLogprob(
                    row: &firstGenerationLogitsRow,
                    chosenToken: cur, topK: topK
                ))
                tokenChunks.append(chunk)
                textOffsets.append(offsetBeforeChunk)
            }

            let scanLen = chunk.count + stopWindow
            if let cutoff = matchedStopSequenceRange(
                in: accumulated, sequences: stopSequences, scanLen: scanLen
            ) {
                accumulated = String(accumulated[..<cutoff.lowerBound])
                finishReason = .stop
                // Drop trailing generation entries whose token text doesn't
                // survive the truncation. Prompt-side entries (textOffset
                // < generationOffsetBase) are always safe — they predate
                // the generation buffer entirely.
                assert(
                    (logprobs?.count ?? tokenChunks.count) == tokenChunks.count
                        && tokenChunks.count == textOffsets.count,
                    "logprobs/tokenChunks/textOffsets out of lockstep"
                )
                let cutoffOffset = generationOffsetBase + accumulated.utf8.count
                while let lastOffset = textOffsets.last,
                      let lastChunk = tokenChunks.last,
                      lastOffset >= generationOffsetBase,
                      lastOffset + lastChunk.utf8.count > cutoffOffset
                {
                    textOffsets.removeLast()
                    tokenChunks.removeLast()
                    logprobs?.removeLast()
                }
                break
            }

            cur = try runtime.decodeStep(
                tokenId: cur,
                position: Int32(pos),
                selectionMode: selectionMode
            )
            pos += 1
        }

        // Store the cold-path snapshot in the prefix cache for the
        // benefit of future requests sharing this prompt's prefix.
        // The snapshot was captured RIGHT AFTER the cold prefill
        // completed (before any generation tokens were appended to
        // K/V), so its K/V represents positions [0, inputIds.count).
        storeColdPrefixIfPossible(
            inputIds: inputIds, snapshot: capturedSnapshotForCache
        )

        return CompletionsGenerationResult(
            tokens: generated,
            text: accumulated,
            finishReason: finishReason,
            logprobs: logprobs,
            tokenChunks: tokenChunks,
            textOffsets: textOffsets,
            textOffsetsIncludePrompt: promptLogprobsPath
        )
    }

    private struct PrefillResult {
        /// Full per-position rows. nil when the caller passed
        /// wantFullRows=false (chat path, which only needs lastRow).
        let rows: [[Float16]]?
        /// The final logit row — always populated. Used to sample
        /// the first generation token AND stash as
        /// firstGenerationLogitsRow for the loop's first iteration
        /// compute-from-logits.
        let lastRow: [Float16]
        /// Absolute position of `rows[0]` when rows != nil. 0 on
        /// cold path, effectiveLCP - 1 on cache-hit path.
        let firstRowAbsolutePosition: Int
        /// Post-prefill snapshot for cache.store(). nil when the
        /// cache is disabled at the handler level.
        let capturedSnapshot: SmeltPromptSnapshot?
    }

    /// A capability-neutral prompt continuation plan. Both all-logits prefill
    /// and final-selection prefill must restore exactly the same state and run
    /// exactly the same token suffix; keeping that decision in one place avoids
    /// the two routes drifting on prepared-prompt replay tails or cache hits.
    private struct PromptPrefillWork {
        let tokens: [Int32]
        let startPosition: Int
        let firstRowAbsolutePosition: Int
    }

    private func preparePromptPrefillWork(
        inputIds: [Int32],
        preparedPrompt: SmeltPreparedPromptState?,
        skipCache: Bool,
        inputIdentity: SmeltPromptInputIdentity
    ) throws -> PromptPrefillWork {
        // Callers may skip partial-LCP reuse for transcript formats whose
        // divergent suffix has not proved positional restore parity. Exact-
        // position state is only restored at a complete checkpoint.
        let cacheMatch = skipCache ? nil : prefixCache?.tryMatch(
            inputIds,
            inputIdentity: inputIdentity
        )
        logPrefixCacheLookupIfRequested(skipped: skipCache)
        let preparedMatch = inputIdentity == .text && cacheMatch == nil
            ? preparedPrompt : nil

        if let match = cacheMatch {
            let lcp = match.effectiveLCP
            runtime.resetWorkingBuffers()
            if match.restoreExactly {
                switch match.chosenSnapshot.snapshot {
                case .host(let snapshot):
                    try runtime.restorePromptSnapshot(snapshot)
                case .device(let snapshot):
                    try runtime.restoreDevicePromptSnapshot(snapshot)
                }
                return PromptPrefillWork(
                    tokens: Array(inputIds[lcp..<inputIds.count]),
                    startPosition: lcp,
                    firstRowAbsolutePosition: lcp
                )
            }

            // Re-prefill the boundary token so prompt logprobs remain fresh.
            guard case .host(let snapshot) = match.chosenSnapshot.snapshot else {
                preconditionFailure("partial LCP restore requires a host snapshot")
            }
            try runtime.restorePromptSnapshot(snapshot, length: lcp - 1)
            return PromptPrefillWork(
                tokens: Array(inputIds[(lcp - 1)..<inputIds.count]),
                startPosition: lcp - 1,
                firstRowAbsolutePosition: lcp - 1
            )
        }

        if let preparedMatch,
           inputIds.count > preparedMatch.tokenIds.count {
            runtime.resetWorkingBuffers()
            try runtime.restorePromptSnapshot(preparedMatch.snapshot)
            // Automatic package capture may stop on a prefill-aligned
            // boundary and carry the remaining contract IDs as replayTokenIds.
            // Re-enter at the actual captured position so the exact replay tail
            // and request suffix traverse the ordinary graph together.
            let capturedLength = preparedMatch.snapshot.capturedLength
            return PromptPrefillWork(
                tokens: Array(inputIds[capturedLength..<inputIds.count]),
                startPosition: capturedLength,
                firstRowAbsolutePosition: capturedLength
            )
        }

        return PromptPrefillWork(
            tokens: inputIds,
            startPosition: 0,
            firstRowAbsolutePosition: 0
        )
    }

    /// Cache-aware prefill. Tries the LCP prefix cache; on hit
    /// restores K/V at LCP-1 and chunk-prefills only the suffix.
    /// On miss, chunk-prefills the whole prompt. In both cases,
    /// captures a fresh snapshot at the post-prefill state for
    /// `cache.store()` later (so growing-conversation requests
    /// can match on the full token range).
    ///
    /// Chat passes `wantFullRows: false` to save memory — a 4k-tok
    /// prompt at vocab=262144 would otherwise materialize ~2 GB
    /// of fp16 rows in the helper.
    ///
    /// Caller is responsible for calling
    /// `runtime.prepareForRequest(...)` BEFORE this and storing
    /// `capturedSnapshot` via `cache.store(...)` AFTER generation
    /// completes.
    private func runPrefillWithCache(
        inputIds: [Int32],
        preparedPrompt: SmeltPreparedPromptState? = nil,
        wantFullRows: Bool,
        skipCache: Bool = false,
        inputIdentity: SmeltPromptInputIdentity = .text
    ) throws -> PrefillResult {
        let work = try preparePromptPrefillWork(
            inputIds: inputIds,
            preparedPrompt: preparedPrompt,
            skipCache: skipCache,
            inputIdentity: inputIdentity
        )

        let rows: [[Float16]]?
        let lastRow: [Float16]
        if wantFullRows {
            let r = try chunkedPrefillAllLogits(
                tokens: work.tokens, startPos: Int32(work.startPosition)
            )
            lastRow = r.last!
            rows = r
        } else {
            lastRow = try chunkedPrefillLastRow(
                tokens: work.tokens, startPos: Int32(work.startPosition)
            )
            rows = nil
        }

        // Capture the post-prefill state for cache.store() — the
        // K/V now covers [0, inputIds.count). Both hit and miss
        // paths capture (refreshes the cache on hit so growing-
        // conversation requests can match on the full token range).
        // skipCache requests don't poison the cache with snapshots
        // a future request might match against.
        let captured: SmeltPromptSnapshot?
        if prefixCache != nil && !skipCache {
            captured = runtime.capturePromptSnapshot(
                capturedLength: inputIds.count,
                promptLength: inputIds.count,
                nextToken: 0
            )
        } else {
            captured = nil
        }

        return PrefillResult(
            rows: rows,
            lastRow: lastRow,
            firstRowAbsolutePosition: work.firstRowAbsolutePosition,
            capturedSnapshot: captured
        )
    }

    private struct SelectedPrefillResult {
        let token: Int32
        let capturedSnapshot: SmeltPromptSnapshot?
    }

    /// Cache-aware Metal prefill for packages whose prefill table performs the
    /// final token selection instead of emitting every logit row. This is the
    /// native prompt route for recurrent MLX graphs: an exact checkpoint seeds
    /// their opaque state, then the ordinary batched table advances the suffix.
    private func runSelectedPrefillWithCache(
        inputIds: [Int32],
        preparedPrompt: SmeltPreparedPromptState?,
        selectionMode: SmeltSelectionMode,
        allowedTokenMask: SmeltAllowedTokenMaskProvider?,
        skipCache: Bool,
        inputIdentity: SmeltPromptInputIdentity
    ) throws -> SelectedPrefillResult {
        let started = CFAbsoluteTimeGetCurrent()
        let work = try preparePromptPrefillWork(
            inputIds: inputIds,
            preparedPrompt: preparedPrompt,
            skipCache: skipCache,
            inputIdentity: inputIdentity
        )
        precondition(!work.tokens.isEmpty, "prompt prefill requires a suffix token")

        let chunkSize = runtime.maxPrefillBatchSize
        var offset = 0
        var token: Int32 = 0
        while offset < work.tokens.count {
            let end = min(offset + chunkSize, work.tokens.count)
            let isLast = end == work.tokens.count
            token = try runtime.prefillStep(
                tokenIds: Array(work.tokens[offset..<end]),
                startPos: Int32(work.startPosition + offset),
                selectionMode: isLast ? selectionMode : .argmax,
                allowedTokenMask: isLast ? allowedTokenMask : nil
            )
            offset = end
        }

        let captured: SmeltPromptSnapshot?
        if prefixCache != nil && !skipCache
            && prefixCache?.requiresExactRestore != true
        {
            captured = runtime.capturePromptSnapshot(
                capturedLength: inputIds.count,
                promptLength: inputIds.count,
                nextToken: token
            )
        } else {
            captured = nil
        }

        if ProcessInfo.processInfo.environment[
            "SMELT_SERVE_PREFIX_CACHE_DIAGNOSTICS"
        ] == "1" {
            let elapsedMS = (CFAbsoluteTimeGetCurrent() - started) * 1_000
            fputs(
                "smelt serve prefix-cache phases route=metal-prefill-select tokens="
                    + "\(work.tokens.count) start=\(work.startPosition) prefill_ms="
                    + "\(String(format: "%.1f", elapsedMS))\n",
                stderr
            )
        }

        return SelectedPrefillResult(
            token: token,
            capturedSnapshot: captured
        )
    }

    /// Chunked prefillAllLogits walking `tokens` in
    /// `runtime.maxPrefillBatchSize` windows with sequentially
    /// advancing startPos. Each chunk's startPos continues the K/V
    /// cache from the previous chunk, so the concatenated rows match
    /// what a single full prefill would have produced.
    private func chunkedPrefillAllLogits(
        tokens: [Int32], startPos: Int32
    ) throws -> [[Float16]] {
        let chunkSize = runtime.maxPrefillBatchSize
        var rows: [[Float16]] = []
        rows.reserveCapacity(tokens.count)
        var chunkStart = 0
        while chunkStart < tokens.count {
            let chunkEnd = min(chunkStart + chunkSize, tokens.count)
            let chunk = Array(tokens[chunkStart..<chunkEnd])
            let chunkRows = try runtime.prefillAllLogits(
                tokens: chunk, startPos: startPos + Int32(chunkStart)
            )
            rows.append(contentsOf: chunkRows)
            chunkStart = chunkEnd
        }
        return rows
    }

    /// Chunked prefill that only retains the LAST row. Used by the
    /// chat path (which doesn't compute prompt-side logprobs) so
    /// long prompts don't materialize ~512 KB × N intermediate fp16
    /// rows that would otherwise be dropped immediately. Peak memory
    /// is one chunk's worth (~256 MB at max_prefill_batch=512) which
    /// drops between iterations.
    /// Store a single-checkpoint entry covering the full
    /// inputIds range. No-op when the prefix cache is disabled or
    /// the captured snapshot is nil (e.g., cache miss with no
    /// runtime support for prompt-side caching).
    private func storeColdPrefixIfPossible(
        inputIds: [Int32],
        snapshot: SmeltPromptSnapshot?,
        inputIdentity: SmeltPromptInputIdentity = .text
    ) {
        guard let cache = prefixCache, let snapshot else { return }
        cache.store(SmeltPromptStateCacheEntry(
            tokens: inputIds,
            inputIdentity: inputIdentity,
            snapshots: [SmeltPromptStateCheckpoint(
                position: inputIds.count, snapshot: .host(snapshot)
            )]
        ))
    }

    private func logPrefixCacheLookupIfRequested(skipped: Bool) {
        guard ProcessInfo.processInfo.environment[
            "SMELT_SERVE_PREFIX_CACHE_DIAGNOSTICS"
        ] == "1" else { return }
        if skipped {
            fputs("smelt serve prefix-cache miss reason=request-policy-skip\n", stderr)
        } else if let prefixCache {
            fputs("smelt serve prefix-cache \(prefixCache.lastLookup)\n", stderr)
        } else {
            fputs("smelt serve prefix-cache miss reason=disabled\n", stderr)
        }
    }

    private func chunkedPrefillLastRow(
        tokens: [Int32], startPos: Int32
    ) throws -> [Float16] {
        precondition(!tokens.isEmpty, "chunkedPrefillLastRow needs ≥1 token")
        let chunkSize = runtime.maxPrefillBatchSize
        var lastRow: [Float16] = []
        var chunkStart = 0
        while chunkStart < tokens.count {
            let chunkEnd = min(chunkStart + chunkSize, tokens.count)
            let chunk = Array(tokens[chunkStart..<chunkEnd])
            let chunkRows = try runtime.prefillAllLogits(
                tokens: chunk, startPos: startPos + Int32(chunkStart)
            )
            // Only keep the final row of the final chunk.
            if chunkEnd == tokens.count, let last = chunkRows.last {
                lastRow = last
            }
            chunkStart = chunkEnd
        }
        return lastRow
    }

    private func matchedStopSequenceRange(
        in text: String,
        sequences: [String],
        scanLen: Int
    ) -> Range<String.Index>? {
        // Pick the EARLIEST match across all sequences, not the first
        // sequence that happens to match. Otherwise stop=["\nclass",
        // "\ndef"] on "...foo\ndef bar\nclass baz..." would truncate
        // at the later `\nclass` because it's checked first.
        //
        // scanLen is sized by the caller to cover the most recent
        // decoded chunk PLUS one max-stop-length of pre-chunk context.
        // That's enough to catch any stop that newly appears via the
        // chunk (even tokens that decode to many chars, like BPE
        // multi-space tokens up to ▁×8 = 8 spaces) while still bounding
        // the per-token scan to a small constant.
        guard !sequences.isEmpty, scanLen > 0 else { return nil }
        let windowLen = min(text.count, scanLen)
        let windowStart = text.index(text.endIndex, offsetBy: -windowLen)
        var best: Range<String.Index>? = nil
        for stop in sequences where !stop.isEmpty {
            guard let range = text.range(of: stop, range: windowStart..<text.endIndex) else {
                continue
            }
            if best == nil || range.lowerBound < best!.lowerBound {
                best = range
            }
        }
        return best
    }

    private struct ChatGenerationResult {
        let tokens: [Int32]
        let text: String
        let finishReason: OpenAIFinishReason
        let logprobs: [LogprobEntry]?
    }

    /// State produced by chat prefill: ready to enter the per-token
    /// generation loop with the first sampled token and the
    /// position-just-past-the-prompt. Used by both buffered and
    /// streaming chat paths.
    private struct PreludeState {
        var cur: Int32
        var pos: Int
        var firstGenerationLogitsRow: [Float16]?
        var capturedSnapshotForCache: SmeltPromptSnapshot?
    }

    /// Compute the logprob for the just-sampled `chosenToken`. After
    /// a batched prefill the runtime's allLogitsHalf buffer holds
    /// the wrong row (row 0 of the last chunk, not row N-1 that
    /// produced cur). The handler stashes the correct boundary row
    /// in `row` for the first iteration; this helper consumes it
    /// and resets to nil so subsequent iterations read from the
    /// runtime buffer (which decodeStep keeps current).
    private func consumeFirstGenerationLogprob(
        row: inout [Float16]?,
        chosenToken: Int32,
        topK: Int
    ) -> LogprobEntry {
        if let prefillRow = row {
            row = nil
            return LogprobsCompute.computeFromLogits(
                logits: prefillRow, chosenToken: chosenToken, topK: topK
            )
        }
        return LogprobsCompute.compute(
            runtime: runtime, chosenToken: chosenToken, topK: topK
        )
    }

    /// Prefill prelude: cache-aware batched prefill OR per-token
    /// fallback, then optionally the thought-channel skip if cur ==
    /// thinkToken. Thought-skip is off for legacy completions and
    /// for constrained-decode modes (see DecodeMode).
    private func runPrelude(
        inputIds: [Int32],
        inputIdentity: SmeltPromptInputIdentity = .text,
        preparedPrompt: SmeltPreparedPromptState?,
        selectionMode: SmeltSelectionMode,
        mode: DecodeMode,
        skipPrefixCache: Bool,
        allowedTokenMask: (() throws -> [UInt32])?,
        isCancelled: () -> Bool
    ) throws -> PreludeState {
        if isCancelled() { throw CancellationError() }
        let canBatchedPrefill = runtime.supportsBatchedPromptPrefill
        let canBatchedSelection = runtime.supportsBatchedPromptSelection
        var cur: Int32
        var capturedSnapshotForCache: SmeltPromptSnapshot? = nil
        // On the batched path, allLogitsHalf doesn't hold the correct
        // boundary row after prefill — we stash the actual one here
        // for the first generation iteration's logprob compute. Set
        // to nil whenever a decodeStep runs (thinkSkip path) since
        // the runtime buffer is then fresh again.
        var firstGenerationLogitsRow: [Float16]? = nil
        // decodeStep evaluates mask providers between commit and GPU wait,
        // hiding the mask compute under the forward; only the batched path
        // (CPU-side select on the prefill rows) needs an eager mask.
        let maskProvider: SmeltAllowedTokenMaskProvider? =
            allowedTokenMask.map { cb in { () throws -> [UInt32]? in try cb() } }

        if canBatchedPrefill {
            let initialMask = try allowedTokenMask?()
            // Chat doesn't compute prompt-side logprobs — pass
            // wantFullRows: false so the helper only retains the
            // final row (caps memory at one chunk's worth instead
            // of N × ~512 KB per row).
            let prefill = try runPrefillWithCache(
                inputIds: inputIds, preparedPrompt: preparedPrompt,
                wantFullRows: false,
                skipCache: skipPrefixCache,
                inputIdentity: inputIdentity
            )
            if isCancelled() { throw CancellationError() }
            capturedSnapshotForCache = prefill.capturedSnapshot
            cur = prefill.lastRow.withUnsafeBufferPointer { buf in
                SmeltLogitsSelector.select(
                    logits: buf,
                    position: Int32(inputIds.count - 1),
                    mode: selectionMode,
                    allowedTokenMask: initialMask
                )
            }
            firstGenerationLogitsRow = prefill.lastRow
        } else if canBatchedSelection {
            let prefill = try runSelectedPrefillWithCache(
                inputIds: inputIds,
                preparedPrompt: preparedPrompt,
                selectionMode: selectionMode,
                allowedTokenMask: maskProvider,
                skipCache: skipPrefixCache,
                inputIdentity: inputIdentity
            )
            if isCancelled() { throw CancellationError() }
            cur = prefill.token
            capturedSnapshotForCache = prefill.capturedSnapshot
        } else {
            // Recurrent packages can reuse only a complete exact-prefix
            // checkpoint. Their conv/rec state cannot be rewound to an
            // arbitrary LCP, so a miss resets and evaluates from position zero;
            // a hit restores every persistent state family at its captured
            // position and evaluates only the newly appended suffix.
            let phaseDiagnostics = ProcessInfo.processInfo.environment[
                "SMELT_SERVE_PREFIX_CACHE_DIAGNOSTICS"
            ] == "1"
            var phaseStart = CFAbsoluteTimeGetCurrent()
            runtime.resetWorkingBuffers()
            let resetMS = (CFAbsoluteTimeGetCurrent() - phaseStart) * 1_000
            let cacheMatch = skipPrefixCache
                ? nil : prefixCache?.tryMatch(
                    inputIds,
                    inputIdentity: inputIdentity
                )
            logPrefixCacheLookupIfRequested(skipped: skipPrefixCache)
            var startPosition = 0
            phaseStart = CFAbsoluteTimeGetCurrent()
            if let cacheMatch {
                precondition(
                    cacheMatch.restoreExactly,
                    "sequential prefix reuse requires an exact-position checkpoint"
                )
                switch cacheMatch.chosenSnapshot.snapshot {
                case .host(let snapshot):
                    try runtime.restorePromptSnapshot(snapshot)
                case .device(let snapshot):
                    try runtime.restoreDevicePromptSnapshot(snapshot)
                }
                startPosition = cacheMatch.effectiveLCP
            } else if inputIdentity == .text,
                      let prepared = preparedPrompt,
                      inputIds.count > prepared.tokenIds.count {
                try runtime.restorePromptSnapshot(prepared.snapshot)
                // Replay an automatic capture's aligned tail before the
                // request suffix. The prepared-state loader has already
                // proved that these IDs are the exact contract suffix.
                startPosition = prepared.snapshot.capturedLength
            }
            let restoreMS = (CFAbsoluteTimeGetCurrent() - phaseStart) * 1_000
            cur = 0
            phaseStart = CFAbsoluteTimeGetCurrent()
            for position in startPosition..<inputIds.count {
                if isCancelled() { throw CancellationError() }
                let tokenId = inputIds[position]
                let isLast = position == inputIds.count - 1
                cur = try runtime.decodeStep(
                    tokenId: tokenId,
                    position: Int32(position),
                    selectionMode: isLast ? selectionMode : .argmax,
                    allowedTokenMask: isLast ? maskProvider : nil
                )
            }
            let prefillMS = (CFAbsoluteTimeGetCurrent() - phaseStart) * 1_000
            if prefixCache != nil && !skipPrefixCache
                && prefixCache?.requiresExactRestore != true
            {
                capturedSnapshotForCache = runtime.capturePromptSnapshot(
                    capturedLength: inputIds.count,
                    promptLength: inputIds.count,
                    nextToken: 0
                )
            }
            if phaseDiagnostics {
                fputs(
                    "smelt serve prefix-cache phases route=decode-per-token tokens="
                        + "\(inputIds.count - startPosition) reset_ms="
                        + "\(String(format: "%.1f", resetMS)) restore_ms="
                        + "\(String(format: "%.1f", restoreMS)) prefill_ms="
                        + "\(String(format: "%.1f", prefillMS))\n",
                    stderr
                )
            }
        }
        var pos = inputIds.count

        if mode.skipThought,
           let thinkTok = inference.thinkToken,
           let thinkEnd = inference.thinkEndToken,
           cur == thinkTok
        {
            if isCancelled() { throw CancellationError() }
            // Force-feed thinkEnd to advance the KV cache, then re-sample
            // so cur reflects the post-thought token instead of the stale
            // thinkTok. If a thinkSkipSuffix is configured (an instruct model
            // does this), feed that next. These tokens are NOT included
            // in logprobs.content because they're not part of the user-
            // visible response.
            cur = try runtime.decodeStep(
                tokenId: thinkEnd,
                position: Int32(pos),
                selectionMode: selectionMode
            )
            pos += 1
            // decodeStep refreshed allLogitsHalf with the row at
            // thinkEnd's position, so the stashed boundary row from
            // the batched prefill is now stale.
            firstGenerationLogitsRow = nil
            if let suffix = inference.thinkSkipSuffix {
                if isCancelled() { throw CancellationError() }
                cur = try runtime.decodeStep(
                    tokenId: suffix,
                    position: Int32(pos),
                    selectionMode: selectionMode
                )
                pos += 1
            }
        }
        return PreludeState(
            cur: cur, pos: pos,
            firstGenerationLogitsRow: firstGenerationLogitsRow,
            capturedSnapshotForCache: capturedSnapshotForCache
        )
    }

    /// Non-streaming chat generation. Delegates to the shared
    /// streaming core with an accumulating closure that collects
    /// chunks/tokens/logprobs into the buffered ChatGenerationResult
    /// shape. The core handles prepareForRequest, prelude, stop-
    /// sequence detection + truncation, tail-buffer + UTF-8 boundary
    /// snap, and prefix-cache store — chat-non-streaming just
    /// re-assembles them post-hoc into a buffered result.
    ///
    /// Subtle: when stop-sequence truncation cuts mid-token, the core
    /// ships the visible prefix as an unattributed flush (nil token,
    /// nil logprob), matching the streaming wire where the cutoff
    /// token isn't a real emitted token. The accumulated `logprobs`
    /// can therefore hold one fewer entry than `result.tokens.count`.
    /// OpenAI's spec doesn't pin per-token logprob/token-count
    /// equality, and reusing the streaming core beats maintaining a
    /// second generator.
    private func runChatGeneration(
        inputIds: [Int32],
        preparedPrompt: SmeltPreparedPromptState?,
        maxTokens: Int,
        selectionMode: SmeltSelectionMode,
        logprobsTopK: Int?,
        stopSequences: [String],
        mode: DecodeMode,
        skipPrefixCache: Bool
    ) async throws -> ChatGenerationResult {
        var accumulated = ""
        var logprobs: [LogprobEntry]? = (logprobsTopK != nil) ? [] : nil
        let result = try await runGenerationStreamingCore(
            inputIds: inputIds,
            preparedPrompt: preparedPrompt,
            maxTokens: maxTokens,
            selectionMode: selectionMode,
            logprobsTopK: logprobsTopK,
            stopSequences: stopSequences,
            mode: mode,
            continuingExistingText: false,
            skipPrefixCache: skipPrefixCache,
            onToken: { _, chunk, lp in
                accumulated += chunk
                if let lp { logprobs?.append(lp) }
            }
        )
        return ChatGenerationResult(
            tokens: result.generatedTokens,
            text: accumulated,
            finishReason: result.finishReason,
            logprobs: logprobs
        )
    }

    /// Shared scaffolding for both streaming endpoints:
    /// prepareForRequest + beginStream + body + DONE + end + abort
    /// catch. The body closure receives the live stream handle and
    /// emits chunks via writeChunk. Errors before beginStream return
    /// .complete with 500; errors after surface via stderr and
    /// best-effort end().
    private func runStream(
        contextCapacity: Int,
        requestId: SmeltServeRequestId,
        transport: any SmeltServeTransport,
        extraHeaders: [String: String] = [:],
        body: (_ stream: SmeltServeStreamHandle) async throws -> Void
    ) async -> HandlerResult {
        let stream: SmeltServeStreamHandle
        do {
            try runtime.prepareForRequest(
                batchCapacity: 1, contextCapacity: contextCapacity
            )
            stream = try await transport.beginStream(
                contentType: "text/event-stream",
                requestId: requestId,
                extraHeaders: extraHeaders
            )
        } catch {
            return .complete(OpenAIJSON.errorResponse(
                status: 500, code: .internalError,
                message: "Stream start failed: \(error)"
            ))
        }
        do {
            try await body(stream)
            try await stream.writeChunk(OpenAIJSON.sseDoneFrame)
            try await stream.end()
        } catch {
            fputs("smelt serve: stream aborted: \(error)\n", stderr)
            try? await stream.end()
        }
        return .streamed
    }

    /// Emits the OpenAI-spec trailing usage chunk on a streaming
    /// chat-completions response when the client opted in via
    /// `stream_options.include_usage: true`. The chunk carries
    /// `choices: []` and a populated `usage` block; the surrounding
    /// stream then proceeds to the `[DONE]` sentinel as usual.
    private func emitChatUsageChunkIfNeeded(
        includeUsage: Bool,
        chunkId: String,
        created: Int,
        promptTokens: Int,
        completionTokens: Int,
        stream: SmeltServeStreamHandle
    ) async throws {
        guard includeUsage else { return }
        let chunk = OpenAIChatStreamChunk(
            id: chunkId, object: "chat.completion.chunk",
            created: created, model: modelId,
            choices: [],
            usage: OpenAIUsage(
                promptTokens: promptTokens,
                completionTokens: completionTokens,
                totalTokens: promptTokens + completionTokens
            )
        )
        try await stream.writeChunk(OpenAIJSON.sseFrame(chunk))
    }

    /// Streaming response for a deterministic answer — either a
    /// resume-replay (answer already in message history) or a
    /// direct-resolve (answer mechanically extracted from the prompt
    /// or the last tool result). This path deliberately does not
    /// call prepareForRequest: no model state or Metal device is
    /// needed.
    private func runDeterministicAnswerStream(
        content: String,
        promptTokens: Int,
        completionTokens: Int,
        includeUsage: Bool,
        extraHeaders: [String: String],
        requestId: SmeltServeRequestId,
        transport: any SmeltServeTransport
    ) async -> HandlerResult {
        let chunkId = OpenAIJSON.chatCompletionId()
        let created = Int(Date().timeIntervalSince1970)
        let stream: SmeltServeStreamHandle
        do {
            stream = try await transport.beginStream(
                contentType: "text/event-stream",
                requestId: requestId,
                extraHeaders: extraHeaders
            )
        } catch {
            return .complete(OpenAIJSON.errorResponse(
                status: 500, code: .internalError,
                message: "Stream start failed: \(error)"
            ))
        }

        do {
            func sendChunk(
                delta: OpenAIChatStreamDelta,
                finishReason: OpenAIFinishReason? = nil
            ) async throws {
                let chunk = OpenAIChatStreamChunk(
                    id: chunkId, object: "chat.completion.chunk",
                    created: created, model: modelId,
                    choices: [OpenAIChatStreamChoice(
                        index: 0, delta: delta, logprobs: nil,
                        finishReason: finishReason
                    )]
                )
                try await stream.writeChunk(OpenAIJSON.sseFrame(chunk))
            }

            try await sendChunk(delta: OpenAIChatStreamDelta(
                role: .assistant, content: nil
            ))
            try await sendChunk(delta: OpenAIChatStreamDelta(
                role: nil, content: content
            ))
            try await sendChunk(
                delta: OpenAIChatStreamDelta(role: nil, content: nil),
                finishReason: .stop
            )
            try await emitChatUsageChunkIfNeeded(
                includeUsage: includeUsage,
                chunkId: chunkId, created: created,
                promptTokens: promptTokens,
                completionTokens: completionTokens,
                stream: stream
            )
            try await stream.writeChunk(OpenAIJSON.sseDoneFrame)
            try await stream.end()
        } catch {
            fputs("smelt serve: deterministic answer stream aborted: \(error)\n", stderr)
            try? await stream.end()
        }
        return .streamed
    }

    /// Streaming /v1/chat/completions per OpenAI's stream:true
    /// contract. Sends SSE-encoded `data: {chunk}\n\n` frames over
    /// the transport's chunked-encoding pipe, terminated by
    /// `data: [DONE]\n\n`. The first chunk carries the role-only
    /// delta; subsequent chunks carry content/logprobs; the final
    /// chunk carries `finish_reason` with an empty delta.
    private func runStreamingChat(
        inputIds: [Int32],
        preparedPrompt: SmeltPreparedPromptState?,
        maxTokens: Int,
        selectionMode: SmeltSelectionMode,
        logprobsTopK: Int?,
        stopSequences: [String],
        agentMatcher: SmeltLLGuidanceMatcher?,
        includeUsage: Bool,
        skipPrefixCache: Bool,
        extraHeaders: [String: String],
        requestId: SmeltServeRequestId,
        transport: any SmeltServeTransport
    ) async -> HandlerResult {
        let chunkId = OpenAIJSON.chatCompletionId()
        let created = Int(Date().timeIntervalSince1970)
        return await runStream(
            contextCapacity: inputIds.count + maxTokens,
            requestId: requestId, transport: transport,
            extraHeaders: extraHeaders
        ) { stream in
            func sendChunk(
                delta: OpenAIChatStreamDelta,
                finishReason: OpenAIFinishReason? = nil,
                logprobs: OpenAILogprobs? = nil
            ) async throws {
                let chunk = OpenAIChatStreamChunk(
                    id: chunkId, object: "chat.completion.chunk",
                    created: created, model: modelId,
                    choices: [OpenAIChatStreamChoice(
                        index: 0, delta: delta, logprobs: logprobs,
                        finishReason: finishReason
                    )]
                )
                try await stream.writeChunk(OpenAIJSON.sseFrame(chunk))
            }
            try await sendChunk(delta: OpenAIChatStreamDelta(
                role: .assistant, content: nil
            ))
            let coreResult = try await runGenerationStreamingCore(
                inputIds: inputIds,
                preparedPrompt: preparedPrompt,
                maxTokens: maxTokens,
                selectionMode: selectionMode,
                logprobsTopK: logprobsTopK,
                stopSequences: stopSequences,
                mode: agentMatcher.map(DecodeMode.constrained)
                    ?? .freeText(skipThought: inference.thinkingPolicy != .enabled),
                continuingExistingText: false,
                skipPrefixCache: skipPrefixCache,
                isCancelled: { stream.isCancelled },
                onToken: { _, chunkText, lp in
                    let logprobsField = lp.map { entry in
                        OpenAILogprobs(content: [
                            LogprobsCompute.chatTokenLogprob(entry, tokenizer: tokenizer)
                        ])
                    }
                    try await sendChunk(
                        delta: OpenAIChatStreamDelta(role: nil, content: chunkText),
                        logprobs: logprobsField
                    )
                }
            )
            try await sendChunk(
                delta: OpenAIChatStreamDelta(role: nil, content: nil),
                finishReason: coreResult.finishReason
            )
            try await emitChatUsageChunkIfNeeded(
                includeUsage: includeUsage,
                chunkId: chunkId, created: created,
                promptTokens: inputIds.count,
                completionTokens: coreResult.generatedTokens.count,
                stream: stream
            )
        }
    }

    /// Streaming /v1/chat/completions when tools are active. JSON-native calls
    /// stream their name and argument bytes incrementally. Package-native XML
    /// calls expose each schema-bound function name as soon as its opening tag
    /// closes, then emit canonical arguments after the completed call validates.
    ///
    /// `useUnion` selects between strict (required/specific) and union
    /// (auto) semantics for the matcher; see DecodeMode docs. In union
    /// mode the model may emit free text instead of a tool call, in
    /// which case we stream text deltas with finish_reason="stop".
    private func runStreamingChatTools(
        inputIds: [Int32],
        preparedPrompt: SmeltPreparedPromptState?,
        maxTokens: Int,
        selectionMode: SmeltSelectionMode,
        toolMatcher: SmeltLLGuidanceMatcher,
        useUnion: Bool,
        toolDescriptors: [SmeltToolDescriptor],
        logprobsTopK: Int?,
        stopSequences: [String],
        includeUsage: Bool,
        skipPrefixCache: Bool,
        extraHeaders: [String: String],
        requestId: SmeltServeRequestId,
        transport: any SmeltServeTransport
    ) async -> HandlerResult {
        let chunkId = OpenAIJSON.chatCompletionId()
        let created = Int(Date().timeIntervalSince1970)
        return await runStream(
            contextCapacity: inputIds.count + maxTokens,
            requestId: requestId, transport: transport,
            extraHeaders: extraHeaders
        ) { stream in
            // Logprobs are only meaningful on the text arm of union
            // mode. The JSON tool-call arm's per-token logprobs are
            // not part of the OpenAI streaming tool_call delta shape,
            // so they're not emitted for tool_call deltas.
            func sendChunk(
                delta: OpenAIChatStreamDelta,
                logprobs: OpenAILogprobs? = nil,
                finishReason: OpenAIFinishReason? = nil
            ) async throws {
                let chunk = OpenAIChatStreamChunk(
                    id: chunkId, object: "chat.completion.chunk",
                    created: created, model: modelId,
                    choices: [OpenAIChatStreamChoice(
                        index: 0, delta: delta, logprobs: logprobs,
                        finishReason: finishReason
                    )]
                )
                try await stream.writeChunk(OpenAIJSON.sseFrame(chunk))
            }

            // Union mode disambiguates the arm by what the model
            // emits. The `<|tool_call>` start token (an instruct model's
            // native trained shape) is `special: true` and so
            // chunkText is empty when it arrives — we detect it via
            // token ID and switch to .nativeBuffered. Otherwise the
            // first non-whitespace character decides: `{` commits to
            // the JSON arm, anything else commits to the text arm.
            // Pure whitespace chunks before any decision defer the
            // arm (all arms remain consistent under the lark grammar).
            // Strict mode is always tool-arm.
            //
            // JSON tool-arm emission is incremental. Package-native XML emits
            // each schema-bound function name as soon as its opener closes;
            // parameter values remain buffered until the complete call passes
            // schema validation, then canonical arguments are emitted.
            var accumulated = ""
            // Buffer per-token logprobs received while the union arm
            // is still undecided. The text arm commits on the first
            // non-whitespace char and ships `accumulated` (which may
            // span multiple deferred whitespace tokens) as a single
            // content delta — the OpenAI streaming schema's
            // `logprobs.content` is an array per chunk, so each
            // accumulated token's lp can travel with that delta.
            // Deltas for tool/native arms don't carry content
            // logprobs (per OpenAI's tool-call delta shape), so on
            // those commits the buffered entries are discarded.
            var pendingLogprobs: [LogprobEntry] = []
            enum Arm { case undecided, tool, nativeBuffered, text }
            var arm: Arm = useUnion
                ? .undecided
                : (toolTranscriptCodec != nil ? .nativeBuffered : .tool)
            let toolArm = ToolArmStreamState()
            let nativeToolArm = toolTranscriptCodec?.makeStreamDecoder(
                toolNames: toolDescriptors.map(\.name)
            )
            var nativeCallIDs: [String] = []
            var nativeRoleEmitted = false
            let nativeGenerationStart = CFAbsoluteTimeGetCurrent()
            func emitNativeRoleIfNeeded() async throws {
                guard toolTranscriptCodec != nil, !nativeRoleEmitted else { return }
                nativeRoleEmitted = true
                try await sendChunk(delta: OpenAIChatStreamDelta(
                    role: .assistant,
                    content: nil
                ))
                if ProcessInfo.processInfo.environment[
                    "SMELT_SERVE_PREFIX_CACHE_DIAGNOSTICS"
                ] == "1" {
                    let elapsedMS = (
                        CFAbsoluteTimeGetCurrent() - nativeGenerationStart
                    ) * 1_000
                    fputs(
                        "smelt serve tool-stream first_token_ms="
                            + "\(String(format: "%.1f", elapsedMS))\n",
                        stderr
                    )
                }
            }
            func advanceNativeToolArm(_ text: String) async throws {
                guard let nativeToolArm else { return }
                for start in try nativeToolArm.consume(text) {
                    if let leadingText = start.leadingText, !leadingText.isEmpty {
                        try await sendChunk(delta: OpenAIChatStreamDelta(
                            role: nativeRoleEmitted ? nil : .assistant,
                            content: leadingText
                        ))
                        nativeRoleEmitted = true
                    }
                    let id = SmeltToolCallID.next()
                    nativeCallIDs.append(id)
                    try await sendChunk(delta: OpenAIChatStreamDelta(
                        role: nativeRoleEmitted ? nil : .assistant,
                        toolCalls: [OpenAIChatStreamToolCallDelta(
                            index: start.index,
                            id: id,
                            type: .function,
                            function: OpenAIChatStreamFunctionDelta(
                                name: start.name,
                                arguments: ""
                            )
                        )]
                    ))
                    nativeRoleEmitted = true
                }
            }
            let nativeStartId = nativeToolCallStartTokenId
            let coreResult = try await runGenerationStreamingCore(
                inputIds: inputIds,
                preparedPrompt: preparedPrompt,
                maxTokens: maxTokens,
                selectionMode: selectionMode,
                logprobsTopK: logprobsTopK,
                stopSequences: stopSequences,
                mode: useUnion
                    ? .constrainedUnion(toolMatcher)
                    : .constrained(toolMatcher),
                continuingExistingText: false,
                skipPrefixCache: skipPrefixCache,
                isCancelled: { stream.isCancelled },
                onToken: { token, chunkText, lp in
                    accumulated += chunkText
                    if arm == .undecided {
                        if let lp { pendingLogprobs.append(lp) }
                    }
                    if arm == .undecided, let nativeStartId,
                       token == nativeStartId
                    {
                        arm = .nativeBuffered
                        pendingLogprobs.removeAll()
                        return
                    }
                    if arm == .undecided {
                        guard let first = accumulated.first(
                            where: { !$0.isWhitespace }
                        ) else { return }
                        if let toolTranscriptCodec,
                           toolTranscriptCodec.couldBeginToolCall(
                            firstSignificantCharacter: first
                           ) {
                            arm = .nativeBuffered
                            pendingLogprobs.removeAll()
                            try await emitNativeRoleIfNeeded()
                            try await advanceNativeToolArm(accumulated)
                            return
                        }
                        if first == "{" {
                            arm = .tool
                            pendingLogprobs.removeAll()
                            // Feed the whole accumulated buffer so the
                            // tool-arm scanner sees the leading
                            // whitespace prefix it tolerates.
                            try await toolArm.advance(
                                chunkText: accumulated,
                                sendDelta: { delta in
                                    try await sendChunk(delta: delta)
                                }
                            )
                            return
                        }
                        arm = .text
                        try await sendChunk(delta: OpenAIChatStreamDelta(
                            role: .assistant, content: nil
                        ))
                        let logprobsField: OpenAILogprobs? = pendingLogprobs.isEmpty
                            ? nil
                            : OpenAILogprobs(content: pendingLogprobs.map {
                                LogprobsCompute.chatTokenLogprob(
                                    $0, tokenizer: tokenizer
                                )
                            })
                        pendingLogprobs.removeAll()
                        try await sendChunk(
                            delta: OpenAIChatStreamDelta(
                                content: accumulated
                            ),
                            logprobs: logprobsField
                        )
                        return
                    }
                    // Post-commit text-arm: emit every callback that
                    // carries either a non-empty chunk or a logprob
                    // entry. Dropping empty-text callbacks would lose
                    // per-token lp accounting for special tokens that
                    // decode to zero bytes (the `bytes` field in
                    // OpenAI's logprob schema exists precisely for
                    // this multi-byte-token case).
                    if arm == .text, !chunkText.isEmpty || lp != nil {
                        let logprobsField = lp.map { entry in
                            OpenAILogprobs(content: [
                                LogprobsCompute.chatTokenLogprob(
                                    entry, tokenizer: tokenizer
                                )
                            ])
                        }
                        try await sendChunk(
                            delta: OpenAIChatStreamDelta(
                                content: chunkText
                            ),
                            logprobs: logprobsField
                        )
                        return
                    }
                    if arm == .nativeBuffered {
                        try await emitNativeRoleIfNeeded()
                        try await advanceNativeToolArm(chunkText)
                        return
                    }
                    if arm == .tool {
                        try await toolArm.advance(
                            chunkText: chunkText,
                            sendDelta: { delta in
                                try await sendChunk(delta: delta)
                            }
                        )
                    }
                }
            )

            let isToolCallShape =
                coreResult.finishReason == .toolCalls
                || (useUnion && (arm == .tool || arm == .nativeBuffered))
            if isToolCallShape {
                // Validate the call parses, even though the JSON path
                // already streamed name+args; a parse failure means
                // truncation (max_tokens before close).
                let calls: [OpenAIToolCall]
                let leadingContent: String?
                do {
                    if arm == .nativeBuffered {
                        if toolTranscriptCodec != nil {
                            let decoded = try decodeNativeTranscriptToolCalls(
                                accumulated,
                                descriptors: toolDescriptors
                            )
                            calls = decoded.calls
                            leadingContent = decoded.content
                        } else {
                            calls = [try decodeNativeToolCall(
                                accumulated,
                                allowedToolNames: Set(
                                    toolDescriptors.map { $0.name }
                                )
                            )]
                            leadingContent = nil
                        }
                    } else {
                        calls = [try decodeGeneratedToolCall(accumulated)]
                        leadingContent = nil
                    }
                } catch {
                    try await Self.writeToolCallFailureFrame(
                        stream: stream,
                        message: "Tool-call did not decode: \(error) "
                            + "(finish_reason=\(coreResult.finishReason.rawValue))",
                        partialJson: accumulated,
                        generatedTokenIds: coreResult.generatedTokens
                    )
                    try await emitChatUsageChunkIfNeeded(
                        includeUsage: includeUsage,
                        chunkId: chunkId, created: created,
                        promptTokens: inputIds.count,
                        completionTokens: coreResult.generatedTokens.count,
                        stream: stream
                    )
                    return
                }
                if arm == .nativeBuffered && !nativeCallIDs.isEmpty {
                    guard nativeCallIDs.count == calls.count else {
                        throw SmeltServeError(
                            "streamed \(nativeCallIDs.count) function starts "
                                + "but decoded \(calls.count) complete calls"
                        )
                    }
                    for (index, call) in calls.enumerated() {
                        try await sendChunk(delta: OpenAIChatStreamDelta(
                            toolCalls: [OpenAIChatStreamToolCallDelta(
                                index: index,
                                id: nil,
                                type: nil,
                                function: OpenAIChatStreamFunctionDelta(
                                    name: nil,
                                    arguments: call.function.arguments
                                )
                            )]
                        ))
                    }
                } else if !toolArm.initialEmitted {
                    // Defensive fallback for a completed shape whose
                    // incremental recognizer never emitted a function start.
                    if let leadingContent, !leadingContent.isEmpty {
                        try await sendChunk(delta: OpenAIChatStreamDelta(
                            role: .assistant,
                            content: leadingContent
                        ))
                    }
                    try await sendChunk(delta: OpenAIChatStreamDelta(
                        role: .assistant,
                        toolCalls: calls.enumerated().map { index, call in
                            OpenAIChatStreamToolCallDelta(
                                index: index, id: call.id, type: call.type,
                                function: OpenAIChatStreamFunctionDelta(
                                    name: call.function.name,
                                    arguments: call.function.arguments
                                )
                            )
                        }
                    ))
                }
                try await sendChunk(
                    delta: OpenAIChatStreamDelta(),
                    finishReason: .toolCalls
                )
            } else if useUnion {
                // Flush deferred text. If the stream ended with arm
                // still .undecided, the buffer was either pure
                // whitespace or a `.partial` native-prefix match
                // (e.g. `<|tool_ca` truncated by max_tokens). Treat
                // it as text from the client's perspective so the
                // raw bytes aren't silently dropped.
                if arm == .undecided, !accumulated.isEmpty {
                    try await sendChunk(delta: OpenAIChatStreamDelta(
                        role: .assistant, content: nil
                    ))
                    try await sendChunk(delta: OpenAIChatStreamDelta(
                        content: accumulated
                    ))
                }
                try await sendChunk(
                    delta: OpenAIChatStreamDelta(),
                    finishReason: coreResult.finishReason
                )
            } else {
                try await Self.writeToolCallFailureFrame(
                    stream: stream,
                    message: "Tool-call generation did not complete: "
                        + "matcher did not accept within max_tokens "
                        + "(finish_reason=\(coreResult.finishReason.rawValue))",
                    partialJson: accumulated,
                    generatedTokenIds: coreResult.generatedTokens
                )
                try await emitChatUsageChunkIfNeeded(
                    includeUsage: includeUsage,
                    chunkId: chunkId, created: created,
                    promptTokens: inputIds.count,
                    completionTokens: coreResult.generatedTokens.count,
                    stream: stream
                )
                return
            }

            try await emitChatUsageChunkIfNeeded(
                includeUsage: includeUsage,
                chunkId: chunkId, created: created,
                promptTokens: inputIds.count,
                completionTokens: coreResult.generatedTokens.count,
                stream: stream
            )
        }
    }

    /// Per-token streaming generator shared by chat and legacy
    /// completions endpoints. Yields each generated token via an
    /// async callback so streaming handlers can emit SSE frames as
    /// the model generates.
    ///
    /// Variances handled via parameters:
    /// - `logprobsTopK == nil` skips the per-token logprob compute.
    /// - `continuingExistingText: true` configures the streaming
    ///   decoder NOT to strip the first leading space — appropriate
    ///   when the generated text continues a non-empty prefix
    ///   (legacy completions echoes the prompt's tokenization
    ///   context). Defaults false for chat (assistant starts a fresh
    ///   response after the user turn).
    ///
    /// The callback receives (token, decoded chunk text, optional
    /// logprob entry). It can throw to abort generation (e.g.,
    /// client disconnect triggers writeChunk to throw).
    ///
    /// onToken fires once per generated token, in order, each paired
    /// with that token's own logprob — even when a tail reserve
    /// delays the bytes (the token is buffered and drained once its
    /// bytes clear the reserve). A nil `token` signals an
    /// unattributed flush: visible bytes shipped without a token
    /// identity or logprob. The only such case today is the text
    /// before a stop-sequence cutoff when the cutoff lands mid-token
    /// — that prefix is shown, but the truncated token is not a real
    /// emitted token.
    ///
    /// prepareForRequest runs per-call so n>1 streaming iterations
    /// start from a fresh runtime state on cache misses.
    ///
    /// Returns `(finishReason, generatedTokens)`. The generated-
    /// tokens list is tracked independently of onToken: a token whose
    /// chunk decodes to empty bytes and carries no requested logprob
    /// produces no onToken call, and a mid-token stop cutoff adds a
    /// nil-token flush. Buffered callers use the returned list for
    /// usage.completion_tokens accounting; streaming callers can
    /// ignore it.
    private struct StreamingCoreResult {
        let finishReason: OpenAIFinishReason
        let generatedTokens: [Int32]
    }
    private func runGenerationStreamingCore(
        inputIds: [Int32],
        inputIdentity: SmeltPromptInputIdentity = .text,
        preparedPrompt: SmeltPreparedPromptState? = nil,
        maxTokens: Int,
        selectionMode: SmeltSelectionMode,
        logprobsTopK: Int?,
        stopSequences: [String],
        mode: DecodeMode,
        continuingExistingText: Bool,
        skipPrefixCache: Bool,
        isCancelled: () -> Bool = { false },
        onToken: (_ token: Int32?, _ chunk: String, _ logprob: LogprobEntry?)
            async throws -> Void
    ) async throws -> StreamingCoreResult {
        if isCancelled() { throw CancellationError() }
        try runtime.prepareForRequest(
            batchCapacity: 1, contextCapacity: inputIds.count + maxTokens
        )
        // Apply the min-tokens EOS gate ONLY in free-text mode. The
        // union-grammar text arm encodes the 3-char floor structurally
        // (see SmeltToolGrammar.larkUnion) so the matcher
        // composes the constraint with the rest of the grammar; the
        // strict-tool path auto-terminates on JSON close and doesn't
        // need a floor.
        let minTokensGate: MinTokensEOSGate? = {
            switch mode {
            case .freeText:
                return MinTokensEOSGate(
                    runtimeEOSTokens: inference.eosTokens,
                    vocabSize: tokenizer.vocabularySize
                )
            case .constrainedUnion, .constrained:
                return nil
            }
        }()
        // When the gate is inactive in free-text mode (threshold met),
        // gate.computeMask() returns nil and we need a no-constraint
        // mask to feed the runtime. Precompute the full-allow mask
        // once so per-step allocation stays cheap.
        let modeCallback = mode.allowedTokenMaskCallback
        let fullAllowMask: [UInt32]? = {
            guard minTokensGate != nil, modeCallback == nil else { return nil }
            let size = tokenizer.vocabularySize
            let words = (size + 31) / 32
            var m = Array(repeating: UInt32.max, count: words)
            let tail = size % 32
            if tail != 0 { m[words - 1] = (UInt32(1) << tail) - 1 }
            return m
        }()
        let maskCallback: (() throws -> [UInt32])?
        if let gate = minTokensGate {
            maskCallback = {
                if let gated = try gate.computeMask() { return gated }
                if let fallback = modeCallback { return try fallback() }
                return fullAllowMask ?? []
            }
        } else {
            maskCallback = modeCallback
        }
        // After the gate latches inactive in free-text mode (no
        // mode-side mask either), we can drop the maskCallback for
        // the rest of generation so the runtime's argmax fast-path
        // (`allowedTokenMask == nil && usesArgmaxFastPath`) kicks
        // back in. Without this, every remaining decode step
        // materializes and applies the precomputed fullAllowMask,
        // disabling the Metal argmax shortcut for the bulk of the
        // response.
        let gateBypassable = minTokensGate != nil && modeCallback == nil
        let prelude = try runPrelude(
            inputIds: inputIds,
            inputIdentity: inputIdentity,
            preparedPrompt: preparedPrompt,
            selectionMode: selectionMode,
            mode: mode,
            skipPrefixCache: skipPrefixCache,
            allowedTokenMask: maskCallback,
            isCancelled: isCancelled
        )
        var cur = prelude.cur
        var pos = prelude.pos
        var firstGenerationLogitsRow = prelude.firstGenerationLogitsRow
        let capturedSnapshotForCache = prelude.capturedSnapshotForCache

        let eosTokens = Set(inference.eosTokens)
        var streamingDecoder = SmeltStreamingTokenDecoder(
            continuingExistingText: continuingExistingText
        )
        var generatedTokens: [Int32] = []
        var finishReason: OpenAIFinishReason = .length
        // Tail-buffer for stop-sequence detection: we accumulate
        // generated text but only emit the prefix that's safe to ship
        // (no partial stop-sequence match can still complete). The
        // tail-buffer size = max(stopSequenceLength) - 1; smaller
        // values would let a stop sequence straddle the emitted/
        // buffered boundary undetected.
        let tailReserve = stopSequences.lazy.map { $0.utf8.count }.max().map { $0 - 1 } ?? 0
        let streamStopWindow = stopSequences.lazy.map(\.count).max() ?? 0
        var accumulated = ""
        var emittedByteOffset = 0
        // onToken fires once per *emit*, not once per token: with a
        // tail reserve, the bytes that clear the reserve in a given
        // step can span several earlier tokens. Buffer each token's
        // byte span + logprob and drain whole tokens once their bytes
        // are safe, so every onToken call carries exactly one token's
        // text paired with that token's logprob. HF/vLLM/TGI keep
        // per-token logprobs aligned to the token text they describe;
        // coalescing tokens into one emit with a single logprob
        // misaligns logprobs.content on the stop+stream+logprobs path.
        struct PendingTokenEmit {
            let token: Int32
            let startByte: Int
            let endByte: Int
            let logprob: LogprobEntry?
        }
        var pending: [PendingTokenEmit] = []
        var stoppedByStopSequence = false
        var completedCanonicalResponse = false
        var processedGeneratedTokens = 0
        func drainSafeTokens(upTo safeEnd: Int) async throws {
            while let first = pending.first, first.endByte <= safeEnd {
                let from = max(emittedByteOffset, first.startByte)
                let text = sliceUTF8(
                    of: accumulated, fromByte: from, toByte: first.endByte
                )
                if !text.isEmpty || first.logprob != nil {
                    try await onToken(first.token, text, first.logprob)
                }
                emittedByteOffset = max(emittedByteOffset, first.endByte)
                pending.removeFirst()
            }
        }
        while generatedTokens.count < maxTokens {
            if isCancelled() { throw CancellationError() }
            if eosTokens.contains(cur) {
                finishReason = .stop
                completedCanonicalResponse = true
                break
            }
            var lp: LogprobEntry? = nil
            if let topK = logprobsTopK {
                lp = consumeFirstGenerationLogprob(
                    row: &firstGenerationLogitsRow,
                    chosenToken: cur, topK: topK
                )
            }
            let chunkText = visibleGeneratedControlTokenText[cur]
                ?? streamingDecoder.decode(tokenId: cur, tokenizer: tokenizer)
            minTokensGate?.observe(decodedText: chunkText)
            generatedTokens.append(cur)
            let tokenStartByte = accumulated.utf8.count
            accumulated += chunkText
            pending.append(PendingTokenEmit(
                token: cur,
                startByte: tokenStartByte,
                endByte: accumulated.utf8.count,
                logprob: lp
            ))

            // Stop-sequence scan over the accumulated text — once a
            // match appears, truncate at the match start and emit
            // only the bytes up to the cutoff (the user explicitly
            // asked not to see the stop text or anything after it).
            // Bound the scan window to the new chunk plus the longest
            // stop length (matches the non-streaming sibling's
            // `chunk.count + stopWindow` bound) so a single request
            // doesn't grow scan cost as O(N²) over a long accumulated
            // buffer.
            if !stopSequences.isEmpty,
               let cutoff = matchedStopSequenceRange(
                   in: accumulated,
                   sequences: stopSequences,
                   scanLen: chunkText.count + streamStopWindow
               )
            {
                accumulated = String(accumulated[..<cutoff.lowerBound])
                finishReason = .stop
                stoppedByStopSequence = true
                let safeEnd = accumulated.utf8.count
                try await drainSafeTokens(upTo: safeEnd)
                // A token straddling the cutoff is truncated: ship its
                // visible prefix without a logprob (the token was cut,
                // so it isn't a fully-emitted token on the wire). nil
                // token marks the bytes as an unattributed flush.
                if safeEnd > emittedByteOffset {
                    let partial = sliceUTF8(
                        of: accumulated,
                        fromByte: emittedByteOffset, toByte: safeEnd
                    )
                    if !partial.isEmpty {
                        try await onToken(nil, partial, nil)
                    }
                    emittedByteOffset = safeEnd
                }
                break
            }

            // Drain whole tokens whose bytes have cleared the tail
            // reserve (which might still complete a stop match in the
            // next token). safeEnd is only a threshold — drainSafeTokens
            // slices at token endByte boundaries, never at safeEnd — so
            // it needn't land on a codepoint boundary. With no stop
            // sequences tailReserve=0 and every token drains at once.
            let safeEnd = max(emittedByteOffset, accumulated.utf8.count - tailReserve)
            try await drainSafeTokens(upTo: safeEnd)
            // Advance the constrained-decoding matcher AFTER the
            // emit so the user sees the token even if it completes
            // the JSON. Strict mode auto-terminates on accept (a
            // closed JSON object). Union mode stays open-ended:
            // the matcher mask, plus the union EOS gate for short
            // text responses, lets the model's eventual EOS end
            // the loop via the eosTokens branch above.
            if let matcher = mode.matcher {
                try matcher.consume(tokenIds: [cur])
                if matcher.isAccepting && mode.autoTerminatesOnAccept {
                    finishReason = .toolCalls
                    completedCanonicalResponse = true
                    break
                }
            }
            let stepMask: SmeltAllowedTokenMaskProvider?
            if gateBypassable, let gate = minTokensGate, !gate.isActive {
                stepMask = nil
            } else {
                stepMask = maskCallback.map { cb in
                    { () throws -> [UInt32]? in try cb() }
                }
            }
            cur = try runtime.decodeStep(
                tokenId: cur, position: Int32(pos),
                selectionMode: selectionMode,
                allowedTokenMask: stepMask
            )
            pos += 1
            processedGeneratedTokens += 1
        }
        // Final flush: drain the tail-reserve tokens held back during
        // the loop, each with its own logprob. EOS and max_tokens both
        // reach here and must flush; only a stop-sequence truncation
        // skips it (that branch already drained up to its cutoff).
        // Gate on stoppedByStopSequence, not finishReason — EOS also
        // sets .stop, so a finishReason check would drop the last
        // tailReserve bytes on the common EOS exit.
        if !stoppedByStopSequence {
            try await drainSafeTokens(upTo: accumulated.utf8.count)
        }

        storeColdPrefixIfPossible(
            inputIds: inputIds,
            snapshot: capturedSnapshotForCache,
            inputIdentity: inputIdentity
        )
        if completedCanonicalResponse && !stoppedByStopSequence {
            try storeCompletedAssistantTurnIfPossible(
                inputIds: inputIds,
                inputIdentity: inputIdentity,
                generatedTokens: generatedTokens,
                processedGeneratedTokens: processedGeneratedTokens,
                nextPosition: &pos
            )
        }
        return StreamingCoreResult(
            finishReason: finishReason, generatedTokens: generatedTokens
        )
    }

    /// Advance an exact successful execution through the package transcript's
    /// assistant-turn closure and cache the resulting all-state checkpoint.
    /// Future requests only use it when these exact token IDs are their prefix,
    /// so re-tokenization drift or a differently rendered history becomes a
    /// clean miss rather than a recurrent-state correctness risk.
    private func storeCompletedAssistantTurnIfPossible(
        inputIds: [Int32],
        inputIdentity: SmeltPromptInputIdentity,
        generatedTokens: [Int32],
        processedGeneratedTokens: Int,
        nextPosition: inout Int
    ) throws {
        guard let cache = prefixCache,
              cache.requiresExactRestore,
              let toolTranscriptCodec
        else { return }

        var processed = processedGeneratedTokens
        while processed < generatedTokens.count {
            _ = try runtime.decodeStep(
                tokenId: generatedTokens[processed],
                position: Int32(nextPosition),
                selectionMode: .argmax
            )
            processed += 1
            nextPosition += 1
        }

        let closure = try toolTranscriptCodec.completedAssistantTurnClosure(
            tokenizer: tokenizer
        )
        for token in closure {
            _ = try runtime.decodeStep(
                tokenId: token,
                position: Int32(nextPosition),
                selectionMode: .argmax
            )
            nextPosition += 1
        }
        let completeTokens = inputIds + generatedTokens + closure
        if ProcessInfo.processInfo.environment[
            "SMELT_SERVE_PROMPT_DIAGNOSTICS"
        ] == "ids" {
            let generatedText = generatedTokens.map { String($0) }
                .joined(separator: ",")
            let closureText = closure.map { String($0) }.joined(separator: ",")
            let diagnostic = "smelt serve completion generated_ids="
                + generatedText + " closure_ids=" + closureText + "\n"
            fputs(
                diagnostic, stderr
            )
        }
        let snapshot = try runtime.captureDevicePromptSnapshot(
            capturedLength: completeTokens.count
        )
        cache.store(SmeltPromptStateCacheEntry(
            tokens: completeTokens,
            inputIdentity: inputIdentity,
            snapshots: [SmeltPromptStateCheckpoint(
                position: completeTokens.count,
                snapshot: .device(snapshot)
            )]
        ))
    }

    /// Pick the EARLIEST stop-sequence match in `text`, considering
    /// all sequences. Returns nil if none match. Earliest, not
    /// first-listed, so `stop=["\nclass","\ndef"]` on
    /// `"...foo\ndef bar\nclass baz..."` truncates at `\ndef` (the
    /// earlier match), not whichever stop was listed first.
    /// Slice `text` by UTF-8 byte offsets. Returns the empty string
    /// if the requested range crosses a UTF-8 codepoint boundary.
    /// The streaming drain always passes token-boundary offsets, so
    /// the boundary guard is a safety net rather than the common path.
    private func sliceUTF8(
        of text: String, fromByte start: Int, toByte end: Int
    ) -> String {
        guard start <= end, end <= text.utf8.count else { return "" }
        let utf8 = text.utf8
        let startIdx = utf8.index(utf8.startIndex, offsetBy: start)
        let endIdx = utf8.index(utf8.startIndex, offsetBy: end)
        return String(utf8[startIdx..<endIdx]) ?? ""
    }

    /// Streaming /v1/completions. Sequentially streams n choices on
    /// one SSE response: choice 0 fully streams, then choice 1, etc.
    /// One `data: [DONE]\n\n` at the very end. With temperature ≤ 0
    /// the n-replicate optimization from non-streaming applies: run
    /// once and clone the chunks across the remaining indices.
    private func runStreamingCompletions(
        inputIds: [Int32],
        maxTokens: Int,
        temperature: Double?,
        topK: Int?,
        topP: Double?,
        seed: Int?,
        stopSequences: [String],
        n: Int,
        requestId: SmeltServeRequestId,
        transport: any SmeltServeTransport
    ) async -> HandlerResult {
        let chunkId = OpenAIJSON.completionId()
        let created = Int(Date().timeIntervalSince1970)
        return await runStream(
            contextCapacity: inputIds.count + maxTokens,
            requestId: requestId, transport: transport
        ) { stream in
            func sendChunk(
                index: Int, text: String,
                finishReason: OpenAIFinishReason? = nil
            ) async throws {
                let chunk = OpenAICompletionStreamChunk(
                    id: chunkId, object: "text_completion",
                    created: created, model: modelId,
                    choices: [OpenAICompletionStreamChoice(
                        index: index, text: text, finishReason: finishReason
                    )]
                )
                try await stream.writeChunk(OpenAIJSON.sseFrame(chunk))
            }

            // Deterministic-mode replication: when temperature ≤ 0 the
            // generator is byte-for-byte reproducible. Run the model
            // ONCE, buffer the emitted chunk texts + final
            // finish_reason, then replay those exact chunks for
            // choices 1..n-1 at the SSE layer (no model work).
            // Matches the non-streaming clone-the-choice optimization.
            // Buffer size: maxTokens × a few bytes per chunk ≈ a few
            // KB per request. Trivial.
            let isDeterministic = (temperature ?? 0) <= 0
            if isDeterministic {
                var bufferedChunks: [String] = []
                let selectionMode = resolveSelectionMode(
                    temperature: temperature, seed: seed,
                    topK: topK, topP: topP
                )
                let coreResult = try await runGenerationStreamingCore(
                    inputIds: inputIds,
                    maxTokens: maxTokens,
                    selectionMode: selectionMode,
                    logprobsTopK: nil,
                    stopSequences: stopSequences,
                    mode: .freeText(skipThought: false),
                    continuingExistingText: true,
                    skipPrefixCache: false,
                    isCancelled: { stream.isCancelled },
                    onToken: { _, chunkText, _ in
                        bufferedChunks.append(chunkText)
                        try await sendChunk(index: 0, text: chunkText)
                    }
                )
                let finish = coreResult.finishReason
                try await sendChunk(index: 0, text: "", finishReason: finish)
                for index in 1 ..< n {
                    for chunkText in bufferedChunks {
                        try await sendChunk(index: index, text: chunkText)
                    }
                    try await sendChunk(
                        index: index, text: "", finishReason: finish
                    )
                }
            } else {
                for index in 0 ..< n {
                    let perChoiceSeed: Int? = seed.map { $0 &+ index }
                    let selectionMode = resolveSelectionMode(
                        temperature: temperature, seed: perChoiceSeed,
                        topK: topK, topP: topP
                    )
                    let coreResult = try await runGenerationStreamingCore(
                        inputIds: inputIds,
                        maxTokens: maxTokens,
                        selectionMode: selectionMode,
                        logprobsTopK: nil,
                        stopSequences: stopSequences,
                        mode: .freeText(skipThought: false),
                        continuingExistingText: true,
                        skipPrefixCache: false,
                        isCancelled: { stream.isCancelled },
                        onToken: { _, chunkText, _ in
                            try await sendChunk(index: index, text: chunkText)
                        }
                    )
                    try await sendChunk(
                        index: index, text: "", finishReason: coreResult.finishReason
                    )
                }
            }
        }
    }

}

private func resolveSelectionMode(
    temperature: Double?,
    seed: Int?,
    topK: Int? = nil,
    topP: Double? = nil
) -> SmeltSelectionMode {
    if let t = temperature, t > 0 {
        // Reinterpret the Int bit-pattern as UInt64 so negative seeds
        // (legitimately produced by `seed &+ index` wrap on n>1 when
        // the base seed is close to Int.max) don't trap the
        // `UInt64(seed)` initializer on out-of-range values.
        let s: UInt64
        if let seed {
            s = UInt64(bitPattern: Int64(seed))
        } else {
            s = UInt64(bitPattern: Int64(Int.random(in: 0..<Int.max)))
        }
        if topK != nil || (topP ?? 1) < 1 {
            return .filteredTemperature(
                Float(t),
                topK: topK,
                topP: Float(topP ?? 1),
                seed: s
            )
        }
        return .temperature(Float(t), seed: s)
    }
    return .argmax
}

private func invalidSamplingParameters(
    temperature: Double?,
    topK: Int?,
    topP: Double?
) -> String? {
    if let temperature,
       !temperature.isFinite || temperature < 0 {
        return "temperature must be finite and non-negative"
    }
    if let topK, topK <= 0 {
        return "top_k must be positive"
    }
    if let topP,
       !topP.isFinite || topP <= 0 || topP > 1 {
        return "top_p must be in (0, 1]"
    }
    return nil
}
