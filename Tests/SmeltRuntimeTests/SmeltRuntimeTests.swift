import Foundation
import Testing
import SmeltSchema
@testable import SmeltRuntime

@Test func agentEventRoundTripsAsStableJSON() throws {
    let metrics = SmeltMetrics(
        promptTokens: 12,
        generatedTokens: 3,
        prefillTimeMs: 1.5,
        generateTimeMs: 9.0,
        tokensPerSecond: 333.0,
        snapshotBytes: 4096
    )
    let event = SmeltEvent(
        type: .metrics,
        sessionId: "session",
        generationId: "generation",
        traceId: "trace",
        metrics: metrics,
        timestampUs: 42
    )

    let data = try JSONEncoder().encode(event)
    let decoded = try JSONDecoder().decode(SmeltEvent.self, from: data)

    #expect(decoded == event)
    #expect(decoded.metrics?.snapshotBytes == 4096)
}

@Test func agentDecodingPolicyRoundTripsThroughEventAndTrace() throws {
    let policy = SmeltDecodingPolicy(
        name: "balanced",
        phase: "assistant_text",
        sampler: "temperature",
        temperature: 0.27,
        seed: "123456",
        source: "PI_SMELT_TEXT_TEMPERATURE",
        reason: "policy gate",
        contextPressure: SmeltContextPressure(
            contextLimit: 1024,
            promptTokens: 48,
            estimatedPromptTokens: 52,
            requestedMaxTokens: 64,
            resolvedMaxTokens: 64,
            effectiveMaxTokens: 64,
            toolCallMinTokens: 64,
            messageCount: 1,
            availableInputTokens: 976,
            availableOutputTokens: 975,
            pressureRatio: 0.046875,
            action: "none",
            reason: "runtime context has room for the requested output budget"
        ),
        signals: SmeltPolicySignals(
            requestedName: "balanced",
            resolvedName: "balanced",
            phase: "assistant_text",
            intent: "general",
            contextPressureBand: "normal",
            hasTools: false,
            constrainedOutput: false,
            explicitTemperature: false,
            textTemperature: true,
            toolTemperature: false,
            seeded: true
        )
    )
    let event = SmeltEvent(
        type: .textStart,
        sessionId: "session",
        generationId: "generation",
        traceId: "trace",
        decodingPolicy: policy,
        timestampUs: 42
    )

    let eventData = try JSONEncoder().encode(event)
    let decodedEvent = try JSONDecoder().decode(SmeltEvent.self, from: eventData)

    #expect(decodedEvent == event)
    #expect(decodedEvent.decodingPolicy == policy)
    #expect(decodedEvent.decodingPolicy?.contextPressure?.promptTokens == 48)
    #expect(decodedEvent.decodingPolicy?.signals?.resolvedName == "balanced")

    let record = SmeltTraceRecord(
        traceId: "trace",
        sessionId: "session",
        generationId: "generation",
        eventType: SmeltEventType.textStart.rawValue,
        packageHash: "package",
        tokenizerHash: "tokenizer",
        prefixCacheKey: "prefix",
        sampler: "temperature:0.270000:seed:123456",
        promptHash: "prompt-hash",
        decodingPolicy: policy,
        timestampUs: 43
    )
    let traceData = try JSONEncoder().encode(record)
    let decodedTrace = try JSONDecoder().decode(SmeltTraceRecord.self, from: traceData)

    #expect(decodedTrace.decodingPolicy == policy)
    #expect(decodedTrace.decodingPolicy?.contextPressure?.contextLimit == 1024)
    #expect(decodedTrace.decodingPolicy?.signals?.intent == "general")
}

@Test func agentDecodingPolicyResolverHandlesNativePolicyInputs() throws {
    let balanced = SmeltDecodingPolicyResolver.resolve(
        SmeltDecodingPolicyRequest(
            name: "balanced",
            phase: "assistant_text",
            textTemperature: 0.27,
            seed: "123",
            contextPressure: SmeltContextPressure(
                contextLimit: 1024,
                promptTokens: 64,
                pressureRatio: 0.0625
            )
        )
    )
    #expect(balanced.name == "balanced")
    #expect(balanced.phase == "assistant_text")
    #expect(balanced.sampler == "temperature")
    #expect(balanced.temperature == 0.27)
    #expect(balanced.seed == "123")
    #expect(balanced.source == "text_temperature")

    let tool = SmeltDecodingPolicyResolver.resolve(
        SmeltDecodingPolicyRequest(
            name: "balanced",
            phase: "tool_call",
            textTemperature: 0.8,
            latestUserText: "brainstorm options"
        )
    )
    #expect(tool.name == "balanced")
    #expect(tool.phase == "tool_call")
    #expect(tool.sampler == "argmax")
    #expect(tool.temperature == nil)
    #expect(tool.source == "tool-call-default")

    let pressure = SmeltDecodingPolicyResolver.resolve(
        SmeltDecodingPolicyRequest(
            name: "adaptive",
            phase: "assistant_text",
            latestUserText: "brainstorm options",
            contextPressure: SmeltContextPressure(pressureRatio: 0.9)
        ),
        randomSeed: { 456 }
    )
    #expect(pressure.sampler == "temperature")
    #expect(pressure.temperature == 0.2)
    #expect(pressure.seed == "456")
    #expect(pressure.source == "context-pressure")

    let mode = try SmeltDecodingPolicyResolver.selectionMode(for: pressure)
    if case let .temperature(temp, seed) = mode {
        #expect(temp == Float(0.2))
        #expect(seed == 456)
    } else {
        Issue.record("Expected temperature selection mode")
    }
}

@Test func agentDecodingPolicyResolverMatrixCapturesSignals() throws {
    struct MatrixCase {
        let label: String
        let request: SmeltDecodingPolicyRequest
        let name: String
        let phase: String
        let sampler: String
        let temperature: Double?
        let source: String
        let intent: String
        let contextPressureBand: String
        let constrainedOutput: Bool
    }

    let cases = [
        MatrixCase(
            label: "tool JSON stays deterministic",
            request: SmeltDecodingPolicyRequest(
                name: "balanced",
                phase: "tool_call",
                textTemperature: 0.8,
                latestUserText: "brainstorm options",
                hasTools: true
            ),
            name: "balanced",
            phase: "tool_call",
            sampler: "argmax",
            temperature: nil,
            source: "tool-call-default",
            intent: "exploratory",
            contextPressureBand: "unknown",
            constrainedOutput: true
        ),
        MatrixCase(
            label: "tool temperature remains explicit experiment",
            request: SmeltDecodingPolicyRequest(
                phase: "tool_call",
                toolTemperature: 0.1,
                hasTools: true
            ),
            name: "adaptive",
            phase: "tool_call",
            sampler: "temperature",
            temperature: 0.1,
            source: "tool_temperature",
            intent: "unknown",
            contextPressureBand: "unknown",
            constrainedOutput: true
        ),
        MatrixCase(
            label: "explicit text temperature creates custom policy",
            request: SmeltDecodingPolicyRequest(
                explicitTemperature: 0.42,
                latestUserText: "fix the compile bug"
            ),
            name: "custom",
            phase: "assistant_text",
            sampler: "temperature",
            temperature: 0.42,
            source: "request.temperature",
            intent: "coding",
            contextPressureBand: "unknown",
            constrainedOutput: false
        ),
        MatrixCase(
            label: "deterministic profile wins over explicit temperature",
            request: SmeltDecodingPolicyRequest(
                name: "deterministic",
                explicitTemperature: 0.9,
                latestUserText: "answer exactly"
            ),
            name: "deterministic",
            phase: "assistant_text",
            sampler: "argmax",
            temperature: nil,
            source: "decoding-policy",
            intent: "exact",
            contextPressureBand: "unknown",
            constrainedOutput: false
        ),
        MatrixCase(
            label: "creative profile warms assistant text",
            request: SmeltDecodingPolicyRequest(
                name: "creative",
                latestUserText: "write a short scene"
            ),
            name: "creative",
            phase: "assistant_text",
            sampler: "temperature",
            temperature: 0.8,
            source: "policy-default",
            intent: "general",
            contextPressureBand: "unknown",
            constrainedOutput: false
        ),
        MatrixCase(
            label: "adaptive pressure tightens before exploratory warming",
            request: SmeltDecodingPolicyRequest(
                name: "adaptive",
                latestUserText: "brainstorm options",
                contextPressure: SmeltContextPressure(pressureRatio: 0.9)
            ),
            name: "adaptive",
            phase: "assistant_text",
            sampler: "temperature",
            temperature: 0.2,
            source: "context-pressure",
            intent: "exploratory",
            contextPressureBand: "high",
            constrainedOutput: false
        ),
        MatrixCase(
            label: "adaptive exploratory text warms sampling",
            request: SmeltDecodingPolicyRequest(
                name: "adaptive",
                latestUserText: "compare a few alternatives"
            ),
            name: "adaptive",
            phase: "assistant_text",
            sampler: "temperature",
            temperature: 0.55,
            source: "adaptive-intent",
            intent: "exploratory",
            contextPressureBand: "unknown",
            constrainedOutput: false
        ),
        MatrixCase(
            label: "adaptive coding text stays conservative",
            request: SmeltDecodingPolicyRequest(
                name: "adaptive",
                latestUserText: "implement the fix and update tests",
                hasTools: true,
                contextPressure: SmeltContextPressure(pressureRatio: 0.62)
            ),
            name: "adaptive",
            phase: "assistant_text",
            sampler: "temperature",
            temperature: 0.35,
            source: "policy-default",
            intent: "coding",
            contextPressureBand: "elevated",
            constrainedOutput: false
        ),
        MatrixCase(
            label: "balanced summary uses stable default",
            request: SmeltDecodingPolicyRequest(
                name: "balanced",
                latestUserText: "summarize the report"
            ),
            name: "balanced",
            phase: "assistant_text",
            sampler: "temperature",
            temperature: 0.35,
            source: "policy-default",
            intent: "summarization",
            contextPressureBand: "unknown",
            constrainedOutput: false
        )
    ]

    for item in cases {
        let policy = SmeltDecodingPolicyResolver.resolve(
            item.request,
            randomSeed: { 999 }
        )
        #expect(policy.name == item.name, "\(item.label): name")
        #expect(policy.phase == item.phase, "\(item.label): phase")
        #expect(policy.sampler == item.sampler, "\(item.label): sampler")
        #expect(policy.source == item.source, "\(item.label): source")
        if let expectedTemperature = item.temperature {
            #expect(policy.temperature == expectedTemperature, "\(item.label): temperature")
        } else {
            #expect(policy.temperature == nil, "\(item.label): temperature")
        }
        #expect(policy.signals?.intent == item.intent, "\(item.label): intent")
        #expect(
            policy.signals?.contextPressureBand == item.contextPressureBand,
            "\(item.label): pressure band"
        )
        #expect(
            policy.signals?.constrainedOutput == item.constrainedOutput,
            "\(item.label): constrained output"
        )
    }
}

@Test func agentContextPressureReducesOutputBudgetAtRuntime() {
    let pressure = SmeltContextPressure(
        requestedMaxTokens: 128,
        toolCallMinTokens: 64,
        messageCount: 2,
        availableOutputTokens: 999,
        action: "none",
        reason: "stale harness estimate"
    ).withRuntime(
        promptTokens: 60,
        contextLimit: 64,
        effectiveMaxTokens: 4,
        requestedMaxTokens: 128,
        resolvedMaxTokens: 4,
        toolCallMinTokens: 64,
        messageCount: 2
    )

    #expect(pressure.contextLimit == 64)
    #expect(pressure.promptTokens == 60)
    #expect(pressure.availableOutputTokens == 4)
    #expect(pressure.requestedMaxTokens == 128)
    #expect(pressure.resolvedMaxTokens == 4)
    #expect(pressure.effectiveMaxTokens == 4)
    #expect(pressure.action == "reduce_output_budget")
}

@Test func agentContextPressureRejectsFullRuntimeContext() {
    let pressure = SmeltContextPressure(
        requestedMaxTokens: 8,
        toolCallMinTokens: 64,
        messageCount: 1
    ).withRuntime(
        promptTokens: 64,
        contextLimit: 64,
        effectiveMaxTokens: nil,
        requestedMaxTokens: 8,
        resolvedMaxTokens: nil,
        toolCallMinTokens: 64,
        messageCount: 1
    )

    #expect(pressure.availableInputTokens == 0)
    #expect(pressure.availableOutputTokens == 0)
    #expect(pressure.pressureRatio == 1)
    #expect(pressure.action == "reject_prompt")
}

@Test func agentStatsRoundTripAsStableJSON() throws {
    let info = SmeltSessionInfo(
        id: "session",
        forkedFrom: "parent",
        prefixCacheKey: "prefix",
        promptLength: 32,
        transcriptTokenCount: 24,
        createdAtUs: 10,
        updatedAtUs: 20
    )
    let sessionStats = SmeltSessionStats(
        info: info,
        systemTokens: 8,
        transcriptTokens: 24,
        capturedTokens: 32,
        replayTokens: 0,
        snapshotBytes: 16_384,
        toolCount: 2,
        metadata: ["surface": "test"]
    )
    let memory = SmeltRuntime.MemoryStats(
        totalAllocatedBytes: 100,
        weightBytes: 40,
        persistentBytes: 30,
        batchScopedBytes: 20,
        contextScopedBytes: 10,
        currentBatchCapacity: 4,
        currentContextCapacity: 32
    )
    let runtimeStats = SmeltTextRuntimeStats(
        maxContextTokens: 4096,
        liveSessionCount: 1,
        maxLiveSessions: 32,
        prefixCacheEntryCount: 1,
        maxPrefixCacheEntries: 64,
        activeGenerationCount: 0,
        memory: memory
    )

    let sessionData = try JSONEncoder().encode(sessionStats)
    let runtimeData = try JSONEncoder().encode(runtimeStats)

    #expect(try JSONDecoder().decode(SmeltSessionStats.self, from: sessionData) == sessionStats)
    #expect(try JSONDecoder().decode(SmeltTextRuntimeStats.self, from: runtimeData) == runtimeStats)
}

@Test func promptSnapshotRoundTripsThroughMappedFile() throws {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("smelt-agent-snapshot-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(
        at: dir,
        withIntermediateDirectories: true,
        attributes: [.posixPermissions: 0o700]
    )
    defer { try? FileManager.default.removeItem(at: dir) }

    let url = dir.appendingPathComponent("snapshot.smkvcache")
    let snapshot = SmeltPromptSnapshot(
        promptLength: 5,
        nextToken: 42,
        byteCount: 7,
        capturedLength: 4,
        replayTokenIds: [9],
        convStates: [Data([1, 2, 3])],
        recStates: [Data([4])],
        keyCaches: [Data([5, 6])],
        valueCaches: [Data([7])]
    )

    let serialized = try snapshot.write(to: url)
    let loaded = try SmeltPromptSnapshot.read(from: url)
    let serializedData = try Data(contentsOf: url)

    #expect(serialized.mode == .serialized)
    #expect(serialized.fileBytes == serializedData.count)
    #expect(loaded.promptLength == snapshot.promptLength)
    #expect(loaded.nextToken == snapshot.nextToken)
    #expect(loaded.byteCount == snapshot.byteCount)
    #expect(loaded.capturedLength == snapshot.capturedLength)
    #expect(loaded.replayTokenIds == snapshot.replayTokenIds)
    #expect(loaded.convStates == snapshot.convStates)
    #expect(loaded.recStates == snapshot.recStates)
    #expect(loaded.keyCaches == snapshot.keyCaches)
    #expect(loaded.valueCaches == snapshot.valueCaches)

    let samePath = try loaded.write(to: url)
    let samePathData = try Data(contentsOf: url)
    #expect(samePath.mode == .linked)
    #expect(samePathData == serializedData)

    let linkedURL = dir.appendingPathComponent("snapshot-linked.smkvcache")
    let linked = try loaded.write(to: linkedURL)
    let linkedData = try Data(contentsOf: linkedURL)
    #expect(linked.mode == .linked)
    #expect(linked.fileBytes == serialized.fileBytes)
    #expect(linkedData == serializedData)

    let originalAttributes = try FileManager.default.attributesOfItem(atPath: url.path)
    let linkedAttributes = try FileManager.default.attributesOfItem(atPath: linkedURL.path)
    #expect(originalAttributes[.systemFileNumber] as? NSNumber == linkedAttributes[.systemFileNumber] as? NSNumber)
}

@Test func agentToolCallEventRoundTripsWithArguments() throws {
    let event = SmeltEvent(
        type: .toolCallEnd,
        sessionId: "session",
        generationId: "generation",
        traceId: "trace",
        id: "call_1",
        name: "read_file",
        delta: #"{"path":"Package.swift"}"#,
        arguments: [
            "path": .string("Package.swift"),
            "limit": .number(120)
        ],
        timestampUs: 42
    )

    let data = try JSONEncoder().encode(event)
    let decoded = try JSONDecoder().decode(SmeltEvent.self, from: data)

    #expect(decoded == event)
    #expect(decoded.id == "call_1")
    #expect(decoded.name == "read_file")
    #expect(decoded.arguments?["path"] == .string("Package.swift"))
}

@Test func agentToolCallFailureDiagnosticRoundTripsThroughEventAndTrace() throws {
    let failure = SmeltToolCallFailureDiagnostic(
        sessionId: "session",
        generationId: "generation",
        traceId: "trace",
        promptHash: "prompt-hash",
        generatedTokenIds: [101, 202],
        partialJSON: #"{"name":"read_file""#,
        stopCause: "max_tokens",
        maxTokens: 2,
        requestedMaxTokens: 1,
        effectiveMaxTokens: 2,
        toolCallMinTokens: 2,
        toolCallBudgetWasLifted: true,
        runtimeMaxGeneratedTokens: 512,
        toolCount: 1
    )
    let event = SmeltEvent(
        type: .error,
        sessionId: "session",
        generationId: "generation",
        traceId: "trace",
        error: failure.message,
        toolCallFailure: failure,
        timestampUs: 42
    )

    let eventData = try JSONEncoder().encode(event)
    let decoded = try JSONDecoder().decode(SmeltEvent.self, from: eventData)

    #expect(decoded == event)
    #expect(decoded.toolCallFailure?.generatedTokenIds == [101, 202])
    #expect(decoded.toolCallFailure?.partialJSON == #"{"name":"read_file""#)
    #expect(decoded.toolCallFailure?.partialJSONByteCount == Data(#"{"name":"read_file""#.utf8).count)
    #expect(decoded.toolCallFailure?.isAccepting == false)
    #expect(decoded.toolCallFailure?.stopCause == "max_tokens")
    #expect(decoded.toolCallFailure?.requestedMaxTokens == 1)
    #expect(decoded.toolCallFailure?.effectiveMaxTokens == 2)
    #expect(decoded.toolCallFailure?.toolCallBudgetWasLifted == true)

    let record = SmeltTraceRecord(
        traceId: "trace",
        sessionId: "session",
        generationId: "generation",
        eventType: SmeltEventType.error.rawValue,
        packageHash: "package",
        tokenizerHash: "tokenizer",
        prefixCacheKey: "prefix",
        sampler: "argmax",
        promptHash: "prompt-hash",
        error: failure.message,
        toolCallFailure: failure,
        timestampUs: 43
    )
    let traceData = try JSONEncoder().encode(record)
    let decodedTrace = try JSONDecoder().decode(SmeltTraceRecord.self, from: traceData)

    #expect(decodedTrace.toolCallFailure == failure)
}

@Test func agentToolGrammarBuildsCanonicalMultiToolUnion() throws {
    let schema = try SmeltToolGrammar.jsonSchema(
        for: [
            SmeltToolDescriptor(
                name: "read_file",
                schemaJSON: #"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}"#,
                description: "Read a file"
            ),
            SmeltToolDescriptor(
                name: "list_dir",
                schemaJSON: #"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}"#,
                description: "List a directory"
            )
        ]
    )
    let object = try JSONSerialization.jsonObject(with: Data(schema.utf8)) as? [String: Any]
    let branches = object?["oneOf"] as? [[String: Any]]
    let first = branches?.first
    let properties = first?["properties"] as? [String: Any]
    let name = properties?["name"] as? [String: Any]

    #expect(branches?.count == 2)
    #expect(name?["enum"] as? [String] == ["read_file"])
    #expect(properties?["arguments"] != nil)
    #expect(schema.contains(#""properties":{"name":"#))
}

@Test func agentToolGrammarLarkUnionContainsBothArms() throws {
    let lark = try SmeltToolGrammar.larkUnion(
        for: [
            SmeltToolDescriptor(
                name: "read_file",
                schemaJSON: #"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}"#
            )
        ]
    )
    #expect(lark.contains("start: tool_call | text_response"))
    #expect(lark.contains("%json"))
    #expect(lark.contains("text_response:"))
    #expect(lark.contains(#""enum":["read_file"]"#))
    // First non-whitespace char must be non-`{` so leading whitespace
    // + `{...}` routes only to the JSON arm.
    #expect(lark.contains("[^{\\s]"))
}

@Test func agentToolGrammarLarkUnionTextArmRequiresMinSubstantiveChars() throws {
    // The text arm is decomposed as
    //   text_response: leading_ws first_char gap nonws gap nonws text_tail
    // so the matcher requires at least 3 non-whitespace characters
    // before EOS becomes acceptable. The `gap` rule allows optional
    // whitespace between the substantive chars so natural prose like
    // "I can ..." parses correctly. This test pins the structural
    // shape; a regression that flattens these to a single regex
    // terminal would defeat llguidance's per-rule state tracking
    // and let the model EOS at one character.
    let lark = try SmeltToolGrammar.larkUnion(
        for: [
            SmeltToolDescriptor(
                name: "read_file",
                schemaJSON: #"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}"#
            )
        ]
    )
    #expect(lark.contains("text_response: leading_ws first_char gap nonws gap nonws text_tail"))
    #expect(lark.contains("first_char: /[^{\\s]/"))
    #expect(lark.contains("nonws: /\\S/"))
    #expect(lark.contains("gap: /\\s*/"))
    #expect(lark.contains("leading_ws: /\\s*/"))
    #expect(lark.contains("text_tail: /[\\s\\S]*/"))
}

@Test(.enabled(if: qwenLLGuidanceFixtureAvailable(), "Build Qwen 4B and llguidance fixtures to run the integration test"))
func qwenTokenizerDrivesLLGuidanceLarkUnion() throws {
    let packagePath = qwenLLGuidancePackagePath()
    let tokenizer = try SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
    let manifestData = try Data(contentsOf: URL(fileURLWithPath: "\(packagePath)/manifest.json"))
    let manifest = try SmeltManifest.decode(from: manifestData)
    let eosTokens = manifest.inference?.eosTokens ?? []
    let lark = try SmeltToolGrammar.larkUnion(
        for: [
            SmeltToolDescriptor(
                name: "read_file",
                schemaJSON: #"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}"#
            ),
            SmeltToolDescriptor(
                name: "list_dir",
                schemaJSON: #"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}"#
            )
        ]
    )
    let llgTokenizer = try SmeltLLGuidanceTokenizer(
        tokenizer: tokenizer,
        eosTokens: eosTokens
    )

    // Text-arm path: a non-`{` first character must be accepted and
    // the matcher should remain non-stopped.
    let textMatcher = try SmeltLLGuidanceMatcher(tokenizer: llgTokenizer, lark: lark)
    try textMatcher.consume(tokenIds: tokenizer.encode("Hello there!"))
    #expect(!textMatcher.isStopped)

    // JSON-arm path: once the model commits to JSON output (emits
    // `{`), the text arm is dead and the schema constraint is the
    // ONLY arm tracking. The mask after `{"name":"` must reject
    // undeclared tool names just like the strict json_schema matcher.
    let jsonMatcher = try SmeltLLGuidanceMatcher(tokenizer: llgTokenizer, lark: lark)
    try jsonMatcher.consume(tokenIds: tokenizer.encode("{\"name\":\""))
    let mask = try jsonMatcher.computeMask()
    let readFirst = try #require(tokenizer.encode("read_file").first)
    let listFirst = try #require(tokenizer.encode("list_dir").first)
    let writeFirst = try #require(tokenizer.encode("write_file").first)
    #expect(SmeltLLGuidanceMatcher.tokenIsAllowed(readFirst, in: mask))
    #expect(SmeltLLGuidanceMatcher.tokenIsAllowed(listFirst, in: mask))
    #expect(!SmeltLLGuidanceMatcher.tokenIsAllowed(writeFirst, in: mask))
}

@Test(.enabled(if: qwenLLGuidanceFixtureAvailable(), "Build Qwen 4B and llguidance fixtures to run the integration test"))
func qwenTokenizerDrivesLLGuidanceToolSchemaMask() throws {
    let packagePath = qwenLLGuidancePackagePath()
    let tokenizer = try SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
    let manifestData = try Data(contentsOf: URL(fileURLWithPath: "\(packagePath)/manifest.json"))
    let manifest = try SmeltManifest.decode(from: manifestData)
    let eosTokens = manifest.inference?.eosTokens ?? []
    let grammar = try SmeltToolGrammar.jsonSchema(
        for: [
            SmeltToolDescriptor(
                name: "read_file",
                schemaJSON: #"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}"#
            ),
            SmeltToolDescriptor(
                name: "list_dir",
                schemaJSON: #"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}"#
            )
        ]
    )
    let llgTokenizer = try SmeltLLGuidanceTokenizer(
        tokenizer: tokenizer,
        eosTokens: eosTokens
    )
    let matcher = try SmeltLLGuidanceMatcher(
        tokenizer: llgTokenizer,
        jsonSchema: grammar
    )

    try matcher.consume(tokenIds: tokenizer.encode("{\"name\":\""))
    let mask = try matcher.computeMask()
    let readFirst = try #require(tokenizer.encode("read_file").first)
    let listFirst = try #require(tokenizer.encode("list_dir").first)
    let writeFirst = try #require(tokenizer.encode("write_file").first)

    #expect(llgTokenizer.vocabSize > 200_000)
    #expect(SmeltLLGuidanceMatcher.tokenIsAllowed(readFirst, in: mask))
    #expect(SmeltLLGuidanceMatcher.tokenIsAllowed(listFirst, in: mask))
    #expect(!SmeltLLGuidanceMatcher.tokenIsAllowed(writeFirst, in: mask))

    try matcher.consume(
        tokenIds: tokenizer.encode(
            "read_file\",\"arguments\":{\"path\":\"Package.swift\"}}"
        )
    )
    #expect(matcher.isAccepting)
}

@Test func agentCancellationIsIdempotent() {
    let cancellation = SmeltCancellation()

    #expect(!cancellation.isCancelled)
    cancellation.cancel()
    cancellation.cancel()
    #expect(cancellation.isCancelled)
}

@Test func agentGenerateOptionsResolvePerTurnTools() {
    let sessionTools = [
        SmeltToolDescriptor(
            name: "read_file",
            schemaJSON: #"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}"#
        )
    ]

    #expect(SmeltGenerateOptions().resolvedTools(sessionTools: sessionTools) == sessionTools)
    #expect(SmeltGenerateOptions(tools: []).resolvedTools(sessionTools: sessionTools).isEmpty)

    let overrideTools = [
        SmeltToolDescriptor(
            name: "list_dir",
            schemaJSON: #"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}"#
        )
    ]
    #expect(SmeltGenerateOptions(tools: overrideTools).resolvedTools(sessionTools: sessionTools) == overrideTools)

    let constrained = SmeltGenerateOptions(maxTokens: 1, toolCallMinTokens: 64)
    #expect(constrained.effectiveMaxTokens(hasActiveTools: true) == 64)
    #expect(constrained.effectiveMaxTokens(hasActiveTools: false) == 1)
    #expect(SmeltGenerateOptions(maxTokens: 128, toolCallMinTokens: 64).effectiveMaxTokens(hasActiveTools: true) == 128)
    #expect(SmeltGenerateOptions(toolCallMinTokens: 0).toolCallMinTokens == 1)
}

@Test func traceWriterAppendsJSONLRecords() throws {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("smelt-agent-trace-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: dir) }

    let writer = SmeltTraceWriter(directoryPath: dir.path)
    let record = SmeltTraceRecord(
        traceId: "trace-1",
        sessionId: "session",
        generationId: "generation",
        eventType: "text_delta",
        packageHash: "pkg",
        tokenizerHash: "tok",
        prefixCacheKey: "prefix",
        sampler: "argmax",
        promptHash: "prompt",
        contextTokenIds: [1, 2, 3],
        tokenId: 7,
        position: 3,
        textByteCount: 1,
        stepLatencyUs: 100,
        timestampUs: 123
    )

    writer.write(record)
    writer.write(record)

    let file = dir.appendingPathComponent("trace-1.jsonl")
    let text = try String(contentsOf: file, encoding: .utf8)
    let lines = text.split(separator: "\n")

    #expect(lines.count == 2)
    #expect(lines[0].contains("\"eventType\":\"text_delta\""))
    #expect(lines[0].contains("\"contextTokenIds\":[1,2,3]"))
}

@Test func agentHashForTokensIsStable() {
    let left = SmeltHash.tokenHash([1, 2, 3])
    let right = SmeltHash.tokenHash([1, 2, 3])
    let different = SmeltHash.tokenHash([1, 2, 4])

    #expect(left == right)
    #expect(left != different)
}

@Test func legacyContextManifestUsesDefaultAndMaxLimitNames() throws {
    let json = Data(#"{"default_limit":4096,"max_limit":32768}"#.utf8)
    let context = try JSONDecoder().decode(SmeltContextManifest.self, from: json)

    #expect(context.defaultLimit == 4096)
    #expect(context.maxLimit == 32768)
}

private func qwenLLGuidanceFixtureAvailable() -> Bool {
    let fm = FileManager.default
    return fm.fileExists(atPath: "\(qwenLLGuidancePackagePath())/tokenizer.json")
        && fm.fileExists(atPath: "third_party/llguidance/lib/libllguidance.a")
}

private func qwenLLGuidancePackagePath() -> String {
    let env = ProcessInfo.processInfo.environment
    if let explicit = env["SMELT_QWEN_4B_PACKAGE"], !explicit.isEmpty {
        return explicit
    }
    return FileManager.default.currentDirectoryPath
        + "/artifacts/qwen35-4b-qmm16x128/Qwen_Qwen3.5-4B.smeltpkg"
}
