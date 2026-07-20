import CryptoKit
import Darwin
import Foundation
import SmeltSchema

public struct SmeltToolDescriptor: Codable, Equatable, Sendable {
    public let name: String
    public let schemaJSON: String
    public let description: String?
    /// Optional prompt-facing representation of the complete tool object.
    /// Native adapters retain this when a frozen model template renders the
    /// wire descriptor itself (for example via Jinja `tojson`). Validation and
    /// constrained decoding continue to use the normalized fields above.
    public let promptJSON: String?

    public init(
        name: String,
        schemaJSON: String,
        description: String? = nil,
        promptJSON: String? = nil
    ) {
        self.name = name
        self.schemaJSON = schemaJSON
        self.description = description
        self.promptJSON = promptJSON
    }
}

public enum SmeltToolGrammarError: Error, CustomStringConvertible {
    case noTools
    case invalidSchema(tool: String)

    public var description: String {
        switch self {
        case .noTools:
            return "Smelt tool grammar requires at least one tool"
        case .invalidSchema(let tool):
            return "Smelt tool \(tool) has an invalid JSON schema"
        }
    }
}

public enum SmeltToolGrammar {
    public static func jsonSchema(
        for tools: [SmeltToolDescriptor]
    ) throws -> String {
        guard !tools.isEmpty else {
            throw SmeltToolGrammarError.noTools
        }

        let branches = try tools.map { tool in
            let name = try jsonString(tool.name)
            let schema = try normalizedSchemaJSON(for: tool)
            return #"{"type":"object","properties":{"name":{"type":"string","enum":["#
                + name
                + #"]},"arguments":"#
                + schema
                + #"},"required":["name","arguments"],"additionalProperties":false}"#
        }
        return #"{"oneOf":["# + branches.joined(separator: ",") + "]}"
    }

    /// Lark grammar with two alternation arms — a tool-call JSON
    /// object matching `jsonSchema(for:)`, OR arbitrary free text.
    ///
    /// The text arm enforces two invariants STRUCTURALLY (not via
    /// flat regex quantifiers, which lark fuses into a single greedy
    /// DFA that the matcher accepts at any length):
    ///
    /// 1. Disambiguation — the first non-whitespace character must
    ///    be neither `{` nor JSON-prefix. llguidance keeps every
    ///    consistent arm alive while the prefix is ambiguous; a bare
    ///    `[^{]` would let a leading newline + `{...}` (a valid JSON
    ///    object per the JSON spec, since whitespace is allowed
    ///    before the root value) stay on the text arm and drop
    ///    schema constraints from the JSON tokens that follow.
    ///
    ///    `leading_ws + first_char` (first_char regex `[^{\\s]`)
    ///    enforces this.
    ///
    /// 2. Minimum substantive body — the text arm requires at least
    ///    3 non-whitespace characters before EOS is acceptable.
    ///    An instruct model under tool_choice:"auto" otherwise degenerates
    ///    ~20% of the time to a single-letter response ("w", "r")
    ///    then EOS. Encoding the floor in the grammar (rather than as
    ///    a separate mask layer in MinTokensEOSGate) means llguidance's
    ///    own matcher composes it with the rest of the grammar
    ///    constraints — no risk of the gate's bit-clearing colliding
    ///    with a matcher state that only permits EOS.
    ///
    /// Used for OpenAI `tool_choice:"auto"`. For `required` /
    /// `specific`, use `jsonSchema(for:)` directly — strict mode
    /// auto-terminates on JSON close, no min floor needed.
    public static func larkUnion(
        for tools: [SmeltToolDescriptor]
    ) throws -> String {
        let schema = try jsonSchema(for: tools)
        return """
        start: tool_call | text_response
        tool_call: %json \(schema)
        text_response: leading_ws first_char gap nonws gap nonws text_tail
        leading_ws: /\\s*/
        first_char: /[^{\\s]/
        gap: /\\s*/
        nonws: /\\S/
        text_tail: /[\\s\\S]*/
        """
    }

    private static func normalizedSchemaJSON(
        for tool: SmeltToolDescriptor
    ) throws -> String {
        let rawData = Data(tool.schemaJSON.utf8)
        let raw = try JSONSerialization.jsonObject(with: rawData)
        guard let object = raw as? [String: Any],
              JSONSerialization.isValidJSONObject(object)
        else {
            throw SmeltToolGrammarError.invalidSchema(tool: tool.name)
        }
        let canonicalData = try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys]
        )
        return String(decoding: canonicalData, as: UTF8.self)
    }

    private static func jsonString(_ value: String) throws -> String {
        let data = try JSONSerialization.data(withJSONObject: [value])
        let encoded = String(decoding: data, as: UTF8.self)
        return String(encoded.dropFirst().dropLast())
    }
}

public enum SmeltJSONValue: Codable, Equatable, Sendable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([SmeltJSONValue])
    case object([String: SmeltJSONValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([SmeltJSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode([String: SmeltJSONValue].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case .bool(let value):
            try container.encode(value)
        case .number(let value):
            try container.encode(value)
        case .string(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .object(let value):
            try container.encode(value)
        }
    }
}

/// JSON shape emitted by `SmeltToolGrammar` when the matcher
/// accepts.
public struct SmeltGeneratedToolCall: Decodable, Sendable {
    public let name: String
    public let arguments: [String: SmeltJSONValue]

    public init(
        name: String,
        arguments: [String: SmeltJSONValue]
    ) {
        self.name = name
        self.arguments = arguments
    }
}

public enum SmeltToolCallID {
    public static func next() -> String {
        "call_\(UUID().uuidString.lowercased())"
    }
}

public struct SmeltLimits: Sendable {
    public let maxLiveSessions: Int
    public let maxPrefixCacheEntries: Int
    public let maxTranscriptTokens: Int
    public let maxGeneratedTokensPerTurn: Int
    public let maxIdleSeconds: Int?

    public init(
        maxLiveSessions: Int = 32,
        maxPrefixCacheEntries: Int = 64,
        maxTranscriptTokens: Int = Int(Int32.max) - 1,
        maxGeneratedTokensPerTurn: Int = 4_096,
        maxIdleSeconds: Int? = nil
    ) {
        self.maxLiveSessions = maxLiveSessions
        self.maxPrefixCacheEntries = maxPrefixCacheEntries
        self.maxTranscriptTokens = maxTranscriptTokens
        self.maxGeneratedTokensPerTurn = maxGeneratedTokensPerTurn
        self.maxIdleSeconds = maxIdleSeconds
    }
}

public struct SmeltSessionConfig: Sendable {
    public let id: String?
    public let systemTokenIds: [Int32]
    public let tools: [SmeltToolDescriptor]
    public let metadata: [String: String]

    public init(
        id: String? = nil,
        systemTokenIds: [Int32] = [],
        tools: [SmeltToolDescriptor] = [],
        metadata: [String: String] = [:]
    ) {
        self.id = id
        self.systemTokenIds = systemTokenIds
        self.tools = tools
        self.metadata = metadata
    }
}

public struct SmeltContextPressure: Codable, Equatable, Sendable {
    public let contextLimit: Int?
    public let promptTokens: Int?
    public let estimatedPromptTokens: Int?
    public let requestedMaxTokens: Int?
    public let resolvedMaxTokens: Int?
    public let effectiveMaxTokens: Int?
    public let toolCallMinTokens: Int?
    public let messageCount: Int?
    public let availableInputTokens: Int?
    public let availableOutputTokens: Int?
    public let pressureRatio: Double?
    public let action: String?
    public let reason: String?

    public init(
        contextLimit: Int? = nil,
        promptTokens: Int? = nil,
        estimatedPromptTokens: Int? = nil,
        requestedMaxTokens: Int? = nil,
        resolvedMaxTokens: Int? = nil,
        effectiveMaxTokens: Int? = nil,
        toolCallMinTokens: Int? = nil,
        messageCount: Int? = nil,
        availableInputTokens: Int? = nil,
        availableOutputTokens: Int? = nil,
        pressureRatio: Double? = nil,
        action: String? = nil,
        reason: String? = nil
    ) {
        self.contextLimit = contextLimit
        self.promptTokens = promptTokens
        self.estimatedPromptTokens = estimatedPromptTokens
        self.requestedMaxTokens = requestedMaxTokens
        self.resolvedMaxTokens = resolvedMaxTokens
        self.effectiveMaxTokens = effectiveMaxTokens
        self.toolCallMinTokens = toolCallMinTokens
        self.messageCount = messageCount
        self.availableInputTokens = availableInputTokens
        self.availableOutputTokens = availableOutputTokens
        self.pressureRatio = pressureRatio
        self.action = action
        self.reason = reason
    }

    public func withRuntime(
        promptTokens: Int,
        contextLimit: Int,
        effectiveMaxTokens: Int?,
        requestedMaxTokens: Int? = nil,
        resolvedMaxTokens: Int? = nil,
        toolCallMinTokens: Int? = nil,
        messageCount: Int? = nil
    ) -> SmeltContextPressure {
        let availableOutputTokens = max(contextLimit - promptTokens, 0)
        let requestedOutputTokens = requestedMaxTokens ?? self.requestedMaxTokens
        let resolvedAction: String
        let resolvedReason: String
        if availableOutputTokens <= 0 {
            resolvedAction = "reject_prompt"
            resolvedReason = "runtime rejected prompt because it exceeded the active context limit"
        } else if let requestedOutputTokens, requestedOutputTokens > availableOutputTokens {
            resolvedAction = "reduce_output_budget"
            resolvedReason = "context pressure reduced output budget before generation"
        } else {
            resolvedAction = "none"
            resolvedReason = "runtime context has room for the requested output budget"
        }
        return SmeltContextPressure(
            contextLimit: contextLimit,
            promptTokens: promptTokens,
            estimatedPromptTokens: estimatedPromptTokens ?? promptTokens,
            requestedMaxTokens: requestedOutputTokens,
            resolvedMaxTokens: resolvedMaxTokens ?? self.resolvedMaxTokens ?? requestedOutputTokens,
            effectiveMaxTokens: effectiveMaxTokens ?? self.effectiveMaxTokens,
            toolCallMinTokens: toolCallMinTokens ?? self.toolCallMinTokens,
            messageCount: messageCount ?? self.messageCount,
            availableInputTokens: max(contextLimit - promptTokens, 0),
            availableOutputTokens: availableOutputTokens,
            pressureRatio: contextLimit > 0 ? Double(promptTokens) / Double(contextLimit) : pressureRatio,
            action: resolvedAction,
            reason: resolvedReason
        )
    }
}

public struct SmeltPolicySignals: Codable, Equatable, Sendable {
    public let requestedName: String?
    public let resolvedName: String
    public let phase: String
    public let intent: String
    public let contextPressureBand: String
    public let hasTools: Bool
    public let constrainedOutput: Bool
    public let explicitTemperature: Bool
    public let textTemperature: Bool
    public let toolTemperature: Bool
    public let seeded: Bool

    public init(
        requestedName: String? = nil,
        resolvedName: String,
        phase: String,
        intent: String,
        contextPressureBand: String,
        hasTools: Bool,
        constrainedOutput: Bool,
        explicitTemperature: Bool,
        textTemperature: Bool,
        toolTemperature: Bool,
        seeded: Bool
    ) {
        self.requestedName = requestedName
        self.resolvedName = resolvedName
        self.phase = phase
        self.intent = intent
        self.contextPressureBand = contextPressureBand
        self.hasTools = hasTools
        self.constrainedOutput = constrainedOutput
        self.explicitTemperature = explicitTemperature
        self.textTemperature = textTemperature
        self.toolTemperature = toolTemperature
        self.seeded = seeded
    }
}

public struct SmeltDecodingPolicy: Codable, Equatable, Sendable {
    public let name: String
    public let phase: String
    public let sampler: String
    public let temperature: Double?
    public let seed: String?
    public let source: String?
    public let reason: String?
    public let contextPressure: SmeltContextPressure?
    public let signals: SmeltPolicySignals?

    public init(
        name: String,
        phase: String,
        sampler: String,
        temperature: Double? = nil,
        seed: String? = nil,
        source: String? = nil,
        reason: String? = nil,
        contextPressure: SmeltContextPressure? = nil,
        signals: SmeltPolicySignals? = nil
    ) {
        self.name = name
        self.phase = phase
        self.sampler = sampler
        self.temperature = temperature
        self.seed = seed
        self.source = source
        self.reason = reason
        self.contextPressure = contextPressure
        self.signals = signals
    }

    public func withContextPressure(
        _ contextPressure: SmeltContextPressure?
    ) -> SmeltDecodingPolicy {
        SmeltDecodingPolicy(
            name: name,
            phase: phase,
            sampler: sampler,
            temperature: temperature,
            seed: seed,
            source: source,
            reason: reason,
            contextPressure: contextPressure,
            signals: signals
        )
    }
}

public struct SmeltDecodingPolicyRequest: Codable, Equatable, Sendable {
    public let name: String?
    public let phase: String?
    public let explicitTemperature: Double?
    public let textTemperature: Double?
    public let toolTemperature: Double?
    public let seed: String?
    public let latestUserText: String?
    public let hasTools: Bool?
    public let contextPressure: SmeltContextPressure?

    public init(
        name: String? = nil,
        phase: String? = nil,
        explicitTemperature: Double? = nil,
        textTemperature: Double? = nil,
        toolTemperature: Double? = nil,
        seed: String? = nil,
        latestUserText: String? = nil,
        hasTools: Bool? = nil,
        contextPressure: SmeltContextPressure? = nil
    ) {
        self.name = name
        self.phase = phase
        self.explicitTemperature = explicitTemperature
        self.textTemperature = textTemperature
        self.toolTemperature = toolTemperature
        self.seed = seed
        self.latestUserText = latestUserText
        self.hasTools = hasTools
        self.contextPressure = contextPressure
    }

    public func withRuntime(
        contextPressure: SmeltContextPressure?,
        hasTools: Bool
    ) -> SmeltDecodingPolicyRequest {
        SmeltDecodingPolicyRequest(
            name: name,
            phase: phase,
            explicitTemperature: explicitTemperature,
            textTemperature: textTemperature,
            toolTemperature: toolTemperature,
            seed: seed,
            latestUserText: latestUserText,
            hasTools: self.hasTools ?? hasTools,
            contextPressure: contextPressure ?? self.contextPressure
        )
    }
}

public enum SmeltDecodingPolicyResolver {
    private struct ProfileDecision {
        let resolvedName: String
        let temperature: Double
        let source: String
        let reason: String
    }

    public static func resolve(
        _ request: SmeltDecodingPolicyRequest,
        randomSeed: @Sendable () -> UInt64 = {
            UInt64.random(in: UInt64.min...UInt64.max)
        }
    ) -> SmeltDecodingPolicy {
        let signals = signals(for: request)
        let decision = resolveProfile(request, signals: signals)
        var seed = request.seed

        guard decision.temperature.isFinite, decision.temperature > 0 else {
            return SmeltDecodingPolicy(
                name: decision.resolvedName,
                phase: signals.phase,
                sampler: "argmax",
                seed: seed,
                source: decision.source,
                reason: decision.reason,
                contextPressure: request.contextPressure,
                signals: signals
            )
        }

        seed = seed ?? String(randomSeed())
        return SmeltDecodingPolicy(
            name: decision.resolvedName,
            phase: signals.phase,
            sampler: "temperature",
            temperature: decision.temperature,
            seed: seed,
            source: decision.source,
            reason: decision.reason,
            contextPressure: request.contextPressure,
            signals: signals
        )
    }

    public static func signals(
        for request: SmeltDecodingPolicyRequest
    ) -> SmeltPolicySignals {
        let phase = normalizePhase(request.phase)
        let configuredName = normalizeName(request.name)
        return SmeltPolicySignals(
            requestedName: request.name,
            resolvedName: configuredName,
            phase: phase,
            intent: classifyIntent(request.latestUserText),
            contextPressureBand: classifyContextPressure(request.contextPressure),
            hasTools: request.hasTools ?? false,
            constrainedOutput: phase == "tool_call",
            explicitTemperature: request.explicitTemperature != nil,
            textTemperature: request.textTemperature != nil,
            toolTemperature: request.toolTemperature != nil,
            seeded: request.seed != nil
        )
    }

    public static func selectionMode(
        for policy: SmeltDecodingPolicy
    ) throws -> SmeltSelectionMode {
        switch policy.sampler {
        case "argmax":
            return .argmax
        case "temperature":
            guard let temperature = policy.temperature, temperature > 0 else {
                return .argmax
            }
            let seed: UInt64
            if let rawSeed = policy.seed, let parsed = UInt64(rawSeed) {
                seed = parsed
            } else {
                seed = UInt64.random(in: UInt64.min...UInt64.max)
            }
            return .temperature(Float(temperature), seed: seed)
        default:
            throw NSError(
                domain: "SmeltDecodingPolicyResolver",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Unknown decoding policy sampler: \(policy.sampler)"
                ]
            )
        }
    }

    private static func normalizeName(_ raw: String?) -> String {
        let value = raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch value {
        case "deterministic", "argmax", "greedy":
            return "deterministic"
        case "balanced":
            return "balanced"
        case "creative":
            return "creative"
        default:
            return "adaptive"
        }
    }

    private static func normalizePhase(_ raw: String?) -> String {
        raw == "tool_call" ? "tool_call" : "assistant_text"
    }

    private static func resolveProfile(
        _ request: SmeltDecodingPolicyRequest,
        signals: SmeltPolicySignals
    ) -> ProfileDecision {
        let configuredName = signals.resolvedName

        if signals.phase == "tool_call" {
            let temperature = request.toolTemperature ?? 0
            return ProfileDecision(
                resolvedName: request.name == nil ? "adaptive" : configuredName,
                temperature: temperature,
                source: temperature > 0 ? "tool_temperature" : "tool-call-default",
                reason: "constrained tool-call JSON must be maximally reproducible"
            )
        }

        if request.name != nil && configuredName == "deterministic" {
            return ProfileDecision(
                resolvedName: configuredName,
                temperature: 0,
                source: "decoding-policy",
                reason: "deterministic policy forces argmax"
            )
        }

        if let explicitTemperature = request.explicitTemperature {
            return ProfileDecision(
                resolvedName: "custom",
                temperature: explicitTemperature,
                source: "request.temperature",
                reason: "request supplied an explicit temperature"
            )
        }

        if let textTemperature = request.textTemperature {
            return ProfileDecision(
                resolvedName: configuredName,
                temperature: textTemperature,
                source: "text_temperature",
                reason: "request configured assistant text temperature"
            )
        }

        switch configuredName {
        case "deterministic":
            return ProfileDecision(
                resolvedName: configuredName,
                temperature: 0,
                source: "decoding-policy",
                reason: "deterministic policy forces argmax"
            )
        case "creative":
            return ProfileDecision(
                resolvedName: configuredName,
                temperature: 0.8,
                source: "policy-default",
                reason: "creative policy uses a warmer assistant text sampler"
            )
        case "adaptive" where signals.contextPressureBand == "high":
            return ProfileDecision(
                resolvedName: configuredName,
                temperature: 0.2,
                source: "context-pressure",
                reason: "adaptive policy tightened sampling under high context pressure"
            )
        case "adaptive" where signals.intent == "exploratory":
            return ProfileDecision(
                resolvedName: configuredName,
                temperature: 0.55,
                source: "adaptive-intent",
                reason: "adaptive policy detected an exploratory request"
            )
        default:
            return ProfileDecision(
                resolvedName: configuredName,
                temperature: 0.35,
                source: "policy-default",
                reason: configuredName == "balanced"
                    ? "balanced policy uses modest assistant text sampling"
                    : "adaptive policy selected modest sampling for \(signals.hasTools ? "post-tool text" : "assistant text")"
            )
        }
    }

    private static func classifyIntent(_ text: String?) -> String {
        guard let text else { return "unknown" }
        let pattern = #"\b(brainstorm|explore|creative|ideas?|options?|sketch|draft|alternatives?|think through|compare)\b"#
        if text.range(
            of: pattern,
            options: [.regularExpression, .caseInsensitive]
        ) != nil {
            return "exploratory"
        }
        if text.range(
            of: #"\b(code|coding|bug|fix|test|compile|implement|refactor|patch|diff)\b"#,
            options: [.regularExpression, .caseInsensitive]
        ) != nil {
            return "coding"
        }
        if text.range(
            of: #"\b(summarize|summary|tl;dr|extract|condense)\b"#,
            options: [.regularExpression, .caseInsensitive]
        ) != nil {
            return "summarization"
        }
        if text.range(
            of: #"\b(exact(?:ly)?|literal|only|yes or no|one word|deterministic)\b"#,
            options: [.regularExpression, .caseInsensitive]
        ) != nil {
            return "exact"
        }
        return "general"
    }

    private static func classifyContextPressure(
        _ contextPressure: SmeltContextPressure?
    ) -> String {
        guard let pressureRatio = contextPressure?.pressureRatio else {
            return "unknown"
        }
        if pressureRatio > 0.8 {
            return "high"
        }
        if pressureRatio > 0.55 {
            return "elevated"
        }
        return "normal"
    }
}

public struct SmeltGenerateOptions: Sendable {
    public let generationId: String?
    public let selectionMode: SmeltSelectionMode
    public let decodingPolicy: SmeltDecodingPolicy?
    public let decodingPolicyRequest: SmeltDecodingPolicyRequest?
    public let maxTokens: Int?
    public let toolCallMinTokens: Int
    public let commitOnCancel: Bool
    public let tools: [SmeltToolDescriptor]?

    public init(
        generationId: String? = nil,
        selectionMode: SmeltSelectionMode = .argmax,
        decodingPolicy: SmeltDecodingPolicy? = nil,
        decodingPolicyRequest: SmeltDecodingPolicyRequest? = nil,
        maxTokens: Int? = nil,
        toolCallMinTokens: Int = 64,
        commitOnCancel: Bool = false,
        tools: [SmeltToolDescriptor]? = nil
    ) {
        self.generationId = generationId
        self.selectionMode = selectionMode
        self.decodingPolicy = decodingPolicy
        self.decodingPolicyRequest = decodingPolicyRequest
        self.maxTokens = maxTokens
        self.toolCallMinTokens = max(1, toolCallMinTokens)
        self.commitOnCancel = commitOnCancel
        self.tools = tools
    }

    func resolvedTools(sessionTools: [SmeltToolDescriptor]) -> [SmeltToolDescriptor] {
        tools ?? sessionTools
    }

    func effectiveMaxTokens(hasActiveTools: Bool) -> Int? {
        guard hasActiveTools, let maxTokens else {
            return maxTokens
        }
        return max(maxTokens, toolCallMinTokens)
    }
}

public struct SmeltSessionInfo: Codable, Equatable, Sendable {
    public let id: String
    public let forkedFrom: String?
    public let prefixCacheKey: String
    public let promptLength: Int
    public let transcriptTokenCount: Int
    public let createdAtUs: UInt64
    public let updatedAtUs: UInt64

    public init(
        id: String,
        forkedFrom: String?,
        prefixCacheKey: String,
        promptLength: Int,
        transcriptTokenCount: Int,
        createdAtUs: UInt64,
        updatedAtUs: UInt64
    ) {
        self.id = id
        self.forkedFrom = forkedFrom
        self.prefixCacheKey = prefixCacheKey
        self.promptLength = promptLength
        self.transcriptTokenCount = transcriptTokenCount
        self.createdAtUs = createdAtUs
        self.updatedAtUs = updatedAtUs
    }
}

public enum SmeltEventType: String, Codable, Sendable {
    case textStart = "text_start"
    case textDelta = "text_delta"
    case toolCallStart = "toolcall_start"
    case toolCallDelta = "toolcall_delta"
    case toolCallEnd = "toolcall_end"
    case metrics
    case done
    case canceled
    case error
}

public struct SmeltMetrics: Codable, Equatable, Sendable {
    public let promptTokens: Int
    public let generatedTokens: Int
    public let prefillTimeMs: Double
    public let generateTimeMs: Double
    public let tokensPerSecond: Double
    public let snapshotBytes: Int
    public let requestedMaxTokens: Int?
    public let effectiveMaxTokens: Int?
    public let toolCallMinTokens: Int?
    public let toolCallBudgetWasLifted: Bool?

    public init(
        promptTokens: Int,
        generatedTokens: Int,
        prefillTimeMs: Double,
        generateTimeMs: Double,
        tokensPerSecond: Double,
        snapshotBytes: Int,
        requestedMaxTokens: Int? = nil,
        effectiveMaxTokens: Int? = nil,
        toolCallMinTokens: Int? = nil,
        toolCallBudgetWasLifted: Bool? = nil
    ) {
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.prefillTimeMs = prefillTimeMs
        self.generateTimeMs = generateTimeMs
        self.tokensPerSecond = tokensPerSecond
        self.snapshotBytes = snapshotBytes
        self.requestedMaxTokens = requestedMaxTokens
        self.effectiveMaxTokens = effectiveMaxTokens
        self.toolCallMinTokens = toolCallMinTokens
        self.toolCallBudgetWasLifted = toolCallBudgetWasLifted
    }
}

public struct SmeltToolCallFailureDiagnostic: Codable, Equatable, Sendable {
    public let message: String
    public let sessionId: String
    public let generationId: String
    public let traceId: String
    public let promptHash: String
    public let generatedTokenIds: [Int32]
    public let generatedTokenCount: Int
    public let partialJSON: String
    public let partialJSONByteCount: Int
    public let isAccepting: Bool
    public let stopCause: String
    public let maxTokens: Int?
    public let requestedMaxTokens: Int?
    public let effectiveMaxTokens: Int?
    public let toolCallMinTokens: Int
    public let toolCallBudgetWasLifted: Bool
    public let runtimeMaxGeneratedTokens: Int
    public let toolCount: Int
    public let matcherError: String?

    public init(
        message: String = "Smelt constrained tool call did not reach an accepting JSON state",
        sessionId: String,
        generationId: String,
        traceId: String,
        promptHash: String,
        generatedTokenIds: [Int32],
        partialJSON: String,
        isAccepting: Bool = false,
        stopCause: String,
        maxTokens: Int?,
        requestedMaxTokens: Int?,
        effectiveMaxTokens: Int?,
        toolCallMinTokens: Int,
        toolCallBudgetWasLifted: Bool,
        runtimeMaxGeneratedTokens: Int,
        toolCount: Int,
        matcherError: String? = nil
    ) {
        self.message = message
        self.sessionId = sessionId
        self.generationId = generationId
        self.traceId = traceId
        self.promptHash = promptHash
        self.generatedTokenIds = generatedTokenIds
        self.generatedTokenCount = generatedTokenIds.count
        self.partialJSON = partialJSON
        self.partialJSONByteCount = Data(partialJSON.utf8).count
        self.isAccepting = isAccepting
        self.stopCause = stopCause
        self.maxTokens = maxTokens
        self.requestedMaxTokens = requestedMaxTokens
        self.effectiveMaxTokens = effectiveMaxTokens
        self.toolCallMinTokens = toolCallMinTokens
        self.toolCallBudgetWasLifted = toolCallBudgetWasLifted
        self.runtimeMaxGeneratedTokens = runtimeMaxGeneratedTokens
        self.toolCount = toolCount
        self.matcherError = matcherError
    }
}

public struct SmeltEvent: Codable, Equatable, Sendable {
    public let type: SmeltEventType
    public let sessionId: String
    public let generationId: String
    public let traceId: String
    public let id: String?
    public let name: String?
    public let tokenId: Int32?
    public let position: Int?
    public let text: String?
    public let delta: String?
    public let arguments: [String: SmeltJSONValue]?
    public let metrics: SmeltMetrics?
    public let error: String?
    public let toolCallFailure: SmeltToolCallFailureDiagnostic?
    public let decodingPolicy: SmeltDecodingPolicy?
    public let timestampUs: UInt64

    public init(
        type: SmeltEventType,
        sessionId: String,
        generationId: String,
        traceId: String,
        id: String? = nil,
        name: String? = nil,
        tokenId: Int32? = nil,
        position: Int? = nil,
        text: String? = nil,
        delta: String? = nil,
        arguments: [String: SmeltJSONValue]? = nil,
        metrics: SmeltMetrics? = nil,
        error: String? = nil,
        toolCallFailure: SmeltToolCallFailureDiagnostic? = nil,
        decodingPolicy: SmeltDecodingPolicy? = nil,
        timestampUs: UInt64 = SmeltClock.nowUs()
    ) {
        self.type = type
        self.sessionId = sessionId
        self.generationId = generationId
        self.traceId = traceId
        self.id = id
        self.name = name
        self.tokenId = tokenId
        self.position = position
        self.text = text
        self.delta = delta
        self.arguments = arguments
        self.metrics = metrics
        self.error = error
        self.toolCallFailure = toolCallFailure
        self.decodingPolicy = decodingPolicy
        self.timestampUs = timestampUs
    }
}

public struct SmeltTurnResult: Sendable {
    public let session: SmeltSessionInfo
    public let generationId: String
    public let traceId: String
    public let tokens: [Int32]
    public let canceled: Bool
    public let committed: Bool
    public let metrics: SmeltMetrics

    public init(
        session: SmeltSessionInfo,
        generationId: String,
        traceId: String,
        tokens: [Int32],
        canceled: Bool,
        committed: Bool,
        metrics: SmeltMetrics
    ) {
        self.session = session
        self.generationId = generationId
        self.traceId = traceId
        self.tokens = tokens
        self.canceled = canceled
        self.committed = committed
        self.metrics = metrics
    }
}

public struct SmeltSessionStats: Codable, Equatable, Sendable {
    public let info: SmeltSessionInfo
    public let systemTokens: Int
    public let transcriptTokens: Int
    public let capturedTokens: Int
    public let replayTokens: Int
    public let snapshotBytes: Int
    public let toolCount: Int
    public let metadata: [String: String]

    public init(
        info: SmeltSessionInfo,
        systemTokens: Int,
        transcriptTokens: Int,
        capturedTokens: Int,
        replayTokens: Int,
        snapshotBytes: Int,
        toolCount: Int,
        metadata: [String: String]
    ) {
        self.info = info
        self.systemTokens = systemTokens
        self.transcriptTokens = transcriptTokens
        self.capturedTokens = capturedTokens
        self.replayTokens = replayTokens
        self.snapshotBytes = snapshotBytes
        self.toolCount = toolCount
        self.metadata = metadata
    }
}

public struct SmeltTextRuntimeStats: Codable, Equatable, Sendable {
    public let maxContextTokens: Int
    public let liveSessionCount: Int
    public let maxLiveSessions: Int
    public let prefixCacheEntryCount: Int
    public let maxPrefixCacheEntries: Int
    public let activeGenerationCount: Int
    public let memory: SmeltRuntime.MemoryStats

    public init(
        maxContextTokens: Int,
        liveSessionCount: Int,
        maxLiveSessions: Int,
        prefixCacheEntryCount: Int,
        maxPrefixCacheEntries: Int,
        activeGenerationCount: Int,
        memory: SmeltRuntime.MemoryStats
    ) {
        self.maxContextTokens = maxContextTokens
        self.liveSessionCount = liveSessionCount
        self.maxLiveSessions = maxLiveSessions
        self.prefixCacheEntryCount = prefixCacheEntryCount
        self.maxPrefixCacheEntries = maxPrefixCacheEntries
        self.activeGenerationCount = activeGenerationCount
        self.memory = memory
    }
}

public struct SmeltSessionBackupInfo: Codable, Equatable, Sendable {
    public let directoryPath: String
    public let manifestPath: String
    public let snapshotPath: String
    public let snapshotFileBytes: Int
    public let snapshotWriteMode: SmeltPromptSnapshotWriteMode
    public let session: SmeltSessionInfo
    public let sessionStats: SmeltSessionStats

    public init(
        directoryPath: String,
        manifestPath: String,
        snapshotPath: String,
        snapshotFileBytes: Int = 0,
        snapshotWriteMode: SmeltPromptSnapshotWriteMode = .serialized,
        session: SmeltSessionInfo,
        sessionStats: SmeltSessionStats
    ) {
        self.directoryPath = directoryPath
        self.manifestPath = manifestPath
        self.snapshotPath = snapshotPath
        self.snapshotFileBytes = snapshotFileBytes
        self.snapshotWriteMode = snapshotWriteMode
        self.session = session
        self.sessionStats = sessionStats
    }
}

private struct SmeltSessionBackupManifest: Codable {
    let schemaVersion: Int
    let sessionId: String
    let forkedFrom: String?
    let prefixCacheKey: String
    let systemTokenIds: [Int32]
    let transcriptTokenIds: [Int32]
    let tools: [SmeltToolDescriptor]
    let metadata: [String: String]
}

public struct SmeltTraceRecord: Codable, Sendable {
    public let schemaVersion: Int
    public let traceId: String
    public let sessionId: String
    public let generationId: String
    public let eventType: String
    public let packageHash: String
    public let tokenizerHash: String?
    public let camSemanticSHA256: String?
    public let exportABISHA256: String?
    public let prefixCacheKey: String?
    public let sampler: String
    public let promptHash: String?
    public let contextTokenIds: [Int32]?
    public let id: String?
    public let name: String?
    public let tokenId: Int32?
    public let position: Int?
    public let textByteCount: Int?
    public let deltaByteCount: Int?
    public let arguments: [String: SmeltJSONValue]?
    public let stepLatencyUs: UInt64?
    public let metrics: SmeltMetrics?
    public let error: String?
    public let toolCallFailure: SmeltToolCallFailureDiagnostic?
    public let decodingPolicy: SmeltDecodingPolicy?
    public let timestampUs: UInt64

    public init(
        schemaVersion: Int = 1,
        traceId: String,
        sessionId: String,
        generationId: String,
        eventType: String,
        packageHash: String,
        tokenizerHash: String?,
        camSemanticSHA256: String? = nil,
        exportABISHA256: String? = nil,
        prefixCacheKey: String?,
        sampler: String,
        promptHash: String?,
        contextTokenIds: [Int32]? = nil,
        id: String? = nil,
        name: String? = nil,
        tokenId: Int32? = nil,
        position: Int? = nil,
        textByteCount: Int? = nil,
        deltaByteCount: Int? = nil,
        arguments: [String: SmeltJSONValue]? = nil,
        stepLatencyUs: UInt64? = nil,
        metrics: SmeltMetrics? = nil,
        error: String? = nil,
        toolCallFailure: SmeltToolCallFailureDiagnostic? = nil,
        decodingPolicy: SmeltDecodingPolicy? = nil,
        timestampUs: UInt64 = SmeltClock.nowUs()
    ) {
        self.schemaVersion = schemaVersion
        self.traceId = traceId
        self.sessionId = sessionId
        self.generationId = generationId
        self.eventType = eventType
        self.packageHash = packageHash
        self.tokenizerHash = tokenizerHash
        self.camSemanticSHA256 = camSemanticSHA256
        self.exportABISHA256 = exportABISHA256
        self.prefixCacheKey = prefixCacheKey
        self.sampler = sampler
        self.promptHash = promptHash
        self.contextTokenIds = contextTokenIds
        self.id = id
        self.name = name
        self.tokenId = tokenId
        self.position = position
        self.textByteCount = textByteCount
        self.deltaByteCount = deltaByteCount
        self.arguments = arguments
        self.stepLatencyUs = stepLatencyUs
        self.metrics = metrics
        self.error = error
        self.toolCallFailure = toolCallFailure
        self.decodingPolicy = decodingPolicy
        self.timestampUs = timestampUs
    }
}

public enum SmeltTextRuntimeError: Error, CustomStringConvertible, CodedError {
    case sessionNotFound(String)
    case sessionLimitExceeded(Int)
    case transcriptLimitExceeded(limit: Int, requested: Int)
    case generationAlreadyActive(String)
    case invalidMaxTokens(Int)
    case toolCallingRequiresTokenizer
    case toolCallDidNotComplete(SmeltToolCallFailureDiagnostic)
    case invalidBackupPath(String)

    public var description: String {
        switch self {
        case .sessionNotFound(let id):
            return "Smelt session not found: \(id)"
        case .sessionLimitExceeded(let limit):
            return "Smelt session limit exceeded: \(limit)"
        case .transcriptLimitExceeded(let limit, let requested):
            return "Smelt transcript limit exceeded: \(requested) > \(limit)"
        case .generationAlreadyActive(let id):
            return "Smelt generation already active: \(id)"
        case .invalidMaxTokens(let value):
            return "Smelt maxTokens must be positive, got \(value)"
        case .toolCallingRequiresTokenizer:
            return "Smelt tool calling requires a package tokenizer"
        case .toolCallDidNotComplete(let diagnostic):
            return diagnostic.message
        case .invalidBackupPath(let path):
            return "Smelt invalid backup path: \(path)"
        }
    }

    public var code: String {
        switch self {
        case .sessionNotFound: return "session_not_found"
        case .sessionLimitExceeded: return "session_limit_exceeded"
        case .transcriptLimitExceeded: return "transcript_limit_exceeded"
        case .generationAlreadyActive: return "generation_already_active"
        case .invalidMaxTokens: return "invalid_max_tokens"
        case .toolCallingRequiresTokenizer: return "tool_calling_requires_tokenizer"
        case .toolCallDidNotComplete: return "tool_call_failed"
        case .invalidBackupPath: return "invalid_backup_path"
        }
    }

    public var toolCallFailure: SmeltToolCallFailureDiagnostic? {
        switch self {
        case .toolCallDidNotComplete(let diagnostic):
            return diagnostic
        default:
            return nil
        }
    }
}

public final class SmeltCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var canceled = false

    public init() {}

    public func cancel() {
        lock.lock()
        canceled = true
        lock.unlock()
    }

    public var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return canceled
    }
}

public final class SmeltTextRuntime: @unchecked Sendable {
    public typealias EventHandler = @Sendable (SmeltEvent) -> Void

    private struct Session {
        var id: String
        var forkedFrom: String?
        var prefixCacheKey: String
        var systemTokenIds: [Int32]
        var snapshot: SmeltPromptSnapshot
        var transcriptTokenIds: [Int32]
        var tools: [SmeltToolDescriptor]
        var metadata: [String: String]
        var createdAtUs: UInt64
        var updatedAtUs: UInt64

        var info: SmeltSessionInfo {
            SmeltSessionInfo(
                id: id,
                forkedFrom: forkedFrom,
                prefixCacheKey: prefixCacheKey,
                promptLength: snapshot.promptLength,
                transcriptTokenCount: transcriptTokenIds.count,
                createdAtUs: createdAtUs,
                updatedAtUs: updatedAtUs
            )
        }
    }

    private let queue = DispatchQueue(label: "smelt.smeltpkg.runtime")
    private let activeLock = NSLock()
    private var activeGenerations: [String: SmeltCancellation] = [:]

    private let model: SmeltModel
    private let tokenizer: SmeltTokenizer?
    private let packageHash: String
    private let tokenizerHash: String?
    private let camSemanticSHA256: String?
    private let exportABISHA256: String?
    private let inferenceEOSTokens: [Int32]
    private let traceWriter: SmeltTraceWriter?
    private let limits: SmeltLimits

    private var sessions: [String: Session] = [:]
    private var prefixSnapshots: [String: SmeltPromptSnapshot] = [:]
    private var prefixOrder: [String] = []

    public var maxContextTokens: Int { model.maxContextTokens }

    public init(
        model: SmeltModel,
        tokenizer: SmeltTokenizer?,
        inferenceEOSTokens: [Int32],
        packagePath: String,
        camSemanticSHA256: String? = nil,
        exportABISHA256: String? = nil,
        traceDirectory: String? = ProcessInfo.processInfo.environment["SMELT_TRACE_DIR"],
        limits: SmeltLimits = SmeltLimits()
    ) throws {
        self.model = model
        self.tokenizer = tokenizer
        self.inferenceEOSTokens = inferenceEOSTokens
        let tokenizerPath = "\(packagePath)/tokenizer.json"
        self.tokenizerHash = tokenizer == nil || !FileManager.default.fileExists(atPath: tokenizerPath)
            ? nil
            : try SmeltHash.fileHash(path: tokenizerPath)
        self.packageHash = try SmeltHash.packageHash(packagePath: packagePath)
        self.camSemanticSHA256 = camSemanticSHA256
        self.exportABISHA256 = exportABISHA256
        self.traceWriter = traceDirectory.map { SmeltTraceWriter(directoryPath: $0) }
        self.limits = limits
    }

    /// Close any sessions whose `updatedAtUs` is older than `limits.maxIdleSeconds`.
    /// Returns the IDs of removed sessions. No-op when `maxIdleSeconds` is nil.
    /// Do not call from inside another runtime call on the same thread — the
    /// underlying `queue.sync` will deadlock the serial queue.
    @discardableResult
    public func sweepIdleSessions() -> [String] {
        queue.sync { sweepIdleSessionsLocked() }
    }

    private func sweepIdleSessionsLocked() -> [String] {
        guard let maxIdleSeconds = limits.maxIdleSeconds, maxIdleSeconds > 0 else {
            return []
        }
        let thresholdUs = UInt64(maxIdleSeconds) * 1_000_000
        let now = SmeltClock.nowUs()
        let toRemove = sessions.filter { _, session in
            now > session.updatedAtUs && now - session.updatedAtUs > thresholdUs
        }.map(\.key)
        for id in toRemove {
            sessions.removeValue(forKey: id)
        }
        return toRemove
    }

    private func requireSessionSlotLocked() throws {
        _ = sweepIdleSessionsLocked()
        guard sessions.count < limits.maxLiveSessions else {
            throw SmeltTextRuntimeError.sessionLimitExceeded(limits.maxLiveSessions)
        }
    }

    public func openSession(
        _ config: SmeltSessionConfig = SmeltSessionConfig()
    ) throws -> SmeltSessionInfo {
        try sync {
            try requireSessionSlotLocked()
            let prefixKey = try makePrefixCacheKey(config: config)
            let snapshot: SmeltPromptSnapshot
            if let cached = prefixSnapshots[prefixKey] {
                snapshot = cached
            } else {
                snapshot = try model.captureBasePrompt(tokenIds: config.systemTokenIds)
                rememberPrefixSnapshot(snapshot, key: prefixKey)
            }

            let now = SmeltClock.nowUs()
            let id = config.id ?? "smelt-session-\(UUID().uuidString.lowercased())"
            let session = Session(
                id: id,
                forkedFrom: nil,
                prefixCacheKey: prefixKey,
                systemTokenIds: config.systemTokenIds,
                snapshot: snapshot,
                transcriptTokenIds: [],
                tools: config.tools,
                metadata: config.metadata,
                createdAtUs: now,
                updatedAtUs: now
            )
            sessions[id] = session
            return session.info
        }
    }

    public func forkSession(
        _ sourceSessionId: String,
        newSessionId: String? = nil
    ) throws -> SmeltSessionInfo {
        try sync {
            guard var source = sessions[sourceSessionId] else {
                throw SmeltTextRuntimeError.sessionNotFound(sourceSessionId)
            }
            try requireSessionSlotLocked()
            let now = SmeltClock.nowUs()
            source.id = newSessionId ?? "smelt-session-\(UUID().uuidString.lowercased())"
            source.forkedFrom = sourceSessionId
            source.createdAtUs = now
            source.updatedAtUs = now
            sessions[source.id] = source
            return source.info
        }
    }

    public func closeSession(_ sessionId: String) {
        _ = queue.sync {
            sessions.removeValue(forKey: sessionId)
        }
    }

    @discardableResult
    public func cancel(generationId: String) -> Bool {
        activeLock.lock()
        defer { activeLock.unlock() }
        guard let token = activeGenerations[generationId] else {
            return false
        }
        token.cancel()
        return true
    }

    public func generateTurn(
        sessionId: String,
        userTokenIds: [Int32],
        options: SmeltGenerateOptions = SmeltGenerateOptions(),
        onEvent: EventHandler
    ) throws -> SmeltTurnResult {
        try sync {
            guard var session = sessions[sessionId] else {
                throw SmeltTextRuntimeError.sessionNotFound(sessionId)
            }
            if let maxTokens = options.maxTokens, maxTokens <= 0 {
                throw SmeltTextRuntimeError.invalidMaxTokens(maxTokens)
            }
            let activeTools = options.resolvedTools(sessionTools: session.tools)
            let promptTokens = session.snapshot.promptLength + userTokenIds.count
            guard promptTokens < model.maxContextTokens
            else {
                throw SmeltTextRuntimeError.transcriptLimitExceeded(
                    limit: model.maxContextTokens,
                    requested: promptTokens
                )
            }
            let availableOutputTokens = max(model.maxContextTokens - promptTokens, 0)
            let requestedEffectiveMaxTokens = options.effectiveMaxTokens(
                hasActiveTools: !activeTools.isEmpty
            )
            let effectiveMaxTokens = requestedEffectiveMaxTokens.map {
                max(1, min($0, limits.maxGeneratedTokensPerTurn, availableOutputTokens))
            }
            let resolvedMaxTokens = options.maxTokens.map {
                max(1, min($0, limits.maxGeneratedTokensPerTurn, availableOutputTokens))
            }
            let toolCallBudgetWasLifted: Bool
            if !activeTools.isEmpty,
               let requestedMaxTokens = options.maxTokens,
               let effectiveMaxTokens
            {
                toolCallBudgetWasLifted = effectiveMaxTokens > requestedMaxTokens
            } else {
                toolCallBudgetWasLifted = false
            }

            let requestedTranscriptTokens =
                session.transcriptTokenIds.count + userTokenIds.count
                + min(effectiveMaxTokens ?? limits.maxGeneratedTokensPerTurn,
                      limits.maxGeneratedTokensPerTurn)
            guard requestedTranscriptTokens <= limits.maxTranscriptTokens else {
                throw SmeltTextRuntimeError.transcriptLimitExceeded(
                    limit: limits.maxTranscriptTokens,
                    requested: requestedTranscriptTokens
                )
            }

            let generationId =
                options.generationId ?? "smelt-generation-\(UUID().uuidString.lowercased())"
            let traceId = UUID().uuidString.lowercased()
            let cancellation = SmeltCancellation()
            try registerActiveGeneration(generationId, cancellation: cancellation)
            defer { unregisterActiveGeneration(generationId) }

            let contextTokenIds =
                session.systemTokenIds + session.transcriptTokenIds + userTokenIds
            let promptHash = SmeltHash.tokenHash(
                contextTokenIds
            )
            let runtimeContextPressure =
                (options.decodingPolicyRequest?.contextPressure
                    ?? options.decodingPolicy?.contextPressure
                    ?? SmeltContextPressure())
                .withRuntime(
                    promptTokens: promptTokens,
                    contextLimit: model.maxContextTokens,
                    effectiveMaxTokens: effectiveMaxTokens,
                    requestedMaxTokens: options.maxTokens,
                    resolvedMaxTokens: resolvedMaxTokens,
                    toolCallMinTokens: options.toolCallMinTokens
                )
            let eventDecodingPolicy: SmeltDecodingPolicy?
            let selectionMode: SmeltSelectionMode
            if let request = options.decodingPolicyRequest {
                let resolvedPolicy = SmeltDecodingPolicyResolver.resolve(
                    request.withRuntime(
                        contextPressure: runtimeContextPressure,
                        hasTools: !activeTools.isEmpty
                    )
                )
                eventDecodingPolicy = resolvedPolicy
                selectionMode = try SmeltDecodingPolicyResolver.selectionMode(
                    for: resolvedPolicy
                )
            } else {
                eventDecodingPolicy = options.decodingPolicy?.withContextPressure(
                    runtimeContextPressure
                )
                selectionMode = options.selectionMode
            }
            let sampler = SmeltHash.samplerDescription(selectionMode)
            var emittedTokens: [Int32] = []
            var streamDecoder = tokenizer?.makeStreamingDecoder()
            let constrainedMatcher: SmeltLLGuidanceMatcher?
            if !activeTools.isEmpty {
                guard let tokenizer else {
                    throw SmeltTextRuntimeError.toolCallingRequiresTokenizer
                }
                let grammar = try SmeltToolGrammar.jsonSchema(for: activeTools)
                let llgTokenizer = try SmeltLLGuidanceTokenizer(
                    tokenizer: tokenizer,
                    eosTokens: inferenceEOSTokens
                )
                constrainedMatcher = try SmeltLLGuidanceMatcher(
                    tokenizer: llgTokenizer,
                    jsonSchema: grammar
                )
            } else {
                constrainedMatcher = nil
            }
            var generatedToolJSON = ""
            var emittedToolCall = false
            var generationError: Error?
            var constrainedStopCause: String?
            var lastStepAt = SmeltClock.nowUs()

            let startEvent = SmeltEvent(
                type: .textStart,
                sessionId: session.id,
                generationId: generationId,
                traceId: traceId,
                decodingPolicy: eventDecodingPolicy
            )
            emit(
                startEvent,
                promptHash: promptHash,
                session: session,
                sampler: sampler,
                contextTokenIds: contextTokenIds,
                stepLatencyUs: nil,
                onEvent: onEvent
            )

            let stateful = try model.generateAndCapture(
                from: session.snapshot,
                tokenIds: userTokenIds,
                selectionMode: selectionMode,
                allowedTokenMask: constrainedMatcher.map { matcher in
                    { try matcher.computeMask() }
                },
                // Tool-calling uses constrained decode that expects the call
                // immediately — force non-thinking regardless of package policy.
                suppressThinking: true
            ) { token in
                emittedTokens.append(token.id)
                let now = SmeltClock.nowUs()
                let stepLatency = now >= lastStepAt ? now - lastStepAt : 0
                lastStepAt = now
                let text = tokenizer.flatMap { tok in
                    streamDecoder?.decode(tokenId: token.id, tokenizer: tok)
                }
                if let constrainedMatcher {
                    if let text {
                        generatedToolJSON += text
                    }
                    do {
                        try constrainedMatcher.consume(tokenIds: [token.id])
                        if constrainedMatcher.isAccepting {
                            try emitCompletedToolCall(
                                jsonText: generatedToolJSON,
                                session: session,
                                generationId: generationId,
                                traceId: traceId,
                                promptHash: promptHash,
                                sampler: sampler,
                                stepLatencyUs: stepLatency,
                                onEvent: onEvent
                            )
                            emittedToolCall = true
                            return false
                        }
                    } catch {
                        generationError = error
                        return false
                    }
                } else {
                    let deltaEvent = SmeltEvent(
                        type: .textDelta,
                        sessionId: session.id,
                        generationId: generationId,
                        traceId: traceId,
                        tokenId: token.id,
                        position: token.position,
                        text: text
                    )
                    emit(
                        deltaEvent,
                        promptHash: promptHash,
                        session: session,
                        sampler: sampler,
                        contextTokenIds: nil,
                        stepLatencyUs: stepLatency,
                        onEvent: onEvent
                    )
                }

                if cancellation.isCancelled {
                    constrainedStopCause = "canceled"
                    return false
                }
                if emittedTokens.count >= limits.maxGeneratedTokensPerTurn {
                    constrainedStopCause = "runtime_max_generated_tokens"
                    return false
                }
                if let maxTokens = effectiveMaxTokens, emittedTokens.count >= maxTokens {
                    constrainedStopCause = "max_tokens"
                    return false
                }
                return true
            }

            func makeToolCallFailureDiagnostic(
                stopCause: String,
                matcherError: String? = nil
            ) -> SmeltToolCallFailureDiagnostic {
                SmeltToolCallFailureDiagnostic(
                    sessionId: session.id,
                    generationId: generationId,
                    traceId: traceId,
                    promptHash: promptHash,
                    generatedTokenIds: emittedTokens,
                    partialJSON: generatedToolJSON,
                    stopCause: stopCause,
                    maxTokens: effectiveMaxTokens,
                    requestedMaxTokens: options.maxTokens,
                    effectiveMaxTokens: effectiveMaxTokens,
                    toolCallMinTokens: options.toolCallMinTokens,
                    toolCallBudgetWasLifted: toolCallBudgetWasLifted,
                    runtimeMaxGeneratedTokens: limits.maxGeneratedTokensPerTurn,
                    toolCount: activeTools.count,
                    matcherError: matcherError
                )
            }

            func throwToolCallFailure(_ diagnostic: SmeltToolCallFailureDiagnostic) throws -> Never {
                let errorEvent = SmeltEvent(
                    type: .error,
                    sessionId: session.id,
                    generationId: generationId,
                    traceId: traceId,
                    error: diagnostic.message,
                    toolCallFailure: diagnostic
                )
                emit(
                    errorEvent,
                    promptHash: promptHash,
                    session: session,
                    sampler: sampler,
                    contextTokenIds: nil,
                    stepLatencyUs: nil,
                    onEvent: onEvent
                )
                throw SmeltTextRuntimeError.toolCallDidNotComplete(diagnostic)
            }

            if let generationError {
                if constrainedMatcher != nil && !emittedToolCall {
                    try throwToolCallFailure(
                        makeToolCallFailureDiagnostic(
                            stopCause: "matcher_error",
                            matcherError: "\(generationError)"
                        )
                    )
                }
                throw generationError
            }

            let canceled = cancellation.isCancelled
            if constrainedMatcher != nil && !emittedToolCall && !canceled {
                try throwToolCallFailure(
                    makeToolCallFailureDiagnostic(
                        stopCause: constrainedStopCause ?? "model_stopped"
                    )
                )
            }
            let committed = !canceled || options.commitOnCancel
            let metrics = SmeltMetrics(
                promptTokens: session.snapshot.promptLength + userTokenIds.count,
                generatedTokens: stateful.result.tokens.count,
                prefillTimeMs: stateful.result.prefillTime * 1_000,
                generateTimeMs: stateful.result.generateTime * 1_000,
                tokensPerSecond: stateful.result.tokensPerSecond,
                snapshotBytes: stateful.snapshot.byteCount,
                requestedMaxTokens: !activeTools.isEmpty ? options.maxTokens : nil,
                effectiveMaxTokens: !activeTools.isEmpty ? effectiveMaxTokens : nil,
                toolCallMinTokens: !activeTools.isEmpty ? options.toolCallMinTokens : nil,
                toolCallBudgetWasLifted: !activeTools.isEmpty ? toolCallBudgetWasLifted : nil
            )

            if committed {
                session.snapshot = stateful.snapshot
                session.transcriptTokenIds += userTokenIds + stateful.result.tokens
                session.updatedAtUs = SmeltClock.nowUs()
                sessions[session.id] = session
            }

            let metricsEvent = SmeltEvent(
                type: .metrics,
                sessionId: session.id,
                generationId: generationId,
                traceId: traceId,
                metrics: metrics
            )
            emit(
                metricsEvent,
                promptHash: promptHash,
                session: session,
                sampler: sampler,
                contextTokenIds: nil,
                stepLatencyUs: nil,
                onEvent: onEvent
            )

            let terminalEvent = SmeltEvent(
                type: canceled ? .canceled : .done,
                sessionId: session.id,
                generationId: generationId,
                traceId: traceId
            )
            emit(
                terminalEvent,
                promptHash: promptHash,
                session: session,
                sampler: sampler,
                contextTokenIds: nil,
                stepLatencyUs: nil,
                onEvent: onEvent
            )

            return SmeltTurnResult(
                session: (sessions[session.id] ?? session).info,
                generationId: generationId,
                traceId: traceId,
                tokens: stateful.result.tokens,
                canceled: canceled,
                committed: committed,
                metrics: metrics
            )
        }
    }

    private func emitCompletedToolCall(
        jsonText: String,
        session: Session,
        generationId: String,
        traceId: String,
        promptHash: String,
        sampler: String,
        stepLatencyUs: UInt64,
        onEvent: EventHandler
    ) throws {
        let data = Data(jsonText.utf8)
        let call = try JSONDecoder().decode(SmeltGeneratedToolCall.self, from: data)
        let callId = SmeltToolCallID.next()
        let argumentsJSON = try canonicalJSONString(call.arguments)

        let startEvent = SmeltEvent(
            type: .toolCallStart,
            sessionId: session.id,
            generationId: generationId,
            traceId: traceId,
            id: callId,
            name: call.name
        )
        emit(
            startEvent,
            promptHash: promptHash,
            session: session,
            sampler: sampler,
            contextTokenIds: nil,
            stepLatencyUs: nil,
            onEvent: onEvent
        )

        let deltaEvent = SmeltEvent(
            type: .toolCallDelta,
            sessionId: session.id,
            generationId: generationId,
            traceId: traceId,
            id: callId,
            name: call.name,
            delta: argumentsJSON
        )
        emit(
            deltaEvent,
            promptHash: promptHash,
            session: session,
            sampler: sampler,
            contextTokenIds: nil,
            stepLatencyUs: stepLatencyUs,
            onEvent: onEvent
        )

        let endEvent = SmeltEvent(
            type: .toolCallEnd,
            sessionId: session.id,
            generationId: generationId,
            traceId: traceId,
            id: callId,
            name: call.name,
            arguments: call.arguments
        )
        emit(
            endEvent,
            promptHash: promptHash,
            session: session,
            sampler: sampler,
            contextTokenIds: nil,
            stepLatencyUs: nil,
            onEvent: onEvent
        )
    }

    public func sessionInfo(_ sessionId: String) throws -> SmeltSessionInfo {
        try sync {
            guard let session = sessions[sessionId] else {
                throw SmeltTextRuntimeError.sessionNotFound(sessionId)
            }
            return session.info
        }
    }

    public func sessionInfos() -> [SmeltSessionInfo] {
        syncNonThrowing {
            sessions.values
                .map(\.info)
                .sorted { $0.createdAtUs < $1.createdAtUs }
        }
    }

    public func sessionStats(_ sessionId: String) throws -> SmeltSessionStats {
        try sync {
            guard let session = sessions[sessionId] else {
                throw SmeltTextRuntimeError.sessionNotFound(sessionId)
            }
            return stats(for: session)
        }
    }

    public func runtimeStats() -> SmeltTextRuntimeStats {
        let activeCount = activeGenerationCount()
        return syncNonThrowing {
            SmeltTextRuntimeStats(
                maxContextTokens: model.maxContextTokens,
                liveSessionCount: sessions.count,
                maxLiveSessions: limits.maxLiveSessions,
                prefixCacheEntryCount: prefixSnapshots.count,
                maxPrefixCacheEntries: limits.maxPrefixCacheEntries,
                activeGenerationCount: activeCount,
                memory: model.memoryStats
            )
        }
    }

    public func backupSession(
        _ sessionId: String,
        toDirectory directoryPath: String
    ) throws -> SmeltSessionBackupInfo {
        try sync {
            guard var session = sessions[sessionId] else {
                throw SmeltTextRuntimeError.sessionNotFound(sessionId)
            }
            let directoryURL = URL(fileURLWithPath: directoryPath, isDirectory: true)
            try FileManager.default.createDirectory(
                at: directoryURL,
                withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700]
            )
            let manifestURL = directoryURL.appendingPathComponent("session.json")
            let snapshotURL = directoryURL.appendingPathComponent("snapshot.smkvcache")
            try? FileManager.default.removeItem(at: manifestURL)

            let manifest = SmeltSessionBackupManifest(
                schemaVersion: 1,
                sessionId: session.id,
                forkedFrom: session.forkedFrom,
                prefixCacheKey: session.prefixCacheKey,
                systemTokenIds: session.systemTokenIds,
                transcriptTokenIds: session.transcriptTokenIds,
                tools: session.tools,
                metadata: session.metadata
            )
            let manifestData = try JSONEncoder().encode(manifest)
            guard FileManager.default.createFile(
                atPath: manifestURL.path,
                contents: manifestData,
                attributes: [.posixPermissions: 0o600]
            ) else {
                throw SmeltTextRuntimeError.invalidBackupPath(manifestURL.path)
            }
            let snapshotWrite = try session.snapshot.write(to: snapshotURL)
            session.snapshot = try SmeltPromptSnapshot.read(from: snapshotURL)
            sessions[sessionId] = session

            return SmeltSessionBackupInfo(
                directoryPath: directoryURL.path,
                manifestPath: manifestURL.path,
                snapshotPath: snapshotURL.path,
                snapshotFileBytes: snapshotWrite.fileBytes,
                snapshotWriteMode: snapshotWrite.mode,
                session: session.info,
                sessionStats: stats(for: session)
            )
        }
    }

    public func restoreSession(
        fromDirectory directoryPath: String,
        newSessionId: String? = nil
    ) throws -> SmeltSessionInfo {
        let directoryURL = URL(fileURLWithPath: directoryPath, isDirectory: true)
        let manifestURL = directoryURL.appendingPathComponent("session.json")
        let snapshotURL = directoryURL.appendingPathComponent("snapshot.smkvcache")
        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(
            SmeltSessionBackupManifest.self,
            from: manifestData
        )
        let snapshot = try SmeltPromptSnapshot.read(from: snapshotURL)

        return try sync {
            let id = newSessionId ?? manifest.sessionId
            if sessions[id] == nil {
                try requireSessionSlotLocked()
            }
            let now = SmeltClock.nowUs()
            let session = Session(
                id: id,
                forkedFrom: newSessionId == nil ? manifest.forkedFrom : manifest.sessionId,
                prefixCacheKey: manifest.prefixCacheKey,
                systemTokenIds: manifest.systemTokenIds,
                snapshot: snapshot,
                transcriptTokenIds: manifest.transcriptTokenIds,
                tools: manifest.tools,
                metadata: manifest.metadata,
                createdAtUs: now,
                updatedAtUs: now
            )
            sessions[id] = session
            return session.info
        }
    }

    private func sync<T>(_ body: () throws -> T) throws -> T {
        var output: Result<T, Error>!
        queue.sync {
            output = Result { try body() }
        }
        return try output.get()
    }

    private func syncNonThrowing<T>(_ body: () -> T) -> T {
        queue.sync(execute: body)
    }

    private func rememberPrefixSnapshot(_ snapshot: SmeltPromptSnapshot, key: String) {
        prefixSnapshots[key] = snapshot
        prefixOrder.removeAll { $0 == key }
        prefixOrder.append(key)

        while prefixOrder.count > limits.maxPrefixCacheEntries {
            let evicted = prefixOrder.removeFirst()
            prefixSnapshots.removeValue(forKey: evicted)
        }
    }

    private func registerActiveGeneration(
        _ generationId: String,
        cancellation: SmeltCancellation
    ) throws {
        activeLock.lock()
        defer { activeLock.unlock() }
        guard activeGenerations[generationId] == nil else {
            throw SmeltTextRuntimeError.generationAlreadyActive(generationId)
        }
        activeGenerations[generationId] = cancellation
    }

    private func unregisterActiveGeneration(_ generationId: String) {
        activeLock.lock()
        activeGenerations.removeValue(forKey: generationId)
        activeLock.unlock()
    }

    private func activeGenerationCount() -> Int {
        activeLock.lock()
        defer { activeLock.unlock() }
        return activeGenerations.count
    }

    private func stats(for session: Session) -> SmeltSessionStats {
        SmeltSessionStats(
            info: session.info,
            systemTokens: session.systemTokenIds.count,
            transcriptTokens: session.transcriptTokenIds.count,
            capturedTokens: session.snapshot.capturedLength,
            replayTokens: session.snapshot.replayTokenIds.count,
            snapshotBytes: session.snapshot.byteCount,
            toolCount: session.tools.count,
            metadata: session.metadata
        )
    }

    private func makePrefixCacheKey(
        config: SmeltSessionConfig
    ) throws -> String {
        var data = Data()
        data.append(Data(packageHash.utf8))
        data.append(SmeltHash.tokenData(config.systemTokenIds))
        data.append(try SmeltHash.canonicalData(config.tools))
        data.append(try SmeltHash.canonicalData(config.metadata))
        return SmeltHash.sha256Hex(data)
    }

    private func canonicalJSONString<T: Encodable>(_ value: T) throws -> String {
        try SmeltJSON.canonicalString(value)
    }

    private func emit(
        _ event: SmeltEvent,
        promptHash: String?,
        session: Session,
        sampler: String,
        contextTokenIds: [Int32]?,
        stepLatencyUs: UInt64?,
        onEvent: EventHandler
    ) {
        traceWriter?.write(
            SmeltTraceRecord(
                traceId: event.traceId,
                sessionId: event.sessionId,
                generationId: event.generationId,
                eventType: event.type.rawValue,
                packageHash: packageHash,
                tokenizerHash: tokenizerHash,
                camSemanticSHA256: camSemanticSHA256,
                exportABISHA256: exportABISHA256,
                prefixCacheKey: session.prefixCacheKey,
                sampler: sampler,
                promptHash: promptHash,
                contextTokenIds: contextTokenIds,
                id: event.id,
                name: event.name,
                tokenId: event.tokenId,
                position: event.position,
                textByteCount: event.text.map { Data($0.utf8).count },
                deltaByteCount: event.delta.map { Data($0.utf8).count },
                arguments: event.arguments,
                stepLatencyUs: stepLatencyUs,
                metrics: event.metrics,
                error: event.error,
                toolCallFailure: event.toolCallFailure,
                decodingPolicy: event.decodingPolicy,
                timestampUs: event.timestampUs
            )
        )
        onEvent(event)
    }
}

public enum SmeltClock {
    public static func nowUs() -> UInt64 {
        UInt64(Date().timeIntervalSince1970 * 1_000_000)
    }
}

public final class SmeltTraceWriter: @unchecked Sendable {
    private let directoryURL: URL
    private let lock = NSLock()

    public init(directoryPath: String) {
        self.directoryURL = URL(fileURLWithPath: directoryPath, isDirectory: true)
        try? FileManager.default.createDirectory(
            at: directoryURL,
            withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700]
        )
    }

    public func write(_ record: SmeltTraceRecord) {
        lock.lock()
        defer { lock.unlock() }

        do {
            var data = try SmeltJSON.canonicalData(record)
            data.append(0x0A)
            let fileURL = directoryURL.appendingPathComponent(
                "\(sanitize(record.traceId)).jsonl"
            )
            if !FileManager.default.fileExists(atPath: fileURL.path) {
                FileManager.default.createFile(
                    atPath: fileURL.path,
                    contents: nil,
                    attributes: [.posixPermissions: 0o600]
                )
            }
            let handle = try FileHandle(forWritingTo: fileURL)
            defer { try? handle.close() }
            try handle.seekToEnd()
            try handle.write(contentsOf: data)
        } catch {
            fputs("Smelt trace write failed: \(error)\n", stderr)
        }
    }

    private func sanitize(_ value: String) -> String {
        value.filter { char in
            char.isLetter || char.isNumber || char == "-" || char == "_"
        }
    }
}

enum SmeltHash {
    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func fileHash(path: String) throws -> String {
        try sha256Hex(Data(contentsOf: URL(fileURLWithPath: path)))
    }

    static func packageHash(packagePath: String) throws -> String {
        var data = Data()
        let manifestPath = "\(packagePath)/manifest.json"
        data.append(try Data(contentsOf: URL(fileURLWithPath: manifestPath)))
        let tokenizerPath = "\(packagePath)/tokenizer.json"
        if FileManager.default.fileExists(atPath: tokenizerPath) {
            data.append(try Data(contentsOf: URL(fileURLWithPath: tokenizerPath)))
        }
        return sha256Hex(data)
    }

    static func tokenHash(_ tokenIds: [Int32]) -> String {
        sha256Hex(tokenData(tokenIds))
    }

    static func tokenData(_ tokenIds: [Int32]) -> Data {
        var data = Data()
        data.reserveCapacity(tokenIds.count * MemoryLayout<Int32>.size)
        for token in tokenIds {
            var value = token.littleEndian
            withUnsafeBytes(of: &value) { bytes in
                data.append(contentsOf: bytes)
            }
        }
        return data
    }

    static func canonicalData<T: Encodable>(_ value: T) throws -> Data {
        try SmeltJSON.canonicalData(value)
    }

    static func samplerDescription(_ mode: SmeltSelectionMode) -> String {
        switch mode {
        case .argmax:
            return "argmax"
        case let .temperature(temp, seed):
            return String(format: "temperature:%.6f:seed:%llu", temp, seed)
        case let .filteredTemperature(temp, topK, topP, seed):
            return String(
                format: "temperature:%.6f:top-k:%@:top-p:%.6f:seed:%llu",
                temp,
                topK.map(String.init) ?? "all",
                topP,
                seed
            )
        }
    }
}
