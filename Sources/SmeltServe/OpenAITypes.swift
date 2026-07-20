import Foundation
import SmeltRuntime

// OpenAI-compatible wire types for the reusable serving surface.

enum OpenAIRole: String, Codable, Sendable {
    case system, user, assistant, tool
}

enum OpenAIFinishReason: String, Codable, Sendable {
    case stop
    case length
    case contentFilter = "content_filter"
    case toolCalls = "tool_calls"
}

enum OpenAIToolType: String, Codable, Sendable {
    case function
}

struct OpenAIToolCallFunction: Codable, Sendable {
    let name: String
    let arguments: String
}

struct OpenAIToolCall: Codable, Sendable {
    let id: String
    let type: OpenAIToolType
    let function: OpenAIToolCallFunction
}

enum OpenAIChatContentPart: Sendable, Equatable {
    case text(String)
    case imageURL(url: String, detail: String?)
    case unsupported(type: String)

    var text: String? {
        guard case .text(let value) = self else { return nil }
        return value
    }

    var typeName: String {
        switch self {
        case .text: return "text"
        case .imageURL: return "image_url"
        case .unsupported(let type): return type
        }
    }
}

struct OpenAIChatMessage: Codable, Sendable {
    let role: OpenAIRole
    let content: String?
    /// Lossless ordered request content. String content becomes one text part;
    /// array content retains media placement and unsupported part types so the
    /// selected CAM executor can handle or reject them explicitly.
    let contentParts: [OpenAIChatContentPart]?
    let name: String?
    let toolCallId: String?
    let toolCalls: [OpenAIToolCall]?

    init(
        role: OpenAIRole,
        content: String?,
        name: String? = nil,
        toolCallId: String? = nil,
        toolCalls: [OpenAIToolCall]? = nil
    ) {
        self.role = role
        self.content = content
        self.contentParts = content.map { [.text($0)] }
        self.name = name
        self.toolCallId = toolCallId
        self.toolCalls = toolCalls
    }

    private enum CodingKeys: String, CodingKey {
        case role, content, name
        case toolCallId = "tool_call_id"
        case toolCalls = "tool_calls"
    }

    private struct ContentPart: Decodable {
        let type: String
        let text: String?
        let imageURL: ImageURL?

        private enum CodingKeys: String, CodingKey {
            case type, text
            case imageURL = "image_url"
        }

        struct ImageURL: Decodable {
            let url: String
            let detail: String?

            init(from decoder: Decoder) throws {
                let value = try decoder.singleValueContainer()
                if let url = try? value.decode(String.self) {
                    self.url = url
                    self.detail = nil
                    return
                }
                struct Object: Decodable {
                    let url: String
                    let detail: String?
                }
                let object = try value.decode(Object.self)
                self.url = object.url
                self.detail = object.detail
            }
        }
    }

    // OpenAI's chat completions spec allows `content` to be either
    // a plain string or an array of typed content parts (multimodal:
    // {"type":"text","text":"..."}, {"type":"image_url",...}). Keep
    // the array losslessly and expose its text projection separately: prompt
    // rendering still consumes `content`, while CAM media execution consumes
    // `contentParts`. Nothing is silently erased at the protocol boundary.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.role = try c.decode(OpenAIRole.self, forKey: .role)
        self.name = try c.decodeIfPresent(String.self, forKey: .name)
        self.toolCallId = try c.decodeIfPresent(String.self, forKey: .toolCallId)
        self.toolCalls = try c.decodeIfPresent([OpenAIToolCall].self, forKey: .toolCalls)
        // content is optional in two senses: missing key (assistant
        // history with only tool_calls) AND explicit null. decodeNil
        // throws keyNotFound when the key is absent, so guard with
        // contains first.
        if !c.contains(.content) {
            self.content = nil
            self.contentParts = nil
        } else if (try? c.decodeNil(forKey: .content)) == true {
            self.content = nil
            self.contentParts = nil
        } else if let s = try? c.decode(String.self, forKey: .content) {
            self.content = s
            self.contentParts = [.text(s)]
        } else if let parts = try? c.decode([ContentPart].self, forKey: .content) {
            let retained = parts.map { part -> OpenAIChatContentPart in
                switch part.type {
                case "text": return .text(part.text ?? "")
                case "image_url":
                    guard let image = part.imageURL else {
                        return .unsupported(type: "image_url")
                    }
                    return .imageURL(url: image.url, detail: image.detail)
                default: return .unsupported(type: part.type)
                }
            }
            self.contentParts = retained
            self.content = retained.compactMap(\.text).joined()
        } else {
            self.content = nil
            self.contentParts = nil
        }
    }

    // OpenAI's wire shape always includes `content` on assistant
    // responses: null when tool_calls is present, string otherwise.
    // The synthesized encoder uses encodeIfPresent which would omit
    // the field — some strict client SDKs reject that.
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(role, forKey: .role)
        if let content {
            try c.encode(content, forKey: .content)
        } else {
            try c.encodeNil(forKey: .content)
        }
        try c.encodeIfPresent(name, forKey: .name)
        try c.encodeIfPresent(toolCallId, forKey: .toolCallId)
        try c.encodeIfPresent(toolCalls, forKey: .toolCalls)
    }
}

struct OpenAIFunctionDef: Decodable, Sendable {
    let name: String
    let description: String?
    let parameters: SmeltJSONValue?
}

struct OpenAIChatTool: Decodable, Sendable {
    let type: OpenAIToolType
    let function: OpenAIFunctionDef
}

/// `.disabled` corresponds to OpenAI's wire value `"none"`; the case
/// is renamed to avoid colliding with `Optional<OpenAIToolChoice>.none`
/// at use sites (`request.toolChoice == .none` would silently match
/// both the explicit-none case and the field being absent).
enum OpenAIToolChoice: Decodable, Sendable, Equatable {
    case auto
    case disabled
    case required
    case specific(String)

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            switch s {
            case "auto": self = .auto
            case "none": self = .disabled
            case "required": self = .required
            default:
                throw DecodingError.dataCorruptedError(
                    in: container,
                    debugDescription: "tool_choice string must be \"auto\", \"none\", or \"required\""
                )
            }
            return
        }
        struct Specific: Decodable {
            let type: String
            let function: NameOnly
            struct NameOnly: Decodable { let name: String }
        }
        let spec = try container.decode(Specific.self)
        self = .specific(spec.function.name)
    }
}

enum OpenAIStopValue: Codable, Sendable {
    case none
    case one(String)
    case many([String])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .none
        } else if let single = try? container.decode(String.self) {
            self = .one(single)
        } else {
            self = .many(try container.decode([String].self))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .none: try container.encodeNil()
        case .one(let s): try container.encode(s)
        case .many(let arr): try container.encode(arr)
        }
    }

    var sequences: [String] {
        switch self {
        case .none: return []
        case .one(let s): return [s]
        case .many(let arr): return arr
        }
    }
}

// Streaming chunk shapes for /v1/chat/completions when stream=true.
// SSE wire format: each event is `data: <json>\n\n`; the stream
// terminates with `data: [DONE]\n\n`.

struct OpenAIChatStreamFunctionDelta: Encodable, Sendable {
    let name: String?
    let arguments: String?
}

struct OpenAIChatStreamToolCallDelta: Encodable, Sendable {
    let index: Int
    let id: String?
    let type: OpenAIToolType?
    let function: OpenAIChatStreamFunctionDelta?
}

struct OpenAIChatStreamDelta: Encodable, Sendable {
    let role: OpenAIRole?
    let content: String?
    let toolCalls: [OpenAIChatStreamToolCallDelta]?

    init(
        role: OpenAIRole? = nil,
        content: String? = nil,
        toolCalls: [OpenAIChatStreamToolCallDelta]? = nil
    ) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
    }

    private enum CodingKeys: String, CodingKey {
        case role, content
        case toolCalls = "tool_calls"
    }
}

struct OpenAIChatStreamChoice: Encodable, Sendable {
    let index: Int
    let delta: OpenAIChatStreamDelta
    let logprobs: OpenAILogprobs?
    let finishReason: OpenAIFinishReason?
    private enum CodingKeys: String, CodingKey {
        case index, delta, logprobs
        case finishReason = "finish_reason"
    }
}

struct OpenAIChatStreamChunk: Encodable, Sendable {
    let id: String
    let object: String   // "chat.completion.chunk"
    let created: Int
    let model: String
    let choices: [OpenAIChatStreamChoice]
    /// Populated only on the trailing chunk when the request opted in
    /// via `stream_options.include_usage: true`. The trailing chunk
    /// also carries `choices: []` per the OpenAI spec.
    let usage: OpenAIUsage?

    init(
        id: String, object: String, created: Int, model: String,
        choices: [OpenAIChatStreamChoice], usage: OpenAIUsage? = nil
    ) {
        self.id = id; self.object = object; self.created = created
        self.model = model; self.choices = choices; self.usage = usage
    }
}

/// OpenAI's `stream_options` request field. We only honor
/// `include_usage` today; other fields (e.g. `include_obfuscation`) are
/// silently ignored, matching upstream lenient behavior.
struct OpenAIStreamOptions: Decodable, Sendable {
    let includeUsage: Bool?

    private enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
    }
}

struct OpenAIChatCompletionsRequest: Decodable, Sendable {
    let model: String
    let messages: [OpenAIChatMessage]
    let effectiveMaxTokens: Int?
    let temperature: Double?
    let topP: Double?
    /// Common OpenAI-compatible extension used by local runtimes.
    let topK: Int?
    let stop: OpenAIStopValue?
    let seed: Int?
    let logprobs: Bool?
    let topLogprobs: Int?
    let n: Int?
    let stream: Bool?
    let streamOptions: OpenAIStreamOptions?
    let tools: [OpenAIChatTool]?
    let toolChoice: OpenAIToolChoice?
    /// Smelt extension: resume a previously-allocated server session.
    ///
    /// **Contract**: session_id is a *cache-affinity hint*, not a
    /// substitute for the message history. Each request MUST still
    /// contain the full conversation in `messages`. The server uses
    /// the id to identify the in-progress conversation for prefix-
    /// cache locality and to return 404 `session_not_found` when the
    /// id is unknown (so clients can retry with `create_session:
    /// true`). It does NOT cause the server to "remember" prior
    /// messages on its own.
    let sessionId: String?
    /// Smelt extension: allocate a new server session for this turn.
    /// Mutually exclusive with `sessionId`: if both are set, sessionId
    /// wins and create_session is ignored.
    let createSession: Bool?
    /// Adapter extension selecting a package-authored prepared prompt state.
    /// The server still requires an exact token-prefix match before restore.
    let promptContract: String?

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: WireKeys.self)
        self.model = try c.decode(String.self, forKey: .model)
        self.messages = try c.decode([OpenAIChatMessage].self, forKey: .messages)
        // OpenAI ships both names; max_completion_tokens is the newer one
        // and wins when both are present.
        let mct = try c.decodeIfPresent(Int.self, forKey: .maxCompletionTokens)
        let mt  = try c.decodeIfPresent(Int.self, forKey: .maxTokens)
        self.effectiveMaxTokens = mct ?? mt
        self.temperature = try c.decodeIfPresent(Double.self, forKey: .temperature)
        self.topP = try c.decodeIfPresent(Double.self, forKey: .topP)
        self.topK = try c.decodeIfPresent(Int.self, forKey: .topK)
        self.stop = try c.decodeIfPresent(OpenAIStopValue.self, forKey: .stop)
        self.seed = try c.decodeIfPresent(Int.self, forKey: .seed)
        self.logprobs = try c.decodeIfPresent(Bool.self, forKey: .logprobs)
        self.topLogprobs = try c.decodeIfPresent(Int.self, forKey: .topLogprobs)
        self.n = try c.decodeIfPresent(Int.self, forKey: .n)
        self.stream = try c.decodeIfPresent(Bool.self, forKey: .stream)
        self.streamOptions = try c.decodeIfPresent(OpenAIStreamOptions.self, forKey: .streamOptions)
        self.tools = try c.decodeIfPresent([OpenAIChatTool].self, forKey: .tools)
        self.toolChoice = try c.decodeIfPresent(OpenAIToolChoice.self, forKey: .toolChoice)
        self.sessionId = try c.decodeIfPresent(String.self, forKey: .sessionId)
        self.createSession = try c.decodeIfPresent(Bool.self, forKey: .createSession)
        self.promptContract = try c.decodeIfPresent(String.self, forKey: .promptContract)
    }

    private enum WireKeys: String, CodingKey {
        case model, messages
        case maxTokens           = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case temperature
        case topP                = "top_p"
        case topK                = "top_k"
        case stop, seed, logprobs
        case topLogprobs         = "top_logprobs"
        case stream
        case streamOptions       = "stream_options"
        case n
        case tools
        case toolChoice          = "tool_choice"
        case sessionId           = "session_id"
        case createSession       = "create_session"
        case promptContract      = "prompt_contract"
    }
}

// OpenAI's legacy /v1/completions accepts `prompt` as String OR an
// array of token-ids. lm-eval-harness's local-completions client
// pre-tokenizes and sends [Int]. The string form is what humans
// (and HumanEval-style benches) use.
enum OpenAICompletionsPrompt: Sendable {
    case text(String)
    case tokens([Int32])
}

struct OpenAICompletionsRequest: Decodable, Sendable {
    let model: String
    let prompt: OpenAICompletionsPrompt
    let effectiveMaxTokens: Int?
    let temperature: Double?
    let topP: Double?
    let topK: Int?
    let stop: OpenAIStopValue?
    let seed: Int?
    let logprobs: Int?
    let echo: Bool?
    /// Number of completions to generate. OpenAI default is 1; we
    /// cap at 16 to bound per-request compute. nil ⇒ treat as 1.
    let n: Int?
    let stream: Bool?

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: WireKeys.self)
        self.model = try c.decode(String.self, forKey: .model)
        // OpenAI accepts String | [Int] | [[Int]] | [String]; lm-eval
        // sends [[Int]] (one batch with one tokenized prompt). Try the
        // shapes in order of likelihood.
        if let s = try? c.decode(String.self, forKey: .prompt) {
            self.prompt = .text(s)
        } else if let nested = try? c.decode([[Int]].self, forKey: .prompt),
                  let first = nested.first {
            self.prompt = .tokens(first.map { Int32($0) })
        } else if let ids = try? c.decode([Int].self, forKey: .prompt) {
            self.prompt = .tokens(ids.map { Int32($0) })
        } else if let arr = try? c.decode([String].self, forKey: .prompt),
                  let first = arr.first {
            self.prompt = .text(first)
        } else {
            self.prompt = .text(
                try c.decode(String.self, forKey: .prompt)
            )
        }
        let mct = try c.decodeIfPresent(Int.self, forKey: .maxCompletionTokens)
        let mt  = try c.decodeIfPresent(Int.self, forKey: .maxTokens)
        self.effectiveMaxTokens = mct ?? mt
        self.temperature = try c.decodeIfPresent(Double.self, forKey: .temperature)
        self.topP = try c.decodeIfPresent(Double.self, forKey: .topP)
        self.topK = try c.decodeIfPresent(Int.self, forKey: .topK)
        self.stop = try c.decodeIfPresent(OpenAIStopValue.self, forKey: .stop)
        self.seed = try c.decodeIfPresent(Int.self, forKey: .seed)
        self.logprobs = try c.decodeIfPresent(Int.self, forKey: .logprobs)
        self.echo = try c.decodeIfPresent(Bool.self, forKey: .echo)
        self.n = try c.decodeIfPresent(Int.self, forKey: .n)
        self.stream = try c.decodeIfPresent(Bool.self, forKey: .stream)
    }

    private enum WireKeys: String, CodingKey {
        case model, prompt
        case maxTokens           = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case temperature
        case topP                = "top_p"
        case topK                = "top_k"
        case stop, seed, logprobs, echo, n, stream
    }
}

// Streaming chunk for legacy /v1/completions. Wire shape:
//   {choices: [{index, text: "...", finish_reason: nil}]}
// One chunk per generated token (more or less — UTF-8 boundary
// buffering may delay a chunk to the next token).

struct OpenAICompletionStreamChoice: Encodable, Sendable {
    let index: Int
    let text: String
    let finishReason: OpenAIFinishReason?
    private enum CodingKeys: String, CodingKey {
        case index, text
        case finishReason = "finish_reason"
    }
}

struct OpenAICompletionStreamChunk: Encodable, Sendable {
    let id: String
    let object: String   // "text_completion"
    let created: Int
    let model: String
    let choices: [OpenAICompletionStreamChoice]
}

struct OpenAIUsage: Encodable, Sendable {
    let promptTokens: Int
    let completionTokens: Int
    let totalTokens: Int

    private enum CodingKeys: String, CodingKey {
        case promptTokens     = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens      = "total_tokens"
    }
}

struct OpenAITopLogprob: Encodable, Sendable {
    let token: String
    let logprob: Double
    let bytes: [Int]?
}

struct OpenAITokenLogprob: Encodable, Sendable {
    let token: String
    let logprob: Double
    let bytes: [Int]?
    let topLogprobs: [OpenAITopLogprob]

    private enum CodingKeys: String, CodingKey {
        case token, logprob, bytes
        case topLogprobs = "top_logprobs"
    }
}

struct OpenAILogprobs: Encodable, Sendable {
    let content: [OpenAITokenLogprob]
}

struct OpenAICompletionLogprobs: Encodable, Sendable {
    let tokens: [String]
    let tokenLogprobs: [Double?]
    let topLogprobs: [[String: Double]]
    let textOffset: [Int]

    private enum CodingKeys: String, CodingKey {
        case tokens
        case tokenLogprobs = "token_logprobs"
        case topLogprobs   = "top_logprobs"
        case textOffset    = "text_offset"
    }
}

struct OpenAIChoice: Encodable, Sendable {
    let index: Int
    let message: OpenAIChatMessage
    let finishReason: OpenAIFinishReason
    let logprobs: OpenAILogprobs?

    private enum CodingKeys: String, CodingKey {
        case index, message
        case finishReason = "finish_reason"
        case logprobs
    }
}

struct OpenAICompletionChoice: Encodable, Sendable {
    let index: Int
    let text: String
    let finishReason: OpenAIFinishReason
    let logprobs: OpenAICompletionLogprobs?

    private enum CodingKeys: String, CodingKey {
        case index, text
        case finishReason = "finish_reason"
        case logprobs
    }
}

struct OpenAIChatCompletionsResponse: Encodable, Sendable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [OpenAIChoice]
    let usage: OpenAIUsage
}

struct OpenAICompletionsResponse: Encodable, Sendable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [OpenAICompletionChoice]
    let usage: OpenAIUsage
}

package struct OpenAIModelEntry: Encodable, Sendable {
    let id: String
    let object: String
    let created: Int
    let ownedBy: String

    package init(id: String, object: String, created: Int, ownedBy: String) {
        self.id = id
        self.object = object
        self.created = created
        self.ownedBy = ownedBy
    }

    private enum CodingKeys: String, CodingKey {
        case id, object, created
        case ownedBy = "owned_by"
    }
}

package struct OpenAIModelsResponse: Encodable, Sendable {
    let object: String
    let data: [OpenAIModelEntry]

    package init(object: String, data: [OpenAIModelEntry]) {
        self.object = object
        self.data = data
    }
}

package enum OpenAIErrorCode: String, Sendable {
    case invalidRequest      = "invalid_request_error"
    case contextLengthExceeded = "context_length_exceeded"
    case internalError       = "internal_error"
    case notImplemented      = "not_implemented"
    case methodNotAllowed    = "method_not_allowed"
    case notFound            = "not_found"
    /// Smelt session_id extension: client sent a session_id the
    /// server has no record of (evicted, restarted, or never
    /// allocated). Client should clear its persisted id and retry
    /// with `create_session: true`.
    case sessionNotFound     = "session_not_found"
}

struct OpenAIErrorBody: Encodable, Sendable {
    let message: String
    let type: String
    let code: String?
    /// Smelt extension: when a tool-call grammar accepted but the
    /// resulting JSON failed to decode (or generation hit max_tokens
    /// mid-call), surface the partial text + the token IDs that
    /// produced it so callers can replay or recover.
    let partialJson: String?
    let generatedTokenIds: [Int32]?
    let toolCallFailure: Bool?

    init(
        message: String,
        type: String,
        code: String?,
        partialJson: String? = nil,
        generatedTokenIds: [Int32]? = nil,
        toolCallFailure: Bool? = nil
    ) {
        self.message = message
        self.type = type
        self.code = code
        self.partialJson = partialJson
        self.generatedTokenIds = generatedTokenIds
        self.toolCallFailure = toolCallFailure
    }

    private enum CodingKeys: String, CodingKey {
        case message, type, code
        case partialJson       = "partial_json"
        case generatedTokenIds = "generated_token_ids"
        case toolCallFailure   = "tool_call_failure"
    }
}

struct OpenAIErrorEnvelope: Encodable, Sendable {
    let error: OpenAIErrorBody
}

package enum OpenAIJSON {
    static let decoder = JSONDecoder()
    static let encoder = JSONEncoder()

    package static func encode<T: Encodable>(_ value: T) throws -> Data {
        try encoder.encode(value)
    }

    static func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        try decoder.decode(type, from: data)
    }

    /// Encode `payload` as a single Server-Sent Events frame:
    /// `data: <json>\n\n`. Used by both streaming chat-completions
    /// and streaming legacy-completions endpoints.
    static func sseFrame<T: Encodable>(_ payload: T) throws -> Data {
        var frame = Data("data: ".utf8)
        frame.append(try encoder.encode(payload))
        frame.append(Data("\n\n".utf8))
        return frame
    }

    /// Static terminating frame for SSE responses, per OpenAI's
    /// streaming contract.
    static let sseDoneFrame = Data("data: [DONE]\n\n".utf8)

    static func chatCompletionId() -> String {
        "chatcmpl-\(UUID().uuidString.lowercased())"
    }

    static func completionId() -> String {
        "cmpl-\(UUID().uuidString.lowercased())"
    }

    /// Encoding a fixed-shape OpenAIErrorEnvelope of String/String/String?
    /// is infallible — try! catches any future regression that violates
    /// that assumption.
    static func errorEnvelope(
        message: String,
        code: OpenAIErrorCode
    ) -> Data {
        let env = OpenAIErrorEnvelope(error: OpenAIErrorBody(
            message: message,
            type: code.rawValue,
            code: code.rawValue
        ))
        return try! encoder.encode(env)
    }

    package static func errorResponse(
        status: Int,
        code: OpenAIErrorCode,
        message: String
    ) -> SmeltServeRawResponse {
        SmeltServeRawResponse(
            statusCode: status,
            body: errorEnvelope(message: message, code: code)
        )
    }

    static func toolCallFailureEnvelope(
        message: String,
        partialJson: String,
        generatedTokenIds: [Int32]
    ) -> Data {
        let env = OpenAIErrorEnvelope(error: OpenAIErrorBody(
            message: message,
            type: OpenAIErrorCode.internalError.rawValue,
            code: OpenAIErrorCode.internalError.rawValue,
            partialJson: partialJson,
            generatedTokenIds: generatedTokenIds,
            toolCallFailure: true
        ))
        return try! encoder.encode(env)
    }

    static func toolCallFailureResponse(
        message: String,
        partialJson: String,
        generatedTokenIds: [Int32]
    ) -> SmeltServeRawResponse {
        SmeltServeRawResponse(
            statusCode: 500,
            body: toolCallFailureEnvelope(
                message: message,
                partialJson: partialJson,
                generatedTokenIds: generatedTokenIds
            )
        )
    }
}
