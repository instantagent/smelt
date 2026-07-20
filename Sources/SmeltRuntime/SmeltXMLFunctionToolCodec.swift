import Foundation

/// Neutral decoded form for the package-owned XML function/parameter tool
/// transcript. SmeltTextRuntime does not depend on OpenAI wire types; adapters map
/// this value to their own tool-call representation.
public struct SmeltDecodedXMLToolCall: Equatable, Sendable {
    public let name: String
    public let argumentsJSON: String

    public init(name: String, argumentsJSON: String) {
        self.name = name
        self.argumentsJSON = argumentsJSON
    }
}

public struct SmeltDecodedXMLToolResponse: Equatable, Sendable {
    public let leadingText: String?
    public let calls: [SmeltDecodedXMLToolCall]

    public init(leadingText: String?, calls: [SmeltDecodedXMLToolCall]) {
        self.leadingText = leadingText
        self.calls = calls
    }
}

/// The earliest semantically stable event in the native XML tool stream. A
/// function name is safe to expose once its closing `>` has arrived; arguments
/// remain buffered until the complete call can be schema-validated and
/// canonicalized by ``SmeltXMLFunctionToolCodec/decode(_:tools:)``.
public struct SmeltXMLFunctionToolStreamCallStart: Equatable, Sendable {
    public let index: Int
    public let name: String
    public let leadingText: String?

    public init(index: Int, name: String, leadingText: String? = nil) {
        self.index = index
        self.name = name
        self.leadingText = leadingText
    }
}

/// Incremental recognizer for package-native XML tool calls. It deliberately
/// emits only function starts: partial native parameter text is not necessarily
/// valid JSON (and may represent a number, boolean, object, or string depending
/// on the active schema), so argument bytes are emitted only after the complete
/// response passes the canonical decoder.
public final class SmeltXMLFunctionToolStreamDecoder {
    private let allowedToolNames: Set<String>
    private var accumulated = ""
    private var emittedCount = 0

    public init(toolNames: [String]) {
        self.allowedToolNames = Set(toolNames)
    }

    public func consume(_ chunk: String) throws -> [SmeltXMLFunctionToolStreamCallStart] {
        accumulated += chunk
        let discovered = try Self.discoverCallStarts(
            in: accumulated,
            allowedToolNames: allowedToolNames
        )
        guard discovered.count > emittedCount else { return [] }
        let new = Array(discovered[emittedCount...])
        emittedCount = discovered.count
        return new
    }

    private static func discoverCallStarts(
        in text: String,
        allowedToolNames: Set<String>
    ) throws -> [SmeltXMLFunctionToolStreamCallStart] {
        var starts: [SmeltXMLFunctionToolStreamCallStart] = []
        var cursor = text.startIndex
        while let call = text.range(
            of: "<tool_call>", range: cursor..<text.endIndex
        ) {
            guard let function = text.range(
                of: "<function=", range: call.upperBound..<text.endIndex
            ) else { break }
            if let nextCall = text.range(
                of: "<tool_call>", range: call.upperBound..<text.endIndex
            ), nextCall.lowerBound < function.lowerBound {
                cursor = nextCall.lowerBound
                continue
            }
            guard let nameEnd = text[function.upperBound...].firstIndex(of: ">") else {
                break
            }
            let name = String(text[function.upperBound..<nameEnd])
            guard allowedToolNames.contains(name) else {
                throw SmeltXMLFunctionToolCodecError.unknownTool(name)
            }
            let leading: String?
            if starts.isEmpty {
                let prefix = text[..<call.lowerBound]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                leading = prefix.isEmpty ? nil : prefix
            } else {
                leading = nil
            }
            starts.append(SmeltXMLFunctionToolStreamCallStart(
                index: starts.count,
                name: name,
                leadingText: leading
            ))
            cursor = text.index(after: nameEnd)
        }
        return starts
    }
}

public enum SmeltXMLFunctionToolCodecError: Error, CustomStringConvertible {
    case noToolCall
    case malformed(String)
    case unknownTool(String)
    case unknownParameter(tool: String, parameter: String)
    case duplicateParameter(tool: String, parameter: String)
    case missingParameter(tool: String, parameter: String)
    case invalidParameter(tool: String, parameter: String, expected: String)
    case invalidSchema(tool: String)
    case invalidArguments(tool: String)

    public var description: String {
        switch self {
        case .noToolCall:
            return "no <tool_call> block found"
        case .malformed(let detail):
            return "malformed XML tool call: \(detail)"
        case .unknownTool(let name):
            return "tool \"\(name)\" is not in the active descriptor set"
        case .unknownParameter(let tool, let parameter):
            return "tool \"\(tool)\" has no parameter \"\(parameter)\""
        case .duplicateParameter(let tool, let parameter):
            return "tool \"\(tool)\" repeats parameter \"\(parameter)\""
        case .missingParameter(let tool, let parameter):
            return "tool \"\(tool)\" is missing required parameter \"\(parameter)\""
        case .invalidParameter(let tool, let parameter, let expected):
            return "tool \"\(tool)\" parameter \"\(parameter)\" is not \(expected)"
        case .invalidSchema(let tool):
            return "tool \"\(tool)\" has an invalid object schema"
        case .invalidArguments(let tool):
            return "tool \"\(tool)\" arguments are not a JSON object"
        }
    }
}

/// Codec and constrained-decoding grammar for the semantic transcript shape:
///
/// ```
/// <tool_call>
/// <function=name>
/// <parameter=argument>
/// value
/// </parameter>
/// </function>
/// </tool_call>
/// ```
///
/// Selection is by package prompt-format capability. Nothing in this type
/// depends on a repository or checkpoint identity.
public enum SmeltXMLFunctionToolCodec {
    /// Special-token literals that are semantic payload for this codec rather
    /// than hidden generation control. API adapters re-expose only these IDs to
    /// the XML parser; EOS, turn, and thinking tokens remain suppressed.
    public static let visibleGeneratedControlTokens = [
        "<tool_call>", "</tool_call>",
    ]

    /// Render the package-native tools system block. This intentionally mirrors
    /// the pinned Jinja transcript byte-for-byte for string-only chat content;
    /// keeping it beside the decoder and grammar makes the native protocol one
    /// reusable brick rather than three loosely synchronized CLI conventions.
    public static func renderSystemMessage(
        tools: [SmeltToolDescriptor]
    ) throws -> String {
        var result = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
        for tool in tools {
            if let promptJSON = tool.promptJSON {
                guard let descriptor = try? SmeltOrderedJSONValue.parse(
                    promptJSON
                ), case .object = descriptor else {
                    throw SmeltXMLFunctionToolCodecError.invalidSchema(
                        tool: tool.name
                    )
                }
                result += "\n" + descriptor.templateJSON
                continue
            }
            guard let parameters = try? SmeltOrderedJSONValue.parse(
                tool.schemaJSON
            ), case .object = parameters else {
                throw SmeltXMLFunctionToolCodecError.invalidSchema(tool: tool.name)
            }
            var functionFields = [
                "\"name\": " + templateJSONString(tool.name),
            ]
            if let description = tool.description, !description.isEmpty {
                functionFields.append(
                    "\"description\": " + templateJSONString(description)
                )
            }
            functionFields.append(
                "\"parameters\": " + parameters.templateJSON
            )
            result += "\n{\"type\": \"function\", \"function\": {"
                + functionFields.joined(separator: ", ") + "}}"
        }
        result += "\n</tools>"
        result += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:"
        result += "\n\n<tool_call>\n<function=example_function_name>"
        result += "\n<parameter=example_parameter_1>\nvalue_1\n</parameter>"
        result += "\n<parameter=example_parameter_2>\nThis is the value for the second parameter"
        result += "\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>"
        result += "\n\n<IMPORTANT>\nReminder:"
        result += "\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags"
        result += "\n- Required parameters MUST be specified"
        result += "\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after"
        result += "\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls"
        result += "\n</IMPORTANT>"
        return result
    }

    /// Render prior assistant calls into the same XML protocol accepted by
    /// `decode`. Argument keys are sorted deliberately: OpenAI arguments arrive
    /// as a JSON string whose object-member order is not semantic, while stable
    /// rendering is essential to exact prefix-cache identity.
    public static func renderCalls(
        _ calls: [SmeltDecodedXMLToolCall]
    ) throws -> String {
        try calls.map { call in
            guard let value = try? JSONDecoder().decode(
                SmeltJSONValue.self,
                from: Data((call.argumentsJSON.isEmpty ? "{}" : call.argumentsJSON).utf8)
            ), case .object(let arguments) = value else {
                throw SmeltXMLFunctionToolCodecError.invalidArguments(
                    tool: call.name
                )
            }
            var result = "<tool_call>\n<function=\(call.name)>\n"
            for name in arguments.keys.sorted() {
                let argument = arguments[name]!
                let rendered: String
                if case .string(let string) = argument {
                    rendered = string
                } else {
                    rendered = templateJSON(argument)
                }
                result += "<parameter=\(name)>\n\(rendered)\n</parameter>\n"
            }
            result += "</function>\n</tool_call>"
            return result
        }.joined(separator: "\n")
    }

    public static func decode(
        _ text: String,
        tools: [SmeltToolDescriptor]
    ) throws -> SmeltDecodedXMLToolResponse {
        guard let firstCall = text.range(of: "<tool_call>") else {
            throw SmeltXMLFunctionToolCodecError.noToolCall
        }
        let prefix = String(text[..<firstCall.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let byName = Dictionary(uniqueKeysWithValues: tools.map { ($0.name, $0) })
        var calls: [SmeltDecodedXMLToolCall] = []
        var cursor = firstCall.lowerBound

        while cursor < text.endIndex {
            let remainder = text[cursor...]
            if remainder.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                break
            }
            guard let callStart = text.range(
                of: "<tool_call>", range: cursor..<text.endIndex
            ) else {
                throw SmeltXMLFunctionToolCodecError.malformed(
                    "non-whitespace suffix after final call"
                )
            }
            let between = text[cursor..<callStart.lowerBound]
            guard between.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw SmeltXMLFunctionToolCodecError.malformed(
                    "text is only allowed before the first call"
                )
            }
            guard let callEnd = text.range(
                of: "</tool_call>",
                range: callStart.upperBound..<text.endIndex
            ) else {
                throw SmeltXMLFunctionToolCodecError.malformed(
                    "missing </tool_call>"
                )
            }
            let bodyRange = callStart.upperBound..<callEnd.lowerBound
            calls.append(try decodeCall(
                String(text[bodyRange]), toolsByName: byName
            ))
            cursor = callEnd.upperBound
        }

        guard !calls.isEmpty else {
            throw SmeltXMLFunctionToolCodecError.noToolCall
        }
        return SmeltDecodedXMLToolResponse(
            leadingText: prefix.isEmpty ? nil : prefix,
            calls: calls
        )
    }

    /// Lark grammar for native constrained generation. It constrains envelope,
    /// active function names, and declared parameter names. Schema value types
    /// are validated by `decode` after acceptance because the native wire form
    /// intentionally represents string values without JSON quotes.
    public static func larkGrammar(
        for tools: [SmeltToolDescriptor],
        allowText: Bool
    ) throws -> String {
        guard !tools.isEmpty else { throw SmeltToolGrammarError.noTools }
        let toolNames = try tools.map { try larkLiteral($0.name) }
        var parameters = Set<String>()
        for tool in tools {
            let schema = try schemaObject(tool)
            if let properties = schema["properties"] as? [String: Any] {
                parameters.formUnion(properties.keys)
            }
        }
        let parameterNames = try parameters.sorted().map(larkLiteral)
        let parameterRule: String
        if parameterNames.isEmpty {
            parameterRule = "parameter_name: \"__agent_no_parameters__\""
                + "\nparameter: \"<parameter=\" parameter_name \">\\n\" parameter_text \"</parameter>\\n\""
        } else {
            parameterRule = "parameter_name: "
                + parameterNames.joined(separator: " | ")
                + "\nparameter: \"<parameter=\" parameter_name \">\\n\" parameter_text \"</parameter>\\n\""
        }
        let startRule = allowText
            ? "start: tool_call | text_response"
            : "start: tool_call"
        let textRules = allowText ? """
        text_response: leading_ws first_char text_tail
        leading_ws: /\\s*/
        first_char: /[^<\\s]/
        text_tail: /[^<]*/
        """ : ""
        return """
        \(startRule)
        tool_call: "<tool_call>\\n<function=" function_name ">\\n" parameter* "</function>\\n</tool_call>"
        function_name: \(toolNames.joined(separator: " | "))
        \(parameterRule)
        // Include the protocol's required trailing newline in the value token.
        // Keeping it out of the closing-tag literal avoids an ambiguous lexer
        // split where a greedy value consumes the newline and permanently
        // masks the following `<` token.
        parameter_text: /[^<]*\\n/
        \(textRules)
        """
    }

    private static func decodeCall(
        _ body: String,
        toolsByName: [String: SmeltToolDescriptor]
    ) throws -> SmeltDecodedXMLToolCall {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("<function=") else {
            throw SmeltXMLFunctionToolCodecError.malformed(
                "call does not begin with <function=>"
            )
        }
        guard let nameEnd = trimmed.firstIndex(of: ">") else {
            throw SmeltXMLFunctionToolCodecError.malformed(
                "function opener is not closed"
            )
        }
        let nameStart = trimmed.index(
            trimmed.startIndex, offsetBy: "<function=".count
        )
        let name = String(trimmed[nameStart..<nameEnd])
        guard let tool = toolsByName[name] else {
            throw SmeltXMLFunctionToolCodecError.unknownTool(name)
        }
        guard let functionClose = trimmed.range(of: "</function>") else {
            throw SmeltXMLFunctionToolCodecError.malformed(
                "missing </function>"
            )
        }
        let afterClose = trimmed[functionClose.upperBound...]
        guard afterClose.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw SmeltXMLFunctionToolCodecError.malformed(
                "content follows </function>"
            )
        }
        var parameterText = String(trimmed[trimmed.index(after: nameEnd)..<functionClose.lowerBound])
        var rawArguments: [String: String] = [:]
        while !parameterText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            parameterText = parameterText.trimmingCharacters(in: .whitespacesAndNewlines)
            guard parameterText.hasPrefix("<parameter=") else {
                throw SmeltXMLFunctionToolCodecError.malformed(
                    "expected <parameter=>"
                )
            }
            guard let openerEnd = parameterText.firstIndex(of: ">") else {
                throw SmeltXMLFunctionToolCodecError.malformed(
                    "parameter opener is not closed"
                )
            }
            let parameterStart = parameterText.index(
                parameterText.startIndex, offsetBy: "<parameter=".count
            )
            let parameter = String(parameterText[parameterStart..<openerEnd])
            guard rawArguments[parameter] == nil else {
                throw SmeltXMLFunctionToolCodecError.duplicateParameter(
                    tool: name, parameter: parameter
                )
            }
            let valueStart = parameterText.index(after: openerEnd)
            guard let close = parameterText.range(
                of: "</parameter>", range: valueStart..<parameterText.endIndex
            ) else {
                throw SmeltXMLFunctionToolCodecError.malformed(
                    "missing </parameter> for \(parameter)"
                )
            }
            var raw = String(parameterText[valueStart..<close.lowerBound])
            if raw.hasPrefix("\n") { raw.removeFirst() }
            if raw.hasSuffix("\n") { raw.removeLast() }
            rawArguments[parameter] = raw
            parameterText = String(parameterText[close.upperBound...])
        }

        let schema = try schemaObject(tool)
        let properties = schema["properties"] as? [String: Any] ?? [:]
        let required = Set(schema["required"] as? [String] ?? [])
        for parameter in rawArguments.keys where properties[parameter] == nil {
            throw SmeltXMLFunctionToolCodecError.unknownParameter(
                tool: name, parameter: parameter
            )
        }
        for parameter in required where rawArguments[parameter] == nil {
            throw SmeltXMLFunctionToolCodecError.missingParameter(
                tool: name, parameter: parameter
            )
        }

        var arguments: [String: SmeltJSONValue] = [:]
        for (parameter, raw) in rawArguments {
            let property = properties[parameter] as? [String: Any] ?? [:]
            arguments[parameter] = try decodedValue(
                raw,
                schema: property,
                tool: name,
                parameter: parameter
            )
        }
        return SmeltDecodedXMLToolCall(
            name: name,
            argumentsJSON: try SmeltJSON.canonicalString(arguments)
        )
    }

    private static func decodedValue(
        _ raw: String,
        schema: [String: Any],
        tool: String,
        parameter: String
    ) throws -> SmeltJSONValue {
        let types: [String]
        if let type = schema["type"] as? String {
            types = [type]
        } else if let listed = schema["type"] as? [String] {
            types = listed
        } else {
            types = []
        }
        if types == ["string"] { return .string(raw) }
        if let decoded = try? JSONDecoder().decode(
            SmeltJSONValue.self, from: Data(raw.utf8)
        ) {
            if types.isEmpty || value(decoded, matchesAny: types) {
                return decoded
            }
        }
        if types.contains("string") { return .string(raw) }
        throw SmeltXMLFunctionToolCodecError.invalidParameter(
            tool: tool,
            parameter: parameter,
            expected: types.isEmpty ? "valid JSON" : types.joined(separator: " or ")
        )
    }

    private static func value(
        _ value: SmeltJSONValue,
        matchesAny types: [String]
    ) -> Bool {
        types.contains { type in
            switch (type, value) {
            case ("null", .null), ("boolean", .bool), ("number", .number),
                 ("string", .string), ("array", .array), ("object", .object):
                return true
            case ("integer", .number(let number)):
                return number.rounded() == number
            default:
                return false
            }
        }
    }

    private static func schemaObject(
        _ tool: SmeltToolDescriptor
    ) throws -> [String: Any] {
        guard let object = try JSONSerialization.jsonObject(
            with: Data(tool.schemaJSON.utf8)
        ) as? [String: Any] else {
            throw SmeltXMLFunctionToolCodecError.invalidSchema(tool: tool.name)
        }
        return object
    }

    private static func larkLiteral(_ value: String) throws -> String {
        let data = try JSONSerialization.data(withJSONObject: [value])
        return String(String(decoding: data, as: UTF8.self).dropFirst().dropLast())
    }

    /// Jinja's `tojson` shape uses a space after commas/colons. Dictionaries
    /// are sorted here to make semantically identical request schemas render to
    /// identical token IDs regardless of the JSON decoder's hash order.
    private static func templateJSON(_ value: SmeltJSONValue) -> String {
        switch value {
        case .null:
            return "null"
        case .bool(let value):
            return value ? "true" : "false"
        case .number(let value):
            if value.isFinite, value.rounded() == value {
                return String(format: "%.0f", value)
            }
            return String(value)
        case .string(let value):
            return templateJSONString(value)
        case .array(let values):
            return "[" + values.map(templateJSON).joined(separator: ", ") + "]"
        case .object(let fields):
            return "{" + fields.keys.sorted().map { key in
                templateJSONString(key) + ": " + templateJSON(fields[key]!)
            }.joined(separator: ", ") + "}"
        }
    }

    private static func templateJSONString(_ value: String) -> String {
        let data = try! JSONSerialization.data(
            withJSONObject: [value], options: [.withoutEscapingSlashes]
        )
        return String(String(decoding: data, as: UTF8.self).dropFirst().dropLast())
    }
}
