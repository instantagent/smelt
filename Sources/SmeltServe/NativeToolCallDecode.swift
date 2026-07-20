import Foundation
import SmeltRuntime

// Native tool-call decoding belongs to the reusable serving surface.

/// Buffered classification of a fully-generated chat-completions
/// response when a tool matcher was active. Used by the
/// non-streaming handler to decide between JSON-arm decode, native
/// arm decode, plain text, and strict-no-accept failure.
enum BufferedShape {
    case json, native, text, strictNoAccept
}

/// Pick the terminal interpretation of a fully-generated buffer.
/// Native shape is keyed on the presence of the start token ID
/// in the emitted tokens (the literal `<|tool_call>` is stripped
/// from `text` by the byte-level BPE decoder, so a text-prefix
/// check would never fire).
func classifyBufferedShape(
    text: String,
    tokens: [Int32],
    nativeToolCallStartTokenId: Int32?,
    finishReason: OpenAIFinishReason,
    toolMatcherSet: Bool,
    useUnion: Bool
) -> BufferedShape {
    if finishReason == .toolCalls { return .json }
    guard toolMatcherSet else { return .text }
    if !useUnion { return .strictNoAccept }
    if let startId = nativeToolCallStartTokenId,
       tokens.contains(startId)
    {
        return .native
    }
    return text.first(where: { !$0.isWhitespace }) == "{"
        ? .json : .text
}

enum NativeToolCallParseError: Error, CustomStringConvertible {
    case missingCallPrefix
    case missingArgs
    case emptyName
    case truncatedArgs
    case unknownToolName(String)

    var description: String {
        switch self {
        case .missingCallPrefix:
            return "expected `call:` after <|tool_call>"
        case .missingArgs:
            return "no `{` after function name"
        case .emptyName:
            return "function name is empty"
        case .truncatedArgs:
            return "args JSON object not closed"
        case .unknownToolName(let name):
            return "tool name \"\(name)\" is not in the active descriptor set"
        }
    }
}

/// Some instruct models were trained to emit tool calls as native special
/// tokens: `<|tool_call>call:NAME{args}<tool_call|>` (per
/// tokenizer_config.json's response_schema). The `<|tool_call>`
/// and `<tool_call|>` tokens are flagged `special: true` and so
/// get stripped by the byte-level BPE decoder before reaching
/// any text buffer (see SmeltTokenizer.decodeByteLevelBPE +
/// decodeStreamingToken). The decoded text therefore looks like
/// `call:NAME{args}` (with whatever leading/trailing whitespace
/// the tokenizer happened to emit around the special tokens).
/// Triggering on the start-token ID — not on text prefix — is
/// the only reliable signal.
///
/// Why this path matters: under tool_choice:"auto"'s union
/// grammar, the text-arm allows the `call:` body, so the matcher
/// doesn't reject the output. The previous code surfaced this as
/// plain assistant text content even though the model had every
/// intention of emitting a tool call.
func decodeNativeToolCall(
    _ text: String,
    allowedToolNames: Set<String>
) throws -> OpenAIToolCall {
    // Strict ASCII (RFC 8259) whitespace: space / tab / LF / CR.
    // Character.isWhitespace would also match NBSP / U+3000 etc.
    // and diverge from the streaming-path's byte-level
    // isJSONWhitespace check in ToolArmStreamState, producing
    // inconsistent classification of identical model output
    // across the buffered and streaming paths.
    //
    // Operate on the UTF-8 byte view rather than `Character`:
    // Swift collapses CRLF (`\r\n`) into a single grapheme
    // cluster, so a Character-level predicate would skip neither
    // `\r` nor `\n` when both arrive together and `\r\ncall:...`
    // would fail the prefix check.
    let asciiWhitespaceBytes: Set<UInt8> = [0x20, 0x09, 0x0A, 0x0D]
    let strippedBytes = Array(text.utf8.drop(while: {
        asciiWhitespaceBytes.contains($0)
    }))
    let stripped = String(decoding: strippedBytes, as: UTF8.self)
    let callLit = "call:"
    guard stripped.hasPrefix(callLit) else {
        throw NativeToolCallParseError.missingCallPrefix
    }
    let afterCall = stripped.dropFirst(callLit.count)
    guard let braceIndex = afterCall.firstIndex(of: "{") else {
        throw NativeToolCallParseError.missingArgs
    }
    let name = afterCall[afterCall.startIndex..<braceIndex]
        .trimmingCharacters(in: .whitespaces)
    guard !name.isEmpty else {
        throw NativeToolCallParseError.emptyName
    }
    // The strict JSON arm constrains the function name to the
    // descriptor enum via the matcher's grammar; the native arm
    // arrives via the union grammar's text arm which accepts
    // arbitrary `call:NAME{...}` bytes, so the descriptor check
    // is the only thing keeping a model from invoking a tool
    // outside the declared set.
    guard allowedToolNames.contains(name) else {
        throw NativeToolCallParseError.unknownToolName(name)
    }
    guard let argsEnd = scanJSONObjectEnd(
        in: afterCall, from: braceIndex
    ) else {
        throw NativeToolCallParseError.truncatedArgs
    }
    let argsSlice = afterCall[braceIndex..<argsEnd]
    let argsValue = try OpenAIJSON.decode(
        SmeltJSONValue.self,
        from: Data(String(argsSlice).utf8)
    )
    return OpenAIToolCall(
        id: SmeltToolCallID.next(),
        type: .function,
        function: OpenAIToolCallFunction(
            name: name,
            arguments: try SmeltJSON.canonicalString(argsValue)
        )
    )
}

/// Brace-depth + string-state scan that returns the index just
/// past the matching close brace of a JSON object starting at
/// `start`. Returns nil if the object is unterminated (so a
/// truncated emission fails decode rather than silently chopping
/// args). Whatever follows the close — the model's
/// `<tool_call|>` close tag, EOS, or nothing on truncation — is
/// ignored.
func scanJSONObjectEnd(
    in s: Substring, from start: Substring.Index
) -> Substring.Index? {
    var i = start
    guard i < s.endIndex, s[i] == "{" else { return nil }
    var depth = 0
    var inString = false
    var escaped = false
    while i < s.endIndex {
        let c = s[i]
        if escaped {
            escaped = false
        } else if inString {
            if c == "\\" { escaped = true }
            else if c == "\"" { inString = false }
        } else {
            switch c {
            case "\"": inString = true
            case "{": depth += 1
            case "}":
                depth -= 1
                if depth == 0 {
                    return s.index(after: i)
                }
            default: break
            }
        }
        i = s.index(after: i)
    }
    return nil
}
