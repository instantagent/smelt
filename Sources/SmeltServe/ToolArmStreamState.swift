import Foundation
import SmeltRuntime

// Streaming tool-call state shared by all text server consumers.

/// Streaming state for the tool-arm of `/v1/chat/completions`.
/// Drives incremental emission of the OpenAI tool_call deltas:
/// initial chunk carries id + parsed function name + empty args;
/// subsequent chunks carry args-string fragments as the JSON
/// object accumulates. A depth counter (string- and escape-aware)
/// stops args emission when the args-object closes — anything
/// past that point is the outer JSON wrapper `}` which belongs to
/// the wire protocol shape, not to the args value.
final class ToolArmStreamState {
    var initialEmitted = false
    // The class owns the UTF-8 byte buffer and appends only the
    // new chunk's bytes per call — O(chunkSize) per advance, not
    // O(totalSize). Stored Int offsets index directly into this
    // buffer; we never materialize a fresh [UInt8] from the full
    // String per call, never use `utf8.index(startIndex,
    // offsetBy: N)` (O(N) on non-RandomAccess UTF8View), and
    // never use `utf8.distance(...)`. Total per-stream parse
    // work is O(N) as documented.
    private var bytes: [UInt8] = []
    private var argsEmittedEndOffset: Int? = nil
    private var argsDepth = 0
    private var argsInString = false
    private var argsEscape = false
    private var argsClosed = false
    private let toolCallId: String = SmeltToolCallID.next()
    // Independent resume cursors for the two top-level keys we
    // care about. The name scan walks the whole buffer until it
    // matches `"name":"`; the args scan walks the whole buffer
    // until it matches `"arguments":`. Running them as separate
    // cursors costs O(2N) total (vs O(N) if the cursor was
    // shared after a match) but correctly handles either key
    // ordering: JSON Schema's `required:[name,arguments]`
    // doesn't pin emission order, and a shared cursor would
    // miss `arguments` if the matcher ever emits it first.
    private var nameScanState = ScanState(
        offset: 0, depth: 0, inString: false, escape: false
    )
    private var argsScanState = ScanState(
        offset: 0, depth: 0, inString: false, escape: false
    )
    // Set once name's `"name":"` is matched; the value's closing
    // quote may take more tokens to arrive, so the resume point
    // is cached separately from the scanner state.
    private var nameValueOpenOffset: Int? = nil
    // ASCII byte needles for the two top-level keys we care
    // about, cached once instead of rebuilt per call.
    private static let nameNeedle: [UInt8] = Array("name".utf8)
    private static let argsNeedle: [UInt8] = Array("arguments".utf8)

    struct ScanState {
        var offset: Int   // UTF-8 byte offset into bytes buffer
        var depth: Int
        var inString: Bool
        var escape: Bool
    }

    /// `chunkText` is ONLY the bytes newly generated since the
    /// previous advance() call (not the accumulated buffer); the
    /// class maintains its own running byte buffer to avoid
    /// per-call O(N) copies.
    func advance(
        chunkText: String,
        sendDelta: (OpenAIChatStreamDelta) async throws -> Void
    ) async throws {
        bytes.append(contentsOf: chunkText.utf8)
        if !initialEmitted {
            if nameValueOpenOffset == nil {
                guard let valueOpen = Self.locateAfterKey(
                    needle: Self.nameNeedle,
                    bytes: bytes,
                    state: &nameScanState
                ) else { return }
                nameValueOpenOffset = valueOpen
            }
            // Walk to the closing `"` of the name value. If we
            // haven't seen the close yet, defer to the next call.
            let valueOpen = nameValueOpenOffset!
            var probe = valueOpen
            var escape = false
            var closingQuote: Int? = nil
            while probe < bytes.count {
                let byte = bytes[probe]
                if escape { escape = false }
                else if byte == 0x5C { escape = true }       // '\\'
                else if byte == 0x22 {                       // '"'
                    closingQuote = probe
                    break
                }
                probe += 1
            }
            guard let closeOff = closingQuote else { return }
            let nameStr = String(
                decoding: bytes[valueOpen..<closeOff],
                as: UTF8.self
            )
            try await sendDelta(
                OpenAIChatStreamDelta(
                    role: .assistant,
                    toolCalls: [OpenAIChatStreamToolCallDelta(
                        index: 0,
                        id: toolCallId,
                        type: .function,
                        function: OpenAIChatStreamFunctionDelta(
                            name: nameStr, arguments: ""
                        )
                    )]
                )
            )
            initialEmitted = true
        }
        if argsEmittedEndOffset == nil {
            guard let argsAfter = Self.locateAfterKey(
                needle: Self.argsNeedle,
                bytes: bytes,
                state: &argsScanState
            ) else { return }
            argsEmittedEndOffset = argsAfter
        }
        guard !argsClosed,
              let startOffset = argsEmittedEndOffset,
              startOffset < bytes.count
        else { return }
        // State machine operates byte-by-byte. JSON's structural
        // bytes (`"` `\` `{` `}`) are ASCII and never appear as
        // continuation bytes of a multi-byte UTF-8 sequence (those
        // are 0x80-0xBF), so byte-level state transitions are safe
        // even inside non-ASCII string values. The delta is
        // reconstructed from the byte slice at the end so multi-
        // byte characters survive verbatim.
        var i = startOffset
        while i < bytes.count, !argsClosed {
            let byte = bytes[i]
            if argsEscape {
                argsEscape = false
            } else if argsInString {
                if byte == 0x5C { argsEscape = true }         // '\\'
                else if byte == 0x22 { argsInString = false } // '"'
            } else {
                switch byte {
                case 0x22: argsInString = true                // '"'
                case 0x7B: argsDepth += 1                     // '{'
                case 0x7D:                                    // '}'
                    argsDepth -= 1
                    if argsDepth == 0 { argsClosed = true }
                default: break
                }
            }
            i += 1
        }
        argsEmittedEndOffset = i
        if startOffset != i {
            let deltaStr = String(
                decoding: bytes[startOffset..<i],
                as: UTF8.self
            )
            try await sendDelta(
                OpenAIChatStreamDelta(
                    toolCalls: [OpenAIChatStreamToolCallDelta(
                        index: 0, id: nil, type: nil,
                        function: OpenAIChatStreamFunctionDelta(
                            name: nil, arguments: deltaStr
                        )
                    )]
                )
            )
        }
    }

    /// Locate the index immediately after a JSON key's value-open
    /// quote (for string values) or AT the value's `{`/etc (for
    /// object values). Caller passes in `state` whose `offset` is
    /// the resume point from the previous call — the scanner
    /// only walks new bytes, making the per-stream cost O(N)
    /// instead of O(N²). On match, `state.offset` is left past
    /// the closing quote of `"<key>"` (caller may further
    /// advance past the value). On miss, `state.offset` is at
    /// end-of-bytes and the next call resumes from there with
    /// preserved depth/string/escape.
    ///
    /// Only matches keys at top-level depth (depth==1, directly
    /// inside the outer `{`) — a tool with an argument literally
    /// named `name` won't trick this scanner because JSON Schema's
    /// `required:[name,arguments]` doesn't pin key emission order.
    private static func locateAfterKey(
        needle: [UInt8],
        bytes: [UInt8],
        state: inout ScanState
    ) -> Int? {
        var i = state.offset
        var depth = state.depth
        var inString = state.inString
        var escape = state.escape
        let end = bytes.count
        while i < end {
            let byte = bytes[i]
            if escape {
                escape = false
                i += 1
                continue
            }
            if inString {
                if byte == 0x5C { escape = true }       // '\\'
                else if byte == 0x22 { inString = false } // '"'
                i += 1
                continue
            }
            switch byte {
            case 0x7B: depth += 1                       // '{'
            case 0x7D: depth -= 1                       // '}'
            case 0x22:                                  // '"'
                if depth == 1 {
                    switch matchKeyStart(bytes: bytes, at: i, needle: needle) {
                    case .match(let matchEnd):
                        if let resumeAfter = advancePastKeySeparator(
                            bytes: bytes, from: matchEnd
                        ) {
                            state.offset = matchEnd
                            state.depth = depth
                            state.inString = inString
                            state.escape = escape
                            return resumeAfter
                        }
                        // Key bytes matched but value-open hasn't
                        // arrived yet. Pause AT the opening `"`
                        // so the next call re-runs the full
                        // match (key + separator + value-open).
                        state.offset = i
                        state.depth = depth
                        state.inString = false
                        state.escape = false
                        return nil
                    case .needsMore:
                        // Pause AT the opening quote; next call
                        // resumes with more bytes available.
                        state.offset = i
                        state.depth = depth
                        state.inString = false
                        state.escape = false
                        return nil
                    case .mismatch:
                        inString = true
                    }
                } else {
                    inString = true
                }
            default: break
            }
            i += 1
        }
        state.offset = i
        state.depth = depth
        state.inString = inString
        state.escape = escape
        return nil
    }

    enum KeyMatchResult {
        case match(Int)   // UTF-8 offset just after the key's closing `"`
        case mismatch
        case needsMore
    }

    /// Three-state result so the scanner can distinguish "ran out
    /// of buffered bytes mid-match" from "definitely not this
    /// key". Operates on the UTF-8 byte buffer — tool-call keys
    /// are ASCII so byte comparison is correct.
    private static func matchKeyStart(
        bytes: [UInt8], at idx: Int, needle: [UInt8]
    ) -> KeyMatchResult {
        var probe = idx + 1
        for needleByte in needle {
            if probe >= bytes.count { return .needsMore }
            if bytes[probe] != needleByte { return .mismatch }
            probe += 1
        }
        if probe >= bytes.count { return .needsMore }
        guard bytes[probe] == 0x22 else { return .mismatch }   // '"'
        return .match(probe + 1)
    }

    /// After a matched `"<key>"`, skip optional whitespace + `:`
    /// + optional whitespace, then position past `"` (string
    /// value) or AT `{`/etc (object value). Whitespace match is
    /// JSON-strict (space/tab/LF/CR per RFC 8259). Operates on
    /// UTF-8 bytes — all of these are ASCII so byte comparison
    /// is correct. Returns nil if the buffer ends mid-separator
    /// (caller pauses and resumes when more bytes arrive).
    private static func advancePastKeySeparator(
        bytes: [UInt8], from start: Int
    ) -> Int? {
        var i = start
        while i < bytes.count, Self.isJSONWhitespace(bytes[i]) {
            i += 1
        }
        guard i < bytes.count, bytes[i] == 0x3A else { return nil } // ':'
        i += 1
        while i < bytes.count, Self.isJSONWhitespace(bytes[i]) {
            i += 1
        }
        guard i < bytes.count else { return nil }
        if bytes[i] == 0x22 { return i + 1 }                        // '"'
        return i
    }

    /// RFC 8259 whitespace: space, tab, line feed, carriage
    /// return. Restricted to ASCII byte comparison; matches the
    /// llguidance JSON grammar exactly (Character.isWhitespace
    /// would also match NBSP / U+3000 etc.).
    private static func isJSONWhitespace(_ byte: UInt8) -> Bool {
        return byte == 0x20 || byte == 0x09
            || byte == 0x0A || byte == 0x0D
    }
}
