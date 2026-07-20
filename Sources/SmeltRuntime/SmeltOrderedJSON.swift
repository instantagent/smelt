import Foundation

/// A JSON value that retains object-member order.
///
/// JSON object order is not semantic, but it is part of the token stream when
/// a model template renders tool descriptors with Jinja's `tojson` filter.
/// `JSONDecoder` deliberately erases that order, so native prompt adapters use
/// this small parser whenever byte-for-byte template parity matters.
public indirect enum SmeltOrderedJSONValue: Equatable, Sendable {
    public struct Member: Equatable, Sendable {
        public let name: String
        public let value: SmeltOrderedJSONValue

        public init(name: String, value: SmeltOrderedJSONValue) {
            self.name = name
            self.value = value
        }
    }

    case null
    case bool(Bool)
    /// The validated JSON number spelling. Retaining the spelling also avoids
    /// a Double round trip changing a frozen prompt.
    case number(String)
    case string(String)
    case array([SmeltOrderedJSONValue])
    case object([Member])

    public static func parse(_ data: Data) throws -> Self {
        var parser = SmeltOrderedJSONParser(bytes: Array(data))
        let value = try parser.parseValue()
        try parser.requireEnd()
        return value
    }

    public static func parse(_ text: String) throws -> Self {
        try parse(Data(text.utf8))
    }

    public func member(named name: String) -> Self? {
        guard case .object(let members) = self else { return nil }
        return members.first(where: { $0.name == name })?.value
    }

    public var arrayValues: [Self]? {
        guard case .array(let values) = self else { return nil }
        return values
    }

    /// Compact valid JSON that retains the source object's member order.
    public var compactJSON: String {
        rendered(memberSeparator: ",", nameSeparator: ":")
    }

    /// The spacing used by the pinned Transformers/Jinja `tojson` filter.
    public var templateJSON: String {
        rendered(memberSeparator: ", ", nameSeparator: ": ")
    }

    private func rendered(
        memberSeparator: String,
        nameSeparator: String
    ) -> String {
        switch self {
        case .null:
            return "null"
        case .bool(let value):
            return value ? "true" : "false"
        case .number(let spelling):
            return spelling
        case .string(let value):
            return Self.jsonString(value)
        case .array(let values):
            return "[" + values.map {
                $0.rendered(
                    memberSeparator: memberSeparator,
                    nameSeparator: nameSeparator
                )
            }.joined(separator: memberSeparator) + "]"
        case .object(let members):
            return "{" + members.map { member in
                Self.jsonString(member.name) + nameSeparator
                    + member.value.rendered(
                        memberSeparator: memberSeparator,
                        nameSeparator: nameSeparator
                    )
            }.joined(separator: memberSeparator) + "}"
        }
    }

    private static func jsonString(_ value: String) -> String {
        let data = try! JSONSerialization.data(
            withJSONObject: [value],
            options: [.withoutEscapingSlashes]
        )
        return String(String(decoding: data, as: UTF8.self).dropFirst().dropLast())
    }
}

public enum SmeltOrderedJSONError: Error, CustomStringConvertible {
    case invalid(String)

    public var description: String {
        switch self {
        case .invalid(let detail):
            return "invalid ordered JSON: \(detail)"
        }
    }
}

private struct SmeltOrderedJSONParser {
    let bytes: [UInt8]
    var index = 0

    mutating func parseValue() throws -> SmeltOrderedJSONValue {
        skipWhitespace()
        guard index < bytes.count else { throw error("unexpected end of input") }
        switch bytes[index] {
        case 0x6E: // n
            try consume("null")
            return .null
        case 0x74: // t
            try consume("true")
            return .bool(true)
        case 0x66: // f
            try consume("false")
            return .bool(false)
        case 0x22: // "
            return .string(try parseString())
        case 0x5B: // [
            return try parseArray()
        case 0x7B: // {
            return try parseObject()
        case 0x2D, 0x30...0x39:
            return .number(try parseNumber())
        default:
            throw error("unexpected byte \(bytes[index])")
        }
    }

    mutating func requireEnd() throws {
        skipWhitespace()
        guard index == bytes.count else { throw error("trailing content") }
    }

    private mutating func parseArray() throws -> SmeltOrderedJSONValue {
        index += 1
        skipWhitespace()
        if consumeIf(0x5D) { return .array([]) }
        var values: [SmeltOrderedJSONValue] = []
        while true {
            values.append(try parseValue())
            skipWhitespace()
            if consumeIf(0x5D) { return .array(values) }
            try require(0x2C, expected: "',' or ']'")
        }
    }

    private mutating func parseObject() throws -> SmeltOrderedJSONValue {
        index += 1
        skipWhitespace()
        if consumeIf(0x7D) { return .object([]) }
        var members: [SmeltOrderedJSONValue.Member] = []
        while true {
            skipWhitespace()
            guard index < bytes.count, bytes[index] == 0x22 else {
                throw error("expected object member name")
            }
            let name = try parseString()
            skipWhitespace()
            try require(0x3A, expected: "':'")
            let value = try parseValue()
            members.append(.init(name: name, value: value))
            skipWhitespace()
            if consumeIf(0x7D) { return .object(members) }
            try require(0x2C, expected: "',' or '}'")
        }
    }

    private mutating func parseString() throws -> String {
        let start = index
        index += 1
        var escaped = false
        while index < bytes.count {
            let byte = bytes[index]
            index += 1
            if escaped {
                escaped = false
            } else if byte == 0x5C {
                escaped = true
            } else if byte == 0x22 {
                let data = Data(bytes[start..<index])
                do {
                    return try JSONDecoder().decode(String.self, from: data)
                } catch {
                    throw self.error("invalid string")
                }
            } else if byte < 0x20 {
                throw error("unescaped control character in string")
            }
        }
        throw error("unterminated string")
    }

    private mutating func parseNumber() throws -> String {
        let start = index
        if consumeIf(0x2D), index == bytes.count {
            throw error("incomplete number")
        }
        if consumeIf(0x30) {
            if index < bytes.count, bytes[index] >= 0x30, bytes[index] <= 0x39 {
                throw error("leading zero in number")
            }
        } else {
            try consumeDigits(required: true)
        }
        if consumeIf(0x2E) { try consumeDigits(required: true) }
        if consumeIf(0x65) || consumeIf(0x45) {
            _ = consumeIf(0x2B) || consumeIf(0x2D)
            try consumeDigits(required: true)
        }
        return String(decoding: bytes[start..<index], as: UTF8.self)
    }

    private mutating func consumeDigits(required: Bool) throws {
        let start = index
        while index < bytes.count, bytes[index] >= 0x30, bytes[index] <= 0x39 {
            index += 1
        }
        if required, start == index { throw error("expected digit") }
    }

    private mutating func consume(_ literal: StaticString) throws {
        let expected = Array(String(describing: literal).utf8)
        guard index + expected.count <= bytes.count,
              Array(bytes[index..<(index + expected.count)]) == expected
        else { throw error("expected \(literal)") }
        index += expected.count
    }

    private mutating func require(_ byte: UInt8, expected: String) throws {
        skipWhitespace()
        guard consumeIf(byte) else { throw error("expected \(expected)") }
    }

    private mutating func consumeIf(_ byte: UInt8) -> Bool {
        guard index < bytes.count, bytes[index] == byte else { return false }
        index += 1
        return true
    }

    private mutating func skipWhitespace() {
        while index < bytes.count,
              bytes[index] == 0x20 || bytes[index] == 0x0A
                || bytes[index] == 0x0D || bytes[index] == 0x09 {
            index += 1
        }
    }

    private func error(_ detail: String) -> SmeltOrderedJSONError {
        .invalid("\(detail) at byte \(index)")
    }
}
