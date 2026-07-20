import Foundation
import SmeltSchema

public enum SmeltNativeToolTranscriptCodecError: Error, CustomStringConvertible {
    case unsupported(String)
    case missingToken(codec: String, token: String)

    public var description: String {
        switch self {
        case .unsupported(let name):
            return "native tool transcript codec '\(name)' is not implemented"
        case .missingToken(let codec, let token):
            return "native tool transcript codec '\(codec)' requires tokenizer token \(token)"
        }
    }
}

/// Type-erased incremental decoder owned by a native tool transcript codec.
/// API adapters consume stable function starts without knowing the codec's
/// delimiters or wire syntax.
public protocol SmeltToolTranscriptStreamDecoding: AnyObject {
    func consume(_ chunk: String) throws -> [SmeltXMLFunctionToolStreamCallStart]
}

extension SmeltXMLFunctionToolStreamDecoder: SmeltToolTranscriptStreamDecoding {}

/// A package-selected native tool transcript brick. The package manifest owns
/// the stable identifier; this value owns every coupled runtime behavior for
/// that identifier so prompt rendering, grammar, decode, streaming, visible
/// control tokens, and checkpoint closure cannot drift independently.
public struct SmeltNativeToolTranscriptCodec: Sendable, Equatable {
    private enum Implementation: Sendable, Equatable {
        case xmlFunctionParameters
    }

    public let name: String
    private let implementation: Implementation

    private init(name: String, implementation: Implementation) {
        self.name = name
        self.implementation = implementation
    }

    public static func resolve(
        _ name: String?
    ) throws -> SmeltNativeToolTranscriptCodec? {
        guard let name else { return nil }
        switch name {
        case SmeltToolTranscriptCodecName.xmlFunctionParameters:
            return SmeltNativeToolTranscriptCodec(
                name: name,
                implementation: .xmlFunctionParameters
            )
        default:
            throw SmeltNativeToolTranscriptCodecError.unsupported(name)
        }
    }

    public var visibleGeneratedControlTokens: [String] {
        switch implementation {
        case .xmlFunctionParameters:
            return SmeltXMLFunctionToolCodec.visibleGeneratedControlTokens
        }
    }

    public func renderSystemMessage(
        tools: [SmeltToolDescriptor]
    ) throws -> String {
        switch implementation {
        case .xmlFunctionParameters:
            return try SmeltXMLFunctionToolCodec.renderSystemMessage(tools: tools)
        }
    }

    public func renderCalls(
        _ calls: [SmeltDecodedXMLToolCall]
    ) throws -> String {
        switch implementation {
        case .xmlFunctionParameters:
            return try SmeltXMLFunctionToolCodec.renderCalls(calls)
        }
    }

    public func decode(
        _ text: String,
        tools: [SmeltToolDescriptor]
    ) throws -> SmeltDecodedXMLToolResponse {
        switch implementation {
        case .xmlFunctionParameters:
            return try SmeltXMLFunctionToolCodec.decode(text, tools: tools)
        }
    }

    public func larkGrammar(
        for tools: [SmeltToolDescriptor],
        allowText: Bool
    ) throws -> String {
        switch implementation {
        case .xmlFunctionParameters:
            return try SmeltXMLFunctionToolCodec.larkGrammar(
                for: tools,
                allowText: allowText
            )
        }
    }

    public func containsToolCall(in text: String) -> Bool {
        switch implementation {
        case .xmlFunctionParameters:
            return text.contains("<tool_call>")
        }
    }

    public func couldBeginToolCall(
        firstSignificantCharacter: Character
    ) -> Bool {
        switch implementation {
        case .xmlFunctionParameters:
            return firstSignificantCharacter == "<"
        }
    }

    public func makeStreamDecoder(
        toolNames: [String]
    ) -> any SmeltToolTranscriptStreamDecoding {
        switch implementation {
        case .xmlFunctionParameters:
            return SmeltXMLFunctionToolStreamDecoder(toolNames: toolNames)
        }
    }

    /// Tokens that commit a completed assistant response to the canonical
    /// package transcript. Exact-state caches advance through these bytes
    /// before capturing the reusable checkpoint.
    public func completedAssistantTurnClosure(
        tokenizer: SmeltTokenizer
    ) throws -> [Int32] {
        switch implementation {
        case .xmlFunctionParameters:
            guard let end = tokenizer.addedTokenId(for: "<|im_end|>") else {
                throw SmeltNativeToolTranscriptCodecError.missingToken(
                    codec: name,
                    token: "<|im_end|>"
                )
            }
            return [Int32(end)] + tokenizer.encode("\n")
        }
    }
}
