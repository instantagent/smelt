// SmeltPreparedPrefix — a prompt prefix whose prefill was computed at package build time.
//
// Package preparation can prefill a fixed prompt prefix (typically a system
// preamble) once and store the resulting KV + recurrent state in the
// package:
//
//   prepared_prefix.json      — { "version": 1, "token_ids": [...] }
//   prepared_prefix.snapshot  — SmeltPromptSnapshot file (mmap'd on load)
//
// At runtime, SmeltModel.generate restores this state instead of prefilling
// whenever the request's token IDs start with the prepared IDs (exact token
// match, so chat-template or tokenizer drift safely falls back to a full
// prefill). Set SMELT_NO_PREPARED_PREFIX=1 to ignore a prepared prefix entirely.

import Foundation
import SmeltSchema

public struct SmeltPreparedPromptContinuation: Codable, Sendable, Equatable {
    public static let tokenizerEncodePromptEncoding = "tokenizer.encode"

    public let version: Int
    public let template: String
    public let thinkingPolicy: SmeltThinkingPolicy
    public let promptEncoding: String
    public let promptSuffixTokenIds: [Int32]

    public init(
        version: Int = 1,
        template: String,
        thinkingPolicy: SmeltThinkingPolicy,
        promptEncoding: String = Self.tokenizerEncodePromptEncoding,
        promptSuffixTokenIds: [Int32]
    ) {
        self.version = version
        self.template = template
        self.thinkingPolicy = thinkingPolicy
        self.promptEncoding = promptEncoding
        self.promptSuffixTokenIds = promptSuffixTokenIds
    }

    public func matches(
        template: String,
        thinkingPolicy: SmeltThinkingPolicy
    ) -> Bool {
        version == 1
            && self.template == template
            && self.thinkingPolicy == thinkingPolicy
            && promptEncoding == Self.tokenizerEncodePromptEncoding
    }

    enum CodingKeys: String, CodingKey {
        case version
        case template
        case thinkingPolicy = "thinking_policy"
        case promptEncoding = "prompt_encoding"
        case promptSuffixTokenIds = "prompt_suffix_token_ids"
    }
}

public enum SmeltPromptTemplateError: Error, CustomStringConvertible, Equatable {
    case missingChatTemplateTokens(template: String)

    public var description: String {
        switch self {
        case .missingChatTemplateTokens(let template):
            return "missing chat-template tokens for \(template)"
        }
    }
}

public struct SmeltPreparedSystemPromptContinuationPlan: Sendable, Equatable {
    public let prefixTailTokenIds: [Int32]
    public let continuation: SmeltPreparedPromptContinuation

    public init(
        prefixTailTokenIds: [Int32],
        continuation: SmeltPreparedPromptContinuation
    ) {
        self.prefixTailTokenIds = prefixTailTokenIds
        self.continuation = continuation
    }
}

public enum SmeltPreparedPromptContinuationBuilder {
    public static func systemPromptPlan(
        tokenizer: SmeltTokenizer,
        template: String,
        thinkingPolicy: SmeltThinkingPolicy
    ) throws -> SmeltPreparedSystemPromptContinuationPlan? {
        switch template {
        case SmeltPromptTemplateName.chatML,
             SmeltPromptTemplateName.chatMLXMLTools:
            guard let imStart = tokenizer.addedTokenId(for: "<|im_start|>"),
                  let imEnd = tokenizer.addedTokenId(for: "<|im_end|>")
            else {
                throw SmeltPromptTemplateError.missingChatTemplateTokens(template: "chatml")
            }

            var prefixTail: [Int32] = [Int32(imStart)]
            prefixTail += tokenizer.encode("user")
            prefixTail += tokenizer.encode("\n")

            var suffix: [Int32] = [Int32(imEnd)]
            suffix += tokenizer.encode("\n")
            suffix += try chatMLAssistantPrelude(
                tokenizer: tokenizer,
                thinkingPolicy: thinkingPolicy
            )

            return SmeltPreparedSystemPromptContinuationPlan(
                prefixTailTokenIds: prefixTail,
                continuation: SmeltPreparedPromptContinuation(
                    template: template,
                    thinkingPolicy: thinkingPolicy,
                    promptSuffixTokenIds: suffix
                )
            )

        default:
            return nil
        }
    }

    public static func inputIds(
        prompt: String,
        tokenizer: SmeltTokenizer,
        preparedPrefixTokenIds: [Int32],
        continuation: SmeltPreparedPromptContinuation,
        template: String,
        thinkingPolicy: SmeltThinkingPolicy
    ) -> [Int32]? {
        guard continuation.matches(
            template: template,
            thinkingPolicy: thinkingPolicy
        ) else {
            return nil
        }

        return preparedPrefixTokenIds
            + tokenizer.encode(prompt)
            + continuation.promptSuffixTokenIds
    }

    public static func inputIds(
        prompt: String,
        tokenizer: SmeltTokenizer,
        preparedPrefixTokenIds: [Int32],
        continuation: SmeltPreparedPromptContinuation?,
        template: String,
        thinkingPolicy: SmeltThinkingPolicy,
        fullInputIds: [Int32]
    ) -> [Int32] {
        guard let continuation else {
            return preparedPrefixTokenIds + fullInputIds
        }
        return inputIds(
            prompt: prompt,
            tokenizer: tokenizer,
            preparedPrefixTokenIds: preparedPrefixTokenIds,
            continuation: continuation,
            template: template,
            thinkingPolicy: thinkingPolicy
        ) ?? fullInputIds
    }

    private static func chatMLAssistantPrelude(
        tokenizer: SmeltTokenizer,
        thinkingPolicy: SmeltThinkingPolicy
    ) throws -> [Int32] {
        guard let imStart = tokenizer.addedTokenId(for: "<|im_start|>") else {
            throw SmeltPromptTemplateError.missingChatTemplateTokens(template: "chatml")
        }
        var ids: [Int32] = [Int32(imStart)]
        ids += tokenizer.encode("assistant")
        ids += tokenizer.encode("\n")
        guard thinkingPolicy == .disabled else { return ids }
        guard let think = tokenizer.addedTokenId(for: "<think>"),
              let thinkEnd = tokenizer.addedTokenId(for: "</think>")
        else {
            throw SmeltPromptTemplateError.missingChatTemplateTokens(template: "chatml")
        }
        ids += [Int32(think)]
        ids += tokenizer.encode("\n\n")
        ids += [Int32(thinkEnd)]
        ids += tokenizer.encode("\n\n")
        return ids
    }
}

struct SmeltPreparedPrefix {
    static let metaFileName = SmeltPreparedArtifacts.prefixMetadata
    static let snapshotFileName = SmeltPreparedArtifacts.prefixSnapshot

    struct Meta: Codable {
        let version: Int
        let tokenIds: [Int32]
        let continuation: SmeltPreparedPromptContinuation?

        enum CodingKeys: String, CodingKey {
            case version
            case tokenIds = "token_ids"
            case continuation
        }
    }

    let tokenIds: [Int32]
    let continuation: SmeltPreparedPromptContinuation?
    let snapshot: SmeltPromptSnapshot

    /// Load the prepared prefix from a package directory. The package file
    /// inventory is authoritative, so a partial or corrupt pair fails loudly.
    static func load(packagePath: String) throws -> SmeltPreparedPrefix? {
        guard ProcessInfo.processInfo.environment["SMELT_NO_PREPARED_PREFIX"] != "1"
        else { return nil }
        let fileManager = FileManager.default
        let metadataPresent = fileManager.fileExists(
            atPath: "\(packagePath)/\(metaFileName)"
        )
        let snapshotPresent = fileManager.fileExists(
            atPath: "\(packagePath)/\(snapshotFileName)"
        )
        guard metadataPresent || snapshotPresent else { return nil }
        guard metadataPresent, snapshotPresent else {
            throw SmeltPreparedArtifactError.incomplete(
                component: "prepared prefix",
                missing: metadataPresent ? snapshotFileName : metaFileName
            )
        }
        return try parse(packagePath: packagePath)
    }

    private static func parse(packagePath: String) throws -> SmeltPreparedPrefix {
        let meta = try JSONDecoder().decode(
            Meta.self,
            from: Data(contentsOf: URL(
                fileURLWithPath: "\(packagePath)/\(metaFileName)"))
        )
        guard meta.version == 1, !meta.tokenIds.isEmpty else {
            throw SmeltPromptSnapshotFileError.invalidMagic
        }
        let snapshot = try SmeltPromptSnapshot.read(
            from: URL(fileURLWithPath: "\(packagePath)/\(snapshotFileName)")
        )
        guard snapshot.promptLength == meta.tokenIds.count else {
            throw SmeltPromptSnapshotFileError.truncatedPayload
        }
        return SmeltPreparedPrefix(
            tokenIds: meta.tokenIds,
            continuation: meta.continuation,
            snapshot: snapshot
        )
    }

    /// Write a prepared prefix while assembling a package.
    static func write(
        packagePath: String,
        tokenIds: [Int32],
        snapshot: SmeltPromptSnapshot,
        continuation: SmeltPreparedPromptContinuation? = nil
    ) throws -> SmeltPromptSnapshotWriteInfo {
        let info = try snapshot.write(
            to: URL(fileURLWithPath: "\(packagePath)/\(snapshotFileName)")
        )
        let meta = Meta(version: 1, tokenIds: tokenIds, continuation: continuation)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(meta).write(
            to: URL(fileURLWithPath: "\(packagePath)/\(metaFileName)"),
            options: .atomic
        )
        return info
    }
}

/// The package-authored prompt contract needed by non-`SmeltModel` frontends
/// such as the OpenAI serve transport. Snapshot bytes remain private to the
/// runtime; callers receive only the exact token prefix and its continuation.
public struct SmeltPreparedPromptPrefix: Sendable {
    public let tokenIds: [Int32]
    public let continuation: SmeltPreparedPromptContinuation?
    public let snapshot: SmeltPromptSnapshot

    public init(
        tokenIds: [Int32],
        continuation: SmeltPreparedPromptContinuation?,
        snapshot: SmeltPromptSnapshot
    ) {
        self.tokenIds = tokenIds
        self.continuation = continuation
        self.snapshot = snapshot
    }

    public static func load(packagePath: String) throws -> SmeltPreparedPromptPrefix? {
        try SmeltPreparedPrefix.load(packagePath: packagePath).map {
            SmeltPreparedPromptPrefix(
                tokenIds: $0.tokenIds,
                continuation: $0.continuation,
                snapshot: $0.snapshot
            )
        }
    }
}

/// A JSON schema compiled into the package. `smelt run` loads it into an
/// llguidance matcher and constrains every generation.
/// Set SMELT_NO_COMPILED_GRAMMAR=1 to ignore it.
public struct SmeltCompiledGrammar {
    public static let fileName = SmeltPreparedArtifacts.grammarMetadata
    /// Serialized llguidance token trie (`SmeltLLGuidanceTokenizer.serializedTrie()`).
    /// Optional: when present, `smelt run` reconstructs the llguidance
    /// tokenizer from it instead of re-building the trie (~0.4s on a 250k
    /// vocab); when absent or unloadable it falls back to the full build.
    public static let trieFileName = SmeltPreparedArtifacts.grammarTrie

    struct Meta: Codable {
        let version: Int
        let jsonSchema: String

        enum CodingKeys: String, CodingKey {
            case version
            case jsonSchema = "json_schema"
        }
    }

    public let jsonSchema: String
    public let serializedTrie: Data?

    /// Load a compiled grammar when present. Metadata is required; the trie is
    /// an optional accelerator and rebuilds from metadata when absent.
    public static func load(packagePath: String) throws -> SmeltCompiledGrammar? {
        guard ProcessInfo.processInfo.environment["SMELT_NO_COMPILED_GRAMMAR"] != "1"
        else { return nil }
        let fileManager = FileManager.default
        let metadataPresent = fileManager.fileExists(atPath: "\(packagePath)/\(fileName)")
        let triePresent = fileManager.fileExists(atPath: "\(packagePath)/\(trieFileName)")
        guard metadataPresent || triePresent else { return nil }
        guard metadataPresent else {
            throw SmeltPreparedArtifactError.incomplete(
                component: "compiled grammar",
                missing: fileName
            )
        }
        return try parse(packagePath: packagePath)
    }

    private static func parse(packagePath: String) throws -> SmeltCompiledGrammar {
        let meta = try JSONDecoder().decode(
            Meta.self,
            from: Data(contentsOf: URL(
                fileURLWithPath: "\(packagePath)/\(fileName)"))
        )
        guard meta.version == 1, !meta.jsonSchema.isEmpty else {
            throw SmeltCompiledGrammarError.malformed
        }
        // The trie is a perf accelerator: a missing/unreadable one just means
        // the slow rebuild path, never a failure.
        let serializedTrie = try? Data(
            contentsOf: URL(fileURLWithPath: "\(packagePath)/\(trieFileName)"),
            options: .mappedIfSafe
        )
        return SmeltCompiledGrammar(
            jsonSchema: meta.jsonSchema, serializedTrie: serializedTrie
        )
    }

    public static func write(
        packagePath: String,
        jsonSchema: String,
        serializedTrie: Data? = nil
    ) throws {
        let meta = Meta(version: 1, jsonSchema: jsonSchema)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(meta).write(
            to: URL(fileURLWithPath: "\(packagePath)/\(fileName)"),
            options: .atomic
        )
        if let serializedTrie {
            try serializedTrie.write(
                to: URL(fileURLWithPath: "\(packagePath)/\(trieFileName)"),
                options: .atomic
            )
        }
    }
}

public enum SmeltCompiledGrammarError: Error, CustomStringConvertible, Equatable {
    case malformed

    public var description: String {
        switch self {
        case .malformed:
            return "compiled grammar metadata is malformed (bad version or empty schema)"
        }
    }
}

public enum SmeltPreparedArtifactError: Error, CustomStringConvertible, Equatable {
    case incomplete(component: String, missing: String)

    public var description: String {
        switch self {
        case .incomplete(let component, let missing):
            return "\(component) is incomplete; missing '\(missing)'"
        }
    }
}
