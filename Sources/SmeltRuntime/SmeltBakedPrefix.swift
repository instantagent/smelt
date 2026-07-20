// SmeltBakedPrefix — a prompt prefix whose prefill was computed at bake time.
//
// Package preparation can prefill a fixed prompt prefix (typically a system
// preamble) once and store the resulting KV + recurrent state in the
// package:
//
//   baked_prefix.json      — { "version": 1, "token_ids": [...] }
//   baked_prefix.snapshot  — SmeltPromptSnapshot file (mmap'd on load)
//
// At runtime, SmeltModel.generate restores this state instead of prefilling
// whenever the request's token IDs start with the baked IDs (exact token
// match, so chat-template or tokenizer drift safely falls back to a full
// prefill). Set SMELT_NO_BAKED_PREFIX=1 to ignore a baked prefix entirely —
// useful for A/B-ing baked vs. fresh prefill.

import Foundation
import SmeltSchema

public struct SmeltBakedPromptContinuation: Codable, Sendable, Equatable {
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

public struct SmeltBakedSystemPromptContinuationPlan: Sendable, Equatable {
    public let prefixTailTokenIds: [Int32]
    public let continuation: SmeltBakedPromptContinuation

    public init(
        prefixTailTokenIds: [Int32],
        continuation: SmeltBakedPromptContinuation
    ) {
        self.prefixTailTokenIds = prefixTailTokenIds
        self.continuation = continuation
    }
}

public enum SmeltBakedPromptContinuationBuilder {
    public static func systemPromptPlan(
        tokenizer: SmeltTokenizer,
        template: String,
        thinkingPolicy: SmeltThinkingPolicy
    ) throws -> SmeltBakedSystemPromptContinuationPlan? {
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

            return SmeltBakedSystemPromptContinuationPlan(
                prefixTailTokenIds: prefixTail,
                continuation: SmeltBakedPromptContinuation(
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
        bakedPrefixTokenIds: [Int32],
        continuation: SmeltBakedPromptContinuation,
        template: String,
        thinkingPolicy: SmeltThinkingPolicy
    ) -> [Int32]? {
        guard continuation.matches(
            template: template,
            thinkingPolicy: thinkingPolicy
        ) else {
            return nil
        }

        return bakedPrefixTokenIds
            + tokenizer.encode(prompt)
            + continuation.promptSuffixTokenIds
    }

    public static func inputIds(
        prompt: String,
        tokenizer: SmeltTokenizer,
        bakedPrefixTokenIds: [Int32],
        continuation: SmeltBakedPromptContinuation?,
        template: String,
        thinkingPolicy: SmeltThinkingPolicy,
        unbakedInputIds: [Int32]
    ) -> [Int32] {
        guard let continuation else {
            return bakedPrefixTokenIds + unbakedInputIds
        }
        return inputIds(
            prompt: prompt,
            tokenizer: tokenizer,
            bakedPrefixTokenIds: bakedPrefixTokenIds,
            continuation: continuation,
            template: template,
            thinkingPolicy: thinkingPolicy
        ) ?? unbakedInputIds
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

struct SmeltBakedPrefix {
    static let metaFileName = SmeltBakeArtifacts.prefixMeta
    static let snapshotFileName = SmeltBakeArtifacts.prefixSnapshot

    struct Meta: Codable {
        let version: Int
        let tokenIds: [Int32]
        let continuation: SmeltBakedPromptContinuation?

        enum CodingKeys: String, CodingKey {
            case version
            case tokenIds = "token_ids"
            case continuation
        }
    }

    let tokenIds: [Int32]
    let continuation: SmeltBakedPromptContinuation?
    let snapshot: SmeltPromptSnapshot

    /// Load the baked prefix from a package directory, or nil when absent.
    /// A present-but-unloadable prefix warns and returns nil — generation
    /// then just pays the full prefill. This is the legacy / opted-out path; a
    /// package whose `baked.json` *declares* a prefix is loaded via `loadStrict`.
    static func load(packagePath: String) -> SmeltBakedPrefix? {
        guard ProcessInfo.processInfo.environment["SMELT_NO_BAKED_PREFIX"] != "1"
        else { return nil }
        guard FileManager.default.fileExists(
            atPath: "\(packagePath)/\(metaFileName)") else { return nil }
        do {
            return try parse(packagePath: packagePath)
        } catch {
            fputs(
                "SmeltModel: baked prefix at \(packagePath)/\(metaFileName) failed "
                    + "to load (\(error)); requests will prefill from scratch\n",
                stderr
            )
            return nil
        }
    }

    /// Strict load for a *declared* baked prefix: throws on a missing or corrupt
    /// artifact instead of silently degrading to a full prefill. Honors no env
    /// opt-out — the caller decides whether the component is enforced.
    static func loadStrict(packagePath: String) throws -> SmeltBakedPrefix {
        try parse(packagePath: packagePath)
    }

    private static func parse(packagePath: String) throws -> SmeltBakedPrefix {
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
        return SmeltBakedPrefix(
            tokenIds: meta.tokenIds,
            continuation: meta.continuation,
            snapshot: snapshot
        )
    }

    /// Write a baked prefix into a package directory.
    static func write(
        packagePath: String,
        tokenIds: [Int32],
        snapshot: SmeltPromptSnapshot,
        continuation: SmeltBakedPromptContinuation? = nil
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
public struct SmeltBakedPromptPrefix: Sendable {
    public let tokenIds: [Int32]
    public let continuation: SmeltBakedPromptContinuation?
    public let snapshot: SmeltPromptSnapshot

    public init(
        tokenIds: [Int32],
        continuation: SmeltBakedPromptContinuation?,
        snapshot: SmeltPromptSnapshot
    ) {
        self.tokenIds = tokenIds
        self.continuation = continuation
        self.snapshot = snapshot
    }

    public static func load(packagePath: String) throws -> SmeltBakedPromptPrefix? {
        let ignored = SmeltBakeManifest.ignoredFromEnv()
        guard !ignored.contains(.prefix) else { return nil }
        let marker = try SmeltBakeManifest.load(packagePath: packagePath)
        let prefix: SmeltBakedPrefix?
        if marker?.declares(.prefix) == true {
            prefix = try SmeltBakedPrefix.loadStrict(packagePath: packagePath)
        } else {
            prefix = SmeltBakedPrefix.load(packagePath: packagePath)
        }
        return prefix.map {
            SmeltBakedPromptPrefix(
                tokenIds: $0.tokenIds,
                continuation: $0.continuation,
                snapshot: $0.snapshot
            )
        }
    }
}

/// A JSON schema sealed into the package (`smelt create --json-schema`).
/// `smelt run` compiles it into an llguidance matcher and constrains every
/// generation, so the package's output contract ships with the package.
/// Set SMELT_NO_BAKED_GRAMMAR=1 to ignore it.
public struct SmeltBakedGrammar {
    public static let fileName = SmeltBakeArtifacts.grammarMeta
    /// Serialized llguidance token trie (`SmeltLLGuidanceTokenizer.serializedTrie()`).
    /// Optional: when present, `smelt run` reconstructs the llguidance
    /// tokenizer from it instead of re-building the trie (~0.4s on a 250k
    /// vocab); when absent or unloadable it falls back to the full build.
    public static let trieFileName = SmeltBakeArtifacts.grammarTrie

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

    /// Legacy / opted-out load: nil on absent or unloadable (generation then runs
    /// unconstrained). A package whose `baked.json` *declares* a grammar is
    /// validated via `loadStrict`.
    public static func load(packagePath: String) -> SmeltBakedGrammar? {
        guard ProcessInfo.processInfo.environment["SMELT_NO_BAKED_GRAMMAR"] != "1"
        else { return nil }
        guard FileManager.default.fileExists(
            atPath: "\(packagePath)/\(fileName)") else { return nil }
        do {
            return try parse(packagePath: packagePath)
        } catch {
            fputs(
                "SmeltBakedGrammar: \(packagePath)/\(fileName) failed to load "
                    + "(\(error)); generation will be unconstrained\n",
                stderr
            )
            return nil
        }
    }

    /// Strict load for a *declared* baked grammar: throws on a missing or corrupt
    /// schema artifact. Honors no env opt-out (the caller decides enforcement).
    public static func loadStrict(packagePath: String) throws -> SmeltBakedGrammar {
        try parse(packagePath: packagePath)
    }

    private static func parse(packagePath: String) throws -> SmeltBakedGrammar {
        let meta = try JSONDecoder().decode(
            Meta.self,
            from: Data(contentsOf: URL(
                fileURLWithPath: "\(packagePath)/\(fileName)"))
        )
        guard meta.version == 1, !meta.jsonSchema.isEmpty else {
            throw SmeltBakedGrammarError.malformed
        }
        // The trie is a perf accelerator: a missing/unreadable one just means
        // the slow rebuild path, never a failure.
        let serializedTrie = try? Data(
            contentsOf: URL(fileURLWithPath: "\(packagePath)/\(trieFileName)"),
            options: .mappedIfSafe
        )
        return SmeltBakedGrammar(
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

public enum SmeltBakedGrammarError: Error, CustomStringConvertible, Equatable {
    case malformed

    public var description: String {
        switch self {
        case .malformed:
            return "baked grammar metadata is malformed (bad version or empty schema)"
        }
    }
}
