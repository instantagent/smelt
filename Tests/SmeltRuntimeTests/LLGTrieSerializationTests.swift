// Gates for the serialized llguidance token trie (baked_grammar.trie):
// a tokenizer reconstructed from serializedTrie() must behave identically
// to one built from the vocabulary — same vocab/EOS metadata and, through
// a JSON-schema matcher, identical masks before and after consuming
// tokens. Gated on a locally built package (full 250k-vocab round trip).

import Foundation
import Testing
@testable import SmeltRuntime

private let qwen08bPackage =
    "artifacts/qwen35-0.8b-qmm16x128/Qwen_Qwen3.5-0.8B.smeltpkg"

private let schema = """
    {
      "type": "object",
      "properties": {
        "answer": {"type": "string", "maxLength": 80}
      },
      "required": ["answer"],
      "additionalProperties": false
    }
    """

@Suite struct LLGTrieSerializationTests {
    @Test func serializedTrieMatchesFromVocabularyBuild() throws {
        guard FileManager.default.fileExists(atPath: qwen08bPackage) else { return }
        let tokenizer = try SmeltTokenizer(path: "\(qwen08bPackage)/tokenizer.json")
        // Bake configuration: slicer built and embedded in the container.
        // The restored tokenizer's masks must equal a plain from-vocabulary
        // build's (the slicer is an optimization, never a semantic change).
        let baked = try SmeltLLGuidanceTokenizer(
            tokenizer: tokenizer, buildSlicer: true
        )
        let trie = try baked.serializedTrie()
        let built = try SmeltLLGuidanceTokenizer(tokenizer: tokenizer)
        let restored = try SmeltLLGuidanceTokenizer(
            tokenizer: tokenizer, serializedTrie: trie
        )

        #expect(restored.vocabSize == built.vocabSize)
        #expect(restored.eosTokens == built.eosTokens)

        let builtMatcher = try SmeltLLGuidanceMatcher(
            tokenizer: built, jsonSchema: schema
        )
        let restoredMatcher = try SmeltLLGuidanceMatcher(
            tokenizer: restored, jsonSchema: schema
        )

        var builtMask = try builtMatcher.computeMask()
        var restoredMask = try restoredMatcher.computeMask()
        #expect(builtMask == restoredMask)

        // Walk a few constrained steps: at each, consume the first allowed
        // token on both matchers and re-compare the next mask.
        for _ in 0..<8 {
            guard let word = builtMask.firstIndex(where: { $0 != 0 }) else { break }
            let bit = builtMask[word].trailingZeroBitCount
            let token = Int32(word * 32 + bit)
            try builtMatcher.consume(tokenIds: [token])
            try restoredMatcher.consume(tokenIds: [token])
            if builtMatcher.isStopped { break }
            builtMask = try builtMatcher.computeMask()
            restoredMask = try restoredMatcher.computeMask()
            #expect(builtMask == restoredMask)
        }
    }

    @Test func corruptTrieBytesThrow() throws {
        guard FileManager.default.fileExists(atPath: qwen08bPackage) else { return }
        let tokenizer = try SmeltTokenizer(path: "\(qwen08bPackage)/tokenizer.json")
        #expect(throws: SmeltLLGuidanceError.self) {
            _ = try SmeltLLGuidanceTokenizer(
                tokenizer: tokenizer, serializedTrie: Data("garbage".utf8)
            )
        }
        let built = try SmeltLLGuidanceTokenizer(tokenizer: tokenizer)
        var truncated = try built.serializedTrie()
        truncated.removeLast()
        #expect(throws: SmeltLLGuidanceError.self) {
            _ = try SmeltLLGuidanceTokenizer(
                tokenizer: tokenizer, serializedTrie: truncated
            )
        }
    }
}
