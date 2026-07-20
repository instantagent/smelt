import Darwin
import Foundation
import Testing
@testable import SmeltRuntime

private func makeManagedTempRoot(_ name: String) -> URL {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("\(name)-\(getpid())", isDirectory: true)
    try? FileManager.default.removeItem(at: root)
    try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    atexit_b {
        try? FileManager.default.removeItem(at: root)
    }
    return root
}

private let tokenizerTestTempRoot: URL = {
    makeManagedTempRoot("smelt-tokenizer-tests")
}()

private func writeFixture(_ fixture: [String: Any]) throws -> String {
    let data = try JSONSerialization.data(withJSONObject: fixture)
    let path = tokenizerTestTempRoot
        .appendingPathComponent("test_tokenizer_\(UUID().uuidString).json")
        .path
    try data.write(to: URL(fileURLWithPath: path))
    return path
}

private nonisolated(unsafe) let byteLevelFixture: [String: Any] = [
    "model": [
        "type": "BPE",
        "vocab": [
            "H": 0, "e": 1, "l": 2, "o": 3,
            "\u{0120}": 4,
            "w": 5, "r": 6, "d": 7,
            "ll": 8, "He": 9, "llo": 10, "Hello": 11,
        ] as [String: Int],
        "merges": ["l l", "H e", "ll o", "He llo"],
    ] as [String: Any],
    "pre_tokenizer": [
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true,
        "use_regex": true,
    ] as [String: Any],
    "added_tokens": [
        ["content": "<|begin_of_text|>", "id": 100, "special": true],
        ["content": "<|end_of_text|>", "id": 101, "special": true],
    ] as [[String: Any]],
]

private nonisolated(unsafe) let atomicGrammarTokenFixture: [String: Any] = [
    "model": [
        "type": "BPE",
        "vocab": ["x": 0],
        "merges": [],
    ] as [String: Any],
    "pre_tokenizer": [
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true,
        "use_regex": true,
    ] as [String: Any],
    "added_tokens": [
        ["content": "<tool_call>", "id": 100, "special": true],
        ["content": "<eos>", "id": 101, "special": true],
    ] as [[String: Any]],
]

private nonisolated(unsafe) let sentencePieceFixture: [String: Any] = [
    "model": [
        "type": "BPE",
        "vocab": [
            "▁": 0, "H": 1, "e": 2, "l": 3, "o": 4,
            "ll": 5, "He": 6, "llo": 7, "Hello": 8,
        ] as [String: Int],
        "merges": ["l l", "H e", "ll o", "He llo"],
    ] as [String: Any],
    "added_tokens": [
        ["content": "<s>", "id": 100, "special": true],
        ["content": "</s>", "id": 101, "special": true],
        ["content": "<unk>", "id": 102, "special": true],
        ["content": "<|im_start|>", "id": 103, "special": true],
    ] as [[String: Any]],
]

private nonisolated(unsafe) let sentencePieceLikeFixture: [String: Any] = [
    "model": [
        "type": "BPE",
        "vocab": [
            "T": 10,
            "h": 11,
            "e": 12,
            "Th": 13,
            "The": 818,
            "▁The": 669,
            "▁": 14,
        ] as [String: Int],
        "merges": [
            "T h",
            "Th e",
            "▁ T",
            "▁T h",
            "▁Th e",
        ] as [String],
    ] as [String: Any],
    "pre_tokenizer": [
        "type": "Split",
        "pattern": ["String": " "],
        "behavior": "MergedWithPrevious",
        "invert": false,
    ] as [String: Any],
    "normalizer": [
        "type": "Replace",
        "pattern": ["String": " "],
        "content": "▁",
    ] as [String: Any],
    "post_processor": [
        "type": "TemplateProcessing",
        "single": [
            ["SpecialToken": ["id": "<bos>", "type_id": 0]],
            ["Sequence": ["id": "A", "type_id": 0]],
        ],
        "special_tokens": [
            "<bos>": [
                "id": "<bos>",
                "ids": [2],
                "tokens": ["<bos>"],
            ]
        ],
    ] as [String: Any],
    "added_tokens": [
        ["content": "<pad>", "id": 0, "special": true],
        ["content": "<eos>", "id": 1, "special": true],
        ["content": "<bos>", "id": 2, "special": true],
        ["content": "<unk>", "id": 3, "special": true],
    ] as [[String: Any]],
]

private nonisolated(unsafe) let byteFallbackSentencePieceFixture: [String: Any] = [
    "model": [
        "type": "BPE",
        "vocab": [
            "<0xE2>": 0,
            "<0x82>": 1,
            "<0xAC>": 2,
            "▁": 3,
            "A": 4,
        ] as [String: Int],
        "merges": [],
    ] as [String: Any],
    "added_tokens": [
        ["content": "<s>", "id": 100, "special": true],
        ["content": "</s>", "id": 101, "special": true],
    ] as [[String: Any]],
]

@Test func byteLevelBPE_detectsBosEos() throws {
    let path = try writeFixture(byteLevelFixture)
    let tok = try SmeltTokenizer(path: path)
    #expect(tok.bosTokenId == 100)
    #expect(tok.eosTokenId == 101)
}

@Test func llguidanceGrammarLiteralsUseAtomicAddedTokens() throws {
    let path = try writeFixture(atomicGrammarTokenFixture)
    let tokenizer = try SmeltTokenizer(path: path)
    #expect(tokenizer.encode("<tool_call>") != [100])
    #expect(tokenizer.encodeWithSpecials("<tool_call>") == [100])

    let built = try SmeltLLGuidanceTokenizer(
        tokenizer: tokenizer, eosTokens: [101]
    )
    let restored = try SmeltLLGuidanceTokenizer(
        tokenizer: tokenizer,
        serializedTrie: try built.serializedTrie()
    )
    for llgTokenizer in [built, restored] {
        let matcher = try SmeltLLGuidanceMatcher(
            tokenizer: llgTokenizer,
            lark: #"start: "<tool_call>""#
        )
        let mask = try matcher.computeMask()
        #expect(SmeltLLGuidanceMatcher.tokenIsAllowed(100, in: mask))
        #expect(!SmeltLLGuidanceMatcher.tokenIsAllowed(0, in: mask))
        try matcher.consume(tokenIds: [100])
        #expect(matcher.isAccepting)
    }
}

@Test func byteLevelBPE_encodeSingleWord() throws {
    let path = try writeFixture(byteLevelFixture)
    let tok = try SmeltTokenizer(path: path)
    #expect(tok.encode("Hello") == [11])
}

@Test func byteLevelBPE_roundtrip() throws {
    let path = try writeFixture(byteLevelFixture)
    let tok = try SmeltTokenizer(path: path)
    let ids = tok.encode("Hello world")
    #expect(ids == [11, 4, 5, 3, 6, 2, 7])
    #expect(tok.decode(ids) == "Hello world")
}

@Test func sentencePiece_detectsBosEos() throws {
    let path = try writeFixture(sentencePieceFixture)
    let tok = try SmeltTokenizer(path: path)
    #expect(tok.bosTokenId == 100)
    #expect(tok.eosTokenId == 101)
}

@Test func sentencePiece_roundtrip() throws {
    let path = try writeFixture(sentencePieceFixture)
    let tok = try SmeltTokenizer(path: path)
    let ids = tok.encode("Hello")
    #expect(ids == [0, 8])
    #expect(tok.decode(ids) == "Hello")
}

@Test func tokenizerExposesAddedTokenId() throws {
    let path = try writeFixture(sentencePieceFixture)
    let tok = try SmeltTokenizer(path: path)
    #expect(tok.addedTokenId(for: "<|im_start|>") == 103)
}

@Test func sentencePieceLike_detectsBosEos() throws {
    let path = try writeFixture(sentencePieceLikeFixture)
    let tok = try SmeltTokenizer(path: path)
    #expect(tok.bosTokenId == 2)
    #expect(tok.eosTokenId == 1)
}

@Test func sentencePieceLike_doesNotInjectDummyPrefix() throws {
    let path = try writeFixture(sentencePieceLikeFixture)
    let tok = try SmeltTokenizer(path: path)
    #expect(tok.encode("The") == [818])
}

@Test func streamingDecoderBuffersSentencePieceByteFallbackUntilValidUTF8() throws {
    let path = try writeFixture(byteFallbackSentencePieceFixture)
    let tok = try SmeltTokenizer(path: path)
    var decoder = tok.makeStreamingDecoder()

    #expect(decoder.decode(tokenId: 0, tokenizer: tok) == "")
    #expect(decoder.decode(tokenId: 1, tokenizer: tok) == "")
    #expect(decoder.decode(tokenId: 2, tokenizer: tok) == "€")
}

@Test func streamingDecoderDropsInitialSentencePieceSpaceMarker() throws {
    let path = try writeFixture(sentencePieceFixture)
    let tok = try SmeltTokenizer(path: path)
    var decoder = tok.makeStreamingDecoder()

    #expect(decoder.decode(tokenId: 0, tokenizer: tok) == "")
    #expect(decoder.decode(tokenId: 8, tokenizer: tok) == "Hello")
}
