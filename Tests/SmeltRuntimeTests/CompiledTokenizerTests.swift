// Parity gates for the compiled tokenizer (tokenizer.bin): the binary
// round-trip must behave identically to the tokenizer.json parse it was
// compiled from — encode, decode, specials, and the llguidance vocabulary.

import Darwin
import Foundation
import Testing
@testable import SmeltRuntime

private let compiledTokTempRoot: URL = {
    let root = FileManager.default.temporaryDirectory
        .appendingPathComponent("smelt-compiled-tokenizer-tests-\(getpid())", isDirectory: true)
    try? FileManager.default.removeItem(at: root)
    try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    atexit_b {
        try? FileManager.default.removeItem(at: root)
    }
    return root
}()

/// Writes the fixture as tokenizer.json in a fresh directory and returns both paths.
private func writePackageDir(_ fixture: [String: Any]) throws -> (json: String, dir: String) {
    let dir = compiledTokTempRoot
        .appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let json = dir.appendingPathComponent("tokenizer.json").path
    let data = try JSONSerialization.data(withJSONObject: fixture)
    try data.write(to: URL(fileURLWithPath: json))
    return (json, dir.path)
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

private nonisolated(unsafe) let sentencePieceFixture: [String: Any] = [
    "model": [
        "type": "BPE",
        "vocab": [
            "▁": 0, "H": 1, "e": 2, "l": 3, "o": 4,
            "ll": 5, "He": 6, "llo": 7, "Hello": 8,
            "<0x0A>": 9,
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

private func assertParity(
    _ json: SmeltTokenizer,
    _ compiled: SmeltTokenizer,
    samples: [String],
    idSequences: [[Int32]]
) {
    #expect(json.bosTokenId == compiled.bosTokenId)
    #expect(json.eosTokenId == compiled.eosTokenId)
    #expect(json.vocabularySize == compiled.vocabularySize)
    for text in samples {
        #expect(json.encode(text) == compiled.encode(text), "encode(\(text))")
        #expect(
            json.encodeWithSpecials(text) == compiled.encodeWithSpecials(text),
            "encodeWithSpecials(\(text))"
        )
    }
    for ids in idSequences {
        #expect(json.decode(ids) == compiled.decode(ids), "decode(\(ids))")
        for id in ids {
            #expect(
                json.tokenBytes(for: id) == compiled.tokenBytes(for: id),
                "tokenBytes(\(id))"
            )
        }
    }
    let jsonVocab = json.llguidanceVocabulary()
    let compiledVocab = compiled.llguidanceVocabulary()
    #expect(jsonVocab.vocabSize == compiledVocab.vocabSize)
    #expect(jsonVocab.eosTokens == compiledVocab.eosTokens)
    #expect(jsonVocab.tokenLengths == compiledVocab.tokenLengths)
    #expect(jsonVocab.tokenBytes == compiledVocab.tokenBytes)
}

@Suite struct CompiledTokenizerTests {
    @Test func byteLevelRoundTripParity() throws {
        let (jsonPath, dir) = try writePackageDir(byteLevelFixture)
        let json = try SmeltTokenizer(jsonPath: jsonPath)
        let binPath = "\(dir)/\(SmeltTokenizer.compiledFileName)"
        try json.writeCompiledTokenizer(to: binPath)
        let compiled = try SmeltTokenizer(compiledPath: binPath)
        assertParity(
            json, compiled,
            samples: ["Hello world", "Hello", " Hello<|end_of_text|>", ""],
            idSequences: [[11, 4, 5, 3, 6, 2, 7], [9, 10], [100, 11, 101]]
        )
    }

    @Test func sentencePieceRoundTripParity() throws {
        let (jsonPath, dir) = try writePackageDir(sentencePieceFixture)
        let json = try SmeltTokenizer(jsonPath: jsonPath)
        let binPath = "\(dir)/\(SmeltTokenizer.compiledFileName)"
        try json.writeCompiledTokenizer(to: binPath)
        let compiled = try SmeltTokenizer(compiledPath: binPath)
        assertParity(
            json, compiled,
            samples: ["Hello", "Hello Hello", ""],
            idSequences: [[8, 0, 8], [9, 9, 1], [100, 8, 101, 103]]
        )
    }

    @Test func pathInitPrefersCompiled() throws {
        let (jsonPath, dir) = try writePackageDir(byteLevelFixture)
        let json = try SmeltTokenizer(jsonPath: jsonPath)
        try json.writeCompiledTokenizer(to: "\(dir)/\(SmeltTokenizer.compiledFileName)")
        // Make the JSON unparseable: init(path:) must succeed via the binary.
        try Data("not json".utf8).write(to: URL(fileURLWithPath: jsonPath))
        let viaPath = try SmeltTokenizer(path: jsonPath)
        #expect(viaPath.encode("Hello world") == json.encode("Hello world"))
    }

    @Test func corruptPayloadFailsChecksumWhenVerified() throws {
        let (jsonPath, dir) = try writePackageDir(byteLevelFixture)
        let json = try SmeltTokenizer(jsonPath: jsonPath)
        let binPath = "\(dir)/\(SmeltTokenizer.compiledFileName)"
        try json.writeCompiledTokenizer(to: binPath)
        var bytes = try Data(contentsOf: URL(fileURLWithPath: binPath))
        bytes[bytes.count - 1] ^= 0xFF
        try bytes.write(to: URL(fileURLWithPath: binPath))
        #expect(throws: SmeltTokenizerError.self) {
            _ = try SmeltTokenizer(compiledPath: binPath, verifyChecksum: true)
        }
    }

    @Test func structurallyCorruptFallsBackToJSON() throws {
        let (jsonPath, dir) = try writePackageDir(byteLevelFixture)
        let json = try SmeltTokenizer(jsonPath: jsonPath)
        let binPath = "\(dir)/\(SmeltTokenizer.compiledFileName)"
        try json.writeCompiledTokenizer(to: binPath)
        // Zero the body: the section walk won't consume the full file, so the
        // structural validation must reject and init(path:) must fall back.
        var bytes = try Data(contentsOf: URL(fileURLWithPath: binPath))
        bytes.replaceSubrange(
            16..<bytes.count,
            with: Data(count: bytes.count - 16)
        )
        try bytes.write(to: URL(fileURLWithPath: binPath))
        let viaPath = try SmeltTokenizer(path: jsonPath)
        #expect(viaPath.encode("Hello world") == json.encode("Hello world"))
    }

    @Test func truncatedCompiledThrows() throws {
        let (jsonPath, dir) = try writePackageDir(byteLevelFixture)
        let json = try SmeltTokenizer(jsonPath: jsonPath)
        let binPath = "\(dir)/\(SmeltTokenizer.compiledFileName)"
        try json.writeCompiledTokenizer(to: binPath)
        let full = try Data(contentsOf: URL(fileURLWithPath: binPath))
        try full.prefix(full.count / 2).write(to: URL(fileURLWithPath: binPath))
        #expect(throws: SmeltTokenizerError.self) {
            _ = try SmeltTokenizer(compiledPath: binPath)
        }
    }

    /// Full-vocabulary parity against a real package, when one is built locally.
    @Test func realPackageParity() throws {
        let jsonPath =
            "artifacts/qwen35-0.8b-qmm16x128/Qwen_Qwen3.5-0.8B.smeltpkg/tokenizer.json"
        guard FileManager.default.fileExists(atPath: jsonPath) else { return }
        let json = try SmeltTokenizer(jsonPath: jsonPath)
        let binPath = compiledTokTempRoot
            .appendingPathComponent(SmeltTokenizer.compiledFileName).path
        try json.writeCompiledTokenizer(to: binPath)
        let compiled = try SmeltTokenizer(compiledPath: binPath)
        assertParity(
            json, compiled,
            samples: [
                "The capital of France is Paris.",
                "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)",
                "<|im_start|>user\nHi<|im_end|>",
                "数字 123 and émojis 🚀",
            ],
            idSequences: [Array(0..<64), [248044]]
        )
    }
}
