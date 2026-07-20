// SmeltTokenizer — Minimal BPE tokenizer for HuggingFace tokenizer.json files.
//
// Auto-detects two styles from the JSON:
//   - SentencePiece: ▁ space markers (Qwen, TinyLlama, Llama 2)
//   - Byte-level BPE: GPT-2 byte encoding (Llama 3, GPT, Mistral v2+)
//
// Not a general-purpose tokenizer library. Just enough to make
// `smelt run --prompt "..."` work.

import Foundation
import SmeltSchema

public struct SmeltLLGuidanceVocabulary: Sendable {
    public let vocabSize: Int
    public let eosTokens: [UInt32]
    public let tokenLengths: [UInt32]
    public let tokenBytes: [UInt8]

    public init(
        vocabSize: Int,
        eosTokens: [UInt32],
        tokenLengths: [UInt32],
        tokenBytes: [UInt8]
    ) {
        self.vocabSize = vocabSize
        self.eosTokens = eosTokens
        self.tokenLengths = tokenLengths
        self.tokenBytes = tokenBytes
    }
}

public struct SmeltTokenizer {

    // MARK: - Tokenizer style

    enum Style {
        case sentencePiece
        case byteLevelBPE
    }

    /// Token/merge lookup backend. The JSON inits build Swift dictionaries;
    /// a compiled tokenizer.bin maps precomputed hash tables instead, so
    /// load cost is independent of vocabulary size.
    enum Lookup {
        case dictionaries(
            vocab: [String: Int],
            reverseVocab: [Int: String],
            mergeRanks: [String: Int]
        )
        case mapped(MappedTokenizerTables)
    }
    private let lookup: Lookup

    /// Token string → token ID
    private func vocabId(_ token: String) -> Int? {
        switch lookup {
        case .dictionaries(let vocab, _, _):
            return vocab[token]
        case .mapped(let tables):
            return tables.lookupVocab(token)
        }
    }

    /// Token ID → token string
    private func tokenString(forId id: Int) -> String? {
        switch lookup {
        case .dictionaries(_, let reverseVocab, _):
            return reverseVocab[id]
        case .mapped(let tables):
            return tables.tokenString(forId: id)
        }
    }

    /// Merge pair ("tokA tokB") → priority (lower = higher priority)
    private func mergeRank(_ pair: String) -> Int? {
        switch lookup {
        case .dictionaries(_, _, let mergeRanks):
            return mergeRanks[pair]
        case .mapped(let tables):
            return tables.lookupMerge(pair)
        }
    }

    private var maxModelTokenId: Int {
        switch lookup {
        case .dictionaries(_, let reverseVocab, _):
            return reverseVocab.keys.max() ?? -1
        case .mapped(let tables):
            return tables.idCapacity - 1
        }
    }
    /// Added token content → token ID
    private let addedTokenIds: [String: Int]
    /// Added token ID → content string (reverse of addedTokenIds).
    /// Lets decode emit non-special added_tokens as their literal content;
    /// special-flagged
    /// added_tokens are still stripped via specialTokenIds first.
    private let addedTokenContentById: [Int: String]
    /// BOS token ID (if any)
    public let bosTokenId: Int?
    /// EOS token ID (if any)
    public let eosTokenId: Int?
    /// Special token IDs to skip during decode
    private let specialTokenIds: Set<Int>
    /// Detected tokenizer style
    private let style: Style
    /// Regex for pre-tokenization (byte-level BPE only)
    private let preTokenPattern: NSRegularExpression?
    /// Whether sentencepiece-style encoding should inject a leading dummy space marker.
    private let prependSentencePieceSpaceMarker: Bool
    /// Whether `encode` NFC-normalizes input first (Qwen2 has an NFC normalizer; the
    /// tokenizer.json path leaves normalization to its own normalizer block / caller).
    private let normalizesInputNFC: Bool

    // MARK: - GPT-2 byte encoding

    /// Byte (0-255) → Unicode character used in BPE vocabulary.
    /// Printable bytes map to themselves; others shift to U+0100+.
    private static let byteToUnicode: [UInt8: Character] = {
        var table: [UInt8: Character] = [:]
        var n: UInt32 = 0
        for b in 0..<256 {
            let byte = UInt8(b)
            let codePoint: UInt32
            if (byte >= 0x21 && byte <= 0x7E)
                || (byte >= 0xA1 && byte <= 0xAC)
                || (byte >= 0xAE)
            {
                codePoint = UInt32(byte)
            } else {
                codePoint = 256 + n
                n += 1
            }
            table[byte] = Character(UnicodeScalar(codePoint)!)
        }
        return table
    }()

    /// Reverse: Unicode character → byte
    private static let unicodeToByte: [Character: UInt8] = {
        var table: [Character: UInt8] = [:]
        for (byte, char) in byteToUnicode {
            table[char] = byte
        }
        return table
    }()

    /// Default pre-tokenization regex (GPT-4 / Llama 3 pattern)
    private static let defaultByteLevelPattern =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    /// Qwen2 pre-tokenization regex. Differs from the default on digit runs (`\p{N}`
    /// single-digit, not `\p{N}{1,3}`) — Qwen2 emits one token per digit. Verbatim from the
    /// checkpoint's fast-tokenizer pre_tokenizer Split pattern.
    private static let qwen2ByteLevelPattern =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    // MARK: - Init

    /// Load the tokenizer for a package file path. Prefers a compiled
    /// `tokenizer.bin` sibling (written at package build time) — it loads in
    /// milliseconds where the 12 MB tokenizer.json parse costs ~900ms — and
    /// falls back to parsing the JSON when the binary is absent or invalid.
    public init(path: String) throws {
        let dir = (path as NSString).deletingLastPathComponent
        let compiledPath =
            dir.isEmpty ? Self.compiledFileName : "\(dir)/\(Self.compiledFileName)"
        if FileManager.default.fileExists(atPath: compiledPath) {
            do {
                try self.init(compiledPath: compiledPath)
                return
            } catch {
                fputs(
                    "SmeltTokenizer: compiled tokenizer at \(compiledPath) "
                        + "failed to load (\(error)); falling back to JSON\n",
                    stderr
                )
            }
        }
        try self.init(jsonPath: path)
    }

    /// Load from a tokenizer.json file inside a .smeltpkg.
    public init(jsonPath path: String) throws {
        let fm = FileManager.default
        guard fm.fileExists(atPath: path) else {
            throw SmeltTokenizerError.fileNotFound(path)
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = json["model"] as? [String: Any],
              let rawMerges = model["merges"] as? [Any]
        else {
            throw SmeltTokenizerError.invalidFormat
        }

        // JSONSerialization silently strips leading U+FEFF (BOM) from
        // JSON string keys, collapsing BOM-prefixed vocab entries onto
        // their bare twins. Some tokenizers have several such
        // collisions including bare `#` (id 236865) being overwritten
        // by `"﻿#"` (id 208867). JSONDecoder preserves BOM-prefixed
        // keys, so a second pass scoped to the vocab block recovers
        // those entries.
        //
        // Caveat: ~430 additional vocab entries with NFC-
        // canonical-equivalent forms (Devanagari combining marks,
        // SentencePiece accent compositions, etc.) are still lost via
        // JSONDecoder because Swift's String canonicalizes on
        // assignment. Recovering those would require a byte-level
        // JSON parser. None show up in ASCII English code generation,
        // which is what unblocked the HumanEval bench — track that
        // as a follow-up if multi-script completion quality regresses.
        let vocabPairs: [(String, Int)]
        do {
            let root = try JSONDecoder().decode(TokenizerJSONRoot.self, from: data)
            vocabPairs = root.model.vocab.pairs
        } catch {
            fputs("SmeltTokenizer: vocab JSON decode failed: \(error)\n", stderr)
            throw SmeltTokenizerError.invalidFormat
        }
        var vocabDict: [String: Int] = [:]
        var reverseVocabBuf: [Int: String] = [:]
        vocabDict.reserveCapacity(vocabPairs.count)
        reverseVocabBuf.reserveCapacity(vocabPairs.count)
        for (key, value) in vocabPairs {
            vocabDict[key] = value
            reverseVocabBuf[value] = key
        }

        var ranks: [String: Int] = [:]
        for (idx, merge) in rawMerges.enumerated() {
            if let str = merge as? String {
                let parts = str.split(separator: " ", maxSplits: 1)
                guard parts.count == 2 else { continue }
                ranks["\(parts[0]) \(parts[1])"] = idx
            } else if let arr = merge as? [String], arr.count == 2 {
                ranks["\(arr[0]) \(arr[1])"] = idx
            }
        }
        self.lookup = .dictionaries(
            vocab: vocabDict,
            reverseVocab: reverseVocabBuf,
            mergeRanks: ranks
        )

        var isByteLevelBPE = false
        var regexPattern: String? = nil
        var prependSentencePieceSpaceMarker = true
        if let preTokenizer = json["pre_tokenizer"] as? [String: Any] {
            (isByteLevelBPE, regexPattern, prependSentencePieceSpaceMarker) =
                Self.inspectPreTokenizer(preTokenizer)
        }

        if isByteLevelBPE {
            self.style = .byteLevelBPE
            let pat = regexPattern ?? Self.defaultByteLevelPattern
            self.preTokenPattern = try? NSRegularExpression(pattern: pat)
        } else {
            self.style = .sentencePiece
            self.preTokenPattern = nil
        }
        self.prependSentencePieceSpaceMarker = prependSentencePieceSpaceMarker
        self.normalizesInputNFC = false

        var bos: Int? = nil
        var eos: Int? = nil
        var specials: Set<Int> = []
        var addedIds: [String: Int] = [:]
        if let addedTokens = json["added_tokens"] as? [[String: Any]] {
            for tok in addedTokens {
                guard let content = tok["content"] as? String,
                      let id = tok["id"] as? Int
                else { continue }
                addedIds[content] = id
                let isSpecial = tok["special"] as? Bool ?? false
                if isSpecial { specials.insert(id) }
                if content == "<s>" || content == "<bos>" { bos = id; specials.insert(id) }
                if content == "</s>" || content == "<eos>" { eos = id; specials.insert(id) }
                if content == "<unk>" { specials.insert(id) }
                if content == "<|begin_of_text|>" { bos = bos ?? id; specials.insert(id) }
                if content == "<|end_of_text|>" { eos = eos ?? id; specials.insert(id) }
                if content == "<|endoftext|>" { eos = eos ?? id; specials.insert(id) }
            }
        }
        self.addedTokenIds = addedIds
        self.addedTokenContentById = Dictionary(
            uniqueKeysWithValues: addedIds.map { ($1, $0) }
        )
        self.bosTokenId = bos
        self.eosTokenId = eos
        self.specialTokenIds = specials
    }

    /// Load a Qwen2-style byte-level BPE tokenizer from the HF *slow* files
    /// (`vocab.json` + `merges.txt` + `tokenizer_config.json`). Used for checkpoints that
    /// ship no `tokenizer.json` (e.g. Qwen3-TTS). Uses the exact Qwen2 pre-tokenizer regex;
    /// special/added tokens come from `tokenizer_config.json`'s `added_tokens_decoder`.
    public init(qwenVocabJSONPath: String, mergesTxtPath: String, tokenizerConfigPath: String) throws {
        let fm = FileManager.default
        for p in [qwenVocabJSONPath, mergesTxtPath, tokenizerConfigPath] where !fm.fileExists(atPath: p) {
            throw SmeltTokenizerError.fileNotFound(p)
        }

        let vocabData = try Data(contentsOf: URL(fileURLWithPath: qwenVocabJSONPath))
        let vocabPairs: [(String, Int)]
        do {
            vocabPairs = try JSONDecoder().decode(TokenizerJSONVocab.self, from: vocabData).pairs
        } catch {
            fputs("SmeltTokenizer: vocab.json decode failed: \(error)\n", stderr)
            throw SmeltTokenizerError.invalidFormat
        }
        var vocabDict: [String: Int] = [:]
        var reverseVocabBuf: [Int: String] = [:]
        vocabDict.reserveCapacity(vocabPairs.count)
        reverseVocabBuf.reserveCapacity(vocabPairs.count)
        for (key, value) in vocabPairs { vocabDict[key] = value; reverseVocabBuf[value] = key }

        // merges.txt: an optional `#version` header on line 0, then one `tokA tokB` pair per line.
        // Only the version header is a comment — `#` is itself a byte-level token, so real merges
        // like `# #` / `# include` start with `#` and must NOT be skipped. Rank = pair order.
        let mergesText = try String(contentsOf: URL(fileURLWithPath: mergesTxtPath), encoding: .utf8)
        var ranks: [String: Int] = [:]
        var rank = 0
        for (idx, line) in mergesText.split(separator: "\n", omittingEmptySubsequences: true).enumerated() {
            if idx == 0, line.hasPrefix("#version") { continue }
            let parts = line.split(separator: " ", maxSplits: 1)
            guard parts.count == 2 else { continue }
            ranks["\(parts[0]) \(parts[1])"] = rank
            rank += 1
        }
        self.lookup = .dictionaries(
            vocab: vocabDict,
            reverseVocab: reverseVocabBuf,
            mergeRanks: ranks
        )

        self.style = .byteLevelBPE
        self.preTokenPattern = try? NSRegularExpression(pattern: Self.qwen2ByteLevelPattern)
        self.prependSentencePieceSpaceMarker = false
        self.normalizesInputNFC = true

        // added_tokens_decoder: { "<id>": { "content": "<|im_start|>", "special": true, ... }, ... }
        var bos: Int? = nil, eos: Int? = nil
        var specials: Set<Int> = []
        var addedIds: [String: Int] = [:]
        let cfgData = try Data(contentsOf: URL(fileURLWithPath: tokenizerConfigPath))
        if let cfg = try JSONSerialization.jsonObject(with: cfgData) as? [String: Any],
           let decoder = cfg["added_tokens_decoder"] as? [String: Any] {
            for (idKey, raw) in decoder {
                guard let id = Int(idKey), let tok = raw as? [String: Any],
                      let content = tok["content"] as? String else { continue }
                addedIds[content] = id
                if tok["special"] as? Bool ?? false { specials.insert(id) }
                // Resolved bos/eos are always special even if the decoder entry omits the flag —
                // decode/tokenBytes must strip them (matches the tokenizer.json init).
                if content == "</s>" || content == "<eos>" { eos = eos ?? id; specials.insert(id) }
                if content == "<|end_of_text|>" || content == "<|endoftext|>" { eos = eos ?? id; specials.insert(id) }
                if content == "<s>" || content == "<bos>" || content == "<|begin_of_text|>" { bos = bos ?? id; specials.insert(id) }
            }
        }
        // A Qwen chat checkpoint always defines added tokens (<|im_start|> etc.). An empty set
        // means a missing/corrupt config — fail loudly, else encodeWithSpecials would silently
        // byte-BPE the chat markers instead of emitting their atomic ids.
        guard !addedIds.isEmpty else { throw SmeltTokenizerError.invalidFormat }
        self.addedTokenIds = addedIds
        self.addedTokenContentById = Dictionary(uniqueKeysWithValues: addedIds.map { ($1, $0) })
        self.bosTokenId = bos
        self.eosTokenId = eos
        self.specialTokenIds = specials
    }

    // MARK: - Metadata

    public func addedTokenId(for content: String) -> Int? {
        addedTokenIds[content]
    }

    /// Reverse of `addedTokenId(for:)`: given an added-token ID,
    /// return the literal content string (e.g., `<|custom|>`).
    public func addedTokenContent(for id: Int) -> String? {
        addedTokenContentById[id]
    }

    /// Raw bytes encoded by a single token ID, or nil for special
    /// tokens (which contribute no bytes to a decoded string). Word
    /// boundary detection in byte-level BPE relies on the first
    /// byte of a token (0x20 = space → new word). Going via this
    /// method avoids the multi-byte UTF-8 split bug that
    /// `decode([id])` has when a Unicode codepoint straddles two
    /// BPE tokens — each piece would fail UTF-8 decoding alone, but
    /// the byte sequence concatenates cleanly.
    public func tokenBytes(for id: Int32) -> [UInt8]? {
        let intId = Int(id)
        if specialTokenIds.contains(intId) { return nil }
        if let token = tokenString(forId: intId) {
            switch style {
            case .byteLevelBPE:
                var bytes: [UInt8] = []
                for char in token {
                    if let byte = Self.unicodeToByte[char] {
                        bytes.append(byte)
                    }
                }
                return bytes
            case .sentencePiece:
                if token.hasPrefix("<0x") && token.hasSuffix(">") {
                    let hex = String(token.dropFirst(3).dropLast(1))
                    if let byte = UInt8(hex, radix: 16) { return [byte] }
                }
                return Array(token.replacingOccurrences(of: "▁", with: " ").utf8)
            }
        }
        if let content = addedTokenContentById[intId] {
            return Array(content.utf8)
        }
        return nil
    }

    public func makeStreamingDecoder() -> SmeltStreamingTokenDecoder {
        SmeltStreamingTokenDecoder()
    }

    public var vocabularySize: Int {
        let maxAddedId = addedTokenIds.values.max() ?? -1
        return max(maxModelTokenId, maxAddedId) + 1
    }

    public func llguidanceVocabulary(
        eosTokens requestedEOSTokens: [Int32] = []
    ) -> SmeltLLGuidanceVocabulary {
        if case .mapped(let tables) = lookup, let baked = tables.llgVocabulary() {
            return SmeltLLGuidanceVocabulary(
                vocabSize: baked.lengths.count,
                eosTokens: resolvedLLGEOSTokens(
                    requestedEOSTokens, size: baked.lengths.count
                ),
                tokenLengths: baked.lengths,
                tokenBytes: baked.bytes
            )
        }

        let addedById = Dictionary(
            uniqueKeysWithValues: addedTokenIds.map { ($1, $0) }
        )
        let size = vocabularySize
        var lengths: [UInt32] = []
        var bytes: [UInt8] = []
        lengths.reserveCapacity(size)

        for id in 0..<size {
            let token = tokenString(forId: id) ?? addedById[id]
            let tokenBytes = token.map { llguidanceTokenBytes(for: $0) } ?? []
            lengths.append(UInt32(tokenBytes.count))
            bytes.append(contentsOf: tokenBytes)
        }

        return SmeltLLGuidanceVocabulary(
            vocabSize: size,
            eosTokens: resolvedLLGEOSTokens(requestedEOSTokens, size: size),
            tokenLengths: lengths,
            tokenBytes: bytes
        )
    }

    private func resolvedLLGEOSTokens(
        _ requested: [Int32], size: Int
    ) -> [UInt32] {
        var eos = requested
            .filter { $0 >= 0 }
            .map { UInt32($0) }
        if eos.isEmpty, let eosTokenId {
            eos = [UInt32(eosTokenId)]
        }
        if eos.isEmpty, size > 0 {
            eos = [UInt32(size - 1)]
        }
        return eos
    }

    // MARK: - Pre-tokenizer inspection

    private static func inspectPreTokenizer(_ config: [String: Any]) -> (Bool, String?, Bool) {
        let type = config["type"] as? String ?? ""

        if type == "ByteLevel" {
            return (true, nil, false)
        }

        if type == "Split",
           let patternObj = config["pattern"] as? [String: Any],
           let pattern = patternObj["String"] as? String,
           pattern == " ",
           let behavior = config["behavior"] as? String,
           behavior == "MergedWithPrevious"
        {
            // This BPE uses space->▁ replacement without the classic dummy
            // sentencepiece prefix on the first token.
            return (false, nil, false)
        }

        if type == "Sequence",
           let pretokenizers = config["pretokenizers"] as? [[String: Any]]
        {
            var hasByteLevel = false
            var pattern: String? = nil
            var prependSentencePieceSpaceMarker = true
            for pt in pretokenizers {
                let ptType = pt["type"] as? String ?? ""
                if ptType == "ByteLevel" { hasByteLevel = true }
                if ptType == "Split",
                   let patternObj = pt["pattern"] as? [String: Any],
                   let regex = patternObj["Regex"] as? String
                {
                    pattern = regex
                }
                if ptType == "Split",
                   let patternObj = pt["pattern"] as? [String: Any],
                   let splitString = patternObj["String"] as? String,
                   splitString == " ",
                   let behavior = pt["behavior"] as? String,
                   behavior == "MergedWithPrevious"
                {
                    prependSentencePieceSpaceMarker = false
                }
            }
            return (hasByteLevel, pattern, prependSentencePieceSpaceMarker)
        }

        return (false, nil, true)
    }

    // MARK: - Encode

    public func encode(_ text: String) -> [Int32] {
        let t = normalizesInputNFC ? text.precomposedStringWithCanonicalMapping : text
        switch style {
        case .sentencePiece: return encodeSentencePiece(t)
        case .byteLevelBPE: return encodeByteLevelBPE(t)
        }
    }

    /// Encode with added/special tokens kept atomic, the way HF tokenizers do (a literal
    /// `<|im_start|>` inside text becomes its single id, not byte-level BPE). NFC-normalizes
    /// first (Qwen2 normalizer), then splits on the added-token literals (longest-match) and
    /// byte-level-BPEs the spans between them. Specials are ASCII so NFC leaves them intact.
    public func encodeWithSpecials(_ text: String) -> [Int32] {
        let norm = text.precomposedStringWithCanonicalMapping
        guard !addedTokenIds.isEmpty else { return encode(norm) }
        let literals = addedTokenIds.sorted { $0.key.count > $1.key.count }  // longest-match wins
        var ids: [Int32] = []
        var buf = ""
        var i = norm.startIndex
        scan: while i < norm.endIndex {
            for (lit, id) in literals where norm[i...].hasPrefix(lit) {
                if !buf.isEmpty { ids.append(contentsOf: encode(buf)); buf.removeAll(keepingCapacity: true) }
                ids.append(Int32(id))
                i = norm.index(i, offsetBy: lit.count)
                continue scan
            }
            buf.append(norm[i])
            i = norm.index(after: i)
        }
        if !buf.isEmpty { ids.append(contentsOf: encode(buf)) }
        return ids
    }

    private func encodeSentencePiece(_ text: String) -> [Int32] {
        let normalizedCore = text.replacingOccurrences(of: " ", with: "▁")
        let normalized = prependSentencePieceSpaceMarker ? "▁" + normalizedCore : normalizedCore
        var tokens: [String] = []
        for char in normalized {
            let s = String(char)
            if vocabId(s) != nil {
                tokens.append(s)
            } else {
                for byte in s.utf8 {
                    tokens.append(String(format: "<0x%02X>", byte))
                }
            }
        }
        return tokensToIds(applyBPEMerges(tokens))
    }

    private func encodeByteLevelBPE(_ text: String) -> [Int32] {
        let pieces = preTokenize(text)
        var allIds: [Int32] = []
        for piece in pieces {
            let tokens = piece.utf8.map { String(Self.byteToUnicode[$0]!) }
            allIds.append(contentsOf: tokensToIds(applyBPEMerges(tokens)))
        }
        return allIds
    }

    private func preTokenize(_ text: String) -> [String] {
        guard let regex = preTokenPattern else { return [text] }
        let nsText = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: nsText.length))
        return matches.map { nsText.substring(with: $0.range) }
    }

    // MARK: - Shared BPE

    private func applyBPEMerges(_ initial: [String]) -> [String] {
        var tokens = initial
        while tokens.count > 1 {
            var bestIdx = -1
            var bestRank = Int.max
            for i in 0..<(tokens.count - 1) {
                let key = "\(tokens[i]) \(tokens[i + 1])"
                if let rank = mergeRank(key), rank < bestRank {
                    bestRank = rank
                    bestIdx = i
                }
            }
            if bestIdx < 0 { break }
            tokens[bestIdx] += tokens[bestIdx + 1]
            tokens.remove(at: bestIdx + 1)
        }
        return tokens
    }

    private func tokensToIds(_ tokens: [String]) -> [Int32] {
        tokens.compactMap { token -> Int32? in
            if let id = vocabId(token) { return Int32(id) }
            if token.utf8.count == 1, let byte = token.utf8.first {
                let byteTok = String(format: "<0x%02X>", byte)
                if let id = vocabId(byteTok) { return Int32(id) }
            }
            return vocabId("<unk>").map { Int32($0) }
        }
    }

    // MARK: - Decode

    public func decode(_ tokenIds: [Int32]) -> String {
        switch style {
        case .sentencePiece: return decodeSentencePiece(tokenIds)
        case .byteLevelBPE: return decodeByteLevelBPE(tokenIds)
        }
    }

    private func decodeSentencePiece(_ tokenIds: [Int32]) -> String {
        var pieces: [String] = []
        var pendingBytes: [UInt8] = []

        func flushBytes() {
            if !pendingBytes.isEmpty {
                if let s = String(bytes: pendingBytes, encoding: .utf8) {
                    pieces.append(s)
                } else {
                    for b in pendingBytes {
                        pieces.append(String(UnicodeScalar(b)))
                    }
                }
                pendingBytes.removeAll()
            }
        }

        for id in tokenIds {
            if specialTokenIds.contains(Int(id)) { continue }
            if let token = tokenString(forId: Int(id)) {
                if token.hasPrefix("<0x") && token.hasSuffix(">") {
                    let hex = String(token.dropFirst(3).dropLast(1))
                    if let byte = UInt8(hex, radix: 16) {
                        pendingBytes.append(byte)
                        continue
                    }
                }
                flushBytes()
                pieces.append(token)
            } else if let content = addedTokenContentById[Int(id)] {
                flushBytes()
                pieces.append(content)
            }
        }
        flushBytes()
        var text = pieces.joined()
        text = text.replacingOccurrences(of: "▁", with: " ")
        if text.hasPrefix(" ") { text = String(text.dropFirst()) }
        return text
    }

    private func decodeByteLevelBPE(_ tokenIds: [Int32]) -> String {
        var bytes: [UInt8] = []
        for id in tokenIds {
            if specialTokenIds.contains(Int(id)) { continue }
            if let token = tokenString(forId: Int(id)) {
                for char in token {
                    if let byte = Self.unicodeToByte[char] {
                        bytes.append(byte)
                    }
                }
            } else if let content = addedTokenContentById[Int(id)] {
                // Non-special added token marked `special: false`; emit
                // content verbatim, no byte-level re-encoding.
                bytes.append(contentsOf: Array(content.utf8))
            }
        }
        return String(bytes: bytes, encoding: .utf8) ?? ""
    }

    private func llguidanceTokenBytes(for token: String) -> [UInt8] {
        switch style {
        case .sentencePiece:
            if token.hasPrefix("<0x") && token.hasSuffix(">") {
                let hex = String(token.dropFirst(3).dropLast(1))
                if let byte = UInt8(hex, radix: 16) {
                    return [byte]
                }
            }
            return Array(token.replacingOccurrences(of: "▁", with: " ").utf8)

        case .byteLevelBPE:
            var bytes: [UInt8] = []
            for char in token {
                if let byte = Self.unicodeToByte[char] {
                    bytes.append(byte)
                } else {
                    bytes.append(contentsOf: String(char).utf8)
                }
            }
            return bytes
        }
    }

    fileprivate func decodeStreamingToken(
        _ tokenId: Int32,
        pendingBytes: inout [UInt8],
        emittedAnyText: inout Bool
    ) -> String {
        guard !specialTokenIds.contains(Int(tokenId)),
              let token = tokenString(forId: Int(tokenId))
        else {
            return ""
        }

        switch style {
        case .sentencePiece:
            if token.hasPrefix("<0x") && token.hasSuffix(">") {
                let hex = String(token.dropFirst(3).dropLast(1))
                if let byte = UInt8(hex, radix: 16) {
                    pendingBytes.append(byte)
                    guard let text = String(bytes: pendingBytes, encoding: .utf8) else {
                        return ""
                    }
                    pendingBytes.removeAll()
                    emittedAnyText = emittedAnyText || !text.isEmpty
                    return text
                }
            }

            var output = flushStreamingBytes(
                pendingBytes: &pendingBytes
            )
            output += token.replacingOccurrences(of: "▁", with: " ")
            if !emittedAnyText, output.hasPrefix(" ") {
                output.removeFirst()
            }
            emittedAnyText = emittedAnyText || !output.isEmpty
            return output

        case .byteLevelBPE:
            for char in token {
                if let byte = Self.unicodeToByte[char] {
                    pendingBytes.append(byte)
                }
            }
            guard let text = String(bytes: pendingBytes, encoding: .utf8) else {
                return ""
            }
            pendingBytes.removeAll()
            emittedAnyText = emittedAnyText || !text.isEmpty
            return text
        }
    }

    private func flushStreamingBytes(
        pendingBytes: inout [UInt8]
    ) -> String {
        guard !pendingBytes.isEmpty else { return "" }
        defer { pendingBytes.removeAll() }
        if let text = String(bytes: pendingBytes, encoding: .utf8) {
            return text
        }
        return String(decoding: pendingBytes, as: UTF8.self)
    }
}

// MARK: - Compiled binary tokenizer (tokenizer.bin)
//
// The compiler serializes the tokenizer as ready-to-query tables so runtime
// load cost is an mmap plus a header walk — independent of vocabulary size.
// String→id lookups probe precomputed open-addressing hash tables in place;
// id→string reads an offset table into a shared string heap.
//
//   "SMTK" | version u32 | fnv1a-64 of body u64 | body
//   body: flags u32 | [bos i32] | [eos i32] | [pattern str]
//         | added tokens (count, [len,bytes,id]) | specials (count, [id])
//         | heap (size u32, bytes)
//         | id table (capacity u32, [heapOff u32, len u32] × capacity)
//         | vocab slots (count u32, [heapOff u32, len u32, id i32] × count)
//         | merge slots (count u32, [heapOff u32, len u32, rank i32] × count)
//         | llg lengths (count u32, u32 × count)
//         | llg bytes (size u32, bytes)
//
// All integers little-endian (the runtime is Apple-Silicon-only). Strings are
// UTF-8 with u32 byte lengths; len == UInt32.max marks an empty slot / hole.
// Slot counts are powers of two; probing is linear. Hash keys are the
// NFC-normalized UTF-8 bytes of the string — Swift Dictionary matches keys by
// Unicode canonical equivalence, so both table build and query normalize to
// keep the JSON and compiled paths behaviorally identical.
//
// The body checksum is verified on load only under SMELT_VERIFY_PACKAGE=1
// (or an explicit flag); the default load is lazy like weights.bin, with
// every table access bounds-checked instead. Structural fields (section
// sizes, slot counts) are always validated up front, and any failure makes
// the caller fall back to tokenizer.json.

/// mmap-backed lookup tables for a compiled tokenizer.
struct MappedTokenizerTables {
    let data: Data
    let heapOffset: Int
    let heapSize: Int
    let idTableOffset: Int
    let idCapacity: Int
    let vocabSlotsOffset: Int
    let vocabSlotCount: Int
    let mergeSlotsOffset: Int
    let mergeSlotCount: Int
    /// Precomputed llguidance vocabulary (token byte lengths + byte blob),
    /// so constrained decoding skips the ~1s per-token conversion pass.
    let llgLengthsOffset: Int
    let llgCount: Int
    let llgBytesOffset: Int
    let llgBytesSize: Int

    /// Bulk-materialize the baked llguidance arrays.
    func llgVocabulary() -> (lengths: [UInt32], bytes: [UInt8])? {
        guard llgCount > 0 else { return nil }
        return data.withUnsafeBytes { raw -> ([UInt32], [UInt8]) in
            let lengths = [UInt32](unsafeUninitializedCapacity: llgCount) {
                buffer, initialized in
                memcpy(buffer.baseAddress!, raw.baseAddress! + llgLengthsOffset, llgCount * 4)
                initialized = llgCount
            }
            let bytes = [UInt8](unsafeUninitializedCapacity: llgBytesSize) {
                buffer, initialized in
                memcpy(buffer.baseAddress!, raw.baseAddress! + llgBytesOffset, llgBytesSize)
                initialized = llgBytesSize
            }
            return (lengths, bytes)
        }
    }

    static func hash<S: Sequence>(_ bytes: S) -> UInt64 where S.Element == UInt8 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in bytes {
            hash = (hash ^ UInt64(byte)) &* 0x0000_0100_0000_01b3
        }
        return hash
    }

    func lookupVocab(_ key: String) -> Int? {
        probe(key, slotsOffset: vocabSlotsOffset, slotCount: vocabSlotCount)
    }

    func lookupMerge(_ key: String) -> Int? {
        probe(key, slotsOffset: mergeSlotsOffset, slotCount: mergeSlotCount)
    }

    private func probe(_ key: String, slotsOffset: Int, slotCount: Int) -> Int? {
        guard slotCount > 0 else { return nil }
        let keyBytes = Array(key.precomposedStringWithCanonicalMapping.utf8)
        let mask = slotCount - 1
        var index = Int(truncatingIfNeeded: Self.hash(keyBytes)) & mask
        return data.withUnsafeBytes { raw -> Int? in
            guard let base = raw.baseAddress else { return nil }
            let heap = base + heapOffset
            return keyBytes.withUnsafeBytes { kb -> Int? in
                for _ in 0..<slotCount {
                    let slot = base + slotsOffset + index * 12
                    let length = slot.loadUnaligned(fromByteOffset: 4, as: UInt32.self)
                    if length == UInt32.max { return nil }
                    if Int(length) == keyBytes.count {
                        let offset = Int(slot.loadUnaligned(as: UInt32.self))
                        if offset + keyBytes.count <= heapSize,
                           memcmp(heap + offset, kb.baseAddress, keyBytes.count) == 0
                        {
                            return Int(slot.loadUnaligned(fromByteOffset: 8, as: Int32.self))
                        }
                    }
                    index = (index + 1) & mask
                }
                return nil
            }
        }
    }

    func tokenString(forId id: Int) -> String? {
        guard id >= 0, id < idCapacity else { return nil }
        return data.withUnsafeBytes { raw -> String? in
            guard let base = raw.baseAddress else { return nil }
            let entry = base + idTableOffset + id * 8
            let length = entry.loadUnaligned(fromByteOffset: 4, as: UInt32.self)
            if length == UInt32.max { return nil }
            let offset = Int(entry.loadUnaligned(as: UInt32.self))
            guard offset + Int(length) <= heapSize else { return nil }
            let bytes = UnsafeRawBufferPointer(
                start: base + heapOffset + offset, count: Int(length)
            )
            return String(decoding: bytes, as: UTF8.self)
        }
    }
}

extension SmeltTokenizer {
    public static let compiledFileName = SmeltTokenizerPackageLayout.compiledFileName

    private static let compiledMagic: [UInt8] = Array("SMTK".utf8)
    private static let compiledVersion: UInt32 = 3

    private struct CompiledFlags {
        static let byteLevelBPE: UInt32 = 1 << 0
        static let prependSPMarker: UInt32 = 1 << 1
        static let normalizesNFC: UInt32 = 1 << 2
        static let hasPattern: UInt32 = 1 << 3
        static let hasBOS: UInt32 = 1 << 4
        static let hasEOS: UInt32 = 1 << 5
    }

    private static func fnv1a64<Bytes: ContiguousBytes>(_ bytes: Bytes) -> UInt64 {
        bytes.withUnsafeBytes { raw in
            var hash: UInt64 = 0xcbf2_9ce4_8422_2325
            for byte in raw.bindMemory(to: UInt8.self) {
                hash = (hash ^ UInt64(byte)) &* 0x0000_0100_0000_01b3
            }
            return hash
        }
    }

    /// Write the compiled binary form. Called at package build time; only
    /// JSON-built (dictionary-backed) tokenizers can be compiled.
    public func writeCompiledTokenizer(to path: String) throws {
        guard case .dictionaries(let vocab, let reverseVocab, let mergeRanks) = lookup
        else { throw SmeltTokenizerError.invalidFormat }

        var heap = Data()
        var heapRefs: [String: (UInt32, UInt32)] = [:]
        func heapRef(_ s: String) -> (UInt32, UInt32) {
            if let existing = heapRefs[s] { return existing }
            let bytes = Array(s.utf8)
            let ref = (UInt32(heap.count), UInt32(bytes.count))
            heap.append(contentsOf: bytes)
            heapRefs[s] = ref
            return ref
        }

        let idCapacity = (reverseVocab.keys.max() ?? -1) + 1
        var idEntries = [(UInt32, UInt32)](
            repeating: (0, UInt32.max), count: idCapacity
        )
        for (id, token) in reverseVocab {
            idEntries[id] = heapRef(token)
        }

        // Open-addressing tables at ≤50% load. Keys are NFC bytes (see the
        // format comment); entries sorted by value for deterministic output.
        func buildSlots(_ pairs: [(key: String, value: Int)]) -> [(UInt32, UInt32, Int32)] {
            guard !pairs.isEmpty else { return [] }
            var slotCount = 4
            while slotCount < pairs.count * 2 { slotCount <<= 1 }
            var slots = [(UInt32, UInt32, Int32)](
                repeating: (0, UInt32.max, 0), count: slotCount
            )
            let mask = slotCount - 1
            for (key, value) in pairs.sorted(by: { $0.value < $1.value }) {
                let nfc = key.precomposedStringWithCanonicalMapping
                let (offset, length) = heapRef(nfc)
                var index = Int(
                    truncatingIfNeeded: MappedTokenizerTables.hash(nfc.utf8)
                ) & mask
                while slots[index].1 != UInt32.max { index = (index + 1) & mask }
                slots[index] = (offset, length, Int32(value))
            }
            return slots
        }
        let vocabSlots = buildSlots(vocab.map { ($0.key, $0.value) })
        let mergeSlots = buildSlots(mergeRanks.map { ($0.key, $0.value) })

        var body = Data()
        func putU32(_ v: UInt32) {
            withUnsafeBytes(of: v.littleEndian) { body.append(contentsOf: $0) }
        }
        func putI32(_ v: Int) { putU32(UInt32(bitPattern: Int32(v))) }
        func putString(_ s: String) {
            let bytes = Array(s.utf8)
            putU32(UInt32(bytes.count))
            body.append(contentsOf: bytes)
        }

        var flags: UInt32 = 0
        if style == .byteLevelBPE { flags |= CompiledFlags.byteLevelBPE }
        if prependSentencePieceSpaceMarker { flags |= CompiledFlags.prependSPMarker }
        if normalizesInputNFC { flags |= CompiledFlags.normalizesNFC }
        if preTokenPattern != nil { flags |= CompiledFlags.hasPattern }
        if bosTokenId != nil { flags |= CompiledFlags.hasBOS }
        if eosTokenId != nil { flags |= CompiledFlags.hasEOS }
        putU32(flags)
        if let bosTokenId { putI32(bosTokenId) }
        if let eosTokenId { putI32(eosTokenId) }
        if let pattern = preTokenPattern?.pattern { putString(pattern) }

        let sortedAdded = addedTokenIds.sorted { $0.value < $1.value }
        putU32(UInt32(sortedAdded.count))
        for (content, id) in sortedAdded {
            putString(content)
            putI32(id)
        }

        let sortedSpecials = specialTokenIds.sorted()
        putU32(UInt32(sortedSpecials.count))
        for id in sortedSpecials { putI32(id) }

        putU32(UInt32(heap.count))
        body.append(heap)

        putU32(UInt32(idCapacity))
        for (offset, length) in idEntries {
            putU32(offset)
            putU32(length)
        }

        for slots in [vocabSlots, mergeSlots] {
            putU32(UInt32(slots.count))
            for (offset, length, value) in slots {
                putU32(offset)
                putU32(length)
                putU32(UInt32(bitPattern: value))
            }
        }

        // llguidance vocabulary (token byte lengths + concatenated bytes),
        // precomputed so constrained decoding skips the per-token conversion
        // pass at load. eosTokens stay a runtime parameter.
        let llg = llguidanceVocabulary()
        putU32(UInt32(llg.tokenLengths.count))
        for length in llg.tokenLengths { putU32(length) }
        putU32(UInt32(llg.tokenBytes.count))
        body.append(contentsOf: llg.tokenBytes)

        var out = Data()
        out.append(contentsOf: Self.compiledMagic)
        withUnsafeBytes(of: Self.compiledVersion.littleEndian) {
            out.append(contentsOf: $0)
        }
        withUnsafeBytes(of: Self.fnv1a64(body).littleEndian) {
            out.append(contentsOf: $0)
        }
        out.append(body)
        try out.write(to: URL(fileURLWithPath: path), options: .atomic)
    }

    /// Load from a compiled tokenizer.bin. The file is memory-mapped; table
    /// payloads are not read until queried.
    public init(
        compiledPath: String,
        verifyChecksum: Bool =
            ProcessInfo.processInfo.environment["SMELT_VERIFY_PACKAGE"] == "1"
    ) throws {
        let data = try Data(
            contentsOf: URL(fileURLWithPath: compiledPath),
            options: .mappedIfSafe
        )
        let headerSize = 4 + 4 + 8
        guard data.count >= headerSize,
              Array(data.prefix(4)) == Self.compiledMagic
        else { throw SmeltTokenizerError.invalidFormat }

        func headerU32(at offset: Int) -> UInt32 {
            var v: UInt32 = 0
            withUnsafeMutableBytes(of: &v) {
                $0.copyBytes(from: data.subdata(in: offset..<offset + 4))
            }
            return UInt32(littleEndian: v)
        }
        guard headerU32(at: 4) == Self.compiledVersion else {
            throw SmeltTokenizerError.invalidFormat
        }
        if verifyChecksum {
            var storedChecksum: UInt64 = 0
            withUnsafeMutableBytes(of: &storedChecksum) {
                $0.copyBytes(from: data.subdata(in: 8..<16))
            }
            let body = data.subdata(in: headerSize..<data.count)
            guard UInt64(littleEndian: storedChecksum) == Self.fnv1a64(body) else {
                throw SmeltTokenizerError.invalidFormat
            }
        }

        var cursor = headerSize
        func readU32() throws -> UInt32 {
            guard cursor + 4 <= data.count else {
                throw SmeltTokenizerError.invalidFormat
            }
            let v = headerU32(at: cursor)
            cursor += 4
            return v
        }
        func readI32() throws -> Int { Int(Int32(bitPattern: try readU32())) }
        func readString() throws -> String {
            let length = Int(try readU32())
            guard cursor + length <= data.count else {
                throw SmeltTokenizerError.invalidFormat
            }
            let s = String(
                decoding: data.subdata(in: cursor..<cursor + length),
                as: UTF8.self
            )
            cursor += length
            return s
        }
        // Validates a table section: reads its element count, bounds-checks
        // the payload, returns (payloadOffset, count) and skips past it.
        func skipTable(elementSize: Int, requirePowerOfTwo: Bool) throws -> (Int, Int) {
            let count = Int(try readU32())
            guard count <= (1 << 28),
                  !requirePowerOfTwo || count == 0 || count & (count - 1) == 0,
                  cursor + count * elementSize <= data.count
            else { throw SmeltTokenizerError.invalidFormat }
            let offset = cursor
            cursor += count * elementSize
            return (offset, count)
        }

        let flags = try readU32()
        self.style = flags & CompiledFlags.byteLevelBPE != 0
            ? .byteLevelBPE : .sentencePiece
        self.prependSentencePieceSpaceMarker =
            flags & CompiledFlags.prependSPMarker != 0
        self.normalizesInputNFC = flags & CompiledFlags.normalizesNFC != 0
        self.bosTokenId = flags & CompiledFlags.hasBOS != 0 ? try readI32() : nil
        self.eosTokenId = flags & CompiledFlags.hasEOS != 0 ? try readI32() : nil
        if flags & CompiledFlags.hasPattern != 0 {
            self.preTokenPattern = try? NSRegularExpression(
                pattern: try readString()
            )
        } else {
            self.preTokenPattern = nil
        }

        let addedCount = Int(try readU32())
        var addedIds: [String: Int] = [:]
        addedIds.reserveCapacity(addedCount)
        for _ in 0..<addedCount {
            let content = try readString()
            addedIds[content] = try readI32()
        }
        self.addedTokenIds = addedIds
        self.addedTokenContentById = Dictionary(
            uniqueKeysWithValues: addedIds.map { ($1, $0) }
        )

        let specialCount = Int(try readU32())
        var specials: Set<Int> = []
        specials.reserveCapacity(specialCount)
        for _ in 0..<specialCount { specials.insert(try readI32()) }
        self.specialTokenIds = specials

        let (heapOffset, heapSize) = try skipTable(
            elementSize: 1, requirePowerOfTwo: false
        )
        let (idTableOffset, idCapacity) = try skipTable(
            elementSize: 8, requirePowerOfTwo: false
        )
        let (vocabSlotsOffset, vocabSlotCount) = try skipTable(
            elementSize: 12, requirePowerOfTwo: true
        )
        let (mergeSlotsOffset, mergeSlotCount) = try skipTable(
            elementSize: 12, requirePowerOfTwo: true
        )
        let (llgLengthsOffset, llgCount) = try skipTable(
            elementSize: 4, requirePowerOfTwo: false
        )
        let (llgBytesOffset, llgBytesSize) = try skipTable(
            elementSize: 1, requirePowerOfTwo: false
        )
        guard cursor == data.count else {
            throw SmeltTokenizerError.invalidFormat
        }

        self.lookup = .mapped(
            MappedTokenizerTables(
                data: data,
                heapOffset: heapOffset,
                heapSize: heapSize,
                idTableOffset: idTableOffset,
                idCapacity: idCapacity,
                vocabSlotsOffset: vocabSlotsOffset,
                vocabSlotCount: vocabSlotCount,
                mergeSlotsOffset: mergeSlotsOffset,
                mergeSlotCount: mergeSlotCount,
                llgLengthsOffset: llgLengthsOffset,
                llgCount: llgCount,
                llgBytesOffset: llgBytesOffset,
                llgBytesSize: llgBytesSize
            )
        )
    }
}

public struct SmeltStreamingTokenDecoder: Sendable {
    private var pendingBytes: [UInt8] = []
    private var emittedAnyText: Bool

    public init(continuingExistingText: Bool = false) {
        // SentencePiece decode strips the FIRST leading space because
        // assistant messages don't want " Hello" → " Hello". For a raw
        // /v1/completions continuation that mid-line ends in
        // indentation, the first generated token IS a space and must
        // be preserved verbatim — caller passes true.
        self.emittedAnyText = continuingExistingText
    }

    public mutating func decode(
        tokenId: Int32,
        tokenizer: SmeltTokenizer
    ) -> String {
        tokenizer.decodeStreamingToken(
            tokenId,
            pendingBytes: &pendingBytes,
            emittedAnyText: &emittedAnyText
        )
    }
}

private struct TokenizerJSONRoot: Decodable {
    let model: TokenizerJSONModel
}

private struct TokenizerJSONModel: Decodable {
    let vocab: TokenizerJSONVocab
}

private struct TokenizerJSONVocab: Decodable {
    let pairs: [(String, Int)]

    private struct VocabKey: CodingKey {
        var stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: VocabKey.self)
        var out: [(String, Int)] = []
        out.reserveCapacity(container.allKeys.count)
        for key in container.allKeys {
            let value = try container.decode(Int.self, forKey: key)
            out.append((key.stringValue, value))
        }
        self.pairs = out
    }
}

public enum SmeltTokenizerError: Error, CustomStringConvertible {
    case invalidFormat
    case fileNotFound(String)

    public var description: String {
        switch self {
        case .invalidFormat:
            return "Invalid tokenizer.json format"
        case let .fileNotFound(path):
            return "Tokenizer not found at \(path)"
        }
    }
}
