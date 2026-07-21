import CLLGuidance
import Foundation

public enum SmeltLLGuidanceError: Error, CustomStringConvertible {
    case tokenizerInitFailed(String)
    case matcherInitFailed(String)
    case matcherError(String)
    case invalidToken(Int32)

    public var description: String {
        switch self {
        case .tokenizerInitFailed(let message):
            return "llguidance tokenizer init failed: \(message)"
        case .matcherInitFailed(let message):
            return "llguidance matcher init failed: \(message)"
        case .matcherError(let message):
            return "llguidance matcher error: \(message)"
        case .invalidToken(let token):
            return "llguidance token id must be non-negative, got \(token)"
        }
    }
}

private final class SmeltLLGuidanceTokenizationBox: @unchecked Sendable {
    private let tokenizer: SmeltTokenizer

    init(tokenizer: SmeltTokenizer) {
        self.tokenizer = tokenizer
    }

    func tokenize(_ bytes: UnsafeBufferPointer<UInt8>) -> [UInt32] {
        let text = String(decoding: bytes, as: UTF8.self)
        // Grammar literals pass through the same tokenizer semantics as the
        // official transcript renderer. Without the atomic-added-token path,
        // a native control literal such as `<tool_call>` compiles to ordinary
        // BPE pieces even though the vocabulary contains its trained single
        // token; generation and historical replay then have different token
        // identities despite identical decoded text.
        return tokenizer.encodeWithSpecials(text).compactMap { token in
            token >= 0 ? UInt32(token) : nil
        }
    }
}

private let agentLLGuidanceTokenizeFn: LlgTokenizeFn = {
    userData,
    bytesPtr,
    byteCount,
    outputTokens,
    outputTokenCapacity in
    guard let userData else { return 0 }
    let box = Unmanaged<SmeltLLGuidanceTokenizationBox>
        .fromOpaque(userData)
        .takeUnretainedValue()
    let bytes = UnsafeBufferPointer(
        start: bytesPtr,
        count: bytesPtr == nil ? 0 : byteCount
    )
    let tokens = box.tokenize(bytes)
    if let outputTokens {
        let writable = min(tokens.count, outputTokenCapacity)
        for index in 0..<writable {
            outputTokens[index] = tokens[index]
        }
    }
    return tokens.count
}

public final class SmeltLLGuidanceTokenizer: @unchecked Sendable {
    private let box: SmeltLLGuidanceTokenizationBox
    private let handle: OpaquePointer
    private let storedVocabSize: Int
    private let storedEOSTokens: [UInt32]

    /// `buildSlicer: true` enables llguidance's slicer optimization
    /// (precomputed vocabulary partitions; ~20x cheaper per-step masks at
    /// ~0.65s build cost on a 250k vocab). Preparation uses it so the built
    /// slicer ships in `serializedTrie()`; runtime callers leave it off —
    /// `SMELT_LLG_SLICER=1` forces it on for A/B runs.
    public init(
        tokenizer: SmeltTokenizer,
        eosTokens: [Int32] = [],
        buildSlicer: Bool = false
    ) throws {
        let vocabStart = CFAbsoluteTimeGetCurrent()
        let builtVocabulary = tokenizer.llguidanceVocabulary(eosTokens: eosTokens)
        if ProcessInfo.processInfo.environment["SMELT_STARTUP_TRACE"] == "1" {
            let ms = (CFAbsoluteTimeGetCurrent() - vocabStart) * 1000
            fputs(
                "startup: \(String(format: "%+7.1fms", ms))  llg vocabulary build\n",
                stderr
            )
        }
        self.box = SmeltLLGuidanceTokenizationBox(tokenizer: tokenizer)
        self.storedVocabSize = builtVocabulary.vocabSize
        self.storedEOSTokens = builtVocabulary.eosTokens

        guard builtVocabulary.vocabSize > 0 else {
            throw SmeltLLGuidanceError.tokenizerInitFailed("empty vocabulary")
        }

        var initV2 = LlgTokenizerInitV2()
        initV2.struct_size = MemoryLayout<LlgTokenizerInitV2>.size
        initV2.vocab_size = UInt32(builtVocabulary.vocabSize)
        initV2.tok_eos = builtVocabulary.eosTokens[0]
        initV2.tokenize_assumes_string = true
        initV2.tokenize_fn = agentLLGuidanceTokenizeFn
        initV2.tokenize_user_data = UnsafeRawPointer(Unmanaged.passUnretained(box).toOpaque())

        let extraEOS = Array(builtVocabulary.eosTokens.dropFirst())
        var errorBuffer = [CChar](repeating: 0, count: 1024)
        // Disable the slicer optimization (empty argv-style array) unless
        // requested: its default partitions cost ~0.65s to precompute over
        // a 250k vocab at tokenizer build, which dominates constrained-
        // decoding cold start. Package preparation requests it (and serializes the built
        // slicer); SMELT_LLG_SLICER=1 forces it for runtime A/B runs.
        let useSlicer = buildSlicer
            || ProcessInfo.processInfo.environment["SMELT_LLG_SLICER"] == "1"
        let emptySlices: [UnsafePointer<CChar>?] = [nil]
        let created = builtVocabulary.tokenLengths.withUnsafeBufferPointer { lengthsPtr in
            builtVocabulary.tokenBytes.withUnsafeBufferPointer { bytesPtr in
                extraEOS.withUnsafeBufferPointer { extraEOSPtr in
                    emptySlices.withUnsafeBufferPointer { slicesPtr in
                        initV2.token_lens = lengthsPtr.baseAddress
                        initV2.token_bytes = bytesPtr.baseAddress
                        initV2.tok_eos_extra = extraEOSPtr.baseAddress
                        initV2.tok_eos_extra_count = UInt32(extraEOS.count)
                        if !useSlicer {
                            initV2.slices = slicesPtr.baseAddress
                        }
                        return errorBuffer.withUnsafeMutableBufferPointer { errorPtr in
                            llg_new_tokenizer_v2(
                                &initV2,
                                errorPtr.baseAddress,
                                errorPtr.count
                            )
                        }
                    }
                }
            }
        }

        guard let created else {
            throw SmeltLLGuidanceError.tokenizerInitFailed(
                Self.cString(errorBuffer)
            )
        }
        self.handle = created
    }

    /// Reconstruct a tokenizer from trie bytes captured by
    /// `serializedTrie()` (typically `compiled_grammar.trie` in a package).
    /// Skips the ~0.4s trie build over the vocabulary; the bytes already
    /// carry the vocabulary and EOS tokens. Corrupt bytes throw — callers
    /// fall back to the from-vocabulary init.
    public init(tokenizer: SmeltTokenizer, serializedTrie: Data) throws {
        let header = try Self.parseTrieHeader(serializedTrie)
        self.box = SmeltLLGuidanceTokenizationBox(tokenizer: tokenizer)
        self.storedVocabSize = header.vocabSize
        self.storedEOSTokens = header.eosTokens

        var initV2 = LlgTokenizerInitV2()
        initV2.struct_size = MemoryLayout<LlgTokenizerInitV2>.size
        initV2.tokenize_assumes_string = true
        initV2.tokenize_fn = agentLLGuidanceTokenizeFn
        initV2.tokenize_user_data = UnsafeRawPointer(Unmanaged.passUnretained(box).toOpaque())

        var errorBuffer = [CChar](repeating: 0, count: 1024)
        // Same slicer policy as the from-vocabulary init above.
        let useSlicer = ProcessInfo.processInfo.environment["SMELT_LLG_SLICER"] == "1"
        let emptySlices: [UnsafePointer<CChar>?] = [nil]
        let created = serializedTrie.withUnsafeBytes { (trieBytes: UnsafeRawBufferPointer) in
            emptySlices.withUnsafeBufferPointer { slicesPtr in
                if !useSlicer {
                    initV2.slices = slicesPtr.baseAddress
                }
                return errorBuffer.withUnsafeMutableBufferPointer { errorPtr in
                    llg_new_tokenizer_from_bytes(
                        trieBytes.bindMemory(to: UInt8.self).baseAddress,
                        trieBytes.count,
                        &initV2,
                        errorPtr.baseAddress,
                        errorPtr.count
                    )
                }
            }
        }

        guard let created else {
            throw SmeltLLGuidanceError.tokenizerInitFailed(
                Self.cString(errorBuffer)
            )
        }
        self.handle = created
    }

    /// Serialize the llguidance token trie — plus the built slicer when
    /// this tokenizer was created with `buildSlicer: true` — for
    /// `init(tokenizer:serializedTrie:)`. Native-endian flat arrays — a
    /// same-architecture cache, not an interchange format.
    public func serializedTrie() throws -> Data {
        var size = 0
        guard let bytes = llg_tokenizer_to_bytes(handle, &size) else {
            throw SmeltLLGuidanceError.tokenizerInitFailed(
                "llg_tokenizer_to_bytes returned null"
            )
        }
        defer { llg_free_bytes(bytes, size) }
        return Data(bytes: bytes, count: size)
    }

    /// Read vocab size and EOS tokens out of the serialized trie header
    /// (layout owned by tools/llguidance-serialize.patch). Accepts both
    /// the v1 raw trie ("TKTR" magic, 13 u32 header words, then token
    /// offsets, nodes, EOS tokens, token data) and the "LLGC" container
    /// (u32 magic, u32 version, u64 trie length, trie, u64 slicer length,
    /// slicer) whose trie section starts at byte 16.
    private static func parseTrieHeader(
        _ data: Data
    ) throws -> (vocabSize: Int, eosTokens: [UInt32]) {
        func word(at byteOffset: Int) throws -> UInt32 {
            guard byteOffset >= 0, byteOffset + 4 <= data.count else {
                throw SmeltLLGuidanceError.tokenizerInitFailed(
                    "serialized trie header out of range"
                )
            }
            var value: UInt32 = 0
            withUnsafeMutableBytes(of: &value) { dest in
                _ = data.copyBytes(
                    to: dest, from: byteOffset..<(byteOffset + 4)
                )
            }
            return value
        }
        // "LLGC" container: u32 magic, u32 version, u64 trie length, then
        // the raw trie section. v1 files are the raw trie at offset 0.
        var base = 0
        if try word(at: 0) == 0x434c_4c47 {
            guard try word(at: 4) == 1 else {
                throw SmeltLLGuidanceError.tokenizerInitFailed(
                    "unsupported serialized tokenizer container version"
                )
            }
            base = 16
        }
        guard try word(at: base) == 0x544b_5452,  // "TKTR"
              try word(at: base + 4) == 1
        else {
            throw SmeltLLGuidanceError.tokenizerInitFailed(
                "unrecognized serialized trie header"
            )
        }
        let vocabSize = Int(try word(at: base + 8))
        let numOffsets = Int(try word(at: base + 36))
        let numNodes = Int(try word(at: base + 44))
        let numEOS = Int(try word(at: base + 48))
        guard numEOS > 0, numEOS <= vocabSize else {
            throw SmeltLLGuidanceError.tokenizerInitFailed(
                "serialized trie header out of range"
            )
        }
        let eosStart = base + 52 + numOffsets * 8 + numNodes * 8
        var eosTokens: [UInt32] = []
        eosTokens.reserveCapacity(numEOS)
        for index in 0..<numEOS {
            eosTokens.append(try word(at: eosStart + index * 4))
        }
        return (vocabSize, eosTokens)
    }

    deinit {
        llg_free_tokenizer(handle)
    }

    public var vocabSize: Int {
        storedVocabSize
    }

    public var eosTokens: [UInt32] {
        storedEOSTokens
    }

    fileprivate var opaqueHandle: OpaquePointer {
        handle
    }

    private static func cString(_ buffer: [CChar]) -> String {
        let end = buffer.firstIndex(of: 0) ?? buffer.count
        return String(decoding: buffer[..<end].map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
}

public final class SmeltLLGuidanceMatcher: @unchecked Sendable {
    private let tokenizer: SmeltLLGuidanceTokenizer
    private let handle: OpaquePointer

    public convenience init(
        tokenizer: SmeltLLGuidanceTokenizer,
        jsonSchema: String
    ) throws {
        try self.init(
            tokenizer: tokenizer,
            constraintType: "json_schema",
            data: jsonSchema
        )
    }

    public convenience init(
        tokenizer: SmeltLLGuidanceTokenizer,
        lark: String
    ) throws {
        try self.init(
            tokenizer: tokenizer,
            constraintType: "lark",
            data: lark
        )
    }

    private init(
        tokenizer: SmeltLLGuidanceTokenizer,
        constraintType: String,
        data: String
    ) throws {
        self.tokenizer = tokenizer

        var initConfig = LlgConstraintInit()
        llg_constraint_init_set_defaults(&initConfig, tokenizer.opaqueHandle)
        initConfig.log_buffer_level = 0
        initConfig.log_stderr_level = 0

        let created = data.withCString { dataPtr in
            constraintType.withCString { typePtr in
                llg_new_matcher(&initConfig, typePtr, dataPtr)
            }
        }
        guard let created else {
            throw SmeltLLGuidanceError.matcherInitFailed("llg_new_matcher returned null")
        }
        self.handle = created

        if llg_matcher_is_error(handle) {
            throw SmeltLLGuidanceError.matcherInitFailed(matcherErrorMessage())
        }
    }

    /// Adopt an already-owned matcher handle (e.g. from
    /// llg_clone_matcher). The instance takes ownership and frees it
    /// in deinit, so the caller must not free or reuse the handle.
    private init(
        tokenizer: SmeltLLGuidanceTokenizer,
        ownedHandle: OpaquePointer
    ) throws {
        self.tokenizer = tokenizer
        self.handle = ownedHandle
        if llg_matcher_is_error(handle) {
            throw SmeltLLGuidanceError.matcherInitFailed(matcherErrorMessage())
        }
    }

    deinit {
        llg_free_matcher(handle)
    }

    /// Clone this matcher's current state into an independent matcher.
    /// Used to reuse a compiled grammar across requests: keep one
    /// pristine (never-consumed) matcher and hand each request a fresh
    /// copy, since a consumed matcher cannot be rewound to the start.
    public func freshCopy() throws -> SmeltLLGuidanceMatcher {
        guard let cloned = llg_clone_matcher(handle) else {
            throw SmeltLLGuidanceError.matcherInitFailed(
                "llg_clone_matcher returned null"
            )
        }
        return try SmeltLLGuidanceMatcher(
            tokenizer: tokenizer, ownedHandle: cloned
        )
    }

    public func computeMask() throws -> [UInt32] {
        guard llg_matcher_compute_mask(handle) == 0 else {
            throw SmeltLLGuidanceError.matcherError(matcherErrorMessage())
        }
        let byteCount = llg_matcher_get_mask_byte_size(handle)
        guard let rawMask = llg_matcher_get_mask(handle) else {
            return []
        }
        return Array(
            UnsafeBufferPointer(
                start: rawMask,
                count: byteCount / MemoryLayout<UInt32>.size
            )
        )
    }

    public func consume(tokenIds: [Int32]) throws {
        guard !tokenIds.isEmpty else { return }
        let tokens = try tokenIds.map { token -> UInt32 in
            guard token >= 0 else {
                throw SmeltLLGuidanceError.invalidToken(token)
            }
            return UInt32(token)
        }
        let result = tokens.withUnsafeBufferPointer { ptr in
            llg_matcher_consume_tokens(handle, ptr.baseAddress, ptr.count)
        }
        guard result == 0 else {
            throw SmeltLLGuidanceError.matcherError(matcherErrorMessage())
        }
    }

    public var isAccepting: Bool {
        llg_matcher_is_accepting(handle)
    }

    public var isStopped: Bool {
        llg_matcher_is_stopped(handle)
    }

    public var eosTokens: [UInt32] {
        tokenizer.eosTokens
    }

    public static func tokenIsAllowed(_ tokenId: Int32, in mask: [UInt32]) -> Bool {
        guard tokenId >= 0 else { return false }
        let wordIndex = Int(tokenId) / 32
        guard wordIndex < mask.count else { return false }
        let bit = UInt32(tokenId) % 32
        return ((mask[wordIndex] >> bit) & 1) == 1
    }

    private func matcherErrorMessage() -> String {
        if let error = llg_matcher_get_error(handle) {
            return String(cString: error)
        }
        return "unknown llguidance matcher error"
    }
}

public enum SmeltLLGuidance {
    public static var version: String {
        guard let raw = llg_get_version() else {
            return "unknown"
        }
        return String(cString: raw)
    }
}
