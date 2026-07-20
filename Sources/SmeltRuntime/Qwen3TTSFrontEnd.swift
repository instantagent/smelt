// Qwen3TTSFrontEnd — the text→inputsEmbeds front-end for the VoiceDesign (instruct-designed
// voice) and CustomVoice (named speaker) paths. Turns raw (text, instruct, language, speaker)
// inputs into the wrapped token id streams the dual-track prefill assembly
// (Qwen3TTSTalkerPrefill.voiceDesignPrefill) consumes. language "Auto" selects the nothink
// branch; named speakers resolve through talker_config.spk_id with the upstream dialect
// remap. Not handled: streaming, voice-clone / ICL.

import Foundation

public enum Qwen3TTSFrontEnd {

    // Chat templates verbatim from qwen_tts/inference/qwen3_tts_model.py (_build_*_text).
    static func instructTemplate(_ instruct: String) -> String {
        "<|im_start|>user\n\(instruct)<|im_end|>\n"
    }
    static func inputTemplate(_ text: String) -> String {
        "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
    }

    /// Tokenize the wrapped instruct + input strings into the two id streams
    /// `voiceDesignPrefill` expects. Special-token-aware so the `<|im_start|>`/`<|im_end|>`
    /// markers (and any added-token literal in the user text) stay atomic, matching HF.
    /// nil/empty instruct ⇒ no instruct rows (upstream instruct=None).
    public static func wrap(text: String, instruct: String?, tokenizer: SmeltTokenizer)
        -> (instructIds: [Int], inputIds: [Int]) {
        let instructIds: [Int]
        if let instruct, !instruct.isEmpty {
            instructIds = tokenizer.encodeWithSpecials(instructTemplate(instruct)).map(Int.init)
        } else {
            instructIds = []
        }
        let inputIds = tokenizer.encodeWithSpecials(inputTemplate(text)).map(Int.init)
        return (instructIds, inputIds)
    }

    /// The prefix special-token ids the dual-track assembly needs, parsed from the checkpoint's
    /// `config.json` (single source of truth — same fields the capture script reads), plus the
    /// per-language codec id map. Lets the front-end build `Qwen3TTSTalkerPrefill.Ids` without
    /// hardcoding any token.
    public struct Config {
        public let ttsBos, ttsEos, ttsPad: Int
        public let codecThink, codecThinkBos, codecThinkEos, codecPad, codecBos: Int
        public let codecNothink: Int?            // required only for language "Auto"
        public let languageIds: [String: Int]    // keys lowercased
        public let speakerIds: [String: Int]     // talker_config.spk_id; empty = no named speakers
        public let speakerDialect: [String: String]  // speaker → dialect language key (spk_is_dialect)

        public static func load(configJSONPath: String) throws -> Config {
            let data = try Data(contentsOf: URL(fileURLWithPath: configJSONPath))
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let talker = root["talker_config"] as? [String: Any],
                  let langs = talker["codec_language_id"] as? [String: Any] else {
                throw Error.invalidConfig
            }
            func top(_ k: String) throws -> Int {
                guard let v = root[k] as? Int else { throw Error.missingConfigKey(k) }
                return v
            }
            func tk(_ k: String) throws -> Int {
                guard let v = talker[k] as? Int else { throw Error.missingConfigKey("talker_config.\(k)") }
                return v
            }
            var langMap: [String: Int] = [:]
            for (k, v) in langs {
                guard let id = v as? Int else { throw Error.invalidConfig }
                langMap[k.lowercased()] = id
            }
            // spk_id: name → codec-vocab id. spk_is_dialect values are bool false OR a
            // dialect-name string that must key into codec_language_id (upstream semantics).
            var spkMap: [String: Int] = [:]
            if let spks = talker["spk_id"] as? [String: Any] {
                for (k, v) in spks {
                    guard let id = v as? Int else { throw Error.invalidConfig }
                    spkMap[k.lowercased()] = id
                }
            }
            var dialectMap: [String: String] = [:]
            if let dial = talker["spk_is_dialect"] as? [String: Any] {
                for (k, v) in dial {
                    if let b = v as? Bool {
                        if b { throw Error.invalidSpeakerDialect(k, "true") }  // upstream never emits bare true
                        continue
                    }
                    // NSNumber bridging: JSON false can also surface as 0 — treat any number as false-y only for 0.
                    if let n = v as? NSNumber, n == 0 { continue }
                    guard let d = v as? String else { throw Error.invalidSpeakerDialect(k, "\(v)") }
                    guard langMap[d.lowercased()] != nil else { throw Error.invalidSpeakerDialect(k, d) }
                    dialectMap[k.lowercased()] = d.lowercased()
                }
            }
            return Config(
                ttsBos: try top("tts_bos_token_id"), ttsEos: try top("tts_eos_token_id"),
                ttsPad: try top("tts_pad_token_id"),
                codecThink: try tk("codec_think_id"), codecThinkBos: try tk("codec_think_bos_id"),
                codecThinkEos: try tk("codec_think_eos_id"), codecPad: try tk("codec_pad_id"),
                codecBos: try tk("codec_bos_id"),
                codecNothink: talker["codec_nothink_id"] as? Int,
                languageIds: langMap, speakerIds: spkMap, speakerDialect: dialectMap)
        }

        /// Build the prefill `Ids` for a (language, speaker) pair, applying upstream semantics:
        /// language "Auto" → nothink branch; a dialect speaker remaps the language id when
        /// language is Chinese or Auto. Throws on unsupported language/speaker so a typo'd
        /// public input fails loudly rather than silently picking a wrong voice.
        public func ids(language: String, speaker: String? = nil) throws -> Qwen3TTSTalkerPrefill.Ids {
            let langKey = language.lowercased()
            var langId: Int?
            if langKey == "auto" {
                langId = nil
            } else if let id = languageIds[langKey] {
                langId = id
            } else {
                throw Error.unknownLanguage(language, Array(languageIds.keys).sorted() + ["auto"])
            }

            var spkId: Int?
            if let speaker, !speaker.isEmpty {
                let spkKey = speaker.lowercased()
                guard let id = speakerIds[spkKey] else {
                    throw Error.unknownSpeaker(speaker, Array(speakerIds.keys).sorted())
                }
                spkId = id
                if langKey == "chinese" || langKey == "auto",
                   let dialect = speakerDialect[spkKey] {
                    langId = languageIds[dialect]  // load() verified the dialect keys langMap
                }
            }

            if langId == nil {
                guard codecNothink != nil else { throw Error.missingConfigKey("talker_config.codec_nothink_id") }
            }
            return Qwen3TTSTalkerPrefill.Ids(
                ttsBos: ttsBos, ttsEos: ttsEos, ttsPad: ttsPad,
                codecThink: codecThink, codecThinkBos: codecThinkBos, codecThinkEos: codecThinkEos,
                codecPad: codecPad, codecBos: codecBos, languageId: langId,
                codecNothink: codecNothink, speakerId: spkId)
        }
    }

    /// `(text, instruct, language, speaker)` → talker prefill `inputsEmbeds`, via the proven
    /// dual-track `voiceDesignPrefill`.
    public static func textToInputsEmbeds(
        text: String, instruct: String?, language: String, speaker: String? = nil,
        tokenizer: SmeltTokenizer, config: Config,
        textEmbedding: [Float], fc1W: [Float], fc1B: [Float], fc2W: [Float], fc2B: [Float],
        codecEmbedding: [Float], dim: Int = 2048, projInter: Int? = nil, hidden: Int? = nil
    ) throws -> (embeds: [Float], frames: Int) {
        let (instructIds, inputIds) = wrap(text: text, instruct: instruct, tokenizer: tokenizer)
        let ids = try config.ids(language: language, speaker: speaker)
        // Delegate to the [Float]-table prefill overload — it owns the bounds-checked row slicing.
        return Qwen3TTSTalkerPrefill.voiceDesignPrefill(
            instructIds: instructIds, inputIds: inputIds, ids: ids,
            textEmbedding: textEmbedding, fc1W: fc1W, fc1B: fc1B, fc2W: fc2W, fc2B: fc2B,
            codecEmbedding: codecEmbedding, dim: dim, projInter: projInter, hidden: hidden)
    }

    /// Row-accessor variant: the packaged driver passes `textRow`/`codecRow` that read straight
    /// from its mapped weight buffers, so the 1.2 GB text_embedding table is never copied whole.
    public static func textToInputsEmbeds(
        text: String, instruct: String?, language: String, speaker: String? = nil,
        tokenizer: SmeltTokenizer, config: Config,
        textRow: (Int) -> [Float], codecRow: (Int) -> [Float],
        fc1W: [Float], fc1B: [Float], fc2W: [Float], fc2B: [Float],
        dim: Int = 2048, projInter: Int? = nil, hidden: Int? = nil
    ) throws -> (embeds: [Float], frames: Int) {
        let (instructIds, inputIds) = wrap(text: text, instruct: instruct, tokenizer: tokenizer)
        let ids = try config.ids(language: language, speaker: speaker)
        return Qwen3TTSTalkerPrefill.voiceDesignPrefill(
            instructIds: instructIds, inputIds: inputIds, ids: ids,
            textRow: textRow, codecRow: codecRow,
            fc1W: fc1W, fc1B: fc1B, fc2W: fc2W, fc2B: fc2B,
            dim: dim, projInter: projInter, hidden: hidden)
    }

    public enum Error: Swift.Error, CustomStringConvertible {
        case invalidConfig
        case missingConfigKey(String)
        case unknownLanguage(String, [String])
        case unknownSpeaker(String, [String])
        case invalidSpeakerDialect(String, String)

        public var description: String {
            switch self {
            case .invalidConfig: return "config.json missing talker_config.codec_language_id"
            case let .missingConfigKey(k): return "config.json missing key \(k)"
            case let .unknownLanguage(l, known): return "unknown language '\(l)'; known: \(known)"
            case let .unknownSpeaker(s, known):
                return known.isEmpty
                    ? "speaker '\(s)' requested but this package has no named speakers (spk_id empty — VoiceDesign/Base variant?)"
                    : "unknown speaker '\(s)'; known: \(known)"
            case let .invalidSpeakerDialect(s, v):
                return "spk_is_dialect['\(s)'] = '\(v)' is neither false nor a known codec_language_id key"
            }
        }
    }
}
