// Qwen3TTSTalkerPrefill — the dual-track talker input-embedding assembly for the
// non-streaming path of Qwen3TTSForConditionalGeneration.generate: VoiceDesign
// (instruct-designed voice) and CustomVoice (named speaker via `speakerId`, a row of
// the codec embedding table). Text tokens enter via text_projection (ResizeMLP) over
// text_embedding rows; codec tags enter via the talker's codec_embedding; the two
// tracks merge by additive sum into one [T, hidden] stream (no cross-attention).
// Explicit language uses the 4-tag "think" prefix; `languageId == nil` (upstream
// language="Auto") uses the 3-tag "nothink" prefix [nothink, think_bos, think_eos].
// Instruct rows are optional (empty `instructIds` = upstream instruct=None).
// Not handled here: streaming mode and voice-clone / ICL prompts.

import Foundation

public enum Qwen3TTSTalkerPrefill {

    /// Config token ids for the prefix tags (talker_config + top-level tts_* ids).
    /// `languageId == nil` selects the nothink prefix (requires `codecNothink`);
    /// `speakerId` is a codec-vocab id from talker_config.spk_id (CustomVoice only).
    public struct Ids {
        public var ttsBos, ttsEos, ttsPad: Int
        public var codecThink, codecThinkBos, codecThinkEos, codecPad, codecBos: Int
        public var languageId: Int?
        public var codecNothink: Int?
        public var speakerId: Int?
        public init(ttsBos: Int, ttsEos: Int, ttsPad: Int,
                    codecThink: Int, codecThinkBos: Int, codecThinkEos: Int,
                    codecPad: Int, codecBos: Int, languageId: Int?,
                    codecNothink: Int? = nil, speakerId: Int? = nil) {
            self.ttsBos = ttsBos; self.ttsEos = ttsEos; self.ttsPad = ttsPad
            self.codecThink = codecThink; self.codecThinkBos = codecThinkBos
            self.codecThinkEos = codecThinkEos; self.codecPad = codecPad; self.codecBos = codecBos
            self.languageId = languageId
            self.codecNothink = codecNothink; self.speakerId = speakerId
        }
    }

    /// One assembled prefill row: the text_embedding id to project (`textId`) and, for dual-track
    /// rows, the codec_embedding id added on top (`codecId`; nil for the instruct/role rows, which
    /// are projection-only). This is the row-aligned form of the layout below — one textId per output
    /// row, so a batched projection comes out already in output order.
    public struct PrefillRow: Equatable {
        public let textId: Int
        public let codecId: Int?
        public init(textId: Int, codecId: Int?) { self.textId = textId; self.codecId = codecId }
    }

    /// The ordered (textId, codecId?) layout of the dual-track prefill — the single structural spec
    /// `voiceDesignPrefill` (host float-assembly) and the GPU front-end (`frontEndPrefillHiddenA`)
    /// both follow. Pure index/branch logic (no embeddings). `voiceDesignPrefill`'s assembly below
    /// must produce this same row order; `qwen3TTSPrefillLayoutDrivenAssemblyMatchesVoiceDesignPrefill`
    /// (fast CI) pins the two together so a drift in either is loud.
    public static func layout(instructIds: [Int], inputIds: [Int], ids: Ids) -> [PrefillRow] {
        precondition(ids.languageId != nil || ids.codecNothink != nil,
                     "languageId nil (Auto) requires codecNothink")
        precondition(inputIds.count >= 8, "inputIds too short for the role+trailing wrapper")
        let textLen = inputIds.count - 8
        var rows: [PrefillRow] = []
        rows.reserveCapacity(instructIds.count + 3 + 6 + textLen + 2)
        for id in instructIds { rows.append(.init(textId: id, codecId: nil)) }          // instruct: projection-only
        for r in 0..<3 { rows.append(.init(textId: inputIds[r], codecId: nil)) }         // role: projection-only
        if let lang = ids.languageId {
            rows.append(.init(textId: ids.ttsPad, codecId: ids.codecThink))
            rows.append(.init(textId: ids.ttsPad, codecId: ids.codecThinkBos))
            rows.append(.init(textId: ids.ttsPad, codecId: lang))
            rows.append(.init(textId: ids.ttsPad, codecId: ids.codecThinkEos))
        } else {
            rows.append(.init(textId: ids.ttsPad, codecId: ids.codecNothink!))
            rows.append(.init(textId: ids.ttsPad, codecId: ids.codecThinkBos))
            rows.append(.init(textId: ids.ttsPad, codecId: ids.codecThinkEos))
        }
        if let spk = ids.speakerId { rows.append(.init(textId: ids.ttsPad, codecId: spk)) }
        rows.append(.init(textId: ids.ttsBos, codecId: ids.codecPad))
        for r in 0..<textLen { rows.append(.init(textId: inputIds[3 + r], codecId: ids.codecPad)) }
        rows.append(.init(textId: ids.ttsEos, codecId: ids.codecPad))
        rows.append(.init(textId: ids.ttsPad, codecId: ids.codecBos))
        return rows
    }

    /// Assemble the prefill inputs_embeds [T, dim]. `inputIds` is the wrapped assistant
    /// text (role[0..2] + text + trailing[-5..]); `instructIds` the wrapped user
    /// instruction ([] = no instruct). Layout:
    ///   instruct rows | role rows
    ///   | think:   [tts_pad+codec(think/think_bos/lang/think_eos)]      (explicit language)
    ///     nothink: [tts_pad+codec(nothink/think_bos/think_eos)]         (languageId nil)
    ///   | tts_pad+codec(speakerId)  (CustomVoice only)
    ///   | tts_bos+codec(pad) | textLen×(text + codec(pad)) | tts_eos+codec(pad) | tts_pad+codec(bos)
    public static func voiceDesignPrefill(
        instructIds: [Int], inputIds: [Int], ids: Ids,
        textEmbedding: [Float], fc1W: [Float], fc1B: [Float], fc2W: [Float], fc2B: [Float],
        codecEmbedding: [Float], dim: Int = 2048, projInter: Int? = nil, hidden: Int? = nil
    ) -> (embeds: [Float], frames: Int) {
        precondition(dim > 0, "dim must be positive")
        let h = hidden ?? dim
        precondition(textEmbedding.count % dim == 0 && codecEmbedding.count % h == 0,
                     "embedding tables must be a whole number of rows (text \(dim)-wide, codec \(h)-wide)")
        let textRows = textEmbedding.count / dim, codecRows = codecEmbedding.count / h
        return voiceDesignPrefill(
            instructIds: instructIds, inputIds: inputIds, ids: ids,
            textRow: { id in
                precondition(id >= 0 && id < textRows, "text token id \(id) out of [0,\(textRows))")
                return Array(textEmbedding[(id * dim)..<((id + 1) * dim)])
            },
            codecRow: { id in
                precondition(id >= 0 && id < codecRows, "codec id \(id) out of [0,\(codecRows))")
                return Array(codecEmbedding[(id * h)..<((id + 1) * h)])
            },
            fc1W: fc1W, fc1B: fc1B, fc2W: fc2W, fc2B: fc2B, dim: dim, projInter: projInter, hidden: hidden)
    }

    /// Row-accessor core: `textRow(id)` returns the raw `dim`-wide text-embedding row and
    /// `codecRow(id)` the `hidden`-wide codec-embedding row (`hidden`/`projInter` default to
    /// `dim` — they coincide on the 1.7B; the 0.6B has textEmbedDim 2048 over hidden 1024).
    /// Lets the packaged driver gather only the ~40 referenced text_embedding rows straight
    /// from its mapped buffer rather than copying the full 1.2 GB table; the [Float]-table
    /// overload above is a thin bounds-checked wrapper.
    public static func voiceDesignPrefill(
        instructIds: [Int], inputIds: [Int], ids: Ids,
        textRow: (Int) -> [Float], codecRow: (Int) -> [Float],
        fc1W: [Float], fc1B: [Float], fc2W: [Float], fc2B: [Float],
        dim: Int = 2048, projInter: Int? = nil, hidden: Int? = nil
    ) -> (embeds: [Float], frames: Int) {
        precondition(dim > 0, "dim must be positive")
        let h = hidden ?? dim
        let inter = projInter ?? dim
        precondition(ids.languageId != nil || ids.codecNothink != nil,
                     "languageId nil (Auto) requires codecNothink")
        // inputIds is the wrapped assistant stream: 3 role tokens + text + 5 trailing tokens.
        precondition(inputIds.count >= 8, "inputIds too short for the role+trailing wrapper")

        func projText(_ tokenIds: [Int]) -> [Float] {
            var gathered = [Float](repeating: 0, count: tokenIds.count * dim)
            for (r, id) in tokenIds.enumerated() {
                let rowVals = textRow(id)
                precondition(rowVals.count == dim, "textRow(\(id)) returned \(rowVals.count) != dim \(dim)")
                for d in 0..<dim { gathered[r * dim + d] = rowVals[d] }
            }
            return Qwen3TTSTalker.textProjection(gathered, rows: tokenIds.count,
                                                 fc1W: fc1W, fc1B: fc1B, fc2W: fc2W, fc2B: fc2B,
                                                 inDim: dim, interDim: inter, outDim: h)
        }
        func crow(_ id: Int) -> [Float] {
            let v = codecRow(id)
            precondition(v.count == h, "codecRow(\(id)) returned \(v.count) != hidden \(h)")
            return v
        }
        func row(_ buf: [Float], _ r: Int) -> [Float] { Array(buf[(r * h)..<((r + 1) * h)]) }
        func add(_ a: [Float], _ b: [Float]) -> [Float] { zip(a, b).map(+) }

        let textLen = inputIds.count - 8           // drop 3 role + 5 trailing
        let instructProj = instructIds.isEmpty ? [] : projText(instructIds)
        let roleProj = projText(Array(inputIds[0..<3]))
        let textProj = projText(Array(inputIds[3..<(3 + textLen)]))
        let special = projText([ids.ttsBos, ids.ttsEos, ids.ttsPad])
        let ttsBosV = row(special, 0), ttsEosV = row(special, 1), ttsPadV = row(special, 2)

        var out: [Float] = []
        out.reserveCapacity((instructIds.count + 3 + 6 + textLen + 2) * h)
        for r in 0..<instructIds.count { out += row(instructProj, r) }
        for r in 0..<3 { out += row(roleProj, r) }
        if let lang = ids.languageId {
            out += add(ttsPadV, crow(ids.codecThink))
            out += add(ttsPadV, crow(ids.codecThinkBos))
            out += add(ttsPadV, crow(lang))
            out += add(ttsPadV, crow(ids.codecThinkEos))
        } else {
            out += add(ttsPadV, crow(ids.codecNothink!))
            out += add(ttsPadV, crow(ids.codecThinkBos))
            out += add(ttsPadV, crow(ids.codecThinkEos))
        }
        if let spk = ids.speakerId { out += add(ttsPadV, crow(spk)) }
        out += add(ttsBosV, crow(ids.codecPad))
        for r in 0..<textLen { out += add(row(textProj, r), crow(ids.codecPad)) }
        out += add(ttsEosV, crow(ids.codecPad))
        out += add(ttsPadV, crow(ids.codecBos))

        return (out, out.count / h)
    }
}
