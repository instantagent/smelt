import Foundation
import SmeltRuntime

// Logprob projection shared by the server and Smelt's benchmark adapter.

/// Per-token logprob payload independent of the OpenAI wire shape.
/// Chat and legacy endpoints project this into different responses.
package struct LogprobEntry {
    package let token: Int32
    package let logprob: Double
    package let topAlternatives: [(token: Int32, logprob: Double)]
}

package enum LogprobsCompute {
    /// OpenAI's convention for "this token's probability rounds to
    /// zero": floor the reported logprob at this very negative value
    /// instead of -infinity (which doesn't survive JSON encoding).
    static let minReportableLogprob: Double = -9999.0

    /// Read the runtime's current logits buffer and compute:
    ///   - log P(chosenToken | context) for the chosen sampled token
    ///   - log P(token_i | context) for the top-K alternatives
    ///
    /// Caller must invoke this AFTER the runtime's decode step that
    /// produced `chosenToken` — the logits buffer still holds the
    /// distribution that token was sampled from. The logits are the
    /// runtime's RAW pre-temperature distribution; per OpenAI spec
    /// logprobs are not temperature-adjusted even when the sampler
    /// applies temperature.
    ///
    /// log-softmax is computed with a Double accumulator over fp32
    /// `exp(logit_i - max)` terms — wider precision than the input
    /// fp16 buffer so the long tail doesn't underflow.
    static func compute(
        runtime: SmeltRuntime,
        chosenToken: Int32,
        topK: Int
    ) -> LogprobEntry {
        return computeFromLogits(
            logits: runtime.allLogitsHalf(),
            chosenToken: chosenToken,
            topK: topK
        )
    }

    /// Same math as `compute(runtime:...)` but takes an explicit
    /// `[Float16]` row. Used by the prompt-side logprobs path which
    /// gets logits per prompt position via `prefillAllLogits`
    /// rather than the runtime's current decode-slot buffer.
    package static func computeFromLogits(
        logits: [Float16],
        chosenToken: Int32,
        topK: Int
    ) -> LogprobEntry {
        let vocab = logits.count
        precondition(vocab > 0, "logits buffer is empty")

        var maxLogit: Float = -.infinity
        for i in 0 ..< vocab {
            let v = Float(logits[i])
            if v > maxLogit { maxLogit = v }
        }
        // Only zero out a -inf max (all logits -inf, degenerate).
        // A +inf max means at least one logit is +inf; subtract-max
        // still works there (the +inf token contributes exp(0)=1).
        if maxLogit == -.infinity { maxLogit = 0 }

        var sumExp: Double = 0
        for i in 0 ..< vocab {
            let shifted = Double(Float(logits[i])) - Double(maxLogit)
            sumExp += exp(shifted)
        }
        let logZ = Double(maxLogit) + log(sumExp)

        let chosenIdx = Int(chosenToken)
        let chosenLogit: Float = (chosenIdx >= 0 && chosenIdx < vocab)
            ? Float(logits[chosenIdx])
            : -.infinity
        let chosenLogprob = clampLogprob(Double(chosenLogit) - logZ)

        var topAlternatives: [(Int32, Double)] = []
        if topK > 0 {
            topAlternatives.reserveCapacity(topK)
            let topRaw = topKLogits(logits: logits, k: topK)
            for (id, logit) in topRaw {
                topAlternatives.append((id, clampLogprob(Double(logit) - logZ)))
            }
        }

        return LogprobEntry(
            token: chosenToken,
            logprob: chosenLogprob,
            topAlternatives: topAlternatives
        )
    }

    private static func clampLogprob(_ x: Double) -> Double {
        // +inf would be a math bug (logprob is always ≤ 0 by
        // construction), but if a degenerate kernel ever emits it,
        // clamp to 0 (probability=1) rather than reporting it as
        // -9999 (probability ≈ 0) which inverts the meaning.
        if x.isNaN { return minReportableLogprob }
        if x == .infinity { return 0 }
        return max(x, minReportableLogprob)
    }

    /// Single-pass insertion-shift top-K over a fp16 logits buffer.
    /// Returns `[(token_id, logit)]` of length ≤ k, sorted by logit
    /// descending. Equivalent shape and roughly equivalent wall
    /// time as SmeltRuntime.topKLogits (the Float16→Float conversion
    /// + vocab walk dominate the per-call cost). Kept as a local
    /// helper rather than calling through the runtime because Unit
    /// 5's prompt-side logprobs path needs to top-K over an
    /// arbitrary `[Float16]` row from prefillAllLogits, not just the
    /// runtime's current decode-slot buffer.
    private static func topKLogits(
        logits: [Float16],
        k: Int
    ) -> [(Int32, Float)] {
        precondition(k > 0, "topKLogits requires k > 0")
        var top: [(Int32, Float)] = []
        top.reserveCapacity(k)
        for i in 0 ..< logits.count {
            let v = Float(logits[i])
            if top.count < k {
                insertSorted(&top, (Int32(i), v))
            } else if v > top[top.count - 1].1 {
                top.removeLast()
                insertSorted(&top, (Int32(i), v))
            }
        }
        return top
    }

    private static func insertSorted(
        _ array: inout [(Int32, Float)],
        _ entry: (Int32, Float)
    ) {
        var lo = 0
        var hi = array.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if array[mid].1 >= entry.1 {
                lo = mid + 1
            } else {
                hi = mid
            }
        }
        array.insert(entry, at: lo)
    }

    /// Project a LogprobEntry into the Chat Completions wire shape.
    /// Token strings come from the raw token bytes (preserving the
    /// leading space on SentencePiece ▁-prefixed tokens) rather than
    /// from `tokenizer.decode([id])`, which strips the first leading
    /// space as if every token were the start of a response. Without
    /// this, `▁world` would serialize as `world` in the logprobs
    /// content even though the response.text contains ` world`.
    /// Bytes are nil when the token has no decodable byte sequence
    /// (special tokens).
    static func chatTokenLogprob(
        _ entry: LogprobEntry,
        tokenizer: SmeltTokenizer
    ) -> OpenAITokenLogprob {
        return OpenAITokenLogprob(
            token: rawTokenString(entry.token, tokenizer: tokenizer),
            logprob: entry.logprob,
            bytes: rawTokenBytes(entry.token, tokenizer: tokenizer),
            topLogprobs: entry.topAlternatives.map { (id, lp) in
                OpenAITopLogprob(
                    token: rawTokenString(id, tokenizer: tokenizer),
                    logprob: lp,
                    bytes: rawTokenBytes(id, tokenizer: tokenizer)
                )
            }
        )
    }

    /// Render a single token id to its string form preserving the
    /// leading space on SentencePiece ▁-prefixed tokens (unlike
    /// `tokenizer.decode([id])` which strips it). Used by both the
    /// chat-completions logprobs projection AND the prompt-side
    /// logprobs path so prompt tokens align with response.text.
    static func rawTokenString(
        _ id: Int32,
        tokenizer: SmeltTokenizer
    ) -> String {
        guard let bytes = tokenizer.tokenBytes(for: id) else {
            // Special tokens have no decodable bytes — fall back to
            // the streaming decode (which handles added-token names
            // like `<|en|>`).
            return tokenizer.decode([id])
        }
        return String(bytes: bytes, encoding: .utf8)
            ?? tokenizer.decode([id])
    }

    private static func rawTokenBytes(
        _ id: Int32,
        tokenizer: SmeltTokenizer
    ) -> [Int]? {
        tokenizer.tokenBytes(for: id).map { $0.map { Int($0) } }
    }

    /// Project a sequence of LogprobEntries into the legacy
    /// /v1/completions parallel-array wire shape. `tokenTexts` and
    /// `textOffsets` must come from the caller. A nil entry at
    /// position i represents the first token of the prompt (BOS) on
    /// the echo:true path — no preceding distribution exists, so
    /// `tokenLogprobs[i] = nil` and `topLogprobs[i] = [:]` per
    /// OpenAI spec.
    static func completionLogprobs(
        _ entries: [LogprobEntry?],
        tokenTexts: [String],
        textOffsets: [Int],
        tokenizer: SmeltTokenizer
    ) -> OpenAICompletionLogprobs {
        precondition(
            entries.count == tokenTexts.count && entries.count == textOffsets.count,
            "entries/tokenTexts/textOffsets must be parallel"
        )
        let tokenLogprobs: [Double?] = entries.map { $0?.logprob }
        let topLogprobs: [[String: Double]] = entries.map { entry in
            guard let entry else { return [:] }
            var dict: [String: Double] = [:]
            dict.reserveCapacity(entry.topAlternatives.count)
            // topAlternatives is logit-descending; keep the FIRST
            // (highest-logit) entry on string collisions so byte-
            // fallback / whitespace-normalized duplicates don't drop
            // the better alternative.
            for (id, lp) in entry.topAlternatives {
                let key = tokenizer.decode([id])
                if dict[key] == nil { dict[key] = lp }
            }
            return dict
        }
        return OpenAICompletionLogprobs(
            tokens: tokenTexts,
            tokenLogprobs: tokenLogprobs,
            topLogprobs: topLogprobs,
            textOffset: textOffsets
        )
    }
}
