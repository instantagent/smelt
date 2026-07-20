import Foundation

public enum SmeltAudioWER {

    /// Word error rate via Levenshtein distance over whitespace-
    /// tokenized words. Reference is the source string; hypothesis
    /// is the model output. Both are normalized (lowercased,
    /// punctuation stripped) before tokenization. Returns the count
    /// of word edits divided by the reference's word count, so
    /// 0 == identical, 1 == every word wrong.
    public static func wordErrorRate(reference: String, hypothesis: String) -> Double {
        let refWords = normalize(reference).split(separator: " ").map(String.init)
        let hypWords = normalize(hypothesis).split(separator: " ").map(String.init)
        if refWords.isEmpty { return hypWords.isEmpty ? 0.0 : 1.0 }
        return Double(levenshtein(refWords, hypWords)) / Double(refWords.count)
    }

    public static func normalize(_ s: String) -> String {
        let lower = s.lowercased()
        let stripped = lower.unicodeScalars.map { sc -> Character in
            CharacterSet.alphanumerics.contains(sc) ? Character(sc) : " "
        }
        return String(stripped)
            .split(separator: " ", omittingEmptySubsequences: true)
            .joined(separator: " ")
    }

    private static func levenshtein(_ a: [String], _ b: [String]) -> Int {
        if a.isEmpty { return b.count }
        if b.isEmpty { return a.count }
        var prev = Array(0...b.count)
        var curr = [Int](repeating: 0, count: b.count + 1)
        for i in 1...a.count {
            curr[0] = i
            for j in 1...b.count {
                let cost = a[i - 1] == b[j - 1] ? 0 : 1
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            }
            swap(&prev, &curr)
        }
        return prev[b.count]
    }
}
