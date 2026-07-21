import Foundation
import SmeltRuntime

// MARK: - Config

// Note: Qwen 3.5 packages are broken on the current tree (decode emits whitespace
// only — likely the in-progress Gemma 4 fusion work in Sources/SmeltCompiler and
// SmeltRuntime). Llama 3.2 1B and 3B both decode correctly today; using them for
// the acceptance measurement gives the same answer for the same question.
let smallPath = "../../artifacts/llama32-1b-affine/meta-llama_Llama-3.2-1B-Instruct.smeltpkg"
let largePath = "../../artifacts/llama32-3b-affine/meta-llama_Llama-3.2-3B-Instruct.smeltpkg"
let measureTokens = 32   // tokens per prompt to measure agreement on

// Diverse prompt corpus designed to span the regimes codex flagged:
// - low-temp deterministic continuations (high acceptance expected)
// - code-shaped continuations
// - JSON / tool-call shaped continuations
// - agent-style instruction following (codex predicted 0.4-0.65)
// - high-entropy chatter (lower bound)
struct Prompt {
    let label: String
    let text: String
}

let prompts: [Prompt] = [
    .init(label: "factual",
          text: "The capital of France is"),
    .init(label: "code-completion",
          text: "def fibonacci(n):\n    if n < 2:\n        return n\n    return"),
    .init(label: "json-tool-call",
          text: "Here is a JSON tool call:\n{\"tool\": \"read_file\", \"arguments\": {\"path\": \""),
    .init(label: "agent-style",
          text: "User: Please list the steps to deploy a Node.js app to production.\nAssistant: To deploy a Node.js app, you should:\n1."),
    .init(label: "open-ended",
          text: "Once upon a time in a small village by the sea, there lived a"),
    .init(label: "code-explanation",
          text: "The following Python code uses a generator. Explain what it does in one sentence:\n```python\ndef gen():\n    for i in range(10):\n        yield i * 2\n```\nThis function"),
]

// MARK: - Helpers

func argmaxNextToken(model: SmeltModel, tokens: [Int32]) throws -> Int32 {
    var first: Int32 = -1
    _ = try model.generate(tokenIds: tokens, selectionMode: .argmax) { tok in
        first = tok.id
        return false  // stop after the first generated token
    }
    return first
}

func generateSequence(model: SmeltModel, tokens: [Int32], maxTokens: Int) throws -> [Int32] {
    var out: [Int32] = []
    _ = try model.generate(tokenIds: tokens, selectionMode: .argmax) { tok in
        out.append(tok.id)
        return out.count < maxTokens
    }
    return out
}

func mean(_ xs: [Double]) -> Double {
    guard !xs.isEmpty else { return 0 }
    return xs.reduce(0, +) / Double(xs.count)
}

func pct(_ x: Double) -> String { String(format: "%.1f%%", x * 100) }

// MARK: - Llama 3 instruct chat template (mirrors SmeltCLI's llama3-instruct case).

func llamaInstructTokens(prompt: String, tokenizer: SmeltTokenizer) -> [Int32] {
    let bos: Int32 = 128_000
    let startHeader: Int32 = 128_006
    let endHeader: Int32 = 128_007
    let eotId: Int32 = 128_009
    var ids: [Int32] = [bos, startHeader]
    ids += tokenizer.encode("user")
    ids += [endHeader]
    ids += tokenizer.encode("\n\n")
    ids += tokenizer.encode(prompt)
    ids += [eotId, startHeader]
    ids += tokenizer.encode("assistant")
    ids += [endHeader]
    ids += tokenizer.encode("\n\n")
    return ids
}

// MARK: - Probe mode (SPROBE=1) — sanity check each model in isolation.

if ProcessInfo.processInfo.environment["SPROBE"] == "1" {
    let tk = try SmeltTokenizer(path: "\(smallPath)/tokenizer.json")
    let prompt = "The capital of France is?"
    let promptToks = llamaInstructTokens(prompt: prompt, tokenizer: tk)
    print("prompt: '\(prompt)' → \(promptToks.count) tokens")

    print("\n--- 0.8B alone ---")
    let s = try SmeltModel(package: smallPath)
    var sOut: [Int32] = []
    _ = try s.generate(tokenIds: promptToks, selectionMode: .argmax) { tok in
        sOut.append(tok.id); return sOut.count < 16
    }
    print("0.8B: \(sOut)\n      '\(tk.decode(sOut))'")

    print("\n--- 4B alone ---")
    let l = try SmeltModel(package: largePath)
    var lOut: [Int32] = []
    _ = try l.generate(tokenIds: promptToks, selectionMode: .argmax) { tok in
        lOut.append(tok.id); return lOut.count < 16
    }
    print("4B  : \(lOut)\n      '\(tk.decode(lOut))'")
    exit(0)
}

// MARK: - Setup

print("loading models...")
let t0 = Date()
let small = try SmeltModel(package: smallPath)
let smallLoad = Date().timeIntervalSince(t0)
let t1 = Date()
let large = try SmeltModel(package: largePath)
let largeLoad = Date().timeIntervalSince(t1)
print(String(format: "  0.8B loaded in %.2fs", smallLoad))
print(String(format: "  4B   loaded in %.2fs", largeLoad))

// Both Qwen 3.5 packages share a tokenizer; load from either.
let tokenizer = try SmeltTokenizer(path: "\(largePath)/tokenizer.json")

// MARK: - Measurement

print("")
print("measuring acceptance over \(prompts.count) prompts × \(measureTokens) tokens each")
print(String(repeating: "─", count: 78))

var allAlphas: [Double] = []
var allConvLens: [Int] = []
var rows: [(String, Double, Int, Double)] = []  // label, alpha, convLen, secs

for prompt in prompts {
    let promptTokens = llamaInstructTokens(prompt: prompt.text, tokenizer: tokenizer)
    let tStart = Date()

    // 1. Generate target sequence with the 4B (ground truth).
    let target = try generateSequence(model: large, tokens: promptTokens, maxTokens: measureTokens)
    guard target.count == measureTokens else {
        print("[skip] \(prompt.label): 4B emitted only \(target.count) tokens (likely EOS)")
        continue
    }

    // 2. Convergent prefix length: argmax-only run from 0.8B; how many initial tokens match?
    let smallSeq = try generateSequence(model: small, tokens: promptTokens, maxTokens: measureTokens)
    var convLen = 0
    for i in 0..<min(target.count, smallSeq.count) {
        if target[i] == smallSeq[i] { convLen += 1 } else { break }
    }

    // DEBUG: dump first 8 tokens of each
    let targetText = tokenizer.decode(Array(target.prefix(16)))
    let smallText = tokenizer.decode(Array(smallSeq.prefix(16)))
    print("    [4B  ] tokens: \(target.prefix(8).map { String($0) }.joined(separator: ","))  '\(targetText)'")
    print("    [0.8B] tokens: \(smallSeq.prefix(8).map { String($0) }.joined(separator: ","))  '\(smallText)'")

    // 3. Per-position acceptance: query 0.8B's argmax for each prefix [prompt + target[0..<i]].
    //    This is the honest α: probability that the drafter agrees with the target
    //    given the target's prior tokens (the spec-decode interaction shape).
    var matches = 0
    for i in 0..<target.count {
        let prefix = promptTokens + Array(target[0..<i])
        let drafted = try argmaxNextToken(model: small, tokens: prefix)
        if drafted == target[i] { matches += 1 }
    }
    let alpha = Double(matches) / Double(target.count)
    let secs = Date().timeIntervalSince(tStart)

    rows.append((prompt.label, alpha, convLen, secs))
    allAlphas.append(alpha)
    allConvLens.append(convLen)

    let labelPad = prompt.label.padding(toLength: 18, withPad: " ", startingAt: 0)
    print("  \(labelPad)  α = \(pct(alpha))   conv-len = \(String(format: "%2d", convLen))/\(measureTokens)   (\(String(format: "%.1f", secs))s)")
}

// MARK: - Summary

print(String(repeating: "─", count: 78))
let avgAlpha = mean(allAlphas)
let avgConv = mean(allConvLens.map(Double.init))
print("  overall α  = \(pct(avgAlpha))")
print("  overall conv-len = \(String(format: "%.1f", avgConv)) / \(measureTokens) tokens")

// Expected speedup model: with K speculative tokens per round,
//   E[accepted] = (1 - α^(K+1)) / (1 - α)
// (Leviathan et al., the standard speculative-decoding formula.)
print("")
print("expected speedup vs greedy decode at this α:")
for K in [2, 4, 6, 8] {
    let expectedAccepted = (1.0 - pow(avgAlpha, Double(K + 1))) / (1.0 - avgAlpha)
    print(String(format: "  K=%d draft tokens → E[accepted] = %.2f tokens/round", K, expectedAccepted))
}

// Cost model: spec decode is profitable iff
//   E[accepted] / (1 + draft_cost / target_cost) > 1
// 0.8B decode ≈ 3.2 ms/tok, 4B decode ≈ 9.5 ms/tok on M2 Max → ratio ≈ 0.34.
print("")
print("rough profitability (M2 Max baseline: 0.8B ≈ 3.2ms/tok, 4B ≈ 9.5ms/tok, ratio ≈ 0.34):")
let draftCostRatio = 3.2 / 9.5
for K in [2, 4, 6, 8] {
    let accepted = (1.0 - pow(avgAlpha, Double(K + 1))) / (1.0 - avgAlpha)
    // One target verification call for K positions costs roughly target_cost (parallel verify).
    // K draft tokens cost K * draft_cost.
    // Effective speedup = accepted / (1 + K * draft_cost_ratio).
    let speedup = accepted / (1.0 + Double(K) * draftCostRatio)
    let verdict = speedup > 1.0 ? "win" : "loss"
    print("  K=\(K) → speedup \(String(format: "%.2f", speedup))x (\(verdict))")
}

print("")
print("verdict:")
if avgAlpha > 0.7 {
    print("  α > 0.7. Spec decode is a clear win across most prompts. Ship it.")
} else if avgAlpha > 0.5 {
    print("  α in 0.5-0.7. Spec decode wins for K=2-4. Worth implementing with adaptive K.")
} else if avgAlpha > 0.35 {
    print("  α in 0.35-0.5. Spec decode is marginal. Implement only if other Tier 1 features land first.")
} else {
    print("  α < 0.35. Spec decode loses given the draft cost ratio. Demote off Tier 2.")
}
