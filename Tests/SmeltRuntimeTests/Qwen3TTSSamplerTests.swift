// Qwen3TTSSamplerTests — pure-CPU gates for the temperature/top-k sampler (no model needed).
import Foundation
import Testing
@testable import SmeltRuntime

struct Qwen3TTSSamplerTests {

    @Test func uniformInRangeAndDeterministic() {
        for frame in 0..<8 { for cb in 0..<16 {
            let u = Qwen3TTSSampler.uniform(seed: 42, frame: frame, codebook: cb)
            #expect(u >= 0 && u < 1)
            #expect(u == Qwen3TTSSampler.uniform(seed: 42, frame: frame, codebook: cb))  // deterministic
        } }
        // distinct (seed, frame, codebook) decorrelate
        #expect(Qwen3TTSSampler.uniform(seed: 1, frame: 0, codebook: 0)
                != Qwen3TTSSampler.uniform(seed: 2, frame: 0, codebook: 0))
        #expect(Qwen3TTSSampler.uniform(seed: 1, frame: 0, codebook: 0)
                != Qwen3TTSSampler.uniform(seed: 1, frame: 1, codebook: 0))
        #expect(Qwen3TTSSampler.uniform(seed: 1, frame: 0, codebook: 0)
                != Qwen3TTSSampler.uniform(seed: 1, frame: 0, codebook: 1))
    }

    @Test func topK1IsArgmaxRegardlessOfU() {
        let logits: [Float] = [0.1, 3.5, -2.0, 3.4, 1.0]
        for u: Float in [0.0, 0.25, 0.5, 0.9, 0.999] {
            #expect(Qwen3TTSSampler.sampleTopK(logits, temperature: 0.9, topK: 1, u: u) == 1)
        }
    }

    @Test func suppressedTokensNeverSelected() {
        // -inf logits (the cb0 suppress path) must never be drawn, at any u.
        var logits = [Float](repeating: 1.0, count: 8)
        logits[3] = -.infinity; logits[5] = -.infinity
        for i in 0..<200 {
            let u = Qwen3TTSSampler.uniform(seed: 7, frame: i, codebook: 0)
            let idx = Qwen3TTSSampler.sampleTopK(logits, temperature: 0.9, topK: 8, u: u)
            #expect(idx != 3 && idx != 5)
        }
    }

    @Test func suppressedLeadingTokenNeverSelectedAtThresholdEdge() {
        // topK reaches into the suppressed tail (threshold == -inf), and a suppressed token is FIRST.
        // At u=0 a naive `acc >= target` would return the leading zero-weight -inf token.
        let logits: [Float] = [-.infinity, 1.0, -.infinity, 2.0]
        for u: Float in [0.0, 0.0001, 0.5, 0.999] {
            let idx = Qwen3TTSSampler.sampleTopK(logits, temperature: 0.9, topK: 4, u: u)
            #expect(idx == 1 || idx == 3)   // only the finite tokens
        }
    }

    @Test func keepsOnlyTopK() {
        // 6 logits, top_k=2 → only the two largest (indices 4,2) are reachable.
        let logits: [Float] = [0.0, 1.0, 3.0, 2.0, 4.0, 0.5]
        var seen = Set<Int>()
        for i in 0..<500 {
            let u = Qwen3TTSSampler.uniform(seed: 99, frame: i, codebook: 0)
            seen.insert(Qwen3TTSSampler.sampleTopK(logits, temperature: 1.0, topK: 2, u: u))
        }
        #expect(seen == Set([4, 2]))
    }

    @Test func empiricalFrequencyMatchesSoftmax() {
        // Two-token distribution after top_k: P(i) = softmax(logit/temp). This sweeps u *uniformly*
        // over [0,1) rather than drawing randomly, so inverse-CDF sampling makes the selection
        // frequency equal p0 EXACTLY up to the sweep step 1/trials (~5e-5) — not Monte-Carlo noise
        // (~1/sqrt(trials)). The tight bound is the point: a loose tolerance would hide a real skew.
        let logits: [Float] = [2.0, 1.0, -5.0, -6.0]   // top_k=2 keeps indices 0,1
        let temp: Float = 0.8
        let w0 = exp(Double(logits[0]) / Double(temp)), w1 = exp(Double(logits[1]) / Double(temp))
        let p0 = w0 / (w0 + w1)
        let trials = 20_000
        var c0 = 0
        for i in 0..<trials {
            let u = Float(i) / Float(trials)            // uniform sweep of [0,1)
            if Qwen3TTSSampler.sampleTopK(logits, temperature: temp, topK: 2, u: u) == 0 { c0 += 1 }
        }
        let freq0 = Double(c0) / Double(trials)
        #expect(abs(freq0 - p0) < 2.0 / Double(trials))   // within one sweep step
    }
}
