// Qwen3TTSSampler — temperature/top-k categorical sampling for the talker decode, the model's
// configured generation mode (generation_config.json: do_sample=true). Greedy argmax degenerates
// into stuck single-token loops on harder prompts (drops content, never emits EOS); sampling breaks
// those loops. cb0 samples on the host (this file); the MTP sub-talker samples on-GPU via the
// sample_topk_f32 kernel, which mirrors this algorithm. Top-p is 1.0 in the checkpoint (identity),
// so only temperature + top-k are implemented; a non-1.0 top_p must fail loudly, not be ignored.

import Foundation

public enum Qwen3TTSSampler {

    /// Fallback sampling knobs (the checkpoint's generation_config.json values). The package path now
    /// reads these from the manifest (Qwen3TTSManifest.Decode) — these statics only fill `Params`
    /// defaults when a caller hand-builds a Params without specifying them.
    public enum Config {
        public static let talkerTemperature: Float = 0.9   // temperature
        public static let talkerTopK = 50                  // top_k
        public static let subtalkerTemperature: Float = 0.9 // subtalker_temperature
        public static let subtalkerTopK = 50               // subtalker_top_k
        public static let topP: Float = 1.0                // top_p / subtalker_top_p (identity)
    }

    /// One decode's sampling state: the seed plus the active temperature/top-k. Greedy is selected by
    /// passing `sampling: nil` to generate/generateCodes (the deterministic argmax path that keeps the
    /// bit-exact gate) — NOT by a zero temperature: a Params always samples and requires temperature > 0.
    public struct Params {
        public var seed: UInt64
        public var talkerTemperature: Float
        public var talkerTopK: Int
        public var subtalkerTemperature: Float
        public var subtalkerTopK: Int
        public init(seed: UInt64,
                    talkerTemperature: Float = Config.talkerTemperature, talkerTopK: Int = Config.talkerTopK,
                    subtalkerTemperature: Float = Config.subtalkerTemperature, subtalkerTopK: Int = Config.subtalkerTopK) {
            self.seed = seed
            self.talkerTemperature = talkerTemperature; self.talkerTopK = talkerTopK
            self.subtalkerTemperature = subtalkerTemperature; self.subtalkerTopK = subtalkerTopK
        }
    }

    /// How a public generate() call selects its decode policy. `.packageDefault` (the default) follows
    /// the package's compiled generation config: when `manifest.decode.doSample` it samples with a FRESH
    /// random seed per call (matching the reference model's unseeded sampling — repeated calls vary),
    /// else greedy. `.sampleSeeded` is the same policy but with a fixed seed — reproducible package
    /// sampling (uses the package's params, unlike `.sample` which carries its own). `.greedy` forces
    /// the deterministic argmax path (the bit-exact gate). `.sample` overrides with explicit params.
    public enum DecodeMode {
        case packageDefault
        case sampleSeeded(UInt64)
        case greedy
        case sample(Params)
    }

    /// Deterministic uniform in [0, 1) for sampling step (frame, codebook) under `seed`. Codebook 0
    /// is cb0; 1...15 are the MTP sub-passes. An independent SmeltDeterministicRng stream per step
    /// keeps draws stateless (no RNG threaded through the frame loop or the GPU chain); the top 24
    /// bits give an exactly-representable fraction.
    public static func uniform(seed: UInt64, frame: Int, codebook: Int) -> Float {
        SmeltSamplingRandom.uniform(seed: seed, step: frame, stream: codebook)
    }

    /// Draw one index from `softmax(top_k(logits) / temperature)` using the precomputed uniform `u`.
    /// Threshold = the k-th largest logit; every token with `logit >= threshold` is kept (matching
    /// HF TopKLogitsWarper tie behavior: values strictly below the k-th are removed). Temperature is
    /// applied in the softmax weights only — positive temperature preserves rank, so the top-k cut on
    /// raw logits is identical to cutting after scaling. Inverse-CDF walk in index order; `Double`
    /// accumulation keeps the cumulative sum stable. Requires `temperature > 0` (greedy is the
    /// caller's argmax path).
    public static func sampleTopK(_ logits: [Float], temperature: Float, topK: Int, u: Float) -> Int {
        precondition(temperature > 0, "sampleTopK needs temperature > 0; greedy is the argmax path")
        let n = logits.count
        precondition(n > 0, "sampleTopK on empty logits")
        let k = min(max(topK, 1), n)
        let maxLogit = logits.max()!
        precondition(maxLogit > -.infinity, "sampleTopK: all logits suppressed (-inf); no token to draw")
        let threshold = logits.sorted(by: >)[k - 1]           // k-th largest
        let invT = 1.0 / Double(temperature)
        // Materialize the kept set once (≤k+ties entries), so exp isn't recomputed in the CDF walk.
        var kept: [(index: Int, weight: Double)] = []
        kept.reserveCapacity(k)
        var total = 0.0
        // `> -inf` excludes suppressed tokens even when topK reaches into the suppressed tail
        // (threshold == -inf): a zero-weight -inf entry must never be selectable. No-op when the
        // threshold is finite (the normal case), since -inf < any finite threshold already.
        for i in 0..<n where logits[i] > -.infinity && logits[i] >= threshold {
            let w = Foundation.exp(Double(logits[i] - maxLogit) * invT)
            kept.append((i, w)); total += w
        }
        let target = Double(u) * total
        var acc = 0.0
        for (index, weight) in kept {
            acc += weight
            if acc >= target { return index }
        }
        return kept[kept.count - 1].index                     // u≈1 rounding: the last kept token
    }
}
