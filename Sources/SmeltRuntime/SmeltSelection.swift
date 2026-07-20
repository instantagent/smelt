import Foundation

/// Runtime token selection policy for decode and Metal prefill.
public enum SmeltSelectionMode: Sendable, Equatable {
    case argmax
    case temperature(Float, seed: UInt64)
    /// Temperature sampling after applying top-k and nucleus truncation.
    /// `topK == nil` keeps the whole allowed vocabulary; `topP == 1`
    /// disables nucleus truncation. This stays a semantic sampler brick:
    /// callers choose it from package/request policy, never model identity.
    case filteredTemperature(
        Float,
        topK: Int?,
        topP: Float,
        seed: UInt64
    )

    public var usesArgmaxFastPath: Bool {
        switch self {
        case .argmax:
            return true
        case let .temperature(temp, _):
            return !temp.isFinite || temp <= 0
        case let .filteredTemperature(temp, _, _, _):
            return !temp.isFinite || temp <= 0
        }
    }

    var gpuSamplingParameters: (invTemperature: Float, seed: UInt64)? {
        switch self {
        case .argmax:
            return nil
        case let .temperature(temp, seed):
            guard temp.isFinite, temp > 0 else { return nil }
            return (1 / temp, seed)
        case .filteredTemperature:
            // The package metallib's full-vocabulary sampler has no
            // truncation inputs. Fall back to the shared host selector until
            // a compatible generic filtered-sampling kernel is available.
            return nil
        }
    }
}

/// Independent deterministic streams used by speculative decoding. Keeping
/// proposal, acceptance, residual, and bonus draws in separate domains avoids
/// accidentally correlating random choices made by otherwise generic bricks.
public enum SmeltSamplingDomain: UInt64, Sendable {
    case drafter             = 0x4452_4146_5445_5253
    case acceptUniform       = 0x4143_4350_545F_554D
    case residualCategorical = 0x5245_5331_4443_4154
    case bonus               = 0x424F_4E55_535F_5443
}

/// splitMix64-backed deterministic RNG. Reproducible across runs,
/// conforms to Swift's `RandomNumberGenerator`.
public struct SmeltDeterministicRng: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) {
        self.state = seed
    }

    public init(
        masterSeed: UInt64,
        domain: SmeltSamplingDomain,
        position: Int32
    ) {
        let posBits = UInt64(UInt32(bitPattern: position))
        self.state = (masterSeed ^ domain.rawValue) &+ posBits
            &+ 0x9E37_79B9_7F4A_7C15
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

public enum SmeltLogitsSelector {
    public static func select(
        logits: UnsafeBufferPointer<Float16>,
        position: Int32,
        mode: SmeltSelectionMode,
        allowedTokenMask: [UInt32]? = nil
    ) -> Int32 {
        select(
            count: logits.count,
            position: position,
            mode: mode,
            allowedTokenMask: allowedTokenMask
        ) { index in Float(logits[index]) }
    }

    static func select(
        logits: [Float],
        position: Int32,
        mode: SmeltSelectionMode,
        allowedTokenMask: [UInt32]? = nil
    ) -> Int32 {
        select(
            count: logits.count,
            position: position,
            mode: mode,
            allowedTokenMask: allowedTokenMask
        ) { index in logits[index] }
    }

    private static func select(
        count: Int,
        position: Int32,
        mode: SmeltSelectionMode,
        allowedTokenMask: [UInt32]?,
        valueAt: (Int) -> Float
    ) -> Int32 {
        guard count > 0 else { return 0 }

        switch mode {
        case .argmax:
            return argmax(
                count: count,
                allowedTokenMask: allowedTokenMask,
                valueAt: valueAt
            )
        case let .temperature(temp, seed):
            guard temp.isFinite, temp > 0 else {
                return argmax(
                    count: count,
                    allowedTokenMask: allowedTokenMask,
                    valueAt: valueAt
                )
            }
            return sampleTemperature(
                count: count,
                temperature: temp,
                seed: seed,
                position: position,
                allowedTokenMask: allowedTokenMask,
                valueAt: valueAt
            )
        case let .filteredTemperature(temp, topK, topP, seed):
            guard temp.isFinite, temp > 0 else {
                return argmax(
                    count: count,
                    allowedTokenMask: allowedTokenMask,
                    valueAt: valueAt
                )
            }
            return sampleFilteredTemperature(
                count: count,
                temperature: temp,
                topK: topK,
                topP: topP,
                seed: seed,
                position: position,
                allowedTokenMask: allowedTokenMask,
                valueAt: valueAt
            )
        }
    }

    private static func argmax(
        count: Int,
        allowedTokenMask: [UInt32]?,
        valueAt: (Int) -> Float
    ) -> Int32 {
        var bestValue = -Float.infinity
        var bestIndex = -1
        if let mask = allowedTokenMask {
            // Visit only set bits: grammar masks at structural positions
            // allow a handful of tokens, so this beats a full-vocab scan
            // with a per-token bit test (and matches its ascending-index
            // tie-breaking exactly).
            for wordIndex in 0..<mask.count {
                var word = mask[wordIndex]
                guard word != 0 else { continue }
                let base = wordIndex * 32
                guard base < count else { break }
                while word != 0 {
                    let index = base + word.trailingZeroBitCount
                    word &= word - 1
                    guard index < count else { break }
                    let value = valueAt(index)
                    if value > bestValue {
                        bestValue = value
                        bestIndex = index
                    }
                }
            }
        } else {
            for index in 0..<count {
                let value = valueAt(index)
                if value > bestValue {
                    bestValue = value
                    bestIndex = index
                }
            }
        }
        precondition(bestIndex >= 0, "selection mask allowed no tokens")
        return Int32(bestIndex)
    }

    private static func sampleTemperature(
        count: Int,
        temperature: Float,
        seed: UInt64,
        position: Int32,
        allowedTokenMask: [UInt32]?,
        valueAt: (Int) -> Float
    ) -> Int32 {
        var maxLogit = -Float.infinity
        var maxIndex = -1
        for index in 0..<count {
            guard tokenIsAllowed(index, by: allowedTokenMask) else { continue }
            let value = valueAt(index)
            if value > maxLogit {
                maxLogit = value
                maxIndex = index
            }
        }
        precondition(maxIndex >= 0, "selection mask allowed no tokens")

        let invTemp = 1 / temperature
        var totalMass: Float = 0
        for index in 0..<count {
            guard tokenIsAllowed(index, by: allowedTokenMask) else { continue }
            totalMass += Float(
                Foundation.exp(Double((valueAt(index) - maxLogit) * invTemp))
            )
        }
        guard totalMass.isFinite, totalMass > 0 else {
            return Int32(maxIndex)
        }

        let threshold = deterministicUnit(seed: seed, position: position) * totalMass
        var runningMass: Float = 0
        for index in 0..<count {
            guard tokenIsAllowed(index, by: allowedTokenMask) else { continue }
            runningMass += Float(
                Foundation.exp(Double((valueAt(index) - maxLogit) * invTemp))
            )
            if runningMass >= threshold {
                return Int32(index)
            }
        }
        return Int32(maxIndex)
    }

    /// Sample from the smallest top-logit candidate set admitted by top-k,
    /// then from the smallest descending-probability prefix whose cumulative
    /// mass reaches top-p. Rank truncation happens after the grammar mask and
    /// temperature is applied before nucleus mass, matching the conventional
    /// Transformers/MLX ordering.
    private static func sampleFilteredTemperature(
        count: Int,
        temperature: Float,
        topK: Int?,
        topP: Float,
        seed: UInt64,
        position: Int32,
        allowedTokenMask: [UInt32]?,
        valueAt: (Int) -> Float
    ) -> Int32 {
        let requestedK = min(max(topK ?? count, 1), count)
        var candidates: [(index: Int, logit: Float)] = []
        candidates.reserveCapacity(requestedK)

        for index in 0..<count {
            guard tokenIsAllowed(index, by: allowedTokenMask) else { continue }
            let candidate = (index: index, logit: valueAt(index))
            if candidates.count < requestedK {
                candidates.append(candidate)
                if candidates.count == requestedK {
                    candidates.sort {
                        $0.logit == $1.logit
                            ? $0.index < $1.index : $0.logit > $1.logit
                    }
                }
            } else if candidate.logit > candidates.last!.logit
                        || (candidate.logit == candidates.last!.logit
                            && candidate.index < candidates.last!.index) {
                candidates[candidates.count - 1] = candidate
                candidates.sort {
                    $0.logit == $1.logit
                        ? $0.index < $1.index : $0.logit > $1.logit
                }
            }
        }
        precondition(!candidates.isEmpty, "selection mask allowed no tokens")

        let invTemp = 1 / temperature
        let maxLogit = candidates[0].logit
        var masses = candidates.map {
            Float(Foundation.exp(Double(($0.logit - maxLogit) * invTemp)))
        }
        var total = masses.reduce(Float(0), +)
        guard total.isFinite, total > 0 else {
            return Int32(candidates[0].index)
        }

        let nucleus = min(max(topP, Float.leastNonzeroMagnitude), 1)
        if nucleus < 1 {
            let cutoff = total * nucleus
            var cumulative: Float = 0
            var keep = 0
            while keep < masses.count && cumulative < cutoff {
                cumulative += masses[keep]
                keep += 1
            }
            candidates.removeSubrange(keep..<candidates.count)
            masses.removeSubrange(keep..<masses.count)
            total = cumulative
        }

        let threshold = deterministicUnit(seed: seed, position: position) * total
        var cumulative: Float = 0
        for (candidate, mass) in zip(candidates, masses) {
            cumulative += mass
            if cumulative >= threshold { return Int32(candidate.index) }
        }
        return Int32(candidates.last!.index)
    }

    /// O(1) bit test against an allowed-token mask.
    static func isAllowed(_ tokenId: Int32, in mask: [UInt32]) -> Bool {
        guard tokenId >= 0 else { return false }
        let wordIndex = Int(tokenId) / 32
        guard wordIndex < mask.count else { return false }
        return ((mask[wordIndex] >> UInt32(tokenId % 32)) & 1) == 1
    }

    private static func tokenIsAllowed(
        _ tokenId: Int,
        by mask: [UInt32]?
    ) -> Bool {
        guard let mask else { return true }
        return isAllowed(Int32(tokenId), in: mask)
    }

    private static func deterministicUnit(seed: UInt64, position: Int32) -> Float {
        // Match the Metal kernel's splitmix64 initial state. The
        // shader runs `splitmix64(seed + golden + position)`, whose
        // inner increment adds golden again before mixing; the
        // CPU stream must produce the same uniform per (seed,
        // position) so seeded sampling stays reproducible when the
        // GPU sampler is unavailable (e.g. allowedTokenMask) and
        // falls through to this path.
        var rng = SmeltDeterministicRng(
            seed: seed &+ 0x9E37_79B9_7F4A_7C15
                &+ UInt64(UInt32(bitPattern: position))
        )
        let bits = rng.next()
        let mantissa = UInt32((bits >> 40) & 0x00FF_FFFF)
        let unit = (Float(mantissa) + 0.5) * (1 / 16_777_216)
        return min(max(unit, Float.leastNonzeroMagnitude), 1 - Float.ulpOfOne)
    }
}
