/// Stateless deterministic sampling draws shared by autoregressive consumers.
public enum SmeltSamplingRandom {
    /// Returns one exactly representable uniform value in `[0, 1)` for a
    /// `(seed, step, stream)` address.
    public static func uniform(seed: UInt64, step: Int, stream: Int) -> Float {
        var rng = SmeltDeterministicRng(
            seed: seed
                &+ 0x9E3779B97F4A7C15 &* UInt64(bitPattern: Int64(step))
                &+ UInt64(bitPattern: Int64(stream))
        )
        return Float(rng.next() >> 40) * (1.0 / Float(1 << 24))
    }
}
