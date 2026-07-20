public struct SmeltBenchmarkGateProfile: Sendable {
    public let minDecodeTokensPerSecond: Double
    public let maxDecodeP95Ms: Double
    public let minPrefillTokensPerSecond: [Int: Double]
    public let maxPrefillP95Ms: [Int: Double]
}
