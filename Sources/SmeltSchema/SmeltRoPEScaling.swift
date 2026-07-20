public enum SmeltRoPEScalingType: String, Codable, Sendable, Hashable {
    case llama3
}

public struct SmeltRoPEScaling: Codable, Sendable, Hashable {
    public let type: SmeltRoPEScalingType
    public let factor: Float
    public let lowFreqFactor: Float
    public let highFreqFactor: Float
    public let originalMaxPositionEmbeddings: Int

    public init(
        type: SmeltRoPEScalingType,
        factor: Float,
        lowFreqFactor: Float,
        highFreqFactor: Float,
        originalMaxPositionEmbeddings: Int
    ) {
        self.type = type
        self.factor = factor
        self.lowFreqFactor = lowFreqFactor
        self.highFreqFactor = highFreqFactor
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case factor
        case lowFreqFactor = "low_freq_factor"
        case highFreqFactor = "high_freq_factor"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
    }
}
