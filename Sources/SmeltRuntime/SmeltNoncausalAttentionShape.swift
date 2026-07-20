/// Validation failures for the standard noncausal-attention runtime ABI.
public enum SmeltNoncausalAttentionShapeError: Error, Equatable, Sendable {
    /// Query token count must be positive.
    case emptyQuery
    /// Key/value token count must be positive because softmax over an empty
    /// source set is undefined.
    case emptyKeyValue
    /// Attention must contain at least one head.
    case emptyHeads
    /// The exact noncausal kernel maps at most two values to each of 32 SIMD
    /// lanes, so supported head dimensions are 1 through 64.
    case unsupportedHeadDimension(Int)
    /// A public count cannot be represented by the UInt32 Metal ABI.
    case countExceedsUInt32
    /// A derived scalar count overflowed Int.
    case scalarCountOverflow
}

/// Checked shape for Smelt's standard batch-one noncausal-attention ABI.
public struct SmeltNoncausalAttentionShape: Equatable, Sendable {
    /// Number of independently packed query rows.
    public let queryTokens: Int
    /// Number of shared key and value rows.
    public let keyValueTokens: Int
    /// Number of attention heads.
    public let heads: Int
    /// Scalars in each head.
    public let headDim: Int
    /// Scalars in one token-major row.
    public let hiddenSize: Int
    /// Total query/output scalar count.
    public let queryScalarCount: Int
    /// Total key or value scalar count.
    public let keyValueScalarCount: Int

    /// Creates a shape only when every dimension is representable by the
    /// current Metal ABI and by the D64 SIMD implementation.
    public init(
        queryTokens: Int,
        keyValueTokens: Int,
        heads: Int,
        headDim: Int
    ) throws {
        guard queryTokens > 0 else {
            throw SmeltNoncausalAttentionShapeError.emptyQuery
        }
        guard keyValueTokens > 0 else {
            throw SmeltNoncausalAttentionShapeError.emptyKeyValue
        }
        guard heads > 0 else {
            throw SmeltNoncausalAttentionShapeError.emptyHeads
        }
        guard (1...64).contains(headDim) else {
            throw SmeltNoncausalAttentionShapeError.unsupportedHeadDimension(headDim)
        }
        guard queryTokens <= UInt32.max,
              keyValueTokens <= UInt32.max,
              heads <= UInt32.max,
              headDim <= UInt32.max
        else {
            throw SmeltNoncausalAttentionShapeError.countExceedsUInt32
        }

        let (hiddenSize, hiddenOverflow) = heads.multipliedReportingOverflow(by: headDim)
        let (queryScalarCount, queryOverflow) = queryTokens.multipliedReportingOverflow(
            by: hiddenSize
        )
        let (keyValueScalarCount, keyValueOverflow) = keyValueTokens
            .multipliedReportingOverflow(by: hiddenSize)
        guard !hiddenOverflow, !queryOverflow, !keyValueOverflow else {
            throw SmeltNoncausalAttentionShapeError.scalarCountOverflow
        }

        self.queryTokens = queryTokens
        self.keyValueTokens = keyValueTokens
        self.heads = heads
        self.headDim = headDim
        self.hiddenSize = hiddenSize
        self.queryScalarCount = queryScalarCount
        self.keyValueScalarCount = keyValueScalarCount
    }
}
