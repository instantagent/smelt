import Testing

@testable import SmeltRuntime

@Suite("Smelt noncausal-attention shape")
struct SmeltNoncausalAttentionShapeTests {
  @Test("Production SkinTokens orientations are representable")
    func productionOrientationsAreRepresentable() throws {
        let orientations = [
            (512, 54_000, 8, 64),
            (512, 512, 8, 64),
            (384, 54_000, 12, 64),
            (54_000, 388, 12, 64),
        ]
        for (queryTokens, keyValueTokens, heads, headDim) in orientations {
            let shape = try SmeltNoncausalAttentionShape(
                queryTokens: queryTokens,
                keyValueTokens: keyValueTokens,
                heads: heads,
                headDim: headDim
            )
            #expect(shape.hiddenSize == heads * headDim)
            #expect(shape.queryScalarCount == queryTokens * heads * headDim)
            #expect(shape.keyValueScalarCount == keyValueTokens * heads * headDim)
        }
    }

    @Test("Undefined or unsupported shapes fail loudly")
    func invalidShapesFailLoudly() {
        #expect(throws: SmeltNoncausalAttentionShapeError.emptyQuery) {
            try SmeltNoncausalAttentionShape(
                queryTokens: 0,
                keyValueTokens: 1,
                heads: 1,
                headDim: 64
            )
        }
        #expect(throws: SmeltNoncausalAttentionShapeError.emptyKeyValue) {
            try SmeltNoncausalAttentionShape(
                queryTokens: 1,
                keyValueTokens: 0,
                heads: 1,
                headDim: 64
            )
        }
        #expect(throws: SmeltNoncausalAttentionShapeError.emptyHeads) {
            try SmeltNoncausalAttentionShape(
                queryTokens: 1,
                keyValueTokens: 1,
                heads: 0,
                headDim: 64
            )
        }
        #expect(throws: SmeltNoncausalAttentionShapeError.unsupportedHeadDimension(65)) {
            try SmeltNoncausalAttentionShape(
                queryTokens: 1,
                keyValueTokens: 1,
                heads: 1,
                headDim: 65
            )
        }
    }
}
