import Foundation
import Testing
@testable import SmeltRuntime
import SmeltSchema

@Test func ropeTableBuilderUsesExpectedPairDuplication() {
    let tables = SmeltRoPETables.build(
        rowCount: 2,
        dim: 4,
        theta: 10_000,
        freqDim: nil
    )

    #expect(tables.cos.count == 8)
    #expect(tables.sin.count == 8)

    // position 0 is the identity rotation
    #expect(tables.cos[0] == Float16(1))
    #expect(tables.cos[1] == Float16(1))
    #expect(tables.sin[0] == Float16(0))
    #expect(tables.sin[1] == Float16(0))
    #expect(tables.cos[2] == Float16(1))
    #expect(tables.cos[3] == Float16(1))

    let angle0 = Float(1.0)
    let angle1 = Float(1.0 / pow(10_000.0, 0.5))
    let expectedCos0 = Float16(Float(cos(Double(angle0))))
    let expectedSin0 = Float16(Float(sin(Double(angle0))))
    let expectedCos1 = Float16(Float(cos(Double(angle1))))
    let expectedSin1 = Float16(Float(sin(Double(angle1))))

    // position 1, pair 0
    #expect(tables.cos[4] == expectedCos0)
    #expect(tables.cos[5] == expectedCos0)
    #expect(tables.sin[4] == expectedSin0)
    #expect(tables.sin[5] == expectedSin0)
    // position 1, pair 1
    #expect(tables.cos[6] == expectedCos1)
    #expect(tables.cos[7] == expectedCos1)
    #expect(tables.sin[6] == expectedSin1)
    #expect(tables.sin[7] == expectedSin1)
}

@Test func ropeTableBuilderHonorsFrequencyBaseOverride() {
    let tables = SmeltRoPETables.build(
        rowCount: 2,
        dim: 128,
        theta: 1_000_000,
        freqDim: 512
    )

    let pair = 16
    let exponent = Float(2 * pair) / 512.0
    let angle = Float(1.0 / pow(1_000_000.0, Double(exponent)))
    let expectedCos = Float16(Float(cos(Double(angle))))
    let expectedSin = Float16(Float(sin(Double(angle))))
    let d0 = 128 + pair * 2
    let d1 = d0 + 1

    #expect(tables.cos[d0] == expectedCos)
    #expect(tables.cos[d1] == expectedCos)
    #expect(tables.sin[d0] == expectedSin)
    #expect(tables.sin[d1] == expectedSin)
}

@Test func ropeTableBuilderSupportsSplitHalfLayout() {
    let tables = SmeltRoPETables.build(
        rowCount: 2,
        dim: 4,
        theta: 10_000,
        freqDim: nil,
        layout: "split_half"
    )

    let angle0 = Float(1.0)
    let angle1 = Float(1.0 / pow(10_000.0, 0.5))
    let expectedCos0 = Float16(Float(cos(Double(angle0))))
    let expectedSin0 = Float16(Float(sin(Double(angle0))))
    let expectedCos1 = Float16(Float(cos(Double(angle1))))
    let expectedSin1 = Float16(Float(sin(Double(angle1))))

    #expect(tables.cos[4] == expectedCos0)
    #expect(tables.cos[6] == expectedCos0)
    #expect(tables.sin[4] == expectedSin0)
    #expect(tables.sin[6] == expectedSin0)
    #expect(tables.cos[5] == expectedCos1)
    #expect(tables.cos[7] == expectedCos1)
    #expect(tables.sin[5] == expectedSin1)
    #expect(tables.sin[7] == expectedSin1)
}

@Test func ropeTableBuilderAppliesLlama3FrequencyScaling() {
    let scaling = SmeltRoPEScaling(
        type: .llama3,
        factor: 32,
        lowFreqFactor: 1,
        highFreqFactor: 4,
        originalMaxPositionEmbeddings: 8192
    )
    let scaled = SmeltRoPETables.build(
        rowCount: 4096,
        dim: 64,
        theta: 500_000,
        freqDim: nil,
        scaling: scaling,
        layout: "split_half"
    )
    let plain = SmeltRoPETables.build(
        rowCount: 4096,
        dim: 64,
        theta: 500_000,
        freqDim: nil,
        layout: "split_half"
    )

    #expect(scaled.cos[64] == plain.cos[64])
    #expect(scaled.sin[64] == plain.sin[64])

    let lowFrequencyPair = 31
    let invFreq = Float(pow(500_000.0, -Double(2 * lowFrequencyPair) / 64.0))
    let position = 4095
    let expectedScaledAngle = Float(position) * invFreq / 32.0
    let rowOffset = position * 64

    #expect(abs(Float(scaled.cos[rowOffset + lowFrequencyPair]) - cos(expectedScaledAngle)) < 0.001)
    #expect(abs(Float(scaled.sin[rowOffset + lowFrequencyPair]) - sin(expectedScaledAngle)) < 0.001)
    #expect(abs(Float(scaled.sin[rowOffset + lowFrequencyPair])
        - Float(plain.sin[rowOffset + lowFrequencyPair])) > 0.0001)
}

@Test func ropeTableResolutionFallsBackForLegacyManifest() {
    let manifest = SmeltManifest(
        modelName: "legacy/test",
        config: SmeltManifestConfig(
            hiddenSize: 1,
            numLayers: 1,
            vocabSize: 1,
            staticSeqCapacity: 4,
            ropeDim: 64,
            numDeltaLayers: 0,
            numAttnLayers: 1,
            attnQProjDim: 64,
            attnKProjDim: 64,
            attnVProjDim: 64,
            attnOutDim: 64,
            ffnDim: 1
        ),
        checksums: SmeltManifestChecksums(
            weightsBin: "",
            metallib: "",
            generatedSwift: "",
            dispatchesBin: ""
        ),
        device: SmeltDeviceRequirements(
            metalFamily: .apple7,
            minMemoryBytes: 1
        ),
        weights: SmeltWeightManifest(totalBytes: 0, entries: []),
        buffers: SmeltBufferTable(slots: []),
        pipelines: [],
        slotLayout: SmeltSlotLayout(
            convStateBaseSlot: 0,
            recStateBaseSlot: 0,
            keyCacheBaseSlot: 0,
            valCacheBaseSlot: 0,
            ropeCosSlot: 31,
            ropeSinSlot: 32,
            tokenIdSlot: 33,
            positionSlot: 34,
            weightsSlot: 30
        )
    )

    let pairs = SmeltRoPETables.resolvedPairs(from: manifest)
    #expect(pairs.count == 1)
    #expect(pairs[0].theta == 10_000)
    #expect(pairs[0].dim == 64)
    #expect(pairs[0].freqDim == nil)
    #expect(pairs[0].cosSlot == 31)
    #expect(pairs[0].sinSlot == 32)
}
