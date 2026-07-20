import Testing

@testable import SmeltRuntime

@Suite("Qwen3.5 vision frozen component plan")
struct Qwen35VisionFrozenPlanTests {
    @Test("Component lowers its selected route to generic frozen operations")
    func genericFrozenRoute() throws {
        let config = SmeltQwen35VisionConfig(
            hiddenSize: 8,
            layerCount: 1,
            headCount: 2,
            headDim: 4,
            intermediateSize: 16,
            layerNormEpsilon: 1e-6,
            inChannels: 3,
            patchSize: 2,
            temporalPatchSize: 2,
            positionEmbeddingCount: 16,
            spatialMergeSize: 2,
            outputHiddenSize: 12,
            activation: "gelu_pytorch_tanh"
        )
        let plan = try SmeltQwen35VisionRuntime.frozenPlan(
            config: config,
            grids: [.init(temporal: 1, height: 2, width: 2)],
            provenanceKey: "fixture:vision-component"
        )

        #expect(plan.planID == "component-dense-vision")
        #expect(plan.provenanceKey == "fixture:vision-component")
        #expect(plan.context.sequenceLength == 4)
        #expect(plan.records.count == 29)
        #expect(plan.records.allSatisfy { $0.isDescribed })
        #expect(plan.records.map(\.recordIndex) == Array(0..<29))
        #expect(plan.records.map(\.dispatchOrdinal) == (1...29).map(Optional.some))

        let groups = Dictionary(grouping: plan.records) {
            $0.operationGroup ?? "unknown"
        }.mapValues(\.count)
        #expect(groups["dense.matmul"] == 7)
        #expect(groups["attention.matmul"] == 4)
        #expect(groups["attention.softmax"] == 2)
        #expect(groups["elementwise.bias"] == 7)
        #expect(groups["elementwise.add"] == 3)
        #expect(groups["normalization.layer"] == 3)
        #expect(groups["attention.qkv_rope_split"] == 1)
        #expect(groups["activation.gelu_tanh"] == 2)
    }
}
