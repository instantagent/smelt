import Foundation
import SmeltModels
import Testing

@testable import SmeltCompiler
@testable import SmeltRuntime

@Suite("Smelt checkpoint tensor plan")
struct SmeltCheckpointTensorPlanTests {
    @Test("CAM owns the complete inventory and explicit exclusions")
    func authoredInventoryIsComplete() throws {
        let module = try #require(SmeltModels.definition(id: "skintokens_articulation"))
        #expect(module.tensors.count == 672)
        #expect(Set(module.tensors.map { $0.selector.pattern }).count == 672)
        #expect(module.tensors.filter { $0.disposition == .carried }.count == 622)
        #expect(module.tensors.filter { $0.disposition == .trainingOnly }.count == 50)
        #expect(module.tensors.filter { $0.owner == "skin-vae" }.count == 252)
        #expect(module.tensors.filter { $0.owner == "mesh-encoder" }.count == 106)
        #expect(module.tensors.filter { $0.owner == "language" }.count == 314)
        #expect(Set(module.tensors.compactMap(\.layoutOrdinal)).count == 672)
    }

    @Test("Canonical checkpoint has exact generic CAM coverage")
    func canonicalCheckpointHasExactCoverage() throws {
        guard let path = ProcessInfo.processInfo.environment["SMELT_SKINNING_CHECKPOINT"] else {
            return
        }
        let checkpoint = try PyTorchCheckpointLoader(path: path)
        let module = try #require(SmeltModels.definition(id: "skintokens_articulation"))
        let plan = try SmeltCheckpointTensorPlan(
            module: module,
            checkpoints: ["checkpoint": checkpoint]
        )
        #expect(plan.tensors.count == 672)
        #expect(plan.carriedTensors.count == 622)
        #expect(plan.omittedTensors.count == 50)
        #expect(plan.tensors.allSatisfy { $0.dtype == "BF16" })
        #expect(plan.carriedTensors.allSatisfy { $0.byteCount > 0 })
        #expect(plan.omittedTensors.map(\.name).contains("vae.model.encoder.learned_queries"))
        #expect(plan.omittedTensors.map(\.name).contains("vae.model.FSQ.project_in.weight"))
    }
}
