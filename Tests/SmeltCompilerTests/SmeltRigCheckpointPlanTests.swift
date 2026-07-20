import Foundation
import Testing

@testable import SmeltCompiler
@testable import SmeltRuntime

@Suite("Smelt rig checkpoint plan")
struct SmeltRigCheckpointPlanTests {
    @Test("Pinned inventory is complete and exclusions are explicit")
    func pinnedInventoryIsComplete() {
        let expected = SmeltRigCheckpointPlan.expectedTensors
        #expect(expected.count == 672)
        #expect(Set(expected.map(\.name)).count == expected.count)
        #expect(expected.filter { $0.disposition == .carried }.count == 622)
        #expect(expected.filter { $0.disposition == .trainingOnly }.count == 50)
        #expect(expected.filter { $0.component == .vae }.count == 252)
        #expect(expected.filter { $0.component == .meshEncoder }.count == 106)
        #expect(expected.filter { $0.component == .qwen }.count == 311)
        #expect(expected.filter { $0.component == .outputProjection }.count == 3)
    }

    @Test("Canonical checkpoint has exact carried coverage")
    func canonicalCheckpointHasExactCoverage() throws {
        guard let path = ProcessInfo.processInfo.environment["SMELT_RIG_CHECKPOINT"] else {
            return
        }
        let checkpoint = try PyTorchCheckpointLoader(path: path)
        let plan = try SmeltRigCheckpointPlan(checkpoint: checkpoint)
        #expect(plan.tensors.count == 672)
        #expect(plan.carriedTensors.count == 622)
        #expect(plan.trainingOnlyTensors.count == 50)
        #expect(plan.carriedTensors.allSatisfy { $0.byteCount > 0 })
        #expect(
            plan.trainingOnlyTensors.map(\.name).contains(
                "vae.model.encoder.learned_queries"
            )
        )
        #expect(
            plan.trainingOnlyTensors.map(\.name).contains(
                "vae.model.FSQ.project_in.weight"
            )
        )
    }
}
