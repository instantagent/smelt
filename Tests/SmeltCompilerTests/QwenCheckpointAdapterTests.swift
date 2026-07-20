import Foundation
import Testing
@testable import SmeltCompiler
import SmeltSchema

@Test func qwenCheckpointAdapterMapsCanonicalTensorNames() {
    #expect(
        QwenCheckpointAdapter.mapName("model.layers.0.self_attn.q_proj.weight")
            == "layers_0_self_attn_q_proj_weight"
    )
    #expect(
        QwenCheckpointAdapter.mapName("model.layers.0.self_attn.q_proj.bias")
            == "layers_0_self_attn_q_proj_bias"
    )
    #expect(QwenCheckpointAdapter.mapName("model.embed_tokens.weight") == "embed_tokens")
    #expect(QwenCheckpointAdapter.mapName("model.norm.weight") == "norm_weight")
    #expect(QwenCheckpointAdapter.mapName("lm_head.weight") == "lm_head_weight")
}

@Test func qwenStrippedDecodeFusesEveryResidualProjectionAdd() throws {
    for ir in [SmeltModelIR.qwen35_0_8B, SmeltModelIR.qwen35_2B] {
        let plan = buildBufferPlan(from: ir)
        let layout = SmeltWeightLayout.computeLayout(from: ir)
        let full = try TopLevelEmitter.generate(
            ir: ir,
            plan: plan,
            weightLayout: layout,
            traceMode: .full
        )
        let stripped = try TopLevelEmitter.generate(
            ir: ir,
            plan: plan,
            weightLayout: layout,
            traceMode: .strippedMarkers
        )
        let elementwiseAdd = UInt16(SmeltPipeline.elementwiseAdd.rawValue)
        let fullDispatches = full.dispatchRecords.filter {
            $0.opKind == SmeltDispatchRecord.opDispatch
        }
        let strippedDispatches = stripped.dispatchRecords.filter {
            $0.opKind == SmeltDispatchRecord.opDispatch
        }

        #expect(fullDispatches.contains { $0.pipeline == elementwiseAdd })
        #expect(strippedDispatches.allSatisfy { $0.pipeline != elementwiseAdd })
        #expect(strippedDispatches.count < fullDispatches.count)
        #expect(stripped.traceMarkers.allSatisfy {
            !$0.label.hasSuffix(".delta_out") && !$0.label.hasSuffix(".ffn_down")
        })
    }
}

@Test func qwenAdapterRejectsUnsupportedMappedBiasTensors() throws {
    let ir = SmeltModelIR.qwen35_2B
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    var mappedNames = Set(layout.map(\.name))
    mappedNames.insert("layers_3_self_attn_q_proj_bias")
    mappedNames.insert("layers_3_self_attn_k_proj_bias")
    mappedNames.insert("layers_3_self_attn_v_proj_bias")

    #expect(throws: SmeltCompilerError.self) {
        try SmeltCompiler.rejectUnsupportedMappedBiasTensorsIfNeeded(
            mappedNames: mappedNames,
            expectedLayout: layout,
            adapter: .qwen
        )
    }
}
