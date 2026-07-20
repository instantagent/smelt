import Foundation
import Testing
@testable import SmeltCompiler
import SmeltSchema

// MARK: - Pure suffix predicate

@Test func gptqScopeMatchesCanonicalProjections() {
    let inProjections = [
        "layers_0_self_attn_q_proj_weight",
        "layers_3_self_attn_k_proj_weight",
        "layers_7_self_attn_v_proj_weight",
        "layers_23_self_attn_o_proj_weight",
        "layers_0_mlp_gate_proj_weight",
        "layers_11_mlp_up_proj_weight",
        "layers_23_mlp_down_proj_weight",
    ]
    for name in inProjections {
        #expect(SmeltGPTQScope.isTransformerProjection(name), "should match: \(name)")
    }

    // Not attention/MLP projections: embeddings, head, norms, biases, and the
    // DeltaNet linear-attn projections (linear_attn_* / out_proj).
    let outProjections = [
        "embed_tokens",
        "lm_head_weight",
        "layers_0_input_layernorm_weight",
        "layers_3_self_attn_q_norm_weight",
        "layers_0_linear_attn_in_proj_qkv_weight",
        "layers_0_linear_attn_out_proj_weight",
        "layers_0_self_attn_q_proj_bias",
    ]
    for name in outProjections {
        #expect(!SmeltGPTQScope.isTransformerProjection(name), "should not match: \(name)")
    }
}

// MARK: - Authoritative affine_u4 decision

@Test func gptqScopeTracksTheRealQuantDecision() {
    let q = "layers_0_self_attn_q_proj_weight"
    let gate = "layers_0_mlp_gate_proj_weight"

    // Plain affine build: attn/MLP projections are in scope.
    let affine = SmeltQuantizationConfig(strategy: .affineU4, groupSize: 64, excludePatterns: [])
    #expect(SmeltGPTQScope.isInScope(name: q, quantization: affine))
    #expect(SmeltGPTQScope.isInScope(name: gate, quantization: affine))
    // Embeddings are not projections even when quantized to affine_u4.
    #expect(!SmeltGPTQScope.isInScope(
        name: "embed_tokens",
        quantization: SmeltQuantizationConfig(
            strategy: .affineU4, groupSize: 64, excludePatterns: [], quantizeEmbedding: true)))

    // Non-affine strategy: nothing is in scope (GPTQ produces affine_u4 blocks).
    let lut = SmeltQuantizationConfig(strategy: .lutU4, groupSize: 16, excludePatterns: [])
    #expect(!SmeltGPTQScope.isInScope(name: q, quantization: lut))

    // TQH routing wins over the affine layout dtype: a TQH-matched gate_proj is
    // NOT in scope, but a sibling q_proj still is. (The weight layout would
    // report both as affine_u4 — this is why scope keys on the real decision.)
    let tqh = SmeltQuantizationConfig(
        strategy: .affineU4, groupSize: 64, excludePatterns: [],
        turboQuantHPatterns: ["*_mlp_gate_proj_weight"])
    #expect(!SmeltGPTQScope.isInScope(name: gate, quantization: tqh))
    #expect(SmeltGPTQScope.isInScope(name: q, quantization: tqh))

    // excludePatterns drop a projection from scope.
    let excl = SmeltQuantizationConfig(
        strategy: .affineU4, groupSize: 64, excludePatterns: ["*_self_attn_q_proj_weight"])
    #expect(!SmeltGPTQScope.isInScope(name: q, quantization: excl))
    #expect(SmeltGPTQScope.isInScope(name: "layers_0_self_attn_o_proj_weight", quantization: excl))
}

// MARK: - Against a real resolved layout

@Test func gptqScopeValidatesAgainstRealLayout() {
    // qwen35_2B is a hybrid delta/attention model: 6× [delta, delta, delta, attn],
    // so attention layers (3,7,…,23) emit q/k/v/o and every layer emits MLP.
    let layout = SmeltWeightLayout.computeLayout(from: .qwen35_2B)
    let names = Set(layout.map(\.name))
    #expect(names.contains("layers_3_self_attn_q_proj_weight"))
    #expect(names.contains("layers_0_linear_attn_out_proj_weight"))

    // Under an affine build, scope is exactly the attn/MLP projection names.
    let affine = SmeltQuantizationConfig(
        strategy: .affineU4, groupSize: 64, excludePatterns: ["*_norm_weight"])
    let scoped = layout.map(\.name).filter { SmeltGPTQScope.isInScope(name: $0, quantization: affine) }
    #expect(scoped.allSatisfy { SmeltGPTQScope.isTransformerProjection($0) })
    #expect(scoped.contains("layers_3_self_attn_q_proj_weight"))
    #expect(scoped.contains("layers_3_self_attn_o_proj_weight"))
    #expect(scoped.contains("layers_0_mlp_down_proj_weight"))
    #expect(!scoped.contains { $0.contains("linear_attn") })
    #expect(!scoped.contains { $0.contains("_norm") })
    #expect(!scoped.contains("embed_tokens"))

    // Under the preset's actual lut_u4 strategy, GPTQ scope is empty.
    #expect(layout.map(\.name).allSatisfy {
        !SmeltGPTQScope.isInScope(name: $0, quantization: SmeltModelIR.qwen35_2B.quantization)
    })
}
