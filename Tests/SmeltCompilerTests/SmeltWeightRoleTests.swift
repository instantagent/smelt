// SmeltWeightRoleTests — pin SmeltWeightRole.classify over the FULL canonical SmeltWeightPacker
// name inventory (dtype-building-blocks U2c). The safety-critical assertions: down_proj, norms,
// lm_head, and embeddings must NEVER classify as `.projection` (the only preserve-eligible role) —
// a misclassification there would preserve a weight whose runtime/kernel path can't take bf16/fp32.

import Testing
@testable import SmeltCompiler

private typealias Role = SmeltWeightRole

@Test func classifyEligibleProjections() {
    // Standard attention + FFN, layer-prefixed (gate_proj/up_proj are FFN-in projections).
    for name in ["layers_5_self_attn_q_proj_weight", "layers_5_self_attn_k_proj_weight",
                 "layers_5_self_attn_v_proj_weight", "layers_5_self_attn_o_proj_weight",
                 "layers_5_mlp_gate_proj_weight", "layers_5_mlp_up_proj_weight"] {
        #expect(Role.classify(name) == .projection, "expected .projection: \(name)")
        #expect(Role.classify(name).isNativePreserveEligible)
    }
    // Exotic linear projections — DeltaNet in/out, per-layer, and pre/post projections.
    for name in ["delta_3_out_proj_weight", "delta_3_in_proj_a_weight", "delta_3_in_proj_b_weight",
                 "delta_3_in_proj_qkv_weight", "delta_3_in_proj_z_weight",
                 "layers_2_per_layer_input_gate_weight", "layers_2_per_layer_projection_weight",
                 "per_layer_model_projection_weight", "pre_projection_weight",
                 "post_projection_weight"] {
        #expect(Role.classify(name) == .projection, "expected .projection: \(name)")
    }
}

@Test func classifyDownProjIsDeferredNotProjection() {
    // SAFETY-CRITICAL: down_proj must be .downProjection, NOT .projection — its fp32-output
    // range-protection path is fp16-weight-only, so it must never be preserved bf16/fp32.
    let name = "layers_5_mlp_down_proj_weight"
    #expect(Role.classify(name) == .downProjection)
    #expect(!Role.classify(name).isNativePreserveEligible)
}

@Test func classifyNonProjectionRolesAreNotEligible() {
    let cases: [(String, Role)] = [
        // Norms — incl. q/k norms, per-layer & projection norms, layernorms, the final norm.
        ("layers_5_self_attn_q_norm_weight", .norm),
        ("layers_5_self_attn_k_norm_weight", .norm),
        ("layers_5_input_layernorm_weight", .norm),
        ("layers_5_post_attention_layernorm_weight", .norm),
        ("layers_5_pre_feedforward_layernorm_weight", .norm),
        ("layers_5_post_feedforward_layernorm_weight", .norm),
        ("per_layer_projection_norm_weight", .norm),
        ("layers_2_post_per_layer_input_norm_weight", .norm),
        ("layers_5_norm_weight", .norm),
        ("norm_weight", .norm),
        // LM head + embeddings.
        ("lm_head_weight", .lmHead),
        ("masked_embedding_centroids_weight", .embedding),
        ("embed_tokens_weight", .embedding),
        ("embed_tokens_per_layer_weight", .embedding),
        // Auxiliary — DeltaNet state, conv, scalars, biases, quant sidecars.
        ("delta_3_A_log", .auxiliary),
        ("delta_3_conv1d_weight", .auxiliary),
        ("delta_3_dt_bias", .auxiliary),
        ("layers_5_layer_scalar", .auxiliary),
        ("layers_5_self_attn_q_proj_weight_scales", .auxiliary),
        ("layers_5_self_attn_q_proj_weight_biases", .auxiliary),
        ("layers_5_self_attn_q_proj_weight_lut", .auxiliary),
    ]
    for (name, expected) in cases {
        #expect(Role.classify(name) == expected, "\(name): expected \(expected)")
        #expect(!Role.classify(name).isNativePreserveEligible, "\(name) must NOT be eligible")
    }
}

@Test func onlyProjectionRoleIsEligible() {
    for role in Role.allCases {
        #expect(role.isNativePreserveEligible == (role == .projection))
    }
}

@Test func unknownNameIsConservativelyAuxiliary() {
    // A weight name no rule matches must default to .auxiliary (never preserve-eligible) — a new
    // packer weight can't silently become preserve-eligible.
    #expect(Role.classify("some_future_unknown_weight") == .auxiliary)
    #expect(!Role.classify("some_future_unknown_weight").isNativePreserveEligible)
}
