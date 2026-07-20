// PreserveNativePatternsTests — the `preserve_native` quantization opt-in plumbing
// (dtype-building-blocks plan U2c, commit 1). preserve_native is a per-tensor glob list whose
// matched matvec-PROJECTION weights are kept at native bf16/fp32 instead of fp16-downcast. This
// commit only THREADS the field (config → manifest → fingerprint); no packer/quantizer
// consumes it yet, so these tests pin the LAYOUT consequence, not any byte behavior.

import Foundation
import Testing
@testable import SmeltCompiler

@Test func preserveNativeDefaultsEmpty() throws {
    let ir = FixtureModelIRs.qwen35_2b_affine()
    #expect(ir.quantization.preserveNativePatterns.isEmpty)
}

@Test func preserveNativeRejectsOverlapWithTurboQuantH() throws {
    // A weight cannot be both preserved-native AND TurboQuant-H quantized. The retired model-spec
    // DSL parser rejected this contradiction at parse time; validateSmeltIR now enforces it for
    // every authoring path, so the invariant moved to the universal validator rather than being
    // lost with the parser.
    let ir = FixtureModelIRs.qwen35_2b_affine(
        turboQuantH: ["*_q_proj_weight"], preserveNative: ["*_q_proj_weight"])
    #expect(throws: SmeltIRValidationError.self) {
        try validateSmeltIR(ir)
    }
}

@Test func preserveNativeLayoutTagsProjectionsBF16() throws {
    // The LAYOUT consequence: under affine_u4 strategy, a preserve_native glob flips the matched
    // projections to native .bf16 (the U2 dense kernel) while down_proj (deferred) and norms
    // (excluded) keep their non-bf16 dtype. Per-layer projections are prefixed, so the glob needs
    // a wildcard.
    let ir = FixtureModelIRs.qwen35_2b_affine(preserveNative: ["*_q_proj_weight", "*_k_proj_weight"])
    #expect(ir.config.activationDtype == .fp16)  // preserve_native only applies on the fp16-act path
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    func dtypes(suffix: String) -> Set<SmeltDType> {
        Set(layout.filter { $0.name.hasSuffix(suffix) }.map(\.dtype))
    }
    // Matched projections → preserved native bf16.
    #expect(dtypes(suffix: "_q_proj_weight") == [.bf16])
    #expect(dtypes(suffix: "_k_proj_weight") == [.bf16])
    // UNmatched projections stay affine-quantized (the default strategy still applies).
    #expect(dtypes(suffix: "_v_proj_weight") == [.affineU4])
    // down_proj is DEFERRED (never preserved) — stays affine, never bf16.
    let down = dtypes(suffix: "_down_proj_weight")
    #expect(!down.contains(.bf16) && !down.isEmpty)
    // norms are excluded-from-quant → fp16, never bf16.
    #expect(!dtypes(suffix: "_norm_weight").contains(.bf16))
}

@Test func preserveNativeLayoutNoOpWhenUnset() throws {
    // With no preserve_native, the layout is unchanged — projections quantize per strategy, nothing
    // is bf16 (the fp16-act path has no native-bf16 entries by default). Pins zero behavior drift.
    let ir = FixtureModelIRs.qwen35_2b_affine()
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    #expect(!layout.contains { $0.name.hasSuffix("_q_proj_weight") && $0.dtype == .bf16 })
}
