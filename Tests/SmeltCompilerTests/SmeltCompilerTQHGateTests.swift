import Foundation
import Testing
@testable import SmeltCompiler

// The affine_u4 Qwen3.5-2B golden (FixtureModelIRs.qwen35_2b_affine) carries a
// `turboQuantH` glob list; these pin the ensureCodegenSupport gate on it.

@Test
func tqhGateAcceptsTiedLMHeadWithEmbedTokensPattern() throws {
    let ir = FixtureModelIRs.qwen35_2b_affine(turboQuantH: ["embed_tokens"])
    #expect(ir.config.tiedLMHead == true)
    #expect(ir.quantization.turboQuantHPatterns == ["embed_tokens"])

    try SmeltCompiler.ensureCodegenSupport(for: ir)
}

@Test
func tqhGateAcceptsPerLayerOnly() throws {
    let ir = FixtureModelIRs.qwen35_2b_affine(turboQuantH: ["embed_tokens_per_layer"])
    #expect(ir.config.tiedLMHead == true)
    #expect(ir.quantization.turboQuantHPatterns == ["embed_tokens_per_layer"])

    try SmeltCompiler.ensureCodegenSupport(for: ir)
}

@Test
func tqhGateRejectsUnsupportedPattern() throws {
    let ir = FixtureModelIRs.qwen35_2b_affine(turboQuantH: ["lm_head_weight"])
    #expect(ir.quantization.turboQuantHPatterns == ["lm_head_weight"])

    do {
        try SmeltCompiler.ensureCodegenSupport(for: ir)
        Issue.record("ensureCodegenSupport unexpectedly accepted TQH-on-lm_head_weight")
    } catch let SmeltCompilerError.unsupportedRuntimeFeature(message) {
        #expect(
            message.contains("lm_head_weight") && message.contains("Supported today:"),
            Comment(rawValue: "expected supportedPatterns gate message, got: \(message)")
        )
    }
}
