// PreserveNativeCodegenTests — the CODEGEN consequence of preserve_native (dtype-building-blocks
// plan U2c.5). PreserveNativePatternsTests pins the LAYOUT (a matched projection's entry is tagged
// .bf16); this file pins the next link: a .bf16 projection entry COMPILES to the fp16_matvec_bf16w
// dense kernel (the U2 lego), a non-preserved projection keeps its own kernel (selective lifting),
// and the preserve_native glob list survives manifest provenance round-trip.
//
// The U2 thesis is "a dtype selects a kernel LEGO, it never gates what compiles." The existing U2b
// gate (emitMatvecAndVarNowEmitBF16FP32Dense) only proves bf16/fp32 EMIT something; here we pin the
// SPECIFIC pipeline each dtype routes to, so a regression that downcasts bf16→fp16 (or routes it to
// the wrong kernel) fails loudly.

import Foundation
import Testing
@testable import SmeltCompiler
@testable import SmeltSchema

private func denseEntry(_ name: String, _ dtype: SmeltDType) -> SmeltWeightEntry {
    SmeltWeightEntry(name: name, offset: 0, sizeBytes: 64 * 64 * 2, shape: [64, 64], dtype: dtype)
}

private func pipelineStateLine(_ pipeline: SmeltPipeline) -> String {
    "setComputePipelineState(p[\(pipeline.rawValue)])"
}

/// The U2 thesis at codegen: a dense matvec weight's dtype picks its kernel lego — bf16 → the
/// fp16_matvec_bf16w kernel, fp32 → fp16_matvec_fp32w, fp16 → fp16_matvec — for BOTH the fixed and
/// variable-slot emit paths. (A non-preserved down_proj that stays fp16 routes via the .fp16 case.)
@Test func denseProjectionDtypeSelectsItsKernelLego() throws {
    let cases: [(dtype: SmeltDType, pipeline: SmeltPipeline)] = [
        (.bf16, .fp16MatvecBF16W),
        (.fp32, .fp16MatvecFP32W),
        (.fp16, .fp16Matvec),
    ]
    for (dtype, pipeline) in cases {
        var fixed = SmeltCodeEmitter()
        let fixedLines = try fixed.emitMatvec(
            weightEntry: denseEntry("w_\(dtype.rawValue)", dtype),
            weightsSlot: 30, inputSlot: 0, outputSlot: 1, rows: 64, cols: 64, groupSize: 32)
        #expect(fixedLines.joined(separator: "\n").contains(pipelineStateLine(pipeline)),
                "emitMatvec(\(dtype)) must route to \(pipeline)")

        var variable = SmeltCodeEmitter()
        let varLines = try variable.emitMatvecVar(
            weightEntry: denseEntry("w_\(dtype.rawValue)", dtype),
            weightsSlot: 30, inputSlotVar: "cur", outputSlot: 1, rows: 64, cols: 64, groupSize: 32)
        #expect(varLines.joined(separator: "\n").contains(pipelineStateLine(pipeline)),
                "emitMatvecVar(\(dtype)) must route to \(pipeline)")
    }
}

/// End-to-end: a preserve_native build tags the matched projection .bf16 in the layout, and that
/// layout entry COMPILES to fp16_matvec_bf16w — while an unmatched projection keeps its strategy
/// kernel (selective lifting: only the matched glob is lifted to the bf16 lego, not the whole model).
@Test func preserveNativeBuildCompilesProjectionToBF16WKernel() throws {
    let ir = FixtureModelIRs.qwen35_2b_affine(preserveNative: ["*_q_proj_weight"])
    #expect(ir.config.activationDtype == .fp16)  // preserve_native only applies on the fp16-act path
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    // The matched projection is preserved native bf16; an unmatched one stays affine-quantized.
    let qProj = try #require(layout.first { $0.name.hasSuffix("_q_proj_weight") })
    let vProj = try #require(layout.first { $0.name.hasSuffix("_v_proj_weight") })
    #expect(qProj.dtype == .bf16)
    #expect(vProj.dtype == .affineU4)  // selective lifting — NOT bf16

    // The preserved .bf16 layout entry compiles to the bf16-weight dense kernel.
    var emitter = SmeltCodeEmitter()
    let lines = try emitter.emitMatvec(
        weightEntry: qProj, weightsSlot: 30, inputSlot: 0, outputSlot: 1,
        rows: qProj.shape[0], cols: qProj.shape[1], groupSize: 64)
    #expect(lines.joined(separator: "\n").contains(pipelineStateLine(.fp16MatvecBF16W)))
    // And NOT the plain fp16 kernel — proving the bytes are read as bf16, not downcast.
    #expect(!lines.joined(separator: "\n").contains(pipelineStateLine(.fp16Matvec)))
}

/// Manifest provenance: the preserve_native glob list survives a resolved-build-options round-trip
/// (so a built package records WHY a projection is bf16), and a legacy manifest lacking the key
/// decodes to an empty list (the decodeIfPresent backward-compat U2c.1 added).
@Test func preserveNativePatternsRoundTripThroughResolvedBuildOptions() throws {
    let options = SmeltResolvedBuildOptions(
        layerPatternUnit: ["attn"], layerPatternRepeats: 1,
        quantizationStrategy: "affine_u4", groupSize: 64,
        excludePatterns: ["*_norm_weight"], quantizeEmbedding: true,
        loadingStrategy: "mmap_prefault", packing: "monolithic",
        prefillEngine: "metal", maxPrefillBatch: 256, prefillHandoffFamilies: [],
        inferenceMaxTokens: 512, eosTokens: [1], thinkToken: nil, thinkEndToken: nil,
        thinkSkipSuffix: nil, tiedLMHead: true, traceMode: "full",
        turboQuantHPatterns: [], preserveNativePatterns: ["*_q_proj_weight", "*_k_proj_weight"])

    let data = try JSONEncoder().encode(options)
    let decoded = try JSONDecoder().decode(SmeltResolvedBuildOptions.self, from: data)
    #expect(decoded.preserveNativePatterns == ["*_q_proj_weight", "*_k_proj_weight"])
    #expect(decoded.compilationGeneratedKernels == "auto")
    #expect(decoded.compilationWeightLayout == "memory_neutral")

    // A legacy manifest without the key → empty list, not a decode failure.
    var obj = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    obj.removeValue(forKey: "preserveNativePatterns")
    obj.removeValue(forKey: "compilationGeneratedKernels")
    obj.removeValue(forKey: "compilationWeightLayout")
    let legacy = try JSONSerialization.data(withJSONObject: obj)
    let legacyDecoded = try JSONDecoder().decode(SmeltResolvedBuildOptions.self, from: legacy)
    #expect(legacyDecoded.preserveNativePatterns.isEmpty)
    #expect(legacyDecoded.compilationGeneratedKernels == "auto")
    #expect(legacyDecoded.compilationWeightLayout == "memory_neutral")
}
