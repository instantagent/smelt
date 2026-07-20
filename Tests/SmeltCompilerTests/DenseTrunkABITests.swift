import XCTest
@testable import SmeltCompiler
@testable import SmeltRuntime
import SmeltSchema

/// W0.1 (docs/talker-trunk-fit-audit.md): the dense trunk ABI at the
/// buffer-planning layer. Port topology fixes layout; BF16 and FP32 select
/// storage size and operation cells within the same layout.
final class DenseTrunkABITests: XCTestCase {

    private func denseIR(
        activationDtype: SmeltDType, prefill: SmeltPrefillConfig? = nil
    ) -> SmeltModelIR {
        let attention = SmeltAttentionConfig(
            qHeads: 16, kvHeads: 8, headDim: 128, gatedQ: false,
            qkNorm: true, qkNormMode: .weight,
            attnScale: 1, ropeTheta: 1_000_000
        )
        let config = SmeltConfig(
            hiddenSize: 2048, numLayers: 2, vocabSize: 3072,
            staticSeqCapacity: 256,
            ropeDim: 128, rmsEps: 1e-6, normMode: .weight,
            activationDtype: activationDtype,
            portTopology: .embeddingsInHiddenOut,
            attention: attention,
            ffn: SmeltFFNConfig(dim: 6144, activation: .swiglu),
            tiedLMHead: false
        )
        return SmeltModelIR(
            modelName: "dense-trunk-abi-test",
            config: config,
            layerPattern: SmeltLayerPattern(unit: [.attention], repeats: 2),
            quantization: SmeltModelIR.qwen35_2B.quantization,
            loading: SmeltModelIR.qwen35_2B.loading,
            prefill: prefill
        )
    }

    private func metalPrefill(maxBatch: Int) -> SmeltPrefillConfig {
        SmeltPrefillConfig(
            engine: "metal", modelPath: "", maxBatchSize: maxBatch,
            handoffFamilies: ["key_cache", "value_cache"])
    }

    private func slot(_ plan: SmeltBufferPlan, _ name: String) -> PlannedSlot? {
        plan.slots.first { $0.name == name }
    }

    func testActivationSlotsFollowABI() {
        let bf16Plan = buildBufferPlan(from: denseIR(activationDtype: .bf16))
        let fp32Plan = buildBufferPlan(from: denseIR(activationDtype: .fp32))
        for name in ["hiddenA", "normOutBuf", "ffnGateBuf", "residualBuf", "attnQBuf"] {
            let half = slot(bf16Plan, name)!
            let full = slot(fp32Plan, name)!
            XCTAssertEqual(half.dtype, .bf16, name)
            XCTAssertEqual(full.dtype, .fp32, name)
            XCTAssertEqual(full.sizeBytes, half.sizeBytes * 2,
                           "\(name): fp32 slot must be exactly double")
        }
    }

    func testKVCacheFollowsABIAndDeclaresRowContiguousLayout() {
        let bf16Plan = buildBufferPlan(from: denseIR(activationDtype: .bf16))
        let fp32Plan = buildBufferPlan(from: denseIR(activationDtype: .fp32))
        let halfK = slot(bf16Plan, "keyCache_0")!
        let fullK = slot(fp32Plan, "keyCache_0")!
        XCTAssertEqual(halfK.dtype, .bf16)
        XCTAssertEqual(fullK.dtype, .fp32)
        XCTAssertEqual(fullK.sizeBytes, halfK.sizeBytes * 2)
        // Both storage families retain the topology's row-contiguous cache.
        XCTAssertEqual(halfK.shape.count, 2)
        XCTAssertEqual(fullK.shape.count, 2)
        XCTAssertEqual(halfK.shape, fullK.shape)
        XCTAssertEqual(fullK.shape.last, 8 * 128)
        let fullV = slot(fp32Plan, "valCache_0")!
        XCTAssertEqual(fullV.dtype, .fp32)
        XCTAssertEqual(fullV.shape.count, 2)
    }

    func testExplicitDtypeSlotsAreUnchangedByABI() {
        let fp32Plan = buildBufferPlan(from: denseIR(activationDtype: .fp32))
        // int32/raw scratch and always-fp32 numerics buffers keep their
        // explicit dtypes regardless of the trunk ABI.
        XCTAssertEqual(slot(fp32Plan, "argmaxBuf")!.dtype, .raw)
        XCTAssertEqual(slot(fp32Plan, "normScaleScratch")!.dtype, .raw)
    }

    func testRoPETablesFollowABI() {
        let bf16Plan = buildBufferPlan(from: denseIR(activationDtype: .bf16))
        let fp32Plan = buildBufferPlan(from: denseIR(activationDtype: .fp32))
        XCTAssertEqual(slot(bf16Plan, "ropeCos")!.dtype, .bf16)
        XCTAssertEqual(slot(fp32Plan, "ropeCos")!.dtype, .fp32)
        XCTAssertEqual(slot(fp32Plan, "ropeSin")!.dtype, .fp32)
        XCTAssertEqual(slot(fp32Plan, "ropeCos")!.sizeBytes,
                       slot(bf16Plan, "ropeCos")!.sizeBytes * 2)
    }

    func testF32RoPEValuesAreThePreCastFloats() {
        // buildF32 must produce the Float values BEFORE the half cast —
        // parity with the talker hand path's pure-Float tables.
        let f32 = SmeltRoPETables.buildF32(
            rowCount: 4, dim: 8, theta: 1_000_000, freqDim: nil,
            layout: "split_half")
        let f16 = SmeltRoPETables.build(
            rowCount: 4, dim: 8, theta: 1_000_000, freqDim: nil,
            layout: "split_half")
        XCTAssertEqual(f32.cos.count, f16.cos.count)
        for i in 0..<f32.cos.count {
            XCTAssertEqual(Float16(f32.cos[i]), f16.cos[i], "cos[\(i)] half-cast mismatch")
            XCTAssertNotEqual(f32.cos[i].isNaN, true)
        }
    }

    func testF32ValidationGates() {
        // Graph topology and dtype are independent axes. The dense FP32 cell
        // is registered; the dense FP16 cell is still missing and must fail
        // loudly instead of silently selecting the token/logits graph.
        XCTAssertThrowsError(try validateSmeltIR(denseIR(activationDtype: .fp16)))
        XCTAssertNoThrow(try validateSmeltIR(denseIR(activationDtype: .fp32)))
    }

    func testF32PrefillValidationGuards() {
        // W2 lifted the fp32+prefill rejection — but only for a metal-native,
        // 1..2048-token chunk (B3.2b's causal_gqa_attn_cached_f32 threadgroup cap).
        XCTAssertNoThrow(try validateSmeltIR(
            denseIR(activationDtype: .fp32, prefill: metalPrefill(maxBatch: 256))))
        XCTAssertNoThrow(try validateSmeltIR(
            denseIR(activationDtype: .fp32, prefill: metalPrefill(maxBatch: 2048))))
        // Nonpositive batch (plan multiplier vs runtime clamp-to-1 mismatch).
        XCTAssertThrowsError(try validateSmeltIR(
            denseIR(activationDtype: .fp32, prefill: metalPrefill(maxBatch: 0))))
        // Over the cached-attn threadgroup score cap.
        XCTAssertThrowsError(try validateSmeltIR(
            denseIR(activationDtype: .fp32, prefill: metalPrefill(maxBatch: 2049))))
        // CoreML prefill has no fp32 path.
        XCTAssertThrowsError(try validateSmeltIR(denseIR(
            activationDtype: .fp32,
            prefill: SmeltPrefillConfig(
                engine: "coreml", modelPath: "prefill.mlmodelc", maxBatchSize: 64,
                handoffFamilies: ["key_cache"]))))
    }
}
