// Dense headless-trunk prefill lowering tests.
import XCTest
@testable import SmeltCompiler
@testable import SmeltSchema

/// The dense-trunk prefill table —
/// Record-level assertions for the registered unfused M>1 sequence (kernels,
/// runtime-seqLen
/// grids, startPos-strided cache-row writes, embeddings-in / hidden-out), and
/// that it shares the decode table's dense-topology guards. The bit-exact
/// validation against real weights is the W2 parity gate.
final class DenseTrunkPrefillEmitterTests: XCTestCase {

    private let layers = 2
    private let hidden = 2048, heads = 16, kvHeads = 8, headDim = 128, inter = 6144

    private func denseAttention(
        ropeLayout: SmeltRoPELayout? = .splitHalf
    ) -> SmeltAttentionConfig {
        SmeltAttentionConfig(
            qHeads: heads, kvHeads: kvHeads, headDim: headDim, gatedQ: false,
            qkNorm: true, qkNormMode: .weight, vNorm: false,
            attnScale: 1, ropeTheta: 1_000_000, ropeDim: nil,
            ropeLayout: ropeLayout, slidingWindow: 0, externalKV: false)
    }

    private func ir(attention: SmeltAttentionConfig? = nil, maxPrefillBatch: Int = 256) -> SmeltModelIR {
        let attention = attention ?? denseAttention()
        let config = SmeltConfig(
            hiddenSize: hidden, numLayers: layers, vocabSize: 3072,
            staticSeqCapacity: 64,
            ropeDim: headDim, rmsEps: 1e-6, normMode: .weight,
            activationDtype: .fp32,
            portTopology: .embeddingsInHiddenOut,
            attention: attention,
            ffn: SmeltFFNConfig(dim: inter, activation: .swiglu),
            tiedLMHead: false)
        return SmeltModelIR(
            modelName: "f32-trunk-prefill-test",
            config: config,
            layerPattern: SmeltLayerPattern(unit: [.attention], repeats: layers),
            quantization: SmeltModelIR.qwen35_2B.quantization,
            loading: SmeltModelIR.qwen35_2B.loading,
            prefill: SmeltPrefillConfig(
                engine: "metal", modelPath: "", maxBatchSize: maxPrefillBatch,
                handoffFamilies: ["key_cache", "value_cache"]))
    }

    private func fabricatedLayout(projDtype: SmeltDType = .bf16) -> [SmeltWeightEntry] {
        var entries: [SmeltWeightEntry] = []
        var offset: UInt64 = 0x1000
        func add(_ name: String, _ dtype: SmeltDType) {
            entries.append(SmeltWeightEntry(
                name: name, offset: offset, sizeBytes: 4096, shape: [1], dtype: dtype))
            offset += 0x1000
        }
        for l in 0..<layers {
            let p = "layers_\(l)"
            for w in ["q", "k", "v", "o"] { add("\(p)_self_attn_\(w)_proj_weight", projDtype) }
            for w in ["gate", "up", "down"] { add("\(p)_mlp_\(w)_proj_weight", projDtype) }
            add("\(p)_input_layernorm_weight", .fp32)
            add("\(p)_post_attention_layernorm_weight", .fp32)
            add("\(p)_self_attn_q_norm_weight", .fp32)
            add("\(p)_self_attn_k_norm_weight", .fp32)
        }
        add("norm_weight", .fp32)
        return entries
    }

    private func generate(
        attention: SmeltAttentionConfig? = nil, projDtype: SmeltDType = .bf16,
        maxPrefillBatch: Int = 256
    ) throws -> ([SmeltDispatchRecord], SmeltBufferPlan) {
        let model = ir(attention: attention, maxPrefillBatch: maxPrefillBatch)
        let plan = buildBufferPlan(from: model)
        let result = try DenseTrunkPrefillEmitter.generate(
            ir: model, plan: plan, weightLayout: fabricatedLayout(projDtype: projDtype))
        return (result.dispatchRecords, plan)
    }

    func testEmitsTheUnfusedPrefillSequence() throws {
        let (records, _) = try generate()
        // Per layer: norm, q/k/v gemm, q/k head-norm, q/k rope, attn, o gemm,
        // residual, SWAP, post-norm, gate/up gemm, swiglu, down gemm, residual,
        // SWAP = 17 dispatches + 2 swaps; plus the final norm.
        let perLayer: [SmeltPipeline] = [
            .rmsNormCodecF32, .gemmBF16WF32, .gemmBF16WF32, .gemmBF16WF32,
            .rmsNormHeadF32, .rmsNormHeadF32, .ropeApplyF32, .ropeApplyF32,
            .causalGQAAttnCachedF32, .gemmBF16WF32, .scaleResidualTCF32,
            .rmsNormCodecF32, .gemmBF16WF32, .gemmBF16WF32, .swigluF32,
            .gemmBF16WF32, .scaleResidualTCF32,
        ]
        let dispatched = records
            .filter { $0.opKind == SmeltDispatchRecord.opDispatch }
            .map { Int($0.pipeline) }
        var expected: [Int] = []
        for _ in 0..<layers { expected += perLayer.map(\.rawValue) }
        expected.append(SmeltPipeline.rmsNormCodecF32.rawValue)
        XCTAssertEqual(dispatched, expected)

        XCTAssertEqual(records.count, layers * 19 + 1)
        let swapIndices = records.enumerated()
            .filter { $0.element.opKind == SmeltDispatchRecord.opSwap }.map(\.offset)
        XCTAssertEqual(swapIndices, (0..<layers).flatMap { [$0 * 19 + 11, $0 * 19 + 18] })
        XCTAssertEqual(records.last!.pipeline, UInt16(SmeltPipeline.rmsNormCodecF32.rawValue))
    }

    /// Phase 4 U2a: the prefill table emits a GPTQ capture point for every projection, with the
    /// FP32 dtype (the trunk activation ports are fp32, not the LLM fp16 ABI) and a dispatchCount
    /// that points AT that projection's gemm in the verbatim table (no optimizer reorders it), so
    /// SmeltRuntime.captureGPTQActivations reads the right [seqLen, k] input.
    func testEmitsGPTQCapturePointsForEveryProjection() throws {
        let model = ir()
        let plan = buildBufferPlan(from: model)
        let result = try DenseTrunkPrefillEmitter.generate(
            ir: model, plan: plan, weightLayout: fabricatedLayout(projDtype: .bf16))
        let points = result.gptqCapturePoints

        XCTAssertEqual(points.count, 7 * layers, "one capture point per projection per layer")
        XCTAssertTrue(points.allSatisfy { !$0.inputIsFloat16 }, "trunk capture input is FP32")
        let suffixes = ["q_proj_weight", "k_proj_weight", "v_proj_weight", "o_proj_weight",
                        "gate_proj_weight", "up_proj_weight", "down_proj_weight"]
        XCTAssertTrue(points.allSatisfy { p in suffixes.contains { p.weightName.hasSuffix($0) } },
                      "every point names a GPTQ-scope projection")

        // Non-swap gemm positions per layer (from testEmitsTheUnfusedPrefillSequence): q=2 k=3 v=4
        // o=10 gate=13 up=14 down=16, with 17 non-swap dispatches per layer.
        let expected = (0..<layers).flatMap { l in [2, 3, 4, 10, 13, 14, 16].map { l * 17 + $0 } }.sorted()
        XCTAssertEqual(points.map(\.dispatchCount).sorted(), expected)

        // Cross-check: the dispatchCount-th non-swap record IS a gemm reading the point's inputSlot.
        var byBoundary: [Int: SmeltDispatchRecord] = [:]
        var nonSwap = 0
        for rec in result.dispatchRecords where rec.opKind != SmeltDispatchRecord.opSwap {
            nonSwap += 1; byBoundary[nonSwap] = rec
        }
        for p in points {
            guard let rec = byBoundary[p.dispatchCount] else {
                return XCTFail("no dispatch at boundary \(p.dispatchCount) for \(p.weightName)")
            }
            XCTAssertEqual(rec.pipeline, UInt16(SmeltPipeline.gemmBF16WF32.rawValue),
                           "boundary \(p.dispatchCount) (\(p.weightName)) is not a gemm")
            XCTAssertEqual(Int(getBuffer(rec, index: 0).slot), p.inputSlot,
                           "capture \(p.weightName) inputSlot \(p.inputSlot) != the gemm's x slot")
        }
        // down_proj reads the FFN intermediate (k = inter); q_proj reads hidden.
        XCTAssertEqual(points.first { $0.weightName.hasSuffix("down_proj_weight") }!.k, inter)
        XCTAssertEqual(points.first { $0.weightName.hasSuffix("q_proj_weight") }!.k, hidden)
    }

    func testCacheWritesAndRopeAreStartPosStrided() throws {
        let (records, plan) = try generate()
        let kvRowStride = UInt64(kvHeads * headDim * 4)
        let gemms = records.filter { $0.pipeline == UInt16(SmeltPipeline.gemmBF16WF32.rawValue) }
        // Layer-0 gemms in order: q, k, v, o, gate, up, down — v writes the cache.
        let vOut = getBuffer(gemms[2], index: 3)
        XCTAssertEqual(Int(vOut.slot), plan.valCacheBaseSlot)
        XCTAssertEqual(vOut.offsetKind, 1)
        XCTAssertEqual(vOut.offset, kvRowStride)
        // q/o gemms write plain slots (not the cache).
        XCTAssertEqual(getBuffer(gemms[0], index: 3).offsetKind, 0)
        XCTAssertEqual(getBuffer(gemms[3], index: 3).offsetKind, 0)

        let ropes = records.filter { $0.pipeline == UInt16(SmeltPipeline.ropeApplyF32.rawValue) }
        // Layer-0 ropes: q (plain), k (→ cache).
        let kOut = getBuffer(ropes[1], index: 3)
        XCTAssertEqual(Int(kOut.slot), plan.keyCacheBaseSlot)
        XCTAssertEqual(kOut.offsetKind, 1)
        XCTAssertEqual(kOut.offset, kvRowStride)
        XCTAssertEqual(getBuffer(ropes[0], index: 3).offsetKind, 0)
        // rope cos/sin rows: startPos-strided by headDim*4 (absolute pos = startPos + local t).
        let cos = getBuffer(ropes[1], index: 1)
        XCTAssertEqual(Int(cos.slot), plan.ropeCosSlot)
        XCTAssertEqual(cos.offsetKind, 1)
        XCTAssertEqual(cos.offset, UInt64(headDim * 4))
    }

    func testRuntimeSeqLenGridsAndKeyConstants() throws {
        let (records, _) = try generate()
        // rms_norm_codec: width = seqLen*32, frames constant = __seqLen__.
        let norm = records.first { $0.pipeline == UInt16(SmeltPipeline.rmsNormCodecF32.rawValue) }!
        XCTAssertEqual(norm.gridWKind, SmeltDispatchRecord.gridSeqLenMulLiteral)
        XCTAssertEqual(norm.gridW, 32)
        XCTAssertEqual(getConstant(norm, index: 0).kind, SmeltConstantRecord.kindSeqLen)
        // gemm: height = seqLen (runtime M), M constant = __seqLen__.
        let gemm = records.first { $0.pipeline == UInt16(SmeltPipeline.gemmBF16WF32.rawValue) }!
        XCTAssertEqual(gemm.gridHKind, SmeltDispatchRecord.gridSeqLen)
        XCTAssertEqual(getConstant(gemm, index: 0).kind, SmeltConstantRecord.kindSeqLen)
        XCTAssertEqual(getConstant(gemm, index: 3).kind, SmeltConstantRecord.kindLiteralU32, "has_bias")
        XCTAssertEqual(getConstant(gemm, index: 3).value, 0)
        // residual: has_scale = 0 → out = residual + x (bit-identical to 1.0*x).
        let resid = records.first { $0.pipeline == UInt16(SmeltPipeline.scaleResidualTCF32.rawValue) }!
        XCTAssertEqual(getBuffer(resid, index: 1).slot, SmeltBufferRecord.slotCur)
        XCTAssertEqual(getBuffer(resid, index: 3).slot, SmeltBufferRecord.slotAlt)
        let hasScale = getConstant(resid, index: 2)
        XCTAssertEqual(hasScale.kind, SmeltConstantRecord.kindLiteralU32)
        XCTAssertEqual(hasScale.value, 0)
        // swiglu: count = __seqLen__ * inter.
        let swiglu = records.first { $0.pipeline == UInt16(SmeltPipeline.swigluF32.rawValue) }!
        let count = getConstant(swiglu, index: 0)
        XCTAssertEqual(count.kind, SmeltConstantRecord.kindSeqLenMulLiteral)
        XCTAssertEqual(count.value, UInt32(inter))
    }

    func testBindingCountsMatchTheCatalogSignatures() throws {
        let (records, _) = try generate()
        for rec in records where rec.opKind == SmeltDispatchRecord.opDispatch {
            let sig = SmeltKernelCatalog.signatures[Int(rec.pipeline)]
            XCTAssertEqual(Int(rec.bufferCount), sig.bufferBindingCount, "\(sig.metalFunctionName) buffers")
            XCTAssertEqual(Int(rec.constantCount), sig.constantCount, "\(sig.metalFunctionName) constants")
        }
    }

    func testProjectionDtypeDisciplineIsLoud() throws {
        // Projections are a valid kernel LEGO in bf16/fp16/fp32/affine_u4 — fp16 COMPILES (commit
        // 4a2acbb retired the bf16-only gate; legality routes through the MatvecKernelTable
        // gateway). The prefill sibling of DenseTrunkEmitterTests.testDtypeDisciplineIsLoud.
        XCTAssertNoThrow(try generate(projDtype: .fp16))
        // A dtype with no matvec kernel (int32) is still LOUD through the gateway.
        XCTAssertThrowsError(try generate(projDtype: .int32)) { error in
            let text = "\(error)"
            XCTAssertTrue(text.contains("MatvecKernelTable") && text.contains("int32"),
                          "expected the gateway's int32 legality diagnostic, got: \(text)")
        }
    }

    func testSharesDenseTopologyGuards() throws {
        // The shared DenseTrunkGuards reject the same out-of-subset knobs as
        // decode — interleaved rope is the canonical silent-parity trap.
        XCTAssertThrowsError(try generate(attention: denseAttention(ropeLayout: .interleaved)))
        XCTAssertThrowsError(try generate(attention: denseAttention(ropeLayout: nil)))
    }

    func testRejectsPrefillBatchOutOf1To2048() throws {
        // The emitter owns the batch bound (a direct caller can bypass
        // validateSmeltIR): ≥1 for the buffer-plan slab multiplier, ≤2048 for the
        // causal_gqa_attn_cached_f32 threadgroup score buffer (B3.2b).
        XCTAssertThrowsError(try generate(maxPrefillBatch: 0))
        XCTAssertThrowsError(try generate(maxPrefillBatch: 2049))
        XCTAssertNoThrow(try generate(maxPrefillBatch: 1))
        XCTAssertNoThrow(try generate(maxPrefillBatch: 2048))
    }
}
