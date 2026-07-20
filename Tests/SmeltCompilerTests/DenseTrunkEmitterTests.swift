// Dense headless-trunk decode lowering tests.
import XCTest
@testable import SmeltCompiler
@testable import SmeltSchema

/// The dense-trunk decode table —
/// Record-level assertions for the registered fused sequence: kernels,
/// bindings, strides, the
/// pipelined cacheLen, and the embeddings-in / hidden-out ports.
final class DenseTrunkEmitterTests: XCTestCase {

    private let layers = 2
    private let hidden = 2048, heads = 16, kvHeads = 8, headDim = 128, inter = 6144

    private func denseAttention(
        gatedQ: Bool = false,
        ropeLayout: SmeltRoPELayout? = .splitHalf,
        vNorm: Bool = false,
        externalKV: Bool = false,
        attnScale: Float = 1,
        ropeDim: Int? = nil,
        slidingWindow: Int = 0
    ) -> SmeltAttentionConfig {
        SmeltAttentionConfig(
            qHeads: heads, kvHeads: kvHeads, headDim: headDim, gatedQ: gatedQ,
            qkNorm: true, qkNormMode: .weight, vNorm: vNorm,
            attnScale: attnScale, ropeTheta: 1_000_000, ropeDim: ropeDim,
            ropeLayout: ropeLayout, slidingWindow: slidingWindow,
            externalKV: externalKV
        )
    }

    private func ir(
        attention: SmeltAttentionConfig? = nil,
        ffnActivation: SmeltActivation = .swiglu
    ) -> SmeltModelIR {
        let attention = attention ?? denseAttention()
        let config = SmeltConfig(
            hiddenSize: hidden, numLayers: layers, vocabSize: 3072,
            staticSeqCapacity: 64,
            ropeDim: headDim, rmsEps: 1e-6, normMode: .weight,
            activationDtype: .fp32,
            portTopology: .embeddingsInHiddenOut,
            attention: attention,
            ffn: SmeltFFNConfig(dim: inter, activation: ffnActivation),
            tiedLMHead: false
        )
        return SmeltModelIR(
            modelName: "dense-trunk-test",
            config: config,
            layerPattern: SmeltLayerPattern(unit: [.attention], repeats: layers),
            quantization: SmeltModelIR.qwen35_2B.quantization,
            loading: SmeltModelIR.qwen35_2B.loading
        )
    }

    private func entry(_ name: String, _ offset: UInt64, _ dtype: SmeltDType) -> SmeltWeightEntry {
        SmeltWeightEntry(name: name, offset: offset, sizeBytes: 4096, shape: [1], dtype: dtype)
    }

    private func fabricatedLayout(projDtype: SmeltDType = .bf16) -> [SmeltWeightEntry] {
        var entries: [SmeltWeightEntry] = []
        var offset: UInt64 = 0x1000
        func add(_ name: String, _ dtype: SmeltDType) {
            entries.append(entry(name, offset, dtype))
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

    private func generate() throws -> ([SmeltDispatchRecord], SmeltBufferPlan) {
        let model = ir()
        let plan = buildBufferPlan(from: model)
        let result = try TopLevelEmitter.generate(
            ir: model, plan: plan, weightLayout: fabricatedLayout())
        return (result.dispatchRecords, plan)
    }

    // Dispatch-safety (the Phase-12 lesson): the f32 emitter realises only
    // the dense topology; any knob outside it must throw at codegen, not
    // silently mis-emit the dense route. These pin the loud rejections.
    private func assertEmitterRejects(
        _ model: SmeltModelIR, _ why: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let plan = buildBufferPlan(from: model)
        XCTAssertThrowsError(
            try TopLevelEmitter.generate(
                ir: model, plan: plan, weightLayout: fabricatedLayout()),
            why, file: file, line: line)
    }

    func testRejectsGatedQ() {
        // gated_q makes qProjDim = 2·heads·headDim; the fused qkv route
        // hardcodes qDim = heads·headDim and would read past Q silently.
        assertEmitterRejects(ir(attention: denseAttention(gatedQ: true)),
                             "gated_q must be rejected, not silently mis-emitted")
    }

    func testRejectsInterleavedRope() {
        // head_norm_rope_f32 is split-half; interleaved is the topology
        // default and a silent parity trap.
        assertEmitterRejects(ir(attention: denseAttention(ropeLayout: .interleaved)),
                             "interleaved RoPE must be rejected on the split-half kernel")
        assertEmitterRejects(ir(attention: denseAttention(ropeLayout: nil)),
                             "an unset rope_layout (defaults interleaved) must be rejected")
    }

    func testRejectsVNormAndExternalKV() {
        assertEmitterRejects(ir(attention: denseAttention(vNorm: true)),
                             "v_norm is not in the fused f32 sequence")
        assertEmitterRejects(ir(attention: denseAttention(externalKV: true)),
                             "external_kv has no own k/v projection")
    }

    func testRejectsUnregisteredKernelAssumptions() {
        // GeGLU FFN: the f32 route always emits gemv_gateup_swiglu_bf16w_f32.
        assertEmitterRejects(ir(ffnActivation: .geglu),
                             "geglu is not the emitted swiglu FFN route")
        // Sliding window: decode_gqa_attn_f32 attends the full position+1 cache.
        assertEmitterRejects(ir(attention: denseAttention(slidingWindow: 256)),
                             "sliding-window attention is not emitted")
        // Non-unit score scale: the kernel hardcodes 1/sqrt(head_dim).
        assertEmitterRejects(ir(attention: denseAttention(attnScale: 2)),
                             "a non-unit attn_scale is not emitted")
        // Partial RoPE: the f32 route strides rope rows by the full head_dim.
        assertEmitterRejects(ir(attention: denseAttention(ropeDim: headDim / 2)),
                             "partial rope_dim != head_dim is not emitted")
    }

    func testRecordSequenceIsTheRegisteredFusedShape() throws {
        let (records, _) = try generate()
        // Per layer: norm, qkv, hnr-q, hnr-k, attn, gemv_add, SWAP, norm,
        // gateup-swiglu, gemv_add, SWAP = 11; plus the final norm.
        XCTAssertEqual(records.count, layers * 11 + 1)
        let perLayer: [SmeltPipeline] = [
            .rmsNormCodecF32, .gemvQKVBF16WF32, .headNormRopeF32, .headNormRopeF32,
            .decodeGQAAttnF32, .gemvAddBF16WF32, .rmsNormCodecF32,
            .gemvGateUpSwigluBF16WF32, .gemvAddBF16WF32,
        ]
        let dispatched = records
            .filter { $0.opKind == SmeltDispatchRecord.opDispatch }
            .map { Int($0.pipeline) }
        var expected: [Int] = []
        for _ in 0..<layers { expected += perLayer.map(\.rawValue) }
        expected.append(SmeltPipeline.rmsNormCodecF32.rawValue)
        XCTAssertEqual(dispatched, expected)
        // Swaps: two per layer (cur returns to hiddenA every step), sitting
        // immediately after each residual gemv_add.
        let swapIndices = records.enumerated()
            .filter { $0.element.opKind == SmeltDispatchRecord.opSwap }.map(\.offset)
        XCTAssertEqual(swapIndices, (0..<layers).flatMap { [$0 * 11 + 6, $0 * 11 + 10] })
        // Final record: the hidden-output port's final norm.
        XCTAssertEqual(records.last!.pipeline, UInt16(SmeltPipeline.rmsNormCodecF32.rawValue))
    }

    func testCacheRowsArePositionStridedAndAttentionUsesPositionPlus1() throws {
        let (records, plan) = try generate()
        let kvRowStride = UInt64(kvHeads * headDim * 4)

        // gemv_qkv's v-output binds the value cache at position * rowStride.
        let qkv = records.first { $0.pipeline == UInt16(SmeltPipeline.gemvQKVBF16WF32.rawValue) }!
        let vBind = getBuffer(qkv, index: 6)
        XCTAssertEqual(Int(vBind.slot), plan.valCacheBaseSlot)
        XCTAssertEqual(vBind.offsetKind, 1)
        XCTAssertEqual(vBind.offset, kvRowStride)

        // k head_norm_rope writes the key cache row; q's writes a slot buffer.
        let hnrs = records.filter { $0.pipeline == UInt16(SmeltPipeline.headNormRopeF32.rawValue) }
        let kHnr = hnrs[1]
        let kOut = getBuffer(kHnr, index: 4)
        XCTAssertEqual(Int(kOut.slot), plan.keyCacheBaseSlot)
        XCTAssertEqual(kOut.offsetKind, 1)
        XCTAssertEqual(kOut.offset, kvRowStride)
        // Rope rows: fp32 stride headDim*4, position-strided.
        let cos = getBuffer(kHnr, index: 2)
        XCTAssertEqual(Int(cos.slot), plan.ropeCosSlot)
        XCTAssertEqual(cos.offsetKind, 1)
        XCTAssertEqual(cos.offset, UInt64(headDim * 4))

        // Attention: cacheLen constant resolves to position + 1.
        let attn = records.first { $0.pipeline == UInt16(SmeltPipeline.decodeGQAAttnF32.rawValue) }!
        let cacheLen = getConstant(attn, index: 0)
        XCTAssertEqual(cacheLen.kind, SmeltConstantRecord.kindPositionPlus1)
        XCTAssertEqual(cacheLen.bindingIndex, 4)
    }

    func testResidualStreamUsesCurAltWithEpsAsF32Literal() throws {
        let (records, _) = try generate()
        let gemvAdd = records.first { $0.pipeline == UInt16(SmeltPipeline.gemvAddBF16WF32.rawValue) }!
        XCTAssertEqual(getBuffer(gemvAdd, index: 2).slot, SmeltBufferRecord.slotCur, "residual reads cur")
        XCTAssertEqual(getBuffer(gemvAdd, index: 3).slot, SmeltBufferRecord.slotAlt, "output writes alt")

        let norm = records.first { $0.pipeline == UInt16(SmeltPipeline.rmsNormCodecF32.rawValue) }!
        XCTAssertEqual(getBuffer(norm, index: 0).slot, SmeltBufferRecord.slotCur, "norm reads the stream")
        let eps = getConstant(norm, index: 2)
        XCTAssertEqual(eps.kind, SmeltConstantRecord.kindLiteralF32)
        XCTAssertEqual(eps.value, Float(1e-6).bitPattern)
    }

    func testBindingCountsMatchTheCatalogSignatures() throws {
        let (records, _) = try generate()
        for rec in records where rec.opKind == SmeltDispatchRecord.opDispatch {
            let sig = SmeltKernelCatalog.signatures[Int(rec.pipeline)]
            XCTAssertEqual(Int(rec.bufferCount), sig.bufferBindingCount,
                           "\(sig.metalFunctionName) buffers")
            XCTAssertEqual(Int(rec.constantCount), sig.constantCount,
                           "\(sig.metalFunctionName) constants")
        }
    }

    func testEveryRecordFieldOfLayerZeroIsPinned() throws {
        // Table-driven snapshot of layer 0's complete records (review low):
        // a q/k weight swap, a constant reorder, or a grid-width change
        // must fail HERE, not at the W5 parity gate.
        let (records, plan) = try generate()
        func snap(_ rec: SmeltDispatchRecord) -> String {
            if rec.opKind == SmeltDispatchRecord.opSwap { return "SWAP" }
            var parts = ["p\(rec.pipeline)",
                         "g\(rec.gridW)x\(rec.gridH)x\(rec.gridD)",
                         "t\(rec.tgW)x\(rec.tgH)x\(rec.tgD)"]
            for i in 0..<Int(rec.bufferCount) {
                let b = getBuffer(rec, index: i)
                parts.append("b\(b.bindingIndex):s\(b.slot):k\(b.offsetKind):o\(b.offset)")
            }
            for i in 0..<Int(rec.constantCount) {
                let c = getConstant(rec, index: i)
                parts.append("c\(c.bindingIndex):k\(c.kind):v\(c.value)")
            }
            return parts.joined(separator: " ")
        }
        // Fabricated offsets: layer-0 weights start at 0x1000, 0x1000 apart,
        // in declaration order q,k,v,o,gate,up,down,inNorm,postNorm,qNorm,kNorm.
        let w = SmeltFixedSlot.weights.rawValue
        let kc = plan.keyCacheBaseSlot, vc = plan.valCacheBaseSlot
        let eps = Float(1e-6).bitPattern
        let qDim = heads * headDim, kvDim = kvHeads * headDim
        let kvStride = kvDim * 4
        let expected = [
            // input norm: cur, inNorm(0x8000), normOut | frames=1, dim, eps
            "p363 g32x1x1 t32x1x1 b0:s-1:k0:o0 b1:s\(w):k0:o\(0x8000) b2:s\(SmeltFixedSlot.normOutBuf.rawValue):k0:o0 "
                + "c3:k0:v1 c4:k0:v\(hidden) c5:k1:v\(eps)",
            // fused qkv: normOut, q(0x1000), k(0x2000), v(0x3000), attnQ, attnK, valCache row
            "p395 g\((qDim + kvDim + kvDim) * 32)x1x1 t32x1x1 b0:s\(SmeltFixedSlot.normOutBuf.rawValue):k0:o0 "
                + "b1:s\(w):k0:o\(0x1000) b2:s\(w):k0:o\(0x2000) b3:s\(w):k0:o\(0x3000) "
                + "b4:s\(SmeltFixedSlot.attnQBuf.rawValue):k0:o0 "
                + "b5:s\(SmeltFixedSlot.attnKBuf.rawValue):k0:o0 "
                + "b6:s\(vc):k1:o\(kvStride) "
                + "c7:k0:v\(qDim) c8:k0:v\(kvDim) c9:k0:v\(kvDim) c10:k0:v\(hidden)",
            // q head-norm+rope: attnQ, qNorm(0xA000), cos, sin, attnOut | heads, headDim, eps
            "p397 g\(heads * 32)x1x1 t32x1x1 b0:s\(SmeltFixedSlot.attnQBuf.rawValue):k0:o0 "
                + "b1:s\(w):k0:o\(0xA000) b2:s\(plan.ropeCosSlot):k1:o\(headDim * 4) "
                + "b3:s\(plan.ropeSinSlot):k1:o\(headDim * 4) "
                + "b4:s\(SmeltFixedSlot.attnOutBuf.rawValue):k0:o0 "
                + "c5:k0:v\(heads) c6:k0:v\(headDim) c7:k1:v\(eps)",
            // k head-norm+rope → key cache row, kNorm(0xB000)
            "p397 g\(kvHeads * 32)x1x1 t32x1x1 b0:s\(SmeltFixedSlot.attnKBuf.rawValue):k0:o0 "
                + "b1:s\(w):k0:o\(0xB000) b2:s\(plan.ropeCosSlot):k1:o\(headDim * 4) "
                + "b3:s\(plan.ropeSinSlot):k1:o\(headDim * 4) "
                + "b4:s\(kc):k1:o\(kvStride) "
                + "c5:k0:v\(kvHeads) c6:k0:v\(headDim) c7:k1:v\(eps)",
            // decode attention: qRoped, kCache, vCache, attnGate | pos+1, heads, kvHeads, headDim
            "p372 g\(heads * 32)x1x1 t32x1x1 b0:s\(SmeltFixedSlot.attnOutBuf.rawValue):k0:o0 "
                + "b1:s\(kc):k0:o0 b2:s\(vc):k0:o0 "
                + "b3:s\(SmeltFixedSlot.attnGateBuf.rawValue):k0:o0 "
                + "c4:k\(SmeltConstantRecord.kindPositionPlus1):v0 "
                + "c5:k0:v\(heads) c6:k0:v\(kvHeads) c7:k0:v\(headDim)",
            // o-proj + residual: attnGate, o(0x4000), cur, alt | n=hidden, K=qDim
            "p398 g\(hidden * 32)x1x1 t32x1x1 b0:s\(SmeltFixedSlot.attnGateBuf.rawValue):k0:o0 "
                + "b1:s\(w):k0:o\(0x4000) b2:s-1:k0:o0 b3:s-2:k0:o0 "
                + "c4:k0:v\(hidden) c5:k0:v\(qDim)",
            "SWAP",
            // post-attn norm: cur, postNorm(0x9000), normOut
            "p363 g32x1x1 t32x1x1 b0:s-1:k0:o0 b1:s\(w):k0:o\(0x9000) b2:s\(SmeltFixedSlot.normOutBuf.rawValue):k0:o0 "
                + "c3:k0:v1 c4:k0:v\(hidden) c5:k1:v\(eps)",
            // gate/up + swiglu: normOut, gate(0x5000), up(0x6000), ffnInt | inter, hidden
            "p396 g\(inter * 32)x1x1 t32x1x1 b0:s\(SmeltFixedSlot.normOutBuf.rawValue):k0:o0 "
                + "b1:s\(w):k0:o\(0x5000) b2:s\(w):k0:o\(0x6000) "
                + "b3:s\(SmeltFixedSlot.ffnIntBuf.rawValue):k0:o0 "
                + "c4:k0:v\(inter) c5:k0:v\(hidden)",
            // down-proj + residual: ffnInt, down(0x7000), cur, alt | hidden, inter
            "p398 g\(hidden * 32)x1x1 t32x1x1 b0:s\(SmeltFixedSlot.ffnIntBuf.rawValue):k0:o0 "
                + "b1:s\(w):k0:o\(0x7000) b2:s-1:k0:o0 b3:s-2:k0:o0 "
                + "c4:k0:v\(hidden) c5:k0:v\(inter)",
            "SWAP",
        ]
        for (idx, want) in expected.enumerated() {
            XCTAssertEqual(snap(records[idx]), want, "layer-0 record \(idx)")
        }
    }

    func testDtypeDisciplineIsLoud() {
        let model = ir()
        let plan = buildBufferPlan(from: model)
        // fp16/fp32/bf16/u4 projections are valid kernel LEGOS — they COMPILE, never gated (the
        // dtype-building-blocks invariant; commit 4a2acbb retired the bf16-only projection gate
        // and routes fp16/fp32 through the unfused dense kernels). fp16 must NOT throw.
        XCTAssertNoThrow(
            try TopLevelEmitter.generate(
                ir: model, plan: plan, weightLayout: fabricatedLayout(projDtype: .fp16)),
            "fp16 projections are a valid lego and must compile, not be gated")
        // But a dtype with NO matvec kernel is still LOUD — resolveProjection routes legality
        // through the ONE gateway (MatvecKernelTable.select), which throws for a non-matvec dtype
        // rather than silently mis-binding int32 bytes as a dense projection.
        XCTAssertThrowsError(
            try TopLevelEmitter.generate(
                ir: model, plan: plan, weightLayout: fabricatedLayout(projDtype: .int32)),
            "an int32 projection has no matvec kernel and must throw at codegen"
        ) { error in
            // Pin that the throw came from the GATEWAY (the mapped DenseTrunkEmitterError.wrongDtype
            // carrying MatvecKernelTable's .notMeaningful diagnostic) — not from WeightLocator
            // resolving first, which would also throw on int32 and make this gate pass even if
            // resolveProjection reverted to resolve-before-authorize.
            guard case DenseTrunkEmitterError.wrongDtype = error else {
                return XCTFail("expected DenseTrunkEmitterError.wrongDtype, got \(error)")
            }
            let text = "\(error)"
            XCTAssertTrue(text.contains("MatvecKernelTable") && text.contains("int32"),
                          "expected the gateway's int32 legality diagnostic, got: \(text)")
        }
    }
}
