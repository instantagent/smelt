import XCTest
import Metal
@testable import SmeltCompiler
@testable import SmeltRuntime
import SmeltSchema

/// B3.2c (docs/talker-trunk-in-package-plan.md): the compiled fp32 talker trunk
/// emitted INTO the TTS package as a `trunk/` sidecar sharing weights.bin +
/// model.metallib. Two pure gates (IR synthesis == the frozen talker-trunk golden;
/// the layout adapter re-points onto the shared blob) + one heavy real-weights gate
/// (the in-package trunk decodes BIT-EXACT to the hand path reading the SAME bytes).
final class Qwen3TTSTrunkSidecarTests: XCTestCase {

    // 1.7B-class talker dims (matches FixtureModelIRs.talkerTrunk).
    private static let layers = 28, hidden = 2048, headDim = 128
    private static let qDim = 2048, kvDim = 1024, inter = 6144, vocab = 3072

    /// A synthetic full-pipeline TTS manifest carrying just the talker.* entries the
    /// sidecar reads, at the 1.7B shapes. `projDtype` lets a test flip the projections
    /// to a non-bf16 build (which must NOT ship a trunk). Offsets are distinct + page-ish
    /// (only their pass-through into the layout matters here).
    private static func syntheticManifest(projDtype: String?) -> Qwen3TTSManifest {
        var entries: [Qwen3TTSManifest.Entry] = []
        var cursor: UInt64 = 0
        func add(_ name: String, _ shape: [Int], _ dtype: String?) {
            let bytes = UInt64(shape.reduce(1, *) * (dtype == "bf16" || dtype == "f16" ? 2 : 4))
            entries.append(.init(name: name, offset: cursor, byteLength: bytes, shape: shape, dtype: dtype))
            cursor += (bytes + 16383) & ~16383   // page-ish align so offsets stay distinct
        }
        for l in 0..<layers {
            let p = "talker.model.layers.\(l)"
            add("\(p).self_attn.q_proj.weight", [qDim, hidden], projDtype)
            add("\(p).self_attn.k_proj.weight", [kvDim, hidden], projDtype)
            add("\(p).self_attn.v_proj.weight", [kvDim, hidden], projDtype)
            add("\(p).self_attn.o_proj.weight", [hidden, qDim], projDtype)
            add("\(p).mlp.gate_proj.weight", [inter, hidden], projDtype)
            add("\(p).mlp.up_proj.weight", [inter, hidden], projDtype)
            add("\(p).mlp.down_proj.weight", [hidden, inter], projDtype)
            add("\(p).input_layernorm.weight", [hidden], nil)
            add("\(p).post_attention_layernorm.weight", [hidden], nil)
            add("\(p).self_attn.q_norm.weight", [headDim], nil)
            add("\(p).self_attn.k_norm.weight", [headDim], nil)
        }
        add("talker.model.norm.weight", [hidden], nil)
        add("talker.model.codec_embedding.weight", [vocab, hidden], projDtype)
        return Qwen3TTSManifest(
            version: 1, modelName: "synthetic-tts", pageSize: 16384,
            pipelines: [], eosTokens: [2150], totalBytes: cursor, weights: entries)
    }

    /// A synthetic full-pipeline TTS manifest whose talker.* PROJECTIONS are u4 (norms stay
    /// fp32), laid out with the REAL `Qwen3TTSPackageBuilder.u4Layout` so the gate exercises
    /// the production relative-offset arithmetic, not a hand-picked toy. Each u4 block carries
    /// page-aligned nibbles + RELATIVE scale/bias offsets — the convention Phase 2a translates
    /// to absolute. Dims default to the 1.7B class; a caller can pass SMALL dims (where the
    /// logical nibble size is below one page, so page-aligned scaleOffset ≠ nibble sizeBytes)
    /// to make the relative-vs-size translation field-distinguishable (codex adversarial #2).
    /// The single u4 Entry-from-u4Layout wiring (block at `offset`, page-aligned nibbles +
    /// RELATIVE scale/bias) — shared by syntheticU4Manifest's addU4 and flipEntryToU4.
    private static func u4Entry(name: String, shape: [Int], offset: UInt64, groupSize: Int,
                                page: Int = 16384) -> Qwen3TTSManifest.Entry {
        let lay = Qwen3TTSPackageBuilder.u4Layout(shape: shape, groupSize: groupSize, pageSize: page)
        return .init(name: name, offset: offset, byteLength: lay.blockBytes, shape: shape,
                     dtype: "u4", groupSize: groupSize,
                     scaleOffset: lay.scaleOffset, scaleByteLength: UInt64(lay.scaleBytes),
                     biasOffset: lay.biasOffset, biasByteLength: UInt64(lay.biasBytes))
    }

    /// Copy a manifest with a swapped weight list (+ optional model name) — collapses the
    /// repeated Qwen3TTSManifest(...) literal across the flip/corrupt/drop test helpers.
    private static func reweight(_ m: Qwen3TTSManifest, _ weights: [Qwen3TTSManifest.Entry],
                                 name: String? = nil) -> Qwen3TTSManifest {
        Qwen3TTSManifest(version: 1, modelName: name ?? m.modelName, pageSize: 16384,
                         pipelines: [], eosTokens: [2150], totalBytes: m.totalBytes, weights: weights)
    }

    private static func syntheticU4Manifest(
        groupSize: Int = 64, layers: Int = layers, hidden: Int = hidden, headDim: Int = headDim,
        qDim: Int = qDim, kvDim: Int = kvDim, inter: Int = inter, vocab: Int = vocab
    ) -> Qwen3TTSManifest {
        var entries: [Qwen3TTSManifest.Entry] = []
        var cursor: UInt64 = 0
        let page = 16384
        func addU4(_ name: String, _ shape: [Int]) {
            let off = Qwen3TTSPackageBuilder.pageAlign(cursor, UInt64(page))
            let entry = Self.u4Entry(name: name, shape: shape, offset: off, groupSize: groupSize, page: page)
            entries.append(entry)
            cursor = off + entry.byteLength
        }
        func addF32(_ name: String, _ shape: [Int]) {
            let off = Qwen3TTSPackageBuilder.pageAlign(cursor, UInt64(page))
            let bytes = UInt64(shape.reduce(1, *) * 4)
            entries.append(.init(name: name, offset: off, byteLength: (bytes + 16383) & ~16383,
                                 shape: shape, dtype: nil))
            cursor = off + ((bytes + 16383) & ~16383)
        }
        for l in 0..<layers {
            let p = "talker.model.layers.\(l)"
            addU4("\(p).self_attn.q_proj.weight", [qDim, hidden])
            addU4("\(p).self_attn.k_proj.weight", [kvDim, hidden])
            addU4("\(p).self_attn.v_proj.weight", [kvDim, hidden])
            addU4("\(p).self_attn.o_proj.weight", [hidden, qDim])
            addU4("\(p).mlp.gate_proj.weight", [inter, hidden])
            addU4("\(p).mlp.up_proj.weight", [inter, hidden])
            addU4("\(p).mlp.down_proj.weight", [hidden, inter])
            addF32("\(p).input_layernorm.weight", [hidden])
            addF32("\(p).post_attention_layernorm.weight", [hidden])
            addF32("\(p).self_attn.q_norm.weight", [headDim])
            addF32("\(p).self_attn.k_norm.weight", [headDim])
        }
        addF32("talker.model.norm.weight", [hidden])
        addF32("talker.model.codec_embedding.weight", [vocab, hidden])   // vocab-ref; weightLayout ignores it
        return Qwen3TTSManifest(
            version: 1, modelName: "synthetic-tts-u4", pageSize: 16384,
            pipelines: [], eosTokens: [2150], totalBytes: cursor, weights: entries)
    }

    // Small-but-valid trunk dims where the logical u4 nibble size is below one 16384-byte page,
    // so the real u4Layout page-aligns scaleOffset ABOVE the nibble sizeBytes (q [64,64]:
    // nibble = 64·32 = 2048, page-aligned scaleOffset = 16384). All dims meet the IR + F32Trunk
    // + u4 contracts (headDim a multiple of 32 and ≤ 128; everything a multiple of 4). Two
    // layers so layer 1 has a nonzero base.
    private static let sLayers = 2, sHidden = 64, sHeadDim = 32
    private static let sqDim = 64, skvDim = 32, sInter = 128, sVocab = 128, sGroup = 16
    private static func smallU4Manifest() -> Qwen3TTSManifest {
        syntheticU4Manifest(groupSize: sGroup, layers: sLayers, hidden: sHidden, headDim: sHeadDim,
                            qDim: sqDim, kvDim: skvDim, inter: sInter, vocab: sVocab)
    }

    /// Phase 2a (docs/lego-phase2-plan.md): a u4-projection build now PRODUCES a trunk layout
    /// (the old `XCTAssertThrowsError` flips), with each u4 projection mapped to an `.affineU4`
    /// entry whose scale/bias offsets are the parent's RELATIVE offsets translated to ABSOLUTE
    /// (`offset + scaleOffset`). Tested on a NON-FIRST layer so the parent offset is nonzero and
    /// the translation isn't a 0+0 no-op; norms stay fp32; `WeightLocator.resolve` then surfaces
    /// the three absolute regions. THE risk (relative→absolute) is gated here before any u4 kernel.
    func testTrunkWeightLayoutMapsU4ProjectionsToAbsoluteRegions() throws {
        let m = Self.syntheticU4Manifest(groupSize: 64)
        let ir = try Qwen3TTSTrunkSidecar.synthesizeIR(from: m)   // synthesis reads shapes, dtype-agnostic
        let layout = try Qwen3TTSTrunkSidecar.weightLayout(from: m, ir: ir)
        XCTAssertEqual(layout.count, Self.layers * 11 + 1)

        let byName = Dictionary(m.weights.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
        // Layer 1 (not 0): its q_proj sits past several blocks, so parent.offset > 0 — the
        // relative→absolute translation is genuinely exercised, not masked by a zero base.
        let parent = byName["talker.model.layers.1.self_attn.q_proj.weight"]!
        XCTAssertGreaterThan(parent.offset, 0, "non-first layer must have a nonzero base offset")
        XCTAssertGreaterThan(parent.scaleOffset!, 0)
        XCTAssertNotEqual(parent.scaleOffset!, parent.biasOffset!, "scale ≠ bias relative offsets")

        let q1 = layout.first { $0.name == "layers_1_self_attn_q_proj_weight" }!
        XCTAssertEqual(q1.dtype, .affineU4)
        XCTAssertEqual(q1.offset, parent.offset)                                 // nibbles: unchanged
        XCTAssertEqual(q1.groupSize, 64)
        XCTAssertEqual(q1.scalesOffset, parent.offset + parent.scaleOffset!)     // relative → ABSOLUTE
        XCTAssertEqual(q1.biasesOffset, parent.offset + parent.biasOffset!)
        XCTAssertEqual(q1.scalesSizeBytes, parent.scaleByteLength!)
        XCTAssertEqual(q1.biasesSizeBytes, parent.biasByteLength!)
        XCTAssertEqual(q1.sizeBytes, UInt64(Self.qDim * SmeltAffineU4.packedRowStride(cols: Self.hidden)))

        // Norms still map to fp32 dense entries.
        XCTAssertEqual(layout.first { $0.name == "layers_1_input_layernorm_weight" }?.dtype, .fp32)
        XCTAssertEqual(layout.first { $0.name == "norm_weight" }?.dtype, .fp32)

        // And the locator (U1) surfaces the three absolute regions the affine kernel would bind.
        let loc = try WeightLocator.resolve(q1)
        XCTAssertEqual(loc.kind, .affineU4(groupSize: 64))
        XCTAssertEqual(loc.regions.map(\.offset), [
            parent.offset,
            parent.offset + parent.scaleOffset!,
            parent.offset + parent.biasOffset!,
        ])
    }

    /// Codex adversarial #2: the 1.7B gate above is partially vacuous — at those dims every u4
    /// nibble region is exactly page-sized, so `sizeBytes == scaleOffset` and a buggy mapper
    /// using `offset + sizeBytes` instead of `offset + scaleOffset` would pass. This gate uses
    /// SMALL dims where the logical nibble size (2048 B) is below one page, so the real layout
    /// page-aligns scaleOffset to 16384 ≠ sizeBytes — the translation FIELD is now
    /// distinguishable, and we assert it reads the RELATIVE scale/bias offsets, not sizeBytes.
    func testTrunkWeightLayoutU4UsesRelativeScaleOffsetNotNibbleSize() throws {
        let m = Self.smallU4Manifest()
        let ir = try Qwen3TTSTrunkSidecar.synthesizeIR(from: m)
        let layout = try Qwen3TTSTrunkSidecar.weightLayout(from: m, ir: ir)

        let parent = Dictionary(m.weights.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })[
            "talker.model.layers.1.self_attn.q_proj.weight"]!
        let q1 = layout.first { $0.name == "layers_1_self_attn_q_proj_weight" }!
        // The premise that makes this non-vacuous: page-aligned scaleOffset strictly exceeds the
        // logical nibble sizeBytes, so the two candidate translations diverge.
        XCTAssertEqual(q1.sizeBytes, UInt64(Self.sqDim * SmeltAffineU4.packedRowStride(cols: Self.sHidden)))
        XCTAssertGreaterThan(parent.scaleOffset!, q1.sizeBytes,
                             "small-dim premise: page-aligned scaleOffset must exceed nibble sizeBytes")
        // The translation uses the RELATIVE scale/bias offsets…
        XCTAssertEqual(q1.scalesOffset, parent.offset + parent.scaleOffset!)
        XCTAssertEqual(q1.biasesOffset, parent.offset + parent.biasOffset!)
        // …NOT offset + nibble sizeBytes (the field-confusion bug this gate exists to catch).
        XCTAssertNotEqual(q1.scalesOffset, parent.offset + q1.sizeBytes,
                          "scalesOffset must come from the RELATIVE scaleOffset, not offset+sizeBytes")
    }

    /// Phase 3 U1 (docs/lego-phase3-plan.md): a u4-projection trunk emits the UNFUSED u4 route
    /// (gemv_u4_f32 per projection + shared swiglu/scale_residual/norm), mirroring the hand u4
    /// path — NOT the fused bf16 kernels and NOT affine_matvec. Anti-fallback: asserts the u4
    /// kernels are PRESENT and the bf16-fused / affine_matvec kernels are ABSENT (and the
    /// converse for bf16), so a silent wrong-kernel or fallback can't pass as green.
    func testU4TrunkDecodeEmitsUnfusedU4Route() throws {
        func raw(_ p: SmeltPipeline) -> UInt16 { UInt16(p.rawValue) }
        func decodePipes(_ m: Qwen3TTSManifest) throws -> Set<UInt16> {
            let ir = try Qwen3TTSTrunkSidecar.synthesizeIR(from: m)
            return Set(try DenseTrunkEmitter.generate(
                ir: ir, plan: buildBufferPlan(from: ir),
                weightLayout: try Qwen3TTSTrunkSidecar.weightLayout(from: m, ir: ir)).dispatchRecords.map { $0.pipeline })
        }
        func prefillPipes(_ m: Qwen3TTSManifest) throws -> Set<UInt16> {
            let ir = try Qwen3TTSTrunkSidecar.synthesizeIR(from: m)
            return Set(try DenseTrunkPrefillEmitter.generate(
                ir: ir, plan: buildBufferPlan(from: ir),
                weightLayout: try Qwen3TTSTrunkSidecar.weightLayout(from: m, ir: ir)).dispatchRecords.map { $0.pipeline })
        }

        // u4 DECODE: gemv_u4_f32 present; bf16-fused + affine_matvec absent.
        let u4d = try decodePipes(Self.syntheticU4Manifest())
        XCTAssertTrue(u4d.contains(raw(.gemvU4F32)), "u4 decode must emit gemv_u4_f32")
        XCTAssertTrue(u4d.contains(raw(.swigluF32)) && u4d.contains(raw(.scaleResidualTCF32)),
                      "u4 decode uses the shared swiglu + scale_residual merges")
        for absent: SmeltPipeline in [.gemvQKVBF16WF32, .gemvAddBF16WF32, .gemvGateUpSwigluBF16WF32, .affineMatvec] {
            XCTAssertFalse(u4d.contains(raw(absent)), "u4 decode must NOT contain \(absent) (no fused/affine fallback)")
        }
        // u4 PREFILL: gemm_u4_f32 present; gemm_bf16w + affine_matvec absent.
        let u4p = try prefillPipes(Self.syntheticU4Manifest())
        XCTAssertTrue(u4p.contains(raw(.gemmU4F32)), "u4 prefill must emit gemm_u4_f32")
        for absent: SmeltPipeline in [.gemmBF16WF32, .affineMatvec] {
            XCTAssertFalse(u4p.contains(raw(absent)), "u4 prefill must NOT contain \(absent)")
        }
        // bf16 (converse): fused/gemm_bf16w present; no u4 kernels.
        let bf16d = try decodePipes(Self.syntheticManifest(projDtype: "bf16"))
        XCTAssertTrue(bf16d.contains(raw(.gemvQKVBF16WF32)), "bf16 decode stays fused")
        XCTAssertFalse(bf16d.contains(raw(.gemvU4F32)), "bf16 decode has no gemv_u4_f32")
        let bf16p = try prefillPipes(Self.syntheticManifest(projDtype: "bf16"))
        XCTAssertTrue(bf16p.contains(raw(.gemmBF16WF32)), "bf16 prefill stays gemm_bf16w_f32")
        XCTAssertFalse(bf16p.contains(raw(.gemmU4F32)), "bf16 prefill has no gemm_u4_f32")
    }

    /// Dtype-completeness: f32/f16 PROJECTION weights compile through the UNFUSED dense route
    /// (gemv_f32 / gemv_f16w_f32 decode, gemm_f32 / gemm_f16w_f32 prefill) — the same scaffold the
    /// u4 route uses, differing only in the gemv/gemm kernel. Proves dtype picks a kernel lego, not a
    /// capability gate: every weight dtype with a kernel is WIRED, none gated out.
    func testDenseF32F16TrunkEmitsUnfusedDenseRoute() throws {
        func raw(_ p: SmeltPipeline) -> UInt16 { UInt16(p.rawValue) }
        func decodePipes(_ m: Qwen3TTSManifest) throws -> Set<UInt16> {
            let ir = try Qwen3TTSTrunkSidecar.synthesizeIR(from: m)
            return Set(try DenseTrunkEmitter.generate(
                ir: ir, plan: buildBufferPlan(from: ir),
                weightLayout: try Qwen3TTSTrunkSidecar.weightLayout(from: m, ir: ir)).dispatchRecords.map { $0.pipeline })
        }
        func prefillPipes(_ m: Qwen3TTSManifest) throws -> Set<UInt16> {
            let ir = try Qwen3TTSTrunkSidecar.synthesizeIR(from: m)
            return Set(try DenseTrunkPrefillEmitter.generate(
                ir: ir, plan: buildBufferPlan(from: ir),
                weightLayout: try Qwen3TTSTrunkSidecar.weightLayout(from: m, ir: ir)).dispatchRecords.map { $0.pipeline })
        }
        // f32: gemv_f32 / gemm_f32 present; the bf16-fused, u4, and f16 kernels all absent + the
        // shared scaffold (swiglu + scale_residual merge) is reused, like u4.
        let f32d = try decodePipes(Self.syntheticManifest(projDtype: "f32"))
        XCTAssertTrue(f32d.contains(raw(.gemvF32)), "f32 decode must emit gemv_f32")
        XCTAssertTrue(f32d.contains(raw(.swigluF32)) && f32d.contains(raw(.scaleResidualTCF32)),
                      "f32 decode reuses the shared swiglu + scale_residual merges")
        for absent: SmeltPipeline in [.gemvQKVBF16WF32, .gemvU4F32, .gemvF16WF32] {
            XCTAssertFalse(f32d.contains(raw(absent)), "f32 decode must NOT contain \(absent)")
        }
        let f32p = try prefillPipes(Self.syntheticManifest(projDtype: "f32"))
        XCTAssertTrue(f32p.contains(raw(.gemmF32)), "f32 prefill must emit gemm_f32")
        for absent: SmeltPipeline in [.gemmBF16WF32, .gemmU4F32, .gemmF16WF32] {
            XCTAssertFalse(f32p.contains(raw(absent)), "f32 prefill must NOT contain \(absent)")
        }
        // f16: gemv_f16w_f32 / gemm_f16w_f32 present; f32/bf16/u4 absent.
        let f16d = try decodePipes(Self.syntheticManifest(projDtype: "f16"))
        XCTAssertTrue(f16d.contains(raw(.gemvF16WF32)), "f16 decode must emit gemv_f16w_f32")
        for absent: SmeltPipeline in [.gemvF32, .gemvQKVBF16WF32, .gemvU4F32] {
            XCTAssertFalse(f16d.contains(raw(absent)), "f16 decode must NOT contain \(absent)")
        }
        let f16p = try prefillPipes(Self.syntheticManifest(projDtype: "f16"))
        XCTAssertTrue(f16p.contains(raw(.gemmF16WF32)), "f16 prefill must emit gemm_f16w_f32")
        for absent: SmeltPipeline in [.gemmF32, .gemmBF16WF32, .gemmU4F32] {
            XCTAssertFalse(f16p.contains(raw(absent)), "f16 prefill must NOT contain \(absent)")
        }
    }

    /// The dual talker+MTP trunk emit ships ALL-OR-NONE and only for ONE shared emittable dtype.
    /// shouldShipTrunks requires the codec_embedding vocab-ref + layer-0 q_proj in BOTH networks
    /// at the SAME dtype ∈ {f32, f16, bf16, u4} (dtype-completeness: every kernel-having dtype ships;
    /// f32/f16 via the unfused dense route). A cross-network mix (bf16 talker / u4 MTP) still ships
    /// NEITHER (not a real build, and the all-or-none preflight keeps a partial commit impossible).
    func testShouldShipTrunksRequiresBothNetworksSameShippableDtype() {
        func e(_ name: String, _ dtype: String?) -> Qwen3TTSManifest.Entry {
            .init(name: name, offset: 0, byteLength: 0, shape: [4, 4], dtype: dtype)
        }
        let codec = e("talker.model.codec_embedding.weight", "bf16")
        let talkerQ = "talker.model.layers.0.self_attn.q_proj.weight"
        let mtpQ = "talker.code_predictor.model.layers.0.self_attn.q_proj.weight"

        XCTAssertTrue(Qwen3TTSPackageBuilder.shouldShipTrunks(
            [codec, e(talkerQ, "bf16"), e(mtpQ, "bf16")]), "bf16 both → ship")
        XCTAssertTrue(Qwen3TTSPackageBuilder.shouldShipTrunks(
            [codec, e(talkerQ, "u4"), e(mtpQ, "u4")]), "u4 both → ship (Phase 3)")
        XCTAssertFalse(Qwen3TTSPackageBuilder.shouldShipTrunks(
            [codec, e(talkerQ, "bf16"), e(mtpQ, "u4")]), "cross-network mix → no ship")
        XCTAssertTrue(Qwen3TTSPackageBuilder.shouldShipTrunks(
            [codec, e(talkerQ, "f16"), e(mtpQ, "f16")]), "f16 both → ship (dtype-complete: unfused dense)")
        XCTAssertTrue(Qwen3TTSPackageBuilder.shouldShipTrunks(
            [codec, e(talkerQ, nil), e(mtpQ, nil)]), "f32 both (nil dtype) → ship (dtype-complete)")
        XCTAssertFalse(Qwen3TTSPackageBuilder.shouldShipTrunks(
            [e(talkerQ, "bf16"), e(mtpQ, "bf16")]), "missing codec_embedding vocab-ref → no trunk")
        XCTAssertFalse(Qwen3TTSPackageBuilder.shouldShipTrunks(
            [codec, e(talkerQ, "bf16")]), "missing MTP network → no trunk")
    }

    /// Codex re-verdict #1: shouldShipTrunks' layer-0 q_proj check is only an INTENT sentinel —
    /// it can't catch a deep failure (a non-q0 projection that's u4/f16, or a missing vocab-ref).
    /// The builder's all-or-none safety instead comes from `prepare` (the pure, no-FS-write prefix
    /// of emit), which fully resolves every layer + both dispatch tables and throws on ANY
    /// emitability failure. The builder prepares BOTH specs before committing either, so a deep
    /// MTP failure can't follow a committed talker sidecar. This gates prepare's deep-catch.
    func testTrunkPreparePreflightCatchesDeepEmitabilityFailures() throws {
        let clean = Self.syntheticMTPManifest()
        XCTAssertNoThrow(try Qwen3TTSTrunkSidecar.prepare(manifest: clean, spec: .mtp), "clean MTP prepares")

        // Layer-0 q_proj stays bf16 (the sentinel would pass) but a DEEP layer mixes dtypes (its
        // gate_proj is u4, the rest bf16) → prepare must throw via the homogeneous-dtype guard
        // (a trunk layer must be uniformly bf16 or u4), not commit a sidecar.
        let deepMixed = Self.flipEntryToU4(
            clean, name: "talker.code_predictor.model.layers.1.mlp.gate_proj.weight", groupSize: 16)
        XCTAssertEqual(
            deepMixed.weights.first { $0.name == "talker.code_predictor.model.layers.0.self_attn.q_proj.weight" }?.dtype,
            "bf16", "the layer-0 sentinel still passes — only a deep layer mixes dtypes")
        XCTAssertThrowsError(try Qwen3TTSTrunkSidecar.prepare(manifest: deepMixed, spec: .mtp),
                             "a deep within-layer dtype mix must fail the preflight")

        // Missing the lm_head.0 vocab-ref → synthesizeIR can't derive the MTP vocab → prepare throws.
        let noLMHead = Self.reweight(
            clean, clean.weights.filter { $0.name != "talker.code_predictor.lm_head.0.weight" },
            name: "no-lm-head")
        XCTAssertThrowsError(try Qwen3TTSTrunkSidecar.prepare(manifest: noLMHead, spec: .mtp),
                             "a missing lm_head.0 vocab-ref must fail the preflight")
    }

    /// Replace one entry with a u4 version (real u4Layout block), keeping its offset — for
    /// preflight tests that need a single deep projection flipped off bf16.
    private static func flipEntryToU4(
        _ m: Qwen3TTSManifest, name: String, groupSize: Int
    ) -> Qwen3TTSManifest {
        let weights = m.weights.map { e in
            e.name == name ? Self.u4Entry(name: e.name, shape: e.shape, offset: e.offset, groupSize: groupSize) : e
        }
        return Self.reweight(m, weights)
    }

    /// Codex adversarial #3: a re-pointed u4 entry must bind the SAME bytes the hand matvec
    /// reads — so weightLayout validates the affine regions (exact fp16 scale/bias lengths +
    /// in-block ordering) and throws loud on a corrupt manifest, rather than producing a
    /// plausible .affineU4 entry that silently binds wrong-but-in-range bytes.
    func testU4WeightLayoutRejectsMalformedRegions() throws {
        let good = Self.smallU4Manifest()
        let ir = try Qwen3TTSTrunkSidecar.synthesizeIR(from: good)
        // A valid manifest passes (control).
        XCTAssertNoThrow(try Qwen3TTSTrunkSidecar.weightLayout(from: good, ir: ir))

        // Corrupt ONE u4 projection's scale region length → loud .u4MalformedRegions.
        func corrupt(_ transform: (Qwen3TTSManifest.Entry) -> Qwen3TTSManifest.Entry) -> Qwen3TTSManifest {
            let target = "talker.model.layers.1.self_attn.q_proj.weight"
            return Self.reweight(good, good.weights.map { $0.name == target ? transform($0) : $0 }, name: "corrupt-u4")
        }
        func dup(_ x: Qwen3TTSManifest.Entry, offset: UInt64? = nil,
                 scaleByteLength: UInt64? = nil, biasOffset: UInt64? = nil) -> Qwen3TTSManifest.Entry {
            .init(name: x.name, offset: offset ?? x.offset, byteLength: x.byteLength, shape: x.shape,
                  dtype: x.dtype, groupSize: x.groupSize,
                  scaleOffset: x.scaleOffset, scaleByteLength: scaleByteLength ?? x.scaleByteLength,
                  biasOffset: biasOffset ?? x.biasOffset, biasByteLength: x.biasByteLength)
        }
        for bad in [
            corrupt { dup($0, scaleByteLength: $0.scaleByteLength! + 2) },   // wrong region length
            corrupt { dup($0, biasOffset: $0.byteLength + 4096) },          // region past the block
            // hostile huge base offset: the in-block regions stay valid, but offset + byteLength
            // overflows UInt64 — must throw, not trap on the absolute-offset addition.
            corrupt { dup($0, offset: UInt64.max - 1024) },
        ] {
            XCTAssertThrowsError(try Qwen3TTSTrunkSidecar.weightLayout(from: bad, ir: ir)) { error in
                guard case Qwen3TTSTrunkSidecar.SidecarError.u4MalformedRegions = error else {
                    return XCTFail("expected .u4MalformedRegions, got \(error)")
                }
            }
        }
    }

    /// Phase 2b (docs/lego-phase2-plan.md): both trunk emitters route their projection binding
    /// through WeightLocator.resolve. A bf16 build emits a non-empty decode + prefill table (the
    /// routing keeps .dense(.bf16) emittable — the heavy real-weights gate pins it byte-identical).
    /// Phase 3 (U1/U2) SUPERSEDED the u4 assert-and-defer: a u4 layout now ALSO emits non-empty
    /// tables (the unfused u4 route); the kernel-level assertions live in
    /// testU4TrunkDecodeEmitsUnfusedU4Route. Pure + fast.
    func testTrunkEmittersRouteProjectionsThroughLocator() throws {
        for m in [Self.syntheticManifest(projDtype: "bf16"), Self.syntheticU4Manifest()] {
            let ir = try Qwen3TTSTrunkSidecar.synthesizeIR(from: m)
            let plan = buildBufferPlan(from: ir)
            let layout = try Qwen3TTSTrunkSidecar.weightLayout(from: m, ir: ir)
            XCTAssertFalse(
                try DenseTrunkEmitter.generate(ir: ir, plan: plan, weightLayout: layout).dispatchRecords.isEmpty,
                "decode trunk must emit through the locator")
            XCTAssertFalse(
                try DenseTrunkPrefillEmitter.generate(ir: ir, plan: plan, weightLayout: layout).dispatchRecords.isEmpty,
                "prefill trunk must emit through the locator")
        }
    }

    func testPreparedTrunkUsesCompilationPlanForSharedLayout() throws {
        let manifest = Self.syntheticU4Manifest()
        let prepared = try Qwen3TTSTrunkSidecar.prepare(manifest: manifest)

        XCTAssertEqual(prepared.compilationPlan.bufferPlan, prepared.plan)
        XCTAssertEqual(prepared.compilationPlan.plannedWeightEntries, prepared.layout)
        XCTAssertEqual(prepared.compilationPlan.weightStoragePlan.plannedUses.count, 196)
        XCTAssertEqual(prepared.compilationPlan.weightStoragePlan.decisions.count, 196)
        XCTAssertTrue(prepared.compilationPlan.weightStoragePlan.isMemoryNeutral)
        XCTAssertTrue(prepared.compilationPlan.weightStoragePlan.plannedUses.allSatisfy {
            $0.capability.phase == .storage
        })
        XCTAssertFalse(prepared.decode.dispatchRecords.isEmpty)
        XCTAssertFalse(prepared.prefill.dispatchRecords.isEmpty)
    }

    func testCommittedTrunkManifestUsesExplicitMarkerOnly() throws {
        let packageRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3-tts-trunk-sidecar-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: packageRoot) }
        try FileManager.default.createDirectory(at: packageRoot, withIntermediateDirectories: true)
        try Data("weights".utf8).write(to: packageRoot.appendingPathComponent("weights.bin"))
        try Data("metallib".utf8).write(to: packageRoot.appendingPathComponent("model.metallib"))

        let prepared = try Qwen3TTSTrunkSidecar.prepare(manifest: Self.syntheticU4Manifest())
        try Qwen3TTSTrunkSidecar.commit(prepared, intoPackage: packageRoot.path)

        let manifestURL = packageRoot
            .appendingPathComponent(prepared.spec.sidecarDir, isDirectory: true)
            .appendingPathComponent("manifest.json")
        let sidecar = try SmeltManifest.decode(from: Data(contentsOf: manifestURL))
        XCTAssertNil(sidecar.kind)
        XCTAssertEqual(sidecar.headlessTrunkABI, true)
        XCTAssertNil(sidecar.blocks)
        XCTAssertNil(sidecar.loop)
    }

    // Synthetic MTP (code_predictor) entries the .mtp spec reads, at small distinct dims
    // (5L / hidden 1024 / 8-2×128 / inter 3072 / vocab 2048). lm_head.0 is the [vocab,hidden]
    // vocab-ref (its hidden width = the MTP's OWN hidden, unlike codec_embedding).
    private static let mLayers = 5, mHidden = 1024, mHeadDim = 128
    private static let mqDim = 1024, mkvDim = 256, mInter = 3072, mVocab = 2048
    private static func syntheticMTPManifest() -> Qwen3TTSManifest {
        var entries: [Qwen3TTSManifest.Entry] = []
        var cursor: UInt64 = 0
        func add(_ name: String, _ shape: [Int], _ dtype: String?) {
            let bytes = UInt64(shape.reduce(1, *) * (dtype == "bf16" || dtype == "f16" ? 2 : 4))
            entries.append(.init(name: name, offset: cursor, byteLength: bytes, shape: shape, dtype: dtype))
            cursor += (bytes + 16383) & ~16383
        }
        for l in 0..<mLayers {
            let p = "talker.code_predictor.model.layers.\(l)"
            add("\(p).self_attn.q_proj.weight", [mqDim, mHidden], "bf16")
            add("\(p).self_attn.k_proj.weight", [mkvDim, mHidden], "bf16")
            add("\(p).self_attn.v_proj.weight", [mkvDim, mHidden], "bf16")
            add("\(p).self_attn.o_proj.weight", [mHidden, mqDim], "bf16")
            add("\(p).mlp.gate_proj.weight", [mInter, mHidden], "bf16")
            add("\(p).mlp.up_proj.weight", [mInter, mHidden], "bf16")
            add("\(p).mlp.down_proj.weight", [mHidden, mInter], "bf16")
            add("\(p).input_layernorm.weight", [mHidden], nil)
            add("\(p).post_attention_layernorm.weight", [mHidden], nil)
            add("\(p).self_attn.q_norm.weight", [mHeadDim], nil)
            add("\(p).self_attn.k_norm.weight", [mHeadDim], nil)
        }
        add("talker.code_predictor.model.norm.weight", [mHidden], nil)
        add("talker.code_predictor.lm_head.0.weight", [mVocab, mHidden], "bf16")
        return Qwen3TTSManifest(
            version: 1, modelName: "synthetic-tts", pageSize: 16384,
            pipelines: [], eosTokens: [2150], totalBytes: cursor, weights: entries)
    }

    /// B3.2d M0 — the spec primitive GENERALIZES: the `.mtp` spec derives the MTP's OWN dims
    /// from the code_predictor shapes and maps the same canonical `layers_N_…` names onto the
    /// `code_predictor.*` offsets — proving the sidecar is block-agnostic, not talker-renamed.
    func testMTPSpecSynthesizesValidTrunkIRFromCodePredictor() throws {
        let m = Self.syntheticMTPManifest()
        let ir = try Qwen3TTSTrunkSidecar.synthesizeIR(from: m, spec: .mtp)
        XCTAssertEqual(ir.config.numLayers, Self.mLayers)
        XCTAssertEqual(ir.config.hiddenSize, Self.mHidden)            // MTP hidden, NOT talker hidden
        XCTAssertEqual(ir.config.attention?.qHeads, Self.mqDim / Self.mHeadDim)
        XCTAssertEqual(ir.config.attention?.kvHeads, Self.mkvDim / Self.mHeadDim)
        XCTAssertEqual(ir.config.attention?.headDim, Self.mHeadDim)
        XCTAssertEqual(ir.config.ffn.dim, Self.mInter)

        let layout = try Qwen3TTSTrunkSidecar.weightLayout(from: m, ir: ir, spec: .mtp)
        XCTAssertEqual(layout.count, Self.mLayers * 11 + 1)
        let byName = Dictionary(m.weights.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
        let q0 = layout.first { $0.name == "layers_0_self_attn_q_proj_weight" }!  // canonical = prefix-independent
        XCTAssertEqual(q0.dtype, .bf16)
        XCTAssertEqual(q0.offset, byName["talker.code_predictor.model.layers.0.self_attn.q_proj.weight"]!.offset)
        XCTAssertEqual(layout.first { $0.name == "norm_weight" }?.dtype, .fp32)

        // canShareWeights/synthesis key off the SPEC prefix: .mtp accepts this manifest,
        // .talker rejects it (no talker.model.* projections present).
        XCTAssertTrue(Qwen3TTSTrunkSidecar.canShareWeights(m, spec: .mtp))
        XCTAssertFalse(Qwen3TTSTrunkSidecar.canShareWeights(m, spec: .talker))
    }

    /// The synthesised IR must equal the frozen talker-trunk golden
    /// (`FixtureModelIRs.talkerTrunk`) — same dims, config, and buffer plan. Pure +
    /// fast (no checkpoint, no GPU).
    func testSynthesizedTrunkIRMatchesFixtureOracle() throws {
        let fixtureIR = FixtureModelIRs.talkerTrunk
        let synthIR = try Qwen3TTSTrunkSidecar.synthesizeIR(from: Self.syntheticManifest(projDtype: "bf16"))

        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        XCTAssertEqual(
            try enc.encode(SmeltCompiler.manifestConfigSnapshot(from: synthIR)),
            try enc.encode(SmeltCompiler.manifestConfigSnapshot(from: fixtureIR)),
            "synthesised trunk config != talker-trunk golden")
        XCTAssertEqual(
            try enc.encode(buildBufferPlan(from: synthIR).toSlotLayout()),
            try enc.encode(buildBufferPlan(from: fixtureIR).toSlotLayout()),
            "synthesised trunk buffer plan != oracle")

        // The config-snapshot + slot-layout above cover the STRUCTURAL dims, but NOT the
        // fp32-trunk ABI CONSTANTS the emitters bake as kernel scalars (RoPE θ/layout,
        // qk-norm, attn scale, rms eps, activation dtype, tied LM head). A factory that got
        // any of those wrong (e.g. ropeTheta 10 000 instead of 1e6) would still match the
        // snapshot + layout + pipeline sets. Assert them field-for-field against the frozen
        // golden so a drift fails HERE — the prove-equal gate for the direct-construction
        // migration.
        let a = synthIR.config, b = fixtureIR.config
        XCTAssertEqual(a.rmsEps, b.rmsEps)
        XCTAssertEqual(a.normMode, b.normMode)
        XCTAssertEqual(a.activationDtype, b.activationDtype)
        XCTAssertEqual(a.tiedLMHead, b.tiedLMHead)
        XCTAssertEqual(a.staticSeqCapacity, b.staticSeqCapacity)
        XCTAssertEqual(a.ropeDim, b.ropeDim)
        XCTAssertEqual(a.ffn.dim, b.ffn.dim)
        XCTAssertEqual(a.ffn.activation, b.ffn.activation)
        let qa = try XCTUnwrap(a.attention), qb = try XCTUnwrap(b.attention)
        XCTAssertEqual(qa.qHeads, qb.qHeads)
        XCTAssertEqual(qa.kvHeads, qb.kvHeads)
        XCTAssertEqual(qa.headDim, qb.headDim)
        XCTAssertEqual(qa.gatedQ, qb.gatedQ)
        XCTAssertEqual(qa.qkNorm, qb.qkNorm)
        XCTAssertEqual(qa.qkNormMode, qb.qkNormMode)
        XCTAssertEqual(qa.attnScale, qb.attnScale)
        XCTAssertEqual(qa.ropeTheta, qb.ropeTheta)
        XCTAssertEqual(qa.ropeLayout, qb.ropeLayout)
        XCTAssertEqual(synthIR.quantization.strategy, fixtureIR.quantization.strategy)
        XCTAssertEqual(synthIR.loading.checkpointMap, fixtureIR.loading.checkpointMap)
        // Prefill: the golden is the batch-256 talker trunk, which matches
        // synthesizeIR's default maxPrefillBatch (256), so engine/batch/handoff line up.
        XCTAssertEqual(synthIR.prefill?.engine, fixtureIR.prefill?.engine)
        XCTAssertEqual(synthIR.prefill?.maxBatchSize, fixtureIR.prefill?.maxBatchSize)
        XCTAssertEqual(synthIR.prefill?.handoffFamilies, fixtureIR.prefill?.handoffFamilies)
    }

    /// The layout adapter produces exactly the emitter's expected weights (11/layer +
    /// final norm), each at the SHARED blob's offset with the trunk dtype + logical size. ALL
    /// projection dtypes map (dtype-completeness): bf16/f16 → 2-byte, f32 → 4-byte, u4 → affine —
    /// dtype picks a kernel lego, never gates the layout. Pure + fast.
    func testTrunkWeightLayoutMapsSharedOffsetsForAllDtypes() throws {
        let m = Self.syntheticManifest(projDtype: "bf16")
        let ir = try Qwen3TTSTrunkSidecar.synthesizeIR(from: m)
        let layout = try Qwen3TTSTrunkSidecar.weightLayout(from: m, ir: ir)
        XCTAssertEqual(layout.count, Self.layers * 11 + 1)

        let byName = Dictionary(m.weights.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
        let q0 = layout.first { $0.name == "layers_0_self_attn_q_proj_weight" }!
        XCTAssertEqual(q0.dtype, .bf16)
        XCTAssertEqual(q0.offset, byName["talker.model.layers.0.self_attn.q_proj.weight"]!.offset)
        XCTAssertEqual(q0.sizeBytes, UInt64(Self.qDim * Self.hidden * 2))   // logical, not page-rounded

        let n0 = layout.first { $0.name == "layers_0_input_layernorm_weight" }!
        XCTAssertEqual(n0.dtype, .fp32)
        XCTAssertEqual(n0.sizeBytes, UInt64(Self.hidden * 4))
        XCTAssertEqual(layout.first { $0.name == "norm_weight" }?.dtype, .fp32)

        XCTAssertTrue(Qwen3TTSTrunkSidecar.canShareWeights(m))
        // f32 projections map to 4-byte fp32 entries (unfused dense route).
        let f32m = Self.syntheticManifest(projDtype: nil)
        XCTAssertTrue(Qwen3TTSTrunkSidecar.canShareWeights(f32m))
        let f32q0 = try Qwen3TTSTrunkSidecar.weightLayout(from: f32m, ir: ir)
            .first { $0.name == "layers_0_self_attn_q_proj_weight" }!
        XCTAssertEqual(f32q0.dtype, .fp32)
        XCTAssertEqual(f32q0.sizeBytes, UInt64(Self.qDim * Self.hidden * 4))
        // f16 projections map to 2-byte fp16 entries.
        let f16m = Self.syntheticManifest(projDtype: "f16")
        XCTAssertTrue(Qwen3TTSTrunkSidecar.canShareWeights(f16m))
        let f16q0 = try Qwen3TTSTrunkSidecar.weightLayout(from: f16m, ir: ir)
            .first { $0.name == "layers_0_self_attn_q_proj_weight" }!
        XCTAssertEqual(f16q0.dtype, .fp16)
        XCTAssertEqual(f16q0.sizeBytes, UInt64(Self.qDim * Self.hidden * 2))
    }

}
