import XCTest
@testable import SmeltCompiler
@testable import SmeltSchema
import SmeltRuntime

/// A minimal in-memory CheckpointTensorSource: vends descriptors by name over a
/// single scratch buffer (the exact-coverage check is name-only and fires before
/// any data is read, so the shared dummy pointer is never dereferenced).
private final class ArrayTensorSource: CheckpointTensorSource {
    let checkpointTensors: [CheckpointTensorDescriptor]
    private let scratch: UnsafeMutableRawPointer
    init(names: [String]) {
        scratch = .allocate(byteCount: 16, alignment: 16)
        checkpointTensors = names.enumerated().map { i, n in
            CheckpointTensorDescriptor(index: i, name: n, dtype: "BF16", shape: [1], byteCount: 2)
        }
    }
    deinit { scratch.deallocate() }
    func checkpointTensorData(_ d: CheckpointTensorDescriptor) -> UnsafeRawPointer {
        UnsafeRawPointer(scratch)
    }
}

/// The full set of talker-trunk checkpoint (HF) names for `layers` layers — the
/// 11 per-layer module suffixes plus the final norm. Maps 1:1 onto the canonical
/// trunk layout, so a source built from exactly these covers expectedLayout.
private func trunkHFNames(layers: Int) -> [String] {
    let suffixes = [
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        "input_layernorm.weight", "post_attention_layernorm.weight",
    ]
    var names = ["talker.model.norm.weight"]
    for l in 0..<layers { for s in suffixes { names.append("talker.model.layers.\(l).\(s)") } }
    return names
}

/// W5 (docs/talker-trunk-fit-audit.md): the talker-trunk checkpoint adapter.
/// Proves the generic compiler can now (a) keep the authored Qwen3-TTS trunk
/// tensor map, (b) rename it to the canonical `layers_N_...` schema, (c) bind
/// `.talker` from `loading.checkpoint_map`, and (d) validate the nested
/// `talker_config`. The real-weights decode parity is the W5c gate.
final class TalkerCheckpointAdapterTests: XCTestCase {

    private func ir(_ name: String) throws -> SmeltModelIR {
        precondition(name == "talker-trunk", "only the talker-trunk fixture is modeled")
        let ir = FixtureModelIRs.talkerTrunk
        try validateSmeltIR(ir)
        return ir
    }

    // MARK: - name mapping

    func testMapsEveryTrunkTensorKindToCanonical() {
        let cases: [(String, String)] = [
            ("talker.model.layers.0.self_attn.q_proj.weight", "layers_0_self_attn_q_proj_weight"),
            ("talker.model.layers.0.self_attn.k_proj.weight", "layers_0_self_attn_k_proj_weight"),
            ("talker.model.layers.0.self_attn.v_proj.weight", "layers_0_self_attn_v_proj_weight"),
            ("talker.model.layers.0.self_attn.o_proj.weight", "layers_0_self_attn_o_proj_weight"),
            ("talker.model.layers.0.self_attn.q_norm.weight", "layers_0_self_attn_q_norm_weight"),
            ("talker.model.layers.0.self_attn.k_norm.weight", "layers_0_self_attn_k_norm_weight"),
            ("talker.model.layers.27.mlp.gate_proj.weight", "layers_27_mlp_gate_proj_weight"),
            ("talker.model.layers.27.mlp.up_proj.weight", "layers_27_mlp_up_proj_weight"),
            ("talker.model.layers.27.mlp.down_proj.weight", "layers_27_mlp_down_proj_weight"),
            ("talker.model.layers.5.input_layernorm.weight", "layers_5_input_layernorm_weight"),
            ("talker.model.layers.5.post_attention_layernorm.weight", "layers_5_post_attention_layernorm_weight"),
            ("talker.model.norm.weight", "norm_weight"),
        ]
        for (hf, canonical) in cases {
            XCTAssertEqual(TalkerCheckpointAdapter.mapName(hf), canonical, hf)
        }
    }

    // MARK: - trunk filter

    func testKeepsTrunkTensorsOnly() {
        // Trunk: every layers.* tensor and the final norm.
        for keep in [
            "talker.model.layers.0.self_attn.q_proj.weight",
            "talker.model.layers.27.mlp.down_proj.weight",
            "talker.model.layers.13.input_layernorm.weight",
            "talker.model.norm.weight",
        ] {
            XCTAssertTrue(TalkerCheckpointAdapter.isTrunkTensor(keep), "should keep \(keep)")
        }
        // Not trunk: front-end embeddings (live under talker.model. but no
        // .layers.), the MTP, the codec head, the text projection, the codec
        // decoder — every weight the compiled trunk must NOT ingest.
        for skip in [
            "talker.model.codec_embedding.weight",
            "talker.model.text_embedding.weight",
            "talker.code_predictor.layers.0.self_attn.q_proj.weight",
            "talker.codec_head.weight",
            "talker.text_projection.0.weight",
            "decoder.layers.0.conv.weight",
            // Unexpected layer tensors a future checkpoint could ship: the
            // allowlist rejects them rather than letting assembleCheckpointTensors
            // silently keep-and-ignore (no .bias on a bias-free GQA, no rotary
            // inv_freq buffer, no stray norm).
            "talker.model.layers.0.self_attn.q_proj.bias",
            "talker.model.layers.0.self_attn.rotary_emb.inv_freq",
            "talker.model.layers.0.self_attn.qk_norm.weight",
            "talker.model.layers.0.mlp.router.weight",
        ] {
            XCTAssertFalse(TalkerCheckpointAdapter.isTrunkTensor(skip), "should skip \(skip)")
        }
    }

    func testTrunkHasNoTiedWeights() throws {
        // Hidden-out port: no LM head, nothing tied.
        let cfg = try ir("talker-trunk").config
        XCTAssertFalse(TalkerCheckpointAdapter.isTiedWeight("norm_weight", config: cfg))
        XCTAssertFalse(TalkerCheckpointAdapter.isTiedWeight("lm_head_weight", config: cfg))
    }

    // MARK: - adapter selection

    func testAuthoredTalkerMapSelectsTalkerAdapter() throws {
        XCTAssertEqual(try SmeltCheckpointAdapter.authored(for: ir("talker-trunk")), .talker)
    }

    func testOnlyTalkerRequiresExactTrunkCoverage() {
        XCTAssertTrue(SmeltCheckpointAdapter.talker.requiresExactTrunkCoverage)
        XCTAssertFalse(SmeltCheckpointAdapter.qwen.requiresExactTrunkCoverage)
        XCTAssertFalse(SmeltCheckpointAdapter.llama.requiresExactTrunkCoverage)
    }

    func testExactCoverageRejectsUnexpectedTrunkTensor() throws {
        // A checkpoint shipping an out-of-range layer index passes the suffix
        // allowlist but maps to a name absent from the layout. Exact coverage
        // must reject it loudly, not keep-and-ignore it.
        let irT = try ir("talker-trunk")
        let layout = SmeltWeightLayout.computeLayout(from: irT)
        var names = trunkHFNames(layers: irT.config.numLayers)
        names.append("talker.model.layers.999.self_attn.q_proj.weight")
        XCTAssertThrowsError(try SmeltCompiler.assembleCheckpointTensors(
            source: ArrayTensorSource(names: names),
            adapter: .talker, expectedLayout: layout, ir: irT)
        ) { err in
            XCTAssertTrue("\(err)".contains("does not expect"),
                          "expected exact-coverage rejection, got \(err)")
        }
    }

    func testExactCoverageRejectsDuplicateMappedTensor() throws {
        // A source vending the same trunk tensor twice (duplicate shards) would
        // collapse in the name dict and pass exact coverage; the injective-map
        // guard rejects it loudly.
        let irT = try ir("talker-trunk")
        let layout = SmeltWeightLayout.computeLayout(from: irT)
        var names = trunkHFNames(layers: irT.config.numLayers)
        names.append("talker.model.layers.0.self_attn.q_proj.weight")  // duplicate
        XCTAssertThrowsError(try SmeltCompiler.assembleCheckpointTensors(
            source: ArrayTensorSource(names: names),
            adapter: .talker, expectedLayout: layout, ir: irT)
        ) { err in
            XCTAssertTrue("\(err)".contains("same trunk name"),
                          "expected duplicate-name rejection, got \(err)")
        }
    }

    func testQwen3TTSWithoutFp32IsNotTalker() throws {
        // The full TTS pipeline (non-fp32) stays on the hand path: the
        // talker-trunk checkpoint map must reject it.
        let fp16IR = FixtureModelIRs.talkerTrunkFP16
        XCTAssertEqual(fp16IR.config.activationDtype, .fp16)
        XCTAssertThrowsError(try SmeltCheckpointAdapter.authored(for: fp16IR)) { error in
            XCTAssertTrue("\(error)".contains("hf.qwen3-tts-talker-trunk"))
        }
    }

    // MARK: - config validation

    private func talkerConfig(
        hidden: Int = 2048, layers: Int = 28, vocab: Int = 3072,
        theta: Int = 1_000_000, eps: Double = 1e-6
    ) -> [String: Any] {
        ["talker_config": [
            "hidden_size": hidden, "num_hidden_layers": layers, "vocab_size": vocab,
            "rope_theta": theta, "rms_norm_eps": eps,
        ]]
    }

    func testValidateConfigAcceptsMatchingNestedConfig() throws {
        XCTAssertNoThrow(try SmeltCheckpointAdapter.talker.validateConfig(
            hfConfig: talkerConfig(), ir: try ir("talker-trunk")))
    }

    func testValidateConfigToleratesFloatSpelledNumbers() throws {
        // JSON may spell rope_theta as 1000000.0 (and hidden_size etc. integrally
        // either way) — NSNumber reads must not false-fail on the decimal form.
        let cfg: [String: Any] = ["talker_config": [
            "hidden_size": 2048.0, "num_hidden_layers": 28, "vocab_size": 3072,
            "rope_theta": 1_000_000.0, "rms_norm_eps": 1e-6,
        ]]
        XCTAssertNoThrow(try SmeltCheckpointAdapter.talker.validateConfig(
            hfConfig: cfg, ir: try ir("talker-trunk")))
    }

    func testValidateConfigRejectsDimMismatch() throws {
        let irT = try ir("talker-trunk")
        XCTAssertThrowsError(try SmeltCheckpointAdapter.talker.validateConfig(
            hfConfig: talkerConfig(hidden: 4096), ir: irT))
        XCTAssertThrowsError(try SmeltCheckpointAdapter.talker.validateConfig(
            hfConfig: talkerConfig(theta: 10_000), ir: irT))
    }

    func testValidateConfigRejectsMissingTalkerConfig() throws {
        XCTAssertThrowsError(try SmeltCheckpointAdapter.talker.validateConfig(
            hfConfig: ["model_type": "qwen3_tts"], ir: try ir("talker-trunk")))
    }

    func testValidateConfigRejectsNonIntegralDim() throws {
        // Int(exactly:) must not let a truncating 2048.5 validate as 2048.
        let cfg: [String: Any] = ["talker_config": [
            "hidden_size": 2048.5, "num_hidden_layers": 28, "vocab_size": 3072,
            "rope_theta": 1_000_000, "rms_norm_eps": 1e-6,
        ]]
        XCTAssertThrowsError(try SmeltCheckpointAdapter.talker.validateConfig(
            hfConfig: cfg, ir: try ir("talker-trunk")))
    }
}
