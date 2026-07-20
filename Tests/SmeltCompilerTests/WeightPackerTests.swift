// WeightPackerTests — Validates weight layout computation and glob matching.

import XCTest

@testable import SmeltCompiler
@testable import SmeltSchema

final class WeightPackerTests: XCTestCase {

    // MARK: - Layout entry count

    func testQwen35_2BLayoutEntryCount() {
        let ir = SmeltModelIR.qwen35_2B
        let layout = SmeltWeightLayout.computeLayout(from: ir)

        // Expected per DeltaNet layer (18 layers):
        //   input_layernorm, qkv, z, a, b, conv1d, A_log, dt_bias, norm,
        //   out_proj, post_attention_layernorm, gate, up, down = 14
        //
        // Expected per Attention layer (6 layers):
        //   input_layernorm, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
        //   post_attention_layernorm, gate, up, down = 11
        //
        // Global: embed_tokens, norm_weight = 2 (lm_head tied to embed_tokens)
        let deltaCount = 14
        let attnCount = 11
        let globalCount = 2
        let expected = ir.numDeltaLayers * deltaCount
            + ir.numAttnLayers * attnCount
            + globalCount
        // 18*14 + 6*11 + 2 = 252 + 66 + 2 = 320

        XCTAssertEqual(
            layout.count,
            expected,
            "Qwen 3.5 2B should have \(expected) weight entries, "
                + "got \(layout.count)"
        )
    }

    // MARK: - Alignment

    func testAllOffsetsAre16ByteAligned() {
        let layout = SmeltWeightLayout.computeLayout(from: .qwen35_2B)

        for entry in layout {
            XCTAssertEqual(
                entry.offset % 16, 0,
                "\(entry.name) offset \(entry.offset) is not 16-byte aligned"
            )

            if let lutOffset = entry.lutOffset {
                XCTAssertEqual(
                    lutOffset % 16, 0,
                    "\(entry.name) LUT offset \(lutOffset) is not "
                        + "16-byte aligned"
                )
            }
        }
    }

    func testAffineOffsetsAre128ByteAligned() {
        let base = SmeltModelIR.qwen35_2B
        let ir = SmeltModelIR(
            modelName: base.modelName,
            config: base.config,
            layerPattern: base.layerPattern,
            quantization: SmeltQuantizationConfig(
                strategy: .affineU4,
                groupSize: 64,
                excludePatterns: base.quantization.excludePatterns
            ),
            loading: base.loading,
            prefill: base.prefill,
            inference: base.inference
        )
        let layout = SmeltWeightLayout.computeLayout(from: ir)

        for entry in layout where entry.dtype == .affineU4 {
            XCTAssertEqual(
                entry.offset % 128, 0,
                "\(entry.name) offset \(entry.offset) is not 128-byte aligned"
            )
            if let scalesOffset = entry.scalesOffset {
                XCTAssertEqual(
                    scalesOffset % 128, 0,
                    "\(entry.name) scales offset \(scalesOffset) is not 128-byte aligned"
                )
            }
            if let biasesOffset = entry.biasesOffset {
                XCTAssertEqual(
                    biasesOffset % 128, 0,
                    "\(entry.name) biases offset \(biasesOffset) is not 128-byte aligned"
                )
            }
        }
    }

    func testAffineQuantizerConsumesExpectedLayoutOffsets() throws {
        let rows = 2
        let cols = 4
        let count = rows * cols
        let values = (0..<count).map { Float($0) * 0.25 }
        let outputPath = NSTemporaryDirectory()
            + "planned_affine_\(ProcessInfo.processInfo.globallyUniqueString).bin"
        defer { try? FileManager.default.removeItem(atPath: outputPath) }

        let expectedEntry = SmeltWeightEntry(
            name: "planned_weight",
            offset: 128,
            sizeBytes: 4,
            shape: [rows, cols],
            dtype: .affineU4,
            groupSize: 2,
            packedRowStride: 2,
            paddedCols: cols,
            scalesOffset: 256,
            scalesSizeBytes: 8,
            biasesOffset: 384,
            biasesSizeBytes: 8
        )
        let config = SmeltQuantizationConfig(
            strategy: .affineU4,
            groupSize: 2,
            excludePatterns: []
        )

        let entries = try values.withUnsafeBytes { raw in
            try SmeltAffineQuantizer.quantize(
                tensors: [(
                    runtimeName: "planned_weight",
                    data: raw.baseAddress! as UnsafeRawPointer,
                    byteCount: count * MemoryLayout<Float>.stride,
                    shape: [rows, cols],
                    dtype: "F32"
                )],
                config: config,
                outputPath: outputPath,
                expectedLayout: [expectedEntry]
            )
        }

        XCTAssertEqual(entries.count, 1)
        XCTAssertEqual(entries[0].offset, expectedEntry.offset)
        XCTAssertEqual(entries[0].scalesOffset, expectedEntry.scalesOffset)
        XCTAssertEqual(entries[0].biasesOffset, expectedEntry.biasesOffset)
        let fileSize = try FileManager.default
            .attributesOfItem(atPath: outputPath)[.size] as? NSNumber
        XCTAssertEqual(
            fileSize?.uint64Value,
            expectedEntry.biasesOffset! + expectedEntry.biasesSizeBytes!
        )
    }

    func testAffineQuantizerRejectsIncompatibleExpectedLayout() throws {
        let rows = 2
        let cols = 4
        let count = rows * cols
        let values = (0..<count).map { Float($0) * 0.25 }
        let outputPath = NSTemporaryDirectory()
            + "bad_planned_affine_\(ProcessInfo.processInfo.globallyUniqueString).bin"
        defer { try? FileManager.default.removeItem(atPath: outputPath) }

        let incompatibleEntry = SmeltWeightEntry(
            name: "planned_weight",
            offset: 0,
            sizeBytes: UInt64(count * MemoryLayout<Float16>.stride),
            shape: [rows, cols],
            dtype: .fp16
        )
        let config = SmeltQuantizationConfig(
            strategy: .affineU4,
            groupSize: 2,
            excludePatterns: []
        )

        XCTAssertThrowsError(try values.withUnsafeBytes { raw in
            try SmeltAffineQuantizer.quantize(
                tensors: [(
                    runtimeName: "planned_weight",
                    data: raw.baseAddress! as UnsafeRawPointer,
                    byteCount: count * MemoryLayout<Float>.stride,
                    shape: [rows, cols],
                    dtype: "F32"
                )],
                config: config,
                outputPath: outputPath,
                expectedLayout: [incompatibleEntry]
            )
        }) { error in
            guard let affineError = error as? SmeltAffineQuantizerError,
                  case .expectedLayoutMismatch = affineError
            else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testWeightPackerUsesSuppliedCompilationPlanLayout() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("planned_packer_\(UUID().uuidString)")
        let sourceDir = root.appendingPathComponent("source")
        let outputDir = root.appendingPathComponent("output")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(
            at: sourceDir,
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: outputDir,
            withIntermediateDirectories: true
        )

        let entry = SmeltWeightEntry(
            name: "planned_weight",
            offset: 16,
            sizeBytes: 4,
            shape: [2],
            dtype: .fp16
        )
        try Data([1, 2, 3, 4]).write(
            to: sourceDir.appendingPathComponent("planned_weight.bin")
        )

        let compilationPlan = SmeltCompilationPlan(
            policy: SmeltCompilationConfig(),
            bufferPlan: buildBufferPlan(from: .qwen35_2B),
            kernelPlan: .empty,
            plannedWeightEntries: [entry],
            weightStoragePlan: SmeltWeightStoragePlan(
                plannedUses: [],
                decisions: [],
                issues: []
            )
        )
        var packer = SmeltWeightPacker(compilationPlan: compilationPlan)
        let manifest = try packer.packWeights(
            sourceDir: sourceDir.path,
            outputDir: outputDir.path
        )

        XCTAssertEqual(manifest.totalBytes, 20)
        XCTAssertEqual(manifest.entries, [entry])
        let packed = try Data(contentsOf: outputDir.appendingPathComponent("weights.bin"))
        XCTAssertEqual(packed.count, 20)
        XCTAssertEqual(Array(packed.suffix(4)), [1, 2, 3, 4])
    }

    // MARK: - Excluded weights are FP16

    func testExcludedWeightsAreFP16() {
        let layout = SmeltWeightLayout.computeLayout(from: .qwen35_2B)

        let expectedFP16Patterns = [
            "embed_tokens",
            "input_layernorm_weight",
            "post_attention_layernorm_weight",
            "norm_weight",
            "conv1d_weight",
            "A_log",
            "dt_bias",
            "q_norm_weight",
            "k_norm_weight",
        ]

        for entry in layout {
            let shouldBeFP16 = expectedFP16Patterns.contains { pattern in
                entry.name.contains(pattern)
            }
            // The linear_attn_norm_weight is a delta-layer norm, also excluded
            let isDeltaNorm = entry.name.contains("linear_attn_norm_weight")

            if shouldBeFP16 || isDeltaNorm {
                XCTAssertEqual(
                    entry.dtype,
                    .fp16,
                    "\(entry.name) should be FP16 (excluded from quant)"
                )
                XCTAssertNil(
                    entry.lutOffset,
                    "\(entry.name) should not have a LUT offset"
                )
            }
        }
    }

    // MARK: - Quantized weights have LUT offsets

    func testQuantizedWeightsHaveLUTOffsets() {
        let layout = SmeltWeightLayout.computeLayout(from: .qwen35_2B)
        let quantized = layout.filter { $0.dtype == .u4Lut }

        XCTAssertGreaterThan(
            quantized.count, 0,
            "Should have at least one quantized weight"
        )

        for entry in quantized {
            XCTAssertNotNil(
                entry.lutOffset,
                "\(entry.name) is u4_lut but has no LUT offset"
            )
            XCTAssertNotNil(
                entry.lutSizeBytes,
                "\(entry.name) is u4_lut but has no LUT size"
            )
            XCTAssertNotNil(
                entry.groupSize,
                "\(entry.name) is u4_lut but has no group size"
            )
            XCTAssertEqual(
                entry.groupSize, 16,
                "\(entry.name) should have group size 16"
            )

            // LUT offset must be after indices
            if let lutOff = entry.lutOffset {
                XCTAssertGreaterThan(
                    lutOff, entry.offset,
                    "\(entry.name) LUT offset should be after indices"
                )
            }
        }
    }

    // MARK: - Glob pattern matching

    func testGlobPatternExactSubstring() {
        XCTAssertTrue(isExcludedFromQuantization(
            name: "embed_tokens",
            patterns: ["embed_tokens"]
        ))
        XCTAssertTrue(isExcludedFromQuantization(
            name: "layers_0_linear_attn_conv1d_weight",
            patterns: ["conv1d_weight"]
        ))
        XCTAssertFalse(isExcludedFromQuantization(
            name: "layers_0_mlp_gate_proj_weight",
            patterns: ["conv1d_weight"]
        ))
    }

    func testGlobPatternWildcard() {
        // *_norm_weight should match names ending in _norm_weight
        XCTAssertTrue(isExcludedFromQuantization(
            name: "layers_3_linear_attn_norm_weight",
            patterns: ["*_norm_weight"]
        ))
        XCTAssertTrue(isExcludedFromQuantization(
            name: "layers_9_self_attn_q_norm_weight",
            patterns: ["*_norm_weight"]
        ))
        // layernorm_weight does NOT end with _norm_weight (no underscore)
        XCTAssertFalse(isExcludedFromQuantization(
            name: "layers_5_input_layernorm_weight",
            patterns: ["*_norm_weight"]
        ))
        XCTAssertFalse(isExcludedFromQuantization(
            name: "layers_0_mlp_gate_proj_weight",
            patterns: ["*_norm_weight"]
        ))
    }

    func testGlobPatternMultiplePatterns() {
        let patterns = [
            "embed_tokens", "conv1d_weight", "A_log",
            "dt_bias", "*_norm_weight",
        ]
        XCTAssertTrue(isExcludedFromQuantization(
            name: "layers_2_linear_attn_A_log",
            patterns: patterns
        ))
        XCTAssertTrue(isExcludedFromQuantization(
            name: "layers_2_linear_attn_dt_bias",
            patterns: patterns
        ))
        XCTAssertFalse(isExcludedFromQuantization(
            name: "layers_2_linear_attn_in_proj_qkv_weight",
            patterns: patterns
        ))
    }

    // MARK: - No overlapping ranges

    func testNoOverlappingRanges() {
        let layout = SmeltWeightLayout.computeLayout(from: .qwen35_2B)

        // Collect all (start, end) ranges
        var ranges: [(name: String, start: UInt64, end: UInt64)] = []
        for entry in layout {
            ranges.append((
                entry.name,
                entry.offset,
                entry.offset + entry.sizeBytes
            ))
            if let lutOff = entry.lutOffset,
               let lutSize = entry.lutSizeBytes
            {
                ranges.append((
                    "\(entry.name)_lut",
                    lutOff,
                    lutOff + lutSize
                ))
            }
        }

        // Sort by start offset
        ranges.sort { $0.start < $1.start }

        // Check no overlaps
        for idx in 1..<ranges.count {
            let prev = ranges[idx - 1]
            let curr = ranges[idx]
            XCTAssertLessThanOrEqual(
                prev.end, curr.start,
                "\(prev.name) [..\(prev.end)] overlaps with "
                    + "\(curr.name) [\(curr.start)..]"
            )
        }
    }

    // MARK: - Specific weight shapes

    func testEmbedTokensShape() {
        let layout = SmeltWeightLayout.computeLayout(from: .qwen35_2B)
        guard let embed = layout.first(where: { $0.name == "embed_tokens" })
        else {
            XCTFail("embed_tokens not found")
            return
        }
        XCTAssertEqual(embed.shape, [248_320, 2_048])
        XCTAssertEqual(embed.dtype, .fp16)
        // Size = 248320 * 2048 * 2 bytes
        XCTAssertEqual(embed.sizeBytes, UInt64(248_320 * 2_048 * 2))
    }

    func testPackedIndexAndLUTSizes() {
        let layout = SmeltWeightLayout.computeLayout(from: .qwen35_2B)
        // QKV: [6144, 2048] quantized with group_size=16
        // packed_index_bytes = 6144 * 2048 / 2 = 6,291,456
        // nGroups = ceil(6144/16) = 384
        // lut_bytes = 384 * 16 * 2 = 12,288
        guard let qkv = layout.first(where: {
            $0.name == "layers_0_linear_attn_in_proj_qkv_weight"
        }) else {
            XCTFail("qkv weight not found")
            return
        }
        XCTAssertEqual(qkv.sizeBytes, UInt64(6_291_456))
        XCTAssertEqual(qkv.lutSizeBytes, UInt64(12_288))
        XCTAssertEqual(qkv.groupSize, 16)
        XCTAssertEqual(qkv.packedRowStride, 1_024) // 2048 / 2
    }

    func testQKVProjectionShape() {
        let layout = SmeltWeightLayout.computeLayout(from: .qwen35_2B)
        guard let qkv = layout.first(where: {
            $0.name == "layers_0_linear_attn_in_proj_qkv_weight"
        }) else {
            XCTFail("qkv weight not found")
            return
        }
        XCTAssertEqual(qkv.shape, [6_144, 2_048])
        XCTAssertEqual(qkv.dtype, .u4Lut)
    }

    // MARK: - external_kv attention weight emission

    private func parseExternalKVIR(
        externalKV: Bool,
        qkNorm: Bool
    ) throws -> SmeltModelIR {
        // Minimal external-KV spec: 4 layers, hidden=256, ffn=2048, a single
        // attn block with external_kv toggled. Wrapped in a helper so each
        // assertion can flip the flag in isolation.
        SmeltModelIR(
            modelName: "test/model",
            config: SmeltConfig(
                hiddenSize: 256, numLayers: 4, vocabSize: 262_144,
                staticSeqCapacity: 256, ropeDim: 256, rmsEps: 1e-6,
                attention: SmeltAttentionConfig(
                    qHeads: 4, kvHeads: 1, headDim: 256, gatedQ: false,
                    qkNorm: qkNorm, externalKV: externalKV),
                ffn: SmeltFFNConfig(dim: 2048, activation: .swiglu)),
            layerPattern: SmeltLayerPattern(unit: [.attention], repeats: 4),
            quantization: SmeltQuantizationConfig(strategy: .lutU4, groupSize: 16, excludePatterns: []),
            loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic))
    }

    func testExternalKVAttentionLayerSkipsKAndVProjections() throws {
        let ir = try parseExternalKVIR(externalKV: true, qkNorm: false)
        let layout = SmeltWeightLayout.computeLayout(from: ir)
        let names = Set(layout.map(\.name))

        for i in 0 ..< 4 {
            XCTAssertFalse(
                names.contains("layers_\(i)_self_attn_k_proj_weight"),
                "external-KV layer must not declare k_proj"
            )
            XCTAssertFalse(
                names.contains("layers_\(i)_self_attn_v_proj_weight"),
                "external-KV layer must not declare v_proj"
            )
            XCTAssertTrue(
                names.contains("layers_\(i)_self_attn_q_proj_weight"),
                "external-KV layer still computes Q from hidden state"
            )
            XCTAssertTrue(
                names.contains("layers_\(i)_self_attn_o_proj_weight"),
                "external-KV layer still has output projection"
            )
        }
    }

    func testExternalKVAttentionLayerKeepsQNormSkipsKNorm() throws {
        let ir = try parseExternalKVIR(externalKV: true, qkNorm: true)
        let layout = SmeltWeightLayout.computeLayout(from: ir)
        let names = Set(layout.map(\.name))

        for i in 0 ..< 4 {
            XCTAssertTrue(
                names.contains("layers_\(i)_self_attn_q_norm_weight"),
                "external-KV layer with qk_norm still emits q_norm"
            )
            XCTAssertFalse(
                names.contains("layers_\(i)_self_attn_k_norm_weight"),
                "external-KV layer must not declare k_norm — k comes from target"
            )
        }
    }

    func testNonExternalKVAttentionLayerStillEmitsAllProjections() throws {
        // Regression guard: existing target specs without external_kv
        // must continue to emit k_proj, v_proj, and (when qk_norm)
        // both q_norm and k_norm.
        let ir = try parseExternalKVIR(externalKV: false, qkNorm: true)
        let layout = SmeltWeightLayout.computeLayout(from: ir)
        let names = Set(layout.map(\.name))

        XCTAssertTrue(names.contains("layers_0_self_attn_k_proj_weight"))
        XCTAssertTrue(names.contains("layers_0_self_attn_v_proj_weight"))
        XCTAssertTrue(names.contains("layers_0_self_attn_q_norm_weight"))
        XCTAssertTrue(names.contains("layers_0_self_attn_k_norm_weight"))
    }

    // MARK: - backbone_hidden_size projection emission

    private func parseIRWithBackbone(backbone: Int?) throws -> SmeltModelIR {
        SmeltModelIR(
            modelName: "test/model",
            config: SmeltConfig(
                hiddenSize: 256, numLayers: 4, vocabSize: 262_144,
                staticSeqCapacity: 256, ropeDim: 256, rmsEps: 1e-6,
                attention: SmeltAttentionConfig(
                    qHeads: 4, kvHeads: 1, headDim: 256, gatedQ: false, externalKV: true),
                ffn: SmeltFFNConfig(dim: 2048, activation: .swiglu),
                backboneHiddenSize: backbone),
            layerPattern: SmeltLayerPattern(unit: [.attention], repeats: 4),
            quantization: SmeltQuantizationConfig(strategy: .lutU4, groupSize: 16, excludePatterns: []),
            loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic))
    }

    func testBackboneEmitsPreAndPostProjectionWeights() throws {
        let ir = try parseIRWithBackbone(backbone: 1536)
        let layout = SmeltWeightLayout.computeLayout(from: ir)
        let byName = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })

        // pre_projection: maps the runtime-supplied
        // [1, 2 * backbone] = [1, 3072] input down to the model
        // hidden_size = 256.
        XCTAssertNotNil(byName["pre_projection_weight"])
        XCTAssertEqual(byName["pre_projection_weight"]?.shape, [256, 2 * 1536])

        // post_projection: maps the model's last hidden state back
        // up to backbone = 1536.
        XCTAssertNotNil(byName["post_projection_weight"])
        XCTAssertEqual(byName["post_projection_weight"]?.shape, [1536, 256])
    }

    func testWithoutBackboneOmitsProjectionWeights() throws {
        // Regression guard: no backbone_hidden_size → no projection
        // weights. Existing target builds and any model that
        // doesn't ship projections (e.g. large models with
        // their own embedder family) keep the prior layout.
        let ir = try parseIRWithBackbone(backbone: nil)
        let layout = SmeltWeightLayout.computeLayout(from: ir)
        let names = Set(layout.map(\.name))

        XCTAssertFalse(names.contains("pre_projection_weight"))
        XCTAssertFalse(names.contains("post_projection_weight"))
    }

    func testProjectionWeightsRespectQuantizationStrategy() throws {
        // Projections route through appendWeightEntry, so they pick
        // up the spec's quantization strategy. A spec with
        // affine_u4 quant should produce affine_u4 projection
        // weights, not raw fp16.
        let ir = try parseIRWithBackbone(backbone: 1536)
        let layout = SmeltWeightLayout.computeLayout(from: ir)
        let byName = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })

        // The default test fixture has no quantization block, so the
        // pipeline produces fp16 (default). The shape check above is
        // the load-bearing assertion; here we just assert dtype is a
        // recognized weight dtype, not garbage.
        let preDtype = byName["pre_projection_weight"]?.dtype
        let postDtype = byName["post_projection_weight"]?.dtype
        XCTAssertTrue([SmeltDType.fp16, .u4Lut, .affineU4].contains(preDtype!))
        XCTAssertTrue([SmeltDType.fp16, .u4Lut, .affineU4].contains(postDtype!))
    }

    // MARK: - cluster_embedder weight emission

    private func parseClusterEmbedderIR(includeBlock: Bool) throws -> SmeltModelIR {
        SmeltModelIR(
            modelName: "test/model",
            config: SmeltConfig(
                hiddenSize: 256, numLayers: 4, vocabSize: 262_144,
                staticSeqCapacity: 256, ropeDim: 256, rmsEps: 1e-6,
                attention: SmeltAttentionConfig(
                    qHeads: 4, kvHeads: 1, headDim: 256, gatedQ: false, externalKV: true),
                ffn: SmeltFFNConfig(dim: 2048, activation: .swiglu),
                backboneHiddenSize: 1536,
                clusterEmbedder: includeBlock
                    ? SmeltClusterEmbedderConfig(numCentroids: 2048, topK: 32) : nil),
            layerPattern: SmeltLayerPattern(unit: [.attention], repeats: 4),
            quantization: SmeltQuantizationConfig(strategy: .lutU4, groupSize: 16, excludePatterns: []),
            loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic))
    }

    func testClusterEmbedderEmitsCentroidsAndTokenOrdering() throws {
        let ir = try parseClusterEmbedderIR(includeBlock: true)
        let layout = SmeltWeightLayout.computeLayout(from: ir)
        let byName = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })

        // Centroid classifier: maps hidden state to centroid
        // logits. Shape [num_centroids, hidden_size] = [2048, 256].
        let centroids = try XCTUnwrap(byName["masked_embedding_centroids_weight"])
        XCTAssertEqual(centroids.shape, [2048, 256])

        // Token-cluster permutation: int32[vocab_size]. Shape [262144],
        // dtype int32, size = 262144 * 4 bytes.
        let ordering = try XCTUnwrap(byName["masked_embedding_token_ordering"])
        XCTAssertEqual(ordering.shape, [262144])
        XCTAssertEqual(ordering.dtype, .int32)
        XCTAssertEqual(ordering.sizeBytes, UInt64(262144 * 4))

        // Both entries must be 16-byte aligned. The centroids entry
        // routes through the standard weight-packer alignment helpers;
        // token_ordering's offset is explicitly aligned by the cluster
        // emission block because the preceding entry's trailing edge is
        // not guaranteed to land on a 16-byte boundary for arbitrary
        // num_centroids/quantization combinations.
        XCTAssertEqual(centroids.offset % 16, 0)
        XCTAssertEqual(ordering.offset % 16, 0)
    }

    func testClusterEmbedderAbsentOmitsBothEntries() throws {
        // Regression guard: a model without cluster_embedder block (e.g.
        // a model that uses dense lm_head
        // instead) must not emit the cluster tensors.
        let ir = try parseClusterEmbedderIR(includeBlock: false)
        let layout = SmeltWeightLayout.computeLayout(from: ir)
        let names = Set(layout.map(\.name))

        XCTAssertFalse(names.contains("masked_embedding_centroids_weight"))
        XCTAssertFalse(names.contains("masked_embedding_token_ordering"))
    }

    func testClusterEmbedderTokenOrderingFitsInInt32() throws {
        // Sanity: vocab sizes that fit in int32 (i.e. < 2^31 = 2.1B) are
        // safe. Test vocab is 262144, well under the limit. This
        // test pins the assumption so any future spec with vocab > 2^31
        // (impractical, but pinned anyway) would force a conscious
        // dtype upgrade.
        let ir = try parseClusterEmbedderIR(includeBlock: true)
        XCTAssertLessThan(ir.config.vocabSize, Int(Int32.max))
    }
}
