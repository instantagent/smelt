import Foundation
import Testing
@testable import SmeltCompiler

// MARK: - IR validation tests

@Test func qwen35IRPassesValidation() throws {
    // The reference config must pass all validation — if this fails, the IR is wrong.
    try validateSmeltIR(SmeltModelIR.qwen35_2B)
}

@Test func qwen35IRDimensions() {
    let ir = SmeltModelIR.qwen35_2B
    #expect(ir.totalLayers == 24)
    #expect(ir.numDeltaLayers == 18)
    #expect(ir.numAttnLayers == 6)
    #expect(ir.config.hiddenSize == 2_048)
    #expect(ir.config.vocabSize == 248_320)
    #expect(ir.config.staticSeqCapacity == 256)
    #expect(ir.config.ropeDim == 64)
    #expect(ir.config.normMode == .onePlusWeight)
    #expect(ir.config.delta!.numHeads == 16)
    #expect(ir.config.delta!.headDim == 128)
    #expect(ir.config.delta!.qkvDim == 6_144)
    #expect(ir.config.attention!.qHeads == 8)
    #expect(ir.config.attention!.kvHeads == 2)
    #expect(ir.config.attention!.headDim == 256)
    #expect(ir.config.attention!.gqaRatio == 4)
    #expect(ir.config.attention!.gatedQ == true)
    #expect(ir.config.attention!.ropeTheta == 10_000_000)
    #expect(ir.config.attention!.qProjDim == 4_096) // 8 * 256 * 2
    #expect(ir.config.ffn.dim == 6_144)
}

@Test func qwen3508IRPassesValidation() throws {
    try validateSmeltIR(SmeltModelIR.qwen35_0_8B)
}

@Test func qwen3508IRDimensions() {
    let ir = SmeltModelIR.qwen35_0_8B
    #expect(ir.totalLayers == 24)
    #expect(ir.numDeltaLayers == 18)
    #expect(ir.numAttnLayers == 6)
    #expect(ir.config.hiddenSize == 1_024)
    #expect(ir.config.vocabSize == 248_320)
    #expect(ir.config.staticSeqCapacity == nil)
    #expect(ir.config.ropeDim == 64)
    #expect(ir.config.normMode == .onePlusWeight)
    #expect(ir.config.delta!.numHeads == 16)
    #expect(ir.config.delta!.headDim == 128)
    #expect(ir.config.delta!.qkvDim == 6_144)
    #expect(ir.config.delta!.zDim == 2_048)
    #expect(ir.config.delta!.valueDim == 2_048)
    #expect(ir.config.attention!.qHeads == 8)
    #expect(ir.config.attention!.kvHeads == 2)
    #expect(ir.config.attention!.headDim == 256)
    #expect(ir.config.attention!.gqaRatio == 4)
    #expect(ir.config.attention!.gatedQ == true)
    #expect(ir.config.attention!.ropeTheta == 10_000_000)
    #expect(ir.config.attention!.qProjDim == 4_096)
    #expect(ir.config.ffn.dim == 3_584)
}

@Test func qwen354BIRPassesValidation() throws {
    try validateSmeltIR(SmeltModelIR.qwen35_4B)
}

@Test func qwen354BIRDimensions() {
    let ir = SmeltModelIR.qwen35_4B
    #expect(ir.totalLayers == 32)
    #expect(ir.numDeltaLayers == 24)
    #expect(ir.numAttnLayers == 8)
    #expect(ir.config.hiddenSize == 2_560)
    #expect(ir.config.vocabSize == 248_320)
    #expect(ir.config.staticSeqCapacity == nil)
    #expect(ir.config.ropeDim == 64)
    #expect(ir.config.normMode == .onePlusWeight)
    #expect(ir.config.delta!.numHeads == 32)
    #expect(ir.config.delta!.headDim == 128)
    #expect(ir.config.delta!.qkvDim == 8_192)
    #expect(ir.config.delta!.zDim == 4_096)
    #expect(ir.config.delta!.valueDim == 4_096)
    #expect(ir.config.delta!.qkHeads == 16)
    #expect(ir.config.delta!.qkRepeatFactor == 2)
    #expect(ir.config.attention!.qHeads == 16)
    #expect(ir.config.attention!.kvHeads == 4)
    #expect(ir.config.attention!.headDim == 256)
    #expect(ir.config.attention!.gqaRatio == 4)
    #expect(ir.config.attention!.gatedQ == true)
    #expect(ir.config.attention!.ropeTheta == 10_000_000)
    #expect(ir.config.attention!.qProjDim == 8_192)
    #expect(ir.config.ffn.dim == 9_216)
}

@Test func qwen35LayerPattern() {
    let pattern = SmeltModelIR.qwen35_2B.layerPattern
    let expanded = pattern.expanded
    #expect(expanded.count == 24)
    // Pattern: [delta, delta, delta, attn] × 6
    for group in 0..<6 {
        let base = group * 4
        #expect(expanded[base] == .delta)
        #expect(expanded[base + 1] == .delta)
        #expect(expanded[base + 2] == .delta)
        #expect(expanded[base + 3] == .attention)
    }
}

@Test func layerCountMismatchRejected() {
    // Pattern says 24 layers, but config says 12
    var ir = SmeltModelIR.qwen35_2B
    ir = SmeltModelIR(
        modelName: ir.modelName,
        config: SmeltConfig(
            hiddenSize: 2_048, numLayers: 12, vocabSize: 248_320,
            staticSeqCapacity: 256, ropeDim: 64, rmsEps: 1e-6,
            delta: ir.config.delta,
            attention: ir.config.attention,
            ffn: ir.config.ffn
        ),
        layerPattern: ir.layerPattern,
        quantization: ir.quantization,
        loading: ir.loading,
        prefill: ir.prefill
    )
    #expect(throws: SmeltIRValidationError.self) {
        try validateSmeltIR(ir)
    }
}

@Test func badQKVDimRejected() {
    let ir = SmeltModelIR(
        modelName: "test",
        config: SmeltConfig(
            hiddenSize: 2_048, numLayers: 24, vocabSize: 248_320,
            staticSeqCapacity: 256, ropeDim: 64, rmsEps: 1e-6,
            delta: SmeltDeltaConfig(
                numHeads: 16, headDim: 128, convKernel: 4,
                qkvDim: 5_000, zDim: 2_048, aDim: 16, bDim: 16 // invalid compact Q/K layout
            ),
            attention: SmeltModelIR.qwen35_2B.config.attention,
            ffn: SmeltModelIR.qwen35_2B.config.ffn
        ),
        layerPattern: SmeltLayerPattern(unit: [.delta, .delta, .delta, .attention], repeats: 6),
        quantization: SmeltModelIR.qwen35_2B.quantization,
        loading: SmeltModelIR.qwen35_2B.loading
    )
    #expect(throws: SmeltIRValidationError.self) {
        try validateSmeltIR(ir)
    }
}

@Test func badGQARatioRejected() {
    let ir = SmeltModelIR(
        modelName: "test",
        config: SmeltConfig(
            hiddenSize: 2_048, numLayers: 24, vocabSize: 248_320,
            staticSeqCapacity: 256, ropeDim: 64, rmsEps: 1e-6,
            delta: SmeltModelIR.qwen35_2B.config.delta,
            attention: SmeltAttentionConfig(
                qHeads: 8, kvHeads: 3, headDim: 256, gatedQ: true // 8/3 is not integer
            ),
            ffn: SmeltModelIR.qwen35_2B.config.ffn
        ),
        layerPattern: SmeltLayerPattern(unit: [.delta, .delta, .delta, .attention], repeats: 6),
        quantization: SmeltModelIR.qwen35_2B.quantization,
        loading: SmeltModelIR.qwen35_2B.loading
    )
    // This will fail at attn hidden size mismatch first (8*256=2048, but 3 kvHeads => gqa not integer)
    #expect(throws: SmeltIRValidationError.self) {
        try validateSmeltIR(ir)
    }
}

// U3 (the activation-dtype axis). qwen35_2B validates at fp16; reconstructing its config with only
// the activation dtype flipped isolates the activation guard as the failure, so these are non-vacuous.

/// qwen35_2B (which has DeltaNet layers) with only `activationDtype` overridden — otherwise valid, so
/// `validateSmeltIR` fails ONLY on the activation guard.
private func qwen35IR(activationDtype: SmeltDType) -> SmeltModelIR {
    let c = SmeltModelIR.qwen35_2B.config
    return SmeltModelIR(
        modelName: "test",
        config: SmeltConfig(
            hiddenSize: c.hiddenSize, numLayers: c.numLayers, vocabSize: c.vocabSize,
            staticSeqCapacity: c.staticSeqCapacity, ropeDim: c.ropeDim, rmsEps: c.rmsEps,
            normMode: c.normMode, activationDtype: activationDtype,
            portTopology: activationDtype == .fp16
                ? .tokenInLogitsOut
                : .embeddingsInHiddenOut,
            delta: c.delta, attention: c.attention, ffn: c.ffn),
        layerPattern: SmeltModelIR.qwen35_2B.layerPattern,
        quantization: SmeltModelIR.qwen35_2B.quantization,
        loading: SmeltModelIR.qwen35_2B.loading)
}

/// Dense non-FP16 activation paths do not yet lower DeltaNet layers.
@Test func fp32ActivationRejectsDeltaNet() throws {
    do {
        try validateSmeltIR(qwen35IR(activationDtype: .fp32))
        Issue.record("fp32 activation with DeltaNet layers was not rejected")
    } catch let SmeltIRValidationError.unsupportedConfiguration(msg) {
        #expect(msg.contains("DeltaNet"))
    }
}

@Test func bf16ActivationRejectsDeltaNet() throws {
    do {
        try validateSmeltIR(qwen35IR(activationDtype: .bf16))
        Issue.record("bf16 activation with DeltaNet layers was not rejected")
    } catch let SmeltIRValidationError.unsupportedConfiguration(msg) {
        #expect(msg.contains("DeltaNet"))
    }
}

@Test func bf16ActivationValidatesForDenseAttentionTrunks() throws {
    let ir = SmeltModelIR.denseTrunk(
        modelName: "bf16-dense-trunk",
        hidden: 128,
        numLayers: 2,
        vocab: 256,
        heads: 4,
        kvHeads: 2,
        headDim: 32,
        inter: 256,
        maxPrefillBatch: 64,
        staticSeqCapacity: 128,
        activationDtype: .bf16
    )
    try validateSmeltIR(ir)
}

@Test func oddRopeDimRejected() {
    let ir = SmeltModelIR(
        modelName: "test",
        config: SmeltConfig(
            hiddenSize: 2_048, numLayers: 24, vocabSize: 248_320,
            staticSeqCapacity: 256, ropeDim: 63, rmsEps: 1e-6, // odd
            delta: SmeltModelIR.qwen35_2B.config.delta,
            attention: SmeltModelIR.qwen35_2B.config.attention,
            ffn: SmeltModelIR.qwen35_2B.config.ffn
        ),
        layerPattern: SmeltLayerPattern(unit: [.delta, .delta, .delta, .attention], repeats: 6),
        quantization: SmeltModelIR.qwen35_2B.quantization,
        loading: SmeltModelIR.qwen35_2B.loading
    )
    #expect(throws: SmeltIRValidationError.self) {
        try validateSmeltIR(ir)
    }
}

@Test func largeStaticSeqCapacityPassesValidationWhenAttentionUsesGenericPath() throws {
    let ir = SmeltModelIR(
        modelName: "test",
        config: SmeltConfig(
            hiddenSize: 2_048, numLayers: 24, vocabSize: 248_320,
            staticSeqCapacity: 512, ropeDim: 64, rmsEps: 1e-6,
            delta: SmeltModelIR.qwen35_2B.config.delta,
            attention: SmeltModelIR.qwen35_2B.config.attention,
            ffn: SmeltModelIR.qwen35_2B.config.ffn
        ),
        layerPattern: SmeltLayerPattern(unit: [.delta, .delta, .delta, .attention], repeats: 6),
        quantization: SmeltModelIR.qwen35_2B.quantization,
        loading: SmeltModelIR.qwen35_2B.loading
    )
    try validateSmeltIR(ir)
}

@Test func badGroupSizeRejected() {
    let ir = SmeltModelIR(
        modelName: "test",
        config: SmeltModelIR.qwen35_2B.config,
        layerPattern: SmeltModelIR.qwen35_2B.layerPattern,
        quantization: SmeltQuantizationConfig(
            strategy: .lutU4, groupSize: 7, excludePatterns: [] // not power of 2
        ),
        loading: SmeltModelIR.qwen35_2B.loading
    )
    #expect(throws: SmeltIRValidationError.self) {
        try validateSmeltIR(ir)
    }
}

// MARK: - Buffer plan tests

@Test func qwen35BufferPlanSlotCount() throws {
    try validateSmeltIR(SmeltModelIR.qwen35_2B)
    let plan = buildBufferPlan(from: SmeltModelIR.qwen35_2B)

    // Verify fixed activation slots exist
    let slotIndices = Set(plan.slots.map(\.index))
    #expect(slotIndices.contains(SmeltFixedSlot.hiddenA.rawValue))
    #expect(slotIndices.contains(SmeltFixedSlot.hiddenB.rawValue))
    #expect(slotIndices.contains(SmeltFixedSlot.argmaxBuf.rawValue))
    #expect(slotIndices.contains(SmeltFixedSlot.weights.rawValue))

    // Verify dynamic slots are contiguous starting at 33. The dynamic
    // base sits one above SmeltFixedSlot.tqhMatvecXHatBuf.rawValue
    // (32) — both that slot and centroidLogitsBuf are only registered
    // for the packages that use them, but their rawValues advance the
    // dynamic floor unconditionally to keep slot indices stable
    // across TQH-matvec and non-TQH builds.
    #expect(plan.convStateBaseSlot == 33)
    #expect(plan.recStateBaseSlot == 33 + 18)  // 51
    #expect(plan.keyCacheBaseSlot == 33 + 18 + 18)  // 69
    #expect(plan.valCacheBaseSlot == 33 + 18 + 18 + 6)  // 75

    // Verify no slot collisions
    let counts = Dictionary(grouping: plan.slots, by: \.index)
    for (idx, group) in counts where group.count > 1 {
        #expect(Bool(false), "Slot \(idx) has \(group.count) entries (collision)")
    }

    // 27 fixed + 1 weight + 18 conv + 18 rec + 6 key + 6 val + 2 rope
    // + 2 reusable MLX attention scratch + 2 dynamic = 82
    #expect(plan.slots.count == 82)
}

@Test func qwen35BufferPlanSizes() throws {
    try validateSmeltIR(SmeltModelIR.qwen35_2B)
    let plan = buildBufferPlan(from: SmeltModelIR.qwen35_2B)

    let byIndex = Dictionary(uniqueKeysWithValues: plan.slots.map { ($0.index, $0) })

    // hiddenA: 2048 * 2 = 4096 bytes
    #expect(byIndex[0]?.sizeBytes == 4_096)
    // qkvBuf: 6144 * 2 = 12288 bytes
    #expect(byIndex[2]?.sizeBytes == 12_288)
    // logitsBuf: 248320 * 2 = 496640 bytes
    #expect(byIndex[16]?.sizeBytes == 496_640)
    // argmaxBuf: min 16 bytes (aligned up from 8)
    #expect(byIndex[17]?.sizeBytes == 16)
    // kvMemBuf: 16 * 128 * 4 = 8192 bytes (FP32)
    #expect(byIndex[18]?.sizeBytes == 8_192)
    // attnQBuf: 8 * 256 * 2 * 2 = 8192 bytes (gated)
    #expect(byIndex[20]?.sizeBytes == 8_192)
    // convState_0: qkvDim * convKernel * fp16 = 6144 * 4 * 2 = 49152 bytes
    #expect(byIndex[plan.convStateBaseSlot]?.sizeBytes == 49_152)
    // recState_0: 16 * 128 * 128 * 2 = 524288 bytes (FP16 recurrent state)
    #expect(byIndex[plan.recStateBaseSlot]?.sizeBytes == 524_288)
    // keyCache_0: 2 * 256 * 256 * 2 = 262144 bytes
    #expect(byIndex[plan.keyCacheBaseSlot]?.sizeBytes == 262_144)
    #expect(byIndex[plan.keyCacheBaseSlot]?.shape == [2, 256, 256])
    // ropeCos: 256 * 64 * 2 = 32768 bytes
    #expect(byIndex[plan.ropeCosSlot]?.sizeBytes == 32_768)
    let mlxPartials = try #require(plan.slots.first {
        $0.name == "mlxAttentionPartialsD256B128"
    })
    let mlxStats = try #require(plan.slots.first {
        $0.name == "mlxAttentionStatsD256B128"
    })
    #expect(mlxPartials.shape == [8, 128, 256])
    #expect(mlxPartials.sizeBytes == 524_288)
    #expect(mlxStats.shape == [2, 8, 128])
    #expect(mlxStats.sizeBytes == 8_192)
    // dynamic scalars: min 16 bytes (aligned up from 4)
    #expect(byIndex[plan.tokenIdSlot]?.sizeBytes == 16)
    #expect(byIndex[plan.positionSlot]?.sizeBytes == 16)
}

@Test func bufferPlanDeterministic() throws {
    try validateSmeltIR(SmeltModelIR.qwen35_2B)
    let plan1 = buildBufferPlan(from: SmeltModelIR.qwen35_2B)
    let plan2 = buildBufferPlan(from: SmeltModelIR.qwen35_2B)
    // Same IR must produce identical plans
    #expect(plan1.slots.count == plan2.slots.count)
    for (slot1, slot2) in zip(plan1.slots, plan2.slots) {
        #expect(slot1.index == slot2.index)
        #expect(slot1.sizeBytes == slot2.sizeBytes)
        #expect(slot1.name == slot2.name)
    }
}

@Test func bufferPlanToManifestTable() throws {
    try validateSmeltIR(SmeltModelIR.qwen35_2B)
    let plan = buildBufferPlan(from: SmeltModelIR.qwen35_2B)
    let table = plan.toBufferTable()
    #expect(table.slots.count == plan.slots.count)
    // Verify JSON round-trip works
    let encoded = try JSONEncoder().encode(table)
    let decoded = try JSONDecoder().decode(SmeltBufferTable.self, from: encoded)
    #expect(decoded.slots.count == table.slots.count)
}

@Test func manifestJSONRoundTrip() throws {
    let plan = buildBufferPlan(from: SmeltModelIR.qwen35_2B)
    let manifest = SmeltManifest(
        blocks: .tokenFeedbackText,
        loop: .tokenFeedbackText,
        modelName: "Qwen/Qwen3.5-2B",
        config: SmeltManifestConfig(
            hiddenSize: 2_048, numLayers: 24, vocabSize: 248_320,
            staticSeqCapacity: 256, ropeDim: 64, numDeltaLayers: 18, numAttnLayers: 6,
            deltaNumHeads: 16, deltaQKVDim: 6_144,
            attnQProjDim: 2_048, attnKProjDim: 512, attnVProjDim: 512,
            attnOutDim: 2_048, ffnDim: 6_144
        ),
        context: nil,
        checksums: SmeltManifestChecksums(
            weightsBin: "abc123",
            metallib: "def456",
            generatedSwift: "ghi789",
            dispatchesBin: "jkl012",
            prefillDispatchesBin: "mno345",
            tokenizerJSON: "pqr678"
        ),
        buildProvenance: SmeltBuildProvenance(
            buildFingerprint: "build123",
            weightsFingerprint: "weights123",
            specSHA256: "spec123",
            compilerSourcesSHA256: "compiler123",
            shaderSourcesSHA256: "shader123",
            resolvedOptions: SmeltResolvedBuildOptions(
                layerPatternUnit: ["delta", "delta", "delta", "attn"],
                layerPatternRepeats: 6,
                quantizationStrategy: "affine_u4",
                groupSize: 64,
                excludePatterns: ["conv1d_weight", "*_norm_weight"],
                quantizeEmbedding: true,
                loadingStrategy: "mmap_prefault",
                packing: "monolithic",
                prefillEngine: "metal",
                maxPrefillBatch: 256,
                prefillHandoffFamilies: ["conv_state", "rec_state"],
                inferenceMaxTokens: 512,
                eosTokens: [151645],
                thinkToken: 151648,
                thinkEndToken: 151649,
                thinkSkipSuffix: 198,
                tiedLMHead: true,
                traceMode: "full"
            )
        ),
        device: SmeltDeviceRequirements(
            metalFamily: .apple7, minMemoryBytes: 2_000_000_000
        ),
        weights: SmeltWeightManifest(totalBytes: 1_000_000, entries: [
            SmeltWeightEntry(
                name: "embed_tokens.weight", offset: 0,
                sizeBytes: 500_000, shape: [248_320, 2_048], dtype: .fp16
            )
        ]),
        buffers: plan.toBufferTable(),
        pipelines: ["fused_lut_matvec", "rms_norm_1pw"],
        slotLayout: plan.toSlotLayout(),
        inference: SmeltInferenceManifest(
            maxTokens: 512,
            eosTokens: [151645],
            thinkToken: 151648,
            thinkEndToken: 151649,
            thinkSkipSuffix: 198,
            chatTemplate: "chatml",
            thinkingPolicy: .disabled
        ),
        decode: SmeltPackageSpec.DecodePolicy(
            sampler: SmeltPackageSpec.DecodePolicy.Sampler(mode: .greedy),
            maxSteps: 512
        ),
        validation: SmeltPackageSpec.Validation(
            parityFixture: "manifest-round-trip",
            performanceGate: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
            performanceProfile: SmeltPackagePerformanceProfiles.profile(
                for: SmeltPackagePerformanceGateID.textDecodePrefillStartup,
                modelName: "Qwen/Qwen3.5-2B"
            )
        )
    )

    let json = try manifest.encodeJSON()
    let decoded = try SmeltManifest.decode(from: json)
    #expect(decoded.modelName == "Qwen/Qwen3.5-2B")
    #expect(decoded.config.hiddenSize == 2_048)
    #expect(decoded.checksums.weightsBin == "abc123")
    #expect(decoded.checksums.dispatchesBin == "jkl012")
    #expect(decoded.checksums.prefillDispatchesBin == "mno345")
    #expect(decoded.checksums.tokenizerJSON == "pqr678")
    #expect(decoded.buildProvenance?.buildFingerprint == "build123")
    #expect(decoded.buildProvenance?.weightsFingerprint == "weights123")
    #expect(decoded.buildProvenance?.resolvedOptions.maxPrefillBatch == 256)
    #expect(decoded.weights.entries.count == 1)
    #expect(decoded.pipelines.count == 2)
    #expect(decoded.slotLayout.convStateBaseSlot == plan.convStateBaseSlot)
    #expect(decoded.slotLayout.weightsSlot == SmeltFixedSlot.weights.rawValue)
    #expect(decoded.blocks == SmeltBlockGraph.tokenFeedbackText)
    #expect(decoded.loop == SmeltLoopSchedule.tokenFeedbackText)
    #expect(decoded.inference?.chatTemplate == "chatml")
    #expect(decoded.decode?.maxSteps == 512)
    #expect(decoded.validation?.performanceGate == SmeltPackagePerformanceGateID.textDecodePrefillStartup)
}

@Test func manifestResolvedBuildOptionsDefaultTraceModeToFull() throws {
    let json = """
    {
      "version": 1,
      "modelName": "test/model",
      "blocks": {
        "version": 1,
        "blocks": [
          {
            "name": "tokenizer",
            "role": "frontend",
            "impl": "native",
            "inputs": ["text"],
            "output": "tokens"
          },
          {
            "name": "trunk",
            "role": "trunk",
            "impl": "compiled",
            "inputs": ["tokens"],
            "output": "logits",
            "feedback": "tokens",
            "state": ["kv-cache"],
            "compiledDelivery": "baked-inline"
          },
          {
            "name": "text-head",
            "role": "head",
            "impl": "native",
            "inputs": ["logits"],
            "output": "text",
            "state": ["sampler"]
          }
        ]
      },
      "loop": {
        "version": 1,
        "setup": [
          {"name": "prefill", "blocks": ["trunk", "text-head"]}
        ],
        "perStep": [
          {"name": "decode", "blocks": ["trunk", "text-head"]}
        ],
        "emission": {"perStep": {}},
        "stop": ["eos-token", "max-steps", "host-cancel"]
      },
      "inference": {
        "max_tokens": 8,
        "eos_tokens": [1],
        "chat_template": "chatml",
        "thinking_policy": "disabled"
      },
      "decode": {
        "sampler": {
          "mode": "greedy"
        },
        "max_steps": 8
      },
      "validation": {
        "parity_fixture": "ir-validation",
        "performance_gate": "text.decode-prefill-startup",
        "performance_profile": {
          "gate": "text.decode-prefill-startup",
          "command": "run",
          "required_trace_labels": [
            "exec -> main (dyld)",
            "tokenizer load",
            "SmeltModel init (total)"
          ],
          "required_output_metrics": [
            "decode_median_ms_per_token",
            "decode_tokens_per_second",
            "decode_p95_ms_per_token",
            "prefill64_wall_ms",
            "prefill64_tokens_per_second",
            "prefill64_p95_ms",
            "prefill256_wall_ms",
            "prefill256_tokens_per_second",
            "prefill256_p95_ms",
            "trace_first_token_ms"
          ],
          "max_bounds": [
            {"metric": "trace_first_token_ms", "max": 100, "unit": "ms"}
          ]
        }
      },
      "config": {
        "hiddenSize": 1,
        "numLayers": 1,
        "vocabSize": 1,
        "staticSeqCapacity": 1,
        "ropeDim": 1,
        "numDeltaLayers": 0,
        "numAttnLayers": 0,
        "deltaNumHeads": 0,
        "deltaQKVDim": 0,
        "attnQProjDim": 0,
        "attnKProjDim": 0,
        "attnVProjDim": 0,
        "attnOutDim": 0,
        "ffnDim": 1
      },
      "checksums": {
        "weights_bin": "a",
        "metallib": "b",
        "generated_swift": "c",
        "dispatches_bin": "d"
      },
      "buildProvenance": {
        "buildFingerprint": "build123",
        "weightsFingerprint": "weights123",
        "specSHA256": "spec123",
        "compilerSourcesSHA256": "compiler123",
        "shaderSourcesSHA256": "shader123",
        "resolvedOptions": {
          "layerPatternUnit": ["attn"],
          "layerPatternRepeats": 1,
          "quantizationStrategy": "affine_u4",
          "groupSize": 128,
          "excludePatterns": [],
          "quantizeEmbedding": true,
          "loadingStrategy": "mmap_prefault",
          "packing": "monolithic",
          "prefillHandoffFamilies": [],
          "inferenceMaxTokens": 8,
          "eosTokens": [],
          "tiedLMHead": true
        }
      },
      "device": {
        "metal_family": "apple7",
        "min_memory_bytes": 1
      },
      "weights": {
        "total_bytes": 0,
        "entries": []
      },
      "buffers": {
        "slots": []
      },
      "pipelines": [],
      "slotLayout": {
        "conv_state_base": 0,
        "rec_state_base": 0,
        "key_cache_base": 0,
        "val_cache_base": 0,
        "rope_cos": 0,
        "rope_sin": 0,
        "token_id": 0,
        "position": 0,
        "weights": 0
      }
    }
    """.data(using: .utf8)!

    let decoded = try SmeltManifest.decode(from: json)
    #expect(decoded.buildProvenance?.resolvedOptions.traceMode == "full")
}

@Test func qwen35TotalActivationBytes() throws {
    try validateSmeltIR(SmeltModelIR.qwen35_2B)
    let plan = buildBufferPlan(from: SmeltModelIR.qwen35_2B)
    // Verified by hand against FP16 recurrent state, split argmax, and the
    // reusable D256 two-pass attention scratch (524,288 B partials + 8,192 B
    // stats for eight query heads).
    // This pins the exact value so any formula change is caught.
    #expect(plan.totalActivationBytes == 14_680_704)
}

@Test func headScaleValues() {
    let ir = SmeltModelIR.qwen35_2B
    // DeltaNet: 1/sqrt(128) ≈ 0.0883883
    #expect(abs(ir.config.delta!.headScale - 0.0883883) < 0.0001)
    // Attention: 1/sqrt(256) = 0.0625
    #expect(ir.config.attention!.headScale == 0.0625)
}

@Test func effectiveAttentionScoreScaleValues() throws {
    let qwen = SmeltModelIR.qwen35_2B.config.attention!
    #expect(abs(qwen.effectiveScoreScale(blockTopology: .standard) - 0.0625) < 0.0001)
}

// MARK: - Prefill buffer plan widening

/// Helper: Qwen 3.5 2B with Metal prefill engine.
private func qwen35MetalPrefill(batchSize: Int = 64) -> SmeltModelIR {
    var ir = SmeltModelIR.qwen35_2B
    ir = SmeltModelIR(
        modelName: ir.modelName,
        config: ir.config,
        layerPattern: ir.layerPattern,
        quantization: ir.quantization,
        loading: ir.loading,
        prefill: SmeltPrefillConfig(
            engine: "metal",
            modelPath: "",
            cachePath: "cache",
            maxBatchSize: batchSize,
            handoffFamilies: ["conv_state", "rec_state", "key_cache", "value_cache", "rope"]
        )
    )
    return ir
}

private func qwen35MetalPrefillVerifyArgmax(
    batchSize: Int = 64,
    verifyTokenCapacity: Int? = nil
) -> SmeltModelIR {
    let ir = SmeltModelIR.qwen35_2B
    return SmeltModelIR(
        modelName: ir.modelName,
        config: ir.config,
        layerPattern: ir.layerPattern,
        quantization: ir.quantization,
        loading: ir.loading,
        prefill: SmeltPrefillConfig(
            engine: "metal",
            modelPath: "",
            cachePath: "cache",
            maxBatchSize: batchSize,
            handoffFamilies: ["conv_state", "rec_state", "key_cache", "value_cache", "rope"],
            verifyArgmax: true,
            verifyTokenCapacity: verifyTokenCapacity
        )
    )
}

@Test func prefillBufferPlanWidensActivations() throws {
    let ir = qwen35MetalPrefill(batchSize: 64)
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let byIndex = Dictionary(uniqueKeysWithValues: plan.slots.map { ($0.index, $0) })

    // hiddenA: 2048 * 64 * 2 = 262144 bytes (64× wider)
    #expect(byIndex[0]?.sizeBytes == 2048 * 64 * 2)
    // qkvBuf: 6144 * 64 * 2 = 786432 bytes
    #expect(byIndex[2]?.sizeBytes == 6144 * 64 * 2)
    // ffnGateBuf: 6144 * 64 * 2
    #expect(byIndex[11]?.sizeBytes == 6144 * 64 * 2)
    // attnQBuf: 8*256*2 * 64 * 2 = 524288 bytes (gated Q)
    #expect(byIndex[20]?.sizeBytes == 8 * 256 * 2 * 64 * 2)
}

@Test func prefillBufferPlanKeepsPersistentStateAndUsesDynamicContextSlots() throws {
    let irDecode = SmeltModelIR.qwen35_2B
    let irPrefill = qwen35MetalPrefill(batchSize: 64)
    try validateSmeltIR(irDecode)
    try validateSmeltIR(irPrefill)
    let decodePlan = buildBufferPlan(from: irDecode)
    let prefillPlan = buildBufferPlan(from: irPrefill)

    // Conv and recurrence state are persistent model state. KV, RoPE, and mask
    // are request-context scoped and dynamic packages start them at one row.
    let decodeByIdx = Dictionary(uniqueKeysWithValues: decodePlan.slots.map { ($0.index, $0) })
    let prefillByIdx = Dictionary(uniqueKeysWithValues: prefillPlan.slots.map { ($0.index, $0) })

    #expect(decodeByIdx[decodePlan.convStateBaseSlot]?.sizeBytes ==
            prefillByIdx[prefillPlan.convStateBaseSlot]?.sizeBytes)
    #expect(decodeByIdx[decodePlan.recStateBaseSlot]?.sizeBytes ==
            prefillByIdx[prefillPlan.recStateBaseSlot]?.sizeBytes)
    #expect(prefillByIdx[prefillPlan.keyCacheBaseSlot]?.sizeBytes == 2 * 1 * 256 * 2)
    #expect(prefillByIdx[prefillPlan.keyCacheBaseSlot]?.shape == [2, 0, 256])
    #expect(prefillByIdx[prefillPlan.ropeCosSlot]?.sizeBytes == 1 * 64 * 2)
    #expect(prefillByIdx[prefillPlan.ropeCosSlot]?.shape == [0, 64])
}

@Test func prefillBufferPlanLogitsNotWidened() throws {
    let ir = qwen35MetalPrefill(batchSize: 64)
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let byIndex = Dictionary(uniqueKeysWithValues: plan.slots.map { ($0.index, $0) })

    // logitsBuf: vocabSize * 2 (NOT * 64) — LM head uses matvec on last token only
    #expect(byIndex[16]?.sizeBytes == 248320 * 2)
    // argmaxBuf unchanged
    #expect(byIndex[17]?.sizeBytes == 16)
}

@Test func prefillVerifyArgmaxBufferPlanUsesPartialScratch() throws {
    let ir = qwen35MetalPrefillVerifyArgmax(batchSize: 64)
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let byIndex = Dictionary(uniqueKeysWithValues: plan.slots.map { ($0.index, $0) })

    // partialKeys: B * ceil(vocab / ROWS_PER_TG) * sizeof(uint2)
    #expect(byIndex[16]?.sizeBytes == 64 * ((248_320 + 7) / 8) * 8)
    #expect(byIndex[17]?.sizeBytes == 64 * 4)
}

@Test func prefillVerifyArgmaxBufferPlanUsesModuleTransactionCapacity() throws {
    let ir = qwen35MetalPrefillVerifyArgmax(
        batchSize: 64,
        verifyTokenCapacity: 32
    )
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let convHistory = plan.slots.filter {
        $0.name.hasPrefix("convStateHistory_")
    }
    let recHistory = plan.slots.filter {
        $0.name.hasPrefix("recStateHistory_")
    }

    #expect(convHistory.count == ir.numDeltaLayers)
    #expect(recHistory.count == ir.numDeltaLayers)
    #expect(convHistory.allSatisfy { $0.shape.first == 33 })
    #expect(recHistory.allSatisfy { $0.shape.first == 33 })
}

@Test func rejectsInvalidVerifyArgmaxTransactionCapacity() throws {
    let zero = qwen35MetalPrefillVerifyArgmax(
        batchSize: 64,
        verifyTokenCapacity: 0
    )
    #expect(throws: SmeltIRValidationError.self) {
        try validateSmeltIR(zero)
    }

    let oversized = qwen35MetalPrefillVerifyArgmax(
        batchSize: 64,
        verifyTokenCapacity: 65
    )
    #expect(throws: SmeltIRValidationError.self) {
        try validateSmeltIR(oversized)
    }
}

@Test func signedPrefillVerifyArgmaxBufferPlanReservesMaterializedRows() throws {
    let base = qwen35MetalPrefillVerifyArgmax(batchSize: 4)
    let ir = SmeltModelIR(
        modelName: base.modelName,
        config: base.config,
        layerPattern: base.layerPattern,
        quantization: SmeltQuantizationConfig(
            strategy: .binary1,
            groupSize: 128,
            excludePatterns: []
        ),
        loading: base.loading,
        prefill: base.prefill
    )
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let byIndex = Dictionary(uniqueKeysWithValues: plan.slots.map { ($0.index, $0) })

    #expect(byIndex[16]?.sizeBytes == 4 * 248_320 * 2)
    #expect(byIndex[17]?.sizeBytes == 4 * 4)
}

@Test func prefillBufferPlanHasTokenIdsBatchSlot() throws {
    let ir = qwen35MetalPrefill(batchSize: 64)
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)

    #expect(plan.tokenIdsBatchSlot == 26)
    let byIndex = Dictionary(uniqueKeysWithValues: plan.slots.map { ($0.index, $0) })
    // 64 * 4 bytes = 256, aligned to min 16
    #expect(byIndex[26]?.sizeBytes == 256)
    #expect(byIndex[26]?.name == "tokenIdsBatch")
}

@Test func decodeBufferPlanNoTokenIdsBatchSlot() throws {
    let plan = buildBufferPlan(from: SmeltModelIR.qwen35_2B)
    #expect(plan.tokenIdsBatchSlot == -1)
    let byIndex = Dictionary(uniqueKeysWithValues: plan.slots.map { ($0.index, $0) })
    // Slot 26 should not exist for decode-only
    #expect(byIndex[26] == nil)
}

// MARK: - backbone_hidden_size validation

@Test func backboneHiddenSizeZeroIsRejected() throws {
    // backbone_hidden_size must be positive when set; 0 silently
    // skipped at packer time would let a malformed backbone spec
    // compile with required projections silently omitted.
    let ir = SmeltModelIR(
        modelName: "test/model",
        config: SmeltConfig(
            hiddenSize: 256, numLayers: 4, vocabSize: 262_144,
            staticSeqCapacity: 256, ropeDim: 256, rmsEps: 1e-6,
            attention: SmeltAttentionConfig(
                qHeads: 4, kvHeads: 1, headDim: 256, gatedQ: false, externalKV: true),
            ffn: SmeltFFNConfig(dim: 2048, activation: .swiglu),
            backboneHiddenSize: 0),
        layerPattern: SmeltLayerPattern(unit: [.attention], repeats: 4),
        quantization: SmeltQuantizationConfig(strategy: .lutU4, groupSize: 16, excludePatterns: []),
        loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic))
    #expect(throws: SmeltIRValidationError.self) {
        try validateSmeltIR(ir)
    }
}

@Test func backboneHiddenSizeAbsentPassesValidation() throws {
    // Regression guard: existing target specs (no backbone_hidden_size)
    // must continue to pass IR validation.
    try validateSmeltIR(SmeltModelIR.qwen35_2B)
}

// MARK: - cluster_embedder validation

private func clusterEmbedderIR(
    numCentroids: Int, topK: Int, vocabSize: Int = 262_144
) -> SmeltModelIR {
    SmeltModelIR(
        modelName: "test/model",
        config: SmeltConfig(
            hiddenSize: 256, numLayers: 4, vocabSize: vocabSize,
            staticSeqCapacity: 256, ropeDim: 256, rmsEps: 1e-6,
            attention: SmeltAttentionConfig(
                qHeads: 4, kvHeads: 1, headDim: 256, gatedQ: false, externalKV: true),
            ffn: SmeltFFNConfig(dim: 2048, activation: .swiglu),
            backboneHiddenSize: 1536,
            clusterEmbedder: SmeltClusterEmbedderConfig(numCentroids: numCentroids, topK: topK)),
        layerPattern: SmeltLayerPattern(unit: [.attention], repeats: 4),
        quantization: SmeltQuantizationConfig(strategy: .lutU4, groupSize: 16, excludePatterns: []),
        loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic))
}

@Test func clusterEmbedderE2BPassesValidation() throws {
    let ir = clusterEmbedderIR(numCentroids: 2048, topK: 32)
    try validateSmeltIR(ir)
}

@Test func clusterEmbedderZeroCentroidsRejected() throws {
    let ir = clusterEmbedderIR(numCentroids: 0, topK: 32)
    #expect(throws: SmeltIRValidationError.self) { try validateSmeltIR(ir) }
}

@Test func clusterEmbedderTopKExceedingCentroidsRejected() throws {
    let ir = clusterEmbedderIR(numCentroids: 16, topK: 32)
    #expect(throws: SmeltIRValidationError.self) { try validateSmeltIR(ir) }
}

@Test func clusterEmbedderNonDivisibleVocabRejected() throws {
    let ir = clusterEmbedderIR(numCentroids: 1000, topK: 32, vocabSize: 262_144)
    #expect(throws: SmeltIRValidationError.self) { try validateSmeltIR(ir) }
}

@Test func clusterEmbedderTokensPerClusterTooLargeRejected() throws {
    // The cluster_sparse_lm_head v1 kernel dispatch caps tgWidth at
    // min(tokens_per_cluster, 256). For tokens_per_cluster > 256,
    // threads beyond the cap never run and their vocab slots leak
    // uninitialized memory. Validator must reject. With vocab=8192
    // and num_centroids=16, tokens_per_cluster = 512 → reject.
    let ir = clusterEmbedderIR(numCentroids: 16, topK: 4, vocabSize: 8192)
    #expect(throws: SmeltIRValidationError.self) { try validateSmeltIR(ir) }
}

@Test func clusterEmbedderUntiedLMHeadRejected() throws {
    // Untied lm_head + cluster_embedder → lm_head_weight is
    // quantized via the spec's strategy (independent of
    // quantize_embedding). The v1 sparse lm_head kernel binds a
    // single weight buffer with no LUT / scales / biases slots;
    // packed u4/affine bytes would be read as fp16. Validator
    // forces tied_lm_head=true for cluster_embedder packages.
    let ir = SmeltModelIR(
        modelName: "test/model",
        config: SmeltConfig(
            hiddenSize: 256, numLayers: 4, vocabSize: 262_144,
            staticSeqCapacity: 256, ropeDim: 256, rmsEps: 1e-6,
            attention: SmeltAttentionConfig(
                qHeads: 4, kvHeads: 1, headDim: 256, gatedQ: false, externalKV: true),
            ffn: SmeltFFNConfig(dim: 2048, activation: .swiglu),
            tiedLMHead: false,
            backboneHiddenSize: 1536,
            clusterEmbedder: SmeltClusterEmbedderConfig(numCentroids: 2048, topK: 32)),
        layerPattern: SmeltLayerPattern(unit: [.attention], repeats: 4),
        quantization: SmeltQuantizationConfig(strategy: .lutU4, groupSize: 16, excludePatterns: []),
        loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic))
    #expect(throws: SmeltIRValidationError.self) { try validateSmeltIR(ir) }
}

@Test func clusterEmbedderQuantizedEmbeddingRejected() throws {
    // The v1 cluster_sparse_lm_head Metal kernel binds a single
    // lm_head weight buffer with no LUT / scales / biases slots.
    // Quantized embed_tokens (which the tied lm_head reads from)
    // would be silently misread. Validator must reject.
    let ir = SmeltModelIR(
        modelName: "test/model",
        config: SmeltConfig(
            hiddenSize: 256, numLayers: 4, vocabSize: 262_144,
            staticSeqCapacity: 256, ropeDim: 256, rmsEps: 1e-6,
            attention: SmeltAttentionConfig(
                qHeads: 4, kvHeads: 1, headDim: 256, gatedQ: false, externalKV: true),
            ffn: SmeltFFNConfig(dim: 2048, activation: .swiglu),
            backboneHiddenSize: 1536,
            clusterEmbedder: SmeltClusterEmbedderConfig(numCentroids: 2048, topK: 32)),
        layerPattern: SmeltLayerPattern(unit: [.attention], repeats: 4),
        quantization: SmeltQuantizationConfig(
            strategy: .affineU4, groupSize: 128,
            excludePatterns: ["*_norm_weight"], quantizeEmbedding: true),
        loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic))
    #expect(throws: SmeltIRValidationError.self) { try validateSmeltIR(ir) }
}
