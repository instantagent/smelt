import Foundation
import Testing
@testable import SmeltCompiler
@testable import SmeltSchema

private func constantRecords(_ record: SmeltDispatchRecord) -> [SmeltConstantRecord] {
    let constants = [
        record.con0, record.con1, record.con2, record.con3,
        record.con4, record.con5, record.con6, record.con7,
    ]
    return Array(constants.prefix(Int(record.constantCount)))
}

private func d256H24KV4HybridIR() -> SmeltModelIR {
    SmeltModelIR(
        modelName: "test/D256-H24-KV4-hybrid",
        config: SmeltConfig(
            hiddenSize: 5_120,
            numLayers: 4,
            vocabSize: 248_320,
            ropeDim: 64,
            rmsEps: 1e-6,
            normMode: .onePlusWeight,
            delta: SmeltDeltaConfig(
                numHeads: 48,
                headDim: 128,
                convKernel: 4,
                qkvDim: 10_240,
                zDim: 6_144,
                aDim: 48,
                bDim: 48
            ),
            attention: SmeltAttentionConfig(
                qHeads: 24,
                kvHeads: 4,
                headDim: 256,
                gatedQ: true,
                qkNorm: true,
                ropeTheta: 10_000_000
            ),
            ffn: SmeltFFNConfig(dim: 17_408, activation: .swiglu)
        ),
        layerPattern: SmeltLayerPattern(
            unit: [.delta, .delta, .delta, .attention],
            repeats: 1
        ),
        quantization: SmeltModelIR.qwen35_4B.quantization,
        loading: SmeltModelIR.qwen35_4B.loading,
        prefill: SmeltModelIR.qwen35_4B.prefill
    )
}

@Test func attentionPluginEmitsCorrectDispatchCount() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    var emitter = SmeltCodeEmitter(indent: 8)

    // Layer 3 is first attention layer (attn index 0)
    let lines = try AttentionPlugin.emitLayer(
        layerIndex: 3,
        attnIndex: 0,
        config: ir.config,
        plan: plan,
        weightEntries: weightEntries,
        weightsSlot: SmeltFixedSlot.weights.rawValue,
        groupSize: ir.quantization.groupSize,
        emitter: &emitter
    )

    let dispatchLines = lines.filter { $0.contains("dispatchThread") }
    // Q/K/V projections(3) + gate_split(1) + 2x per-head norm(2) + 2x RoPE(2)
    // + 2x KV cache(2) + direct/one-pass/two-pass attention(4)
    // + sigmoid_mul(1) + O proj(1) = 16
    #expect(dispatchLines.count == 16)
}

@Test func attentionPluginUsesCorrectSlots() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    var emitter = SmeltCodeEmitter(indent: 4)

    let lines = try AttentionPlugin.emitLayer(
        layerIndex: 3,
        attnIndex: 0,
        config: ir.config,
        plan: plan,
        weightEntries: weightEntries,
        weightsSlot: SmeltFixedSlot.weights.rawValue,
        groupSize: ir.quantization.groupSize,
        emitter: &emitter
    )

    let allCode = lines.joined(separator: "\n")

    // Uses attention buffer slots
    #expect(allCode.contains("b[\(SmeltFixedSlot.attnQBuf.rawValue)]"))
    #expect(allCode.contains("b[\(SmeltFixedSlot.attnKBuf.rawValue)]"))
    #expect(allCode.contains("b[\(SmeltFixedSlot.attnVBuf.rawValue)]"))
    #expect(allCode.contains("b[\(SmeltFixedSlot.attnOutBuf.rawValue)]"))
    #expect(allCode.contains("b[\(SmeltFixedSlot.attnGateBuf.rawValue)]"))

    // Uses KV cache at dynamic base slot
    #expect(allCode.contains("b[\(plan.keyCacheBaseSlot)]"))
    #expect(allCode.contains("b[\(plan.valCacheBaseSlot)]"))

    // Uses RoPE tables
    #expect(allCode.contains("b[\(plan.ropeCosSlot)]"))
    #expect(allCode.contains("b[\(plan.ropeSinSlot)]"))
}

@Test func attentionPluginEmitsPositionDependentCode() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    var emitter = SmeltCodeEmitter(indent: 4)

    let lines = try AttentionPlugin.emitLayer(
        layerIndex: 3,
        attnIndex: 0,
        config: ir.config,
        plan: plan,
        weightEntries: weightEntries,
        weightsSlot: SmeltFixedSlot.weights.rawValue,
        groupSize: ir.quantization.groupSize,
        emitter: &emitter
    )

    let allCode = lines.joined(separator: "\n")

    // RoPE uses position-dependent offset
    #expect(allCode.contains("Int(position)"))
    // KV cache uses position
    #expect(allCode.contains("UInt32(position)"))
    // Attention decode uses seqLen = position + 1
    #expect(allCode.contains("UInt32(position + 1)"))
}

@Test func attentionPluginStrippedFusesRopeAndKCacheUpdate() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })

    for traceMode in [SmeltTraceMode.stripped, .strippedMarkers] {
        var emitter = SmeltCodeEmitter(indent: 4)
        _ = try AttentionPlugin.emitLayer(
            layerIndex: 3,
            attnIndex: 0,
            config: ir.config,
            plan: plan,
            weightEntries: weightEntries,
            weightsSlot: SmeltFixedSlot.weights.rawValue,
            groupSize: ir.quantization.groupSize,
            emitter: &emitter,
            traceMode: traceMode
        )

        let dispatches = emitter.dispatchRecords.filter {
            $0.opKind == SmeltDispatchRecord.opDispatch
        }
        #expect(
            dispatches.filter {
                $0.pipeline == UInt16(SmeltPipeline.ropeKVCacheUpdate.rawValue)
            }.count == 1
        )
        #expect(
            dispatches.filter {
                $0.pipeline == UInt16(SmeltPipeline.applyRope.rawValue)
            }.count == 1
        )
        #expect(
            dispatches.filter {
                $0.pipeline == UInt16(SmeltPipeline.kvCacheUpdate.rawValue)
            }.count == 1
        )
    }
}

@Test func attentionPluginGeneratesValidSwift() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    var emitter = SmeltCodeEmitter(indent: 4)

    let lines = try AttentionPlugin.emitLayer(
        layerIndex: 3,
        attnIndex: 0,
        config: ir.config,
        plan: plan,
        weightEntries: weightEntries,
        weightsSlot: SmeltFixedSlot.weights.rawValue,
        groupSize: ir.quantization.groupSize,
        emitter: &emitter
    )

    // No string lookups in generated code
    for line in lines {
        #expect(!line.contains("[\""))
    }

    // All pipeline references are integer indices
    let pipelineLines = lines.filter { $0.contains("setComputePipelineState") }
    for line in pipelineLines {
        #expect(line.contains("p["))
    }
}

@Test func attentionPluginUsesGenericMLXD256DecodeTopology() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    var emitter = SmeltCodeEmitter(indent: 4)

    let lines = try AttentionPlugin.emitLayer(
        layerIndex: 3,
        attnIndex: 0,
        config: ir.config,
        plan: plan,
        weightEntries: weightEntries,
        weightsSlot: SmeltFixedSlot.weights.rawValue,
        groupSize: ir.quantization.groupSize,
        emitter: &emitter
    )

    let allCode = lines.joined(separator: "\n")
    #expect(allCode.contains("p[\(SmeltPipeline.attentionDecodeD256H8KV2.rawValue)]"))
    #expect(allCode.contains("p[\(SmeltPipeline.attentionDecodeMLXVectorD256.rawValue)]"))
    #expect(allCode.contains("p[\(SmeltPipeline.attentionDecodeMLXVector2Pass1D256B128.rawValue)]"))
    #expect(allCode.contains("p[\(SmeltPipeline.attentionDecodeMLXVector2Pass2D256B128.rawValue)]"))
    #expect(!allCode.contains("p[\(SmeltPipeline.attentionDecode.rawValue)]"))
}

@Test func attentionPluginEncodesHybridDecodeGuardsInDispatchTable() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    var emitter = SmeltCodeEmitter(indent: 4)

    _ = try AttentionPlugin.emitLayer(
        layerIndex: 3,
        attnIndex: 0,
        config: ir.config,
        plan: plan,
        weightEntries: weightEntries,
        weightsSlot: SmeltFixedSlot.weights.rawValue,
        groupSize: ir.quantization.groupSize,
        emitter: &emitter
    )

    let shortRecord = emitter.dispatchRecords.first {
        $0.opKind == SmeltDispatchRecord.opDispatch
            && $0.pipeline == UInt16(SmeltPipeline.attentionDecodeD256H8KV2.rawValue)
    }
    let vectorRecord = emitter.dispatchRecords.first {
        $0.opKind == SmeltDispatchRecord.opDispatch
            && $0.pipeline == UInt16(SmeltPipeline.attentionDecodeMLXVectorD256.rawValue)
    }
    let twoPassRecord = emitter.dispatchRecords.first {
        $0.opKind == SmeltDispatchRecord.opDispatch
            && $0.pipeline == UInt16(
                SmeltPipeline.attentionDecodeMLXVector2Pass1D256B128.rawValue
            )
    }

    #expect(shortRecord != nil)
    #expect(vectorRecord != nil)
    #expect(twoPassRecord != nil)

    guard let shortRecord, let vectorRecord, let twoPassRecord else { return }

    let shortConstants = constantRecords(shortRecord)
    let vectorConstants = constantRecords(vectorRecord)
    let twoPassConstants = constantRecords(twoPassRecord)

    #expect(
        shortConstants.contains {
            $0.kind == SmeltConstantRecord.kindPositionPlus1LessThanLiteralSkipIfFalse
                && $0.value == 2
        }
    )
    #expect(
        vectorConstants.contains {
            $0.kind == SmeltConstantRecord.kindPositionPlus1GreaterEqualLiteralSkipIfFalse
                && $0.value == 2
        }
    )
    #expect(
        vectorConstants.contains {
            $0.kind == SmeltConstantRecord.kindPositionPlus1LessThanLiteralSkipIfFalse
                && $0.value == 1024
        }
    )
    #expect(
        twoPassConstants.contains {
            $0.kind == SmeltConstantRecord.kindPositionPlus1GreaterEqualLiteralSkipIfFalse
                && $0.value == 1024
        }
    )
    #expect(
        shortConstants.contains { $0.kind == SmeltConstantRecord.kindPositionPlus1 }
    )
    #expect(
        vectorConstants.contains { $0.kind == SmeltConstantRecord.kindPositionPlus1 }
    )
    #expect(
        shortConstants.contains { $0.kind == SmeltConstantRecord.kindCacheSeqCapacity }
    )
    #expect(
        vectorConstants.contains { $0.kind == SmeltConstantRecord.kindCacheSeqCapacity }
    )
}

@Test func attentionPluginD256H16KV4UsesGenericMLXDecodeTopology() throws {
    let ir = SmeltModelIR.qwen35_4B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    var emitter = SmeltCodeEmitter(indent: 4)

    let lines = try AttentionPlugin.emitLayer(
        layerIndex: 3,
        attnIndex: 0,
        config: ir.config,
        plan: plan,
        weightEntries: weightEntries,
        weightsSlot: SmeltFixedSlot.weights.rawValue,
        groupSize: ir.quantization.groupSize,
        emitter: &emitter
    )

    let allCode = lines.joined(separator: "\n")
    #expect(allCode.contains("p[\(SmeltPipeline.attentionDecodeD256H16KV4.rawValue)]"))
    #expect(allCode.contains("p[\(SmeltPipeline.attentionDecodeMLXVectorD256.rawValue)]"))
    #expect(allCode.contains("p[\(SmeltPipeline.attentionDecodeMLXVector2Pass1D256B128.rawValue)]"))
    #expect(allCode.contains("p[\(SmeltPipeline.attentionDecodeMLXVector2Pass2D256B128.rawValue)]"))
    #expect(!allCode.contains("p[\(SmeltPipeline.attentionDecode.rawValue)]"))
}

@Test func attentionPluginD256H16KV4EncodesMLXDecodeGuardsInDispatchTable() throws {
    let ir = SmeltModelIR.qwen35_4B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    var emitter = SmeltCodeEmitter(indent: 4)

    _ = try AttentionPlugin.emitLayer(
        layerIndex: 3,
        attnIndex: 0,
        config: ir.config,
        plan: plan,
        weightEntries: weightEntries,
        weightsSlot: SmeltFixedSlot.weights.rawValue,
        groupSize: ir.quantization.groupSize,
        emitter: &emitter
    )

    let shortRecord = emitter.dispatchRecords.first {
        $0.opKind == SmeltDispatchRecord.opDispatch
            && $0.pipeline == UInt16(SmeltPipeline.attentionDecodeD256H16KV4.rawValue)
    }
    let vectorRecord = emitter.dispatchRecords.first {
        $0.opKind == SmeltDispatchRecord.opDispatch
            && $0.pipeline == UInt16(SmeltPipeline.attentionDecodeMLXVectorD256.rawValue)
    }
    let twoPassRecord = emitter.dispatchRecords.first {
        $0.opKind == SmeltDispatchRecord.opDispatch
            && $0.pipeline == UInt16(
                SmeltPipeline.attentionDecodeMLXVector2Pass1D256B128.rawValue
            )
    }

    #expect(shortRecord != nil)
    #expect(vectorRecord != nil)
    #expect(twoPassRecord != nil)

    guard let shortRecord, let vectorRecord, let twoPassRecord else { return }

    let shortConstants = constantRecords(shortRecord)
    let vectorConstants = constantRecords(vectorRecord)
    let twoPassConstants = constantRecords(twoPassRecord)

    #expect(
        shortConstants.contains {
            $0.kind == SmeltConstantRecord.kindPositionPlus1LessThanLiteralSkipIfFalse
                && $0.value == 2
        }
    )
    #expect(
        vectorConstants.contains {
            $0.kind == SmeltConstantRecord.kindPositionPlus1GreaterEqualLiteralSkipIfFalse
                && $0.value == 2
        }
    )
    #expect(
        vectorConstants.contains {
            $0.kind == SmeltConstantRecord.kindPositionPlus1LessThanLiteralSkipIfFalse
                && $0.value == 1024
        }
    )
    #expect(
        twoPassConstants.contains {
            $0.kind == SmeltConstantRecord.kindPositionPlus1GreaterEqualLiteralSkipIfFalse
                && $0.value == 1024
        }
    )
}

@Test func attentionPluginD256H24KV4UsesShortAndLongContextBricks() throws {
    let ir = d256H24KV4HybridIR()
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    var emitter = SmeltCodeEmitter(indent: 4)

    let lines = try AttentionPlugin.emitLayer(
        layerIndex: 3,
        attnIndex: 0,
        config: ir.config,
        plan: plan,
        weightEntries: weightEntries,
        weightsSlot: SmeltFixedSlot.weights.rawValue,
        groupSize: ir.quantization.groupSize,
        emitter: &emitter
    )

    let allCode = lines.joined(separator: "\n")
    #expect(allCode.contains("p[\(SmeltPipeline.attentionDecodeD256H24KV4.rawValue)]"))
    #expect(allCode.contains("p[\(SmeltPipeline.attentionDecodeMLXVectorD256.rawValue)]"))
    #expect(allCode.contains("p[\(SmeltPipeline.attentionDecodeMLXVector2Pass1D256B128.rawValue)]"))
    #expect(allCode.contains("p[\(SmeltPipeline.attentionDecodeMLXVector2Pass2D256B128.rawValue)]"))
    #expect(!allCode.contains("p[\(SmeltPipeline.attentionDecode.rawValue)]"))

    let shortRecord = emitter.dispatchRecords.first {
        $0.opKind == SmeltDispatchRecord.opDispatch
            && $0.pipeline == UInt16(SmeltPipeline.attentionDecodeD256H24KV4.rawValue)
    }
    let vectorRecord = emitter.dispatchRecords.first {
        $0.opKind == SmeltDispatchRecord.opDispatch
            && $0.pipeline == UInt16(SmeltPipeline.attentionDecodeMLXVectorD256.rawValue)
    }
    let twoPassRecord = emitter.dispatchRecords.first {
        $0.opKind == SmeltDispatchRecord.opDispatch
            && $0.pipeline == UInt16(
                SmeltPipeline.attentionDecodeMLXVector2Pass1D256B128.rawValue
            )
    }
    let shortConstants = try #require(shortRecord).constantCount
    let vectorConstants = try #require(vectorRecord).constantCount
    let twoPassConstants = try #require(twoPassRecord).constantCount
    #expect(shortConstants == 3)
    #expect(vectorConstants == 6)
    #expect(twoPassConstants == 5)
    #expect(
        constantRecords(try #require(shortRecord)).contains {
            $0.kind == SmeltConstantRecord.kindPositionPlus1LessThanLiteralSkipIfFalse
                && $0.value == 2
        }
    )
    #expect(
        constantRecords(try #require(vectorRecord)).contains {
            $0.kind == SmeltConstantRecord.kindPositionPlus1GreaterEqualLiteralSkipIfFalse
                && $0.value == 2
        }
    )
    #expect(
        constantRecords(try #require(twoPassRecord)).contains {
            $0.kind == SmeltConstantRecord.kindPositionPlus1GreaterEqualLiteralSkipIfFalse
                && $0.value == 1024
        }
    )
}

@Test func attentionPluginRoutesPreviouslyUnlistedD256ShapeThroughMLXBricks() throws {
    let base = d256H24KV4HybridIR()
    let config = SmeltConfig(
        hiddenSize: base.config.hiddenSize,
        numLayers: base.config.numLayers,
        vocabSize: base.config.vocabSize,
        ropeDim: base.config.ropeDim,
        rmsEps: base.config.rmsEps,
        normMode: base.config.normMode,
        delta: base.config.delta,
        attention: SmeltAttentionConfig(
            qHeads: 12,
            kvHeads: 3,
            headDim: 256,
            gatedQ: true,
            qkNorm: true,
            ropeTheta: 10_000_000
        ),
        ffn: base.config.ffn
    )
    let ir = SmeltModelIR(
        modelName: "test/unlisted-D256-H12-KV3",
        config: config,
        layerPattern: base.layerPattern,
        quantization: base.quantization,
        loading: base.loading,
        prefill: base.prefill
    )
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    var emitter = SmeltCodeEmitter(indent: 4)

    _ = try AttentionPlugin.emitLayer(
        layerIndex: 3,
        attnIndex: 0,
        config: ir.config,
        plan: plan,
        weightEntries: weightEntries,
        weightsSlot: SmeltFixedSlot.weights.rawValue,
        groupSize: ir.quantization.groupSize,
        emitter: &emitter
    )

    let pipelines = Set(emitter.dispatchRecords.map(\.pipeline))
    #expect(pipelines.contains(UInt16(SmeltPipeline.attentionDecodeMLXVectorD256.rawValue)))
    #expect(pipelines.contains(UInt16(
        SmeltPipeline.attentionDecodeMLXVector2Pass1D256B128.rawValue
    )))
    #expect(pipelines.contains(UInt16(
        SmeltPipeline.attentionDecodeMLXVector2Pass2D256B128.rawValue
    )))
    #expect(!pipelines.contains(UInt16(SmeltPipeline.attentionDecode.rawValue)))
}

// MARK: - external_kv codegen smoke

private func parseExternalKVIR() throws -> SmeltModelIR {
    SmeltModelIR(
        modelName: "test/model",
        config: SmeltConfig(
            hiddenSize: 256, numLayers: 4, vocabSize: 262_144,
            staticSeqCapacity: 256, ropeDim: 256, rmsEps: 1e-6,
            attention: SmeltAttentionConfig(
                qHeads: 4, kvHeads: 1, headDim: 256, gatedQ: false,
                qkNorm: true, externalKV: true),
            ffn: SmeltFFNConfig(dim: 2048, activation: .swiglu)),
        layerPattern: SmeltLayerPattern(unit: [.attention], repeats: 4),
        quantization: SmeltQuantizationConfig(strategy: .lutU4, groupSize: 16, excludePatterns: []),
        loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic))
}

@Test func attentionPluginExternalKVOmitsKAndVProjections() throws {
    // Smoke regression for the codex P1 caught at unit 2b1 codex
    // review: an externalKV layer must not reference k_proj_weight or
    // v_proj_weight in generated code, since those tensors aren't
    // packed into the model's safetensors at all.
    let ir = try parseExternalKVIR()
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    var emitter = SmeltCodeEmitter(indent: 4)

    let lines = try AttentionPlugin.emitLayer(
        layerIndex: 0,
        attnIndex: 0,
        config: ir.config,
        plan: plan,
        weightEntries: weightEntries,
        weightsSlot: SmeltFixedSlot.weights.rawValue,
        groupSize: ir.quantization.groupSize,
        emitter: &emitter
    )
    let allCode = lines.joined(separator: "\n")

    // The two externalKV branches in the K/V projection and KV-cache
    // sections each emit a deterministic comment marker.
    #expect(allCode.contains("External KV: K/V supplied by target"))
    #expect(allCode.contains("External KV: K/V cache pre-populated"))

    // K and V projection comments come from the matvec emit calls
    // that we now skip on externalKV layers; they must not appear.
    #expect(!allCode.contains("// K projection"))
    #expect(!allCode.contains("// V projection"))
    #expect(!allCode.contains("K per-head RMS norm"))

    // K-side RoPE and KV-cache writes are also gated on externalKV.
    // Target package owns the cache; the model must not write to it.
    #expect(!allCode.contains("// RoPE on K"))
    #expect(!allCode.contains("// K cache update"))
    #expect(!allCode.contains("// V cache update"))

    // Q-side computations still run because the model computes Q
    // from the layer's hidden state; their comments must remain.
    #expect(allCode.contains("Q projection"))
    #expect(allCode.contains("Q per-head RMS norm"))
    #expect(allCode.contains("RoPE on Q"))
}
