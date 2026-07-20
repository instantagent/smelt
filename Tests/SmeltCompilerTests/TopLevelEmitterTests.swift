import Foundation
import Testing
@testable import SmeltCompiler
import SmeltSchema

private let topLevelEmitterTestTempRoot: URL = makeManagedTempRoot(
    "smelt-top-level-emitter-tests"
)

private func makeTempDir() throws -> URL {
    try makeTempDir(under: topLevelEmitterTestTempRoot)
}

private func writeLocalWeightsBundle(
    layout: [SmeltWeightEntry]
) throws -> URL {
    try writeLocalWeightsBundle(layout: layout, into: topLevelEmitterTestTempRoot)
}

@Test func topLevelEmitterGeneratesCompleteFunctionForQwen() throws {
    let ir = SmeltModelIR.qwen35_2B
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try TopLevelEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )
    let source = result.source

    // Should contain function signature
    #expect(source.contains("func encodeDecodeStep("))
    #expect(source.contains("_ enc: MTLComputeCommandEncoder"))
    #expect(source.contains("UnsafeBufferPointer<MTLComputePipelineState>"))
    #expect(source.contains("UnsafeBufferPointer<MTLBuffer>"))

    // Should contain bounds check
    #expect(source.contains("precondition(position >= 0"))

    // Should contain embedding
    #expect(source.contains("Embedding"))

    // Should contain double-buffer tracking
    #expect(source.contains("var cur ="))
    #expect(source.contains("var alt ="))
    #expect(source.contains("swap(&cur, &alt)"))

    // Should contain all 24 layers
    #expect(source.contains("Layer 0 (delta)"))
    #expect(source.contains("Layer 3 (attn)"))
    #expect(source.contains("Layer 23 (attn)"))

    // Should contain final norm + LM head + argmax
    #expect(source.contains("Final norm"))
    #expect(source.contains("LM head"))
    #expect(source.contains("Argmax"))

    // Should close the function
    #expect(source.hasSuffix("}"))
}

@Test func topLevelEmitterDispatchBudget() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try TopLevelEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )
    let source = result.source

    // Count total dispatches
    let dispatchCount = source.components(separatedBy: "dispatchThread").count - 1

    // Generated source dispatches:
    // DeltaNet plugin: 8 × 18 = 144
    // Attention plugin: 16 × 6 = 96 (four guarded D256 attention bricks)
    // Shared per layer: input_norm + residual_add + post_norm + FFN gate/up + FFN down + residual_add = 6 × 24 = 144
    // Global: embedding + final_norm + lm_head + split argmax partials + split argmax reduce = 5
    // Total: 144 + 96 + 144 + 5 = 389
    #expect(dispatchCount == 389, "Expected 389 source dispatches, got \(dispatchCount)")
}

@Test func topLevelEmitterNoStringLookups() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try TopLevelEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )
    let source = result.source

    // No dictionary access in generated code (except comments)
    let codeLines = source.components(separatedBy: "\n")
        .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }

    for line in codeLines {
        #expect(!line.contains("[\""), "String lookup found: \(line)")
    }
}

@Test func topLevelEmitterUsesIntegerPipelineIndices() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try TopLevelEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )
    let source = result.source

    let pipelineLines = source.components(separatedBy: "\n")
        .filter { $0.contains("setComputePipelineState") }

    #expect(pipelineLines.count == 389, "Expected 389 pipeline sets")
    for line in pipelineLines {
        #expect(line.contains("p["), "Non-integer pipeline: \(line)")
    }
}

// MARK: - Pre/post projection emission

@Test func topLevelEmitterEmitsProjectionsForBackboneModel() throws {
    let ir = try backboneModelIR()
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let result = try TopLevelEmitter.generate(ir: ir, plan: plan, weightLayout: layout)

    // Pre-projection runs before the layer loop's double-buffer
    // tracking, replacing the embedding-gather as the layer-0 input
    // path. Post-projection runs after the final norm, in parallel
    // with the lm_head dispatch.
    #expect(result.source.contains(
        "Pre-projection (drafter input → drafter hidden)"
    ))
    #expect(result.source.contains(
        "Post-projection (drafter hidden → backbone output)"
    ))

    // Trace markers for both projections fire so a future runtime
    // can pin them by label.
    let labels = result.traceMarkers.map(\.label)
    #expect(labels.contains("pre_projection_out"))
    #expect(labels.contains("post_projection_out"))
}

@Test func topLevelEmitterSkipsProjectionsWithoutBackbone() throws {
    // Regression: Qwen 3.5 2B has no backbone_hidden_size, so the
    // pre/post projection emission paths must stay dormant.
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let result = try TopLevelEmitter.generate(ir: ir, plan: plan, weightLayout: layout)

    #expect(!result.source.contains("Pre-projection"))
    #expect(!result.source.contains("Post-projection"))
}

@Test func bufferPlanWidensHiddenSlotsForBackboneModel() throws {
    // Backbone model: hidden_size=256, backbone=1536. pre_projection
    // reads 2*backbone=3072 fp16 elements from hiddenB; post_projection
    // writes backbone=1536 elements to hiddenB. Widening means hiddenB
    // must be at least 3072 elements (pre-projection input dominates),
    // hiddenA at least max(hiddenSize, backbone) = 1536 elements.
    let ir = try backboneModelIR()
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let fp16 = 2

    let hiddenA = try #require(plan.slots.first { $0.name == "hiddenA" })
    let hiddenB = try #require(plan.slots.first { $0.name == "hiddenB" })

    #expect(hiddenA.sizeBytes >= 1536 * fp16)
    #expect(hiddenB.sizeBytes >= 2 * 1536 * fp16)
}

@Test func bufferPlanLeavesHiddenSlotsWithoutBackbone() throws {
    // Regression: Qwen 3.5 2B (no backbone_hidden_size, no metal
    // prefill in the reference IR → B=1). hiddenA/hiddenB stay at
    // hidden_size fp16 elements, no widening.
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let fp16 = 2
    let hidden = ir.config.hiddenSize

    let hiddenA = try #require(plan.slots.first { $0.name == "hiddenA" })
    let hiddenB = try #require(plan.slots.first { $0.name == "hiddenB" })

    #expect(hiddenA.sizeBytes == hidden * fp16)
    #expect(hiddenB.sizeBytes == hidden * fp16)
}

// MARK: - Cluster sparse LM head

/// A 4-layer external-KV attention model with a backbone projection (hidden 256,
/// backbone 1536). Shared base for the projection + cluster codegen tests.
private func backboneModelIR() throws -> SmeltModelIR {
    SmeltModelIR(
        modelName: "test/model",
        config: SmeltConfig(
            hiddenSize: 256, numLayers: 4, vocabSize: 262_144,
            staticSeqCapacity: 256, ropeDim: 256, rmsEps: 1e-6,
            attention: SmeltAttentionConfig(
                qHeads: 4, kvHeads: 1, headDim: 256, gatedQ: false, externalKV: true),
            ffn: SmeltFFNConfig(dim: 2048, activation: .swiglu),
            backboneHiddenSize: 1536),
        layerPattern: SmeltLayerPattern(unit: [.attention], repeats: 4),
        quantization: SmeltQuantizationConfig(strategy: .lutU4, groupSize: 16, excludePatterns: []),
        loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic))
}

private func parseIRWithCluster() throws -> SmeltModelIR {
    SmeltModelIR(
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
        quantization: SmeltQuantizationConfig(strategy: .lutU4, groupSize: 16, excludePatterns: []),
        loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic))
}

private func clusterWithLogitCapModelIR() throws -> SmeltModelIR {
    SmeltModelIR(
        modelName: "test/model",
        config: SmeltConfig(
            hiddenSize: 256, numLayers: 4, vocabSize: 262_144,
            staticSeqCapacity: 256, ropeDim: 256, rmsEps: 1e-6,
            logitCap: 30,
            attention: SmeltAttentionConfig(
                qHeads: 4, kvHeads: 1, headDim: 256, gatedQ: false, externalKV: true),
            ffn: SmeltFFNConfig(dim: 2048, activation: .swiglu),
            backboneHiddenSize: 1536,
            clusterEmbedder: SmeltClusterEmbedderConfig(numCentroids: 2048, topK: 32)),
        layerPattern: SmeltLayerPattern(unit: [.attention], repeats: 4),
        quantization: SmeltQuantizationConfig(strategy: .lutU4, groupSize: 16, excludePatterns: []),
        loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic))
}

@Test func topLevelEmitterEmitsClusterSparseLMHead() throws {
    // Spec with cluster_embedder set: dense lm_head matvec
    // must be skipped, cluster sparse lm_head dispatches must fire.
    let ir = try parseIRWithCluster()
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let result = try TopLevelEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )

    // Comment markers fire. The "Centroid projection" matvec produces
    // [num_centroids] logits and the sparse-lm-head dispatch gathers
    // top-k clusters into the vocab-sized logits buffer.
    #expect(result.source.contains("Final norm (cluster lm_head path)"))
    #expect(result.source.contains("Centroid projection (cluster lm_head)"))
    #expect(result.source.contains("Cluster sparse LM head"))

    // The dispatch table references the new pipeline.
    let pipelines = result.pipelineNames
    let clusterIdx = pipelines.firstIndex { $0.contains("cluster_sparse_lm_head") }
    #expect(clusterIdx != nil, "cluster_sparse_lm_head pipeline should be registered")
    if let clusterIdx {
        let dispatched = result.dispatchRecords.contains {
            Int($0.pipeline) == clusterIdx
        }
        #expect(dispatched, "cluster_sparse_lm_head should be dispatched")
    }

    // Trace markers for centroid_logits and cluster_lm_head_logits.
    let labels = result.traceMarkers.map(\.label)
    #expect(labels.contains("centroid_logits"))
    #expect(labels.contains("cluster_lm_head_logits"))

    // The model still has projections from unit 2d2.
    #expect(result.source.contains("Pre-projection"))
    #expect(result.source.contains("Post-projection"))
}

@Test func topLevelEmitterUsesDenseLMHeadForNonClusterModel() throws {
    // Regression: Qwen 3.5 2B (no cluster_embedder) must keep using
    // the dense lm_head matvec. None of the cluster comment markers
    // or pipeline references should appear.
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let result = try TopLevelEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )

    #expect(!result.source.contains("Cluster sparse LM head"))
    #expect(!result.source.contains("Centroid projection"))
    let labels = result.traceMarkers.map(\.label)
    #expect(!labels.contains("centroid_logits"))
    #expect(!labels.contains("cluster_lm_head_logits"))

    // The dense LM head pipeline should still fire — i.e. some
    // matvec dispatching to logitsBuf is present.
    #expect(result.source.contains("LM head"))
}

@Test func topLevelEmitterSkipsLogitCapForClusterPath() throws {
    // Without skipping, cap * tanh(-inf / cap) == -cap unmasks every
    // non-candidate -inf entry the cluster sparse lm_head wrote.
    // Confirm a spec with both cluster_embedder AND
    // logit_cap set produces NO "Logit capping" comment in the
    // generated source — softcap belongs inside the cluster kernel
    // (unit 2c3 implementation detail).
    let ir = try clusterWithLogitCapModelIR()
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let result = try TopLevelEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )

    #expect(result.source.contains("Cluster sparse LM head"))
    #expect(!result.source.contains("Logit capping"))

    // Regression guard: the dense-lm_head path still uses logit_cap
    // when cluster_embedder is unset. Verified separately by
    // existing fixture tests that exercise logit_cap.
}

@Test func bufferPlanAllocatesCentroidLogitsBufWhenClusterSet() throws {
    let ir = try parseIRWithCluster()
    let plan = buildBufferPlan(from: ir)
    let centroid = try #require(plan.slots.first { $0.name == "centroidLogitsBuf" })

    // num_centroids=2048 fp16 → 4 KiB
    #expect(centroid.sizeBytes == 2048 * 2)
    // Slot index matches the SmeltFixedSlot rawValue (31).
    #expect(centroid.index == SmeltFixedSlot.centroidLogitsBuf.rawValue)
}

@Test func bufferPlanOmitsCentroidLogitsBufForNonCluster() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    #expect(plan.slots.first { $0.name == "centroidLogitsBuf" } == nil)
}
