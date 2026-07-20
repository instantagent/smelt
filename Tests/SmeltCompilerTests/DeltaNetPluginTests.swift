import Foundation
import Testing
@testable import SmeltCompiler

@Test func deltaNetPluginEmitsCorrectDispatchCount() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    var emitter = SmeltCodeEmitter(indent: 8)

    let lines = try DeltaNetPlugin.emitLayer(
        layerIndex: 0,
        deltaIndex: 0,
        config: ir.config,
        plan: plan,
        weightEntries: weightEntries,
        weightsSlot: SmeltFixedSlot.weights.rawValue,
        groupSize: ir.quantization.groupSize,
        emitter: &emitter
    )

    // Count dispatch calls (lines containing "dispatch")
    let dispatchLines = lines.filter {
        $0.contains("dispatchThread")
    }
    // QKV + Z + fused A+B + conv + Q/K scale + recurrence + gated norm + out_proj = 8
    #expect(dispatchLines.count == 8)
}

@Test func deltaNetPluginUsesCorrectSlots() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    var emitter = SmeltCodeEmitter(indent: 4)

    let lines = try DeltaNetPlugin.emitLayer(
        layerIndex: 0,
        deltaIndex: 0,
        config: ir.config,
        plan: plan,
        weightEntries: weightEntries,
        weightsSlot: SmeltFixedSlot.weights.rawValue,
        groupSize: ir.quantization.groupSize,
        emitter: &emitter
    )

    let allCode = lines.joined(separator: "\n")

    // Uses qkvBuf slot
    #expect(allCode.contains("b[\(SmeltFixedSlot.qkvBuf.rawValue)]"))
    // Uses zBuf slot
    #expect(allCode.contains("b[\(SmeltFixedSlot.zBuf.rawValue)]"))
    // Uses conv state at dynamic base slot
    #expect(allCode.contains("b[\(plan.convStateBaseSlot)]"))
    // Uses rec state at dynamic base slot
    #expect(allCode.contains("b[\(plan.recStateBaseSlot)]"))
    // Uses weight buffer
    #expect(allCode.contains("b[\(SmeltFixedSlot.weights.rawValue)]"))
    // Uses normOutBuf for output
    #expect(allCode.contains("b[\(SmeltFixedSlot.normOutBuf.rawValue)]"))
}

@Test func deltaNetPluginLayer5UsesCorrectStateSlots() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    var emitter = SmeltCodeEmitter(indent: 4)

    // Layer 4 is delta index 3 (layers 0,1,2 are delta, 3 is attn, 4 is delta index 3)
    let lines = try DeltaNetPlugin.emitLayer(
        layerIndex: 4,
        deltaIndex: 3,
        config: ir.config,
        plan: plan,
        weightEntries: weightEntries,
        weightsSlot: SmeltFixedSlot.weights.rawValue,
        groupSize: ir.quantization.groupSize,
        emitter: &emitter
    )

    let allCode = lines.joined(separator: "\n")

    // Conv state for delta index 3
    #expect(allCode.contains("b[\(plan.convStateBaseSlot + 3)]"))
    // Rec state for delta index 3
    #expect(allCode.contains("b[\(plan.recStateBaseSlot + 3)]"))
}

@Test func deltaNetPluginGeneratesValidSwift() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    var emitter = SmeltCodeEmitter(indent: 4)

    let lines = try DeltaNetPlugin.emitLayer(
        layerIndex: 0,
        deltaIndex: 0,
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
        #expect(!line.contains("dictionary"))
    }

    // All pipeline references are integer indices
    let pipelineLines = lines.filter { $0.contains("setComputePipelineState") }
    for line in pipelineLines {
        #expect(line.contains("p["))
    }
}

@Test func deltaNetPluginAllWeightOffsetsAreNonZero() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    var emitter = SmeltCodeEmitter(indent: 4)

    // Layer 1 — weights should have non-zero offsets (not all at beginning)
    let lines = try DeltaNetPlugin.emitLayer(
        layerIndex: 1,
        deltaIndex: 1,
        config: ir.config,
        plan: plan,
        weightEntries: weightEntries,
        weightsSlot: SmeltFixedSlot.weights.rawValue,
        groupSize: ir.quantization.groupSize,
        emitter: &emitter
    )

    // Weight buffer references should include non-zero offsets
    let weightLines = lines.filter {
        $0.contains("b[\(SmeltFixedSlot.weights.rawValue)]") && $0.contains("offset:")
    }
    let hasNonZeroOffset = weightLines.contains { !$0.contains("offset: 0") }
    #expect(hasNonZeroOffset, "Layer 1 weights should have non-zero offsets")
}

@Test func deltaNetPluginD128H16UsesShapeSpecializedRecurrence() throws {
    let ir = SmeltModelIR.qwen35_0_8B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    var emitter = SmeltCodeEmitter(indent: 4)

    let lines = try DeltaNetPlugin.emitLayer(
        layerIndex: 0,
        deltaIndex: 0,
        config: ir.config,
        plan: plan,
        weightEntries: weightEntries,
        weightsSlot: SmeltFixedSlot.weights.rawValue,
        groupSize: ir.quantization.groupSize,
        emitter: &emitter
    )

    let allCode = lines.joined(separator: "\n")
    #expect(allCode.contains("p[\(SmeltPipeline.conv1dUpdateSilu.rawValue)]"))
    #expect(allCode.contains("p[\(SmeltPipeline.rmsScaleQK.rawValue)]"))
    #expect(allCode.contains("p[\(SmeltPipeline.deltanetRecurrenceMlxDecodeD128H16.rawValue)]"))
    #expect(!allCode.contains("p[\(SmeltPipeline.deltanetRecurrenceMlxDecode.rawValue)]"))
}

@Test func deltaNetPluginD128H32QK16UsesShapeSpecializedRecurrence() throws {
    let ir = SmeltModelIR.qwen35_4B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let weightEntries = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    var emitter = SmeltCodeEmitter(indent: 4)

    let lines = try DeltaNetPlugin.emitLayer(
        layerIndex: 0,
        deltaIndex: 0,
        config: ir.config,
        plan: plan,
        weightEntries: weightEntries,
        weightsSlot: SmeltFixedSlot.weights.rawValue,
        groupSize: ir.quantization.groupSize,
        emitter: &emitter
    )

    let allCode = lines.joined(separator: "\n")
    #expect(allCode.contains("p[\(SmeltPipeline.conv1dUpdateSilu.rawValue)]"))
    #expect(allCode.contains("p[\(SmeltPipeline.rmsScaleQK.rawValue)]"))
    #expect(allCode.contains("p[\(SmeltPipeline.deltanetRecurrenceMlxDecodeD128H32QK16.rawValue)]"))
    #expect(!allCode.contains("p[\(SmeltPipeline.conv1dUpdateSilu6144x4.rawValue)]"))
    #expect(!allCode.contains("p[\(SmeltPipeline.deltanetRecurrenceMlxDecode.rawValue)]"))
    #expect(!allCode.contains("p[\(SmeltPipeline.deltanetRecurrenceMlxDecodeD128H16.rawValue)]"))
}
