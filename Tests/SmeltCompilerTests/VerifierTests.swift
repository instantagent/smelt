import Testing
@testable import SmeltCompiler

@Test func verifyQwen35GeneratedCode() throws {
    let ir = SmeltModelIR.qwen35_2B
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try TopLevelEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )

    let report = SmeltVerifier.verify(source: result.source, expectedLayers: 24)

    // Structural checks
    #expect(report.validFunctionStructure)
    #expect(!report.hasStringLookups)
    #expect(report.allIntegerPipelines)

    // Swap count: 2 per layer × 24 layers = 48
    #expect(report.correctSwapCount == 48)
    #expect(report.expectedSwapCount == 48)

    // Overall pass
    #expect(report.passed)
}

@Test func verifyDispatchBudget() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try TopLevelEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )

    let report = SmeltVerifier.verify(source: result.source, expectedLayers: 24)

    let expected = SmeltDispatchBudget.expectedTotal(
        numDelta: 18, numAttn: 6, numLayers: 24
    )

    // Budget check against optimized dispatch records (the runtime artifact)
    let optimizedDispatches = result.dispatchRecords.filter {
        $0.opKind == SmeltDispatchRecord.opDispatch
    }.count
    #expect(
        optimizedDispatches == expected,
        "Dispatch budget: expected \(expected), got \(optimizedDispatches)"
    )

    // Pipeline state changes must equal dispatch count in source (1:1 invariant)
    #expect(report.dispatchPipelineParity)
}

@Test func verifyNoPipelineStringLookups() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try TopLevelEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )
    let source = result.source
    let report = SmeltVerifier.verify(source: source, expectedLayers: 24)

    let pipelineLines = source.components(separatedBy: "\n")
        .filter { $0.contains("setComputePipelineState") }

    // Every pipeline set must use p[N] syntax
    for line in pipelineLines {
        #expect(line.contains("p["), "Non-integer pipeline: \(line)")
    }

    // Pipeline count must equal source dispatch count.
    #expect(pipelineLines.count == report.pipelineStateChanges)
    #expect(report.dispatchPipelineParity)
}

@Test func verifyBoundsChecksPresent() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)

    let result = try TopLevelEmitter.generate(
        ir: ir, plan: plan, weightLayout: layout
    )
    let source = result.source

    // Must have position bounds check
    #expect(source.contains("precondition(position >= 0"))
    // Must have tokenId bounds check
    #expect(source.contains("precondition(tokenId >= 0"))
}

@Test func verifyDispatchBudgetMatchesEmittedCode() throws {
    // Verify budget constants match ACTUAL emitted dispatch count
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let genResult = try TopLevelEmitter.generate(ir: ir, plan: plan, weightLayout: layout)
    let budgetTotal = SmeltDispatchBudget.expectedTotal(
        numDelta: 18, numAttn: 6, numLayers: 24
    )

    // Budget must match optimized dispatch records (the runtime artifact)
    let optimizedDispatches = genResult.dispatchRecords.filter {
        $0.opKind == SmeltDispatchRecord.opDispatch
    }.count
    #expect(
        optimizedDispatches == budgetTotal,
        "Budget says \(budgetTotal), optimized records have \(optimizedDispatches)"
    )
}

@Test func verifyHandoffResolverIntegration() throws {
    let ir = SmeltModelIR.qwen35_2B
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)

    guard let prefill = ir.prefill else {
        #expect(Bool(false), "Qwen 3.5 2B should have prefill config")
        return
    }

    let table = SmeltHandoffResolver.resolve(
        families: prefill.handoffFamilies,
        ir: ir,
        plan: plan
    )

    // All slot indices should be within buffer plan range
    for entry in table.entries {
        #expect(
            entry.slotIndex >= 0 && entry.slotIndex < plan.slotCount,
            "Slot \(entry.slotIndex) for \(entry.tensorName) out of range"
        )
    }

    // RoPE slots should be valid
    #expect(table.ropeCosSlot >= 0 && table.ropeCosSlot < plan.slotCount)
    #expect(table.ropeSinSlot >= 0 && table.ropeSinSlot < plan.slotCount)
}
