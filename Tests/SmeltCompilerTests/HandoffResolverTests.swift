import Foundation
import Testing
@testable import SmeltCompiler

@Test func resolveQwen35HandoffFamilies() throws {
    let ir = SmeltModelIR.qwen35_2B
    try validateSmeltIR(ir)
    let plan = buildBufferPlan(from: ir)

    let table = SmeltHandoffResolver.resolve(
        families: ir.prefill!.handoffFamilies,
        ir: ir,
        plan: plan
    )

    // 18 conv + 18 rec + 6 key + 6 val = 48 entries
    #expect(table.entries.count == 48)

    // Conv states start at plan.convStateBaseSlot
    let convEntries = table.entries.filter { $0.tensorName.hasPrefix("conv_state_") }
    #expect(convEntries.count == 18)
    #expect(convEntries[0].slotIndex == plan.convStateBaseSlot)
    #expect(convEntries[17].slotIndex == plan.convStateBaseSlot + 17)
    #expect(convEntries[0].convertFP16toFP32 == false)

    // Rec states are now stored in FP16 end-to-end.
    let recEntries = table.entries.filter { $0.tensorName.hasPrefix("rec_state_") }
    #expect(recEntries.count == 18)
    #expect(recEntries[0].slotIndex == plan.recStateBaseSlot)
    #expect(recEntries[0].convertFP16toFP32 == false)

    // Rec state element count: numHeads * headDim * headDim = 16 * 128 * 128 = 262144
    #expect(recEntries[0].expectedElements == 262_144)

    // Key caches
    let keyEntries = table.entries.filter { $0.tensorName.hasPrefix("key_cache_") }
    #expect(keyEntries.count == 6)
    #expect(keyEntries[0].slotIndex == plan.keyCacheBaseSlot)

    // Value caches
    let valEntries = table.entries.filter { $0.tensorName.hasPrefix("value_cache_") }
    #expect(valEntries.count == 6)
    #expect(valEntries[0].slotIndex == plan.valCacheBaseSlot)

    // RoPE slots
    #expect(table.ropeCosSlot == plan.ropeCosSlot)
    #expect(table.ropeSinSlot == plan.ropeSinSlot)
}

@Test func resolveWithoutRopeFamilyOmitsRopeEntries() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)

    // Only conv_state — no rope family
    let table = SmeltHandoffResolver.resolve(
        families: ["conv_state"],
        ir: ir,
        plan: plan
    )

    #expect(table.entries.count == 18)
    // RoPE slots still provided for the runtime to use
    #expect(table.ropeCosSlot == plan.ropeCosSlot)
}

@Test func resolvedHandoffJSONRoundTrip() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)

    let table = SmeltHandoffResolver.resolve(
        families: ir.prefill!.handoffFamilies,
        ir: ir,
        plan: plan
    )

    let encoded = try JSONEncoder().encode(table)
    let decoded = try JSONDecoder().decode(SmeltHandoffTable.self, from: encoded)

    #expect(decoded.entries.count == table.entries.count)
    #expect(decoded.entries[0].tensorName == table.entries[0].tensorName)
    #expect(decoded.entries[0].slotIndex == table.entries[0].slotIndex)
    #expect(decoded.ropeCosSlot == table.ropeCosSlot)
}

@Test func tensorNamesMatchCoreMLOutputConvention() throws {
    let ir = SmeltModelIR.qwen35_2B
    let plan = buildBufferPlan(from: ir)

    let table = SmeltHandoffResolver.resolve(
        families: ir.prefill!.handoffFamilies,
        ir: ir,
        plan: plan
    )

    // CoreML outputs use "_out" suffix
    for entry in table.entries {
        #expect(entry.tensorName.hasSuffix("_out"),
                "Tensor name '\(entry.tensorName)' should end with _out")
    }
}
