// MatvecKernelTableTests — the parity gate for the dtype-building-blocks gateway
// (docs/dtype-building-blocks-plan.md §0.5/§1, Unit 1).
//
// These are the standing invariant tests, not vacuous coverage:
//   - PARITY: registered ∪ knownMissing == meaningfulCells, registered ∩ knownMissing == ∅,
//     so the three sets can never silently drift (registered is derived, but we still assert
//     the partition holds against the declared sets).
//   - NON-VACUITY: the table's matrix IS the §0 kernel matrix, and the MISSING dense cells are
//     EXACTLY the fp16-act × {bf16,fp32} holes — not an empty/over-broad set that would make
//     the gate pass for free.
//   - GATEWAY BEHAVIOR: select() returns the right family for registered cells, throws .missing
//     for the holes, throws .notMeaningful for combos no consumer needs — no silent default.

import Foundation
import Testing

@testable import SmeltCompiler
@testable import SmeltSchema

private typealias Table = MatvecKernelTable
private typealias Cell = MatvecKernelTable.Cell

// MARK: - Parity (the sets partition cleanly)

@Test func parityKnownMissingIsSubsetOfMeaningful() {
    #expect(Table.knownMissing.isSubset(of: Table.meaningfulCells))
}

@Test func parityRegisteredAndKnownMissingPartitionMeaningful() {
    // registered := meaningfulCells \ knownMissing (derived); assert the partition explicitly.
    #expect(Table.registered.union(Table.knownMissing) == Table.meaningfulCells)
    #expect(Table.registered.isDisjoint(with: Table.knownMissing))
}

// MARK: - Non-vacuity (the matrix is the real §0 kernel matrix)

@Test func nonVacuityMatrixIsNonTrivial() {
    // Guard against the gate passing because the sets are empty/degenerate. knownMissing is EMPTY
    // as of U2 (every meaningful cell has a kernel — the no-hole claim is hard), so non-vacuity is
    // carried by meaningfulCells/registered being large, asserted here, and by the registered-cell
    // pins below (which prove the matrix didn't shrink to nothing).
    #expect(Table.meaningfulCells.count >= 18)
    #expect(!Table.registered.isEmpty)
    #expect(Table.knownMissing.isEmpty)
}

@Test func noHolesRemainEveryMeaningfulCellResolves() {
    // The hard no-hole claim (U2): knownMissing is empty AND every meaningful cell resolves to a
    // family — select() never throws for a declared cell. A future op that declares a meaningful
    // cell without a kernel (re-growing knownMissing) breaks this immediately.
    #expect(Table.knownMissing.isEmpty)
    for cell in Table.meaningfulCells {
        #expect((try? Table.select(cell)) != nil, "meaningful cell has no kernel: \(cell)")
    }
}

@Test func formerDenseHolesAreNowRegistered() {
    // Non-vacuity replacement for the old "missing holes are exactly {bf16,fp32}" pin: the U2 win
    // is that those EXACT cells are now REGISTERED and select to `.dense(w)` — proving the matrix
    // didn't simply DROP the cells (which would pass a weight-projected check while silently
    // shrinking coverage). gemv (fixed + variable slot) + gemm, for bf16 and fp32.
    for w in [SmeltDType.bf16, .fp32] {
        for slot in [Table.Slot.fixed, .variable] {
            let gemv = Cell(activation: .fp16, weight: w, slot: slot)
            #expect(Table.registered.contains(gemv), "expected registered: \(gemv)")
            #expect((try? Table.select(gemv)) == .dense(w))
        }
        let gemm = Cell(activation: .fp16, weight: w, shape: .gemm)
        #expect(Table.registered.contains(gemm), "expected registered: \(gemm)")
        #expect((try? Table.select(gemm)) == .dense(w))
    }
}

@Test func nonVacuityFP16ActDenseFP16AndQuantAreRegistered() {
    // fp16-act × {fp16 dense, affine_u4, u4_lut} have kernels today — they must be registered,
    // not missing, for both input-slot kinds (fixed/variable) where a consumer exists.
    for slot in [Table.Slot.fixed, .variable] {
        for w in [SmeltDType.fp16, .affineU4, .u4Lut] {
            let cell = Cell(activation: .fp16, weight: w, slot: slot)
            #expect(Table.registered.contains(cell), "expected registered: \(cell)")
        }
    }
}

@Test func nonVacuityFP32ActDenseMatrixIsRegistered() {
    // The TTS talker fp32-act dense matrix (f32/f16/bf16 × gemv/gemm/gemmTN) all have kernels.
    for shape in [Table.Shape.gemv, .gemm, .gemmTN] {
        for w in [SmeltDType.fp32, .fp16, .bf16] {
            let cell = Cell(activation: .fp32, weight: w, shape: shape, output: .fp32)
            #expect(Table.registered.contains(cell), "expected registered: \(cell)")
        }
    }
}

@Test func nonVacuityBF16ActDenseMatrixIsRegistered() {
    for shape in [Table.Shape.gemv, .gemm] {
        let cell = Cell(
            activation: .bf16,
            weight: .bf16,
            shape: shape,
            output: .bf16
        )
        #expect(Table.registered.contains(cell), "expected registered: \(cell)")
        #expect((try? Table.select(cell)) == .dense(.bf16))
    }
}

// MARK: - Gateway behavior (total, no silent default)

@Test func selectReturnsFamilyForRegisteredCells() throws {
    #expect(try Table.select(Cell(activation: .fp16, weight: .fp16)) == .dense(.fp16))
    #expect(try Table.select(Cell(activation: .fp16, weight: .affineU4)) == .affineU4)
    #expect(try Table.select(Cell(activation: .fp16, weight: .u4Lut)) == .lutU4)
    #expect(try Table.select(Cell(activation: .fp16, weight: .turboQuantH, slot: .fixed)) == .tqh)
    #expect(try Table.select(Cell(activation: .fp32, weight: .bf16, output: .fp32)) == .dense(.bf16))
    #expect(
        try Table.select(Cell(activation: .fp32, weight: .affineU4, shape: .gemm, output: .fp32))
            == .affineU4)
    // fp16-act GEMM (emitBatchedMatmul): fp16 dense + affine_u4/u4_lut/tqh are registered.
    #expect(try Table.select(Cell(activation: .fp16, weight: .fp16, shape: .gemm)) == .dense(.fp16))
    #expect(try Table.select(Cell(activation: .fp16, weight: .affineU4, shape: .gemm)) == .affineU4)
    #expect(try Table.select(Cell(activation: .fp16, weight: .u4Lut, shape: .gemm)) == .lutU4)
    #expect(try Table.select(Cell(activation: .fp16, weight: .turboQuantH, shape: .gemm)) == .tqh)
    // fp16-act fused DUAL (DeltaNet A+B, gemv): only u4_lut/affine_u4 have a fused-dual kernel.
    #expect(
        try Table.select(Cell(activation: .fp16, weight: .u4Lut, shape: .gemv, fusion: .dualMatvec))
            == .lutU4)
    #expect(
        try Table.select(Cell(activation: .fp16, weight: .affineU4, shape: .gemv, fusion: .dualMatvec))
            == .affineU4)
    // fp16-act fused GATE+UP (gemv decode): SwiGLU u4_lut/affine_u4/binary1,
    // GeGLU affine_u4.
    #expect(
        try Table.select(Cell(activation: .fp16, weight: .u4Lut, shape: .gemv, fusion: .gateUpSwiglu))
            == .lutU4)
    #expect(
        try Table.select(Cell(activation: .fp16, weight: .affineU4, shape: .gemv, fusion: .gateUpSwiglu))
            == .affineU4)
    #expect(
        try Table.select(Cell(activation: .fp16, weight: .binary1, shape: .gemv, fusion: .gateUpSwiglu))
            == .binary1)
    #expect(
        try Table.select(Cell(activation: .fp16, weight: .ternary2, shape: .gemv, fusion: .gateUpSwiglu))
            == .ternary2)
    #expect(
        try Table.select(Cell(activation: .fp16, weight: .affineU4, shape: .gemv, fusion: .gateUpGeGLU))
            == .affineU4)
    // fp16-act PREFILL (gemm) fused: affine_u4 gate+up (SwiGLU/GeGLU) + dual all have BatchedFull
    // kernels. u4_lut has no batched fused gate+up/dual.
    #expect(
        try Table.select(Cell(activation: .fp16, weight: .affineU4, shape: .gemm, fusion: .gateUpSwiglu))
            == .affineU4)
    #expect(
        try Table.select(Cell(activation: .fp16, weight: .binary1, shape: .gemm, fusion: .gateUpSwiglu))
            == .binary1)
    #expect(
        try Table.select(Cell(activation: .fp16, weight: .affineU4, shape: .gemm, fusion: .gateUpGeGLU))
            == .affineU4)
    #expect(
        try Table.select(Cell(activation: .fp16, weight: .affineU4, shape: .gemm, fusion: .dualMatvec))
            == .affineU4)
    expectNotMeaningful(
        Cell(activation: .fp16, weight: .u4Lut, shape: .gemm, fusion: .gateUpSwiglu))
}

@Test func selectRejectsFusedDualForNonQuantWeights() {
    // A dense/tqh dual has no fused kernel — the probe must get .notMeaningful (→ nil → the
    // fused emitter falls back to separate matvecs), NOT a family.
    expectNotMeaningful(Cell(activation: .fp16, weight: .fp16, shape: .gemv, fusion: .dualMatvec))
    expectNotMeaningful(
        Cell(activation: .fp16, weight: .turboQuantH, shape: .gemv, fusion: .dualMatvec))
}

/// Assert select(cell) throws specifically `.notMeaningful`, not `.missing`.
private func expectNotMeaningful(_ cell: Cell) {
    do {
        _ = try Table.select(cell)
        Issue.record("expected .notMeaningful for \(cell), got a family")
    } catch let Table.SelectError.notMeaningful(c) {
        #expect(c == cell)
    } catch {
        Issue.record("expected .notMeaningful for \(cell), got \(error)")
    }
}

@Test func selectRejectsVariableSlotTqhAsNotMeaningful() throws {
    // Variable-slot tqh has NO emitMatvecVar consumer (the only caller is the per-layer input
    // gate, never tqh-packed), so it is notMeaningful — NOT a missing hole. The fixed-slot tqh
    // (tied LM head) stays registered. (The former bf16/fp32 dense holes now resolve — see
    // formerDenseHolesAreNowRegistered.)
    expectNotMeaningful(Cell(activation: .fp16, weight: .turboQuantH, slot: .variable))
    #expect(try Table.select(Cell(activation: .fp16, weight: .turboQuantH, slot: .fixed)) == .tqh)
}

@Test func selectThrowsNotMeaningfulForUnneededCombos() {
    // int32/raw are not matvec weights; a transposed u4 fp32-act matvec has no consumer.
    expectNotMeaningful(Cell(activation: .fp16, weight: .int32))
    expectNotMeaningful(Cell(activation: .fp32, weight: .affineU4, shape: .gemmTN, output: .fp32))
}

// MARK: - Authority / region-resolver agreement (resolveProjection rests on this)

private func projectionEntry(_ dt: SmeltDType) -> SmeltWeightEntry {
    if dt == .affineU4 {
        return SmeltWeightEntry(
            name: "q_proj", offset: 8192, sizeBytes: 1024, shape: [16, 64], dtype: .affineU4,
            groupSize: 64, scalesOffset: 12000, scalesSizeBytes: 32,
            biasesOffset: 20000, biasesSizeBytes: 32)
    }
    return SmeltWeightEntry(name: "w", offset: 4096, sizeBytes: 64, shape: [4, 8], dtype: dt)
}

@Test func gatewayAuthorizesExactlyWhatWeightLocatorResolvesForFP32Projections() {
    // resolveProjection makes select() the legality AUTHORITY and WeightLocator.resolve the
    // region resolver. That split only holds if the two agree on the dtype set: every dtype the
    // gateway authorizes as an fp32-act projection, WeightLocator can resolve — and vice versa.
    // Pin it so the two dtype switches can never drift.
    var authorized: Set<SmeltDType> = []
    for dt in [SmeltDType.fp16, .fp32, .bf16, .int32, .raw, .u4Lut, .affineU4, .turboQuantH] {
        let cell = Cell(activation: .fp32, weight: dt, shape: .gemv, fusion: .none,
                        output: .fp32, slot: .fixed)
        let gatewayAuthorizes = ((try? Table.select(cell)) != nil)
        let weightLocatorResolves = ((try? WeightLocator.resolve(projectionEntry(dt))) != nil)
        #expect(gatewayAuthorizes == weightLocatorResolves,
                "gateway vs WeightLocator disagree on fp32-act projection dtype \(dt)")
        if gatewayAuthorizes { authorized.insert(dt) }
    }
    // Pin the POSITIVE set too — a table that authorized nothing (or everything) would still pass
    // the boolean-agreement check above if WeightLocator drifted in lockstep.
    #expect(authorized == Set([.fp16, .fp32, .bf16, .affineU4]),
            "fp32-act projection authority must be exactly {fp16,fp32,bf16,affineU4}, got \(authorized)")
}
