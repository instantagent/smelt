// MatvecKernelTable — the single dtype→kernel-FAMILY selection authority that every
// matvec consumer MUST call (the "gateway" of the dtype-building-blocks plan,
// docs/dtype-building-blocks-plan.md §0.5/§1).
//
// THE PROBLEM IT RETIRES: the matvec kernel matrix is split by ACTIVATION dtype and was
// selected by TWO divergent inline switches — SmeltCodeEmitter.emitMatvec (fp16-act:
// fp16/affine_u4/u4_lut/tqh, throws bf16/fp32) and WeightLocator.resolve /
// DenseTrunkGuards.resolveProjection (fp32-act: f32/f16/bf16 dense + affine_u4) — with
// OPPOSITE coverage. Two authorities that can drift is the architectural debt the building-
// blocks goal retires: a dtype/quant selects a KERNEL LEGO; it never silently gates WHAT
// COMPILES, and a new matvec consumer can never silently inherit fp16.
//
// THE INVARIANT, ENFORCED (not merely tracked) by three mechanisms together:
//   1. GATEWAY (this type). `select(cell)` is TOTAL over the typed `Cell` space and has NO
//      silent default — an unregistered MEANINGFUL cell is a LOUD throw (`.missing`), exactly
//      like emitMatvec's old exhaustive switch but in ONE place. A new op hitting an unfilled
//      cell FAILS at emit/test time; it cannot silently get fp16.
//   2. NO-BYPASS LINT (a later unit). A structural test asserts the matvec pipelines +
//      `weightEntry.dtype` matvec-routing live ONLY inside allowlisted lowering helpers — so
//      every consumer is FORCED through `select()`. (No-silent-default is necessary but not
//      sufficient: a caller could still hand-construct a `.fp16Matvec` dispatch; the lint
//      forbids that.)
//   3. PARITY GATE. `meaningfulCells` is the declared truth of what SHOULD exist;
//      `registered == meaningfulCells \ knownMissing` is derived so the sets can't drift. As of U2
//      `knownMissing` is EMPTY — the bf16/fp32 dense kernels are hand-written and wired, so every
//      meaningful cell resolves and the no-hole claim is HARD (the gate pins knownMissing empty; a
//      future partial op must add a kernel rather than silently re-growing the hole set).
//
// THE FAMILY IS ACTIVATION-AGNOSTIC. `Family` is the dtype→family choice ONLY — it carries no
// bindings/constants/grids/specialization (each emitter keeps its own lowering). It unifies
// `WeightLocator.Kind` (`.dense(dtype)` / `.affineU4`) with the fp16-act-only `.lutU4`/`.tqh`
// families. The CALLER is already in a fixed activation context (emitMatvec is fp16-act, the
// DenseTrunk emitters select their activation axis explicitly, so it maps
// `(activation + family)` to the concrete
// kernel: `.dense(.bf16)` lowers to fp16_matvec-shaped bf16 in the LLM path and to
// gemv_bf16w_f32 in the dense-trunk path. One family, two lowerings — the unification the goal asks
// for.

import SmeltSchema

public enum MatvecKernelTable {

    // MARK: - Cell axes (the typed contract — NOT a cartesian product; the table DEFINES
    // which combinations are meaningful)

    /// Activation storage dtype. A consumer lowers the selected family to the
    /// operation kernel for this storage axis.
    public enum Activation: String, CaseIterable, Hashable, Sendable {
        case fp16
        case bf16
        case fp32
    }

    /// Matvec shape family. gemv = M=1 decode; gemm = M>1 prefill (non-transposed);
    /// gemmTN = transposed prefill.
    public enum Shape: String, CaseIterable, Hashable, Sendable { case gemv, gemm, gemmTN }

    /// Output storage dtype of the matvec.
    public enum Output: String, CaseIterable, Hashable, Sendable {
        case fp16
        case bf16
        case fp32
    }

    /// Input-slot kind: a fixed slot index vs a variable (named) slot.
    public enum Slot: String, CaseIterable, Hashable, Sendable { case fixed, variable }

    /// What is fused into the matvec. Fused families are added here WITH their consumers (never
    /// ahead of one). The fusion axis is LOAD-BEARING for AUTHORIZATION: a fused cell
    /// (e.g. affine_u4 × gate-up) is its OWN registration, so "a generic affine GEMM exists"
    /// cannot authorize "a fused gate-up kernel exists" — the plan's §0.5 invariant. The concrete
    /// fused pipeline (fused_dual_lut_matvec vs fused_dual_affine_matvec, the GeGLU/SwiGLU split,
    /// shape specialization) is still the emitter's choice; the table only authorizes the family.
    public enum Fusion: String, CaseIterable, Hashable, Sendable {
        case none
        case dualMatvec     // two same-dtype weights → one fused dispatch (DeltaNet A+B)
        case gateUpSwiglu   // fused gate+up projection + SwiGLU
        case gateUpGeGLU    // fused gate+up projection + GeGLU
    }

    /// One typed point in the (activation × weight × shape × fusion × output × slot) matrix.
    public struct Cell: Hashable, Sendable, CustomStringConvertible {
        public let activation: Activation
        public let weight: SmeltDType
        public let shape: Shape
        public let fusion: Fusion
        public let output: Output
        public let slot: Slot

        public init(
            activation: Activation, weight: SmeltDType, shape: Shape = .gemv,
            fusion: Fusion = .none, output: Output = .fp16, slot: Slot = .fixed
        ) {
            self.activation = activation
            self.weight = weight
            self.shape = shape
            self.fusion = fusion
            self.output = output
            self.slot = slot
        }

        public var description: String {
            "\(activation.rawValue)-act × \(weight.rawValue) × \(shape.rawValue)"
                + " × \(fusion.rawValue) × \(output.rawValue)-out × \(slot.rawValue)-slot"
        }
    }

    // MARK: - The family (dtype→family choice ONLY)

    /// The kernel-family choice. Activation-agnostic: the caller's known activation picks the
    /// concrete kernel (`.dense(.bf16)` → fp16_matvec-shaped bf16 in the LLM path,
    /// gemv_bf16w_f32 in the dense-trunk path). Carries no bindings/constants/specialization —
    /// `.affineU4`'s group size, regions, etc. stay at the binding site (it reads the entry).
    ///
    /// `WeightLocator.Kind` (`.dense(SmeltDType)`/`.affineU4(groupSize:)`) is a DELIBERATELY
    /// separate type (plan §1): it is the DenseTrunk REGION resolver's binding output (it carries
    /// group size + offsets), whereas `Family` is the matvec-wide dtype→family AUTHORITY tag
    /// (it also covers `.lutU4`/`.tqh`, which the trunk-scoped WeightLocator throws on). They are
    /// kept apart on purpose, not unified. Drift between the two dtype switches is prevented two
    /// ways: both `family(for:)` and `WeightLocator.resolve` are `default`-free (a new SmeltDType
    /// is a compile error in BOTH), and `gatewayAuthorizesExactlyWhatWeightLocatorResolves...`
    /// pins that they agree on the fp32-act projection dtype set.
    public enum Family: Hashable, Sendable {
        case dense(SmeltDType)   // fp16/bf16/fp32 dense weight
        case affineU4            // group-wise affine int4
        case binary1             // signed 1-bit + per-group scale
        case ternary2            // signed ternary 2-bit + per-group scale
        case lutU4               // u4 LUT (fp16-act only)
        case tqh                 // turbo_quant_h (fp16-act only)
    }

    public enum SelectError: Error, CustomStringConvertible {
        /// The cell is meaningful but has no registered kernel yet — the building-block hole a
        /// future unit fills. Loud by construction (no silent fp16 fallback).
        case missing(Cell, hint: String)
        /// No consumer needs this combination — reaching it is a caller bug, not a missing kernel.
        case notMeaningful(Cell)

        public var description: String {
            switch self {
            case let .missing(cell, hint):
                return "MatvecKernelTable: no registered kernel for \(cell) — \(hint)"
            case let .notMeaningful(cell):
                return "MatvecKernelTable: \(cell) is not a meaningful matvec cell "
                    + "(no consumer needs this activation/weight/shape/fusion/output/slot combo)"
            }
        }
    }

    // MARK: - The matrix

    /// Declared truth of every cell that SHOULD have a kernel. Built explicitly (not a blind
    /// cartesian product) — the table DEFINES meaningfulness, so combos no consumer needs are
    /// simply absent. Grows as fused/output/shape lowering sites are routed through `select()`.
    public static let meaningfulCells: Set<Cell> = {
        var cells: Set<Cell> = []

        // fp16-ACTIVATION generic LLM matvec (gemv, fusion none, output fp16). fp16 +
        // affine_u4/u4_lut + bf16/fp32 (the U2 fp16_matvec_{bf16,fp32}w kernels) all registered.
        // Both input-slot kinds (emitMatvec=fixed, emitMatvecVar=variable).
        for slot in [Slot.fixed, .variable] {
            for w in [SmeltDType.fp16, .bf16, .fp32, .affineU4, .binary1,
                      .ternary2, .u4Lut] {
                cells.insert(Cell(activation: .fp16, weight: w, slot: slot))
            }
        }
        // tqh: the FIXED-slot kernel (tied LM head) is registered. There is NO variable-slot tqh
        // CONSUMER — emitMatvecVar's only caller is the per-layer input gate, packed via
        // appendWeightEntry (fp16 / affine_u4 / u4_lut, never tqh). So a variable-slot tqh cell is
        // notMeaningful, NOT a hole: per §0.5 the table never registers ahead of a consumer, so it
        // is added WITH its consumer + an emitTQHMatvecVar kernel if a TQH-packed emitMatvecVar
        // weight ever appears. Until then select() throws .notMeaningful for it.
        cells.insert(Cell(activation: .fp16, weight: .turboQuantH, slot: .fixed))

        // fp16-ACTIVATION batched prefill matmul (gemm, output fp16) — emitBatchedMatmul. fp16 +
        // affine_u4/u4_lut/tqh registered (the batched/fused/per-batch kernels); bf16/fp32 also
        // registered (U2 — the per-batch dense path reuses the gemv fp16_matvec_{bf16,fp32}w kernels).
        for w in [SmeltDType.fp16, .bf16, .fp32, .affineU4, .binary1,
                  .ternary2, .u4Lut, .turboQuantH] {
            cells.insert(Cell(activation: .fp16, weight: w, shape: .gemm))
        }

        // fp16-ACTIVATION fused DUAL projection (DeltaNet A+B, gemv decode): two same-dtype
        // weights → one fused dispatch. u4_lut → fused_dual_lut_matvec, affine_u4 →
        // fused_dual_affine_matvec. Only these two quant families have a fused-dual kernel; a
        // dense/tqh dual has none, so the gateway authorizes only u4_lut/affine_u4 here and the
        // fused emitter falls back to separate matvecs otherwise.
        for w in [SmeltDType.u4Lut, .affineU4] {
            cells.insert(Cell(activation: .fp16, weight: w, shape: .gemv, fusion: .dualMatvec))
        }

        // fp16-ACTIVATION fused GATE+UP FFN — gate+up projection + activation in one dispatch.
        // DECODE (gemv, TopLevelEmitter): SwiGLU has u4_lut (fused_gate_up_swiglu) + affine_u4
        // (fused_affine_gate_up_swiglu); GeGLU has affine_u4 (fused_affine_gate_up_geglu). The
        // unfused fallback emits separate gateway-routed matvecs.
        cells.insert(Cell(activation: .fp16, weight: .u4Lut, shape: .gemv, fusion: .gateUpSwiglu))
        cells.insert(Cell(activation: .fp16, weight: .affineU4, shape: .gemv, fusion: .gateUpSwiglu))
        cells.insert(Cell(activation: .fp16, weight: .binary1, shape: .gemv, fusion: .gateUpSwiglu))
        cells.insert(Cell(activation: .fp16, weight: .ternary2, shape: .gemv, fusion: .gateUpSwiglu))
        cells.insert(Cell(activation: .fp16, weight: .affineU4, shape: .gemv, fusion: .gateUpGeGLU))

        // PREFILL/batched (gemm, PrefillEmitter): the affine_u4 fused families exist as
        // BatchedFull kernels — gate+up (SwiGLU + GeGLU) and dual (DeltaNet A+B + fused K+V).
        // u4_lut has no batched fused gate+up/dual kernel, so those gemm fused cells are absent
        // (a u4_lut prefill FFN/dual falls to separate gateway-routed matvecs).
        cells.insert(Cell(activation: .fp16, weight: .affineU4, shape: .gemm, fusion: .gateUpSwiglu))
        cells.insert(Cell(activation: .fp16, weight: .binary1, shape: .gemm, fusion: .gateUpSwiglu))
        cells.insert(Cell(activation: .fp16, weight: .affineU4, shape: .gemm, fusion: .gateUpGeGLU))
        cells.insert(Cell(activation: .fp16, weight: .affineU4, shape: .gemm, fusion: .dualMatvec))

        // fp32-ACTIVATION TTS dense-trunk matvec — gemv (decode) + gemm/gemmTN (prefill),
        // output fp32.
        for shape in [Shape.gemv, .gemm, .gemmTN] {
            for w in [SmeltDType.fp32, .fp16, .bf16] {
                cells.insert(Cell(activation: .fp32, weight: w, shape: shape, output: .fp32))
            }
        }
        // fp32-act affine u4 — gemv + gemm (no transposed u4 consumer).
        for shape in [Shape.gemv, .gemm] {
            cells.insert(Cell(activation: .fp32, weight: .affineU4, shape: shape, output: .fp32))
        }

        // BF16-activation dense trunk. Decode and prefill share the same
        // operation family; the shape axis selects GEMV or GEMM lowering.
        for shape in [Shape.gemv, .gemm] {
            cells.insert(
                Cell(
                    activation: .bf16,
                    weight: .bf16,
                    shape: shape,
                    output: .bf16
                )
            )
        }

        return cells
    }()

    /// Meaningful cells with no registered kernel yet. EMPTY as of U2: the fp16-act × {bf16,fp32}
    /// dense matvecs (the headline holes — no fp16-activation bf16/fp32 dense kernel existed; the
    /// dense-trunk gemv_bf16w_f32 etc. are fp32-act, a different shape) now have hand-written kernels
    /// (`fp16_matvec_{bf16,fp32}w`, gated by FP16ActDenseKernelTests) wired through
    /// emitMatvec / emitMatvecVar / emitBatchedMatmul. The no-hole claim is now HARD: every
    /// meaningful cell resolves. A future partial op must add a kernel — select() throws or the
    /// no-bypass lint fires; it can never silently re-grow this set to dodge the parity gate (the
    /// test pins it empty).
    public static let knownMissing: Set<Cell> = []

    /// Registered = meaningful and NOT missing. Derived (computed once) so the three sets can
    /// never drift apart.
    public static let registered: Set<Cell> = meaningfulCells.subtracting(knownMissing)

    // MARK: - Selection (total, no silent default)

    /// THE gateway. Total over the typed cell space; no silent default. Returns the dtype→family
    /// choice for a registered cell, throws `.missing` for a meaningful hole, throws
    /// `.notMeaningful` for a combo no consumer needs.
    public static func select(_ cell: Cell) throws -> Family {
        guard meaningfulCells.contains(cell) else {
            throw SelectError.notMeaningful(cell)
        }
        if knownMissing.contains(cell) {
            throw SelectError.missing(cell, hint: missingHint(cell))
        }
        guard let fam = family(for: cell.weight) else {
            // Unreachable: registered cells never carry int32/raw. Kept loud, never silent.
            throw SelectError.notMeaningful(cell)
        }
        return fam
    }

    /// Weight dtype → family. Exhaustive (no default) so a new SmeltDType is a compile error
    /// here, not a silent miscategorization. Returns nil for dtypes that are not matvec weights.
    static func family(for weight: SmeltDType) -> Family? {
        switch weight {
        case .fp16, .bf16, .fp32: return .dense(weight)
        case .affineU4: return .affineU4
        case .binary1: return .binary1
        case .ternary2: return .ternary2
        case .u4Lut: return .lutU4
        case .turboQuantH: return .tqh
        case .int32, .raw: return nil
        }
    }

    private static func missingHint(_ cell: Cell) -> String {
        // knownMissing is empty as of U2, so this is reachable only when a FUTURE op declares a
        // meaningful cell without a kernel (it added the cell to knownMissing). The fix is always
        // the same: write the kernel for this (activation, weight, shape, fusion, output, slot) and
        // remove the cell from knownMissing — never fall back to fp16.
        "\(cell.weight.rawValue)-weight matvec for this cell has no registered kernel; hand-write "
            + "it and remove the cell from MatvecKernelTable.knownMissing — do not fall back to fp16"
    }
}
