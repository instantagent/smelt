// WeightLocator — Phase 2 (docs/lego-phase2-plan.md): the dtype→region RESOLUTION
// half of SmeltCodeEmitter.emitMatvec's switch, extracted as a pure value type so the
// DenseTrunk emitters (the only matvec emitters NOT already routed through emitMatvec —
// they predate it and special-case bf16) can share ONE entry→{kind, absolute region
// offsets} mapping instead of each hardcoding a single bf16 offset. A compiled block's
// weight binding goes through this resolver; a fixed-dtype emitter that assumes one
// dtype is a latent dtype-coverage bug — the CLAUDE.md "Dtype dispatch safety"
// invariant, generalized from the matvec emit site to the binding site.
//
// SCOPE: DenseTrunk emitters only (codex §8). It is COMPLETE for that domain — a trunk's
// projections are only ever dense bf16/f16/fp32 or affine u4; a trunk never carries
// u4_lut/turbo_quant_h weights — and throws loud on anything else (dispatch-safe). Do
// NOT route the generic emitMatvec through it: emitMatvec already handles u4_lut +
// turbo_quant_h, which this resolver throws on, so it would be a PARTIAL replacement of
// the proven generic switch. The generic text path stays exactly as-is.
//
// Offsets are read VERBATIM from the entry, which already carries them ABSOLUTE into
// weights.bin. Any relative→absolute translation (the TTS native convention stores
// scale/bias offsets relative to the block start) happens where the entry is
// CONSTRUCTED (Phase 2a), never here — the locator is pure offset pass-through.

import Foundation
import SmeltSchema

public struct WeightLocator: Equatable, Sendable {
    /// How the weight's bytes are typed at the binding site.
    public enum Kind: Equatable, Sendable {
        case dense(SmeltDType)            // bf16/f16/fp32 — one contiguous region
        case affineU4(groupSize: Int)    // group-wise affine int4 — nibbles + scales + biases
        case signed(SmeltDType, groupSize: Int) // binary1/ternary2 — codes + scales
    }

    /// An absolute byte range into weights.bin.
    public struct Region: Equatable, Sendable {
        public let offset: UInt64
        public let len: UInt64
        public init(offset: UInt64, len: UInt64) {
            self.offset = offset
            self.len = len
        }
    }

    public let kind: Kind
    /// `.dense` → `[weights]`; `.affineU4` → `[nibbles, scales, biases]`, in the order
    /// the affine kernel binds them (weights@0, scales@1, biases@2). Absolute offsets.
    public let regions: [Region]

    public init(kind: Kind, regions: [Region]) {
        self.kind = kind
        self.regions = regions
    }

    public enum ResolveError: Error, CustomStringConvertible {
        case unsupportedDType(name: String, dtype: SmeltDType)
        case missingAffineRegions(name: String)
        case missingSignedRegions(name: String, dtype: SmeltDType)

        public var description: String {
            switch self {
            case let .unsupportedDType(name, dtype):
                return "WeightLocator: no DenseTrunk binding for dtype \(dtype.rawValue) "
                    + "(weight '\(name)') — this resolver covers dense bf16/f16/fp32 + "
                    + "affine_u4 only; u4_lut/turbo_quant_h are generic emitMatvec dtypes; "
                    + "int32/raw are unsupported there too"
            case .missingAffineRegions(let name):
                return "WeightLocator: affine_u4 weight '\(name)' is missing its "
                    + "scales/biases offsets, sizes, or group size"
            case let .missingSignedRegions(name, dtype):
                return "WeightLocator: \(dtype.rawValue) weight '\(name)' is missing its "
                    + "scale offset, size, or group size"
            }
        }
    }

    /// Resolve one weight entry to its binding kind + absolute region offsets. Exhaustive
    /// on dtype (no default) so a new SmeltDType is a compile error here, not a silent
    /// drop to a wrong binding.
    public static func resolve(_ e: SmeltWeightEntry) throws -> WeightLocator {
        switch e.dtype {
        case .bf16, .fp16, .fp32:
            return WeightLocator(
                kind: .dense(e.dtype),
                regions: [Region(offset: e.offset, len: e.sizeBytes)])
        case .affineU4:
            guard let scalesOff = e.scalesOffset, let scalesLen = e.scalesSizeBytes,
                  let biasesOff = e.biasesOffset, let biasesLen = e.biasesSizeBytes,
                  let groupSize = e.groupSize else {
                throw ResolveError.missingAffineRegions(name: e.name)
            }
            return WeightLocator(
                kind: .affineU4(groupSize: groupSize),
                regions: [
                    Region(offset: e.offset, len: e.sizeBytes),
                    Region(offset: scalesOff, len: scalesLen),
                    Region(offset: biasesOff, len: biasesLen),
                ])
        case .binary1, .ternary2:
            guard let scalesOff = e.scalesOffset,
                  let scalesLen = e.scalesSizeBytes,
                  let groupSize = e.groupSize
            else {
                throw ResolveError.missingSignedRegions(name: e.name, dtype: e.dtype)
            }
            return WeightLocator(
                kind: .signed(e.dtype, groupSize: groupSize),
                regions: [
                    Region(offset: e.offset, len: e.sizeBytes),
                    Region(offset: scalesOff, len: scalesLen),
                ])
        case .u4Lut, .turboQuantH, .int32, .raw:
            throw ResolveError.unsupportedDType(name: e.name, dtype: e.dtype)
        }
    }
}
