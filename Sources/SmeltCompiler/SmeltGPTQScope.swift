import Foundation
import SmeltSchema

/// Which weights the GPTQ calibrator targets.
///
/// GPTQ needs a 2-D linear projection with a clean per-input activation Hessian,
/// i.e. the standard transformer projections (attention q/k/v/o, MLP
/// gate/up/down). Scope = those projections that resolve to affine_u4.
///
/// "Resolves to affine_u4" is the authoritative `SmeltAffineQuantizer`
/// decision, not the weight layout's dtype: the layout assigns dtype *before*
/// TQH routing, so a gate/up/down matched by `turboQuantHPatterns` shows up as
/// affine_u4 in the layout yet is actually quantized as turbo_quant_h. Keying
/// on the real decision means TQH routing, `excludePatterns`, unquantized
/// embeddings, and non-affine strategies all fall out correctly. The
/// projection-suffix check then drops the affine_u4 weights that are not
/// attention/MLP matmuls — `embed_tokens`, `lm_head`, DeltaNet `linear_attn_*` /
/// `out_proj`, draft-model/cluster projections.
enum SmeltGPTQScope {

    /// Canonical attention + MLP projection suffixes (see `SmeltWeightLayout`).
    /// DeltaNet uses `linear_attn_*` / `_out_proj_weight`, so it does not match.
    private static let projectionSuffixes = [
        "_self_attn_q_proj_weight",
        "_self_attn_k_proj_weight",
        "_self_attn_v_proj_weight",
        "_self_attn_o_proj_weight",
        "_mlp_gate_proj_weight",
        "_mlp_up_proj_weight",
        "_mlp_down_proj_weight",
    ]

    /// Whether `name` is a standard transformer attention/MLP projection.
    static func isTransformerProjection(_ name: String) -> Bool {
        projectionSuffixes.contains { name.hasSuffix($0) }
    }

    /// Whether `name` is in GPTQ scope under `quantization`.
    static func isInScope(name: String, quantization: SmeltQuantizationConfig) -> Bool {
        SmeltAffineQuantizer.shouldQuantizeAffine(name: name, config: quantization)
            && isTransformerProjection(name)
    }

    /// The in-scope subset of an already-resolved layout — entries whose `dtype`
    /// reflects final quant routing, as produced by `SmeltAffineQuantizer`
    /// (where TQH-routed and odd-column projections already carry their real
    /// non-affine dtype). Key on the resolved dtype here; for a spec-level
    /// decision from a name + config (before a layout is resolved), use
    /// `isInScope(name:quantization:)`. Do not pass `SmeltWeightLayout`
    /// pre-routing entries — their dtype can't distinguish TQH.
    static func inResolvedScope(_ entries: [SmeltWeightEntry]) -> [SmeltWeightEntry] {
        entries.filter(isResolvedInScope)
    }

    /// Whether a single already-resolved layout entry is in GPTQ scope. The emit
    /// path uses this so the capture-point predicate can't drift from `inResolvedScope`.
    static func isResolvedInScope(_ entry: SmeltWeightEntry) -> Bool {
        entry.dtype == .affineU4 && isTransformerProjection(entry.name)
    }
}
