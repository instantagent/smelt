// SmeltWeightRole — a name-based classifier for canonical Smelt tensor names, used to scope the
// `preserve_native` dtype policy (dtype-building-blocks plan U2c). A weight is kept at its native
// bf16/fp32 dtype ONLY when it is an eligible matvec PROJECTION that also matches a preserve_native
// glob — so the classifier's job is to identify projections AND, critically, to EXCLUDE the roles
// that must stay fp16 (down_proj's fp32-output range-protection path is fp16-weight-only; embeddings
// are rejected by embedToken for bf16/fp32; the cluster-sparse lm_head expects half; norms follow
// the activation ABI).
//
// Per the design consult: match EXACT canonical names + known SUFFIXES, never a broad
// `contains("_proj")` — `lm_head_weight`, `*_down_proj_weight`, and `masked_embedding_centroids_weight`
// are all easy to misclassify with a loose rule. The full canonical inventory (SmeltWeightPacker) is
// pinned by SmeltWeightRoleTests so a new weight name can't silently fall through.

import Foundation

/// The semantic role of a packed weight, for native-dtype-preservation eligibility. ONLY
/// `.projection` is preserve-eligible; every other role stays fp16 regardless of the glob.
public enum SmeltWeightRole: String, Sendable, Equatable, CaseIterable {
    /// Standard attention/FFN linear projections (q/k/v/o, gate/up) + the exotic linear projections
    /// (DeltaNet in/out, per-layer, draft-model pre/post). The matvec weights the U2 fp16-act
    /// dense kernels can lower — eligible for native-dtype preservation.
    case projection
    /// `*_down_proj_weight`. A projection, but the down-proj has an fp32-OUTPUT
    /// range-protection path that is fp16-WEIGHT-only (TopLevelEmitter/PrefillEmitter); a bf16/fp32
    /// down-proj would regress it. DEFERRED — never preserve-eligible until fp32-output bf16/fp32
    /// kernels land.
    case downProjection
    /// `embed_tokens*`, per-layer embeddings, cluster centroids. `embedToken` rejects bf16/fp32.
    case embedding
    /// `lm_head_weight`. The cluster-sparse LM-head path expects half weights.
    case lmHead
    /// Layer / q / k / projection norms (any `*_norm_weight` / `*_layernorm_weight`). Follow the
    /// activation ABI (fp16 or fp32), small — not a quantization/preservation target.
    case norm
    /// Conv / DeltaNet state (A_log, dt_bias) / scalars / biases / quant-metadata sidecars
    /// (`*_scales`/`*_biases`/`*_lut`). Not dense matvec weights.
    case auxiliary

    /// True iff a weight of this role may be kept at its native bf16/fp32 dtype (when it also
    /// matches a preserve_native glob). Only `.projection` — the safety gate the policy rests on.
    public var isNativePreserveEligible: Bool { self == .projection }
}

extension SmeltWeightRole {
    /// Classify a canonical Smelt tensor name. Weights are layer-prefixed (e.g.
    /// `layers_5_self_attn_q_proj_weight`, `delta_3_in_proj_qkv_weight`), so match SUFFIXES; a few
    /// model-global weights match exactly. Order is most-specific-first: metadata/aux and the
    /// exact non-projection names are checked BEFORE the projection suffixes so nothing leaks into
    /// `.projection`.
    public static func classify(_ name: String) -> SmeltWeightRole {
        // Quant-metadata sidecars + non-matvec state/scalars/biases. Checked first: a quantized
        // projection's sidecar is `..._q_proj_weight_scales`, which must NOT read as a projection.
        if name.hasSuffix("_scales") || name.hasSuffix("_biases") || name.hasSuffix("_lut")
            || name.hasSuffix("_A_log") || name.hasSuffix("_conv1d_weight")
            || name.hasSuffix("_dt_bias") || name.hasSuffix("_bias")
            || name.hasSuffix("_layer_scalar") {
            return .auxiliary
        }
        // Model-global exact names (before suffix rules).
        if name == "lm_head_weight" { return .lmHead }
        if name == "masked_embedding_centroids_weight" { return .embedding }
        // Embeddings: embed_tokens + per-layer embedding tables. embedToken rejects bf16/fp32.
        if name.contains("embed_tokens") || name.hasSuffix("_embedding_weight") {
            return .embedding
        }
        // Norms (covers *_norm_weight incl. q/k/projection/per-layer norms, *_layernorm_weight, and
        // the model-final exact `norm_weight`). BEFORE projections — a `*_projection_norm_weight`
        // is a norm, not a projection.
        if name.hasSuffix("_norm_weight") || name.hasSuffix("_layernorm_weight")
            || name == "norm_weight" {
            return .norm
        }
        // Down-proj — a projection, but deferred (fp32-output path is fp16-weight-only).
        if name.hasSuffix("_down_proj_weight") { return .downProjection }
        // Eligible linear projections — EXACT suffixes (never a generic `_proj_weight`).
        if isProjectionSuffix(name) { return .projection }
        // Unknown → auxiliary (conservative: never preserve-eligible).
        return .auxiliary
    }

    /// THE storage policy: true iff this weight must be kept at its native bf16 dtype instead of
    /// fp16-downcasting / quantizing. The single authority shared by the LAYOUT (appendWeightEntry)
    /// and BOTH quantizers (SmeltQuantizer / SmeltAffineQuantizer) so they can never disagree on a
    /// weight's stored dtype (the layout/quantizer drift the design consult flagged).
    ///
    /// Preserve iff: the token/logits graph's registered FP16 activation cell
    /// (embeddings/hidden trunks retain BF16 via their port-topology policy) AND
    /// the role is a preserve-eligible `.projection` (NOT down_proj / embedding / norm / lm_head) AND
    /// the name matches a `preserve_native` glob. preserve_native WINS over turbo_quant_h at the
    /// tensor level (callers check this BEFORE their TQH/affine/lut branches, so a preserved tensor
    /// leaves GPTQ/affine scope). bf16-SOURCE only in U2c — the layout tags `.bf16` optimistically;
    /// the quantizer is authoritative and throws if a matched tensor's source isn't BF16 (fp16 source
    /// makes preserve a no-op; fp32-source preservation is a deferred unit — needs a source-aware
    /// layout step the IR can't provide today).
    public static func preservesNativeBF16(
        name: String, activationDtype: SmeltDType, config: SmeltQuantizationConfig
    ) -> Bool {
        guard activationDtype == .fp16 else { return false }
        guard classify(name).isNativePreserveEligible else { return false }
        // The preserve_native globs follow the SAFE exact-or-glob convention shared with
        // turbo_quant_h (matchesExactOrGlob): a bare `q_proj_weight` matches exactly, so per-layer
        // projections that should span layers must use a wildcard (e.g. `*_q_proj_weight`).
        return matchesExactOrGlob(name: name, patterns: config.preserveNativePatterns)
    }

    /// Validate a preserve_native source tensor is true bf16 of exactly `rows*cols` elements.
    /// Returns nil when the source is valid; otherwise the detail string for the caller's
    /// per-quantizer `preserveNativeRequiresBF16Source` error. Single-sources the load-bearing
    /// preflight so the LUT and affine quantizers can't drift (U2c). bf16-source only — the
    /// optimistic `.bf16` layout tag is only safe if the real source is genuinely 2-byte bf16.
    static func preserveNativeSourceRejection(
        dtype: String, byteCount: Int, rows: Int, cols: Int
    ) -> String? {
        guard dtype == "BF16" else { return dtype }
        let expected = rows * cols * 2
        guard byteCount == expected else { return "\(dtype) byteCount=\(byteCount)≠\(expected)" }
        return nil
    }

    /// The exact projection suffixes / global projection names that are preserve-eligible.
    private static func isProjectionSuffix(_ name: String) -> Bool {
        // Standard attention + FFN (down_proj handled separately above).
        for s in ["_q_proj_weight", "_k_proj_weight", "_v_proj_weight", "_o_proj_weight",
                  "_gate_proj_weight", "_up_proj_weight",
                  // DeltaNet linear in/out projections.
                  "_out_proj_weight", "_in_proj_a_weight", "_in_proj_b_weight",
                  "_in_proj_qkv_weight", "_in_proj_z_weight",
                  // AltUp per-layer projections (per_layer_input_gate routes emitMatvecVar).
                  "_per_layer_input_gate_weight", "_per_layer_projection_weight"]
        where name.hasSuffix(s) {
            return true
        }
        // Model-global projections (draft-model pre/post, the per-layer model projection).
        return name == "per_layer_model_projection_weight"
            || name == "pre_projection_weight"
            || name == "post_projection_weight"
    }
}
