// DenseTrunkGuards — shared dense headless-trunk legality and routing.
// One shared contract keeps decode and prefill in lockstep. Every IR feature
// that would change topology must be registered explicitly or rejected loudly.

import Foundation
import SmeltSchema

public enum DenseTrunkGuards {

    /// Reject any IR knob outside the currently registered dense topology.
    public static func validate(
        cfg: SmeltConfig,
        attn: SmeltAttentionConfig,
        layerPattern: SmeltLayerPattern
    ) throws {
        guard layerPattern.expanded.allSatisfy({ $0 == .attention }) else {
            throw DenseTrunkEmitterError.unsupported(
                "dense trunks support uniform dense attention patterns only")
        }
        guard cfg.normMode == .weight, attn.qkNormMode == .weight else {
            throw DenseTrunkEmitterError.unsupported(
                "dense norm kernels are weight-direct; norm_mode/qk_norm_mode "
                + "must be 'weight' (no build-time w−1 transform on this path)")
        }
        guard attn.qkNorm else {
            throw DenseTrunkEmitterError.unsupported(
                "the dense trunk applies per-head qk-norm; qk_norm=false "
                + "trunks need a rope-only route (not emitted yet)")
        }
        guard !attn.gatedQ else {
            throw DenseTrunkEmitterError.unsupported(
                "gated_q doubles q_proj (qProjDim = 2·heads·headDim); the qkv "
                + "route assumes qDim = heads·headDim — not emitted")
        }
        guard !attn.qkvBias else {
            throw DenseTrunkEmitterError.unsupported(
                "qkv_bias adds projection-side bias tensors; the dense trunk qkv "
                + "route does not apply them — not emitted")
        }
        guard !attn.vNorm else {
            throw DenseTrunkEmitterError.unsupported(
                "v_norm (scale-less per-head V RMS) is not in the dense sequence — not emitted")
        }
        guard !attn.externalKV else {
            throw DenseTrunkEmitterError.unsupported(
                "external_kv (cross-attention over a target's cache) has no own "
                + "k/v projection — the dense trunk projects its own — not emitted")
        }
        guard attn.ropeLayout == .splitHalf else {
            throw DenseTrunkEmitterError.unsupported(
                "dense rope kernels apply split-half/rotate-half RoPE; the trunk "
                + "must declare rope_layout split_half explicitly (the topology "
                + "default is interleaved — a silent parity trap)")
        }
        guard cfg.hiddenSizePerLayerInput == 0 else {
            throw DenseTrunkEmitterError.unsupported(
                "per-layer residual input slabs are not in the dense sequence — not emitted")
        }
        guard cfg.attnLogitCap == nil else {
            throw DenseTrunkEmitterError.unsupported(
                "attn_logit_cap has no registered dense GQA kernel — not emitted")
        }
        guard cfg.sharedKVLayers == 0 else {
            throw DenseTrunkEmitterError.unsupported(
                "shared-KV layers ship a different per-layer weight set and "
                + "sequence — not emitted")
        }
        guard cfg.ffn.activation == .swiglu else {
            throw DenseTrunkEmitterError.unsupported(
                "the dense FFN route is swiglu only (got \(cfg.ffn.activation))")
        }
        guard attn.slidingWindow == 0 else {
            throw DenseTrunkEmitterError.unsupported(
                "the dense GQA kernels attend the full causal cache; sliding-window "
                + "attention is not emitted")
        }
        guard attn.attnScale == 1 else {
            throw DenseTrunkEmitterError.unsupported(
                "dense GQA kernels hardcode the 1/sqrt(head_dim) score scale; "
                + "a non-unit attn_scale is not emitted")
        }
        guard attn.ropeScaling == nil else {
            throw DenseTrunkEmitterError.unsupported(
                "rope_scaling rewrites the RoPE tables; the dense trunk has no "
                + "scaled-rope parity coverage — not emitted")
        }
        let ropeDim = attn.effectiveRopeDim(default: cfg.ropeDim)
        guard ropeDim == attn.headDim else {
            throw DenseTrunkEmitterError.unsupported(
                "the dense route applies RoPE over the full head_dim and strides "
                + "rope rows by head_dim; partial rope_dim (\(ropeDim) != "
                + "head_dim \(attn.headDim)) is not emitted")
        }
        guard attn.headDim <= 128 else {
            throw DenseTrunkEmitterError.unsupported(
                "dense GQA kernels have fixed 128-wide per-head registers; "
                + "head_dim > 128 is not emitted")
        }
        guard attn.headDim % 4 == 0, cfg.hiddenSize % 4 == 0, cfg.ffn.dim % 4 == 0 else {
            throw DenseTrunkEmitterError.unsupported(
                "the bf16 GEMV/GEMM kernels contract in float4 chunks and drop the "
                + "tail; head_dim, hidden_size and ffn.dim must each be multiples of 4")
        }
    }

    /// Resolve a trunk PROJECTION weight to the dtype-region kind the emitter binds. This is the
    /// projection-dtype guard — the one binding-site check that otherwise leaks into each
    /// emitter's `proj()` — so the decode and prefill tables can never drift on which projection
    /// dtypes the trunk supports.
    ///
    /// LEGALITY now comes from the ONE matvec gateway (`MatvecKernelTable.select`), the SAME
    /// authority the fp16-activation `emitMatvec` switch consults — so "which dtypes are a legal
    /// matvec" cannot drift between the dense and token-graph paths (the dtype-building-blocks collapse,
    /// docs/dtype-building-blocks-plan.md §1). `WeightLocator` stays the dense-region resolver
    /// (offsets/sizes/group size). The emitter then branches on the returned kind to pick its
    /// route: `.dense(.bf16)` → fused bf16; `.dense(.fp32)`/`.dense(.fp16)` → the UNFUSED dense
    /// route (gemv_f32/gemv_f16w_f32); `.affineU4` → the UNFUSED u4 route (gemv_u4_f32/
    /// gemm_u4_f32). Dtype picks a KERNEL lego, never gates what compiles.
    public static func resolveProjection(
        _ entry: SmeltWeightEntry,
        activationDtype: SmeltDType,
        shape: MatvecKernelTable.Shape,
        name: String
    ) throws -> WeightLocator.Kind {
        // Legality AUTHORITY — runs FIRST so the gateway, not a local switch, decides which
        // dtypes are a legal trunk projection. Shape is supplied by the caller, so a decode GEMV
        // can never accidentally authorize a prefill GEMM (or vice versa). The gateway throws for
        // any non-projection cell; map it to this surface's error type so callers keep the dense
        // trunk error surface.
        let activation: MatvecKernelTable.Activation
        let output: MatvecKernelTable.Output
        switch activationDtype {
        case .bf16:
            activation = .bf16
            output = .bf16
        case .fp32:
            activation = .fp32
            output = .fp32
        default:
            throw DenseTrunkEmitterError.unsupported(
                "dense trunk projection routing needs bf16 or fp32 activation storage, got "
                    + activationDtype.rawValue
            )
        }
        do {
            _ = try MatvecKernelTable.select(MatvecKernelTable.Cell(
                activation: activation,
                weight: entry.dtype,
                shape: shape,
                fusion: .none,
                output: output,
                slot: .fixed
            ))
        } catch let e as MatvecKernelTable.SelectError {
            throw DenseTrunkEmitterError.wrongDtype(
                name: name, expected: "bf16/fp16/fp32 or u4 (a legal trunk projection)",
                got: "\(entry.dtype.rawValue) [\(e)]")
        }
        // Region resolution for the authorized dtype. WeightLocator.resolve admits EXACTLY the
        // dtypes the gateway authorizes for an fp32-act projection (asserted by a consistency
        // gate), so for any authorized dtype it returns a kind — its own dtype switch is the
        // unreachable-defensive backstop here.
        return try WeightLocator.resolve(entry).kind
    }

    /// Assert a layer's projection weights are a UNIFORM dtype-region kind (all one of f32/f16/bf16/u4)
    /// and return that shared kind. The trunk's kind-branched layer emit (fused bf16 qkv/gateup
    /// vs the f32/f16/u4 unfused per-projection route) picks ONE route per layer from q_proj's kind, so a within-layer
    /// mix would silently bind the odd-dtype weights through the wrong kernel — exactly the
    /// dispatch-safety footgun u4 emission introduces. A real build is uniform; a mixed manifest
    /// fails loud here (the homogeneous-dtype guard) rather than mis-emitting. Each `name` pairs
    /// with its entry for a precise diagnostic.
    public static func uniformProjectionKind(
        _ named: [(name: String, entry: SmeltWeightEntry)],
        activationDtype: SmeltDType,
        shape: MatvecKernelTable.Shape
    ) throws -> WeightLocator.Kind {
        guard let first = named.first else {
            throw DenseTrunkEmitterError.unsupported("a trunk layer must have projection weights")
        }
        let kind = try resolveProjection(
            first.entry,
            activationDtype: activationDtype,
            shape: shape,
            name: first.name
        )
        for p in named.dropFirst()
        where try resolveProjection(
            p.entry,
            activationDtype: activationDtype,
            shape: shape,
            name: p.name
        ) != kind {
            throw DenseTrunkEmitterError.unsupported(
                "trunk layer mixes projection dtypes ('\(p.name)' differs from '\(first.name)') — "
                + "a layer must be uniformly ONE dtype (f32/f16/bf16/u4); the kind-branched emit picks "
                + "one route (bf16 fused vs the f32/f16/u4 unfused route) off the layer's q_proj kind")
        }
        return kind
    }
}
