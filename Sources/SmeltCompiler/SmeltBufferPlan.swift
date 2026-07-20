// SmeltBufferPlan — Compute buffer slot allocations from a validated IR.
//
// The buffer plan is deterministic: given the same IR, it produces the same
// slot indices and sizes. This is critical because generated Swift code
// references slots by integer index — if the plan changes, the generated
// code is invalid.
//
// Slot layout:
//   0-25   Fixed activation/intermediate slots (shared across all models)
//   26-29  Reserved
//   30     Monolithic weight buffer (mmap, size set by weight packer)
//   31+    Dynamically allocated per-layer state, tables, and dynamic scalars
//
// The dynamic region uses a running counter — no hard-coded ranges.
// This supports arbitrary layer counts without slot collisions.

// MARK: - Buffer plan

public struct SmeltRoPEParams: Sendable, Hashable {
    public let theta: Float
    public let dim: Int
    public let freqDim: Int?
    public let scaling: SmeltRoPEScaling?
    public let layout: String

    public init(
        theta: Float,
        dim: Int,
        freqDim: Int? = nil,
        scaling: SmeltRoPEScaling? = nil,
        layout: String = "interleaved"
    ) {
        self.theta = theta
        self.dim = dim
        self.freqDim = freqDim
        self.scaling = scaling
        self.layout = layout
    }
}

public struct PlannedRoPETablePair: Sendable, Equatable {
    public let params: SmeltRoPEParams
    public let cosSlot: Int
    public let sinSlot: Int

    public init(params: SmeltRoPEParams, cosSlot: Int, sinSlot: Int) {
        self.params = params
        self.cosSlot = cosSlot
        self.sinSlot = sinSlot
    }
}

/// Complete buffer allocation plan, produced from a validated SmeltModelIR.
public struct SmeltBufferPlan: Sendable, Equatable {
    /// All allocated slots, sorted by index.
    public let slots: [PlannedSlot]

    /// Index where conv states start (for generated code).
    public let convStateBaseSlot: Int
    /// Index where rec states start.
    public let recStateBaseSlot: Int
    /// Index where key caches start.
    public let keyCacheBaseSlot: Int
    /// Index where value caches start.
    public let valCacheBaseSlot: Int
    /// Slot index for RoPE cos table.
    public let ropeCosSlot: Int
    /// Slot index for RoPE sin table.
    public let ropeSinSlot: Int
    /// All allocated RoPE table pairs. Single-family models have one pair.
    public let ropeTablePairs: [PlannedRoPETablePair]
    /// Slot index for dynamic tokenId.
    public let tokenIdSlot: Int
    /// Slot index for dynamic position.
    public let positionSlot: Int
    /// Slot index for batch token IDs (prefill only, -1 if no Metal prefill).
    public let tokenIdsBatchSlot: Int
    /// Optional FP32 FFN down-projection scratch for BF16-trained activations.
    public let ffnDownFp32Slot: Int
    /// Optional shared low-bit activation view for CAM projection banks.
    public let projectionActivationPlanesSlot: Int
    public let projectionActivationScalesSlot: Int

    /// Total bytes across all activation + state buffers (excludes weight buffer).
    public var totalActivationBytes: Int {
        slots.filter { $0.category != .weight }.reduce(0) { $0 + $1.sizeBytes }
    }

    /// Total slot count (highest index + 1).
    public var slotCount: Int {
        guard let last = slots.last else { return 0 }
        return last.index + 1
    }
}

/// One planned buffer slot.
public struct PlannedSlot: Sendable, Equatable {
    public let index: Int
    public let name: String
    public let sizeBytes: Int
    public let dtype: SmeltDType
    public let shape: [Int]
    public let category: SmeltBufferCategory

    public init(
        index: Int,
        name: String,
        sizeBytes: Int,
        dtype: SmeltDType,
        shape: [Int] = [],
        category: SmeltBufferCategory
    ) {
        self.index = index
        self.name = name
        self.sizeBytes = sizeBytes
        self.dtype = dtype
        self.shape = shape
        self.category = category
    }
}

// MARK: - Well-known fixed slot indices (shared activation buffers)

/// Fixed slot indices for activation buffers.
/// Slot numbers are stable across model variants. Indices 0-30 are reserved;
/// per-layer state starts at 31 via dynamic allocation.
/// Not all slots are allocated for every model — delta-specific and
/// attention-specific slots are only created when those layer types exist.
public enum SmeltFixedSlot: Int, CaseIterable, Sendable {
    // --- Shared (allocated for all models) ---
    case hiddenA = 0        // Double-buffered hidden state A
    case hiddenB = 1        // Double-buffered hidden state B
    case normOutBuf = 8     // RMS norm output
    case ffnGateBuf = 11    // FFN gate projection (SwiGLU)
    case ffnUpBuf = 12      // FFN up projection
    case ffnIntBuf = 13     // FFN intermediate
    case ffnDownBuf = 14    // FFN down projection
    case residualBuf = 15   // Residual accumulation
    case logitsBuf = 16     // Final logits [vocab_size]
    case argmaxBuf = 17     // Argmax result (8 bytes)

    // --- Prefill input (allocated only when Metal prefill enabled) ---
    case tokenIdsBatch = 26 // [maxPrefillBatch] Int32 token IDs for prefill
    case perLayerInputsBuf = 27   // Per-layer input tensor [layers * hidden_size_per_layer_input]

    // --- DeltaNet-specific (allocated only when delta layers present) ---
    case qkvBuf = 2         // QKV projection output
    case zBuf = 3           // Z (output) projection
    case aBuf = 4           // Decay parameter per head
    case bBuf = 5           // Beta parameter per head
    case betaBuf = 6        // Computed beta per head
    case gBuf = 7           // Gate per head
    case recOutBuf = 9      // Recurrence output
    case gatedOutBuf = 10   // Gated activation output
    case kvMemBuf = 18      // Key-value memory (FP32)
    case deltaBuf = 19      // Delta intermediate (FP32)

    // --- Attention-specific (allocated only when attention layers present) ---
    case attnQBuf = 20      // Q projections (includes gate when gated_q)
    case attnKBuf = 21      // K projections
    case attnVBuf = 22      // V projections
    case attnOutBuf = 23    // Attention output
    case attnGateBuf = 24   // Gate buffer after split
    case attnMaskBuf = 25   // Causal mask [current context capacity]

    // --- Scratch ---
    case normScaleScratch = 28  // RMS norm rsqrt scale scratch ([maxPrefillBatch] floats for prefill)
    case perLayerScratchBuf = 29 // Per-layer projection scratch [layers * hidden_size_per_layer_input]

    // --- Weight buffer ---
    case weights = 30       // Monolithic mmap'd weights.bin

    // --- EAGLE-class draft-model cluster embedder ---
    /// Holds the [num_centroids] fp16 centroid-projection logits
    /// produced before the sparse lm_head dispatch. Allocated only
    /// when `cfg.clusterEmbedder != nil`; non-draft packages and
    /// 26B/31B-style draft models (dense lm_head) leave this slot
    /// unregistered.
    case centroidLogitsBuf = 31

    // --- TurboQuant-H matvec ---
    /// Scratch buffer for `tqh_matvec_prepare_input` → `tqh_matvec`
    /// X_hat (per-group Hadamard transform of the matvec input).
    /// Sized to `ceil(maxCols/G) * G * fp32` where maxCols is the
    /// largest cols across every TQH-encoded matvec consumer
    /// enabled by the spec (lm_head, FFN down, FFN gate/up, attn
    /// q/k/v/o projs — see the allocation site in buildBufferPlan
    /// for the per-pattern cols map). Allocated only when at
    /// least one TQH matvec consumer pattern is allowlisted.
    ///
    /// Single slot shared across all TQH matvec dispatches in a
    /// forward pass — safe because Metal compute encoders run
    /// dispatches serially by default; prepare(N)→matvec(N) is
    /// guaranteed to retire before prepare(N+1) starts.
    /// `MTLDispatchType.concurrent` would invalidate that
    /// assumption.
    case tqhMatvecXHatBuf = 32
}

// MARK: - Minimum buffer size

/// Metal buffers should be at least 16 bytes (matches reference allocator behavior).
private let minBufferBytes = 16

/// Round up to minimum buffer size.
private func alignedSize(_ size: Int) -> Int {
    max(size, minBufferBytes)
}

private func ropeParams(
    for attn: SmeltAttentionConfig,
    layerType: SmeltLayerType,
    defaultRopeDim: Int,
    blockTopology: SmeltBlockTopology
) -> SmeltRoPEParams {
    let effectiveDim = attn.effectiveRopeDim(default: defaultRopeDim)
    let freqDim: Int?
    if layerType != .attention, effectiveDim < attn.headDim {
        freqDim = attn.headDim
    } else {
        freqDim = nil
    }
    return SmeltRoPEParams(
        theta: attn.ropeTheta,
        dim: effectiveDim,
        freqDim: freqDim,
        scaling: attn.ropeScaling,
        layout: attn.effectiveRoPETableLayout(blockTopology: blockTopology)
    )
}

// MARK: - Plan builder

/// Build the buffer plan from a validated IR.
/// This must be called AFTER validateSmeltIR succeeds.
public func buildBufferPlan(from ir: SmeltModelIR) -> SmeltBufferPlan {
    let cfg = ir.config
    let fp16 = 2  // bytes per FP16 element
    let fp32 = 4  // bytes per FP32 element
    let int32 = 4

    // Prefill batch multiplier: activation buffers are B× wider when
    // Metal prefill is enabled, so batched kernels can operate on
    // [B, dim] tensors. Decode still uses only the first [1, dim] slice.
    let metalPrefill = ir.prefill?.engine == "metal"
    let B = metalPrefill ? ir.prefill!.maxBatchSize : 1
    let dynamicContext = ir.usesDynamicContext
    let initialContextCapacity = dynamicContext ? 1 : ir.compiledSeqCapacity
    let contextShapeExtent = dynamicContext ? 0 : initialContextCapacity

    var slots: [PlannedSlot] = []

    // The trunk's activation/state ABI (W0, docs/talker-trunk-fit-audit.md):
    // fp16 is the LLM default; fp32 is the talker-class trunk. Slots with an
    // explicit dtype (int32 scratch, fp32 numerics buffers) keep it.
    let actDtype = cfg.activationDtype
    let actBytes = actDtype == .fp32 ? fp32 : fp16

    // Helper: append a slot at a fixed index. `dtype: nil` follows the
    // trunk's activation ABI.
    func fixedSlot(
        _ slot: SmeltFixedSlot, _ name: String, elements: Int,
        bytes: Int = 0, dtype: SmeltDType? = nil,
        shape: [Int] = [],
        category: SmeltBufferCategory = .activation
    ) {
        let resolved = dtype ?? actDtype
        let rawSize = bytes > 0 ? bytes : elements * (resolved == .fp32 ? fp32 : fp16)
        slots.append(PlannedSlot(
            index: slot.rawValue, name: name, sizeBytes: alignedSize(rawSize),
            dtype: resolved, shape: shape, category: category
        ))
    }

    let numDelta = ir.numDeltaLayers
    let numAttn = ir.numAttnLayers
    let attentionConfigs = ir.layerPattern.expanded.compactMap { cfg.attentionConfig(for: $0) }
    let maxAttnQProjDim = attentionConfigs.map(\.qProjDim).max() ?? 0
    let maxAttnKProjDim = attentionConfigs.map(\.kProjDim).max() ?? 0
    let maxAttnVProjDim = attentionConfigs.map(\.vProjDim).max() ?? 0
    let maxAttnOutDim = attentionConfigs.map { $0.qHeads * $0.headDim }.max() ?? 0
    // --- Fixed activation buffers (0-25) ---
    // Activation buffers are widened by B for Metal prefill.
    // logitsBuf is NOT widened by default — the LM head uses
    // fused_lut_matvec on only the last token's hidden state.
    // Multi-position verify targets opt into a [B, vocab] logitsBuf
    // via prefill.emit_all_logits=true so stochastic verify can
    // read K+1 logit rows from one chunked-prefill pass. The
    // greedy affine verify-argmax sibling table reuses logitsBuf as its
    // partial-key scratch and writes final Int32 tokens into argmaxBuf.
    // Signed LM heads currently materialize their active fp16 rows before the
    // shared argmax reducer, so signed verify packages reserve [B, vocab].
    // The affine argmax-only scratch needs B * (vocab / ROWS_PER_TG) uint2
    // entries, which is half the bytes of a full [B, vocab] fp16 slab.
    // argmaxBuf keeps the 8-byte decode ABI floor.
    // attnMaskBuf is NOT widened — it is sized to the active context capacity.
    let emitsAllLogits = ir.prefill?.emitAllLogits == true
    let emitsVerifyArgmax = ir.prefill?.verifyArgmax == true
    // Transactional recurrent verification is a generic multi-token execution
    // capability. The module selects its bounded memory/performance cell even
    // when ordinary prompt prefill advertises hundreds of rows. Row zero is
    // the state on entry and row n is the successor after n verified tokens.
    let recurrentVerifyHistoryRows = emitsVerifyArgmax && numDelta > 0
        ? min(B, max(0, ir.prefill?.verifyTokenCapacity ?? 0)) + 1
        : 0
    let signedVerifyArgmax = emitsVerifyArgmax
        && (ir.quantization.strategy == .binary1
            || ir.quantization.strategy == .ternary2)
    let singleLogitsBytes = cfg.vocabSize * fp16
    let fullBatchLogitsBytes = (emitsAllLogits || signedVerifyArgmax)
        ? B * singleLogitsBytes
        : 0
    let verifyArgmaxPartialBytes = emitsVerifyArgmax
        ? B * ((cfg.vocabSize + 7) / 8) * 8
        : 0
    let decodeArgmaxPartialBytes = ((cfg.vocabSize + 2_047) / 2_048) * 8
    let logitsBytes = max(
        singleLogitsBytes,
        fullBatchLogitsBytes,
        verifyArgmaxPartialBytes
    )

    // The same activation slots also carry runtime feature-fusion input and
    // optional projected output. Size from the generic graph primitive so
    // EAGLE, MTP, and future composers follow one path.
    let fusion = cfg.resolvedInputFusion
    let hiddenAElements = max(cfg.hiddenSize, fusion?.postProjectionWidth ?? 0)
    let hiddenBElements = max(
        cfg.hiddenSize,
        max(fusion?.concatenatedWidth ?? 0, fusion?.postProjectionWidth ?? 0)
    )

    // Shared activation buffers (always allocated)
    fixedSlot(.hiddenA, "hiddenA", elements: hiddenAElements * B)
    fixedSlot(.hiddenB, "hiddenB", elements: hiddenBElements * B)
    fixedSlot(.normOutBuf, "normOutBuf", elements: cfg.hiddenSize * B)
    fixedSlot(.ffnGateBuf, "ffnGateBuf", elements: cfg.maxFFNDim * B)
    fixedSlot(.ffnUpBuf, "ffnUpBuf", elements: cfg.maxFFNDim * B)
    fixedSlot(.ffnIntBuf, "ffnIntBuf", elements: cfg.maxFFNDim * B)
    fixedSlot(.ffnDownBuf, "ffnDownBuf", elements: cfg.hiddenSize * B)
    fixedSlot(.residualBuf, "residualBuf", elements: cfg.hiddenSize * B)
    fixedSlot(.logitsBuf, "logitsBuf", elements: (logitsBytes + fp16 - 1) / fp16)
    let argmaxEntries = (emitsAllLogits || emitsVerifyArgmax) ? B : 1
    let argmaxBytes = max(8, argmaxEntries * int32)
    fixedSlot(.argmaxBuf, "argmaxBuf", elements: 0, bytes: argmaxBytes, dtype: .raw)
    fixedSlot(
        .normScaleScratch,
        "normScaleScratch",
        elements: 0,
        bytes: max(B * fp32, decodeArgmaxPartialBytes),
        dtype: .raw
    )
    if cfg.hiddenSizePerLayerInput > 0 {
        let perLayerTotalDim = cfg.numLayers * cfg.hiddenSizePerLayerInput
        fixedSlot(.perLayerInputsBuf, "perLayerInputsBuf", elements: perLayerTotalDim * B)
        fixedSlot(.perLayerScratchBuf, "perLayerScratchBuf", elements: perLayerTotalDim * B)
    }

    // EAGLE-class draft-model cluster-embedder intermediate. Sized to
    // `num_centroids` fp16 elements (4 KiB at the canonical
    // 2048-centroid E2B/E4B draft-model shape). Holds the per-step
    // centroid scores produced before the topk + sparse lm_head
    // dispatch consumes them.
    if let cluster = cfg.clusterEmbedder {
        // cluster_sparse_lm_head is half-in/half-out — pinned regardless of
        // the trunk ABI (validateSmeltIR forbids fp32+cluster anyway).
        fixedSlot(.centroidLogitsBuf, "centroidLogitsBuf",
                  elements: cluster.numCentroids, dtype: .fp16)
    }

    // tqhMatvecXHatBuf needs to fit X_hat for the WIDEST cols of
    // any TQH-encoded matvec consumer enabled by the spec. Walk
    // each TQH pattern, map to its consumer's matvec cols, take
    // the max. Patterns that don't reach a matvec consumer
    // (e.g. embed_tokens_per_layer, which uses tqh_embedding_gather
    // and a different scratch path) contribute 0.
    let g = 128
    var tqhMatvecMaxCols = 0
    for pattern in ir.quantization.turboQuantHPatterns {
        if pattern == SmeltCanonicalTensorNames.embedTokens {
            // tied lm_head matvec: cols == hiddenSize
            tqhMatvecMaxCols = max(tqhMatvecMaxCols, cfg.hiddenSize)
        } else if pattern.hasSuffix("down_proj_weight") {
            // FFN down: W is [hidden, ffnDim], matvec cols == ffnDim
            tqhMatvecMaxCols = max(tqhMatvecMaxCols, cfg.maxFFNDim)
        } else if pattern.hasSuffix("o_proj_weight") {
            // o_proj: W is [hidden, qHeads*headDim], matvec cols ==
            // qHeads*headDim, which can EXCEED hiddenSize on models
            // with more attention than width (e.g. Qwen 3.5 0.8B
            // has hidden=1024, qHeads*headDim=2048).
            tqhMatvecMaxCols = max(tqhMatvecMaxCols, maxAttnOutDim)
        } else if pattern.hasSuffix("proj_weight") {
            // gate/up/q/k/v proj: input dim == hiddenSize.
            tqhMatvecMaxCols = max(tqhMatvecMaxCols, cfg.hiddenSize)
        }
    }
    if tqhMatvecMaxCols > 0 {
        let numGroups = SmeltTurboQuantHCodec.numGroups(
            cols: tqhMatvecMaxCols, groupSize: g
        )
        // Metal-prefill builds use the batched TQH matvec
        // (tqh_matvec_batched, Unit 55). Each of the B prefill
        // positions needs its own X_hat slice, so total scratch is
        // B * numGroups * G fp32. For decode-only builds, B=1.
        fixedSlot(
            .tqhMatvecXHatBuf, "tqhMatvecXHatBuf",
            elements: B * numGroups * g, dtype: .fp32
        )
    }

    // DeltaNet-specific activation buffers (only if delta layers exist)
    if let delta = cfg.delta, numDelta > 0 {
        let valueDim = delta.zDim
        fixedSlot(.qkvBuf, "qkvBuf", elements: delta.qkvDim * B)
        fixedSlot(.zBuf, "zBuf", elements: valueDim * B)
        fixedSlot(.aBuf, "aBuf", elements: delta.numHeads * B)
        fixedSlot(.bBuf, "bBuf", elements: delta.numHeads * B)
        fixedSlot(.betaBuf, "betaBuf", elements: delta.numHeads * B)
        fixedSlot(.gBuf, "gBuf", elements: delta.numHeads * B)
        fixedSlot(.recOutBuf, "recOutBuf", elements: valueDim * B)
        fixedSlot(.gatedOutBuf, "gatedOutBuf", elements: valueDim * B)
        // kvMem and delta are per-position intermediates (not batched)
        fixedSlot(.kvMemBuf, "kvMemBuf", elements: delta.numHeads * delta.headDim, dtype: .fp32)
        fixedSlot(.deltaBuf, "deltaBuf", elements: delta.numHeads * delta.headDim, dtype: .fp32)
    }

    // Attention-specific activation buffers (only if attention layers exist)
    if numAttn > 0 {
        fixedSlot(.attnQBuf, "attnQBuf", elements: maxAttnQProjDim * B)
        fixedSlot(.attnKBuf, "attnKBuf", elements: maxAttnKProjDim * B)
        fixedSlot(.attnVBuf, "attnVBuf", elements: maxAttnVProjDim * B)
        fixedSlot(.attnOutBuf, "attnOutBuf", elements: maxAttnOutDim * B)
        fixedSlot(.attnGateBuf, "attnGateBuf", elements: maxAttnOutDim * B)
        fixedSlot(
            .attnMaskBuf,
            "attnMaskBuf",
            elements: initialContextCapacity,
            shape: [contextShapeExtent],
            category: .dynamic
        )
    }

    // Prefill batch token IDs (only when Metal prefill is enabled)
    if metalPrefill {
        fixedSlot(
            .tokenIdsBatch, "tokenIdsBatch", elements: 0,
            bytes: B * int32, dtype: .int32, category: .dynamic
        )
    }

    // --- Monolithic weight buffer (slot 30) ---
    // Size is 0 here — the weight packer determines the real size via
    // SmeltWeightManifest.totalBytes. The runtime treats this slot specially:
    // it's the mmap'd weights.bin buffer, NOT a pre-allocated MTLBuffer.
    // We bypass alignedSize() to keep it at 0 (signals "externally sized").
    slots.append(PlannedSlot(
        index: SmeltFixedSlot.weights.rawValue, name: "weights",
        sizeBytes: 0, dtype: .raw, shape: [], category: .weight
    ))

    // --- Dynamic per-layer state ---

    // Dynamic slots start above the highest fixed-slot rawValue, which
    // is now `tqhMatvecXHatBuf` (32). Like `centroidLogitsBuf`, the
    // rawValue advances the dynamic floor unconditionally so slot
    // indices stay stable whether or not the TQH matvec slot is
    // *registered* (registration is conditional on
    // `turboQuantHPatterns` containing a matvec consumer).
    var nextSlot = SmeltFixedSlot.tqhMatvecXHatBuf.rawValue + 1  // 33

    var projectionActivationPlanesSlot = -1
    var projectionActivationScalesSlot = -1
    if let activationBits = cfg.projectionBanks.compactMap({
        $0.activationView?.bitCount
    }).max() {
        let maxProjectionActivationWidth = max(cfg.hiddenSize, cfg.maxFFNDim)
        let groups = (maxProjectionActivationWidth + 127) / 128
        projectionActivationPlanesSlot = nextSlot
        slots.append(PlannedSlot(
            index: nextSlot,
            name: "projectionActivationPlanes",
            sizeBytes: alignedSize(B * groups * activationBits * 4 * int32),
            dtype: .raw,
            shape: metalPrefill
                ? [B, groups, activationBits, 4]
                : [groups, activationBits, 4],
            category: .activation
        ))
        nextSlot += 1
        projectionActivationScalesSlot = nextSlot
        slots.append(PlannedSlot(
            index: nextSlot,
            name: "projectionActivationScales",
            sizeBytes: alignedSize(B * groups * fp16),
            dtype: .fp16,
            shape: metalPrefill ? [B, groups] : [groups],
            category: .activation
        ))
        nextSlot += 1
    }

    // Conv states: one per DeltaNet layer (only if delta layers exist)
    let convStateBase = nextSlot
    if let delta = cfg.delta, numDelta > 0 {
        let convStateBytes = delta.qkvDim * delta.convKernel * fp16
        for layerIdx in 0..<numDelta {
            slots.append(PlannedSlot(
                index: nextSlot, name: "convState_\(layerIdx)",
                sizeBytes: alignedSize(convStateBytes), dtype: .fp16,
                shape: [delta.qkvDim, delta.convKernel],
                category: .state
            ))
            nextSlot += 1
        }
    }

    // Recurrence states: one per DeltaNet layer
    let recStateBase = nextSlot
    if let delta = cfg.delta, numDelta > 0 {
        let recStateBytes = delta.numHeads * delta.headDim * delta.headDim * fp16
        for layerIdx in 0..<numDelta {
            slots.append(PlannedSlot(
                index: nextSlot, name: "recState_\(layerIdx)",
                sizeBytes: alignedSize(recStateBytes), dtype: .fp16,
                shape: [delta.numHeads, delta.headDim, delta.headDim],
                category: .state
            ))
            nextSlot += 1
        }
    }

    // Verify-only recurrent transaction storage. Ordinary prompt prefill and
    // decode never bind these stable, model-independent history slot names.
    if let delta = cfg.delta, recurrentVerifyHistoryRows > 0 {
        let convStateBytes = delta.qkvDim * delta.convKernel * fp16
        for layerIdx in 0..<numDelta {
            slots.append(PlannedSlot(
                index: nextSlot,
                name: "convStateHistory_\(layerIdx)",
                sizeBytes: alignedSize(
                    recurrentVerifyHistoryRows * convStateBytes
                ),
                dtype: .fp16,
                shape: [
                    recurrentVerifyHistoryRows,
                    delta.qkvDim,
                    delta.convKernel,
                ],
                category: .state
            ))
            nextSlot += 1
        }

        let recStateBytes = delta.numHeads * delta.headDim * delta.headDim * fp16
        for layerIdx in 0..<numDelta {
            slots.append(PlannedSlot(
                index: nextSlot,
                name: "recStateHistory_\(layerIdx)",
                sizeBytes: alignedSize(
                    recurrentVerifyHistoryRows * recStateBytes
                ),
                dtype: .fp16,
                shape: [
                    recurrentVerifyHistoryRows,
                    delta.numHeads,
                    delta.headDim,
                    delta.headDim,
                ],
                category: .state
            ))
            nextSlot += 1
        }
    }

    // Key caches: one per attention layer (only if attention layers exist)
    let keyCacheBase = nextSlot
    if numAttn > 0 {
        for (layerIdx, attn) in attentionConfigs.enumerated() {
            let keyCacheElements = attn.kvHeads * initialContextCapacity * attn.headDim
            let keyCacheBytes = keyCacheElements * actBytes
            // Port topology selects cache layout; activation dtype only sizes
            // and types the storage inside that layout.
            let kvShape = cfg.portTopology == .embeddingsInHiddenOut
                ? [contextShapeExtent, attn.kvHeads * attn.headDim]
                : [attn.kvHeads, contextShapeExtent, attn.headDim]
            slots.append(PlannedSlot(
                index: nextSlot, name: "keyCache_\(layerIdx)",
                sizeBytes: alignedSize(keyCacheBytes), dtype: actDtype,
                shape: kvShape,
                category: .state
            ))
            nextSlot += 1
        }
    }

    // Value caches: one per attention layer
    let valCacheBase = nextSlot
    if numAttn > 0 {
        for (layerIdx, attn) in attentionConfigs.enumerated() {
            let valCacheElements = attn.kvHeads * initialContextCapacity * attn.headDim
            let valCacheBytes = valCacheElements * actBytes
            let kvShape = actDtype != .fp16
                ? [contextShapeExtent, attn.kvHeads * attn.headDim]
                : [attn.kvHeads, contextShapeExtent, attn.headDim]
            slots.append(PlannedSlot(
                index: nextSlot, name: "valCache_\(layerIdx)",
                sizeBytes: alignedSize(valCacheBytes), dtype: actDtype,
                shape: kvShape,
                category: .state
            ))
            nextSlot += 1
        }
    }

    // --- RoPE tables ---
    var ropeTableParams: [SmeltRoPEParams] = []
    if attentionConfigs.isEmpty {
        ropeTableParams = [SmeltRoPEParams(theta: 10_000, dim: cfg.ropeDim)]
    } else {
        var seen = Set<SmeltRoPEParams>()
        for layerType in ir.layerPattern.expanded where layerType.isAttentionFamily {
            guard let attn = cfg.attentionConfig(for: layerType) else { continue }
            let params = ropeParams(
                for: attn,
                layerType: layerType,
                defaultRopeDim: cfg.ropeDim,
                blockTopology: cfg.blockTopology
            )
            if seen.insert(params).inserted {
                ropeTableParams.append(params)
            }
        }
    }

    var ropePairs: [PlannedRoPETablePair] = []
    ropePairs.reserveCapacity(ropeTableParams.count)
    for (pairIndex, params) in ropeTableParams.enumerated() {
        // f32 RoPE kernels take `const float*` cos/sin — the tables follow
        // the trunk ABI (W0 review finding).
        let ropeBytes = initialContextCapacity * params.dim * actBytes
        let cosSlot = nextSlot
        slots.append(PlannedSlot(
            index: cosSlot,
            name: pairIndex == 0 ? "ropeCos" : "ropeCos_\(pairIndex)",
            sizeBytes: alignedSize(ropeBytes),
            dtype: actDtype,
            shape: [contextShapeExtent, params.dim],
            category: .table
        ))
        nextSlot += 1

        let sinSlot = nextSlot
        slots.append(PlannedSlot(
            index: sinSlot,
            name: pairIndex == 0 ? "ropeSin" : "ropeSin_\(pairIndex)",
            sizeBytes: alignedSize(ropeBytes),
            dtype: actDtype,
            shape: [contextShapeExtent, params.dim],
            category: .table
        ))
        nextSlot += 1

        ropePairs.append(PlannedRoPETablePair(
            params: params,
            cosSlot: cosSlot,
            sinSlot: sinSlot
        ))
    }

    // MLX's D=256 vector attention changes from one-pass to a B128 two-pass
    // reduction once the key sequence reaches 1024 on supported Apple GPUs.
    // Allocate reusable topology scratch from the attention capabilities, not
    // from any model-family identity. Every attention layer serially reuses
    // these two slots.
    if actDtype == .fp16,
       let maxMlxD256QHeads = (
           attentionConfigs
            .filter {
                $0.headDim == 256
                    && $0.kvHeads > 0
                    && $0.qHeads.isMultiple(of: $0.kvHeads)
            }
            .map(\.qHeads)
            .max()
       )
    {
        let blockCount = 128
        slots.append(PlannedSlot(
            index: nextSlot,
            name: "mlxAttentionPartialsD256B128",
            sizeBytes: alignedSize(
                maxMlxD256QHeads * blockCount * 256 * fp16
            ),
            dtype: .fp16,
            shape: [maxMlxD256QHeads, blockCount, 256],
            category: .activation
        ))
        nextSlot += 1
        slots.append(PlannedSlot(
            index: nextSlot,
            name: "mlxAttentionStatsD256B128",
            sizeBytes: alignedSize(
                2 * maxMlxD256QHeads * blockCount * fp32
            ),
            dtype: .fp32,
            shape: [2, maxMlxD256QHeads, blockCount],
            category: .activation
        ))
        nextSlot += 1
    }

    let ropeCosIdx = ropePairs.first?.cosSlot ?? nextSlot
    let ropeSinIdx = ropePairs.first?.sinSlot ?? nextSlot

    let ffnDownFp32Idx = -1

    // --- Dynamic scalar slots ---
    let tokenIdIdx = nextSlot
    slots.append(PlannedSlot(
        index: nextSlot, name: "dynamicTokenId",
        sizeBytes: alignedSize(int32), dtype: .int32, category: .dynamic
    ))
    nextSlot += 1

    let positionIdx = nextSlot
    slots.append(PlannedSlot(
        index: nextSlot, name: "dynamicPosition",
        sizeBytes: alignedSize(int32), dtype: .int32, category: .dynamic
    ))

    // Sort by index — fixed slots may be non-contiguous with optional configs
    slots.sort { $0.index < $1.index }

    return SmeltBufferPlan(
        slots: slots,
        convStateBaseSlot: convStateBase,
        recStateBaseSlot: recStateBase,
        keyCacheBaseSlot: keyCacheBase,
        valCacheBaseSlot: valCacheBase,
        ropeCosSlot: ropeCosIdx,
        ropeSinSlot: ropeSinIdx,
        ropeTablePairs: ropePairs,
        tokenIdSlot: tokenIdIdx,
        positionSlot: positionIdx,
        tokenIdsBatchSlot: metalPrefill ? SmeltFixedSlot.tokenIdsBatch.rawValue : -1,
        ffnDownFp32Slot: ffnDownFp32Idx,
        projectionActivationPlanesSlot: projectionActivationPlanesSlot,
        projectionActivationScalesSlot: projectionActivationScalesSlot
    )
}

// MARK: - Conversion to manifest buffer table

extension SmeltBufferPlan {
    /// Convert to the manifest's slot layout for JSON serialization.
    public func toSlotLayout() -> SmeltSlotLayout {
        SmeltSlotLayout(
            convStateBaseSlot: convStateBaseSlot,
            recStateBaseSlot: recStateBaseSlot,
            keyCacheBaseSlot: keyCacheBaseSlot,
            valCacheBaseSlot: valCacheBaseSlot,
            ropeCosSlot: ropeCosSlot,
            ropeSinSlot: ropeSinSlot,
            ropeTablePairs: ropeTablePairs.map {
                SmeltRoPETablePairManifest(
                    theta: $0.params.theta,
                    dim: $0.params.dim,
                    freqDim: $0.params.freqDim,
                    scaling: $0.params.scaling,
                    layout: $0.params.layout,
                    cosSlot: $0.cosSlot,
                    sinSlot: $0.sinSlot
                )
            },
            tokenIdSlot: tokenIdSlot,
            positionSlot: positionSlot,
            weightsSlot: SmeltFixedSlot.weights.rawValue
        )
    }

    /// Convert to the manifest's buffer table format for JSON serialization.
    public func toBufferTable() -> SmeltBufferTable {
        SmeltBufferTable(slots: slots.map { planned in
            SmeltBufferSlot(
                index: planned.index,
                name: planned.name,
                sizeBytes: planned.sizeBytes,
                dtype: planned.dtype,
                shape: planned.shape,
                category: planned.category
            )
        })
    }
}
