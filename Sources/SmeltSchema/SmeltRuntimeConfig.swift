// SmeltRuntimeConfig — Minimal runtime-only configuration.
//
// Extracted from SmeltManifest at load time, then the manifest is dropped.
// This struct contains ONLY what the decode loop needs: integer slot indices,
// buffer sizes as a flat array, pipeline count, and bounds for dynamic inputs.
//
// No strings. No arrays of metadata. No Foundation types.
// This is the boundary between "framework overhead" and "zero-cost runtime".

/// Minimal config for the decode hot path. All integers, no heap allocations.
/// Extracted from SmeltManifest once at load time.
public struct SmeltRuntimeConfig: Sendable {
    // --- Dynamic input bounds (checked once per decodeStep call) ---
    public let vocabSize: Int32
    /// Static package sequence allocation. Dynamic runtimes keep this at 0.
    public let staticSeqCapacity: Int32
    /// Active invocation context ceiling. Dynamic runtimes default this to the
    /// largest representable position unless the caller passes an explicit limit.
    public let contextLimit: Int32

    // --- Slot layout (integer indices for generated code) ---
    public let convStateBaseSlot: Int32
    public let recStateBaseSlot: Int32
    public let keyCacheBaseSlot: Int32
    public let valCacheBaseSlot: Int32
    public let ropeCosSlot: Int32
    public let ropeSinSlot: Int32
    public let tokenIdSlot: Int32
    public let positionSlot: Int32
    public let weightsSlot: Int32
    /// Slot index for argmax result buffer.
    public let argmaxSlot: Int32

    // --- Fixed slot constants (must match SmeltFixedSlot in SmeltCompiler) ---
    /// Argmax result buffer — fixed at slot 17.
    public static let fixedArgmaxSlot: Int32 = 17
    /// Logits buffer — fixed at slot 16.
    public static let fixedLogitsSlot: Int32 = 16
    /// Post-final-norm hidden state buffer — fixed at slot 8.
    /// Both the fused norm+matvec path and the cluster-embedder path
    /// write the post-norm hidden state here, so the draft-model taps
    /// surface can read it without knowing which lm_head variant the
    /// target uses.
    public static let fixedNormOutBufSlot: Int32 = 8

    // --- Counts ---
    public let pipelineCount: Int32
    public let bufferCount: Int32
    public let numDeltaLayers: Int32
    public let numAttnLayers: Int32

    // --- Prefill ---
    /// Maximum token chunk size for Metal prefill (0 if no Metal prefill).
    public let maxPrefillBatchSize: Int32
    /// Slot index for batch token IDs (-1 if no Metal prefill).
    public let tokenIdsBatchSlot: Int32

    public init(
        vocabSize: Int32,
        staticSeqCapacity: Int32,
        contextLimit: Int32,
        convStateBaseSlot: Int32,
        recStateBaseSlot: Int32,
        keyCacheBaseSlot: Int32,
        valCacheBaseSlot: Int32,
        ropeCosSlot: Int32,
        ropeSinSlot: Int32,
        tokenIdSlot: Int32,
        positionSlot: Int32,
        weightsSlot: Int32,
        argmaxSlot: Int32,
        pipelineCount: Int32,
        bufferCount: Int32,
        numDeltaLayers: Int32,
        numAttnLayers: Int32,
        maxPrefillBatchSize: Int32 = 0,
        tokenIdsBatchSlot: Int32 = -1
    ) {
        self.vocabSize = vocabSize
        self.staticSeqCapacity = staticSeqCapacity
        self.contextLimit = contextLimit
        self.convStateBaseSlot = convStateBaseSlot
        self.recStateBaseSlot = recStateBaseSlot
        self.keyCacheBaseSlot = keyCacheBaseSlot
        self.valCacheBaseSlot = valCacheBaseSlot
        self.ropeCosSlot = ropeCosSlot
        self.ropeSinSlot = ropeSinSlot
        self.tokenIdSlot = tokenIdSlot
        self.positionSlot = positionSlot
        self.weightsSlot = weightsSlot
        self.argmaxSlot = argmaxSlot
        self.pipelineCount = pipelineCount
        self.bufferCount = bufferCount
        self.numDeltaLayers = numDeltaLayers
        self.numAttnLayers = numAttnLayers
        self.maxPrefillBatchSize = maxPrefillBatchSize
        self.tokenIdsBatchSlot = tokenIdsBatchSlot
    }
}

// MARK: - Extract from manifest

extension SmeltRuntimeConfig {
    /// Extract the minimal runtime config from a full manifest.
    /// Call once at load time, then drop the manifest.
    public init(from manifest: SmeltManifest, contextLimit: Int? = nil) {
        let dynamicRequestBuffersEnabled = manifest.prefill?.engine != "coreml"
        let resolvedContextLimit =
            contextLimit
            ?? (dynamicRequestBuffersEnabled ? Int(Int32.max) : manifest.defaultContextLimit)
        self.init(
            vocabSize: Int32(manifest.config.vocabSize),
            staticSeqCapacity: Int32(manifest.config.staticSeqCapacity ?? 0),
            contextLimit: Int32(resolvedContextLimit),
            convStateBaseSlot: Int32(manifest.slotLayout.convStateBaseSlot),
            recStateBaseSlot: Int32(manifest.slotLayout.recStateBaseSlot),
            keyCacheBaseSlot: Int32(manifest.slotLayout.keyCacheBaseSlot),
            valCacheBaseSlot: Int32(manifest.slotLayout.valCacheBaseSlot),
            ropeCosSlot: Int32(manifest.slotLayout.ropeCosSlot),
            ropeSinSlot: Int32(manifest.slotLayout.ropeSinSlot),
            tokenIdSlot: Int32(manifest.slotLayout.tokenIdSlot),
            positionSlot: Int32(manifest.slotLayout.positionSlot),
            weightsSlot: Int32(manifest.slotLayout.weightsSlot),
            argmaxSlot: SmeltRuntimeConfig.fixedArgmaxSlot,
            pipelineCount: Int32(manifest.pipelines.count),
            bufferCount: Int32(
                (manifest.buffers.slots.map(\.index).max() ?? 0) + 1
            ),
            numDeltaLayers: Int32(manifest.config.numDeltaLayers),
            numAttnLayers: Int32(manifest.config.numAttnLayers),
            maxPrefillBatchSize: Int32(manifest.prefill?.maxBatchSize ?? 0),
            tokenIdsBatchSlot: {
                // Slot 26 if Metal prefill is configured
                if let prefill = manifest.prefill, prefill.engine == "metal" {
                    return 26
                }
                return -1
            }()
        )
    }
}
