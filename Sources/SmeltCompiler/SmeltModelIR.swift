// SmeltModelIR — Typed intermediate representation for a Smelt model.
//
// The IR is the compiler's internal data model: every dimension, every layer,
// every quantization setting represented as a concrete Swift type. The parser
// produces an IR, the validator rejects impossible configurations, and all
// downstream phases (weight packing, slot allocation, code emission) consume it.
//
// All types are value types — cheap to copy, no ARC overhead.

import SmeltSchema

// MARK: - Top-level IR

/// Complete typed description of a model, produced by constructing + validating an IR.
public struct SmeltModelIR: Sendable {
    public let modelName: String
    /// Exact remote source revision when authored by a module. Nil preserves
    /// the default `main` behavior for hand-built IR fixtures.
    public let modelRevision: String?
    public let config: SmeltConfig
    public let layerPattern: SmeltLayerPattern
    public let quantization: SmeltQuantizationConfig
    public let loading: SmeltLoadingConfig
    public let compilation: SmeltCompilationConfig
    public let runtime: SmeltRuntimePolicyConfig
    public let prefill: SmeltPrefillConfig?
    public let decode: SmeltDecodeConfig
    public let inference: SmeltInferenceConfig

    public init(
        modelName: String,
        modelRevision: String? = nil,
        config: SmeltConfig,
        layerPattern: SmeltLayerPattern,
        quantization: SmeltQuantizationConfig,
        loading: SmeltLoadingConfig,
        compilation: SmeltCompilationConfig = SmeltCompilationConfig(),
        runtime: SmeltRuntimePolicyConfig = SmeltRuntimePolicyConfig(),
        prefill: SmeltPrefillConfig? = nil,
        decode: SmeltDecodeConfig = SmeltDecodeConfig(),
        inference: SmeltInferenceConfig = SmeltInferenceConfig()
    ) {
        self.modelName = modelName
        self.modelRevision = modelRevision
        self.config = config
        self.layerPattern = layerPattern
        self.quantization = quantization
        self.loading = loading
        self.compilation = compilation
        self.runtime = runtime
        self.prefill = prefill
        self.decode = decode
        self.inference = inference
    }

    /// Total number of layers (must equal config.numLayers).
    public var totalLayers: Int { layerPattern.unit.count * layerPattern.repeats }

    /// Number of DeltaNet layers.
    public var numDeltaLayers: Int {
        layerPattern.unit.filter { $0 == .delta }.count * layerPattern.repeats
    }

    /// Number of attention layers.
    public var numAttnLayers: Int {
        layerPattern.unit.filter(\.isAttentionFamily).count * layerPattern.repeats
    }

    public var numSlidingLayers: Int {
        layerPattern.unit.filter { $0 == .sliding }.count * layerPattern.repeats
    }

    public var numGlobalLayers: Int {
        layerPattern.unit.filter { $0 == .global }.count * layerPattern.repeats
    }

    public var usesDynamicContext: Bool {
        prefill?.engine == "metal"
    }

    public var compiledSeqCapacity: Int {
        config.compiledSeqCapacity
    }

    public var firstKVSharedLayerIndex: Int? {
        guard config.sharedKVLayers > 0 else { return nil }
        return config.numLayers - config.sharedKVLayers
    }

    public func isKVSharedLayer(_ layerIndex: Int) -> Bool {
        kvSharedSourceGlobalLayerIndex(for: layerIndex) != nil
    }

    public func kvSharedSourceGlobalLayerIndex(for layerIndex: Int) -> Int? {
        guard let firstShared = firstKVSharedLayerIndex else { return nil }
        guard layerIndex >= firstShared, layerIndex < totalLayers else { return nil }

        let expanded = layerPattern.expanded
        let layerType = expanded[layerIndex]
        guard layerType.isAttentionFamily else { return nil }

        for idx in stride(from: firstShared - 1, through: 0, by: -1) {
            if expanded[idx] == layerType {
                return idx
            }
        }
        return nil
    }

    public func attentionLocalIndex(forGlobalLayerIndex layerIndex: Int) -> Int? {
        let expanded = layerPattern.expanded
        guard layerIndex >= 0, layerIndex < expanded.count else { return nil }
        guard expanded[layerIndex].isAttentionFamily else { return nil }
        return expanded[..<layerIndex].filter(\.isAttentionFamily).count
    }

    public func kvSharedSourceAttentionIndex(forGlobalLayerIndex layerIndex: Int) -> Int? {
        guard let sourceGlobal = kvSharedSourceGlobalLayerIndex(for: layerIndex) else { return nil }
        return attentionLocalIndex(forGlobalLayerIndex: sourceGlobal)
    }
}

// MARK: - Config

/// Runtime-supplied feature rows concatenated and projected into a transformer
/// stack. This graph primitive is shared by EAGLE-style and MTP auxiliaries.
public struct SmeltInputFusionConfig: Sendable, Equatable {
    public let sourceWidth: Int
    public let sourceCount: Int
    public let normalizeSources: Bool
    public let postProjectionWidth: Int?

    public init(
        sourceWidth: Int,
        sourceCount: Int = 2,
        normalizeSources: Bool = false,
        postProjectionWidth: Int? = nil
    ) {
        self.sourceWidth = sourceWidth
        self.sourceCount = sourceCount
        self.normalizeSources = normalizeSources
        self.postProjectionWidth = postProjectionWidth
    }

    public var concatenatedWidth: Int { sourceWidth * sourceCount }
}

public struct SmeltProjectionBankConfig: Sendable, Equatable {
    public let id: String
    public let source: SmeltCAMIR.ProjectionSource
    public let outputs: [SmeltCAMIR.ProjectionEndpoint]
    public let activationView: SmeltCAMIR.ProjectionActivationView?
    public let activationViewLayerSpans: [SmeltCAMIR.ActivationViewLayerSpan]?

    public init(
        id: String,
        source: SmeltCAMIR.ProjectionSource,
        outputs: [SmeltCAMIR.ProjectionEndpoint],
        activationView: SmeltCAMIR.ProjectionActivationView? = nil,
        activationViewLayerSpans: [SmeltCAMIR.ActivationViewLayerSpan]? = nil
    ) {
        self.id = id
        self.source = source
        self.outputs = outputs
        self.activationView = activationView
        self.activationViewLayerSpans = activationViewLayerSpans
    }

    public func usesActivationView(at layerIndex: Int) -> Bool {
        guard activationView != nil else { return false }
        guard let activationViewLayerSpans else { return true }
        return activationViewLayerSpans.contains { span in
            span.start..<(span.start + span.count) ~= layerIndex
        }
    }
}

/// The model's external port contract, independent of activation storage.
public enum SmeltPortTopology: Sendable, Equatable {
    /// Token IDs enter the package and vocabulary logits leave it.
    case tokenInLogitsOut

    /// Embeddings enter the package and normalized hidden states leave it.
    case embeddingsInHiddenOut
}

/// Global model configuration — all dimensions are compile-time constants.
public struct SmeltConfig: Sendable {
    public let hiddenSize: Int
    public let numLayers: Int
    public let vocabSize: Int
    public let vocabSizePerLayerInput: Int
    public let hiddenSizePerLayerInput: Int
    public let hiddenActivation: SmeltHiddenActivation?
    public let staticSeqCapacity: Int?
    public let ropeDim: Int
    public let rmsEps: Float
    public let normMode: SmeltNormMode
    /// Activation/state storage: FP16, BF16, or FP32. Drives operation-family
    /// selection and activation/KV-cache slot sizing, never graph topology.
    public let activationDtype: SmeltDType
    /// External port topology. This selects the graph independently of dtype.
    public let portTopology: SmeltPortTopology
    public let blockTopology: SmeltBlockTopology
    public let logitCap: Float?
    public let attnLogitCap: Float?
    public let sharedKVLayers: Int
    /// Optional FFN width used by the trailing shared-KV region.
    /// Some models use a wider MLP on the trailing shared-KV layers.
    public let sharedKVFFNDim: Int?
    /// DeltaNet config — nil if model has no DeltaNet layers.
    public let delta: SmeltDeltaConfig?
    /// Attention configs keyed by layer family.
    public let attentionConfigs: [SmeltLayerType: SmeltAttentionConfig]
    public let ffn: SmeltFFNConfig
    /// When true, LM head reuses embed_tokens weight (no separate weight entry).
    /// When false, a separate lm_head_weight is packed and may be quantized.
    public let tiedLMHead: Bool
    /// Hidden size of an external model that a projection package pairs
    /// with. Set on packages that ship pre/post projection weights so the
    /// compiler can size them:
    ///   - pre_projection: maps a `2 * backboneHiddenSize` input down to
    ///     `hiddenSize` (this package's own dim).
    ///   - post_projection: maps the last hidden state back up to
    ///     `backboneHiddenSize` so the next-step input can reuse it.
    /// Nil on regular packages — they have no projection layers.
    public let backboneHiddenSize: Int?

    /// Explicit model-agnostic form of the legacy backbone projection graph.
    public let inputFusion: SmeltInputFusionConfig?

    /// Common-input projection topology authored by the module. Backends may
    /// lower a bank as separate matvecs or as one packed projection brick.
    public let projectionBanks: [SmeltProjectionBankConfig]

    /// Cluster-embedder config for sparse `lm_head`. Set on packages that
    /// ship a top-k centroid classifier; nil on packages that use a full
    /// dense lm_head over the vocab.
    public let clusterEmbedder: SmeltClusterEmbedderConfig?

    public init(
        hiddenSize: Int,
        numLayers: Int,
        vocabSize: Int,
        vocabSizePerLayerInput: Int = 0,
        hiddenSizePerLayerInput: Int = 0,
        hiddenActivation: SmeltHiddenActivation? = nil,
        staticSeqCapacity: Int? = nil,
        ropeDim: Int,
        rmsEps: Float,
        normMode: SmeltNormMode = .weight,
        activationDtype: SmeltDType = .fp16,
        portTopology: SmeltPortTopology = .tokenInLogitsOut,
        blockTopology: SmeltBlockTopology = .standard,
        logitCap: Float? = nil,
        attnLogitCap: Float? = nil,
        sharedKVLayers: Int = 0,
        sharedKVFFNDim: Int? = nil,
        delta: SmeltDeltaConfig? = nil,
        attention: SmeltAttentionConfig? = nil,
        attentionConfigs: [SmeltLayerType: SmeltAttentionConfig] = [:],
        ffn: SmeltFFNConfig,
        tiedLMHead: Bool = true,
        backboneHiddenSize: Int? = nil,
        clusterEmbedder: SmeltClusterEmbedderConfig? = nil,
        inputFusion: SmeltInputFusionConfig? = nil,
        projectionBanks: [SmeltProjectionBankConfig] = []
    ) {
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.vocabSize = vocabSize
        self.vocabSizePerLayerInput = vocabSizePerLayerInput
        self.hiddenSizePerLayerInput = hiddenSizePerLayerInput
        self.hiddenActivation = hiddenActivation
        self.staticSeqCapacity = staticSeqCapacity
        self.ropeDim = ropeDim
        self.rmsEps = rmsEps
        self.normMode = normMode
        self.activationDtype = activationDtype
        self.portTopology = portTopology
        self.blockTopology = blockTopology
        self.logitCap = logitCap
        self.attnLogitCap = attnLogitCap
        self.sharedKVLayers = sharedKVLayers
        self.sharedKVFFNDim = sharedKVFFNDim
        self.delta = delta
        var mergedAttentionConfigs = attentionConfigs
        if let attention {
            mergedAttentionConfigs[.attention] = attention
        }
        self.attentionConfigs = mergedAttentionConfigs
        self.ffn = ffn
        self.tiedLMHead = tiedLMHead
        self.backboneHiddenSize = backboneHiddenSize
        self.clusterEmbedder = clusterEmbedder
        self.inputFusion = inputFusion
        self.projectionBanks = projectionBanks
    }

    public func projectionBank(
        source: SmeltCAMIR.ProjectionSource,
        containing endpoints: [SmeltCAMIR.ProjectionEndpoint]
    ) -> SmeltProjectionBankConfig? {
        projectionBanks.first { bank in
            bank.source == source && endpoints.allSatisfy(bank.outputs.contains)
        }
    }

    public var attention: SmeltAttentionConfig? {
        attentionConfigs[.attention]
    }

    public var resolvedInputFusion: SmeltInputFusionConfig? {
        if let inputFusion { return inputFusion }
        guard let backboneHiddenSize else { return nil }
        return SmeltInputFusionConfig(
            sourceWidth: backboneHiddenSize,
            sourceCount: 2,
            normalizeSources: false,
            postProjectionWidth: backboneHiddenSize
        )
    }

    /// Initial/static sequence allocation for packages that still require one.
    /// Dynamic packages deliberately compile without a context ceiling and start
    /// request-scoped buffers at one token.
    public var compiledSeqCapacity: Int {
        staticSeqCapacity ?? 1
    }

    public func attentionConfig(for layerType: SmeltLayerType) -> SmeltAttentionConfig? {
        guard layerType.isAttentionFamily else { return nil }
        return attentionConfigs[layerType]
    }

    public var maxFFNDim: Int {
        max(ffn.dim, sharedKVFFNDim ?? 0)
    }

    public func ffnDim(for layerIndex: Int) -> Int {
        guard let sharedKVFFNDim, sharedKVLayers > 0 else { return ffn.dim }
        let firstShared = numLayers - sharedKVLayers
        guard layerIndex >= firstShared else { return ffn.dim }
        return sharedKVFFNDim
    }
}

/// DeltaNet-specific dimensions.
public struct SmeltDeltaConfig: Sendable {
    public let numHeads: Int
    public let headDim: Int
    public let convKernel: Int
    public let qkvDim: Int         // 2 * qkDim + valueDim
    public let zDim: Int           // valueDim (gated RMS / out-proj input)
    public let aDim: Int           // valueHeads (decay)
    public let bDim: Int           // valueHeads (beta)

    public init(
        numHeads: Int,
        headDim: Int,
        convKernel: Int,
        qkvDim: Int,
        zDim: Int,
        aDim: Int,
        bDim: Int
    ) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.convKernel = convKernel
        self.qkvDim = qkvDim
        self.zDim = zDim
        self.aDim = aDim
        self.bDim = bDim
    }

    /// DeltaNet recurrent/value head count.
    public var valueHeads: Int { numHeads }

    /// DeltaNet recurrent/value width: valueHeads * headDim.
    public var valueDim: Int { numHeads * headDim }

    /// Compact Q/K width inside qkvDim.
    public var qkDim: Int { qkvDim - zDim }

    /// Number of compact Q/K heads encoded in qkvDim.
    /// Validation guarantees this is positive and integral.
    public var qkHeads: Int { qkDim / (2 * headDim) }

    /// How many value heads share one compact Q/K head.
    /// Validation guarantees this is integral.
    public var qkRepeatFactor: Int { valueHeads / qkHeads }

    /// 1/sqrt(headDim), precomputed for dispatch constants.
    public var headScale: Float { 1.0 / Float(headDim).squareRoot() }
}

/// Full attention layer dimensions.
public struct SmeltAttentionConfig: Sendable {
    public let qHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let gatedQ: Bool
    /// Whether q_proj/k_proj/v_proj carry additive projection bias tensors.
    public let qkvBias: Bool
    /// Whether this attention layer has per-head Q/K normalization weights.
    /// Qwen uses these; Llama does not.
    public let qkNorm: Bool
    /// Per-head Q/K norm weight semantics.
    /// Some models use one_plus_weight; others use weight.
    public let qkNormMode: SmeltNormMode
    /// Whether this attention layer applies scale-less per-head RMS norm to V.
    /// Some models apply this; others do not.
    public let vNorm: Bool
    /// Attention score scale requested by the model family.
    /// Standard decoder kernels use 1/sqrt(head_dim) * attnScale.
    /// Some models use attnScale directly.
    public let attnScale: Float
    public let ropeTheta: Float
    public let ropeDim: Int?
    public let ropeLayout: SmeltRoPELayout?
    public let ropeScaling: SmeltRoPEScaling?
    public let slidingWindow: Int
    /// When true, this attention layer takes its K and V from an
    /// **external** source (a separately-loaded package's last-layer
    /// cache) rather than projecting its own from the layer's hidden
    /// state — every layer cross-attends over the external last-layer
    /// K/V instead of running its own k_proj / v_proj.
    /// Downstream consequences (no own k_proj/v_proj/k_norm/v_norm
    /// weights, cross-attention kernel dispatch, frozen position_ids)
    /// land in subsequent units; this field is the parser-level flag
    /// that gates them.
    public let externalKV: Bool

    public init(
        qHeads: Int,
        kvHeads: Int,
        headDim: Int,
        gatedQ: Bool,
        qkvBias: Bool = false,
        qkNorm: Bool = false,
        qkNormMode: SmeltNormMode = .onePlusWeight,
        vNorm: Bool = false,
        attnScale: Float = 1.0,
        ropeTheta: Float = 10_000,
        ropeDim: Int? = nil,
        ropeLayout: SmeltRoPELayout? = nil,
        ropeScaling: SmeltRoPEScaling? = nil,
        slidingWindow: Int = 0,
        externalKV: Bool = false
    ) {
        self.qHeads = qHeads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.gatedQ = gatedQ
        self.qkvBias = qkvBias
        self.qkNorm = qkNorm
        self.qkNormMode = qkNormMode
        self.vNorm = vNorm
        self.attnScale = attnScale
        self.ropeTheta = ropeTheta
        self.ropeDim = ropeDim
        self.ropeLayout = ropeLayout
        self.ropeScaling = ropeScaling
        self.slidingWindow = slidingWindow
        self.externalKV = externalKV
    }

    /// GQA ratio: qHeads / kvHeads.
    public var gqaRatio: Int { qHeads / kvHeads }

    /// 1/sqrt(headDim), precomputed for dispatch constants.
    public var headScale: Float { 1.0 / Float(headDim).squareRoot() }

    public func effectiveScoreScale(blockTopology: SmeltBlockTopology) -> Float {
        headScale * attnScale
    }

    /// Q projection output dim: qHeads * headDim * (gatedQ ? 2 : 1).
    public var qProjDim: Int { qHeads * headDim * (gatedQ ? 2 : 1) }

    /// K projection output dim: kvHeads * headDim.
    public var kProjDim: Int { kvHeads * headDim }

    /// V projection output dim: kvHeads * headDim.
    public var vProjDim: Int { kvHeads * headDim }

    public func effectiveRopeDim(default defaultRopeDim: Int) -> Int {
        ropeDim ?? defaultRopeDim
    }

    public func effectiveRoPETableLayout(blockTopology: SmeltBlockTopology) -> String {
        if let ropeLayout {
            return ropeLayout.rawValue
        }
        return "interleaved"
    }

    public func effectiveRoPELayoutConstant(
        blockTopology: SmeltBlockTopology,
        ropeDim: Int
    ) -> Int {
        if let ropeLayout {
            switch ropeLayout {
            case .interleaved:
                return 0
            case .splitHalf:
                return 1
            }
        }
        return 0
    }
}

/// FFN dimensions.
public struct SmeltFFNConfig: Sendable {
    public let dim: Int
    public let activation: SmeltActivation

    public init(dim: Int, activation: SmeltActivation) {
        self.dim = dim
        self.activation = activation
    }
}

/// Cluster-embedder config for sparse `lm_head`.
///
/// Replaces the dense `lm_head` matvec over the full vocab with a top-k
/// centroid gather:
///   1. Project the last hidden state to `numCentroids` centroid logits.
///   2. Pick the top-k centroids.
///   3. Gather the `(vocabSize / numCentroids)` tokens belonging to
///      each picked centroid (`token_ordering` permutes the vocab into
///      contiguous clusters of that size).
///   4. Dot-product those `topK * (vocabSize / numCentroids)` lm_head
///      rows with the hidden state.
///   5. Scatter into a sparse logits tensor with `-inf` elsewhere.
///
/// Example: `numCentroids=2048, topK=32` yields ~64× fewer dot-products
/// than the dense lm_head.
///
/// Constraint: `vocabSize` must be divisible by `numCentroids` so
/// that every cluster has the same number of tokens. Validated by
/// `validateSmeltIR`.
public struct SmeltClusterEmbedderConfig: Sendable, Equatable {
    /// Total number of centroids.
    public let numCentroids: Int
    /// Number of centroids selected per step.
    public let topK: Int

    public init(numCentroids: Int, topK: Int) {
        self.numCentroids = numCentroids
        self.topK = topK
    }

    /// Tokens-per-cluster, derived from a vocab-size context.
    public func tokensPerCluster(vocabSize: Int) -> Int {
        vocabSize / numCentroids
    }
}

// MARK: - Enums

/// Supported FFN activations.
public enum SmeltActivation: String, Sendable {
    case swiglu
    case geglu
}

public enum SmeltHiddenActivation: String, Sendable {
    case geluPytorchTanh = "gelu_pytorch_tanh"
}

public enum SmeltNormMode: String, Sendable {
    case weight
    case onePlusWeight = "one_plus_weight"
}

public enum SmeltRoPELayout: String, Sendable {
    case interleaved
    case splitHalf = "split_half"
}

public enum SmeltBlockTopology: String, Sendable {
    case standard
}

/// Layer type in the pattern.
public enum SmeltLayerType: String, Sendable {
    case delta
    case attention = "attn"
    case sliding
    case global

    public var isAttentionFamily: Bool {
        switch self {
        case .attention, .sliding, .global:
            return true
        case .delta:
            return false
        }
    }
}

/// Quantization strategies.
public enum SmeltQuantStrategy: String, Sendable {
    case lutU4 = "lut_u4"
    case affineU4 = "affine_u4"
    case binary1
    case ternary2
    case fp16
    /// TurboQuant-H 2.125-bpw codebook quantization. Currently
    /// activated per-tensor via SmeltQuantizationConfig.turboQuantHPatterns
    /// only; setting this as the default model strategy would
    /// quantize every weight (including small norms and head bias
    /// tensors that have neither the size nor the distribution to
    /// benefit), so we don't expose it as a model-wide default yet.
    case turboQuantH = "turbo_quant_h"
}

/// Weight loading strategies.
public enum SmeltLoadStrategy: String, Sendable {
    case mmapPrefault = "mmap_prefault"
}

/// Weight packing modes.
public enum SmeltPackingMode: String, Sendable {
    case monolithic
}

/// Authored HuggingFace tensor-name map used for checkpoint ingestion.
public enum SmeltCheckpointMap: String, Sendable, Equatable {
    case qwenHF = "hf.qwen"
    case qwenMTPHF = "hf.qwen-mtp"
    case llamaHF = "hf.llama"
    case qwen3TTSTalkerTrunkHF = "hf.qwen3-tts-talker-trunk"
}

public enum SmeltGeneratedKernelPolicy: String, Sendable {
    case auto
    case disabled
}

public enum SmeltWeightLayoutPolicy: String, Sendable {
    case memoryNeutral = "memory_neutral"
}

public struct SmeltCompilationConfig: Sendable, Equatable {
    public let generatedKernels: SmeltGeneratedKernelPolicy
    public let generatedKernelConsumerKinds: Set<SmeltKernelConsumerKind>?
    public let weightLayout: SmeltWeightLayoutPolicy

    public init(
        generatedKernels: SmeltGeneratedKernelPolicy = .auto,
        generatedKernelConsumerKinds: Set<SmeltKernelConsumerKind>? = nil,
        weightLayout: SmeltWeightLayoutPolicy = .memoryNeutral
    ) {
        self.generatedKernels = generatedKernels
        self.generatedKernelConsumerKinds = generatedKernelConsumerKinds
        self.weightLayout = weightLayout
    }

    public var generatedKernelConsumerKindNames: [String] {
        guard generatedKernels == .auto else { return [] }
        let allowedKinds = generatedKernelConsumerKinds ?? Set(SmeltKernelConsumerKind.allCases)
        return SmeltKernelConsumerKind.allCases
            .filter { allowedKinds.contains($0) }
            .map(\.rawValue)
    }

    func allowsGeneratedKernelConsumer(kind: SmeltKernelConsumerKind?) -> Bool {
        guard generatedKernels == .auto else { return false }
        guard let allowedKinds = generatedKernelConsumerKinds else { return true }
        guard let kind else { return false }
        return allowedKinds.contains(kind)
    }
}

// MARK: - Layer pattern

/// Describes the repeating layer pattern.
public struct SmeltLayerPattern: Sendable {
    /// The base pattern unit, e.g. [delta, delta, delta, attn].
    public let unit: [SmeltLayerType]

    /// Number of times the unit repeats.
    public let repeats: Int

    public init(unit: [SmeltLayerType], repeats: Int) {
        self.unit = unit
        self.repeats = repeats
    }

    /// Fully expanded layer list.
    public var expanded: [SmeltLayerType] {
        (0..<repeats).flatMap { _ in unit }
    }
}

// MARK: - Quantization config

/// Quantization specification.
public struct SmeltQuantizationConfig: Sendable {
    public let strategy: SmeltQuantStrategy
    public let groupSize: Int
    /// Weight name patterns excluded from quantization (glob-style).
    public let excludePatterns: [String]
    /// When true, embed_tokens is quantized and uses lut_embedding_gather.
    /// Saves ~970MB for vocab=248K, hidden=2048 models. Default false.
    public let quantizeEmbedding: Bool
    /// Weight name patterns routed to TurboQuant-H (2.125 bpw) instead
    /// of the model-default strategy. Glob-style; matches before
    /// excludePatterns. Intended for big embedding tables (e.g. AltUp
    /// per-layer input tables), which trade ~half of their bytes for a
    /// ~0.94 cosine reconstruction with negligible PPL impact. Empty by
    /// default.
    public let turboQuantHPatterns: [String]
    /// Glob-style weight name patterns whose matched matvec-PROJECTION
    /// weights are kept at their NATIVE bf16 dtype instead of being
    /// downcast to fp16 (the opt-in for bf16-direct LLM projections,
    /// dtype-building-blocks plan U2). Per-layer projections are
    /// layer-prefixed, so a span-all-layers entry needs a wildcard
    /// (e.g. `*_q_proj_weight`), like `turboQuantHPatterns`. bf16-SOURCE
    /// only today (the quantizer rejects a non-bf16 match); fp32-source
    /// preservation is a deferred unit. Like `turboQuantHPatterns` this is
    /// per-tensor and must be part of the reuse fingerprint: changing the
    /// list shifts the weights.bin layout. Empty by default.
    public let preserveNativePatterns: [String]

    public init(
        strategy: SmeltQuantStrategy,
        groupSize: Int,
        excludePatterns: [String],
        quantizeEmbedding: Bool = false,
        turboQuantHPatterns: [String] = [],
        preserveNativePatterns: [String] = []
    ) {
        self.strategy = strategy
        self.groupSize = groupSize
        self.excludePatterns = excludePatterns
        self.quantizeEmbedding = quantizeEmbedding
        self.turboQuantHPatterns = turboQuantHPatterns
        self.preserveNativePatterns = preserveNativePatterns
    }
}

// MARK: - Loading config

/// How weights are loaded at runtime.
public struct SmeltLoadingConfig: Sendable {
    public let strategy: SmeltLoadStrategy
    public let packing: SmeltPackingMode
    public let checkpointMap: SmeltCheckpointMap?

    public init(
        strategy: SmeltLoadStrategy,
        packing: SmeltPackingMode,
        checkpointMap: SmeltCheckpointMap? = nil
    ) {
        self.strategy = strategy
        self.packing = packing
        self.checkpointMap = checkpointMap
    }
}

// MARK: - Prefill config

/// Optional prefill configuration.
public struct SmeltPrefillConfig: Sendable {
    public let engine: String  // "coreml" or "metal"
    public let modelPath: String
    /// Path to prompt cache directory (.npy files + meta.json).
    public let cachePath: String
    /// Maximum number of tokens processed in one prefill chunk.
    public let maxBatchSize: Int
    /// Handoff families from the DSL (e.g. "conv_state", "key_cache").
    /// The compiler expands these to per-layer concrete mappings using the buffer plan.
    public let handoffFamilies: [String]
    /// Emit lm_head + logit_cap for every position in the batch
    /// instead of only the last. Required for full-distribution verify
    /// (reading K+1 logit rows from one chunked-prefill pass instead of
    /// K+1 sequential decodes). Greedy verify packages additionally get
    /// a separate argmax-only prefill table when the LM-head shape has a
    /// fused verifier kernel. Costs `maxBatchSize * vocabSize * 2`
    /// bytes of logitsBuf and `maxBatchSize` matvec dispatches per
    /// full-logits prefill call. Default off.
    public let emitAllLogits: Bool
    /// Emit only the argmax verifier prefill table used by verify gates.
    /// Unlike `emitAllLogits`, this does not change the normal prefill
    /// dispatch table or allocate full batch logits.
    public let verifyArgmax: Bool
    /// Maximum number of recurrent token successors retained transactionally
    /// by the verifier. The module owns this memory/performance tradeoff;
    /// non-recurrent packages ignore it.
    public let verifyTokenCapacity: Int

    public init(
        engine: String,
        modelPath: String,
        cachePath: String = "",
        maxBatchSize: Int = 64,
        handoffFamilies: [String],
        emitAllLogits: Bool = false,
        verifyArgmax: Bool = false,
        verifyTokenCapacity: Int? = nil
    ) {
        self.engine = engine
        self.modelPath = modelPath
        self.cachePath = cachePath
        self.maxBatchSize = maxBatchSize
        self.handoffFamilies = handoffFamilies
        self.emitAllLogits = emitAllLogits
        self.verifyArgmax = verifyArgmax
        self.verifyTokenCapacity = verifyTokenCapacity
            ?? (verifyArgmax ? min(maxBatchSize, 8) : 0)
    }
}

// MARK: - Runtime config

public enum SmeltConfigValueSource: String, Sendable, Equatable {
    case explicit
    case modelPreset
    case defaultValue
}

public typealias SmeltInferenceValueSource = SmeltConfigValueSource
public typealias SmeltDecodeValueSource = SmeltConfigValueSource

/// Runtime dispatch policy parsed from source config.
public struct SmeltRuntimePolicyConfig: Sendable {
    /// Schema-owned runtime policy label.
    public let architecture: String?
    /// Whether architecture came from source config or a compatibility guess.
    public let architectureSource: SmeltConfigValueSource

    public init(
        architecture: String? = nil,
        architectureSource: SmeltConfigValueSource? = nil
    ) {
        self.architecture = architecture
        self.architectureSource = architectureSource
            ?? (architecture == nil ? .defaultValue : .explicit)
    }
}

// MARK: - Decode config

/// Runtime decode policy parsed from source config.
public struct SmeltDecodeConfig: Sendable {
    /// Schema-owned decode policy used by CAM package assembly.
    public let policy: SmeltPackageSpec.DecodePolicy?
    /// Whether the policy came from source config or a compatibility default.
    public let policySource: SmeltDecodeValueSource

    public init(
        policy: SmeltPackageSpec.DecodePolicy? = nil,
        policySource: SmeltDecodeValueSource? = nil
    ) {
        self.policy = policy
        self.policySource = policySource ?? (policy == nil ? .defaultValue : .explicit)
    }
}

// MARK: - Inference config

/// Runtime inference parameters — drives the generate loop from the DSL.
public struct SmeltInferenceConfig: Sendable {
    /// Maximum tokens to generate per request.
    public let maxTokens: Int
    /// Whether max tokens came from source config or a compatibility default.
    public let maxTokensSource: SmeltInferenceValueSource
    /// Token IDs that signal end of generation.
    public let eosTokens: [Int32]
    /// Whether EOS tokens came from source config or a compatibility default.
    public let eosTokensSource: SmeltInferenceValueSource
    /// Token ID for <think> (if model uses think-skip). Nil to disable.
    public let thinkToken: Int32?
    /// Token ID for </think>.
    public let thinkEndToken: Int32?
    /// Token to inject after </think> during think-skip (typically newline).
    public let thinkSkipSuffix: Int32?
    /// Chat-template name packaged with the model (nil -> raw prompt fallback outside CAM).
    public let chatTemplate: String?
    /// Thinking-channel policy for the chat template.
    /// Nil means unspecified; CAM lowering requires it explicitly.
    public let thinkingPolicy: SmeltThinkingPolicy?
    /// Package-native tool transcript codec, independent of role/turn framing.
    public let toolTranscriptCodec: String?
    /// Prompt-state restoration semantics derived from CAM state declarations.
    public let promptStateRestoreMode: SmeltPromptStateRestoreMode?

    public init(
        maxTokens: Int = 512,
        maxTokensSource: SmeltInferenceValueSource? = nil,
        eosTokens: [Int32] = [],
        eosTokensSource: SmeltInferenceValueSource? = nil,
        thinkToken: Int32? = nil,
        thinkEndToken: Int32? = nil,
        thinkSkipSuffix: Int32? = nil,
        chatTemplate: String? = nil,
        thinkingPolicy: SmeltThinkingPolicy? = nil,
        toolTranscriptCodec: String? = nil,
        promptStateRestoreMode: SmeltPromptStateRestoreMode? = nil
    ) {
        self.maxTokens = maxTokens
        self.maxTokensSource = maxTokensSource ?? (maxTokens == 512 ? .defaultValue : .explicit)
        self.eosTokens = eosTokens
        self.eosTokensSource = eosTokensSource ?? (eosTokens.isEmpty ? .defaultValue : .explicit)
        self.thinkToken = thinkToken
        self.thinkEndToken = thinkEndToken
        self.thinkSkipSuffix = thinkSkipSuffix
        self.chatTemplate = chatTemplate
        self.thinkingPolicy = thinkingPolicy
        self.toolTranscriptCodec = toolTranscriptCodec
        self.promptStateRestoreMode = promptStateRestoreMode
    }
}

// MARK: - Validation

/// Compile-time validation errors — every structural problem is caught before code-gen.
public enum SmeltIRValidationError: Error, CustomStringConvertible {
    /// A config combination the compiler cannot build (yet) — always loud,
    /// never a silent fall-through.
    case unsupportedConfiguration(String)
    case layerCountMismatch(expected: Int, got: Int)
    case qkvDimMismatch(expected: Int, got: Int)
    case qkvLayoutInvalid(qkvDim: Int, zDim: Int, headDim: Int)
    case zDimMismatch(expected: Int, got: Int)
    case aDimMismatch(expected: Int, got: Int)
    case bDimMismatch(expected: Int, got: Int)
    case qkValueHeadRatioNotInteger(qkHeads: Int, valueHeads: Int)
    case gqaRatioNotInteger(qHeads: Int, kvHeads: Int)
    case hiddenSizeMismatch(deltaHeads: Int, deltaDim: Int, product: Int, hiddenSize: Int)
    case groupSizeInvalid(groupSize: Int)
    case ropeDimExceedsHeadDim(ropeDim: Int, headDim: Int)
    case ropeDimOdd(ropeDim: Int)
    case vocabSizeInvalid(vocabSize: Int)
    case rmsEpsInvalid(rmsEps: Float)
    case dimensionNotPositive(name: String, value: Int)
    case missingLayerConfig(layerType: String)
    case alignmentViolation(name: String, required: Int, actual: Int)
    /// Used for structural cluster-embedder constraints whose violation
    /// is *not* a "dimension is non-positive" failure (top_k exceeds
    /// num_centroids, vocab not divisible, etc.). Carries a free-form
    /// description so the reader gets the actual constraint name in the
    /// error message instead of a confusing "is not positive" suffix.
    case clusterEmbedderConstraint(detail: String)

    public var description: String {
        switch self {
        case let .unsupportedConfiguration(why):
            return why
        case let .layerCountMismatch(expected, got):
            return "layer count mismatch: config says \(expected), pattern expands to \(got)"
        case let .qkvDimMismatch(expected, got):
            return "qkv_dim must be 3 * delta value_dim (\(expected)), got \(got)"
        case let .qkvLayoutInvalid(qkvDim, zDim, headDim):
            return "delta qkv_dim \(qkvDim) with z_dim \(zDim) must leave 2 * qk_heads * head_dim chunks; head_dim=\(headDim)"
        case let .zDimMismatch(expected, got):
            return "delta z_dim must equal delta value_dim (\(expected)), got \(got)"
        case let .aDimMismatch(expected, got):
            return "delta a_dim must equal num_heads (\(expected)), got \(got)"
        case let .bDimMismatch(expected, got):
            return "delta b_dim must equal num_heads (\(expected)), got \(got)"
        case let .qkValueHeadRatioNotInteger(qkHeads, valueHeads):
            return "delta value head count \(valueHeads) must be an integer multiple of compact qk head count \(qkHeads)"
        case let .gqaRatioNotInteger(qHeads, kvHeads):
            return "GQA ratio must be integer: \(qHeads) / \(kvHeads) is not"
        case let .hiddenSizeMismatch(deltaHeads, deltaDim, product, hiddenSize):
            return "delta heads * head_dim (\(deltaHeads) * \(deltaDim) = \(product)) != hidden_size (\(hiddenSize))"
        case let .groupSizeInvalid(groupSize):
            return "quantization group_size must be power of 2 >= 4, got \(groupSize)"
        case let .ropeDimExceedsHeadDim(ropeDim, headDim):
            return "rope_dim \(ropeDim) exceeds attention head_dim \(headDim)"
        case let .ropeDimOdd(ropeDim):
            return "rope_dim must be even (rotate in pairs), got \(ropeDim)"
        case let .vocabSizeInvalid(vocabSize):
            return "vocab_size must be > 0, got \(vocabSize)"
        case let .rmsEpsInvalid(rmsEps):
            return "rms_eps must be > 0, got \(rmsEps)"
        case let .dimensionNotPositive(name, value):
            return "\(name) must be > 0, got \(value)"
        case let .missingLayerConfig(layerType):
            return "pattern contains \(layerType) layers but no \(layerType) config block"
        case let .alignmentViolation(name, required, actual):
            return "\(name) must be a multiple of \(required), got \(actual)"
        case let .clusterEmbedderConstraint(detail):
            return "cluster_embedder constraint: \(detail)"
        }
    }
}

/// Validate the IR — all structural checks that catch bad configs before code-gen.
public func validateSmeltIR(_ ir: SmeltModelIR) throws {
    // Dense-trunk ABI (docs/talker-trunk-fit-audit.md): decode emission,
    // weight packing, and runtime e2e have landed. What's still tied to the
    // generic fp16 graph is gated below:
    // DeltaNet, cluster embedders, backbone projection slabs, and prefill (W2).
    // The activation ABI is fp16, bf16, or fp32, full stop — a programmatic IR
    // with any other dtype must not reach the fp16 emitter sized as
    // 2-byte-something (the parser already rejects it; direct construction
    // must too).
    guard [.fp16, .bf16, .fp32].contains(ir.config.activationDtype) else {
        throw SmeltIRValidationError.unsupportedConfiguration(
            "activation_dtype must be fp16, bf16, or fp32, got "
                + ir.config.activationDtype.rawValue)
    }
    if ir.config.portTopology == .embeddingsInHiddenOut {
        guard ir.config.activationDtype == .bf16 || ir.config.activationDtype == .fp32 else {
            throw SmeltIRValidationError.unsupportedConfiguration(
                "embeddings-in/hidden-out trunks have no registered "
                    + "\(ir.config.activationDtype.rawValue) operation cells")
        }
        if ir.config.delta != nil || ir.numDeltaLayers > 0 {
            throw SmeltIRValidationError.unsupportedConfiguration(
                "dense activation trunks do not support DeltaNet layers")
        }
        if ir.config.clusterEmbedder != nil {
            throw SmeltIRValidationError.unsupportedConfiguration(
                "dense activation trunks do not support cluster embedders (fp16-only kernels)")
        }
        if ir.config.resolvedInputFusion != nil {
            throw SmeltIRValidationError.unsupportedConfiguration(
                "dense activation trunks do not support backbone projection slabs "
                    + "(fp16-only paths)")
        }
        if let prefill = ir.prefill {
            // W2/B3.2b: the fp32 prefill table (DenseTrunkPrefillEmitter) exists. It is
            // a Metal-native, cross-chunk table whose attention uses
            // causal_gqa_attn_cached_f32 (threadgroup score buffer sized 2048), so a
            // per-chunk batch must be metal-engine and ≤ 2048 tokens. The chunked
            // harness bounds TOTAL prefill seqLen ≤ 2048 at runtime.
            guard prefill.engine == "metal" else {
                throw SmeltIRValidationError.unsupportedConfiguration(
                    "dense activation trunks support only metal-native prefill "
                    + "(got engine '\(prefill.engine)')")
            }
            guard prefill.maxBatchSize >= 1 else {
                // The buffer plan uses max_prefill_batch as the activation-slab
                // multiplier; a nonpositive value would mismatch the runtime's
                // clamp-to-1 and trap on real seqLen > 1.
                throw SmeltIRValidationError.unsupportedConfiguration(
                    "dense activation prefill needs max_prefill_batch ≥ 1 "
                    + "(got \(prefill.maxBatchSize))")
            }
            guard prefill.maxBatchSize <= 2048 else {
                throw SmeltIRValidationError.unsupportedConfiguration(
                    "dense activation prefill uses a cached GQA score buffer "
                        + "(threadgroup score buffer caps at 2048); max_prefill_batch "
                        + "\(prefill.maxBatchSize) > 2048")
            }
        }
    } else if ir.config.activationDtype != .fp16 {
        throw SmeltIRValidationError.unsupportedConfiguration(
            "token-in/logits-out graphs have no registered "
                + "\(ir.config.activationDtype.rawValue) operation cells")
    }
    let cfg = ir.config

    if let prefill = ir.prefill {
        guard prefill.maxBatchSize >= 1 else {
            throw SmeltIRValidationError.unsupportedConfiguration(
                "prefill max_batch_size must be at least 1; got "
                    + "\(prefill.maxBatchSize)"
            )
        }
        if prefill.verifyArgmax {
            guard (1...prefill.maxBatchSize).contains(
                prefill.verifyTokenCapacity
            ) else {
                throw SmeltIRValidationError.unsupportedConfiguration(
                    "verify-argmax transaction capacity must be in [1, "
                        + "max_batch_size]; got \(prefill.verifyTokenCapacity) "
                        + "for batch \(prefill.maxBatchSize)"
                )
            }
        } else if prefill.verifyTokenCapacity != 0 {
            throw SmeltIRValidationError.unsupportedConfiguration(
                "prefill transaction capacity requires verify-argmax"
            )
        }
    }

    // --- Global positivity guards ---
    let globalChecks: [(String, Int)] = [
        ("hidden_size", cfg.hiddenSize),
        ("num_layers", cfg.numLayers),
        ("ffn.dim", cfg.ffn.dim),
        ("rope_dim", cfg.ropeDim),
    ]
    for (name, value) in globalChecks {
        if value <= 0 {
            throw SmeltIRValidationError.dimensionNotPositive(name: name, value: value)
        }
    }

    // Layer count must match pattern expansion
    if ir.totalLayers != cfg.numLayers {
        throw SmeltIRValidationError.layerCountMismatch(
            expected: cfg.numLayers, got: ir.totalLayers
        )
    }

    if cfg.sharedKVLayers < 0 {
        throw SmeltIRValidationError.dimensionNotPositive(name: "shared_kv_layers", value: cfg.sharedKVLayers)
    }
    if let staticSeqCapacity = cfg.staticSeqCapacity, staticSeqCapacity <= 0 {
        throw SmeltIRValidationError.dimensionNotPositive(
            name: "static_seq_capacity",
            value: staticSeqCapacity
        )
    }
    if cfg.staticSeqCapacity == nil, !ir.usesDynamicContext {
        throw SmeltIRValidationError.dimensionNotPositive(name: "static_seq_capacity", value: 0)
    }
    if let sharedKVFFNDim = cfg.sharedKVFFNDim, sharedKVFFNDim <= 0 {
        throw SmeltIRValidationError.dimensionNotPositive(name: "shared_kv_ffn_dim", value: sharedKVFFNDim)
    }
    if let backboneHiddenSize = cfg.backboneHiddenSize, backboneHiddenSize <= 0 {
        throw SmeltIRValidationError.dimensionNotPositive(
            name: "backbone_hidden_size",
            value: backboneHiddenSize
        )
    }
    if cfg.backboneHiddenSize != nil, cfg.inputFusion != nil {
        throw SmeltIRValidationError.unsupportedConfiguration(
            "backbone_hidden_size and input_fusion are alternative spellings; set only one"
        )
    }
    if let fusion = cfg.inputFusion {
        if cfg.tiedLMHead {
            throw SmeltIRValidationError.unsupportedConfiguration(
                "input_fusion packages omit embed_tokens and require tied_lm_head false"
            )
        }
        if fusion.sourceWidth <= 0 {
            throw SmeltIRValidationError.dimensionNotPositive(
                name: "input_fusion.source_width", value: fusion.sourceWidth)
        }
        if fusion.sourceCount <= 0 {
            throw SmeltIRValidationError.dimensionNotPositive(
                name: "input_fusion.source_count", value: fusion.sourceCount)
        }
        if let postWidth = fusion.postProjectionWidth, postWidth <= 0 {
            throw SmeltIRValidationError.dimensionNotPositive(
                name: "input_fusion.post_projection_width", value: postWidth)
        }
    }
    if let cluster = cfg.clusterEmbedder {
        if cluster.numCentroids <= 0 {
            throw SmeltIRValidationError.dimensionNotPositive(
                name: "cluster_embedder.num_centroids",
                value: cluster.numCentroids
            )
        }
        if cluster.topK <= 0 {
            throw SmeltIRValidationError.dimensionNotPositive(
                name: "cluster_embedder.top_k",
                value: cluster.topK
            )
        }
        if cluster.topK > cluster.numCentroids {
            throw SmeltIRValidationError.clusterEmbedderConstraint(
                detail: "top_k (\(cluster.topK)) exceeds num_centroids (\(cluster.numCentroids))"
            )
        }
        if cfg.vocabSize % cluster.numCentroids != 0 {
            // Each cluster must hold exactly vocab_size / num_centroids
            // tokens; the token_ordering permutation buffer's contiguous
            // cluster blocks rely on this divisibility. A non-divisible
            // setup would mean some clusters have ragged sizes and the
            // top-k gather would index past the cluster end.
            throw SmeltIRValidationError.clusterEmbedderConstraint(
                detail: "vocab_size (\(cfg.vocabSize)) must be divisible by num_centroids (\(cluster.numCentroids))"
            )
        }
        if ir.quantization.quantizeEmbedding {
            // The cluster_sparse_lm_head v1 kernel binds a single
            // lm_head weight buffer with no LUT / scales / biases
            // slots. Quantized embed_tokens (which is the tied
            // lm_head source) would be silently misread. Draft models
            // are ~78 MB in fp16 — quantization isn't required.
            // Future unit can widen the kernel signature; v1 keeps
            // the binding count at 5.
            throw SmeltIRValidationError.clusterEmbedderConstraint(
                detail: "cluster_embedder requires quantize_embedding false; the v1 sparse lm_head kernel does not support quantized embed_tokens / lm_head"
            )
        }
        if !cfg.tiedLMHead {
            // For untied lm_head, the weight packer quantizes
            // lm_head_weight per the spec's quantization strategy
            // (independent of quantize_embedding). The sparse kernel
            // would then read packed u4/affine bytes as fp16. The only
            // known consumers ship with tied_lm_head=true, so this gate
            // matches them; future architectures with separate lm_heads
            // would need the quantized binding extension referenced above.
            throw SmeltIRValidationError.clusterEmbedderConstraint(
                detail: "cluster_embedder requires tied_lm_head true; the v1 sparse lm_head kernel cannot read a quantized untied lm_head_weight"
            )
        }
        let tokensPerCluster = cluster.tokensPerCluster(vocabSize: cfg.vocabSize)
        if tokensPerCluster > 256 {
            // The cluster_sparse_lm_head dispatch caps tgWidth at
            // min(tokens_per_cluster, 256). When tokens_per_cluster
            // exceeds 256, threads with tid >= 256 are never spawned
            // and their vocab slots remain uninitialized. Until the
            // kernel grows an inner serial loop over slots, reject
            // shapes that would trip the cap. Canonical E2B/E4B
            // shape (2048 centroids over 262 K vocab) has 128
            // tokens/cluster, well under the limit.
            throw SmeltIRValidationError.clusterEmbedderConstraint(
                detail: "tokens_per_cluster (vocab_size / num_centroids = \(tokensPerCluster)) exceeds the v1 kernel's per-threadgroup cap of 256; pick a num_centroids value that yields tokens_per_cluster <= 256"
            )
        }
    }
    if cfg.vocabSizePerLayerInput < 0 {
        throw SmeltIRValidationError.dimensionNotPositive(
            name: "vocab_size_per_layer_input",
            value: cfg.vocabSizePerLayerInput
        )
    }
    if cfg.hiddenSizePerLayerInput < 0 {
        throw SmeltIRValidationError.dimensionNotPositive(
            name: "hidden_size_per_layer_input",
            value: cfg.hiddenSizePerLayerInput
        )
    }
    if cfg.sharedKVLayers > cfg.numLayers {
        throw SmeltIRValidationError.layerCountMismatch(
            expected: cfg.numLayers, got: cfg.sharedKVLayers
        )
    }
    if cfg.sharedKVFFNDim != nil, cfg.sharedKVLayers == 0 {
        throw SmeltIRValidationError.dimensionNotPositive(name: "shared_kv_layers", value: cfg.sharedKVLayers)
    }

    if let logitCap = cfg.logitCap, logitCap <= 0 {
        throw SmeltIRValidationError.dimensionNotPositive(name: "logit_cap", value: Int(logitCap))
    }
    if let attnLogitCap = cfg.attnLogitCap, attnLogitCap <= 0 {
        throw SmeltIRValidationError.dimensionNotPositive(name: "attn_logit_cap", value: Int(attnLogitCap))
    }

    // Pattern must only use layer types that have configs
    let hasDelta = ir.numDeltaLayers > 0
    let hasAttn = ir.numAttnLayers > 0

    if hasDelta && cfg.delta == nil {
        throw SmeltIRValidationError.missingLayerConfig(layerType: "delta")
    }
    if hasAttn {
        let attentionTypes = Set(ir.layerPattern.expanded.filter(\.isAttentionFamily))
        for layerType in attentionTypes where cfg.attentionConfigs[layerType] == nil {
            throw SmeltIRValidationError.missingLayerConfig(layerType: layerType.rawValue)
        }
    }

    // --- DeltaNet validation (only if delta layers exist) ---
    if let delta = cfg.delta, hasDelta {
        let deltaChecks: [(String, Int)] = [
            ("delta.num_heads", delta.numHeads),
            ("delta.head_dim", delta.headDim),
            ("delta.conv_kernel", delta.convKernel),
        ]
        for (name, value) in deltaChecks {
            if value <= 0 {
                throw SmeltIRValidationError.dimensionNotPositive(name: name, value: value)
            }
        }

        let deltaValueDim = delta.valueDim
        if delta.zDim != deltaValueDim {
            throw SmeltIRValidationError.zDimMismatch(
                expected: deltaValueDim, got: delta.zDim
            )
        }
        let qkDim = delta.qkvDim - delta.zDim
        if qkDim <= 0 || qkDim % (2 * delta.headDim) != 0 {
            throw SmeltIRValidationError.qkvLayoutInvalid(
                qkvDim: delta.qkvDim,
                zDim: delta.zDim,
                headDim: delta.headDim
            )
        }
        let qkHeads = qkDim / (2 * delta.headDim)
        if delta.numHeads % qkHeads != 0 {
            throw SmeltIRValidationError.qkValueHeadRatioNotInteger(
                qkHeads: qkHeads,
                valueHeads: delta.numHeads
            )
        }
        if delta.aDim != delta.numHeads {
            throw SmeltIRValidationError.aDimMismatch(
                expected: delta.numHeads, got: delta.aDim
            )
        }
        if delta.bDim != delta.numHeads {
            throw SmeltIRValidationError.bDimMismatch(
                expected: delta.numHeads, got: delta.bDim
            )
        }
        // Kernel constraint: SIMD reduction needs multiples of 32
        if delta.headDim % 32 != 0 {
            throw SmeltIRValidationError.alignmentViolation(
                name: "delta.head_dim", required: 32, actual: delta.headDim
            )
        }
    }

    // --- Attention validation (only if attention layers exist) ---
    if cfg.ropeDim % 2 != 0 {
        throw SmeltIRValidationError.ropeDimOdd(ropeDim: cfg.ropeDim)
    }

    // --- Attention validation (only if attention layers exist) ---
    if hasAttn {
        let attentionTypes = Set(ir.layerPattern.expanded.filter(\.isAttentionFamily))
        for layerType in attentionTypes {
            guard let attn = cfg.attentionConfigs[layerType] else { continue }
            let attnChecks: [(String, Int)] = [
                ("\(layerType.rawValue).q_heads", attn.qHeads),
                ("\(layerType.rawValue).kv_heads", attn.kvHeads),
                ("\(layerType.rawValue).head_dim", attn.headDim),
            ]
            for (name, value) in attnChecks {
                if value <= 0 {
                    throw SmeltIRValidationError.dimensionNotPositive(name: name, value: value)
                }
            }
            if attn.attnScale <= 0 {
                throw SmeltIRValidationError.dimensionNotPositive(
                    name: "\(layerType.rawValue).attn_scale",
                    value: Int(attn.attnScale)
                )
            }
            if cfg.portTopology == .embeddingsInHiddenOut && attn.qkvBias {
                throw SmeltIRValidationError.unsupportedConfiguration(
                    "dense activation trunks do not support qkv_bias projection biases"
                )
            }

            if attn.qHeads % attn.kvHeads != 0 {
                throw SmeltIRValidationError.gqaRatioNotInteger(
                    qHeads: attn.qHeads, kvHeads: attn.kvHeads
                )
            }
            if attn.headDim % 32 != 0 {
                throw SmeltIRValidationError.alignmentViolation(
                    name: "\(layerType.rawValue).head_dim", required: 32, actual: attn.headDim
                )
            }
            let effectiveRopeDim = attn.effectiveRopeDim(default: cfg.ropeDim)
            if effectiveRopeDim % 2 != 0 {
                throw SmeltIRValidationError.ropeDimOdd(ropeDim: effectiveRopeDim)
            }
            if effectiveRopeDim > attn.headDim {
                throw SmeltIRValidationError.ropeDimExceedsHeadDim(
                    ropeDim: effectiveRopeDim, headDim: attn.headDim
                )
            }
        }
    }

    // --- Quantization (u4-specific checks for both LUT and affine strategies) ---
    if ir.quantization.strategy == .lutU4 || ir.quantization.strategy == .affineU4 {
        let gs = ir.quantization.groupSize
        if gs < 4 || (gs & (gs - 1)) != 0 {
            throw SmeltIRValidationError.groupSizeInvalid(groupSize: gs)
        }

        // u4 packing requires even columns (kernel uses cols/2 floor stride)
        if cfg.hiddenSize % 2 != 0 {
            throw SmeltIRValidationError.alignmentViolation(
                name: "hidden_size (u4 packing)", required: 2, actual: cfg.hiddenSize
            )
        }
        if cfg.ffn.dim % 2 != 0 {
            throw SmeltIRValidationError.alignmentViolation(
                name: "ffn.dim (u4 packing)", required: 2, actual: cfg.ffn.dim
            )
        }
        if let sharedKVFFNDim = cfg.sharedKVFFNDim, sharedKVFFNDim % 2 != 0 {
            throw SmeltIRValidationError.alignmentViolation(
                name: "shared_kv_ffn_dim (u4 packing)", required: 2, actual: sharedKVFFNDim
            )
        }
    }

    // A weight cannot be both preserved at its native dtype AND TurboQuant-H quantized —
    // the two are contradictory storage requests. The retired model-spec DSL parser rejected
    // this at parse time; now that every authoring path (CAM module, programmatic factory)
    // reaches this validator instead, enforce it here so the invariant holds regardless of source.
    for pattern in ir.quantization.preserveNativePatterns
    where ir.quantization.turboQuantHPatterns.contains(pattern) {
        throw SmeltIRValidationError.unsupportedConfiguration(
            "quantization pattern '\(pattern)' is in both preserve_native and turbo_quant_h "
                + "(a weight cannot be both preserved-native and TurboQuant-H quantized)"
        )
    }

    // Kernel constraint: SIMD reduction on hidden_size
    if cfg.hiddenSize % 32 != 0 {
        throw SmeltIRValidationError.alignmentViolation(
            name: "hidden_size", required: 32, actual: cfg.hiddenSize
        )
    }

    // --- General sanity ---
    if cfg.vocabSize <= 0 {
        throw SmeltIRValidationError.vocabSizeInvalid(vocabSize: cfg.vocabSize)
    }
    if cfg.rmsEps <= 0 {
        throw SmeltIRValidationError.rmsEpsInvalid(rmsEps: cfg.rmsEps)
    }
}

// MARK: - Qwen 3.5 2B factory (reference config for tests + bootstrapping)

extension SmeltModelIR {
    /// The exact Qwen 3.5 2B configuration used for testing and as a reference.
    public static let qwen35_2B = SmeltModelIR(
        modelName: "Qwen/Qwen3.5-2B",
        config: SmeltConfig(
            hiddenSize: 2_048,
            numLayers: 24,
            vocabSize: 248_320,
            staticSeqCapacity: 256,
            ropeDim: 64,
            rmsEps: 1e-6,
            normMode: .onePlusWeight,
            delta: SmeltDeltaConfig(
                numHeads: 16,
                headDim: 128,
                convKernel: 4,
                qkvDim: 6_144,
                zDim: 2_048,
                aDim: 16,
                bDim: 16
            ),
            attention: SmeltAttentionConfig(
                qHeads: 8,
                kvHeads: 2,
                headDim: 256,
                gatedQ: true,
                qkNorm: true,
                ropeTheta: 10_000_000
            ),
            ffn: SmeltFFNConfig(
                dim: 6_144,
                activation: .swiglu
            )
        ),
        layerPattern: SmeltLayerPattern(
            unit: [.delta, .delta, .delta, .attention],
            repeats: 6
        ),
        quantization: SmeltQuantizationConfig(
            strategy: .lutU4,
            groupSize: 16,
            excludePatterns: [
                SmeltCanonicalTensorNames.embedTokens,
                "conv1d_weight", "A_log", "dt_bias", "*_norm_weight",
            ]
        ),
        loading: SmeltLoadingConfig(
            strategy: .mmapPrefault,
            packing: .monolithic,
            checkpointMap: .qwenHF
        ),
        prefill: SmeltPrefillConfig(
            engine: "coreml",
            modelPath: "prefill.mlmodelc",
            cachePath: "cache",
            handoffFamilies: [
                "conv_state", "rec_state", "key_cache", "value_cache", "rope",
            ]
        )
    )

    /// Qwen 3.5 0.8B reference config for Metal-prefill bring-up and tuning.
    public static let qwen35_0_8B = SmeltModelIR(
        modelName: "Qwen/Qwen3.5-0.8B",
        config: SmeltConfig(
            hiddenSize: 1_024,
            numLayers: 24,
            vocabSize: 248_320,
            ropeDim: 64,
            rmsEps: 1e-6,
            normMode: .onePlusWeight,
            delta: SmeltDeltaConfig(
                numHeads: 16,
                headDim: 128,
                convKernel: 4,
                qkvDim: 6_144,
                zDim: 2_048,
                aDim: 16,
                bDim: 16
            ),
            attention: SmeltAttentionConfig(
                qHeads: 8,
                kvHeads: 2,
                headDim: 256,
                gatedQ: true,
                qkNorm: true,
                ropeTheta: 10_000_000
            ),
            ffn: SmeltFFNConfig(
                dim: 3_584,
                activation: .swiglu
            )
        ),
        layerPattern: SmeltLayerPattern(
            unit: [.delta, .delta, .delta, .attention],
            repeats: 6
        ),
        quantization: SmeltQuantizationConfig(
            strategy: .affineU4,
            groupSize: 64,
            excludePatterns: [
                "conv1d_weight", "A_log", "dt_bias", "*_norm_weight"
            ],
            quantizeEmbedding: true
        ),
        loading: SmeltLoadingConfig(
            strategy: .mmapPrefault,
            packing: .monolithic,
            checkpointMap: .qwenHF
        ),
        prefill: SmeltPrefillConfig(
            engine: "metal",
            modelPath: "",
            cachePath: "cache",
            maxBatchSize: 256,
            handoffFamilies: [
                "conv_state", "rec_state", "key_cache", "value_cache", "rope",
            ]
        )
    )

    /// Qwen 3.5 4B reference config for bring-up and specialization work.
    public static let qwen35_4B = SmeltModelIR(
        modelName: "Qwen/Qwen3.5-4B",
        config: SmeltConfig(
            hiddenSize: 2_560,
            numLayers: 32,
            vocabSize: 248_320,
            ropeDim: 64,
            rmsEps: 1e-6,
            normMode: .onePlusWeight,
            delta: SmeltDeltaConfig(
                numHeads: 32,
                headDim: 128,
                convKernel: 4,
                qkvDim: 8_192,
                zDim: 4_096,
                aDim: 32,
                bDim: 32
            ),
            attention: SmeltAttentionConfig(
                qHeads: 16,
                kvHeads: 4,
                headDim: 256,
                gatedQ: true,
                qkNorm: true,
                ropeTheta: 10_000_000
            ),
            ffn: SmeltFFNConfig(
                dim: 9_216,
                activation: .swiglu
            )
        ),
        layerPattern: SmeltLayerPattern(
            unit: [.delta, .delta, .delta, .attention],
            repeats: 8
        ),
        quantization: SmeltQuantizationConfig(
            strategy: .affineU4,
            groupSize: 64,
            excludePatterns: [
                "conv1d_weight", "A_log", "dt_bias", "*_norm_weight"
            ],
            quantizeEmbedding: true
        ),
        loading: SmeltLoadingConfig(
            strategy: .mmapPrefault,
            packing: .monolithic,
            checkpointMap: .qwenHF
        ),
        prefill: SmeltPrefillConfig(
            engine: "metal",
            modelPath: "",
            cachePath: "cache",
            maxBatchSize: 256,
            handoffFamilies: [
                "conv_state", "rec_state", "key_cache", "value_cache", "rope",
            ]
        )
    )
}

// MARK: - Dense headless-trunk factory

extension SmeltModelIR {
    /// Builds the shared attention-only, embeddings-in/hidden-out trunk IR.
    /// Port topology defines the graph; `activationDtype` selects operation and
    /// storage cells within it. RoPE and norm semantics are explicit constants
    /// because checkpoint tensor shapes cannot recover them.
    static func denseTrunk(
        modelName: String,
        hidden: Int,
        numLayers: Int,
        vocab: Int,
        heads: Int,
        kvHeads: Int,
        headDim: Int,
        inter: Int,
        maxPrefillBatch: Int,
        staticSeqCapacity: Int = 256,
        activationDtype: SmeltDType = .fp32
    ) -> SmeltModelIR {
        SmeltModelIR(
            modelName: modelName,
            config: SmeltConfig(
                hiddenSize: hidden,
                numLayers: numLayers,
                vocabSize: vocab,
                staticSeqCapacity: staticSeqCapacity,
                ropeDim: headDim,
                rmsEps: 1e-6,
                normMode: .weight,
                activationDtype: activationDtype,
                portTopology: .embeddingsInHiddenOut,
                attention: SmeltAttentionConfig(
                    qHeads: heads,
                    kvHeads: kvHeads,
                    headDim: headDim,
                    gatedQ: false,
                    qkNorm: true,
                    qkNormMode: .weight,
                    ropeTheta: 1_000_000,
                    ropeLayout: .splitHalf
                ),
                ffn: SmeltFFNConfig(dim: inter, activation: .swiglu),
                tiedLMHead: false
            ),
            layerPattern: SmeltLayerPattern(unit: [.attention], repeats: numLayers),
            quantization: SmeltQuantizationConfig(
                strategy: .fp16,
                groupSize: 64,
                excludePatterns: [],
                quantizeEmbedding: false
            ),
            loading: SmeltLoadingConfig(
                strategy: .mmapPrefault,
                packing: .monolithic,
                checkpointMap: .qwen3TTSTalkerTrunkHF
            ),
            prefill: SmeltPrefillConfig(
                engine: "metal",
                modelPath: "",
                cachePath: "",
                maxBatchSize: maxPrefillBatch,
                handoffFamilies: ["key_cache", "value_cache"]
            )
        )
    }
}
