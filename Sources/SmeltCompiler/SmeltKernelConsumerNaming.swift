// Shared logical consumer and weight naming for capability-planned kernels.
//
// Planner, codegen, and weight-layout planning must agree on these IDs. Keeping
// the spelling in one place avoids shape-equivalent routes drifting by caller.

public enum SmeltKernelConsumerKind: String, Sendable, Equatable, Hashable, CaseIterable {
    case qProjBiasDecode
    case kProjBiasDecode
    case vProjBiasDecode
    case kvProjBiasDecode
    case attentionOutputResidualDecode
    case ffnDownResidualDecode
    case ffnGateUpPrefill
    case ffnDownPrefill

    public static let allCases: [SmeltKernelConsumerKind] = [
        .qProjBiasDecode,
        .kProjBiasDecode,
        .vProjBiasDecode,
        .kvProjBiasDecode,
        .attentionOutputResidualDecode,
        .ffnDownResidualDecode,
        .ffnGateUpPrefill,
        .ffnDownPrefill,
    ]
}

struct SmeltKernelLayerConsumerContext: Sendable {
    let layerIndex: Int
    let hiddenSize: Int
    let ffnDim: Int
    let groupSize: Int
    let blockTopology: SmeltBlockTopology
    let attention: SmeltAttentionConfig?
    let attentionHasOwnKV: Bool
    let prefillEngine: String?
    let ffnActivation: SmeltActivation

    init(
        layerIndex: Int,
        hiddenSize: Int,
        ffnDim: Int,
        groupSize: Int,
        blockTopology: SmeltBlockTopology,
        attention: SmeltAttentionConfig?,
        attentionHasOwnKV: Bool,
        prefillEngine: String?,
        ffnActivation: SmeltActivation
    ) {
        self.layerIndex = layerIndex
        self.hiddenSize = hiddenSize
        self.ffnDim = ffnDim
        self.groupSize = groupSize
        self.blockTopology = blockTopology
        self.attention = attention
        self.attentionHasOwnKV = attentionHasOwnKV
        self.prefillEngine = prefillEngine
        self.ffnActivation = ffnActivation
    }

    init(
        ir: SmeltModelIR,
        layerIndex: Int,
        layerType: SmeltLayerType,
        groupSize: Int? = nil
    ) {
        let attention = ir.config.attentionConfig(for: layerType)
        self.init(
            layerIndex: layerIndex,
            hiddenSize: ir.config.hiddenSize,
            ffnDim: ir.config.ffnDim(for: layerIndex),
            groupSize: groupSize ?? ir.quantization.groupSize,
            blockTopology: ir.config.blockTopology,
            attention: attention,
            attentionHasOwnKV: attention.map {
                !$0.externalKV && !ir.isKVSharedLayer(layerIndex)
            } ?? false,
            prefillEngine: ir.prefill?.engine,
            ffnActivation: ir.config.ffn.activation
        )
    }

    init(
        config: SmeltConfig,
        layerIndex: Int,
        groupSize: Int,
        attention: SmeltAttentionConfig? = nil,
        attentionHasOwnKV: Bool = false,
        prefillEngine: String? = nil
    ) {
        self.init(
            layerIndex: layerIndex,
            hiddenSize: config.hiddenSize,
            ffnDim: config.ffnDim(for: layerIndex),
            groupSize: groupSize,
            blockTopology: config.blockTopology,
            attention: attention,
            attentionHasOwnKV: attentionHasOwnKV,
            prefillEngine: prefillEngine,
            ffnActivation: config.ffn.activation
        )
    }
}

struct SmeltKernelConsumerDescriptor: Sendable, Equatable {
    let kind: SmeltKernelConsumerKind
    let operation: SmeltKernelOperationPattern

    init(kind: SmeltKernelConsumerKind) {
        self.kind = kind
        self.operation = Self.operation(for: kind)
    }

    var phase: SmeltKernelPhase {
        operation.plannedPhase
    }

    func isAvailable(in context: SmeltKernelLayerConsumerContext) -> Bool {
        switch kind {
        case .qProjBiasDecode:
            context.attention?.qkvBias == true
        case .kProjBiasDecode, .vProjBiasDecode:
            context.attention?.qkvBias == true && context.attentionHasOwnKV
        case .kvProjBiasDecode:
            context.attention?.qkvBias == true
                && context.attentionHasOwnKV
                && context.attention?.kProjDim == context.attention?.vProjDim
        case .attentionOutputResidualDecode:
            context.attention != nil && context.blockTopology == .standard
        case .ffnDownResidualDecode:
            context.blockTopology == .standard
        case .ffnGateUpPrefill:
            context.prefillEngine == "metal" && context.ffnActivation == .swiglu
        case .ffnDownPrefill:
            context.prefillEngine == "metal"
        }
    }

    func consumerID(layerIndex: Int) -> String {
        switch kind {
        case .qProjBiasDecode:
            return SmeltKernelConsumerNaming.qProjBiasDecodeConsumerID(
                layerIndex: layerIndex
            )
        case .kProjBiasDecode:
            return SmeltKernelConsumerNaming.kProjBiasDecodeConsumerID(
                layerIndex: layerIndex
            )
        case .vProjBiasDecode:
            return SmeltKernelConsumerNaming.vProjBiasDecodeConsumerID(
                layerIndex: layerIndex
            )
        case .kvProjBiasDecode:
            return SmeltKernelConsumerNaming.kvProjBiasDecodeConsumerID(
                layerIndex: layerIndex
            )
        case .attentionOutputResidualDecode:
            return SmeltKernelConsumerNaming.attentionOutputResidualDecodeConsumerID(
                layerIndex: layerIndex
            )
        case .ffnDownResidualDecode:
            return SmeltKernelConsumerNaming.ffnDownResidualDecodeConsumerID(
                layerIndex: layerIndex
            )
        case .ffnGateUpPrefill:
            return SmeltKernelConsumerNaming.ffnGateUpPrefillConsumerID(
                layerIndex: layerIndex
            )
        case .ffnDownPrefill:
            return SmeltKernelConsumerNaming.ffnDownPrefillConsumerID(
                layerIndex: layerIndex
            )
        }
    }

    func shape(in context: SmeltKernelLayerConsumerContext) -> SmeltKernelShape? {
        guard isAvailable(in: context) else { return nil }
        switch kind {
        case .qProjBiasDecode:
            guard let attn = context.attention else { return nil }
            return SmeltKernelShape(
                rows: attn.qProjDim,
                cols: context.hiddenSize,
                groupSize: context.groupSize
            )
        case .kProjBiasDecode:
            guard let attn = context.attention else { return nil }
            return SmeltKernelShape(
                rows: attn.kProjDim,
                cols: context.hiddenSize,
                groupSize: context.groupSize
            )
        case .vProjBiasDecode:
            guard let attn = context.attention else { return nil }
            return SmeltKernelShape(
                rows: attn.vProjDim,
                cols: context.hiddenSize,
                groupSize: context.groupSize
            )
        case .kvProjBiasDecode:
            guard let attn = context.attention,
                  attn.kProjDim == attn.vProjDim
            else {
                return nil
            }
            return SmeltKernelShape(
                rows: attn.kProjDim,
                cols: context.hiddenSize,
                groupSize: context.groupSize
            )
        case .attentionOutputResidualDecode:
            guard let attn = context.attention else { return nil }
            return SmeltKernelShape(
                rows: context.hiddenSize,
                cols: attn.qHeads * attn.headDim,
                groupSize: context.groupSize
            )
        case .ffnDownResidualDecode, .ffnDownPrefill:
            return SmeltKernelShape(
                rows: context.hiddenSize,
                cols: context.ffnDim,
                groupSize: context.groupSize
            )
        case .ffnGateUpPrefill:
            return SmeltKernelShape(
                rows: context.ffnDim,
                cols: context.hiddenSize,
                groupSize: context.groupSize
            )
        }
    }

    func weights(layerIndex: Int) -> [SmeltPlannedKernelWeight] {
        switch kind {
        case .qProjBiasDecode:
            return [SmeltKernelConsumerNaming.qProjAffineWeight(layerIndex: layerIndex)]
        case .kProjBiasDecode:
            return [SmeltKernelConsumerNaming.kProjAffineWeight(layerIndex: layerIndex)]
        case .vProjBiasDecode:
            return [SmeltKernelConsumerNaming.vProjAffineWeight(layerIndex: layerIndex)]
        case .kvProjBiasDecode:
            return SmeltKernelConsumerNaming.kvProjAffineWeights(layerIndex: layerIndex)
        case .attentionOutputResidualDecode:
            return [SmeltKernelConsumerNaming.oProjAffineWeight(layerIndex: layerIndex)]
        case .ffnDownResidualDecode, .ffnDownPrefill:
            return [SmeltKernelConsumerNaming.ffnDownAffineWeight(layerIndex: layerIndex)]
        case .ffnGateUpPrefill:
            return SmeltKernelConsumerNaming.ffnGateUpWeights(layerIndex: layerIndex)
        }
    }

    func candidate(
        in context: SmeltKernelLayerConsumerContext
    ) -> SmeltPlannedKernelCandidate? {
        guard let shape = shape(in: context) else { return nil }
        return SmeltPlannedKernelCandidate(
            consumerID: consumerID(layerIndex: context.layerIndex),
            operation: operation,
            shape: shape,
            weights: weights(layerIndex: context.layerIndex),
            kind: kind
        )
    }

    private static func operation(
        for kind: SmeltKernelConsumerKind
    ) -> SmeltKernelOperationPattern {
        switch kind {
        case .qProjBiasDecode,
             .kProjBiasDecode,
             .vProjBiasDecode,
             .attentionOutputResidualDecode,
             .ffnDownResidualDecode:
            return .affineMatvecResidualAdd
        case .kvProjBiasDecode:
            return .fusedDualAffineMatvecResidualAdd
        case .ffnGateUpPrefill:
            return .fusedGateUpSwigluPrefillFull
        case .ffnDownPrefill:
            return .affineMatvecPrefillFull
        }
    }
}

enum SmeltKernelConsumerNaming {
    static let generatedConsumerDescriptors: [SmeltKernelConsumerDescriptor] =
        SmeltKernelConsumerKind.allCases.map(SmeltKernelConsumerDescriptor.init(kind:))

    static func descriptor(
        for kind: SmeltKernelConsumerKind
    ) -> SmeltKernelConsumerDescriptor {
        generatedConsumerDescriptors.first { $0.kind == kind }
            ?? SmeltKernelConsumerDescriptor(kind: kind)
    }

    static func attentionPrefix(layerIndex: Int) -> String {
        "layers_\(layerIndex)_self_attn"
    }

    static func ffnPrefix(layerIndex: Int) -> String {
        "layers_\(layerIndex)_mlp"
    }

    static func deltaPrefix(layerIndex: Int) -> String {
        "layers_\(layerIndex)_linear_attn"
    }

    static func deltaQKVWeight(layerIndex: Int) -> String {
        "\(deltaPrefix(layerIndex: layerIndex))_in_proj_qkv_weight"
    }

    static func deltaZWeight(layerIndex: Int) -> String {
        "\(deltaPrefix(layerIndex: layerIndex))_in_proj_z_weight"
    }

    static func deltaAWeight(layerIndex: Int) -> String {
        "\(deltaPrefix(layerIndex: layerIndex))_in_proj_a_weight"
    }

    static func deltaBWeight(layerIndex: Int) -> String {
        "\(deltaPrefix(layerIndex: layerIndex))_in_proj_b_weight"
    }

    static func deltaOutputWeight(layerIndex: Int) -> String {
        "\(deltaPrefix(layerIndex: layerIndex))_out_proj_weight"
    }

    /// A small-batch affine projection is identified by the tensor it consumes.
    /// This keeps the generated route model-agnostic while still giving package
    /// planning and code emission one stable authorization key.
    static func affinePrefillSmallBatchConsumerID(weightName: String) -> String {
        "\(weightName).prefill.small_batch"
    }

    static func affinePrefillSmallBatchCandidate(
        weightName: String,
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> SmeltPlannedKernelCandidate {
        SmeltPlannedKernelCandidate(
            consumerID: affinePrefillSmallBatchConsumerID(weightName: weightName),
            operation: .affineMatvecPrefillSmallBatch,
            shape: SmeltKernelShape(
                rows: rows,
                cols: cols,
                groupSize: groupSize
            ),
            weights: [
                SmeltPlannedKernelWeight(
                    weightName: weightName,
                    role: .affine
                ),
            ]
        )
    }

    static func fusedDualAffinePrefillSmallBatchCandidate(
        firstWeightName: String,
        secondWeightName: String,
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> SmeltPlannedKernelCandidate {
        SmeltPlannedKernelCandidate(
            consumerID: "\(firstWeightName)+\(secondWeightName).prefill.small_batch",
            operation: .fusedDualAffineMatvecPrefillSmallBatch,
            shape: SmeltKernelShape(
                rows: rows,
                cols: cols,
                groupSize: groupSize
            ),
            weights: [
                SmeltPlannedKernelWeight(
                    weightName: firstWeightName,
                    role: .first
                ),
                SmeltPlannedKernelWeight(
                    weightName: secondWeightName,
                    role: .second
                ),
            ]
        )
    }

    static func qProjWeight(layerIndex: Int) -> String {
        "\(attentionPrefix(layerIndex: layerIndex))_q_proj_weight"
    }

    static func kProjWeight(layerIndex: Int) -> String {
        "\(attentionPrefix(layerIndex: layerIndex))_k_proj_weight"
    }

    static func vProjWeight(layerIndex: Int) -> String {
        "\(attentionPrefix(layerIndex: layerIndex))_v_proj_weight"
    }

    static func oProjWeight(layerIndex: Int) -> String {
        "\(attentionPrefix(layerIndex: layerIndex))_o_proj_weight"
    }

    static func qProjBiasDecodeConsumerID(layerIndex: Int) -> String {
        "\(attentionPrefix(layerIndex: layerIndex)).q_proj.bias.decode"
    }

    static func kProjBiasDecodeConsumerID(layerIndex: Int) -> String {
        "\(attentionPrefix(layerIndex: layerIndex)).k_proj.bias.decode"
    }

    static func vProjBiasDecodeConsumerID(layerIndex: Int) -> String {
        "\(attentionPrefix(layerIndex: layerIndex)).v_proj.bias.decode"
    }

    static func kvProjBiasDecodeConsumerID(layerIndex: Int) -> String {
        "\(attentionPrefix(layerIndex: layerIndex)).kv_proj.bias.decode"
    }

    static func attentionOutputResidualDecodeConsumerID(layerIndex: Int) -> String {
        "\(attentionPrefix(layerIndex: layerIndex)).o_proj.residual.decode"
    }

    static func ffnGateWeight(layerIndex: Int) -> String {
        "\(ffnPrefix(layerIndex: layerIndex))_gate_proj_weight"
    }

    static func ffnUpWeight(layerIndex: Int) -> String {
        "\(ffnPrefix(layerIndex: layerIndex))_up_proj_weight"
    }

    static func ffnDownWeight(layerIndex: Int) -> String {
        "\(ffnPrefix(layerIndex: layerIndex))_down_proj_weight"
    }

    static func ffnGateUpPrefillConsumerID(layerIndex: Int) -> String {
        "\(ffnPrefix(layerIndex: layerIndex)).gate_up.prefill"
    }

    static func ffnDownPrefillConsumerID(layerIndex: Int) -> String {
        "\(ffnPrefix(layerIndex: layerIndex)).down_proj.prefill"
    }

    static func ffnDownResidualDecodeConsumerID(layerIndex: Int) -> String {
        "\(ffnPrefix(layerIndex: layerIndex)).down_proj.residual.decode"
    }

    static func qProjAffineWeight(layerIndex: Int) -> SmeltPlannedKernelWeight {
        SmeltPlannedKernelWeight(
            weightName: qProjWeight(layerIndex: layerIndex),
            role: .affine
        )
    }

    static func kProjAffineWeight(layerIndex: Int) -> SmeltPlannedKernelWeight {
        SmeltPlannedKernelWeight(
            weightName: kProjWeight(layerIndex: layerIndex),
            role: .affine
        )
    }

    static func vProjAffineWeight(layerIndex: Int) -> SmeltPlannedKernelWeight {
        SmeltPlannedKernelWeight(
            weightName: vProjWeight(layerIndex: layerIndex),
            role: .affine
        )
    }

    static func kvProjAffineWeights(layerIndex: Int) -> [SmeltPlannedKernelWeight] {
        [
            SmeltPlannedKernelWeight(
                weightName: kProjWeight(layerIndex: layerIndex),
                role: .key
            ),
            SmeltPlannedKernelWeight(
                weightName: vProjWeight(layerIndex: layerIndex),
                role: .value
            ),
        ]
    }

    static func oProjAffineWeight(layerIndex: Int) -> SmeltPlannedKernelWeight {
        SmeltPlannedKernelWeight(
            weightName: oProjWeight(layerIndex: layerIndex),
            role: .affine
        )
    }

    static func ffnDownAffineWeight(layerIndex: Int) -> SmeltPlannedKernelWeight {
        SmeltPlannedKernelWeight(
            weightName: ffnDownWeight(layerIndex: layerIndex),
            role: .affine
        )
    }

    static func ffnGateUpWeights(layerIndex: Int) -> [SmeltPlannedKernelWeight] {
        [
            SmeltPlannedKernelWeight(
                weightName: ffnGateWeight(layerIndex: layerIndex),
                role: .gate
            ),
            SmeltPlannedKernelWeight(
                weightName: ffnUpWeight(layerIndex: layerIndex),
                role: .up
            ),
        ]
    }

    static func candidateKinds(
        for context: SmeltKernelLayerConsumerContext
    ) -> [SmeltKernelConsumerKind] {
        generatedConsumerDescriptors
            .filter { $0.isAvailable(in: context) }
            .map(\.kind)
    }

    static func candidates(
        for context: SmeltKernelLayerConsumerContext
    ) -> [SmeltPlannedKernelCandidate] {
        generatedConsumerDescriptors.compactMap { $0.candidate(in: context) }
    }

    static func candidate(
        kind: SmeltKernelConsumerKind,
        context: SmeltKernelLayerConsumerContext
    ) -> SmeltPlannedKernelCandidate? {
        descriptor(for: kind).candidate(in: context)
    }
}
