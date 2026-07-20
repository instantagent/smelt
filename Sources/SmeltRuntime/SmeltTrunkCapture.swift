/// Exactness-only layer boundaries from one compiled dense-trunk execution.
public struct SmeltTrunkLayerCapture: Sendable {
    /// Hidden state emitted by each dense block in execution order.
    public let layerOutputs: [[Float]]

    /// Final normalized hidden state emitted by the trunk output port.
    public let finalHiddenStates: [Float]

    /// Creates a capture from per-layer and final hidden-state snapshots.
    public init(layerOutputs: [[Float]], finalHiddenStates: [Float]) {
        self.layerOutputs = layerOutputs
        self.finalHiddenStates = finalHiddenStates
    }
}

/// One model-agnostic exactness probe at a dense-trunk dispatch boundary.
struct SmeltTrunkDispatchCaptureRequest: Sendable {
    enum Source: Sendable {
        case current
        case alternate
        case slot(String)
        case slotOffset(String, elementOffset: Int)
    }

    let label: String
    let afterDispatch: Int
    let source: Source
    let count: Int
}
