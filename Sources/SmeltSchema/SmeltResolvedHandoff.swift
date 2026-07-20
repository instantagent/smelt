// SmeltResolvedHandoff — Resolved prefill state handoff mappings.
//
// Produced by the compiler from SmeltPrefillConfig.handoffFamilies + SmeltBufferPlan.
// Serialized into the manifest so the runtime knows where to copy CoreML outputs.
//
// Each entry maps a CoreML output tensor name to a specific buffer slot
// with expected shape for validation.

// MARK: - Resolved handoff

/// One resolved mapping from a CoreML output tensor to a Metal buffer slot.
public struct SmeltResolvedHandoff: Codable, Sendable {
    /// CoreML output tensor name (e.g. "conv_state_0_out").
    public let tensorName: String
    /// Target buffer slot index (from SmeltBufferPlan).
    public let slotIndex: Int
    /// Expected element count for size validation.
    public let expectedElements: Int
    /// Whether this state requires FP16→FP32 conversion (e.g. rec_state).
    public let convertFP16toFP32: Bool

    public init(
        tensorName: String,
        slotIndex: Int,
        expectedElements: Int,
        convertFP16toFP32: Bool = false
    ) {
        self.tensorName = tensorName
        self.slotIndex = slotIndex
        self.expectedElements = expectedElements
        self.convertFP16toFP32 = convertFP16toFP32
    }

    private enum CodingKeys: String, CodingKey {
        case tensorName = "tensor_name"
        case slotIndex = "slot_index"
        case expectedElements = "expected_elements"
        case convertFP16toFP32 = "convert_fp16_to_fp32"
    }
}

// MARK: - Resolved handoff table

/// Complete handoff table for the manifest.
public struct SmeltHandoffTable: Codable, Sendable {
    public let entries: [SmeltResolvedHandoff]
    /// Slot indices for RoPE tables (separate from per-layer state).
    public let ropeCosSlot: Int
    public let ropeSinSlot: Int

    public init(
        entries: [SmeltResolvedHandoff],
        ropeCosSlot: Int,
        ropeSinSlot: Int
    ) {
        self.entries = entries
        self.ropeCosSlot = ropeCosSlot
        self.ropeSinSlot = ropeSinSlot
    }

    private enum CodingKeys: String, CodingKey {
        case entries
        case ropeCosSlot = "rope_cos_slot"
        case ropeSinSlot = "rope_sin_slot"
    }
}
