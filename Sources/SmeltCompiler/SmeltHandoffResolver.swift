// SmeltHandoffResolver — Resolves prefill handoff families to per-layer slot mappings.
//
// Takes SmeltPrefillConfig.handoffFamilies (e.g. ["conv_state", "rec_state", ...])
// and expands them into concrete SmeltResolvedHandoff entries using the buffer plan.
//
// This is a compile-time operation — the resolved table is serialized into the manifest.

/// Resolves handoff family names to concrete slot mappings.
public struct SmeltHandoffResolver {

    /// Resolve handoff families into per-layer slot mappings.
    ///
    /// - Parameters:
    ///   - families: Family names from SmeltPrefillConfig (e.g. "conv_state", "rope").
    ///   - ir: Validated model IR.
    ///   - plan: Buffer plan with slot indices.
    /// - Returns: Complete handoff table for the manifest.
    public static func resolve(
        families: [String],
        ir: SmeltModelIR,
        plan: SmeltBufferPlan
    ) -> SmeltHandoffTable {
        let numDelta = ir.numDeltaLayers
        let numAttn = ir.numAttnLayers
        let dynamicContextElements = ir.usesDynamicContext ? 0 : nil

        var entries: [SmeltResolvedHandoff] = []

        for family in families {
            switch family {
            case "conv_state":
                guard let delta = ir.config.delta else { continue }
                let elemCount = delta.qkvDim * delta.convKernel
                for idx in 0..<numDelta {
                    entries.append(SmeltResolvedHandoff(
                        tensorName: "conv_state_\(idx)_out",
                        slotIndex: plan.convStateBaseSlot + idx,
                        expectedElements: elemCount
                    ))
                }

            case "rec_state":
                guard let delta = ir.config.delta else { continue }
                let elemCount = delta.numHeads * delta.headDim * delta.headDim
                for idx in 0..<numDelta {
                    entries.append(SmeltResolvedHandoff(
                        tensorName: "rec_state_\(idx)_out",
                        slotIndex: plan.recStateBaseSlot + idx,
                        expectedElements: elemCount
                    ))
                }

            case "key_cache":
                guard let attn = ir.config.attention else { continue }
                let elemCount =
                    dynamicContextElements
                    ?? attn.kvHeads * ir.compiledSeqCapacity * attn.headDim
                for idx in 0..<numAttn {
                    entries.append(SmeltResolvedHandoff(
                        tensorName: "key_cache_\(idx)_out",
                        slotIndex: plan.keyCacheBaseSlot + idx,
                        expectedElements: elemCount
                    ))
                }

            case "value_cache":
                guard let attn = ir.config.attention else { continue }
                let elemCount =
                    dynamicContextElements
                    ?? attn.kvHeads * ir.compiledSeqCapacity * attn.headDim
                for idx in 0..<numAttn {
                    entries.append(SmeltResolvedHandoff(
                        tensorName: "value_cache_\(idx)_out",
                        slotIndex: plan.valCacheBaseSlot + idx,
                        expectedElements: elemCount
                    ))
                }

            case "rope":
                // RoPE is handled separately via ropeCosSlot/ropeSinSlot
                break

            default:
                // Unknown family — skip (validation should catch this earlier)
                break
            }
        }

        return SmeltHandoffTable(
            entries: entries,
            ropeCosSlot: plan.ropeCosSlot,
            ropeSinSlot: plan.ropeSinSlot
        )
    }
}
