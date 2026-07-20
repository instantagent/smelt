// SmeltDrafterTaps — Read-side handles a target runtime exposes for
// an EAGLE-class drafter to consume. Bridges the architectural gap
// between the target's `decodeStep` (which writes the post-final-
// norm hidden state and last-layer K/V cache as side effects) and
// the drafter's `external_kv` attention (which reads them as inputs).
//
// Architecture-agnostic: the K/V taps are keyed by attention family
// name from the manifest's expanded layer pattern. Gemma 4 has both
// "sliding" and "global"; Llama / Qwen / Mistral drafters tap a
// single "attn" family; future architectures plug in by adding a
// new family name without touching this surface.
//
// Layout responsibility:
// - The hidden-state tap points at `normOutBuf` (slot 8). Smelt's
//   final-norm dispatch writes the post-norm hidden state into that
//   slot on every text model — both the fused norm+matvec production
//   path and the cluster-embedder drafter path.
// - Each family entry points at the last attention layer of that
//   family that owns its K/V cache slot (i.e., is not in the
//   trailing kv-shared region for Gemma 4 E2B/E4B targets).

import Foundation
import Metal
import SmeltSchema

/// Canonical attention-family names that count as attention layers
/// in `expandedLayerPattern` walking. Drafter taps key on these;
/// `lastAttentionLayerIndex` advances its counter when a layer
/// matches one of these names. Adding a new attention family is a
/// single-line edit here — both walkers reference the same set.
///
/// Gemma 4 uses sliding/global; Llama / Qwen / Mistral use attn.
/// Storage on `SmeltDrafterTaps.attention` stays an open
/// `[String: SmeltDrafterAttentionTap]` so an architecture-specific
/// drafter can stash a non-canonical key without touching this set,
/// but pattern-walking only counts what's listed here.
public enum SmeltAttentionFamily {
    public static let known: Set<String> = ["sliding", "global", "attn"]
}

/// Per-attention-family K/V cache reference plus the shape needed
/// to bind it as an external_kv input.
public struct SmeltDrafterAttentionTap {
    public let keyCache: MTLBuffer
    public let valueCache: MTLBuffer
    public let kvHeads: Int
    public let headDim: Int
    /// Currently-allocated context capacity in tokens. The cache is
    /// laid out as `[kvHeads, contextCapacity, headDim]` fp16.
    /// Drafter dispatches must use this as the seq stride.
    public let contextCapacity: Int

    public init(
        keyCache: MTLBuffer,
        valueCache: MTLBuffer,
        kvHeads: Int,
        headDim: Int,
        contextCapacity: Int
    ) {
        self.keyCache = keyCache
        self.valueCache = valueCache
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.contextCapacity = contextCapacity
    }
}

/// Bundle of read-side handles a target runtime exposes for drafter
/// consumption. Returned by `SmeltRuntime.drafterTaps()`.
public struct SmeltDrafterTaps {
    /// Post-final-norm hidden state of the target's most recent
    /// decode step. Decode-only models lay this out as
    /// `[hiddenSize]` fp16 at offset 0. Metal-prefill models lay
    /// it out as `[B, hiddenSize]` fp16 — the most-recent token's
    /// hidden lives at `(seqLen - 1) * hiddenSize * 2` bytes after
    /// a chunked prefill, not at the buffer head. Phase 5 will
    /// thread the active offset through; this surface returns the
    /// buffer head.
    public let lastHiddenState: MTLBuffer
    public let hiddenSize: Int

    /// Per-family last-layer K/V references. Keys are attention
    /// family names from the manifest's layer pattern:
    ///   - Gemma 4 (sliding/global mix): `["sliding", "global"]`
    ///   - Llama / Qwen / Mistral: `["attn"]`
    ///   - Future split-attention architectures plug in by name.
    public let attention: [String: SmeltDrafterAttentionTap]

    public init(
        lastHiddenState: MTLBuffer,
        hiddenSize: Int,
        attention: [String: SmeltDrafterAttentionTap]
    ) {
        self.lastHiddenState = lastHiddenState
        self.hiddenSize = hiddenSize
        self.attention = attention
    }
}

/// Sugar for the canonical family names in
/// `SmeltAttentionFamily.known`. Equivalent to
/// `attention["<name>"]`; use the dictionary directly for any
/// non-canonical family.
extension SmeltDrafterTaps {
    public var sliding: SmeltDrafterAttentionTap? { attention["sliding"] }
    public var global: SmeltDrafterAttentionTap? { attention["global"] }
    public var attn: SmeltDrafterAttentionTap? { attention["attn"] }
}

/// Find the attention-layer index of the last layer in an expanded
/// pattern that matches the given family AND owns its K/V cache slot.
///
/// `sharedKVLayers` is the trailing layer count whose K/V cache slot
/// is allocated but never written — those layers cross-attend to an
/// earlier non-shared layer. Returning the shared layer's own slot
/// would hand back zeros (Gemma 4 E2B / E4B trip this), so we walk
/// back into the non-shared region instead. Pass 0 for non-Gemma or
/// Gemma packages without kv sharing.
///
/// Returns nil when no non-shared layer of the family exists.
public func lastAttentionLayerIndex(
    forFamily family: String,
    in expandedPattern: [String],
    sharedKVLayers: Int = 0
) -> Int? {
    let firstSharedLayer = sharedKVLayers > 0
        ? expandedPattern.count - sharedKVLayers
        : expandedPattern.count
    var attnIdx = -1
    var lastMatch: Int?
    for (layerIdx, layer) in expandedPattern.enumerated() {
        guard SmeltAttentionFamily.known.contains(layer) else { continue }
        attnIdx += 1
        if layer == family, layerIdx < firstSharedLayer {
            lastMatch = attnIdx
        }
    }
    return lastMatch
}

/// Walk an expanded layer pattern and return distinct attention
/// family names in first-seen order. Skips non-attention layers
/// (e.g., "delta" recurrent layers) so a runtime caller can
/// iterate exactly the families that appear and resolve a tap
/// for each.
public func attentionFamilies(in expandedPattern: [String]) -> [String] {
    var seen: [String] = []
    for layer in expandedPattern where SmeltAttentionFamily.known.contains(layer) {
        if !seen.contains(layer) { seen.append(layer) }
    }
    return seen
}
