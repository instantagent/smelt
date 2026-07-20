// Canonical Smelt tensor names. These strings are part of the
// on-disk manifest contract: builders look them up via
// requireWeight(name:), runtime callers reference them by name,
// gates check matches against them. A typo at any one site is a
// silent build break (manifest entry missing or routing wrong
// codepath); centralising the literals here both documents
// intent and lets a future codebase-wide sweep find every
// reference.
//
// Today these are the two embed-tensor names used by the
// TurboQuant-H gate, the writer's pattern matcher, the per-layer
// embed dispatch in both emitters, and the runtime's embedToken
// path. Adding new canonical names (e.g. for FFN or attn matvec
// targets) belongs here.

public enum SmeltCanonicalTensorNames {
    /// Decode-time / prefill-time token embedding table.
    /// Shape: [vocab_size, hidden_size].
    public static let embedTokens = "embed_tokens"
    /// AltUp per-layer input embedding table.
    /// Shape: [vocab_size_per_layer_input, num_layers * hidden_size_per_layer_input].
    /// Distinct from `embedTokens` despite the substring overlap.
    public static let embedTokensPerLayer = "embed_tokens_per_layer"
    /// Cluster-embedder token-ordering permutation buffer (int32).
    /// Shape: [vocab_size]. Groups vocab tokens into contiguous clusters.
    public static let maskedEmbeddingTokenOrdering = "masked_embedding_token_ordering"
}
