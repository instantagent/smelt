/// Canonical package-relative names for immutable state produced during
/// package compilation. The package inventory is the declaration; there is no
/// mutable secondary manifest.
public enum SmeltPreparedArtifacts {
    public static let prefixMetadata = "prepared_prefix.json"
    public static let prefixSnapshot = "prepared_prefix.snapshot"
    public static let promptsMetadata = "prepared_prompts.json"
    public static let grammarMetadata = "compiled_grammar.json"
    public static let grammarTrie = "compiled_grammar.trie"
}
