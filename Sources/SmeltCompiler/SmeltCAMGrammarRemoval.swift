import Foundation

/// Fail-closed guard for the deleted `.cam` authoring grammar.
///
/// The `.cam` text grammar was removed in Phase C of the module-authoring
/// migration (public-release Phase 2); models are authored as Swift values and
/// emitted to `<id>.module.json`. A user who still passes a `.cam` path to
/// `smelt build` or `smelt module check|admission|ir`
/// must be told the format is gone — otherwise the input falls into a JSON
/// decode that fails with a confusing "not valid JSON" error, or (worse) is
/// silently misinterpreted. This gateway centralizes the suffix check and the
/// diagnostic so every command fails closed with the same explicit message.
public enum SmeltCAMGrammarRemoval {
    /// Returns a fail-closed diagnostic if `path` is a removed `.cam` grammar
    /// input, otherwise `nil`. Callers exit non-zero on a non-nil result before
    /// attempting to decode the input.
    public static func rejectionDiagnostic(forInputPath path: String) -> String? {
        guard path.hasSuffix(".cam") else { return nil }
        return "the .cam authoring grammar was removed; author models as .module.json "
            + "(offending input: \(path))"
    }
}
