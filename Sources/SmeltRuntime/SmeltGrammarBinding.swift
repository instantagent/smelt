import Foundation

// Runtime-bound grammar parameters. A baked JSON schema may declare bindable
// choice slots: an array element that is exactly the string "$bind:NAME",
// e.g. `"enum": ["$bind:routes"]`. At invocation,
// `smelt run --bind routes=billing,auth,infra` splices the bound values into
// that array, so one baked package constrains output to a choice set that is
// not known until run time. The placeholder string is itself a valid enum
// literal, so bake-time schema validation needs no special casing; the
// expensive baked artifact (the token trie) is vocab-bound and unaffected.

public enum SmeltGrammarBindingError: Error, CustomStringConvertible {
    /// The schema declares slots that the invocation did not bind.
    case unboundPlaceholders([String])
    /// The invocation bound names the schema does not declare.
    case unknownBindings([String])
    /// A "$bind:NAME" string sits outside an array — only enum/array
    /// positions are bindable.
    case placeholderOutsideArray([String])
    case malformedSchema(String)

    public var description: String {
        switch self {
        case .unboundPlaceholders(let names):
            return "grammar has unbound slots: \(names.joined(separator: ", "))"
                + " — bind them with --bind NAME=V1,V2,..."
        case .unknownBindings(let names):
            return "no such grammar slot: \(names.joined(separator: ", "))"
        case .placeholderOutsideArray(let names):
            return "grammar placeholder outside an array position: "
                + names.joined(separator: ", ")
                + " — only enum/array elements are bindable"
        case .malformedSchema(let message):
            return "grammar is not valid JSON: \(message)"
        }
    }
}

public enum SmeltGrammarBinding {
    public static let placeholderPrefix = "$bind:"

    /// Slot names declared by the schema, in first-appearance order.
    /// Throws when a placeholder sits outside an array position. Always
    /// parses — a raw-substring scan would miss JSON-escaped placeholders
    /// (e.g. the dollar sign written as a \u escape).
    public static func placeholders(inJSONSchema schema: String) throws -> [String] {
        var walk = Walk()
        _ = try walk.substitute(parse(schema), bindings: [:])
        guard walk.misplaced.isEmpty else {
            throw SmeltGrammarBindingError.placeholderOutsideArray(walk.misplaced)
        }
        return walk.found
    }

    /// Splice `bindings` into the schema's "$bind:NAME" slots. Every slot
    /// must be bound and every binding must match a slot; a schema without
    /// slots passes through byte-identical when `bindings` is empty.
    public static func apply(
        bindings: [String: [String]], toJSONSchema schema: String
    ) throws -> String {
        var walk = Walk()
        let substituted = try walk.substitute(parse(schema), bindings: bindings)
        guard walk.misplaced.isEmpty else {
            throw SmeltGrammarBindingError.placeholderOutsideArray(walk.misplaced)
        }
        let unbound = walk.found.filter { bindings[$0] == nil }
        guard unbound.isEmpty else {
            throw SmeltGrammarBindingError.unboundPlaceholders(unbound)
        }
        let unknown = bindings.keys.filter { !walk.found.contains($0) }.sorted()
        guard unknown.isEmpty else {
            throw SmeltGrammarBindingError.unknownBindings(unknown)
        }
        // No bindings and no slots: return the original string so a slotless
        // package's schema is never perturbed by re-serialization.
        if bindings.isEmpty {
            return schema
        }
        let data = try JSONSerialization.data(
            withJSONObject: substituted, options: [.sortedKeys]
        )
        guard let result = String(data: data, encoding: .utf8) else {
            throw SmeltGrammarBindingError.malformedSchema(
                "re-serialization produced non-UTF-8 output"
            )
        }
        return result
    }

    private static func parse(_ schema: String) throws -> Any {
        do {
            return try JSONSerialization.jsonObject(
                with: Data(schema.utf8), options: [.fragmentsAllowed]
            )
        } catch {
            throw SmeltGrammarBindingError.malformedSchema("\(error)")
        }
    }

    /// A placeholder is a string exactly equal to "$bind:NAME" — a substring
    /// occurrence (say, inside a description) is not a slot.
    private static func placeholderName(_ value: Any) -> String? {
        guard let string = value as? String,
              string.hasPrefix(placeholderPrefix)
        else { return nil }
        let name = String(string.dropFirst(placeholderPrefix.count))
        return name.isEmpty ? nil : name
    }

    private struct Walk {
        var found: [String] = []
        var misplaced: [String] = []

        mutating func substitute(
            _ node: Any, bindings: [String: [String]]
        ) throws -> Any {
            if let array = node as? [Any] {
                var result: [Any] = []
                for element in array {
                    if let name = SmeltGrammarBinding.placeholderName(element) {
                        if !found.contains(name) { found.append(name) }
                        if let values = bindings[name] {
                            result.append(contentsOf: values)
                        } else {
                            result.append(element)
                        }
                    } else {
                        result.append(try substitute(element, bindings: bindings))
                    }
                }
                return result
            }
            if let object = node as? [String: Any] {
                var result: [String: Any] = [:]
                for (key, value) in object {
                    if let name = SmeltGrammarBinding.placeholderName(key) {
                        if !misplaced.contains(name) { misplaced.append(name) }
                    }
                    result[key] = try substitute(value, bindings: bindings)
                }
                return result
            }
            if let name = SmeltGrammarBinding.placeholderName(node) {
                if !misplaced.contains(name) { misplaced.append(name) }
            }
            return node
        }
    }
}
