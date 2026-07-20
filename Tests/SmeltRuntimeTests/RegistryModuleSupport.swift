// RegistryModuleSupport — grammar-free access to the authored module IR for
// SmeltRuntimeTests. Mirrors the SmeltCompilerTests helper of the same name so
// runtime tests can consume `SmeltModels` values instead of parsing `.cam`.

import Foundation
import SmeltModels
import SmeltSchema

/// The authored registry IR for a fixture, keyed by module id (a `.cam` or
/// `.module.json` suffix is tolerated). Byte-identical to the grammar parser's
/// output for `Examples/CAM/<id>.cam` — `ModuleAuthoringParityTests` pins that
/// equality — so this is a drop-in for tests that previously loaded the grammar
/// fixture.
func registryModuleIR(_ name: String) -> SmeltCAMIR {
    let id = name.hasSuffix(".module.json") ? String(name.dropLast(".module.json".count))
        : name.hasSuffix(".cam") ? String(name.dropLast(".cam".count)) : name
    guard let ir = SmeltModels.definition(id: id) else {
        preconditionFailure("no authored registry module for '\(name)' (id '\(id)')")
    }
    return ir
}

/// Apply a JSON-value mutation to an authored module IR and re-decode it — the
/// grammar-free equivalent of loading a fixture, editing its `.cam` text, and
/// reparsing. The module JSON *is* the lowered IR, so editing the lowered field
/// reproduces the same drift the text edit encoded.
func mutatedModuleIR(
    _ base: SmeltCAMIR,
    _ mutate: (inout [String: Any]) throws -> Void
) throws -> SmeltCAMIR {
    let data = try base.canonicalJSONData()
    guard var object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        preconditionFailure("module IR did not encode to a JSON object")
    }
    try mutate(&object)
    let mutatedData = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    return try JSONDecoder().decode(SmeltCAMIR.self, from: mutatedData).validated()
}
