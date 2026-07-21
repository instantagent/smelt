import Foundation
import Testing
import SmeltSchema

@testable import SmeltRuntime

private let routerSchema = #"""
{"type":"object","properties":{"route":{"type":"string","enum":["$bind:routes"]}},"required":["route"],"additionalProperties":false}
"""#

private func enumValues(in schema: String) throws -> [String] {
    let root = try JSONSerialization.jsonObject(with: Data(schema.utf8))
    let object = try #require(root as? [String: Any])
    let properties = try #require(object["properties"] as? [String: Any])
    let route = try #require(properties["route"] as? [String: Any])
    return try #require(route["enum"] as? [String])
}

@Test func bindingSplicesValuesIntoEnumSlot() throws {
    let bound = try SmeltGrammarBinding.apply(
        bindings: ["routes": ["billing", "auth", "infra"]],
        toJSONSchema: routerSchema
    )
    #expect(try enumValues(in: bound) == ["billing", "auth", "infra"])
    #expect(!bound.contains("$bind:"))
}

@Test func bindingPreservesFixedSiblingsInMixedEnum() throws {
    let schema = #"{"properties":{"route":{"enum":["fallback","$bind:routes"]}}}"#
    let bound = try SmeltGrammarBinding.apply(
        bindings: ["routes": ["billing", "auth"]], toJSONSchema: schema
    )
    #expect(try enumValues(in: bound) == ["fallback", "billing", "auth"])
}

@Test func unboundSlotThrows() {
    #expect(throws: SmeltGrammarBindingError.self) {
        try SmeltGrammarBinding.apply(bindings: [:], toJSONSchema: routerSchema)
    }
    do {
        _ = try SmeltGrammarBinding.apply(
            bindings: [:], toJSONSchema: routerSchema
        )
        Issue.record("expected unboundPlaceholders")
    } catch let SmeltGrammarBindingError.unboundPlaceholders(names) {
        #expect(names == ["routes"])
    } catch {
        Issue.record("unexpected error: \(error)")
    }
}

@Test func unknownBindingNameThrows() {
    do {
        _ = try SmeltGrammarBinding.apply(
            bindings: ["routes": ["a"], "destinations": ["b"]],
            toJSONSchema: routerSchema
        )
        Issue.record("expected unknownBindings")
    } catch let SmeltGrammarBindingError.unknownBindings(names) {
        #expect(names == ["destinations"])
    } catch {
        Issue.record("unexpected error: \(error)")
    }
}

@Test func bindingsOnSlotlessSchemaThrow() {
    let schema = #"{"type":"string","enum":["low","high"]}"#
    #expect(throws: SmeltGrammarBindingError.self) {
        try SmeltGrammarBinding.apply(
            bindings: ["routes": ["a"]], toJSONSchema: schema
        )
    }
}

@Test func slotlessSchemaPassesThroughUntouched() throws {
    let schema = #"{"type":"string","enum":["low","high"]}"#
    let bound = try SmeltGrammarBinding.apply(bindings: [:], toJSONSchema: schema)
    #expect(bound == schema)
}

@Test func placeholderOutsideArrayThrows() {
    let schema = #"{"properties":{"route":{"const":"$bind:routes"}}}"#
    #expect(throws: SmeltGrammarBindingError.self) {
        try SmeltGrammarBinding.placeholders(inJSONSchema: schema)
    }
    #expect(throws: SmeltGrammarBindingError.self) {
        try SmeltGrammarBinding.apply(
            bindings: ["routes": ["a"]], toJSONSchema: schema
        )
    }
}

@Test func placeholderSubstringInDescriptionIsNotASlot() throws {
    let schema = #"{"description":"slots use $bind:name syntax","enum":["a"]}"#
    #expect(try SmeltGrammarBinding.placeholders(inJSONSchema: schema) == [])
    let bound = try SmeltGrammarBinding.apply(bindings: [:], toJSONSchema: schema)
    #expect(bound.contains("$bind:name syntax"))
}

@Test func placeholdersReportsSlotsInAppearanceOrder() throws {
    let schema = #"""
    {"properties":{"route":{"enum":["$bind:routes"]},"priority":{"enum":["$bind:priorities"]}}}
    """#
    let slots = try SmeltGrammarBinding.placeholders(inJSONSchema: schema)
    #expect(Set(slots) == ["routes", "priorities"])
}

@Test func jsonEscapedPlaceholderIsStillASlot() throws {
    // The dollar sign spelled as a \u escape decodes to the same slot; a
    // raw-substring scan would miss this spelling and let the slot
    // flow into execution as a literal enum value.
    let schema = "{\"enum\":[\"\\u0024bind:routes\"]}"
    #expect(!schema.contains("$bind:"))
    #expect(try SmeltGrammarBinding.placeholders(inJSONSchema: schema) == ["routes"])
    #expect(throws: SmeltGrammarBindingError.self) {
        try SmeltGrammarBinding.apply(bindings: [:], toJSONSchema: schema)
    }
    let bound = try SmeltGrammarBinding.apply(
        bindings: ["routes": ["a", "b"]], toJSONSchema: schema
    )
    #expect(!bound.contains("bind:"))
}

@Test func placeholderAsObjectKeyThrows() {
    let schema = #"{"properties":{"$bind:routes":{"type":"string"}}}"#
    #expect(throws: SmeltGrammarBindingError.self) {
        try SmeltGrammarBinding.placeholders(inJSONSchema: schema)
    }
    #expect(throws: SmeltGrammarBindingError.self) {
        try SmeltGrammarBinding.apply(
            bindings: ["routes": ["a"]], toJSONSchema: schema
        )
    }
}

@Test func bindingPreservesHighPrecisionNumbers() throws {
    let schema = #"{"properties":{"x":{"const":0.30000000000000004},"route":{"enum":["$bind:routes"]}}}"#
    let bound = try SmeltGrammarBinding.apply(
        bindings: ["routes": ["a"]], toJSONSchema: schema
    )
    let root = try #require(
        try JSONSerialization.jsonObject(with: Data(bound.utf8)) as? [String: Any]
    )
    let properties = try #require(root["properties"] as? [String: Any])
    let x = try #require(properties["x"] as? [String: Any])
    let const = try #require(x["const"] as? Double)
    #expect(const == 0.30000000000000004)
}

@Test func malformedSchemaThrows() {
    #expect(throws: SmeltGrammarBindingError.self) {
        try SmeltGrammarBinding.apply(
            bindings: ["routes": ["a"]], toJSONSchema: #"{"enum": ["$bind:routes"#
        )
    }
}

// MARK: - Matcher integration (needs the Qwen 4B fixture + llguidance)

private func qwenFixtureAvailable() -> Bool {
    let fm = FileManager.default
    return fm.fileExists(atPath: "\(qwenFixturePackagePath())/tokenizer.json")
        && fm.fileExists(atPath: "third_party/llguidance/lib/libllguidance.a")
}

private func qwenFixturePackagePath() -> String {
    let env = ProcessInfo.processInfo.environment
    if let explicit = env["SMELT_QWEN_4B_PACKAGE"], !explicit.isEmpty {
        return explicit
    }
    return FileManager.default.currentDirectoryPath
        + "/artifacts/qwen35-4b-qmm16x128/Qwen_Qwen3.5-4B.smeltpkg"
}

@Test(.enabled(if: qwenFixtureAvailable(), "Build Qwen 4B and llguidance fixtures to run the integration test"))
func boundEnumConstrainsMaskToRuntimeRoutes() throws {
    let packagePath = qwenFixturePackagePath()
    let tokenizer = try SmeltTokenizer(path: "\(packagePath)/tokenizer.json")
    let manifestData = try Data(
        contentsOf: URL(fileURLWithPath: "\(packagePath)/manifest.json")
    )
    let manifest = try SmeltManifest.decode(from: manifestData)
    let llgTokenizer = try SmeltLLGuidanceTokenizer(
        tokenizer: tokenizer,
        eosTokens: manifest.inference?.eosTokens ?? []
    )

    let bound = try SmeltGrammarBinding.apply(
        bindings: ["routes": ["billing", "zebra"]], toJSONSchema: routerSchema
    )
    let matcher = try SmeltLLGuidanceMatcher(
        tokenizer: llgTokenizer, jsonSchema: bound
    )

    try matcher.consume(tokenIds: tokenizer.encode("{\"route\":\""))
    let mask = try matcher.computeMask()
    let billingFirst = try #require(tokenizer.encode("billing").first)
    let zebraFirst = try #require(tokenizer.encode("zebra").first)
    let outsideFirst = try #require(tokenizer.encode("queue").first)
    #expect(SmeltLLGuidanceMatcher.tokenIsAllowed(billingFirst, in: mask))
    #expect(SmeltLLGuidanceMatcher.tokenIsAllowed(zebraFirst, in: mask))
    #expect(!SmeltLLGuidanceMatcher.tokenIsAllowed(outsideFirst, in: mask))

    try matcher.consume(tokenIds: tokenizer.encode("billing\"}"))
    #expect(matcher.isAccepting)
}
