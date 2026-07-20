import Foundation
import SmeltSchema
import Testing

@Suite("Generic module runtime binding")
struct SmeltModuleRuntimeBindingPlanTests {
    @Test("Qwen vision and text blocks bind from the selected multimodal flow")
    func qwenVisionBindsAsLegoBricks() throws {
        let module = try Self.loadModule()
        let plan = try Self.bindingPlan(for: module)

        #expect(plan.executionShape == .iterative)
        #expect(plan.phases.map(\.role) == ["setup", "step"])
        #expect(plan.phases.map { $0.invocations.map(\.bindingKey) } == [
            [
                "native:text-tokenizer",
                "native:media-preprocessor",
                "compiled:transformer-encoder",
                "compiled:adapter",
                "native:visual-token-fusion",
            ],
            ["compiled:transformer", "native:sampler"],
        ])
        #expect(plan.emittedNodeIDs == ["multimodal-detokenizer"])
        #expect(plan.regionContract.nodeBindings.contains {
            $0.hasSuffix("=compiled:transformer-encoder")
        })
        #expect(plan.regionContract.nodeBindings.contains {
            $0.hasSuffix("=compiled:adapter")
        })
        #expect(plan.regionContract.nodeBindings.contains {
            $0.hasSuffix("=native:visual-token-fusion")
        })
    }

    @Test("Binding never selects on model id or semantic hash")
    func bindingIgnoresIdentity() throws {
        let originalModule = try Self.loadModule()
        let source = try String(contentsOf: Self.moduleURL, encoding: .utf8)
        let renamed = source.replacingOccurrences(
            of: "\"id\" : \"qwen35_4b\"",
            with: "\"id\" : \"renamed_multimodal_model\""
        )
        try #require(renamed != source)
        let renamedModule = try JSONDecoder().decode(
            SmeltCAMIR.self,
            from: Data(renamed.utf8)
        )

        let original = try Self.bindingPlan(for: originalModule)
        let drifted = try Self.bindingPlan(for: renamedModule)
        #expect(try originalModule.semanticSHA256()
            != renamedModule.semanticSHA256())
        #expect(drifted.executionShape == original.executionShape)
        #expect(drifted.phases == original.phases)
        #expect(drifted.feedbackEdges == original.feedbackEdges)
        #expect(drifted.regionContract == original.regionContract)
    }

    private static func bindingPlan(
        for module: SmeltCAMIR
    ) throws -> SmeltModuleRuntimeBindingPlan {
        let descriptor = try SmeltCAMPackageDescriptor(from: module)
        let capabilities = try SmeltCAMPackageCapabilities(descriptor: descriptor)
        return try SmeltModuleRuntimeBindingPlan(
            capabilities: capabilities,
            decision: capabilities.resolve(.runMultimodalText)
        )
    }

    private static func loadModule() throws -> SmeltCAMIR {
        let data = try Data(contentsOf: moduleURL)
        return try JSONDecoder().decode(SmeltCAMIR.self, from: data).validated()
    }

    private static var moduleURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent("qwen35_4b.module.json")
    }
}
