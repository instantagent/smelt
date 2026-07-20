import Foundation
import SmeltSchema
import Testing

@Suite("Inkling CAM topology")
struct InklingCAMTests {
    @Test("Pinned full model expresses every architecture brick")
    func exactFullTopology() throws {
        let module = try Self.load("inkling")
        let trunk = try #require(module.blocks.first { $0.id == "trunk" })
        let shape = try #require(trunk.shape.transformer)

        #expect(shape.hiddenSize == 6_144)
        #expect(shape.layers?.roles == [.sliding, .sliding, .sliding, .sliding, .sliding, .global])
        #expect(shape.layers?.repeatCount == 11)
        #expect(shape.layers?.roles.count == 6)
        #expect(shape.denseLayerCount == 2)
        #expect(shape.ffn?.dim == 24_576)
        #expect(shape.expert?.ffn.dim == 3_072)
        #expect(shape.router?.experts == 256)
        #expect(shape.router?.sharedExperts == 2)
        #expect(shape.router?.topK == 6)
        #expect(shape.router?.activation == .sigmoid)
        #expect(shape.router?.normalization == .selectedAndShared)
        #expect(shape.router?.scoreCorrectionBias == true)
        #expect(shape.router?.routeScale == "8")
        #expect(shape.router?.globalScale == true)
        #expect(shape.router?.sharedExpertSink == true)
        #expect(shape.vocab?.size == 201_024)
        #expect(shape.vocab?.tiedHead == false)

        let attention = try #require(shape.attentionByRole)
        let sliding = try #require(attention.first { $0.role == .sliding }?.attention)
        let global = try #require(attention.first { $0.role == .global }?.attention)
        #expect(sliding.qHeads == 64)
        #expect(sliding.kvHeads == 16)
        #expect(sliding.headDim == 128)
        #expect(sliding.window == 512)
        #expect(sliding.scaling == .inverseHeadDim)
        #expect(sliding.relativePosition?.projectionDim == 16)
        #expect(sliding.relativePosition?.extent == 512)
        #expect(global.kvHeads == 8)
        #expect(global.relativePosition?.extent == 1_024)
        #expect(global.relativePosition?.logScalingFloor == 128_000)
        #expect(global.relativePosition?.logScalingAlpha == "0.1")

        #expect(Set(shape.shortConvolutions?.map(\.site) ?? []) == Set([
            .attentionKey,
            .attentionValue,
            .attentionBranchOutput,
            .ffnBranchOutput,
        ]))
        #expect(shape.shortConvolutions?.allSatisfy {
            $0.kernelSize == 4 && $0.residual == .addInput
        } == true)
        #expect(shape.projectionBanks?.first { $0.id == "attention-input" }?.outputs == [
            .attentionQ, .attentionK, .attentionV, .attentionRelative,
        ])

        let image = try #require(module.blocks.first { $0.id == "image-patch-encoder" })
        let audio = try #require(module.blocks.first { $0.id == "audio-encoder" })
        #expect(image.operatorName == .patchEncoder)
        #expect(audio.operatorName == .discreteAudioEncoder)
        #expect(Self.requirements(image)["fold-plan"] == "1x5x5,1x2x2,1x4x4,2x1x1")
        #expect(Self.requirements(image)["patch-size"] == "40")
        #expect(Self.requirements(audio)["embedding-shape"] == "1280x6144")
        #expect(Self.requirements(audio)["values-per-token"] == "16")

        let sources = Dictionary(uniqueKeysWithValues: module.sources.map { ($0.id, $0) })
        #expect(sources["weights"]?.revision == "e4aa5ee880fbb0d2c1a93b1e4a39f2d4b97eb28a")
        #expect(sources["weights-nvfp4"]?.revision == "1fa46988f638221367b5fdeee4e86d5c9882ae23")
        #expect(sources["reference"]?.revision == "28596623762cb409bb1c9234f04bfb1269b1ece1")
    }

    @Test("Multimodal blocks bind generically and ignore module identity")
    func genericRuntimeBinding() throws {
        let original = try Self.load("inkling")
        let source = try String(contentsOf: Self.url("inkling"), encoding: .utf8)
        let renamedSource = source.replacingOccurrences(
            of: "\"id\" : \"inkling\"",
            with: "\"id\" : \"arbitrary_composed_model\""
        )
        try #require(renamedSource != source)
        let renamed = try JSONDecoder().decode(
            SmeltCAMIR.self,
            from: Data(renamedSource.utf8)
        ).validated()

        let originalPlan = try Self.multimodalBindingPlan(original)
        let renamedPlan = try Self.multimodalBindingPlan(renamed)
        #expect(originalPlan.phases.map { $0.invocations.map(\.bindingKey) } == [
            [
                "native:prompt-renderer",
                "native:text-tokenizer",
                "native:media-preprocessor",
                "compiled:patch-encoder",
                "compiled:discrete-audio-encoder",
                "native:multimodal-token-fusion",
            ],
            ["compiled:transformer", "native:sampler"],
        ])
        #expect(originalPlan.regionContract == renamedPlan.regionContract)
        #expect(originalPlan.phases == renamedPlan.phases)
        #expect(try original.semanticSHA256() != renamed.semanticSHA256())
    }

    @Test("MTP remains an independently composable exact-topology brick")
    func mtpIsIndependentBrick() throws {
        let module = try Self.load("inkling_mtp")
        let mtp = try #require(module.blocks.first { $0.id == "mtp" })
        let shape = try #require(mtp.shape.transformer)

        #expect(shape.layers?.roles == [
            .sliding, .global, .sliding, .global,
            .sliding, .sliding, .sliding, .sliding,
        ])
        #expect(shape.layers?.repeatCount == 1)
        #expect(shape.ffn?.dim == 24_576)
        #expect(shape.router == nil)
        #expect(Self.requirements(mtp)["input-projection-shape"] == "6144x12288")
        #expect(Self.requirements(mtp)["output-unembedding"] == "base-module")
        #expect(module.tensors.map(\.selector.pattern) == ["model.mtp.*"])
        #expect(module.graphNodes.first?.block == "mtp")
    }

    private static func multimodalBindingPlan(
        _ module: SmeltCAMIR
    ) throws -> SmeltModuleRuntimeBindingPlan {
        let descriptor = try SmeltCAMPackageDescriptor(from: module)
        let capabilities = try SmeltCAMPackageCapabilities(descriptor: descriptor)
        return try SmeltModuleRuntimeBindingPlan(
            capabilities: capabilities,
            decision: capabilities.resolve(.runMultimodalText)
        )
    }

    private static func requirements(_ block: SmeltCAMIR.Block) -> [String: String] {
        Dictionary(uniqueKeysWithValues: block.shape.requirements.compactMap { requirement in
            requirement.value.map { (requirement.key, $0) }
        })
    }

    private static func load(_ id: String) throws -> SmeltCAMIR {
        try SmeltCAMIR.decodeModule(at: url(id))
    }

    private static func url(_ id: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent("\(id).module.json")
    }
}
