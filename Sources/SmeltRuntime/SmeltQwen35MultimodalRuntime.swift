import Foundation
import Metal

/// Runnable composition of a Qwen3.5 vision component and an existing text
/// component. Configuration is derived from the component's carried CAM graph;
/// paths decide placement only, never model behavior or compatibility.
public final class SmeltQwen35MultimodalRuntime {
    public struct Result: Sendable {
        public let generatedTokenIDs: [Int32]
        public let text: String
        public let promptTokenCount: Int
        public let visualTokenCount: Int
        public let grid: SmeltQwen35VisionRuntime.Grid
        public let ropeDelta: Int
        public let preprocessSeconds: Double
        public let visionSeconds: Double
        public let fusionSeconds: Double
        public let imageToFusedSeconds: Double
        public let textPrefillSeconds: Double
        public let decodeSeconds: Double
    }

    public let fusionConfig: SmeltQwen35MultimodalFusionConfig
    public let visionConfig: SmeltQwen35VisionConfig

    private let component: SmeltQwen35VisionArtifact
    private let preprocessor: SmeltQwen35ImagePreprocessor
    private let vision: SmeltQwen35VisionRuntime
    private let text: SmeltRuntime
    private let tokenizer: SmeltTokenizer
    private let eosTokenIDs: Set<Int32>
    private let contextLimit: Int

    public init(
        textComponentPath: String,
        visionComponentPath: String,
        contextLimit: Int = 4_096,
        verifyVisionComponent: Bool = true,
        device requestedDevice: MTLDevice? = nil
    ) throws {
        guard contextLimit > 1 else {
            throw SmeltQwen35MultimodalRuntimeError.invalidRequest(
                "contextLimit must be greater than one"
            )
        }
        let component = try SmeltQwen35VisionArtifact(
            path: visionComponentPath,
            verify: verifyVisionComponent
        )
        self.component = component
        fusionConfig = try SmeltQwen35MultimodalFusionConfig(module: component.module)
        let imageConfig = try SmeltQwen35ImagePreprocessorConfig(module: component.module)
        preprocessor = SmeltQwen35ImagePreprocessor(config: imageConfig)
        let checkpointPlan = try SmeltQwen35VisionCheckpointPlan(
            module: component.module,
            checkpoint: component
        )
        visionConfig = checkpointPlan.config

        guard let device = requestedDevice ?? MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else {
            throw SmeltQwen35MultimodalRuntimeError.metalUnavailable
        }
        let weights = try SmeltQwen35VisionWeights(
            device: device,
            checkpoint: component,
            plan: checkpointPlan
        )
        let pipelines = try SmeltQwen35VisionPipelines(
            device: device,
            library: component.makeLibrary(device: device)
        )
        vision = SmeltQwen35VisionRuntime(
            device: device,
            queue: queue,
            config: checkpointPlan.config,
            weights: weights,
            pipelines: pipelines
        )

        let text = try SmeltRuntime(
            packagePath: textComponentPath,
            device: device,
            contextLimit: contextLimit
        )
        guard text.manifest.config.hiddenSize == fusionConfig.hiddenSize,
              text.manifest.config.ropeDim == fusionConfig.mropeSections.reduce(0, +) * 2,
              text.manifest.slotLayout.ropeTablePairs.count == 1,
              text.manifest.slotLayout.ropeTablePairs[0].layout == "interleaved",
              text.manifest.slotLayout.ropeTablePairs[0].theta == fusionConfig.ropeTheta
        else {
            throw SmeltQwen35MultimodalRuntimeError.incompatibleTextComponent(
                "hidden size or RoPE contract differs from the selected fusion graph"
            )
        }
        self.text = text
        tokenizer = try SmeltTokenizer(path: "\(textComponentPath)/tokenizer.json")
        eosTokenIDs = Set(
            try text.manifest.resolvedInferencePolicy().inference.eosTokens
        )
        self.contextLimit = contextLimit
    }

    /// Generate from already-templated token prefix/suffix around one image.
    /// The API keeps chat templating outside the graph executor; typed media
    /// placement, visual placeholders, fusion, MRoPE, and decode are owned here.
    public func generate(
        imageAt imageURL: URL,
        prefixTokenIDs: [Int32]? = nil,
        suffixTokenIDs: [Int32]? = nil,
        maxTokens: Int = 32
    ) throws -> Result {
        guard maxTokens > 0 else {
            throw SmeltQwen35MultimodalRuntimeError.invalidRequest(
                "maxTokens must be positive"
            )
        }
        let preprocessStart = CFAbsoluteTimeGetCurrent()
        let image = try preprocessor.preprocess(imageAt: imageURL)
        let preprocessSeconds = CFAbsoluteTimeGetCurrent() - preprocessStart

        let visionStart = CFAbsoluteTimeGetCurrent()
        let visual = try vision.encode(
            patches: image.patches,
            grids: [image.grid],
            diagnoseNonFinite: false
        )
        let visualValues = visual.values()
        guard visualValues.allSatisfy(\.isFinite) else {
            throw SmeltQwen35MultimodalRuntimeError.nonFiniteVisionEmbeddings
        }
        let visionSeconds = CFAbsoluteTimeGetCurrent() - visionStart

        let prefix = prefixTokenIDs ?? [fusionConfig.visionStartTokenID]
        let suffix = suffixTokenIDs ?? [fusionConfig.visionEndTokenID]
        let prompt = prefix
            + [Int32](repeating: fusionConfig.imageTokenID, count: visual.tokenCount)
            + suffix
        guard prompt.count + maxTokens <= contextLimit else {
            throw SmeltQwen35MultimodalRuntimeError.invalidRequest(
                "prompt \(prompt.count) + maxTokens \(maxTokens) exceeds "
                    + "contextLimit \(contextLimit)"
            )
        }
        let positions = try SmeltQwen35MultimodalPositionPlan(
            tokenIDs: prompt,
            grids: [image.grid],
            config: fusionConfig
        )
        let rope = try positions.ropeTables(
            rowCount: prompt.count + maxTokens,
            ropeDim: text.manifest.config.ropeDim,
            config: fusionConfig
        )

        text.resetWorkingBuffers()
        try text.prepareForRequest(
            batchCapacity: min(prompt.count, text.maxPrefillBatchSize),
            contextCapacity: prompt.count + maxTokens
        )
        let fusionStart = CFAbsoluteTimeGetCurrent()
        let embeddings = try SmeltQwen35MultimodalFusion.embeddings(
            tokenIDs: prompt,
            visualEmbeddings: visualValues,
            runtime: text,
            config: fusionConfig
        )
        let fusionSeconds = CFAbsoluteTimeGetCurrent() - fusionStart
        let prefillStart = CFAbsoluteTimeGetCurrent()
        var current = try text.prefillEmbeddings(
            embeddings,
            tokenIds: prompt,
            ropeCos: rope.cos,
            ropeSin: rope.sin
        )
        let textPrefillSeconds = CFAbsoluteTimeGetCurrent() - prefillStart

        var generated: [Int32] = []
        generated.reserveCapacity(maxTokens)
        let decodeStart = CFAbsoluteTimeGetCurrent()
        while generated.count < maxTokens, !eosTokenIDs.contains(current) {
            generated.append(current)
            guard generated.count < maxTokens else { break }
            current = try text.decodeStep(
                tokenId: current,
                position: Int32(prompt.count + generated.count - 1)
            )
        }
        let decodeSeconds = CFAbsoluteTimeGetCurrent() - decodeStart
        guard text.allLogits().allSatisfy(\.isFinite) else {
            throw SmeltQwen35MultimodalRuntimeError.nonFiniteTextLogits
        }
        return Result(
            generatedTokenIDs: generated,
            text: tokenizer.decode(generated),
            promptTokenCount: prompt.count,
            visualTokenCount: visual.tokenCount,
            grid: image.grid,
            ropeDelta: positions.ropeDelta,
            preprocessSeconds: preprocessSeconds,
            visionSeconds: visionSeconds,
            fusionSeconds: fusionSeconds,
            imageToFusedSeconds: preprocessSeconds + visionSeconds + fusionSeconds,
            textPrefillSeconds: textPrefillSeconds,
            decodeSeconds: decodeSeconds
        )
    }
}

public enum SmeltQwen35MultimodalRuntimeError: Error, CustomStringConvertible {
    case invalidRequest(String)
    case incompatibleTextComponent(String)
    case metalUnavailable
    case nonFiniteVisionEmbeddings
    case nonFiniteTextLogits

    public var description: String {
        switch self {
        case .invalidRequest(let message):
            return "invalid Qwen3.5 multimodal request: \(message)"
        case .incompatibleTextComponent(let message):
            return "incompatible Qwen3.5 text component: \(message)"
        case .metalUnavailable:
            return "Qwen3.5 multimodal runtime could not create a Metal device/queue"
        case .nonFiniteVisionEmbeddings:
            return "Qwen3.5 multimodal runtime produced non-finite vision embeddings"
        case .nonFiniteTextLogits:
            return "Qwen3.5 multimodal runtime produced non-finite text logits"
        }
    }
}
