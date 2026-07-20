import SmeltRuntime
import SmeltSchema

/// One-request text API for downstream Smelt consumers.
///
/// This is the same admitted CAM text path used by `smelt run`: the package
/// owns route selection, prompt adaptation, template, thinking, stop, and
/// maximum-generation policy. Consumers provide semantic input and optional
/// sampling policy without importing or spawning the `smelt` executable.
public final class SmeltTextGenerator: @unchecked Sendable {
    public struct Response: Sendable {
        public let text: String
        public let tokenIDs: [Int32]
        public let promptTokenCount: Int
        public let prefillTime: Double
        public let generateTime: Double

        public init(
            text: String,
            tokenIDs: [Int32],
            promptTokenCount: Int,
            prefillTime: Double,
            generateTime: Double
        ) {
            self.text = text
            self.tokenIDs = tokenIDs
            self.promptTokenCount = promptTokenCount
            self.prefillTime = prefillTime
            self.generateTime = generateTime
        }
    }

    private let construction: CAMTextRuntimeConstruction
    private let tokenizer: SmeltTokenizer
    private let model: SmeltModel
    private let template: String
    private let thinkingPolicy: SmeltThinkingPolicy

    public init(packagePath: String, contextLimit: Int? = nil) throws {
        let admission = try SmeltRuntimeAdmission.resolve(
            packagePath: packagePath,
            requests: [.runText]
        )
        let construction = try admission.makeTextConstruction()
        let (manifest, inference) = try construction.loadManifestConfig()
        self.construction = construction
        self.tokenizer = try construction.makeTokenizer()
        self.model = try construction.makeModel(
            contextLimit: contextLimit,
            manifest: manifest
        )
        self.template = try construction.resolveTemplate(cliOverride: nil)
        self.thinkingPolicy = resolvedThinkingPolicy(inference)
    }

    public func generate(
        prompt: String,
        systemPrompt: String = "",
        maxTokens: Int = 512,
        selectionMode: SmeltSelectionMode = .argmax
    ) throws -> Response {
        guard maxTokens > 0 else {
            throw SmeltTextGeneratorError.invalidMaxTokens(maxTokens)
        }
        let rendered = try construction.renderPrompt(
            prompt: prompt,
            systemPrompt: systemPrompt
        )
        var inputIDs = try buildInputIds(
            prompt: rendered.prompt,
            tokenizer: tokenizer,
            template: template,
            thinkingPolicy: thinkingPolicy
        )
        if !rendered.systemPrompt.isEmpty {
            inputIDs = try buildSystemIds(
                systemPrompt: rendered.systemPrompt,
                tokenizer: tokenizer,
                template: template
            ) + inputIDs
        } else if let baked = model.bakedPrefixTokenIds {
            inputIDs = buildInputIdsApplyingBakedPrefix(
                prompt: rendered.prompt,
                tokenizer: tokenizer,
                unbakedInputIds: inputIDs,
                bakedPrefixTokenIds: baked,
                continuation: model.bakedPrefixContinuation,
                template: template,
                thinkingPolicy: thinkingPolicy
            )
        }

        let effectiveMaxTokens = construction.effectiveMaxTokens(maxTokens)
        var emitted = 0
        let result = try model.generate(
            tokenIds: inputIDs,
            selectionMode: selectionMode
        ) { _ in
            emitted += 1
            return emitted < effectiveMaxTokens
        }
        return Response(
            text: tokenizer.decode(result.tokens),
            tokenIDs: result.tokens,
            promptTokenCount: inputIDs.count,
            prefillTime: result.prefillTime,
            generateTime: result.generateTime
        )
    }
}

public enum SmeltTextGeneratorError: Error, CustomStringConvertible, Equatable {
    case invalidMaxTokens(Int)

    public var description: String {
        switch self {
        case .invalidMaxTokens(let value):
            return "maxTokens must be positive, got \(value)"
        }
    }
}
