import SmeltRuntime
import SmeltServe

package enum AgentRunner {
    package static func run(
        _ artifact: AgentArtifact,
        prompt: String,
        maxTokens: Int = 512,
        temperature: Float? = nil,
        seed: UInt64 = 0
    ) throws -> AgentResponse {
        try AgentSession(artifact: artifact).run(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            seed: seed
        )
    }
}

package final class AgentSession: @unchecked Sendable {
    package let artifact: AgentArtifact
    private let generator: SmeltTextGenerator

    package init(artifact: AgentArtifact) throws {
        self.artifact = artifact
        let package = try artifact.manifest.resolveModel()
        self.generator = try SmeltTextGenerator(packagePath: package.packageURL.path)
    }

    package func run(
        prompt: String,
        maxTokens: Int = 512,
        temperature: Float? = nil,
        seed: UInt64 = 0
    ) throws -> AgentResponse {
        let selection: SmeltSelectionMode
        if let temperature, temperature > 0 {
            selection = .temperature(temperature, seed: seed)
        } else {
            selection = .argmax
        }
        return AgentResponse(smeltResponse: try generator.generate(
            prompt: prompt,
            systemPrompt: artifact.manifest.instructions ?? "",
            maxTokens: maxTokens,
            selectionMode: selection
        ))
    }
}

extension AgentResponse {
    init(smeltResponse response: SmeltTextGenerator.Response) {
        self.init(
            text: response.text,
            tokenIDs: response.tokenIDs,
            promptTokenCount: response.promptTokenCount,
            prefillTime: response.prefillTime,
            generateTime: response.generateTime
        )
    }
}
