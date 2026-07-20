import Foundation
import Testing
@testable import SmeltServe

@Suite("OpenAI request extensions")
struct OpenAIRequestExtensionTests {
    @Test("chat requests retain prepared prompt and filtered sampling policy")
    func decodesPreparedPromptContract() throws {
        let data = Data(#"""
        {
            "model":"current",
            "messages":[{"role":"user","content":"hello"}],
            "temperature":0.7,
            "top_k":20,
            "top_p":0.95,
            "prompt_contract":"interactive/pi-v1"
        }
        """#.utf8)

        let request = try JSONDecoder().decode(
            OpenAIChatCompletionsRequest.self,
            from: data
        )
        #expect(request.temperature == 0.7)
        #expect(request.topK == 20)
        #expect(request.topP == 0.95)
        #expect(request.promptContract == "interactive/pi-v1")
    }

    @Test("legacy completions retain filtered sampling policy")
    func decodesCompletionsFilters() throws {
        let data = Data(#"""
        {
            "model":"current",
            "prompt":"hello",
            "temperature":0.8,
            "top_k":12,
            "top_p":0.9
        }
        """#.utf8)

        let request = try JSONDecoder().decode(
            OpenAICompletionsRequest.self,
            from: data
        )
        #expect(request.temperature == 0.8)
        #expect(request.topK == 12)
        #expect(request.topP == 0.9)
    }
}
