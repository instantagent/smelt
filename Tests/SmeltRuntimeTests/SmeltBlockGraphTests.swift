import Foundation
import Testing
@testable import SmeltSchema

// SmeltBlockGraph — the declared block composition (B1 of
// docs/block-spec-plan.md): canonical graphs, structural validation, and
// explicit endpoint signatures.

@Suite struct SmeltBlockGraphTests {

    @Test func canonicalGraphsValidateAndExposeSignatures() throws {
        try SmeltBlockGraph.tokenFeedbackText.validate()
        try SmeltBlockGraph.qwen3TTS.validate()
        try SmeltBlockGraph.qwen3TTSCompiledTalker.validate()
        try SmeltBlockGraph.qwen3TTSCodecDecoder.validate()
        try SmeltLoopSchedule.qwen3TTSCodecDecoder.validate(against: .qwen3TTSCodecDecoder)
        #expect(SmeltBlockGraph.tokenFeedbackText.signature?.input == .text)
        #expect(SmeltBlockGraph.tokenFeedbackText.signature?.output == .text)
        #expect(SmeltBlockGraph.qwen3TTS.signature?.input == .text)
        #expect(SmeltBlockGraph.qwen3TTS.signature?.output == .audio)
        #expect(SmeltBlockGraph.qwen3TTSCompiledTalker.signature?.output == .audio)
        #expect(SmeltBlockGraph.qwen3TTSCodecDecoder.signature?.input == .codecFrames)
        #expect(SmeltBlockGraph.qwen3TTSCodecDecoder.signature?.output == .audio)
    }

    @Test func canonicalGraphImplAndDeliveryAreHonest() {
        // The graph is an honest map of what RUNS: `impl` (compiled vs native) + how a
        // compiled block is DELIVERED. Pins so a future change that makes the graph
        // dishonest (e.g. codec-decoder back to .native, or a sidecar mislabeled) fails here.
        func b(_ g: SmeltBlockGraph, _ name: String) -> SmeltBlockGraph.Block {
            g.blocks.first { $0.name == name }!
        }
        // LLM trunk: compiled, baked dispatch table in the MAIN package (not a sidecar subdir).
        #expect(b(.tokenFeedbackText, "trunk").impl == .compiled)
        #expect(b(.tokenFeedbackText, "trunk").compiledDelivery == .bakedInline)
        // TTS non-bf16 graph: the front-end + talker + MTP are HAND; codec-decoder is COMPILED
        // (runtime-emit) in EVERY build (its dtype-aware realizer covers u4/f16/f32 too).
        #expect(b(.qwen3TTS, "tts-frontend").impl == .native)
        #expect(b(.qwen3TTS, "tts-frontend").compiledDelivery == nil)
        #expect(b(.qwen3TTS, "talker").impl == .native)
        #expect(b(.qwen3TTS, "talker").compiledDelivery == nil)
        #expect(b(.qwen3TTS, "mtp-head").impl == .native)
        #expect(b(.qwen3TTS, "mtp-head").compiledDelivery == nil)
        #expect(b(.qwen3TTS, "codec-decoder").impl == .compiled)
        #expect(b(.qwen3TTS, "codec-decoder").compiledDelivery == .runtimeEmit)
        // TTS bf16 graph: front-end is a COMPILED runtime-emit record table (1a-ii); talker baked in
        // its own sidecar subdir; MTP native head wrapping a compiled internal transformer; codec
        // still runtime-emit.
        #expect(b(.qwen3TTSCompiledTalker, "tts-frontend").impl == .compiled)
        #expect(b(.qwen3TTSCompiledTalker, "tts-frontend").compiledDelivery == .runtimeEmit)
        #expect(b(.qwen3TTSCompiledTalker, "talker").impl == .compiled)
        #expect(b(.qwen3TTSCompiledTalker, "talker").compiledDelivery == .bakedSidecar)
        #expect(b(.qwen3TTSCompiledTalker, "mtp-head").impl == .native)
        #expect(b(.qwen3TTSCompiledTalker, "mtp-head").compiledDelivery == .internalSidecar)
        #expect(b(.qwen3TTSCompiledTalker, "codec-decoder").compiledDelivery == .runtimeEmit)
        // TTS u4 graph (Phase 3): trunks compiled (talker baked sidecar, MTP internal sidecar),
        // but the front-end stays NATIVE — text_embedding is u4, no compiled u4 gather yet. The
        // honest u4 graph, distinct from the bf16 graph's compiled-front-end claim.
        #expect(b(.qwen3TTSCompiledTrunkNativeFrontEnd, "tts-frontend").impl == .native)
        #expect(b(.qwen3TTSCompiledTrunkNativeFrontEnd, "tts-frontend").compiledDelivery == nil)
        #expect(b(.qwen3TTSCompiledTrunkNativeFrontEnd, "talker").impl == .compiled)
        #expect(b(.qwen3TTSCompiledTrunkNativeFrontEnd, "talker").compiledDelivery == .bakedSidecar)
        #expect(b(.qwen3TTSCompiledTrunkNativeFrontEnd, "mtp-head").impl == .native)
        #expect(b(.qwen3TTSCompiledTrunkNativeFrontEnd, "mtp-head").compiledDelivery == .internalSidecar)
        #expect(b(.qwen3TTSCompiledTrunkNativeFrontEnd, "codec-decoder").impl == .compiled)
        #expect(b(.qwen3TTSCodecDecoder, "codec-decoder").compiledDelivery == .runtimeEmit)
    }

    @Test func roundTripsThroughJSON() throws {
        for graph in [SmeltBlockGraph.tokenFeedbackText, .qwen3TTS, .qwen3TTSCompiledTalker, .qwen3TTSCodecDecoder] {
            let data = try JSONEncoder().encode(graph)
            let back = try JSONDecoder().decode(SmeltBlockGraph.self, from: data)
            #expect(back == graph)
        }
    }

    @Test func brokenWiringRejected() {
        // talker expects embeddings; tokenizer produces tokens.
        let graph = SmeltBlockGraph(blocks: [
            .init(name: "tok", role: .frontend, impl: .native,
                  inputs: [.text], output: .tokens),
            .init(name: "talker", role: .trunk, impl: .native,
                  inputs: [.embeddings], output: .logits),
            .init(name: "head", role: .head, impl: .native,
                  inputs: [.logits], output: .text),
        ])
        #expect(throws: SmeltBlockGraph.GraphError.self) { try graph.validate() }
    }

    @Test func internalEndpointsRejected() {
        let graph = SmeltBlockGraph(blocks: [
            .init(name: "trunk", role: .trunk, impl: .compiled,
                  inputs: [.tokens], output: .logits)
        ])
        #expect(throws: SmeltBlockGraph.GraphError.self) { try graph.validate() }
    }

    @Test func roleOrderAndTrunkRequired() {
        let headFirst = SmeltBlockGraph(blocks: [
            .init(name: "head", role: .head, impl: .native,
                  inputs: [.text], output: .tokens),
            .init(name: "trunk", role: .trunk, impl: .compiled,
                  inputs: [.tokens], output: .logits),
            .init(name: "out", role: .head, impl: .native,
                  inputs: [.logits], output: .text),
        ])
        #expect(throws: SmeltBlockGraph.GraphError.self) { try headFirst.validate() }

        let trunkless = SmeltBlockGraph(blocks: [
            .init(name: "tok", role: .frontend, impl: .native,
                  inputs: [.text], output: .tokens),
            .init(name: "detok", role: .head, impl: .native,
                  inputs: [.tokens], output: .text),
        ])
        #expect(throws: SmeltBlockGraph.GraphError.self) { try trunkless.validate() }
    }

    @Test func extraInputsMustBeProducedUpstream() {
        // A head consuming hidden that nothing upstream produced.
        let graph = SmeltBlockGraph(blocks: [
            .init(name: "tok", role: .frontend, impl: .native,
                  inputs: [.text], output: .tokens),
            .init(name: "trunk", role: .trunk, impl: .compiled,
                  inputs: [.tokens], output: .logits),
            .init(name: "head", role: .head, impl: .native,
                  inputs: [.logits, .hidden], output: .text),
        ])
        #expect(throws: SmeltBlockGraph.GraphError.self) { try graph.validate() }
    }

    @Test func feedbackEdgesAreDeclaredOnARBlocks() {
        // Self-feedback is a loop edge, not wiring: the talker re-consumes
        // codec embedding sums, and the LLM trunk re-consumes its sampled token.
        #expect(SmeltBlockGraph.tokenFeedbackText.blocks[1].feedback == .tokens)
        #expect(SmeltBlockGraph.qwen3TTS.blocks[1].feedback == .embeddings)
    }

    @Test func ttsGraphSeparatesCb0HeadFromMTP() {
        // The cb0 selection (GPU sampler: repetition penalty, suppress
        // ranges, min-new-token EOS) is its own head; MTP consumes the cb0
        // token AND the talker hidden.
        let names = SmeltBlockGraph.qwen3TTS.blocks.map(\.name)
        #expect(names == ["tts-frontend", "talker", "codec-head", "mtp-head", "codec-decoder"])
        let mtp = SmeltBlockGraph.qwen3TTS.blocks[3]
        #expect(mtp.inputs == [.tokens, .hidden])
        #expect(mtp.state == [.kvCache, .sampler])
    }

    @Test func firstBlockExtraInputsRejected() {
        // Nothing upstream can feed a first block's extra input.
        let graph = SmeltBlockGraph(blocks: [
            .init(name: "front", role: .frontend, impl: .native,
                  inputs: [.text, .hidden], output: .tokens),
            .init(name: "trunk", role: .trunk, impl: .compiled,
                  inputs: [.tokens], output: .logits),
            .init(name: "head", role: .head, impl: .native,
                  inputs: [.logits], output: .text),
        ])
        #expect(throws: SmeltBlockGraph.GraphError.self) { try graph.validate() }
    }

    @Test func frontendFirstHeadLastRequired() {
        // A bare trunk with external endpoints is not a canonical package.
        let trunkOnly = SmeltBlockGraph(blocks: [
            .init(name: "trunk", role: .trunk, impl: .compiled,
                  inputs: [.text], output: .text)
        ])
        #expect(throws: SmeltBlockGraph.GraphError.self) { try trunkOnly.validate() }
    }

    @Test func duplicateNamesRejected() {
        let graph = SmeltBlockGraph(blocks: [
            .init(name: "x", role: .frontend, impl: .native,
                  inputs: [.text], output: .tokens),
            .init(name: "x", role: .trunk, impl: .compiled,
                  inputs: [.tokens], output: .logits),
            .init(name: "head", role: .head, impl: .native,
                  inputs: [.logits], output: .text),
        ])
        #expect(throws: SmeltBlockGraph.GraphError.self) { try graph.validate() }
    }
}

@Suite struct SmeltBlockGraphDetectionTests {

    private func manifestJSON(label: String? = nil, graph: SmeltBlockGraph?) throws -> Data {
        var object: [String: Any] = ["modelName": "m"]
        if let label { object["kind"] = label }
        if let graph {
            let graphData = try JSONEncoder().encode(graph)
            object["blocks"] = try JSONSerialization.jsonObject(with: graphData)
        }
        return try JSONSerialization.data(withJSONObject: object)
    }

    @Test func graphEndpointsResolveRuntimePolicyWithoutPackageKind() throws {
        let ttsData = try manifestJSON(graph: .qwen3TTS)
        #expect(try SmeltRuntimeGraphPolicy.resolve(manifestData: ttsData) == .sidecarTextToCodecAudio)

        let codecData = try manifestJSON(graph: .qwen3TTSCodecDecoder)
        #expect(try SmeltRuntimeGraphPolicy.resolve(manifestData: codecData) == .codecAudio)
    }

    @Test func textGraphResolvesFromBlocks() throws {
        let data = try manifestJSON(label: "audio", graph: .tokenFeedbackText)
        #expect(try SmeltRuntimeGraphPolicy.resolve(manifestData: data) == .textGeneration)
    }

    @Test func invalidGraphFailsDetection() throws {
        let broken = SmeltBlockGraph(blocks: [
            .init(name: "tok", role: .frontend, impl: .native,
                  inputs: [.text], output: .tokens)
        ])
        let data = try manifestJSON(label: "text", graph: broken)
        #expect(throws: SmeltRuntimeGraphPolicy.ResolveError.self) {
            try SmeltRuntimeGraphPolicy.resolve(manifestData: data)
        }
    }

    @Test func newerGraphVersionIsNamedNotGeneric() throws {
        var object: [String: Any] = [:]
        let graphData = try JSONEncoder().encode(SmeltBlockGraph.tokenFeedbackText)
        var graphObject = try JSONSerialization.jsonObject(with: graphData) as! [String: Any]
        graphObject["version"] = 2
        object["blocks"] = graphObject
        let data = try JSONSerialization.data(withJSONObject: object)
        #expect(throws: SmeltRuntimeGraphPolicy.ResolveError.self) {
            try SmeltRuntimeGraphPolicy.resolve(manifestData: data)
        }
    }

    @Test func unknownSideOutputIsNamedNotIgnored() throws {
        // sideOutputs is a closed vocabulary: a future label fails the
        // typed decode and reports as newer-smelt, never silently drops.
        let data = Data(#"""
        {"blocks": {"version": 1, "blocks": [
          {"name": "f", "role": "frontend", "impl": "native",
           "inputs": ["text"], "output": "tokens"},
          {"name": "t", "role": "trunk", "impl": "compiled",
           "inputs": ["tokens"], "output": "logits",
           "sideOutputs": ["entropy-trace"]},
          {"name": "h", "role": "head", "impl": "native",
           "inputs": ["logits"], "output": "text"}]}}
        """#.utf8)
        #expect(throws: DecodingError.self) {
            try SmeltRuntimeGraphPolicy.resolve(manifestData: data)
        }
    }

    @Test func newerBlockVocabularyIsNamedNotJunk() throws {
        // A graph from a future smelt (unknown port type) must be reported
        // as such, not as corrupt JSON.
        let data = Data(#"""
        {"blocks": {"version": 1, "blocks": [
          {"name": "x", "role": "trunk", "impl": "compiled",
           "inputs": ["image"], "output": "image"}]}}
        """#.utf8)
        #expect(throws: DecodingError.self) {
            try SmeltRuntimeGraphPolicy.resolve(manifestData: data)
        }
    }
}
