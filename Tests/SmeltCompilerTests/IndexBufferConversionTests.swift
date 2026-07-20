import Foundation
import Testing
@testable import SmeltCompiler

private func loadAssistantIR() throws -> SmeltModelIR {
    SmeltModelIR(
        modelName: "test-assistant",
        config: SmeltConfig(
            hiddenSize: 256, numLayers: 4, vocabSize: 262_144,
            ropeDim: 256, rmsEps: 1e-6,
            attentionConfigs: [.sliding: SmeltAttentionConfig(
                qHeads: 4, kvHeads: 1, headDim: 256, gatedQ: false, externalKV: true)],
            ffn: SmeltFFNConfig(dim: 2048, activation: .swiglu),
            tiedLMHead: true,
            backboneHiddenSize: 1536,
            clusterEmbedder: SmeltClusterEmbedderConfig(numCentroids: 2048, topK: 32)),
        layerPattern: SmeltLayerPattern(unit: [.sliding], repeats: 4),
        quantization: SmeltQuantizationConfig(strategy: .lutU4, groupSize: 16, excludePatterns: []),
        loading: SmeltLoadingConfig(strategy: .mmapPrefault, packing: .monolithic))
}

@Test func indexBufferTransformPassesThroughInt32() throws {
    let ir = try loadAssistantIR()
    let vocab = ir.config.vocabSize
    let source = (0 ..< vocab).map { Int32($0 % vocab) }

    let result = try source.withUnsafeBytes { rawBytes -> [String] in
        let tensors: [(runtimeName: String, data: UnsafeRawPointer,
                       byteCount: Int, shape: [Int], dtype: String)] = [
            (
                runtimeName: "masked_embedding_token_ordering",
                data: rawBytes.baseAddress!,
                byteCount: vocab * 4,
                shape: [vocab],
                dtype: "I32"
            )
        ]
        let (out, owned) =
            try SmeltCompiler.adjustedCheckpointTensorsForIndexBuffers(
                tensors, config: ir.config
            )
        #expect(owned.isEmpty)  // pass-through allocates nothing
        #expect(out.count == 1)
        #expect(out[0].dtype == "I32")
        #expect(out[0].byteCount == vocab * 4)
        #expect(out[0].data == rawBytes.baseAddress!)
        return [out[0].runtimeName]
    }
    #expect(result == ["masked_embedding_token_ordering"])
}

@Test func indexBufferTransformDowncastsInt64ToInt32() throws {
    let ir = try loadAssistantIR()
    let vocab = ir.config.vocabSize
    // Use a permutation that includes the vocab boundary - 1 to exercise
    // the upper edge of the range.
    let source: [Int64] = (0 ..< vocab).map { Int64(($0 + 7) % vocab) }

    try source.withUnsafeBytes { rawBytes in
        let tensors: [(runtimeName: String, data: UnsafeRawPointer,
                       byteCount: Int, shape: [Int], dtype: String)] = [
            (
                runtimeName: "masked_embedding_token_ordering",
                data: rawBytes.baseAddress!,
                byteCount: vocab * 8,
                shape: [vocab],
                dtype: "I64"
            )
        ]
        let (out, owned) =
            try SmeltCompiler.adjustedCheckpointTensorsForIndexBuffers(
                tensors, config: ir.config
            )
        #expect(owned.count == 1)
        #expect(out.count == 1)
        #expect(out[0].dtype == "I32")
        #expect(out[0].byteCount == vocab * 4)

        let dst = out[0].data.bindMemory(to: Int32.self, capacity: vocab)
        for idx in stride(from: 0, to: vocab, by: 1024) {
            #expect(dst[idx] == Int32(source[idx]))
        }
        // Withdraw the lifetime hold so `owned` survives until we leave.
        withExtendedLifetime(owned) {}
    }
}

@Test func indexBufferTransformRejectsOutOfRangeInt64() throws {
    let ir = try loadAssistantIR()
    let vocab = ir.config.vocabSize
    var source: [Int64] = (0 ..< vocab).map { Int64($0) }
    source[42] = Int64(vocab)  // out of range — must equal vocab or higher

    source.withUnsafeBytes { rawBytes in
        let tensors: [(runtimeName: String, data: UnsafeRawPointer,
                       byteCount: Int, shape: [Int], dtype: String)] = [
            (
                runtimeName: "masked_embedding_token_ordering",
                data: rawBytes.baseAddress!,
                byteCount: vocab * 8,
                shape: [vocab],
                dtype: "I64"
            )
        ]
        do {
            _ = try SmeltCompiler.adjustedCheckpointTensorsForIndexBuffers(
                tensors, config: ir.config
            )
            Issue.record("expected throw for out-of-range index")
        } catch SmeltCompilerError.unsupportedConfiguration(let message) {
            #expect(message.contains("masked_embedding_token_ordering"))
            #expect(message.contains("entry 42"))
            #expect(message.contains("\(vocab)"))
        } catch {
            Issue.record("expected unsupportedConfiguration, got \(error)")
        }
    }
}

@Test func indexBufferTransformRejectsNegativeInt64() throws {
    let ir = try loadAssistantIR()
    let vocab = ir.config.vocabSize
    var source: [Int64] = (0 ..< vocab).map { Int64($0) }
    source[1] = -1  // negative — common int64 sign-extension footgun

    source.withUnsafeBytes { rawBytes in
        let tensors: [(runtimeName: String, data: UnsafeRawPointer,
                       byteCount: Int, shape: [Int], dtype: String)] = [
            (
                runtimeName: "masked_embedding_token_ordering",
                data: rawBytes.baseAddress!,
                byteCount: vocab * 8,
                shape: [vocab],
                dtype: "I64"
            )
        ]
        do {
            _ = try SmeltCompiler.adjustedCheckpointTensorsForIndexBuffers(
                tensors, config: ir.config
            )
            Issue.record("expected throw for negative index")
        } catch SmeltCompilerError.unsupportedConfiguration {
            // Expected.
        } catch {
            Issue.record("expected unsupportedConfiguration, got \(error)")
        }
    }
}

@Test func indexBufferTransformRejectsUnsupportedDtype() throws {
    let ir = try loadAssistantIR()
    let bytes = [UInt8](repeating: 0, count: 16)
    bytes.withUnsafeBytes { rawBytes in
        let tensors: [(runtimeName: String, data: UnsafeRawPointer,
                       byteCount: Int, shape: [Int], dtype: String)] = [
            (
                runtimeName: "masked_embedding_token_ordering",
                data: rawBytes.baseAddress!,
                byteCount: 16,
                shape: [4],
                dtype: "F16"
            )
        ]
        do {
            _ = try SmeltCompiler.adjustedCheckpointTensorsForIndexBuffers(
                tensors, config: ir.config
            )
            Issue.record("expected throw for unsupported dtype")
        } catch SmeltCompilerError.unsupportedConfiguration(let message) {
            #expect(message.contains("F16"))
        } catch {
            Issue.record("expected unsupportedConfiguration, got \(error)")
        }
    }
}

@Test func indexBufferTransformIgnoresUnrelatedTensors() throws {
    let ir = try loadAssistantIR()
    let bytes = [UInt8](repeating: 0, count: 16)
    try bytes.withUnsafeBytes { rawBytes in
        let tensors: [(runtimeName: String, data: UnsafeRawPointer,
                       byteCount: Int, shape: [Int], dtype: String)] = [
            (
                runtimeName: "embed_tokens",
                data: rawBytes.baseAddress!,
                byteCount: 16,
                shape: [4, 2],
                dtype: "F16"
            )
        ]
        let (out, owned) =
            try SmeltCompiler.adjustedCheckpointTensorsForIndexBuffers(
                tensors, config: ir.config
            )
        #expect(owned.isEmpty)
        #expect(out.count == 1)
        #expect(out[0].runtimeName == "embed_tokens")
        #expect(out[0].dtype == "F16")
    }
}
