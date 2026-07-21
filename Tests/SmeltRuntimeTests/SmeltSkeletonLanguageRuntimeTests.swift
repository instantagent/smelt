import CryptoKit
import Foundation
import Testing

@testable import SmeltRuntime

@Suite("Smelt skeleton language runtime")
struct SmeltSkeletonLanguageRuntimeTests {
    private struct ReferenceManifest: Decodable {
        struct Tensor: Decodable {
            let file: String
            let shape: [Int]
            let dtype: String
            let sha256: String
        }

        let schema: String
        let sourceCommit: String
        let checkpointSHA256: String
        let attentionImplementation: String
        let modelDtype: String?
        let sequenceLength: Int
        let incrementalPrefixLength: Int?
        let incrementalCheckpoints: [Int]?
        let incrementalJointCount: Int?
        let tensors: [String: Tensor]

        enum CodingKeys: String, CodingKey {
            case schema
            case sourceCommit = "source_commit"
            case checkpointSHA256 = "checkpoint_sha256"
            case attentionImplementation = "attention_implementation"
            case modelDtype = "model_dtype"
            case sequenceLength = "sequence_length"
            case incrementalPrefixLength = "incremental_prefix_length"
            case incrementalCheckpoints = "incremental_checkpoints"
            case incrementalJointCount = "incremental_joint_count"
            case tensors
        }
    }

    private struct Metrics {
        let cosine: Double
        let relativeL2: Double
        let maximumAbsoluteDifference: Float
        let bitExactCount: Int
    }

    private struct LongContextReceipt: Encodable {
        let updates: Int
        let decisions: Int
        let prefixLength: Int
        let lastDecodedPosition: Int
        let finalContextLength: Int
        let prefillMilliseconds: Double
        let prefillTokensPerSecond: Double
        let decodeMedianMilliseconds: Double
        let decodeP95Milliseconds: Double
        let decodeTokensPerSecond: Double
        let worstHiddenCosine: Double
        let worstHiddenRelativeL2: Double
        let worstHiddenMaximumAbsoluteDifference: Float
        let worstHiddenMaximumAbsoluteDifferenceUpdate: Int
        let worstLogitCosine: Double
        let worstLogitRelativeL2: Double
        let worstLogitMaximumAbsoluteDifference: Float
        let worstLogitMaximumAbsoluteDifferenceUpdate: Int
        let firstDivergentLayerUpdate: Int?
        let firstDivergentLayer: Int?
        let firstDecisionMismatchUpdate: Int?
    }

    private func milliseconds(_ duration: Duration) -> Double {
        let components = duration.components
        return Double(components.seconds) * 1_000
            + Double(components.attoseconds) / 1_000_000_000_000_000
    }

    private func percentile(_ sortedValues: [Double], numerator: Int) -> Double {
        precondition(!sortedValues.isEmpty)
        precondition((0...100).contains(numerator))
        let index = (sortedValues.count - 1) * numerator / 100
        return sortedValues[index]
    }

    private func metrics(actual: [Float], expected: [Float]) -> Metrics {
        precondition(actual.count == expected.count)
        var dot: Double = 0
        var actualSquared: Double = 0
        var expectedSquared: Double = 0
        var differenceSquared: Double = 0
        var maximum: Float = 0
        var bitExactCount = 0
        for index in actual.indices {
            let a = Double(actual[index])
            let e = Double(expected[index])
            let difference = a - e
            dot += a * e
            actualSquared += a * a
            expectedSquared += e * e
            differenceSquared += difference * difference
            maximum = max(maximum, abs(actual[index] - expected[index]))
            if actual[index].bitPattern == expected[index].bitPattern {
                bitExactCount += 1
            }
        }
        return Metrics(
            cosine: dot / sqrt(actualSquared * expectedSquared),
            relativeL2: sqrt(differenceSquared / expectedSquared),
            maximumAbsoluteDifference: maximum,
            bitExactCount: bitExactCount
        )
    }

    private func argmax(_ values: [Float]) -> Int? {
        values.indices.max { values[$0] < values[$1] }
    }

    private func sourceCompatibleGreedyToken(
        logits: [Float],
        sequence: [Int],
        generatedHistory: [Int]
    ) throws -> Int {
    let allowed = try SmeltSkeletonGenerationPolicy.allowedNextTokens(
            sequence: sequence,
            mode: .sourceCompatible
        )
    return try SmeltSkeletonGenerationPolicy.sample(
            logits: logits,
            history: generatedHistory,
            allowedTokens: allowed,
            repetitionPenalty: 2,
            temperature: 1,
            topK: 1,
            topP: 1,
            uniform: 0
        ).token
    }

    private func floats(_ path: String) throws -> [Float] {
        let tensor = try NpyLoader.load(path: path)
        #expect(tensor.dtype == "f4")
        return Array(
            UnsafeBufferPointer(
                start: tensor.fp32Pointer,
                count: tensor.elementCount
            )
        )
    }

    private func loadReference(_ path: String) throws -> ReferenceManifest {
        let data = try Data(contentsOf: URL(fileURLWithPath: "\(path)/manifest.json"))
        let manifest = try JSONDecoder().decode(ReferenceManifest.self, from: data)
    #expect(manifest.schema == "smelt.skintokens.language-reference.v1")
        #expect(manifest.sourceCommit == "273b691d35989d71cd17ff2895fdc735097b92d1")
        #expect(
            manifest.checkpointSHA256
                == "f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692"
        )
        #expect(["eager", "flash_attention_2"].contains(manifest.attentionImplementation))
        #expect(manifest.modelDtype == "torch.bfloat16")
        return manifest
    }

    private func verifiedFloats(
        referencePath: String,
        manifest: ReferenceManifest,
        tensorName: String
    ) throws -> [Float] {
        let tensor = try #require(manifest.tensors[tensorName])
        #expect(tensor.dtype == "float32")
        let path = "\(referencePath)/\(tensor.file)"
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        #expect(digest == tensor.sha256)
        let values = try floats(path)
        #expect(values.count == tensor.shape.reduce(1, *))
        return values
    }

    private func verifiedInt32(
        referencePath: String,
        manifest: ReferenceManifest,
        tensorName: String
    ) throws -> [Int32] {
        let tensor = try #require(manifest.tensors[tensorName])
        #expect(tensor.dtype == "int32")
        let path = "\(referencePath)/\(tensor.file)"
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        #expect(digest == tensor.sha256)
        let values = try NpyLoader.load(path: path)
        #expect(values.dtype == "i4")
        #expect(values.elementCount == tensor.shape.reduce(1, *))
        return Array(
            UnsafeBufferPointer(
                start: values.data.bindMemory(
                    to: Int32.self,
                    capacity: values.elementCount
                ),
                count: values.elementCount
            )
        )
    }

    @Test("Canonical sidecar loads and runs checkpoint-backed prefill")
    func canonicalPrefillSmoke() throws {
    guard let package = ProcessInfo.processInfo.environment["SMELT_SKINNING_PACKAGE"] else {
            return
        }
        let runtime = try SmeltSkeletonLanguageRuntime(packagePath: package)
        let embeddingIDs = [257, 263, 33_035]
        let gatheredEmbeddings = try runtime.embeddings(tokenIDs: embeddingIDs)
        #expect(gatheredEmbeddings.count == embeddingIDs.count * 896)
        let embeddingDescriptor = try #require(
            runtime.artifact.checkpointTensors.first {
                $0.name == "transformer.model.embed_tokens.weight"
            }
        )
        let embeddingSource = runtime.artifact.checkpointTensorData(
            embeddingDescriptor
        ).bindMemory(to: UInt16.self, capacity: 33_036 * 896)
        for (row, tokenID) in embeddingIDs.enumerated() {
            for column in 0..<896 {
                #expect(
                    gatheredEmbeddings[row * 896 + column].bitPattern
                        == UInt32(embeddingSource[tokenID * 896 + column]) << 16
                )
            }
        }
        let hiddenSize = 896
        let referencePath = ProcessInfo.processInfo.environment[
      "SMELT_SKINNING_LANGUAGE_REFERENCE"
        ]
        let referenceManifest = try referencePath.map(loadReference)
        let sequenceLength = referenceManifest?.sequenceLength ?? 2
        let embeddings: [Float]
        if let referencePath, let referenceManifest {
            embeddings = try verifiedFloats(
                referencePath: referencePath,
                manifest: referenceManifest,
                tensorName: "input_embeddings"
            )
        } else {
            embeddings = (0..<(sequenceLength * hiddenSize)).map { index in
                let value = sin(Float(index * 37 + 11) * 0.0031) * 0.31
                let bits = UInt16(truncatingIfNeeded: value.bitPattern >> 16)
                return Float(bitPattern: UInt32(bits) << 16)
            }
        }
        let output = try runtime.prefill(
            embeddings: embeddings,
            sequenceLength: sequenceLength
        )
        #expect(output.hiddenStates.count == sequenceLength * hiddenSize)
        #expect(output.hiddenStates.allSatisfy { $0.isFinite })
        #expect((output.hiddenStates.map(abs).max() ?? 0) > 1e-3)
        #expect(output.finalLogits.count == 33_036)
        #expect(output.finalLogits.allSatisfy { $0.isFinite })
        #expect((output.finalLogits.map(abs).max() ?? 0) > 1e-3)

        guard let referencePath, let referenceManifest else {
            return
        }
        let allReferenceHidden = try verifiedFloats(
            referencePath: referencePath,
            manifest: referenceManifest,
            tensorName: "prefill_hidden_states"
        )
        let expectedHiddenCount = sequenceLength * hiddenSize
        let expectedHidden = Array(allReferenceHidden.suffix(expectedHiddenCount))
        let allReferenceLogits = try verifiedFloats(
            referencePath: referencePath,
            manifest: referenceManifest,
            tensorName: "prefill_logits"
        )
        let expectedLogits = Array(allReferenceLogits.suffix(33_036))
        let hiddenMetrics = metrics(
            actual: output.hiddenStates,
            expected: expectedHidden
        )
        let logitMetrics = metrics(
            actual: output.finalLogits,
            expected: expectedLogits
        )
        print(
      "SkinTokens language source parity: hidden cosine=\(hiddenMetrics.cosine) "
                + "relL2=\(hiddenMetrics.relativeL2) "
                + "maxAbs=\(hiddenMetrics.maximumAbsoluteDifference); "
                + "logits cosine=\(logitMetrics.cosine) "
                + "relL2=\(logitMetrics.relativeL2) "
                + "maxAbs=\(logitMetrics.maximumAbsoluteDifference)"
        )
        if sequenceLength <= 2 {
            #expect(hiddenMetrics.cosine > 0.9998)
            #expect(hiddenMetrics.relativeL2 < 0.02)
            #expect(hiddenMetrics.maximumAbsoluteDifference < 0.1)
            #expect(logitMetrics.cosine > 0.9996)
            #expect(logitMetrics.relativeL2 < 0.03)
            #expect(logitMetrics.maximumAbsoluteDifference < 0.15)
        } else {
            #expect(hiddenMetrics.cosine > 0.9993)
            #expect(hiddenMetrics.relativeL2 < 0.04)
            #expect(hiddenMetrics.maximumAbsoluteDifference < 0.5)
            #expect(logitMetrics.cosine > 0.9992)
            #expect(logitMetrics.relativeL2 < 0.045)
            #expect(logitMetrics.maximumAbsoluteDifference < 0.2)
        }
        let actualToken = output.finalLogits.indices.max {
            output.finalLogits[$0] < output.finalLogits[$1]
        }
        let expectedToken = expectedLogits.indices.max {
            expectedLogits[$0] < expectedLogits[$1]
        }
        #expect(actualToken == expectedToken)

        let teacherTokens = try verifiedInt32(
            referencePath: referencePath,
            manifest: referenceManifest,
            tensorName: "teacher_tokens"
        )
        #expect(teacherTokens.count == 2)
        for (step, teacherToken) in teacherTokens.enumerated() {
            let decoded = try runtime.decodeTeacherForced(
                tokenID: Int(teacherToken),
                position: sequenceLength + step
            )
            let allStepHidden = try verifiedFloats(
                referencePath: referencePath,
                manifest: referenceManifest,
                tensorName: "step_\(step)_hidden_states"
            )
            let expectedStepHidden = Array(allStepHidden.suffix(hiddenSize))
            let allStepLogits = try verifiedFloats(
                referencePath: referencePath,
                manifest: referenceManifest,
                tensorName: "step_\(step)_logits"
            )
            let expectedStepLogits = Array(allStepLogits.suffix(33_036))
            let stepHiddenMetrics = metrics(
                actual: decoded.hiddenState,
                expected: expectedStepHidden
            )
            let stepLogitMetrics = metrics(
                actual: decoded.logits,
                expected: expectedStepLogits
            )
            print(
        "SkinTokens language teacher step \(step): hidden cosine="
                    + "\(stepHiddenMetrics.cosine) relL2="
                    + "\(stepHiddenMetrics.relativeL2) maxAbs="
                    + "\(stepHiddenMetrics.maximumAbsoluteDifference); logits cosine="
                    + "\(stepLogitMetrics.cosine) relL2="
                    + "\(stepLogitMetrics.relativeL2) maxAbs="
                    + "\(stepLogitMetrics.maximumAbsoluteDifference)"
            )
            if sequenceLength <= 2 {
                #expect(stepHiddenMetrics.cosine > 0.9998)
                #expect(stepHiddenMetrics.relativeL2 < 0.02)
                #expect(stepHiddenMetrics.maximumAbsoluteDifference < 0.1)
                #expect(stepLogitMetrics.cosine > 0.9996)
                #expect(stepLogitMetrics.relativeL2 < 0.03)
                #expect(stepLogitMetrics.maximumAbsoluteDifference < 0.15)
            } else {
                #expect(stepHiddenMetrics.cosine > 0.9995)
                #expect(stepHiddenMetrics.relativeL2 < 0.035)
                #expect(stepHiddenMetrics.maximumAbsoluteDifference < 0.12)
                #expect(stepLogitMetrics.cosine > 0.9995)
                #expect(stepLogitMetrics.relativeL2 < 0.035)
                #expect(stepLogitMetrics.maximumAbsoluteDifference < 0.15)
            }
            let actualStepToken = decoded.logits.indices.max {
                decoded.logits[$0] < decoded.logits[$1]
            }
            let expectedStepToken = expectedStepLogits.indices.max {
                expectedStepLogits[$0] < expectedStepLogits[$1]
            }
            #expect(actualStepToken == expectedStepToken)
        }

        let capture = try runtime.trunk.captureTrunkPrefillLayerOutputs(
            embeddings: embeddings,
            seqLen: sequenceLength
        )
        #expect(capture.layerOutputs.count == 28)
        #expect(capture.finalHiddenStates.count == output.hiddenStates.count)
        for index in output.hiddenStates.indices {
            #expect(
                capture.finalHiddenStates[index].bitPattern
                    == output.hiddenStates[index].bitPattern
            )
        }
        let referenceLayers = try verifiedFloats(
            referencePath: referencePath,
            manifest: referenceManifest,
            tensorName: "prefill_layer_outputs"
        )
        var worstCosine = 1.0
        var worstRelativeL2 = 0.0
        var worstMaximumAbsoluteDifference: Float = 0
        var worstLayer = -1
        for layer in 0..<28 {
            let start = layer * expectedHiddenCount
            let expected = Array(referenceLayers[start..<(start + expectedHiddenCount)])
            let layerMetrics = metrics(
                actual: capture.layerOutputs[layer],
                expected: expected
            )
      if ProcessInfo.processInfo.environment["SMELT_SKINNING_DIAGNOSTICS"] == "1" {
                print(
          "SkinTokens language prefill layer \(layer): exact="
                        + "\(layerMetrics.bitExactCount)/\(expected.count) relL2="
                        + "\(layerMetrics.relativeL2) maxAbs="
                        + "\(layerMetrics.maximumAbsoluteDifference)"
                )
            }
            if layerMetrics.relativeL2 > worstRelativeL2 {
                worstLayer = layer
                worstRelativeL2 = layerMetrics.relativeL2
            }
            worstCosine = min(worstCosine, layerMetrics.cosine)
            worstMaximumAbsoluteDifference = max(
                worstMaximumAbsoluteDifference,
                layerMetrics.maximumAbsoluteDifference
            )
            if sequenceLength <= 2 {
                #expect(layerMetrics.cosine > 0.9998)
                #expect(layerMetrics.relativeL2 < 0.02)
                #expect(layerMetrics.maximumAbsoluteDifference < 1.5)
            } else {
                #expect(layerMetrics.cosine > 0.9997)
                #expect(layerMetrics.relativeL2 < 0.025)
                #expect(layerMetrics.maximumAbsoluteDifference < 3.2)
            }
        }
        print(
      "SkinTokens language per-layer source parity: worstLayer=\(worstLayer) "
                + "minimumCosine=\(worstCosine) "
                + "maximumRelL2=\(worstRelativeL2) "
                + "maximumAbs=\(worstMaximumAbsoluteDifference)"
        )

        if referenceManifest.tensors["greedy_mesh_embeddings"] != nil,
           referenceManifest.tensors["greedy_generated_tokens"] != nil
        {
            let meshEmbeddings = try verifiedFloats(
                referencePath: referencePath,
                manifest: referenceManifest,
                tensorName: "greedy_mesh_embeddings"
            )
            let expectedGeneratedTokens = try verifiedInt32(
                referencePath: referencePath,
                manifest: referenceManifest,
                tensorName: "greedy_generated_tokens"
            ).map(Int.init)
            let generator = SmeltSkeletonGenerator(languageModel: runtime)
            let generated = try generator.generateTokenSequence(
                meshEmbeddings: meshEmbeddings,
                startTokenIDs: [257, 263, 128, 128, 128, 258],
                configuration: .init(
                    policyMode: .sourceCompatible,
                    decodeMode: .greedy,
                    maximumGeneratedTokens: expectedGeneratedTokens.count
                )
            )
            #expect(generated.generatedTokenIDs == expectedGeneratedTokens)
            #expect(
                runtime.trunk.memoryStats().currentContextCapacity
                    == 512 + 6 + expectedGeneratedTokens.count
            )
        }
    }

    @Test("Long incremental cache preserves the source greedy trajectory")
    func longIncrementalCacheParity() throws {
        let environment = ProcessInfo.processInfo.environment
    guard let package = environment["SMELT_SKINNING_PACKAGE"],
      let referencePath = environment["SMELT_SKINNING_LANGUAGE_LONG_REFERENCE"]
        else {
      #expect(environment["SMELT_SKINNING_REQUIRE_FIXTURES"] != "1")
            return
        }
        let manifest = try loadReference(referencePath)
        let checkpoints = try #require(manifest.incrementalCheckpoints)
        let prefixLength = try #require(manifest.incrementalPrefixLength)
        let jointCount = try #require(manifest.incrementalJointCount)
        #expect(checkpoints == checkpoints.sorted())
        #expect((checkpoints.first ?? 0) >= 1)
        let fixtureMaximumUpdate = try #require(checkpoints.last)
        let requestedMaximumUpdate = environment["SMELT_DENSE_TRUNK_LONG_MAX_UPDATE"]
            .flatMap(Int.init)
        let maximumUpdate = min(requestedMaximumUpdate ?? fixtureMaximumUpdate, fixtureMaximumUpdate)
        #expect(jointCount == 382)
        #expect(prefixLength == 512 + 3 * jointCount + 3)
        #expect(fixtureMaximumUpdate <= 4 * jointCount - 1)
        #expect(prefixLength + (4 * jointCount - 1) <= 3_192)
        #expect(maximumUpdate > 0)

        let meshEmbeddings = try verifiedFloats(
            referencePath: referencePath,
            manifest: manifest,
            tensorName: "incremental_mesh_embeddings"
        )
        let startTokens = try verifiedInt32(
            referencePath: referencePath,
            manifest: manifest,
            tensorName: "incremental_start_tokens"
        ).map(Int.init)
        let inputTokens = try verifiedInt32(
            referencePath: referencePath,
            manifest: manifest,
            tensorName: "incremental_input_tokens"
        ).map(Int.init)
        let nextTokens = try verifiedInt32(
            referencePath: referencePath,
            manifest: manifest,
            tensorName: "incremental_next_tokens"
        ).map(Int.init)
        let nextMargins = try verifiedFloats(
            referencePath: referencePath,
            manifest: manifest,
            tensorName: "incremental_next_margins"
        )
        #expect(meshEmbeddings.count == 512 * 896)
        #expect(startTokens.count == 3 * jointCount + 3)
        #expect(startTokens.prefix(2) == [257, 263])
        #expect(startTokens.last == 258)
        #expect(inputTokens.count == fixtureMaximumUpdate)
        #expect(nextTokens.count == fixtureMaximumUpdate)
        #expect(nextMargins.count == fixtureMaximumUpdate)
        #expect(Array(inputTokens.dropFirst()) == Array(nextTokens.dropLast()))

        let runtime = try SmeltSkeletonLanguageRuntime(packagePath: package)
        try runtime.trunk.ensureContextCapacity(3_192)
        let startEmbeddings = try runtime.embeddings(tokenIDs: startTokens)
        let prefix = meshEmbeddings + startEmbeddings
        let clock = ContinuousClock()
        let prefillStart = clock.now
        let prefill = try runtime.prefill(
            embeddings: prefix,
            sequenceLength: prefixLength
        )
        let prefillMilliseconds = milliseconds(prefillStart.duration(to: clock.now))
        var sequence = startTokens
        var generatedHistory: [Int] = []
        let initialToken = try sourceCompatibleGreedyToken(
            logits: prefill.finalLogits,
            sequence: sequence,
            generatedHistory: generatedHistory
        )
        #expect(initialToken == inputTokens[0])

        let checkpointSet = Set(checkpoints.filter { $0 <= maximumUpdate })
    var firstDecisionMismatch:
      (
            update: Int,
            expected: Int,
            actual: Int,
            sourceMargin: Float,
            expectedLogit: Float,
            actualLogit: Float
        )?
        var worstHiddenCosine = 1.0
        var worstHiddenRelativeL2 = 0.0
        var worstHiddenMaximumAbsoluteDifference: Float = 0
        var worstHiddenMaximumAbsoluteDifferenceUpdate = 0
        var worstLogitCosine = 1.0
        var worstLogitRelativeL2 = 0.0
        var worstLogitMaximumAbsoluteDifference: Float = 0
        var worstLogitMaximumAbsoluteDifferenceUpdate = 0
        var firstDivergentLayerUpdate: Int?
        var firstDivergentLayerIndex: Int?
        var decodeMilliseconds: [Double] = []
        decodeMilliseconds.reserveCapacity(maximumUpdate)

        for (index, inputToken) in inputTokens.prefix(maximumUpdate).enumerated() {
            let update = index + 1
            sequence.append(inputToken)
            generatedHistory.append(inputToken)
            let layerTensorName = "incremental_\(update)_layer_outputs"
      let captureLayers =
        checkpointSet.contains(update)
                && manifest.tensors[layerTensorName] != nil
            let decoded: SmeltSkeletonLanguageRuntime.DecodeOutput
            let layerCapture: SmeltTrunkLayerCapture?
            let decodeStart = clock.now
            if captureLayers {
                let result = try runtime.decodeTeacherForcedCapturingLayers(
                    tokenID: inputToken,
                    position: prefixLength + index
                )
                decoded = result.output
                layerCapture = result.capture
            } else {
                decoded = try runtime.decodeTeacherForced(
                    tokenID: inputToken,
                    position: prefixLength + index
                )
                layerCapture = nil
            }
            decodeMilliseconds.append(
                milliseconds(decodeStart.duration(to: clock.now))
            )
            let actualNext = try sourceCompatibleGreedyToken(
                logits: decoded.logits,
                sequence: sequence,
                generatedHistory: generatedHistory
            )
            if firstDecisionMismatch == nil, actualNext != nextTokens[index] {
                firstDecisionMismatch = (
                    update,
                    nextTokens[index],
                    actualNext,
                    nextMargins[index],
                    decoded.logits[nextTokens[index]],
                    decoded.logits[actualNext]
                )
            }
            guard checkpointSet.contains(update) else {
                continue
            }

            let expectedHidden = try verifiedFloats(
                referencePath: referencePath,
                manifest: manifest,
                tensorName: "incremental_\(update)_hidden_state"
            )
            let expectedLogits = try verifiedFloats(
                referencePath: referencePath,
                manifest: manifest,
                tensorName: "incremental_\(update)_logits"
            )
            if let layerCapture {
                let expectedLayers = try verifiedFloats(
                    referencePath: referencePath,
                    manifest: manifest,
                    tensorName: layerTensorName
                )
                #expect(layerCapture.layerOutputs.count == 28)
                #expect(expectedLayers.count == 28 * 896)
                var firstDivergentLayer: Int?
                for layer in 0..<28 {
                    let start = layer * 896
                    let expectedLayer = Array(expectedLayers[start..<(start + 896)])
                    let layerMetrics = metrics(
                        actual: layerCapture.layerOutputs[layer],
                        expected: expectedLayer
                    )
                    if firstDivergentLayer == nil,
                       layerMetrics.bitExactCount != expectedLayer.count
                    {
                        firstDivergentLayer = layer
                        if firstDivergentLayerUpdate == nil {
                            firstDivergentLayerUpdate = update
                            firstDivergentLayerIndex = layer
                        }
                        print(
                            "Dense trunk long update \(update) first divergent layer "
                                + "\(layer): exact=\(layerMetrics.bitExactCount)/896 "
                                + "relL2=\(layerMetrics.relativeL2) maxAbs="
                                + "\(layerMetrics.maximumAbsoluteDifference)"
                        )
                    }
                }
            }
            let hiddenMetrics = metrics(
                actual: decoded.hiddenState,
                expected: expectedHidden
            )
            let logitMetrics = metrics(
                actual: decoded.logits,
                expected: expectedLogits
            )
            worstHiddenCosine = min(worstHiddenCosine, hiddenMetrics.cosine)
            worstHiddenRelativeL2 = max(
                worstHiddenRelativeL2,
                hiddenMetrics.relativeL2
            )
            if hiddenMetrics.maximumAbsoluteDifference
                > worstHiddenMaximumAbsoluteDifference
            {
                worstHiddenMaximumAbsoluteDifference =
                    hiddenMetrics.maximumAbsoluteDifference
                worstHiddenMaximumAbsoluteDifferenceUpdate = update
            }
            worstLogitCosine = min(worstLogitCosine, logitMetrics.cosine)
            worstLogitRelativeL2 = max(
                worstLogitRelativeL2,
                logitMetrics.relativeL2
            )
            if logitMetrics.maximumAbsoluteDifference
                > worstLogitMaximumAbsoluteDifference
            {
                worstLogitMaximumAbsoluteDifference =
                    logitMetrics.maximumAbsoluteDifference
                worstLogitMaximumAbsoluteDifferenceUpdate = update
            }
            print(
        "SkinTokens language long update \(update): hidden cosine="
                    + "\(hiddenMetrics.cosine) relL2=\(hiddenMetrics.relativeL2) "
                    + "maxAbs=\(hiddenMetrics.maximumAbsoluteDifference); "
                    + "logits cosine=\(logitMetrics.cosine) "
                    + "relL2=\(logitMetrics.relativeL2) "
                    + "maxAbs=\(logitMetrics.maximumAbsoluteDifference); "
                    + "sourceMargin=\(nextMargins[index])"
            )
            #expect(hiddenMetrics.cosine > 0.999)
            #expect(hiddenMetrics.relativeL2 < 0.05)
            #expect(hiddenMetrics.maximumAbsoluteDifference < 0.75)
            #expect(logitMetrics.cosine > 0.999)
            #expect(logitMetrics.relativeL2 < 0.05)
            #expect(logitMetrics.maximumAbsoluteDifference < 0.3)
        }

        if let firstDecisionMismatch {
            print(
        "SkinTokens language first long-horizon decision mismatch: update="
                    + "\(firstDecisionMismatch.update) expected="
                    + "\(firstDecisionMismatch.expected) actual="
                    + "\(firstDecisionMismatch.actual) sourceMargin="
                    + "\(firstDecisionMismatch.sourceMargin) smeltExpectedLogit="
                    + "\(firstDecisionMismatch.expectedLogit) smeltActualLogit="
                    + "\(firstDecisionMismatch.actualLogit)"
            )
        }
        #expect(firstDecisionMismatch == nil)
        print(
      "SkinTokens language long-horizon summary: updates=\(maximumUpdate) "
                + "decisions=\(maximumUpdate + 1) "
                + "worstHiddenCosine=\(worstHiddenCosine) "
                + "worstHiddenRelL2=\(worstHiddenRelativeL2) "
                + "worstLogitCosine=\(worstLogitCosine) "
                + "worstLogitRelL2=\(worstLogitRelativeL2)"
        )
        let sortedDecodeMilliseconds = decodeMilliseconds.sorted()
        let totalDecodeMilliseconds = decodeMilliseconds.reduce(0, +)
        let receipt = LongContextReceipt(
            updates: maximumUpdate,
            decisions: maximumUpdate + 1,
            prefixLength: prefixLength,
            lastDecodedPosition: prefixLength + maximumUpdate - 1,
            finalContextLength: prefixLength + maximumUpdate,
            prefillMilliseconds: prefillMilliseconds,
            prefillTokensPerSecond: Double(prefixLength) * 1_000 / prefillMilliseconds,
            decodeMedianMilliseconds: percentile(
                sortedDecodeMilliseconds,
                numerator: 50
            ),
            decodeP95Milliseconds: percentile(
                sortedDecodeMilliseconds,
                numerator: 95
            ),
            decodeTokensPerSecond: Double(maximumUpdate) * 1_000
                / totalDecodeMilliseconds,
            worstHiddenCosine: worstHiddenCosine,
            worstHiddenRelativeL2: worstHiddenRelativeL2,
            worstHiddenMaximumAbsoluteDifference:
                worstHiddenMaximumAbsoluteDifference,
            worstHiddenMaximumAbsoluteDifferenceUpdate:
                worstHiddenMaximumAbsoluteDifferenceUpdate,
            worstLogitCosine: worstLogitCosine,
            worstLogitRelativeL2: worstLogitRelativeL2,
            worstLogitMaximumAbsoluteDifference:
                worstLogitMaximumAbsoluteDifference,
            worstLogitMaximumAbsoluteDifferenceUpdate:
                worstLogitMaximumAbsoluteDifferenceUpdate,
            firstDivergentLayerUpdate: firstDivergentLayerUpdate,
            firstDivergentLayer: firstDivergentLayerIndex,
            firstDecisionMismatchUpdate: firstDecisionMismatch?.update
        )
        let receiptData = try JSONEncoder().encode(receipt)
        let receiptJSON = try #require(
            String(data: receiptData, encoding: .utf8)
        )
    print("SKINNING_U0_LONG_RECEIPT \(receiptJSON)")
    }

    @Test("Dense trunk decode bricks preserve BF16 boundaries")
    func decodeLayerPrimitiveParity() throws {
        let environment = ProcessInfo.processInfo.environment
    guard let package = environment["SMELT_SKINNING_PACKAGE"],
      let referencePath = environment["SMELT_SKINNING_LANGUAGE_LONG_REFERENCE"],
              let updateText = environment["SMELT_DENSE_TRUNK_DECODE_REGION_UPDATE"],
              let update = Int(updateText),
              let layerText = environment["SMELT_DENSE_TRUNK_DECODE_REGION_LAYER"],
              let layer = Int(layerText)
        else {
            return
        }
        let manifest = try loadReference(referencePath)
        let prefixLength = try #require(manifest.incrementalPrefixLength)
        let meshEmbeddings = try verifiedFloats(
            referencePath: referencePath,
            manifest: manifest,
            tensorName: "incremental_mesh_embeddings"
        )
        let startTokens = try verifiedInt32(
            referencePath: referencePath,
            manifest: manifest,
            tensorName: "incremental_start_tokens"
        ).map(Int.init)
        let inputTokens = try verifiedInt32(
            referencePath: referencePath,
            manifest: manifest,
            tensorName: "incremental_input_tokens"
        ).map(Int.init)
        #expect(update > 0 && update <= inputTokens.count)
        #expect(layer >= 0 && layer < 28)

        let runtime = try SmeltSkeletonLanguageRuntime(packagePath: package)
        try runtime.trunk.ensureContextCapacity(3_192)
        let startEmbeddings = try runtime.embeddings(tokenIDs: startTokens)
        _ = try runtime.prefill(
            embeddings: meshEmbeddings + startEmbeddings,
            sequenceLength: prefixLength
        )
        if update > 1 {
            for index in 0..<(update - 1) {
                _ = try runtime.decodeTeacherForced(
                    tokenID: inputTokens[index],
                    position: prefixLength + index
                )
            }
        }

        // Decode lowers each dense block to nine dispatches. Several source
        // operations are deliberately fused, so probe only the boundaries
        // that actually exist in the compiled graph.
        let dispatchBase = layer * 9
        let prefix = "incremental_\(update)_layer\(layer)"
        let position = prefixLength + update - 1
        let cacheLength = position + 1
        let requests: [SmeltTrunkDispatchCaptureRequest] = [
      .init(
        label: "\(prefix)_input_norm", afterDispatch: dispatchBase + 1, source: .slot("normOutBuf"),
        count: 896),
      .init(
        label: "\(prefix)_q_projection", afterDispatch: dispatchBase + 2, source: .slot("attnQBuf"),
        count: 2_048),
      .init(
        label: "\(prefix)_k_projection", afterDispatch: dispatchBase + 3, source: .slot("attnKBuf"),
        count: 1_024),
      .init(
        label: "\(prefix)_v_projection", afterDispatch: dispatchBase + 2,
        source: .slotOffset("valCache_\(layer)", elementOffset: position * 1_024), count: 1_024),
      .init(
        label: "\(prefix)_q_roped", afterDispatch: dispatchBase + 3, source: .slot("attnOutBuf"),
        count: 2_048),
      .init(
        label: "\(prefix)_k_roped", afterDispatch: dispatchBase + 4,
        source: .slotOffset("keyCache_\(layer)", elementOffset: position * 1_024), count: 1_024),
      .init(
        label: "\(prefix)_key_cache", afterDispatch: dispatchBase + 4,
        source: .slot("keyCache_\(layer)"), count: cacheLength * 1_024),
      .init(
        label: "\(prefix)_value_cache", afterDispatch: dispatchBase + 4,
        source: .slot("valCache_\(layer)"), count: cacheLength * 1_024),
      .init(
        label: "\(prefix)_attention_output", afterDispatch: dispatchBase + 5,
        source: .slot("attnGateBuf"), count: 2_048),
      .init(
        label: "\(prefix)_attention_residual", afterDispatch: dispatchBase + 6, source: .alternate,
        count: 896),
      .init(
        label: "\(prefix)_post_attention_norm", afterDispatch: dispatchBase + 7,
        source: .slot("normOutBuf"), count: 896),
      .init(
        label: "\(prefix)_swiglu", afterDispatch: dispatchBase + 8, source: .slot("ffnIntBuf"),
        count: 3_072),
      .init(
        label: "\(prefix)_layer_output", afterDispatch: dispatchBase + 9, source: .alternate,
        count: 896),
        ]
        let actual = try runtime.decodeTeacherForcedCapturingDispatches(
            tokenID: inputTokens[update - 1],
            position: position,
            requests: requests
        )
        let expectedLayerOutputs = try verifiedFloats(
            referencePath: referencePath,
            manifest: manifest,
            tensorName: "incremental_\(update)_layer_outputs"
        )
        let layerOutputStart = layer * 896
        let expectedLayerOutput = Array(
            expectedLayerOutputs[layerOutputStart..<(layerOutputStart + 896)]
        )
        let expectedLayerInput: [Float]
        if layer == 0 {
            expectedLayerInput = try runtime.embeddings(
                tokenIDs: [inputTokens[update - 1]]
            )
        } else {
            let layerInputStart = (layer - 1) * 896
            expectedLayerInput = Array(
                expectedLayerOutputs[layerInputStart..<(layerInputStart + 896)]
            )
        }
        let expectedOProjection = try verifiedFloats(
            referencePath: referencePath,
            manifest: manifest,
            tensorName: "\(prefix)_o_projection"
        )
        let expectedAttentionResidual = zip(
            expectedLayerInput,
            expectedOProjection
        ).map { residual, update in
            SmeltBF16.decode(SmeltBF16.encode(residual + update))
        }
        func tokenMajorCache(_ source: [Float]) -> [Float] {
            let heads = 8
            let headDimension = 128
            #expect(source.count == cacheLength * heads * headDimension)
            var result = [Float](repeating: 0, count: source.count)
            for head in 0..<heads {
                for token in 0..<cacheLength {
                    let sourceBase = (head * cacheLength + token) * headDimension
                    let resultBase = (token * heads + head) * headDimension
                    result.replaceSubrange(
                        resultBase..<(resultBase + headDimension),
                        with: source[sourceBase..<(sourceBase + headDimension)]
                    )
                }
            }
            return result
        }
        let expectedKeyCache = tokenMajorCache(
            try verifiedFloats(
                referencePath: referencePath,
                manifest: manifest,
                tensorName: "\(prefix)_key_cache"
            )
        )
        let expectedValueCache = tokenMajorCache(
            try verifiedFloats(
                referencePath: referencePath,
                manifest: manifest,
                tensorName: "\(prefix)_value_cache"
            )
        )

        for request in requests {
            let values = try #require(actual[request.label])
            let expected: [Float]
            switch request.label {
            case "\(prefix)_attention_residual":
                expected = expectedAttentionResidual
            case "\(prefix)_layer_output":
                expected = expectedLayerOutput
            case "\(prefix)_key_cache":
                expected = expectedKeyCache
            case "\(prefix)_value_cache":
                expected = expectedValueCache
            default:
                expected = try verifiedFloats(
                    referencePath: referencePath,
                    manifest: manifest,
                    tensorName: request.label
                )
            }
            let regionMetrics = metrics(actual: values, expected: expected)
            print(
                "Dense trunk decode boundary \(request.label): exact="
                    + "\(regionMetrics.bitExactCount)/\(expected.count) relL2="
                    + "\(regionMetrics.relativeL2) maxAbs="
                    + "\(regionMetrics.maximumAbsoluteDifference)"
            )
            let mismatchIndices = values.indices.lazy.filter {
                values[$0].bitPattern != expected[$0].bitPattern
            }.prefix(10)
            for mismatch in mismatchIndices {
                print(
                    "Dense trunk decode mismatch \(request.label)[\(mismatch)]: "
                        + "expected=\(expected[mismatch]) actual=\(values[mismatch])"
                )
            }
            #expect(regionMetrics.bitExactCount == expected.count)
        }
    }

    @Test("Dense trunk bricks preserve per-layer BF16 boundaries")
    func layerPrimitiveParity() throws {
        let environment = ProcessInfo.processInfo.environment
    guard let package = environment["SMELT_SKINNING_PACKAGE"],
      let referencePath = environment["SMELT_SKINNING_LANGUAGE_REGION_REFERENCE"]
        else {
            return
        }
        let manifest = try loadReference(referencePath)
        let layer = Int(environment["SMELT_DENSE_TRUNK_REGION_LAYER"] ?? "0") ?? 0
        let dispatchBase = layer * 17
        let prefix = "layer\(layer)"
        let sequenceLength = manifest.sequenceLength
        let hidden = 896
        let embeddings = try verifiedFloats(
            referencePath: referencePath,
            manifest: manifest,
            tensorName: "input_embeddings"
        )
        var requests: [SmeltTrunkDispatchCaptureRequest] = [
      .init(
        label: "\(prefix)_input_norm", afterDispatch: dispatchBase + 1, source: .slot("normOutBuf"),
        count: sequenceLength * hidden),
      .init(
        label: "\(prefix)_q_projection", afterDispatch: dispatchBase + 2, source: .slot("attnQBuf"),
        count: sequenceLength * 2_048),
      .init(
        label: "\(prefix)_k_projection", afterDispatch: dispatchBase + 3, source: .slot("attnKBuf"),
        count: sequenceLength * 1_024),
      .init(
        label: "\(prefix)_v_projection", afterDispatch: dispatchBase + 4,
        source: .slot("valCache_\(layer)"), count: sequenceLength * 1_024),
      .init(
        label: "\(prefix)_q_norm", afterDispatch: dispatchBase + 5, source: .slot("attnQBuf"),
        count: sequenceLength * 2_048),
      .init(
        label: "\(prefix)_k_norm", afterDispatch: dispatchBase + 6, source: .slot("attnKBuf"),
        count: sequenceLength * 1_024),
      .init(
        label: "\(prefix)_q_roped", afterDispatch: dispatchBase + 7, source: .slot("attnQBuf"),
        count: sequenceLength * 2_048),
      .init(
        label: "\(prefix)_k_roped", afterDispatch: dispatchBase + 8,
        source: .slot("keyCache_\(layer)"), count: sequenceLength * 1_024),
      .init(
        label: "\(prefix)_attention_output", afterDispatch: dispatchBase + 9,
        source: .slot("attnOutBuf"), count: sequenceLength * 2_048),
      .init(
        label: "\(prefix)_o_projection", afterDispatch: dispatchBase + 10,
        source: .slot("normOutBuf"), count: sequenceLength * hidden),
      .init(
        label: "\(prefix)_post_attention_norm", afterDispatch: dispatchBase + 12,
        source: .slot("normOutBuf"), count: sequenceLength * hidden),
      .init(
        label: "\(prefix)_gate_projection", afterDispatch: dispatchBase + 13,
        source: .slot("ffnGateBuf"), count: sequenceLength * 3_072),
      .init(
        label: "\(prefix)_up_projection", afterDispatch: dispatchBase + 14,
        source: .slot("ffnUpBuf"), count: sequenceLength * 3_072),
      .init(
        label: "\(prefix)_swiglu", afterDispatch: dispatchBase + 15, source: .slot("ffnIntBuf"),
        count: sequenceLength * 3_072),
      .init(
        label: "\(prefix)_down_projection", afterDispatch: dispatchBase + 16,
        source: .slot("normOutBuf"), count: sequenceLength * hidden),
        ]
        if layer == 0 {
            requests.append(
                .init(
                    label: "\(prefix)_attention_residual",
                    afterDispatch: dispatchBase + 11,
                    source: .alternate,
                    count: sequenceLength * hidden
                )
            )
        }
        let runtime = try SmeltSkeletonLanguageRuntime(packagePath: package)
        let actual = try runtime.trunk.captureTrunkPrefillDispatchOutputs(
            embeddings: embeddings,
            seqLen: sequenceLength,
            requests: requests
        )
        for request in requests {
            let values = try #require(actual[request.label])
            let expected: [Float]
            if request.label == "layer0_attention_residual" {
                let projection = try verifiedFloats(
                    referencePath: referencePath,
                    manifest: manifest,
                    tensorName: "layer0_o_projection"
                )
                expected = zip(embeddings, projection).map { residual, update in
                    SmeltBF16.decode(SmeltBF16.encode(residual + update))
                }
            } else {
                expected = try verifiedFloats(
                    referencePath: referencePath,
                    manifest: manifest,
                    tensorName: request.label
                )
            }
            let regionMetrics = metrics(actual: values, expected: expected)
            print(
                "Dense trunk boundary \(request.label): exact="
                    + "\(regionMetrics.bitExactCount)/\(expected.count) relL2="
                    + "\(regionMetrics.relativeL2) maxAbs="
                    + "\(regionMetrics.maximumAbsoluteDifference)"
            )
            let mismatchIndices = values.indices.lazy.filter {
                values[$0].bitPattern != expected[$0].bitPattern
            }.prefix(10)
            for mismatch in mismatchIndices {
                print(
                    "Dense trunk mismatch \(request.label)[\(mismatch)]: expected="
                        + "\(expected[mismatch]) actual=\(values[mismatch]) expectedBits="
                        + "0x\(String(expected[mismatch].bitPattern, radix: 16)) actualBits="
                        + "0x\(String(values[mismatch].bitPattern, radix: 16))"
                )
            }
            #expect(regionMetrics.bitExactCount == expected.count)
        }
    }
}
