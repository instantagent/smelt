import CryptoKit
import Foundation
import Metal
import Testing

@testable import SmeltRuntime

@Suite("Smelt skin decoder runtime", .serialized)
struct SmeltSkinDecoderTests {
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
        let tensors: [String: Tensor]

        enum CodingKeys: String, CodingKey {
            case schema
            case sourceCommit = "source_commit"
            case checkpointSHA256 = "checkpoint_sha256"
            case tensors
        }
    }

    private struct Metrics {
        let cosine: Double
        let relativeL2: Double
        let maximumAbsoluteDifference: Float
    }

    private struct BitMismatchSummary {
        let count: Int
        let firstIndex: Int?
        let maximumAbsoluteDifference: Float
    }

    private struct ProductionDecoderFixture {
    let artifact: SmeltComponentArtifact
    let decoder: SmeltSkinDecoder
        let pointNormals: [Float]
        let conditionTokens: [Float]
        let jointIndices: [[UInt32]]
    }

    private enum DiagnosticError: Error {
        case invalidKernelPolicy(String)
    case metalUnavailable
    }

    @Test("Reduced decoder passes every pinned source boundary")
    func reducedSourceParity() throws {
    guard let package = ProcessInfo.processInfo.environment["SMELT_SKINNING_PACKAGE"],
            let reference = ProcessInfo.processInfo.environment[
        "SMELT_SKINNING_SKIN_DECODER_REFERENCE"
            ]
        else {
            return
        }
        let manifest = try loadManifest(reference)
    let artifact = try SmeltComponentArtifact(path: package, verify: true)
        let runtime = try SmeltSkinDecoder(artifact: artifact)
        let indices = try verifiedInt32(
            path: reference,
            manifest: manifest,
            name: "indices"
        ).map(UInt32.init(bitPattern:))
        let condition = try verifiedFloats(
            path: reference,
            manifest: manifest,
            name: "condition_tokens"
        )
        let pointNormals = try verifiedFloats(
            path: reference,
            manifest: manifest,
            name: "point_normals"
        )
        let capture = try runtime.decodeReduced(
            indices: indices,
            conditionTokens: condition,
            pointNormals: pointNormals
        )
        try check(
            actual: capture.fsqCodes,
            expected: verifiedFloats(
                path: reference,
                manifest: manifest,
                name: "fsq_codes"
            ),
            label: "fsq",
            minimumCosine: 0.999,
            maximumRelativeL2: 0.04,
            maximumAbsoluteDifference: 0.1
        )
        try check(
            actual: capture.postQuantOutput,
            expected: verifiedFloats(
                path: reference,
                manifest: manifest,
                name: "post_quant_output"
            ),
            label: "post_quant",
            minimumCosine: 0.999,
            maximumRelativeL2: 0.04,
            maximumAbsoluteDifference: 0.2
        )
        let sourceLayers = try verifiedFloats(
            path: reference,
            manifest: manifest,
            name: "self_layer_outputs"
        )
        #expect(capture.selfLayerOutputs.count == 10)
        let elementsPerLayer = 7 * 768
        for layer in 0..<10 {
            let start = layer * elementsPerLayer
            try check(
                actual: capture.selfLayerOutputs[layer],
                expected: Array(sourceLayers[start..<(start + elementsPerLayer)]),
                label: "self.\(layer)",
                minimumCosine: 0.999,
                maximumRelativeL2: 0.04,
                maximumAbsoluteDifference: 2
            )
        }
        try check(
            actual: capture.projectedQueries,
            expected: verifiedFloats(
                path: reference,
                manifest: manifest,
                name: "projected_queries"
            ),
            label: "query_projection",
            minimumCosine: 0.999,
            maximumRelativeL2: 0.04,
            maximumAbsoluteDifference: 0.2
        )
        try check(
            actual: capture.crossOutput,
            expected: verifiedFloats(
                path: reference,
                manifest: manifest,
                name: "cross_output"
            ),
            label: "cross",
            minimumCosine: 0.999,
            maximumRelativeL2: 0.04,
            maximumAbsoluteDifference: 2
        )
        try check(
            actual: capture.normalizedOutput,
            expected: verifiedFloats(
                path: reference,
                manifest: manifest,
                name: "normalized_output"
            ),
            label: "final_norm",
            minimumCosine: 0.999,
            maximumRelativeL2: 0.04,
            maximumAbsoluteDifference: 0.2
        )
        try check(
            actual: capture.rawLogits,
            expected: verifiedFloats(
                path: reference,
                manifest: manifest,
                name: "raw_logits"
            ),
            label: "raw_logits",
            minimumCosine: 0.999,
            maximumRelativeL2: 0.04,
            maximumAbsoluteDifference: 0.2
        )
        try check(
            actual: capture.probabilities,
            expected: verifiedFloats(
                path: reference,
                manifest: manifest,
                name: "probabilities"
            ),
            label: "probabilities",
            minimumCosine: 0.999,
            maximumRelativeL2: 0.04,
            maximumAbsoluteDifference: 0.05
        )
    }

    @Test("Production query chunking is byte-identical to the staged route")
    func productionChunkingParity() throws {
    guard let package = ProcessInfo.processInfo.environment["SMELT_SKINNING_PACKAGE"] else {
            return
        }
    let artifact = try SmeltComponentArtifact(path: package, verify: true)
        let runtime = try SmeltSkinDecoder(artifact: artifact)
        let condition = (0..<(384 * 512)).map { index in
            widenedBF16(sin(Float(index * 31 + 3) * 0.0017) * 0.19)
        }
        let pointNormals = (0..<(513 * 6)).map { index in
            sin(Float(index * 13 + 17) * 0.0071) * 0.73
        }
        let indices: [UInt32] = [0, 1, 1_234, 32_767]
        let staged = try runtime.decodeReduced(
            indices: indices,
            conditionTokens: condition,
            pointNormals: pointNormals
        ).probabilities
        let chunked = try runtime.decode(
            indices: indices,
            conditionTokens: condition,
            pointNormals: pointNormals
        )
        #expect(staged.count == 513)
        #expect(chunked.count == staged.count)
        for index in staged.indices {
            #expect(
                chunked[index].bitPattern == staged[index].bitPattern,
                "chunked decoder bit divergence at vertex \(index)"
            )
        }
    }

    @Test("Production 54,000-vertex decoder smoke")
    func productionShapeSmoke() throws {
    guard ProcessInfo.processInfo.environment["SMELT_SKINNING_FULL_SMOKE"] == "1",
      let package = ProcessInfo.processInfo.environment["SMELT_SKINNING_PACKAGE"]
        else {
            return
        }
    let artifact = try SmeltComponentArtifact(path: package, verify: true)
        let runtime = try SmeltSkinDecoder(artifact: artifact)
        let condition = (0..<(384 * 512)).map { index in
            widenedBF16(sin(Float(index * 19 + 23) * 0.0013) * 0.17)
        }
        let pointNormals = (0..<(54_000 * 6)).map { index in
            sin(Float(index * 11 + 29) * 0.0021) * 0.79
        }
        let output = try runtime.decode(
            indices: [0, 1, 1_234, 32_767],
            conditionTokens: condition,
            pointNormals: pointNormals
        )
        #expect(output.count == 54_000)
        #expect(output.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 })
    }

    @Test("Multi-joint correctness route preserves individual decode bits")
    func multiJointParity() throws {
    guard let package = ProcessInfo.processInfo.environment["SMELT_SKINNING_PACKAGE"] else {
            return
        }
    let artifact = try SmeltComponentArtifact(path: package, verify: true)
        let runtime = try SmeltSkinDecoder(artifact: artifact)
        let condition = (0..<(384 * 512)).map { index in
            widenedBF16(sin(Float(index * 7 + 41) * 0.0023) * 0.21)
        }
        let pointNormals = (0..<(33 * 6)).map { index in
            sin(Float(index * 43 + 13) * 0.0059) * 0.67
        }
        let joints: [[UInt32]] = [
            [0, 1, 2, 3],
            [32_767, 4_096, 512, 64],
        ]
        let individual = try joints.map {
            try runtime.decode(
                indices: $0,
                conditionTokens: condition,
                pointNormals: pointNormals
            )
        }
        let combined = try runtime.decodeJoints(
            indicesByJoint: joints,
            conditionTokens: condition,
            pointNormals: pointNormals
        )
        #expect(combined.vertexCount == 33)
        #expect(combined.jointCount == 2)
        #expect(combined.values.count == 66)
        for vertex in 0..<33 {
            for joint in 0..<2 {
                #expect(
                    combined.values[vertex * 2 + joint].bitPattern
                        == individual[joint][vertex].bitPattern
                )
            }
        }
    }

    @Test("Production decoder policies match on exact generated inputs")
    func productionDecoderPolicyParity() throws {
        guard
            ProcessInfo.processInfo.environment[
        "SMELT_SKINNING_FULL_DECODER_PARITY"
            ] == "1",
      let package = ProcessInfo.processInfo.environment["SMELT_SKINNING_PACKAGE"],
            let fixture = ProcessInfo.processInfo.environment[
        "SMELT_SKINNING_GLB_FIXTURE"
            ]
        else {
            return
        }
        let production = try productionDecoderFixture(
            package: package,
            glb: fixture
        )

        let candidateJoints = Set([11, 33, 36, 44, 55])
        let prefixCount = (try #require(candidateJoints.max())) + 1
        let jointIndices = Array(production.jointIndices.prefix(prefixCount))
        let policies: [(String, SmeltSkinDecoderKernelPolicy)] = [
            ("optimized-repeat", .optimized),
            (
                "scalar-attention",
                .init(
                    maximumDenseRowsPerThreadgroup: 8,
                    fuseDenseEpilogue: true,
                    attentionQueryTile: 1
                )
            ),
            (
                "scalar-self-attention",
                .init(
                    maximumDenseRowsPerThreadgroup: 8,
                    fuseDenseEpilogue: true,
                    selfAttentionQueryTile: 1,
                    crossAttentionQueryTile: 8
                )
            ),
            (
                "scalar-cross-attention",
                .init(
                    maximumDenseRowsPerThreadgroup: 8,
                    fuseDenseEpilogue: true,
                    selfAttentionQueryTile: 8,
                    crossAttentionQueryTile: 1
                )
            ),
            (
                "scalar-dense",
                .init(
                    maximumDenseRowsPerThreadgroup: 1,
                    fuseDenseEpilogue: true,
                    attentionQueryTile: 8
                )
            ),
            (
                "staged-epilogue",
                .init(
                    maximumDenseRowsPerThreadgroup: 8,
                    fuseDenseEpilogue: false,
                    attentionQueryTile: 8
                )
            ),
            (
                "fully-staged",
                .init(
                    maximumDenseRowsPerThreadgroup: 1,
                    fuseDenseEpilogue: false,
                    attentionQueryTile: 1
                )
            ),
        ]
    let reference = try production.decoder.decodeJoints(
            indicesByJoint: jointIndices,
            conditionTokens: production.conditionTokens,
            pointNormals: production.pointNormals
        )
        for (name, policy) in policies {
            let decoder = try SmeltSkinDecoder(
        artifact: production.artifact,
                kernelPolicy: policy
            )
            let actual = try decoder.decodeJoints(
                indicesByJoint: jointIndices,
                conditionTokens: production.conditionTokens,
                pointNormals: production.pointNormals
            )
            let mismatch = bitMismatch(
                actual: actual.values,
                expected: reference.values
            )
            let candidateMismatchCount = candidateJoints.reduce(0) { count, joint in
                count
                    + (0..<actual.vertexCount).reduce(0) { subtotal, vertex in
                        let index = vertex * actual.jointCount + joint
                        return subtotal
                            + (actual.values[index].bitPattern
                                == reference.values[index].bitPattern ? 0 : 1)
                    }
            }
            let firstMismatch = mismatch.firstIndex.map(String.init) ?? "none"
            print(
        "SKINNING_DECODER_POLICY_PARITY policy=\(name) "
                    + "mismatches=\(mismatch.count) "
                    + "candidateMismatches=\(candidateMismatchCount) "
                    + "first=\(firstMismatch) "
                    + "maxAbs=\(mismatch.maximumAbsoluteDifference)"
            )
            #expect(
                mismatch.count == 0,
                "\(name) first bit mismatch: \(firstMismatch)"
            )
        }
    }

    @Test("Production repeated-joint decoder stress is bit-exact")
    func productionRepeatedJointStress() throws {
        guard
            ProcessInfo.processInfo.environment[
        "SMELT_SKINNING_DECODER_STRESS"
            ] == "1",
      let package = ProcessInfo.processInfo.environment["SMELT_SKINNING_PACKAGE"],
            let glb = ProcessInfo.processInfo.environment[
        "SMELT_SKINNING_GLB_FIXTURE"
            ]
        else {
            return
        }
        let production = try productionDecoderFixture(package: package, glb: glb)
        let environment = ProcessInfo.processInfo.environment
    let repeatCount = Int(environment["SMELT_SKINNING_DECODER_STRESS_REPEATS"] ?? "381")
    let joint = Int(environment["SMELT_SKINNING_DECODER_STRESS_JOINT"] ?? "11")
        let policyName =
      environment["SMELT_SKINNING_DECODER_STRESS_POLICY"]
            ?? "optimized"
        guard let repeatCount, repeatCount > 1,
            let joint, production.jointIndices.indices.contains(joint)
        else {
            throw SmeltSkinDecoderError.invalidJointCount(repeatCount ?? 0)
        }
        let decoder = try SmeltSkinDecoder(
      artifact: production.artifact,
            kernelPolicy: try kernelPolicy(named: policyName)
        )
        let actual = try decoder.decodeJoints(
            indicesByJoint: Array(
                repeating: production.jointIndices[joint],
                count: repeatCount
            ),
            conditionTokens: production.conditionTokens,
            pointNormals: production.pointNormals
        )
        var reference = [Float](repeating: 0, count: actual.vertexCount)
        for vertex in 0..<actual.vertexCount {
            reference[vertex] = actual.values[vertex * repeatCount]
        }
        let referenceSHA = sha256(reference)
        #expect(
            referenceSHA
                == "af166532814788a4841d1eef45e3cc5b3cb6651cf89c00c45c3d2c538121d303"
        )
        #expect(reference[2_742].bitPattern == 0x3A39_D3C6)

        var mismatchCount = 0
        var firstRepetition: Int?
        var firstVertex: Int?
        var firstActualBits: UInt32?
        for repetition in 1..<repeatCount {
            for vertex in 0..<actual.vertexCount {
                let value = actual.values[vertex * repeatCount + repetition]
                if value.bitPattern != reference[vertex].bitPattern {
                    mismatchCount += 1
                    if firstRepetition == nil {
                        firstRepetition = repetition
                        firstVertex = vertex
                        firstActualBits = value.bitPattern
                    }
                }
            }
        }
        print(
      "SKINNING_DECODER_STRESS policy=\(policyName) "
                + "joint=\(joint) repeats=\(repeatCount) "
                + "mismatches=\(mismatchCount) "
                + "firstRepetition=\(firstRepetition.map(String.init) ?? "none") "
                + "firstVertex=\(firstVertex.map(String.init) ?? "none") "
                + "firstActualBits=\(firstActualBits.map(String.init) ?? "none")"
        )
        #expect(mismatchCount == 0)
    }

    @Test("Production decoder prefix-cycle stress is bit-exact")
    func productionPrefixCycleStress() throws {
        guard
            ProcessInfo.processInfo.environment[
        "SMELT_SKINNING_DECODER_PREFIX_STRESS"
            ] == "1",
      let package = ProcessInfo.processInfo.environment["SMELT_SKINNING_PACKAGE"],
            let glb = ProcessInfo.processInfo.environment[
        "SMELT_SKINNING_GLB_FIXTURE"
            ]
        else {
            return
        }
        let production = try productionDecoderFixture(package: package, glb: glb)
        let environment = ProcessInfo.processInfo.environment
    let cycleCount = Int(environment["SMELT_SKINNING_DECODER_STRESS_CYCLES"] ?? "32")
    let prefixCount = Int(environment["SMELT_SKINNING_DECODER_STRESS_PREFIX"] ?? "12")
        let policyName =
      environment["SMELT_SKINNING_DECODER_STRESS_POLICY"]
            ?? "optimized"
        guard let cycleCount, cycleCount > 1,
            let prefixCount, prefixCount > 0,
            prefixCount <= production.jointIndices.count
        else {
            throw SmeltSkinDecoderError.invalidJointCount(cycleCount ?? 0)
        }
        let prefix = Array(production.jointIndices.prefix(prefixCount))
        let decoder = try SmeltSkinDecoder(
      artifact: production.artifact,
            kernelPolicy: try kernelPolicy(named: policyName)
        )
        let actual = try decoder.decodeJoints(
            indicesByJoint: Array(repeating: prefix, count: cycleCount).flatMap { $0 },
            conditionTokens: production.conditionTokens,
            pointNormals: production.pointNormals
        )
        let totalColumns = cycleCount * prefixCount
        var referenceJoint11 = [Float](repeating: 0, count: actual.vertexCount)
        if prefixCount > 11 {
            for vertex in 0..<actual.vertexCount {
                referenceJoint11[vertex] = actual.values[vertex * totalColumns + 11]
            }
            #expect(
                sha256(referenceJoint11)
                    == "af166532814788a4841d1eef45e3cc5b3cb6651cf89c00c45c3d2c538121d303"
            )
            #expect(referenceJoint11[2_742].bitPattern == 0x3A39_D3C6)
        }

        var mismatchCount = 0
        var firstCycle: Int?
        var firstPrefixJoint: Int?
        var firstVertex: Int?
        var firstActualBits: UInt32?
        for cycle in 1..<cycleCount {
            for prefixJoint in 0..<prefixCount {
                for vertex in 0..<actual.vertexCount {
                    let referenceIndex = vertex * totalColumns + prefixJoint
                    let actualIndex = referenceIndex + cycle * prefixCount
                    let expected = actual.values[referenceIndex]
                    let value = actual.values[actualIndex]
                    if value.bitPattern != expected.bitPattern {
                        mismatchCount += 1
                        if firstCycle == nil {
                            firstCycle = cycle
                            firstPrefixJoint = prefixJoint
                            firstVertex = vertex
                            firstActualBits = value.bitPattern
                        }
                    }
                }
            }
        }
        print(
      "SKINNING_DECODER_PREFIX_STRESS policy=\(policyName) "
                + "cycles=\(cycleCount) prefix=\(prefixCount) "
                + "mismatches=\(mismatchCount) "
                + "firstCycle=\(firstCycle.map(String.init) ?? "none") "
                + "firstPrefixJoint=\(firstPrefixJoint.map(String.init) ?? "none") "
                + "firstVertex=\(firstVertex.map(String.init) ?? "none") "
                + "firstActualBits=\(firstActualBits.map(String.init) ?? "none")"
        )
        #expect(mismatchCount == 0)
    }

    private func productionDecoderFixture(
        package: String,
        glb: String
    ) throws -> ProductionDecoderFixture {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw DiagnosticError.metalUnavailable
    }
    let artifact = try SmeltComponentArtifact(path: package, verify: true)
    let meshEncoder = try SmeltMeshEncoder(artifact: artifact, device: device)
    let conditionEncoder = try SmeltSkinConditionEncoder(
      artifact: artifact,
      device: device
    )
    let generator = SmeltSkeletonGenerator(
      languageModel: try SmeltSkeletonLanguageRuntime(
        packagePath: package,
        device: device,
        verifyPackage: true
      )
    )
    let decoder = try SmeltSkinDecoder(artifact: artifact, device: device)
        let mesh = try SmeltGLB.readMesh(from: URL(fileURLWithPath: glb))
        let sampling = try SmeltMeshGeometry.sample(mesh: mesh, seed: 0)
    let meshEncoding = try meshEncoder.encode(
            pointNormals: sampling.pointNormals
        )
    let condition = try conditionEncoder.encode(
            pointNormals: sampling.pointNormals
        )
    let generation = try generator.generate(
            meshEmbeddings: meshEncoding.embeddings,
            startTokenIDs: [
        SmeltSkeletonVocabulary.skeletonBOS,
        SmeltSkeletonVocabulary.noClass,
            ],
            configuration: .init(
                policyMode: .validated,
                decodeMode: .greedy
            )
        )
        #expect(generation.skinCodeIndices.count == 381)
        #expect(
            sha256(generation.tokenIDs)
                == "43e24a52a5ccf2a74deec5c0fd5b1a312672df3c21af954bfc52db8c27711f25"
        )
        return ProductionDecoderFixture(
      artifact: artifact,
      decoder: decoder,
            pointNormals: sampling.pointNormals,
            conditionTokens: condition.conditionTokens,
            jointIndices: generation.skinCodeIndices.map { indices in
                indices.map(UInt32.init)
            }
        )
    }

    private func kernelPolicy(
        named name: String
    ) throws -> SmeltSkinDecoderKernelPolicy {
        switch name {
        case "optimized":
            return .optimized
        case "scalar-attention":
            return .init(
                maximumDenseRowsPerThreadgroup: 8,
                fuseDenseEpilogue: true,
                attentionQueryTile: 1
            )
        case "scalar-self-attention":
            return .init(
                maximumDenseRowsPerThreadgroup: 8,
                fuseDenseEpilogue: true,
                selfAttentionQueryTile: 1,
                crossAttentionQueryTile: 8
            )
        case "scalar-cross-attention":
            return .init(
                maximumDenseRowsPerThreadgroup: 8,
                fuseDenseEpilogue: true,
                selfAttentionQueryTile: 8,
                crossAttentionQueryTile: 1
            )
        case "scalar-dense":
            return .init(
                maximumDenseRowsPerThreadgroup: 1,
                fuseDenseEpilogue: true,
                attentionQueryTile: 8
            )
        case "staged-epilogue":
            return .init(
                maximumDenseRowsPerThreadgroup: 8,
                fuseDenseEpilogue: false,
                attentionQueryTile: 8
            )
        case "fully-staged":
            return .init(
                maximumDenseRowsPerThreadgroup: 1,
                fuseDenseEpilogue: false,
                attentionQueryTile: 1
            )
        default:
            throw DiagnosticError.invalidKernelPolicy(name)
        }
    }

    private func loadManifest(_ path: String) throws -> ReferenceManifest {
        let data = try Data(contentsOf: URL(fileURLWithPath: "\(path)/manifest.json"))
        let manifest = try JSONDecoder().decode(ReferenceManifest.self, from: data)
    #expect(manifest.schema == "smelt.skintokens.decoder-reference.v1")
        #expect(manifest.sourceCommit == "273b691d35989d71cd17ff2895fdc735097b92d1")
        #expect(
            manifest.checkpointSHA256
                == "f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692"
        )
        return manifest
    }

    private func verifiedFloats(
        path: String,
        manifest: ReferenceManifest,
        name: String
    ) throws -> [Float] {
        let descriptor = try #require(manifest.tensors[name])
        #expect(descriptor.dtype == "float32")
        let file = "\(path)/\(descriptor.file)"
        try verifyHash(file: file, expected: descriptor.sha256)
        let tensor = try NpyLoader.load(path: file)
        #expect(tensor.dtype == "f4")
        #expect(tensor.elementCount == descriptor.shape.reduce(1, *))
        return Array(
            UnsafeBufferPointer(
                start: tensor.fp32Pointer,
                count: tensor.elementCount
            )
        )
    }

    private func verifiedInt32(
        path: String,
        manifest: ReferenceManifest,
        name: String
    ) throws -> [Int32] {
        let descriptor = try #require(manifest.tensors[name])
        #expect(descriptor.dtype == "int32")
        let file = "\(path)/\(descriptor.file)"
        try verifyHash(file: file, expected: descriptor.sha256)
        let tensor = try NpyLoader.load(path: file)
        #expect(tensor.dtype == "i4")
        #expect(tensor.elementCount == descriptor.shape.reduce(1, *))
        return Array(
            UnsafeBufferPointer(
                start: tensor.data.bindMemory(
                    to: Int32.self,
                    capacity: tensor.elementCount
                ),
                count: tensor.elementCount
            )
        )
    }

    private func verifyHash(file: String, expected: String) throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: file))
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        #expect(digest == expected)
    }

    private func sha256<T>(_ values: [T]) -> String {
        values.withUnsafeBytes { bytes in
            SHA256.hash(data: Data(bytes)).map {
                String(format: "%02x", $0)
            }.joined()
        }
    }

    private func bitMismatch(
        actual: [Float],
        expected: [Float]
    ) -> BitMismatchSummary {
        precondition(actual.count == expected.count)
        var count = 0
        var firstIndex: Int?
        var maximumAbsoluteDifference: Float = 0
        for index in actual.indices
        where actual[index].bitPattern != expected[index].bitPattern {
            count += 1
            if firstIndex == nil { firstIndex = index }
            maximumAbsoluteDifference = max(
                maximumAbsoluteDifference,
                abs(actual[index] - expected[index])
            )
        }
        return BitMismatchSummary(
            count: count,
            firstIndex: firstIndex,
            maximumAbsoluteDifference: maximumAbsoluteDifference
        )
    }

    private func check(
        actual: [Float],
        expected: [Float],
        label: String,
        minimumCosine: Double,
        maximumRelativeL2: Double,
        maximumAbsoluteDifference: Float
    ) throws {
        let value = metrics(actual: actual, expected: expected)
        print(
      "SkinTokens decoder \(label): cosine=\(value.cosine) "
                + "relL2=\(value.relativeL2) "
                + "maxAbs=\(value.maximumAbsoluteDifference)"
        )
        #expect(value.cosine > minimumCosine)
        #expect(value.relativeL2 < maximumRelativeL2)
        #expect(value.maximumAbsoluteDifference < maximumAbsoluteDifference)
    }

    private func metrics(actual: [Float], expected: [Float]) -> Metrics {
        precondition(actual.count == expected.count)
        var dot: Double = 0
        var actualSquared: Double = 0
        var expectedSquared: Double = 0
        var differenceSquared: Double = 0
        var maximum: Float = 0
        for index in actual.indices {
            let actualValue = Double(actual[index])
            let expectedValue = Double(expected[index])
            let difference = actualValue - expectedValue
            dot += actualValue * expectedValue
            actualSquared += actualValue * actualValue
            expectedSquared += expectedValue * expectedValue
            differenceSquared += difference * difference
            maximum = max(maximum, abs(actual[index] - expected[index]))
        }
        return Metrics(
            cosine: dot / sqrt(actualSquared * expectedSquared),
            relativeL2: sqrt(differenceSquared / expectedSquared),
            maximumAbsoluteDifference: maximum
        )
    }

    private func widenedBF16(_ value: Float) -> Float {
        let bits = value.bitPattern
        let leastSignificantRetainedBit = (bits >> 16) & 1
        let rounded = bits &+ 0x7FFF &+ leastSignificantRetainedBit
        return Float(bitPattern: rounded & 0xFFFF_0000)
    }
}
