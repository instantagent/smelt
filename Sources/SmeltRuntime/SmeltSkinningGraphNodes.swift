import CryptoKit
import Foundation
import Metal

enum SmeltSkinningGraphNodes {
    static let registrations: [SmeltGraphNodeRegistration] = [
        .init(entrypoint: "glb.decode", make: { GLBDecode() }),
        .init(entrypoint: "surface.sample", make: { SurfaceSample() }),
        .init(entrypoint: "mesh.encode", make: { MeshEncode() }),
        .init(entrypoint: "condition.encode", make: { ConditionEncode() }),
        .init(entrypoint: "sequence.generate", make: { SequenceGenerate() }),
        .init(entrypoint: "skin.neighbors", make: { SkinNeighbors() }),
        .init(entrypoint: "skin.decode", make: { SkinDecode() }),
        .init(entrypoint: "skin.transfer", make: { SkinTransfer() }),
        .init(entrypoint: "glb.encode", make: { GLBEncode() }),
    ]

    struct GLBDecode: SmeltGraphNodeRuntime {
        func run(
            inputs: [String: SmeltGraphValue],
            context: SmeltGraphExecutionContext
        ) throws -> [String: SmeltGraphValue] {
            let url = try inputs.required("input", as: URL.self, node: "glb.decode")
            return ["mesh": .init(try SmeltGLB.readMesh(from: url))]
        }
    }

    struct SurfaceSample: SmeltGraphNodeRuntime {
        func run(
            inputs: [String: SmeltGraphValue],
            context: SmeltGraphExecutionContext
        ) throws -> [String: SmeltGraphValue] {
            let mesh = try inputs.required("mesh", as: SmeltMesh.self, node: "surface.sample")
            let seed = try unsignedOption("sampling-seed", context: context)
            let sampled = try SmeltMeshGeometry.sample(mesh: mesh, seed: seed)
            record(sampled.pointNormals, as: "surface.point-normals", context: context)
            context.recordEvidence(
                "surface.point-count",
                value: String(sampled.pointNormals.count / 6)
            )
            return ["sampled": .init(sampled)]
        }
    }

    struct MeshEncode: SmeltGraphNodeRuntime {
        func run(
            inputs: [String: SmeltGraphValue],
            context: SmeltGraphExecutionContext
        ) throws -> [String: SmeltGraphValue] {
            let sampled = try inputs.required(
                "sampled",
                as: SmeltSampledSurface.self,
                node: "mesh.encode"
            )
            let encoding = try meshEncoder(context).encode(pointNormals: sampled.pointNormals)
            record(encoding.embeddings, as: "mesh.embedding", context: context)
            return ["encoding": .init(encoding)]
        }
    }

    struct ConditionEncode: SmeltGraphNodeRuntime {
        func run(
            inputs: [String: SmeltGraphValue],
            context: SmeltGraphExecutionContext
        ) throws -> [String: SmeltGraphValue] {
            let sampled = try inputs.required(
                "sampled",
                as: SmeltSampledSurface.self,
                node: "condition.encode"
            )
            let condition = try conditionEncoder(context).encode(
                pointNormals: sampled.pointNormals
            )
            record(condition.conditionTokens, as: "condition.tokens", context: context)
            return ["condition": .init(condition)]
        }
    }

    struct SequenceGenerate: SmeltGraphNodeRuntime {
        func run(
            inputs: [String: SmeltGraphValue],
            context: SmeltGraphExecutionContext
        ) throws -> [String: SmeltGraphValue] {
            let encoding = try inputs.required(
                "encoding",
                as: SmeltMeshEncoding.self,
                node: "sequence.generate"
            )
            let startTokens = try skeletonTokens(context.options["skeleton-tokens"])
            let beamCount = try positiveOption("beam-count", context: context)
            let mode: SmeltSkeletonDecodeMode
            if let sampleSeedText = context.options["sample-seed"] {
                let sampleSeed = try unsignedValue(sampleSeedText, flag: "sample-seed")
                mode = beamCount > 1
                    ? .beamSampled(seed: sampleSeed, width: beamCount)
                    : .sampled(seed: sampleSeed)
            } else {
                mode = .greedy
            }
            let generation = try generator(context).generate(
                meshEmbeddings: encoding.embeddings,
                startTokenIDs: startTokens,
                configuration: .init(policyMode: .validated, decodeMode: mode)
            )
            record(generation.tokenIDs, as: "generation.tokens", context: context)
            if context.capturesEvidence {
                record(
                    generation.skinCodeIndices.flatMap { $0 },
                    as: "generation.skin-codes",
                    context: context
                )
            }
            context.recordEvidence(
                "generation.joint-count",
                value: String(generation.skinCodeIndices.count)
            )
            return ["generation": .init(generation)]
        }
    }

    struct SkinNeighbors: SmeltGraphNodeRuntime {
        func run(
            inputs: [String: SmeltGraphValue],
            context: SmeltGraphExecutionContext
        ) throws -> [String: SmeltGraphValue] {
            let mesh = try inputs.required("mesh", as: SmeltMesh.self, node: "skin.neighbors")
            let sampled = try inputs.required(
                "sampled",
                as: SmeltSampledSurface.self,
                node: "skin.neighbors"
            )
            let plan = try SmeltMeshGeometry.prepareSkinTransfer(mesh: mesh, sampled: sampled)
            return [
                "queries": .init(plan.queryPointNormals),
                "plan": .init(plan),
            ]
        }
    }

    struct SkinDecode: SmeltGraphNodeRuntime {
        func run(
            inputs: [String: SmeltGraphValue],
            context: SmeltGraphExecutionContext
        ) throws -> [String: SmeltGraphValue] {
            let generation = try inputs.required(
                "generation",
                as: SmeltSkeletonGenerationResult.self,
                node: "skin.decode"
            )
            let condition = try inputs.required(
                "condition",
                as: SmeltSkinConditionEncoding.self,
                node: "skin.decode"
            )
            let pointNormals = try inputs.required(
                "queries",
                as: [Float].self,
                node: "skin.decode"
            )
            let indices = generation.skinCodeIndices.map { $0.map(UInt32.init) }
            let weights = try decoder(context).decodeJointField(
                indicesByJoint: indices,
                conditionTokens: condition.conditionTokens,
                pointNormals: pointNormals
            )
            record(
                weights.jointMajorWeights,
                elementCount: weights.vertexCount * weights.jointCount,
                as: "skin.decoded-field",
                context: context
            )
            recordJointColumns(weights, context: context)
            return ["skin": .init(weights)]
        }
    }

    struct SkinTransfer: SmeltGraphNodeRuntime {
        func run(
            inputs: [String: SmeltGraphValue],
            context: SmeltGraphExecutionContext
        ) throws -> [String: SmeltGraphValue] {
            let plan = try inputs.required(
                "plan",
                as: SmeltSkinTransferPlan.self,
                node: "skin.transfer"
            )
            let weights = try inputs.required(
                "skin",
                as: SmeltGPUSkinField.self,
                node: "skin.transfer"
            )
            let skin = try transfer(context).transfer(plan: plan, sampledField: weights)
            record(skin.jointIndices, as: "skin.vertex-joints", context: context)
            record(skin.weights, as: "skin.vertex-weights", context: context)
            return ["vertex-skin": .init(skin)]
        }
    }

    struct GLBEncode: SmeltGraphNodeRuntime {
        func run(
            inputs: [String: SmeltGraphValue],
            context: SmeltGraphExecutionContext
        ) throws -> [String: SmeltGraphValue] {
            let mesh = try inputs.required("mesh", as: SmeltMesh.self, node: "glb.encode")
            let sampled = try inputs.required(
                "sampled",
                as: SmeltSampledSurface.self,
                node: "glb.encode"
            )
            let generation = try inputs.required(
                "generation",
                as: SmeltSkeletonGenerationResult.self,
                node: "glb.encode"
            )
            let skin = try inputs.required(
                "vertex-skin",
                as: SmeltVertexSkin.self,
                node: "glb.encode"
            )
            try SmeltGLB.writeSkin(
                mesh: mesh,
                skeleton: generation.skeleton,
                normalization: sampled.normalization,
                skin: skin,
                to: context.outputURL
            )
            context.setSummary(
                "Wrote \(skin.vertexCount) vertices / \(skin.jointCount) joints: "
                    + context.outputURL.path
            )
            return ["output": .init(context.outputURL)]
        }
    }

    private static func device(_ context: SmeltGraphExecutionContext) throws -> MTLDevice {
        try context.resource("metal.device") {
            guard let device = MTLCreateSystemDefaultDevice() else {
                throw SmeltSkinningGraphNodeError.metalUnavailable
            }
            return device
        }
    }

    private static func artifact(
        _ context: SmeltGraphExecutionContext
    ) throws -> SmeltComponentArtifact {
        try context.resource("component.artifact") {
            try SmeltComponentArtifact(path: context.packagePath, verify: true)
        }
    }

    private static func meshEncoder(
        _ context: SmeltGraphExecutionContext
    ) throws -> SmeltMeshEncoder {
        try context.resource("mesh.encoder") {
            try SmeltMeshEncoder(artifact: artifact(context), device: device(context))
        }
    }

    private static func conditionEncoder(
        _ context: SmeltGraphExecutionContext
    ) throws -> SmeltSkinConditionEncoder {
        try context.resource("condition.encoder") {
            try SmeltSkinConditionEncoder(artifact: artifact(context), device: device(context))
        }
    }

    private static func generator(
        _ context: SmeltGraphExecutionContext
    ) throws -> SmeltSkeletonGenerator {
        try context.resource("sequence.generator") {
            SmeltSkeletonGenerator(
                languageModel: try SmeltSkeletonLanguageRuntime(
                    packagePath: context.packagePath,
                    device: device(context),
                    verifyPackage: true
                )
            )
        }
    }

    private static func decoder(
        _ context: SmeltGraphExecutionContext
    ) throws -> SmeltSkinDecoder {
        try context.resource("skin.decoder") {
            try SmeltSkinDecoder(artifact: artifact(context), device: device(context))
        }
    }

    private static func transfer(
        _ context: SmeltGraphExecutionContext
    ) throws -> SmeltSkinTransferRuntime {
        try context.resource("skin.transfer") {
            try SmeltSkinTransferRuntime(artifact: artifact(context), device: device(context))
        }
    }

    private static func skeletonTokens(_ value: String?) throws -> [Int] {
        guard let value else {
            return [SmeltSkeletonVocabulary.skeletonBOS, SmeltSkeletonVocabulary.noClass]
        }
        let fields = value.split(separator: ",", omittingEmptySubsequences: false)
        let tokens = fields.compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard tokens.count == fields.count else {
            throw SmeltPackageRunnerError.invalidOption("skeleton-tokens", value)
        }
        return tokens
    }

    private static func unsignedOption(
        _ flag: String,
        context: SmeltGraphExecutionContext
    ) throws -> UInt64 {
        guard let value = context.options[flag] else {
            throw SmeltPackageRunnerError.missingOptionDefault(flag)
        }
        return try unsignedValue(value, flag: flag)
    }

    private static func positiveOption(
        _ flag: String,
        context: SmeltGraphExecutionContext
    ) throws -> Int {
        guard let value = context.options[flag], let parsed = Int(value), parsed > 0 else {
            throw SmeltPackageRunnerError.missingOptionDefault(flag)
        }
        return parsed
    }

    private static func unsignedValue(_ value: String, flag: String) throws -> UInt64 {
        guard let parsed = UInt64(value) else {
            throw SmeltPackageRunnerError.invalidOption(flag, value)
        }
        return parsed
    }

    private static func record<T>(
        _ values: [T],
        as key: String,
        context: SmeltGraphExecutionContext
    ) {
        guard context.capturesEvidence else { return }
        values.withUnsafeBytes { bytes in
            context.recordEvidence(key, value: sha256(bytes))
        }
    }

    private static func record(
        _ buffer: MTLBuffer,
        elementCount: Int,
        as key: String,
        context: SmeltGraphExecutionContext
    ) {
        guard context.capturesEvidence else { return }
        let bytes = UnsafeRawBufferPointer(
            start: buffer.contents(),
            count: elementCount * MemoryLayout<Float>.stride
        )
        context.recordEvidence(key, value: sha256(bytes))
    }

    private static func recordJointColumns(
        _ field: SmeltGPUSkinField,
        context: SmeltGraphExecutionContext
    ) {
        guard context.capturesEvidence else { return }
        let byteCount = field.vertexCount * MemoryLayout<Float>.stride
        for joint in 0..<field.jointCount {
            let bytes = UnsafeRawBufferPointer(
                start: field.jointMajorWeights.contents().advanced(by: joint * byteCount),
                count: byteCount
            )
            context.recordEvidence(
                "skin.decoded-field.joint.\(joint)",
                value: sha256(bytes)
            )
        }
    }

    private static func sha256(_ bytes: UnsafeRawBufferPointer) -> String {
        SHA256.hash(data: Data(bytes)).map { String(format: "%02x", $0) }.joined()
    }
}

private enum SmeltSkinningGraphNodeError: Error {
    case metalUnavailable
}
