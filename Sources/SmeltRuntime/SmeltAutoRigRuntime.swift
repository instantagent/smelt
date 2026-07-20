import Foundation
import Metal

/// End-to-end neural output for already sampled rig model point/normal rows.
public struct SmeltRiggingResult: Sendable, Equatable {
    public let meshQuerySourceIndices: [Int]
    public let conditionQuerySourceIndices: [Int]
    public let generation: SmeltRigGenerationResult
    public let skinWeights: SmeltSkinWeights

    public init(
        meshQuerySourceIndices: [Int],
        conditionQuerySourceIndices: [Int],
        generation: SmeltRigGenerationResult,
        skinWeights: SmeltSkinWeights
    ) {
        self.meshQuerySourceIndices = meshQuerySourceIndices
        self.conditionQuerySourceIndices = conditionQuerySourceIndices
        self.generation = generation
        self.skinWeights = skinWeights
    }
}

/// End-to-end native asset result, including the exact sampling and transfer
/// artifacts used to write the output GLB.
public struct SmeltGLBRiggingResult: Sendable, Equatable {
    public let sampling: SmeltSampledSurface
    public let neural: SmeltRiggingResult
    public let vertexSkin: SmeltVertexSkin
    public let outputURL: URL

    public init(
        sampling: SmeltSampledSurface,
        neural: SmeltRiggingResult,
        vertexSkin: SmeltVertexSkin,
        outputURL: URL
    ) {
        self.sampling = sampling
        self.neural = neural
        self.vertexSkin = vertexSkin
        self.outputURL = outputURL
    }
}

/// Pure-Smelt rig model neural pipeline from sampled points through rig weights.
/// Native GLB import, surface sampling, transfer, and export are separate I/O
/// boundaries so geometry policy cannot obscure neural first divergence.
public final class SmeltAutoRigRuntime {
    public let artifact: SmeltRigArtifact
    public let meshEncoder: SmeltMeshEncoder
    public let conditionEncoder: SmeltSkinConditionEncoder
    public let generator: SmeltSkeletonGenerator
    public let decoder: SmeltSkinDecoder

    public init(
        packagePath: String,
        device: MTLDevice? = nil,
        verifyPackage: Bool = true
    ) throws {
        guard let resolvedDevice = device ?? MTLCreateSystemDefaultDevice() else {
            throw SmeltAutoRigRuntimeError.metalUnavailable
        }
        let artifact = try SmeltRigArtifact(
            path: packagePath,
            verify: verifyPackage
        )
        self.artifact = artifact
        meshEncoder = try SmeltMeshEncoder(
            artifact: artifact,
            device: resolvedDevice
        )
        conditionEncoder = try SmeltSkinConditionEncoder(
            artifact: artifact,
            device: resolvedDevice
        )
        generator = SmeltSkeletonGenerator(
            languageModel: try SmeltSkeletonLanguageRuntime(
                packagePath: packagePath,
                device: resolvedDevice,
                verifyPackage: verifyPackage
            )
        )
        decoder = try SmeltSkinDecoder(
            artifact: artifact,
            device: resolvedDevice
        )
    }

    /// Runs the complete neural path over exactly 54,000 `[xyz, normal]` rows.
    public func rig(
        pointNormals: [Float],
        startTokenIDs: [Int] = [
            SmeltRigVocabulary.skeletonBOS,
            SmeltRigVocabulary.noClass,
        ],
        generationConfiguration: SmeltRigGenerationConfiguration = .init()
    ) throws -> SmeltRiggingResult {
        guard pointNormals.count == 54_000 * 6,
              pointNormals.allSatisfy(\.isFinite)
        else {
            throw SmeltAutoRigRuntimeError.invalidPointNormalCount(
                expected: 54_000 * 6,
                got: pointNormals.count
            )
        }
        let mesh: SmeltMeshEncoding
        do {
            mesh = try meshEncoder.encode(pointNormals: pointNormals)
        } catch {
            throw SmeltAutoRigRuntimeError.stageFailed("mesh-encoder", "\(error)")
        }
        let condition: SmeltSkinConditionEncoding
        do {
            condition = try conditionEncoder.encode(pointNormals: pointNormals)
        } catch {
            throw SmeltAutoRigRuntimeError.stageFailed("condition-encoder", "\(error)")
        }
        let generation: SmeltRigGenerationResult
        do {
            generation = try generator.generate(
                meshEmbeddings: mesh.embeddings,
                startTokenIDs: startTokenIDs,
                configuration: generationConfiguration
            )
        } catch {
            throw SmeltAutoRigRuntimeError.stageFailed("languageModel-generation", "\(error)")
        }
        let indices = generation.skinCodeIndices.map { joint in
            joint.map(UInt32.init)
        }
        let skinWeights: SmeltSkinWeights
        do {
            skinWeights = try decoder.decodeJoints(
                indicesByJoint: indices,
                conditionTokens: condition.conditionTokens,
                pointNormals: pointNormals
            )
        } catch {
            throw SmeltAutoRigRuntimeError.stageFailed("skin-decoder", "\(error)")
        }
        return SmeltRiggingResult(
            meshQuerySourceIndices: mesh.selectedSourceIndices,
            conditionQuerySourceIndices: condition.selectedSourceIndices,
            generation: generation,
            skinWeights: skinWeights
        )
    }

    /// Imports, samples, rigs, transfers, and exports one GLB without Blender,
    /// PyTorch, Python, CUDA, or a network service.
    public func rigGLB(
        inputURL: URL,
        outputURL: URL,
        samplingSeed: UInt64 = 0,
        startTokenIDs: [Int] = [
            SmeltRigVocabulary.skeletonBOS,
            SmeltRigVocabulary.noClass,
        ],
        generationConfiguration: SmeltRigGenerationConfiguration = .init()
    ) throws -> SmeltGLBRiggingResult {
        let mesh: SmeltMesh
        do {
            mesh = try SmeltGLB.readMesh(from: inputURL)
        } catch {
            throw SmeltAutoRigRuntimeError.stageFailed("glb-import", "\(error)")
        }
        let sampling: SmeltSampledSurface
        do {
            sampling = try SmeltMeshGeometry.sample(mesh: mesh, seed: samplingSeed)
        } catch {
            throw SmeltAutoRigRuntimeError.stageFailed("surface-sampling", "\(error)")
        }
        let neural = try rig(
            pointNormals: sampling.pointNormals,
            startTokenIDs: startTokenIDs,
            generationConfiguration: generationConfiguration
        )
        let vertexSkin: SmeltVertexSkin
        do {
            vertexSkin = try SmeltMeshGeometry.transferSkin(
                mesh: mesh,
                sampled: sampling,
                sampledWeights: neural.skinWeights
            )
        } catch {
            throw SmeltAutoRigRuntimeError.stageFailed("skin-transfer", "\(error)")
        }
        do {
            try SmeltGLB.writeRig(
                mesh: mesh,
                skeleton: neural.generation.skeleton,
                normalization: sampling.normalization,
                skin: vertexSkin,
                to: outputURL
            )
        } catch {
            throw SmeltAutoRigRuntimeError.stageFailed("glb-export", "\(error)")
        }
        return SmeltGLBRiggingResult(
            sampling: sampling,
            neural: neural,
            vertexSkin: vertexSkin,
            outputURL: outputURL
        )
    }
}

public enum SmeltAutoRigRuntimeError: Error, Equatable {
    case metalUnavailable
    case invalidPointNormalCount(expected: Int, got: Int)
    case stageFailed(String, String)
}
