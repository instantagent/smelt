import Foundation
import Testing

@testable import SmeltRuntime

@Suite("Smelt CAM graph execution")
struct SmeltSkinningGraphExecutionTests {
    @Test("GPU skin transfer is bit-exact against the staged brick")
    func skinTransferParity() throws {
        guard let package = ProcessInfo.processInfo.environment[
            "SMELT_SKINNING_PACKAGE"
        ] else {
            return
        }
        let plan = SmeltSkinTransferPlan(
            vertexCount: 3,
            sampledPointCount: 4,
            querySourceIndices: [0, 1, 2, 3],
            queryPointNormals: [],
            neighborOffsets: [0, 2, 5, 9],
            neighborQueryRows: [0, 1, 1, 2, 3, 0, 1, 2, 3],
            neighborBlends: [
                0.31, 0.69,
                0.17, 0.29, 0.54,
                0.11, 0.23, 0.19, 0.47,
            ]
        )
        let sampled = SmeltSkinWeights(
            vertexCount: 4,
            jointCount: 6,
            values: [
                0.13, 0.51, 0.07, 0.89, 0.31, 0.43,
                0.67, 0.29, 0.83, 0.17, 0.59, 0.37,
                0.41, 0.73, 0.19, 0.61, 0.23, 0.97,
                0.79, 0.11, 0.53, 0.47, 0.71, 0.05,
            ]
        )
        let expected = try SmeltMeshGeometry.transferSkin(
            plan: plan,
            sampledWeights: sampled
        )
        let artifact = try SmeltComponentArtifact(path: package, verify: true)
        let actual = try SmeltSkinTransferRuntime(artifact: artifact).transfer(
            plan: plan,
            sampledWeights: sampled
        )
        #expect(actual.jointIndices == expected.jointIndices)
        #expect(actual.weights.count == expected.weights.count)
        for index in expected.weights.indices {
            #expect(
                actual.weights[index].bitPattern == expected.weights[index].bitPattern,
                "skin transfer bit divergence at lane \(index)"
            )
        }
    }

    @Test("Pinned mesh runs through the package-authored graph")
    func productionGLBPath() throws {
        guard ProcessInfo.processInfo.environment[
            "SMELT_SKINNING_FULL_GLB_SMOKE"
        ] == "1",
            let package = ProcessInfo.processInfo.environment[
                "SMELT_SKINNING_PACKAGE"
            ],
            let fixture = ProcessInfo.processInfo.environment[
                "SMELT_SKINNING_GLB_FIXTURE"
            ]
        else {
            return
        }
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-skinning-graph-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = directory.appendingPathComponent("giraffe-skinned.glb")
        let runner = try SmeltPackageRunner(packagePath: package)
        let result = try runner.run(
            inputURL: URL(fileURLWithPath: fixture),
            outputURL: output,
            options: [
                "sampling-seed": "0",
                "skeleton-tokens": "257,263,128,128,128,258",
            ]
        )
        #expect(result.outputURL == output)
        let summary = try SmeltGLB.validateSkin(from: output)
        #expect(summary.vertexCount == 14_807)
        #expect(summary.jointCount == 1)
        #expect(summary.weightedVertexCount == summary.vertexCount)
    }
}
