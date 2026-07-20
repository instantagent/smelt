import Foundation
import Testing

@testable import SmeltRuntime

@Suite("Smelt auto-rig runtime")
struct SmeltAutoRigRuntimeTests {
    @Test("Production neural path emits one complete joint-weight column")
    func productionNeuralPath() throws {
        guard ProcessInfo.processInfo.environment["SMELT_RIG_FULL_SMOKE"] == "1",
              let package = ProcessInfo.processInfo.environment[
                  "SMELT_RIG_PACKAGE"
              ]
        else {
            return
        }
        var points = [Float](repeating: 0, count: 54_000 * 6)
        for index in 0..<54_000 {
            points[index * 6] = Float((index * 37) % 4_096 - 2_048) / 2_048
            points[index * 6 + 1] = Float((index * 109) % 8_192 - 4_096) / 4_096
            points[index * 6 + 2] = Float((index * 251) % 16_384 - 8_192) / 8_192
            points[index * 6 + 3] = Float((index * 13) % 256 - 128) / 128
            points[index * 6 + 4] = Float((index * 29) % 256 - 128) / 128
            points[index * 6 + 5] = Float((index * 61) % 256 - 128) / 128
        }
        let runtime = try SmeltAutoRigRuntime(packagePath: package)
        let result = try runtime.rig(
            pointNormals: points,
            startTokenIDs: [257, 263, 128, 128, 128, 258],
            generationConfiguration: .init(
                policyMode: .validated,
                decodeMode: .greedy,
                maximumGeneratedTokens: 5
            )
        )
        #expect(result.meshQuerySourceIndices.count == 512)
        #expect(result.conditionQuerySourceIndices.count == 384)
        #expect(result.generation.skeleton.joints.count == 1)
        #expect(result.skinWeights.vertexCount == 54_000)
        #expect(result.skinWeights.jointCount == 1)
        #expect(result.skinWeights.values.count == 54_000)
        #expect(
            result.skinWeights.values.allSatisfy {
                $0.isFinite && $0 >= 0 && $0 <= 1
            }
        )
    }

    @Test("Pinned giraffe rigs to a native skinned GLB")
    func productionGLBPath() throws {
        guard ProcessInfo.processInfo.environment[
            "SMELT_RIG_FULL_GLB_SMOKE"
        ] == "1",
            let package = ProcessInfo.processInfo.environment[
                "SMELT_RIG_PACKAGE"
            ],
            let fixture = ProcessInfo.processInfo.environment[
                "SMELT_RIG_GLB_FIXTURE"
            ]
        else {
            return
        }
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tokenrig-full-glb-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = directory.appendingPathComponent("giraffe-rigged.glb")
        let runtime = try SmeltAutoRigRuntime(packagePath: package)
        let result = try runtime.rigGLB(
            inputURL: URL(fileURLWithPath: fixture),
            outputURL: output,
            samplingSeed: 0,
            startTokenIDs: [257, 263, 128, 128, 128, 258],
            generationConfiguration: .init(
                policyMode: .validated,
                decodeMode: .greedy,
                maximumGeneratedTokens: 5
            )
        )
        #expect(FileManager.default.fileExists(atPath: output.path))
        #expect(result.sampling.pointNormals.count == 54_000 * 6)
        #expect(result.neural.generation.skeleton.joints.count == 1)
        #expect(result.vertexSkin.jointCount == 1)
        let imported = try SmeltGLB.readMesh(from: output)
        #expect(imported.positions.count == result.vertexSkin.vertexCount)
        #expect(imported.triangleIndices.count > 3_000)
        let summary = try SmeltGLB.validateRig(from: output)
        #expect(summary.vertexCount == result.vertexSkin.vertexCount)
        #expect(summary.jointCount == result.vertexSkin.jointCount)
        #expect(summary.weightedVertexCount == result.vertexSkin.vertexCount)
    }
}
