import Foundation
import Testing

@testable import SmeltRuntime

@Suite("Smelt GLB")
struct SmeltGLBTests {
    @Test("Generated skinned GLB roundtrips its world-space mesh")
    func roundTrip() throws {
        let mesh = try SmeltMesh(
            positions: [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(0, 1, 0)],
            normals: Array(repeating: SIMD3(0, 0, 1), count: 3),
            triangleIndices: [0, 1, 2]
        )
        let skeleton = SmeltSkeleton(joints: [
            .init(position: SIMD3(0, 0, 0), parent: -1),
            .init(position: SIMD3(0, 1, 0), parent: 0),
        ])
        let skin = SmeltVertexSkin(
            vertexCount: 3,
            jointCount: 2,
            jointIndices: [
                0, 1, 0, 0,
                0, 1, 0, 0,
                1, 0, 0, 0,
            ],
            weights: [
                1, 0, 0, 0,
                0.5, 0.5, 0, 0,
                1, 0, 0, 0,
            ]
        )
        let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("skinning-glb-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }
    let path = directory.appendingPathComponent("skin.glb")
    try SmeltGLB.writeSkin(
            mesh: mesh,
            skeleton: skeleton,
            normalization: .init(center: .zero, scale: 1),
            skin: skin,
            to: path
        )
        let imported = try SmeltGLB.readMesh(from: path)
        #expect(imported.positions == mesh.positions)
        #expect(imported.normals == mesh.normals)
        #expect(imported.triangleIndices == mesh.triangleIndices)

        let data = try Data(contentsOf: path)
    let jsonLength = Int(
      data.withUnsafeBytes {
            UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 12, as: UInt32.self))
        })
    let document =
      try JSONSerialization.jsonObject(
            with: data.subdata(in: 20..<(20 + jsonLength))
        ) as? [String: Any]
        #expect((document?["skins"] as? [[String: Any]])?.count == 1)
        #expect((document?["nodes"] as? [[String: Any]])?.count == 3)
    let summary = try SmeltGLB.validateSkin(from: path)
        #expect(summary.vertexCount == 3)
        #expect(summary.jointCount == 2)
        #expect(summary.weightedVertexCount == 3)
        #expect(summary.inverseBindMatrixCount == 2)
        #expect(summary.maximumWeightSumError == 0)
    }

    @Test("Pinned upstream giraffe imports without Blender")
    func upstreamGiraffe() throws {
    guard
      let path = ProcessInfo.processInfo.environment[
        "SMELT_SKINNING_GLB_FIXTURE"
      ]
    else {
            return
        }
        let mesh = try SmeltGLB.readMesh(
            from: URL(fileURLWithPath: path)
        )
        #expect(mesh.positions.count > 1_000)
        #expect(mesh.triangleIndices.count > 3_000)
        #expect(mesh.normals.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite })
    }

  @Test("Persisted fixture contains a complete finite normalized skin")
  func persistedSkinFixture() throws {
    guard
      let path = ProcessInfo.processInfo.environment[
        "SMELT_SKINNING_GLB_VALIDATE_PATH"
      ]
    else {
            #expect(
        ProcessInfo.processInfo.environment["SMELT_SKINNING_REQUIRE_FIXTURES"]
                    != "1"
            )
            return
        }
    let summary = try SmeltGLB.validateSkin(from: URL(fileURLWithPath: path))
        #expect(summary.vertexCount > 1_000)
        #expect(summary.jointCount > 0)
        #expect(summary.weightedVertexCount == summary.vertexCount)
        #expect(summary.inverseBindMatrixCount == summary.jointCount)
        let data = try JSONEncoder().encode(summary)
        let json = try #require(String(data: data, encoding: .utf8))
    print("SKINNING_U0_GLB_RECEIPT \(json)")
    }
}
