import Testing

@testable import SmeltRuntime

@Suite("Smelt mesh geometry")
struct SmeltMeshGeometryTests {
    private func mesh() throws -> SmeltMesh {
        try SmeltMesh(
            positions: [
                SIMD3(-2, 0, 0),
                SIMD3(2, 0, 0),
                SIMD3(0, 4, 0),
                SIMD3(0, 0, 1),
            ],
            normals: Array(repeating: SIMD3(0, 0, 1), count: 4),
            triangleIndices: [0, 1, 2, 0, 3, 1]
        )
    }

    @Test("Sampling is deterministic and receipt replay is bit-identical")
    func sampleReplay() throws {
        let mesh = try mesh()
        let first = try SmeltMeshGeometry.sample(mesh: mesh, seed: 7)
        let second = try SmeltMeshGeometry.sample(mesh: mesh, seed: 7)
        #expect(first == second)
        #expect(first.pointNormals.count == 54_000 * 6)
        #expect(first.receipt.sourceVertexIndices.count == 4)
        #expect(first.receipt.sourceTriangleIndices.count == 53_996)
        let replayed = try SmeltMeshGeometry.replay(
            mesh: mesh,
            normalization: first.normalization,
            receipt: first.receipt
        )
        #expect(replayed.count == first.pointNormals.count)
        for index in replayed.indices {
            #expect(replayed[index].bitPattern == first.pointNormals[index].bitPattern)
        }
        #expect(first.normalization.normalize(SIMD3(-2, 0, 0)).x == -1)
        #expect(first.normalization.normalize(SIMD3(2, 4, 0)).x == 1)
    }

    @Test("Eight-neighbor transfer normalizes and prunes to four lanes")
    func transfer() throws {
        let mesh = try mesh()
        let sampled = try SmeltMeshGeometry.sample(mesh: mesh, seed: 3)
        let jointCount = 6
        var values = [Float](repeating: 0, count: 54_000 * jointCount)
        for vertex in 0..<54_000 {
            for joint in 0..<jointCount {
                values[vertex * jointCount + joint] = Float(joint + 1)
            }
        }
        let skin = try SmeltMeshGeometry.transferSkin(
            mesh: mesh,
            sampled: sampled,
            sampledWeights: .init(
                vertexCount: 54_000,
                jointCount: jointCount,
                values: values
            )
        )
        #expect(skin.vertexCount == 4)
        #expect(skin.jointIndices.count == 16)
        for vertex in 0..<4 {
            let row = skin.weights[(vertex * 4)..<(vertex * 4 + 4)]
            #expect(abs(row.reduce(0, +) - 1) < 1e-6)
            #expect(skin.jointIndices[vertex * 4] == 5)
            #expect(skin.jointIndices[vertex * 4 + 3] == 2)
        }
    }
}
