import Foundation

/// Flattened triangle mesh in world space for skinning component preprocessing.
public struct SmeltMesh: Sendable, Equatable {
    public let positions: [SIMD3<Float>]
    public let normals: [SIMD3<Float>]
    public let triangleIndices: [UInt32]

    public init(
        positions: [SIMD3<Float>],
        normals: [SIMD3<Float>],
        triangleIndices: [UInt32]
    ) throws {
        guard !positions.isEmpty,
              normals.count == positions.count,
              !triangleIndices.isEmpty,
              triangleIndices.count.isMultiple(of: 3),
              triangleIndices.allSatisfy({ Int($0) < positions.count }),
              positions.allSatisfy(Self.finite),
              normals.allSatisfy(Self.finite)
        else {
            throw SmeltMeshGeometryError.invalidMesh
        }
        self.positions = positions
        self.normals = normals
        self.triangleIndices = triangleIndices
    }

    private static func finite(_ value: SIMD3<Float>) -> Bool {
        value.x.isFinite && value.y.isFinite && value.z.isFinite
    }
}

/// Reversible uniform affine that maps the longest asset dimension to `[-1,1]`.
public struct SmeltMeshNormalization: Sendable, Equatable {
    public let center: SIMD3<Float>
    public let scale: Float

    public init(center: SIMD3<Float>, scale: Float) {
        self.center = center
        self.scale = scale
    }

    public func normalize(_ point: SIMD3<Float>) -> SIMD3<Float> {
        (point - center) / scale
    }

    public func denormalize(_ point: SIMD3<Float>) -> SIMD3<Float> {
        point * scale + center
    }
}

/// Full deterministic sampling receipt. These arrays, rather than a seed alone,
/// are the orchestration-exact boundary because upstream sampling is unseeded.
public struct SmeltSurfaceSamplingReceipt: Sendable, Equatable {
    public let sourceVertexIndices: [Int]
    public let sourceTriangleIndices: [Int]
    public let barycentricUV: [SIMD2<Float>]

    public init(
        sourceVertexIndices: [Int],
        sourceTriangleIndices: [Int],
        barycentricUV: [SIMD2<Float>]
    ) {
        self.sourceVertexIndices = sourceVertexIndices
        self.sourceTriangleIndices = sourceTriangleIndices
        self.barycentricUV = barycentricUV
    }
}

/// the skinning component's normalized sampled point cloud and its replay receipt.
public struct SmeltSampledSurface: Sendable, Equatable {
    public let normalization: SmeltMeshNormalization
    public let pointNormals: [Float]
    public let receipt: SmeltSurfaceSamplingReceipt

    public init(
        normalization: SmeltMeshNormalization,
        pointNormals: [Float],
        receipt: SmeltSurfaceSamplingReceipt
    ) {
        self.normalization = normalization
        self.pointNormals = pointNormals
        self.receipt = receipt
    }
}

/// Four-influence GLB-ready skin data in vertex-major order.
public struct SmeltVertexSkin: Sendable, Equatable {
    public let vertexCount: Int
    public let jointCount: Int
    public let jointIndices: [UInt16]
    public let weights: [Float]

    public init(
        vertexCount: Int,
        jointCount: Int,
        jointIndices: [UInt16],
        weights: [Float]
    ) {
        self.vertexCount = vertexCount
        self.jointCount = jointCount
        self.jointIndices = jointIndices
        self.weights = weights
    }
}

/// Exact eight-neighbor transfer program plus the minimal sampled query union.
public struct SmeltSkinTransferPlan: Sendable, Equatable {
  public let vertexCount: Int
  public let sampledPointCount: Int
  public let querySourceIndices: [Int]
  public let queryPointNormals: [Float]
  public let neighborOffsets: [Int]
  public let neighborQueryRows: [Int]
  public let neighborBlends: [Float]

  public init(
    vertexCount: Int,
    sampledPointCount: Int,
    querySourceIndices: [Int],
    queryPointNormals: [Float],
    neighborOffsets: [Int],
    neighborQueryRows: [Int],
    neighborBlends: [Float]
  ) {
    self.vertexCount = vertexCount
    self.sampledPointCount = sampledPointCount
    self.querySourceIndices = querySourceIndices
    self.queryPointNormals = queryPointNormals
    self.neighborOffsets = neighborOffsets
    self.neighborQueryRows = neighborQueryRows
    self.neighborBlends = neighborBlends
  }
}

/// Native geometry policy used before and after the neural skinning component path.
public enum SmeltMeshGeometry {
    public static let sampledPointCount = 54_000
    public static let maximumAuthoredVertexSamples = 16_384

    /// Normalizes and samples the exact production row count. A seed makes the
    /// Smelt route reproducible; the returned receipt is the stronger evidence.
    public static func sample(
        mesh: SmeltMesh,
        seed: UInt64 = 0
    ) throws -> SmeltSampledSurface {
        let normalization = try makeNormalization(mesh.positions)
        let positions = mesh.positions.map(normalization.normalize)
        let normals = mesh.normals.map(normalized)
        let vertexSampleCount = min(
            maximumAuthoredVertexSamples,
            positions.count,
            sampledPointCount
        )
        var rng = SmeltDeterministicRng(seed: seed)
        var permutation = Array(positions.indices)
        if permutation.count > 1 {
            for index in stride(from: permutation.count - 1, through: 1, by: -1) {
                let selected = Int(rng.next() % UInt64(index + 1))
                permutation.swapAt(index, selected)
            }
        }
        let vertexIndices = Array(permutation.prefix(vertexSampleCount))
        let surfaceCount = sampledPointCount - vertexSampleCount
        let triangleCount = mesh.triangleIndices.count / 3
        var cumulativeArea: [Double] = []
        cumulativeArea.reserveCapacity(triangleCount)
        var totalArea = 0.0
        var faceNormals: [SIMD3<Float>] = []
        faceNormals.reserveCapacity(triangleCount)
        for triangle in 0..<triangleCount {
            let base = triangle * 3
            let a = positions[Int(mesh.triangleIndices[base])]
            let b = positions[Int(mesh.triangleIndices[base + 1])]
            let c = positions[Int(mesh.triangleIndices[base + 2])]
            let cross = crossProduct(b - a, c - a)
            let length = sqrt(
                Double(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z)
            )
            totalArea += length
            cumulativeArea.append(totalArea)
            faceNormals.append(normalized(cross))
        }
        guard totalArea.isFinite, totalArea > 0 else {
            throw SmeltMeshGeometryError.zeroSurfaceArea
        }

        var triangleIndices: [Int] = []
        var barycentric: [SIMD2<Float>] = []
        triangleIndices.reserveCapacity(surfaceCount)
        barycentric.reserveCapacity(surfaceCount)
        for _ in 0..<surfaceCount {
            let target = uniform(&rng) * totalArea
            triangleIndices.append(lowerBound(cumulativeArea, target))
        }
        for _ in 0..<surfaceCount {
            var u = Float(uniform(&rng))
            var v = Float(uniform(&rng))
            if u + v > 1 {
                u = 1 - u
                v = 1 - v
            }
            barycentric.append(SIMD2(u, v))
        }

        var pointNormals: [Float] = []
        pointNormals.reserveCapacity(sampledPointCount * 6)
        for index in vertexIndices {
            append(positions[index], normals[index], to: &pointNormals)
        }
        for sample in 0..<surfaceCount {
            let triangle = triangleIndices[sample]
            let base = triangle * 3
            let a = positions[Int(mesh.triangleIndices[base])]
            let b = positions[Int(mesh.triangleIndices[base + 1])]
            let c = positions[Int(mesh.triangleIndices[base + 2])]
            let uv = barycentric[sample]
            let point = a + (b - a) * uv.x + (c - a) * uv.y
            append(point, faceNormals[triangle], to: &pointNormals)
        }
        return SmeltSampledSurface(
            normalization: normalization,
            pointNormals: pointNormals,
            receipt: SmeltSurfaceSamplingReceipt(
                sourceVertexIndices: vertexIndices,
                sourceTriangleIndices: triangleIndices,
                barycentricUV: barycentric
            )
        )
    }

    /// Replays a captured sampling receipt without consuming random state.
    public static func replay(
        mesh: SmeltMesh,
        normalization: SmeltMeshNormalization,
        receipt: SmeltSurfaceSamplingReceipt
    ) throws -> [Float] {
        let positions = mesh.positions.map(normalization.normalize)
        let normals = mesh.normals.map(normalized)
    guard
      receipt.sourceVertexIndices.count
                + receipt.sourceTriangleIndices.count == sampledPointCount,
              receipt.sourceTriangleIndices.count == receipt.barycentricUV.count,
              receipt.sourceVertexIndices.allSatisfy(positions.indices.contains),
              receipt.sourceTriangleIndices.allSatisfy({
                  $0 >= 0 && $0 < mesh.triangleIndices.count / 3
              })
        else {
            throw SmeltMeshGeometryError.invalidSamplingReceipt
        }
        var pointNormals: [Float] = []
        pointNormals.reserveCapacity(sampledPointCount * 6)
        for index in receipt.sourceVertexIndices {
            append(positions[index], normals[index], to: &pointNormals)
        }
        for sample in receipt.sourceTriangleIndices.indices {
            let triangle = receipt.sourceTriangleIndices[sample]
            let base = triangle * 3
            let a = positions[Int(mesh.triangleIndices[base])]
            let b = positions[Int(mesh.triangleIndices[base + 1])]
            let c = positions[Int(mesh.triangleIndices[base + 2])]
            let uv = receipt.barycentricUV[sample]
            guard uv.x >= 0, uv.y >= 0, uv.x + uv.y <= 1 else {
                throw SmeltMeshGeometryError.invalidSamplingReceipt
            }
            let point = a + (b - a) * uv.x + (c - a) * uv.y
            let normal = normalized(crossProduct(b - a, c - a))
            append(point, normal, to: &pointNormals)
        }
        return pointNormals
    }

    /// Transfers sampled weights to authored vertices with upstream's eight
    /// inverse-distance neighbors, then normalizes and prunes to four GLB lanes.
    public static func transferSkin(
        mesh: SmeltMesh,
        sampled: SmeltSampledSurface,
        sampledWeights: SmeltSkinWeights
    ) throws -> SmeltVertexSkin {
    let plan = try prepareSkinTransfer(mesh: mesh, sampled: sampled)
    guard sampledWeights.vertexCount == plan.sampledPointCount else {
      throw SmeltMeshGeometryError.invalidSampledWeights
    }
    var selected = [Float](
      repeating: 0,
      count: plan.querySourceIndices.count * sampledWeights.jointCount
    )
    for (queryRow, sourceRow) in plan.querySourceIndices.enumerated() {
      let source = sourceRow * sampledWeights.jointCount
      let destination = queryRow * sampledWeights.jointCount
      selected.replaceSubrange(
        destination..<(destination + sampledWeights.jointCount),
        with: sampledWeights.values[source..<(source + sampledWeights.jointCount)]
      )
    }
    return try transferSkin(
      plan: plan,
      sampledWeights: .init(
        vertexCount: plan.querySourceIndices.count,
        jointCount: sampledWeights.jointCount,
        values: selected
      )
    )
  }

  public static func prepareSkinTransfer(
    mesh: SmeltMesh,
    sampled: SmeltSampledSurface
  ) throws -> SmeltSkinTransferPlan {
    guard sampled.pointNormals.count.isMultiple(of: 6) else {
            throw SmeltMeshGeometryError.invalidSampledWeights
        }
    let sampledPointCount = sampled.pointNormals.count / 6
    let sampledPositions = stride(from: 0, to: sampled.pointNormals.count, by: 6).map {
            SIMD3(
                sampled.pointNormals[$0],
                sampled.pointNormals[$0 + 1],
                sampled.pointNormals[$0 + 2]
            )
        }
        let tree = KDTree(points: sampledPositions)
    var sourceNeighbors: [[Int]] = []
    var blends: [[Float]] = []
    sourceNeighbors.reserveCapacity(mesh.positions.count)
    blends.reserveCapacity(mesh.positions.count)
    var sourceUnion = Set<Int>()
        for vertex in mesh.positions.indices {
            let query = sampled.normalization.normalize(mesh.positions[vertex])
            let neighbors = tree.nearest(query, count: min(8, sampledPositions.count))
            var inverseDistances: [Double] = []
            inverseDistances.reserveCapacity(neighbors.count)
            var totalInverse = 0.0
            for neighbor in neighbors {
                let inverse = 1.0 / (sqrt(neighbor.distanceSquared) + 1e-8)
                inverseDistances.append(inverse)
                totalInverse += inverse
        sourceUnion.insert(neighbor.index)
      }
      sourceNeighbors.append(neighbors.map(\.index))
      blends.append(inverseDistances.map { Float($0 / totalInverse) })
    }
    let querySourceIndices = sourceUnion.sorted()
    let queryRowBySource = Dictionary(
      uniqueKeysWithValues: querySourceIndices.enumerated().map { ($0.element, $0.offset) }
    )
    var queryPointNormals: [Float] = []
    queryPointNormals.reserveCapacity(querySourceIndices.count * 6)
    for source in querySourceIndices {
      queryPointNormals.append(contentsOf: sampled.pointNormals[(source * 6)..<(source * 6 + 6)])
    }
    var neighborOffsets = [0]
    var neighborQueryRows: [Int] = []
    var neighborBlends: [Float] = []
    for vertex in sourceNeighbors.indices {
      for (source, blend) in zip(sourceNeighbors[vertex], blends[vertex]) {
        guard let row = queryRowBySource[source] else {
          throw SmeltMeshGeometryError.invalidSampledWeights
        }
        neighborQueryRows.append(row)
        neighborBlends.append(blend)
            }
      neighborOffsets.append(neighborQueryRows.count)
    }
    return SmeltSkinTransferPlan(
      vertexCount: mesh.positions.count,
      sampledPointCount: sampledPointCount,
      querySourceIndices: querySourceIndices,
      queryPointNormals: queryPointNormals,
      neighborOffsets: neighborOffsets,
      neighborQueryRows: neighborQueryRows,
      neighborBlends: neighborBlends
    )
  }

  public static func transferSkin(
    plan: SmeltSkinTransferPlan,
    sampledWeights: SmeltSkinWeights
  ) throws -> SmeltVertexSkin {
    guard sampledWeights.vertexCount == plan.querySourceIndices.count,
      sampledWeights.jointCount > 0,
      sampledWeights.jointCount <= Int(UInt16.max),
      sampledWeights.values.count
        == sampledWeights.vertexCount * sampledWeights.jointCount
    else {
      throw SmeltMeshGeometryError.invalidSampledWeights
    }
    var joints = [UInt16](repeating: 0, count: plan.vertexCount * 4)
    var weights = [Float](repeating: 0, count: plan.vertexCount * 4)
    var accumulated = [Float](repeating: 0, count: sampledWeights.jointCount)
    for vertex in 0..<plan.vertexCount {
      for joint in accumulated.indices { accumulated[joint] = 0 }
      let start = plan.neighborOffsets[vertex]
      let end = plan.neighborOffsets[vertex + 1]
      for neighbor in start..<end {
        let blend = plan.neighborBlends[neighbor]
        let source = plan.neighborQueryRows[neighbor] * sampledWeights.jointCount
                for joint in 0..<sampledWeights.jointCount {
                    accumulated[joint] += sampledWeights.values[source + joint] * blend
                }
            }
            let ranked = accumulated.indices.sorted {
                accumulated[$0] == accumulated[$1]
                    ? $0 < $1
                    : accumulated[$0] > accumulated[$1]
            }.prefix(4)
            let sum = ranked.reduce(Float(0)) { $0 + max(accumulated[$1], 0) }
            if sum > 1e-8 {
                for (lane, joint) in ranked.enumerated() {
                    joints[vertex * 4 + lane] = UInt16(joint)
                    weights[vertex * 4 + lane] = max(accumulated[joint], 0) / sum
                }
            } else {
                weights[vertex * 4] = 1
            }
        }
        return SmeltVertexSkin(
      vertexCount: plan.vertexCount,
            jointCount: sampledWeights.jointCount,
            jointIndices: joints,
            weights: weights
        )
    }

    private static func makeNormalization(
        _ positions: [SIMD3<Float>]
    ) throws -> SmeltMeshNormalization {
        var minimum = positions[0]
        var maximum = positions[0]
        for point in positions.dropFirst() {
            minimum = SIMD3(
                min(minimum.x, point.x),
                min(minimum.y, point.y),
                min(minimum.z, point.z)
            )
            maximum = SIMD3(
                max(maximum.x, point.x),
                max(maximum.y, point.y),
                max(maximum.z, point.z)
            )
        }
        let extent = maximum - minimum
        let scale = max(extent.x, extent.y, extent.z) / 2
        guard scale.isFinite, scale > 0 else {
            throw SmeltMeshGeometryError.zeroExtent
        }
        return SmeltMeshNormalization(
            center: (minimum + maximum) / 2,
            scale: scale
        )
    }

    private static func append(
        _ point: SIMD3<Float>,
        _ normal: SIMD3<Float>,
        to values: inout [Float]
    ) {
        values.append(contentsOf: [
            point.x, point.y, point.z,
            normal.x, normal.y, normal.z,
        ])
    }

    private static func uniform(_ rng: inout SmeltDeterministicRng) -> Double {
        Double(rng.next() >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }

    private static func lowerBound(_ values: [Double], _ target: Double) -> Int {
        var lower = 0
        var upper = values.count
        while lower < upper {
            let middle = (lower + upper) / 2
      if values[middle] < target { lower = middle + 1 } else { upper = middle }
        }
        return min(lower, values.count - 1)
    }

    private static func crossProduct(
        _ left: SIMD3<Float>,
        _ right: SIMD3<Float>
    ) -> SIMD3<Float> {
        SIMD3(
            left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x
        )
    }

    private static func normalized(_ value: SIMD3<Float>) -> SIMD3<Float> {
        let length = sqrt(value.x * value.x + value.y * value.y + value.z * value.z)
        return length > 1e-10 && length.isFinite ? value / length : .zero
    }
}

private final class KDTree {
    struct Neighbor {
        let index: Int
        let distanceSquared: Double
    }

    private final class Node {
        let index: Int
        let axis: Int
        let left: Node?
        let right: Node?

        init(index: Int, axis: Int, left: Node?, right: Node?) {
            self.index = index
            self.axis = axis
            self.left = left
            self.right = right
        }
    }

    private let points: [SIMD3<Float>]
    private let root: Node?

    init(points: [SIMD3<Float>]) {
        self.points = points
        root = Self.build(points: points, indices: Array(points.indices), depth: 0)
    }

    func nearest(_ point: SIMD3<Float>, count: Int) -> [Neighbor] {
        var best: [Neighbor] = []
        search(root, point: point, count: count, best: &best)
        return best
    }

    private static func build(
        points: [SIMD3<Float>],
        indices: [Int],
        depth: Int
    ) -> Node? {
        guard !indices.isEmpty else { return nil }
        let axis = depth % 3
        let sorted = indices.sorted {
            component(points[$0], axis) == component(points[$1], axis)
                ? $0 < $1
                : component(points[$0], axis) < component(points[$1], axis)
        }
        let middle = sorted.count / 2
        return Node(
            index: sorted[middle],
            axis: axis,
            left: build(
                points: points,
                indices: Array(sorted[..<middle]),
                depth: depth + 1
            ),
            right: build(
                points: points,
                indices: Array(sorted[(middle + 1)...]),
                depth: depth + 1
            )
        )
    }

    private func search(
        _ node: Node?,
        point: SIMD3<Float>,
        count: Int,
        best: inout [Neighbor]
    ) {
        guard let node else { return }
        let delta = points[node.index] - point
        let distance = Double(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z)
        best.append(Neighbor(index: node.index, distanceSquared: distance))
        best.sort {
            $0.distanceSquared == $1.distanceSquared
                ? $0.index < $1.index
                : $0.distanceSquared < $1.distanceSquared
        }
        if best.count > count { best.removeLast() }
        let difference = Double(
            Self.component(point, node.axis) - Self.component(points[node.index], node.axis)
        )
        let near = difference < 0 ? node.left : node.right
        let far = difference < 0 ? node.right : node.left
        search(near, point: point, count: count, best: &best)
    let worst =
      best.count < count
            ? Double.greatestFiniteMagnitude
            : best[best.count - 1].distanceSquared
        if difference * difference <= worst {
            search(far, point: point, count: count, best: &best)
        }
    }

    private static func component(_ value: SIMD3<Float>, _ axis: Int) -> Float {
        switch axis {
        case 0: return value.x
        case 1: return value.y
        default: return value.z
        }
    }
}

public enum SmeltMeshGeometryError: Error, Equatable {
    case invalidMesh
    case zeroExtent
    case zeroSurfaceArea
    case invalidSamplingReceipt
    case invalidSampledWeights
}
