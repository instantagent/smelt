import Foundation

/// Dependency-free glTF 2.0 binary import/export for supported skinned meshes.
/// and skin subset. Unsupported compressed/sparse/non-triangle data fails loud.
public enum SmeltGLB {
    /// Validates the persisted skeleton, inverse bind matrices, joint indices,
    /// and normalized skin weights in a generated GLB.
  public static func validateSkin(
        from url: URL,
        weightSumTolerance: Float = 1e-5
  ) throws -> SmeltGLBSkinSummary {
        guard weightSumTolerance >= 0, weightSumTolerance.isFinite else {
            throw SmeltGLBError.invalidSkin
        }
        let container = try Container(data: Data(contentsOf: url))
        let document = container.document
        let nodes = try array(document, "nodes")
        let meshes = try array(document, "meshes")
        let skins = try array(document, "skins")
        guard skins.count == 1,
              let jointNodes = skins[0]["joints"] as? [Int],
              !jointNodes.isEmpty,
              Set(jointNodes).count == jointNodes.count,
              jointNodes.allSatisfy({ nodes.indices.contains($0) }),
              let skeletonNode = skins[0]["skeleton"] as? Int,
              jointNodes.contains(skeletonNode),
              let inverseBindAccessor = skins[0]["inverseBindMatrices"] as? Int
        else {
            throw SmeltGLBError.invalidSkin
        }
        let inverseBindMatrices = try container.float16(accessor: inverseBindAccessor)
        guard inverseBindMatrices.count == jointNodes.count,
              inverseBindMatrices.allSatisfy({ $0.allSatisfy(\.isFinite) })
        else {
            throw SmeltGLBError.invalidSkin
        }

        let jointSet = Set(jointNodes)
        var reachedJoints = Set<Int>()
        var visiting = Set<Int>()
        func walkJoint(_ nodeIndex: Int) throws {
            guard jointSet.contains(nodeIndex), visiting.insert(nodeIndex).inserted,
                  reachedJoints.insert(nodeIndex).inserted
            else {
                throw SmeltGLBError.invalidSkin
            }
            for child in nodes[nodeIndex]["children"] as? [Int] ?? [] {
                guard nodes.indices.contains(child) else {
                    throw SmeltGLBError.invalidSkin
                }
                if jointSet.contains(child) {
                    try walkJoint(child)
                }
            }
            visiting.remove(nodeIndex)
        }
        try walkJoint(skeletonNode)
        guard reachedJoints == jointSet else {
            throw SmeltGLBError.invalidSkin
        }

        var vertexCount = 0
        var weightedVertexCount = 0
        var maximumWeightSumError: Float = 0
        var skinnedMeshCount = 0
        for node in nodes where node["skin"] as? Int == 0 {
            guard let meshIndex = node["mesh"] as? Int,
                  meshes.indices.contains(meshIndex),
                  let primitives = meshes[meshIndex]["primitives"] as? [[String: Any]]
            else {
                throw SmeltGLBError.invalidSkin
            }
            skinnedMeshCount += 1
            for primitive in primitives {
                guard let attributes = primitive["attributes"] as? [String: Any],
                      let positionAccessor = attributes["POSITION"] as? Int,
                      let jointAccessor = attributes["JOINTS_0"] as? Int,
                      let weightAccessor = attributes["WEIGHTS_0"] as? Int
                else {
                    throw SmeltGLBError.invalidSkin
                }
                let positions = try container.float3(accessor: positionAccessor)
                let joints = try container.unsigned4(accessor: jointAccessor)
                let weights = try container.float4(accessor: weightAccessor)
                guard positions.count == joints.count, joints.count == weights.count else {
                    throw SmeltGLBError.invalidSkin
                }
                for row in positions.indices {
                    let rowWeights = weights[row]
                    guard rowWeights.allSatisfy({ $0.isFinite && $0 >= 0 }),
                          joints[row].allSatisfy({ Int($0) < jointNodes.count })
                    else {
                        throw SmeltGLBError.invalidSkin
                    }
                    let error = abs(rowWeights.reduce(0, +) - 1)
                    maximumWeightSumError = max(maximumWeightSumError, error)
                    guard error <= weightSumTolerance else {
                        throw SmeltGLBError.invalidSkin
                    }
                    if rowWeights.contains(where: { $0 > 0 }) {
                        weightedVertexCount += 1
                    }
                }
                vertexCount += positions.count
            }
        }
        guard skinnedMeshCount > 0, vertexCount > 0,
              weightedVertexCount == vertexCount
        else {
            throw SmeltGLBError.invalidSkin
        }
    return SmeltGLBSkinSummary(
            vertexCount: vertexCount,
            jointCount: jointNodes.count,
            weightedVertexCount: weightedVertexCount,
            inverseBindMatrixCount: inverseBindMatrices.count,
            maximumWeightSumError: maximumWeightSumError
        )
    }

    /// Imports every triangle primitive referenced by the active scene and
    /// flattens node transforms into one world-space mesh.
    public static func readMesh(from url: URL) throws -> SmeltMesh {
        let container = try Container(data: Data(contentsOf: url))
        let document = container.document
        let nodes = try array(document, "nodes")
        let meshes = try array(document, "meshes")
        let scenes = try array(document, "scenes")
        let sceneIndex = document["scene"] as? Int ?? 0
        guard scenes.indices.contains(sceneIndex),
              let roots = scenes[sceneIndex]["nodes"] as? [Int]
        else {
            throw SmeltGLBError.invalidDocument("active scene is missing")
        }
        var positions: [SIMD3<Float>] = []
        var normals: [SIMD3<Float>] = []
        var indices: [UInt32] = []
        var visited = Set<Int>()

        func walk(_ nodeIndex: Int, parent: Matrix4) throws {
            guard nodes.indices.contains(nodeIndex) else {
                throw SmeltGLBError.invalidDocument("node index out of range")
            }
            guard visited.insert(nodeIndex).inserted else {
                throw SmeltGLBError.invalidDocument("node graph is cyclic/shared")
            }
            let node = nodes[nodeIndex]
            let world = parent * (try Matrix4(node: node))
            if let meshIndex = node["mesh"] as? Int {
                guard meshes.indices.contains(meshIndex),
                      let primitives = meshes[meshIndex]["primitives"]
                        as? [[String: Any]]
                else {
                    throw SmeltGLBError.invalidDocument("mesh index is invalid")
                }
                for primitive in primitives {
                    guard primitive["mode"] as? Int ?? 4 == 4,
                          let attributes = primitive["attributes"] as? [String: Any],
                          let positionAccessor = attributes["POSITION"] as? Int
                    else {
                        throw SmeltGLBError.unsupported("triangle POSITION required")
                    }
                    let localPositions = try container.float3(accessor: positionAccessor)
                    let localNormals: [SIMD3<Float>]
                    if let normalAccessor = attributes["NORMAL"] as? Int {
                        localNormals = try container.float3(accessor: normalAccessor)
                    } else {
                        localNormals = Array(repeating: .zero, count: localPositions.count)
                    }
                    guard localNormals.count == localPositions.count else {
                        throw SmeltGLBError.invalidDocument(
                            "POSITION/NORMAL counts differ"
                        )
                    }
                    let localIndices: [UInt32]
                    if let accessor = primitive["indices"] as? Int {
                        localIndices = try container.scalarIndices(accessor: accessor)
                    } else {
                        localIndices = localPositions.indices.map(UInt32.init)
                    }
                    guard localIndices.count.isMultiple(of: 3),
                          localIndices.allSatisfy({ Int($0) < localPositions.count })
                    else {
                        throw SmeltGLBError.invalidDocument("indices are invalid")
                    }
                    let base = UInt32(positions.count)
                    positions.append(contentsOf: localPositions.map(world.transformPoint))
                    normals.append(contentsOf: localNormals.map(world.transformNormal))
                    indices.append(contentsOf: localIndices.map { $0 + base })
                }
            }
            if let children = node["children"] as? [Int] {
                for child in children { try walk(child, parent: world) }
            }
        }
        for root in roots { try walk(root, parent: .identity) }
        guard !positions.isEmpty else {
            throw SmeltGLBError.invalidDocument("scene contains no mesh")
        }
        if normals.allSatisfy({ $0 == .zero }) {
            normals = rebuiltNormals(positions: positions, indices: indices)
        }
        return try SmeltMesh(
            positions: positions,
            normals: normals,
            triangleIndices: indices
        )
    }

    /// Writes one standards-compliant, four-influence skinned GLB. The mesh is
    /// flattened intentionally; the generated skeleton remains hierarchical.
  public static func writeSkin(
        mesh: SmeltMesh,
        skeleton: SmeltSkeleton,
        normalization: SmeltMeshNormalization,
        skin: SmeltVertexSkin,
        to url: URL
    ) throws {
        guard skin.vertexCount == mesh.positions.count,
              skin.jointCount == skeleton.joints.count,
              skin.jointIndices.count == mesh.positions.count * 4,
              skin.weights.count == mesh.positions.count * 4,
              !skeleton.joints.isEmpty
        else {
            throw SmeltGLBError.invalidSkin
        }
        var binary = Data()
        var bufferViews: [[String: Any]] = []
        var accessors: [[String: Any]] = []

        func appendView(_ bytes: Data, target: Int? = nil) -> Int {
            while !binary.count.isMultiple(of: 4) { binary.append(0) }
            let offset = binary.count
            binary.append(bytes)
            var view: [String: Any] = [
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": bytes.count,
            ]
            if let target { view["target"] = target }
            bufferViews.append(view)
            return bufferViews.count - 1
        }

        func appendAccessor(
            view: Int,
            componentType: Int,
            count: Int,
            type: String,
            minimum: [Float]? = nil,
            maximum: [Float]? = nil
        ) -> Int {
            var accessor: [String: Any] = [
                "bufferView": view,
                "componentType": componentType,
                "count": count,
                "type": type,
            ]
            if let minimum { accessor["min"] = minimum }
            if let maximum { accessor["max"] = maximum }
            accessors.append(accessor)
            return accessors.count - 1
        }

        let positionBytes = data(mesh.positions.flatMap { [$0.x, $0.y, $0.z] })
        let normalBytes = data(mesh.normals.flatMap { [$0.x, $0.y, $0.z] })
        let jointBytes = data(skin.jointIndices)
        let weightBytes = data(skin.weights)
        let indexBytes = data(mesh.triangleIndices)
        let worldJoints = skeleton.joints.map {
            normalization.denormalize($0.position)
        }
    let inverseBindBytes = data(
      worldJoints.flatMap { point in
            [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                -point.x, -point.y, -point.z, 1,
            ] as [Float]
        })
        var minimum = mesh.positions[0]
        var maximum = mesh.positions[0]
        for point in mesh.positions.dropFirst() {
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
        let positionAccessor = appendAccessor(
            view: appendView(positionBytes, target: 34_962),
            componentType: 5_126,
            count: mesh.positions.count,
            type: "VEC3",
            minimum: [minimum.x, minimum.y, minimum.z],
            maximum: [maximum.x, maximum.y, maximum.z]
        )
        let normalAccessor = appendAccessor(
            view: appendView(normalBytes, target: 34_962),
            componentType: 5_126,
            count: mesh.normals.count,
            type: "VEC3"
        )
        let jointAccessor = appendAccessor(
            view: appendView(jointBytes, target: 34_962),
            componentType: 5_123,
            count: mesh.positions.count,
            type: "VEC4"
        )
        let weightAccessor = appendAccessor(
            view: appendView(weightBytes, target: 34_962),
            componentType: 5_126,
            count: mesh.positions.count,
            type: "VEC4"
        )
        let indexAccessor = appendAccessor(
            view: appendView(indexBytes, target: 34_963),
            componentType: 5_125,
            count: mesh.triangleIndices.count,
            type: "SCALAR"
        )
        let inverseBindAccessor = appendAccessor(
            view: appendView(inverseBindBytes),
            componentType: 5_126,
            count: skeleton.joints.count,
            type: "MAT4"
        )

    var nodes: [[String: Any]] = [
      [
        // Serializer strings are part of the pinned public byte oracle.
            "name": "rig modelMesh",
            "mesh": 0,
            "skin": 0,
      ]
    ]
        for joint in skeleton.joints.indices {
            let parent = skeleton.joints[joint].parent
      let translation =
        parent < 0
                ? worldJoints[joint]
                : worldJoints[joint] - worldJoints[parent]
            var node: [String: Any] = [
                "name": "bone_\(joint)",
                "translation": [translation.x, translation.y, translation.z],
            ]
            let children = skeleton.joints.indices.filter {
                skeleton.joints[$0].parent == joint
            }.map { $0 + 1 }
            if !children.isEmpty { node["children"] = children }
            nodes.append(node)
        }
        let roots = skeleton.joints.indices.filter {
            skeleton.joints[$0].parent < 0
        }
        guard roots.count == 1 else {
            throw SmeltGLBError.invalidSkin
        }
        let jointNodes = skeleton.joints.indices.map { $0 + 1 }
        var document: [String: Any] = [
            "asset": ["version": "2.0", "generator": "Smelt Rig"],
            "scene": 0,
            "scenes": [["nodes": [0, roots[0] + 1]]],
            "nodes": nodes,
      "meshes": [
        [
                "name": "rig modelMesh",
          "primitives": [
            [
                    "attributes": [
                        "POSITION": positionAccessor,
                        "NORMAL": normalAccessor,
                        "JOINTS_0": jointAccessor,
                        "WEIGHTS_0": weightAccessor,
                    ],
                    "indices": indexAccessor,
                    "mode": 4,
            ]
          ],
        ]
      ],
      "skins": [
        [
                "name": "rig modelArmature",
                "inverseBindMatrices": inverseBindAccessor,
                "skeleton": roots[0] + 1,
                "joints": jointNodes,
        ]
      ],
            "bufferViews": bufferViews,
            "accessors": accessors,
            "buffers": [["byteLength": binary.count]],
        ]
        var json = try JSONSerialization.data(
            withJSONObject: document,
            options: [.sortedKeys]
        )
        while !json.count.isMultiple(of: 4) { json.append(0x20) }
        while !binary.count.isMultiple(of: 4) { binary.append(0) }
        document["buffers"] = [["byteLength": binary.count]]
        json = try JSONSerialization.data(withJSONObject: document, options: [.sortedKeys])
        while !json.count.isMultiple(of: 4) { json.append(0x20) }

        var output = Data()
        append(UInt32(0x4654_6C67), to: &output)
        append(UInt32(2), to: &output)
        append(UInt32(12 + 8 + json.count + 8 + binary.count), to: &output)
        append(UInt32(json.count), to: &output)
        append(UInt32(0x4E4F_534A), to: &output)
        output.append(json)
        append(UInt32(binary.count), to: &output)
        append(UInt32(0x004E_4942), to: &output)
        output.append(binary)
        try output.write(to: url, options: .atomic)
    }

    private static func array(
        _ dictionary: [String: Any],
        _ key: String
    ) throws -> [[String: Any]] {
        guard let value = dictionary[key] as? [[String: Any]] else {
            throw SmeltGLBError.invalidDocument("\(key) is missing")
        }
        return value
    }

    private static func rebuiltNormals(
        positions: [SIMD3<Float>],
        indices: [UInt32]
    ) -> [SIMD3<Float>] {
        var normals = Array(repeating: SIMD3<Float>.zero, count: positions.count)
        for triangle in stride(from: 0, to: indices.count, by: 3) {
            let smelt = Int(indices[triangle])
            let ib = Int(indices[triangle + 1])
            let ic = Int(indices[triangle + 2])
            let normal = cross(positions[ib] - positions[smelt], positions[ic] - positions[smelt])
            normals[smelt] += normal
            normals[ib] += normal
            normals[ic] += normal
        }
        return normals.map(normalized)
    }

    private static func cross(
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
        return length > 1e-10 ? value / length : .zero
    }

    private static func data<T>(_ values: [T]) -> Data {
        values.withUnsafeBytes { Data($0) }
    }

    private static func append(_ value: UInt32, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }

    private struct Container {
        let document: [String: Any]
        let binary: Data

        init(data: Data) throws {
            guard data.count >= 20,
                  data.u32(0) == 0x4654_6C67,
                  data.u32(4) == 2,
                  Int(data.u32(8)) == data.count
            else {
                throw SmeltGLBError.invalidContainer
            }
            var cursor = 12
            var json: Data?
            var binary: Data?
            while cursor + 8 <= data.count {
                let length = Int(data.u32(cursor))
                let type = data.u32(cursor + 4)
                cursor += 8
                guard length >= 0, cursor + length <= data.count else {
                    throw SmeltGLBError.invalidContainer
                }
                let chunk = data.subdata(in: cursor..<(cursor + length))
                if type == 0x4E4F_534A { json = chunk }
                if type == 0x004E_4942 { binary = chunk }
                cursor += length
            }
            guard let json, let binary,
                  let object = try JSONSerialization.jsonObject(with: json)
                    as? [String: Any]
            else {
                throw SmeltGLBError.invalidContainer
            }
            document = object
            self.binary = binary
        }

        func float3(accessor index: Int) throws -> [SIMD3<Float>] {
            let view = try accessor(index)
            guard view.componentType == 5_126, view.type == "VEC3" else {
                throw SmeltGLBError.unsupported("expected FLOAT VEC3")
            }
            var result: [SIMD3<Float>] = []
            result.reserveCapacity(view.count)
            for row in 0..<view.count {
                let offset = view.offset + row * view.stride
        result.append(
          SIMD3(
                    binary.f32(offset),
                    binary.f32(offset + 4),
                    binary.f32(offset + 8)
                ))
            }
            return result
        }

        func scalarIndices(accessor index: Int) throws -> [UInt32] {
            let view = try accessor(index)
            guard view.type == "SCALAR", [5_121, 5_123, 5_125].contains(view.componentType)
            else {
                throw SmeltGLBError.unsupported("index accessor type")
            }
            return (0..<view.count).map { row in
                let offset = view.offset + row * view.stride
                switch view.componentType {
                case 5_121: return UInt32(binary[offset])
                case 5_123: return UInt32(binary.u16(offset))
                default: return binary.u32(offset)
                }
            }
        }

        func unsigned4(accessor index: Int) throws -> [[UInt32]] {
            let view = try accessor(index)
            guard view.type == "VEC4", [5_121, 5_123].contains(view.componentType) else {
                throw SmeltGLBError.unsupported("expected unsigned VEC4")
            }
            let componentBytes = view.componentType == 5_121 ? 1 : 2
            return (0..<view.count).map { row in
                let offset = view.offset + row * view.stride
                return (0..<4).map { component in
                    let componentOffset = offset + component * componentBytes
                    return view.componentType == 5_121
                        ? UInt32(binary[componentOffset])
                        : UInt32(binary.u16(componentOffset))
                }
            }
        }

        func float4(accessor index: Int) throws -> [[Float]] {
            try floats(accessor: index, type: "VEC4", componentCount: 4)
        }

        func float16(accessor index: Int) throws -> [[Float]] {
            try floats(accessor: index, type: "MAT4", componentCount: 16)
        }

        private func floats(
            accessor index: Int,
            type: String,
            componentCount: Int
        ) throws -> [[Float]] {
            let view = try accessor(index)
            guard view.componentType == 5_126, view.type == type else {
                throw SmeltGLBError.unsupported("expected FLOAT \(type)")
            }
            return (0..<view.count).map { row in
                let offset = view.offset + row * view.stride
                return (0..<componentCount).map { component in
                    binary.f32(offset + component * 4)
                }
            }
        }

        private func accessor(_ index: Int) throws -> AccessorView {
            let accessors = try SmeltGLB.array(document, "accessors")
            let views = try SmeltGLB.array(document, "bufferViews")
            guard accessors.indices.contains(index),
                  accessors[index]["sparse"] == nil,
                  let viewIndex = accessors[index]["bufferView"] as? Int,
                  views.indices.contains(viewIndex),
                  let count = accessors[index]["count"] as? Int,
                  let componentType = accessors[index]["componentType"] as? Int,
                  let type = accessors[index]["type"] as? String,
                  views[viewIndex]["buffer"] as? Int ?? 0 == 0,
                  let viewLength = views[viewIndex]["byteLength"] as? Int
            else {
                throw SmeltGLBError.invalidDocument("accessor is invalid")
            }
            let components: Int
            switch type {
            case "SCALAR": components = 1
            case "VEC3": components = 3
            case "VEC4": components = 4
            case "MAT4": components = 16
            default: throw SmeltGLBError.unsupported("accessor type \(type)")
            }
            let componentBytes: Int
            switch componentType {
            case 5_121: componentBytes = 1
            case 5_123: componentBytes = 2
            case 5_125, 5_126: componentBytes = 4
            default: throw SmeltGLBError.unsupported("component type")
            }
            let elementBytes = components * componentBytes
            let stride = views[viewIndex]["byteStride"] as? Int ?? elementBytes
            let viewOffset = views[viewIndex]["byteOffset"] as? Int ?? 0
            let accessorOffset = accessors[index]["byteOffset"] as? Int ?? 0
            let offset = viewOffset + accessorOffset
            guard count >= 0, stride >= elementBytes,
                  offset >= viewOffset,
        count == 0
          || offset + (count - 1) * stride + elementBytes
                    <= viewOffset + viewLength,
                  viewOffset + viewLength <= binary.count
            else {
                throw SmeltGLBError.invalidDocument("accessor exceeds buffer")
            }
            return AccessorView(
                offset: offset,
                stride: stride,
                count: count,
                componentType: componentType,
                type: type
            )
        }
    }

    private struct AccessorView {
        let offset: Int
        let stride: Int
        let count: Int
        let componentType: Int
        let type: String
    }

    private struct Matrix4 {
        let values: [Float]

        static let identity = Matrix4(values: [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        ])

        init(values: [Float]) {
            precondition(values.count == 16)
            self.values = values
        }

        init(node: [String: Any]) throws {
            if let matrix = node["matrix"] as? [NSNumber], matrix.count == 16 {
                self.init(values: matrix.map(\.floatValue))
                return
            }
      let translation =
        (node["translation"] as? [NSNumber])?.map(\.floatValue)
                ?? [0, 0, 0]
      let scale =
        (node["scale"] as? [NSNumber])?.map(\.floatValue)
                ?? [1, 1, 1]
      let rotation =
        (node["rotation"] as? [NSNumber])?.map(\.floatValue)
                ?? [0, 0, 0, 1]
            guard translation.count == 3, scale.count == 3, rotation.count == 4 else {
                throw SmeltGLBError.invalidDocument("node TRS is invalid")
            }
            let x = rotation[0]
            let y = rotation[1]
            let z = rotation[2]
            let w = rotation[3]
      let xx = x * x
      let yy = y * y
      let zz = z * z
      let xy = x * y
      let xz = x * z
      let yz = y * z
      let wx = w * x
      let wy = w * y
      let wz = w * z
            self.init(values: [
                (1 - 2 * (yy + zz)) * scale[0],
                (2 * (xy + wz)) * scale[0],
                (2 * (xz - wy)) * scale[0], 0,
                (2 * (xy - wz)) * scale[1],
                (1 - 2 * (xx + zz)) * scale[1],
                (2 * (yz + wx)) * scale[1], 0,
                (2 * (xz + wy)) * scale[2],
                (2 * (yz - wx)) * scale[2],
                (1 - 2 * (xx + yy)) * scale[2], 0,
                translation[0], translation[1], translation[2], 1,
            ])
        }

        static func * (left: Matrix4, right: Matrix4) -> Matrix4 {
            var result = [Float](repeating: 0, count: 16)
            for column in 0..<4 {
                for row in 0..<4 {
                    for inner in 0..<4 {
            result[column * 4 + row] +=
              left.values[inner * 4 + row]
                            * right.values[column * 4 + inner]
                    }
                }
            }
            return Matrix4(values: result)
        }

        func transformPoint(_ point: SIMD3<Float>) -> SIMD3<Float> {
            SIMD3(
                values[0] * point.x + values[4] * point.y + values[8] * point.z
                    + values[12],
                values[1] * point.x + values[5] * point.y + values[9] * point.z
                    + values[13],
                values[2] * point.x + values[6] * point.y + values[10] * point.z
                    + values[14]
            )
        }

        func transformNormal(_ normal: SIMD3<Float>) -> SIMD3<Float> {
      let a = values[0]
      let b = values[4]
      let c = values[8]
      let d = values[1]
      let e = values[5]
      let f = values[9]
      let g = values[2]
      let h = values[6]
      let i = values[10]
      let determinant =
        a * (e * i - f * h) - b * (d * i - f * g)
                + c * (d * h - e * g)
            guard abs(determinant) > 1e-12 else { return .zero }
            let inverse = 1 / determinant
            let result = SIMD3(
                ((e * i - f * h) * normal.x + (f * g - d * i) * normal.y
                    + (d * h - e * g) * normal.z) * inverse,
                ((c * h - b * i) * normal.x + (a * i - c * g) * normal.y
                    + (b * g - a * h) * normal.z) * inverse,
                ((b * f - c * e) * normal.x + (c * d - a * f) * normal.y
                    + (a * e - b * d) * normal.z) * inverse
            )
            return SmeltGLB.normalized(result)
        }
    }
}

/// Structural and numerical facts recovered from a persisted skinned GLB.
public struct SmeltGLBSkinSummary: Codable, Equatable, Sendable {
    /// Number of vertices carrying joint and weight attributes.
    public let vertexCount: Int

    /// Number of joint nodes referenced by the skin.
    public let jointCount: Int

    /// Number of vertices with at least one positive skin weight.
    public let weightedVertexCount: Int

    /// Number of persisted inverse bind matrices.
    public let inverseBindMatrixCount: Int

    /// Largest absolute error from a per-vertex weight sum of one.
    public let maximumWeightSumError: Float
}

extension Data {
  fileprivate func u16(_ offset: Int) -> UInt16 {
        withUnsafeBytes { bytes in
            UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt16.self))
        }
    }

  fileprivate func u32(_ offset: Int) -> UInt32 {
        withUnsafeBytes { bytes in
            UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
        }
    }

  fileprivate func f32(_ offset: Int) -> Float {
        Float(bitPattern: u32(offset))
    }
}

public enum SmeltGLBError: Error, Equatable {
    case invalidContainer
    case invalidDocument(String)
    case unsupported(String)
    case invalidSkin
}
