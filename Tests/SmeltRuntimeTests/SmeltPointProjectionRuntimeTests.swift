import Metal
import XCTest

@testable import SmeltRuntime

final class SmeltPointProjectionRuntimeTests: XCTestCase {
    func testPackagedPointProjectionsMatchIndependentCPUOracle() throws {
        guard let packagePath = ProcessInfo.processInfo.environment["SMELT_RIG_PACKAGE"] else {
            throw XCTSkip("SMELT_RIG_PACKAGE is not set")
        }
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("No Metal device available")
        }
        let artifact = try SmeltRigArtifact(path: packagePath, verify: true)
        let runtime = try SmeltPointProjectionRuntime(artifact: artifact)
        let pointNormals: [Float] = [
            -0.875, 0.125, 0.75, 0.25, -0.5, 1,
            0, -0.375, 0.625, -1, 0.75, -0.25,
            0.3125, 0.9375, -0.6875, 0.5, 0.125, -0.875,
            1.125, -1.25, 0.0625, -0.375, -0.625, 0.75,
        ]

        for projection in [SmeltPointProjection.mesh, .condition] {
            let actual = try runtime.project(
                pointNormals: pointNormals,
                projection: projection
            )
            let expected = try cpuProjection(
                pointNormals: pointNormals,
                projection: projection,
                artifact: artifact
            )
            XCTAssertEqual(actual.count, expected.count)
            XCTAssertTrue(actual.allSatisfy(\.isFinite))
            var maximumDifference: Float = 0
            for index in actual.indices {
                maximumDifference = max(maximumDifference, abs(actual[index] - expected[index]))
            }
            XCTAssertLessThan(
                maximumDifference,
                3e-5,
                "\(projection) packaged projection diverged from the source-formula CPU oracle"
            )
        }
    }

    func testPointProjectionPreservesRowsAcrossCommandBufferBoundary() throws {
        guard let packagePath = ProcessInfo.processInfo.environment["SMELT_RIG_PACKAGE"] else {
            throw XCTSkip("SMELT_RIG_PACKAGE is not set")
        }
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("No Metal device available")
        }
        let artifact = try SmeltRigArtifact(path: packagePath, verify: true)
        let runtime = try SmeltPointProjectionRuntime(artifact: artifact)
        let rowCount = SmeltPointProjectionRuntime.maximumRowsPerCommandBuffer + 1
        var pointNormals = [Float](repeating: 0, count: rowCount * 6)
        for row in 0..<rowCount {
            for column in 0..<6 {
                pointNormals[row * 6 + column] = Float((row * 13 + column * 17) % 257) / 128 - 1
            }
        }
        let inspectedRows = [
            0,
            SmeltPointProjectionRuntime.maximumRowsPerCommandBuffer - 1,
            SmeltPointProjectionRuntime.maximumRowsPerCommandBuffer,
        ]
        let inspectedInputs = inspectedRows.flatMap { row in
            Array(pointNormals[(row * 6)..<(row * 6 + 6)])
        }

        for projection in [SmeltPointProjection.mesh, .condition] {
            let actual = try runtime.project(
                pointNormals: pointNormals,
                projection: projection
            )
            let expected = try cpuProjection(
                pointNormals: inspectedInputs,
                projection: projection,
                artifact: artifact
            )
            let outputDimension = actual.count / rowCount
            for (expectedRow, actualRow) in inspectedRows.enumerated() {
                for column in 0..<outputDimension {
                    XCTAssertEqual(
                        actual[actualRow * outputDimension + column],
                        expected[expectedRow * outputDimension + column],
                        accuracy: 3e-5
                    )
                }
            }
        }
    }

    private func cpuProjection(
        pointNormals: [Float],
        projection: SmeltPointProjection,
        artifact: SmeltRigArtifact
    ) throws -> [Float] {
        let names: (weight: String, bias: String, output: Int, includePi: Bool, pmpe: Bool)
        switch projection {
        case .mesh:
            names = (
                "mesh_encoder.encoder.input_proj.weight",
                "mesh_encoder.encoder.input_proj.bias",
                512,
                false,
                false
            )
        case .condition:
            names = (
                "vae.model.cond_encoder.proj_in.weight",
                "vae.model.cond_encoder.proj_in.bias",
                768,
                true,
                true
            )
        }
        let descriptors = Dictionary(
            uniqueKeysWithValues: artifact.checkpointTensors.map { ($0.name, $0) }
        )
        let weightDescriptor = try XCTUnwrap(descriptors[names.weight])
        let biasDescriptor = try XCTUnwrap(descriptors[names.bias])
        let weight = artifact.checkpointTensorData(weightDescriptor)
            .bindMemory(to: UInt16.self, capacity: names.output * 54)
        let bias = artifact.checkpointTensorData(biasDescriptor)
            .bindMemory(to: UInt16.self, capacity: names.output)
        let rows = pointNormals.count / 6
        var output = [Float](repeating: 0, count: rows * names.output)
        for row in 0..<rows {
            let xyz = Array(pointNormals[(row * 6)..<(row * 6 + 3)])
            var features = xyz
            for coordinate in xyz {
                for frequencyIndex in 0..<8 {
                    let frequency = Float(1 << frequencyIndex)
                        * (names.includePi ? Float.pi : 1)
                    var value = sin(coordinate * frequency)
                    if names.pmpe {
                        let fraction = Float(frequencyIndex + 1) / 8
                        let phase = (pow(8, 1 - fraction) + fraction) * (2 * Float.pi)
                        value += sin(coordinate * (0.5 * Float.pi) + phase)
                    }
                    features.append(value)
                }
            }
            for coordinate in xyz {
                for frequencyIndex in 0..<8 {
                    let frequency = Float(1 << frequencyIndex)
                        * (names.includePi ? Float.pi : 1)
                    var value = cos(coordinate * frequency)
                    if names.pmpe {
                        let fraction = Float(frequencyIndex + 1) / 8
                        let phase = (pow(8, 1 - fraction) + fraction) * (2 * Float.pi)
                        value += cos(coordinate * (0.5 * Float.pi) + phase)
                    }
                    features.append(value)
                }
            }
            features.append(contentsOf: pointNormals[(row * 6 + 3)..<(row * 6 + 6)])
            XCTAssertEqual(features.count, 54)
            for column in 0..<names.output {
                var value = widenBF16(bias[column])
                for inner in 0..<54 {
                    value += features[inner] * widenBF16(weight[column * 54 + inner])
                }
                output[row * names.output + column] = value
            }
        }
        return output
    }

    private func widenBF16(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }
}
