import Foundation
import Metal
import XCTest

@testable import SmeltRuntime

final class SmeltSkinConditionEncoderTests: XCTestCase {
    func testReducedConditionEncoderMatchesPinnedTorchFP32Fixture() throws {
    guard let packagePath = ProcessInfo.processInfo.environment["SMELT_SKINNING_PACKAGE"] else {
      throw XCTSkip("SMELT_SKINNING_PACKAGE is not set")
        }
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("No Metal device available")
        }
    let artifact = try SmeltComponentArtifact(path: packagePath, verify: true)
        let runtime = try SmeltSkinConditionEncoder(artifact: artifact)
        let query = decode(skinConditionQueryBase64, count: 3 * 768)
        let data = decode(skinConditionDataBase64, count: 5 * 768)
        let expected = decode(skinConditionExpectedBase64, count: 3 * 512)
        let actual = try runtime.encodeReduced(query: query, data: data)
        XCTAssertEqual(actual.count, expected.count)
        XCTAssertTrue(actual.allSatisfy(\.isFinite))
        var maximumDifference: Float = 0
        var squaredError: Double = 0
        var squaredReference: Double = 0
        for index in actual.indices {
            let difference = actual[index] - expected[index]
            maximumDifference = max(maximumDifference, abs(difference))
            squaredError += Double(difference * difference)
            squaredReference += Double(expected[index] * expected[index])
        }
        XCTAssertLessThan(maximumDifference, 8e-4)
        XCTAssertLessThan(sqrt(squaredError / squaredReference), 8e-4)
    }

    func testProductionShapeConditionEncoderSmoke() throws {
    guard ProcessInfo.processInfo.environment["SMELT_SKINNING_FULL_SMOKE"] == "1" else {
      throw XCTSkip("SMELT_SKINNING_FULL_SMOKE is not enabled")
        }
    guard let packagePath = ProcessInfo.processInfo.environment["SMELT_SKINNING_PACKAGE"] else {
      throw XCTSkip("SMELT_SKINNING_PACKAGE is not set")
        }
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("No Metal device available")
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
    let artifact = try SmeltComponentArtifact(path: packagePath, verify: true)
        let runtime = try SmeltSkinConditionEncoder(artifact: artifact)
        let result = try runtime.encode(pointNormals: points)
        XCTAssertEqual(result.selectedSourceIndices.count, 384)
        XCTAssertEqual(result.conditionTokens.count, 384 * 512)
        XCTAssertTrue(result.conditionTokens.allSatisfy(\.isFinite))
        XCTAssertGreaterThan(result.conditionTokens.map(abs).max() ?? 0, 0.01)
    }

    private func decode(_ encoded: String, count: Int) -> [Float] {
        guard let data = Data(base64Encoded: encoded), data.count == count * 4 else {
            XCTFail("corrupt skin-condition fixture")
            return []
        }
        return data.withUnsafeBytes { bytes in
            (0..<count).map { index in
                Float(
                    bitPattern: bytes.loadUnaligned(
                        fromByteOffset: index * 4,
                        as: UInt32.self
                    ).littleEndian
                )
            }
        }
    }
}
