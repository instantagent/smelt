import Foundation
import Metal
import XCTest

@testable import SmeltCompiler
@testable import SmeltRuntime

final class SmeltComponentArtifactParityTests: XCTestCase {
    func testCanonicalPackageIsBitExactAndRoutable() throws {
        guard let checkpointPath = ProcessInfo.processInfo.environment["SMELT_SKINNING_CHECKPOINT"],
              let packagePath = ProcessInfo.processInfo.environment["SMELT_SKINNING_PACKAGE"]
        else {
            throw XCTSkip("SMELT_SKINNING_CHECKPOINT and SMELT_SKINNING_PACKAGE are not set")
        }
        let checkpoint = try PyTorchCheckpointLoader(path: checkpointPath)
        let artifact = try SmeltComponentArtifact(path: packagePath, verify: true)
        let sourceDescriptors = Dictionary(
            uniqueKeysWithValues: checkpoint.checkpointTensors.map { ($0.name, $0) }
        )
        let artifactDescriptors = Dictionary(
            uniqueKeysWithValues: artifact.checkpointTensors.map { ($0.name, $0) }
        )
        XCTAssertEqual(
            Set(artifactDescriptors.keys).union(artifact.manifest.omittedTensors),
            Set(sourceDescriptors.keys)
        )

        var comparedBytes = 0
        for (name, artifactDescriptor) in artifactDescriptors {
            let sourceDescriptor = try XCTUnwrap(sourceDescriptors[name])
            XCTAssertEqual(artifactDescriptor.dtype, sourceDescriptor.dtype)
            XCTAssertEqual(artifactDescriptor.shape, sourceDescriptor.shape)
            XCTAssertEqual(artifactDescriptor.byteCount, sourceDescriptor.byteCount)
            let source = checkpoint.checkpointTensorData(sourceDescriptor)
            let packaged = artifact.checkpointTensorData(artifactDescriptor)
            XCTAssertEqual(
                memcmp(source, packaged, sourceDescriptor.byteCount),
                0,
                "package tensor bytes differ for \(name)"
            )
            comparedBytes += sourceDescriptor.byteCount
        }
        XCTAssertGreaterThan(comparedBytes, 1_000_000_000)

        let byName = Dictionary(
            uniqueKeysWithValues: artifact.manifest.tensors.map { ($0.name, $0) }
        )
        let embedding = try XCTUnwrap(byName["transformer.model.embed_tokens.weight"])
        let head = try XCTUnwrap(byName["transformer.lm_head.weight"])
        XCTAssertEqual(embedding.offset, head.offset)
        XCTAssertEqual(embedding.storageID, head.storageID)

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device available")
        }
        let library = try artifact.makeLibrary(device: device)
        for pipeline in artifact.manifest.pipelines {
            XCTAssertNotNil(
                library.makeFunction(name: pipeline),
                "declared SkinTokens Metal function is absent: \(pipeline)"
            )
        }
        print(
            "SKINNING_U0_PACKAGE_PARITY tensors=\(artifactDescriptors.count) "
                + "bytes=\(comparedBytes) pipelines=\(artifact.manifest.pipelines.count)"
        )
    }
}
