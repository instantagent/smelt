import Foundation
import Metal
import XCTest

@testable import SmeltCompiler
@testable import SmeltRuntime

final class SmeltRigArtifactParityTests: XCTestCase {
    func testCanonicalPackageIsBitExactAndRoutable() throws {
        guard let checkpointPath = ProcessInfo.processInfo.environment["SMELT_RIG_CHECKPOINT"],
              let packagePath = ProcessInfo.processInfo.environment["SMELT_RIG_PACKAGE"]
        else {
            throw XCTSkip("SMELT_RIG_CHECKPOINT and SMELT_RIG_PACKAGE are not set")
        }
        let checkpoint = try PyTorchCheckpointLoader(path: checkpointPath)
        let plan = try SmeltRigCheckpointPlan(checkpoint: checkpoint)
        let artifact = try SmeltRigArtifact(path: packagePath, verify: true)
        let sourceDescriptors = checkpoint.checkpointTensors
        let artifactDescriptors = Dictionary(
            uniqueKeysWithValues: artifact.checkpointTensors.map { ($0.name, $0) }
        )
        XCTAssertEqual(artifactDescriptors.count, plan.carriedTensors.count)

        var comparedBytes = 0
        for tensor in plan.carriedTensors {
            let sourceDescriptor = sourceDescriptors[tensor.descriptorIndex]
            let artifactDescriptor = try XCTUnwrap(artifactDescriptors[tensor.name])
            XCTAssertEqual(artifactDescriptor.dtype, sourceDescriptor.dtype)
            XCTAssertEqual(artifactDescriptor.shape, sourceDescriptor.shape)
            XCTAssertEqual(artifactDescriptor.byteCount, sourceDescriptor.byteCount)
            let source = checkpoint.checkpointTensorData(sourceDescriptor)
            let packaged = artifact.checkpointTensorData(artifactDescriptor)
            XCTAssertEqual(
                memcmp(source, packaged, sourceDescriptor.byteCount),
                0,
                "BF16 package bytes differ for \(tensor.name)"
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
                "declared TokenRig Metal function is absent: \(pipeline)"
            )
        }
        print(
            "RIG_U0_PACKAGE_PARITY tensors=\(artifactDescriptors.count) "
                + "bytes=\(comparedBytes) pipelines=\(artifact.manifest.pipelines.count)"
        )
    }
}
