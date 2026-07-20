import Foundation
import Testing
@testable import SmeltCompiler
import SmeltSchema

// Integration test for the general (runtime-driven) GPTQ calibrator against the
// in-repo metal-prefill vehicle. Gated on both the built package and the HF
// checkpoint; skips cleanly when either is absent.

private let qwen35PkgPath = "/tmp/qwen35-0.8b-pkg/Qwen_Qwen3.5-0.8B.smeltpkg"
private let qwen35Checkpoint = "/tmp/qwen35-0.8b"

@Test func gptqRuntimeCalibratorProducesInjectableBlocks() throws {
    guard FileManager.default.fileExists(atPath: "\(qwen35PkgPath)/gptq_capture_points.json"),
          FileManager.default.fileExists(atPath: "\(qwen35Checkpoint)/config.json")
    else { return }

    let ir = SmeltModelIR.qwen35_0_8B
    let tokens: [[Int32]] = [
        (0..<12).map { Int32(100 + $0) },
        (0..<8).map { Int32(2000 + $0 * 3) },
    ]

    let (blocks, ranks) = try SmeltGPTQCalibrator.calibrateRuntime(
        packagePath: qwen35PkgPath,
        checkpointDir: qwen35Checkpoint,
        ir: ir,
        calibrationTokens: tokens,
        layersPerPass: 6
    )

    // Every in-scope affine_u4 projection got a block, and nothing out of scope did.
    let layout = SmeltWeightLayout.computeLayout(from: ir)
    let expected = Set(SmeltGPTQScope.inResolvedScope(layout).map(\.name))
    #expect(!expected.isEmpty)
    #expect(Set(blocks.keys) == expected,
            "missing \(expected.subtracting(blocks.keys)); extra \(Set(blocks.keys).subtracting(expected))")

    // Each block's shape matches its layout entry, so it injects via fillAffineFromBlock,
    // and its packed arrays satisfy the affine_u4 length invariants.
    let entryByName = Dictionary(uniqueKeysWithValues: layout.map { ($0.name, $0) })
    for (name, block) in blocks {
        let entry = try #require(entryByName[name])
        #expect(block.rows == entry.shape[0])
        #expect(block.cols == entry.shape[1])
        #expect(block.nibbles.count == block.rows * ((block.cols + 1) / 2))
        let groups = (block.cols + block.groupSize - 1) / block.groupSize
        #expect(block.scales.count == block.rows * groups)
        #expect(block.biases.count == block.rows * groups)
        // A real calibration produces non-degenerate scales (not all zero).
        #expect(!block.scales.allSatisfy { $0 == 0 })
        // Hessian rank is bounded by the calibration token count (sum of seqLens).
        let totalTokens = tokens.reduce(0) { $0 + $1.count }
        let rank = try #require(ranks[name])
        #expect(rank >= 1 && rank <= min(totalTokens, block.cols))
    }
}
