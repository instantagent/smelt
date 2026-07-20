import XCTest
@testable import SmeltCompiler
import SmeltSchema

/// Phase 2 Unit 1 (docs/lego-phase2-plan.md): the pure dtype→region resolver. Asserts the
/// dense dtypes resolve to a single contiguous region at the entry's own offset, affine_u4
/// resolves to the three absolute regions read VERBATIM from the entry (nibbles, scales,
/// biases), and every out-of-scope dtype throws loud (no silent drop to a wrong binding).
final class WeightLocatorTests: XCTestCase {

    private func dense(_ dtype: SmeltDType, offset: UInt64, sizeBytes: UInt64) -> SmeltWeightEntry {
        SmeltWeightEntry(name: "w", offset: offset, sizeBytes: sizeBytes, shape: [4, 8], dtype: dtype)
    }

    func testDenseDTypesResolveToOneRegionAtEntryOffset() throws {
        for dt in [SmeltDType.bf16, .fp16, .fp32] {
            let loc = try WeightLocator.resolve(dense(dt, offset: 4096, sizeBytes: 64))
            XCTAssertEqual(loc.kind, .dense(dt), "\(dt) must resolve to .dense(\(dt))")
            XCTAssertEqual(loc.regions, [WeightLocator.Region(offset: 4096, len: 64)],
                           "\(dt) must be one region at the entry's own offset/size")
        }
    }

    func testAffineU4ResolvesToThreeAbsoluteRegionsVerbatim() throws {
        // Deliberately NON-CONTIGUOUS offsets (gaps between every region): a buggy resolver
        // that recomputed a contiguous layout (scales == offset+sizeBytes, biases ==
        // scales+scalesSizeBytes) instead of reading scalesOffset/biasesOffset verbatim
        // would land on the wrong bytes — so contiguous offsets would let it pass spuriously.
        let e = SmeltWeightEntry(
            name: "q_proj", offset: 8192, sizeBytes: 1024, shape: [16, 64], dtype: .affineU4,
            groupSize: 64,
            scalesOffset: 12000, scalesSizeBytes: 32,
            biasesOffset: 20000, biasesSizeBytes: 32)
        let loc = try WeightLocator.resolve(e)
        XCTAssertEqual(loc.kind, .affineU4(groupSize: 64))
        XCTAssertEqual(loc.regions, [
            WeightLocator.Region(offset: 8192, len: 1024),    // nibbles @ entry.offset
            WeightLocator.Region(offset: 12000, len: 32),     // scales  @ entry.scalesOffset (absolute, gap)
            WeightLocator.Region(offset: 20000, len: 32),     // biases  @ entry.biasesOffset (absolute, gap)
        ], "affine_u4 must surface all three regions verbatim, in weights/scales/biases order")
    }

    func testAffineU4MissingRegionsThrows() {
        // affine_u4 entry with the quant metadata dropped (the exact failure Phase 2a guards
        // against) — must throw, not produce a one-region locator that binds garbage.
        let e = SmeltWeightEntry(
            name: "q_proj", offset: 8192, sizeBytes: 1024, shape: [16, 64], dtype: .affineU4)
        XCTAssertThrowsError(try WeightLocator.resolve(e)) { error in
            guard case WeightLocator.ResolveError.missingAffineRegions = error else {
                return XCTFail("expected .missingAffineRegions, got \(error)")
            }
        }
    }

    func testOutOfScopeDTypesThrowLoud() {
        for dt in [SmeltDType.u4Lut, .turboQuantH, .int32, .raw] {
            let e = SmeltWeightEntry(name: "w", offset: 0, sizeBytes: 16, shape: [4, 4], dtype: dt)
            XCTAssertThrowsError(try WeightLocator.resolve(e), "\(dt) is out of F32Trunk scope") { error in
                guard case WeightLocator.ResolveError.unsupportedDType = error else {
                    return XCTFail("expected .unsupportedDType for \(dt), got \(error)")
                }
            }
        }
    }
}
