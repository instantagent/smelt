import Metal
import XCTest
@testable import SmeltRuntime

final class SuffixLookupDrafterTests: XCTestCase {
    private func taps() throws -> SmeltDrafterTaps {
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let hidden = try XCTUnwrap(
            device.makeBuffer(length: 2, options: .storageModeShared)
        )
        return SmeltDrafterTaps(
            lastHiddenState: hidden,
            hiddenSize: 1,
            attention: [:]
        )
    }

    func testLongestSuffixProposesACompleteBlock() throws {
        let drafter = SmeltSuffixLookupDrafter(maxNeedleLength: 4)
        drafter.resetSuffixCache(
            promptTokens: [1, 2, 3, 4, 5, 6, 7, 8, 3, 4]
        )

        let batch = try drafter.draftStep(
            targetTaps: taps(),
            lastToken: 4,
            position: 9,
            K: 4,
            selectionMode: .argmax
        )

        XCTAssertEqual(batch.candidates, [5, 6, 7, 8])
        XCTAssertEqual(drafter.hits, 1)
        XCTAssertEqual(drafter.misses, 0)
        if case .none = batch.q { } else {
            XCTFail("greedy suffix proposals must not materialize q logits")
        }
    }

    func testHistogramTieBreakIsDeterministic() throws {
        let drafter = SmeltSuffixLookupDrafter(maxNeedleLength: 1)
        drafter.resetSuffixCache(promptTokens: [9, 7, 0, 9, 5, 0, 9])

        let batch = try drafter.draftStep(
            targetTaps: taps(),
            lastToken: 9,
            position: 6,
            K: 1,
            selectionMode: .argmax
        )

        XCTAssertEqual(batch.candidates, [5])
    }

    func testMissDeclinesInsteadOfGuessing() throws {
        let drafter = SmeltSuffixLookupDrafter()
        drafter.resetSuffixCache(promptTokens: [1, 2, 3, 4])

        let batch = try drafter.draftStep(
            targetTaps: taps(),
            lastToken: 4,
            position: 3,
            K: 3,
            selectionMode: .argmax
        )

        XCTAssertTrue(batch.candidates.isEmpty)
        XCTAssertEqual(drafter.hits, 0)
        XCTAssertEqual(drafter.misses, 1)
    }

    func testPreflightDeclinesWithoutTargetTaps() throws {
        let drafter: any SmeltDrafter = SmeltSuffixLookupDrafter()
        let preflight = try XCTUnwrap(drafter as? any SmeltPreflightDrafter)
        preflight.resetSuffixCache(promptTokens: [1, 2, 3, 4])

        let batch = try preflight.preflightDraft(
            lastToken: 4,
            position: 3,
            K: 3,
            selectionMode: .argmax
        )

        XCTAssertTrue(batch.candidates.isEmpty)
    }

    func testGeneratedTokensParticipateWithoutCopyingThePrompt() throws {
        let drafter = SmeltSuffixLookupDrafter(maxNeedleLength: 2)
        drafter.resetSuffixCache(promptTokens: [3, 4, 10, 11, 12, 13])
        drafter.recordGeneratedTokens([3, 4])

        let batch = try drafter.draftStep(
            targetTaps: taps(),
            lastToken: 4,
            position: 7,
            K: 4,
            selectionMode: .argmax
        )

        XCTAssertEqual(batch.candidates, [10, 11, 12, 13])
    }

    func testReplacingGeneratedTailRebuildsOccurrenceIndex() throws {
        let drafter = SmeltSuffixLookupDrafter(maxNeedleLength: 2)
        drafter.resetSuffixCache(promptTokens: [1, 2, 3, 4, 5])
        drafter.recordGeneratedTokens([9, 1, 2])

        let hit = try drafter.preflightDraft(
            lastToken: 2,
            position: 7,
            K: 2,
            selectionMode: .argmax
        )
        XCTAssertEqual(hit.candidates, [3, 4])

        drafter.recordGeneratedTokens([8, 7])
        let miss = try drafter.preflightDraft(
            lastToken: 7,
            position: 6,
            K: 2,
            selectionMode: .argmax
        )
        XCTAssertTrue(miss.candidates.isEmpty)
    }
}
