import Foundation
import Testing
@testable import SmeltRuntime

@Suite struct SmeltPromptStateCacheTests {
    private func checkpoint(position: Int, bytes: Int = 100) -> SmeltPromptStateCheckpoint {
        let snapshot = SmeltPromptSnapshot(
            promptLength: position,
            nextToken: 0,
            byteCount: bytes,
            capturedLength: position,
            replayTokenIds: [],
            convStates: [],
            recStates: [],
            keyCaches: [],
            valueCaches: []
        )
        return SmeltPromptStateCheckpoint(
            position: position,
            snapshot: .host(snapshot)
        )
    }

    private func entry(
        _ tokens: [Int32],
        bytes: Int = 100,
        inputIdentity: SmeltPromptInputIdentity = .text
    ) -> SmeltPromptStateCacheEntry {
        SmeltPromptStateCacheEntry(
            tokens: tokens,
            inputIdentity: inputIdentity,
            snapshots: [checkpoint(position: tokens.count, bytes: bytes)]
        )
    }

    @Test func exactModeRestoresOnlyCompleteCheckpointPrefixes() throws {
        let cache = SmeltPromptStateCache(
            maxBytes: 1_000,
            minMatchTokens: 2,
            tailFreshTokens: 0,
            requiresExactRestore: true
        )
        cache.store(entry([1, 2, 3]))

        let match = try #require(cache.tryMatch(
            [1, 2, 3, 4], inputIdentity: .text
        ))
        #expect(match.effectiveLCP == 3)
        #expect(match.restoreExactly)
        #expect(match.chosenSnapshot.position == 3)

        #expect(cache.tryMatch([1, 2, 3], inputIdentity: .text) == nil)
        #expect(cache.lastLookup.description.contains("no-complete-checkpoint-prefix"))
        #expect(cache.tryMatch([1, 2, 9, 4], inputIdentity: .text) == nil)
        #expect(cache.lastLookup.description.contains("best_lcp=2"))
    }

    @Test func lcpModeKeepsTailFreshAndReportsPartialRestore() throws {
        let cache = SmeltPromptStateCache(
            maxBytes: 1_000,
            minMatchTokens: 2,
            tailFreshTokens: 1,
            requiresExactRestore: false
        )
        cache.store(entry([1, 2, 3, 4]))

        let match = try #require(cache.tryMatch(
            [1, 2, 3, 9, 10], inputIdentity: .text
        ))
        #expect(match.effectiveLCP == 3)
        #expect(!match.restoreExactly)
        #expect(cache.lastLookup.description.contains("restored=2"))
        #expect(cache.lastLookup.description.contains("prefill=3"))
    }

    @Test func duplicateTokensReplaceAndLRUEvictsLeastRecentEntry() throws {
        let cache = SmeltPromptStateCache(
            maxBytes: 200,
            minMatchTokens: 2,
            tailFreshTokens: 0,
            requiresExactRestore: true
        )
        let first: [Int32] = [1, 1]
        let second: [Int32] = [2, 2]
        let third: [Int32] = [3, 3]
        cache.store(entry(first))
        cache.store(entry(second))
        #expect(cache.entryCount == 2)
        #expect(cache.bytesUsed == 200)

        _ = try #require(cache.tryMatch(first + [9], inputIdentity: .text))
        cache.store(entry(third))
        #expect(cache.entryCount == 2)
        #expect(cache.tryMatch(second + [9], inputIdentity: .text) == nil)
        #expect(cache.tryMatch(first + [9], inputIdentity: .text) != nil)
        #expect(cache.tryMatch(third + [9], inputIdentity: .text) != nil)

        cache.store(entry(first, bytes: 80))
        #expect(cache.entryCount == 2)
        #expect(cache.bytesUsed == 180)
    }

    @Test func oversizedAndDisabledEntriesDoNotDisplaceUsefulState() {
        let cache = SmeltPromptStateCache(
            maxBytes: 100,
            minMatchTokens: 1,
            tailFreshTokens: 0,
            requiresExactRestore: true
        )
        cache.store(entry([1], bytes: 100))
        cache.store(entry([2], bytes: 101))
        #expect(cache.entryCount == 1)
        #expect(cache.tryMatch([1, 3], inputIdentity: .text) != nil)
        #expect(cache.tryMatch([2, 3], inputIdentity: .text) == nil)
    }

    @Test func multimodalIdentityPreventsSameTokenCrossMediaRestore() throws {
        let cache = SmeltPromptStateCache(
            maxBytes: 1_000,
            minMatchTokens: 2,
            tailFreshTokens: 0,
            requiresExactRestore: true
        )
        let imageA = SmeltPromptInputIdentity.nonTokenInputsSHA256("image-a")
        let imageB = SmeltPromptInputIdentity.nonTokenInputsSHA256("image-b")
        cache.store(entry([1, 2, 3], inputIdentity: imageA))

        #expect(cache.tryMatch([1, 2, 3, 4], inputIdentity: imageB) == nil)
        #expect(cache.lastLookup.description.contains("input-identity-mismatch"))
        #expect(cache.tryMatch([1, 2, 3, 4], inputIdentity: .text) == nil)
        #expect(cache.tryMatch([1, 2, 3, 4], inputIdentity: imageA) != nil)
    }

    @Test func nonTokenIdentityIsCanonicalAndOrderSensitive() {
        let imageA = SmeltNonTokenPromptInput(
            kind: "image", bytes: Data([0, 1, 2, 3])
        )
        let imageB = SmeltNonTokenPromptInput(
            kind: "image", bytes: Data([4, 5, 6])
        )
        let audioA = SmeltNonTokenPromptInput(
            kind: "audio", bytes: imageA.bytes
        )

        let first = SmeltPromptInputIdentity.nonTokenInputs([imageA, imageB])
        #expect(first == SmeltPromptInputIdentity.nonTokenInputs([imageA, imageB]))
        #expect(first != SmeltPromptInputIdentity.nonTokenInputs([imageB, imageA]))
        #expect(
            SmeltPromptInputIdentity.nonTokenInputs([imageA])
                != SmeltPromptInputIdentity.nonTokenInputs([audioA])
        )
        #expect(
            SmeltPromptInputIdentity.nonTokenInputs([
                .init(kind: "image", bytes: Data([0, 1])),
                .init(kind: "image", bytes: Data([2, 3])),
            ])
                != SmeltPromptInputIdentity.nonTokenInputs([imageA])
        )
    }
}
