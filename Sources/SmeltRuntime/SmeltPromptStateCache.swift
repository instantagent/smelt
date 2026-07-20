import CryptoKit
import Foundation

/// Token-prefix cache for complete runtime prompt-state checkpoints.
///
/// Exact-position mode is used whenever a package owns persistent state that
/// cannot be trimmed to an arbitrary token position. LCP mode is available to
/// state layouts whose snapshots have positional prefix semantics. The cache
/// itself is architecture-neutral; the runtime declares which contract holds.
public final class SmeltPromptStateCache {
    public enum Lookup: CustomStringConvertible {
        case notAttempted(reason: String)
        case miss(requested: Int, entries: Int, bestLCP: Int, reason: String)
        case hit(requested: Int, restored: Int, suffix: Int, exact: Bool)

        public var description: String {
            switch self {
            case .notAttempted(let reason):
                return "miss reason=\(reason)"
            case .miss(let requested, let entries, let bestLCP, let reason):
                return "miss requested=\(requested) entries=\(entries) "
                    + "best_lcp=\(bestLCP) reason=\(reason)"
            case .hit(let requested, let restored, let suffix, let exact):
                return "hit requested=\(requested) restored=\(restored) "
                    + "prefill=\(suffix) mode=\(exact ? "exact" : "lcp")"
            }
        }
    }

    public let maxBytes: Int
    public let minMatchTokens: Int
    public let tailFreshTokens: Int
    public let requiresExactRestore: Bool

    private var entries: [SmeltPromptStateCacheEntry] = []
    private var totalBytes = 0
    private var clock: UInt64 = 0
    public private(set) var lastLookup: Lookup = .notAttempted(
        reason: "not-requested"
    )

    public init(
        maxBytes: Int,
        minMatchTokens: Int,
        tailFreshTokens: Int,
        requiresExactRestore: Bool
    ) {
        self.maxBytes = max(0, maxBytes)
        self.minMatchTokens = max(1, minMatchTokens)
        self.tailFreshTokens = max(0, tailFreshTokens)
        self.requiresExactRestore = requiresExactRestore
    }

    public func tryMatch(
        _ tokens: [Int32],
        inputIdentity: SmeltPromptInputIdentity
    ) -> SmeltPromptStateMatch? {
        guard maxBytes > 0, !tokens.isEmpty, !entries.isEmpty else {
            lastLookup = .miss(
                requested: tokens.count,
                entries: entries.count,
                bestLCP: 0,
                reason: entries.isEmpty ? "empty" : "disabled-or-empty-request"
            )
            return nil
        }

        if requiresExactRestore {
            var bestEntryIndex: Int?
            var bestLength = 0
            var bestLCP = 0
            for (index, entry) in entries.enumerated() {
                guard entry.inputIdentity == inputIdentity else { continue }
                let length = entry.tokens.count
                bestLCP = max(bestLCP, computeLCP(entry.tokens, tokens))
                // A suffix token is required to produce boundary logits. An
                // equal-token request misses until logits become cache state.
                guard length >= minMatchTokens,
                      length < tokens.count,
                      length > bestLength,
                      tokens.starts(with: entry.tokens),
                      let snapshot = entry.snapshots.last,
                      snapshot.position == length
                else { continue }
                bestEntryIndex = index
                bestLength = length
            }
            guard let entryIndex = bestEntryIndex,
                  let snapshot = entries[entryIndex].snapshots.last
            else {
                lastLookup = .miss(
                    requested: tokens.count,
                    entries: entries.count,
                    bestLCP: bestLCP,
                    reason: entries.contains(where: {
                        $0.inputIdentity == inputIdentity
                    })
                        ? (bestLCP < minMatchTokens
                            ? "below-minimum" : "no-complete-checkpoint-prefix")
                        : "input-identity-mismatch"
                )
                return nil
            }
            bump(entryAt: entryIndex)
            lastLookup = .hit(
                requested: tokens.count,
                restored: bestLength,
                suffix: tokens.count - bestLength,
                exact: true
            )
            return SmeltPromptStateMatch(
                effectiveLCP: bestLength,
                chosenSnapshot: snapshot,
                restoreExactly: true
            )
        }

        let tailCap = tokens.count - tailFreshTokens
        guard tailCap >= minMatchTokens else {
            lastLookup = .miss(
                requested: tokens.count,
                entries: entries.count,
                bestLCP: 0,
                reason: "tail-fresh-below-minimum"
            )
            return nil
        }

        var bestEntryIndex: Int?
        var bestLCP = 0
        for (index, entry) in entries.enumerated() {
            guard entry.inputIdentity == inputIdentity else { continue }
            let effective = min(computeLCP(entry.tokens, tokens), tailCap)
            if effective >= minMatchTokens && effective > bestLCP {
                bestLCP = effective
                bestEntryIndex = index
            }
        }
        guard let entryIndex = bestEntryIndex else {
            lastLookup = .miss(
                requested: tokens.count,
                entries: entries.count,
                bestLCP: bestLCP,
                reason: entries.contains(where: {
                    $0.inputIdentity == inputIdentity
                }) ? "no-usable-lcp" : "input-identity-mismatch"
            )
            return nil
        }
        guard let snapshot = entries[entryIndex].snapshots.last,
              snapshot.position >= bestLCP - 1
        else {
            lastLookup = .miss(
                requested: tokens.count,
                entries: entries.count,
                bestLCP: bestLCP,
                reason: "snapshot-does-not-cover-lcp"
            )
            return nil
        }
        bump(entryAt: entryIndex)
        lastLookup = .hit(
            requested: tokens.count,
            restored: bestLCP - 1,
            suffix: tokens.count - (bestLCP - 1),
            exact: false
        )
        return SmeltPromptStateMatch(
            effectiveLCP: bestLCP,
            chosenSnapshot: snapshot,
            restoreExactly: false
        )
    }

    public func store(_ entry: SmeltPromptStateCacheEntry) {
        guard maxBytes > 0, !entry.tokens.isEmpty, entry.bytes <= maxBytes else {
            return
        }
        if let existing = entries.firstIndex(where: {
            $0.tokens == entry.tokens && $0.inputIdentity == entry.inputIdentity
        }) {
            totalBytes -= entries[existing].bytes
            entries.remove(at: existing)
        }
        var newEntry = entry
        clock &+= 1
        newEntry.lastUsed = clock
        entries.append(newEntry)
        totalBytes += newEntry.bytes
        evictLRUUntilWithinBudget()
    }

    public var entryCount: Int { entries.count }
    public var bytesUsed: Int { totalBytes }

    private func bump(entryAt index: Int) {
        clock &+= 1
        entries[index].lastUsed = clock
    }

    private func evictLRUUntilWithinBudget() {
        guard totalBytes > maxBytes else { return }
        entries.sort { $0.lastUsed < $1.lastUsed }
        while totalBytes > maxBytes, !entries.isEmpty {
            totalBytes -= entries.removeFirst().bytes
        }
    }

    private func computeLCP(_ left: [Int32], _ right: [Int32]) -> Int {
        let count = min(left.count, right.count)
        var index = 0
        while index < count && left[index] == right[index] { index += 1 }
        return index
    }
}

/// Identity of prompt inputs that are not represented by token IDs. Text-only
/// requests use `.text`. Multimodal callers hash the canonical ordered media
/// inputs or fused embeddings and must provide that digest on both lookup and
/// store; identical placeholder tokens with different media can never match.
public enum SmeltPromptInputIdentity: Equatable, Hashable, Sendable {
    case text
    case nonTokenInputsSHA256(String)

    /// Canonical identity for ordered prompt inputs that token IDs do not
    /// describe (images, audio, adapter payloads, or future CAM port values).
    /// Length-prefixing both the kind and payload keeps concatenations
    /// unambiguous; preserving order makes graph input order part of identity.
    public static func nonTokenInputs(
        _ inputs: [SmeltNonTokenPromptInput]
    ) -> SmeltPromptInputIdentity {
        var hasher = SHA256()
        hasher.update(data: Data("smelt.prompt.non-token-inputs.v1".utf8))
        update(UInt64(inputs.count), in: &hasher)
        for input in inputs {
            let kind = Data(input.kind.utf8)
            update(UInt64(kind.count), in: &hasher)
            hasher.update(data: kind)
            update(UInt64(input.bytes.count), in: &hasher)
            hasher.update(data: input.bytes)
        }
        let digest = hasher.finalize().map {
            String(format: "%02x", $0)
        }.joined()
        return .nonTokenInputsSHA256(digest)
    }

    private static func update(_ value: UInt64, in hasher: inout SHA256) {
        var bigEndian = value.bigEndian
        withUnsafeBytes(of: &bigEndian) { bytes in
            hasher.update(bufferPointer: bytes)
        }
    }
}

public struct SmeltNonTokenPromptInput: Equatable, Sendable {
    public let kind: String
    public let bytes: Data

    public init(kind: String, bytes: Data) {
        precondition(!kind.isEmpty, "non-token prompt input kind must not be empty")
        self.kind = kind
        self.bytes = bytes
    }
}

public struct SmeltPromptStateCacheEntry {
    public let tokens: [Int32]
    public let inputIdentity: SmeltPromptInputIdentity
    public let snapshots: [SmeltPromptStateCheckpoint]
    public let bytes: Int
    var lastUsed: UInt64

    public init(
        tokens: [Int32],
        inputIdentity: SmeltPromptInputIdentity,
        snapshots: [SmeltPromptStateCheckpoint]
    ) {
        precondition(!snapshots.isEmpty, "prompt-state entry requires a snapshot")
        let sorted = snapshots.sorted { $0.position < $1.position }
        precondition(
            sorted.last!.position == tokens.count,
            "highest-position snapshot must cover the complete token range"
        )
        self.tokens = tokens
        self.inputIdentity = inputIdentity
        self.snapshots = sorted
        self.bytes = sorted.reduce(0) { $0 + $1.snapshot.byteCount }
        self.lastUsed = 0
    }
}

public struct SmeltPromptStateCheckpoint {
    public let position: Int
    public let snapshot: SmeltCachedPromptSnapshot

    public init(position: Int, snapshot: SmeltCachedPromptSnapshot) {
        self.position = position
        self.snapshot = snapshot
    }
}

public enum SmeltCachedPromptSnapshot {
    case host(SmeltPromptSnapshot)
    case device(SmeltDevicePromptSnapshot)

    public var byteCount: Int {
        switch self {
        case .host(let snapshot): return snapshot.byteCount
        case .device(let snapshot): return snapshot.byteCount
        }
    }
}

public struct SmeltPromptStateMatch {
    public let effectiveLCP: Int
    public let chosenSnapshot: SmeltPromptStateCheckpoint
    public let restoreExactly: Bool

    public init(
        effectiveLCP: Int,
        chosenSnapshot: SmeltPromptStateCheckpoint,
        restoreExactly: Bool
    ) {
        self.effectiveLCP = effectiveLCP
        self.chosenSnapshot = chosenSnapshot
        self.restoreExactly = restoreExactly
    }
}
