import Foundation

/// Bounded registry of opaque API session identifiers.
///
/// A session id is only a cache-affinity/liveness handle: transcript tokens and
/// model state remain owned by `SmeltPromptStateCache`. Keeping those roles
/// separate means editing or branching a conversation cannot make the registry
/// substitute stale history. The registry contributes TTL, LRU capacity, and a
/// distinct unknown/evicted signal for API adapters.
public final class SmeltSessionRegistry: @unchecked Sendable {
    private struct Entry {
        let id: String
        var lastUsed: Date
        var sequence: UInt64
    }

    private let maxSessions: Int
    private let idleTimeoutSeconds: TimeInterval
    private let now: @Sendable () -> Date
    private let makeID: @Sendable () -> String
    private let lock = NSLock()
    private var entries: [String: Entry] = [:]
    private var sequence: UInt64 = 0

    public init(
        maxSessions: Int,
        idleTimeoutSeconds: TimeInterval,
        now: @escaping @Sendable () -> Date = Date.init,
        makeID: @escaping @Sendable () -> String = {
            "sess_\(UUID().uuidString.lowercased())"
        }
    ) {
        self.maxSessions = max(1, maxSessions)
        self.idleTimeoutSeconds = max(1, idleTimeoutSeconds)
        self.now = now
        self.makeID = makeID
    }

    /// Allocate an id without publishing it. API adapters can put the id in
    /// response headers first, then call `touch` at their chosen commit point.
    public func allocate() -> String { makeID() }

    /// Validate and refresh an existing id. False distinguishes evicted,
    /// expired, and never-allocated sessions from an ordinary cold cache miss.
    public func contains(_ id: String) -> Bool {
        lock.withLock {
            let instant = now()
            sweepExpired(at: instant)
            guard entries[id] != nil else { return false }
            update(id, at: instant)
            return true
        }
    }

    public func touch(_ id: String) {
        lock.withLock {
            let instant = now()
            sweepExpired(at: instant)
            update(id, at: instant)
            evictIfOverCap()
        }
    }

    public func evict(_ id: String) {
        lock.withLock { _ = entries.removeValue(forKey: id) }
    }

    public var count: Int {
        lock.withLock {
            sweepExpired(at: now())
            return entries.count
        }
    }

    private func update(_ id: String, at instant: Date) {
        sequence &+= 1
        entries[id] = Entry(id: id, lastUsed: instant, sequence: sequence)
    }

    private func sweepExpired(at instant: Date) {
        let cutoff = instant.addingTimeInterval(-idleTimeoutSeconds)
        entries = entries.filter { $0.value.lastUsed >= cutoff }
    }

    private func evictIfOverCap() {
        guard entries.count > maxSessions else { return }
        let victims = entries.values.sorted {
            if $0.lastUsed != $1.lastUsed { return $0.lastUsed < $1.lastUsed }
            return $0.sequence < $1.sequence
        }.prefix(entries.count - maxSessions)
        for victim in victims { entries.removeValue(forKey: victim.id) }
    }
}

private extension NSLock {
    func withLock<R>(_ body: () -> R) -> R {
        lock(); defer { unlock() }
        return body()
    }
}
