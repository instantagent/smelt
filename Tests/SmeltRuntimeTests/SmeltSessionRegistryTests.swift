import Foundation
import Testing
@testable import SmeltRuntime

@Suite struct SmeltSessionRegistryTests {
    private final class TestClock: @unchecked Sendable {
        private let lock = NSLock()
        private var stored = Date(timeIntervalSince1970: 1_000)

        var now: Date {
            lock.lock(); defer { lock.unlock() }
            return stored
        }

        func advance(_ seconds: TimeInterval) {
            lock.lock(); defer { lock.unlock() }
            stored = stored.addingTimeInterval(seconds)
        }
    }

    @Test func allocatedIDIsNotLiveUntilTouched() {
        let registry = SmeltSessionRegistry(
            maxSessions: 2,
            idleTimeoutSeconds: 30,
            makeID: { "sess_test" }
        )
        let id = registry.allocate()
        #expect(id == "sess_test")
        #expect(!registry.contains(id))
        registry.touch(id)
        #expect(registry.contains(id))
        #expect(registry.count == 1)
    }

    @Test func containsRefreshesIdleDeadlineAndExpiredIDsDisappear() {
        let clock = TestClock()
        let registry = SmeltSessionRegistry(
            maxSessions: 2,
            idleTimeoutSeconds: 10,
            now: { clock.now }
        )
        registry.touch("a")
        clock.advance(9)
        #expect(registry.contains("a"))
        clock.advance(9)
        #expect(registry.contains("a"))
        clock.advance(11)
        #expect(!registry.contains("a"))
        #expect(registry.count == 0)
    }

    @Test func capacityEvictsLeastRecentlyUsedWithStableTieBreak() {
        let clock = TestClock()
        let registry = SmeltSessionRegistry(
            maxSessions: 2,
            idleTimeoutSeconds: 100,
            now: { clock.now }
        )
        registry.touch("a")
        registry.touch("b")
        #expect(registry.contains("a"))
        registry.touch("c")

        #expect(registry.contains("a"))
        #expect(!registry.contains("b"))
        #expect(registry.contains("c"))
        #expect(registry.count == 2)
    }

    @Test func explicitEvictionInvalidatesOnlyNamedSession() {
        let registry = SmeltSessionRegistry(
            maxSessions: 2,
            idleTimeoutSeconds: 100
        )
        registry.touch("branch-a")
        registry.touch("branch-b")
        registry.evict("branch-a")
        #expect(!registry.contains("branch-a"))
        #expect(registry.contains("branch-b"))
    }
}
