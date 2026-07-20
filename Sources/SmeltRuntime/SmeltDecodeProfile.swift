// SmeltDecodeProfile — env-flag-gated per-decodeStep timing recorder.
// The static state is process-wide because multi-position decode threads
// every decodeStep call (both draft-model and target refresh) through
// one shared global, and the draft-model wrapper drains the records
// per K-loop iteration.

import Foundation

enum SmeltDecodeProfile {
    static let enabled: Bool =
        ProcessInfo.processInfo.environment["SMELT_DECODE_PROFILE"] == "1"

    /// SMELT_TTS_UNBATCHED=1: `batched {}` scopes run each dispatch as its own committed+waited
    /// command buffer, so the profiled `encode()` path records ISOLATED per-kernel GPU times
    /// (k_<function> buckets). Diagnostic only — much slower, never set in production.
    static let unbatched: Bool =
        ProcessInfo.processInfo.environment["SMELT_TTS_UNBATCHED"] == "1"

    struct Record {
        let encodeUs: Double
        let submitUs: Double
        let gpuWaitUs: Double
    }

    nonisolated(unsafe) private static var pending: [Record] = []
    private static let lock = NSLock()

    static func record(
        encodeUs: Double, submitUs: Double, gpuWaitUs: Double
    ) {
        lock.lock()
        defer { lock.unlock() }
        pending.append(
            Record(
                encodeUs: encodeUs,
                submitUs: submitUs,
                gpuWaitUs: gpuWaitUs
            )
        )
    }

    static func flush() -> [Record] {
        lock.lock()
        defer { lock.unlock() }
        let drained = pending
        pending = []
        return drained
    }

    // MARK: - Category × stage attribution
    // A general per-bucket accumulator keyed "stage.category", on top of the existing
    // per-dispatch Record path. Always enabled at the API level (single dict add under the
    // lock when the caller decides to record); call sites gate on `enabled` so the hot path
    // pays only a bool check when SMELT_DECODE_PROFILE is off.

    nonisolated(unsafe) private static var bucketNs: [String: UInt64] = [:]
    nonisolated(unsafe) private static var bucketCount: [String: Int] = [:]
    nonisolated(unsafe) private static var stage: String = "none"

    static func setStage(_ s: String) {
        guard enabled else { return }
        lock.lock(); stage = s; lock.unlock()
    }

    /// Clear the accumulators so each profiled run starts from a clean window (a prior run that
    /// threw before `report()` would otherwise leak its buckets into this run's attribution).
    static func reset() {
        lock.lock()
        bucketNs = [:]; bucketCount = [:]; stage = "none"
        lock.unlock()
    }

    /// Accumulate `ns` nanoseconds (and a hit) into `<currentStage>.<category>`.
    static func add(_ category: String, _ ns: UInt64) {
        lock.lock()
        defer { lock.unlock() }
        let key = "\(stage).\(category)"
        bucketNs[key, default: 0] += ns
        bucketCount[key, default: 0] += 1
    }

    /// Drain the bucket accumulators (ns, count) keyed by "stage.category".
    static func flushBuckets() -> (ns: [String: UInt64], count: [String: Int]) {
        lock.lock()
        defer { lock.unlock() }
        let r = (bucketNs, bucketCount)
        bucketNs = [:]; bucketCount = [:]; stage = "none"
        return r
    }

    /// Format the drained buckets against a measured top-level wall, sorted by time, with the
    /// `uncategorized` remainder (so attribution is honest about what it did NOT capture).
    static func report(totalWallS: Double) -> String {
        let (ns, count) = flushBuckets()
        let attributedS = ns.values.reduce(0) { $0 + Double($1) / 1e9 }
        var lines = ["=== Qwen3-TTS GPU perf attribution ===",
                     String(format: "total wall: %.2f s", totalWallS)]
        for (key, n) in ns.sorted(by: { $0.value > $1.value }) {
            let s = Double(n) / 1e9
            lines.append(String(format: "  %-22@  %8.2f s  %5.1f%%  (n=%d)",
                                key as NSString, s, 100 * s / totalWallS, count[key] ?? 0))
        }
        let uncat = totalWallS - attributedS
        lines.append(String(format: "  %-22@  %8.2f s  %5.1f%%", "uncategorized" as NSString,
                            uncat, 100 * uncat / totalWallS))
        return lines.joined(separator: "\n")
    }
}
