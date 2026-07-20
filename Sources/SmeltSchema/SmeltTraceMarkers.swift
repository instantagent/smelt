import Foundation

/// Optional debug trace metadata written into a .smeltpkg.
///
/// This is not used by the hot path. It lets the CLI stop prefill/decode at
/// named boundaries and compare intermediate buffers.
public struct SmeltTraceMarkers: Codable, Sendable {
    public let decode: [SmeltTraceMarker]
    public let prefill: [SmeltTraceMarker]

    public init(
        decode: [SmeltTraceMarker] = [],
        prefill: [SmeltTraceMarker] = []
    ) {
        self.decode = decode
        self.prefill = prefill
    }
}

/// One named boundary in the dispatch stream.
public struct SmeltTraceMarker: Codable, Sendable {
    /// Human-readable stage label, for example `L3.mid` or `L3.out`.
    public let label: String
    /// Number of dispatch-table ops to execute before stopping.
    /// Counts both Metal dispatches and swap records.
    public let dispatchCount: Int
    /// Buffer slot containing the logical output at this boundary.
    public let bufferSlot: Int

    public init(label: String, dispatchCount: Int, bufferSlot: Int) {
        self.label = label
        self.dispatchCount = dispatchCount
        self.bufferSlot = bufferSlot
    }
}
