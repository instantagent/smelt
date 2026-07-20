import CryptoKit
import Foundation
import SmeltSchema

/// Package-authored sampling defaults associated with a prepared prompt
/// contract. Request fields still win; these values fill only omitted fields.
public struct SmeltPreparedPromptSampling: Codable, Equatable, Sendable {
    public let temperature: Double?
    public let topK: Int?
    public let topP: Double?

    public init(
        temperature: Double? = nil,
        topK: Int? = nil,
        topP: Double? = nil
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
    }

    enum CodingKeys: String, CodingKey {
        case temperature
        case topK = "top_k"
        case topP = "top_p"
    }
}

/// One frozen prompt contract and the state produced after evaluating it.
/// Contract IDs are adapter vocabulary; matching and restoration are entirely
/// generic and always guarded by exact token-prefix identity.
public struct SmeltPreparedPromptState: Sendable {
    public let id: String
    public let tokenIds: [Int32]
    public let sampling: SmeltPreparedPromptSampling?
    public let snapshot: SmeltPromptSnapshot

    public init(
        id: String,
        tokenIds: [Int32],
        sampling: SmeltPreparedPromptSampling? = nil,
        snapshot: SmeltPromptSnapshot
    ) {
        self.id = id
        self.tokenIds = tokenIds
        self.sampling = sampling
        self.snapshot = snapshot
    }
}

public enum SmeltPreparedPromptError: Error, CustomStringConvertible, Equatable {
    case malformed(String)

    public var description: String {
        switch self {
        case .malformed(let detail):
            return "prepared prompt metadata is malformed: \(detail)"
        }
    }
}

/// A package can carry several prepared prompt states without creating
/// adapter-specific runtime paths. Selection is by contract ID when supplied,
/// then by longest exact token prefix.
public struct SmeltPreparedPromptSet: Sendable {
    public static let fileName = SmeltBakeArtifacts.preparedPromptsMeta

    struct Meta: Codable {
        let version: Int
        let entries: [Entry]
    }

    struct Entry: Codable {
        let id: String
        let tokenIds: [Int32]
        let snapshotFile: String
        let sampling: SmeltPreparedPromptSampling?

        enum CodingKeys: String, CodingKey {
            case id
            case tokenIds = "token_ids"
            case snapshotFile = "snapshot_file"
            case sampling
        }
    }

    public let states: [SmeltPreparedPromptState]

    public init(states: [SmeltPreparedPromptState]) {
        self.states = states
    }

    public func longestMatch(
        tokenIds: [Int32],
        contract: String? = nil
    ) -> SmeltPreparedPromptState? {
        states.lazy
            .filter { contract == nil || $0.id == contract }
            .filter {
                !$0.tokenIds.isEmpty
                    && tokenIds.count >= $0.tokenIds.count
                    && tokenIds.starts(with: $0.tokenIds)
            }
            .max { left, right in left.tokenIds.count < right.tokenIds.count }
    }

    public static func load(packagePath: String) throws -> Self? {
        let url = URL(fileURLWithPath: packagePath).appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: url))
        guard meta.version == 1, !meta.entries.isEmpty else {
            throw SmeltPreparedPromptError.malformed("bad version or empty entries")
        }
        var seen = Set<String>()
        var states: [SmeltPreparedPromptState] = []
        states.reserveCapacity(meta.entries.count)
        for entry in meta.entries {
            guard validID(entry.id), seen.insert(entry.id).inserted else {
                throw SmeltPreparedPromptError.malformed(
                    "invalid or duplicate id '\(entry.id)'"
                )
            }
            guard !entry.tokenIds.isEmpty,
                  validSnapshotFile(entry.snapshotFile)
            else {
                throw SmeltPreparedPromptError.malformed(
                    "invalid entry '\(entry.id)'"
                )
            }
            try validateSampling(entry.sampling, id: entry.id)
            let snapshot = try SmeltPromptSnapshot.read(
                from: URL(fileURLWithPath: packagePath)
                    .appendingPathComponent(entry.snapshotFile)
            )
            guard snapshot.promptLength == entry.tokenIds.count else {
                throw SmeltPreparedPromptError.malformed(
                    "snapshot length for '\(entry.id)' does not match token_ids"
                )
            }
            guard snapshot.capturedLength <= entry.tokenIds.count,
                  snapshot.replayTokenIds
                    == Array(entry.tokenIds.dropFirst(snapshot.capturedLength))
            else {
                throw SmeltPreparedPromptError.malformed(
                    "snapshot replay tail for '\(entry.id)' does not match token_ids"
                )
            }
            states.append(SmeltPreparedPromptState(
                id: entry.id,
                tokenIds: entry.tokenIds,
                sampling: entry.sampling,
                snapshot: snapshot
            ))
        }
        return Self(states: states)
    }

    /// Add or replace one named state. Snapshot filenames are content-stable by
    /// contract ID, while metadata replacement is atomic.
    @discardableResult
    public static func write(
        packagePath: String,
        state: SmeltPreparedPromptState
    ) throws -> SmeltPromptSnapshotWriteInfo {
        guard validID(state.id), !state.tokenIds.isEmpty,
              state.snapshot.promptLength == state.tokenIds.count,
              state.snapshot.capturedLength <= state.tokenIds.count,
              state.snapshot.replayTokenIds
                == Array(state.tokenIds.dropFirst(state.snapshot.capturedLength))
        else {
            throw SmeltPreparedPromptError.malformed("invalid state '\(state.id)'")
        }
        try validateSampling(state.sampling, id: state.id)
        let root = URL(fileURLWithPath: packagePath)
        let metaURL = root.appendingPathComponent(fileName)
        let existing: Meta
        if FileManager.default.fileExists(atPath: metaURL.path) {
            existing = try JSONDecoder().decode(
                Meta.self, from: Data(contentsOf: metaURL)
            )
            guard existing.version == 1 else {
                throw SmeltPreparedPromptError.malformed("unsupported version")
            }
        } else {
            existing = Meta(version: 1, entries: [])
        }

        let snapshotFile = snapshotFileName(for: state.id)
        let info = try state.snapshot.write(
            to: root.appendingPathComponent(snapshotFile)
        )
        let newEntry = Entry(
            id: state.id,
            tokenIds: state.tokenIds,
            snapshotFile: snapshotFile,
            sampling: state.sampling
        )
        var entries = existing.entries.filter { $0.id != state.id }
        entries.append(newEntry)
        entries.sort { $0.id < $1.id }
        let meta = Meta(version: 1, entries: entries)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(meta).write(to: metaURL, options: .atomic)
        return info
    }

    public static func declaredFiles(packagePath: String) throws -> [String] {
        let url = URL(fileURLWithPath: packagePath).appendingPathComponent(fileName)
        let meta = try JSONDecoder().decode(Meta.self, from: Data(contentsOf: url))
        return [fileName] + meta.entries.map(\.snapshotFile)
    }

    private static func snapshotFileName(for id: String) -> String {
        let digest = SHA256.hash(data: Data(id.utf8)).prefix(12).map {
            String(format: "%02x", $0)
        }.joined()
        return "prepared_prompt_\(digest).snapshot"
    }

    private static func validID(_ id: String) -> Bool {
        !id.isEmpty && id.utf8.count <= 128
            && id.unicodeScalars.allSatisfy {
                CharacterSet.alphanumerics.contains($0)
                    || $0 == "/" || $0 == "." || $0 == "_" || $0 == "-"
            }
    }

    private static func validSnapshotFile(_ file: String) -> Bool {
        file.hasPrefix("prepared_prompt_")
            && file.hasSuffix(".snapshot")
            && !file.contains("/") && !file.contains("\\")
    }

    private static func validateSampling(
        _ sampling: SmeltPreparedPromptSampling?,
        id: String
    ) throws {
        guard let sampling else { return }
        if let temperature = sampling.temperature,
           !temperature.isFinite || temperature < 0 {
            throw SmeltPreparedPromptError.malformed(
                "invalid sampling temperature for '\(id)'"
            )
        }
        if let topK = sampling.topK, topK <= 0 {
            throw SmeltPreparedPromptError.malformed(
                "invalid sampling top_k for '\(id)'"
            )
        }
        if let topP = sampling.topP,
           !topP.isFinite || topP <= 0 || topP > 1 {
            throw SmeltPreparedPromptError.malformed(
                "invalid sampling top_p for '\(id)'"
            )
        }
    }
}
