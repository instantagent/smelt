// HuggingFaceCheckpointProbe — Lightweight HF checkpoint preflight metadata fetch.
//
// Fetches config.json and enough checkpoint metadata to validate config and
// tensor-name coverage before downloading full safetensors weights.

import Foundation

public struct SmeltHFSafetensorInventory: Codable, Equatable, Sendable {
    public struct Count: Codable, Equatable, Sendable {
        public let name: String
        public let count: Int
    }

    public struct File: Codable, Equatable, Sendable {
        public let name: String
        public let tensorCount: Int
        public let tensorBytes: UInt64
    }

    public let schemaVersion: Int
    public let modelID: String
    public let revision: String
    public let tensorCount: Int
    public let tensorBytes: UInt64
    public let files: [File]
    public let dtypes: [Count]
    public let transferPolicy: String

    public func encodeJSON(prettyPrinted: Bool = true) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = prettyPrinted
            ? [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            : [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(self)
    }
}

/// Public, bounded metadata-only front door for checkpoint bring-up. It uses
/// the same probe as package construction but never accepts a full-body HTTP
/// response for a Range request.
public enum SmeltHFCheckpointInventoryProbe {
    public static func probe(
        modelID: String,
        revision: String = "main"
    ) throws -> SmeltHFSafetensorInventory {
        summarize(
            modelID: modelID,
            revision: revision,
            tensors: try HuggingFaceCheckpointProbe.probe(
                modelId: modelID,
                revision: revision
            ).tensors
        )
    }

    static func summarize(
        modelID: String,
        revision: String,
        tensors: [HFCheckpointTensorMetadata]
    ) -> SmeltHFSafetensorInventory {
        let byFile = Dictionary(grouping: tensors, by: \.filename)
        let files = byFile.map { filename, values in
            SmeltHFSafetensorInventory.File(
                name: filename,
                tensorCount: values.count,
                tensorBytes: values.reduce(UInt64(0)) { total, tensor in
                    guard let start = tensor.fileOffsetStart,
                          let end = tensor.fileOffsetEnd,
                          end >= start else { return total }
                    return total + UInt64(end - start)
                }
            )
        }.sorted { $0.name < $1.name }
        let dtypeCounts = Dictionary(grouping: tensors, by: { $0.dtype ?? "unknown" })
            .map { SmeltHFSafetensorInventory.Count(name: $0.key, count: $0.value.count) }
            .sorted { $0.name < $1.name }
        return SmeltHFSafetensorInventory(
            schemaVersion: 1,
            modelID: modelID,
            revision: revision,
            tensorCount: tensors.count,
            tensorBytes: files.reduce(UInt64(0)) { $0 + $1.tensorBytes },
            files: files,
            dtypes: dtypeCounts,
            transferPolicy: "safetensors-prefix-and-json-header-ranges-only"
        )
    }
}

struct HFCheckpointTensorMetadata {
    let name: String
    let dtype: String?
    let shape: [Int]?
    let filename: String
    let fileOffsetStart: Int?
    let fileOffsetEnd: Int?
}

struct HFCheckpointProbeResult {
    let config: [String: Any]
    let tensors: [HFCheckpointTensorMetadata]
}

enum HuggingFaceCheckpointProbe {

    private static let maxAttempts = 4

    static func probe(
        modelId: String,
        revision: String = "main"
    ) throws -> HFCheckpointProbeResult {
        let config = try fetchJSONObject(
            modelId: modelId,
            revision: revision,
            filename: "config.json"
        )

        if let index = try fetchJSONObjectIfExists(
            modelId: modelId,
            revision: revision,
            filename: "model.safetensors.index.json"
        ),
           let weightMap = index["weight_map"] as? [String: String]
        {
            let tensors = try fetchIndexedSafetensorsMetadata(
                modelId: modelId,
                revision: revision,
                weightMap: weightMap
            )
            return HFCheckpointProbeResult(config: config, tensors: tensors)
        }

        let header = try fetchSafetensorsHeader(
            modelId: modelId,
            revision: revision,
            filename: "model.safetensors"
        )
        return HFCheckpointProbeResult(
            config: config,
            tensors: try parseSafetensorsHeader(header, filename: "model.safetensors")
        )
    }

    private static func fetchIndexedSafetensorsMetadata(
        modelId: String,
        revision: String,
        weightMap: [String: String]
    ) throws -> [HFCheckpointTensorMetadata] {
        let namesByFilename = Dictionary(grouping: weightMap.keys) { name in
            weightMap[name] ?? "model.safetensors"
        }
        var metadataByName: [String: HFCheckpointTensorMetadata] = [:]

        for filename in namesByFilename.keys.sorted() {
            let header = try fetchSafetensorsHeader(
                modelId: modelId,
                revision: revision,
                filename: filename
            )
            let shardMetadata = try parseSafetensorsHeader(header, filename: filename)
            let shardMetadataByName = Dictionary(uniqueKeysWithValues: shardMetadata.map {
                ($0.name, $0)
            })

            for name in (namesByFilename[filename] ?? []).sorted() {
                guard let metadata = shardMetadataByName[name] else {
                    throw HFCheckpointProbeError.invalidHeader(
                        "\(modelId) \(filename): index maps missing tensor '\(name)'"
                    )
                }
                metadataByName[name] = metadata
            }
        }

        return weightMap.keys.sorted().compactMap { metadataByName[$0] }
    }

    private static func fetchSafetensorsHeader(
        modelId: String,
        revision: String,
        filename: String
    ) throws -> Data {
        let prefix = try fetchData(
            modelId: modelId,
            revision: revision,
            filename: filename,
            byteRange: 0...7
        )
        guard prefix.count == 8 else {
            throw HFCheckpointProbeError.invalidHeader(
                "\(modelId) \(filename): expected 8-byte safetensors prefix, got \(prefix.count)"
            )
        }

        let headerLength = readUInt64LE(prefix)
        guard headerLength > 0, headerLength < 100_000_000 else {
            throw HFCheckpointProbeError.invalidHeader(
                "\(modelId) \(filename): invalid header length \(headerLength)"
            )
        }

        let headerEnd = Int(headerLength) + 7
        return try fetchData(
            modelId: modelId,
            revision: revision,
            filename: filename,
            byteRange: 0...headerEnd
        )
    }

    private static func parseSafetensorsHeader(
        _ data: Data,
        filename: String
    ) throws -> [HFCheckpointTensorMetadata] {
        guard data.count >= 8 else {
            throw HFCheckpointProbeError.invalidHeader("header response shorter than 8 bytes")
        }
        let headerLength = Int(readUInt64LE(data))
        guard headerLength > 0, data.count >= headerLength + 8 else {
            throw HFCheckpointProbeError.invalidHeader(
                "header response shorter than declared safetensors header"
            )
        }

        let headerData = data.subdata(in: 8..<(8 + headerLength))
        guard let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw HFCheckpointProbeError.invalidHeader("failed to decode safetensors header JSON")
        }

        let dataRegionStart = 8 + headerLength
        return json.compactMap { name, value in
            guard name != "__metadata__",
                  let info = value as? [String: Any]
            else {
                return nil
            }

            let dtype = info["dtype"] as? String
            let shape = info["shape"] as? [Int]
            let offsets = info["data_offsets"] as? [Int]
            let start = offsets?[safe: 0].map { dataRegionStart + $0 }
            let end = offsets?[safe: 1].map { dataRegionStart + $0 }
            return HFCheckpointTensorMetadata(
                name: name,
                dtype: dtype,
                shape: shape,
                filename: filename,
                fileOffsetStart: start,
                fileOffsetEnd: end
            )
        }
        .sorted { $0.name < $1.name }
    }

    private static func fetchJSONObject(
        modelId: String,
        revision: String,
        filename: String
    ) throws -> [String: Any] {
        guard let json = try fetchJSONObjectIfExists(
            modelId: modelId,
            revision: revision,
            filename: filename
        ) else {
            throw HFCheckpointProbeError.missingRemoteFile(
                "\(modelId)/resolve/\(revision)/\(filename)"
            )
        }
        return json
    }

    private static func fetchJSONObjectIfExists(
        modelId: String,
        revision: String,
        filename: String
    ) throws -> [String: Any]? {
        let data = try fetchDataIfExists(
            modelId: modelId,
            revision: revision,
            filename: filename
        )
        guard let data else { return nil }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw HFCheckpointProbeError.invalidJSON(
                "\(modelId)/resolve/\(revision)/\(filename)"
            )
        }
        return json
    }

    private static func fetchDataIfExists(
        modelId: String,
        revision: String,
        filename: String,
        byteRange: ClosedRange<Int>? = nil
    ) throws -> Data? {
        do {
            return try fetchData(
                modelId: modelId,
                revision: revision,
                filename: filename,
                byteRange: byteRange
            )
        } catch let HFCheckpointProbeError.httpError(code, _) where code == 404 {
            return nil
        }
    }

    private static func fetchData(
        modelId: String,
        revision: String,
        filename: String,
        byteRange: ClosedRange<Int>? = nil
    ) throws -> Data {
        let urlString = "https://huggingface.co/\(modelId)/resolve/\(revision)/\(filename)"
        guard let url = URL(string: urlString) else {
            throw HFCheckpointProbeError.invalidURL(urlString)
        }

        var lastError: Error?
        for attempt in 1...maxAttempts {
            do {
                return try fetchDataAttempt(url: url, urlString: urlString, byteRange: byteRange)
            } catch {
                lastError = error
                if !shouldRetry(after: error) || attempt == maxAttempts {
                    throw error
                }
                Thread.sleep(forTimeInterval: retryDelay(forAttempt: attempt))
            }
        }
        throw lastError ?? HFCheckpointProbeError.invalidResponse(urlString)
    }

    private static func fetchDataAttempt(
        url: URL,
        urlString: String,
        byteRange: ClosedRange<Int>?
    ) throws -> Data {
        var request = URLRequest(url: url)
        request.timeoutInterval = 120
        if let byteRange {
            request.setValue(
                "bytes=\(byteRange.lowerBound)-\(byteRange.upperBound)",
                forHTTPHeaderField: "Range"
            )
        }
        HuggingFaceAuth.applyAuthorization(to: &request)

        let allowedCodes: Set<Int> = byteRange == nil ? [200] : [206]
        let maximumBytes = byteRange.map {
            max($0.upperBound - $0.lowerBound + 1, 1)
        } ?? 100_000_000
        let receiver = HFCheckpointBoundedReceiver(
            urlString: urlString,
            allowedStatusCodes: allowedCodes,
            maximumBytes: maximumBytes
        )
        let session = URLSession(
            configuration: .ephemeral,
            delegate: receiver,
            delegateQueue: nil
        )
        session.dataTask(with: request).resume()
        receiver.wait()
        session.invalidateAndCancel()

        if let responseError = receiver.error {
            throw responseError
        }
        guard receiver.responseCode != nil else {
            throw HFCheckpointProbeError.invalidResponse(urlString)
        }
        return receiver.data
    }

    private static func shouldRetry(after error: Error) -> Bool {
        if let probeError = error as? HFCheckpointProbeError,
           case let .httpError(code, _) = probeError
        {
            return code == 408 || code == 429 || (500...599).contains(code)
        }
        if let urlError = error as? URLError {
            switch urlError.code {
            case .timedOut,
                 .cannotFindHost,
                 .cannotConnectToHost,
                 .dnsLookupFailed,
                 .networkConnectionLost,
                 .notConnectedToInternet,
                 .resourceUnavailable,
                 .badServerResponse:
                return true
            default:
                return false
            }
        }
        return false
    }

    private static func retryDelay(forAttempt attempt: Int) -> TimeInterval {
        min(pow(2.0, Double(attempt - 1)), 8.0)
    }

    private static func readUInt64LE(_ data: Data) -> UInt64 {
        precondition(data.count >= 8)
        var value: UInt64 = 0
        for (idx, byte) in data.prefix(8).enumerated() {
            value |= UInt64(byte) << UInt64(idx * 8)
        }
        return value
    }
}

private final class HFCheckpointBoundedReceiver: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let urlString: String
    private let allowedStatusCodes: Set<Int>
    private let maximumBytes: Int
    private let semaphore = DispatchSemaphore(value: 0)
    private var didFinish = false

    private(set) var data = Data()
    private(set) var responseCode: Int?
    private(set) var error: Error?

    init(urlString: String, allowedStatusCodes: Set<Int>, maximumBytes: Int) {
        self.urlString = urlString
        self.allowedStatusCodes = allowedStatusCodes
        self.maximumBytes = maximumBytes
    }

    func wait() {
        semaphore.wait()
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let response = response as? HTTPURLResponse else {
            error = HFCheckpointProbeError.invalidResponse(urlString)
            completionHandler(.cancel)
            return
        }
        responseCode = response.statusCode
        guard allowedStatusCodes.contains(response.statusCode) else {
            error = HFCheckpointProbeError.httpError(response.statusCode, urlString)
            completionHandler(.cancel)
            return
        }
        completionHandler(.allow)
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive chunk: Data
    ) {
        guard data.count <= maximumBytes - chunk.count else {
            error = HFCheckpointProbeError.invalidResponse(
                "\(urlString) exceeded bounded metadata response of \(maximumBytes) bytes"
            )
            dataTask.cancel()
            return
        }
        data.append(chunk)
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError completionError: Error?
    ) {
        if error == nil {
            error = completionError
        }
        guard !didFinish else { return }
        didFinish = true
        semaphore.signal()
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

enum HFCheckpointProbeError: Error, CustomStringConvertible {
    case invalidURL(String)
    case invalidResponse(String)
    case httpError(Int, String)
    case missingRemoteFile(String)
    case invalidJSON(String)
    case invalidHeader(String)

    var description: String {
        switch self {
        case let .invalidURL(url):
            return "Invalid URL: \(url)"
        case let .invalidResponse(url):
            return "Invalid response: \(url)"
        case let .httpError(code, url):
            return "HTTP \(code): \(url)"
        case let .missingRemoteFile(path):
            return "Missing remote file: \(path)"
        case let .invalidJSON(path):
            return "Invalid JSON: \(path)"
        case let .invalidHeader(message):
            return "Invalid safetensors header: \(message)"
        }
    }
}
