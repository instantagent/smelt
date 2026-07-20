// HuggingFaceCache — Download and cache HuggingFace model files.
//
// Downloads safetensors, config.json, tokenizer files from HF Hub.
// Caches in ~/.cache/smelt/models/{model_id}/{revision}/.
// Supports standard Hugging Face token env vars, token files, and git credentials.

import Foundation

/// Resolved cache entry with paths to all model files.
public struct HFCacheEntry {
    public let directory: String
    public let safetensorsPaths: [String]
    public let configPath: String
    public let tokenizerPath: String?
    public let tokenizerConfigPath: String?
}

struct HFTokenizerAssets {
    let tokenizerPath: String?
    let tokenizerConfigPath: String?
}

/// Downloads and caches HuggingFace model files.
public struct HuggingFaceCache {

    private static let maxAttempts = 8

    /// Resolve only tokenizer-side assets. Packed-weight source builds use
    /// this path so a fresh package does not download or inspect model shards
    /// merely because its reusable weights directory is tokenizer-free.
    static func resolveTokenizerAssets(
        modelId: String,
        revision: String = "main"
    ) throws -> HFTokenizerAssets {
        let cacheDir = cacheDirectory(for: modelId, revision: revision)
        let fm = FileManager.default
        if !fm.fileExists(atPath: cacheDir) {
            try fm.createDirectory(atPath: cacheDir, withIntermediateDirectories: true)
        }

        let tokenizerPath = "\(cacheDir)/tokenizer.json"
        if !fm.fileExists(atPath: tokenizerPath) {
            _ = try? downloadFile(
                modelId: modelId,
                filename: "tokenizer.json",
                revision: revision,
                destination: tokenizerPath
            )
        }

        let tokenizerConfigPath = "\(cacheDir)/tokenizer_config.json"
        if !fm.fileExists(atPath: tokenizerConfigPath) {
            _ = try? downloadFile(
                modelId: modelId,
                filename: "tokenizer_config.json",
                revision: revision,
                destination: tokenizerConfigPath
            )
        }

        return HFTokenizerAssets(
            tokenizerPath: fm.fileExists(atPath: tokenizerPath) ? tokenizerPath : nil,
            tokenizerConfigPath: fm.fileExists(atPath: tokenizerConfigPath)
                ? tokenizerConfigPath
                : nil
        )
    }

    /// Resolve a model ID to local cache paths, downloading if needed.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model ID (e.g. "Qwen/Qwen3.5-2B").
    ///   - revision: Git revision (default "main").
    /// - Returns: Cache entry with local file paths.
    static func resolve(
        modelId: String,
        revision: String = "main",
        probe: HFCheckpointProbeResult? = nil,
        requiredTensors: [HFCheckpointTensorMetadata]? = nil
    ) throws -> HFCacheEntry {
        let cacheDir = cacheDirectory(for: modelId, revision: revision)
        let fm = FileManager.default

        if !fm.fileExists(atPath: cacheDir) {
            try fm.createDirectory(atPath: cacheDir, withIntermediateDirectories: true)
        }

        // Download config.json (small, always needed)
        let configPath = "\(cacheDir)/config.json"
        if !fm.fileExists(atPath: configPath) {
            try downloadFile(
                modelId: modelId, filename: "config.json",
                revision: revision, destination: configPath
            )
        }

        var safetensorsPaths: [String] = []

        if let subsetPath = try resolveSubsetIfPossible(
            modelId: modelId,
            revision: revision,
            cacheDir: cacheDir,
            probe: probe,
            requiredTensors: requiredTensors
        ) {
            safetensorsPaths = [subsetPath]
        } else {
            // Check for sharded vs single safetensors
            let indexPath = "\(cacheDir)/model.safetensors.index.json"

            // Try downloading index first
            let hasIndex: Bool
            if fm.fileExists(atPath: indexPath) {
                hasIndex = true
            } else {
                hasIndex = (try? downloadFile(
                    modelId: modelId,
                    filename: "model.safetensors.index.json",
                    revision: revision,
                    destination: indexPath
                )) != nil
            }

            if hasIndex {
                // Sharded model — parse index to find shard files
                let indexData = try Data(contentsOf: URL(fileURLWithPath: indexPath))
                guard let index = try JSONSerialization.jsonObject(with: indexData)
                    as? [String: Any],
                      let weightMap = index["weight_map"] as? [String: String]
                else {
                    throw HFCacheError.invalidIndex(modelId)
                }

                var shardFiles: Set<String> = []
                for file in weightMap.values { shardFiles.insert(file) }

                for file in shardFiles.sorted() {
                    let safeName = URL(fileURLWithPath: file).lastPathComponent
                    let shardPath = "\(cacheDir)/\(safeName)"
                    if !fm.fileExists(atPath: shardPath) {
                        try downloadFile(
                            modelId: modelId, filename: file,
                            revision: revision, destination: shardPath
                        )
                    }
                    safetensorsPaths.append(shardPath)
                }
            } else {
                let singlePath = "\(cacheDir)/model.safetensors"
                if !fm.fileExists(atPath: singlePath) {
                    try downloadFile(
                        modelId: modelId, filename: "model.safetensors",
                        revision: revision, destination: singlePath
                    )
                }
                safetensorsPaths = [singlePath]
            }
        }

        // Download tokenizer files (optional)
        let tokenizerPath = "\(cacheDir)/tokenizer.json"
        if !fm.fileExists(atPath: tokenizerPath) {
            _ = try? downloadFile(
                modelId: modelId, filename: "tokenizer.json",
                revision: revision, destination: tokenizerPath
            )
        }

        let tokenizerConfigPath = "\(cacheDir)/tokenizer_config.json"
        if !fm.fileExists(atPath: tokenizerConfigPath) {
            _ = try? downloadFile(
                modelId: modelId, filename: "tokenizer_config.json",
                revision: revision, destination: tokenizerConfigPath
            )
        }

        return HFCacheEntry(
            directory: cacheDir,
            safetensorsPaths: safetensorsPaths,
            configPath: configPath,
            tokenizerPath: fm.fileExists(atPath: tokenizerPath) ? tokenizerPath : nil,
            tokenizerConfigPath: fm.fileExists(atPath: tokenizerConfigPath)
                ? tokenizerConfigPath : nil
        )
    }

    private static func resolveSubsetIfPossible(
        modelId: String,
        revision: String,
        cacheDir: String,
        probe: HFCheckpointProbeResult?,
        requiredTensors: [HFCheckpointTensorMetadata]?
    ) throws -> String? {
        guard let probe, let requiredTensors, !requiredTensors.isEmpty else {
            return nil
        }
        let eligibleTensors = requiredTensors.filter {
            $0.filename == "model.safetensors"
                && $0.fileOffsetStart != nil
                && $0.fileOffsetEnd != nil
                && $0.dtype != nil
                && $0.shape != nil
        }
        guard eligibleTensors.count == requiredTensors.count,
              probe.tensors.contains(where: { $0.filename == "model.safetensors" })
        else {
            return nil
        }

        let subsetKey = stableSubsetKey(for: eligibleTensors.map(\.name))
        let subsetPath = "\(cacheDir)/text-\(subsetKey).safetensors"
        if FileManager.default.fileExists(atPath: subsetPath) {
            return subsetPath
        }

        do {
            try materializeSingleFileSubset(
                modelId: modelId,
                revision: revision,
                cacheDir: cacheDir,
                destination: subsetPath,
                tensors: eligibleTensors
            )
        } catch {
            guard shouldRetry(after: error) else {
                throw error
            }

            let fullPath = "\(cacheDir)/model.safetensors"
            if !FileManager.default.fileExists(atPath: fullPath) {
                fputs(
                    "  Range-based subset fetch failed; downloading full model.safetensors for local subsetting...\n",
                    stderr
                )
                try downloadFile(
                    modelId: modelId,
                    filename: "model.safetensors",
                    revision: revision,
                    destination: fullPath
                )
            }

            try materializeSingleFileSubset(
                modelId: modelId,
                revision: revision,
                cacheDir: cacheDir,
                destination: subsetPath,
                tensors: eligibleTensors
            )
        }
        return subsetPath
    }

    private static func materializeSingleFileSubset(
        modelId: String,
        revision: String,
        cacheDir: String,
        destination: String,
        tensors: [HFCheckpointTensorMetadata]
    ) throws {
        let ordered = tensors.sorted { lhs, rhs in
            let l = lhs.fileOffsetStart ?? .max
            let r = rhs.fileOffsetStart ?? .max
            if l == r { return lhs.name < rhs.name }
            return l < r
        }

        var header: [String: Any] = [:]
        var payloadOffset = 0
        for tensor in ordered {
            guard let dtype = tensor.dtype,
                  let shape = tensor.shape,
                  let start = tensor.fileOffsetStart,
                  let end = tensor.fileOffsetEnd
            else {
                throw HFCacheError.invalidSubset("missing tensor metadata for \(tensor.name)")
            }
            header[tensor.name] = [
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [payloadOffset, payloadOffset + (end - start)],
            ]
            payloadOffset += end - start
        }

        let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        let tempPath = destination + ".tmp"
        let fm = FileManager.default
        if fm.fileExists(atPath: tempPath) {
            try fm.removeItem(atPath: tempPath)
        }
        if fm.fileExists(atPath: destination) {
            try fm.removeItem(atPath: destination)
        }
        fm.createFile(atPath: tempPath, contents: nil)
        let outputURL = URL(fileURLWithPath: tempPath)
        let output = try FileHandle(forWritingTo: outputURL)
        defer { try? output.close() }

        var headerLengthLE = UInt64(headerData.count).littleEndian
        try withUnsafeBytes(of: &headerLengthLE) { bytes in
            try output.write(contentsOf: bytes)
        }
        try output.write(contentsOf: headerData)

        let localFullPath = "\(cacheDir)/model.safetensors"
        let localInput = fm.fileExists(atPath: localFullPath)
            ? try FileHandle(forReadingFrom: URL(fileURLWithPath: localFullPath))
            : nil
        defer { try? localInput?.close() }

        let chunkSize = 64 * 1024 * 1024
        for (idx, tensor) in ordered.enumerated() {
            guard let start = tensor.fileOffsetStart,
                  let end = tensor.fileOffsetEnd
            else {
                throw HFCacheError.invalidSubset("missing file offsets for \(tensor.name)")
            }

            var cursor = start
            while cursor < end {
                let nextEnd = min(cursor + chunkSize, end)
                let chunk: Data
                if let localInput {
                    try localInput.seek(toOffset: UInt64(cursor))
                    chunk = try localInput.read(upToCount: nextEnd - cursor) ?? Data()
                } else {
                    chunk = try fetchRangeData(
                        modelId: modelId,
                        revision: revision,
                        filename: tensor.filename,
                        byteRange: cursor...(nextEnd - 1)
                    )
                }

                guard chunk.count == nextEnd - cursor else {
                    throw HFCacheError.invalidSubset(
                        "range fetch for \(tensor.name) returned \(chunk.count) bytes, expected \(nextEnd - cursor)"
                    )
                }
                try output.write(contentsOf: chunk)
                cursor = nextEnd
            }

            if idx == 0 || (idx + 1) % 64 == 0 || idx + 1 == ordered.count {
                fputs("  Cached text subset: \(idx + 1)/\(ordered.count) tensors\r", stderr)
            }
        }
        fputs("\n", stderr)
        try output.synchronize()
        try fm.moveItem(atPath: tempPath, toPath: destination)

        let sizeMB = ((try? fm.attributesOfItem(atPath: destination)[.size] as? NSNumber)?
            .intValue ?? 0) / 1024 / 1024
        fputs("  Materialized text-only safetensors subset (\(sizeMB) MB)\n", stderr)
    }

    // MARK: - Cache directory

    private static func cacheDirectory(for modelId: String, revision: String) -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let safeId = modelId.replacingOccurrences(of: "/", with: "--")
            .replacingOccurrences(of: "..", with: "_")
        let safeRev = revision.replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "..", with: "_")
        return "\(home)/.cache/smelt/models/\(safeId)/\(safeRev)"
    }

    // MARK: - Download

    @discardableResult
    private static func downloadFile(
        modelId: String,
        filename: String,
        revision: String,
        destination: String
    ) throws -> Bool {
        let urlString = "https://huggingface.co/\(modelId)/resolve/\(revision)/\(filename)"
        guard let url = URL(string: urlString) else {
            throw HFCacheError.invalidURL(urlString)
        }

        var lastError: Error?
        for attempt in 1...maxAttempts {
            do {
                try downloadFileAttempt(url: url, urlString: urlString, destination: destination)
                let fm = FileManager.default
                let fileSizeBytes = (try? fm.attributesOfItem(atPath: destination)[.size] as? NSNumber)?
                    .intValue ?? 0
                fputs("  Downloaded: \(filename) (\(fileSizeBytes / 1024 / 1024) MB)\n", stderr)
                return true
            } catch {
                lastError = error
                if !shouldRetry(after: error) || attempt == maxAttempts {
                    throw error
                }
                Thread.sleep(forTimeInterval: retryDelay(forAttempt: attempt))
            }
        }
        throw lastError ?? HFCacheError.downloadFailed(urlString)
    }

    private static func fetchRangeData(
        modelId: String,
        revision: String,
        filename: String,
        byteRange: ClosedRange<Int>
    ) throws -> Data {
        let urlString = "https://huggingface.co/\(modelId)/resolve/\(revision)/\(filename)"
        guard let url = URL(string: urlString) else {
            throw HFCacheError.invalidURL(urlString)
        }

        var lastError: Error?
        for attempt in 1...maxAttempts {
            do {
                return try fetchRangeDataAttempt(url: url, urlString: urlString, byteRange: byteRange)
            } catch {
                lastError = error
                if !shouldRetry(after: error) || attempt == maxAttempts {
                    throw error
                }
                Thread.sleep(forTimeInterval: retryDelay(forAttempt: attempt))
            }
        }
        throw lastError ?? HFCacheError.downloadFailed(urlString)
    }

    private static func stableSubsetKey(for names: [String]) -> String {
        var hash: UInt64 = 14695981039346656037
        for byte in names.sorted().joined(separator: "\n").utf8 {
            hash ^= UInt64(byte)
            hash &*= 1099511628211
        }
        return String(format: "%016llx", hash)
    }

    private static func downloadFileAttempt(
        url: URL,
        urlString: String,
        destination: String
    ) throws {
        final class DownloadBox: @unchecked Sendable {
            var location: URL?
            var response: URLResponse?
            var error: Error?
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 7200

        HuggingFaceAuth.applyAuthorization(to: &request)

        let semaphore = DispatchSemaphore(value: 0)
        let downloadBox = DownloadBox()

        let task = URLSession.shared.downloadTask(with: request) { location, response, error in
            downloadBox.location = location
            downloadBox.response = response
            downloadBox.error = error
            semaphore.signal()
        }
        task.resume()
        semaphore.wait()

        if let error = downloadBox.error { throw error }
        guard let httpResponse = downloadBox.response as? HTTPURLResponse else {
            throw HFCacheError.downloadFailed(urlString)
        }
        guard httpResponse.statusCode == 200 else {
            throw HFCacheError.httpError(httpResponse.statusCode, urlString)
        }
        guard let location = downloadBox.location else {
            throw HFCacheError.downloadFailed(urlString)
        }

        let fm = FileManager.default
        let destinationURL = URL(fileURLWithPath: destination)
        if fm.fileExists(atPath: destination) {
            try fm.removeItem(at: destinationURL)
        }
        try fm.moveItem(at: location, to: destinationURL)
    }

    private static func fetchRangeDataAttempt(
        url: URL,
        urlString: String,
        byteRange: ClosedRange<Int>
    ) throws -> Data {
        final class ResponseBox: @unchecked Sendable {
            var data: Data?
            var response: URLResponse?
            var error: Error?
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 7200
        request.setValue(
            "bytes=\(byteRange.lowerBound)-\(byteRange.upperBound)",
            forHTTPHeaderField: "Range"
        )
        HuggingFaceAuth.applyAuthorization(to: &request)

        let semaphore = DispatchSemaphore(value: 0)
        let responseBox = ResponseBox()
        URLSession.shared.dataTask(with: request) { data, response, error in
            responseBox.data = data
            responseBox.response = response
            responseBox.error = error
            semaphore.signal()
        }.resume()
        semaphore.wait()

        if let error = responseBox.error { throw error }
        guard let httpResponse = responseBox.response as? HTTPURLResponse else {
            throw HFCacheError.downloadFailed(urlString)
        }
        guard [200, 206].contains(httpResponse.statusCode) else {
            throw HFCacheError.httpError(httpResponse.statusCode, urlString)
        }
        guard let data = responseBox.data else {
            throw HFCacheError.downloadFailed(urlString)
        }
        return data
    }

    private static func shouldRetry(after error: Error) -> Bool {
        if let cacheError = error as? HFCacheError,
           case let .httpError(code, _) = cacheError
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
}

// MARK: - Errors

public enum HFCacheError: Error, CustomStringConvertible {
    case invalidURL(String)
    case httpError(Int, String)
    case downloadFailed(String)
    case invalidIndex(String)
    case invalidSubset(String)

    public var description: String {
        switch self {
        case let .invalidURL(url): return "Invalid URL: \(url)"
        case let .httpError(code, url): return "HTTP \(code): \(url)"
        case let .downloadFailed(url): return "Download failed: \(url)"
        case let .invalidIndex(model): return "Invalid safetensors index for \(model)"
        case let .invalidSubset(message): return "Invalid text-only subset: \(message)"
        }
    }
}
