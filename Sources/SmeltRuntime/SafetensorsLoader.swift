// SafetensorsLoader — Minimal mmap-based safetensors reader.
//
// Safetensors format:
//   8 bytes: header length (uint64 LE)
//   N bytes: JSON header {tensor_name: {dtype, shape, data_offsets: [start, end]}}
//   Remaining: raw tensor data (mmap-able)
//
// Supports sharded models (model.safetensors.index.json → multiple files).
// No external dependencies.

import Foundation

/// One tensor's metadata from the safetensors header.
public struct SafetensorInfo {
    public let name: String
    public let dtype: String        // "F16", "BF16", "F32", etc.
    public let shape: [Int]
    public let dataStart: Int       // byte offset in data region
    public let dataEnd: Int         // byte offset end
    public let shardIndex: Int      // which shard file (0 for single file)

    public var byteCount: Int { dataEnd - dataStart }
}

/// Loads tensors from safetensors files via mmap.
public struct SafetensorsLoader {

    /// All tensor metadata across all shards.
    public let tensors: [SafetensorInfo]

    /// Shard mmap handles.
    private let shards: [MmapHandle]

    /// Load from a single file or directory with sharded files.
    public init(paths: [String]) throws {
        var allTensors: [SafetensorInfo] = []
        var handles: [MmapHandle] = []

        for (shardIdx, path) in paths.enumerated() {
            let handle = try MmapHandle(path: path)

            // File must be at least 8 bytes for the header length field
            guard handle.length >= 8 else {
                throw SafetensorsError.invalidHeader(path)
            }

            let headerLen = handle.readUInt64LE(at: 0)
            // Guard against absurdly large header or UInt64→Int overflow
            guard headerLen < 100_000_000,
                  headerLen <= UInt64(handle.length),
                  Int(headerLen) + 8 <= handle.length
            else {
                throw SafetensorsError.invalidHeader(path)
            }

            let headerStart = 8
            let headerEnd = headerStart + Int(headerLen)

            guard headerEnd <= handle.length else {
                throw SafetensorsError.invalidHeader(path)
            }

            // Parse JSON header
            let headerData = Data(
                bytes: handle.pointer.advanced(by: headerStart),
                count: Int(headerLen)
            )
            guard let json = try JSONSerialization.jsonObject(with: headerData)
                as? [String: Any]
            else {
                throw SafetensorsError.invalidHeader(path)
            }

            let dataRegionStart = headerEnd

            for (name, value) in json {
                // Skip __metadata__ key
                if name == "__metadata__" { continue }

                guard let info = value as? [String: Any],
                      let dtype = info["dtype"] as? String,
                      let shape = info["shape"] as? [Int],
                      let offsets = info["data_offsets"] as? [Int],
                      offsets.count == 2
                else { continue }

                let absStart = dataRegionStart + offsets[0]
                let absEnd = dataRegionStart + offsets[1]
                // Validate offsets within mapped file
                guard absStart >= 0, absEnd >= absStart, absEnd <= handle.length else {
                    continue
                }
                // Validate shape × dtype size matches declared byte span
                let dtypeSize: Int
                switch dtype {
                case "F16", "BF16": dtypeSize = 2
                case "F32": dtypeSize = 4
                case "F64": dtypeSize = 8
                case "I32", "U32": dtypeSize = 4
                case "I64", "U64": dtypeSize = 8
                case "I8", "U8", "BOOL": dtypeSize = 1
                case "I16", "U16": dtypeSize = 2
                default: continue  // Skip unsupported dtypes entirely
                }
                if dtypeSize > 0 {
                    // Checked multiplication to prevent Int overflow on huge shapes
                    var expectedBytes = dtypeSize
                    var overflow = false
                    for dim in shape {
                        let (result, didOverflow) = expectedBytes.multipliedReportingOverflow(by: dim)
                        if didOverflow { overflow = true; break }
                        expectedBytes = result
                    }
                    guard !overflow, absEnd - absStart == expectedBytes else { continue }
                }

                allTensors.append(SafetensorInfo(
                    name: name,
                    dtype: dtype,
                    shape: shape,
                    dataStart: absStart,
                    dataEnd: absEnd,
                    shardIndex: shardIdx
                ))
            }

            handles.append(handle)
        }

        self.tensors = allTensors
        self.shards = handles
    }

    /// Load from a model directory (auto-detects single vs sharded).
    public init(directory: String) throws {
        let fm = FileManager.default
        let indexPath = "\(directory)/model.safetensors.index.json"

        if fm.fileExists(atPath: indexPath) {
            // Sharded model
            let indexData = try Data(contentsOf: URL(fileURLWithPath: indexPath))
            guard let index = try JSONSerialization.jsonObject(with: indexData)
                as? [String: Any],
                  let weightMap = index["weight_map"] as? [String: String]
            else {
                throw SafetensorsError.invalidIndex(indexPath)
            }

            // Collect unique shard filenames in order
            var shardFiles: [String] = []
            var seen: Set<String> = []
            for file in weightMap.values.sorted() {
                if seen.insert(file).inserted {
                    shardFiles.append("\(directory)/\(file)")
                }
            }

            try self.init(paths: shardFiles)
        } else {
            // Single file
            let singlePath = "\(directory)/model.safetensors"
            guard fm.fileExists(atPath: singlePath) else {
                throw SafetensorsError.fileNotFound(singlePath)
            }
            try self.init(paths: [singlePath])
        }
    }

    /// Get a pointer to a tensor's raw data (mmap'd, zero-copy).
    public func tensorData(_ info: SafetensorInfo) -> UnsafeRawPointer {
        shards[info.shardIndex].pointer.advanced(by: info.dataStart)
    }

    /// Find a tensor by name.
    public func tensor(named name: String) -> SafetensorInfo? {
        tensors.first { $0.name == name }
    }
}

// MARK: - Mmap handle

private final class MmapHandle {
    let pointer: UnsafeRawPointer
    let length: Int

    init(path: String) throws {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw SafetensorsError.fileNotFound(path)
        }
        defer { close(fd) }

        let fileSize = lseek(fd, 0, SEEK_END)
        lseek(fd, 0, SEEK_SET)
        self.length = Int(fileSize)

        guard let ptr = mmap(nil, length, PROT_READ, MAP_PRIVATE, fd, 0),
              ptr != MAP_FAILED
        else {
            throw SafetensorsError.mmapFailed(path)
        }

        madvise(UnsafeMutableRawPointer(mutating: ptr), length, MADV_SEQUENTIAL)
        self.pointer = UnsafeRawPointer(ptr)
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: pointer), length)
    }

    func readUInt64LE(at offset: Int) -> UInt64 {
        pointer.advanced(by: offset).load(as: UInt64.self)
    }
}

// MARK: - Errors

public enum SafetensorsError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case invalidHeader(String)
    case invalidIndex(String)
    case mmapFailed(String)

    public var description: String {
        switch self {
        case let .fileNotFound(path): return "Safetensors file not found: \(path)"
        case let .invalidHeader(path): return "Invalid safetensors header: \(path)"
        case let .invalidIndex(path): return "Invalid safetensors index: \(path)"
        case let .mmapFailed(path): return "Failed to mmap: \(path)"
        }
    }
}
