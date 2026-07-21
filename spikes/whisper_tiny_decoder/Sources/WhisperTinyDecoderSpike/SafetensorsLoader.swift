import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public struct SafetensorInfo {
    public let name: String
    public let dtype: String
    public let shape: [Int]
    public let dataStart: Int
    public let dataEnd: Int
    public let shardIndex: Int

    public var byteCount: Int { dataEnd - dataStart }
}

public struct SafetensorsLoader {
    public let tensors: [SafetensorInfo]
    private let shards: [MmapHandle]

    public init(paths: [String]) throws {
        var allTensors: [SafetensorInfo] = []
        var handles: [MmapHandle] = []

        for (shardIdx, path) in paths.enumerated() {
            let handle = try MmapHandle(path: path)
            guard handle.length >= 8 else {
                throw SafetensorsError.invalidHeader(path)
            }

            let headerLen = handle.readUInt64LE(at: 0)
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

            let headerData = Data(
                bytes: handle.pointer.advanced(by: headerStart),
                count: Int(headerLen)
            )
            guard let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
            else {
                throw SafetensorsError.invalidHeader(path)
            }

            let dataRegionStart = headerEnd
            for (name, value) in json {
                if name == "__metadata__" { continue }
                guard let info = value as? [String: Any],
                      let dtype = info["dtype"] as? String,
                      let shape = info["shape"] as? [Int],
                      let offsets = info["data_offsets"] as? [Int],
                      offsets.count == 2
                else {
                    continue
                }

                let absStart = dataRegionStart + offsets[0]
                let absEnd = dataRegionStart + offsets[1]
                guard absStart >= 0, absEnd >= absStart, absEnd <= handle.length else {
                    continue
                }

                let dtypeSize: Int
                switch dtype {
                case "F16", "BF16":
                    dtypeSize = 2
                case "F32":
                    dtypeSize = 4
                case "F64":
                    dtypeSize = 8
                case "I32", "U32":
                    dtypeSize = 4
                case "I64", "U64":
                    dtypeSize = 8
                case "I8", "U8", "BOOL":
                    dtypeSize = 1
                case "I16", "U16":
                    dtypeSize = 2
                default:
                    continue
                }

                var expectedBytes = dtypeSize
                var overflow = false
                for dim in shape {
                    let (result, didOverflow) = expectedBytes.multipliedReportingOverflow(by: dim)
                    if didOverflow {
                        overflow = true
                        break
                    }
                    expectedBytes = result
                }
                guard !overflow, absEnd - absStart == expectedBytes else {
                    continue
                }

                allTensors.append(
                    SafetensorInfo(
                        name: name,
                        dtype: dtype,
                        shape: shape,
                        dataStart: absStart,
                        dataEnd: absEnd,
                        shardIndex: shardIdx
                    )
                )
            }

            handles.append(handle)
        }

        self.tensors = allTensors
        self.shards = handles
    }

    public func tensorData(_ info: SafetensorInfo) -> UnsafeRawPointer {
        shards[info.shardIndex].pointer.advanced(by: info.dataStart)
    }

    public func tensor(named name: String) -> SafetensorInfo? {
        tensors.first { $0.name == name }
    }
}

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

        madvise(ptr, length, MADV_SEQUENTIAL)
        self.pointer = UnsafeRawPointer(ptr)
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: pointer), length)
    }

    func readUInt64LE(at offset: Int) -> UInt64 {
        pointer.advanced(by: offset).load(as: UInt64.self)
    }
}

public enum SafetensorsError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case invalidHeader(String)
    case mmapFailed(String)

    public var description: String {
        switch self {
        case let .fileNotFound(path):
            return "Safetensors file not found: \(path)"
        case let .invalidHeader(path):
            return "Invalid safetensors header: \(path)"
        case let .mmapFailed(path):
            return "Failed to mmap: \(path)"
        }
    }
}
