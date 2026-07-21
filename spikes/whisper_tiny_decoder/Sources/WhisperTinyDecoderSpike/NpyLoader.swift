import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public struct NpyTensor {
    public let data: UnsafeRawPointer
    public let byteCount: Int
    public let shape: [Int]
    public let dtype: String
    public let elementCount: Int

    public var fp16Pointer: UnsafePointer<UInt16> {
        data.bindMemory(to: UInt16.self, capacity: elementCount)
    }

    public var fp32Pointer: UnsafePointer<Float> {
        data.bindMemory(to: Float.self, capacity: elementCount)
    }
}

public struct NpyLoader {
    public static func load(path: String) throws -> NpyTensor {
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw NpyError.fileNotFound(path)
        }
        defer { close(fd) }

        let fileSize = Int(lseek(fd, 0, SEEK_END))
        lseek(fd, 0, SEEK_SET)

        guard fileSize >= 10 else {
            throw NpyError.invalidFormat(path, "file too small")
        }

        guard let ptr = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0),
              ptr != MAP_FAILED
        else {
            throw NpyError.mmapFailed(path)
        }

        let bytes = ptr.bindMemory(to: UInt8.self, capacity: fileSize)

        guard bytes[0] == 0x93,
              bytes[1] == 0x4E,
              bytes[2] == 0x55,
              bytes[3] == 0x4D,
              bytes[4] == 0x50,
              bytes[5] == 0x59
        else {
            munmap(ptr, fileSize)
            throw NpyError.invalidFormat(path, "bad magic bytes")
        }

        let majorVersion = bytes[6]
        let minorVersion = bytes[7]

        let headerLen: Int
        let headerStart: Int
        if majorVersion == 1 {
            headerLen = Int(bytes[8]) | (Int(bytes[9]) << 8)
            headerStart = 10
        } else if majorVersion == 2 {
            guard fileSize >= 12 else {
                munmap(ptr, fileSize)
                throw NpyError.invalidFormat(path, "v2 file too small")
            }
            headerLen = Int(bytes[8])
                | (Int(bytes[9]) << 8)
                | (Int(bytes[10]) << 16)
                | (Int(bytes[11]) << 24)
            headerStart = 12
        } else {
            munmap(ptr, fileSize)
            throw NpyError.invalidFormat(path, "unsupported version \(majorVersion).\(minorVersion)")
        }

        let dataStart = headerStart + headerLen
        guard dataStart <= fileSize else {
            munmap(ptr, fileSize)
            throw NpyError.invalidFormat(path, "header extends past file")
        }

        let headerBytes = Data(bytes: ptr.advanced(by: headerStart), count: headerLen)
        guard let headerStr = String(data: headerBytes, encoding: .ascii) else {
            munmap(ptr, fileSize)
            throw NpyError.invalidFormat(path, "header not ASCII")
        }

        if headerStr.contains("'fortran_order': True") {
            munmap(ptr, fileSize)
            throw NpyError.invalidFormat(path, "fortran_order not supported")
        }

        guard let dtype = extractDtype(from: headerStr) else {
            munmap(ptr, fileSize)
            throw NpyError.invalidFormat(path, "cannot parse dtype")
        }

        let shape = extractShape(from: headerStr)
        guard !shape.isEmpty else {
            munmap(ptr, fileSize)
            throw NpyError.invalidFormat(path, "cannot parse shape")
        }

        let elementCount = shape.reduce(1, *)
        let dtypeSize: Int
        switch dtype {
        case "f2": dtypeSize = 2
        case "f4": dtypeSize = 4
        case "i4": dtypeSize = 4
        default:
            munmap(ptr, fileSize)
            throw NpyError.invalidFormat(path, "unsupported dtype '\(dtype)'")
        }

        let expectedBytes = elementCount * dtypeSize
        let actualBytes = fileSize - dataStart
        guard actualBytes >= expectedBytes else {
            munmap(ptr, fileSize)
            throw NpyError.invalidFormat(
                path,
                "data too small: need \(expectedBytes), got \(actualBytes)"
            )
        }

        return NpyTensor(
            data: UnsafeRawPointer(ptr.advanced(by: dataStart)),
            byteCount: expectedBytes,
            shape: shape,
            dtype: dtype,
            elementCount: elementCount
        )
    }

    private static func extractDtype(from header: String) -> String? {
        guard let descrRange = header.range(of: "'descr'") else { return nil }
        let after = header[descrRange.upperBound...]
        guard let openQuote = after.firstIndex(of: "'") else { return nil }
        let afterOpen = after[after.index(after: openQuote)...]
        guard let closeQuote = afterOpen.firstIndex(of: "'") else { return nil }
        var dtype = String(afterOpen[..<closeQuote])
        if dtype.hasPrefix("<") || dtype.hasPrefix(">") || dtype.hasPrefix("|") {
            dtype = String(dtype.dropFirst())
        }
        return dtype
    }

    private static func extractShape(from header: String) -> [Int] {
        guard let shapeRange = header.range(of: "'shape'") else { return [] }
        let after = header[shapeRange.upperBound...]
        guard let openParen = after.firstIndex(of: "("),
              let closeParen = after.firstIndex(of: ")")
        else {
            return []
        }
        let shapeStr = after[after.index(after: openParen)..<closeParen]
        return shapeStr
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .compactMap(Int.init)
    }
}

public enum NpyError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case invalidFormat(String, String)
    case mmapFailed(String)

    public var description: String {
        switch self {
        case let .fileNotFound(path):
            return ".npy file not found: \(path)"
        case let .invalidFormat(path, detail):
            return "Invalid .npy format (\(path)): \(detail)"
        case let .mmapFailed(path):
            return "Failed to mmap .npy: \(path)"
        }
    }
}
