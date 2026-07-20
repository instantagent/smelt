// NpyLoader — Minimal validated .npy file reader.
//
// NumPy .npy format:
//   6 bytes: magic "\x93NUMPY"
//   1 byte: major version
//   1 byte: minor version
//   2 bytes (v1) or 4 bytes (v2): header length (little-endian)
//   N bytes: ASCII header dict (shape, dtype, fortran_order)
//   Remaining: raw tensor data
//
// We support: float16, float32, bfloat16, int32. C-order only.

import Foundation

/// Loaded .npy tensor data.
public struct NpyTensor {
    public let data: UnsafeRawPointer
    public let byteCount: Int
    public let shape: [Int]
    public let dtype: String        // "f2" (FP16), "f4" (FP32), "i4" (Int32)
    public let elementCount: Int

    /// Typed access for FP16 data.
    public var fp16Pointer: UnsafePointer<UInt16> {
        data.bindMemory(to: UInt16.self, capacity: elementCount)
    }

    /// Typed access for FP32 data.
    public var fp32Pointer: UnsafePointer<Float> {
        data.bindMemory(to: Float.self, capacity: elementCount)
    }
}

/// Loads .npy files via mmap (zero-copy for read-only access).
public struct NpyLoader {

    /// Load a .npy file and return typed tensor data.
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

        // mmap the file
        guard let ptr = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0),
              ptr != MAP_FAILED
        else {
            throw NpyError.mmapFailed(path)
        }

        let bytes = ptr.bindMemory(to: UInt8.self, capacity: fileSize)

        // Check magic: \x93NUMPY
        guard bytes[0] == 0x93,
              bytes[1] == 0x4E,  // N
              bytes[2] == 0x55,  // U
              bytes[3] == 0x4D,  // M
              bytes[4] == 0x50,  // P
              bytes[5] == 0x59   // Y
        else {
            munmap(ptr, fileSize)
            throw NpyError.invalidFormat(path, "bad magic bytes")
        }

        let majorVersion = bytes[6]
        let minorVersion = bytes[7]

        // Parse header length
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

        // Parse ASCII header
        let headerBytes = Data(bytes: ptr.advanced(by: headerStart), count: headerLen)
        guard let headerStr = String(data: headerBytes, encoding: .ascii) else {
            munmap(ptr, fileSize)
            throw NpyError.invalidFormat(path, "header not ASCII")
        }

        // Extract fortran_order
        if headerStr.contains("'fortran_order': True") {
            munmap(ptr, fileSize)
            throw NpyError.invalidFormat(path, "fortran_order not supported")
        }

        // Extract dtype
        let dtype = extractDtype(from: headerStr)
        guard let dtype else {
            munmap(ptr, fileSize)
            throw NpyError.invalidFormat(path, "cannot parse dtype")
        }

        // Extract shape
        let shape = extractShape(from: headerStr)
        guard !shape.isEmpty else {
            munmap(ptr, fileSize)
            throw NpyError.invalidFormat(path, "cannot parse shape")
        }

        let elementCount = shape.reduce(1, *)
        let dtypeSize: Int
        switch dtype {
        case "f2": dtypeSize = 2    // float16
        case "f4": dtypeSize = 4    // float32
        case "i4": dtypeSize = 4    // int32
        default:
            munmap(ptr, fileSize)
            throw NpyError.invalidFormat(path, "unsupported dtype '\(dtype)'")
        }

        let expectedBytes = elementCount * dtypeSize
        let actualBytes = fileSize - dataStart
        guard actualBytes >= expectedBytes else {
            munmap(ptr, fileSize)
            throw NpyError.invalidFormat(
                path, "data too small: need \(expectedBytes), got \(actualBytes)"
            )
        }

        let dataPtr = UnsafeRawPointer(ptr.advanced(by: dataStart))

        // Note: we don't munmap here — the caller holds a reference to the data.
        // The mapping stays alive until the process exits (acceptable for cached state).
        // For proper cleanup, wrap in a class with deinit (like SafetensorsLoader).

        return NpyTensor(
            data: dataPtr,
            byteCount: expectedBytes,
            shape: shape,
            dtype: dtype,
            elementCount: elementCount
        )
    }

    // MARK: - Header parsing

    private static func extractDtype(from header: String) -> String? {
        // Look for 'descr': '<f4' or '|f2' or similar
        guard let descrRange = header.range(of: "'descr'") else { return nil }
        let after = header[descrRange.upperBound...]
        // Find the dtype string between quotes after colon
        guard let openQuote = after.firstIndex(of: "'") else { return nil }
        let afterOpen = after[after.index(after: openQuote)...]
        guard let closeQuote = afterOpen.firstIndex(of: "'") else { return nil }
        var dtype = String(afterOpen[..<closeQuote])
        // Strip endian prefix (< or > or |)
        if dtype.hasPrefix("<") || dtype.hasPrefix(">") || dtype.hasPrefix("|") {
            dtype = String(dtype.dropFirst())
        }
        return dtype
    }

    private static func extractShape(from header: String) -> [Int] {
        // Look for 'shape': (N, M, ...) or 'shape': (N,)
        guard let shapeRange = header.range(of: "'shape'") else { return [] }
        let after = header[shapeRange.upperBound...]
        guard let openParen = after.firstIndex(of: "(") else { return [] }
        guard let closeParen = after.firstIndex(of: ")") else { return [] }
        let shapeStr = after[after.index(after: openParen)..<closeParen]
        let parts = shapeStr.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespaces)
        }
        return parts.compactMap { Int($0) }
    }
}

// MARK: - Errors

public enum NpyError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case invalidFormat(String, String)
    case mmapFailed(String)

    public var description: String {
        switch self {
        case let .fileNotFound(path): return ".npy file not found: \(path)"
        case let .invalidFormat(path, detail): return "Invalid .npy format (\(path)): \(detail)"
        case let .mmapFailed(path): return "Failed to mmap .npy: \(path)"
        }
    }
}
