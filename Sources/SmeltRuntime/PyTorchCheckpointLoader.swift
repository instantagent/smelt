// PyTorchCheckpointLoader — restricted, zero-copy reader for PyTorch ZIP
// state-dict checkpoints.
//
// PyTorch's current ZIP serialization stores an uncompressed protocol-2
// pickle plus one uncompressed file per storage. Rig checkpoints use this
// representation. This loader interprets only the small pickle opcode and
// reducer surface required to recover dense tensor metadata; it never imports
// Python, executes pickle globals, or materializes tensor payloads.

import Foundation

/// One dense tensor view recovered from a PyTorch ZIP state dictionary.
public struct PyTorchCheckpointTensorInfo: Sendable, Equatable {
    /// State-dict key.
    public let name: String
    /// Smelt checkpoint dtype spelling.
    public let dtype: String
    /// Logical tensor shape.
    public let shape: [Int]
    /// Logical tensor strides in elements.
    public let strides: [Int]
    /// Storage identifier inside the ZIP's data directory.
    public let storageKey: String
    /// Logical view offset in storage elements.
    public let storageOffset: Int
    /// Logical dense tensor byte count.
    public let byteCount: Int
}

/// Fail-loud errors from restricted PyTorch checkpoint ingestion.
public enum PyTorchCheckpointError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case mmapFailed(String)
    case invalidZip(String)
    case unsupportedZipFeature(String)
    case missingArchiveEntry(String)
    case duplicateArchiveEntry(String)
    case invalidPickle(String)
    case unsupportedPickleOpcode(UInt8, Int)
    case unsupportedStorageType(String)
    case duplicateTensor(String)
    case missingStorage(String, tensor: String)
    case invalidTensor(String, reason: String)

    public var description: String {
        switch self {
        case let .fileNotFound(path):
            return "PyTorch checkpoint not found: \(path)"
        case let .mmapFailed(path):
            return "Failed to mmap PyTorch checkpoint: \(path)"
        case let .invalidZip(reason):
            return "Invalid PyTorch ZIP checkpoint: \(reason)"
        case let .unsupportedZipFeature(reason):
            return "Unsupported PyTorch ZIP feature: \(reason)"
        case let .missingArchiveEntry(name):
            return "PyTorch checkpoint is missing archive entry: \(name)"
        case let .duplicateArchiveEntry(name):
            return "PyTorch checkpoint contains duplicate archive entry: \(name)"
        case let .invalidPickle(reason):
            return "Invalid restricted PyTorch pickle: \(reason)"
        case let .unsupportedPickleOpcode(opcode, offset):
            return String(
                format: "Unsupported PyTorch pickle opcode 0x%02x at byte %d",
                opcode,
                offset
            )
        case let .unsupportedStorageType(name):
            return "Unsupported PyTorch storage type: \(name)"
        case let .duplicateTensor(name):
            return "PyTorch state dictionary contains duplicate tensor: \(name)"
        case let .missingStorage(key, tensor):
            return "PyTorch tensor \(tensor) references missing storage \(key)"
        case let .invalidTensor(name, reason):
            return "Invalid PyTorch tensor \(name): \(reason)"
        }
    }
}

/// Mmap-backed, zero-copy source for uncompressed PyTorch ZIP checkpoints.
public final class PyTorchCheckpointLoader: CheckpointTensorSource {
    private struct ResolvedTensor {
        let descriptor: CheckpointTensorDescriptor
        let info: PyTorchCheckpointTensorInfo
        let dataOffset: Int
    }

    /// Recovered state-dict tensor views in authored order.
    public let tensors: [PyTorchCheckpointTensorInfo]

    /// PyTorch state dictionaries carry their model-declared norm semantics.
    public let normWeightSemantics: CheckpointNormWeightSemantics = .modelDeclared

    private let mapping: PyTorchCheckpointMmap
    private let resolved: [ResolvedTensor]

    /// Opens and validates an uncompressed PyTorch ZIP checkpoint.
    public init(path: String) throws {
        let mapping = try PyTorchCheckpointMmap(path: path)
        let entries = try PyTorchZipDirectory(mapping: mapping).entries
        let byName = Dictionary(grouping: entries, by: \.name)
        if let duplicate = byName.first(where: { $0.value.count != 1 })?.key {
            throw PyTorchCheckpointError.duplicateArchiveEntry(duplicate)
        }
        let unique = byName.compactMapValues(\.first)
        let pickleEntries = entries.filter { $0.name.hasSuffix("/data.pkl") }
        guard pickleEntries.count == 1, let pickleEntry = pickleEntries.first else {
            throw PyTorchCheckpointError.invalidZip(
                "expected exactly one */data.pkl entry, found \(pickleEntries.count)"
            )
        }
        let root = String(pickleEntry.name.dropLast("data.pkl".count))
        guard let byteOrder = unique["\(root)byteorder"] else {
            throw PyTorchCheckpointError.missingArchiveEntry("\(root)byteorder")
        }
        let byteOrderValue = mapping.string(at: byteOrder.dataOffset, count: byteOrder.byteCount)
        guard byteOrderValue == "little" else {
            throw PyTorchCheckpointError.unsupportedZipFeature(
                "checkpoint byteorder is \(byteOrderValue), expected little"
            )
        }

        let pickle = Data(
            bytes: mapping.pointer.advanced(by: pickleEntry.dataOffset),
            count: pickleEntry.byteCount
        )
        let parsed = try PyTorchRestrictedPickle.parse(pickle)
        var names = Set<String>()
        var resolved: [ResolvedTensor] = []
        resolved.reserveCapacity(parsed.count)

        for (index, tensor) in parsed.enumerated() {
            guard names.insert(tensor.name).inserted else {
                throw PyTorchCheckpointError.duplicateTensor(tensor.name)
            }
            guard tensor.storageOffset >= 0 else {
                throw PyTorchCheckpointError.invalidTensor(
                    tensor.name,
                    reason: "negative storage offset"
                )
            }
            guard tensor.shape.allSatisfy({ $0 >= 0 }),
                  tensor.shape.count == tensor.strides.count
            else {
                throw PyTorchCheckpointError.invalidTensor(
                    tensor.name,
                    reason: "invalid shape/stride rank"
                )
            }
            let expectedStrides = Self.contiguousStrides(shape: tensor.shape)
            guard tensor.strides == expectedStrides else {
                throw PyTorchCheckpointError.invalidTensor(
                    tensor.name,
                    reason: "non-contiguous view strides \(tensor.strides)"
                )
            }
            let elementSize = try Self.elementSize(dtype: tensor.dtype)
            let elementCount = try Self.elementCount(
                shape: tensor.shape,
                tensorName: tensor.name
            )
            let byteCount = try Self.checkedMultiply(
                elementCount,
                elementSize,
                tensorName: tensor.name,
                role: "byte count"
            )
            let storageByteOffset = try Self.checkedMultiply(
                tensor.storageOffset,
                elementSize,
                tensorName: tensor.name,
                role: "storage byte offset"
            )
            let storageByteCount = try Self.checkedMultiply(
                tensor.storageElementCount,
                elementSize,
                tensorName: tensor.name,
                role: "storage byte count"
            )
            let storageName = "\(root)data/\(tensor.storageKey)"
            guard let storage = unique[storageName] else {
                throw PyTorchCheckpointError.missingStorage(
                    tensor.storageKey,
                    tensor: tensor.name
                )
            }
            guard storage.byteCount == storageByteCount,
                  storageByteOffset <= storage.byteCount,
                  byteCount <= storage.byteCount - storageByteOffset
            else {
                throw PyTorchCheckpointError.invalidTensor(
                    tensor.name,
                    reason: "view exceeds storage \(tensor.storageKey)"
                )
            }
            let info = PyTorchCheckpointTensorInfo(
                name: tensor.name,
                dtype: tensor.dtype,
                shape: tensor.shape,
                strides: tensor.strides,
                storageKey: tensor.storageKey,
                storageOffset: tensor.storageOffset,
                byteCount: byteCount
            )
            let descriptor = CheckpointTensorDescriptor(
                index: index,
                name: tensor.name,
                dtype: tensor.dtype,
                shape: tensor.shape,
                byteCount: byteCount
            )
            resolved.append(
                ResolvedTensor(
                    descriptor: descriptor,
                    info: info,
                    dataOffset: storage.dataOffset + storageByteOffset
                )
            )
        }
        guard !resolved.isEmpty else {
            throw PyTorchCheckpointError.invalidPickle("state dictionary contains no tensors")
        }
        self.mapping = mapping
        self.resolved = resolved
        tensors = resolved.map(\.info)
    }

    /// Common source descriptors consumed by Smelt package builders.
    public var checkpointTensors: [CheckpointTensorDescriptor] {
        resolved.map(\.descriptor)
    }

    /// Returns a zero-copy pointer into the mapped storage entry.
    public func checkpointTensorData(
        _ descriptor: CheckpointTensorDescriptor
    ) -> UnsafeRawPointer {
        precondition(
            descriptor.index >= 0 && descriptor.index < resolved.count,
            "descriptor does not belong to this PyTorch checkpoint"
        )
        let tensor = resolved[descriptor.index]
        precondition(
            tensor.descriptor.name == descriptor.name,
            "descriptor does not match its PyTorch checkpoint index"
        )
        return mapping.pointer.advanced(by: tensor.dataOffset)
    }

    private static func elementSize(dtype: String) throws -> Int {
        switch dtype {
        case "BF16", "F16", "I16", "U16": return 2
        case "F32", "I32", "U32": return 4
        case "F64", "I64", "U64": return 8
        case "I8", "U8", "BOOL": return 1
        default: throw PyTorchCheckpointError.unsupportedStorageType(dtype)
        }
    }

    private static func elementCount(shape: [Int], tensorName: String) throws -> Int {
        var count = 1
        for dimension in shape {
            count = try checkedMultiply(
                count,
                dimension,
                tensorName: tensorName,
                role: "element count"
            )
        }
        return count
    }

    private static func checkedMultiply(
        _ lhs: Int,
        _ rhs: Int,
        tensorName: String,
        role: String
    ) throws -> Int {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else {
            throw PyTorchCheckpointError.invalidTensor(
                tensorName,
                reason: "\(role) overflow"
            )
        }
        return value
    }

    private static func contiguousStrides(shape: [Int]) -> [Int] {
        guard !shape.isEmpty else { return [] }
        var strides = [Int](repeating: 1, count: shape.count)
        if shape.count > 1 {
            for index in stride(from: shape.count - 2, through: 0, by: -1) {
                strides[index] = strides[index + 1] * shape[index + 1]
            }
        }
        return strides
    }
}

private final class PyTorchCheckpointMmap {
    let pointer: UnsafeRawPointer
    let length: Int

    init(path: String) throws {
        let descriptor = open(path, O_RDONLY)
        guard descriptor >= 0 else {
            throw PyTorchCheckpointError.fileNotFound(path)
        }
        defer { close(descriptor) }
        let fileSize = lseek(descriptor, 0, SEEK_END)
        guard fileSize > 0, fileSize <= Int.max else {
            throw PyTorchCheckpointError.invalidZip("invalid file size")
        }
        length = Int(fileSize)
        guard let mapped = mmap(nil, length, PROT_READ, MAP_PRIVATE, descriptor, 0),
              mapped != MAP_FAILED
        else {
            throw PyTorchCheckpointError.mmapFailed(path)
        }
        pointer = UnsafeRawPointer(mapped)
        madvise(UnsafeMutableRawPointer(mutating: mapped), length, MADV_SEQUENTIAL)
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: pointer), length)
    }

    func byte(at offset: Int) -> UInt8 {
        pointer.load(fromByteOffset: offset, as: UInt8.self)
    }

    func uint16(at offset: Int) -> UInt16 {
        UInt16(byte(at: offset)) | (UInt16(byte(at: offset + 1)) << 8)
    }

    func uint32(at offset: Int) -> UInt32 {
        UInt32(byte(at: offset))
            | (UInt32(byte(at: offset + 1)) << 8)
            | (UInt32(byte(at: offset + 2)) << 16)
            | (UInt32(byte(at: offset + 3)) << 24)
    }

    func string(at offset: Int, count: Int) -> String {
        let bytes = UnsafeBufferPointer(
            start: pointer.advanced(by: offset).assumingMemoryBound(to: UInt8.self),
            count: count
        )
        return String(decoding: bytes, as: UTF8.self)
    }
}

private struct PyTorchZipEntry {
    let name: String
    let dataOffset: Int
    let byteCount: Int
}

private struct PyTorchZipDirectory {
    let entries: [PyTorchZipEntry]

    init(mapping: PyTorchCheckpointMmap) throws {
        let minimumEOCD = 22
        guard mapping.length >= minimumEOCD else {
            throw PyTorchCheckpointError.invalidZip("file is smaller than EOCD")
        }
        let firstCandidate = max(0, mapping.length - minimumEOCD - 65_535)
        var eocd: Int?
        var cursor = mapping.length - minimumEOCD
        while cursor >= firstCandidate {
            if mapping.uint32(at: cursor) == 0x0605_4b50 {
                eocd = cursor
                break
            }
            cursor -= 1
        }
        guard let eocd else {
            throw PyTorchCheckpointError.invalidZip("EOCD signature not found")
        }
        guard mapping.uint16(at: eocd + 4) == 0,
              mapping.uint16(at: eocd + 6) == 0
        else {
            throw PyTorchCheckpointError.unsupportedZipFeature("multi-disk archive")
        }
        let entryCount = Int(mapping.uint16(at: eocd + 10))
        let centralSize = Int(mapping.uint32(at: eocd + 12))
        let centralOffset = Int(mapping.uint32(at: eocd + 16))
        guard entryCount != 0xffff,
              centralOffset != Int(UInt32.max),
              centralSize != Int(UInt32.max)
        else {
            throw PyTorchCheckpointError.unsupportedZipFeature("ZIP64 archive")
        }
        guard centralOffset >= 0,
              centralSize >= 0,
              centralOffset <= mapping.length,
              centralSize <= mapping.length - centralOffset
        else {
            throw PyTorchCheckpointError.invalidZip("central directory exceeds file")
        }

        var entries: [PyTorchZipEntry] = []
        entries.reserveCapacity(entryCount)
        cursor = centralOffset
        for _ in 0..<entryCount {
            guard cursor <= mapping.length - 46,
                  mapping.uint32(at: cursor) == 0x0201_4b50
            else {
                throw PyTorchCheckpointError.invalidZip("bad central-directory entry")
            }
            let flags = mapping.uint16(at: cursor + 8)
            let compression = mapping.uint16(at: cursor + 10)
            let compressedSize = Int(mapping.uint32(at: cursor + 20))
            let byteCount = Int(mapping.uint32(at: cursor + 24))
            let nameLength = Int(mapping.uint16(at: cursor + 28))
            let extraLength = Int(mapping.uint16(at: cursor + 30))
            let commentLength = Int(mapping.uint16(at: cursor + 32))
            let localOffset = Int(mapping.uint32(at: cursor + 42))
            guard flags & 1 == 0 else {
                throw PyTorchCheckpointError.unsupportedZipFeature("encrypted entry")
            }
            guard compression == 0, compressedSize == byteCount else {
                throw PyTorchCheckpointError.unsupportedZipFeature(
                    "compressed entry at central offset \(cursor)"
                )
            }
            let variableLength = nameLength + extraLength + commentLength
            guard cursor + 46 <= mapping.length,
                  variableLength <= mapping.length - cursor - 46
            else {
                throw PyTorchCheckpointError.invalidZip("truncated central entry")
            }
            let name = mapping.string(at: cursor + 46, count: nameLength)
            guard localOffset <= mapping.length - 30,
                  mapping.uint32(at: localOffset) == 0x0403_4b50
            else {
                throw PyTorchCheckpointError.invalidZip("bad local header for \(name)")
            }
            let localNameLength = Int(mapping.uint16(at: localOffset + 26))
            let localExtraLength = Int(mapping.uint16(at: localOffset + 28))
            let dataOffset = localOffset + 30 + localNameLength + localExtraLength
            guard dataOffset >= 0,
                  dataOffset <= mapping.length,
                  byteCount <= mapping.length - dataOffset
            else {
                throw PyTorchCheckpointError.invalidZip("entry payload exceeds file: \(name)")
            }
            entries.append(
                PyTorchZipEntry(name: name, dataOffset: dataOffset, byteCount: byteCount)
            )
            cursor += 46 + variableLength
        }
        self.entries = entries
    }
}

private struct PyTorchParsedTensor {
    let name: String
    let dtype: String
    let storageKey: String
    let storageElementCount: Int
    let storageOffset: Int
    let shape: [Int]
    let strides: [Int]
}

private enum PyTorchRestrictedPickle {
    private struct Storage {
        let dtype: String
        let key: String
        let elementCount: Int
    }

    private indirect enum Value {
        case marker
        case string(String)
        case integer(Int)
        case boolean(Bool)
        case none
        case global(module: String, name: String)
        case tuple([Value])
        case list([Value])
        case dictionary
        case storage(Storage)
        case tensor(PyTorchParsedTensor)
        case opaque
    }

    static func parse(_ data: Data) throws -> [PyTorchParsedTensor] {
        let bytes = [UInt8](data)
        var cursor = 0
        var stack: [Value] = []
        var memo: [Int: Value] = [:]
        var tensors: [PyTorchParsedTensor] = []
        var stopped = false

        func require(_ count: Int) throws {
            guard count >= 0, cursor <= bytes.count, count <= bytes.count - cursor else {
                throw PyTorchCheckpointError.invalidPickle("truncated opcode at \(cursor)")
            }
        }
        func readByte() throws -> UInt8 {
            try require(1)
            defer { cursor += 1 }
            return bytes[cursor]
        }
        func readUInt16() throws -> UInt16 {
            try require(2)
            let value = UInt16(bytes[cursor]) | (UInt16(bytes[cursor + 1]) << 8)
            cursor += 2
            return value
        }
        func readUInt32() throws -> UInt32 {
            try require(4)
            let value = UInt32(bytes[cursor])
                | (UInt32(bytes[cursor + 1]) << 8)
                | (UInt32(bytes[cursor + 2]) << 16)
                | (UInt32(bytes[cursor + 3]) << 24)
            cursor += 4
            return value
        }
        func readString(count: Int) throws -> String {
            try require(count)
            defer { cursor += count }
            return String(decoding: bytes[cursor..<(cursor + count)], as: UTF8.self)
        }
        func readLine() throws -> String {
            guard let end = bytes[cursor...].firstIndex(of: 0x0a) else {
                throw PyTorchCheckpointError.invalidPickle("unterminated line at \(cursor)")
            }
            let value = String(decoding: bytes[cursor..<end], as: UTF8.self)
            cursor = end + 1
            return value
        }
        func pop() throws -> Value {
            guard let value = stack.popLast() else {
                throw PyTorchCheckpointError.invalidPickle("stack underflow at \(cursor)")
            }
            return value
        }
        func valuesAfterMark() throws -> [Value] {
            guard let mark = stack.lastIndex(where: {
                if case .marker = $0 { return true }
                return false
            }) else {
                throw PyTorchCheckpointError.invalidPickle("missing MARK at \(cursor)")
            }
            let values = Array(stack[(mark + 1)...])
            stack.removeSubrange(mark...)
            return values
        }
        func integer(_ value: Value, role: String) throws -> Int {
            guard case let .integer(result) = value else {
                throw PyTorchCheckpointError.invalidPickle("\(role) is not an integer")
            }
            return result
        }
        func string(_ value: Value, role: String) throws -> String {
            guard case let .string(result) = value else {
                throw PyTorchCheckpointError.invalidPickle("\(role) is not a string")
            }
            return result
        }
        func integers(_ value: Value, role: String) throws -> [Int] {
            guard case let .tuple(values) = value else {
                throw PyTorchCheckpointError.invalidPickle("\(role) is not a tuple")
            }
            return try values.map { try integer($0, role: role) }
        }
        func capture(_ value: Value) {
            guard case let .tensor(tensor) = value,
                  let candidate = stack.last,
                  case let .string(name) = candidate
            else { return }
            tensors.append(
                PyTorchParsedTensor(
                    name: name,
                    dtype: tensor.dtype,
                    storageKey: tensor.storageKey,
                    storageElementCount: tensor.storageElementCount,
                    storageOffset: tensor.storageOffset,
                    shape: tensor.shape,
                    strides: tensor.strides
                )
            )
        }

        while cursor < bytes.count, !stopped {
            let opcodeOffset = cursor
            let opcode = try readByte()
            switch opcode {
            case 0x80: // PROTO
                let version = try readByte()
                guard version == 2 else {
                    throw PyTorchCheckpointError.invalidPickle(
                        "unsupported protocol \(version)"
                    )
                }
            case 0x7d: // EMPTY_DICT
                stack.append(.dictionary)
            case 0x5d: // EMPTY_LIST
                stack.append(.list([]))
            case 0x29: // EMPTY_TUPLE
                stack.append(.tuple([]))
            case 0x28: // MARK
                stack.append(.marker)
            case 0x58: // BINUNICODE
                let count = Int(try readUInt32())
                stack.append(.string(try readString(count: count)))
            case 0x63: // GLOBAL
                stack.append(.global(module: try readLine(), name: try readLine()))
            case 0x4b: // BININT1
                stack.append(.integer(Int(try readByte())))
            case 0x4d: // BININT2
                stack.append(.integer(Int(try readUInt16())))
            case 0x4a: // BININT
                stack.append(.integer(Int(Int32(bitPattern: try readUInt32()))))
            case 0x47: // BINFLOAT
                try require(8)
                cursor += 8
                stack.append(.opaque)
            case 0x89: // NEWFALSE
                stack.append(.boolean(false))
            case 0x88: // NEWTRUE
                stack.append(.boolean(true))
            case 0x4e: // NONE
                stack.append(.none)
            case 0x71: // BINPUT
                let index = Int(try readByte())
                guard let value = stack.last else {
                    throw PyTorchCheckpointError.invalidPickle("BINPUT stack underflow")
                }
                memo[index] = value
            case 0x72: // LONG_BINPUT
                let index = Int(try readUInt32())
                guard let value = stack.last else {
                    throw PyTorchCheckpointError.invalidPickle("LONG_BINPUT stack underflow")
                }
                memo[index] = value
            case 0x68: // BINGET
                let index = Int(try readByte())
                guard let value = memo[index] else {
                    throw PyTorchCheckpointError.invalidPickle("unknown memo \(index)")
                }
                capture(value)
                stack.append(value)
            case 0x6a: // LONG_BINGET
                let index = Int(try readUInt32())
                guard let value = memo[index] else {
                    throw PyTorchCheckpointError.invalidPickle("unknown memo \(index)")
                }
                capture(value)
                stack.append(value)
            case 0x74: // TUPLE
                stack.append(.tuple(try valuesAfterMark()))
            case 0x85: // TUPLE1
                stack.append(.tuple([try pop()]))
            case 0x86: // TUPLE2
                let second = try pop()
                let first = try pop()
                stack.append(.tuple([first, second]))
            case 0x87: // TUPLE3
                let third = try pop()
                let second = try pop()
                let first = try pop()
                stack.append(.tuple([first, second, third]))
            case 0x51: // BINPERSID
                let persistent = try pop()
                guard case let .tuple(values) = persistent,
                      values.count == 5
                else {
                    throw PyTorchCheckpointError.invalidPickle("unsupported persistent ID")
                }
                guard try string(values[0], role: "persistent tag") == "storage",
                      case let .global(module, storageName) = values[1],
                      module == "torch"
                else {
                    throw PyTorchCheckpointError.invalidPickle("unsupported persistent object")
                }
                let dtype = try storageDType(storageName)
                stack.append(
                    .storage(
                        Storage(
                            dtype: dtype,
                            key: try string(values[2], role: "storage key"),
                            elementCount: try integer(values[4], role: "storage element count")
                        )
                    )
                )
            case 0x52: // REDUCE
                let arguments = try pop()
                let callable = try pop()
                let reduced: Value
                if case let .global(module, name) = callable,
                   module == "collections", name == "OrderedDict"
                {
                    reduced = .dictionary
                } else if case let .global(module, name) = callable,
                          module == "torch._utils", name == "_rebuild_tensor_v2"
                {
                    guard case let .tuple(values) = arguments,
                          values.count >= 4,
                          case let .storage(storage) = values[0]
                    else {
                        throw PyTorchCheckpointError.invalidPickle(
                            "invalid _rebuild_tensor_v2 arguments"
                        )
                    }
                    let tensor = PyTorchParsedTensor(
                        name: "",
                        dtype: storage.dtype,
                        storageKey: storage.key,
                        storageElementCount: storage.elementCount,
                        storageOffset: try integer(values[1], role: "storage offset"),
                        shape: try integers(values[2], role: "tensor shape"),
                        strides: try integers(values[3], role: "tensor strides")
                    )
                    reduced = .tensor(tensor)
                } else {
                    throw PyTorchCheckpointError.invalidPickle("unsupported REDUCE callable")
                }
                capture(reduced)
                stack.append(reduced)
            case 0x73: // SETITEM
                _ = try pop()
                _ = try pop()
            case 0x75: // SETITEMS
                _ = try valuesAfterMark()
            case 0x61: // APPEND
                let value = try pop()
                guard let container = stack.popLast(), case var .list(values) = container else {
                    throw PyTorchCheckpointError.invalidPickle("APPEND target is not a list")
                }
                values.append(value)
                stack.append(.list(values))
            case 0x65: // APPENDS
                let values = try valuesAfterMark()
                guard let container = stack.popLast(), case var .list(existing) = container else {
                    throw PyTorchCheckpointError.invalidPickle("APPENDS target is not a list")
                }
                existing.append(contentsOf: values)
                stack.append(.list(existing))
            case 0x62: // BUILD
                _ = try pop()
            case 0x2e: // STOP
                stopped = true
            default:
                throw PyTorchCheckpointError.unsupportedPickleOpcode(opcode, opcodeOffset)
            }
        }
        guard stopped else {
            throw PyTorchCheckpointError.invalidPickle("missing STOP opcode")
        }
        return tensors
    }

    private static func storageDType(_ name: String) throws -> String {
        switch name {
        case "BFloat16Storage": return "BF16"
        case "HalfStorage": return "F16"
        case "FloatStorage": return "F32"
        case "DoubleStorage": return "F64"
        case "ByteStorage": return "U8"
        case "CharStorage": return "I8"
        case "ShortStorage": return "I16"
        case "IntStorage": return "I32"
        case "LongStorage": return "I64"
        case "BoolStorage": return "BOOL"
        default: throw PyTorchCheckpointError.unsupportedStorageType(name)
        }
    }
}
