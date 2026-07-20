import Foundation
@preconcurrency import Metal
#if os(macOS)
import Darwin
#endif
import SmeltSchema

struct SmeltMetalWeightSegmentLayout: Equatable, Sendable {
    let logicalRange: Range<Int>
    let mappedFileOffset: Int
    let mappedLength: Int
}

enum SmeltMetalWeightStorageError: Error, CustomStringConvertible {
    case invalidLayout(String)
    case mappingFailed(String)
    case bufferCreationFailed(String)

    var description: String {
        switch self {
        case .invalidLayout(let message):
            return "invalid Metal weight layout: \(message)"
        case .mappingFailed(let message):
            return "Metal weight mmap failed: \(message)"
        case .bufferCreationFailed(let message):
            return "Metal weight buffer creation failed: \(message)"
        }
    }
}

enum SmeltMetalWeightSegmentPlanner {
    static func plan(
        totalBytes: UInt64,
        entries: [SmeltWeightEntry],
        maximumBufferLength: Int,
        pageSize: Int = Int(getpagesize())
    ) throws -> [SmeltMetalWeightSegmentLayout] {
        guard totalBytes > 0, totalBytes <= UInt64(Int.max) else {
            throw SmeltMetalWeightStorageError.invalidLayout(
                "total_bytes \(totalBytes) is outside the host address range"
            )
        }
        let total = Int(totalBytes)
        guard maximumBufferLength > 0, pageSize > 0 else {
            throw SmeltMetalWeightStorageError.invalidLayout(
                "buffer limit and page size must be positive"
            )
        }

        let sorted = try entries.map { ($0, try storageRange($0)) }.sorted {
            if $0.1.lowerBound == $1.1.lowerBound { return $0.0.name < $1.0.name }
            return $0.1.lowerBound < $1.1.lowerBound
        }
        guard !sorted.isEmpty else {
            guard total <= maximumBufferLength else {
                throw SmeltMetalWeightStorageError.invalidLayout(
                    "weights exceed one Metal buffer but the manifest has no tensor boundaries"
                )
            }
            return [try segment(0..<total, total: total, limit: maximumBufferLength, pageSize: pageSize)]
        }

        var layouts: [SmeltMetalWeightSegmentLayout] = []
        var logicalStart = 0
        var coveredEnd = 0
        for (entry, storageRange) in sorted {
            let start = storageRange.lowerBound
            let extentEnd = storageRange.upperBound
            guard start >= logicalStart, extentEnd <= total else {
                throw SmeltMetalWeightStorageError.invalidLayout(
                    "tensor \(entry.name) range [\(start), \(extentEnd)) is outside total_bytes=\(total)"
                )
            }

            let proposedEnd = max(coveredEnd, extentEnd)
            if try mappedLength(
                logicalStart..<proposedEnd,
                total: total,
                pageSize: pageSize
            ) > maximumBufferLength {
                guard coveredEnd > logicalStart else {
                    throw SmeltMetalWeightStorageError.invalidLayout(
                        "tensor \(entry.name) requires more than device maxBufferLength "
                            + "(\(maximumBufferLength) bytes)"
                    )
                }
                guard coveredEnd <= start else {
                    throw SmeltMetalWeightStorageError.invalidLayout(
                        "tensor storage crosses candidate split before \(entry.name)"
                    )
                }
                layouts.append(
                    try segment(
                        logicalStart..<start,
                        total: total,
                        limit: maximumBufferLength,
                        pageSize: pageSize
                    )
                )
                logicalStart = start
                coveredEnd = extentEnd
                guard try mappedLength(
                    logicalStart..<coveredEnd,
                    total: total,
                    pageSize: pageSize
                ) <= maximumBufferLength else {
                    throw SmeltMetalWeightStorageError.invalidLayout(
                        "tensor \(entry.name) requires more than device maxBufferLength "
                            + "(\(maximumBufferLength) bytes)"
                    )
                }
            } else {
                coveredEnd = proposedEnd
            }
        }
        layouts.append(
            try segment(
                logicalStart..<total,
                total: total,
                limit: maximumBufferLength,
                pageSize: pageSize
            )
        )
        return layouts
    }

    private static func storageRange(_ entry: SmeltWeightEntry) throws -> Range<Int> {
        let ranges: [(UInt64, UInt64?)] = [
            (entry.offset, entry.sizeBytes),
            (entry.lutOffset ?? 0, entry.lutOffset == nil ? nil : entry.lutSizeBytes),
            (entry.scalesOffset ?? 0, entry.scalesOffset == nil ? nil : entry.scalesSizeBytes),
            (entry.biasesOffset ?? 0, entry.biasesOffset == nil ? nil : entry.biasesSizeBytes),
            (entry.codebookOffset ?? 0, entry.codebookOffset == nil ? nil : entry.codebookSizeBytes),
        ]
        var minimum = Int.max
        var maximum = 0
        for (offset, size) in ranges {
            guard let size else { continue }
            let end = offset.addingReportingOverflow(size)
            guard !end.overflow else {
                throw SmeltMetalWeightStorageError.invalidLayout(
                    "tensor \(entry.name) storage range overflows UInt64"
                )
            }
            minimum = min(
                minimum,
                try checkedInt(offset, label: "\(entry.name).storage_start")
            )
            maximum = max(maximum, try checkedInt(
                end.partialValue,
                label: "\(entry.name).storage_end"
            ))
        }
        guard minimum < maximum else {
            throw SmeltMetalWeightStorageError.invalidLayout(
                "tensor \(entry.name) has no non-empty storage"
            )
        }
        return minimum..<maximum
    }

    private static func checkedInt(_ value: UInt64, label: String) throws -> Int {
        guard value <= UInt64(Int.max) else {
            throw SmeltMetalWeightStorageError.invalidLayout(
                "\(label)=\(value) exceeds Int.max"
            )
        }
        return Int(value)
    }

    private static func segment(
        _ logicalRange: Range<Int>,
        total: Int,
        limit: Int,
        pageSize: Int
    ) throws -> SmeltMetalWeightSegmentLayout {
        guard !logicalRange.isEmpty else {
            throw SmeltMetalWeightStorageError.invalidLayout("empty segment")
        }
        let mappedStart = alignDown(logicalRange.lowerBound, to: pageSize)
        let mappedEnd = min(try alignUp(logicalRange.upperBound, to: pageSize), total)
        let length = mappedEnd - mappedStart
        guard length > 0, length <= limit else {
            throw SmeltMetalWeightStorageError.invalidLayout(
                "mapped segment [\(mappedStart), \(mappedEnd)) exceeds limit \(limit)"
            )
        }
        return SmeltMetalWeightSegmentLayout(
            logicalRange: logicalRange,
            mappedFileOffset: mappedStart,
            mappedLength: length
        )
    }

    private static func mappedLength(
        _ logicalRange: Range<Int>,
        total: Int,
        pageSize: Int
    ) throws -> Int {
        guard !logicalRange.isEmpty else { return 0 }
        return min(try alignUp(logicalRange.upperBound, to: pageSize), total)
            - alignDown(logicalRange.lowerBound, to: pageSize)
    }

    private static func alignDown(_ value: Int, to alignment: Int) -> Int {
        value - value % alignment
    }

    private static func alignUp(_ value: Int, to alignment: Int) throws -> Int {
        let remainder = value % alignment
        guard remainder != 0 else { return value }
        let result = value.addingReportingOverflow(alignment - remainder)
        guard !result.overflow else {
            throw SmeltMetalWeightStorageError.invalidLayout("page alignment overflow")
        }
        return result.partialValue
    }
}

final class SmeltMetalWeightBuffers: @unchecked Sendable {
    struct Segment {
        let layout: SmeltMetalWeightSegmentLayout
        let buffer: MTLBuffer
    }

    let segments: [Segment]
    let totalLogicalBytes: Int

    init(
        path: String,
        manifest: SmeltWeightManifest,
        device: MTLDevice,
        maximumBufferLength: Int? = nil
    ) throws {
        let limit = min(maximumBufferLength ?? device.maxBufferLength, device.maxBufferLength)
        let layouts = try SmeltMetalWeightSegmentPlanner.plan(
            totalBytes: manifest.totalBytes,
            entries: manifest.entries,
            maximumBufferLength: limit
        )
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw SmeltMetalWeightStorageError.mappingFailed("cannot open \(path)")
        }
        defer { close(fd) }
        let fileSize = lseek(fd, 0, SEEK_END)
        guard fileSize >= off_t(manifest.totalBytes) else {
            throw SmeltMetalWeightStorageError.mappingFailed(
                "\(path) is \(fileSize) bytes; expected \(manifest.totalBytes)"
            )
        }

        var created: [Segment] = []
        created.reserveCapacity(layouts.count)
        for (index, layout) in layouts.enumerated() {
            let pointer = mmap(
                nil,
                layout.mappedLength,
                PROT_READ,
                MAP_SHARED,
                fd,
                off_t(layout.mappedFileOffset)
            )
            guard pointer != MAP_FAILED, let pointer else {
                throw SmeltMetalWeightStorageError.mappingFailed(
                    "segment \(index) at \(layout.mappedFileOffset)"
                )
            }
            guard let buffer = device.makeBuffer(
                bytesNoCopy: pointer,
                length: layout.mappedLength,
                options: .storageModeShared,
                deallocator: { pointer, length in munmap(pointer, length) }
            ) else {
                munmap(pointer, layout.mappedLength)
                throw SmeltMetalWeightStorageError.bufferCreationFailed(
                    "segment \(index) [\(layout.logicalRange.lowerBound), "
                        + "\(layout.logicalRange.upperBound))"
                )
            }
            buffer.label = "smelt.weights.\(index)"
            created.append(Segment(layout: layout, buffer: buffer))
        }
        segments = created
        totalLogicalBytes = Int(manifest.totalBytes)
    }

    @inline(__always)
    func resolve(logicalOffset: Int, length: Int = 1) -> (buffer: MTLBuffer, offset: Int)? {
        guard logicalOffset >= 0, length >= 0, length <= totalLogicalBytes,
              logicalOffset <= totalLogicalBytes - length
        else { return nil }
        if segments.count == 1, let segment = segments.first {
            let local = logicalOffset - segment.layout.mappedFileOffset
            guard local >= 0, length <= segment.buffer.length,
                  local <= segment.buffer.length - length
            else { return nil }
            return (segment.buffer, local)
        }
        var low = 0
        var high = segments.count
        while low < high {
            let middle = (low + high) / 2
            let range = segments[middle].layout.logicalRange
            if logicalOffset < range.lowerBound {
                high = middle
            } else if logicalOffset >= range.upperBound {
                low = middle + 1
            } else {
                let segment = segments[middle]
                guard length <= range.upperBound - logicalOffset else { return nil }
                let local = logicalOffset - segment.layout.mappedFileOffset
                guard local >= 0, local <= segment.buffer.length - length else { return nil }
                return (segment.buffer, local)
            }
        }
        return nil
    }
}
