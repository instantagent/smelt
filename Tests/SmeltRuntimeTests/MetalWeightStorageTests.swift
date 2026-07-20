import XCTest
import Metal
import Darwin
import SmeltSchema
@testable import SmeltRuntime

final class MetalWeightStorageTests: XCTestCase {
    private func entry(
        _ name: String,
        offset: UInt64,
        size: UInt64,
        scalesOffset: UInt64? = nil,
        scalesSize: UInt64? = nil
    ) -> SmeltWeightEntry {
        SmeltWeightEntry(
            name: name,
            offset: offset,
            sizeBytes: size,
            shape: [Int(size)],
            dtype: .raw,
            scalesOffset: scalesOffset,
            scalesSizeBytes: scalesSize
        )
    }

    func testPlannerSplitsAtTensorBoundariesAndIncludesAuxiliaryStorage() throws {
        let layouts = try SmeltMetalWeightSegmentPlanner.plan(
            totalBytes: 768,
            entries: [
                entry("a", offset: 0, size: 100, scalesOffset: 128, scalesSize: 120),
                entry("b", offset: 256, size: 200),
                entry("c", offset: 512, size: 200),
            ],
            maximumBufferLength: 320,
            pageSize: 64
        )

        XCTAssertEqual(layouts.map(\.logicalRange), [0..<256, 256..<512, 512..<768])
        XCTAssertTrue(layouts.allSatisfy { $0.mappedLength <= 320 })
    }

    func testPlannerRejectsSplitThroughAuxiliaryStorage() throws {
        XCTAssertThrowsError(
            try SmeltMetalWeightSegmentPlanner.plan(
                totalBytes: 512,
                entries: [
                    entry("a", offset: 0, size: 100, scalesOffset: 128, scalesSize: 172),
                    entry("b", offset: 256, size: 144),
                ],
                maximumBufferLength: 320,
                pageSize: 64
            )
        )
    }

    func testPlannerKeepsAuxiliaryStorageBeforePrimaryTensorTogether() throws {
        let layouts = try SmeltMetalWeightSegmentPlanner.plan(
            totalBytes: 640,
            entries: [
                entry("a", offset: 64, size: 192),
                entry("b", offset: 384, size: 128, scalesOffset: 320, scalesSize: 64),
            ],
            maximumBufferLength: 384,
            pageSize: 64
        )

        XCTAssertEqual(layouts.map(\.logicalRange), [0..<320, 320..<640])
    }

    func testMappedSegmentsResolveOneLogicalWeightFile() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device unavailable")
        }
        let page = Int(getpagesize())
        let total = page * 72
        let boundary1 = page * 24
        let boundary2 = page * 48
        var data = Data(repeating: 0, count: total)
        data[page] = 11
        data[boundary1 + page] = 22
        data[boundary2 + page] = 33
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("weights.bin")
        try data.write(to: file)

        let manifest = SmeltWeightManifest(
            totalBytes: UInt64(total),
            entries: [
                entry("a", offset: 0, size: UInt64(page * 20)),
                entry("b", offset: UInt64(boundary1), size: UInt64(page * 20)),
                entry("c", offset: UInt64(boundary2), size: UInt64(page * 20)),
            ]
        )
        let buffers = try SmeltMetalWeightBuffers(
            path: file.path,
            manifest: manifest,
            device: device,
            maximumBufferLength: page * 28
        )

        XCTAssertEqual(buffers.segments.count, 3)
        for (offset, expected) in [
            (page, UInt8(11)),
            (boundary1 + page, UInt8(22)),
            (boundary2 + page, UInt8(33)),
        ] {
            let resolved = try XCTUnwrap(buffers.resolve(logicalOffset: offset))
            XCTAssertEqual(
                resolved.buffer.contents().advanced(by: resolved.offset).load(as: UInt8.self),
                expected
            )
        }
    }
}
