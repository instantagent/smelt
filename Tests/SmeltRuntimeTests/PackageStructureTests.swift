import Darwin
import Foundation
import XCTest
@testable import SmeltRuntime
import SmeltSchema

final class PackageStructureTests: XCTestCase {

    private static func makeManagedTempRoot(_ name: String) -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(name)-\(getpid())", isDirectory: true)
        try? FileManager.default.removeItem(at: root)
        try? FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )
        atexit_b {
            try? FileManager.default.removeItem(at: root)
        }
        return root
    }

    private static let tempRoot: URL = {
        makeManagedTempRoot("smelt-package-structure-tests")
    }()

    private func makeTempPackage() throws -> String {
        let root = Self.tempRoot
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )
        return root.path
    }

    private func writeManifest(
        at packagePath: String,
        pipelines: [String]
    ) throws {
        let manifest = SmeltManifest(
            modelName: "test/model",
            config: SmeltManifestConfig(
                hiddenSize: 1,
                numLayers: 1,
                vocabSize: 1,
                staticSeqCapacity: 1,
                ropeDim: 1,
                numDeltaLayers: 0,
                numAttnLayers: 0,
                ffnDim: 1
            ),
            checksums: SmeltManifestChecksums(
                weightsBin: "",
                metallib: "",
                generatedSwift: "",
                dispatchesBin: ""
            ),
            device: SmeltDeviceRequirements(
                metalFamily: .apple7,
                minMemoryBytes: 1
            ),
            weights: SmeltWeightManifest(totalBytes: 0, entries: []),
            buffers: SmeltBufferTable(slots: []),
            pipelines: pipelines,
            slotLayout: SmeltSlotLayout(
                convStateBaseSlot: 0,
                recStateBaseSlot: 0,
                keyCacheBaseSlot: 0,
                valCacheBaseSlot: 0,
                ropeCosSlot: 0,
                ropeSinSlot: 0,
                tokenIdSlot: 0,
                positionSlot: 0,
                weightsSlot: 0
            ),
            prefill: SmeltPrefillManifest(
                engine: "metal",
                modelPath: "",
                maxBatchSize: 64,
                handoff: SmeltHandoffTable(
                    entries: [],
                    ropeCosSlot: 0,
                    ropeSinSlot: 0
                ),
                inputContract: SmeltPrefillInputContract()
            )
        )
        try manifest.encodePrettyJSON().write(
            to: URL(fileURLWithPath: "\(packagePath)/manifest.json")
        )
    }

    private func writePrefillTable(
        at packagePath: String,
        records: [SmeltDispatchRecord],
        fileName: String = "prefill_dispatches.bin"
    ) throws {
        let data = records.withUnsafeBufferPointer { buffer in
            Data(
                bytes: buffer.baseAddress!,
                count: buffer.count * MemoryLayout<SmeltDispatchRecord>.stride
            )
        }
        try data.write(to: URL(fileURLWithPath: "\(packagePath)/\(fileName)"))
    }

    func testPrefillStructureCountsDispatchesAndSwaps() throws {
        let packagePath = try makeTempPackage()
        try writeManifest(at: packagePath, pipelines: ["alpha", "beta", "gamma"])

        var alpha = SmeltDispatchRecord.empty()
        alpha.opKind = SmeltDispatchRecord.opDispatch
        alpha.pipeline = 0

        var beta = SmeltDispatchRecord.empty()
        beta.opKind = SmeltDispatchRecord.opDispatch
        beta.pipeline = 1

        var gamma = SmeltDispatchRecord.empty()
        gamma.opKind = SmeltDispatchRecord.opDispatch
        gamma.pipeline = 2

        try writePrefillTable(
            at: packagePath,
            records: [alpha, beta, SmeltDispatchRecord.swap(), gamma, beta]
        )

        let report = try XCTUnwrap(
            SmeltPackageStructure.inspectPrefill(packagePath: packagePath)
        )
        XCTAssertEqual(report.totalRecords, 5)
        XCTAssertEqual(report.dispatchCount, 4)
        XCTAssertEqual(report.swapCount, 1)
        XCTAssertEqual(report.dispatchCount(named: "beta"), 2)
        XCTAssertEqual(report.dispatchCount(named: "alpha"), 1)
        XCTAssertEqual(report.dispatchCount(named: "gamma"), 1)
        XCTAssertEqual(report.pipelineUsages.first?.name, "beta")
        XCTAssertEqual(report.tableName, "prefill_dispatches.bin")
    }

    func testPrefillVerifyArgmaxStructureCountsDispatchesAndSwaps() throws {
        let packagePath = try makeTempPackage()
        try writeManifest(at: packagePath, pipelines: ["alpha", "beta"])

        var alpha = SmeltDispatchRecord.empty()
        alpha.opKind = SmeltDispatchRecord.opDispatch
        alpha.pipeline = 0

        var beta = SmeltDispatchRecord.empty()
        beta.opKind = SmeltDispatchRecord.opDispatch
        beta.pipeline = 1

        try writePrefillTable(
            at: packagePath,
            records: [alpha, SmeltDispatchRecord.swap(), beta, beta],
            fileName: "prefill_verify_argmax_dispatches.bin"
        )

        let report = try XCTUnwrap(
            SmeltPackageStructure.inspectPrefillVerifyArgmax(packagePath: packagePath)
        )
        XCTAssertEqual(report.totalRecords, 4)
        XCTAssertEqual(report.dispatchCount, 3)
        XCTAssertEqual(report.swapCount, 1)
        XCTAssertEqual(report.dispatchCount(named: "beta"), 2)
        XCTAssertEqual(report.dispatchCount(named: "alpha"), 1)
        XCTAssertEqual(report.tableName, "prefill_verify_argmax_dispatches.bin")
    }

    func testPrefillStructureReturnsNilWithoutPrefillTable() throws {
        let packagePath = try makeTempPackage()
        try writeManifest(at: packagePath, pipelines: [])

        let report = try SmeltPackageStructure.inspectPrefill(packagePath: packagePath)
        XCTAssertNil(report)
    }

    func testDecodeStructureCountsDispatchesAndSwaps() throws {
        let packagePath = try makeTempPackage()
        try writeManifest(at: packagePath, pipelines: ["alpha", "beta"])

        var alpha = SmeltDispatchRecord.empty()
        alpha.opKind = SmeltDispatchRecord.opDispatch
        alpha.pipeline = 0

        var beta = SmeltDispatchRecord.empty()
        beta.opKind = SmeltDispatchRecord.opDispatch
        beta.pipeline = 1

        let data = [alpha, SmeltDispatchRecord.swap(), beta, beta]
            .withUnsafeBufferPointer { buffer in
                Data(
                    bytes: buffer.baseAddress!,
                    count: buffer.count * MemoryLayout<SmeltDispatchRecord>.stride
                )
            }
        try data.write(to: URL(fileURLWithPath: "\(packagePath)/dispatches.bin"))

        let report = try XCTUnwrap(
            SmeltPackageStructure.inspectDecode(packagePath: packagePath)
        )
        XCTAssertEqual(report.tableName, "dispatches.bin")
        XCTAssertEqual(report.totalRecords, 4)
        XCTAssertEqual(report.dispatchCount, 3)
        XCTAssertEqual(report.swapCount, 1)
        XCTAssertEqual(report.dispatchCount(named: "beta"), 2)
    }
}
