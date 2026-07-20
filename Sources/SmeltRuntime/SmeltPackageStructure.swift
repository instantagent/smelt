import Foundation
import SmeltSchema

public struct SmeltDispatchPipelineUsage: Sendable {
    public let pipelineIndex: UInt16
    public let name: String
    public let dispatchCount: Int
}

public struct SmeltDispatchStructureReport: Sendable {
    public let packagePath: String
    public let tableName: String
    public let totalRecords: Int
    public let dispatchCount: Int
    public let swapCount: Int
    public let pipelineUsages: [SmeltDispatchPipelineUsage]

    public func dispatchCount(named pipelineName: String) -> Int {
        pipelineUsages.first(where: { $0.name == pipelineName })?.dispatchCount ?? 0
    }
}

public typealias SmeltPrefillPipelineUsage = SmeltDispatchPipelineUsage
public typealias SmeltPrefillStructureReport = SmeltDispatchStructureReport
public typealias SmeltDecodePipelineUsage = SmeltDispatchPipelineUsage
public typealias SmeltDecodeStructureReport = SmeltDispatchStructureReport

public enum SmeltPackageStructure {
    public static func inspectDecode(packagePath: String) throws -> SmeltDecodeStructureReport? {
        let manifestPath = "\(packagePath)/manifest.json"
        let manifestData = try Data(contentsOf: URL(fileURLWithPath: manifestPath))
        let manifest = try JSONDecoder().decode(SmeltManifest.self, from: manifestData)
        return try inspectTable(
            packagePath: packagePath,
            manifest: manifest,
            fileName: "dispatches.bin"
        )
    }

    public static func inspectPrefill(packagePath: String) throws -> SmeltPrefillStructureReport? {
        let manifestPath = "\(packagePath)/manifest.json"
        let manifestData = try Data(contentsOf: URL(fileURLWithPath: manifestPath))
        let manifest = try JSONDecoder().decode(SmeltManifest.self, from: manifestData)
        return try inspectPrefill(packagePath: packagePath, manifest: manifest)
    }

    public static func inspectPrefillVerifyArgmax(
        packagePath: String
    ) throws -> SmeltPrefillStructureReport? {
        let manifestPath = "\(packagePath)/manifest.json"
        let manifestData = try Data(contentsOf: URL(fileURLWithPath: manifestPath))
        let manifest = try JSONDecoder().decode(SmeltManifest.self, from: manifestData)
        return try inspectTable(
            packagePath: packagePath,
            manifest: manifest,
            fileName: "prefill_verify_argmax_dispatches.bin"
        )
    }

    static func inspectPrefill(
        packagePath: String,
        manifest: SmeltManifest
    ) throws -> SmeltPrefillStructureReport? {
        try inspectTable(
            packagePath: packagePath,
            manifest: manifest,
            fileName: "prefill_dispatches.bin"
        )
    }

    static func inspectTable(
        packagePath: String,
        manifest: SmeltManifest,
        fileName: String
    ) throws -> SmeltDispatchStructureReport? {
        let tablePath = "\(packagePath)/\(fileName)"
        guard FileManager.default.fileExists(atPath: tablePath) else {
            return nil
        }

        let tableData = try Data(contentsOf: URL(fileURLWithPath: tablePath))
        let stride = MemoryLayout<SmeltDispatchRecord>.stride
        guard tableData.count % stride == 0 else {
            throw SmeltRuntimeError.invalidPackage(
                "\(fileName) size \(tableData.count) is not a multiple of dispatch stride \(stride)"
            )
        }

        let tableCount = tableData.count / stride
        var swapCount = 0
        var pipelineCounts: [UInt16: Int] = [:]

        tableData.withUnsafeBytes { ptr in
            let table = ptr.bindMemory(to: SmeltDispatchRecord.self)
            for idx in 0..<tableCount {
                let rec = table[idx]
                if rec.opKind == SmeltDispatchRecord.opSwap {
                    swapCount += 1
                    continue
                }
                if rec.opKind != SmeltDispatchRecord.opDispatch {
                    continue
                }
                pipelineCounts[rec.pipeline, default: 0] += 1
            }
        }

        let usages = pipelineCounts.map { pipeline, count in
            let name = Int(pipeline) < manifest.pipelines.count
                ? manifest.pipelines[Int(pipeline)]
                : "pipeline_\(pipeline)"
            return SmeltDispatchPipelineUsage(
                pipelineIndex: pipeline,
                name: name,
                dispatchCount: count
            )
        }
        .sorted {
            if $0.dispatchCount != $1.dispatchCount {
                return $0.dispatchCount > $1.dispatchCount
            }
            if $0.name != $1.name {
                return $0.name < $1.name
            }
            return $0.pipelineIndex < $1.pipelineIndex
        }

        return SmeltDispatchStructureReport(
            packagePath: packagePath,
            tableName: fileName,
            totalRecords: tableCount,
            dispatchCount: pipelineCounts.values.reduce(0, +),
            swapCount: swapCount,
            pipelineUsages: usages
        )
    }
}
