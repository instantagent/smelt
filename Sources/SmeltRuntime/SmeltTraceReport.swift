import CryptoKit
import Foundation
import SmeltSchema

/// Static package execution trace: what this package declares, which compiled/native
/// routes its artifacts make possible, and where the declaration and artifacts disagree.
///
/// This is intentionally cheaper than a model run. It is the package-level witness you
/// run before trusting intermediate labels, benchmarks, or parity claims.
public struct SmeltTraceReport: Codable, Sendable {
    public let schemaVersion: Int
    public let packagePath: String
    public let packageKind: String
    public let modelName: String?
    public let manifestKind: String?
    public let manifestSHA256: String
    public let buildFingerprint: String?
    public let graph: SmeltTraceGraph?
    public let loop: SmeltTraceLoop?
    public let artifacts: [SmeltTraceArtifact]
    public let blocks: [SmeltTraceBlockRoute]
    public let dispatchTables: [SmeltTraceDispatchTable]
    public let sidecars: [SmeltTraceSidecar]
    public let dtypeSummary: [SmeltTraceDTypeCount]
    public let issues: [SmeltTraceIssue]

    public var errorCount: Int {
        issues.filter { $0.severity == .error }.count
    }

    public var warningCount: Int {
        issues.filter { $0.severity == .warning }.count
    }

    public var hasErrors: Bool { errorCount > 0 }
}

public struct SmeltTraceGraph: Codable, Sendable, Equatable {
    public let source: String
    public let signature: String?
    public let blockCount: Int
}

public struct SmeltTraceLoop: Codable, Sendable, Equatable {
    public let source: String
    public let setup: [SmeltTraceLoopPhase]
    public let perStep: [SmeltTraceLoopPhase]
    public let emission: String
    public let stop: [String]
}

public struct SmeltTraceLoopPhase: Codable, Sendable, Equatable {
    public let name: String
    public let blocks: [String]
    public let feedsNextStep: Bool
}

public struct SmeltTraceBlockRoute: Codable, Sendable, Equatable {
    public let name: String
    public let role: String
    public let inputs: [String]
    public let output: String
    public let feedback: String?
    public let state: [String]
    public let sideOutputs: [String]
    public let declaredImpl: String
    public let declaredDelivery: String?
    public let route: String
    public let status: SmeltTraceStatus
    public let evidence: [String]
}

public struct SmeltTraceArtifact: Codable, Sendable {
    public let name: String
    public let path: String
    public let kind: String
    public let exists: Bool
    public let bytes: UInt64?
    public let declaredSHA256: String?
    public let actualSHA256: String?
    public let hashSkippedReason: String?
}

public struct SmeltTraceDispatchTable: Codable, Sendable {
    public let name: String
    public let path: String
    public let exists: Bool
    public let sha256: String?
    public let parseError: String?
    public let totalRecords: Int?
    public let dispatchCount: Int?
    public let swapCount: Int?
    public let topPipelines: [SmeltTracePipelineUsage]
}

public struct SmeltTracePipelineUsage: Codable, Sendable, Equatable {
    public let name: String
    public let dispatchCount: Int
}

public struct SmeltTraceSidecar: Codable, Sendable {
    public let name: String
    public let path: String
    public let exists: Bool
    public let packageKind: String?
    public let modelName: String?
    public let headlessTrunkABI: Bool?
    public let manifestSHA256: String?
    public let dispatchTables: [SmeltTraceDispatchTable]
    public let issues: [SmeltTraceIssue]
}

public struct SmeltTraceDTypeCount: Codable, Sendable, Equatable {
    public let dtype: String
    public let count: Int
}

public struct SmeltTraceIssue: Codable, Sendable, Equatable {
    public let severity: SmeltTraceSeverity
    public let code: String
    public let message: String
}

public enum SmeltTraceSeverity: String, Codable, Sendable {
    case warning
    case error
}

public enum SmeltTraceStatus: String, Codable, Sendable {
    case ok
    case warning
    case error
}

public struct SmeltTraceOptions: Sendable {
    public let hashLargeArtifacts: Bool

    public init(hashLargeArtifacts: Bool = false) {
        self.hashLargeArtifacts = hashLargeArtifacts
    }
}

public enum SmeltTrace {
    private static let schemaVersion = 1
    private static let smallHashLimit: UInt64 = 64 * 1024 * 1024
    private static let emittableTrunkDTypes: Set<String> = ["f32", "f16", "bf16", "u4"]

    public static func inspect(
        packagePath: String,
        options: SmeltTraceOptions = SmeltTraceOptions()
    ) throws -> SmeltTraceReport {
        let manifestURL = URL(fileURLWithPath: packagePath).appendingPathComponent("manifest.json")
        let manifestData = try Data(contentsOf: manifestURL)
        let manifestHash = sha256Hex(manifestData)
        let slim = try decodeSlimManifest(manifestData)

        if SmeltManifest.isHeadlessTrunk(headlessTrunkABI: slim.headlessTrunkABI) {
            return try inspectTextGeneration(
                packagePath: packagePath,
                manifestData: manifestData,
                manifestSHA256: manifestHash,
                forcedKind: "headless-trunk",
                options: options
            )
        }

        let graphPolicy: SmeltRuntimeGraphPolicy?
        do {
            graphPolicy = try SmeltRuntimeGraphPolicy.resolve(manifestData: manifestData)
        } catch SmeltRuntimeGraphPolicy.ResolveError.missingGraph {
            if (try? Qwen3TTSManifest.decode(from: manifestData)) != nil {
                return try inspectTTS(
                    packagePath: packagePath,
                    manifestData: manifestData,
                    manifestSHA256: manifestHash,
                    slim: slim,
                    options: options
                )
            }
            graphPolicy = nil
        } catch {
            let reportLabel = reportPackageLabel(from: slim, manifestData: manifestData)
            return try inspectGeneric(
                packagePath: packagePath,
                manifestSHA256: manifestHash,
                slim: slim,
                inferredKind: reportLabel,
                options: options,
                issues: [
                    .error(
                        "invalidRuntimeGraphPolicy",
                        "manifest blocks could not resolve runtime graph policy: \(error)"
                    )
                ]
            )
        }

        if let graphPolicy {
            switch graphPolicy {
            case .sidecarTextToCodecAudio:
                return try inspectTTS(
                    packagePath: packagePath,
                    manifestData: manifestData,
                    manifestSHA256: manifestHash,
                    slim: slim,
                    options: options
                )
            case .codecAudio:
                return try inspectTTS(
                    packagePath: packagePath,
                    manifestData: manifestData,
                    manifestSHA256: manifestHash,
                    slim: slim,
                    options: options
                )
            case .textGeneration:
                return try inspectTextGeneration(
                    packagePath: packagePath,
                    manifestData: manifestData,
                    manifestSHA256: manifestHash,
                    forcedKind: SmeltRuntimeGraphPolicy.textGeneration.rawValue,
                    options: options
                )
            }
        }

        return try inspectGeneric(
            packagePath: packagePath,
            manifestSHA256: manifestHash,
            slim: slim,
            inferredKind: reportPackageLabel(from: slim, manifestData: manifestData),
            options: options
        )
    }

    private struct SlimManifest: Decodable {
        let kind: String?
        let headlessTrunkABI: Bool?
        let blocks: SmeltBlockGraph?
        let modelName: String?
        let architecture: String?
    }

    private static func decodeSlimManifest(_ data: Data) throws -> SlimManifest {
        do {
            return try JSONDecoder().decode(SlimManifest.self, from: data)
        } catch {
            if let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               object["blocks"] != nil {
                return SlimManifest(
                    kind: object["kind"] as? String,
                    headlessTrunkABI: object["headlessTrunkABI"] as? Bool,
                    blocks: nil,
                    modelName: object["modelName"] as? String,
                    architecture: object["architecture"] as? String
                )
            }
            throw error
        }
    }

    private static func reportPackageLabel(from slim: SlimManifest, manifestData: Data) -> String {
        if SmeltManifest.isHeadlessTrunk(headlessTrunkABI: slim.headlessTrunkABI) {
            return "headless-trunk"
        }
        if let policy = try? SmeltRuntimeGraphPolicy.resolve(manifestData: manifestData) {
            return policy.rawValue
        }
        if let kind = slim.kind?.nilIfEmpty {
            return kind
        }
        return "graphless-package"
    }

    private static func unsupportedRuntimeAdapterIssue(
        kind: String,
        architecture: String?
    ) -> SmeltTraceIssue {
        .warning(
            "unsupportedRuntimeArchitecture",
            "package kind '\(kind)' architecture '\(architecture?.nilIfEmpty ?? "(unspecified)")' "
                + "has no trace graph adapter; recorded package artifacts only"
        )
    }

    @discardableResult
    private static func appendValidationIssue(
        from validate: () throws -> Void,
        code: String,
        into issues: inout [SmeltTraceIssue]
    ) -> Bool {
        do {
            try validate()
            return true
        } catch {
            issues.append(.error(code, String(describing: error)))
            return false
        }
    }

    // MARK: - Generic / future package kinds

    private static func inspectGeneric(
        packagePath: String,
        manifestSHA256: String,
        slim: SlimManifest,
        inferredKind: String,
        options: SmeltTraceOptions,
        issues: [SmeltTraceIssue]? = nil
    ) throws -> SmeltTraceReport {
        let artifacts = try genericArtifacts(packagePath: packagePath, options: options)
        return SmeltTraceReport(
            schemaVersion: schemaVersion,
            packagePath: packagePath,
            packageKind: inferredKind,
            modelName: slim.modelName,
            manifestKind: slim.kind,
            manifestSHA256: manifestSHA256,
            buildFingerprint: nil,
            graph: nil,
            loop: nil,
            artifacts: artifacts,
            blocks: [],
            dispatchTables: [],
            sidecars: [],
            dtypeSummary: [],
            issues: issues ?? [
                .warning(
                    "genericTrace",
                    "package kind '\(inferredKind)' has no trace graph adapter; recorded package artifacts only"
                )
            ]
        )
    }

    // MARK: - Text generation / headless trunk

    private static func inspectTextGeneration(
        packagePath: String,
        manifestData: Data,
        manifestSHA256: String,
        forcedKind: String,
        options: SmeltTraceOptions
    ) throws -> SmeltTraceReport {
        let manifest = try JSONDecoder().decode(SmeltManifest.self, from: manifestData)
        var issues: [SmeltTraceIssue] = []
        let isHeadlessTrunk = forcedKind == "headless-trunk"
        let manifestPolicyValid = appendValidationIssue(
            from: { try manifest.validatePackageOwnedRuntimePolicy() },
            code: "invalidTextCAMPolicy",
            into: &issues
        )
        let graph: SmeltBlockGraph?
        let graphSource: String
        if let declared = manifest.blocks {
            graph = declared
            graphSource = "declared"
        } else if isHeadlessTrunk {
            graph = nil
            graphSource = "headless-trunk"
        } else {
            graph = nil
            graphSource = "missing"
            issues.append(.error("missingTextGraph", "text-generation trace requires a declared block graph"))
        }
        let loop: SmeltLoopSchedule?
        let loopSource: String
        if let declared = manifest.loop {
            loop = declared
            loopSource = "declared"
        } else if isHeadlessTrunk {
            loop = nil
            loopSource = "headless-trunk"
        } else {
            loop = nil
            loopSource = "missing"
            issues.append(.error("missingTextLoop", "text-generation trace requires a declared loop schedule"))
        }

        var blocks = graph.map { blockRoutes(from: $0, evidence: textGenerationEvidence(manifest: manifest)) } ?? [
            SmeltTraceBlockRoute(
                name: "headless-trunk",
                role: "trunk",
                inputs: ["embeddings"],
                output: "hidden",
                feedback: nil,
                state: ["kv-cache"],
                sideOutputs: [],
                declaredImpl: "compiled",
                declaredDelivery: "compiled-inline",
                route: "compiled:compiled-inline:headless-trunk",
                status: .ok,
                evidence: [
                    "headlessTrunkABI=\(manifest.headlessTrunkABI == true)",
                    "kind=\(manifest.kind ?? "nil")",
                ]
            )
        ]

        let decodeTable = dispatchTable(packagePath: packagePath, manifest: manifest, fileName: "dispatches.bin")
        let prefillTable = dispatchTable(packagePath: packagePath, manifest: manifest, fileName: "prefill_dispatches.bin")
        let verifyTable = dispatchTable(
            packagePath: packagePath,
            manifest: manifest,
            fileName: "prefill_verify_argmax_dispatches.bin"
        )
        let dispatchTables = [decodeTable, prefillTable, verifyTable]
        for table in dispatchTables {
            if let parseError = table.parseError {
                issues.append(.error("invalidDispatchTable", "\(table.name): \(parseError)"))
            }
        }
        if manifestPolicyValid, let graph, let loop {
            do {
                try loop.validate(against: graph)
            } catch {
                issues.append(.error("loopMismatch", "\(error)"))
            }
        }

        if !isHeadlessTrunk, decodeTable.exists == false {
            issues.append(.error("missingDecodeTable", "text-generation trace expected dispatches.bin for the compiled trunk"))
        }
        if manifest.prefill != nil, prefillTable.exists == false {
            issues.append(.error("missingPrefillTable", "manifest declares metal prefill but prefill_dispatches.bin is missing"))
        }
        blocks = blocks.map { block in
            guard block.name == "trunk", decodeTable.exists == false else { return block }
            return block.withStatus(.error, adding: "missing dispatches.bin")
        }

        let artifacts = [
            artifact(packagePath, "manifest.json", kind: "manifest", declaredSHA256: nil, options: options),
            artifact(packagePath, "weights.bin", kind: "weights", declaredSHA256: manifest.checksums.weightsBin, options: options),
            artifact(packagePath, "model.metallib", kind: "metallib", declaredSHA256: manifest.checksums.metallib, options: options),
            artifact(packagePath, "SmeltGenerated.swift", kind: "generated", declaredSHA256: manifest.checksums.generatedSwift, options: options),
            artifact(packagePath, "dispatches.bin", kind: "dispatch-table", declaredSHA256: manifest.checksums.dispatchesBin, options: options),
            artifact(packagePath, "prefill_dispatches.bin", kind: "dispatch-table", declaredSHA256: manifest.checksums.prefillDispatchesBin, options: options),
            artifact(
                packagePath, "prefill_verify_argmax_dispatches.bin", kind: "dispatch-table",
                declaredSHA256: manifest.checksums.prefillVerifyArgmaxDispatchesBin, options: options
            ),
            artifact(packagePath, "trace_markers.json", kind: "trace-markers", declaredSHA256: nil, options: options),
            artifact(packagePath, "tokenizer.json", kind: "tokenizer", declaredSHA256: manifest.checksums.tokenizerJSON, options: options),
        ]
        var requiredArtifacts: Set<String> = ["weights.bin", "model.metallib", "SmeltGenerated.swift", "dispatches.bin"]
        if manifest.prefill != nil {
            requiredArtifacts.insert("prefill_dispatches.bin")
        }
        if manifest.checksums.prefillVerifyArgmaxDispatchesBin?.nilIfEmpty != nil {
            requiredArtifacts.insert("prefill_verify_argmax_dispatches.bin")
        }
        if manifest.checksums.tokenizerJSON?.nilIfEmpty != nil {
            requiredArtifacts.insert("tokenizer.json")
        }
        issues += missingArtifactIssues(
            artifacts,
            requiredNames: requiredArtifacts,
            code: "missingTextArtifact"
        )

        return SmeltTraceReport(
            schemaVersion: schemaVersion,
            packagePath: packagePath,
            packageKind: forcedKind,
            modelName: manifest.modelName,
            manifestKind: manifest.kind,
            manifestSHA256: manifestSHA256,
            buildFingerprint: manifest.buildProvenance?.buildFingerprint,
            graph: traceGraph(graph, source: graphSource),
            loop: traceLoop(loop, source: loopSource),
            artifacts: artifacts,
            blocks: blocks,
            dispatchTables: dispatchTables,
            sidecars: [],
            dtypeSummary: dtypeSummary(manifest.weights.entries.map(\.dtype.rawValue)),
            issues: issues
        )
    }

    private static func textGenerationEvidence(manifest: SmeltManifest) -> [String] {
        var evidence = [
            "pipelines=\(manifest.pipelines.count)",
            "weights=\(manifest.weights.entries.count)",
        ]
        if let traceMode = manifest.buildProvenance?.resolvedOptions.traceMode {
            evidence.append("traceMode=\(traceMode)")
        }
        if let prefill = manifest.prefill {
            evidence.append("prefill=\(prefill.engine):maxBatch=\(prefill.maxBatchSize)")
        }
        return evidence
    }

    // MARK: - TTS

    private static func inspectTTS(
        packagePath: String,
        manifestData: Data,
        manifestSHA256: String,
        slim: SlimManifest,
        options: SmeltTraceOptions
    ) throws -> SmeltTraceReport {
        let manifest = try Qwen3TTSManifest.decode(from: manifestData)
        var issues: [SmeltTraceIssue] = []
        let topologyValid = appendValidationIssue(
            from: { try manifest.validateQwen3TTSGraphAndLoop() },
            code: "invalidQwen3TTSTopology",
            into: &issues
        )
        let graph = manifest.blocks
        if graph == nil {
            issues.append(.error("missingTTSGraph", "TTS manifest must declare a CAM block graph"))
        }
        let loop = manifest.loop
        if loop == nil {
            issues.append(.error("missingTTSLoop", "TTS manifest must declare a CAM loop schedule"))
        }
        if topologyValid, let graph, let loop {
            do {
                try loop.validate(against: graph)
            } catch {
                issues.append(.error("loopMismatch", "\(error)"))
            }
        }

        let graphBlocks = graph?.blocks ?? []
        let declaresTrunkSidecar = graphBlocks.contains {
            $0.name == "talker" && $0.compiledDelivery == .compiledSidecar
        }
        let declaresMTPSidecar = graphBlocks.contains {
            $0.name == "mtp-head" && $0.compiledDelivery == .internalSidecar
        }
        let trunkSidecar = declaresTrunkSidecar || sidecarExists(packagePath: packagePath, name: "trunk")
            ? try inspectSidecar(packagePath: packagePath, name: "trunk", options: options)
            : emptySidecar(packagePath: packagePath, name: "trunk")
        let mtpSidecar = declaresMTPSidecar || sidecarExists(packagePath: packagePath, name: "trunk-mtp")
            ? try inspectSidecar(packagePath: packagePath, name: "trunk-mtp", options: options)
            : emptySidecar(packagePath: packagePath, name: "trunk-mtp")
        var sidecars = [trunkSidecar, mtpSidecar].filter {
            $0.exists
                || ($0.name == "trunk" && declaresTrunkSidecar)
                || ($0.name == "trunk-mtp" && declaresMTPSidecar)
        }
        issues += sidecars.flatMap(\.issues)

        let textDType = ttsDType(manifest, "talker.model.text_embedding.weight")
        let codecPipelines = Set(Qwen3TTSCodecEmitter.pipelineNames)
        let missingCodecPipelines = codecPipelines.subtracting(Set(manifest.pipelines)).sorted()
        let trunkIntent = shouldShipTTSCompiledTrunks(manifest)

        if declaresTrunkSidecar && trunkIntent && !trunkSidecar.exists {
            issues.append(.error("missingTalkerSidecar", "talker/MTP projection dtypes can ship trunks but trunk/ is missing"))
        }
        if declaresMTPSidecar && trunkIntent && !mtpSidecar.exists {
            issues.append(.error("missingMTPSidecar", "talker/MTP projection dtypes can ship trunks but trunk-mtp/ is missing"))
        }
        if !trunkIntent && (trunkSidecar.exists || mtpSidecar.exists) {
            issues.append(.warning("unexpectedTrunkSidecar", "trunk sidecar exists although layer-0 talker/MTP dtypes do not form one emittable pair"))
        }
        if !missingCodecPipelines.isEmpty {
            issues.append(.error("missingCodecPipelines", "codec runtime-emitted route is missing pipelines: \(missingCodecPipelines.joined(separator: ", "))"))
        }

        let blocks = (graph?.blocks ?? []).map { block in
            ttsBlockRoute(
                block,
                textDType: textDType,
                trunkSidecar: trunkSidecar,
                mtpSidecar: mtpSidecar,
                missingCodecPipelines: missingCodecPipelines,
                trunkIntent: trunkIntent
            )
        }
        for block in blocks where block.status == .error {
            issues.append(.error("blockRouteMismatch", "\(block.name): \(block.evidence.joined(separator: "; "))"))
        }

        let tokenizerFiles = manifest.tokenizerFiles ?? []
        var artifacts: [SmeltTraceArtifact] = [
            artifact(packagePath, "manifest.json", kind: "manifest", declaredSHA256: nil, options: options),
            artifact(packagePath, "weights.bin", kind: "weights", declaredSHA256: nil, options: options),
            artifact(packagePath, "model.metallib", kind: "metallib", declaredSHA256: nil, options: options),
        ]
        artifacts += tokenizerFiles.map {
            artifact(packagePath, $0, kind: "tokenizer-or-config", declaredSHA256: nil, options: options)
        }
        var requiredArtifacts = Set(["weights.bin", "model.metallib"])
        requiredArtifacts.formUnion(tokenizerFiles)
        issues += missingArtifactIssues(
            artifacts,
            requiredNames: requiredArtifacts,
            code: "missingTTSArtifact"
        )

        // Preserve stable ordering even if one sidecar was absent.
        sidecars.sort { $0.name < $1.name }
        return SmeltTraceReport(
            schemaVersion: schemaVersion,
            packagePath: packagePath,
            packageKind: "tts",
            modelName: manifest.modelName,
            manifestKind: nil,
            manifestSHA256: manifestSHA256,
            buildFingerprint: nil,
            graph: traceGraph(graph, source: "declared"),
            loop: traceLoop(loop, source: "declared"),
            artifacts: artifacts,
            blocks: blocks,
            dispatchTables: [],
            sidecars: sidecars,
            dtypeSummary: dtypeSummary(manifest.weights.map { $0.dtype ?? "f32" }),
            issues: issues
        )
    }

    private static func ttsBlockRoute(
        _ block: SmeltBlockGraph.Block,
        textDType: String,
        trunkSidecar: SmeltTraceSidecar,
        mtpSidecar: SmeltTraceSidecar,
        missingCodecPipelines: [String],
        trunkIntent: Bool
    ) -> SmeltTraceBlockRoute {
        var evidence: [String] = []
        var status: SmeltTraceStatus = .ok
        let route = "\(block.impl.rawValue):\(block.compiledDelivery?.rawValue ?? "native")"
        switch block.name {
        case "tts-frontend":
            evidence.append("text_embedding.dtype=\(textDType)")
            evidence.append("compiledFrontEndSupported=\(textDType == "bf16")")
            if block.impl == .compiled && textDType != "bf16" {
                status = .error
                evidence.append("graph declares compiled front-end but runtime only compiles bf16 text_embedding")
            }
        case "talker":
            evidence.append("trunkIntent=\(trunkIntent)")
            evidence.append("trunk.exists=\(trunkSidecar.exists)")
            if block.compiledDelivery == .compiledSidecar && !trunkSidecar.exists {
                status = .error
                evidence.append("graph declares compiled sidecar but trunk/ is absent")
            } else if block.compiledDelivery != .compiledSidecar && trunkSidecar.exists {
                status = .warning
                evidence.append("trunk/ exists but graph does not declare compiled sidecar")
            }
        case "mtp-head":
            evidence.append("trunk-mtp.exists=\(mtpSidecar.exists)")
            if block.compiledDelivery == .internalSidecar && !mtpSidecar.exists {
                status = .error
                evidence.append("graph declares internal sidecar but trunk-mtp/ is absent")
            } else if block.compiledDelivery != .internalSidecar && mtpSidecar.exists {
                status = .warning
                evidence.append("trunk-mtp/ exists but graph does not declare internal sidecar")
            }
        case "codec-decoder":
            evidence.append("codecPipelines.missing=\(missingCodecPipelines.count)")
            if block.impl == .compiled && !missingCodecPipelines.isEmpty {
                status = .error
                evidence.append("runtime-emitted codec route missing required package pipelines")
            }
        default:
            break
        }
        return routeBlock(block, route: route, status: status, evidence: evidence)
    }

    private static func ttsDType(_ manifest: Qwen3TTSManifest, _ name: String) -> String {
        manifest.weights.first(where: { $0.name == name })?.dtype ?? "f32"
    }

    private static func shouldShipTTSCompiledTrunks(_ manifest: Qwen3TTSManifest) -> Bool {
        func q0(_ prefix: String) -> String? {
            guard let e = manifest.weights.first(where: {
                $0.name == "\(prefix)layers.0.self_attn.q_proj.weight"
            }) else { return nil }
            return e.dtype ?? "f32"
        }
        let talker = q0("talker.model.")
        let mtp = q0("talker.code_predictor.model.")
        return manifest.weights.contains(where: { $0.name.hasPrefix("talker.") })
            && manifest.weights.contains(where: { $0.name == "talker.model.codec_embedding.weight" })
            && talker != nil
            && talker == mtp
            && emittableTrunkDTypes.contains(talker!)
    }

    // MARK: - Sidecars / dispatch / artifacts

    private static func inspectSidecar(
        packagePath: String,
        name: String,
        options: SmeltTraceOptions
    ) throws -> SmeltTraceSidecar {
        let path = "\(packagePath)/\(name)"
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else {
            return SmeltTraceSidecar(
                name: name,
                path: path,
                exists: false,
                packageKind: nil,
                modelName: nil,
                headlessTrunkABI: nil,
                manifestSHA256: nil,
                dispatchTables: [],
                issues: []
            )
        }
        let manifestPath = "\(path)/manifest.json"
        var issues: [SmeltTraceIssue] = []
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: manifestPath)) else {
            return SmeltTraceSidecar(
                name: name,
                path: path,
                exists: true,
                packageKind: nil,
                modelName: nil,
                headlessTrunkABI: nil,
                manifestSHA256: nil,
                dispatchTables: [],
                issues: [.error("sidecarMissingManifest", "\(name)/ exists but has no manifest.json")]
            )
        }
        let manifest: SmeltManifest
        do {
            manifest = try SmeltManifest.decode(from: data)
        } catch {
            return SmeltTraceSidecar(
                name: name,
                path: path,
                exists: true,
                packageKind: nil,
                modelName: nil,
                headlessTrunkABI: nil,
                manifestSHA256: sha256Hex(data),
                dispatchTables: [],
                issues: [.error("sidecarInvalidManifest", "\(name)/ manifest.json is invalid: \(error)")]
            )
        }
        let isHeadless = SmeltManifest.isHeadlessTrunk(headlessTrunkABI: manifest.headlessTrunkABI)
        let packageKind = isHeadless ? "headless-trunk" : (manifest.kind ?? "graphless-package")
        if !isHeadless {
            issues.append(.error(
                "sidecarNotHeadless",
                "\(name)/ is expected to be a headless trunk sidecar, got label \(packageKind)"
            ))
        }
        let decode = dispatchTable(packagePath: path, manifest: manifest, fileName: "dispatches.bin")
        let prefill = dispatchTable(packagePath: path, manifest: manifest, fileName: "prefill_dispatches.bin")
        if !decode.exists {
            issues.append(.error("sidecarMissingDecodeTable", "\(name)/ is missing dispatches.bin"))
        }
        if !prefill.exists {
            issues.append(.error("sidecarMissingPrefillTable", "\(name)/ is missing prefill_dispatches.bin"))
        }
        for table in [decode, prefill] {
            if let parseError = table.parseError {
                issues.append(.error("sidecarInvalidDispatchTable", "\(name)/\(table.name): \(parseError)"))
            }
        }
        _ = artifact(path, "manifest.json", kind: "manifest", declaredSHA256: nil, options: options)
        return SmeltTraceSidecar(
            name: name,
            path: path,
            exists: true,
            packageKind: packageKind,
            modelName: manifest.modelName,
            headlessTrunkABI: manifest.headlessTrunkABI,
            manifestSHA256: sha256Hex(data),
            dispatchTables: [decode, prefill],
            issues: issues
        )
    }

    private static func sidecarExists(packagePath: String, name: String) -> Bool {
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(
            atPath: "\(packagePath)/\(name)",
            isDirectory: &isDir
        ) && isDir.boolValue
    }

    private static func emptySidecar(packagePath: String, name: String) -> SmeltTraceSidecar {
        SmeltTraceSidecar(
            name: name,
            path: "\(packagePath)/\(name)",
            exists: false,
            packageKind: nil,
            modelName: nil,
            headlessTrunkABI: nil,
            manifestSHA256: nil,
            dispatchTables: [],
            issues: []
        )
    }

    private static func dispatchTable(
        packagePath: String,
        manifest: SmeltManifest,
        fileName: String
    ) -> SmeltTraceDispatchTable {
        let path = "\(packagePath)/\(fileName)"
        guard FileManager.default.fileExists(atPath: path) else {
            return SmeltTraceDispatchTable(
                name: fileName,
                path: path,
                exists: false,
                sha256: nil,
                parseError: nil,
                totalRecords: nil,
                dispatchCount: nil,
                swapCount: nil,
                topPipelines: []
            )
        }
        let report: SmeltDispatchStructureReport?
        let parseError: String?
        do {
            report = try SmeltPackageStructure.inspectTableForTrace(
                packagePath: packagePath,
                manifest: manifest,
                fileName: fileName
            )
            parseError = nil
        } catch {
            report = nil
            parseError = String(describing: error)
        }
        return SmeltTraceDispatchTable(
            name: fileName,
            path: path,
            exists: true,
            sha256: try? sha256Hex(ofFileAt: path),
            parseError: parseError,
            totalRecords: report?.totalRecords,
            dispatchCount: report?.dispatchCount,
            swapCount: report?.swapCount,
            topPipelines: (report?.pipelineUsages.prefix(8) ?? []).map {
                SmeltTracePipelineUsage(name: $0.name, dispatchCount: $0.dispatchCount)
            }
        )
    }

    private static func artifact(
        _ packagePath: String,
        _ relativePath: String,
        kind: String,
        declaredSHA256: String?,
        options: SmeltTraceOptions
    ) -> SmeltTraceArtifact {
        let path = "\(packagePath)/\(relativePath)"
        let fm = FileManager.default
        guard fm.fileExists(atPath: path) else {
            return SmeltTraceArtifact(
                name: relativePath,
                path: path,
                kind: kind,
                exists: false,
                bytes: nil,
                declaredSHA256: declaredSHA256?.nilIfEmpty,
                actualSHA256: nil,
                hashSkippedReason: nil
            )
        }
        let bytes = fileSize(path)
        let shouldHash = options.hashLargeArtifacts || (bytes ?? 0) <= smallHashLimit
        let actual = shouldHash ? (try? sha256Hex(ofFileAt: path)) : nil
        let skipped = shouldHash ? nil : "larger than \(smallHashLimit) bytes; pass --hash-large-artifacts to hash"
        return SmeltTraceArtifact(
            name: relativePath,
            path: path,
            kind: kind,
            exists: true,
            bytes: bytes,
            declaredSHA256: declaredSHA256?.nilIfEmpty,
            actualSHA256: actual,
            hashSkippedReason: skipped
        )
    }

    private static func genericArtifacts(
        packagePath: String,
        options: SmeltTraceOptions
    ) throws -> [SmeltTraceArtifact] {
        let packageURL = URL(fileURLWithPath: packagePath)
        let packageRoot = packageURL.resolvingSymlinksInPath().path
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .isHiddenKey]
        guard let enumerator = FileManager.default.enumerator(
            at: packageURL,
            includingPropertiesForKeys: Array(keys),
            options: [.skipsHiddenFiles]
        ) else {
            return [
                artifact(packagePath, "manifest.json", kind: "manifest", declaredSHA256: nil, options: options)
            ]
        }

        var paths: [String] = []
        for case let fileURL as URL in enumerator {
            let values = try fileURL.resourceValues(forKeys: keys)
            guard values.isRegularFile == true else { continue }
            let filePath = fileURL.resolvingSymlinksInPath().path
            guard filePath.hasPrefix(packageRoot) else { continue }
            let relative = String(filePath.dropFirst(packageRoot.count))
                .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            guard !relative.isEmpty else { continue }
            paths.append(relative)
        }

        if !paths.contains("manifest.json") {
            paths.append("manifest.json")
        }
        return paths.sorted().map { relative in
            artifact(
                packagePath,
                relative,
                kind: relative == "manifest.json" ? "manifest" : "package-file",
                declaredSHA256: nil,
                options: options
            )
        }
    }

    // MARK: - Shared shaping

    private static func blockRoutes(from graph: SmeltBlockGraph, evidence: [String]) -> [SmeltTraceBlockRoute] {
        graph.blocks.map {
            routeBlock(
                $0,
                route: "\($0.impl.rawValue):\($0.compiledDelivery?.rawValue ?? "native")",
                status: .ok,
                evidence: evidence
            )
        }
    }

    private static func routeBlock(
        _ block: SmeltBlockGraph.Block,
        route: String,
        status: SmeltTraceStatus,
        evidence: [String]
    ) -> SmeltTraceBlockRoute {
        SmeltTraceBlockRoute(
            name: block.name,
            role: block.role.rawValue,
            inputs: block.inputs.map(\.rawValue),
            output: block.output.rawValue,
            feedback: block.feedback?.rawValue,
            state: (block.state ?? []).map(\.rawValue),
            sideOutputs: (block.sideOutputs ?? []).map(\.rawValue),
            declaredImpl: block.impl.rawValue,
            declaredDelivery: block.compiledDelivery?.rawValue,
            route: route,
            status: status,
            evidence: evidence
        )
    }

    private static func traceGraph(_ graph: SmeltBlockGraph?, source: String) -> SmeltTraceGraph? {
        guard let graph else { return nil }
        let signature = graph.signature.map { "\($0.input.rawValue)->\($0.output.rawValue)" }
        return SmeltTraceGraph(source: source, signature: signature, blockCount: graph.blocks.count)
    }

    private static func traceLoop(_ loop: SmeltLoopSchedule?, source: String) -> SmeltTraceLoop? {
        guard let loop else { return nil }
        return SmeltTraceLoop(
            source: source,
            setup: loop.setup.map(traceLoopPhase),
            perStep: loop.perStep.map(traceLoopPhase),
            emission: traceEmission(loop.emission),
            stop: loop.stop.map(\.rawValue)
        )
    }

    private static func traceLoopPhase(_ phase: SmeltLoopSchedule.Phase) -> SmeltTraceLoopPhase {
        SmeltTraceLoopPhase(
            name: phase.name,
            blocks: phase.blocks,
            feedsNextStep: phase.feedsNextStep ?? false
        )
    }

    private static func traceEmission(_ emission: SmeltLoopSchedule.Emission) -> String {
        switch emission {
        case .perStep:
            return "per-step"
        case .chunked(let first, let max, let growth, let via):
            return "chunked:first=\(first):max=\(max):growth=\(growth.rawValue):via=\(via)"
        case .final(let via):
            return "final:via=\(via)"
        }
    }

    private static func dtypeSummary(_ dtypes: [String]) -> [SmeltTraceDTypeCount] {
        Dictionary(grouping: dtypes, by: { $0 }).map {
            SmeltTraceDTypeCount(dtype: $0.key, count: $0.value.count)
        }.sorted {
            if $0.count != $1.count { return $0.count > $1.count }
            return $0.dtype < $1.dtype
        }
    }

    private static func missingArtifactIssues(
        _ artifacts: [SmeltTraceArtifact],
        requiredNames: Set<String>,
        code: String
    ) -> [SmeltTraceIssue] {
        artifacts
            .filter { requiredNames.contains($0.name) && !$0.exists }
            .map { .error(code, "package artifact missing: \($0.path)") }
    }

    private static func fileSize(_ path: String) -> UInt64? {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attrs[.size] as? NSNumber else {
            return nil
        }
        return size.uint64Value
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func sha256Hex(ofFileAt path: String) throws -> String {
        let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let chunk = try handle.read(upToCount: 1 << 20) ?? Data()
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

private extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}

private extension SmeltTraceIssue {
    static func warning(_ code: String, _ message: String) -> SmeltTraceIssue {
        SmeltTraceIssue(severity: .warning, code: code, message: message)
    }

    static func error(_ code: String, _ message: String) -> SmeltTraceIssue {
        SmeltTraceIssue(severity: .error, code: code, message: message)
    }
}

private extension SmeltTraceBlockRoute {
    func withStatus(_ newStatus: SmeltTraceStatus, adding evidenceItem: String) -> SmeltTraceBlockRoute {
        SmeltTraceBlockRoute(
            name: name,
            role: role,
            inputs: inputs,
            output: output,
            feedback: feedback,
            state: state,
            sideOutputs: sideOutputs,
            declaredImpl: declaredImpl,
            declaredDelivery: declaredDelivery,
            route: route,
            status: newStatus,
            evidence: evidence + [evidenceItem]
        )
    }
}

extension SmeltPackageStructure {
    static func inspectTableForTrace(
        packagePath: String,
        manifest: SmeltManifest,
        fileName: String
    ) throws -> SmeltDispatchStructureReport? {
        try inspectTable(packagePath: packagePath, manifest: manifest, fileName: fileName)
    }
}
