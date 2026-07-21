// SmeltPackageSpecBuilder — temporary module projection package assembler.
//
// This is deliberately narrow: it assembles a package from already-materialized
// local files projected from checked CAM. Family builders still own heavy
// artifact generation until their config-built equivalence gates are recorded.

import Foundation
import CryptoKit
import SmeltSchema
#if canImport(Darwin)
import Darwin
#endif

public enum SmeltPackageSpecBuilderError: Error, CustomStringConvertible, Equatable {
    case malformed(String)
    case sourceArtifactMissing(String)

    public var description: String {
        switch self {
        case .malformed(let why): return "package spec build: \(why)"
        case .sourceArtifactMissing(let path):
            return "package spec build: source artifact missing: \(path)"
        }
    }
}

public enum SmeltPackageSpecBuilder {
    static let buildCommandValueFlags: Set<String> = [
        "--output",
        "--module-build-evidence-json",
    ]
    private static let ignoredCompilerByproductSourcePaths: Set<String> = [
        "SmeltGeneratedKernels.metal",
        "gptq_capture_points.json",
        "model.metalarchive",
        "trace_markers.json",
    ]

    public struct BuildResult: Sendable, Equatable {
        public let packagePath: String
        public let specPath: String
        public let specSHA256: String
        public let sourceRoot: String
        public let sourceManifestSHA256: String
        public let sourcePackageFiles: [String]
        public let sourcePayloadSHA256: String
        public let generatedPackageFiles: [String]
        public let generatedPayloadSHA256: String?
        public let packagePayloadSHA256: String
        public let resolvedPlanSignature: [String]
        public let resolvedPlanSignatureSHA256: String
        public let packageFiles: [String]

        public init(
            packagePath: String,
            specPath: String,
            specSHA256: String,
            sourceRoot: String,
            sourceManifestSHA256: String,
            sourcePackageFiles: [String],
            sourcePayloadSHA256: String,
            generatedPackageFiles: [String] = [],
            generatedPayloadSHA256: String? = nil,
            packagePayloadSHA256: String,
            resolvedPlanSignature: [String],
            resolvedPlanSignatureSHA256: String,
            packageFiles: [String]
        ) {
            self.packagePath = packagePath
            self.specPath = specPath
            self.specSHA256 = specSHA256
            self.sourceRoot = sourceRoot
            self.sourceManifestSHA256 = sourceManifestSHA256
            self.sourcePackageFiles = sourcePackageFiles
            self.sourcePayloadSHA256 = sourcePayloadSHA256
            self.generatedPackageFiles = generatedPackageFiles
            self.generatedPayloadSHA256 = generatedPayloadSHA256
            self.packagePayloadSHA256 = packagePayloadSHA256
            self.resolvedPlanSignature = resolvedPlanSignature
            self.resolvedPlanSignatureSHA256 = resolvedPlanSignatureSHA256
            self.packageFiles = packageFiles
        }
    }

    struct GeneratedPackageFile: Sendable, Equatable {
        let path: String
        let data: Data

        init(path: String, data: Data) {
            self.path = path
            self.data = data
        }
    }

    struct CAMBuildIdentity: Sendable, Equatable {
        let camPath: String
        let packageProjectionID: String
        let packageProjectionVersion: Int
        let camSemanticSHA256: String
        let exportABISHA256: String
        let descriptorVersion: Int
        let descriptorGraphSignatureSHA256: String
        let projectedPackageSpecSHA256: String

        init(
            camPath: String,
            packageProjectionID: String,
            packageProjectionVersion: Int,
            camSemanticSHA256: String,
            exportABISHA256: String,
            descriptorVersion: Int,
            descriptorGraphSignatureSHA256: String,
            projectedPackageSpecSHA256: String
        ) {
            self.camPath = camPath
            self.packageProjectionID = packageProjectionID
            self.packageProjectionVersion = packageProjectionVersion
            self.camSemanticSHA256 = camSemanticSHA256
            self.exportABISHA256 = exportABISHA256
            self.descriptorVersion = descriptorVersion
            self.descriptorGraphSignatureSHA256 = descriptorGraphSignatureSHA256
            self.projectedPackageSpecSHA256 = projectedPackageSpecSHA256
        }
    }

    private struct BuildEvidence: Encodable {
        struct PackageFile: Encodable {
            let path: String
            let roles: [String]
        }

        struct Runtime: Encodable {
            let architecture: String
        }

        let evidenceSchema: String
        let schema: Int
        let kind: String
        let generatedAt: String
        let command: [String]
        let toolSHA256: String
        let workingDirectory: String
        let packagePath: String
        let specPath: String
        let specSHA256: String
        let sourceRoot: String
        let sourceManifestSHA256: String
        let sourcePayloadSHA256: String
        let packagePayloadSHA256: String
        let resolvedPlanSignature: [String]
        let resolvedPlanSignatureSHA256: String
        let resolvedPlanPackageFiles: [PackageFile]
        let modelName: String
        let runtime: Runtime
        let packageFiles: [String]

        enum CodingKeys: String, CodingKey {
            case evidenceSchema = "evidence_schema"
            case schema
            case kind
            case generatedAt = "generated_at"
            case command
            case toolSHA256 = "tool_sha256"
            case workingDirectory = "working_directory"
            case packagePath = "package_path"
            case specPath = "spec_path"
            case specSHA256 = "spec_sha256"
            case sourceRoot = "source_root"
            case sourceManifestSHA256 = "source_manifest_sha256"
            case sourcePayloadSHA256 = "source_payload_sha256"
            case packagePayloadSHA256 = "package_payload_sha256"
            case resolvedPlanSignature = "resolved_plan_signature"
            case resolvedPlanSignatureSHA256 = "resolved_plan_signature_sha256"
            case resolvedPlanPackageFiles = "resolved_plan_package_files"
            case modelName = "model_name"
            case runtime
            case packageFiles = "package_files"
        }
    }

    private struct CAMBuildEvidence: Encodable {
        struct PackageFile: Encodable {
            let path: String
            let roles: [String]
        }

        let evidenceSchema: String
        let generatedAt: String
        let command: [String]
        let toolSHA256: String
        let workingDirectory: String
        let packagePath: String
        let camPath: String
        let packageProjectionID: String
        let packageProjectionVersion: Int
        let camSemanticSHA256: String
        let exportABISHA256: String
        let descriptorVersion: Int
        let descriptorGraphSignatureSHA256: String
        let projectedPackageSpecSHA256: String
        let sourceRoot: String
        let sourcePackageFiles: [String]
        let sourcePayloadSHA256: String
        let generatedPackageFiles: [String]
        let generatedPayloadSHA256: String
        let camDescriptorSHA256: String
        let packagePayloadSHA256: String
        let resolvedPlanSignature: [String]
        let resolvedPlanSignatureSHA256: String
        let resolvedPlanPackageFiles: [PackageFile]
        let packageFiles: [String]
        let buildElapsedMS: Int

        enum CodingKeys: String, CodingKey {
            case evidenceSchema = "evidence_schema"
            case generatedAt = "generated_at"
            case command
            case toolSHA256 = "tool_sha256"
            case workingDirectory = "working_directory"
            case packagePath = "package_path"
            case camPath = "module_path"
            case packageProjectionID = "package_projection_id"
            case packageProjectionVersion = "package_projection_version"
            case camSemanticSHA256 = "module_semantic_sha256"
            case exportABISHA256 = "export_abi_sha256"
            case descriptorVersion = "descriptor_version"
            case descriptorGraphSignatureSHA256 = "descriptor_graph_signature_sha256"
            case projectedPackageSpecSHA256 = "projected_package_spec_sha256"
            case sourceRoot = "source_root"
            case sourcePackageFiles = "source_package_files"
            case sourcePayloadSHA256 = "source_payload_sha256"
            case generatedPackageFiles = "generated_package_files"
            case generatedPayloadSHA256 = "generated_payload_sha256"
            case camDescriptorSHA256 = "module_descriptor_sha256"
            case packagePayloadSHA256 = "package_payload_sha256"
            case resolvedPlanSignature = "resolved_plan_signature"
            case resolvedPlanSignatureSHA256 = "resolved_plan_signature_sha256"
            case resolvedPlanPackageFiles = "resolved_plan_package_files"
            case packageFiles = "package_files"
            case buildElapsedMS = "build_elapsed_ms"
        }
    }

    @discardableResult
    static func build(
        specPath: String,
        outputDirectory: String,
        evidencePath: String? = nil,
        command: [String] = []
    ) throws -> BuildResult {
        if evidencePath != nil,
           let commandError = validateBuildCommandArguments(command, requireEvidenceFlag: true) {
            throw SmeltPackageSpecBuilderError.malformed(
                "invalid build evidence command: \(commandError)"
            )
        }
        let specURL = URL(fileURLWithPath: specPath)
        let data = try Data(contentsOf: specURL)
        let specSHA256 = sha256Hex(data)
        let spec = try SmeltPackageSpec.decode(from: data)
        let (result, plan) = try assemble(
            spec: spec,
            specPath: specURL.standardizedFileURL.path,
            specSHA256: specSHA256,
            sourceBaseURL: specURL.deletingLastPathComponent(),
            outputDirectory: outputDirectory
        )
        if let evidencePath {
            try writeEvidence(
                result,
                plan: plan,
                evidencePath: evidencePath,
                command: command
            )
        }
        return result
    }

    @discardableResult
    static func build(
        spec: SmeltPackageSpec,
        camIdentity: CAMBuildIdentity,
        sourceBaseDirectory: String,
        outputDirectory: String,
        generatedFiles: [GeneratedPackageFile] = [],
        evidencePath: String? = nil,
        command: [String] = []
    ) throws -> BuildResult {
        let start = Date()
        let sourceBaseURL = URL(fileURLWithPath: sourceBaseDirectory, isDirectory: true)
        let (result, plan) = try assemble(
            spec: spec,
            specPath: URL(fileURLWithPath: camIdentity.camPath).standardizedFileURL.path,
            specSHA256: camIdentity.projectedPackageSpecSHA256,
            sourceBaseURL: sourceBaseURL,
            outputDirectory: outputDirectory,
            generatedFiles: generatedFiles
        )
        if let evidencePath {
            try writeCAMEvidence(
                result,
                plan: plan,
                identity: camIdentity,
                evidencePath: evidencePath,
                command: command,
                buildElapsedMS: max(0, Int(Date().timeIntervalSince(start) * 1000))
            )
        }
        return result
    }

    static func validateBuildCommandArguments(
        _ command: [String],
        requireEvidenceFlag: Bool = false
    ) -> String? {
        var seenFlags: Set<String> = []
        var index = 3
        while index < command.count {
            let flag = command[index]
            guard buildCommandValueFlags.contains(flag) else {
                if flag.hasPrefix("--") {
                    return "unsupported option for module package spec JSON: \(flag)"
                }
                return "unexpected argument for module package spec JSON: \(flag)"
            }
            if seenFlags.contains(flag) {
                return "duplicate option for module package spec JSON: \(flag)"
            }
            seenFlags.insert(flag)
            guard index + 1 < command.count else {
                return "missing value for module package spec JSON option: \(flag)"
            }
            let value = command[index + 1]
            if value.isEmpty || value.hasPrefix("--") {
                return "missing value for module package spec JSON option: \(flag)"
            }
            index += 2
        }
        if !seenFlags.contains("--output") {
            return "missing required option for module package spec JSON: --output"
        }
        if requireEvidenceFlag, !seenFlags.contains("--module-build-evidence-json") {
            return "missing required option for module package spec JSON: --module-build-evidence-json"
        }
        return nil
    }

    private static func assemble(
        spec: SmeltPackageSpec,
        specPath: String,
        specSHA256: String,
        sourceBaseURL: URL,
        outputDirectory: String,
        generatedFiles: [GeneratedPackageFile] = []
    ) throws -> (BuildResult, SmeltPackageResolvedPlan) {
        let plan = try SmeltPackageResolvedPlan.resolve(spec)
        let sourceRoot = try localSourceRoot(for: plan, relativeTo: sourceBaseURL)
        let generatedPayloads = try validateGeneratedFiles(
            generatedFiles,
            plan: plan,
            sourceRoot: sourceRoot
        )
        let generatedPaths = Set(generatedPayloads.map(\.path))
        let sourcePackageFiles = plan.packageFiles.filter { !generatedPaths.contains($0.path) }
        try validateSourceInventory(
            packageFiles: sourcePackageFiles,
            sourceRoot: sourceRoot
        )
        try validateManifestAgreement(spec: spec, plan: plan, sourceRoot: sourceRoot)
        try validatePackageInterface(sourceRoot: sourceRoot)
        let sourcePayloadSHA256 = try sourcePayloadDigest(
            packageFiles: sourcePackageFiles,
            sourceRoot: sourceRoot
        )
        let generatedPayloadSHA256 = generatedPayloads.isEmpty
            ? nil
            : dataPayloadDigest(payloads: generatedPayloads)
        let sourceManifestSHA256 = try sha256Hex(
            ofFileAt: sourceRoot.appendingPathComponent("manifest.json").path
        )
        let resolvedPlanSignature = plan.signature.lines
        let resolvedPlanSignatureSHA256 = sha256Hex(
            Data(resolvedPlanSignature.joined(separator: "\n").utf8)
        )
        let payloads = try payloads(
            for: sourcePackageFiles,
            sourceRoot: sourceRoot
        ) + generatedPayloads.map {
            SmeltPackageAssembler.FilePayload(path: $0.path, body: .data($0.data))
        }

        try SmeltPackageAssembler.assemble(
            plan: plan,
            packagePath: outputDirectory,
            payloads: payloads
        )
        let packagePayloadSHA256 = try sourcePayloadDigest(
            packageFiles: plan.packageFiles,
            sourceRoot: URL(fileURLWithPath: outputDirectory, isDirectory: true)
        )
        let result = BuildResult(
            packagePath: URL(fileURLWithPath: outputDirectory, isDirectory: true)
                .standardizedFileURL
                .path,
            specPath: specPath,
            specSHA256: specSHA256,
            sourceRoot: sourceRoot.standardizedFileURL.path,
            sourceManifestSHA256: sourceManifestSHA256,
            sourcePackageFiles: sourcePackageFiles.map(\.path),
            sourcePayloadSHA256: sourcePayloadSHA256,
            generatedPackageFiles: generatedPayloads.map(\.path),
            generatedPayloadSHA256: generatedPayloadSHA256,
            packagePayloadSHA256: packagePayloadSHA256,
            resolvedPlanSignature: resolvedPlanSignature,
            resolvedPlanSignatureSHA256: resolvedPlanSignatureSHA256,
            packageFiles: plan.packageFiles.map(\.path)
        )
        return (result, plan)
    }

    private static func localSourceRoot(
        for plan: SmeltPackageResolvedPlan,
        specURL: URL
    ) throws -> URL {
        try localSourceRoot(for: plan, relativeTo: specURL.deletingLastPathComponent())
    }

    private static func localSourceRoot(
        for plan: SmeltPackageResolvedPlan,
        relativeTo baseURL: URL
    ) throws -> URL {
        let localSources = plan.sources.filter { $0.kind == .localDirectory }
        guard localSources.count == 1, let source = localSources.first else {
            throw SmeltPackageSpecBuilderError.malformed(
                "generic package-spec build requires exactly one local-directory source"
            )
        }
        guard source.revision == nil else {
            throw SmeltPackageSpecBuilderError.malformed(
                "generic package-spec build cannot resolve source revisions"
            )
        }
        let raw = source.locator
        let url = raw.hasPrefix("/")
            ? URL(fileURLWithPath: raw, isDirectory: true)
            : baseURL
                .appendingPathComponent(raw, isDirectory: true)
                .standardizedFileURL
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue
        else {
            throw SmeltPackageSpecBuilderError.sourceArtifactMissing(url.path)
        }
        return url
    }

    private static func validateSourceInventory(
        packageFiles: [SmeltPackageResolvedPlan.PackageFile],
        sourceRoot: URL
    ) throws {
        let root = sourceRoot.resolvingSymlinksInPath()
        let planned = Set(packageFiles.map(\.path))
        let opaqueSubtrees = Set(packageFiles.compactMap { file in
            file.roles.contains { $0.hasPrefix("sidecar:") } ? file.path : nil
        })
        var allowed = planned
        for path in planned {
            var parts = path.split(separator: "/").map(String.init)
            while parts.count > 1 {
                parts.removeLast()
                allowed.insert(parts.joined(separator: "/"))
            }
        }

        guard let enumerator = FileManager.default.enumerator(atPath: root.path) else {
            throw SmeltPackageSpecBuilderError.sourceArtifactMissing(root.path)
        }

        var undeclared: [String] = []
        for case let relative as String in enumerator {
            if ignoredCompilerByproductSourcePaths.contains(relative) {
                continue
            }
            if !allowed.contains(relative)
                && !opaqueSubtrees.contains(where: { relative.hasPrefix($0 + "/") }) {
                undeclared.append(relative)
                if undeclared.count >= 8 { break }
            }
        }

        guard undeclared.isEmpty else {
            throw SmeltPackageSpecBuilderError.malformed(
                "source has undeclared package file(s): "
                    + undeclared.sorted().joined(separator: ", ")
            )
        }
    }

    private static func validateGeneratedFiles(
        _ generatedFiles: [GeneratedPackageFile],
        plan: SmeltPackageResolvedPlan,
        sourceRoot: URL
    ) throws -> [GeneratedPackageFile] {
        let plannedByPath = Dictionary(uniqueKeysWithValues: plan.packageFiles.map {
            ($0.path, $0)
        })
        var byPath: [String: GeneratedPackageFile] = [:]
        for file in generatedFiles {
            try validateGeneratedPackagePath(file.path)
            guard let planned = plannedByPath[file.path] else {
                throw SmeltPackageSpecBuilderError.malformed(
                    "generated package file '\(file.path)' is not declared by resolved plan"
                )
            }
            guard !planned.roles.contains(where: { $0.hasPrefix("sidecar:") }) else {
                throw SmeltPackageSpecBuilderError.malformed(
                    "generated package file '\(file.path)' cannot replace a sidecar directory"
                )
            }
            guard byPath[file.path] == nil else {
                throw SmeltPackageSpecBuilderError.malformed(
                    "generated package file '\(file.path)' declared twice"
                )
            }
            let source = sourceRoot.appendingPathComponent(file.path)
            if FileManager.default.fileExists(atPath: source.path) {
                throw SmeltPackageSpecBuilderError.malformed(
                    "generated package file '\(file.path)' already exists in source root"
                )
            }
            byPath[file.path] = file
        }
        return plan.packageFiles.compactMap { byPath[$0.path] }
    }

    private static func validateGeneratedPackagePath(_ path: String) throws {
        guard !path.isEmpty,
              !path.hasPrefix("/"),
              !path.hasPrefix("~"),
              !path.contains("\\"),
              !path.contains("\0")
        else {
            throw SmeltPackageSpecBuilderError.malformed(
                "generated package file path is unsafe: \(path)"
            )
        }
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !parts.isEmpty,
              parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." })
        else {
            throw SmeltPackageSpecBuilderError.malformed(
                "generated package file path is unsafe: \(path)"
            )
        }
    }

    private struct PackageManifestSummary: Decodable {
        let kind: String?
        let architecture: String?
        let modelName: String?
        let blocks: SmeltBlockGraph?
        let loop: SmeltLoopSchedule?
        let validation: SmeltPackageSpec.Validation?
        let checksums: SmeltManifestChecksums?
        let hasInference: Bool
        let hasConfig: Bool
        let hasFiles: Bool
        let hasBuffers: Bool
        let hasSlotLayout: Bool
        let hasEosTokens: Bool
        let hasTokenizerFiles: Bool
        let hasStartup: Bool

        enum CodingKeys: String, CodingKey {
            case kind
            case architecture
            case modelName
            case blocks
            case loop
            case inference
            case validation
            case checksums
            case config
            case files
            case buffers
            case slotLayout
            case eosTokens
            case tokenizerFiles
            case startup
        }

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            kind = try container.decodeIfPresent(String.self, forKey: .kind)
            architecture = try container.decodeIfPresent(String.self, forKey: .architecture)
            modelName = try container.decodeIfPresent(String.self, forKey: .modelName)
            blocks = try container.decodeIfPresent(SmeltBlockGraph.self, forKey: .blocks)
            loop = try container.decodeIfPresent(SmeltLoopSchedule.self, forKey: .loop)
            validation = try container.decodeIfPresent(
                SmeltPackageSpec.Validation.self,
                forKey: .validation
            )
            checksums = try container.decodeIfPresent(SmeltManifestChecksums.self, forKey: .checksums)
            hasInference = container.contains(.inference)
            hasConfig = container.contains(.config)
            hasFiles = container.contains(.files)
            hasBuffers = container.contains(.buffers)
            hasSlotLayout = container.contains(.slotLayout)
            hasEosTokens = container.contains(.eosTokens)
            hasTokenizerFiles = container.contains(.tokenizerFiles)
            hasStartup = container.contains(.startup)
        }
    }

    private struct PackageManifestPolicyValidator {
        let accepts: @Sendable (PackageManifestSummary) -> Bool
        let validate: @Sendable (Data, SmeltPackageSpec, URL) throws -> Void
    }

    private static let packageManifestPolicyValidators: [PackageManifestPolicyValidator] = [
        .init(
            accepts: { $0.hasEosTokens || $0.hasTokenizerFiles },
            validate: { manifestData, spec, sourceRoot in
                try validateQwen3TTSManifestAgreement(
                    manifestData: manifestData,
                    spec: spec,
                    sourceRoot: sourceRoot
                )
            }
        ),
        .init(
            accepts: { $0.hasBuffers || $0.hasSlotLayout || $0.hasInference },
            validate: { manifestData, spec, _ in
                try validateTextManifestAgreement(
                    manifestData: manifestData,
                    spec: spec
                )
            }
        ),
    ]

    private static func validateManifestAgreement(
        spec: SmeltPackageSpec,
        plan: SmeltPackageResolvedPlan,
        sourceRoot: URL
    ) throws {
        let manifestURL = sourceRoot.appendingPathComponent("manifest.json")
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: manifestURL.path, isDirectory: &isDirectory),
              !isDirectory.boolValue
        else {
            throw SmeltPackageSpecBuilderError.sourceArtifactMissing(manifestURL.path)
        }

        let manifestData: Data
        let summary: PackageManifestSummary
        do {
            manifestData = try Data(contentsOf: manifestURL)
            summary = try JSONDecoder().decode(PackageManifestSummary.self, from: manifestData)
        } catch {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json cannot be decoded for package-spec preflight: \(error)"
            )
        }

        try validateManifestRuntimePolicy(summary.blocks, spec: spec)
        try validateManifestModelName(summary.modelName, plan: plan)
        guard summary.blocks == spec.blocks else {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json blocks disagree with package spec blocks"
            )
        }
        guard summary.loop == spec.loop else {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json loop disagrees with package spec loop"
            )
        }
        guard (summary.validation ?? .init()) == spec.validation else {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json validation policy disagrees with package spec validation policy"
            )
        }
        try validatePackageManifestPolicyAgreement(
            summary: summary,
            manifestData: manifestData,
            spec: spec,
            sourceRoot: sourceRoot
        )
        try validateDeclaredChecksums(
            summary.checksums,
            plan: plan,
            sourceRoot: sourceRoot
        )
    }

    private static func validateManifestModelName(
        _ modelName: String?,
        plan: SmeltPackageResolvedPlan
    ) throws {
        guard let modelName else { return }
        guard modelName == plan.modelName else {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json modelName '\(modelName)' disagrees with "
                    + "package spec model_name '\(plan.modelName)'"
            )
        }
    }

    private static func validateManifestRuntimePolicy(
        _ blocks: SmeltBlockGraph?,
        spec: SmeltPackageSpec
    ) throws {
        guard let blocks else {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json runtime graph is missing"
            )
        }
        let manifestPolicy: SmeltRuntimeGraphPolicy
        do {
            manifestPolicy = try SmeltRuntimeGraphPolicy.resolve(blocks: blocks)
        } catch {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json runtime graph policy could not be resolved: \(error)"
            )
        }
        let specPolicy = try SmeltRuntimeGraphPolicy.resolve(blocks: spec.blocks)
        guard manifestPolicy == specPolicy else {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json runtime graph policy '\(manifestPolicy.rawValue)' disagrees with "
                    + "package spec runtime graph policy '\(specPolicy.rawValue)'"
            )
        }
    }

    private static func validatePackageManifestPolicyAgreement(
        summary: PackageManifestSummary,
        manifestData: Data,
        spec: SmeltPackageSpec,
        sourceRoot: URL
    ) throws {
        let validators = packageManifestPolicyValidators.filter { $0.accepts(summary) }
        guard validators.count == 1, let validator = validators.first else {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json package policy shape is not recognized"
            )
        }
        try validator.validate(manifestData, spec, sourceRoot)
    }

    private static func validateTextManifestAgreement(
        manifestData: Data,
        spec: SmeltPackageSpec
    ) throws {
        let manifest: SmeltManifest
        do {
            manifest = try SmeltManifest.decode(from: manifestData)
        } catch {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json cannot be validated as text package policy: \(error)"
            )
        }
        if try manifestDeclaresRootArchitecture(manifestData) {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json text package must not declare root architecture"
            )
        }

        let manifestPolicy = try textArchitectureConfig(from: manifest)
        let (comparableManifestPolicy, comparableSpecPolicy) = normalizedTextArchitecturePoliciesForAgreement(
            manifestPolicy: manifestPolicy,
            specPolicy: spec.architectureConfig
        )
        guard comparableManifestPolicy == comparableSpecPolicy else {
            let mismatchSummary = architecturePolicyMismatchSummary(
                manifestPolicy: comparableManifestPolicy,
                specPolicy: comparableSpecPolicy
            )
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json text architecture_config policy disagrees with "
                    + "package spec architecture_config\(mismatchSummary)"
            )
        }

        guard let manifestInference = manifest.inference else {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json text inference policy is missing"
            )
        }
        guard let specInference = spec.inference else {
            throw SmeltPackageSpecBuilderError.malformed(
                "package spec text inference policy is missing"
            )
        }
        guard try packageSpecValue(manifestInference) == packageSpecValue(specInference) else {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json text inference policy disagrees with package spec inference policy"
            )
        }

        guard let manifestDecode = manifest.decode else {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json text decode policy is missing"
            )
        }
        guard let packageDecodePolicy = spec.decode else {
            throw SmeltPackageSpecBuilderError.malformed(
                "package spec text decode policy is missing"
            )
        }
        guard try packageSpecValue(manifestDecode) == packageSpecValue(packageDecodePolicy) else {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json text decode policy disagrees with package spec decode policy"
            )
        }
    }

    private static func validateQwen3TTSManifestAgreement(
        manifestData: Data,
        spec: SmeltPackageSpec,
        sourceRoot: URL
    ) throws {
        let manifest: Qwen3TTSManifest
        do {
            manifest = try Qwen3TTSManifest.decode(from: manifestData)
            try manifest.validateQwen3TTSValidation()
        } catch {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json cannot be validated as Qwen3-TTS package policy: \(error)"
            )
        }

        let manifestPolicy = try qwen3TTSArchitectureConfig(from: manifest)
        guard manifestPolicy == spec.architectureConfig else {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json Qwen3-TTS architecture_config policy disagrees with "
                    + "package spec architecture_config"
            )
        }
        try validateQwen3TTSTopLevelPolicy(manifest: manifest, spec: spec)
        try validateQwen3TTSSidecarPolicy(
            parentManifest: manifest,
            spec: spec,
            sourceRoot: sourceRoot
        )
    }

    private static func textArchitectureConfig(
        from manifest: SmeltManifest
    ) throws -> SmeltPackageSpecValue {
        guard let totalBytes = Int(exactly: manifest.weights.totalBytes) else {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json text total_bytes exceeds module integer range"
            )
        }
        var object: [String: SmeltPackageSpecValue] = [
            "hidden_size": .int(manifest.config.hiddenSize),
            "num_layers": .int(manifest.config.numLayers),
            "vocab_size": .int(manifest.config.vocabSize),
            "static_seq_capacity": .int(manifest.config.staticContextCapacity),
            "rope_dim": .int(manifest.config.ropeDim),
            "num_delta_layers": .int(manifest.config.numDeltaLayers),
            "num_attn_layers": .int(manifest.config.numAttnLayers),
            "ffn_dim": .int(manifest.config.ffnDim),
            "pipelines": .array(manifest.pipelines.map { .string($0) }),
            "weight_total_bytes": .int(totalBytes),
        ]
        if let options = manifest.buildProvenance?.resolvedOptions {
            var loading: [String: SmeltPackageSpecValue] = [
                "strategy": .string(options.loadingStrategy),
                "packing": .string(options.packing),
            ]
            if let checkpointMap = options.checkpointMap {
                loading["checkpoint_map"] = .string(checkpointMap)
            }
            object["loading"] = .object(loading)
        }
        if let hiddenActivation = manifest.config.hiddenActivation {
            object["hidden_activation"] = .string(hiddenActivation)
        }
        if let blockTopology = manifest.config.blockTopology {
            object["block_topology"] = .string(blockTopology)
        }
        if let layerPattern = manifest.config.layerPattern {
            object["layers"] = .object([
                "pattern": .array(layerPattern.pattern.map { .string($0) }),
                "repeats": .int(layerPattern.repeats),
            ])
        }
        if !manifest.config.attentionByRole.isEmpty {
            object["attention"] = .object(
                Dictionary(uniqueKeysWithValues: manifest.config.attentionByRole.map { role, attention in
                    (
                        role,
                        .object([
                            "q_heads": .int(attention.qHeads),
                            "kv_heads": .int(attention.kvHeads),
                            "head_dim": .int(attention.headDim),
                            "qk_norm": .bool(attention.qkNorm),
                            "v_norm": .bool(attention.vNorm),
                            "rope_theta": .number(attention.ropeTheta),
                            "rope_dim": .int(attention.ropeDim),
                            "sliding_window": .int(attention.slidingWindow),
                        ])
                    )
                })
            )
        }
        if let perLayerInput = manifest.config.perLayerInput {
            object["per_layer_input"] = .object([
                "hidden_size": .int(perLayerInput.hiddenSize),
                "vocab_size": .int(perLayerInput.vocabSize),
            ])
        }
        if let sharedKVLayers = manifest.config.sharedKVLayers, sharedKVLayers > 0 {
            object["shared_kv_layers"] = .int(sharedKVLayers)
        }
        if let logitCap = manifest.config.logitCap {
            object["logit_cap"] = .number(Double(logitCap))
        }
        if !manifest.config.turboQuantHPatterns.isEmpty {
            object["turbo_quant_h"] = .array(
                manifest.config.turboQuantHPatterns.map { .string($0) }
            )
        }
        if let prefill = manifest.prefill {
            object["prefill"] = .object([
                "engine": .string(prefill.engine),
                "model": .string(prefill.modelPath),
                "max_batch_size": .int(prefill.maxBatchSize),
                "handoff_entries": .int(prefill.handoff.entries.count),
            ])
        }
        return .object(object)
    }

    private static func manifestDeclaresRootArchitecture(_ manifestData: Data) throws -> Bool {
        let object = try JSONSerialization.jsonObject(with: manifestData)
        guard let manifest = object as? [String: Any] else {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json package policy must be a JSON object"
            )
        }
        return manifest.keys.contains("architecture")
    }

    private static func normalizedTextArchitecturePoliciesForAgreement(
        manifestPolicy: SmeltPackageSpecValue,
        specPolicy: SmeltPackageSpecValue
    ) -> (SmeltPackageSpecValue, SmeltPackageSpecValue) {
        guard case .object(var manifestObject) = manifestPolicy,
              case .object(let specObject) = specPolicy,
              specObject["pipelines"] == .array([])
        else {
            return (manifestPolicy, specPolicy)
        }

        // CAM-authored text specs leave generated Metal pipeline inventory empty:
        // pipeline names are package output, while the other architecture fields
        // are the authored/runtime policy we need to keep strict.
        manifestObject["pipelines"] = .array([])
        return (.object(manifestObject), specPolicy)
    }

    private static func architecturePolicyMismatchSummary(
        manifestPolicy: SmeltPackageSpecValue,
        specPolicy: SmeltPackageSpecValue
    ) -> String {
        let mismatches = packageSpecValueMismatches(
            manifestPolicy,
            specPolicy,
            prefix: ""
        )
        guard !mismatches.isEmpty else { return "" }
        let shown = mismatches.prefix(8).joined(separator: ", ")
        let remaining = mismatches.count - min(mismatches.count, 8)
        if remaining > 0 {
            return ": \(shown), +\(remaining) more"
        }
        return ": \(shown)"
    }

    private static func packageSpecValueMismatches(
        _ manifestValue: SmeltPackageSpecValue,
        _ specValue: SmeltPackageSpecValue,
        prefix: String
    ) -> [String] {
        if case .object(let manifestObject) = manifestValue,
           case .object(let specObject) = specValue
        {
            let keys = Set(manifestObject.keys).union(specObject.keys).sorted()
            return keys.flatMap { key -> [String] in
                let path = prefix.isEmpty ? key : "\(prefix).\(key)"
                guard let manifestChild = manifestObject[key] else {
                    return ["\(path) manifest=<missing> spec=\(shortPackageSpecValue(specObject[key]))"]
                }
                guard let specChild = specObject[key] else {
                    return ["\(path) manifest=\(shortPackageSpecValue(manifestChild)) spec=<missing>"]
                }
                if manifestChild == specChild {
                    return []
                }
                return packageSpecValueMismatches(
                    manifestChild,
                    specChild,
                    prefix: path
                )
            }
        }
        return [
            "\(prefix) manifest=\(shortPackageSpecValue(manifestValue)) "
                + "spec=\(shortPackageSpecValue(specValue))"
        ]
    }

    private static func shortPackageSpecValue(_ value: SmeltPackageSpecValue?) -> String {
        guard let value else { return "<missing>" }
        switch value {
        case .string(let string):
            return "\"\(string)\""
        case .int(let int):
            return "\(int)"
        case .number(let double):
            return "\(double)"
        case .bool(let bool):
            return "\(bool)"
        case .array(let array):
            return "[\(array.count) items]"
        case .object(let object):
            return "{\(object.count) keys}"
        case .null:
            return "null"
        }
    }

    private static func validateQwen3TTSTopLevelPolicy(
        manifest: Qwen3TTSManifest,
        spec: SmeltPackageSpec
    ) throws {
        let profile = SmeltQwen3TTSPackageProfiles.runnable
        try requireTokenizerPolicy(
            spec.tokenizer,
            format: "byte-bpe",
            files: manifest.tokenizerFiles ?? [],
            family: "Qwen3-TTS"
        )
        guard let inference = spec.inference else {
            throw SmeltPackageSpecBuilderError.malformed(
                "package spec Qwen3-TTS inference policy is missing"
            )
        }
        guard inference.maxTokens == profile.maxTokens,
              inference.eosTokens == manifest.eosTokens
        else {
            throw SmeltPackageSpecBuilderError.malformed(
                "package spec Qwen3-TTS inference policy disagrees with manifest policy"
            )
        }
        let manifestDecode = try qwen3TTSDecodePolicy(from: manifest.decode)
        guard let packageDecodePolicy = spec.decode,
              try packageSpecValue(manifestDecode) == packageSpecValue(packageDecodePolicy)
        else {
            throw SmeltPackageSpecBuilderError.malformed(
                "package spec Qwen3-TTS decode policy disagrees with manifest policy"
            )
        }
    }

    private static func requireTokenizerPolicy(
        _ tokenizer: SmeltPackageSpec.Tokenizer?,
        format: String,
        files: [String],
        family: String
    ) throws {
        guard let tokenizer,
              tokenizer.format == format,
              tokenizer.files == files
        else {
            throw SmeltPackageSpecBuilderError.malformed(
                "package spec \(family) tokenizer policy disagrees with manifest policy"
            )
        }
    }

    private static func qwen3TTSDecodePolicy(
        from decode: Qwen3TTSManifest.Decode?
    ) throws -> SmeltPackageSpec.DecodePolicy {
        guard let decode else {
            return .init(sampler: .init(mode: .greedy))
        }
        if decode.doSample {
            guard decode.temperature > 0, decode.topK > 0,
                  decode.subtalkerTemperature > 0, decode.subtalkerTopK > 0
            else {
                throw SmeltPackageSpecBuilderError.malformed(
                    "manifest.json Qwen3-TTS decode policy has invalid sampling values"
                )
            }
        }
        return .init(
            sampler: .init(
                mode: decode.doSample ? .sample : .greedy,
                temperature: decode.doSample ? Double(decode.temperature) : nil,
                topK: decode.doSample ? decode.topK : nil
            ),
            subSampler: decode.doSample ? .init(
                mode: .sample,
                temperature: Double(decode.subtalkerTemperature),
                topK: decode.subtalkerTopK
            ) : nil
        )
    }

    private static func validateQwen3TTSSidecarPolicy(
        parentManifest: Qwen3TTSManifest,
        spec: SmeltPackageSpec,
        sourceRoot: URL
    ) throws {
        let expectedProfiles = SmeltPackageSidecarProfiles.qwen3TTSRunnableHeadlessTrunks
        let expectedIDs = Set(expectedProfiles.map(\.id))
        let declaredIDs = Set(spec.sidecars.map(\.id))
        guard declaredIDs == expectedIDs else {
            throw SmeltPackageSpecBuilderError.malformed(
                "package spec Qwen3-TTS sidecars disagree with runnable sidecar policy"
            )
        }

        let sidecarsByID = Dictionary(uniqueKeysWithValues: spec.sidecars.map {
            ($0.id, $0)
        })
        for expected in expectedProfiles {
            guard let sidecar = sidecarsByID[expected.id],
                  sidecar.path == expected.path,
                  sidecar.kind == expected.kind
            else {
                throw SmeltPackageSpecBuilderError.malformed(
                    "package spec Qwen3-TTS sidecar '\(expected.id)' must be path "
                        + "'\(expected.path)' and kind '\(expected.kind)'"
                )
            }
            try validateQwen3TTSSidecarPayload(
                parentManifest: parentManifest,
                expected: expected,
                sourceRoot: sourceRoot
            )
        }
    }

    private static func validateQwen3TTSSidecarPayload(
        parentManifest: Qwen3TTSManifest,
        expected: SmeltHeadlessTrunkSidecarProfile,
        sourceRoot: URL
    ) throws {
        let sidecarRoot = sourceRoot.appendingPathComponent(
            expected.path,
            isDirectory: true
        )
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(
            atPath: sidecarRoot.path,
            isDirectory: &isDirectory
        ), isDirectory.boolValue else {
            throw SmeltPackageSpecBuilderError.sourceArtifactMissing(sidecarRoot.path)
        }

        let requiredFiles = [
            "manifest.json",
            "dispatches.bin",
            "prefill_dispatches.bin",
            "SmeltGenerated.swift",
            "weights.bin",
            "model.metallib",
        ]
        for file in requiredFiles {
            let url = sidecarRoot.appendingPathComponent(file)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw SmeltPackageSpecBuilderError.sourceArtifactMissing(url.path)
            }
        }

        let manifestURL = sidecarRoot.appendingPathComponent("manifest.json")
        let sidecarManifest: SmeltManifest
        do {
            sidecarManifest = try SmeltManifest.decode(from: Data(contentsOf: manifestURL))
        } catch {
            throw SmeltPackageSpecBuilderError.malformed(
                "Qwen3-TTS sidecar '\(expected.path)' manifest.json "
                    + "cannot be validated: \(error)"
            )
        }

        guard SmeltManifest.isHeadlessTrunk(headlessTrunkABI: sidecarManifest.headlessTrunkABI) else {
            throw SmeltPackageSpecBuilderError.malformed(
                "Qwen3-TTS sidecar '\(expected.path)' manifest is not a headless trunk"
            )
        }
        guard sidecarManifest.blocks == nil, sidecarManifest.loop == nil else {
            throw SmeltPackageSpecBuilderError.malformed(
                "Qwen3-TTS sidecar '\(expected.path)' must not declare runnable graph policy"
            )
        }
        guard sidecarManifest.modelName == expected.modelName else {
            throw SmeltPackageSpecBuilderError.malformed(
                "Qwen3-TTS sidecar '\(expected.path)' modelName "
                    + "'\(sidecarManifest.modelName)' disagrees with expected "
                    + "'\(expected.modelName)'"
            )
        }
        guard sidecarManifest.weights.totalBytes == parentManifest.totalBytes else {
            throw SmeltPackageSpecBuilderError.malformed(
                "Qwen3-TTS sidecar '\(expected.path)' shared weight byte count "
                    + "disagrees with parent manifest"
            )
        }
        guard sidecarManifest.inference == nil, sidecarManifest.decode == nil else {
            throw SmeltPackageSpecBuilderError.malformed(
                "Qwen3-TTS sidecar '\(expected.path)' must not declare runnable inference/decode policy"
            )
        }
        guard sidecarManifest.prefill != nil else {
            throw SmeltPackageSpecBuilderError.malformed(
                "Qwen3-TTS sidecar '\(expected.path)' prefill policy is missing"
            )
        }

        try validateQwen3TTSSidecarSharedLink(
            "weights.bin",
            sidecarRoot: sidecarRoot,
            sidecarPath: expected.path,
            sourceRoot: sourceRoot
        )
        try validateQwen3TTSSidecarSharedLink(
            "model.metallib",
            sidecarRoot: sidecarRoot,
            sidecarPath: expected.path,
            sourceRoot: sourceRoot
        )
    }

    private static func validateQwen3TTSSidecarSharedLink(
        _ file: String,
        sidecarRoot: URL,
        sidecarPath: String,
        sourceRoot: URL
    ) throws {
        let url = sidecarRoot.appendingPathComponent(file)
        guard try isSymlink(url) else {
            throw SmeltPackageSpecBuilderError.malformed(
                "Qwen3-TTS sidecar '\(sidecarPath)' shared \(file) "
                    + "must be a package-internal symlink"
            )
        }
        let target = try validateSidecarSymlink(
            url,
            sidecarPath: sidecarPath,
            relativePath: file,
            sourceRoot: sourceRoot
        )
        guard sourceRelativePath(target, under: sourceRoot) == file else {
            throw SmeltPackageSpecBuilderError.malformed(
                "Qwen3-TTS sidecar '\(sidecarPath)' shared \(file) "
                    + "must resolve to parent \(file)"
            )
        }
    }

    private static func qwen3TTSArchitectureConfig(
        from manifest: Qwen3TTSManifest
    ) throws -> SmeltPackageSpecValue {
        guard let totalBytes = Int(exactly: manifest.totalBytes) else {
            throw SmeltPackageSpecBuilderError.malformed(
                "manifest.json Qwen3-TTS total_bytes exceeds module integer range"
            )
        }
        var object: [String: SmeltPackageSpecValue] = [
            "architecture": .string(SmeltQwen3TTSPackageProfiles.runnable.runtimeArchitecture),
            "model_name": .string(manifest.modelName),
            "page_size": .int(manifest.pageSize),
            "total_bytes": .int(totalBytes),
            "pipelines": .array(manifest.pipelines.map { .string($0) }),
            "eos_tokens": .array(manifest.eosTokens.map { .int(Int($0)) }),
            "tokenizer_files": .array((manifest.tokenizerFiles ?? []).map { .string($0) }),
            "sidecars": .array(
                SmeltQwen3TTSPackageProfiles.runnable.sidecarPaths.map { .string($0) }
            ),
            "weight_layout": try packageSpecValue(manifest.weights),
        ]
        if let decode = manifest.decode {
            object["decode"] = try packageSpecValue(decode)
        }
        return .object(object)
    }

    private static func packageSpecValue<T: Encodable>(_ value: T) throws -> SmeltPackageSpecValue {
        do {
            let data = try JSONEncoder().encode(value)
            return try JSONDecoder().decode(SmeltPackageSpecValue.self, from: data)
        } catch {
            throw SmeltPackageSpecBuilderError.malformed(
                "could not project manifest policy into module architecture_config: \(error)"
            )
        }
    }

    private static func validateDeclaredChecksums(
        _ checksums: SmeltManifestChecksums?,
        plan: SmeltPackageResolvedPlan,
        sourceRoot: URL
    ) throws {
        guard let checksums else { return }
        let declared = [
            "weights.bin": checksums.weightsBin,
            "model.metallib": checksums.metallib,
            "SmeltGenerated.swift": checksums.generatedSwift,
            "dispatches.bin": checksums.dispatchesBin,
            "prefill_dispatches.bin": checksums.prefillDispatchesBin,
            "prefill_verify_argmax_dispatches.bin": checksums.prefillVerifyArgmaxDispatchesBin,
            "tokenizer.json": checksums.tokenizerJSON,
        ].compactMapValues { value -> String? in
            guard let value, !value.isEmpty else { return nil }
            return value
        }
        let planned = Set(plan.packageFiles.map(\.path))
        for path in declared.keys.sorted() where planned.contains(path) {
            let source = sourceRoot.appendingPathComponent(path)
            let actual = try sha256Hex(ofFileAt: source.path)
            guard actual == declared[path] else {
                throw SmeltPackageSpecBuilderError.malformed(
                    "manifest.json checksum for '\(path)' disagrees with source bytes"
                )
            }
        }
    }

    private static func validatePackageInterface(sourceRoot: URL) throws {
        do {
            guard let interface = try SmeltPackageInterface.load(
                packagePath: sourceRoot.path
            ) else { return }
            let manifestData = try Data(
                contentsOf: sourceRoot.appendingPathComponent("manifest.json")
            )
            let context = try SmeltPackageInterface.packageValidationContext(
                manifestData: manifestData
            )
            try interface.validate(packageContext: context)
        } catch {
            throw SmeltPackageSpecBuilderError.malformed(
                "args.json cannot be validated for package-spec preflight: \(error)"
            )
        }
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

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func sourcePayloadDigest(
        packageFiles: [SmeltPackageResolvedPlan.PackageFile],
        sourceRoot: URL
    ) throws -> String {
        let resolvedSourceRoot = sourceRoot.resolvingSymlinksInPath().standardizedFileURL
        var lines: [String] = []
        for file in packageFiles {
            let source = sourceRoot.appendingPathComponent(file.path)
            if file.roles.contains(where: { $0.hasPrefix("sidecar:") }) {
                lines.append("directory:\(file.path)")
                lines += try directoryDigestLines(
                    root: source,
                    packagePath: file.path,
                    sourceRoot: resolvedSourceRoot
                )
            } else {
                let digest = try plannedFileDigest(
                    at: source,
                    packagePath: file.path,
                    sourceRoot: resolvedSourceRoot
                )
                lines.append("file:\(file.path):\(digest)")
            }
        }
        return sha256Hex(Data((lines.sorted().joined(separator: "\n") + "\n").utf8))
    }

    private static func dataPayloadDigest(payloads: [GeneratedPackageFile]) -> String {
        let lines = payloads.map {
            "file:\($0.path):\(sha256Hex($0.data))"
        }
        return sha256Hex(Data((lines.sorted().joined(separator: "\n") + "\n").utf8))
    }

    private static func plannedFileDigest(
        at url: URL,
        packagePath: String,
        sourceRoot: URL
    ) throws -> String {
        try rejectSymlinkFile(at: url, packagePath: packagePath, sourceRoot: sourceRoot)
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              !isDirectory.boolValue
        else {
            throw SmeltPackageSpecBuilderError.sourceArtifactMissing(url.path)
        }
        return try sha256Hex(ofFileAt: url.path)
    }

    private static func directoryDigestLines(
        root: URL,
        packagePath: String,
        sourceRoot: URL
    ) throws -> [String] {
        try rejectSymlinkDirectory(at: root, packagePath: packagePath, sourceRoot: sourceRoot)
        guard let enumerator = FileManager.default.enumerator(atPath: root.path) else {
            throw SmeltPackageSpecBuilderError.sourceArtifactMissing(root.path)
        }

        var lines: [String] = []
        for case let relative as String in enumerator {
            let source = root.appendingPathComponent(relative)
            if try isSymlink(source) {
                let target = try validateSidecarSymlink(
                    source,
                    sidecarPath: packagePath,
                    relativePath: relative,
                    sourceRoot: sourceRoot
                )
                let targetRelative = sourceRelativePath(target, under: sourceRoot)
                lines.append("symlink:\(packagePath)/\(relative):\(targetRelative)")
                lines.append(
                    "file:\(packagePath)/\(relative):\(try sha256Hex(ofFileAt: target.path))"
                )
                continue
            }
            var isDirectory = ObjCBool(false)
            guard FileManager.default.fileExists(atPath: source.path, isDirectory: &isDirectory)
            else {
                throw SmeltPackageSpecBuilderError.sourceArtifactMissing(source.path)
            }
            if isDirectory.boolValue { continue }
            lines.append(
                "file:\(packagePath)/\(relative):\(try sha256Hex(ofFileAt: source.path))"
            )
        }
        return lines.sorted()
    }

    private static func rejectSymlinkFile(
        at url: URL,
        packagePath: String,
        sourceRoot: URL
    ) throws {
        guard try isSymlink(url) else { return }
        _ = try validateSymlinkTarget(
            url,
            sourceRoot: sourceRoot,
            context: "package file '\(packagePath)'"
        )
        throw SmeltPackageSpecBuilderError.malformed(
            "package file '\(packagePath)' is a symlink; materialize file payloads before module assembly"
        )
    }

    private static func rejectSymlinkDirectory(
        at url: URL,
        packagePath: String,
        sourceRoot: URL
    ) throws {
        guard try isSymlink(url) else { return }
        _ = try validateSymlinkTarget(
            url,
            sourceRoot: sourceRoot,
            context: "sidecar '\(packagePath)'"
        )
        throw SmeltPackageSpecBuilderError.malformed(
            "sidecar '\(packagePath)' is a symlink; materialize sidecar directories before module assembly"
        )
    }

    private static func validateSidecarSymlink(
        _ url: URL,
        sidecarPath: String,
        relativePath: String,
        sourceRoot: URL
    ) throws -> URL {
        let target = try validateSymlinkTarget(
            url,
            sourceRoot: sourceRoot,
            context: "sidecar '\(sidecarPath)' symlink '\(relativePath)'"
        )
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: target.path, isDirectory: &isDirectory) else {
            throw SmeltPackageSpecBuilderError.sourceArtifactMissing(target.path)
        }
        guard !isDirectory.boolValue else {
            throw SmeltPackageSpecBuilderError.malformed(
                "sidecar '\(sidecarPath)' symlink '\(relativePath)' resolves to a directory"
            )
        }
        return target
    }

    private static func validateSymlinkTarget(
        _ url: URL,
        sourceRoot: URL,
        context: String
    ) throws -> URL {
        let target = url.resolvingSymlinksInPath().standardizedFileURL
        guard FileManager.default.fileExists(atPath: target.path) else {
            throw SmeltPackageSpecBuilderError.sourceArtifactMissing(target.path)
        }
        guard isWithinSourceRoot(target, sourceRoot: sourceRoot) else {
            throw SmeltPackageSpecBuilderError.malformed(
                "\(context) resolves outside source root: \(target.path)"
            )
        }
        return target
    }

    private static func isSymlink(_ url: URL) throws -> Bool {
        #if canImport(Darwin)
        var info = stat()
        guard lstat(url.path, &info) == 0 else {
            if errno == ENOENT {
                return false
            }
            throw SmeltPackageSpecBuilderError.malformed(
                "could not inspect source path for symlink: \(url.path)"
            )
        }
        return (info.st_mode & S_IFMT) == S_IFLNK
        #else
        do {
            let values = try url.resourceValues(forKeys: [.isSymbolicLinkKey])
            return values.isSymbolicLink == true
        } catch {
            return false
        }
        #endif
    }

    private static func isWithinSourceRoot(_ url: URL, sourceRoot: URL) -> Bool {
        let rootPath = sourceRoot.standardizedFileURL.path
        let path = url.standardizedFileURL.path
        return path == rootPath || path.hasPrefix(rootPath + "/")
    }

    private static func sourceRelativePath(_ url: URL, under sourceRoot: URL) -> String {
        let rootPath = sourceRoot.standardizedFileURL.path
        let path = url.standardizedFileURL.path
        guard path != rootPath else { return "." }
        return String(path.dropFirst(rootPath.count + 1))
    }

    private static func writeEvidence(
        _ result: BuildResult,
        plan: SmeltPackageResolvedPlan,
        evidencePath: String,
        command: [String]
    ) throws {
        let evidence = BuildEvidence(
            evidenceSchema: "smelt.package_spec.build_evidence.v1",
            schema: 1,
            kind: "smelt.module.package_spec_build_evidence",
            generatedAt: ISO8601DateFormatter().string(from: Date()),
            command: command,
            toolSHA256: try commandToolSHA256(command),
            workingDirectory: URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .standardizedFileURL
                .path,
            packagePath: result.packagePath,
            specPath: result.specPath,
            specSHA256: result.specSHA256,
            sourceRoot: result.sourceRoot,
            sourceManifestSHA256: result.sourceManifestSHA256,
            sourcePayloadSHA256: result.sourcePayloadSHA256,
            packagePayloadSHA256: result.packagePayloadSHA256,
            resolvedPlanSignature: result.resolvedPlanSignature,
            resolvedPlanSignatureSHA256: result.resolvedPlanSignatureSHA256,
            resolvedPlanPackageFiles: plan.packageFiles.map {
                .init(path: $0.path, roles: $0.roles)
            },
            modelName: plan.modelName,
            runtime: .init(
                architecture: plan.runtime.architecture
            ),
            packageFiles: result.packageFiles
        )
        let outputURL = URL(fileURLWithPath: evidencePath)
        let parent = outputURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(evidence).write(to: outputURL, options: .atomic)
    }

    private static func writeCAMEvidence(
        _ result: BuildResult,
        plan: SmeltPackageResolvedPlan,
        identity: CAMBuildIdentity,
        evidencePath: String,
        command: [String],
        buildElapsedMS: Int
    ) throws {
        guard let generatedPayloadSHA256 = result.generatedPayloadSHA256 else {
            throw SmeltPackageSpecBuilderError.malformed(
                "module build evidence v3 requires generated package files"
            )
        }
        guard let camDescriptorSHA256 = try camDescriptorSHA256(in: result) else {
            throw SmeltPackageSpecBuilderError.malformed(
                "module build evidence v3 requires generated \(SmeltCAMPackageDescriptor.packageFileName)"
            )
        }
        let evidence = CAMBuildEvidence(
            evidenceSchema: "smelt.module.build_evidence.v3",
            generatedAt: ISO8601DateFormatter().string(from: Date()),
            command: command,
            toolSHA256: try commandToolSHA256(command),
            workingDirectory: URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .standardizedFileURL
                .path,
            packagePath: result.packagePath,
            camPath: URL(fileURLWithPath: identity.camPath).standardizedFileURL.path,
            packageProjectionID: identity.packageProjectionID,
            packageProjectionVersion: identity.packageProjectionVersion,
            camSemanticSHA256: identity.camSemanticSHA256,
            exportABISHA256: identity.exportABISHA256,
            descriptorVersion: identity.descriptorVersion,
            descriptorGraphSignatureSHA256: identity.descriptorGraphSignatureSHA256,
            projectedPackageSpecSHA256: identity.projectedPackageSpecSHA256,
            sourceRoot: result.sourceRoot,
            sourcePackageFiles: result.sourcePackageFiles,
            sourcePayloadSHA256: result.sourcePayloadSHA256,
            generatedPackageFiles: result.generatedPackageFiles,
            generatedPayloadSHA256: generatedPayloadSHA256,
            camDescriptorSHA256: camDescriptorSHA256,
            packagePayloadSHA256: result.packagePayloadSHA256,
            resolvedPlanSignature: result.resolvedPlanSignature,
            resolvedPlanSignatureSHA256: result.resolvedPlanSignatureSHA256,
            resolvedPlanPackageFiles: plan.packageFiles.map {
                .init(path: $0.path, roles: $0.roles)
            },
            packageFiles: result.packageFiles,
            buildElapsedMS: buildElapsedMS
        )
        let outputURL = URL(fileURLWithPath: evidencePath)
        let parent = outputURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(evidence).write(to: outputURL, options: .atomic)
    }

    private static func camDescriptorSHA256(in result: BuildResult) throws -> String? {
        guard result.generatedPackageFiles.contains(SmeltCAMPackageDescriptor.packageFileName)
        else {
            return nil
        }
        let url = URL(fileURLWithPath: result.packagePath, isDirectory: true)
            .appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName)
        return try sha256Hex(ofFileAt: url.path)
    }

    private static func commandToolSHA256(_ command: [String]) throws -> String {
        guard let toolPath = command.first else {
            throw SmeltPackageSpecBuilderError.malformed(
                "build evidence requires a command tool"
            )
        }
        let url = toolPath.hasPrefix("/")
            ? URL(fileURLWithPath: toolPath)
            : URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(toolPath)
                .standardizedFileURL
        return try sha256Hex(ofFileAt: url.path)
    }

    private static func payloads(
        for packageFiles: [SmeltPackageResolvedPlan.PackageFile],
        sourceRoot: URL
    ) throws -> [SmeltPackageAssembler.FilePayload] {
        try packageFiles.map { file in
            if file.roles.contains(where: { $0.hasPrefix("sidecar:") }) {
                let source = sourceRoot.appendingPathComponent(file.path, isDirectory: true)
                var isDirectory = ObjCBool(false)
                guard FileManager.default.fileExists(atPath: source.path, isDirectory: &isDirectory),
                      isDirectory.boolValue
                else {
                    throw SmeltPackageSpecBuilderError.sourceArtifactMissing(source.path)
                }
                return SmeltPackageAssembler.FilePayload(
                    path: file.path,
                    body: .copyDirectory(source.path)
                )
            }
            let source = sourceRoot.appendingPathComponent(file.path)
            var isDirectory = ObjCBool(false)
            guard FileManager.default.fileExists(atPath: source.path, isDirectory: &isDirectory),
                  !isDirectory.boolValue
            else {
                throw SmeltPackageSpecBuilderError.sourceArtifactMissing(source.path)
            }
            return SmeltPackageAssembler.FilePayload(
                path: file.path,
                body: file.roles.contains(where: {
                    $0.hasPrefix("artifact:") && $0.hasSuffix(":weights")
                }) ? .sharedFile(source.path) : .copyFile(source.path)
            )
        }
    }
}
