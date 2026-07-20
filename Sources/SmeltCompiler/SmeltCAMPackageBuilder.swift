import Foundation
import SmeltSchema

public enum SmeltCAMPackageBuilderError: Error, CustomStringConvertible, Equatable {
    case malformed(String)

    public var description: String {
        switch self {
        case .malformed(let reason):
            return "module package build: \(reason)"
        }
    }
}

public enum SmeltCAMPackageBuilder {
    public static let buildCommandValueFlags: Set<String> = [
        "--output",
        "--module-artifact-root",
        "--module-build-evidence-json",
    ]

    @discardableResult
    public static func build(
        camPath: String,
        artifactRoot: String,
        outputDirectory: String,
        evidencePath: String,
        command: [String] = []
    ) throws -> SmeltPackageSpecBuilder.BuildResult {
        let artifactRootURL = URL(fileURLWithPath: artifactRoot, isDirectory: true)
            .standardizedFileURL
            .resolvingSymlinksInPath()
        let outputDirectoryURL = URL(fileURLWithPath: outputDirectory, isDirectory: true)
            .standardizedFileURL
            .resolvingSymlinksInPath()
        guard artifactRootURL.path != outputDirectoryURL.path else {
            throw SmeltCAMPackageBuilderError.malformed(
                "--module-artifact-root and --output must be distinct paths"
            )
        }
        let camURL = URL(fileURLWithPath: camPath)
        let cam = try SmeltCAMIR.decodeModule(at: camURL)
        let projection = try SmeltCAMCheckedPackageProjector.project(
            cam: cam,
            artifactRoot: artifactRoot
        )
        guard projection.buildCommandCovered else {
            throw SmeltCAMPackageBuilderError.malformed(
                "module projection '\(projection.packageProjectionID)' is checked-package-projected "
                    + "but not build-command-covered"
            )
        }
        let descriptorData = try SmeltCAMPackageDescriptor(from: cam).canonicalJSONData()
        let identity = SmeltPackageSpecBuilder.CAMBuildIdentity(
            camPath: camURL.standardizedFileURL.path,
            packageProjectionID: projection.packageProjectionID,
            packageProjectionVersion: projection.packageProjectionVersion,
            camSemanticSHA256: projection.camSemanticSHA256,
            exportABISHA256: projection.exportABISHA256,
            descriptorVersion: projection.descriptorVersion,
            descriptorGraphSignatureSHA256: projection.descriptorGraphSignatureSHA256,
            projectedPackageSpecSHA256: projection.projectedPackageSpecSHA256
        )
        return try SmeltPackageSpecBuilder.build(
            spec: projection.spec,
            camIdentity: identity,
            sourceBaseDirectory: FileManager.default.currentDirectoryPath,
            outputDirectory: outputDirectory,
            generatedFiles: [
                .init(
                    path: SmeltCAMPackageDescriptor.packageFileName,
                    data: descriptorData
                ),
            ],
            evidencePath: evidencePath,
            command: command
        )
    }

    public static func validateBuildCommandArguments(_ command: [String]) -> String? {
        var seenFlags: Set<String> = []
        var index = 3
        while index < command.count {
            let flag = command[index]
            guard buildCommandValueFlags.contains(flag) else {
                if flag.hasPrefix("--") {
                    return "unsupported option for module build: \(flag)"
                }
                return "unexpected argument for module build: \(flag)"
            }
            if seenFlags.contains(flag) {
                return "duplicate option for module build: \(flag)"
            }
            seenFlags.insert(flag)
            guard index + 1 < command.count else {
                return "missing value for module build option: \(flag)"
            }
            let value = command[index + 1]
            if value.isEmpty || value.hasPrefix("--") {
                return "missing value for module build option: \(flag)"
            }
            index += 2
        }
        for required in ["--output", "--module-artifact-root", "--module-build-evidence-json"] {
            if !seenFlags.contains(required) {
                return "missing required option for module build: \(required)"
            }
        }
        return nil
    }
}
