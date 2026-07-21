import Foundation
import SmeltSchema

public enum SmeltCAMSourcePackageBuilderError: Error, CustomStringConvertible, Equatable {
    case malformed(String)

    public var description: String {
        switch self {
        case .malformed(let reason):
            return "module source package build: \(reason)"
        }
    }
}

public enum SmeltCAMSourcePackageBuilder {
  public struct BuildResult: Sendable, Equatable {
    public let packagePath: String
    public let generatedSwiftPath: String?
    public let metallibPath: String
    public let manifestPath: String
  }

    public static let buildCommandValueFlags: Set<String> = [
        "--weights-dir",
        "--shader-dir",
        "--output",
        "--trace-mode",
    "--source",
    ]

    public static let buildCommandBoolFlags: Set<String> = [
        "--module-source-package",
        "--optimizer-report",
    ]

    @discardableResult
    public static func build(
        camPath: String,
        outputDirectory: String,
        weightsDir: String?,
    sourceOverrides: [String: String] = [:],
        shaderDir: String,
        traceMode: SmeltTraceMode
  ) throws -> BuildResult {
        let camURL = URL(fileURLWithPath: camPath)
        let cam = try SmeltCAMIR.decodeModule(at: camURL)
    if cam.run != nil,
      cam.graphNodes.allSatisfy({ $0.implementation == .native || $0.implementation == .adapter })
    {
      let result = try SmeltComponentPackageBuilder.build(
        module: cam,
        sourceOverrides: sourceOverrides,
        shaderDirectory: shaderDir,
        outputDirectory: outputDirectory
      )
      return BuildResult(
        packagePath: result.packagePath,
        generatedSwiftPath: nil,
        metallibPath: result.metallibPath,
        manifestPath: result.manifestPath
      )
    }
        let ir = try SmeltCAMCheckedPackageProjector.sourceModelIR(cam: cam)
        let result = try SmeltCompiler.build(
            ir: ir,
            inputName: camURL.standardizedFileURL.path,
            sourceBaseDirectory: camURL.deletingLastPathComponent().path,
            outputDir: outputDirectory,
            weightsDir: weightsDir,
            shaderDir: shaderDir,
            traceMode: traceMode
        )
        try writePackageDescriptor(cam: cam, packagePath: result.packagePath)
    return BuildResult(
      packagePath: result.packagePath,
      generatedSwiftPath: result.generatedSwiftPath,
      metallibPath: result.metallibPath,
      manifestPath: result.manifestPath
    )
    }

    /// Source-built packages are ordinary module packages at runtime. Emit the
    /// same canonical descriptor as the checked-package projection so every
    /// command resolves exports and runtime construction through one path.
    static func writePackageDescriptor(cam: SmeltCAMIR, packagePath: String) throws {
        let data = try SmeltCAMPackageDescriptor(from: cam).canonicalJSONData()
        let destination = URL(fileURLWithPath: packagePath, isDirectory: true)
            .appendingPathComponent(SmeltCAMPackageDescriptor.packageFileName)
        try data.write(to: destination, options: .atomic)
    }

    public static func validateBuildCommandArguments(_ command: [String]) -> String? {
        var seenFlags: Set<String> = []
        var values: [String: String] = [:]
        var index = 3
        while index < command.count {
            let flag = command[index]
            if buildCommandBoolFlags.contains(flag) {
                if seenFlags.contains(flag) {
                    return "duplicate option for module source build: \(flag)"
                }
                seenFlags.insert(flag)
                index += 1
                continue
            }
            guard buildCommandValueFlags.contains(flag) else {
                if flag.hasPrefix("--") {
                    return "unsupported option for module source build: \(flag)"
                }
                return "unexpected argument for module source build: \(flag)"
            }
      if seenFlags.contains(flag), flag != "--source" {
                return "duplicate option for module source build: \(flag)"
            }
            seenFlags.insert(flag)
            guard index + 1 < command.count else {
                return "missing value for module source build option: \(flag)"
            }
            let value = command[index + 1]
            if value.isEmpty || value.hasPrefix("--") {
                return "missing value for module source build option: \(flag)"
            }
            values[flag] = value
            index += 2
        }
        for required in ["--module-source-package", "--output", "--shader-dir"] {
            if !seenFlags.contains(required) {
                return "missing required option for module source build: \(required)"
            }
        }
    if !seenFlags.contains("--weights-dir"), !seenFlags.contains("--source") {
      return "missing source input for module source build: pass --weights-dir or --source ID=PATH"
        }
        if let output = values["--output"],
      URL(fileURLWithPath: output).pathExtension == "smeltpkg"
    {
            return "--output is a parent directory; do not pass a final .smeltpkg package path"
        }
        return nil
    }
}
