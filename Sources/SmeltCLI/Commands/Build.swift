import Foundation
import SmeltCompiler
import SmeltRuntime
import SmeltSchema

private struct BuildInputHeader: Decodable {
    let kind: String?
    let version: Int?
    let packageName: String?

    enum CodingKeys: String, CodingKey {
        case kind
        case version
        case packageName = "package_name"
    }

    /// True when the decoded JSON carries internal package-spec fields
    /// (`version` / `package_name` / `kind`) rather than being an authored
    /// `.module.json` module. Kept as a member so the field access stays bare
    /// and this input-format probe never reads like a runtime kind/architecture
    /// dispatch selector (check-module-selector-deletion `manifest-kind-architecture-gate`).
    var looksLikeInternalPackageSpec: Bool {
        version != nil || packageName != nil || kind != nil
    }
}

func runBuildCommand() {
    guard args.count >= 3 else {
    fputs(
      "Usage: smelt build <model.module.json> [--module-source-package] [--weights-dir DIR | --source ID=PATH ...] [--shader-dir DIR] --output DIR [--trace-mode full|stripped|stripped-markers] [--optimizer-report] [--module-artifact-root DIR] [--module-build-evidence-json FILE]\n",
      stderr)
    fputs(
      "  module packages: --module-artifact-root and --module-build-evidence-json are required for checked builds.\n",
      stderr)
    fputs(
      "  module source packages: --module-source-package, source input, --shader-dir, and --output are required.\n",
      stderr)
        exit(1)
    }
    let agentFile = args[2]
    if let diagnostic = SmeltCAMGrammarRemoval.rejectionDiagnostic(forInputPath: agentFile) {
        fputs("Error: \(diagnostic)\n", stderr)
        exit(1)
    }
    let outputDir = parseArg("--output", default: ".")
    let camArtifactRoot = parseArg("--module-artifact-root")
    let camBuildEvidencePath = parseArg("--module-build-evidence-json")

    if agentFile.hasSuffix(".module.json") {
        if hasArg("--module-source-package") {
            if let argumentError = SmeltCAMSourcePackageBuilder.validateBuildCommandArguments(args) {
                fputs("Error: \(argumentError)\n", stderr)
                exit(1)
            }
      guard
        let traceMode = SmeltTraceMode(
                rawValue: parseArg("--trace-mode", default: SmeltTraceMode.full.rawValue)
        )
      else {
                fputs("Error: --trace-mode must be one of: full, stripped, stripped-markers\n", stderr)
                exit(1)
            }
            do {
                let result = try SmeltCAMSourcePackageBuilder.build(
                    camPath: agentFile,
                    outputDirectory: outputDir,
                    weightsDir: parseArg("--weights-dir"),
          sourceOverrides: try parseSourceOverrides(args),
                    shaderDir: parseArg("--shader-dir"),
                    traceMode: traceMode
                )
                fputs("Built: \(result.packagePath)\n", stderr)
        if let generatedSwiftPath = result.generatedSwiftPath {
          fputs("  Generated Swift: \(generatedSwiftPath)\n", stderr)
        }
                fputs("  Metal library:   \(result.metallibPath)\n", stderr)
                fputs("  Manifest:        \(result.manifestPath)\n", stderr)
                if hasArg("--optimizer-report") {
                    let optimizerReportPath = "\(result.packagePath)/optimizer-report.md"
                    try SmeltOptimizerReportGenerator.writeMarkdown(
                        packagePath: result.packagePath,
                        outputPath: optimizerReportPath
                    )
                    fputs("  Optimizer report: \(optimizerReportPath)\n", stderr)
                }
            } catch {
                fputs("Build failed: \(error)\n", stderr)
                exit(1)
            }
            return
        }
        if let argumentError = SmeltCAMPackageBuilder.validateBuildCommandArguments(args) {
            fputs("Error: \(argumentError)\n", stderr)
            exit(1)
        }
        do {
            let result = try SmeltCAMPackageBuilder.build(
                camPath: agentFile,
                artifactRoot: camArtifactRoot,
                outputDirectory: outputDir,
                evidencePath: camBuildEvidencePath,
                command: args
            )
            fputs("Built: \(result.packagePath)\n", stderr)
        } catch {
            fputs("Build failed: \(error)\n", stderr)
            exit(1)
        }
        return
    }

    if hasArg("--module-source-package") {
        fputs("Error: --module-source-package requires a .module.json input\n", stderr)
        exit(1)
    }

    // `.module.json` is the ONLY supported build input. The legacy LLM model-spec DSL and its
    // parser were retired — the release surface is authored as Swift-defined `.module.json`
    // modules (Sources/SmeltModels/*.swift). Give a targeted message for internal package-spec
    // JSON (which used to be sniffed here before falling through to the DSL parser).
    if let data = try? Data(contentsOf: URL(fileURLWithPath: agentFile)),
       let header = try? JSONDecoder().decode(BuildInputHeader.self, from: data),
    header.looksLikeInternalPackageSpec
  {
        fputs("Error: package spec JSON is internal; build a .module.json module instead\n", stderr)
        exit(1)
    }
    fputs(
        "Error: build input '\(agentFile)' is not a .module.json module "
            + "(the legacy model-spec DSL was retired)\n",
        stderr
    )
    exit(1)
}

private func parseSourceOverrides(_ command: [String]) throws -> [String: String] {
  var overrides: [String: String] = [:]
  var index = 3
  while index < command.count {
    guard command[index] == "--source" else {
      index +=
        command[index].hasPrefix("--")
          && !["--module-source-package", "--optimizer-report"].contains(command[index])
        ? 2 : 1
      continue
    }
    guard index + 1 < command.count else {
      throw SmeltCAMSourcePackageBuilderError.malformed("--source requires ID=PATH")
    }
    let value = command[index + 1]
    let fields = value.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
    guard fields.count == 2, !fields[0].isEmpty, !fields[1].isEmpty else {
      throw SmeltCAMSourcePackageBuilderError.malformed("--source requires ID=PATH")
    }
    let id = String(fields[0])
    guard overrides[id] == nil else {
      throw SmeltCAMSourcePackageBuilderError.malformed(
        "duplicate source override '\(id)'"
      )
    }
    overrides[id] = String(fields[1])
    index += 2
  }
  return overrides
}
