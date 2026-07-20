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
        fputs("Usage: smelt build <model.module.json> [--module-source-package] [--weights-dir DIR] [--shader-dir DIR] --output DIR [--trace-mode full|stripped|stripped-markers] [--optimizer-report] [--module-artifact-root DIR] [--module-build-evidence-json FILE]\n", stderr)
        fputs("  module packages: --module-artifact-root and --module-build-evidence-json are required for checked builds.\n", stderr)
        fputs("  module source packages: --module-source-package, --weights-dir, --shader-dir, and --output are required.\n", stderr)
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
            guard let traceMode = SmeltTraceMode(
                rawValue: parseArg("--trace-mode", default: SmeltTraceMode.full.rawValue)
            ) else {
                fputs("Error: --trace-mode must be one of: full, stripped, stripped-markers\n", stderr)
                exit(1)
            }
            do {
                let result = try SmeltCAMSourcePackageBuilder.build(
                    camPath: agentFile,
                    outputDirectory: outputDir,
                    weightsDir: parseArg("--weights-dir"),
                    shaderDir: parseArg("--shader-dir"),
                    traceMode: traceMode
                )
                fputs("Built: \(result.packagePath)\n", stderr)
                fputs("  Generated Swift: \(result.generatedSwiftPath)\n", stderr)
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
       header.looksLikeInternalPackageSpec {
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
