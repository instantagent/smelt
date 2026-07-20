import Foundation
import SmeltCompiler

func runQwen35VisionComponentBuildCommand() {
    guard args.count >= 3, !args[2].hasPrefix("--") else {
        fputs(
            "Usage: smelt vision-component-build <checkpoint-dir> --module <module.json> "
                + "--output <vision-component-dir> [--shader-dir DIR]\n",
            stderr
        )
        exit(1)
    }
    let checkpoint = args[2]
    let module = parseArg("--module")
    let output = parseArg("--output")
    let shaderDirectory = parseArg("--shader-dir", default: "Resources/Shaders")
    guard !module.isEmpty, !output.isEmpty else {
        fputs("vision-component-build requires --module and --output\n", stderr)
        exit(1)
    }
    do {
        try SmeltQwen35VisionArtifactBuilder.build(
            modulePath: module,
            checkpointPath: checkpoint,
            shaderDirectory: shaderDirectory,
            outputPath: output
        )
        fputs("Built: \(output)\n", stderr)
    } catch {
        fputs("vision-component-build failed: \(error)\n", stderr)
        exit(1)
    }
}
