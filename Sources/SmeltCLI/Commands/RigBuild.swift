import SmeltCompiler
import Foundation

func runRigBuildCommand() {
    guard args.count >= 3, !args[2].hasPrefix("--") else {
        fputs(
            "Usage: smelt rig-build <grpo_1400.ckpt> --output <rig.smeltpkg> "
                + "[--shader-dir DIR]\n",
            stderr
        )
        exit(1)
    }
    let checkpoint = args[2]
    let output = parseArg("--output")
    let shaderDirectory = parseArg("--shader-dir", default: "Resources/Shaders")
    guard !output.isEmpty else {
        fputs("rig-build requires --output\n", stderr)
        exit(1)
    }
    do {
        try SmeltRigPackageBuilder.build(
            checkpointPath: checkpoint,
            shaderDirectory: shaderDirectory,
            outputPath: output
        )
        fputs("Built rig package: \(output)\n", stderr)
    } catch {
        fputs("rig-build failed: \(error)\n", stderr)
        exit(1)
    }
}
