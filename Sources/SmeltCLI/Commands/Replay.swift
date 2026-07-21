import Foundation
import SmeltRuntime

func runReplayCommand(_ args: [String]) {
    guard args.count >= 3 else {
        fputs(
            "Usage: smelt lab replay <trace.jsonl> [--package <model.smeltpkg>]"
                + " [--failure-bundle <dir>]\n",
            stderr
        )
        exit(1)
    }
    let tracePath = args[2]
    let packagePath = parseArg(args, "--package", default: "")
    let failureBundleDir = parseArg(args, "--failure-bundle", default: "")
    let construction: CAMTextRuntimeConstruction?
    if !packagePath.isEmpty {
        construction = requireCAMTextRuntimePlanOrExit(
            packagePath: packagePath,
            request: .runText,
            verb: "lab replay",
            requireAuthoredInventory: true
        )
    } else {
        construction = nil
    }
    do {
        try replayTrace(
            tracePath: tracePath,
            packagePath: packagePath.isEmpty ? nil : packagePath,
            construction: construction,
            failureBundleDir: failureBundleDir.isEmpty ? nil : failureBundleDir
        )
    } catch {
        fputs("Replay failed: \(error)\n", stderr)
        exit(1)
    }
}
