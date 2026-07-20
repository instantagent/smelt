import Foundation
import SmeltRuntime

func runReplayCommand() {
    guard args.count >= 3 else {
        fputs(
            "Usage: smelt replay <trace.jsonl> [--package <model.smeltpkg>]"
                + " [--failure-bundle <dir>]\n",
            stderr
        )
        exit(1)
    }
    let tracePath = args[2]
    let packagePath = parseArg("--package", default: "")
    let failureBundleDir = parseArg("--failure-bundle", default: "")
    let construction: CAMTextRuntimeConstruction?
    if !packagePath.isEmpty {
        construction = requireCAMTextRuntimePlanOrExit(
            packagePath: packagePath,
            request: .runText,
            verb: "replay",
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
