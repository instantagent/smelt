import Foundation
import SmeltRuntime

func runPrefillKernelsCommand(_ args: [String]) {
    guard args.count >= 3 else {
        fputs("Usage: smelt lab profile prefill <model.smeltpkg> [--tokens N] [--iterations N]\n", stderr)
        exit(1)
    }
    let pkgPath = args[2]
    let construction = requireCAMTextRuntimePlanOrExit(
        packagePath: pkgPath,
        request: .profilePrefillKernels,
        verb: "lab profile prefill"
    )
    let numTokens = Int(parseArg(args, "--tokens", default: "16")) ?? 16
    let iterations = Int(parseArg(args, "--iterations", default: "3")) ?? 3
    do {
        try construction.requirePrefillCapacity(tokenCount: numTokens)
    } catch {
        fputs("smelt lab profile prefill: \(error)\n", stderr)
        exit(1)
    }

    do {
        let runtime = try construction.makeRuntime(contextLimit: nil)
        guard runtime.hasMetalPrefill else {
            fputs("Package does not have Metal prefill (engine != metal)\n", stderr)
            exit(1)
        }
        let tokenIds = (0..<numTokens).map { Int32($0) }
        fputs("Per-kernel prefill GPU profile (\(iterations) iterations, \(numTokens) tokens)...\n\n", stderr)

        runtime.resetWorkingBuffers()
        _ = try runtime.prefillStep(tokenIds: tokenIds, startPos: 0)
        runtime.resetWorkingBuffers()

        var accumulated: [String: (totalUs: Double, count: Int, dispatches: Int)] = [:]
        for _ in 0..<iterations {
            runtime.resetWorkingBuffers()
            let results = try runtime.profilePrefillKernels(
                tokenIds: tokenIds, startPos: 0
            )
            for r in results {
                let prev = accumulated[r.name, default: (0, 0, 0)]
                accumulated[r.name] = (
                    prev.totalUs + r.totalGpuUs,
                    prev.count + 1,
                    r.dispatchCount
                )
            }
        }

        printKernelProfileTable(
            accumulated: accumulated,
            iterations: iterations,
            perTokenUnits: "ms/prefill"
        )
    } catch {
        fputs("Prefill kernel profile failed: \(error)\n", stderr)
        exit(1)
    }
}
