import Foundation
import SmeltRuntime

func runPrefillCommand() {
    guard args.count >= 3 else {
        fputs("Usage: smelt prefill <model.smeltpkg> [--tokens N]\n", stderr)
        exit(1)
    }
    let pkgPath = args[2]
    let construction = requireCAMTextRuntimePlanOrExit(
        packagePath: pkgPath,
        request: .prefillParity,
        verb: "prefill"
    )
    let numTokens = Int(parseArg("--tokens", default: "4")) ?? 4

    do {
        let tokenIds = (0..<numTokens).map { Int32($0) }

        fputs("Decode reference: \(numTokens) steps...\n", stderr)
        let refRuntime = try construction.makeRuntime(contextLimit: nil)
        var refToken: Int32 = 0
        let refStart = CFAbsoluteTimeGetCurrent()
        for (i, tok) in tokenIds.enumerated() {
            refToken = try refRuntime.decodeStep(tokenId: tok, position: Int32(i))
        }
        let refElapsed = CFAbsoluteTimeGetCurrent() - refStart
        fputs("  Decode ref token: \(refToken)  (\(String(format: "%.1f", refElapsed * 1000))ms)\n", stderr)

        let pfRuntime = try construction.makeRuntime(contextLimit: nil)
        guard pfRuntime.hasMetalPrefill else {
            fputs("Package does not have Metal prefill (engine != metal)\n", stderr)
            exit(1)
        }
        fputs("Metal prefill: \(numTokens) tokens...\n", stderr)
        let pfStart = CFAbsoluteTimeGetCurrent()
        var pfToken: Int32 = 0
        let chunkSize = max(pfRuntime.maxPrefillBatchSize, 1)
        var start = 0
        while start < tokenIds.count {
            let end = min(start + chunkSize, tokenIds.count)
            pfToken = try pfRuntime.prefillStep(
                tokenIds: Array(tokenIds[start..<end]),
                startPos: Int32(start)
            )
            start = end
        }
        let pfElapsed = CFAbsoluteTimeGetCurrent() - pfStart
        fputs("  Prefill token:    \(pfToken)  (\(String(format: "%.1f", pfElapsed * 1000))ms)\n", stderr)

        if pfToken == refToken {
            fputs("  ✓ MATCH\n", stderr)
        } else {
            fputs("  ✗ MISMATCH — prefill=\(pfToken) decode=\(refToken)\n", stderr)
        }
    } catch {
        fputs("Prefill failed: \(error)\n", stderr)
        exit(1)
    }
}
