import Foundation
import SmeltRuntime

func runVerifyProfileCommand(_ args: [String]) {
    let usage = "Usage: smelt lab profile verify <model.smeltpkg> --ids <csv> [--batch N] [--match substring[,substring...]] [--output FILE]\n"
    guard args.count >= 3 else {
        fputs(usage, stderr)
        exit(1)
    }
    let packagePath = args[2]
    let ids: [Int32]
    do {
        ids = try parseVerifyProfileIDs(parseArg(args, "--ids"))
    } catch {
        fputs("smelt lab profile verify: \(error)\n", stderr)
        exit(1)
    }
    let batch = Int(parseArg(args, "--batch", default: "\(ids.count)")) ?? 0
    guard batch > 0, batch <= ids.count else {
        fputs("smelt lab profile verify: --batch must be in [1, \(ids.count)]\n", stderr)
        exit(1)
    }
    let tokens = Array(ids.prefix(batch))
    let matches = parseArg(args, "--match")
        .split(separator: ",")
        .map(String.init)
        .filter { !$0.isEmpty }
    let outputPath = URL(fileURLWithPath: parseArg(args,
        "--output", default: "verify-profile.csv"
    )).standardizedFileURL.path
    let construction = requireCAMTextRuntimePlanOrExit(
        packagePath: packagePath,
        request: .profileVerifyArgmax,
        verb: "lab profile verify",
        requireAuthoredInventory: true
    )

    do {
        let verify = try construction.makeRuntime(contextLimit: batch + 1)
        guard verify.canChunkedPrefillVerifyArgmax(tokenCount: batch) else {
            throw NSError(
                domain: "SmeltCLI",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "package cannot run transactional verify-argmax at batch \(batch)"]
            )
        }
        try verify.prepareForRequest(
            batchCapacity: batch,
            contextCapacity: batch + 1
        )

        let sequential = try construction.makeRuntime(contextLimit: batch + 1)
        try sequential.prepareForRequest(
            batchCapacity: 1,
            contextCapacity: batch + 1
        )
        var expected: [Int32] = []
        expected.reserveCapacity(batch)
        for (index, token) in tokens.enumerated() {
            expected.append(try sequential.decodeStep(
                tokenId: token,
                position: Int32(index),
                selectionMode: .argmax
            ))
        }

        verify.resetWorkingBuffers()
        verify.armVerifyArgmaxProfile(
            outputPath: outputPath,
            pipelineNameMatches: matches
        )
        let started = CFAbsoluteTimeGetCurrent()
        let actual = try verify.prefillVerifyArgmax(tokens: tokens, startPos: 0)
        let profileWallMs = (CFAbsoluteTimeGetCurrent() - started) * 1_000
        guard actual == expected else {
            let mismatch = zip(actual, expected).enumerated().first {
                $0.element.0 != $0.element.1
            }
            let detail = mismatch.map {
                "row \($0.offset): verify=\($0.element.0) sequential=\($0.element.1)"
            } ?? "row-count mismatch verify=\(actual.count) sequential=\(expected.count)"
            throw NSError(
                domain: "SmeltCLI",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "verify parity failed: \(detail)"]
            )
        }
        fputs(
            "verify-profile package=\(packagePath) batch=\(batch) parity=exact"
                + " profile_wall_ms=\(String(format: "%.3f", profileWallMs))"
                + " output=\(outputPath)"
                + (matches.isEmpty ? "" : " match=\(matches.joined(separator: ","))")
                + "\n",
            stderr
        )
    } catch {
        fputs("smelt lab profile verify failed: \(error)\n", stderr)
        exit(1)
    }
}

private func parseVerifyProfileIDs(_ text: String) throws -> [Int32] {
    let parts = text.split(separator: ",")
    guard !parts.isEmpty else {
        throw NSError(
            domain: "SmeltCLI", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "--ids must not be empty"]
        )
    }
    return try parts.map { part in
        guard let value = Int32(part.trimmingCharacters(in: .whitespaces)) else {
            throw NSError(
                domain: "SmeltCLI", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "invalid token id '\(part)'"]
            )
        }
        return value
    }
}
