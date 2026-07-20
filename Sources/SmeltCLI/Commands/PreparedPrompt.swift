import Foundation
import CryptoKit
import SmeltRuntime
import SmeltSchema

/// Internal package-construction command. Adapter/release tooling supplies an
/// already-tokenized frozen contract; the runtime artifact stays independent
/// of the adapter that produced those IDs.
func runPreparePromptCommand() {
    let usage = "Usage: smelt prepare-prompt <package.smeltpkg> --id ID "
        + "(--ids CSV | --ids-file FILE) [--prefix-count N] "
        + "[--capture automatic|sequential] [--temperature N] "
        + "[--top-k N] [--top-p N]\n"
    guard args.count >= 3 else {
        fputs(usage, stderr)
        exit(1)
    }
    let packagePath = URL(fileURLWithPath: args[2], isDirectory: true)
        .standardizedFileURL.path
    let construction = requireCAMTextRuntimePlanOrExit(
        packagePath: packagePath,
        request: .runText,
        verb: "prepare-prompt"
    )
    let id = parseArg("--id")
    let idsText: String
    let inline = parseArg("--ids")
    let idsFile = parseArg("--ids-file")
    if !inline.isEmpty, idsFile.isEmpty {
        idsText = inline
    } else if inline.isEmpty, !idsFile.isEmpty {
        do {
            idsText = try String(contentsOfFile: idsFile, encoding: .utf8)
        } catch {
            fputs("prepare-prompt: could not read --ids-file: \(error)\n", stderr)
            exit(1)
        }
    } else {
        fputs(usage, stderr)
        exit(1)
    }

    do {
        guard !id.isEmpty else {
            throw CLIError("--id is required")
        }
        var tokenIds = try parsePreparedPromptIDs(idsText)
        let prefixCountRaw = parseArg("--prefix-count")
        if !prefixCountRaw.isEmpty {
            guard let prefixCount = Int(prefixCountRaw),
                  prefixCount > 0, prefixCount <= tokenIds.count
            else { throw CLIError("--prefix-count is outside the supplied IDs") }
            tokenIds = Array(tokenIds.prefix(prefixCount))
        }

        let temperature = try optionalDouble("--temperature", minimum: 0)
        let topK = try optionalInt("--top-k", minimum: 1)
        let topP = try optionalDouble("--top-p", minimum: 0, maximum: 1)
        if topP == 0 { throw CLIError("--top-p must be greater than zero") }
        let sampling: SmeltPreparedPromptSampling? =
            temperature != nil || topK != nil || topP != nil
                ? SmeltPreparedPromptSampling(
                    temperature: temperature, topK: topK, topP: topP
                ) : nil

        let capture = parseArg("--capture", default: "automatic")
        let (manifest, _) = try construction.loadManifestConfig()
        let model = try construction.makeModel(
            contextLimit: tokenIds.count + 16,
            manifest: manifest
        )
        let snapshot: SmeltPromptSnapshot
        switch capture {
        case "automatic":
            snapshot = try model.captureBasePrompt(tokenIds: tokenIds)
        case "sequential":
            snapshot = try model.captureBasePromptSequential(tokenIds: tokenIds)
        default:
            throw CLIError("--capture must be automatic or sequential")
        }
        let state = SmeltPreparedPromptState(
            id: id,
            tokenIds: tokenIds,
            sampling: sampling,
            snapshot: snapshot
        )
        let info = try SmeltPreparedPromptSet.write(
            packagePath: packagePath,
            state: state
        )
        let files = try SmeltPreparedPromptSet.declaredFiles(
            packagePath: packagePath
        )
        try SmeltBakeManifest.record(
            [SmeltBakeManifest.preparedPrompts(requiredFiles: files)],
            packagePath: packagePath
        )
        var tokenBytes = Data(capacity: tokenIds.count * MemoryLayout<Int32>.size)
        for token in tokenIds {
            var littleEndian = token.littleEndian
            withUnsafeBytes(of: &littleEndian) {
                tokenBytes.append(contentsOf: $0)
            }
        }
        let tokenHash = SHA256.hash(data: tokenBytes).map {
            String(format: "%02x", $0)
        }.joined()
        fputs(
            "Prepared \(id): tokens=\(tokenIds.count) "
                + "int32le_sha256=\(tokenHash) "
                + "snapshot_bytes=\(info.fileBytes) capture=\(capture)\n",
            stderr
        )
    } catch {
        fputs("prepare-prompt failed: \(error)\n", stderr)
        exit(1)
    }
}

private func parsePreparedPromptIDs(_ text: String) throws -> [Int32] {
    let pieces = text.split { $0 == "," || $0.isWhitespace }
    guard !pieces.isEmpty else { throw CLIError("no token IDs supplied") }
    return try pieces.map { piece in
        guard let value = Int32(piece), value >= 0 else {
            throw CLIError("invalid token ID '\(piece)'")
        }
        return value
    }
}

private func optionalInt(_ flag: String, minimum: Int) throws -> Int? {
    let raw = parseArg(flag)
    guard !raw.isEmpty else { return nil }
    guard let value = Int(raw), value >= minimum else {
        throw CLIError("\(flag) must be at least \(minimum)")
    }
    return value
}

private func optionalDouble(
    _ flag: String,
    minimum: Double,
    maximum: Double? = nil
) throws -> Double? {
    let raw = parseArg(flag)
    guard !raw.isEmpty else { return nil }
    guard let value = Double(raw), value.isFinite, value >= minimum,
          maximum.map({ value <= $0 }) ?? true
    else { throw CLIError("invalid \(flag) value") }
    return value
}
