// Generic target-seeded auxiliary / speculative-decode benchmark.
//
// Mirrors the generic speculative-decode benchmark logic, but runs as a plain
// executable so the swift-testing-1501 objc_retain crash and the
// XCTest swift_getErrorValue trap on this signature don't block
// measurement. Both framework wrappers SIGTRAP/SIGSEGV inside
// their own runners before/around the Smelt runtime — the same
// `runtime.decodeStep` body runs to completion here.
//
// Usage:
//   smelt mtp-bench <target.smeltpkg> <drafter.smeltpkg>
//   smelt mtp-bench <target.smeltpkg> --suffix-only
//     [--K N] [--measure-tokens N] [--prompt-file FILE]
//     [--prompt STRING] [--skip-plain] [--suffix-max-n N]

import Foundation
import SmeltRuntime
import SmeltServe
import SmeltSchema

func runMtpBenchCommand() {
    let usage = "Usage: smelt mtp-bench <target.smeltpkg> (<drafter.smeltpkg> | --suffix-only) [--K N] [--measure-tokens N] [--prompt STRING | --prompt-file FILE] [--skip-plain] [--no-stop-at-eot] [--temperature T --seed N] [--suffix-max-n N]\n"
    guard args.count >= 4 else {
        fputs(usage, stderr)
        exit(1)
    }
    let targetPkg = args[2]
    let suffixOnly = hasArg("--suffix-only")
    let drafterPkg: String?
    if suffixOnly {
        drafterPkg = nil
    } else if !args[3].hasPrefix("--") {
        drafterPkg = args[3]
    } else {
        fputs(usage, stderr)
        exit(1)
    }
    let targetConstruction = requireCAMTextRuntimePlanOrExit(
        packagePath: targetPkg,
        request: .mtpBenchTarget,
        verb: "mtp-bench",
        requireAuthoredInventory: true
    )

    do {
        let K = try parsePositiveIntArg("--K") ?? SmeltSpeculativeRuntime.defaultK
        let suffixMaxN = try parsePositiveIntArg("--suffix-max-n") ?? 4
        let measureTokens = try parsePositiveIntArg("--measure-tokens") ?? 60
        let promptOverride = parseArg("--prompt").nilIfEmpty
        let promptFile = parseArg("--prompt-file").nilIfEmpty
        let skipPlain = hasArg("--skip-plain")
        let stopAtEOT = !hasArg("--no-stop-at-eot")
        let selectionMode = try parseMtpSelectionMode()

        try runMtpBench(
            targetConstruction: targetConstruction,
            drafterPkg: drafterPkg,
            suffixMaxN: suffixMaxN,
            K: K,
            measureTokens: measureTokens,
            promptOverride: promptOverride,
            promptFile: promptFile,
            skipPlain: skipPlain,
            stopAtEOT: stopAtEOT,
            selectionMode: selectionMode
        )
    } catch {
        fputs("mtp-bench failed: \(error)\n", stderr)
        exit(1)
    }
}

private func parseMtpSelectionMode() throws -> SmeltSelectionMode {
    guard let temp = try parseNonNegativeDoubleArg("--temperature") else {
        return .argmax
    }
    guard temp.isFinite, temp > 0 else {
        return .argmax
    }
    let seedRaw = parseArg("--seed")
    guard !seedRaw.isEmpty else {
        throw NSError(
            domain: "SmeltCLI", code: 1,
            userInfo: [NSLocalizedDescriptionKey:
                "--temperature requires --seed N (UInt64) so stochastic bench results are reproducible across runs"]
        )
    }
    guard let seed = UInt64(seedRaw) else {
        throw NSError(
            domain: "SmeltCLI", code: 1,
            userInfo: [NSLocalizedDescriptionKey:
                "--seed must be a UInt64; got '\(seedRaw)'"]
        )
    }
    return .temperature(Float(temp), seed: seed)
}

private func mtpLog(_ msg: String) {
    fputs("[mtp-bench] \(msg)\n", stderr)
}

private func runMtpBench(
    targetConstruction: CAMTextRuntimeConstruction,
    drafterPkg: String?,
    suffixMaxN: Int,
    K: Int,
    measureTokens: Int,
    promptOverride: String?,
    promptFile: String?,
    skipPlain: Bool,
    stopAtEOT: Bool,
    selectionMode: SmeltSelectionMode
) throws {
    let stride = K + 1
    guard measureTokens % stride == 0 else {
        fputs("Error: --measure-tokens must be divisible by K+1=\(stride)\n", stderr)
        exit(1)
    }
    let measureRounds = measureTokens / stride

    let (_, inference) = try targetConstruction.loadManifestConfig()
    let eotTokens = Set(inference.eosTokens)
    let tokenizer = try targetConstruction.makeTokenizer()
    let template = try targetConstruction.resolveTemplate(cliOverride: nil)
    let defaultPrompt = "List the first five prime numbers."
    let prompt: String
    if let promptFile {
        prompt = try readTextFileArg(promptFile, label: "prompt")
    } else {
        prompt = promptOverride ?? defaultPrompt
    }
    let promptTokens = try buildInputIds(
        prompt: prompt,
        tokenizer: tokenizer,
        template: template,
        thinkingPolicy: resolvedThinkingPolicy(inference)
    )
    let promptLen = Int32(promptTokens.count)
    let contextCap = Int(promptLen) + measureTokens + 8

    let plainElapsed: Double
    var plainTokens: [Int32] = []
    var plainStepsRun = 0
    if skipPlain {
        mtpLog("--skip-plain set; no plain-decode baseline")
        plainElapsed = 0
    } else {
        mtpLog("loading plain target...")
        let plainTarget = try targetConstruction.makeRuntime(contextLimit: nil)
        try plainTarget.prepareForRequest(
            batchCapacity: drafterPkg == nil
                ? min(plainTarget.maxPrefillBatchSize, promptTokens.count)
                : 1,
            contextCapacity: contextCap
        )
        try plainTarget.ensureContextCapacity(contextCap)
        var plainTok = drafterPkg == nil
            ? try primeMTPPromptWithMetalPrefill(
                runtime: plainTarget,
                promptTokens: promptTokens,
                finalSelectionMode: selectionMode
            )
            : try primeMTPPrompt(
                runtime: plainTarget,
                promptTokens: promptTokens,
                finalSelectionMode: selectionMode
            )
        var plainPos = promptLen
        plainTokens.append(plainTok)
        mtpLog("plain decode \(measureTokens) tokens...")
        let plainStart = CFAbsoluteTimeGetCurrent()
        for i in 0 ..< measureTokens {
            if stopAtEOT, eotTokens.contains(plainTok) { break }
            plainTok = try plainTarget.decodeStep(
                tokenId: plainTok, position: plainPos,
                selectionMode: selectionMode
            )
            plainPos += 1
            plainStepsRun += 1
            if i + 1 < measureTokens { plainTokens.append(plainTok) }
        }
        plainElapsed = CFAbsoluteTimeGetCurrent() - plainStart
    }

    mtpLog("loading spec runtime...")
    let specTargetRuntime = try targetConstruction.makeRuntime(contextLimit: nil)
    let runtime: SmeltSpeculativeRuntime
    if let drafterPkg {
        runtime = try SmeltSpeculativeRuntime(
            target: specTargetRuntime,
            drafterPath: drafterPkg,
            K: K
        )
    } else {
        runtime = try SmeltSpeculativeRuntime(
            target: specTargetRuntime,
            drafter: SmeltSuffixLookupDrafter(
                maxNeedleLength: suffixMaxN
            ),
            K: K
        )
    }
    let specTarget = runtime.target
    try specTarget.prepareForRequest(
        batchCapacity: drafterPkg == nil
            ? min(specTarget.maxPrefillBatchSize, promptTokens.count)
            : 1,
        contextCapacity: contextCap
    )
    let primed: (nextToken: Int32, hiddenStates: [Data])
    if drafterPkg == nil {
        primed = (
            try primeMTPPromptWithMetalPrefill(
                runtime: specTarget,
                promptTokens: promptTokens,
                finalSelectionMode: .argmax
            ),
            []
        )
    } else {
        primed = try primeMTPPromptWithHiddenStates(
            runtime: specTarget,
            promptTokens: promptTokens,
            finalSelectionMode: .argmax
        )
    }
    runtime.adoptCurrentTargetLogits(argmax: primed.nextToken)
    if drafterPkg != nil {
        try runtime.primeTargetSeededDrafter(
            promptTokens: promptTokens,
            targetNextToken: primed.nextToken,
            targetHiddenStates: primed.hiddenStates
        )
    }
    runtime.resetSuffixCache(promptTokens: promptTokens)

    mtpLog("spec decode \(measureRounds) rounds (K=\(K))...")
    var lastTok = promptTokens.last!
    var lastPos = promptLen - 1
    var committed = 0
    var specCommittedTokens: [Int32] = []
    var roundAccepts: [Int] = []
    let draftBonusActive =
        ProcessInfo.processInfo.environment["SMELT_SPEC_DRAFT_BONUS"] == "1"
        && !selectionMode.usesArgmaxFastPath
    var drafterSecondsTotal: Double = 0
    var verifySecondsTotal: Double = 0
    var leviathanSecondsTotal: Double = 0
    var refreshSecondsTotal: Double = 0
    var roundsRun = 0
    var hitEOT = false
    let specStart = CFAbsoluteTimeGetCurrent()
    for _ in 0 ..< measureRounds {
        let r = try runtime.decodeStep(
            lastToken: lastTok, position: lastPos,
            selectionMode: selectionMode
        )
        // Truncate committed tokens at the first EOT so spec
        // doesn't get credit for tokens past where plain greedy
        // stops — keeps observed ratio honest when EOT lands mid-
        // round.
        let eotIdx = stopAtEOT
            ? r.committedTokens.firstIndex(where: eotTokens.contains)
            : nil
        let inTurnTokens = eotIdx.map { Array(r.committedTokens[...$0]) }
            ?? r.committedTokens
        committed += inTurnTokens.count
        roundsRun += 1
        // A full greedy verified-tail commit intentionally emits K accepted
        // candidates with no bonus refresh. Partial/default rounds still emit
        // accepted+1, so clamping by the observed token count handles both.
        let acceptedForLog = min(r.acceptedCount, inTurnTokens.count)
        roundAccepts.append(max(acceptedForLog, 0))
        specCommittedTokens.append(contentsOf: inTurnTokens)
        runtime.recordGeneratedTokens(specCommittedTokens)
        drafterSecondsTotal += r.phaseTimings.drafterSeconds
        verifySecondsTotal += r.phaseTimings.verifySeconds
        leviathanSecondsTotal += r.phaseTimings.leviathanSeconds
        refreshSecondsTotal += r.phaseTimings.refreshSeconds
        lastTok = r.nextToken
        lastPos = r.nextPosition
        if eotIdx != nil {
            hitEOT = true
            break
        }
    }
    let specElapsed = CFAbsoluteTimeGetCurrent() - specStart

    if !skipPlain, selectionMode.usesArgmaxFastPath {
        let cmp = min(
            plainTokens.count,
            specCommittedTokens.count,
            measureTokens
        )
        let mismatch = (0 ..< cmp).first(where: { plainTokens[$0] != specCommittedTokens[$0] })
        print("  parity:       plain[0..\(cmp))=\(plainTokens.prefix(cmp).map { String($0) }.joined(separator: ","))")
        print("                spec[0..\(cmp))=\(specCommittedTokens.prefix(cmp).map { String($0) }.joined(separator: ","))")
        if let m = mismatch {
            print("  parity:       FIRST MISMATCH at index \(m): plain=\(plainTokens[m]) spec=\(specCommittedTokens[m])")
        } else {
            print("  parity:       MATCH for first \(cmp) tokens")
        }
    } else if !skipPlain {
        print("  parity:       SKIPPED (stochastic spec-decode diverges from greedy plain decode by design)")
    }

    let specPerRound = specElapsed / Double(roundsRun) * 1000
    let specPerToken = specElapsed / Double(committed) * 1000
    let avgAccept = Double(roundAccepts.reduce(0, +)) / Double(roundsRun)
    let drafterPerRound = drafterSecondsTotal / Double(roundsRun) * 1000
    let verifyPerRound = verifySecondsTotal / Double(roundsRun) * 1000
    let leviathanPerRound = leviathanSecondsTotal / Double(roundsRun) * 1000
    let refreshPerRound = refreshSecondsTotal / Double(roundsRun) * 1000
    let acceptedBudget = draftBonusActive ? K + 1 : K

    print("---- mtp-bench: spec-decode \(drafterPkg == nil ? "(suffix lookup)" : "(real HF weights)") ----")
    let modeLabel: String
    let plainLabel: String
    switch selectionMode {
    case .argmax:
        modeLabel = "argmax (greedy)"
        plainLabel = "plain greedy"
    case let .temperature(t, s):
        modeLabel = String(format: "temperature=%.3f seed=%llu", t, s)
        plainLabel = "plain stochastic"
    case let .filteredTemperature(t, topK, topP, s):
        modeLabel = String(
            format: "temperature=%.3f top-k=%@ top-p=%.3f seed=%llu",
            t,
            topK.map(String.init) ?? "all",
            topP,
            s
        )
        plainLabel = "plain filtered stochastic"
    }
    print("  mode:         \(modeLabel)")
    print("  prompt:       \(promptTokens.count) tokens")
    if skipPlain {
        print("  \(plainLabel): N/A (--skip-plain)")
        print("  observed:     N/A (--skip-plain)")
    } else if plainStepsRun == 0 {
        print("  \(plainLabel): N/A (prefill selected token was EOT; no decode steps timed)")
        print("  observed:     N/A")
    } else {
        let plainPerToken = plainElapsed / Double(plainStepsRun) * 1000
        let observedRatio = plainPerToken / specPerToken
        print(String(format: "  %@: %.3f ms/tok (%.2f tok/s)",
                     plainLabel,
                     plainPerToken, 1000 / plainPerToken))
        print(String(format: "  observed:     %.3fx",
                     observedRatio))
    }
    if stopAtEOT, eotTokens.isEmpty {
        print("  stop:         eos_tokens manifest entry is empty; --stop-at-eot is a no-op")
    } else if stopAtEOT, hitEOT {
        print("  stop:         hit eos after \(roundsRun)/\(measureRounds) rounds")
    } else if stopAtEOT {
        print("  stop:         exhausted \(measureRounds) rounds without eos")
    }
    print(String(format: "  spec round:   %.3f ms/round (K+1=%d)",
                 specPerRound, K + 1))
    print(String(format: "  spec decode:  %.3f ms/tok over %d committed (%.2f tok/s)",
                 specPerToken, committed, 1000 / specPerToken))
    print(String(format: "  avg accept:   %.2f of %d",
                 avgAccept, acceptedBudget))
    print("  per-round:    \(roundAccepts.map { String($0) }.joined(separator: ","))")
    if let suffix = runtime.drafter as? SmeltSuffixLookupDrafter {
        print("  suffix:       hits=\(suffix.hits) misses=\(suffix.misses) "
            + "last-n=\(suffix.lastNeedleLength.map(String.init) ?? "none")")
        if hasArg("--suffix-debug"), !suffix.lastCandidates.isEmpty {
            print("  suffix draft: \(suffix.lastCandidates.map(String.init).joined(separator: ","))")
        }
    }
    print(String(format: "  phase split:  drafter=%.2fms verify=%.2fms leviathan=%.2fms refresh=%.2fms",
                 drafterPerRound, verifyPerRound, leviathanPerRound, refreshPerRound))
}

private func primeMTPPrompt(
    runtime: SmeltRuntime,
    promptTokens: [Int32],
    finalSelectionMode: SmeltSelectionMode
) throws -> Int32 {
    var next: Int32 = 0
    for (index, token) in promptTokens.enumerated() {
        next = try runtime.decodeStep(
            tokenId: token,
            position: Int32(index),
            selectionMode: index == promptTokens.count - 1 ? finalSelectionMode : .argmax
        )
    }
    return next
}

private func primeMTPPromptWithMetalPrefill(
    runtime: SmeltRuntime,
    promptTokens: [Int32],
    finalSelectionMode: SmeltSelectionMode
) throws -> Int32 {
    precondition(!promptTokens.isEmpty)
    let chunkSize = max(runtime.maxPrefillBatchSize, 1)
    var start = 0
    var next: Int32 = 0
    while start < promptTokens.count {
        let end = min(start + chunkSize, promptTokens.count)
        next = try runtime.prefillStep(
            tokenIds: Array(promptTokens[start..<end]),
            startPos: Int32(start),
            selectionMode: end == promptTokens.count
                ? finalSelectionMode
                : .argmax
        )
        start = end
    }
    return next
}

private func primeMTPPromptWithHiddenStates(
    runtime: SmeltRuntime,
    promptTokens: [Int32],
    finalSelectionMode: SmeltSelectionMode
) throws -> (nextToken: Int32, hiddenStates: [Data]) {
    var next: Int32 = 0
    var hiddenStates: [Data] = []
    hiddenStates.reserveCapacity(promptTokens.count)
    for (index, token) in promptTokens.enumerated() {
        next = try runtime.decodeStep(
            tokenId: token,
            position: Int32(index),
            selectionMode: index == promptTokens.count - 1
                ? finalSelectionMode
                : .argmax
        )
        hiddenStates.append(try runtime.currentHiddenStateBytes())
    }
    return (next, hiddenStates)
}
