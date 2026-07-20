import CryptoKit
import Foundation
import SmeltRuntime
import SmeltSchema

func runTraceCommand() {
    let usage = """
    Usage: smelt trace inspect <model.smeltpkg> [--json] [--hash-large-artifacts]
           smelt trace record <model.smeltpkg> [--output trace.smttrace] [--events events.json] [--case-text TEXT] [--json] [--hash-large-artifacts]
           smelt trace compare <expected.smttrace> <actual.smttrace> [--json]
           smelt trace verify <model.smeltpkg> [--golden trace.smttrace] [--events events.json] [--case-text TEXT] [--json] [--hash-large-artifacts]
           smelt trace replay <model.smeltpkg> <trace.smttrace> [--events events.json] [--case-text TEXT] [--json] [--hash-large-artifacts]
           smelt trace suite <suite.json> [--package model.smeltpkg] [--update] [--json] [--hash-large-artifacts]
       Case options:
           --case-max-steps N
           TTS only: --case-language LANG --case-instruct TEXT --case-speaker TEXT
                     --case-first-chunk N --case-max-chunk N --case-seed N --case-greedy
    """
    var idx = 2
    var mode = "inspect"
    let modes = Set(["inspect", "record", "compare", "verify", "replay", "suite"])
    if idx < args.count, modes.contains(args[idx]) {
        mode = args[idx]
        idx += 1
    }

    var packagePath: String?
    var outputPath: String?
    var goldenPath: String?
    var eventsPath: String?
    var caseText: String?
    var caseLanguage = "English"
    var caseInstruct: String?
    var caseSpeaker: String?
    var caseMaxSteps = 8
    var caseFirstChunk: Int?
    var caseMaxChunk: Int?
    var caseSeed: UInt64?
    var caseGreedy = false
    var updateSuite = false
    var emitJSON = false
    var hashLargeArtifacts = false
    var positional: [String] = []
    while idx < args.count {
        let arg = args[idx]
        switch arg {
        case "--json":
            emitJSON = true
            idx += 1
        case "--hash-large-artifacts":
            hashLargeArtifacts = true
            idx += 1
        case "--package":
            guard idx + 1 < args.count else {
                fputs("smelt trace: --package requires a path\n", stderr)
                exit(1)
            }
            packagePath = args[idx + 1]
            idx += 2
        case "--output":
            guard idx + 1 < args.count else {
                fputs("smelt trace: --output requires a path\n", stderr)
                exit(1)
            }
            outputPath = args[idx + 1]
            idx += 2
        case "--golden":
            guard idx + 1 < args.count else {
                fputs("smelt trace: --golden requires a path\n", stderr)
                exit(1)
            }
            goldenPath = args[idx + 1]
            idx += 2
        case "--events":
            guard idx + 1 < args.count else {
                fputs("smelt trace: --events requires a path\n", stderr)
                exit(1)
            }
            eventsPath = args[idx + 1]
            idx += 2
        case "--case-text":
            guard idx + 1 < args.count else {
                fputs("smelt trace: --case-text requires text\n", stderr)
                exit(1)
            }
            caseText = args[idx + 1]
            idx += 2
        case "--case-language":
            guard idx + 1 < args.count else {
                fputs("smelt trace: --case-language requires a value\n", stderr)
                exit(1)
            }
            caseLanguage = args[idx + 1]
            idx += 2
        case "--case-instruct":
            guard idx + 1 < args.count else {
                fputs("smelt trace: --case-instruct requires text\n", stderr)
                exit(1)
            }
            caseInstruct = args[idx + 1]
            idx += 2
        case "--case-speaker":
            guard idx + 1 < args.count else {
                fputs("smelt trace: --case-speaker requires text\n", stderr)
                exit(1)
            }
            caseSpeaker = args[idx + 1]
            idx += 2
        case "--case-max-steps":
            guard idx + 1 < args.count, let parsed = Int(args[idx + 1]), parsed > 0 else {
                fputs("smelt trace: --case-max-steps requires a positive integer\n", stderr)
                exit(1)
            }
            caseMaxSteps = parsed
            idx += 2
        case "--case-first-chunk":
            guard idx + 1 < args.count, let parsed = Int(args[idx + 1]), parsed > 0 else {
                fputs("smelt trace: --case-first-chunk requires a positive integer\n", stderr)
                exit(1)
            }
            caseFirstChunk = parsed
            idx += 2
        case "--case-max-chunk":
            guard idx + 1 < args.count, let parsed = Int(args[idx + 1]), parsed > 0 else {
                fputs("smelt trace: --case-max-chunk requires a positive integer\n", stderr)
                exit(1)
            }
            caseMaxChunk = parsed
            idx += 2
        case "--case-seed":
            guard idx + 1 < args.count, let parsed = UInt64(args[idx + 1]) else {
                fputs("smelt trace: --case-seed requires an unsigned integer\n", stderr)
                exit(1)
            }
            caseSeed = parsed
            idx += 2
        case "--case-greedy":
            caseGreedy = true
            idx += 1
        case "--update":
            updateSuite = true
            idx += 1
        case "--help", "-h":
            print(usage)
            return
        default:
            if arg.hasPrefix("--") {
                fputs("smelt trace: unknown option \(arg)\n", stderr)
                exit(1)
            }
            positional.append(arg)
            idx += 1
        }
    }

    do {
        let inspectOptions = SmeltTraceOptions(hashLargeArtifacts: hashLargeArtifacts)
        let runtimeCase = try makeTraceRuntimeCase(
            text: caseText,
            language: caseLanguage,
            instruct: caseInstruct,
            speaker: caseSpeaker,
            maxSteps: caseMaxSteps,
            firstChunk: caseFirstChunk,
            maxChunk: caseMaxChunk,
            seed: caseSeed,
            greedy: caseGreedy
        )
        switch mode {
        case "inspect":
            let resolvedPackagePath = try resolveTracePackagePath(packagePath, positional, usage: usage)
            _ = requireCAMPackageCapabilitiesOrExit(
                packagePath: resolvedPackagePath,
                verb: "trace"
            )
            let report = try SmeltTrace.inspect(packagePath: resolvedPackagePath, options: inspectOptions)
            if emitJSON {
                try printJSON(report)
            } else {
                printTraceReport(report)
            }
            if report.hasErrors {
                exit(2)
            }
        case "record":
            let resolvedPackagePath = try resolveTracePackagePath(packagePath, positional, usage: usage)
            let capabilities = requireCAMPackageCapabilitiesOrExit(
                packagePath: resolvedPackagePath,
                verb: "trace"
            )
            let runtimeEvents = try resolveRuntimeEvents(
                packagePath: resolvedPackagePath,
                eventsPath: eventsPath,
                runtimeCase: runtimeCase,
                capabilities: capabilities
            )
            let recordOptions = SmeltTraceRecordOptions(
                inspectOptions: inspectOptions,
                runtimeEvents: runtimeEvents
            )
            let witness = try SmeltTrace.record(
                packagePath: resolvedPackagePath,
                options: recordOptions
            )
            if let outputPath {
                try SmeltTrace.writeWitness(witness, to: outputPath)
                if emitJSON {
                    try printJSON(witness)
                } else {
                    print("wrote trace witness: \(outputPath)")
                    print("contract sha256: \(shortHash(witness.contractSHA256))")
                }
            } else {
                try printJSON(witness)
            }
            if !witness.contract.issues.filter({ $0.severity == .error }).isEmpty {
                exit(2)
            }
        case "compare":
            guard positional.count == 2 else {
                fputs(usage + "\n", stderr)
                exit(1)
            }
            let expected = try SmeltTrace.loadWitness(from: positional[0])
            let actual = try SmeltTrace.loadWitness(from: positional[1])
            let comparison = SmeltTrace.compare(expected: expected, actual: actual)
            if emitJSON {
                try printJSON(comparison)
            } else {
                printTraceComparison(comparison)
            }
            if !comparison.matches {
                exit(2)
            }
        case "verify":
            let resolvedPackagePath = try resolveTracePackagePath(packagePath, positional, usage: usage)
            let capabilities = requireCAMPackageCapabilitiesOrExit(
                packagePath: resolvedPackagePath,
                verb: "trace"
            )
            if let goldenPath {
                let runtimeEvents = try resolveRuntimeEvents(
                    packagePath: resolvedPackagePath,
                    eventsPath: eventsPath,
                    runtimeCase: runtimeCase,
                    capabilities: capabilities
                )
                let recordOptions = SmeltTraceRecordOptions(
                    inspectOptions: inspectOptions,
                    runtimeEvents: runtimeEvents
                )
                let comparison = try SmeltTrace.verify(
                    packagePath: resolvedPackagePath,
                    against: goldenPath,
                    options: recordOptions
                )
                if emitJSON {
                    try printJSON(comparison)
                } else {
                    printTraceComparison(comparison)
                }
                if !comparison.matches {
                    exit(2)
                }
            } else {
                let report = try SmeltTrace.inspect(packagePath: resolvedPackagePath, options: inspectOptions)
                if emitJSON {
                    try printJSON(report)
                } else {
                    printTraceReport(report)
                }
                if report.hasErrors {
                    exit(2)
                }
            }
        case "replay":
            guard positional.count >= 2 || (packagePath != nil && goldenPath != nil) else {
                fputs(usage + "\n", stderr)
                exit(1)
            }
            let resolvedPackagePath = try resolveTracePackagePath(
                packagePath,
                positional.isEmpty ? [] : [positional[0]],
                usage: usage
            )
            let tracePath = goldenPath ?? positional.dropFirst().first!
            let capabilities = requireCAMPackageCapabilitiesOrExit(
                packagePath: resolvedPackagePath,
                verb: "trace"
            )
            let explicitRuntimeEvents = eventsPath != nil || runtimeCase != nil
                ? try resolveRuntimeEvents(
                    packagePath: resolvedPackagePath,
                    eventsPath: eventsPath,
                    runtimeCase: runtimeCase,
                    capabilities: capabilities
                )
                : nil
            let replayOptions = SmeltTraceReplayOptions(
                inspectOptions: inspectOptions,
                runtimeEvents: explicitRuntimeEvents
            )
            let comparison = try SmeltTrace.replay(
                packagePath: resolvedPackagePath,
                from: tracePath,
                options: replayOptions
            )
            if emitJSON {
                try printJSON(comparison)
            } else {
                printTraceComparison(comparison, label: "trace replay")
            }
            if !comparison.matches {
                exit(2)
            }
        case "suite":
            guard positional.count == 1 else {
                fputs(usage + "\n", stderr)
                exit(1)
            }
            if runtimeCase != nil || eventsPath != nil || goldenPath != nil || outputPath != nil {
                throw traceCLIError(
                    "suite cases must declare their own golden/events/case input in the suite JSON"
                )
            }
            let result = try runTraceSuite(
                path: positional[0],
                packageOverride: packagePath,
                updateGoldens: updateSuite,
                inspectOptions: inspectOptions
            )
            if emitJSON {
                try printJSON(result)
            } else {
                printTraceSuiteResult(result)
            }
            if !result.matches {
                exit(2)
            }
        default:
            fputs(usage + "\n", stderr)
            exit(1)
        }
    } catch {
        fputs("smelt trace: \(error)\n", stderr)
        exit(1)
    }
}

private func loadTraceEvents(_ path: String) throws -> [SmeltTraceEvent] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    return try JSONDecoder().decode([SmeltTraceEvent].self, from: data)
}

private enum TraceRuntimeCase {
    case text(TraceTextRuntimeCase)
}

private struct TraceTextRuntimeCase {
    let text: String
    let language: String
    let instruct: String?
    let speaker: String?
    let maxSteps: Int
    let firstChunk: Int?
    let maxChunk: Int?
    let seed: UInt64?
    let greedy: Bool
}

private func makeTraceRuntimeCase(
    text: String?,
    language: String,
    instruct: String?,
    speaker: String?,
    maxSteps: Int,
    firstChunk: Int?,
    maxChunk: Int?,
    seed: UInt64?,
    greedy: Bool
) throws -> TraceRuntimeCase? {
    if let text {
        return .text(TraceTextRuntimeCase(
            text: text,
            language: language,
            instruct: instruct,
            speaker: speaker,
            maxSteps: maxSteps,
            firstChunk: firstChunk,
            maxChunk: maxChunk,
            seed: seed,
            greedy: greedy
        ))
    }
    if language != "English" || instruct != nil || speaker != nil
        || firstChunk != nil || maxChunk != nil || seed != nil || greedy
    {
        throw traceCLIError(
            "--case-language, --case-instruct, --case-speaker, "
                + "--case-first-chunk, --case-max-chunk, --case-seed, "
                + "and --case-greedy require --case-text with a tts package"
        )
    }
    return nil
}

private func resolveRuntimeEvents(
    packagePath: String,
    eventsPath: String?,
    runtimeCase: TraceRuntimeCase?,
    capabilities: SmeltCAMPackageCapabilities
) throws -> [SmeltTraceEvent] {
    if eventsPath != nil, runtimeCase != nil {
        throw traceCLIError(
            "--events and --case-text are mutually exclusive runtime event sources"
        )
    }
    if let eventsPath {
        return try loadTraceEvents(eventsPath)
    }
    guard let runtimeCase else { return [] }
    return try captureRuntimeEvents(
        packagePath: packagePath,
        runtimeCase: runtimeCase,
        capabilities: capabilities
    )
}

private func captureRuntimeEvents(
    packagePath: String,
    runtimeCase: TraceRuntimeCase,
    capabilities: SmeltCAMPackageCapabilities
) throws -> [SmeltTraceEvent] {
    try captureCAMRuntimeEvents(
        packagePath: packagePath,
        capabilities: capabilities,
        runtimeCase: runtimeCase
    )
}

private struct CAMTraceTextRoute {
    let decision: SmeltCAMPackageCapabilities.Decision

    func runtimeRouteOrExit(
        capabilities: SmeltCAMPackageCapabilities
    ) -> CAMRuntimeRoute {
        resolveCAMRuntimeRouteOrExit(
            capabilities: capabilities,
            decision: decision,
            verb: "trace"
        )
    }
}

private func captureCAMRuntimeEvents(
    packagePath: String,
    capabilities: SmeltCAMPackageCapabilities,
    runtimeCase: TraceRuntimeCase
) throws -> [SmeltTraceEvent] {
    switch runtimeCase {
    case .text(let textCase):
        if textCase.greedy, textCase.seed != nil {
            throw traceCLIError("--case-greedy and --case-seed are mutually exclusive")
        }
        let route = resolveCAMTraceTextRouteOrExit(capabilities)
        let runtimeRoute = route.runtimeRouteOrExit(capabilities: capabilities)
        switch runtimeRoute {
        case .textToText:
            let construction = makeCAMTextRuntimeConstructionOrExit(
                packagePath: packagePath,
                capabilities: capabilities,
                decision: route.decision,
                verb: "trace"
            )
            let report = try SmeltTrace.inspect(packagePath: packagePath)
            return SmeltTraceCAMRoute.events(
                witness: makeCAMTraceRouteWitness(
                    decision: route.decision,
                    capabilities: capabilities
                ),
                followedBy: try captureTextToTextCAMTraceEvents(
                    packagePath: packagePath,
                    report: report,
                    runtimeCase: textCase,
                    construction: construction
                )
            )
        case .textToPCM(let outputRate):
            switch outputRate {
            case "24khz":
                let construction = makeCAMTextToPCMRuntimeConstructionOrExit(
                    packagePath: packagePath,
                    capabilities: capabilities,
                    decision: route.decision,
                    verb: "trace"
                )
                let report = try SmeltTrace.inspect(packagePath: packagePath)
                return SmeltTraceCAMRoute.events(
                    witness: makeCAMTraceRouteWitness(
                        decision: route.decision,
                        capabilities: capabilities
                    ),
                    followedBy: try captureCAMTextToPCMTraceEvents(
                        packagePath: packagePath,
                        report: report,
                        runtimeCase: textCase,
                        construction: construction
                    )
                )
            default:
                throw traceCLIError(
                    "unsupported CAM audio output rate '\(outputRate)'"
                )
            }
    }
}
}

private func resolveCAMTraceTextRouteOrExit(
    _ capabilities: SmeltCAMPackageCapabilities
) -> CAMTraceTextRoute {
    let text = resolveOptionalCAMTraceDecision(capabilities, request: .traceTextGenerate)
    let audio = resolveOptionalCAMTraceDecision(capabilities, request: .traceTextSynthesize)
        ?? resolveOptionalCAMTraceDecision(capabilities, request: .traceAudioSynthesis)
    switch (text, audio) {
    case (.some(let text), nil):
        return CAMTraceTextRoute(decision: text)
    case (nil, .some(let audio)):
        return CAMTraceTextRoute(decision: audio)
    case (.some(let text), .some(let audio)):
        fputs(
            "smelt trace: CAM --case-text is ambiguous: "
                + "\(describeCAMDecision(text)) and \(describeCAMDecision(audio))\n",
            stderr
        )
        exit(1)
    case (nil, nil):
        fputs("smelt trace: no CAM export satisfies trace text request\n", stderr)
        exit(1)
    }
}

private func resolveOptionalCAMTraceDecision(
    _ capabilities: SmeltCAMPackageCapabilities,
    request: SmeltCAMCapabilityRequest
) -> SmeltCAMPackageCapabilities.Decision? {
    do {
        return try capabilities.resolve(request)
    } catch SmeltCAMPackageCapabilitiesError.noMatchingExport {
        return nil
    } catch {
        fputs("smelt trace: \(error)\n", stderr)
        exit(1)
    }
}

private func captureTextToTextCAMTraceEvents(
    packagePath: String,
    report: SmeltTraceReport,
    runtimeCase: TraceTextRuntimeCase,
    construction: CAMTextRuntimeConstruction
) throws -> [SmeltTraceEvent] {
    try captureTextToTextTraceBackendEvents(
        packagePath: packagePath,
        report: report,
        runtimeCase: runtimeCase,
        caseLabel: .camTextToText,
        construction: construction
    )
}

private func captureCAMTextToPCMTraceEvents(
    packagePath: String,
    report: SmeltTraceReport,
    runtimeCase: TraceTextRuntimeCase,
    construction: CAMTextToPCMRuntimeConstruction
) throws -> [SmeltTraceEvent] {
    try construction.requirePackagePath(packagePath)
    return try captureTextToPCMTraceBackendEvents(
        packagePath: packagePath,
        report: report,
        runtimeCase: runtimeCase,
        caseLabel: .camTextToPCM,
        construction: construction
    )
}

private enum TextToTextTraceCaseLabel {
    case manifestTextToText
    case camTextToText

    func fields(runtimeCase: TraceTextRuntimeCase) -> [String] {
        let leadingField: String
        switch self {
        case .manifestTextToText:
            leadingField = "io=text->text"
        case .camTextToText:
            leadingField = "io=text->text"
        }
        return [
            leadingField,
            "textSHA256=\(sha256Hex(Data(runtimeCase.text.utf8)))",
            "maxTokens=\(runtimeCase.maxSteps)",
            "decode=argmax",
        ]
    }
}

private enum TextToPCMTraceCaseLabel {
    case camTextToPCM

    func fields(runtimeCase: TraceTextRuntimeCase) -> [String] {
        return [
            "io=text->pcm",
            "textSHA256=\(sha256Hex(Data(runtimeCase.text.utf8)))",
            "language=\(runtimeCase.language)",
            "maxSteps=\(runtimeCase.maxSteps)",
            "decode=\(runtimeCase.seed.map { "sampleSeeded:\($0)" } ?? "greedy")",
        ]
    }
}

private func captureTextToTextTraceBackendEvents(
    packagePath: String,
    report: SmeltTraceReport,
    runtimeCase: TraceTextRuntimeCase,
    caseLabel: TextToTextTraceCaseLabel = .manifestTextToText,
    construction: CAMTextRuntimeConstruction? = nil
) throws -> [SmeltTraceEvent] {
    try captureTextToTextTraceRuntimeEvents(
        packagePath: packagePath,
        report: report,
        runtimeCase: runtimeCase,
        caseLabel: caseLabel,
        construction: construction
    )
}

private func captureTextToPCMTraceBackendEvents(
    packagePath: String,
    report: SmeltTraceReport,
    runtimeCase: TraceTextRuntimeCase,
    caseLabel: TextToPCMTraceCaseLabel,
    construction: CAMTextToPCMRuntimeConstruction
) throws -> [SmeltTraceEvent] {
    try captureTextToPCMTraceRuntimeEvents(
        packagePath: packagePath,
        report: report,
        runtimeCase: runtimeCase,
        caseLabel: caseLabel,
        construction: construction
    )
}

private func captureTextToPCMTraceRuntimeEvents(
    packagePath: String,
    report: SmeltTraceReport,
    runtimeCase: TraceTextRuntimeCase,
    caseLabel: TextToPCMTraceCaseLabel,
    construction: CAMTextToPCMRuntimeConstruction
) throws -> [SmeltTraceEvent] {
    let recorder = SmeltRuntimeTraceRecorder()
    let routesByBlock = traceRoutesByBlock(report)
    let decode: CAMTextToPCMDecodeMode = runtimeCase.seed.map {
        .sampleSeeded($0)
    } ?? .greedy
    recorder.record(
        kind: "case",
        witness: caseLabel.fields(runtimeCase: runtimeCase).joined(separator: ":")
    )
    recorder.record(
        kind: "block-finish",
        phase: "input:text",
        block: "tts-frontend",
        route: routesByBlock["tts-frontend"],
        witness: "textSHA256=\(sha256Hex(Data(runtimeCase.text.utf8)))"
    )

    let runtime = try construction.make24KRuntime(verb: "trace")
    var chunkCount = 0
    var sampleCount = 0
    try runtime.generateStreaming(
        text: runtimeCase.text,
        instruct: runtimeCase.instruct,
        language: runtimeCase.language,
        speaker: runtimeCase.speaker,
        maxFrames: runtimeCase.maxSteps,
        decode: decode,
        firstChunkFrames: runtimeCase.firstChunk,
        maxChunkFrames: runtimeCase.maxChunk,
        trace: recorder
    ) { chunk in
        chunkCount += 1
        sampleCount += chunk.samples.count
        recorder.record(
            kind: "output-chunk",
            phase: "emission",
            block: "codec-decoder",
            route: routesByBlock["codec-decoder"],
            step: chunk.frameOffset,
            witness: [
                "frames=\(chunk.frameCount)",
                "samples=\(chunk.samples.count)",
                "final=\(chunk.isFinal)",
                "sha256=\(sha256Hex(chunk.samples))",
            ].joined(separator: ":")
        )
        return true
    }
    recorder.record(
        kind: "case-result",
        witness: "chunks=\(chunkCount):samples=\(sampleCount)"
    )
    return recorder.events
}

private func captureTextToTextTraceRuntimeEvents(
    packagePath: String,
    report: SmeltTraceReport,
    runtimeCase: TraceTextRuntimeCase,
    caseLabel: TextToTextTraceCaseLabel = .manifestTextToText,
    construction: CAMTextRuntimeConstruction? = nil
) throws -> [SmeltTraceEvent] {
    if runtimeCase.seed != nil {
        throw traceCLIError("--case-seed is currently implemented for tts case recording only")
    }
    if runtimeCase.instruct != nil || runtimeCase.speaker != nil
        || runtimeCase.firstChunk != nil || runtimeCase.maxChunk != nil
        || runtimeCase.language != "English"
    {
        throw traceCLIError(
            "--case-language, --case-instruct, --case-speaker, "
                + "--case-first-chunk, and --case-max-chunk are tts-only"
        )
    }

    let recorder = SmeltRuntimeTraceRecorder()
    let routesByBlock = traceRoutesByBlock(report)
    recorder.record(
        kind: "case",
        witness: caseLabel.fields(runtimeCase: runtimeCase).joined(separator: ":")
    )
    recorder.record(
        kind: "block-finish",
        phase: "input:tokenize",
        block: "tokenizer",
        route: routesByBlock["tokenizer"],
        witness: "textSHA256=\(sha256Hex(Data(runtimeCase.text.utf8)))"
    )

    let result = try evaluatePrompt(
        packagePath: packagePath,
        prompt: runtimeCase.text,
        maxTokens: runtimeCase.maxSteps,
        template: "",
        selectionMode: .argmax,
        selectionDescription: "argmax",
        construction: construction
    )
    recorder.record(
        kind: "prefill",
        phase: "setup:prefill",
        witness: [
            "promptTokens=\(result.promptTokens)",
            "mode=\(result.mode)",
            "selection=\(result.selection)",
        ].joined(separator: ":")
    )
    recorder.record(
        kind: "block-finish",
        phase: "setup:prefill",
        block: "trunk",
        route: routesByBlock["trunk"],
        witness: "promptTokens=\(result.promptTokens):mode=\(result.mode)"
    )
    recorder.record(
        kind: "block-finish",
        phase: "setup:prefill",
        block: "text-head",
        route: routesByBlock["text-head"],
        witness: "selection=\(result.selection)"
    )
    for (offset, token) in result.generated.enumerated() {
        recorder.record(
            kind: "block-finish",
            phase: "per-step:decode",
            block: "trunk",
            route: routesByBlock["trunk"],
            step: offset,
            witness: "token=\(token)"
        )
        recorder.record(
            kind: "block-finish",
            phase: "per-step:decode",
            block: "text-head",
            route: routesByBlock["text-head"],
            step: offset,
            witness: "token=\(token)"
        )
        recorder.record(
            kind: "output-token",
            phase: "decode",
            step: offset,
            witness: "token=\(token)"
        )
    }
    recorder.record(
        kind: "case-result",
        witness: [
            "tokens=\(result.generated.count)",
            "completionSHA256=\(sha256Hex(Data(result.completion.utf8)))",
        ].joined(separator: ":")
    )
    return recorder.events
}

private func traceRoutesByBlock(_ report: SmeltTraceReport) -> [String: String] {
    Dictionary(report.blocks.map { ($0.name, $0.route) }, uniquingKeysWith: { first, _ in first })
}

private struct TraceSuite: Codable {
    let schemaVersion: Int?
    let package: String?
    let cases: [TraceSuiteCase]
}

private struct TraceSuiteCase: Codable {
    let name: String
    let package: String?
    let golden: String
    let events: String?
    let caseText: String?
    let caseLanguage: String?
    let caseInstruct: String?
    let caseSpeaker: String?
    let caseMaxSteps: Int?
    let caseFirstChunk: Int?
    let caseMaxChunk: Int?
    let caseSeed: UInt64?
    let caseGreedy: Bool?
}

private struct TraceSuiteResult: Codable {
    let suite: String
    let matches: Bool
    let cases: [TraceSuiteCaseResult]
}

private struct TraceSuiteCaseResult: Codable {
    let name: String
    let package: String
    let golden: String
    let matches: Bool
    let updated: Bool
    let differences: [SmeltTraceDifference]
    let error: String?
}

private func runTraceSuite(
    path: String,
    packageOverride: String?,
    updateGoldens: Bool,
    inspectOptions: SmeltTraceOptions
) throws -> TraceSuiteResult {
    let suiteURL = URL(fileURLWithPath: path)
    let suiteData = try Data(contentsOf: suiteURL)
    let suite = try JSONDecoder().decode(TraceSuite.self, from: suiteData)
    try validateTraceSuite(suite)
    if updateGoldens, packageOverride != nil {
        throw traceCLIError("trace suite cannot update goldens while using a package override")
    }
    let baseURL = suiteURL.deletingLastPathComponent()
    if !suiteRequiresCLIRuntime(suite) {
        try preflightTraceSuitePackages(
            suite,
            suiteBaseURL: baseURL,
            packageOverride: packageOverride
        )
        let result = try SmeltTrace.verifySuite(
            path: path,
            options: SmeltTraceSuiteOptions(
                inspectOptions: inspectOptions,
                packageOverride: packageOverride,
                updateGoldens: updateGoldens
            )
        )
        return TraceSuiteResult(
            suite: result.suite,
            matches: result.matches,
            cases: result.cases.map {
                TraceSuiteCaseResult(
                    name: $0.name,
                    package: $0.package,
                    golden: $0.golden,
                    matches: $0.matches,
                    updated: $0.updated,
                    differences: $0.differences,
                    error: $0.error
                )
            }
        )
    }
    let results = suite.cases.map { testCase in
        runTraceSuiteCase(
            testCase,
            suite: suite,
            suiteBaseURL: baseURL,
            packageOverride: packageOverride,
            updateGolden: updateGoldens,
            inspectOptions: inspectOptions
        )
    }
    return TraceSuiteResult(
        suite: path,
        matches: results.allSatisfy(\.matches),
        cases: results
    )
}

private func validateTraceSuite(_ suite: TraceSuite) throws {
    let staticSuite = SmeltTraceSuiteSpec(
        schemaVersion: suite.schemaVersion,
        package: suite.package,
        cases: suite.cases.map {
            SmeltTraceSuiteCaseSpec(
                name: $0.name,
                package: $0.package,
                golden: $0.golden,
                events: $0.events
            )
        }
    )
    try SmeltTrace.validateSuite(staticSuite)

    for testCase in suite.cases {
        let name = testCase.name.trimmedTraceValue
        if let caseLanguage = testCase.caseLanguage, caseLanguage.trimmedTraceValue.isEmpty {
            throw traceCLIError("case '\(name)' has blank caseLanguage")
        }
        if let maxSteps = testCase.caseMaxSteps, maxSteps <= 0 {
            throw traceCLIError("case '\(name)' caseMaxSteps must be positive")
        }
        if let firstChunk = testCase.caseFirstChunk, firstChunk <= 0 {
            throw traceCLIError("case '\(name)' caseFirstChunk must be positive")
        }
        if let maxChunk = testCase.caseMaxChunk, maxChunk <= 0 {
            throw traceCLIError("case '\(name)' caseMaxChunk must be positive")
        }

        let hasRuntimeSource = testCase.caseText != nil
        let hasRuntimeFields = testCase.hasRuntimeFields
        if testCase.events != nil, hasRuntimeFields {
            throw traceCLIError(
                "case '\(name)' cannot mix events with caseText or case options"
            )
        }
        if !hasRuntimeSource, hasRuntimeFields {
            throw traceCLIError(
                "case '\(name)' case options require caseText"
            )
        }
    }
}

private func suiteRequiresCLIRuntime(_ suite: TraceSuite) -> Bool {
    suite.cases.contains {
        $0.hasRuntimeFields
    }
}

private func preflightTraceSuitePackages(
    _ suite: TraceSuite,
    suiteBaseURL: URL,
    packageOverride: String?
) throws {
    for testCase in suite.cases {
        let package = try resolveTraceSuitePackage(
            testCase,
            suite: suite,
            suiteBaseURL: suiteBaseURL,
            packageOverride: packageOverride
        )
        _ = requireCAMPackageCapabilitiesOrExit(packagePath: package, verb: "trace")
    }
}

private func resolveTraceSuitePackage(
    _ testCase: TraceSuiteCase,
    suite: TraceSuite,
    suiteBaseURL: URL,
    packageOverride: String?
) throws -> String {
    let packageRaw = packageOverride ?? testCase.package ?? suite.package ?? ""
    guard !packageRaw.isEmpty else {
        throw traceCLIError("case '\(testCase.name)' has no package")
    }
    return packageOverride != nil
        ? packageRaw
        : resolveSuitePath(packageRaw, relativeTo: suiteBaseURL)
}

private func runTraceSuiteCase(
    _ testCase: TraceSuiteCase,
    suite: TraceSuite,
    suiteBaseURL: URL,
    packageOverride: String?,
    updateGolden: Bool,
    inspectOptions: SmeltTraceOptions
) -> TraceSuiteCaseResult {
    var package = ""
    let golden = resolveSuitePath(testCase.golden, relativeTo: suiteBaseURL)
    do {
        package = try resolveTraceSuitePackage(
            testCase,
            suite: suite,
            suiteBaseURL: suiteBaseURL,
            packageOverride: packageOverride
        )
        let capabilities = requireCAMPackageCapabilitiesOrExit(
            packagePath: package,
            verb: "trace"
        )
        let eventsPath = testCase.events.map {
            resolveSuitePath($0, relativeTo: suiteBaseURL)
        }
        let runtimeCase = try makeTraceRuntimeCase(
            text: testCase.caseText,
            language: testCase.caseLanguage ?? "English",
            instruct: testCase.caseInstruct,
            speaker: testCase.caseSpeaker,
            maxSteps: testCase.caseMaxSteps ?? 8,
            firstChunk: testCase.caseFirstChunk,
            maxChunk: testCase.caseMaxChunk,
            seed: testCase.caseSeed,
            greedy: testCase.caseGreedy ?? false
        )
        let runtimeEvents = try resolveRuntimeEvents(
            packagePath: package,
            eventsPath: eventsPath,
            runtimeCase: runtimeCase,
            capabilities: capabilities
        )
        if updateGolden {
            let witness = try SmeltTrace.record(
                packagePath: package,
                options: SmeltTraceRecordOptions(
                    inspectOptions: inspectOptions,
                    runtimeEvents: runtimeEvents
                )
            )
            let errorIssues = witness.contract.issues.filter { $0.severity == .error }
            if !errorIssues.isEmpty {
                let codes = errorIssues.map(\.code).joined(separator: ", ")
                throw traceCLIError("case '\(testCase.name)' produced trace errors: \(codes)")
            }
            let goldenURL = URL(fileURLWithPath: golden)
            try FileManager.default.createDirectory(
                at: goldenURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try SmeltTrace.writeWitness(witness, to: golden)
            return TraceSuiteCaseResult(
                name: testCase.name,
                package: package,
                golden: golden,
                matches: true,
                updated: true,
                differences: [],
                error: nil
            )
        }
        let comparison = try SmeltTrace.verify(
            packagePath: package,
            against: golden,
            options: SmeltTraceRecordOptions(
                inspectOptions: inspectOptions,
                runtimeEvents: runtimeEvents
            )
        )
        return TraceSuiteCaseResult(
            name: testCase.name,
            package: package,
            golden: golden,
            matches: comparison.matches,
            updated: false,
            differences: comparison.differences,
            error: nil
        )
    } catch {
        return TraceSuiteCaseResult(
            name: testCase.name,
            package: package,
            golden: golden,
            matches: false,
            updated: false,
            differences: [],
            error: "\(error)"
        )
    }
}

private func resolveSuitePath(_ path: String, relativeTo baseURL: URL) -> String {
    guard !(path as NSString).isAbsolutePath else { return path }
    return baseURL.appendingPathComponent(path).path
}

private func traceCLIError(_ message: String) -> NSError {
    NSError(
        domain: "SmeltTraceCLI",
        code: 1,
        userInfo: [NSLocalizedDescriptionKey: message]
    )
}

private extension TraceSuiteCase {
    var hasRuntimeFields: Bool {
        caseText != nil
            || caseLanguage != nil
            || caseInstruct != nil
            || caseSpeaker != nil
            || caseMaxSteps != nil
            || caseFirstChunk != nil
            || caseMaxChunk != nil
            || caseSeed != nil
            || caseGreedy != nil
    }

}

private extension String {
    var trimmedTraceValue: String {
        trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

private func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func sha256Hex(_ samples: [Float]) -> String {
    let data = samples.withUnsafeBufferPointer { Data(buffer: $0) }
    return sha256Hex(data)
}

private func resolveTracePackagePath(
    _ flagged: String?,
    _ positional: [String],
    usage: String
) throws -> String {
    if let flagged { return flagged }
    if let first = positional.first { return first }
    if let inferred = inferPackagePathFromCWD() { return inferred }
    fputs(usage + "\n", stderr)
    exit(1)
}

private func printJSON<T: Encodable>(_ value: T) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    let data = try encoder.encode(value)
    FileHandle.standardOutput.write(data)
    print()
}

private func printTraceReport(_ report: SmeltTraceReport) {
    print("smelt trace")
    print("package: \(report.packagePath)")
    print("kind: \(report.packageKind)")
    if let modelName = report.modelName {
        print("model: \(modelName)")
    }
    if let manifestKind = report.manifestKind {
        print("manifest kind: \(manifestKind)")
    }
    print("manifest sha256: \(shortHash(report.manifestSHA256))")
    if let buildFingerprint = report.buildFingerprint {
        print("build fingerprint: \(buildFingerprint)")
    }
    if let graph = report.graph {
        print("graph: \(graph.source) \(graph.signature ?? "unknown") blocks=\(graph.blockCount)")
    } else {
        print("graph: none")
    }
    if let loop = report.loop {
        print("loop: \(loop.source) setup=\(loop.setup.count) per-step=\(loop.perStep.count) emission=\(loop.emission)")
    } else {
        print("loop: none")
    }
    print("status: \(report.hasErrors ? "error" : "ok") errors=\(report.errorCount) warnings=\(report.warningCount)")

    printSection("loop")
    if let loop = report.loop {
        print("  setup:")
        for phase in loop.setup {
            print("    \(phase.name): \(phase.blocks.joined(separator: ", ")) feedsNextStep=\(phase.feedsNextStep)")
        }
        print("  per-step:")
        for phase in loop.perStep {
            print("    \(phase.name): \(phase.blocks.joined(separator: ", ")) feedsNextStep=\(phase.feedsNextStep)")
        }
        print("  emission: \(loop.emission)")
        print("  stop: \(loop.stop.joined(separator: ", "))")
    } else {
        print("  none")
    }

    printSection("blocks")
    if report.blocks.isEmpty {
        print("  none")
    } else {
        for block in report.blocks {
            print(
                "  \(statusLabel(block.status)) \(block.name) "
                    + "\(block.role) \(block.route) "
                    + "\(block.inputs.joined(separator: "+")) -> \(block.output)"
            )
            if let feedback = block.feedback {
                print("      feedback: \(feedback)")
            }
            if !block.state.isEmpty {
                print("      state: \(block.state.joined(separator: ", "))")
            }
            if !block.sideOutputs.isEmpty {
                print("      side outputs: \(block.sideOutputs.joined(separator: ", "))")
            }
            for item in block.evidence {
                print("      \(item)")
            }
        }
    }

    printSection("sidecars")
    if report.sidecars.isEmpty {
        print("  none")
    } else {
        for sidecar in report.sidecars {
            let status: SmeltTraceStatus = sidecar.exists
                ? (sidecar.issues.contains { $0.severity == .error } ? .error : .ok)
                : .warning
            print(
                "  \(statusLabel(status)) \(sidecar.name) "
                    + "\(sidecar.packageKind ?? "missing") "
                    + "manifest=\(sidecar.manifestSHA256.map(shortHash) ?? "none")"
            )
            for table in sidecar.dispatchTables {
                print("      \(dispatchSummary(table))")
            }
            for issue in sidecar.issues {
                print("      \(issue.severity.rawValue): \(issue.code) \(issue.message)")
            }
        }
    }

    printSection("dispatch tables")
    if report.dispatchTables.isEmpty {
        print("  none")
    } else {
        for table in report.dispatchTables {
            print("  \(dispatchSummary(table))")
            if !table.topPipelines.isEmpty {
                let top = table.topPipelines
                    .map { "\($0.name)=\($0.dispatchCount)" }
                    .joined(separator: ", ")
                print("      top: \(top)")
            }
        }
    }

    printSection("artifacts")
    if report.artifacts.isEmpty {
        print("  none")
    } else {
        for artifact in report.artifacts {
            let status = artifact.exists ? "OK" : "MISS"
            var line = "  \(status) \(artifact.name) \(artifact.kind)"
            if let bytes = artifact.bytes {
                line += " bytes=\(bytes)"
            }
            if let actual = artifact.actualSHA256 {
                line += " sha256=\(shortHash(actual))"
            }
            if let skipped = artifact.hashSkippedReason {
                line += " hash-skipped=\"\(skipped)\""
            }
            print(line)
        }
    }

    printSection("dtypes")
    if report.dtypeSummary.isEmpty {
        print("  none")
    } else {
        for count in report.dtypeSummary {
            print("  \(count.dtype): \(count.count)")
        }
    }

    printSection("issues")
    if report.issues.isEmpty {
        print("  none")
    } else {
        for issue in report.issues {
            print("  \(issue.severity.rawValue): \(issue.code) \(issue.message)")
        }
    }
}

private func printSection(_ name: String) {
    print("")
    print("\(name):")
}

private func printTraceComparison(
    _ comparison: SmeltTraceComparison,
    label: String = "trace compare"
) {
    print(label)
    print("status: \(comparison.matches ? "match" : "mismatch")")
    guard !comparison.differences.isEmpty else { return }
    print("")
    print("differences:")
    for difference in comparison.differences {
        print("  \(difference.path)")
        print("    expected: \(oneLine(difference.expected))")
        print("    actual:   \(oneLine(difference.actual))")
    }
}

private func printTraceSuiteResult(_ result: TraceSuiteResult) {
    print("trace suite")
    print("suite: \(result.suite)")
    print("status: \(result.matches ? "match" : "mismatch") cases=\(result.cases.count)")
    guard !result.cases.isEmpty else { return }
    print("")
    print("cases:")
    for testCase in result.cases {
        let status = testCase.updated ? "UPDATED" : (testCase.matches ? "OK" : "FAIL")
        print("  \(status) \(testCase.name)")
        print("      package: \(testCase.package)")
        print("      golden: \(testCase.golden)")
        if let error = testCase.error {
            print("      error: \(error)")
        }
        for difference in testCase.differences {
            print("      \(difference.path)")
            print("        expected: \(oneLine(difference.expected))")
            print("        actual:   \(oneLine(difference.actual))")
        }
    }
}

private func oneLine(_ text: String?) -> String {
    guard let text else { return "nil" }
    return text
        .split(whereSeparator: \.isNewline)
        .map(String.init)
        .joined(separator: " ")
}

private func statusLabel(_ status: SmeltTraceStatus) -> String {
    switch status {
    case .ok: return "OK"
    case .warning: return "WARN"
    case .error: return "ERR"
    }
}

private func dispatchSummary(_ table: SmeltTraceDispatchTable) -> String {
    guard table.exists else {
        return "MISS \(table.name)"
    }
    var line = "OK \(table.name)"
    if let total = table.totalRecords {
        line += " records=\(total)"
    }
    if let dispatches = table.dispatchCount {
        line += " dispatches=\(dispatches)"
    }
    if let swaps = table.swapCount {
        line += " swaps=\(swaps)"
    }
    if let hash = table.sha256 {
        line += " sha256=\(shortHash(hash))"
    }
    if let parseError = table.parseError {
        line += " parse-error=\"\(parseError)\""
    }
    return line
}

private func shortHash(_ hash: String) -> String {
    String(hash.prefix(12))
}
