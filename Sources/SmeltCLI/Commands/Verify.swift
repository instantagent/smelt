import Foundation
import SmeltRuntime

func runVerifyCommand() {
    guard args.count >= 3 else {
        fputs(
            "Usage: smelt verify <model.smeltpkg> [--baseline pkg] [--prompt-file FILE] [--template NAME] [--max-tokens N] [--decode-iterations N] [--prefill-iterations N] [--prefill-tokens 64,256] [--prefill-verify-tokens 63,64,65,...] [--prefill-continuation-tokens N] [--prefill-continuation-route plain|suffix] [--prefill-continuation-k N] [--suffix-max-n N] [--min-decode-tps N] [--max-decode-p95-ms N] [--min-prefill-tps 64=...,256=...] [--max-prefill-p95-ms 64=...,256=...] [--benchmark-settle-timeout N] [--benchmark-settle-interval N]\n",
            stderr
        )
        exit(1)
    }
    let candidatePath = args[2]
    let retiredGateFlags = [
        "--gate-qwen35",
        "--gate-llama32",
    ]
    if let retiredGateFlag = retiredGateFlags.first(where: args.contains) {
        fputs(
            "smelt verify: \(retiredGateFlag) was removed; use module gate contracts or explicit threshold flags\n",
            stderr
        )
        exit(1)
    }
    let baselinePath = parseArg("--baseline")
    let promptFile = parseArg("--prompt-file")
    let templateOverride = hasArg("--template") ? parseArg("--template") : nil
    let requestedMaxTokens = Int(parseArg("--max-tokens", default: "8")) ?? 8
    let decodeIterations = Int(parseArg("--decode-iterations", default: "20")) ?? 20
    let prefillIterations = Int(parseArg("--prefill-iterations", default: "5")) ?? 5
    let prefillTokenCounts = parseCSVInts(
        parseArg("--prefill-tokens", default: "64,256")
    )
    let explicitPrefillVerifyTokenCounts = parseCSVInts(
        parseArg("--prefill-verify-tokens", default: "")
    )
    let prefillContinuationTokens = max(
        Int(parseArg("--prefill-continuation-tokens", default: "0")) ?? 0,
        0
    )
    let prefillContinuationK = max(
        Int(parseArg("--prefill-continuation-k", default: "31")) ?? 31,
        1
    )
    let suffixMaxNeedleLength = max(
        Int(parseArg("--suffix-max-n", default: "4")) ?? 4,
        1
    )
    let prefillContinuationRoute: PrefillContinuationRoute
    switch parseArg("--prefill-continuation-route", default: "plain") {
    case "plain":
        prefillContinuationRoute = .plain
    case "suffix":
        prefillContinuationRoute = .suffixLookup(
            K: prefillContinuationK,
            maxNeedleLength: suffixMaxNeedleLength
        )
    case let route:
        fputs(
            "smelt verify: --prefill-continuation-route requires plain or suffix; got '\(route)'\n",
            stderr
        )
        exit(1)
    }
    let minDecodeTpsArg = Double(parseArg("--min-decode-tps", default: ""))
    let maxDecodeP95MsArg = Double(parseArg("--max-decode-p95-ms", default: ""))
    let minPrefillTps = parseCSVIntDoubleMap(
        parseArg("--min-prefill-tps", default: "")
    )
    let maxPrefillP95Ms = parseCSVIntDoubleMap(
        parseArg("--max-prefill-p95-ms", default: "")
    )
    let settleConfig = parseBenchmarkSettleConfig()

    let candidateConstruction = requireCAMTextRuntimePlanOrExit(
        packagePath: candidatePath,
        request: .benchDecode,
        verb: "verify"
    )
    let baselineConstruction = baselinePath.isEmpty
        ? nil
        : requireCAMTextRuntimePlanOrExit(
            packagePath: baselinePath,
            request: .benchDecode,
            verb: "verify"
        )
    let maxTokens = candidateConstruction.effectiveMaxTokens(requestedMaxTokens)

    do {
        let minDecodeTps = minDecodeTpsArg
        let maxDecodeP95Ms = maxDecodeP95MsArg
        var gateFailures: [String] = []
        let candidateIntegrity = try SmeltPackageIntegrity.verify(
            packagePath: candidatePath,
            includeWeights: true
        )
        printPackageIntegrityReport(candidateIntegrity, label: "Candidate")
        if let candidateDecodeStructure = try SmeltPackageStructure.inspectDecode(
            packagePath: candidatePath
        ) {
            printDispatchStructureReport(
                candidateDecodeStructure,
                label: "Candidate",
                kind: "decode"
            )
        }
        if let candidatePrefillStructure = try SmeltPackageStructure.inspectPrefill(
            packagePath: candidatePath
        ) {
            printDispatchStructureReport(
                candidatePrefillStructure,
                label: "Candidate",
                kind: "prefill"
            )
        }
        if let candidateVerifyArgmaxStructure = try SmeltPackageStructure
            .inspectPrefillVerifyArgmax(packagePath: candidatePath) {
            printDispatchStructureReport(
                candidateVerifyArgmaxStructure,
                label: "Candidate",
                kind: "prefill_verify_argmax"
            )
        }
        if !baselinePath.isEmpty {
            let baselineIntegrity = try SmeltPackageIntegrity.verify(
                packagePath: baselinePath,
                includeWeights: true
            )
            printPackageIntegrityReport(baselineIntegrity, label: "Baseline")
        }

        let prompts = try loadVerifyPrompts(
            path: promptFile.isEmpty ? nil : promptFile
        )
        guard !prompts.isEmpty else {
            throw NSError(
                domain: "SmeltCLI",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "No prompts found for verification"
                ]
            )
        }
        let template = try resolveVerifyChatTemplate(
            construction: candidateConstruction,
            cliOverride: templateOverride
        )

        fputs("Verify candidate: \(candidatePath)\n", stderr)
        if !baselinePath.isEmpty {
            fputs("Verify baseline:  \(baselinePath)\n", stderr)
        }
        fputs(
            "Prompt corpus: \(prompts.count) prompts, template=\(template), maxTokens=\(maxTokens)\n\n",
            stderr
        )

        var mismatchCount = 0
        for (idx, prompt) in prompts.enumerated() {
            let candidate = try evaluatePrompt(
                packagePath: candidatePath,
                prompt: prompt,
                maxTokens: maxTokens,
                template: template,
                construction: candidateConstruction
            )

            if baselinePath.isEmpty {
                fputs(
                    "  [\(idx + 1)/\(prompts.count)] ok  tokens=\(candidate.generated.count)"
                        + " promptTokens=\(candidate.promptTokens)"
                        + " mode=\(candidate.mode)\n",
                    stderr
                )
                continue
            }

            let baseline = try evaluatePrompt(
                packagePath: baselinePath,
                prompt: prompt,
                maxTokens: maxTokens,
                template: template,
                construction: baselineConstruction
            )

            if candidate.generated == baseline.generated {
                fputs(
                    "  [\(idx + 1)/\(prompts.count)] ok  ids=\(candidate.generated)\n",
                    stderr
                )
            } else {
                mismatchCount += 1
                fputs(
                    "  [\(idx + 1)/\(prompts.count)] mismatch\n"
                        + "    prompt: \(prompt)\n"
                        + "    baseline ids:  \(baseline.generated)\n"
                        + "    candidate ids: \(candidate.generated)\n"
                        + "    baseline completion:\n\(baseline.completion)\n"
                        + "    candidate completion:\n\(candidate.completion)\n",
                    stderr
                )
            }
        }

        fputs("\nDecode benchmark:\n", stderr)
        let decode = try benchmarkDecode(
            packagePath: candidatePath,
            iterations: decodeIterations,
            construction: candidateConstruction
        )
        fputs(
            "  median total:   \(String(format: "%.2f", decode.medianTotalMs))ms/tok"
                + "  (\(String(format: "%.1f", decode.tokensPerSecond)) tok/s)\n",
            stderr
        )
        fputs(
            "  p95 total:      \(String(format: "%.2f", decode.p95TotalMs))ms/tok\n",
            stderr
        )
        fputs(
            "  median pure GPU: \(String(format: "%.2f", decode.medianPureGpuMs))ms/tok\n",
            stderr
        )
        fputs(
            "  p95 pure GPU:    \(String(format: "%.2f", decode.p95PureGpuMs))ms/tok\n",
            stderr
        )
        fputs(
            "  median CPU:      \(String(format: "%.3f", decode.medianCpuMs))ms/tok\n",
            stderr
        )
        fputs(
            "  p95 CPU:         \(String(format: "%.3f", decode.p95CpuMs))ms/tok\n",
            stderr
        )
        fputs(
            "  median submit:   \(String(format: "%.3f", decode.medianSubmitGapMs))ms/tok\n",
            stderr
        )
        fputs(
            "  p95 submit:      \(String(format: "%.3f", decode.p95SubmitGapMs))ms/tok\n",
            stderr
        )
        fputs(
            "  median read:     \(String(format: "%.3f", decode.medianReadMs))ms/tok\n",
            stderr
        )
        fputs(
            "  p95 read:        \(String(format: "%.3f", decode.p95ReadMs))ms/tok\n",
            stderr
        )
        gateFailures += decodeGateFailures(
            result: decode,
            minTps: minDecodeTps,
            maxP95Ms: maxDecodeP95Ms
        )

        let candidateRuntime = try candidateConstruction.makeRuntime(contextLimit: nil)
        var prefillMismatchCount = 0
        if candidateRuntime.hasMetalPrefill {
            let prefillVerifyTokenCounts =
                explicitPrefillVerifyTokenCounts.isEmpty
                ? defaultPrefillVerifyTokenCounts(
                    maxPrefillBatchSize: candidateRuntime.maxPrefillBatchSize
                )
                : explicitPrefillVerifyTokenCounts

            if !prefillVerifyTokenCounts.isEmpty {
                fputs("\nPrefill parity:\n", stderr)
                let parityResults = try verifyPrefillParity(
                    packagePath: candidatePath,
                    tokenCounts: prefillVerifyTokenCounts,
                    continuationTokenCount: prefillContinuationTokens,
                    continuationRoute: prefillContinuationRoute,
                    template: template,
                    construction: candidateConstruction
                )
                if prefillContinuationTokens > 0 {
                    fputs(
                        "  continuation route: \(prefillContinuationRoute.label)\n",
                        stderr
                    )
                }
                for result in parityResults {
                    if result.matches {
                        let continuation = result.decodeContinuation.isEmpty
                            ? ""
                            : "  continuation=\(result.decodeContinuation.count) exact"
                        fputs(
                            "  \(result.tokenCount) tokens: ok  token=\(result.prefillToken)"
                                + continuation + "\n",
                            stderr
                        )
                    } else {
                        prefillMismatchCount += 1
                        fputs(
                            "  \(result.tokenCount) tokens: mismatch  prefill=\(result.prefillToken) decode=\(result.decodeToken)"
                                + "\n    prefill continuation: \(result.prefillContinuation)"
                                + "\n    decode continuation:  \(result.decodeContinuation)\n",
                            stderr
                        )
                    }
                }
            }
        }

        if candidateRuntime.hasMetalPrefill && !prefillTokenCounts.isEmpty {
            fputs("\nPrefill benchmark:\n", stderr)
            for tokens in prefillTokenCounts {
                let result = try benchmarkPrefillWithSettle(
                    packagePath: candidatePath,
                    numTokens: tokens,
                    iterations: prefillIterations,
                    warmupIterations: 2,
                    minTps: minPrefillTps[tokens],
                    maxP95Ms: maxPrefillP95Ms[tokens],
                    config: settleConfig,
                    construction: candidateConstruction
                )
                fputs(
                    "  \(tokens) tokens: \(String(format: "%.1f", result.medianWallMs))ms/prefill"
                        + "  (\(String(format: "%.1f", result.tokensPerSecond)) tok/s)"
                        + "  p95=\(String(format: "%.1f", result.p95WallMs))ms"
                        + "  final=\(result.finalToken)\n",
                    stderr
                )
                gateFailures += prefillGateFailures(
                    result: result,
                    tokenCount: tokens,
                    minTps: minPrefillTps[tokens],
                    maxP95Ms: maxPrefillP95Ms[tokens]
                )
            }
        }

        if !gateFailures.isEmpty {
            fputs("\nVerify gates:\n", stderr)
            for failure in gateFailures {
                fputs("  \(failure)\n", stderr)
            }
        }

        if mismatchCount > 0 || prefillMismatchCount > 0 || !gateFailures.isEmpty {
            fputs(
                "\nVERIFY FAILED: \(mismatchCount) prompt mismatches, \(prefillMismatchCount) prefill mismatches, \(gateFailures.count) gate failures\n",
                stderr
            )
            exit(2)
        }

        fputs("\nVERIFY PASSED\n", stderr)
    } catch {
        fputs("Verify failed: \(error)\n", stderr)
        exit(1)
    }
}

private func resolveVerifyChatTemplate(
    construction: CAMTextRuntimeConstruction,
    cliOverride: String?
) throws -> String {
    try construction.resolveTemplate(cliOverride: cliOverride)
}
