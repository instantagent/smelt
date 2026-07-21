import Foundation
import SmeltRuntime
import SmeltSchema

func runRunCommand() {
    let usage = [
        "Usage: smelt run <model.smeltpkg> [input] [package-declared options]\n",
        "   Text packages: [prompt] [--max-tokens N] [--context-limit N] [--template NAME] [--temp T] [--seed N] [--system TEXT | --system-file PATH] [--prompt-file PATH] [--linger SECONDS] [--bind NAME=V1,V2,...]\n",
        "   or: smelt run --package <model.smeltpkg> --prompt \"text\"\n",
        "   --help shows this text plus the package's declared flags; `--` ends flags\n",
        "   Text-to-PCM packages: smelt run <voice.smeltpkg> \"text\" > output.wav\n"
    ]
    // `--` ends flag parsing for EVERYTHING — including package resolution
    // (`smelt run pkg -- --package other` is prompt text, not a redirect).
    // The scanner gets the full argv (positionals after `--` are prompt
    // material); every `--flag VALUE` lookup reads the truncated view.
    let fullArgv = args
    if let dash = args.firstIndex(of: "--") {
        args = Array(args[..<dash])
    }

    let (packagePath, promptStartIndex) = resolvePackagePath(usage: usage)

    if dispatchDeclaredPackageRunIfPresent(
        packagePath: packagePath,
        promptStartIndex: promptStartIndex,
        fullArgv: fullArgv
    ) {
        return
    }

    // Preserve package capability admission as the first package-content
    // authority. Text returns a plan; other modalities dispatch through their
    // own run handler, which applies the same mode resolver before execution.
    let runPlan = resolveRunTextRuntimePlanOrDispatchOtherOrExit(
        packagePath: packagePath,
        promptStartIndex: promptStartIndex,
        fullArgv: fullArgv
    )
    let lingerCAMIdentity = runPlan.lingerCAMIdentity
    let construction = runPlan.construction

    // Strict argv: every --flag must be a built-in or declared by the
    // package's args.json; a typo'd flag is an error, not prompt text.
    let runInterface = loadRunPackageInterface(
        packagePath: packagePath
    )
    let scanned: SmeltPackageInterface.Scan
    do {
        scanned = try SmeltPackageInterface.scan(
            argv: fullArgv,
            startIndex: promptStartIndex,
            builtinsWithValues: runInterface.builtins.value,
            builtinBools: runInterface.builtins.bool,
            declared: runInterface.declared
        )
    } catch {
        fputs("smelt run: \(error)\n", stderr)
        for line in usage { fputs(line, stderr) }
        exit(1)
    }
    if scanned.helpRequested {
        for line in usage { fputs(line, stderr) }
        for line in runInterface.declared?.helpLines() ?? [] { fputs(line + "\n", stderr) }
        exit(0)
    }
    let declaredFlags = scopedDeclaredRunFlags(runInterface.declared)
    let scopedFlags = ScopedFlagReader(
        argv: fullArgv,
        startIndex: promptStartIndex,
        terminatorIndex: scanned.terminatorIndex,
        valueFlags: runInterface.builtins.value.union(declaredFlags.value),
        boolFlags: runInterface.builtins.bool.union(declaredFlags.bool)
    )

    let promptFlag = scopedFlags.value("--prompt")
    let positionalPrompt = scanned.positionals.joined(separator: " ")
    let promptFile = scopedFlags.value("--prompt-file")
    let explicitPrompt = promptFlag.isEmpty ? positionalPrompt : promptFlag
    let promptFromFile: String?
    do {
        promptFromFile = promptFile.isEmpty
            ? nil
            : try readTextFileArg(promptFile, label: "prompt")
    } catch {
        fputs("smelt run: \(error)\n", stderr)
        exit(1)
    }
    let pipedPrompt = explicitPrompt.isEmpty && promptFromFile == nil
        ? readPromptFromStdin()
        : nil
    let inputText = promptFromFile ?? (explicitPrompt.isEmpty ? pipedPrompt ?? "" : explicitPrompt)

    let resolvedDeclared: [String: SmeltPackageArgumentValue]
    let grammarBindings: [String: [String]]
    let contextLimit = scopedFlags.positiveIntOrExit("--context-limit", verb: "run")
    var systemPrompt = scopedFlags.value("--system")
    do {
        resolvedDeclared = try runInterface.declared?.resolve(
            declaredRaw: scanned.declaredRaw
        ) ?? [:]
        grammarBindings = try mergeDeclaredBindings(
            interface: runInterface.declared,
            scanned: scanned,
            resolved: resolvedDeclared,
            explicit: parseGrammarBindings(rawValues: scopedFlags.values("--bind"))
        )
        let systemFile = scopedFlags.value("--system-file")
        if !systemFile.isEmpty {
            systemPrompt = try String(
                contentsOf: URL(fileURLWithPath: systemFile), encoding: .utf8
            )
        }
    } catch {
        fputs("smelt run: \(error)\n", stderr)
        exit(1)
    }

    // With a compiled grammar, generation stops at the schema's accepting
    // state, so the cap is a backstop — but 32 truncates real JSON outputs
    // mid-object. Schema maxLength bounds keep the backstop honest.
    let hasCompiledGrammar: Bool
    do {
        hasCompiledGrammar = try SmeltCompiledGrammar.load(packagePath: packagePath) != nil
    } catch {
        fputs("smelt run: \(error)\n", stderr)
        exit(1)
    }
    let defaultMaxTokens = hasCompiledGrammar
        ? 512 : construction.executionPlan.stopPolicy.maxSteps
    let maxTokens = construction.effectiveMaxTokens(
        scopedFlags.positiveIntOrExit("--max-tokens", verb: "run") ?? defaultMaxTokens
    )
    let template = scopedFlags.value("--template")
    do {
        // Declared args: typed values feed the package's prompt template
        // ({flag}/{input} placeholders) and its grammar bind slots.
        let effectivePrompt = try runInterface.declared?.fillPrompt(
            resolved: resolvedDeclared, input: inputText
        ) ?? inputText
        let (selectionMode, selectionDescription) = try resolveSelectionMode(
            tempArg: scopedFlags.value("--temp", default: "0"),
            seedArg: scopedFlags.value("--seed", default: "")
        )
        if effectivePrompt.isEmpty {
            fputs("smelt run: one-shot mode requires input\n", stderr)
            exit(1)
        } else {
            // --linger N: forward to a warm worker when one is listening;
            // otherwise run inline and leave a worker behind for N idle
            // seconds so repeat invocations skip package load.
            let lingerSeconds = scopedFlags.nonNegativeIntOrExit("--linger", verb: "run") ?? 0
            let useLinger = lingerSeconds > 0 && !scopedFlags.has("--debug")
            if useLinger {
                let socketPath = lingerSocketPath(
                    packagePath: packagePath,
                    contextLimit: contextLimit,
                    grammarBindings: grammarBindings,
                    camIdentity: lingerCAMIdentity
                )
                let request = LingerRequest(
                    prompt: effectivePrompt,
                    systemPrompt: systemPrompt,
                    template: template,
                    maxTokens: maxTokens,
                    tempArg: scopedFlags.value("--temp", default: "0"),
                    seedArg: scopedFlags.value("--seed", default: "")
                )
                if let result = tryLingerForward(
                    socketPath: socketPath, request: request
                ) {
                    printPromptRunResult(result, packagePath: packagePath)
                    fputs("Linger: served by warm worker\n", stderr)
                    return
                }
            }

            try runPrompt(
                packagePath: packagePath,
                prompt: effectivePrompt,
                maxTokens: maxTokens,
                template: template,
                selectionMode: selectionMode,
                selectionDescription: selectionDescription,
                contextLimit: contextLimit,
                systemPrompt: systemPrompt,
                grammarBindings: grammarBindings,
                construction: construction
            )

            if useLinger {
                spawnLingerWorker(
                    packagePath: packagePath,
                    socketPath: lingerSocketPath(
                        packagePath: packagePath,
                        contextLimit: contextLimit,
                        grammarBindings: grammarBindings,
                        camIdentity: lingerCAMIdentity
                    ),
                    idleSeconds: lingerSeconds,
                    contextLimit: contextLimit,
                    grammarBindings: grammarBindings,
                    camIdentity: lingerCAMIdentity
                )
            }

            // First successful run: deduplicate large artifacts into the
            // shared store in the background (SMELT_CAS=0 disables).
            spawnCasAdoptIfUseful(packagePath: packagePath)
        }
    } catch {
        fputs("Run failed: \(error)\n", stderr)
        exit(1)
    }
}
