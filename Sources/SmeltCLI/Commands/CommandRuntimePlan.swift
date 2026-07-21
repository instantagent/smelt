import Foundation
import SmeltRuntime
import SmeltSchema

struct RunTextRuntimePlan {
    let construction: CAMTextRuntimeConstruction
    let lingerCAMIdentity: LingerCAMIdentity?
}

func resolveRunTextRuntimePlanOrDispatchOtherOrExit(
    packagePath: String,
    promptStartIndex: Int,
    fullArgv: [String]
) -> RunTextRuntimePlan {
    let runFlags = scopedRunFlags(
        packagePath: packagePath,
        promptStartIndex: promptStartIndex,
        fullArgv: fullArgv
    )
    let lingerSeconds = runFlags.nonNegativeIntOrExit("--linger", verb: "run") ?? 0
    let requirements: [SmeltRuntimeRequestRequirement] = lingerSeconds > 0
        ? [SmeltRuntimeRequestRequirement(
            request: .runAudio,
            authoredCapabilities: ["run.stream"]
        )]
        : []
    let admission: SmeltRuntimeAdmission
    do {
        admission = try SmeltRuntimeAdmission.resolve(
            packagePath: packagePath,
            requests: [.runText, .runAudio],
            requirements: requirements
        )
    } catch SmeltRuntimeAdmissionError.noMatchingExport {
        fputs("smelt run: no CAM export satisfies run request\n", stderr)
        exit(1)
    } catch SmeltRuntimeAdmissionError.missingAuthoredCapabilities {
        fputs("smelt run: no CAM export satisfies linger request\n", stderr)
        exit(1)
    } catch {
        fputs("smelt run: \(error)\n", stderr)
        exit(1)
    }
    return resolveCAMRunTextRuntimePlanOrDispatchOtherOrExit(
        admission: admission,
        packagePath: packagePath,
        promptStartIndex: promptStartIndex,
        fullArgv: fullArgv,
        runFlags: runFlags,
        lingerSeconds: lingerSeconds
    )
}

private func resolveCAMRunTextRuntimePlanOrDispatchOtherOrExit(
    admission: SmeltRuntimeAdmission,
    packagePath: String,
    promptStartIndex: Int,
    fullArgv: [String],
    runFlags: ScopedFlagReader,
    lingerSeconds: Int
) -> RunTextRuntimePlan {
    switch admission.runtimeRoute {
    case .textToText:
        let textWantsLinger = admission.request == .runText
            && lingerSeconds > 0
            && !runFlags.has("--debug")
        let construction: CAMTextRuntimeConstruction
        do {
            construction = try admission.makeTextConstruction()
        } catch {
            fputs("smelt run: CAM text construction failed: \(error)\n", stderr)
            exit(1)
        }
        let lingerIdentity: LingerCAMIdentity?
        if textWantsLinger {
            lingerIdentity = makeLingerCAMIdentity(
                decision: admission.decision,
                capabilities: admission.capabilities
            )
        } else {
            lingerIdentity = nil
        }
        return RunTextRuntimePlan(
            construction: construction,
            lingerCAMIdentity: lingerIdentity
        )
    case .textToPCM(let outputRate):
        switch outputRate {
        case "24khz":
            let audioWantsLinger = admission.request == .runAudio
                && lingerSeconds > 0
            let lingerIdentity: LingerCAMIdentity?
            if audioWantsLinger {
                lingerIdentity = makeLingerCAMIdentity(
                    decision: admission.decision,
                    capabilities: admission.capabilities
                )
            } else {
                lingerIdentity = nil
            }
            let construction: CAMTextToPCMRuntimeConstruction
            do {
                construction = try CAMTextToPCMRuntimeConstruction(
                    runtimeAdmission: admission
                )
            } catch {
                fputs("smelt run: CAM text-to-PCM construction failed: \(error)\n", stderr)
                exit(1)
            }
            dispatchCAMTextToPCMRunHandlerOrExit(
                packagePath: packagePath,
                construction: construction,
                promptStartIndex: promptStartIndex,
                fullArgv: fullArgv,
                camIdentity: lingerIdentity
            )
        default:
            fputs("smelt run: unsupported CAM audio output rate '\(outputRate)'\n", stderr)
            exit(1)
        }
    }
}

private func scopedRunFlags(
    packagePath: String,
    promptStartIndex: Int,
    fullArgv: [String]
) -> ScopedFlagReader {
    let safeStart = min(max(0, promptStartIndex), fullArgv.count)
    let terminatorIndex = fullArgv[safeStart...].firstIndex(of: "--")
    let declared = scopedDeclaredRunFlags(packagePath: packagePath)
    return ScopedFlagReader(
        argv: fullArgv,
        startIndex: promptStartIndex,
        terminatorIndex: terminatorIndex,
        valueFlags: RunFlags.textValue
            .union(RunFlags.textToPCMValue)
            .union(declared.value),
        boolFlags: RunFlags.textBool
            .union(RunFlags.textToPCMBool)
            .union(declared.bool)
    )
}

private func scopedDeclaredRunFlags(packagePath: String) -> SmeltPackageInterface.RunBuiltinFlags {
    guard let declared = try? SmeltPackageInterface.load(packagePath: packagePath) else {
        return .init(value: [], bool: [])
    }
    return scopedDeclaredRunFlags(declared)
}

private func resolveCAMLingerIdentityOrExit(
    _ capabilities: SmeltCAMPackageCapabilities,
    request: SmeltCAMCapabilityRequest,
    requireStreamingExport: Bool = false,
    verb: String
) -> LingerCAMIdentity {
    do {
        let decision = try capabilities.resolve(request)
        if requireStreamingExport {
            requireCAMRunStreamCapabilityOrExit(decision, verb: verb)
        }
        return makeLingerCAMIdentity(
            decision: decision,
            capabilities: capabilities
        )
    } catch SmeltCAMPackageCapabilitiesError.noMatchingExport {
        fputs("smelt \(verb): no CAM export satisfies linger request\n", stderr)
        exit(1)
    } catch {
        fputs("smelt \(verb): \(error)\n", stderr)
        exit(1)
    }
}

private struct CAMBenchPlanRoute {
    let request: SmeltCAMCapabilityRequest
    let decision: SmeltCAMPackageCapabilities.Decision

    func runtimeRouteOrExit(
        capabilities: SmeltCAMPackageCapabilities
    ) -> CAMRuntimeRoute {
        resolveCAMRuntimeRouteOrExit(
            capabilities: capabilities,
            decision: decision,
            verb: "lab bench decode"
        )
    }
}

func requireBenchTextRuntimePlanOrExit(packagePath: String) -> CAMTextRuntimeConstruction {
    let capabilities = requireCAMPackageCapabilitiesOrExit(
        packagePath: packagePath,
        verb: "lab bench decode"
    )
    return requireCAMBenchTextRuntimePlanOrExit(
        capabilities: capabilities,
        packagePath: packagePath
    )
}

private func requireCAMBenchTextRuntimePlanOrExit(
    capabilities: SmeltCAMPackageCapabilities,
    packagePath: String
) -> CAMTextRuntimeConstruction {
    let route = resolveCAMBenchPlanRouteOrExit(capabilities)
    let runtimeRoute = route.runtimeRouteOrExit(capabilities: capabilities)
    switch runtimeRoute {
    case .textToText:
        return makeCAMTextRuntimeConstructionOrExit(
            packagePath: packagePath,
            capabilities: capabilities,
            decision: route.decision,
            verb: "lab bench decode"
        )
    case .textToPCM:
        fputs("smelt lab bench decode: module audio exports use a specialized audio benchmark harness\n", stderr)
        exit(1)
    }
}

private func resolveCAMBenchPlanRouteOrExit(
    _ capabilities: SmeltCAMPackageCapabilities
) -> CAMBenchPlanRoute {
    do {
        return CAMBenchPlanRoute(
            request: .benchDecode,
            decision: try capabilities.resolve(.benchDecode)
        )
    } catch SmeltCAMPackageCapabilitiesError.noMatchingExport {
    } catch {
        fputs("smelt lab bench decode: \(error)\n", stderr)
        exit(1)
    }

    do {
        return CAMBenchPlanRoute(
            request: .benchAudio,
            decision: try capabilities.resolve(.benchAudio)
        )
    } catch SmeltCAMPackageCapabilitiesError.noMatchingExport {
    } catch {
        fputs("smelt lab bench decode: \(error)\n", stderr)
        exit(1)
    }

    do {
        return CAMBenchPlanRoute(
            request: .runAudio,
            decision: try capabilities.resolve(.runAudio)
        )
    } catch SmeltCAMPackageCapabilitiesError.noMatchingExport {
        fputs("smelt lab bench decode: no CAM export satisfies decode benchmark request\n", stderr)
        exit(1)
    } catch {
        fputs("smelt lab bench decode: \(error)\n", stderr)
        exit(1)
    }
}

func requireCAMTextRuntimePlanOrExit(
    packagePath: String,
    request: SmeltCAMCapabilityRequest,
    verb: String,
    requireAuthoredInventory: Bool = false
) -> CAMTextRuntimeConstruction {
    let capabilities = requireCAMPackageCapabilitiesOrExit(
        packagePath: packagePath,
        verb: verb
    )
    let decision: SmeltCAMPackageCapabilities.Decision
    do {
        decision = try capabilities.resolve(request)
    } catch SmeltCAMPackageCapabilitiesError.noMatchingExport {
        fputs("smelt \(verb): no CAM export satisfies \(verb) text request\n", stderr)
        exit(1)
    } catch {
        fputs("smelt \(verb): \(error)\n", stderr)
        exit(1)
    }
    return makeCAMTextRuntimeConstructionOrExit(
        packagePath: packagePath,
        capabilities: capabilities,
        decision: decision,
        requiredCapabilityFiles: request.requiredPackageFiles,
        requireAuthoredInventory: requireAuthoredInventory,
        verb: verb
    )
}
