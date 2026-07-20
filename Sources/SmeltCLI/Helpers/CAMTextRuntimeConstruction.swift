import Foundation
import SmeltRuntime
import SmeltSchema

func makeCAMTextRuntimeConstructionOrExit(
    packagePath: String,
    capabilities: SmeltCAMPackageCapabilities,
    decision: SmeltCAMPackageCapabilities.Decision,
    requiredCapabilityFiles: Set<String> = [],
    requireAuthoredInventory: Bool = false,
    verb: String
) -> CAMTextRuntimeConstruction {
    let runtimeRoute = resolveCAMRuntimeRouteOrExit(
        capabilities: capabilities,
        decision: decision,
        verb: verb
    )
    guard runtimeRoute == .textToText else {
        fputs("smelt \(verb): CAM route is not text-to-text\n", stderr)
        exit(1)
    }
    let optionalCapabilityFiles: Set<String> = [
        "dispatches.bin",
        "prefill_dispatches.bin",
        "prefill_verify_argmax_dispatches.bin",
    ]
    let allowMissingInventoryFiles = requiredCapabilityFiles.isEmpty
        ? []
        : optionalCapabilityFiles.subtracting(requiredCapabilityFiles)
    requireCAMPackageInventoryOrExit(
        capabilities,
        packagePath: packagePath,
        verb: verb,
        requireAuthoredInventory: requireAuthoredInventory,
        allowMissingFiles: allowMissingInventoryFiles
    )
    do {
        return try CAMTextRuntimeConstruction(
            packagePath: packagePath,
            capabilities: capabilities,
            decision: decision
        )
    } catch {
        fputs("smelt \(verb): CAM text construction failed: \(error)\n", stderr)
        exit(1)
    }
}
