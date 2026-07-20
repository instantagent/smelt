import Foundation
import SmeltSchema

func requireCAMRuntimeAssemblyFeatureContractOrExit(
    capabilities: SmeltCAMPackageCapabilities,
    decision: SmeltCAMPackageCapabilities.Decision,
    verb: String
) {
    let contract: SmeltCAMPackageCapabilities.RuntimeAssemblyFeatureContract
    do {
        contract = try capabilities.runtimeAssemblyFeatureContract(for: decision)
    } catch {
        fputs("smelt \(verb): CAM runtime feature contract failed: \(error)\n", stderr)
        exit(1)
    }
    guard contract.schema == SmeltCAMPackageCapabilities.RuntimeAssemblyFeatureContract.currentSchema,
          !contract.featureSet.isEmpty
    else {
        fputs(
            "smelt \(verb): CAM runtime feature contract is empty or unsupported for "
                + "\(describeCAMDecision(decision))\n",
            stderr
        )
        exit(1)
    }
}

func makeCAMTraceRouteWitness(
    decision: SmeltCAMPackageCapabilities.Decision,
    capabilities: SmeltCAMPackageCapabilities
) -> String {
    return decision.traceRouteWitnessV6(
        camSemanticSHA256: capabilities.camSemanticSHA256,
        exportABISHA256: capabilities.exportABISHA256
    )
}
