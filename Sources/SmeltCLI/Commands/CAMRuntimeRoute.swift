import Foundation
import SmeltRuntime
import SmeltSchema

func resolveCAMRuntimeRouteOrExit(
    capabilities: SmeltCAMPackageCapabilities,
    decision: SmeltCAMPackageCapabilities.Decision,
    verb: String
) -> CAMRuntimeRoute {
    if let route = CAMRuntimeRouteResolver.resolve(
        decision: decision,
        capabilities: capabilities
    ) {
        return route
    }
    fputs(
        "smelt \(verb): no CAM runtime route for \(describeCAMDecision(decision))\n",
        stderr
    )
    exit(1)
}
