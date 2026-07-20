import Foundation
import SmeltSchema

package enum SmeltRuntimeAdmissionError: Error, CustomStringConvertible, Sendable {
    case emptyRequestSet
    case missingPackageDescriptor(String)
    case unsupportedFeatures(stage: String, diagnostic: String)
    case noMatchingExport
    case noRuntimeRoute(exportID: String, flowID: String)
    case missingAuthoredCapabilities([String])
    case textRouteRequired

    package var description: String {
        switch self {
        case .emptyRequestSet:
            return "runtime admission requires at least one capability request"
        case .missingPackageDescriptor(let fileName):
            return "expected module descriptor but package has no \(fileName)"
        case .unsupportedFeatures(let stage, let diagnostic):
            return "module feature admission failed at \(stage): \(diagnostic)"
        case .noMatchingExport:
            return "no CAM export satisfies runtime request"
        case .noRuntimeRoute(let exportID, let flowID):
            return "no CAM runtime route for export '\(exportID)' flow '\(flowID)'"
        case .missingAuthoredCapabilities(let capabilities):
            return "runtime export is missing authored capabilities: "
                + capabilities.joined(separator: ",")
        case .textRouteRequired:
            return "CAM runtime route is not text-to-text"
        }
    }
}

package struct SmeltRuntimeRequestRequirement: Sendable {
    package let request: SmeltCAMCapabilityRequest
    package let authoredCapabilities: Set<String>

    package init(
        request: SmeltCAMCapabilityRequest,
        authoredCapabilities: Set<String>
    ) {
        self.request = request
        self.authoredCapabilities = authoredCapabilities
    }
}

/// One throwing authority for package loading, feature admission, request
/// selection, runtime routing, and package inventory validation. Executable
/// and embedded adapters consume this evidence instead of resolving packages
/// independently.
package struct SmeltRuntimeAdmission: Sendable {
    package let packagePath: String
    package let request: SmeltCAMCapabilityRequest
    package let runtimeRoute: CAMRuntimeRoute
    package let capabilities: SmeltCAMPackageCapabilities
    package let decision: SmeltCAMPackageCapabilities.Decision

    package static func resolve(
        packagePath: String,
        requests: [SmeltCAMCapabilityRequest],
        requirements: [SmeltRuntimeRequestRequirement] = []
    ) throws -> SmeltRuntimeAdmission {
        guard !requests.isEmpty else {
            throw SmeltRuntimeAdmissionError.emptyRequestSet
        }
        let packageURL = URL(
            fileURLWithPath: packagePath,
            isDirectory: true
        ).standardizedFileURL
        guard let capabilities = try SmeltCAMPackageCapabilities.loadIfPresent(
            packageURL: packageURL
        ) else {
            throw SmeltRuntimeAdmissionError.missingPackageDescriptor(
                SmeltCAMPackageDescriptor.packageFileName
            )
        }

        let featureAdmission = capabilities.featureAdmission
        guard !featureAdmission.hasUnsupportedFeatures else {
            throw SmeltRuntimeAdmissionError.unsupportedFeatures(
                stage: featureAdmission.stage,
                diagnostic: featureAdmission.unsupportedDiagnostic
            )
        }

        var selected: (
            request: SmeltCAMCapabilityRequest,
            decision: SmeltCAMPackageCapabilities.Decision
        )? = nil
        for request in requests {
            do {
                selected = (request, try capabilities.resolve(request))
                break
            } catch SmeltCAMPackageCapabilitiesError.noMatchingExport {
                continue
            }
        }
        guard let selected else {
            throw SmeltRuntimeAdmissionError.noMatchingExport
        }
        guard let runtimeRoute = CAMRuntimeRouteResolver.resolve(
            decision: selected.decision,
            capabilities: capabilities
        ) else {
            throw SmeltRuntimeAdmissionError.noRuntimeRoute(
                exportID: selected.decision.exportID,
                flowID: selected.decision.flowID
            )
        }
        let requiredCapabilities = requirements
            .filter { $0.request == selected.request }
            .reduce(into: Set<String>()) { result, requirement in
                result.formUnion(requirement.authoredCapabilities)
            }
        let missingCapabilities = requiredCapabilities
            .subtracting(selected.decision.authoredCapabilities)
            .sorted()
        guard missingCapabilities.isEmpty else {
            throw SmeltRuntimeAdmissionError.missingAuthoredCapabilities(
                missingCapabilities
            )
        }

        switch runtimeRoute {
        case .textToText:
            try SmeltCAMPackageInventoryValidator.validate(
                capabilities,
                packageURL: packageURL
            )
        case .textToPCM:
            try SmeltCAMPackageInventoryValidator.validate(
                capabilities,
                packageURL: packageURL,
                requireAuthoredInventory: true
            )
        }

        return SmeltRuntimeAdmission(
            packagePath: packageURL.path,
            request: selected.request,
            runtimeRoute: runtimeRoute,
            capabilities: capabilities,
            decision: selected.decision
        )
    }

    package func makeTextConstruction() throws -> CAMTextRuntimeConstruction {
        guard runtimeRoute == .textToText else {
            throw SmeltRuntimeAdmissionError.textRouteRequired
        }
        return try CAMTextRuntimeConstruction(
            packagePath: packagePath,
            capabilities: capabilities,
            decision: decision
        )
    }
}
