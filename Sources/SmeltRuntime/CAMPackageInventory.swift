import Foundation
import SmeltSchema

package struct SmeltCAMPackageInventoryError: Error, CustomStringConvertible, Sendable {
    package let description: String

    package init(_ description: String) {
        self.description = description
    }
}

/// One filesystem-backed authority for validating the package inventory
/// projected by CAM gates. CLI adapters may translate the thrown error into
/// process diagnostics, while embedded consumers keep normal error semantics.
package enum SmeltCAMPackageInventoryValidator {
    package static func validate(
        _ capabilities: SmeltCAMPackageCapabilities,
        packageURL: URL,
        requireAuthoredInventory: Bool = false,
        allowMissingFiles: Set<String> = []
    ) throws {
        let inventories = capabilities.gateContracts
            .flatMap(\.requirements)
            .compactMap { requirement -> [String]? in
                guard requirement.subject == "package-files",
                      requirement.relation == "include"
                else { return nil }
                return requirement.value
                    .split(separator: ",")
                    .map(String.init)
                    .sorted()
            }
        let uniqueInventories = Set(inventories)
        guard uniqueInventories.count <= 1 else {
            throw SmeltCAMPackageInventoryError(
                "module package inventory requirements disagree"
            )
        }
        guard let requiredFiles = uniqueInventories.first else {
            if requireAuthoredInventory {
                throw SmeltCAMPackageInventoryError(
                    "module package inventory requirement missing"
                )
            }
            return
        }
        let missing = requiredFiles.filter { relativePath in
            if allowMissingFiles.contains(relativePath) { return false }
            return !FileManager.default.fileExists(
                atPath: packageURL.appendingPathComponent(relativePath).path
            )
        }
        guard missing.isEmpty else {
            throw SmeltCAMPackageInventoryError(
                "module package inventory missing files: "
                    + missing.joined(separator: ",")
            )
        }
    }
}
