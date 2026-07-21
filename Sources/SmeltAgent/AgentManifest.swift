import Foundation
import SmeltRuntime

package struct AgentManifest: Codable, Sendable, Equatable {
    package static let currentVersion = 1
    package static let fileName = "agent.json"

    package struct ModelReference: Codable, Sendable, Equatable {
        package let smeltPackageIdentity: String

        package init(smeltPackageIdentity: String) {
            self.smeltPackageIdentity = smeltPackageIdentity
        }

        enum CodingKeys: String, CodingKey {
            case smeltPackageIdentity = "smelt_package_identity"
        }
    }

    package enum DefaultMode: String, Codable, Sendable {
        case once
        case interactive
    }

    package let version: Int
    package let name: String
    package let model: ModelReference
    package let instructions: String?
    package let tools: [String]
    package let defaultMode: DefaultMode

    package init(
        version: Int = Self.currentVersion,
        name: String,
        model: ModelReference,
        instructions: String? = nil,
        tools: [String] = [],
        defaultMode: DefaultMode = .once
    ) {
        self.version = version
        self.name = name
        self.model = model
        self.instructions = instructions
        self.tools = tools
        self.defaultMode = defaultMode
    }

    package func validate() throws {
        guard version == Self.currentVersion else {
            throw AgentManifestError.unsupportedVersion(version)
        }
        try Self.validateName(name)
        guard model.smeltPackageIdentity.count == 64,
              model.smeltPackageIdentity.utf8.allSatisfy({
                  ($0 >= 48 && $0 <= 57) || ($0 >= 97 && $0 <= 102)
              })
        else {
            throw AgentManifestError.invalidModelIdentity(
                model.smeltPackageIdentity
            )
        }
        guard Set(tools).count == tools.count,
              tools.allSatisfy(Self.isValidName)
        else {
            throw AgentManifestError.invalidTools(tools)
        }
    }

    package static func validateName(_ value: String) throws {
        guard isValidName(value) else {
            throw AgentManifestError.invalidName(value)
        }
    }

    package func resolveModel() throws -> SmeltStoredPackage {
        try validate()
        guard let package = try SmeltPackageStore.locate(
            identity: model.smeltPackageIdentity
        ) else {
            throw AgentManifestError.modelNotInstalled(model.smeltPackageIdentity)
        }
        return package
    }

    private static func isValidName(_ value: String) -> Bool {
        guard let first = value.first, first.isLetter || first.isNumber else {
            return false
        }
        return value.allSatisfy {
            $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" || $0 == "."
        }
    }
}

package enum AgentManifestError: Error, CustomStringConvertible, Equatable {
    case unsupportedVersion(Int)
    case invalidName(String)
    case invalidModelIdentity(String)
    case invalidTools([String])
    case modelNotInstalled(String)

    package var description: String {
        switch self {
        case .unsupportedVersion(let version):
            return "unsupported .agent version \(version)"
        case .invalidName(let name):
            return "invalid agent name '\(name)'"
        case .invalidModelIdentity(let identity):
            return "invalid Smelt package identity '\(identity)'"
        case .invalidTools(let tools):
            return "invalid or duplicate tool names: \(tools.joined(separator: ", "))"
        case .modelNotInstalled(let identity):
            return "Smelt package \(identity) is not installed"
        }
    }
}
