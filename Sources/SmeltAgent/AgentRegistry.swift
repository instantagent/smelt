import Foundation
import SmeltRuntime

/// Filesystem registry for thin `.agent` overlays and portable Smelt package
/// payloads. Smelt owns package identity and materialization; the registry owns
/// only distribution layout and agent naming.
package enum AgentRegistry {
    package static func publish(
        _ artifact: AgentArtifact,
        name: String,
        registry: URL,
        fileManager: FileManager = .default
    ) throws -> URL {
        try AgentManifest.validateName(name)
        let model = try artifact.manifest.resolveModel()
        let overlays = registry.appendingPathComponent("agents", isDirectory: true)
        let models = registry.appendingPathComponent("models", isDirectory: true)
        try fileManager.createDirectory(at: overlays, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: models, withIntermediateDirectories: true)
        let modelDestination = models.appendingPathComponent(
            "\(model.identity).smeltpkg",
            isDirectory: true
        )
        if fileManager.fileExists(atPath: modelDestination.path) {
            let actual = try SmeltPackageIdentity.compute(packagePath: modelDestination.path)
            guard actual == model.identity else {
                throw AgentRegistryError.modelIdentityMismatch(
                    expected: model.identity,
                    actual: actual
                )
            }
        } else {
            try SmeltPackageStore.materialize(
                identity: model.identity,
                at: modelDestination
            )
        }
        let destination = overlays.appendingPathComponent("\(name).agent", isDirectory: true)
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.copyItem(at: artifact.url, to: destination)
        return destination
    }

    package static func install(
        name: String,
        registry: URL
    ) throws -> AgentArtifact {
        try AgentManifest.validateName(name)
        let source = registry
            .appendingPathComponent("agents", isDirectory: true)
            .appendingPathComponent("\(name).agent", isDirectory: true)
        let artifact = try AgentArtifact.load(at: source)
        let identity = artifact.manifest.model.smeltPackageIdentity
        if try SmeltPackageStore.locate(identity: identity) == nil {
            let model = registry
                .appendingPathComponent("models", isDirectory: true)
                .appendingPathComponent("\(identity).smeltpkg", isDirectory: true)
            guard FileManager.default.fileExists(atPath: model.path) else {
                throw AgentRegistryError.modelMissing(identity)
            }
            let installed = try SmeltPackageStore.install(packagePath: model.path)
            guard installed.identity == identity else {
                throw AgentRegistryError.modelIdentityMismatch(
                    expected: identity,
                    actual: installed.identity
                )
            }
        }
        return try AgentStore.install(artifact)
    }
}

package enum AgentRegistryError: Error, CustomStringConvertible, Equatable {
    case modelMissing(String)
    case modelIdentityMismatch(expected: String, actual: String)

    package var description: String {
        switch self {
        case .modelMissing(let identity):
            return "registry does not contain Smelt package \(identity)"
        case .modelIdentityMismatch(let expected, let actual):
            return "registry Smelt package identity mismatch: expected \(expected), got \(actual)"
        }
    }
}
