import Foundation
import SmeltRuntime

package struct AgentArtifact: Sendable {
    package let url: URL
    package let manifest: AgentManifest

    package init(url: URL, manifest: AgentManifest) {
        self.url = url
        self.manifest = manifest
    }

    package static func load(at url: URL) throws -> AgentArtifact {
        let url = url.standardizedFileURL
        guard url.pathExtension == "agent" else {
            throw AgentArtifactError.invalidPath(url.path)
        }
        let manifestURL = url.appendingPathComponent(AgentManifest.fileName)
        let manifest = try JSONDecoder().decode(
            AgentManifest.self,
            from: Data(contentsOf: manifestURL)
        )
        try manifest.validate()
        return AgentArtifact(url: url, manifest: manifest)
    }

    @discardableResult
    package static func create(
        at output: URL,
        name: String,
        modelPackagePath: String,
        instructions: String? = nil,
        tools: [String] = [],
        defaultMode: AgentManifest.DefaultMode = .once,
        fileManager: FileManager = .default
    ) throws -> AgentArtifact {
        let output = output.standardizedFileURL
        guard output.pathExtension == "agent" else {
            throw AgentArtifactError.invalidPath(output.path)
        }
        let stored = try SmeltPackageStore.install(packagePath: modelPackagePath)
        let manifest = AgentManifest(
            name: name,
            model: .init(smeltPackageIdentity: stored.identity),
            instructions: instructions?.isEmpty == false ? instructions : nil,
            tools: tools,
            defaultMode: defaultMode
        )
        try manifest.validate()

        let parent = output.deletingLastPathComponent()
        try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
        let temporary = parent.appendingPathComponent(
            ".\(output.lastPathComponent).tmp-\(UUID().uuidString)",
            isDirectory: true
        )
        try fileManager.createDirectory(at: temporary, withIntermediateDirectories: false)
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(manifest).write(
                to: temporary.appendingPathComponent(AgentManifest.fileName),
                options: .atomic
            )
            if fileManager.fileExists(atPath: output.path) {
                _ = try fileManager.replaceItemAt(output, withItemAt: temporary)
            } else {
                try fileManager.moveItem(at: temporary, to: output)
            }
        } catch {
            try? fileManager.removeItem(at: temporary)
            throw error
        }
        return AgentArtifact(url: output, manifest: manifest)
    }
}

package enum AgentArtifactError: Error, CustomStringConvertible, Equatable {
    case invalidPath(String)

    package var description: String {
        switch self {
        case .invalidPath(let path):
            return "agent artifact path must end in .agent: \(path)"
        }
    }
}
