import Foundation
import SmeltRuntime
import Darwin

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
        let entries = try FileManager.default.contentsOfDirectory(atPath: url.path).sorted()
        guard entries == [AgentManifest.fileName] else {
            throw AgentArtifactError.unexpectedEntries(entries)
        }
        let manifestURL = url.appendingPathComponent(AgentManifest.fileName)
        var metadata = stat()
        guard lstat(manifestURL.path, &metadata) == 0,
              metadata.st_mode & S_IFMT == S_IFREG
        else {
            throw AgentArtifactError.invalidManifestEntry(manifestURL.path)
        }
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

    package func copy(
        to destination: URL,
        fileManager: FileManager = .default
    ) throws -> AgentArtifact {
        let source = try Self.load(at: url)
        let destination = destination.standardizedFileURL
        guard destination.pathExtension == "agent" else {
            throw AgentArtifactError.invalidPath(destination.path)
        }
        let parent = destination.deletingLastPathComponent()
        try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
        let staging = parent.appendingPathComponent(
            ".\(destination.lastPathComponent).tmp-\(UUID().uuidString).agent",
            isDirectory: true
        )
        do {
            try fileManager.copyItem(at: source.url, to: staging)
            _ = try Self.load(at: staging)
            if fileManager.fileExists(atPath: destination.path) {
                _ = try fileManager.replaceItemAt(destination, withItemAt: staging)
            } else {
                try fileManager.moveItem(at: staging, to: destination)
            }
            return try Self.load(at: destination)
        } catch {
            try? fileManager.removeItem(at: staging)
            throw error
        }
    }
}

package enum AgentArtifactError: Error, CustomStringConvertible, Equatable {
    case invalidPath(String)
    case invalidManifestEntry(String)
    case unexpectedEntries([String])

    package var description: String {
        switch self {
        case .invalidPath(let path):
            return "agent artifact path must end in .agent: \(path)"
        case .invalidManifestEntry(let path):
            return "agent manifest must be a regular file inside the artifact: \(path)"
        case .unexpectedEntries(let entries):
            return "agent artifact must contain only \(AgentManifest.fileName); found: "
                + entries.joined(separator: ", ")
        }
    }
}
