import Foundation

package enum AgentStore {
    package static func rootURL() -> URL {
        if let override = ProcessInfo.processInfo.environment["SMELT_AGENT_STORE_DIR"],
           !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        return FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        )[0]
        .appendingPathComponent("smelt", isDirectory: true)
        .appendingPathComponent("agents", isDirectory: true)
    }

    package static func resolve(_ nameOrPath: String) throws -> AgentArtifact {
        let direct = URL(fileURLWithPath: nameOrPath)
        if FileManager.default.fileExists(atPath: direct.path) {
            return try AgentArtifact.load(at: direct)
        }
        let name = nameOrPath.hasSuffix(".agent")
            ? nameOrPath : "\(nameOrPath).agent"
        return try AgentArtifact.load(at: rootURL().appendingPathComponent(name))
    }

    @discardableResult
    package static func install(
        _ artifact: AgentArtifact,
        fileManager: FileManager = .default
    ) throws -> AgentArtifact {
        let root = rootURL()
        try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
        let destination = root.appendingPathComponent(
            "\(artifact.manifest.name).agent",
            isDirectory: true
        )
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.copyItem(at: artifact.url, to: destination)
        return try AgentArtifact.load(at: destination)
    }
}
