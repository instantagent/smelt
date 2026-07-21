import Foundation
import Testing
@testable import SmeltCLI

@Suite("Smelt agent Pi launcher")
struct AgentPiLauncherTests {
    @Test("resolves bundled resources before the installation share fallback")
    func resolvesPackagedPiResources() throws {
        let fileManager = FileManager.default
        let root = fileManager.temporaryDirectory.appendingPathComponent(
            "smelt-pi-resource-\(UUID().uuidString)",
            isDirectory: true
        )
        defer { try? fileManager.removeItem(at: root) }

        let bin = root.appendingPathComponent("bin", isDirectory: true)
        let executable = bin.appendingPathComponent("smelt")
        let share = root.appendingPathComponent("share/smelt/agent/pi", isDirectory: true)
        try fileManager.createDirectory(at: share, withIntermediateDirectories: true)
        try Data().write(to: share.appendingPathComponent("index.ts"))

        let isolatedWorkingDirectory = root.appendingPathComponent(
            "isolated-working-directory",
            isDirectory: true
        )
        try fileManager.createDirectory(
            at: isolatedWorkingDirectory,
            withIntermediateDirectories: true
        )
        #expect(try resolvePiExtensionPath(
            environment: [:],
            executablePath: executable.path,
            currentDirectory: isolatedWorkingDirectory.path
        ) == share.path)

        let bundled = bin
            .appendingPathComponent("Smelt_SmeltCLI.bundle", isDirectory: true)
            .appendingPathComponent("pi-smelt-agent", isDirectory: true)
        try fileManager.createDirectory(at: bundled, withIntermediateDirectories: true)
        try Data().write(to: bundled.appendingPathComponent("index.ts"))
        #expect(try resolvePiExtensionPath(
            environment: [:],
            executablePath: executable.path,
            currentDirectory: isolatedWorkingDirectory.path
        ) == bundled.path)
    }

    @Test("rejects an invalid explicit Pi resource path")
    func rejectsMissingConfiguredResource() throws {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("missing-pi-resource-\(UUID().uuidString)")
        #expect(throws: PiLauncherError.self) {
            try resolvePiExtensionPath(
                environment: ["SMELT_AGENT_PI_EXTENSION_PATH": missing.path],
                executablePath: "/unused/smelt"
            )
        }
    }
}
