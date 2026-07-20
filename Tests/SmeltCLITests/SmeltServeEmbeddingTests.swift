import Foundation
import Testing
import SmeltServe

@Test("embedded server validates packages without a smelt executable")
func embeddedServerOwnsAdmission() {
    let missingPackage = FileManager.default.temporaryDirectory
        .appendingPathComponent("missing-\(UUID().uuidString).smeltpkg", isDirectory: true)

    #expect(throws: SmeltServeError.self) {
        try SmeltTextServer.run(SmeltTextServerConfiguration(
            packagePath: missingPackage.path,
            endpoint: .http(host: "127.0.0.1", port: 8080)
        ))
    }
}

@Test("serve library leaves process lifecycle to executable adapters")
func serveLibraryDoesNotOwnProcessSignals() throws {
    let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let serveRoot = repoRoot
        .appendingPathComponent("Sources", isDirectory: true)
        .appendingPathComponent("SmeltServe", isDirectory: true)
    let transportSources = try ["HTTPTransport.swift", "StdioTransport.swift"]
        .map { name in
            try String(
                contentsOf: serveRoot.appendingPathComponent(name),
                encoding: .utf8
            )
        }
        .joined(separator: "\n")
    #expect(!transportSources.contains("installFatalSignalHandlers"))
    #expect(!transportSources.contains("exit("))

    let serveCommand = try String(
        contentsOf: repoRoot
            .appendingPathComponent("Sources/SmeltCLI/Commands/ServeCommand.swift"),
        encoding: .utf8
    )
    #expect(serveCommand.contains("installFatalSignalHandlers"))
    #expect(serveCommand.contains("SmeltServeAdmission.resolve"))
    #expect(!serveCommand.contains("resolveServeRuntimePlanOrExit"))
}

@Test("CLI run, embedded generation, and serving share runtime admission")
func textEntryPointsShareRuntimeAdmission() throws {
    let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let sources = try [
        "Sources/SmeltCLI/Commands/CommandRuntimePlan.swift",
        "Sources/SmeltServe/SmeltTextGenerator.swift",
        "Sources/SmeltServe/SmeltServeAdmission.swift",
    ].map { relativePath in
        try String(
            contentsOf: repoRoot.appendingPathComponent(relativePath),
            encoding: .utf8
        )
    }

    for source in sources {
        #expect(source.contains("SmeltRuntimeAdmission.resolve"))
    }
    let generator = sources[1]
    #expect(!generator.contains("SmeltManifest.decode"))
    #expect(!generator.contains("SmeltModel("))
    #expect(generator.contains("construction.renderPrompt"))
    #expect(generator.contains("construction.effectiveMaxTokens"))
}
