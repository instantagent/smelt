import Darwin
import Foundation
import SmeltAgent

enum PiLauncherError: Error, CustomStringConvertible {
    case piMissing(String)
    case extensionMissing([String])
    case terminalRequired

    var description: String {
        switch self {
        case .piMissing(let executable):
            return "Pi is required (looked for \(executable)); install it with `brew install pi-coding-agent`"
        case .extensionMissing(let candidates):
            return "Smelt agent's Pi extension is missing; looked in: \(candidates.joined(separator: ", "))"
        case .terminalRequired:
            return "interactive Smelt agent use requires a terminal"
        }
    }
}

func launchPiInteractiveAgent(
    artifact: AgentArtifact,
    initialPrompt: String?,
    maxTokens: Int,
    temperature: Float?,
    seed: UInt64
) throws -> Never {
    try requireTerminal()
    let environment = ProcessInfo.processInfo.environment
    let piExecutable = environment["SMELT_AGENT_PI_EXECUTABLE"] ?? "pi"
    guard executableExists(piExecutable, environment: environment) else {
        throw PiLauncherError.piMissing(piExecutable)
    }
    let extensionPath = try resolvePiExtensionPath(environment: environment)

    setenv("SMELT_AGENT_PI_AGENT_PACKAGE", artifact.url.path, 1)
    setenv("SMELT_AGENT_PI_AGENT_ID", "current", 1)
    setenv("SMELT_AGENT_PI_AGENT_NAME", artifact.manifest.name, 1)
    setenv("SMELT_AGENT_PI_BIN", currentSmeltExecutablePath(), 1)
    setenv("SMELT_AGENT_PI_MAX_TOKENS", String(maxTokens), 1)
    setenv("SMELT_AGENT_PI_SEED", String(seed), 1)
    if let temperature {
        setenv("SMELT_AGENT_PI_TEMPERATURE", String(temperature), 1)
    } else {
        unsetenv("SMELT_AGENT_PI_TEMPERATURE")
    }
    if environment["SMELT_AGENT_PI_OPENAI_PORT"] == nil {
        setenv("SMELT_AGENT_PI_OPENAI_PORT", String(stableAgentPort(artifact.url.path)), 1)
    }

    var arguments = [
        piExecutable,
        "-e", extensionPath,
        "--no-extensions",
        "--no-context-files",
        "--no-skills",
        "--no-prompt-templates",
        "--model", "smelt-agent/current",
        "--system-prompt", "",
    ]
    if artifact.manifest.tools.isEmpty {
        arguments.append("--no-tools")
    } else {
        arguments += ["--tools", artifact.manifest.tools.joined(separator: ",")]
    }
    if let initialPrompt, !initialPrompt.isEmpty {
        arguments.append(initialPrompt)
    }
    try execPi(arguments)
}

func launchPiAuthoringAgent(name: String) throws -> Never {
    try requireTerminal()
    let normalized = name.hasSuffix(".agent")
        ? String(name.dropLast(".agent".count))
        : name
    try AgentManifest.validateName(normalized)

    let environment = ProcessInfo.processInfo.environment
    let piExecutable = environment["SMELT_AGENT_PI_EXECUTABLE"] ?? "pi"
    guard executableExists(piExecutable, environment: environment) else {
        throw PiLauncherError.piMissing(piExecutable)
    }
    let extensionDirectory = try resolvePiExtensionPath(environment: environment)
    let authorExtension = URL(fileURLWithPath: extensionDirectory, isDirectory: true)
        .appendingPathComponent("author.ts").path
    guard FileManager.default.fileExists(atPath: authorExtension) else {
        throw PiLauncherError.extensionMissing([authorExtension])
    }

    let workingDirectory = URL(
        fileURLWithPath: FileManager.default.currentDirectoryPath,
        isDirectory: true
    )
    let draft = workingDirectory.appendingPathComponent(normalized, isDirectory: true)
    let output = workingDirectory.appendingPathComponent("\(normalized).agent", isDirectory: true)
    try prepareAuthoringDraft(at: draft)

    setenv("SMELT_AGENT_AUTHOR_DRAFT", draft.path, 1)
    setenv("SMELT_AGENT_AUTHOR_NAME", normalized, 1)
    setenv("SMELT_AGENT_AUTHOR_OUTPUT", output.path, 1)
    setenv("SMELT_AGENT_PI_BIN", currentSmeltExecutablePath(), 1)

    let sessions = draft.appendingPathComponent(".pi-sessions", isDirectory: true)
    try FileManager.default.createDirectory(at: sessions, withIntermediateDirectories: true)
    let hasSession = ((try? FileManager.default.contentsOfDirectory(
        at: sessions,
        includingPropertiesForKeys: nil
    )) ?? []).contains { $0.pathExtension == "jsonl" }

    var arguments = [
        piExecutable,
        "-e", authorExtension,
        "--no-extensions",
        "--no-context-files",
        "--no-skills",
        "--no-prompt-templates",
        "--no-builtin-tools",
        "--session-dir", sessions.path,
        "--system-prompt", "",
    ]
    if hasSession { arguments.append("--continue") }
    arguments.append("Help me finish the \(normalized) agent. Start by asking what job it should do.")
    try execPi(arguments)
}

private func requireTerminal() throws {
    guard isatty(STDIN_FILENO) != 0, isatty(STDOUT_FILENO) != 0 else {
        throw PiLauncherError.terminalRequired
    }
}

private func prepareAuthoringDraft(at draft: URL) throws {
    let fileManager = FileManager.default
    try fileManager.createDirectory(at: draft, withIntermediateDirectories: true)
    func writeIfMissing(_ name: String, _ data: Data) throws {
        let destination = draft.appendingPathComponent(name)
        guard !fileManager.fileExists(atPath: destination.path) else { return }
        try data.write(to: destination, options: .atomic)
    }
    let source: [String: Any] = [
        "version": 1,
        "model": "",
        "systemFile": "instructions.md",
        "tools": [],
        "defaultMode": "once",
    ]
    try writeIfMissing(
        "Agentfile",
        try JSONSerialization.data(
            withJSONObject: source,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        ) + Data("\n".utf8)
    )
    try writeIfMissing("instructions.md", Data())
    try writeIfMissing("cases.jsonl", Data())
}

private func resolvePiExtensionPath(environment: [String: String]) throws -> String {
    if let configured = environment["SMELT_AGENT_PI_EXTENSION_PATH"], piExtensionExists(configured) {
        return URL(fileURLWithPath: configured, isDirectory: true).standardizedFileURL.path
    }
    let executable = URL(fileURLWithPath: currentSmeltExecutablePath()).standardizedFileURL
    var candidates = [
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
            .appendingPathComponent("integrations/pi-smelt-agent").path,
        executable.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("share/smelt/agent/pi").path,
    ]
    var directory = executable.deletingLastPathComponent()
    while directory.path != "/" {
        if directory.lastPathComponent == ".build" {
            candidates.append(
                directory.deletingLastPathComponent()
                    .appendingPathComponent("integrations/pi-smelt-agent").path
            )
        }
        let parent = directory.deletingLastPathComponent()
        guard parent.path != directory.path else { break }
        directory = parent
    }
    var unique: [String] = []
    for candidate in candidates {
        let path = URL(fileURLWithPath: candidate, isDirectory: true).standardizedFileURL.path
        if !unique.contains(path) { unique.append(path) }
    }
    if let match = unique.first(where: piExtensionExists) { return match }
    throw PiLauncherError.extensionMissing(unique)
}

private func piExtensionExists(_ path: String) -> Bool {
    FileManager.default.fileExists(
        atPath: URL(fileURLWithPath: path, isDirectory: true)
            .appendingPathComponent("index.ts").path
    )
}

private func executableExists(_ executable: String, environment: [String: String]) -> Bool {
    if executable.contains("/") {
        return FileManager.default.isExecutableFile(atPath: executable)
    }
    return (environment["PATH"] ?? "").split(separator: ":").contains { directory in
        FileManager.default.isExecutableFile(
            atPath: URL(fileURLWithPath: String(directory), isDirectory: true)
                .appendingPathComponent(executable).path
        )
    }
}

private func currentSmeltExecutablePath() -> String {
    if let bundled = Bundle.main.executableURL?.path,
       FileManager.default.isExecutableFile(atPath: bundled) {
        return URL(fileURLWithPath: bundled).standardizedFileURL.path
    }
    let raw = CommandLine.arguments[0]
    return raw.contains("/") ? URL(fileURLWithPath: raw).standardizedFileURL.path : raw
}

private func stableAgentPort(_ packagePath: String) -> Int {
    var hash: UInt32 = 2_166_136_261
    for byte in packagePath.utf8 {
        hash ^= UInt32(byte)
        hash &*= 16_777_619
    }
    return 20_000 + Int(hash % 20_000)
}

private func execPi(_ arguments: [String]) throws -> Never {
    let executable = arguments[0]
    var cArguments = arguments.map { strdup($0) }
    cArguments.append(nil)
    defer { cArguments.dropLast().forEach { free($0) } }
    _ = cArguments.withUnsafeMutableBufferPointer { buffer in
        execvp(executable, buffer.baseAddress)
    }
    throw PiLauncherError.piMissing("\(executable): \(String(cString: strerror(errno)))")
}
