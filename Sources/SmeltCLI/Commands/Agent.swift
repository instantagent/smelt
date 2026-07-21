import Foundation
import SmeltAgent
import SmeltServe
import Darwin

private let agentUsage = """
Smelt agent

  smelt agent create <name> --model <model.smeltpkg>
                     [--system TEXT|--system-file FILE]
                     [--tool NAME ...] [--interactive] [--output NAME.agent]
  smelt agent create <name> --from Agentfile [--output NAME.agent]
  smelt agent run <name|path.agent> [prompt]
                  [--max-tokens N] [--temperature T] [--seed N]
  smelt agent run <name|path.agent> --interactive
  smelt agent install <name> --registry DIR
  smelt agent publish <name|path.agent> [--name NAME] --registry DIR
"""

private struct AgentCLIError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

private func agentFail(_ message: String) -> Never {
    fputs("smelt agent: \(message)\n", stderr)
    exit(1)
}

private func value(_ flag: String, in arguments: [String]) -> String? {
    guard let index = arguments.firstIndex(of: flag), index + 1 < arguments.count else {
        return nil
    }
    return arguments[index + 1]
}

private func values(_ flag: String, in arguments: [String]) -> [String] {
    arguments.indices.compactMap { index in
        arguments[index] == flag && index + 1 < arguments.count
            ? arguments[index + 1] : nil
    }
}

private func positionalTail(
    _ arguments: [String],
    valueFlags: Set<String>
) -> [String] {
    var result: [String] = []
    var skip = false
    for argument in arguments {
        if skip { skip = false; continue }
        if valueFlags.contains(argument) { skip = true; continue }
        if argument.hasPrefix("-") { continue }
        result.append(argument)
    }
    return result
}

private struct Agentfile: Decodable {
    enum Tools: Decodable {
        case names([String])
        case file(String)

        init(from decoder: Decoder) throws {
            let values = try decoder.singleValueContainer()
            if let names = try? values.decode([String].self) {
                self = .names(names)
            } else {
                self = .file(try values.decode(String.self))
            }
        }
    }

    let version: Int
    let model: String
    let instructions: String?
    let system: String?
    let systemFile: String?
    let tools: Tools?
    let args: String?
    let defaultMode: AgentManifest.DefaultMode?

    enum CodingKeys: String, CodingKey {
        case version, model, instructions, system, systemFile, tools, args, defaultMode
    }
}

private struct AgentfileTools: Decodable {
    let version: Int
    let tools: [String]
}

private struct AgentfileInterface: Decodable {
    struct Run: Decodable { let defaultMode: String? }
    let version: Int
    let run: Run?
}

private func create(_ arguments: [String]) throws {
    guard let name = arguments.first, !name.hasPrefix("-") else {
        throw AgentCLIError("create requires a name")
    }
    let output = URL(fileURLWithPath: value("--output", in: arguments)
        ?? "\(name).agent")

    let model: String
    let instructions: String?
    let tools: [String]
    let mode: AgentManifest.DefaultMode
    if let sourcePath = value("--from", in: arguments) {
        let sourceURL = URL(fileURLWithPath: sourcePath).standardizedFileURL
        let source = try JSONDecoder().decode(
            Agentfile.self,
            from: Data(contentsOf: sourceURL)
        )
        guard source.version == 1 else {
            throw AgentCLIError("unsupported Agentfile version \(source.version)")
        }
        let root = sourceURL.deletingLastPathComponent()
        model = source.model.hasPrefix("/")
            ? source.model
            : root.appendingPathComponent(source.model).standardizedFileURL.path
        if let file = source.systemFile {
            instructions = try String(
                contentsOf: root.appendingPathComponent(file),
                encoding: .utf8
            )
        } else {
            instructions = source.instructions ?? source.system
        }
        switch source.tools {
        case .names(let names):
            tools = names
        case .file(let file):
            let toolSource = try JSONDecoder().decode(
                AgentfileTools.self,
                from: Data(contentsOf: root.appendingPathComponent(file))
            )
            guard toolSource.version == 1 else {
                throw AgentCLIError("unsupported tools file version \(toolSource.version)")
            }
            tools = toolSource.tools
        case nil:
            tools = []
        }
        if let declared = source.defaultMode {
            mode = declared
        } else if let file = source.args {
            let interface = try JSONDecoder().decode(
                AgentfileInterface.self,
                from: Data(contentsOf: root.appendingPathComponent(file))
            )
            guard interface.version == 1 else {
                throw AgentCLIError("unsupported agent interface version \(interface.version)")
            }
            mode = interface.run?.defaultMode == "interactive" ? .interactive : .once
        } else {
            mode = .once
        }
    } else {
        guard let modelValue = value("--model", in: arguments) else {
            try launchPiAuthoringAgent(name: name)
        }
        model = modelValue
        if let file = value("--system-file", in: arguments) {
            instructions = try String(contentsOfFile: file, encoding: .utf8)
        } else {
            instructions = value("--system", in: arguments)
        }
        tools = values("--tool", in: arguments)
        mode = arguments.contains("--interactive") ? .interactive : .once
    }

    let artifact = try AgentArtifact.create(
        at: output,
        name: name,
        modelPackagePath: model,
        instructions: instructions,
        tools: tools,
        defaultMode: mode
    )
    fputs("Created \(artifact.url.path)\n", stderr)
    fputs("Model: \(artifact.manifest.model.smeltPackageIdentity)\n", stderr)
}

private func run(_ arguments: [String]) throws {
    guard let target = arguments.first else { throw AgentCLIError("run requires an agent") }
    let artifact = try AgentStore.resolve(target)
    let maxTokens = Int(value("--max-tokens", in: arguments) ?? "512") ?? 0
    let temperature = value("--temperature", in: arguments).flatMap(Float.init)
    let seed = UInt64(value("--seed", in: arguments) ?? "0") ?? 0
    let interactive = arguments.contains("--interactive")
        || arguments.contains("-i")
        || (artifact.manifest.defaultMode == .interactive
            && !arguments.contains("--once"))
    let words = positionalTail(
        Array(arguments.dropFirst()),
        valueFlags: ["--max-tokens", "--temperature", "--seed", "--prompt"]
    )
    let explicitPrompt = value("--prompt", in: arguments)
        ?? (words.isEmpty ? nil : words.joined(separator: " "))
    if interactive {
        try launchPiInteractiveAgent(
            artifact: artifact,
            initialPrompt: explicitPrompt,
            maxTokens: maxTokens,
            temperature: temperature,
            seed: seed
        )
    }

    let session = try AgentSession(artifact: artifact)
    func generate(_ prompt: String) throws {
        let response = try session.run(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            seed: seed
        )
        print(response.text)
    }

    let prompt: String
    if let explicitPrompt {
        prompt = explicitPrompt
    } else if isatty(STDIN_FILENO) == 0 {
        prompt = String(data: FileHandle.standardInput.readDataToEndOfFile(), encoding: .utf8) ?? ""
    } else {
        throw AgentCLIError("run requires a prompt, piped stdin, or --interactive")
    }
    try generate(prompt)
}

/// Internal adapter used by the Pi integration while it supervises the linked
/// Smelt HTTP transport. Serving remains a Smelt library surface; this command
/// is deliberately absent from `smelt agent help` and carries no agent policy.
private func serveModelTransport(_ arguments: [String]) throws {
    guard let target = arguments.first else {
        throw AgentCLIError("_serve-model requires an agent")
    }
    let artifact = try AgentStore.resolve(target)
    let package = try artifact.manifest.resolveModel()
    let transport = value("--transport", in: arguments) ?? "http"
    guard transport == "http" else {
        throw AgentCLIError("_serve-model supports only --transport http")
    }
    let host = value("--host", in: arguments) ?? "127.0.0.1"
    let portValue = value("--port", in: arguments) ?? "8080"
    guard let port = UInt16(portValue), port > 0 else {
        throw AgentCLIError("--port must be a positive integer in [1, 65535]")
    }
    let contextLimit: Int?
    if let raw = value("--context-limit", in: arguments) {
        guard let parsed = Int(raw), parsed > 0 else {
            throw AgentCLIError("--context-limit must be positive")
        }
        contextLimit = parsed
    } else {
        contextLimit = nil
    }
    installFatalSignalHandlers(label: "smelt agent model server")
    try SmeltTextServer.run(SmeltTextServerConfiguration(
        packagePath: package.packageURL.path,
        endpoint: .http(host: host, port: port),
        contextLimit: contextLimit,
        templateOverride: value("--template", in: arguments)
    ))
}

private func install(_ arguments: [String]) throws {
    guard let name = arguments.first,
          let registry = value("--registry", in: arguments)
    else { throw AgentCLIError("install requires a name and --registry DIR") }
    let installed = try AgentRegistry.install(
        name: name,
        registry: URL(fileURLWithPath: registry, isDirectory: true)
    )
    fputs("Installed \(installed.url.path)\n", stderr)
}

private func publish(_ arguments: [String]) throws {
    guard let target = arguments.first,
          let registry = value("--registry", in: arguments)
    else { throw AgentCLIError("publish requires an agent and --registry DIR") }
    let artifact = try AgentStore.resolve(target)
    let name = value("--name", in: arguments) ?? artifact.manifest.name
    let output = try AgentRegistry.publish(
        artifact,
        name: name,
        registry: URL(fileURLWithPath: registry, isDirectory: true)
    )
    fputs("Published \(output.path)\n", stderr)
}

private func listModelPackages() throws {
    for path in try AgentModelCatalog.installedPackagePaths() {
        print(path)
    }
}

func runAgentCommand(_ arguments: [String]) {
    guard let command = arguments.first else { agentFail("\n\(agentUsage)") }
    do {
        switch command {
        case "help", "--help", "-h": print(agentUsage)
        case "create": try create(Array(arguments.dropFirst()))
        case "run": try run(Array(arguments.dropFirst()))
        case "_serve-model": try serveModelTransport(Array(arguments.dropFirst()))
        case "_list-model-packages": try listModelPackages()
        case "install": try install(Array(arguments.dropFirst()))
        case "publish": try publish(Array(arguments.dropFirst()))
        default: throw AgentCLIError("unknown command '\(command)'\n\n\(agentUsage)")
        }
    } catch {
        agentFail("\(error)")
    }
}
