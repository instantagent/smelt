// Linger — ControlPersist-style warm workers for `smelt run --linger N`.
//
// The first `--linger` invocation runs inline, then leaves a detached worker
// process behind holding the loaded model. Follow-up invocations on the same
// package find the worker's Unix socket and forward the request, skipping
// package load entirely. The worker exits after N seconds idle; if it's gone
// or unreachable, the client falls back to an inline run.
//
// The socket name keys on the package's resolved path, context limit, and the
// manifest + prepared-artifact mtimes — rebuilding a package routes
// around any still-running stale worker.
//
// Protocol: one JSON request line, client half-closes, worker replies with one
// JSON response and closes. Requests are served sequentially.

import Darwin
import Foundation
import SmeltRuntime
import SmeltSchema

private let lingerCAMIdentityArgument = "--module-linger-identity"

struct LingerCAMIdentity: Codable, Equatable, Sendable {
    let camSemanticSHA256: String
    let exportABISHA256: String
    let exportID: String
    let flowID: String
    let inputPorts: [LingerCAMPortRecord]
    let outputPorts: [LingerCAMPortRecord]
    let authoredCapabilities: [String]
    let matchedGateIDs: [String]

    func isWorkerCompatible(with actual: LingerCAMIdentity) -> Bool {
        camSemanticSHA256 == actual.camSemanticSHA256
            && exportABISHA256 == actual.exportABISHA256
            && exportID == actual.exportID
            && flowID == actual.flowID
            && inputPorts == actual.inputPorts
            && outputPorts == actual.outputPorts
            && authoredCapabilities == actual.authoredCapabilities
            && matchedGateIDs == actual.matchedGateIDs
    }
}

struct LingerCAMPortRecord: Codable, Equatable, Sendable, Comparable {
    let name: String
    let typeName: String
    let attributes: [String: String]
    let optional: Bool

    static func < (lhs: LingerCAMPortRecord, rhs: LingerCAMPortRecord) -> Bool {
        if lhs.name != rhs.name { return lhs.name < rhs.name }
        if lhs.typeName != rhs.typeName { return lhs.typeName < rhs.typeName }
        if lhs.optional != rhs.optional { return !lhs.optional && rhs.optional }
        let lhsAttributes = lhs.attributes.sorted { left, right in
            left.key == right.key ? left.value < right.value : left.key < right.key
        }
        let rhsAttributes = rhs.attributes.sorted { left, right in
            left.key == right.key ? left.value < right.value : left.key < right.key
        }
        for (left, right) in zip(lhsAttributes, rhsAttributes) {
            if left.key != right.key { return left.key < right.key }
            if left.value != right.value { return left.value < right.value }
        }
        return lhsAttributes.count < rhsAttributes.count
    }
}

struct LingerRequest: Codable {
    let prompt: String
    let systemPrompt: String
    let template: String
    let maxTokens: Int
    let tempArg: String
    let seedArg: String
}

struct LingerResponse: Codable {
    let ok: Bool
    let error: String?
    let result: LingerResult?
}

struct LingerResult: Codable {
    let prompt: String
    let promptTokens: Int
    let mode: String
    let selection: String
    let generated: [Int32]
    let completion: String
    let prefillElapsed: Double
    let genElapsed: Double

    init(_ r: PromptRunResult) {
        prompt = r.prompt
        promptTokens = r.promptTokens
        mode = r.mode
        selection = r.selection
        generated = r.generated
        completion = r.completion
        prefillElapsed = r.prefillElapsed
        genElapsed = r.genElapsed
    }

    var promptRunResult: PromptRunResult {
        PromptRunResult(
            prompt: prompt,
            promptTokens: promptTokens,
            mode: mode,
            selection: selection,
            generated: generated,
            completion: completion,
            prefillElapsed: prefillElapsed,
            genElapsed: genElapsed
        )
    }
}

private func fnv1a64Hex(_ s: String) -> String {
    var hash: UInt64 = 0xcbf2_9ce4_8422_2325
    for byte in s.utf8 {
        hash = (hash ^ UInt64(byte)) &* 0x0000_0100_0000_01b3
    }
    return String(format: "%016llx", hash)
}

private func fileMTime(_ path: String) -> Int {
    // Nanosecond precision (APFS): a rebuild landing in the same second as
    // the file it replaces must still rotate the socket key.
    let attrs = try? FileManager.default.attributesOfItem(atPath: path)
    let date = attrs?[.modificationDate] as? Date
    return Int((date?.timeIntervalSince1970 ?? 0) * 1_000_000_000)
}

func lingerSocketPath(
    packagePath: String,
    contextLimit: Int?,
    grammarBindings: [String: [String]] = [:],
    camIdentity: LingerCAMIdentity? = nil
) -> String {
    let resolved = URL(fileURLWithPath: packagePath).resolvingSymlinksInPath().path
    // A worker's matcher is built at load with the invocation's grammar
    // bindings, so different binding sets get different workers. JSON with
    // sorted keys is the canonical encoding — delimiter-joining would let
    // crafted values collide (e.g. a value containing "&" or "=").
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let bindingsKey = (try? encoder.encode(grammarBindings))
        .flatMap { String(data: $0, encoding: .utf8) } ?? ""
    var keyParts = [
        resolved,
        String(contextLimit ?? -1),
        String(fileMTime("\(resolved)/manifest.json")),
        String(fileMTime("\(resolved)/\(SmeltPreparedArtifacts.prefixMetadata)")),
        String(fileMTime("\(resolved)/\(SmeltPreparedArtifacts.grammarMetadata)")),
        String(fileMTime("\(resolved)/\(SmeltPackageInterface.fileName)")),
        bindingsKey,
    ]
    if let camIdentity {
        keyParts += ["cam", encodedLingerCAMIdentity(camIdentity)]
    }
    let key = keyParts.joined(separator: "|")
    let dir = NSTemporaryDirectory()
    return "\(dir)smelt-linger-\(fnv1a64Hex(key)).sock"
}

func makeLingerCAMIdentity(
    decision: SmeltCAMPackageCapabilities.Decision,
    capabilities: SmeltCAMPackageCapabilities
) -> LingerCAMIdentity {
    return LingerCAMIdentity(
        camSemanticSHA256: capabilities.camSemanticSHA256,
        exportABISHA256: capabilities.exportABISHA256,
        exportID: decision.exportID,
        flowID: decision.flowID,
        inputPorts: decision.inputPorts.map(portRecord).sorted(),
        outputPorts: decision.outputPorts.map(portRecord).sorted(),
        authoredCapabilities: decision.authoredCapabilities.sorted(),
        matchedGateIDs: decision.matchedGateIDs.sorted()
    )
}

func encodedLingerCAMIdentity(_ identity: LingerCAMIdentity) -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let data = (try? encoder.encode(identity)) ?? Data()
    return data.base64EncodedString()
}

private func decodedLingerCAMIdentity(_ raw: String) -> LingerCAMIdentity? {
    guard let data = Data(base64Encoded: raw) else { return nil }
    guard let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
        return nil
    }
    let allowedFields: Set<String> = [
        "camSemanticSHA256",
        "exportABISHA256",
        "exportID",
        "flowID",
        "inputPorts",
        "outputPorts",
        "authoredCapabilities",
        "matchedGateIDs",
    ]
    guard Set(object.keys).isSubset(of: allowedFields) else { return nil }
    return try? JSONDecoder().decode(LingerCAMIdentity.self, from: data)
}

private func portRecord(_ port: SmeltCAMPackageDescriptor.Port) -> LingerCAMPortRecord {
    LingerCAMPortRecord(
        name: port.portName,
        typeName: port.type.typeName,
        attributes: port.type.attributes,
        optional: port.optional
    )
}

private func verifyExpectedLingerCAMIdentity(
    expected: String,
    actual: LingerCAMIdentity
) {
    guard !expected.isEmpty else { return }
    guard let decoded = decodedLingerCAMIdentity(expected) else {
        fputs("smelt linger-worker: invalid module linger identity\n", stderr)
        exit(1)
    }
    guard decoded.isWorkerCompatible(with: actual) else {
        fputs("smelt linger-worker: module linger identity mismatch\n", stderr)
        exit(1)
    }
}

// MARK: - Socket plumbing

func withSockAddrUnix<T>(
    _ path: String, _ body: (UnsafePointer<sockaddr>, socklen_t) -> T
) -> T? {
    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)
    let pathBytes = Array(path.utf8)
    let capacity = MemoryLayout.size(ofValue: addr.sun_path) - 1
    guard pathBytes.count <= capacity else { return nil }
    withUnsafeMutableBytes(of: &addr.sun_path) { raw in
        raw.copyBytes(from: pathBytes)
    }
    let len = socklen_t(MemoryLayout<sockaddr_un>.size)
    return withUnsafeBytes(of: &addr) { raw in
        body(raw.baseAddress!.assumingMemoryBound(to: sockaddr.self), len)
    }
}

private func readToEOF(_ fd: Int32) -> Data {
    var out = Data()
    var buf = [UInt8](repeating: 0, count: 64 * 1024)
    while true {
        let n = read(fd, &buf, buf.count)
        if n <= 0 { break }
        out.append(contentsOf: buf[0..<n])
    }
    return out
}

private func writeAll(_ fd: Int32, _ data: Data) -> Bool {
    var offset = 0
    return data.withUnsafeBytes { raw -> Bool in
        let base = raw.baseAddress!
        while offset < raw.count {
            let n = write(fd, base + offset, raw.count - offset)
            if n <= 0 { return false }
            offset += n
        }
        return true
    }
}

// MARK: - Client

/// Forward a request to a live linger worker. Returns nil when no worker is
/// reachable (caller runs inline); a stale socket is unlinked on the way out.
func tryLingerForward(
    socketPath: String, request: LingerRequest
) -> PromptRunResult? {
    guard FileManager.default.fileExists(atPath: socketPath) else { return nil }
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { return nil }
    defer { close(fd) }

    let connected = withSockAddrUnix(socketPath) { sa, len in
        connect(fd, sa, len) == 0
    } ?? false
    guard connected else {
        unlink(socketPath)
        return nil
    }

    // Generation can legitimately take a while; 10 minutes guards against a
    // wedged worker without capping real generations.
    var timeout = timeval(tv_sec: 600, tv_usec: 0)
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))

    guard let payload = try? JSONEncoder().encode(request),
          writeAll(fd, payload)
    else { return nil }
    shutdown(fd, SHUT_WR)

    let responseData = readToEOF(fd)
    guard let response = try? JSONDecoder().decode(LingerResponse.self, from: responseData)
    else { return nil }
    if let error = response.error {
        fputs("Linger worker error: \(error)\n", stderr)
        exit(1)
    }
    return response.result?.promptRunResult
}

/// Spawn a detached linger worker for the package unless one is already
/// listening. The worker is a fresh exec (never a fork — Metal state does not
/// survive fork) in its own session with stdio on /dev/null.
func spawnLingerWorker(
    packagePath: String,
    socketPath: String,
    idleSeconds: Int,
    contextLimit: Int?,
    grammarBindings: [String: [String]] = [:],
    camIdentity: LingerCAMIdentity? = nil
) {
    guard !FileManager.default.fileExists(atPath: socketPath) else { return }
    guard let exe = Bundle.main.executablePath else { return }

    var arguments = [
        exe, "linger-worker", packagePath,
        "--socket", socketPath,
        "--idle", String(idleSeconds),
    ]
    if let contextLimit {
        arguments += ["--context-limit", String(contextLimit)]
    }
    if let camIdentity {
        arguments += [lingerCAMIdentityArgument, encodedLingerCAMIdentity(camIdentity)]
    }
    for name in grammarBindings.keys.sorted() {
        arguments += ["--bind", "\(name)=\(grammarBindings[name]!.joined(separator: ","))"]
    }

    var attr: posix_spawnattr_t?
    posix_spawnattr_init(&attr)
    defer { posix_spawnattr_destroy(&attr) }
    posix_spawnattr_setflags(&attr, Int16(POSIX_SPAWN_SETSID))

    var fileActions: posix_spawn_file_actions_t?
    posix_spawn_file_actions_init(&fileActions)
    defer { posix_spawn_file_actions_destroy(&fileActions) }
    posix_spawn_file_actions_addopen(&fileActions, 0, "/dev/null", O_RDONLY, 0)
    posix_spawn_file_actions_addopen(&fileActions, 1, "/dev/null", O_WRONLY, 0)
    posix_spawn_file_actions_addopen(&fileActions, 2, "/dev/null", O_WRONLY, 0)

    let argv: [UnsafeMutablePointer<CChar>?] =
        arguments.map { strdup($0) } + [nil]
    defer { for arg in argv { free(arg) } }
    var pid: pid_t = 0
    _ = posix_spawn(&pid, exe, &fileActions, &attr, argv, environ)
}

// MARK: - Worker

func runLingerWorkerCommand() {
    guard args.count >= 3 else {
        fputs("Usage: smelt linger-worker <model.smeltpkg> --socket PATH --idle N\n", stderr)
        exit(1)
    }
    let packagePath = args[2]
    if hasArg("--cam-linger-identity") {
        fputs("smelt linger-worker: --cam-linger-identity was renamed to --module-linger-identity\n", stderr)
        exit(1)
    }
    let socketPath = parseArg("--socket")
    let idleSeconds = Int(parseArg("--idle", default: "30")) ?? 30
    let expectedCAMIdentity = parseArg(lingerCAMIdentityArgument)
    guard !socketPath.isEmpty else { exit(1) }

    let capabilities = requireCAMPackageCapabilitiesOrExit(
        packagePath: packagePath,
        verb: "linger-worker"
    )
    let construction = resolveCAMLingerWorkerOrExit(
        capabilities: capabilities,
        packagePath: packagePath,
        socketPath: socketPath,
        idleSeconds: idleSeconds,
        expectedCAMIdentity: expectedCAMIdentity
    )

    let contextLimit = (try? parsePositiveIntArg("--context-limit")) ?? nil

    // Bind before loading the model: early clients block on the backlog for
    // the load instead of spawning duplicate workers. A live listener on the
    // path means another worker won the race — exit quietly.
    let listenFd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard listenFd >= 0 else { exit(1) }
    if FileManager.default.fileExists(atPath: socketPath) {
        let probe = socket(AF_UNIX, SOCK_STREAM, 0)
        let alive = withSockAddrUnix(socketPath) { sa, len in
            connect(probe, sa, len) == 0
        } ?? false
        close(probe)
        if alive { exit(0) }
        unlink(socketPath)
    }
    let bound = withSockAddrUnix(socketPath) { sa, len in
        bind(listenFd, sa, len) == 0
    } ?? false
    guard bound, listen(listenFd, 16) == 0 else { exit(1) }
    chmod(socketPath, 0o600)
    defer { unlink(socketPath) }

    let context: RunContext
    do {
        context = try RunContext.load(
            packagePath: packagePath,
            contextLimit: contextLimit,
            grammarBindings: parseGrammarBindings(),
            construction: construction
        )
    } catch {
        unlink(socketPath)
        exit(1)
    }

    while true {
        var pollFd = pollfd(fd: listenFd, events: Int16(POLLIN), revents: 0)
        let ready = poll(&pollFd, 1, Int32(idleSeconds * 1000))
        if ready <= 0 { break }  // idle timeout (or poll error) — expire

        let clientFd = accept(listenFd, nil, nil)
        guard clientFd >= 0 else { continue }

        let requestData = readToEOF(clientFd)
        let response: LingerResponse
        if let request = try? JSONDecoder().decode(LingerRequest.self, from: requestData) {
            do {
                let (selectionMode, selectionDescription) = try resolveSelectionMode(
                    tempArg: request.tempArg,
                    seedArg: request.seedArg
                )
                let result = try context.evaluate(
                    prompt: request.prompt,
                    maxTokens: request.maxTokens,
                    template: request.template,
                    selectionMode: selectionMode,
                    selectionDescription: selectionDescription,
                    systemPrompt: request.systemPrompt
                )
                response = LingerResponse(
                    ok: true, error: nil, result: LingerResult(result)
                )
            } catch {
                response = LingerResponse(
                    ok: false, error: "\(error)", result: nil
                )
            }
        } else {
            response = LingerResponse(
                ok: false, error: "malformed linger request", result: nil
            )
        }
        if let payload = try? JSONEncoder().encode(response) {
            _ = writeAll(clientFd, payload)
        }
        close(clientFd)
    }
}

private struct CAMLingerRoute {
    let decision: SmeltCAMPackageCapabilities.Decision
    let runtimeRoute: CAMRuntimeRoute
}

private func resolveCAMLingerWorkerOrExit(
    capabilities: SmeltCAMPackageCapabilities,
    packagePath: String,
    socketPath: String,
    idleSeconds: Int,
    expectedCAMIdentity: String
) -> CAMTextRuntimeConstruction {
    let route = resolveCAMLingerRouteOrExit(capabilities)
    switch route.runtimeRoute {
    case .textToText:
        verifyExpectedLingerCAMIdentity(
            expected: expectedCAMIdentity,
            actual: makeLingerCAMIdentity(
                decision: route.decision,
                capabilities: capabilities
            )
        )
        return makeCAMTextRuntimeConstructionOrExit(
            packagePath: packagePath,
            capabilities: capabilities,
            decision: route.decision,
            verb: "linger-worker"
        )
    case .textToPCM(let outputRate):
        switch outputRate {
        case "24khz":
            verifyExpectedLingerCAMIdentity(
                expected: expectedCAMIdentity,
                actual: makeLingerCAMIdentity(
                    decision: route.decision,
                    capabilities: capabilities
                )
            )
            let construction = makeCAMTextToPCMRuntimeConstructionOrExit(
                packagePath: packagePath,
                capabilities: capabilities,
                decision: route.decision,
                verb: "linger-worker"
            )
            dispatchCAMTextToPCMWarmRuntimeOrExit(
                packagePath: packagePath,
                construction: construction,
                socketPath: socketPath,
                idleSeconds: idleSeconds
            )
        default:
            fputs(
                "smelt linger-worker: unsupported CAM audio output rate '\(outputRate)'\n",
                stderr
            )
            exit(1)
        }
    }
}

func requireCAMRunStreamCapabilityOrExit(
    _ decision: SmeltCAMPackageCapabilities.Decision,
    verb: String
) {
    guard decision.authoredCapabilities.contains("run.stream") else {
        fputs("smelt \(verb): no CAM export satisfies linger request\n", stderr)
        exit(1)
    }
}

private func resolveCAMLingerRouteOrExit(
    _ capabilities: SmeltCAMPackageCapabilities
) -> CAMLingerRoute {
    do {
        let decision = try capabilities.resolve(.runText)
        return makeCAMLingerRouteOrExit(
            decision: decision,
            capabilities: capabilities
        )
    } catch SmeltCAMPackageCapabilitiesError.noMatchingExport {
    } catch {
        fputs("smelt linger-worker: \(error)\n", stderr)
        exit(1)
    }

    do {
        let decision = try capabilities.resolve(.runAudio)
        requireCAMRunStreamCapabilityOrExit(decision, verb: "linger-worker")
        return makeCAMLingerRouteOrExit(
            decision: decision,
            capabilities: capabilities
        )
    } catch SmeltCAMPackageCapabilitiesError.noMatchingExport {
        fputs("smelt linger-worker: no CAM export satisfies linger request\n", stderr)
        exit(1)
    } catch {
        fputs("smelt linger-worker: \(error)\n", stderr)
        exit(1)
    }
}

private func makeCAMLingerRouteOrExit(
    decision: SmeltCAMPackageCapabilities.Decision,
    capabilities: SmeltCAMPackageCapabilities
) -> CAMLingerRoute {
    CAMLingerRoute(
        decision: decision,
        runtimeRoute: resolveCAMRuntimeRouteOrExit(
            capabilities: capabilities,
            decision: decision,
            verb: "linger-worker"
        )
    )
}

private func dispatchCAMTextToPCMWarmRuntimeOrExit(
    packagePath: String,
    construction: CAMTextToPCMRuntimeConstruction,
    socketPath: String,
    idleSeconds: Int
) -> Never {
    do {
        try construction.requirePackagePath(packagePath)
    } catch {
        fputs("smelt linger-worker: \(error)\n", stderr)
        exit(1)
    }
    runCAMTextToPCMWarmRuntime(
        packagePath: packagePath,
        construction: construction,
        socketPath: socketPath,
        idleSeconds: idleSeconds
    )
    exit(0)
}
