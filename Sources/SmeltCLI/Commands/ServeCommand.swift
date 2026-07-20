import Foundation
import SmeltRuntime
import SmeltServe

func runServeCommand() {
    let (packagePath, _) = resolvePackagePath(usage: [
        "Usage: smelt serve <model.smeltpkg> [--transport http|stdio]",
        "       [--port N] [--host 127.0.0.1|0.0.0.0] [--context-limit N] [--template NAME]\n",
    ])

    let transportName = parseArg("--transport", default: "stdio")
    let portString = parseArg("--port", default: "8080")
    guard let port = UInt16(portString), port > 0 else {
        fputs("smelt serve: --port must be a positive integer in [1, 65535]\n", stderr)
        exit(1)
    }
    let host = parseArg("--host", default: "127.0.0.1")

    if transportName == "http" && (host == "0.0.0.0" || host == "::") {
        fputs(
            "smelt serve: WARNING --host \(host) makes the HTTP listener reachable on the LAN without auth. Front with a reverse proxy if exposing.\n",
            stderr
        )
    }

    do {
        installFatalSignalHandlers(label: "smelt serve")
        switch try SmeltServeAdmission.resolve(packagePath: packagePath) {
        case .textToText(let construction):
            let endpoint: SmeltServeEndpoint
            switch transportName {
            case "stdio":
                endpoint = .stdio
            case "http":
                endpoint = .http(host: host, port: port)
            default:
                throw CLIError("unknown transport '\(transportName)'")
            }
            let contextLimit = try parsePositiveIntArg("--context-limit")
            let template = parseArg("--template").nilIfEmpty
            try SmeltTextServer.run(SmeltTextServerConfiguration(
                packagePath: packagePath,
                endpoint: endpoint,
                contextLimit: contextLimit,
                templateOverride: template
            ), construction: construction)
        case .textToPCM(let admission):
            let audioConstruction = try CAMTextToPCMRuntimeConstruction(
                serveAdmission: admission
            )
            try dispatchCAMTextToPCMServeRuntimeOrExit(
                packagePath: packagePath,
                construction: audioConstruction,
                transportName: transportName,
                port: port,
                host: host
            )
        }
    } catch {
        fputs("smelt serve: \(error)\n", stderr)
        exit(1)
    }
}

private func dispatchCAMTextToPCMServeRuntimeOrExit(
    packagePath: String,
    construction: CAMTextToPCMRuntimeConstruction,
    transportName: String,
    port: UInt16,
    host: String
) throws -> Never {
    try construction.requirePackagePath(packagePath)
    try dispatchCAMTextToPCMHTTPServeOrExit(
        packagePath: packagePath,
        construction: construction,
        transportName: transportName,
        port: port,
        host: host
    )
}

private func dispatchCAMTextToPCMHTTPServeOrExit(
    packagePath: String,
    construction: CAMTextToPCMRuntimeConstruction,
    transportName: String,
    port: UInt16,
    host: String
) throws -> Never {
    guard transportName == "http" else {
        fputs("smelt serve: a text-to-PCM package streams binary audio — use --transport http\n", stderr)
        exit(1)
    }
    try runCAMTextToPCMServeRuntime(
        packagePath: packagePath,
        construction: construction,
        port: port,
        host: host
    )
    exit(0)
}

/// Serve loop for a text-to-PCM package: same transport + serial request loop as
/// the text path, with SmeltTextToPCMServeHandler answering /v1/audio/* and /v1/models.
private func runCAMTextToPCMServeRuntime(
    packagePath: String,
    construction: CAMTextToPCMRuntimeConstruction,
    port: UInt16,
    host: String
) throws {
    let runtime = try construction.makeServeRuntime(verb: "serve")
    try runtime.prewarmForServe()
    let modelId = runtime.modelID
    let packageIdentity = try SmeltPackageIdentity.compute(packagePath: packagePath)
    let handler = SmeltTextToPCMServeHandler(
        runtime: runtime,
        modelId: modelId,
        packageIdentity: packageIdentity
    )
    let transport = try HTTPTransport(port: port, host: host)

    let semaphore = DispatchSemaphore(value: 0)
    Task {
        do {
            try await transport.start()
        } catch {
            fputs("smelt serve: start failed: \(error)\n", stderr)
            await transport.stop()
            semaphore.signal()
            return
        }
        fputs("smelt serve ready (transport=http host=\(host) port=\(port), text-to-pcm model=\(modelId))\n", stderr)

        loop: while true {
            let req: SmeltServeRawRequest?
            do {
                req = try await transport.read()
            } catch {
                fputs("smelt serve: read error: \(error) (continuing)\n", stderr)
                do {
                    try await transport.write(OpenAIJSON.errorResponse(
                        status: 400, code: .invalidRequest,
                        message: "Bad request: \(error)"
                    ), requestId: 0)
                } catch {
                    fputs("smelt serve: failed to write error: \(error) (exiting)\n", stderr)
                    break loop
                }
                continue
            }
            guard let req else { break }

            do {
                switch await handler.handle(req, transport: transport) {
                case .complete(let res):
                    try await transport.write(res, requestId: req.id)
                case .streamed:
                    break
                }
            } catch {
                fputs("smelt serve: write error: \(error) (exiting)\n", stderr)
                break loop
            }
        }

        await transport.stop()
        semaphore.signal()
    }
    semaphore.wait()
}
