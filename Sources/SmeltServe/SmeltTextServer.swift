import Foundation
import SmeltRuntime
import SmeltSchema

public struct SmeltServeError: Error, CustomStringConvertible, Sendable {
    public let description: String

    public init(_ description: String) {
        self.description = description
    }
}

public enum SmeltServeEndpoint: Sendable, Equatable {
    case http(host: String, port: UInt16)
    case stdio
}

public struct SmeltTextServerConfiguration: Sendable, Equatable {
    public let packagePath: String
    public let endpoint: SmeltServeEndpoint
    public let contextLimit: Int?
    public let templateOverride: String?

    public init(
        packagePath: String,
        endpoint: SmeltServeEndpoint,
        contextLimit: Int? = nil,
        templateOverride: String? = nil
    ) {
        self.packagePath = packagePath
        self.endpoint = endpoint
        self.contextLimit = contextLimit
        self.templateOverride = templateOverride
    }
}

/// Blocking text-generation server entry point shared by `smelt serve` and
/// consumers that embed Smelt, including Instant Agent. This API owns model
/// construction and transport setup; callers do not need the `smelt` executable.
public enum SmeltTextServer {
    public static func run(_ configuration: SmeltTextServerConfiguration) throws {
        switch try SmeltServeAdmission.resolve(packagePath: configuration.packagePath) {
        case .textToText(let construction):
            try run(configuration, construction: construction)
        case .textToPCM:
            throw SmeltServeError("CAM serve route is not text-to-text")
        }
    }

    package static func run(
        _ configuration: SmeltTextServerConfiguration,
        construction: CAMTextRuntimeConstruction
    ) throws {
        let packageURL = URL(
            fileURLWithPath: configuration.packagePath,
            isDirectory: true
        ).standardizedFileURL
        try construction.requirePackagePath(packageURL.path)
        let runtime = try construction.makeRuntime(contextLimit: configuration.contextLimit)
        let (manifest, inference) = try construction.loadManifestConfig()
        let tokenizer = try construction.makeTokenizer()
        let template = try construction.resolveTemplate(
            cliOverride: configuration.templateOverride
        )
        let handler = try SmeltServeHandler(
            packagePath: packageURL.path,
            runtime: runtime,
            tokenizer: tokenizer,
            inference: inference,
            modelId: manifest.modelName,
            template: template
        )

        let transport: any SmeltServeTransport
        let endpointDescription: String
        switch configuration.endpoint {
        case .stdio:
            transport = StdioTransport()
            endpointDescription = "transport=stdio"
        case .http(let host, let port):
            guard port > 0 else {
                throw SmeltServeError("HTTP port must be in [1, 65535]")
            }
            transport = try HTTPTransport(port: port, host: host)
            endpointDescription = "transport=http host=\(host) port=\(port)"
        }

        try runLoop(
            handler: handler,
            transport: transport,
            readyMessage: "smelt serve ready (\(endpointDescription), model=\(manifest.modelName))"
        )
    }

    private static func runLoop(
        handler: SmeltServeHandler,
        transport: any SmeltServeTransport,
        readyMessage: String
    ) throws {
        let completion = BlockingServeCompletion()
        Task {
            do {
                try await transport.start()
                fputs(readyMessage + "\n", stderr)

                while let request = try await readRequest(from: transport) {
                    do {
                        switch await handler.handle(request, transport: transport) {
                        case .complete(let response):
                            try await transport.write(response, requestId: request.id)
                        case .streamed:
                            break
                        }
                    } catch {
                        throw SmeltServeError("write error: \(error)")
                    }
                }
                await transport.stop()
                completion.finish(.success(()))
            } catch {
                await transport.stop()
                completion.finish(.failure(error))
            }
        }
        try completion.wait().get()
    }

    private static func readRequest(
        from transport: any SmeltServeTransport
    ) async throws -> SmeltServeRawRequest? {
        while true {
            do {
                return try await transport.read()
            } catch {
                fputs("smelt serve: read error: \(error) (continuing)\n", stderr)
                do {
                    try await transport.write(
                        OpenAIJSON.errorResponse(
                            status: 400,
                            code: .invalidRequest,
                            message: "Bad request: \(error)"
                        ),
                        requestId: 0
                    )
                } catch {
                    throw SmeltServeError("failed to write error response: \(error)")
                }
            }
        }
    }
}

private final class BlockingServeCompletion: @unchecked Sendable {
    private let lock = NSLock()
    private let semaphore = DispatchSemaphore(value: 0)
    private var result: Result<Void, Error>?

    func finish(_ result: Result<Void, Error>) {
        lock.lock()
        self.result = result
        lock.unlock()
        semaphore.signal()
    }

    func wait() -> Result<Void, Error> {
        semaphore.wait()
        lock.lock()
        defer { lock.unlock() }
        return result ?? .failure(SmeltServeError("serve loop ended without a result"))
    }
}
