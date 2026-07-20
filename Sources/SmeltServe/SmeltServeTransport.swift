import Foundation

// Transport-neutral request, response, and streaming contracts.

public typealias SmeltServeRequestId = UInt64

public enum SmeltServeMethod: String, Sendable {
    case get = "GET"
    case post = "POST"
}

public enum SmeltServePath: String, Sendable {
    case chatCompletions = "/v1/chat/completions"
    case completions     = "/v1/completions"
    case models          = "/v1/models"
    case audioSpeech     = "/v1/audio/speech"   // text-to-PCM packages
    case audioVoices     = "/v1/audio/voices"   // text-to-PCM speakers/languages listing
}

public struct SmeltServeRawRequest: Sendable {
    public let id: SmeltServeRequestId
    public let method: SmeltServeMethod
    public let path: SmeltServePath
    public let headers: [String: String]
    public let body: Data

    public init(
        id: SmeltServeRequestId,
        method: SmeltServeMethod,
        path: SmeltServePath,
        headers: [String: String] = [:],
        body: Data
    ) {
        self.id = id
        self.method = method
        self.path = path
        self.headers = headers
        self.body = body
    }
}

public struct SmeltServeRawResponse: Sendable {
    public let statusCode: Int
    public let headers: [String: String]
    public let body: Data

    public init(
        statusCode: Int,
        headers: [String: String] = [:],
        body: Data
    ) {
        self.statusCode = statusCode
        self.headers = headers
        self.body = body
    }
}

public protocol SmeltServeTransport: AnyObject, Sendable {
    func start() async throws
    func read() async throws -> SmeltServeRawRequest?
    func write(_ response: SmeltServeRawResponse, requestId: SmeltServeRequestId) async throws
    /// Begin a streamed response. After this returns, the handler
    /// emits chunks via the returned handle and calls `end()` once
    /// the response is complete. The transport sends HTTP headers
    /// (HTTP/1.1 200 + Content-Type: text/event-stream +
    /// Transfer-Encoding: chunked) immediately so the client can
    /// start parsing while the server generates.
    func beginStream(
        contentType: String,
        requestId: SmeltServeRequestId,
        extraHeaders: [String: String]
    ) async throws -> SmeltServeStreamHandle
    func stop() async
}

/// A streaming response in flight. writeChunk emits one data
/// frame; end() flushes the terminating frame AND runs the post-
/// response keep-alive resume logic so the connection can serve
/// the next request.
///
/// Contract: end() MUST be called exactly once before the handle
/// goes out of scope — including on error paths — or the
/// transport's connection state leaks (the connection stays
/// pending and never re-arms for the next request).
public protocol SmeltServeStreamHandle: AnyObject, Sendable {
    /// True once the transport knows the peer can no longer consume the
    /// response. Generation checks this at model-step boundaries so a closed
    /// client does not keep advancing mutable model state in the background.
    var isCancelled: Bool { get }
    func writeChunk(_ data: Data) async throws
    func end() async throws
}

public extension SmeltServeStreamHandle {
    /// Non-network transports do not have an asynchronous peer-close signal.
    var isCancelled: Bool { false }
}
