import Foundation
import Network

// HTTP transport implementation shared by CLI and embedded consumers.

public final class HTTPTransport: SmeltServeTransport, @unchecked Sendable {
    private static let reservedStreamHeaders: Set<String> = [
        "content-type", "transfer-encoding",
        "cache-control", "connection",
    ]
    private let port: UInt16
    private let listener: NWListener
    private var nextRequestId: SmeltServeRequestId = 0
    private var pending: [SmeltServeRequestId: PendingRequest] = [:]

    /// Per-pending-request bookkeeping for the keep-alive write path:
    /// after responding we either restart the receive loop on the same
    /// connection (keep-alive) or close it (Connection: close).
    private struct PendingRequest {
        let connection: NWConnection
        let state: HTTPConnectionState
    }

    private let queueLock = NSLock()
    private let acceptQueue = DispatchQueue(label: "smelt.serve.http.accept")
    private let connectionsQueue = DispatchQueue(label: "smelt.serve.http.connections")

    private var requestStream: AsyncStream<SmeltServeRawRequest>!
    private var requestContinuation: AsyncStream<SmeltServeRawRequest>.Continuation!
    // AsyncStream is single-consumer; store one iterator and drive it
    // from read() instead of materializing a fresh iterator each call
    // (Apple docs: creating multiple iterators from one AsyncStream is
    // unsupported). The serial run-loop invariant — one outstanding
    // read at a time — keeps the mutating next() race-free here.
    private var requestIterator: AsyncStream<SmeltServeRawRequest>.AsyncIterator!

    public init(port: UInt16, host: String = "127.0.0.1") throws {
        self.port = port
        let params = NWParameters.tcp
        // Don't enable allowLocalEndpointReuse — on macOS that turns into
        // SO_REUSEPORT, which lets a second `smelt serve` silently coexist
        // on the same port and have the kernel load-balance between them.
        // That breaks the listener-readiness check downstream.
        // Default to loopback because the listener has no auth; binding
        // all interfaces exposes an unauthenticated chat endpoint.
        switch host {
        case "127.0.0.1", "::1", "localhost":
            params.requiredInterfaceType = .loopback
        case "0.0.0.0", "::":
            break
        default:
            throw HTTPTransportError.unsupportedHost(host)
        }
        self.listener = try NWListener(
            using: params,
            on: NWEndpoint.Port(rawValue: port) ?? .any
        )

        var continuation: AsyncStream<SmeltServeRawRequest>.Continuation!
        let stream = AsyncStream<SmeltServeRawRequest> { c in
            continuation = c
        }
        self.requestStream = stream
        self.requestContinuation = continuation
        self.requestIterator = stream.makeAsyncIterator()
    }

    public func start() async throws {
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            let connectionState = HTTPConnectionState()
            connection.stateUpdateHandler = { [weak self] state in
                guard let self else { return }
                switch state {
                case .failed, .cancelled:
                    connectionState.markCancelled()
                    self.dropPending(for: connection)
                default: break
                }
            }
            connection.start(queue: self.connectionsQueue)
            self.receiveRequest(on: connection, state: connectionState)
        }

        // NWListener.start reports bind failures asynchronously via the
        // state handler. Without awaiting `.ready` here, a busy-port
        // condition silently passes start(), the CLI prints "ready", and
        // read() hangs forever — turn that into a throw at start-time.
        let gate = ListenerStartGate()
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            listener.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    gate.fire(cont, error: nil)
                case .failed(let error):
                    gate.fire(cont, error: error)
                case .cancelled:
                    gate.fire(cont, error: HTTPTransportError.listenerCancelled)
                default: break
                }
            }
            listener.start(queue: acceptQueue)
        }
    }

    private func dropPending(for connection: NWConnection) {
        queueLock.withLock {
            pending = pending.filter { $0.value.connection !== connection }
        }
    }

    public func read() async throws -> SmeltServeRawRequest? {
        await requestIterator.next()
    }

    public func write(
        _ response: SmeltServeRawResponse,
        requestId: SmeltServeRequestId
    ) async throws {
        let pending = queueLock.withLock {
            self.pending.removeValue(forKey: requestId)
        }
        guard let pending else {
            fputs("smelt serve: HTTP request id \(requestId) had no live connection (client disconnected mid-request)\n", stderr)
            return
        }
        let connection = pending.connection
        let state = pending.state
        let keepAlive = state.keepAlive
        let bytes = encodeHTTPResponse(response, keepAlive: keepAlive)
        let sendSucceeded = await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
            connection.send(
                content: bytes,
                completion: .contentProcessed { error in
                    if let error {
                        fputs("smelt serve: HTTP send failed for request id \(requestId): \(error)\n", stderr)
                        cont.resume(returning: false)
                    } else {
                        cont.resume(returning: true)
                    }
                }
            )
        }
        resumeKeepAliveOrClose(
            connection: connection, state: state,
            sendSucceeded: sendSucceeded, keepAlive: keepAlive
        )
    }

    /// Shared post-response cleanup: on a successful send with
    /// keep-alive, try to parse a pipelined next request from the
    /// existing buffer or restart the receive loop. Otherwise
    /// cancel the connection. Used by both the buffered write()
    /// and the streaming end() paths.
    private func resumeKeepAliveOrClose(
        connection: NWConnection,
        state: HTTPConnectionState,
        sendSucceeded: Bool,
        keepAlive: Bool
    ) {
        if !sendSucceeded || !keepAlive {
            connection.cancel()
            return
        }
        // A receive may still be armed from beginStream. The state lock makes
        // parsing safe, and receiveLoop's one-reader gate makes this resume a
        // no-op until that existing receive completes.
        switch state.withMutableBuffer({ buffer in
            tryParseRequest(buffer: &buffer, on: connection, state: state)
        }) {
        case .complete:
            return
        case .incomplete:
            receiveLoop(connection: connection, state: state)
        case .failed(let status, let message):
            failConnection(connection, status: status, message: message)
        }
    }

    public func beginStream(
        contentType: String,
        requestId: SmeltServeRequestId,
        extraHeaders: [String: String]
    ) async throws -> SmeltServeStreamHandle {
        let pending = queueLock.withLock {
            self.pending.removeValue(forKey: requestId)
        }
        guard let pending else {
            throw HTTPTransportError.unknownRequestId(requestId)
        }
        let connection = pending.connection
        let state = pending.state
        // Send headers immediately so the client can start parsing
        // while we generate. Transfer-Encoding: chunked lets us emit
        // arbitrary-length data frames terminated by 0\r\n\r\n.
        var headerBytes = Data()
        headerBytes.append("HTTP/1.1 200 OK\r\n".data(using: .utf8)!)
        headerBytes.append("Content-Type: \(contentType)\r\n".data(using: .utf8)!)
        headerBytes.append("Transfer-Encoding: chunked\r\n".data(using: .utf8)!)
        headerBytes.append("Cache-Control: no-store\r\n".data(using: .utf8)!)
        headerBytes.append(
            "Connection: \(state.keepAlive ? "keep-alive" : "close")\r\n".data(using: .utf8)!
        )
        for (k, v) in extraHeaders
            where !Self.reservedStreamHeaders.contains(k.lowercased()) {
            headerBytes.append("\(k): \(v)\r\n".data(using: .utf8)!)
        }
        headerBytes.append("\r\n".data(using: .utf8)!)
        try await sendOrThrow(connection: connection, bytes: headerBytes)
        // Parsing a complete request used to leave no receive armed until the
        // response ended. For a long prefill that made a peer FIN invisible
        // until the first generated chunk. Keep exactly one receive in flight
        // during the stream; the gate in HTTPConnectionState makes the normal
        // keep-alive resume below idempotent.
        receiveLoop(connection: connection, state: state)
        return HTTPStreamHandle(transport: self, connection: connection, state: state)
    }

    fileprivate func sendOrThrow(
        connection: NWConnection, bytes: Data
    ) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            connection.send(
                content: bytes,
                completion: .contentProcessed { error in
                    if let error { cont.resume(throwing: error) }
                    else { cont.resume() }
                }
            )
        }
    }

    fileprivate func finalizeStream(
        connection: NWConnection, state: HTTPConnectionState
    ) async throws {
        // Terminating chunk = "0\r\n\r\n". Then either re-arm
        // keep-alive (matching the buffered write() post-send logic)
        // or close.
        let terminator = "0\r\n\r\n".data(using: .utf8)!
        var sendSucceeded = true
        do {
            try await sendOrThrow(connection: connection, bytes: terminator)
        } catch {
            fputs("smelt serve: HTTP stream terminator send failed: \(error)\n", stderr)
            sendSucceeded = false
        }
        resumeKeepAliveOrClose(
            connection: connection, state: state,
            sendSucceeded: sendSucceeded, keepAlive: state.keepAlive
        )
    }

    public func stop() async {
        listener.cancel()
        requestContinuation.finish()
        queueLock.withLock {
            for (_, pendingReq) in pending { pendingReq.connection.cancel() }
            pending.removeAll()
        }
    }

    private func receiveRequest(
        on connection: NWConnection,
        state: HTTPConnectionState
    ) {
        receiveLoop(connection: connection, state: state)
    }

    private func receiveLoop(connection: NWConnection, state: HTTPConnectionState) {
        guard state.beginReceive() else { return }
        connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) {
            [weak self] data, _, isComplete, error in
            guard let self else { return }
            state.endReceive()
            if let data {
                state.withMutableBuffer { buffer in buffer.append(data) }
            }

            switch state.withMutableBuffer({ buffer in
                self.tryParseRequest(buffer: &buffer, on: connection, state: state)
            }) {
            case .complete:
                return
            case .incomplete:
                if let error {
                    state.markCancelled()
                    self.failConnection(connection, status: 400, message: "Receive error: \(error)")
                    return
                }
                if isComplete {
                    state.markCancelled()
                    // Clean EOF with no buffered partial request is the
                    // expected end of a keep-alive client session.
                    if state.bufferIsEmpty {
                        connection.cancel()
                    } else {
                        self.failConnection(connection, status: 400, message: "Connection closed before request complete")
                    }
                    return
                }
                self.receiveLoop(connection: connection, state: state)
            case .failed(let status, let message):
                self.failConnection(connection, status: status, message: message)
            }
        }
    }

    private enum ParseOutcome {
        case incomplete
        case complete
        case failed(status: Int, message: String)
    }

    private func tryParseRequest(
        buffer: inout Data,
        on connection: NWConnection,
        state: HTTPConnectionState
    ) -> ParseOutcome {
        let headerDelim = Data([0x0D, 0x0A, 0x0D, 0x0A])
        guard let headerEnd = buffer.range(of: headerDelim) else {
            if buffer.count > 64 * 1024 {
                return .failed(status: 400, message: "Request headers exceed 64 KiB")
            }
            return .incomplete
        }

        let headerData = buffer[..<headerEnd.lowerBound]
        guard let headerString = String(data: Data(headerData), encoding: .utf8) else {
            return .failed(status: 400, message: "Headers are not valid UTF-8")
        }

        var lines = headerString.split(separator: "\r\n", omittingEmptySubsequences: false)
        guard !lines.isEmpty else {
            return .failed(status: 400, message: "Empty request")
        }
        let requestLine = String(lines.removeFirst())
        let requestParts = requestLine.split(separator: " ", maxSplits: 2).map(String.init)
        guard requestParts.count == 3,
              let method = SmeltServeMethod(rawValue: requestParts[0])
        else {
            return .failed(status: 400, message: "Malformed request line: \(requestLine)")
        }
        let pathOnly = requestParts[1].split(separator: "?", maxSplits: 1).map(String.init).first ?? requestParts[1]
        guard let path = SmeltServePath(rawValue: pathOnly) else {
            return .failed(status: 404, message: "Unknown path: \(pathOnly)")
        }
        // HTTP/1.0 default is `Connection: close`; HTTP/1.1 default is
        // keep-alive. Bias accordingly so HTTP/1.0 clients aren't left
        // hanging waiting for a FIN that never comes.
        let httpVersion = requestParts[2]
        if httpVersion.hasPrefix("HTTP/1.0") {
            state.keepAlive = false
        }

        var headers: [String: String] = [:]
        var contentLength = 0
        for line in lines where !line.isEmpty {
            if let colon = line.firstIndex(of: ":") {
                let key = String(line[..<colon]).trimmingCharacters(in: .whitespaces).lowercased()
                let value = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
                headers[key] = value
                if key == "content-length" {
                    guard let n = Int(value), n >= 0 else {
                        return .failed(status: 400, message: "Invalid Content-Length: \(value)")
                    }
                    contentLength = n
                }
            }
        }

        if contentLength > 10 * 1024 * 1024 {
            return .failed(status: 413, message: "Body exceeds 10 MiB limit")
        }

        let bodyStart = buffer.index(headerEnd.lowerBound, offsetBy: 4)
        let availableBody = buffer.distance(from: bodyStart, to: buffer.endIndex)
        if availableBody < contentLength {
            return .incomplete
        }
        let bodyEnd = buffer.index(bodyStart, offsetBy: contentLength)
        let bodyData = buffer[bodyStart..<bodyEnd]

        // HTTP/1.1 default is keep-alive; flip off on explicit
        // `Connection: close`. (HTTP/1.0's default-close was already
        // handled above by the protocol-version check.) The header is
        // comma-separated; tokenize and look for exact "close" rather
        // than substring-matching so "keep-alive, Upgrade" doesn't trip.
        if let conn = headers["connection"]?.lowercased() {
            let tokens = conn.split(whereSeparator: { $0 == "," || $0.isWhitespace })
            if tokens.contains("close") {
                state.keepAlive = false
            } else if tokens.contains("keep-alive") {
                // HTTP/1.0 clients can opt in via this header.
                state.keepAlive = true
            }
        }

        let id = nextId()
        let request = SmeltServeRawRequest(
            id: id,
            method: method,
            path: path,
            headers: headers,
            body: Data(bodyData)
        )
        queueLock.withLock {
            pending[id] = PendingRequest(connection: connection, state: state)
        }
        // Consume the parsed bytes so any pipelined next request that
        // arrived in the same recv() can be parsed next.
        buffer.removeSubrange(buffer.startIndex..<bodyEnd)
        requestContinuation.yield(request)
        return .complete
    }

    private func nextId() -> SmeltServeRequestId {
        queueLock.withLock {
            let id = nextRequestId
            nextRequestId &+= 1
            return id
        }
    }

    private func failConnection(
        _ connection: NWConnection,
        status: Int,
        message: String
    ) {
        let code: OpenAIErrorCode = status == 404 ? .notFound : .invalidRequest
        let response = OpenAIJSON.errorResponse(status: status, code: code, message: message)
        let bytes = encodeHTTPResponse(response, keepAlive: false)
        connection.send(
            content: bytes,
            completion: .contentProcessed { _ in connection.cancel() }
        )
    }

    private func encodeHTTPResponse(
        _ response: SmeltServeRawResponse,
        keepAlive: Bool
    ) -> Data {
        var out = Data()
        let statusText = httpStatusText(response.statusCode)
        // JSON unless the handler set an explicit Content-Type (e.g. audio/wav from the
        // TTS surface); the header loop below skips it so it isn't emitted twice.
        let contentType = response.headers.first { $0.key.lowercased() == "content-type" }?.value
            ?? "application/json"
        out.append("HTTP/1.1 \(response.statusCode) \(statusText)\r\n".data(using: .utf8)!)
        out.append("Content-Type: \(contentType)\r\n".data(using: .utf8)!)
        out.append("Content-Length: \(response.body.count)\r\n".data(using: .utf8)!)
        out.append(
            "Connection: \(keepAlive ? "keep-alive" : "close")\r\n".data(using: .utf8)!
        )
        for (k, v) in response.headers where k.lowercased() != "content-type" {
            out.append("\(k): \(v)\r\n".data(using: .utf8)!)
        }
        out.append("\r\n".data(using: .utf8)!)
        out.append(response.body)
        return out
    }

    private func httpStatusText(_ code: Int) -> String {
        switch code {
        case 200: return "OK"
        case 400: return "Bad Request"
        case 404: return "Not Found"
        case 405: return "Method Not Allowed"
        case 413: return "Payload Too Large"
        case 500: return "Internal Server Error"
        case 501: return "Not Implemented"
        default:  return "Unknown"
        }
    }
}

// Per-connection parser, receive, and cancellation state.
private final class HTTPConnectionState: @unchecked Sendable {
    // A receive remains armed while streamed generation runs, whereas response
    // finalization happens on the handler task. Protect the shared parser state
    // and use a recursive lock because tryParseRequest updates keepAlive while
    // it already owns the mutable-buffer lock.
    private let lock = NSRecursiveLock()
    private var buffer = Data()
    private var receiveInFlight = false
    private var cancelled = false
    // HTTP/1.1 default is keep-alive; flip off only when the client
    // explicitly sends `Connection: close`.
    //
    private var storedKeepAlive = true

    var keepAlive: Bool {
        get { lock.withLock { storedKeepAlive } }
        set { lock.withLock { storedKeepAlive = newValue } }
    }

    var isCancelled: Bool { lock.withLock { cancelled } }
    var bufferIsEmpty: Bool { lock.withLock { buffer.isEmpty } }

    func markCancelled() {
        lock.withLock { cancelled = true }
    }

    func beginReceive() -> Bool {
        lock.withLock {
            guard !receiveInFlight else { return false }
            receiveInFlight = true
            return true
        }
    }

    func endReceive() {
        lock.withLock { receiveInFlight = false }
    }

    func withMutableBuffer<R>(_ body: (inout Data) -> R) -> R {
        lock.withLock { body(&buffer) }
    }
}

// One-shot resume gate for NWListener.stateUpdateHandler, which
// keeps firing after `.ready`. Without this, a `.cancelled` after
// `.ready` would attempt a second `cont.resume` and trap.
private final class ListenerStartGate: @unchecked Sendable {
    private let lock = NSLock()
    private var fired = false

    func fire(_ cont: CheckedContinuation<Void, Error>, error: Error?) {
        lock.lock(); defer { lock.unlock() }
        guard !fired else { return }
        fired = true
        if let error { cont.resume(throwing: error) }
        else { cont.resume() }
    }
}

enum HTTPTransportError: Error, CustomStringConvertible {
    case unknownRequestId(SmeltServeRequestId)
    case listenerCancelled
    case streamHandleOrphaned
    case unsupportedHost(String)

    var description: String {
        switch self {
        case .unknownRequestId(let id):
            return "HTTP transport: no pending connection for request id \(id)"
        case .listenerCancelled:
            return "HTTP transport: listener cancelled before becoming ready"
        case .streamHandleOrphaned:
            return "HTTP transport: stream handle outlived its transport"
        case .unsupportedHost(let host):
            return "HTTP transport: --host must be 127.0.0.1, ::1, localhost, 0.0.0.0, or :: (got \"\(host)\")"
        }
    }
}

private extension NSLock {
    func withLock<R>(_ body: () -> R) -> R {
        self.lock(); defer { self.unlock() }
        return body()
    }
}

private extension NSRecursiveLock {
    func withLock<R>(_ body: () -> R) -> R {
        self.lock(); defer { self.unlock() }
        return body()
    }
}

private final class HTTPStreamHandle: SmeltServeStreamHandle, @unchecked Sendable {
    private weak var transport: HTTPTransport?
    private let connection: NWConnection
    private let state: HTTPConnectionState

    init(transport: HTTPTransport, connection: NWConnection, state: HTTPConnectionState) {
        self.transport = transport
        self.connection = connection
        self.state = state
    }

    var isCancelled: Bool { state.isCancelled }

    func writeChunk(_ data: Data) async throws {
        guard let transport else {
            throw HTTPTransportError.streamHandleOrphaned
        }
        // HTTP/1.1 chunked-encoding frame: `<hex-length>\r\n<data>\r\n`
        var frame = Data()
        frame.append(String(data.count, radix: 16).data(using: .utf8)!)
        frame.append("\r\n".data(using: .utf8)!)
        frame.append(data)
        frame.append("\r\n".data(using: .utf8)!)
        try await transport.sendOrThrow(connection: connection, bytes: frame)
    }

    func end() async throws {
        guard let transport else { return }
        try await transport.finalizeStream(connection: connection, state: state)
    }
}
