import Foundation

// Line-oriented transport retained for the smelt CLI adapter.

// Serial-dispatch invariant: SmeltServe's run loop awaits each
// read()/write() to completion before issuing the next. Re-entrant
// calls from multiple Tasks would race on nextRequestId — if v2
// adds concurrency, switch to an actor-isolated counter.
public final class StdioTransport: SmeltServeTransport, @unchecked Sendable {
    private var nextRequestId: SmeltServeRequestId = 0
    private let readQueue = DispatchQueue(label: "smelt.serve.stdio.read")
    private let writeQueue = DispatchQueue(label: "smelt.serve.stdio.write")

    public init() {}

    public func start() async throws {}

    public func read() async throws -> SmeltServeRawRequest? {
        while true {
            let line = await readLineOffMainThread()
            guard let line else { return nil }
            if line.trimmingCharacters(in: .whitespaces).isEmpty { continue }
            let id = nextRequestId
            nextRequestId &+= 1
            return try decodeEnvelope(line: line, id: id)
        }
    }

    public func write(
        _ response: SmeltServeRawResponse,
        requestId: SmeltServeRequestId
    ) async throws {
        await writeOffMainThread(response.body)
    }

    public func beginStream(
        contentType: String,
        requestId: SmeltServeRequestId,
        extraHeaders: [String: String]
    ) async throws -> SmeltServeStreamHandle {
        // Streaming over stdio sends JSON envelopes per chunk so the
        // line-oriented reader can distinguish stream events from the
        // legacy single-response shape. extraHeaders are dropped on
        // stdio — the envelope shape doesn't have a per-message header
        // channel and stdio consumers don't rely on response headers.
        await writeOffMainThread(envelope("stream-start", contentType: contentType))
        return StdioStreamHandle(transport: self)
    }

    public func stop() async {}

    fileprivate func writeStreamChunk(_ data: Data) async {
        await writeOffMainThread(envelope("stream-chunk", data: data))
    }

    fileprivate func writeStreamEnd() async {
        await writeOffMainThread(envelope("stream-end"))
    }

    private func envelope(
        _ event: String,
        contentType: String? = nil,
        data: Data? = nil
    ) -> Data {
        var obj: [String: Any] = ["event": event]
        if let contentType { obj["content_type"] = contentType }
        if let data { obj["data"] = String(data: data, encoding: .utf8) ?? "" }
        return (try? JSONSerialization.data(withJSONObject: obj)) ?? Data()
    }

    private func readLineOffMainThread() async -> String? {
        await withCheckedContinuation { (cont: CheckedContinuation<String?, Never>) in
            readQueue.async {
                cont.resume(returning: readLine(strippingNewline: true))
            }
        }
    }

    private func writeOffMainThread(_ body: Data) async {
        let lineBytes = body + Data([0x0A])
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            writeQueue.async {
                FileHandle.standardOutput.write(lineBytes)
                cont.resume()
            }
        }
    }

    private func decodeEnvelope(
        line: String,
        id: SmeltServeRequestId
    ) throws -> SmeltServeRawRequest {
        let data = Data(line.utf8)
        guard
            let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw StdioTransportError.malformedEnvelope
        }
        guard let pathString = parsed["path"] as? String else {
            throw StdioTransportError.missingPath
        }
        guard let path = SmeltServePath(rawValue: pathString) else {
            throw StdioTransportError.unknownPath(pathString)
        }
        let method: SmeltServeMethod
        if let methodString = parsed["method"] as? String {
            guard let parsed = SmeltServeMethod(rawValue: methodString) else {
                throw StdioTransportError.unknownMethod(methodString)
            }
            method = parsed
        } else {
            method = .post
        }

        let bodyData: Data
        if let body = parsed["body"] {
            bodyData = try JSONSerialization.data(withJSONObject: body)
        } else {
            bodyData = "{}".data(using: .utf8)!
        }

        return SmeltServeRawRequest(
            id: id,
            method: method,
            path: path,
            headers: [:],
            body: bodyData
        )
    }
}

enum StdioTransportError: Error, CustomStringConvertible {
    case malformedEnvelope
    case missingPath
    case unknownPath(String)
    case unknownMethod(String)

    var description: String {
        switch self {
        case .malformedEnvelope: return "Stdio request was not a JSON object"
        case .missingPath: return "Stdio request missing required 'path' field"
        case .unknownPath(let p): return "Unknown stdio request path: \(p)"
        case .unknownMethod(let m): return "Unknown stdio request method: \(m)"
        }
    }
}

private final class StdioStreamHandle: SmeltServeStreamHandle, @unchecked Sendable {
    private let transport: StdioTransport
    init(transport: StdioTransport) { self.transport = transport }
    func writeChunk(_ data: Data) async throws {
        await transport.writeStreamChunk(data)
    }
    func end() async throws {
        await transport.writeStreamEnd()
    }
}
