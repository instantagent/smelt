// CAM text-to-PCM warm runtime — the audio side of `smelt run --linger N`.
// Same ControlPersist lifecycle as text generation, with PCM frames instead
// of one final text blob.
//
// Audio can't wait for the end, so warm runtime replies are streams of typed
// frames (type byte + u32 LE length + payload):
//
//   'M'  JSON stream metadata, used by routes whose shape is runtime-defined
//   'A'  Float32 PCM chunk, forwarded as generated
//        (segmented so no frame exceeds 1 MB)
//   'S'  stats trailer (JSON, currently empty) — clean completion, then close
//   'E'  UTF-8 error message — terminal
//
// The client resolves every parameter (declared args, voice defaults, flags)
// before forwarding; only resolved primitives cross the socket. If a frame
// write fails (client hung up), the warm runtime cancels generation by returning
// false from the generateStreaming chunk callback. Requests are sequential.

import Darwin
import Foundation
import SmeltRuntime
import SmeltSchema

private let camTextToPCMWarmSocketIdentityKey = ["text", "to", "pcm", "adapter"]
    .joined(separator: "-")

struct CAMTextToPCMWarmRequest: Codable {
    let text: String
    let speaker: String?
    let language: String
    let instruct: String?
    let maxFrames: Int
    let firstChunkFrames: Int
    let maxChunkFrames: Int
    let greedy: Bool
    let seed: UInt64?
}

private enum CAMTextToPCMWarmFrame {
    static let audio = UInt8(ascii: "A")
    static let stats = UInt8(ascii: "S")
    static let error = UInt8(ascii: "E")
    static let metadata = UInt8(ascii: "M")
}

/// Socket key: package identity + rebuild markers. Params travel
/// per-request, but a rebuild still rotates the warm process so nothing read at
/// load time (now or later) can go stale.
func camTextToPCMWarmSocketPath(
    packagePath: String,
    camIdentity: LingerCAMIdentity? = nil
) -> String {
    lingerSocketPath(
        packagePath: packagePath,
        contextLimit: nil,
        grammarBindings: [
            camTextToPCMWarmSocketIdentityKey: camTextToPCMWarmArtifactSignatures(
                packagePath: packagePath
            )
        ],
        camIdentity: camIdentity
    )
}

private func camTextToPCMWarmArtifactSignatures(
    packagePath: String
) -> [String] {
    let files = Set(CAMTextToPCM24KPackageLayout.lingerIdentityFiles)
    return files.sorted().map {
        camTextToPCMWarmArtifactSignature(packagePath: packagePath, relativePath: $0)
    }
}

private func camTextToPCMWarmArtifactSignature(
    packagePath: String,
    relativePath: String
) -> String {
    let fileManager = FileManager.default
    let packageURL = URL(fileURLWithPath: packagePath)
        .resolvingSymlinksInPath()
    let url = packageURL.appendingPathComponent(relativePath)
    var isDirectory = ObjCBool(false)
    guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
        return "\(relativePath)=missing"
    }
    guard isDirectory.boolValue else {
        return "\(relativePath)=\(camTextToPCMWarmArtifactStamp(url))"
    }
    let children = fileManager.enumerator(
        at: url,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles]
    )?.compactMap { entry -> String? in
        guard let childURL = entry as? URL else { return nil }
        let childPath = childURL.path
        let basePath = url.path + "/"
        let childRelativePath = childPath.hasPrefix(basePath)
            ? String(childPath.dropFirst(basePath.count))
            : childURL.lastPathComponent
        return "\(relativePath)/\(childRelativePath)=\(camTextToPCMWarmArtifactStamp(childURL))"
    }.sorted() ?? []
    return ([relativePath + "=" + camTextToPCMWarmArtifactStamp(url)] + children)
        .joined(separator: ";")
}

private func camTextToPCMWarmArtifactStamp(_ url: URL) -> String {
    let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
    let date = attrs?[.modificationDate] as? Date
    let size = (attrs?[.size] as? NSNumber)?.uint64Value ?? 0
    let type = (attrs?[.type] as? FileAttributeType) == .typeDirectory ? "dir" : "file"
    let mtime = Int((date?.timeIntervalSince1970 ?? 0) * 1_000_000_000)
    return "\(type):\(mtime):\(size)"
}

// MARK: - Frame IO

private func writeFrame(_ fd: Int32, type: UInt8, payload: Data) -> Bool {
    var header = Data(capacity: 5)
    header.append(type)
    var len = UInt32(payload.count).littleEndian
    withUnsafeBytes(of: &len) { header.append(contentsOf: $0) }
    return writeAllData(fd, header) && writeAllData(fd, payload)
}

private func writeAllData(_ fd: Int32, _ data: Data) -> Bool {
    guard !data.isEmpty else { return true }
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

/// Read until EOF with a hard size cap; nil when the cap is exceeded or the
/// socket times out / errors before EOF.
private func readToEOFBounded(_ fd: Int32, maxBytes: Int) -> Data? {
    var out = Data()
    var buf = [UInt8](repeating: 0, count: 64 * 1024)
    while true {
        let n = read(fd, &buf, buf.count)
        if n == 0 { return out }
        guard n > 0 else { return nil }
        out.append(contentsOf: buf[0..<n])
        guard out.count <= maxBytes else { return nil }
    }
}

private func readExact(_ fd: Int32, _ count: Int) -> Data? {
    guard count > 0 else { return Data() }
    var out = Data(capacity: count)
    var buf = [UInt8](repeating: 0, count: min(count, 64 * 1024))
    while out.count < count {
        let want = min(buf.count, count - out.count)
        let n = read(fd, &buf, want)
        guard n > 0 else { return nil }
        out.append(contentsOf: buf[0..<n])
    }
    return out
}

// MARK: - Client

/// Forward a synthesis request to a live CAM text-to-PCM warm runtime. Returns
/// false when no runtime is reachable before audio flowed; a failure after
/// first audio is terminal because partial audio already played or streamed.
func tryCAMTextToPCMWarmForward(
    socketPath: String,
    request: CAMTextToPCMWarmRequest,
    onSamples: ([Float]) -> Void
) -> Bool {
    guard FileManager.default.fileExists(atPath: socketPath) else { return false }
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else { return false }
    defer { close(fd) }
    var nosigpipe: Int32 = 1
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &nosigpipe, socklen_t(MemoryLayout<Int32>.size))

    let connected = withSockAddrUnix(socketPath) { sa, len in
        connect(fd, sa, len) == 0
    } ?? false
    guard connected else {
        unlink(socketPath)
        return false
    }
    // Synthesis is bounded by maxFrames, but guard a wedged warm runtime.
    var timeout = timeval(tv_sec: 600, tv_usec: 0)
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))

    guard let payload = try? JSONEncoder().encode(request),
          writeAllData(fd, payload)
    else { return false }
    shutdown(fd, SHUT_WR)

    var sawAudio = false
    while true {
        guard let header = readExact(fd, 5) else {
            // Warm runtime vanished. Before audio: clean fallback to inline.
            // After: the stream is unrecoverable.
            if sawAudio {
                fputs("smelt run: CAM text-to-PCM warm runtime died mid-stream\n", stderr)
                exit(1)
            }
            return false
        }
        let type = header[0]
        // load (not loadUnaligned) traps when the Data slice's base isn't
        // 4-byte aligned — read the LE length without an alignment assumption.
        let length = header.subdata(in: 1..<5).withUnsafeBytes {
            Int(UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self)))
        }
        guard length <= 64 * 1024 * 1024,
              let payload = readExact(fd, length) else {
            if sawAudio {
                fputs("smelt run: CAM text-to-PCM warm stream truncated\n", stderr)
                exit(1)
            }
            // Connected but speaking garbage: retire the socket so the
            // inline fallback can leave a fresh warm runtime behind, instead of
            // every future call cold-falling-back to this broken one.
            unlink(socketPath)
            return false
        }
        switch type {
        case CAMTextToPCMWarmFrame.audio:
            // A payload that isn't a whole number of float samples is a
            // corrupt/stale warm runtime, not audio — fail the protocol rather than
            // silently dropping the trailing bytes.
            guard payload.count % MemoryLayout<Float>.stride == 0 else {
                if sawAudio {
                    fputs("smelt run: CAM text-to-PCM warm audio frame not float-aligned\n", stderr)
                    exit(1)
                }
                unlink(socketPath)
                return false
            }
            sawAudio = true
            let samples = [Float](unsafeUninitializedCapacity: payload.count / 4) {
                dest, count in
                _ = payload.copyBytes(to: dest)
                count = payload.count / 4
            }
            onSamples(samples)
        case CAMTextToPCMWarmFrame.stats:
            return true
        case CAMTextToPCMWarmFrame.error:
            let message = String(data: payload, encoding: .utf8) ?? "unknown error"
            fputs("smelt run: CAM text-to-PCM warm runtime error: \(message)\n", stderr)
            exit(1)
        default:
            if sawAudio {
                fputs("smelt run: CAM text-to-PCM warm protocol error (frame '\(type)')\n", stderr)
                exit(1)
            }
            unlink(socketPath)
            return false
        }
    }
}

// MARK: - Worker

/// Serve CAM text-to-PCM requests until idle. Same bind-before-load + probe-alive
/// dance as the text runtime, so racing clients block on the backlog instead
/// of spawning duplicates.
func runCAMTextToPCMWarmRuntime(
    packagePath: String,
    construction: CAMTextToPCMRuntimeConstruction,
    socketPath: String,
    idleSeconds: Int
) {
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

    let runtime: CAMTextToPCM24KRuntime
    do {
        try construction.requirePackagePath(packagePath)
        runtime = try construction.make24KRuntime(verb: "linger-worker")
        // Warm the compiled trunk (bf16 packages) here, while the runtime is starting up,
        // so its ~400-pipeline compile never lands on a lingered request's TTFA.
        try runtime.prewarmCompiledTrunk()
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
        var nosigpipe: Int32 = 1
        setsockopt(
            clientFd, SOL_SOCKET, SO_NOSIGPIPE,
            &nosigpipe, socklen_t(MemoryLayout<Int32>.size)
        )
        // The request is one small JSON blob sent immediately; a client
        // that stalls or never half-closes must not pin the sequential
        // warm runtime (or grow the buffer without bound).
        var requestTimeout = timeval(tv_sec: 10, tv_usec: 0)
        setsockopt(
            clientFd, SOL_SOCKET, SO_RCVTIMEO,
            &requestTimeout, socklen_t(MemoryLayout<timeval>.size)
        )

        let requestData = readToEOFBounded(clientFd, maxBytes: 64 * 1024)
        guard let requestData,
              let request = try? JSONDecoder().decode(
                  CAMTextToPCMWarmRequest.self, from: requestData
              ) else {
            _ = writeFrame(
                clientFd, type: CAMTextToPCMWarmFrame.error,
                payload: Data("malformed CAM text-to-PCM warm request".utf8)
            )
            close(clientFd)
            continue
        }

        do {
            var clientGone = false
            try runtime.generateStreaming(
                text: request.text,
                instruct: request.instruct,
                language: request.language,
                speaker: request.speaker,
                maxFrames: request.maxFrames,
                decode: camTextToPCMWarmDecodeMode(for: request),
                firstChunkFrames: request.firstChunkFrames,
                maxChunkFrames: request.maxChunkFrames
            ) { chunk in
                guard !chunk.samples.isEmpty else { return true }
                // Segment so no frame exceeds 1 MB — a huge --max-frames
                // request must not produce frames past the client's cap.
                let samplesPerFrame = 256 * 1024
                var start = 0
                while start < chunk.samples.count {
                    let end = min(start + samplesPerFrame, chunk.samples.count)
                    let payload = chunk.samples[start..<end].withUnsafeBufferPointer {
                        Data(buffer: $0)
                    }
                    guard writeFrame(clientFd, type: CAMTextToPCMWarmFrame.audio, payload: payload)
                    else {
                        // Client hung up (barge-in or died): stop generating.
                        clientGone = true
                        return false
                    }
                    start = end
                }
                return true
            }
            if !clientGone {
                _ = writeFrame(clientFd, type: CAMTextToPCMWarmFrame.stats, payload: Data("{}".utf8))
            }
        } catch {
            _ = writeFrame(
                clientFd, type: CAMTextToPCMWarmFrame.error,
                payload: Data("\(error)".utf8)
            )
        }
        close(clientFd)
    }
}

private func camTextToPCMWarmDecodeMode(
    for request: CAMTextToPCMWarmRequest
) -> CAMTextToPCMDecodeMode {
    if request.greedy { return .greedy }
    if let seed = request.seed { return .sampleSeeded(seed) }
    return .packageDefault
}
