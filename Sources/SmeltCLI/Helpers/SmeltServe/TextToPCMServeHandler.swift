import Foundation
import SmeltRuntime
import SmeltSchema
import SmeltServe

// SmeltTextToPCMServeHandler — the serve surface for text-to-PCM packages
// (`smelt serve <audio.smeltpkg> --transport http`). OpenAI-compatible where
// OpenAI has a shape:
//
//   POST /v1/audio/speech   {input, voice?, language?, instruct?, response_format?,
//                            seed?, greedy?, max_frames?, first_chunk_frames?,
//                            max_chunk_frames?}
    //     response_format "pcm" (default): chunked raw s16le at the runtime's
    //       declared sample rate and channel count, each chunk written the moment
    //       the model emits it.
//       Client disconnect mid-stream cancels the remaining generation (barge-in).
//     response_format "wav": buffered 16-bit WAV (Content-Type audio/wav).
//   GET /v1/audio/voices    speakers + languages from the package's config tables,
//                           plus the sealed package voice if present.
//   GET /v1/models          single-entry model listing.
//
// A prepared `voice.json` supplies per-field defaults; request fields override it.
// Generation is strictly serial (the serve loop awaits each request): the runtime
// backend is touched only by the producer thread per request.

final class SmeltTextToPCMServeHandler: @unchecked Sendable {
    private let runtime: CAMTextToPCMServeRuntime
    private let modelId: String
    private let packageIdentity: String

    /// Frame cap ceiling (~164s of 24 kHz audio): bounds per-request KV/RoPE allocation.
    private static let maxFramesCeiling = 2048

    init(
        runtime: CAMTextToPCMServeRuntime,
        modelId: String,
        packageIdentity: String
    ) {
        self.runtime = runtime
        self.modelId = modelId
        self.packageIdentity = packageIdentity
    }

    func handle(
        _ raw: SmeltServeRawRequest,
        transport: any SmeltServeTransport
    ) async -> SmeltServeHandler.HandlerResult {
        switch (raw.method, raw.path) {
        case (.post, .audioSpeech):
            return await handleSpeech(raw.body, requestId: raw.id, transport: transport)
        case (.get, .audioVoices):
            return .complete(handleVoices())
        case (.get, .models):
            let now = Int(Date().timeIntervalSince1970)
            let body = try! OpenAIJSON.encode(OpenAIModelsResponse(
                object: "list",
                data: [OpenAIModelEntry(id: modelId, object: "model", created: now, ownedBy: "smelt")]
            ))
            return .complete(SmeltServeRawResponse(
                statusCode: 200,
                headers: ["X-Smelt-Package-Identity": packageIdentity],
                body: body
            ))
        case (_, .chatCompletions), (_, .completions):
            return .complete(OpenAIJSON.errorResponse(
                status: 400, code: .invalidRequest,
                message: "\(raw.path.rawValue) requires a text generation package; this server is "
                    + "serving the text-to-PCM package '\(modelId)' — use /v1/audio/speech"
            ))
        case (.get, .audioSpeech), (.post, .audioVoices), (.post, .models):
            return .complete(OpenAIJSON.errorResponse(
                status: 405, code: .methodNotAllowed,
                message: "Method not allowed for \(raw.path.rawValue)"
            ))
        }
    }

    // MARK: - /v1/audio/speech

    private struct SpeechRequest: Decodable {
        let input: String
        let model: String?            // accepted for OpenAI compat; single-model server
        let voice: String?            // named speaker (spk_id table)
        let language: String?
        let instruct: String?
        let responseFormat: String?   // "pcm" (default) | "wav"
        let seed: UInt64?
        let greedy: Bool?
        let maxFrames: Int?
        let firstChunkFrames: Int?
        let maxChunkFrames: Int?

        enum CodingKeys: String, CodingKey {
            case input, model, voice, language, instruct, seed, greedy
            case responseFormat = "response_format"
            case maxFrames = "max_frames"
            case firstChunkFrames = "first_chunk_frames"
            case maxChunkFrames = "max_chunk_frames"
        }
    }

    private func handleSpeech(
        _ body: Data,
        requestId: SmeltServeRequestId,
        transport: any SmeltServeTransport
    ) async -> SmeltServeHandler.HandlerResult {
        let req: SpeechRequest
        do {
            req = try JSONDecoder().decode(SpeechRequest.self, from: body)
        } catch {
            return .complete(OpenAIJSON.errorResponse(
                status: 400, code: .invalidRequest, message: "Bad request body: \(error)"))
        }
        guard !req.input.isEmpty else {
            return .complete(OpenAIJSON.errorResponse(
                status: 400, code: .invalidRequest, message: "'input' must be non-empty"))
        }
        let format = req.responseFormat ?? "pcm"
        guard format == "pcm" || format == "wav" else {
            return .complete(OpenAIJSON.errorResponse(
                status: 400, code: .invalidRequest,
                message: "response_format '\(format)' unsupported (pcm | wav)"))
        }

        let voice = runtime.voiceDefaults
        // Effective voice: request field ?? prepared voice.json ?? declared
        // package default (manifest loop schedule — B2.2's one truth)
        // ?? built-in.
        let speaker = req.voice ?? voice?.speaker
        let language = req.language ?? voice?.language ?? "Auto"
        let instruct = req.instruct ?? voice?.instruct
        let declaredChunks = runtime.declaredChunkSchedule
        let firstChunk = req.firstChunkFrames ?? voice?.firstChunkFrames
            ?? declaredChunks?.first ?? 1
        let maxChunk = req.maxChunkFrames ?? voice?.maxChunkFrames
            ?? max(firstChunk, declaredChunks?.max ?? 1)
        let maxFrames = min(
            req.maxFrames ?? voice?.maxFrames ?? 256,
            Self.maxFramesCeiling
        )
        guard firstChunk >= 1,
              maxChunk >= firstChunk,
              maxFrames >= 1
        else {
            return .complete(OpenAIJSON.errorResponse(
                status: 400,
                code: .invalidRequest,
                message: "need 1 <= first_chunk_frames <= max_chunk_frames and max_frames >= 1"
            ))
        }
        let decode: CAMTextToPCMDecodeMode
        if req.greedy == true {
            decode = .greedy
        } else if let seed = req.seed {
            decode = .sampleSeeded(seed)
        } else {
            decode = .packageDefault
        }

        // Blocking generateStreaming runs on its own dedicated thread; chunks cross to
        // this async context through the stream. JOIN CONTRACT: every path below
        // consumes the stream to completion before returning, so the producer thread is
        // past all backend access when the serial serve loop picks up the next
        // request (the single-threaded-driver contract). On consumer death (client
        // disconnected — barge-in) `cancel` flips, the producer's next onChunk return
        // stops the model mid-utterance, and the loop keeps draining until finish.
        let cancel = CancelFlag()
        let runtime = self.runtime
        let generationRequest = CAMTextToPCMServeRequest(
            text: req.input,
            speaker: speaker,
            language: language,
            instruct: instruct,
            maxFrames: maxFrames,
            firstChunkFrames: firstChunk,
            maxChunkFrames: maxChunk,
            decode: decode,
            seed: req.seed
        )
        func chunkStream() -> AsyncThrowingStream<[Float], Error> {
            AsyncThrowingStream { continuation in
                Thread.detachNewThread {
                    do {
                        try runtime.generateStreaming(request: generationRequest) { samples in
                            if cancel.isSet { return false }
                            if !samples.isEmpty { continuation.yield(samples) }
                            return true
                        }
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
            }
        }

        if format == "wav" {
            var samples: [Float] = []
            do {
                for try await s in chunkStream() { samples += s }   // runs to finish = joined
            } catch {
                return .complete(OpenAIJSON.errorResponse(
                    status: 500, code: .internalError, message: "synthesis failed: \(error)"))
            }
            return .complete(SmeltServeRawResponse(
                statusCode: 200,
                headers: ["Content-Type": "audio/wav"],
                body: SmeltAudioOutput.wavData(
                    channelMajor: samples,
                    sampleRate: runtime.sampleRate,
                    channels: runtime.audioChannels
                )))
        }

        // pcm: stream chunks as they land. The producer is spawned only after the
        // response stream is open (a beginStream failure must not leave an unjoined
        // generation running). Errors after the first byte can't change the status;
        // log and end the stream (the truncation is the client's error signal).
        let stream: SmeltServeStreamHandle
        do {
            stream = try await transport.beginStream(
                contentType: "audio/pcm",
                requestId: requestId,
                extraHeaders: [
                    "X-Sample-Rate": "\(runtime.sampleRate)",
                    "X-Audio-Format": "s16le",
                    "X-Channels": "\(runtime.audioChannels)",
                ])
        } catch {
            return .complete(OpenAIJSON.errorResponse(
                status: 500, code: .internalError, message: "stream start failed: \(error)"))
        }
        do {
            var consumerAlive = true
            for try await samples in chunkStream() {
                guard consumerAlive else { continue }   // draining to join the producer
                do {
                    try await stream.writeChunk(SmeltAudioOutput.pcm16Data(
                        channelMajor: samples,
                        sampleRate: runtime.sampleRate,
                        channels: runtime.audioChannels
                    ))
                } catch {
                    cancel.set()   // client gone: stop the model (barge-in), keep draining
                    consumerAlive = false
                }
            }
        } catch {
            fputs("smelt serve: text-to-PCM synthesis failed mid-stream: \(error)\n", stderr)
        }
        try? await stream.end()
        return .streamed
    }

    // MARK: - /v1/audio/voices

    private func handleVoices() -> SmeltServeRawResponse {
        struct VoiceDefaults: Encodable {
            let speaker: String?
            let language: String?
            let instruct: String?
            let firstChunkFrames: Int?
            let maxChunkFrames: Int?
            let maxFrames: Int?
            enum CodingKeys: String, CodingKey {
                case speaker, language, instruct
                case firstChunkFrames = "first_chunk_frames"
                case maxChunkFrames = "max_chunk_frames"
                case maxFrames = "max_frames"
            }
        }
        struct VoicesResponse: Encodable {
            let model: String
            let voices: [String]
            let languages: [String]
            let defaults: VoiceDefaults?
        }
        let catalog = runtime.voiceCatalog
        let response = VoicesResponse(
            model: modelId,
            voices: catalog.speakers,
            languages: catalog.languages,
            defaults: runtime.voiceDefaults.map {
                VoiceDefaults(speaker: $0.speaker, language: $0.language, instruct: $0.instruct,
                           firstChunkFrames: $0.firstChunkFrames, maxChunkFrames: $0.maxChunkFrames,
                           maxFrames: $0.maxFrames)
            })
        let body = try! OpenAIJSON.encode(response)
        return SmeltServeRawResponse(statusCode: 200, body: body)
    }
}

/// Thread-safe one-way cancellation latch shared by the stream consumer (sets it on
/// client disconnect) and the generation producer thread (polls it per chunk).
final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    func set() { lock.lock(); value = true; lock.unlock() }
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }
}
