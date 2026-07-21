import AVFoundation
import Darwin
import Foundation
import SmeltRuntime
import SmeltSchema

// smelt run on a CAM text-to-PCM package. Audio is written to stdout as WAV:
//   smelt run voice.smeltpkg "text" > out.wav  file: exact WAV (sizes patched)
//   smelt run voice.smeltpkg "text" | ffplay - pipe: streaming WAV, chunks flush
//                                                    as generated (first audio ~80ms warm)
//   echo "text" | smelt run voice.smeltpkg      stdin is the text source
// Package defaults may be overridden by explicit flags. Stats go to stderr.

func runCAMTextToPCMRunCommand(
    packagePath: String,
    construction: CAMTextToPCMRuntimeConstruction,
    promptStartIndex: Int,
    fullArgv: [String],
    camIdentity: LingerCAMIdentity? = nil
) {
    let usage = [
        "Usage: smelt run <voice.smeltpkg> \"text\"\n",
        "       [--speaker NAME] [--language L] [--instruct TEXT]\n",
        "       [--seed N | --greedy] [--max-frames N] [--first-chunk N] [--max-chunk N]\n",
        "       [--linger SECONDS]\n",
        "  audio goes to stdout as WAV\n",
        "  --linger N      leave a warm runtime behind for N idle seconds; repeat\n",
        "                  invocations skip the model load (warm first audio)\n",
        "  --seed N        deterministic sampling; --greedy disables sampling (debug)\n",
        "  --first/max-chunk  streaming schedule in 80ms frames (voice defaults, else 1/1\n",
        "                     = minimum-latency zero-buffer; use 4/4 for long-form throughput)\n",
        "  --help          this text plus the package's declared flags; `--` ends flags\n"
    ]
    // Strict argv: every --flag must be a built-in or declared by the
    // package's args.json; a typo'd flag is an error, not speech text.
    let interface = loadPackageInterface(
        packagePath: packagePath,
        graphPolicy: .sidecarTextToCodecAudio
    )
    let scanned: SmeltPackageInterface.Scan
    do {
        scanned = try SmeltPackageInterface.scan(
            argv: fullArgv,
            startIndex: promptStartIndex,
            builtinsWithValues: RunFlags.textToPCMValue,
            builtinBools: RunFlags.textToPCMBool,
            declared: interface
        )
    } catch {
        fputs("smelt run: \(error)\n", stderr)
        for line in usage { fputs(line, stderr) }
        exit(1)
    }
    if scanned.helpRequested {
        for line in usage { fputs(line, stderr) }
        for line in interface?.helpLines() ?? [] { fputs(line + "\n", stderr) }
        exit(0)
    }
    let declaredFlags = scopedDeclaredRunFlags(interface)
    let scopedFlags = ScopedFlagReader(
        argv: fullArgv,
        startIndex: promptStartIndex,
        terminatorIndex: scanned.terminatorIndex,
        valueFlags: RunFlags.textToPCMValue.union(declaredFlags.value),
        boolFlags: RunFlags.textToPCMBool.union(declaredFlags.bool)
    )

    let promptFlag = scopedFlags.value("--prompt")
    let argText = promptFlag.isEmpty
        ? scanned.positionals.joined(separator: " ") : promptFlag
    let stdinIsTTY = isatty(FileHandle.standardInput.fileDescriptor) == 1
    let pipedText = argText.isEmpty && !stdinIsTTY ? readPromptFromStdin() : nil
    let text = argText.isEmpty ? pipedText ?? "" : argText
    guard !text.isEmpty else {
        for line in usage { fputs(line, stderr) }
        exit(1)
    }

    // Resolve every parameter before the (multi-second) model load: a bad
    // flag value fails in milliseconds.
    let speaker: String?
    let language: String
    let instruct: String?
    let firstChunk: Int
    let maxChunk: Int
    let maxFrames: Int
    do {
        let voice = try construction.load24KVoiceDefaults()
        // One precedence chain for every voice field:
        //   explicit flag (built-in XOR declared — both is a conflict)
        //   > prepared voice.json > args.json default > built-in default.
        let merge = CAMTextToPCMArgMerge(
            interface: interface,
            scanned: scanned,
            resolved: try interface?.resolve(declaredRaw: scanned.declaredRaw) ?? [:]
        )
        speaker = try merge.string("speaker", "--speaker", voice?.speaker)
        language = try merge.string("language", "--language", voice?.language) ?? "Auto"
        instruct = try merge.string("instruct", "--instruct", voice?.instruct)
        // Strict numeric flags: a malformed or non-positive value is a CLI error, not a
        // silent fall-through to the default (which would later abort on a runtime
        // precondition). The merged schedule is re-checked below because a hand-edited
        // voice.json or args.json can carry invalid values from external tooling.
        // Chunk-schedule defaults come from the manifest's declared loop
        // (one truth — B2.2); pre-stamping packages keep the old 1/1.
        let declaredChunks = try construction.declared24KChunkSchedule()
        firstChunk = try merge.int("first-chunk", "--first-chunk", voice?.firstChunkFrames)
            ?? declaredChunks?.first ?? 1
        // The max fallback couples to the RESOLVED first (an explicit
        // --first-chunk 4 must not collide with a declared max of 1).
        maxChunk = try merge.int("max-chunk", "--max-chunk", voice?.maxChunkFrames)
            ?? max(firstChunk, declaredChunks?.max ?? 1)
        maxFrames = try merge.int("max-frames", "--max-frames", voice?.maxFrames) ?? 256
        guard firstChunk >= 1, maxChunk >= firstChunk, maxFrames >= 1 else {
            fputs("smelt run: need 1 <= first-chunk (\(firstChunk)) <= max-chunk (\(maxChunk)) and max-frames (\(maxFrames)) >= 1 (check voice.json/args.json)\n", stderr)
            exit(1)
        }
        // Resolve the names against the package's tables now — an unknown
        // speaker must not wait for generateStreaming to throw after the
        // multi-second model load.
        try construction.validate24KVoice(language: language, speaker: speaker)
    } catch {
        fputs("smelt run: \(error)\n", stderr)
        exit(1)
    }
    let decode: CAMTextToPCMDecodeMode
    let seedRaw = scopedFlags.value("--seed")
    var seedValue: UInt64?
    if scopedFlags.has("--greedy") {
        decode = .greedy
    } else if !seedRaw.isEmpty {
        guard let seed = UInt64(seedRaw) else {
            fputs("smelt run: --seed must be an unsigned integer, got '\(seedRaw)'\n", stderr)
            exit(1)
        }
        seedValue = seed
        decode = .sampleSeeded(seed)
    } else {
        decode = .packageDefault
    }
    let lingerSeconds = scopedFlags.nonNegativeIntOrExit("--linger", verb: "run") ?? 0

    do {
        let sink = try CAMTextToPCMRunOutputSink()

        // --linger N: same warm-runtime semantics as text generation: forward
        // to a listening runtime (warm TTFA, no model load), else run inline
        // and leave a warm runtime behind for N idle seconds.
        if lingerSeconds > 0 {
            let socketPath = camTextToPCMWarmSocketPath(
                packagePath: packagePath,
                camIdentity: camIdentity
            )
            let request = CAMTextToPCMWarmRequest(
                text: text, speaker: speaker, language: language,
                instruct: instruct, maxFrames: maxFrames,
                firstChunkFrames: firstChunk, maxChunkFrames: maxChunk,
                greedy: scopedFlags.has("--greedy"), seed: seedValue
            )
            // Warm-path TTFA: time from here to the runtime's first audio. markStart is
            // last-call-wins (resets t0), so the cold fallback below re-marks AFTER its
            // gpu load + trunk prewarm — neither is counted in either path's TTFA.
            sink.markStart()
            if tryCAMTextToPCMWarmForward(
                socketPath: socketPath, request: request,
                onSamples: { sink.consume($0) }
            ) {
                sink.finish(speaker: speaker)
                fputs("Linger: served by warm runtime\n", stderr)
                return
            }
        }

        let runtime = try construction.make24KRuntime(verb: "run")
        // Compile the trunk (bf16 packages) before the TTFA timer starts, not inside it.
        try runtime.prewarmCompiledTrunk()
        sink.markStart()
        try runtime.generateStreaming(
            text: text, instruct: instruct, language: language, speaker: speaker,
            maxFrames: maxFrames, decode: decode,
            firstChunkFrames: firstChunk, maxChunkFrames: maxChunk
        ) { chunk in
            sink.consume(chunk.samples)
            return true
        }
        sink.finish(speaker: speaker)

        if lingerSeconds > 0 {
            spawnLingerWorker(
                packagePath: packagePath,
                socketPath: camTextToPCMWarmSocketPath(
                    packagePath: packagePath,
                    camIdentity: camIdentity
                ),
                idleSeconds: lingerSeconds,
                contextLimit: nil,
                camIdentity: camIdentity
            )
        }
    } catch {
        fputs("text-to-pcm run failed: \(error)\n", stderr)
        exit(1)
    }
}

/// Output routing for synthesized audio, shared by inline runs and linger
/// forwards. WAV is always streamed to stdout so `smelt run` has one stable
/// output contract independent of TTY state. The header is written lazily at
/// first audio, so a failed run leaves an empty redirected file.
final class CAMTextToPCMRunOutputSink {
    private let stdout = FileHandle.standardOutput
    private var headerWritten = false
    private var wavHeaderStart: off_t = -1
    private var pcmBytes = 0
    private var sampleCount = 0
    private var ttfa: Double = 0
    private var t0 = CFAbsoluteTimeGetCurrent()

    init() throws {
    }

    /// Reset the clock to the start of the attempt. Inline runs call this
    /// after the model load (TTFA = synthesis only, as before); the linger
    /// path calls it before connecting, so a forwarded TTFA includes socket
    /// + any wait on a still-loading worker — the user-perceived latency.
    func markStart() { t0 = CFAbsoluteTimeGetCurrent() }

    func consume(_ samples: [Float]) {
        guard !samples.isEmpty else { return }
        if ttfa == 0 { ttfa = CFAbsoluteTimeGetCurrent() - t0 }
        if !headerWritten {
            // Header start for the post-stream size patch, captured from
            // the position AFTER the header write: with `>>` (O_APPEND)
            // the header lands at the old EOF, not 0, and the write
            // itself moves the offset there. -1 (pipe, unseekable) = no
            // patch.
            stdout.write(CAMTextToPCM24KAudio.wavHeader(pcmBytes: nil))
            let posAfterHeader = lseek(STDOUT_FILENO, 0, SEEK_CUR)
            if posAfterHeader >= 44 { wavHeaderStart = posAfterHeader - 44 }
            headerWritten = true
        }
        let pcm = CAMTextToPCM24KAudio.pcm16(samples)
        stdout.write(pcm)
        pcmBytes += pcm.count
        sampleCount += samples.count
    }

    func finish(speaker: String?) {
        let wall = CFAbsoluteTimeGetCurrent() - t0
        // Seekable stdout (a redirected file): patch the exact sizes into
        // the streamed header, at wherever the header actually landed. A
        // pipe keeps the read-until-EOF streaming header.
        if wavHeaderStart >= 0 {
            for patch in CAMTextToPCM24KAudio.wavSizePatches(pcmBytes: pcmBytes) {
                var v = patch.value.littleEndian
                _ = withUnsafeBytes(of: &v) {
                    pwrite(STDOUT_FILENO, $0.baseAddress, 4, wavHeaderStart + off_t(patch.offset))
                }
            }
        }
        let audioSeconds = Double(sampleCount) / Double(CAMTextToPCM24KAudio.sampleRate)
        var status = String(
            format: "text-to-pcm: first audio %.0fms, wall %.2fs, %.2fs audio", ttfa * 1000, wall, audioSeconds)
        if let speaker { status += " (speaker \(speaker))" }
        fputs(status + "\n", stderr)
    }
}

/// Per-target merge of explicit flags (built-in or declared), prepared
/// voice, and declared defaults, with conflicts rejected at equal precedence.
struct CAMTextToPCMArgMerge {
    let interface: SmeltPackageInterface?
    let scanned: SmeltPackageInterface.Scan
    let resolved: [String: SmeltPackageArgumentValue]

    struct MergeError: Error, CustomStringConvertible {
        let description: String
    }

    private func declaredArg(_ target: String) -> SmeltPackageInterface.Arg? {
        interface?.args.first { $0.target == target }
    }

    /// The declared arg's value split by provenance: explicitly passed on
    /// the command line vs filled from its args.json default.
    private func declaredValues(
        _ target: String
    ) -> (explicit: SmeltPackageArgumentValue?, fallback: SmeltPackageArgumentValue?) {
        guard let arg = declaredArg(target), let value = resolved[arg.flag] else {
            return (nil, nil)
        }
        return scanned.declaredRaw[arg.flag] != nil ? (value, nil) : (nil, value)
    }

    private func rejectConflict(
        _ target: String, _ builtinFlag: String,
        builtinExplicit: Bool, declaredExplicit: Bool
    ) throws {
        if builtinExplicit, declaredExplicit, let arg = declaredArg(target) {
            throw MergeError(
                description: "--\(arg.flag) and \(builtinFlag) both set \(target)"
            )
        }
    }

    func string(
        _ target: String, _ builtinFlag: String, _ prepared: String?
    ) throws -> String? {
        let builtin = parseArg(builtinFlag).nilIfEmpty
        let (explicit, fallback) = declaredValues(target)
        try rejectConflict(
            target, builtinFlag,
            builtinExplicit: builtin != nil, declaredExplicit: explicit != nil
        )
        if let builtin { return builtin }
        if case .string(let s)? = explicit { return s }
        if let prepared { return prepared }
        if case .string(let s)? = fallback { return s }
        return nil
    }

    func int(
        _ target: String, _ builtinFlag: String, _ prepared: Int?
    ) throws -> Int? {
        let raw = parseArg(builtinFlag)
        var builtin: Int?
        if !raw.isEmpty {
            guard let v = Int(raw), v > 0 else {
                throw MergeError(
                    description: "\(builtinFlag) must be a positive integer, got '\(raw)'"
                )
            }
            builtin = v
        }
        let (explicit, fallback) = declaredValues(target)
        try rejectConflict(
            target, builtinFlag,
            builtinExplicit: builtin != nil, declaredExplicit: explicit != nil
        )
        if let builtin { return builtin }
        if case .int(let v)? = explicit { return v }
        if let prepared { return prepared }
        if case .int(let v)? = fallback { return v }
        return nil
    }
}

func dispatchCAMTextToPCMRunHandlerOrExit(
    packagePath: String,
    construction: CAMTextToPCMRuntimeConstruction,
    promptStartIndex: Int,
    fullArgv: [String],
    camIdentity: LingerCAMIdentity? = nil
) -> Never {
    do {
        try construction.requirePackagePath(packagePath)
    } catch {
        fputs("smelt run: \(error)\n", stderr)
        exit(1)
    }
    runCAMTextToPCMRunCommand(
        packagePath: packagePath,
        construction: construction,
        promptStartIndex: promptStartIndex,
        fullArgv: fullArgv,
        camIdentity: camIdentity
    )
    exit(0)
}

/// Streams Float32 24 kHz mono chunks to the default output device as they arrive.
/// AVAudioPlayerNode consumes scheduled buffers gaplessly, so feeding it each chunk
/// the moment generateStreaming emits it plays with no jitter buffer — the engine
/// IS the consumer the zero-buffer schedules are designed for.
final class CAMTextToPCMLivePlayer {
    private let player: SmeltAudioPlayer

    init() throws {
        player = try SmeltAudioPlayer(
            sampleRate: CAMTextToPCM24KAudio.sampleRate, channels: 1)
    }

    /// Text-to-PCM output is mono, so channel-major == the sample order.
    func enqueue(_ samples: [Float]) {
        guard !samples.isEmpty else { return }
        player.enqueue(channelMajor: samples)
    }

    /// Block until every scheduled buffer has played out, then stop the engine.
    func finish() {
        player.finish()
    }
}
