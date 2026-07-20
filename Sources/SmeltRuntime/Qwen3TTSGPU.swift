// Qwen3TTSGPU — GPU runtime driver for a hand-built Qwen3-TTS `.smeltpkg`.
//
// Loads the package itself rather than going through SmeltRuntime.init (which
// builds one monolithic weights MTLBuffer — a 7 GB single buffer can exceed
// device.maxBufferLength). Instead it mmaps weights.bin ONCE and wraps each
// weight as an MTLBuffer(bytesNoCopy:) at its page-aligned offset; the owning
// mmap stays alive for the driver's lifetime and is released in deinit.
//
// U4a is this loader + accessors; the codec / talker / MTP dispatch graph
// (graduated from the GPU test harness) lands in later sub-units.

import Foundation
import Metal
import SmeltSchema

public final class Qwen3TTSGPU {

    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let manifest: Qwen3TTSManifest
    public let packagePath: String
    /// Loaded from the package's bundled HF files when present, so `generate(text:…)` is a
    /// self-contained text→24 kHz entry point. nil for a weights-only package.
    public let tokenizer: SmeltTokenizer?
    public let frontEndConfig: Qwen3TTSFrontEnd.Config?
    /// The prepared voice defaults (`voice.json`): package defaults
    /// for speaker / language / instruct / streaming schedule. nil when nothing was baked.
    public let voice: Qwen3TTSVoice?

    private let pipelinesByName: [String: MTLComputePipelineState]
    private let weightBuffers: [String: MTLBuffer]
    private let weightEntries: [String: Qwen3TTSManifest.Entry]
    private let mmapBase: UnsafeMutableRawPointer
    private let mmapLength: Int
    /// Set inside a `batched { }` scope so dispatch helpers append to one command buffer (one
    /// commit + sync per forward) instead of committing each kernel. Single-threaded driver loop.
    var batchCmd: MTLCommandBuffer?
    var batchEnc: MTLComputeCommandEncoder?
    /// CPU-derived codec-stream constants (RVQ codebook embeddings, output_proj concat, RoPE
    /// tables), built by the first Qwen3TTSCodecStream init and reused by every later stream —
    /// they cost ~20ms to derive and are pure functions of the package weights. Same
    /// single-threaded-driver contract as the batched state above.
    var codecStreamShared: Qwen3TTSCodecStream.Shared?
    /// Widened-to-f32 codec-decoder weights for the COMPILED streaming codec, keyed by checkpoint
    /// name, materialized ONCE on first use and reused across chunks + streams. A u4 build stores the
    /// rank≥2 `decoder.*` conv/transformer weights as bf16 (Qwen3TTSPackageBuilder: "read back via
    /// f32(), widened"); the stream's f32 record kernels need f32 bytes, so the realizer widens the
    /// non-f32 ones here (f32 weights bind raw via wbuf — no copy). Same single-threaded-driver contract.
    var codecStreamWidened: [String: MTLBuffer] = [:]
    /// Memoized talkerShape() — manifest shapes and the bundled config.json are immutable for
    /// the life of the loaded package; the uncached path re-read + re-parsed config.json from
    /// disk on every inputsEmbeds/generateCodes call (a fixed TTFA tax). Same single-threaded-
    /// driver contract as the batched state above.
    var talkerShapeCache: TalkerShape?
    /// Memoized text_projection fc weights as [Float] (BLAS inputs for the CPU front-end) and
    /// the projected tts_pad row keyed by token id — pure functions of the package weights;
    /// uncached, f32() re-copied the full matrices on every inputsEmbeds/generateCodes call.
    var textProjWeightsCache: (fc1W: [Float], fc1B: [Float], fc2W: [Float], fc2B: [Float])?
    var ttsPadEmbedCache: [Int: [Float]] = [:]
    /// Per-batched-scope output-buffer recycling. Within one `batched { }` every dispatch's output
    /// must be a distinct live buffer (the command buffer hasn't run yet), so `checkedOutBufs` holds
    /// this scope's buffers; at the NEXT scope's start they return to `bufferPool` (keyed by element
    /// count) for reuse — the prior command buffer has completed and its readback is done by then.
    /// Eliminates the per-dispatch `makeBuffer` allocation + first-touch page faults across frames.
    var bufferPool: [Int: [MTLBuffer]] = [:]
    var checkedOutBufs: [MTLBuffer] = []

    public init(packagePath: String, device: MTLDevice? = nil) throws {
        guard let dev = device ?? MTLCreateSystemDefaultDevice() else {
            throw LoadError.noMetalDevice
        }
        self.device = dev
        guard let q = dev.makeCommandQueue() else { throw LoadError.noCommandQueue }
        self.queue = q
        self.packagePath = packagePath

        let manifestData = try Data(
            contentsOf: URL(fileURLWithPath: "\(packagePath)/manifest.json"))
        let manifest = try Qwen3TTSManifest.decode(from: manifestData)
        self.manifest = manifest

        guard manifest.version == 1 else { throw LoadError.unsupportedVersion(manifest.version) }
        guard !manifest.eosTokens.isEmpty else { throw LoadError.noEosTokens }
        do {
            try manifest.validateQwen3TTSValidation(packagePath: packagePath)
        } catch {
            throw LoadError.validationInvalid(String(describing: error))
        }
        // bytesNoCopy needs each weight pointer aligned to THIS host's page size. Offsets
        // are aligned to manifest.pageSize, so the host page must divide it (e.g. a 4 KB
        // package can't be loaded on a 16 KB-page host).
        let hostPage = Int(getpagesize())
        guard manifest.pageSize > 0, manifest.pageSize % hostPage == 0 else {
            throw LoadError.pageSizeMismatch(manifest: manifest.pageSize, host: hostPage)
        }

        // Bake honesty for the TTS root package. Qwen3TTSGPU opens the package
        // directly (not via SmeltRuntime.init), and its inner SmeltRuntime opens
        // only the trunk sidecar sub-path — so the root baked.json (declaring
        // voice/args) would otherwise be unenforced.
        try SmeltBakeManifest.enforce(
            packagePath: packagePath, ignoring: SmeltBakeManifest.ignoredFromEnv())

        // A bundled tokenizer makes this a text→24 kHz package; load it from the package dir.
        // The four files are copied in by Qwen3TTSPackageBuilder; require the full set or none.
        if let files = manifest.tokenizerFiles, !files.isEmpty {
            guard Set(Qwen3TTSManifest.requiredTokenizerFiles).isSubset(of: Set(files)) else {
                throw LoadError.tokenizerIncomplete(files)
            }
            self.tokenizer = try SmeltTokenizer(
                qwenVocabJSONPath: "\(packagePath)/vocab.json",
                mergesTxtPath: "\(packagePath)/merges.txt",
                tokenizerConfigPath: "\(packagePath)/tokenizer_config.json")
            self.frontEndConfig = try Qwen3TTSFrontEnd.Config.load(
                configJSONPath: "\(packagePath)/config.json")
        } else {
            self.tokenizer = nil
            self.frontEndConfig = nil
        }
        self.voice = try Qwen3TTSVoice.load(packagePath: packagePath)

        // Pipeline states from the package metallib, looked up by name. Exact
        // presence — a missing function is a hard error, not a placeholder (the
        // monolithic runtime would silently substitute one).
        let lib = try dev.makeLibrary(
            URL: URL(fileURLWithPath: "\(packagePath)/model.metallib"))
        var psos: [String: MTLComputePipelineState] = [:]
        for fn in manifest.pipelines {
            guard let f = lib.makeFunction(name: fn) else {
                throw LoadError.missingFunction(fn)
            }
            // Label = function name so per-kernel profiling (SMELT_DECODE_PROFILE) can attribute
            // dispatch GPU time by kernel.
            let desc = MTLComputePipelineDescriptor()
            desc.computeFunction = f
            desc.label = fn
            psos[fn] = try dev.makeComputePipelineState(descriptor: desc, options: [], reflection: nil)
        }
        self.pipelinesByName = psos

        // mmap weights.bin once.
        let weightsPath = "\(packagePath)/weights.bin"
        let fd = open(weightsPath, O_RDONLY)
        guard fd >= 0 else { throw LoadError.weightsOpenFailed(weightsPath) }
        defer { close(fd) }
        // Validate before the trapping UInt64→Int conversion: a malformed package must
        // throw, not crash.
        guard manifest.totalBytes <= UInt64(Int.max) else {
            throw LoadError.weightsSizeMismatch(weightsPath)
        }
        let size = Int(manifest.totalBytes)
        let fileSize = lseek(fd, 0, SEEK_END)
        lseek(fd, 0, SEEK_SET)
        guard size > 0, fileSize >= off_t(size) else {
            throw LoadError.weightsSizeMismatch(weightsPath)
        }
        guard let base = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0),
              base != MAP_FAILED else {
            throw LoadError.weightsMmapFailed(weightsPath)
        }
        madvise(base, size, MADV_WILLNEED)   // first-touch prefault, as the monolithic loader does
        self.mmapBase = base
        self.mmapLength = size

        // One bytesNoCopy buffer per weight at its page-aligned offset. The pointer
        // and length are page-aligned (guaranteed by the builder), as bytesNoCopy
        // requires. No-op deallocator: the single mmap is freed in deinit, not per slice.
        // These wrap PROT_READ pages, so every weight is a read-only kernel INPUT —
        // a kernel that binds one as an output buffer would SIGBUS.
        var bufs: [String: MTLBuffer] = [:]
        var entries: [String: Qwen3TTSManifest.Entry] = [:]
        bufs.reserveCapacity(manifest.weights.count)
        entries.reserveCapacity(manifest.weights.count)
        let ps = UInt64(manifest.pageSize)
        for e in manifest.weights {
            // Each slice must honour the builder's contract: page-aligned, and within the
            // mapping. Otherwise bytesNoCopy mis-aligns (host-page dependent) or the first
            // kernel read SIGBUSes — catch it cleanly at load. Bounds test is written to
            // avoid UInt64 overflow on a corrupt offset+byteLength.
            guard e.offset % ps == 0, e.byteLength % ps == 0 else {
                munmap(base, size)
                throw LoadError.weightMisaligned(e.name)
            }
            guard e.offset <= manifest.totalBytes,
                  e.byteLength <= manifest.totalBytes - e.offset else {
                munmap(base, size)
                throw LoadError.weightOutOfBounds(e.name)
            }
            let ptr = base.advanced(by: Int(e.offset))
            guard let buf = dev.makeBuffer(
                bytesNoCopy: ptr, length: Int(e.byteLength),
                options: .storageModeShared, deallocator: nil) else {
                munmap(base, size)
                throw LoadError.bufferCreateFailed(e.name)
            }
            bufs[e.name] = buf
            entries[e.name] = e
        }
        self.weightBuffers = bufs
        self.weightEntries = entries
    }

    // The per-weight buffers alias this one mmap (a no-op deallocator each — munmap-per-slice
    // would double-free), so the mapping is released here, once, when the driver is released.
    // Driver dispatch methods are synchronous (waitUntilCompleted), so no command buffer
    // referencing a weight buffer is in flight across a public-method boundary; don't release
    // the driver mid-dispatch.
    deinit { munmap(mmapBase, mmapLength) }

    public func pipeline(_ name: String) -> MTLComputePipelineState? { pipelinesByName[name] }
    public func weight(_ name: String) -> MTLBuffer? { weightBuffers[name] }
    public func weightShape(_ name: String) -> [Int]? { weightEntries[name]?.shape }
    /// Packed weight element type. f16/bf16 both store 2 bytes and widen to fp32 in-kernel; bf16 is the
    /// source checkpoint's native dtype (bit-exact), f16 is a lossy narrowing; u4 is group-wise affine
    /// int4 (packed nibbles + per-group fp16 scale/bias, all in one block — see weightU4).
    public enum WeightPackDType { case f32, f16, bf16, u4 }

    /// Exhaustive dtype lookup (CLAUDE.md dispatch-safety: no silent dtype fallthrough).
    public func weightDType(_ name: String) -> WeightPackDType {
        switch weightEntries[name]?.dtype {
        case nil, "f32": return .f32
        case "f16": return .f16
        case "bf16": return .bf16
        case "u4": return .u4
        case let other: preconditionFailure("weight \(name) unknown dtype \(other ?? "nil")")
        }
    }

    /// The u4 block's scale/bias byte offsets (RELATIVE to the weight buffer's start) + column group
    /// size — for binding the single mmap'd block buffer at three offsets in the gemv_u4/gemm_u4 kernels.
    public func weightU4(_ name: String) -> (scaleOffset: Int, biasOffset: Int, groupSize: Int)? {
        guard let e = weightEntries[name], e.dtype == "u4",
              let s = e.scaleOffset, let b = e.biasOffset, let g = e.groupSize else { return nil }
        return (Int(s), Int(b), g)
    }

    // MARK: - Compiled trunk (B3.2c)

    // The compiled fp32 talker trunk, loaded ONCE from the package's trunk/ sidecar (bf16
    // builds carry it) and reused across SEQUENTIAL requests. Text-to-PCM serving is serial by
    // contract (SmeltTextToPCMServeHandler / SmeltServe / CAMTextToPCMWarmRuntime each
    // complete one request before the next), so one long-lived trunk runtime is safe;
    // concurrent serving would need a mutex/actor or per-request KV isolation.
    // Lazy + memoized so calibration/profiling gpus (willRunBatched == false ⇒
    // the session never resolves it) and codec-only /
    // f32/f16 packages don't pay the ~400-pipeline compile (a u4 build DOES carry the
    // trunk now — Phase 3 — and resolves it like bf16).
    // Cache of resolved co-resident trunk sidecars, keyed by dir name ("trunk" / "trunk-mtp").
    // The double-optional encodes three states: absent key = unresolved; .some(nil) = resolved
    // and the dir doesn't exist (settled — never a trunk); .some(rt) = loaded. A load THROW is
    // NOT cached, so the next call re-attempts and re-throws rather than silently degrading: a
    // package advertising a compiled block with a broken sidecar is corrupt, not a
    // fallback. Loaded once and reused across SEQUENTIAL requests (TTS serving is serial by
    // contract — see resolveCompiledSidecar callers).
    private var compiledSidecars: [String: SmeltRuntime?] = [:]

    /// Resolve (load-once, cache) the compiled trunk sidecar at `<packagePath>/<dir>`.
    /// nil when that DIR is absent (f32/f16 / codec-only / no-such-block builds; bf16 AND u4
    /// carry it). A PRESENT dir that fails to load THROWS (uncached, so a retry re-throws).
    public func resolveCompiledSidecar(_ dir: String) throws -> SmeltRuntime? {
        if let cached = compiledSidecars[dir] { return cached }
        let path = "\(packagePath)/\(dir)"
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else {
            compiledSidecars[dir] = .some(nil)   // dir absent — settled, never a trunk
            return nil
        }
        let rt = try SmeltRuntime(packagePath: path, device: device)
        compiledSidecars[dir] = .some(rt)
        return rt
    }

    /// The MAIN talker trunk (B3.2c, trunk/) and the MTP/code_predictor trunk (B3.2d, trunk-mtp/).
    public func resolveCompiledTrunk() throws -> SmeltRuntime? { try resolveCompiledSidecar("trunk") }
    public func resolveCompiledMTPTrunk() throws -> SmeltRuntime? { try resolveCompiledSidecar("trunk-mtp") }

    /// Force BOTH trunk loads NOW (during warmup) so their pipeline compiles never land on a
    /// request's TTFA. The serve handler / linger worker / run path call this right after
    /// constructing the GPU. No-op for blocks the package doesn't carry, or under a non-batched
    /// scope (SMELT_TTS_UNBATCHED) — where generation won't run anyway (it requires batched mode).
    public func prewarmCompiledTrunk() throws {
        guard willRunBatched else { return }
        _ = try resolveCompiledSidecar("trunk")
        _ = try resolveCompiledSidecar("trunk-mtp")
    }

    public enum LoadError: Error, CustomStringConvertible {
        case noMetalDevice
        case noCommandQueue
        case unsupportedVersion(Int)
        case noEosTokens
        case pageSizeMismatch(manifest: Int, host: Int)
        case missingFunction(String)
        case weightsOpenFailed(String)
        case weightsSizeMismatch(String)
        case weightsMmapFailed(String)
        case weightOutOfBounds(String)
        case weightMisaligned(String)
        case bufferCreateFailed(String)
        case tokenizerIncomplete([String])
        case validationInvalid(String)

        public var description: String {
            switch self {
            case .noMetalDevice: return "no Metal device"
            case .noCommandQueue: return "could not create command queue"
            case let .unsupportedVersion(v): return "unsupported Qwen3TTS package version \(v)"
            case .noEosTokens: return "manifest has no eosTokens"
            case let .pageSizeMismatch(m, h): return "package pageSize \(m) not a multiple of host page \(h)"
            case let .missingFunction(f): return "metallib missing kernel \(f)"
            case let .weightsOpenFailed(p): return "weights.bin open failed: \(p)"
            case let .weightsSizeMismatch(p): return "weights.bin smaller than manifest totalBytes: \(p)"
            case let .weightsMmapFailed(p): return "weights.bin mmap failed: \(p)"
            case let .weightOutOfBounds(n): return "weight \(n) slice extends past weights.bin"
            case let .weightMisaligned(n): return "weight \(n) offset/length not page-aligned"
            case let .bufferCreateFailed(n): return "bytesNoCopy buffer failed for weight \(n)"
            case let .tokenizerIncomplete(f): return "package tokenizerFiles incomplete: \(f)"
            case let .validationInvalid(detail): return "Qwen3-TTS validation is invalid: \(detail)"
            }
        }
    }
}
