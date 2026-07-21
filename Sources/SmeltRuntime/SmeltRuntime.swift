// SmeltRuntime — Thin loader for .smeltpkg artifacts.
//
// ~200 lines. Five responsibilities:
// 1. Decode manifest.json → SmeltManifest → SmeltRuntimeConfig
// 2. mmap weights.bin → device-sized MTLBuffer segments with logical offsets
// 3. Load model.metallib → create pipeline states
// 4. Pre-allocate all buffers from manifest.buffers.slots
// 5. decodeStep: one command buffer, one encoder, zero ARC
//
// After init, the manifest is dropped. Only SmeltRuntimeConfig (all Int32,
// no heap) survives into the decode path.

import Foundation
@preconcurrency import Metal
import SmeltSchema
#if os(macOS)
import Darwin
#endif

public protocol CodedError: Error {
    var code: String { get }
}

public enum SmeltJSON {
    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }()

    public static func canonicalString<T: Encodable>(_ value: T) throws -> String {
        let data = try encoder.encode(value)
        return String(decoding: data, as: UTF8.self)
    }

    public static func canonicalData<T: Encodable>(_ value: T) throws -> Data {
        try encoder.encode(value)
    }
}

// MARK: - Runtime

/// Compact reusable runtime state captured after prefilling a base prompt.
///
/// Stores only the live recurrent/KV prefix state, not the widened request
/// workspace, so it can be restored into a trimmed runtime for later requests.
public struct SmeltPromptSnapshot: Sendable {
    public let promptLength: Int
    public let nextToken: Int32
    public let byteCount: Int

    /// Number of prompt tokens physically represented by the captured state.
    /// This may be shorter than `promptLength` when an automatic capture keeps
    /// an alignment tail for exact replay through the ordinary model graph.
    public let capturedLength: Int
    let replayTokenIds: [Int32]
    let convStates: [Data]
    let recStates: [Data]
    let keyCaches: [Data]
    let valueCaches: [Data]
    private let backing: SmeltPromptSnapshotBacking?

    init(
        promptLength: Int,
        nextToken: Int32,
        byteCount: Int,
        capturedLength: Int,
        replayTokenIds: [Int32],
        convStates: [Data],
        recStates: [Data],
        keyCaches: [Data],
        valueCaches: [Data],
        backing: SmeltPromptSnapshotBacking? = nil
    ) {
        self.promptLength = promptLength
        self.nextToken = nextToken
        self.byteCount = byteCount
        self.capturedLength = capturedLength
        self.replayTokenIds = replayTokenIds
        self.convStates = convStates
        self.recStates = recStates
        self.keyCaches = keyCaches
        self.valueCaches = valueCaches
        self.backing = backing
    }
}

/// In-memory checkpoint whose persistent model state remains device-resident.
/// Unlike the serialized `SmeltPromptSnapshot`, this representation never
/// stages recurrent/KV tensors through CPU `Data`; capture and restore use
/// Metal blits so unified-memory page ownership stays with the GPU.
public final class SmeltDevicePromptSnapshot: @unchecked Sendable {
    fileprivate enum Layout {
        case whole
        case cache(
            heads: Int,
            headDim: Int,
            bytesPerValue: Int,
            sourceStride: Int
        )
    }

    fileprivate struct Slot {
        let slotIndex: Int
        let buffer: MTLBuffer
        let layout: Layout
    }

    public let capturedLength: Int
    public let byteCount: Int
    fileprivate let slots: [Slot]

    fileprivate init(capturedLength: Int, byteCount: Int, slots: [Slot]) {
        self.capturedLength = capturedLength
        self.byteCount = byteCount
        self.slots = slots
    }
}

final class SmeltPromptSnapshotBacking: @unchecked Sendable {
    let data: NSData
    let sourceURL: URL?
    let sourceFileSize: Int?

    init(
        data: NSData,
        sourceURL: URL? = nil,
        sourceFileSize: Int? = nil
    ) {
        self.data = data
        self.sourceURL = sourceURL
        self.sourceFileSize = sourceFileSize
    }
}

public enum SmeltPromptSnapshotWriteMode: String, Codable, Equatable, Sendable {
    case serialized
    case linked
    case cloned
    case copied
}

public struct SmeltPromptSnapshotWriteInfo: Codable, Equatable, Sendable {
    public let path: String
    public let fileBytes: Int
    public let mode: SmeltPromptSnapshotWriteMode

    public init(
        path: String,
        fileBytes: Int,
        mode: SmeltPromptSnapshotWriteMode
    ) {
        self.path = path
        self.fileBytes = fileBytes
        self.mode = mode
    }
}

private struct SmeltPromptSnapshotFileHeader: Codable {
    let schemaVersion: Int
    let promptLength: Int
    let nextToken: Int32
    let capturedLength: Int
    let replayTokenIds: [Int32]
    let convStateLengths: [Int]
    let recStateLengths: [Int]
    let keyCacheLengths: [Int]
    let valueCacheLengths: [Int]
}

public enum SmeltPromptSnapshotFileError: Error, CustomStringConvertible {
    case invalidMagic
    case truncatedHeader
    case truncatedPayload
    case invalidPath(String)

    public var description: String {
        switch self {
        case .invalidMagic:
            return "Invalid Smelt prompt snapshot magic"
        case .truncatedHeader:
            return "Truncated Smelt prompt snapshot header"
        case .truncatedPayload:
            return "Truncated Smelt prompt snapshot payload"
        case .invalidPath(let path):
            return "Invalid Smelt prompt snapshot path: \(path)"
        }
    }
}

extension SmeltPromptSnapshot {
    private static let fileMagic = Array("AGENTKV1".utf8)

    @discardableResult
    public func write(to url: URL) throws -> SmeltPromptSnapshotWriteInfo {
        guard url.isFileURL else {
            throw SmeltPromptSnapshotFileError.invalidPath(url.absoluteString)
        }
        if let materialized = try materializeBackedSnapshot(to: url) {
            return materialized
        }
        let header = SmeltPromptSnapshotFileHeader(
            schemaVersion: 1,
            promptLength: promptLength,
            nextToken: nextToken,
            capturedLength: capturedLength,
            replayTokenIds: replayTokenIds,
            convStateLengths: convStates.map(\.count),
            recStateLengths: recStates.map(\.count),
            keyCacheLengths: keyCaches.map(\.count),
            valueCacheLengths: valueCaches.map(\.count)
        )
        let headerData = try JSONEncoder().encode(header)

        let path = url.path
        try? FileManager.default.removeItem(at: url)
        guard FileManager.default.createFile(
            atPath: path,
            contents: nil,
            attributes: [.posixPermissions: 0o600]
        ) else {
            throw SmeltPromptSnapshotFileError.invalidPath(path)
        }
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.write(contentsOf: Data(Self.fileMagic))
        var headerLength = Data()
        headerLength.appendUInt64LE(UInt64(headerData.count))
        try handle.write(contentsOf: headerLength)
        try handle.write(contentsOf: headerData)
        try Self.writeChunks(convStates, to: handle)
        try Self.writeChunks(recStates, to: handle)
        try Self.writeChunks(keyCaches, to: handle)
        try Self.writeChunks(valueCaches, to: handle)
        try Self.secureSnapshotFile(url)
        return SmeltPromptSnapshotWriteInfo(
            path: path,
            fileBytes: Self.fileMagic.count + MemoryLayout<UInt64>.size + headerData.count + byteCount,
            mode: .serialized
        )
    }

    public static func read(from url: URL) throws -> SmeltPromptSnapshot {
        guard url.isFileURL else {
            throw SmeltPromptSnapshotFileError.invalidPath(url.absoluteString)
        }
        let mapped = try NSData(
            contentsOfFile: url.path,
            options: [.mappedIfSafe]
        )
        let totalLength = mapped.length
        guard totalLength >= fileMagic.count + MemoryLayout<UInt64>.size else {
            throw SmeltPromptSnapshotFileError.truncatedHeader
        }
        let base = mapped.bytes.assumingMemoryBound(to: UInt8.self)
        for (index, byte) in fileMagic.enumerated() {
            guard base[index] == byte else {
                throw SmeltPromptSnapshotFileError.invalidMagic
            }
        }

        var offset = fileMagic.count
        let headerLength = Int(readUInt64LE(from: base.advanced(by: offset)))
        offset += MemoryLayout<UInt64>.size
        guard headerLength >= 0, offset + headerLength <= totalLength else {
            throw SmeltPromptSnapshotFileError.truncatedHeader
        }
        let headerData = Data(bytes: base.advanced(by: offset), count: headerLength)
        offset += headerLength
        let header = try JSONDecoder().decode(
            SmeltPromptSnapshotFileHeader.self,
            from: headerData
        )
        let backing = SmeltPromptSnapshotBacking(
            data: mapped,
            sourceURL: url,
            sourceFileSize: totalLength
        )
        let convStates = try readChunks(
            lengths: header.convStateLengths,
            base: base,
            totalLength: totalLength,
            offset: &offset
        )
        let recStates = try readChunks(
            lengths: header.recStateLengths,
            base: base,
            totalLength: totalLength,
            offset: &offset
        )
        let keyCaches = try readChunks(
            lengths: header.keyCacheLengths,
            base: base,
            totalLength: totalLength,
            offset: &offset
        )
        let valueCaches = try readChunks(
            lengths: header.valueCacheLengths,
            base: base,
            totalLength: totalLength,
            offset: &offset
        )
        let byteCount =
            header.convStateLengths.reduce(0, +)
            + header.recStateLengths.reduce(0, +)
            + header.keyCacheLengths.reduce(0, +)
            + header.valueCacheLengths.reduce(0, +)

        return SmeltPromptSnapshot(
            promptLength: header.promptLength,
            nextToken: header.nextToken,
            byteCount: byteCount,
            capturedLength: header.capturedLength,
            replayTokenIds: header.replayTokenIds,
            convStates: convStates,
            recStates: recStates,
            keyCaches: keyCaches,
            valueCaches: valueCaches,
            backing: backing
        )
    }

    private static func writeChunks(_ chunks: [Data], to handle: FileHandle) throws {
        for chunk in chunks {
            try handle.write(contentsOf: chunk)
        }
    }

    private func materializeBackedSnapshot(to url: URL) throws -> SmeltPromptSnapshotWriteInfo? {
        guard let sourceURL = backing?.sourceURL,
              let sourceFileSize = backing?.sourceFileSize
        else {
            return nil
        }
        let sourcePath = sourceURL.standardizedFileURL.path
        let destinationPath = url.standardizedFileURL.path
        if sourcePath == destinationPath {
            guard FileManager.default.fileExists(atPath: destinationPath) else {
                return nil
            }
            try Self.secureSnapshotFile(url)
            return SmeltPromptSnapshotWriteInfo(
                path: destinationPath,
                fileBytes: sourceFileSize,
                mode: .linked
            )
        }

        try? FileManager.default.removeItem(at: url)
        do {
            try FileManager.default.linkItem(at: sourceURL, to: url)
            try Self.secureSnapshotFile(url)
            return SmeltPromptSnapshotWriteInfo(
                path: destinationPath,
                fileBytes: sourceFileSize,
                mode: .linked
            )
        } catch {
            try? FileManager.default.removeItem(at: url)
        }

        if Self.cloneFile(sourceURL, to: url) {
            try Self.secureSnapshotFile(url)
            return SmeltPromptSnapshotWriteInfo(
                path: destinationPath,
                fileBytes: sourceFileSize,
                mode: .cloned
            )
        }
        try? FileManager.default.removeItem(at: url)

        do {
            try FileManager.default.copyItem(at: sourceURL, to: url)
            try Self.secureSnapshotFile(url)
            return SmeltPromptSnapshotWriteInfo(
                path: destinationPath,
                fileBytes: sourceFileSize,
                mode: .copied
            )
        } catch {
            try? FileManager.default.removeItem(at: url)
            return nil
        }
    }

    private static func secureSnapshotFile(_ url: URL) throws {
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o600],
            ofItemAtPath: url.path
        )
    }

    private static func cloneFile(_ sourceURL: URL, to destinationURL: URL) -> Bool {
        #if os(macOS)
        copyfile(sourceURL.path, destinationURL.path, nil, UInt32(COPYFILE_CLONE)) == 0
        #else
        false
        #endif
    }

    private static func readChunks(
        lengths: [Int],
        base: UnsafePointer<UInt8>,
        totalLength: Int,
        offset: inout Int
    ) throws -> [Data] {
        try lengths.map { length in
            guard length >= 0, offset + length <= totalLength else {
                throw SmeltPromptSnapshotFileError.truncatedPayload
            }
            defer { offset += length }
            return Data(
                bytesNoCopy: UnsafeMutableRawPointer(mutating: base.advanced(by: offset)),
                count: length,
                deallocator: .none
            )
        }
    }

    private static func readUInt64LE(from pointer: UnsafePointer<UInt8>) -> UInt64 {
        var value: UInt64 = 0
        for index in 0..<8 {
            value |= UInt64(pointer[index]) << UInt64(index * 8)
        }
        return value
    }
}

private extension Data {
    mutating func appendUInt64LE(_ value: UInt64) {
        for index in 0..<8 {
            append(UInt8((value >> UInt64(index * 8)) & 0xff))
        }
    }
}

/// Loads a .smeltpkg and provides decode functionality.
/// After init, the hot path (decodeStep) touches only:
/// - Pipeline table already materialized for the active dispatch table
/// - UnsafeBufferPointer<MTLBuffer> (buffers)
/// - SmeltRuntimeConfig (Int32 scalars, no heap)
public final class SmeltRuntime {
    public struct MemoryStats: Codable, Equatable, Sendable {
        public let totalAllocatedBytes: Int
        public let weightBytes: Int
        public let persistentBytes: Int
        public let batchScopedBytes: Int
        public let contextScopedBytes: Int
        public let currentBatchCapacity: Int
        public let currentContextCapacity: Int
    }

    private enum PipelineSlot {
        case ready(MTLComputePipelineState)
        case fallbackToPrevious
    }

    private final class PipelineStartupCompilation: @unchecked Sendable {
        private let lock = NSLock()
        private var results: [Result<PipelineSlot, Error>?]
        private var archives: [any MTLBinaryArchive] = []
        private let group = DispatchGroup()

        init(count: Int) {
            self.results = [Result<PipelineSlot, Error>?](
                repeating: nil, count: count)
        }

        func start(
            indices: [Int],
            entries: [String],
            library: MTLLibrary,
            device: MTLDevice,
            icbCapable: Bool,
            archiveLoader: @escaping @Sendable () -> [any MTLBinaryArchive]
        ) {
            guard !indices.isEmpty else { return }
            group.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                let archives = archiveLoader()
                self.setArchives(archives)
                DispatchQueue.concurrentPerform(iterations: indices.count) { ordinal in
                    let index = indices[ordinal]
                    let result = Result {
                        try SmeltRuntime.compilePipelineSlot(
                            entry: entries[index],
                            library: library,
                            device: device,
                            icbCapable: icbCapable,
                            archives: archives
                        )
                    }
                    self.set(result, at: index)
                }
                self.group.leave()
            }
        }

        func waitForResults() -> (
            results: [Result<PipelineSlot, Error>?],
            archives: [any MTLBinaryArchive]
        ) {
            group.wait()
            lock.lock()
            defer { lock.unlock() }
            return (results, archives)
        }

        private func set(_ result: Result<PipelineSlot, Error>, at index: Int) {
            lock.lock()
            results[index] = result
            lock.unlock()
        }

        private func setArchives(_ archives: [any MTLBinaryArchive]) {
            lock.lock()
            self.archives = archives
            lock.unlock()
        }
    }

    let device: MTLDevice
    private let queue: MTLCommandQueue
    /// Keeps the weights buffer wired into GPU residency for the queue's
    /// lifetime (committed at init so the first dispatch skips page-table
    /// setup for the whole weights file).
    private let residencySet: (any MTLResidencySet)?
    private let weightBuffers: SmeltMetalWeightBuffers

    /// The Metal device this runtime was constructed on. Composers
    /// (e.g., a drafter that binds the target's K/V buffers into its
    /// own slots) must use the same device — Metal resources are
    /// device-owned and a cross-device bind would fail at encode.
    public var metalDevice: MTLDevice { device }
    private let config: SmeltRuntimeConfig
    var pipelines: [MTLComputePipelineState?]
    private var materializedPipelines: [Bool]
    private let pipelineLibrary: MTLLibrary
    private var pipelineArchives: [any MTLBinaryArchive]
    private let pipelineICBCapable: Bool
    private let pipelineMaterializationLock = NSLock()
    private var startupPipelineCompilation: PipelineStartupCompilation?
    private var temperatureSamplingPipeline: MTLComputePipelineState?
    private var gumbelSamplingPipeline: MTLComputePipelineState?
    private var specBVK3Pipeline: MTLComputePipelineState?
    var buffers: [MTLBuffer]
    private let packagePath: String
    let manifest: SmeltManifest
    private let slotsByIndex: [SmeltBufferSlot?]
    private let dynamicRequestBuffersEnabled: Bool
    private var currentBatchCapacity: Int
    private var currentContextCapacity: Int
    private var loggedTrimFailure = false

    /// The cross-chunk cached attention's threadgroup score-buffer cap: the SIMD kernel
    /// `causal_gqa_attn_cached_f32` stages `sc[2048]`, so it can attend a cache of at most
    /// 2048 rows. A chunk whose cache length (startPos+seqLen) exceeds this is routed to the
    /// uncapped scalar-streaming variant instead (Phase 4 U1, variable-length prefill).
    static let cachedAttnSimdCap = 2048

    /// Pipeline indices of the capped SIMD cross-chunk attention and its uncapped scalar
    /// streaming variant. When a prefill chunk's cache length exceeds `cachedAttnSimdCap`,
    /// `emitDispatchRecord` substitutes the scalar kernel — numerically equivalent (≤~1e-6),
    /// register-only `acc[headDim]`, no length cap. nil when the package predates the kernel
    /// (a stale package: `prefillTrunkChunked` rejects a >2048 prefill rather than mis-run).
    private lazy var cachedAttnCappedPipelineIndex: Int? =
        manifest.pipelines.firstIndex(of: "causal_gqa_attn_cached_f32")
    private lazy var cachedAttnScalarPipelineIndex: Int? =
        manifest.pipelines.firstIndex(of: "causal_gqa_attn_cached_scalar_f32")

    /// Test observability (Phase 4 U1 non-vacuity): counts scalar-attention substitutions,
    /// so a >2048 gate can prove the uncapped path actually ran (the SIMD kernel would
    /// otherwise read past its sc[2048] stage). Not load-bearing for production behavior.
    nonisolated(unsafe) static var cachedAttnScalarSubstitutions = 0

    /// Pipeline binary archives are useful for cold shader-cache experiments,
    /// but the warm-start LLM path is faster when the small required PSO set is
    /// created directly from the packaged metallib.
    private static let pipelineArchivesEnabled: Bool =
        ProcessInfo.processInfo.environment["SMELT_PIPELINE_ARCHIVES"] == "1"

    /// Binary dispatch table (mmap'd from dispatches.bin).
    /// Nil if package was built without a dispatch table.
    private let dispatchTable: UnsafeBufferPointer<SmeltDispatchRecord>?
    private let dispatchTableData: Data?  // retains the mmap

    /// Prefill dispatch table (mmap'd from prefill_dispatches.bin).
    /// Nil if package was built without Metal prefill.
    private let prefillDispatchTable: UnsafeBufferPointer<SmeltDispatchRecord>?
    private let prefillDispatchTableData: Data?  // retains the mmap
    private let prefillVerifyArgmaxDispatchTable: UnsafeBufferPointer<SmeltDispatchRecord>?
    private let prefillVerifyArgmaxDispatchTableData: Data?

    private var verifyICB: VerifyICBState = .pending
    private let verifyICBEnabled: Bool =
        ProcessInfo.processInfo.environment["SMELT_VERIFY_ICB"] == "1"

    /// Opt-in stderr trace of every `resizeRequestScopedBuffers` event.
    /// `bindExternalKVBuffer` references go stale on resize, so this
    /// trace is the diagnostic surface for spotting that hazard.
    /// Set via `SMELT_RESIZE_TRACE=1`.
    static let resizeTraceEnabled: Bool =
        ProcessInfo.processInfo.environment["SMELT_RESIZE_TRACE"] == "1"

    var verifyProfile: VerifyProfileState =
        ProcessInfo.processInfo.environment["SMELT_PROFILE_VERIFY"] != nil
            ? .armed : .disabled
    var verifyProfileOutputPath: String? =
        ProcessInfo.processInfo.environment["SMELT_PROFILE_VERIFY"]
    var verifyProfilePipelineMatches: [String] =
        ProcessInfo.processInfo.environment["SMELT_PROFILE_VERIFY_MATCH"]?
            .split(separator: ",")
            .map(String.init)
            .filter { !$0.isEmpty } ?? []

    /// Set true by `armProfileForNextPrefill()` so SMELT_PROFILE_VERIFY
    /// captures the spec-decode verify dispatch, not the caller's
    /// earlier prompt prefill. Cleared on consumption inside
    /// `prefillAllLogits`.
    private var profileNextPrefillAsVerify: Bool = false
    private var specBVTargetLogitsBuffer: MTLBuffer?
    private var specBVCandidatesBuffer: MTLBuffer?
    private var specBVOutputBuffer: MTLBuffer?

    /// Spec-decode opt-in: arm the SMELT_PROFILE_VERIFY one-shot to
    /// fire on the very next `prefillAllLogits` call (the upcoming
    /// chunked-verify dispatch). Has no effect when SMELT_PROFILE_VERIFY
    /// is unset or has already fired.
    public func armProfileForNextPrefill() {
        profileNextPrefillAsVerify = true
    }

    /// Arm one explicit per-dispatch profile of the next transactional
    /// verify-argmax call. This is the first-class equivalent of the legacy
    /// SMELT_PROFILE_VERIFY environment hook and keeps installed `smelt lab`
    /// tooling self-contained.
    public func armVerifyArgmaxProfile(
        outputPath: String,
        pipelineNameMatches: [String] = []
    ) {
        verifyProfileOutputPath = outputPath
        verifyProfilePipelineMatches = pipelineNameMatches.filter { !$0.isEmpty }
        verifyProfile = .armed
    }

    /// Load a .smeltpkg artifact.
    ///
    /// - Parameters:
    ///   - packagePath: Path to the .smeltpkg directory.
    ///   - device: Metal device to use. Defaults to system default.
    public convenience init(
        packagePath: String,
        device: MTLDevice? = nil,
        verifyPackage: Bool = false,
        contextLimit: Int? = nil,
        manifest predecodedManifest: SmeltManifest? = nil
    ) throws {
        try self.init(
            packagePath: packagePath,
            device: device,
            verifyPackage: verifyPackage,
            contextLimit: contextLimit,
            manifest: predecodedManifest,
            maximumWeightBufferLengthForTesting: nil
        )
    }

    init(
        packagePath: String,
        device: MTLDevice? = nil,
        verifyPackage: Bool = false,
        contextLimit: Int? = nil,
        manifest predecodedManifest: SmeltManifest? = nil,
        maximumWeightBufferLengthForTesting: Int?
    ) throws {
        // SMELT_STARTUP_TRACE=1 prints per-stage init timings to stderr.
        let traceStartup =
            ProcessInfo.processInfo.environment["SMELT_STARTUP_TRACE"] == "1"
        let initStart = CFAbsoluteTimeGetCurrent()
        var stageStart = initStart
        func stamp(_ label: String) {
            guard traceStartup else { return }
            let now = CFAbsoluteTimeGetCurrent()
            let stage = String(format: "%+7.1fms", (now - stageStart) * 1000)
            let total = String(format: "%7.1fms", (now - initStart) * 1000)
            fputs("startup: \(stage)  (total \(total))  \(label)\n", stderr)
            stageStart = now
        }

        // Metal device + queue creation costs ~40-50ms and needs nothing from
        // the package, so it runs on a background thread while the manifest
        // decode and weights mmap hide under it. The empty command buffer
        // kicks queue/firmware spin-up during the rest of init instead of
        // taxing the first real dispatch.
        let providedDevice = device
        nonisolated(unsafe) var deviceResult:
            Result<(MTLDevice, MTLCommandQueue), Error> =
                .failure(SmeltRuntimeError.noMetalDevice)
        let deviceGroup = DispatchGroup()
        deviceGroup.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            deviceResult = Result {
                guard let dev = providedDevice ?? MTLCreateSystemDefaultDevice()
                else { throw SmeltRuntimeError.noMetalDevice }
                guard let q = dev.makeCommandQueue() else {
                    throw SmeltRuntimeError.noCommandQueue
                }
                q.makeCommandBuffer()?.commit()
                return (dev, q)
            }
            deviceGroup.leave()
        }

        // 1. Load and decode manifest, unless the caller already needed it
        // for package routing and can hand the same value through.
        let manifest: SmeltManifest
        if let predecodedManifest {
            manifest = predecodedManifest
        } else {
            let manifestPath = "\(packagePath)/manifest.json"
            let manifestData = try Data(contentsOf: URL(fileURLWithPath: manifestPath))
            manifest = try SmeltManifest.decode(from: manifestData)
        }
        try manifest.validatePackageOwnedRuntimePolicy()
        if verifyPackage || ProcessInfo.processInfo.environment["SMELT_VERIFY_PACKAGE"] == "1" {
            _ = try SmeltPackageIntegrity.verify(
                packagePath: packagePath,
                manifest: manifest,
                includeWeights: true
            )
        }

        stamp("manifest decode")

        // Prepared artifacts are declared by the package inventory itself.
        // Validate compiled grammar metadata at the universal package-open point
        // so direct consumers cannot silently run unconstrained on corruption.
        _ = try SmeltCompiledGrammar.load(packagePath: packagePath)
        stamp("prepared artifacts")

        let dynamicRequestBuffersEnabled = manifest.prefill?.engine != "coreml"
        let resolvedContextLimit = try Self.resolveContextLimit(
            manifest: manifest,
            requestedLimit: contextLimit,
            dynamicRequestBuffersEnabled: dynamicRequestBuffersEnabled
        )

        // Extract runtime config (Int32-only, no heap) then drop manifest
        self.config = SmeltRuntimeConfig(
            from: manifest,
            contextLimit: resolvedContextLimit
        )
        self.manifest = manifest
        self.packagePath = packagePath
        self.dynamicRequestBuffersEnabled = dynamicRequestBuffersEnabled
        self.currentBatchCapacity =
            self.dynamicRequestBuffersEnabled ? 1 : max(Int(self.config.maxPrefillBatchSize), 1)
        self.currentContextCapacity =
            self.dynamicRequestBuffersEnabled ? 1 : max(Int(self.config.contextLimit), 1)

        // 2. Weight mappings are created after the device join because the
        // segment plan comes from that device's actual maxBufferLength.
        let weightsPath = "\(packagePath)/weights.bin"

        deviceGroup.wait()
        let (dev, q) = try deviceResult.get()
        self.device = dev
        self.queue = q
        stamp("Metal device + queue (overlapped)")

        let archivePath = "\(packagePath)/model.metalarchive"
        let pipelineArchivePath =
            Self.pipelineArchivesEnabled
            && FileManager.default.fileExists(atPath: archivePath)
            ? archivePath
            : nil
        nonisolated(unsafe) var preloadedPipelineArchives: [any MTLBinaryArchive] = []
        let pipelineArchiveGroup = DispatchGroup()
        if pipelineArchivePath != nil {
            pipelineArchiveGroup.enter()
            DispatchQueue.global(qos: .userInitiated).async {
                preloadedPipelineArchives = Self.loadPipelineArchives(
                    device: dev,
                    archivePath: pipelineArchivePath
                )
                pipelineArchiveGroup.leave()
            }
        }
        let pipelineArchiveLoader: @Sendable () -> [any MTLBinaryArchive] = {
            pipelineArchiveGroup.wait()
            return preloadedPipelineArchives
        }

        // Weights buffer creation + residency wiring run in the background,
        // overlapped with pipeline creation and working-buffer allocation;
        // joined at the end of init. Wiring residency here (macOS 15
        // residency sets) keeps GPU page-table setup for the whole weights
        // file out of the first real dispatch.
        nonisolated(unsafe) var weightBufferResult:
            Result<(SmeltMetalWeightBuffers, (any MTLResidencySet)?), Error> =
                .failure(SmeltRuntimeError.noMetalDevice)
        let weightsGroup = DispatchGroup()
        weightsGroup.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            weightBufferResult = Result {
                let buffers = try SmeltMetalWeightBuffers(
                    path: weightsPath,
                    manifest: manifest.weights,
                    device: dev,
                    maximumBufferLength: maximumWeightBufferLengthForTesting
                )
                let descriptor = MTLResidencySetDescriptor()
                descriptor.initialCapacity = buffers.segments.count
                let residency = try? dev.makeResidencySet(descriptor: descriptor)
                if let residency {
                    for segment in buffers.segments {
                        residency.addAllocation(segment.buffer)
                    }
                    residency.commit()
                    residency.requestResidency()
                    q.addResidencySet(residency)
                }
                return (buffers, residency)
            }
            weightsGroup.leave()
        }

        // 3. Load model.metallib and the binary dispatch tables. Pipeline
        // states are then materialized only for manifest entries the package
        // can actually dispatch. SMELT_EAGER_PIPELINES=1 preserves the old
        // all-pipelines startup mode for lab/debug surfaces that synthesize
        // arbitrary records.
        let metallibPath = "\(packagePath)/model.metallib"
        let metallibURL = URL(fileURLWithPath: metallibPath)
        let library = try dev.makeLibrary(URL: metallibURL)
        stamp("metallib load")

        let dispatchLoad = try Self.loadDispatchTable(
            packagePath: packagePath,
            fileName: "dispatches.bin"
        )
        let prefillDispatchLoad = try Self.loadDispatchTable(
            packagePath: packagePath,
            fileName: "prefill_dispatches.bin"
        )
        let prefillVerifyArgmaxDispatchLoad = try Self.loadDispatchTable(
            packagePath: packagePath,
            fileName: "prefill_verify_argmax_dispatches.bin"
        )
        self.dispatchTableData = dispatchLoad?.data
        self.dispatchTable = dispatchLoad?.table
        self.prefillDispatchTableData = prefillDispatchLoad?.data
        self.prefillDispatchTable = prefillDispatchLoad?.table
        self.prefillVerifyArgmaxDispatchTableData =
            prefillVerifyArgmaxDispatchLoad?.data
        self.prefillVerifyArgmaxDispatchTable =
            prefillVerifyArgmaxDispatchLoad?.table
        try Self.validateDeclaredPrefillArtifacts(
            packagePath: packagePath,
            manifest: manifest,
            prefillDispatchTable: prefillDispatchLoad?.table
        )
        stamp("dispatch tables")

        // ICB-encodable pipelines need supportIndirectCommandBuffers on the
        // descriptor; the flag may trigger an extra shader variant, so it's
        // gated on the same env flag the ICB build path is.
        let icbCapable =
            ProcessInfo.processInfo.environment["SMELT_VERIFY_ICB"] == "1"

        // Pre-compiled pipeline archive (written by `smelt build`). It remains
        // opt-in because the warm-start LLM gate only materializes a small PSO
        // set, and direct metallib creation is materially faster there.
        self.pipelineLibrary = library
        self.pipelineArchives = []
        self.pipelineICBCapable = icbCapable
        // Pipeline creation is thread-safe and embarrassingly parallel;
        // entries whose function can't be materialized fall back to the
        // previous pipeline (never dispatched), resolved in a sequential
        // fixup pass to preserve the original semantics.
        let pipelineEntries = manifest.pipelines
        let eagerPipelines =
            ProcessInfo.processInfo.environment["SMELT_EAGER_PIPELINES"] == "1"
        let startupDispatchTables: [
            (name: String, table: UnsafeBufferPointer<SmeltDispatchRecord>?)
        ] = prefillDispatchLoad != nil
            ? [("prefill_dispatches.bin", prefillDispatchLoad?.table)]
            : [("dispatches.bin", dispatchLoad?.table)]
        let requiredPipelineIndices = try Self.requiredPipelineIndices(
            manifest: manifest,
            dispatchTables: startupDispatchTables,
            eager: eagerPipelines
        )
        let requiredPipelineList = requiredPipelineIndices.sorted()
        let materializedPipelines =
            [Bool](repeating: false, count: pipelineEntries.count)
        self.pipelines = [MTLComputePipelineState?](
            repeating: nil, count: pipelineEntries.count)
        self.materializedPipelines = materializedPipelines
        self.temperatureSamplingPipeline = nil
        self.gumbelSamplingPipeline = nil
        self.specBVK3Pipeline = nil
        if !requiredPipelineList.isEmpty {
            let startupCompilation = PipelineStartupCompilation(
                count: pipelineEntries.count)
            self.startupPipelineCompilation = startupCompilation
            startupCompilation.start(
                indices: requiredPipelineList,
                entries: pipelineEntries,
                library: library,
                device: dev,
                icbCapable: icbCapable,
                archiveLoader: pipelineArchiveLoader
            )
        }
        let materializedCount = materializedPipelines.lazy.filter { $0 }.count
        stamp(
            "pipeline states (\(materializedCount)/\(pipelines.count), "
                + "\(requiredPipelineList.count) async)"
        )

        // 4. Pre-allocate all buffers from manifest
        let slotCount = Int(config.bufferCount)
        var bufs = [MTLBuffer]()
        bufs.reserveCapacity(slotCount)

        // Create a lookup from slot index to slot metadata.
        var slotLookup = Array<SmeltBufferSlot?>(repeating: nil, count: slotCount)
        for slot in manifest.buffers.slots {
            guard slot.index >= 0 && slot.index < slotCount else { continue }
            slotLookup[slot.index] = slot
        }
        self.slotsByIndex = slotLookup

        for idx in 0..<slotCount {
            if idx == Int(config.weightsSlot) {
                // Placeholder — replaced by the mmap'd weights buffer at the
                // join below (its creation overlaps pipeline setup).
                guard let buf = dev.makeBuffer(length: 16, options: .storageModeShared)
                else {
                    throw SmeltRuntimeError.invalidPackage(
                        "Failed to allocate weights placeholder buffer"
                    )
                }
                bufs.append(buf)
            } else if let slot = slotLookup[idx], slot.sizeBytes > 0 {
                let size = Self.requestedSize(
                    for: slot,
                    config: manifest.config,
                    compiledMaxBatch: max(Int(self.config.maxPrefillBatchSize), 1),
                    batchCapacity: currentBatchCapacity,
                    contextCapacity: currentContextCapacity,
                    dynamicEnabled: dynamicRequestBuffersEnabled
                )
                guard let buf = dev.makeBuffer(
                    length: size, options: .storageModeShared
                ) else {
                    throw SmeltRuntimeError.invalidPackage(
                        "Failed to allocate buffer slot \(idx) (\(size) bytes)"
                    )
                }
                // Zero-init: MTLBuffer contents are NOT guaranteed zeroed.
                // State buffers (conv, rec, KV cache) and the attention mask
                // must start at zero or the first decode step reads garbage.
                memset(buf.contents(), 0, size)
                bufs.append(buf)
            } else {
                // Slot not in manifest or zero-size — allocate minimum
                guard let buf = dev.makeBuffer(
                    length: 16, options: .storageModeShared
                ) else {
                    throw SmeltRuntimeError.invalidPackage(
                        "Failed to allocate placeholder buffer slot \(idx)"
                    )
                }
                memset(buf.contents(), 0, 16)
                bufs.append(buf)
            }
        }
        stamp("buffer alloc")
        weightsGroup.wait()
        let (weightBuffers, residency) = try weightBufferResult.get()
        self.weightBuffers = weightBuffers
        self.residencySet = residency
        if Int(config.weightsSlot) >= 0, Int(config.weightsSlot) < bufs.count {
            bufs[Int(config.weightsSlot)] = weightBuffers.segments[0].buffer
        }
        Self.populateStaticTables(
            manifest: manifest,
            config: config,
            contextCapacity: currentContextCapacity,
            buffers: &bufs
        )
        self.buffers = bufs
        Self.commitAsyncQueueWarmup(device: dev, queue: q)
        stamp("weights join + residency")

    }

    private static func commitAsyncQueueWarmup(
        device: MTLDevice,
        queue: MTLCommandQueue
    ) {
        guard let buffer = device.makeBuffer(length: 4, options: .storageModeShared),
              let commandBuffer = queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder()
        else { return }
        blit.fill(buffer: buffer, range: 0..<buffer.length, value: 0)
        blit.endEncoding()
        commandBuffer.addCompletedHandler { _ in
            _ = buffer.length
        }
        commandBuffer.commit()
    }

    private static func resolveContextLimit(
        manifest: SmeltManifest,
        requestedLimit: Int?,
        dynamicRequestBuffersEnabled: Bool
    ) throws -> Int {
        if let requestedLimit, requestedLimit <= 0 {
            throw SmeltRuntimeError.invalidPackage(
                "--context-limit must be positive, got \(requestedLimit)"
            )
        }

        let activeLimit: Int
        if dynamicRequestBuffersEnabled {
            activeLimit = requestedLimit ?? Int(Int32.max)
        } else {
            activeLimit = requestedLimit ?? manifest.config.staticContextCapacity
        }
        guard activeLimit <= Int(Int32.max) else {
            throw SmeltRuntimeError.invalidPackage(
                "context limit \(activeLimit) exceeds Int32.max"
            )
        }

        if !dynamicRequestBuffersEnabled && activeLimit > manifest.config.staticContextCapacity {
            throw SmeltRuntimeError.invalidPackage(
                "--context-limit \(activeLimit) exceeds static package static_seq_capacity \(manifest.config.staticContextCapacity)"
            )
        }

        return activeLimit
    }

    private static func populateStaticTables(
        manifest: SmeltManifest,
        config: SmeltRuntimeConfig,
        contextCapacity: Int,
        buffers: inout [MTLBuffer]
    ) {
        for pair in SmeltRoPETables.resolvedPairs(from: manifest) {
            guard pair.cosSlot >= 0, pair.cosSlot < buffers.count else { continue }
            guard pair.sinSlot >= 0, pair.sinSlot < buffers.count else { continue }
            let tableDtype = manifest.buffers.slots.first { $0.index == pair.cosSlot }?.dtype ?? .fp16
            SmeltRoPETables.populate(
                cosBuffer: buffers[pair.cosSlot],
                sinBuffer: buffers[pair.sinSlot],
                rowCount: contextCapacity,
                dim: pair.dim,
                theta: pair.theta,
                freqDim: pair.freqDim,
                scaling: pair.scaling,
                layout: pair.layout,
                dtype: tableDtype
            )
        }
    }

    // MARK: - Prefill factory

    /// Create a prefill runner from the package manifest.
    /// - Parameter cacheDir: Path to cached prompt states. If nil, uses package/cache/.
    public func makePrefillRunner(cacheDir: String? = nil) throws -> SmeltPrefillRunner {
        guard let prefillManifest = manifest.prefill else {
            throw SmeltRuntimeError.invalidPackage("No prefill section in manifest")
        }
        let resolvedCacheDir = cacheDir ?? "\(packagePath)/cache"
        return try SmeltPrefillRunner(
            prefillManifest: prefillManifest,
            packagePath: packagePath,
            cacheDir: resolvedCacheDir,
            buffers: buffers
        )
    }

    /// Reset mutable working buffers before starting a fresh request.
    ///
    /// This clears activation scratch, recurrent/KV state, and dynamic inputs,
    /// while leaving weights and lookup tables intact.
    public func resetWorkingBuffers() {
        for slot in manifest.buffers.slots {
            guard buffers[slot.index].length > 0 else { continue }
            switch slot.category {
            case .activation, .state, .dynamic:
                memset(buffers[slot.index].contents(), 0, buffers[slot.index].length)
            case .weight, .table:
                continue
            }
        }
    }

    /// Grow request-scoped buffers for a fresh request.
    ///
    /// The runtime keeps weights, tables, and persistent layer state resident,
    /// but widens prefill activations and KV capacity only as much as needed.
    public func prepareForRequest(batchCapacity: Int, contextCapacity: Int) throws {
        try resizeRequestScopedBuffers(
            batchCapacity: clampedBatchCapacity(batchCapacity),
            contextCapacity: clampedContextCapacity(contextCapacity),
            preserveContext: false
        )
    }

    /// Trim request-scoped buffers back to a decode-sized floor.
    /// Best-effort: if allocation fails, keep the larger buffers and log once.
    public func trimRequestBuffers() {
        guard dynamicRequestBuffersEnabled else { return }
        do {
            try resizeRequestScopedBuffers(
                batchCapacity: 1,
                contextCapacity: 1,
                preserveContext: false
            )
        } catch {
            if !loggedTrimFailure {
                fputs("Smelt trim failed (keeping inflated buffers): \(error)\n", stderr)
                loggedTrimFailure = true
            }
        }
    }

    /// Report the runtime's current memory footprint.
    public func memoryStats() -> MemoryStats {
        let weight = weightBuffers.totalLogicalBytes
        var total = weight
        var batchScoped = 0
        var contextScoped = 0

        for idx in buffers.indices {
            if idx == Int(config.weightsSlot) { continue }
            let bytes = buffers[idx].length
            total += bytes
            guard let slot = slotsByIndex[idx] else { continue }
            if Self.isBatchScopedSlot(slot) {
                batchScoped += bytes
            } else if Self.isContextScopedSlot(slot) {
                contextScoped += bytes
            }
        }

        return MemoryStats(
            totalAllocatedBytes: total,
            weightBytes: weight,
            persistentBytes: total - weight - batchScoped - contextScoped,
            batchScopedBytes: batchScoped,
            contextScopedBytes: contextScoped,
            currentBatchCapacity: currentBatchCapacity,
            currentContextCapacity: currentContextCapacity
        )
    }

    /// Physical Metal mappings backing the package's one logical weight file.
    /// One is the fast path; larger packages are segmented by device limits.
    public var metalWeightBufferLengths: [Int] {
        weightBuffers.segments.map { $0.buffer.length }
    }

    var cacheSeqCapacityValue: UInt32 {
        UInt32(max(usesDynamicCacheSeqCapacity ? currentContextCapacity : Int(config.contextLimit), 1))
    }

    private var usesDynamicCacheSeqCapacity: Bool {
        guard dynamicRequestBuffersEnabled, config.numAttnLayers > 0 else {
            return false
        }
        let firstKeyCacheSlot = Int(config.keyCacheBaseSlot)
        guard firstKeyCacheSlot >= 0,
              firstKeyCacheSlot < slotsByIndex.count,
              let slot = slotsByIndex[firstKeyCacheSlot]
        else {
            return false
        }
        return Self.isContextScopedSlot(slot) && slot.shape.count >= 3
    }

    private func resolveDecodeConstant(
        _ con: SmeltConstantRecord,
        position: Int32,
        skipDispatch: inout Bool
    ) -> UInt32 {
        switch con.kind {
        case SmeltConstantRecord.kindPosition:
            return UInt32(bitPattern: position)
        case SmeltConstantRecord.kindPositionPlus1:
            return UInt32(bitPattern: position + 1)
        case SmeltConstantRecord.kindPositionPlus1LessThanLiteralSkipIfFalse:
            let val: UInt32 = UInt32(bitPattern: position + 1) < con.value ? 1 : 0
            if val == 0 { skipDispatch = true }
            return val
        case SmeltConstantRecord.kindPositionPlus1GreaterEqualLiteralSkipIfFalse:
            let val: UInt32 = UInt32(bitPattern: position + 1) >= con.value ? 1 : 0
            if val == 0 { skipDispatch = true }
            return val
        case SmeltConstantRecord.kindCacheSeqCapacity:
            return cacheSeqCapacityValue
        default:
            return con.value
        }
    }

    func resolvePrefillConstant(
        _ con: SmeltConstantRecord,
        seqLen: Int32,
        startPos: Int32,
        skipDispatch: inout Bool
    ) -> UInt32 {
        switch con.kind {
        case SmeltConstantRecord.kindPosition:
            return UInt32(bitPattern: startPos)
        case SmeltConstantRecord.kindPositionPlus1:
            return UInt32(bitPattern: startPos + 1)
        case SmeltConstantRecord.kindSeqLen:
            return UInt32(bitPattern: seqLen)
        case SmeltConstantRecord.kindSeqLenMulLiteral:
            return UInt32(bitPattern: seqLen) &* con.value
        case SmeltConstantRecord.kindSeqLenModLiteral:
            return con.value == 0 ? 0 : UInt32(bitPattern: seqLen) % con.value
        case SmeltConstantRecord.kindSeqLenModLiteralSkipIfZero:
            let val = con.value == 0 ? 0 : UInt32(bitPattern: seqLen) % con.value
            if val == 0 { skipDispatch = true }
            return val
        case SmeltConstantRecord.kindSeqLenLessThanLiteralSkipIfFalse:
            let val: UInt32 = UInt32(bitPattern: seqLen) < con.value ? 1 : 0
            if val == 0 { skipDispatch = true }
            return val
        case SmeltConstantRecord.kindStartPos:
            return UInt32(bitPattern: startPos)
        case SmeltConstantRecord.kindStartPosPlusLiteral:
            return UInt32(bitPattern: startPos) &+ con.value
        case SmeltConstantRecord.kindCacheSeqCapacity:
            return cacheSeqCapacityValue
        default:
            return con.value
        }
    }

    /// Capture the live recurrent/KV state after prefilling a prompt.
    ///
    /// The snapshot is compact: fixed per-layer states are copied whole, while
    /// KV caches keep only the live prefix for `promptLength`.
    public func capturePromptSnapshot(
        capturedLength: Int,
        promptLength: Int,
        nextToken: Int32,
        replayTokenIds: [Int32] = []
    ) -> SmeltPromptSnapshot {
        let clampedCapturedLength = min(max(capturedLength, 0), Int(config.contextLimit))
        let clampedPromptLength = min(max(promptLength, 0), Int(config.contextLimit))

        var convStates: [Data] = []
        var recStates: [Data] = []
        var keyCaches: [Data] = []
        var valueCaches: [Data] = []
        var totalBytes = 0

        for layerIdx in 0..<Int(config.numDeltaLayers) {
            let convSlot = Int(config.convStateBaseSlot) + layerIdx
            if let data = copyWholeSlot(slotIndex: convSlot) {
                convStates.append(data)
                totalBytes += data.count
            }

            let recSlot = Int(config.recStateBaseSlot) + layerIdx
            if let data = copyWholeSlot(slotIndex: recSlot) {
                recStates.append(data)
                totalBytes += data.count
            }
        }

        for layerIdx in 0..<Int(config.numAttnLayers) {
            let keySlot = Int(config.keyCacheBaseSlot) + layerIdx
            if let data = copyCachePrefix(slotIndex: keySlot, length: clampedCapturedLength) {
                keyCaches.append(data)
                totalBytes += data.count
            }

            let valueSlot = Int(config.valCacheBaseSlot) + layerIdx
            if let data = copyCachePrefix(slotIndex: valueSlot, length: clampedCapturedLength) {
                valueCaches.append(data)
                totalBytes += data.count
            }
        }

        return SmeltPromptSnapshot(
            promptLength: clampedPromptLength,
            nextToken: nextToken,
            byteCount: totalBytes,
            capturedLength: clampedCapturedLength,
            replayTokenIds: replayTokenIds,
            convStates: convStates,
            recStates: recStates,
            keyCaches: keyCaches,
            valueCaches: valueCaches
        )
    }

    /// Restore a previously captured prompt snapshot into the current runtime.
    ///
    /// Call `resetWorkingBuffers()` before restore so stale request state does not
    /// leak across restores. This method ensures context capacity for the live
    /// prefix and repopulates only the compact snapshot state.
    public func restorePromptSnapshot(_ snapshot: SmeltPromptSnapshot) throws {
        try restorePromptSnapshot(snapshot, length: snapshot.capturedLength)
    }

    /// Restore a captured prompt snapshot, truncating the K/V cache to `length`
    /// positions instead of the snapshot's full captured length. Used by the
    /// Prompt-state cache LCP path where the cached snapshot is longer than the
    /// shared prefix.
    ///
    /// Opaque recurrent/convolutional states don't have positional prefix
    /// semantics: they were captured at the snapshot's full length and cannot
    /// be rewound. The package's CAM-derived restore mode decides whether a
    /// partial restore is legal; no model architecture or layer family is
    /// consulted here.
    public func restorePromptSnapshot(
        _ snapshot: SmeltPromptSnapshot,
        length: Int
    ) throws {
        guard length >= 0, length <= snapshot.capturedLength else {
            throw SmeltRuntimeError.invalidPackage(
                "restorePromptSnapshot length \(length) outside "
                + "[0, \(snapshot.capturedLength)]"
            )
        }
        if length < snapshot.capturedLength
            && promptStateRestoreMode == .exactPosition {
            throw SmeltRuntimeError.invalidPackage(
                "restorePromptSnapshot partial restore (length=\(length) < "
                + "captured=\(snapshot.capturedLength)) unsupported on package "
                + "whose CAM state declares exact-position restore semantics"
            )
        }

        let clampedLength = clampedContextCapacity(length)
        if clampedLength > currentContextCapacity {
            try resizeRequestScopedBuffers(
                batchCapacity: currentBatchCapacity,
                contextCapacity: clampedLength,
                preserveContext: false
            )
        }

        for (layerIdx, data) in snapshot.convStates.enumerated() {
            copyWholeSlot(data: data, slotIndex: Int(config.convStateBaseSlot) + layerIdx)
        }

        for (layerIdx, data) in snapshot.recStates.enumerated() {
            copyWholeSlot(data: data, slotIndex: Int(config.recStateBaseSlot) + layerIdx)
        }

        for (layerIdx, data) in snapshot.keyCaches.enumerated() {
            restoreCachePrefix(
                data,
                slotIndex: Int(config.keyCacheBaseSlot) + layerIdx,
                length: clampedLength,
                sourceStride: snapshot.capturedLength
            )
        }

        for (layerIdx, data) in snapshot.valueCaches.enumerated() {
            restoreCachePrefix(
                data,
                slotIndex: Int(config.valCacheBaseSlot) + layerIdx,
                length: clampedLength,
                sourceStride: snapshot.capturedLength
            )
        }
    }

    /// Capture every package-declared persistent state family into private
    /// Metal buffers. This is the fast in-process checkpoint path; serialized
    /// package snapshots continue to use `SmeltPromptSnapshot` above.
    public func captureDevicePromptSnapshot(
        capturedLength: Int
    ) throws -> SmeltDevicePromptSnapshot {
        let length = min(max(capturedLength, 0), Int(config.contextLimit))
        guard let commandBuffer = queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder()
        else { throw SmeltRuntimeError.metalCommandBufferUnavailable }

        var slots: [SmeltDevicePromptSnapshot.Slot] = []
        var totalBytes = 0

        func captureWhole(_ slotIndex: Int) throws {
            guard slotIndex >= 0, slotIndex < buffers.count else { return }
            let source = buffers[slotIndex]
            guard source.length > 0 else { return }
            guard let destination = device.makeBuffer(
                length: source.length,
                options: .storageModePrivate
            ) else {
                throw SmeltRuntimeError.invalidPackage(
                    "could not allocate \(source.length)-byte device checkpoint buffer"
                )
            }
            blit.copy(
                from: source, sourceOffset: 0,
                to: destination, destinationOffset: 0,
                size: source.length
            )
            slots.append(.init(
                slotIndex: slotIndex,
                buffer: destination,
                layout: .whole
            ))
            totalBytes += source.length
        }

        func captureCache(_ slotIndex: Int) throws {
            guard slotIndex >= 0, slotIndex < buffers.count,
                  let slot = slotsByIndex[slotIndex],
                  slot.shape.count >= 3
            else { return }
            let heads = slot.shape[0]
            let headDim = slot.shape[2]
            let bytesPerValue = Self.bytesPerElement(slot.dtype)
            let source = buffers[slotIndex]
            let currentStride = max(
                1,
                source.length / max(heads * headDim * bytesPerValue, 1)
            )
            let liveLength = min(length, currentStride)
            let sourceHeadBytes = currentStride * headDim * bytesPerValue
            let checkpointHeadBytes = liveLength * headDim * bytesPerValue
            let checkpointBytes = heads * checkpointHeadBytes
            guard checkpointBytes > 0,
                  let destination = device.makeBuffer(
                    length: checkpointBytes,
                    options: .storageModePrivate
                  )
            else {
                throw SmeltRuntimeError.invalidPackage(
                    "could not allocate \(checkpointBytes)-byte device KV checkpoint buffer"
                )
            }
            for head in 0..<heads {
                blit.copy(
                    from: source,
                    sourceOffset: head * sourceHeadBytes,
                    to: destination,
                    destinationOffset: head * checkpointHeadBytes,
                    size: checkpointHeadBytes
                )
            }
            slots.append(.init(
                slotIndex: slotIndex,
                buffer: destination,
                layout: .cache(
                    heads: heads,
                    headDim: headDim,
                    bytesPerValue: bytesPerValue,
                    sourceStride: liveLength
                )
            ))
            totalBytes += checkpointBytes
        }

        do {
            for slot in manifest.buffers.slots where slot.category == .state {
                if Self.isKVCacheSlot(slot) {
                    try captureCache(slot.index)
                } else {
                    try captureWhole(slot.index)
                }
            }
        } catch {
            blit.endEncoding()
            throw error
        }
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
        return SmeltDevicePromptSnapshot(
            capturedLength: length,
            byteCount: totalBytes,
            slots: slots
        )
    }

    public func restoreDevicePromptSnapshot(
        _ snapshot: SmeltDevicePromptSnapshot
    ) throws {
        let length = clampedContextCapacity(snapshot.capturedLength)
        if length > currentContextCapacity {
            try resizeRequestScopedBuffers(
                batchCapacity: currentBatchCapacity,
                contextCapacity: length,
                preserveContext: false
            )
        }
        guard let commandBuffer = queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder()
        else { throw SmeltRuntimeError.metalCommandBufferUnavailable }

        for saved in snapshot.slots {
            guard saved.slotIndex >= 0, saved.slotIndex < buffers.count else {
                continue
            }
            let destination = buffers[saved.slotIndex]
            switch saved.layout {
            case .whole:
                blit.copy(
                    from: saved.buffer, sourceOffset: 0,
                    to: destination, destinationOffset: 0,
                    size: min(saved.buffer.length, destination.length)
                )
            case .cache(
                let heads,
                let headDim,
                let bytesPerValue,
                let sourceStride
            ):
                let destinationStride = max(
                    1,
                    destination.length
                        / max(heads * headDim * bytesPerValue, 1)
                )
                let liveLength = min(sourceStride, destinationStride)
                let sourceHeadBytes = sourceStride * headDim * bytesPerValue
                let destinationHeadBytes = destinationStride * headDim * bytesPerValue
                let copyBytes = liveLength * headDim * bytesPerValue
                for head in 0..<heads {
                    blit.copy(
                        from: saved.buffer,
                        sourceOffset: head * sourceHeadBytes,
                        to: destination,
                        destinationOffset: head * destinationHeadBytes,
                        size: copyBytes
                    )
                }
            }
        }
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
    }

    // MARK: - Decode

    private func makeCommandBufferAndEncoder() throws
        -> (MTLCommandBuffer, MTLComputeCommandEncoder)
    {
        try finishStartupPipelineMaterialization()
        guard let cmdBuf = queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder()
        else {
            throw SmeltRuntimeError.metalCommandBufferUnavailable
        }
        return (cmdBuf, enc)
    }

    /// Run one decode step: token in → token out.
    /// Interprets the binary dispatch table from dispatches.bin.
    /// - Throws: `SmeltRuntimeError.metalCommandBufferUnavailable` if the GPU
    ///   command buffer cannot be created, or `metalAllocationFailed` if the
    ///   request-scoped buffers cannot be grown to fit `position`.
    public func decodeStep(
        tokenId: Int32,
        position: Int32,
        selectionMode: SmeltSelectionMode = .argmax,
        allowedTokenMask: SmeltAllowedTokenMaskProvider? = nil
    ) throws -> Int32 {
        try ensureContextCapacity(Int(position) + 1)

        guard let table = dispatchTable else {
            throw SmeltRuntimeError.dispatchTableMissing(
                "no dispatches.bin — cannot decode"
            )
        }
        try ensurePipelineStatesMaterialized(
            for: table,
            tableName: "dispatches.bin"
        )
        let profile = SmeltDecodeProfile.enabled
        let t0 = profile ? CFAbsoluteTimeGetCurrent() : 0
        let (cmdBuf, enc) = try makeCommandBufferAndEncoder()

        interpretDispatchTable(
            table, enc: enc, tokenId: tokenId, position: position
        )

        enc.endEncoding()
        let usedGPUSampler =
            allowedTokenMask == nil
            && encodeSelectionKernelIfNeeded(
                commandBuffer: cmdBuf,
                position: position,
                selectionMode: selectionMode
            )
        let t1 = profile ? CFAbsoluteTimeGetCurrent() : 0
        cmdBuf.commit()
        let t2 = profile ? CFAbsoluteTimeGetCurrent() : 0
        // The mask is only consumed by CPU sampling after the GPU wait, so
        // evaluating the provider here hides its cost (llguidance mask
        // compute is ~ the cost of a small-model forward) under the GPU.
        // When profiling, mask time lands in gpuWaitUs.
        let mask: [UInt32]?
        do {
            mask = try allowedTokenMask?()
        } catch {
            cmdBuf.waitUntilCompleted()
            throw error
        }
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw error }
        let t3 = profile ? CFAbsoluteTimeGetCurrent() : 0
        if profile {
            SmeltDecodeProfile.record(
                encodeUs: (t1 - t0) * 1_000_000,
                submitUs: (t2 - t1) * 1_000_000,
                gpuWaitUs: (t3 - t2) * 1_000_000
            )
        }

        if usedGPUSampler {
            return buffers[Int(config.argmaxSlot)]
                .contents().load(as: Int32.self)
        }

        return selectCurrentToken(
            position: position,
            selectionMode: selectionMode,
            allowedTokenMask: mask
        )
    }

    /// Debug decode: execute only the first N dispatches, then stop.
    /// Returns argmax result (may be meaningless if stopped before the LM head).
    public func debugDecodeStep(
        tokenId: Int32,
        position: Int32,
        maxDispatches: Int
    ) throws -> Int32 {
        try ensureContextCapacity(Int(position) + 1)

        guard let table = dispatchTable else {
            throw SmeltRuntimeError.dispatchTableMissing(
                "no dispatches.bin — cannot debug-decode"
            )
        }
        try materializeAllPipelineStates()
        let (cmdBuf, enc) = try makeCommandBufferAndEncoder()

        let recordCount = Self.recordCount(
            coveringDispatches: maxDispatches,
            in: table
        )
        let truncated = UnsafeBufferPointer(
            start: table.baseAddress,
            count: recordCount
        )
        interpretDispatchTable(
            truncated, enc: enc, tokenId: tokenId, position: position
        )

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw error }

        return buffers[Int(config.argmaxSlot)]
            .contents().load(as: Int32.self)
    }

    /// Vocab size from config.
    public var vocabSize: Int { Int(config.vocabSize) }

    /// Transformer hidden width from the package manifest.
    public var hiddenSize: Int { manifest.config.hiddenSize }

    /// Number of recurrent (DeltaNet) layers. Speculative-decode
    /// callers MUST check this before any decode side effect:
    /// recurrent state isn't position-indexed and verify can't
    /// roll back state mutations that happen before a rejection-
    /// triggered fallback. See verifyDraft's numDeltaLayers guard.
    public var numDeltaLayers: Int { manifest.config.numDeltaLayers }

    public var promptStateRestoreMode: SmeltPromptStateRestoreMode {
        if let declared = manifest.inference?.promptStateRestoreMode {
            return declared
        }
        // Compatibility for packages emitted before the explicit CAM-derived
        // field existed. Infer from the package's concrete persistent buffer
        // inventory, not an architecture or layer-count heuristic.
        let hasOpaquePersistentState = manifest.buffers.slots.contains {
            $0.category == .state && !Self.isKVCacheSlot($0)
        }
        return hasOpaquePersistentState ? .exactPosition : .positionIndexed
    }

    /// Whether a prompt checkpoint can only be restored at its exact captured
    /// token position. Packages with non-positional persistent state declare
    /// this through their state layout; callers must not trim such snapshots.
    public var promptStateRequiresExactPositionRestore: Bool {
        promptStateRestoreMode == .exactPosition
    }

    /// Whether generation callers may use the package's chunked all-logits
    /// prefill as a positionally composable prompt operation.
    public var supportsBatchedPromptPrefill: Bool {
        supportsChunkedPrefillVerify
            && !promptStateRequiresExactPositionRestore
    }

    /// Whether prompt generation can use the package's Metal prefill table to
    /// select the continuation token directly. Unlike all-logits prefill this
    /// is safe for exact-position recurrent state: the table consumes the live
    /// state at `startPos`, advances it through the supplied suffix, and only
    /// exposes the final selection. It is unavailable on emit-all-logits
    /// packages because `prefillStep` deliberately has no selection contract
    /// for that output layout.
    public var supportsBatchedPromptSelection: Bool {
        hasMetalPrefill
            && maxPrefillBatchSize > 0
            && !supportsChunkedPrefillVerify
    }

    /// EOS / end-of-turn token IDs from the package manifest's
    /// `inference.eos_tokens`. Bench callers that stop on EOT need
    /// the model-specific set: Gemma 4 stops on 106 (`<turn|>`),
    /// Qwen on 248044/248046, Llama 3 on 128001/128008/128009.
    /// Empty when the package manifest omits an `inference` block.
    public var eosTokens: [Int32] { manifest.inference?.eosTokens ?? [] }

    /// True if the package was built with Metal prefill +
    /// `prefill.emit_all_logits=true` (Phase 7.1) — required for
    /// `prefillAllLogits` and the chunked-prefill verify path in
    /// SmeltSpeculativeRuntime. The engine check rejects
    /// CoreML+emit_all_logits packages: those have the flag set
    /// but no `prefill_dispatches.bin`, and prefillAllLogits would
    /// throw `dispatchTableMissing` mid-round instead of
    /// gracefully falling back to sequential verify.
    public var supportsChunkedPrefillVerify: Bool {
        let opts = manifest.buildProvenance?.resolvedOptions
        return opts?.prefillEmitAllLogits == true
            && opts?.prefillEngine == "metal"
    }

    public var supportsChunkedPrefillVerifyArgmax: Bool {
        let hasTable =
            manifest.buildProvenance?.resolvedOptions.prefillEngine == "metal"
            && prefillVerifyArgmaxDispatchTable != nil
        return hasTable
            && (numDeltaLayers == 0 || recurrentVerifyTokenCapacity > 0)
    }

    /// Whether `prefillAllLogits(tokens:startPos:)` can run with
    /// the given token count. False on packages without
    /// `emit_all_logits` or when the count exceeds the compiled
    /// `max_prefill_batch`.
    public func canChunkedPrefillVerify(tokenCount: Int) -> Bool {
        supportsChunkedPrefillVerify && tokenCount <= maxPrefillBatchSize
    }

    public func canChunkedPrefillVerifyArgmax(tokenCount: Int) -> Bool {
        supportsChunkedPrefillVerifyArgmax
            && tokenCount <= maxPrefillBatchSize
            && (numDeltaLayers == 0
                || tokenCount <= recurrentVerifyTokenCapacity)
    }

    /// Number of recurrent input-token checkpoints compiled into the
    /// verify-argmax table. Zero means this package predates transactional
    /// recurrent verification (or is not recurrent).
    public var recurrentVerifyTokenCapacity: Int {
        guard numDeltaLayers > 0 else { return 0 }
        var capacity = Int.max
        for layer in 0..<numDeltaLayers {
            guard let convState = manifest.buffers.slots.first(where: {
                      $0.name == "convState_\(layer)"
                  }),
                  let convHistory = manifest.buffers.slots.first(where: {
                      $0.name == "convStateHistory_\(layer)"
                  }),
                  let recState = manifest.buffers.slots.first(where: {
                      $0.name == "recState_\(layer)"
                  }),
                  let recHistory = manifest.buffers.slots.first(where: {
                      $0.name == "recStateHistory_\(layer)"
                  }),
                  convState.sizeBytes > 0,
                  recState.sizeBytes > 0,
                  convHistory.sizeBytes % convState.sizeBytes == 0,
                  recHistory.sizeBytes % recState.sizeBytes == 0
            else { return 0 }

            let convRows = convHistory.sizeBytes / convState.sizeBytes
            let recRows = recHistory.sizeBytes / recState.sizeBytes
            capacity = min(capacity, min(convRows - 1, recRows - 1))
        }
        return capacity == Int.max ? 0 : max(capacity, 0)
    }

    public var supportsRecurrentVerifyTransactions: Bool {
        numDeltaLayers > 0 && recurrentVerifyTokenCapacity > 0
    }

    /// Queue a GPU-only rollback/commit from verify history into every live
    /// recurrent state. The caller must enqueue its next target operation on
    /// this runtime's command queue; Metal queue ordering then folds the blit
    /// into the normal refresh wait without another CPU synchronization.
    func enqueueRecurrentVerifyStateCommit(historyRow: Int) throws {
        guard numDeltaLayers > 0 else { return }
        guard historyRow >= 0,
              historyRow <= recurrentVerifyTokenCapacity else {
            throw SmeltRuntimeError.invalidPackage(
                "recurrent verify history row \(historyRow) outside [0, "
                    + "\(recurrentVerifyTokenCapacity)]"
            )
        }
        guard let commandBuffer = queue.makeCommandBuffer(),
              let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw SmeltRuntimeError.metalCommandBufferUnavailable
        }

        for layer in 0..<numDeltaLayers {
            for (liveName, historyName) in [
                ("convState_\(layer)", "convStateHistory_\(layer)"),
                ("recState_\(layer)", "recStateHistory_\(layer)"),
            ] {
                guard let liveSlot = manifest.buffers.slots.first(where: {
                          $0.name == liveName
                      }),
                      let historySlot = manifest.buffers.slots.first(where: {
                          $0.name == historyName
                      }),
                      liveSlot.index >= 0,
                      historySlot.index >= 0,
                      liveSlot.index < buffers.count,
                      historySlot.index < buffers.count else {
                    blit.endEncoding()
                    throw SmeltRuntimeError.invalidPackage(
                        "recurrent verify transaction missing \(liveName) or "
                            + historyName
                    )
                }
                let live = buffers[liveSlot.index]
                let history = buffers[historySlot.index]
                let sourceOffset = historyRow * live.length
                guard sourceOffset + live.length <= history.length else {
                    blit.endEncoding()
                    throw SmeltRuntimeError.invalidPackage(
                        "recurrent verify history \(historyName) has "
                            + "\(history.length) bytes; row \(historyRow) needs "
                            + "\(sourceOffset + live.length)"
                    )
                }
                blit.copy(
                    from: history,
                    sourceOffset: sourceOffset,
                    to: live,
                    destinationOffset: 0,
                    size: live.length
                )
            }
        }
        blit.endEncoding()
        commandBuffer.commit()
    }

    /// Whether `prefillAllLogits(tokens:startPos:)` /
    /// `prefillAllLogitsResident(...)` can run end-to-end for the
    /// supplied token batch. Stricter than
    /// `canChunkedPrefillVerify` because it also rules out packages
    /// with recurrent (DeltaNet) layers, which the prefill-all-logits
    /// path refuses at runtime.
    public func canEmitPrefillLogits(for tokenCount: Int) -> Bool {
        canChunkedPrefillVerify(tokenCount: tokenCount) && numDeltaLayers == 0
    }

    /// Read-side handles an EAGLE-class drafter consumes after the
    /// target's most recent decode/prefill step: the post-final-norm
    /// hidden state plus per-family last-layer K/V cache.
    ///
    /// Buffer references reflect the runtime's current allocation at
    /// the time of the call. `prepareForRequest` may grow context-
    /// or batch-scoped buffers, which replaces the underlying
    /// `MTLBuffer` (Metal can't grow in place). Callers who cache
    /// the returned references must re-call `drafterTaps()` and
    /// re-encode dispatch tables after any `prepareForRequest`.
    /// See `SmeltDrafterTaps.lastHiddenState` for buffer-layout
    /// caveats on the Metal-prefill path.
    public func drafterTaps() throws -> SmeltDrafterTaps {
        let normOutSlot = Int(SmeltRuntimeConfig.fixedNormOutBufSlot)
        guard normOutSlot < buffers.count else {
            throw SmeltRuntimeError.invalidPackage(
                "normOutBuf slot \(normOutSlot) out of range"
            )
        }
        let hiddenBuf = buffers[normOutSlot]
        let hidden = manifest.config.hiddenSize

        guard let provenance = manifest.buildProvenance else {
            throw SmeltRuntimeError.invalidPackage(
                "manifest missing buildProvenance — drafterTaps requires a layer pattern"
            )
        }
        let expanded = provenance.resolvedOptions.expandedLayerPattern
        guard let sharedKV = manifest.config.sharedKVLayers else {
            // Legacy manifests predate the sharedKVLayers field. We
            // cannot tell a shared-KV Gemma target from a non-shared
            // model from a missing key, and getting it wrong on E2B/
            // E4B silently hands the drafter zero caches. Refuse.
            throw SmeltRuntimeError.invalidPackage(
                "manifest predates the sharedKVLayers field — rebuild the package with a current compiler before invoking drafterTaps"
            )
        }

        var attention: [String: SmeltDrafterAttentionTap] = [:]
        for family in attentionFamilies(in: expanded) {
            if let tap = try makeAttentionTap(
                family: family, expandedPattern: expanded, sharedKVLayers: sharedKV
            ) {
                attention[family] = tap
            }
        }

        return SmeltDrafterTaps(
            lastHiddenState: hiddenBuf,
            hiddenSize: hidden,
            attention: attention
        )
    }

    /// After a Metal-prefill verify, copy the hidden state at
    /// position offset `offsetTokens` (= row `offsetTokens` of the
    /// [B, hiddenSize] normOutBuf layout) to offset 0. Used by the
    /// no-refresh spec-decode path to make the drafter see the
    /// commit-boundary hidden state without paying the cost of a
    /// dedicated `target.decodeStep` refresh.
    ///
    /// normOutBuf is fp16 by package invariant — `SmeltDrafterTaps`
    /// documents the buffer's layout and the runtime's final-norm
    /// dispatch chain commits to that dtype.
    public func copyVerifyHiddenToHead(offsetTokens: Int) throws {
        guard offsetTokens >= 0 else {
            throw SmeltRuntimeError.invalidPackage(
                "copyVerifyHiddenToHead: negative offset \(offsetTokens)"
            )
        }
        let normOutSlot = Int(SmeltRuntimeConfig.fixedNormOutBufSlot)
        guard normOutSlot < buffers.count else {
            throw SmeltRuntimeError.invalidPackage(
                "normOutBuf slot \(normOutSlot) out of range"
            )
        }
        let buffer = buffers[normOutSlot]
        let rowBytes = manifest.config.hiddenSize * MemoryLayout<Float16>.stride
        let srcOffset = offsetTokens * rowBytes
        guard srcOffset + rowBytes <= buffer.length else {
            throw SmeltRuntimeError.invalidPackage(
                "copyVerifyHiddenToHead: row \(offsetTokens) "
                + "(byte range [\(srcOffset)..+\(rowBytes)]) exceeds "
                + "normOutBuf length \(buffer.length)"
            )
        }
        if srcOffset == 0 { return }
        memmove(
            buffer.contents(),
            buffer.contents().advanced(by: srcOffset),
            rowBytes
        )
    }

    /// After a Metal-prefill verify with `emit_all_logits=true`, copy
    /// one `[vocab]` logits row from the batched `[B, vocab]` layout
    /// back to row 0. Speculative-decode uses this when a verified
    /// draft tail becomes the committed boundary: the next round can
    /// consume row-0 logits exactly like a preceding `decodeStep`
    /// refresh had produced them, without re-decoding the same token.
    public func copyPrefillLogitsToHead(offsetTokens: Int) throws {
        guard offsetTokens >= 0 else {
            throw SmeltRuntimeError.invalidPackage(
                "copyPrefillLogitsToHead: negative offset \(offsetTokens)"
            )
        }
        let logitsSlot = Int(SmeltRuntimeConfig.fixedLogitsSlot)
        guard logitsSlot < buffers.count else {
            throw SmeltRuntimeError.invalidPackage(
                "logits slot \(logitsSlot) out of range"
            )
        }
        let buffer = buffers[logitsSlot]
        let rowBytes = Int(config.vocabSize) * MemoryLayout<Float16>.stride
        let srcOffset = offsetTokens * rowBytes
        guard srcOffset + rowBytes <= buffer.length else {
            throw SmeltRuntimeError.invalidPackage(
                "copyPrefillLogitsToHead: row \(offsetTokens) "
                + "(byte range [\(srcOffset)..+\(rowBytes)]) exceeds "
                + "logits buffer length \(buffer.length)"
            )
        }
        if srcOffset == 0 { return }
        memmove(
            buffer.contents(),
            buffer.contents().advanced(by: srcOffset),
            rowBytes
        )
    }

    /// Replace the MTLBuffer at a keyCache or valCache slot with
    /// `external`. Used by speculative-decode setups to point a
    /// drafter's keyCache_<n>/valCache_<n> slots at the target's
    /// last-layer K/V cache, so the drafter's external_kv
    /// attention dispatches read target's K/V instead of the
    /// drafter's own zero-init buffers.
    ///
    /// Constraints:
    ///   - `slotIndex` must reference a `keyCache_<n>` or
    ///     `valCache_<n>` slot. Other slot kinds throw —
    ///     swapping them is not safe (RoPE tables, attention
    ///     masks, and weights have implicit shape contracts the
    ///     runtime relies on).
    ///   - The external buffer's `length` must EQUAL the current
    ///     slot's allocated size. KV slots are laid out
    ///     `[kvHeads, contextCapacity, headDim]` fp16 and the
    ///     attention kernels read with the drafter's own
    ///     `cacheSeqCapacity` as the per-head stride; a larger
    ///     buffer doesn't help — it'd be a target K/V at the
    ///     target's contextCapacity stride and the drafter's
    ///     kernels would land on the wrong rows. Caller must
    ///     `prepareForRequest` BOTH runtimes to the same
    ///     contextCapacity first.
    ///   - The original buffer is dropped from this runtime's
    ///     `buffers` array — caller's responsibility (or ARC's)
    ///     for its lifetime now.
    ///   - `prepareForRequest` resizes overwrite this binding.
    ///     Re-bind after any context-capacity change.
    public func bindExternalKVBuffer(
        at slotIndex: Int, buffer external: MTLBuffer
    ) throws {
        guard slotIndex >= 0, slotIndex < buffers.count else {
            throw SmeltRuntimeError.invalidPackage(
                "bindExternalKVBuffer slot \(slotIndex) out of range"
            )
        }
        guard let slot = slotsByIndex[slotIndex] else {
            throw SmeltRuntimeError.invalidPackage(
                "bindExternalKVBuffer slot \(slotIndex) has no slot metadata"
            )
        }
        guard Self.isKVCacheSlot(slot) else {
            throw SmeltRuntimeError.invalidPackage(
                "bindExternalKVBuffer rejects slot '\(slot.name)' "
                + "(only keyCache_<n>/valCache_<n> are bindable)"
            )
        }
        let expected = buffers[slotIndex].length
        guard external.length == expected else {
            throw SmeltRuntimeError.invalidPackage(
                "bindExternalKVBuffer slot '\(slot.name)' needs exactly "
                + "\(expected) bytes (matching cacheSeqCapacity stride); "
                + "got \(external.length). prepareForRequest both runtimes "
                + "to the same contextCapacity before binding."
            )
        }
        guard buffers[slotIndex] !== external else { return }
        buffers[slotIndex] = external
        invalidateVerifyICB()
    }

    /// Copy `bytes` into the buffer at `slotIndex`. Used by tests
    /// and runtime composers that need to install pre-computed
    /// activations (e.g., the EAGLE drafter's pre-projection input
    /// `hiddenB`, which holds the concat of target's last token
    /// embedding and target's last hidden state).
    ///
    /// Distinct from `bindExternalKVBuffer`:
    ///   - `bindExternalKVBuffer` swaps the underlying MTLBuffer
    ///     reference (no-copy handoff). KV-only.
    ///   - `installSlotBytes` copies bytes into the existing
    ///     buffer's contents, leaving the MTLBuffer reference
    ///     unchanged.
    ///
    /// Accepts `.activation`, `.state`, and `.dynamic` slots.
    /// Rejects `.weight` (the logically contiguous mmap'd weights file
    /// is PROT_READ — memcpy would SIGBUS) and `.table` (RoPE
    /// cos/sin and attention masks are populated once at init
    /// from the spec; overwriting them silently corrupts every
    /// downstream attention dispatch). KV-cache `.state` slots
    /// pass; for those, prefer `bindExternalKVBuffer` when the
    /// caller already has an MTLBuffer to swap in (no-copy).
    ///
    /// Constraints:
    ///   - `slotIndex` must be in range. Out-of-range throws.
    ///   - `bytes.count` must equal the slot's current MTLBuffer
    ///     length (which reflects post-`prepareForRequest`
    ///     dynamic sizing for context-scoped slots, not the
    ///     manifest's static-capacity upper bound).
    ///   - `prepareForRequest` resizes overwrite the install on
    ///     context-scoped slots — re-install after any resize.
    public func installSlotBytes(at slotIndex: Int, bytes: Data) throws {
        if let error = installSlotBytesFailure(at: slotIndex, bytes: bytes) {
            throw error
        }
    }

    /// Install the runtime-owned source rows declared by the package's generic
    /// input-fusion ABI. This is the public composition surface for compiled
    /// auxiliary modules: callers name semantic rows in manifest order, while
    /// slot lookup, fp16 byte sizing, and concatenation remain runtime-owned.
    public func installInputFusionSources(_ sources: [Data]) throws {
        guard let fusion = manifest.config.inputFusion else {
            throw SmeltRuntimeError.invalidPackage(
                "installInputFusionSources requires manifest.config.inputFusion"
            )
        }
        guard sources.count == fusion.sourceCount else {
            throw SmeltRuntimeError.invalidPackage(
                "installInputFusionSources needs \(fusion.sourceCount) sources; got \(sources.count)"
            )
        }
        let rowBytes = fusion.sourceWidth * MemoryLayout<Float16>.stride
        for (index, source) in sources.enumerated() where source.count != rowBytes {
            throw SmeltRuntimeError.invalidPackage(
                "installInputFusionSources source \(index) needs \(rowBytes) fp16 bytes; got \(source.count)"
            )
        }
        guard let slot = manifest.buffers.slots.first(where: { $0.name == "hiddenB" }) else {
            throw SmeltRuntimeError.invalidPackage(
                "input-fusion package is missing hiddenB activation slot"
            )
        }
        var slab = Data()
        slab.reserveCapacity(rowBytes * sources.count)
        for source in sources { slab.append(source) }
        try installSlotBytes(at: slot.index, bytes: slab)
    }

    /// Current post-final-norm hidden row as fp16 bytes. Ordinary text models
    /// and input-fusion auxiliary modules expose the same hidden-state seam.
    public func currentHiddenStateBytes() throws -> Data {
        let normOutSlot = Int(SmeltRuntimeConfig.fixedNormOutBufSlot)
        let rowBytes = manifest.config.hiddenSize * MemoryLayout<Float16>.stride
        return try readSlotBytes(at: normOutSlot, offset: 0, length: rowBytes)
    }

    /// Non-throwing install path for callers that need to avoid a
    /// Swift `throws` ABI boundary around the raw Data -> MTLBuffer
    /// copy. Returns the same validation failure that
    /// `installSlotBytes(at:bytes:)` would throw.
    @inline(never)
    func installSlotBytesFailure(at slotIndex: Int, bytes: Data) -> SmeltRuntimeError? {
        guard slotIndex >= 0, slotIndex < buffers.count else {
            return SmeltRuntimeError.invalidPackage(
                "installSlotBytes slot \(slotIndex) out of range"
            )
        }
        guard let slot = slotsByIndex[slotIndex] else {
            return SmeltRuntimeError.invalidPackage(
                "installSlotBytes slot \(slotIndex) has no slot metadata"
            )
        }
        switch slot.category {
        case .activation, .state, .dynamic:
            break
        case .weight:
            return SmeltRuntimeError.invalidPackage(
                "installSlotBytes rejects slot '\(slot.name)' "
                + "(category=.weight is mmap'd PROT_READ; "
                + "memcpy would SIGBUS)"
            )
        case .table:
            return SmeltRuntimeError.invalidPackage(
                "installSlotBytes rejects slot '\(slot.name)' "
                + "(category=.table holds RoPE cos/sin or attention "
                + "masks populated once at init; overwriting silently "
                + "corrupts downstream dispatches)"
            )
        }
        let buffer = buffers[slotIndex]
        guard bytes.count == buffer.length else {
            return SmeltRuntimeError.invalidPackage(
                "installSlotBytes slot \(slotIndex) needs exactly "
                + "\(buffer.length) bytes; got \(bytes.count)"
            )
        }
        bytes.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return }
            memcpy(buffer.contents(), base, bytes.count)
        }
        return nil
    }

    /// Read `length` bytes from `slotIndex` starting at `offset`.
    ///
    /// Accepts any in-range slot regardless of category. The caller
    /// is responsible for knowing what's in the slot — reading from
    /// `.weight` slots is safe (mmap'd PROT_READ), but the bytes are
    /// the raw dispatch-ready layout, not user-friendly.
    public func readSlotBytes(
        at slotIndex: Int, offset: Int, length: Int
    ) throws -> Data {
        guard slotIndex >= 0, slotIndex < buffers.count else {
            throw SmeltRuntimeError.invalidPackage(
                "readSlotBytes slot \(slotIndex) out of range"
            )
        }
        let buffer = buffers[slotIndex]
        // Validate non-overflowing form: `offset + length` traps on
        // overflow (e.g., offset=Int.max, length=1), aborting the
        // process before we can throw a clean error. Compare against
        // `buffer.length - length` after individually bounding each
        // input; the subtraction is safe once `length <= buffer.length`.
        guard offset >= 0, length >= 0,
              length <= buffer.length,
              offset <= buffer.length - length
        else {
            throw SmeltRuntimeError.invalidPackage(
                "readSlotBytes slot \(slotIndex) range "
                + "[\(offset), \(offset)+\(length)) outside buffer "
                + "[0, \(buffer.length))"
            )
        }
        return Data(
            bytes: buffer.contents().advanced(by: offset),
            count: length
        )
    }

    /// Return the embed_tokens row for `tokenId` as raw fp16 bytes
    /// (length = `manifest.config.hiddenSize * 2`). Used by
    /// speculative-decode setups: the EAGLE drafter's pre_projection
    /// input concatenates target's embedding of the most-recent
    /// committed token with target's last hidden state, both in
    /// target hidden dim.
    private func weightPointer(
        logicalOffset: Int,
        length: Int,
        context: String
    ) throws -> UnsafeMutableRawPointer {
        guard let resolved = weightBuffers.resolve(
            logicalOffset: logicalOffset,
            length: length
        ) else {
            throw SmeltRuntimeError.invalidPackage(
                "\(context) range [\(logicalOffset), \(logicalOffset + length)) "
                    + "crosses or exceeds a Metal weight segment"
            )
        }
        return resolved.buffer.contents().advanced(by: resolved.offset)
    }

    public func embedToken(_ tokenId: Int32) throws -> Data {
        guard tokenId >= 0, tokenId < manifest.config.vocabSize else {
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: tokenId \(tokenId) out of range "
                + "[0, \(manifest.config.vocabSize))"
            )
        }
        guard let entry = manifest.weights.entries.first(
            where: { $0.name == SmeltCanonicalTensorNames.embedTokens }
        ) else {
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: manifest has no embed_tokens entry"
            )
        }
        let hiddenSize = manifest.config.hiddenSize
        let rowBytes = hiddenSize * MemoryLayout<Float16>.stride

        switch entry.dtype {
        case .fp16:
            let totalRowOffset = Int(entry.offset) + Int(tokenId) * rowBytes
            let entryEnd = Int(entry.offset) + Int(entry.sizeBytes)
            guard totalRowOffset + rowBytes <= entryEnd else {
                // Tighter bound than `weightsBuf.length` — catches a
                // malformed manifest where `entry.sizeBytes` doesn't
                // cover all `vocabSize * hiddenSize` rows.
                throw SmeltRuntimeError.invalidPackage(
                    "embedToken: row offset \(totalRowOffset)+\(rowBytes) "
                    + "exceeds embed_tokens entry end \(entryEnd)"
                )
            }
            let src = try weightPointer(
                logicalOffset: totalRowOffset,
                length: rowBytes,
                context: "embedToken fp16 row"
            )
            return Data(bytes: src, count: rowBytes)

        case .affineU4:
            return try dequantizeAffineEmbeddingRow(
                entry: entry,
                tokenId: Int(tokenId),
                hiddenSize: hiddenSize
            )

        case .u4Lut:
            return try dequantizeLUTEmbeddingRow(
                entry: entry,
                tokenId: Int(tokenId),
                hiddenSize: hiddenSize
            )

        case .turboQuantH:
            return try dequantizeTurboQuantHEmbeddingRow(
                entry: entry,
                tokenId: Int(tokenId),
                hiddenSize: hiddenSize
            )

        case .binary1, .ternary2:
            return try dequantizeSignedEmbeddingRow(
                entry: entry,
                tokenId: Int(tokenId),
                hiddenSize: hiddenSize
            )

        default:
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: embed_tokens dtype is \(entry.dtype); "
                + "expected fp16, affine_u4, u4_lut, turbo_quant_h, "
                + "binary1, or ternary2"
            )
        }
    }

    private func dequantizeSignedEmbeddingRow(
        entry: SmeltWeightEntry,
        tokenId: Int,
        hiddenSize: Int
    ) throws -> Data {
        guard let groupSize = entry.groupSize,
              let packedRowStride = entry.packedRowStride,
              let paddedCols = entry.paddedCols,
              let scalesOffset = entry.scalesOffset,
              let scalesSizeBytes = entry.scalesSizeBytes
        else {
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: signed embed_tokens missing "
                    + "group/stride/padded-cols/scale metadata"
            )
        }
        guard entry.shape.count == 2,
              entry.shape[0] == manifest.config.vocabSize,
              entry.shape[1] == hiddenSize
        else {
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: signed entry shape \(entry.shape) does not match "
                    + "[\(manifest.config.vocabSize), \(hiddenSize)]"
            )
        }
        let format: SmeltSignedQuantFormat
        switch entry.dtype {
        case .binary1: format = .binary1
        case .ternary2: format = .ternary2
        default:
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: internal signed dtype mismatch \(entry.dtype)"
            )
        }
        let groupsPerRow: Int
        do {
            groupsPerRow = try SmeltSignedQuantCodec.groupsPerRow(
                paddedCols: paddedCols,
                groupSize: groupSize
            )
        } catch {
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: signed geometry is invalid: \(error)"
            )
        }
        let scaleRowBytes = groupsPerRow * MemoryLayout<UInt16>.stride
        let packedRowOffset = Int(entry.offset) + tokenId * packedRowStride
        let packedEnd = Int(entry.offset) + Int(entry.sizeBytes)
        guard packedRowOffset + packedRowStride <= packedEnd else {
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: signed packed row exceeds embed_tokens entry"
            )
        }
        let scaleRowOffset = Int(scalesOffset) + tokenId * scaleRowBytes
        let scalesEnd = Int(scalesOffset) + Int(scalesSizeBytes)
        guard scaleRowOffset + scaleRowBytes <= scalesEnd else {
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: signed scale row exceeds embed_tokens scale region"
            )
        }

        let codes = try weightPointer(
            logicalOffset: packedRowOffset,
            length: packedRowStride,
            context: "embedToken signed codes"
        )
        let scales = try weightPointer(
            logicalOffset: scaleRowOffset,
            length: scaleRowBytes,
            context: "embedToken signed scales"
        )
        var output = [UInt16](repeating: 0, count: hiddenSize)
        do {
            try output.withUnsafeMutableBufferPointer { buffer in
                try SmeltSignedQuantCodec.dequantizeRowToFloat16Bits(
                    format: format,
                    codes: codes,
                    codeByteCount: packedRowStride,
                    scales: scales,
                    scaleByteCount: scaleRowBytes,
                    cols: hiddenSize,
                    paddedCols: paddedCols,
                    groupSize: groupSize,
                    into: buffer.baseAddress!
                )
            }
        } catch {
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: signed row decode failed: \(error)"
            )
        }
        return output.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    /// Dequantize one TurboQuant-H row via SmeltTurboQuantHCodec.
    private func dequantizeTurboQuantHEmbeddingRow(
        entry: SmeltWeightEntry,
        tokenId: Int,
        hiddenSize: Int
    ) throws -> Data {
        guard let groupSize = entry.groupSize,
              let codesPerRow = entry.packedRowStride,
              let codebookOffset = entry.codebookOffset
        else {
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: turbo_quant_h embed_tokens missing "
                + "group/stride/codebook metadata"
            )
        }
        guard groupSize > 0, (groupSize & (groupSize - 1)) == 0 else {
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: turbo_quant_h groupSize \(groupSize) "
                + "must be a power of two"
            )
        }
        let codesEnd = Int(entry.offset) + Int(entry.sizeBytes)
        let rowStart = Int(entry.offset) + tokenId * codesPerRow
        guard rowStart + codesPerRow <= codesEnd else {
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: turbo_quant_h row offset out of bounds "
                + "for entry [\(entry.offset)..\(codesEnd))"
            )
        }
        // entry.shape[1] is the writer-side cols. If config.hiddenSize
        // drifts from that the per-row byte counts and per-group
        // codebook indexing both go off. Reject early instead of
        // silently reading the wrong codes/centroids.
        guard entry.shape.count == 2, entry.shape[1] == hiddenSize else {
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: turbo_quant_h entry shape "
                + "\(entry.shape) does not match hiddenSize=\(hiddenSize)"
            )
        }
        let numGroups = SmeltTurboQuantHCodec.numGroups(
            cols: hiddenSize, groupSize: groupSize
        )
        // Validate the writer's declared stride covers the full
        // num_groups * groupSize nibble positions the codec will
        // read. The earlier rowStart-in-entry guard only proves
        // the row's declared bytes fit; a manifest where
        // packedRowStride is smaller than required would let the
        // codec walk past the row into either the next row or
        // padding bytes.
        let requiredCodesPerRow = (numGroups * groupSize + 3) / 4
        guard codesPerRow >= requiredCodesPerRow else {
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: turbo_quant_h packed_row_stride "
                + "\(codesPerRow) < required \(requiredCodesPerRow) "
                + "for hiddenSize=\(hiddenSize), groupSize=\(groupSize)"
            )
        }
        let expectedCodebookBytes = numGroups * 4 * 2
        guard let entryCodebookSize = entry.codebookSizeBytes,
              Int(entryCodebookSize) >= expectedCodebookBytes
        else {
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: turbo_quant_h codebook region "
                    + "[\(codebookOffset)..+\(expectedCodebookBytes)] "
                    + "has codebook_size_bytes=\(entry.codebookSizeBytes ?? 0)"
            )
        }

        let codesPtr = try weightPointer(
            logicalOffset: rowStart,
            length: codesPerRow,
            context: "embedToken turbo_quant_h codes"
        )
            .assumingMemoryBound(to: UInt8.self)
        let codebookPtr = try weightPointer(
            logicalOffset: Int(codebookOffset),
            length: expectedCodebookBytes,
            context: "embedToken turbo_quant_h codebook"
        )
            .assumingMemoryBound(to: UInt16.self)

        var out = [UInt16](repeating: 0, count: hiddenSize)
        out.withUnsafeMutableBufferPointer { buf in
            SmeltTurboQuantHCodec.dequantizeRowInto(
                codes: codesPtr,
                codebook: codebookPtr,
                output: buf.baseAddress!,
                cols: hiddenSize,
                groupSize: groupSize
            )
        }
        return out.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func dequantizeAffineEmbeddingRow(
        entry: SmeltWeightEntry,
        tokenId: Int,
        hiddenSize: Int
    ) throws -> Data {
        guard let groupSize = entry.groupSize,
              let packedRowStride = entry.packedRowStride,
              let scalesOffset = entry.scalesOffset,
              let biasesOffset = entry.biasesOffset
        else {
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: affine_u4 embed_tokens missing group/stride/scale/bias metadata"
            )
        }

        let numColGroups = (hiddenSize + groupSize - 1) / groupSize
        let packedRowOffset = Int(entry.offset) + tokenId * packedRowStride
        let packedEnd = Int(entry.offset) + Int(entry.sizeBytes)
        guard packedRowOffset + packedRowStride <= packedEnd else {
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: affine_u4 packed row exceeds embed_tokens entry"
            )
        }

        let packed = try weightPointer(
            logicalOffset: packedRowOffset,
            length: packedRowStride,
            context: "embedToken affine_u4 packed row"
        )
            .assumingMemoryBound(to: UInt8.self)
        let sbBase = tokenId * numColGroups
        let scaleRowOffset = sbBase * MemoryLayout<UInt16>.stride
        let scaleBytes = numColGroups * MemoryLayout<UInt16>.stride
        let scales = try weightPointer(
            logicalOffset: Int(scalesOffset) + scaleRowOffset,
            length: scaleBytes,
            context: "embedToken affine_u4 scales"
        )
            .assumingMemoryBound(to: UInt16.self)
        let biases = try weightPointer(
            logicalOffset: Int(biasesOffset) + scaleRowOffset,
            length: scaleBytes,
            context: "embedToken affine_u4 biases"
        )
            .assumingMemoryBound(to: UInt16.self)
        var out = [UInt16](repeating: 0, count: hiddenSize)
        // Walk one column-group at a time so scale/bias are loaded
        // once per group instead of redundantly on every byte (e.g.,
        // 1 lookup vs 64 redundant lookups within a 128-element
        // group). The tail group handles odd-hiddenSize / partial
        // bytes.
        let bytesPerGroup = groupSize / 2
        for groupIdx in 0 ..< numColGroups {
            let scale = Float(Float16(bitPattern: scales[groupIdx]))
            let bias = Float(Float16(bitPattern: biases[groupIdx]))
            let groupColStart = groupIdx * groupSize
            let groupColEnd = min(groupColStart + groupSize, hiddenSize)
            let groupByteStart = groupIdx * bytesPerGroup
            let groupByteEnd = min(groupByteStart + bytesPerGroup, packedRowStride)
            for byteIndex in groupByteStart ..< groupByteEnd {
                let packedByte = packed[byteIndex]
                let col0 = byteIndex * 2
                if col0 < groupColEnd {
                    out[col0] = Float16(Float(packedByte & 0x0F) * scale + bias).bitPattern
                }
                let col1 = col0 + 1
                if col1 < groupColEnd {
                    out[col1] = Float16(Float(packedByte >> 4) * scale + bias).bitPattern
                }
            }
        }

        return out.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    private func dequantizeLUTEmbeddingRow(
        entry: SmeltWeightEntry,
        tokenId: Int,
        hiddenSize: Int
    ) throws -> Data {
        guard let groupSize = entry.groupSize,
              let packedRowStride = entry.packedRowStride,
              let lutOffset = entry.lutOffset
        else {
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: u4_lut embed_tokens missing group/stride/lut metadata"
            )
        }

        let packedRowOffset = Int(entry.offset) + tokenId * packedRowStride
        let packedEnd = Int(entry.offset) + Int(entry.sizeBytes)
        guard packedRowOffset + packedRowStride <= packedEnd else {
            throw SmeltRuntimeError.invalidPackage(
                "embedToken: u4_lut packed row exceeds embed_tokens entry"
            )
        }

        let packed = try weightPointer(
            logicalOffset: packedRowOffset,
            length: packedRowStride,
            context: "embedToken u4_lut packed row"
        )
            .assumingMemoryBound(to: UInt8.self)
        let lutBase = (tokenId / groupSize) * 16
        let lut = try weightPointer(
            logicalOffset: Int(lutOffset) + lutBase * MemoryLayout<UInt16>.stride,
            length: 16 * MemoryLayout<UInt16>.stride,
            context: "embedToken u4_lut table"
        )
            .assumingMemoryBound(to: UInt16.self)

        var out = [UInt16](repeating: 0, count: hiddenSize)
        for byteIndex in 0..<packedRowStride {
            let packedByte = packed[byteIndex]
            let col0 = byteIndex * 2
            if col0 < hiddenSize {
                out[col0] = lut[Int(packedByte & 0x0F)]
            }
            let col1 = col0 + 1
            if col1 < hiddenSize {
                out[col1] = lut[Int(packedByte >> 4)]
            }
        }

        return out.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    /// MTLBuffer length at `slotIndex` (post-`prepareForRequest`
    /// dynamic size, NOT the manifest's static upper bound). Tests
    /// that round-trip through `installSlotBytes` need this to
    /// size the payload correctly.
    public func bufferLength(at slotIndex: Int) -> Int? {
        guard slotIndex >= 0, slotIndex < buffers.count else {
            return nil
        }
        if slotIndex == Int(config.weightsSlot) {
            return weightBuffers.totalLogicalBytes
        }
        return buffers[slotIndex].length
    }

    private func makeAttentionTap(
        family: String,
        expandedPattern: [String],
        sharedKVLayers: Int
    ) throws -> SmeltDrafterAttentionTap? {
        guard let attnIdx = lastAttentionLayerIndex(
            forFamily: family,
            in: expandedPattern,
            sharedKVLayers: sharedKVLayers
        ) else {
            return nil
        }

        let keySlot = Int(config.keyCacheBaseSlot) + attnIdx
        let valSlot = Int(config.valCacheBaseSlot) + attnIdx
        guard keySlot < buffers.count, valSlot < buffers.count else {
            throw SmeltRuntimeError.invalidPackage(
                "\(family) K/V slots \(keySlot)/\(valSlot) out of range"
            )
        }

        guard let keySlotInfo = slotsByIndex[keySlot],
              keySlotInfo.shape.count == 3
        else {
            throw SmeltRuntimeError.invalidPackage(
                "keyCache slot \(keySlot) missing 3-D shape metadata"
            )
        }
        let kvHeads = keySlotInfo.shape[0]
        let headDim = keySlotInfo.shape[2]

        return SmeltDrafterAttentionTap(
            keyCache: buffers[keySlot],
            valueCache: buffers[valSlot],
            kvHeads: kvHeads,
            headDim: headDim,
            contextCapacity: currentContextCapacity
        )
    }

    /// Whether this runtime can run temperature sampling on the GPU.
    public var supportsGPUTemperatureSampling: Bool {
        pipelineLibrary.functionNames.contains("sample_temperature_fp16")
    }

    /// Whether this runtime's package metallib contains the opt-in K=3
    /// stochastic Block Verification kernel.
    public var supportsGPUBlockVerificationK3: Bool {
        pipelineLibrary.functionNames.contains("spec_bv_k3")
    }

    private func makeSharedRuntimeBuffer(length: Int, label: String) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(
            length: max(length, 16), options: .storageModeShared
        ) else {
            throw SmeltRuntimeError.invalidPackage(
                "failed to allocate \(label) (\(length) bytes)"
            )
        }
        memset(buffer.contents(), 0, buffer.length)
        buffer.label = label
        return buffer
    }

    private func ensureSpecBVTargetLogitsBuffer(rowCount: Int) throws -> MTLBuffer {
        let bytes = rowCount * vocabSize * MemoryLayout<Float16>.stride
        if let buffer = specBVTargetLogitsBuffer, buffer.length >= bytes {
            return buffer
        }
        let buffer = try makeSharedRuntimeBuffer(
            length: bytes, label: "smelt.spec_bv.target_logits"
        )
        specBVTargetLogitsBuffer = buffer
        return buffer
    }

    private func ensureSpecBVCandidatesBuffer() throws -> MTLBuffer {
        let bytes = 3 * MemoryLayout<Int32>.stride
        if let buffer = specBVCandidatesBuffer, buffer.length >= bytes {
            return buffer
        }
        let buffer = try makeSharedRuntimeBuffer(
            length: bytes, label: "smelt.spec_bv.candidates"
        )
        specBVCandidatesBuffer = buffer
        return buffer
    }

    private func ensureSpecBVOutputBuffer() throws -> MTLBuffer {
        let bytes = 2 * MemoryLayout<Int32>.stride
        if let buffer = specBVOutputBuffer, buffer.length >= bytes {
            return buffer
        }
        let buffer = try makeSharedRuntimeBuffer(
            length: bytes, label: "smelt.spec_bv.output"
        )
        specBVOutputBuffer = buffer
        return buffer
    }

    /// Sample the currently-live logits buffer on GPU with an arbitrary
    /// sampling position. Used by stochastic drafters whose decode position
    /// is frozen but whose sampling RNG position advances per drafted token.
    public func sampleCurrentLogitsGPU(
        position: Int32,
        selectionMode: SmeltSelectionMode
    ) throws -> Int32 {
        guard !selectionMode.usesArgmaxFastPath else {
            return buffers[Int(config.argmaxSlot)]
                .contents().load(as: Int32.self)
        }
        guard ensureTemperatureSamplingPipeline() != nil else {
            return selectCurrentToken(position: position, selectionMode: selectionMode)
        }
        guard let cmdBuf = queue.makeCommandBuffer() else {
            throw SmeltRuntimeError.metalCommandBufferUnavailable
        }
        let encoded = encodeSelectionKernelIfNeeded(
            commandBuffer: cmdBuf,
            position: position,
            selectionMode: selectionMode
        )
        guard encoded else {
            return selectCurrentToken(position: position, selectionMode: selectionMode)
        }
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        return buffers[Int(config.argmaxSlot)]
            .contents().load(as: Int32.self)
    }

    /// Copy the current single-row logits buffer into an external staging
    /// buffer. The copy is a Metal blit so stochastic spec-decode can keep
    /// full-vocab q rows out of Swift arrays.
    public func copyCurrentLogits(
        to destination: MTLBuffer,
        destinationOffset: Int
    ) throws {
        let byteCount = vocabSize * MemoryLayout<Float16>.stride
        guard destinationOffset >= 0,
              destinationOffset <= destination.length - byteCount
        else {
            throw SmeltRuntimeError.invalidPackage(
                "copyCurrentLogits destination range "
                + "[\(destinationOffset)..+\(byteCount)] exceeds "
                + "buffer length \(destination.length)"
            )
        }
        guard let cmdBuf = queue.makeCommandBuffer(),
              let blit = cmdBuf.makeBlitCommandEncoder()
        else {
            throw SmeltRuntimeError.metalCommandBufferUnavailable
        }
        blit.copy(
            from: buffers[Int(SmeltRuntimeConfig.fixedLogitsSlot)],
            sourceOffset: 0,
            to: destination,
            destinationOffset: destinationOffset,
            size: byteCount
        )
        blit.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    /// Select a token from the current logits buffer.
    public func selectCurrentToken(
        position: Int32,
        selectionMode: SmeltSelectionMode = .argmax,
        allowedTokenMask: [UInt32]? = nil
    ) -> Int32 {
        if selectionMode.usesArgmaxFastPath {
            if allowedTokenMask == nil {
                return buffers[Int(config.argmaxSlot)]
                    .contents().load(as: Int32.self)
            }
            // Speculative masked argmax: the dispatch table computes the
            // global GPU argmax regardless of the mask. When the global
            // top token is allowed it IS the masked argmax (it maximizes
            // the logits over a superset), so an O(1) bit test replaces
            // the CPU scan. Grammar-dense steps (e.g. inside JSON string
            // values) almost always take this path; the CPU fallback
            // below then only runs on sparse structural masks, where the
            // set-bit scan is cheap.
            if let mask = allowedTokenMask {
                let pick = buffers[Int(config.argmaxSlot)]
                    .contents().load(as: Int32.self)
                if SmeltLogitsSelector.isAllowed(pick, in: mask) {
                    return pick
                }
            }
        }

        let logitsBuf = buffers[Int(SmeltRuntimeConfig.fixedLogitsSlot)]
        let ptr = logitsBuf.contents().bindMemory(
            to: Float16.self,
            capacity: Int(config.vocabSize)
        )
        let logits = UnsafeBufferPointer(
            start: ptr,
            count: Int(config.vocabSize)
        )
        return SmeltLogitsSelector.select(
            logits: logits,
            position: position,
            mode: selectionMode,
            allowedTokenMask: allowedTokenMask
        )
    }

    private func encodeSelectionKernelIfNeeded(
        commandBuffer: MTLCommandBuffer,
        position: Int32,
        selectionMode: SmeltSelectionMode
    ) -> Bool {
        guard let params = selectionMode.gpuSamplingParameters,
              let pipeline = activeSamplingPipelineForSelection(),
              let enc = commandBuffer.makeComputeCommandEncoder()
        else {
            return false
        }

        let executionWidth = max(pipeline.threadExecutionWidth, 1)
        let maxThreads = max(pipeline.maxTotalThreadsPerThreadgroup, executionWidth)
        let preferredThreads = min(256, maxThreads)
        let threads = max(
            executionWidth,
            (preferredThreads / executionWidth) * executionWidth
        )

        var vocabSize = UInt32(config.vocabSize)
        var invTemperature = params.invTemperature
        var seed = params.seed
        var positionBits = UInt32(bitPattern: position)

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(buffers[Int(SmeltRuntimeConfig.fixedLogitsSlot)], offset: 0, index: 0)
        enc.setBuffer(buffers[Int(config.argmaxSlot)], offset: 0, index: 1)
        enc.setBytes(&vocabSize, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&invTemperature, length: MemoryLayout<Float>.size, index: 3)
        enc.setBytes(&seed, length: MemoryLayout<UInt64>.size, index: 4)
        enc.setBytes(&positionBits, length: MemoryLayout<UInt32>.size, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        enc.endEncoding()
        return true
    }

    /// Read top-K logits from the logits buffer after a decode/prefill step.
    /// Returns `(tokenId, logit)` pairs sorted by logit descending.
    public func topKLogits(k: Int = 5) -> [(Int32, Float)] {
        let logitsBuf = buffers[Int(SmeltRuntimeConfig.fixedLogitsSlot)]
        let ptr = logitsBuf.contents().bindMemory(
            to: Float16.self,
            capacity: Int(config.vocabSize)
        )
        var topK: [(Int32, Float)] = []
        for i in 0..<Int(config.vocabSize) {
            let val = Float(ptr[i])
            if topK.count < k {
                topK.append((Int32(i), val))
                topK.sort { $0.1 > $1.1 }
            } else if val > topK.last!.1 {
                topK[topK.count - 1] = (Int32(i), val)
                topK.sort { $0.1 > $1.1 }
            }
        }
        return topK
    }

    /// Write fp32 values into a slot buffer — the embeddings-input port's
    /// manual entry for FP32 activation ports (also
    /// lets parity gates pin table contents, e.g. RoPE rows).
    public func writeSlot(_ slot: Int, f32 values: [Float], elementOffset: Int = 0) {
        let buf = buffers[slot]
        let bytes = values.count * MemoryLayout<Float>.stride
        precondition(elementOffset * 4 + bytes <= buf.length,
                     "writeSlot [\(elementOffset), +\(values.count)) overruns slot \(slot)")
        values.withUnsafeBytes { src in
            _ = memcpy(buf.contents().advanced(by: elementOffset * 4), src.baseAddress!, bytes)
        }
    }

    /// Write FP32 values narrowed to BF16 into a BF16 activation slot.
    public func writeSlot(_ slot: Int, bf16 values: [Float], elementOffset: Int = 0) {
        let buffer = buffers[slot]
        let bits = SmeltBF16.encode(values)
        let byteOffset = elementOffset * MemoryLayout<UInt16>.stride
        let byteCount = bits.count * MemoryLayout<UInt16>.stride
        precondition(
            byteOffset + byteCount <= buffer.length,
            "writeSlot bf16 [\(elementOffset), +\(values.count)) overruns slot \(slot)"
        )
        bits.withUnsafeBytes { source in
            _ = memcpy(
                buffer.contents().advanced(by: byteOffset),
                source.baseAddress!,
                byteCount
            )
        }
    }

    private func writeSlotFloat16(
        _ slot: Int,
        values: ArraySlice<Float16>,
        elementOffset: Int = 0,
        label: String
    ) throws {
        guard slot >= 0, slot < buffers.count else {
            throw SmeltRuntimeError.invalidPackage(
                "\(label): slot \(slot) is outside the runtime buffer table"
            )
        }
        let buf = buffers[slot]
        let byteOffset = elementOffset * MemoryLayout<Float16>.stride
        let bytes = values.count * MemoryLayout<Float16>.stride
        guard elementOffset >= 0, byteOffset <= buf.length,
              bytes <= buf.length - byteOffset
        else {
            throw SmeltRuntimeError.invalidPackage(
                "\(label): [\(elementOffset), +\(values.count)) overruns "
                    + "slot \(slot) (\(buf.length / 2) fp16 values)"
            )
        }
        let copied = values.withContiguousStorageIfAvailable { src in
            guard let base = src.baseAddress else { return }
            memcpy(buf.contents().advanced(by: byteOffset), base, bytes)
        } != nil
        guard copied else {
            throw SmeltRuntimeError.invalidPackage(
                "\(label): fp16 source was not contiguous"
            )
        }
    }

    /// Read `count` values from a buffer slot starting at `elementOffset`.
    /// `asFP32=true` interprets the slot as `Float`; `asBF16=true` widens
    /// BF16 bits; otherwise the slot is interpreted as `Float16`.
    public func dumpSlot(
        _ slot: Int,
        elementOffset: Int = 0,
        count: Int = 10,
        asFP32: Bool = false,
        asBF16: Bool = false
    ) -> [Float] {
        let buf = buffers[slot]
        if asFP32 {
            let ptr = buf.contents().bindMemory(
                to: Float.self,
                capacity: elementOffset + count
            )
            return (0..<count).map { ptr[elementOffset + $0] }
        } else if asBF16 {
            let pointer = buf.contents().bindMemory(
                to: UInt16.self,
                capacity: elementOffset + count
            )
            return SmeltBF16.decode(pointer.advanced(by: elementOffset), count: count)
        } else {
            let ptr = buf.contents().bindMemory(
                to: Float16.self,
                capacity: elementOffset + count
            )
            return (0..<count).map { Float(ptr[elementOffset + $0]) }
        }
    }

    /// Read the full logits buffer as `Float`s after a decode/prefill step.
    public func allLogits() -> [Float] {
        let logitsBuf = buffers[Int(SmeltRuntimeConfig.fixedLogitsSlot)]
        let ptr = logitsBuf.contents().bindMemory(
            to: Float16.self,
            capacity: Int(config.vocabSize)
        )
        return (0..<Int(config.vocabSize)).map { Float(ptr[$0]) }
    }

    /// Read the full logits buffer as `Float16`s, skipping the
    /// fp32 round-trip in `allLogits` for callers that consume
    /// fp16 directly (e.g., the speculative-decode drafter, which
    /// stashes per-step logits in `SmeltDrafterQ.dense` as
    /// `[[Float16]]`). Logits are stored fp16 in the buffer; this
    /// is one stride-copy of `vocabSize * 2` bytes instead of two
    /// vocab-size allocations and a precision round-trip.
    public func allLogitsHalf() -> [Float16] {
        let logitsBuf = buffers[Int(SmeltRuntimeConfig.fixedLogitsSlot)]
        let ptr = logitsBuf.contents().bindMemory(
            to: Float16.self,
            capacity: Int(config.vocabSize)
        )
        return Array(UnsafeBufferPointer(
            start: ptr, count: Int(config.vocabSize)
        ))
    }

    /// Run K+1 sequential decode steps for the speculative-decode
    /// verify path. Each step writes K/V at its position and
    /// produces the target's logit distribution one token ahead;
    /// the returned rows are stored fp16 (`[[Float16]]`) so the
    /// rejection sampler compares against drafter logits in the
    /// same precision.
    ///
    /// Inputs:
    ///   - `tokens`: the K+1 token sequence the drafter proposed —
    ///     `[lastCommittedToken, candidate_0, …, candidate_{K-1}]`.
    ///   - `startPosition`: position of `tokens[0]`. Each successive
    ///     token decodes at `startPosition + i`.
    ///
    /// Output: `tokens.count` logit rows. Row `i` is the target's
    /// distribution at `startPosition + i + 1` — the position whose
    /// token follows `tokens[i]`. The rejection sampler uses row
    /// `k` to verify `candidate_k`; row K is the bonus distribution
    /// sampled from after a full accept.
    ///
    /// V1: sequential decodes (K+1 GPU-sync round-trips). The
    /// chunked-prefill verify path emits logits only at the LAST
    /// position (PrefillEmitter.swift:506), which doesn't fit
    /// rejection sampling — a per-position-logits prefill kernel
    /// is the follow-up optimization.
    ///
    /// Side effect: writes K/V at positions `[startPosition,
    /// startPosition + tokens.count - 1]`. After the rejection
    /// sampler decides a commit point j, the orchestrator must
    /// re-decode at `startPosition + j` with the accepted/replaced
    /// token to overwrite its K/V. K/V positions strictly above
    /// the commit boundary stay populated with verify-time values;
    /// since subsequent decodes only read `[0..position]` (their
    /// own position), this is harmless. Architectures with cross-
    /// position state would break this assumption — see the
    /// `numDeltaLayers` guard above.
    public func verifyDraft(
        tokens: [Int32],
        startPosition: Int32,
        firstRowFromLogitsBuf: Bool = false
    ) throws -> [[Float16]] {
        // Recurrent (DeltaNet) layers carry per-step state that
        // isn't position-indexed — verifyDraft would advance
        // convState/recState through the proposed suffix and the
        // K/V-only re-decode at the commit boundary couldn't undo
        // those updates. Snapshot/restore lands as a follow-up; v1
        // refuses to verify against recurrent-state packages so
        // partial-accept rejection can't silently corrupt the
        // target's recurrent trajectory.
        if manifest.config.numDeltaLayers > 0 {
            throw SmeltRuntimeError.invalidPackage(
                "verifyDraft: target has \(manifest.config.numDeltaLayers) "
                + "recurrent layers; speculative-decode verify is unsafe "
                + "without convState/recState snapshot+restore (follow-up)"
            )
        }
        // Validate every proposed token before any decode work.
        // decodeStep's tokenId precondition traps the process on
        // out-of-range ids; a buggy / mismatched drafter would
        // therefore abort instead of letting the speculative
        // runtime surface a recoverable mismatch error. Throw
        // invalidPackage with the offending value so callers can
        // fall back cleanly.
        let vocab = manifest.config.vocabSize
        for tok in tokens {
            guard tok >= 0, tok < vocab else {
                throw SmeltRuntimeError.invalidPackage(
                    "verifyDraft: proposed token \(tok) out of vocab "
                    + "range [0, \(vocab))"
                )
            }
        }
        var rows: [[Float16]] = []
        rows.reserveCapacity(tokens.count)
        for (i, tok) in tokens.enumerated() {
            if i == 0 && firstRowFromLogitsBuf {
                // Caller asserts that `tokens[0]` was already
                // decoded at `startPosition` and the row-0 logits
                // are still live in logitsBuf. K/V[startPosition]
                // is also already populated; we can skip straight
                // to decoding `tokens[1..]`.
                rows.append(allLogitsHalf())
                continue
            }
            let pos = startPosition + Int32(i)
            _ = try decodeStep(tokenId: tok, position: pos)
            rows.append(allLogitsHalf())
        }
        return rows
    }

    // MARK: - Dispatch table interpreter

    @inline(__always)
    private func bindBuffer(
        _ encoder: MTLComputeCommandEncoder,
        slot: Int,
        offset: Int,
        index: Int
    ) {
        if slot == Int(config.weightsSlot) {
            guard let resolved = weightBuffers.resolve(logicalOffset: offset) else {
                preconditionFailure(
                    "weight offset \(offset) is outside the mapped weight segments"
                )
            }
            encoder.setBuffer(resolved.buffer, offset: resolved.offset, index: index)
        } else {
            encoder.setBuffer(buffers[slot], offset: offset, index: index)
        }
    }

    private func prepareDecodeDynamicState(tokenId: Int32, position: Int32) {
        buffers[Int(config.tokenIdSlot)].contents()
            .storeBytes(of: tokenId, as: Int32.self)
        buffers[Int(config.positionSlot)].contents()
            .storeBytes(of: position, as: Int32.self)
        precondition(
            position >= 0 && position < config.contextLimit,
            "position out of range [0, \(config.contextLimit))"
        )
        precondition(
            tokenId >= 0 && tokenId < config.vocabSize,
            "tokenId out of range [0, \(config.vocabSize))"
        )
    }

    @discardableResult
    private func emitDecodeDispatchRecord(
        _ rec: SmeltDispatchRecord,
        enc: MTLComputeCommandEncoder,
        cur: Int,
        alt: Int,
        position: Int32,
        pipelineOverrides: [UInt16: MTLComputePipelineState] = [:]
    ) -> Bool {
        guard rec.opKind == SmeltDispatchRecord.opDispatch else { return false }
        if let override = pipelineOverrides[rec.pipeline] {
            enc.setComputePipelineState(override)
        } else {
            enc.setComputePipelineState(pipelineState(for: Int(rec.pipeline)))
        }
        for bufferIndex in 0..<Int(rec.bufferCount) {
            let buffer = getBuffer(rec, index: bufferIndex)
            let slot = resolveDispatchBufferSlot(buffer, cur: cur, alt: alt)
            let offset = buffer.offsetKind == 0
                ? Int(buffer.offset)
                : Int(position) * Int(buffer.offset)
            bindBuffer(
                enc,
                slot: slot,
                offset: offset,
                index: Int(buffer.bindingIndex)
            )
        }
        var skipDispatch = false
        for constantIndex in 0..<Int(rec.constantCount) {
            let constant = getConstant(rec, index: constantIndex)
            var value = resolveDecodeConstant(
                constant,
                position: position,
                skipDispatch: &skipDispatch
            )
            if constant.bindingIndex != UInt8.max {
                enc.setBytes(
                    &value,
                    length: MemoryLayout<UInt32>.size,
                    index: Int(constant.bindingIndex)
                )
            }
        }
        guard !skipDispatch else { return false }
        let grid = MTLSize(
            width: Int(rec.gridW),
            height: Int(rec.gridH),
            depth: Int(rec.gridD)
        )
        let threadgroup = MTLSize(
            width: Int(rec.tgW),
            height: Int(rec.tgH),
            depth: Int(rec.tgD)
        )
        if rec.style == SmeltDispatchRecord.styleThreadgroups {
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
        } else {
            enc.dispatchThreads(grid, threadsPerThreadgroup: threadgroup)
        }
        return true
    }

    /// Interpret the binary dispatch table into Metal compute commands.
    /// No allocations, no heap, no ARC. Pure scalar loop over POD records.
    private func interpretDispatchTable(
        _ table: UnsafeBufferPointer<SmeltDispatchRecord>,
        enc: MTLComputeCommandEncoder,
        tokenId: Int32,
        position: Int32,
        pipelineOverrides: [UInt16: MTLComputePipelineState] = [:]
    ) {
        var current = 0
        var alternate = 1
        prepareDecodeDynamicState(tokenId: tokenId, position: position)
        for record in table {
            if record.opKind == SmeltDispatchRecord.opSwap {
                swap(&current, &alternate)
            } else {
                emitDecodeDispatchRecord(
                    record,
                    enc: enc,
                    cur: current,
                    alt: alternate,
                    position: position,
                    pipelineOverrides: pipelineOverrides
                )
            }
        }
    }

    /// Profile decode: splits CPU encode vs GPU execute vs readback.
    /// Returns (token, cpuMs, gpuMs, readMs).
    public func profileDecodeStep(
        tokenId: Int32, position: Int32
    ) throws -> (token: Int32, cpuMs: Double, gpuMs: Double, readMs: Double, pureGpuMs: Double) {
        try ensureContextCapacity(Int(position) + 1)

        let t0 = CFAbsoluteTimeGetCurrent()

        guard let table = dispatchTable else {
            throw SmeltRuntimeError.dispatchTableMissing(
                "no dispatches.bin — cannot profile"
            )
        }
        try materializeAllPipelineStates()
        let (cmdBuf, enc) = try makeCommandBufferAndEncoder()
        interpretDispatchTable(
            table, enc: enc, tokenId: tokenId, position: position
        )

        enc.endEncoding()
        let t1 = CFAbsoluteTimeGetCurrent()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        let t2 = CFAbsoluteTimeGetCurrent()

        let token = buffers[Int(config.argmaxSlot)]
            .contents().load(as: Int32.self)
        let t3 = CFAbsoluteTimeGetCurrent()

        // Pure GPU time from Metal timestamps (excludes CPU→GPU submission latency)
        let pureGpu = (cmdBuf.gpuEndTime - cmdBuf.gpuStartTime) * 1_000

        return (
            token: token,
            cpuMs: (t1 - t0) * 1_000,
            gpuMs: (t2 - t1) * 1_000,
            readMs: (t3 - t2) * 1_000,
            pureGpuMs: pureGpu
        )
    }

    /// Profile one decode step using an in-memory dispatch table.
    ///
    /// This is intentionally a lab/probe surface: it reuses the exact runtime
    /// interpreter, buffers, pipelines, guards, swaps, and dynamic constants,
    /// while allowing tooling to rewrite `dispatches.bin` records without
    /// rebuilding a package.
    public func profileDecodeStep(
        tokenId: Int32,
        position: Int32,
        dispatchRecords: [SmeltDispatchRecord],
        pipelineOverrides: [UInt16: MTLComputePipelineState] = [:]
    ) throws -> (token: Int32, cpuMs: Double, gpuMs: Double, readMs: Double, pureGpuMs: Double) {
        try ensureContextCapacity(Int(position) + 1)

        let t0 = CFAbsoluteTimeGetCurrent()

        try dispatchRecords.withUnsafeBufferPointer { table in
            guard !table.isEmpty else {
                throw SmeltRuntimeError.dispatchTableMissing(
                    "empty override dispatch table — cannot profile"
                )
            }
            try ensurePipelineStatesMaterialized(
                for: table,
                tableName: "override dispatch records"
            )
        }

        let (cmdBuf, enc) = try makeCommandBufferAndEncoder()
        dispatchRecords.withUnsafeBufferPointer { table in
            interpretDispatchTable(
                table,
                enc: enc,
                tokenId: tokenId,
                position: position,
                pipelineOverrides: pipelineOverrides
            )
        }

        enc.endEncoding()
        let t1 = CFAbsoluteTimeGetCurrent()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        let t2 = CFAbsoluteTimeGetCurrent()

        if let error = cmdBuf.error {
            throw SmeltRuntimeError.metalDispatchFailed(error)
        }

        let token = buffers[Int(config.argmaxSlot)]
            .contents().load(as: Int32.self)
        let t3 = CFAbsoluteTimeGetCurrent()

        let pureGpu = (cmdBuf.gpuEndTime - cmdBuf.gpuStartTime) * 1_000

        return (
            token: token,
            cpuMs: (t1 - t0) * 1_000,
            gpuMs: (t2 - t1) * 1_000,
            readMs: (t3 - t2) * 1_000,
            pureGpuMs: pureGpu
        )
    }

    // MARK: - Per-kernel GPU profiling

    /// Per-kernel GPU timing result.
    public struct KernelProfile {
        public let name: String
        public let dispatchCount: Int
        public let totalGpuUs: Double
        public let avgGpuUs: Double
        public let pctOfTotal: Double
    }

    public struct DispatchBenchmarkProfile {
        public let name: String
        public let dispatchOrdinal: Int
        public let recordIndex: Int
        public let medianGpuUs: Double
        public let p95GpuUs: Double
        public let avgGpuUs: Double
        public let minGpuUs: Double
        public let maxGpuUs: Double
        public let samplesGpuUs: [Double]
    }

    private struct ResolvedDecodeDispatch {
        let recordIndex: Int
        let record: SmeltDispatchRecord
        let cur: Int
        let alt: Int
    }

    private func resolveDecodeDispatch(
        ordinal: Int
    ) -> ResolvedDecodeDispatch? {
        guard ordinal > 0, let table = dispatchTable else {
            return nil
        }

        var cur = 0
        var alt = 1
        var seenDispatches = 0
        for idx in 0..<table.count {
            let rec = table[idx]
            if rec.opKind == SmeltDispatchRecord.opSwap {
                let tmp = cur
                cur = alt
                alt = tmp
                continue
            }

            guard rec.opKind == SmeltDispatchRecord.opDispatch else { continue }

            seenDispatches += 1
            if seenDispatches == ordinal {
                return ResolvedDecodeDispatch(
                    recordIndex: idx,
                    record: rec,
                    cur: cur,
                    alt: alt
                )
            }
        }
        return nil
    }

    @discardableResult
    private func encodeResolvedDecodeDispatch(
        _ resolved: ResolvedDecodeDispatch,
        encoder enc: MTLComputeCommandEncoder,
        tokenId: Int32,
        position: Int32
    ) -> Bool {
        buffers[Int(config.tokenIdSlot)].contents()
            .storeBytes(of: tokenId, as: Int32.self)
        buffers[Int(config.positionSlot)].contents()
            .storeBytes(of: position, as: Int32.self)

        let rec = resolved.record
        enc.setComputePipelineState(pipelineState(for: Int(rec.pipeline)))

        for bidx in 0..<Int(rec.bufferCount) {
            let buf = getBuffer(rec, index: bidx)
            let slot: Int
            if buf.slot >= 0 {
                slot = Int(buf.slot)
            } else if buf.slot == SmeltBufferRecord.slotCur {
                slot = resolved.cur
            } else {
                slot = resolved.alt
            }
            let offset: Int
            if buf.offsetKind == 0 {
                offset = Int(buf.offset)
            } else {
                offset = Int(position) * Int(buf.offset)
            }
            bindBuffer(enc, slot: slot, offset: offset, index: Int(buf.bindingIndex))
        }

        var skipDispatch = false
        for cidx in 0..<Int(rec.constantCount) {
            let con = getConstant(rec, index: cidx)
            var val = resolveDecodeConstant(
                con,
                position: position,
                skipDispatch: &skipDispatch
            )
            if con.bindingIndex != UInt8.max {
                enc.setBytes(&val, length: 4, index: Int(con.bindingIndex))
            }
        }

        if skipDispatch {
            return false
        }

        if rec.style == SmeltDispatchRecord.styleThreadgroups {
            enc.dispatchThreadgroups(
                MTLSize(
                    width: Int(rec.gridW),
                    height: Int(rec.gridH),
                    depth: Int(rec.gridD)
                ),
                threadsPerThreadgroup: MTLSize(
                    width: Int(rec.tgW),
                    height: Int(rec.tgH),
                    depth: Int(rec.tgD)
                )
            )
        } else {
            enc.dispatchThreads(
                MTLSize(
                    width: Int(rec.gridW),
                    height: Int(rec.gridH),
                    depth: Int(rec.gridD)
                ),
                threadsPerThreadgroup: MTLSize(
                    width: Int(rec.tgW),
                    height: Int(rec.tgH),
                    depth: Int(rec.tgD)
                )
            )
        }

        return true
    }

    public func benchmarkDecodeDispatch(
        tokenId: Int32,
        position: Int32,
        dispatchOrdinal: Int,
        warmup: Int = 3,
        iterations: Int = 20
    ) throws -> DispatchBenchmarkProfile? {
        try ensureContextCapacity(Int(position) + 1)

        guard iterations > 0,
              let resolved = resolveDecodeDispatch(ordinal: dispatchOrdinal)
        else {
            return nil
        }
        try materializeAllPipelineStates()

        let totalRuns = warmup + iterations
        var samplesUs: [Double] = []
        samplesUs.reserveCapacity(iterations)

        for run in 0..<totalRuns {
            let (cmdBuf, enc) = try makeCommandBufferAndEncoder()

            let shouldDispatch = encodeResolvedDecodeDispatch(
                resolved,
                encoder: enc,
                tokenId: tokenId,
                position: position
            )
            enc.endEncoding()

            guard shouldDispatch else {
                return nil
            }

            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            if let err = cmdBuf.error {
                throw SmeltRuntimeError.metalDispatchFailed(err)
            }

            if run >= warmup {
                let gpuUs = (cmdBuf.gpuEndTime - cmdBuf.gpuStartTime) * 1_000_000
                samplesUs.append(gpuUs)
            }
        }

        guard !samplesUs.isEmpty else {
            return nil
        }

        let sorted = samplesUs.sorted()
        let median = sorted[sorted.count / 2]
        let p95Index = min(
            max(Int(ceil(Double(sorted.count) * 0.95)) - 1, 0),
            sorted.count - 1
        )
        let p95 = sorted[p95Index]
        let avg = samplesUs.reduce(0, +) / Double(samplesUs.count)
        let minUs = sorted.first ?? 0
        let maxUs = sorted.last ?? 0
        let name = Int(resolved.record.pipeline) < manifest.pipelines.count
            ? manifest.pipelines[Int(resolved.record.pipeline)]
            : "pipeline_\(resolved.record.pipeline)"

        return DispatchBenchmarkProfile(
            name: name,
            dispatchOrdinal: dispatchOrdinal,
            recordIndex: resolved.recordIndex,
            medianGpuUs: median,
            p95GpuUs: p95,
            avgGpuUs: avg,
            minGpuUs: minUs,
            maxGpuUs: maxUs,
            samplesGpuUs: samplesUs
        )
    }

    private struct ResolvedPrefillDispatch {
        let recordIndex: Int
        let record: SmeltDispatchRecord
        let cur: Int
        let alt: Int
    }

    private func resolvePrefillDispatch(
        ordinal: Int
    ) -> ResolvedPrefillDispatch? {
        guard ordinal > 0, let table = prefillDispatchTable else {
            return nil
        }

        var cur = 0
        var alt = 1
        var seenDispatches = 0
        for idx in 0..<table.count {
            let rec = table[idx]
            if rec.opKind == SmeltDispatchRecord.opSwap {
                swap(&cur, &alt)
                continue
            }
            guard rec.opKind == SmeltDispatchRecord.opDispatch else { continue }
            seenDispatches += 1
            if seenDispatches == ordinal {
                return ResolvedPrefillDispatch(
                    recordIndex: idx,
                    record: rec,
                    cur: cur,
                    alt: alt
                )
            }
        }
        return nil
    }

    /// Benchmark one dispatch from the package's normal prefill table at the
    /// exact dynamic sequence geometry used by the full-plan prefill path.
    /// This is a profiling surface: the command buffer intentionally contains
    /// only the selected dispatch so frozen-plan tooling can calibrate one
    /// stable dispatch-cost key at a time.
    public func benchmarkPrefillDispatch(
        tokenIds: [Int32],
        startPos: Int32,
        dispatchOrdinal: Int,
        warmup: Int = 3,
        iterations: Int = 20
    ) throws -> DispatchBenchmarkProfile? {
        try ensurePrefillCapacity(tokenIds: tokenIds, startPos: startPos)
        guard iterations > 0,
              let resolved = resolvePrefillDispatch(ordinal: dispatchOrdinal),
              let table = prefillDispatchTable
        else { return nil }
        try ensurePipelineStatesMaterialized(
            for: table,
            tableName: "prefill_dispatches.bin"
        )
        writePrefillTokenIds(tokenIds)

        let totalRuns = max(warmup, 0) + iterations
        var samplesUs: [Double] = []
        samplesUs.reserveCapacity(iterations)
        let seqLen = Int32(tokenIds.count)

        for run in 0..<totalRuns {
            let (cmdBuf, enc) = try makeCommandBufferAndEncoder()
            let shouldDispatch = emitDispatchRecord(
                resolved.record,
                enc: enc,
                cur: resolved.cur,
                alt: resolved.alt,
                seqLen: seqLen,
                startPos: startPos
            )
            enc.endEncoding()
            guard shouldDispatch else { return nil }

            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error {
                throw SmeltRuntimeError.metalDispatchFailed(error)
            }
            if run >= max(warmup, 0) {
                samplesUs.append(
                    (cmdBuf.gpuEndTime - cmdBuf.gpuStartTime) * 1_000_000
                )
            }
        }

        guard !samplesUs.isEmpty else { return nil }
        let sorted = samplesUs.sorted()
        let p95Index = min(
            max(Int(ceil(Double(sorted.count) * 0.95)) - 1, 0),
            sorted.count - 1
        )
        let pipeline = Int(resolved.record.pipeline) < manifest.pipelines.count
            ? manifest.pipelines[Int(resolved.record.pipeline)]
            : "pipeline_\(resolved.record.pipeline)"
        return DispatchBenchmarkProfile(
            name: pipeline,
            dispatchOrdinal: dispatchOrdinal,
            recordIndex: resolved.recordIndex,
            medianGpuUs: sorted[sorted.count / 2],
            p95GpuUs: sorted[p95Index],
            avgGpuUs: samplesUs.reduce(0, +) / Double(samplesUs.count),
            minGpuUs: sorted.first ?? 0,
            maxGpuUs: sorted.last ?? 0,
            samplesGpuUs: samplesUs
        )
    }

    /// Profile one decode step with per-kernel GPU timing.
    ///
    /// Uses one command buffer per dispatch with gpuStartTime/gpuEndTime
    /// to get per-kernel GPU timestamps. Profiling mode only.
    public func profileKernels(
        tokenId: Int32, position: Int32
    ) throws -> [KernelProfile] {
        try ensureContextCapacity(Int(position) + 1)

        guard let tableData = dispatchTableData else {
            throw SmeltRuntimeError.dispatchTableMissing(
                "no dispatches.bin — cannot profile kernels"
            )
        }
        if dispatchTable != nil { try materializeAllPipelineStates() }

        // Access dispatch table safely within withUnsafeBytes
        let stride = MemoryLayout<SmeltDispatchRecord>.stride
        let tableCount = tableData.count / stride

        var cur = 0
        var alt = 1
        buffers[Int(config.tokenIdSlot)].contents()
            .storeBytes(of: tokenId, as: Int32.self)
        buffers[Int(config.positionSlot)].contents()
            .storeBytes(of: position, as: Int32.self)

        // Aggregate by pipeline
        var totalUs: [UInt16: Double] = [:]
        var counts: [UInt16: Int] = [:]

        tableData.withUnsafeBytes { ptr in
            let table = ptr.bindMemory(to: SmeltDispatchRecord.self)
        for idx in 0..<tableCount {
            let rec = table[idx]

            if rec.opKind == SmeltDispatchRecord.opSwap {
                let tmp = cur; cur = alt; alt = tmp
                continue
            }

            guard let cmdBuf = queue.makeCommandBuffer(),
                  let enc = cmdBuf.makeComputeCommandEncoder()
            else { continue }

            enc.setComputePipelineState(pipelineState(for: Int(rec.pipeline)))

            for bidx in 0..<Int(rec.bufferCount) {
                let buf = getBuffer(rec, index: bidx)
                let slot: Int
                if buf.slot >= 0 { slot = Int(buf.slot) }
                else if buf.slot == SmeltBufferRecord.slotCur { slot = cur }
                else { slot = alt }
                let offset: Int
                if buf.offsetKind == 0 { offset = Int(buf.offset) }
                else { offset = Int(position) * Int(buf.offset) }
                bindBuffer(enc, slot: slot, offset: offset, index: Int(buf.bindingIndex))
            }

            var skipDispatch = false
            for cidx in 0..<Int(rec.constantCount) {
                let con = getConstant(rec, index: cidx)
                var val = resolveDecodeConstant(
                    con,
                    position: position,
                    skipDispatch: &skipDispatch
                )
                if con.bindingIndex != UInt8.max {
                    enc.setBytes(&val, length: 4, index: Int(con.bindingIndex))
                }
            }

            if skipDispatch {
                enc.endEncoding()
                continue
            }

            if rec.style == SmeltDispatchRecord.styleThreadgroups {
                enc.dispatchThreadgroups(
                    MTLSize(width: Int(rec.gridW), height: Int(rec.gridH), depth: Int(rec.gridD)),
                    threadsPerThreadgroup: MTLSize(width: Int(rec.tgW), height: Int(rec.tgH), depth: Int(rec.tgD))
                )
            } else {
                enc.dispatchThreads(
                    MTLSize(width: Int(rec.gridW), height: Int(rec.gridH), depth: Int(rec.gridD)),
                    threadsPerThreadgroup: MTLSize(width: Int(rec.tgW), height: Int(rec.tgH), depth: Int(rec.tgD))
                )
            }

            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            if let err = cmdBuf.error {
                fputs("  dispatch \(idx) error: \(err)\n", stderr)
                continue
            }
            let gpuUs = (cmdBuf.gpuEndTime - cmdBuf.gpuStartTime) * 1_000_000
            let pipe = rec.pipeline
            totalUs[pipe, default: 0] += gpuUs
            counts[pipe, default: 0] += 1
        }
        } // end withUnsafeBytes

        let grandTotalUs = totalUs.values.reduce(0, +)

        var results: [KernelProfile] = []
        for (pipe, us) in totalUs {
            let name = Int(pipe) < manifest.pipelines.count
                ? manifest.pipelines[Int(pipe)]
                : "pipeline_\(pipe)"
            let count = counts[pipe, default: 0]
            results.append(KernelProfile(
                name: name,
                dispatchCount: count,
                totalGpuUs: us,
                avgGpuUs: us / Double(count),
                pctOfTotal: grandTotalUs > 0 ? (us / grandTotalUs) * 100 : 0
            ))
        }
        results.sort { $0.totalGpuUs > $1.totalGpuUs }
        return results
    }

    /// Profile each prefill dispatch independently using Metal GPU timestamps.
    /// Uses one command buffer per dispatch to attribute time to kernels.
    public func profilePrefillKernels(
        tokenIds: [Int32], startPos: Int32
    ) throws -> [KernelProfile] {
        try ensurePrefillCapacity(tokenIds: tokenIds, startPos: startPos)

        guard let tableData = prefillDispatchTableData else {
            throw SmeltRuntimeError.dispatchTableMissing(
                "no prefill_dispatches.bin — cannot profile prefill kernels"
            )
        }

        let stride = MemoryLayout<SmeltDispatchRecord>.stride
        let tableCount = tableData.count / stride
        let seqLen = Int32(tokenIds.count)

        var cur = 0
        var alt = 1
        writePrefillTokenIds(tokenIds)

        var totalUs: [UInt16: Double] = [:]
        var counts: [UInt16: Int] = [:]

        tableData.withUnsafeBytes { ptr in
            let table = ptr.bindMemory(to: SmeltDispatchRecord.self)
            for idx in 0..<tableCount {
                let rec = table[idx]

                if rec.opKind == SmeltDispatchRecord.opSwap {
                    let tmp = cur
                    cur = alt
                    alt = tmp
                    continue
                }

                if rec.minSeqLen > 0 && UInt32(seqLen) < UInt32(rec.minSeqLen) {
                    continue
                }

                guard let cmdBuf = queue.makeCommandBuffer(),
                      let enc = cmdBuf.makeComputeCommandEncoder()
                else { continue }

                enc.setComputePipelineState(pipelineState(for: Int(rec.pipeline)))

                for bidx in 0..<Int(rec.bufferCount) {
                    let buf = getBuffer(rec, index: bidx)
                    let slot: Int
                    if buf.slot >= 0 {
                        slot = Int(buf.slot)
                    } else if buf.slot == SmeltBufferRecord.slotCur {
                        slot = cur
                    } else {
                        slot = alt
                    }

                    let offset = resolveDispatchBufferOffset(
                        buf, seqLen: seqLen, startPos: startPos)
                    bindBuffer(
                        enc, slot: slot, offset: offset, index: Int(buf.bindingIndex)
                    )
                }

                var skipDispatch = false
                for cidx in 0..<Int(rec.constantCount) {
                    let con = getConstant(rec, index: cidx)
                    var val = resolvePrefillConstant(
                        con,
                        seqLen: seqLen,
                        startPos: startPos,
                        skipDispatch: &skipDispatch
                    )
                    if con.bindingIndex != UInt8.max {
                        enc.setBytes(&val, length: 4, index: Int(con.bindingIndex))
                    }
                }

                if skipDispatch {
                    enc.endEncoding()
                    continue
                }

                let gridW = resolvePrefillGrid(rec.gridW, kind: rec.gridWKind, seqLen: seqLen)
                let gridH = resolvePrefillGrid(rec.gridH, kind: rec.gridHKind, seqLen: seqLen)
                let gridD = resolvePrefillGrid(rec.gridD, kind: rec.gridDKind, seqLen: seqLen)
                if gridW == 0 || gridH == 0 || gridD == 0 {
                    enc.endEncoding()
                    continue
                }

                if rec.dispatchStyle == SmeltDispatchRecord.styleThreadgroups {
                    enc.dispatchThreadgroups(
                        MTLSize(
                            width: gridW,
                            height: gridH,
                            depth: gridD
                        ),
                        threadsPerThreadgroup: MTLSize(
                            width: Int(rec.tgW), height: Int(rec.tgH),
                            depth: Int(rec.tgD)
                        )
                    )
                } else {
                    enc.dispatchThreads(
                        MTLSize(
                            width: gridW,
                            height: gridH,
                            depth: gridD
                        ),
                        threadsPerThreadgroup: MTLSize(
                            width: Int(rec.tgW), height: Int(rec.tgH),
                            depth: Int(rec.tgD)
                        )
                    )
                }

                enc.endEncoding()
                cmdBuf.commit()
                cmdBuf.waitUntilCompleted()

                if let err = cmdBuf.error {
                    fputs("  prefill dispatch \(idx) error: \(err)\n", stderr)
                    continue
                }
                let gpuUs = (cmdBuf.gpuEndTime - cmdBuf.gpuStartTime) * 1_000_000
                let pipe = rec.pipeline
                totalUs[pipe, default: 0] += gpuUs
                counts[pipe, default: 0] += 1
            }
        }

        let grandTotalUs = totalUs.values.reduce(0, +)

        var results: [KernelProfile] = []
        for (pipe, us) in totalUs {
            let name = Int(pipe) < manifest.pipelines.count
                ? manifest.pipelines[Int(pipe)]
                : "pipeline_\(pipe)"
            let count = counts[pipe, default: 0]
            results.append(KernelProfile(
                name: name,
                dispatchCount: count,
                totalGpuUs: us,
                avgGpuUs: us / Double(count),
                pctOfTotal: grandTotalUs > 0 ? (us / grandTotalUs) * 100 : 0
            ))
        }
        results.sort { $0.totalGpuUs > $1.totalGpuUs }
        return results
    }

    // MARK: - Metal prefill

    /// Whether this package has a Metal prefill dispatch table.
    public var hasMetalPrefill: Bool { prefillDispatchTable != nil }

    /// Maximum token batch supported by the Metal prefill dispatch table.
    public var maxPrefillBatchSize: Int { Int(config.maxPrefillBatchSize) }

    /// Active invocation context limit. Dynamic runtimes use the caller's value
    /// or the largest representable position, not a package default.
    public var maxContextTokens: Int { Int(config.contextLimit) }

    /// Run Metal prefill: batch of token IDs in → first decoded token out.
    /// Interprets prefill_dispatches.bin in a single command buffer.
    public func prefillStep(
        tokenIds: [Int32],
        startPos: Int32,
        selectionMode: SmeltSelectionMode = .argmax,
        allowedTokenMask: SmeltAllowedTokenMaskProvider? = nil
    ) throws -> Int32 {
        // emit_all_logits packages skip the per-prefill argmax
        // dispatch and write logits at [B, vocab] instead of
        // [vocab]. prefillStep's "return last-token argmax"
        // contract can't be satisfied without re-doing the
        // selection at the (seqLen-1)*vocab offset; the supported
        // entry for such packages is `prefillAllLogits`.
        if supportsChunkedPrefillVerify {
            throw SmeltRuntimeError.invalidPackage(
                "prefillStep is undefined on packages built with "
                + "prefill.emit_all_logits=true; use prefillAllLogits"
            )
        }
        try ensurePrefillCapacity(tokenIds: tokenIds, startPos: startPos)

        guard let table = prefillDispatchTable else {
            throw SmeltRuntimeError.dispatchTableMissing(
                "no prefill_dispatches.bin — cannot prefill"
            )
        }
        let (cmdBuf, enc) = try makeCommandBufferAndEncoder()

        interpretPrefillDispatchTable(
            table, enc: enc, tokenIds: tokenIds,
            seqLen: Int32(tokenIds.count), startPos: startPos
        )

        enc.endEncoding()
        let selectionPosition = startPos + Int32(tokenIds.count) - 1
        let usedGPUSampler =
            allowedTokenMask == nil
            && encodeSelectionKernelIfNeeded(
                commandBuffer: cmdBuf,
                position: selectionPosition,
                selectionMode: selectionMode
            )
        cmdBuf.commit()
        // Evaluate under the GPU wait, same as decodeStep.
        let mask: [UInt32]?
        do {
            mask = try allowedTokenMask?()
        } catch {
            cmdBuf.waitUntilCompleted()
            throw error
        }
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw error }

        if usedGPUSampler {
            return buffers[Int(config.argmaxSlot)]
                .contents().load(as: Int32.self)
        }

        return selectCurrentToken(
            position: selectionPosition,
            selectionMode: selectionMode,
            allowedTokenMask: mask
        )
    }

    /// Run the normal full-model prefill from caller-supplied fp16 embeddings.
    ///
    /// This is the generic fusion seam for encoders whose output replaces token
    /// embeddings (images, audio, or another future modality). It deliberately
    /// reuses the package's ordinary prefill table and skips only its first,
    /// structurally verified token-gather dispatch. Every subsequent swap,
    /// transformer dispatch, cache write, LM head, and selection stays on the
    /// package-owned path.
    ///
    /// `ropeCos` and `ropeSin`, when present, replace complete table prefixes.
    /// They must contain the same number of whole fp16 RoPE rows and cover this
    /// prefill. Supplying additional rows pins the positions used by subsequent
    /// decode, which is how an MRoPE caller carries its prompt delta forward.
    public func prefillEmbeddings(
        _ embeddings: [Float16],
        tokenIds: [Int32],
        startPos: Int32 = 0,
        ropeCos: [Float16]? = nil,
        ropeSin: [Float16]? = nil,
        selectionMode: SmeltSelectionMode = .argmax,
        allowedTokenMask: SmeltAllowedTokenMaskProvider? = nil
    ) throws -> Int32 {
        guard !tokenIds.isEmpty else {
            throw SmeltRuntimeError.invalidPackage(
                "prefillEmbeddings requires at least one token"
            )
        }
        guard startPos >= 0,
              Int(startPos) <= maxContextTokens - tokenIds.count
        else {
            throw SmeltRuntimeError.invalidPackage(
                "prefillEmbeddings range [\(startPos), +\(tokenIds.count)) "
                    + "exceeds context limit \(maxContextTokens)"
            )
        }
        guard !supportsChunkedPrefillVerify else {
            throw SmeltRuntimeError.invalidPackage(
                "prefillEmbeddings is undefined on packages built with "
                    + "prefill.emit_all_logits=true"
            )
        }
        guard let table = prefillDispatchTable else {
            throw SmeltRuntimeError.dispatchTableMissing(
                "no prefill_dispatches.bin — cannot prefill embeddings"
            )
        }
        guard table.count > 0,
              table[0].opKind == SmeltDispatchRecord.opDispatch,
              Int(table[0].pipeline) < manifest.pipelines.count,
              manifest.pipelines[Int(table[0].pipeline)]
                == "affine_embedding_gather_batched"
        else {
            let firstName: String
            if table.count > 0,
               table[0].opKind == SmeltDispatchRecord.opDispatch,
               Int(table[0].pipeline) < manifest.pipelines.count {
                firstName = manifest.pipelines[Int(table[0].pipeline)]
            } else {
                firstName = "<non-dispatch>"
            }
            throw SmeltRuntimeError.invalidPackage(
                "prefillEmbeddings requires the first prefill record to be "
                    + "affine_embedding_gather_batched; got \(firstName)"
            )
        }

        guard let hiddenSlot = manifest.buffers.slots.first(
            where: { $0.name == "hiddenA" }
        ), hiddenSlot.index == 0, hiddenSlot.dtype == .fp16 else {
            throw SmeltRuntimeError.invalidPackage(
                "prefillEmbeddings requires the full-model fp16 hiddenA ABI "
                    + "at double-buffer slot 0"
            )
        }
        let hidden = manifest.config.hiddenSize
        guard embeddings.count == tokenIds.count * hidden else {
            throw SmeltRuntimeError.invalidPackage(
                "prefillEmbeddings expected [\(tokenIds.count), \(hidden)] = "
                    + "\(tokenIds.count * hidden) fp16 values; got "
                    + "\(embeddings.count)"
            )
        }
        guard maxPrefillBatchSize > 0 else {
            throw SmeltRuntimeError.invalidPackage(
                "prefillEmbeddings requires max_prefill_batch_size > 0"
            )
        }

        let chunkCapacity = min(tokenIds.count, maxPrefillBatchSize)
        try ensurePrefillCapacity(seqLen: chunkCapacity, startPos: startPos)
        try ensureContextCapacity(Int(startPos) + tokenIds.count)

        switch (ropeCos, ropeSin) {
        case (nil, nil):
            break
        case let (.some(cos), .some(sin)):
            let ropeDim = manifest.config.ropeDim
            guard ropeDim > 0,
                  cos.count == sin.count,
                  cos.count % ropeDim == 0,
                  cos.count / ropeDim >= Int(startPos) + tokenIds.count,
                  cos.count / ropeDim <= maxContextTokens
            else {
                throw SmeltRuntimeError.invalidPackage(
                    "prefillEmbeddings RoPE overrides must be equal whole-row "
                        + "tables covering the prefill (ropeDim=\(ropeDim))"
                )
            }
            guard let cosSlot = manifest.buffers.slots.first(
                where: { $0.name == "ropeCos" }
            ), let sinSlot = manifest.buffers.slots.first(
                where: { $0.name == "ropeSin" }
            ), cosSlot.dtype == .fp16, sinSlot.dtype == .fp16 else {
                throw SmeltRuntimeError.invalidPackage(
                    "prefillEmbeddings fp16 RoPE overrides require fp16 "
                        + "ropeCos/ropeSin slots"
                )
            }
            // The override may intentionally include later decode rows. Grow
            // through that prefix before pinning; a grow rebuilds production
            // RoPE and would otherwise erase the caller's MRoPE rows.
            try ensureContextCapacity(cos.count / ropeDim)
            try writeSlotFloat16(
                cosSlot.index, values: cos[...], label: "prefillEmbeddings ropeCos"
            )
            try writeSlotFloat16(
                sinSlot.index, values: sin[...], label: "prefillEmbeddings ropeSin"
            )
        default:
            throw SmeltRuntimeError.invalidPackage(
                "prefillEmbeddings requires ropeCos and ropeSin together"
            )
        }

        var offset = 0
        while offset < tokenIds.count {
            let end = min(offset + chunkCapacity, tokenIds.count)
            let chunkTokens = Array(tokenIds[offset..<end])
            let embeddingStart = offset * hidden
            let embeddingEnd = end * hidden
            try writeSlotFloat16(
                hiddenSlot.index,
                values: embeddings[embeddingStart..<embeddingEnd],
                label: "prefillEmbeddings hiddenA"
            )

            let (cmdBuf, enc) = try makeCommandBufferAndEncoder()
            interpretPrefillDispatchTable(
                table,
                enc: enc,
                tokenIds: chunkTokens,
                seqLen: Int32(chunkTokens.count),
                startPos: startPos + Int32(offset),
                skipInitialEmbeddingGather: true
            )
            enc.endEncoding()

            let isLast = end == tokenIds.count
            let selectionPosition = startPos + Int32(tokenIds.count) - 1
            let usedGPUSampler = isLast
                && allowedTokenMask == nil
                && encodeSelectionKernelIfNeeded(
                    commandBuffer: cmdBuf,
                    position: selectionPosition,
                    selectionMode: selectionMode
                )
            cmdBuf.commit()

            let mask: [UInt32]?
            do {
                mask = isLast ? try allowedTokenMask?() : nil
            } catch {
                cmdBuf.waitUntilCompleted()
                throw error
            }
            cmdBuf.waitUntilCompleted()
            if let error = cmdBuf.error { throw error }

            if isLast {
                if usedGPUSampler {
                    return buffers[Int(config.argmaxSlot)]
                        .contents().load(as: Int32.self)
                }
                return selectCurrentToken(
                    position: selectionPosition,
                    selectionMode: selectionMode,
                    allowedTokenMask: mask
                )
            }
            offset = end
        }

        preconditionFailure("non-empty prefillEmbeddings loop did not return")
    }

    /// Run the compiled prefill table over a pre-filled embeddings batch — the
    /// dense trunk's embeddings-in / hidden-out ports (no token gather, no
    /// selection, no LM head). Writes `[seqLen, hidden]` embeddings into hiddenA,
    /// runs the prefill table at `startPos`, and returns `[seqLen, hidden]` from
    /// normOutBuf. K/V cache rows `[startPos, startPos + seqLen)` are populated
    /// for a subsequent decode loop. Optional rope tables are pinned AFTER
    /// capacity is ensured (a grow repopulates them from the production fill).
    /// The dense trunk prefill entry (B3.x); the W2 parity path.
    ///
    /// SINGLE-CHUNK ONLY: the DenseTrunkPrefillEmitter table's attention attends
    /// cache rows `[0, seqLen)` (chunk-local), so it is only correct at
    /// `startPos == 0`. Cross-chunk prefill (attending the full `[0, startPos +
    /// seqLen)` prefix) is a separate unit; until it lands, `startPos != 0` is
    /// rejected rather than silently computed against the wrong rows.
    public func prefillTrunk(
        embeddings: [Float], seqLen: Int,
        ropeCos: [Float]? = nil, ropeSin: [Float]? = nil,
        startPos: Int32 = 0
    ) throws -> [Float] {
        let table = try guardedPrefillTrunkTable(seqLen: seqLen, startPos: startPos)
        try ensurePipelineStatesMaterialized(
            for: table,
            tableName: "prefill_dispatches.bin"
        )
        let ports = try guardDenseTrunkABI("prefillTrunk")
        let hiddenASlot = ports.hiddenA
        let normOutSlot = ports.normOut
        // `hidden` is the package's, never the caller's: a caller-supplied value
        // that happened to match a shorter `embeddings.count` would write a
        // partial hiddenA slab, run full hidden-size kernels, and return a
        // truncated hidden state — a silent shape bug.
        let hidden = manifest.config.hiddenSize
        precondition(embeddings.count == seqLen * hidden,
                     "embeddings must be [seqLen, hidden] = \(seqLen * hidden), got \(embeddings.count)")
        func slotIndex(_ name: String) -> Int? {
            manifest.buffers.slots.first { $0.name == name }?.index
        }
        // Capacity first (grow-only): a grow repopulates the rope tables, so pin
        // them AFTER, then write the embeddings into the (now correctly sized) slab.
        try ensurePrefillCapacity(
            tokenIds: [Int32](repeating: 0, count: seqLen), startPos: startPos)
        if let ropeCos, let slot = slotIndex("ropeCos") {
            if ports.dtype == .bf16 {
                writeSlot(slot, bf16: ropeCos)
            } else {
                writeSlot(slot, f32: ropeCos)
            }
        }
        if let ropeSin, let slot = slotIndex("ropeSin") {
            if ports.dtype == .bf16 {
                writeSlot(slot, bf16: ropeSin)
            } else {
                writeSlot(slot, f32: ropeSin)
            }
        }
        if ports.dtype == .bf16 {
            writeSlot(hiddenASlot, bf16: embeddings)
        } else {
            writeSlot(hiddenASlot, f32: embeddings)
        }

        let (cmdBuf, enc) = try makeCommandBufferAndEncoder()
        interpretPrefillDispatchTable(
            table, enc: enc, tokenIds: [], seqLen: Int32(seqLen), startPos: startPos)
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        return dumpSlot(
            normOutSlot,
            count: seqLen * hidden,
            asFP32: ports.dtype == .fp32,
            asBF16: ports.dtype == .bf16
        )
    }

    /// Runs the same compiled headless-trunk prefill table while draining at
    /// each semantic decoder-layer boundary. Dense trunk layers contain exactly
    /// two hidden-stream swaps; the second swap commits the post-FFN residual.
    /// This exactness harness validates that invariant and returns every raw
    /// layer output plus the final normalized hidden tensor. It is deliberately
    /// synchronous and is not a production inference path.
    public func captureTrunkPrefillLayerOutputs(
        embeddings: [Float],
        seqLen: Int,
        ropeCos: [Float]? = nil,
        ropeSin: [Float]? = nil
    ) throws -> SmeltTrunkLayerCapture {
        let table = try guardedPrefillTrunkTable(seqLen: seqLen, startPos: 0)
        try ensurePipelineStatesMaterialized(
            for: table,
            tableName: "prefill_dispatches.bin"
        )
        let ports = try guardDenseTrunkABI("captureTrunkPrefillLayerOutputs")
        let hiddenASlot = ports.hiddenA
        let normOutSlot = ports.normOut
        guard let hiddenBSlot = manifest.buffers.slots.first(where: {
            $0.name == "hiddenB"
        })?.index else {
            throw SmeltRuntimeError.invalidPackage(
                "captureTrunkPrefillLayerOutputs: headless trunk has no hiddenB slot"
            )
        }
        let hidden = manifest.config.hiddenSize
        precondition(
            embeddings.count == seqLen * hidden,
            "embeddings must be [seqLen, hidden] = \(seqLen * hidden), got \(embeddings.count)"
        )
        try ensurePrefillCapacity(
            tokenIds: [Int32](repeating: 0, count: seqLen),
            startPos: 0
        )
        func slotIndex(_ name: String) -> Int? {
            manifest.buffers.slots.first { $0.name == name }?.index
        }
        if let ropeCos, let slot = slotIndex("ropeCos") {
            if ports.dtype == .bf16 {
                writeSlot(slot, bf16: ropeCos)
            } else {
                writeSlot(slot, f32: ropeCos)
            }
        }
        if let ropeSin, let slot = slotIndex("ropeSin") {
            if ports.dtype == .bf16 {
                writeSlot(slot, bf16: ropeSin)
            } else {
                writeSlot(slot, f32: ropeSin)
            }
        }
        resetWorkingBuffers()
        if ports.dtype == .bf16 {
            writeSlot(hiddenASlot, bf16: embeddings)
        } else {
            writeSlot(hiddenASlot, f32: embeddings)
        }

        var current = hiddenASlot
        var alternate = hiddenBSlot
        var swapCount = 0
        var layerOutputs: [[Float]] = []
        layerOutputs.reserveCapacity(manifest.config.numLayers)
        var (commandBuffer, encoder) = try makeCommandBufferAndEncoder()

        func drain(
            _ commandBuffer: MTLCommandBuffer,
            _ encoder: MTLComputeCommandEncoder
        ) throws {
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error { throw error }
        }

        for record in table {
            if record.opKind == SmeltDispatchRecord.opSwap {
                swap(&current, &alternate)
                swapCount += 1
                if swapCount.isMultiple(of: 2) {
                    try drain(commandBuffer, encoder)
                    layerOutputs.append(
                        dumpSlot(
                            current,
                            count: seqLen * hidden,
                            asFP32: ports.dtype == .fp32,
                            asBF16: ports.dtype == .bf16
                        )
                    )
                    (commandBuffer, encoder) = try makeCommandBufferAndEncoder()
                }
                continue
            }
            guard record.opKind == SmeltDispatchRecord.opDispatch else { continue }
            emitDispatchRecord(
                record,
                enc: encoder,
                cur: current,
                alt: alternate,
                seqLen: Int32(seqLen),
                startPos: 0
            )
        }
        try drain(commandBuffer, encoder)
        guard swapCount == manifest.config.numLayers * 2,
              layerOutputs.count == manifest.config.numLayers
        else {
            throw SmeltRuntimeError.invalidPackage(
                "captureTrunkPrefillLayerOutputs: saw \(swapCount) swaps / "
                    + "\(layerOutputs.count) layers, expected "
                    + "\(manifest.config.numLayers * 2) / \(manifest.config.numLayers)"
            )
        }
        return SmeltTrunkLayerCapture(
            layerOutputs: layerOutputs,
            finalHiddenStates: dumpSlot(
                normOutSlot,
                count: seqLen * hidden,
                asFP32: ports.dtype == .fp32,
                asBF16: ports.dtype == .bf16
            )
        )
    }

    /// Runs one dense-trunk decode while draining at each decoder-layer
    /// boundary. The caller must populate the hidden-input port first; the
    /// existing K/V cache is advanced exactly once at `position`.
    func captureTrunkDecodeLayerOutputs(
        tokenId: Int32,
        position: Int32
    ) throws -> SmeltTrunkLayerCapture {
        guard let table = dispatchTable else {
            throw SmeltRuntimeError.dispatchTableMissing(
                "no dispatches.bin — cannot capture decode layers"
            )
        }
        try ensurePipelineStatesMaterialized(
            for: table,
            tableName: "dispatches.bin"
        )
        let ports = try guardDenseTrunkABI("captureTrunkDecodeLayerOutputs")
        guard let hiddenBSlot = manifest.buffers.slots.first(where: {
            $0.name == "hiddenB"
        })?.index else {
            throw SmeltRuntimeError.invalidPackage(
                "captureTrunkDecodeLayerOutputs: headless trunk has no hiddenB slot"
            )
        }
        precondition(
            Int(position) < currentContextCapacity,
            "captureTrunkDecodeLayerOutputs: ensure context capacity before capture"
        )
        prepareDecodeDynamicState(tokenId: tokenId, position: position)

        let hidden = manifest.config.hiddenSize
        var current = ports.hiddenA
        var alternate = hiddenBSlot
        var swapCount = 0
        var layerOutputs: [[Float]] = []
        layerOutputs.reserveCapacity(manifest.config.numLayers)
        var (commandBuffer, encoder) = try makeCommandBufferAndEncoder()

        func drain() throws {
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error { throw error }
        }

        for record in table {
            if record.opKind == SmeltDispatchRecord.opSwap {
                swap(&current, &alternate)
                swapCount += 1
                if swapCount.isMultiple(of: 2) {
                    try drain()
                    layerOutputs.append(
                        dumpSlot(
                            current,
                            count: hidden,
                            asFP32: ports.dtype == .fp32,
                            asBF16: ports.dtype == .bf16
                        )
                    )
                    (commandBuffer, encoder) = try makeCommandBufferAndEncoder()
                }
                continue
            }
            emitDecodeDispatchRecord(
                record,
                enc: encoder,
                cur: current,
                alt: alternate,
                position: position
            )
        }
        try drain()
        guard swapCount == manifest.config.numLayers * 2,
              layerOutputs.count == manifest.config.numLayers
        else {
            throw SmeltRuntimeError.invalidPackage(
                "captureTrunkDecodeLayerOutputs: saw \(swapCount) swaps / "
                    + "\(layerOutputs.count) layers, expected "
                    + "\(manifest.config.numLayers * 2) / \(manifest.config.numLayers)"
            )
        }
        return SmeltTrunkLayerCapture(
            layerOutputs: layerOutputs,
            finalHiddenStates: dumpSlot(
                ports.normOut,
                count: hidden,
                asFP32: ports.dtype == .fp32,
                asBF16: ports.dtype == .bf16
            )
        )
    }

    /// Runs one dense-trunk decode until requested dispatch boundaries and
    /// snapshots logical or named slots without resetting the live K/V cache.
    func captureTrunkDecodeDispatchOutputs(
        tokenId: Int32,
        position: Int32,
        requests: [SmeltTrunkDispatchCaptureRequest]
    ) throws -> [String: [Float]] {
        guard !requests.isEmpty else { return [:] }
        guard let table = dispatchTable else {
            throw SmeltRuntimeError.dispatchTableMissing(
                "no dispatches.bin — cannot capture decode dispatches"
            )
        }
        try ensurePipelineStatesMaterialized(
            for: table,
            tableName: "dispatches.bin"
        )
        let ports = try guardDenseTrunkABI("captureTrunkDecodeDispatchOutputs")
        guard let hiddenBSlot = manifest.buffers.slots.first(where: {
            $0.name == "hiddenB"
        })?.index else {
            throw SmeltRuntimeError.invalidPackage(
                "captureTrunkDecodeDispatchOutputs: headless trunk has no hiddenB slot"
            )
        }
        guard Set(requests.map(\.label)).count == requests.count,
              requests.allSatisfy({ $0.afterDispatch > 0 && $0.count > 0 })
        else {
            throw SmeltRuntimeError.invalidPackage(
                "captureTrunkDecodeDispatchOutputs: labels must be unique and bounds positive"
            )
        }
        precondition(
            Int(position) < currentContextCapacity,
            "captureTrunkDecodeDispatchOutputs: ensure context capacity before capture"
        )
        prepareDecodeDynamicState(tokenId: tokenId, position: position)

        func slot(named name: String) throws -> Int {
            guard let index = manifest.buffers.slots.first(where: {
                $0.name == name
            })?.index else {
                throw SmeltRuntimeError.invalidPackage(
                    "captureTrunkDecodeDispatchOutputs: unknown slot \(name)"
                )
            }
            return index
        }

        func snapshot(
            _ request: SmeltTrunkDispatchCaptureRequest,
            current: Int,
            alternate: Int
        ) throws -> [Float] {
            let sourceSlot: Int
            switch request.source {
            case .current:
                sourceSlot = current
            case .alternate:
                sourceSlot = alternate
            case .slot(let name):
                sourceSlot = try slot(named: name)
            case .slotOffset(let name, _):
                sourceSlot = try slot(named: name)
            }
            let elementOffset: Int
            if case .slotOffset(_, let offset) = request.source {
                elementOffset = offset
            } else {
                elementOffset = 0
            }
            return dumpSlot(
                sourceSlot,
                elementOffset: elementOffset,
                count: request.count,
                asFP32: ports.dtype == .fp32,
                asBF16: ports.dtype == .bf16
            )
        }

        let requestsByDispatch = Dictionary(grouping: requests, by: \.afterDispatch)
        let lastRequestedDispatch = requests.map(\.afterDispatch).max() ?? 0
        var current = ports.hiddenA
        var alternate = hiddenBSlot
        var dispatchOrdinal = 0
        var captures: [String: [Float]] = [:]
        var (commandBuffer, encoder) = try makeCommandBufferAndEncoder()

        func drain() throws {
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error { throw error }
        }

        for record in table {
            if record.opKind == SmeltDispatchRecord.opSwap {
                swap(&current, &alternate)
                continue
            }
            guard record.opKind == SmeltDispatchRecord.opDispatch else { continue }
            dispatchOrdinal += 1
            emitDecodeDispatchRecord(
                record,
                enc: encoder,
                cur: current,
                alt: alternate,
                position: position
            )
            guard let boundaryRequests = requestsByDispatch[dispatchOrdinal] else {
                continue
            }
            try drain()
            for request in boundaryRequests {
                captures[request.label] = try snapshot(
                    request,
                    current: current,
                    alternate: alternate
                )
            }
            if dispatchOrdinal == lastRequestedDispatch {
                return captures
            }
            (commandBuffer, encoder) = try makeCommandBufferAndEncoder()
        }
        try drain()
        throw SmeltRuntimeError.invalidPackage(
            "captureTrunkDecodeDispatchOutputs: table ended at dispatch "
                + "\(dispatchOrdinal), before requested dispatch \(lastRequestedDispatch)"
        )
    }

    /// Runs a dense headless trunk until requested dispatch boundaries and
    /// snapshots the requested logical or named slots. This is an exactness
    /// bring-up primitive, not a production inference path.
    func captureTrunkPrefillDispatchOutputs(
        embeddings: [Float],
        seqLen: Int,
        requests: [SmeltTrunkDispatchCaptureRequest]
    ) throws -> [String: [Float]] {
        guard !requests.isEmpty else { return [:] }
        let table = try guardedPrefillTrunkTable(seqLen: seqLen, startPos: 0)
        try ensurePipelineStatesMaterialized(
            for: table,
            tableName: "prefill_dispatches.bin"
        )
        let ports = try guardDenseTrunkABI("captureTrunkPrefillDispatchOutputs")
        guard let hiddenBSlot = manifest.buffers.slots.first(where: {
            $0.name == "hiddenB"
        })?.index else {
            throw SmeltRuntimeError.invalidPackage(
                "captureTrunkPrefillDispatchOutputs: headless trunk has no hiddenB slot"
            )
        }
        let hidden = manifest.config.hiddenSize
        precondition(
            embeddings.count == seqLen * hidden,
            "embeddings must be [seqLen, hidden] = \(seqLen * hidden), got \(embeddings.count)"
        )
        guard Set(requests.map(\.label)).count == requests.count,
              requests.allSatisfy({ $0.afterDispatch > 0 && $0.count > 0 })
        else {
            throw SmeltRuntimeError.invalidPackage(
                "captureTrunkPrefillDispatchOutputs: labels must be unique and bounds positive"
            )
        }
        let requestsByDispatch = Dictionary(grouping: requests, by: \.afterDispatch)
        let lastRequestedDispatch = requests.map(\.afterDispatch).max() ?? 0

        try ensurePrefillCapacity(
            tokenIds: [Int32](repeating: 0, count: seqLen),
            startPos: 0
        )
        resetWorkingBuffers()
        if ports.dtype == .bf16 {
            writeSlot(ports.hiddenA, bf16: embeddings)
        } else {
            writeSlot(ports.hiddenA, f32: embeddings)
        }

        func slot(named name: String) throws -> Int {
            guard let index = manifest.buffers.slots.first(where: {
                $0.name == name
            })?.index else {
                throw SmeltRuntimeError.invalidPackage(
                    "captureTrunkPrefillDispatchOutputs: unknown slot \(name)"
                )
            }
            return index
        }

        func snapshot(
            _ request: SmeltTrunkDispatchCaptureRequest,
            current: Int,
            alternate: Int
        ) throws -> [Float] {
            let sourceSlot: Int
            switch request.source {
            case .current:
                sourceSlot = current
            case .alternate:
                sourceSlot = alternate
            case .slot(let name):
                sourceSlot = try slot(named: name)
            case .slotOffset(let name, _):
                sourceSlot = try slot(named: name)
            }
            let elementOffset: Int
            if case .slotOffset(_, let offset) = request.source {
                elementOffset = offset
            } else {
                elementOffset = 0
            }
            return dumpSlot(
                sourceSlot,
                elementOffset: elementOffset,
                count: request.count,
                asFP32: ports.dtype == .fp32,
                asBF16: ports.dtype == .bf16
            )
        }

        var current = ports.hiddenA
        var alternate = hiddenBSlot
        var dispatchOrdinal = 0
        var captures: [String: [Float]] = [:]
        var (commandBuffer, encoder) = try makeCommandBufferAndEncoder()

        func drain() throws {
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            if let error = commandBuffer.error { throw error }
        }

        for record in table {
            if record.opKind == SmeltDispatchRecord.opSwap {
                swap(&current, &alternate)
                continue
            }
            guard record.opKind == SmeltDispatchRecord.opDispatch else { continue }
            dispatchOrdinal += 1
            emitDispatchRecord(
                record,
                enc: encoder,
                cur: current,
                alt: alternate,
                seqLen: Int32(seqLen),
                startPos: 0
            )
            guard let boundaryRequests = requestsByDispatch[dispatchOrdinal] else {
                continue
            }
            try drain()
            for request in boundaryRequests {
                captures[request.label] = try snapshot(
                    request,
                    current: current,
                    alternate: alternate
                )
            }
            if dispatchOrdinal == lastRequestedDispatch {
                return captures
            }
            (commandBuffer, encoder) = try makeCommandBufferAndEncoder()
        }
        try drain()
        throw SmeltRuntimeError.invalidPackage(
            "captureTrunkPrefillDispatchOutputs: table ended at dispatch "
                + "\(dispatchOrdinal), before requested dispatch \(lastRequestedDispatch)"
        )
    }

    /// Cross-chunk prefill of a compiled dense trunk over an arbitrary-length prompt
    /// (B3.2b, docs/talker-trunk-crosschunk-prefill-plan.md). `source` holds the
    /// `[seqLen, hidden]` embeddings on THIS runtime's device; the FINAL hidden row
    /// (absolute position `seqLen-1`) is blitted into `dest`. The trunk owns the KV.
    ///
    /// Drives the prefill table once per `max_prefill_batch` chunk at advancing
    /// `startPos` (the start-aware `causal_gqa_attn_cached_f32` makes each chunk
    /// attend the absolute cache prefix `[0, startPos+t]`; KV/rope are startPos-
    /// strided). hiddenA is rewritten between chunks, so each chunk is its OWN command
    /// buffer (chunks can't share a live encoder) on THIS runtime's queue — same-queue
    /// serial commit orders the chunks among themselves, and the host wait on the LAST
    /// buffer orders them against the caller's subsequent read of `dest`.
    ///
    /// Capacity is a CALLER precondition (buffers can't resize between chunks):
    /// `ensurePrefillCapacity(seqLen:)` + `ensureContextCapacity(seqLen + …)` and pin
    /// RoPE BEFORE calling (the session does this at init). Returns the chunk count.
    @discardableResult
    public func prefillTrunkChunked(
        source: MTLBuffer, dest: MTLBuffer, seqLen: Int, hidden: Int
    ) throws -> Int {
        let ports = try guardDenseTrunkABI("prefillTrunkChunked")
        let hiddenASlot = ports.hiddenA
        let normOutSlot = ports.normOut
        guard let table = prefillDispatchTable else {
            throw SmeltRuntimeError.dispatchTableMissing("no prefill_dispatches.bin — cannot prefill")
        }
        try ensurePipelineStatesMaterialized(
            for: table,
            tableName: "prefill_dispatches.bin"
        )
        precondition(hidden == manifest.config.hiddenSize,
                     "prefillTrunkChunked hidden \(hidden) != package hidden \(manifest.config.hiddenSize)")
        guard seqLen >= 1 else {
            throw SmeltRuntimeError.invalidPackage("prefillTrunkChunked seqLen \(seqLen) must be >= 1")
        }
        // Variable-length prefill (Phase 4 U1): seqLen > 2048 is served by routing the
        // over-cap chunks to the uncapped scalar attention (emitDispatchRecord). A package
        // that predates that kernel can't, so reject loudly rather than mis-run on the SIMD
        // kernel's 2048-capped score buffer (the silent-drop-vs-loud-throw invariant).
        // The scalar attention is O(chunkLen · heads · cacheLen · headDim) per layer in ONE
        // dispatch (one thread per (t,head), each streaming the whole cache twice). The only
        // ceiling is the GPU watchdog (~seconds); at chunkLen = max_prefill_batch and a cache
        // of ~hundreds of thousands of rows that is reachable, but it is FAR beyond any real
        // long dense-trunk prompt, so no hard cap is imposed — a watchdog timeout on a
        // pathological prompt surfaces as a Metal command-buffer error, not silent corruption.
        if seqLen > Self.cachedAttnSimdCap {
            guard cachedAttnScalarPipelineIndex != nil else {
                throw SmeltRuntimeError.invalidPackage(
                    "prefillTrunkChunked seqLen \(seqLen) exceeds the SIMD attention cap "
                    + "\(Self.cachedAttnSimdCap) and this package has no "
                    + "causal_gqa_attn_cached_scalar_f32 — rebuild it for variable-length prompts")
            }
        }
        let rowBytes = hidden * (ports.dtype == .fp32 ? 4 : 2)
        precondition(source.length >= seqLen * rowBytes,
                     "prefillTrunkChunked source \(source.length)B < [seqLen,hidden] \(seqLen * rowBytes)B")
        precondition(dest.length >= rowBytes,
                     "prefillTrunkChunked dest \(dest.length)B < hidden \(rowBytes)B")
        let maxBatch = maxPrefillBatchSize
        // Cross-chunk (startPos > 0) requires the start-aware attention kernel. A stale
        // pre-B3.2b trunk emits the chunk-local causal_gqa_attn_simd_f32, which IGNORES
        // startPos — bit-exact for a single chunk but SILENTLY WRONG for any continuation.
        // Fail loud rather than mis-compute (the silent-drop-vs-loud-throw invariant).
        if seqLen > maxBatch {
            let hasStartAware = table.contains { rec in
                rec.opKind == SmeltDispatchRecord.opDispatch
                    && Int(rec.pipeline) < manifest.pipelines.count
                    && ["causal_gqa_attn_cached_f32", "causal_gqa_attn_cached_bf16"]
                        .contains(manifest.pipelines[Int(rec.pipeline)])
            }
            guard hasStartAware else {
                throw SmeltRuntimeError.invalidPackage(
                    "prefillTrunkChunked needs cross-chunk attention but the prefill table has "
                    + "no causal_gqa_attn_cached_f32 dispatch (stale pre-B3.2b trunk) — rebuild it")
            }
        }
        let hiddenABuf = buffers[hiddenASlot]
        let normOutBuf = buffers[normOutSlot]
        var startPos = 0
        var chunks = 0
        while startPos < seqLen {
            let chunkLen = min(maxBatch, seqLen - startPos)
            precondition(startPos + chunkLen <= currentContextCapacity,
                         "prefillTrunkChunked: context capacity \(currentContextCapacity) < "
                         + "\(startPos + chunkLen) — ensure capacity before calling")
            precondition(chunkLen <= currentBatchCapacity,
                         "prefillTrunkChunked: batch capacity \(currentBatchCapacity) < chunkLen "
                         + "\(chunkLen) — ensure capacity before calling")
            guard let cmdBuf = queue.makeCommandBuffer() else {
                throw SmeltRuntimeError.metalCommandBufferUnavailable
            }
            // (1) GPU bind-in: this chunk's source rows → hiddenA[0..chunkLen).
            guard let blitIn = cmdBuf.makeBlitCommandEncoder() else {
                throw SmeltRuntimeError.metalCommandBufferUnavailable
            }
            blitIn.copy(from: source, sourceOffset: startPos * rowBytes,
                        to: hiddenABuf, destinationOffset: 0, size: chunkLen * rowBytes)
            blitIn.endEncoding()
            // (2) the prefill table at this startPos (own compute encoder).
            guard let enc = cmdBuf.makeComputeCommandEncoder() else {
                throw SmeltRuntimeError.metalCommandBufferUnavailable
            }
            interpretPrefillDispatchTable(
                table, enc: enc, tokenIds: [], seqLen: Int32(chunkLen), startPos: Int32(startPos))
            enc.endEncoding()
            let isLast = startPos + chunkLen >= seqLen
            if isLast {
                // (3) the final hidden row (this chunk's last) → dest.
                guard let blitOut = cmdBuf.makeBlitCommandEncoder() else {
                    throw SmeltRuntimeError.metalCommandBufferUnavailable
                }
                blitOut.copy(from: normOutBuf, sourceOffset: (chunkLen - 1) * rowBytes,
                             to: dest, destinationOffset: 0, size: rowBytes)
                blitOut.endEncoding()
            }
            cmdBuf.commit()
            if isLast { cmdBuf.waitUntilCompleted() }
            startPos += chunkLen
            chunks += 1
        }
        return chunks
    }

    /// The shared single-chunk prefill-table guard for `prefillTrunk` and
    /// `encodeTrunkPrefill`: single-chunk (startPos 0, the table's attention is
    /// chunk-local), seqLen in 1...max_prefill_batch (the buffer-plan slab
    /// multiplier; ensurePrefillCapacity would otherwise clamp and the embeddings
    /// write would trap), and a prefill table present.
    private func guardedPrefillTrunkTable(
        seqLen: Int, startPos: Int32
    ) throws -> UnsafeBufferPointer<SmeltDispatchRecord> {
        guard startPos == 0 else {
            throw SmeltRuntimeError.invalidPackage(
                "single-chunk prefill only (attention is chunk-local); "
                + "startPos \(startPos) != 0 is not supported yet")
        }
        guard seqLen >= 1, seqLen <= maxPrefillBatchSize else {
            throw SmeltRuntimeError.invalidPackage(
                "prefill seqLen \(seqLen) must be in 1...\(maxPrefillBatchSize) "
                + "(the package's max_prefill_batch)")
        }
        guard let table = prefillDispatchTable else {
            throw SmeltRuntimeError.dispatchTableMissing(
                "no prefill_dispatches.bin — cannot prefill")
        }
        return table
    }

    /// Resolves the model-agnostic dense headless-trunk ports. BF16 and FP32
    /// are storage families of the same embeddings-in/hidden-out contract.
    static func resolveDenseTrunkPorts(
        caller: String, headlessTrunkABI: Bool?, kind: String?,
        hiddenA: (index: Int, dtype: SmeltDType)?,
        normOut: (index: Int, dtype: SmeltDType)?
    ) throws -> (hiddenA: Int, normOut: Int, dtype: SmeltDType) {
        guard headlessTrunkABI == true else {
            throw SmeltRuntimeError.invalidPackage(
                "\(caller): the headless-trunk API requires a package marked headlessTrunkABI "
                    + "true; this package is headlessTrunkABI="
                    + "\(headlessTrunkABI.map(String.init) ?? "nil") "
                    + "kind=\(kind ?? "nil") — not a compiled headless trunk")
        }
        guard let hiddenA, let normOut else {
            throw SmeltRuntimeError.invalidPackage(
                "\(caller): the dense trunk API needs hiddenA + normOutBuf port slots")
        }
        guard hiddenA.dtype == normOut.dtype,
              hiddenA.dtype == .bf16 || hiddenA.dtype == .fp32
        else {
            throw SmeltRuntimeError.invalidPackage(
                "\(caller): dense trunk ports must share bf16 or fp32 storage; "
                    + "hiddenA=\(hiddenA.dtype) normOutBuf=\(normOut.dtype)")
        }
        return (hiddenA.index, normOut.index, hiddenA.dtype)
    }

    private func guardDenseTrunkABI(
        _ caller: String
    ) throws -> (hiddenA: Int, normOut: Int, dtype: SmeltDType) {
        func port(_ name: String) -> (index: Int, dtype: SmeltDType)? {
            manifest.buffers.slots.first(where: { $0.name == name }).map { ($0.index, $0.dtype) }
        }
        return try Self.resolveDenseTrunkPorts(
            caller: caller,
            headlessTrunkABI: manifest.headlessTrunkABI,
            kind: manifest.kind,
            hiddenA: port("hiddenA"),
            normOut: port("normOutBuf")
        )
    }

    /// Encode the compiled DECODE table into a CALLER-OWNED compute encoder — the
    /// trunk as a block in someone else's command-buffer scope (the talker
    /// session's `advance` phase: feedback-gather → trunk → cb0/MTP in one buffer).
    /// No command buffer, no commit/wait, no selection — the caller owns all of
    /// that. Capacity MUST already be ensured (buffers can't reallocate under a
    /// live encoder): call `ensureContextCapacity(position + 1)` before opening the
    /// encoder. The dense trunk's embeddings-in (hiddenA) / hidden-out (normOutBuf)
    /// ports mean the caller's prior dispatches fill hiddenA and its later ones
    /// read normOutBuf. (W3 of docs/talker-trunk-fit-audit.md.)
    ///
    /// ORDERING CONSTRAINT: this does a host-side write of `tokenId`/`position`
    /// into the runtime's dynamic slots (via the shared interpreter) before
    /// encoding — vestigial for the embeddings-in trunk (its kernels take position
    /// from baked offsets/constants, not the slot), but those slots are shared
    /// runtime state, so a command already encoded into the same buffer that reads
    /// them would observe these values at execution. Treat the tokenId/position
    /// slots as trunk-scoped within the encode.
    public func encodeTrunkDecode(
        into enc: MTLComputeCommandEncoder, tokenId: Int32, position: Int32
    ) throws {
        guard let table = dispatchTable else {
            throw SmeltRuntimeError.dispatchTableMissing(
                "no dispatches.bin — cannot decode")
        }
        try ensurePipelineStatesMaterialized(
            for: table,
            tableName: "dispatches.bin"
        )
        _ = try guardDenseTrunkABI("encodeTrunkDecode")
        precondition(Int(position) < currentContextCapacity,
                     "encodeTrunkDecode: ensure context capacity (≥ \(position + 1)) "
                     + "before opening the encoder — buffers can't resize mid-encoder")
        interpretDispatchTable(table, enc: enc, tokenId: tokenId, position: position)
    }

    /// Encode the compiled PREFILL table into a CALLER-OWNED compute encoder — the
    /// prefill counterpart of `encodeTrunkDecode` (the session's `prefill` scope).
    /// Single-chunk (startPos 0), seqLen in 1...max_prefill_batch. Capacity MUST
    /// already be ensured: call `ensurePrefillCapacity(seqLen:)` before opening the
    /// encoder. No command buffer / commit / wait / selection. The caller's
    /// front-end fills hiddenA (in the same encoder, before the trunk dispatches)
    /// and reads normOutBuf afterward — this does not write embeddings.
    ///
    /// ORDERING CONSTRAINT (as in `encodeTrunkDecode`): the shared interpreter
    /// host-side-writes the token-id batch slot (zero-fill) before encoding —
    /// vestigial for the embeddings-in trunk, but a command already encoded into
    /// the same buffer that reads that slot would observe it at execution. Treat
    /// the token-id batch slot as trunk-scoped within the encode.
    public func encodeTrunkPrefill(
        into enc: MTLComputeCommandEncoder, seqLen: Int, startPos: Int32 = 0
    ) throws {
        let table = try guardedPrefillTrunkTable(seqLen: seqLen, startPos: startPos)
        try ensurePipelineStatesMaterialized(
            for: table,
            tableName: "prefill_dispatches.bin"
        )
        _ = try guardDenseTrunkABI("encodeTrunkPrefill")
        precondition(seqLen <= currentBatchCapacity
                     && Int(startPos) + seqLen <= currentContextCapacity,
                     "encodeTrunkPrefill: ensure prefill capacity before opening the encoder")
        interpretPrefillDispatchTable(
            table, enc: enc, tokenIds: [], seqLen: Int32(seqLen), startPos: startPos)
    }

    /// The live MTLBuffer backing a trunk port slot — the bind/blit handle for
    /// running the compiled trunk as a sub-step of a hand-built pipeline (W4 of
    /// docs/talker-trunk-session-integration-plan.md). A caller binds a GPU
    /// dispatch's output AT `hiddenA` (embeddings-in) and blits `normOutBuf`
    /// (hidden-out) out, inside its own encoder scope — no host round-trip.
    ///
    /// Restricted to the two dense-trunk ports and gated by the same dense-trunk ABI
    /// marker as the encode entries, so it can't hand out arbitrary internal
    /// buffers. CAPACITY PRECONDITION: a context/prefill grow swaps the slot's
    /// MTLBuffer (resizeRequestScopedBuffers), so ensure final capacity FIRST and
    /// re-fetch after any grow — a buffer cached across a grow is stale.
    public func portSlotBuffer(_ name: String) throws -> MTLBuffer {
        let ports = try guardDenseTrunkABI("portSlotBuffer")
        switch name {
        case "hiddenA": return buffers[ports.hiddenA]
        case "normOutBuf": return buffers[ports.normOut]
        default:
            throw SmeltRuntimeError.invalidPackage(
                "portSlotBuffer: only 'hiddenA' / 'normOutBuf' are exposable (got '\(name)')")
        }
    }

    /// Capture GPTQ activation statistics for one prefill of `tokenIds` (a fresh
    /// `startPos`-based prefill; the caller manages any cross-prompt reset).
    ///
    /// Walks the prefill dispatch table once, breaking the command buffer after
    /// each capture point's boundary so the projection's input slot — host-visible
    /// shared storage — can be read and fed to `capture`. Calibration-only and slow
    /// (one command buffer per boundary), but reuses the proven host-side `XᵀX`
    /// accumulation. Requires a metal-prefill package.
    ///
    /// The boundary is a dispatch-*op* count (swaps excluded), matching how
    /// `SmeltCodeEmitter.buildCapturePoints` counts on the post-optimization stream;
    /// runtime-skipped dispatches (minSeqLen guards) still advance the count, so
    /// boundaries stay aligned with the build.
    public func captureGPTQActivations(
        tokenIds: [Int32],
        startPos: Int32 = 0,
        capturePoints: [SmeltGPTQCapturePoint],
        into capture: SmeltActivationCapture
    ) throws {
        guard let table = prefillDispatchTable else {
            throw SmeltRuntimeError.dispatchTableMissing(
                "no prefill_dispatches.bin — cannot capture GPTQ activations"
            )
        }
        try ensurePrefillCapacity(tokenIds: tokenIds, startPos: startPos)
        // Each capture is an independent fresh prefill: zero the working/state buffers
        // (incl. DeltaNet conv/recurrent state) so a prior call's leftover state can't
        // contaminate this one's activations — then write the tokens.
        resetWorkingBuffers()
        writePrefillTokenIds(tokenIds)
        var byBoundary: [Int: [SmeltGPTQCapturePoint]] = [:]
        for point in capturePoints { byBoundary[point.dispatchCount, default: []].append(point) }
        try runGPTQCaptureWalk(table, seqLen: Int32(tokenIds.count), startPos: startPos,
                               byBoundary: byBoundary, into: capture)
    }

    /// GPTQ capture for an FP32 operation cell: seed hiddenA with `source` — the
    /// concatenated [prompt embeds ++ teacher-forced decode-frame inputs] — and capture each
    /// projection's Hessian through the COMPILED trunk's prefill, so u4 calibration rides the
    /// compiled generation path. The trunk consumes hiddenA (no token-embedding gather), so this
    /// is the hiddenA-seeded twin of `captureGPTQActivations`.
    ///
    /// CHUNKED like `prefillTrunkChunked`: an arbitrarily long concatenated sequence runs in
    /// max_prefill_batch chunks (over-cap chunks use the scalar attention, Phase 4 U1), and the
    /// Hessian accumulates across chunks (XᵀX is additive; `accumulate` also tallies the per-weight
    /// calibration-row count across calls). The KV cache is reset ONCE before the first chunk and
    /// carried across chunks (cross-chunk causality), so every position's projection input matches
    /// the real generation's. Caller pins ropeCos/ropeSin + grows capacity first.
    public func captureGPTQActivationsFromHidden(
        source: MTLBuffer, seqLen: Int, hidden: Int,
        capturePoints: [SmeltGPTQCapturePoint], into capture: SmeltActivationCapture
    ) throws {
        let ports = try guardDenseTrunkABI("captureGPTQActivationsFromHidden")
        guard ports.dtype == .fp32 else {
            throw SmeltRuntimeError.invalidPackage(
                "captureGPTQActivationsFromHidden has no registered \(ports.dtype.rawValue) "
                    + "source-copy cell")
        }
        let hiddenASlot = ports.hiddenA
        guard let table = prefillDispatchTable else {
            throw SmeltRuntimeError.dispatchTableMissing(
                "no prefill_dispatches.bin — cannot capture GPTQ activations")
        }
        precondition(hidden == manifest.config.hiddenSize,
                     "captureGPTQActivationsFromHidden hidden \(hidden) != package hidden \(manifest.config.hiddenSize)")
        let rowBytes = hidden * 4
        precondition(source.length >= seqLen * rowBytes,
                     "capture source \(source.length)B < [seqLen,hidden] \(seqLen * rowBytes)B")
        guard seqLen >= 1 else { return }
        if seqLen > Self.cachedAttnSimdCap {
            guard cachedAttnScalarPipelineIndex != nil else {
                throw SmeltRuntimeError.invalidPackage(
                    "capture seqLen \(seqLen) > \(Self.cachedAttnSimdCap) needs causal_gqa_attn_cached_scalar_f32 — rebuild")
            }
        }
        var byBoundary: [Int: [SmeltGPTQCapturePoint]] = [:]
        for point in capturePoints { byBoundary[point.dispatchCount, default: []].append(point) }
        let hiddenABuf = buffers[hiddenASlot]
        // Fresh prefill: reset state ONCE; chunks then carry the KV cache.
        resetWorkingBuffers()
        var startPos = 0
        while startPos < seqLen {
            let chunkLen = min(maxPrefillBatchSize, seqLen - startPos)
            precondition(startPos + chunkLen <= currentContextCapacity,
                         "capture: context capacity \(currentContextCapacity) < \(startPos + chunkLen)")
            precondition(chunkLen <= currentBatchCapacity,
                         "capture: batch capacity \(currentBatchCapacity) < chunkLen \(chunkLen)")
            // Bind this chunk's source rows → hiddenA[0..chunkLen).
            guard let blitCmd = queue.makeCommandBuffer(),
                  let blit = blitCmd.makeBlitCommandEncoder() else {
                throw SmeltRuntimeError.metalCommandBufferUnavailable
            }
            blit.copy(from: source, sourceOffset: startPos * rowBytes,
                      to: hiddenABuf, destinationOffset: 0, size: chunkLen * rowBytes)
            blit.endEncoding(); blitCmd.commit(); blitCmd.waitUntilCompleted()
            try runGPTQCaptureWalk(table, seqLen: Int32(chunkLen), startPos: Int32(startPos),
                                   byBoundary: byBoundary, into: capture)
            startPos += chunkLen
        }
    }

    /// Walk a prefill dispatch table once, breaking the command buffer after each capture
    /// point's boundary so the projection's (host-visible shared) input slot can be read and
    /// fed to `capture`. The boundary is a dispatch-op count (swaps excluded), matching how
    /// the build counts; runtime-skipped dispatches still advance the count. Shared by the
    /// token-seeded and hiddenA-seeded capture entries so they cannot drift.
    private func runGPTQCaptureWalk(
        _ table: UnsafeBufferPointer<SmeltDispatchRecord>,
        seqLen: Int32, startPos: Int32,
        byBoundary: [Int: [SmeltGPTQCapturePoint]],
        into capture: SmeltActivationCapture
    ) throws {
        var (cmdBuf, enc) = try makeCommandBufferAndEncoder()
        var cur = 0
        var alt = 1
        var dispatchOpCount = 0
        for idx in 0..<table.count {
            let rec = table[idx]
            if rec.opKind == SmeltDispatchRecord.opSwap {
                swap(&cur, &alt)
                continue
            }
            emitDispatchRecord(rec, enc: enc, cur: cur, alt: alt, seqLen: seqLen, startPos: startPos)
            dispatchOpCount += 1
            guard let points = byBoundary[dispatchOpCount] else { continue }
            // The projection just executed and does not write its input, so the
            // input slot is intact. Drain and read it before the next layer's
            // norm overwrites it.
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
            for point in points {
                capture.accumulate(
                    point.weightName, buffers[point.inputSlot],
                    m: Int(seqLen), k: point.k, inputIsFloat16: point.inputIsFloat16
                )
            }
            (cmdBuf, enc) = try makeCommandBufferAndEncoder()
        }
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    /// Run Metal prefill and return per-position logit rows. Only
    /// valid on packages built with `prefill.emit_all_logits=true`
    /// (see Phase 7.1) — those write `[B, vocab]` fp16 to logitsBuf
    /// and skip the argmax dispatch. Used by speculative-decode
    /// verify: K+1 logit rows from one chunked-prefill call instead
    /// of K+1 sequential decodes.
    ///
    /// `tokens.count` must be ≤ the package's `max_prefill_batch`.
    /// Returns one fp16 row per token, in order. Side effects:
    /// writes K/V at positions `[startPos, startPos +
    /// tokens.count - 1]` and updates normOutBuf for the batch.
    public func prefillAllLogits(
        tokens: [Int32], startPos: Int32
    ) throws -> [[Float16]] {
        guard supportsChunkedPrefillVerify else {
            throw SmeltRuntimeError.invalidPackage(
                "prefillAllLogits requires a package built with "
                + "prefill.emit_all_logits=true"
            )
        }
        if numDeltaLayers > 0 {
            throw SmeltRuntimeError.invalidPackage(
                "prefillAllLogits unsafe on target with "
                + "\(numDeltaLayers) recurrent layers"
            )
        }
        let vocab = vocabSize
        let maxBatch = maxPrefillBatchSize
        guard !tokens.isEmpty, tokens.count <= maxBatch else {
            throw SmeltRuntimeError.invalidPackage(
                "prefillAllLogits: tokens.count=\(tokens.count) outside "
                + "[1, \(maxBatch)]"
            )
        }
        for tok in tokens {
            guard tok >= 0, tok < vocab else {
                throw SmeltRuntimeError.invalidPackage(
                    "prefillAllLogits: token \(tok) out of vocab range "
                    + "[0, \(vocab))"
                )
            }
        }

        try ensurePrefillCapacity(tokenIds: tokens, startPos: startPos)
        guard let table = prefillDispatchTable else {
            throw SmeltRuntimeError.dispatchTableMissing(
                "no prefill_dispatches.bin — cannot prefill"
            )
        }

        // TQH-tied LM head packages can't build the argmax-only
        // verify table, so spec-decode falls into this all-logits
        // path. Arm SMELT_PROFILE_VERIFY only when spec-decode opted
        // in via profileNextPrefillAsVerify — otherwise the bench's
        // prompt prefill would consume the one-shot arming before
        // any verify dispatch runs.
        let profileCtx: SmeltVerifyProfileContext?
        if profileNextPrefillAsVerify {
            profileNextPrefillAsVerify = false
            profileCtx = tryArmVerifyProfile(recordCount: table.count)
        } else {
            profileCtx = nil
        }

        let cmdBuf: MTLCommandBuffer
        if let ctx = profileCtx {
            guard let cb = queue.makeCommandBuffer() else {
                throw SmeltRuntimeError.metalCommandBufferUnavailable
            }
            cmdBuf = cb
            interpretPrefillDispatchTableProfiled(
                table, cmdBuf: cmdBuf, tokenIds: tokens,
                seqLen: Int32(tokens.count), startPos: startPos,
                profile: ctx
            )
        } else {
            let (cb, enc) = try makeCommandBufferAndEncoder()
            cmdBuf = cb
            interpretPrefillDispatchTable(
                table, enc: enc, tokenIds: tokens,
                seqLen: Int32(tokens.count), startPos: startPos
            )
            enc.endEncoding()
        }
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let ctx = profileCtx {
            finalizeVerifyProfile(ctx)
        }

        // Pull tokens.count rows from the [B, vocab] logitsBuf.
        // Positions ≥ tokens.count contain garbage from running
        // dispatches over uninitialized normOut slices; the caller
        // never sees them.
        let logitsBuf = buffers[Int(SmeltRuntimeConfig.fixedLogitsSlot)]
        let ptr = logitsBuf.contents().bindMemory(
            to: Float16.self, capacity: vocab * tokens.count
        )
        var rows: [[Float16]] = []
        rows.reserveCapacity(tokens.count)
        for pos in 0 ..< tokens.count {
            let base = pos * vocab
            rows.append(Array(UnsafeBufferPointer(
                start: ptr.advanced(by: base), count: vocab
            )))
        }
        return rows
    }

    /// Run the same Metal prefill path as `prefillAllLogits` but leave
    /// logits resident in `logitsBuf` instead of materializing Swift arrays.
    /// Used by opt-in GPU Block Verification.
    public func prefillAllLogitsResident(
        tokens: [Int32], startPos: Int32
    ) throws {
        guard supportsChunkedPrefillVerify else {
            throw SmeltRuntimeError.invalidPackage(
                "prefillAllLogitsResident requires a package built with "
                + "prefill.emit_all_logits=true"
            )
        }
        if numDeltaLayers > 0 {
            throw SmeltRuntimeError.invalidPackage(
                "prefillAllLogitsResident unsafe on target with "
                + "\(numDeltaLayers) recurrent layers"
            )
        }
        let vocab = vocabSize
        let maxBatch = maxPrefillBatchSize
        guard !tokens.isEmpty, tokens.count <= maxBatch else {
            throw SmeltRuntimeError.invalidPackage(
                "prefillAllLogitsResident: tokens.count=\(tokens.count) outside "
                + "[1, \(maxBatch)]"
            )
        }
        for tok in tokens {
            guard tok >= 0, tok < vocab else {
                throw SmeltRuntimeError.invalidPackage(
                    "prefillAllLogitsResident: token \(tok) out of vocab range "
                    + "[0, \(vocab))"
                )
            }
        }

        try ensurePrefillCapacity(tokenIds: tokens, startPos: startPos)
        guard let table = prefillDispatchTable else {
            throw SmeltRuntimeError.dispatchTableMissing(
                "no prefill_dispatches.bin — cannot prefill"
            )
        }

        let profileCtx: SmeltVerifyProfileContext?
        if profileNextPrefillAsVerify {
            profileNextPrefillAsVerify = false
            profileCtx = tryArmVerifyProfile(recordCount: table.count)
        } else {
            profileCtx = nil
        }

        let cmdBuf: MTLCommandBuffer
        if let ctx = profileCtx {
            guard let cb = queue.makeCommandBuffer() else {
                throw SmeltRuntimeError.metalCommandBufferUnavailable
            }
            cmdBuf = cb
            interpretPrefillDispatchTableProfiled(
                table, cmdBuf: cmdBuf, tokenIds: tokens,
                seqLen: Int32(tokens.count), startPos: startPos,
                profile: ctx
            )
        } else {
            let (cb, enc) = try makeCommandBufferAndEncoder()
            cmdBuf = cb
            interpretPrefillDispatchTable(
                table, enc: enc, tokenIds: tokens,
                seqLen: Int32(tokens.count), startPos: startPos
            )
            enc.endEncoding()
        }
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let ctx = profileCtx {
            finalizeVerifyProfile(ctx)
        }
    }

    public func stageCurrentLogitsForSpecBV(row: Int) throws {
        let scratch = try ensureSpecBVTargetLogitsBuffer(rowCount: 4)
        let rowBytes = vocabSize * MemoryLayout<Float16>.stride
        try copyCurrentLogits(
            to: scratch,
            destinationOffset: row * rowBytes
        )
    }

    public func stageResidentPrefillLogitsForSpecBV(
        rowCount: Int,
        destinationStartRow: Int
    ) throws {
        guard rowCount > 0, rowCount + destinationStartRow <= 4 else {
            throw SmeltRuntimeError.invalidPackage(
                "stageResidentPrefillLogitsForSpecBV row range "
                + "\(destinationStartRow)..<\(destinationStartRow + rowCount) "
                + "outside K=3 target rows"
            )
        }
        let scratch = try ensureSpecBVTargetLogitsBuffer(rowCount: 4)
        let rowBytes = vocabSize * MemoryLayout<Float16>.stride
        let byteCount = rowCount * rowBytes
        guard let cmdBuf = queue.makeCommandBuffer(),
              let blit = cmdBuf.makeBlitCommandEncoder()
        else {
            throw SmeltRuntimeError.metalCommandBufferUnavailable
        }
        blit.copy(
            from: buffers[Int(SmeltRuntimeConfig.fixedLogitsSlot)],
            sourceOffset: 0,
            to: scratch,
            destinationOffset: destinationStartRow * rowBytes,
            size: byteCount
        )
        blit.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    /// Phase J: Daliri-coupling — sample target's gumbel-argmax from a
    /// specific row of the prefill logits buffer using the same Gumbel
    /// noise the drafter used (via shared seed). The drafter's GPU
    /// sampler runs `sample_temperature_gumbel_fp16` with
    /// `seed, position`; this method dispatches the same kernel against
    /// `buffers[fixedLogitsSlot]` at `rowIndex * vocabBytes` offset.
    /// Identical RNG → identical Gumbel noise → coupled sample.
    public func sampleGumbelArgmaxFromPrefillRow(
        rowIndex: Int,
        temperature: Float,
        seed: UInt64,
        position: Int32
    ) throws -> Int32 {
        guard let pipeline = ensureGumbelSamplingPipeline() else {
            throw SmeltRuntimeError.invalidPackage(
                "sampleGumbelArgmaxFromPrefillRow: gumbel pipeline missing; "
                + "rebuild package"
            )
        }
        guard temperature.isFinite, temperature > 0 else {
            throw SmeltRuntimeError.invalidPackage(
                "sampleGumbelArgmaxFromPrefillRow: temperature must be "
                + "finite and positive"
            )
        }
        let rowBytes = vocabSize * MemoryLayout<Float16>.stride
        let logitsBuf = buffers[Int(SmeltRuntimeConfig.fixedLogitsSlot)]
        guard rowIndex >= 0,
              (rowIndex + 1) * rowBytes <= logitsBuf.length
        else {
            throw SmeltRuntimeError.invalidPackage(
                "sampleGumbelArgmaxFromPrefillRow: row \(rowIndex) outside "
                + "logits buffer (\(logitsBuf.length / rowBytes) rows)"
            )
        }
        let resultBuf = try ensureGumbelCoupledResultBuffer()
        guard let cmdBuf = queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder()
        else {
            throw SmeltRuntimeError.metalCommandBufferUnavailable
        }
        let executionWidth = max(pipeline.threadExecutionWidth, 1)
        let maxThreads = max(pipeline.maxTotalThreadsPerThreadgroup, executionWidth)
        let preferredThreads = min(256, maxThreads)
        let threads = max(
            executionWidth,
            (preferredThreads / executionWidth) * executionWidth
        )
        var vocab = UInt32(vocabSize)
        var invTemp = 1.0 / temperature
        var seedArg = seed
        var positionArg = UInt32(bitPattern: position)
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(logitsBuf, offset: rowIndex * rowBytes, index: 0)
        enc.setBuffer(resultBuf, offset: 0, index: 1)
        enc.setBytes(&vocab, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&invTemp, length: MemoryLayout<Float>.size, index: 3)
        enc.setBytes(&seedArg, length: MemoryLayout<UInt64>.size, index: 4)
        enc.setBytes(&positionArg, length: MemoryLayout<UInt32>.size, index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        return resultBuf.contents().load(as: Int32.self)
    }

    private var gumbelCoupledResultBuffer: MTLBuffer?
    private func ensureGumbelCoupledResultBuffer() throws -> MTLBuffer {
        if let buf = gumbelCoupledResultBuffer { return buf }
        let buf = try makeSharedRuntimeBuffer(
            length: MemoryLayout<Int32>.stride,
            label: "smelt.gumbel_coupled.result"
        )
        gumbelCoupledResultBuffer = buf
        return buf
    }

    /// Phase J: Gumbel-coupled stochastic verify (Daliri et al. 2024,
    /// arXiv:2408.07978). Each of K verify positions calls the same
    /// `sample_temperature_gumbel_fp16` kernel the drafter used, with
    /// `stepSeed = baseSeed + k * 0x9E37_79B9_7F4A_7C15`. The Gumbel
    /// noise is identical to what the drafter saw, so target's argmax
    /// is the coupled sample of `p`. Accept iff equal to drafter's
    /// candidate; on disagreement commit target's pick. On full accept,
    /// bonus from row K with its own seed extension.
    public func runGumbelCoupledStep(
        candidates: [Int32],
        baseSeed: UInt64,
        position: Int32,
        temperature: Float
    ) throws -> (acceptedCount: Int, token: Int32) {
        let K = candidates.count
        precondition(K >= 1, "runGumbelCoupledStep: K must be >= 1")
        let golden: UInt64 = 0x9E37_79B9_7F4A_7C15
        for k in 0 ..< K {
            let stepSeed = baseSeed &+ UInt64(k) &* golden
            let targetPick = try sampleGumbelArgmaxFromPrefillRow(
                rowIndex: k,
                temperature: temperature,
                seed: stepSeed,
                position: position
            )
            if targetPick != candidates[k] {
                return (acceptedCount: k, token: targetPick)
            }
        }
        // All K matched; bonus from row K with extended seed.
        let bonusSeed = baseSeed &+ UInt64(K) &* golden
        let bonus = try sampleGumbelArgmaxFromPrefillRow(
            rowIndex: K,
            temperature: temperature,
            seed: bonusSeed,
            position: position
        )
        return (acceptedCount: K, token: bonus)
    }

    public func runSpecBVK3(
        draftLogits: MTLBuffer,
        candidates: [Int32],
        temperature: Float,
        seed: UInt64,
        position: Int32
    ) throws -> (acceptedCount: Int, token: Int32) {
        guard let pipeline = ensureSpecBVK3Pipeline() else {
            throw SmeltRuntimeError.invalidPackage(
                "spec_bv_k3 kernel missing from model.metallib; rebuild package"
            )
        }
        guard candidates.count == 3 else {
            throw SmeltRuntimeError.invalidPackage(
                "runSpecBVK3 requires exactly 3 candidates, got \(candidates.count)"
            )
        }
        guard temperature.isFinite, temperature > 0 else {
            throw SmeltRuntimeError.invalidPackage(
                "runSpecBVK3 temperature must be finite and positive"
            )
        }
        let targetLogits = try ensureSpecBVTargetLogitsBuffer(rowCount: 4)
        let candidateBuffer = try ensureSpecBVCandidatesBuffer()
        let outputBuffer = try ensureSpecBVOutputBuffer()
        _ = candidates.withUnsafeBufferPointer { src in
            memcpy(
                candidateBuffer.contents(),
                src.baseAddress!,
                candidates.count * MemoryLayout<Int32>.stride
            )
        }
        memset(outputBuffer.contents(), 0, 2 * MemoryLayout<Int32>.stride)

        guard let cmdBuf = queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder()
        else {
            throw SmeltRuntimeError.metalCommandBufferUnavailable
        }

        var vocab = UInt32(vocabSize)
        var invTemp = 1 / temperature
        var seedValue = seed
        var posBits = UInt32(bitPattern: position)
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(targetLogits, offset: 0, index: 0)
        enc.setBuffer(draftLogits, offset: 0, index: 1)
        enc.setBuffer(candidateBuffer, offset: 0, index: 2)
        enc.setBuffer(outputBuffer, offset: 0, index: 3)
        enc.setBytes(&vocab, length: MemoryLayout<UInt32>.stride, index: 4)
        enc.setBytes(&invTemp, length: MemoryLayout<Float>.stride, index: 5)
        enc.setBytes(&seedValue, length: MemoryLayout<UInt64>.stride, index: 6)
        enc.setBytes(&posBits, length: MemoryLayout<UInt32>.stride, index: 7)
        let executionWidth = max(pipeline.threadExecutionWidth, 1)
        let maxThreads = max(pipeline.maxTotalThreadsPerThreadgroup, executionWidth)
        let preferredThreads = min(256, maxThreads)
        let threads = max(
            executionWidth,
            (preferredThreads / executionWidth) * executionWidth
        )
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = outputBuffer.contents().bindMemory(to: Int32.self, capacity: 2)
        let accepted = Int(ptr[0])
        guard accepted >= 0, accepted <= 3 else {
            throw SmeltRuntimeError.invalidPackage(
                "spec_bv_k3 returned invalid acceptedCount \(accepted)"
            )
        }
        return (accepted, ptr[1])
    }

    /// Cache key is the runtime `verifySeqLen` (= `tokens.count`); a
    /// different live seqLen invalidates the cached ICB because grid
    /// sizes and seqLen-derived constants are baked at build time.
    private func ensureVerifyICBBuilt(
        table: UnsafeBufferPointer<SmeltDispatchRecord>,
        verifySeqLen: Int32
    ) {
        guard verifyICBEnabled else { return }
        switch verifyICB {
        case .built(let result) where result.stats.verifySeqLen == verifySeqLen:
            return
        case .failed:
            return
        default:
            break
        }

        if let result = buildVerifyICB(table: table, verifySeqLen: verifySeqLen) {
            verifyICB = .built(result)
            let s = result.stats
            print(
                "[SmeltVerifyICB] static=\(s.staticDispatchCount) "
                + "dynamic=\(s.dynamicDispatchCount) "
                + "swap=\(s.swapCount) "
                + "skip_minSeqLen=\(s.skippedByMinSeqLen) "
                + "skip_zeroGrid=\(s.skippedByZeroGrid) "
                + "skip_zeroMod=\(s.skippedBySkipIfZero) "
                + "total=\(s.totalRecords) "
                + "verifySeqLen=\(s.verifySeqLen) "
                + "constsBuf=\(s.constantsBufferBytes)B"
            )
        } else {
            verifyICB = .failed
            print("[SmeltVerifyICB] build failed or no static records")
        }
    }

    /// Drop the cached verify ICB. Call when underlying buffers or
    /// cache-capacity-derived constants change — the ICB has baked-in
    /// `MTLBuffer` references and `cacheSeqCapacity` literals from build
    /// time, so any mutation of those would leave the ICB stale.
    private func invalidateVerifyICB() {
        verifyICB = .pending
    }

    /// Writes K/V and normOut exactly like `prefillAllLogits`, but
    /// the final LM head is a batched argmax kernel + small GPU
    /// reduction — no materialized `[B, vocab]` logits.
    public func prefillVerifyArgmax(
        tokens: [Int32], startPos: Int32
    ) throws -> [Int32] {
        guard supportsChunkedPrefillVerifyArgmax else {
            throw SmeltRuntimeError.invalidPackage(
                "prefillVerifyArgmax requires a package with "
                + "prefill_verify_argmax_dispatches.bin"
            )
        }
        let vocab = vocabSize
        let maxBatch = maxPrefillBatchSize
        guard !tokens.isEmpty, tokens.count <= maxBatch else {
            throw SmeltRuntimeError.invalidPackage(
                "prefillVerifyArgmax: tokens.count=\(tokens.count) outside "
                + "[1, \(maxBatch)]"
            )
        }
        for tok in tokens {
            guard tok >= 0, tok < vocab else {
                throw SmeltRuntimeError.invalidPackage(
                    "prefillVerifyArgmax: token \(tok) out of vocab range "
                    + "[0, \(vocab))"
                )
            }
        }

        try ensurePrefillCapacity(tokenIds: tokens, startPos: startPos)
        guard let table = prefillVerifyArgmaxDispatchTable else {
            throw SmeltRuntimeError.dispatchTableMissing(
                "no prefill_verify_argmax_dispatches.bin — cannot run verify argmax"
            )
        }
        try ensurePipelineStatesMaterialized(
            for: table,
            tableName: "prefill_verify_argmax_dispatches.bin"
        )

        // Profile mode forces the interpret path: per-dispatch GPU
        // sampling inside an MTLIndirectCommandBuffer needs an in-ICB
        // sample API Metal doesn't expose, so we can't get per-command
        // timing through the ICB executor.
        let profileCtx = tryArmVerifyProfile(recordCount: table.count)
        if profileCtx == nil {
            ensureVerifyICBBuilt(table: table, verifySeqLen: Int32(tokens.count))
        }

        let argmaxBuf = buffers[Int(config.argmaxSlot)]
        memset(
            argmaxBuf.contents(),
            0,
            tokens.count * MemoryLayout<Int32>.stride
        )

        let cmdBuf: MTLCommandBuffer
        if let ctx = profileCtx {
            guard let cb = queue.makeCommandBuffer() else {
                throw SmeltRuntimeError.metalCommandBufferUnavailable
            }
            cmdBuf = cb
            interpretPrefillDispatchTableProfiled(
                table, cmdBuf: cmdBuf, tokenIds: tokens,
                seqLen: Int32(tokens.count), startPos: startPos,
                profile: ctx
            )
        } else {
            let (cb, enc) = try makeCommandBufferAndEncoder()
            cmdBuf = cb
            if case .built(let icbResult) = verifyICB {
                writePrefillTokenIds(tokens)
                executeVerifyICB(
                    icbResult,
                    enc: enc,
                    seqLen: Int32(tokens.count), startPos: startPos
                )
            } else {
                interpretPrefillDispatchTable(
                    table, enc: enc, tokenIds: tokens,
                    seqLen: Int32(tokens.count), startPos: startPos
                )
            }
            enc.endEncoding()
        }
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let ctx = profileCtx {
            finalizeVerifyProfile(ctx)
        }

        let ptr = argmaxBuf.contents().bindMemory(
            to: Int32.self, capacity: tokens.count
        )
        return Array(UnsafeBufferPointer(start: ptr, count: tokens.count))
    }

    /// Profile one Metal prefill chunk: splits CPU encode vs GPU execute vs readback.
    /// Returns timings for a single prefillStep invocation.
    public func profilePrefillStep(
        tokenIds: [Int32], startPos: Int32
    ) throws -> (token: Int32, cpuMs: Double, gpuMs: Double, readMs: Double, pureGpuMs: Double) {
        try ensurePrefillCapacity(tokenIds: tokenIds, startPos: startPos)

        let t0 = CFAbsoluteTimeGetCurrent()

        guard let table = prefillDispatchTable else {
            throw SmeltRuntimeError.dispatchTableMissing(
                "no prefill_dispatches.bin — cannot prefill"
            )
        }
        let (cmdBuf, enc) = try makeCommandBufferAndEncoder()

        interpretPrefillDispatchTable(
            table, enc: enc, tokenIds: tokenIds,
            seqLen: Int32(tokenIds.count), startPos: startPos
        )

        enc.endEncoding()
        let t1 = CFAbsoluteTimeGetCurrent()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        let t2 = CFAbsoluteTimeGetCurrent()

        let token = buffers[Int(config.argmaxSlot)]
            .contents().load(as: Int32.self)
        let t3 = CFAbsoluteTimeGetCurrent()

        let pureGpu = (cmdBuf.gpuEndTime - cmdBuf.gpuStartTime) * 1_000

        return (
            token: token,
            cpuMs: (t1 - t0) * 1_000,
            gpuMs: (t2 - t1) * 1_000,
            readMs: (t3 - t2) * 1_000,
            pureGpuMs: pureGpu
        )
    }

    /// Debug prefill: execute only the first N prefill dispatches, then stop.
    /// Returns argmax result (may be meaningless if stopped before the LM head).
    public func debugPrefillStep(
        tokenIds: [Int32],
        startPos: Int32,
        maxDispatches: Int
    ) throws -> Int32 {
        try ensurePrefillCapacity(tokenIds: tokenIds, startPos: startPos)

        guard let table = prefillDispatchTable else {
            throw SmeltRuntimeError.dispatchTableMissing(
                "no prefill_dispatches.bin — cannot prefill"
            )
        }
        let (cmdBuf, enc) = try makeCommandBufferAndEncoder()

        let recordCount = Self.recordCount(
            coveringDispatches: maxDispatches,
            in: table
        )
        let truncated = UnsafeBufferPointer(
            start: table.baseAddress,
            count: recordCount
        )
        interpretPrefillDispatchTable(
            truncated,
            enc: enc,
            tokenIds: tokenIds,
            seqLen: Int32(tokenIds.count),
            startPos: startPos
        )

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error { throw error }

        return buffers[Int(config.argmaxSlot)]
            .contents().load(as: Int32.self)
    }

    private static func recordCount(
        coveringDispatches maxDispatches: Int,
        in table: UnsafeBufferPointer<SmeltDispatchRecord>
    ) -> Int {
        guard maxDispatches > 0 else { return 0 }
        var seenDispatches = 0
        var recordCount = 0
        for rec in table {
            recordCount += 1
            if rec.opKind == SmeltDispatchRecord.opDispatch {
                seenDispatches += 1
                if seenDispatches >= maxDispatches {
                    break
                }
            }
        }
        return min(recordCount, table.count)
    }

    /// Return the last few prefill dispatch names up to `endExclusive`.
    public func prefillDispatchWindow(
        endingAt endExclusive: Int,
        count: Int = 12
    ) -> [String] {
        guard let tableData = prefillDispatchTableData else { return [] }
        let stride = MemoryLayout<SmeltDispatchRecord>.stride
        let tableCount = tableData.count / stride
        let end = min(endExclusive, tableCount)
        let start = max(0, end - count)

        return tableData.withUnsafeBytes { ptr in
            let table = ptr.bindMemory(to: SmeltDispatchRecord.self)
            return (start..<end).map { idx in
                let rec = table[idx]
                if rec.opKind == SmeltDispatchRecord.opSwap {
                    return "\(idx): swap"
                }
                let name = Int(rec.pipeline) < manifest.pipelines.count
                    ? manifest.pipelines[Int(rec.pipeline)]
                    : "pipeline_\(rec.pipeline)"
                return "\(idx): \(name)"
            }
        }
    }

    /// Without side effects, predicts whether `emitDispatchRecord` would
    /// emit a dispatch for `rec` (vs short-circuit via minSeqLen guard,
    /// skipDispatch constant, or zero-grid). Used by the profile path
    /// to avoid allocating an encoder + sample-buffer slot for records
    /// that would skip.
    func willDispatchRecord(
        _ rec: SmeltDispatchRecord, seqLen: Int32, startPos: Int32
    ) -> Bool {
        if rec.opKind == SmeltDispatchRecord.opSwap { return false }
        if rec.minSeqLen > 0 && UInt32(seqLen) < UInt32(rec.minSeqLen) {
            return false
        }
        var skipDispatch = false
        for cidx in 0..<Int(rec.constantCount) {
            let con = getConstant(rec, index: cidx)
            _ = resolvePrefillConstant(
                con, seqLen: seqLen, startPos: startPos,
                skipDispatch: &skipDispatch
            )
            if skipDispatch { return false }
        }
        let gridW = resolvePrefillGrid(rec.gridW, kind: rec.gridWKind, seqLen: seqLen)
        let gridH = resolvePrefillGrid(rec.gridH, kind: rec.gridHKind, seqLen: seqLen)
        let gridD = resolvePrefillGrid(rec.gridD, kind: rec.gridDKind, seqLen: seqLen)
        return gridW != 0 && gridH != 0 && gridD != 0
    }

    /// Encode one prefill dispatch record into the compute encoder. The
    /// caller owns the cur/alt double-buffer state and must handle
    /// `opSwap` records before invoking this method. Returns true when
    /// a dispatch was actually emitted; false on minSeqLen guard /
    /// skipDispatch constant / zero-grid short-circuit.
    @discardableResult
    func emitDispatchRecord(
        _ rec: SmeltDispatchRecord,
        enc: MTLComputeCommandEncoder,
        cur: Int, alt: Int,
        seqLen: Int32, startPos: Int32
    ) -> Bool {
        if rec.minSeqLen > 0 && UInt32(seqLen) < UInt32(rec.minSeqLen) {
            return false
        }
        // Variable-length prefill (Phase 4 U1): the SIMD cross-chunk attention caps at a
        // 2048-row cache (its threadgroup sc[] stage). When THIS chunk's cache length
        // (startPos+seqLen) exceeds that, swap in the uncapped scalar-streaming variant —
        // numerically equivalent, no cap. Mirrors the hand causalGQAAttnDispatch's simd/
        // scalar choice; the scalar kernel runs correctly on the SIMD grid (lanes with
        // t>=frames return early). Keyed on a precomputed index, so it's inert (false branch)
        // for every other kernel and for packages without the scalar variant. INVARIANT: the
        // scalar kernel's acc[] registers assume headDim <= 128 — held because the ONLY emitter
        // of causal_gqa_attn_cached_f32 is the dense-trunk FP32 prefill (DenseTrunkGuards
        // caps headDim <= 128). A future emitter using this pipeline with a wider head would
        // need a headDim gate here.
        var pipelineIndex = Int(rec.pipeline)
        if pipelineIndex == cachedAttnCappedPipelineIndex,
           Int(startPos) + Int(seqLen) > Self.cachedAttnSimdCap,
           let scalarIndex = cachedAttnScalarPipelineIndex {
            pipelineIndex = scalarIndex
            Self.cachedAttnScalarSubstitutions += 1
        }
        enc.setComputePipelineState(pipelineState(for: pipelineIndex))
        for bidx in 0..<Int(rec.bufferCount) {
            let buf = getBuffer(rec, index: bidx)
            let slot = resolveDispatchBufferSlot(buf, cur: cur, alt: alt)
            let offset = resolveDispatchBufferOffset(
                buf, seqLen: seqLen, startPos: startPos
            )
            bindBuffer(
                enc, slot: slot, offset: offset, index: Int(buf.bindingIndex)
            )
        }
        var skipDispatch = false
        for cidx in 0..<Int(rec.constantCount) {
            let con = getConstant(rec, index: cidx)
            var val = resolvePrefillConstant(
                con,
                seqLen: seqLen,
                startPos: startPos,
                skipDispatch: &skipDispatch
            )
            if con.bindingIndex != UInt8.max {
                enc.setBytes(&val, length: 4, index: Int(con.bindingIndex))
            }
        }
        if skipDispatch { return false }
        let gridW = resolvePrefillGrid(rec.gridW, kind: rec.gridWKind, seqLen: seqLen)
        let gridH = resolvePrefillGrid(rec.gridH, kind: rec.gridHKind, seqLen: seqLen)
        let gridD = resolvePrefillGrid(rec.gridD, kind: rec.gridDKind, seqLen: seqLen)
        if gridW == 0 || gridH == 0 || gridD == 0 { return false }
        let grid = MTLSize(width: gridW, height: gridH, depth: gridD)
        let tg = MTLSize(width: Int(rec.tgW), height: Int(rec.tgH), depth: Int(rec.tgD))
        if rec.dispatchStyle == SmeltDispatchRecord.styleThreadgroups {
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        } else {
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        }
        return true
    }

    /// Interpret the prefill dispatch table into Metal compute commands.
    /// Same flat loop as decode, but resolves seqLen/startPos constants
    /// and supports prefill-specific offset kinds.
    private func interpretPrefillDispatchTable(
        _ table: UnsafeBufferPointer<SmeltDispatchRecord>,
        enc: MTLComputeCommandEncoder,
        tokenIds: [Int32],
        seqLen: Int32,
        startPos: Int32,
        skipInitialEmbeddingGather: Bool = false
    ) {
        var cur = 0
        var alt = 1
        writePrefillTokenIds(tokenIds)
        for idx in 0..<table.count {
            let rec = table[idx]
            if skipInitialEmbeddingGather && idx == 0 {
                continue
            }
            if rec.opKind == SmeltDispatchRecord.opSwap {
                swap(&cur, &alt)
                continue
            }
            emitDispatchRecord(
                rec, enc: enc, cur: cur, alt: alt,
                seqLen: seqLen, startPos: startPos
            )
        }
    }

    /// Profile variant: each dispatch goes in its own compute encoder
    /// whose pass descriptor wires sample-buffer attachments at the
    /// encoder's start/end boundaries. Apple Silicon only supports
    /// `.atStageBoundary` sampling, so one-encoder-per-dispatch is the
    /// only way to get per-dispatch GPU timing. CPU overhead is high
    /// (one encoder + one descriptor per record) but the path only
    /// fires on the first profile-armed prefillVerifyArgmax call.
    func interpretPrefillDispatchTableProfiled(
        _ table: UnsafeBufferPointer<SmeltDispatchRecord>,
        cmdBuf: MTLCommandBuffer,
        tokenIds: [Int32],
        seqLen: Int32,
        startPos: Int32,
        profile: SmeltVerifyProfileContext
    ) {
        var cur = 0
        var alt = 1
        writePrefillTokenIds(tokenIds)
        for idx in 0..<table.count {
            let rec = table[idx]
            if rec.opKind == SmeltDispatchRecord.opSwap {
                swap(&cur, &alt)
                continue
            }
            guard willDispatchRecord(rec, seqLen: seqLen, startPos: startPos)
            else { continue }

            // Try to encode with sample-buffer attachment for timing.
            // If anything declines (no more profile slots, attachment
            // descriptor nil, encoder creation failure), fall back to a
            // plain encoder so the verify pass still produces correct
            // output. The CSV will show fewer than the total dispatch
            // count for that run; functional correctness is preserved.
            let name = pipelineName(for: rec.pipeline)
            let sampledEnc = profile.shouldSample(pipelineName: name)
                ? makeSampledEncoder(cmdBuf: cmdBuf, profile: profile)
                : nil
            let enc = sampledEnc ?? cmdBuf.makeComputeCommandEncoder()
            guard let enc else { continue }
            emitDispatchRecord(
                rec, enc: enc, cur: cur, alt: alt,
                seqLen: seqLen, startPos: startPos
            )
            enc.endEncoding()
            if sampledEnc != nil {
                profile.kernelNames.append(name)
                profile.dispatchCount += 1
            }
        }
    }

    private func makeSampledEncoder(
        cmdBuf: MTLCommandBuffer, profile: SmeltVerifyProfileContext
    ) -> MTLComputeCommandEncoder? {
        guard profile.dispatchCount < profile.maxDispatches else { return nil }
        let sampleSlot = profile.dispatchCount
        let desc = MTLComputePassDescriptor()
        guard let attach = desc.sampleBufferAttachments[0] else { return nil }
        attach.sampleBuffer = profile.buffer
        attach.startOfEncoderSampleIndex = 2 * sampleSlot
        attach.endOfEncoderSampleIndex = 2 * sampleSlot + 1
        return cmdBuf.makeComputeCommandEncoder(descriptor: desc)
    }

    func resolveDispatchBufferSlot(
        _ buf: SmeltBufferRecord, cur: Int, alt: Int
    ) -> Int {
        if buf.slot >= 0 { return Int(buf.slot) }
        if buf.slot == SmeltBufferRecord.slotCur { return cur }
        return alt
    }

    func resolveDispatchBufferOffset(
        _ buf: SmeltBufferRecord, seqLen: Int32, startPos: Int32
    ) -> Int {
        switch buf.offsetKind {
        case 0:
            return Int(buf.offset)
        case 1:
            return Int(startPos) * Int(buf.offset)
        case 2:
            let stride = Int(buf.offset & 0xFFFF_FFFF)
            let addend = Int(buf.offset >> 32)
            return Int(startPos) * stride + addend
        case 3:
            return Int(seqLen - 1) * Int(buf.offset)
        case 4:
            let stride = Int(buf.offset & 0xFFFF_FFFF)
            let divisor = max(Int(buf.offset >> 32), 1)
            return (Int(seqLen) / divisor) * stride
        default:
            // Unknown kinds are an emitter/runtime version mismatch, not data:
            // fail loud rather than silently treat the payload as a literal
            // offset (the silent-kind-fallthrough class — cf. the dtype-dispatch
            // and checkpoint-coverage invariants).
            preconditionFailure(
                "unknown dispatch offsetKind \(buf.offsetKind) — emitter/runtime mismatch")
        }
    }

    func resolvePrefillGrid(
        _ literal: UInt32,
        kind: UInt8,
        seqLen: Int32
    ) -> Int {
        let s = max(Int(seqLen), 0)
        switch kind {
        case SmeltDispatchRecord.gridSeqLen:
            return s
        case SmeltDispatchRecord.gridSeqLenMulLiteral:
            return s * Int(literal)
        case SmeltDispatchRecord.gridSeqLenCeilDivLiteral:
            let raw = Int(literal)
            let divisor = max(raw & 0x7FFF_FFFF, 1)
            if (literal & 0x8000_0000) != 0 {
                return s / divisor
            }
            return (s + divisor - 1) / divisor
        default:
            return Int(literal)
        }
    }

    // MARK: - Weight loading (mmap)

    /// mmap weights.bin into a single MTLBuffer with munmap deallocator.
    /// Kernel names that require function constants (FC_COLS, FC_GROUP_SIZE).
    /// Base entries for these are placeholders — all dispatches use
    /// specialized "name:cols:groupSize" variants.
    private static let fcRequiredKernels: Set<String> = [
        "fused_lut_matvec", "fused_lut_matvec_add", "fused_gate_up_swiglu",
        "fused_dual_lut_matvec", "affine_matvec", "fused_affine_matvec_add",
        "fused_dual_affine_matvec", "fused_affine_gate_up_swiglu",
        "fused_affine_gate_up_geglu",
        "fused_rms_norm_affine_gate_up_swiglu", "fused_rms_norm_affine_matvec",
        "norm_scale_affine_gate_up_swiglu", "norm_scale_affine_matvec",
        "atomic_norm_affine_gate_up_swiglu", "atomic_norm_affine_matvec"
    ]

    /// Materialize the MTLFunction for a manifest pipeline entry. Returns nil
    /// for entries whose function can't be built (never dispatched — the
    /// runtime substitutes the previous pipeline). Shared with the compiler's
    /// binary-archive pre-compilation so the two can't drift on entry syntax.
    public static func pipelineFunction(
        for entry: String,
        library: MTLLibrary
    ) throws -> MTLFunction? {
        if entry.contains(":") {
            // Function-constant-specialized pipeline: "name:cols:groupSize"
            let parts = entry.split(separator: ":")
            guard parts.count == 3,
                  let cols = UInt32(parts[1]),
                  let groupSize = UInt32(parts[2])
            else {
                throw SmeltRuntimeError.invalidPackage(
                    "Malformed specialized pipeline entry: '\(entry)'"
                )
            }
            let constants = MTLFunctionConstantValues()
            var colsVal = cols
            var gsVal = groupSize
            constants.setConstantValue(&colsVal, type: .uint, index: 0)
            constants.setConstantValue(&gsVal, type: .uint, index: 1)
            return try library.makeFunction(
                name: String(parts[0]), constantValues: constants
            )
        }
        if fcRequiredKernels.contains(entry) {
            // Base entry for a function-constant kernel — dummy values.
            let constants = MTLFunctionConstantValues()
            var dummyCols: UInt32 = 1
            var dummyGs: UInt32 = 1
            constants.setConstantValue(&dummyCols, type: .uint, index: 0)
            constants.setConstantValue(&dummyGs, type: .uint, index: 1)
            return try? library.makeFunction(name: entry, constantValues: constants)
        }
        return library.makeFunction(name: entry)
    }

    static func makePipelineState(
        function fn: MTLFunction,
        device: MTLDevice,
        icbCapable: Bool,
        archives: [any MTLBinaryArchive] = []
    ) throws -> MTLComputePipelineState {
        guard icbCapable || !archives.isEmpty else {
            return try device.makeComputePipelineState(function: fn)
        }
        // Pipeline states encoded into an MTLIndirectCommandBuffer must be
        // built with `supportIndirectCommandBuffers = true` on their
        // descriptor — Metal validation rejects ICB encodes otherwise.
        let desc = MTLComputePipelineDescriptor()
        desc.computeFunction = fn
        desc.supportIndirectCommandBuffers = icbCapable
        if !archives.isEmpty { desc.binaryArchives = archives }
        return try device.makeComputePipelineState(
            descriptor: desc, options: [], reflection: nil
        )
    }

    private static func loadPipelineArchives(
        device: MTLDevice,
        archivePath: String?
    ) -> [any MTLBinaryArchive] {
        guard let archivePath else { return [] }
        let descriptor = MTLBinaryArchiveDescriptor()
        descriptor.url = URL(fileURLWithPath: archivePath)
        guard let archive = try? device.makeBinaryArchive(descriptor: descriptor)
        else { return [] }
        return [archive]
    }

    private static func compilePipelineSlot(
        entry: String,
        library: MTLLibrary,
        device: MTLDevice,
        icbCapable: Bool,
        archives: [any MTLBinaryArchive]
    ) throws -> PipelineSlot {
        guard let fn = try pipelineFunction(for: entry, library: library)
        else { return .fallbackToPrevious }
        return .ready(
            try makePipelineState(
                function: fn,
                device: device,
                icbCapable: icbCapable,
                archives: archives
            )
        )
    }

    private typealias DispatchTableLoad = (
        data: Data,
        table: UnsafeBufferPointer<SmeltDispatchRecord>
    )

    private static func loadDispatchTable(
        packagePath: String,
        fileName: String
    ) throws -> DispatchTableLoad? {
        let tablePath = "\(packagePath)/\(fileName)"
        guard FileManager.default.fileExists(atPath: tablePath) else {
            return nil
        }

        let data = try Data(contentsOf: URL(fileURLWithPath: tablePath))
        let stride = MemoryLayout<SmeltDispatchRecord>.stride
        guard data.count % stride == 0 else {
            throw SmeltRuntimeError.invalidPackage(
                "\(fileName) size \(data.count) is not a multiple of "
                    + "dispatch stride \(stride)"
            )
        }

        let count = data.count / stride
        let table = data.withUnsafeBytes { ptr in
            UnsafeBufferPointer(
                start: ptr.baseAddress?.assumingMemoryBound(
                    to: SmeltDispatchRecord.self
                ),
                count: count
            )
        }
        return (data: data, table: table)
    }

    private static func validateDeclaredPrefillArtifacts(
        packagePath: String,
        manifest: SmeltManifest,
        prefillDispatchTable: UnsafeBufferPointer<SmeltDispatchRecord>?
    ) throws {
        guard let prefill = manifest.prefill else { return }
        if prefill.engine == "metal" {
            guard prefillDispatchTable != nil else {
                throw SmeltRuntimeError.invalidPackage(
                    "manifest declares metal prefill but prefill_dispatches.bin is missing"
                )
            }
            return
        }

        let fm = FileManager.default
        let modelPath = "\(packagePath)/\(prefill.modelPath)"
        guard !prefill.modelPath.isEmpty, fm.fileExists(atPath: modelPath) else {
            throw SmeltRuntimeError.invalidPackage(
                "manifest declares \(prefill.engine) prefill but \(prefill.modelPath) is missing"
            )
        }
        guard fm.fileExists(atPath: "\(packagePath)/cache/meta.json") else {
            throw SmeltRuntimeError.invalidPackage(
                "manifest declares \(prefill.engine) prefill but cache/meta.json is missing"
            )
        }
    }

    static func requiredPipelineIndices(
        manifest: SmeltManifest,
        dispatchTables: [
            (name: String, table: UnsafeBufferPointer<SmeltDispatchRecord>?)
        ],
        eager: Bool
    ) throws -> Set<Int> {
        let allIndices = Set(manifest.pipelines.indices)
        guard !eager else { return allIndices }

        var required: Set<Int> = []
        for (tableName, maybeTable) in dispatchTables {
            guard let table = maybeTable else { continue }
            for rec in table where rec.opKind == SmeltDispatchRecord.opDispatch {
                let index = Int(rec.pipeline)
                guard index >= 0, index < manifest.pipelines.count else {
                    throw SmeltRuntimeError.invalidPackage(
                        "\(tableName) references pipeline \(index), "
                            + "but manifest has \(manifest.pipelines.count)"
                    )
                }
                required.insert(index)
            }
        }

        guard !required.isEmpty else { return allIndices }

        if let cappedIndex = manifest.pipelines.firstIndex(
            of: "causal_gqa_attn_cached_f32"
        ),
           required.contains(cappedIndex),
           let scalarIndex = manifest.pipelines.firstIndex(
            of: "causal_gqa_attn_cached_scalar_f32"
           ) {
            required.insert(scalarIndex)
        }

        return required
    }

    private func finishStartupPipelineMaterialization() throws {
        pipelineMaterializationLock.lock()
        let startupCompilation = self.startupPipelineCompilation
        pipelineMaterializationLock.unlock()
        guard let startupCompilation else { return }

        let traceStartup =
            ProcessInfo.processInfo.environment["SMELT_STARTUP_TRACE"] == "1"
        let waitStart = traceStartup ? CFAbsoluteTimeGetCurrent() : 0
        let compiled = startupCompilation.waitForResults()
        if traceStartup {
            let waitMs = (CFAbsoluteTimeGetCurrent() - waitStart) * 1000
            let formattedWait = String(format: "%+7.1fms", waitMs)
            fputs(
                "startup: \(formattedWait)  pipeline materialization wait\n",
                stderr
            )
        }
        pipelineMaterializationLock.lock()
        defer { pipelineMaterializationLock.unlock() }
        guard self.startupPipelineCompilation === startupCompilation else { return }
        if !compiled.archives.isEmpty {
            self.pipelineArchives = compiled.archives
        }

        for (index, result) in compiled.results.enumerated() {
            guard let result else { continue }
            switch try result.get() {
            case .ready(let state):
                pipelines[index] = state
                materializedPipelines[index] = true
            case .fallbackToPrevious:
                if index > 0 {
                    try materializePipelineStateLocked(index - 1)
                    pipelines[index] = pipelines[index - 1]
                }
                materializedPipelines[index] = true
            }
        }
        self.startupPipelineCompilation = nil
    }

    func ensurePipelineStatesMaterialized(
        for table: UnsafeBufferPointer<SmeltDispatchRecord>,
        tableName: String
    ) throws {
        try finishStartupPipelineMaterialization()
        let required = try Self.requiredPipelineIndices(
            manifest: manifest,
            dispatchTables: [(tableName, table)],
            eager: false
        )
        try materializePipelineStates(indices: required)
    }

    private func materializePipelineStates(indices: Set<Int>) throws {
        let missing = indices.filter {
            $0 >= 0 && $0 < materializedPipelines.count
                && !materializedPipelines[$0]
        }
        guard !missing.isEmpty else { return }

        pipelineMaterializationLock.lock()
        defer { pipelineMaterializationLock.unlock() }

        for index in missing.sorted() {
            try materializePipelineStateLocked(index)
        }
    }

    private func materializeAllPipelineStates() throws {
        try materializePipelineStates(indices: Set(manifest.pipelines.indices))
    }

    private func materializePipelineStateLocked(_ index: Int) throws {
        guard index >= 0, index < manifest.pipelines.count else {
            throw SmeltRuntimeError.invalidPackage(
                "pipeline index \(index) outside manifest pipeline table"
            )
        }
        if materializedPipelines[index] { return }

        if let fn = try Self.pipelineFunction(
            for: manifest.pipelines[index],
            library: pipelineLibrary
        ) {
            pipelines[index] = try Self.makePipelineState(
                function: fn,
                device: device,
                icbCapable: pipelineICBCapable,
                archives: pipelineArchives
            )
        } else if index > 0 {
            try materializePipelineStateLocked(index - 1)
            guard pipelines[index - 1] != nil else {
                throw SmeltRuntimeError.invalidPackage(
                    "pipeline \(index) fell back to previous pipeline, "
                        + "but pipeline \(index - 1) was not materialized"
                )
            }
            pipelines[index] = pipelines[index - 1]
        } else {
            pipelines[index] = try Self.makePipelineState(
                function: pipelineLibrary.makeFunction(
                    name: pipelineLibrary.functionNames.first!
                )!,
                device: device,
                icbCapable: pipelineICBCapable,
                archives: pipelineArchives
            )
        }
        materializedPipelines[index] = true
    }

    private func makeStandalonePipeline(named name: String) throws -> MTLComputePipelineState? {
        guard let fn = pipelineLibrary.makeFunction(name: name) else { return nil }
        return try Self.makePipelineState(
            function: fn,
            device: device,
            icbCapable: pipelineICBCapable,
            archives: pipelineArchives
        )
    }

    private func ensureTemperatureSamplingPipeline() -> MTLComputePipelineState? {
        pipelineMaterializationLock.lock()
        defer { pipelineMaterializationLock.unlock() }
        if let pipeline = temperatureSamplingPipeline { return pipeline }
        temperatureSamplingPipeline =
            try? makeStandalonePipeline(named: "sample_temperature_fp16")
        return temperatureSamplingPipeline
    }

    private func ensureGumbelSamplingPipeline() -> MTLComputePipelineState? {
        pipelineMaterializationLock.lock()
        defer { pipelineMaterializationLock.unlock() }
        if let pipeline = gumbelSamplingPipeline { return pipeline }
        gumbelSamplingPipeline =
            try? makeStandalonePipeline(named: "sample_temperature_gumbel_fp16")
        return gumbelSamplingPipeline
    }

    private func activeSamplingPipelineForSelection() -> MTLComputePipelineState? {
        // Default to the Gumbel-max sampler (single parallel pass; same
        // distribution as the inverse-CDF kernel). The opt-out exists
        // for bit-exact reproduction of pre-Phase-G samples.
        let useLegacySampler = ProcessInfo.processInfo
            .environment["SMELT_LEGACY_SAMPLER"] == "1"
        if useLegacySampler {
            return ensureTemperatureSamplingPipeline()
        }
        return ensureGumbelSamplingPipeline() ?? ensureTemperatureSamplingPipeline()
    }

    private func ensureSpecBVK3Pipeline() -> MTLComputePipelineState? {
        pipelineMaterializationLock.lock()
        defer { pipelineMaterializationLock.unlock() }
        if let pipeline = specBVK3Pipeline { return pipeline }
        specBVK3Pipeline = try? makeStandalonePipeline(named: "spec_bv_k3")
        return specBVK3Pipeline
    }

    func pipelineState(for index: Int) -> MTLComputePipelineState {
        precondition(
            index >= 0 && index < pipelines.count,
            "pipeline index \(index) outside manifest pipeline table"
        )
        precondition(
            index < materializedPipelines.count && materializedPipelines[index],
            "pipeline index \(index) was not materialized at startup; "
                + "call ensurePipelineStatesMaterialized before dispatch"
        )
        guard let pipeline = pipelines[index] else {
            preconditionFailure(
                "pipeline index \(index) was marked materialized without a state"
            )
        }
        return pipeline
    }

    private func writePrefillTokenIds(_ tokenIds: [Int32]) {
        guard config.tokenIdsBatchSlot >= 0 else { return }
        let batchSlot = Int(config.tokenIdsBatchSlot)
        let batchBuf = buffers[batchSlot]
        let capacity = batchBuf.length / MemoryLayout<Int32>.stride
        let idPtr = batchBuf.contents().bindMemory(
            to: Int32.self,
            capacity: capacity
        )
        for i in 0..<capacity {
            idPtr[i] = i < tokenIds.count ? tokenIds[i] : 0
        }
    }

    private func ensurePrefillCapacity(tokenIds: [Int32], startPos: Int32) throws {
        precondition(
            startPos >= 0 && Int(startPos) + tokenIds.count <= Int(config.contextLimit),
            "prefill range out of context limit [0, \(config.contextLimit))"
        )
        // Grow-only within a request: shrinking here would invalidate the
        // verify ICB on every round (new MTLBuffer instances) and waste
        // the per-round encode pre-allocation. Fresh requests still
        // shrink via prepareForRequest(preserveContext: false).
        let neededBatch = clampedBatchCapacity(tokenIds.count)
        let neededContext = clampedContextCapacity(Int(startPos) + tokenIds.count)
        try resizeRequestScopedBuffers(
            batchCapacity: max(neededBatch, currentBatchCapacity),
            contextCapacity: max(neededContext, currentContextCapacity),
            preserveContext: true
        )
    }

    /// Grow request-scoped capacity (batch + context, grow-only) for a prefill of
    /// `seqLen` tokens at `startPos`. The caller's capacity-ensure for the
    /// `encodeTrunkPrefill` external-encoder flow: a grow reallocates buffers, so
    /// it must happen BEFORE the command encoder is opened (W3). Idempotent once
    /// the capacity is adequate.
    public func ensurePrefillCapacity(seqLen: Int, startPos: Int32 = 0) throws {
        try ensurePrefillCapacity(
            tokenIds: [Int32](repeating: 0, count: seqLen), startPos: startPos)
    }

    /// Grow context capacity while preserving K/V cache contents.
    ///
    /// Counterpart to `prepareForRequest` for grow-only resizes. Unlike
    /// `prepareForRequest` (which uses `preserveContext: false` because
    /// it's intended for fresh requests), this preserves the current
    /// K/V cache prefix — safe to call between `prefillAllLogits` and
    /// the decode loop to lift buffers from the prefill batch size to
    /// the planned context capacity in one shot, instead of paying a
    /// per-decode-step grow.
    public func ensureContextCapacity(_ requiredContextCapacity: Int) throws {
        let clamped = clampedContextCapacity(requiredContextCapacity)
        guard clamped > currentContextCapacity else { return }
        try resizeRequestScopedBuffers(
            batchCapacity: currentBatchCapacity,
            contextCapacity: clamped,
            preserveContext: true
        )
    }

    private func resizeRequestScopedBuffers(
        batchCapacity: Int,
        contextCapacity: Int,
        preserveContext: Bool
    ) throws {
        guard dynamicRequestBuffersEnabled else { return }

        // Allocate all new buffers first; only commit if every allocation
        // succeeds. A throw mid-loop must leave the runtime's existing slot
        // assignments and capacity counters untouched.
        var pending: [(slotIndex: Int, newBuffer: MTLBuffer)] = []
        pending.reserveCapacity(manifest.buffers.slots.count)
        for slot in manifest.buffers.slots {
            guard slot.index != Int(config.weightsSlot) else { continue }
            let newSize = Self.requestedSize(
                for: slot,
                config: manifest.config,
                compiledMaxBatch: max(Int(config.maxPrefillBatchSize), 1),
                batchCapacity: batchCapacity,
                contextCapacity: contextCapacity,
                dynamicEnabled: dynamicRequestBuffersEnabled
            )
            guard buffers[slot.index].length != newSize else { continue }
            let newBuffer = try makeResizedBuffer(
                slot: slot,
                old: buffers[slot.index],
                newSize: newSize,
                preservePrefix: preserveContext && Self.shouldPreserveContents(slot)
            )
            pending.append((slot.index, newBuffer))
        }

        let contextChanged = contextCapacity != currentContextCapacity
        let buffersChanged = !pending.isEmpty
        if Self.resizeTraceEnabled, buffersChanged || contextChanged {
            let changedSlotNames = pending.compactMap {
                slotsByIndex[$0.slotIndex]?.name
            }
            fputs(
                "[smelt-resize] \(manifest.modelName) "
                + "ctx \(currentContextCapacity)→\(contextCapacity) "
                + "batch \(currentBatchCapacity)→\(batchCapacity) "
                + "replaced=\(pending.count) "
                + "slots=\(changedSlotNames.prefix(6).joined(separator: ","))"
                + (changedSlotNames.count > 6 ? "…" : "")
                + "\n",
                stderr
            )
        }
        for entry in pending {
            buffers[entry.slotIndex] = entry.newBuffer
        }
        currentBatchCapacity = batchCapacity
        currentContextCapacity = contextCapacity
        // populateStaticTables rebuilds RoPE tables (millions of trig ops on
        // a 4B-context model) — skip when context capacity is unchanged.
        if contextChanged {
            Self.populateStaticTables(
                manifest: manifest,
                config: config,
                contextCapacity: currentContextCapacity,
                buffers: &buffers
            )
        }
        if buffersChanged || contextChanged {
            invalidateVerifyICB()
        }
    }

    private func makeResizedBuffer(
        slot: SmeltBufferSlot,
        old: MTLBuffer,
        newSize: Int,
        preservePrefix: Bool
    ) throws -> MTLBuffer {
        guard let buf = device.makeBuffer(
            length: newSize,
            options: .storageModeShared
        ) else {
            throw SmeltRuntimeError.metalAllocationFailed(newSize)
        }
        memset(buf.contents(), 0, newSize)
        if preservePrefix {
            preserveResizedPrefix(
                slot: slot,
                old: old,
                new: buf
            )
        }
        return buf
    }

    private func preserveResizedPrefix(
        slot: SmeltBufferSlot,
        old: MTLBuffer,
        new: MTLBuffer
    ) {
        if Self.isContextScopedSlot(slot), slot.shape.count >= 3 {
            let heads = slot.shape[0]
            let headDim = slot.shape[2]
            let bytesPerValue = Self.bytesPerElement(slot.dtype)
            let elemsPerSeq = max(heads * headDim * bytesPerValue, 1)
            let oldSeq = max(1, old.length / elemsPerSeq)
            let newSeq = max(1, new.length / elemsPerSeq)
            let prefixStride = min(oldSeq, newSeq) * headDim * bytesPerValue
            let oldStride = oldSeq * headDim * bytesPerValue
            let newStride = newSeq * headDim * bytesPerValue

            if prefixStride > 0 {
                let oldBase = old.contents()
                let newBase = new.contents()
                for head in 0..<heads {
                    memcpy(
                        newBase.advanced(by: head * newStride),
                        oldBase.advanced(by: head * oldStride),
                        prefixStride
                    )
                }
            }
            return
        }

        let prefix = min(old.length, new.length)
        if prefix > 0 {
            memcpy(new.contents(), old.contents(), prefix)
        }
    }

    private func clampedBatchCapacity(_ requested: Int) -> Int {
        let floor = 1
        let compiledMax = max(Int(config.maxPrefillBatchSize), floor)
        if config.maxPrefillBatchSize <= 0 {
            return floor
        }
        return min(max(requested, floor), compiledMax)
    }

    private func clampedContextCapacity(_ requested: Int) -> Int {
        min(max(requested, 1), Int(config.contextLimit))
    }

    private static func requestedSize(
        for slot: SmeltBufferSlot,
        config: SmeltManifestConfig,
        compiledMaxBatch: Int,
        batchCapacity: Int,
        contextCapacity: Int,
        dynamicEnabled: Bool
    ) -> Int {
        guard dynamicEnabled else { return slot.sizeBytes }
        if let size = batchScopedSize(
            for: slot,
            config: config,
            compiledMaxBatch: compiledMaxBatch,
            batchCapacity: batchCapacity
        ) {
            return max(size, 16)
        }
        if let size = contextScopedSize(
            for: slot,
            contextCapacity: contextCapacity
        ) {
            return max(size, 16)
        }
        return slot.sizeBytes
    }

    private static func batchScopedSize(
        for slot: SmeltBufferSlot,
        config: SmeltManifestConfig,
        compiledMaxBatch: Int,
        batchCapacity: Int
    ) -> Int? {
        if slot.name == "tokenIdsBatch" {
            return batchCapacity * MemoryLayout<Int32>.stride
        }

        if isRecurrentVerifyHistorySlot(slot),
           let compiledRows = slot.shape.first,
           compiledRows > 1 {
            // History row 0 is the state on verify entry; every input token
            // adds one more row. The compiler caps compiledRows independently
            // of the package's potentially much larger prompt-prefill batch.
            let bytesPerRow = slot.sizeBytes / compiledRows
            let requestedRows = min(batchCapacity + 1, compiledRows)
            return bytesPerRow * requestedRows
        }

        if isBatchScopedSlot(slot), compiledMaxBatch > 0 {
            let bytesPerBatch = slot.sizeBytes / compiledMaxBatch
            if bytesPerBatch > 0 {
                return bytesPerBatch * batchCapacity
            }
        }

        let elems: Int
        switch slot.name {
        case "hiddenA", "hiddenB", "normOutBuf", "ffnDownBuf", "residualBuf",
             "zBuf", "recOutBuf", "gatedOutBuf":
            guard config.hiddenSize > 0 else { return nil }
            elems = config.hiddenSize * batchCapacity
        case "ffnGateBuf", "ffnUpBuf", "ffnIntBuf":
            guard config.ffnDim > 0 else { return nil }
            elems = config.ffnDim * batchCapacity
        case "qkvBuf":
            guard config.deltaQKVDim > 0 else { return nil }
            elems = config.deltaQKVDim * batchCapacity
        case "aBuf", "bBuf", "betaBuf", "gBuf":
            guard config.deltaNumHeads > 0 else { return nil }
            elems = config.deltaNumHeads * batchCapacity
        case "attnQBuf":
            guard config.attnQProjDim > 0 else { return nil }
            elems = config.attnQProjDim * batchCapacity
        case "attnKBuf":
            guard config.attnKProjDim > 0 else { return nil }
            elems = config.attnKProjDim * batchCapacity
        case "attnVBuf":
            guard config.attnVProjDim > 0 else { return nil }
            elems = config.attnVProjDim * batchCapacity
        case "attnOutBuf", "attnGateBuf":
            guard config.attnOutDim > 0 else { return nil }
            elems = config.attnOutDim * batchCapacity
        default:
            return nil
        }

        return elems * bytesPerElement(slot.dtype)
    }

    private static func contextScopedSize(
        for slot: SmeltBufferSlot,
        contextCapacity: Int
    ) -> Int? {
        if isKVCacheSlot(slot), slot.shape.count >= 3 {
            let heads = slot.shape[0]
            let headDim = slot.shape[2]
            return heads * contextCapacity * headDim * bytesPerElement(slot.dtype)
        }
        // Dense-trunk ABI: row-contiguous [rows, kvDim] KV — rows are
        // the context-scoped dimension.
        if isKVCacheSlot(slot), slot.shape.count == 2 {
            return contextCapacity * slot.shape[1] * bytesPerElement(slot.dtype)
        }
        if isRoPETableSlot(slot), slot.shape.count >= 2 {
            let dim = slot.shape[1]
            return contextCapacity * dim * bytesPerElement(slot.dtype)
        }
        if isAttentionMaskSlot(slot) {
            return contextCapacity * bytesPerElement(slot.dtype)
        }
        return nil
    }

    private static func bytesPerElement(_ dtype: SmeltDType) -> Int {
        dtype.bytesPerElement ?? 1
    }

    private static func isBatchScopedSlot(_ slot: SmeltBufferSlot) -> Bool {
        switch slot.name {
        case "hiddenA", "hiddenB", "normOutBuf", "ffnDownBuf", "residualBuf",
             "zBuf", "recOutBuf", "gatedOutBuf",
             "ffnGateBuf", "ffnUpBuf", "ffnIntBuf",
             "qkvBuf", "aBuf", "bBuf", "betaBuf", "gBuf",
             "attnQBuf", "attnKBuf", "attnVBuf", "attnOutBuf", "attnGateBuf",
             "tokenIdsBatch":
            return true
        default:
            return false
        }
    }

    private static func isRecurrentVerifyHistorySlot(
        _ slot: SmeltBufferSlot
    ) -> Bool {
        slot.name.hasPrefix("convStateHistory_")
            || slot.name.hasPrefix("recStateHistory_")
    }

    private static func isContextScopedSlot(_ slot: SmeltBufferSlot) -> Bool {
        isKVCacheSlot(slot) || isRoPETableSlot(slot) || isAttentionMaskSlot(slot)
    }

    private static func isKVCacheSlot(_ slot: SmeltBufferSlot) -> Bool {
        slot.name.hasPrefix("keyCache_") || slot.name.hasPrefix("valCache_")
    }

    private static func isRoPETableSlot(_ slot: SmeltBufferSlot) -> Bool {
        slot.name == "ropeCos" || slot.name == "ropeSin"
            || slot.name.hasPrefix("ropeCos_") || slot.name.hasPrefix("ropeSin_")
    }

    private static func isAttentionMaskSlot(_ slot: SmeltBufferSlot) -> Bool {
        slot.name == "attnMaskBuf"
    }

    private static func shouldPreserveContents(_ slot: SmeltBufferSlot) -> Bool {
        isKVCacheSlot(slot)
    }

    private func copyWholeSlot(slotIndex: Int) -> Data? {
        guard slotIndex >= 0, slotIndex < buffers.count else { return nil }
        let buf = buffers[slotIndex]
        guard buf.length > 0 else { return nil }
        return Data(bytes: buf.contents(), count: buf.length)
    }

    private func copyWholeSlot(data: Data, slotIndex: Int) {
        guard slotIndex >= 0, slotIndex < buffers.count else { return }
        let buf = buffers[slotIndex]
        let count = min(data.count, buf.length)
        data.withUnsafeBytes { src in
            guard let base = src.baseAddress, count > 0 else { return }
            memcpy(buf.contents(), base, count)
        }
    }

    private func copyCachePrefix(slotIndex: Int, length: Int) -> Data? {
        guard slotIndex >= 0, slotIndex < buffers.count,
              let slot = slotsByIndex[slotIndex],
              slot.shape.count >= 3
        else {
            return nil
        }

        let heads = slot.shape[0]
        let headDim = slot.shape[2]
        let bytesPerValue = Self.bytesPerElement(slot.dtype)
        let currentSeq = max(
            1,
            buffers[slotIndex].length
                / max(heads * headDim * bytesPerValue, 1)
        )
        let clampedLength = min(length, currentSeq)
        let fullStride = currentSeq * headDim * bytesPerValue
        let prefixStride = clampedLength * headDim * bytesPerValue
        var data = Data(count: heads * prefixStride)
        let srcBase = buffers[slotIndex].contents()

        data.withUnsafeMutableBytes { dst in
            guard let dstBase = dst.baseAddress else { return }
            for head in 0..<heads {
                memcpy(
                    dstBase.advanced(by: head * prefixStride),
                    srcBase.advanced(by: head * fullStride),
                    prefixStride
                )
            }
        }
        return data
    }

    /// `sourceStride` is the per-head token count in `data` (i.e., the
    /// `capturedLength` the snapshot was taken at). For full restore
    /// `sourceStride == length`; for partial restore
    /// `length < sourceStride`. Required (non-optional) — letting it
    /// default silently would produce corrupted partial restores
    /// because head 1+ would read from head 0's tail.
    private func restoreCachePrefix(
        _ data: Data,
        slotIndex: Int,
        length: Int,
        sourceStride: Int
    ) {
        guard slotIndex >= 0, slotIndex < buffers.count,
              let slot = slotsByIndex[slotIndex],
              slot.shape.count >= 3
        else {
            return
        }

        let heads = slot.shape[0]
        let headDim = slot.shape[2]
        let bytesPerValue = Self.bytesPerElement(slot.dtype)
        let currentSeq = max(
            1,
            buffers[slotIndex].length
                / max(heads * headDim * bytesPerValue, 1)
        )
        let clampedLength = min(length, currentSeq, sourceStride)
        let destHeadStride = currentSeq * headDim * bytesPerValue
        let sourceHeadStride = sourceStride * headDim * bytesPerValue
        let copyBytes = clampedLength * headDim * bytesPerValue
        let dstBase = buffers[slotIndex].contents()

        data.withUnsafeBytes { src in
            guard let srcBase = src.baseAddress else { return }
            for head in 0..<heads {
                memcpy(
                    dstBase.advanced(by: head * destHeadStride),
                    srcBase.advanced(by: head * sourceHeadStride),
                    copyBytes
                )
            }
        }
    }
}

// MARK: - Errors

/// Errors from the Smelt runtime.
public enum SmeltRuntimeError: Error, CustomStringConvertible, CodedError {
    case noMetalDevice
    case noCommandQueue
    case invalidPackage(String)
    case checksumMismatch(String)
    case deviceIncompatible(String)
    case metalCommandBufferUnavailable
    case metalDispatchFailed(Error)
    case metalAllocationFailed(Int)
    case dispatchTableMissing(String)
    case inputExceedsContext(limit: Int, requested: Int)

    public var description: String {
        switch self {
        case .noMetalDevice:
            return "No Metal device available"
        case .noCommandQueue:
            return "Failed to create Metal command queue"
        case let .invalidPackage(detail):
            return "Invalid .smeltpkg: \(detail)"
        case let .checksumMismatch(detail):
            return "Checksum mismatch: \(detail)"
        case let .deviceIncompatible(detail):
            return "Device incompatible: \(detail)"
        case .metalCommandBufferUnavailable:
            return "Failed to create Metal command buffer or encoder"
        case let .metalDispatchFailed(err):
            return "Metal dispatch failed: \(err)"
        case let .metalAllocationFailed(bytes):
            return "Failed to allocate Metal buffer of \(bytes) bytes"
        case let .dispatchTableMissing(detail):
            return "Dispatch table missing in package: \(detail)"
        case let .inputExceedsContext(limit, requested):
            return "Input is too long for context_limit=\(limit): \(requested) tokens"
        }
    }

    public var code: String {
        switch self {
        case .noMetalDevice: return "no_metal_device"
        case .noCommandQueue: return "no_command_queue"
        case .invalidPackage: return "invalid_package"
        case .checksumMismatch: return "checksum_mismatch"
        case .deviceIncompatible: return "device_incompatible"
        case .metalCommandBufferUnavailable: return "metal_command_buffer_unavailable"
        case .metalDispatchFailed: return "metal_dispatch_failed"
        case .metalAllocationFailed: return "metal_allocation_failed"
        case .dispatchTableMissing: return "dispatch_table_missing"
        case .inputExceedsContext: return "input_exceeds_context"
        }
    }
}

// Decode is driven by dispatches.bin — no compiled Swift needed.
// The binary dispatch table is mmap'd at init and interpreted per token.
