import Darwin
import Foundation
import Metal

public struct SmeltFrozenOperationTimingSpan: Sendable, Equatable {
    public let recordIndices: [Int]
    public let gpuUs: Double

    public init(recordIndices: [Int], gpuUs: Double) {
        self.recordIndices = recordIndices
        self.gpuUs = gpuUs
    }
}

/// Per-operation GPU timing produced by a frozen component execution. Spans
/// preserve the device's real scheduling: multiple opaque operations may share
/// a span when no natural stage boundary exists between them.
public struct SmeltFrozenOperationTimingProfile: Sendable, Equatable {
    public let measurementMethod: String
    public let spans: [SmeltFrozenOperationTimingSpan]
    public let wholePlanGPUUs: Double

    public init(
        measurementMethod: String,
        spans: [SmeltFrozenOperationTimingSpan],
        wholePlanGPUUs: Double
    ) {
        self.measurementMethod = measurementMethod
        self.spans = spans
        self.wholePlanGPUUs = wholePlanGPUUs
    }
}

enum SmeltMetalFrozenOperationProfilerError: Error, CustomStringConvertible {
    case stageBoundarySamplingUnavailable(String)
    case timestampCounterUnavailable(String)
    case markerCompilationFailed(String)
    case counterBufferCreationFailed(String)
    case commandEncoderUnavailable
    case invalidEncoderState(String)
    case recordOverflow(expected: Int)
    case recordCountMismatch(expected: Int, got: Int)
    case sampleOverflow(capacity: Int)
    case unresolvedCounters
    case invalidCounterPayload(expectedBytes: Int, gotBytes: Int)
    case invalidTimestamp(records: [Int], start: UInt64, end: UInt64)
    case invalidTimestampScale

    var description: String {
        switch self {
        case .stageBoundarySamplingUnavailable(let device):
            return "frozen operation profiler: \(device) does not support stage-boundary counters"
        case .timestampCounterUnavailable(let device):
            return "frozen operation profiler: \(device) has no timestamp counter set"
        case .markerCompilationFailed(let detail):
            return "frozen operation profiler: dependency marker compile failed: \(detail)"
        case .counterBufferCreationFailed(let detail):
            return "frozen operation profiler: counter buffer creation failed: \(detail)"
        case .commandEncoderUnavailable:
            return "frozen operation profiler: could not create a compute encoder"
        case .invalidEncoderState(let detail):
            return "frozen operation profiler: invalid encoder state: \(detail)"
        case .recordOverflow(let expected):
            return "frozen operation profiler: execution exceeded \(expected) frozen records"
        case .recordCountMismatch(let expected, let got):
            return "frozen operation profiler: executed \(got) records; frozen plan has \(expected)"
        case .sampleOverflow(let capacity):
            return "frozen operation profiler: execution exceeded \(capacity) counter samples"
        case .unresolvedCounters:
            return "frozen operation profiler: timestamp counter payload was unavailable"
        case .invalidCounterPayload(let expected, let got):
            return "frozen operation profiler: counter payload has \(got) bytes; expected at least \(expected)"
        case .invalidTimestamp(let records, let start, let end):
            return "frozen operation profiler: records \(records) have invalid timestamps \(start)...\(end)"
        case .invalidTimestampScale:
            return "frozen operation profiler: GPU timestamp scale is invalid"
        }
    }
}

/// In-place profiler for a component's already-selected execution route.
/// Existing compute encoders receive start/end samples without being split.
/// Opaque MPS calls are measured in the gaps between those real stages. A
/// bit-preserving dependency marker is used only where the graph has no natural
/// boundary (command start and attention-to-projection transitions).
final class SmeltMetalFrozenOperationProfiler {
    private struct PendingSpan {
        var records: [Int]
        let startSample: Int
        let endSample: Int
    }

    private static let markerSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void agent_frozen_profile_dependency_marker(
            device atomic_uint *value [[buffer(0)]])
        {
            atomic_fetch_or_explicit(value, 0u, memory_order_relaxed);
        }
        """

    private let device: MTLDevice
    private let recordCapacity: Int
    private let sampleCapacity: Int
    private let sampleBuffer: MTLCounterSampleBuffer
    private let markerPipeline: MTLComputePipelineState
    private let clockStart: (cpu: UInt64, gpu: UInt64)
    private var nextRecord = 0
    private var nextSample = 0
    private var spans: [PendingSpan] = []
    private var activeComputeSpan: Int?
    private var lastBoundarySample: Int?
    private var pendingOpaqueRecords: [Int] = []
    private var opaqueStartSample: Int?

    init(device: MTLDevice, recordCapacity: Int) throws {
        guard recordCapacity > 0 else {
            throw SmeltMetalFrozenOperationProfilerError.recordCountMismatch(
                expected: recordCapacity,
                got: 0
            )
        }
        guard device.supportsCounterSampling(.atStageBoundary) else {
            throw SmeltMetalFrozenOperationProfilerError
                .stageBoundarySamplingUnavailable(device.name)
        }
        guard let counterSet = (device.counterSets ?? []).first(where: {
            $0.name == MTLCommonCounterSet.timestamp.rawValue
        }) else {
            throw SmeltMetalFrozenOperationProfilerError
                .timestampCounterUnavailable(device.name)
        }
        self.sampleCapacity = recordCapacity * 2 + 256
        let descriptor = MTLCounterSampleBufferDescriptor()
        descriptor.counterSet = counterSet
        descriptor.label = "smelt.frozen-operation-profile"
        descriptor.storageMode = .shared
        descriptor.sampleCount = sampleCapacity
        do {
            sampleBuffer = try device.makeCounterSampleBuffer(descriptor: descriptor)
        } catch {
            throw SmeltMetalFrozenOperationProfilerError
                .counterBufferCreationFailed(String(describing: error))
        }
        do {
            let library = try device.makeLibrary(source: Self.markerSource, options: nil)
            guard let function = library.makeFunction(
                name: "agent_frozen_profile_dependency_marker"
            ) else {
                throw SmeltMetalFrozenOperationProfilerError
                    .markerCompilationFailed("function is absent")
            }
            markerPipeline = try device.makeComputePipelineState(function: function)
        } catch let error as SmeltMetalFrozenOperationProfilerError {
            throw error
        } catch {
            throw SmeltMetalFrozenOperationProfilerError
                .markerCompilationFailed(String(describing: error))
        }
        self.device = device
        self.recordCapacity = recordCapacity
        let clocks = device.sampleTimestamps()
        clockStart = (UInt64(clocks.cpu), UInt64(clocks.gpu))
    }

    func makeComputeEncoder(
        commandBuffer: MTLCommandBuffer,
        label: String
    ) throws -> MTLComputeCommandEncoder {
        guard activeComputeSpan == nil else {
            throw SmeltMetalFrozenOperationProfilerError.invalidEncoderState(
                "a compute segment is already active"
            )
        }
        let (start, end) = try reserveSamplePair()
        try closeOpaqueSpan(at: start)
        let descriptor = MTLComputePassDescriptor()
        let attachment = descriptor.sampleBufferAttachments[0]!
        attachment.sampleBuffer = sampleBuffer
        attachment.startOfEncoderSampleIndex = start
        attachment.endOfEncoderSampleIndex = end
        guard let encoder = commandBuffer.makeComputeCommandEncoder(
            descriptor: descriptor
        ) else {
            throw SmeltMetalFrozenOperationProfilerError.commandEncoderUnavailable
        }
        spans.append(PendingSpan(records: [], startSample: start, endSample: end))
        activeComputeSpan = spans.count - 1
        encoder.label = label
        return encoder
    }

    func recordMetalOperation(label: String) throws {
        guard let activeComputeSpan else {
            throw SmeltMetalFrozenOperationProfilerError.invalidEncoderState(
                "Metal operation '\(label)' has no active compute segment"
            )
        }
        spans[activeComputeSpan].records.append(try reserveRecord())
    }

    func endComputeEncoder(_ encoder: MTLComputeCommandEncoder) {
        encoder.endEncoding()
        if let activeComputeSpan {
            lastBoundarySample = spans[activeComputeSpan].endSample
        }
        activeComputeSpan = nil
    }

    func encodeOpaqueOperation(
        label: String,
        _ body: () throws -> Void
    ) throws {
        guard activeComputeSpan == nil else {
            throw SmeltMetalFrozenOperationProfilerError.invalidEncoderState(
                "opaque operation '\(label)' began inside a compute segment"
            )
        }
        if pendingOpaqueRecords.isEmpty {
            guard let lastBoundarySample else {
                throw SmeltMetalFrozenOperationProfilerError.invalidEncoderState(
                    "opaque operation '\(label)' has no preceding timestamp boundary"
                )
            }
            opaqueStartSample = lastBoundarySample
        }
        pendingOpaqueRecords.append(try reserveRecord())
        try body()
    }

    func encodeBoundaryMarker(
        commandBuffer: MTLCommandBuffer,
        buffer: MTLBuffer,
        label: String
    ) throws {
        guard activeComputeSpan == nil else {
            throw SmeltMetalFrozenOperationProfilerError.invalidEncoderState(
                "boundary marker '\(label)' began inside a compute segment"
            )
        }
        let (start, end) = try reserveSamplePair()
        try closeOpaqueSpan(at: start)
        let descriptor = MTLComputePassDescriptor()
        let attachment = descriptor.sampleBufferAttachments[0]!
        attachment.sampleBuffer = sampleBuffer
        attachment.startOfEncoderSampleIndex = start
        attachment.endOfEncoderSampleIndex = end
        guard let encoder = commandBuffer.makeComputeCommandEncoder(
            descriptor: descriptor
        ) else {
            throw SmeltMetalFrozenOperationProfilerError.commandEncoderUnavailable
        }
        encoder.label = label
        encoder.setComputePipelineState(markerPipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        encoder.dispatchThreads(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
        )
        encoder.endEncoding()
        lastBoundarySample = end
    }

    func finish(commandBuffer: MTLCommandBuffer) throws -> SmeltFrozenOperationTimingProfile {
        guard activeComputeSpan == nil else {
            throw SmeltMetalFrozenOperationProfilerError.invalidEncoderState(
                "a compute segment remained active at completion"
            )
        }
        guard pendingOpaqueRecords.isEmpty else {
            throw SmeltMetalFrozenOperationProfilerError.invalidEncoderState(
                "opaque records \(pendingOpaqueRecords) have no ending boundary"
            )
        }
        guard nextRecord == recordCapacity else {
            throw SmeltMetalFrozenOperationProfilerError.recordCountMismatch(
                expected: recordCapacity,
                got: nextRecord
            )
        }
        let expectedBytes = nextSample * MemoryLayout<UInt64>.stride
        guard let data = try sampleBuffer.resolveCounterRange(0..<nextSample) else {
            throw SmeltMetalFrozenOperationProfilerError.unresolvedCounters
        }
        guard data.count >= expectedBytes else {
            throw SmeltMetalFrozenOperationProfilerError.invalidCounterPayload(
                expectedBytes: expectedBytes,
                gotBytes: data.count
            )
        }
        let clockEndRaw = device.sampleTimestamps()
        let nanosecondsPerGPUTick = try timestampNanosecondsPerTick(
            start: clockStart,
            end: (UInt64(clockEndRaw.cpu), UInt64(clockEndRaw.gpu))
        )
        let values = data.withUnsafeBytes { raw -> [UInt64] in
            Array(raw.bindMemory(to: UInt64.self).prefix(nextSample))
        }
        var measured: [SmeltFrozenOperationTimingSpan] = []
        measured.reserveCapacity(spans.count)
        for span in spans where !span.records.isEmpty {
            let start = values[span.startSample]
            let end = values[span.endSample]
            guard start != UInt64.max, end != UInt64.max, end >= start else {
                throw SmeltMetalFrozenOperationProfilerError.invalidTimestamp(
                    records: span.records,
                    start: start,
                    end: end
                )
            }
            measured.append(SmeltFrozenOperationTimingSpan(
                recordIndices: span.records,
                gpuUs: Double(end - start) * nanosecondsPerGPUTick / 1_000
            ))
        }
        return SmeltFrozenOperationTimingProfile(
            measurementMethod:
                "in-place-metal-stage-boundaries+opaque-mps-gaps",
            spans: measured,
            wholePlanGPUUs: max(
                commandBuffer.gpuEndTime - commandBuffer.gpuStartTime,
                0
            ) * 1_000_000
        )
    }

    private func closeOpaqueSpan(at endSample: Int) throws {
        guard !pendingOpaqueRecords.isEmpty else { return }
        guard let opaqueStartSample else {
            throw SmeltMetalFrozenOperationProfilerError.invalidEncoderState(
                "opaque records have no starting boundary"
            )
        }
        spans.append(PendingSpan(
            records: pendingOpaqueRecords,
            startSample: opaqueStartSample,
            endSample: endSample
        ))
        pendingOpaqueRecords = []
        self.opaqueStartSample = nil
    }

    private func reserveRecord() throws -> Int {
        guard nextRecord < recordCapacity else {
            throw SmeltMetalFrozenOperationProfilerError.recordOverflow(
                expected: recordCapacity
            )
        }
        defer { nextRecord += 1 }
        return nextRecord
    }

    private func reserveSamplePair() throws -> (Int, Int) {
        guard nextSample + 1 < sampleCapacity else {
            throw SmeltMetalFrozenOperationProfilerError.sampleOverflow(
                capacity: sampleCapacity
            )
        }
        let result = (nextSample, nextSample + 1)
        nextSample += 2
        return result
    }

    private func timestampNanosecondsPerTick(
        start: (cpu: UInt64, gpu: UInt64),
        end: (cpu: UInt64, gpu: UInt64)
    ) throws -> Double {
        guard end.cpu > start.cpu, end.gpu > start.gpu else {
            throw SmeltMetalFrozenOperationProfilerError.invalidTimestampScale
        }
        var timebase = mach_timebase_info_data_t()
        mach_timebase_info(&timebase)
        let cpuNanoseconds = Double(end.cpu - start.cpu)
            * Double(timebase.numer) / Double(timebase.denom)
        let scale = cpuNanoseconds / Double(end.gpu - start.gpu)
        guard scale.isFinite, scale > 0 else {
            throw SmeltMetalFrozenOperationProfilerError.invalidTimestampScale
        }
        return scale
    }
}
