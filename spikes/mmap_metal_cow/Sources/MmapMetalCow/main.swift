import Foundation
import Metal
import Darwin

// MARK: - Output

enum Outcome { case pass, fail, info }

func emit(_ kind: Outcome, _ msg: String) {
    let tag: String
    switch kind {
    case .pass: tag = "[PASS]"
    case .fail: tag = "[FAIL]"
    case .info: tag = "[INFO]"
    }
    print("\(tag) \(msg)")
}

func section(_ name: String) {
    print("")
    print("========== \(name) ==========")
}

// MARK: - Metal kernel

let kernelSource = """
#include <metal_stdlib>
using namespace metal;

kernel void writeMagic(device uchar *buf [[buffer(0)]],
                       constant uint &offset [[buffer(1)]],
                       constant uchar &value [[buffer(2)]],
                       uint tid [[thread_position_in_grid]]) {
    if (tid == 0) {
        buf[offset] = value;
    }
}

kernel void writePattern(device uchar *buf [[buffer(0)]],
                         constant uint &count [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {
    if (tid < count) {
        buf[tid] = uchar(0xA0 | (tid & 0x0F));
    }
}

// Touch one byte per page across the buffer. One thread per page.
kernel void touchPages(device uchar *buf [[buffer(0)]],
                       constant uint &numPages [[buffer(1)]],
                       constant uint &pageSize [[buffer(2)]],
                       uint tid [[thread_position_in_grid]]) {
    if (tid < numPages) {
        uint offset = tid * pageSize;
        buf[offset] = uchar(tid & 0xFF);
    }
}

// Sparse write: touch every Kth page. Used to force COW on a small subset.
kernel void sparseWrite(device uchar *buf [[buffer(0)]],
                        constant uint &numPages [[buffer(1)]],
                        constant uint &pageSize [[buffer(2)]],
                        constant uint &stride [[buffer(3)]],
                        uint tid [[thread_position_in_grid]]) {
    uint pageIdx = tid * stride;
    if (pageIdx < numPages) {
        buf[pageIdx * pageSize] = uchar(tid & 0xFF);
    }
}
"""

// MARK: - Helpers

func mustOpenRW(_ path: String, sizeBytes: Int, fillByte: UInt8) -> Int32 {
    let fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0o644)
    precondition(fd >= 0, "open failed: \(String(cString: strerror(errno)))")
    let r = ftruncate(fd, off_t(sizeBytes))
    precondition(r == 0, "ftruncate failed")
    var fill = [UInt8](repeating: fillByte, count: sizeBytes)
    let n = pwrite(fd, &fill, sizeBytes, 0)
    precondition(n == sizeBytes, "pwrite filled \(n) of \(sizeBytes)")
    let s = fsync(fd)
    precondition(s == 0, "fsync failed")
    return fd
}

func mustMmap(_ fd: Int32, length: Int, flags: Int32) -> UnsafeMutableRawPointer {
    let prot = PROT_READ | PROT_WRITE
    let p = mmap(nil, length, prot, flags, fd, 0)!
    precondition(Int(bitPattern: p) != -1, "mmap failed: \(String(cString: strerror(errno)))")
    return p
}

func readByteFromFreshFD(_ path: String, offset: Int) -> UInt8? {
    let fd = open(path, O_RDONLY)
    guard fd >= 0 else { return nil }
    defer { close(fd) }
    var byte: UInt8 = 0
    let n = pread(fd, &byte, 1, off_t(offset))
    return n == 1 ? byte : nil
}

func gpuWriteMagic(device: MTLDevice, queue: MTLCommandQueue, pipeline: MTLComputePipelineState,
                   buffer: MTLBuffer, offset: UInt32, value: UInt8) {
    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(buffer, offset: 0, index: 0)
    var off = offset
    var val = value
    enc.setBytes(&off, length: 4, index: 1)
    enc.setBytes(&val, length: 1, index: 2)
    enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
}

func gpuWritePattern(device: MTLDevice, queue: MTLCommandQueue, pipeline: MTLComputePipelineState,
                     buffer: MTLBuffer, count: UInt32) {
    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(buffer, offset: 0, index: 0)
    var c = count
    enc.setBytes(&c, length: 4, index: 1)
    let tg = min(Int(count), pipeline.maxTotalThreadsPerThreadgroup)
    enc.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
}

// MARK: - Test setup

let pageSize = Int(getpagesize())
let bufferSize = pageSize * 4   // four pages

print("page size:    \(pageSize) bytes")
print("buffer size:  \(bufferSize) bytes (\(bufferSize / pageSize) pages)")

guard let device = MTLCreateSystemDefaultDevice() else {
    emit(.fail, "no Metal device")
    exit(1)
}
print("device: \(device.name)")

let queue = device.makeCommandQueue()!
let library: MTLLibrary
do {
    library = try device.makeLibrary(source: kernelSource, options: nil)
} catch {
    emit(.fail, "compile kernel: \(error)")
    exit(1)
}
let writeMagic = library.makeFunction(name: "writeMagic")!
let writePattern = library.makeFunction(name: "writePattern")!
let writeMagicPipe = try device.makeComputePipelineState(function: writeMagic)
let writePatternPipe = try device.makeComputePipelineState(function: writePattern)
let touchPagesPipe = try device.makeComputePipelineState(function: library.makeFunction(name: "touchPages")!)
let sparseWritePipe = try device.makeComputePipelineState(function: library.makeFunction(name: "sparseWrite")!)

func gpuTouchPages(buffer: MTLBuffer, numPages: UInt32, pageSize: UInt32) {
    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(touchPagesPipe)
    enc.setBuffer(buffer, offset: 0, index: 0)
    var n = numPages
    var p = pageSize
    enc.setBytes(&n, length: 4, index: 1)
    enc.setBytes(&p, length: 4, index: 2)
    let tg = min(Int(numPages), touchPagesPipe.maxTotalThreadsPerThreadgroup)
    enc.dispatchThreads(MTLSize(width: Int(numPages), height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
}

func gpuSparseWrite(buffer: MTLBuffer, numPages: UInt32, pageSize: UInt32, stride: UInt32) {
    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(sparseWritePipe)
    enc.setBuffer(buffer, offset: 0, index: 0)
    var n = numPages
    var p = pageSize
    var s = stride
    enc.setBytes(&n, length: 4, index: 1)
    enc.setBytes(&p, length: 4, index: 2)
    enc.setBytes(&s, length: 4, index: 3)
    let touchCount = (numPages + stride - 1) / stride
    let tg = min(Int(touchCount), sparseWritePipe.maxTotalThreadsPerThreadgroup)
    enc.dispatchThreads(MTLSize(width: Int(touchCount), height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
}

func currentRSSBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? info.resident_size : 0
}

func mb(_ bytes: UInt64) -> String { String(format: "%.1f MiB", Double(bytes) / (1024.0 * 1024.0)) }

// Working files in /tmp so we don't pollute the repo.
let testDir = "/tmp/mmap_metal_cow_\(getpid())"
mkdir(testDir, 0o755)
defer {
    // Best-effort cleanup; don't fail the run on this.
    let task = Process()
    task.launchPath = "/bin/rm"
    task.arguments = ["-rf", testDir]
    try? task.run()
    task.waitUntilExit()
}

// MARK: - TEST A: COW from GPU writes (the killer question)

section("TEST A: GPU write to MAP_PRIVATE — does COW trigger?")

do {
    let path = "\(testDir)/test_a.bin"
    let fd_a = mustOpenRW(path, sizeBytes: bufferSize, fillByte: 0x42)
    let fd_b = open(path, O_RDWR)
    precondition(fd_b >= 0, "open fd_b")

    let ptrA = mustMmap(fd_a, length: bufferSize, flags: MAP_PRIVATE)
    let ptrB = mustMmap(fd_b, length: bufferSize, flags: MAP_PRIVATE)

    // Wrap A as a Metal buffer (no copy).
    let bufA = device.makeBuffer(bytesNoCopy: ptrA,
                                 length: bufferSize,
                                 options: .storageModeShared,
                                 deallocator: nil)!

    // Pre-conditions: both views start at 0x42.
    let preA = ptrA.load(fromByteOffset: 0, as: UInt8.self)
    let preB = ptrB.load(fromByteOffset: 0, as: UInt8.self)
    emit(preA == 0x42 ? .pass : .fail, "pre-write A[0] == 0x42 (got 0x\(String(preA, radix: 16)))")
    emit(preB == 0x42 ? .pass : .fail, "pre-write B[0] == 0x42 (got 0x\(String(preB, radix: 16)))")

    // GPU writes 0xFF to A[0].
    let t0 = Date()
    gpuWriteMagic(device: device, queue: queue, pipeline: writeMagicPipe,
                  buffer: bufA, offset: 0, value: 0xFF)
    let dt = Date().timeIntervalSince(t0) * 1000
    emit(.info, "Metal write completed in \(String(format: "%.3f", dt)) ms")

    // A's view should reflect the GPU write.
    let postA = ptrA.load(fromByteOffset: 0, as: UInt8.self)
    emit(postA == 0xFF ? .pass : .fail,
         "post-write A[0] == 0xFF via CPU pointer (got 0x\(String(postA, radix: 16)))")

    // B's view: this is the answer.
    let postB = ptrB.load(fromByteOffset: 0, as: UInt8.self)
    if postB == 0x42 {
        emit(.pass, "post-write B[0] == 0x42 — COW WORKED. GPU write triggered page fault, A diverged from B. **Per-fork isolation via MAP_PRIVATE is real.**")
    } else if postB == 0xFF {
        emit(.fail, "post-write B[0] == 0xFF — NO COW. GPU bypassed write-protect; A and B share the page. Fall back to read-only-prefix architecture.")
    } else {
        emit(.fail, "post-write B[0] == 0x\(String(postB, radix: 16)) — unexpected value")
    }

    // File on disk: should be 0x42 regardless (MAP_PRIVATE never writes back).
    if let fileByte = readByteFromFreshFD(path, offset: 0) {
        emit(fileByte == 0x42 ? .pass : .fail,
             "file on disk[0] == 0x42 via fresh fd (got 0x\(String(fileByte, radix: 16))) — confirms MAP_PRIVATE doesn't propagate to file")
    }

    // Pages 1..3 of B should still be 0x42 (no GPU write near them).
    let postBpage1 = ptrB.load(fromByteOffset: pageSize, as: UInt8.self)
    emit(postBpage1 == 0x42 ? .pass : .fail,
         "B[page 1, offset 0] == 0x42 (got 0x\(String(postBpage1, radix: 16))) — only the touched page diverged")

    munmap(ptrA, bufferSize)
    munmap(ptrB, bufferSize)
    close(fd_a)
    close(fd_b)
}

// MARK: - TEST B: Control — CPU write to MAP_PRIVATE triggers COW (sanity)

section("TEST B: CPU write to MAP_PRIVATE — does standard COW work? (control)")

do {
    let path = "\(testDir)/test_b.bin"
    let fd_a = mustOpenRW(path, sizeBytes: bufferSize, fillByte: 0x42)
    let fd_b = open(path, O_RDWR)

    let ptrA = mustMmap(fd_a, length: bufferSize, flags: MAP_PRIVATE)
    let ptrB = mustMmap(fd_b, length: bufferSize, flags: MAP_PRIVATE)

    // CPU write to A[0].
    ptrA.storeBytes(of: UInt8(0xFE), as: UInt8.self)

    let postA = ptrA.load(fromByteOffset: 0, as: UInt8.self)
    let postB = ptrB.load(fromByteOffset: 0, as: UInt8.self)
    emit(postA == 0xFE ? .pass : .fail, "A[0] == 0xFE after CPU write")
    emit(postB == 0x42 ? .pass : .fail,
         "B[0] == 0x42 after A's CPU write (got 0x\(String(postB, radix: 16))) — confirms baseline COW works")

    munmap(ptrA, bufferSize)
    munmap(ptrB, bufferSize)
    close(fd_a)
    close(fd_b)
}

// MARK: - TEST C: Persistence via MAP_SHARED + GPU writes

section("TEST C: GPU write to MAP_SHARED + msync — does it persist to disk?")

do {
    let path = "\(testDir)/test_c.bin"
    let fd = mustOpenRW(path, sizeBytes: bufferSize, fillByte: 0x42)

    let ptr = mustMmap(fd, length: bufferSize, flags: MAP_SHARED)
    let buf = device.makeBuffer(bytesNoCopy: ptr,
                                length: bufferSize,
                                options: .storageModeShared,
                                deallocator: nil)!

    let writeCount: UInt32 = 256
    let t0 = Date()
    gpuWritePattern(device: device, queue: queue, pipeline: writePatternPipe,
                    buffer: buf, count: writeCount)
    let dt = Date().timeIntervalSince(t0) * 1000
    emit(.info, "Metal pattern write (256 bytes) completed in \(String(format: "%.3f", dt)) ms")

    // Verify in-memory.
    let mid = ptr.load(fromByteOffset: 7, as: UInt8.self)
    let expected: UInt8 = 0xA0 | 7
    emit(mid == expected ? .pass : .fail,
         "in-memory[7] == 0x\(String(expected, radix: 16)) (got 0x\(String(mid, radix: 16)))")

    // msync to push to disk.
    let s = msync(ptr, bufferSize, MS_SYNC)
    precondition(s == 0, "msync failed")

    // Read file via fresh fd.
    if let fileByte = readByteFromFreshFD(path, offset: 7) {
        if fileByte == expected {
            emit(.pass, "file on disk[7] == 0x\(String(expected, radix: 16)) via fresh fd — **GPU writes survive to disk via MAP_SHARED + msync. Free KV persistence is real.**")
        } else {
            emit(.fail, "file on disk[7] == 0x\(String(fileByte, radix: 16)) — persistence broken")
        }
    }

    // Spot-check a byte beyond the write region — should still be 0x42.
    if let untouched = readByteFromFreshFD(path, offset: bufferSize - 1) {
        emit(untouched == 0x42 ? .pass : .fail,
             "file on disk[end] == 0x42 (got 0x\(String(untouched, radix: 16))) — only written bytes changed")
    }

    munmap(ptr, bufferSize)
    close(fd)
}

// MARK: - TEST D: Timing — mmap+wrap vs makeBuffer(length:)

section("TEST D: timing — mmap'd Metal buffer vs allocated Metal buffer")

do {
    let path = "\(testDir)/test_d.bin"
    let fd = mustOpenRW(path, sizeBytes: bufferSize, fillByte: 0x00)

    let iters = 100

    // Time mmap + wrap.
    var mmapTotal: TimeInterval = 0
    for _ in 0..<iters {
        let t0 = Date()
        let p = mustMmap(fd, length: bufferSize, flags: MAP_PRIVATE)
        let b = device.makeBuffer(bytesNoCopy: p,
                                  length: bufferSize,
                                  options: .storageModeShared,
                                  deallocator: nil)!
        _ = b
        mmapTotal += Date().timeIntervalSince(t0)
        munmap(p, bufferSize)
    }
    let mmapAvg = mmapTotal / Double(iters) * 1_000_000  // µs

    // Time plain device allocation.
    var allocTotal: TimeInterval = 0
    for _ in 0..<iters {
        let t0 = Date()
        let b = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
        _ = b
        allocTotal += Date().timeIntervalSince(t0)
    }
    let allocAvg = allocTotal / Double(iters) * 1_000_000  // µs

    emit(.info, "mmap + makeBuffer(bytesNoCopy:): \(String(format: "%.1f", mmapAvg)) µs avg over \(iters) iters")
    emit(.info, "device.makeBuffer(length:):      \(String(format: "%.1f", allocAvg)) µs avg over \(iters) iters")

    close(fd)
}

// MARK: - TEST E: Page-fault latency under cold GPU touch

section("TEST E: cold-page first-touch latency (validates 'free cold start')")

do {
    let bigSize = 64 * 1024 * 1024  // 64 MiB
    let numPages = UInt32(bigSize / pageSize)
    let path = "\(testDir)/test_e.bin"
    let fd = mustOpenRW(path, sizeBytes: bigSize, fillByte: 0x42)

    emit(.info, "buffer: \(bigSize / (1024*1024)) MiB across \(numPages) pages")

    // COLD: fresh mmap, GPU first-touches every page.
    let ptrCold = mustMmap(fd, length: bigSize, flags: MAP_PRIVATE)
    let bufCold = device.makeBuffer(bytesNoCopy: ptrCold,
                                    length: bigSize,
                                    options: .storageModeShared,
                                    deallocator: nil)!
    let t0 = Date()
    gpuTouchPages(buffer: bufCold, numPages: numPages, pageSize: UInt32(pageSize))
    let coldDt = Date().timeIntervalSince(t0)
    let perPageCold = coldDt / Double(numPages) * 1_000_000  // µs/page
    emit(.info, "COLD touch: \(String(format: "%.2f", coldDt * 1000)) ms total, \(String(format: "%.2f", perPageCold)) µs/page")

    // WARM: same buffer, second pass.
    let t1 = Date()
    gpuTouchPages(buffer: bufCold, numPages: numPages, pageSize: UInt32(pageSize))
    let warmDt = Date().timeIntervalSince(t1)
    let perPageWarm = warmDt / Double(numPages) * 1_000_000
    emit(.info, "WARM touch: \(String(format: "%.2f", warmDt * 1000)) ms total, \(String(format: "%.2f", perPageWarm)) µs/page")

    let coldOverhead = perPageCold - perPageWarm
    emit(.info, "cold overhead: \(String(format: "%.2f", coldOverhead)) µs/page (page-fault + zero-fill cost)")

    // Realistic prefix size: 4K context, ~28KB per layer per position for Qwen 2B → ~448 MiB
    // for full 4K cache. Compute "cold start cost" for a typical 1K prefix.
    let bytesPer1KPrefix = 28 * 1024 * 28_672 / 4  // ~196 MiB but let's go with measured
    _ = bytesPer1KPrefix
    let pagesPer4MiB = 4 * 1024 * 1024 / pageSize
    let coldCostFor4MiB = Double(pagesPer4MiB) * perPageCold
    emit(.info, "extrapolated: warm-up cost for a 4 MiB prefix block = \(String(format: "%.2f", coldCostFor4MiB / 1000)) ms")

    if perPageCold < 5.0 {
        emit(.pass, "cold-page touch is sub-5µs/page. **Prefix cache cold-start claim holds: a 4 MiB prefix block warms in <\(String(format: "%.0f", coldCostFor4MiB/1000)) ms.**")
    } else if perPageCold < 50.0 {
        emit(.info, "cold-page touch in 5-50µs range — usable but plan to MADV_WILLNEED ahead of decode")
    } else {
        emit(.fail, "cold-page touch >50µs — prefix cache 'free cold start' claim weakens; need explicit prefetch")
    }

    munmap(ptrCold, bigSize)
    close(fd)
}

// MARK: - TEST F: Many-fork mmap stress

section("TEST F: 100 simultaneous mmap forks of a shared KV file")

do {
    let kvSize = 16 * 1024 * 1024  // 16 MiB shared KV file
    let numPages = UInt32(kvSize / pageSize)
    let numForks = 100
    let writePagesPerFork: UInt32 = 8  // each fork writes to 8 pages = 128 KiB diverged
    let stride = numPages / writePagesPerFork

    let path = "\(testDir)/test_f.bin"
    let initFd = mustOpenRW(path, sizeBytes: kvSize, fillByte: 0x42)
    close(initFd)

    let rssBefore = currentRSSBytes()
    emit(.info, "RSS before: \(mb(rssBefore))")
    emit(.info, "creating \(numForks) MAP_PRIVATE mmaps of a \(kvSize / (1024*1024)) MiB file")

    var ptrs: [UnsafeMutableRawPointer] = []
    var bufs: [MTLBuffer] = []
    var fds: [Int32] = []

    let tOpen = Date()
    for _ in 0..<numForks {
        let fd = open(path, O_RDWR)
        precondition(fd >= 0)
        let p = mustMmap(fd, length: kvSize, flags: MAP_PRIVATE)
        let b = device.makeBuffer(bytesNoCopy: p,
                                  length: kvSize,
                                  options: .storageModeShared,
                                  deallocator: nil)!
        ptrs.append(p)
        bufs.append(b)
        fds.append(fd)
    }
    let openDt = Date().timeIntervalSince(tOpen) * 1000
    emit(.info, "open + mmap + wrap × \(numForks): \(String(format: "%.1f", openDt)) ms (\(String(format: "%.2f", openDt / Double(numForks))) ms each)")

    let rssAfterOpen = currentRSSBytes()
    emit(.info, "RSS after \(numForks) mmaps (no writes yet): \(mb(rssAfterOpen)) (delta \(mb(rssAfterOpen - rssBefore)))")

    // Each fork sparsely writes via GPU, forcing per-fork COW on writePagesPerFork pages.
    let tWrite = Date()
    for buf in bufs {
        gpuSparseWrite(buffer: buf, numPages: numPages, pageSize: UInt32(pageSize), stride: stride)
    }
    let writeDt = Date().timeIntervalSince(tWrite) * 1000
    emit(.info, "GPU sparse write × \(numForks) (\(writePagesPerFork) pages each): \(String(format: "%.1f", writeDt)) ms")

    let rssAfterWrite = currentRSSBytes()
    let extraBytes = rssAfterWrite - rssAfterOpen
    let expectedExtraBytes = UInt64(numForks) * UInt64(writePagesPerFork) * UInt64(pageSize)
    emit(.info, "RSS after writes: \(mb(rssAfterWrite)) (delta \(mb(extraBytes)))")
    emit(.info, "expected from COW: \(numForks) forks × \(writePagesPerFork) pages × \(pageSize) bytes = \(mb(expectedExtraBytes))")

    // Spot-check: fork 0 sees its own writes; fork 1 sees only its own.
    let fork0First = ptrs[0].load(fromByteOffset: 0, as: UInt8.self)  // wrote tid=0 → 0x00
    let fork1First = ptrs[1].load(fromByteOffset: 0, as: UInt8.self)  // also wrote tid=0 → 0x00 to its own COW page
    let fork0Mid = ptrs[0].load(fromByteOffset: Int(stride * UInt32(pageSize)), as: UInt8.self)  // wrote tid=1 → 0x01
    emit(fork0First == 0x00 ? .pass : .fail, "fork[0][0] == 0x00 (got 0x\(String(fork0First, radix: 16)))")
    emit(fork1First == 0x00 ? .pass : .fail, "fork[1][0] == 0x00 (got 0x\(String(fork1First, radix: 16)))")
    emit(fork0Mid == 0x01 ? .pass : .fail, "fork[0][page \(stride)] == 0x01 (got 0x\(String(fork0Mid, radix: 16)))")

    // The big check: did RSS grow by roughly the expected COW amount, or did the OS
    // explode and decide to materialize all pages?
    let ratio = Double(extraBytes) / Double(expectedExtraBytes)
    if ratio < 3.0 {
        emit(.pass, "RSS growth ratio \(String(format: "%.2fx", ratio)) of expected COW. **Many-fork architecture is bounded; the OS isn't materializing untouched pages.**")
    } else {
        emit(.fail, "RSS growth ratio \(String(format: "%.2fx", ratio)) of expected — OS over-materialized, many-fork story weakens")
    }

    // Cleanup
    for i in 0..<numForks {
        munmap(ptrs[i], kvSize)
        close(fds[i])
    }
}

// MARK: - Verdict

print("")
print("=================================================================")
print(" Verdict: scroll up. Tests A, C, E, F together establish the")
print(" paged-mmap KV architecture as physically real on Apple Silicon.")
print(" Failures in any of these block the elegant path.")
print("=================================================================")
