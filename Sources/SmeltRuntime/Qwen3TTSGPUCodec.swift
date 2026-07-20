// Qwen3TTSGPU — codec-decoder graph (P6-U4b). Graduated verbatim from the proven
// Qwen3TTSCodecGPUTests dispatch helpers: the only changes are the weight source
// (f32(name) reads the package's bytesNoCopy buffer instead of a fresh SafetensorsLoader)
// and the pipeline lookup (pso(fn) uses the loaded package metallib instead of compiling
// a .metal). decodeCodec(codes -> 24 kHz wav) mirrors the full-codec-graph gate, which
// matches the real model. Dispatches are synchronous (waitUntilCompleted), honouring the
// driver's single-mmap lifetime invariant.

import Foundation
import Metal

extension Qwen3TTSGPU {

    // MARK: - buffer + dispatch helpers

    /// Time `body` into the `<stage>.<cat>` bucket when profiling is on; otherwise zero overhead.
    @inline(__always) func profTime<T>(_ cat: String, _ body: () -> T) -> T {
        guard SmeltDecodeProfile.enabled else { return body() }
        let t = CFAbsoluteTimeGetCurrent()
        let r = body()
        SmeltDecodeProfile.add(cat, UInt64(max(0, CFAbsoluteTimeGetCurrent() - t) * 1e9))
        return r
    }
    /// Throwing variant of `profTime` (wall-clock attribution around whole scopes/calls).
    @inline(__always) func profTimeThrows<T>(_ cat: String, _ body: () throws -> T) rethrows -> T {
        guard SmeltDecodeProfile.enabled else { return try body() }
        let t = CFAbsoluteTimeGetCurrent()
        let r = try body()
        SmeltDecodeProfile.add(cat, UInt64(max(0, CFAbsoluteTimeGetCurrent() - t) * 1e9))
        return r
    }

    func bufF32(_ v: [Float]) -> MTLBuffer {
        profTime("upload") { device.makeBuffer(bytes: v, length: v.count * 4, options: .storageModeShared)! }
    }
    /// `zero: false` skips the memset — ONLY for buffers whose consuming kernel FULLY overwrites every
    /// element (matmul/gemv/gemm outputs, nil-bias placeholders that `has_bias=0` never reads). Keep the
    /// default for pad buffers and accumulate/partial-write kernels. Saves the redundant DRAM zero-write +
    /// CPU memset on a recycled buffer that's about to be clobbered.
    func outF32(_ count: Int, zero: Bool = true) -> MTLBuffer {
        profTime("outAlloc") {
            // Inside a batched scope, recycle a completed-scope buffer of the same element count
            // instead of allocating. On a resident pooled buffer the zeroing is a cache rewrite,
            // not a page-fault.
            if batchEnc != nil, let b = bufferPool[count]?.popLast() {
                checkedOutBufs.append(b)
                if zero { memset(b.contents(), 0, count * 4) }
                return b
            }
            let b = device.makeBuffer(length: count * 4, options: .storageModeShared)!
            if zero { memset(b.contents(), 0, count * 4) }
            if batchEnc != nil { checkedOutBufs.append(b) }
            return b
        }
    }
    func readF32(_ buf: MTLBuffer, _ count: Int) -> [Float] {
        profTime("readback") {
            let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
            // Bulk copy (memcpy-class) — an element-wise `.map` over millions of floats was the
            // dominant cost measured in perf phase 0.
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        }
    }
    func pso(_ fn: String) throws -> MTLComputePipelineState {
        guard let p = pipeline(fn) else { throw LoadError.missingFunction(fn) }
        return p
    }
    /// fp32 weight values from the package, by checkpoint name.
    func f32(_ name: String) -> [Float] {
        guard let buf = weight(name), let shape = weightShape(name) else {
            preconditionFailure("Qwen3TTSGPU: missing weight \(name)")
        }
        let count = shape.reduce(1, *)
        let dt = weightDType(name)
        let elemBytes = dt == .f32 ? 4 : 2
        // The slice byte length (page-rounded) must cover the declared element count — guards
        // a manifest whose shape overstates the packed bytes from reading past the mapping.
        precondition(count >= 0 && count * elemBytes <= buf.length, "Qwen3TTSGPU: \(name) shape overruns its buffer")
        return profTime("weightMat") {
            switch dt {
            case .f32:
                let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
                return Array(UnsafeBufferPointer(start: ptr, count: count))
            case .bf16:  // exact widen bf16→fp32 (top 16 bits); codec weights stored bf16 are read here
                let ptr = buf.contents().bindMemory(to: UInt16.self, capacity: count)
                return (0..<count).map { Float(bitPattern: UInt32(ptr[$0]) << 16) }
            case .f16:
                let ptr = buf.contents().bindMemory(to: Float16.self, capacity: count)
                return (0..<count).map { Float(ptr[$0]) }
            case .u4:
                preconditionFailure("Qwen3TTSGPU.f32: \(name) is u4 — use weightRow / a u4 matmul path")
            }
        }
    }

    /// One dim-wide row of a 2-D weight, read straight from its mapped buffer — lets the front-end
    /// gather the handful of referenced text_embedding rows without copying the 1.2 GB table.
    func weightRow(_ name: String, _ row: Int, _ dim: Int) -> [Float] {
        guard let buf = weight(name), let shape = weightShape(name) else {
            preconditionFailure("Qwen3TTSGPU: missing weight \(name)")
        }
        precondition(shape.count == 2 && shape[1] == dim && dim > 0, "\(name) shape \(shape) not [rows, \(dim)]")
        let dt = weightDType(name)
        // u4: the row lives in a packed block (nibbles + per-group fp16 scale/bias at the entry's
        // relative offsets) — dequantize it on the host (mirrors SmeltAffineU4.dequantizeRow). Handled
        // first because its bytes-per-element (~0.5) breaks the dense dim*elemBytes bound below.
        if dt == .u4 {
            guard let u = weightU4(name) else { preconditionFailure("weightRow u4 \(name) missing metadata") }
            precondition(row >= 0 && row < shape[0], "\(name) row \(row) out of [0, \(shape[0]))")
            // Validate the (manifest-supplied) u4 block layout before indexing into the mmap — a
            // malformed package must trap cleanly here, not divide by zero or read OOB (parity with
            // matmulU4Dispatch's contract checks).
            let g = u.groupSize, groups = (dim + g - 1) / g, rowStride = (dim + 1) / 2
            precondition(g > 0, "\(name) u4 groupSize \(g) must be > 0")
            let rows = shape[0]
            precondition(u.scaleOffset >= 0 && u.biasOffset >= 0
                && rows * rowStride <= u.scaleOffset
                && u.scaleOffset + rows * groups * 2 <= u.biasOffset
                && u.biasOffset + rows * groups * 2 <= buf.length,
                "\(name) u4 block (scaleOffset \(u.scaleOffset), biasOffset \(u.biasOffset)) overruns buffer \(buf.length)")
            return profTime("weightMat") {
                let bytes = buf.contents()
                let nib = bytes.advanced(by: row * rowStride).assumingMemoryBound(to: UInt8.self)
                let sc = bytes.advanced(by: u.scaleOffset + row * groups * 2).assumingMemoryBound(to: UInt16.self)
                let bi = bytes.advanced(by: u.biasOffset + row * groups * 2).assumingMemoryBound(to: UInt16.self)
                return (0..<dim).map { c in
                    let grp = c / g
                    let scale = Float(Float16(bitPattern: sc[grp])), bias = Float(Float16(bitPattern: bi[grp]))
                    let byte = nib[c >> 1]
                    let n = (c & 1 == 0) ? (byte & 0x0F) : (byte >> 4)
                    return Float(n) * scale + bias
                }
            }
        }
        let elemBytes = dt == .f32 ? 4 : 2
        // Bound rows by the buffer BEFORE multiplying, so a corrupt manifest shape can't overflow Int.
        precondition(shape[0] >= 0 && shape[0] <= buf.length / (dim * elemBytes), "\(name) shape \(shape) overruns buffer")
        precondition(row >= 0 && row < shape[0], "\(name) row \(row) out of [0, \(shape[0]))")
        let base = row * dim
        return profTime("weightMat") {
            switch dt {
            case .f32:
                let ptr = buf.contents().bindMemory(to: Float.self, capacity: base + dim)
                return Array(UnsafeBufferPointer(start: ptr + base, count: dim))
            case .bf16:  // exact widen bf16→fp32 (the bf16 bits are the top 16 of the fp32)
                let ptr = buf.contents().bindMemory(to: UInt16.self, capacity: base + dim)
                return (0..<dim).map { Float(bitPattern: UInt32(ptr[base + $0]) << 16) }
            case .f16:
                let ptr = buf.contents().bindMemory(to: Float16.self, capacity: base + dim)
                return (0..<dim).map { Float(ptr[base + $0]) }
            case .u4:
                preconditionFailure("unreachable: u4 handled above")
            }
        }
    }

    func encode(_ pso: MTLComputePipelineState, _ grid: MTLSize, _ tg: MTLSize,
                _ body: (MTLComputeCommandEncoder) -> Void) {
        // Inside a `batched { }` scope: append to the shared encoder, no commit/sync. A default
        // compute encoder uses serial dispatch, so a later kernel reading an earlier kernel's output
        // buffer sees the completed write — the chain is ordered without explicit barriers.
        if let enc = batchEnc {
            enc.setComputePipelineState(pso); body(enc)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
            return
        }
        guard SmeltDecodeProfile.enabled else {
            let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pso); body(enc)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
            enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
            return
        }
        // encode / submit / GPU-compute / host-wait-minus-GPU split (sums to the dispatch wall).
        let t0 = CFAbsoluteTimeGetCurrent()
        let cmd = queue.makeCommandBuffer()!, enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso); body(enc)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        let t1 = CFAbsoluteTimeGetCurrent()
        cmd.commit()
        let t2 = CFAbsoluteTimeGetCurrent()
        cmd.waitUntilCompleted()
        let t3 = CFAbsoluteTimeGetCurrent()
        let hostWait = max(0, t3 - t2)
        // GPU timestamps are a different clock domain than CFAbsoluteTime; clamp the GPU-busy slice
        // to the host wait window so gpuCompute + waitNonGpu == hostWait exactly (no over-attribution
        // → no negative uncategorized).
        let pureGpu = min(max(0, cmd.gpuEndTime - cmd.gpuStartTime), hostWait)
        SmeltDecodeProfile.add("encode", UInt64(max(0, t1 - t0) * 1e9))
        SmeltDecodeProfile.add("submit", UInt64(max(0, t2 - t1) * 1e9))
        SmeltDecodeProfile.add("gpuCompute", UInt64(pureGpu * 1e9))
        SmeltDecodeProfile.add("waitNonGpu", UInt64((hostWait - pureGpu) * 1e9))
        // Per-kernel attribution (unbatched profiled path only): PSO labels are the function names.
        SmeltDecodeProfile.add("k_\(pso.label ?? "unlabeled")", UInt64(pureGpu * 1e9))
    }

    /// Run `body` with every `encode()` dispatch (and `blitCopy`) appended to ONE command buffer,
    /// committed + waited once at the end — instead of a commit+sync per kernel. Bit-exact: a default
    /// compute encoder runs its dispatches in serial order. `batched` owns the perf buckets for the run.
    /// Whether `batched {}` will actually batch (one command buffer for the whole
    /// scope) vs commit+wait per dispatch. UNBATCHED only under per-kernel profiling
    /// (SMELT_TTS_UNBATCHED). Single source of truth: the compiled-trunk session asks
    /// this instead of re-deriving the predicate (which would silently drift here).
    public var willRunBatched: Bool { !SmeltDecodeProfile.unbatched }

    func batched<T>(_ body: () throws -> T) rethrows -> T {
        precondition(batchEnc == nil, "batched { } must not nest — a nested scope corrupts the command-buffer lifecycle")
        // Reclaim the previous scope's outputs now (its command buffer completed and its readback ran
        // before control returned here), keyed by element count, so this scope's outF32 can recycle them.
        for b in checkedOutBufs { bufferPool[b.length / 4, default: []].append(b) }
        checkedOutBufs.removeAll(keepingCapacity: true)
        // Per-kernel profiling (SMELT_TTS_UNBATCHED) runs UNBATCHED: batchCmd stays
        // nil → encode/blit commit+wait per dispatch.
        if !willRunBatched { return try body() }
        let cmd = queue.makeCommandBuffer()!
        batchCmd = cmd
        batchEnc = cmd.makeComputeCommandEncoder()!
        let prof = SmeltDecodeProfile.enabled
        let t0 = prof ? CFAbsoluteTimeGetCurrent() : 0
        do {
            let r = try body()
            batchEnc?.endEncoding(); batchEnc = nil
            let t2 = prof ? CFAbsoluteTimeGetCurrent() : 0
            cmd.commit(); cmd.waitUntilCompleted()
            batchCmd = nil
            if prof {
                let hostWait = max(0, CFAbsoluteTimeGetCurrent() - t2)
                let pureGpu = min(max(0, cmd.gpuEndTime - cmd.gpuStartTime), hostWait)
                SmeltDecodeProfile.add("batchEncode", UInt64(max(0, t2 - t0) * 1e9))
                SmeltDecodeProfile.add("gpuCompute", UInt64(pureGpu * 1e9))
                SmeltDecodeProfile.add("waitNonGpu", UInt64((hostWait - pureGpu) * 1e9))
            }
            return r
        } catch {
            batchEnc?.endEncoding(); batchEnc = nil; batchCmd = nil
            throw error
        }
    }

    /// 1D RoPE cos/sin tables [frames, headDim] (duplicated halves), theta default 10000.
    func ropeTables(frames: Int, headDim: Int, theta: Float = 10000) -> ([Float], [Float]) {
        let half = headDim / 2
        var cosT = [Float](repeating: 0, count: frames * headDim)
        var sinT = [Float](repeating: 0, count: frames * headDim)
        for t in 0..<frames {
            for i in 0..<half {
                let invFreq = 1.0 / powf(theta, Float(2 * i) / Float(headDim))
                let ang = Float(t) * invFreq
                let c = cosf(ang), s = sinf(ang)
                cosT[t * headDim + i] = c; cosT[t * headDim + i + half] = c
                sinT[t * headDim + i] = s; sinT[t * headDim + i + half] = s
            }
        }
        return (cosT, sinT)
    }

    // MARK: - shared transformer dispatches (also used by talker/MTP in U4c)

    func matmulDispatch(_ xBuf: MTLBuffer, _ M: Int, _ K: Int, _ weight: [Float],
                        _ bias: [Float]?, _ N: Int) throws -> MTLBuffer {
        let wBuf = bufF32(weight), bBuf = bufF32(bias ?? [Float](repeating: 0, count: N)), outBuf = outF32(M * N, zero: false)
        // M>1, first choice: direct multi-N GEMM (see the resident overload below for the shape
        // rationale; bit-identical per (n,m) to gemv/gemm). Pso guard → gemm/matmul fallbacks.
        if M > 1, K % 4 == 0, let tnpipe = pipeline("gemm_tn_f32") {
            precondition(tnpipe.threadExecutionWidth == 32, "gemm_tn assumes 32-wide SIMD, got \(tnpipe.threadExecutionWidth)")
            let nGroups = (N + gemmTNFeatures - 1) / gemmTNFeatures
            let mRows = (M + gemmTNTileM - 1) / gemmTNTileM * gemmTNTileM
            encode(tnpipe, MTLSize(width: nGroups * 32, height: mRows, depth: 1),
                   MTLSize(width: 32, height: gemmTNTileM, depth: 1)) { enc in
                enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
                enc.setBuffer(bBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
                var mU = UInt32(M), nU = UInt32(N), kU = UInt32(K), hb = UInt32(bias == nil ? 0 : 1)
                enc.setBytes(&mU, length: 4, index: 4); enc.setBytes(&nU, length: 4, index: 5)
                enc.setBytes(&kU, length: 4, index: 6); enc.setBytes(&hb, length: 4, index: 7)
            }
            return outBuf
        }
        // M>1 (codec pre_transformer): coalesced batched GEMV, same as the resident overload — one SIMD
        // per (n,m), K-strided, bit-identical to gemv per row. The codec is all M=frames matmuls; this
        // gives them the prefill's coalescing win. K%4==0 / pso-present guard → naive matmul fallback.
        if M > 1, K % 4 == 0, let gpipe = pipeline("gemm_f32") {
            precondition(gpipe.threadExecutionWidth == 32, "gemm assumes 32-wide SIMD, got \(gpipe.threadExecutionWidth)")
            encode(gpipe, MTLSize(width: N * 32, height: M, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
                enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
                enc.setBuffer(bBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
                var mU = UInt32(M), nU = UInt32(N), kU = UInt32(K), hb = UInt32(bias == nil ? 0 : 1)
                enc.setBytes(&mU, length: 4, index: 4); enc.setBytes(&nU, length: 4, index: 5)
                enc.setBytes(&kU, length: 4, index: 6); enc.setBytes(&hb, length: 4, index: 7)
            }
            return outBuf
        }
        let pipe = try pso("matmul_f32")
        encode(pipe, MTLSize(width: N, height: M, depth: 1), MTLSize(width: min(N, 32), height: 1, depth: 1)) { enc in
            enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
            enc.setBuffer(bBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
            var mU = UInt32(M), nU = UInt32(N), kU = UInt32(K), hb = UInt32(bias == nil ? 0 : 1)
            enc.setBytes(&mU, length: 4, index: 4); enc.setBytes(&nU, length: 4, index: 5)
            enc.setBytes(&kU, length: 4, index: 6); enc.setBytes(&hb, length: 4, index: 7)
        }
        return outBuf
    }

    func rmsNormCodecDispatch(_ xBuf: MTLBuffer, _ frames: Int, _ dim: Int, _ w: [Float],
                              eps: Float = 1e-5) throws -> MTLBuffer {
        let pipe = try pso("rms_norm_codec_f32")
        // One 32-lane SIMD per frame (simd_sum reduction) — assumes a 32-wide SIMD (true on Apple GPUs).
        precondition(pipe.threadExecutionWidth == 32, "rms_norm_codec assumes 32-wide SIMD, got \(pipe.threadExecutionWidth)")
        let wBuf = bufF32(w), outBuf = outF32(frames * dim)
        encode(pipe, MTLSize(width: frames * 32, height: 1, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
            enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
            enc.setBuffer(outBuf, offset: 0, index: 2)
            var fr = UInt32(frames), dm = UInt32(dim), e = eps
            enc.setBytes(&fr, length: 4, index: 3); enc.setBytes(&dm, length: 4, index: 4); enc.setBytes(&e, length: 4, index: 5)
        }
        return outBuf
    }

    /// `into`: write the rotated output DIRECTLY at (buffer, byteOffset) — e.g. a KV-cache row —
    /// instead of a transient buffer + blit (each blit inside a batched scope splits the compute
    /// encoder, a measured per-layer stall in the decode/MTP paths). Same kernel, same bytes.
    func ropeDispatch(_ xBuf: MTLBuffer, _ frames: Int, _ heads: Int, _ headDim: Int,
                      _ cBuf: MTLBuffer, _ sBuf: MTLBuffer, into: (MTLBuffer, Int)? = nil) throws -> MTLBuffer {
        precondition(headDim % 2 == 0, "rope_apply_f32 rotates pairs; headDim must be even")
        let pipe = try pso("rope_apply_f32")
        if let (buf, off) = into {
            precondition(off >= 0 && off % 4 == 0 && off + frames * heads * headDim * 4 <= buf.length,
                         "rope into-write [\(off), +\(frames * heads * headDim * 4)) overruns buffer \(buf.length)")
        }
        let outBuf = into?.0 ?? outF32(frames * heads * headDim)
        // One thread per (frame, head, pair) — no reduction, so a flat 1D grid maximizes occupancy.
        let total = frames * heads * (headDim / 2)
        encode(pipe, MTLSize(width: total, height: 1, depth: 1), MTLSize(width: min(total, 256), height: 1, depth: 1)) { enc in
            enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(cBuf, offset: 0, index: 1)
            enc.setBuffer(sBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: into?.1 ?? 0, index: 3)
            var fr = UInt32(frames), hh = UInt32(heads), hdd = UInt32(headDim)
            enc.setBytes(&fr, length: 4, index: 4); enc.setBytes(&hh, length: 4, index: 5); enc.setBytes(&hdd, length: 4, index: 6)
        }
        return outBuf
    }

    func slidingAttnDispatch(_ qBuf: MTLBuffer, _ kBuf: MTLBuffer, _ vBuf: MTLBuffer,
                             _ frames: Int, _ heads: Int, _ headDim: Int, _ window: Int) throws -> MTLBuffer {
        precondition(window > 0 && window <= 72, "sliding_attn score stage is sized for window<=72")
        let outBuf = outF32(frames * heads * headDim)
        // First choice: one SIMD per (t, head) — bit-identical to the scalar kernel (see
        // sliding_attn_simd_f32.metal; same scalar-thread pathology as causal_gqa_attn_f32).
        // Pso guard → the original one-thread-per-(t,head) kernel for older packages.
        if let spipe = pipeline("sliding_attn_simd_f32") {
            precondition(spipe.threadExecutionWidth == 32, "sliding_attn_simd assumes 32-wide SIMD, got \(spipe.threadExecutionWidth)")
            encode(spipe, MTLSize(width: frames * 32, height: heads, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
                enc.setBuffer(qBuf, offset: 0, index: 0); enc.setBuffer(kBuf, offset: 0, index: 1)
                enc.setBuffer(vBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
                var fr = UInt32(frames), hh = UInt32(heads), hdd = UInt32(headDim), win = UInt32(window)
                enc.setBytes(&fr, length: 4, index: 4); enc.setBytes(&hh, length: 4, index: 5)
                enc.setBytes(&hdd, length: 4, index: 6); enc.setBytes(&win, length: 4, index: 7)
            }
            return outBuf
        }
        let pipe = try pso("sliding_attn_f32")
        encode(pipe, MTLSize(width: frames, height: heads, depth: 1), MTLSize(width: min(frames, 32), height: 1, depth: 1)) { enc in
            enc.setBuffer(qBuf, offset: 0, index: 0); enc.setBuffer(kBuf, offset: 0, index: 1)
            enc.setBuffer(vBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
            var fr = UInt32(frames), hh = UInt32(heads), hdd = UInt32(headDim), win = UInt32(window)
            enc.setBytes(&fr, length: 4, index: 4); enc.setBytes(&hh, length: 4, index: 5)
            enc.setBytes(&hdd, length: 4, index: 6); enc.setBytes(&win, length: 4, index: 7)
        }
        return outBuf
    }

    /// SiLU (swish) over `count` elements → new buffer (y = x/(1+exp(-x))), matching the host
    /// Qwen3TTSTalker.textProjection activation. The text_projection's fc1→fc2 gate (1a-i front-end).
    func siluDispatch(_ input: MTLBuffer, _ count: Int) throws -> MTLBuffer {
        let pipe = try pso("silu_f32")
        let outBuf = outF32(count, zero: false)
        encode(pipe, MTLSize(width: count, height: 1, depth: 1), MTLSize(width: min(count, 256), height: 1, depth: 1)) { enc in
            enc.setBuffer(input, offset: 0, index: 0); enc.setBuffer(outBuf, offset: 0, index: 1)
            var c = UInt32(count); enc.setBytes(&c, length: 4, index: 2)
        }
        return outBuf
    }

    func swigluDispatch(_ gateBuf: MTLBuffer, _ upBuf: MTLBuffer, _ count: Int) throws -> MTLBuffer {
        let pipe = try pso("swiglu_f32")
        let outBuf = outF32(count)
        encode(pipe, MTLSize(width: count, height: 1, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
            enc.setBuffer(gateBuf, offset: 0, index: 0); enc.setBuffer(upBuf, offset: 0, index: 1)
            enc.setBuffer(outBuf, offset: 0, index: 2)
            var c = UInt32(count); enc.setBytes(&c, length: 4, index: 3)
        }
        return outBuf
    }

    func scaleResidualTCDispatch(_ xBuf: MTLBuffer, _ resBuf: MTLBuffer, _ scale: [Float],
                                 _ channels: Int, _ frames: Int) throws -> MTLBuffer {
        let pipe = try pso("scale_residual_tc_f32")
        let sBuf = bufF32(scale), outBuf = outF32(frames * channels)
        encode(pipe, MTLSize(width: frames, height: channels, depth: 1), MTLSize(width: min(frames, 32), height: 1, depth: 1)) { enc in
            enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(resBuf, offset: 0, index: 1)
            enc.setBuffer(sBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
            var ch = UInt32(channels), fr = UInt32(frames), hs = UInt32(1)
            enc.setBytes(&ch, length: 4, index: 4); enc.setBytes(&fr, length: 4, index: 5)
            enc.setBytes(&hs, length: 4, index: 6)
        }
        return outBuf
    }

    // MARK: - Resident-weight dispatch overloads
    // Bind the package's already-mapped weight MTLBuffers directly (read-only inputs) instead of
    // bufF32-ing a fresh copy every dispatch. The length asserts guard against an UNDERSIZED bind
    // (page-rounded buffers, so they catch OOB reads, not an exact-shape mismatch within padding).

    /// Force-unwrap a resident weight buffer by name (the package guarantees presence at load).
    func wbuf(_ name: String) -> MTLBuffer {
        guard let b = weight(name) else { preconditionFailure("Qwen3TTSGPU: missing weight \(name)") }
        return b
    }

    /// gemm_tn_* output features per SIMD / threadgroup rows — MUST match GEMM_TN / GEMM_TN_TGM
    /// in gemm_tn_*.metal. (TN=4/TG_M=16 won the measured sweep: TN=8 and TG_M=32 were equal or
    /// worse; the threadgroup-staged and per-lane-acc[TM] shapes were 1.7-2× slower.)
    var gemmTNFeatures: Int { 4 }
    var gemmTNTileM: Int { 16 }

    /// `into`: write the output DIRECTLY at (buffer, byteOffset) — e.g. the V rows of a KV-cache —
    /// instead of a transient buffer + blit (a blit inside a batched scope splits the compute encoder;
    /// the per-layer K/V blits were a measured stall in the decode/MTP paths). Same kernels, same bytes.
    func matmulDispatch(_ xBuf: MTLBuffer, _ M: Int, _ K: Int, _ weightBuf: MTLBuffer,
                        _ biasBuf: MTLBuffer?, _ N: Int, dtype: WeightPackDType = .f32,
                        u4ScaleOffset: Int = 0, u4BiasOffset: Int = 0, u4GroupSize: Int = 0,
                        into: (MTLBuffer, Int)? = nil) throws -> MTLBuffer {
        if let (buf, off) = into {
            precondition(off >= 0 && off % 4 == 0 && off + M * N * 4 <= buf.length,
                         "matmul into-write [\(off), +\(M * N * 4)) overruns buffer \(buf.length)")
        }
        // Group-wise affine int4: weightBuf is ONE block — nibbles at 0, fp16 scales at u4ScaleOffset,
        // fp16 biases at u4BiasOffset (both page-aligned, so binding the same buffer at three offsets is
        // valid). Routes M=1→gemv_u4, M>1→gemm_u4 (no naive fallback). Returns before the f-suffix paths.
        if dtype == .u4 {
            return try matmulU4Dispatch(xBuf, M, K, weightBuf, biasBuf, N,
                                        scaleOffset: u4ScaleOffset, biasOffset: u4BiasOffset, groupSize: u4GroupSize,
                                        into: into)
        }
        // The f16w/bf16w kernels share matmul's buffer/constant layout — only the weight element size
        // (2 vs 4) and the pso suffix differ. f16/bf16 route to the coalesced gemv/gemm (float4 chunking,
        // K%4==0); the naive matmul fallback has no 2-byte variant.
        let suffix: String
        switch dtype {
        case .f32: suffix = "f32"; case .f16: suffix = "f16w_f32"; case .bf16: suffix = "bf16w_f32"
        case .u4: preconditionFailure("u4 routes to matmulU4Dispatch above, never the suffix path")
        }
        let elemBytes = dtype == .f32 ? 4 : 2
        precondition(weightBuf.length >= N * K * elemBytes, "matmul weight \(weightBuf.length)B < \(N*K*elemBytes)B [N=\(N),K=\(K)]")
        precondition(dtype == .f32 || K % 4 == 0, "f16/bf16 weights require K%4==0 (gemv/gemm float4); K=\(K)")
        if let biasBuf { precondition(biasBuf.length >= N * 4, "matmul bias \(biasBuf.length)B < \(N*4)B") }
        // M=1 (the dominant decode case): route to the coalesced GEMV — one threadgroup per output
        // feature, threads stride K contiguously → coalesced weight reads (bandwidth-bound, unlike
        // the strided one-thread-per-output matmul). The M>1 prefill keeps the naive matmul. Guarded
        // on the GEMV pso being present so an older package (no gemv kernels) falls back to matmul.
        if M == 1, let pipe = pipeline("gemv_\(suffix)") {
            // The GEMV uses one SIMD per output feature with simd_sum — assumes a 32-wide SIMD
            // (true on Apple GPUs). Trap clearly on a hypothetical non-32-wide GPU rather than
            // silently miscomputing the reduction.
            precondition(pipe.threadExecutionWidth == 32, "gemv assumes 32-wide SIMD, got \(pipe.threadExecutionWidth)")
            let tg = 32   // one SIMD per output feature (simd_sum reduction, no barriers)
            let bBuf = biasBuf ?? outF32(N, zero: false)   // has_bias=0 → never read
            let outBuf = into?.0 ?? outF32(N, zero: false) // gemv writes every out[n]
            encode(pipe, MTLSize(width: N * tg, height: 1, depth: 1), MTLSize(width: tg, height: 1, depth: 1)) { enc in
                enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(weightBuf, offset: 0, index: 1)
                enc.setBuffer(bBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: into?.1 ?? 0, index: 3)
                var mU = UInt32(1), nU = UInt32(N), kU = UInt32(K), hb = UInt32(biasBuf == nil ? 0 : 1)
                enc.setBytes(&mU, length: 4, index: 4); enc.setBytes(&nU, length: 4, index: 5)
                enc.setBytes(&kU, length: 4, index: 6); enc.setBytes(&hb, length: 4, index: 7)
            }
            return outBuf
        }
        // M>1, first choice: direct multi-N GEMM — TG_M row-SIMDs per threadgroup, each computing
        // TN output features (x chunk read once per TN dots, no barriers/staging). One-output-per-
        // threadgroup shapes measured launch-granularity-bound (~1.15M tiny TGs per 30-token
        // prefill, ~54GB/s); this keeps the ceil(M/TG_M)× weight dedup at TN× fewer threadgroups.
        // Per-(n,m) reduction bit-identical to gemv/gemm. Pso guard → gemm/matmul fallbacks below.
        if M > 1, K % 4 == 0, let tnpipe = pipeline("gemm_tn_\(suffix)") {
            precondition(tnpipe.threadExecutionWidth == 32, "gemm_tn assumes 32-wide SIMD, got \(tnpipe.threadExecutionWidth)")
            let bBuf = biasBuf ?? outF32(N, zero: false)        // has_bias=0 → never read
            let outBuf = into?.0 ?? outF32(M * N, zero: false)  // every (n,m) is written
            let nGroups = (N + gemmTNFeatures - 1) / gemmTNFeatures
            let mRows = (M + gemmTNTileM - 1) / gemmTNTileM * gemmTNTileM
            encode(tnpipe, MTLSize(width: nGroups * 32, height: mRows, depth: 1),
                   MTLSize(width: 32, height: gemmTNTileM, depth: 1)) { enc in
                enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(weightBuf, offset: 0, index: 1)
                enc.setBuffer(bBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: into?.1 ?? 0, index: 3)
                var mU = UInt32(M), nU = UInt32(N), kU = UInt32(K), hb = UInt32(biasBuf == nil ? 0 : 1)
                enc.setBytes(&mU, length: 4, index: 4); enc.setBytes(&nU, length: 4, index: 5)
                enc.setBytes(&kU, length: 4, index: 6); enc.setBytes(&hb, length: 4, index: 7)
            }
            return outBuf
        }
        // M>1 (prefill): route to the coalesced batched GEMV — one SIMD per (n,m) output, K-strided
        // (coalesced), vs matmul_f32's one-thread-per-output strided reads. The prefill measured ~2×
        // the decode's coalesced per-token cost purely from that uncoalescing; this closes it while
        // keeping the naive kernel's high occupancy. Guarded on the pso (older package → matmul) and
        // K%4==0 (float4 chunking). Per-(n,m) reduction is bit-identical to gemv → codes==gen_codes.
        if M > 1, K % 4 == 0, let gpipe = pipeline("gemm_\(suffix)") {
            precondition(gpipe.threadExecutionWidth == 32, "gemm assumes 32-wide SIMD, got \(gpipe.threadExecutionWidth)")
            let bBuf = biasBuf ?? outF32(N, zero: false)        // has_bias=0 → never read
            let outBuf = into?.0 ?? outF32(M * N, zero: false)  // gemm writes every (n,m)
            encode(gpipe, MTLSize(width: N * 32, height: M, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
                enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(weightBuf, offset: 0, index: 1)
                enc.setBuffer(bBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: into?.1 ?? 0, index: 3)
                var mU = UInt32(M), nU = UInt32(N), kU = UInt32(K), hb = UInt32(biasBuf == nil ? 0 : 1)
                enc.setBytes(&mU, length: 4, index: 4); enc.setBytes(&nU, length: 4, index: 5)
                enc.setBytes(&kU, length: 4, index: 6); enc.setBytes(&hb, length: 4, index: 7)
            }
            return outBuf
        }
        // bf16 has no naive matmul variant; it always routes to gemv/gemm above (K%4==0, pso packaged).
        precondition(dtype != .bf16, "bf16 weights must route to gemv/gemm, not the naive matmul fallback")
        let pipe = try pso("matmul_\(suffix)")
        let bBuf = biasBuf ?? outF32(N)   // hb=0 below makes the kernel ignore it; outF32 zeroes anyway
        let outBuf = into?.0 ?? outF32(M * N)
        encode(pipe, MTLSize(width: N, height: M, depth: 1), MTLSize(width: min(N, 32), height: 1, depth: 1)) { enc in
            enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(weightBuf, offset: 0, index: 1)
            enc.setBuffer(bBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: into?.1 ?? 0, index: 3)
            var mU = UInt32(M), nU = UInt32(N), kU = UInt32(K), hb = UInt32(biasBuf == nil ? 0 : 1)
            enc.setBytes(&mU, length: 4, index: 4); enc.setBytes(&nU, length: 4, index: 5)
            enc.setBytes(&kU, length: 4, index: 6); enc.setBytes(&hb, length: 4, index: 7)
        }
        return outBuf
    }

    /// Group-wise affine int4 matmul: out = x · dequant(W) (+bias). `weightBuf` is one block with packed
    /// nibbles at offset 0, fp16 scales at `scaleOffset`, fp16 biases at `biasOffset` (both page-aligned),
    /// bound at three offsets. M=1 → gemv_u4_f32, M>1 → gemm_u4_f32 (no naive fallback).
    func matmulU4Dispatch(_ xBuf: MTLBuffer, _ M: Int, _ K: Int, _ weightBuf: MTLBuffer,
                          _ biasBuf: MTLBuffer?, _ N: Int,
                          scaleOffset: Int, biasOffset: Int, groupSize: Int,
                          into: (MTLBuffer, Int)? = nil) throws -> MTLBuffer {
        precondition(K % 4 == 0, "u4 gemv/gemm require K%4==0 (float4 chunks); K=\(K)")
        precondition(groupSize >= 4 && groupSize % 4 == 0, "u4 group_size must be a multiple of 4 ≥ 4; got \(groupSize)")
        let groups = (K + groupSize - 1) / groupSize
        // Layout invariant: scales [scaleOffset, +N·groups·2) sit before biases [biasOffset, +N·groups·2),
        // and the block holds the (higher) bias region — so both fp16 regions are in bounds.
        precondition(scaleOffset + N * groups * 2 <= biasOffset, "u4 scale region overlaps bias region")
        precondition(weightBuf.length >= biasOffset + N * groups * 2,
                     "u4 weight block \(weightBuf.length)B < biasOffset \(biasOffset) + \(N*groups*2)B")
        if let biasBuf { precondition(biasBuf.length >= N * 4, "u4 bias \(biasBuf.length)B < \(N*4)B") }
        let fn = M == 1 ? "gemv_u4_f32" : "gemm_u4_f32"
        guard let pipe = pipeline(fn) else { preconditionFailure("u4 weight needs \(fn) (older package?)") }
        precondition(pipe.threadExecutionWidth == 32, "\(fn) assumes 32-wide SIMD, got \(pipe.threadExecutionWidth)")
        let bBuf = biasBuf ?? outF32(N, zero: false)        // has_bias=0 → never read
        let outBuf = into?.0 ?? outF32(M * N, zero: false)  // every (n,m) written
        // grid (N·32, M): one SIMD (32 lanes) per output feature (gemv) / per (n,m) (gemm).
        encode(pipe, MTLSize(width: N * 32, height: M, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
            enc.setBuffer(xBuf, offset: 0, index: 0)
            enc.setBuffer(weightBuf, offset: 0, index: 1)
            enc.setBuffer(weightBuf, offset: scaleOffset, index: 2)
            enc.setBuffer(weightBuf, offset: biasOffset, index: 3)
            enc.setBuffer(bBuf, offset: 0, index: 4); enc.setBuffer(outBuf, offset: into?.1 ?? 0, index: 5)
            var mU = UInt32(M), nU = UInt32(N), kU = UInt32(K), hb = UInt32(biasBuf == nil ? 0 : 1), gs = UInt32(groupSize)
            enc.setBytes(&mU, length: 4, index: 6); enc.setBytes(&nU, length: 4, index: 7)
            enc.setBytes(&kU, length: 4, index: 8); enc.setBytes(&hb, length: 4, index: 9)
            enc.setBytes(&gs, length: 4, index: 10)
        }
        return outBuf
    }

    /// GPU argmax of `logits[n]` → `outIdx[slot]` (uint), matching the CPU `for v where lg[v]>mx`
    /// (lowest index wins ties). Lets the MTP sub-passes chain without a CPU readback.
    func argmaxDispatch(_ logits: MTLBuffer, _ outIdx: MTLBuffer, slot: Int, n: Int) throws {
        // Trap a too-small logits/out buffer on CPU rather than a silent GPU OOB read (parity with gatherRowDispatch).
        precondition(logits.length >= n * 4, "argmax logits \(logits.length)B < \(n*4)B [n=\(n)]")
        precondition(outIdx.length >= (slot + 1) * 4, "argmax outIdx \(outIdx.length)B < \((slot+1)*4)B [slot=\(slot)]")
        let pipe = try pso("argmax_f32")
        precondition(pipe.threadExecutionWidth == 32, "argmax_f32 assumes 32-wide SIMD, got \(pipe.threadExecutionWidth)")
        encode(pipe, MTLSize(width: 32, height: 1, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
            enc.setBuffer(logits, offset: 0, index: 0); enc.setBuffer(outIdx, offset: 0, index: 1)
            var nU = UInt32(n), s = UInt32(slot)
            enc.setBytes(&nU, length: 4, index: 2); enc.setBytes(&s, length: 4, index: 3)
        }
    }

    /// Temperature/top-k sampling counterpart to argmaxDispatch: draws idx into outIdx[slot] from
    /// softmax(top_k(logits)/temperature) using uniforms[slot] (the host-precomputed RNG draw for this
    /// sub-pass). Same single-threadgroup shape, so it slots into the MTP chain in place of the argmax.
    func sampleTopKDispatch(_ logits: MTLBuffer, uniforms: MTLBuffer, _ outIdx: MTLBuffer,
                            slot: Int, n: Int, temperature: Float, topK: Int) throws {
        precondition(logits.length >= n * 4, "sampleTopK logits \(logits.length)B < \(n*4)B [n=\(n)]")
        precondition(outIdx.length >= (slot + 1) * 4, "sampleTopK outIdx \(outIdx.length)B < \((slot+1)*4)B [slot=\(slot)]")
        precondition(uniforms.length >= (slot + 1) * 4, "sampleTopK uniforms \(uniforms.length)B < \((slot+1)*4)B [slot=\(slot)]")
        precondition(temperature > 0, "sampleTopK needs temperature > 0; greedy is argmaxDispatch")
        let pipe = try pso("sample_topk_f32")
        precondition(pipe.threadExecutionWidth == 32, "sample_topk_f32 assumes 32-wide SIMD, got \(pipe.threadExecutionWidth)")
        encode(pipe, MTLSize(width: 32, height: 1, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
            enc.setBuffer(logits, offset: 0, index: 0); enc.setBuffer(uniforms, offset: 0, index: 1)
            enc.setBuffer(outIdx, offset: 0, index: 2)
            var nU = UInt32(n), s = UInt32(slot), temp = temperature, kk = UInt32(max(topK, 1))
            enc.setBytes(&nU, length: 4, index: 3); enc.setBytes(&s, length: 4, index: 4)
            enc.setBytes(&temp, length: 4, index: 5); enc.setBytes(&kk, length: 4, index: 6)
        }
    }

    /// Gather row `idx[slot]` of `table[rows,dim]` → new `[dim]` buffer. The index is read from the
    /// GPU buffer (the prior sub-pass's argmax), so the gather chains with no CPU round-trip.
    func gatherRowDispatch(_ table: MTLBuffer, _ idx: MTLBuffer, slot: Int, dim: Int, rows: Int,
                           dtype: WeightPackDType = .f32) throws -> MTLBuffer {
        // The index is read from a GPU buffer, so a too-small table (a dtype/shape regression) would be a
        // silent GPU OOB read — trap on CPU instead. `rows` is the gather domain (the argmax vocab).
        // bf16 tables (BF16-source codec embeddings) widen in-kernel → bit-identical to f32; same bindings.
        precondition(dtype == .f32 || dtype == .bf16, "gather_row supports f32/bf16 tables, not \(dtype)")
        let elemBytes = dtype == .f32 ? 4 : 2
        precondition(table.length >= rows * dim * elemBytes, "gather table \(table.length)B < \(rows*dim*elemBytes)B [rows=\(rows),dim=\(dim)]")
        precondition(idx.length >= (slot + 1) * 4, "gather idx \(idx.length)B < \((slot+1)*4)B [slot=\(slot)]")
        let pipe = try pso(dtype == .bf16 ? "gather_row_bf16w_f32" : "gather_row_f32")
        let out = outF32(dim)
        encode(pipe, MTLSize(width: dim, height: 1, depth: 1), MTLSize(width: min(dim, 256), height: 1, depth: 1)) { enc in
            enc.setBuffer(table, offset: 0, index: 0); enc.setBuffer(idx, offset: 0, index: 1)
            enc.setBuffer(out, offset: 0, index: 2)
            var d = UInt32(dim), s = UInt32(slot)
            enc.setBytes(&d, length: 4, index: 3); enc.setBytes(&s, length: 4, index: 4)
        }
        return out
    }

    func rmsNormCodecDispatch(_ xBuf: MTLBuffer, _ frames: Int, _ dim: Int, _ wBuf: MTLBuffer,
                              eps: Float = 1e-5) throws -> MTLBuffer {
        precondition(wBuf.length >= dim * 4, "rmsNorm weight \(wBuf.length)B < \(dim*4)B")
        let pipe = try pso("rms_norm_codec_f32")
        precondition(pipe.threadExecutionWidth == 32, "rms_norm_codec assumes 32-wide SIMD, got \(pipe.threadExecutionWidth)")
        let outBuf = outF32(frames * dim)
        encode(pipe, MTLSize(width: frames * 32, height: 1, depth: 1), MTLSize(width: 32, height: 1, depth: 1)) { enc in
            enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(wBuf, offset: 0, index: 1)
            enc.setBuffer(outBuf, offset: 0, index: 2)
            var fr = UInt32(frames), dm = UInt32(dim), e = eps
            enc.setBytes(&fr, length: 4, index: 3); enc.setBytes(&dm, length: 4, index: 4); enc.setBytes(&e, length: 4, index: 5)
        }
        return outBuf
    }

    /// `scaled: false` = add-only (`out = res + x`); the kernel skips the scale read+multiply. Bit-exact
    /// for an all-ones scale (1.0*x == x), which is what the talker/MTP residual-adds pass.
    func scaleResidualTCDispatch(_ xBuf: MTLBuffer, _ resBuf: MTLBuffer, _ scaleBuf: MTLBuffer,
                                 _ channels: Int, _ frames: Int, scaled: Bool = true) throws -> MTLBuffer {
        precondition(!scaled || scaleBuf.length >= channels * 4, "scale buffer \(scaleBuf.length)B < \(channels*4)B")
        let pipe = try pso("scale_residual_tc_f32")
        let outBuf = outF32(frames * channels)
        encode(pipe, MTLSize(width: frames, height: channels, depth: 1), MTLSize(width: min(frames, 32), height: 1, depth: 1)) { enc in
            enc.setBuffer(xBuf, offset: 0, index: 0); enc.setBuffer(resBuf, offset: 0, index: 1)
            enc.setBuffer(scaleBuf, offset: 0, index: 2); enc.setBuffer(outBuf, offset: 0, index: 3)
            var ch = UInt32(channels), fr = UInt32(frames), hs = UInt32(scaled ? 1 : 0)
            enc.setBytes(&ch, length: 4, index: 4); enc.setBytes(&fr, length: 4, index: 5)
            enc.setBytes(&hs, length: 4, index: 6)
        }
        return outBuf
    }


    // MARK: - C2-FULL: compiled codec record-table path (docs/codec-c2-full-plan.md)

    /// Fixed Qwen3-TTS-12Hz codec topology with the three checkpoint-derived dims (`restN`, `firstN`,
    /// `upsampleInter`) read from the loaded weight shapes.
    func makeCodecConfig() -> Qwen3TTSCodecConfig {
        let preConvDim = 1024, restCount = 15
        // Fail loud on a missing/empty tensor rather than baking a wrong plan constant from a 1-product.
        func count(_ name: String) -> Int {
            guard let shape = weightShape(name) else { preconditionFailure("codec config: missing weight \(name)") }
            return shape.reduce(1, *)
        }
        let firstN = count("decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage")
        let restN = count("decoder.quantizer.rvq_rest.vq.layers.0._codebook.cluster_usage")
        precondition(firstN > 0 && restN > 0, "codec config: empty codebook vocab (firstN \(firstN), restN \(restN))")
        // ONE restN is baked into the rvq_gather_sum constant — every rest codebook must share it.
        for k in 1..<restCount {
            let n = count("decoder.quantizer.rvq_rest.vq.layers.\(k)._codebook.cluster_usage")
            precondition(n == restN, "codec config: rest codebook \(k) vocab \(n) != layer 0 (\(restN))")
        }
        func upInter(_ b: Int) -> Int {
            let total = count("decoder.upsample.\(b).1.pwconv1.weight")
            precondition(total % preConvDim == 0, "codec config: upsample \(b) pwconv1 (\(total)) not divisible by \(preConvDim)")
            return total / preConvDim
        }
        return Qwen3TTSCodecConfig(restN: restN, firstN: firstN, upsampleInter: [upInter(0), upInter(1)])
    }

    /// A codec-decoder weight as f32 bytes for the COMPILED STREAMING codec (whose record kernels are
    /// all f32). f32 weights bind RAW (resident, zero copy — the common bf16/f16/f32 package, whose codec
    /// weights are f32). A u4 build stores the rank≥2 `decoder.*` conv/transformer weights as bf16
    /// (Qwen3TTSPackageBuilder: "read back via f32(), widened") — widen those to f32 ONCE, cached on the
    /// gpu and reused across chunks + streams, so they produce the SAME bytes offline `realizeCodecPlan`
    /// uploads via `bufF32(f32(name))`. (`f32()` traps loud on an unexpected u4 codec weight.)
    func codecStreamWeight(_ name: String) -> MTLBuffer {
        if weightDType(name) == .f32 { return wbuf(name) }
        if let cached = codecStreamWidened[name] { return cached }
        let buf = bufF32(f32(name))
        codecStreamWidened[name] = buf
        return buf
    }

    /// Materialize each `CodecSlotDesc` → an `MTLBuffer`. Raw weights via `f32(name)` (the canonical
    /// dtype-aware source → bit-identical); derived tensors (codebooks, projConcat,
    /// RoPE) via the existing host helpers; `.scratch` fresh-zeroed (channel_copy pad contract).
    func realizeCodecPlan(_ plan: Qwen3TTSCodecPlan, codes: [Int32], frames: Int,
                          config c: Qwen3TTSCodecConfig) -> [MTLBuffer] {
        let rvqDim = c.rvqDim
        // Preserve the gather's OOB guard (rvqGatherSum relied on this check) before binding codes.
        for t in 0..<frames { precondition(codes[t] >= 0 && Int(codes[t]) < c.firstN, "semantic code OOB") }
        for k in 0..<c.restCount { for t in 0..<frames {
            precondition(codes[(k + 1) * frames + t] >= 0 && Int(codes[(k + 1) * frames + t]) < c.restN, "acoustic code OOB")
        } }
        func codebookFirst() -> [Float] {
            Qwen3TTSCodec.codebookEmbedding(
                embeddingSum: f32("decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"),
                clusterUsage: f32("decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage"), dim: rvqDim)
        }
        func codebookRestFlat() -> [Float] {
            var flat: [Float] = []
            for k in 0..<c.restCount {
                flat += Qwen3TTSCodec.codebookEmbedding(
                    embeddingSum: f32("decoder.quantizer.rvq_rest.vq.layers.\(k)._codebook.embedding_sum"),
                    clusterUsage: f32("decoder.quantizer.rvq_rest.vq.layers.\(k)._codebook.cluster_usage"), dim: rvqDim)
            }
            return flat
        }
        func projConcat() -> [Float] {
            let firstProj = f32("decoder.quantizer.rvq_first.output_proj.weight")
            let restProj = f32("decoder.quantizer.rvq_rest.output_proj.weight")
            let outDim = firstProj.count / rvqDim
            var pc = [Float](repeating: 0, count: outDim * 2 * rvqDim)
            for o in 0..<outDim {
                for d in 0..<rvqDim { pc[o * 2 * rvqDim + d] = firstProj[o * rvqDim + d] }
                for d in 0..<rvqDim { pc[o * 2 * rvqDim + rvqDim + d] = restProj[o * rvqDim + d] }
            }
            return pc
        }
        let (cosT, sinT) = ropeTables(frames: frames, headDim: c.ptHeadDim)
        return plan.slots.map { desc in
            switch desc {
            case .weight(let name): return bufF32(f32(name))
            case .codes: return device.makeBuffer(bytes: codes, length: codes.count * 4, options: .storageModeShared)!
            case .codebookFirst: return bufF32(codebookFirst())
            case .codebookRestFlat: return bufF32(codebookRestFlat())
            case .projConcat: return bufF32(projConcat())
            case .ropeCos: return bufF32(cosT)
            case .ropeSin: return bufF32(sinT)
            case .ones(let n): return bufF32([Float](repeating: 1, count: n))
            case .scratch(let n):
                let buf = device.makeBuffer(length: n * 4, options: .storageModeShared)!
                memset(buf.contents(), 0, n * 4)
                return buf
            }
        }
    }

    /// Build the codec pipeline table (dense local index -> PSO). Fail-loud on a missing kernel: the
    /// sole-compiled path has no fallback, and every codec kernel is in the package metallib.
    private func codecPipelines() throws -> [MTLComputePipelineState] {
        try Qwen3TTSCodecEmitter.pipelineNames.map { try pso($0) }
    }

    // MARK: - full codec graph: RVQ codes -> 24 kHz waveform

    /// Decode `codes` (row-major [16, frames] of codebook ids, as the real model emits) into a 24 kHz
    /// mono waveform. The compiled record table (docs/codec-default-flip-plan.md) is the SOLE path;
    /// it runs on its OWN command buffer (one serial encoder = bit-exact ordering), independent of the
    /// shared `batched` scope, so it works in every mode (the codec is never run during capture).
    public func decodeCodec(codes: [Int32], frames: Int) throws -> [Float] {
        SmeltDecodeProfile.setStage("codec")
        precondition(batchEnc == nil, "decodeCodec must not run inside an open batched scope")
        let cfg = makeCodecConfig()
        let plan = Qwen3TTSCodecEmitter.plan(frames: frames, config: cfg)
        let buffers = realizeCodecPlan(plan, codes: codes, frames: frames, config: cfg)
        let pipelines = try codecPipelines()
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        SmeltCodecRecordRunner.encode(plan.records, pipelines: pipelines, buffers: buffers, into: enc)
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()
        return readF32(buffers[plan.outputSlot], plan.outputLength)
    }
}
